/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * See the License in the specific file packaged with this Software for details.
 */

// 文件职责：用真实 SQLite 故障路径验证 SQLiteEngine 的错误语义：
// 连接失败、非法状态、约束冲突、损坏文件、条件字段越界和备份目标不可写。
// 旧映射标识符/复合主键校验用例随映射层删除；路由校验随路由模型删除。
#include <gtest/gtest.h>

#include <sqlite3.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "storage_backends/sqlite/engine/sqlite_engine.h"
#include "storage/query_expression.h"
#include "storage/record_contract.h"
#include "storage/storage_catalog.h"

namespace HcclSim::Storage {
// ROLLBACK 失败故障注入测试点（经 friend 关系访问 SQLiteEngine 私有注入标志）：
// 真实 WAL 连接上 ROLLBACK 几乎不可构造 IO 故障（容器无 mount 能力、root 绕过
// 权限位），按合同以最小注入点验证收尾语义；仅本测试文件定义，产品代码不得引用。
struct SQLiteRollbackFaultTestPeer {
    static void FailNextRollback(SQLiteEngine& engine) { engine.failNextRollbackForTest_ = true; }
};
} // namespace HcclSim::Storage

namespace fault_semantics_test {
struct FaultTab {
    uint64_t id{0};
    std::string name;
};
struct DynTab {
    uint64_t id{0};
    std::string note;
};
} // namespace fault_semantics_test

using namespace HcclSim::Storage::Ddl;
#define Field HCCL_VM_DDL_FIELD
#define Version HCCL_VM_DDL_VERSION
#define PrimaryKey HCCL_VM_DDL_PRIMARY_KEY
#define Index HCCL_VM_DDL_INDEX
#define Scope Ddl::ScopeKind

#define SQLITE_FAULT_TEST_DATABASE(Table)                                                                \
    Table(                                                                                               \
        fault_semantics_test::FaultTab,                                                                  \
        (Version(1), Scope::Static, Field(id, AutoIncrement), Field(name), PrimaryKey(id), Index(name))) \
        Table(                                                                                           \
            fault_semantics_test::DynTab,                                                                \
            (Version(1), Scope::DynamicPid, Field(id, AutoIncrement), Field(note), PrimaryKey(id), Index(note)))

#define HCCL_VM_DDL_CURRENT_DATABASE "SQLITE_FAULT_TEST_DB"
HCCL_VM_DEFINE_DATABASE(kSqliteFaultTestDatabase, SQLITE_FAULT_TEST_DB, SQLITE_FAULT_TEST_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE
#undef Field
#undef Version
#undef PrimaryKey
#undef Index
#undef Scope

namespace {
using namespace HcclSim::Storage;
using Fault = fault_semantics_test::FaultTab;

constexpr SqliteConnectionProfile kFaultProfile{10000, false, 200};

// 生成一次性临时数据库路径，并清理上次运行可能遗留的同名文件。
std::string TempDatabasePath(const std::string& tag)
{
    static std::atomic<uint64_t> counter{0};
    const std::string path
        = testing::TempDir() + "sqlite_fault_" + tag + "_" + std::to_string(counter.fetch_add(1)) + ".db";
    (void)std::remove(path.c_str());
    return path;
}

EngineOpenRequest MakeOpenRequest(StorageOpenMode mode = StorageOpenMode::READ_WRITE_CREATE)
{
    return EngineOpenRequest{mode, "ns"};
}

TEST(SQLiteEngineFaultTest, OpenWithUnwritablePathFailsAsBackendUnavailable)
{
    SQLiteEngine engine("/nonexistent-sqlite-fault-dir/missing.db", &Ddl::kSqliteFaultTestDatabase, kFaultProfile);
    const auto opened = engine.Open(MakeOpenRequest());
    ASSERT_FALSE(opened.ok());
    EXPECT_EQ(opened.code, DbCode::BACKEND_UNAVAILABLE);
    EXPECT_FALSE(opened.diagnostic.empty());
}

TEST(SQLiteEngineFaultTest, OpenExistingModeOnMissingFileFails)
{
    SQLiteEngine engine(TempDatabasePath("missing"), &Ddl::kSqliteFaultTestDatabase, kFaultProfile);
    const auto opened = engine.Open(MakeOpenRequest(StorageOpenMode::READ_EXISTING));
    ASSERT_FALSE(opened.ok());
    EXPECT_EQ(opened.code, DbCode::BACKEND_UNAVAILABLE);
}

TEST(SQLiteEngineFaultTest, RepeatedOpenAndCloseAreInvalidState)
{
    SQLiteEngine engine(TempDatabasePath("states"), &Ddl::kSqliteFaultTestDatabase, kFaultProfile);
    ASSERT_TRUE(engine.Open(MakeOpenRequest()).ok());
    EXPECT_EQ(engine.Open(MakeOpenRequest()).code, DbCode::INVALID_STATE);
    ASSERT_TRUE(engine.Prepare().ok());
    ASSERT_TRUE(engine.Close().ok());
    EXPECT_EQ(engine.Close().code, DbCode::INVALID_STATE);
    EXPECT_EQ(engine.Open(MakeOpenRequest()).code, DbCode::INVALID_STATE);
}

TEST(SQLiteEngineFaultTest, OperationsBeforeOpenAndPrepareAreInvalidState)
{
    SQLiteEngine engine(TempDatabasePath("unprepared"), &Ddl::kSqliteFaultTestDatabase, kFaultProfile);
    const auto& meta = RecordSchema<Fault>::Meta();
    Fault record;
    record.name = "early";
    EXPECT_EQ(engine.Insert(meta, TableScope::Static(), &record, nullptr).code, DbCode::INVALID_STATE);
    std::vector<Fault> rows;
    auto sink = ResultSink::For(rows);
    EXPECT_EQ(engine.Select(meta, TableScope::Static(), All<Fault>().data, {}, sink).code, DbCode::INVALID_STATE);
    EXPECT_EQ(engine.Count(meta, TableScope::Static(), All<Fault>().data).code, DbCode::INVALID_STATE);
    TransactionBinding transaction;
    EXPECT_EQ(engine.Begin(transaction).code, DbCode::INVALID_STATE);
    EXPECT_EQ(engine.Insert(meta, TableScope::Static(), &record, &transaction).code, DbCode::INVALID_STATE);
    EXPECT_EQ(engine.Commit(transaction).code, DbCode::INVALID_STATE);
    EXPECT_EQ(engine.Rollback(transaction).code, DbCode::INVALID_STATE);
    EXPECT_EQ(engine.BackupRelational({"/tmp/fault-backup"}).code, DbCode::INVALID_STATE);

    // Open 后未 Prepare 仍然拒绝执行。
    ASSERT_TRUE(engine.Open(MakeOpenRequest()).ok());
    EXPECT_EQ(engine.Insert(meta, TableScope::Static(), &record, nullptr).code, DbCode::INVALID_STATE);
}

TEST(SQLiteEngineFaultTest, CorruptDatabaseFailsOpenAsIOErrorAndLeavesNoOpenConnection)
{
    const std::string path = TempDatabasePath("corrupt");
    {
        std::ofstream out(path, std::ios::binary);
        out << "this is definitely not a sqlite database file 0123456789";
    }
    SQLiteEngine engine(path, &Ddl::kSqliteFaultTestDatabase, kFaultProfile);
    const auto opened = engine.Open(MakeOpenRequest(StorageOpenMode::READ_EXISTING));
    ASSERT_FALSE(opened.ok());
    EXPECT_EQ(opened.code, DbCode::IO_ERROR);
    // 原生句柄已关闭，引擎未进入 OPENED；不能 Prepare/Close 或被当作已打开。
    EXPECT_EQ(engine.Prepare().code, DbCode::INVALID_STATE);
    EXPECT_EQ(engine.Close().code, DbCode::INVALID_STATE);
    EXPECT_EQ(engine.Open(MakeOpenRequest(StorageOpenMode::READ_EXISTING)).code, DbCode::IO_ERROR);

    SQLiteEngine healthy(TempDatabasePath("healthy-after-corrupt"), &Ddl::kSqliteFaultTestDatabase, kFaultProfile);
    ASSERT_TRUE(healthy.Open(MakeOpenRequest()).ok());
    ASSERT_TRUE(healthy.Prepare().ok());
    EXPECT_TRUE(healthy.Close().ok());
}

TEST(SQLiteEngineFaultTest, DuplicatePrimaryKeyMapsToConflict)
{
    SQLiteEngine engine(TempDatabasePath("conflict"), &Ddl::kSqliteFaultTestDatabase, kFaultProfile);
    ASSERT_TRUE(engine.Open(MakeOpenRequest()).ok());
    ASSERT_TRUE(engine.Prepare().ok());
    const auto& meta = RecordSchema<Fault>::Meta();

    Fault first;
    first.id = 5U;
    first.name = "first";
    ASSERT_TRUE(engine.Insert(meta, TableScope::Static(), &first, nullptr).ok());

    Fault duplicate;
    duplicate.id = 5U;
    duplicate.name = "duplicate";
    const auto conflicted = engine.Insert(meta, TableScope::Static(), &duplicate, nullptr);
    ASSERT_FALSE(conflicted.ok());
    EXPECT_EQ(conflicted.code, DbCode::CONFLICT);
    EXPECT_FALSE(conflicted.diagnostic.empty());
}

TEST(SQLiteEngineFaultTest, UnmappedQueryFieldIsInvalidArgument)
{
    SQLiteEngine engine(TempDatabasePath("unmapped"), &Ddl::kSqliteFaultTestDatabase, kFaultProfile);
    ASSERT_TRUE(engine.Open(MakeOpenRequest()).ok());
    ASSERT_TRUE(engine.Prepare().ok());
    const auto& meta = RecordSchema<Fault>::Meta();
    // 直接构造越界字段编号的条件：引擎按目录边界拒绝
    ConditionClause clause;
    clause.fieldId = 99U;
    clause.compare = Compare::EQUAL;
    clause.value = ConditionValue{DbScalar{int64_t{1}}};
    ConditionData where{std::vector<ConditionClause>{clause}};
    std::vector<Fault> rows;
    auto sink = ResultSink::For(rows);
    EXPECT_EQ(engine.Select(meta, TableScope::Static(), where, {}, sink).code, DbCode::INVALID_ARGUMENT);
}

TEST(SQLiteEngineFaultTest, DynamicTableWithIndexPreparesAndCreatesIndexLazily)
{
    using Dyn = fault_semantics_test::DynTab;
    const std::string path = TempDatabasePath("dynindex");
    SQLiteEngine engine(path, &Ddl::kSqliteFaultTestDatabase, kFaultProfile);
    ASSERT_TRUE(engine.Open(MakeOpenRequest()).ok());
    const auto prepared = engine.Prepare();
    ASSERT_TRUE(prepared.ok()) << prepared.diagnostic;
    // 动态表不在 Prepare 中创建；首次访问惰性建表并建索引
    Dyn record;
    record.note = "first";
    const auto inserted = engine.Insert(RecordSchema<Dyn>::Meta(), TableScope::Process(3U), &record, nullptr);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    sqlite3* db = nullptr;
    ASSERT_EQ(sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr), SQLITE_OK);
    sqlite3_stmt* stmt = nullptr;
    ASSERT_EQ(
        sqlite3_prepare_v2(
            db, "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name = 'idx_DynTab_note_DynTab_P_3'", -1,
            &stmt, nullptr),
        SQLITE_OK);
    ASSERT_EQ(sqlite3_step(stmt), SQLITE_ROW);
    EXPECT_EQ(sqlite3_column_int64(stmt, 0), 1);
    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

TEST(SQLiteEngineFaultTest, BackupToUnwritableDestinationFailsAsIOError)
{
    SQLiteEngine engine(TempDatabasePath("backup-source"), &Ddl::kSqliteFaultTestDatabase, kFaultProfile);
    ASSERT_TRUE(engine.Open(MakeOpenRequest()).ok());
    ASSERT_TRUE(engine.Prepare().ok());
    const auto& meta = RecordSchema<Fault>::Meta();
    Fault record;
    record.name = "backed";
    ASSERT_TRUE(engine.Insert(meta, TableScope::Static(), &record, nullptr).ok());

    EngineBackupRequest request;
    request.destinationPath = "/nonexistent-sqlite-fault-dir/backup.db";
    const auto backed = engine.BackupRelational(request);
    ASSERT_FALSE(backed.ok());
    EXPECT_EQ(backed.code, DbCode::IO_ERROR);
}

TEST(SQLiteEngineFaultTest, TransactionWithoutPhysicalExecutionRejectsCommit)
{
    SQLiteEngine engine(TempDatabasePath("empty-tx"), &Ddl::kSqliteFaultTestDatabase, kFaultProfile);
    ASSERT_TRUE(engine.Open(MakeOpenRequest()).ok());
    ASSERT_TRUE(engine.Prepare().ok());
    TransactionBinding transaction;
    EXPECT_EQ(engine.Commit(transaction).code, DbCode::INVALID_STATE);
    EXPECT_EQ(engine.Rollback(transaction).code, DbCode::INVALID_STATE);
}

namespace {

    /// 对物理库执行一条无参 SQL（预置同名对象用）。
    void ExecDirectSql(const std::string& path, const char* sql)
    {
        sqlite3* db = nullptr;
        ASSERT_EQ(sqlite3_open(path.c_str(), &db), SQLITE_OK);
        char* message = nullptr;
        ASSERT_EQ(sqlite3_exec(db, sql, nullptr, nullptr, &message), SQLITE_OK) << (message ? message : sql);
        if (message != nullptr) {
            sqlite3_free(message);
        }
        sqlite3_close(db);
    }

} // namespace

// owner 发布失败且 ROLLBACK 自身也失败：诊断必须同时保留原始错误与 rollback
// 错误；连接状态不可证明——立即关闭且不得被同一引擎继续复用（后续 Prepare 按
// "连接未打开"拒绝，不伪装成普通可重试失败）。触发路径：预置同名 VIEW 占位，
// CREATE TABLE IF NOT EXISTS 静默跳过，随后 CREATE INDEX 落在 VIEW 上真实失败，
// 进入 Prepare 失败路径的 ROLLBACK（注入失败）。真实 WAL 连接上无法构造该故障，
// 故经注入点验证（注入点本身仅存在于测试路径）。
TEST(SQLiteEngineFaultTest, RollbackFailureIsReportedAndConnectionIsNotReused)
{
    const std::string path = TempDatabasePath("rollback-failure");
    {
        // 预置同名 VIEW：表创建被 IF NOT EXISTS 吞掉，索引创建必然失败。
        ExecDirectSql(path, "CREATE VIEW FaultTab AS SELECT 1 AS id, 'x' AS name");
    }
    SQLiteEngine engine(path, &Ddl::kSqliteFaultTestDatabase, kFaultProfile);
    ASSERT_TRUE(engine.Open(MakeOpenRequest()).ok());
    HcclSim::Storage::SQLiteRollbackFaultTestPeer::FailNextRollback(engine);
    const auto prepared = engine.Prepare();
    ASSERT_FALSE(prepared.ok()) << "the view-shadowed table must fail owner preparation";
    // 诊断同时携带原始错误（索引失败）与 rollback 失败说明。
    EXPECT_NE(prepared.diagnostic.find("rollback also failed"), std::string::npos)
        << "the rollback failure must not be swallowed";
    EXPECT_NE(prepared.diagnostic.find("connection was closed"), std::string::npos);
    // 连接不可复用：同一引擎的后续 Prepare 按未打开拒绝（不触达 SQLite、不伪装可重试）。
    const auto retried = engine.Prepare();
    ASSERT_FALSE(retried.ok());
    EXPECT_EQ(retried.code, DbCode::INVALID_STATE);
    // 关闭收尾仍幂等成功。
    ASSERT_TRUE(engine.Close().ok());
    // 普通可回滚失败的重试合同不受影响：全新引擎（新连接）在对象修复后发布成功。
    {
        ExecDirectSql(path, "DROP VIEW FaultTab");
        SQLiteEngine fresh(path, &Ddl::kSqliteFaultTestDatabase, kFaultProfile);
        ASSERT_TRUE(fresh.Open(MakeOpenRequest()).ok());
        ASSERT_TRUE(fresh.Prepare().ok()) << "a fresh engine must retry publication normally";
    }
    for (const char* suffix : {"", "-wal", "-shm"}) {
        std::remove((path + suffix).c_str());
    }
}

} // namespace
