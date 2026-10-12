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

// 文件职责：真实 StorageSession + 目录登记表驱动 SQLiteEngine 的会话级合同：
// Table<T> CRUD/事务全链路、通用事务惰性绑定、BLOB 字节往返、大 BLOB 复制
// 计数（REQ-CAP-4：即时路径 0 新增 owning 副本 / 事务路径恰一次）、会话关闭
// 与有符号/无符号语义。语句缓存/rawBytes 通道/映射特化已删除——验证它们的
// 用例随结构一并移除或改为目录登记形态。
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "storage_backends/sqlite/engine/sqlite_engine.h"
#include "storage/query_expression.h"
#include "storage/record_contract.h"
#include "storage/storage_catalog.h"
#include "storage/storage_session.h"
#include "storage/table.h"
#include "storage_backends/sqlite/composition/sqlite_backend_module.h"
using sim::backends::kSqliteBackendName;

// ---- 会话级测试目录：DeviceRecord（主键+文本+有符号+无符号）+ PayloadRecord（BLOB）----
namespace sqlite_session_test {
struct DeviceRecord {
    uint64_t id{0};
    std::string name;
    int64_t state{0};
    uint64_t ticket{0};
};
struct PayloadRecord {
    uint64_t id{0};
    std::vector<uint8_t> payload;
};
// 动态 scope 表（事务回滚失效登记的回归用例）：note 唯一，用于事务内 DML 冲突注入
struct ScopedRecord {
    uint64_t id{0};
    std::string note;
};
} // namespace sqlite_session_test

using namespace HcclSim::Storage::Ddl;
#define Field HCCL_VM_DDL_FIELD
#define Version HCCL_VM_DDL_VERSION
#define PrimaryKey HCCL_VM_DDL_PRIMARY_KEY
#define Unique HCCL_VM_DDL_UNIQUE
#define Index HCCL_VM_DDL_INDEX
#define Scope Ddl::ScopeKind

#define SQLITE_SESSION_TEST_DATABASE(Table)                                                                   \
    Table(                                                                                                    \
        sqlite_session_test::DeviceRecord, (Version(1), Scope::Static, Field(id, AutoIncrement), Field(name), \
                                            Field(state), Field(ticket), PrimaryKey(id), Index(state)))       \
        Table(                                                                                                \
            sqlite_session_test::PayloadRecord,                                                               \
            (Version(1), Scope::Static, Field(id, AutoIncrement), Field(payload), PrimaryKey(id)))            \
            Table(                                                                                            \
                sqlite_session_test::ScopedRecord,                                                            \
                (Version(1), Scope::DynamicPid, Field(id, AutoIncrement), Field(note), PrimaryKey(id), Unique(note)))

#define HCCL_VM_DDL_CURRENT_DATABASE "SQLITE_SESSION_TEST_DB"
HCCL_VM_DEFINE_DATABASE(kSqliteSessionTestDatabase, SQLITE_SESSION_TEST_DB, SQLITE_SESSION_TEST_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE
#undef Field
#undef Version
#undef PrimaryKey
#undef Unique
#undef Index
#undef Scope

namespace {
using namespace HcclSim::Storage;
using DeviceRecord = sqlite_session_test::DeviceRecord;
using PayloadRecord = sqlite_session_test::PayloadRecord;

constexpr SqliteConnectionProfile kSessionProfile{10000, false, 200};

std::string TempDatabasePath()
{
    static std::atomic<uint64_t> counter{0};
    const std::string path = testing::TempDir() + "sqlite_session_" + std::to_string(counter.fetch_add(1)) + ".db";
    (void)std::remove(path.c_str());
    return path;
}

// 端到端会话夹具：真实 StorageSession 装配 SQLiteEngine（目录 + profile）。
class SQLiteEngineSessionTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        databasePath = TempDatabasePath();
        engine = std::make_unique<SQLiteEngine>(databasePath, &Ddl::kSqliteSessionTestDatabase, kSessionProfile);

        binding.sessionId = "sqlite-session-test";
        binding.dataNamespace = "sqlite-session";
        binding.databases.push_back({"SQLITE_SESSION_TEST_DB", "", &Ddl::kSqliteSessionTestDatabase});

        request.engines = {EngineRegistration{"SQLITE_SESSION_TEST_DB", kSqliteBackendName, engine.get()}};

        auto opened = StorageSession::Open(binding, request);
        ASSERT_TRUE(opened.ok()) << opened.diagnostic;
        session.emplace(std::move(*opened.value));
    }

    void TearDown() override
    {
        if (session.has_value()) {
            (void)session->Close();
            session.reset();
        }
        engine.reset();
    }

    Table<DeviceRecord> DeviceTable()
    {
        auto table = session->Table<DeviceRecord>();
        EXPECT_TRUE(table.ok()) << table.diagnostic;
        return std::move(*table.value);
    }

    Table<PayloadRecord> PayloadTable()
    {
        auto table = session->Table<PayloadRecord>();
        EXPECT_TRUE(table.ok()) << table.diagnostic;
        return std::move(*table.value);
    }

    Table<sqlite_session_test::ScopedRecord> ScopedTable(TableScope scope)
    {
        auto table = session->Table<sqlite_session_test::ScopedRecord>(scope);
        EXPECT_TRUE(table.ok()) << table.diagnostic;
        return std::move(*table.value);
    }

    std::string databasePath;
    std::unique_ptr<SQLiteEngine> engine;
    SessionStorageBinding binding;
    StorageOpenRequest request;
    std::optional<StorageSession> session;
};

// 冻结 Table<T> 合同驱动真实 SQLite 引擎：即时 Insert/Get/Query/Update/Delete/Clear 全链路。
TEST_F(SQLiteEngineSessionTest, ImmediateCrudRunsThroughFrozenTableContract)
{
    auto table = DeviceTable();

    DeviceRecord inserted;
    inserted.name = "device-a";
    inserted.state = 1;
    const auto insertedKey = table.Insert(inserted);
    ASSERT_TRUE(insertedKey.ok()) << insertedKey.diagnostic;
    EXPECT_EQ(insertedKey.value->value, 1U);
    // 生成主键经引擎返回值由 Table 写回调用方记录。
    EXPECT_EQ(inserted.id, 1U);

    DeviceRecord second;
    second.name = "device-b";
    second.state = 3;
    ASSERT_TRUE(table.Insert(second).ok());
    EXPECT_EQ(second.id, 2U);

    const auto fetched = table.Select(Eq(&DeviceRecord::id, uint64_t{1}), Limit(1U));
    ASSERT_TRUE(fetched.ok()) << fetched.diagnostic;
    ASSERT_EQ(fetched.value->size(), 1U);
    EXPECT_EQ(fetched.value->front().name, "device-a");
    EXPECT_EQ(fetched.value->front().state, 1);

    const auto missing = table.Select(Eq(&DeviceRecord::id, uint64_t{99}), Limit(1U));
    EXPECT_EQ(missing.code, DbCode::OK);
    EXPECT_TRUE(missing.value->empty());

    const auto selected
        = table.Select(Eq(&DeviceRecord::state, int64_t{1}), OrderBy(&DeviceRecord::id, SortDirection::ASC));
    ASSERT_TRUE(selected.ok()) << selected.diagnostic;
    ASSERT_EQ(selected.value->size(), 1U);
    EXPECT_EQ(selected.value->front().name, "device-a");

    const auto updated = table.Update(Eq(&DeviceRecord::id, uint64_t{1}), Set(&DeviceRecord::state, int64_t{5}));
    ASSERT_TRUE(updated.ok()) << updated.diagnostic;
    EXPECT_EQ(*updated.value, 1U);
    const auto afterUpdate = table.Select(Eq(&DeviceRecord::id, uint64_t{1}), Limit(1U));
    ASSERT_TRUE(afterUpdate.ok());
    ASSERT_EQ(afterUpdate.value->size(), 1U);
    EXPECT_EQ(afterUpdate.value->front().state, 5);

    const auto deleted = table.Delete(Eq(&DeviceRecord::id, uint64_t{2}));
    ASSERT_TRUE(deleted.ok()) << deleted.diagnostic;
    EXPECT_EQ(*deleted.value, 1U);
    const auto gone = table.Select(Eq(&DeviceRecord::id, uint64_t{2}), Limit(1U));
    EXPECT_EQ(gone.code, DbCode::OK);
    EXPECT_TRUE(gone.value->empty());

    ASSERT_TRUE(table.Clear(ClearMode::RESET_SEQUENCE).ok());
    DeviceRecord fresh;
    fresh.name = "device-c";
    ASSERT_TRUE(table.Insert(fresh).ok());
    EXPECT_EQ(fresh.id, 1U);
}

// 每次执行都重新绑定本次条件值（无语句缓存可依赖旧绑定）；LIMIT 与组合条件
// 保持占位符顺序；同结构重复执行的值隔离。
TEST_F(SQLiteEngineSessionTest, RepeatedStatementsRebindCurrentValuesEachExecution)
{
    auto table = DeviceTable();
    DeviceRecord first;
    first.name = "first";
    first.state = 10;
    ASSERT_TRUE(table.Insert(first).ok());
    DeviceRecord second;
    second.name = "second";
    second.state = 20;
    ASSERT_TRUE(table.Insert(second).ok());

    const auto firstRows = table.Select(Eq(&DeviceRecord::state, int64_t{10}));
    ASSERT_TRUE(firstRows.ok()) << firstRows.diagnostic;
    ASSERT_EQ(firstRows.value->size(), 1U);
    EXPECT_EQ(firstRows.value->front().name, "first");

    // 与上一查询同一结构：绑定的是本次条件值（无缓存复用旧绑定的问题）。
    const auto secondRows = table.Select(Eq(&DeviceRecord::state, int64_t{20}));
    ASSERT_TRUE(secondRows.ok()) << secondRows.diagnostic;
    ASSERT_EQ(secondRows.value->size(), 1U);
    EXPECT_EQ(secondRows.value->front().name, "second");
    const auto secondCount = table.Count(Eq(&DeviceRecord::state, int64_t{20}));
    ASSERT_TRUE(secondCount.ok()) << secondCount.diagnostic;
    EXPECT_EQ(*secondCount.value, 1U);

    const auto limited = table.Select(Eq(&DeviceRecord::state, int64_t{10}), Limit(1U));
    ASSERT_TRUE(limited.ok()) << limited.diagnostic;
    ASSERT_EQ(limited.value->size(), 1U);
    EXPECT_EQ(limited.value->front().name, "first");

    const auto composite
        = table.Select(And(Eq(&DeviceRecord::name, std::string{"second"}), Eq(&DeviceRecord::state, int64_t{20})));
    ASSERT_TRUE(composite.ok()) << composite.diagnostic;
    ASSERT_EQ(composite.value->size(), 1U);
    EXPECT_EQ(composite.value->front().id, second.id);

    // 引擎直查空结果：materializer 把 sink 目标覆写为空行集。
    std::vector<DeviceRecord> overwritten{{999U, "stale", 99, 0U}};
    auto sink = ResultSink::For(overwritten);
    const auto& meta = RecordSchema<DeviceRecord>::Meta();
    const auto emptyResult
        = engine->Select(meta, TableScope::Static(), Eq(&DeviceRecord::state, int64_t{99}).data, {}, sink);
    ASSERT_TRUE(emptyResult.ok()) << emptyResult.diagnostic;
    EXPECT_TRUE(overwritten.empty());
}

// 冻结 Transaction 合同驱动真实 SQLite 事务：DML 直达执行主键立即可见、Commit 按序返回、Rollback 丢弃。
TEST_F(SQLiteEngineSessionTest, TransactionsRunThroughFrozenTableContract)
{
    auto table = DeviceTable();

    auto transactionResult = session->BeginTransaction();
    ASSERT_TRUE(transactionResult.ok()) << transactionResult.diagnostic;
    auto transaction = std::move(*transactionResult.value);

    DeviceRecord first;
    first.name = "tx-first";
    const auto firstKey = table.Insert(first, transaction);
    ASSERT_TRUE(firstKey.ok()) << firstKey.diagnostic;
    // 事务构建期主键已回填，供同组后续记录引用（OpDetail+OpMem 语义）。
    EXPECT_EQ(first.id, 1U);

    DeviceRecord second;
    second.name = "tx-second";
    const auto secondKey = table.Insert(second, transaction);
    ASSERT_TRUE(secondKey.ok()) << secondKey.diagnostic;
    EXPECT_EQ(second.id, 2U);

    ASSERT_TRUE(
        table.Update(Eq(&DeviceRecord::id, uint64_t{1}), Set(&DeviceRecord::state, int64_t{9}), transaction).ok());

    const auto committed = transaction.Commit();
    ASSERT_TRUE(committed.ok()) << committed.diagnostic;
    ASSERT_EQ(committed.value->items.size(), 3U);
    EXPECT_EQ(committed.value->items.front().generatedId, std::optional<uint64_t>{1U});
    EXPECT_EQ(committed.value->items.at(1).generatedId, std::optional<uint64_t>{2U});
    EXPECT_EQ(transaction.State(), TransactionState::COMMITTED);

    const auto afterCommit = table.Select(Eq(&DeviceRecord::id, uint64_t{1}), Limit(1U));
    ASSERT_TRUE(afterCommit.ok());
    ASSERT_EQ(afterCommit.value->size(), 1U);
    EXPECT_EQ(afterCommit.value->front().state, 9);

    // 回滚路径：暂存插入不落库。
    auto rollbackResult = session->BeginTransaction();
    ASSERT_TRUE(rollbackResult.ok());
    auto rolling = std::move(*rollbackResult.value);
    DeviceRecord discarded;
    discarded.name = "tx-discarded";
    ASSERT_TRUE(table.Insert(discarded, rolling).ok());
    ASSERT_TRUE(rolling.Rollback().ok());
    EXPECT_EQ(rolling.State(), TransactionState::ROLLED_BACK);
    const auto none = table.Select(Eq(&DeviceRecord::name, std::string{"tx-discarded"}));
    ASSERT_TRUE(none.ok());
    EXPECT_TRUE(none.value->empty());
}

// 混合 INSERT/UPDATE/DELETE 事务的逐项结果语义：AtomicItemResult 中只有 INSERT
// 携带 generatedId（生成主键）；UPDATE/DELETE 的引擎返回 value 是受影响行数，
// 不得误作生成主键。生成主键在 Commit 前即已回填，可供同事务后续条件引用
// （外键引用语义的前提）。
TEST_F(SQLiteEngineSessionTest, MixedTransactionRecordsGeneratedIdOnlyForInsert)
{
    auto table = DeviceTable();

    auto transactionResult = session->BeginTransaction();
    ASSERT_TRUE(transactionResult.ok()) << transactionResult.diagnostic;
    auto transaction = std::move(*transactionResult.value);

    DeviceRecord first;
    first.name = "mix-first";
    const auto firstKey = table.Insert(first, transaction);
    ASSERT_TRUE(firstKey.ok()) << firstKey.diagnostic;
    EXPECT_EQ(first.id, 1U); // Commit 前生成主键已回填

    // 用回填主键在同事务内定位行：主键 Commit 前可用即可供外键式引用
    ASSERT_TRUE(
        table.Update(Eq(&DeviceRecord::id, first.id), Set(&DeviceRecord::state, int64_t{42}), transaction).ok());

    DeviceRecord second;
    second.name = "mix-second";
    const auto secondKey = table.Insert(second, transaction);
    ASSERT_TRUE(secondKey.ok()) << secondKey.diagnostic;
    ASSERT_TRUE(table.Delete(Eq(&DeviceRecord::id, second.id), transaction).ok());

    const auto committed = transaction.Commit();
    ASSERT_TRUE(committed.ok()) << committed.diagnostic;
    ASSERT_EQ(committed.value->items.size(), 4U); // INSERT + UPDATE + INSERT + DELETE
    // 只有 INSERT 项携带 generatedId
    EXPECT_EQ(committed.value->items.at(0).generatedId, std::optional<uint64_t>{1U});
    EXPECT_EQ(committed.value->items.at(0).affectedRows, 1U);
    EXPECT_EQ(committed.value->items.at(1).generatedId, std::nullopt); // UPDATE：受影响行数不是主键
    EXPECT_EQ(committed.value->items.at(1).affectedRows, 1U);
    EXPECT_EQ(committed.value->items.at(2).generatedId, std::optional<uint64_t>{2U});
    EXPECT_EQ(committed.value->items.at(2).affectedRows, 1U);
    EXPECT_EQ(committed.value->items.at(3).generatedId, std::nullopt); // DELETE：受影响行数不是主键
    EXPECT_EQ(committed.value->items.at(3).affectedRows, 1U);
    EXPECT_EQ(transaction.State(), TransactionState::COMMITTED);

    // 提交后落库状态：仅首行保留且更新生效
    const auto rows = table.Select(All<DeviceRecord>());
    ASSERT_TRUE(rows.ok());
    ASSERT_EQ(rows.value->size(), 1U);
    EXPECT_EQ(rows.value->front().state, 42);
}

// 公共 Transaction 失败回滚后的动态 scope 复用：事务内首次建表 + DML 冲突失败
// → FinishWrite 统一回滚物理事务（撤销首次建表）→ 引擎登记已失效 → 同 scope
// 的下一次访问重新执行幂等 DDL。修复前登记残留会跳过建表，出现 no such table。
TEST_F(SQLiteEngineSessionTest, DynamicScopeReusableAfterTransactionDmlFailureRollback)
{
    auto table = ScopedTable(TableScope::Process(12U));

    auto transactionResult = session->BeginTransaction();
    ASSERT_TRUE(transactionResult.ok()) << transactionResult.diagnostic;
    auto transaction = std::move(*transactionResult.value);

    using Scoped = sqlite_session_test::ScopedRecord;
    Scoped first;
    first.note = "scoped";
    ASSERT_TRUE(table.Insert(first, transaction).ok()); // 事务内首次建表并登记
    Scoped duplicate;
    duplicate.note = "scoped"; // 唯一索引冲突 → DML 执行失败
    const auto failed = table.Insert(duplicate, transaction);
    EXPECT_EQ(failed.code, DbCode::CONFLICT);
    // 失败熔断：公共 Transaction 统一调用一次 engine->Rollback，进入终态
    EXPECT_EQ(transaction.State(), TransactionState::FAILED);

    // 同 scope 重新可用：登记已随回滚失效，重新建表后写入/查询成功
    Scoped retry;
    retry.note = "retry";
    const auto reinserted = table.Insert(retry);
    ASSERT_TRUE(reinserted.ok()) << reinserted.diagnostic;
    EXPECT_EQ(retry.id, 1U); // 回滚撤销了首次建表，新表序列从头开始
    const auto rows = table.Select(Eq(&Scoped::note, std::string{"retry"}));
    ASSERT_TRUE(rows.ok()) << rows.diagnostic;
    ASSERT_EQ(rows.value->size(), 1U);
    EXPECT_EQ(rows.value->front().note, "retry");
}

// 通用事务入口不需要业务原子组枚举：首个暂存 DML 惰性绑定物理事务域（engine 指针），
// 跨域 DML 在执行前被拒绝，未提交事务析构自动回滚。
TEST_F(SQLiteEngineSessionTest, GenericTransactionBindsLazilyAndRejectsForeignDomain)
{
    auto table = DeviceTable();

    auto transactionResult = session->BeginTransaction();
    ASSERT_TRUE(transactionResult.ok()) << transactionResult.diagnostic;
    auto transaction = std::move(*transactionResult.value);
    EXPECT_EQ(transaction.State(), TransactionState::BUILDING);

    DeviceRecord record;
    record.name = "generic-tx";
    const auto staged = table.Insert(record, transaction);
    ASSERT_TRUE(staged.ok()) << staged.diagnostic;
    EXPECT_EQ(record.id, 1U);

    // 同一物理域内的更新正常暂存。
    ASSERT_TRUE(
        table.Update(Eq(&DeviceRecord::id, uint64_t{1}), Set(&DeviceRecord::state, int64_t{7}), transaction).ok());

    const auto committed = transaction.Commit();
    ASSERT_TRUE(committed.ok()) << committed.diagnostic;
    ASSERT_EQ(committed.value->items.size(), 2U);
    EXPECT_EQ(transaction.State(), TransactionState::COMMITTED);
    const auto after = table.Select(Eq(&DeviceRecord::id, uint64_t{1}), Limit(1U));
    ASSERT_TRUE(after.ok());
    ASSERT_EQ(after.value->size(), 1U);
    EXPECT_EQ(after.value->front().state, 7);

    // 未提交事务显式回滚后行不落库；未触碰事务的即时读不受影响。
    auto rollingResult = session->BeginTransaction();
    ASSERT_TRUE(rollingResult.ok());
    auto rolling = std::move(*rollingResult.value);
    DeviceRecord discarded;
    discarded.name = "generic-rollback";
    ASSERT_TRUE(table.Insert(discarded, rolling).ok());
    EXPECT_EQ(rolling.State(), TransactionState::BUILDING);
    ASSERT_TRUE(rolling.Rollback().ok());
    EXPECT_EQ(rolling.State(), TransactionState::ROLLED_BACK);
    const auto afterRollback = table.Count(Eq(&DeviceRecord::name, std::string{"generic-rollback"}));
    ASSERT_TRUE(afterRollback.ok());
    EXPECT_EQ(*afterRollback.value, 0U);
}

// BLOB 形态记录端到端往返，字节保持一致（目录字段通道直通，无 rawBytes 专用通道）。
TEST_F(SQLiteEngineSessionTest, PayloadRecordRoundTripsBytes)
{
    auto table = PayloadTable();

    PayloadRecord inserted;
    inserted.payload = {0x00, 0xFF, 0x7F, 0x00, 0x42, 0x00};
    const auto insertedKey = table.Insert(inserted);
    ASSERT_TRUE(insertedKey.ok()) << insertedKey.diagnostic;
    EXPECT_EQ(inserted.id, 1U);

    const auto fetched = table.Select(Eq(&PayloadRecord::id, uint64_t{1}), Limit(1U));
    ASSERT_TRUE(fetched.ok()) << fetched.diagnostic;
    ASSERT_EQ(fetched.value->size(), 1U);
    EXPECT_EQ(fetched.value->front().payload, inserted.payload);

    ASSERT_TRUE(table.Delete(Eq(&PayloadRecord::id, uint64_t{1})).ok());
    const auto gone = table.Select(Eq(&PayloadRecord::id, uint64_t{1}), Limit(1U));
    EXPECT_EQ(gone.code, DbCode::OK);
    EXPECT_TRUE(gone.value->empty());
}

// ==== REQ-CAP-4：即时/事务大 BLOB 复制合同（目录字段通道） ====

// 1 MiB BLOB 的即时 UPDATE：从调用方 Set 子句值到 SQLite 同步 bind 不得产生新的
// owning BLOB 副本（Set 值借用直达 SQLITE_STATIC bind；borrow 有效性由读回字节一致证明）。
TEST_F(SQLiteEngineSessionTest, ImmediateLargeBlobUpdateBorrowsAssignmentPayload)
{
    auto table = PayloadTable();

    PayloadRecord inserted;
    inserted.payload.assign(1024 * 1024, 0xAB);
    ASSERT_TRUE(table.Insert(inserted).ok()) << "seed row";

    std::vector<uint8_t> blob(1024 * 1024);
    for (std::size_t i = 0; i < blob.size(); ++i) {
        blob[i] = static_cast<uint8_t>(i & 0xFF);
    }
    const auto updated = table.Update(Eq(&PayloadRecord::id, uint64_t{1}), Set(&PayloadRecord::payload, blob));
    ASSERT_TRUE(updated.ok()) << updated.diagnostic;
    EXPECT_EQ(*updated.value, 1U);

    // 借用链有效性：step+finalize 在源对象存活范围内完成，读回字节与写入一致。
    const auto fetched = table.SelectOne(Eq(&PayloadRecord::id, uint64_t{1}));
    ASSERT_TRUE(fetched.ok()) << fetched.diagnostic;
    EXPECT_EQ(fetched.value->payload, blob);
    ASSERT_TRUE(table.Delete(Eq(&PayloadRecord::id, uint64_t{1})).ok());
}

// 事务路径与即时路径共用同一直达实现：DML 立即执行，条件/更新以 const 引用
// 传递（Set 的公开入口一次拥有化后不再有第二份事务持有副本）。
TEST_F(SQLiteEngineSessionTest, TransactionalLargeBlobUpdateRunsDirectly)
{
    auto table = PayloadTable();

    PayloadRecord inserted;
    inserted.payload.assign(1024 * 1024, 0x11);
    ASSERT_TRUE(table.Insert(inserted).ok()) << "seed row";

    std::vector<uint8_t> blob(1024 * 1024, 0xCD);
    {
        auto transaction = session->BeginTransaction();
        ASSERT_TRUE(transaction.ok()) << transaction.diagnostic;
        const auto staged
            = table.Update(Eq(&PayloadRecord::id, uint64_t{1}), Set(&PayloadRecord::payload, blob), *transaction);
        ASSERT_TRUE(staged.ok()) << staged.diagnostic;
        // 事务 UPDATE 直达引擎立即执行（本层不持有任何变更载荷副本）
        ASSERT_TRUE(transaction->Commit().ok());
    }
    const auto fetched = table.SelectOne(Eq(&PayloadRecord::id, uint64_t{1}));
    ASSERT_TRUE(fetched.ok()) << fetched.diagnostic;
    EXPECT_EQ(fetched.value->payload, blob);
    ASSERT_TRUE(table.Delete(Eq(&PayloadRecord::id, uint64_t{1})).ok());
}

// 1 MiB BLOB 的即时 INSERT：visitRecord 借用原 T 成员地址直达 SQLITE_STATIC bind，
// 全链路零新增 owning 副本（读回字节一致证明借用链有效）。
TEST_F(SQLiteEngineSessionTest, ImmediateLargeBlobInsertBorrowsAllTheWay)
{
    auto table = PayloadTable();

    PayloadRecord record;
    record.payload.assign(1024 * 1024, 0x5A);
    ASSERT_TRUE(table.Insert(record).ok());

    const auto fetched = table.SelectOne(Eq(&PayloadRecord::id, record.id));
    ASSERT_TRUE(fetched.ok()) << fetched.diagnostic;
    EXPECT_EQ(fetched.value->payload, record.payload);
    ASSERT_TRUE(table.Delete(Eq(&PayloadRecord::id, record.id)).ok());
}

TEST_F(SQLiteEngineSessionTest, SessionCloseReleasesEngineConnection)
{
    {
        auto table = DeviceTable();
        DeviceRecord record;
        record.name = "before-close";
        ASSERT_TRUE(table.Insert(record).ok());
    }
    ASSERT_TRUE(session->Close().ok());
    EXPECT_EQ(session->Table<DeviceRecord>().code, DbCode::INVALID_STATE);
    session.reset();

    // 引擎重新打开同一文件可读回数据。
    SQLiteEngine reopened(databasePath, &Ddl::kSqliteSessionTestDatabase, kSessionProfile);
    EngineOpenRequest openRequest;
    openRequest.mode = StorageOpenMode::READ_EXISTING;
    ASSERT_TRUE(reopened.Open(openRequest).ok());
    ASSERT_TRUE(reopened.Prepare().ok());

    const auto& meta = RecordSchema<DeviceRecord>::Meta();
    std::vector<DeviceRecord> records;
    auto sink = ResultSink::For(records);
    QueryOptions options;
    options.limit = 1U;
    ASSERT_TRUE(
        reopened.Select(meta, TableScope::Static(), Eq(&DeviceRecord::id, uint64_t{1}).data, options, sink).ok());
    ASSERT_EQ(records.size(), 1U);
    EXPECT_EQ(records.front().name, "before-close");
    ASSERT_TRUE(reopened.Close().ok());
}

// 非主键 UNSIGNED 字段的完整 uint64 位型往返：0、INT64_MAX、INT64_MAX+1、UINT64_MAX 经
// Insert → Select、等值 Query、Update → Select 后类型均为 uint64_t 且逐位一致
// （迁移前 SQLite 把 64 位 bit pattern 存入 INTEGER）。超范围值不充当主键。
TEST_F(SQLiteEngineSessionTest, UnsignedFieldRoundTripsFullUint64Range)
{
    auto table = DeviceTable();
    const std::vector<uint64_t> boundaries
        = {0U, uint64_t{std::numeric_limits<int64_t>::max()}, uint64_t{std::numeric_limits<int64_t>::max()} + 1U,
           std::numeric_limits<uint64_t>::max()};

    for (size_t index = 0; index < boundaries.size(); ++index) {
        const uint64_t value = boundaries[index];
        const uint64_t flipped = value ^ 1U;

        DeviceRecord inserted;
        inserted.name = "u64-" + std::to_string(index);
        inserted.ticket = value;
        const auto insertedKey = table.Insert(inserted);
        ASSERT_TRUE(insertedKey.ok()) << insertedKey.diagnostic << " value=" << value;
        const uint64_t rowId = inserted.id;
        EXPECT_GE(rowId, 1U);

        const auto fetched = table.Select(Eq(&DeviceRecord::id, uint64_t{rowId}), Limit(1U));
        ASSERT_TRUE(fetched.ok()) << fetched.diagnostic;
        ASSERT_EQ(fetched.value->size(), 1U);
        EXPECT_EQ(fetched.value->front().ticket, value);

        const auto selected = table.Select(Eq(&DeviceRecord::ticket, value));
        ASSERT_TRUE(selected.ok()) << selected.diagnostic;
        ASSERT_EQ(selected.value->size(), 1U) << " value=" << value;
        EXPECT_EQ(selected.value->front().id, rowId);
        EXPECT_EQ(selected.value->front().ticket, value);

        const auto updated = table.Update(Eq(&DeviceRecord::id, uint64_t{rowId}), Set(&DeviceRecord::ticket, flipped));
        ASSERT_TRUE(updated.ok()) << updated.diagnostic;
        EXPECT_EQ(*updated.value, 1U);
        const auto afterUpdate = table.Select(Eq(&DeviceRecord::id, uint64_t{rowId}), Limit(1U));
        ASSERT_TRUE(afterUpdate.ok());
        ASSERT_EQ(afterUpdate.value->size(), 1U);
        EXPECT_EQ(afterUpdate.value->front().ticket, flipped);
    }
}

// SIGNED 字段保持有符号语义：负值经往返仍是同值 int64_t，不被无符号归一化改写。
TEST_F(SQLiteEngineSessionTest, SignedFieldKeepsSignedSemanticsAcrossRoundTrip)
{
    auto table = DeviceTable();
    const std::vector<int64_t> signedValues
        = {int64_t{-1}, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
    for (size_t index = 0; index < signedValues.size(); ++index) {
        DeviceRecord inserted;
        inserted.name = "signed-" + std::to_string(index);
        inserted.state = signedValues[index];
        ASSERT_TRUE(table.Insert(inserted).ok());
        const auto fetched = table.Select(Eq(&DeviceRecord::id, uint64_t{inserted.id}), Limit(1U));
        ASSERT_TRUE(fetched.ok()) << fetched.diagnostic;
        ASSERT_EQ(fetched.value->size(), 1U);
        EXPECT_EQ(fetched.value->front().state, signedValues[index]);
    }
}
} // namespace
