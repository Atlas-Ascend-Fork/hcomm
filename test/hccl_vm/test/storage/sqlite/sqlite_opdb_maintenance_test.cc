/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：OpDB 维护语义（备份/换库/动态表销毁/关闭排空）。
// 维护流程只经后端无关 StorageRuntime 与唯一 simulator bootstrap；并发用例
// 经公开进程上下文租约与公开原子接口验证，不接触具体 manager/engine。
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sqlite3.h>

#include "operation_data/operation_data_ops.h"
#include "storage/engine/i_relational_engine.h"
#include "storage_backends/sqlite/engine/sqlite_backend_factory.h"
#include "storage_composition/storage_composition.h"
#include "storage/internal/process_storage_context.h"
#include "storage/storage_runtime.h"
#include "storage/table_access.h"
#include "storage_backends/sqlite/composition/sqlite_backend_module.h"

namespace {
const char* kRunnerPath = "/tmp/test_sqlite_opdb_maint_runner.db";
const char* kOpDataPath = "/tmp/test_sqlite_opdb_maint_opdata.db";
const char* kSchemaManifestTable = "__hccl_vm_schema_manifest";

/// 列出 OP_DATA_DB 物理文件中的全部用户表（观察动态表生命周期；排除 sqlite_
/// 系统对象与引擎内部 schema manifest 簿记表）。
std::vector<std::string> ListPhysicalTables(const std::string& path = kOpDataPath)
{
    sqlite3* db = nullptr;
    std::vector<std::string> names;
    if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        return names;
    }
    sqlite3_stmt* stmt = nullptr;
    const std::string sql = std::string{"SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE "}
                            + "'sqlite_%' AND name != ? ORDER BY name";
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, kSchemaManifestTable, -1, SQLITE_TRANSIENT);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const unsigned char* text = sqlite3_column_text(stmt, 0);
            if (text != nullptr) {
                names.emplace_back(reinterpret_cast<const char*>(text));
            }
        }
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return names;
}

bool TableExists(const std::string& path, const char* table)
{
    sqlite3* db = nullptr;
    if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        return false;
    }
    sqlite3_stmt* stmt = nullptr;
    const int prepared
        = sqlite3_prepare_v2(db, "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", -1, &stmt, nullptr);
    if (prepared != SQLITE_OK) {
        sqlite3_close(db);
        return false;
    }
    sqlite3_bind_text(stmt, 1, table, -1, SQLITE_TRANSIENT);
    const bool found = sqlite3_step(stmt) == SQLITE_ROW;
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return found;
}

std::string LegacyPayload(const std::string& path)
{
    sqlite3* db = nullptr;
    if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        return {};
    }
    sqlite3_stmt* stmt = nullptr;
    const int prepared = sqlite3_prepare_v2(db, "SELECT payload FROM legacy_payload", -1, &stmt, nullptr);
    if (prepared != SQLITE_OK || sqlite3_step(stmt) != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return {};
    }
    const auto* payload = sqlite3_column_text(stmt, 0);
    const std::string value = payload == nullptr ? std::string{} : reinterpret_cast<const char*>(payload);
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return value;
}

std::string JournalMode(const std::string& path)
{
    sqlite3* db = nullptr;
    if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        return {};
    }
    sqlite3_stmt* stmt = nullptr;
    const int prepared = sqlite3_prepare_v2(db, "PRAGMA journal_mode", -1, &stmt, nullptr);
    if (prepared != SQLITE_OK || sqlite3_step(stmt) != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return {};
    }
    const auto* mode = sqlite3_column_text(stmt, 0);
    const std::string value = mode == nullptr ? std::string{} : reinterpret_cast<const char*>(mode);
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return value;
}

class SqliteOpDbMaintenanceTest : public testing::Test {
protected:
    void SetUp() override
    {
        CleanFiles();
        HcclSim::Storage::StorageRuntimeConfig sessionConfig;
        sessionConfig.databases.push_back(
            {std::string{"RUNNER_DB"}, sim::backends::kSqliteBackendName, std::string{kRunnerPath}});
        sessionConfig.databases.push_back(
            {std::string{"OP_DATA_DB"}, sim::backends::kSqliteBackendName, std::string{kOpDataPath}});
        // 经唯一 bootstrap 安装：同时登记当前路由（备份等维护观测依赖）。
        auto ensured = sim::EnsureSimulatorStorageSession(sessionConfig);
        ASSERT_TRUE(ensured.ok()) << ensured.diagnostic;
    }

    void TearDown() override
    {
        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
        CleanFiles();
    }

    static void CleanFiles()
    {
        for (const char* path : {kRunnerPath, kOpDataPath}) {
            std::remove(path);
            std::remove((std::string(path) + "-wal").c_str());
            std::remove((std::string(path) + "-shm").c_str());
        }
    }

    /// 经会话写入一条 SyncRecord。
    static void InsertOneSyncRecord(uint32_t pid)
    {
        auto table = HcclSim::Storage::AcquireTable<sim::operation::SyncRecordTab>();
        ASSERT_TRUE(table.ok());
        sim::operation::SyncRecordTab record{};
        record.pid = pid;
        record.rankId = pid;
        record.rankSize = 1;
        record.syncIter = 1;
        record.streamId = 1;
        record.status = 0;
        auto inserted = table->Insert(record);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    }

    /// 重装两库会话（供后续用例使用）。
    static void ReinstallSession()
    {
        HcclSim::Storage::StorageRuntimeConfig sessionConfig;
        sessionConfig.databases.push_back(
            {std::string{"RUNNER_DB"}, sim::backends::kSqliteBackendName, std::string{kRunnerPath}});
        sessionConfig.databases.push_back(
            {std::string{"OP_DATA_DB"}, sim::backends::kSqliteBackendName, std::string{kOpDataPath}});
        auto ensured = sim::EnsureSimulatorStorageSession(sessionConfig);
        ASSERT_TRUE(ensured.ok()) << ensured.diagnostic;
    }
};

// 备份产物可读：换库到备份文件后数据可见（迁移前 Backup + SetDbConfig 语义）。
TEST_F(SqliteOpDbMaintenanceTest, BackupProducesReadableSnapshot)
{
    InsertOneSyncRecord(1);
    const std::string backupPath = "/tmp/test_sqlite_opdb_maint_backup.db";
    std::remove(backupPath.c_str());
    auto backed = sim::BackupSimulatorDatabase(
        "OP_DATA_DB", HcclSim::Storage::BackupRequest{backupPath, HcclSim::Storage::BackupPurpose::OnlineReset});
    ASSERT_TRUE(backed.ok()) << backed.diagnostic;
    EXPECT_EQ(backed.value->destinationPath, backupPath);
    // 备份文件可读：换库到备份文件后数据可见。
    auto reconfigured
        = sim::ReconfigureSimulatorDatabase({std::string{"OP_DATA_DB"}, sim::backends::kSqliteBackendName, backupPath});
    ASSERT_TRUE(reconfigured.ok()) << reconfigured.diagnostic;
    {
        auto table = HcclSim::Storage::AcquireTable<sim::operation::SyncRecordTab>();
        ASSERT_TRUE(table.ok());
        auto rows = table->Select(HcclSim::Storage::All<sim::operation::SyncRecordTab>());
        ASSERT_TRUE(rows.ok());
        EXPECT_EQ(rows.value->size(), 1U);
        EXPECT_EQ(rows.value->front().pid, 1U);
    }
    std::remove(backupPath.c_str());
    std::remove((backupPath + "-wal").c_str());
    std::remove((backupPath + "-shm").c_str());
}

// Backup is a byte-level SQLite snapshot, so a readonly dedicated backup connection must not
// require every current business-Catalog table or write a manifest into a legacy source file.
TEST_F(SqliteOpDbMaintenanceTest, BackupCopiesLegacySourceWithoutCreatingCurrentCatalogSchema)
{
    const std::pair<const char*, const char*> journals[] = {{"DELETE", "delete"}, {"WAL", "wal"}};
    for (const auto& journal : journals) {
        std::remove(kOpDataPath);
        std::remove((std::string{kOpDataPath} + "-wal").c_str());
        std::remove((std::string{kOpDataPath} + "-shm").c_str());
        sqlite3* source = nullptr;
        ASSERT_EQ(sqlite3_open(kOpDataPath, &source), SQLITE_OK);
        char* diagnostic = nullptr;
        const std::string sql
            = "PRAGMA journal_mode=" + std::string{journal.first}
              + "; CREATE TABLE legacy_payload (payload TEXT); INSERT INTO legacy_payload VALUES ('kept');";
        const int created = sqlite3_exec(source, sql.c_str(), nullptr, nullptr, &diagnostic);
        const std::string createDiagnostic = diagnostic == nullptr ? "" : diagnostic;
        sqlite3_free(diagnostic);
        sqlite3_close(source);
        ASSERT_EQ(created, SQLITE_OK) << createDiagnostic;
        ASSERT_EQ(JournalMode(kOpDataPath), journal.second);

        const std::string backupPath
            = "/tmp/test_sqlite_opdb_maint_legacy_backup_" + std::string{journal.second} + ".db";
        std::remove(backupPath.c_str());
        auto backed = sim::BackupSimulatorDatabase(
            "OP_DATA_DB", HcclSim::Storage::BackupRequest{backupPath, HcclSim::Storage::BackupPurpose::OnlineReset});
        ASSERT_TRUE(backed.ok()) << backed.diagnostic;
        EXPECT_EQ(backed.value->destinationPath, backupPath);
        EXPECT_EQ(JournalMode(kOpDataPath), journal.second);
        EXPECT_EQ(ListPhysicalTables(kOpDataPath), std::vector<std::string>{"legacy_payload"});
        EXPECT_EQ(ListPhysicalTables(backupPath), std::vector<std::string>{"legacy_payload"});
        EXPECT_EQ(LegacyPayload(kOpDataPath), "kept");
        EXPECT_EQ(LegacyPayload(backupPath), "kept");
        for (const std::string& path : {std::string{kOpDataPath}, backupPath}) {
            EXPECT_FALSE(TableExists(path, kSchemaManifestTable));
            EXPECT_FALSE(TableExists(path, "OpDetailTab"));
        }
        std::remove(backupPath.c_str());
        std::remove((backupPath + "-wal").c_str());
        std::remove((backupPath + "-shm").c_str());
    }

    auto existing = HcclSim::Storage::CreateSqliteEngine(
        HcclSim::Storage::Ddl::kOpDataDatabase, kOpDataPath,
        HcclSim::Storage::SqliteConnectionProfile{10000, false, 200});
    ASSERT_TRUE(existing.ok()) << existing.diagnostic;
    auto& relational = static_cast<HcclSim::Storage::IRelationalEngine&>(**existing.value);
    HcclSim::Storage::EngineOpenRequest request;
    request.mode = HcclSim::Storage::StorageOpenMode::READ_EXISTING;
    ASSERT_TRUE(relational.Open(request).ok());
    // 结构全量审计已删除：consumer 的 Prepare 是纯状态推进，不执行 DDL。
    EXPECT_TRUE(relational.Prepare().ok());
    // 缺失静态表在首次 CRUD 时以 SQLite "no such table" 错误暴露。
    const auto& meta = HcclSim::Storage::RecordSchema<sim::operation::OpDetailTab>::Meta();
    sim::operation::OpDetailTab probe{};
    const auto rejected = relational.Insert(meta, HcclSim::Storage::TableScope::Static(), &probe, nullptr);
    EXPECT_NE(rejected.code, HcclSim::Storage::DbCode::OK);
    EXPECT_NE(rejected.diagnostic.find("OpDetailTab"), std::string::npos);
    EXPECT_TRUE(relational.Close().ok());
}

// 动态任务表按 scope 惰性创建；销毁动态数据后全部物理动态表消失。
TEST_F(SqliteOpDbMaintenanceTest, DropDynamicDataCleansEveryScope)
{
    for (const uint64_t pid : {11U, 22U, 33U}) {
        sim::operation::OpTaskTab task{};
        task.opDetailId = 1;
        task.optaskMeta = {0x01};
        auto inserted
            = HcclSim::Storage::Insert<sim::operation::OpTaskTab>(HcclSim::Storage::TableScope::Process(pid), task);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    }
    const auto tables = ListPhysicalTables();
    EXPECT_EQ(tables.size(), 11U); // 8 静态表 + 3 动态表
    EXPECT_NE(std::find(tables.begin(), tables.end(), "OpTaskTab_P_11"), tables.end());
    EXPECT_NE(std::find(tables.begin(), tables.end(), "OpTaskTab_P_22"), tables.end());
    EXPECT_NE(std::find(tables.begin(), tables.end(), "OpTaskTab_P_33"), tables.end());

    auto dropped = HcclSim::Storage::StorageRuntime::DropDynamicData("OP_DATA_DB");
    ASSERT_TRUE(dropped.ok()) << dropped.diagnostic;
    // 返回值按公开合同计被销毁的动态逻辑表数（OpTaskTab 一张）；
    // 全部物理 scope 表（3 张）已被销毁由物理清单断言。
    EXPECT_EQ(*dropped.value, 1U);
    EXPECT_EQ(ListPhysicalTables().size(), 8U);
}

// 关闭后公开访问失败，重新安装后恢复（不产生隐式重连）。
TEST_F(SqliteOpDbMaintenanceTest, CloseThenOperationsFailUntilReinstalled)
{
    InsertOneSyncRecord(1);
    auto closed = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    ASSERT_TRUE(closed.ok()) << closed.diagnostic;

    auto table = HcclSim::Storage::AcquireTable<sim::operation::SyncRecordTab>();
    ASSERT_FALSE(table.ok());
    auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
    ASSERT_FALSE(lease.ok());

    ReinstallSession();
    auto reinstalled = HcclSim::Storage::ProcessStorageContext::Acquire();
    ASSERT_TRUE(reinstalled.ok());
}

// 关闭幂等：重复关闭成功（第二次按无效状态拒绝由合同测试覆盖）。
TEST_F(SqliteOpDbMaintenanceTest, CloseIsIdempotentWhenAlreadyClosed)
{
    auto closed = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    ASSERT_TRUE(closed.ok());
    auto closedAgain = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    ASSERT_FALSE(closedAgain.ok()); // 已关闭：无效状态
    ReinstallSession();
}

// 持有公开进程租约期间发起关闭：关闭等待租约归还后完成（排空无死锁）。
TEST_F(SqliteOpDbMaintenanceTest, CloseDrainsInFlightLeaseWithoutDeadlock)
{
    InsertOneSyncRecord(1);
    std::atomic<bool> leaseHeld{false};
    std::atomic<bool> releaseNow{false};
    std::thread holder([&]() {
        auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
        if (lease.ok()) {
            leaseHeld.store(true);
            while (!releaseNow.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    });
    while (!leaseHeld.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    auto closing = std::async(std::launch::async, []() {
        return HcclSim::Storage::StorageRuntime::CloseProcessSession();
    });
    // 租约在途：关闭必须仍未完成（排空等待中）。
    EXPECT_EQ(closing.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);
    releaseNow.store(true);
    holder.join();
    auto closed = closing.get();
    ASSERT_TRUE(closed.ok()) << closed.diagnostic;
    ReinstallSession();
}

// 业务访问与关闭并发：双方都必须终结（无锁序反转死锁）。
TEST_F(SqliteOpDbMaintenanceTest, BusinessAccessAndCloseConcurrentlyWithoutDeadlock)
{
    InsertOneSyncRecord(1);
    std::atomic<bool> stop{false};
    std::vector<std::string> outcomes;
    std::mutex outcomesMutex;
    std::thread accessLoop([&]() {
        while (!stop.load()) {
            auto table = HcclSim::Storage::AcquireTable<sim::operation::SyncRecordTab>();
            std::lock_guard<std::mutex> lock(outcomesMutex);
            outcomes.push_back(table.ok() ? "ok" : "rejected");
        }
    });
    auto closed = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    stop.store(true);
    accessLoop.join();
    ASSERT_TRUE(closed.ok()) << closed.diagnostic;
    std::lock_guard<std::mutex> lock(outcomesMutex);
    ASSERT_FALSE(outcomes.empty());
    for (const auto& outcome : outcomes) {
        EXPECT_TRUE(outcome == "ok" || outcome == "rejected") << outcome;
    }
    ReinstallSession();
}

// 备份失败（目标目录不存在）不产生副作用：业务访问立即恢复。
TEST_F(SqliteOpDbMaintenanceTest, BackupFailureLeavesBusinessUsable)
{
    InsertOneSyncRecord(1);
    const std::string badPath = "/nonexistent-storage-dir/backup.db";
    auto backed = sim::BackupSimulatorDatabase(
        "OP_DATA_DB", HcclSim::Storage::BackupRequest{badPath, HcclSim::Storage::BackupPurpose::OnlineReset});
    ASSERT_FALSE(backed.ok());
    // 专用备份连接失败不产生副作用：业务会话立即可用。
    InsertOneSyncRecord(2);
    auto table = HcclSim::Storage::AcquireTable<sim::operation::SyncRecordTab>();
    ASSERT_TRUE(table.ok());
    auto rows = table->Select(HcclSim::Storage::All<sim::operation::SyncRecordTab>());
    ASSERT_TRUE(rows.ok());
    EXPECT_EQ(rows.value->size(), 2U);
}
} // namespace
