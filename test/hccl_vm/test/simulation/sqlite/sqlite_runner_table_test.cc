/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：直接验证进程全局 StorageSession 取得的 Runner Table<T> 行为。
// 测试不经过 RunnerDB、Operations 或其它 CRUD facade，防止业务迁移后出现第二条数据库访问路径。
#include <gtest/gtest.h>

#include <sqlite3.h>

#include <cstdint>
#include <cstdio>
#include <string>

#include "sqlite/simulation_storage_test_helper.h"
#include "storage_composition/storage_composition.h"
#include "storage/internal/process_storage_context.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/storage_session.h"
#include "storage_backends/sqlite/composition/sqlite_backend_module.h"

namespace {
const std::string kRunnerPath = "/tmp/test_sqlite_runner_table_runner.db";
const std::string kOpDataPath = "/tmp/test_sqlite_runner_table_opdata.db";

class IndependentConnection {
public:
    explicit IndependentConnection(const std::string& path)
    {
        opened_ = sqlite3_open_v2(path.c_str(), &db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_URI, nullptr) == SQLITE_OK;
    }

    ~IndependentConnection()
    {
        if (db_ != nullptr) {
            sqlite3_close(db_);
        }
    }

    IndependentConnection(const IndependentConnection&) = delete;
    IndependentConnection& operator=(const IndependentConnection&) = delete;

    bool Ok() const { return opened_; }

    int Exec(const char* sql) { return sqlite3_exec(db_, sql, nullptr, nullptr, nullptr); }

    bool ReadColumnInt64(uint64_t id, const char* column, uint64_t& value, const char* table = "Stream")
    {
        sqlite3_stmt* select = nullptr;
        const std::string sql = "SELECT " + std::string(column) + " FROM " + std::string(table) + " WHERE id = ?";
        if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &select, nullptr) != SQLITE_OK) {
            return false;
        }
        sqlite3_bind_int64(select, 1, static_cast<sqlite3_int64>(id));
        const bool found = sqlite3_step(select) == SQLITE_ROW;
        if (found) {
            value = static_cast<uint64_t>(sqlite3_column_int64(select, 0));
        }
        sqlite3_finalize(select);
        return found;
    }

    bool SetColumnInt64(uint64_t id, const char* column, uint64_t value, const char* table = "Stream")
    {
        sqlite3_stmt* update = nullptr;
        const std::string sql = "UPDATE " + std::string(table) + " SET " + std::string(column) + " = ? WHERE id = ?";
        if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &update, nullptr) != SQLITE_OK) {
            return false;
        }
        sqlite3_bind_int64(update, 1, static_cast<sqlite3_int64>(value));
        sqlite3_bind_int64(update, 2, static_cast<sqlite3_int64>(id));
        const bool done = sqlite3_step(update) == SQLITE_DONE && sqlite3_changes(db_) == 1;
        sqlite3_finalize(update);
        return done;
    }

    uint64_t InsertStreamRow(uint64_t ctxId, uint64_t priority, uint64_t activated)
    {
        sqlite3_stmt* insert = nullptr;
        const char* sql = "INSERT INTO Stream (ctx_id, sq_base_addr, is_primary_default, is_other_default, priority, "
                          "schedule_strategy, failure_mode, user_tag, overflow_switch, activated, capture_status, "
                          "task_complete_status) VALUES (?, 0, 0, 0, ?, 0, 0, 0, 0, ?, 0, 0)";
        if (sqlite3_prepare_v2(db_, sql, -1, &insert, nullptr) != SQLITE_OK) {
            return 0;
        }
        sqlite3_bind_int64(insert, 1, static_cast<sqlite3_int64>(ctxId));
        sqlite3_bind_int64(insert, 2, static_cast<sqlite3_int64>(priority));
        sqlite3_bind_int64(insert, 3, static_cast<sqlite3_int64>(activated));
        const bool done = sqlite3_step(insert) == SQLITE_DONE;
        sqlite3_finalize(insert);
        return done ? static_cast<uint64_t>(sqlite3_last_insert_rowid(db_)) : 0;
    }

private:
    sqlite3* db_{nullptr};
    bool opened_{false};
};

class SqliteRunnerTableTest : public testing::Test {
protected:
    void SetUp() override { ASSERT_TRUE(runnerdb_test::ResetTestSession(kRunnerPath, kOpDataPath)); }

    void TearDown() override { runnerdb_test::CleanUpDatabases(kRunnerPath, kOpDataPath); }
};

TEST_F(SqliteRunnerTableTest, InsertAssignsIdAndReadsBack)
{
    auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
    ASSERT_TRUE(lease.ok());
    auto table = lease.value->Get().Table<sim::runtime::Server>();
    ASSERT_TRUE(table.ok());

    sim::runtime::Server server{};
    server.pod_id = 31;
    std::snprintf(server.version, sizeof(server.version), "v3.1");
    auto inserted = table->Insert(server);
    ASSERT_TRUE(inserted.ok());
    ASSERT_TRUE(inserted.value.has_value());
    EXPECT_EQ(inserted.value->value, 1U);
    EXPECT_EQ(server.id, 1U);

    auto found = table->SelectOne(HcclSim::Storage::Eq(&sim::runtime::Server::id, server.id));
    ASSERT_TRUE(found.ok());
    ASSERT_TRUE(found.value.has_value());
    EXPECT_EQ(found.value->pod_id, 31U);
    EXPECT_STREQ(found.value->version, "v3.1");
}

TEST_F(SqliteRunnerTableTest, SelectPushesConditionsAndKeepsLegacyOrder)
{
    auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
    ASSERT_TRUE(lease.ok());
    auto table = lease.value->Get().Table<sim::runtime::Device>();
    ASSERT_TRUE(table.ok());
    for (uint32_t index = 0; index < 4; ++index) {
        sim::runtime::Device device{};
        device.server_id = index % 2;
        device.logic_id = index;
        ASSERT_TRUE(table->Insert(device).ok());
    }

    auto selected = table->Select(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, uint64_t{1}),
        HcclSim::Storage::OrderBy(&sim::runtime::Device::id, HcclSim::Storage::SortDirection::ASC));
    ASSERT_TRUE(selected.ok());
    ASSERT_TRUE(selected.value.has_value());
    ASSERT_EQ(selected.value->size(), 2U);
    EXPECT_LT((*selected.value)[0].id, (*selected.value)[1].id);
    EXPECT_EQ((*selected.value)[0].server_id, 1U);
}

TEST_F(SqliteRunnerTableTest, UpdateDeclaredFieldsAndDeleteTargetOnly)
{
    auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
    ASSERT_TRUE(lease.ok());
    auto table = lease.value->Get().Table<sim::runtime::Stream>();
    ASSERT_TRUE(table.ok());
    sim::runtime::Stream first{};
    first.ctx_id = 5;
    first.priority = 1;
    sim::runtime::Stream second{};
    second.ctx_id = 6;
    second.priority = 2;
    ASSERT_TRUE(table->Insert(first).ok());
    ASSERT_TRUE(table->Insert(second).ok());

    auto updated = table->Update(
        HcclSim::Storage::Eq(&sim::runtime::Stream::id, first.id),
        HcclSim::Storage::Set(&sim::runtime::Stream::priority, uint64_t{8}));
    ASSERT_TRUE(updated.ok());
    ASSERT_TRUE(updated.value.has_value());
    EXPECT_EQ(*updated.value, 1U);

    auto untouched = table->SelectOne(HcclSim::Storage::Eq(&sim::runtime::Stream::id, second.id));
    ASSERT_TRUE(untouched.ok());
    ASSERT_TRUE(untouched.value.has_value());
    EXPECT_EQ(untouched.value->priority, 2U);

    auto deleted = table->Delete(HcclSim::Storage::Eq(&sim::runtime::Stream::id, first.id));
    ASSERT_TRUE(deleted.ok());
    ASSERT_TRUE(deleted.value.has_value());
    EXPECT_EQ(*deleted.value, 1U);
    const auto afterDelete = table->SelectOne(HcclSim::Storage::Eq(&sim::runtime::Stream::id, first.id));
    // 删除后的行未命中：SelectOne 返回 NOT_FOUND。
    ASSERT_FALSE(afterDelete.ok());
    EXPECT_EQ(afterDelete.code, HcclSim::Storage::DbCode::NOT_FOUND);
}

TEST_F(SqliteRunnerTableTest, ClearResetsAutoIncrementSequence)
{
    auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
    ASSERT_TRUE(lease.ok());
    auto table = lease.value->Get().Table<sim::runtime::Runner>();
    ASSERT_TRUE(table.ok());
    sim::runtime::Runner first{};
    ASSERT_TRUE(table->Insert(first).ok());
    ASSERT_TRUE(table->Clear(HcclSim::Storage::ClearMode::RESET_SEQUENCE).ok());
    sim::runtime::Runner second{};
    auto inserted = table->Insert(second);
    ASSERT_TRUE(inserted.ok());
    ASSERT_TRUE(inserted.value.has_value());
    EXPECT_EQ(inserted.value->value, 1U);
}

TEST_F(SqliteRunnerTableTest, FieldUpdatesPreserveIndependentConnectionWrites)
{
    auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
    ASSERT_TRUE(lease.ok());
    auto table = lease.value->Get().Table<sim::runtime::Stream>();
    ASSERT_TRUE(table.ok());
    sim::runtime::Stream stream{};
    stream.ctx_id = 5;
    stream.priority = 1;
    ASSERT_TRUE(table->Insert(stream).ok());

    IndependentConnection other(kRunnerPath);
    ASSERT_TRUE(other.Ok());
    ASSERT_TRUE(other.SetColumnInt64(stream.id, "activated", 77));
    ASSERT_TRUE(table
                    ->Update(
                        HcclSim::Storage::Eq(&sim::runtime::Stream::id, stream.id),
                        HcclSim::Storage::Set(&sim::runtime::Stream::priority, uint64_t{42}))
                    .ok());

    auto finalRow = table->SelectOne(HcclSim::Storage::Eq(&sim::runtime::Stream::id, stream.id));
    ASSERT_TRUE(finalRow.ok());
    ASSERT_TRUE(finalRow.value.has_value());
    EXPECT_EQ(finalRow.value->priority, 42U);
    EXPECT_EQ(finalRow.value->activated, 77U);
    EXPECT_EQ(finalRow.value->ctx_id, 5U);
}

TEST_F(SqliteRunnerTableTest, UpdateReportsZeroWhenAnotherConnectionDeletesRow)
{
    auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
    ASSERT_TRUE(lease.ok());
    auto table = lease.value->Get().Table<sim::runtime::Stream>();
    ASSERT_TRUE(table.ok());
    sim::runtime::Stream stream{};
    ASSERT_TRUE(table->Insert(stream).ok());
    IndependentConnection other(kRunnerPath);
    ASSERT_TRUE(other.Ok());
    ASSERT_EQ(other.Exec("DELETE FROM Stream WHERE id = 1"), SQLITE_OK);

    auto updated = table->Update(
        HcclSim::Storage::Eq(&sim::runtime::Stream::id, stream.id),
        HcclSim::Storage::Set(&sim::runtime::Stream::priority, uint64_t{7}));
    ASSERT_TRUE(updated.ok());
    ASSERT_TRUE(updated.value.has_value());
    EXPECT_EQ(*updated.value, 0U);
}

TEST_F(SqliteRunnerTableTest, BackendErrorIsReturnedWhenPhysicalTableIsMissing)
{
    auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
    ASSERT_TRUE(lease.ok());
    auto table = lease.value->Get().Table<sim::runtime::Stream>();
    ASSERT_TRUE(table.ok());
    sim::runtime::Stream stream{};
    ASSERT_TRUE(table->Insert(stream).ok());
    IndependentConnection other(kRunnerPath);
    ASSERT_TRUE(other.Ok());
    ASSERT_EQ(other.Exec("DROP TABLE Stream"), SQLITE_OK);

    EXPECT_FALSE(table
                     ->Update(
                         HcclSim::Storage::Eq(&sim::runtime::Stream::id, stream.id),
                         HcclSim::Storage::Set(&sim::runtime::Stream::priority, uint64_t{7}))
                     .ok());
}

TEST_F(SqliteRunnerTableTest, ClosedProcessSessionCannotBeAcquired)
{
    ASSERT_TRUE(HcclSim::Storage::StorageRuntime::CloseProcessSession().ok());
    auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
    EXPECT_FALSE(lease.ok());
    HcclSim::Storage::StorageRuntimeConfig sessionConfig;
    sessionConfig.databases.push_back(
        {std::string{"RUNNER_DB"}, sim::backends::kSqliteBackendName, std::string{kRunnerPath}});
    sessionConfig.databases.push_back(
        {std::string{"OP_DATA_DB"}, sim::backends::kSqliteBackendName, std::string{kOpDataPath}});
    ASSERT_TRUE(HcclSim::Storage::StorageRuntime::EnsureProcessSession(sessionConfig).ok());
}

TEST_F(SqliteRunnerTableTest, ExplicitIdIsPreservedByTableContract)
{
    auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
    ASSERT_TRUE(lease.ok());
    auto table = lease.value->Get().Table<sim::runtime::Stream>();
    ASSERT_TRUE(table.ok());
    sim::runtime::Stream record{};
    record.id = 7;
    record.ctx_id = 11;
    record.priority = 2;
    auto inserted = table->Insert(record);
    ASSERT_TRUE(inserted.ok());
    ASSERT_TRUE(inserted.value.has_value());
    EXPECT_EQ(inserted.value->value, 7U);
    EXPECT_EQ(record.id, inserted.value->value);
}

TEST_F(SqliteRunnerTableTest, UpdateChangesOnlyDeclaredPhysicalColumn)
{
    auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
    ASSERT_TRUE(lease.ok());
    auto table = lease.value->Get().Table<sim::runtime::Stream>();
    ASSERT_TRUE(table.ok());
    sim::runtime::Stream record{};
    record.ctx_id = 5;
    record.priority = 1;
    ASSERT_TRUE(table->Insert(record).ok());
    ASSERT_TRUE(table
                    ->Update(
                        HcclSim::Storage::Eq(&sim::runtime::Stream::id, record.id),
                        HcclSim::Storage::Set(&sim::runtime::Stream::priority, uint64_t{8}))
                    .ok());

    IndependentConnection other(kRunnerPath);
    ASSERT_TRUE(other.Ok());
    uint64_t priority = 0;
    uint64_t ctxId = 0;
    uint64_t activated = 0;
    ASSERT_TRUE(other.ReadColumnInt64(record.id, "priority", priority));
    ASSERT_TRUE(other.ReadColumnInt64(record.id, "ctx_id", ctxId));
    ASSERT_TRUE(other.ReadColumnInt64(record.id, "activated", activated));
    EXPECT_EQ(priority, 8U);
    EXPECT_EQ(ctxId, 5U);
    EXPECT_EQ(activated, 0U);
}

TEST_F(SqliteRunnerTableTest, ReadsAndUpdatesRowsInsertedByStructuredSql)
{
    auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
    ASSERT_TRUE(lease.ok());
    auto table = lease.value->Get().Table<sim::runtime::Stream>();
    ASSERT_TRUE(table.ok());
    IndependentConnection other(kRunnerPath);
    ASSERT_TRUE(other.Ok());
    const uint64_t id = other.InsertStreamRow(21, 100, 0);
    ASSERT_NE(id, 0U);

    auto selected = table->SelectOne(HcclSim::Storage::Eq(&sim::runtime::Stream::id, id));
    ASSERT_TRUE(selected.ok());
    ASSERT_TRUE(selected.value.has_value());
    EXPECT_EQ(selected.value->ctx_id, 21U);
    EXPECT_EQ(selected.value->priority, 100U);

    ASSERT_TRUE(table
                    ->Update(
                        HcclSim::Storage::Eq(&sim::runtime::Stream::id, id),
                        HcclSim::Storage::Set(&sim::runtime::Stream::priority, uint64_t{31}))
                    .ok());
    uint64_t priority = 0;
    ASSERT_TRUE(other.ReadColumnInt64(id, "priority", priority));
    EXPECT_EQ(priority, 31U);
}
TEST_F(SqliteRunnerTableTest, HcommResourceTables_PreserveEndpointAndMemoryRecords)
{
    sim::runtime::HcommEndpoint endpoint{};
    endpoint.runner_id = 11;
    endpoint.south_endpoint_id = 22;
    endpoint.protocol = 1;
    endpoint.addr_type = 3;
    endpoint.addr[0] = 0xAB;
    endpoint.loc_type = 0;
    endpoint.loc[0] = 7;
    endpoint.extension[0] = 9;
    endpoint.bind_state = sim::runtime::HCOMM_ENDPOINT_BOUND;

    const uint64_t endpointId = runnerdb_test::InsertRecord<sim::runtime::HcommEndpoint>(endpoint);
    ASSERT_GT(endpointId, 0u);

    auto foundEndpoint = runnerdb_test::SelectRecord<sim::runtime::HcommEndpoint>(endpointId);
    ASSERT_TRUE(foundEndpoint.has_value());
    EXPECT_EQ(foundEndpoint->runner_id, 11U);
    EXPECT_EQ(foundEndpoint->south_endpoint_id, 22U);
    EXPECT_EQ(foundEndpoint->addr[0], 0xAB);
    EXPECT_EQ(foundEndpoint->bind_state, sim::runtime::HCOMM_ENDPOINT_BOUND);

    sim::runtime::HcommMemReg memory{};
    memory.endpoint_id = endpointId;
    memory.runner_id = 11;
    memory.descriptor_key = 33;
    memory.addr = 0x1000;
    memory.size = 4096;
    memory.mem_type = 0;
    memory.record_kind = sim::runtime::HCOMM_MEM_LOCAL_REGISTERED;
    std::snprintf(memory.mem_tag, sizeof(memory.mem_tag), "input");

    const uint64_t memoryId = runnerdb_test::InsertRecord<sim::runtime::HcommMemReg>(memory);
    ASSERT_GT(memoryId, 0u);

    auto foundMemory = runnerdb_test::SelectRecord<sim::runtime::HcommMemReg>(memoryId);
    ASSERT_TRUE(foundMemory.has_value());
    EXPECT_EQ(foundMemory->endpoint_id, endpointId);
    EXPECT_EQ(foundMemory->descriptor_key, 33U);
    EXPECT_EQ(foundMemory->addr, 0x1000U);
    EXPECT_EQ(foundMemory->size, 4096U);
    EXPECT_EQ(foundMemory->record_kind, sim::runtime::HCOMM_MEM_LOCAL_REGISTERED);
    EXPECT_STREQ(foundMemory->mem_tag, "input");
}
} // namespace
