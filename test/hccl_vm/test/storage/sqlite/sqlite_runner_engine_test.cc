/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：验证 RunnerDB 结构化记录在存储抽象层上的端到端语义：
// 自增主键回填、逻辑字段往返（含 uint64 高位与 C 字符串列）、天然 BLOB 载荷逐字节一致、
// 结构化条件查询下推（等值/范围/IN/前缀/AND/OR/排序/limit）、显式字段更新、
// 声明字段条件更新防丢失（陈旧条件影响 0 行）、删除与序列重置。
#include <gtest/gtest.h>

#include <sqlite3.h>

#include <cstring>
#include <string>
#include <vector>

#include "storage_composition/storage_composition.h"
#include "storage/table_access.h"
#include "storage/storage_session.h"
#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage_backends/sqlite/composition/sqlite_backend_module.h"

namespace {

const char* kRunnerPath = "/tmp/test_sqlite_runner_engine_runner.db";
const char* kOpDataPath = "/tmp/test_sqlite_runner_engine_opdata.db";

class SqliteRunnerEngineTest : public testing::Test {
protected:
    void SetUp() override
    {
        CleanFiles();
        ASSERT_TRUE(sim::RegisterSimulatorStorageBackends().ok());
        HcclSim::Storage::StorageRuntimeConfig sessionConfig;
        sessionConfig.databases.push_back(
            {std::string{"RUNNER_DB"}, sim::backends::kSqliteBackendName, std::string{kRunnerPath}});
        sessionConfig.databases.push_back(
            {std::string{"OP_DATA_DB"}, sim::backends::kSqliteBackendName, std::string{kOpDataPath}});
        auto ensured = HcclSim::Storage::StorageRuntime::EnsureProcessSession(sessionConfig);
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
};

TEST_F(SqliteRunnerEngineTest, InsertAssignsAutoIncrementIdAndRoundTripsStructuredFields)
{
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::Server>();
    ASSERT_TRUE(table.ok()) << table.diagnostic;

    sim::runtime::Server record{};
    record.id = 0;
    record.pod_id = 42;
    record.server_id = 7;
    snprintf(record.version, sizeof(record.version), "v9.9");
    auto inserted = table->Insert(record);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    EXPECT_EQ(inserted.value->value, 1U);
    EXPECT_EQ(record.id, 1U);

    auto found = table->SelectOne(HcclSim::Storage::Eq(&sim::runtime::Server::id, inserted->value));
    ASSERT_TRUE(found.ok()) << found.diagnostic;
    EXPECT_EQ(found.value->pod_id, 42U);
    EXPECT_EQ(found.value->server_id, 7U);
    EXPECT_STREQ(found.value->version, "v9.9");
}

TEST_F(SqliteRunnerEngineTest, TypedInsertKeepsExplicitAutoIncrementPrimaryKey)
{
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::Server>();
    ASSERT_TRUE(table.ok()) << table.diagnostic;

    sim::runtime::Server record{};
    record.id = 73;
    record.pod_id = 4;
    record.server_id = 8;
    auto inserted = table->Insert(record);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    EXPECT_EQ(inserted.value->value, 73U);
    EXPECT_EQ(record.id, 73U);
}

TEST_F(SqliteRunnerEngineTest, InsertAssignsSequentialIds)
{
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::Host>();
    ASSERT_TRUE(table.ok());
    for (uint64_t expected = 1; expected <= 3; ++expected) {
        sim::runtime::Host record{};
        record.server_id = expected;
        auto inserted = table->Insert(record);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
        EXPECT_EQ(inserted.value->value, expected);
    }
}

TEST_F(SqliteRunnerEngineTest, UnsignedHighBitsRoundTripThroughIntegerColumns)
{
    // 完整 uint64 高位值（含最高位为 1）经结构化 INTEGER 列逐位往返。
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::VirtualMemBlock>();
    ASSERT_TRUE(table.ok()) << table.diagnostic;
    const uint64_t high = std::numeric_limits<uint64_t>::max() - 12U;
    sim::runtime::VirtualMemBlock record{};
    record.start_ptr = high;
    record.dev_mapped_ptr = std::numeric_limits<uint64_t>::max();
    record.size = 0x1000;
    auto inserted = table->Insert(record);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    auto found = table->SelectOne(HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::id, inserted->value));
    ASSERT_TRUE(found.ok()) << found.diagnostic;
    EXPECT_EQ(found.value->start_ptr, high);
    EXPECT_EQ(found.value->dev_mapped_ptr, std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(found.value->size, 0x1000U);
}

TEST_F(SqliteRunnerEngineTest, SelectByPrimaryKeyConditionMissingIdReturnsEmptySet)
{
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::Server>();
    ASSERT_TRUE(table.ok());
    auto found = table->SelectOne(HcclSim::Storage::Eq(&sim::runtime::Server::id, 999));
    ASSERT_FALSE(found.ok());
    EXPECT_EQ(found.code, HcclSim::Storage::DbCode::NOT_FOUND);
}

TEST_F(SqliteRunnerEngineTest, EqualityPredicatesArePushedDownToTheDatabase)
{
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::Device>();
    ASSERT_TRUE(table.ok());
    for (uint32_t index = 0; index < 5; ++index) {
        sim::runtime::Device record{};
        record.server_id = index % 2;
        record.logic_id = index;
        record.physical_id = 100 + index;
        auto inserted = table->Insert(record);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    }
    // 结构化等值条件下推:数据库只返回 server_id == 1 的行,业务侧不再全表过滤。
    auto rows = table->Select(HcclSim::Storage::Eq(&sim::runtime::Device::server_id, uint64_t{1}));
    ASSERT_TRUE(rows.ok()) << rows.diagnostic;
    ASSERT_EQ(rows.value->size(), 2U);
    for (const auto& row : *rows.value) {
        EXPECT_EQ(row.server_id, 1U);
    }
}

TEST_F(SqliteRunnerEngineTest, RangeInPrefixAndOrPredicatesArePushedDown)
{
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::VirtualMemBlock>();
    ASSERT_TRUE(table.ok()) << table.diagnostic;
    // 三块虚拟内存:主机区间 [0x1000,0x2000) / [0x3000,0x4000) / 设备映射 [0x5000,0x6000)。
    const std::pair<uint64_t, uint64_t> ranges[] = {{0x1000, 0x1000}, {0x3000, 0x1000}, {0x5000, 0x1000}};
    for (size_t index = 0; index < 3; ++index) {
        sim::runtime::VirtualMemBlock record{};
        record.start_ptr = ranges[index].first;
        record.size = ranges[index].second;
        record.dev_mapped_ptr = ranges[index].first;
        record.src_type
            = static_cast<uint8_t>(index == 2 ? sim::runtime::VIR_MEM_TYPE_DEV : sim::runtime::VIR_MEM_TYPE_HOST);
        auto inserted = table->Insert(record);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    }

    // 派生端点列承载包含查询:0x1500 落在第一块内,不落在他处。
    auto contained = table->Select(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::start_ptr, uint64_t{0x1500}),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::start_end_ptr, uint64_t{0x1500})));
    ASSERT_TRUE(contained.ok()) << contained.diagnostic;
    ASSERT_EQ(contained.value->size(), 1U);
    EXPECT_EQ(contained.value->front().start_ptr, 0x1000U);

    // 范围条件:start_ptr >= 0x3000 排除第一块。
    auto above = table->Select(HcclSim::Storage::Ge(&sim::runtime::VirtualMemBlock::start_ptr, uint64_t{0x3000}));
    ASSERT_TRUE(above.ok()) << above.diagnostic;
    ASSERT_EQ(above.value->size(), 2U);

    // IN 条件与 AND 组合。
    auto devBlocks = table->Select(HcclSim::Storage::And(
        HcclSim::Storage::In(
            &sim::runtime::VirtualMemBlock::start_ptr, std::vector<uint64_t>{uint64_t{0x1000}, uint64_t{0x5000}}),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, uint64_t{sim::runtime::VIR_MEM_TYPE_DEV})));
    ASSERT_TRUE(devBlocks.ok()) << devBlocks.diagnostic;
    ASSERT_EQ(devBlocks.value->size(), 1U);
    EXPECT_EQ(devBlocks.value->front().start_ptr, 0x5000U);

    // OR 连接:主机类型或起始地址为 0x5000。
    auto either = table->Select(HcclSim::Storage::Or(
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, uint64_t{sim::runtime::VIR_MEM_TYPE_HOST}),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::start_ptr, uint64_t{0x5000})));
    ASSERT_TRUE(either.ok()) << either.diagnostic;
    ASSERT_EQ(either.value->size(), 3U);
}

TEST_F(SqliteRunnerEngineTest, TextPrefixPredicateIsPushedDown)
{
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::IpcNotify>();
    ASSERT_TRUE(table.ok()) << table.diagnostic;
    const char* names[] = {"alpha-notify", "alpha-other", "beta-notify"};
    for (const char* name : names) {
        sim::runtime::IpcNotify record{};
        record.notify_id = 1;
        std::snprintf(reinterpret_cast<char*>(record.name_or_key), sizeof(record.name_or_key), "%s", name);
        auto inserted = table->Insert(record);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    }
    // 前缀查询(迁移前 memcmp+strlen 语义):TEXT 列 PREFIX 下推。
    auto matched
        = table->Select(HcclSim::Storage::StartsWith(&sim::runtime::IpcNotify::name_or_key, std::string{"alpha"}));
    ASSERT_TRUE(matched.ok()) << matched.diagnostic;
    ASSERT_EQ(matched.value->size(), 2U);
}

TEST_F(SqliteRunnerEngineTest, OrderAndLimitArePushedDown)
{
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::Device>();
    ASSERT_TRUE(table.ok());
    for (uint32_t index = 1; index <= 4; ++index) {
        sim::runtime::Device record{};
        record.server_id = 10;
        record.logic_id = 10 - index; // 逻辑号倒序插入:9,8,7,6。
        auto inserted = table->Insert(record);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    }
    auto rows = table->Select(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, uint64_t{10}),
        HcclSim::Storage::OrderBy(&sim::runtime::Device::logic_id, HcclSim::Storage::SortDirection::ASC),
        HcclSim::Storage::Limit(3U));
    ASSERT_TRUE(rows.ok()) << rows.diagnostic;
    ASSERT_EQ(rows.value->size(), 3U);
    EXPECT_EQ(rows.value->at(0).logic_id, 6U);
    EXPECT_EQ(rows.value->at(1).logic_id, 7U);
    EXPECT_EQ(rows.value->at(2).logic_id, 8U);
}

TEST_F(SqliteRunnerEngineTest, UpdateByPrimaryKeySetsDeclaredFieldsOnly)
{
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::Stream>();
    ASSERT_TRUE(table.ok());
    sim::runtime::Stream record{};
    record.ctx_id = 11;
    record.priority = 1;
    record.activated = 0;
    auto inserted = table->Insert(record);
    ASSERT_TRUE(inserted.ok());
    const auto key = inserted->value;

    // 显式字段更新:只改 priority,其余列保持。
    auto result = table->Update(
        HcclSim::Storage::Eq(&sim::runtime::Stream::id, key),
        HcclSim::Storage::Set(&sim::runtime::Stream::priority, uint64_t{9}));
    ASSERT_TRUE(result.ok()) << result.diagnostic;
    EXPECT_EQ(*result.value, 1U);

    auto reloaded = table->SelectOne(HcclSim::Storage::Eq(&sim::runtime::Stream::id, key));
    ASSERT_TRUE(reloaded.ok());
    EXPECT_EQ(reloaded.value->priority, 9U);
    EXPECT_EQ(reloaded.value->ctx_id, 11U);
    EXPECT_EQ(reloaded.value->activated, 0U);
}

TEST_F(SqliteRunnerEngineTest, IncrementIsAtomicInDatabase)
{
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::RaSocketPair>();
    ASSERT_TRUE(table.ok());
    sim::runtime::RaSocketPair record{};
    record.ref_cnt = 5;
    auto inserted = table->Insert(record);
    ASSERT_TRUE(inserted.ok());
    const auto key = inserted->value;

    // 数据库端原子自增:不读不改不写。
    auto result = table->Update(
        HcclSim::Storage::Eq(&sim::runtime::RaSocketPair::id, key),
        HcclSim::Storage::Increment(&sim::runtime::RaSocketPair::ref_cnt, uint64_t{1}));
    ASSERT_TRUE(result.ok()) << result.diagnostic;
    EXPECT_EQ(*result.value, 1U);
    auto reloaded = table->SelectOne(HcclSim::Storage::Eq(&sim::runtime::RaSocketPair::id, key));
    ASSERT_TRUE(reloaded.ok());
    EXPECT_EQ(reloaded.value->ref_cnt, 6U);
}

TEST_F(SqliteRunnerEngineTest, DeleteByPrimaryKeyRemovesOnlyThatRow)
{
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::Event>();
    ASSERT_TRUE(table.ok());
    sim::runtime::Event first{};
    auto firstKey = table->Insert(first);
    ASSERT_TRUE(firstKey.ok());
    sim::runtime::Event second{};
    auto secondKey = table->Insert(second);
    ASSERT_TRUE(secondKey.ok());

    auto deleted = table->Delete(HcclSim::Storage::Eq(&sim::runtime::Event::id, firstKey->value));
    ASSERT_TRUE(deleted.ok()) << deleted.diagnostic;
    EXPECT_EQ(*deleted.value, 1U);

    const auto removedRow = table->SelectOne(HcclSim::Storage::Eq(&sim::runtime::Event::id, firstKey->value));
    // 删除后的行未命中：SelectOne 返回 NOT_FOUND。
    ASSERT_FALSE(removedRow.ok());
    EXPECT_EQ(removedRow.code, HcclSim::Storage::DbCode::NOT_FOUND);
    const auto keptRow = table->SelectOne(HcclSim::Storage::Eq(&sim::runtime::Event::id, secondKey->value));
    ASSERT_TRUE(keptRow.ok());
    ASSERT_TRUE(keptRow.value.has_value());
    // 未命中删除影响 0 行。
    auto missing = table->Delete(HcclSim::Storage::Eq(&sim::runtime::Event::id, 777));
    ASSERT_TRUE(missing.ok());
    EXPECT_EQ(*missing.value, 0U);
}

TEST_F(SqliteRunnerEngineTest, ClearWithResetSequenceRestartsAutoIncrement)
{
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::RunModeConfig>();
    ASSERT_TRUE(table.ok());
    sim::runtime::RunModeConfig first{};
    first.mode = 1;
    auto firstKey = table->Insert(first);
    ASSERT_TRUE(firstKey.ok());
    EXPECT_EQ(firstKey.value->value, 1U);

    auto cleared = table->Clear(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    ASSERT_TRUE(cleared.ok()) << cleared.diagnostic;

    sim::runtime::RunModeConfig second{};
    second.mode = 2;
    auto secondKey = table->Insert(second);
    ASSERT_TRUE(secondKey.ok()) << secondKey.diagnostic;
    // 序列已重置:新记录从 1 重新分配。
    EXPECT_EQ(secondKey.value->value, 1U);
}

TEST_F(SqliteRunnerEngineTest, ClearWithKeepSequenceContinuesAutoIncrement)
{
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::Port>();
    ASSERT_TRUE(table.ok());
    sim::runtime::Port first{};
    auto firstKey = table->Insert(first);
    ASSERT_TRUE(firstKey.ok());
    auto cleared = table->Clear(HcclSim::Storage::ClearMode::KEEP_SEQUENCE);
    ASSERT_TRUE(cleared.ok());
    sim::runtime::Port second{};
    auto secondKey = table->Insert(second);
    ASSERT_TRUE(secondKey.ok());
    EXPECT_EQ(secondKey.value->value, 2U);
}

TEST_F(SqliteRunnerEngineTest, NaturalBlobPayloadRoundTripsByteByByte)
{
    // 天然字节载荷(RaDevice.mac_addr)按独立 BLOB 列逐字节保存和读取（原
    // CcuResource.instr_space 载体随指令空间列删除，改写为同形态存活表）。
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::RaDevice>();
    ASSERT_TRUE(table.ok());
    sim::runtime::RaDevice record{};
    record.device_id = 5;
    record.state = 3;
    record.endpoint_id = 0;
    for (size_t index = 0; index < sizeof(record.mac_addr); ++index) {
        record.mac_addr[index] = static_cast<uint8_t>(index & 0xFF);
    }
    auto inserted = table->Insert(record);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;

    auto found = table->SelectOne(HcclSim::Storage::Eq(&sim::runtime::RaDevice::id, inserted->value));
    ASSERT_TRUE(found.ok()) << found.diagnostic;
    EXPECT_EQ(found.value->device_id, 5U);
    EXPECT_EQ(found.value->state, 3U);
    EXPECT_EQ(std::memcmp(found.value->mac_addr, record.mac_addr, sizeof(record.mac_addr)), 0);
}

TEST_F(SqliteRunnerEngineTest, ConditionalUpdateRejectsStaleFieldValueThenAcceptsFreshOne)
{
    // 声明字段等值条件更新是显式 Update 防丢失的引擎层基础:
    // 读出旧值后被独立连接抢先改写,以旧值为条件的更新影响 0 行;
    // 重读最新值后条件更新恰影响 1 行。
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::Stream>();
    ASSERT_TRUE(table.ok());
    sim::runtime::Stream record{};
    record.ctx_id = 21;
    record.priority = 1;
    auto inserted = table->Insert(record);
    ASSERT_TRUE(inserted.ok());
    const auto key = inserted->value;

    // 独立连接抢先改写 priority 列(模拟另一进程会话的显式字段更新)。
    sqlite3* other = nullptr;
    ASSERT_EQ(sqlite3_open_v2(kRunnerPath, &other, SQLITE_OPEN_READWRITE | SQLITE_OPEN_URI, nullptr), SQLITE_OK);
    {
        sqlite3_stmt* update = nullptr;
        ASSERT_EQ(
            sqlite3_prepare_v2(other, "UPDATE Stream SET priority = 55 WHERE id = ?", -1, &update, nullptr), SQLITE_OK);
        sqlite3_bind_int64(update, 1, static_cast<sqlite3_int64>(key));
        ASSERT_EQ(sqlite3_step(update), SQLITE_DONE);
        ASSERT_EQ(sqlite3_changes(other), 1);
        sqlite3_finalize(update);
    }
    sqlite3_close(other);

    // 陈旧条件(priority 仍为读出时的 1)做条件更新:影响 0 行,调用方必须重读重试。
    auto staleResult = table->Update(
        HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Stream::id, key),
            HcclSim::Storage::Eq(&sim::runtime::Stream::priority, uint64_t{1})),
        HcclSim::Storage::Set(&sim::runtime::Stream::priority, uint64_t{30}));
    ASSERT_TRUE(staleResult.ok()) << staleResult.diagnostic;
    EXPECT_EQ(*staleResult.value, 0U);

    // 重读最新值:以 priority == 55 为条件回写,恰影响 1 行。
    auto reloaded = table->SelectOne(HcclSim::Storage::Eq(&sim::runtime::Stream::id, key));
    ASSERT_TRUE(reloaded.ok());
    EXPECT_EQ(reloaded.value->priority, 55U);
    auto freshResult = table->Update(
        HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Stream::id, key),
            HcclSim::Storage::Eq(&sim::runtime::Stream::priority, uint64_t{55})),
        HcclSim::Storage::Set(&sim::runtime::Stream::priority, uint64_t{30}));
    ASSERT_TRUE(freshResult.ok()) << freshResult.diagnostic;
    EXPECT_EQ(*freshResult.value, 1U);

    // 最终值保留独立连接与本方的修改链。
    auto finalRow = table->SelectOne(HcclSim::Storage::Eq(&sim::runtime::Stream::id, key));
    ASSERT_TRUE(finalRow.ok());
    EXPECT_EQ(finalRow.value->priority, 30U);
    EXPECT_EQ(finalRow.value->ctx_id, 21U);
}

TEST_F(SqliteRunnerEngineTest, TransactionBoundToOpDataDomainRejectsRunnerTableStage)
{
    // 迁移前语义:Runner 表无事务(逐语句自动提交)。通用事务首个暂存 DML 绑定 OP_DATA_DB
    // 物理域后,Runner 表(RUNNER_DB)的暂存属于跨物理域,执行前被拒绝,不会产生半提交状态。
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::Task>();
    ASSERT_TRUE(table.ok()) << table.diagnostic;
    auto opDetailTable = HcclSim::Storage::AcquireTable<sim::operation::OpDetailTab>();
    ASSERT_TRUE(opDetailTable.ok()) << opDetailTable.diagnostic;

    sim::operation::OpDetailTab detail{};
    detail.pid = 1;
    detail.rankId = 1;
    sim::runtime::Task staged{};
    staged.stream_id = 1;
    staged.seq_number = 10;
    {
        auto transaction = HcclSim::Storage::BeginTransaction();
        ASSERT_TRUE(transaction.ok()) << transaction.diagnostic;
        // 首个暂存 DML 把事务惰性绑定到 OP_DATA_DB 物理域。
        const auto stagedOpDetail = opDetailTable->Insert(detail, *transaction.value);
        ASSERT_TRUE(stagedOpDetail.ok()) << stagedOpDetail.diagnostic;
        const auto stagedIntoForeignDomain = table->Insert(staged, *transaction.value);
        ASSERT_FALSE(stagedIntoForeignDomain.ok());
        EXPECT_EQ(stagedIntoForeignDomain.code, HcclSim::Storage::DbCode::INVALID_ARGUMENT);
        // 拒绝发生在路由校验:显式回滚,不产生半提交状态。
        ASSERT_TRUE(transaction.value->Rollback().ok());
    }
    auto rows = table->Select(HcclSim::Storage::All<sim::runtime::Task>());
    ASSERT_TRUE(rows.ok()) << rows.diagnostic;
    EXPECT_TRUE(rows.value->empty());
    auto details = opDetailTable->Select(HcclSim::Storage::All<sim::operation::OpDetailTab>());
    ASSERT_TRUE(details.ok()) << details.diagnostic;
    EXPECT_TRUE(details.value->empty());
}
} // namespace
