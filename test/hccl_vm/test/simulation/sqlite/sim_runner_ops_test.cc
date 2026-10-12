/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <type_traits>

#include <gtest/gtest.h>
#include <unistd.h>

#include "storage/internal/process_storage_context.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "storage/storage_session.h"
#include "sqlite/simulation_storage_test_helper.h"

namespace {
const std::string kTestDbPath = "/tmp/test_sim_runner_ops.db";
const std::string kTestOpDataDbPath = "/tmp/test_sim_runner_ops.db.opdata";

void CleanUpDb() { runnerdb_test::CleanUpDatabases(kTestDbPath, kTestOpDataDbPath); }

void SetupTestData()
{
    CleanUpDb();
    (void)runnerdb_test::ResetTestSession(kTestDbPath, kTestOpDataDbPath);

    sim::runtime::Server server{};
    server.pod_id = 100;
    snprintf(server.version, sizeof(server.version), "v1.0");
    runnerdb_test::InsertRecord<sim::runtime::Server>(server);

    sim::runtime::Host host{};
    host.server_id = 1;
    snprintf(host.ip_addr, sizeof(host.ip_addr), "192.168.1.100");
    host.arch = 1;
    runnerdb_test::InsertRecord<sim::runtime::Host>(host);

    sim::runtime::Device device{};
    device.server_id = 1;
    device.logic_id = 0;
    device.physical_id = 0;
    runnerdb_test::InsertRecord<sim::runtime::Device>(device);

    sim::runtime::Context ctx{};
    ctx.device_id = 1;
    ctx.is_default = 1;
    ctx.run_id = 1;
    runnerdb_test::InsertRecord<sim::runtime::Context>(ctx);

    sim::runtime::Communicator comm{};
    std::strncpy(comm.comm_id, "test_comm", sizeof(comm.comm_id) - 1);
    comm.rank_size = 1;
    comm.rank_id = 0;
    comm.device_id = 1;
    sim::runtime::g_cur_comm_key = runnerdb_test::InsertRecord<sim::runtime::Communicator>(comm);
}
} // namespace

class SimRunnerOpsTest : public testing::Test {
protected:
    void SetUp() override { SetupTestData(); }

    void TearDown() override { CleanUpDb(); }
};

// SetRunMode：清空运行模式单行表（重置序列）后写入新模式；重复设置覆盖前值。
TEST_F(SimRunnerOpsTest, SetRunModeRewritesSingleRowMode)
{
    ASSERT_TRUE(sim::runtime::SetRunMode(true));
    const auto checkOnly = runnerdb_test::SelectFirstRecord<sim::runtime::RunModeConfig>(
        HcclSim::Storage::All<sim::runtime::RunModeConfig>(),
        HcclSim::Storage::OrderBy(&sim::runtime::RunModeConfig::id, HcclSim::Storage::SortDirection::ASC));
    ASSERT_TRUE(checkOnly.second);
    EXPECT_EQ(checkOnly.first.mode, 1U);

    ASSERT_TRUE(sim::runtime::SetRunMode(false));
    const auto normal = runnerdb_test::SelectFirstRecord<sim::runtime::RunModeConfig>(
        HcclSim::Storage::All<sim::runtime::RunModeConfig>(),
        HcclSim::Storage::OrderBy(&sim::runtime::RunModeConfig::id, HcclSim::Storage::SortDirection::ASC));
    ASSERT_TRUE(normal.second);
    EXPECT_EQ(normal.first.mode, 0U);
    // 单行表语义：重复设置不累积行。
    const auto rows = runnerdb_test::SelectRecords<sim::runtime::RunModeConfig>(
        HcclSim::Storage::All<sim::runtime::RunModeConfig>());
    EXPECT_EQ(rows.size(), 1U);
}

TEST_F(SimRunnerOpsTest, SetAndGetLastStreamId)
{
    sim::runtime::SetLastStreamIdTls(12345);

    uint64_t streamId = sim::runtime::GetLastStreamIdTls();

    EXPECT_EQ(streamId, 12345);
}

TEST_F(SimRunnerOpsTest, SetAndGetLastTaskId)
{
    sim::runtime::SetLastTaskIdTls(67890);

    uint64_t taskId = sim::runtime::GetLastTaskIdTls();

    EXPECT_EQ(taskId, 67890);
}

TEST_F(SimRunnerOpsTest, SetTsDevice)
{
    sim::runtime::SetTsDevice(42);
    sim::runtime::SetTsDevice(0);
}

// 一行业务函数（内部一次数据库端 COUNT 聚合，不加载行集）。
TEST_F(SimRunnerOpsTest, GetRankSize_ReturnsCorrectCount)
{
    uint32_t size = sim::runtime::GetRankSize();

    EXPECT_GE(size, 1);
}

TEST_F(SimRunnerOpsTest, GetHostSize_ReturnsCorrectCount)
{
    uint32_t size = sim::runtime::GetHostSize();

    EXPECT_GE(size, 1);
}

TEST_F(SimRunnerOpsTest, SetCurrCtxTls_UpdatesContext)
{
    bool ret = sim::runtime::SetCurrCtxTls(1);

    EXPECT_TRUE(ret);
}

TEST_F(SimRunnerOpsTest, GetCurrDeviceKey_WithValidCtx_ReturnsDeviceKey)
{
    sim::runtime::SetCurrCtxTls(1);

    uint64_t deviceKey = sim::runtime::GetCurrDeviceKey();

    EXPECT_EQ(deviceKey, 1);
}

TEST_F(SimRunnerOpsTest, GetCurrDeviceKey_WithInvalidCtx_ReturnsZero)
{
    sim::runtime::SetCurrCtxTls(99999);

    uint64_t deviceKey = sim::runtime::GetCurrDeviceKey();

    EXPECT_EQ(deviceKey, 0);
}

TEST_F(SimRunnerOpsTest, GetDeviceIdByCtxId_WithValidCtx_ReturnsDeviceId)
{
    uint64_t deviceId = sim::runtime::GetDeviceIdByCtxId(1);

    EXPECT_EQ(deviceId, 1);
}

TEST_F(SimRunnerOpsTest, GetDeviceIdByCtxId_WithInvalidCtx_ReturnsZero)
{
    uint64_t deviceId = sim::runtime::GetDeviceIdByCtxId(99999);

    EXPECT_EQ(deviceId, 0);
}

TEST_F(SimRunnerOpsTest, GetCurrDeviceId_WithValidContext)
{
    sim::runtime::SetCurrCtxTls(1);

    uint64_t deviceId = sim::runtime::GetCurrDeviceId();

    EXPECT_EQ(deviceId, 1);
}

// ==================== GetCurrDeviceId 错误路径 ====================

class SimRunnerOpsGetCurrDeviceIdTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();
        sim::runtime::Runner runner;
        sim::runtime::GetCurrRunnerTls(1, runner);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerOpsGetCurrDeviceIdTest, GetCurrDeviceId_InvalidCtx_ReturnsZero)
{
    // 设置一个不存在的 context id
    sim::runtime::SetCurrCtxTls(99999);
    uint64_t deviceId = sim::runtime::GetCurrDeviceId();
    EXPECT_EQ(deviceId, 0);
}

TEST_F(SimRunnerOpsGetCurrDeviceIdTest, GetCurrDeviceId_MissingContextDevice_ReturnsZero)
{
    // 添加一个 context，其 device_id 没有对应的设备。
    sim::runtime::Context ctx2{};
    ctx2.device_id = 999;
    ctx2.is_default = 1;
    ctx2.run_id = 1;
    const uint64_t ctxId = runnerdb_test::InsertRecord<sim::runtime::Context>(ctx2);

    sim::runtime::SetCurrCtxTls(ctxId);
    uint64_t deviceId = sim::runtime::GetCurrDeviceId();
    EXPECT_EQ(deviceId, 0);
}

// ==================== GetDeviceIdByCtxId 错误路径 ====================

class SimRunnerOpsGetDeviceIdByCtxIdTest : public testing::Test {
protected:
    void SetUp() override { SetupTestData(); }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerOpsGetDeviceIdByCtxIdTest, GetDeviceIdByCtxId_InvalidCtx_ReturnsZero)
{
    uint64_t deviceId = sim::runtime::GetDeviceIdByCtxId(99999);
    EXPECT_EQ(deviceId, 0);
}

TEST_F(SimRunnerOpsGetDeviceIdByCtxIdTest, GetDeviceIdByCtxId_NoDevice_ReturnsZero)
{
    // 添加一个 context，其 device_id 没有对应的 device
    sim::runtime::Context ctx2{};
    ctx2.device_id = 999; // device_id=999 不存在
    ctx2.is_default = 1;
    ctx2.run_id = 1;
    const uint64_t ctxId = runnerdb_test::InsertRecord<sim::runtime::Context>(ctx2);

    uint64_t deviceId = sim::runtime::GetDeviceIdByCtxId(ctxId);
    EXPECT_EQ(deviceId, 0);
}

// ==================== 新增：GetCurrDeviceKey 错误路径 ====================

TEST_F(SimRunnerOpsGetCurrDeviceIdTest, GetCurrDeviceKey_InvalidCtx_ReturnsZero)
{
    sim::runtime::SetCurrCtxTls(99999);
    uint64_t deviceKey = sim::runtime::GetCurrDeviceKey();
    EXPECT_EQ(deviceKey, 0);
}

// ==================== Db 薄兼容接口（一对一转发合同） ====================

// Add/GetById：新增后按主键取回，与直接原子调用返回同一记录；未命中保持 NOT_FOUND。
TEST_F(SimRunnerOpsTest, DbAddAndGetById_RoundTripMatchesAtomicCall)
{
    sim::runtime::Stream stream{};
    stream.ctx_id = 1;
    stream.is_primary_default = 1;
    stream.priority = 3;
    auto added = sim::runtime::Db::Add(stream);
    ASSERT_TRUE(added.ok()) << added.diagnostic;
    ASSERT_TRUE(added.value.has_value());
    const auto streamId = added.value->value;
    EXPECT_GT(streamId, 0U);

    auto byThin = sim::runtime::Db::GetById<sim::runtime::Stream>(streamId);
    auto byAtomic
        = HcclSim::Storage::SelectOne<sim::runtime::Stream>(HcclSim::Storage::Eq(&sim::runtime::Stream::id, streamId));
    ASSERT_TRUE(byThin.ok()) << byThin.diagnostic;
    ASSERT_TRUE(byAtomic.ok()) << byAtomic.diagnostic;
    EXPECT_EQ(byThin->id, byAtomic->id);
    EXPECT_EQ(byThin->ctx_id, byAtomic->ctx_id);
    EXPECT_EQ(byThin->priority, byAtomic->priority);

    auto missing = sim::runtime::Db::GetById<sim::runtime::Stream>(streamId + 1000U);
    EXPECT_FALSE(missing.ok());
    EXPECT_EQ(missing.code, HcclSim::Storage::DbCode::NOT_FOUND);
}

// 兼容接口负向合同（编译期）：GetOneByPred/GetByPred 只接收类型化 Condition<T>，
// 不接收任何查询子句（OrderBy/Limit）与 std::function/函数指针谓词；
// Pred 表示"可下推数据库端的类型化条件"，不是迁移前的内存谓词。
static_assert(
    std::is_invocable_v<
        decltype(sim::runtime::Db::GetOneByPred<sim::runtime::Stream>),
        const HcclSim::Storage::Condition<sim::runtime::Stream>&>,
    "GetOneByPred must accept Condition<T>");
static_assert(
    !std::is_invocable_v<
        decltype(sim::runtime::Db::GetOneByPred<sim::runtime::Stream>),
        const HcclSim::Storage::Condition<sim::runtime::Stream>&,
        const HcclSim::Storage::SortClause<sim::runtime::Stream>&>,
    "GetOneByPred must not accept OrderBy");
static_assert(
    !std::is_invocable_v<
        decltype(sim::runtime::Db::GetOneByPred<sim::runtime::Stream>),
        const HcclSim::Storage::Condition<sim::runtime::Stream>&, const HcclSim::Storage::LimitClause&>,
    "GetOneByPred must not accept Limit");
static_assert(
    !std::is_invocable_v<
        decltype(sim::runtime::Db::GetOneByPred<sim::runtime::Stream>),
        const std::function<bool(const sim::runtime::Stream&)>&>,
    "GetOneByPred must not accept std::function predicate");
static_assert(
    !std::is_invocable_v<
        decltype(sim::runtime::Db::GetOneByPred<sim::runtime::Stream>), bool (*)(const sim::runtime::Stream&)>,
    "GetOneByPred must not accept function-pointer predicate");
static_assert(
    std::is_invocable_v<
        decltype(sim::runtime::Db::GetByPred<sim::runtime::Stream>),
        const HcclSim::Storage::Condition<sim::runtime::Stream>&>,
    "GetByPred must accept Condition<T>");
static_assert(
    !std::is_invocable_v<
        decltype(sim::runtime::Db::GetByPred<sim::runtime::Stream>),
        const HcclSim::Storage::Condition<sim::runtime::Stream>&,
        const HcclSim::Storage::SortClause<sim::runtime::Stream>&>,
    "GetByPred must not accept OrderBy");
static_assert(
    !std::is_invocable_v<
        decltype(sim::runtime::Db::GetByPred<sim::runtime::Stream>),
        const HcclSim::Storage::Condition<sim::runtime::Stream>&, const HcclSim::Storage::LimitClause&>,
    "GetByPred must not accept Limit");
static_assert(
    !std::is_invocable_v<
        decltype(sim::runtime::Db::GetByPred<sim::runtime::Stream>),
        const std::function<bool(const sim::runtime::Stream&)>&>,
    "GetByPred must not accept std::function predicate");

// GetOneByPred：固定主键 id 升序取首条匹配；不接收调用方排序子句。
TEST_F(SimRunnerOpsTest, DbGetOneByPred_ReturnsFirstMatchInFixedIdAscOrder)
{
    sim::runtime::Stream first{};
    first.ctx_id = 7;
    sim::runtime::Stream second{};
    second.ctx_id = 7;
    ASSERT_GT(runnerdb_test::InsertRecord(first), 0U);
    ASSERT_GT(runnerdb_test::InsertRecord(second), 0U);
    ASSERT_LT(first.id, second.id);

    auto byPred = sim::runtime::Db::GetOneByPred<sim::runtime::Stream>(
        HcclSim::Storage::Eq(&sim::runtime::Stream::ctx_id, uint64_t{7}));
    ASSERT_TRUE(byPred.ok()) << byPred.diagnostic;
    EXPECT_EQ(byPred->id, first.id);

    // 固定 id ASC 与显式带排序的直接原子调用等价。
    auto byAtomicAsc = HcclSim::Storage::SelectOne<sim::runtime::Stream>(
        HcclSim::Storage::Eq(&sim::runtime::Stream::ctx_id, uint64_t{7}),
        HcclSim::Storage::OrderBy(&sim::runtime::Stream::id, HcclSim::Storage::SortDirection::ASC));
    ASSERT_TRUE(byAtomicAsc.ok());
    EXPECT_EQ(byPred->id, byAtomicAsc->id);

    // 自定义排序（DESC）不属于兼容接口固定语义：业务直接使用原子接口表达。
    auto byAtomicDesc = HcclSim::Storage::SelectOne<sim::runtime::Stream>(
        HcclSim::Storage::Eq(&sim::runtime::Stream::ctx_id, uint64_t{7}),
        HcclSim::Storage::OrderBy(&sim::runtime::Stream::id, HcclSim::Storage::SortDirection::DESC));
    ASSERT_TRUE(byAtomicDesc.ok());
    EXPECT_EQ(byAtomicDesc->id, second.id);
}

// GetOneByPred：未命中返回 NOT_FOUND（单行读取唯一错误形态）。
TEST_F(SimRunnerOpsTest, DbGetOneByPred_MissReturnsNotFound)
{
    auto missing = sim::runtime::Db::GetOneByPred<sim::runtime::Stream>(
        HcclSim::Storage::Eq(&sim::runtime::Stream::ctx_id, uint64_t{999}));
    EXPECT_FALSE(missing.ok());
    EXPECT_EQ(missing.code, HcclSim::Storage::DbCode::NOT_FOUND);
}

// GetByPred：返回全部匹配行且固定 id 升序；Count 与行数一致。
TEST_F(SimRunnerOpsTest, DbGetByPred_ReturnsAllMatchingRowsInFixedIdAscOrder)
{
    for (uint64_t i = 0; i < 3U; i++) {
        sim::runtime::Stream stream{};
        stream.ctx_id = 8;
        stream.priority = i;
        ASSERT_GT(runnerdb_test::InsertRecord(stream), 0U);
    }
    sim::runtime::Stream other{};
    other.ctx_id = 9;
    ASSERT_GT(runnerdb_test::InsertRecord(other), 0U);

    auto rows = sim::runtime::Db::GetByPred<sim::runtime::Stream>(
        HcclSim::Storage::Eq(&sim::runtime::Stream::ctx_id, uint64_t{8}));
    ASSERT_TRUE(rows.ok()) << rows.diagnostic;
    ASSERT_TRUE(rows.value.has_value());
    ASSERT_EQ(rows.value->size(), 3U);
    for (size_t i = 1; i < rows.value->size(); i++) {
        EXPECT_LT((*rows.value)[i - 1].id, (*rows.value)[i].id);
    }

    auto counted = sim::runtime::Db::Count<sim::runtime::Stream>(
        HcclSim::Storage::Eq(&sim::runtime::Stream::ctx_id, uint64_t{8}));
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    ASSERT_TRUE(counted.value.has_value());
    EXPECT_EQ(*counted.value, 3U);

    auto atomicCounted = HcclSim::Storage::Count<sim::runtime::Stream>(
        HcclSim::Storage::Eq(&sim::runtime::Stream::ctx_id, uint64_t{8}));
    ASSERT_TRUE(atomicCounted.ok());
    EXPECT_EQ(*atomicCounted.value, *counted.value);
}

// GetByPred：全表读取的兼容形态（All<T>() 条件），固定 id 升序返回全部行。
TEST_F(SimRunnerOpsTest, DbGetByPred_AllConditionReadsWholeTableInIdAscOrder)
{
    sim::runtime::Stream seeded{};
    seeded.ctx_id = 11;
    ASSERT_GT(runnerdb_test::InsertRecord(seeded), 0U);

    auto rows = sim::runtime::Db::GetByPred<sim::runtime::Stream>(HcclSim::Storage::All<sim::runtime::Stream>());
    ASSERT_TRUE(rows.ok()) << rows.diagnostic;
    ASSERT_TRUE(rows.value.has_value());
    ASSERT_GE(rows.value->size(), 1U);
    for (size_t i = 1; i < rows.value->size(); i++) {
        EXPECT_LT((*rows.value)[i - 1].id, (*rows.value)[i].id);
    }
    const auto total = HcclSim::Storage::Count<sim::runtime::Stream>(HcclSim::Storage::All<sim::runtime::Stream>());
    ASSERT_TRUE(total.ok());
    EXPECT_EQ(rows.value->size(), static_cast<size_t>(*total.value));
}

// Update：命中返回 affectedRows=1；未命中为既有零行合同（OK + affectedRows=0）。
TEST_F(SimRunnerOpsTest, DbUpdate_ReportsAffectedRows)
{
    sim::runtime::Stream stream{};
    stream.ctx_id = 1;
    stream.priority = 1;
    ASSERT_GT(runnerdb_test::InsertRecord(stream), 0U);

    auto updated = sim::runtime::Db::Update<sim::runtime::Stream>(
        HcclSim::Storage::Eq(&sim::runtime::Stream::id, stream.id),
        HcclSim::Storage::Set(&sim::runtime::Stream::priority, uint64_t{8}));
    ASSERT_TRUE(updated.ok()) << updated.diagnostic;
    ASSERT_TRUE(updated.value.has_value());
    EXPECT_EQ(*updated.value, 1U);

    auto reloaded = sim::runtime::Db::GetById<sim::runtime::Stream>(stream.id);
    ASSERT_TRUE(reloaded.ok());
    EXPECT_EQ(reloaded->priority, 8U);

    auto untouched = sim::runtime::Db::Update<sim::runtime::Stream>(
        HcclSim::Storage::Eq(&sim::runtime::Stream::id, stream.id + 1000U),
        HcclSim::Storage::Set(&sim::runtime::Stream::priority, uint64_t{9}));
    ASSERT_TRUE(untouched.ok()) << untouched.diagnostic;
    ASSERT_TRUE(untouched.value.has_value());
    EXPECT_EQ(*untouched.value, 0U);
}

// Delete：命中 affectedRows=1；重复删除为零行合同（OK + affectedRows=0）。
TEST_F(SimRunnerOpsTest, DbDelete_ReportsAffectedRows)
{
    sim::runtime::Stream stream{};
    stream.ctx_id = 1;
    ASSERT_GT(runnerdb_test::InsertRecord(stream), 0U);

    auto deleted
        = sim::runtime::Db::Delete<sim::runtime::Stream>(HcclSim::Storage::Eq(&sim::runtime::Stream::id, stream.id));
    ASSERT_TRUE(deleted.ok()) << deleted.diagnostic;
    ASSERT_TRUE(deleted.value.has_value());
    EXPECT_EQ(*deleted.value, 1U);

    auto redeleted
        = sim::runtime::Db::Delete<sim::runtime::Stream>(HcclSim::Storage::Eq(&sim::runtime::Stream::id, stream.id));
    ASSERT_TRUE(redeleted.ok()) << redeleted.diagnostic;
    ASSERT_TRUE(redeleted.value.has_value());
    EXPECT_EQ(*redeleted.value, 0U);
}

// DeleteAll(KEEP_SEQUENCE)：清空数据但自增序列延续（下一条 id 继续递增）。
TEST_F(SimRunnerOpsTest, DbDeleteAll_KeepSequenceContinuesAutoIncrement)
{
    sim::runtime::Stream first{};
    sim::runtime::Stream second{};
    ASSERT_EQ(runnerdb_test::InsertRecord(first), 1U);
    ASSERT_EQ(runnerdb_test::InsertRecord(second), 2U);

    auto cleared = sim::runtime::Db::DeleteAll<sim::runtime::Stream>(HcclSim::Storage::ClearMode::KEEP_SEQUENCE);
    ASSERT_TRUE(cleared.ok()) << cleared.diagnostic;

    sim::runtime::Stream third{};
    ASSERT_EQ(runnerdb_test::InsertRecord(third), 3U);
    auto remaining = sim::runtime::Db::Count<sim::runtime::Stream>(HcclSim::Storage::All<sim::runtime::Stream>());
    ASSERT_TRUE(remaining.ok());
    EXPECT_EQ(*remaining.value, 1U);
}

// DeleteAll(RESET_SEQUENCE)：清空数据并重置自增序列（下一条 id 从 1 重新开始）。
TEST_F(SimRunnerOpsTest, DbDeleteAll_ResetSequenceRestartsAutoIncrement)
{
    sim::runtime::Stream first{};
    sim::runtime::Stream second{};
    ASSERT_EQ(runnerdb_test::InsertRecord(first), 1U);
    ASSERT_EQ(runnerdb_test::InsertRecord(second), 2U);

    auto cleared = sim::runtime::Db::DeleteAll<sim::runtime::Stream>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    ASSERT_TRUE(cleared.ok()) << cleared.diagnostic;

    sim::runtime::Stream third{};
    ASSERT_EQ(runnerdb_test::InsertRecord(third), 1U);
}

// ==================== VirtualMemBlock 领域接口（地址区间包含查询） ====================

namespace {
// 构造一条 VirtualMemBlock 测试记录（区间端点由编码层按基址+大小派生入库）。
sim::runtime::VirtualMemBlock MakeMemBlock(
    uint64_t startPtr, uint64_t devMappedPtr, uint64_t size, uint8_t srcType, uint64_t deviceId, uint64_t rankId)
{
    sim::runtime::VirtualMemBlock block{};
    block.start_ptr = startPtr;
    block.dev_mapped_ptr = devMappedPtr;
    block.size = size;
    block.src_type = srcType;
    block.device_id = deviceId;
    block.rank_id = rankId;
    return block;
}
} // namespace

class SimRunnerOpsVirtualMemTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();
        // id 1：device 1 / rank 0，映射区间 [0x2000, 0x2100)，虚拟区间 [0x1000, 0x1100)。
        auto blockA = MakeMemBlock(0x1000, 0x2000, 0x100, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV, 1, 0);
        ASSERT_EQ(runnerdb_test::InsertRecord(blockA), 1U);
        // id 2：device 2 / rank 1，映射区间 [0x4000, 0x4100)。
        auto blockB = MakeMemBlock(0x3000, 0x4000, 0x100, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV, 2, 1);
        ASSERT_EQ(runnerdb_test::InsertRecord(blockB), 2U);
        // id 3：device 1 / rank 2，映射区间与 id 1 重叠（[0x2000, 0x2100)）。
        auto blockC = MakeMemBlock(0x5000, 0x2000, 0x100, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV, 1, 2);
        ASSERT_EQ(runnerdb_test::InsertRecord(blockC), 3U);
        // id 4：HOST 记录，映射区间 [0x7000, 0x7100)、虚拟区间 [0x6000, 0x6100)。
        auto blockD = MakeMemBlock(0x6000, 0x7000, 0x100, (uint8_t)sim::runtime::VIR_MEM_TYPE_HOST, 1, 0);
        ASSERT_EQ(runnerdb_test::InsertRecord(blockD), 4U);
    }

    void TearDown() override { CleanUpDb(); }
};

// 设备映射地址包含查询（DEV + device 限定）：下界命中（dev_mapped_ptr == address）。
// 条件形态与业务调用点一致：And(Le(dev_mapped_ptr), Gt(dev_mapped_end_ptr), Eq(src_type,DEV),
// Eq(device_id))，固定 id ASC 取首条。
TEST_F(SimRunnerOpsVirtualMemTest, MappedDeviceBlockByDevice_LowerBoundAddressHits)
{
    auto found = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, uint64_t{0x2000}),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, uint64_t{0x2000}),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, uint64_t{1})));
    ASSERT_TRUE(found.ok()) << found.diagnostic;
    ASSERT_TRUE(found.value.has_value());
    EXPECT_EQ(found->id, 1U);
    EXPECT_EQ(found->dev_mapped_ptr, 0x2000U);
}

// 设备映射地址包含查询：上界不命中（address == 基址+大小，半开区间）。
TEST_F(SimRunnerOpsVirtualMemTest, MappedDeviceBlockByDevice_UpperBoundAddressMisses)
{
    auto found = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, uint64_t{0x2100}),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, uint64_t{0x2100}),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, uint64_t{1})));
    ASSERT_FALSE(found.ok());
    EXPECT_EQ(found.code, HcclSim::Storage::DbCode::NOT_FOUND);
}

// HOST 记录不能被 DEV 限定查询命中（映射区间与虚拟区间两形态都不命中）。
TEST_F(SimRunnerOpsVirtualMemTest, MappedDeviceBlockQueries_IgnoreHostRecords)
{
    auto mapped = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, uint64_t{0x7050}),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, uint64_t{0x7050}),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV)));
    EXPECT_FALSE(mapped.ok());
    EXPECT_EQ(mapped.code, HcclSim::Storage::DbCode::NOT_FOUND);

    auto virtualBlock = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::start_ptr, uint64_t{0x6050}),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::start_end_ptr, uint64_t{0x6050}),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV)));
    EXPECT_FALSE(virtualBlock.ok());
    EXPECT_EQ(virtualBlock.code, HcclSim::Storage::DbCode::NOT_FOUND);
}

// device 隔离：其他 device 的映射区间不命中；本 device 命中。
TEST_F(SimRunnerOpsVirtualMemTest, MappedDeviceBlockByDevice_IsolatesByDevice)
{
    auto other = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, uint64_t{0x4000}),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, uint64_t{0x4000}),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, uint64_t{1})));
    EXPECT_FALSE(other.ok());
    EXPECT_EQ(other.code, HcclSim::Storage::DbCode::NOT_FOUND);

    auto own = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, uint64_t{0x4000}),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, uint64_t{0x4000}),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, uint64_t{2})));
    ASSERT_TRUE(own.ok()) << own.diagnostic;
    EXPECT_EQ(own->id, 2U);
}

// rank 隔离：按 rank 匹配的映射区间查询只命中该 rank 的记录。
TEST_F(SimRunnerOpsVirtualMemTest, MappedDeviceBlockByRank_IsolatesByRank)
{
    auto own = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, uint64_t{0x2050}),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, uint64_t{0x2050}),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::rank_id, uint64_t{2})));
    ASSERT_TRUE(own.ok()) << own.diagnostic;
    EXPECT_EQ(own->id, 3U);

    auto other = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, uint64_t{0x2050}),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, uint64_t{0x2050}),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::rank_id, uint64_t{1})));
    EXPECT_FALSE(other.ok());
    EXPECT_EQ(other.code, HcclSim::Storage::DbCode::NOT_FOUND);
}

// 多条匹配时 id ASC 选择保持（重叠区间内取最小 id，与迁移前逐行扫描首中语义一致）。
TEST_F(SimRunnerOpsVirtualMemTest, MappedDeviceBlockByDevice_PicksLowestIdOnOverlap)
{
    auto found = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, uint64_t{0x2050}),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, uint64_t{0x2050}),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, uint64_t{1})));
    ASSERT_TRUE(found.ok()) << found.diagnostic;
    EXPECT_EQ(found->id, 1U);
}

// 虚拟地址区间（DEV 限定）：命中 DEV 记录且不限定 device；上界不命中。
TEST_F(SimRunnerOpsVirtualMemTest, DevVirtualBlock_MatchesStartRangeDevOnly)
{
    auto found = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::start_ptr, uint64_t{0x3050}),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::start_end_ptr, uint64_t{0x3050}),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV)));
    ASSERT_TRUE(found.ok()) << found.diagnostic;
    EXPECT_EQ(found->id, 2U);

    auto upperMiss = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::start_ptr, uint64_t{0x1100}),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::start_end_ptr, uint64_t{0x1100}),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV)));
    EXPECT_FALSE(upperMiss.ok());
    EXPECT_EQ(upperMiss.code, HcclSim::Storage::DbCode::NOT_FOUND);
}

// 虚拟地址区间（不限类型）：HOST 记录也能命中（IPC 导出键场景）。
TEST_F(SimRunnerOpsVirtualMemTest, VirtualBlock_MatchesAnySrcType)
{
    auto hostFound = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::start_ptr, uint64_t{0x6050}),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::start_end_ptr, uint64_t{0x6050})));
    ASSERT_TRUE(hostFound.ok()) << hostFound.diagnostic;
    EXPECT_EQ(hostFound->id, 4U);

    auto devFound = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::start_ptr, uint64_t{0x1050}),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::start_end_ptr, uint64_t{0x1050})));
    ASSERT_TRUE(devFound.ok()) << devFound.diagnostic;
    EXPECT_EQ(devFound->id, 1U);

    auto miss = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::start_ptr, uint64_t{0x9000}),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::start_end_ptr, uint64_t{0x9000})));
    EXPECT_FALSE(miss.ok());
    EXPECT_EQ(miss.code, HcclSim::Storage::DbCode::NOT_FOUND);
}
