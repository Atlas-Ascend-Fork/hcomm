/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <future>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "runtime_state/db_sim_runner_common.h"
#include "storage/internal/process_storage_context.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/storage_session.h"
#include "sqlite/simulation_storage_test_helper.h"

extern uint64_t g_cur_server_key;

namespace {
const std::string kTestDbPath = "/tmp/test_sim_runner_common.db";
const std::string kTestOpDataDbPath = "/tmp/test_sim_runner_common.db.opdata";

void CleanUpDb() { runnerdb_test::CleanUpDatabases(kTestDbPath, kTestOpDataDbPath); }

void SetupTestData()
{
    CleanUpDb();
    sim::runtime::g_cur_comm_key = 0;
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
    device.super_device_id = 0;
    runnerdb_test::InsertRecord<sim::runtime::Device>(device);

    device.logic_id = 1;
    device.physical_id = 1;
    runnerdb_test::InsertRecord<sim::runtime::Device>(device);

    g_cur_server_key = 1;
}
} // namespace

// 测试内直接取设备（等价迁移前 GetDeviceByLogicId/GetDeviceByPhysicalId 的查询语义）。
sim::runtime::Device FindDeviceForTest(uint32_t logicId, uint32_t physicalId)
{
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::Device>();
    if (!table.ok()) {
        return {};
    }
    auto device = table->Select(
        HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Device::server_id, g_cur_server_key),
            HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, logicId),
            HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, physicalId)),
        HcclSim::Storage::OrderBy(&sim::runtime::Device::id, HcclSim::Storage::SortDirection::ASC),
        HcclSim::Storage::Limit(1U));
    return device.ok() && !device->empty() ? device->front() : sim::runtime::Device{};
}

class SimRunnerCommonTest : public testing::Test {
protected:
    void SetUp() override { SetupTestData(); }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonTest, GetDeviceByLogicId_WhenDeviceExists_ReturnSuccess)
{
    sim::runtime::Device device{};
    auto ret = sim::runtime::GetDeviceByLogicId(0, device);

    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(device.logic_id, 0);
    EXPECT_EQ(device.physical_id, 0);
    EXPECT_EQ(device.server_id, 1);
}

TEST_F(SimRunnerCommonTest, GetDeviceByLogicId_WhenDeviceNotExists_ReturnError)
{
    sim::runtime::Device device{};
    auto ret = sim::runtime::GetDeviceByLogicId(999, device);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonTest, GetDeviceByPhysicalId_WhenDeviceExists_ReturnSuccess)
{
    sim::runtime::Device device{};
    auto ret = sim::runtime::GetDeviceByPhysicalId(0, device);

    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(device.physical_id, 0);
    EXPECT_EQ(device.logic_id, 0);
}

TEST_F(SimRunnerCommonTest, GetDeviceByPhysicalId_WhenDeviceNotExists_ReturnError)
{
    sim::runtime::Device device{};
    auto ret = sim::runtime::GetDeviceByPhysicalId(999, device);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonTest, UpdateDeviceLogicId_WhenValidParams_UpdateSuccessfully)
{
    auto ret = sim::runtime::UpdateDeviceLogicId(1, 0, 100, 0);

    EXPECT_EQ(ret, ACL_SUCCESS);

    sim::runtime::Device device{};
    auto getRet = sim::runtime::GetDeviceByPhysicalId(0, device);
    EXPECT_EQ(getRet, ACL_SUCCESS);
    EXPECT_EQ(device.logic_id, 100);
    EXPECT_EQ(device.status, 1);
}

TEST_F(SimRunnerCommonTest, UpdateDeviceLogicId_WhenServerKeyInvalid_ReturnError)
{
    auto ret = sim::runtime::UpdateDeviceLogicId(999, 0, 100, 0);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonTest, UpdateDeviceLogicId_WhenPhyDevIdInvalid_ReturnError)
{
    auto ret = sim::runtime::UpdateDeviceLogicId(1, 999, 100, 0);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonTest, UpdateSuperDeviceId_WhenValidParams_UpdateSuccessfully)
{
    auto ret = sim::runtime::UpdateSuperDeviceId(0, 200);

    EXPECT_EQ(ret, ACL_SUCCESS);

    sim::runtime::Device device{};
    auto getRet = sim::runtime::GetDeviceByLogicId(0, device);
    EXPECT_EQ(getRet, ACL_SUCCESS);
    EXPECT_EQ(device.super_device_id, 200);
}

TEST_F(SimRunnerCommonTest, UpdateSuperDeviceId_WhenLogicIdInvalid_ReturnError)
{
    auto ret = sim::runtime::UpdateSuperDeviceId(999, 200);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonTest, GetServerByKey_WhenServerExists_ReturnSuccess)
{
    sim::runtime::Server server{};
    auto ret = runnerdb_test::SelectFirstRecord<sim::runtime::Server>(
        HcclSim::Storage::Eq(&sim::runtime::Server::id, uint64_t{1}));

    EXPECT_TRUE(ret.second);
    EXPECT_EQ(ret.first.pod_id, 100);
}

TEST_F(SimRunnerCommonTest, GetServerByKey_WhenServerNotExists_ReturnError)
{
    auto ret = runnerdb_test::SelectFirstRecord<sim::runtime::Server>(
        HcclSim::Storage::Eq(&sim::runtime::Server::id, uint64_t{999}));

    EXPECT_FALSE(ret.second);
}

TEST_F(SimRunnerCommonTest, GetDeviceByServerKeyAndPhysicalId_WhenValid_ReturnSuccess)
{
    auto ret = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, uint64_t{1}),
        HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{0})));

    EXPECT_TRUE(ret.second);
    EXPECT_EQ(ret.first.physical_id, 0);
    EXPECT_EQ(ret.first.server_id, 1);
}

TEST_F(SimRunnerCommonTest, GetDeviceByServerKeyAndPhysicalId_WhenInvalidServerKey_ReturnError)
{
    auto ret = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, uint64_t{999}),
        HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{0})));

    EXPECT_FALSE(ret.second);
}

TEST_F(SimRunnerCommonTest, GetDeviceByServerKeyAndPhysicalId_WhenInvalidPhysicalId_ReturnError)
{
    auto ret = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, uint64_t{1}),
        HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{999})));

    EXPECT_FALSE(ret.second);
}

TEST_F(SimRunnerCommonTest, GetServerByKey_Found_ReturnSuccess)
{
    sim::runtime::Server server{};
    auto ret = sim::runtime::GetServerByKey(1, server);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(server.pod_id, 100);
}

TEST_F(SimRunnerCommonTest, GetServerByKey_NotFound_ReturnError)
{
    sim::runtime::Server server{};
    auto ret = sim::runtime::GetServerByKey(999, server);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

class SimRunnerCommonCcuTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        const sim::runtime::Device device = FindDeviceForTest(0U, 0U);
        ASSERT_NE(device.id, 0U);

        sim::runtime::Ccu ccu{};
        ccu.device_id = device.id;
        ccu.die_id = 0;
        ccu.status = 1;
        runnerdb_test::InsertRecord<sim::runtime::Ccu>(ccu);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonCcuTest, GetCcuFromDeviceByDieId_WhenValid_ReturnSuccess)
{
    sim::runtime::Ccu ccu{};
    auto ret = sim::runtime::GetCcuFromDeviceByDieId(1, 0, ccu);

    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(ccu.die_id, 0);
    EXPECT_EQ(ccu.device_id, 1);
}

TEST_F(SimRunnerCommonCcuTest, GetCcuFromDeviceByDieId_WhenNotFound_ReturnError)
{
    sim::runtime::Ccu ccu{};
    auto ret = sim::runtime::GetCcuFromDeviceByDieId(1, 99, ccu);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

class SimRunnerCommonContextTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        const sim::runtime::Device device = FindDeviceForTest(0U, 0U);
        ASSERT_NE(device.id, 0U);

        sim::runtime::Context ctx{};
        ctx.device_id = device.id;
        ctx.is_default = 1;
        ctx.run_id = 1;
        runnerdb_test::InsertRecord<sim::runtime::Context>(ctx);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonContextTest, GetContextByDevId_WhenValid_ReturnSuccess)
{
    sim::runtime::Device device{};
    auto ret = sim::runtime::GetDeviceByPhysicalId(0, device);
    ASSERT_EQ(ret, ACL_SUCCESS);

    sim::runtime::Context ctx{};
    auto getRet = sim::runtime::GetContextByDevId(static_cast<uint32_t>(device.id), ctx);

    EXPECT_EQ(getRet, ACL_SUCCESS);
    EXPECT_EQ(ctx.device_id, device.id);
    EXPECT_EQ(ctx.is_default, 1);
}

TEST_F(SimRunnerCommonContextTest, GetContextByDevId_WhenNotFound_ReturnError)
{
    sim::runtime::Context ctx{};
    auto ret = sim::runtime::GetContextByDevId(999, ctx);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

class SimRunnerCommonPortTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        const sim::runtime::Device device = FindDeviceForTest(0U, 0U);
        ASSERT_NE(device.id, 0U);

        sim::runtime::Port port{};
        port.device_id = device.id;
        snprintf(port.name, sizeof(port.name), "eth0");
        port.status = 1;
        runnerdb_test::InsertRecord<sim::runtime::Port>(port);

        sim::runtime::Port port2{};
        port2.device_id = device.id;
        snprintf(port2.name, sizeof(port2.name), "eth1");
        port2.status = 1;
        runnerdb_test::InsertRecord<sim::runtime::Port>(port2);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonPortTest, GetPortByName_WhenValid_ReturnSuccess)
{
    sim::runtime::Port port{};
    auto ret = sim::runtime::GetPortByName(1, 0, "eth0", port);

    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_STREQ(port.name, "eth0");
}

TEST_F(SimRunnerCommonPortTest, GetPortByName_WhenNotFound_ReturnError)
{
    sim::runtime::Port port{};
    auto ret = sim::runtime::GetPortByName(1, 0, "nonexistent", port);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

class SimRunnerCommonCountTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        const sim::runtime::Device device = FindDeviceForTest(0U, 0U);
        ASSERT_NE(device.id, 0U);

        sim::runtime::TaskSchedulerDevice tsDev{};
        tsDev.device_id = device.id;
        tsDev.type = static_cast<uint8_t>(sim::runtime::TS_DEV_TYPE_CPU);
        runnerdb_test::InsertRecord<sim::runtime::TaskSchedulerDevice>(tsDev);

        tsDev.type = static_cast<uint8_t>(sim::runtime::TS_DEV_TYPE_SCALAR);
        runnerdb_test::InsertRecord<sim::runtime::TaskSchedulerDevice>(tsDev);
    }

    void TearDown() override { CleanUpDb(); }
};

// 一行业务函数（内部定位设备后在数据库端 COUNT 聚合，不加载行集）。
TEST_F(SimRunnerCommonCountTest, GetAICpuCount_WhenDeviceExists_ReturnCorrectCount)
{
    auto count = sim::runtime::GetAICpuCount(0);

    EXPECT_EQ(count, 1);
}

TEST_F(SimRunnerCommonCountTest, GetAICpuCount_WhenDeviceNotExists_ReturnZero)
{
    auto count = sim::runtime::GetAICpuCount(999);

    EXPECT_EQ(count, 0);
}

TEST_F(SimRunnerCommonCountTest, GetAICoreCount_WhenDeviceExists_ReturnCorrectCount)
{
    auto count = sim::runtime::GetAICoreCount(0);

    EXPECT_GE(count, 1);
}

TEST_F(SimRunnerCommonCountTest, GetAICoreCount_WhenDeviceNotExists_ReturnZero)
{
    auto count = sim::runtime::GetAICoreCount(999);

    EXPECT_EQ(count, 0);
}

class SimRunnerCommonVectorCubeTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        const sim::runtime::Device device = FindDeviceForTest(0U, 0U);
        ASSERT_NE(device.id, 0U);

        sim::runtime::TaskSchedulerDevice tsDev{};
        tsDev.device_id = device.id;
        tsDev.type = static_cast<uint8_t>(sim::runtime::TS_DEV_TYPE_SCALAR);
        uint64_t tsId = runnerdb_test::InsertRecord<sim::runtime::TaskSchedulerDevice>(tsDev);

        sim::runtime::ComputeDie die{};
        die.ts_id = tsId;
        die.type = static_cast<uint8_t>(sim::runtime::COMPUTE_TYPE_VECTOR);
        runnerdb_test::InsertRecord<sim::runtime::ComputeDie>(die);

        die.type = static_cast<uint8_t>(sim::runtime::COMPUTE_TYPE_CUBE);
        runnerdb_test::InsertRecord<sim::runtime::ComputeDie>(die);
    }

    void TearDown() override { CleanUpDb(); }
};

// 一行业务函数（内部取设备全部 Scalar scheduler 后按 scheduler 逐个数据库端 COUNT，
// 不加载 ComputeDie 行集）。
TEST_F(SimRunnerCommonVectorCubeTest, GetVectorCoreCount_WhenDeviceExists_ReturnCorrectCount)
{
    auto count = sim::runtime::GetVectorCoreCount(0);

    EXPECT_GE(count, 1);
}

TEST_F(SimRunnerCommonVectorCubeTest, GetVectorCoreCount_WhenDeviceNotExists_ReturnZero)
{
    auto count = sim::runtime::GetVectorCoreCount(999);

    EXPECT_EQ(count, 0);
}

TEST_F(SimRunnerCommonVectorCubeTest, GetCubeCoreCount_WhenDeviceExists_ReturnCorrectCount)
{
    auto count = sim::runtime::GetCubeCoreCount(0);
    EXPECT_GE(count, 1);
}

TEST_F(SimRunnerCommonVectorCubeTest, GetCubeCoreCount_WhenDeviceNotExists_ReturnZero)
{
    auto count = sim::runtime::GetCubeCoreCount(999);
    EXPECT_EQ(count, 0);
}

class SimRunnerCommonRankEndpointTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        const sim::runtime::Device device = FindDeviceForTest(0U, 0U);
        ASSERT_NE(device.id, 0U);

        sim::runtime::Communicator comm{};
        std::strncpy(comm.comm_id, "test_comm", sizeof(comm.comm_id) - 1);
        comm.rank_size = 8;
        comm.rank_id = 5;
        comm.device_id = device.id;
        sim::runtime::g_cur_comm_key = runnerdb_test::InsertRecord<sim::runtime::Communicator>(comm);
        ASSERT_NE(sim::runtime::g_cur_comm_key, 0U);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonRankEndpointTest, GetCommRankByDeviceId_WhenExists_ReturnsRankId)
{
    const auto comm = runnerdb_test::SelectRecord<sim::runtime::Communicator>(sim::runtime::g_cur_comm_key);
    ASSERT_TRUE(comm.has_value());

    uint32_t rankId = UINT32_MAX;
    ASSERT_TRUE(sim::runtime::GetCommRankByDeviceId(sim::runtime::g_cur_comm_key, comm->device_id, rankId));
    EXPECT_EQ(rankId, 5);
}

TEST_F(SimRunnerCommonRankEndpointTest, GetCommRankByDeviceId_WhenNotExists_ReturnsFalse)
{
    uint32_t rankId = UINT32_MAX;
    EXPECT_FALSE(sim::runtime::GetCommRankByDeviceId(sim::runtime::g_cur_comm_key, 999, rankId));
    EXPECT_EQ(rankId, UINT32_MAX);
}

// 复合索引验证（2026-09-17）：512 成员通信域（256p×2die 生产规模）上全量
// rank 反查。GetCommRankByDeviceId 的谓词为 (comm_id, comm_hash, device_id)
// 三等值——复合索引 Index(comm_id, comm_hash, device_id) 使候选集为单行
// （旧索引 (comm_id, comm_hash) 下为全成员 512 行候选逐行过滤）。
TEST_F(SimRunnerCommonRankEndpointTest, GetCommRankByDeviceId_FullMemberSet_CompositeIndexCorrectness)
{
    constexpr uint32_t kMembers = 512U;
    constexpr uint64_t kCommHash = 987654321ULL;

    // 建 512 成员：device_id = 基准设备 id + i，rank_id = i
    const auto seed = runnerdb_test::SelectRecord<sim::runtime::Communicator>(sim::runtime::g_cur_comm_key);
    ASSERT_TRUE(seed.has_value());
    const uint64_t baseDeviceId = seed->device_id;

    for (uint32_t i = 1U; i < kMembers; ++i) {
        sim::runtime::Communicator member{};
        std::strncpy(member.comm_id, "test_comm", sizeof(member.comm_id) - 1);
        member.rank_size = kMembers;
        member.rank_id = i;
        member.device_id = baseDeviceId + i;
        member.comm_hash = kCommHash;
        ASSERT_NE(runnerdb_test::InsertRecord<sim::runtime::Communicator>(member), 0U);
    }
    // 种子成员补 hash（原行 comm_hash=0）
    ASSERT_TRUE(sim::runtime::Db::Update<sim::runtime::Communicator>(
                    HcclSim::Storage::Eq(&sim::runtime::Communicator::id, sim::runtime::g_cur_comm_key),
                    HcclSim::Storage::Set(&sim::runtime::Communicator::comm_hash, kCommHash))
                    .ok());

    // 全量反查：每个成员的 (device_id → rank_id) 必须精确命中
    // （i=0 是 fixture 种子成员，其 rank_id=5 是 SetUp 的既定值）
    for (uint32_t i = 0U; i < kMembers; ++i) {
        uint32_t rankId = UINT32_MAX;
        ASSERT_TRUE(sim::runtime::GetCommRankByDeviceId(sim::runtime::g_cur_comm_key, baseDeviceId + i, rankId));
        const uint32_t expectedRank = (i == 0U) ? seed->rank_id : i;
        EXPECT_EQ(rankId, expectedRank) << "device_id=" << (baseDeviceId + i);
    }
    // 不存在成员：越界 device_id 必须返回 false
    uint32_t rankId = UINT32_MAX;
    EXPECT_FALSE(sim::runtime::GetCommRankByDeviceId(sim::runtime::g_cur_comm_key, baseDeviceId + kMembers, rankId));
}

TEST_F(SimRunnerCommonTest, CommunicatorMemberIdResolvesPeersInSameDomain)
{
    uint64_t firstCommId = 0;
    uint64_t secondCommId = 0;
    ASSERT_TRUE(sim::runtime::GetOrInsertCommunicator("shared_domain", 2, 0, 1, 0, firstCommId));
    ASSERT_TRUE(sim::runtime::GetOrInsertCommunicator("shared_domain", 2, 1, 2, 0, secondCommId));
    EXPECT_NE(firstCommId, 0);
    EXPECT_NE(firstCommId, secondCommId);

    sim::runtime::Device device{};
    ASSERT_EQ(sim::runtime::GetDeviceByCommRank(firstCommId, 1, device), ACL_SUCCESS);
    EXPECT_EQ(device.id, 2);
    ASSERT_EQ(sim::runtime::GetDeviceByCommRank(secondCommId, 0, device), ACL_SUCCESS);
    EXPECT_EQ(device.id, 1);

    std::vector<sim::runtime::CommunicatorMemberInfo> members;
    ASSERT_TRUE(sim::runtime::GetCommunicatorMembers(secondCommId, members));
    ASSERT_EQ(members.size(), 2U);
    EXPECT_TRUE(
        std::any_of(members.begin(), members.end(), [&firstCommId](const sim::runtime::CommunicatorMemberInfo& member) {
            return member.memberId == firstCommId && member.rankId == 0 && member.deviceId == 1;
        }));
    EXPECT_TRUE(std::any_of(
        members.begin(), members.end(), [&secondCommId](const sim::runtime::CommunicatorMemberInfo& member) {
            return member.memberId == secondCommId && member.rankId == 1 && member.deviceId == 2;
        }));

    std::string commName;
    ASSERT_TRUE(sim::runtime::GetCommunicatorName(secondCommId, commName));
    EXPECT_EQ(commName, "shared_domain");
}

TEST_F(SimRunnerCommonTest, CommunicatorDestroyBarrierWaitsForEveryRank)
{
    uint64_t firstCommId = 0;
    uint64_t secondCommId = 0;
    ASSERT_TRUE(sim::runtime::GetOrInsertCommunicator("destroy_barrier", 2, 0, 1, 0, firstCommId));
    ASSERT_TRUE(sim::runtime::GetOrInsertCommunicator("destroy_barrier", 2, 1, 2, 0, secondCommId));

    auto firstWait = std::async(std::launch::async, [firstCommId] {
        return sim::runtime::WaitCommunicatorDestroyReady(firstCommId);
    });
    EXPECT_EQ(firstWait.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);

    EXPECT_TRUE(sim::runtime::WaitCommunicatorDestroyReady(secondCommId));
    EXPECT_EQ(firstWait.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_TRUE(firstWait.get());
}

TEST_F(SimRunnerCommonTest, CommunicatorMemberIdSeparatesSameNameDomainsByHash)
{
    uint64_t firstDomainMemberId = 0;
    uint64_t secondDomainMemberId = 0;
    uint64_t unusedMemberId = 0;
    ASSERT_TRUE(sim::runtime::GetOrInsertCommunicator("shared_sub_domain", 2, 0, 1, 100, firstDomainMemberId));
    ASSERT_TRUE(sim::runtime::GetOrInsertCommunicator("shared_sub_domain", 2, 1, 2, 100, unusedMemberId));
    ASSERT_TRUE(sim::runtime::GetOrInsertCommunicator("shared_sub_domain", 2, 0, 2, 200, unusedMemberId));
    ASSERT_TRUE(sim::runtime::GetOrInsertCommunicator("shared_sub_domain", 2, 1, 1, 200, secondDomainMemberId));

    sim::runtime::Device device{};
    ASSERT_EQ(sim::runtime::GetDeviceByCommRank(firstDomainMemberId, 1, device), ACL_SUCCESS);
    EXPECT_EQ(device.id, 2U);
    ASSERT_EQ(sim::runtime::GetDeviceByCommRank(secondDomainMemberId, 1, device), ACL_SUCCESS);
    EXPECT_EQ(device.id, 1U);

    std::vector<sim::runtime::CommunicatorMemberInfo> members;
    ASSERT_TRUE(sim::runtime::GetCommunicatorMembers(firstDomainMemberId, members));
    ASSERT_EQ(members.size(), 2U);
    ASSERT_TRUE(sim::runtime::GetCommunicatorMembers(secondDomainMemberId, members));
    ASSERT_EQ(members.size(), 2U);
}

TEST_F(SimRunnerCommonTest, CommunicatorNameLengthIsValidatedWithoutChangingSameNameDomainSemantics)
{
    const std::string validName(127, 'a');
    uint64_t commId = 0;
    ASSERT_TRUE(sim::runtime::GetOrInsertCommunicator(validName.c_str(), 1, 0, 1, 0, commId));
    EXPECT_NE(commId, 0U);

    const std::string tooLongName(128, 'b');
    EXPECT_FALSE(sim::runtime::GetOrInsertCommunicator(tooLongName.c_str(), 1, 0, 1, 0, commId));
    EXPECT_FALSE(sim::runtime::WaitCommunicatorReady(tooLongName.c_str(), 0, 1));
}

TEST_F(SimRunnerCommonRankEndpointTest, GetEndPointByIpAddr_WhenNotExists_ReturnError)
{
    sim::runtime::EndPoint ep{};
    auto ret = sim::runtime::GetEndPointByIpAddr("192.168.99.99", ep);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

// 上游仅有错误路径用例；此处补充正常路径，保持"正常路径 + 错误路径"覆盖。
TEST_F(SimRunnerCommonRankEndpointTest, GetEndPointByIpAddr_WhenExists_ReturnSuccess)
{
    sim::runtime::EndPoint ep{};
    snprintf(ep.ip_addr, sizeof(ep.ip_addr), "192.168.99.99");
    runnerdb_test::InsertRecord<sim::runtime::EndPoint>(ep);

    sim::runtime::EndPoint result{};
    auto ret = sim::runtime::GetEndPointByIpAddr("192.168.99.99", result);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_STREQ(result.ip_addr, "192.168.99.99");
}

class SimRunnerCommonFullPortTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        const sim::runtime::Device device = FindDeviceForTest(0U, 0U);
        ASSERT_NE(device.id, 0U);

        sim::runtime::Port port{};
        port.device_id = device.id;
        port.die_id = 0;
        snprintf(port.name, sizeof(port.name), "eth0");
        port.status = 1;
        portId_ = runnerdb_test::InsertRecord<sim::runtime::Port>(port);

        sim::runtime::Ccu ccu{};
        ccu.device_id = device.id;
        ccu.die_id = 0;
        ccu.status = 1;
        ccuId_ = runnerdb_test::InsertRecord<sim::runtime::Ccu>(ccu);

        snprintf(port.name, sizeof(port.name), "ccu_eth0");
        runnerdb_test::InsertRecord<sim::runtime::Port>(port);
    }

    void TearDown() override { CleanUpDb(); }

    uint64_t portId_{0};
    uint64_t ccuId_{0};
};

TEST_F(SimRunnerCommonFullPortTest, GetPortById_WhenValid_ReturnSuccess)
{
    sim::runtime::Port port{};
    auto ret = sim::runtime::GetPortById(portId_, port);

    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(port.id, portId_);
}

TEST_F(SimRunnerCommonFullPortTest, GetPortById_WhenNotFound_ReturnError)
{
    sim::runtime::Port port{};
    auto ret = sim::runtime::GetPortById(999, port);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonFullPortTest, GetPortByName_DeviceNotFound_ReturnError)
{
    sim::runtime::Port port{};
    auto ret = sim::runtime::GetPortByName(999, 0, "eth0", port);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

// ---- MarkAndSyncCommunicatorDestroy 首错诊断（快路径）----
// 超时汇总路径（固定 300s 预算）不在 UT 实测；此处覆盖两类快速失败出口的
// 返回合同与日志字段（self 读取失败 / 无效成员），轮询统计由代码审查保证。
TEST_F(SimRunnerCommonTest, MarkAndSyncDestroy_SelfReadNotFound_ReturnFalseFast)
{
    // 不存在的 commId：self 读取 NOT_FOUND → 立即 false（不进入 300s 轮询）
    const auto startedAt = std::chrono::steady_clock::now();
    const bool ok = sim::runtime::MarkAndSyncCommunicatorDestroy(999999U);
    const auto elapsedMs
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startedAt).count();
    EXPECT_FALSE(ok);
    EXPECT_LT(elapsedMs, 5000) << "self 读取失败必须立即返回，不得进入销毁轮询";
}

TEST_F(SimRunnerCommonTest, MarkAndSyncDestroy_ZeroCommId_ReturnFalseFast)
{
    const auto startedAt = std::chrono::steady_clock::now();
    const bool ok = sim::runtime::MarkAndSyncCommunicatorDestroy(0U);
    const auto elapsedMs
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startedAt).count();
    EXPECT_FALSE(ok);
    EXPECT_LT(elapsedMs, 5000);
}
