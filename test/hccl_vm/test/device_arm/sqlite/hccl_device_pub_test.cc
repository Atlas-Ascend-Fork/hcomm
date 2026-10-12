/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <set>
#include <thread>
#include <unistd.h>
#include <vector>

#include "hccl_device_pub.h"
#include "sim_capacity_limits.h"
#include "store_sim_memory_manager.h"
#include "storage/internal/process_storage_context.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/storage_session.h"
#include "sqlite/simulation_storage_test_helper.h"

using namespace HcclSim;

extern uint32_t g_rankId;
extern std::map<uint32_t, uint32_t> sqTailMap;

namespace {
const std::string kTestDbPath = "/tmp/test_hccl_device_pub.db";

void CleanUpDb()
{
    std::remove(kTestDbPath.c_str());
    std::remove((kTestDbPath + "-wal").c_str());
    std::remove((kTestDbPath + "-shm").c_str());
}

/// 插入一行 PhyMemBlock（owner=deviceId），返回主键（0=失败）
uint64_t InsertPhyOwnedBy(uint64_t deviceId, const char* name)
{
    sim::runtime::PhyMemBlock phyMem{};
    phyMem.device_id = deviceId;
    snprintf(phyMem.name, sizeof(phyMem.name), "%s", name);
    phyMem.size = 0x10000;
    phyMem.type = 0;
    phyMem.ref_count = 1;
    return runnerdb_test::InsertRecord<sim::runtime::PhyMemBlock>(phyMem);
}

/// 插入一行 DEV 型 VirtualMemBlock（映射设备=deviceId），返回主键（0=失败）。
/// 区间端点为派生列（base+size 编码期计算），测试只赋基址与大小。
uint64_t InsertDevVmb(uint64_t deviceId, uint64_t startPtr, uint64_t mappedPtr, uint64_t size, uint64_t phyMemId)
{
    sim::runtime::VirtualMemBlock virMem{};
    virMem.start_ptr = startPtr;
    virMem.dev_mapped_ptr = mappedPtr;
    virMem.size = size;
    virMem.ctx_id = 1;
    virMem.device_id = deviceId;
    virMem.phy_mem_id = phyMemId;
    virMem.owner_pid = 0;
    virMem.src_type = (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV;
    virMem.policy = 0;
    return runnerdb_test::InsertRecord<sim::runtime::VirtualMemBlock>(virMem);
}

/// 取 fixture SetupTestData 插入的首个 Device（physical_id=0）主键
uint64_t FirstDeviceId()
{
    auto devRet = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
        HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{0}));
    return devRet.second ? devRet.first.id : 0;
}

void SetupTestData()
{
    CleanUpDb();
    // 迁移前 SetDbPath + ClearAll 的等价语义：以本用例数据库路径重建进程存储会话。
    (void)runnerdb_test::ResetTestSession(kTestDbPath, kTestDbPath);

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
    sim::runtime::g_cur_comm_key = 0;
}
} // namespace

class HcclDevicePubTest : public testing::Test {
protected:
    void SetUp() override
    {
        g_rankId = 0;
        sqTailMap.clear();
        SetupTestData();
    }

    void TearDown() override
    {
        runnerdb_test::ClearRecords<sim::runtime::EndPoint>();
        runnerdb_test::ClearRecords<sim::runtime::VirtualMemBlock>();
        runnerdb_test::ClearRecords<sim::runtime::PhyMemBlock>();
        CleanUpDb();
    }
};

TEST_F(HcclDevicePubTest, SetCurRankId_Valid)
{
    uint32_t rankId = 5;
    HcclVmResult result = SetCurRankId(rankId);
    EXPECT_EQ(result, HcclVmResult::HCCL_SIM_SUCCESS);
}

// ==================== AICPU Jetty 编号合同（sim_capacity_limits.h）====================

TEST_F(HcclDevicePubTest, AicpuJettyReservationIsMonotonicAndFailsClosed)
{
    // 单调递增、发布真实硬件编号
    std::atomic<uint32_t> seq{0};
    uint32_t first = 0;
    uint32_t second = 0;
    ASSERT_TRUE(HcclSim::TryReserveAicpuJettyHwId(seq, first));
    ASSERT_TRUE(HcclSim::TryReserveAicpuJettyHwId(seq, second));
    EXPECT_EQ(first, HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN);
    EXPECT_EQ(second, HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN + 1);

    // 边界：最后一个合法编号是 END；再预留即失败
    std::atomic<uint32_t> tail{HcclSim::HCCL_VM_AICPU_JETTY_CAPACITY_PER_RANK - 1};
    uint32_t last = 0;
    ASSERT_TRUE(HcclSim::TryReserveAicpuJettyHwId(tail, last));
    EXPECT_EQ(last, HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_END);
    EXPECT_FALSE(HcclSim::TryReserveAicpuJettyHwId(tail, last));

    // 耗尽后连续失败：计数停留在上限不增长、不回绕，出参不被污染
    std::atomic<uint32_t> exhausted{HcclSim::HCCL_VM_AICPU_JETTY_CAPACITY_PER_RANK};
    uint32_t unchanged = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    for (int i = 0; i < 3; i++) {
        EXPECT_FALSE(HcclSim::TryReserveAicpuJettyHwId(exhausted, unchanged));
    }
    EXPECT_EQ(exhausted.load(), HcclSim::HCCL_VM_AICPU_JETTY_CAPACITY_PER_RANK);
    EXPECT_EQ(unchanged, HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN);
}

TEST_F(HcclDevicePubTest, AicpuJettyReservationConcurrentNoDuplicate)
{
    // 并发预留无重复：全部落在 [BEGIN, END] 内且互不相同
    std::atomic<uint32_t> seq{0};
    constexpr int kThreads = 8;
    constexpr int kPerThread = 64;
    std::vector<std::thread> workers;
    std::vector<uint32_t> ids(kThreads * kPerThread, 0);
    for (int t = 0; t < kThreads; t++) {
        workers.emplace_back([&, t]() {
            for (int i = 0; i < kPerThread; i++) {
                uint32_t id = 0;
                if (HcclSim::TryReserveAicpuJettyHwId(seq, id)) {
                    ids[static_cast<size_t>(t) * kPerThread + i] = id;
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    std::set<uint32_t> unique(ids.begin(), ids.end());
    unique.erase(0); // 0 为未成功哨兵（合法编号从 BEGIN=5312 起）
    EXPECT_EQ(unique.size(), static_cast<size_t>(kThreads * kPerThread));
    ASSERT_FALSE(unique.empty());
    EXPECT_LE(*unique.rbegin(), HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_END);
    EXPECT_GE(*unique.begin(), HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN);
}

TEST_F(HcclDevicePubTest, SetCurRankId_Zero)
{
    HcclVmResult result = SetCurRankId(0);
    EXPECT_EQ(result, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(HcclDevicePubTest, SetCurRankId_MaxValue)
{
    HcclVmResult result = SetCurRankId(UINT32_MAX);
    EXPECT_EQ(result, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(HcclDevicePubTest, GetCurRankId_AfterSet)
{
    uint32_t setRankId = 10;
    SetCurRankId(setRankId);

    uint32_t getRankId = 0;
    HcclVmResult result = GetCurRankId(&getRankId);
    EXPECT_EQ(result, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(getRankId, setRankId);
}

TEST_F(HcclDevicePubTest, GetSqBufferAddr_Valid)
{
    uint8_t* sqBuff = nullptr;
    HcclVmResult result = GetSqBufferAddr(&sqBuff);
    EXPECT_EQ(result, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_NE(sqBuff, nullptr);
}

TEST_F(HcclDevicePubTest, GetSqBufferAddr_BufferSize)
{
    uint8_t* sqBuff = nullptr;
    GetSqBufferAddr(&sqBuff);
    EXPECT_NE(sqBuff, nullptr);
}

TEST_F(HcclDevicePubTest, UpdatePiValByJettyId_Valid)
{
    uint32_t jettyId = 1;
    uint32_t piValue = 100;

    HcclVmResult result = UpdatePiValByJettyId(jettyId, piValue);
    EXPECT_EQ(result, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(HcclDevicePubTest, GetPiValByJettyId_AfterUpdate)
{
    uint32_t jettyId = 2;
    uint32_t piValue = 200;

    UpdatePiValByJettyId(jettyId, piValue);

    uint32_t getValue = 0;
    HcclVmResult result = GetPiValByJettyId(jettyId, &getValue);
    EXPECT_EQ(result, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(getValue, piValue);
}

TEST_F(HcclDevicePubTest, GetPiValByJettyId_NotExist)
{
    uint32_t jettyId = 9999;
    uint32_t getValue = 0;

    HcclVmResult result = GetPiValByJettyId(jettyId, &getValue);
    EXPECT_EQ(result, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(HcclDevicePubTest, UpdateKfcStatus_ZeroAddr) { EXPECT_NO_THROW(UpdateKfcStatus(0)); }

TEST_F(HcclDevicePubTest, UpdateKfcStatus_ValidAddr)
{
    constexpr uint32_t kfcBufSize = 4096;
    std::vector<uint8_t> kfcBuf(kfcBufSize, 0);
    uint64_t d2hAddr = reinterpret_cast<uint64_t>(kfcBuf.data());
    EXPECT_NO_THROW(UpdateKfcStatus(d2hAddr));
}

TEST_F(HcclDevicePubTest, RegisterSignalHandler_NoThrow) { EXPECT_NO_THROW(RegisterSignalHandler()); }

// ==================== GetSqTail / UpdateSqTail Tests ====================

TEST_F(HcclDevicePubTest, UpdateSqTail_And_GetSqTail)
{
    uint32_t sqId = 1;
    uint32_t newTail = 42;

    UpdateSqTail(sqId, newTail);
    uint32_t tail = GetSqTail(sqId);

    EXPECT_EQ(tail, newTail);
}

TEST_F(HcclDevicePubTest, GetSqTail_NotExist)
{
    uint32_t sqId = 99;
    uint32_t tail = GetSqTail(sqId);

    EXPECT_EQ(tail, 0);
}

TEST_F(HcclDevicePubTest, UpdateSqTail_Overwrite)
{
    uint32_t sqId = 2;
    UpdateSqTail(sqId, 10);
    UpdateSqTail(sqId, 20);

    uint32_t tail = GetSqTail(sqId);
    EXPECT_EQ(tail, 20);
}

// ==================== GetDeviceIdByDevAddr Tests ====================

TEST_F(HcclDevicePubTest, GetDeviceIdByDevAddr_NotFound)
{
    uint64_t devAddr = 0x1000;
    uint32_t deviceId = GetDeviceIdByDevAddr(devAddr);

    EXPECT_EQ(deviceId, 0);
}

TEST_F(HcclDevicePubTest, GetDeviceIdByDevAddr_Found)
{
    auto devRet = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
        HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{0}));
    ASSERT_TRUE(devRet.second);
    uint64_t deviceId = devRet.first.id;

    sim::runtime::PhyMemBlock phyMem{};
    phyMem.device_id = deviceId;
    snprintf(phyMem.name, sizeof(phyMem.name), "test_phy_mem");
    phyMem.size = 4096;
    phyMem.type = 0;
    phyMem.ref_count = 1;
    uint64_t phyMemId = runnerdb_test::InsertRecord<sim::runtime::PhyMemBlock>(phyMem);

    sim::runtime::VirtualMemBlock virMem{};
    virMem.start_ptr = 0x10000000;
    virMem.size = 4096;
    virMem.ctx_id = 1;
    virMem.phy_mem_id = phyMemId;
    virMem.owner_pid = 0;
    virMem.src_type = (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV;
    virMem.policy = 0;
    runnerdb_test::InsertRecord<sim::runtime::VirtualMemBlock>(virMem);

    sim::runtime::Communicator comm{};
    std::strncpy(comm.comm_id, "test_comm", sizeof(comm.comm_id) - 1);
    comm.rank_size = 8;
    comm.rank_id = 5;
    comm.device_id = deviceId;
    sim::runtime::g_cur_comm_key = runnerdb_test::InsertRecord<sim::runtime::Communicator>(comm);

    uint64_t devAddr = virMem.start_ptr + 0x100;
    uint32_t resultDeviceId = GetDeviceIdByDevAddr(devAddr);

    EXPECT_EQ(resultDeviceId, deviceId);
}

TEST_F(HcclDevicePubTest, GetDeviceIdByDevAddr_VirMemFound_ButPhyMemNotFound)
{
    auto devRet = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
        HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{0}));
    ASSERT_TRUE(devRet.second);
    uint64_t deviceId = devRet.first.id;

    sim::runtime::VirtualMemBlock virMem{};
    virMem.start_ptr = 0x20000000;
    virMem.size = 4096;
    virMem.ctx_id = 1;
    virMem.phy_mem_id = 99999;
    virMem.owner_pid = 0;
    virMem.src_type = (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV;
    virMem.policy = 0;
    runnerdb_test::InsertRecord<sim::runtime::VirtualMemBlock>(virMem);

    uint64_t devAddr = virMem.start_ptr + 0x100;
    uint32_t resultDeviceId = GetDeviceIdByDevAddr(devAddr);

    EXPECT_EQ(resultDeviceId, 0);
}

TEST_F(HcclDevicePubTest, GetDeviceIdByDevAddr_DeviceNotFound)
{
    auto devRet = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
        HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{0}));
    ASSERT_TRUE(devRet.second);
    uint64_t deviceId = devRet.first.id;

    sim::runtime::PhyMemBlock phyMem{};
    phyMem.device_id = deviceId;
    snprintf(phyMem.name, sizeof(phyMem.name), "test_phy_mem2");
    phyMem.size = 4096;
    phyMem.type = 0;
    phyMem.ref_count = 1;
    uint64_t phyMemId = runnerdb_test::InsertRecord<sim::runtime::PhyMemBlock>(phyMem);

    sim::runtime::VirtualMemBlock virMem{};
    virMem.start_ptr = 0x30000000;
    virMem.size = 4096;
    virMem.ctx_id = 1;
    virMem.phy_mem_id = phyMemId;
    virMem.owner_pid = 0;
    virMem.src_type = (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV;
    virMem.policy = 0;
    runnerdb_test::InsertRecord<sim::runtime::VirtualMemBlock>(virMem);

    uint64_t devAddr = virMem.start_ptr + 0x100;
    uint32_t resultDeviceId = GetDeviceIdByDevAddr(devAddr);

    // 基线红修复：GetDeviceIdByDevAddr 的合同是"按虚拟地址定位 VMB，返回其
    // phy_mem_id 所指 PhyMemBlock 行的 device_id"，不校验 Device 表存在性。
    // 旧用例名 DeviceNotFound 期望 0，但本用例的 phy/vir 行均有效，函数按
    // 合同返回物理属主（=deviceId），期望 0 在基线即失败。修正为合同语义；
    // "物理行缺失返回 0"的合同由上方 VirMemFound_ButPhyMemNotFound 用例覆盖。
    EXPECT_EQ(resultDeviceId, deviceId);
}

// ==================== GetDeviceIdByIpAddr Tests ====================

TEST_F(HcclDevicePubTest, GetDeviceIdByIpAddr_NotFound)
{
    uint32_t deviceId = 0;

    EXPECT_FALSE(GetDeviceIdByIpAddr("192.168.1.999", deviceId));
}

TEST_F(HcclDevicePubTest, GetDeviceIdByIpAddr_Found)
{
    auto devRet = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
        HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{0}));
    ASSERT_TRUE(devRet.second);
    uint64_t deviceId = devRet.first.id;

    sim::runtime::EndPoint ep{};
    ep.device_id = deviceId;
    snprintf(ep.ip_addr, sizeof(ep.ip_addr), "192.168.1.200");
    runnerdb_test::InsertRecord<sim::runtime::EndPoint>(ep);

    sim::runtime::Communicator comm{};
    std::strncpy(comm.comm_id, "test_comm", sizeof(comm.comm_id) - 1);
    comm.rank_size = 8;
    comm.rank_id = 7;
    comm.device_id = deviceId;
    sim::runtime::g_cur_comm_key = runnerdb_test::InsertRecord<sim::runtime::Communicator>(comm);

    uint32_t resultDeviceId = 0;

    EXPECT_TRUE(GetDeviceIdByIpAddr("192.168.1.200", resultDeviceId));
    EXPECT_EQ(resultDeviceId, deviceId);
}

TEST_F(HcclDevicePubTest, GetDeviceIdByIpAddr_DeviceNotFound)
{
    auto devRet = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
        HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{0}));
    ASSERT_TRUE(devRet.second);
    uint64_t deviceId = devRet.first.id;

    sim::runtime::EndPoint ep{};
    ep.device_id = deviceId;
    snprintf(ep.ip_addr, sizeof(ep.ip_addr), "192.168.1.201");
    runnerdb_test::InsertRecord<sim::runtime::EndPoint>(ep);

    uint32_t resultDeviceId = 0;

    EXPECT_TRUE(GetDeviceIdByIpAddr("192.168.1.201", resultDeviceId));
    EXPECT_EQ(resultDeviceId, deviceId);
}

// ==================== GetDevMapperAddrByDevAddr Tests ====================

TEST_F(HcclDevicePubTest, GetDevMapperAddrByDevAddr_NotFound)
{
    uint64_t devAddr = 0xDEADBEEF;
    uint64_t realAddr = GetDevMapperAddrByDevAddr(devAddr);

    EXPECT_EQ(realAddr, 0ULL);
}

TEST_F(HcclDevicePubTest, GetDevMapperAddrByDevAddr_FoundButShmNotExist)
{
    auto devRet = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
        HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{0}));
    ASSERT_TRUE(devRet.second);
    uint64_t deviceId = devRet.first.id;

    sim::runtime::PhyMemBlock phyMem{};
    phyMem.device_id = deviceId;
    snprintf(phyMem.name, sizeof(phyMem.name), "test_real_ptr_mem");
    phyMem.size = 4096;
    phyMem.type = 0;
    phyMem.ref_count = 1;
    uint64_t phyMemId = runnerdb_test::InsertRecord<sim::runtime::PhyMemBlock>(phyMem);

    sim::runtime::VirtualMemBlock virMem{};
    virMem.start_ptr = 0x40000000;
    virMem.dev_mapped_ptr = 0x40000000;
    virMem.size = 4096;
    virMem.ctx_id = 1;
    virMem.phy_mem_id = phyMemId;
    virMem.owner_pid = (uint64_t)getppid();
    virMem.src_type = (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV;
    virMem.is_dev_access = 0;
    virMem.policy = 0;
    runnerdb_test::InsertRecord<sim::runtime::VirtualMemBlock>(virMem);

    uint64_t devAddr = virMem.dev_mapped_ptr + 0x100;
    uint64_t realAddr = GetDevMapperAddrByDevAddr(devAddr);

    // PhyMem name "test_real_ptr_mem" has no actual shm, AcquireMemByName returns nullptr
    EXPECT_EQ(realAddr, 0ULL);
}

TEST_F(HcclDevicePubTest, GetDevMapperAddrByDevAddr_PhyMemNotFound)
{
    auto devRet = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
        HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{0}));
    ASSERT_TRUE(devRet.second);
    uint64_t deviceId = devRet.first.id;

    sim::runtime::PhyMemBlock phyMem{};
    phyMem.device_id = deviceId;
    snprintf(phyMem.name, sizeof(phyMem.name), "test_real_ptr_mem2");
    phyMem.size = 4096;
    phyMem.type = 0;
    phyMem.ref_count = 1;
    uint64_t phyMemId = runnerdb_test::InsertRecord<sim::runtime::PhyMemBlock>(phyMem);

    sim::runtime::VirtualMemBlock virMem{};
    virMem.start_ptr = 0x50000000;
    virMem.dev_mapped_ptr = 0x50000000;
    virMem.size = 4096;
    virMem.ctx_id = 1;
    virMem.phy_mem_id = phyMemId;
    virMem.owner_pid = (uint64_t)getppid();
    virMem.src_type = (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV;
    virMem.is_dev_access = 0;
    virMem.policy = 0;
    runnerdb_test::InsertRecord<sim::runtime::VirtualMemBlock>(virMem);

    // Delete the phyMem so GetById returns empty
    runnerdb_test::ClearRecords<sim::runtime::PhyMemBlock>();

    uint64_t devAddr = virMem.dev_mapped_ptr + 0x100;
    uint64_t realAddr = GetDevMapperAddrByDevAddr(devAddr);

    EXPECT_EQ(realAddr, 0ULL);
}

// ==================== ResolveTaskAddress Tests ====================
// 任务书 v3 §A：SQE 任务构造复用一次设备作用域区间查找，同时取得虚拟地址与
// 物理属主；替代"翻译后再按虚拟地址全局反查 VMB→PhyMemBlock"。以下用例锁定
// 语义（回归清单 1-4）：新旧输出一致、设备身份不串、映射设备与物理属主可不同、
// 边界/空洞/空关联/属主缺失/释放后/无效设备/溢出/后端错误全部失败关闭。

TEST_F(HcclDevicePubTest, ResolveTaskAddress_MatchesLegacyTranslationOutput)
{
    // 回归1：同设备普通映射，新旧实现输出相同 virtualAddress、physicalDeviceId
    const uint64_t deviceId = FirstDeviceId();
    ASSERT_NE(deviceId, 0U);
    SetCurDeviceKey(static_cast<uint32_t>(deviceId));

    const uint64_t phyMemId = InsertPhyOwnedBy(deviceId, "resolve_phy_same");
    ASSERT_NE(phyMemId, 0U);
    constexpr uint64_t kStart = 0x10000000ULL;
    constexpr uint64_t kMapped = 0x700000000ULL;
    ASSERT_NE(InsertDevVmb(deviceId, kStart, kMapped, 0x10000, phyMemId), 0U);

    constexpr uint64_t kAddr = kMapped + 0x400;
    // 旧路径：翻译 + 全局反查
    const uint64_t legacyVir = TransLocalAddrToVirtual(kAddr);
    const uint32_t legacyOwner = GetDeviceIdByDevAddr(legacyVir);

    auto resolved = ResolveTaskAddress(kAddr, deviceId);
    ASSERT_TRUE(resolved.ok()) << resolved.diagnostic;
    EXPECT_EQ(resolved->virtualAddress, kStart + 0x400);
    EXPECT_EQ(resolved->virtualAddress, legacyVir);
    EXPECT_EQ(resolved->physicalDeviceId, static_cast<uint32_t>(deviceId));
    EXPECT_EQ(resolved->physicalDeviceId, legacyOwner);
    SetCurDeviceKey(0);
}

TEST_F(HcclDevicePubTest, ResolveTaskAddress_SameMappedAddressDifferentDevicesSelectsOwnRow)
{
    // 回归2：两设备映射地址数值相同但记录/start_ptr 不同——(device,addr) 身份选对行
    const uint64_t devA = FirstDeviceId();
    ASSERT_NE(devA, 0U);
    sim::runtime::Device deviceB{};
    deviceB.server_id = 1;
    deviceB.physical_id = 1;
    const uint64_t devB = runnerdb_test::InsertRecord<sim::runtime::Device>(deviceB);
    ASSERT_NE(devB, 0U);

    const uint64_t phyA = InsertPhyOwnedBy(devA, "resolve_phy_a");
    const uint64_t phyB = InsertPhyOwnedBy(devB, "resolve_phy_b");
    ASSERT_NE(phyA, 0U);
    ASSERT_NE(phyB, 0U);
    constexpr uint64_t kMapped = 0x800000000ULL;
    constexpr uint64_t kStartA = 0x10000000ULL;
    constexpr uint64_t kStartB = 0x50000000ULL;
    ASSERT_NE(InsertDevVmb(devA, kStartA, kMapped, 0x10000, phyA), 0U);
    ASSERT_NE(InsertDevVmb(devB, kStartB, kMapped, 0x10000, phyB), 0U);

    auto resA = ResolveTaskAddress(kMapped + 0x100, devA);
    ASSERT_TRUE(resA.ok()) << resA.diagnostic;
    EXPECT_EQ(resA->virtualAddress, kStartA + 0x100);
    EXPECT_EQ(resA->physicalDeviceId, static_cast<uint32_t>(devA));

    auto resB = ResolveTaskAddress(kMapped + 0x100, devB);
    ASSERT_TRUE(resB.ok()) << resB.diagnostic;
    EXPECT_EQ(resB->virtualAddress, kStartB + 0x100);
    EXPECT_EQ(resB->physicalDeviceId, static_cast<uint32_t>(devB));

    // 不同进程相同数值地址不代表同一映射：两设备结果互不相同、不串设备
    EXPECT_NE(resA->virtualAddress, resB->virtualAddress);
    EXPECT_NE(resA->physicalDeviceId, resB->physicalDeviceId);
}

TEST_F(HcclDevicePubTest, ResolveTaskAddress_PhysicalOwnerMayDifferFromMappingDevice)
{
    // 回归3：映射设备 A、物理属主 B（aclrtMapMem 允许的关联模型）——必须返回 B，
    // 且 VMB 定位仍按 A；PhyMemBlock 主键读取保留（不能用 A 替换）
    const uint64_t devA = FirstDeviceId();
    ASSERT_NE(devA, 0U);
    sim::runtime::Device deviceB{};
    deviceB.server_id = 1;
    deviceB.physical_id = 1;
    const uint64_t devB = runnerdb_test::InsertRecord<sim::runtime::Device>(deviceB);
    ASSERT_NE(devB, 0U);

    const uint64_t phyOwnedByB = InsertPhyOwnedBy(devB, "resolve_phy_cross");
    ASSERT_NE(phyOwnedByB, 0U);
    constexpr uint64_t kMapped = 0x900000000ULL;
    constexpr uint64_t kStart = 0x20000000ULL;
    ASSERT_NE(InsertDevVmb(devA, kStart, kMapped, 0x10000, phyOwnedByB), 0U);

    auto resolved = ResolveTaskAddress(kMapped + 0x80, devA);
    ASSERT_TRUE(resolved.ok()) << resolved.diagnostic;
    EXPECT_EQ(resolved->virtualAddress, kStart + 0x80);
    EXPECT_EQ(resolved->physicalDeviceId, static_cast<uint32_t>(devB));

    // 以 B 为映射设备查同一数值地址不命中该行（身份是 (device,addr)）
    auto wrongScope = ResolveTaskAddress(kMapped + 0x80, devB);
    EXPECT_FALSE(wrongScope.ok());
    EXPECT_EQ(wrongScope.code, HcclSim::Storage::DbCode::NOT_FOUND);
    EXPECT_FALSE(wrongScope.value.has_value());
}

TEST_F(HcclDevicePubTest, ResolveTaskAddress_IntervalBoundariesAndHoles)
{
    // 回归4：起点、末字节命中；end 边界与空洞确定未命中
    const uint64_t deviceId = FirstDeviceId();
    ASSERT_NE(deviceId, 0U);
    const uint64_t phy1 = InsertPhyOwnedBy(deviceId, "resolve_phy_b1");
    const uint64_t phy2 = InsertPhyOwnedBy(deviceId, "resolve_phy_b2");
    ASSERT_NE(phy1, 0U);
    ASSERT_NE(phy2, 0U);
    constexpr uint64_t kMapped1 = 0x1000ULL;
    constexpr uint64_t kStart1 = 0x11000000ULL;
    constexpr uint64_t kMapped2 = 0x4000ULL;
    constexpr uint64_t kStart2 = 0x12000000ULL;
    ASSERT_NE(InsertDevVmb(deviceId, kStart1, kMapped1, 0x1000, phy1), 0U);
    ASSERT_NE(InsertDevVmb(deviceId, kStart2, kMapped2, 0x1000, phy2), 0U);

    auto first = ResolveTaskAddress(kMapped1, deviceId); // 起点
    ASSERT_TRUE(first.ok()) << first.diagnostic;
    EXPECT_EQ(first->virtualAddress, kStart1);
    auto lastByte = ResolveTaskAddress(kMapped1 + 0xFFF, deviceId); // 末字节
    ASSERT_TRUE(lastByte.ok()) << lastByte.diagnostic;
    EXPECT_EQ(lastByte->virtualAddress, kStart1 + 0xFFF);
    auto endBoundary = ResolveTaskAddress(kMapped1 + 0x1000, deviceId); // end 边界（空洞）
    EXPECT_FALSE(endBoundary.ok());
    EXPECT_EQ(endBoundary.code, HcclSim::Storage::DbCode::NOT_FOUND);
    auto below = ResolveTaskAddress(kMapped1 - 1, deviceId);
    EXPECT_FALSE(below.ok());
    EXPECT_EQ(below.code, HcclSim::Storage::DbCode::NOT_FOUND);
    auto second = ResolveTaskAddress(kMapped2, deviceId);
    ASSERT_TRUE(second.ok()) << second.diagnostic;
    EXPECT_EQ(second->virtualAddress, kStart2);
}

TEST_F(HcclDevicePubTest, ResolveTaskAddress_MissingPhysicalAssociationFailsClosed)
{
    // 回归4：未映射(phy_mem_id=0)、物理行缺失、释放后查询——明确错误，
    // 不生成零地址/零属主的"有效"结果；仅需地址的旧翻译函数不受影响
    const uint64_t deviceId = FirstDeviceId();
    ASSERT_NE(deviceId, 0U);
    SetCurDeviceKey(static_cast<uint32_t>(deviceId));

    // 未映射物理块（Reserve 后未 MapMem 的形状）
    constexpr uint64_t kUnmapped = 0x20000000ULL;
    ASSERT_NE(InsertDevVmb(deviceId, 0x13000000ULL, kUnmapped, 0x10000, 0), 0U);
    auto unassociated = ResolveTaskAddress(kUnmapped + 0x10, deviceId);
    EXPECT_FALSE(unassociated.ok());
    EXPECT_EQ(unassociated.code, HcclSim::Storage::DbCode::NOT_FOUND);
    EXPECT_FALSE(unassociated.value.has_value());
    // 旧翻译（仅需地址）不因物理关联缺失失败，也不额外读取 PhyMemBlock
    EXPECT_EQ(TransLocalAddrToVirtual(kUnmapped + 0x10), 0x13000000ULL + 0x10);
    SetCurDeviceKey(0);

    // 物理行缺失（悬空 phy_mem_id）
    constexpr uint64_t kDangling = 0x21000000ULL;
    ASSERT_NE(InsertDevVmb(deviceId, 0x14000000ULL, kDangling, 0x10000, 99999), 0U);
    auto dangling = ResolveTaskAddress(kDangling + 0x10, deviceId);
    EXPECT_FALSE(dangling.ok());
    EXPECT_EQ(dangling.code, HcclSim::Storage::DbCode::NOT_FOUND);

    // 释放后查询：VMB 行删除 → NOT_FOUND
    const uint64_t phyMemId = InsertPhyOwnedBy(deviceId, "resolve_phy_released");
    ASSERT_NE(phyMemId, 0U);
    constexpr uint64_t kReleased = 0x22000000ULL;
    const uint64_t vmbId = InsertDevVmb(deviceId, 0x15000000ULL, kReleased, 0x10000, phyMemId);
    ASSERT_NE(vmbId, 0U);
    ASSERT_TRUE(runnerdb_test::DeleteRecord<sim::runtime::VirtualMemBlock>(vmbId));
    auto released = ResolveTaskAddress(kReleased + 0x10, deviceId);
    EXPECT_FALSE(released.ok());
    EXPECT_EQ(released.code, HcclSim::Storage::DbCode::NOT_FOUND);
}

TEST_F(HcclDevicePubTest, ResolveTaskAddress_InvalidMappingDeviceRejected)
{
    // 回归4：mappingDeviceId=0（"无当前设备"哨兵）——即使库中存在 device_id=0
    // 的脏行覆盖该地址，也必须显式拒绝，不得返回零属主有效任务
    constexpr uint64_t kAddr = 0x23000000ULL;
    ASSERT_NE(InsertDevVmb(0, 0x16000000ULL, kAddr, 0x10000, 0), 0U);
    auto resolved = ResolveTaskAddress(kAddr + 0x10, 0);
    EXPECT_FALSE(resolved.ok());
    EXPECT_EQ(resolved.code, HcclSim::Storage::DbCode::INVALID_ARGUMENT);
    EXPECT_FALSE(resolved.value.has_value());
}

TEST_F(HcclDevicePubTest, ResolveTaskAddress_VirtualAddressOverflowRejected)
{
    // 回归4：start_ptr + diff 上溢——显式拒绝，不返回回绕地址
    const uint64_t deviceId = FirstDeviceId();
    ASSERT_NE(deviceId, 0U);
    const uint64_t phyMemId = InsertPhyOwnedBy(deviceId, "resolve_phy_ovf");
    ASSERT_NE(phyMemId, 0U);
    constexpr uint64_t kMapped = 0x24000000ULL;
    ASSERT_NE(InsertDevVmb(deviceId, UINT64_MAX - 8, kMapped, 0x1000, phyMemId), 0U);
    auto resolved = ResolveTaskAddress(kMapped + 16, deviceId);
    EXPECT_FALSE(resolved.ok());
    EXPECT_EQ(resolved.code, HcclSim::Storage::DbCode::INVALID_ARGUMENT);
    EXPECT_FALSE(resolved.value.has_value());
}

TEST_F(HcclDevicePubTest, ResolveTaskAddress_BackendErrorPropagates)
{
    // 回归4：注入后端错误（会话关闭）——错误码原样传播，不冒充 OK/零值结果
    const uint64_t deviceId = FirstDeviceId();
    ASSERT_NE(deviceId, 0U);
    const uint64_t phyMemId = InsertPhyOwnedBy(deviceId, "resolve_phy_err");
    ASSERT_NE(phyMemId, 0U);
    constexpr uint64_t kMapped = 0x25000000ULL;
    ASSERT_NE(InsertDevVmb(deviceId, 0x17000000ULL, kMapped, 0x10000, phyMemId), 0U);

    const auto closed = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    ASSERT_TRUE(closed.ok()) << closed.diagnostic;
    auto resolved = ResolveTaskAddress(kMapped + 0x10, deviceId);
    EXPECT_FALSE(resolved.ok());
    EXPECT_FALSE(resolved.value.has_value()) << "失败结果不得携带值载荷";
    EXPECT_FALSE(resolved.diagnostic.empty());

    // 恢复会话供 TearDown 清理
    ASSERT_TRUE(runnerdb_test::ResetTestSession(kTestDbPath, kTestDbPath));
}
