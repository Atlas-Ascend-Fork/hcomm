/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <tuple>
#include <vector>

#include "hccl_device_pub.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/table_access.h"
#include "sqlite/simulation_storage_test_helper.h"

// 查询次数观察点用的内存语义引擎（backend conformance 的 Fake 后端）
#include "../../storage/conformance_memory_engine.h"

// 将实现纳入当前翻译单元，以直接验证匿名namespace中的纯转换辅助函数。
#include "../../../src/device_arm/level1/data_op_device_stub.cc"

using namespace HcclSim;

namespace {

const std::string kTestDbPath = "/tmp/test_data_op_device_stub.db";
const char* kTestCommName = "test_comm";
// 本端/远端设备虚拟编址基址
constexpr uint64_t kLocalVirBase = 0x10000000ULL;
constexpr uint64_t kRemoteVirBase = 0x20000000ULL;
// 设备地址范围 [0, kDevAddrRange) 内可被翻译
constexpr uint64_t kDevAddrRange = 0x100000ULL;

uint64_t g_localDevId = 0;
uint64_t g_remoteDevId = 0;
uint64_t g_commKey = 0;

void CleanUpDb()
{
    std::remove(kTestDbPath.c_str());
    std::remove((kTestDbPath + "-wal").c_str());
    std::remove((kTestDbPath + "-shm").c_str());
}

void SetupTestData()
{
    CleanUpDb();
    // 迁移前 SetDbPath + ClearAll 的等价语义：以本用例数据库路径重建进程存储会话。
    (void)runnerdb_test::ResetTestSession(kTestDbPath, kTestDbPath);

    sim::runtime::Device devLocal{};
    devLocal.server_id = 1;
    devLocal.logic_id = 1;
    devLocal.physical_id = 1;
    g_localDevId = runnerdb_test::InsertRecord<sim::runtime::Device>(devLocal);

    sim::runtime::Device devRemote{};
    devRemote.server_id = 1;
    devRemote.logic_id = 2;
    devRemote.physical_id = 2;
    g_remoteDevId = runnerdb_test::InsertRecord<sim::runtime::Device>(devRemote);

    SetCurDeviceKey(static_cast<uint32_t>(g_localDevId));

    sim::runtime::VirtualMemBlock localMem{};
    localMem.start_ptr = kLocalVirBase;
    localMem.dev_mapped_ptr = 0;
    localMem.size = kDevAddrRange;
    localMem.ctx_id = 1;
    localMem.device_id = g_localDevId;
    localMem.src_type = static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV);
    runnerdb_test::InsertRecord<sim::runtime::VirtualMemBlock>(localMem);

    sim::runtime::VirtualMemBlock remoteMem{};
    remoteMem.start_ptr = kRemoteVirBase;
    remoteMem.dev_mapped_ptr = 0;
    remoteMem.size = kDevAddrRange;
    remoteMem.ctx_id = 1;
    remoteMem.device_id = g_remoteDevId;
    remoteMem.src_type = static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV);
    runnerdb_test::InsertRecord<sim::runtime::VirtualMemBlock>(remoteMem);

    sim::runtime::Communicator self{};
    std::strncpy(self.comm_id, kTestCommName, sizeof(self.comm_id) - 1);
    self.rank_size = 2;
    self.rank_id = 1;
    self.device_id = g_localDevId;
    g_commKey = runnerdb_test::InsertRecord<sim::runtime::Communicator>(self);

    sim::runtime::Communicator peer{};
    std::strncpy(peer.comm_id, kTestCommName, sizeof(peer.comm_id) - 1);
    peer.rank_size = 2;
    peer.rank_id = 2;
    peer.device_id = g_remoteDevId;
    runnerdb_test::InsertRecord<sim::runtime::Communicator>(peer);
}

} // namespace

class DataOpDeviceStubTest : public testing::Test {
protected:
    void SetUp() override { SetupTestData(); }

    void TearDown() override
    {
        runnerdb_test::ClearRecords<sim::runtime::Communicator>();
        runnerdb_test::ClearRecords<sim::runtime::VirtualMemBlock>();
        runnerdb_test::ClearRecords<sim::runtime::Device>();
        CleanUpDb();
    }
};

namespace {

std::tuple<sim::runtime::HcclThread, sim::runtime::HcclChannel, uint32_t, uint32_t> MakeContext()
{
    sim::runtime::HcclThread thread{};
    thread.commId = g_commKey;
    thread.streamId = 34;
    sim::runtime::HcclChannel channel{};
    channel.id = 56;
    return {thread, channel, 1U, 2U};
}

TEST_F(DataOpDeviceStubTest, BatchDescriptorAbiIsStable)
{
    EXPECT_EQ(sizeof(HcommBatchTransferDesc), 64U);
    EXPECT_EQ(alignof(HcommBatchTransferDesc), 8U);
    EXPECT_EQ(offsetof(HcommBatchTransferDesc, transferInfo), 8U);
    EXPECT_EQ(sizeof(HcclHcommBatchTransferDesc), 64U);
    EXPECT_EQ(alignof(HcclHcommBatchTransferDesc), 8U);
    EXPECT_EQ(offsetof(HcclHcommBatchTransferDesc, transferInfo), 8U);
}

TEST_F(DataOpDeviceStubTest, CopyTaskMapsDirectionAndAddresses)
{
    const auto [thread, channel, rankId, remoteRankId] = MakeContext();
    (void)channel;
    const void* src = reinterpret_cast<const void*>(0x1000ULL);
    const void* dst = reinterpret_cast<const void*>(0x2000ULL);
    HcclTaskMetaData task = MakeCopyTask(thread, rankId, rankId, src, remoteRankId, dst, 128);
    EXPECT_EQ(task.taskType, HccLTaskMetaType::MEM_CPY);
    EXPECT_EQ(task.commId, g_commKey);
    EXPECT_EQ(task.streamId, 34U);
    EXPECT_EQ(task.jettyId, std::numeric_limits<uint32_t>::max());
    // src为本端(rank1)，dst为远端(rank2)
    EXPECT_EQ(task.taskData.transMem.srcDeviceId, 1U);
    EXPECT_EQ(task.taskData.transMem.dstDeviceId, 2U);
    EXPECT_EQ(task.taskData.transMem.srcOffset, kLocalVirBase + 0x1000U);
    EXPECT_EQ(task.taskData.transMem.dstOffset, kRemoteVirBase + 0x2000U);
    EXPECT_EQ(task.taskData.transMem.len, 128U);
}

TEST_F(DataOpDeviceStubTest, ReduceTaskKeepsElementCount)
{
    const auto [thread, channel, rankId, remoteRankId] = MakeContext();
    (void)channel;
    (void)remoteRankId;
    HcclTaskMetaData task = MakeReduceTask(
        thread, rankId, 2, reinterpret_cast<const void*>(0x3000ULL), 1, reinterpret_cast<const void*>(0x4000ULL), 17, 4,
        1);
    EXPECT_EQ(task.taskType, HccLTaskMetaType::REDUCE);
    // src为远端(rank2)，dst为本端(rank1)
    EXPECT_EQ(task.taskData.reduce.srcDeviceId, 2U);
    EXPECT_EQ(task.taskData.reduce.dstDeviceId, 1U);
    EXPECT_EQ(task.taskData.reduce.srcOffset, kRemoteVirBase + 0x3000U);
    EXPECT_EQ(task.taskData.reduce.dstOffset, kLocalVirBase + 0x4000U);
    EXPECT_EQ(task.taskData.reduce.dataCount, 17U);
    EXPECT_EQ(task.taskData.reduce.dataType, 4U);
    EXPECT_EQ(task.taskData.reduce.reduceOp, 1U);
}

TEST_F(DataOpDeviceStubTest, A5ReduceValidationRejectsInvalidAndOverflow)
{
    uint32_t typeSize = 0;
    EXPECT_EQ(ValidateA5Reduce(4, 1, 16, typeSize), HCCL_SUCCESS);
    EXPECT_EQ(typeSize, 4U);
    EXPECT_EQ(ValidateA5Reduce(13, 0, 1, typeSize), HCCL_E_PARA);
    EXPECT_EQ(ValidateA5Reduce(4, 255, 1, typeSize), HCCL_E_PARA);
    EXPECT_EQ(ValidateA5Reduce(12, 0, std::numeric_limits<uint64_t>::max(), typeSize), HCCL_E_PARA);
}

TEST_F(DataOpDeviceStubTest, TimeoutValidationTruncatesAndRejectsInvalidValues)
{
    uint32_t normalized = 0;
    EXPECT_EQ(NormalizeTimeout(12.75F, normalized), HCCL_SUCCESS);
    EXPECT_EQ(normalized, 12U);
    EXPECT_EQ(NormalizeTimeout(-1.0F, normalized), HCCL_E_PARA);
    EXPECT_EQ(NormalizeTimeout(std::numeric_limits<float>::quiet_NaN(), normalized), HCCL_E_PARA);
    EXPECT_EQ(NormalizeTimeout(4294967296.0F, normalized), HCCL_E_PARA);
}

TEST_F(DataOpDeviceStubTest, BatchWriteAndReadMapOppositeDirections)
{
    const auto [thread, channel, rankId, remoteRankId] = MakeContext();
    std::vector<HcclTaskMetaData> tasks;
    HcommBatchTransferDesc write{};
    write.transType = TRANSFER_TYPE_WRITE;
    write.transferInfo.write.src = reinterpret_cast<void*>(0x1000ULL);
    write.transferInfo.write.dst = reinterpret_cast<void*>(0x2000ULL);
    write.transferInfo.write.len = 64;
    EXPECT_EQ(ExpandBatchDescriptor(write, 0, thread, channel, rankId, remoteRankId, tasks), HCCL_SUCCESS);
    // write: src为本端(rank1)，dst为远端(rank2)
    EXPECT_EQ(tasks[0].taskData.transMem.srcDeviceId, 1U);
    EXPECT_EQ(tasks[0].taskData.transMem.dstDeviceId, 2U);
    EXPECT_EQ(tasks[0].taskData.transMem.srcOffset, kLocalVirBase + 0x1000U);
    EXPECT_EQ(tasks[0].taskData.transMem.dstOffset, kRemoteVirBase + 0x2000U);

    HcommBatchTransferDesc read{};
    read.transType = TRANSFER_TYPE_READ;
    read.transferInfo.read.src = reinterpret_cast<void*>(0x3000ULL);
    read.transferInfo.read.dst = reinterpret_cast<void*>(0x4000ULL);
    read.transferInfo.read.len = 32;
    EXPECT_EQ(ExpandBatchDescriptor(read, 1, thread, channel, rankId, remoteRankId, tasks), HCCL_SUCCESS);
    // read: src为远端(rank2)，dst为本端(rank1)
    EXPECT_EQ(tasks[1].taskData.transMem.srcDeviceId, 2U);
    EXPECT_EQ(tasks[1].taskData.transMem.dstDeviceId, 1U);
    EXPECT_EQ(tasks[1].taskData.transMem.srcOffset, kRemoteVirBase + 0x3000U);
    EXPECT_EQ(tasks[1].taskData.transMem.dstOffset, kLocalVirBase + 0x4000U);
}

TEST_F(DataOpDeviceStubTest, UnsupportedAndInvalidBatchTypesDoNotAppendTasks)
{
    const auto [thread, channel, rankId, remoteRankId] = MakeContext();
    std::vector<HcclTaskMetaData> tasks;
    HcommBatchTransferDesc desc{};
    desc.transType = TRANSFER_TYPE_NOTIFY_WAIT;
    EXPECT_EQ(ExpandBatchDescriptor(desc, 0, thread, channel, rankId, remoteRankId, tasks), HCCL_E_NOT_SUPPORT);
    desc.transType = TRANSFER_TYPE_NOTIFY_WAIT_DEFAULT;
    EXPECT_EQ(ExpandBatchDescriptor(desc, 1, thread, channel, rankId, remoteRankId, tasks), HCCL_E_NOT_SUPPORT);
    desc.transType = 99;
    EXPECT_EQ(ExpandBatchDescriptor(desc, 2, thread, channel, rankId, remoteRankId, tasks), HCCL_E_PARA);
    EXPECT_TRUE(tasks.empty());
}

TEST_F(DataOpDeviceStubTest, PublicParameterErrorsAreStableAcrossRepeatedCalls)
{
    EXPECT_EQ(HcommLocalCopyOnThread(1, nullptr, reinterpret_cast<void*>(1), 0), HCCL_E_PARA);
    EXPECT_EQ(HcommLocalReduceOnThread(1, reinterpret_cast<void*>(1), nullptr, 0, 4, 0), HCCL_E_PARA);
    EXPECT_EQ(HcommBatchTransferOnThread(1, 1, nullptr, 1), HCCL_E_PTR);
    HcommBatchTransferDesc desc{};
    EXPECT_EQ(HcommBatchTransferOnThread(1, 1, &desc, 0), HCCL_E_PARA);
    EXPECT_EQ(HcommBatchTransferOnThread(1, 1, &desc, 0), HCCL_E_PARA);
}

TEST_F(DataOpDeviceStubTest, BatchModeBoundariesAllowRepeatedCalls)
{
    EXPECT_EQ(HcommBatchModeStart("repeat"), HCCL_SUCCESS);
    EXPECT_EQ(HcommBatchModeStart("repeat"), HCCL_SUCCESS);
    EXPECT_EQ(HcommBatchModeEnd("repeat"), HCCL_SUCCESS);
    EXPECT_EQ(HcommBatchModeEnd("repeat"), HCCL_SUCCESS);
}

} // namespace
