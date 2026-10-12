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
#include <cstring>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "store_sim_store_pub.h"
#include "store_sim_memory_manager.h"
#include "operation_data/operation_data_ops.h"
#include "storage/internal/process_storage_context.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/storage_session.h"
#include "sqlite/simulation_storage_test_helper.h"

using namespace HcclSim;

namespace {
constexpr const char* TEST_MEM_NAME = "/HcclShmPubUtMem";
constexpr size_t TEST_MEM_SIZE = 4096;
constexpr uint64_t TEST_DEV_BASE = 0x10000000;
constexpr uint64_t TEST_DEV_OFFSET = 128;

void CleanupTestMem()
{
    sim::MemoryManager::GetInstance().FreeMemByName(TEST_MEM_NAME);
    shm_unlink(TEST_MEM_NAME);
}

std::vector<uint8_t> ToBlob(const HcclTaskMetaData& task)
{
    const auto* begin = reinterpret_cast<const uint8_t*>(&task);
    return std::vector<uint8_t>(begin, begin + sizeof(HcclTaskMetaData));
}
} // namespace

class HcclShmPubTest : public testing::Test {
protected:
    void SetUp() override
    {
        // 迁移前单库 ClearAll() 的等价语义：重建干净会话。
        (void)runnerdb_test::ResetTestSession("/tmp/test_hccl_shm_pub.db", "/tmp/test_hccl_shm_pub_op.db");
        CleanupTestMem();
    }

    void TearDown() override
    {
        CleanupTestMem();
        (void)runnerdb_test::ResetTestSession("/tmp/test_hccl_shm_pub.db", "/tmp/test_hccl_shm_pub_op.db");
    }
};

TEST_F(HcclShmPubTest, GetAddrByOffset_ReturnsErrorWhenOutputAlreadyHoldsPointer)
{
    VmUniquePtr addrPtr(reinterpret_cast<void*>(0x1), VmPtrReleaser{sim::runtime::PhyMemBlock{}});
    EXPECT_EQ(GetAddrByOffset(TEST_DEV_BASE, addrPtr), HcclVmResult::HCCL_SIM_E_PARA);
}

TEST_F(HcclShmPubTest, GetAddrByOffset_ReturnsErrorWhenVirtualMemMissing)
{
    VmUniquePtr addrPtr(nullptr);
    EXPECT_EQ(GetAddrByOffset(TEST_DEV_BASE, addrPtr), HcclVmResult::HCCL_SIM_E_PTR);
    EXPECT_EQ(addrPtr.get(), nullptr);
}

TEST_F(HcclShmPubTest, GetAddrByOffset_MapsRunnerDbVirtualAddressToSharedMemory)
{
    void* shm = sim::MemoryManager::GetInstance().AllocMemByName(TEST_MEM_NAME, TEST_MEM_SIZE);
    ASSERT_NE(shm, nullptr);
    static_cast<uint8_t*>(shm)[TEST_DEV_OFFSET] = 0x5a;

    sim::runtime::PhyMemBlock phyMem{};
    phyMem.device_id = 0;
    std::strncpy(phyMem.name, TEST_MEM_NAME, sizeof(phyMem.name) - 1);
    phyMem.size = TEST_MEM_SIZE;
    phyMem.type = 0;
    phyMem.ref_count = 1;
    uint64_t phyMemId = runnerdb_test::InsertRecord(phyMem);
    ASSERT_NE(phyMemId, 0U);

    sim::runtime::VirtualMemBlock virMem{};
    virMem.start_ptr = TEST_DEV_BASE;
    virMem.size = TEST_MEM_SIZE;
    virMem.phy_mem_id = phyMemId;
    virMem.owner_pid = static_cast<uint64_t>(getpid());
    virMem.src_type = static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV);
    uint64_t virMemId = runnerdb_test::InsertRecord(virMem);
    ASSERT_NE(virMemId, 0U);

    VmUniquePtr addrPtr(nullptr);
    EXPECT_EQ(GetAddrByOffset(TEST_DEV_BASE + TEST_DEV_OFFSET, addrPtr), HcclVmResult::HCCL_SIM_SUCCESS);
    ASSERT_NE(addrPtr.get(), nullptr);
    EXPECT_EQ(static_cast<uint8_t*>(addrPtr.get())[0], 0x5a);
}

TEST_F(HcclShmPubTest, InsertTaskToCollection_ReturnsErrorForNullTask)
{
    uint32_t index = 0;
    EXPECT_EQ(InsertTaskToCollection(nullptr, &index), HcclVmResult::HCCL_SIM_E_PARA);
}

TEST_F(HcclShmPubTest, InsertTaskToCollection_ReturnsErrorForNullIndex)
{
    HcclTaskMetaData task{};
    EXPECT_EQ(InsertTaskToCollection(&task, nullptr), HcclVmResult::HCCL_SIM_E_PARA);
}

TEST_F(HcclShmPubTest, InsertTaskToCollection_InsertsSerializedTaskIntoOpDb)
{
    ASSERT_EQ(sim::operation::EnsureProcessStorageSession(), 0);

    HcclTaskMetaData task{};
    task.deviceId = 7;
    task.streamId = 3;
    task.taskType = HccLTaskMetaType::SYNC_STREAM;
    task.taskData.syncStreamTask.syncIdx = 1;

    uint32_t index = 0;
    EXPECT_EQ(InsertTaskToCollection(&task, &index), HcclVmResult::HCCL_SIM_SUCCESS);

    sim::operation::OpTaskTab expected{};
    expected.optaskMeta = ToBlob(task);
    EXPECT_EQ(expected.optaskMeta.size(), sizeof(HcclTaskMetaData));
}

TEST_F(HcclShmPubTest, InsertTaskToCollection_RejectsNonSyncTaskWithoutCommunicator)
{
    // 迁移前 sim::InitOpDataDb() 的建库语义已由夹具 ResetTestSession 覆盖（惰性建表）。

    HcclTaskMetaData task{};
    task.deviceId = 0;
    task.streamId = 3;
    task.taskType = HccLTaskMetaType::MEM_CPY;

    uint32_t index = 0;
    EXPECT_EQ(InsertTaskToCollection(&task, &index), HcclVmResult::HCCL_SIM_E_PARA);
}
