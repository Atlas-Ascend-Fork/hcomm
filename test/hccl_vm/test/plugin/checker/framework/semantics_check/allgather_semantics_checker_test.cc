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
#include <map>
#include <vector>

#include "allgather_semantics_checker.h"
#include "check_utils.h"

namespace HcclSim {
class AllGatherSemanticsCheckerTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // Helper function to create valid AllGather semantics
    void CreateValidAllGatherSemantics(PhysicalMemorySemantics& memorySemantics, u32 rankSize, u64 dataSize)
    {
        for (DeviceId deviceId = 0; deviceId < rankSize; deviceId++) {
            BufferSemanticMap deviceSemantics;
            BufferSemanticMap outputSemantics;

            u64 totalSize = 0;
            for (DeviceId srcDeviceId = 0; srcDeviceId < rankSize; srcDeviceId++) {
                BufferSemantic bufSem(totalSize, dataSize);
                bufSem.srcBufs.insert(SrcBufDes(srcDeviceId, BufferType::INPUT, 0));
                outputSemantics.emplace(bufSem.startAddr, bufSem);
                totalSize += dataSize;
            }

            deviceSemantics = outputSemantics;
            memorySemantics[deviceId] = deviceSemantics;
        }
    }
};

// Test normal case: valid AllGather semantics with 2 ranks
TEST_F(AllGatherSemanticsCheckerTest, ValidSemantics_TwoRanks)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    CreateValidAllGatherSemantics(memorySemantics, rankSize, dataSize);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherSemantics(memorySemantics, {}, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: valid AllGather semantics with 4 ranks
TEST_F(AllGatherSemanticsCheckerTest, ValidSemantics_FourRanks)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 4;
    u64 dataSize = 2048;

    CreateValidAllGatherSemantics(memorySemantics, rankSize, dataSize);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherSemantics(memorySemantics, {}, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: valid AllGather semantics with 8 ranks
TEST_F(AllGatherSemanticsCheckerTest, ValidSemantics_EightRanks)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 8;
    u64 dataSize = 4096;

    CreateValidAllGatherSemantics(memorySemantics, rankSize, dataSize);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherSemantics(memorySemantics, {}, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: single rank
TEST_F(AllGatherSemanticsCheckerTest, ValidSemantics_SingleRank)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 1;
    u64 dataSize = 1024;

    CreateValidAllGatherSemantics(memorySemantics, rankSize, dataSize);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherSemantics(memorySemantics, {}, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: zero data size
TEST_F(AllGatherSemanticsCheckerTest, ValidSemantics_ZeroDataSize)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 0;

    CreateValidAllGatherSemantics(memorySemantics, rankSize, dataSize);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherSemantics(memorySemantics, {}, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test abnormal case: missing rank
TEST_F(AllGatherSemanticsCheckerTest, Abnormal_MissingRank)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    // Only add rank 0, missing rank 1
    BufferSemanticMap deviceSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem0(0, dataSize);
    bufSem0.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem0.startAddr, bufSem0);

    BufferSemantic bufSem1(dataSize, dataSize);
    bufSem1.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem1.startAddr, bufSem1);

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherSemantics(memorySemantics, {}, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong start addr
TEST_F(AllGatherSemanticsCheckerTest, Abnormal_WrongStartAddr)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    // Create semantics with wrong start addr
    BufferSemanticMap deviceSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem0(100, dataSize); // Wrong start addr
    bufSem0.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem0.startAddr, bufSem0);

    BufferSemantic bufSem1(dataSize + 100, dataSize);
    bufSem1.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem1.startAddr, bufSem1);

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;
    memorySemantics[1] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherSemantics(memorySemantics, {}, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: multiple source buffers (should be single)
TEST_F(AllGatherSemanticsCheckerTest, Abnormal_MultipleSourceBuffers)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    BufferSemanticMap deviceSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem0(0, dataSize);
    bufSem0.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    bufSem0.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0)); // Extra source
    outputSemantics.emplace(bufSem0.startAddr, bufSem0);

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;
    memorySemantics[1] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherSemantics(memorySemantics, {}, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong source rank
TEST_F(AllGatherSemanticsCheckerTest, Abnormal_WrongSourceRank)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    BufferSemanticMap deviceSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem0(0, dataSize);
    bufSem0.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0)); // Wrong source rank
    outputSemantics.emplace(bufSem0.startAddr, bufSem0);

    BufferSemantic bufSem1(dataSize, dataSize);
    bufSem1.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0)); // Wrong source rank
    outputSemantics.emplace(bufSem1.startAddr, bufSem1);

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;
    memorySemantics[1] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherSemantics(memorySemantics, {}, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong buffer type (not INPUT)
TEST_F(AllGatherSemanticsCheckerTest, Abnormal_WrongBufferType)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    BufferSemanticMap deviceSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem0(0, dataSize);
    bufSem0.srcBufs.insert(SrcBufDes(0, BufferType::OUTPUT, 0)); // Wrong buffer type
    outputSemantics.emplace(bufSem0.startAddr, bufSem0);

    BufferSemantic bufSem1(dataSize, dataSize);
    bufSem1.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem1.startAddr, bufSem1);

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;
    memorySemantics[1] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherSemantics(memorySemantics, {}, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong source addr
TEST_F(AllGatherSemanticsCheckerTest, Abnormal_WrongSourceAddr)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    BufferSemanticMap deviceSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem0(0, dataSize);
    bufSem0.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 100)); // Wrong source addr
    outputSemantics.emplace(bufSem0.startAddr, bufSem0);

    BufferSemantic bufSem1(dataSize, dataSize);
    bufSem1.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem1.startAddr, bufSem1);

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;
    memorySemantics[1] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherSemantics(memorySemantics, {}, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: incomplete total size
TEST_F(AllGatherSemanticsCheckerTest, Abnormal_IncompleteTotalSize)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    BufferSemanticMap deviceSemantics;
    BufferSemanticMap outputSemantics;

    // Only add one buffer, missing the second
    BufferSemantic bufSem0(0, dataSize);
    bufSem0.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem0.startAddr, bufSem0);

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;
    memorySemantics[1] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherSemantics(memorySemantics, {}, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test boundary case: large data size
TEST_F(AllGatherSemanticsCheckerTest, ValidSemantics_LargeDataSize)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024 * 1024 * 1024; // 1GB

    CreateValidAllGatherSemantics(memorySemantics, rankSize, dataSize);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherSemantics(memorySemantics, {}, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: large number of ranks
TEST_F(AllGatherSemanticsCheckerTest, ValidSemantics_LargeRankSize)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 128;
    u64 dataSize = 1024;

    CreateValidAllGatherSemantics(memorySemantics, rankSize, dataSize);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherSemantics(memorySemantics, {}, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}
} // namespace HcclSim
