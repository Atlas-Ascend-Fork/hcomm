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
#include <set>
#include <vector>

#include "check_utils.h"
#include "send_recv_semantics_checker.h"

namespace HcclSim {
class SendRecvSemanticsCheckerTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // Helper function to create valid Send/Recv semantics
    void
    CreateValidSendRecvSemantics(PhysicalMemorySemantics& memorySemantics, u64 dataSize, RankId srcRank, RankId dstRank)
    {
        // Only destination rank has output with data from source rank
        BufferSemanticMap dstMemSemantics;
        BufferSemanticMap outputSemantics;

        BufferSemantic bufSem(0, dataSize);
        bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, 0));
        outputSemantics.emplace(bufSem.startAddr, bufSem);

        dstMemSemantics = outputSemantics;
        memorySemantics[dstRank] = dstMemSemantics;

        // Source rank has empty semantics
        BufferSemanticMap srcMemSemantics;
        memorySemantics[srcRank] = srcMemSemantics;
    }
};

// Test normal case: valid Send/Recv semantics
TEST_F(SendRecvSemanticsCheckerTest, ValidSemantics_Basic)
{
    PhysicalMemorySemantics memorySemantics;
    u64 dataSize = 1024;
    RankId srcRank = 0;
    RankId dstRank = 1;

    CreateValidSendRecvSemantics(memorySemantics, dataSize, srcRank, dstRank);

    std::vector<DeviceId> rankToDevice(memorySemantics.size());
    for (DeviceId i = 0; i < memorySemantics.size(); i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckSendRecvSemantics(
        memorySemantics, {}, dataSize, rankToDevice[srcRank], rankToDevice[dstRank], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: reversed ranks
TEST_F(SendRecvSemanticsCheckerTest, ValidSemantics_ReversedRanks)
{
    PhysicalMemorySemantics memorySemantics;
    u64 dataSize = 2048;
    RankId srcRank = 1;
    RankId dstRank = 0;

    CreateValidSendRecvSemantics(memorySemantics, dataSize, srcRank, dstRank);

    std::vector<DeviceId> rankToDevice(memorySemantics.size());
    for (DeviceId i = 0; i < memorySemantics.size(); i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckSendRecvSemantics(
        memorySemantics, {}, dataSize, rankToDevice[srcRank], rankToDevice[dstRank], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: zero data size
TEST_F(SendRecvSemanticsCheckerTest, ValidSemantics_ZeroDataSize)
{
    PhysicalMemorySemantics memorySemantics;
    u64 dataSize = 0;
    RankId srcRank = 0;
    RankId dstRank = 1;

    CreateValidSendRecvSemantics(memorySemantics, dataSize, srcRank, dstRank);

    std::vector<DeviceId> rankToDevice(memorySemantics.size());
    for (DeviceId i = 0; i < memorySemantics.size(); i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckSendRecvSemantics(
        memorySemantics, {}, dataSize, rankToDevice[srcRank], rankToDevice[dstRank], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test abnormal case: not exactly 2 ranks
TEST_F(SendRecvSemanticsCheckerTest, Abnormal_NotTwoRanks)
{
    PhysicalMemorySemantics memorySemantics;
    u64 dataSize = 1024;
    RankId srcRank = 0;
    RankId dstRank = 1;

    // Add 3 ranks instead of 2
    BufferSemanticMap rank0MemSemantics;
    memorySemantics[0] = rank0MemSemantics;

    BufferSemanticMap rank1MemSemantics;
    memorySemantics[1] = rank1MemSemantics;

    BufferSemanticMap rank2MemSemantics;
    memorySemantics[2] = rank2MemSemantics;

    std::vector<DeviceId> rankToDevice(memorySemantics.size());
    for (DeviceId i = 0; i < memorySemantics.size(); i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckSendRecvSemantics(
        memorySemantics, {}, dataSize, rankToDevice[srcRank], rankToDevice[dstRank], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: single rank
TEST_F(SendRecvSemanticsCheckerTest, Abnormal_SingleRank)
{
    PhysicalMemorySemantics memorySemantics;
    u64 dataSize = 1024;
    RankId srcRank = 0;
    RankId dstRank = 0;

    BufferSemanticMap deviceSemantics;
    memorySemantics[0] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(memorySemantics.size());
    for (DeviceId i = 0; i < memorySemantics.size(); i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckSendRecvSemantics(
        memorySemantics, {}, dataSize, rankToDevice[srcRank], rankToDevice[dstRank], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong source rank in semantics
TEST_F(SendRecvSemanticsCheckerTest, Abnormal_WrongSourceRank)
{
    PhysicalMemorySemantics memorySemantics;
    u64 dataSize = 1024;
    RankId srcRank = 0;
    RankId dstRank = 1;

    BufferSemanticMap dstMemSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize);
    bufSem.srcBufs.insert(SrcBufDes(2, BufferType::INPUT, 0)); // Wrong source rank
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    dstMemSemantics = outputSemantics;
    memorySemantics[dstRank] = dstMemSemantics;

    BufferSemanticMap srcMemSemantics;
    memorySemantics[srcRank] = srcMemSemantics;

    std::vector<DeviceId> rankToDevice(memorySemantics.size());
    for (DeviceId i = 0; i < memorySemantics.size(); i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckSendRecvSemantics(
        memorySemantics, {}, dataSize, rankToDevice[srcRank], rankToDevice[dstRank], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong buffer type
TEST_F(SendRecvSemanticsCheckerTest, Abnormal_WrongBufferType)
{
    PhysicalMemorySemantics memorySemantics;
    u64 dataSize = 1024;
    RankId srcRank = 0;
    RankId dstRank = 1;

    BufferSemanticMap dstMemSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize);
    bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::OUTPUT, 0)); // Wrong buffer type
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    dstMemSemantics = outputSemantics;
    memorySemantics[dstRank] = dstMemSemantics;

    BufferSemanticMap srcMemSemantics;
    memorySemantics[srcRank] = srcMemSemantics;

    std::vector<DeviceId> rankToDevice(memorySemantics.size());
    for (DeviceId i = 0; i < memorySemantics.size(); i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckSendRecvSemantics(
        memorySemantics, {}, dataSize, rankToDevice[srcRank], rankToDevice[dstRank], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong source addr
TEST_F(SendRecvSemanticsCheckerTest, Abnormal_WrongSourceAddr)
{
    PhysicalMemorySemantics memorySemantics;
    u64 dataSize = 1024;
    RankId srcRank = 0;
    RankId dstRank = 1;

    BufferSemanticMap dstMemSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize);
    bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, 100)); // Wrong source addr
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    dstMemSemantics = outputSemantics;
    memorySemantics[dstRank] = dstMemSemantics;

    BufferSemanticMap srcMemSemantics;
    memorySemantics[srcRank] = srcMemSemantics;

    std::vector<DeviceId> rankToDevice(memorySemantics.size());
    for (DeviceId i = 0; i < memorySemantics.size(); i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckSendRecvSemantics(
        memorySemantics, {}, dataSize, rankToDevice[srcRank], rankToDevice[dstRank], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: incomplete total size
TEST_F(SendRecvSemanticsCheckerTest, Abnormal_IncompleteTotalSize)
{
    PhysicalMemorySemantics memorySemantics;
    u64 dataSize = 1024;
    RankId srcRank = 0;
    RankId dstRank = 1;

    BufferSemanticMap dstMemSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize / 2); // Only half the size
    bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    dstMemSemantics = outputSemantics;
    memorySemantics[dstRank] = dstMemSemantics;

    BufferSemanticMap srcMemSemantics;
    memorySemantics[srcRank] = srcMemSemantics;

    std::vector<DeviceId> rankToDevice(memorySemantics.size());
    for (DeviceId i = 0; i < memorySemantics.size(); i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckSendRecvSemantics(
        memorySemantics, {}, dataSize, rankToDevice[srcRank], rankToDevice[dstRank], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test boundary case: large data size
TEST_F(SendRecvSemanticsCheckerTest, ValidSemantics_LargeDataSize)
{
    PhysicalMemorySemantics memorySemantics;
    u64 dataSize = 1024 * 1024 * 1024; // 1GB
    RankId srcRank = 0;
    RankId dstRank = 1;

    CreateValidSendRecvSemantics(memorySemantics, dataSize, srcRank, dstRank);

    std::vector<DeviceId> rankToDevice(memorySemantics.size());
    for (DeviceId i = 0; i < memorySemantics.size(); i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckSendRecvSemantics(
        memorySemantics, {}, dataSize, rankToDevice[srcRank], rankToDevice[dstRank], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}
} // namespace HcclSim
