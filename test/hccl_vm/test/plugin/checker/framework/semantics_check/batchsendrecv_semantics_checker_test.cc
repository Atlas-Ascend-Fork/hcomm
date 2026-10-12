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

#include "batchsendrecv_semantics_checker.h"
#include "check_utils.h"

namespace HcclSim {
class BatchSendRecvSemanticsCheckerTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // Helper function to create valid BatchSendRecv semantics
    void CreateValidBatchSendRecvSemantics(PhysicalMemorySemantics& memorySemantics, u32 rankSize, u64 dataSize)
    {
        for (RankId rankId = 0; rankId < rankSize; rankId++) {
            BufferSemanticMap deviceSemantics;
            BufferSemanticMap outputSemantics;

            u64 totalSize = 0;
            for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
                BufferSemantic bufSem(totalSize, dataSize);
                bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, rankId * dataSize));
                outputSemantics.emplace(bufSem.startAddr, bufSem);
                totalSize += dataSize;
            }

            deviceSemantics = outputSemantics;
            memorySemantics[rankId] = deviceSemantics;
        }
    }

    void CreateValidBatchSendRecvRingSemantics(
        PhysicalMemorySemantics& memorySemantics, u32 rankSize, u64 dataSize, bool splitOutput = false)
    {
        for (RankId rankId = 0; rankId < rankSize; ++rankId) {
            BufferSemanticMap deviceSemantics;
            const RankId srcRank = (rankId + rankSize - 1U) % rankSize;
            if (dataSize != 0) {
                const u64 firstSize = splitOutput ? dataSize / 2 : dataSize;
                BufferSemantic first(0, firstSize);
                first.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, 0));
                deviceSemantics.emplace(first.startAddr, first);
                if (firstSize != dataSize) {
                    BufferSemantic second(firstSize, dataSize - firstSize);
                    second.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, firstSize));
                    deviceSemantics.emplace(second.startAddr, second);
                }
            }
            memorySemantics[rankId] = deviceSemantics;
        }
    }
};

// Test normal case: valid BatchSendRecv semantics with 2 ranks
TEST_F(BatchSendRecvSemanticsCheckerTest, ValidSemantics_TwoRanks)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    CreateValidBatchSendRecvSemantics(memorySemantics, rankSize, dataSize);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBatchSendRecvSemantics(memorySemantics, {}, rankSize, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: valid BatchSendRecv semantics with 4 ranks
TEST_F(BatchSendRecvSemanticsCheckerTest, ValidSemantics_FourRanks)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;

    CreateValidBatchSendRecvSemantics(memorySemantics, rankSize, dataSize);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBatchSendRecvSemantics(memorySemantics, {}, rankSize, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: single rank
TEST_F(BatchSendRecvSemanticsCheckerTest, ValidSemantics_SingleRank)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 1;
    u64 dataSize = 1024;

    CreateValidBatchSendRecvSemantics(memorySemantics, rankSize, dataSize);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBatchSendRecvSemantics(memorySemantics, {}, rankSize, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: zero data size - manually construct zero-size BufferSemantic
// to satisfy checker's curRankId advancement logic (empty set would fail curRankId != rankSize)
TEST_F(BatchSendRecvSemanticsCheckerTest, ValidSemantics_ZeroDataSize)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 1;
    u64 dataSize = 0;

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        BufferSemanticMap deviceSemantics;
        BufferSemanticMap outputSemantics;

        u64 totalSize = 0;
        for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
            BufferSemantic bufSem(totalSize, dataSize);
            bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, dataSize * rankId));
            outputSemantics.emplace(bufSem.startAddr, bufSem);
            totalSize += dataSize;
        }

        deviceSemantics = outputSemantics;
        memorySemantics[rankId] = deviceSemantics;
    }

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBatchSendRecvSemantics(memorySemantics, {}, rankSize, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test abnormal case: missing rank
TEST_F(BatchSendRecvSemanticsCheckerTest, Abnormal_MissingRank)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    // Only add rank 0
    BufferSemanticMap deviceSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem0(0, dataSize);
    bufSem0.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem0.startAddr, bufSem0);

    BufferSemantic bufSem1(dataSize, dataSize);
    bufSem1.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, dataSize));
    outputSemantics.emplace(bufSem1.startAddr, bufSem1);

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBatchSendRecvSemantics(memorySemantics, {}, rankSize, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

TEST_F(BatchSendRecvSemanticsCheckerTest, Abnormal_ExpectedRankSetIsNotInferredFromActualMap)
{
    PhysicalMemorySemantics memorySemantics;
    constexpr u64 dataSize = 1024;

    BufferSemanticMap deviceSemantics;
    BufferSemantic output(0, dataSize);
    output.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    deviceSemantics.emplace(output.startAddr, output);
    memorySemantics[0] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(2);
    for (DeviceId i = 0; i < 2; i++)
        rankToDevice[i] = i;
    EXPECT_EQ(TaskCheckBatchSendRecvSemantics(memorySemantics, {}, 2, dataSize, rankToDevice), HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong source rank
TEST_F(BatchSendRecvSemanticsCheckerTest, Abnormal_WrongSourceRank)
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
    bufSem1.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, dataSize)); // Wrong source rank
    outputSemantics.emplace(bufSem1.startAddr, bufSem1);

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;
    memorySemantics[1] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBatchSendRecvSemantics(memorySemantics, {}, rankSize, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong buffer type
TEST_F(BatchSendRecvSemanticsCheckerTest, Abnormal_WrongBufferType)
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
    bufSem1.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, dataSize));
    outputSemantics.emplace(bufSem1.startAddr, bufSem1);

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;
    memorySemantics[1] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBatchSendRecvSemantics(memorySemantics, {}, rankSize, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test boundary case: large number of ranks
TEST_F(BatchSendRecvSemanticsCheckerTest, ValidSemantics_LargeRankSize)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 128;
    u64 dataSize = 1024;

    CreateValidBatchSendRecvSemantics(memorySemantics, rankSize, dataSize);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBatchSendRecvSemantics(memorySemantics, {}, rankSize, dataSize, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

TEST_F(BatchSendRecvSemanticsCheckerTest, RingValidSemantics_FourRanksWithSplitOutput)
{
    PhysicalMemorySemantics memorySemantics;
    CreateValidBatchSendRecvRingSemantics(memorySemantics, 4, 1024, true);

    std::vector<DeviceId> rankToDevice(4);
    for (DeviceId i = 0; i < 4; i++)
        rankToDevice[i] = i;
    EXPECT_EQ(
        TaskCheckBatchSendRecvRingSemantics(memorySemantics, {}, 4, 1024, rankToDevice), HcclResult::HCCL_SUCCESS);
}

TEST_F(BatchSendRecvSemanticsCheckerTest, RingValidSemantics_ZeroDataSize)
{
    PhysicalMemorySemantics memorySemantics;
    CreateValidBatchSendRecvRingSemantics(memorySemantics, 2, 0);

    std::vector<DeviceId> rankToDevice(2);
    for (DeviceId i = 0; i < 2; i++)
        rankToDevice[i] = i;
    EXPECT_EQ(TaskCheckBatchSendRecvRingSemantics(memorySemantics, {}, 2, 0, rankToDevice), HcclResult::HCCL_SUCCESS);
}

TEST_F(BatchSendRecvSemanticsCheckerTest, RingAbnormal_SingleRank)
{
    PhysicalMemorySemantics memorySemantics;
    CreateValidBatchSendRecvRingSemantics(memorySemantics, 1, 1024);

    std::vector<DeviceId> rankToDevice(1);
    for (DeviceId i = 0; i < 1; i++)
        rankToDevice[i] = i;
    EXPECT_EQ(TaskCheckBatchSendRecvRingSemantics(memorySemantics, {}, 1, 1024, rankToDevice), HcclResult::HCCL_E_PARA);
}

TEST_F(BatchSendRecvSemanticsCheckerTest, RingAbnormal_MissingRank)
{
    PhysicalMemorySemantics memorySemantics;
    CreateValidBatchSendRecvRingSemantics(memorySemantics, 4, 1024);
    memorySemantics.erase(3);

    std::vector<DeviceId> rankToDevice(4);
    for (DeviceId i = 0; i < 4; i++)
        rankToDevice[i] = i;
    EXPECT_EQ(TaskCheckBatchSendRecvRingSemantics(memorySemantics, {}, 4, 1024, rankToDevice), HcclResult::HCCL_E_PARA);
}

TEST_F(BatchSendRecvSemanticsCheckerTest, RingAbnormal_WrongSourceRank)
{
    PhysicalMemorySemantics memorySemantics;
    CreateValidBatchSendRecvRingSemantics(memorySemantics, 4, 1024);
    memorySemantics[2].clear();
    BufferSemantic output(0, 1024);
    output.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    memorySemantics[2].emplace(output.startAddr, output);

    std::vector<DeviceId> rankToDevice(4);
    for (DeviceId i = 0; i < 4; i++)
        rankToDevice[i] = i;
    EXPECT_EQ(TaskCheckBatchSendRecvRingSemantics(memorySemantics, {}, 4, 1024, rankToDevice), HcclResult::HCCL_E_PARA);
}

TEST_F(BatchSendRecvSemanticsCheckerTest, RingAbnormal_WrongSourceOffset)
{
    PhysicalMemorySemantics memorySemantics;
    CreateValidBatchSendRecvRingSemantics(memorySemantics, 2, 1024);
    memorySemantics[0].clear();
    BufferSemantic output(0, 1024);
    output.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 128));
    memorySemantics[0].emplace(output.startAddr, output);

    std::vector<DeviceId> rankToDevice(2);
    for (DeviceId i = 0; i < 2; i++)
        rankToDevice[i] = i;
    EXPECT_EQ(TaskCheckBatchSendRecvRingSemantics(memorySemantics, {}, 2, 1024, rankToDevice), HcclResult::HCCL_E_PARA);
}

TEST_F(BatchSendRecvSemanticsCheckerTest, RingAbnormal_WrongBufferType)
{
    PhysicalMemorySemantics memorySemantics;
    CreateValidBatchSendRecvRingSemantics(memorySemantics, 2, 1024);
    memorySemantics[0].clear();
    BufferSemantic output(0, 1024);
    output.srcBufs.insert(SrcBufDes(1, BufferType::OUTPUT, 0));
    memorySemantics[0].emplace(output.startAddr, output);

    std::vector<DeviceId> rankToDevice(2);
    for (DeviceId i = 0; i < 2; i++)
        rankToDevice[i] = i;
    EXPECT_EQ(TaskCheckBatchSendRecvRingSemantics(memorySemantics, {}, 2, 1024, rankToDevice), HcclResult::HCCL_E_PARA);
}

TEST_F(BatchSendRecvSemanticsCheckerTest, RingAbnormal_OutputGap)
{
    PhysicalMemorySemantics memorySemantics;
    CreateValidBatchSendRecvRingSemantics(memorySemantics, 2, 1024);
    memorySemantics[0].clear();
    BufferSemantic output(128, 896);
    output.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 128));
    memorySemantics[0].emplace(output.startAddr, output);

    std::vector<DeviceId> rankToDevice(2);
    for (DeviceId i = 0; i < 2; i++)
        rankToDevice[i] = i;
    EXPECT_EQ(TaskCheckBatchSendRecvRingSemantics(memorySemantics, {}, 2, 1024, rankToDevice), HcclResult::HCCL_E_PARA);
}
} // namespace HcclSim
