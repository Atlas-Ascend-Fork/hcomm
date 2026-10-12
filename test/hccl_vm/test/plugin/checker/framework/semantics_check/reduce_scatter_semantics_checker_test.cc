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
#include "reduce_scatter_semantics_checker.h"

namespace HcclSim {
class ReduceScatterSemanticsCheckerTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // Helper function to create valid ReduceScatter semantics
    void CreateValidReduceScatterSemantics(
        PhysicalMemorySemantics& memorySemantics, u32 rankSize, u64 dataSize, HcclReduceOp reduceType)
    {
        for (RankId rankId = 0; rankId < rankSize; rankId++) {
            BufferSemanticMap deviceSemantics;
            BufferSemanticMap outputSemantics;

            BufferSemantic bufSem(0, dataSize, true, reduceType);
            for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
                bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, rankId * dataSize));
            }
            outputSemantics.emplace(bufSem.startAddr, bufSem);

            deviceSemantics = outputSemantics;
            memorySemantics[rankId] = deviceSemantics;
        }
    }
};

// Test normal case: valid ReduceScatter semantics with 2 ranks
TEST_F(ReduceScatterSemanticsCheckerTest, ValidSemantics_TwoRanks)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    CreateValidReduceScatterSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckReduceScatterSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: valid ReduceScatter semantics with 4 ranks
TEST_F(ReduceScatterSemanticsCheckerTest, ValidSemantics_FourRanks)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;

    CreateValidReduceScatterSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckReduceScatterSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: different reduce operations
TEST_F(ReduceScatterSemanticsCheckerTest, ValidSemantics_ReduceProd)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;

    CreateValidReduceScatterSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_PROD);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckReduceScatterSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_PROD, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: single rank
TEST_F(ReduceScatterSemanticsCheckerTest, ValidSemantics_SingleRank)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 1;
    u64 dataSize = 1024;

    CreateValidReduceScatterSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckReduceScatterSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: zero data size
TEST_F(ReduceScatterSemanticsCheckerTest, ValidSemantics_ZeroDataSize)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 4;
    u64 dataSize = 0;

    CreateValidReduceScatterSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckReduceScatterSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test abnormal case: missing rank
TEST_F(ReduceScatterSemanticsCheckerTest, Abnormal_MissingRank)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    // Only add rank 0
    BufferSemanticMap deviceSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize, true, HcclReduceOp::HCCL_REDUCE_SUM);
    bufSem.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    bufSem.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, dataSize));
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckReduceScatterSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong reduce type
TEST_F(ReduceScatterSemanticsCheckerTest, Abnormal_WrongReduceType)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    CreateValidReduceScatterSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckReduceScatterSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_PROD, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: insufficient source buffers
TEST_F(ReduceScatterSemanticsCheckerTest, Abnormal_InsufficientSourceBuffers)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;

    BufferSemanticMap deviceSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize, true, HcclReduceOp::HCCL_REDUCE_SUM);
    // Only add 2 source buffers instead of 4
    bufSem.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    bufSem.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, dataSize));
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;
    memorySemantics[1] = deviceSemantics;
    memorySemantics[2] = deviceSemantics;
    memorySemantics[3] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckReduceScatterSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test boundary case: large number of ranks
TEST_F(ReduceScatterSemanticsCheckerTest, ValidSemantics_LargeRankSize)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 128;
    u64 dataSize = 1024;

    CreateValidReduceScatterSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckReduceScatterSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}
} // namespace HcclSim
