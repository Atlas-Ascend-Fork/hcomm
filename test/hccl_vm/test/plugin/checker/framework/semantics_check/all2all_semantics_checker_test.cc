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

#include "all2all_semantics_checker.h"
#include "check_utils.h"
#include "checker_def.h"

namespace HcclSim {
class All2AllSemanticsCheckerTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // Helper function to create valid All2All semantics
    void CreateValidAll2AllSemantics(
        PhysicalMemorySemantics& memorySemantics, All2AllDataDesTagInner& all2AllDataDes, u32 rankSize,
        u64 countPerRank)
    {
        all2AllDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_INT32;
        all2AllDataDes.sendCountMatrix.clear();
        all2AllDataDes.sendCountMatrix.resize(rankSize * rankSize, countPerRank);

        u64 dataSize = countPerRank * CHECK_SIZE_TABLE[all2AllDataDes.recvType];

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
};

// Test normal case: valid All2All semantics with 2 ranks
TEST_F(All2AllSemanticsCheckerTest, ValidSemantics_TwoRanks)
{
    PhysicalMemorySemantics memorySemantics;
    All2AllDataDesTagInner all2AllDataDes;
    u32 rankSize = 2;
    u64 countPerRank = 100;

    CreateValidAll2AllSemantics(memorySemantics, all2AllDataDes, rankSize, countPerRank);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAll2AllSemantics(memorySemantics, {}, all2AllDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: valid All2All semantics with 4 ranks
TEST_F(All2AllSemanticsCheckerTest, ValidSemantics_FourRanks)
{
    PhysicalMemorySemantics memorySemantics;
    All2AllDataDesTagInner all2AllDataDes;
    u32 rankSize = 4;
    u64 countPerRank = 100;

    CreateValidAll2AllSemantics(memorySemantics, all2AllDataDes, rankSize, countPerRank);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAll2AllSemantics(memorySemantics, {}, all2AllDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: single rank
TEST_F(All2AllSemanticsCheckerTest, ValidSemantics_SingleRank)
{
    PhysicalMemorySemantics memorySemantics;
    All2AllDataDesTagInner all2AllDataDes;
    u32 rankSize = 1;
    u64 countPerRank = 100;

    CreateValidAll2AllSemantics(memorySemantics, all2AllDataDes, rankSize, countPerRank);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAll2AllSemantics(memorySemantics, {}, all2AllDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: zero count - no data transferred, buffer semantics should be empty
TEST_F(All2AllSemanticsCheckerTest, ValidSemantics_ZeroCount)
{
    PhysicalMemorySemantics memorySemantics;
    All2AllDataDesTagInner all2AllDataDes;
    u32 rankSize = 2;
    u64 countPerRank = 0;

    all2AllDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_INT32;
    all2AllDataDes.sendCountMatrix.clear();
    all2AllDataDes.sendCountMatrix.resize(rankSize * rankSize, countPerRank);

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        BufferSemanticMap deviceSemantics;
        memorySemantics[rankId] = deviceSemantics;
    }

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAll2AllSemantics(memorySemantics, {}, all2AllDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test abnormal case: missing rank
TEST_F(All2AllSemanticsCheckerTest, Abnormal_MissingRank)
{
    PhysicalMemorySemantics memorySemantics;
    All2AllDataDesTagInner all2AllDataDes;
    u32 rankSize = 2;
    u64 countPerRank = 100;

    all2AllDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_INT32;
    all2AllDataDes.sendCountMatrix.resize(rankSize * rankSize, countPerRank);

    // Only add rank 0
    u64 dataSize = countPerRank * CHECK_SIZE_TABLE[all2AllDataDes.recvType];
    BufferSemanticMap deviceSemantics;
    BufferSemanticMap outputSemantics;

    for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
        BufferSemantic bufSem(srcRank * dataSize, dataSize);
        bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, 0));
        outputSemantics.emplace(bufSem.startAddr, bufSem);
    }

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAll2AllSemantics(memorySemantics, {}, all2AllDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong source rank
TEST_F(All2AllSemanticsCheckerTest, Abnormal_WrongSourceRank)
{
    PhysicalMemorySemantics memorySemantics;
    All2AllDataDesTagInner all2AllDataDes;
    u32 rankSize = 2;
    u64 countPerRank = 100;

    all2AllDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_INT32;
    all2AllDataDes.sendCountMatrix.resize(rankSize * rankSize, countPerRank);

    u64 dataSize = countPerRank * CHECK_SIZE_TABLE[all2AllDataDes.recvType];

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
    HcclResult result = TaskCheckAll2AllSemantics(memorySemantics, {}, all2AllDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test boundary case: large rank size
TEST_F(All2AllSemanticsCheckerTest, ValidSemantics_LargeRankSize)
{
    PhysicalMemorySemantics memorySemantics;
    All2AllDataDesTagInner all2AllDataDes;
    u32 rankSize = 16;
    u64 countPerRank = 100;

    CreateValidAll2AllSemantics(memorySemantics, all2AllDataDes, rankSize, countPerRank);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAll2AllSemantics(memorySemantics, {}, all2AllDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}
} // namespace HcclSim
