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

#include "allreduce_semantics_checker.h"
#include "check_utils.h"

namespace HcclSim {
class AllReduceSemanticsCheckerTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // Helper function to create valid AllReduce semantics
    void CreateValidAllReduceSemantics(
        PhysicalMemorySemantics& memorySemantics, u32 rankSize, u64 dataSize, HcclReduceOp reduceType)
    {
        for (RankId rankId = 0; rankId < rankSize; rankId++) {
            BufferSemanticMap deviceSemantics;
            BufferSemanticMap outputSemantics;

            BufferSemantic bufSem(0, dataSize, true, reduceType);
            for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
                bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, 0));
            }
            outputSemantics.emplace(bufSem.startAddr, bufSem);

            deviceSemantics = outputSemantics;
            memorySemantics[rankId] = deviceSemantics;
        }
    }
};

// Test normal case: valid AllReduce semantics with 2 ranks
TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_TwoRanks)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    CreateValidAllReduceSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: valid AllReduce semantics with 4 ranks
TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_FourRanks)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 4;
    u64 dataSize = 2048;

    CreateValidAllReduceSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: different reduce operations
TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_ReduceProd)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;

    CreateValidAllReduceSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_PROD);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_PROD, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_ReduceMax)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;

    CreateValidAllReduceSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_MAX);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_MAX, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_ReduceMin)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;

    CreateValidAllReduceSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_MIN);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_MIN, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: single rank
TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_SingleRank)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 1;
    u64 dataSize = 1024;

    CreateValidAllReduceSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: zero data size
TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_ZeroDataSize)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 0;

    CreateValidAllReduceSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test abnormal case: missing rank
TEST_F(AllReduceSemanticsCheckerTest, Abnormal_MissingRank)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    // Only add rank 0
    BufferSemanticMap deviceSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize, true, HcclReduceOp::HCCL_REDUCE_SUM);
    bufSem.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    bufSem.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong reduce type
TEST_F(AllReduceSemanticsCheckerTest, Abnormal_WrongReduceType)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    // Create with SUM but check with PROD
    CreateValidAllReduceSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_PROD, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: insufficient source buffers
TEST_F(AllReduceSemanticsCheckerTest, Abnormal_InsufficientSourceBuffers)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;

    BufferSemanticMap deviceSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize, true, HcclReduceOp::HCCL_REDUCE_SUM);
    // Only add 2 source buffers instead of 4
    bufSem.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    bufSem.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
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
        = TaskCheckAllReduceSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong source buffer type
TEST_F(AllReduceSemanticsCheckerTest, Abnormal_WrongBufferType)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    BufferSemanticMap deviceSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize, true, HcclReduceOp::HCCL_REDUCE_SUM);
    bufSem.srcBufs.insert(SrcBufDes(0, BufferType::OUTPUT, 0)); // Wrong buffer type
    bufSem.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;
    memorySemantics[1] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: invalid source rank range
TEST_F(AllReduceSemanticsCheckerTest, Abnormal_InvalidSourceRankRange)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    BufferSemanticMap deviceSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize, true, HcclReduceOp::HCCL_REDUCE_SUM);
    bufSem.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    bufSem.srcBufs.insert(SrcBufDes(2, BufferType::INPUT, 0)); // Invalid rank (out of range)
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;
    memorySemantics[1] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong total size
TEST_F(AllReduceSemanticsCheckerTest, Abnormal_WrongTotalSize)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    BufferSemanticMap deviceSemantics;
    BufferSemanticMap outputSemantics;

    // Create buffer with wrong size
    BufferSemantic bufSem(0, 512, true, HcclReduceOp::HCCL_REDUCE_SUM); // Wrong size
    bufSem.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    bufSem.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    deviceSemantics = outputSemantics;
    memorySemantics[0] = deviceSemantics;
    memorySemantics[1] = deviceSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test boundary case: large number of ranks
TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_LargeRankSize)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 128;
    u64 dataSize = 1024;

    CreateValidAllReduceSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: large data size
TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_LargeDataSize)
{
    PhysicalMemorySemantics memorySemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024 * 1024 * 1024; // 1GB

    CreateValidAllReduceSemantics(memorySemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(memorySemantics, {}, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}
} // namespace HcclSim

namespace HcclSim {
// 物理语义合并后可跨越输出边界；校验应只读取注册窗口并正确平移数据来源。
TEST_F(AllReduceSemanticsCheckerTest, AbsoluteOutputWindowInsideMergedSemantic)
{
    PhysicalMemorySemantics memory;
    DeviceBufferAddressLayouts layouts;
    const std::vector<DeviceId> devices{3, 7};
    for (DeviceId device : devices) {
        auto& layout = layouts[device];
        layout.hasInput = true;
        layout.hasOutput = true;
        layout.inputBase = 0x100000 + device * 0x1000;
        layout.outputBase = 0x200000 + device * 0x1000;
        layout.outputSize = 64;
    }
    for (DeviceId device : devices) {
        BufferSemantic segment(layouts[device].outputBase - 16, 96, true, HCCL_REDUCE_SUM);
        for (DeviceId source : devices) {
            segment.srcBufs.emplace(source, BufferType::INPUT, layouts[source].inputBase - 16);
        }
        memory[device].emplace(segment.startAddr, segment);
    }
    EXPECT_EQ(TaskCheckAllReduceSemantics(memory, layouts, 64, HCCL_REDUCE_SUM, devices), HCCL_SUCCESS);
    // 裁剪不能修改原始语义表，也不能掩盖窗口内的错误来源。
    EXPECT_EQ(memory.at(3).begin()->second.size, 96U);
    memory.at(3).begin()->second.srcBufs.clear();
    memory.at(3).begin()->second.srcBufs.emplace(3, BufferType::INPUT, layouts[3].inputBase);
    memory.at(3).begin()->second.srcBufs.emplace(7, BufferType::INPUT, layouts[7].inputBase);
    EXPECT_NE(TaskCheckAllReduceSemantics(memory, layouts, 64, HCCL_REDUCE_SUM, devices), HCCL_SUCCESS);
}

// 注册窗口大于算子结果时，窗口内的多余数据仍必须报错。
TEST_F(AllReduceSemanticsCheckerTest, AbsoluteOutputRejectsExtraDataInsideRegisteredBuffer)
{
    DeviceBufferAddressLayouts layouts;
    layouts[3].hasInput = layouts[3].hasOutput = true;
    layouts[3].inputBase = 0x1000;
    layouts[3].outputBase = 0x2000;
    layouts[3].outputSize = 128;
    PhysicalMemorySemantics memory;
    BufferSemantic segment(0x2000, 128);
    segment.srcBufs.emplace(3, BufferType::INPUT, 0x1000);
    memory[3].emplace(segment.startAddr, segment);
    EXPECT_NE(TaskCheckAllReduceSemantics(memory, layouts, 64, HCCL_REDUCE_SUM, {3}), HCCL_SUCCESS);
}
} // namespace HcclSim
