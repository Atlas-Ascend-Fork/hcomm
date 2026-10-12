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
#include "check_utils.h"
#include "data_slice.h"

namespace HcclSim {
bool IsSendRecvType(HcclCMDType opType);

class CheckUtilsTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(CheckUtilsTest, IsAllToAllSeries_AllToAll) { EXPECT_TRUE(IsAllToAllSeries(HcclCMDType::HCCL_CMD_ALLTOALL)); }

TEST_F(CheckUtilsTest, IsAllToAllSeries_AllToAllV) { EXPECT_TRUE(IsAllToAllSeries(HcclCMDType::HCCL_CMD_ALLTOALLV)); }

TEST_F(CheckUtilsTest, IsAllToAllSeries_AllToAllVC) { EXPECT_TRUE(IsAllToAllSeries(HcclCMDType::HCCL_CMD_ALLTOALLVC)); }

TEST_F(CheckUtilsTest, IsAllToAllSeries_NotAllToAll)
{
    EXPECT_FALSE(IsAllToAllSeries(HcclCMDType::HCCL_CMD_ALLREDUCE));
    EXPECT_FALSE(IsAllToAllSeries(HcclCMDType::HCCL_CMD_ALLGATHER));
    EXPECT_FALSE(IsAllToAllSeries(HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
}

TEST_F(CheckUtilsTest, IsSendRecvType_Send) { EXPECT_TRUE(IsSendRecvType(HcclCMDType::HCCL_CMD_SEND)); }

TEST_F(CheckUtilsTest, IsSendRecvType_Recv) { EXPECT_TRUE(IsSendRecvType(HcclCMDType::HCCL_CMD_RECEIVE)); }

TEST_F(CheckUtilsTest, IsSendRecvType_NotSendRecv)
{
    EXPECT_FALSE(IsSendRecvType(HcclCMDType::HCCL_CMD_ALLREDUCE));
    EXPECT_FALSE(IsSendRecvType(HcclCMDType::HCCL_CMD_BROADCAST));
}

TEST_F(CheckUtilsTest, CalcDataSize_Int8)
{
    uint64_t dataSize = 0;
    CalcDataSize(HcclCMDType::HCCL_CMD_ALLREDUCE, 100, HcclDataType::HCCL_DATA_TYPE_INT8, dataSize);
    EXPECT_EQ(dataSize, 100);
}

TEST_F(CheckUtilsTest, CalcDataSize_Int32)
{
    uint64_t dataSize = 0;
    CalcDataSize(HcclCMDType::HCCL_CMD_ALLREDUCE, 100, HcclDataType::HCCL_DATA_TYPE_INT32, dataSize);
    EXPECT_EQ(dataSize, 400);
}

TEST_F(CheckUtilsTest, CalcDataSize_Fp16)
{
    uint64_t dataSize = 0;
    CalcDataSize(HcclCMDType::HCCL_CMD_ALLREDUCE, 100, HcclDataType::HCCL_DATA_TYPE_FP16, dataSize);
    EXPECT_EQ(dataSize, 200);
}

TEST_F(CheckUtilsTest, CalcDataSize_Fp32)
{
    uint64_t dataSize = 0;
    CalcDataSize(HcclCMDType::HCCL_CMD_ALLREDUCE, 100, HcclDataType::HCCL_DATA_TYPE_FP32, dataSize);
    EXPECT_EQ(dataSize, 400);
}

TEST_F(CheckUtilsTest, CalcDataSize_ZeroCount)
{
    uint64_t dataSize = 0;
    CalcDataSize(HcclCMDType::HCCL_CMD_ALLREDUCE, 0, HcclDataType::HCCL_DATA_TYPE_INT32, dataSize);
    EXPECT_EQ(dataSize, 0);
}

TEST_F(CheckUtilsTest, CalcDataSize_LargeCount)
{
    uint64_t dataSize = 0;
    CalcDataSize(HcclCMDType::HCCL_CMD_ALLREDUCE, 1000000, HcclDataType::HCCL_DATA_TYPE_INT32, dataSize);
    EXPECT_EQ(dataSize, 4000000);
}

TEST_F(CheckUtilsTest, SrcBufDes_Constructor)
{
    SrcBufDes srcBuf(1, BufferType::INPUT, 0x1000);

    EXPECT_EQ(srcBuf.deviceId, 1);
    EXPECT_EQ(srcBuf.bufType, BufferType::INPUT);
    EXPECT_EQ(srcBuf.srcAddr, 0x1000);
}

TEST_F(CheckUtilsTest, SrcBufDes_Comparison)
{
    SrcBufDes srcBuf1(1, BufferType::INPUT, 0x1000);
    SrcBufDes srcBuf2(2, BufferType::INPUT, 0x2000);

    EXPECT_TRUE(srcBuf1 < srcBuf2);
    EXPECT_FALSE(srcBuf2 < srcBuf1);
}

TEST_F(CheckUtilsTest, BufferSemantic_Constructor)
{
    BufferSemantic bufSem(0x1000, 1024);

    EXPECT_EQ(bufSem.startAddr, 0x1000);
    EXPECT_EQ(bufSem.size, 1024);
    EXPECT_EQ(bufSem.isReduce, false);
}

TEST_F(CheckUtilsTest, BufferSemantic_WithReduce)
{
    BufferSemantic bufSem(0x1000, 1024, true, HcclReduceOp::HCCL_REDUCE_SUM);

    EXPECT_EQ(bufSem.isReduce, true);
    EXPECT_EQ(bufSem.reduceType, HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(CheckUtilsTest, BufferSemantic_Comparison)
{
    BufferSemantic bufSem1(0x1000, 1024);
    BufferSemantic bufSem2(0x2000, 2048);

    EXPECT_TRUE(bufSem1 < bufSem2);
    EXPECT_FALSE(bufSem2 < bufSem1);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_AllReduce)
{
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_ALLREDUCE, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0, 0, {},
        {});
    EXPECT_EQ(inputSize, 400);
    EXPECT_EQ(outputSize, 400);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_Broadcast)
{
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_BROADCAST, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0, 0, {},
        {});
    EXPECT_EQ(inputSize, 400);
    EXPECT_EQ(outputSize, 400);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_Send)
{
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_SEND, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0, 1, {}, {});
    EXPECT_EQ(inputSize, 400);
    EXPECT_EQ(outputSize, 0);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_Receive)
{
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_RECEIVE, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 1, 0, 1, {},
        {});
    EXPECT_EQ(inputSize, 0);
    EXPECT_EQ(outputSize, 400);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_Reduce)
{
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_REDUCE, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0, 0, {},
        {});
    EXPECT_EQ(inputSize, 400);
    EXPECT_EQ(outputSize, 400);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_AllGather)
{
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_ALLGATHER, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0, 0, {},
        {});
    EXPECT_EQ(inputSize, 400);
    EXPECT_EQ(outputSize, 1600);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_ReduceScatter)
{
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0,
        0, {}, {});
    EXPECT_EQ(inputSize, 1600);
    EXPECT_EQ(outputSize, 400);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_Scatter)
{
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_SCATTER, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0, 0, {},
        {});
    EXPECT_EQ(inputSize, 1600);
    EXPECT_EQ(outputSize, 400);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_BatchSendRecv)
{
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0,
        0, {}, {});
    EXPECT_EQ(inputSize, 400);
    EXPECT_EQ(outputSize, 400);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_UnsupportedOp)
{
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_INVALID, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0, 0, {},
        {});
    // For unsupported ops, the function just logs a warning
    EXPECT_EQ(inputSize, 0);
    EXPECT_EQ(outputSize, 0);
}
} // namespace HcclSim
