/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/MockObject.h>
#include <cstdint>
#include <vector>
#define private public
#include "orion_adapter_rts.h"
#include "p2p_transport_lite_impl.h"
#include "stream_lite.h"
#include "rtsq_a5.h"
#include "sqe.h"
#include "ascend_hal.h"
#include "drv_api_exception.h"
#include "invalid_params_exception.h"
#undef private

using namespace Hccl;

class P2PTransportLiteImplReadReduceTest : public testing::Test {
protected:
    struct ReduceCall {
        u64 src;
        u64 dst;
        u32 size;
    };

    static void SetUpTestCase() { std::cout << "P2PTransportLiteImplReadReduceTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "P2PTransportLiteImplReadReduceTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        reduceCalls_.clear();
        mockSq.assign(AC_SQE_MAX_CNT * RTSQ_SQE_SIZE, 0);
        MOCKER(HrtGetDeviceType)
            .stubs()
            .with(outBound(Hccl::DevType((DevType)DevType::DEV_TYPE_910A2)))
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(mockSq.data())));
        MOCKER_CPP(&RtsqBase::QuerySqDepth)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(static_cast<u32>(AC_SQE_MAX_CNT)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();
        std::cout << "A Test case in P2PTransportLiteImplReadReduceTest SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in P2PTransportLiteImplReadReduceTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }

    std::vector<char> GetNotifyUniqueId(u32 notifyId, u32 devPhyId)
    {
        BinaryStream binaryStream;
        binaryStream << notifyId;
        binaryStream << devPhyId;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 按 P2PTransportLiteImpl 构造函数的解析格式序列化：
    // [type][notifyNum][bufferNum][notifyUniqueIds][rmtNotifyUniqueIds][rmtBufferUniqueIds]。
    std::vector<char> BuildP2PTransportUniqueId(u32 devPhyId)
    {
        u32 transportType = 0;
        u32 notifyNum = 1;
        u32 bufferNum = 0;
        std::vector<char> notifyUniqueId = GetNotifyUniqueId(0, devPhyId);
        BinaryStream rmtNotifyStream;
        u64 fakeAddr = 0x1000;
        u64 fakeSize = 0x100;
        u32 fakeNotifyId = 0;
        rmtNotifyStream << fakeAddr;
        rmtNotifyStream << fakeSize;
        rmtNotifyStream << fakeNotifyId;
        std::vector<char> rmtNotifyUniqueId;
        rmtNotifyStream.Dump(rmtNotifyUniqueId);
        std::vector<char> rmtBufferUniqueId;

        BinaryStream binaryStream;
        binaryStream << transportType;
        binaryStream << notifyNum;
        binaryStream << bufferNum;
        binaryStream << notifyUniqueId;
        binaryStream << rmtNotifyUniqueId;
        binaryStream << rmtBufferUniqueId;
        std::vector<char> p2pUniqueId;
        binaryStream.Dump(p2pUniqueId);
        return p2pUniqueId;
    }

    std::unique_ptr<StreamLite> MakeStream(u32 fakeDevPhyId)
    {
        u32 fakeStreamId = 1;
        u32 fakeSqId = 1;
        BinaryStream liteBinaryStream;
        liteBinaryStream << fakeStreamId;
        liteBinaryStream << fakeSqId;
        liteBinaryStream << fakeDevPhyId;
        std::vector<char> uniqueId{};
        liteBinaryStream.Dump(uniqueId);
        std::unique_ptr<StreamLite> stream = std::make_unique<StreamLite>(uniqueId);
        stream->rtsq = std::make_unique<RtsqA5>(fakeDevPhyId, fakeStreamId, fakeSqId);
        return stream;
    }

    std::unique_ptr<P2PTransportLiteImpl> MakeTransport(u32 fakeDevPhyId)
    {
        std::vector<char> p2pUniqueId = BuildP2PTransportUniqueId(fakeDevPhyId);
        auto callback = [this](u32 streamId, u32 taskId, const TaskParam& taskParam) {
            (void)streamId;
            (void)taskId;
            const ParaReduce& reduce = taskParam.taskPara.Reduce;
            reduceCalls_.push_back(
                {static_cast<u64>(reinterpret_cast<uintptr_t>(reduce.src)),
                 static_cast<u64>(reinterpret_cast<uintptr_t>(reduce.dst)), static_cast<u32>(reduce.size)});
        };
        return std::make_unique<P2PTransportLiteImpl>(p2pUniqueId, callback);
    }

    std::vector<u8> mockSq;
    std::vector<ReduceCall> reduceCalls_;
};

TEST_F(P2PTransportLiteImplReadReduceTest, P2PTransportLiteImpl_ReadReduce_LargeData_SplitAligned)
{
    const u64 elemSize = 4;
    const u64 maxTaskBytes = (UINT32_MAX / elemSize) * elemSize;
    const u64 totalSize = 8ULL * 1024 * 1024 * 1024;
    const u64 rmtBase = 0x100000000ULL;
    const u64 locBase = 0x200000000ULL;

    std::unique_ptr<StreamLite> stream = MakeStream(1);
    std::unique_ptr<P2PTransportLiteImpl> transportLite = MakeTransport(1);

    RmaBufferLite loc(locBase, totalSize);
    Buffer rmt(rmtBase, totalSize);
    ReduceIn reduceIn(DataType::FP32, ReduceOp::SUM);

    transportLite->ReadReduce(loc, rmt, reduceIn, *stream);

    ASSERT_EQ(reduceCalls_.size(), 3U);
    EXPECT_EQ(reduceCalls_[0].size, static_cast<u32>(maxTaskBytes));
    EXPECT_EQ(reduceCalls_[1].size, static_cast<u32>(maxTaskBytes));
    EXPECT_EQ(reduceCalls_[2].size, static_cast<u32>(totalSize - maxTaskBytes * 2));
    EXPECT_EQ(reduceCalls_[0].src, rmtBase);
    EXPECT_EQ(reduceCalls_[1].src, rmtBase + maxTaskBytes);
    EXPECT_EQ(reduceCalls_[2].src, rmtBase + maxTaskBytes * 2);
    EXPECT_EQ(reduceCalls_[0].dst, locBase);
    EXPECT_EQ(reduceCalls_[1].dst, locBase + maxTaskBytes);
    EXPECT_EQ(reduceCalls_[2].dst, locBase + maxTaskBytes * 2);
}

TEST_F(P2PTransportLiteImplReadReduceTest, P2PTransportLiteImpl_ReadReduce_MisalignedSize_Throw)
{
    const u64 totalSize = 8ULL * 1024 * 1024 * 1024 + 1;

    std::unique_ptr<StreamLite> stream = MakeStream(1);
    std::unique_ptr<P2PTransportLiteImpl> transportLite = MakeTransport(1);

    RmaBufferLite loc(0x200000000ULL, totalSize);
    Buffer rmt(0x100000000ULL, totalSize);
    ReduceIn reduceIn(DataType::FP32, ReduceOp::SUM);

    EXPECT_THROW(transportLite->ReadReduce(loc, rmt, reduceIn, *stream), InvalidParamsException);
    EXPECT_TRUE(reduceCalls_.empty());
}

TEST_F(P2PTransportLiteImplReadReduceTest, P2PTransportLiteImpl_ReadReduce_BoundarySize_SingleBlock)
{
    const u64 elemSize = 4;
    const u64 maxTaskBytes = (UINT32_MAX / elemSize) * elemSize;

    std::unique_ptr<StreamLite> stream = MakeStream(1);
    std::unique_ptr<P2PTransportLiteImpl> transportLite = MakeTransport(1);
    ReduceIn reduceIn(DataType::FP32, ReduceOp::SUM);

    {
        RmaBufferLite loc(0x200000000ULL, 4096);
        Buffer rmt(0x100000000ULL, 4096);
        transportLite->ReadReduce(loc, rmt, reduceIn, *stream);
        ASSERT_EQ(reduceCalls_.size(), 1U);
        EXPECT_EQ(reduceCalls_[0].size, 4096U);
        reduceCalls_.clear();
    }
    {
        RmaBufferLite loc(0x200000000ULL, maxTaskBytes);
        Buffer rmt(0x100000000ULL, maxTaskBytes);
        transportLite->ReadReduce(loc, rmt, reduceIn, *stream);
        ASSERT_EQ(reduceCalls_.size(), 1U);
        EXPECT_EQ(reduceCalls_[0].size, static_cast<u32>(maxTaskBytes));
        reduceCalls_.clear();
    }
}

TEST_F(P2PTransportLiteImplReadReduceTest, P2PTransportLiteImpl_ReadReduce_InvalidDataType_Throw)
{
    std::unique_ptr<StreamLite> stream = MakeStream(1);
    std::unique_ptr<P2PTransportLiteImpl> transportLite = MakeTransport(1);

    RmaBufferLite loc(0x200000000ULL, 4096);
    Buffer rmt(0x100000000ULL, 4096);
    ReduceIn reduceIn(DataType::INVALID, ReduceOp::SUM);

    EXPECT_THROW(transportLite->ReadReduce(loc, rmt, reduceIn, *stream), InvalidParamsException);
    EXPECT_TRUE(reduceCalls_.empty());
}

TEST_F(P2PTransportLiteImplReadReduceTest, P2PTransportLiteImpl_ReadReduce_ZeroSize_NoTask)
{
    std::unique_ptr<StreamLite> stream = MakeStream(1);
    std::unique_ptr<P2PTransportLiteImpl> transportLite = MakeTransport(1);

    RmaBufferLite loc(0x200000000ULL, 0);
    Buffer rmt(0x100000000ULL, 0);
    ReduceIn reduceIn(DataType::FP32, ReduceOp::SUM);

    transportLite->ReadReduce(loc, rmt, reduceIn, *stream);
    EXPECT_TRUE(reduceCalls_.empty());
}

TEST_F(P2PTransportLiteImplReadReduceTest, P2PTransportLiteImpl_ReadReduce_Int16Element_SplitAligned)
{
    const u64 elemSize = 2;
    const u64 maxTaskBytes = (UINT32_MAX / elemSize) * elemSize;
    const u64 totalSize = 8ULL * 1024 * 1024 * 1024;

    std::unique_ptr<StreamLite> stream = MakeStream(1);
    std::unique_ptr<P2PTransportLiteImpl> transportLite = MakeTransport(1);

    RmaBufferLite loc(0x200000000ULL, totalSize);
    Buffer rmt(0x100000000ULL, totalSize);
    ReduceIn reduceIn(DataType::INT16, ReduceOp::SUM);

    transportLite->ReadReduce(loc, rmt, reduceIn, *stream);

    ASSERT_EQ(reduceCalls_.size(), 3U);
    EXPECT_EQ(reduceCalls_[0].size, static_cast<u32>(maxTaskBytes));
    EXPECT_EQ(reduceCalls_[1].size, static_cast<u32>(maxTaskBytes));
    EXPECT_EQ(reduceCalls_[2].size, static_cast<u32>(totalSize - maxTaskBytes * 2));
}

TEST_F(P2PTransportLiteImplReadReduceTest, P2PTransportLiteImpl_BatchTransfer_ReadReduce_SplitAligned)
{
    const u64 elemSize = 4;
    const u64 maxTaskBytes = (UINT32_MAX / elemSize) * elemSize;
    const u64 blockSize = maxTaskBytes * 2;
    const u64 locBase = 0x200000000ULL;
    const u64 rmtBase = 0x100000000ULL;

    std::unique_ptr<StreamLite> stream = MakeStream(1);
    std::unique_ptr<P2PTransportLiteImpl> transportLite = MakeTransport(1);

    std::vector<RmaBufferLite> loc;
    std::vector<Buffer> rmt;
    std::vector<BaseTransportLiteImpl::TransferOp> transferOp;
    for (u32 i = 0; i < 2; i++) {
        loc.emplace_back(locBase + i * blockSize, blockSize);
        rmt.emplace_back(rmtBase + i * blockSize, blockSize);
        transferOp.push_back({TransferType::READ, ReduceIn(DataType::FP32, ReduceOp::SUM)});
    }

    transportLite->BatchTransfer(loc, rmt, transferOp, *stream);

    ASSERT_EQ(reduceCalls_.size(), 4U);
    for (u32 i = 0; i < 2; i++) {
        const u64 srcBase = rmtBase + i * blockSize;
        const u64 dstBase = locBase + i * blockSize;
        EXPECT_EQ(reduceCalls_[2 * i].size, static_cast<u32>(maxTaskBytes));
        EXPECT_EQ(reduceCalls_[2 * i].src, srcBase);
        EXPECT_EQ(reduceCalls_[2 * i].dst, dstBase);
        EXPECT_EQ(reduceCalls_[2 * i + 1].size, static_cast<u32>(maxTaskBytes));
        EXPECT_EQ(reduceCalls_[2 * i + 1].src, srcBase + maxTaskBytes);
        EXPECT_EQ(reduceCalls_[2 * i + 1].dst, dstBase + maxTaskBytes);
    }
}
