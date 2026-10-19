/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#define private public
#include "orion_adapter_rts.h"
#include "p2p_transport_lite_impl.h"
#include "stream_lite.h"
#include "rtsq_a5.h"
#include "sqe.h"
#include "ascend_hal.h"
#include "drv_api_exception.h"
#undef private

using namespace Hccl;

class P2PTransportLiteImplWaitWithTimeoutTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "P2PTransportLiteImplWaitWithTimeoutTest tests set up." << std::endl; }

    static void TearDownTestCase()
    {
        std::cout << "P2PTransportLiteImplWaitWithTimeoutTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqDepth)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(static_cast<u32>(AC_SQE_MAX_CNT)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();
        std::cout << "A Test case in P2PTransportLiteImplWaitWithTimeoutTest SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in P2PTransportLiteImplWaitWithTimeoutTest TearDown" << std::endl;
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

    // 按 P2PTransportLiteImpl 构造函数的解析格式序列化(参照生产端 P2PTransport::GetUniqueId)：
    // [type][notifyNum][bufferNum][notifyUniqueIds][rmtNotifyUniqueIds][rmtBufferUniqueIds]。
    std::vector<char> BuildP2PTransportUniqueId(u32 devPhyId)
    {
        u32 transportType = 0; // 构造函数读后未使用
        u32 notifyNum = 1;
        u32 bufferNum = 0; // bufferNum=0 时 ParseRmtBufferVec 直接返回，无需远端 buffer 数据
        std::vector<char> notifyUniqueId = GetNotifyUniqueId(0, devPhyId);
        // 远端 notify DTO，字段与 RmtP2PNotifyLite(addr/size/id) 对应
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

    u8 mockSq[64];
};

TEST_F(P2PTransportLiteImplWaitWithTimeoutTest, P2PTransportLiteImpl_WaitWithTimeout_Normal_Success)
{
    u32 fakeStreamId = 1;
    u32 fakeSqId = 1;
    u32 fakeDevPhyId = 1;

    BinaryStream liteBinaryStream;
    liteBinaryStream << fakeStreamId;
    liteBinaryStream << fakeSqId;
    liteBinaryStream << fakeDevPhyId;
    std::vector<char> uniqueId{};
    liteBinaryStream.Dump(uniqueId);

    StreamLite stream(uniqueId);
    stream.rtsq = std::make_unique<RtsqA5>(fakeDevPhyId, fakeStreamId, fakeSqId);
    RtsqA5& rtsq = *static_cast<RtsqA5*>(stream.rtsq.get());

    std::vector<char> p2pUniqueId = BuildP2PTransportUniqueId(fakeDevPhyId);

    auto callback = [](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam) {
        (void)streamId;
        (void)taskId;
        (void)taskParam;
    };
    P2PTransportLiteImpl transportLite(p2pUniqueId, callback);

    u32 timeout = 1800;
    EXPECT_NO_THROW(transportLite.WaitWithTimeout(0, stream, timeout));
}

TEST_F(P2PTransportLiteImplWaitWithTimeoutTest, P2PTransportLiteImpl_WaitWithTimeout_ZeroTimeout_Success)
{
    u32 fakeStreamId = 1;
    u32 fakeSqId = 1;
    u32 fakeDevPhyId = 1;

    BinaryStream liteBinaryStream;
    liteBinaryStream << fakeStreamId;
    liteBinaryStream << fakeSqId;
    liteBinaryStream << fakeDevPhyId;
    std::vector<char> uniqueId{};
    liteBinaryStream.Dump(uniqueId);

    StreamLite stream(uniqueId);
    stream.rtsq = std::make_unique<RtsqA5>(fakeDevPhyId, fakeStreamId, fakeSqId);
    RtsqA5& rtsq = *static_cast<RtsqA5*>(stream.rtsq.get());

    std::vector<char> p2pUniqueId = BuildP2PTransportUniqueId(fakeDevPhyId);

    auto callback = [](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam) {
        (void)streamId;
        (void)taskId;
        (void)taskParam;
    };
    P2PTransportLiteImpl transportLite(p2pUniqueId, callback);

    u32 timeout = 0;
    EXPECT_NO_THROW(transportLite.WaitWithTimeout(0, stream, timeout));
}

TEST_F(P2PTransportLiteImplWaitWithTimeoutTest, P2PTransportLiteImpl_WaitWithTimeout_MaxTimeout_Success)
{
    u32 fakeStreamId = 1;
    u32 fakeSqId = 1;
    u32 fakeDevPhyId = 1;

    BinaryStream liteBinaryStream;
    liteBinaryStream << fakeStreamId;
    liteBinaryStream << fakeSqId;
    liteBinaryStream << fakeDevPhyId;
    std::vector<char> uniqueId{};
    liteBinaryStream.Dump(uniqueId);

    StreamLite stream(uniqueId);
    stream.rtsq = std::make_unique<RtsqA5>(fakeDevPhyId, fakeStreamId, fakeSqId);
    RtsqA5& rtsq = *static_cast<RtsqA5*>(stream.rtsq.get());

    std::vector<char> p2pUniqueId = BuildP2PTransportUniqueId(fakeDevPhyId);

    auto callback = [](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam) {
        (void)streamId;
        (void)taskId;
        (void)taskParam;
    };
    P2PTransportLiteImpl transportLite(p2pUniqueId, callback);

    u32 timeout = 0xFFFFFFFF;
    EXPECT_NO_THROW(transportLite.WaitWithTimeout(0, stream, timeout));
}
