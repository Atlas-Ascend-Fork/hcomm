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

#include <string>
#include "adapter_error_manager_pub.h"
#define private public
#define protected public
#include "hccl_common_v2.h"
#include "preempt_port_manager_v2.h"
#include "socket.h"
#undef private
#undef protected

using namespace Hccl;
using namespace std;

namespace {
// 捕获 RptInputErr 上报内容, 用于断言 EI0019/EI0020 错误码与 reason 信息
std::string g_preemptRptErrorCode;
std::vector<std::string> g_preemptRptValues;

void stubPreemptRptInputErrCapture(std::string error_code, std::vector<std::string> key, std::vector<std::string> value)
{
    g_preemptRptErrorCode = error_code;
    g_preemptRptValues = value;
}
} // namespace

class HcclPreemptPortManagerV2Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclPreemptPortManagerV2Test SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "HcclPreemptPortManagerV2Test TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() {}

    virtual void TearDown() { GlobalMockObject::verify(); }

    PreemptPortManager& ppm = PreemptPortManager::GetInstance(0);
    PreemptPortManager& ppm1 = PreemptPortManager::GetInstance(-1);
    PreemptPortManager& ppm2 = PreemptPortManager::GetInstance(MAX_MODULE_DEVICE_NUM);
};

TEST_F(HcclPreemptPortManagerV2Test, Ut_GetRangeStr_When_InputValue_Expect_NO_THROW)
{
    // when
    std::vector<SocketPortRange> portRange;
    SocketPortRange range = {50000, 50000};
    portRange.push_back(range);

    // check
    EXPECT_NO_THROW(ppm.GetRangeStr(portRange));
}

TEST_F(HcclPreemptPortManagerV2Test, Ut_ListenPreempt_When_InputValue_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&PreemptPortManager::PreemptPortInRange).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.01"};
    std::vector<SocketPortRange> portRangeList;
    SocketPortRange range = {50000, 50005};
    portRangeList.push_back(range);
    u32 usePort = 50000;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    // check
    EXPECT_NO_THROW(ppm.ListenPreempt(listenSocket, portRangeList, usePort));
}

TEST_F(HcclPreemptPortManagerV2Test, Ut_PreemptPortInRange_When_Exist_IP_Expect_Use_Occupied)
{
    // when
    MOCKER_CPP(&Socket::Listen, bool(Socket::*)(u32 & port)).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.01"};
    std::vector<SocketPortRange> portRangeList;
    SocketPortRange range = {50000, 50005};
    portRangeList.push_back(range);
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    u32 usePort = 0;

    // check
    IpPortRef hostPortRef;
    hostPortRef.insert({localIp.GetIpStr().c_str(), std::make_pair(50000, Referenced())});
    ppm.preemptSockets_[HrtNetworkMode::PEER] = hostPortRef;
    EXPECT_NO_THROW(ppm.PreemptPortInRange(listenSocket, HrtNetworkMode::PEER, portRangeList, usePort));
    EXPECT_EQ(usePort, 50000);
}

TEST_F(HcclPreemptPortManagerV2Test, Ut_PreemptPortInRange_When_New_IP_Expect_HCCL_E_PARA)
{
    // when
    MOCKER_CPP(&Socket::Listen, bool(Socket::*)(u32 & port)).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.01"};
    std::vector<SocketPortRange> portRange;
    SocketPortRange range = {50000, 50005};
    portRange.push_back(range);
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    u32 usePort = 0;

    // check
    ppm.preemptSockets_[HrtNetworkMode::PEER] = IpPortRef();
    EXPECT_THROW(
        ppm.PreemptPortInRange(listenSocket, HrtNetworkMode::PEER, portRange, usePort), InvalidParamsException);
}

// 复用已抢占端口重监听失败(设备网卡): 上报 EI0020 并抛 InvalidParamsException
TEST_F(HcclPreemptPortManagerV2Test, Ut_PreemptPortInRange_When_ReusePortListenFail_Expect_ThrowAndReportEI0020)
{
    // when
    MOCKER_CPP(&Socket::Listen, bool(Socket::*)(u32 & port)).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(RptInputErr).stubs().will(invoke(stubPreemptRptInputErrCapture));

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.01"};
    std::vector<SocketPortRange> portRangeList;
    SocketPortRange range = {50000, 50005};
    portRangeList.push_back(range);
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    u32 usePort = 0;

    // 预置: 该 IP 已抢占端口 50000, 走端口复用分支而非轮询抢占
    IpPortRef portRef;
    portRef.insert({localIp.GetIpStr().c_str(), std::make_pair(50000, Referenced())});
    ppm.preemptSockets_[HrtNetworkMode::PEER] = portRef;

    // check
    EXPECT_THROW(
        ppm.PreemptPortInRange(listenSocket, HrtNetworkMode::PEER, portRangeList, usePort), InvalidParamsException);
    EXPECT_EQ(g_preemptRptErrorCode, "EI0020");
    ASSERT_EQ(g_preemptRptValues.size(), 1u);
    EXPECT_NE(g_preemptRptValues[0].find("50000"), std::string::npos);

    ppm.preemptSockets_[HrtNetworkMode::PEER] = IpPortRef();
}

// 端口耗尽且调用方未指定端口(host网卡): 上报 EI0019, reason 报端口范围而非无效端口哨兵值
TEST_F(HcclPreemptPortManagerV2Test, Ut_ReportPortAllocFail_When_Invalid_Port_Expect_ReportRangeNotSentinel)
{
    // when
    MOCKER_CPP(&Socket::Listen, bool(Socket::*)(u32 & port)).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(RptInputErr).stubs().will(invoke(stubPreemptRptInputErrCapture));

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.01"};
    std::vector<SocketPortRange> portRangeList;
    SocketPortRange range = {50001, 50001};
    portRangeList.push_back(range);
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket
        = std::make_shared<Socket>(socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::HOST_NIC_TYPE);
    u32 usePort = HCCL_INVALID_PORT;

    // check
    ppm.preemptSockets_[HrtNetworkMode::PEER] = IpPortRef();
    EXPECT_THROW(
        ppm.PreemptPortInRange(listenSocket, HrtNetworkMode::PEER, portRangeList, usePort), InvalidParamsException);
    EXPECT_EQ(g_preemptRptErrorCode, "EI0019");
    ASSERT_EQ(g_preemptRptValues.size(), 1u);
    EXPECT_NE(g_preemptRptValues[0].find("no available port in range"), std::string::npos);
    EXPECT_NE(g_preemptRptValues[0].find("[50001, 50001]"), std::string::npos);
    EXPECT_EQ(g_preemptRptValues[0].find("65536"), std::string::npos);

    ppm.preemptSockets_[HrtNetworkMode::PEER] = IpPortRef();
}

// 端口耗尽且调用方已指定端口(host网卡): 上报 EI0019, reason 报实际指定端口
TEST_F(HcclPreemptPortManagerV2Test, Ut_ReportPortAllocFail_When_Specific_Port_Expect_ReportPort)
{
    // when
    MOCKER_CPP(&Socket::Listen, bool(Socket::*)(u32 & port)).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(RptInputErr).stubs().will(invoke(stubPreemptRptInputErrCapture));

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.01"};
    std::vector<SocketPortRange> portRangeList;
    SocketPortRange range = {50001, 50001};
    portRangeList.push_back(range);
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket
        = std::make_shared<Socket>(socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::HOST_NIC_TYPE);
    u32 usePort = 50001;

    // check
    ppm.preemptSockets_[HrtNetworkMode::PEER] = IpPortRef();
    EXPECT_THROW(
        ppm.PreemptPortInRange(listenSocket, HrtNetworkMode::PEER, portRangeList, usePort), InvalidParamsException);
    EXPECT_EQ(g_preemptRptErrorCode, "EI0019");
    ASSERT_EQ(g_preemptRptValues.size(), 1u);
    EXPECT_NE(g_preemptRptValues[0].find("port 50001"), std::string::npos);

    ppm.preemptSockets_[HrtNetworkMode::PEER] = IpPortRef();
}

TEST_F(HcclPreemptPortManagerV2Test, Ut_IsAlreadyListening_When_Ref_0_Expect_false)
{
    // then
    IpPortRef hostPortRef;
    hostPortRef.insert({"10.10.10.02", std::make_pair(5000, Referenced())});

    // check
    EXPECT_EQ(ppm.IsAlreadyListening(hostPortRef, "10.10.10.02", 5000), false);
}

TEST_F(HcclPreemptPortManagerV2Test, Ut_IsAlreadyListening_When_Right_Expect_true)
{
    // then
    Referenced ref;
    ref.Ref();
    IpPortRef hostPortRef;
    hostPortRef.insert({"10.10.10.02", std::make_pair(5000, ref)});

    // check
    EXPECT_EQ(ppm.IsAlreadyListening(hostPortRef, "10.10.10.02", 5000), true);
}

TEST_F(HcclPreemptPortManagerV2Test, Ut_ReleasePreempt_When_InputValue_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&PreemptPortManager::IsAlreadyListening).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.1"};
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        socketHandle, localIp, 5000, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    IpPortRef hostPortRef;
    Referenced ref;
    ref.refCount = 1;
    hostPortRef.insert({localIp.GetIpStr().c_str(), std::make_pair(5000, ref)});

    // check
    EXPECT_NO_THROW(ppm.ReleasePreempt(hostPortRef, listenSocket, HrtNetworkMode::PEER));
    EXPECT_EQ(hostPortRef.find(localIp.GetIpStr().c_str()), hostPortRef.end());
}

TEST_F(HcclPreemptPortManagerV2Test, Ut_ReleasePreempt_When_Ref_Count_2_Expect_NO_ERASE)
{
    // when
    MOCKER_CPP(&PreemptPortManager::IsAlreadyListening).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.1"};
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        socketHandle, localIp, 5000, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    IpPortRef hostPortRef;
    Referenced ref;
    ref.refCount = 2;
    hostPortRef.insert({localIp.GetIpStr().c_str(), std::make_pair(5000, ref)});

    // check
    EXPECT_NO_THROW(ppm.ReleasePreempt(hostPortRef, listenSocket, HrtNetworkMode::PEER));
    auto it = hostPortRef.equal_range(localIp.GetIpStr().c_str());
    Referenced& outRef = it.first->second.second;
    EXPECT_EQ(outRef.refCount, 1);

    // check
    IpPortRef multiPortRef;
    Referenced ref1;
    ref1.refCount = 1;
    Referenced ref2;
    ref2.refCount = 1;
    multiPortRef.insert({localIp.GetIpStr().c_str(), std::make_pair(5000, ref1)});
    multiPortRef.insert({localIp.GetIpStr().c_str(), std::make_pair(5001, ref2)});

    std::shared_ptr<Socket> listenSocket2 = std::make_shared<Socket>(
        socketHandle, localIp, 5001, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    EXPECT_NO_THROW(ppm.ReleasePreempt(multiPortRef, listenSocket2, HrtNetworkMode::PEER));

    // 5001应被删除，5000应保留
    auto rangeIt = multiPortRef.equal_range(localIp.GetIpStr().c_str());
    int count = 0;
    u32 remainingPort = 0;
    for (auto rangeEntry = rangeIt.first; rangeEntry != rangeIt.second; ++rangeEntry) {
        count++;
        remainingPort = rangeEntry->second.first;
    }
    EXPECT_EQ(count, 1);
    EXPECT_EQ(remainingPort, 5000u);
}

TEST_F(HcclPreemptPortManagerV2Test, Ut_ReleasePreempt_When_Ref_Count_ERROR_Expect_THROW)
{
    // when
    MOCKER_CPP(&PreemptPortManager::IsAlreadyListening).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.1"};
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        socketHandle, localIp, 5000, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    IpPortRef hostPortRef;
    Referenced ref;
    ref.refCount = 0;
    hostPortRef.insert({localIp.GetIpStr().c_str(), std::make_pair(5000, ref)});

    // check
    EXPECT_THROW(ppm.ReleasePreempt(hostPortRef, listenSocket, HrtNetworkMode::PEER), InvalidParamsException);
}

TEST_F(HcclPreemptPortManagerV2Test, Ut_Release_When_InputValue_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&PreemptPortManager::ReleasePreempt).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.1"};
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    // check
    EXPECT_NO_THROW(ppm.Release(listenSocket));
}
