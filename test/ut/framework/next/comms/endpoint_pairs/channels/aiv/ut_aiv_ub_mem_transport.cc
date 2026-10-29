/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#define private public
#include "channels/aiv/aiv_ub_mem_transport.h"
#include "p2p_enable_manager.h"
#include "runtime_api_exception.h"
#include "socket_exception.h"

using namespace hcomm;

static void StubSocketSendAsyncNoop(Hccl::Socket* self, const void* sendBuf, u32 size)
{
    (void)self;
    (void)sendBuf;
    (void)size;
}

static void StubSocketRecvAsyncNoop(Hccl::Socket* self, void* recvBuf, u32 size)
{
    (void)self;
    (void)recvBuf;
    (void)size;
}

// WaitP2PEnabled 的 isEnabled 出参由全局开关控制：false 表示驱动侧尚未完成使能，true 表示已完成。
// 引用出参必须通过 invoke 桩原生传递，不能使用 outBoundP（指针约束），否则 any_cast 断言崩溃
static bool g_p2pEnabled = false;
static HcclResult StubWaitP2PEnabled(
    Hccl::P2PEnableManager* self, uint32_t localDeviceLogicID, uint32_t remoteDevicePhysicID, bool& isEnabled)
{
    (void)self;
    (void)localDeviceLogicID;
    (void)remoteDevicePhysicID;
    isEnabled = g_p2pEnabled;
    return HCCL_SUCCESS;
}

// 夹具名与 ut_AivUbMemTransport_API_test.cc 中的同名夹具区分：两者编进同一可执行文件，
// 同名同命名空间但成员布局不同会构成 ODR 冲突（弱符号任取一份，构造/析构布局错位导致堆破坏）
class AivUbMemTransportStatusTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        std::cout << "AivUbMemTransportStatusTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        std::cout << "AivUbMemTransportStatusTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in AivUbMemTransportStatusTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in AivUbMemTransportStatusTest TearDown" << std::endl;
    }

    std::vector<Hccl::LocalRmaBuffer*> localRmaBufferVec_{};
};

TEST_F(AivUbMemTransportStatusTest, St_GetStatus_When_SOCKET_OK_Expect_Success)
{
    Hccl::Socket* fakeSocket = reinterpret_cast<Hccl::Socket*>(0x1);
    HcommChannelDesc desc{};
    AivUbMemTransport aivTransport(localRmaBufferVec_, fakeSocket, desc);

    aivTransport.baseStatus_ = Hccl::TransportStatus::SOCKET_OK;
    aivTransport.aivUbStatus_ = hcomm::AivUbMemTransport::AivUbMemTransportStatus::SOCKET_OK;

    MOCKER_CPP(&AivUbMemTransport::RmtBufferUnpackProc).stubs();
    MOCKER_CPP(&Hccl::Socket::SendAsync, void(Hccl::Socket::*)(const void*, u32))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(StubSocketSendAsyncNoop));
    MOCKER_CPP(&Hccl::Socket::RecvAsync, void(Hccl::Socket::*)(void*, u32))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(StubSocketRecvAsyncNoop));

    EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::SOCKET_OK);
    EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::SEND_DATA_SIZE);

    EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::SOCKET_OK);
    EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::RECV_DATA_SIZE);

    EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::SOCKET_OK);
    EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::SEND_MEM_INFO);

    EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::SOCKET_OK);
    EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::RECV_MEM_INFO);

    EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::SOCKET_OK);
    EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::RECV_MEM_FIN);

    EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::READY);
    EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::READY);
}

TEST_F(AivUbMemTransportStatusTest, Ut_GetStatus_When_RecvData_Fail_Expect_Status_Invalid)
{
    Hccl::Socket* fakeSocket = reinterpret_cast<Hccl::Socket*>(0x1);
    HcommChannelDesc desc{};
    AivUbMemTransport aivTransport(localRmaBufferVec_, fakeSocket, desc);

    aivTransport.baseStatus_ = Hccl::TransportStatus::SOCKET_OK;
    aivTransport.aivUbStatus_ = hcomm::AivUbMemTransport::AivUbMemTransportStatus::RECV_MEM_INFO;

    MOCKER_CPP(&AivUbMemTransport::RmtBufferUnpackProc).stubs().will(throws(Hccl::RuntimeApiException("test_fail")));

    EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::INVALID);
    EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::RECV_MEM_FIN);
}

TEST_F(AivUbMemTransportStatusTest, Ut_GetStatus_When_SocketReady_Fail_Expect_Status_Invalid)
{
    Hccl::Socket* fakeSocket = reinterpret_cast<Hccl::Socket*>(0x1);
    HcommChannelDesc desc{};
    AivUbMemTransport aivTransport(localRmaBufferVec_, fakeSocket, desc);

    aivTransport.baseStatus_ = Hccl::TransportStatus::SOCKET_OK;
    aivTransport.aivUbStatus_ = hcomm::AivUbMemTransport::AivUbMemTransportStatus::RECV_MEM_INFO;

    MOCKER_CPP(&Hccl::Socket::GetAsyncStatus).stubs().will(throws(Hccl::SocketException("test_fail")));

    EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::INVALID);
}

// PCIE 全流程：INIT 首轮触发 enable 一次，未使能时停留 P2P_ENABLING，使能后进入 socket 建链，析构对称 disable 一次
TEST_F(AivUbMemTransportStatusTest, Ut_GetStatus_When_PCIE_Expect_EnableOnceAndSymmetricDisable)
{
    Hccl::Socket* fakeSocket = reinterpret_cast<Hccl::Socket*>(0x1);
    HcommChannelDesc desc{};
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_PCIE;
    desc.remoteEndpoint.loc.device.devPhyId = 20; // stub 环境本地物理 id 固定为 1，非自环

    Hccl::SocketStatus socketStatusInit = Hccl::SocketStatus::INIT;
    MOCKER_CPP(&Hccl::Socket::GetAsyncStatus).stubs().will(returnValue(socketStatusInit));

    MOCKER_CPP(&Hccl::P2PEnableManager::EnableP2P, HcclResult(Hccl::P2PEnableManager::*)(std::vector<uint32_t>))
        .expects(once())
        .will(returnValue(HCCL_SUCCESS)); // INIT 首轮触发一次，后续轮询不重复下发
    g_p2pEnabled = false;
    MOCKER_CPP(
        &Hccl::P2PEnableManager::WaitP2PEnabled, HcclResult(Hccl::P2PEnableManager::*)(uint32_t, uint32_t, bool&))
        .stubs()
        .will(invoke(StubWaitP2PEnabled));
    MOCKER_CPP(
        &Hccl::P2PEnableManager::DisableP2P, HcclResult(Hccl::P2PEnableManager::*)(uint32_t, std::vector<uint32_t>))
        .expects(once())
        .will(returnValue(HCCL_SUCCESS)); // 析构对称 disable 恰好一次

    {
        AivUbMemTransport aivTransport(localRmaBufferVec_, fakeSocket, desc);
        aivTransport.baseStatus_ = Hccl::TransportStatus::INIT;
        aivTransport.aivUbStatus_ = hcomm::AivUbMemTransport::AivUbMemTransportStatus::INIT;

        // 第 1 轮：触发 enable p2p 后停留 P2P_ENABLING，socket 未就绪
        EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::INIT);
        EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::P2P_ENABLING);
        EXPECT_TRUE(aivTransport.p2pEnableStarted_);

        // 第 2 轮：驱动侧尚未完成使能，停留 P2P_ENABLING
        EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::INIT);
        EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::P2P_ENABLING);

        // 第 3 轮：p2p 使能完成进入 socket 建链，socket 未就绪停留
        g_p2pEnabled = true;
        EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::INIT);
        EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::SOCKET_OK);
    } // 析构：p2pEnableStarted_ 为 true，对称执行 disable 恰好一次
}

// 非 PCIE（HCCS）链路：不触发 enable p2p，直接进入 socket 建链，析构不执行 disable
TEST_F(AivUbMemTransportStatusTest, Ut_GetStatus_When_NonPCIE_Expect_SkipEnable)
{
    Hccl::Socket* fakeSocket = reinterpret_cast<Hccl::Socket*>(0x1);
    HcommChannelDesc desc{}; // protocol 默认 0 即 HCCS
    desc.remoteEndpoint.loc.device.devPhyId = 20;

    Hccl::SocketStatus socketStatusInit = Hccl::SocketStatus::INIT;
    MOCKER_CPP(&Hccl::Socket::GetAsyncStatus).stubs().will(returnValue(socketStatusInit));

    MOCKER_CPP(&Hccl::P2PEnableManager::EnableP2P, HcclResult(Hccl::P2PEnableManager::*)(std::vector<uint32_t>))
        .expects(never());
    MOCKER_CPP(
        &Hccl::P2PEnableManager::DisableP2P, HcclResult(Hccl::P2PEnableManager::*)(uint32_t, std::vector<uint32_t>))
        .expects(never());

    {
        AivUbMemTransport aivTransport(localRmaBufferVec_, fakeSocket, desc);
        aivTransport.baseStatus_ = Hccl::TransportStatus::INIT;
        aivTransport.aivUbStatus_ = hcomm::AivUbMemTransport::AivUbMemTransportStatus::INIT;

        EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::INIT);
        EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::SOCKET_OK);
        EXPECT_FALSE(aivTransport.p2pEnableStarted_);
    }
}

// 自环链路（对端物理 id 等于本地物理 id）：不支持 enable p2p，直接跳过
TEST_F(AivUbMemTransportStatusTest, Ut_GetStatus_When_SelfLoop_Expect_SkipEnable)
{
    Hccl::Socket* fakeSocket = reinterpret_cast<Hccl::Socket*>(0x1);
    HcommChannelDesc desc{};
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_PCIE;
    desc.remoteEndpoint.loc.device.devPhyId = 1; // stub 环境本地物理 id 固定为 1，构成自环

    Hccl::SocketStatus socketStatusInit = Hccl::SocketStatus::INIT;
    MOCKER_CPP(&Hccl::Socket::GetAsyncStatus).stubs().will(returnValue(socketStatusInit));

    MOCKER_CPP(&Hccl::P2PEnableManager::EnableP2P, HcclResult(Hccl::P2PEnableManager::*)(std::vector<uint32_t>))
        .expects(never());
    MOCKER_CPP(
        &Hccl::P2PEnableManager::DisableP2P, HcclResult(Hccl::P2PEnableManager::*)(uint32_t, std::vector<uint32_t>))
        .expects(never());

    {
        AivUbMemTransport aivTransport(localRmaBufferVec_, fakeSocket, desc);
        aivTransport.baseStatus_ = Hccl::TransportStatus::INIT;
        aivTransport.aivUbStatus_ = hcomm::AivUbMemTransport::AivUbMemTransportStatus::INIT;

        EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::INIT);
        EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::SOCKET_OK);
        EXPECT_FALSE(aivTransport.p2pEnableStarted_);
    }
}

// enable p2p 下发失败：状态置 INVALID，不持有引用，析构不执行 disable
TEST_F(AivUbMemTransportStatusTest, Ut_GetStatus_When_EnableP2PFail_Expect_Status_Invalid)
{
    Hccl::Socket* fakeSocket = reinterpret_cast<Hccl::Socket*>(0x1);
    HcommChannelDesc desc{};
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_PCIE;
    desc.remoteEndpoint.loc.device.devPhyId = 20;

    MOCKER_CPP(&Hccl::P2PEnableManager::EnableP2P, HcclResult(Hccl::P2PEnableManager::*)(std::vector<uint32_t>))
        .stubs()
        .will(returnValue(HCCL_E_RUNTIME));
    MOCKER_CPP(
        &Hccl::P2PEnableManager::DisableP2P, HcclResult(Hccl::P2PEnableManager::*)(uint32_t, std::vector<uint32_t>))
        .expects(never()); // enable 失败未增加引用计数，析构不执行 disable

    {
        AivUbMemTransport aivTransport(localRmaBufferVec_, fakeSocket, desc);
        aivTransport.baseStatus_ = Hccl::TransportStatus::INIT;
        aivTransport.aivUbStatus_ = hcomm::AivUbMemTransport::AivUbMemTransportStatus::INIT;

        EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::INVALID);
        EXPECT_FALSE(aivTransport.p2pEnableStarted_);
    }
}
