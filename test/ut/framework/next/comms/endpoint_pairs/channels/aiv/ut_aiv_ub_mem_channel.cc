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
#include "mockcpp/mokc.h"
#include <cstdint>
#include <cstring>
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "aiv_ub_mem_channel.h"
#include "../sockets/socket_mgr.h"
#undef protected
#undef private
#include "hcomm_c_adpt.h"
#include "endpoint.h"

using namespace hcomm;
using namespace Hccl;

namespace {

uint32_t g_listenPort = 0;
uint32_t g_stopListenPort = 0;

HcommResult StubEndpointStartListen(EndpointHandle, uint32_t port, const HcommEndpointListenConfig*)
{
    g_listenPort = port;
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

HcommResult StubEndpointStartListenFail(EndpointHandle, uint32_t port, const HcommEndpointListenConfig*)
{
    g_listenPort = port;
    return static_cast<HcommResult>(HCCL_E_INTERNAL);
}

HcommResult StubEndpointStopListen(EndpointHandle, uint32_t port)
{
    g_stopListenPort = port;
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

class StubEndpointForAivUbMemChannel : public Endpoint {
public:
    explicit StubEndpointForAivUbMemChannel(const EndpointDesc& desc) : Endpoint(desc) {}

    HcclResult Init() override { return HCCL_SUCCESS; }
    LocalRegedMemMgr* GetLocalRegMemMgr() override { return nullptr; }
    void* GetRdmaHandle() override { return nullptr; }
    bool IsCtxHandleValid() const override { return false; }
};

Socket MakeFakeSocket()
{
    return Socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
}

} // namespace

class AivUbMemChannelTest : public testing::Test {
protected:
    static void TearDownTestCase() { GlobalMockObject::verify(); }

    void TearDown() override { GlobalMockObject::verify(); }

    HcommChannelDesc MakeDefaultDesc()
    {
        HcommChannelDesc desc;
        (void)std::memset(&desc, 0, sizeof(desc));
        return desc;
    }

    AivUbMemChannel* CreateChannel()
    {
        EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
        HcommChannelDesc desc = MakeDefaultDesc();
        return new AivUbMemChannel(ep, desc);
    }

    void DestroyChannel(AivUbMemChannel* ch) { delete ch; }
};

// ===================== 已有用例：Clean / Resume =====================

// Clean: 调用应返回 HCCL_SUCCESS，无论调用次数
TEST_F(AivUbMemChannelTest, Ut_CleanWhenCalledExpectHCCLSuccess)
{
    AivUbMemChannel* ch = CreateChannel();
    EXPECT_EQ(ch->Clean(), HCCL_SUCCESS);
    // 重复调用覆盖相同行为
    EXPECT_EQ(ch->Clean(), HCCL_SUCCESS);
    DestroyChannel(ch);
}

// Resume: 调用应返回 HCCL_SUCCESS，无论调用次数
TEST_F(AivUbMemChannelTest, Ut_ResumeWhenCalledExpectHCCLSuccess)
{
    AivUbMemChannel* ch = CreateChannel();
    EXPECT_EQ(ch->Resume(), HCCL_SUCCESS);
    // 重复调用覆盖相同行为
    EXPECT_EQ(ch->Resume(), HCCL_SUCCESS);
    DestroyChannel(ch);
}

// ===================== BuildSocket UT =====================

// BuildSocket 当 socket_ 已预创建（非 nullptr）时直接返回 SUCCESS
TEST_F(AivUbMemChannelTest, Ut_BuildSocket_When_PreCreatedSocket_Expect_ReturnSuccess)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    HcommChannelDesc desc = MakeDefaultDesc();
    AivUbMemChannel ch(ep, desc);
    Socket socket = MakeFakeSocket();
    ch.socket_ = &socket;

    // SocketMgr::GetSocket 不应被调用
    MOCKER_CPP(&SocketMgr::GetSocket).expects(never());

    EXPECT_EQ(ch.BuildSocket(), HCCL_SUCCESS);
    EXPECT_EQ(ch.socket_, &socket);
    ch.socket_ = nullptr;
}

// BuildSocket 当 socket_ 为 nullptr 时通过 SocketMgr 创建 Socket
TEST_F(AivUbMemChannelTest, Ut_BuildSocket_When_SocketNull_Expect_CreateBySocketMgr)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 60001;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.channelName = "test_channel";
    AivUbMemChannel ch(ep, desc);
    Socket socket = MakeFakeSocket();
    ch.socket_ = nullptr;
    ch.localEp_.loc.device.devPhyId = 3;

    MOCKER_CPP(&SocketMgr::GetSocket).stubs().with(mockcpp::any(), outBound(&socket)).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(ch.BuildSocket(), HCCL_SUCCESS);
    EXPECT_EQ(ch.socket_, &socket);
    ASSERT_NE(ch.socketConfigHolder_, nullptr);
    EXPECT_EQ(ch.socketConfig_, ch.socketConfigHolder_.get());
    EXPECT_EQ(ch.socketConfig_->listeningPort, desc.port);
    EXPECT_EQ(ch.socketConfig_->GetRole(), SocketRole::SERVER);
    ch.socket_ = nullptr;
}

// BuildSocket 当 port=0 时使用默认端口 DEFAULT_LISTENING_PORT(60001)
TEST_F(AivUbMemChannelTest, Ut_BuildSocket_When_PortZero_Expect_UseDefaultPort)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 0;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.channelName = "test";
    AivUbMemChannel ch(ep, desc);
    Socket socket = MakeFakeSocket();
    ch.socket_ = nullptr;
    ch.localEp_.loc.device.devPhyId = 3;

    MOCKER_CPP(&SocketMgr::GetSocket).stubs().with(mockcpp::any(), outBound(&socket)).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(ch.BuildSocket(), HCCL_SUCCESS);
    ASSERT_NE(ch.socketConfigHolder_, nullptr);
    EXPECT_EQ(ch.socketConfig_->listeningPort, 60001);
    ch.socket_ = nullptr;
}

// BuildSocket 当 channelName=nullptr 时使用默认标签 "AUTOMATIC_SOCKET_TAG"
TEST_F(AivUbMemChannelTest, Ut_BuildSocket_When_ChannelNameNull_Expect_UseDefaultTag)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 60001;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.channelName = nullptr;
    AivUbMemChannel ch(ep, desc);
    Socket socket = MakeFakeSocket();
    ch.socket_ = nullptr;
    ch.localEp_.loc.device.devPhyId = 3;

    MOCKER_CPP(&SocketMgr::GetSocket).stubs().with(mockcpp::any(), outBound(&socket)).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(ch.BuildSocket(), HCCL_SUCCESS);
    ASSERT_NE(ch.socketConfigHolder_, nullptr);
    // channelName 为 nullptr 时使用默认标签
    EXPECT_EQ(ch.socketConfig_->tag, "AUTOMATIC_SOCKET_TAG");
    ch.socket_ = nullptr;
}

// BuildSocket 当 role=HCOMM_SOCKET_ROLE_RESERVED 时设 noRankId=true
TEST_F(AivUbMemChannelTest, Ut_BuildSocket_When_RoleReserved_Expect_SetNoRankIdTrue)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 60001;
    desc.role = HCOMM_SOCKET_ROLE_RESERVED;
    desc.channelName = "test";
    AivUbMemChannel ch(ep, desc);
    Socket socket = MakeFakeSocket();
    ch.socket_ = nullptr;
    ch.localEp_.loc.device.devPhyId = 3;

    MOCKER_CPP(&SocketMgr::GetSocket).stubs().with(mockcpp::any(), outBound(&socket)).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(ch.BuildSocket(), HCCL_SUCCESS);
    ASSERT_NE(ch.socketConfigHolder_, nullptr);
    EXPECT_TRUE(ch.socketConfig_->noRankId);
    ch.socket_ = nullptr;
}

// BuildSocket 当 SocketMgr::GetSocket 返回失败时透传错误码
TEST_F(AivUbMemChannelTest, Ut_BuildSocket_When_GetSocketFail_Expect_ReturnError)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 60001;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.channelName = "test";
    AivUbMemChannel ch(ep, desc);
    ch.socket_ = nullptr;
    ch.localEp_.loc.device.devPhyId = 3;

    MOCKER_CPP(&SocketMgr::GetSocket).stubs().will(returnValue(HCCL_E_INTERNAL));

    EXPECT_EQ(ch.BuildSocket(), HCCL_E_INTERNAL);
    EXPECT_EQ(ch.socket_, nullptr);
}

// ===================== StartListen UT =====================

// StartListen 当 role=SERVER 时调用 HcommEndpointStartListen
TEST_F(AivUbMemChannelTest, Ut_StartListen_When_RoleServer_Expect_CallEndpointStartListen)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 60001;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUbMemChannel ch(ep, desc);
    g_listenPort = 0;

    MOCKER(HcommEndpointStartListen).stubs().will(invoke(StubEndpointStartListen));

    EXPECT_EQ(ch.StartListen(), HCCL_SUCCESS);
    EXPECT_EQ(g_listenPort, 60001);
}

// StartListen 当 role=CLIENT 时不调用 HcommEndpointStartListen 直接返回 SUCCESS
TEST_F(AivUbMemChannelTest, Ut_StartListen_When_RoleClient_Expect_ReturnSuccessWithoutListen)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 60001;
    desc.role = HCOMM_SOCKET_ROLE_CLIENT;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUbMemChannel ch(ep, desc);
    g_listenPort = 0;

    MOCKER(HcommEndpointStartListen).expects(never());

    EXPECT_EQ(ch.StartListen(), HCCL_SUCCESS);
    EXPECT_EQ(g_listenPort, 0);
}

// StartListen 当 port=0 且 role=SERVER 时使用默认端口 60001
TEST_F(AivUbMemChannelTest, Ut_StartListen_When_PortZero_Expect_UseDefaultPort)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 0;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUbMemChannel ch(ep, desc);
    g_listenPort = 0;

    MOCKER(HcommEndpointStartListen).stubs().will(invoke(StubEndpointStartListen));

    EXPECT_EQ(ch.StartListen(), HCCL_SUCCESS);
    EXPECT_EQ(g_listenPort, 60001);
}

// StartListen 当 role=RESERVED 时不调用 HcommEndpointStartListen
TEST_F(AivUbMemChannelTest, Ut_StartListen_When_RoleReserved_Expect_ReturnSuccessWithoutListen)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 60001;
    desc.role = HCOMM_SOCKET_ROLE_RESERVED;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUbMemChannel ch(ep, desc);
    g_listenPort = 0;

    MOCKER(HcommEndpointStartListen).expects(never());

    EXPECT_EQ(ch.StartListen(), HCCL_SUCCESS);
    EXPECT_EQ(g_listenPort, 0);
}

// StartListen 当 HcommEndpointStartListen 返回失败时透传错误码
TEST_F(AivUbMemChannelTest, Ut_StartListen_When_StartListenFail_Expect_ReturnError)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 60001;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUbMemChannel ch(ep, desc);
    g_listenPort = 0;

    MOCKER(HcommEndpointStartListen).stubs().will(invoke(StubEndpointStartListenFail));

    EXPECT_EQ(ch.StartListen(), HCCL_E_INTERNAL);
}

// ===================== StopListen UT =====================

// StopListen 当 role=SERVER 且 serverListened_=true 时调用 HcommEndpointStopListen 并重置 serverListened_
TEST_F(AivUbMemChannelTest, Ut_StopListen_When_RoleServerAndListened_Expect_CallEndpointStopListen)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 60011;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUbMemChannel ch(ep, desc);
    ch.serverListened_ = true;
    g_stopListenPort = 0;

    MOCKER(HcommEndpointStopListen).stubs().will(invoke(StubEndpointStopListen));

    ch.StopListen();
    EXPECT_EQ(g_stopListenPort, 60011);
    EXPECT_FALSE(ch.serverListened_);
}

// StopListen 当 port=0 且 role=SERVER 时使用默认端口 DEFAULT_LISTENING_PORT(60001)
TEST_F(AivUbMemChannelTest, Ut_StopListen_When_PortZero_Expect_UseDefaultPort)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 0;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUbMemChannel ch(ep, desc);
    ch.serverListened_ = true;
    g_stopListenPort = 0;

    MOCKER(HcommEndpointStopListen).stubs().will(invoke(StubEndpointStopListen));

    ch.StopListen();
    EXPECT_EQ(g_stopListenPort, 60001);
    EXPECT_FALSE(ch.serverListened_);
}

// StopListen 当 serverListened_=false 时不调用 HcommEndpointStopListen
TEST_F(AivUbMemChannelTest, Ut_StopListen_When_NotListened_Expect_NotCallStopListen)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 60001;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUbMemChannel ch(ep, desc);
    ch.serverListened_ = false;

    MOCKER(HcommEndpointStopListen).expects(never());

    ch.StopListen();
    EXPECT_FALSE(ch.serverListened_);
}

// ===================== Init UT =====================

// Init 按 ParseInputParam -> StartListen -> BuildSocket -> BuildTransport 顺序执行
TEST_F(AivUbMemChannelTest, Ut_Init_When_Normal_Expect_ExecuteInCorrectOrder)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.port = 60001;
    desc.channelName = "test";
    AivUbMemChannel ch(ep, desc);

    MOCKER_CPP(&AivUbMemChannel::ParseInputParam, HcclResult(AivUbMemChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AivUbMemChannel::StartListen, HcclResult(AivUbMemChannel::*)())
        .expects(once())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AivUbMemChannel::BuildSocket, HcclResult(AivUbMemChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AivUbMemChannel::BuildTransport, HcclResult(AivUbMemChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(ch.Init(), HCCL_SUCCESS);
}

// Init 当 channelDesc_.socket 预创建时走预创建路径（BuildSocket 跳过）
TEST_F(AivUbMemChannelTest, Ut_Init_When_PreCreatedSocket_Expect_BuildSocketSkipCreate)
{
    EndpointDesc local{};
    local.loc.device.devPhyId = 3;
    StubEndpointForAivUbMemChannel endpoint(local);
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.port = 60001;
    Socket socket = MakeFakeSocket();
    desc.socket = static_cast<HcommSocket>(&socket);
    AivUbMemChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(HcommEndpointStartListen).stubs().will(invoke(StubEndpointStartListen));
    // SocketMgr::GetSocket 不应被调用（预创建路径跳过 BuildSocket）
    MOCKER_CPP(&SocketMgr::GetSocket).expects(never());
    // BuildTransport mock 为 SUCCESS（隔离验证 BuildSocket 被跳过）
    MOCKER_CPP(&AivUbMemChannel::BuildTransport, HcclResult(AivUbMemChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(ch.Init(), HCCL_SUCCESS);
    // socket_ 应保持为预创建的值
    EXPECT_EQ(ch.socket_, &socket);
}

// ===================== ParseInputParam UT =====================

// ParseInputParam 正确赋值 remoteEp_ 和 devicePhyId_
TEST_F(AivUbMemChannelTest, Ut_ParseInputParam_When_Normal_Expect_AssignRemoteEpAndDevicePhyId)
{
    EndpointDesc local{};
    local.loc.device.devPhyId = 7;
    StubEndpointForAivUbMemChannel endpoint(local);
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.remoteEndpoint.loc.device.devPhyId = 99;
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_MEM;
    Socket socket = MakeFakeSocket();
    desc.socket = static_cast<HcommSocket>(&socket);
    AivUbMemChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    EXPECT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_EQ(ch.devicePhyId_, 7);
    EXPECT_EQ(ch.remoteEp_.loc.device.devPhyId, 99);
    EXPECT_EQ(ch.remoteEp_.protocol, COMM_PROTOCOL_UB_MEM);
    EXPECT_EQ(ch.socket_, &socket);
}

// ParseInputParam 当 endpointHandle_ 为 nullptr 时返回 HCCL_E_PTR
TEST_F(AivUbMemChannelTest, Ut_ParseInputParam_When_NullEndpoint_Expect_ReturnEPtr)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    AivUbMemChannel ch(nullptr, desc);

    EXPECT_EQ(ch.ParseInputParam(), HCCL_E_PTR);
}

// ===================== PutSocketIfNeeded UT =====================

// PutSocketIfNeeded: 自创建 socket 路径（channelDesc_.socket == nullptr）→ 调用 SocketMgr::DestroySocket
TEST_F(AivUbMemChannelTest, Ut_PutSocketIfNeeded_When_SelfCreatedSocket_Expect_CallDestroySocket)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.socket = nullptr; // 自创建路径
    AivUbMemChannel ch(ep, desc);
    Socket socket = MakeFakeSocket();
    ch.socket_ = &socket;

    // 构造一个 SocketConfig 使 socketConfig_ 非 nullptr
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData linkData(portType, 0, 1, 0, 0);
    ch.socketConfigHolder_ = std::make_unique<Hccl::SocketConfig>(linkData, 60001, "ut", true);
    ch.socketConfig_ = ch.socketConfigHolder_.get();

    MOCKER_CPP(&SocketMgr::DestroySocket, HcclResult(SocketMgr::*)(const Hccl::SocketConfig&))
        .expects(once())
        .will(returnValue(HCCL_SUCCESS));

    ch.PutSocketIfNeeded();
    EXPECT_EQ(ch.socket_, nullptr);
}

// PutSocketIfNeeded: 预创建 socket 路径（channelDesc_.socket != nullptr）→ 不调用 SocketMgr::DestroySocket
TEST_F(AivUbMemChannelTest, Ut_PutSocketIfNeeded_When_PreCreatedSocket_Expect_NotCallDestroySocket)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    Socket preSocket = MakeFakeSocket();
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.socket = static_cast<HcommSocket>(&preSocket); // 预创建路径
    AivUbMemChannel ch(ep, desc);
    Socket socket = MakeFakeSocket();
    ch.socket_ = &socket;

    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData linkData(portType, 0, 1, 0, 0);
    ch.socketConfigHolder_ = std::make_unique<Hccl::SocketConfig>(linkData, 60001, "ut", true);
    ch.socketConfig_ = ch.socketConfigHolder_.get();

    // DestroySocket 不应被调用
    MOCKER_CPP(&SocketMgr::DestroySocket, HcclResult(SocketMgr::*)(const Hccl::SocketConfig&)).expects(never());

    ch.PutSocketIfNeeded();
    EXPECT_EQ(ch.socket_, nullptr);
}

// PutSocketIfNeeded: socket_ 为 nullptr 时直接返回
TEST_F(AivUbMemChannelTest, Ut_PutSocketIfNeeded_When_SocketNull_Expect_ReturnImmediately)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    HcommChannelDesc desc = MakeDefaultDesc();
    AivUbMemChannel ch(ep, desc);
    ch.socket_ = nullptr;

    // DestroySocket 不应被调用
    MOCKER_CPP(&SocketMgr::DestroySocket, HcclResult(SocketMgr::*)(const Hccl::SocketConfig&)).expects(never());

    ch.PutSocketIfNeeded();
    EXPECT_EQ(ch.socket_, nullptr);
}
