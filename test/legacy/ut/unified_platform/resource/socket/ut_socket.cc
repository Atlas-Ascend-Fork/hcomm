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
#define private public
#include "socket.h"
#include "hccp_common.h"
#include "null_ptr_exception.h"
#include "socket_exception.h"
#undef private

using namespace Hccl;
class SocketTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Socket tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "Socket tests tear down." << std::endl; }

    virtual void SetUp()
    {
        socketServer
            = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
        socketClient
            = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
        std::cout << "A Test case in Socket SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete socketServer;
        delete socketClient;
        std::cout << "A Test case in Socket TearDown" << std::endl;
    }
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    std::string tag = "testxxxxx";
    Socket* socketServer;
    Socket* socketClient;
    FdHandle fakeFdHandle = (void*)100;
};

TEST_F(SocketTest, listen_stop_listen_ok)
{
    // Given
    MOCKER(HrtRaSocketListenOneStart).stubs().with(mockcpp::any(), mockcpp::any());
    MOCKER(HrtRaSocketListenOneStop).stubs().with(mockcpp::any());
    // when
    socketServer->Listen();

    socketServer->StopListen();
}

TEST_F(SocketTest, server_connect_ok_then_close)
{
    // Given
    int fakeFdStatus = SOCKET_CONNECTED;
    RaSocketFdHandleParam fakeParam(fakeFdHandle, fakeFdStatus, remoteIp);

    MOCKER(HrtRaBlockGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParam));

    // when
    socketServer->Connect();
    auto status = socketServer->GetStatus();
    // then
    EXPECT_EQ(status, SocketStatus::OK);

    MOCKER(HrtRaSocketCloseOne).stubs().with(mockcpp::any());
    socketServer->Close();
}

TEST_F(SocketTest, server_connect_timeout)
{
    // Given
    int fakeFdStatus = SOCKET_CONNECT_TIMEOUT;
    RaSocketFdHandleParam fakeParam(fakeFdHandle, fakeFdStatus, remoteIp);

    MOCKER(HrtRaBlockGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParam));

    // when
    socketServer->Connect();

    auto status = socketServer->GetStatus();
    // then
    EXPECT_EQ(status, SocketStatus::TIMEOUT);
}

TEST_F(SocketTest, server_connect_connecting)
{
    // Given
    int fakeFdStatus = SOCKET_CONNECTING;
    RaSocketFdHandleParam fakeParam(fakeFdHandle, fakeFdStatus, remoteIp);

    MOCKER(HrtRaBlockGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParam));

    // when
    socketServer->Connect();

    auto status = socketServer->GetStatus();
    // then
    EXPECT_EQ(status, SocketStatus::CONNECTING);
}

TEST_F(SocketTest, server_connect_init)
{
    // Given
    int fakeFdStatus = SOCKET_NOT_CONNECTED;
    RaSocketFdHandleParam fakeParam(fakeFdHandle, fakeFdStatus, remoteIp);

    MOCKER(HrtRaBlockGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParam));

    // when
    socketServer->Connect();

    auto status = socketServer->GetStatus();
    // then
    EXPECT_EQ(status, SocketStatus::INIT);
}

TEST_F(SocketTest, server_connect_ok_then_async_send_recv_close)
{
    // Given
    int fakeFdStatus = SOCKET_CONNECTED;
    RaSocketFdHandleParam fakeParam(fakeFdHandle, fakeFdStatus, remoteIp);

    MOCKER(RaGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParam));

    u8 buf[32] = {0};
    RequestHandle fakeReqHandle = 1;
    unsigned long long dataSize = 32;
    MOCKER(HrtRaSocketSendAsync)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(dataSize))
        .will(returnValue(fakeReqHandle));

    MOCKER(HrtRaSocketRecvAsync)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(dataSize))
        .will(returnValue(fakeReqHandle));

    // when
    socketServer->Connect();
    auto status = socketServer->GetAsyncStatus();
    // then
    EXPECT_EQ(status, SocketStatus::OK);

    socketServer->SendAsync(buf, dataSize);

    status = socketServer->GetAsyncStatus();
    EXPECT_EQ(status, SocketStatus::OK);

    socketServer->RecvAsync(buf, dataSize);

    status = socketServer->GetAsyncStatus();
    EXPECT_EQ(status, SocketStatus::OK);

    MOCKER(HrtRaSocketCloseOne).stubs().with(mockcpp::any());
    socketServer->Close();
}

TEST_F(SocketTest, async_listen_stop_ok)
{
    // Given
    // when
    socketServer->ListenAsync();
    EXPECT_EQ(socketServer->socketStatus, SocketStatus::LISTEN_STARTING);
    EXPECT_EQ(socketClient->GetAsyncStatus(), SocketStatus::OK);

    socketServer->StopListen();
    EXPECT_EQ(socketServer->isListening, false);
}

TEST_F(SocketTest, async_connect_close_ok)
{
    // Given
    int fakeFdStatus = SOCKET_CONNECTED;
    RaSocketFdHandleParam fakeParam(fakeFdHandle, fakeFdStatus, remoteIp);
    MOCKER(HrtRaBlockGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParam));

    // when
    socketClient->ConnectAsync();
    EXPECT_EQ(socketClient->socketStatus, SocketStatus::CONNECT_STARTING);
    EXPECT_EQ(socketClient->GetAsyncStatus(), SocketStatus::OK);

    socketClient->Close();
    EXPECT_EQ(socketClient->isConnected, false);
}

TEST_F(SocketTest, async_listen_eagain_stop_ok)
{
    // Given
    ReqHandleResult notCompletedResult = ReqHandleResult::NOT_COMPLETED;
    ReqHandleResult sockEAagainResult = ReqHandleResult::SOCK_E_AGAIN;
    ReqHandleResult completedResult = ReqHandleResult::COMPLETED;
    MOCKER(HrtRaGetAsyncReqResult)
        .stubs()
        .will(returnValue(notCompletedResult))
        .then(returnValue(sockEAagainResult))
        .then(returnValue(notCompletedResult))
        .then(returnValue(completedResult));

    // when
    socketServer->ListenAsync();
    EXPECT_EQ(socketServer->socketStatus, SocketStatus::LISTEN_STARTING);
    EXPECT_EQ(socketClient->GetAsyncStatus(), SocketStatus::OK);

    socketServer->StopListen();
    EXPECT_EQ(socketServer->isListening, false);
}

TEST_F(SocketTest, async_connect_eagain_close_ok)
{
    // Given
    ReqHandleResult notCompletedResult = ReqHandleResult::NOT_COMPLETED;
    ReqHandleResult sockEAagainResult = ReqHandleResult::SOCK_E_AGAIN;
    ReqHandleResult completedResult = ReqHandleResult::COMPLETED;
    MOCKER(HrtRaGetAsyncReqResult)
        .stubs()
        .will(returnValue(notCompletedResult))
        .then(returnValue(sockEAagainResult))
        .then(returnValue(notCompletedResult))
        .then(returnValue(completedResult));

    int fakeFdStatus = SOCKET_CONNECTED;
    RaSocketFdHandleParam fakeParam(fakeFdHandle, fakeFdStatus, remoteIp);
    MOCKER(HrtRaBlockGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParam));

    // when
    socketClient->ConnectAsync();
    EXPECT_EQ(socketClient->GetAsyncStatus(), SocketStatus::CONNECT_STARTING);
    EXPECT_EQ(socketClient->GetAsyncStatus(), SocketStatus::CONNECT_STARTING);
    EXPECT_EQ(socketClient->GetAsyncStatus(), SocketStatus::CONNECT_STARTING);
    EXPECT_EQ(socketClient->GetAsyncStatus(), SocketStatus::OK);

    socketClient->Close();
    EXPECT_EQ(socketClient->isConnected, false);
}

TEST_F(SocketTest, async_listen_error)
{
    // Given
    ReqHandleResult invalidParaResult = ReqHandleResult::INVALID_PARA;
    MOCKER(HrtRaGetAsyncReqResult).stubs().will(returnValue(invalidParaResult));

    // when
    socketServer->ListenAsync();
    EXPECT_THROW(socketServer->GetAsyncStatus(), SocketException);
}

TEST_F(SocketTest, async_connect_error)
{
    // Given
    ReqHandleResult invalidParaResult = ReqHandleResult::INVALID_PARA;
    MOCKER(HrtRaGetAsyncReqResult).stubs().will(returnValue(invalidParaResult));

    // when
    socketClient->ConnectAsync();
    EXPECT_THROW(socketClient->GetAsyncStatus(), SocketException);
}

TEST_F(SocketTest, server_connect_async_ok_then_async_send_recv_close)
{
    // Given
    int fakeFdStatus = SOCKET_CONNECTED;
    RaSocketFdHandleParam fakeParam(fakeFdHandle, fakeFdStatus, remoteIp);

    MOCKER(RaGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParam));

    u8 buf[32] = {0};
    RequestHandle fakeReqHandle = 1;
    unsigned long long dataSize = 32;
    MOCKER(HrtRaSocketSendAsync)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(dataSize))
        .will(returnValue(fakeReqHandle));

    MOCKER(HrtRaSocketRecvAsync)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(dataSize))
        .will(returnValue(fakeReqHandle));

    // when
    socketServer->ConnectAsync();

    // then
    auto status = socketServer->GetAsyncStatus();
    EXPECT_EQ(status, SocketStatus::OK);

    socketServer->SendAsync(buf, dataSize);

    status = socketServer->GetAsyncStatus();
    EXPECT_EQ(status, SocketStatus::OK);

    socketServer->RecvAsync(buf, dataSize);

    status = socketServer->GetAsyncStatus();
    EXPECT_EQ(status, SocketStatus::OK);

    status = socketServer->GetAsyncStatus();
    EXPECT_EQ(status, SocketStatus::OK);

    MOCKER(HrtRaSocketCloseOne).stubs().with(mockcpp::any());
    socketServer->Close();
}

TEST_F(SocketTest, server_connect_async_ok_then_async_send_recv_egain_close)
{
    int fakeFdStatus = SOCKET_CONNECTED;
    RaSocketFdHandleParam fakeParam(fakeFdHandle, fakeFdStatus, remoteIp);

    MOCKER(RaGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParam));

    u8 buf[32] = {0};
    RequestHandle fakeReqHandle = 1;
    unsigned long long dataSize = 32;
    MOCKER(HrtRaSocketSendAsync)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(dataSize))
        .will(returnValue(fakeReqHandle));

    MOCKER(HrtRaSocketRecvAsync)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(dataSize))
        .will(returnValue(fakeReqHandle));

    // when
    socketServer->ConnectAsync();

    // given
    SocketStatus status;
    ReqHandleResult notCompletedResult = ReqHandleResult::NOT_COMPLETED;
    ReqHandleResult sockEAagainResult = ReqHandleResult::SOCK_E_AGAIN;
    ReqHandleResult completedResult = ReqHandleResult::COMPLETED;
    MOCKER(HrtRaGetAsyncReqResult)
        .stubs()
        .will(returnValue(notCompletedResult)) // send wait
        .then(returnValue(sockEAagainResult))  // send retry
        .then(returnValue(completedResult))    // send done
        .then(returnValue(notCompletedResult)) // recv wait
        .then(returnValue(sockEAagainResult))  // recv retry
        .then(returnValue(completedResult));   // recv done

    // then
    status = socketServer->GetAsyncStatus();
    EXPECT_EQ(status, SocketStatus::OK);

    // send test
    socketServer->SendAsync(buf, dataSize);

    status = socketServer->GetAsyncStatus();
    EXPECT_EQ(status, SocketStatus::SENDING);

    status = socketServer->GetAsyncStatus();
    EXPECT_EQ(status, SocketStatus::SENDING);

    status = socketServer->GetAsyncStatus();
    EXPECT_EQ(status, SocketStatus::OK);

    // recv test
    socketServer->RecvAsync(buf, dataSize);

    status = socketServer->GetAsyncStatus();
    EXPECT_EQ(status, SocketStatus::RECVING);

    status = socketServer->GetAsyncStatus();
    EXPECT_EQ(status, SocketStatus::RECVING);

    status = socketServer->GetAsyncStatus();
    EXPECT_EQ(status, SocketStatus::OK);

    status = socketServer->GetAsyncStatus();
    EXPECT_EQ(status, SocketStatus::OK);

    MOCKER(HrtRaSocketCloseOne).stubs().with(mockcpp::any());
    socketServer->Close();
}

TEST_F(SocketTest, send_nullptr) { EXPECT_THROW(socketServer->SendAsync(nullptr, 1), SocketException); }

TEST_F(SocketTest, recv_nullptr) { EXPECT_THROW(socketServer->RecvAsync(nullptr, 1), SocketException); }

TEST_F(SocketTest, send_size_zero)
{
    u8 fakeBuf[1] = {0};
    EXPECT_THROW(socketServer->SendAsync(fakeBuf, 0), SocketException);
}

TEST_F(SocketTest, recv_size_zero)
{
    u8 fakeBuf[1] = {0};
    EXPECT_THROW(socketServer->RecvAsync(fakeBuf, 0), SocketException);
}

TEST_F(SocketTest, Ut_SetRemoteAddr_When_Called_Expect_OnlyRemoteAddrChanged)
{
    // Given：完整 LinkData（type/protocol/ranks/addrs）
    LinkData linkData(
        PortDeploymentType::DEV_NET, LinkProtocol::UB_CTP, 0, 1, IpAddress("127.0.0.1"), IpAddress("192.168.0.1"));
    IpAddress newRemoteAddr("10.1.2.3");

    // when
    linkData.SetRemoteAddr(newRemoteAddr);

    // then：仅 remoteAddr 变化，其余字段保持
    EXPECT_EQ(linkData.GetRemoteAddr().GetIpStr(), newRemoteAddr.GetIpStr());
    EXPECT_EQ(linkData.GetLocalAddr().GetIpStr(), IpAddress("127.0.0.1").GetIpStr());
    EXPECT_EQ(linkData.GetLocalRankId(), 0U);
    EXPECT_EQ(linkData.GetRemoteRankId(), 1U);
    EXPECT_EQ(linkData.GetLocalPortId(), 0U);
    EXPECT_EQ(linkData.GetRemotePortId(), 0U);
}

TEST_F(SocketTest, Ut_SocketConfig_When_RemoteAddrChanged_Expect_HccpTagSameAndKeyDifferent)
{
    // Given：verbatim tag 的连接 config 模板
    LinkData linkData(
        PortDeploymentType::DEV_NET, LinkProtocol::UB_CTP, 0, 1, IpAddress("127.0.0.1"), IpAddress("192.168.0.1"));
    SocketConfig templateConfig(linkData, "UT_ACCEPT", SocketRole::SERVER, 8080, true);

    // when：拷贝模板后仅替换 remoteAddr（与 SocketMgr::AcceptSocket 的语义一致）
    SocketConfig peerA = templateConfig;
    peerA.link.SetRemoteAddr(IpAddress("10.1.2.3"));
    SocketConfig peerB = templateConfig;
    peerB.link.SetRemoteAddr(IpAddress("10.1.2.4"));

    // then：hccpTag 保持模板值，remoteAddr 不同即不同 key
    EXPECT_EQ(peerA.GetHccpTag(), templateConfig.GetHccpTag());
    EXPECT_EQ(peerB.GetHccpTag(), templateConfig.GetHccpTag());
    EXPECT_NE(peerA.link.GetRemoteAddr().GetIpStr(), peerB.link.GetRemoteAddr().GetIpStr());
    EXPECT_FALSE(std::equal_to<SocketConfig>{}(peerA, peerB));
}

TEST_F(SocketTest, Ut_SocketAccept_When_ErrorStatusesAndRole_Expect_MappedCodes)
{
    std::shared_ptr<Socket> acceptedSocket = nullptr;

    // 轮1：非 SERVER 角色 → PARA
    EXPECT_EQ(socketClient->Accept("testxxxxx", acceptedSocket, 1), HCCL_E_PARA);
    EXPECT_EQ(acceptedSocket, nullptr);

    // 轮2：连接超时 → TIMEOUT
    int fakeFdStatus = SOCKET_CONNECT_TIMEOUT;
    RaSocketFdHandleParam fakeParamTimeout(fakeFdHandle, fakeFdStatus, remoteIp);
    MOCKER(HrtRaBlockGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParamTimeout));
    EXPECT_EQ(socketServer->Accept("testxxxxx", acceptedSocket, 1), HCCL_E_TIMEOUT);
    EXPECT_EQ(acceptedSocket, nullptr);
    GlobalMockObject::verify();

    // 轮3：连接中 → AGAIN
    fakeFdStatus = SOCKET_CONNECTING;
    RaSocketFdHandleParam fakeParamConnecting(fakeFdHandle, fakeFdStatus, remoteIp);
    MOCKER(HrtRaBlockGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParamConnecting));
    EXPECT_EQ(socketServer->Accept("testxxxxx", acceptedSocket, 1), HCCL_E_AGAIN);
    EXPECT_EQ(acceptedSocket, nullptr);
    GlobalMockObject::verify();

    // 轮4：其他状态（未连接）→ TCP_CONNECT
    fakeFdStatus = SOCKET_NOT_CONNECTED;
    RaSocketFdHandleParam fakeParamInit(fakeFdHandle, fakeFdStatus, remoteIp);
    MOCKER(HrtRaBlockGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParamInit));
    EXPECT_EQ(socketServer->Accept("testxxxxx", acceptedSocket, 1), HCCL_E_TCP_CONNECT);
    EXPECT_EQ(acceptedSocket, nullptr);
    GlobalMockObject::verify();
}

TEST_F(SocketTest, Ut_SocketAccept_When_StatusOk_Expect_SuccessAndSocketValid)
{
    int fakeFdStatus = SOCKET_CONNECTED;
    RaSocketFdHandleParam fakeParam(fakeFdHandle, fakeFdStatus, remoteIp);
    MOCKER(HrtRaBlockGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParam));

    std::shared_ptr<Socket> acceptedSocket = nullptr;
    EXPECT_EQ(socketServer->Accept("testxxxxx", acceptedSocket, 1), HCCL_SUCCESS);
    ASSERT_NE(acceptedSocket, nullptr);
    EXPECT_EQ(acceptedSocket->GetFdHandle(), fakeFdHandle);
}
