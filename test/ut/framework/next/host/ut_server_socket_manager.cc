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
#include "hcomm_c_adpt.h"
#include "server_socket/server_socket_manager.h"
#include "socket_exception.h"
#define private public
using namespace hcomm;

class ServerSocketManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ServerSocketManagerTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "ServerSocketManagerTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in ServerSocketManagerTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in ServerSocketManagerTest TearDown" << std::endl;
    }
};

TEST_F(ServerSocketManagerTest, Ut_When_Device_Socket_Listen_Expect_SUCCESS)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 60001;
    HcclResult ret = ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::DEVICE_NIC_TYPE, 60001);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ServerSocketManagerTest, Ut_When_Host_Socket_Listen_Expect_SUCCESS)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 60001;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 60001);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ServerSocketManagerTest, Ut_When_Stop_Listen_While_Not_Start_Listen_Expect_Fail)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::DEVICE_NIC_TYPE, 60001);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(ServerSocketManagerTest, Ut_When_Stop_Listen_While_Listen_Count_Is_Zero_Listen_Expect_Fail)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 60001;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ServerSocketManager::GetInstance().hostServerSocketMap_.clear();
    ret = ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 60001);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}
TEST_F(ServerSocketManagerTest, Ut_When_ServerSocketStartListen_Device_Expect_SUCCESS)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 60001;
    HcclResult ret = ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(port, 0);
    ret = ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::DEVICE_NIC_TYPE, port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ServerSocketManagerTest, Ut_When_ServerSocketStartListen_Host_Expect_SUCCESS)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 0;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(port, 0);
    ret = ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::HOST_NIC_TYPE, port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ServerSocketManagerTest, Ut_When_ServerSocketStartListen_Illegal_NicType_Expect_Fail)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 0;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::INVALID, 0, &port);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeInit_WithEmptyMaps_Expect_NoCrash)
{
    ServerSocketManager::GetInstance().DeInit(0);
    EXPECT_EQ(ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);
    EXPECT_EQ(ServerSocketManager::GetInstance().hostServerSocketMap_.size(), 0);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeInit_WithDeviceSocket_Expect_Clear)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 43210;
    HcclResult ret = ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);

    ServerSocketManager::GetInstance().DeInit(0);
    EXPECT_EQ(ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeInit_WithHostSocket_Expect_Clear)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 0;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(port, 0);
    EXPECT_NE(ServerSocketManager::GetInstance().hostServerSocketMap_.size(), 0);

    ServerSocketManager::GetInstance().DeInit(0);
    EXPECT_EQ(ServerSocketManager::GetInstance().hostServerSocketMap_.size(), 0);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeInitDeviceSockets_Expect_OnlyMatchingDevPhyId)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 43210;
    HcclResult ret = ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ServerSocketManager::GetInstance().DeInitDeviceSockets(0);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeInitHostSockets_Expect_OnlyMatchingDevPhyId)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 0;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ServerSocketManager::GetInstance().DeInitHostSockets(0);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeInitDeviceSockets_NonMatchingDevPhyId_Expect_NotClear)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 43210;
    HcclResult ret = ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);

    ServerSocketManager::GetInstance().DeInitDeviceSockets(999);
    EXPECT_NE(ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);

    ServerSocketManager::GetInstance().DeInitDeviceSockets(0);
    EXPECT_EQ(ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeInitHostSockets_NonMatchingDevPhyId_Expect_NotClear)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 0;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(ServerSocketManager::GetInstance().hostServerSocketMap_.size(), 0);

    ServerSocketManager::GetInstance().DeInitHostSockets(999);
    EXPECT_NE(ServerSocketManager::GetInstance().hostServerSocketMap_.size(), 0);

    ServerSocketManager::GetInstance().DeInitHostSockets(0);
    EXPECT_EQ(ServerSocketManager::GetInstance().hostServerSocketMap_.size(), 0);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeviceSocketReuse_Expect_CountIncrement)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 43211;
    HcclResult ret = ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ServerSocketManager::GetInstance().deviceServerSocketMap_[localPort][43211].second, 1);

    ret = ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ServerSocketManager::GetInstance().deviceServerSocketMap_[localPort][43211].second, 2);

    ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::DEVICE_NIC_TYPE, 43211);
    ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::DEVICE_NIC_TYPE, 43211);
}

TEST_F(ServerSocketManagerTest, Ut_When_HostSocketReuse_Expect_CountIncrement)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 43212;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ServerSocketManager::GetInstance().hostServerSocketMap_[localPort][43212].second, 1);

    ret = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ServerSocketManager::GetInstance().hostServerSocketMap_[localPort][43212].second, 2);

    ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 43212);
    ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 43212);
}

// HostSocketStopListen refcount 归 0 时，应调用 socketMgrCompat_->ServerDeInit 擦除 GetServerSocketMap 条目
// stub 中 CheckServerPortListening 固定返回 true，因此 ServerDeInit 必然被调用
TEST_F(ServerSocketManagerTest, Ut_HostSocketStopListen_RefcountZero_Expect_ServerDeInitCalled)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 43213;

    // StartListen: stub CheckServerPortListening 返回 true → isListen=true → 跳过 Listen，建占位
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // mock ServerDeInit，期望被调用一次（refcount 1→0 时触发）
    MOCKER_CPP(&Hccl::SocketManager::ServerDeInit).expects(once()).with(mockcpp::any());

    // StopListen: refcount 1→0 → CheckServerPortListening(true) → ServerDeInit
    ret = ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::HOST_NIC_TYPE, port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(ServerSocketManagerTest, Ut_When_AddWhiteList_Device_Expect_SUCCESS)
{
    Hccl::IpAddress ipAddr("1.0.0.1");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 60002;
    HcclResult ret = ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(Hccl::HrtRaSocketWhiteListAdd).stubs().will(ignoreReturnValue());
    ret = ServerSocketManager::GetInstance().AddWhiteList(localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, "server", 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(Hccl::HrtRaSocketWhiteListDel).stubs().will(ignoreReturnValue());
    ret = ServerSocketManager::GetInstance().DeleteWhiteList(localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, "server", 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::DEVICE_NIC_TYPE, port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ServerSocketManagerTest, Ut_When_AddWhiteList_Host_Expect_SUCCESS)
{
    Hccl::IpAddress ipAddr("1.0.0.2");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 60003;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(Hccl::HrtRaSocketWhiteListAdd).stubs().will(ignoreReturnValue());
    ret = ServerSocketManager::GetInstance().AddWhiteList(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, "server", 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(Hccl::HrtRaSocketWhiteListDel).stubs().will(ignoreReturnValue());
    ret = ServerSocketManager::GetInstance().DeleteWhiteList(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, "server", 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::HOST_NIC_TYPE, port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ServerSocketManagerTest, Ut_When_AddWhiteList_HandleNotFound_Expect_Fail)
{
    Hccl::IpAddress ipAddr("9.9.9.1");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);

    HcclResult ret
        = ServerSocketManager::GetInstance().AddWhiteList(localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, "server", 1);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeleteWhiteList_HandleNotFound_Expect_Fail)
{
    Hccl::IpAddress ipAddr("9.9.9.2");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);

    HcclResult ret
        = ServerSocketManager::GetInstance().DeleteWhiteList(localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, "server", 1);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

// UT 环境 HostSocketHandleManager::Get 为 stub（恒返回非空），mock 成 nullptr 覆盖 host 侧找不到 handle 的分支
TEST_F(ServerSocketManagerTest, Ut_When_AddWhiteList_HostHandleNotFound_Expect_Fail)
{
    Hccl::IpAddress ipAddr("9.9.9.3");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);

    MOCKER_CPP(&Hccl::HostSocketHandleManager::Get).stubs().will(returnValue(static_cast<Hccl::SocketHandle>(nullptr)));
    HcclResult ret
        = ServerSocketManager::GetInstance().AddWhiteList(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, "server", 1);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(ServerSocketManagerTest, Ut_GetServerSocket_When_ListeningHostAndDevice_Expect_ReturnSocket)
{
    // 已监听的 host/device 取回同一 socket（重复调用返回同一指针）
    Hccl::IpAddress hostIpAddr("1.0.0.10");
    Hccl::DevNetPortType hostType = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData hostPortData = Hccl::PortData(0, hostType, 0, hostIpAddr);
    uint32_t hostPort = 60020;
    EXPECT_EQ(
        ServerSocketManager::GetInstance().ServerSocketStartListen(
            hostPortData, Hccl::NicType::HOST_NIC_TYPE, 0, &hostPort),
        HCCL_SUCCESS);
    std::shared_ptr<Hccl::Socket> hostSocket = nullptr;
    std::shared_ptr<Hccl::Socket> hostSocketAgain = nullptr;
    EXPECT_EQ(
        ServerSocketManager::GetInstance().GetServerSocket(
            hostPortData, Hccl::NicType::HOST_NIC_TYPE, hostPort, hostSocket),
        HCCL_SUCCESS);
    ASSERT_NE(hostSocket, nullptr);
    EXPECT_EQ(
        ServerSocketManager::GetInstance().GetServerSocket(
            hostPortData, Hccl::NicType::HOST_NIC_TYPE, hostPort, hostSocketAgain),
        HCCL_SUCCESS);
    EXPECT_EQ(hostSocket.get(), hostSocketAgain.get());

    Hccl::IpAddress devIpAddr("1.0.0.11");
    Hccl::DevNetPortType devType = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData devPortData = Hccl::PortData(0, devType, 0, devIpAddr);
    uint32_t devPort = 60021;
    EXPECT_EQ(
        ServerSocketManager::GetInstance().ServerSocketStartListen(
            devPortData, Hccl::NicType::DEVICE_NIC_TYPE, 0, &devPort),
        HCCL_SUCCESS);
    std::shared_ptr<Hccl::Socket> devSocket = nullptr;
    EXPECT_EQ(
        ServerSocketManager::GetInstance().GetServerSocket(
            devPortData, Hccl::NicType::DEVICE_NIC_TYPE, devPort, devSocket),
        HCCL_SUCCESS);
    ASSERT_NE(devSocket, nullptr);
    EXPECT_NE(hostSocket.get(), devSocket.get());

    EXPECT_EQ(
        ServerSocketManager::GetInstance().ServerSocketStopListen(hostPortData, Hccl::NicType::HOST_NIC_TYPE, hostPort),
        HCCL_SUCCESS);
    EXPECT_EQ(
        ServerSocketManager::GetInstance().ServerSocketStopListen(devPortData, Hccl::NicType::DEVICE_NIC_TYPE, devPort),
        HCCL_SUCCESS);
}

TEST_F(ServerSocketManagerTest, Ut_GetServerSocket_When_NotListeningOrIllegalNic_Expect_ErrorMapping)
{
    Hccl::IpAddress ipAddr("9.9.9.30");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    std::shared_ptr<Hccl::Socket> serverSocket = nullptr;

    // 轮1：host 未监听 → NOT_FOUND
    EXPECT_EQ(
        ServerSocketManager::GetInstance().GetServerSocket(
            localPort, Hccl::NicType::HOST_NIC_TYPE, 60030, serverSocket),
        HCCL_E_NOT_FOUND);
    EXPECT_EQ(serverSocket, nullptr);

    // 轮2：device 未监听 → NOT_FOUND
    EXPECT_EQ(
        ServerSocketManager::GetInstance().GetServerSocket(
            localPort, Hccl::NicType::DEVICE_NIC_TYPE, 60030, serverSocket),
        HCCL_E_NOT_FOUND);
    EXPECT_EQ(serverSocket, nullptr);

    // 轮3：非法 nicType → PARA
    EXPECT_EQ(
        ServerSocketManager::GetInstance().GetServerSocket(localPort, Hccl::NicType::INVALID, 60030, serverSocket),
        HCCL_E_PARA);
}

TEST_F(ServerSocketManagerTest, Ut_WhiteList_When_ErrorPaths_Expect_NotFoundThenInternal)
{
    Hccl::IpAddress ipAddr("9.9.9.31");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);

    // 轮1：host 删除白名单但 handle 找不到 → NOT_FOUND
    MOCKER_CPP(&Hccl::HostSocketHandleManager::Get).stubs().will(returnValue(static_cast<Hccl::SocketHandle>(nullptr)));
    EXPECT_EQ(
        ServerSocketManager::GetInstance().DeleteWhiteList(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, "server", 1),
        HCCL_E_NOT_FOUND);
    GlobalMockObject::verify();

    // 轮2：底层加白名单抛异常 → INTERNAL
    uint32_t port = 60031;
    EXPECT_EQ(
        ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port),
        HCCL_SUCCESS);
    MOCKER(Hccl::HrtRaSocketWhiteListAdd).stubs().will(throws(Hccl::SocketException("ut")));
    EXPECT_EQ(
        ServerSocketManager::GetInstance().AddWhiteList(localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, "server", 1),
        HCCL_E_INTERNAL);
    GlobalMockObject::verify();

    // 轮3：底层删白名单抛异常 → INTERNAL
    MOCKER(Hccl::HrtRaSocketWhiteListDel).stubs().will(throws(Hccl::SocketException("ut")));
    EXPECT_EQ(
        ServerSocketManager::GetInstance().DeleteWhiteList(localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, "server", 1),
        HCCL_E_INTERNAL);
    GlobalMockObject::verify();

    EXPECT_EQ(
        ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::DEVICE_NIC_TYPE, port),
        HCCL_SUCCESS);
}

TEST_F(ServerSocketManagerTest, Ut_When_AddWhiteList_IllegalNicType_Expect_Fail)
{
    Hccl::IpAddress ipAddr("1.0.0.3");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);

    HcclResult ret = ServerSocketManager::GetInstance().AddWhiteList(localPort, Hccl::NicType::INVALID, 0, "server", 1);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(ServerSocketManagerTest, Ut_BuildWhiteList_When_TagAndConnLimit_Expect_EntryMatched)
{
    std::vector<Hccl::RaSocketWhitelist> wlists = ServerSocketManager::GetInstance().BuildWhiteList("server", 3);
    ASSERT_EQ(wlists.size(), 1u);
    EXPECT_EQ(wlists[0].tag, "server");
    EXPECT_EQ(wlists[0].connLimit, 3u);
}
