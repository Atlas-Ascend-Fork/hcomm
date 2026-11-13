/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define private public
#define protected public
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "dev_ub_connection.h"
#include "rma_conn_manager.h"
#include "socket.h"
#include "orion_adapter_rts.h"
#include "not_support_exception.h"
#include "rma_conn_exception.h"
#undef private

using namespace Hccl;

class DevUbConnectionTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DevUbConnection tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "DevUbConnection tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in DevUbConnection SetUP" << std::endl;
        fakeSocket
            = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete fakeSocket;
        std::cout << "A Test case in DevUbConnection TearDown" << std::endl;
    }
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    std::string tag = "test";
};

TEST(DevUbConnectionTest, rma_ub_connection_get_pi_sqDepth_ok)
{
    // Given: socket time out
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::TIMEOUT));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    // Then
    EXPECT_EQ(devUbConnection.piVal, 0);
    EXPECT_EQ(devUbConnection.sqDepth, 8192);
}

TEST(DevUbConnectionTest, rma_ub_connection_get_jfcMode_and_jettyHandle_ok)
{
    // Given: socket time out
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::TIMEOUT));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    JettyHandle jettyHandle = 100;
    devUbConnection.jettyHandle = jettyHandle;

    // Then
    EXPECT_EQ(devUbConnection.jfcMode, HrtUbJfcMode::STARS_POLL);
    EXPECT_EQ(devUbConnection.GetJettyHandle(), jettyHandle);
}

TEST(DevUbConnectionTest, tp_import_test)
{
    MOCKER(HrtGetDevicePhyIdByUserDevId).defaults().will(returnValue(static_cast<DevId>(0)));
    // construct DevUbConnection
    RdmaHandle rdmaHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 10, 11, 10, 11);
    linkData.localAddr_ = IpAddress("10.0.0.1");
    linkData.remoteAddr_ = IpAddress("10.0.0.2");
    linkData.linkProtocol_ = LinkProtocol::UB_TP;
    std::string tag = "test";
    DevUbTpConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    EXPECT_EQ(devUbConnection.GetStatus(), RmaConnStatus::INIT); // TP_INFO_GETTING
    EXPECT_EQ(devUbConnection.GetStatus(), RmaConnStatus::EXCHANGEABLE);

    // Then
    auto rmtDto = devUbConnection.GetExchangeDto();
    devUbConnection.ParseRmtExchangeDto(*rmtDto);
    devUbConnection.ImportRmtDto();
    EXPECT_EQ(devUbConnection.status, RmaConnStatus::EXCHANGEABLE);
    EXPECT_EQ(DevUbConnection::UbConnStatus::JETTY_IMPORTING, devUbConnection.ubConnStatus);

    EXPECT_EQ(devUbConnection.GetStatus(), RmaConnStatus::READY);
    EXPECT_EQ(DevUbConnection::UbConnStatus::READY, devUbConnection.ubConnStatus);
}

TEST(DevUbConnectionTest, ctp_import_test)
{
    // construct DevUbConnection
    RdmaHandle rdmaHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 10, 11, 10, 11);
    linkData.localAddr_ = IpAddress("11.0.0.1");
    linkData.remoteAddr_ = IpAddress("11.0.0.2");
    linkData.linkProtocol_ = LinkProtocol::UB_CTP;
    std::string tag = "test";
    DevUbCtpConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    EXPECT_EQ(devUbConnection.GetStatus(), RmaConnStatus::INIT); // TP_INFO_GETTING
    EXPECT_EQ(devUbConnection.GetStatus(), RmaConnStatus::EXCHANGEABLE);

    // Then
    auto rmtDto = devUbConnection.GetExchangeDto();
    devUbConnection.ParseRmtExchangeDto(*rmtDto);
    devUbConnection.ImportRmtDto();
    EXPECT_EQ(devUbConnection.status, RmaConnStatus::EXCHANGEABLE);
    EXPECT_EQ(DevUbConnection::UbConnStatus::JETTY_IMPORTING, devUbConnection.ubConnStatus);

    EXPECT_EQ(devUbConnection.GetStatus(), RmaConnStatus::READY);
    EXPECT_EQ(DevUbConnection::UbConnStatus::READY, devUbConnection.ubConnStatus);
}
