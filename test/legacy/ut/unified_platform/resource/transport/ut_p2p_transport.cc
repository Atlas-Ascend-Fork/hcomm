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
#include "virtual_topo.h"
#include "p2p_connection.h"
#include "task.h"
#include "p2p_enable_manager.h" // HrtEnableP2P/HrtDisableP2P/HrtGetP2PStatus 及 DRV_P2P_STATUS 状态定义由此带入
#define protected public
#include "p2p_transport.h"
#undef protected
#include "runtime_api_exception.h"
#include "data_type.h"
#include "reduce_op.h"
#include "stream.h"
#include "internal_exception.h"
#include "timeout_exception.h"
#include "socket_exception.h"
#include "local_ipc_rma_buffer_v2.h"
#include "ipc_local_notify.h"
#include "dev_buffer.h"
#include "rma_buffer.h"
using namespace Hccl;

class StubP2PRmaConnection : public P2PConnection {
public:
    StubP2PRmaConnection() : P2PConnection(nullptr, "tag") {}

    unique_ptr<BaseTask>
    PrepareRead(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config) override
    {
        return make_unique<TaskP2pMemcpy>(localMemBuf.addr, remoteMemBuf.addr, localMemBuf.size, MemcpyKind::D2D);
    }

    unique_ptr<BaseTask> PrepareReadReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
        const SqeConfig& config) override
    {
        return make_unique<TaskSdmaReduce>(localMemBuf.addr, remoteMemBuf.addr, localMemBuf.size, datatype, reduceOp);
    }

    unique_ptr<BaseTask>
    PrepareWrite(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config) override
    {
        return make_unique<TaskP2pMemcpy>(remoteMemBuf.addr, localMemBuf.addr, localMemBuf.size, MemcpyKind::D2D);
    }

    unique_ptr<BaseTask> PrepareWriteReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
        const SqeConfig& config) override
    {
        return nullptr;
    }

    string Describe() const override { return "StubP2PRmaConnection"; }

    void Connect() override {}
};

class P2PTransportTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "P2PTransport tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "P2PTransport tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in P2PTransport SetUP" << std::endl;
        MOCKER(HrtMemAsyncCopy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtReduceAsync).stubs().with(mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));
        // aclrtCreateStreamWithConfig 新签名将 stream 改为出参且返回 aclError，mock 出参繁琐；
        // 直接 mock 上层 HrtStreamCreateWithFlags 返回非空 fake 指针，避免真实 HrtStreamSetMode 空指针检查抛异常
        MOCKER(HrtStreamCreateWithFlags)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), outBound((void*)100))
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtStreamDestroy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtGetStreamId).stubs().with(mockcpp::any(), outBound(0)).will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtGetDeviceType)
            .stubs()
            .with(outBound(Hccl::DevType((DevType)DevType::DEV_TYPE_910A2)))
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtIpcOpenNotify)
            .stubs()
            .with(mockcpp::any(), outBound((void*)fakeNotifyHandleAddr))
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtDeviceGetBareTgid).stubs().with(outBound(fakePid)).will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtGetDevice).stubs().with(outBound(0)).will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtNotifyCreate)
            .stubs()
            .with(mockcpp::any(), outBound((void*)(fakeNotifyHandleAddr)))
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtIpcSetNotifyName)
            .stubs()
            .with(mockcpp::any(), outBoundP(fakeName, sizeof(fakeName)), mockcpp::any())
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtGetNotifyID)
            .stubs()
            .with(mockcpp::any(), outBound(fakeNotifyId))
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtNotifyGetAddr)
            .stubs()
            .with(mockcpp::any(), outBound(fakeAddress))
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtNotifyGetOffset)
            .stubs()
            .with(mockcpp::any(), outBound(fakeOffset))
            .will(returnValue(HcclResult::HCCL_SUCCESS));

        MOCKER(HrtNotifyRecord).stubs().with(mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtNotifyWaitWithTimeOut).stubs().with(mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in P2PTransport TearDown" << std::endl;
        GlobalMockObject::verify();
    }

    RmaBufferSlice locSlice;
    RmtRmaBufferSlice rmtSlice;

    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    char fakeName[65] = "testRtsNotify";
};

TEST_F(P2PTransportTest, P2PTransport_describe)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    P2PTransport transport(locRes, attr, link, fakeSocket);
    transport.Describe();
}

TEST_F(P2PTransportTest, P2PTransport_establish)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    P2PTransport transport(locRes, attr, link, fakeSocket);

    MOCKER_CPP(&P2PTransport::IsSocketReady).stubs().will(returnValue(true));
    EXPECT_NO_THROW(transport.Establish());
}

TEST_F(P2PTransportTest, P2PTransport_is_socket_ready)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    P2PTransport transport(locRes, attr, link, fakeSocket);

    SocketStatus socketStatusInit = SocketStatus::INIT;
    SocketStatus socketStatusOK = SocketStatus::OK;
    SocketStatus socketStatusTimeout = SocketStatus::TIMEOUT;
    MOCKER_CPP(&Socket::GetAsyncStatus)
        .stubs()
        .will(returnValue(socketStatusInit))
        .then(returnValue(socketStatusTimeout))
        .then(returnValue(socketStatusOK));

    EXPECT_FALSE(transport.IsSocketReady()); // INIT 状态未就绪
    EXPECT_FALSE(transport.IsSocketReady()); // TIMEOUT 状态置 baseStatus=SOCKET_TIMEOUT 后返回 false
    EXPECT_TRUE(transport.IsSocketReady());
    EXPECT_TRUE(transport.IsSocketReady()); // baseStatus 为 SOCKET_OK 时，直接返回 true

    transport.socket = nullptr;
    EXPECT_FALSE(transport.IsSocketReady()); // socket 为空时返回 false
}

// PCIE 链路按需 enable p2p 全流程：socket 未就绪也先下发 enable（避免 socket 与 p2p 互等），
// 轮询驱动侧状态直至使能；析构时仅对已持有的引用对称执行 disable p2p
TEST_F(P2PTransportTest, P2PTransport_get_status_pcie_enable_and_symmetric_disable)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    IpAddress ipAddress("1.0.0.0");
    LinkData link(PortDeploymentType::P2P, LinkProtocol::PCIE, 0, 1, ipAddress, ipAddress, 10, 20, 0);
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    MOCKER(HrtGetDevicePhyIdByUserDevId)
        .stubs()
        .with(mockcpp::any(), outBound(static_cast<DevId>((DevId)10)))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtEnableP2P).expects(once()).will(returnValue(HCCL_SUCCESS)); // INIT 首轮触发一次，后续轮询不重复下发
    MOCKER(HrtDisableP2P).expects(once()).will(returnValue(HCCL_SUCCESS)); // 析构对称 disable 恰好一次
    uint32_t drvStatus = DRV_P2P_STATUS_DISABLE;
    MOCKER(HrtGetP2PStatus)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&drvStatus, sizeof(drvStatus)))
        .will(returnValue(HCCL_SUCCESS));
    SocketStatus socketStatusInit = SocketStatus::INIT;
    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue(socketStatusInit)); // socket 全程未就绪

    {
        P2PTransport transport(locRes, attr, link, fakeSocket);
        // 第 1 轮：INIT 触发 enable p2p；socket 未就绪返回 INIT，但 enable 已成功下发
        EXPECT_EQ(transport.GetStatus(), TransportStatus::INIT);
        // 第 2 轮：驱动侧尚未使能，停留等待下次轮询
        EXPECT_EQ(transport.GetStatus(), TransportStatus::INIT);
        // 第 3 轮：驱动侧已使能，p2p 阶段完成进入 socket 建链阶段；socket 仍未就绪，等待建链
        drvStatus = DRV_P2P_STATUS_ENABLE;
        EXPECT_EQ(transport.GetStatus(), TransportStatus::INIT);
    } // 析构：对称 disable p2p，引用计数归零
}

// 非 PCIE（HCCS）链路：设备内互联天然互通，不触发 enable p2p，析构亦不执行 disable
TEST_F(P2PTransportTest, P2PTransport_get_status_non_pcie_skip_enable)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    IpAddress ipAddress("1.0.0.0");
    LinkData link(PortDeploymentType::P2P, LinkProtocol::HCCS, 0, 1, ipAddress, ipAddress, 10, 20, 0);
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    MOCKER(HrtEnableP2P).expects(never());  // 协议过滤：非 PCIE 链路不触发 enable p2p
    MOCKER(HrtDisableP2P).expects(never()); // 未持有引用，析构不执行对称 disable
    SocketStatus socketStatusInit = SocketStatus::INIT;
    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue(socketStatusInit));

    {
        P2PTransport transport(locRes, attr, link, fakeSocket);
        // 首轮轮询：跳过 p2p 使能阶段直接进入 socket 建链阶段，socket 未就绪保持 INIT
        EXPECT_EQ(transport.GetStatus(), TransportStatus::INIT);
        EXPECT_EQ(transport.GetStatus(), TransportStatus::INIT);
    }
}

// 自环链路（对端物理设备即本设备）：enable p2p 不支持与自身执行，跳过使能
TEST_F(P2PTransportTest, P2PTransport_get_status_self_loop_skip_enable)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    IpAddress ipAddress("1.0.0.0");
    LinkData link(PortDeploymentType::P2P, LinkProtocol::PCIE, 0, 1, ipAddress, ipAddress, 10, 10, 0);
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    MOCKER(HrtGetDevicePhyIdByUserDevId)
        .stubs()
        .with(mockcpp::any(), outBound(static_cast<DevId>((DevId)10)))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtEnableP2P).expects(never());  // 自环链路不触发 enable p2p
    MOCKER(HrtDisableP2P).expects(never()); // 未持有引用，析构不执行对称 disable
    SocketStatus socketStatusInit = SocketStatus::INIT;
    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue(socketStatusInit));

    {
        P2PTransport transport(locRes, attr, link, fakeSocket);
        // 首轮轮询：自环跳过 p2p 使能阶段，socket 未就绪保持 INIT
        EXPECT_EQ(transport.GetStatus(), TransportStatus::INIT);
        EXPECT_EQ(transport.GetStatus(), TransportStatus::INIT);
    }
}

// EnableP2P 下发失败：抛出 RuntimeApiException 且不持有引用，析构不执行对称 disable
TEST_F(P2PTransportTest, P2PTransport_get_status_enable_failed_throw)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    IpAddress ipAddress("1.0.0.0");
    LinkData link(PortDeploymentType::P2P, LinkProtocol::PCIE, 0, 1, ipAddress, ipAddress, 10, 20, 0);
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    MOCKER(HrtGetDevicePhyIdByUserDevId)
        .stubs()
        .with(mockcpp::any(), outBound(static_cast<DevId>((DevId)10)))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtEnableP2P).stubs().will(returnValue(HCCL_E_RUNTIME));
    MOCKER(HrtDisableP2P).expects(never()); // enable 失败未增加引用计数，析构不执行 disable

    {
        P2PTransport transport(locRes, attr, link, fakeSocket);
        EXPECT_THROW(transport.GetStatus(), RuntimeApiException);
    }
}

// socket 就绪后推进握手：SOCKET_OK 发送数据长度（SendAsync），SEND_DATA_SIZE 接收对端数据长度（RecvAsync）
TEST_F(P2PTransportTest, P2PTransport_get_status_handshake_advance_when_socket_ready)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    IpAddress ipAddress("1.0.0.0");
    LinkData link(PortDeploymentType::P2P, LinkProtocol::HCCS, 0, 1, ipAddress, ipAddress, 10, 20, 0);
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    MOCKER(HrtEnableP2P).expects(never());
    MOCKER(HrtDisableP2P).expects(never());
    SocketStatus socketStatusOK = SocketStatus::OK;
    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue(socketStatusOK));
    MOCKER_CPP(&Socket::SendAsync).expects(once()); // PrepareSendData 首拍发送交换数据
    MOCKER_CPP(&Socket::RecvAsync).expects(once()); // RecvDataSize 接收对端数据长度

    P2PTransport transport(locRes, attr, link, fakeSocket);
    // 第 1 轮：非 PCIE 跳过使能，socket 就绪，PrepareSendData 推进到 SEND_DATA_SIZE
    EXPECT_EQ(transport.GetStatus(), TransportStatus::SOCKET_OK);
    // 第 2 轮：RecvDataSize 推进到 RECV_DATA_SIZE
    EXPECT_EQ(transport.GetStatus(), TransportStatus::SOCKET_OK);
}

TEST_F(P2PTransportTest, P2PTransport_read_write_read_reduce_write_reduce)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    StubP2PRmaConnection stubRmaConnection;
    RmaConnection* rmaConnection = &stubRmaConnection;
    locRes.connVec.push_back(rmaConnection);

    Stream stream;

    P2PTransport transport(locRes, attr, link, fakeSocket);
    transport.Read(locSlice, rmtSlice, stream);
    transport.Write(locSlice, rmtSlice, stream);

    ReduceIn reduceIn(DataType::INT8, ReduceOp::MAX);
    transport.ReadReduce(locSlice, rmtSlice, reduceIn, stream);
    transport.WriteReduce(locSlice, rmtSlice, reduceIn, stream);
}

TEST_F(P2PTransportTest, P2PTransport_post_wait)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    IpcLocalNotify ipcLocalNotify;
    BaseLocalNotify* validLocalNotify = &ipcLocalNotify;
    locRes.notifyVec.push_back(validLocalNotify);

    std::unique_ptr<IpcRemoteNotify> ipcRemoteNotify = std::make_unique<IpcRemoteNotify>();
    Stream stream;

    P2PTransport transport(locRes, attr, link, fakeSocket);
    transport.rmtNotifyVec.push_back(std::move(ipcRemoteNotify));

    transport.Post(0, stream);
}
