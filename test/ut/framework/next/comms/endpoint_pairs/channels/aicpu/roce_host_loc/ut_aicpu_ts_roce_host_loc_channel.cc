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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include <vector>
#include <iostream>
#include <cstring>
#include "aicpu_ts_roce_host_loc_channel.h"
#include "../sockets/socket_mgr.h"
#include "roce_host_loc_connection.h"
#include "socket.h"
#include "orion_adapter_hccp.h"
#include "hccp.h"
#include "hccp_common.h"
#include "hcomm_c_adpt.h"
#include "buffer/local_rdma_rma_buffer_v2.h"
#include "rdma_handle_manager.h"
#include "cpu_roce_endpoint.h"
#include "binary_stream.h"
#include "mem_device_pub.h"
#include "aicpu_res_package_helper.h"
#include "exchange_rdma_buffer_dto.h"
#include "channels/host/exchange_rdma_conn_dto.h"
#include "user_remote_mem_getter.h"
#include "acl/acl_rt.h"

#define private public

using namespace hcomm;

class AicpuTsRoceHostLocChannelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AicpuTsRoceHostLocChannelTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AicpuTsRoceHostLocChannelTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in AicpuTsRoceHostLocChannelTest SetUP" << std::endl;
        Hccl::DevType dev = Hccl::DevType::DEV_TYPE_950;
        MOCKER(Hccl::HrtGetDevice).stubs().will(returnValue(0));
        MOCKER(Hccl::HrtGetDeviceType).stubs().will(returnValue(dev));
        // hrtGetDeviceType（全局符号，CheckDevTypeSupport 调用）mock 为 950
        MOCKER(hrtGetDeviceType).stubs().will(invoke(+[](DevType& t) -> HcclResult {
            t = DevType::DEV_TYPE_950;
            return HCCL_SUCCESS;
        }));
        MOCKER(Hccl::HrtGetDevicePhyIdByUserDevId)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(static_cast<Hccl::DevId>(0)));
        RdmaHandle rdmaHandle = (void*)0x1000000;
        MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));

        Hccl::IpAddress localIp("1.0.0.0");
        Hccl::IpAddress remoteIp("2.0.0.0");
        fakeSocket = new Hccl::Socket(
            nullptr, localIp, 60001, remoteIp, "_0_1_", Hccl::SocketRole::SERVER, Hccl::NicType::HOST_NIC_TYPE);

        MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
        MOCKER(Hccl::HrtRaNdaQpCreate)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
        MOCKER(Hccl::HrtRaNdaCqCreate)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
        MOCKER(Hccl::HrtRaNdaCqDestroy).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER(Hccl::HrtRaQpDestroy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER(RaGetQpAttr).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
        MOCKER_CPP(&AicpuTsRoceHostLocChannel::BuildNotifyValueBuffer).stubs().will(returnValue(HCCL_SUCCESS));

        EndpointDesc endpointDesc{};
        endpointDesc.protocol = COMM_PROTOCOL_ROCE;
        endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
        endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        endpoint = std::make_unique<CpuRoceEndpoint>(endpointDesc);
        endpoint->Init();
        endpointHandle = static_cast<EndpointHandle>(endpoint.get());

        EndpointDesc endpointDesc2;
        endpointDesc2.protocol = COMM_PROTOCOL_ROCE;
        endpointDesc2.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        endpointDesc2.commAddr.addr = remoteIp.GetBinaryAddress().addr;
        endpointDesc2.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        channelDesc.remoteEndpoint = endpointDesc2;
        channelDesc.notifyNum = 3;
        channelDesc.port = 60001;
        void* fsocket = static_cast<void*>(fakeSocket);
        channelDesc.socket = fsocket;

        localBufferPtr = std::make_shared<Hccl::Buffer>(666);
        localRdmaRmaBuffer = std::make_shared<Hccl::LocalRdmaRmaBuffer>(localBufferPtr, rdmaHandle);
        void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
        channelDesc.memHandles = &memHandle;
        channelDesc.memHandleNum = 1;
        channelDesc.exchangeAllMems = false;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in AicpuTsRoceHostLocChannelTest TearDown" << std::endl;
        delete fakeSocket;
    }

    std::shared_ptr<Hccl::Buffer> localBufferPtr;
    std::shared_ptr<Hccl::LocalRdmaRmaBuffer> localRdmaRmaBuffer;
    std::unique_ptr<CpuRoceEndpoint> endpoint;
    EndpointHandle endpointHandle{};
    HcommChannelDesc channelDesc{};
    Hccl::Socket* fakeSocket;
};

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_Normal_Init_Expect_HCCL_SUCCESS)
{
    std::cout << "Start Ut_When_Normal_Init_Expect_HCCL_SUCCESS" << std::endl;

    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&RoceHostLocConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsRoceHostLocChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsRoceHostLocChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsRoceHostLocChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsRoceHostLocChannel::NotifyVecUnpack)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsRoceHostLocChannel::ConnVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsRoceHostLocChannel::RmtBufferVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    // 本用例只验证状态机流转，数据交换细节由独立用例覆盖，与host侧用例保持一致mock掉ExchangeData
    MOCKER_CPP(&AicpuTsRoceHostLocChannel::ExchangeData).stubs().will(returnValue(HCCL_SUCCESS));

    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(channel->Init(), HCCL_SUCCESS);
    // connect
    hcomm::ChannelStatus status = channel->GetStatus();
    EXPECT_EQ(channel->rdmaStatus_, AicpuTsRoceHostLocChannel::RdmaStatus::SOCKET_OK);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = channel->GetStatus();
    EXPECT_EQ(channel->rdmaStatus_, AicpuTsRoceHostLocChannel::RdmaStatus::QP_CREATED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = channel->GetStatus();
    EXPECT_EQ(channel->rdmaStatus_, AicpuTsRoceHostLocChannel::RdmaStatus::DATA_EXCHANGE);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = channel->GetStatus();
    EXPECT_EQ(channel->rdmaStatus_, AicpuTsRoceHostLocChannel::RdmaStatus::CONN_OK);
    EXPECT_EQ(status, ChannelStatus::READY);

    std::cout << "End Ut_When_Normal_Init_Expect_HCCL_SUCCESS" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetNotifyNum_Expect_Correct)
{
    std::cout << "Start Ut_When_GetNotifyNum_Expect_Correct" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();

    uint32_t notifyNum = 0;
    HcclResult ret = channel->GetNotifyNum(&notifyNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(notifyNum, channelDesc.notifyNum);

    std::cout << "End Ut_When_GetNotifyNum_Expect_Correct" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetBufferNum_Expect_Correct)
{
    std::cout << "Start Ut_When_GetBufferNum_Expect_Correct" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();

    uint32_t bufferNum = 0;
    HcclResult ret = channel->GetBufferNum(&bufferNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_GetBufferNum_Expect_Correct" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_Describe_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_Describe_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();

    std::string desc = channel->Describe();
    std::cout << desc << std::endl;

    std::cout << "End Ut_When_Describe_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetQpNum_Expect_Success)
{
    std::cout << "Start Ut_When_GetQpNum_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();

    uint32_t qpNum = 0;
    HcclResult ret = channel->GetQpNum(&qpNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_GetQpNum_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_BuildSocket_NullSocket_Expect_Success)
{
    std::cout << "Start Ut_When_BuildSocket_NullSocket_Expect_Success" << std::endl;
    channelDesc.socket = nullptr;
    channelDesc.port = 0;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    MOCKER_CPP(&hcomm::SocketMgr::GetSocket)
        .stubs()
        .with(mockcpp::any(), outBound(fakeSocket))
        .will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(channel->Init(), HCCL_SUCCESS);
    std::cout << "End Ut_When_BuildSocket_NullSocket_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_ModifyQp_Expect_Success)
{
    std::cout << "Start Ut_When_ModifyQp_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    MOCKER_CPP(&RoceHostLocConnection::ParseRmtExchangeDto).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    HcclResult ret = channel->ModifyQp();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::cout << "End Ut_When_ModifyQp_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetRemoteMem_NonEmpty_Expect_Success)
{
    std::cout << "Start Ut_When_GetRemoteMem_NonEmpty_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));
    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    char** memInfos = nullptr;
    HcclResult ret = channel->GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memNum, 1u);
    std::cout << "End Ut_When_GetRemoteMems_NonEmpty_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetRemoteMems_CacheValid_Expect_Success)
{
    std::cout << "Start Ut_When_GetRemoteMems_CacheValid_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x2000000));
    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = channel->GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = channel->GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::cout << "End Ut_When_GetRemoteMems_CacheValid_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetRemoteMems_NullPtr_Expect_Error)
{
    std::cout << "Start Ut_When_GetRemoteMems_NullPtr_Expect_Error" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    char** memInfos = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = channel->GetRemoteMems(&memNum, nullptr, &memInfos);
    EXPECT_NE(ret, HCCL_SUCCESS);
    CommMem* remoteMem = nullptr;
    ret = channel->GetRemoteMems(&memNum, &remoteMem, nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
    ret = channel->GetRemoteMems(nullptr, &remoteMem, &memInfos);
    EXPECT_NE(ret, HCCL_SUCCESS);
    std::cout << "End Ut_When_GetRemoteMems_NullPtr_Expect_Error" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetRemoteMems_Empty_Expect_Success)
{
    std::cout << "Start Ut_When_GetRemoteMems_Empty_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->rmtRmaBuffers_.clear();
    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = channel->GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memNum, 0U);
    std::cout << "End Ut_When_GetRemoteMems_Empty_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetLocalNotifyUniqueIds_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetLocalNotifyUniqueIds_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    auto result = channel->GetLocalNotifyUniqueIds();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetLocalNotifyUniqueIds_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetRemoteNotifyUniqueIds_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetRemoteNotifyUniqueIds_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->remoteNotifies_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));
    auto result = channel->GetRemoteNotifyUniqueIds();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetRemoteNotifyUniqueIds_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetNotifyValueBufferUniqueIds_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetNotifyValueBufferUniqueIds_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    // BuildNotifyValueBuffer is mocked in SetUp, so create notifyValueBuffer_ here
    channel->notifyValueBuffer_ = std::make_unique<Hccl::LocalRdmaRmaBuffer>(localBufferPtr, (void*)0x1000000);
    auto result = channel->GetNotifyValueBufferUniqueIds();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetNotifyValueBufferUniqueIds_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetLocBufferUniqueIds_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetLocBufferUniqueIds_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    auto result = channel->GetLocBufferUniqueIds();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetLocBufferUniqueIds_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetLocBufferUniqueIds_NullBuffer_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetLocBufferUniqueIds_NullBuffer_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->localRmaBuffers_.push_back(nullptr);
    channel->bufferNum_ = channel->localRmaBuffers_.size();
    auto result = channel->GetLocBufferUniqueIds();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetLocBufferUniqueIds_NullBuffer_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetRmtBufferUniqueIds_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetRmtBufferUniqueIds_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));
    auto result = channel->GetRmtBufferUniqueIds();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetRmtBufferUniqueIds_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetRmtBufferUniqueIds_NullBuffer_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetRmtBufferUniqueIds_NullBuffer_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->rmtRmaBuffers_.push_back(nullptr);
    auto result = channel->GetRmtBufferUniqueIds();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetRmtBufferUniqueIds_NullBuffer_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetConnUniqueIds_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetConnUniqueIds_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    auto result = channel->GetConnUniqueIds();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetConnUniqueIds_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetUniqueId_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetUniqueId_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));
    // BuildNotifyValueBuffer is mocked in SetUp, so create notifyValueBuffer_ here
    channel->notifyValueBuffer_ = std::make_unique<Hccl::LocalRdmaRmaBuffer>(localBufferPtr, (void*)0x1000000);
    auto result = channel->GetUniqueId();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetUniqueId_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_GetChannelKind_Expect_RoceHostLoc)
{
    std::cout << "Start Ut_When_GetChannelKind_Expect_RoceHostLoc" << std::endl;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    EXPECT_EQ(channel->GetChannelKind(), HcommChannelKind::AICPU_TS_ROCE_HOST_LOC);
    std::cout << "End Ut_When_GetChannelKind_Expect_RoceHostLoc" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_Clean_Expect_Success)
{
    std::cout << "Start Ut_When_Clean_Expect_Success" << std::endl;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    EXPECT_EQ(channel->Clean(), HCCL_SUCCESS);
    std::cout << "End Ut_When_Clean_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_Resume_Expect_Success)
{
    std::cout << "Start Ut_When_Resume_Expect_Success" << std::endl;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    EXPECT_EQ(channel->Resume(), HCCL_SUCCESS);
    std::cout << "End Ut_When_Resume_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_When_H2DResPack_Expect_Success)
{
    std::cout << "Start Ut_When_H2DResPack_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));
    // BuildNotifyValueBuffer is mocked in SetUp, so create notifyValueBuffer_ here
    channel->notifyValueBuffer_ = std::make_unique<Hccl::LocalRdmaRmaBuffer>(localBufferPtr, (void*)0x1000000);
    std::vector<char> buffer;
    HcclResult ret = channel->H2DResPack(buffer);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::cout << "End Ut_When_H2DResPack_Expect_Success" << std::endl;
}

// ==================== 对端声明长度/数量校验（防超大内存申请） ====================

static bool RecvExchangeSizeZeroStub(Hccl::Socket* socket, void* buf, uint32_t size)
{
    (void)socket;
    if (buf != nullptr && size >= sizeof(uint64_t)) {
        uint64_t sizeZero = 0;
        memcpy_s(buf, size, &sizeZero, sizeof(sizeZero));
    }
    return true;
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_ExchangeData_When_RecvSizeZero_Expect_E_PARA)
{
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->socket_ = fakeSocket;
    MOCKER_CPP(&AicpuTsRoceHostLocChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsRoceHostLocChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsRoceHostLocChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::Send).stubs().will(returnValue(true));
    MOCKER_CPP(&Hccl::Socket::Recv).stubs().will(invoke(RecvExchangeSizeZeroStub));

    EXPECT_EQ(channel->ExchangeData(), HCCL_E_PARA);
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_RmtBufferVecUnpackProc_When_RmtNumExceedMax_Expect_E_PARA)
{
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);

    Hccl::BinaryStream binaryStream;
    u32 rmtNum = MAX_BUFFER_NUM + 1;
    binaryStream << rmtNum;
    std::vector<char> data{};
    binaryStream.Dump(data);
    Hccl::BinaryStream recvStream(data);

    EXPECT_EQ(channel->RmtBufferVecUnpackProc(recvStream), HCCL_E_PARA);
}

TEST_F(AicpuTsRoceHostLocChannelTest, Ut_RmtBufferVecUnpackProc_When_RmtNumNormal_Expect_SUCCESS)
{
    auto channel = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->bufferNum_ = 1;

    Hccl::BinaryStream binaryStream;
    u32 rmtNum = 1;
    u32 pos = 0;
    binaryStream << rmtNum;
    binaryStream << pos;
    Hccl::ExchangeRdmaBufferDto dto;
    dto.Serialize(binaryStream);
    std::vector<char> data{};
    binaryStream.Dump(data);
    Hccl::BinaryStream recvStream(data);

    EXPECT_EQ(channel->RmtBufferVecUnpackProc(recvStream), HCCL_SUCCESS);
    EXPECT_EQ(channel->rmtRmaBuffers_.size(), 1U);
}
