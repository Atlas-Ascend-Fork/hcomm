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
#include "aiv_roce_host_loc_channel.h"
#include "socket.h"
#include "orion_adapter_hccp.h"
#include "hccp.h"
#include "hccp_common.h"
#include "hcomm_c_adpt.h"
#include "buffer/local_rdma_rma_buffer_v2.h"
#include "rdma_handle_manager.h"
#include "cpu_roce_endpoint.h"
#include "mem_device_pub.h"
#include "orion_adapter_rts.h"
#include "acl/acl_rt.h"

#define private public

using namespace hcomm;

class AivRoceHostLocChannelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AivRoceHostLocChannelTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AivRoceHostLocChannelTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in AivRoceHostLocChannelTest SetUP" << std::endl;
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
        std::cout << "A Test case in AivRoceHostLocChannelTest TearDown" << std::endl;
        delete fakeSocket;
    }

    std::shared_ptr<Hccl::Buffer> localBufferPtr;
    std::shared_ptr<Hccl::LocalRdmaRmaBuffer> localRdmaRmaBuffer;
    std::unique_ptr<CpuRoceEndpoint> endpoint;
    EndpointHandle endpointHandle{};
    HcommChannelDesc channelDesc{};
    Hccl::Socket* fakeSocket;
};

static bool g_slabAllocated = false;
static void* g_slabPtr = nullptr;
static size_t g_slabSize = 0;

aclError StubAclrtMalloc(void** devPtr, size_t size, aclrtMemMallocPolicy, aclrtMallocConfig*)
{
    if (devPtr == nullptr) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    *devPtr = std::malloc(size);
    if (*devPtr == nullptr) {
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }
    g_slabAllocated = true;
    g_slabPtr = *devPtr;
    g_slabSize = size;
    return ACL_SUCCESS;
}

aclError StubAclrtMallocFail(void** devPtr, size_t, aclrtMemMallocPolicy, aclrtMallocConfig*)
{
    if (devPtr != nullptr) {
        *devPtr = nullptr;
    }
    return ACL_ERROR_RT_MEMORY_ALLOCATION;
}

aclError StubAclrtFree(void* devPtr)
{
    std::free(devPtr);
    if (devPtr == g_slabPtr) {
        g_slabAllocated = false;
        g_slabPtr = nullptr;
        g_slabSize = 0;
    }
    return ACL_SUCCESS;
}

static void StubHrtMemcpyVoid(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind) {}

static void StubHrtMemcpyReal(void* dst, uint64_t dstMax, const void* src, uint64_t count, Hccl::tagRtMemcpyKind)
{
    if (dst != nullptr && src != nullptr && dstMax >= count) {
        (void)std::memcpy(dst, src, static_cast<size_t>(count));
    }
}

TEST_F(AivRoceHostLocChannelTest, Ut_When_BuildAndGetLocNotifyInfo_Expect_Success)
{
    std::cout << "Start Ut_When_BuildAndGetLocNotifyInfo_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;

    RegedNotifyEntity* notify = nullptr;
    HcclResult ret = channel->BuildAndGetLocNotifyInfo(&notify);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_BuildAndGetLocNotifyInfo_Expect_Success" << std::endl;
}

TEST_F(AivRoceHostLocChannelTest, Ut_When_BuildAndGetRmtNotifyInfo_Expect_Success)
{
    std::cout << "Start Ut_When_BuildAndGetRmtNotifyInfo_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;

    RegedNotifyEntity* notify = nullptr;
    HcclResult ret = channel->BuildAndGetRmtNotifyInfo(&notify);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_BuildAndGetRmtNotifyInfo_Expect_Success" << std::endl;
}

TEST_F(AivRoceHostLocChannelTest, Ut_When_BuildAndGetLocBufInfo_Expect_Success)
{
    std::cout << "Start Ut_When_BuildAndGetLocBufInfo_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;

    std::vector<RegedBufferEntity> locBufList;
    locBufList.resize(channel->bufferNum_);
    RegedBufferEntity* bufferEntity = nullptr;
    HcclResult ret = channel->BuildAndGetLocBufInfo(locBufList, &bufferEntity);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_BuildAndGetLocBufInfo_Expect_Success" << std::endl;
}

TEST_F(AivRoceHostLocChannelTest, Ut_When_BuildAndGetRmtBufInfo_Expect_Success)
{
    std::cout << "Start Ut_When_BuildAndGetRmtBufInfo_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    std::vector<RegedBufferEntity> rmtBufList;
    rmtBufList.resize(channel->bufferNum_);
    RegedBufferEntity* bufferEntity = nullptr;
    HcclResult ret = channel->BuildAndGetRmtBufInfo(rmtBufList, &bufferEntity);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_BuildAndGetRmtBufInfo_Expect_Success" << std::endl;
}

TEST_F(AivRoceHostLocChannelTest, Ut_When_BuildAndGetSqContext_Expect_Success)
{
    std::cout << "Start Ut_When_BuildAndGetSqContext_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;

    std::vector<SqContext> sqList;
    sqList.resize(channel->connNum_);
    SqContext* sqContext = nullptr;
    HcclResult ret = channel->BuildAndGetSqContext(sqList, &sqContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_BuildAndGetSqContext_Expect_Success" << std::endl;
}

TEST_F(AivRoceHostLocChannelTest, Ut_When_BuildAndGetCqContext_Expect_Success)
{
    std::cout << "Start Ut_When_BuildAndGetCqContext_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;

    std::vector<CqContext> cqList;
    cqList.resize(channel->connNum_);
    CqContext* cqContext = nullptr;
    HcclResult ret = channel->BuildAndGetCqContext(cqList, &cqContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_BuildAndGetCqContext_Expect_Success" << std::endl;
}

TEST_F(AivRoceHostLocChannelTest, Ut_When_BuildAndGetDevChannelEntity_Expect_Success)
{
    std::cout << "Start Ut_When_BuildAndGetDevChannelEntity_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyReal)));

    uint64_t devChannelEntityPtr = 0;
    HcclResult ret = channel->BuildAndGetDevChannelEntity(&devChannelEntityPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(devChannelEntityPtr, 0);
    channel->FreeDeviceMemories();
    std::cout << "End Ut_When_BuildAndGetDevChannelEntity_Expect_Success" << std::endl;
}

TEST_F(AivRoceHostLocChannelTest, Ut_When_BuildSlab_Expect_Success)
{
    std::cout << "Start Ut_When_BuildSlab_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyReal)));

    g_slabAllocated = false;
    uint64_t devChannelEntityPtr = 0;
    HcclResult ret = channel->BuildAndGetDevChannelEntity(&devChannelEntityPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(devChannelEntityPtr, 0);
    EXPECT_TRUE(g_slabAllocated);
    EXPECT_NE(channel->devChannelEntitySlab_, nullptr);
    EXPECT_GT(channel->devChannelEntitySlabSize_, sizeof(ChannelEntity));

    ChannelEntity* devEntity = reinterpret_cast<ChannelEntity*>(devChannelEntityPtr);
    EXPECT_EQ(devEntity->engine, static_cast<uint32_t>(COMM_ENGINE_AIV));
    EXPECT_EQ(devEntity->sqNum, 1u);
    EXPECT_EQ(devEntity->cqNum, 1u);

    channel->FreeDeviceMemories();
    std::cout << "End Ut_When_BuildSlab_Expect_Success" << std::endl;
}

TEST_F(AivRoceHostLocChannelTest, Ut_When_BuildSlab_CachedReturn_Expect_SamePtr)
{
    std::cout << "Start Ut_When_BuildSlab_CachedReturn_Expect_SamePtr" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyReal)));

    uint64_t firstPtr = 0;
    HcclResult ret = channel->BuildAndGetDevChannelEntity(&firstPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(firstPtr, 0);

    // Second call should return same pointer without re-allocating
    g_slabAllocated = false;
    uint64_t secondPtr = 0;
    ret = channel->BuildAndGetDevChannelEntity(&secondPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(secondPtr, firstPtr);
    EXPECT_FALSE(g_slabAllocated);

    channel->FreeDeviceMemories();
    std::cout << "End Ut_When_BuildSlab_CachedReturn_Expect_SamePtr" << std::endl;
}

TEST_F(AivRoceHostLocChannelTest, Ut_When_BuildSlab_AclrtMallocFail_Expect_MemoryError)
{
    std::cout << "Start Ut_When_BuildSlab_AclrtMallocFail_Expect_MemoryError" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMallocFail));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyVoid)));

    uint64_t devChannelEntityPtr = 0;
    HcclResult ret = channel->BuildAndGetDevChannelEntity(&devChannelEntityPtr);
    EXPECT_EQ(ret, HCCL_E_MEMORY);
    EXPECT_EQ(devChannelEntityPtr, 0);
    EXPECT_EQ(channel->devChannelEntitySlab_, nullptr);
    EXPECT_EQ(channel->devChannelEntitySlabSize_, 0);

    std::cout << "End Ut_When_BuildSlab_AclrtMallocFail_Expect_MemoryError" << std::endl;
}

TEST_F(AivRoceHostLocChannelTest, Ut_When_Clean_FreesSlab_CanRebuild)
{
    std::cout << "Start Ut_When_Clean_FreesSlab_CanRebuild" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyReal)));

    // First build
    uint64_t firstPtr = 0;
    HcclResult ret = channel->BuildAndGetDevChannelEntity(&firstPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(channel->devChannelEntitySlab_, nullptr);

    // Clean should free the slab
    channel->Clean();
    EXPECT_EQ(channel->devChannelEntitySlab_, nullptr);
    EXPECT_EQ(channel->devChannelEntitySlabSize_, 0);

    // Rebuild should succeed
    g_slabAllocated = false;
    uint64_t secondPtr = 0;
    ret = channel->BuildAndGetDevChannelEntity(&secondPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(secondPtr, 0);
    EXPECT_TRUE(g_slabAllocated);

    channel->FreeDeviceMemories();
    std::cout << "End Ut_When_Clean_FreesSlab_CanRebuild" << std::endl;
}

// ===================== PreAllocDevChannelEntity / FillDevChannelEntity UT =====================

// PreAllocDevChannelEntity: devChannelEntityPtr 为 nullptr
TEST_F(AivRoceHostLocChannelTest, Ut_PreAllocDevChannelEntity_When_PtrNull_Returns_E_PTR)
{
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    EXPECT_EQ(channel->PreAllocDevChannelEntity(nullptr), HCCL_E_PTR);
}

// FillDevChannelEntity: 未预分配（slab 为 nullptr）→ E_INTERNAL
TEST_F(AivRoceHostLocChannelTest, Ut_FillDevChannelEntity_When_NotPreAllocated_Returns_E_INTERNAL)
{
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    EXPECT_EQ(channel->FillDevChannelEntity(), HCCL_E_INTERNAL);
}

// FillDevChannelEntity: 已预分配但 channel 未 READY → E_INTERNAL
TEST_F(AivRoceHostLocChannelTest, Ut_FillDevChannelEntity_When_NotReady_Returns_E_INTERNAL)
{
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->channelStatus_ = ChannelStatus::INIT;
    // 模拟已预分配 slab（设置一个非空指针）
    channel->devChannelEntitySlab_ = reinterpret_cast<void*>(0x1000);
    channel->devChannelEntitySlabSize_ = 4096;
    EXPECT_EQ(channel->FillDevChannelEntity(), HCCL_E_INTERNAL);
    channel->devChannelEntitySlab_ = nullptr;
}

// ===================== PreAllocDevChannelEntity / FillDevChannelEntity 正常路径 UT =====================

// PreAllocDevChannelEntity: 正常分配 slab
TEST_F(AivRoceHostLocChannelTest, Ut_PreAllocDevChannelEntity_When_Normal_ReturnsSuccess)
{
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyReal)));

    uint64_t devChannelEntityPtr = 0;
    EXPECT_EQ(channel->PreAllocDevChannelEntity(&devChannelEntityPtr), HCCL_SUCCESS);
    EXPECT_NE(devChannelEntityPtr, 0);
    channel->FreeDeviceMemories();
}

// PreAllocDevChannelEntity: 已分配则返回缓存指针
TEST_F(AivRoceHostLocChannelTest, Ut_PreAllocDevChannelEntity_When_AlreadyBuilt_ReturnsCached)
{
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyReal)));

    uint64_t firstPtr = 0;
    ASSERT_EQ(channel->PreAllocDevChannelEntity(&firstPtr), HCCL_SUCCESS);
    EXPECT_NE(firstPtr, 0);
    uint64_t secondPtr = 0;
    EXPECT_EQ(channel->PreAllocDevChannelEntity(&secondPtr), HCCL_SUCCESS);
    EXPECT_EQ(secondPtr, firstPtr);
    channel->FreeDeviceMemories();
}

// FillDevChannelEntity: 先 PreAlloc 再 Fill，验证成功
TEST_F(AivRoceHostLocChannelTest, Ut_FillDevChannelEntity_When_PreAllocatedAndReady_ReturnsSuccess)
{
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyReal)));

    uint64_t devChannelEntityPtr = 0;
    ASSERT_EQ(channel->PreAllocDevChannelEntity(&devChannelEntityPtr), HCCL_SUCCESS);
    EXPECT_EQ(channel->FillDevChannelEntity(), HCCL_SUCCESS);
    channel->FreeDeviceMemories();
}
