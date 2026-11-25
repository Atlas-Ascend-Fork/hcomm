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

#define private public
#define protected public
#include "aicpu/aicpu_ts_p2p_channel.h"
#include "endpoint.h"
#include "buffer/local_ipc_rma_buffer_v2.h"
#include "makebufs_helper.h"
#include "../sockets/socket_mgr.h"
#undef private
#undef protected
#include "adapter_rts_common.h"
#include "orion_adpt_utils.h"
#include "socket_handle_manager.h"
using namespace hcomm;

namespace {
class StubEndpointForP2pChannel : public Endpoint {
public:
    explicit StubEndpointForP2pChannel(bool idCommAddr = false) : Endpoint(MakeDesc(idCommAddr)) {}

    HcclResult Init() override { return HCCL_SUCCESS; }
    // 内存方法在 mgr 上，Endpoint 不再承载
    LocalRegedMemMgr* GetLocalRegMemMgr() override { return nullptr; }
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override { return ctxHandle_ != nullptr; }

private:
    void* ctxHandle_{nullptr};
    static EndpointDesc MakeDesc(bool idCommAddr)
    {
        EndpointDesc desc{};
        desc.protocol = COMM_PROTOCOL_PCIE;
        desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        desc.loc.device.devPhyId = 3;
        if (idCommAddr) {
            desc.commAddr.type = COMM_ADDR_TYPE_ID; // rank graph 下发的 PCIE 点位以 ID 标识，非 IP
        }
        return desc;
    }
};

class NullBufRmaBuffer : public Hccl::LocalRmaBuffer {
public:
    NullBufRmaBuffer() : Hccl::LocalRmaBuffer(nullptr, Hccl::RmaType::IPC) {}
    std::string Describe() const override { return "NullBufRmaBuffer"; }
};

std::shared_ptr<Hccl::LocalIpcRmaBuffer> MakeLocalIpcRmaBuffer(uintptr_t addr, u64 size, const char* tag)
{
    auto buffer = std::make_shared<Hccl::Buffer>(addr, size, HCCL_MEM_TYPE_DEVICE, tag);
    return std::make_shared<Hccl::LocalIpcRmaBuffer>(buffer);
}

HcommResult StubP2pGetAllMemHandlesOne(EndpointHandle, void** memHandles, uint32_t* memHandleNum)
{
    static auto localBuffer = MakeLocalIpcRmaBuffer(0x4000U, 0x1000U, "p2p_all");
    static std::shared_ptr<Hccl::LocalIpcRmaBuffer> localBuffers[1] = {localBuffer};
    *memHandles = localBuffers;
    *memHandleNum = 1;
    return HCCL_SUCCESS;
}
} // namespace

class AicpuTsP2pChannelTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(AicpuTsP2pChannelTest, UT_MakeRmaBufferVecFromMemHandles_When_NullHandle_Expect_ReturnHCCL_E_PTR)
{
    HcommMemHandle memHandles[1] = {nullptr};
    std::vector<Hccl::LocalRmaBuffer*> bufferVec;

    EXPECT_EQ(MakeRmaBufferVecFromMemHandles(memHandles, 1, bufferVec, "AicpuTsP2pChannel"), HCCL_E_PTR);
    EXPECT_TRUE(bufferVec.empty());
}

TEST_F(AicpuTsP2pChannelTest, UT_MakeRmaBufferVecFromMemHandles_When_ZeroHandles_Expect_EmptyBufferVec)
{
    std::vector<Hccl::LocalRmaBuffer*> bufferVec{reinterpret_cast<Hccl::LocalRmaBuffer*>(0x1)};

    ASSERT_EQ(MakeRmaBufferVecFromMemHandles(nullptr, 0, bufferVec, "AicpuTsP2pChannel"), HCCL_SUCCESS);
    EXPECT_TRUE(bufferVec.empty());
}

TEST_F(AicpuTsP2pChannelTest, UT_MakeRmaBufferVecFromMemHandles_When_ValidHandle_Expect_FillBufferVec)
{
    auto localRmaBuffer = MakeLocalIpcRmaBuffer(0x1000U, 0x2000U, "p2p_ut");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localRmaBuffer.get())};
    std::vector<Hccl::LocalRmaBuffer*> bufferVec;

    ASSERT_EQ(MakeRmaBufferVecFromMemHandles(memHandles, 1, bufferVec, "AicpuTsP2pChannel"), HCCL_SUCCESS);
    ASSERT_EQ(bufferVec.size(), 1U);
    EXPECT_EQ(bufferVec[0], localRmaBuffer.get());
}

TEST_F(AicpuTsP2pChannelTest, UT_MakeRmaBufferVecFromMemHandles_When_GetBufNull_Expect_ReturnHCCL_E_PTR)
{
    NullBufRmaBuffer nullBuf;
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(&nullBuf)};
    std::vector<Hccl::LocalRmaBuffer*> bufferVec;

    EXPECT_EQ(MakeRmaBufferVecFromMemHandles(memHandles, 1, bufferVec, "AicpuTsP2pChannel"), HCCL_E_PTR);
    EXPECT_TRUE(bufferVec.empty());
}

TEST_F(AicpuTsP2pChannelTest, UT_ParseInputParam_When_ExchangeAllMemsFalse_Expect_FillCommonRes)
{
    StubEndpointForP2pChannel endpoint;
    auto localRmaBuffer = MakeLocalIpcRmaBuffer(0x3000U, 0x1000U, "p2p_parse");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localRmaBuffer.get())};
    HcommChannelDesc desc{};
    desc.exchangeAllMems = false;
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;

    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc, COMM_ENGINE_RESERVED);

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.local_.commonRes.bufferVec.size(), 1U);
    EXPECT_EQ(ch.local_.commonRes.bufferVec[0], localRmaBuffer.get());
}

TEST_F(AicpuTsP2pChannelTest, UT_ParseInputParam_When_ExchangeAllMemsTrue_Expect_FillCommonRes)
{
    StubEndpointForP2pChannel endpoint;
    HcommChannelDesc desc{};
    desc.exchangeAllMems = true;
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc, COMM_ENGINE_RESERVED);
    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubP2pGetAllMemHandlesOne));

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.local_.commonRes.bufferVec.size(), 1U);
    EXPECT_EQ(ch.local_.commonRes.bufferVec[0]->GetAddr(), 0x4000U);
}

// PCIE 点位在 rank graph 中以 COMM_ADDR_TYPE_ID 下发（非 IP），建链不依赖 IP：
// ParseInputParam 对 local/remote 的 ID 类型 commAddr 均应成功，不得因地址转换返回 E_NOT_SUPPORT。
TEST_F(AicpuTsP2pChannelTest, Ut_ParseInputParam_When_CommAddrTypeId_Expect_Success)
{
    StubEndpointForP2pChannel endpoint(true);
    auto localRmaBuffer = MakeLocalIpcRmaBuffer(0x3600U, 0x1000U, "p2p_id_addr");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localRmaBuffer.get())};
    HcommChannelDesc desc{};
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_PCIE;
    desc.remoteEndpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_ID;
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc, COMM_ENGINE_RESERVED);

    EXPECT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_EQ(ch.local_.ep.commAddr.type, COMM_ADDR_TYPE_ID);
    EXPECT_EQ(ch.remote_.ep.commAddr.type, COMM_ADDR_TYPE_ID);
}

// P2P 收编走 V2 后，host V2 分流依赖 GetChannelKind()==AICPU_TS_PCIE 识别；对齐
// ut_aicpu_ts_roce_host_loc_channel.cc 的 Ut_When_GetChannelKind_Expect_RoceHostLoc。
TEST_F(AicpuTsP2pChannelTest, Ut_When_GetChannelKind_Expect_Pcie)
{
    StubEndpointForP2pChannel endpoint;
    HcommChannelDesc desc{};
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc, COMM_ENGINE_RESERVED);

    EXPECT_EQ(ch.GetChannelKind(), HcommChannelKind::AICPU_TS_PCIE);
}

// ===================== BuildSocket 收敛基类后的钉住用例 =====================
// P2P 为 LISTEN_AND_BORROW（模式钉住见上例）。但 LISTEN 分支 P2P 无专属监听端口类型：
// GetListenPortType() default 分支返回 DevNetPortType(PCIE)，DevNetPortType 合法集 {TCP,RDMA,UB}
// 不含 PCIE，PortData 构造即抛——原实现同构（自带 "TODO PROTOTYPE P2P?"），存量原型地雷，保真钉住

TEST_F(AicpuTsP2pChannelTest, Ut_ParseInputParam_Expect_ListenAndBorrowMode)
{
    StubEndpointForP2pChannel endpoint;
    auto localRmaBuffer = MakeLocalIpcRmaBuffer(0x3100U, 0x1000U, "p2p_mode");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localRmaBuffer.get())};
    HcommChannelDesc desc{};
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc, COMM_ENGINE_RESERVED);

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_EQ(ch.traits_.socketBuildMode, SocketBuildMode::LISTEN_AND_BORROW);
}

TEST_F(AicpuTsP2pChannelTest, Ut_StartListen_WhenSocketNull_Expect_NoSelfListen)
{
    StubEndpointForP2pChannel endpoint;
    auto localRmaBuffer = MakeLocalIpcRmaBuffer(0x3200U, 0x1000U, "p2p_listen");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localRmaBuffer.get())};
    HcommChannelDesc desc{};
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc, COMM_ENGINE_RESERVED);

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));
    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.local_.socket, nullptr);

    EXPECT_EQ(ch.StartListen(), HCCL_SUCCESS);
    EXPECT_EQ(ch.local_.serverSocket, nullptr);
}

// ===================== Resume 错误码传播钉住 =====================
// Resume 内 BuildSocket/BuildConnection/BuildBuffer 裸调已改 CHK_RET：重建失败须上抛，不得静默吞错后
// 返回 SUCCESS。

TEST_F(AicpuTsP2pChannelTest, Ut_Resume_WhenBuildConnectionFails_Expect_ErrorPropagated)
{
    StubEndpointForP2pChannel endpoint;
    auto localRmaBuffer = MakeLocalIpcRmaBuffer(0x3300U, 0x1000U, "p2p_resume_bc");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localRmaBuffer.get())};
    HcommChannelDesc desc{};
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc, COMM_ENGINE_RESERVED);

    MOCKER_CPP_VIRTUAL(ch, &AicpuTsP2pChannel::BuildSocket, HcclResult(AicpuTsP2pChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(ch, &AicpuTsP2pChannel::BuildConnection, HcclResult(AicpuTsP2pChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    EXPECT_NE(ch.Resume(), HCCL_SUCCESS);
}

TEST_F(AicpuTsP2pChannelTest, Ut_Resume_WhenBuildBufferFails_Expect_ErrorPropagated)
{
    StubEndpointForP2pChannel endpoint;
    auto localRmaBuffer = MakeLocalIpcRmaBuffer(0x3400U, 0x1000U, "p2p_resume_bb");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localRmaBuffer.get())};
    HcommChannelDesc desc{};
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc, COMM_ENGINE_RESERVED);

    MOCKER_CPP_VIRTUAL(ch, &AicpuTsP2pChannel::BuildSocket, HcclResult(AicpuTsP2pChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(ch, &AicpuTsP2pChannel::BuildConnection, HcclResult(AicpuTsP2pChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(ch, &AicpuTsP2pChannel::BuildBuffer, HcclResult(AicpuTsP2pChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    EXPECT_NE(ch.Resume(), HCCL_SUCCESS);
}

// ===================== BuildNotify 收敛基类后的钉住用例 =====================
// PCIE 折算 BUILD_LOCAL/IpcLocalNotify：统一流程构建 + notifyVec/local_.notifies 双注册
// （IpcLocalNotify 构造不依赖 rdmaHandle，RtsNotify/SetIpcName 链走 llt 桩）
TEST_F(AicpuTsP2pChannelTest, Ut_GetNotifyBuildMode_Expect_BUILD_LOCAL)
{
    StubEndpointForP2pChannel endpoint;
    HcommChannelDesc desc{};
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc, COMM_ENGINE_RESERVED);
    EXPECT_EQ(ch.GetNotifyBuildMode(), NotifyBuildMode::BUILD_LOCAL);
}

TEST_F(AicpuTsP2pChannelTest, Ut_BuildNotify_Unified_Expect_IpcLocalNotifyCreatedAndRegistered)
{
    StubEndpointForP2pChannel endpoint;
    HcommChannelDesc desc{};
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc, COMM_ENGINE_RESERVED);
    ch.local_.notifyNum = 2;
    ch.traits_.notifyBuildMode = NotifyBuildMode::BUILD_LOCAL;
    ASSERT_EQ(ch.BuildNotify(), HCCL_SUCCESS);
    EXPECT_EQ(ch.local_.notifies.size(), 2U);
    EXPECT_EQ(ch.local_.commonRes.notifyVec.size(), 2U);
    EXPECT_EQ(ch.local_.commonRes.notifyVec[0], ch.local_.notifies[0].get());
    EXPECT_NE(dynamic_cast<Hccl::IpcLocalNotify*>(ch.local_.notifies[0].get()), nullptr); // 工厂分派 IpcLocalNotify
}
