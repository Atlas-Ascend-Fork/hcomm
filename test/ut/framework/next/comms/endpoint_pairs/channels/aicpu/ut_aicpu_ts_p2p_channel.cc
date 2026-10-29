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
    StubEndpointForP2pChannel() : Endpoint(MakeDesc()) {}

    HcclResult Init() override { return HCCL_SUCCESS; }
    // 内存方法在 mgr 上，Endpoint 不再承载
    LocalRegedMemMgr* GetLocalRegMemMgr() override { return nullptr; }
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override { return ctxHandle_ != nullptr; }

private:
    void* ctxHandle_{nullptr};
    static EndpointDesc MakeDesc()
    {
        EndpointDesc desc{};
        desc.protocol = COMM_PROTOCOL_PCIE;
        desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        desc.loc.device.devPhyId = 3;
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

    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.local_.commonRes.bufferVec.size(), 1U);
    EXPECT_EQ(ch.local_.commonRes.bufferVec[0], localRmaBuffer.get());
}

TEST_F(AicpuTsP2pChannelTest, UT_ParseInputParam_When_ExchangeAllMemsTrue_Expect_FillCommonRes)
{
    StubEndpointForP2pChannel endpoint;
    HcommChannelDesc desc{};
    desc.exchangeAllMems = true;
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);
    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubP2pGetAllMemHandlesOne));

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.local_.commonRes.bufferVec.size(), 1U);
    EXPECT_EQ(ch.local_.commonRes.bufferVec[0]->GetAddr(), 0x4000U);
}

// P2P 收编走 V2 后，host V2 分流依赖 GetChannelKind()==AICPU_TS_PCIE 识别；对齐
// ut_aicpu_ts_roce_host_loc_channel.cc 的 Ut_When_GetChannelKind_Expect_RoceHostLoc。
TEST_F(AicpuTsP2pChannelTest, Ut_When_GetChannelKind_Expect_Pcie)
{
    StubEndpointForP2pChannel endpoint;
    HcommChannelDesc desc{};
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

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
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_EQ(ch.traits_.socketBuildMode, SocketBuildMode::LISTEN_AND_BORROW);
}

TEST_F(AicpuTsP2pChannelTest, Ut_StartListen_WhenSocketNull_Expect_PcieListenPortTypeThrows)
{
    StubEndpointForP2pChannel endpoint; // MakeDesc 未配 commAddr，LISTEN 分支的 CommAddrToIpAddress 需打桩
    auto localRmaBuffer = MakeLocalIpcRmaBuffer(0x3200U, 0x1000U, "p2p_listen");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localRmaBuffer.get())};
    HcommChannelDesc desc{};
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));
    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.local_.socket, nullptr);

    // 打桩 commAddr 解析成功以抵达真实抛点：自建监听已收口进 StartListen 的 LISTEN 分支，P2P 无专属
    // 监听端口类型，GetListenPortType() default 分支返回 DevNetPortType(PCIE)，而 DevNetPortType
    // 合法集 {TCP,RDMA,UB} 不含 PCIE，PortData 构造即抛 InvalidParamsException。与原实现完全一致
    // （原代码即 DevNetPortType(PCIE)，自带 "TODO PROTOTYPE P2P?" 注释），属存量原型地雷而非收口
    // 回归，此处钉住真实行为；PortData 之后的 Create/GetSocket/借出不可达，无需打桩。P2P 原型完善时
    // 应引入 P2P 族端口类型（P2PPortType 合法集含 PCIE）。urma 路径（DevNetPortType(UB) 合法）的
    // LISTEN_AND_BORROW 全流程行为由 ut_aicpu_ts_urma_channel.cc 对应用例钉住。
    // （Listen() 虚函数由链接级 llt 桩接管，MOCKER_CPP 打虚函数拿到虚表偏移伪指针会段错误）
    MOCKER(CommAddrToIpAddress).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_THROW(ch.StartListen(), Hccl::InvalidParamsException);
}

// ===================== Resume 错误码传播钉住 =====================
// Resume 内 BuildSocket/BuildConnection/BuildBuffer 裸调已改 CHK_RET：重建失败须上抛，不得静默吞错后
// 返回 SUCCESS。P2P 的 socketBuildMode 是 LISTEN_AND_BORROW 而非 NONE，BuildSocket 须 mock 直通 SUCCESS
// （不打桩则 LISTEN 分支撞 PCIE 监听端口类型存量地雷直接抛异常），失败注入聚焦后两个 hook

TEST_F(AicpuTsP2pChannelTest, Ut_Resume_WhenBuildConnectionFails_Expect_ErrorPropagated)
{
    StubEndpointForP2pChannel endpoint;
    auto localRmaBuffer = MakeLocalIpcRmaBuffer(0x3300U, 0x1000U, "p2p_resume_bc");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localRmaBuffer.get())};
    HcommChannelDesc desc{};
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

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
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

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
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);
    EXPECT_EQ(ch.GetNotifyBuildMode(), NotifyBuildMode::BUILD_LOCAL);
}

TEST_F(AicpuTsP2pChannelTest, Ut_BuildNotify_Unified_Expect_IpcLocalNotifyCreatedAndRegistered)
{
    StubEndpointForP2pChannel endpoint;
    HcommChannelDesc desc{};
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);
    ch.local_.notifyNum = 2;
    ch.traits_.notifyBuildMode = NotifyBuildMode::BUILD_LOCAL;
    ASSERT_EQ(ch.BuildNotify(), HCCL_SUCCESS);
    EXPECT_EQ(ch.local_.notifies.size(), 2U);
    EXPECT_EQ(ch.local_.commonRes.notifyVec.size(), 2U);
    EXPECT_EQ(ch.local_.commonRes.notifyVec[0], ch.local_.notifies[0].get());
    EXPECT_NE(dynamic_cast<Hccl::IpcLocalNotify*>(ch.local_.notifies[0].get()), nullptr); // 工厂分派 IpcLocalNotify
}
