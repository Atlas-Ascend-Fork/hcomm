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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#include "env_config/env_config_v2.h"

#define private public
#include "ccu_urma_channel.h"
#include "ccu_dev_mgr_imp.h"
#include "local_ub_rma_buffer.h"
#include "orion_adpt_utils.h"
#include "socket.h"
#undef private
#include "hcomm_c_adpt.h" // HcommEndpointGet（BuildSocket 注入路径打桩）

using namespace hcomm;

namespace hcomm {
HcclResult BuildBufferInfos(
    HcommMemHandle* memHandles, uint32_t memHandleNum, std::vector<CcuTransport::CclBufferInfo>& bufferInfos);
}

class CcuUrmaChannelTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

static UrmaEndpoint& TestUrmaEndpoint()
{
    static UrmaEndpoint endpoint{EndpointDesc{}};
    return endpoint;
}

TEST_F(CcuUrmaChannelTest, Ut_Clean_When_ImplIsNull_Expect_HCCL_E_PTR)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);

    auto ret = ch.Clean();
    EXPECT_EQ(ret, HcclResult::HCCL_E_PTR);
}

// Minimal test-only connection deriving from CcuConnection to avoid heavy Init()
class TestCcuConnection : public CcuConnection {
public:
    TestCcuConnection(
        const CommAddr& locAddr, const CommAddr& rmtAddr, const CcuChannelInfo& channelInfo,
        const std::vector<CcuJetty*>& ccuJettys, uint32_t qos = Hccl::UB_QOS_DEFAULT)
        : CcuConnection(locAddr, rmtAddr, channelInfo, ccuJettys, qos)
    {}
    // Do not call Init(); use default base behavior for Clean()
};

TEST_F(CcuUrmaChannelTest, Ut_Clean_When_ImplIsPresent_Expect_HCCL_SUCCESS)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);

    // Prepare minimal safe objects to construct a CcuTransport without heavy Init
    CommAddr locAddr{};
    CommAddr rmtAddr{};
    CcuChannelInfo channelInfo{};
    std::vector<CcuJetty*> jettys{};

    // Create a test connection (does not call Init)
    std::unique_ptr<CcuConnection> conn(
        new TestCcuConnection(locAddr, rmtAddr, channelInfo, jettys, Hccl::UB_QOS_DEFAULT));

    // Fake socket pointer (not dereferenced by Clean())
    Hccl::Socket* fakeSocket = reinterpret_cast<Hccl::Socket*>(0x1);

    // Prepare a simple buffer info
    CcuTransport::CclBufferInfo bufInfo(0x1000, 0x100, 1, 1);

    // Construct transport instance (constructor does not call Init)
    std::unique_ptr<CcuTransport> transport(new CcuTransport(fakeSocket, std::move(conn), bufInfo));

    // Inject into channel (we used #define private public when included)
    ch.impl_ = std::move(transport);

    auto ret = ch.Clean();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuUrmaChannelTest, Ut_Resume_When_Called_Expect_HCCL_SUCCESS)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);

    auto ret = ch.Resume();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuUrmaChannelTest, Ut_BuildBufferInfos_When_LocalUbHandle_Expect_BufferInfoFieldsFromRmaBuffer)
{
    auto rawBuffer = std::make_shared<Hccl::Buffer>(0x34560, 0x200, HCCL_MEM_TYPE_HOST, "ccu_user");
    auto localRmaBuffer = std::make_shared<Hccl::LocalUbRmaBuffer>(rawBuffer);
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localRmaBuffer.get())};

    std::vector<CcuTransport::CclBufferInfo> bufferInfos;
    ASSERT_EQ(BuildBufferInfos(memHandles, 1, bufferInfos), HCCL_SUCCESS);
    ASSERT_EQ(bufferInfos.size(), 1U);
    EXPECT_EQ(bufferInfos[0].addr, localRmaBuffer->GetAddr());
    EXPECT_EQ(bufferInfos[0].size, static_cast<uint32_t>(localRmaBuffer->GetSize()));
    EXPECT_EQ(bufferInfos[0].tokenId, localRmaBuffer->GetTokenId());
    EXPECT_EQ(bufferInfos[0].tokenValue, localRmaBuffer->GetTokenValue());
    EXPECT_EQ(bufferInfos[0].type, COMM_MEM_TYPE_HOST);
    EXPECT_EQ(std::string(bufferInfos[0].memInfo.data()), rawBuffer->GetMemInfo());
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_DfxDescribe_Expect_ReadyOrFailed)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(1);
    CcuUrmaChannel ch(ep, desc);

    ch.ccuEndpoint_ = &TestUrmaEndpoint();
    ch.local_.socket = reinterpret_cast<Hccl::Socket*>(1);
    ch.memHandles_.push_back(reinterpret_cast<HcommMemHandle>(0x1));

    // 1. 构造依赖
    CommAddr locAddr{}, rmtAddr{};
    CcuChannelInfo channelInfo{};
    std::vector<CcuJetty*> jettys{};

    auto conn = std::make_unique<CcuConnection>(locAddr, rmtAddr, channelInfo, jettys, Hccl::UB_QOS_DEFAULT);
    auto fakeSocket = reinterpret_cast<Hccl::Socket*>(1);
    CcuTransport::CclBufferInfo bufInfo(0x1000, 0x100, 1, 1);

    // 2. 创建 transport
    auto transport = std::make_unique<CcuTransport>(fakeSocket, std::move(conn), bufInfo);
    ch.impl_ = std::move(transport);
    ch.impl_->SetLocResStatus(CcuTransport::CcuResStatus::RES_OK);

    // 3. 从 ch.impl_ 操作，不要再用旧的 transport 指针！
    ch.impl_->transStatus_ = CcuTransport::TransStatus::SOCKET_TIMEOUT;
    auto ret = ch.GetStatus();
    EXPECT_NE(ret, ChannelStatus::INIT);

    ch.impl_->transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
    ret = ch.GetStatus();
    EXPECT_NE(ret, ChannelStatus::INIT);

    // 4. Mock 1：成功
    ch.impl_->transStatus_ = CcuTransport::TransStatus::READY;
    MOCKER_CPP(&CcuConnection::Describe, HcclResult(CcuConnection::*)(std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    ret = ch.GetStatus();
    EXPECT_EQ(ret, ChannelStatus::READY);
    GlobalMockObject::verify();
    GlobalMockObject::reset(); // <-- 必须重置 Mock！

    // 5. Mock 2：失败
    ch.impl_->transStatus_ = CcuTransport::TransStatus::READY;
    ch.isFirstPrintChannelInfo_ = true; // 需要重置为true才能触发Describe的调用
    MOCKER_CPP(&CcuConnection::Describe, HcclResult(CcuConnection::*)(std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_E_PARA));
    ret = ch.GetStatus();
    EXPECT_EQ(ret, ChannelStatus::FAILED);
    GlobalMockObject::verify();
    GlobalMockObject::reset();
}

// ==================== CcuUrmaChannel::GetStatus 资源不足终态映射 UT ====================

struct ChannelWithTransport {
    std::unique_ptr<CcuUrmaChannel> ch;
    CcuTransport* impl;
};

ChannelWithTransport MakeChannelWithTransport()
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(1);
    auto ch = std::make_unique<CcuUrmaChannel>(ep, desc);
    ch->ccuEndpoint_ = &TestUrmaEndpoint();
    ch->local_.socket = reinterpret_cast<Hccl::Socket*>(1);
    ch->memHandles_.push_back(reinterpret_cast<HcommMemHandle>(0x1));
    CommAddr locAddr{}, rmtAddr{};
    CcuChannelInfo channelInfo{};
    std::vector<CcuJetty*> jettys{};
    auto conn = std::make_unique<CcuConnection>(locAddr, rmtAddr, channelInfo, jettys, Hccl::UB_QOS_DEFAULT);
    auto fakeSocket = reinterpret_cast<Hccl::Socket*>(1);
    CcuTransport::CclBufferInfo bufInfo(0x1000, 0x100, 1, 1);
    auto transport = std::make_unique<CcuTransport>(fakeSocket, std::move(conn), bufInfo);
    // STC V2.0: locResStatus_ 改用公共 setter, 不直塞私有成员
    transport->SetLocResStatus(CcuTransport::CcuResStatus::RES_OK);
    CcuTransport* impl = transport.get();
    ch->impl_ = std::move(transport);
    ch->isFirstPrintChannelInfo_ = false;
    return {std::move(ch), impl};
}

// 构造可直接进入快照装配路径的 CCU 通道，避免触发真实建链
static std::unique_ptr<CcuUrmaChannel> MakeReadySnapshotChannel()
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(1);
    auto channel = std::make_unique<CcuUrmaChannel>(ep, desc);
    channel->ccuEndpoint_ = &TestUrmaEndpoint();
    channel->local_.socket = reinterpret_cast<Hccl::Socket*>(1);
    channel->memHandles_.push_back(reinterpret_cast<HcommMemHandle>(1));
    channel->isFirstPrintChannelInfo_ = false;

    auto conn = std::make_unique<CcuConnection>(
        CommAddr{}, CommAddr{}, CcuChannelInfo{}, std::vector<CcuJetty*>{}, Hccl::UB_QOS_DEFAULT);
    conn->rmtCcuBufAddr_ = 0x10000ULL;
    conn->rmtCcuBufSize_ = 0x1000ULL;
    conn->rmtCcuBufTokenId_ = 10U;
    conn->rmtCcuBufTokenValue_ = 20U;
    auto impl = std::make_unique<CcuTransport>(
        reinterpret_cast<Hccl::Socket*>(1), std::move(conn), CcuTransport::CclBufferInfo{});
    impl->dieId_ = 1U;
    impl->userDevId_ = 3;
    impl->SetLocResStatus(CcuTransport::CcuResStatus::RES_OK);
    impl->locRes_.ckes = {10U, 11U};
    impl->locRes_.xns = {20U, 21U};
    impl->rmtRes_.ckes = {30U, 31U};
    impl->rmtRes_.xns = {40U, 41U};
    impl->transStatus_ = CcuTransport::TransStatus::READY;
    channel->impl_ = std::move(impl);
    return channel;
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_LocUnavail_Expect_ResLocUnavailable)
{
    auto ctx = MakeChannelWithTransport();
    // 白盒保留: transStatus_/rmtResStatus_ 无公共 setter, 只能直塞私有成员
    ctx.impl->transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
    // STC V2.0: locResStatus_ 改用公共 setter
    ctx.impl->SetLocResStatus(CcuTransport::CcuResStatus::RES_UNAVAIL);
    ctx.impl->rmtResStatus_ = CcuTransport::CcuResStatus::RES_OK;
    EXPECT_EQ(ctx.ch->GetStatus(), ChannelStatus::RES_LOC_UNAVAIL);
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_RmtUnavail_Expect_ResRmtUnavailable)
{
    auto ctx = MakeChannelWithTransport();
    // 白盒保留: transStatus_/rmtResStatus_ 无公共 setter, 只能直塞私有成员
    ctx.impl->transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
    // STC V2.0: locResStatus_ 改用公共 setter
    ctx.impl->SetLocResStatus(CcuTransport::CcuResStatus::RES_OK);
    ctx.impl->rmtResStatus_ = CcuTransport::CcuResStatus::RES_UNAVAIL;
    EXPECT_EQ(ctx.ch->GetStatus(), ChannelStatus::RES_RMT_UNAVAIL);
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_BothUnavail_Expect_LocPriority)
{
    auto ctx = MakeChannelWithTransport();
    // 白盒保留: transStatus_/rmtResStatus_ 无公共 setter, 只能直塞私有成员
    ctx.impl->transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
    // STC V2.0: locResStatus_ 改用公共 setter
    ctx.impl->SetLocResStatus(CcuTransport::CcuResStatus::RES_UNAVAIL);
    ctx.impl->rmtResStatus_ = CcuTransport::CcuResStatus::RES_UNAVAIL;
    EXPECT_EQ(ctx.ch->GetStatus(), ChannelStatus::RES_LOC_UNAVAIL);
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_NoUnavail_Expect_Failed)
{
    auto ctx = MakeChannelWithTransport();
    // 白盒保留: transStatus_/rmtResStatus_ 无公共 setter, 只能直塞私有成员
    ctx.impl->transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
    // STC V2.0: locResStatus_ 改用公共 setter
    ctx.impl->SetLocResStatus(CcuTransport::CcuResStatus::RES_OK);
    ctx.impl->rmtResStatus_ = CcuTransport::CcuResStatus::RES_OK;
    EXPECT_EQ(ctx.ch->GetStatus(), ChannelStatus::FAILED);
}

// TC-CH-016: 本端硬失败(RES_FAILED)终态映射为 FAILED (非资源不足, 不触发回退)
TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_LocFailed_Expect_Failed)
{
    auto ctx = MakeChannelWithTransport();
    ctx.impl->transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
    ctx.impl->SetLocResStatus(CcuTransport::CcuResStatus::RES_FAILED);
    ctx.impl->rmtResStatus_ = CcuTransport::CcuResStatus::RES_OK;
    EXPECT_EQ(ctx.ch->GetStatus(), ChannelStatus::FAILED);
}

// TC-CH-017: 对端硬失败(RES_FAILED)终态映射为 FAILED
TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_RmtFailed_Expect_Failed)
{
    auto ctx = MakeChannelWithTransport();
    ctx.impl->transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
    ctx.impl->SetLocResStatus(CcuTransport::CcuResStatus::RES_OK);
    ctx.impl->rmtResStatus_ = CcuTransport::CcuResStatus::RES_FAILED;
    EXPECT_EQ(ctx.ch->GetStatus(), ChannelStatus::FAILED);
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_Ready_Expect_Ready)
{
    auto ctx = MakeChannelWithTransport();
    ctx.impl->transStatus_ = CcuTransport::TransStatus::READY;
    EXPECT_EQ(ctx.ch->GetStatus(), ChannelStatus::READY);
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_SocketTimeout_Expect_SocketTimeout)
{
    auto ctx = MakeChannelWithTransport();
    ctx.impl->transStatus_ = CcuTransport::TransStatus::SOCKET_TIMEOUT;
    EXPECT_EQ(ctx.ch->GetStatus(), ChannelStatus::SOCKET_TIMEOUT);
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_ConstructFails_Expect_MsgOnlyTransport)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);

    ch.ccuEndpoint_ = &TestUrmaEndpoint();
    ch.local_.socket = reinterpret_cast<Hccl::Socket*>(0x1);
    ch.memHandles_ = {reinterpret_cast<HcommMemHandle>(0x2)};
    ch.channelStatus_ = ChannelStatus::INIT;

    // 第一次 GetStatus: 触发 TryPrepareAndConstruct, 硬失败 -> msg-only transport(RES_FAILED)
    EXPECT_EQ(ch.GetStatus(), ChannelStatus::INIT);
    ASSERT_NE(ch.impl_, nullptr);
    EXPECT_EQ(ch.impl_->GetLocResStatus(), CcuTransport::CcuResStatus::RES_FAILED);
    EXPECT_EQ(ch.impl_->transStatus_, CcuTransport::TransStatus::INIT);

    // 第二次 GetStatus: locResStatus_ != RES_UNKNOWN, 不会重复构造 transport
    CcuTransport* implPtr = ch.impl_.get();
    EXPECT_EQ(ch.GetStatus(), ChannelStatus::INIT);
    EXPECT_EQ(ch.impl_.get(), implPtr);
}

// 未初始化路径: endpoint/socket/memHandles 缺失时直接失败, 不再进入懒建链
TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_Uninitialized_Expect_Failed)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);

    EXPECT_EQ(ch.GetStatus(), ChannelStatus::FAILED);
    EXPECT_EQ(ch.impl_, nullptr);
}

TEST_F(CcuUrmaChannelTest, Ut_BuildHcommCcuChannelEntity_When_NotReady_Expect_HCCL_E_UNAVAIL)
{
    auto channel = MakeReadySnapshotChannel();
    channel->impl_->transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;

    HcommCcuChannelEntity snapshot{};
    std::vector<RegedBufferEntity> remoteBuffers;
    EXPECT_EQ(channel->BuildHcommCcuChannelEntity(snapshot, remoteBuffers), HcclResult::HCCL_E_UNAVAIL);
}

// impl_ 未建立（endpoint/socket/memHandles 缺失）时构建实体返回 HCCL_E_UNAVAIL，日志不携带 UINT32_MAX 哨兵 id
TEST_F(CcuUrmaChannelTest, Ut_BuildHcommCcuChannelEntity_When_ImplNotBuilt_Expect_HCCL_E_UNAVAIL)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);

    HcommCcuChannelEntity snapshot{};
    std::vector<RegedBufferEntity> remoteBuffers;
    EXPECT_EQ(ch.BuildHcommCcuChannelEntity(snapshot, remoteBuffers), HcclResult::HCCL_E_UNAVAIL);
    EXPECT_EQ(ch.impl_, nullptr);
}

TEST_F(CcuUrmaChannelTest, Ut_BuildHcommCcuChannelEntity_When_CcuVersionQueryFails_Expect_QueryError)
{
    auto channel = MakeReadySnapshotChannel();

    HcommCcuChannelEntity snapshot{};
    std::vector<RegedBufferEntity> remoteBuffers;
    EXPECT_EQ(channel->BuildHcommCcuChannelEntity(snapshot, remoteBuffers), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(CcuUrmaChannelTest, Ut_BuildHcommCcuChannelEntity_When_ResourceNumExceedCapacity_Expect_HCCL_E_INTERNAL)
{
    auto channel = MakeReadySnapshotChannel();
    channel->impl_->locRes_.xns.assign(17U, 0U);

    HcommCcuChannelEntity snapshot{};
    std::vector<RegedBufferEntity> remoteBuffers;
    EXPECT_EQ(channel->BuildHcommCcuChannelEntity(snapshot, remoteBuffers), HcclResult::HCCL_E_INTERNAL);
}

static void SetupSnapshotCcuVersion(CcuVersion version)
{
    MOCKER(CcuDevMgrImp::GetCcuVersion).stubs().with(mockcpp::any(), outBound(version)).will(returnValue(HCCL_SUCCESS));
}

TEST_F(CcuUrmaChannelTest, Ut_BuildHcommCcuChannelEntity_When_Normal_Expect_EntityFieldsFilled)
{
    auto channel = MakeReadySnapshotChannel();
    CcuChannelInfo channelInfo{};
    channelInfo.channelId = 101U;
    channelInfo.dieId = 1U;
    channel->impl_->ccuConnection_->channelInfo_ = channelInfo;
    channel->impl_->rmtBufferVec_.push_back(
        std::make_unique<Hccl::RemoteUbRmaBuffer>(0x20000U, 0x2000U, 11U, 12U, HCCL_MEM_TYPE_DEVICE, "ccu_user"));
    SetupSnapshotCcuVersion(CcuVersion::CCU_V2);

    HcommCcuChannelEntity snapshot{};
    std::vector<RegedBufferEntity> remoteBuffers;
    EXPECT_EQ(channel->BuildHcommCcuChannelEntity(snapshot, remoteBuffers), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(snapshot.header.version, HCOMM_CCU_CHANNEL_ABI_VERSION);
    EXPECT_EQ(snapshot.header.magicWord, HCOMM_CCU_CHANNEL_MAGIC_WORD);
    EXPECT_EQ(snapshot.header.size, sizeof(HcommCcuChannelEntity));
    EXPECT_EQ(snapshot.deviceLogicId, 3);
    EXPECT_EQ(snapshot.ccuVersion, 1U);
    EXPECT_EQ(snapshot.dieId, 1U);
    EXPECT_EQ(snapshot.channelId, 101U);
    EXPECT_EQ(snapshot.localVarNum, 2U);
    EXPECT_EQ(snapshot.remoteVarNum, 2U);
    EXPECT_EQ(snapshot.localEventNum, 2U);
    EXPECT_EQ(snapshot.remoteEventNum, 2U);
    EXPECT_EQ(snapshot.localVarIds[0], 20U);
    EXPECT_EQ(snapshot.localVarIds[1], 21U);
    EXPECT_EQ(snapshot.remoteVarIds[0], 40U);
    EXPECT_EQ(snapshot.remoteVarIds[1], 41U);
    EXPECT_EQ(snapshot.localEventIds[0], 10U);
    EXPECT_EQ(snapshot.localEventIds[1], 11U);
    EXPECT_EQ(snapshot.remoteEventIds[0], 30U);
    EXPECT_EQ(snapshot.remoteEventIds[1], 31U);
    EXPECT_EQ(snapshot.rmtCcuResBuffer.type, REGED_BUFFER_RMA);
    EXPECT_EQ(snapshot.rmtCcuResBuffer.bufferInfo.rma.addr, 0x10000U);
    EXPECT_EQ(snapshot.rmtCcuResBuffer.bufferInfo.rma.size, 0x1000U);
    EXPECT_EQ(snapshot.rmtCcuResBuffer.bufferInfo.rma.protectionInfo.type, PROTECTION_TYPE_UB);
    EXPECT_EQ(snapshot.rmtCcuResBuffer.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId, 10U);
    EXPECT_EQ(snapshot.rmtCcuResBuffer.bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue, 20U);

    ASSERT_EQ(snapshot.remoteBufferNum, 1U);
    ASSERT_NE(snapshot.remoteBuffers, nullptr);
    EXPECT_EQ(snapshot.remoteBuffers[0].type, REGED_BUFFER_RMA);
    EXPECT_EQ(snapshot.remoteBuffers[0].bufferInfo.rma.addr, 0x20000U);
    EXPECT_EQ(snapshot.remoteBuffers[0].bufferInfo.rma.size, 0x2000U);

    GlobalMockObject::verify();
    GlobalMockObject::reset();
}

// ===================== BuildSocket 结构异构保留的钉住用例 =====================
// CCU 不走基类 traits_.socketBuildMode 分派（CCU_URMA 折算为 NONE）：socket 由外部注入 + ccuEndpoint_ 解析

TEST_F(CcuUrmaChannelTest, Ut_GetSocketBuildMode_When_CcuUrma_Expect_None)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);
    EXPECT_EQ(ch.GetSocketBuildMode(), SocketBuildMode::NONE);
}

TEST_F(CcuUrmaChannelTest, Ut_BuildSocket_WhenSocketNotInjected_Expect_HCCL_E_PTR)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);
    EXPECT_EQ(ch.BuildSocket(), HCCL_E_PTR);
}

static HcommResult StubCcuHcommEndpointGet(EndpointHandle, void** endpoint)
{
    *endpoint = static_cast<void*>(&TestUrmaEndpoint());
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

TEST_F(CcuUrmaChannelTest, Ut_BuildSocket_WhenSocketInjected_Expect_SuccessAndCcuEndpointResolved)
{
    Hccl::Socket fakeSock(
        nullptr, Hccl::IpAddress(), 0, Hccl::IpAddress(), "ut", Hccl::SocketRole::SERVER,
        Hccl::NicType::DEVICE_NIC_TYPE);
    HcommChannelDesc desc{};
    desc.socket = static_cast<void*>(&fakeSock);
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);

    MOCKER(HcommEndpointGet).stubs().will(invoke(StubCcuHcommEndpointGet));

    EXPECT_EQ(ch.BuildSocket(), HCCL_SUCCESS);
    EXPECT_EQ(ch.local_.socket, &fakeSock);
    EXPECT_EQ(ch.ccuEndpoint_, &TestUrmaEndpoint());
    ch.local_.socket = nullptr; // 防止析构归还打到真实 SocketMgr
}
