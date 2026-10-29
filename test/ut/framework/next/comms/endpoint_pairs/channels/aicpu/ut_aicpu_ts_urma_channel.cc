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
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "aicpu/aicpu_ts_urma_channel.h"
#include "endpoint.h"
#include "buffer/local_ub_rma_buffer.h"
#include "../sockets/socket_mgr.h"
#undef private
#undef protected
#include "socket_handle_manager.h"
using namespace hcomm;

namespace {
class StubEndpointForUrmaChannel : public Endpoint {
public:
    StubEndpointForUrmaChannel() : Endpoint(MakeDesc()) { ctxHandle_ = reinterpret_cast<void*>(0x1); }

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
        Hccl::IpAddress localIp("127.0.0.1");
        desc.protocol = COMM_PROTOCOL_UB_CTP;
        desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        desc.commAddr.addr = localIp.GetBinaryAddress().addr;
        desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        desc.loc.device.devPhyId = 3;
        return desc;
    }
};

std::shared_ptr<Hccl::LocalUbRmaBuffer> MakeUrmaLocalBuffer(uintptr_t addr, u64 size, const char* tag)
{
    auto buffer = std::make_shared<Hccl::Buffer>(addr, size, HCCL_MEM_TYPE_DEVICE, tag);
    return std::make_shared<Hccl::LocalUbRmaBuffer>(buffer);
}

HcommResult StubUrmaGetAllMemHandlesOne(EndpointHandle, void** memHandles, uint32_t* memHandleNum)
{
    static std::shared_ptr<Hccl::Buffer> buffer
        = std::make_shared<Hccl::Buffer>(0x620000U, 0x1000U, HCCL_MEM_TYPE_DEVICE, "urma_all");
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffer = std::make_shared<Hccl::LocalUbRmaBuffer>(buffer);
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffers[1] = {localBuffer};
    *memHandles = localBuffers;
    *memHandleNum = 1;
    return HCCL_SUCCESS;
}

HcommResult StubUrmaGetAllMemHandlesWithNullSlot(EndpointHandle, void** memHandles, uint32_t* memHandleNum)
{
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffers[1] = {nullptr};
    *memHandles = localBuffers;
    *memHandleNum = 1;
    return HCCL_SUCCESS;
}

void AttachFakeUbMemTransport(AicpuTsUrmaChannel& ch)
{
    Hccl::BaseMemTransport::Attribution attr{};
    Hccl::IpAddress locAddrIp{"1.1.1.1"};
    Hccl::IpAddress rmtAddrIp{"2.2.2.2"};
    Hccl::Socket sock{nullptr, locAddrIp, 0, rmtAddrIp, "ut", Hccl::SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE};
    Hccl::BaseMemTransport::CommonLocRes commonLocRes{};
    auto conn = std::make_unique<Hccl::DevUbConnection>(nullptr, locAddrIp, rmtAddrIp, Hccl::OpMode::OPBASE);
    commonLocRes.connVec.push_back(conn.get());
    RdmaHandle rdmaHandle = nullptr;
    Hccl::BaseMemTransport::LocCntNotifyRes locCntNotifyRes{};
    Hccl::LinkData linkData(
        Hccl::PortDeploymentType::HOST_NET, Hccl::LinkProtocol::UB_TP, 0, 0, Hccl::IpAddress(), Hccl::IpAddress());

    ch.connections_.push_back(std::move(conn));
    ch.memTransport_ = std::make_unique<Hccl::UbMemTransport>(
        commonLocRes, attr, linkData, sock, rdmaHandle, locCntNotifyRes, false);
}

std::vector<Hccl::LocalRmaBuffer*> g_capturedUpdateBuffers;

HcclResult StubUrmaUpdateMemInfo(Hccl::UbMemTransport*, std::vector<Hccl::LocalRmaBuffer*>& bufferVec)
{
    g_capturedUpdateBuffers = bufferVec;
    return HCCL_SUCCESS;
}
} // namespace

class AicpuTsUrmaChannelTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(AicpuTsUrmaChannelTest, Ut_Clean_WithoutInit_Returns_SUCCESS)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);

    auto ret = ch.Clean();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AicpuTsUrmaChannelTest, Ut_Resume_MockedBuilds_Returns_SUCCESS)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);

    // Mock 现已虚化的 Init hook（MOCKER_CPP_VIRTUAL 经 ch 的 vtable 取真实函数地址）
    MOCKER_CPP_VIRTUAL(ch, &Channel::BuildSocket, HcclResult(Channel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP_VIRTUAL(ch, &AicpuTsUrmaChannel::BuildConnection, HcclResult(AicpuTsUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&AicpuTsUrmaChannel::ResetLocalNotifies, HcclResult(AicpuTsUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP_VIRTUAL(ch, &AicpuTsUrmaChannel::BuildBuffer, HcclResult(AicpuTsUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    auto ret = ch.Resume();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AicpuTsUrmaChannelTest, Ut_ResetLocalNotifies_When_Empty_Expect_SUCCESS)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);

    EXPECT_EQ(ch.ResetLocalNotifies(), HCCL_SUCCESS);
}

TEST_F(AicpuTsUrmaChannelTest, Ut_Resume_When_ResetLocalNotifiesFailed_Expect_ReturnFailed)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);

    MOCKER_CPP_VIRTUAL(ch, &Channel::BuildSocket, HcclResult(Channel::*)()).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(ch, &AicpuTsUrmaChannel::BuildConnection, HcclResult(AicpuTsUrmaChannel::*)())
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsUrmaChannel::ResetLocalNotifies, HcclResult(AicpuTsUrmaChannel::*)())
        .stubs()
        .will(returnValue(HCCL_E_INTERNAL));
    MOCKER_CPP_VIRTUAL(ch, &AicpuTsUrmaChannel::BuildBuffer, HcclResult(AicpuTsUrmaChannel::*)()).expects(never());

    EXPECT_EQ(ch.Resume(), HCCL_E_INTERNAL);
}

TEST_F(AicpuTsUrmaChannelTest, UT_ParseInputParam_When_ExchangeAllMemsFalse_Expect_FillCommonRes)
{
    StubEndpointForUrmaChannel endpoint;
    auto localBuffer = MakeUrmaLocalBuffer(0x610000U, 0x1000U, "urma_desc");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};
    HcommChannelDesc desc{};
    desc.exchangeAllMems = false;
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AicpuTsUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.local_.commonRes.bufferVec.size(), 1U);
    EXPECT_EQ(ch.local_.commonRes.bufferVec[0], localBuffer.get());
}

TEST_F(AicpuTsUrmaChannelTest, UT_ParseInputParam_When_ExchangeAllMemsTrue_Expect_FillCommonRes)
{
    StubEndpointForUrmaChannel endpoint;
    HcommChannelDesc desc{};
    desc.exchangeAllMems = true;
    AicpuTsUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubUrmaGetAllMemHandlesOne));
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.local_.commonRes.bufferVec.size(), 1U);
    EXPECT_EQ(ch.local_.commonRes.bufferVec[0]->GetAddr(), 0x620000U);
}

TEST_F(AicpuTsUrmaChannelTest, UT_ParseInputParam_When_ExchangeAllMemsTrueAndNullSlot_Expect_ReturnHCCL_E_PTR)
{
    StubEndpointForUrmaChannel endpoint;
    HcommChannelDesc desc{};
    desc.exchangeAllMems = true;
    AicpuTsUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubUrmaGetAllMemHandlesWithNullSlot));

    EXPECT_EQ(ch.ParseInputParam(), HCCL_E_PTR);
}

TEST_F(AicpuTsUrmaChannelTest, UT_UpdateMemInfo_When_ValidMemHandles_Expect_UpdateTransportAndCommonRes)
{
    g_capturedUpdateBuffers.clear();
    HcommChannelDesc desc{};
    AicpuTsUrmaChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    AttachFakeUbMemTransport(ch);
    auto initBuffer = MakeUrmaLocalBuffer(0x620000U, 0x1000U, "urma_init");
    ch.local_.commonRes.bufferVec.push_back(initBuffer.get());
    auto localBuffer = MakeUrmaLocalBuffer(0x630000U, 0x1000U, "urma_update");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};

    MOCKER_CPP(
        &Hccl::UbMemTransport::UpdateMemInfo, HcclResult(Hccl::UbMemTransport::*)(std::vector<Hccl::LocalRmaBuffer*>&))
        .stubs()
        .with(mockcpp::any())
        .will(invoke(StubUrmaUpdateMemInfo));

    EXPECT_EQ(ch.UpdateMemInfo(memHandles, 1), HCCL_SUCCESS);
    ASSERT_EQ(g_capturedUpdateBuffers.size(), 1U);
    EXPECT_EQ(g_capturedUpdateBuffers[0], localBuffer.get());
    ASSERT_EQ(ch.local_.commonRes.bufferVec.size(), 2U);
    EXPECT_EQ(ch.local_.commonRes.bufferVec[0], initBuffer.get());
    EXPECT_EQ(ch.local_.commonRes.bufferVec[1], localBuffer.get());
}

TEST_F(AicpuTsUrmaChannelTest, UT_UpdateMemInfo_When_NullHandle_Expect_ReturnHCCL_E_PTR)
{
    HcommChannelDesc desc{};
    AicpuTsUrmaChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    AttachFakeUbMemTransport(ch);
    HcommMemHandle memHandles[1] = {nullptr};

    EXPECT_EQ(ch.UpdateMemInfo(memHandles, 1), HCCL_E_PTR);
}

TEST_F(AicpuTsUrmaChannelTest, Ut_GetStatus_DfxInfo_TEST)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);

    // 1. 构造依赖
    Hccl::BaseMemTransport::Attribution attr{};
    Hccl::IpAddress locAddrIp{"1.1.1.1"};
    Hccl::IpAddress rmtAddrIp{"2.2.2.2"};
    Hccl::Socket sock{nullptr, locAddrIp, 0, rmtAddrIp, "ut", Hccl::SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE};
    Hccl::BaseMemTransport::CommonLocRes commonLocRes{};
    std::vector<Hccl::RmaConnection*> conns;
    auto conn = std::make_unique<Hccl::DevUbConnection>(nullptr, locAddrIp, rmtAddrIp, Hccl::OpMode::OPBASE);
    conns.push_back(conn.get()); // 仅占位，测试不涉及具体连接对象
    commonLocRes.connVec = conns;
    RdmaHandle rdmaHandle = nullptr;
    Hccl::BaseMemTransport::LocCntNotifyRes locCntNotifyRes{};

    Hccl::PortDeploymentType portDeploymentType = Hccl::PortDeploymentType::HOST_NET;
    Hccl::LinkProtocol linkProtocol = Hccl::LinkProtocol::UB_TP;
    Hccl::IpAddress locAddr;
    Hccl::IpAddress rmtAddr;
    uint32_t locDevPhyId = 0;
    uint32_t rmtDevPhyId = 0;
    Hccl::LinkData ld = Hccl::LinkData(portDeploymentType, linkProtocol, locDevPhyId, rmtDevPhyId, locAddr, rmtAddr);

    auto memTransport
        = std::make_unique<Hccl::UbMemTransport>(commonLocRes, attr, ld, sock, rdmaHandle, locCntNotifyRes, false);
    memTransport->connNum = 1;
    memTransport->baseStatus = Hccl::TransportStatus::READY;
    ch.memTransport_ = std::move(memTransport);

    MOCKER_CPP(&Hccl::UbMemTransport::Describe, HcclResult(Hccl::UbMemTransport::*)(std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS))
        .then(returnValue(HCCL_E_PARA)); // 连续两次调用，第一次成功，第二次失败

    auto ret = ch.GetStatus();
    EXPECT_EQ(ret, ChannelStatus::READY);

    // 再次调用GetStatus，验证isFirstPrintChannelInfo_逻辑分支覆盖
    ch.isFirstPrintChannelInfo_ = true; // 重置为true，模拟第一次打印
    ret = ch.GetStatus();
    EXPECT_EQ(ret, ChannelStatus::FAILED);

    GlobalMockObject::verify();
    GlobalMockObject::reset();
}

TEST_F(AicpuTsUrmaChannelTest, Ut_StartListen_When_RoleNotServer_Expect_SUCCESS)
{
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_CLIENT;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);

    auto ret = ch.StartListen();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ch.local_.serverSocket, nullptr); // CLIENT 角色 → BORROW 模式，不自建监听（ctx 路径亦被角色门控跳过）
}

// ==================== GetMemUpdatePack：覆写把取包原样转发给 transport ====================

TEST_F(AicpuTsUrmaChannelTest, Ut_GetMemUpdatePack_When_TransportAnswers_Expect_SameAsTransport)
{
    HcommChannelDesc desc{};
    AicpuTsUrmaChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    AttachFakeUbMemTransport(ch);
    // LLT 下 UbMemTransport 是桩（llt_next_orion_stub.cc），产不出真实包，这里打桩给出可辨认的包
    const std::vector<char> transportPack{0x02, 0x00, 0x00, 0x00, 0x7A, 0x7B};
    MOCKER_CPP(&Hccl::UbMemTransport::GetMemUpdatePack, std::vector<char>(Hccl::UbMemTransport::*)() const)
        .stubs()
        .will(returnValue(transportPack));

    // 必须走覆写转发给 transport，而不是基类那个恒空的默认实现
    EXPECT_EQ(ch.GetMemUpdatePack(), transportPack);
}

// ===================== BuildConnection 设备属性只取一次 UT =====================

static int g_urmaDevBaseAttrFetchCount = 0;
static HcclResult StubCountUrmaDevBaseAttr(void* ctxHandle, DevBaseAttr* attr)
{
    (void)ctxHandle;
    (void)attr; // 不覆写预置 devBaseAttr_，仅计数
    g_urmaDevBaseAttrFetchCount++;
    return HCCL_SUCCESS;
}

// 首次 BuildConnection 拉取设备属性并置位守卫；重入（Resume 重建链路）不再重复 BuildAttr/HccpRaGetDevBaseAttr
TEST_F(AicpuTsUrmaChannelTest, Ut_BuildConnection_When_Reentry_Expect_DevBaseAttrFetchedOnce)
{
    HcommChannelDesc desc{};
    desc.ubAttr.sqDepth = 512U;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);
    ch.local_.ep.protocol = COMM_PROTOCOL_UBC_TP;
    ch.local_.ep.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ch.local_.ep.commAddr.addr = Hccl::IpAddress("1.0.0.1").GetBinaryAddress().addr;
    ch.remote_.ep.protocol = COMM_PROTOCOL_UBC_TP;
    ch.remote_.ep.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ch.remote_.ep.commAddr.addr = Hccl::IpAddress("2.0.0.1").GetBinaryAddress().addr;
    ch.local_.rdmaHandle = reinterpret_cast<RdmaHandle>(0x1);
    ch.devBaseAttr_.sqMaxDepth = 1024U;
    ch.devBaseAttr_.maxReadSize = 64U;
    ch.devBaseAttr_.maxWriteSize = 64U;

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    g_urmaDevBaseAttrFetchCount = 0;
    MOCKER(HccpRaGetDevBaseAttr).stubs().will(invoke(StubCountUrmaDevBaseAttr));

    EXPECT_EQ(ch.BuildConnection(), HCCL_SUCCESS);
    EXPECT_EQ(ch.connections_.size(), 1U);
    EXPECT_EQ(g_urmaDevBaseAttrFetchCount, 1);
    EXPECT_TRUE(ch.devBaseAttrReady_);

    // 模拟 Resume 等重入：再次建链不再重复拉取设备属性
    EXPECT_EQ(ch.BuildConnection(), HCCL_SUCCESS);
    EXPECT_EQ(ch.connections_.size(), 1U);
    EXPECT_EQ(g_urmaDevBaseAttrFetchCount, 1);
}

// ===================== BuildSocket 收敛基类后的钉住用例 =====================

TEST_F(AicpuTsUrmaChannelTest, Ut_ParseInputParam_WhenRoleReserved_Expect_ListenAndBorrowMode)
{
    StubEndpointForUrmaChannel endpoint;
    auto localBuffer = MakeUrmaLocalBuffer(0x610000U, 0x1000U, "urma_mode_rsv");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_RESERVED;
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AicpuTsUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_EQ(ch.traits_.socketBuildMode, SocketBuildMode::LISTEN_AND_BORROW);
}

TEST_F(AicpuTsUrmaChannelTest, Ut_ParseInputParam_WhenRoleExplicit_Expect_BorrowMode)
{
    StubEndpointForUrmaChannel endpoint;
    auto localBuffer = MakeUrmaLocalBuffer(0x611000U, 0x1000U, "urma_mode_srv");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AicpuTsUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_EQ(ch.traits_.socketBuildMode, SocketBuildMode::BORROW);
}

TEST_F(AicpuTsUrmaChannelTest, Ut_StartListenAndBuildSocket_WhenRoleReserved_Expect_ListenSocketAndNoRankIdConfig)
{
    StubEndpointForUrmaChannel endpoint; // commAddr 已配 V4 127.0.0.1，真实 CommAddrToIpAddress 可解析
    auto localBuffer = MakeUrmaLocalBuffer(0x612000U, 0x1000U, "urma_borrow_rsv");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_RESERVED;
    desc.port = 61001; // 验证自建监听端口不受 desc.port 影响（RESERVED 自建监听固定 defaultListenPort=60001）
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AicpuTsUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));
    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.local_.socket, nullptr);

    // LISTEN 分支外部依赖打桩：handle 创建 / 借出（Listen() 虚函数由链接级 llt 桩接管；
    // MOCKER_CPP 打虚函数拿到虚表偏移伪指针会段错误）
    MOCKER_CPP(&Hccl::SocketHandleManager::Create).stubs().will(returnValue(static_cast<Hccl::SocketHandle>(nullptr)));
    Hccl::Socket borrowed(
        nullptr, Hccl::IpAddress(), 0, Hccl::IpAddress(), "ut", Hccl::SocketRole::CLIENT,
        Hccl::NicType::DEVICE_NIC_TYPE);
    MOCKER_CPP(&SocketMgr::GetSocket).stubs().with(mockcpp::any(), outBound(&borrowed)).will(returnValue(HCCL_SUCCESS));

    // 自建监听已收口进 StartListen（与端点侧监听统一入口）；RESERVED 非 server 角色，ctx 路径被门控跳过
    EXPECT_EQ(ch.StartListen(), HCCL_SUCCESS);
    ASSERT_NE(ch.local_.serverSocket, nullptr);                 // 本端自建监听 socket 建立
    EXPECT_EQ(ch.local_.serverSocket->GetListenPort(), 60001U); // 自建监听端口=defaultListenPort（base 行为）

    EXPECT_EQ(ch.BuildSocket(), HCCL_SUCCESS); // 监听已就绪，BuildSocket 只做配置保存与数据 socket 借出
    ASSERT_NE(ch.local_.socketConfig, nullptr);
    EXPECT_TRUE(ch.local_.socketConfig->noRankId);            // noRankId 三参形态：rank 不进匹配键
    EXPECT_EQ(ch.local_.socketConfig->listeningPort, 60001U); // 三参形态端口不进 config，保持成员默认
    EXPECT_TRUE(ch.local_.isRecvFirst);                       // client 借出 → 先收后发
    ch.local_.socket = nullptr;                               // 防止析构归还打到真实 SocketMgr
}

// ===================== BuildNotify 收敛基类后的钉住用例 =====================
// URMA 折算 BUILD_LOCAL：统一流程 + UbLocalNotify 工厂由基类 BuildNotify 承担
// （流程/双注册钉住见 uboe UT 的 UT_BuildNotify_Expect_BuildLocalModeAndUbLocalNotifyRegistered）
TEST_F(AicpuTsUrmaChannelTest, Ut_GetNotifyBuildMode_Expect_BUILD_LOCAL)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);
    EXPECT_EQ(ch.GetNotifyBuildMode(), NotifyBuildMode::BUILD_LOCAL);
}
