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
#include <cstdint>
#include <cstdlib>
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "aiv_urma_channel.h"
#include "../sockets/socket_mgr.h"
#include "base_comm_node_accessors.h"
#include "aiv_urma_transport.h"
#undef protected
#undef private
#include "hcomm_c_adpt.h"
#include "adapter_rts_common.h" // hrtGetDevice/hrtGetDevicePhyIdByIndex
#include "endpoint.h"
#include "exchange_ub_buffer_dto.h"
#include "runtime_api_exception.h"

using namespace hcomm;
using namespace Hccl;

namespace {

uint32_t g_listenPort = 0;
constexpr u32 STUB_TOKEN_VALUE = 0x12345678U;
constexpr u32 STUB_TOKEN_ID = 0x87654321U;

HcommResult StubEndpointStartListen(EndpointHandle, uint32_t port, const HcommEndpointListenConfig*)
{
    g_listenPort = port;
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

LinkData MakeDefaultLinkData()
{
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    return LinkData(portType, 0, 1, 0, 0);
}

class StubEndpointForAivUrmaChannel : public Endpoint {
public:
    explicit StubEndpointForAivUrmaChannel(const EndpointDesc& desc, void* rdmaHandle) : Endpoint(desc)
    {
        ctxHandle_ = rdmaHandle;
    }

    HcclResult Init() override { return HCCL_SUCCESS; }
    // 本端/远端内存操作走 GetLocalRegMemMgr()/GetRemoteRegMemMgr()，stub 均不实现
    LocalRegedMemMgr* GetLocalRegMemMgr() override { return nullptr; }
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override { return ctxHandle_ != nullptr; }

private:
    void* ctxHandle_{nullptr};
};

class StubLocalUbRmaBufferWithExchangeDto : public LocalUbRmaBuffer {
public:
    explicit StubLocalUbRmaBufferWithExchangeDto(const std::shared_ptr<Buffer>& buffer) : LocalUbRmaBuffer(buffer) {}

    std::unique_ptr<Serializable> GetExchangeDto() override
    {
        return std::make_unique<ExchangeUbBufferDto>(
            GetAddr(), GetSize(), STUB_TOKEN_VALUE, STUB_TOKEN_ID, 0, UINT32_MAX);
    }
};

HcommResult StubHcommMemGetAllMemHandlesFail(EndpointHandle, void**, uint32_t*) { return HCCL_E_INTERNAL; }

HcommResult StubHcommMemGetAllMemHandlesEmpty(EndpointHandle, void** memHandles, uint32_t* memHandleNum)
{
    *memHandles = nullptr;
    *memHandleNum = 0;
    return HCCL_SUCCESS;
}

HcommResult StubHcommMemGetAllMemHandlesOne(EndpointHandle, void** memHandles, uint32_t* memHandleNum)
{
    static std::shared_ptr<Hccl::Buffer> buffer = std::make_shared<Hccl::Buffer>(0x12340000U, 0x2000U, "aiv_urma_ut");
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffer = std::make_shared<Hccl::LocalUbRmaBuffer>(buffer);
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffers[1] = {localBuffer};
    *memHandles = localBuffers;
    *memHandleNum = 1;
    return HCCL_SUCCESS;
}

static hccl::DeviceMem StubDeviceMemAlloc(u64 size, bool /*level2Address*/)
{
    void* ptr = std::malloc(static_cast<size_t>(size));
    // UT uses host memory as fake device memory. Keep owner=false because DeviceMem dtor calls hrtFree for owner
    // memory.
    return hccl::DeviceMem(ptr, size, false);
}

aclError StubAclrtMalloc(void** devPtr, size_t size, aclrtMemMallocPolicy /*policy*/, aclrtMallocConfig*)
{
    if (devPtr == nullptr) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    *devPtr = std::malloc(size);
    return (*devPtr == nullptr) ? ACL_ERROR_RT_MEMORY_ALLOCATION : ACL_SUCCESS;
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
    return ACL_SUCCESS;
}

HcclResult StubRecvDataProcessNeedFinish(AivUrmaTransport*, bool& needFinish)
{
    needFinish = true;
    return HCCL_SUCCESS;
}

HcclResult StubRecvDataProcessNoFinish(AivUrmaTransport*, bool& needFinish)
{
    needFinish = false;
    return HCCL_SUCCESS;
}

HcclResult StubIsResReadyTrue(AivUrmaTransport*, bool& isReady)
{
    isReady = true;
    return HCCL_SUCCESS;
}

HcclResult StubIsResReadyFalse(AivUrmaTransport*, bool& isReady)
{
    isReady = false;
    return HCCL_SUCCESS;
}

bool IsPtrInRange(const void* ptr, const void* base, size_t size)
{
    uintptr_t ptrValue = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t baseValue = reinterpret_cast<uintptr_t>(base);
    return ptrValue >= baseValue && ptrValue < baseValue + size;
}

void BuildReadyTransport(AivUrmaChannel& ch, DevUbCtpConnection& conn)
{
    BaseMemTransport::CommonLocRes commonRes{};
    conn.status = RmaConnStatus::READY;
    conn.jettyId = 3;
    conn.dbAddr = 0x1000;
    conn.sqBuffVa = 0x2000;
    conn.sqDepth = 4;
    conn.tpn = 5;
    conn.cqInfo_.id = 6;
    conn.cqInfo_.va = 0x3000;
    conn.cqInfo_.cqeSize = 64;
    conn.cqInfo_.cqDepth = 8;
    conn.cqInfo_.swdbAddr = 0x4000;
    commonRes.connVec.emplace_back(&conn);

    BaseMemTransport::Attribution attr{};
    LinkData linkData = MakeDefaultLinkData();
    static Socket socket(
        nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    ch.transport_
        = std::make_unique<AivUrmaTransport>(commonRes, attr, linkData, socket, reinterpret_cast<RdmaHandle>(0x1));
    ch.transport_->transportStatus_ = TransportStatus::READY;
}

static void StubHrtMemcpyReal(void* dst, uint64_t dstMax, const void* src, uint64_t count, Hccl::tagRtMemcpyKind)
{
    if (dst != nullptr && src != nullptr && dstMax >= count) {
        (void)std::memcpy(dst, src, static_cast<size_t>(count));
    }
}
} // namespace

class AivUrmaChannelTest : public testing::Test {
protected:
    static void TearDownTestCase() { GlobalMockObject::verify(); }

    void TearDown() override { GlobalMockObject::verify(); }

    HcommChannelDesc MakeDefaultDesc()
    {
        HcommChannelDesc desc{};
        desc.roceAttr.queueNum = 1;
        return desc;
    }
};

TEST_F(AivUrmaChannelTest, Ut_Clean_WhenCalled_Returns_SUCCESS)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());

    EXPECT_EQ(ch.Clean(), HCCL_SUCCESS);
    EXPECT_EQ(ch.Clean(), HCCL_SUCCESS);
}

TEST_F(AivUrmaChannelTest, Ut_Resume_WhenBuildsMocked_Returns_SUCCESS)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());

    MOCKER_CPP_VIRTUAL(ch, &AivUrmaChannel::BuildConnection, HcclResult(AivUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(ch, &AivUrmaChannel::BuildBuffer, HcclResult(AivUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(ch.Resume(), HCCL_SUCCESS);
}

// ===================== Resume 错误码传播钉住 =====================
// Resume 内 BuildConnection/BuildBuffer 裸调已改 CHK_RET：重建失败须上抛，不得静默吞错后返回 SUCCESS

TEST_F(AivUrmaChannelTest, Ut_Resume_WhenBuildConnectionFails_Expect_ErrorPropagated)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());

    MOCKER_CPP_VIRTUAL(ch, &AivUrmaChannel::BuildConnection, HcclResult(AivUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    EXPECT_NE(ch.Resume(), HCCL_SUCCESS);
}

TEST_F(AivUrmaChannelTest, Ut_Resume_WhenBuildBufferFails_Expect_ErrorPropagated)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());

    MOCKER_CPP_VIRTUAL(ch, &AivUrmaChannel::BuildConnection, HcclResult(AivUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(ch, &AivUrmaChannel::BuildBuffer, HcclResult(AivUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    EXPECT_NE(ch.Resume(), HCCL_SUCCESS);
}

TEST_F(AivUrmaChannelTest, Ut_Init_WhenBuildsMocked_Returns_SUCCESS)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());

    MOCKER_CPP_VIRTUAL(ch, &AivUrmaChannel::ParseInputParam, HcclResult(AivUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(ch, &Channel::StartListen, HcclResult(Channel::*)())
        .expects(once())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(ch, &Channel::BuildSocket, HcclResult(Channel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AivUrmaChannel::BuildAttr, HcclResult(AivUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(ch, &AivUrmaChannel::BuildConnection, HcclResult(AivUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(ch, &AivUrmaChannel::BuildBuffer, HcclResult(AivUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(ch.Init(), HCCL_SUCCESS);
}

TEST_F(AivUrmaChannelTest, Ut_BuildAttr_WhenCalled_FillsAivOpBaseAttr)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    ch.local_.ep.loc.device.devPhyId = 3;

    EXPECT_EQ(ch.BuildAttr(), HCCL_SUCCESS);
    EXPECT_EQ(ch.local_.attr.devicePhyId, 3);
    EXPECT_EQ(ch.local_.attr.opMode, OpMode::OPBASE);
    EXPECT_EQ(ch.local_.attr.opAcceState, AcceleratorState::AIV);
}

TEST_F(AivUrmaChannelTest, Ut_BuildConnection_When_UbProtocol_Expect_RetainsConfiguredSqDepth)
{
    enum class ConnectionType {
        TP,
        CTP,
        UBG,
    };

    struct TestCase {
        CommProtocol protocol;
        ConnectionType connectionType;
    };

    const TestCase testCases[] = {
        {COMM_PROTOCOL_UBC_TP, ConnectionType::TP},
        {COMM_PROTOCOL_UBC_CTP, ConnectionType::CTP},
        {COMM_PROTOCOL_UBG, ConnectionType::UBG},
    };

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    // BuildConnection 顶部合并了 HccpRaGetDevBaseAttr 设备属性准备，打桩避免覆写预置的 devBaseAttr_
    MOCKER(HccpRaGetDevBaseAttr).stubs().will(returnValue(HCCL_SUCCESS));
    for (const auto& testCase : testCases) {
        SCOPED_TRACE(testing::Message() << "protocol=" << testCase.protocol);
        HcommChannelDesc desc = MakeDefaultDesc();
        desc.ubAttr.sqDepth = 512U;
        AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
        ch.local_.ep.protocol = testCase.protocol;
        ch.local_.ep.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        ch.local_.ep.commAddr.addr = IpAddress("1.0.0.1").GetBinaryAddress().addr;
        ch.remote_.ep.protocol = testCase.protocol;
        ch.remote_.ep.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        ch.remote_.ep.commAddr.addr = IpAddress("2.0.0.1").GetBinaryAddress().addr;
        ch.local_.rdmaHandle = reinterpret_cast<RdmaHandle>(0x1);
        ch.devBaseAttr_.sqMaxDepth = 1024U;

        ASSERT_EQ(ch.BuildConnection(), HCCL_SUCCESS);
        ASSERT_EQ(ch.connections_.size(), 1U);
        EXPECT_EQ(ch.connections_[0]->GetSqDepth(), 512U);
        switch (testCase.connectionType) {
            case ConnectionType::TP:
                EXPECT_NE(dynamic_cast<DevUbTpConnection*>(ch.connections_[0].get()), nullptr);
                break;
            case ConnectionType::CTP:
                EXPECT_NE(dynamic_cast<DevUbCtpConnection*>(ch.connections_[0].get()), nullptr);
                break;
            case ConnectionType::UBG:
                EXPECT_NE(dynamic_cast<DevUbRtpConnection*>(ch.connections_[0].get()), nullptr);
                break;
        }
    }
}

TEST_F(AivUrmaChannelTest, Ut_BuildSocket_WhenSocketExists_Returns_SUCCESS)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    ch.local_.socket = &socket;

    EXPECT_EQ(ch.BuildSocket(), HCCL_SUCCESS);
    ch.local_.socket = nullptr;
}

TEST_F(AivUrmaChannelTest, Ut_BuildSocket_WhenSocketNull_GetsSocketFromSocketMgr)
{
    // 收敛后 BuildSocket 走基类 traits_.socketBuildMode 分派，须先经 ParseInputParam 折算（AIV_URMA → BORROW）；
    // 构造方式对齐 Ut_Destroy_WhenBorrowOnNonZeroPhyId（真 stub endpoint + hrt 出参折算 local_.devicePhyId）
    EndpointDesc local{};
    local.loc.device.devPhyId = 3;
    // ROCE 协议规避 HCCS 默认值在 LinkData 协议转换上的不支持路径（对齐 Destroy 用例）
    local.protocol = COMM_PROTOCOL_ROCE;
    StubEndpointForAivUrmaChannel endpoint(local, reinterpret_cast<void*>(0x1));
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 61001;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    ch.local_.socket = nullptr;

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    unsigned int devicePhyId = 3U;
    MOCKER(hrtGetDevicePhyIdByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(devicePhyId))
        .will(returnValue(HCCL_SUCCESS));
    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_EQ(ch.traits_.socketBuildMode, SocketBuildMode::BORROW); // AIV_URMA 折算为纯借用形态

    MOCKER_CPP(&SocketMgr::GetSocket).stubs().with(mockcpp::any(), outBound(&socket)).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(ch.BuildSocket(), HCCL_SUCCESS);
    EXPECT_EQ(ch.local_.socket, &socket);
    ASSERT_NE(ch.local_.socketConfigHolder, nullptr);
    EXPECT_EQ(ch.local_.socketConfig, ch.local_.socketConfigHolder.get());
    EXPECT_EQ(ch.local_.socketConfig->listeningPort, desc.port);
    EXPECT_EQ(ch.local_.socketConfig->GetRole(), SocketRole::SERVER);
    ch.local_.socket = nullptr; // 防止析构归还打到真实 SocketMgr
}

TEST_F(AivUrmaChannelTest, Ut_BuildSocket_WhenRoleReserved_PreservesNoRankIdMatching)
{
    EndpointDesc local{};
    local.loc.device.devPhyId = 3;
    local.protocol = COMM_PROTOCOL_ROCE;
    StubEndpointForAivUrmaChannel endpoint(local, reinterpret_cast<void*>(0x1));
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 61002;
    desc.role = HCOMM_SOCKET_ROLE_RESERVED;
    AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    ch.local_.socket = nullptr;

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    unsigned int devicePhyId = 3U;
    MOCKER(hrtGetDevicePhyIdByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(devicePhyId))
        .will(returnValue(HCCL_SUCCESS));
    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_EQ(ch.traits_.socketBuildMode, SocketBuildMode::BORROW);

    MOCKER_CPP(&SocketMgr::GetSocket).stubs().with(mockcpp::any(), outBound(&socket)).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(ch.BuildSocket(), HCCL_SUCCESS);
    EXPECT_EQ(ch.local_.socket, &socket);
    ASSERT_NE(ch.local_.socketConfigHolder, nullptr);
    EXPECT_TRUE(ch.local_.socketConfig->noRankId);
    EXPECT_EQ(ch.local_.socketConfig->listeningPort, desc.port);
    SocketRole expectedRole
        = ch.local_.socketConfig->link.GetLocalAddr() < ch.local_.socketConfig->link.GetRemoteAddr() ?
              SocketRole::SERVER :
              SocketRole::CLIENT;
    EXPECT_EQ(ch.local_.socketConfig->GetRole(), expectedRole);
    ch.local_.socket = nullptr; // 防止析构归还打到真实 SocketMgr
}

TEST_F(AivUrmaChannelTest, Ut_Destroy_WhenBorrowOnNonZeroPhyId_PutsSocketBackToSamePhyIdInstance)
{
    // 回归：非 0 phyId 上借用 socket，析构必须归还到同一 phyId 的 SocketMgr 实例。
    // 曾因子类遮蔽 local_.devicePhyId，PutSocketIfNeeded 读到基类副本恒为 0，
    // 归还打到实例 0 且 DestroySocket 对 miss 静默成功，造成实例 3 的 socket 泄漏。
    EndpointDesc local{};
    local.loc.device.devPhyId = 3;
    // socketMap_ 对 SocketConfig 哈希会经 LinkData::GetLocalPort 调 LinkProtocol2LinkProtoType，
    // 默认 protocol=COMM_PROTOCOL_HCCS(0) 不受支持会抛 NotSupportException，须显式用受支持的 ROCE。
    local.protocol = COMM_PROTOCOL_ROCE;
    StubEndpointForAivUrmaChannel endpoint(local, reinterpret_cast<void*>(0x1));
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 61003;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.channelName = "ut_destroy_phyid3"; // 保证 config 唯一，不与其他用例串扰

    auto ownedSocket = std::make_unique<Socket>(
        nullptr, IpAddress(), 0, IpAddress(), "ut-destroy-phyid3", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    Socket* rawSocket = ownedSocket.get();
    std::unique_ptr<Hccl::SocketConfig> borrowedCfg; // SocketConfig 无默认构造，scope 内拷贝构造
    SocketMgr& phyIdMgr = GetSocketMgr(3);

    {
        AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);
        // 8edfed0cb 起设备号获取前移至 Init 入口，ParseInputParam 不再折算 local_.devicePhyId；
        // 本用例不整链 Init，直接置 3 复现非 0 phyId 归还场景
        EXPECT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
        ch.local_.devicePhyId = 3U;

        // BuildSocket 走 SocketMgr 借用（GetSocket 打桩避免真实建链），config 由通道自己生成
        MOCKER_CPP(&SocketMgr::GetSocket)
            .stubs()
            .with(mockcpp::any(), outBound(rawSocket))
            .will(returnValue(HCCL_SUCCESS));
        EXPECT_EQ(ch.BuildSocket(), HCCL_SUCCESS);
        ASSERT_EQ(ch.local_.socket, rawSocket);
        ASSERT_NE(ch.local_.socketConfig, nullptr);
        borrowedCfg = std::make_unique<Hccl::SocketConfig>(*ch.local_.socketConfig);

        // 实例 3 预登记借用态：refCount=2，析构归还一次后剩 1，不触发真实 Socket::Destroy
        SocketMgr::SocketEntry entry;
        entry.socket = std::move(ownedSocket);
        entry.refCount = 2;
        phyIdMgr.socketMap_[*borrowedCfg] = std::move(entry);
        phyIdMgr.socketInUseMap_[rawSocket] = true;
    } // ch 析构 → ~Channel → PutSocketIfNeeded → GetInstance(local_.devicePhyId).DestroySocket

    // 归还落在实例 3：引用计数 2→1，占用标记已清（若误打到实例 0，此处状态原封不动，用例变红）
    EXPECT_EQ(phyIdMgr.socketMap_.at(*borrowedCfg).refCount, 1U);
    EXPECT_FALSE(phyIdMgr.socketInUseMap_.at(rawSocket).load());
    // 清理静态实例残留，避免影响后续用例
    phyIdMgr.socketMap_.erase(*borrowedCfg);
    phyIdMgr.socketInUseMap_.erase(rawSocket);
}

TEST_F(AivUrmaChannelTest, Ut_StartListen_WhenPortConfigured_UsesConfiguredPort)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 61001;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    // StartListen 已收敛至基类：守卫/端口改读 local_.isSocketServer/local_.serverPort（由 ParseInputParam 折算），
    // 对齐 Ut_Destroy 用例模式：stub endpoint 支撑真实 ParseInputParam，再验基类 StartListen
    EndpointDesc local{};
    local.loc.device.devPhyId = 3;
    local.protocol = COMM_PROTOCOL_ROCE; // SocketConfig 哈希经 LinkData 需受支持的协议，与 Ut_Destroy 保持一致
    StubEndpointForAivUrmaChannel endpoint(local, reinterpret_cast<void*>(0x1));
    AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);
    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    g_listenPort = 0;

    MOCKER(HcommEndpointStartListen).stubs().will(invoke(StubEndpointStartListen));

    EXPECT_EQ(ch.StartListen(), HCCL_SUCCESS);
    EXPECT_EQ(g_listenPort, desc.port);
}

TEST_F(AivUrmaChannelTest, Ut_StartListen_WhenPortZero_UsesDefaultPort)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 0;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    // StartListen 已收敛至基类：守卫/端口改读 local_.isSocketServer/local_.serverPort（由 ParseInputParam 折算），
    // 对齐 Ut_Destroy 用例模式：stub endpoint 支撑真实 ParseInputParam，再验基类 StartListen
    EndpointDesc local{};
    local.loc.device.devPhyId = 3;
    local.protocol = COMM_PROTOCOL_ROCE; // SocketConfig 哈希经 LinkData 需受支持的协议，与 Ut_Destroy 保持一致
    StubEndpointForAivUrmaChannel endpoint(local, reinterpret_cast<void*>(0x1));
    AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);
    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    g_listenPort = 0;

    MOCKER(HcommEndpointStartListen).stubs().will(invoke(StubEndpointStartListen));

    EXPECT_EQ(ch.StartListen(), HCCL_SUCCESS);
    EXPECT_EQ(g_listenPort, 60001);
}

TEST_F(AivUrmaChannelTest, Ut_StartListen_WhenRoleClient_DoesNotStartListen)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 61001;
    desc.role = HCOMM_SOCKET_ROLE_CLIENT;
    // StartListen 已收敛至基类：守卫/端口改读 local_.isSocketServer/local_.serverPort（由 ParseInputParam 折算），
    // 对齐 Ut_Destroy 用例模式：stub endpoint 支撑真实 ParseInputParam，再验基类 StartListen
    EndpointDesc local{};
    local.loc.device.devPhyId = 3;
    local.protocol = COMM_PROTOCOL_ROCE; // SocketConfig 哈希经 LinkData 需受支持的协议，与 Ut_Destroy 保持一致
    StubEndpointForAivUrmaChannel endpoint(local, reinterpret_cast<void*>(0x1));
    AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);
    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    g_listenPort = 0;

    MOCKER(HcommEndpointStartListen).expects(never());

    EXPECT_EQ(ch.StartListen(), HCCL_SUCCESS);
    EXPECT_EQ(g_listenPort, 0);
}

TEST_F(AivUrmaChannelTest, Ut_StartListen_WhenRoleReserved_DoesNotStartListen)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 61002;
    desc.role = HCOMM_SOCKET_ROLE_RESERVED;
    // StartListen 已收敛至基类：守卫/端口改读 local_.isSocketServer/local_.serverPort（由 ParseInputParam 折算），
    // 对齐 Ut_Destroy 用例模式：stub endpoint 支撑真实 ParseInputParam，再验基类 StartListen
    EndpointDesc local{};
    local.loc.device.devPhyId = 3;
    local.protocol = COMM_PROTOCOL_ROCE; // SocketConfig 哈希经 LinkData 需受支持的协议，与 Ut_Destroy 保持一致
    StubEndpointForAivUrmaChannel endpoint(local, reinterpret_cast<void*>(0x1));
    AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);
    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    g_listenPort = 0;

    MOCKER(HcommEndpointStartListen).expects(never());

    EXPECT_EQ(ch.StartListen(), HCCL_SUCCESS);
    EXPECT_EQ(g_listenPort, 0);
}

TEST_F(AivUrmaChannelTest, Ut_PutSocketIfNeeded_WhenCalledTwice_OnlyKeepsSocketNull)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    std::string socketTag = "ut";
    Hccl::SocketConfig socketConfig(MakeDefaultLinkData(), socketTag, true);
    ch.local_.socket = &socket;
    ch.local_.socketConfig = &socketConfig;

    MOCKER_CPP(&SocketMgr::PutSocket).stubs().will(returnValue(HCCL_SUCCESS));

    ch.PutSocketIfNeeded();
    EXPECT_EQ(ch.local_.socket, nullptr);

    ch.PutSocketIfNeeded();
    EXPECT_EQ(ch.local_.socket, nullptr);
}

TEST_F(AivUrmaChannelTest, Ut_UnsupportedDataApis_WhenCalled_Return_NOT_SUPPORT)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    uint32_t notifyNum = 1;

    EXPECT_EQ(ch.GetNotifyNum(&notifyNum), HCCL_SUCCESS);
    EXPECT_EQ(ch.NotifyRecord(0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.NotifyWait(0, 0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.WriteWithNotify(nullptr, nullptr, 0, 0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.Write(nullptr, nullptr, 0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.Read(nullptr, nullptr, 0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.ChannelFence(), HCCL_E_NOT_SUPPORT);
}

TEST_F(AivUrmaChannelTest, Ut_Init_WhenEndpointNull_Returns_E_PTR)
{
    AivUrmaChannel ch(nullptr, MakeDefaultDesc());

    EXPECT_EQ(ch.Init(), HCCL_E_PTR);
}

TEST_F(AivUrmaChannelTest, Ut_ParseInputParam_WhenExchangeAllMemsGetFailed_ReturnsError)
{
    EndpointDesc local{};
    local.loc.device.devPhyId = 3;
    StubEndpointForAivUrmaChannel endpoint(local, reinterpret_cast<void*>(0x1));
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.exchangeAllMems = true;
    AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubHcommMemGetAllMemHandlesFail));

    EXPECT_EQ(ch.ParseInputParam(), HCCL_E_INTERNAL);
    EXPECT_TRUE(ch.local_.commonRes.bufferVec.empty());
}

TEST_F(AivUrmaChannelTest, Ut_ParseInputParam_WhenExchangeAllMemsEmpty_ReturnsSuccess)
{
    EndpointDesc local{};
    local.loc.device.devPhyId = 3;
    StubEndpointForAivUrmaChannel endpoint(local, reinterpret_cast<void*>(0x1));
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.exchangeAllMems = true;
    AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubHcommMemGetAllMemHandlesEmpty));
    EXPECT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_TRUE(ch.local_.commonRes.bufferVec.empty());
}

TEST_F(AivUrmaChannelTest, UT_ParseInputParam_When_ExchangeAllMemsHasOneHandle_Expect_FillCommonRes)
{
    EndpointDesc local{};
    local.loc.device.devPhyId = 3;
    StubEndpointForAivUrmaChannel endpoint(local, reinterpret_cast<void*>(0x1));
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.exchangeAllMems = true;
    AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubHcommMemGetAllMemHandlesOne));

    EXPECT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.local_.commonRes.bufferVec.size(), 1U);
    ASSERT_NE(ch.local_.commonRes.bufferVec[0], nullptr);
    EXPECT_EQ(ch.local_.commonRes.bufferVec[0]->GetAddr(), 0x12340000U);
    EXPECT_EQ(ch.local_.commonRes.bufferVec[0]->GetSize(), 0x2000U);
    EXPECT_FALSE(ch.local_.commonRes.bufferVec[0]->GetBuf()->GetMemInfo().empty());
}

TEST_F(AivUrmaChannelTest, UT_ParseInputParam_When_ExchangeAllMemsFalse_Expect_FillCommonRes)
{
    EndpointDesc local{};
    local.loc.device.devPhyId = 3;
    StubEndpointForAivUrmaChannel endpoint(local, reinterpret_cast<void*>(0x1));
    auto buffer = std::make_shared<Hccl::Buffer>(0x22340000U, 0x1000U, "aiv_urma_desc");
    auto localBuffer = std::make_shared<Hccl::LocalUbRmaBuffer>(buffer);
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.exchangeAllMems = false;
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    EXPECT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.local_.commonRes.bufferVec.size(), 1U);
    EXPECT_EQ(ch.local_.commonRes.bufferVec[0], localBuffer.get());
}

TEST_F(AivUrmaChannelTest, Ut_BuildChannelEntityToDevice_WhenDevPtrNull_Returns_E_PTR)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());

    EXPECT_EQ(ch.GetChannelEntity()->BuildChannelEntityToDevice(nullptr), HCCL_E_PTR);
}

TEST_F(AivUrmaChannelTest, Ut_BuildChannelEntityToDevice_WhenTransportNull_Returns_E_PTR)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    void* devChannelPtr = nullptr;

    EXPECT_EQ(ch.GetChannelEntity()->BuildChannelEntityToDevice(&devChannelPtr), HCCL_E_PTR);
    EXPECT_EQ(devChannelPtr, nullptr);
}

TEST_F(AivUrmaChannelTest, Ut_GetStatus_WhenTransportStatusTimeout_Returns_SOCKET_TIMEOUT)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());

    BaseMemTransport::CommonLocRes commonRes{};
    DevUbCtpConnection conn(reinterpret_cast<RdmaHandle>(0x1), IpAddress(), IpAddress(), OpMode::OPBASE);
    commonRes.connVec.emplace_back(&conn);
    BaseMemTransport::Attribution attr{};
    LinkData linkData = MakeDefaultLinkData();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    ch.transport_
        = std::make_unique<AivUrmaTransport>(commonRes, attr, linkData, socket, reinterpret_cast<RdmaHandle>(0x1));
    SocketStatus fakeSocketStatus = SocketStatus::TIMEOUT;

    MOCKER_CPP(&Socket::GetAsyncStatus, SocketStatus(Socket::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(fakeSocketStatus));

    EXPECT_EQ(ch.GetStatus(), ChannelStatus::SOCKET_TIMEOUT);
}

TEST_F(AivUrmaChannelTest, Ut_BuildChannelEntityToDevice_WhenTransportReady_ReturnsDevicePtr)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    DevUbCtpConnection conn(reinterpret_cast<RdmaHandle>(0x1), IpAddress(), IpAddress(), OpMode::OPBASE);
    BuildReadyTransport(ch, conn);

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));

    void* devChannelPtr = nullptr;
    EXPECT_EQ(ch.GetChannelEntity()->BuildChannelEntityToDevice(&devChannelPtr), HCCL_SUCCESS);
    ASSERT_NE(devChannelPtr, nullptr);
    EXPECT_EQ(ch.aivChannelHelper_.devChannelEntity_, devChannelPtr);
    ASSERT_NE(ch.aivChannelHelper_.devChannelEntitySlab_, nullptr);
    EXPECT_EQ(ch.aivChannelHelper_.devChannelEntitySlab_, devChannelPtr);
    EXPECT_GT(ch.aivChannelHelper_.devChannelEntitySlabSize_, sizeof(ChannelEntity));

    ChannelEntity* devChannel = reinterpret_cast<ChannelEntity*>(devChannelPtr);
    EXPECT_EQ(devChannel->engine, COMM_ENGINE_AIV);
    EXPECT_EQ(devChannel->sqNum, 1);
    EXPECT_EQ(devChannel->cqNum, 1);
    EXPECT_NE(devChannel->sqContextAddr, nullptr);
    EXPECT_NE(devChannel->cqContextAddr, nullptr);
    EXPECT_TRUE(IsPtrInRange(
        devChannel->sqContextAddr, ch.aivChannelHelper_.devChannelEntitySlab_,
        ch.aivChannelHelper_.devChannelEntitySlabSize_));
    EXPECT_TRUE(IsPtrInRange(
        devChannel->cqContextAddr, ch.aivChannelHelper_.devChannelEntitySlab_,
        ch.aivChannelHelper_.devChannelEntitySlabSize_));
    EXPECT_TRUE(IsPtrInRange(
        reinterpret_cast<void*>(devChannel->sqContextAddr[0].contextInfo.ubJfs.headAddr),
        ch.aivChannelHelper_.devChannelEntitySlab_, ch.aivChannelHelper_.devChannelEntitySlabSize_));
    EXPECT_TRUE(IsPtrInRange(
        reinterpret_cast<void*>(devChannel->sqContextAddr[0].contextInfo.ubJfs.tailAddr),
        ch.aivChannelHelper_.devChannelEntitySlab_, ch.aivChannelHelper_.devChannelEntitySlabSize_));
    EXPECT_TRUE(IsPtrInRange(
        reinterpret_cast<void*>(devChannel->cqContextAddr[0].contextInfo.ubJfc.headAddr),
        ch.aivChannelHelper_.devChannelEntitySlab_, ch.aivChannelHelper_.devChannelEntitySlabSize_));
    EXPECT_TRUE(IsPtrInRange(
        reinterpret_cast<void*>(devChannel->cqContextAddr[0].contextInfo.ubJfc.tailAddr),
        ch.aivChannelHelper_.devChannelEntitySlab_, ch.aivChannelHelper_.devChannelEntitySlabSize_));
}

TEST_F(AivUrmaChannelTest, Ut_BuildChannelEntityToDevice_WhenAclrtMallocFail_ReturnsMemoryError)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    DevUbCtpConnection conn(reinterpret_cast<RdmaHandle>(0x1), IpAddress(), IpAddress(), OpMode::OPBASE);
    BuildReadyTransport(ch, conn);

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMallocFail));

    void* devChannelPtr = nullptr;
    EXPECT_EQ(ch.GetChannelEntity()->BuildChannelEntityToDevice(&devChannelPtr), HCCL_E_MEMORY);
    EXPECT_EQ(devChannelPtr, nullptr);
    EXPECT_EQ(ch.aivChannelHelper_.devChannelEntitySlab_, nullptr);
    EXPECT_EQ(ch.aivChannelHelper_.devChannelEntity_, nullptr);
}

TEST_F(AivUrmaChannelTest, Ut_Clean_WhenDeviceChannelEntityBuilt_ReleasesSlab)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    DevUbCtpConnection conn(reinterpret_cast<RdmaHandle>(0x1), IpAddress(), IpAddress(), OpMode::OPBASE);
    BuildReadyTransport(ch, conn);

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));

    void* devChannelPtr = nullptr;
    ASSERT_EQ(ch.GetChannelEntity()->BuildChannelEntityToDevice(&devChannelPtr), HCCL_SUCCESS);
    ASSERT_NE(ch.aivChannelHelper_.devChannelEntitySlab_, nullptr);

    EXPECT_EQ(ch.Clean(), HCCL_SUCCESS);
    EXPECT_EQ(ch.aivChannelHelper_.devChannelEntitySlab_, nullptr);
    EXPECT_EQ(ch.aivChannelHelper_.devChannelEntitySlabSize_, 0);
    EXPECT_EQ(ch.aivChannelHelper_.devChannelEntity_, nullptr);
}

class AivUrmaTransportTest : public testing::Test {
protected:
    static void TearDownTestCase() { GlobalMockObject::verify(); }

    void TearDown() override { GlobalMockObject::verify(); }

    std::unique_ptr<DevUbCtpConnection> MakeConn()
    {
        return std::make_unique<DevUbCtpConnection>(
            reinterpret_cast<RdmaHandle>(0x1), IpAddress(), IpAddress(), OpMode::OPBASE);
    }

    std::unique_ptr<AivUrmaTransport> MakeTransport(DevUbConnection& conn, Socket& socket)
    {
        commonRes_.connVec.clear();
        commonRes_.bufferVec.clear();
        commonRes_.connVec.emplace_back(&conn);
        attr_.opAcceState = AcceleratorState::AIV;
        LinkData linkData = MakeDefaultLinkData();
        return std::make_unique<AivUrmaTransport>(
            commonRes_, attr_, linkData, socket, reinterpret_cast<RdmaHandle>(0x1));
    }

    BaseMemTransport::CommonLocRes commonRes_{};
    BaseMemTransport::Attribution attr_{};
};

TEST_F(AivUrmaTransportTest, Ut_Construct_WhenConnNull_Expect_ThrowInternalException)
{
    BaseMemTransport::CommonLocRes commonRes{};
    commonRes.connVec.emplace_back(nullptr);
    BaseMemTransport::Attribution attr{};
    LinkData linkData = MakeDefaultLinkData();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);

    EXPECT_THROW(
        AivUrmaTransport transport(commonRes, attr, linkData, socket, reinterpret_cast<RdmaHandle>(0x1)),
        InternalException);
}

TEST_F(AivUrmaTransportTest, Ut_DescribeAndLinkDesc_WhenCalled_ReturnsNonEmpty)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);

    EXPECT_FALSE(transport->Describe().empty());
    EXPECT_FALSE(transport->GetLinkDescInfo().empty());
}

TEST_F(AivUrmaTransportTest, Ut_CheckCommonLocRes_WhenConnValid_Returns)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);

    EXPECT_NO_THROW(transport->CheckLocBuffer(commonRes_));
    EXPECT_EQ(transport->CheckLocConn(commonRes_), HCCL_SUCCESS);
    EXPECT_EQ(transport->CheckCommonLocRes(commonRes_), HcclResult::HCCL_SUCCESS);
}

TEST_F(AivUrmaTransportTest, Ut_HandshakePackUnpack_WhenEmptyMsg_Returns)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    BinaryStream binaryStream;

    transport->HandshakeMsgPack(binaryStream);
    EXPECT_EQ(transport->HandshakeMsgUnpack(binaryStream), HCCL_SUCCESS);
    EXPECT_EQ(transport->rmtOpAcceState_, AcceleratorState::AIV);
}

TEST_F(AivUrmaTransportTest, Ut_HandshakeUnpack_WhenAcceleratorMismatch_ReturnsError)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    BinaryStream binaryStream;
    binaryStream << static_cast<uint32_t>(AcceleratorState::CCU_MS);
    binaryStream << std::vector<char>{};

    EXPECT_EQ(transport->HandshakeMsgUnpack(binaryStream), HCCL_E_PARA);
}

TEST_F(AivUrmaTransportTest, Ut_HandshakeUnpack_WhenMsgSizeMismatch_ReturnsError)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    BinaryStream binaryStream;
    binaryStream << static_cast<uint32_t>(AcceleratorState::AIV);
    binaryStream << std::vector<char>{'x'};

    EXPECT_EQ(transport->HandshakeMsgUnpack(binaryStream), HCCL_E_PARA);
}

TEST_F(AivUrmaTransportTest, Ut_BufferVecPack_WhenCalled_FillsBinaryStream)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    BinaryStream binaryStream;

    EXPECT_EQ(transport->BufferVecPack(binaryStream), HCCL_SUCCESS);
}

TEST_F(AivUrmaTransportTest, Ut_BufferVecPack_WhenBufferVecHasLocalBuffer_PreservesRegisteredToken)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    auto buffer = std::make_shared<Hccl::Buffer>(0x99880000U, 0x1000U, "aiv_urma_qpt");
    auto localBuffer = std::make_shared<StubLocalUbRmaBufferWithExchangeDto>(buffer);
    transport->commonLocRes_.bufferVec.emplace_back(localBuffer.get());
    BinaryStream binaryStream;

    EXPECT_EQ(transport->BufferVecPack(binaryStream), HCCL_SUCCESS);
    uint32_t bufferNum = 0;
    uint32_t pos = 0;
    ExchangeUbBufferDto dto;
    binaryStream >> bufferNum >> pos;
    dto.Deserialize(binaryStream);

    EXPECT_EQ(bufferNum, 1U);
    EXPECT_EQ(pos, 0U);
    EXPECT_EQ(dto.tokenId, STUB_TOKEN_ID);
    EXPECT_EQ(dto.tokenValue, STUB_TOKEN_VALUE);
}

TEST_F(AivUrmaTransportTest, Ut_IsResReadyAndConnsReady_WhenConnReady_ReturnsTrue)
{
    auto conn = MakeConn();
    conn->status = RmaConnStatus::READY;
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);

    bool isReady = false;
    EXPECT_EQ(transport->IsResReady(isReady), HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(isReady);
    EXPECT_TRUE(transport->IsConnsReady());
}

TEST_F(AivUrmaTransportTest, Ut_PrepareGetStatus_WhenAlreadyReady_ReturnsFalse)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->transportStatus_ = TransportStatus::READY;

    bool needProcess = true;
    EXPECT_EQ(transport->PrepareGetStatus(needProcess), HcclResult::HCCL_SUCCESS);
    EXPECT_FALSE(needProcess);
    EXPECT_EQ(transport->transportStatus_, TransportStatus::READY);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenInit_SetsSocketOk)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::INIT;

    EXPECT_EQ(transport->ProcessUrmaStatus(), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::SOCKET_OK);
    EXPECT_EQ(transport->transportStatus_, TransportStatus::SOCKET_OK);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenSocketOkAndResNotReady_StaysSocketOk)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::SOCKET_OK;

    MOCKER_CPP(&AivUrmaTransport::IsResReady).stubs().will(invoke(StubIsResReadyFalse));

    EXPECT_EQ(transport->ProcessUrmaStatus(), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::SOCKET_OK);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenSocketOkAndResReady_SendsData)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::SOCKET_OK;

    MOCKER_CPP(&AivUrmaTransport::IsResReady).stubs().will(invoke(StubIsResReadyTrue));
    MOCKER_CPP(&AivUrmaTransport::SendExchangeData).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    EXPECT_EQ(transport->ProcessUrmaStatus(), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::SEND_DATA);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenSendData_RecvsData)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::SEND_DATA;

    MOCKER_CPP(&AivUrmaTransport::RecvExchangeData).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    EXPECT_EQ(transport->ProcessUrmaStatus(), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::RECV_DATA);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenRecvDataNeedsFinish_ProcessesData)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::RECV_DATA;

    MOCKER_CPP(&AivUrmaTransport::RecvDataProcess).stubs().will(invoke(StubRecvDataProcessNeedFinish));

    EXPECT_EQ(transport->ProcessUrmaStatus(), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::PROCESS_DATA);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenRecvDataNeedsNoFinish_BecomesReady)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::RECV_DATA;

    MOCKER_CPP(&AivUrmaTransport::RecvDataProcess).stubs().will(invoke(StubRecvDataProcessNoFinish));

    EXPECT_EQ(transport->ProcessUrmaStatus(), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::RECV_FIN);
    EXPECT_EQ(transport->transportStatus_, TransportStatus::READY);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenProcessDataAndConnsReady_SendsFinish)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::PROCESS_DATA;

    MOCKER_CPP(&AivUrmaTransport::IsConnsReady).stubs().will(returnValue(true));
    MOCKER_CPP(&AivUrmaTransport::SendFinish).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    EXPECT_EQ(transport->ProcessUrmaStatus(), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::CONN_OK);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenConnOk_RecvsFinish)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::CONN_OK;

    MOCKER_CPP(&AivUrmaTransport::RecvFinish).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    EXPECT_EQ(transport->ProcessUrmaStatus(), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::SEND_FIN);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenSendFin_BecomesReady)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::SEND_FIN;

    EXPECT_EQ(transport->ProcessUrmaStatus(), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::RECV_FIN);
    EXPECT_EQ(transport->transportStatus_, TransportStatus::READY);
}

TEST_F(AivUrmaTransportTest, Ut_RmtBufferVecUnpackProc_WhenZeroSizeDto_AppendsNullBuffer)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    BinaryStream binaryStream;
    ExchangeUbBufferDto dto{};
    binaryStream << static_cast<uint32_t>(1);
    binaryStream << static_cast<uint32_t>(0);
    dto.Serialize(binaryStream);
    AivUrmaTransport::RemoteBufferVec bufferVec;

    EXPECT_EQ(transport->RmtBufferVecUnpackProc(1, binaryStream, bufferVec), HCCL_SUCCESS);

    ASSERT_EQ(bufferVec.size(), 1);
    EXPECT_EQ(bufferVec[0], nullptr);
}

TEST_F(AivUrmaTransportTest, Ut_RmtBufferVecUnpackProc_WhenNumMismatch_ReturnsError)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    BinaryStream binaryStream;
    binaryStream << static_cast<uint32_t>(2);
    AivUrmaTransport::RemoteBufferVec bufferVec;

    EXPECT_EQ(transport->RmtBufferVecUnpackProc(1, binaryStream, bufferVec), HCCL_E_PARA);
}

TEST_F(AivUrmaTransportTest, Ut_ConnVecUnpackProc_WhenConnNumZero_ReturnsFalse)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->connNum_ = 0;
    BinaryStream binaryStream;
    binaryStream << static_cast<uint32_t>(0);

    bool result = true;
    EXPECT_EQ(transport->ConnVecUnpackProc(binaryStream, result), HcclResult::HCCL_SUCCESS);
    EXPECT_FALSE(result);
}

TEST_F(AivUrmaTransportTest, Ut_ConnVecUnpackProc_WhenConnNumMismatch_Expect_E_PARA)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    BinaryStream binaryStream;
    binaryStream << static_cast<uint32_t>(2);

    bool result;
    EXPECT_EQ(transport->ConnVecUnpackProc(binaryStream, result), HcclResult::HCCL_E_PARA);
}

TEST_F(AivUrmaTransportTest, Ut_GetRemoteMems_WhenParamNull_Returns_E_PARA)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    uint32_t memNum = 0;

    EXPECT_EQ(transport->GetRemoteMems(&memNum, nullptr, &memInfos), HCCL_E_PARA);
    EXPECT_EQ(transport->GetRemoteMems(&memNum, &remoteMem, nullptr), HCCL_E_PARA);
    EXPECT_EQ(transport->GetRemoteMems(nullptr, &remoteMem, &memInfos), HCCL_E_PARA);
}

TEST_F(AivUrmaTransportTest, Ut_GetRemoteMems_WhenNoRemoteBuffer_Returns_SUCCESS)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    CommMem* remoteMem = reinterpret_cast<CommMem*>(0x1);
    char** memInfos = reinterpret_cast<char**>(0x1);
    uint32_t memNum = 0;

    EXPECT_EQ(transport->GetRemoteMems(&memNum, &remoteMem, &memInfos), HCCL_SUCCESS);
}

TEST_F(AivUrmaTransportTest, Ut_GetRemoteMems_WhenOnlyReservedRemoteBuffer_Returns_SUCCESS)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    void* rdmaHandle = (void*)0x100;
    auto remoteRmaBuffer = std::make_unique<Hccl::RemoteUbRmaBuffer>(rdmaHandle);
    transport->rmtBufferVec_.push_back(std::move(remoteRmaBuffer));
    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    uint32_t memNum = 1;

    EXPECT_EQ(transport->GetRemoteMems(&memNum, &remoteMem, &memInfos), HCCL_SUCCESS);
    EXPECT_EQ(memNum, 1);
}

TEST_F(AivUrmaTransportTest, Ut_GetStatus_WhenSocketTimeout_Returns_SOCKET_TIMEOUT)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    SocketStatus fakeSocketStatus = SocketStatus::TIMEOUT;

    MOCKER_CPP(&Socket::GetAsyncStatus, SocketStatus(Socket::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(fakeSocketStatus));

    EXPECT_EQ(transport->GetStatus(), TransportStatus::SOCKET_TIMEOUT);
}

TEST_F(AivUrmaTransportTest, Ut_GetHostChannelEntity_WhenNotReady_ReturnsError)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    ChannelEntity hostChannel{};

    EXPECT_EQ(transport->GetHostChannelEntity(&hostChannel), HCCL_E_INTERNAL);
}

TEST_F(AivUrmaTransportTest, Ut_GetHostChannelEntity_WhenReady_FillsSqAndCqContext)
{
    auto conn = MakeConn();
    conn->status = RmaConnStatus::READY;
    conn->jettyId = 3;
    conn->dbAddr = 0x1000;
    conn->sqBuffVa = 0x2000;
    conn->sqDepth = 4;
    conn->tpn = 5;
    conn->cqInfo_.id = 6;
    conn->cqInfo_.va = 0x3000;
    conn->cqInfo_.cqeSize = 64;
    conn->cqInfo_.cqDepth = 8;
    conn->cqInfo_.swdbAddr = 0x4000;

    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->transportStatus_ = TransportStatus::READY;
    ChannelEntity hostChannel{};

    using DeviceMemAllocRetByValue = hccl::DeviceMem (*)(u64, bool);
    mockcpp::mockAPI<DeviceMemAllocRetByValue>::get(
        "hccl::DeviceMem::alloc", "AivUrmaTransport_DeviceMem_alloc_by_value",
        static_cast<DeviceMemAllocRetByValue>(&hccl::DeviceMem::alloc))
        .stubs()
        .will(invoke(StubDeviceMemAlloc));

    EXPECT_EQ(transport->GetHostChannelEntity(&hostChannel), HCCL_SUCCESS);

    ASSERT_EQ(hostChannel.sqNum, 1);
    ASSERT_NE(hostChannel.sqContextAddr, nullptr);
    EXPECT_EQ(hostChannel.sqContextAddr[0].type, SQ_CONTEXT_TYPE_UB_JFS);
    EXPECT_EQ(hostChannel.sqContextAddr[0].contextInfo.ubJfs.jfsID, conn->jettyId);
    EXPECT_EQ(hostChannel.sqContextAddr[0].contextInfo.ubJfs.dbVa, conn->dbAddr);
    EXPECT_EQ(hostChannel.sqContextAddr[0].contextInfo.ubJfs.sqVa, conn->sqBuffVa);
    EXPECT_EQ(hostChannel.sqContextAddr[0].contextInfo.ubJfs.tpID, conn->tpn);
    EXPECT_EQ(hostChannel.sqContextAddr[0].contextInfo.ubJfs.wqeSize, 64);
    EXPECT_NE(hostChannel.sqContextAddr[0].contextInfo.ubJfs.headAddr, 0);
    EXPECT_NE(hostChannel.sqContextAddr[0].contextInfo.ubJfs.tailAddr, 0);

    ASSERT_EQ(hostChannel.cqNum, 1);
    ASSERT_NE(hostChannel.cqContextAddr, nullptr);
    EXPECT_EQ(hostChannel.cqContextAddr[0].type, CQ_CONTEXT_TYPE_UB_JFC);
    EXPECT_EQ(hostChannel.cqContextAddr[0].contextInfo.ubJfc.jfcID, conn->cqInfo_.id);
    EXPECT_EQ(hostChannel.cqContextAddr[0].contextInfo.ubJfc.scqVa, conn->cqInfo_.va);
    EXPECT_EQ(hostChannel.cqContextAddr[0].contextInfo.ubJfc.cqeSize, conn->cqInfo_.cqeSize);
    EXPECT_EQ(hostChannel.cqContextAddr[0].contextInfo.ubJfc.cqDepth, conn->cqInfo_.cqDepth);
    EXPECT_EQ(hostChannel.cqContextAddr[0].contextInfo.ubJfc.dbVa, conn->cqInfo_.swdbAddr);
    EXPECT_NE(hostChannel.cqContextAddr[0].contextInfo.ubJfc.headAddr, 0);
    EXPECT_NE(hostChannel.cqContextAddr[0].contextInfo.ubJfc.tailAddr, 0);
}

// ===================== PreAllocChannelEntityToDevice / FillChannelEntityToDevice UT =====================

// PreAllocChannelEntityToDevice: devChannelPtr 为 nullptr
TEST_F(AivUrmaChannelTest, Ut_PreAllocChannelEntityToDevice_When_DevPtrNull_Returns_E_PTR)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    EXPECT_EQ(ch.GetChannelEntity()->PreAllocChannelEntityToDevice(nullptr), HCCL_E_PTR);
}

// PreAllocChannelEntityToDevice: transport_ 为 nullptr
TEST_F(AivUrmaChannelTest, Ut_PreAllocChannelEntityToDevice_When_TransportNull_Returns_E_PTR)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    void* devChannelPtr = nullptr;
    EXPECT_EQ(ch.GetChannelEntity()->PreAllocChannelEntityToDevice(&devChannelPtr), HCCL_E_PTR);
    EXPECT_EQ(devChannelPtr, nullptr);
}

// FillChannelEntityToDevice: 未预分配（slab 为 nullptr）→ E_INTERNAL
TEST_F(AivUrmaChannelTest, Ut_FillChannelEntityToDevice_When_NotPreAllocated_Returns_E_INTERNAL)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    EXPECT_EQ(ch.GetChannelEntity()->FillChannelEntityToDevice(), HCCL_E_INTERNAL);
}

// ===================== PreAllocChannelEntityToDevice / FillChannelEntityToDevice 正常路径 UT =====================

// PreAllocChannelEntityToDevice: transport ready，正常分配 slab
TEST_F(AivUrmaChannelTest, Ut_PreAllocChannelEntityToDevice_When_TransportReady_ReturnsDevicePtr)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    DevUbCtpConnection conn(reinterpret_cast<RdmaHandle>(0x1), IpAddress(), IpAddress(), OpMode::OPBASE);
    BuildReadyTransport(ch, conn);

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));

    void* devChannelPtr = nullptr;
    EXPECT_EQ(ch.GetChannelEntity()->PreAllocChannelEntityToDevice(&devChannelPtr), HCCL_SUCCESS);
    ASSERT_NE(devChannelPtr, nullptr);
    ASSERT_NE(ch.aivChannelHelper_.devChannelEntitySlab_, nullptr);
    EXPECT_EQ(ch.aivChannelHelper_.devChannelEntity_, devChannelPtr);
    EXPECT_GT(ch.aivChannelHelper_.devChannelEntitySlabSize_, 0);
}

// PreAllocChannelEntityToDevice: 已分配则返回缓存指针
TEST_F(AivUrmaChannelTest, Ut_PreAllocChannelEntityToDevice_When_AlreadyBuilt_ReturnsCached)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    DevUbCtpConnection conn(reinterpret_cast<RdmaHandle>(0x1), IpAddress(), IpAddress(), OpMode::OPBASE);
    BuildReadyTransport(ch, conn);

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));

    void* firstPtr = nullptr;
    ASSERT_EQ(ch.GetChannelEntity()->PreAllocChannelEntityToDevice(&firstPtr), HCCL_SUCCESS);
    void* secondPtr = nullptr;
    EXPECT_EQ(ch.GetChannelEntity()->PreAllocChannelEntityToDevice(&secondPtr), HCCL_SUCCESS);
    EXPECT_EQ(secondPtr, firstPtr);
}

// FillChannelEntityToDevice: 先 PreAlloc 再 Fill，验证成功
TEST_F(AivUrmaChannelTest, Ut_FillChannelEntityToDevice_When_PreAllocated_ReturnsSuccess)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    DevUbCtpConnection conn(reinterpret_cast<RdmaHandle>(0x1), IpAddress(), IpAddress(), OpMode::OPBASE);
    BuildReadyTransport(ch, conn);

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyReal)));

    void* devChannelPtr = nullptr;
    ASSERT_EQ(ch.GetChannelEntity()->PreAllocChannelEntityToDevice(&devChannelPtr), HCCL_SUCCESS);
    EXPECT_EQ(ch.GetChannelEntity()->FillChannelEntityToDevice(), HCCL_SUCCESS);
    ASSERT_NE(ch.aivChannelHelper_.devChannelEntity_, nullptr);
}

// ===================== BuildNotify 收敛基类后的钉住用例 =====================
// AIV_URMA 折算 NONE：统一 BuildNotify 直通返回，不建任何 notify（原 aiv 无流程语义保持）
TEST_F(AivUrmaChannelTest, Ut_BuildNotify_WhenModeNone_Expect_DirectPassThrough)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    EXPECT_EQ(ch.GetNotifyBuildMode(), NotifyBuildMode::NONE);
    ch.local_.notifyNum = 3; // 非 0 也不得构建
    EXPECT_EQ(ch.BuildNotify(), HCCL_SUCCESS);
    EXPECT_TRUE(ch.local_.notifies.empty());
    EXPECT_TRUE(ch.local_.commonRes.notifyVec.empty());
}
