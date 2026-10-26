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
#include <unordered_set>
#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "rank_graph_interface.h"
#include "rank_graph_v2.h"
#include "hccl/hccl_res.h"
#include "hccl_types.h"
#include <hccl/hccl_comm.h>
#include "hcomm_c_adpt.h"
#include "channel_process.h"
#include "base_config_legacy.h"
#include "ccu_device_res.h"
#include "ccu_device_pub.h"
#include "config/env_config.h"
#include "env_config/env_config_v2.h"
#define private public
#include "my_rank.h"
#include "roce_channel_desc_configurator.h"
#undef private
#include "hccl_comm_pub.h"
#include "host_multi_qp_config.h"
#include "llt_hccl_stub_rank_graph.h"
#include "llt_dca_stub.h"
#include "coll_comm_mgr.h"
#include "ccu_res.h"
#include "orion_adpt_utils.h"
#include "adapter_rts_common.h"
#include "adapter_rts.h"
#include "launch_aicpu.h"
#include <cstdlib>
#include <cstring>

using namespace hccl;

namespace {

constexpr u32 HOST_MULTI_QP_DEVICE_LOGIC_ID = 4U;
constexpr u32 HOST_MULTI_QP_DEVICE_PHY_ID = 3U;
u32 g_hostMultiQpConfigPhyId = HOST_MULTI_QP_DEVICE_PHY_ID;
std::vector<uint16_t> g_hostMultiQpUdpPorts;

HcclResult StubGetDeviceForHostMultiQp(s32* deviceLogicId)
{
    *deviceLogicId = static_cast<s32>(HOST_MULTI_QP_DEVICE_LOGIC_ID);
    return HCCL_SUCCESS;
}

HcclResult StubGetDevicePhyIdForHostMultiQp(u32 deviceLogicId, u32& devicePhyId, bool isRefresh)
{
    EXPECT_EQ(deviceLogicId, HOST_MULTI_QP_DEVICE_LOGIC_ID);
    EXPECT_FALSE(isRefresh);
    devicePhyId = HOST_MULTI_QP_DEVICE_PHY_ID;
    return HCCL_SUCCESS;
}

const std::vector<uint16_t>* StubGetHostMultiQpUdpPorts(hccl::HostMultiQpConfig*, uint32_t devicePhyId)
{
    return devicePhyId == g_hostMultiQpConfigPhyId ? &g_hostMultiQpUdpPorts : nullptr;
}

HcclResult StubResolveUserHandleToHost(ChannelHandle /*userHandle*/, ChannelHandle& hostHandle)
{
    hostHandle = static_cast<ChannelHandle>(0xAAA); // host 哨兵：锁定新建分支 channels 反查回填
    return HCCL_SUCCESS;
}

HcclResult StubResolveHostHandleToDevice(ChannelHandle /*hostHandle*/, ChannelHandle& deviceHandle)
{
    deviceHandle = static_cast<ChannelHandle>(0xBBB); // device 哨兵：锁定复用分支 deviceChannels 反查回填
    return HCCL_SUCCESS;
}

// WaitForChannelsReady 归类用例复用：首轮填 g_stubStatusFirst、后续轮填 g_stubStatus，返回 SUCCESS 触发归类
static int32_t g_stubStatusFirst = static_cast<int32_t>(hcomm::HCOMM_CHANNEL_STATUS_READY);
static int32_t g_stubStatus = static_cast<int32_t>(hcomm::HCOMM_CHANNEL_STATUS_READY);
static uint32_t g_stubStatusCallCount = 0;

static int32_t StubHcommChannelGetStatus(const ChannelHandle* /*channelList*/, uint32_t listNum, int32_t* statusList)
{
    g_stubStatusCallCount++;
    int32_t status = (g_stubStatusCallCount == 1) ? g_stubStatusFirst : g_stubStatus;
    for (uint32_t i = 0; i < listNum; ++i) {
        statusList[i] = status;
    }
    return static_cast<int32_t>(HCCL_SUCCESS);
}

// WaitForChannelsReady 超时桩复用：g_linkTimeout 控制返回值，避免二次 mock
static s32 g_linkTimeout = 1;
static s32 StubGetLinkTimeOut() { return g_linkTimeout; }

} // namespace

HcclResult GetAllMemoryStub(CommMems*, std::vector<HcclMem>& mems, std::vector<std::string>& tags, uint64_t& version)
{
    HcclMem m;
    m.addr = (void*)0xCC11000;
    m.size = 1024;
    m.type = HCCL_MEM_TYPE_DEVICE;
    mems.push_back(m);
    tags.push_back("HcclBuffer");
    version = 1;
    return HCCL_SUCCESS;
}
HcclResult RegisterMemoryStub(
    hccl::EndpointMgr*, EndpointHandle, const std::vector<std::string>&, const std::vector<HcclMem>& mems, uint64_t)
{
    return HCCL_SUCCESS;
}
HcclResult GetMemHandlesByTagsStub(
    hccl::EndpointMgr* mgr, EndpointHandle, const std::vector<std::string>& tags, std::vector<MemHandle>& memHandleVec)
{
    for (size_t i = 0; i < tags.size(); i++) {
        memHandleVec.push_back((MemHandle)(0xCC11000 + i));
    }
    return HCCL_SUCCESS;
}

// spy 桩：记录 RegisterMemory 调用信息
struct RegisterMemoryCallRecord {
    EndpointHandle ep;
    std::vector<std::string> tags;
    std::vector<HcclMem> mems;
};
static std::vector<RegisterMemoryCallRecord> s_registerMemoryCalls;
HcclResult RegisterMemorySpyStub(
    hccl::EndpointMgr* mgr, EndpointHandle ep, const std::vector<std::string>& tags, const std::vector<HcclMem>& mems,
    uint64_t version)
{
    // 模拟真实 RegisterMemory：版本一致则跳过
    auto [it, _] = mgr->endpointTagMemMap_.try_emplace(ep, ep);
    if (it->second.GetVersion() == version) {
        return HCCL_SUCCESS;
    }
    s_registerMemoryCalls.push_back({ep, tags, mems});
    it->second.SetVersion(version);
    return HCCL_SUCCESS;
}

// E2E 测试用的 spy 桩：统计 HcommMemReg/HcommMemUnreg 调用
static int s_e2eRegCount = 0;
static int s_e2eUnregCount = 0;
static HcommResult E2EMemRegStub(EndpointHandle, const char* tag, const CommMem*, HcommMemHandle* h)
{
    s_e2eRegCount++;
    *h = (HcommMemHandle)(uintptr_t)(0xCC100000 + s_e2eRegCount);
    return HCCL_SUCCESS;
}

static HcommResult E2EMemUnregStub(EndpointHandle, HcommMemHandle)
{
    s_e2eUnregCount++;
    return HCCL_SUCCESS;
}

static HcommResult E2EEndpointDestroyStub(EndpointHandle) { return HCCL_SUCCESS; }

class MyRankTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MyRankTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "MyRankTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        s_registerMemoryCalls.clear();
        std::cout << "A Test case in MyRankTest SetUP" << std::endl;
        MOCKER(hrtGetDeviceType).stubs().will(invoke(+[](DevType& t) -> HcclResult {
            t = DevType::DEV_TYPE_950;
            return HCCL_SUCCESS;
        }));
        rankIpPortMap = std::make_shared<std::unordered_map<u32, std::unordered_map<Hccl::IpAddress, u32>>>();
        (*rankIpPortMap)[0][Hccl::IpAddress("1.0.0.0")] = 16666;
        (*rankIpPortMap)[0][Hccl::IpAddress("2.0.0.0")] = 16666;
        (*rankIpPortMap)[1][Hccl::IpAddress("2.0.0.0")] = 16666;
        (*rankIpPortMap)[1][Hccl::IpAddress("3.0.0.0")] = 16666;
        (*rankIpPortMap)[2][Hccl::IpAddress("0.0.0.0")] = 16666;
        (*rankIpPortMap)[2][Hccl::IpAddress("3.0.0.0")] = 16666;
        rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
        myRank = std::make_unique<MyRank>(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);
        roceDescConfigurator = std::make_unique<RoceChannelDescConfigurator>(1);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in MyRankTest TearDown" << std::endl;
    }

    void CreateCclBuffer(HcclMem& cclBuffer)
    {
        cclBuffer.addr = (void*)0xab;
        cclBuffer.size = 1024;
        cclBuffer.type = HCCL_MEM_TYPE_DEVICE;
    }

    void CreateEndpointDesc(EndpointDesc& ep, CommProtocol protocol, const std::string& ip)
    {
        ep.protocol = protocol;
        ep.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        ep.commAddr.addr = Hccl::IpAddress(ip).GetBinaryAddress().addr;
        ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    }

    void MockerFuncs()
    {
        MOCKER_CPP(&Hccl::SocketManager::GetConnectedSocket)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue((Hccl::Socket*)0xab));
        MOCKER_CPP(&hccl::CommMems::GetAllMemory).stubs().will(invoke(GetAllMemoryStub));
        MOCKER_CPP(&hccl::EndpointMgr::RegisterMemory).stubs().will(invoke(RegisterMemoryStub));
        MOCKER_CPP(&hccl::EndpointMgr::GetMemHandlesByTags).stubs().will(invoke(GetMemHandlesByTagsStub));
        MOCKER(HcommCcuInsCreate).stubs().with(mockcpp::any()).will(returnValue(CcuResult::CCU_SUCCESS));
        MOCKER(HcommCcuInsCreateLegacy).stubs().with(mockcpp::any()).will(returnValue(CcuResult::CCU_SUCCESS));
        MOCKER_CPP(&hccl::MyRank::TryInitCcuInstance).stubs().will(returnValue(HCCL_SUCCESS));
    }

    void SetupForQuery(HcclMem& cclBuffer, EndpointDesc& localEp, EndpointDesc& rmtEp)
    {
        MockerFuncs();
        MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));
        CreateCclBuffer(cclBuffer);
        EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);
        CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
        CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "2.0.0.0");
    }

    hcomm::EndpointPair*
    GetEpPair(uint32_t localRank, uint32_t remoteRank, const EndpointDesc& localEp, const EndpointDesc& rmtEp)
    {
        RankIdPair rankIdPair = std::make_pair(localRank, remoteRank);
        EndpointDescPair epDescPair = std::make_pair(localEp, rmtEp);
        RankPair* rankPair = nullptr;
        myRank->rankPairMgr_->Get(rankIdPair, rankPair);
        hcomm::EndpointPair* epPair = nullptr;
        rankPair->GetEndpointPair(epDescPair, epPair);
        return epPair;
    }

    // 填 channelHandles_ + handleToLoc_ + handleToEpPair_, reuseIdx 即 vector 下标
    void RegisterChannels(hcomm::EndpointPair* epPair, CommEngine engine, const std::vector<ChannelHandle>& handles)
    {
        epPair->channelHandles_[engine] = handles;
        for (uint32_t i = 0; i < handles.size(); ++i) {
            epPair->handleToLoc_[handles[i]] = {engine, i};
            myRank->handleToEpPair_[handles[i]] = epPair;
        }
    }

    uint32_t DEFAULT_MODE = 0;
    uint32_t AICPU_TS_MODE = 2;
    uint32_t CCU_MS_MODE = 5;
    uint32_t CCU_SCHED_MODE = 6;
    Hccl::RankIpPortMapPtr rankIpPortMap;
    aclrtBinHandle binHandle;
    CommConfig config{"my_rank_ut"};
    ManagerCallbacks callbacks;
    // 真实Hccl::RankGraph对象：RankGraphV2经void*持有其裸指针（IsLevel0PcieFallback直读），
    // 传伪造指针会在标志查询时解引用崩溃；其余查询仍走IRankGraph mock，不受影响
    Hccl::RankGraph realRankGraph{0};
    void* rankGraphPtr = (void*)&realRankGraph;
    std::shared_ptr<RankGraph> rankGraph;
    std::unique_ptr<MyRank> myRank;
    std::unique_ptr<RoceChannelDescConfigurator> roceDescConfigurator;
};

TEST_F(MyRankTest, Ut_When_QueryListenPort_Listen_Port_Expect_SUCCESS)
{
    // 非CPU引擎走两级端口表：端口取表值16666（SetUp已按 rank+IP 填表），不再依赖 rankGraph 查询
    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_ROCE, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_ROCE, "2.0.0.0");

    uint32_t listenPort;
    HcommChannelDesc desc;
    HcclResult ret = myRank->QueryListenPort(0, 1, localEp, rmtEp, listenPort, desc, COMM_ENGINE_AICPU_TS);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listenPort, 16666);
    EXPECT_EQ(desc.role, HCOMM_SOCKET_ROLE_SERVER);
    EXPECT_EQ(desc.port, 16666);

    EndpointDesc rmtEp2;
    CreateEndpointDesc(rmtEp2, COMM_PROTOCOL_ROCE, "0.0.0.0");
    ret = myRank->QueryListenPort(0, 2, localEp, rmtEp2, listenPort, desc, COMM_ENGINE_AICPU_TS);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listenPort, 16666);
    EXPECT_EQ(desc.role, HCOMM_SOCKET_ROLE_CLIENT);
    EXPECT_EQ(desc.port, 16666);
}

TEST_F(MyRankTest, Ut_When_QueryListenPort_InValid_Port_Expect_E_PARA)
{
    // 表中出现超过端口上限的非法值，QueryListenPort 应返回 E_PARA
    (*rankIpPortMap)[1][Hccl::IpAddress("2.0.0.0")] = 1919000;

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_ROCE, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_ROCE, "2.0.0.0");

    uint32_t listenPort;
    HcommChannelDesc desc;
    HcclResult ret = myRank->QueryListenPort(0, 1, localEp, rmtEp, listenPort, desc, COMM_ENGINE_AICPU_TS);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(MyRankTest, Ut_When_QueryListenPort_PortMiss_Expect_FallbackPort)
{
    // 表查不到 (rank, IP)：未配置端口段时用默认端口兜底，继续建链
    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_ROCE, "5.5.5.5");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_ROCE, "1.1.1.1");

    uint32_t listenPort;
    HcommChannelDesc desc;
    HcclResult ret = myRank->QueryListenPort(0, 1, localEp, rmtEp, listenPort, desc, COMM_ENGINE_AICPU_TS);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listenPort, Hccl::DEFAULT_VALUE_TCPPORT);
    EXPECT_EQ(desc.role, HCOMM_SOCKET_ROLE_CLIENT);
}

TEST_F(MyRankTest, Ut_When_GetDevicePortByAddr_EntryFound_Expect_SUCCESS)
{
    // SetUp 已按 (rank, IP) 填表，命中条目返回表值端口
    uint32_t port = 0;
    HcclResult ret = myRank->GetDevicePortByAddr(0, Hccl::IpAddress("1.0.0.0"), &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(port, 16666);
}

TEST_F(MyRankTest, Ut_When_GetDevicePortByAddr_RankMiss_Expect_NOT_FOUND)
{
    uint32_t port = 0;
    HcclResult ret = myRank->GetDevicePortByAddr(99, Hccl::IpAddress("1.0.0.0"), &port);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(MyRankTest, Ut_When_GetDevicePortByAddr_AddrMiss_Expect_NOT_FOUND)
{
    uint32_t port = 0;
    HcclResult ret = myRank->GetDevicePortByAddr(0, Hccl::IpAddress("9.9.9.9"), &port);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(MyRankTest, Ut_When_GetDevicePortByAddr_NullMapOrPort_Expect_E_PTR)
{
    uint32_t port = 0;
    EXPECT_EQ(myRank->GetDevicePortByAddr(0, Hccl::IpAddress("1.0.0.0"), nullptr), HCCL_E_PTR);

    Hccl::RankIpPortMapPtr savedMap = myRank->rankIpPortMap_;
    myRank->rankIpPortMap_ = nullptr;
    EXPECT_EQ(myRank->GetDevicePortByAddr(0, Hccl::IpAddress("1.0.0.0"), &port), HCCL_E_PTR);
    myRank->rankIpPortMap_ = savedMap;
}

TEST_F(MyRankTest, Ut_When_BatchCreateChannels_Expect_SUCCESS)
{
    extern s32 log_level_get_stub();
    extern void log_level_set_stub(s32 logLevel);

    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MockerFuncs();
    MOCKER_CPP(&hccl::MyRank::CollectDcaPeerInfo).stubs().will(returnValue(HCCL_SUCCESS));
    ChannelHandle channelHandle = 0xab;
    MOCKER(hcomm::ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));
    // CreateChannelsLoop 被桩掉后 targetChannels 不被填充，AICPU/AIV 预分配会在空句柄上解引用；
    // 出入口两次反查（user→host、host→device）都依赖建链填充的 D2H/H2D 表，一并打桩
    MOCKER(hcomm::ChannelProcess::PrepareUserChannels).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hcomm::ChannelProcess::ResolveUserHandleToHost).stubs().will(invoke(StubResolveUserHandleToHost));
    MOCKER(hcomm::ChannelProcess::ResolveHostHandleToDevice).stubs().will(invoke(StubResolveHostHandleToDevice));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);
    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "2.0.0.0");
    EndpointDesc rmtEp2;
    CreateEndpointDesc(rmtEp2, COMM_PROTOCOL_UB_MEM, "0.0.0.0");

    HcclChannelDesc channelDesc[3]{};
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    channelDesc[0].localEndpoint = localEp;
    channelDesc[0].remoteEndpoint = rmtEp;
    channelDesc[0].ubMemAttr.pathMode = 0;
    channelDesc[1].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[1].remoteRank = 1;
    channelDesc[1].notifyNum = 2;
    channelDesc[1].localEndpoint = localEp;
    channelDesc[1].remoteEndpoint = rmtEp;
    channelDesc[1].ubMemAttr.pathMode = 0;
    channelDesc[2].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[2].remoteRank = 2;
    channelDesc[2].notifyNum = 2;
    channelDesc[2].localEndpoint = localEp;
    channelDesc[2].remoteEndpoint = rmtEp2;
    channelDesc[2].ubMemAttr.pathMode = 0;

    u32 channelIdx0 = 0u;
    u32 channelIdx1 = 1u;
    u32 channelIdx2 = 2u;
    u32 RmtEp1reuseIdx0 = 0u;
    u32 RmtEp1reuseIdx1 = 1u;
    u32 RmtEp2reuseIdx0 = 0u;

    std::vector<HcommChannelDesc> hcommDesc(3);
    for (u32 i = 0; i < 3; ++i) {
        hcommDesc[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDesc[i], myRank->config_);
    }
    EXPECT_EQ(myRank->BatchCreateSockets(channelDesc, 1, "test", hcommDesc, COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);
    std::vector<ChannelHandle> hostChannelHandles(3);
    ChannelHandle* hostChannelHandleList = hostChannelHandles.data();
    std::vector<ChannelHandle> deviceChannelHandles(3);
    ChannelHandle* deviceChannelHandleList = deviceChannelHandles.data();
    std::vector<std::vector<MemHandle>> allHandles1(1);
    const s32 originalLogLevel = log_level_get_stub();
    log_level_set_stub(DLOG_INFO);
    testing::internal::CaptureStdout();
    const HcclResult createRet = myRank->BatchCreateChannels(
        COMM_ENGINE_AICPU_TS, channelDesc, 1, hcommDesc, hostChannelHandleList, deviceChannelHandleList, allHandles1);
    const std::string createLog = testing::internal::GetCapturedStdout();
    log_level_set_stub(originalLogLevel);
    EXPECT_EQ(createRet, HCCL_SUCCESS);
    EXPECT_NE(createLog.find("[LinkInfo]userRank[0] remoteRank[1]"), std::string::npos);
    EXPECT_EQ(myRank->newChannels_.size(), 1);
    EXPECT_EQ(myRank->newChannels_[0], std::make_pair(channelIdx0, RmtEp1reuseIdx0));

    EXPECT_EQ(myRank->BatchCreateSockets(channelDesc, 2, "test", hcommDesc, COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);
    std::vector<std::vector<MemHandle>> allHandles2(2);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_AICPU_TS, channelDesc, 2, hcommDesc, hostChannelHandleList, deviceChannelHandleList,
            allHandles2),
        HCCL_SUCCESS);
    EXPECT_EQ(myRank->newChannels_.size(), 1);
    EXPECT_EQ(myRank->newChannels_[0], std::make_pair(channelIdx1, RmtEp1reuseIdx1));

    EXPECT_EQ(myRank->BatchCreateSockets(channelDesc, 3, "test", hcommDesc, COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);
    std::vector<std::vector<MemHandle>> allHandles3(3);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_AICPU_TS, channelDesc, 3, hcommDesc, hostChannelHandleList, deviceChannelHandleList,
            allHandles3),
        HCCL_SUCCESS);
    EXPECT_EQ(myRank->newChannels_.size(), 1);
    EXPECT_EQ(myRank->newChannels_[0], std::make_pair(channelIdx2, RmtEp2reuseIdx0));

    // 双句柄映射断言：新建分支 channels 反查回填 host 哨兵，复用分支 deviceChannels 反查回填 device 哨兵
    EXPECT_EQ(hostChannelHandles[0], static_cast<ChannelHandle>(0xAAA));
    EXPECT_EQ(deviceChannelHandles[0], static_cast<ChannelHandle>(0xBBB));

    // 超时路径：E_AGAIN 轮询至超时（timeout 0 秒，避免 1s 空转）
    MOCKER_CPP(&Hccl::EnvSocketConfig::GetLinkTimeOut).stubs().with(mockcpp::any()).will(invoke(StubGetLinkTimeOut));
    g_linkTimeout = 0;
    MOCKER(HcommChannelGetStatus).stubs().will(returnValue(static_cast<HcommResult>(HCCL_E_AGAIN)));
    EXPECT_EQ(
        myRank->WaitForChannelsReady(channelDesc, hostChannelHandleList, 3, COMM_ENGINE_AICPU_TS), HCCL_E_TIMEOUT);

    // 归类路径：timeout 回 1s，invoke 桩填 statusList 触发 ClassifyChannelStatus
    g_linkTimeout = 1;
    // 前一步已对 HcommChannelGetStatus 打过 E_AGAIN 桩，重置后再挂 invoke 桩，避免旧桩永远命中
    MOCKER(HcommChannelGetStatus).reset();
    MOCKER(HcommChannelGetStatus).stubs().will(invoke(StubHcommChannelGetStatus));

    // 全 READY → SUCCESS
    g_stubStatusFirst = static_cast<int32_t>(hcomm::HCOMM_CHANNEL_STATUS_READY);
    g_stubStatus = static_cast<int32_t>(hcomm::HCOMM_CHANNEL_STATUS_READY);
    g_stubStatusCallCount = 0;
    EXPECT_EQ(myRank->WaitForChannelsReady(channelDesc, hostChannelHandleList, 3, COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);

    // 含 CONNECTING → 首轮 E_AGAIN 重试后成功
    g_stubStatusFirst = static_cast<int32_t>(hcomm::HCOMM_CHANNEL_STATUS_CONNECTING);
    g_stubStatus = static_cast<int32_t>(hcomm::HCOMM_CHANNEL_STATUS_READY);
    g_stubStatusCallCount = 0;
    EXPECT_EQ(myRank->WaitForChannelsReady(channelDesc, hostChannelHandleList, 3, COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);

    // 含 FAILED → E_NETWORK
    g_stubStatusFirst = static_cast<int32_t>(hcomm::HCOMM_CHANNEL_STATUS_FAILED);
    g_stubStatus = static_cast<int32_t>(hcomm::HCOMM_CHANNEL_STATUS_FAILED);
    g_stubStatusCallCount = 0;
    EXPECT_EQ(
        myRank->WaitForChannelsReady(channelDesc, hostChannelHandleList, 3, COMM_ENGINE_AICPU_TS), HCCL_E_NETWORK);

    // 含 RES_LOC_UNAVAIL → E_UNAVAIL
    g_stubStatusFirst = static_cast<int32_t>(hcomm::HCOMM_CHANNEL_STATUS_RES_LOC_UNAVAIL);
    g_stubStatus = static_cast<int32_t>(hcomm::HCOMM_CHANNEL_STATUS_RES_LOC_UNAVAIL);
    g_stubStatusCallCount = 0;
    EXPECT_EQ(
        myRank->WaitForChannelsReady(channelDesc, hostChannelHandleList, 3, COMM_ENGINE_AICPU_TS), HCCL_E_UNAVAIL);

    // 未知状态（else 分支）→ E_UNAVAIL
    g_stubStatusFirst = 999;
    g_stubStatus = 999;
    g_stubStatusCallCount = 0;
    EXPECT_EQ(
        myRank->WaitForChannelsReady(channelDesc, hostChannelHandleList, 3, COMM_ENGINE_AICPU_TS), HCCL_E_UNAVAIL);
    unsetenv("HCCL_DFS_CONFIG");
}

TEST_F(MyRankTest, Ut_CreateChannels_When_CpuEngine_UbCtp_Expect_PARAM_ERROR)
{
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MockerFuncs();

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_CTP, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_CTP, "2.0.0.0");

    HcclChannelDesc channelDesc[1]{};
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UB_CTP;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    channelDesc[0].localEndpoint = localEp;
    channelDesc[0].remoteEndpoint = rmtEp;

    ChannelHandle handles[1] = {0};
    HcclResult ret = myRank->CreateChannels(COMM_ENGINE_CPU, "test_ubctp_blocked", channelDesc, 1, handles);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(MyRankTest, Ut_CreateChannels_When_CpuEngine_UbcTp_Expect_PARAM_ERROR)
{
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MockerFuncs();

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UBC_TP, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UBC_TP, "2.0.0.0");

    HcclChannelDesc channelDesc[1]{};
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UBC_TP;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    channelDesc[0].localEndpoint = localEp;
    channelDesc[0].remoteEndpoint = rmtEp;

    ChannelHandle handles[1] = {0};
    HcclResult ret = myRank->CreateChannels(COMM_ENGINE_CPU, "test_ubctp_blocked", channelDesc, 1, handles);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// AIV引擎多通道：channel[1]的memHandleNum超限应被拦截（修复前恒校验第0个通道，该非法参数被绕过）
TEST_F(MyRankTest, Ut_CheckChannelParam_When_AivMultiChannelMemHandleNumExceedMax_Expect_PARAM_ERROR)
{
    HcclChannelDesc desc[2] = {};
    memset(desc, 0, sizeof(desc));
    desc[0].remoteRank = 1;
    desc[0].channelProtocol = COMM_PROTOCOL_HCCS;
    desc[1].remoteRank = 2;
    desc[1].channelProtocol = COMM_PROTOCOL_HCCS;
    desc[1].memHandleNum = 257; // MEM_HANDLE_NUM_MAX = 256

    HcclResult ret = myRank->CheckChannelParam(COMM_ENGINE_AIV, desc, 2);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// AIV引擎：memHandleNum非0但memHandles为空应被拦截
TEST_F(MyRankTest, Ut_CheckChannelParam_When_AivMemHandlesNullWithNonZeroNum_Expect_PARAM_ERROR)
{
    HcclChannelDesc desc[1] = {};
    memset(desc, 0, sizeof(desc));
    desc[0].remoteRank = 1;
    desc[0].channelProtocol = COMM_PROTOCOL_HCCS;
    desc[0].memHandleNum = 2; // memHandles 保持 nullptr

    HcclResult ret = myRank->CheckChannelParam(COMM_ENGINE_AIV, desc, 1);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 非AIV引擎：memHandleNum非0仅打告警不拦截，合法notifyNum应校验通过
TEST_F(MyRankTest, Ut_CheckChannelParam_When_NonAivMemHandleNumNonZero_Expect_SUCCESS)
{
    HcclChannelDesc desc[1] = {};
    memset(desc, 0, sizeof(desc));
    desc[0].remoteRank = 1;
    desc[0].channelProtocol = COMM_PROTOCOL_HCCS;
    desc[0].memHandleNum = 1;
    desc[0].notifyNum = 2;

    HcclResult ret = myRank->CheckChannelParam(COMM_ENGINE_AICPU_TS, desc, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 非AIV引擎多通道：channel[1]的notifyNum超限应被拦截（修复前恒校验第0个通道，该非法参数被绕过）
TEST_F(MyRankTest, Ut_CheckChannelParam_When_MultiChannelNotifyNumExceedMax_Expect_PARAM_ERROR)
{
    HcclChannelDesc desc[2] = {};
    memset(desc, 0, sizeof(desc));
    desc[0].remoteRank = 1;
    desc[0].channelProtocol = COMM_PROTOCOL_HCCS;
    desc[0].notifyNum = 2;
    desc[1].remoteRank = 2;
    desc[1].channelProtocol = COMM_PROTOCOL_HCCS;
    desc[1].notifyNum = 65; // NOTIFY_NUM_MAX = 64

    HcclResult ret = myRank->CheckChannelParam(COMM_ENGINE_AICPU_TS, desc, 2);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 测试Init时用户未配置展开模式时，读取环境变量配置
TEST_F(MyRankTest, Ut_Init_When_Default_Mode_Expect_Set_By_Env)
{
    setenv("HCCL_OP_EXPANSION_MODE", "CCU_SCHED", 1);
    // EnvConfig为单例，首次GetInstance时构造并Parse一次；此处强制重新Parse使setenv生效
    Hccl::EnvConfig::GetInstance().Parse();
    MOCKER_CPP(&hccl::MyRank::TryInitCcuInstance).stubs().will(returnValue(HCCL_SUCCESS));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);

    uint32_t defaultOpExpansionMode = DEFAULT_MODE;
    EXPECT_EQ(myRank->Init(cclBuffer, defaultOpExpansionMode, 2), HCCL_SUCCESS);
    EXPECT_EQ(myRank->opExpansionMode_, CCU_SCHED_MODE);
    unsetenv("HCCL_OP_EXPANSION_MODE");
}

// 测试Init时ccu驱动拉起失败回退到aicpu
// 简化后 TryInitCcuInstance 只走 OnDemand（拉起驱动 CcuInitFeature），不再调 HcommCcuInsCreateLegacy 创建实例
TEST_F(MyRankTest, Ut_Init_When_Ccu_Driver_Fail_Expect_Fallback_Aicpu)
{
    setenv("HCCL_CCU_CUSTOM_OP_MODE", "1", 1);
    MOCKER(hcomm::CcuInitFeature).stubs().will(returnValue(CcuResult::CCU_E_DRV_BUSY));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);

    uint32_t opExpansionModeMs = CCU_MS_MODE;
    EXPECT_EQ(myRank->Init(cclBuffer, opExpansionModeMs, 2), HCCL_SUCCESS);
    EXPECT_EQ(myRank->opExpansionMode_, AICPU_TS_MODE);
    unsetenv("HCCL_CCU_CUSTOM_OP_MODE");
}

// 测试Init时ccu驱动拉起返回非预期错误时失败
// 简化后 ccu instance 不在 init 时创建，MS->SCH 回退由 ReserveCcuMsCommOrFallback 在预约失败时完成；
// 此用例验证驱动返回非预期错误（CCU_E_PARA）时直接失败
TEST_F(MyRankTest, Ut_Init_When_Ccu_Driver_Unexpected_Fail_Expect_Fail)
{
    setenv("HCCL_CCU_CUSTOM_OP_MODE", "1", 1);
    MOCKER(hcomm::CcuInitFeature).stubs().will(returnValue(CcuResult::CCU_E_PARA));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);

    uint32_t opExpansionModeMs = CCU_MS_MODE;
    EXPECT_EQ(myRank->Init(cclBuffer, opExpansionModeMs, 2), HCCL_E_PARA);
    unsetenv("HCCL_CCU_CUSTOM_OP_MODE");
}

// 无UB兜底场景：level0为pcie fallback时CCU拦截，直接回退AICPU，不向设备侧拉起CCU驱动
TEST_F(MyRankTest, Ut_TryInitCcuInstanceOnDemand_When_Level0PcieFallback_Expect_FallbackAicpuWithoutDriver)
{
    myRank->opExpansionMode_ = CCU_MS_MODE;
    myRank->mainBoardType_ = Hccl::HcclMainboardId::MAINBOARD_POD; // 跳过主板类型查询，聚焦兜底拦截
    realRankGraph.SetLevel0PcieFallback(true);
    // 不mock CcuInitFeature：若未在标志处拦截而走到驱动拉起，用例将失败

    EXPECT_EQ(myRank->TryInitCcuInstanceOnDemand(), HCCL_SUCCESS);
    EXPECT_EQ(myRank->opExpansionMode_, AICPU_TS_MODE);
    EXPECT_EQ(myRank->ccuInsHandle_, static_cast<CcuInsHandle>(0));
}

// 非兜底场景：标志为false时不拦截，正常走向CCU驱动拉起（模拟驱动忙回退，与兜底拦截路径区分）
TEST_F(MyRankTest, Ut_TryInitCcuInstanceOnDemand_When_NotPcieFallback_Expect_CcuDriverCalled)
{
    myRank->opExpansionMode_ = CCU_MS_MODE;
    myRank->mainBoardType_ = Hccl::HcclMainboardId::MAINBOARD_POD;
    realRankGraph.SetLevel0PcieFallback(false);
    MOCKER(hcomm::CcuInitFeature).stubs().will(returnValue(CcuResult::CCU_E_DRV_BUSY));

    EXPECT_EQ(myRank->TryInitCcuInstanceOnDemand(), HCCL_SUCCESS);
    EXPECT_EQ(myRank->opExpansionMode_, AICPU_TS_MODE);
}

// RankGraphV2标志转发：经void*包装Hccl::RankGraph裸指针，IsLevel0PcieFallback直读被包装对象
TEST_F(MyRankTest, Ut_RankGraphV2_IsLevel0PcieFallback_When_WrappedGraphFlagToggled_Expect_ForwardValue)
{
    realRankGraph.SetLevel0PcieFallback(true);
    EXPECT_EQ(true, rankGraph->IsLevel0PcieFallback());
    realRankGraph.SetLevel0PcieFallback(false);
    EXPECT_EQ(false, rankGraph->IsLevel0PcieFallback());
}

// 测试BatchCreateChannels在资源不足时销毁新申请的channel
TEST_F(MyRankTest, St_BatchCreateChannels_When_Resource_fallback_Expect_Return_HCCL_E_UNAVAIL)
{
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommEndpointStartListen).stubs().with(mockcpp::any()).will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MOCKER(HcommChannelDestroy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MockerFuncs();

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 5, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_CTP, "1.0.0.0");
    EndpointDesc rmtEp1;
    CreateEndpointDesc(rmtEp1, COMM_PROTOCOL_UB_CTP, "2.0.0.0");
    EndpointDesc rmtEp2;
    CreateEndpointDesc(rmtEp2, COMM_PROTOCOL_UB_CTP, "3.0.0.0");

    HcclChannelDesc channelDesc[5]{};
    for (int i = 0; i < 2; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UB_CTP;
        channelDesc[i].remoteRank = 1;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = localEp;
        channelDesc[i].remoteEndpoint = rmtEp1;
    }
    for (int i = 2; i < 5; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UB_CTP;
        channelDesc[i].remoteRank = 2;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = localEp;
        channelDesc[i].remoteEndpoint = rmtEp2;
    }

    // 模拟创建到rmtEp2的第二个channel时资源不足，需要清理前三个channel
    MOCKER(HcommChannelCreate)
        .stubs()
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)))
        .then(returnValue(static_cast<int>(HCCL_SUCCESS)))
        .then(returnValue(static_cast<int>(HCCL_SUCCESS)))
        .then(returnValue(static_cast<int>(HCCL_E_UNAVAIL)));
    // 整条 HcommChannelCreate 被桩掉后 D2H/H2D 表恒空，出入口两次反查都需打桩
    MOCKER(hcomm::ChannelProcess::ResolveUserHandleToHost).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hcomm::ChannelProcess::ResolveHostHandleToDevice).stubs().will(returnValue(HCCL_SUCCESS));
    std::vector<HcommChannelDesc> hcommDesc(5);
    std::vector<ChannelHandle> hostChannelHandles(5);
    ChannelHandle* hostChannelHandleList = hostChannelHandles.data();
    std::vector<ChannelHandle> deviceChannelHandles(5);
    std::vector<std::vector<MemHandle>> allHandles_fb1(5);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_CCU, channelDesc, 5, hcommDesc, hostChannelHandleList, deviceChannelHandles.data(),
            allHandles_fb1),
        HCCL_E_UNAVAIL);
    EXPECT_EQ(myRank->newChannels_.size(), 0);

    // 获取到rmtEp1的endpointPair
    RankIdPair rankIdPair1 = std::make_pair(0, 1);
    EndpointDescPair endpointDescPair1 = std::make_pair(localEp, rmtEp1);
    RankPair* rankPair1 = nullptr;
    hcomm::EndpointPair* endpointPair1 = nullptr;
    myRank->rankPairMgr_->Get(rankIdPair1, rankPair1);
    rankPair1->GetEndpointPair(endpointDescPair1, endpointPair1);

    // 期望channelHandle被清理
    EXPECT_EQ(endpointPair1->channelHandles_.size(), 1);
    EXPECT_NE(endpointPair1->channelHandles_.find(COMM_ENGINE_CCU), endpointPair1->channelHandles_.end());
    EXPECT_EQ(endpointPair1->channelHandles_[COMM_ENGINE_CCU].size(), 0);

    // 获取到rmtEp2的endpointPair
    RankIdPair rankIdPair2 = std::make_pair(0, 2);
    EndpointDescPair endpointDescPair2 = std::make_pair(localEp, rmtEp2);
    RankPair* rankPair2 = nullptr;
    hcomm::EndpointPair* endpointPair2 = nullptr;
    myRank->rankPairMgr_->Get(rankIdPair2, rankPair2);
    rankPair2->GetEndpointPair(endpointDescPair2, endpointPair2);

    // 期望channelHandle被清理
    EXPECT_EQ(endpointPair2->channelHandles_.size(), 1);
    EXPECT_NE(endpointPair2->channelHandles_.find(COMM_ENGINE_CCU), endpointPair2->channelHandles_.end());
    EXPECT_EQ(endpointPair2->channelHandles_[COMM_ENGINE_CCU].size(), 0);
}

// 测试多次调用BatchCreateChannels，在最后一次资源不足时只销毁新申请的channel
TEST_F(MyRankTest, St_BatchCreateChannels_Multi_Times_When_fallback_Expect_Return_HCCL_E_UNAVAIL)
{
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommEndpointStartListen).stubs().with(mockcpp::any()).will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MOCKER(HcommChannelDestroy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MockerFuncs();

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 5, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_CTP, "1.0.0.0");
    EndpointDesc rmtEp1;
    CreateEndpointDesc(rmtEp1, COMM_PROTOCOL_UB_CTP, "2.0.0.0");
    EndpointDesc rmtEp2;
    CreateEndpointDesc(rmtEp2, COMM_PROTOCOL_UB_CTP, "3.0.0.0");

    HcclChannelDesc channelDesc[5]{};
    for (int i = 0; i < 2; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UB_CTP;
        channelDesc[i].remoteRank = 1;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = localEp;
        channelDesc[i].remoteEndpoint = rmtEp1;
    }
    for (int i = 2; i < 5; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UB_CTP;
        channelDesc[i].remoteRank = 2;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = localEp;
        channelDesc[i].remoteEndpoint = rmtEp2;
    }

    // 第一次调用BatchCreateChannels，创建3个channel成功
    // 第二次调用BatchCreateChannels，创建5个channel，前4个channel成功，第5个channel失败
    // 需要只清理到rmtEp2的第二个channel
    MOCKER(HcommChannelCreate)
        .stubs()
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)))    // 第一次调用，到rmtEp1的channel1成功
        .then(returnValue(static_cast<int>(HCCL_SUCCESS)))    // 第一次调用，到rmtEp1的channel2成功
        .then(returnValue(static_cast<int>(HCCL_SUCCESS)))    // 第一次调用，到rmtEp2的channel1成功
        .then(returnValue(static_cast<int>(HCCL_SUCCESS)))    // 第二次调用，到rmtEp2的channel2成功
        .then(returnValue(static_cast<int>(HCCL_E_UNAVAIL))); // 第二次调用，到rmtEp2的channel3失败
    // 整条 HcommChannelCreate 被桩掉后 D2H/H2D 表恒空，出入口两次反查都需打桩
    MOCKER(hcomm::ChannelProcess::ResolveUserHandleToHost).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hcomm::ChannelProcess::ResolveHostHandleToDevice).stubs().will(returnValue(HCCL_SUCCESS));
    std::vector<HcommChannelDesc> hcommDesc(5);
    std::vector<ChannelHandle> hostChannelHandles(5);
    ChannelHandle* hostChannelHandleList = hostChannelHandles.data();
    std::vector<ChannelHandle> deviceChannelHandles(5);
    ChannelHandle* deviceChannelHandleList = deviceChannelHandles.data();
    // 第一次调用BatchCreateChannels成功，创建3个channel
    std::vector<std::vector<MemHandle>> allHandles_multi1(3);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_CCU, channelDesc, 3, hcommDesc, hostChannelHandleList, deviceChannelHandleList,
            allHandles_multi1),
        HCCL_SUCCESS);
    EXPECT_EQ(myRank->newChannels_.size(), 3);
    u32 channelIdx0 = 0u;
    u32 channelIdx1 = 1u;
    u32 channelIdx2 = 2u;
    u32 RmtEp1reuseIdx0 = 0u;
    u32 RmtEp1reuseIdx1 = 1u;
    u32 RmtEp2reuseIdx0 = 0u;
    EXPECT_EQ(myRank->newChannels_[0], std::make_pair(channelIdx0, RmtEp1reuseIdx0));
    EXPECT_EQ(myRank->newChannels_[1], std::make_pair(channelIdx1, RmtEp1reuseIdx1));
    EXPECT_EQ(myRank->newChannels_[2], std::make_pair(channelIdx2, RmtEp2reuseIdx0));

    // 获取到rmtEp1的endpointPair
    RankIdPair rankIdPair1 = std::make_pair(0, 1);
    EndpointDescPair endpointDescPair1 = std::make_pair(localEp, rmtEp1);
    RankPair* rankPair1 = nullptr;
    hcomm::EndpointPair* endpointPair1 = nullptr;
    myRank->rankPairMgr_->Get(rankIdPair1, rankPair1);
    rankPair1->GetEndpointPair(endpointDescPair1, endpointPair1);

    // 获取到rmtEp2的endpointPair
    RankIdPair rankIdPair2 = std::make_pair(0, 2);
    EndpointDescPair endpointDescPair2 = std::make_pair(localEp, rmtEp2);
    RankPair* rankPair2 = nullptr;
    hcomm::EndpointPair* endpointPair2 = nullptr;
    myRank->rankPairMgr_->Get(rankIdPair2, rankPair2);
    rankPair2->GetEndpointPair(endpointDescPair2, endpointPair2);

    // 期望到rmtEp1的channelHandle有两个channel
    EXPECT_EQ(endpointPair1->channelHandles_.size(), 1);
    EXPECT_NE(endpointPair1->channelHandles_.find(COMM_ENGINE_CCU), endpointPair1->channelHandles_.end());
    EXPECT_EQ(endpointPair1->channelHandles_[COMM_ENGINE_CCU].size(), 2);

    // 期望到rmtEp2的channelHandle有一个channel
    EXPECT_EQ(endpointPair2->channelHandles_.size(), 1);
    EXPECT_NE(endpointPair2->channelHandles_.find(COMM_ENGINE_CCU), endpointPair2->channelHandles_.end());
    EXPECT_EQ(endpointPair2->channelHandles_[COMM_ENGINE_CCU].size(), 1);

    // 第二次调用BatchCreateChannels，创建第5个channel失败
    std::vector<std::vector<MemHandle>> allHandles_multi2(5);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_CCU, channelDesc, 5, hcommDesc, hostChannelHandleList, deviceChannelHandleList,
            allHandles_multi2),
        HCCL_E_UNAVAIL);
    EXPECT_EQ(myRank->newChannels_.size(), 0);

    // 期望到rmtEp1的channelHandle不被清理，保持两个channel
    EXPECT_EQ(endpointPair1->channelHandles_.size(), 1);
    EXPECT_NE(endpointPair1->channelHandles_.find(COMM_ENGINE_CCU), endpointPair1->channelHandles_.end());
    EXPECT_EQ(endpointPair1->channelHandles_[COMM_ENGINE_CCU].size(), 2);

    // 期望到rmtEp2的channelHandle只有第二次新创建的channel2被清理，保持第一次创建时的一个channel
    EXPECT_EQ(endpointPair2->channelHandles_.size(), 1);
    EXPECT_NE(endpointPair2->channelHandles_.find(COMM_ENGINE_CCU), endpointPair2->channelHandles_.end());
    EXPECT_EQ(endpointPair2->channelHandles_[COMM_ENGINE_CCU].size(), 1);
}

TEST_F(MyRankTest, Ut_When_ChannelGetHcclBuffer_NoBuffer_Expect_HCCL_E_INTERNAL)
{
    MOCKER(hcomm::ChannelProcess::ChannelGetRemoteMems).stubs().will(returnValue(HCCL_SUCCESS));

    ChannelHandle channel = 0x12345;
    void* buffer = nullptr;
    uint64_t size = 0;
    HcclResult ret = myRank->ChannelGetHcclBuffer(channel, &buffer, &size);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMems_Normal_Expect_SUCCESS)
{
    MOCKER(hcomm::ChannelProcess::ChannelGetRemoteMems).stubs().will(returnValue(HCCL_SUCCESS));

    ChannelHandle channel = 0x12345;
    CommMem* remoteMem = nullptr;
    char** memInfo = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = myRank->ChannelGetRemoteMems(channel, &memNum, &remoteMem, &memInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMems_RemoteMemNull_Expect_E_PTR)
{
    ChannelHandle channel = 0x12345;
    char** memInfo = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = myRank->ChannelGetRemoteMems(channel, &memNum, nullptr, &memInfo);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMems_MemInfoNull_Expect_E_PTR)
{
    ChannelHandle channel = 0x12345;
    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = myRank->ChannelGetRemoteMems(channel, &memNum, &remoteMem, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMems_MemNumNull_Expect_E_PTR)
{
    ChannelHandle channel = 0x12345;
    CommMem* remoteMem = nullptr;
    char** memInfo = nullptr;
    HcclResult ret = myRank->ChannelGetRemoteMems(channel, nullptr, &remoteMem, &memInfo);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMemsVec_Normal_Expect_SUCCESS)
{
    MOCKER(hcomm::ChannelProcess::ChannelGetRemoteMems).stubs().will(returnValue(HCCL_SUCCESS));

    ChannelHandle channel = 0x12345;
    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    std::vector<std::string> memTags;
    HcclResult ret = myRank->ChannelGetRemoteMems(channel, &memNum, &remoteMem, memTags);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMemsVec_RemoteMemNull_Expect_E_PTR)
{
    ChannelHandle channel = 0x12345;
    uint32_t memNum = 0;
    std::vector<std::string> memTags;
    HcclResult ret = myRank->ChannelGetRemoteMems(channel, &memNum, nullptr, memTags);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMemsVec_MemNumNull_Expect_E_PTR)
{
    ChannelHandle channel = 0x12345;
    CommMem* remoteMem = nullptr;
    std::vector<std::string> memTags;
    HcclResult ret = myRank->ChannelGetRemoteMems(channel, nullptr, &remoteMem, memTags);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMemsVec_MultiTagsWithNull_Expect_CopiedToVector)
{
    // 验证 memNum>0 底层返回多个 memTag（含 nullptr）时，char** 能否被正确拷贝为 vector<string>（nullptr 转空串）
    CommMem remoteMems[2] = {};
    char tag0[] = "ut_remote_tag";
    char* rawTags[2] = {tag0, nullptr};
    uint32_t expectMemNum = 2;
    CommMem* expectRemoteMem = remoteMems;
    char** expectTags = rawTags;
    MOCKER(hcomm::ChannelProcess::ChannelGetRemoteMems)
        .stubs()
        .with(mockcpp::any(), outBoundP(&expectMemNum), outBoundP(&expectRemoteMem), outBoundP(&expectTags))
        .will(returnValue(HCCL_SUCCESS));

    ChannelHandle channel = 0x12345;
    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    std::vector<std::string> memTags;
    HcclResult ret = myRank->ChannelGetRemoteMems(channel, &memNum, &remoteMem, memTags);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(memNum, 2U);
    ASSERT_EQ(memTags.size(), 2U);
    EXPECT_EQ(memTags[0], std::string("ut_remote_tag"));
    EXPECT_EQ(memTags[1], std::string());
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_UbcCtp_Sets_CommDomainQos_FromCommConfig)
{
    CommConfig commConfig("ut");
    ASSERT_EQ(commConfig.SetConfigHcclQos(5u), HCCL_SUCCESS);
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UB_CTP;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.qos, 5u);
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_UbcTp_Sets_CommDomainQos_FromCommConfig)
{
    CommConfig commConfig("ut");
    ASSERT_EQ(commConfig.SetConfigHcclQos(2u), HCCL_SUCCESS);
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UBC_TP;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.qos, 2u);
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_Uboe_Sets_CommDomainQos_FromCommConfig)
{
    CommConfig commConfig("ut");
    ASSERT_EQ(commConfig.SetConfigHcclQos(7u), HCCL_SUCCESS);
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UBOE;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.qos, 7u);
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_UbcTp_QosUnset_UsesUbQosDefault)
{
    CommConfig commConfig("ut");
    ASSERT_EQ(commConfig.SetConfigHcclQos(HCCL_COMM_QOS_CONFIG_NOT_SET), HCCL_SUCCESS);
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UBC_TP;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.qos, 4u); // 与 EnvConfig::UB_QOS_DEFAULT 一致
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_Roce_DoesNotUseUbAttrBranch)
{
    CommConfig commConfig("ut");
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_ROCE;
    in.roceAttr.retryCnt = 3u;
    in.roceAttr.retryInterval = 20u;
    in.roceAttr.tc = 8u;
    in.roceAttr.sl = 4u;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.qos, HCCL_COMM_QOS_CONFIG_NOT_SET); // RoCE 不抬默认，透传给 ApplyRoceQosCompatToSlTc
    EXPECT_EQ(out.roceAttr.retryCnt, 3u);
    EXPECT_EQ(out.roceAttr.retryInterval, 20u);
    EXPECT_EQ(out.roceAttr.tc, 8u);
    EXPECT_EQ(out.roceAttr.sl, 4u);
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_Roce_QosSet_PassesThrough)
{
    CommConfig commConfig("ut");
    ASSERT_EQ(commConfig.SetConfigHcclQos(5u), HCCL_SUCCESS);
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_ROCE;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.qos, 5u);
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_UbMem_Sets_PathMode_FromInput)
{
    CommConfig commConfig("ut");
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UB_MEM;
    in.ubMemAttr.pathMode = 2u;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.ubMemAttr.pathMode, 2u);
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_UbMem_PathModeZero_PassedThrough)
{
    CommConfig commConfig("ut");
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UB_MEM;
    in.ubMemAttr.pathMode = 0u;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.ubMemAttr.pathMode, 0u);
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_UbcCtp_DoesNotSetPathMode)
{
    // UBC_CTP 走 qos 分支，不进入 UB_MEM 的 pathMode 显式设置分支；
    // pathMode 仅经 union raws 拷贝传递，这里验证 ubMemAttr.pathMode 仍随 raws 透传
    CommConfig commConfig("ut");
    ASSERT_EQ(commConfig.SetConfigHcclQos(5u), HCCL_SUCCESS);
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UB_CTP;
    in.ubMemAttr.pathMode = 2u;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.qos, 5u);
    EXPECT_EQ(out.ubMemAttr.pathMode, 2u);
}

TEST_F(MyRankTest, Ut_ConfigSqDepthByExpansionMode_When_CCU_MSModel_WithCommConfig)
{
    HcommChannelDesc in{};
    ASSERT_EQ(myRank->config_.SetConfigSqDepth(512U), HCCL_SUCCESS);
    myRank->opExpansionMode_ = CCU_MS_MODE;
    HcclResult ret = myRank->ConfigSqDepthByExpansionMode(COMM_ENGINE_CCU, in);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(in.ubAttr.sqDepth, 128);
}

TEST_F(MyRankTest, Ut_ConfigSqDepthByExpansionMode_When_CCU_SCHEDModel_WithCommConfig)
{
    HcommChannelDesc in{};
    ASSERT_EQ(myRank->config_.SetConfigSqDepth(512U), HCCL_SUCCESS);
    myRank->opExpansionMode_ = CCU_SCHED_MODE;
    HcclResult out = myRank->ConfigSqDepthByExpansionMode(COMM_ENGINE_CCU, in);
    EXPECT_EQ(out, HCCL_SUCCESS);
    EXPECT_EQ(in.ubAttr.sqDepth, 16);
}

TEST_F(MyRankTest, Ut_ConfigSqDepthByExpansionMode_When_EngineAndProtocolVary_Expect_ScopedPropagation)
{
    struct TestCase {
        CommEngine engine;
        CommProtocol protocol;
        uint32_t configuredSqDepth;
        uint32_t initialSqDepth;
        uint32_t expectedSqDepth;
    };

    const TestCase testCases[] = {
        {COMM_ENGINE_AIV, COMM_PROTOCOL_UBC_TP, 128U, 64U, 128U},
        {COMM_ENGINE_AIV, COMM_PROTOCOL_UB_CTP, 128U, 64U, 128U},
        {COMM_ENGINE_AIV, COMM_PROTOCOL_UB_RTP, 128U, 64U, 128U},
        {COMM_ENGINE_AIV, COMM_PROTOCOL_UBC_TP, HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET, HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET,
         HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET},
        {COMM_ENGINE_AIV, COMM_PROTOCOL_UBOE, 128U, 64U, 64U},
        {COMM_ENGINE_AICPU_TS, COMM_PROTOCOL_UBC_TP, 128U, 64U, 64U},
    };

    for (const auto& testCase : testCases) {
        SCOPED_TRACE(
            testing::Message() << "engine=" << testCase.engine << ", protocol=" << testCase.protocol
                               << ", configuredSqDepth=" << testCase.configuredSqDepth);
        ASSERT_EQ(myRank->config_.SetConfigSqDepth(testCase.configuredSqDepth), HCCL_SUCCESS);
        HcommChannelDesc channelDesc{};
        channelDesc.remoteEndpoint.protocol = testCase.protocol;
        channelDesc.ubAttr.sqDepth = testCase.initialSqDepth;

        EXPECT_EQ(myRank->ConfigSqDepthByExpansionMode(testCase.engine, channelDesc), HCCL_SUCCESS);
        EXPECT_EQ(channelDesc.ubAttr.sqDepth, testCase.expectedSqDepth);
    }
}

// 测试TryInitCcuInstance在DEFAULT_MODE(0)时映射为CCU_UNUSED，提前返回成功且不拉起CCU
// 本次提交将DEFAULT_MODE从CCU_SCHED改为CCU_UNUSED，此处验证新行为
TEST_F(MyRankTest, Ut_TryInitCcuInstance_When_DefaultMode_Expect_CcuUnusedAndSuccess)
{
    myRank->opExpansionMode_ = DEFAULT_MODE;
    HcclResult ret = myRank->TryInitCcuInstance();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(myRank->ccuInsHandle_, static_cast<CcuInsHandle>(0));
}

// 测试TryInitCcuInstance在AICPU_TS_MODE(2)时映射为CCU_UNUSED，提前返回成功且不拉起CCU
TEST_F(MyRankTest, Ut_TryInitCcuInstance_When_AicpuTsMode_Expect_CcuUnusedAndSuccess)
{
    myRank->opExpansionMode_ = AICPU_TS_MODE;
    HcclResult ret = myRank->TryInitCcuInstance();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(myRank->ccuInsHandle_, static_cast<CcuInsHandle>(0));
}

// 测试HCCL_OP_EXPANSION_MODE未配置时默认值为AICPU_TS
// 本次提交将env_config默认值由CCU_SCHED改为AICPU_TS，此处直接构造EnvAlgoConfig验证新默认值
TEST_F(MyRankTest, Ut_EnvAlgoConfig_When_NoExpansionModeEnv_Expect_DefaultAicpuTs)
{
    unsetenv("HCCL_OP_EXPANSION_MODE");
    Hccl::EnvAlgoConfig algoConfig;
    algoConfig.Parse();
    EXPECT_EQ(algoConfig.GetHcclAccelerator(), Hccl::HcclAccelerator::AICPU_TS);
}

TEST_F(MyRankTest, Ut_CreateChannels_When_BatchExchangeAndCheckConsistency_Timeout_DevType950_Expect_HCCL_E_TIMEOUT)
{
    MOCKER_CPP(&hccl::MyRank::BatchCreateSockets).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::MyRank::BatchCreateChannels).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MockerFuncs();
    MOCKER_CPP(&hccl::MyRank::CollectDcaPeerInfo).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    // 绕过 NeedCheckConsistency 门控，确保走到 ExchangeInfoMgr 超时分支
    MOCKER_CPP(&hccl::MyRank::NeedCheckConsistency).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::ExchangeInfoMgr::BatchExchangeAndCheckConsistency)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_TIMEOUT));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "2.0.0.0");

    HcclChannelDesc channelDesc[1]{};
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    channelDesc[0].localEndpoint = localEp;
    channelDesc[0].remoteEndpoint = rmtEp;
    channelDesc[0].ubMemAttr.pathMode = 0;

    ChannelHandle channelHandles[1];
    HcclResult ret = myRank->CreateChannels(COMM_ENGINE_AICPU_TS, "test", channelDesc, 1, channelHandles);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);
}

// 验证DEV_TYPE_960设备同样进入通信域一致性校验交换路径（一致性校验超时返回HCCL_E_TIMEOUT）
TEST_F(MyRankTest, Ut_CreateChannels_When_BatchExchangeAndCheckConsistency_Timeout_DevType960_Expect_HCCL_E_TIMEOUT)
{
    MOCKER_CPP(&hccl::MyRank::BatchCreateSockets).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::MyRank::BatchCreateChannels).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MockerFuncs();
    MOCKER_CPP(&hccl::MyRank::CollectDcaPeerInfo).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_960)).will(returnValue(HCCL_SUCCESS));

    // 绕过 NeedCheckConsistency 门控，确保走到 ExchangeInfoMgr 超时分支
    MOCKER_CPP(&hccl::MyRank::NeedCheckConsistency).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::ExchangeInfoMgr::BatchExchangeAndCheckConsistency)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_TIMEOUT));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "2.0.0.0");

    HcclChannelDesc channelDesc[1]{};
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    channelDesc[0].localEndpoint = localEp;
    channelDesc[0].remoteEndpoint = rmtEp;
    channelDesc[0].ubMemAttr.pathMode = 0;

    ChannelHandle channelHandles[1];
    HcclResult ret = myRank->CreateChannels(COMM_ENGINE_AICPU_TS, "test", channelDesc, 1, channelHandles);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);
}

TEST_F(MyRankTest, Ut_CreateChannels_When_NullChannelDesc_Expect_EPtr)
{
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    ChannelHandle channelHandles[1];
    HcclResult ret = myRank->CreateChannels(COMM_ENGINE_AICPU_TS, "test", nullptr, 1, channelHandles);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_CreateChannels_When_NullChannelHandles_Expect_EPtr)
{
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    HcclChannelDesc channelDesc[1];
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    HcclResult ret = myRank->CreateChannels(COMM_ENGINE_AICPU_TS, "test", channelDesc, 1, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_CreateChannels_When_ZeroChannelNum_Expect_EPara)
{
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    HcclChannelDesc channelDesc[1];
    ChannelHandle channelHandles[1];
    HcclResult ret = myRank->CreateChannels(COMM_ENGINE_AICPU_TS, "test", channelDesc, 0, channelHandles);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// ==================== 复用通道追加内存的 device 侧补齐下发 ====================

namespace {
// 4 字节包头 + 2 条 28 字节条目
constexpr uint32_t PACK_BYTES_WITH_TWO_ENTRIES = 60U;

struct MemUpdatePackCall {
    ChannelHandle channel;
    void* hostPack;      // 第一轮恒为 nullptr，第二轮为段首地址
    uint32_t capacityIn; // 第一轮恒为 0，第二轮为 host 侧给的容量
};

struct H2dCopy {
    const void* src;
    std::vector<char> payload; // DeviceMem 出作用域即释放，内容只能拷下来看
};

std::unordered_map<ChannelHandle, uint32_t> g_packSizeByChannel;
std::vector<MemUpdatePackCall> g_packCalls;
std::vector<uint32_t> g_memHandleNumByChannel;
std::vector<H2dCopy> g_h2dCopies;

// 取包接口 spy：按通道给出所需长度，并把 host 缓冲铺成通道相关的填充值
HcommResult SpyChannelGetMemUpdatePack(ChannelHandle channel, void* hostPack, uint32_t* packSize)
{
    if (packSize == nullptr) {
        return HCCL_E_PTR;
    }
    g_packCalls.push_back({channel, hostPack, *packSize});

    auto it = g_packSizeByChannel.find(channel);
    const uint32_t need = (it == g_packSizeByChannel.end()) ? 0U : it->second;
    if (need == 0) {
        *packSize = 0;
        return HCCL_SUCCESS;
    }
    if (hostPack == nullptr) {
        *packSize = need;
        return HCCL_SUCCESS;
    }
    if (*packSize < need) {
        return HCCL_E_PARA;
    }
    std::memset(hostPack, static_cast<int>(0xA0U + (channel & 0x0FU)), need);
    *packSize = need;
    return HCCL_SUCCESS;
}

// 假建链输出：句柄按通道号编址，allHandles[i] 的元素数由用例指定
HcclResult StubBatchCreateChannels(
    hccl::MyRank*, CommEngine, const HcclChannelDesc*, uint32_t channelNum, std::vector<HcommChannelDesc>&,
    ChannelHandle* hostHandles, ChannelHandle* deviceHandles, std::vector<std::vector<MemHandle>>& allHandles)
{
    for (u32 i = 0; i < channelNum; ++i) {
        hostHandles[i] = static_cast<ChannelHandle>(0x1000U + i);
        deviceHandles[i] = static_cast<ChannelHandle>(0x2000U + i);
        const uint32_t handleNum = (i < g_memHandleNumByChannel.size()) ? g_memHandleNumByChannel[i] : 0U;
        allHandles[i].assign(handleNum, reinterpret_cast<MemHandle>(static_cast<uintptr_t>(0x3000U + i)));
    }
    return HCCL_SUCCESS;
}

// llt 的 HrtMalloc 返回常量地址、aclrtMemcpy 是真 memcpy，地址互踩，这里换成真 malloc
HcclResult StubLaunchHrtMalloc(void** devPtr, u64 size, bool /*level2Address*/)
{
    if (devPtr == nullptr || size == 0) {
        return HCCL_E_PTR;
    }
    *devPtr = std::malloc(size);
    return (*devPtr != nullptr) ? HCCL_SUCCESS : HCCL_E_PTR;
}

HcclResult StubLaunchHrtFree(void* devPtr)
{
    std::free(devPtr);
    return HCCL_SUCCESS;
}

HcclResult
StubLaunchHrtMemSyncCopy(void* dst, uint64_t /*destMax*/, const void* src, uint64_t count, HcclRtMemcpyKind kind)
{
    if (dst != nullptr && src != nullptr && count > 0) {
        if (kind == HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE) {
            const char* begin = static_cast<const char*>(src);
            g_h2dCopies.push_back({src, std::vector<char>(begin, begin + count)});
        }
        (void)std::memcpy(dst, src, count);
    }
    return HCCL_SUCCESS;
}

const H2dCopy* FindH2dCopyFrom(const void* src)
{
    for (const auto& copy : g_h2dCopies) {
        if (copy.src == src) {
            return &copy;
        }
    }
    return nullptr;
}

void ResetCreateChannelsRecords()
{
    g_packSizeByChannel.clear();
    g_packCalls.clear();
    g_memHandleNumByChannel.clear();
    g_h2dCopies.clear();
}

// launchNum：DFX 填表 1 次，复用通道有包可推时再加 1 次
void InstallCreateChannelsLaunchMocks(MyRank* myRank, uint32_t launchNum)
{
    MOCKER_CPP(&hccl::MyRank::BatchCreateSockets).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::MyRank::BatchCreateChannels).stubs().will(invoke(StubBatchCreateChannels));
    MOCKER_CPP(&hccl::MyRank::WaitForChannelsReady).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::MyRank::BatchExchangeAndCheckConsistency).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::MyRank::FinalizeChannelsByEngine).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommChannelGetMemUpdatePack).stubs().will(invoke(SpyChannelGetMemUpdatePack));
    // CreateChannels 在 AICPU/AICPU_TS 分支先兜底 EnsureDeviceReady 再下发 DFX register，
    // 需桩掉新接口并标记 commInit 已就绪，否则真实 EnsureDeviceReady 以假句柄执行会失败提前返回
    MOCKER(HcommChannelEnsureDeviceReady).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    myRank->callbacks_.getAicpuCommState = []() {
        return true;
    };
    MOCKER(hrtStreamCreateWithFlags).stubs().will(returnValue(static_cast<HcclResult>(HCCL_E_RUNTIME)));
    MOCKER(hrtGetHcclV2Support).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtStreamSetMode).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtMalloc).stubs().will(invoke(StubLaunchHrtMalloc));
    MOCKER_CPP(hrtFree).stubs().will(invoke(StubLaunchHrtFree));
    MOCKER_CPP(hrtMemSyncCopy).stubs().will(invoke(StubLaunchHrtMemSyncCopy));
    MOCKER(hccl::AicpuAclKernelLaunch).expects(exactly(launchNum)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hcclStreamSynchronize).stubs().will(returnValue(HCCL_SUCCESS));
}

void MakeChannelDescs(HcclChannelDesc* descs, const EndpointDesc& localEp, const EndpointDesc& rmtEp, uint32_t num)
{
    for (u32 i = 0; i < num; ++i) {
        descs[i] = HcclChannelDesc{};
        descs[i].channelProtocol = COMM_PROTOCOL_UB_MEM;
        descs[i].remoteRank = 1U;
        descs[i].notifyNum = 2U;
        descs[i].localEndpoint = localEp;
        descs[i].remoteEndpoint = rmtEp;
        descs[i].ubMemAttr.pathMode = 0;
    }
}
} // namespace

// 整批新建：地址表随建链 kernel 一起下发，本流程一条都不取
TEST_F(MyRankTest, Ut_CreateChannels_When_AllChannelsNew_Expect_NoPackQuery)
{
    ResetCreateChannelsRecords();
    MockerFuncs();
    InstallCreateChannelsLaunchMocks(myRank.get(), 1U);
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "2.0.0.0");
    HcclChannelDesc channelDescs[2];
    MakeChannelDescs(channelDescs, localEp, rmtEp, 2U);

    // BatchCreateChannels 被桩掉，本轮新建列表由用例直接给出
    myRank->newChannels_ = {{0U, 0U}, {1U, 0U}};
    g_memHandleNumByChannel = {3U, 3U};
    g_packSizeByChannel[0x1000U] = PACK_BYTES_WITH_TWO_ENTRIES;
    g_packSizeByChannel[0x1001U] = PACK_BYTES_WITH_TWO_ENTRIES;

    ChannelHandle channelHandles[2];
    EXPECT_EQ(
        myRank->CreateChannels(COMM_ENGINE_AICPU_TS, "test_mem_update_new", channelDescs, 2U, channelHandles),
        HCCL_SUCCESS);

    EXPECT_TRUE(g_packCalls.empty());
}

// 复用但本轮没追加内存（memHandleNum 未增长）：不取包、不下发
TEST_F(MyRankTest, Ut_CreateChannels_When_MemHandleNumNotGrown_Expect_NoPackQuery)
{
    ResetCreateChannelsRecords();
    MockerFuncs();
    InstallCreateChannelsLaunchMocks(myRank.get(), 1U);
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "2.0.0.0");
    HcclChannelDesc channelDescs[2];
    MakeChannelDescs(channelDescs, localEp, rmtEp, 2U);

    myRank->newChannels_.clear();
    g_memHandleNumByChannel = {1U, 1U};
    g_packSizeByChannel[0x1000U] = PACK_BYTES_WITH_TWO_ENTRIES;
    g_packSizeByChannel[0x1001U] = PACK_BYTES_WITH_TWO_ENTRIES;

    ChannelHandle channelHandles[2];
    EXPECT_EQ(
        myRank->CreateChannels(COMM_ENGINE_AICPU_TS, "test_mem_update_reuse", channelDescs, 2U, channelHandles),
        HCCL_SUCCESS);

    EXPECT_TRUE(g_packCalls.empty());
}

// 新建与复用混在同一批：两条跳过条件各自生效，只取复用那条
TEST_F(MyRankTest, Ut_CreateChannels_When_NewAndReusedMixed_Expect_OnlyReusedQueried)
{
    ResetCreateChannelsRecords();
    MockerFuncs();
    InstallCreateChannelsLaunchMocks(myRank.get(), 2U);
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "2.0.0.0");
    HcclChannelDesc channelDescs[2];
    MakeChannelDescs(channelDescs, localEp, rmtEp, 2U);

    myRank->newChannels_ = {{0U, 0U}}; // 通道0 新建，通道1 复用
    g_memHandleNumByChannel = {3U, 3U};
    g_packSizeByChannel[0x1000U] = PACK_BYTES_WITH_TWO_ENTRIES;
    g_packSizeByChannel[0x1001U] = PACK_BYTES_WITH_TWO_ENTRIES;

    ChannelHandle channelHandles[2];
    EXPECT_EQ(
        myRank->CreateChannels(COMM_ENGINE_AICPU_TS, "test_mem_update_mixed", channelDescs, 2U, channelHandles),
        HCCL_SUCCESS);

    ASSERT_EQ(g_packCalls.size(), 2U);
    EXPECT_EQ(g_packCalls[0].channel, 0x1001U);
    EXPECT_EQ(g_packCalls[0].hostPack, nullptr);
    EXPECT_EQ(g_packCalls[0].capacityIn, 0U);
    EXPECT_EQ(g_packCalls[1].channel, 0x1001U);
    ASSERT_NE(g_packCalls[1].hostPack, nullptr);
    EXPECT_EQ(g_packCalls[1].capacityIn, PACK_BYTES_WITH_TWO_ENTRIES);
}

// 复用通道的包长为 0：只问一轮，总长 0 直接返回，不发补包 kernel
TEST_F(MyRankTest, Ut_CreateChannels_When_PackEmpty_Expect_OnlyQueryOnceNoLaunch)
{
    ResetCreateChannelsRecords();
    MockerFuncs();
    InstallCreateChannelsLaunchMocks(myRank.get(), 1U);
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "2.0.0.0");
    HcclChannelDesc channelDescs[1];
    MakeChannelDescs(channelDescs, localEp, rmtEp, 1U);

    myRank->newChannels_.clear();
    g_memHandleNumByChannel = {3U};
    g_packSizeByChannel[0x1000U] = 0U;

    ChannelHandle channelHandles[1];
    EXPECT_EQ(
        myRank->CreateChannels(COMM_ENGINE_AICPU_TS, "test_mem_update_empty", channelDescs, 1U, channelHandles),
        HCCL_SUCCESS);

    ASSERT_EQ(g_packCalls.size(), 1U);
    EXPECT_EQ(g_packCalls[0].channel, 0x1000U);
    EXPECT_EQ(g_packCalls[0].hostPack, nullptr);
}

// 两个复用通道各有包：两轮取包共 4 次，两段拼进同一缓冲后一次下发
TEST_F(MyRankTest, Ut_CreateChannels_When_TwoReusedChannelsGrown_Expect_PackPushedAsOneBlob)
{
    ResetCreateChannelsRecords();
    MockerFuncs();
    InstallCreateChannelsLaunchMocks(myRank.get(), 2U);
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "2.0.0.0");
    HcclChannelDesc channelDescs[2];
    MakeChannelDescs(channelDescs, localEp, rmtEp, 2U);

    myRank->newChannels_.clear();
    g_memHandleNumByChannel = {3U, 3U};
    g_packSizeByChannel[0x1000U] = PACK_BYTES_WITH_TWO_ENTRIES;
    g_packSizeByChannel[0x1001U] = PACK_BYTES_WITH_TWO_ENTRIES;

    ChannelHandle channelHandles[2];
    EXPECT_EQ(
        myRank->CreateChannels(COMM_ENGINE_AICPU_TS, "test_mem_update_two", channelDescs, 2U, channelHandles),
        HCCL_SUCCESS);

    // 第一轮只问长度，第二轮才带缓冲
    ASSERT_EQ(g_packCalls.size(), 4U);
    EXPECT_EQ(g_packCalls[0].channel, 0x1000U);
    EXPECT_EQ(g_packCalls[0].hostPack, nullptr);
    EXPECT_EQ(g_packCalls[0].capacityIn, 0U);
    EXPECT_EQ(g_packCalls[1].channel, 0x1001U);
    EXPECT_EQ(g_packCalls[1].hostPack, nullptr);
    ASSERT_NE(g_packCalls[2].hostPack, nullptr);
    EXPECT_EQ(g_packCalls[2].channel, 0x1000U);
    EXPECT_EQ(g_packCalls[2].capacityIn, PACK_BYTES_WITH_TWO_ENTRIES);
    ASSERT_NE(g_packCalls[3].hostPack, nullptr);
    EXPECT_EQ(g_packCalls[3].channel, 0x1001U);
    EXPECT_EQ(g_packCalls[3].capacityIn, PACK_BYTES_WITH_TWO_ENTRIES);
    // 通道顺序即段顺序：通道1 的段紧接在通道0 之后
    EXPECT_EQ(
        static_cast<const char*>(g_packCalls[3].hostPack),
        static_cast<const char*>(g_packCalls[2].hostPack) + PACK_BYTES_WITH_TWO_ENTRIES);

    // 下发的就是这两段拼起来的那份缓冲
    const H2dCopy* pushed = FindH2dCopyFrom(g_packCalls[2].hostPack);
    ASSERT_NE(pushed, nullptr);
    ASSERT_EQ(pushed->payload.size(), 2U * PACK_BYTES_WITH_TWO_ENTRIES);
    std::vector<char> expectBlob(PACK_BYTES_WITH_TWO_ENTRIES, static_cast<char>(0xA0));
    expectBlob.insert(expectBlob.end(), PACK_BYTES_WITH_TWO_ENTRIES, static_cast<char>(0xA1));
    EXPECT_EQ(pushed->payload, expectBlob);
}

TEST_F(MyRankTest, Ut_ConfigSqDepthByExpansionMode_When_CcuMs_Expect_CorrectDepth)
{
    myRank->opExpansionMode_ = CCU_MS_MODE;
    HcommChannelDesc hcommDesc;
    hcommDesc.ubAttr.sqDepth = 0;
    HcclResult ret = myRank->ConfigSqDepthByExpansionMode(COMM_ENGINE_CCU, hcommDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(hcommDesc.ubAttr.sqDepth, 128u);
}

TEST_F(MyRankTest, Ut_ConfigSqDepthByExpansionMode_When_CcuSched_Expect_CorrectDepth)
{
    myRank->opExpansionMode_ = CCU_SCHED_MODE;
    HcommChannelDesc hcommDesc;
    hcommDesc.ubAttr.sqDepth = 0;
    HcclResult ret = myRank->ConfigSqDepthByExpansionMode(COMM_ENGINE_CCU, hcommDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(hcommDesc.ubAttr.sqDepth, 16u);
}

TEST_F(MyRankTest, Ut_ConfigSqDepthByExpansionMode_When_CcuInvalidMode_Expect_EInternal)
{
    myRank->opExpansionMode_ = DEFAULT_MODE;
    HcommChannelDesc hcommDesc;
    HcclResult ret = myRank->ConfigSqDepthByExpansionMode(COMM_ENGINE_CCU, hcommDesc);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(MyRankTest, Ut_ConfigSqDepthByExpansionMode_When_NonCcuEngine_Expect_Success)
{
    myRank->opExpansionMode_ = DEFAULT_MODE;
    HcommChannelDesc hcommDesc;
    HcclResult ret = myRank->ConfigSqDepthByExpansionMode(COMM_ENGINE_AICPU_TS, hcommDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试重构前后 hcommDescs 输出等价：老代码(GetTagMemoryHandles+RegisterMemory)
// 和新代码(GetAllMemory+PrepareChannelMemHandles) 输出一致
TEST_F(MyRankTest, Ut_BatchCreateChannels_HcommDescsEquivalence)
{
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    // 独立mock: 不走MockerFuncs(), 让GetAllMemory+RegisterMemory跑真实代码
    // COMM_PROTOCOL_UB_MEM → UbMemEndpoint → UbMemRegedMemMgr, 纯内存操作无需硬件资源
    MOCKER_CPP(&Hccl::SocketManager::GetConnectedSocket)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue((Hccl::Socket*)0xab));
    MOCKER(hcomm::ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hcomm::ChannelProcess::PrepareUserChannels).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hcomm::ChannelProcess::ResolveUserHandleToHost).stubs().will(returnValue(HCCL_SUCCESS));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    // 注册两块用户内存，拿到真实 rawHandle
    CommMem memA{};
    memA.addr = (void*)0xA000;
    memA.size = 4096;
    memA.type = COMM_MEM_TYPE_DEVICE;
    void* rawA = nullptr;
    EXPECT_EQ(myRank->commMems_->CommRegMem("tagA", memA, &rawA), HCCL_SUCCESS);
    ASSERT_NE(rawA, nullptr);

    CommMem memB{};
    memB.addr = (void*)0xB000;
    memB.size = 8192;
    memB.type = COMM_MEM_TYPE_DEVICE;
    void* rawB = nullptr;
    EXPECT_EQ(myRank->commMems_->CommRegMem("tagB", memB, &rawB), HCCL_SUCCESS);
    ASSERT_NE(rawB, nullptr);

    // 2 endpoint, 4 channel，用户 mem 子集各不相同
    EndpointDesc epA, epB;
    CreateEndpointDesc(epA, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    CreateEndpointDesc(epB, COMM_PROTOCOL_UB_MEM, "2.0.0.0");

    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "3.0.0.0");

    HcclChannelDesc channelDesc[4]{};
    for (int i = 0; i < 2; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UB_CTP;
        channelDesc[i].remoteRank = 1;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = epA;
        channelDesc[i].remoteEndpoint = rmtEp;
    }
    for (int i = 2; i < 4; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UB_CTP;
        channelDesc[i].remoteRank = 2;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = epB;
        channelDesc[i].remoteEndpoint = rmtEp;
    }
    // ch0: epA, [tagA]
    channelDesc[0].memHandles = &rawA;
    channelDesc[0].memHandleNum = 1;
    // ch1: epA, [tagB]
    channelDesc[1].memHandles = &rawB;
    channelDesc[1].memHandleNum = 1;
    // ch2: epB, [tagA, tagB]
    void* handlesAB[] = {rawA, rawB};
    channelDesc[2].memHandles = handlesAB;
    channelDesc[2].memHandleNum = 2;
    // ch3: epB, memHandleNum=0（仅 cclBuffer）
    channelDesc[3].memHandles = nullptr;
    channelDesc[3].memHandleNum = 0;

    std::vector<HcommChannelDesc> hcommDesc(4);
    for (u32 i = 0; i < 4; ++i) {
        hcommDesc[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDesc[i], myRank->config_);
    }
    EXPECT_EQ(myRank->BatchCreateSockets(channelDesc, 4, "test", hcommDesc, COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);

    std::vector<ChannelHandle> hostChannelHandles(4);
    ChannelHandle* hostChannelHandleList = hostChannelHandles.data();
    std::vector<ChannelHandle> deviceChannelHandles(4);
    std::vector<std::vector<MemHandle>> allHandles_eq(4);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_AICPU_TS, channelDesc, 4, hcommDesc, hostChannelHandleList, deviceChannelHandles.data(),
            allHandles_eq),
        HCCL_SUCCESS);

    // 验证 hcommDescs：输出仅取决于 channelDesc 的 memHandle 子集，与内部注册方式无关
    ASSERT_EQ(hcommDesc[0].memHandleNum, 2u); // cclBuffer + tagA
    ASSERT_EQ(hcommDesc[1].memHandleNum, 2u); // cclBuffer + tagB
    ASSERT_EQ(hcommDesc[2].memHandleNum, 3u); // cclBuffer + tagA + tagB
    ASSERT_EQ(hcommDesc[3].memHandleNum, 1u); // 仅 cclBuffer
    EXPECT_EQ(hcommDesc[0].exchangeAllMems, false);
    EXPECT_EQ(hcommDesc[1].exchangeAllMems, false);
    EXPECT_EQ(hcommDesc[2].exchangeAllMems, false);
    EXPECT_EQ(hcommDesc[3].exchangeAllMems, false);

    unsetenv("HCCL_DFS_CONFIG");
}

// 测试 PrepareChannelMemHandles 在 RegisterMemory 失败时直接返回错误
TEST_F(MyRankTest, Ut_PrepareChannelMemHandles_RegisterMemoryFail)
{
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    MOCKER_CPP(&hccl::EndpointMgr::RegisterMemory).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));

    EndpointHandle epHandle = (EndpointHandle)0x1;
    std::vector<MemHandle> memHandleVec;
    HcclResult ret = myRank->PrepareMemHandles(epHandle, nullptr, 0, memHandleVec);

    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_TRUE(memHandleVec.empty());
}

// 测试 PrepareChannelMemHandles 在 memHandles==nullptr 但 memHandleNum>0 时返回错误
TEST_F(MyRankTest, Ut_PrepareChannelMemHandles_NullMemHandles)
{
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    MockerFuncs();
    EndpointHandle epHandle = (EndpointHandle)0x1;

    // 正常：memHandleNum=0
    std::vector<MemHandle> handlesOk;
    EXPECT_EQ(myRank->PrepareMemHandles(epHandle, nullptr, 0, handlesOk), HCCL_SUCCESS);

    // 异常：memHandleNum>0 但 memHandles==nullptr，cclBuffer 不受影响
    std::vector<MemHandle> handlesBad;
    HcclResult ret = myRank->PrepareMemHandles(epHandle, nullptr, 3, handlesBad);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(handlesBad.size(), 1u); // cclBuffer 不因用户内存异常而丢失
}

// 验证 RegisterMemory 调用粒度：重构前 per-channel，重构后 per-endpoint
// 4 channel 共享 2 个 endpoint → RegisterMemory 只应调用 2 次
TEST_F(MyRankTest, Ut_BatchCreateChannels_RegisterMemoryPerEndpoint)
{
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MOCKER_CPP(&Hccl::SocketManager::GetConnectedSocket)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue((Hccl::Socket*)0xab));
    MOCKER_CPP(&hccl::CommMems::GetAllMemory).stubs().will(invoke(GetAllMemoryStub));
    MOCKER_CPP(&hccl::EndpointMgr::RegisterMemory).stubs().will(invoke(RegisterMemorySpyStub));
    MOCKER_CPP(&hccl::EndpointMgr::GetMemHandlesByTags).stubs().will(invoke(GetMemHandlesByTagsStub));
    MOCKER(hcomm::ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hcomm::ChannelProcess::PrepareUserChannels).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hcomm::ChannelProcess::ResolveUserHandleToHost).stubs().will(returnValue(HCCL_SUCCESS));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    // 2 endpoint, 4 channel
    EndpointDesc epA, epB;
    CreateEndpointDesc(epA, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    CreateEndpointDesc(epB, COMM_PROTOCOL_UB_MEM, "2.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "3.0.0.0");

    HcclChannelDesc channelDesc[4]{};
    for (int i = 0; i < 2; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UB_MEM;
        channelDesc[i].remoteRank = 1;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = epA;
        channelDesc[i].remoteEndpoint = rmtEp;
    }
    for (int i = 2; i < 4; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UB_MEM;
        channelDesc[i].remoteRank = 2;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = epB;
        channelDesc[i].remoteEndpoint = rmtEp;
    }

    std::vector<HcommChannelDesc> hcommDesc(4);
    for (u32 i = 0; i < 4; ++i) {
        hcommDesc[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDesc[i], myRank->config_);
    }
    EXPECT_EQ(myRank->BatchCreateSockets(channelDesc, 4, "test", hcommDesc, COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);

    std::vector<ChannelHandle> hostChannelHandles(4);
    std::vector<ChannelHandle> deviceChannelHandles(4);
    std::vector<std::vector<MemHandle>> allHandles(4);
    ChannelHandle* hostChannelHandleList = hostChannelHandles.data();
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_AICPU_TS, channelDesc, 4, hcommDesc, hostChannelHandleList, deviceChannelHandles.data(),
            allHandles),
        HCCL_SUCCESS);

    // 验证：2个endpoint → 2次RegisterMemory调用（非4次per-channel）
    ASSERT_EQ(s_registerMemoryCalls.size(), 2u);
    // 每次调用传入的是同一份全量数据（GetAllMemoryStub返回的1条cclBuffer）
    EXPECT_EQ(s_registerMemoryCalls[0].tags.size(), 1u);
    EXPECT_EQ(s_registerMemoryCalls[0].tags[0], "HcclBuffer");
    EXPECT_EQ(s_registerMemoryCalls[1].tags.size(), 1u);
    EXPECT_EQ(s_registerMemoryCalls[1].tags[0], "HcclBuffer");

    unsetenv("HCCL_DFS_CONFIG");
}

// 端到端生命周期：CommRegMem → CreateChannels → CommUnregMem → CreateChannels
// 验证解注册推送到全部endpoint，重进建链不含旧tag
TEST_F(MyRankTest, Ut_MemRegAndAcquireLifecycle)
{
    // ---- mock ----
    s_e2eRegCount = 0;
    s_e2eUnregCount = 0;

    // spy: HcommMemReg/Unreg 计数
    MOCKER(HcommMemReg)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(E2EMemRegStub));
    MOCKER(HcommMemUnreg).stubs().with(mockcpp::any(), mockcpp::any()).will(invoke(E2EMemUnregStub));
    MOCKER(HcommEndpointDestroy).stubs().with(mockcpp::any()).will(invoke(E2EEndpointDestroyStub));
    // 硬件
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(static_cast<u32>(0)), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceRefresh).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    // channel 跳过平台层
    MOCKER(hcomm::ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hcomm::ChannelProcess::PrepareUserChannels).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hcomm::ChannelProcess::ResolveUserHandleToHost).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hcomm::ChannelProcess::ResolveHostHandleToDevice).stubs().will(returnValue(HCCL_SUCCESS));
    // socket
    MOCKER_CPP(&Hccl::SocketManager::GetConnectedSocket)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue((Hccl::Socket*)0xab));
    // port / device
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    // CCU（Init 中 TryInitCcuInstance 永不走，但以防万一）
    MOCKER_CPP(&hccl::MyRank::TryInitCcuInstance).stubs().will(returnValue(HCCL_SUCCESS));

    // ---- init CommMems + register tags ----
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    CommMems* commMems = myRank->GetCommMems();
    ASSERT_NE(commMems, nullptr);

    auto regMem = [&](const std::string& tag, void* addr) {
        CommMem m{COMM_MEM_TYPE_DEVICE, addr, 1024};
        void* h = nullptr;
        EXPECT_EQ(commMems->CommRegMem(tag, m, &h), HCCL_SUCCESS);
    };
    regMem("A", (void*)0xA000);
    regMem("B", (void*)0xB000);
    regMem("C", (void*)0xC000);

    // ---- create 2 endpoint × 2 channel ----
    EndpointDesc epA, epB;
    CreateEndpointDesc(epA, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    CreateEndpointDesc(epB, COMM_PROTOCOL_UB_MEM, "2.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "3.0.0.0");

    HcclChannelDesc channelDesc[2]{};
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    channelDesc[0].localEndpoint = epA;
    channelDesc[0].remoteEndpoint = rmtEp;
    channelDesc[0].memHandles = nullptr;
    channelDesc[0].memHandleNum = 0;

    channelDesc[1].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[1].remoteRank = 1;
    channelDesc[1].notifyNum = 2;
    channelDesc[1].localEndpoint = epB;
    channelDesc[1].remoteEndpoint = rmtEp;
    channelDesc[1].memHandles = nullptr;
    channelDesc[1].memHandleNum = 0;

    std::vector<HcommChannelDesc> hcommDesc(2);
    for (u32 i = 0; i < 2; ++i) {
        hcommDesc[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDesc[i], myRank->config_);
    }
    EXPECT_EQ(myRank->BatchCreateSockets(channelDesc, 2, "test", hcommDesc, COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);

    // ---- Round 1: CreateChannels ----
    std::vector<ChannelHandle> hostChannelHandles1(2);
    std::vector<ChannelHandle> deviceChannelHandles1(2);
    std::vector<std::vector<MemHandle>> allHandles1(2);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_AIV, channelDesc, 2, hcommDesc, hostChannelHandles1.data(), deviceChannelHandles1.data(),
            allHandles1),
        HCCL_SUCCESS);

    // cclBuffer + A + B + C = 4 tags, 2 endpoints → 8 HcommMemReg calls
    EXPECT_EQ(s_e2eRegCount, 8);
    EXPECT_EQ(s_e2eUnregCount, 0);

    // ---- Round 1.5: 无 CommMems 变更，版本不变跳过 ----
    std::vector<HcommChannelDesc> hcommDesc1b(2);
    for (u32 i = 0; i < 2; ++i) {
        hcommDesc1b[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDesc[i], myRank->config_);
    }
    EXPECT_EQ(myRank->BatchCreateSockets(channelDesc, 2, "test", hcommDesc1b, COMM_ENGINE_AIV), HCCL_SUCCESS);
    std::vector<ChannelHandle> hostChannelHandles1b(2);
    std::vector<ChannelHandle> deviceChannelHandles1b(2);
    std::vector<std::vector<MemHandle>> allHandles1b(2);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_AIV, channelDesc, 2, hcommDesc1b, hostChannelHandles1b.data(), deviceChannelHandles1b.data(),
            allHandles1b),
        HCCL_SUCCESS);

    // 版本不变，RegisterMemory 跳过，无新增 HcommMemReg
    EXPECT_EQ(s_e2eRegCount, 8);

    // ---- UnregMemByTag("B") ----
    EXPECT_EQ(myRank->UnregMemByTag("B"), HCCL_SUCCESS);
    EXPECT_EQ(s_e2eUnregCount, 2); // epA + epB 各 1 次

    // ---- Round 2: 版本变更后重新注册 ----
    std::vector<HcommChannelDesc> hcommDesc2(2);
    for (u32 i = 0; i < 2; ++i) {
        hcommDesc2[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDesc[i], myRank->config_);
    }
    EXPECT_EQ(myRank->BatchCreateSockets(channelDesc, 2, "test", hcommDesc2, COMM_ENGINE_AIV), HCCL_SUCCESS);
    std::vector<ChannelHandle> hostChannelHandles2(2);
    std::vector<ChannelHandle> deviceChannelHandles2(2);
    std::vector<std::vector<MemHandle>> allHandles2(2);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_AIV, channelDesc, 2, hcommDesc2, hostChannelHandles2.data(), deviceChannelHandles2.data(),
            allHandles2),
        HCCL_SUCCESS);

    // 版本变更，"B"已不在 CommMems 中，"B"不会被注册
    // cclBuffer+A+C 已存在不触发 HcommMemReg，regCount 不变
    EXPECT_EQ(s_e2eRegCount, 8);

    // 在 mock 失效前主动析构 EndpointMgr，确保 TaggedMemMap 析构时
    // HcommMemUnreg / HcommEndpointDestroy 仍是 mock，不会调真函数操作假 handle
    myRank->endpointMgr_.reset();
}

TEST_F(MyRankTest, Ut_BatchCreateChannels_When_DefaultPort_EnvConfigured_Expect_UsePortRangeMin)
{
    uint32_t defaultPort = 16666;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&defaultPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MockerFuncs();

    const Hccl::EnvHostNicConfig& fakeEnvConfig = Hccl::EnvConfig::GetInstance().GetHostNicConfig();
    const_cast<Hccl::EnvHostNicConfig&>(fakeEnvConfig).hcclDeviceSocketPortRange
        = Hccl::CfgField<std::vector<Hccl::SocketPortRange>>{
            "HCCL_NPU_SOCKET_PORT_RANGE",
            {{60001, 60010}},
            [](const std::string& s) -> std::vector<Hccl::SocketPortRange> {
                return Hccl::CastSocketPortRange(s, "HCCL_NPU_SOCKET_PORT_RANGE");
            }};
    const_cast<Hccl::EnvHostNicConfig&>(fakeEnvConfig).hcclDeviceSocketPortRange.isParsed = true;

    MOCKER(HcommEndpointStartListen)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 5, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UBC_CTP, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UBC_CTP, "2.0.0.0");

    HcclChannelDesc channelDesc[1];
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UBC_CTP;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    channelDesc[0].localEndpoint = localEp;
    channelDesc[0].remoteEndpoint = rmtEp;
    channelDesc[0].memHandles = nullptr;
    channelDesc[0].memHandleNum = 0;

    MOCKER(HcommChannelCreate).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    // 整条 HcommChannelCreate 被桩掉后 D2H 表恒空，出口反查需打桩
    MOCKER(hcomm::ChannelProcess::ResolveUserHandleToHost).stubs().will(returnValue(HCCL_SUCCESS));

    std::vector<HcommChannelDesc> hcommDesc(1);
    std::vector<ChannelHandle> hostChannelHandles(1);
    std::vector<ChannelHandle> deviceChannelHandles(1);
    std::vector<std::vector<MemHandle>> allHandles(1);
    ChannelHandle* hostChannelHandleList = hostChannelHandles.data();
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_CCU, channelDesc, 1, hcommDesc, hostChannelHandleList, deviceChannelHandles.data(), allHandles),
        HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(MyRankTest, Ut_BatchCreateChannels_When_DefaultPort_EnvNotConfigured_Expect_UseDefaultPort)
{
    uint32_t defaultPort = 16666;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&defaultPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MockerFuncs();

    const Hccl::EnvHostNicConfig& fakeEnvConfig = Hccl::EnvConfig::GetInstance().GetHostNicConfig();
    const_cast<Hccl::EnvHostNicConfig&>(fakeEnvConfig).hcclDeviceSocketPortRange
        = Hccl::CfgField<std::vector<Hccl::SocketPortRange>>{
            "HCCL_NPU_SOCKET_PORT_RANGE", {}, [](const std::string& s) -> std::vector<Hccl::SocketPortRange> {
                return Hccl::CastSocketPortRange(s, "HCCL_NPU_SOCKET_PORT_RANGE");
            }};
    const_cast<Hccl::EnvHostNicConfig&>(fakeEnvConfig).hcclDeviceSocketPortRange.isParsed = true;

    MOCKER(HcommEndpointStartListen)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 5, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UBC_CTP, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UBC_CTP, "2.0.0.0");

    HcclChannelDesc channelDesc[1];
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UBC_CTP;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    channelDesc[0].localEndpoint = localEp;
    channelDesc[0].remoteEndpoint = rmtEp;
    channelDesc[0].memHandles = nullptr;
    channelDesc[0].memHandleNum = 0;

    MOCKER(HcommChannelCreate).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    // 整条 HcommChannelCreate 被桩掉后 D2H 表恒空，出口反查需打桩
    MOCKER(hcomm::ChannelProcess::ResolveUserHandleToHost).stubs().will(returnValue(HCCL_SUCCESS));

    std::vector<HcommChannelDesc> hcommDesc(1);
    std::vector<ChannelHandle> hostChannelHandles(1);
    std::vector<ChannelHandle> deviceChannelHandles(1);
    std::vector<std::vector<MemHandle>> allHandles(1);
    ChannelHandle* hostChannelHandleList = hostChannelHandles.data();
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_CCU, channelDesc, 1, hcommDesc, hostChannelHandleList, deviceChannelHandles.data(), allHandles),
        HCCL_SUCCESS);

    GlobalMockObject::verify();
}

// ==================== QueryChannels UT ====================

TEST_F(MyRankTest, Ut_When_QueryAllNotExist_Expect_AllZero)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);

    HcclChannelDesc desc[2];
    for (int i = 0; i < 2; ++i) {
        desc[i].channelProtocol = COMM_PROTOCOL_UBC_CTP;
        desc[i].remoteRank = 1;
        desc[i].notifyNum = 2;
        desc[i].localEndpoint = localEp;
        desc[i].remoteEndpoint = rmtEp;
    }
    ChannelHandle handles[2] = {0};
    EXPECT_EQ(myRank->QueryChannels(COMM_ENGINE_CCU, desc, 2, handles), HCCL_SUCCESS);
    EXPECT_EQ(handles[0], 0u);
    EXPECT_EQ(handles[1], 0u);
}

TEST_F(MyRankTest, Ut_When_QueryAllExist_Expect_HandleReturned)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);

    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_CCU, {0x100, 0x101});

    HcclChannelDesc desc[2];
    for (int i = 0; i < 2; ++i) {
        desc[i].channelProtocol = COMM_PROTOCOL_UBC_CTP;
        desc[i].remoteRank = 1;
        desc[i].notifyNum = 2;
        desc[i].localEndpoint = localEp;
        desc[i].remoteEndpoint = rmtEp;
    }
    ChannelHandle handles[2] = {0};
    EXPECT_EQ(myRank->QueryChannels(COMM_ENGINE_CCU, desc, 2, handles), HCCL_SUCCESS);
    EXPECT_EQ(handles[0], 0x100u);
    EXPECT_EQ(handles[1], 0x101u);
}

TEST_F(MyRankTest, Ut_When_QueryPartialExist_Expect_Mixed)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);

    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_CCU, {0x100});

    HcclChannelDesc desc[2];
    for (int i = 0; i < 2; ++i) {
        desc[i].channelProtocol = COMM_PROTOCOL_UBC_CTP;
        desc[i].remoteRank = 1;
        desc[i].notifyNum = 2;
        desc[i].localEndpoint = localEp;
        desc[i].remoteEndpoint = rmtEp;
    }
    ChannelHandle handles[2] = {0};
    EXPECT_EQ(myRank->QueryChannels(COMM_ENGINE_CCU, desc, 2, handles), HCCL_SUCCESS);
    EXPECT_EQ(handles[0], 0x100u);
    EXPECT_EQ(handles[1], 0u);
}

TEST_F(MyRankTest, Ut_When_QueryMultiReuseIdx_Expect_Sequential)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);

    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_CCU, {0xA0, 0xA1, 0xA2});

    HcclChannelDesc desc[3];
    for (int i = 0; i < 3; ++i) {
        desc[i].channelProtocol = COMM_PROTOCOL_UBC_CTP;
        desc[i].remoteRank = 1;
        desc[i].notifyNum = 2;
        desc[i].localEndpoint = localEp;
        desc[i].remoteEndpoint = rmtEp;
    }
    ChannelHandle handles[3] = {0};
    EXPECT_EQ(myRank->QueryChannels(COMM_ENGINE_CCU, desc, 3, handles), HCCL_SUCCESS);
    EXPECT_EQ(handles[0], 0xA0u);
    EXPECT_EQ(handles[1], 0xA1u);
    EXPECT_EQ(handles[2], 0xA2u);
}

TEST_F(MyRankTest, Ut_When_QueryHostNicToDeviceNic_Expect_Zero)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);
    rmtEp.loc.locType = ENDPOINT_LOC_TYPE_HOST;

    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_CCU, {0x100});

    HcclChannelDesc desc[1];
    desc[0].channelProtocol = COMM_PROTOCOL_UBC_CTP;
    desc[0].remoteRank = 1;
    desc[0].notifyNum = 2;
    desc[0].localEndpoint = localEp;
    desc[0].remoteEndpoint = rmtEp;
    ChannelHandle handles[1] = {0};
    EXPECT_EQ(myRank->QueryChannels(COMM_ENGINE_CCU, desc, 1, handles), HCCL_SUCCESS);
    EXPECT_EQ(handles[0], 0u);
}

TEST_F(MyRankTest, Ut_When_QueryNoSideEffect_Expect_NoNewRankPair)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);

    HcclChannelDesc desc[1];
    desc[0].channelProtocol = COMM_PROTOCOL_UBC_CTP;
    desc[0].remoteRank = 9;
    desc[0].notifyNum = 2;
    desc[0].localEndpoint = localEp;
    desc[0].remoteEndpoint = rmtEp;

    size_t sizeBefore = myRank->rankPairMgr_->rankPairMap_.size();

    ChannelHandle handles[1] = {0};
    EXPECT_EQ(myRank->QueryChannels(COMM_ENGINE_CCU, desc, 1, handles), HCCL_SUCCESS);
    EXPECT_EQ(handles[0], 0u);

    EXPECT_EQ(myRank->rankPairMgr_->rankPairMap_.size(), sizeBefore);
    RankPair* rankPair = nullptr;
    EXPECT_EQ(myRank->rankPairMgr_->Find(std::make_pair(0u, 9u), rankPair), HCCL_E_NOT_FOUND);
}

TEST_F(MyRankTest, Ut_When_QueryInvalidParam_Expect_Error)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);

    HcclChannelDesc desc[1];
    desc[0].channelProtocol = COMM_PROTOCOL_UBC_CTP;
    desc[0].remoteRank = 1;
    desc[0].notifyNum = 2;
    desc[0].localEndpoint = localEp;
    desc[0].remoteEndpoint = rmtEp;
    ChannelHandle handles[1] = {0};

    EXPECT_EQ(myRank->QueryChannels(COMM_ENGINE_CCU, nullptr, 1, handles), HCCL_E_PTR);
    EXPECT_EQ(myRank->QueryChannels(COMM_ENGINE_CCU, desc, 1, nullptr), HCCL_E_PTR);
    EXPECT_EQ(myRank->QueryChannels(COMM_ENGINE_CCU, desc, 0, handles), HCCL_E_PARA);
}

// ==================== DestroyChannels UT ====================

TEST_F(MyRankTest, Ut_When_DestroySingleHandle_Expect_Erased)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);
    MOCKER(HcommChannelDestroy).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));

    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_CCU, {0x100, 0x101});

    ChannelHandle toDestroy[1] = {0x100};
    EXPECT_EQ(myRank->DestroyChannels(toDestroy, 1), HCCL_SUCCESS);

    // 行为契约 oracle: 已销毁 handle 再次 Destroy 返回 NOT_FOUND, 存活 handle 返回 SUCCESS
    ChannelHandle destroyed[1] = {0x100};
    EXPECT_EQ(myRank->DestroyChannels(destroyed, 1), HCCL_E_NOT_FOUND);
    ChannelHandle alive[1] = {0x101};
    EXPECT_EQ(myRank->DestroyChannels(alive, 1), HCCL_SUCCESS);
}

// 暂只支持 CCU 引擎：AICPU/AIV 等 device 场景的 channel 销毁返回 NOT_SUPPORT，且不修改反查表
TEST_F(MyRankTest, Ut_When_DestroyNonCcuEngine_Expect_NotSupport)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);
    MOCKER(HcommChannelDestroy).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));

    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_AICPU, {0x100});

    ChannelHandle toDestroy[1] = {0x100};
    EXPECT_EQ(myRank->DestroyChannels(toDestroy, 1), HCCL_E_NOT_SUPPORT);

    // 行为契约 oracle: channel 未被销毁, 再次 Destroy 仍返回 NOT_SUPPORT 而非 NOT_FOUND
    EXPECT_EQ(myRank->DestroyChannels(toDestroy, 1), HCCL_E_NOT_SUPPORT);
}

// 混合引擎批量销毁：CCU 句柄正常销毁，非 CCU 句柄返回 NOT_SUPPORT 且不中断其余句柄处理
TEST_F(MyRankTest, Ut_When_DestroyMixedEngine_Expect_CcuDestroyedAndNonCcuNotSupport)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);
    MOCKER(HcommChannelDestroy).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));

    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_CCU, {0x100});
    RegisterChannels(epPair, COMM_ENGINE_AICPU, {0x200});

    ChannelHandle toDestroy[2] = {0x100, 0x200};
    EXPECT_EQ(myRank->DestroyChannels(toDestroy, 2), HCCL_E_NOT_SUPPORT);

    // 行为契约 oracle: CCU 0x100 已销毁返回 NOT_FOUND, 非 CCU 0x200 仍存活返回 NOT_SUPPORT
    ChannelHandle destroyedCcu[1] = {0x100};
    EXPECT_EQ(myRank->DestroyChannels(destroyedCcu, 1), HCCL_E_NOT_FOUND);
    ChannelHandle aliveNonCcu[1] = {0x200};
    EXPECT_EQ(myRank->DestroyChannels(aliveNonCcu, 1), HCCL_E_NOT_SUPPORT);
}

// hostNIC<->DeviceNIC(UNREUSE)通道不可复用(Query 恒返回 0), 但已创建的通道必须能被 Destroy 正常销毁
TEST_F(MyRankTest, Ut_When_DestroyHostNicToDeviceNic_Expect_Success)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);
    rmtEp.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    MOCKER(HcommChannelDestroy).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));

    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_CCU, {0x100});

    ChannelHandle toDestroy[1] = {0x100};
    EXPECT_EQ(myRank->DestroyChannels(toDestroy, 1), HCCL_SUCCESS);

    // 行为契约 oracle: 已销毁 handle 再次 Destroy 返回 NOT_FOUND
    ChannelHandle destroyed[1] = {0x100};
    EXPECT_EQ(myRank->DestroyChannels(destroyed, 1), HCCL_E_NOT_FOUND);
}

TEST_F(MyRankTest, Ut_When_DestroySameEpMiddleIdx_Expect_Continuous)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);
    MOCKER(HcommChannelDestroy).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));

    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_CCU, {0xA0, 0xA1, 0xA2});

    ChannelHandle toDestroy[1] = {0xA1};
    EXPECT_EQ(myRank->DestroyChannels(toDestroy, 1), HCCL_SUCCESS);

    // 行为契约 oracle: 已销毁的 0xA1 再次 Destroy 返回 NOT_FOUND
    ChannelHandle destroyed[1] = {0xA1};
    EXPECT_EQ(myRank->DestroyChannels(destroyed, 1), HCCL_E_NOT_FOUND);
    // 存活的 0xA0/0xA2 再次 Destroy 返回 SUCCESS
    ChannelHandle aliveA0[1] = {0xA0};
    EXPECT_EQ(myRank->DestroyChannels(aliveA0, 1), HCCL_SUCCESS);
    ChannelHandle aliveA2[1] = {0xA2};
    EXPECT_EQ(myRank->DestroyChannels(aliveA2, 1), HCCL_SUCCESS);
}

TEST_F(MyRankTest, Ut_When_DestroySameEpMultiIdx_Expect_RemainingPreserved)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);
    MOCKER(HcommChannelDestroy).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));

    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_CCU, {0xA0, 0xA1, 0xA2});

    ChannelHandle toDestroy[2] = {0xA0, 0xA2};
    EXPECT_EQ(myRank->DestroyChannels(toDestroy, 2), HCCL_SUCCESS);

    // 行为契约 oracle: 已销毁的 0xA0/0xA2 再次 Destroy 返回 NOT_FOUND
    ChannelHandle destroyed[2] = {0xA0, 0xA2};
    EXPECT_EQ(myRank->DestroyChannels(destroyed, 2), HCCL_E_NOT_FOUND);
    // 存活的 0xA1 再次 Destroy 返回 SUCCESS
    ChannelHandle alive[1] = {0xA1};
    EXPECT_EQ(myRank->DestroyChannels(alive, 1), HCCL_SUCCESS);
}

TEST_F(MyRankTest, Ut_When_DestroyCrossEngine_Expect_Independent)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);
    MOCKER(HcommChannelDestroy).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));

    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_CCU, {0xC0});
    RegisterChannels(epPair, COMM_ENGINE_AICPU_TS, {0xA0, 0xA1});

    // CCU 0xC0 销毁成功；AICPU_TS 0xA0 暂不支持销毁返回 NOT_SUPPORT，不中断其余句柄处理
    ChannelHandle toDestroy[2] = {0xC0, 0xA0};
    EXPECT_EQ(myRank->DestroyChannels(toDestroy, 2), HCCL_E_NOT_SUPPORT);

    // 行为契约 oracle: CCU 0xC0 已销毁返回 NOT_FOUND; AICPU_TS 0xA0/0xA1 未销毁返回 NOT_SUPPORT
    ChannelHandle destroyedCcu[1] = {0xC0};
    EXPECT_EQ(myRank->DestroyChannels(destroyedCcu, 1), HCCL_E_NOT_FOUND);
    ChannelHandle aliveAicpu0[1] = {0xA0};
    EXPECT_EQ(myRank->DestroyChannels(aliveAicpu0, 1), HCCL_E_NOT_SUPPORT);
    ChannelHandle aliveAicpu1[1] = {0xA1};
    EXPECT_EQ(myRank->DestroyChannels(aliveAicpu1, 1), HCCL_E_NOT_SUPPORT);
}

TEST_F(MyRankTest, Ut_When_DestroyCrossEp_Expect_Independent)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp, rmtEp2;
    SetupForQuery(cclBuffer, localEp, rmtEp);
    CreateEndpointDesc(rmtEp2, COMM_PROTOCOL_UB_MEM, "0.0.0.0");
    MOCKER(HcommChannelDestroy).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));

    auto epPair1 = GetEpPair(0, 1, localEp, rmtEp);
    auto epPair2 = GetEpPair(0, 2, localEp, rmtEp2);
    ASSERT_NE(epPair1, nullptr);
    ASSERT_NE(epPair2, nullptr);
    RegisterChannels(epPair1, COMM_ENGINE_CCU, {0x10});
    RegisterChannels(epPair2, COMM_ENGINE_CCU, {0x20});

    ChannelHandle toDestroy[2] = {0x10, 0x20};
    EXPECT_EQ(myRank->DestroyChannels(toDestroy, 2), HCCL_SUCCESS);

    // Query oracle: 两个 ep 各自销毁后返回 0
    HcclChannelDesc desc1[1];
    desc1[0].channelProtocol = COMM_PROTOCOL_UBC_CTP;
    desc1[0].remoteRank = 1;
    desc1[0].notifyNum = 2;
    desc1[0].localEndpoint = localEp;
    desc1[0].remoteEndpoint = rmtEp;
    ChannelHandle query1[1] = {0};
    EXPECT_EQ(myRank->QueryChannels(COMM_ENGINE_CCU, desc1, 1, query1), HCCL_SUCCESS);
    EXPECT_EQ(query1[0], 0u); // 0x10 已销毁

    HcclChannelDesc desc2[1];
    desc2[0].channelProtocol = COMM_PROTOCOL_UBC_CTP;
    desc2[0].remoteRank = 2;
    desc2[0].notifyNum = 2;
    desc2[0].localEndpoint = localEp;
    desc2[0].remoteEndpoint = rmtEp2;
    ChannelHandle query2[1] = {0};
    EXPECT_EQ(myRank->QueryChannels(COMM_ENGINE_CCU, desc2, 1, query2), HCCL_SUCCESS);
    EXPECT_EQ(query2[0], 0u); // 0x20 已销毁
}

TEST_F(MyRankTest, Ut_When_DestroyInvalidHandle_Expect_NotFound)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);
    MOCKER(HcommChannelDestroy).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));

    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_CCU, {0x100});

    ChannelHandle toDestroy[1] = {0xDEAD};
    EXPECT_EQ(myRank->DestroyChannels(toDestroy, 1), HCCL_E_NOT_FOUND);

    // 行为契约 oracle: 无效 handle 未影响存活 handle, 0x100 仍可销毁
    ChannelHandle alive[1] = {0x100};
    EXPECT_EQ(myRank->DestroyChannels(alive, 1), HCCL_SUCCESS);
}

// 反查索引中存在但 EndpointPair 指针为空(悬垂防护分支), 按 NOT_FOUND 处理且不影响其它句柄
TEST_F(MyRankTest, Ut_When_DestroyHandleWithNullEpPair_Expect_NotFound)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);
    MOCKER(HcommChannelDestroy).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));

    myRank->handleToEpPair_[0xBAD] = nullptr;
    ChannelHandle toDestroy[1] = {0xBAD};
    EXPECT_EQ(myRank->DestroyChannels(toDestroy, 1), HCCL_E_NOT_FOUND);

    // 行为契约 oracle: 空指针条目不影响其它存活句柄
    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_CCU, {0x100});
    ChannelHandle alive[1] = {0x100};
    EXPECT_EQ(myRank->DestroyChannels(alive, 1), HCCL_SUCCESS);
}

TEST_F(MyRankTest, Ut_When_DestroyDuplicateHandle_Expect_SecondNotFound)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);
    MOCKER(HcommChannelDestroy).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));

    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_CCU, {0x100, 0x101});

    // 第一次 0x100 销毁成功, 第二次 0x100 已不存在返回 NOT_FOUND
    ChannelHandle toDestroy[2] = {0x100, 0x100};
    EXPECT_EQ(myRank->DestroyChannels(toDestroy, 2), HCCL_E_NOT_FOUND);

    // 行为契约 oracle: 0x100 已销毁, 再次 Destroy 返回 NOT_FOUND; 0x101 仍存活, 返回 SUCCESS
    ChannelHandle destroyed[1] = {0x100};
    EXPECT_EQ(myRank->DestroyChannels(destroyed, 1), HCCL_E_NOT_FOUND);
    ChannelHandle alive[1] = {0x101};
    EXPECT_EQ(myRank->DestroyChannels(alive, 1), HCCL_SUCCESS);
}

TEST_F(MyRankTest, Ut_When_DestroyThenQuery_Expect_Zero)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);
    MOCKER(HcommChannelDestroy).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));

    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_CCU, {0x100});

    ChannelHandle toDestroy[1] = {0x100};
    EXPECT_EQ(myRank->DestroyChannels(toDestroy, 1), HCCL_SUCCESS);

    HcclChannelDesc desc[1];
    desc[0].channelProtocol = COMM_PROTOCOL_UBC_CTP;
    desc[0].remoteRank = 1;
    desc[0].notifyNum = 2;
    desc[0].localEndpoint = localEp;
    desc[0].remoteEndpoint = rmtEp;
    ChannelHandle queryHandles[1] = {0};
    EXPECT_EQ(myRank->QueryChannels(COMM_ENGINE_CCU, desc, 1, queryHandles), HCCL_SUCCESS);
    EXPECT_EQ(queryHandles[0], 0u);
}

TEST_F(MyRankTest, Ut_When_DestroyInvalidParam_Expect_Error)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);

    ChannelHandle handles[1] = {0x100};
    EXPECT_EQ(myRank->DestroyChannels(nullptr, 1), HCCL_E_PTR);
    EXPECT_EQ(myRank->DestroyChannels(handles, 0), HCCL_E_PARA);
}

TEST_F(MyRankTest, Ut_When_DestroyFails_Expect_ErrorPropagated)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);
    MOCKER(HcommChannelDestroy).stubs().will(returnValue(static_cast<int>(HCCL_E_INTERNAL)));

    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_CCU, {0x100, 0x101});

    ChannelHandle toDestroy[1] = {0x100};
    EXPECT_EQ(myRank->DestroyChannels(toDestroy, 1), HCCL_E_INTERNAL);
}

// ==================== 容错销毁 UT====================

// TC-DS-027: 3个handle(A有效、B mock HcommChannelDestroy失败、C有效), 验证返回firstErr且A/C仍被销毁
TEST_F(MyRankTest, Ut_When_DestroyPartialFail_Expect_ContinueAndReturnFirstErr)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp;
    SetupForQuery(cclBuffer, localEp, rmtEp);
    // mock: A和C的HcommChannelDestroy成功, B失败 (按调用顺序依次返回)
    MOCKER(HcommChannelDestroy)
        .stubs()
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)))
        .then(returnValue(static_cast<int>(HCCL_E_INTERNAL)))
        .then(returnValue(static_cast<int>(HCCL_SUCCESS)));

    auto epPair = GetEpPair(0, 1, localEp, rmtEp);
    ASSERT_NE(epPair, nullptr);
    RegisterChannels(epPair, COMM_ENGINE_CCU, {0xA0, 0xA1, 0xA2});

    ChannelHandle toDestroy[3] = {0xA0, 0xA1, 0xA2};
    EXPECT_EQ(myRank->DestroyChannels(toDestroy, 3), HCCL_E_INTERNAL);

    // 行为契约 oracle: 实现约定"无论 HcommChannelDestroy 成败, host 侧索引均同步清理",
    // 因此 A/B/C 的索引都已清理, 再次 Destroy 均返回 NOT_FOUND
    ChannelHandle destroyedAC[2] = {0xA0, 0xA2};
    EXPECT_EQ(myRank->DestroyChannels(destroyedAC, 2), HCCL_E_NOT_FOUND);
    ChannelHandle aliveB[1] = {0xA1};
    EXPECT_EQ(myRank->DestroyChannels(aliveB, 1), HCCL_E_NOT_FOUND);
}

// ==================== CCU回退链路 UT====================

// CCU 回滚链路: WaitForChannelsReady 返回资源不足时, CreateChannels 销毁本轮新建通道并返回 E_UNAVAIL
TEST_F(MyRankTest, Ut_When_RollbackFullChain_ResourceUnavailable_Expect_StateRestored)
{
    HcclMem cclBuffer;
    EndpointDesc localEp, rmtEp, rmtEp2;
    SetupForQuery(cclBuffer, localEp, rmtEp);
    CreateEndpointDesc(rmtEp2, COMM_PROTOCOL_UB_MEM, "0.0.0.0");
    MOCKER(HcommChannelDestroy).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    // 建链阶段 mock 成功, 连接阶段返回资源不足, 触发 CCU 回滚(真实 DestroyNewChannels 路径)
    MOCKER_CPP(&MyRank::BatchCreateSockets).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&MyRank::BatchCreateChannels).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&MyRank::WaitForChannelsReady).stubs().will(returnValue(HCCL_E_UNAVAIL));
    // CCU 引擎需要 CCU 展开模式
    myRank->opExpansionMode_ = CCU_SCHED_MODE;

    HcclChannelDesc desc[2];
    for (int i = 0; i < 2; ++i) {
        desc[i].channelProtocol = COMM_PROTOCOL_UBC_CTP;
        desc[i].remoteRank = (i == 0) ? 1 : 2;
        desc[i].notifyNum = 8;
        desc[i].localEndpoint = localEp;
        desc[i].remoteEndpoint = (i == 0) ? rmtEp : rmtEp2;
    }

    // 模拟 BatchCreateChannels 已创建两个新通道(进入 newChannels_ 与反查表)。
    // 两条 desc 使用不同 endpoint，避免触发 CCU 重复 desc 拦截。
    auto epPair1 = GetEpPair(0, 1, localEp, rmtEp);
    auto epPair2 = GetEpPair(0, 2, localEp, rmtEp2);
    ASSERT_NE(epPair1, nullptr);
    ASSERT_NE(epPair2, nullptr);
    ChannelHandle acquireHandles[2] = {0x200, 0x201};
    RegisterChannels(epPair1, COMM_ENGINE_CCU, {acquireHandles[0]});
    RegisterChannels(epPair2, COMM_ENGINE_CCU, {acquireHandles[1]});
    myRank->newChannels_.emplace_back(std::make_pair(0, 0));
    myRank->newChannels_.emplace_back(std::make_pair(1, 0));

    std::vector<ChannelHandle> outHandles(2, 0);
    EXPECT_EQ(myRank->CreateChannels(COMM_ENGINE_CCU, "test", desc, 2, outHandles.data()), HCCL_E_UNAVAIL);

    // 回滚后状态恢复: newChannels_ 清空、反查表清空、Query 全部为 0
    EXPECT_TRUE(myRank->newChannels_.empty());
    EXPECT_TRUE(myRank->handleToEpPair_.empty());
    ChannelHandle queryHandles[2] = {0, 0};
    EXPECT_EQ(myRank->QueryChannels(COMM_ENGINE_CCU, desc, 2, queryHandles), HCCL_SUCCESS);
    EXPECT_EQ(queryHandles[0], 0u);
    EXPECT_EQ(queryHandles[1], 0u);
}

TEST_F(MyRankTest, Ut_FillRoceSrcPortList_When_CommAddrToIpFail_Expect_SrcPortListNull)
{
    auto& portConfig = const_cast<Hccl::MultiQpSrcPortConfig&>(
        Hccl::EnvConfig::GetInstance().GetRdmaConfig().GetMultiQpSrcPortConfig());
    portConfig.ipPairToPorts.clear();
    portConfig.ipPairToPorts["1.0.0.0,2.0.0.0"] = {10001};

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_E_RUNTIME));
    MOCKER(hcomm::CommAddrToIpAddress).stubs().will(returnValue(HCCL_E_NOT_SUPPORT));

    HcclChannelDesc hcclDesc{};
    ASSERT_EQ(HcclChannelDescInit(&hcclDesc, 1), HCCL_SUCCESS);
    hcclDesc.channelProtocol = COMM_PROTOCOL_ROCE;
    CreateEndpointDesc(hcclDesc.localEndpoint, COMM_PROTOCOL_ROCE, "1.0.0.0");
    hcclDesc.localEndpoint.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    CreateEndpointDesc(hcclDesc.remoteEndpoint, COMM_PROTOCOL_ROCE, "2.0.0.0");

    HcommChannelDesc hcommDesc = MyRankUtils::ChannelDescHccl2Hcomm(hcclDesc, config);
    hcommDesc.exchangeAllMems = false;
    hcommDesc.roceAttr.queueNum = 4;

    EXPECT_EQ(roceDescConfigurator->FillRoceSrcPortList(hcclDesc, 0, hcommDesc), HCCL_E_INTERNAL);
    EXPECT_EQ(hcommDesc.roceAttr.srcPortList, nullptr);

    portConfig.ipPairToPorts.clear();
}

TEST_F(MyRankTest, Ut_FillRoceSrcPortList_When_ConfigAvailable_Expect_SrcPortListFilled)
{
    auto& portConfig = const_cast<Hccl::MultiQpSrcPortConfig&>(
        Hccl::EnvConfig::GetInstance().GetRdmaConfig().GetMultiQpSrcPortConfig());
    portConfig.ipPairToPorts.clear();
    portConfig.ipPairToPorts["1.0.0.0,2.0.0.0"] = {10001, 10002};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_E_RUNTIME));

    HcclChannelDesc hcclDesc{};
    ASSERT_EQ(HcclChannelDescInit(&hcclDesc, 1), HCCL_SUCCESS);
    hcclDesc.channelProtocol = COMM_PROTOCOL_ROCE;
    CreateEndpointDesc(hcclDesc.localEndpoint, COMM_PROTOCOL_ROCE, "1.0.0.0");
    hcclDesc.localEndpoint.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    CreateEndpointDesc(hcclDesc.remoteEndpoint, COMM_PROTOCOL_ROCE, "2.0.0.0");

    HcommChannelDesc hcommDesc = MyRankUtils::ChannelDescHccl2Hcomm(hcclDesc, config);
    hcommDesc.exchangeAllMems = false;
    hcommDesc.roceAttr.queueNum = 4;

    EXPECT_EQ(roceDescConfigurator->FillRoceSrcPortList(hcclDesc, 0, hcommDesc), HCCL_SUCCESS);
    ASSERT_NE(hcommDesc.roceAttr.srcPortList, nullptr);
    EXPECT_EQ(hcommDesc.roceAttr.srcPortList[0], 10001u);
    EXPECT_EQ(hcommDesc.roceAttr.srcPortList[1], 10002u);
    EXPECT_EQ(hcommDesc.roceAttr.srcPortList[2], 10001u);
    EXPECT_EQ(hcommDesc.roceAttr.srcPortList[3], 10002u);

    portConfig.ipPairToPorts.clear();
}

TEST_F(MyRankTest, Ut_FillRoceSrcPortList_When_ExchangeAllMems_Expect_SrcPortListNull)
{
    MOCKER(hrtGetDevice).expects(never());

    HcclChannelDesc hcclDesc{};
    ASSERT_EQ(HcclChannelDescInit(&hcclDesc, 1), HCCL_SUCCESS);
    hcclDesc.channelProtocol = COMM_PROTOCOL_ROCE;
    CreateEndpointDesc(hcclDesc.localEndpoint, COMM_PROTOCOL_ROCE, "1.0.0.0");
    hcclDesc.localEndpoint.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    CreateEndpointDesc(hcclDesc.remoteEndpoint, COMM_PROTOCOL_ROCE, "2.0.0.0");

    HcommChannelDesc hcommDesc = MyRankUtils::ChannelDescHccl2Hcomm(hcclDesc, config);
    hcommDesc.exchangeAllMems = true;

    EXPECT_EQ(roceDescConfigurator->FillRoceSrcPortList(hcclDesc, 0, hcommDesc), HCCL_SUCCESS);
    EXPECT_EQ(hcommDesc.roceAttr.srcPortList, nullptr);
}

TEST_F(MyRankTest, Ut_FillRoceSrcPortList_When_HostConfigAvailable_Expect_HostConfigPreferred)
{
    auto& portConfig = const_cast<Hccl::MultiQpSrcPortConfig&>(
        Hccl::EnvConfig::GetInstance().GetRdmaConfig().GetMultiQpSrcPortConfig());
    portConfig.ipPairToPorts.clear();
    portConfig.ipPairToPorts["1.0.0.0,2.0.0.0"] = {10001, 10002};
    g_hostMultiQpConfigPhyId = HOST_MULTI_QP_DEVICE_PHY_ID;
    g_hostMultiQpUdpPorts = {20001, 20002};
    MOCKER(hrtGetDevice).stubs().will(invoke(StubGetDeviceForHostMultiQp));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(invoke(StubGetDevicePhyIdForHostMultiQp));
    MOCKER_CPP(&hccl::HostMultiQpConfig::GetUdpPorts).stubs().will(invoke(StubGetHostMultiQpUdpPorts));

    HcclChannelDesc hcclDesc{};
    ASSERT_EQ(HcclChannelDescInit(&hcclDesc, 1), HCCL_SUCCESS);
    hcclDesc.channelProtocol = COMM_PROTOCOL_ROCE;
    CreateEndpointDesc(hcclDesc.localEndpoint, COMM_PROTOCOL_ROCE, "1.0.0.0");
    hcclDesc.localEndpoint.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    CreateEndpointDesc(hcclDesc.remoteEndpoint, COMM_PROTOCOL_ROCE, "2.0.0.0");

    HcommChannelDesc hcommDesc = MyRankUtils::ChannelDescHccl2Hcomm(hcclDesc, config);
    hcommDesc.exchangeAllMems = false;
    hcommDesc.roceAttr.queueNum = 5;

    ASSERT_EQ(roceDescConfigurator->FillRoceSrcPortList(hcclDesc, 0, hcommDesc), HCCL_SUCCESS);
    ASSERT_NE(hcommDesc.roceAttr.srcPortList, nullptr);
    EXPECT_EQ(
        std::vector<uint16_t>(hcommDesc.roceAttr.srcPortList, hcommDesc.roceAttr.srcPortList + 5U),
        (std::vector<uint16_t>{20001, 20002, 20001, 20002, 20001}));

    g_hostMultiQpConfigPhyId = HOST_MULTI_QP_DEVICE_PHY_ID + 1U;
    ASSERT_EQ(roceDescConfigurator->FillRoceSrcPortList(hcclDesc, 0, hcommDesc), HCCL_SUCCESS);
    ASSERT_NE(hcommDesc.roceAttr.srcPortList, nullptr);
    EXPECT_EQ(
        std::vector<uint16_t>(hcommDesc.roceAttr.srcPortList, hcommDesc.roceAttr.srcPortList + 5U),
        (std::vector<uint16_t>{10001, 10002, 10001, 10002, 10001}));

    portConfig.ipPairToPorts.clear();
}

// DPU 同 IP 场景（host 侧 CPU 引擎）：双端复用同一 IP 时，rankId 较小的一方为 SERVER
TEST_F(MyRankTest, Ut_QueryListenPort_When_SameIpLocalRankSmaller_Expect_Server)
{
    // CPU 引擎走 rankGraph rank 级查询，mock GetDevicePort 返回 16666
    uint32_t devPort = 16666;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_ROCE, "2.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_ROCE, "2.0.0.0");

    uint32_t listenPort = 0;
    HcommChannelDesc desc;
    HcclResult ret = myRank->QueryListenPort(0, 1, localEp, rmtEp, listenPort, desc, COMM_ENGINE_CPU);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(desc.role, HCOMM_SOCKET_ROLE_SERVER);
    EXPECT_EQ(listenPort, 16666);
    EXPECT_EQ(desc.port, 16666);
}

// DPU 同 IP 场景（host 侧 CPU 引擎）：双端复用同一 IP 时，rankId 较大的一方为 CLIENT
TEST_F(MyRankTest, Ut_QueryListenPort_When_SameIpLocalRankLarger_Expect_Client)
{
    // CPU 引擎走 rankGraph rank 级查询，mock GetDevicePort 返回 16666
    uint32_t devPort = 16666;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_ROCE, "2.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_ROCE, "2.0.0.0");

    uint32_t listenPort = 0;
    HcommChannelDesc desc;
    HcclResult ret = myRank->QueryListenPort(1, 0, localEp, rmtEp, listenPort, desc, COMM_ENGINE_CPU);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(desc.role, HCOMM_SOCKET_ROLE_CLIENT);
    EXPECT_EQ(listenPort, 16666);
    EXPECT_EQ(desc.port, 16666);
}

// NeedCheckConsistency: DEV_TYPE_950 且 rankConsistentState 为 0(first) 时需要校验
TEST_F(MyRankTest, Ut_NeedCheckConsistency_When_DevType950AndModeFirst_Expect_NeedCheck)
{
    // 清除 SetUp 默认的 hrtGetDeviceType(950) mock，使本用例的 mock 生效
    GlobalMockObject::verify();
    setenv("HCCL_DFS_CONFIG", "inconsistent_check:first", 1);
    Hccl::EnvConfig::GetInstance().Parse();
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    bool needCheck = false;
    EXPECT_EQ(myRank->NeedCheckConsistency(needCheck), HCCL_SUCCESS);
    EXPECT_TRUE(needCheck);

    // 恢复环境，避免影响后续用例
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    Hccl::EnvConfig::GetInstance().Parse();
}

// NeedCheckConsistency: DEV_TYPE_960 且 rankConsistentState 为 1(on) 时需要校验
TEST_F(MyRankTest, Ut_NeedCheckConsistency_When_DevType960AndModeOn_Expect_NeedCheck)
{
    // 清除 SetUp 默认的 hrtGetDeviceType(950) mock，使本用例的 mock 生效
    GlobalMockObject::verify();
    setenv("HCCL_DFS_CONFIG", "inconsistent_check:on", 1);
    Hccl::EnvConfig::GetInstance().Parse();
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_960)).will(returnValue(HCCL_SUCCESS));

    bool needCheck = false;
    EXPECT_EQ(myRank->NeedCheckConsistency(needCheck), HCCL_SUCCESS);
    EXPECT_TRUE(needCheck);

    // 恢复环境，避免影响后续用例
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    Hccl::EnvConfig::GetInstance().Parse();
}

// NeedCheckConsistency: rankConsistentState 为 off(-1) 时不需要校验
TEST_F(MyRankTest, Ut_NeedCheckConsistency_When_ModeOff_Expect_NoNeedCheck)
{
    // 清除 SetUp 默认的 hrtGetDeviceType(950) mock，使本用例的 mock 生效
    GlobalMockObject::verify();
    setenv("HCCL_DFS_CONFIG", "inconsistent_check:off", 1);
    Hccl::EnvConfig::GetInstance().Parse();
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    bool needCheck = true;
    EXPECT_EQ(myRank->NeedCheckConsistency(needCheck), HCCL_SUCCESS);
    EXPECT_FALSE(needCheck);

    // 恢复环境，避免影响后续用例
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    Hccl::EnvConfig::GetInstance().Parse();
}

// NeedCheckConsistency: 非 DEV_TYPE_950/960 设备不需要校验
TEST_F(MyRankTest, Ut_NeedCheckConsistency_When_DevTypeNotSupport_Expect_NoNeedCheck)
{
    // 清除 SetUp 默认的 hrtGetDeviceType(950) mock，使本用例的 mock 生效
    GlobalMockObject::verify();
    setenv("HCCL_DFS_CONFIG", "inconsistent_check:on", 1);
    Hccl::EnvConfig::GetInstance().Parse();
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910B)).will(returnValue(HCCL_SUCCESS));

    bool needCheck = true;
    EXPECT_EQ(myRank->NeedCheckConsistency(needCheck), HCCL_SUCCESS);
    EXPECT_FALSE(needCheck);

    // 恢复环境，避免影响后续用例
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    Hccl::EnvConfig::GetInstance().Parse();
}

// NeedCheckConsistency: 设备类型查询失败时向上返回错误码
TEST_F(MyRankTest, Ut_NeedCheckConsistency_When_GetDeviceTypeFails_Expect_Error)
{
    // 清除 SetUp 默认的 hrtGetDeviceType(950) mock，使本用例的 mock 生效
    GlobalMockObject::verify();
    MOCKER(hrtGetDeviceType).stubs().will(returnValue(HCCL_E_RUNTIME));

    bool needCheck = false;
    EXPECT_EQ(myRank->NeedCheckConsistency(needCheck), HCCL_E_RUNTIME);
    EXPECT_FALSE(needCheck);
}

// NeedCheckConsistency 返回错误时，一致性交换流程透传该错误
TEST_F(MyRankTest, Ut_BatchExchangeAndCheckConsistency_When_NeedCheckFails_Expect_Propagate)
{
    MOCKER_CPP(&hccl::MyRank::NeedCheckConsistency).stubs().will(returnValue(HCCL_E_INTERNAL));

    HcclChannelDesc channelDesc[1]{};
    channelDesc[0].remoteRank = 1;
    ChannelHandle channelHandles[1] = {0};
    EXPECT_EQ(myRank->BatchExchangeAndCheckConsistency(channelDesc, channelHandles, 1), HCCL_E_INTERNAL);
}

// needCheck 为 false 时跳过一致性交换，不调用 ExchangeInfoMgr
TEST_F(MyRankTest, Ut_BatchExchangeAndCheckConsistency_When_NeedCheckFalse_Expect_Skip)
{
    MOCKER_CPP(&hccl::MyRank::NeedCheckConsistency).stubs().with(outBound(false)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::ExchangeInfoMgr::BatchExchangeAndCheckConsistency).stubs().will(returnValue(HCCL_E_TIMEOUT));

    HcclChannelDesc channelDesc[1]{};
    channelDesc[0].remoteRank = 1;
    ChannelHandle channelHandles[1] = {0};
    EXPECT_EQ(myRank->BatchExchangeAndCheckConsistency(channelDesc, channelHandles, 1), HCCL_SUCCESS);
}

// ---------------------------------------------------------------------------
// DCA 集成直测（检视 D8）：失败对端收集 / 端口查询 / 探测下发队列的真实路径
// ---------------------------------------------------------------------------
static HcclResult StubDcaGetListenPortOk(
    hccl::MyRank* /*self*/, uint32_t /*rank*/, const Hccl::IpAddress& /*addr*/, EndpointLocType /*locType*/,
    CommEngine /*engine*/, uint32_t* port)
{
    *port = 16666;
    return HCCL_SUCCESS;
}

static HcclResult StubDcaCollectPeerInfo(
    hccl::MyRank* /*self*/, const HcclChannelDesc* /*channelDescs*/,
    const std::unordered_map<uint32_t, uint32_t>& /*failedRanks*/, std::vector<hcomm::DcaPeerInfo>& peers,
    CommEngine /*engine*/)
{
    peers.emplace_back();
    return HCCL_SUCCESS;
}

static HcclResult
StubDcaCollectLocalInfo(hccl::MyRank* /*self*/, std::vector<hcomm::LocalIpInfo>& localIps, CommEngine /*engine*/)
{
    localIps.emplace_back();
    return HCCL_SUCCESS;
}

static HcclResult StubDcaGetNetInstId(hccl::MyRank* /*self*/, uint32_t /*remoteRank*/, std::string& netInstId)
{
    netInstId = "stub_netinst";
    return HCCL_SUCCESS;
}

TEST_F(MyRankTest, Ut_CollectFailedRanks_When_MixedStatus_Expect_OnlyBadRanksCollected)
{
    // 仅 CONNECTING/FAILED/TIMEOUT 计入失败对端，READY 和资源不足状态跳过
    HcclChannelDesc descs[5]{};
    descs[0].remoteRank = 1;
    descs[1].remoteRank = 2;
    descs[2].remoteRank = 3;
    descs[3].remoteRank = 4;
    descs[4].remoteRank = 5;
    int32_t statusList[5] = {
        hcomm::HCOMM_CHANNEL_STATUS_CONNECTING, hcomm::HCOMM_CHANNEL_STATUS_READY, hcomm::HCOMM_CHANNEL_STATUS_FAILED,
        hcomm::HCOMM_CHANNEL_STATUS_TIMEOUT, hcomm::HCOMM_CHANNEL_STATUS_RES_LOC_UNAVAIL};
    std::unordered_map<uint32_t, uint32_t> failedRanks;
    myRank->CollectFailedRanks(descs, statusList, 5, failedRanks);
    ASSERT_EQ(failedRanks.size(), 3u);
    EXPECT_EQ(failedRanks[1], 0u);
    EXPECT_EQ(failedRanks[3], 2u);
    EXPECT_EQ(failedRanks[4], 3u);
}

TEST_F(MyRankTest, Ut_GetDcaPort_When_QueryOk_Expect_QueriedPort_WhenFail_Expect_DefaultPort)
{
    MOCKER_CPP(&hccl::MyRank::GetListenPortByAddr).stubs().will(invoke(StubDcaGetListenPortOk));
    EXPECT_EQ(
        myRank->GetDcaPort(1, Hccl::IpAddress("1.0.0.0"), ENDPOINT_LOC_TYPE_DEVICE, COMM_ENGINE_AICPU_TS), 16666u);
    GlobalMockObject::verify();

    MOCKER_CPP(&hccl::MyRank::GetListenPortByAddr).stubs().will(returnValue(HCCL_E_NOT_FOUND));
    // 查询失败回退默认端口：device=16666（DEFAULT_VALUE_TCPPORT）、host=60001（DEFAULT_LISTENING_PORT）
    EXPECT_EQ(
        myRank->GetDcaPort(1, Hccl::IpAddress("1.0.0.0"), ENDPOINT_LOC_TYPE_DEVICE, COMM_ENGINE_AICPU_TS), 16666u);
    EXPECT_EQ(myRank->GetDcaPort(1, Hccl::IpAddress("1.0.0.0"), ENDPOINT_LOC_TYPE_HOST, COMM_ENGINE_CPU), 60001u);
}

TEST_F(MyRankTest, Ut_CollectDcaPeerInfo_When_DeviceLoc_Expect_PeerFilled)
{
    MOCKER_CPP(&hccl::MyRank::GetDcaPort).stubs().will(returnValue(16666u));
    MOCKER_CPP(&hccl::MyRank::GetNetInstId).stubs().will(invoke(StubDcaGetNetInstId));
    HcclChannelDesc descs[1]{};
    descs[0].remoteRank = 1;
    CreateEndpointDesc(descs[0].localEndpoint, COMM_PROTOCOL_ROCE, "1.0.0.0");
    CreateEndpointDesc(descs[0].remoteEndpoint, COMM_PROTOCOL_ROCE, "3.0.0.0");
    descs[0].remoteEndpoint.loc.device.devPhyId = 3;
    std::unordered_map<uint32_t, uint32_t> failedRanks{{1, 0}};
    std::vector<hcomm::DcaPeerInfo> peers;
    EXPECT_EQ(myRank->CollectDcaPeerInfo(descs, failedRanks, peers, COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);
    ASSERT_EQ(peers.size(), 1u);
    EXPECT_EQ(peers[0].remoteRankId, 1u);
    EXPECT_EQ(peers[0].remoteDevPhyId, 3u);
    EXPECT_EQ(peers[0].remotePort, 16666u);
    EXPECT_STREQ(peers[0].remoteNetInstId, "stub_netinst");
    GlobalMockObject::verify();
}

TEST_F(MyRankTest, Ut_StartDcaDetectByFailedRanks_When_Valid_Expect_AddIpQueueCalled)
{
    // 真实路径：失败对端经 StartDcaDetectByFailedRanks 下发到 DCA 队列（AddIpQueue 桩计数递增）
    hcomm::DcaStubResetCounts();
    HcclChannelDesc descs[1]{};
    descs[0].remoteRank = 1;
    std::unordered_map<uint32_t, uint32_t> failedRanks{{1, 0}};
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::MyRank::CollectDcaPeerInfo).stubs().will(invoke(StubDcaCollectPeerInfo));
    MOCKER_CPP(&hccl::MyRank::CollectDcaLocalInfo).stubs().will(invoke(StubDcaCollectLocalInfo));
    MOCKER_CPP(&hccl::MyRank::GetNetInstId).stubs().will(invoke(StubDcaGetNetInstId));
    myRank->StartDcaDetectByFailedRanks(descs, 1, failedRanks, COMM_ENGINE_AICPU_TS);
    EXPECT_EQ(hcomm::DcaStubAddIpQueueCallCount(), 1u);
    GlobalMockObject::verify();
}

TEST_F(MyRankTest, Ut_StartDcaDetectByFailedRanks_When_GetPhyIdFails_Expect_SkipDetect)
{
    // 本端 phyId 获取失败：跳过本次探测，不下发 DCA 队列
    hcomm::DcaStubResetCounts();
    HcclChannelDesc descs[1]{};
    descs[0].remoteRank = 1;
    std::unordered_map<uint32_t, uint32_t> failedRanks{{1, 0}};
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_E_RUNTIME));
    MOCKER_CPP(&hccl::MyRank::CollectDcaPeerInfo).stubs().will(invoke(StubDcaCollectPeerInfo));
    MOCKER_CPP(&hccl::MyRank::CollectDcaLocalInfo).stubs().will(invoke(StubDcaCollectLocalInfo));
    myRank->StartDcaDetectByFailedRanks(descs, 1, failedRanks, COMM_ENGINE_AICPU_TS);
    EXPECT_EQ(hcomm::DcaStubAddIpQueueCallCount(), 0u);
    GlobalMockObject::verify();
}

TEST_F(MyRankTest, Ut_DcaStub_InitDeinit_Expect_CallCountsBalanced)
{
    // DCA 生命周期桩计数：Init/Deinit 成对到达
    hcomm::DcaStubResetCounts();
    hcomm::DetectConnectionAnomalies& dca = CollCommMgr::GetInstance().GetDetectConnectionAnomalies(0);
    dca.Init();
    dca.Init();
    dca.Deinit();
    dca.Deinit();
    EXPECT_EQ(hcomm::DcaStubInitCallCount(), 2u);
    EXPECT_EQ(hcomm::DcaStubDeinitCallCount(), 2u);
}

// ---------------------------------------------------------------------------
// DCA 覆盖补齐：CollectEndpointDescs/CollectDcaLocalInfo/GetNetInstId/触发点
// ---------------------------------------------------------------------------
static HcclResult StubDcaGetDeviceType960(DevType& devType)
{
    devType = DevType::DEV_TYPE_960; // 设备门未通过（非 950）
    return HCCL_SUCCESS;
}

static EndpointDesc MakeStubEndpointDesc(const char* ip, EndpointLocType locType, CommProtocol protocol)
{
    EndpointDesc desc{};
    desc.protocol = protocol;
    desc.loc.locType = locType;
    (void)hcomm::IpAddressToCommAddr(Hccl::IpAddress(ip), desc.commAddr);
    return desc;
}

static HcclResult StubDcaEndpointDescsDup(hccl::MyRank* /*self*/, std::vector<EndpointDesc>& descs)
{
    descs.clear();
    EndpointDesc desc = MakeStubEndpointDesc("1.0.0.0", ENDPOINT_LOC_TYPE_DEVICE, COMM_PROTOCOL_ROCE);
    descs.push_back(desc);
    descs.push_back(desc); // 同 IP 第二条：验证按 IP 去重
    return HCCL_SUCCESS;
}

static HcclResult StubDcaEndpointDescsInvalid(hccl::MyRank* /*self*/, std::vector<EndpointDesc>& descs)
{
    descs.clear();
    // 非 HOST/DEVICE → 过滤
    EndpointDesc invalidLoc = MakeStubEndpointDesc("1.0.0.0", static_cast<EndpointLocType>(0xFF), COMM_PROTOCOL_ROCE);
    descs.push_back(invalidLoc);
    // 地址解析失败（CommAddrToIpAddress mock 失败）→ 跳过
    EndpointDesc badAddr = MakeStubEndpointDesc("2.0.0.0", ENDPOINT_LOC_TYPE_DEVICE, COMM_PROTOCOL_ROCE);
    descs.push_back(badAddr);
    return HCCL_SUCCESS;
}

// 端点查询 mock 缝隙：RankGraphV2 统一委托 Hccl::IRankGraph（pImpl），与 SetUp 既有 IRankGraph mock 一致。
// 注意：不要直接 mock RankGraphV2 的虚函数入口（mockcpp JmpOnlyApiHook 会崩溃）
static uint32_t g_dcaNetLayers[3] = {0, 1, 2};
static uint32_t g_dcaTopoInsts[1] = {0};
static HcclResult StubDcaGetNetLayers(Hccl::IRankGraph* /*self*/, uint32_t** netLayers, uint32_t* netLayerNum)
{
    *netLayers = g_dcaNetLayers;
    *netLayerNum = 3;
    return HCCL_SUCCESS;
}
static HcclResult
StubDcaGetTopoInsts(Hccl::IRankGraph* /*self*/, uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum)
{
    if (netLayer == 0) {
        return HCCL_E_INTERNAL; // layer0 失败 → continue 到下一层
    }
    *topoInsts = g_dcaTopoInsts;
    *topoInstNum = 1;
    return HCCL_SUCCESS;
}
static HcclResult
StubDcaGetEndpointNum(Hccl::IRankGraph* /*self*/, uint32_t netLayer, uint32_t /*topoInstId*/, uint32_t* num)
{
    *num = (netLayer == 2) ? 0u : 3u; // layer2 数量为 0 → continue
    return HCCL_SUCCESS;
}
static HcclResult StubDcaGetEndpointDesc(
    Hccl::IRankGraph* /*self*/, uint32_t /*netLayer*/, uint32_t /*topoInstId*/, uint32_t* descNum,
    EndpointDesc* endpointDesc)
{
    if (endpointDesc != nullptr && *descNum > 0) {
        endpointDesc[0] = MakeStubEndpointDesc("9.9.9.9", ENDPOINT_LOC_TYPE_DEVICE, COMM_PROTOCOL_ROCE);
    }
    *descNum = 1; // 回写实际填充条数：申请 3 → 1，触发 resize 截断
    return HCCL_SUCCESS;
}

TEST_F(MyRankTest, Ut_CollectFailedRanks_When_ConnectingStatusThenInvalidArgs_Expect_CollectedThenGuarded)
{
    // 轮1：等待超时时仍处于 CONNECTING 的通道也计入失败对端
    HcclChannelDesc descs[2]{};
    descs[0].remoteRank = 1;
    descs[1].remoteRank = 2;
    int32_t statusList[2] = {hcomm::HCOMM_CHANNEL_STATUS_CONNECTING, hcomm::HCOMM_CHANNEL_STATUS_READY};
    std::unordered_map<uint32_t, uint32_t> failedRanks;
    myRank->CollectFailedRanks(descs, statusList, 2, failedRanks);
    ASSERT_EQ(failedRanks.size(), 1u);
    EXPECT_EQ(failedRanks[1], 0u);

    // 轮2：空指针/0 数量守卫：不收集、不崩溃
    failedRanks.clear();
    myRank->CollectFailedRanks(nullptr, statusList, 2, failedRanks);
    myRank->CollectFailedRanks(descs, nullptr, 2, failedRanks);
    myRank->CollectFailedRanks(descs, statusList, 0, failedRanks);
    EXPECT_TRUE(failedRanks.empty());
}

TEST_F(MyRankTest, Ut_CollectDcaPeerInfo_When_HostLocThenInvalidAddr_Expect_PhyIdThenSkip)
{
    MOCKER_CPP(&hccl::MyRank::GetNetInstId).stubs().will(invoke(StubDcaGetNetInstId));
    HcclChannelDesc descs[1]{};
    descs[0].remoteRank = 1;
    CreateEndpointDesc(descs[0].localEndpoint, COMM_PROTOCOL_ROCE, "1.0.0.0");
    CreateEndpointDesc(descs[0].remoteEndpoint, COMM_PROTOCOL_ROCE, "3.0.0.0");
    std::unordered_map<uint32_t, uint32_t> failedRanks{{1, 0}};

    // 轮1：host locType 走 rankGraph_->GetDeviceId 取对端 phyId
    descs[0].remoteEndpoint.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    descs[0].remoteEndpoint.loc.device.devPhyId = 7; // 哨兵：host 路径不得读该值
    MOCKER_CPP(&hccl::MyRank::GetDcaPort).stubs().will(returnValue(16666u));
    std::vector<hcomm::DcaPeerInfo> peers;
    EXPECT_EQ(myRank->CollectDcaPeerInfo(descs, failedRanks, peers, COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);
    ASSERT_EQ(peers.size(), 1u);
    EXPECT_EQ(peers[0].remoteDevPhyId, 0u); // 走 GetDeviceId，未读写回 → 0（而非 desc 的 7）
    GlobalMockObject::verify();

    // 轮2：对端地址解析失败 → 跳过该 peer
    std::vector<hcomm::DcaPeerInfo> peers2;
    MOCKER(hcomm::CommAddrToIpAddress).stubs().will(returnValue(HCCL_E_PARA));
    EXPECT_EQ(myRank->CollectDcaPeerInfo(descs, failedRanks, peers2, COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);
    EXPECT_TRUE(peers2.empty());
    GlobalMockObject::verify();
}

TEST_F(MyRankTest, Ut_StartDcaDetectByFailedRanks_When_GuardsNotPass_Expect_SkipDetect)
{
    hcomm::DcaStubResetCounts();
    HcclChannelDesc descs[1]{};
    descs[0].remoteRank = 1;
    std::unordered_map<uint32_t, uint32_t> failedRanks{{1, 0}};

    // 轮1：参数非法（descs 空 / channelNum 0）
    myRank->StartDcaDetectByFailedRanks(nullptr, 1, failedRanks, COMM_ENGINE_AICPU_TS);
    myRank->StartDcaDetectByFailedRanks(descs, 0, failedRanks, COMM_ENGINE_AICPU_TS);
    EXPECT_EQ(hcomm::DcaStubAddIpQueueCallCount(), 0u);

    // 轮2：failedRanks 为空
    std::unordered_map<uint32_t, uint32_t> emptyRanks;
    myRank->StartDcaDetectByFailedRanks(descs, 1, emptyRanks, COMM_ENGINE_AICPU_TS);
    EXPECT_EQ(hcomm::DcaStubAddIpQueueCallCount(), 0u);

    // 轮3：设备门未通过（非 950）。
    // 先 verify 清掉 SetUp 注册的默认 hrtGetDeviceType 桩，再挂本轮的 960 桩（同函数后注册 MOCKER 不覆盖先注册）
    GlobalMockObject::verify();
    MOCKER(hrtGetDeviceType).stubs().will(invoke(StubDcaGetDeviceType960));
    myRank->StartDcaDetectByFailedRanks(descs, 1, failedRanks, COMM_ENGINE_AICPU_TS);
    EXPECT_EQ(hcomm::DcaStubAddIpQueueCallCount(), 0u);
    GlobalMockObject::verify();
}

TEST_F(MyRankTest, Ut_StartDcaDetectByFailedRanks_When_CollectInfoNotUsable_Expect_SkipDetect)
{
    HcclChannelDesc descs[1]{};
    descs[0].remoteRank = 1;
    std::unordered_map<uint32_t, uint32_t> failedRanks{{1, 0}};

    // 轮1：CollectDcaPeerInfo 返回空 peers → 跳过
    hcomm::DcaStubResetCounts();
    MOCKER_CPP(&hccl::MyRank::CollectDcaPeerInfo).stubs().will(returnValue(HCCL_SUCCESS));
    myRank->StartDcaDetectByFailedRanks(descs, 1, failedRanks, COMM_ENGINE_AICPU_TS);
    EXPECT_EQ(hcomm::DcaStubAddIpQueueCallCount(), 0u);
    GlobalMockObject::verify();

    // 轮2：CollectDcaLocalInfo 失败 → 跳过
    MOCKER_CPP(&hccl::MyRank::CollectDcaPeerInfo).stubs().will(invoke(StubDcaCollectPeerInfo));
    MOCKER_CPP(&hccl::MyRank::CollectDcaLocalInfo).stubs().will(returnValue(HCCL_E_INTERNAL));
    myRank->StartDcaDetectByFailedRanks(descs, 1, failedRanks, COMM_ENGINE_AICPU_TS);
    EXPECT_EQ(hcomm::DcaStubAddIpQueueCallCount(), 0u);
    GlobalMockObject::verify();
}

TEST_F(MyRankTest, Ut_GetNetInstId_When_RankGraphNullThenNotV2_Expect_ErrorCodes)
{
    auto* savedRankGraph = myRank->rankGraph_;
    std::string netInstId;

    // 轮1：rankGraph_ 为空 → E_PTR
    myRank->rankGraph_ = nullptr;
    EXPECT_EQ(myRank->GetNetInstId(1, netInstId), HCCL_E_PTR);

    // 轮2：非 RankGraphV2 实例 → E_NOT_SUPPORT
    hccl::RankGraphV1 nonV2Graph;
    myRank->rankGraph_ = &nonV2Graph;
    EXPECT_EQ(myRank->GetNetInstId(1, netInstId), HCCL_E_NOT_SUPPORT);
    EXPECT_TRUE(netInstId.empty());
    myRank->rankGraph_ = savedRankGraph;
}

TEST_F(MyRankTest, Ut_RankGraphV2_GetNetInstId_When_NullThenNormal_Expect_NotFoundThenNetInstId)
{
    auto* v2 = dynamic_cast<RankGraphV2*>(rankGraph.get());
    ASSERT_NE(v2, nullptr);
    std::string netInstId;

    // 轮1：netLayer0 无 netInstance → E_NOT_FOUND
    MOCKER_CPP(
        &Hccl::RankGraph::GetNetInstanceByRankId,
        const Hccl::NetInstance* (Hccl::RankGraph::*)(u32, Hccl::RankId) const)
        .stubs()
        .will(returnValue(static_cast<const Hccl::NetInstance*>(nullptr)));
    EXPECT_EQ(v2->GetNetInstId(0, netInstId), HCCL_E_NOT_FOUND);
    EXPECT_TRUE(netInstId.empty());
    GlobalMockObject::verify();

    // 轮2：正常返回 netLayer0 的 netInstId
    Hccl::NetInstance* fakeInstance = reinterpret_cast<Hccl::NetInstance*>(0x1);
    MOCKER_CPP(
        &Hccl::RankGraph::GetNetInstanceByRankId,
        const Hccl::NetInstance* (Hccl::RankGraph::*)(u32, Hccl::RankId) const)
        .stubs()
        .will(returnValue(static_cast<const Hccl::NetInstance*>(fakeInstance)));
    MOCKER_CPP(&Hccl::NetInstance::GetNetInstId).stubs().will(returnValue(std::string("S1")));
    EXPECT_EQ(v2->GetNetInstId(0, netInstId), HCCL_SUCCESS);
    EXPECT_EQ(netInstId, "S1");
    GlobalMockObject::verify();
}

TEST_F(MyRankTest, Ut_CollectEndpointDescs_When_EarlyReturn_Expect_ErrorMapping)
{
    auto* savedRankGraph = myRank->rankGraph_;
    std::vector<EndpointDesc> descs;

    // 轮1：非 RankGraphV2 → E_NOT_SUPPORT
    hccl::RankGraphV1 nonV2Graph;
    myRank->rankGraph_ = &nonV2Graph;
    EXPECT_EQ(myRank->CollectEndpointDescs(descs), HCCL_E_NOT_SUPPORT);
    EXPECT_TRUE(descs.empty());

    // 轮2：GetNetLayers 失败 → E_NOT_FOUND（mock pImpl 委托层 IRankGraph）
    myRank->rankGraph_ = savedRankGraph;
    MOCKER_CPP(&Hccl::IRankGraph::GetNetLayers).stubs().will(returnValue(HCCL_E_INTERNAL));
    EXPECT_EQ(myRank->CollectEndpointDescs(descs), HCCL_E_NOT_FOUND);
    myRank->rankGraph_ = savedRankGraph;
    GlobalMockObject::verify();
}

TEST_F(MyRankTest, Ut_CollectEndpointDescs_When_LayerFailsAndDescTruncated_Expect_SkipAndResize)
{
    // 三层遍历：layer0 topoInst 失败跳过、layer1 正常、layer2 端点数为 0 跳过；
    // 且 GetEndpointDesc 回写更小 descNum 时按实际条数截断（mock pImpl 委托层 IRankGraph）
    MOCKER_CPP(&Hccl::IRankGraph::GetNetLayers).stubs().will(invoke(StubDcaGetNetLayers));
    MOCKER_CPP(&Hccl::IRankGraph::GetTopoInstsByLayer).stubs().will(invoke(StubDcaGetTopoInsts));
    MOCKER_CPP(&Hccl::IRankGraph::GetEndpointNum).stubs().will(invoke(StubDcaGetEndpointNum));
    MOCKER_CPP(&Hccl::IRankGraph::GetEndpointDesc).stubs().will(invoke(StubDcaGetEndpointDesc));

    std::vector<EndpointDesc> descs;
    EXPECT_EQ(myRank->CollectEndpointDescs(descs), HCCL_SUCCESS);
    ASSERT_EQ(descs.size(), 1u); // 3 → 1 截断，且只来自 layer1
    Hccl::IpAddress ip;
    EXPECT_EQ(hcomm::CommAddrToIpAddress(descs[0].commAddr, ip), HCCL_SUCCESS);
    EXPECT_EQ(ip.GetIpStr(), "9.9.9.9");
    GlobalMockObject::verify();
}

TEST_F(MyRankTest, Ut_CollectDcaLocalInfo_When_DedupThenInvalidSkip_Expect_SingleThenEmpty)
{
    // 轮1：同 IP 的多个 endpoint 只保留一条，端口经 GetDcaPort 获取
    MOCKER_CPP(&hccl::MyRank::CollectEndpointDescs).stubs().will(invoke(StubDcaEndpointDescsDup));
    MOCKER_CPP(&hccl::MyRank::GetDcaPort).stubs().will(returnValue(16666u));
    std::vector<hcomm::LocalIpInfo> localIps;
    EXPECT_EQ(myRank->CollectDcaLocalInfo(localIps, COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);
    ASSERT_EQ(localIps.size(), 1u);
    EXPECT_EQ(localIps[0].ip.GetIpStr(), "1.0.0.0");
    EXPECT_EQ(localIps[0].port, 16666u);
    GlobalMockObject::verify();

    // 轮2：非法 locType 与地址解析失败均跳过 → 空结果
    MOCKER_CPP(&hccl::MyRank::CollectEndpointDescs).stubs().will(invoke(StubDcaEndpointDescsInvalid));
    MOCKER(hcomm::CommAddrToIpAddress).stubs().will(returnValue(HCCL_E_PARA));
    std::vector<hcomm::LocalIpInfo> localIps2;
    EXPECT_EQ(myRank->CollectDcaLocalInfo(localIps2, COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);
    EXPECT_TRUE(localIps2.empty());
    GlobalMockObject::verify();
}

TEST_F(MyRankTest, Ut_WaitForChannelsReady_When_TimeoutAndStatusFail_Expect_DcaDetectTriggered)
{
    // 两个触发调用点各占一轮：等待超时 / 状态异常返回
    MOCKER_CPP(&hccl::MyRank::CollectDcaPeerInfo).stubs().will(invoke(StubDcaCollectPeerInfo));
    MOCKER_CPP(&hccl::MyRank::CollectDcaLocalInfo).stubs().will(invoke(StubDcaCollectLocalInfo));
    MOCKER_CPP(&hccl::MyRank::GetNetInstId).stubs().will(invoke(StubDcaGetNetInstId));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));

    HcclChannelDesc channelDesc[1]{};
    channelDesc[0].remoteRank = 1;
    ChannelHandle hostChannelHandleList[1] = {0};

    // 轮1：HcommChannelGetStatus 持续 E_AGAIN + LinkTimeOut=1s → 等待超时调用点
    hcomm::DcaStubResetCounts();
    MOCKER(HcommChannelGetStatus).stubs().will(returnValue(static_cast<HcommResult>(HCCL_E_AGAIN)));
    MOCKER_CPP(&Hccl::EnvSocketConfig::GetLinkTimeOut).stubs().with(mockcpp::any()).will(returnValue((s32)(1)));
    EXPECT_EQ(
        myRank->WaitForChannelsReady(channelDesc, hostChannelHandleList, 1, COMM_ENGINE_AICPU_TS), HCCL_E_TIMEOUT);
    EXPECT_EQ(hcomm::DcaStubAddIpQueueCallCount(), 1u);

    // 轮2：HcommChannelGetStatus 直接失败 → 状态异常返回调用点
    hcomm::DcaStubResetCounts();
    MOCKER(HcommChannelGetStatus).reset();
    MOCKER(HcommChannelGetStatus).stubs().will(returnValue(static_cast<HcommResult>(HCCL_E_TIMEOUT)));
    EXPECT_EQ(
        myRank->WaitForChannelsReady(channelDesc, hostChannelHandleList, 1, COMM_ENGINE_AICPU_TS), HCCL_E_TIMEOUT);
    EXPECT_EQ(hcomm::DcaStubAddIpQueueCallCount(), 1u);
    GlobalMockObject::verify();
}

TEST_F(MyRankTest, Ut_CreateChannels_When_ExchangeReturnsTcpConnect_Expect_DcaDetectTriggered)
{
    // 一致性校验返回 E_TCP_CONNECT 时同样触发 DCA（现有仅覆盖 E_TIMEOUT）
    MOCKER_CPP(&hccl::MyRank::BatchCreateSockets).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::MyRank::BatchCreateChannels).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MockerFuncs();
    MOCKER_CPP(&hccl::MyRank::CollectDcaPeerInfo).stubs().will(invoke(StubDcaCollectPeerInfo));
    MOCKER_CPP(&hccl::MyRank::CollectDcaLocalInfo).stubs().will(invoke(StubDcaCollectLocalInfo));
    MOCKER_CPP(&hccl::MyRank::GetNetInstId).stubs().will(invoke(StubDcaGetNetInstId));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::MyRank::NeedCheckConsistency).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::ExchangeInfoMgr::BatchExchangeAndCheckConsistency)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_TCP_CONNECT));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "2.0.0.0");

    HcclChannelDesc channelDesc[1]{};
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    channelDesc[0].localEndpoint = localEp;
    channelDesc[0].remoteEndpoint = rmtEp;
    channelDesc[0].ubMemAttr.pathMode = 0;

    hcomm::DcaStubResetCounts();
    ChannelHandle channelHandles[1];
    HcclResult ret = myRank->CreateChannels(COMM_ENGINE_AICPU_TS, "test", channelDesc, 1, channelHandles);
    EXPECT_EQ(ret, HCCL_E_TCP_CONNECT);
    EXPECT_EQ(hcomm::DcaStubAddIpQueueCallCount(), 1u);
    GlobalMockObject::verify();
}
