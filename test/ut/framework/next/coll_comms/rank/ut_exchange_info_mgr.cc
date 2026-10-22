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
#include <algorithm>
#include <unordered_set>
#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "rank_graph_interface.h"
#include "rank_graph_v2.h"
#include "hcomm_c_adpt.h"
#include <hccl/hccl_comm.h>
#include "channel_process.h"
#include "base_config_legacy.h"
#define private public
#include "my_rank.h"
#undef private
#include "hccl_comm_pub.h"
#include "llt_hccl_stub_rank_graph.h"
#include "rank_consistency_checker_v2.h"
#include "binary_stream.h"
#include "adapter_rts_common.h"

using namespace hccl;

// 对端回包是否携带hcommData / 本轮是否无载荷（由各用例切换），SetUp中恢复默认值
static bool g_exchangeWithHcommData = true;
static bool g_exchangeSkipPayload = false;

class ExchangeInfoMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ExchangeInfoMgrTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "ExchangeInfoMgrTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in ExchangeInfoMgrTest SetUP" << std::endl;
        g_exchangeWithHcommData = true;
        g_exchangeSkipPayload = false;
        rankIpPortMap = std::make_shared<std::unordered_map<u32, std::unordered_map<Hccl::IpAddress, u32>>>();
        (*rankIpPortMap)[0][Hccl::IpAddress("1.0.0.0")] = 16666;
        (*rankIpPortMap)[1][Hccl::IpAddress("2.0.0.0")] = 16666;
        (*rankIpPortMap)[2][Hccl::IpAddress("0.0.0.0")] = 16666;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in ExchangeInfoMgrTest TearDown" << std::endl;
    }

    Hccl::RankIpPortMapPtr rankIpPortMap;
};

static HcclResult InitCollComm(std::shared_ptr<hccl::hcclComm> hcclCommPtr)
{
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    void* commV2 = (void*)0x2000;
    uint32_t rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclOpExpansionMode = 1;
    config.hcclRdmaTrafficClass = 0xFFFFFFFF;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;
    return hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
}

// 一致性交换流程中 ChannelGetDesc 返回的 socket：由用例构造，NicType 决定走同步还是异步收发
static Hccl::Socket* g_exchangeSocketForStub = nullptr;

// mock ChannelProcess::ChannelGetDesc：返回带真实 socket 的 desc，供一致性交换流程使用
static HcclResult StubChannelGetDescForExchange(ChannelHandle channelHandle, HcommChannelDesc* channelDesc)
{
    (void)channelHandle;
    channelDesc->socket = reinterpret_cast<HcommSocket>(g_exchangeSocketForStub);
    channelDesc->role = HCOMM_SOCKET_ROLE_CLIENT;
    return HCCL_SUCCESS;
}

// 模拟对端回包：按PackExchangeInfo的布局填充[size_t checkFrameSize][CheckFrameV2(可选)][vector<u8>]，
// 非跳过时长度与本端交换信息一致(4字节)，跳过时携带空hcclData
// 非批量协议先收uint64_t size再按size收Data：size==sizeof(uint64_t)时回填对端packed大小，否则回填packed内容
static constexpr size_t UT_EXCH_INFO_LEN = 4;

static void FillExchangePayload(void* recvBuf, u32 size)
{
    Hccl::BinaryStream stream;
    const size_t checkFrameSize = g_exchangeWithHcommData ? sizeof(CheckFrameV2) : 0;
    stream << checkFrameSize;
    if (g_exchangeWithHcommData) {
        CheckFrameV2 frame{};
        stream << frame;
    }
    // 跳过时携带空hcclData，与本端 [size_t 0][空hcclData] 包对称
    std::vector<u8> data;
    if (!g_exchangeSkipPayload) {
        data.assign(UT_EXCH_INFO_LEN, 0);
    }
    stream << data;
    std::vector<char> packed;
    stream.Dump(packed);

    auto* dst = static_cast<char*>(recvBuf);
    if (size == sizeof(uint64_t)) {
        const uint64_t remoteSize = static_cast<uint64_t>(packed.size());
        const auto* src = reinterpret_cast<const char*>(&remoteSize);
        for (size_t i = 0; i < sizeof(uint64_t); ++i) {
            dst[i] = src[i];
        }
        return;
    }
    const size_t copyLen = (packed.size() < static_cast<size_t>(size)) ? packed.size() : static_cast<size_t>(size);
    for (size_t i = 0; i < copyLen; ++i) {
        dst[i] = packed[i];
    }
}

static bool StubRecvExchangePayload(Hccl::Socket* self, void* recvBuf, u32 size)
{
    (void)self;
    FillExchangePayload(recvBuf, size);
    return true;
}

static void StubRecvAsyncExchangePayload(Hccl::Socket* self, void* recvBuf, u32 size)
{
    (void)self;
    FillExchangePayload(recvBuf, size);
}

TEST_F(ExchangeInfoMgrTest, Ut_WaitAllAsyncComplete_When_AllOk_Expect_Success)
{
    // mock Socket::GetAsyncStatus返回OK
    MOCKER_CPP(&Hccl::Socket::GetAsyncStatus).stubs().will(returnValue(Hccl::SocketStatus(Hccl::SocketStatus::OK)));

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);

    // 构造socket列表（指针值仅用于mock匹配，不实际调用）
    std::vector<Hccl::Socket*> sockets = {(Hccl::Socket*)0x1, (Hccl::Socket*)0x2};
    std::vector<uint32_t> remoteRanks = {1, 2};
    ExchangeInfoMgr exchangeInfoMgr;
    HcclResult ret = exchangeInfoMgr.WaitAllAsyncComplete(sockets, remoteRanks);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ExchangeInfoMgrTest, Ut_BatchExchange_When_NewRankConsistent_Expect_Success)
{
    HcclResult ret = HCCL_SUCCESS;
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>();
    ASSERT_EQ(InitCollComm(hcclCommPtr), HCCL_SUCCESS);
    hccl::CollComm* collComm = hcclCommPtr->GetCollComm();
    hccl::MyRank* myRank = collComm->GetMyRank();
    CollCommConfigConsistency& collCommConfigConsistency = myRank->GetCollCommConfigConsistency();
    std::vector<u8> localData = {0xDE, 0xAD, 0xBE, 0xEF};
    ret = collCommConfigConsistency.AddExchangeInfo(localData.data(), localData.size());
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // mock Socket异步接口：GetAsyncStatus返回OK
    MOCKER_CPP(&Hccl::Socket::GetAsyncStatus).stubs().will(returnValue(Hccl::SocketStatus(Hccl::SocketStatus::OK)));
    // mock SendAsync/RecvAsync成功（RecvAsync填充与本端等长的对端交换信息）
    MOCKER_CPP(&Hccl::Socket::SendAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::RecvAsync).stubs().will(invoke(StubRecvAsyncExchangePayload));
    // mock 超时配置
    MOCKER_CPP(&Hccl::EnvSocketConfig::GetLinkTimeOut).stubs().will(returnValue((s32)1));
    MOCKER_CPP(&RankConsistencyCheckerV2::CompareCheckFrameV2).stubs().will(returnValue(HCCL_SUCCESS));

    // 构造真实socket（DEVICE_NIC_TYPE：走异步收发），mock Destroy避免析构时调用HCCP接口
    Hccl::Socket exchangeSocket(
        nullptr, Hccl::IpAddress{}, 0, Hccl::IpAddress{}, "ut_exch", Hccl::SocketRole::CLIENT,
        Hccl::NicType::DEVICE_NIC_TYPE);
    g_exchangeSocketForStub = &exchangeSocket;
    MOCKER_CPP(&Hccl::Socket::Destroy).stubs();

    uint32_t remoteRanks[1] = {1};
    ChannelHandle channelHandles[1] = {(ChannelHandle)0x1};
    MOCKER_CPP(&hcomm::ChannelProcess::ChannelGetDesc).stubs().will(invoke(StubChannelGetDescForExchange));
    ExchangeInfoMgr exchangeInfoMgr;
    ret = exchangeInfoMgr.BatchExchangeAndCheckConsistency(remoteRanks, channelHandles, 1, collCommConfigConsistency);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// host 网卡场景，使用 Peer 模式同步收发接口，模拟交换信息成功
TEST_F(ExchangeInfoMgrTest, Ut_BatchExchange_When_NewRankConsistent_Expect_Success_On_Cpu)
{
    // 初始化通信域
    HcclResult ret = HCCL_SUCCESS;
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>();
    ASSERT_EQ(InitCollComm(hcclCommPtr), HCCL_SUCCESS);

    // 添加交换信息
    hccl::CollComm* collComm = hcclCommPtr->GetCollComm();
    hccl::MyRank* myRank = collComm->GetMyRank();
    CollCommConfigConsistency& collCommConfigConsistency = myRank->GetCollCommConfigConsistency();
    std::vector<u8> localData = {0xDE, 0xAD, 0xBE, 0xEF};
    ret = collCommConfigConsistency.AddExchangeInfo(localData.data(), localData.size());
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // mock socket接口
    MOCKER_CPP(&Hccl::Socket::GetAsyncStatus).stubs().will(returnValue(Hccl::SocketStatus(Hccl::SocketStatus::OK)));
    MOCKER_CPP(&Hccl::Socket::Send).stubs().will(returnValue(true));
    MOCKER_CPP(&Hccl::Socket::Recv).stubs().will(invoke(StubRecvExchangePayload));
    // mock 超时配置
    MOCKER_CPP(&Hccl::EnvSocketConfig::GetLinkTimeOut).stubs().will(returnValue((s32)1));
    // mock 比对结果
    MOCKER_CPP(&RankConsistencyCheckerV2::CompareCheckFrameV2).stubs().will(returnValue(HCCL_SUCCESS));

    // 构造真实socket（HOST_NIC_TYPE：走同步收发），mock Destroy避免析构时调用HCCP接口
    Hccl::Socket exchangeSocket(
        nullptr, Hccl::IpAddress{}, 0, Hccl::IpAddress{}, "ut_exch", Hccl::SocketRole::CLIENT,
        Hccl::NicType::HOST_NIC_TYPE);
    g_exchangeSocketForStub = &exchangeSocket;
    MOCKER_CPP(&Hccl::Socket::Destroy).stubs();

    // 构造远端rank
    uint32_t remoteRanks[1] = {1};
    ChannelHandle channelHandles[1] = {(ChannelHandle)0x1};
    MOCKER_CPP(&hcomm::ChannelProcess::ChannelGetDesc).stubs().will(invoke(StubChannelGetDescForExchange));

    // 交换信息
    ExchangeInfoMgr exchangeInfoMgr;
    ret = exchangeInfoMgr.BatchExchangeAndCheckConsistency(remoteRanks, channelHandles, 1, collCommConfigConsistency);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// PackExchangeInfo: 同时打包含/不含hcommData的两个数据包
TEST_F(ExchangeInfoMgrTest, Ut_PackExchangeInfo_When_HcclDataAdded_Expect_BothPackets)
{
    CollCommConfigConsistency collCommConfigConsistency;
    std::vector<u8> localData = {0xDE, 0xAD, 0xBE, 0xEF};
    ASSERT_EQ(collCommConfigConsistency.AddExchangeInfo(localData.data(), localData.size()), HCCL_SUCCESS);

    ExchangeInfoMgr exchangeInfoMgr;
    ExchangeInfoMgr::ExchangeCtx ctx;
    EXPECT_EQ(exchangeInfoMgr.PackExchangeInfo(collCommConfigConsistency, ctx), HCCL_SUCCESS);
    EXPECT_EQ(ctx.sizeAll, sizeof(size_t) + sizeof(CheckFrameV2) + sizeof(size_t) + localData.size());
    EXPECT_EQ(ctx.sizeOnlyHccl, sizeof(size_t) + sizeof(size_t) + localData.size());

    // allData: [uint64_t size][size_t checkFrameSize][CheckFrameV2][hcclData]
    std::vector<char> allBody(ctx.allData.begin() + sizeof(uint64_t), ctx.allData.end());
    Hccl::BinaryStream allStream(allBody);
    size_t allCheckFrameSize = 0;
    allStream >> allCheckFrameSize;
    EXPECT_EQ(allCheckFrameSize, sizeof(CheckFrameV2));
    CheckFrameV2 allFrame{};
    allStream >> allFrame;
    std::vector<u8> allHcclData;
    allStream >> allHcclData;
    EXPECT_EQ(allHcclData, localData);

    // onlyHcclData: [uint64_t size][size_t checkFrameSize=0][hcclData]
    std::vector<char> onlyBody(ctx.onlyHcclData.begin() + sizeof(uint64_t), ctx.onlyHcclData.end());
    Hccl::BinaryStream onlyStream(onlyBody);
    size_t onlyCheckFrameSize = 1;
    onlyStream >> onlyCheckFrameSize;
    EXPECT_EQ(onlyCheckFrameSize, 0U);
    std::vector<u8> onlyHcclData;
    onlyStream >> onlyHcclData;
    EXPECT_EQ(onlyHcclData, localData);
}

// PackExchangeInfo: 刷新device失败时返回错误码，不继续拼包
TEST_F(ExchangeInfoMgrTest, Ut_PackExchangeInfo_When_GetDeviceRefreshFails_Expect_RuntimeError)
{
    CollCommConfigConsistency collCommConfigConsistency;
    std::vector<u8> localData = {0xDE, 0xAD, 0xBE, 0xEF};
    ASSERT_EQ(collCommConfigConsistency.AddExchangeInfo(localData.data(), localData.size()), HCCL_SUCCESS);

    MOCKER(hrtGetDeviceRefresh).stubs().will(returnValue(HCCL_E_RUNTIME));
    ExchangeInfoMgr exchangeInfoMgr;
    ExchangeInfoMgr::ExchangeCtx ctx;
    EXPECT_EQ(exchangeInfoMgr.PackExchangeInfo(collCommConfigConsistency, ctx), HCCL_E_RUNTIME);
    EXPECT_TRUE(ctx.allData.empty());
    EXPECT_EQ(ctx.sizeAll, 0U);
}

// 已校验过的rank：后续轮次仅交换hcclData（不含hcommData），无hcclData时交换空载荷包
TEST_F(ExchangeInfoMgrTest, Ut_BatchExchange_When_RankAlreadyChecked_Expect_OnlyHcclData)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>();
    ASSERT_EQ(InitCollComm(hcclCommPtr), HCCL_SUCCESS);
    hccl::CollComm* collComm = hcclCommPtr->GetCollComm();
    hccl::MyRank* myRank = collComm->GetMyRank();
    CollCommConfigConsistency& collCommConfigConsistency = myRank->GetCollCommConfigConsistency();
    std::vector<u8> localData = {0xDE, 0xAD, 0xBE, 0xEF};

    MOCKER_CPP(&Hccl::Socket::GetAsyncStatus).stubs().will(returnValue(Hccl::SocketStatus(Hccl::SocketStatus::OK)));
    MOCKER_CPP(&Hccl::Socket::SendAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::RecvAsync).stubs().will(invoke(StubRecvAsyncExchangePayload));
    MOCKER_CPP(&Hccl::EnvSocketConfig::GetLinkTimeOut).stubs().will(returnValue((s32)1));
    MOCKER_CPP(&RankConsistencyCheckerV2::CompareCheckFrameV2).stubs().will(returnValue(HCCL_SUCCESS));

    Hccl::Socket exchangeSocket(
        nullptr, Hccl::IpAddress{}, 0, Hccl::IpAddress{}, "ut_exch", Hccl::SocketRole::CLIENT,
        Hccl::NicType::DEVICE_NIC_TYPE);
    g_exchangeSocketForStub = &exchangeSocket;
    MOCKER_CPP(&Hccl::Socket::Destroy).stubs();

    uint32_t remoteRanks[1] = {1};
    ChannelHandle channelHandles[1] = {(ChannelHandle)0x1};
    MOCKER_CPP(&hcomm::ChannelProcess::ChannelGetDesc).stubs().will(invoke(StubChannelGetDescForExchange));

    ExchangeInfoMgr exchangeInfoMgr;
    // 第一轮：未校验过，交换含hcommData的整包
    ASSERT_EQ(collCommConfigConsistency.AddExchangeInfo(localData.data(), localData.size()), HCCL_SUCCESS);
    g_exchangeWithHcommData = true;
    EXPECT_EQ(
        exchangeInfoMgr.BatchExchangeAndCheckConsistency(remoteRanks, channelHandles, 1, collCommConfigConsistency),
        HCCL_SUCCESS);
    EXPECT_EQ(exchangeInfoMgr.checkedRankIdList_.count(1), 1U);

    // 第二轮：已校验过，仅交换hcclData
    ASSERT_EQ(collCommConfigConsistency.AddExchangeInfo(localData.data(), localData.size()), HCCL_SUCCESS);
    g_exchangeWithHcommData = false;
    EXPECT_EQ(
        exchangeInfoMgr.BatchExchangeAndCheckConsistency(remoteRanks, channelHandles, 1, collCommConfigConsistency),
        HCCL_SUCCESS);

    // 第三轮：已校验过且未新增hcclData，交换空载荷包（对端解包后跳过存储）
    g_exchangeSkipPayload = true;
    EXPECT_EQ(
        exchangeInfoMgr.BatchExchangeAndCheckConsistency(remoteRanks, channelHandles, 1, collCommConfigConsistency),
        HCCL_SUCCESS);
}

// CheckRecvedSizes: 数量不一致或size与期望不符返回 E_PARA，一致返回 SUCCESS
TEST_F(ExchangeInfoMgrTest, Ut_CheckRecvedSizes_When_Invalid_Expect_ParaError)
{
    ExchangeInfoMgr exchangeInfoMgr;
    ExchangeInfoMgr::ExchangeCtx ctx;
    ctx.recvBuffs.resize(1);
    ctx.rmtRankIds.push_back(1);
    ctx.expectRecvSizes.assign(1, 100);

    // recvedSizes 数量与期望不一致
    EXPECT_EQ(exchangeInfoMgr.CheckRecvedSizes(ctx), HCCL_E_PARA);

    // size 与期望不一致
    ctx.recvedSizes.assign(1, 50);
    EXPECT_EQ(exchangeInfoMgr.CheckRecvedSizes(ctx), HCCL_E_PARA);

    // size 与期望一致
    ctx.recvedSizes[0] = 100;
    EXPECT_EQ(exchangeInfoMgr.CheckRecvedSizes(ctx), HCCL_SUCCESS);
}

// ConsistencyRecvedDataProc: 对端数据长度与本端不一致返回 E_PARA
TEST_F(ExchangeInfoMgrTest, Ut_ConsistencyRecvedDataProc_When_RemoteDataLenMismatch_Expect_ParaError)
{
    CollCommConfigConsistency collCommConfigConsistency;
    std::vector<u8> localData = {0xDE, 0xAD, 0xBE, 0xEF};
    ASSERT_EQ(collCommConfigConsistency.AddExchangeInfo(localData.data(), localData.size()), HCCL_SUCCESS);

    ExchangeInfoMgr::ExchangeCtx ctx;
    ctx.rmtRankIds.push_back(1);

    // 对端回包长度为 8，与本端 4 字节不一致
    Hccl::BinaryStream stream;
    const size_t checkFrameSize = sizeof(CheckFrameV2);
    CheckFrameV2 frame{};
    std::vector<u8> remoteData = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    stream << checkFrameSize;
    stream << frame;
    stream << remoteData;
    std::vector<char> packed;
    stream.Dump(packed);
    ctx.recvBuffs.emplace_back(packed.size(), 0);
    std::copy(packed.begin(), packed.end(), ctx.recvBuffs[0].begin());

    MOCKER_CPP(&RankConsistencyCheckerV2::CompareCheckFrameV2).stubs().will(returnValue(HCCL_SUCCESS));
    ExchangeInfoMgr exchangeInfoMgr;
    EXPECT_EQ(exchangeInfoMgr.ConsistencyRecvedDataProc(collCommConfigConsistency, ctx), HCCL_E_PARA);
}

// ConsistencyRecvedDataProc: 对端数据为空时不存储，查询返回 actualLength=0
TEST_F(ExchangeInfoMgrTest, Ut_ConsistencyRecvedDataProc_When_RemoteDataEmpty_Expect_SkipStore)
{
    CollCommConfigConsistency collCommConfigConsistency;

    ExchangeInfoMgr::ExchangeCtx ctx;
    ctx.rmtRankIds.push_back(1);

    Hccl::BinaryStream stream;
    const size_t checkFrameSize = 0;
    std::vector<u8> remoteData;
    stream << checkFrameSize;
    stream << remoteData;
    std::vector<char> packed;
    stream.Dump(packed);
    ctx.recvBuffs.emplace_back(packed.size(), 0);
    std::copy(packed.begin(), packed.end(), ctx.recvBuffs[0].begin());

    ExchangeInfoMgr exchangeInfoMgr;
    EXPECT_EQ(exchangeInfoMgr.ConsistencyRecvedDataProc(collCommConfigConsistency, ctx), HCCL_SUCCESS);

    uint32_t actualLength = 0;
    u8 buf[1] = {0};
    EXPECT_EQ(collCommConfigConsistency.GetExchangeInfo(1, sizeof(buf), buf, &actualLength), HCCL_SUCCESS);
    EXPECT_EQ(actualLength, 0U);
}

// ConsistencyRecvedDataProc: 对端CheckFrameV2大小与本端不一致返回 E_PARA
TEST_F(ExchangeInfoMgrTest, Ut_ConsistencyRecvedDataProc_When_RemoteFrameSizeMismatch_Expect_ParaError)
{
    CollCommConfigConsistency collCommConfigConsistency;

    ExchangeInfoMgr::ExchangeCtx ctx;
    ctx.rmtRankIds.push_back(1);

    // 对端声明的帧大小与本端 sizeof(CheckFrameV2) 不一致
    Hccl::BinaryStream stream;
    const size_t checkFrameSize = sizeof(CheckFrameV2) + 1;
    stream << checkFrameSize;
    std::vector<char> packed;
    stream.Dump(packed);
    ctx.recvBuffs.emplace_back(packed.size(), 0);
    std::copy(packed.begin(), packed.end(), ctx.recvBuffs[0].begin());

    ExchangeInfoMgr exchangeInfoMgr;
    EXPECT_EQ(exchangeInfoMgr.ConsistencyRecvedDataProc(collCommConfigConsistency, ctx), HCCL_E_PARA);
}

// ConsistencyRecvedDataProc: recvBuff 为空（当前协议下不应出现）返回 E_INTERNAL
TEST_F(ExchangeInfoMgrTest, Ut_ConsistencyRecvedDataProc_When_RecvBuffEmpty_Expect_Internal)
{
    CollCommConfigConsistency collCommConfigConsistency;

    ExchangeInfoMgr::ExchangeCtx ctx;
    ctx.rmtRankIds.push_back(1);
    ctx.recvBuffs.resize(1);

    ExchangeInfoMgr exchangeInfoMgr;
    EXPECT_EQ(exchangeInfoMgr.ConsistencyRecvedDataProc(collCommConfigConsistency, ctx), HCCL_E_INTERNAL);
}

// ConsistencyRecvedDataProc: 刷新device失败时返回错误码，不继续生成本端校验帧
TEST_F(ExchangeInfoMgrTest, Ut_ConsistencyRecvedDataProc_When_GetDeviceRefreshFails_Expect_RuntimeError)
{
    CollCommConfigConsistency collCommConfigConsistency;

    ExchangeInfoMgr::ExchangeCtx ctx;
    ctx.rmtRankIds.push_back(1);

    // 对端包内含 CheckFrameV2，触发本端按需生成校验帧前的 device 刷新
    Hccl::BinaryStream stream;
    const size_t checkFrameSize = sizeof(CheckFrameV2);
    CheckFrameV2 frame{};
    stream << checkFrameSize;
    stream << frame;
    std::vector<char> packed;
    stream.Dump(packed);
    ctx.recvBuffs.emplace_back(packed.size(), 0);
    std::copy(packed.begin(), packed.end(), ctx.recvBuffs[0].begin());

    MOCKER(hrtGetDeviceRefresh).stubs().will(returnValue(HCCL_E_RUNTIME));
    ExchangeInfoMgr exchangeInfoMgr;
    EXPECT_EQ(exchangeInfoMgr.ConsistencyRecvedDataProc(collCommConfigConsistency, ctx), HCCL_E_RUNTIME);
}

// BatchExchangeAndCheckConsistency: channelNum 为 0 直接返回成功
TEST_F(ExchangeInfoMgrTest, Ut_BatchExchange_When_ChannelNumZero_Expect_Success)
{
    ExchangeInfoMgr exchangeInfoMgr;
    CollCommConfigConsistency collCommConfigConsistency;
    uint32_t remoteRanks[1] = {1};
    ChannelHandle channelHandles[1] = {0};
    EXPECT_EQ(
        exchangeInfoMgr.BatchExchangeAndCheckConsistency(remoteRanks, channelHandles, 0, collCommConfigConsistency),
        HCCL_SUCCESS);
}

// BatchExchangeAndCheckConsistency: ChannelGetDesc 失败时透传错误码
TEST_F(ExchangeInfoMgrTest, Ut_BatchExchange_When_ChannelGetDescFails_Expect_Error)
{
    ExchangeInfoMgr exchangeInfoMgr;
    CollCommConfigConsistency collCommConfigConsistency;
    uint32_t remoteRanks[1] = {1};
    ChannelHandle channelHandles[1] = {(ChannelHandle)0x1};
    MOCKER_CPP(&hcomm::ChannelProcess::ChannelGetDesc).stubs().will(returnValue(HCCL_E_NOT_FOUND));
    EXPECT_EQ(
        exchangeInfoMgr.BatchExchangeAndCheckConsistency(remoteRanks, channelHandles, 1, collCommConfigConsistency),
        HCCL_E_NOT_FOUND);
}

// BatchExchangeAndCheckConsistency: socket 为空返回 E_INTERNAL
TEST_F(ExchangeInfoMgrTest, Ut_BatchExchange_When_SocketNull_Expect_Internal)
{
    ExchangeInfoMgr exchangeInfoMgr;
    CollCommConfigConsistency collCommConfigConsistency;
    uint32_t remoteRanks[1] = {1};
    ChannelHandle channelHandles[1] = {(ChannelHandle)0x1};
    g_exchangeSocketForStub = nullptr;
    MOCKER_CPP(&hcomm::ChannelProcess::ChannelGetDesc).stubs().will(invoke(StubChannelGetDescForExchange));
    EXPECT_EQ(
        exchangeInfoMgr.BatchExchangeAndCheckConsistency(remoteRanks, channelHandles, 1, collCommConfigConsistency),
        HCCL_E_INTERNAL);
}

// PackExchangeInfo: 拼包内容超过上限（用户数据上限 + 帧头）返回 E_PARA
TEST_F(ExchangeInfoMgrTest, Ut_BatchExchange_When_PackSizeExceedsLimit_Expect_ParaError)
{
    ExchangeInfoMgr exchangeInfoMgr;
    CollCommConfigConsistency collCommConfigConsistency;
    // 绕过 AddExchangeInfo 上限，直接构造超限本端数据（HCCL_EXCHANGE_INFO_LEN + 1）
    collCommConfigConsistency.exchangeInfoBuf_.assign(HCCL_EXCHANGE_INFO_LEN + 1, 0x5A);
    uint32_t remoteRanks[1] = {1};
    ChannelHandle channelHandles[1] = {(ChannelHandle)0x1};

    // ExchangeParamInfos 会先读取 socket 的 NicType 再执行拼包上限校验，需构造真实 socket
    Hccl::Socket exchangeSocket(
        nullptr, Hccl::IpAddress{}, 0, Hccl::IpAddress{}, "ut_exch", Hccl::SocketRole::CLIENT,
        Hccl::NicType::HOST_NIC_TYPE);
    g_exchangeSocketForStub = &exchangeSocket;
    MOCKER_CPP(&Hccl::Socket::Destroy).stubs();

    MOCKER_CPP(&hcomm::ChannelProcess::ChannelGetDesc).stubs().will(invoke(StubChannelGetDescForExchange));
    EXPECT_EQ(
        exchangeInfoMgr.BatchExchangeAndCheckConsistency(remoteRanks, channelHandles, 1, collCommConfigConsistency),
        HCCL_E_PARA);
}
