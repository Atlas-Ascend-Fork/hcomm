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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <thread>
#define private public
#include "host_socket_handle_manager.h"
#include "whitelist.h"
#include "hccp_peer_manager.h"
#include "dev_type.h"

#include "socket.h"
#include <string>
#include <unordered_map>
#include <vector>
#include "new_rank_info.h"
#include "rank_table_info.h"
#include "json/common/json_parse_util.h"
#include "root_handle_v2.h"
#include "internal_exception.h"
#include "timeout_exception.h"
#include "socket_exception.h"
#include "null_ptr_exception.h"
#include "rank_info_dispatcher.h"
#include "rank_info_detect_service.h"
#include "orion_adapter_rts.h"
#include "env_config/env_config_v2.h"
#include "base_config_legacy.h"
#include "socket_agent.h"
#include "adapter_error_manager_pub.h"
#undef private

using namespace Hccl;

namespace {
std::string g_serviceCapturedErrorCode;
// RecvRemoteAgentIdStub 的序号计数器：按用例重置，避免用例顺序影响 agentId 取值
// （agentId 会被当作 rankId 校验，跨用例累加会超出 rankSize 触发 FailedConnectionAgentIdString 报错）
int g_recvAgentIdNextRank = 0;
void StubServiceRptInputErr(std::string error_code, std::vector<std::string> key, std::vector<std::string> value)
{
    (void)key;
    (void)value;
    g_serviceCapturedErrorCode = error_code;
}

} // namespace

class RankInfoDetectServiceTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankInfoDetectService tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RankInfoDetectService tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in RankInfoDetectService SetUP" << std::endl;
        g_recvAgentIdNextRank = 0;
        // 初始化模拟 Socket 句柄
        hccpSocketHandle = new int(0);
        MOCKER(HrtRaSocketInit).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(hccpSocketHandle));

        // 1. 构造测试所需参数
        u32 devPhyId_ = 0;
        IpAddress hostIp_ = GetAnIpAddress();
        u32 hostPort_ = 60007;
        IpAddress remoteIp_ = IpAddress("127.0.0.1");
        std::string serverSocketTag_ = "rank_info_test_server";

        // 2. 创建服务器 Socket（用于传入被测试类构造函数）
        auto serverSocket_ = std::make_shared<Socket>(
            hccpSocketHandle, hostIp_, hostPort_, remoteIp_, serverSocketTag_, SocketRole::SERVER,
            NicType::DEVICE_NIC_TYPE);

        MOCKER_CPP(&HccpPeerManager::Init).stubs().with(mockcpp::any());
        MOCKER_CPP(&HccpPeerManager::DeInit).stubs().with(mockcpp::any());
        MOCKER_CPP(&HostSocketHandleManager::Create)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(hccpSocketHandle));

        serverSocket_->Listen(); // 启动监听（模拟真实场景）

        // 4. 初始化被测试对象
        rankInfoDetectService_ = new RankInfoDetectService(devPhyId_, serverSocket_, "test", {});
    }

    virtual void TearDown()
    {
        // 验证所有打桩的预期是否满足（mockcpp 核心校验）
        GlobalMockObject::verify();
        // 释放动态资源
        delete hccpSocketHandle;
        delete rankInfoDetectService_;
        std::cout << "A Test case in RankInfoDetectService TearDown" << std::endl;
    }

    IpAddress GetAnIpAddress() { return IpAddress("1.0.0.0"); }

    // 辅助函数：配置 1s 建链超时并 mock HostSocketHandleManager::Get
    void SetupMockSocketAndTimeoutConfig()
    {
        static EnvSocketConfig envSocketConfig;
        EnvSocketConfig& fakeEnvSocketConfig = envSocketConfig;
        fakeEnvSocketConfig.linkTimeOut = CfgField<s32>{"HCCL_CONNECT_TIMEOUT", s32(1), Str2T<s32>};
        fakeEnvSocketConfig.linkTimeOut.isParsed = true;
        MOCKER_CPP(&EnvConfig::GetSocketConfig).stubs().will(returnValue(fakeEnvSocketConfig));

        MOCKER_CPP(&HostSocketHandleManager::Get)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(hccpSocketHandle));
    }

    // 辅助函数：mock 白名单开关
    void MockHostNicConfigWhitelistDisable(bool whitelistDisable)
    {
        static EnvHostNicConfig envHostNicConfig;
        EnvHostNicConfig& fakeEnvHostNicConfig = envHostNicConfig;
        fakeEnvHostNicConfig.whitelistDisable
            = CfgField<bool>{"HCCL_WHITELIST_DISABLE", whitelistDisable, CastBin2Bool};
        fakeEnvHostNicConfig.whitelistDisable.isParsed = true;
        MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvHostNicConfig));
    }

    // 辅助函数：生成模拟的 RankInfo 消息（用于 GetRankTable 测试）
    std::string GenMockRankInfoMsg(const std::string& rankTableStr, u32 step)
    {
        // 消息格式：[rankTableStr][step(4字节)]
        std::string msg = rankTableStr;
        msg.append(reinterpret_cast<const char*>(&step), sizeof(step));
        return msg;
    }

    // 成员变量
    void* hccpSocketHandle;
    RankInfoDetectService* rankInfoDetectService_;
};

TEST_F(RankInfoDetectServiceTest, Ut_RankInfoDetectService_When_Init_Expect_Success)
{
    auto res1 = rankInfoDetectService_->devPhyId_;
    EXPECT_EQ(0, res1);
    EXPECT_EQ(RankInfoDetectService::GrowState::STARTING, rankInfoDetectService_->GetGrowState());

    EXPECT_NE(nullptr, rankInfoDetectService_->serverSocket_);

    auto res2 = rankInfoDetectService_->serverSocket_->isListening;
    EXPECT_EQ(true, res2);

    auto res3 = rankInfoDetectService_->serverSocket_->listenPort;
    EXPECT_EQ(60007, res3);
}

TEST_F(RankInfoDetectServiceTest, Ut_GetConnections_When_Timeout_Expect_fail)
{
    SetupMockSocketAndTimeoutConfig();

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT));

    EXPECT_EQ(rankInfoDetectService_->GetConnections(true), HcclResult::HCCL_E_INTERNAL);

    auto res1 = rankInfoDetectService_->connSockets_.size();
    EXPECT_EQ(0, res1);
}

TEST_F(RankInfoDetectServiceTest, Ut_GetConnections_When_Timeout_Expect_NoEI0016Report)
{
    SetupMockSocketAndTimeoutConfig();

    // 持续 CONNECTING，模拟对端一直未建链直至超时
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::CONNECTING));

    g_serviceCapturedErrorCode.clear();
    MOCKER(RptInputErr).stubs().will(invoke(StubServiceRptInputErr));

    EXPECT_EQ(rankInfoDetectService_->GetConnections(true), HcclResult::HCCL_E_INTERNAL);
    // 耗时耗尽分支已收敛为不再单独上报 EI0016，避免与超时 EI0015 重复
    EXPECT_NE(g_serviceCapturedErrorCode, "EI0016");
}

TEST_F(RankInfoDetectServiceTest, Ut_GetConnections_When_Normal_Expect_Success)
{
    MOCKER_CPP(&HostSocketHandleManager::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(hccpSocketHandle));

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK));

    u32 rankSize = 1;
    void* msg = &rankSize;
    u64 msgLen = sizeof(u32);
    u64& revMsgLen = msgLen;
    std::string agentId;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId)
        .stubs()
        .with(mockcpp::any(), outBound(agentId))
        .will(invoke(+[](RankInfoDetectService*, SocketAgent&, std::string& id) {
            id = "0";
            return true;
        }));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));
    MOCKER_CPP(&SocketAgent::RecvMsg)
        .stubs()
        .with(outBound(msg), outBound(revMsgLen), mockcpp::any())
        .will(returnValue(true));

    EXPECT_EQ(rankInfoDetectService_->GetConnections(true), HcclResult::HCCL_SUCCESS);

    auto res1 = rankInfoDetectService_->connSockets_.size();
    EXPECT_EQ(1, res1);

    for (auto& iter : rankInfoDetectService_->connSockets_) {
        EXPECT_NE(nullptr, iter.second);
    }
}

bool RecvGrowRemoteAgentIdStub(RankInfoDetectService* thisPtr, SocketAgent& connSocketAgent, std::string& agentId)
{
    (void)thisPtr;
    (void)connSocketAgent;
    agentId = "1";
    return true;
}

TEST_F(RankInfoDetectServiceTest, Ut_GetConnections_When_GrowIncomplete_Expect_FailAndRollback)
{
    SetupMockSocketAndTimeoutConfig();
    MOCKER(RptInputErr).stubs().will(invoke(StubServiceRptInputErr));

    IpAddress hostIp = rankInfoDetectService_->serverSocket_->GetLocalIp();
    auto oldSocket = std::make_shared<Socket>(
        hccpSocketHandle, hostIp, 10001, hostIp, "rank_info_test_old", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    rankInfoDetectService_->connSockets_["0"] = oldSocket;
    rankInfoDetectService_->expectedNewRankCount_ = 2;
    rankInfoDetectService_->newAgentIds_.clear();

    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(10))));

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK));

    u32 rankSize = 2;
    std::string agentId;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId)
        .stubs()
        .with(mockcpp::any(), outBound(agentId))
        .will(invoke(RecvGrowRemoteAgentIdStub));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));

    EXPECT_EQ(rankInfoDetectService_->GetConnections(false), HcclResult::HCCL_E_TIMEOUT);
    EXPECT_EQ(1, rankInfoDetectService_->connSockets_.size());
    EXPECT_NE(rankInfoDetectService_->connSockets_.end(), rankInfoDetectService_->connSockets_.find("0"));
    EXPECT_TRUE(rankInfoDetectService_->newAgentIds_.empty());
    EXPECT_TRUE(rankInfoDetectService_->failedAgentIdList_.empty());
}

TEST_F(RankInfoDetectServiceTest, Ut_GetRankTable_When_Normal_Expect_Success)
{
    u32 port1 = 12345;
    std::string socketKey1 = std::to_string(port1);

    IpAddress hostIp = rankInfoDetectService_->serverSocket_->GetLocalIp();
    std::string tag1 = RANK_INFO_DETECT_TAG + "_" + socketKey1;
    auto mockSocket1 = std::make_shared<Socket>(
        hccpSocketHandle, hostIp, port1, hostIp, tag1, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    rankInfoDetectService_->connSockets_[socketKey1] = mockSocket1;

    auto res0 = rankInfoDetectService_->connSockets_.size();
    EXPECT_EQ(1, res0);

    for (auto& iter : rankInfoDetectService_->connSockets_) {
        EXPECT_NE(nullptr, iter.second);
    }

    // 构造一个ranktableinfo
    RankTableInfo localRankTable;
    localRankTable.version = "1.0";
    localRankTable.rankCount = 1;
    hccl::NewRankInfo rankInfo{};
    rankInfo.rankId = 0;
    rankInfo.rankLevelInfos.emplace_back(RankLevelInfo{});
    localRankTable.ranks.emplace_back(rankInfo);

    // 把构造好的ranktableinfo转为vector<char> rankInfoMsg
    // 消息格式: [ranktable数据(n字节)][step(4字节)]
    BinaryStream binaryStream;
    EXPECT_EQ(localRankTable.GetBinStream(true, binaryStream), HCCL_SUCCESS);
    binaryStream << rankInfoDetectService_->currentStep_;

    // 字节流转换为vector<char>格式
    vector<char> rankInfoMsg;
    binaryStream.Dump(rankInfoMsg);

    // 取rankInfoMsg的size
    u32 rankInfoSize = rankInfoMsg.size();
    u64 expectLen = rankInfoSize;

    // 取rankInfoMsg的data() （const char *）
    MOCKER(aclrtMallocHostWithCfg).stubs().will(returnValue(1));
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue((void*)rankInfoMsg.data()));
    char* rankInfoMsgToSend = rankInfoMsg.data();
    void* msg = rankInfoMsg.data();
    u64 msgLen = rankInfoMsg.size();
    u64& revMsgLen = msgLen;
    MOCKER_CPP(&SocketAgent::RecvMsg)
        .stubs()
        .with(outBoundP(msg, msgLen), outBound(revMsgLen), mockcpp::any())
        .will(returnValue(true));

    EXPECT_EQ(rankInfoDetectService_->GetRankTable(true), HcclResult::HCCL_SUCCESS);
}

TEST_F(RankInfoDetectServiceTest, Ut_BroadcastRankTable_When_Normal_Expect_Success)
{
    MOCKER_CPP(&RankInfoDispatcher::BroadcastRankTable)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    EXPECT_EQ(rankInfoDetectService_->BroadcastRankTable(), HcclResult::HCCL_SUCCESS);

    IpAddress localIp, remoteIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    rankInfoDetectService_->connSockets_[tag] = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    EXPECT_EQ(rankInfoDetectService_->BroadcastRankTable(), HcclResult::HCCL_SUCCESS);
}

TEST_F(RankInfoDetectServiceTest, Ut_Disconnect_When_Normal_Expect_Success)
{
    u32 port1 = 12345;
    u32 port2 = 12346;
    std::string socketKey1 = std::to_string(port1);
    std::string socketKey2 = std::to_string(port2);

    IpAddress hostIp = rankInfoDetectService_->serverSocket_->GetLocalIp();
    std::string tag1 = RANK_INFO_DETECT_TAG + "_" + socketKey1;
    std::string tag2 = RANK_INFO_DETECT_TAG + "_" + socketKey2;
    auto mockSocket1 = std::make_shared<Socket>(
        hccpSocketHandle, hostIp, port1, hostIp, tag1, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    auto mockSocket2 = std::make_shared<Socket>(
        hccpSocketHandle, hostIp, port2, hostIp, tag2, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    rankInfoDetectService_->connSockets_[socketKey1] = mockSocket1;
    rankInfoDetectService_->connSockets_[socketKey2] = mockSocket2;

    auto res1 = rankInfoDetectService_->connSockets_.size();
    EXPECT_EQ(2, res1);

    for (auto& iter : rankInfoDetectService_->connSockets_) {
        EXPECT_NE(nullptr, iter.second);
    }

    (void)rankInfoDetectService_->Disconnect();

    auto res2 = rankInfoDetectService_->connSockets_.size();
    EXPECT_EQ(0, res2);
}

TEST_F(RankInfoDetectServiceTest, Ut_ParseRankTable_When_Normal_Expect_Success)
{
    RankTableInfo localRankTable;
    localRankTable.version = "1.0";
    localRankTable.rankCount = 1;
    hccl::NewRankInfo rankInfo{};
    rankInfo.rankId = 0;
    rankInfo.rankLevelInfos.emplace_back(RankLevelInfo{});
    localRankTable.ranks.emplace_back(rankInfo);

    // 使用真实 UpdateRankTable，验证空表仍由原有校验拒绝，正常上报继续处理。
    for (u32 reportedCount : {0U, 1U}) {
        SCOPED_TRACE(reportedCount);
        localRankTable.ranks.assign(reportedCount, rankInfo);
        localRankTable.rankCount = reportedCount;
        BinaryStream binaryStream;
        if (reportedCount == 0) {
            // 正常序列化入口拒绝空表，按真实布局构造异常报文以覆盖接收边界。
            binaryStream << localRankTable.version << localRankTable.rankCount << size_t{0} << localRankTable.detour;
            binaryStream << localRankTable.logicSuperPodSplitLayer << localRankTable.logicSuperPodGroupIds.size();
        } else {
            ASSERT_EQ(localRankTable.GetBinStream(true, binaryStream), HCCL_SUCCESS);
        }
        binaryStream << rankInfoDetectService_->currentStep_;
        vector<char> rankInfoMsg;
        binaryStream.Dump(rankInfoMsg);
        ASSERT_EQ(
            rankInfoDetectService_->ParseRankTable(rankInfoMsg, false),
            reportedCount == 1 ? HCCL_SUCCESS : HCCL_E_PARA);
        EXPECT_EQ(rankInfoDetectService_->rankTable_.ranks.size(), reportedCount);
        EXPECT_EQ(rankInfoDetectService_->netLayerCheckInfos_.size(), reportedCount);
    }
    EXPECT_EQ(rankInfoDetectService_->rankTable_.GetSource(), RankTableSource::ROOTINFO);
}

TEST_F(RankInfoDetectServiceTest, Ut_LogicSuperPod_When_GrowCollect_Expect_RootInfoSourceInMergedTable)
{
    RankTableInfo localRankTable;
    localRankTable.version = "2.0";
    localRankTable.rankCount = 1;
    hccl::NewRankInfo rankInfo;
    rankInfo.rankId = 1;
    rankInfo.localId = 1;
    rankInfo.replacedLocalId = 1;
    rankInfo.deviceId = 1;
    rankInfo.logicSuperPodId = "B";
    RankLevelInfo level;
    level.netInstId = "server_1";
    rankInfo.rankLevelInfos.push_back(level);
    localRankTable.ranks.push_back(rankInfo);

    rankInfoDetectService_->rankTable_ = localRankTable;
    auto& oldRank = rankInfoDetectService_->rankTable_.ranks[0];
    oldRank.rankId = 0;
    oldRank.localId = 0;
    oldRank.replacedLocalId = 0;
    oldRank.deviceId = 0;
    oldRank.logicSuperPodId = "A";
    oldRank.rankLevelInfos[0].netInstId = "server_0";
    ASSERT_EQ(rankInfoDetectService_->rankTable_.GetSource(), RankTableSource::RANKTABLE);
    rankInfoDetectService_->currentStep_ = 3;

    BinaryStream binaryStream;
    ASSERT_EQ(localRankTable.GetBinStream(true, binaryStream), HCCL_SUCCESS);
    binaryStream << u32{0}; // grow新rank从step0接入。
    vector<char> rankInfoMsg;
    binaryStream.Dump(rankInfoMsg);

    ASSERT_EQ(rankInfoDetectService_->ParseRankTable(rankInfoMsg, true), HCCL_SUCCESS);
    ASSERT_EQ(rankInfoDetectService_->MarkGrowLoopReady(), HCCL_SUCCESS);
    RankTableInfo mergedRankTable;
    ASSERT_EQ(rankInfoDetectService_->WaitForCollectComplete(mergedRankTable), HCCL_SUCCESS);
    EXPECT_EQ(mergedRankTable.GetSource(), RankTableSource::ROOTINFO);
    EXPECT_EQ(mergedRankTable.rankCount, 2u);
    ASSERT_EQ(mergedRankTable.ranks.size(), 2u);
    EXPECT_EQ(mergedRankTable.ranks[0].logicSuperPodId, "A");
    EXPECT_EQ(mergedRankTable.ranks[1].logicSuperPodId, "B");
    EXPECT_EQ(mergedRankTable.ranks[0].rankLevelInfos[0].netInstId, "server_0");
    EXPECT_EQ(mergedRankTable.ranks[1].rankLevelInfos[0].netInstId, "server_1");
}

TEST_F(RankInfoDetectServiceTest, NetLayerCheckComparesAllRanksBeforeBroadcast)
{
    auto& checks = rankInfoDetectService_->netLayerCheckInfos_;
    rankInfoDetectService_->rankTable_.rankCount = 2;
    NetLayerCheckInfo selected;
    selected.version = NetLayerCheckInfo::VERSION;
    selected.explicitSelection = 1;
    selected.layers = 6;
    checks[0] = selected;
    checks[1] = selected;
    EXPECT_EQ(rankInfoDetectService_->GetNetLayerCheckResult().result, HCCL_SUCCESS);
    checks[1].layers = 3; // not(0) 与 0,1 的有效层不同
    EXPECT_EQ(rankInfoDetectService_->GetNetLayerCheckResult().result, HCCL_E_PARA);
    checks[1].layers = 6;
    checks[1].explicitSelection = 0; // 默认选层与显式选层结果相同可以通过
    EXPECT_EQ(rankInfoDetectService_->GetNetLayerCheckResult().result, HCCL_SUCCESS);
    checks[1] = {}; // 有显式选层时不接受旧端缺少校验信息
    EXPECT_EQ(rankInfoDetectService_->GetNetLayerCheckResult().result, HCCL_E_NOT_SUPPORT);
    checks[0].explicitSelection = 0;
    EXPECT_EQ(rankInfoDetectService_->GetNetLayerCheckResult().result, HCCL_SUCCESS);
    checks[0] = selected;
    checks.erase(1);
    EXPECT_EQ(rankInfoDetectService_->GetNetLayerCheckResult().result, HCCL_E_PARA);
}

TEST_F(RankInfoDetectServiceTest, NetLayerCheckCollectsLocalFailureForBroadcast)
{
    RankTableInfo local;
    local.version = "1.0";
    local.rankCount = 1;
    hccl::NewRankInfo rank;
    rank.rankLevelInfos.emplace_back(RankLevelInfo{});
    local.ranks.push_back(rank);
    const struct {
        u32 version;
        HcclResult reported;
        bool truncated;
        HcclResult expected;
    } cases[] = {
        {NetLayerCheckInfo::VERSION, HCCL_E_PARA, false, HCCL_E_PARA},
        {NetLayerCheckInfo::VERSION, HCCL_E_NOT_SUPPORT, false, HCCL_E_NOT_SUPPORT},
        {NetLayerCheckInfo::VERSION + 1, HCCL_SUCCESS, false, HCCL_E_NOT_SUPPORT},
        {NetLayerCheckInfo::VERSION, HCCL_E_PARA, true, HCCL_E_PARA},
    };
    for (const auto& item : cases) {
        SCOPED_TRACE(item.version);
        BinaryStream stream;
        ASSERT_EQ(local.GetBinStream(true, stream), HCCL_SUCCESS);
        stream << rankInfoDetectService_->currentStep_;
        NetLayerCheckInfo failed;
        failed.version = item.version;
        failed.explicitSelection = 1;
        failed.result = item.reported;
        failed.Serialize(stream);
        std::vector<char> bytes;
        stream.Dump(bytes);
        if (item.truncated) {
            bytes.pop_back();
        }
        // 非法或不支持的尾帧均须记录失败，不能中断对其他 rank 的收集。
        rankInfoDetectService_->rankTable_ = {};
        rankInfoDetectService_->netLayerCheckInfos_.clear();
        ASSERT_EQ(rankInfoDetectService_->ParseRankTable(bytes, false), HCCL_SUCCESS);
        EXPECT_EQ(rankInfoDetectService_->GetNetLayerCheckResult().result, item.expected);
        EXPECT_EQ(rankInfoDetectService_->rankTable_.rankCount, 1U);
    }
}

bool RecvRemoteAgentIdStub(RankInfoDetectService* thisPtr, SocketAgent& connSocketAgent, std::string& agentId)
{
    agentId = std::to_string(g_recvAgentIdNextRank);
    g_recvAgentIdNextRank++;
    return true;
}

TEST_F(RankInfoDetectServiceTest, Ut_ProcessScalableConnStatus_When_GroupComplete_Expect_ZeroRemaining)
{
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId).stubs().will(invoke(RecvRemoteAgentIdStub));
    u32 rankSize = 8;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));

    u32 groupSize = 4;
    u32 expectedSocketNum = 4;
    u32 totalRankNum = 0;
    bool isFirstAcceptTimeOut = false;
    for (u32 connected = 1; connected <= 4; ++connected) {
        ASSERT_TRUE(rankInfoDetectService_->ProcessScalableConnStatus(
            rankInfoDetectService_->serverSocket_, SocketStatus::OK, groupSize, expectedSocketNum, totalRankNum,
            isFirstAcceptTimeOut));
        EXPECT_EQ(expectedSocketNum, 4U - connected);
        EXPECT_EQ(totalRankNum, 8U);
    }
    EXPECT_EQ(rankInfoDetectService_->connSockets_.size(), 4U);
}

TEST_F(RankInfoDetectServiceTest, Ut_ProcessScalableConnStatus_When_RankSizeMismatch_Expect_ContinueWaiting)
{
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId).stubs().will(invoke(RecvRemoteAgentIdStub));
    u32 rankSize = 9;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));

    u32 groupSize = 3;
    u32 expectedSocketNum = 3;
    u32 totalRankNum = 8;
    bool isFirstAcceptTimeOut = false;
    // PR 行为变更：握手失败（rankSize 校验失败）后继续等待其余连接（存活 rank 不被误报为失败
    // 随广播下发），仅 Shutdown 中止——非 shutdown 下返回 true；半握手连接已被
    // EraseIncompleteHandshakeConnection 摘除，该 agent 计入本组失败列表
    EXPECT_TRUE(rankInfoDetectService_->ProcessScalableConnStatus(
        rankInfoDetectService_->serverSocket_, SocketStatus::OK, groupSize, expectedSocketNum, totalRankNum,
        isFirstAcceptTimeOut));
    EXPECT_EQ(expectedSocketNum, 3U); // 握手失败不递减计数，该 agent 随后由失败列表统计
    EXPECT_EQ(totalRankNum, 8U);
    EXPECT_TRUE(rankInfoDetectService_->connSockets_.empty()); // 半握手连接已摘除
}

TEST_F(RankInfoDetectServiceTest, Ut_GetConnections_When_Last_Fail_Expect_Print_Connected_Socket)
{
    MOCKER_CPP(&HostSocketHandleManager::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(hccpSocketHandle));

    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        // start time
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        // execute time
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(10))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(20))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(30))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(40))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(50))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(60))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(70))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(80))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(90))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(100))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(100000))));

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        // 第一次连接中成功
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        // 第二次成功
        .then(returnValue((SocketStatus)SocketStatus::OK))
        // 第三次超时后成功
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        // 第四次异常
        .then(throws(InternalException("Internal Error Happens")));

    u32 rankSize = 4;
    void* msg = &rankSize;
    u64 msgLen = sizeof(u32);
    u64& revMsgLen = msgLen;
    MOCKER_CPP(&SocketAgent::RecvMsg)
        .stubs()
        .with(outBound(msg), outBound(revMsgLen), mockcpp::any())
        .will(returnValue(true));

    string agentId{"0"};
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId)
        .stubs()
        .with(mockcpp::any(), outBound(agentId))
        .will(invoke(RecvRemoteAgentIdStub));

    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));

    EXPECT_EQ(rankInfoDetectService_->GetConnections(true), HcclResult::HCCL_SUCCESS);
}

TEST_F(RankInfoDetectServiceTest, Ut_GetConnections_When_Sudden_Fail)
{
    MOCKER_CPP(&HostSocketHandleManager::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(hccpSocketHandle));

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        // 第一次连接中成功
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        // 第二次先超时，后失败
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT))
        .then(throws(InternalException("Internal Error Happens")));

    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        // start time
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        // execute time
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(20))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(40))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(60))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(80))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(100))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(100000))));

    u32 rankSize = 4;
    void* msg = &rankSize;
    u64 msgLen = sizeof(u32);
    u64& revMsgLen = msgLen;
    MOCKER_CPP(&SocketAgent::RecvMsg)
        .stubs()
        .with(outBound(msg), outBound(revMsgLen), mockcpp::any())
        .will(returnValue(true));

    string agentId{"0"};
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId)
        .stubs()
        .with(mockcpp::any(), outBound(agentId))
        .will(invoke(RecvRemoteAgentIdStub));

    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));

    EXPECT_EQ(rankInfoDetectService_->GetConnections(true), HcclResult::HCCL_SUCCESS);

    auto res1 = rankInfoDetectService_->connSockets_.size();
    EXPECT_EQ(1, res1);
}

// PR 行为变更（单 root Init 路径）：首个连接握手失败后 continue 继续等待（旧实现 break 中止），
// 后续存活 rank 正常连入；半握手连接被 EraseIncompleteHandshakeConnection 摘除，
// 不计入已连接，也不把后续存活 rank 误报进失败列表
TEST_F(RankInfoDetectServiceTest, Ut_GetConnections_When_HandshakeFail_Expect_ContinueWaiting)
{
    SetupMockSocketAndTimeoutConfig();
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId).stubs().will(invoke(RecvRemoteAgentIdStub));
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))));

    // 第 1 个连接握手失败（RecvRemoteRankSize false → 摘除 + continue），后续 2 个连接握手成功
    u32 rankSize = 2;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .then(returnValue(false))
        .then(returnValue(true))
        .then(returnValue(true));

    // 连接 1（agentId "0"，握手失败）与后续存活连接（agentId "1"/"2"）均建链成功
    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::OK));

    EXPECT_EQ(rankInfoDetectService_->GetConnections(true), HcclResult::HCCL_SUCCESS);
    // 半握手对端 "0" 已摘除；存活 rank "1"/"2" 正常连入
    EXPECT_EQ(rankInfoDetectService_->connSockets_.size(), 2U);
    EXPECT_EQ(rankInfoDetectService_->connSockets_.count("0"), 0U);
    // 预期连接全部收齐：不生成失败列表（存活 rank 未被误报为失败）
    EXPECT_TRUE(rankInfoDetectService_->failedAgentIdList_.empty());
}

// PR 行为变更：握手失败仅 Shutdown 中止 accept——shutdown_ 置位后握手失败直接 break，
// 零连接成功时 GetConnections 返回 HCCL_E_INTERNAL（半握手连接已摘除）
TEST_F(RankInfoDetectServiceTest, Ut_GetConnections_When_ShutdownAtHandshakeFail_Expect_Abort)
{
    SetupMockSocketAndTimeoutConfig();
    rankInfoDetectService_->shutdown_ = true;
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId).stubs().will(invoke(RecvRemoteAgentIdStub));
    u32 rankSize = 2;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(false));
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))));
    MOCKER_CPP(&Socket::GetStatus).stubs().then(returnValue((SocketStatus)SocketStatus::OK));

    EXPECT_EQ(rankInfoDetectService_->GetConnections(true), HcclResult::HCCL_E_INTERNAL);
    EXPECT_TRUE(rankInfoDetectService_->connSockets_.empty());
}

TEST_F(RankInfoDetectServiceTest, Ut_IsRemoteIpInWhiteList_When_WhitelistDisable_Expect_True)
{
    MockHostNicConfigWhitelistDisable(true);
    EXPECT_TRUE(rankInfoDetectService_->IsRemoteIpInWhiteList(IpAddress("10.0.0.1")));
}

TEST_F(RankInfoDetectServiceTest, Ut_IsRemoteIpInWhiteList_When_WhitelistEmpty_Expect_False)
{
    MockHostNicConfigWhitelistDisable(false);
    rankInfoDetectService_->wlistInfo_.clear();
    EXPECT_FALSE(rankInfoDetectService_->IsRemoteIpInWhiteList(IpAddress("10.0.0.1")));
}

TEST_F(RankInfoDetectServiceTest, Ut_IsRemoteIpInWhiteList_When_RemoteIpInWhiteList_Expect_True)
{
    MockHostNicConfigWhitelistDisable(false);
    RaSocketWhitelist entry;
    entry.remoteIp = IpAddress("10.0.0.1");
    entry.tag = "rank_info_test";
    rankInfoDetectService_->wlistInfo_ = {entry};
    EXPECT_TRUE(rankInfoDetectService_->IsRemoteIpInWhiteList(IpAddress("10.0.0.1")));
}

TEST_F(RankInfoDetectServiceTest, Ut_IsRemoteIpInWhiteList_When_RemoteIpNotInWhiteList_Expect_False)
{
    MockHostNicConfigWhitelistDisable(false);
    RaSocketWhitelist entry;
    entry.remoteIp = IpAddress("10.0.0.1");
    entry.tag = "rank_info_test";
    rankInfoDetectService_->wlistInfo_ = {entry};
    EXPECT_FALSE(rankInfoDetectService_->IsRemoteIpInWhiteList(IpAddress("10.0.0.2")));
}

TEST_F(RankInfoDetectServiceTest, Ut_GetConnections_When_RemoteIpNotInWhiteList_Expect_DiscardAndAcceptNext)
{
    SetupMockSocketAndTimeoutConfig();

    // 时钟前3次为 0（正常路径在此完成第2条连接建链后退出），之后推进到超时（兜底：避免异常场景死循环）
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(10))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(10))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(10))));

    // 第1条连接白名单校验不通过（丢弃后继续等待），第2条校验通过并完成建链
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().then(returnValue(false)).then(returnValue(true));

    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));

    u32 rankSize = 1;
    void* msg = &rankSize;
    u64 msgLen = sizeof(u32);
    u64& revMsgLen = msgLen;
    MOCKER_CPP(&SocketAgent::RecvMsg)
        .stubs()
        .with(outBound(msg), outBound(revMsgLen), mockcpp::any())
        .will(returnValue(true));

    string agentId{"0"};
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId)
        .stubs()
        .with(mockcpp::any(), outBound(agentId))
        .will(invoke(RecvRemoteAgentIdStub));

    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));

    rankInfoDetectService_->GetConnections(true);

    EXPECT_EQ(1, rankInfoDetectService_->connSockets_.size());
}

TEST_F(RankInfoDetectServiceTest, Ut_GetConnections_When_AllRemoteIpNotInWhiteList_Expect_E_INTERNAL)
{
    SetupMockSocketAndTimeoutConfig();

    // 第1轮拒绝后时间推进到超时，外层循环退出
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(10))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(10))));

    // 所有连接白名单校验均不通过
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().will(returnValue(false));
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));

    // 顶层超时退出会调用 RPT_INPUT_ERR，需打桩避免触达真实实现
    MOCKER(RptInputErr).stubs().will(invoke(StubServiceRptInputErr));

    // GetConnections 以错误码返回失败：无成功连接时返回 HCCL_E_INTERNAL
    EXPECT_EQ(rankInfoDetectService_->GetConnections(true), HCCL_E_INTERNAL);

    EXPECT_EQ(0, rankInfoDetectService_->connSockets_.size());
}

// ============================ scalable 主路径用例（补充） ============================

// 按 groupSize 收满本组连接返回成功
TEST_F(RankInfoDetectServiceTest, Ut_GetConnectionsScalable_When_GroupComplete_Expect_Success)
{
    SetupMockSocketAndTimeoutConfig();
    // 固定时钟：linkTimeOut 被 mock 为 1s，若用真实时钟，首轮 remaining = 1s - 微秒级耗时
    // 经 duration_cast<seconds> 截断为 0，会被误判为“超时耗尽”；固定时钟使 usedTime 恒为 0。
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))));
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId).stubs().will(invoke(RecvRemoteAgentIdStub));
    u32 rankSize = 8;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));
    // groupSize=2：每建链一条前先 CONNECTING 一次再 OK（覆盖等待分支）
    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK));

    std::chrono::steady_clock::time_point acceptStartTime{};
    EXPECT_EQ(rankInfoDetectService_->GetConnectionsScalable(2, acceptStartTime), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rankInfoDetectService_->connSockets_.size(), 2U);
}

// 一直无连接时收敛为超时：AcceptGroupConnections 对零连接同样返回 SUSPENDING（整组失败列表已生成），
// GetConnectionsScalable 原样透传——SetupScalable 放行至 mesh（空帧 + 失败列表跨组传播）、
// 跳过本组广播（无接收方）；本函数不再做 count==0 收敛
TEST_F(RankInfoDetectServiceTest, Ut_GetConnectionsScalable_When_Timeout_Expect_E_SUSPENDING)
{
    SetupMockSocketAndTimeoutConfig();
    // 首轮读取起始时间（0s），次轮读取执行时间（2s >= linkTimeOut 1s），触发 AcceptGroupConnections 超时分支
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(2))));
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::CONNECTING));

    g_serviceCapturedErrorCode.clear();
    MOCKER(RptInputErr).stubs().will(invoke(StubServiceRptInputErr));

    std::chrono::steady_clock::time_point acceptStartTime{};
    EXPECT_EQ(rankInfoDetectService_->GetConnectionsScalable(2, acceptStartTime), HcclResult::HCCL_E_SUSPENDING);
    EXPECT_EQ(rankInfoDetectService_->connSockets_.size(), 0U);
    // 整组失败区间已生成（[0,2)），供 mesh 空帧跨组传播
    EXPECT_EQ(rankInfoDetectService_->failedAgentIdList_, "0-1");
}

// 白名单校验不通过时丢弃该连接并继续等待下一条（scalable 场景）
TEST_F(RankInfoDetectServiceTest, Ut_GetConnectionsScalable_When_RemoteIpNotInWhiteList_Expect_DiscardAndAcceptNext)
{
    SetupMockSocketAndTimeoutConfig();
    // 固定时钟：linkTimeOut 被 mock 为 1s，若用真实时钟，首轮 remaining = 1s - 微秒级耗时
    // 经 duration_cast<seconds> 截断为 0，会被误判为“超时耗尽”；固定时钟使 usedTime 恒为 0。
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))));
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().then(returnValue(false)).then(returnValue(true));
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId).stubs().will(invoke(RecvRemoteAgentIdStub));
    u32 rankSize = 8;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));

    std::chrono::steady_clock::time_point acceptStartTime{};
    EXPECT_EQ(rankInfoDetectService_->GetConnectionsScalable(1, acceptStartTime), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rankInfoDetectService_->connSockets_.size(), 1U);
}

// M1：组内 accept 部分失败（≥1 个连接成功）返回 SUSPENDING 并生成本组压缩失败列表，广播由后续步骤携带
TEST_F(RankInfoDetectServiceTest, Ut_AcceptGroupConnections_When_PartialFail_Expect_SuspendingWithFailedList)
{
    SetupMockSocketAndTimeoutConfig();
    // 固定时钟：构造 startTime（0s）→ 首轮未超时（0s）→ 次轮超时（2s ≥ 1s）
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(2))));
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId).stubs().will(invoke(RecvRemoteAgentIdStub));
    u32 rankSize = 8;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));
    // 首条连接成功（rank 0），随后耗尽超时 → 剩余 1 条未连接
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    g_serviceCapturedErrorCode.clear();
    MOCKER(RptInputErr).stubs().will(invoke(StubServiceRptInputErr));

    u32 expectedSocketNum = 2;
    auto startTime = std::chrono::steady_clock::now();
    EXPECT_EQ(
        rankInfoDetectService_->AcceptGroupConnections(
            hccpSocketHandle, 60007, startTime, std::chrono::seconds(1), expectedSocketNum),
        HcclResult::HCCL_E_SUSPENDING);
    EXPECT_EQ(rankInfoDetectService_->connSockets_.size(), 1U);
    EXPECT_EQ(rankInfoDetectService_->failedAgentIdList_, "1");
    EXPECT_EQ(rankInfoDetectService_->totalRankNum_, 8U); // agent 上报的全局 nRanks 存成员，供缺席反推
}

// M2/FR4：失败列表区间压缩编码（连续段 a-b、单点 a、段间逗号）
TEST_F(RankInfoDetectServiceTest, Ut_FailedConnectionAgentIdString_When_PartialFail_Expect_CompressedSegments)
{
    for (u32 rankId : {0U, 1U, 2U, 3U, 6U}) {
        rankInfoDetectService_->connSockets_[std::to_string(rankId)] = nullptr;
    }
    EXPECT_EQ(rankInfoDetectService_->FailedConnectionAgentIdString(8), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rankInfoDetectService_->failedAgentIdList_, "4-5,7");
}

// 26W 卡（65536×4 上限）全失败单段区间（SRS 4.2：消息增量 ≤64 字节的核心收益场景）
TEST_F(RankInfoDetectServiceTest, Ut_FailedConnectionAgentIdString_When_AllFail_Expect_SingleRangeSegment)
{
    EXPECT_EQ(rankInfoDetectService_->FailedConnectionAgentIdString(262144), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rankInfoDetectService_->failedAgentIdList_, "0-262143");
}

// scalable 组偏移：仅统计本组区间 [groupFirstRank_, groupFirstRank_+rankSize)，组外 rank 不误报
TEST_F(RankInfoDetectServiceTest, Ut_FailedConnectionAgentIdString_When_GroupOffset_Expect_GroupScopedList)
{
    rankInfoDetectService_->groupFirstRank_ = 8;
    for (u32 rankId : {9U, 12U, 13U}) {
        rankInfoDetectService_->connSockets_[std::to_string(rankId)] = nullptr;
    }
    EXPECT_EQ(rankInfoDetectService_->FailedConnectionAgentIdString(8), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rankInfoDetectService_->failedAgentIdList_, "8,10-11,14-15");
}

// 非法/非本组 agentId 告警跳过、不中断统计：合法连接正常标记，失败列表不被空/截断污染
TEST_F(RankInfoDetectServiceTest, Ut_FailedConnectionAgentIdString_When_InvalidAgentId_Expect_SkipAndContinue)
{
    rankInfoDetectService_->connSockets_["0"] = nullptr;
    rankInfoDetectService_->connSockets_["2"] = nullptr;
    // 尾部非数字（SalStrToULong 仅消费数字前缀 "1"）：不得计为已连接，rank1 进失败列表
    rankInfoDetectService_->connSockets_["1abc"] = nullptr;
    rankInfoDetectService_->connSockets_["not_a_number"] = nullptr; // 解析失败
    rankInfoDetectService_->connSockets_["999"] = nullptr;          // 越界（rankSize=8）
    EXPECT_EQ(rankInfoDetectService_->FailedConnectionAgentIdString(8), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rankInfoDetectService_->failedAgentIdList_, "1,3-7");
}

// 失败列表统计异常时保守兜底为整组区间（避免把空/截断列表当成本组无失败广播）
TEST_F(RankInfoDetectServiceTest, Ut_AcceptGroupConnections_When_FailedListGenFail_Expect_FallbackWholeGroupRange)
{
    SetupMockSocketAndTimeoutConfig();
    // 固定时钟：构造 startTime（0s）→ 首轮未超时（0s）→ 次轮超时（2s ≥ 1s）
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(2))));
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId).stubs().will(invoke(RecvRemoteAgentIdStub));
    u32 rankSize = 8;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    g_serviceCapturedErrorCode.clear();
    MOCKER(RptInputErr).stubs().will(invoke(StubServiceRptInputErr));
    // 模拟失败列表统计异常：调用点应兜底为整组区间而非空列表
    MOCKER_CPP(&RankInfoDetectService::FailedConnectionAgentIdString)
        .stubs()
        .will(returnValue(HcclResult::HCCL_E_INTERNAL));

    u32 expectedSocketNum = 2;
    auto startTime = std::chrono::steady_clock::now();
    EXPECT_EQ(
        rankInfoDetectService_->AcceptGroupConnections(
            hccpSocketHandle, 60007, startTime, std::chrono::seconds(1), expectedSocketNum),
        HcclResult::HCCL_E_SUSPENDING);
    EXPECT_EQ(rankInfoDetectService_->connSockets_.size(), 1U);
    EXPECT_EQ(rankInfoDetectService_->failedAgentIdList_, "0-1"); // 本组 [0,2) 整组区间
}

// 兜底区间 u32 回绕守卫：firstRank 接近 UINT32_MAX 且 groupSize 较大时 first+groupSize-1 回绕，
// 追加会被 AppendFailedIdSegment 静默截断为单点——守卫跳过追加保持空列表，由 agent 侧 rankCount
// 校验兜底（错误防御数据来源，groupFirstRank_ 由 Get 阶段本地推导，正常不越界）
TEST_F(RankInfoDetectServiceTest, Ut_AcceptGroupConnections_When_FallbackRangeWraps_Expect_SkipAppend)
{
    SetupMockSocketAndTimeoutConfig();
    // 固定时钟：startTime（0s）→ 首轮未超时（0s）→ 次轮超时（2s ≥ 1s）
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(2))));
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId).stubs().will(invoke(RecvRemoteAgentIdStub));
    u32 rankSize = 8;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    g_serviceCapturedErrorCode.clear();
    MOCKER(RptInputErr).stubs().will(invoke(StubServiceRptInputErr));
    // 模拟失败列表统计异常触发兜底 + 构造回绕区间（firstRank=UINT32_MAX-1, groupSize=4 → last 回绕）
    rankInfoDetectService_->groupFirstRank_ = UINT32_MAX - 1;
    MOCKER_CPP(&RankInfoDetectService::FailedConnectionAgentIdString)
        .stubs()
        .will(returnValue(HcclResult::HCCL_E_INTERNAL));

    // groupSize 取自 expectedSocketNum（service.cc groupSize = expectedSocketNum），须传 4 使
    // firstRank=UINT32_MAX-1 > UINT32_MAX-3 命中守卫跳过；传 2 时守卫条件成立会走正常追加分支
    u32 expectedSocketNum = 4;
    auto startTime = std::chrono::steady_clock::now();
    EXPECT_EQ(
        rankInfoDetectService_->AcceptGroupConnections(
            hccpSocketHandle, 60007, startTime, std::chrono::seconds(1), expectedSocketNum),
        HcclResult::HCCL_E_SUSPENDING);
    // 回绕区间被守卫跳过：失败列表保持空（不追加被截断的单点），由 agent 侧 rankCount 校验兜底
    EXPECT_TRUE(rankInfoDetectService_->failedAgentIdList_.empty());
}

// M3：mesh 回调失败（参数类错误）不阻断广播，SetupScalable 返回广播结果（快速广播，遗言可达性优先）
TEST_F(RankInfoDetectServiceTest, Ut_SetupScalable_When_MeshMergeCallbackFail_Expect_StillBroadcast)
{
    SetupMockSocketAndTimeoutConfig();
    // 固定时钟：linkTimeOut 被 mock 为 1s，若用真实时钟，首轮 remaining = 1s - 微秒级耗时
    // 经 duration_cast<seconds> 截断为 0，会被误判为“超时耗尽”；固定时钟使 usedTime 恒为 0。
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))));
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId).stubs().will(invoke(RecvRemoteAgentIdStub));
    u32 rankSize = 8;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));
    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK));

    // 构造组内 rankTable 消息（[ranktable][step]），供 GetRankTable 解析
    RankTableInfo localRankTable;
    localRankTable.version = "2.0";
    localRankTable.rankCount = 1;
    hccl::NewRankInfo rankInfo{};
    rankInfo.rankId = 0;
    rankInfo.rankLevelInfos.emplace_back(RankLevelInfo{});
    localRankTable.ranks.emplace_back(rankInfo);
    BinaryStream binaryStream;
    ASSERT_EQ(localRankTable.GetBinStream(true, binaryStream), HCCL_SUCCESS);
    binaryStream << rankInfoDetectService_->currentStep_;
    vector<char> rankInfoMsg;
    binaryStream.Dump(rankInfoMsg);
    MOCKER(aclrtMallocHostWithCfg).stubs().will(returnValue(1));
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue((void*)rankInfoMsg.data()));
    void* msg = rankInfoMsg.data();
    u64 msgLen = rankInfoMsg.size();
    u64& revMsgLen = msgLen;
    MOCKER_CPP(&SocketAgent::RecvMsg)
        .stubs()
        .with(outBoundP(msg, msgLen), outBound(revMsgLen), mockcpp::any())
        .will(returnValue(true));

    MOCKER_CPP(&RankInfoDetectService::BroadcastRankTable).expects(once()).will(returnValue(HCCL_SUCCESS));
    std::function<HcclResult(RankTableInfo&, std::string&, u32, std::chrono::steady_clock::time_point)> callback
        = [](RankTableInfo&, std::string&, u32, std::chrono::steady_clock::time_point) -> HcclResult {
        return HCCL_E_NETWORK;
    };
    EXPECT_EQ(rankInfoDetectService_->SetupScalable(1, callback), HcclResult::HCCL_SUCCESS);
    // 回调失败后数据自洽（半合并防护）：失败列表不被跨组内容污染（本组 groupSize=1 全连上 → 空）、
    // rankTable 保持本组表（rankCount=1，未被半合并/未合并子表污染）——未合并组状态未知不臆造进
    // 失败列表，agent 侧 rankCount 校验兜底报错
    EXPECT_TRUE(rankInfoDetectService_->failedAgentIdList_.empty());
    EXPECT_EQ(rankInfoDetectService_->rankTable_.rankCount, 1U);
    EXPECT_EQ(rankInfoDetectService_->rankTable_.ranks.size(), 1U);
}

// 广播本身失败仍置错误（M4：广播失败/零连接 → ERROR，服务端状态不收敛 IDLE）
TEST_F(RankInfoDetectServiceTest, Ut_SetupScalable_When_BroadcastFail_Expect_ErrCodePassthrough)
{
    SetupMockSocketAndTimeoutConfig();
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))));
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId).stubs().will(invoke(RecvRemoteAgentIdStub));
    u32 rankSize = 8;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));
    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK));

    RankTableInfo localRankTable;
    localRankTable.version = "2.0";
    localRankTable.rankCount = 1;
    hccl::NewRankInfo rankInfo{};
    rankInfo.rankId = 0;
    rankInfo.rankLevelInfos.emplace_back(RankLevelInfo{});
    localRankTable.ranks.emplace_back(rankInfo);
    BinaryStream binaryStream;
    ASSERT_EQ(localRankTable.GetBinStream(true, binaryStream), HCCL_SUCCESS);
    binaryStream << rankInfoDetectService_->currentStep_;
    vector<char> rankInfoMsg;
    binaryStream.Dump(rankInfoMsg);
    MOCKER(aclrtMallocHostWithCfg).stubs().will(returnValue(1));
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue((void*)rankInfoMsg.data()));
    void* msg = rankInfoMsg.data();
    u64 msgLen = rankInfoMsg.size();
    u64& revMsgLen = msgLen;
    MOCKER_CPP(&SocketAgent::RecvMsg)
        .stubs()
        .with(outBoundP(msg, msgLen), outBound(revMsgLen), mockcpp::any())
        .will(returnValue(true));

    MOCKER_CPP(&RankInfoDetectService::BroadcastRankTable).expects(once()).will(returnValue(HCCL_E_PARA));
    std::function<HcclResult(RankTableInfo&, std::string&, u32, std::chrono::steady_clock::time_point)> callback
        = [](RankTableInfo&, std::string&, u32, std::chrono::steady_clock::time_point) -> HcclResult {
        return HCCL_E_NETWORK;
    }; // mesh 回调失败也不阻断
    EXPECT_EQ(rankInfoDetectService_->SetupScalable(1, callback), HcclResult::HCCL_E_PARA);
}

// 部分失败（GetConnectionsScalable 返回 SUSPENDING）不于第 1 步中止：收集存活子集（GetRankTable
// 仅遍历已连接 agent），继续执行 mesh 回调与广播（遗言可达性优先），广播完成后向上透传 SUSPENDING
// （worker 线程据此把 server 标记 ERROR，仍在 WaitComplete 轮询的 agent 快速失败）
TEST_F(RankInfoDetectServiceTest, Ut_SetupScalable_When_PartialConnectFail_Expect_BroadcastThenSuspending)
{
    SetupMockSocketAndTimeoutConfig();
    MOCKER_CPP(&RankInfoDetectService::GetConnectionsScalable).stubs().will(returnValue(HcclResult::HCCL_E_SUSPENDING));
    // 部分失败语义：塞一个存活连接（zeroConnected 判定依据 connSockets_ 非空，空表会走跳过广播分支）
    {
        std::lock_guard<std::mutex> lock(rankInfoDetectService_->connMtx_);
        rankInfoDetectService_->connSockets_["0"] = nullptr;
    }
    // 部分失败仍需收集存活子集：跳过收集会因空表 GetBinStream 报 E_PARA 使广播发不出去
    MOCKER_CPP(&RankInfoDetectService::GetRankTable).expects(once()).will(returnValue(HCCL_SUCCESS));
    bool meshCallbackInvoked = false;
    std::function<HcclResult(RankTableInfo&, std::string&, u32, std::chrono::steady_clock::time_point)> callback =
        [&meshCallbackInvoked](RankTableInfo&, std::string&, u32, std::chrono::steady_clock::time_point) -> HcclResult {
        meshCallbackInvoked = true;
        return HCCL_SUCCESS;
    };
    MOCKER_CPP(&RankInfoDetectService::BroadcastRankTable).expects(once()).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(rankInfoDetectService_->SetupScalable(4, callback), HcclResult::HCCL_E_SUSPENDING);
    EXPECT_TRUE(meshCallbackInvoked);
}

// 零连接（本组全失联）编排：SUSPENDING 放行——mesh 回调仍执行（空表手工空帧 + 整组失败列表
// 跨组传播，由其他 root 广播给其存活 agent），跳过本组广播（无接收方、空表 GetBinStream 必败），
// 收集为空操作，最终透传 SUSPENDING
TEST_F(RankInfoDetectServiceTest, Ut_SetupScalable_When_ZeroConnected_Expect_MeshOnlyThenSuspending)
{
    SetupMockSocketAndTimeoutConfig();
    // SetupScalable 入口会 clear 失败列表（防跨轮残留），生产路径由 AcceptGroupConnections 在
    // clear 之后生成——mock 需带副作用回填，模拟该时序。
    // invoke 只接受函数指针（mockcpp ProcStub 模板约束），lambda 前缀 + 强制无捕获 lambda 转换
    MOCKER_CPP(&RankInfoDetectService::GetConnectionsScalable)
        .stubs()
        .will(invoke(
            +[](RankInfoDetectService* self, u32,
                std::chrono::steady_clock::time_point& acceptStartTime) -> HcclResult {
                self->failedAgentIdList_ = "0-3";
                acceptStartTime = std::chrono::steady_clock::now(); // 桩值不影响断言（mesh 已打桩）
                return HCCL_E_SUSPENDING;
            }));
    // 零连接：收集为空操作（无已连接 agent 可遍历）
    MOCKER_CPP(&RankInfoDetectService::GetRankTable).expects(once()).will(returnValue(HCCL_SUCCESS));
    bool meshCallbackInvoked = false;
    std::function<HcclResult(RankTableInfo&, std::string&, u32, std::chrono::steady_clock::time_point)> callback
        = [&meshCallbackInvoked](
              RankTableInfo& rankTable, std::string& failedList, u32,
              std::chrono::steady_clock::time_point) -> HcclResult {
        meshCallbackInvoked = true;
        // mesh 回调拿到整组失败列表（空表由 RootMeshAllGather 手工空帧承载，非 service 职责）
        EXPECT_EQ(failedList, "0-3");
        EXPECT_TRUE(rankTable.ranks.empty());
        return HCCL_SUCCESS;
    };
    // 零连接跳过本组广播（无接收方）
    MOCKER_CPP(&RankInfoDetectService::BroadcastRankTable).expects(never());

    // 零连接状态（失败列表由上述 mock 副作用在 clear 之后生成）
    {
        std::lock_guard<std::mutex> lock(rankInfoDetectService_->connMtx_);
        rankInfoDetectService_->connSockets_.clear();
    }

    EXPECT_EQ(rankInfoDetectService_->SetupScalable(4, callback), HcclResult::HCCL_E_SUSPENDING);
    EXPECT_TRUE(meshCallbackInvoked);
}

// SUSPENDING 遗言路径真实收集回归（检视缺陷：跳过收集后空表广播）：走真实 GetConnectionsScalable
// （4 连接中 1 个成功后 socket 异常 → 部分失败）+ 真实 GetRankTable（仅广播打桩——dispatcher 内部
// epoll/线程不在 UT 环境承载范围），断言收集后 rankTable_ 为存活子集且可序列化——
// 空表会使 mesh 交换与广播的 GetBinStream 报 E_PARA，遗言广播整体发不出去
TEST_F(RankInfoDetectServiceTest, Ut_SetupScalable_When_PartialFail_Expect_SurvivorTableSerializable)
{
    SetupMockSocketAndTimeoutConfig();
    // 固定时钟：避免 1s 预算被 duration_cast 截断为 0 误判“超时耗尽”
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))));
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId).stubs().will(invoke(RecvRemoteAgentIdStub));
    u32 rankSize = 8;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));
    // 4 连接中首个成功（rank 0），随后 socket 异常退出循环 → 剩余 3 个未连接 → 部分失败
    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::INVALID));

    // 存活 agent 的子表消息（握手后已发出，GetRankTable 从已连接 socket 读取）
    RankTableInfo localRankTable;
    localRankTable.version = "2.0";
    localRankTable.rankCount = 1;
    hccl::NewRankInfo rankInfo{};
    rankInfo.rankId = 0;
    rankInfo.rankLevelInfos.emplace_back(RankLevelInfo{});
    localRankTable.ranks.emplace_back(rankInfo);
    BinaryStream binaryStream;
    ASSERT_EQ(localRankTable.GetBinStream(true, binaryStream), HCCL_SUCCESS);
    binaryStream << rankInfoDetectService_->currentStep_;
    vector<char> rankInfoMsg;
    binaryStream.Dump(rankInfoMsg);
    MOCKER(aclrtMallocHostWithCfg).stubs().will(returnValue(1));
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue((void*)rankInfoMsg.data()));
    void* msg = rankInfoMsg.data();
    u64 msgLen = rankInfoMsg.size();
    u64& revMsgLen = msgLen;
    MOCKER_CPP(&SocketAgent::RecvMsg)
        .stubs()
        .with(outBoundP(msg, msgLen), outBound(revMsgLen), mockcpp::any())
        .will(returnValue(true));

    MOCKER_CPP(&RankInfoDetectService::BroadcastRankTable).expects(once()).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(rankInfoDetectService_->SetupScalable(4, nullptr), HcclResult::HCCL_E_SUSPENDING);
    // 存活子集非空且可序列化（GetBinStream 对空表返回 E_PARA——检视指出的精确失败点）
    EXPECT_EQ(rankInfoDetectService_->rankTable_.rankCount, 1U);
    BinaryStream outStream;
    EXPECT_EQ(rankInfoDetectService_->rankTable_.GetBinStream(true, outStream), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rankInfoDetectService_->failedAgentIdList_, "1-3"); // 缺席 rank 区间压缩
}

// 失败列表跨轮清理（scalable 路径）——第 1 轮部分失败的列表不残留到第 2 轮成功广播
TEST_F(RankInfoDetectServiceTest, Ut_SetupScalable_When_SecondRoundAllSuccess_Expect_NoResidualFailedList)
{
    SetupMockSocketAndTimeoutConfig();
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId).stubs().will(invoke(RecvRemoteAgentIdStub));
    u32 rankSize = 8;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));
    // 跨轮清理关注点在失败列表，表收集与广播打桩简化
    MOCKER_CPP(&RankInfoDetectService::GetRankTable).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankInfoDetectService::BroadcastRankTable).stubs().will(returnValue(HCCL_SUCCESS));
    // 固定时钟：usedTime 恒为 0，避免 1s 预算被 duration_cast 截断为 0 误判“超时耗尽”（两轮共用恒定值）
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))));

    // 第 1 轮：4 连接中 1 个成功后 socket 异常退出循环 → 部分失败（SUSPENDING）：跳过组内收集、
    // 广播遗言后透传 SUSPENDING
    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::INVALID))
        // 第 2 轮：4 次连接全部成功
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::OK));

    EXPECT_EQ(rankInfoDetectService_->SetupScalable(4, nullptr), HcclResult::HCCL_E_SUSPENDING);
    EXPECT_EQ(rankInfoDetectService_->connSockets_.size(), 1U);
    EXPECT_EQ(rankInfoDetectService_->failedAgentIdList_, "1-3");

    // 第 2 轮复位连接（生产中每轮 worker 新建 service，此处模拟新一轮）
    g_recvAgentIdNextRank = 0;
    {
        std::lock_guard<std::mutex> lock(rankInfoDetectService_->connMtx_);
        rankInfoDetectService_->connSockets_.clear();
    }
    EXPECT_EQ(rankInfoDetectService_->SetupScalable(4, nullptr), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rankInfoDetectService_->connSockets_.size(), 4U);
    EXPECT_TRUE(rankInfoDetectService_->failedAgentIdList_.empty()); // 无第 1 轮残留
}

// 失败列表跨轮清理（单 root Setup 路径，FR1.1/FR7 回归防残留）
TEST_F(RankInfoDetectServiceTest, Ut_Setup_When_SecondRoundAllSuccess_Expect_NoResidualFailedList)
{
    SetupMockSocketAndTimeoutConfig();
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId).stubs().will(invoke(RecvRemoteAgentIdStub));
    u32 rankSize = 8;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));
    MOCKER_CPP(&RankInfoDetectService::GetRankTable).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankInfoDetectService::BroadcastRankTable).stubs().will(returnValue(HCCL_SUCCESS));
    // 固定时钟：usedTime 恒为 0，两轮均不触发超时（断点由 GetStatus 序列控制）
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))));

    // 第 1 轮：首连接成功（expectedSocketNum 被上报 rankSize=8 重写），随后 socket 异常退出 → 部分失败 "1-7"
    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::INVALID))
        // 第 2 轮：8 次连接全部成功
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::OK));

    EXPECT_EQ(rankInfoDetectService_->Setup(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rankInfoDetectService_->connSockets_.size(), 1U);
    EXPECT_EQ(rankInfoDetectService_->failedAgentIdList_, "1-7");

    // 第 2 轮复位连接，全部成功 → 广播列表为空，无第 1 轮残留
    g_recvAgentIdNextRank = 0;
    {
        std::lock_guard<std::mutex> lock(rankInfoDetectService_->connMtx_);
        rankInfoDetectService_->connSockets_.clear();
    }
    EXPECT_EQ(rankInfoDetectService_->Setup(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rankInfoDetectService_->connSockets_.size(), 8U);
    EXPECT_TRUE(rankInfoDetectService_->failedAgentIdList_.empty());
}

// 26W 卡 1% 稀疏失败（每第 100 个 rank 失败，2622 个单点）——压缩串长度量级与消息上限约束
TEST_F(RankInfoDetectServiceTest, Ut_FailedConnectionAgentIdString_When_SparseFail_Expect_BoundedLength)
{
    const u32 rankSize = 262144;
    std::string expected;
    for (u32 rankId = 0; rankId < rankSize; rankId += 100) {
        if (!expected.empty()) {
            expected += ',';
        }
        expected += std::to_string(rankId);
    }
    for (u32 rankId = 0; rankId < rankSize; rankId++) {
        if (rankId % 100 == 0) {
            continue; // 该 rank 建链失败（未连接）
        }
        rankInfoDetectService_->connSockets_[std::to_string(rankId)] = nullptr;
    }
    EXPECT_EQ(rankInfoDetectService_->FailedConnectionAgentIdString(rankSize), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rankInfoDetectService_->failedAgentIdList_, expected);
    EXPECT_GT(rankInfoDetectService_->failedAgentIdList_.size(), size_t(10240)); // 稀疏失败退化为单点序列（~20KB 量级）
    // 生效上界是 mesh 失败列表上限（localFailedList_ 经 mesh 回调发送时由 SendMeshPartial 校验），而非子表缓冲上界
    EXPECT_LT(rankInfoDetectService_->failedAgentIdList_.size(), static_cast<size_t>(MAX_MESH_FAILED_LIST_LEN));
}

// ════════════════════════════════════════════════════════════════════════════
// 持久化状态机与缩容路径 UT 覆盖
// ════════════════════════════════════════════════════════════════════════════

// 验证 Grow 线程就绪后才能触发本轮收集
TEST_F(RankInfoDetectServiceTest, Ut_TriggerGrowCollect_When_Ready_Expect_Running)
{
    EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectService_->MarkGrowLoopReady());
    EXPECT_EQ(RankInfoDetectService::GrowState::IDLE, rankInfoDetectService_->GetGrowState());

    EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectService_->TriggerGrowCollect(1));
    EXPECT_EQ(RankInfoDetectService::GrowState::RUNNING, rankInfoDetectService_->GetGrowState());
}

// 验证 Grow 生命周期中只有完成态/等待态允许触发下一轮
TEST_F(RankInfoDetectServiceTest, Ut_IsReadyForGrow_When_StateTransitions_ExpectExpected)
{
    EXPECT_FALSE(rankInfoDetectService_->IsReadyForGrow());

    EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectService_->MarkGrowLoopReady());
    EXPECT_TRUE(rankInfoDetectService_->IsReadyForGrow());

    EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectService_->TriggerGrowCollect(1));
    EXPECT_FALSE(rankInfoDetectService_->IsReadyForGrow());

    EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectService_->FinishGrowCollect(HCCL_SUCCESS));
    EXPECT_TRUE(rankInfoDetectService_->IsReadyForGrow());

    rankInfoDetectService_->MarkGrowThreadExited(HCCL_E_INTERNAL);
    EXPECT_FALSE(rankInfoDetectService_->IsReadyForGrow());
}

// 验证 Grow 失败后进入 ERROR 终态，等待和后续触发都不再复用 service
TEST_F(RankInfoDetectServiceTest, Ut_GrowCollect_When_Fail_Expect_ErrorTerminal)
{
    EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectService_->MarkGrowLoopReady());
    EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectService_->TriggerGrowCollect(1));
    rankInfoDetectService_->MarkGrowThreadExited(HCCL_E_INTERNAL);

    // ERROR 为当前 service 终态：等待接口只返回错误，不消费状态；后续 Grow 直接失败。
    RankTableInfo rankTable;
    EXPECT_EQ(HCCL_E_INTERNAL, rankInfoDetectService_->WaitForCollectComplete(rankTable));
    EXPECT_EQ(RankInfoDetectService::GrowState::ERROR, rankInfoDetectService_->GetGrowState());
    EXPECT_EQ(HCCL_E_INTERNAL, rankInfoDetectService_->TriggerGrowCollect(1));
    EXPECT_EQ(RankInfoDetectService::GrowState::ERROR, rankInfoDetectService_->GetGrowState());
    EXPECT_FALSE(rankInfoDetectService_->IsSupportGrowLoop());
}

TEST_F(RankInfoDetectServiceTest, Ut_GrowCollect_When_NetLayerCheckFails_Expect_BroadcastThenReject)
{
    MOCKER_CPP(&RankInfoDetectService::GetConnections).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankInfoDetectService::GetRankTableForGrow).stubs().will(invoke(+[](RankInfoDetectService* service) {
        service->rankTable_.rankCount = 2;
        return HCCL_SUCCESS;
    }));
    MOCKER_CPP(&RankInfoDetectService::BroadcastRankTableForGrow)
        .expects(exactly(2))
        .will(invoke(+[](RankInfoDetectService* service) {
            EXPECT_EQ(RankInfoDetectService::GrowState::RUNNING, service->GetGrowState());
            EXPECT_EQ(2U, service->rankTable_.rankCount);
            service->currentStep_ = 1;
            return HCCL_SUCCESS;
        }));
    MOCKER_CPP(&RankInfoDetectService::RollbackNewConnections)
        .expects(exactly(2))
        .will(invoke(+[](RankInfoDetectService* service) {
            EXPECT_EQ(1U, service->currentStep_); // 不能在错误广播前回滚连接。
            return HCCL_SUCCESS;
        }));
    for (HcclResult result : {HCCL_E_PARA, HCCL_E_NOT_SUPPORT}) {
        SCOPED_TRACE(result);
        rankInfoDetectService_->growState_ = RankInfoDetectService::GrowState::RUNNING;
        rankInfoDetectService_->rankTable_.rankCount = 1;
        rankInfoDetectService_->currentStep_ = 0;
        NetLayerCheckInfo report;
        report.version = NetLayerCheckInfo::VERSION;
        report.explicitSelection = 1;
        report.result = result;
        rankInfoDetectService_->netLayerCheckInfos_[0] = report;
        EXPECT_EQ(result, rankInfoDetectService_->SetupGrowLoop());
        EXPECT_EQ(1U, rankInfoDetectService_->rankTable_.rankCount);
        EXPECT_EQ(RankInfoDetectService::GrowState::ERROR, rankInfoDetectService_->GetGrowState());
        RankTableInfo rankTable;
        rankTable.rankCount = 99;
        EXPECT_EQ(result, rankInfoDetectService_->WaitForCollectComplete(rankTable));
        EXPECT_EQ(99U, rankTable.rankCount);
    }
}

// 验证 Shutdown 后 TriggerGrowCollect 返回 HCCL_E_INTERNAL
TEST_F(RankInfoDetectServiceTest, Ut_TriggerGrowCollect_When_Shutdown_Expect_E_INTERNAL)
{
    EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectService_->Shutdown());

    // Shutdown 后 TriggerGrowCollect 应返回失败
    EXPECT_EQ(HCCL_E_INTERNAL, rankInfoDetectService_->TriggerGrowCollect(1));
    EXPECT_EQ(RankInfoDetectService::GrowState::SHUTDOWN, rankInfoDetectService_->GetGrowState());
    EXPECT_FALSE(rankInfoDetectService_->IsSupportGrowLoop());
}

// 验证 Grow 线程仅可从 STARTING 发布 IDLE，终态不可重新就绪
TEST_F(RankInfoDetectServiceTest, Ut_MarkGrowLoopReady_When_Starting_Expect_IdleThenReject)
{
    EXPECT_EQ(RankInfoDetectService::GrowState::STARTING, rankInfoDetectService_->GetGrowState());
    EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectService_->MarkGrowLoopReady());
    EXPECT_EQ(RankInfoDetectService::GrowState::IDLE, rankInfoDetectService_->GetGrowState());
    EXPECT_EQ(HCCL_E_INTERNAL, rankInfoDetectService_->MarkGrowLoopReady());

    rankInfoDetectService_->MarkGrowThreadExited(HCCL_E_INTERNAL);
    EXPECT_EQ(RankInfoDetectService::GrowState::ERROR, rankInfoDetectService_->GetGrowState());
    EXPECT_EQ(HCCL_E_INTERNAL, rankInfoDetectService_->MarkGrowLoopReady());
    EXPECT_EQ(RankInfoDetectService::GrowState::ERROR, rankInfoDetectService_->GetGrowState());
}

// 验证同一常驻 service 连续两轮 Grow 均复用原 serverSocket
TEST_F(RankInfoDetectServiceTest, Ut_GrowStateMachine_When_TwoRounds_ExpectSocketRetained)
{
    auto serverSocket = rankInfoDetectService_->serverSocket_;
    ASSERT_NE(serverSocket, nullptr);

    EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectService_->MarkGrowLoopReady());
    for (uint32_t round = 0; round < 2; ++round) {
        EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectService_->TriggerGrowCollect(1));
        EXPECT_EQ(RankInfoDetectService::GrowState::RUNNING, rankInfoDetectService_->GetGrowState());
        EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectService_->FinishGrowCollect(HCCL_SUCCESS));
        EXPECT_EQ(RankInfoDetectService::GrowState::IDLE, rankInfoDetectService_->GetGrowState());
        EXPECT_EQ(rankInfoDetectService_->serverSocket_.get(), serverSocket.get());
    }
}

// 失败列表跨轮清理（grow 路径）：SetupGrowLoop 每轮起始 clear，
// Init/上一轮残留的失败列表不随本轮 grow 广播下发（FR1.1/FR7 回归防残留）
TEST_F(RankInfoDetectServiceTest, Ut_GrowLoop_When_RoundStart_Expect_FailedListCleared)
{
    SetupMockSocketAndTimeoutConfig();
    MOCKER_CPP(&RankInfoDetectService::IsRemoteIpInWhiteList).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId).stubs().will(invoke(RecvRemoteAgentIdStub));
    u32 rankSize = 2;
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));
    // 固定时钟：避免 1s 预算被 duration_cast 截断为 0 误判"超时耗尽"
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))));
    // 本用例只关注失败列表跨轮清理，表收集与广播打桩简化
    MOCKER_CPP(&RankInfoDetectService::GetRankTableForGrow).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankInfoDetectService::BroadcastRankTableForGrow).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    MOCKER_CPP(&Socket::Close).stubs();

    EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectService_->MarkGrowLoopReady());
    // 模拟 Init/上一轮残留的失败列表
    rankInfoDetectService_->failedAgentIdList_ = "3,5";

    std::thread growThread([&]() {
        rankInfoDetectService_->SetupGrowLoop();
    });
    EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectService_->TriggerGrowCollect(1));
    RankTableInfo rankTable;
    EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectService_->WaitForCollectComplete(rankTable));
    EXPECT_EQ(RankInfoDetectService::GrowState::IDLE, rankInfoDetectService_->GetGrowState());

    EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectService_->Shutdown());
    growThread.join();
    // 本轮 grow 连接全部成功：残留列表在轮起始被 clear，未随广播下发
    EXPECT_TRUE(rankInfoDetectService_->failedAgentIdList_.empty());
}
