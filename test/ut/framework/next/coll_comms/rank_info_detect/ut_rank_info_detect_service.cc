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

    // 把构造好的ranktableinfo转为vector<char> rankInfoMsg
    // 消息格式: [ranktable数据(n字节)][step(4字节)]
    BinaryStream binaryStream;
    EXPECT_EQ(localRankTable.GetBinStream(true, binaryStream), HCCL_SUCCESS);
    binaryStream << rankInfoDetectService_->currentStep_;

    // 字节流转换为vector<char>格式
    vector<char> rankInfoMsg;
    binaryStream.Dump(rankInfoMsg);

    // Mock UpdateRankTable方法
    MOCKER_CPP(&RankTableInfo::UpdateRankTable)
        .expects(exactly(1))
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // 6. 执行测试
    EXPECT_EQ(rankInfoDetectService_->ParseRankTable(rankInfoMsg, false), HcclResult::HCCL_SUCCESS);
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

TEST_F(RankInfoDetectServiceTest, Ut_ProcessScalableConnStatus_When_RankSizeMismatch_Expect_Failure)
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
    EXPECT_FALSE(rankInfoDetectService_->ProcessScalableConnStatus(
        rankInfoDetectService_->serverSocket_, SocketStatus::OK, groupSize, expectedSocketNum, totalRankNum,
        isFirstAcceptTimeOut));
    EXPECT_EQ(expectedSocketNum, 3U);
    EXPECT_EQ(totalRankNum, 8U);
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

    EXPECT_EQ(rankInfoDetectService_->GetConnectionsScalable(2), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rankInfoDetectService_->connSockets_.size(), 2U);
}

// 一直无连接时收敛为超时并返回 HCCL_E_INTERNAL（scalable 缺连接即失败，不降级）
TEST_F(RankInfoDetectServiceTest, Ut_GetConnectionsScalable_When_Timeout_Expect_E_INTERNAL)
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

    EXPECT_EQ(rankInfoDetectService_->GetConnectionsScalable(2), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(rankInfoDetectService_->connSockets_.size(), 0U);
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

    EXPECT_EQ(rankInfoDetectService_->GetConnectionsScalable(1), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rankInfoDetectService_->connSockets_.size(), 1U);
}

// SetupScalable：组内收集成功后 meshMergeCallback 失败时错误码透传（不再广播）
TEST_F(RankInfoDetectServiceTest, Ut_SetupScalable_When_MeshMergeCallbackFail_Expect_ErrCodePassthrough)
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

    std::function<HcclResult(RankTableInfo&)> callback = [](RankTableInfo&) -> HcclResult {
        return HCCL_E_NETWORK;
    };
    EXPECT_EQ(rankInfoDetectService_->SetupScalable(1, callback), HcclResult::HCCL_E_NETWORK);
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
