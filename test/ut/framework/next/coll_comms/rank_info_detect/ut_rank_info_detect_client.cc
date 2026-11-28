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
#include "hccp_peer_manager.h"
#include "timeout_exception.h"
#include <cerrno>
#include "orion_adapter_rts.h"
#include "invalid_params_exception.h"
#include "network_api_exception.h"
#include "host_socket_handle_manager.h"
#include "whitelist.h"
#include "hccp_peer_manager.h"
#include "dev_type.h"

#include "socket.h"
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include "new_rank_info.h"
#include "rank_table_info.h"
#include "json/common/json_parse_util.h"
#include "root_handle_v2.h"
#include "internal_exception.h"
#include "timeout_exception.h"
#include "socket_exception.h"
#include "null_ptr_exception.h"
#include "rank_info_dispatcher.h"
#include "adapter_error_manager_pub.h"
#define private public
#define protected public
#include "rank_info_detect_client.h"
#undef protected
#undef private
#include "orion_adapter_rts.h"
#include "env_config/env_config_v2.h"
#include "env_logic_superpod_config.h"
#include "base_config_legacy.h"
#include "phy_topo.h"
#include "phy_topo_builder.h"
#include "rank_graph_test_data_builder.h"
#include "preempt_port_manager_v2.h"
#include "socket_manager.h"
#include "orion_adapter_hccp.h"
#include "json/json_api.h"

extern "C" int TopoAddrInfoGetSize(int phyId, size_t* size);
extern "C" int TopoAddrInfoGet(int phyId, char* rankInfo, size_t* bufSize);
extern "C" int TopoAddrInfoGetTopoFilePath(int phyId, char* filePath, size_t bufSize);

namespace {
HcclResult SelectLocalHostBackupAddrForTest(Hccl::RankInfoDetectClient& client, nlohmann::json& localDevInfoJson)
{
    std::string localDevInfoJsonStr = localDevInfoJson.dump();
    HcclResult ret = hccl::SelectRootInfoHostAddr(
        localDevInfoJsonStr,
        [&client](const Hccl::IpAddress& addr) {
            return client.ProbeHostRoceAddr(addr);
        },
        client.devPhyId_, client.rankId_);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    localDevInfoJson = nlohmann::json::parse(localDevInfoJsonStr);
    return HCCL_SUCCESS;
}
} // namespace

using namespace Hccl;

TEST(LogicSuperPodRootInfoTest, Ut_LogicSuperPod_When_NewClientAfterEnvChanges_Expect_ProcessCache)
{
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_EXIT(
        {
            setenv("HCCL_LOGIC_SUPERPOD_ID", "group_A", 1);
            RankInfoDetectClient first(0, 1, 0, nullptr);
            RankTableInfo firstTable;
            if (first.ConstructRankTable(firstTable) != HCCL_SUCCESS) {
                std::exit(1);
            }
            setenv("HCCL_LOGIC_SUPERPOD_ID", std::string(129, 'B').c_str(), 1);
            RankInfoDetectClient second(0, 1, 0, nullptr);
            RankTableInfo secondTable;
            std::exit(
                second.ConstructRankTable(secondTable) == HCCL_SUCCESS
                        && hccl::EnvLogicSuperPodConfig::GetInstance().GetLogicSuperPodId() == "group_A" ?
                    0 :
                    1);
        },
        ::testing::ExitedWithCode(0), "");
}

TEST(NetLayerCheckInfoTest, ConsistencyUsesEffectiveLayersAndPeerCapability)
{
    NetLayerCheckInfo local;
    local.version = NetLayerCheckInfo::VERSION;
    local.layers = 6;
    NetLayerCheckInfo peer = local;
    peer.layers = 3;
    EXPECT_EQ(local.CheckConsistency(peer), HCCL_SUCCESS); // 全默认不限制候选层。
    local.explicitSelection = 1;
    EXPECT_EQ(local.CheckConsistency(peer), HCCL_E_PARA);
    peer.layers = 6;
    EXPECT_EQ(local.CheckConsistency(peer), HCCL_SUCCESS); // 默认端与显式端的有效集合相同。
    EXPECT_EQ(peer.CheckConsistency(local), HCCL_SUCCESS);
    peer.version = 0;
    EXPECT_EQ(local.CheckConsistency(peer), HCCL_E_NOT_SUPPORT);
    peer = local;
    peer.result = HCCL_E_PARA;
    EXPECT_EQ(local.CheckConsistency(peer), HCCL_E_PARA);
    EXPECT_EQ(peer.CheckConsistency(local), HCCL_E_PARA);
    peer = local;
    local.layers = peer.layers = 0;
    EXPECT_EQ(local.CheckConsistency(peer), HCCL_E_PARA);
    local.layers = peer.layers = 256;
    EXPECT_EQ(local.CheckConsistency(peer), HCCL_E_PARA);
}

TEST(NetLayerCheckInfoTest, BoundedOptionalTrailer)
{
    NetLayerCheckInfo input;
    input.version = NetLayerCheckInfo::VERSION;
    input.explicitSelection = 1;
    input.layers = 6;
    BinaryStream stream;
    input.Serialize(stream);
    std::vector<char> bytes;
    stream.Dump(bytes);
    BinaryStream valid(bytes);
    NetLayerCheckInfo output;
    ASSERT_EQ(output.Deserialize(valid), HCCL_SUCCESS);
    EXPECT_EQ(output.layers, 6U);
    EXPECT_EQ(output.explicitSelection, 1U);

    auto extendedBytes = bytes;
    extendedBytes.push_back('\0');
    BinaryStream extended(extendedBytes);
    EXPECT_EQ(output.Deserialize(extended), HCCL_E_PARA);
    EXPECT_EQ(output.layers, 6U);
    EXPECT_EQ(output.explicitSelection, 1U);

    bytes.pop_back();
    BinaryStream truncated(bytes);
    EXPECT_EQ(output.Deserialize(truncated), HCCL_E_PARA);
    EXPECT_EQ(output.version, static_cast<u32>(NetLayerCheckInfo::VERSION));
    BinaryStream legacy;
    EXPECT_EQ(output.Deserialize(legacy), HCCL_SUCCESS);
    EXPECT_EQ(output.version, 0U);
    input.layers = 256;
    BinaryStream invalidMask;
    input.Serialize(invalidMask);
    EXPECT_EQ(output.Deserialize(invalidMask), HCCL_E_PARA);
    input.layers = 6;
    input.version = 2;
    BinaryStream future;
    input.Serialize(future);
    EXPECT_EQ(output.Deserialize(future), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(output.version, 0U);
    // 新版本即使扩展帧长，也应识别为版本不支持；失败不改写旧结果。
    BinaryStream futureExtended;
    input.Serialize(futureExtended);
    futureExtended << 0U;
    EXPECT_EQ(output.Deserialize(futureExtended), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(output.version, 0U);
}

class RankInfoDetectClientTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankInfoDetectClientTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RankInfoDetectClientTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in RankInfoDetectClientTest SetUP" << std::endl;
        socketHandle = new int(0);
        MOCKER(HrtRaSocketInit).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(socketHandle));
        MOCKER(HrtRaSocketDeInit).stubs().with(mockcpp::any());
        MOCKER_CPP(&HccpPeerManager::Init).stubs().with(mockcpp::any());
        MOCKER_CPP(&HccpPeerManager::DeInit).stubs().with(mockcpp::any());
        MOCKER(HrtGetDevice).stubs().with(outBound(0)).will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER_CPP(&HostSocketHandleManager::Destroy).stubs().with(mockcpp::any(), mockcpp::any());
        IpAddress serverIp = IpAddress("10.0.0.10");
        u32 hostPort = 60001;
        IpAddress hostIp_ = IpAddress("192.168.1.8");
        u32 rankSize_ = 1;
        u32 devPhyId_ = 0;
        u32 rankId_ = 0;
        std::string clientSocketTag = "rank_info_test_server";

        auto clientSocket_ = std::make_shared<Socket>(
            socketHandle, hostIp_, hostPort, serverIp, clientSocketTag, SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
        SocketAgent socketAgent_ = SocketAgent(clientSocket_.get());
        rankInfoDetectClient_ = new RankInfoDetectClient(devPhyId_, rankSize_, rankId_, clientSocket_);
    }

    virtual void TearDown()
    {
        delete rankInfoDetectClient_;
        rankInfoDetectClient_ = nullptr;
        // RankInfoDetectClient 析构会 detach 线程调 DeInit，需等其完成再 verify 清 mock
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        delete static_cast<int*>(socketHandle);
        socketHandle = nullptr;
        GlobalMockObject::verify();
        std::cout << "A Test case in RankInfoDetectClientTest TearDown" << std::endl;
    }

    RankInfoDetectClient* rankInfoDetectClient_{nullptr};
    SocketHandle socketHandle;

    // 辅助函数：构建含指定 rootRankId/failedAgentIdList 的 Mock rankInfo 消息
    vector<char> BuildMockRankInfoMsg(const std::string& failedAgentIdList, u32 rootRankId = UINT32_MAX)
    {
        RankTableInfo localRankTable;
        localRankTable.version = "1.0";
        localRankTable.rankCount = 1;
        hccl::NewRankInfo rankInfo{};
        rankInfo.rankId = 0;
        rankInfo.rankLevelInfos.emplace_back(hccl::RankLevelInfo{});
        localRankTable.ranks.emplace_back(rankInfo);

        BinaryStream binaryStream;
        EXPECT_EQ(localRankTable.GetBinStream(true, binaryStream), HCCL_SUCCESS);
        binaryStream << rankInfoDetectClient_->currentStep_;
        binaryStream << rootRankId;
        binaryStream << failedAgentIdList;

        vector<char> rankInfoMsg;
        binaryStream.Dump(rankInfoMsg);
        return rankInfoMsg;
    }
};

namespace {
std::string g_clientCapturedErrorCode;
void StubClientRptInputErr(std::string error_code, std::vector<std::string> key, std::vector<std::string> value)
{
    (void)key;
    (void)value;
    g_clientCapturedErrorCode = error_code;
}
std::string BuildRootInfoJsonString()
{
    nlohmann::json rootInfo;
    rootInfo["version"] = "2.0";
    rootInfo["topo_file_path"] = "/tmp";
    rootInfo["rank_count"] = 2;
    rootInfo["rank_list"] = nlohmann::json::array();
    for (u32 rankId = 0; rankId < 2U; ++rankId) {
        nlohmann::json rank;
        rank["rank_id"] = rankId;
        rank["device_id"] = rankId;
        rank["local_id"] = rankId;
        rank["level_list"] = nlohmann::json::array({
            {
                {"net_layer", 0},
                {"net_instance_id", "az0-rack0"},
                {"net_type", "TOPO_FILE_DESC"},
                {"net_attr", ""},
                {"rank_addr_list", nlohmann::json::array({
                                       {
                                           {"addr_type", "IPV4"},
                                           {"addr", rankId == 0U ? "223.0.0.28" : "223.0.0.10"},
                                           {"ports", nlohmann::json::array({rankId == 0U ? "0/0" : "0/1"})},
                                       },
                                   })},
            },
        });
        rootInfo["rank_list"].push_back(rank);
    }
    return rootInfo.dump();
}

int FillTmpTopoPath(int phyId, char* path, size_t bufSize)
{
    (void)phyId;
    constexpr const char* tmpPath = "/tmp";
    const size_t pathSize = std::strlen(tmpPath) + 1U;
    if (path == nullptr || bufSize < pathSize) {
        return -1;
    }
    std::memcpy(path, tmpPath, pathSize);
    return 0;
}

HcclResult BuildEmptyTopo(hccl::PhyTopoBuilder* builder, const std::string& topoPath)
{
    (void)builder;
    (void)topoPath;
    hccl::PhyTopo::GetInstance()->Clear();
    hccl::PhyTopoBuilder::GetInstance().RecoverBuild(test::MakeTopoInfo({0}, {}));
    return HCCL_SUCCESS;
}

HcclResult BuildHostRdmaTopo(hccl::PhyTopoBuilder* builder, const std::string& topoPath)
{
    (void)builder;
    (void)topoPath;
    hccl::PhyTopo::GetInstance()->Clear();
    hccl::PhyTopoBuilder::GetInstance().RecoverBuild(test::MakeHostRdmaTopo());
    return HCCL_SUCCESS;
}

EnvHostNicConfig BuildHostPortRangeConfig()
{
    EnvHostNicConfig fakeConfig;
    std::vector<SocketPortRange> portRanges{SocketPortRange{50000, 50001}};
    fakeConfig.hcclHostSocketPortRange = CfgField<std::vector<SocketPortRange>>{
        "HCCL_HOST_SOCKET_PORT_RANGE", portRanges, [](const std::string& s) -> std::vector<SocketPortRange> {
            return CastSocketPortRange(s, "HCCL_HOST_SOCKET_PORT_RANGE");
        }};
    fakeConfig.hcclHostSocketPortRange.isParsed = true;
    fakeConfig.hcclIfBasePort = CfgField<u32>{"HCCL_IF_BASE_PORT", HCCL_INVALID_PORT, Str2T<u32>};
    fakeConfig.hcclIfBasePort.isParsed = true;
    return fakeConfig;
}

hccl::NewRankInfo BuildHostRdmaRankInfo()
{
    hccl::NewRankInfo rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceId = 0;
    rankInfo.localId = 0;
    hccl::AddressInfo addrInfo;
    addrInfo.addr = IpAddress("192.168.1.101");
    hccl::RankLevelInfo levelInfo;
    levelInfo.netLayer = 3;
    levelInfo.rankAddrs.push_back(addrInfo);
    rankInfo.rankLevelInfos.push_back(levelInfo);
    return rankInfo;
}

void MockTopoPathAndBuild(HcclResult (*buildFunc)(hccl::PhyTopoBuilder*, const std::string&))
{
    MOCKER(TopoAddrInfoGetTopoFilePath)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(FillTmpTopoPath));
    MOCKER_CPP(&hccl::PhyTopoBuilder::Build).stubs().with(mockcpp::any()).will(invoke(buildFunc));
}

hccl::NewRankInfo BuildRankInfoForTls(u32 rankId, TlsStatus tlsStatus)
{
    hccl::NewRankInfo rankInfo{};
    rankInfo.rankId = rankId;
    rankInfo.localId = rankId;
    rankInfo.replacedLocalId = rankId;
    rankInfo.rankLevelInfos.emplace_back(hccl::RankLevelInfo{});
    rankInfo.tlsStatus = tlsStatus;
    return rankInfo;
}

hccl::NewRankInfo BuildRankInfoForHostDpuTls(u32 rankId, TlsStatus hostDpuTlsStatus)
{
    hccl::NewRankInfo rankInfo{};
    rankInfo.rankId = rankId;
    rankInfo.localId = rankId;
    rankInfo.replacedLocalId = rankId;
    rankInfo.rankLevelInfos.emplace_back(hccl::RankLevelInfo{});
    rankInfo.hostDpuTlsStatus = hostDpuTlsStatus;
    return rankInfo;
}

void BuildRankTableForTls(RankTableInfo& rankTable, const std::vector<TlsStatus>& tlsStatusList)
{
    rankTable.version = "2.0";
    rankTable.rankCount = tlsStatusList.size();
    rankTable.ranks.clear();
    for (u32 idx = 0; idx < tlsStatusList.size(); ++idx) {
        rankTable.ranks.emplace_back(BuildRankInfoForTls(idx, tlsStatusList[idx]));
    }
}

void BuildRankTableForHostDpuTls(RankTableInfo& rankTable, const std::vector<TlsStatus>& hostDpuTlsStatusList)
{
    rankTable.version = "2.0";
    rankTable.rankCount = hostDpuTlsStatusList.size();
    rankTable.ranks.clear();
    for (u32 idx = 0; idx < hostDpuTlsStatusList.size(); ++idx) {
        rankTable.ranks.emplace_back(BuildRankInfoForHostDpuTls(idx, hostDpuTlsStatusList[idx]));
    }
}

HcclResult StubLocalTlsStatusQuery(RaInfo* info, TlsStatus& tlsStatus)
{
    EXPECT_EQ(info->phyId, 0U);
    if (info->mode == static_cast<int>(NetworkMode::NETWORK_OFFLINE)) {
        tlsStatus = TlsStatus::DISABLE;
    } else {
        EXPECT_EQ(info->mode, static_cast<int>(NetworkMode::NETWORK_PEER_ONLINE));
        tlsStatus = TlsStatus::ENABLE;
    }
    return HCCL_SUCCESS;
}

HcclResult StubHostDpuTlsQueryNotSupport(RaInfo* info, TlsStatus& tlsStatus)
{
    EXPECT_EQ(info->mode, static_cast<int>(NetworkMode::NETWORK_PEER_ONLINE));
    tlsStatus = TlsStatus::UNKNOWN;
    return HCCL_E_NOT_SUPPORT;
}

HcclResult StubHostDpuTlsQueryRuntimeFail(RaInfo* info, TlsStatus& tlsStatus)
{
    EXPECT_EQ(info->mode, static_cast<int>(NetworkMode::NETWORK_PEER_ONLINE));
    tlsStatus = TlsStatus::DISABLE;
    return HCCL_E_NETWORK;
}

std::string g_reportedErrorCode;
std::vector<std::string> g_reportedKeys;
std::vector<std::string> g_reportedValues;

void StubRptInputErr(std::string errorCode, std::vector<std::string> keys, std::vector<std::string> values)
{
    g_reportedErrorCode = errorCode;
    g_reportedKeys = keys;
    g_reportedValues = values;
}

nlohmann::json BuildLocalDevInfoWithBackupAddrs(
    const std::string& addrType, const std::string& primaryAddr, const nlohmann::json& backupAddrs, u32 netLayer = 3,
    const std::string& port = "host0")
{
    nlohmann::json addrJson = {
        {"addr_type", addrType},
        {"addr", primaryAddr},
        {"backup_addr", backupAddrs},
        {"ports", nlohmann::json::array({port})},
    };
    nlohmann::json levelJson;
    levelJson["net_layer"] = netLayer;
    levelJson["rank_addr_list"] = nlohmann::json::array({addrJson});
    nlohmann::json localDevInfoJson;
    localDevInfoJson["local_id"] = 0;
    localDevInfoJson["level_list"] = nlohmann::json::array({levelJson});
    return localDevInfoJson;
}

nlohmann::json BuildIpv4BackupAddrs(std::size_t count)
{
    nlohmann::json backupAddrs = nlohmann::json::array();
    for (std::size_t idx = 0; idx < count; ++idx) {
        backupAddrs.push_back("192.168.2." + std::to_string(idx + 1));
    }
    return backupAddrs;
}

std::string GetSelectedHostAddr(const nlohmann::json& localDevInfoJson)
{
    return localDevInfoJson["level_list"][0]["rank_addr_list"][0]["addr"].get<std::string>();
}

} // namespace

TEST_F(RankInfoDetectClientTest, Ut_CheckStatus_When_Normal_Expect_Success)
{
    MOCKER_CPP(&Socket::GetStatus).stubs().then(returnValue((SocketStatus)SocketStatus::OK));

    EXPECT_EQ(rankInfoDetectClient_->CheckStatus(), HCCL_SUCCESS);
}

TEST_F(RankInfoDetectClientTest, Ut_SendAgentIdAndRankSize_When_Normal_Expect_Success)
{
    MOCKER(HrtRaSocketBlockSend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(true));

    EXPECT_EQ(rankInfoDetectClient_->SendAgentIdAndRankSize(), HcclResult::HCCL_SUCCESS);
}

TEST_F(RankInfoDetectClientTest, Ut_ConstructSingleRank_When_Normal_Expect_Success)
{
    RankTableInfo localRankTable;

    EXPECT_NO_THROW(rankInfoDetectClient_->ConstructSingleRank(localRankTable));

    EXPECT_EQ(localRankTable.version, "2.0");
    EXPECT_EQ(localRankTable.rankCount, 1U);
    EXPECT_EQ(localRankTable.ranks.size(), 1U);
    const hccl::NewRankInfo& actualRankInfo = localRankTable.ranks[0];
    EXPECT_EQ(actualRankInfo.rankId, 0);
    EXPECT_EQ(actualRankInfo.rankLevelInfos.size(), 1U);
}

TEST_F(RankInfoDetectClientTest, Ut_ConstructSingleRank_When_TlsQueriesSucceed_Expect_BothStatusesUpdated)
{
    RankTableInfo localRankTable;
    MOCKER(HrtRaGetTlsStatus).stubs().will(invoke(StubLocalTlsStatusQuery));

    EXPECT_NO_THROW(rankInfoDetectClient_->ConstructSingleRank(localRankTable));

    ASSERT_EQ(localRankTable.ranks.size(), 1U);
    EXPECT_EQ(localRankTable.ranks[0].tlsStatus, TlsStatus::DISABLE);
    EXPECT_EQ(localRankTable.ranks[0].hostDpuTlsStatus, TlsStatus::ENABLE);
}

TEST_F(RankInfoDetectClientTest, Ut_GetLocalHostDpuTlsStatus_When_NotSupported_Expect_Unknown)
{
    TlsStatus tlsStatus = TlsStatus::ENABLE;
    MOCKER(HrtRaGetTlsStatus).stubs().will(invoke(StubHostDpuTlsQueryNotSupport));

    HcclResult ret = rankInfoDetectClient_->GetLocalHostDpuTlsStatus(tlsStatus);

    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(tlsStatus, TlsStatus::UNKNOWN);
}

TEST_F(RankInfoDetectClientTest, Ut_GetLocalHostDpuTlsStatus_When_QueryFails_Expect_Disable)
{
    TlsStatus tlsStatus = TlsStatus::UNKNOWN;
    MOCKER(HrtRaGetTlsStatus).stubs().will(invoke(StubHostDpuTlsQueryRuntimeFail));

    HcclResult ret = rankInfoDetectClient_->GetLocalHostDpuTlsStatus(tlsStatus);

    EXPECT_EQ(ret, HCCL_E_NETWORK);
    EXPECT_EQ(tlsStatus, TlsStatus::DISABLE);
}

TEST_F(RankInfoDetectClientTest, Ut_ConstructRankTable_When_SelectedSingleRank_Expect_UseActualLayers)
{
    std::string testJsonPath{"/tmp"};

    MOCKER(realpath)
        .stubs()
        .with(mockcpp::any(), outBoundP(const_cast<char*>(testJsonPath.c_str()), testJsonPath.size() + 1))
        .will(returnValue(const_cast<char*>(testJsonPath.c_str())));
    auto rootInfoJson = nlohmann::json::parse(BuildRootInfoJsonString());
    for (auto& rank : rootInfoJson["rank_list"]) {
        auto level3 = rank["level_list"][0];
        level3["net_layer"] = 3;
        level3["net_type"] = "CLOS";
        rank["level_list"].push_back(level3);
    }
    std::string testJsonContent = rootInfoJson.dump();
    MOCKER(hccl::ParseRootInfoConfigFile)
        .stubs()
        .with(mockcpp::any(), outBound(testJsonContent))
        .will(returnValue(HCCL_SUCCESS));
    size_t expectedSize = testJsonContent.size();
    MOCKER(TopoAddrInfoGetSize).stubs().with(0, outBoundP(&expectedSize, sizeof(size_t))).will(returnValue(0));
    const char* rootInfoCtx = testJsonContent.c_str();
    MOCKER(TopoAddrInfoGet)
        .stubs()
        .with(
            0, outBoundP(const_cast<char*>(rootInfoCtx), testJsonContent.size()),
            outBoundP(&expectedSize, sizeof(size_t)))
        .will(returnValue(0));

    struct Case {
        u32 rankSize;
        const char* expression;
        u32 selectedMask;
    };
    const Case cases[] = {{1, "", 1},       {2, "", 9},    {1, "0", 1}, {1, "3", 8},
                          {1, "not(0)", 8}, {1, "0,3", 9}, {1, "2", 0}, {1, "not(0,3)", 0}};
    for (const auto& item : cases) {
        SCOPED_TRACE(item.expression);
        rankInfoDetectClient_->rankSize_ = item.rankSize;
        hccl::NetLayerSelection selection;
        ASSERT_EQ(hccl::NetLayerSelection::Parse(item.expression, selection), HCCL_SUCCESS);
        rankInfoDetectClient_->SetNetLayerSelection(selection);
        RankTableInfo localRankTable;
        ASSERT_EQ(rankInfoDetectClient_->ConstructRankTable(localRankTable), HCCL_SUCCESS);
        EXPECT_EQ(localRankTable.version, "2.0");
        EXPECT_EQ(localRankTable.rankCount, item.rankSize);
        ASSERT_EQ(localRankTable.ranks.size(), 1U);
        const auto& levels = localRankTable.ranks[0].rankLevelInfos;
        const bool defaultSingleRank = item.rankSize == 1 && item.expression[0] == '\0';
        ASSERT_EQ(levels.size(), defaultSingleRank ? 1U : 2U);
        EXPECT_EQ(levels.front().netLayer, 0U);
        if (!defaultSingleRank) {
            EXPECT_EQ(levels.back().netLayer, 3U);
        }
        rankInfoDetectClient_->PrepareNetLayerCheck(localRankTable.ranks.front());
        EXPECT_EQ(rankInfoDetectClient_->localNetLayerCheck_.layers, item.selectedMask);
        EXPECT_EQ(
            rankInfoDetectClient_->localNetLayerCheck_.result, item.selectedMask == 0 ? HCCL_E_PARA : HCCL_SUCCESS);
    }
}

TEST_F(RankInfoDetectClientTest, Ut_ConstructRankTable_When_SingleRankDiscoveryFails_Expect_NoSelectionFallback)
{
    MOCKER(hccl::ParseRootInfoConfigFile).stubs().will(returnValue(HCCL_E_PARA));
    MOCKER(TopoAddrInfoGetSize).stubs().will(returnValue(-1));
    MOCKER(TopoAddrInfoGet).expects(never());
    hccl::NetLayerSelection selection;
    ASSERT_EQ(hccl::NetLayerSelection::Parse("not(0)", selection), HCCL_SUCCESS);
    rankInfoDetectClient_->SetNetLayerSelection(selection);
    RankTableInfo selectedTable;
    EXPECT_EQ(rankInfoDetectClient_->ConstructRankTable(selectedTable), HCCL_E_PARA);
    EXPECT_TRUE(selectedTable.ranks.empty());

    // 默认单卡不依赖拓扑发现，仍保持原快速路径。
    rankInfoDetectClient_->SetNetLayerSelection(hccl::NetLayerSelection{});
    RankTableInfo defaultTable;
    ASSERT_EQ(rankInfoDetectClient_->ConstructRankTable(defaultTable), HCCL_SUCCESS);
    ASSERT_EQ(defaultTable.ranks.size(), 1U);
    ASSERT_EQ(defaultTable.ranks[0].rankLevelInfos.size(), 1U);
    EXPECT_EQ(defaultTable.ranks[0].rankLevelInfos[0].netLayer, 0U);
}

TEST_F(RankInfoDetectClientTest, Ut_LogicSuperPod_When_MultiRankConfiguredOrUnset_Expect_IdCopiedToLocalTable)
{
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    // 独立进程隔离首次解析缓存，分别验证非空配置和未配置。
    for (const char* logicId : {"group_A", static_cast<const char*>(nullptr)}) {
        EXPECT_EXIT(
            {
                if (logicId == nullptr) {
                    EXPECT_EQ(unsetenv("HCCL_LOGIC_SUPERPOD_ID"), 0);
                } else {
                    EXPECT_EQ(setenv("HCCL_LOGIC_SUPERPOD_ID", logicId, 1), 0);
                }
                rankInfoDetectClient_->rankSize_ = 2;
                std::string testJsonPath{"/tmp"};
                MOCKER(realpath)
                    .stubs()
                    .with(mockcpp::any(), outBoundP(const_cast<char*>(testJsonPath.c_str()), testJsonPath.size() + 1))
                    .will(returnValue(const_cast<char*>(testJsonPath.c_str())));
                std::string testJsonContent = BuildRootInfoJsonString();
                size_t expectedSize = testJsonContent.size();
                MOCKER(TopoAddrInfoGetSize)
                    .stubs()
                    .with(0, outBoundP(&expectedSize, sizeof(size_t)))
                    .will(returnValue(0));
                MOCKER(TopoAddrInfoGet)
                    .stubs()
                    .with(
                        0, outBoundP(const_cast<char*>(testJsonContent.c_str()), testJsonContent.size()),
                        outBoundP(&expectedSize, sizeof(size_t)))
                    .will(returnValue(0));

                RankTableInfo localRankTable;
                EXPECT_EQ(rankInfoDetectClient_->ConstructRankTable(localRankTable), HCCL_SUCCESS);
                EXPECT_EQ(localRankTable.rankCount, 2U);
                EXPECT_EQ(localRankTable.ranks.size(), 1U);
                if (localRankTable.ranks.size() == 1U) {
                    EXPECT_EQ(localRankTable.ranks[0].logicSuperPodId, logicId == nullptr ? "" : logicId);
                }
                std::exit(::testing::Test::HasFailure() ? 1 : 0);
            },
            ::testing::ExitedWithCode(0), "");
    }
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_BackupAddrAvailable_Expect_SelectBackup)
{
    nlohmann::json localDevInfoJson = BuildLocalDevInfoWithBackupAddrs(
        "IPV4", "192.168.1.101", nlohmann::json::array({"192.168.1.102", "192.168.1.103"}), 3, "d2h");

    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&))
        .expects(exactly(2))
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCP_ELINKDOWN))
        .then(returnValue(0));

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_SUCCESS);
    EXPECT_EQ(localDevInfoJson["level_list"][0]["rank_addr_list"][0]["addr"].get<std::string>(), "192.168.1.102");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_LevelListMissing_Expect_E_PARA)
{
    nlohmann::json localDevInfoJson = {
        {"rank_id", 0},
        {"device_id", 0},
        {"local_id", 0},
    };

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_E_PARA);
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_LevelListIsNotArray_Expect_E_PARA)
{
    nlohmann::json localDevInfoJson = {
        {"rank_id", 0},
        {"device_id", 0},
        {"local_id", 0},
        {"level_list", "invalid"},
    };

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_E_PARA);
}

TEST_F(
    RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_PrimaryAvailableAndMultipleBackups_Expect_KeepPrimary)
{
    nlohmann::json localDevInfoJson = BuildLocalDevInfoWithBackupAddrs(
        "IPV4", "192.168.1.101", nlohmann::json::array({"192.168.1.102", "192.168.1.103"}));
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&))
        .expects(exactly(1))
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(0));

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_SUCCESS);
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.101");
}

TEST_F(
    RankInfoDetectClientTest,
    Ut_SelectLocalHostBackupAddr_When_FirstBackupUnavailableAndSecondAvailable_Expect_SelectSecondBackup)
{
    nlohmann::json localDevInfoJson = BuildLocalDevInfoWithBackupAddrs(
        "IPV4", "192.168.1.101", nlohmann::json::array({"192.168.1.102", "192.168.1.103"}));
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&))
        .expects(exactly(3))
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCP_ELINKDOWN))
        .then(returnValue(HCCP_ELINKDOWN))
        .then(returnValue(0));

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_SUCCESS);
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.103");
}

TEST_F(
    RankInfoDetectClientTest,
    Ut_SelectLocalHostBackupAddr_When_BackupAddrEmptyAndPrimaryAvailable_Expect_ProbeAndKeepPrimary)
{
    nlohmann::json localDevInfoJson
        = BuildLocalDevInfoWithBackupAddrs("IPV4", "192.168.1.101", nlohmann::json::array());
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&))
        .expects(exactly(1))
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(0));

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_SUCCESS);
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.101");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_Ipv6PrimaryUnavailable_Expect_SelectIpv6Backup)
{
    nlohmann::json localDevInfoJson = BuildLocalDevInfoWithBackupAddrs(
        "IPV6", "2001:db8::1", nlohmann::json::array({"2001:db8::2", "2001:db8::3"}));
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&))
        .expects(exactly(2))
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCP_ELINKDOWN))
        .then(returnValue(0));

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_SUCCESS);
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "2001:db8::2");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_PrimaryAddrInvalid_Expect_E_PARA)
{
    nlohmann::json localDevInfoJson
        = BuildLocalDevInfoWithBackupAddrs("IPV4", "192.168.1", nlohmann::json::array({"192.168.1.102"}));

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_E_PARA);
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_BackupAddrInvalid_Expect_E_PARA)
{
    nlohmann::json localDevInfoJson
        = BuildLocalDevInfoWithBackupAddrs("IPV4", "192.168.1.101", nlohmann::json::array({"192.168.1"}));

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_E_PARA);
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_BackupAddrElementNotString_Expect_E_PARA)
{
    nlohmann::json localDevInfoJson
        = BuildLocalDevInfoWithBackupAddrs("IPV4", "192.168.1.101", nlohmann::json::array({102}));

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_E_PARA);
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_PrimaryProbeError_Expect_E_NETWORK)
{
    nlohmann::json localDevInfoJson = BuildLocalDevInfoWithBackupAddrs(
        "IPV4", "192.168.1.101", nlohmann::json::array({"192.168.1.102", "192.168.1.103"}));
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&))
        .expects(exactly(1))
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(-EINVAL));

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_E_NETWORK);
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.101");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_NoBackupAddr_Expect_KeepPrimaryWithoutProbe)
{
    nlohmann::json localDevInfoJson
        = BuildLocalDevInfoWithBackupAddrs("IPV4", "192.168.1.101", nlohmann::json::array());
    localDevInfoJson["level_list"][0]["rank_addr_list"][0].erase("backup_addr");
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&)).expects(never());

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_SUCCESS);
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.101");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_Layer4HasBackupAddr_Expect_SelectBackup)
{
    nlohmann::json localDevInfoJson
        = BuildLocalDevInfoWithBackupAddrs("IPV4", "192.168.1.101", nlohmann::json::array({"192.168.1.102"}), 4);
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&))
        .expects(exactly(2))
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCP_ELINKDOWN))
        .then(returnValue(0));

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_SUCCESS);
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.102");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_Layer0To2HaveBackupAddr_Expect_KeepPrimary)
{
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&)).expects(never());
    for (u32 netLayer = 0; netLayer <= 2; ++netLayer) {
        nlohmann::json localDevInfoJson = BuildLocalDevInfoWithBackupAddrs(
            "IPV4", "192.168.1.101", nlohmann::json::array({"192.168.1.102"}), netLayer);

        EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_SUCCESS);
        EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.101");
    }
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_Layer3PortsAreUnrelated_Expect_SelectBackup)
{
    nlohmann::json localDevInfoJson = BuildLocalDevInfoWithBackupAddrs(
        "IPV4", "192.168.1.101", nlohmann::json::array({"192.168.1.102"}), 3, "unrelated-port");
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&))
        .expects(exactly(2))
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCP_ELINKDOWN))
        .then(returnValue(0));

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_SUCCESS);
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.102");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_Layer3HasMultipleAddrConfigs_Expect_SelectEachBackup)
{
    nlohmann::json localDevInfoJson = nlohmann::json::parse(R"({
        "local_id": 0,
        "level_list": [{
            "net_layer": 3,
            "rank_addr_list": [
                {
                    "addr_type": "IPV4",
                    "addr": "192.168.1.10",
                    "backup_addr": ["192.168.1.11"],
                    "ports": ["port-a"]
                },
                {
                    "addr_type": "IPV4",
                    "addr": "192.168.2.10",
                    "backup_addr": ["192.168.2.11"],
                    "ports": ["port-b"]
                }
            ]
        }]
    })");
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&))
        .expects(exactly(4))
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCP_ELINKDOWN))
        .then(returnValue(0))
        .then(returnValue(HCCP_ELINKDOWN))
        .then(returnValue(0));

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_SUCCESS);
    EXPECT_EQ(localDevInfoJson["level_list"][0]["rank_addr_list"][0]["addr"].get<std::string>(), "192.168.1.11");
    EXPECT_EQ(localDevInfoJson["level_list"][0]["rank_addr_list"][1]["addr"].get<std::string>(), "192.168.2.11");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectBackupAddr_When_ProbeReturnsOtherError_Expect_E_NETWORK)
{
    nlohmann::json localDevInfoJson
        = BuildLocalDevInfoWithBackupAddrs("IPV4", "192.168.1.101", nlohmann::json::array({"192.168.1.102"}), 3, "d2h");

    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&))
        .expects(exactly(1))
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(-EOPNOTSUPP));

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_E_NETWORK);
    EXPECT_EQ(localDevInfoJson["level_list"][0]["rank_addr_list"][0]["addr"].get<std::string>(), "192.168.1.101");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_AllCandidatesLinkDown_Expect_KeepPrimaryAndContinue)
{
    nlohmann::json localDevInfoJson = nlohmann::json::parse(R"({
        "local_id": 0,
        "level_list": [
            {
                "net_layer": 3,
                "rank_addr_list": [
                    {
                        "addr_type": "IPV4",
                        "addr": "192.168.1.101",
                        "backup_addr": ["192.168.1.102", "192.168.1.103"],
                        "ports": ["d2h"]
                    }
                ]
            }
        ]
    })");

    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&))
        .expects(exactly(3))
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCP_ELINKDOWN));

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_SUCCESS);
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.101");
}

TEST_F(
    RankInfoDetectClientTest,
    Ut_SelectLocalHostBackupAddr_When_BackupAddrCountAtLimitAndAllLinkDown_Expect_KeepPrimaryAndContinue)
{
    nlohmann::json localDevInfoJson = BuildLocalDevInfoWithBackupAddrs(
        "IPV4", "192.168.1.101", BuildIpv4BackupAddrs(hccl::MAX_VALUE_BACKUP_ADDR_SIZE));
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&))
        .expects(exactly(hccl::MAX_VALUE_BACKUP_ADDR_SIZE + 1U))
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCP_ELINKDOWN));

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_SUCCESS);
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.101");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_BackupAddrCountExceedsLimit_Expect_E_PARA)
{
    nlohmann::json localDevInfoJson = BuildLocalDevInfoWithBackupAddrs(
        "IPV4", "192.168.1.101", BuildIpv4BackupAddrs(hccl::MAX_VALUE_BACKUP_ADDR_SIZE + 1U));
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&)).expects(never());

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_E_PARA);
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_BackupAddrIsNotArray_Expect_E_PARA)
{
    nlohmann::json localDevInfoJson = nlohmann::json::parse(R"({
        "local_id": 0,
        "level_list": [
            {
                "net_layer": 3,
                "rank_addr_list": [
                    {
                        "addr_type": "IPV4",
                        "addr": "192.168.1.101",
                        "backup_addr": "192.168.1.102",
                        "ports": ["d2h"]
                    }
                ]
            }
        ]
    })");

    EXPECT_EQ(SelectLocalHostBackupAddrForTest(*rankInfoDetectClient_, localDevInfoJson), HCCL_E_PARA);
}

TEST_F(RankInfoDetectClientTest, Ut_ProbeHostRoceAddr_When_LinkDown_Expect_ReturnRawError)
{
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&))
        .expects(exactly(1))
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCP_ELINKDOWN));

    const s32 ret = rankInfoDetectClient_->ProbeHostRoceAddr(IpAddress("192.168.1.101", AF_INET));
    EXPECT_EQ(ret, HCCP_ELINKDOWN);
}

TEST_F(RankInfoDetectClientTest, Ut_ProbeHostRoceAddr_When_OtherError_Expect_ReturnRawError)
{
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&))
        .expects(exactly(1))
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(-EOPNOTSUPP));

    const s32 ret = rankInfoDetectClient_->ProbeHostRoceAddr(IpAddress("192.168.1.101", AF_INET));
    EXPECT_EQ(ret, -EOPNOTSUPP);
}

TEST_F(RankInfoDetectClientTest, NetLayerCheckUsesEffectiveLayers)
{
    hccl::NewRankInfo rank;
    rank.rankId = 0;
    for (u32 layer : {0U, 1U, 2U}) {
        hccl::RankLevelInfo level;
        level.netLayer = layer;
        rank.rankLevelInfos.push_back(level);
    }
    for (const auto& expression : {"not(0)", "1,2", "2,1,1"}) {
        hccl::NetLayerSelection selection;
        ASSERT_EQ(hccl::NetLayerSelection::Parse(expression, selection), HCCL_SUCCESS);
        rankInfoDetectClient_->SetNetLayerSelection(selection);
        rankInfoDetectClient_->PrepareNetLayerCheck(rank);
        EXPECT_EQ(rankInfoDetectClient_->localNetLayerCheck_.layers, 6U);
        EXPECT_EQ(rankInfoDetectClient_->localNetLayerCheck_.result, HCCL_SUCCESS);
    }
    hccl::NetLayerSelection missing;
    ASSERT_EQ(hccl::NetLayerSelection::Parse("7", missing), HCCL_SUCCESS);
    rankInfoDetectClient_->SetNetLayerSelection(missing);
    rankInfoDetectClient_->PrepareNetLayerCheck(rank);
    EXPECT_EQ(rankInfoDetectClient_->localNetLayerCheck_.result, HCCL_E_PARA);
    rankInfoDetectClient_->SetNetLayerSelection(hccl::NetLayerSelection{});
    rank.rankLevelInfos.back().netLayer = hccl::MAX_VALUE_NETLAYER + 1;
    rankInfoDetectClient_->PrepareNetLayerCheck(rank);
    EXPECT_EQ(rankInfoDetectClient_->localNetLayerCheck_.result, HCCL_E_PARA);
}

TEST_F(RankInfoDetectClientTest, NetLayerCheckRejectsMismatchAndLegacyRoot)
{
    auto& local = rankInfoDetectClient_->localNetLayerCheck_;
    auto& remote = rankInfoDetectClient_->remoteNetLayerCheck_;
    local.version = NetLayerCheckInfo::VERSION;
    local.explicitSelection = 1;
    local.layers = 6;
    EXPECT_EQ(rankInfoDetectClient_->VerifyNetLayerConsistency(), HCCL_E_NOT_SUPPORT);
    remote = local;
    EXPECT_EQ(rankInfoDetectClient_->VerifyNetLayerConsistency(), HCCL_SUCCESS);
    remote.layers = 3;
    EXPECT_EQ(rankInfoDetectClient_->VerifyNetLayerConsistency(), HCCL_E_PARA);
    remote.result = HCCL_E_NOT_SUPPORT;
    EXPECT_EQ(rankInfoDetectClient_->VerifyNetLayerConsistency(), HCCL_E_NOT_SUPPORT);
    local.explicitSelection = 0;
    remote = local;
    remote.explicitSelection = 1; // 默认端也必须与显式端的有效层一致。
    EXPECT_EQ(rankInfoDetectClient_->VerifyNetLayerConsistency(), HCCL_SUCCESS);
    remote.layers = 3;
    EXPECT_EQ(rankInfoDetectClient_->VerifyNetLayerConsistency(), HCCL_E_PARA);
    remote = {};
    EXPECT_EQ(rankInfoDetectClient_->VerifyNetLayerConsistency(), HCCL_SUCCESS);
    local.result = HCCL_E_PARA;
    EXPECT_EQ(rankInfoDetectClient_->VerifyNetLayerConsistency(), HCCL_E_PARA);
}

TEST_F(RankInfoDetectClientTest, NetLayerCheckParsesRootBroadcastResult)
{
    auto bytes = BuildMockRankInfoMsg("");
    NetLayerCheckInfo result;
    result.version = NetLayerCheckInfo::VERSION;
    result.explicitSelection = 1;
    result.result = HCCL_E_PARA;
    BinaryStream tail;
    result.Serialize(tail);
    tail.Dump(bytes);
    MOCKER_CPP(&RankInfoDetectClient::RecvRankTableMsg).stubs().with(outBound(bytes)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankInfoDetectClient::VerifyRankTable).expects(never());
    // 接收入口先拒绝 Root 广播的选层错误，不再进入普通拓扑校验。
    EXPECT_EQ(rankInfoDetectClient_->RecvRankTable(), HCCL_E_PARA);
    EXPECT_EQ(rankInfoDetectClient_->remoteNetLayerCheck_.result, HCCL_E_PARA);
    const auto originalRankSize = rankInfoDetectClient_->rankSize_;
    RankTableInfo nextRankTable;
    nextRankTable.rankCount = 99;
    EXPECT_EQ(rankInfoDetectClient_->WaitForNextBroadcast(nextRankTable, 2), HCCL_E_PARA);
    EXPECT_EQ(rankInfoDetectClient_->rankSize_, originalRankSize);
    EXPECT_EQ(nextRankTable.rankCount, 99U);
}

TEST_F(RankInfoDetectClientTest, NetLayerCheckRejectsUnknownRootVersionAtReceiveEntry)
{
    auto bytes = BuildMockRankInfoMsg("");
    NetLayerCheckInfo result;
    result.version = NetLayerCheckInfo::VERSION + 1;
    BinaryStream tail;
    result.Serialize(tail);
    tail << 0U; // 新版扩展尾帧应报不支持，不得转成参数错误。
    tail.Dump(bytes);
    MOCKER_CPP(&RankInfoDetectClient::RecvRankTableMsg).stubs().with(outBound(bytes)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankInfoDetectClient::VerifyRankTable).expects(never());
    EXPECT_EQ(rankInfoDetectClient_->RecvRankTable(), HCCL_E_NOT_SUPPORT);
    RankTableInfo nextRankTable;
    EXPECT_EQ(rankInfoDetectClient_->WaitForNextBroadcast(nextRankTable, 2), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(rankInfoDetectClient_->remoteNetLayerCheck_.version, 0U);

    // 损坏的选层尾帧同样直接返回参数错误，不再抛异常。
    auto malformedBytes = BuildMockRankInfoMsg("");
    malformedBytes.push_back(0);
    EXPECT_EQ(rankInfoDetectClient_->ParseRankTable(malformedBytes), HCCL_E_PARA);
}

TEST_F(RankInfoDetectClientTest, Ut_ParseRankTable_When_FailedAgentIdListNotEmpty_Expect_NoEI0015Report)
{
    // 非空 failedAgentIdList 表示建链失败
    vector<char> rankInfoMsg = BuildMockRankInfoMsg("1,2");

    g_clientCapturedErrorCode.clear();
    MOCKER(RptInputErr).stubs().will(invoke(StubClientRptInputErr));

    EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectClient_->ParseRankTable(rankInfoMsg));
    // failedAgentIdList 分支已收敛为不再单独上报 EI0015，避免与 root 节点上报重复
    EXPECT_NE(g_clientCapturedErrorCode, "EI0015");
}

TEST_F(RankInfoDetectClientTest, Ut_ParseRankTable_When_RootRankIdBroadcast_Expect_Stored)
{
    vector<char> rankInfoMsg = BuildMockRankInfoMsg("", 3);

    EXPECT_EQ(HCCL_SUCCESS, rankInfoDetectClient_->ParseRankTable(rankInfoMsg));
    EXPECT_EQ(rankInfoDetectClient_->GetRootRankId(), 3U);
}

// M6：区间压缩编码与异常数据均走打印容错（截断 + 解析失败兜底），不影响后续校验路径
TEST_F(RankInfoDetectClientTest, Ut_ParseRankTable_When_CompressedOrMalformedList_Expect_NoThrow)
{
    g_clientCapturedErrorCode.clear();
    MOCKER(RptInputErr).stubs().will(invoke(StubClientRptInputErr));

    vector<char> compressedMsg = BuildMockRankInfoMsg("0-3,100-101,200");
    EXPECT_NO_THROW(rankInfoDetectClient_->ParseRankTable(compressedMsg));

    vector<char> truncatedMsg = BuildMockRankInfoMsg(std::string(512, '7') + ",8-9");
    EXPECT_NO_THROW(rankInfoDetectClient_->ParseRankTable(truncatedMsg)); // 超长列表截断为 256 前缀

    vector<char> malformedMsg = BuildMockRankInfoMsg("0-,x,y--2,");
    EXPECT_NO_THROW(rankInfoDetectClient_->ParseRankTable(malformedMsg)); // 解析失败打印原始串前缀
    // 遗言打印走 HCCL_ERROR 日志路径，不经 RptInputErr 错误码上报（打桩为行为观测，非死桩）
    EXPECT_TRUE(g_clientCapturedErrorCode.empty());
}

// 遗言打印核心逻辑可测化：截断 + 失败总数计数 + 畸形列表判别（行为由 FormatFailedAgentIdList 承载）
TEST_F(RankInfoDetectClientTest, Ut_FormatFailedAgentIdList_When_CompressedOrMalformed_Expect_TruncateAndCount)
{
    u64 failedCount = 0;
    bool parseOk = false;
    // 压缩列表原样输出；总数 = 区间求和（0-3→4 + 100-101→2 + 200→1 = 7）
    EXPECT_EQ(FormatFailedAgentIdList("0-3,100-101,200", failedCount, parseOk), "0-3,100-101,200");
    EXPECT_TRUE(parseOk);
    EXPECT_EQ(failedCount, 7U);

    // 合法但超长（257 段、513 字符 > 256）：输出截断为 256 前缀 + "..."，计数不截断
    std::string longList;
    for (u32 i = 0; i < MAX_FAILED_LIST_PRINT_LEN; ++i) {
        longList += "7,";
    }
    longList += "7";
    const std::string printed = FormatFailedAgentIdList(longList, failedCount, parseOk);
    EXPECT_TRUE(parseOk);
    EXPECT_EQ(failedCount, static_cast<u64>(MAX_FAILED_LIST_PRINT_LEN) + 1U);
    EXPECT_EQ(printed.size(), static_cast<size_t>(MAX_FAILED_LIST_PRINT_LEN) + 3U);
    EXPECT_EQ(printed.substr(0, MAX_FAILED_LIST_PRINT_LEN), longList.substr(0, MAX_FAILED_LIST_PRINT_LEN));
    EXPECT_EQ(printed.substr(MAX_FAILED_LIST_PRINT_LEN), "...");

    // 畸形列表（空段/非数字/倒序区间）：parseOk=false 计数无效，短串原样输出
    EXPECT_EQ(FormatFailedAgentIdList("0-,x,y--2,", failedCount, parseOk), "0-,x,y--2,");
    EXPECT_FALSE(parseOk);

    // 超大数字段（512 个 '7' 单段越界）：整体判解析失败，输出仍按前缀截断
    const std::string oversized = std::string(512, '7') + ",8-9";
    EXPECT_EQ(
        FormatFailedAgentIdList(oversized, failedCount, parseOk),
        oversized.substr(0, MAX_FAILED_LIST_PRINT_LEN) + "...");
    EXPECT_FALSE(parseOk);
}

// 失败 agent 上报窗口计算：设计余量（+SOCKET_LINK_EXTRA_TIMEOUT）+ clamp 上限 + 负哨兵/0 防御分支
TEST_F(RankInfoDetectClientTest, Ut_GetErrorBroadcastWaitSeconds_When_VariousLinkTimeout_Expect_ClampedWindow)
{
    EXPECT_EQ(GetErrorBroadcastWaitSeconds(60), 100U);                      // 正常配置：60 + 40
    EXPECT_EQ(GetErrorBroadcastWaitSeconds(120), 160U);                     // 默认配置：120 + 40
    EXPECT_EQ(GetErrorBroadcastWaitSeconds(0), SOCKET_LINK_EXTRA_TIMEOUT);  // 0 防御：纯余量
    EXPECT_EQ(GetErrorBroadcastWaitSeconds(-1), SOCKET_LINK_EXTRA_TIMEOUT); // 负哨兵：无 s32→u32 相加回绕
    EXPECT_EQ(GetErrorBroadcastWaitSeconds(7200), MAX_ERROR_BROADCAST_WAIT_SEC); // 合法域上限：clamp 300
}

TEST_F(RankInfoDetectClientTest, Ut_RecvRankTable_When_Normal_Expect_Success)
{
    vector<char> rankInfoMsg = BuildMockRankInfoMsg("");

    // 字节流转换为vector<char>格式
    MOCKER(aclrtMallocHostWithCfg).stubs().will(returnValue(1));
    std::vector<char> hostAlloc(MAX_BUFFER_LEN);
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue(static_cast<void*>(hostAlloc.data())));
    MOCKER(HrtFreeHost).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    void* msg = rankInfoMsg.data();
    u64 msgLen = rankInfoMsg.size();
    u64 revMsgLenOut = msgLen;
    u64& revMsgLen = revMsgLenOut;
    MOCKER_CPP(&SocketAgent::RecvMsg)
        .stubs()
        .with(outBoundP(msg, msgLen), outBound(revMsgLen), mockcpp::any())
        .will(returnValue(true));

    MOCKER_CPP(&RankInfoDetectClient::VerifyRankTable).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    EXPECT_EQ(rankInfoDetectClient_->RecvRankTable(), HcclResult::HCCL_SUCCESS);
    RankTableInfo nextRankTable;
    EXPECT_EQ(rankInfoDetectClient_->WaitForNextBroadcast(nextRankTable, 1), HCCL_SUCCESS);
    EXPECT_EQ(nextRankTable.rankCount, 1U);
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyTlsConsistency_When_AllRanksEnable_Expect_ReturnSuccess)
{
    BuildRankTableForTls(rankInfoDetectClient_->rankTable_, {TlsStatus::ENABLE, TlsStatus::ENABLE});

    HcclResult ret = rankInfoDetectClient_->VerifyTlsConsistency();

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyTlsConsistency_When_AllRanksDisable_Expect_ReturnSuccess)
{
    BuildRankTableForTls(rankInfoDetectClient_->rankTable_, {TlsStatus::DISABLE, TlsStatus::DISABLE});

    HcclResult ret = rankInfoDetectClient_->VerifyTlsConsistency();

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyTlsConsistency_When_EnableAndDisableMixed_Expect_ReturnParaError)
{
    BuildRankTableForTls(rankInfoDetectClient_->rankTable_, {TlsStatus::ENABLE, TlsStatus::DISABLE});

    HcclResult ret = rankInfoDetectClient_->VerifyTlsConsistency();

    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyTlsConsistency_When_KnownConsistentAndUnknownExists_Expect_ReturnSuccess)
{
    BuildRankTableForTls(rankInfoDetectClient_->rankTable_, {TlsStatus::ENABLE, TlsStatus::UNKNOWN});

    HcclResult ret = rankInfoDetectClient_->VerifyTlsConsistency();

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyTlsConsistency_When_KnownInconsistentAndUnknownExists_Expect_ReturnParaError)
{
    BuildRankTableForTls(
        rankInfoDetectClient_->rankTable_, {TlsStatus::ENABLE, TlsStatus::DISABLE, TlsStatus::UNKNOWN});

    HcclResult ret = rankInfoDetectClient_->VerifyTlsConsistency();

    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyHostDpuTlsConsistency_When_AllRanksEnable_Expect_ReturnSuccess)
{
    BuildRankTableForHostDpuTls(rankInfoDetectClient_->rankTable_, {TlsStatus::ENABLE, TlsStatus::ENABLE});

    HcclResult ret = rankInfoDetectClient_->VerifyHostDpuTlsConsistency();

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyHostDpuTlsConsistency_When_AllRanksDisable_Expect_ReturnSuccess)
{
    BuildRankTableForHostDpuTls(rankInfoDetectClient_->rankTable_, {TlsStatus::DISABLE, TlsStatus::DISABLE});

    HcclResult ret = rankInfoDetectClient_->VerifyHostDpuTlsConsistency();

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyHostDpuTlsConsistency_When_EnableAndDisableMixed_Expect_ReturnParaError)
{
    BuildRankTableForHostDpuTls(rankInfoDetectClient_->rankTable_, {TlsStatus::ENABLE, TlsStatus::DISABLE});
    g_reportedValues.clear();
    MOCKER(RptInputErr).stubs().will(invoke(StubRptInputErr));

    HcclResult ret = rankInfoDetectClient_->VerifyHostDpuTlsConsistency();

    EXPECT_EQ(ret, HCCL_E_PARA);
    ASSERT_EQ(g_reportedValues.size(), 3U);
    EXPECT_EQ(
        g_reportedValues[2], "\"All ranks are consistent. Current status: rankList for enabled tls: 0; "
                             "rankList for disabled tls: 1\"");
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyHostDpuTlsConsistency_When_KnownConsistentAndUnknownExists_Expect_Success)
{
    BuildRankTableForHostDpuTls(rankInfoDetectClient_->rankTable_, {TlsStatus::ENABLE, TlsStatus::UNKNOWN});

    HcclResult ret = rankInfoDetectClient_->VerifyHostDpuTlsConsistency();

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyHostDpuTlsConsistency_When_KnownInconsistentAndUnknownExists_Expect_ParaError)
{
    BuildRankTableForHostDpuTls(
        rankInfoDetectClient_->rankTable_, {TlsStatus::ENABLE, TlsStatus::DISABLE, TlsStatus::UNKNOWN});

    HcclResult ret = rankInfoDetectClient_->VerifyHostDpuTlsConsistency();

    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyHostDpuTlsConsistency_When_AllRanksUnknown_Expect_ReturnSuccess)
{
    BuildRankTableForHostDpuTls(rankInfoDetectClient_->rankTable_, {TlsStatus::UNKNOWN, TlsStatus::UNKNOWN});

    HcclResult ret = rankInfoDetectClient_->VerifyHostDpuTlsConsistency();

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyRankTable_When_HostDpuTlsInconsistent_Expect_E_PARA)
{
    rankInfoDetectClient_->rankSize_ = 3;
    BuildRankTableForHostDpuTls(
        rankInfoDetectClient_->rankTable_, {TlsStatus::ENABLE, TlsStatus::DISABLE, TlsStatus::UNKNOWN});
    g_reportedErrorCode.clear();
    g_reportedKeys.clear();
    g_reportedValues.clear();
    MOCKER(RptInputErr).stubs().will(invoke(StubRptInputErr));

    EXPECT_EQ(rankInfoDetectClient_->VerifyRankTable(), HCCL_E_PARA);

    EXPECT_EQ(g_reportedErrorCode, "EI0016");
    EXPECT_EQ(g_reportedKeys, std::vector<std::string>({"value", "variable", "expect"}));
    ASSERT_EQ(g_reportedValues.size(), 3U);
    EXPECT_EQ(g_reportedValues[0], "Disable");
    EXPECT_EQ(g_reportedValues[1], "\"hostDpuTls\"");
    EXPECT_EQ(
        g_reportedValues[2], "\"All ranks are consistent. Current status: rankList for enabled tls: 0; "
                             "rankList for disabled tls: 1; rankList for query failure tls: 2\"");
}

TEST_F(RankInfoDetectClientTest, Ut_CheckStatus_When_Timeout_Expect_E_TIMEOUT)
{
    EnvSocketConfig fakeEnvSocketConfig;
    fakeEnvSocketConfig.linkTimeOut = CfgField<s32>{"HCCL_CONNECT_TIMEOUT", s32(1), Str2T<s32>};
    fakeEnvSocketConfig.linkTimeOut.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetSocketConfig).stubs().will(returnValue(fakeEnvSocketConfig));
    MOCKER_CPP(&Socket::GetStatus).stubs().then(returnValue((SocketStatus)SocketStatus::CONNECTING));

    EXPECT_EQ(rankInfoDetectClient_->CheckStatus(), HCCL_E_TIMEOUT);
}

TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_EmptyRankLevelInfos_Expect_NoThrow)
{
    // Given: rankInfo with empty rankLevelInfos
    hccl::NewRankInfo rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceId = 0;
    rankInfo.localId = 0;
    // rankLevelInfos is empty by default

    MOCKER(HrtGetDevice).stubs().with(outBound(0)).will(returnValue(HcclResult::HCCL_SUCCESS));

    MockTopoPathAndBuild(BuildEmptyTopo);

    // When & Then: loop iterates zero times, returns without error
    EXPECT_EQ(rankInfoDetectClient_->HostListenPortDetect(rankInfo), HCCL_SUCCESS);
    EXPECT_EQ(rankInfo.hostPort, hccl::DEFAULT_VALUE_TCPPORT);
    hccl::PhyTopo::GetInstance()->Clear(); // 清理上一次测试的拓扑状态
}

TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_NoTopoGraph_Expect_NoThrow)
{
    // Given: rankInfo with rankLevelInfos but netLayer doesn't match any built topo graph
    hccl::NewRankInfo rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceId = 0;
    rankInfo.localId = 0;

    hccl::AddressInfo addrInfo;
    addrInfo.addr = IpAddress("192.168.1.1");
    addrInfo.socketPort_ = 0;
    hccl::RankLevelInfo levelInfo;
    levelInfo.netLayer = 0;
    levelInfo.rankAddrs.push_back(addrInfo);
    rankInfo.rankLevelInfos.push_back(levelInfo);

    MOCKER(HrtGetDevice).stubs().with(outBound(0)).will(returnValue(HcclResult::HCCL_SUCCESS));

    MockTopoPathAndBuild(BuildEmptyTopo);

    // When: PhyTopo built but netLayer=0 has no graph → GetTopoGraph returns nullptr
    // Then: skip nullptr graph, log debug, continue, return without error
    EXPECT_EQ(rankInfoDetectClient_->HostListenPortDetect(rankInfo), HCCL_SUCCESS);
    EXPECT_EQ(rankInfo.hostPort, hccl::DEFAULT_VALUE_TCPPORT);
    hccl::PhyTopo::GetInstance()->Clear(); // 清理上一次测试的拓扑状态
}

TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_MultipleRankLevelInfos_Expect_NoThrow)
{
    // Given: multiple rankLevelInfos, none with a matching topo graph
    hccl::NewRankInfo rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceId = 0;
    rankInfo.localId = 0;

    for (u32 i = 0; i < 3; i++) {
        hccl::AddressInfo addrInfo;
        addrInfo.addr = IpAddress(StringFormat("192.168.%u.1", i + 1));
        addrInfo.socketPort_ = 0;
        hccl::RankLevelInfo levelInfo;
        levelInfo.netLayer = i;
        levelInfo.rankAddrs.push_back(addrInfo);
        rankInfo.rankLevelInfos.push_back(levelInfo);
    }

    MOCKER(HrtGetDevice).stubs().with(outBound(0)).will(returnValue(HcclResult::HCCL_SUCCESS));

    MockTopoPathAndBuild(BuildEmptyTopo);

    // When & Then: iterate all level infos, all graphs nullptr, no throw
    EXPECT_EQ(rankInfoDetectClient_->HostListenPortDetect(rankInfo), HCCL_SUCCESS);
    EXPECT_EQ(rankInfo.hostPort, hccl::DEFAULT_VALUE_TCPPORT);
    hccl::PhyTopo::GetInstance()->Clear(); // 清理上一次测试的拓扑状态
}

TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_RdmaLinkEmptyRankAddrs_Expect_Continue)
{
    // Given: host RDMA topology exists at netLayer=3, but rankAddrs is empty
    hccl::NewRankInfo rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceId = 0;
    rankInfo.localId = 0;

    hccl::RankLevelInfo levelInfo;
    levelInfo.netLayer = 3; // HOST+ROCE link at netLayer=3
    // rankAddrs left EMPTY — triggers rankLevelInfo.rankAddrs.empty() check
    rankInfo.rankLevelInfos.push_back(levelInfo);

    MOCKER(HrtGetDevice).stubs().with(outBound(0)).will(returnValue(HcclResult::HCCL_SUCCESS));

    MockTopoPathAndBuild(BuildHostRdmaTopo);

    // When & Then: ROCE → RDMA but rankAddrs.empty() → skip, hostPort unchanged
    hccl::PhyTopo::GetInstance()->Clear();
    EXPECT_EQ(rankInfoDetectClient_->HostListenPortDetect(rankInfo), HCCL_SUCCESS);
    EXPECT_EQ(rankInfo.hostPort, hccl::DEFAULT_VALUE_TCPPORT);
    hccl::PhyTopo::GetInstance()->Clear(); // 清理上一次测试的拓扑状态
}

// basePort configured, portRange empty → listenPort = basePort + devPhyId
TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_BasePort_Expect_HostPortSet)
{
    // Given: HOST+ROCE link at netLayer=3, basePort configured
    EnvHostNicConfig fakeConfig;
    fakeConfig.hcclHostSocketPortRange = CfgField<std::vector<SocketPortRange>>{
        "HCCL_HOST_SOCKET_PORT_RANGE", {}, [](const std::string& s) -> std::vector<SocketPortRange> {
            return CastSocketPortRange(s, "HCCL_HOST_SOCKET_PORT_RANGE");
        }};
    fakeConfig.hcclHostSocketPortRange.isParsed = true;
    // hcclIfBasePort: configured to valid value 50000
    fakeConfig.hcclIfBasePort = CfgField<u32>{"HCCL_IF_BASE_PORT", 50000, Str2T<u32>};
    fakeConfig.hcclIfBasePort.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeConfig));
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&)).expects(never());

    hccl::NewRankInfo rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceId = 0;
    rankInfo.localId = 0;

    hccl::AddressInfo addrInfo;
    addrInfo.addr = IpAddress("192.168.1.1");
    hccl::RankLevelInfo levelInfo;
    levelInfo.netLayer = 3; // HOST+ROCE link at netLayer=3
    levelInfo.rankAddrs.push_back(addrInfo);
    rankInfo.rankLevelInfos.push_back(levelInfo);

    MOCKER(HrtGetDevice).stubs().with(outBound(0)).will(returnValue(HcclResult::HCCL_SUCCESS));

    MockTopoPathAndBuild(BuildHostRdmaTopo);

    // When: basePort valid → listenPort = basePort + devPhyId = 50000 + 0
    hccl::PhyTopo::GetInstance()->Clear();
    EXPECT_EQ(rankInfoDetectClient_->HostListenPortDetect(rankInfo), HCCL_SUCCESS);

    // Then: hostPort should be set to basePort + devPhyId
    EXPECT_EQ(rankInfo.hostPort, 50000u);

    // Cleanup: TearDown resets hostSocket_
    rankInfoDetectClient_->SocketTearDown(0);
    hccl::PhyTopo::GetInstance()->Clear(); // 清理上一次测试的拓扑状态
}

TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_PortRange_Expect_ListenOnDpuAddr)
{
    // Given: host port range is configured and the topology provides the DPU address.
    EnvHostNicConfig fakeConfig = BuildHostPortRangeConfig();
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeConfig));
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&))
        .expects(exactly(1))
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(0));

    hccl::NewRankInfo rankInfo = BuildHostRdmaRankInfo();

    MockTopoPathAndBuild(BuildHostRdmaTopo);

    // When: the DPU address is available.
    EXPECT_EQ(rankInfoDetectClient_->HostListenPortDetect(rankInfo), HCCL_SUCCESS);

    // Then: listen on the DPU address and allocate a port from the configured range.
    ASSERT_NE(rankInfoDetectClient_->hostSocket_, nullptr);
    EXPECT_EQ(
        rankInfoDetectClient_->hostSocket_->GetLocalIp().Describe(),
        rankInfo.rankLevelInfos[0].rankAddrs[0].addr.Describe());
    EXPECT_GE(rankInfo.hostPort, 50000u);
    EXPECT_LE(rankInfo.hostPort, 50001u);

    rankInfoDetectClient_->SocketTearDown(0);
    hccl::PhyTopo::GetInstance()->Clear();
}

TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_DpuAddrLinkDown_Expect_SkipListen)
{
    // Given: host port range is configured, but the DPU address is linkdown.
    EnvHostNicConfig fakeConfig = BuildHostPortRangeConfig();
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeConfig));
    MOCKER(HrtRaRdmaInit, s32(HrtNetworkMode, RaInterface&, RdmaHandle&))
        .expects(exactly(1))
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCP_ELINKDOWN));

    hccl::NewRankInfo rankInfo = BuildHostRdmaRankInfo();

    MockTopoPathAndBuild(BuildHostRdmaTopo);

    // When & Then: skip listening and continue communication domain initialization.
    EXPECT_EQ(rankInfoDetectClient_->HostListenPortDetect(rankInfo), HCCL_SUCCESS);
    EXPECT_EQ(rankInfo.hostPort, hccl::DEFAULT_VALUE_TCPPORT);
    EXPECT_EQ(rankInfoDetectClient_->hostSocket_, nullptr);

    hccl::PhyTopo::GetInstance()->Clear();
}

TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_NoHostLink_Expect_HostPortUnchanged)
{
    // Given: graph exists at netLayer=1, but links are DEVICE position instead of HOST
    // This tests the path where graph has edges but none match HOST position
    hccl::NewRankInfo rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceId = 0;
    rankInfo.localId = 0;

    hccl::AddressInfo addrInfo;
    addrInfo.addr = IpAddress("192.168.1.1");
    hccl::RankLevelInfo levelInfo;
    levelInfo.netLayer = 1; // netLayer=1 has DEV links, not HOST
    levelInfo.rankAddrs.push_back(addrInfo);
    rankInfo.rankLevelInfos.push_back(levelInfo);

    MOCKER(HrtGetDevice).stubs().with(outBound(0)).will(returnValue(HcclResult::HCCL_SUCCESS));

    MockTopoPathAndBuild(BuildHostRdmaTopo);

    // When: netLayer=1 graph exists but has no HOST position links
    hccl::PhyTopo::GetInstance()->Clear();
    EXPECT_EQ(rankInfoDetectClient_->HostListenPortDetect(rankInfo), HCCL_SUCCESS);

    // Then: hostPort should remain at default since no HOST RDMA link found
    EXPECT_EQ(rankInfo.hostPort, hccl::DEFAULT_VALUE_TCPPORT);
    hccl::PhyTopo::GetInstance()->Clear(); // 清理上一次测试的拓扑状态
}

TEST_F(RankInfoDetectClientTest, Ut_TearDown_HostSocketNull_Expect_EarlyReturn)
{
    // Given: hostSocket_ is nullptr (default state, no HostListenPortDetect called)

    // When & Then: TearDown should return immediately without any side effects
    EXPECT_NO_THROW(rankInfoDetectClient_->SocketTearDown(0));
}

// 抢占端口成功后，hostSocket_ 应登记到 SocketManager::GetServerSocketMap()，hostSocketRegistered_ 置 true
TEST_F(RankInfoDetectClientTest, Ut_SetupHostListenPort_PreemptSuccess_Expect_RegisteredToMap)
{
    // Given: portRange 空、basePort 无效 → 走默认 range + ListenPreempt 抢占分支
    EnvHostNicConfig fakeConfig;
    fakeConfig.hcclHostSocketPortRange = CfgField<std::vector<SocketPortRange>>{
        "HCCL_HOST_SOCKET_PORT_RANGE", {}, [](const std::string& s) -> std::vector<SocketPortRange> {
            return CastSocketPortRange(s, "HCCL_HOST_SOCKET_PORT_RANGE");
        }};
    fakeConfig.hcclHostSocketPortRange.isParsed = true;
    fakeConfig.hcclIfBasePort = CfgField<u32>{"HCCL_IF_BASE_PORT", HCCL_INVALID_PORT, Str2T<u32>};
    fakeConfig.hcclIfBasePort.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeConfig));

    // stub 的 ListenPreempt 会调 Socket::Listen(port) → HrtRaSocketTryListenOneStart 返回 true，usePort 设为 range
    // 首个端口

    IpAddress hostIp("192.168.1.8");
    uint32_t hostPort = 0;

    // When: 调 SetupHostListenPort 走抢占分支
    EXPECT_NO_THROW(rankInfoDetectClient_->SetupHostListenPort(0, 0, hostIp, hostPort));

    // Then: hostPort 被设为抢占端口，hostSocketRegistered_ 置 true
    EXPECT_NE(hostPort, 0u);
    EXPECT_TRUE(rankInfoDetectClient_->hostSocketRegistered_);
    EXPECT_NE(rankInfoDetectClient_->hostSocket_, nullptr);

    // Cleanup
    rankInfoDetectClient_->SocketTearDown(0);
    GlobalMockObject::verify();
}

// hostSocket_ 已登记 map 时，SocketTearDown 应跳过 Release/Destroy，只释放 shared_ptr
TEST_F(RankInfoDetectClientTest, Ut_SocketTearDown_Registered_Expect_SkipReleaseDestroy)
{
    // Given: 手动构造 hostSocket_ 并标记已登记
    IpAddress hostIp("192.168.1.8");
    auto socket = std::make_shared<Socket>(
        socketHandle, hostIp, 60001, hostIp, "hostport_preempt", SocketRole::SERVER, NicType::HOST_NIC_TYPE);
    rankInfoDetectClient_->hostSocket_ = socket;
    rankInfoDetectClient_->hostSocketRegistered_ = true;

    // mock Release 和 Destroy，期望不被调用
    MOCKER_CPP(&PreemptPortManager::Release).expects(never()).with(mockcpp::any());
    MOCKER_CPP(&HostSocketHandleManager::Destroy).expects(never()).with(mockcpp::any(), mockcpp::any());

    // When: 调 SocketTearDown
    EXPECT_NO_THROW(rankInfoDetectClient_->SocketTearDown(0));

    // Then: hostSocket_ 被置空，且 Release/Destroy 未被调用
    EXPECT_EQ(rankInfoDetectClient_->hostSocket_, nullptr);

    GlobalMockObject::verify();
}
