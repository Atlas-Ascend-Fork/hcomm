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
#include <iostream>
#include <unistd.h>

// 访问私有成员验证分段上界判定路径（与ut_rank_graph.cc同款惯例）
#define private public
#define protected public

#include "detour_service.h"
#include "phy_topo.h"
#include "phy_topo_builder.h"
#include "rank_graph_test_data_builder.h"
#include "rank_gph.h"
#include "rank_graph_builder.h"
#include "nlohmann/json.hpp"

#undef private
#undef protected

#include "rank_graph_builder_bridge.h"

using namespace Hccl;

class RankGraphBuilderTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankGraphBuilderTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RankGraphBuilderTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in RankGraphBuilderTest SetUP" << std::endl;
        GlobalMockObject::reset();
        PhyTopo::GetInstance()->Clear();
    }

    virtual void TearDown()
    {
        PhyTopo::GetInstance()->Clear();
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in RankGraphBuilderTest TearDown" << std::endl;
    }
};

TEST_F(RankGraphBuilderTest, Ut_Build_When_Normal_Expect_Success)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder rankGraphBuilder;
    TopoInfo topoInfo = test::MakeTwoPeerClosTopo();
    std::unique_ptr<RankGraph> rankGraph = rankGraphBuilder.RecoverBuild(test::MakeRankTable2p(), topoInfo, 0);
    EXPECT_NE(nullptr, rankGraph);
    ASSERT_NE(nullptr, rankGraphBuilder.topoInfo_);
    EXPECT_EQ(topoInfo.Describe(), rankGraphBuilder.topoInfo_->Describe());
    auto path1 = rankGraph->GetPaths(0, 0, 1);
    EXPECT_NE(1, path1.size());
    auto path2 = rankGraph->GetPaths(0, 1, 0);
    EXPECT_NE(1, path2.size());
}

// topo解析缺失：告警后写禁止切段值0并跳过建虚拟分组视图，不静默回落全域切段旧行为
TEST_F(RankGraphBuilderTest, Ut_Build_When_TopoInfoNull_Expect_WarningSkipAndZeroBound)
{
    MOCKER_CPP(&PhyTopoBuilder::Build).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&PhyTopoBuilder::GetTopoInfo).stubs().will(returnValue(std::shared_ptr<TopoInfo>()));
    MOCKER_CPP(&RankGraphBuilder::BuildFromRankTable).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&RankGraph::BuildVirtualGroups).expects(never()); // 空指针分支不得建虚拟分组视图
    MOCKER_CPP(&RankGraphBuilder::BuildPeer2PeerLinks).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::UpdateTopoInstForMyRankOnly).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::SetEndpointDesc).stubs();
    MOCKER_CPP(&DetourService::InsertDetourLinks).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph = rankGraphBuilder.Build(test::MakeRankTable2p(), "topo.json", 0);
    ASSERT_NE(nullptr, rankGraph);
    EXPECT_EQ(0U, rankGraph->physicalSuperPodLayer_); // 禁止切段：子图继承后同样不建视图
}
// 无UB兜底场景：rank table中本rank level0带pcie_fallback显式标记时，Build将标志持久化进RankGraph
TEST_F(RankGraphBuilderTest, Ut_Build_When_Level0PcieFallback_Expect_FlagPersistedInRankGraph)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder rankGraphBuilder;
    RankTableInfo rankTable = test::MakeRankTable2p();
    // 模拟rank_info_detect_client插入兜底level0时写入的pcie_fallback显式标记（本rank即rank0）
    rankTable.ranks[0].rankLevelInfos[0].pcieFallback = true;
    std::unique_ptr<RankGraph> rankGraph = rankGraphBuilder.RecoverBuild(rankTable, test::MakeTwoPeerClosTopo(), 0);
    EXPECT_NE(nullptr, rankGraph);
    EXPECT_EQ(true, rankGraph->IsLevel0PcieFallback());
}

// 常规场景：rank table无pcie_fallback标记时，构建出的RankGraph标志保持false
TEST_F(RankGraphBuilderTest, Ut_Build_When_Level0NotPcieFallback_Expect_FlagFalse)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(test::MakeRankTable2p(), test::MakeTwoPeerClosTopo(), 0);
    EXPECT_NE(nullptr, rankGraph);
    EXPECT_EQ(false, rankGraph->IsLevel0PcieFallback());
}

TEST_F(RankGraphBuilderTest, Ut_BuildRankGraph_When_Normal_Expect_Success)
{
    // when
    MOCKER_CPP(&RankGraphBuilder::BuildFromRankTable).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&RankGraph::InitInnerRanks).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::BuildPeer2PeerLinks).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&DetourService::InsertDetourLinks).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::UpdateTopoInstForMyRankOnly).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::SetEndpointDesc).stubs();
    // then
    RankGraphBuilder rankGraphBuilder;
    Hccl::RankTableInfo tmpRankTable;
    tmpRankTable.rankCount = 3;
    rankGraphBuilder.rankTable_ = make_unique<Hccl::RankTableInfo>(tmpRankTable);
    rankGraphBuilder.myRank_ = 0;
    EXPECT_NO_THROW(rankGraphBuilder.BuildRankGraph(false));
}

TEST_F(RankGraphBuilderTest, Ut_Build_When_1pRankTable_Expect_Success)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(test::MakeRankTable1p(), test::MakeOnePeerTopo(true), 0);
    EXPECT_NE(nullptr, rankGraph);
    auto path1 = rankGraph->GetPaths(0, 0, 1);
    EXPECT_EQ(0, path1.size());
    auto path2 = rankGraph->GetPaths(0, 1, 0);
    EXPECT_EQ(0, path2.size());
    vector<u32> subRankIds = {0};
    auto subRankGraph = rankGraph->CreateSubRankGraph(subRankIds);
    EXPECT_EQ(1, subRankGraph->GetInnerRankSize());
}

TEST_F(RankGraphBuilderTest, Ut_Build_When_OnePTopoFileWithoutEdge_Expect_Success)
{
    PhyTopo::GetInstance()->Clear();
    auto graph = std::make_shared<Graph<PhyTopo::Node, PhyTopo::Link>>();
    graph->AddNode(PhyTopo::Peer::GetId(0), std::make_shared<PhyTopo::Peer>(0));
    PhyTopo::GetInstance()->AddTopoGraph(0, graph);
    PhyTopo::GetInstance()->InitFinish();
    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(test::MakeRankTable1p(), test::MakeOnePeerTopo(false), 0);
    EXPECT_NE(nullptr, rankGraph);
    auto rankSize = rankGraph->GetRankSize();
    EXPECT_EQ(rankSize, 1);
    auto peer = rankGraph->GetPeer(rankGraph->GetMyRank());
    ASSERT_NE(peer, nullptr);
    EXPECT_EQ(peer->GetLocalId(), 0);
}

TEST_F(RankGraphBuilderTest, Ut_BuildFromRankTable_When_NetLayerInconsistent_Expect_InvalidParamsException)
{
    // 校验BuildFromRankTable的Add(RankId, Peer)
    // when
    MOCKER_CPP(&PhyTopoBuilder::Build).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&RankGraph::InitInnerRanks).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&DetourService::InsertDetourLinks).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::BuildPeer2PeerLinks).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::AddFabricInfo).stubs().will(ignoreReturnValue());
    // then
    RankGraphBuilder rankGraphBuilder;
    RankTableInfo rankTable = test::MakeRankTable2p();
    rankTable.ranks[0].rankLevelInfos.emplace_back(test::MakeRankLevel(
        2, "missing-layer", NetType::CLOS,
        {
            test::MakeAddress("192.168.200.1", {"0/1"}),
        }));
    EXPECT_THROW(rankGraphBuilder.Build(rankTable, "topo.json", 0), InvalidParamsException);
}

TEST_F(RankGraphBuilderTest, ut_Build_When_4pRankTable_Expect_Success)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(test::MakeRankTable4pForBuilder(), test::MakeFourPeerBuilderTopo(), 0);
    EXPECT_NE(nullptr, rankGraph);
    std::vector<std::string> netIds = {"az0-rack0", "az0", "all"};
    for (s32 rankId = 0; rankId < 3; rankId++) {
        for (u32 netLayer = 0; netLayer < 3; netLayer++) {
            const NetInstance* fabGroup = rankGraph->GetNetInstanceByRankId(netLayer, rankId);
            EXPECT_EQ(netIds[netLayer], fabGroup->GetNetInstId());

            EXPECT_EQ(true, fabGroup->HasNode(NetInstance::Peer(rankId, 0, 0, 0).GetLocalId()));
            if (netLayer == 0) {
                EXPECT_EQ(NetType::TOPO_FILE_DESC, fabGroup->GetNetType());
            } else if (netLayer == 1) {
                EXPECT_EQ(NetType::CLOS, fabGroup->GetNetType());
            } else if (netLayer == 2) {
                EXPECT_EQ(NetType::CLOS, fabGroup->GetNetType());
            }
        }
    }

    std::vector<NetInstance::Path> pathsLayer0 = rankGraph->GetPaths(0, 0, 1);
    EXPECT_EQ(1, pathsLayer0.size());
    // GetPaths检查边0 - 1的level1的边 peer0->net0、net0->peer1、peer0->net1、net1->peer1
    std::vector<NetInstance::Path> pathsLayer1 = rankGraph->GetPaths(1, 0, 1);
    std::vector<std::string> ips1 = {"IpAddress[AF=v4, addr=192.168.101.1]", "IpAddress[AF=v4, addr=124.112.1.1]"};
    std::vector<std::string> ips2 = {"IpAddress[AF=v4, addr=192.168.101.11]", "IpAddress[AF=v4, addr=124.112.1.4]"};
    EXPECT_EQ(1, pathsLayer1.size());
    EXPECT_EQ(2, pathsLayer1[0].links.size());

    EXPECT_EQ(2, pathsLayer1[0].links[0].GetHop());
    EXPECT_EQ(LinkType::PEER2NET, pathsLayer1[0].links[0].GetType());
    EXPECT_EQ(std::set<Hccl::LinkProtocol>{LinkProtocol::UB_CTP}, pathsLayer1[0].links[0].GetLinkProtocols());
    EXPECT_EQ(0, pathsLayer1[0].links[0].GetSourceNode()->GetNodeId());
    EXPECT_EQ(4294967296, pathsLayer1[0].links[0].GetTargetNode()->GetNodeId());
    EXPECT_EQ(htonl(0xC0A86501), pathsLayer1[0].links[0].GetSourceIface()->GetAddr().GetBinaryAddress().addr.s_addr);

    EXPECT_EQ(2, pathsLayer1[0].links[1].GetHop());
    EXPECT_EQ(LinkType::PEER2NET, pathsLayer1[0].links[1].GetType());
    EXPECT_EQ(std::set<Hccl::LinkProtocol>{LinkProtocol::UB_CTP}, pathsLayer1[0].links[0].GetLinkProtocols());
    EXPECT_EQ(4294967296, pathsLayer1[0].links[1].GetSourceNode()->GetNodeId());
    EXPECT_EQ(1, pathsLayer1[0].links[1].GetTargetNode()->GetNodeId());
    EXPECT_EQ(htonl(0xC0A8650B), pathsLayer1[0].links[1].GetTargetIface()->GetAddr().GetBinaryAddress().addr.s_addr);

    std::vector<NetInstance::Path> pathsLayer2 = rankGraph->GetPaths(2, 2, 3);
    EXPECT_EQ(1, pathsLayer1.size());
    EXPECT_EQ(2, pathsLayer1[0].links.size());
}

TEST_F(RankGraphBuilderTest, Ut_RankGraphBuilderRecoverBuild_When_Invalid_Expect_InvalidParamsException)
{
    RankTableInfo rankTableInfo;
    TopoInfo topoInfo;
    RankGraphBuilder rankGraphBuilder;
    EXPECT_THROW(rankGraphBuilder.RecoverBuild(rankTableInfo, topoInfo, 0), InvalidParamsException);
}
namespace {
constexpr RankId RANK_0 = 0;
constexpr RankId RANK_1 = 1;
constexpr RankId RANK_2 = 2;
constexpr RankId RANK_3 = 3;
constexpr u32 TARGET_LAYER = 1;
constexpr u32 INTER_LAYER = 2;
const std::string TARGET_INST_ID = "sp_0_idx_0";

// 4rank超节点表：L0=server实例(每实例2个rank)、L1=sp_0_idx_0(4个rank)、L2=cluster(4个rank)。
RankTableInfo MakeSuperpodTable()
{
    std::vector<NewRankInfo> ranks;
    for (u32 rankId = 0; rankId < 4; ++rankId) {
        const std::string ip = "192.0.2." + std::to_string(10 + rankId);
        std::vector<RankLevelInfo> levels = {
            test::MakeRankLevel(
                0, rankId < 2 ? "srv_0" : "srv_1", NetType::TOPO_FILE_DESC, {test::MakeAddress(ip, {"0/0"})}),
            test::MakeRankLevel(TARGET_LAYER, TARGET_INST_ID, NetType::CLOS, {test::MakeAddress(ip, {"1/0"})}),
            test::MakeRankLevel(INTER_LAYER, "cluster", NetType::CLOS, {test::MakeAddress(ip, {"2/0"})}),
        };
        ranks.push_back(test::MakeRankInfo(rankId, rankId, rankId, levels));
    }
    return test::MakeRankTable(ranks);
}

// 配套topo：L0框内直连(UB)、L1超节点层(UB)、L2跨超层(RoCE)。
TopoInfo MakeSuperpodTopo(bool withInterRoce = true)
{
    std::vector<EdgeInfo> edges = {
        test::MakeEdge(0, LinkType::PEER2PEER, 0, {"0/0"}, 1, {"0/0"}, LinkProtocol::UB_CTP),
        test::MakeEdge(0, LinkType::PEER2PEER, 2, {"0/0"}, 3, {"0/0"}, LinkProtocol::UB_CTP),
    };
    for (u32 localId = 0; localId < 4; ++localId) {
        edges.push_back(
            test::MakeEdge(TARGET_LAYER, LinkType::PEER2NET, localId, {"1/0"}, 0, {}, LinkProtocol::UB_CTP));
        edges.push_back(test::MakeEdge(
            INTER_LAYER, LinkType::PEER2NET, localId, {"2/0"}, 0, {},
            withInterRoce ? LinkProtocol::ROCE : LinkProtocol::UB_CTP));
    }
    return test::MakeTopoInfo({0, 1, 2, 3}, edges);
}
} // namespace

class TableRewriteBuildTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() { PhyTopo::GetInstance()->Clear(); }
    virtual void TearDown()
    {
        PhyTopo::GetInstance()->Clear();
        GlobalMockObject::verify();
    }
};

// FindSuperPodLayer判定跨超层为最低已注册层(NOT_SUPPORT)：告警后写禁止切段值0、不建虚拟分组视图
TEST_F(TableRewriteBuildTest, Ut_RecoverBuild_When_FindNotSupport_Expect_ZeroBoundNoView)
{
    MOCKER(Hccl::FindSuperPodLayer).stubs().will(returnValue(HcclResult::HCCL_E_NOT_SUPPORT));
    MOCKER_CPP(&RankGraph::BuildVirtualGroups).expects(never()); // 告警跳过分支不得建虚拟分组视图

    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(bridge, nullptr);
    RankGraphBuildResult result;
    ASSERT_EQ(bridge->recoverBuild(MakeSuperpodTable(), MakeSuperpodTopo(), RANK_0, true, result), HCCL_SUCCESS);
    auto graph = result.rankGraph;
    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(0U, graph->physicalSuperPodLayer_); // 禁止切段：子图继承后同样不建
}

// FindSuperPodLayer全表无已注册层(NOT_FOUND)：告警后写禁止切段值0、不建虚拟分组视图
TEST_F(TableRewriteBuildTest, Ut_RecoverBuild_When_FindNotFound_Expect_ZeroBoundNoView)
{
    MOCKER(Hccl::FindSuperPodLayer).stubs().will(returnValue(HcclResult::HCCL_E_NOT_FOUND));
    MOCKER_CPP(&RankGraph::BuildVirtualGroups).expects(never()); // 告警跳过分支不得建虚拟分组视图

    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(bridge, nullptr);
    RankGraphBuildResult result;
    ASSERT_EQ(bridge->recoverBuild(MakeSuperpodTable(), MakeSuperpodTopo(), RANK_0, true, result), HCCL_SUCCESS);
    auto graph = result.rankGraph;
    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(0U, graph->physicalSuperPodLayer_); // 禁止切段：子图继承后同样不建
}

// 成功分支：判定出的物理超节点层写入图供BuildVirtualGroups与子图继承；层值超过上界的层保持物理口径
TEST_F(TableRewriteBuildTest, Ut_RecoverBuild_When_FindSucceeded_Expect_BoundWrittenAndLayersAboveBoundPhysical)
{
    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(bridge, nullptr);
    RankGraphBuildResult result;
    ASSERT_EQ(bridge->recoverBuild(MakeSuperpodTable(), MakeSuperpodTopo(), RANK_0, true, result), HCCL_SUCCESS);
    auto graph = result.rankGraph;
    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(TARGET_LAYER, graph->physicalSuperPodLayer_); // 判定值1，非默认8亦非禁止值0

    std::vector<u32> rankList;
    u32 rankNum = 0;
    EXPECT_EQ(graph->GetLocalInstRanks(INTER_LAYER, rankList, rankNum), HCCL_SUCCESS);
    EXPECT_EQ(rankNum, 4U); // 层2超过上界(1)：不切段，按物理实例成员口径返回
    EXPECT_EQ(rankList, std::vector<u32>({RANK_0, RANK_1, RANK_2, RANK_3}));
}

TEST_F(RankGraphBuilderTest, Ut_BuildAndRecover_When_SourceChanges_Expect_GroupingPolicyAndPhysicalTopology)
{
    std::vector<NewRankInfo> ranks;
    std::vector<EdgeInfo> edges;
    for (u32 rank = 0; rank < 4; ++rank) {
        const auto addr = test::MakeAddress("192.168.0." + std::to_string(rank + 1), {"0/6"});
        ranks.push_back(test::MakeRankInfo(
            rank, rank, rank,
            {test::MakeRankLevel(0, "server" + std::to_string(rank), NetType::TOPO_FILE_DESC, {addr}),
             test::MakeRankLevel(1, "pod" + std::to_string(rank % 2), NetType::CLOS, {addr}),
             test::MakeRankLevel(2, "global", NetType::CLOS, {addr})}));
        for (u32 layer = 0; layer < 3; ++layer) {
            edges.push_back(test::MakeEdge(layer, LinkType::PEER2NET, rank, {"0/6"}));
        }
    }
    const auto table = test::MakeRankTable(ranks);
    const auto topo = test::MakeTopoInfo({0, 1, 2, 3}, edges);
    nlohmann::json jsonTable = {{"version", "2.0"}, {"rank_count", 4}, {"rank_list", nlohmann::json::array()}};
    for (const auto& rank : ranks) {
        nlohmann::json levels = nlohmann::json::array();
        for (const auto& level : rank.rankLevelInfos) {
            levels.push_back(
                {{"net_layer", level.netLayer},
                 {"net_instance_id", level.netInstId},
                 {"net_type", level.netLayer == 0 ? "TOPO_FILE_DESC" : "CLOS"},
                 {"rank_addr_list",
                  {{{"addr_type", "IPV4"},
                    {"addr", "192.168.0." + std::to_string(rank.rankId + 1)},
                    {"ports", {"0/6"}},
                    {"plane_id", "0"}}}}});
        }
        jsonTable["rank_list"].push_back(
            {{"rank_id", rank.rankId},
             {"device_id", rank.deviceId},
             {"local_id", rank.localId},
             {"level_list", levels}});
    }
    // Only replace file loading; use the real physical topology and graph builders.
    MOCKER_CPP(&PhyTopoBuilder::Build).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&PhyTopoBuilder::GetTopoInfo).stubs().will(returnValue(std::make_shared<TopoInfo>(topo)));
    for (u32 mode = 0; mode < 4; ++mode) {
        const bool enabled = (mode % 2 != 0);
        PhyTopo::GetInstance()->Clear();
        RankGraphBuilder builder;
        std::unique_ptr<RankGraph> graph;
        if (mode < 2) {
            graph = builder.RecoverBuild(table, topo, 0, enabled);
        } else {
            PhyTopoBuilder::GetInstance().RecoverBuild(topo);
            graph
                = enabled ? builder.Build(table, "unused.json", 0) : builder.Build(jsonTable.dump(), "unused.json", 0);
        }
        ASSERT_NE(graph, nullptr);
        EXPECT_EQ(graph->IsVirtualGroupsEnabled(), enabled);
        std::vector<u32> localRanks;
        u32 count = 0;
        ASSERT_EQ(graph->GetLocalInstRanks(1, localRanks, count), HCCL_SUCCESS);
        EXPECT_EQ(localRanks, enabled ? std::vector<u32>({0}) : std::vector<u32>({0, 2}));
        EXPECT_EQ(graph->GetNetInstanceByRankId(1, 0)->GetRankIds().size(), 2U);
        EXPECT_FALSE(graph->GetPaths(1, 0, 2).empty());
    }
}
