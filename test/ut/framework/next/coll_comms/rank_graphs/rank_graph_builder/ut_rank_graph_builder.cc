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
#include <mockcpp/ApiHook.h>
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <new>
#include <unistd.h>
#include <cstdint>
#include <map>
#include <type_traits>
#include <utility>
#include "hccl/hccl_comm.h"
#include "coll_comm_config.h"

#define private public
#define protected public

#include "detour_service.h"
#include "internal_exception.h"
#include "network_api_exception.h"
#include "phy_topo.h"
#include "phy_topo_builder.h"
#include "rank_graph_test_data_builder.h"
#include "rank_gph.h"
#include "rank_graph_builder.h"
#include "rank_graph_builder_bridge.h"
#include "orion_adapter_hccp.h"
#include <fstream>
#include "nlohmann/json.hpp"
#include "json/json_api.h"
#include "json/topo/topo_info_reader.h"

#undef private
#undef protected

#include "rank_graph_builder_bridge.h"
#include "rank_graph_v2.h"

using namespace Hccl;

class RankGraphBuilderTest : public testing::Test {
protected:
    bool savedUbxClosSplit_ = false;

    static void SetUpTestCase() { std::cout << "RankGraphBuilderTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RankGraphBuilderTest TearDown" << std::endl; }

    static RankGraphBuilder::LayerSelector MakeLayerSelector(const std::string& expression)
    {
        hccl::NetLayerSelection selection;
        const auto ret = hccl::NetLayerSelection::Parse(expression, selection);
        EXPECT_EQ(HCCL_SUCCESS, ret);
        if (expression.empty()) {
            return {};
        }
        return [selection, ret](u32 available, u32& effective) {
            return ret == HCCL_SUCCESS ? selection.Resolve(available, effective) : ret;
        };
    }

    static std::unique_ptr<RankGraph> BuildGraph(
        const std::string& expression = "", const RankTableInfo& ranks = test::MakeRankTable4pForBuilder(),
        const hccl::TopoInfo& topo = test::MakeFourPeerBuilderTopo())
    {
        RankGraphBuilder builder(MakeLayerSelector(expression));
        return builder.RecoverBuild(ranks, topo, 0);
    }

    static void PrepareMockedTopoFile(const std::shared_ptr<hccl::TopoInfo>& topo)
    {
        // 只替代文件加载，物理图和返回元数据始终使用同一输入。
        PhyTopo::GetInstance()->Clear();
        ASSERT_EQ(HCCL_SUCCESS, hccl::PhyTopoBuilder::GetInstance().RecoverBuild(*topo));
        MOCKER_CPP(&hccl::PhyTopoBuilder::Build).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&hccl::PhyTopoBuilder::GetTopoInfo).stubs().will(returnValue(topo));
    }

    static void CheckSelectedGraph(const RankGraph* graph, const std::set<u32>& expected)
    {
        ASSERT_NE(nullptr, graph);
        EXPECT_EQ(expected, graph->GetLevels(0));
        EXPECT_EQ(4U, graph->GetRankSize());
        EXPECT_EQ(4U, graph->GetInnerRankSize());
        for (u32 layer = 0; layer < MAX_NET_LAYER; ++layer) {
            SCOPED_TRACE(layer);
            EXPECT_EQ(expected.count(layer) == 0, graph->GetPaths(layer, 0, 1).empty());
        }
    }

    template <typename Query>
    static void CheckEndpointAccess(RankGraph& source, Query& query, u32 layer, u32 rank, HcclResult expected)
    {
        uint32_t count = 0;
        ASSERT_EQ(HCCL_SUCCESS, source.GetEndpointNum(layer, 0, &count));
        ASSERT_GT(count, 0U);
        std::vector<EndpointDesc> endpoints(count);
        ASSERT_EQ(HCCL_SUCCESS, source.GetEndpointDesc(layer, 0, &count, endpoints.data()));
        EndpointAttrBwCoeff bandwidth = 0;
        EXPECT_EQ(
            expected,
            query.GetEndpointInfo(rank, &endpoints[0], ENDPOINT_ATTR_BW_COEFF, sizeof(bandwidth), &bandwidth));
    }

    static void CheckGraphView(hccl::RankGraphV2& query, RankGraph& graph, const std::set<u32>& expectedLayers)
    {
        uint32_t* layers = nullptr;
        uint32_t count = 0;
        ASSERT_EQ(HCCL_SUCCESS, query.GetNetLayers(&layers, &count));
        ASSERT_EQ(expectedLayers.size(), count);
        ASSERT_NE(nullptr, layers);
        EXPECT_EQ(expectedLayers, std::set<u32>(layers, layers + count));
        void* exported = nullptr;
        uint32_t length = 0;
        ASSERT_EQ(HCCL_SUCCESS, query.GetRankGraphInfo(static_cast<GraphType>(0), &exported, &length));
        EXPECT_EQ(&graph, exported);
        EXPECT_EQ(sizeof(RankGraph), length);
        EXPECT_EQ(HCCL_E_PTR, query.GetRankGraphInfo(static_cast<GraphType>(0), nullptr, &length));
        EXPECT_EQ(HCCL_E_PTR, query.GetRankGraphInfo(static_cast<GraphType>(0), &exported, nullptr));
    }

    static void CheckLayer0MetadataOnly(
        hccl::RankGraphV2& query, RankGraph& graph, const std::vector<u32>& sizes, const std::vector<u32>& ranks)
    {
        uint32_t* values = nullptr;
        uint32_t count = 0;
        ASSERT_EQ(HCCL_SUCCESS, query.GetInstSizeListByNetLayer(0, &values, &count));
        ASSERT_NE(nullptr, values);
        EXPECT_EQ(sizes, std::vector<u32>(values, values + count));
        auto* savedSizes = values;
        ASSERT_EQ(HCCL_SUCCESS, query.GetInstRanksByNetLayer(0, &values, &count));
        ASSERT_NE(nullptr, values);
        EXPECT_EQ(ranks, std::vector<u32>(values, values + count));
        ASSERT_EQ(HCCL_SUCCESS, query.GetInstSizeByNetLayer(0, &count));
        EXPECT_EQ(ranks.size(), count);
        ASSERT_EQ(HCCL_SUCCESS, query.GetInstSizeListByNetLayer(0, &values, &count));
        EXPECT_EQ(savedSizes, values);
        CommLink sentinel{};
        CommLink* links = &sentinel;
        count = 99;
        ASSERT_EQ(HCCL_SUCCESS, query.GetLinks(0, 0, 1, &links, &count));
        EXPECT_EQ(nullptr, links);
        EXPECT_EQ(0U, count);
        EXPECT_EQ(HCCL_E_PARA, query.GetLinks(0, 0, graph.GetRankSize(), &links, &count));
        EXPECT_EQ(HCCL_E_PTR, query.GetLinks(0, 0, 1, nullptr, &count));
        EXPECT_EQ(HCCL_E_PTR, query.GetLinks(0, 0, 1, &links, nullptr));
        EXPECT_EQ(HCCL_E_PARA, query.GetEndpointNum(0, 0, &count));
        EndpointDesc endpoint{};
        EXPECT_EQ(HCCL_E_PARA, query.GetEndpointDesc(0, 0, &count, &endpoint));
        EXPECT_TRUE(graph.GetPaths(0, 0, 1).empty());
        EXPECT_EQ(0U, graph.GetLayerRanks(0));
        ASSERT_NO_FATAL_FAILURE(CheckGraphView(query, graph, graph.GetLevels(graph.GetMyRank())));
    }

    virtual void SetUp()
    {
        std::cout << "A Test case in RankGraphBuilderTest SetUP" << std::endl;
        setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
        GlobalMockObject::reset();
        PhyTopo::GetInstance()->Clear();
        savedUbxClosSplit_ = hccl::PhyTopoBuilder::ubxClosSplit_;
    }

    virtual void TearDown()
    {
        PhyTopo::GetInstance()->Clear();
        hccl::PhyTopoBuilder::ubxClosSplit_ = savedUbxClosSplit_;
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        unsetenv("HCCL_DFS_CONFIG");
        std::cout << "A Test case in RankGraphBuilderTest TearDown" << std::endl;
    }
};

TEST_F(RankGraphBuilderTest, Ut_ConstructConnInterface_When_UbFabric_Expect_Peer2Net)
{
    const std::set<std::string> ports = {"port0", "port1"};
    const std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    auto phyIface
        = std::make_shared<PhyTopo::ConnInterface>(ports, AddrPosition::DEVICE, LinkType::PEER2NET, protocols);
    const IpAddress addr("192.168.1.1");
    const std::map<std::string, std::vector<IpAddress>> portAddrMap = {{"port0", {addr}}, {"port1", {addr}}};

    auto ifaces = ConstructConnIFromPhyTopoConnIAndPortMap(
        phyIface, portAddrMap, Hccl::TopoType::CLOS, 0, ConnIfaceBuildCtx{0U, LinkType::PEER2NET});

    ASSERT_EQ(1U, ifaces.size());
    EXPECT_EQ(LinkType::PEER2NET, ifaces[0]->GetLinkType());
    EXPECT_EQ(addr, ifaces[0]->GetAddr());
    EXPECT_EQ(ports, ifaces[0]->GetPorts());
    EXPECT_EQ(protocols, ifaces[0]->GetLinkProtocols());
}

TEST_F(RankGraphBuilderTest, Ut_ConstructConnInterface_When_UbPeer_Expect_Peer2Peer)
{
    const std::set<std::string> ports = {"port0"};
    const std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    auto phyIface
        = std::make_shared<PhyTopo::ConnInterface>(ports, AddrPosition::DEVICE, LinkType::PEER2PEER, protocols);
    const IpAddress addr("192.168.1.1");
    const std::map<std::string, std::vector<IpAddress>> portAddrMap = {{"port0", {addr}}};

    auto ifaces = ConstructConnIFromPhyTopoConnIAndPortMap(
        phyIface, portAddrMap, Hccl::TopoType::MESH_1D, 0, ConnIfaceBuildCtx{0U});

    ASSERT_EQ(1U, ifaces.size());
    EXPECT_EQ(LinkType::PEER2PEER, ifaces[0]->GetLinkType());
    EXPECT_EQ(ports, ifaces[0]->GetPorts());
}

TEST_F(RankGraphBuilderTest, Ut_RecoverBuild_When_LayersShareEid_Expect_EndpointCountsMatch)
{
    const std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP, LinkProtocol::UB_MEM};
    const std::vector<std::string> eids = {"000000000000002000100000df001007", "000000000000002000100000df001008"};
    std::vector<hccl::NewRankInfo> ranks;
    std::vector<EdgeInfo> edges;
    for (u32 rankId = 0; rankId < eids.size(); ++rankId) {
        hccl::AddressInfo address;
        address.addr = IpAddress(IpAddress::StrToEID(eids[rankId]));
        address.addrType = AddrType::EID;
        address.ports = {"0/6"};
        address.planeId = "plane0";
        ranks.push_back(test::MakeRankInfo(
            rankId, rankId, rankId,
            {test::MakeRankLevel(0, "local", NetType::TOPO_FILE_DESC, {address}),
             test::MakeRankLevel(1, "global", NetType::CLOS, {address})}));
        for (u32 layer = 0; layer < 2; ++layer) {
            auto edge = test::MakeEdge(layer, LinkType::PEER2NET, rankId, {"0/6"});
            edge.protocols = protocols;
            edges.push_back(edge);
        }
    }

    RankGraphBuilder builder;
    auto graph = builder.RecoverBuild(test::MakeRankTable(ranks, true), test::MakeTopoInfo({0, 1}, edges), 0);
    ASSERT_NE(nullptr, graph);
    auto peer = graph->GetPeer(0);
    ASSERT_NE(nullptr, peer);
    auto layer0Ifaces = peer->GetIfacesByLayer(0);
    auto layer1Ifaces = peer->GetIfacesByLayer(1);
    ASSERT_EQ(1U, layer0Ifaces.size());
    ASSERT_EQ(1U, layer1Ifaces.size());
    ASSERT_NE(layer0Ifaces[0], layer1Ifaces[0]);
    EXPECT_EQ(LinkType::PEER2NET, layer0Ifaces[0]->GetLinkType());
    EXPECT_EQ(*layer0Ifaces[0], *layer1Ifaces[0]);

    // Exercise the actual cross-layer overwrite rather than matching an iface to itself.
    const auto endpointMap = peer->GetEndpointToIfaceMap();
    ASSERT_EQ(2U, endpointMap.size());
    for (const auto& entry : endpointMap) {
        EXPECT_EQ(layer1Ifaces[0], entry.second);
    }
    for (u32 layer = 0; layer < 2; ++layer) {
        uint32_t num = 0;
        ASSERT_EQ(HCCL_SUCCESS, graph->GetEndpointNum(layer, 0, &num));
        ASSERT_EQ(2U, num);
        std::vector<EndpointDesc> descs(num);
        uint32_t descNum = num;
        ASSERT_EQ(HCCL_SUCCESS, graph->GetEndpointDesc(layer, 0, &descNum, descs.data()));
        ASSERT_EQ(num, descNum);
        EXPECT_NE(descs[0].protocol, descs[1].protocol);
        for (const auto& desc : descs) {
            // IPoURMA represents an EID-backed IpAddress as IPv6 while preserving all EID bytes.
            EXPECT_EQ(COMM_ADDR_TYPE_IP_V6, desc.commAddr.type);
            const auto expectedEid = IpAddress::StrToEID(eids[0]);
            EXPECT_EQ(0, memcmp(desc.commAddr.addr6.s6_addr, expectedEid.raw, URMA_EID_LEN));
        }
    }
}

TEST_F(RankGraphBuilderTest, Ut_CheckSelectedLayers_When_NonzeroRank_Expect_ReadLocalLayers)
{
    RankGraphBuilder builder(MakeLayerSelector("not(0)"));
    builder.rankTable_ = std::make_unique<RankTableInfo>(test::MakeRankTable4pForBuilder());
    builder.myRank_ = 3;
    auto& levels = builder.rankTable_->ranks.at(3).rankLevelInfos;
    ASSERT_GE(levels.size(), 2U);
    levels.resize(2);
    levels[0].netLayer = 0;
    levels[1].netLayer = 3;
    ASSERT_EQ(HCCL_SUCCESS, builder.CheckSelectedLayers());
    EXPECT_EQ(0x09U, builder.availableLayers_);
    ASSERT_TRUE(builder.selectedLayers_.has_value());
    EXPECT_EQ(0x08U, builder.selectedLayers_.value());
}

TEST_F(RankGraphBuilderTest, Ut_SelectedBuild_When_UbxRestore_Expect_SelectFinalLayers)
{
    const struct {
        bool split;
        u32 meshGroupSize;
        bool layer1;
        bool layer3;
    } shapes[]
        = {{true, 2, true, true},
           {true, 2, true, false},
           {true, 3, true, true},
           {false, 2, true, true},
           {true, 2, false, true}};
    for (const auto& shape : shapes) {
        SCOPED_TRACE(
            testing::Message() << shape.split << ":" << shape.meshGroupSize << ":" << shape.layer1 << ":"
                               << shape.layer3);
        auto ranks = test::MakeRankTable4pForBuilder();
        auto topo = test::MakeFourPeerBuilderTopo();
        for (auto& rank : ranks.ranks) {
            auto& levels = rank.rankLevelInfos;
            levels[0].netInstId = "mesh" + std::to_string(rank.rankId / shape.meshGroupSize);
            levels[2].netLayer = 3;
            if (!shape.layer3) {
                levels.pop_back();
            }
            if (!shape.layer1) {
                // 原生 {0,3} 的 CLOS 地址本来就在 layer 0，不需要执行拆层还原。
                levels[0].netInstId = levels[1].netInstId;
                levels[0].rankAddrs.insert(
                    levels[0].rankAddrs.end(), levels[1].rankAddrs.begin(), levels[1].rankAddrs.end());
                levels[0].portAddrMap.insert(levels[1].portAddrMap.begin(), levels[1].portAddrMap.end());
                levels.erase(levels.begin() + 1);
            }
        }
        topo.edges[3] = std::move(topo.edges[2]);
        topo.edges.erase(2);
        for (auto& edge : topo.edges[3]) {
            edge.netLayer = 3;
        }
        // 模拟物理图已复制 CLOS 边到 layer 1，layer 0 仍保留其副本。
        for (auto edge : topo.edges[1]) {
            edge.netLayer = 0;
            topo.edges[0].push_back(edge);
        }
        topo.edgeCount += topo.edges[1].size();
        if (!shape.layer3) {
            topo.edgeCount -= topo.edges[3].size();
            topo.edges.erase(3);
        }
        PhyTopo::GetInstance()->Clear();
        hccl::PhyTopoBuilder::ubxClosSplit_ = shape.split;
        auto baseline = BuildGraph("", ranks, topo);
        ASSERT_NE(nullptr, baseline);
        const bool keepLayer1 = shape.layer1 && (!shape.split || shape.meshGroupSize == 3);
        const u32 available = 1U | (keepLayer1 ? 2U : 0U) | (shape.layer3 ? 8U : 0U);
        u32 baselineLayers = 0;
        for (u32 layer : baseline->GetLevels(0)) {
            baselineLayers |= 1U << layer;
        }
        ASSERT_EQ(available, baselineLayers);
        EXPECT_FALSE(baseline->GetBuildLayerSelection().has_value());
        const struct {
            const char* expression;
            u32 expected;
        } cases[]
            = {{"0", 1U},
               {"1", keepLayer1 ? 2U : 0U},
               {"not(0)", available & ~1U},
               {"0,3", shape.layer3 ? 9U : 0U},
               {"not(3)", shape.layer3 ? available & ~8U : 0U}};
        for (bool virtualGroups : {false, true}) {
            for (const auto& item : cases) {
                SCOPED_TRACE(testing::Message() << item.expression << ":" << virtualGroups);
                RankGraphBuilder builder(MakeLayerSelector(item.expression));
                if (item.expected == 0) {
                    EXPECT_THROW(builder.RecoverBuild(ranks, topo, 0, virtualGroups), InvalidParamsException);
                    EXPECT_EQ(nullptr, builder.rankGraph_);
                    continue;
                }
                auto graph = builder.RecoverBuild(ranks, topo, 0, virtualGroups);
                ASSERT_NE(nullptr, graph);
                ASSERT_TRUE(graph->GetBuildLayerSelection().has_value());
                EXPECT_EQ(available, graph->GetBuildLayerSelection()->availableLayers);
                EXPECT_EQ(item.expected, graph->GetBuildLayerSelection()->selectedLayers);
                EXPECT_EQ(baseline->GetInnerRanks(), graph->GetInnerRanks());
                u32 actual = 0;
                for (u32 layer : graph->GetLevels(0)) {
                    actual |= 1U << layer;
                }
                EXPECT_EQ(item.expected, actual);
                for (u32 layer : {0U, 1U, 3U}) {
                    const bool selected = (item.expected & (1U << layer)) != 0;
                    for (u32 peer : {1U, 2U}) {
                        EXPECT_EQ(
                            selected ? baseline->GetPaths(layer, 0, peer).size() : 0U,
                            graph->GetPaths(layer, 0, peer).size());
                    }
                    if (!selected) {
                        EXPECT_TRUE(graph->GetPeer(0)->GetIfacesByLayer(layer).empty());
                    }
                }
                hccl::RankGraphV2 query(graph.get());
                ASSERT_EQ(HCCL_SUCCESS, query.InitNetLayerSelection());
                ASSERT_NO_FATAL_FAILURE(CheckGraphView(query, *graph, graph->GetLevels(0)));
            }
        }
    }
}

TEST_F(RankGraphBuilderTest, Ut_SelectedBuildAndQuery_When_LayerMasks_Expect_OnlySelectedGraph)
{
    const auto rankTable = test::MakeRankTable4pForBuilder();
    const auto topo = test::MakeFourPeerBuilderTopo();
    auto original = BuildGraph("", rankTable, topo);
    ASSERT_NE(nullptr, original);
    ASSERT_EQ(3U, original->GetLevelNum());
    EXPECT_FALSE(original->GetBuildLayerSelection().has_value());
    EXPECT_EQ(original->GetNetInstanceByRankId(0, 0)->GetRankIds(), original->GetInnerRanks());
    hccl::RankGraphV2 originalQuery(original.get());
    ASSERT_EQ(HCCL_SUCCESS, originalQuery.InitNetLayerSelection());
    for (u32 rank = 0; rank < rankTable.rankCount; ++rank) {
        std::string localInstanceId;
        ASSERT_EQ(HCCL_SUCCESS, originalQuery.GetNetInstId(rank, localInstanceId));
        EXPECT_EQ(original->GetNetInstanceByRankId(0, rank)->GetNetInstId(), localInstanceId);
    }

    // 覆盖单层、不含0、非连续层和全选；检查真实图而非查询包装。
    const struct {
        const char* expression;
        u32 mask;
    } cases[] = {{"0", 1U}, {"1", 2U}, {"2", 4U}, {"0,2", 5U}, {"not(0)", 6U}, {"0,1,2", 7U}};
    for (const auto& item : cases) {
        SCOPED_TRACE(item.expression);
        auto graph = BuildGraph(item.expression, rankTable, topo);
        ASSERT_NE(nullptr, graph);
        ASSERT_TRUE(graph->GetBuildLayerSelection().has_value());
        EXPECT_EQ(7U, graph->GetBuildLayerSelection()->availableLayers);
        EXPECT_EQ(original->GetRankSize(), graph->GetRankSize());
        EXPECT_EQ(original->GetInnerRankSize(), graph->GetInnerRankSize());
        EXPECT_EQ(original->GetInnerRanks(), graph->GetInnerRanks());
        hccl::RankGraphV2 query(graph.get());
        ASSERT_EQ(HCCL_SUCCESS, query.InitNetLayerSelection());
        ASSERT_NO_FATAL_FAILURE(CheckGraphView(query, *graph, graph->GetLevels(0)));
        std::set<u32> expectedLayers;
        for (u32 layer = 0; layer < 3; ++layer) {
            const bool selected = (item.mask & (1U << layer)) != 0;
            if (selected) {
                expectedLayers.insert(layer);
                EXPECT_EQ(original->GetPaths(layer, 0, 1).size(), graph->GetPaths(layer, 0, 1).size());
                const auto& address = rankTable.ranks[0].rankLevelInfos[layer].rankAddrs[0].addr;
                uint32_t originalIndex = 0;
                uint32_t selectedIndex = 0;
                ASSERT_EQ(HCCL_SUCCESS, original->GetIpIndex(0, address, originalIndex));
                ASSERT_EQ(HCCL_SUCCESS, graph->GetIpIndex(0, address, selectedIndex));
                EXPECT_EQ(originalIndex, selectedIndex);
            } else {
                EXPECT_EQ(0U, graph->GetLayerRanks(layer));
                EXPECT_TRUE(graph->GetPaths(layer, 0, 1).empty());
                for (u32 rank = 0; rank < rankTable.rankCount; ++rank) {
                    EXPECT_TRUE(graph->GetPeer(rank)->GetIfacesByLayer(layer).empty());
                }
            }
            const auto accessResult = selected ? HCCL_SUCCESS : HCCL_E_NOT_FOUND;
            ASSERT_NO_FATAL_FAILURE(CheckEndpointAccess(selected ? *graph : *original, *graph, layer, 0, accessResult));
            ASSERT_NO_FATAL_FAILURE(CheckEndpointAccess(*original, query, layer, 0, accessResult));
            CommLink* links = nullptr;
            uint32_t linkCount = 0;
            EXPECT_EQ(
                selected || layer == 0 ? HCCL_SUCCESS : HCCL_E_PARA, query.GetLinks(layer, 0, 1, &links, &linkCount));
            if (selected) {
                EXPECT_GT(linkCount, 0U);
            } else {
                EXPECT_EQ(nullptr, links);
                EXPECT_EQ(0U, linkCount);
            }
        }
        if ((item.mask & 1U) == 0) {
            ASSERT_NO_FATAL_FAILURE(CheckLayer0MetadataOnly(query, *graph, {4}, {0, 1, 2, 3}));
        }
        EXPECT_EQ(expectedLayers.size(), graph->GetLevelNum());
        for (u32 rank = 0; rank < rankTable.rankCount; ++rank) {
            EXPECT_TRUE(graph->HasRank(rank));
            EXPECT_EQ(expectedLayers, graph->GetLevels(rank));
            std::string localInstanceId;
            ASSERT_EQ(HCCL_SUCCESS, graph->GetLayer0NetInstanceId(rank, localInstanceId));
            EXPECT_EQ(original->GetNetInstanceByRankId(0, rank)->GetNetInstId(), localInstanceId);
            localInstanceId.clear();
            ASSERT_EQ(HCCL_SUCCESS, query.GetNetInstId(rank, localInstanceId));
            EXPECT_EQ(original->GetNetInstanceByRankId(0, rank)->GetNetInstId(), localInstanceId);
        }
    }
    // 各通信域不修改共享物理图，也不影响先前生成的图。
    auto unchanged = BuildGraph("", rankTable, topo);
    ASSERT_NE(nullptr, unchanged);
    EXPECT_EQ(original->GetLevels(0), unchanged->GetLevels(0));
    EXPECT_EQ(original->GetPaths(0, 0, 1).size(), unchanged->GetPaths(0, 0, 1).size());
}

TEST_F(RankGraphBuilderTest, Ut_SelectedBuild_When_NoLayerZero_Expect_SubGraphPreservesLocalMembership)
{
    auto rankTable = test::MakeRankTable4pForBuilder();
    for (auto& rank : rankTable.ranks) {
        rank.rankLevelInfos[0].netInstId = rank.rankId < 2 ? "server0" : "server1";
        rank.localId = rank.rankId % 2;
    }
    auto graph = BuildGraph("2", rankTable);
    ASSERT_NE(nullptr, graph);
    const std::set<RankId> localRanks = {0, 1};
    EXPECT_EQ(localRanks, graph->GetInnerRanks());
    EXPECT_FALSE(graph->GetPaths(2, 0, 3).empty());
    EXPECT_TRUE(graph->GetPaths(2, 2, 3).empty());

    hccl::RankGraphV2 parentQuery(graph.get());
    ASSERT_EQ(HCCL_SUCCESS, parentQuery.InitNetLayerSelection());
    ASSERT_NO_FATAL_FAILURE(CheckLayer0MetadataOnly(parentQuery, *graph, {2, 2}, {0, 1}));

    std::unique_ptr<RankGraph> child;
    ASSERT_EQ(HCCL_SUCCESS, graph->CreateSubRankGraph({3, 0, 1}, child));
    ASSERT_NE(nullptr, child);
    ASSERT_TRUE(child->GetBuildLayerSelection().has_value());
    EXPECT_EQ(7U, child->GetBuildLayerSelection()->availableLayers);
    EXPECT_EQ(4U, child->GetBuildLayerSelection()->selectedLayers);
    const std::set<RankId> childLocalRanks = {1, 2};
    const std::set<u32> layers = {2};
    EXPECT_EQ(childLocalRanks, child->GetInnerRanks());
    EXPECT_EQ(1, child->GetMyRank());
    EXPECT_EQ(3U, child->GetRankSize());
    EXPECT_EQ(layers, child->GetLevels(1));
    EXPECT_EQ(0U, child->GetLayerRanks(0));
    EXPECT_FALSE(child->GetPaths(2, 1, 0).empty());
    EXPECT_TRUE(child->GetPaths(0, 1, 0).empty());
    hccl::RankGraphV2 childQuery(child.get());
    ASSERT_EQ(HCCL_SUCCESS, childQuery.InitNetLayerSelection());
    ASSERT_NO_FATAL_FAILURE(CheckLayer0MetadataOnly(childQuery, *child, {1, 2}, {1, 2}));
    // 父 rank 3 映射为子 rank 0，设备归属不随 rank 重编号变化。
    for (u32 rank = 0; rank < 3; ++rank) {
        std::string localInstanceId;
        ASSERT_EQ(HCCL_SUCCESS, child->GetLayer0NetInstanceId(rank, localInstanceId));
        EXPECT_EQ(rank == 0 ? "server1" : "server0", localInstanceId);
        localInstanceId.clear();
        ASSERT_EQ(HCCL_SUCCESS, childQuery.GetNetInstId(rank, localInstanceId));
        EXPECT_EQ(rank == 0 ? "server1" : "server0", localInstanceId);
    }
    std::string missing = "stale";
    EXPECT_EQ(HCCL_E_NOT_FOUND, child->GetLayer0NetInstanceId(3, missing));
    EXPECT_TRUE(missing.empty());
    missing = "stale";
    EXPECT_EQ(HCCL_E_NOT_FOUND, childQuery.GetNetInstId(3, missing));
    EXPECT_TRUE(missing.empty());
}

TEST_F(RankGraphBuilderTest, Ut_SelectedQuery_When_GraphExceedsBuiltMask_Expect_RejectWithoutFreezing)
{
    auto actual = BuildGraph("2");
    ASSERT_NE(nullptr, actual);
    hccl::RankGraphV2 query(actual.get());
    MOCKER_CPP(&RankGraph::GetLevels).stubs().will(returnValue(std::set<u32>{0, 2}));
    EXPECT_EQ(HCCL_E_INTERNAL, query.InitNetLayerSelection());
    GlobalMockObject::verify();
    EXPECT_EQ(HCCL_SUCCESS, query.InitNetLayerSelection());
}

TEST_F(RankGraphBuilderTest, Ut_SelectedChild_When_AllCurrentLayers_Expect_CopyOriginalEndpointMap)
{
    const std::vector<u32> rankIds{3, 0, 1};
    for (const auto* expression : {"", "2"}) {
        SCOPED_TRACE(expression);
        auto parent = BuildGraph(expression);
        ASSERT_NE(nullptr, parent);
        EXPECT_TRUE(parent->GetPeer(3)->GetEndpointToIfaceMap().empty());
        u32 available = 0;
        for (u32 layer : parent->GetLevels(0)) {
            available |= 1U << layer;
        }
        for (bool explicitSelection : {false, true}) {
            SCOPED_TRACE(explicitSelection);
            std::unique_ptr<RankGraph> child;
            ASSERT_EQ(
                HCCL_SUCCESS, parent->CreateSubRankGraph(
                                  rankIds, child, explicitSelection ? std::optional<u32>{available} : std::nullopt));
            EXPECT_EQ(parent->GetLevels(0), child->GetLevels(child->GetMyRank()));
            for (u32 rank = 0; rank < rankIds.size(); ++rank) {
                EXPECT_EQ(
                    parent->GetPeer(rankIds[rank])->GetEndpointToIfaceMap(),
                    child->GetPeer(rank)->GetEndpointToIfaceMap());
            }
        }
    }
}

TEST_F(RankGraphBuilderTest, Ut_SelectedChild_When_Narrowed_Expect_OnlySelectedResourcesAndStableIdentity)
{
    auto parent = BuildGraph();
    ASSERT_NE(nullptr, parent);
    std::unique_ptr<RankGraph> child;
    ASSERT_EQ(HCCL_SUCCESS, parent->CreateSubRankGraph({3, 0, 1}, child, 4U));
    ASSERT_NE(nullptr, child);
    EXPECT_EQ((std::set<u32>{2}), child->GetLevels(1));
    EXPECT_EQ(3U, child->GetRankSize());
    EXPECT_EQ((std::set<RankId>{0, 1, 2}), child->GetInnerRanks());
    EXPECT_EQ(7U, child->GetBuildLayerSelection()->availableLayers);
    EXPECT_EQ(4U, child->GetBuildLayerSelection()->selectedLayers);
    EXPECT_FALSE(child->GetPaths(2, 1, 0).empty());
    EXPECT_TRUE(child->GetPaths(0, 1, 0).empty());
    for (u32 layer = 0; layer < 3; ++layer) {
        ASSERT_NO_FATAL_FAILURE(
            CheckEndpointAccess(*parent, *child, layer, 1, layer == 2 ? HCCL_SUCCESS : HCCL_E_NOT_FOUND));
    }
    ASSERT_NO_FATAL_FAILURE(CheckEndpointAccess(*child, *child, 2, 1, HCCL_SUCCESS));
    auto* originalOutput = child.get();
    for (u32 mask : {0U, 8U, 256U}) {
        EXPECT_EQ(HCCL_E_PARA, parent->CreateSubRankGraph({0, 1}, child, mask));
        EXPECT_EQ(originalOutput, child.get());
    }
    // 非本 rank 的非法层也必须在子图过滤/移位前拒绝。
    for (u32 layer : {MAX_NET_LAYER, 32U}) {
        SCOPED_TRACE(layer);
        parent->GetPeer(1)->netLayers_.insert(layer);
        for (bool explicitSelection : {false, true}) {
            SCOPED_TRACE(explicitSelection);
            EXPECT_EQ(
                HCCL_E_PARA,
                parent->CreateSubRankGraph({0, 1}, child, explicitSelection ? std::optional<u32>{4U} : std::nullopt));
            EXPECT_EQ(originalOutput, child.get());
        }
        parent->GetPeer(1)->netLayers_.erase(layer);
    }
    std::unique_ptr<RankGraph> grandchild;
    EXPECT_EQ(HCCL_E_PARA, child->CreateSubRankGraph({1, 2}, grandchild, 1U));
    ASSERT_EQ(HCCL_SUCCESS, child->CreateSubRankGraph({1, 2}, grandchild, 4U));
    ASSERT_NO_FATAL_FAILURE(CheckEndpointAccess(*parent, *grandchild, 2, 0, HCCL_SUCCESS));
    EXPECT_EQ((std::set<u32>{2}), grandchild->GetLevels(0));
    EXPECT_EQ((std::set<u32>{0, 1, 2}), parent->GetLevels(0));
}

TEST_F(RankGraphBuilderTest, Ut_SelectedChild_When_ExcludeExpression_Expect_UseParentAllowedCandidates)
{
    auto parent = BuildGraph("not(0)");
    ASSERT_NE(nullptr, parent);
    parent->SetLevel0PcieFallback(true);
    for (const auto* expression : {"2", "not(1)"}) {
        SCOPED_TRACE(expression);
        std::unique_ptr<RankGraph> child;
        HcclCommConfig config{};
        HcclCommConfigInit(&config);
        ASSERT_EQ(EOK, strcpy_s(config.hcclUseNetLayer, sizeof(config.hcclUseNetLayer), expression));
        u32 selected = 0;
        ASSERT_EQ(HCCL_SUCCESS, hccl::NetLayerSelection::ResolveForSubComm(config.hcclUseNetLayer, 6U, 6U, selected));
        ASSERT_EQ(4U, selected);
        ASSERT_EQ(HCCL_SUCCESS, parent->CreateSubRankGraph({0, 1}, child, selected));
        EXPECT_TRUE(child->IsLevel0PcieFallback());
        EXPECT_EQ(6U, child->GetBuildLayerSelection()->availableLayers);
        hccl::RankGraphV2 query(child.get());
        ASSERT_EQ(HCCL_SUCCESS, query.InitNetLayerSelection());
        ASSERT_NO_FATAL_FAILURE(CheckLayer0MetadataOnly(query, *child, {2}, {0, 1}));
        for (bool explicitSelection : {false, true}) {
            SCOPED_TRACE(explicitSelection);
            std::unique_ptr<RankGraph> inherited;
            ASSERT_EQ(
                HCCL_SUCCESS, child->CreateSubRankGraph(
                                  {0, 1}, inherited, explicitSelection ? std::optional<u32>{4U} : std::nullopt));
            EXPECT_TRUE(inherited->IsLevel0PcieFallback());
            EXPECT_EQ(explicitSelection ? 4U : 6U, inherited->GetBuildLayerSelection()->availableLayers);
            hccl::RankGraphV2 inheritedQuery(inherited.get());
            ASSERT_EQ(HCCL_SUCCESS, inheritedQuery.InitNetLayerSelection());
            ASSERT_NO_FATAL_FAILURE(CheckLayer0MetadataOnly(inheritedQuery, *inherited, {2}, {0, 1}));
        }
    }
}

TEST_F(RankGraphBuilderTest, Ut_SelectedChild_When_SingleRankAndSameEndpoint_Expect_FilterMapKeepDefaultTopo)
{
    RankGraph parent(0);
    auto peer = std::make_shared<NetInstance::Peer>(0, 0, 0, 0);
    ASSERT_EQ(HCCL_SUCCESS, parent.AddPeer(peer));
    EndpointDesc endpoint{};
    ASSERT_EQ(HCCL_SUCCESS, GetCommAddr(endpoint.commAddr, IpAddress("192.168.1.1")));
    endpoint.protocol = LinkProtocolToCommProtocol(LinkProtocol::UB_CTP);
    for (u32 layer : {1U, 2U}) {
        auto instance = std::make_shared<ClosNetInstance>(layer, "plane");
        instance->AddRankId(0);
        ASSERT_EQ(HCCL_SUCCESS, instance->AddNode(peer));
        ASSERT_EQ(HCCL_SUCCESS, peer->AddNetInstance(instance));
        ASSERT_EQ(HCCL_SUCCESS, parent.AddNetInstance(instance));
        instance->UpdateTopoInst(layer + 6, Hccl::TopoType::CLOS, 0);
        const std::set<std::string> ports
            = layer == 1 ? std::set<std::string>{"p0"} : std::set<std::string>{"p0", "p1"};
        auto iface = std::make_shared<NetInstance::ConnInterface>(
            IpAddress("192.168.1.1"), ports, AddrPosition::DEVICE, LinkType::PEER2NET,
            std::set<LinkProtocol>{LinkProtocol::UB_CTP}, Hccl::TopoType::CLOS, layer + 6);
        peer->AddConnInterface(layer, iface);
        peer->SetEndpointToIface(endpoint.commAddr, endpoint.protocol, iface);
    }
    ASSERT_EQ(HCCL_SUCCESS, parent.SetInnerRanks({0}));
    parent.InitFinish();
    std::unique_ptr<RankGraph> child;
    ASSERT_EQ(HCCL_SUCCESS, parent.CreateSubRankGraph({0}, child, 2U));
    std::vector<u32> instances;
    u32 count = 0;
    child->GetTopoInstsByLayer(1, instances, count);
    // 单成员实例保持主线的默认拓扑，不额外复制父实例或接口列表。
    EXPECT_EQ((std::vector<u32>{0}), instances);
    Hccl::TopoType topoType{};
    ASSERT_EQ(HCCL_SUCCESS, child->GetTopoType(1, 0, topoType));
    EXPECT_EQ(Hccl::TopoType::MESH_1D, topoType);
    ASSERT_EQ(HCCL_SUCCESS, child->GetEndpointNum(1, 0, &count));
    EXPECT_EQ(0U, count);
    EndpointDesc actual{};
    count = 1;
    ASSERT_EQ(HCCL_SUCCESS, child->GetEndpointDesc(1, 0, &count, &actual));
    EXPECT_EQ(0U, count);
    EndpointAttrBwCoeff bandwidth = 0;
    ASSERT_EQ(
        HCCL_SUCCESS, child->GetEndpointInfo(0, &endpoint, ENDPOINT_ATTR_BW_COEFF, sizeof(bandwidth), &bandwidth));
    EXPECT_EQ(1U, bandwidth);
    EXPECT_EQ((std::set<u32>{1}), child->GetLevels(0));
    EXPECT_TRUE(child->GetPeer(0)->GetIfacesByLayer(2).empty());
    ASSERT_EQ(
        HCCL_SUCCESS, parent.GetEndpointInfo(0, &endpoint, ENDPOINT_ATTR_BW_COEFF, sizeof(bandwidth), &bandwidth));
    EXPECT_EQ(2U, bandwidth);

    // 0 表示不传选层参数；覆盖默认子域再选层、已选层子域的继承、同层及收窄。
    for (u32 childMask : {0U, 2U, 6U}) {
        for (u32 nestedMask : {0U, 2U}) {
            SCOPED_TRACE(::testing::Message() << "childMask=" << childMask << ", nestedMask=" << nestedMask);
            std::unique_ptr<RankGraph> first;
            ASSERT_EQ(
                HCCL_SUCCESS,
                parent.CreateSubRankGraph({0}, first, childMask == 0 ? std::nullopt : std::optional<u32>{childMask}));
            std::unique_ptr<RankGraph> nested;
            ASSERT_EQ(
                HCCL_SUCCESS, first->CreateSubRankGraph(
                                  {0}, nested, nestedMask == 0 ? std::nullopt : std::optional<u32>{nestedMask}));
            ASSERT_EQ(
                HCCL_SUCCESS,
                nested->GetEndpointInfo(0, &endpoint, ENDPOINT_ATTR_BW_COEFF, sizeof(bandwidth), &bandwidth));
            const bool onlyLayer1 = childMask == 2U || nestedMask == 2U;
            EXPECT_EQ(onlyLayer1 ? 1U : 2U, bandwidth);
            EXPECT_EQ((onlyLayer1 ? std::set<u32>{1} : std::set<u32>{1, 2}), nested->GetLevels(0));
            EXPECT_TRUE(nested->GetPeer(0)->GetInterfacesMap().empty());
            instances.clear();
            nested->GetTopoInstsByLayer(1, instances, count);
            EXPECT_EQ((std::vector<u32>{0}), instances);
        }
    }

    // 损坏的接口列表应在重建映射时报错，显式子图失败不能替换已有结果。
    auto* previous = child.get();
    peer->interfacesMap_[1].push_back(nullptr);
    EXPECT_EQ(HCCL_E_PTR, parent.CreateSubRankGraph({0}, child, 2U));
    EXPECT_EQ(previous, child.get());
    peer->interfacesMap_[1].pop_back();

    // 嵌套子域引用的祖先接口来源也不能包含非法层，失败时保留已有输出。
    std::unique_ptr<RankGraph> first;
    ASSERT_EQ(HCCL_SUCCESS, parent.CreateSubRankGraph({0}, first, 6U));
    for (u32 layer : {MAX_NET_LAYER, 32U}) {
        peer->netLayers_.insert(layer);
        EXPECT_EQ(HCCL_E_PARA, first->CreateSubRankGraph({0}, child, 2U));
        EXPECT_EQ(previous, child.get());
        peer->netLayers_.erase(layer);
    }

    // 在子图节点、实例等构造后使本地成员初始化失败，所有调用方式都不得发布半成品。
    parent.innerRanks_.clear();
    for (bool explicitSelection : {false, true}) {
        SCOPED_TRACE(explicitSelection);
        std::unique_ptr<RankGraph> emptyOutput;
        for (auto* output : {&child, &emptyOutput}) {
            auto* original = output->get();
            SCOPED_TRACE(original != nullptr);
            EXPECT_EQ(
                HCCL_E_PARA,
                parent.CreateSubRankGraph({0}, *output, explicitSelection ? std::optional<u32>{2U} : std::nullopt));
            EXPECT_EQ(original, output->get());
        }
    }
}

TEST_F(RankGraphBuilderTest, Ut_BuilderBridge_When_SubConfigured_Expect_NarrowWithoutEnvironment)
{
    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(nullptr, bridge);
    ASSERT_NE(nullptr, bridge->createSubGraph);
    MOCKER(hccl::NetLayerSelection::LoadForIndependentComm).expects(never());
    MOCKER(hccl::ReadHcclCommConfig).expects(never());
    auto parent = BuildGraph("1,2");
    ASSERT_NE(nullptr, parent);
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    std::unique_ptr<RankGraph> child;
    ASSERT_EQ(HCCL_SUCCESS, bridge->createSubGraph(*parent, {0, 1}, config.hcclUseNetLayer, child));
    EXPECT_EQ((std::set<u32>{1, 2}), child->GetLevels(0));
    ASSERT_EQ(EOK, strcpy_s(config.hcclUseNetLayer, sizeof(config.hcclUseNetLayer), "not(1)"));
    ASSERT_EQ(HCCL_SUCCESS, bridge->createSubGraph(*parent, {0, 1}, config.hcclUseNetLayer, child));
    EXPECT_EQ((std::set<u32>{2}), child->GetLevels(0));
    auto* beforeFailure = child.get();
    ASSERT_EQ(EOK, strcpy_s(config.hcclUseNetLayer, sizeof(config.hcclUseNetLayer), "0"));
    EXPECT_EQ(HCCL_E_PARA, bridge->createSubGraph(*parent, {0, 1}, config.hcclUseNetLayer, child));
    EXPECT_EQ(beforeFailure, child.get());
    EXPECT_EQ(HCCL_E_PTR, bridge->createSubGraph(*parent, {0, 1}, nullptr, child));
    EXPECT_EQ(beforeFailure, child.get());
}

TEST_F(RankGraphBuilderTest, Ut_SelectedBuild_When_InvalidSelection_Expect_RejectBeforeBuild)
{
    // 空掩码、输入中不存在的层、超出 8 层范围均拒绝。
    for (u32 mask : {0U, 8U, 256U}) {
        SCOPED_TRACE(mask);
        RankGraphBuilder builder([mask](u32, u32& effective) {
            effective = mask;
            return HCCL_SUCCESS;
        });
        EXPECT_THROW(
            builder.RecoverBuild(test::MakeRankTable4pForBuilder(), test::MakeFourPeerBuilderTopo(), 0),
            InvalidParamsException);
    }

    // 二进制/内存输入的非法层不能因未选中而被静默忽略。
    for (u32 layer : {MAX_NET_LAYER, 32U}) {
        for (const auto* expression : {"", "0"}) {
            SCOPED_TRACE(layer);
            SCOPED_TRACE(expression);
            auto ranks = test::MakeRankTable4pForBuilder();
            ranks.ranks.back().rankLevelInfos.back().netLayer = layer;
            RankGraphBuilder invalidLayerBuilder(MakeLayerSelector(expression));
            EXPECT_THROW(
                invalidLayerBuilder.RecoverBuild(ranks, test::MakeFourPeerBuilderTopo(), 0), InvalidParamsException);
        }
    }

    // 非法 rank 必须在解析候选层之前被拒绝。
    bool selectionCalled = false;
    RankGraphBuilder builder([&selectionCalled](u32 available, u32& effective) {
        selectionCalled = true;
        effective = available;
        return HCCL_SUCCESS;
    });
    const auto ranks = test::MakeRankTable4pForBuilder();
    EXPECT_THROW(builder.RecoverBuild(ranks, test::MakeFourPeerBuilderTopo(), ranks.rankCount), InvalidParamsException);
    EXPECT_FALSE(selectionCalled);
}

TEST_F(RankGraphBuilderTest, Ut_SelectedBuild_When_ExtendedLayers_Expect_DefaultAndSelectedKeepRealLayerId)
{
    // 与前述0/1/2用例共同覆盖0～7，默认候选层不限定为连续层。
    for (u32 layer = 3; layer < MAX_NET_LAYER; ++layer) {
        SCOPED_TRACE(layer);
        PhyTopo::GetInstance()->Clear();
        auto rankTable = test::MakeRankTable4pForBuilder();
        auto topo = test::MakeFourPeerBuilderTopo();
        auto edges = topo.edges.at(2);
        topo.edges.erase(2);
        for (auto& edge : edges) {
            edge.netLayer = layer;
        }
        topo.edges[layer] = edges;
        for (auto& rank : rankTable.ranks) {
            rank.rankLevelInfos[2].netLayer = layer;
        }
        for (const auto& expression : {std::string{}, std::to_string(layer)}) {
            SCOPED_TRACE(expression);
            auto graph = BuildGraph(expression, rankTable, topo);
            ASSERT_NE(nullptr, graph);
            const auto expected = expression.empty() ? std::set<u32>{0, 1, layer} : std::set<u32>{layer};
            EXPECT_EQ(expected, graph->GetLevels(0));
            EXPECT_EQ(expected.size(), graph->GetLevelNum());
            EXPECT_FALSE(graph->GetPaths(layer, 0, 1).empty());
            EXPECT_EQ(!expression.empty(), graph->GetBuildLayerSelection().has_value());
            for (u32 candidate = 0; candidate < MAX_NET_LAYER; ++candidate) {
                if (expected.count(candidate) == 0) {
                    EXPECT_TRUE(graph->GetPaths(candidate, 0, 1).empty());
                    EXPECT_TRUE(graph->GetPeer(0)->GetIfacesByLayer(candidate).empty());
                }
            }
        }
    }
}

TEST_F(RankGraphBuilderTest, Ut_SelectedBuild_When_DisconnectedGroups_Expect_KeepSelectedLayers)
{
    auto rankTable = test::MakeRankTable4pForBuilder();
    rankTable.ranks[3].rankLevelInfos[2].netInstId = "isolated";
    auto graph = BuildGraph("2", rankTable);
    ASSERT_NE(nullptr, graph);
    EXPECT_EQ(4U, graph->GetRankSize());
    EXPECT_EQ(std::set<u32>{2}, graph->GetLevels(0));
    EXPECT_FALSE(graph->GetPaths(2, 0, 1).empty());
    EXPECT_TRUE(graph->GetPaths(2, 0, 3).empty());
    EXPECT_TRUE(graph->GetPaths(1, 0, 3).empty());
    EXPECT_NO_THROW(BuildGraph("1,2", rankTable));
    EXPECT_NO_THROW(BuildGraph("", rankTable));
}

TEST_F(RankGraphBuilderTest, Ut_SelectedBuild_When_TwoServers_Expect_AllowIntraGroupPaths)
{
    auto rankTable = test::MakeRankTable4pForBuilder();
    for (auto& rank : rankTable.ranks) {
        rank.rankLevelInfos.resize(2);
        rank.rankLevelInfos[0].netInstId = rank.rankId < 2 ? "server0" : "server1";
    }
    // 四卡域可以仅在同组执行 Send/Recv，不应在初始化时要求全域互通。
    auto graph = BuildGraph("not(1)", rankTable);
    ASSERT_NE(nullptr, graph);
    EXPECT_EQ(4U, graph->GetRankSize());
    EXPECT_EQ(std::set<u32>{0}, graph->GetLevels(0));
    EXPECT_FALSE(graph->GetPaths(0, 0, 1).empty());
    EXPECT_TRUE(graph->GetPaths(0, 0, 2).empty());
    EXPECT_TRUE(graph->GetPaths(1, 0, 2).empty());
    EXPECT_NO_THROW(BuildGraph("not(0)", rankTable));
    auto parent = BuildGraph("0,1", rankTable);
    ASSERT_NE(nullptr, parent);
    std::unique_ptr<RankGraph> child;
    ASSERT_EQ(HCCL_SUCCESS, parent->CreateSubRankGraph({0, 2}, child, 1U));
    ASSERT_NE(nullptr, child);
    EXPECT_EQ(2U, child->GetRankSize());
    EXPECT_EQ(std::set<u32>{0}, child->GetLevels(0));
    EXPECT_TRUE(child->GetPaths(0, 0, 1).empty());
    EXPECT_TRUE(child->GetPaths(1, 0, 1).empty());
    ASSERT_EQ(HCCL_SUCCESS, parent->CreateSubRankGraph({0, 1}, child, 1U));
    EXPECT_EQ(2U, child->GetRankSize());
    EXPECT_EQ(std::set<u32>{0}, child->GetLevels(0));
    EXPECT_FALSE(child->GetPaths(0, 0, 1).empty());
    ASSERT_EQ(HCCL_SUCCESS, child->CreateSubRankGraph({0}, parent));
    EXPECT_EQ(1U, parent->GetRankSize());

    rankTable.ranks.resize(2);
    rankTable.rankCount = 2;
    EXPECT_NO_THROW(BuildGraph("not(1)", rankTable));
}

TEST_F(RankGraphBuilderTest, Ut_SelectedBuild_When_OverlappingGroups_Expect_KeepSubdomainSelection)
{
    auto rankTable = test::MakeRankTable4pForBuilder();
    for (auto& rank : rankTable.ranks) {
        rank.rankLevelInfos[1].netInstId = std::to_string(rank.rankId / 2);
        rank.rankLevelInfos[2].netInstId = std::to_string(rank.rankId % 2);
    }
    EXPECT_NO_THROW(BuildGraph("1", rankTable));
    EXPECT_NO_THROW(BuildGraph("2", rankTable));
    auto graph = BuildGraph("1,2", rankTable);
    ASSERT_NE(nullptr, graph);
    std::unique_ptr<RankGraph> child;
    ASSERT_EQ(HCCL_SUCCESS, graph->CreateSubRankGraph({0, 3}, child));
    ASSERT_NE(nullptr, child);
    EXPECT_EQ((std::set<u32>{1, 2}), child->GetLevels(0));
    for (u32 layer : {0U, 1U, 2U}) {
        EXPECT_TRUE(child->GetPaths(layer, 0, 1).empty());
    }
}

TEST_F(RankGraphBuilderTest, Ut_SelectedBuild_When_PcieFallbackExcluded_Expect_NoLayerZeroEdges)
{
    auto rankTable = test::MakeRankTable4pForBuilder();
    rankTable.ranks[0].rankLevelInfos[0].pcieFallback = true;
    MOCKER_CPP(&RankGraphBuilder::BuildPcieFallbackLinks).expects(never());
    MOCKER_CPP(&UpdaterFor64Plus1::UpdateRankGraph).expects(never());
    MOCKER_CPP(&DetourService::InsertDetourLinks).expects(never());
    auto graph = BuildGraph("2", rankTable);
    GlobalMockObject::verify();
    GlobalMockObject::reset();
    ASSERT_NE(nullptr, graph);
    EXPECT_TRUE(graph->IsLevel0PcieFallback());
    EXPECT_EQ(0U, graph->GetLayerRanks(0));
    EXPECT_FALSE(graph->GetPaths(2, 0, 1).empty());
}

TEST_F(RankGraphBuilderTest, Ut_Build_When_ConfigSnapshot_Expect_OnlySelectedLayers)
{
    const struct {
        const char* expression;
        std::set<u32> expected;
    } cases[] = {{"2", {2}}, {"not(0)", {1, 2}}, {"2,0,2", {0, 2}}};
    for (const auto& item : cases) {
        SCOPED_TRACE(item.expression);
        HcclCommConfig config{};
        HcclCommConfigInit(&config);
        ASSERT_EQ(strcpy_s(config.hcclUseNetLayer, sizeof(config.hcclUseNetLayer), item.expression), EOK);
        HcclCommConfig snapshot{};
        ASSERT_EQ(hccl::PrepareIndependentCommConfig(&config, snapshot), HCCL_SUCCESS);
        hccl::NetLayerSelection selection;
        ASSERT_EQ(hccl::ReadNetLayerSelection(snapshot.hcclUseNetLayer, selection), HCCL_SUCCESS);
        const auto rankTable = test::MakeRankTable4pForBuilder();
        RankGraphBuilder builder([selection](u32 available, u32& effective) {
            return selection.Resolve(available, effective);
        });

        ASSERT_NO_FATAL_FAILURE(
            PrepareMockedTopoFile(std::make_shared<hccl::TopoInfo>(test::MakeFourPeerBuilderTopo())));
        auto graph = builder.Build(rankTable, "unused-topo-path", 0);
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        CheckSelectedGraph(graph.get(), item.expected);
    }
}

TEST_F(RankGraphBuilderTest, Ut_BuilderBridge_When_InvalidRegistration_Expect_KeepInstalledBridge)
{
    constexpr auto VERSION = RANK_GRAPH_BUILDER_BRIDGE_VERSION;
    const auto size = static_cast<uint32_t>(sizeof(RankGraphBuilderBridge));
    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(nullptr, bridge);
    const auto installed = *bridge;
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, RegisterRankGraphBuilderBridge(installed, 0, size));
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, RegisterRankGraphBuilderBridge(installed, VERSION + 1, size));
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, RegisterRankGraphBuilderBridge(installed, VERSION, size - 1));
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, RegisterRankGraphBuilderBridge(installed, VERSION, size + 1));
    RankGraphBuilderBridge incomplete[] = {installed, installed, installed, installed, installed};
    incomplete[0].buildFromString = nullptr;
    incomplete[1].buildFromRankTable = nullptr;
    incomplete[2].recoverBuild = nullptr;
    incomplete[3].adoptRankGraph = nullptr;
    incomplete[4].createSubGraph = nullptr;
    for (const auto& candidate : incomplete) {
        EXPECT_EQ(HCCL_E_PTR, RegisterRankGraphBuilderBridge(candidate, VERSION, size));
    }
    EXPECT_EQ(HCCL_E_INTERNAL, RegisterRankGraphBuilderBridge(installed, VERSION, size));
    EXPECT_EQ(bridge, GetRankGraphBuilderBridge());
    EXPECT_EQ(installed.buildFromString, bridge->buildFromString);
    EXPECT_EQ(installed.buildFromRankTable, bridge->buildFromRankTable);
    EXPECT_EQ(installed.recoverBuild, bridge->recoverBuild);
    EXPECT_EQ(installed.adoptRankGraph, bridge->adoptRankGraph);
    EXPECT_EQ(installed.createSubGraph, bridge->createSubGraph);
}

TEST_F(RankGraphBuilderTest, Ut_BuilderBridge_When_ZeroHeaderDefault_Expect_BuildAndInherit)
{
    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(nullptr, bridge);
    const auto rankTable = test::MakeRankTable4pForBuilder();
    const auto topo = std::make_shared<hccl::TopoInfo>(test::MakeFourPeerBuilderTopo());
    ASSERT_NO_FATAL_FAILURE(PrepareMockedTopoFile(topo));
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    ASSERT_EQ(EOK, strcpy_s(config.hcclUseNetLayer, sizeof(config.hcclUseNetLayer), "not(0)"));
    RankGraphBuildResult selectedResult{};
    ASSERT_EQ(HCCL_SUCCESS, bridge->buildFromRankTable(rankTable, "unused", 0, config.hcclUseNetLayer, selectedResult));
    // 内部无配置构造保留零头，默认建图与子域继承不能因统一入口而失败。
    HcclCommConfig legacyDefault{};
    RankGraphBuildResult defaultResult{};
    ASSERT_EQ(
        HCCL_SUCCESS, bridge->buildFromRankTable(rankTable, "unused", 0, legacyDefault.hcclUseNetLayer, defaultResult));
    ASSERT_NO_FATAL_FAILURE(CheckSelectedGraph(defaultResult.rankGraph.get(), {0, 1, 2}));
    ASSERT_EQ(
        HCCL_SUCCESS,
        bridge->recoverBuild(RankTableInfo(rankTable), *topo, 0, false, legacyDefault.hcclUseNetLayer, defaultResult));
    ASSERT_NO_FATAL_FAILURE(CheckSelectedGraph(defaultResult.rankGraph.get(), {0, 1, 2}));
    std::unique_ptr<RankGraph> child;
    ASSERT_EQ(
        HCCL_SUCCESS, bridge->createSubGraph(*selectedResult.rankGraph, {0, 1}, legacyDefault.hcclUseNetLayer, child));
    EXPECT_EQ((std::set<u32>{1, 2}), child->GetLevels(0));
    GlobalMockObject::verify();
    GlobalMockObject::reset();
    CheckSelectedGraph(selectedResult.rankGraph.get(), {1, 2});
    CheckSelectedGraph(defaultResult.rankGraph.get(), {0, 1, 2});
    EXPECT_EQ(topo->edgeCount, selectedResult.topoInfo.edgeCount);
    EXPECT_EQ(topo->edgeCount, defaultResult.topoInfo.edgeCount);
}

TEST_F(RankGraphBuilderTest, Ut_BuilderBridge_When_InvalidConfig_Expect_KeepPreviousResult)
{
    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(nullptr, bridge);
    const auto rankTable = test::MakeRankTable4pForBuilder();
    const auto topo = test::MakeFourPeerBuilderTopo();
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    ASSERT_EQ(EOK, strcpy_s(config.hcclUseNetLayer, sizeof(config.hcclUseNetLayer), "2"));
    RankGraphBuildResult result{};
    ASSERT_EQ(
        HCCL_SUCCESS, bridge->recoverBuild(RankTableInfo(rankTable), topo, 0, false, config.hcclUseNetLayer, result));
    const auto previous = result.rankGraph;
    EXPECT_EQ(HCCL_E_PTR, bridge->buildFromString("", "", 0, nullptr, result));
    EXPECT_EQ(HCCL_E_PTR, bridge->buildFromRankTable(rankTable, "", 0, nullptr, result));
    EXPECT_EQ(HCCL_E_PTR, bridge->recoverBuild(RankTableInfo(rankTable), topo, 0, false, nullptr, result));
    ASSERT_EQ(EOK, strcpy_s(config.hcclUseNetLayer, sizeof(config.hcclUseNetLayer), "bad"));
    EXPECT_EQ(HCCL_E_PARA, bridge->buildFromString("", "", 0, config.hcclUseNetLayer, result));
    EXPECT_EQ(HCCL_E_PARA, bridge->buildFromRankTable(rankTable, "", 0, config.hcclUseNetLayer, result));
    EXPECT_EQ(
        HCCL_E_PARA, bridge->recoverBuild(RankTableInfo(rankTable), topo, 0, false, config.hcclUseNetLayer, result));
    for (const auto* expression : {"3", "not(0,1,2)"}) {
        ASSERT_EQ(EOK, strcpy_s(config.hcclUseNetLayer, sizeof(config.hcclUseNetLayer), expression));
        EXPECT_EQ(
            HCCL_E_PARA,
            bridge->recoverBuild(RankTableInfo(rankTable), topo, 0, false, config.hcclUseNetLayer, result));
    }
    EXPECT_EQ(previous, result.rankGraph);
    EXPECT_EQ(rankTable.rankCount, result.rankTableInfo.rankCount);
    EXPECT_EQ(topo.edgeCount, result.topoInfo.edgeCount);
    EXPECT_EQ(std::set<u32>({2}), result.rankGraph->GetLevels(0));
}

TEST_F(RankGraphBuilderTest, Ut_BuilderBridge_When_AllocationFails_Expect_MemoryError)
{
    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(nullptr, bridge);
    auto parent = BuildGraph("0");
    ASSERT_NE(nullptr, parent);
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    ASSERT_EQ(EOK, strcpy_s(config.hcclUseNetLayer, sizeof(config.hcclUseNetLayer), "0"));
    const auto throwBadAlloc = +[](const char*, hccl::NetLayerSelection&) -> HcclResult {
        throw std::bad_alloc();
    };
    // 直接替换入口，避免 MockCpp 将 bad_alloc 转为框架异常；离开作用域时自动恢复。
    mockcpp::ApiHook readSelectionHook(
        reinterpret_cast<void*>(hccl::ReadNetLayerSelection), reinterpret_cast<void*>(throwBadAlloc));
    RankGraphBuildResult result{};
    EXPECT_EQ(HCCL_E_MEMORY, bridge->buildFromString("", "", 0, config.hcclUseNetLayer, result));
    EXPECT_EQ(
        HCCL_E_MEMORY,
        bridge->buildFromRankTable(test::MakeRankTable4pForBuilder(), "", 0, config.hcclUseNetLayer, result));
    EXPECT_EQ(
        HCCL_E_MEMORY, bridge->recoverBuild(
                           test::MakeRankTable4pForBuilder(), test::MakeFourPeerBuilderTopo(), 0, false,
                           config.hcclUseNetLayer, result));
    EXPECT_EQ(nullptr, result.rankGraph);
    const auto throwSubBadAlloc = +[](const char*, uint32_t, uint32_t, uint32_t&) -> HcclResult {
        throw std::bad_alloc();
    };
    mockcpp::ApiHook subSelectionHook(
        reinterpret_cast<void*>(hccl::NetLayerSelection::ResolveForSubComm), reinterpret_cast<void*>(throwSubBadAlloc));
    std::unique_ptr<RankGraph> child;
    EXPECT_EQ(HCCL_E_MEMORY, bridge->createSubGraph(*parent, {0, 1}, config.hcclUseNetLayer, child));
    EXPECT_EQ(nullptr, child);
}

TEST_F(RankGraphBuilderTest, Ut_BuilderBridge_When_ConfiguredBuildAndRecovery_Expect_RealSelectedAndDefaultGraphs)
{
    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(nullptr, bridge);
    MOCKER_CPP(&hccl::NetLayerSelection::LoadForIndependentComm).expects(never());
    MOCKER(hccl::ReadHcclCommConfig).expects(never());
    const auto ranks = test::MakeRankTable4pForBuilder();
    const auto topo = std::make_shared<hccl::TopoInfo>(test::MakeFourPeerBuilderTopo());
    ASSERT_NO_FATAL_FAILURE(PrepareMockedTopoFile(topo));
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    const struct {
        const char* expression;
        std::set<u32> expected;
    } cases[] = {{"", {0, 1, 2}}, {"0", {0}}, {"2", {2}}, {"not(0)", {1, 2}}, {"0,2", {0, 2}}};
    for (const auto& item : cases) {
        SCOPED_TRACE(item.expression);
        ASSERT_EQ(EOK, strcpy_s(config.hcclUseNetLayer, sizeof(config.hcclUseNetLayer), item.expression));
        RankGraphBuildResult built{}, recovered{};
        ASSERT_EQ(HCCL_SUCCESS, bridge->buildFromRankTable(ranks, "unused", 0, config.hcclUseNetLayer, built));
        ASSERT_NO_FATAL_FAILURE(CheckSelectedGraph(built.rankGraph.get(), item.expected));
        for (bool enableVirtualGroups : {false, true}) {
            SCOPED_TRACE(enableVirtualGroups);
            ASSERT_EQ(
                HCCL_SUCCESS,
                bridge->recoverBuild(
                    RankTableInfo(ranks), *topo, 0, enableVirtualGroups, config.hcclUseNetLayer, recovered));
            CheckSelectedGraph(recovered.rankGraph.get(), item.expected);
            EXPECT_EQ(enableVirtualGroups, recovered.rankGraph->IsVirtualGroupsEnabled());
            EXPECT_EQ(ranks.rankCount, recovered.rankTableInfo.rankCount);
            EXPECT_EQ(topo->edgeCount, recovered.topoInfo.edgeCount);
        }
    }
}

TEST_F(RankGraphBuilderTest, Ut_Build_When_Normal_Expect_Success)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder rankGraphBuilder;
    hccl::TopoInfo topoInfo = test::MakeTwoPeerClosTopo();
    std::unique_ptr<RankGraph> rankGraph = rankGraphBuilder.RecoverBuild(test::MakeRankTable2p(), topoInfo, 0);
    EXPECT_NE(nullptr, rankGraph);
    ASSERT_NE(nullptr, rankGraphBuilder.topoInfo_);
    EXPECT_EQ(topoInfo.Describe(), rankGraphBuilder.topoInfo_->Describe());
    auto path1 = rankGraph->GetPaths(0, 0, 1);
    EXPECT_NE(1, path1.size());
    auto path2 = rankGraph->GetPaths(0, 1, 0);
    EXPECT_NE(1, path2.size());
}

// 优化：Build 成功后 rankGraph 全量 Dump 受日志级别门控，DEBUG 未开启时跳过 Dump，构建结果不受影响
TEST_F(RankGraphBuilderTest, Ut_Build_When_LogLevelNotDebug_Expect_Success)
{
    MOCKER_CPP(&hccl::PhyTopoBuilder::Build).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankGraphBuilder::BuildFromRankTable).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankGraph::BuildVirtualGroups).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankGraphBuilder::BuildPeer2PeerLinks).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankGraphBuilder::UpdateTopoInstForMyRankOnly).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankGraphBuilder::SetEndpointDesc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&DetourService::InsertDetourLinks).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclCheckLogLevel).stubs().will(returnValue(false));
    MOCKER_CPP(&RankGraph::Dump).expects(never()); // DEBUG未开启时全量Dump必须被跳过

    RankGraphBuilder rankGraphBuilder;
    EXPECT_NO_THROW(rankGraphBuilder.Build(test::MakeRankTable2p(), "topo.json", 0));
}

// DEBUG 日志开启时执行全量 Dump，构建结果同样不受影响
TEST_F(RankGraphBuilderTest, Ut_Build_When_LogLevelDebug_Expect_Success)
{
    MOCKER_CPP(&hccl::PhyTopoBuilder::Build).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankGraphBuilder::BuildFromRankTable).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankGraph::BuildVirtualGroups).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankGraphBuilder::BuildPeer2PeerLinks).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankGraphBuilder::UpdateTopoInstForMyRankOnly).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankGraphBuilder::SetEndpointDesc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&DetourService::InsertDetourLinks).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclCheckLogLevel).stubs().will(returnValue(true));
    MOCKER_CPP(&RankGraph::Dump).expects(atLeast(1)); // DEBUG开启时全量Dump必须被执行

    RankGraphBuilder rankGraphBuilder;
    EXPECT_NO_THROW(rankGraphBuilder.Build(test::MakeRankTable2p(), "topo.json", 0));
}

// RecoverBuild（快照恢复主链路）成功后的全量 Dump 同样受日志级别门控，DEBUG 未开启时跳过
TEST_F(RankGraphBuilderTest, Ut_RecoverBuild_When_LogLevelNotDebug_Expect_DumpSkipped)
{
    PhyTopo::GetInstance()->Clear();
    MOCKER(HcclCheckLogLevel).stubs().will(returnValue(false));
    MOCKER_CPP(&RankGraph::Dump).expects(never()); // DEBUG未开启时全量Dump必须被跳过

    RankGraphBuilder rankGraphBuilder;
    hccl::TopoInfo topoInfo = test::MakeTwoPeerClosTopo();
    std::unique_ptr<RankGraph> rankGraph = rankGraphBuilder.RecoverBuild(test::MakeRankTable2p(), topoInfo, 0);
    EXPECT_NE(nullptr, rankGraph);
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
    MOCKER_CPP(&RankGraphBuilder::BuildFromRankTable).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&RankGraph::InitInnerRanks).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&RankGraphBuilder::BuildPeer2PeerLinks).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&DetourService::InsertDetourLinks)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&RankGraphBuilder::UpdateTopoInstForMyRankOnly)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankGraphBuilder::SetEndpointDesc).stubs().will(returnValue(HCCL_SUCCESS));
    // then
    RankGraphBuilder rankGraphBuilder;
    Hccl::RankTableInfo tmpRankTable;
    tmpRankTable.rankCount = 3;
    rankGraphBuilder.rankTable_ = make_unique<Hccl::RankTableInfo>(tmpRankTable);
    rankGraphBuilder.myRank_ = 0;
    EXPECT_EQ(rankGraphBuilder.BuildRankGraph(false), HcclResult::HCCL_SUCCESS);
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
    std::unique_ptr<RankGraph> subRankGraph;
    EXPECT_EQ(HCCL_SUCCESS, rankGraph->CreateSubRankGraph(subRankIds, subRankGraph));
    EXPECT_EQ(1, subRankGraph->GetInnerRankSize());
}

TEST_F(RankGraphBuilderTest, Ut_Build_When_OnePTopoFileWithoutEdge_Expect_Success)
{
    PhyTopo::GetInstance()->Clear();
    auto graph = std::make_shared<hccl::Graph<PhyTopo::Node, PhyTopo::Link>>();
    graph->AddNode(PhyTopo::Peer::GetId(0), std::make_shared<PhyTopo::Peer>(0));
    (void)PhyTopo::GetInstance()->AddTopoGraph(0, graph);
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
    MOCKER_CPP(&hccl::PhyTopoBuilder::Build).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankGraph::InitInnerRanks).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&DetourService::InsertDetourLinks)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&RankGraphBuilder::BuildPeer2PeerLinks).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankGraphBuilder::AddFabricInfo).stubs().will(returnValue(HCCL_SUCCESS));
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
    hccl::TopoInfo topoInfo;
    RankGraphBuilder rankGraphBuilder;
    EXPECT_THROW(rankGraphBuilder.RecoverBuild(rankTableInfo, topoInfo, 0), InvalidParamsException);
}

TEST_F(RankGraphBuilderTest, Ut_RecoverBuild_When_SameTopoInstHasDifferentPlanes_Expect_CreateFabricByPlaneId)
{
    PhyTopo::GetInstance()->Clear();
    const std::string rankTableString = R"({
        "version": "2.0",
        "rank_count": 1,
        "rank_list": [
            {
                "rank_id": 0,
                "device_id": 0,
                "local_id": 0,
                "level_list": [
                    {
                        "net_layer": 0,
                        "net_instance_id": "rank-group-0",
                        "net_type": "TOPO_FILE_DESC",
                        "rank_addr_list": [
                            {
                                "addr_type": "IPV4",
                                "addr": "192.168.0.1",
                                "ports": ["0/1"],
                                "plane_id": "planeA"
                            },
                            {
                                "addr_type": "IPV4",
                                "addr": "192.168.0.2",
                                "ports": ["0/2"],
                                "plane_id": "planeB"
                            }
                        ]
                    }
                ]
            }
        ]
    })";
    const std::string topoString = R"({
        "version": "2.0",
        "peer_count": 1,
        "peer_list": [{"local_id": 0}],
        "edge_count": 2,
        "edge_list": [
            {
                "net_layer": 0,
                "link_type": "PEER2NET",
                "protocols": ["UB_CTP"],
                "topo_type": "CLOS",
                "topo_instance_id": 7,
                "local_a": 0,
                "local_a_ports": ["0/1"],
                "position": "DEVICE"
            },
            {
                "net_layer": 0,
                "link_type": "PEER2NET",
                "protocols": ["UB_CTP"],
                "topo_type": "CLOS",
                "topo_instance_id": 7,
                "local_a": 0,
                "local_a_ports": ["0/2"],
                "position": "DEVICE"
            }
        ]
    })";

    RankTableInfo rankTableInfo;
    hccl::TopoInfo topoInfo;
    ASSERT_EQ(HCCL_SUCCESS, hccl::ParseRankTableInfoJson(rankTableString, rankTableInfo));
    ASSERT_EQ(HCCL_SUCCESS, hccl::ParseTopoInfoJson(topoString, topoInfo));

    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph = rankGraphBuilder.RecoverBuild(rankTableInfo, topoInfo, 0);
    ASSERT_NE(nullptr, rankGraph);
    NetInstance* layer0 = rankGraph->GetNetInstanceByNetInstId(0, "rank-group-0");
    ASSERT_NE(nullptr, layer0);
    ASSERT_EQ(2, layer0->GetFabrics().size());
    std::set<PlaneId> planeIds;
    for (const auto& fabric : layer0->GetFabrics()) {
        ASSERT_NE(nullptr, fabric);
        planeIds.insert(fabric->GetPlaneId());
    }
    EXPECT_EQ((std::set<PlaneId>{"planeA", "planeB"}), planeIds);
}

TEST_F(RankGraphBuilderTest, Ut_RecoverBuild_When_TwoRanksHaveNoPcieRankAddress_Expect_PcieClosPaths)
{
    const std::vector<u32> deviceIds = {2, 5};
    const std::vector<IpAddress> vnicIps = {IpAddress("192.168.100.2"), IpAddress("192.168.100.5")};
    const auto rankTable = test::MakeRankTable(
        {test::MakeRankInfo(
             0, deviceIds[0], 0,
             {test::MakeRankLevel(
                 0, "local", NetType::TOPO_FILE_DESC, {test::MakeAddress("192.168.0.1", {"ub0"}, "7")})}),
         test::MakeRankInfo(
             1, deviceIds[1], 1,
             {test::MakeRankLevel(
                 0, "local", NetType::TOPO_FILE_DESC, {test::MakeAddress("192.168.0.2", {"ub0"}, "7")})})});
    const auto topo = test::MakeTopoInfo(
        {0, 1},
        {test::MakeEdge(0, LinkType::PEER2NET, 0, {"d2h"}, 0, {}, LinkProtocol::PCIE, AddrPosition::DEVICE, 7),
         test::MakeEdge(0, LinkType::PEER2NET, 1, {"d2h"}, 0, {}, LinkProtocol::PCIE, AddrPosition::DEVICE, 7)});

    // 分别从两张物理卡建图，校验查询使用物理 deviceId，且只打桩硬件 VNIC 查询。
    for (u32 myRank = 0; myRank < 2; ++myRank) {
        PhyTopo::GetInstance()->Clear();
        for (u32 rank = 0; rank < 2; ++rank) {
            MOCKER(HrtRaSocketGetVnicIpInfos)
                .expects(once())
                .with(
                    eq(deviceIds[myRank]), eq(Hccl::DeviceIdType::DEVICE_ID_TYPE_PHY_ID), eq(deviceIds[rank]),
                    outBound(vnicIps[rank]));
        }
        RankGraphBuilder builder;
        auto graph = builder.RecoverBuild(rankTable, topo, myRank);
        ASSERT_NE(nullptr, graph);
        auto* layer0 = graph->GetNetInstanceByRankId(0, myRank);
        ASSERT_NE(nullptr, layer0);
        ASSERT_EQ(1U, layer0->GetFabrics().size());
        auto fabric = layer0->GetFabrics().front();
        ASSERT_NE(nullptr, fabric);
        EXPECT_EQ("pcie:7", fabric->GetPlaneId());

        std::vector<u32> topoInstIds;
        u32 topoInstNum = 0;
        layer0->GetTopoInstsByLayer(topoInstIds, topoInstNum);
        EXPECT_EQ(1U, topoInstNum);
        EXPECT_EQ((std::vector<u32>{7}), topoInstIds);
        Hccl::TopoType topoType{Hccl::TopoType::MESH_1D};
        ASSERT_EQ(HCCL_SUCCESS, layer0->GetTopoType(7, topoType));
        EXPECT_EQ(Hccl::TopoType::CLOS, topoType);
        std::vector<u32> topoRanks;
        u32 rankNum = 0;
        ASSERT_EQ(HCCL_SUCCESS, layer0->GetRanksByTopoInst(7, topoRanks, rankNum));
        EXPECT_EQ(2U, rankNum);
        EXPECT_EQ((std::vector<u32>{0, 1}), topoRanks);

        for (u32 rank = 0; rank < 2; ++rank) {
            auto peer = graph->GetPeer(rank);
            ASSERT_NE(nullptr, peer);
            const auto ifaces = peer->GetIfacesByLayer(0);
            ASSERT_EQ(1U, ifaces.size());
            ASSERT_NE(nullptr, ifaces.front());
            EXPECT_EQ(vnicIps[rank], ifaces.front()->GetAddr());
            EXPECT_EQ((std::set<std::string>{"d2h"}), ifaces.front()->GetPorts());
            EXPECT_EQ(Hccl::TopoType::CLOS, ifaces.front()->GetTopoType());
            EXPECT_EQ(7U, ifaces.front()->GetTopoInstId());
            EXPECT_EQ(1U, layer0->GetGraph().GetEdges(peer->GetNodeId(), fabric->GetNodeId()).size());
            EXPECT_EQ(1U, layer0->GetGraph().GetEdges(fabric->GetNodeId(), peer->GetNodeId()).size());
            const auto& level = builder.rankTable_->ranks[rank].rankLevelInfos.front();
            EXPECT_EQ(0U, level.portAddrMap.count("d2h"));
            ASSERT_EQ(1U, level.rankAddrs.size());
            EXPECT_EQ("7", level.rankAddrs.front().planeId);

            const auto paths = graph->GetPaths(0, rank, 1 - rank);
            ASSERT_EQ(1U, paths.size());
            ASSERT_EQ(2U, paths.front().links.size());
            const auto& forward = paths.front().links[0];
            const auto& backward = paths.front().links[1];
            ASSERT_NE(nullptr, forward.GetSourceIface());
            ASSERT_NE(nullptr, backward.GetTargetIface());
            EXPECT_EQ(vnicIps[rank], forward.GetSourceIface()->GetAddr());
            EXPECT_EQ(vnicIps[1 - rank], backward.GetTargetIface()->GetAddr());
            EXPECT_EQ(peer->GetNodeId(), forward.GetSourceNode()->GetNodeId());
            EXPECT_EQ(fabric->GetNodeId(), forward.GetTargetNode()->GetNodeId());
            EXPECT_EQ(fabric->GetNodeId(), backward.GetSourceNode()->GetNodeId());
            EXPECT_EQ(graph->GetPeer(1 - rank)->GetNodeId(), backward.GetTargetNode()->GetNodeId());
            EXPECT_EQ(nullptr, forward.GetTargetIface());
            EXPECT_EQ(nullptr, backward.GetSourceIface());
            for (const auto& link : paths.front().links) {
                EXPECT_EQ(LinkType::PEER2NET, link.GetType());
                EXPECT_EQ((std::set<LinkProtocol>{LinkProtocol::PCIE}), link.GetLinkProtocols());
                EXPECT_EQ(LinkDirection::BOTH, link.GetLinkDirection());
                EXPECT_EQ(2U, link.GetHop());
            }
        }

        u32 endpointNum = 0;
        ASSERT_EQ(HCCL_SUCCESS, graph->GetEndpointNum(0, 7, &endpointNum));
        ASSERT_EQ(1U, endpointNum);
        EndpointDesc endpoint{};
        ASSERT_EQ(HCCL_SUCCESS, graph->GetEndpointDesc(0, 7, &endpointNum, &endpoint));
        ASSERT_EQ(1U, endpointNum);
        EXPECT_EQ(COMM_PROTOCOL_PCIE, endpoint.protocol);
        EXPECT_EQ(ENDPOINT_LOC_TYPE_DEVICE, endpoint.loc.locType);
        EXPECT_EQ(COMM_ADDR_TYPE_IP_V4, endpoint.commAddr.type);
        EXPECT_EQ(vnicIps[myRank].GetBinaryAddress().addr.s_addr, endpoint.commAddr.addr.s_addr);
    }
}

TEST_F(RankGraphBuilderTest, Ut_RecoverBuild_When_RankTableUsesReservedPciePlane_Expect_Reject)
{
    const auto rankTable = test::MakeRankTable(
        {test::MakeRankInfo(
             0, 2, 0,
             {test::MakeRankLevel(
                 0, "local", NetType::TOPO_FILE_DESC, {test::MakeAddress("192.168.0.1", {"ub0"}, "pcie:7")})}),
         test::MakeRankInfo(
             1, 5, 1,
             {test::MakeRankLevel(
                 0, "local", NetType::TOPO_FILE_DESC, {test::MakeAddress("192.168.0.2", {"ub0"}, "7")})})});
    const auto topo = test::MakeTopoInfo(
        {0, 1},
        {test::MakeEdge(0, LinkType::PEER2NET, 0, {"d2h"}, 0, {}, LinkProtocol::PCIE, AddrPosition::DEVICE, 7),
         test::MakeEdge(0, LinkType::PEER2NET, 1, {"d2h"}, 0, {}, LinkProtocol::PCIE, AddrPosition::DEVICE, 7)});
    MOCKER(HrtRaSocketGetVnicIpInfos).expects(never());
    RankGraphBuilder builder;
    EXPECT_THROW(builder.RecoverBuild(rankTable, topo, 0), InvalidParamsException);
    EXPECT_EQ(HCCL_E_PARA, builder.AddTopoDescFabricInfo());
}

TEST_F(RankGraphBuilderTest, Ut_ConstructConnInterface_When_PcieRemoteRank_Expect_TargetVnicAddress)
{
    const IpAddress vnicIp("192.168.100.2");
    MOCKER(HrtRaSocketGetVnicIpInfos).expects(once()).with(eq(0U), mockcpp::any(), eq(1U), outBound(vnicIp));
    auto phyIface = std::make_shared<PhyTopo::ConnInterface>(
        std::set<std::string>{"d2h"}, AddrPosition::DEVICE, LinkType::PEER2NET,
        std::set<LinkProtocol>{LinkProtocol::PCIE});
    auto ifaces = ConstructConnIFromPhyTopoConnIAndPortMap(
        phyIface, {}, Hccl::TopoType::CLOS, 7, ConnIfaceBuildCtx{0U, LinkType::PEER2NET, 1U});
    ASSERT_EQ(1U, ifaces.size());
    EXPECT_EQ(vnicIp, ifaces.front()->GetAddr());
}

// 逻辑超节点ID的实例改写与建图测试
namespace {
constexpr RankId RANK_0 = 0;
constexpr RankId RANK_1 = 1;
constexpr RankId RANK_2 = 2;
constexpr RankId RANK_3 = 3;
constexpr u32 TARGET_LAYER = 1;
constexpr u32 INTER_LAYER = 2;
const std::string TARGET_INST_ID = "sp_0_idx_0";

// 4rank超节点表：L0=server实例(每实例2个rank)、L1=sp_0_idx_0(4个rank)、L2=cluster(4个rank)；
// logicIds非空的rank写入logicSuperPodId字段
RankTableInfo MakeSuperpodTable(const std::map<RankId, std::string>& logicIds = {})
{
    std::vector<hccl::NewRankInfo> ranks;
    for (u32 rankId = 0; rankId < 4; ++rankId) {
        const std::string ip = "192.0.2." + std::to_string(10 + rankId);
        std::vector<RankLevelInfo> levels = {
            test::MakeRankLevel(
                0, rankId < 2 ? "srv_0" : "srv_1", NetType::TOPO_FILE_DESC, {test::MakeAddress(ip, {"0/0"})}),
            test::MakeRankLevel(TARGET_LAYER, TARGET_INST_ID, NetType::CLOS, {test::MakeAddress(ip, {"1/0"})}),
            test::MakeRankLevel(INTER_LAYER, "cluster", NetType::CLOS, {test::MakeAddress(ip, {"2/0"})}),
        };
        hccl::NewRankInfo rank = test::MakeRankInfo(rankId, rankId, rankId, levels);
        auto it = logicIds.find(rankId);
        if (it != logicIds.end()) {
            rank.logicSuperPodId = it->second;
        }
        ranks.push_back(rank);
    }
    return test::MakeRankTable(ranks);
}

// 配套topo：L0框内直连(UB)、L1超节点层(UB)、L2跨超层(RoCE)
hccl::TopoInfo MakeSuperpodTopo(bool withInterRoce = true)
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

// 把TopoInfo写成真实topo.json文件（编排用例供PhyTopoBuilder::Build解析）
std::string ProtocolToStr(LinkProtocol proto)
{
    switch (proto) {
        case LinkProtocol::ROCE:
            return "ROCE";
        case LinkProtocol::UB_RTP:
            return "UB_RTP";
        case LinkProtocol::UBOE:
            return "UBOE";
        default:
            return "UB_CTP";
    }
}

void WriteTopoFile(const std::string& path, const hccl::TopoInfo& topo)
{
    nlohmann::json json;
    json["version"] = "2.0";
    json["peer_count"] = topo.peers.size();
    nlohmann::json peerList = nlohmann::json::array();
    for (const auto& peer : topo.peers) {
        peerList.push_back({{"local_id", peer.localId}});
    }
    json["peer_list"] = peerList;
    nlohmann::json edgeList = nlohmann::json::array();
    for (const auto& layerEdges : topo.edges) {
        for (const auto& edge : layerEdges.second) {
            nlohmann::json edgeJson;
            edgeJson["net_layer"] = edge.netLayer;
            edgeJson["link_type"] = edge.linkType == LinkType::PEER2PEER ? "PEER2PEER" : "PEER2NET";
            edgeJson["local_a"] = edge.localA;
            edgeJson["local_a_ports"] = std::vector<std::string>(edge.localAPorts.begin(), edge.localAPorts.end());
            if (edge.linkType == LinkType::PEER2PEER) {
                edgeJson["local_b"] = edge.localB;
                edgeJson["local_b_ports"] = std::vector<std::string>(edge.localBPorts.begin(), edge.localBPorts.end());
            }
            nlohmann::json protocols = nlohmann::json::array();
            for (const auto& proto : edge.protocols) {
                protocols.push_back(ProtocolToStr(proto));
            }
            edgeJson["protocols"] = protocols;
            edgeList.push_back(edgeJson);
        }
    }
    json["edge_count"] = edgeList.size();
    json["edge_list"] = edgeList;
    std::ofstream file(path);
    ASSERT_TRUE(file.is_open());
    file << json.dump();
    ASSERT_TRUE(file.good());
}
} // namespace

class RewriteSuperPodTableTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    virtual void TearDown() { GlobalMockObject::verify(); }

    // 取rank在目标层的新实例ID
    static std::string GetInstId(const RankTableInfo& table, RankId rankId, u32 layer)
    {
        for (const auto& level : table.ranks[rankId].rankLevelInfos) {
            if (level.netLayer == layer) {
                return level.netInstId;
            }
        }
        return "";
    }
};

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

class ApplyLogicSuperPodIdTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp()
    {
        PhyTopo::GetInstance()->Clear();
        char path[] = "/tmp/hcomm_logic_superpod_XXXXXX";
        const int fd = mkstemp(path);
        ASSERT_GE(fd, 0);
        close(fd);
        topoPath_ = path;
    }
    virtual void TearDown()
    {
        if (!topoPath_.empty()) {
            std::remove(topoPath_.c_str());
        }
        PhyTopo::GetInstance()->Clear();
        GlobalMockObject::verify();
    }
    std::string topoPath_;
};

TEST_F(ApplyLogicSuperPodIdTest, Ut_LogicSuperPod_When_NonRootInfo_Expect_NoRewrite)
{
    auto table = MakeSuperpodTable({{RANK_0, "A"}, {RANK_1, "A"}, {RANK_2, "B"}, {RANK_3, "B"}});
    table.SetSource(RankTableSource::RANKTABLE);
    WriteTopoFile(topoPath_, MakeSuperpodTopo());
    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(bridge, nullptr);
    RankGraphBuildResult built;
    ASSERT_EQ(bridge->buildFromRankTable(table, topoPath_, RANK_0, "", built), HCCL_SUCCESS);
    ASSERT_NE(built.rankGraph, nullptr);
    const auto* instance = built.rankGraph->GetNetInstanceByNetInstId(TARGET_LAYER, TARGET_INST_ID);
    ASSERT_NE(instance, nullptr);
    EXPECT_EQ(instance->GetRankSize(), 4u);
    EXPECT_FALSE(built.rankGraph->GetPaths(TARGET_LAYER, RANK_0, RANK_2).empty());
    EXPECT_EQ(built.rankTableInfo.logicSuperPodSplitLayer, MAX_NET_LAYER);
    ASSERT_EQ(built.rankTableInfo.ranks.size(), table.ranks.size());
    for (size_t i = 0; i < table.ranks.size(); ++i) {
        EXPECT_EQ(table.ranks[i].rankLevelInfos[1].netInstId, TARGET_INST_ID);
        EXPECT_EQ(built.rankTableInfo.ranks[i].rankLevelInfos[1].netInstId, TARGET_INST_ID);
        EXPECT_EQ(table.ranks[i].logicSuperPodId, i < 2 ? "A" : "B");
        EXPECT_EQ(built.rankTableInfo.ranks[i].logicSuperPodId, table.ranks[i].logicSuperPodId);
    }
}

TEST_F(ApplyLogicSuperPodIdTest, Ut_LogicSuperPod_When_RootInfoWithoutLogicId_Expect_OriginalGrouping)
{
    auto table = MakeSuperpodTable();
    table.SetSource(RankTableSource::ROOTINFO);
    WriteTopoFile(topoPath_, MakeSuperpodTopo());
    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(bridge, nullptr);
    RankGraphBuildResult built;
    ASSERT_EQ(bridge->buildFromRankTable(table, topoPath_, RANK_0, "", built), HCCL_SUCCESS);
    ASSERT_NE(built.rankGraph, nullptr);
    const auto* instance = built.rankGraph->GetNetInstanceByNetInstId(TARGET_LAYER, TARGET_INST_ID);
    ASSERT_NE(instance, nullptr);
    EXPECT_EQ(instance->GetRankSize(), 4u);
    EXPECT_FALSE(built.rankGraph->GetPaths(TARGET_LAYER, RANK_0, RANK_2).empty());
    EXPECT_EQ(built.rankTableInfo.logicSuperPodSplitLayer, MAX_NET_LAYER);
    ASSERT_EQ(built.rankTableInfo.ranks.size(), table.ranks.size());
    for (size_t i = 0; i < table.ranks.size(); ++i) {
        EXPECT_EQ(table.ranks[i].rankLevelInfos[1].netInstId, TARGET_INST_ID);
        EXPECT_EQ(built.rankTableInfo.ranks[i].rankLevelInfos[1].netInstId, TARGET_INST_ID);
        EXPECT_TRUE(built.rankTableInfo.ranks[i].logicSuperPodId.empty());
    }
}

TEST_F(ApplyLogicSuperPodIdTest, Ut_LogicSuperPod_When_RecoveredFromSnapshot_Expect_SharedLowerLayerStillBlocked)
{
    auto table = MakeSuperpodTable({{RANK_0, "A"}, {RANK_1, "A"}, {RANK_2, "B"}, {RANK_3, "B"}});
    table.SetSource(RankTableSource::ROOTINFO);
    WriteTopoFile(topoPath_, MakeSuperpodTopo(false));
    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(bridge, nullptr);
    RankGraphBuildResult built;
    ASSERT_EQ(bridge->buildFromRankTable(table, topoPath_, RANK_0, "", built), HCCL_SUCCESS);
    const std::map<RankId, std::string> expectedGroups = {{RANK_0, "A"}, {RANK_1, "A"}, {RANK_2, "B"}, {RANK_3, "B"}};
    EXPECT_EQ(built.rankTableInfo.logicSuperPodGroupIds, expectedGroups);
    ASSERT_TRUE(built.rankGraph->GetPaths(TARGET_LAYER, RANK_0, RANK_2).empty());
    ASSERT_FALSE(built.rankGraph->GetPaths(TARGET_LAYER, RANK_0, RANK_1).empty());

    BinaryStream snapshot;
    ASSERT_EQ(built.rankTableInfo.GetBinStream(true, snapshot), HCCL_SUCCESS);
    const u32 nextField = 0x12345678;
    snapshot << nextField;
    RankTableInfo restoredTable(snapshot);
    EXPECT_EQ(restoredTable.logicSuperPodGroupIds, expectedGroups);
    u32 restoredNextField = 0;
    snapshot >> restoredNextField;
    ASSERT_EQ(restoredNextField, nextField);
    RankGraphBuildResult recovered;
    ASSERT_EQ(
        bridge->recoverBuild(
            std::move(restoredTable), built.topoInfo, RANK_0, built.rankGraph->IsVirtualGroupsEnabled(), "", recovered),
        HCCL_SUCCESS);
    EXPECT_EQ(recovered.rankTableInfo.logicSuperPodGroupIds, expectedGroups);
    EXPECT_TRUE(recovered.rankGraph->GetPaths(TARGET_LAYER, RANK_0, RANK_2).empty());
    EXPECT_FALSE(recovered.rankGraph->GetPaths(TARGET_LAYER, RANK_0, RANK_1).empty());
    EXPECT_EQ(recovered.rankTableInfo.ranks[0].rankLevelInfos[2].netInstId, "cluster_A");
}

TEST_F(ApplyLogicSuperPodIdTest, Ut_LogicSuperPod_When_SubgraphRanksReordered_Expect_SharedLowerLayerStillBlocked)
{
    auto table = MakeSuperpodTable({{RANK_0, "A"}, {RANK_1, "A"}, {RANK_2, "B"}, {RANK_3, "B"}});
    table.SetSource(RankTableSource::ROOTINFO);
    WriteTopoFile(topoPath_, MakeSuperpodTopo(false));
    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(bridge, nullptr);
    RankGraphBuildResult built;
    ASSERT_EQ(bridge->buildFromRankTable(table, topoPath_, RANK_0, "", built), HCCL_SUCCESS);
    ASSERT_TRUE(built.rankGraph->GetPaths(TARGET_LAYER, RANK_0, RANK_2).empty());

    std::unique_ptr<RankGraph> child;
    ASSERT_EQ(built.rankGraph->CreateSubRankGraph({2, 3, 0, 1}, child), HCCL_SUCCESS);
    ASSERT_NE(child, nullptr);
    EXPECT_TRUE(child->GetPaths(TARGET_LAYER, RANK_2, RANK_0).empty());
    EXPECT_TRUE(child->GetPaths(TARGET_LAYER, RANK_0, RANK_2).empty());
    EXPECT_FALSE(child->GetPaths(TARGET_LAYER, RANK_2, RANK_3).empty());
}

TEST_F(ApplyLogicSuperPodIdTest, Ut_LogicSuperPod_When_ServerUnconfigured_Expect_DefaultGroupBoundaryPreserved)
{
    auto table = MakeSuperpodTable({{RANK_0, "A"}, {RANK_1, "A"}});
    table.SetSource(RankTableSource::ROOTINFO);
    WriteTopoFile(topoPath_, MakeSuperpodTopo(false));
    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(bridge, nullptr);

    RankGraphBuildResult baseline;
    // recoverBuild 按右值消费入参，而 table 后续仍需用于 buildFromRankTable，故传入拷贝
    // recoverBuild 按右值消费入参，而 table 后续仍需用于 buildFromRankTable，故传入拷贝
    ASSERT_EQ(
        bridge->recoverBuild(RankTableInfo(table), MakeSuperpodTopo(false), RANK_0, false, "", baseline), HCCL_SUCCESS);
    ASSERT_NE(baseline.rankGraph, nullptr);
    ASSERT_FALSE(baseline.rankGraph->GetPaths(TARGET_LAYER, RANK_0, RANK_2).empty());
    PhyTopo::GetInstance()->Clear();

    RankGraphBuildResult built;
    ASSERT_EQ(bridge->buildFromRankTable(table, topoPath_, RANK_0, "", built), HCCL_SUCCESS);
    ASSERT_NE(built.rankGraph, nullptr);
    const std::map<RankId, std::string> expectedGroups = {{RANK_0, "A"}, {RANK_1, "A"}, {RANK_2, ""}, {RANK_3, ""}};
    EXPECT_EQ(built.rankTableInfo.logicSuperPodGroupIds, expectedGroups);
    EXPECT_EQ(built.rankTableInfo.logicSuperPodSplitLayer, INTER_LAYER);
    EXPECT_TRUE(built.rankGraph->GetPaths(TARGET_LAYER, RANK_0, RANK_2).empty());
    EXPECT_TRUE(built.rankGraph->GetPaths(TARGET_LAYER, RANK_2, RANK_0).empty());
    EXPECT_FALSE(built.rankGraph->GetPaths(TARGET_LAYER, RANK_0, RANK_1).empty());
    EXPECT_EQ(built.rankTableInfo.ranks[RANK_2].rankLevelInfos[2].netInstId, "cluster");

    BinaryStream snapshot;
    ASSERT_EQ(built.rankTableInfo.GetBinStream(true, snapshot), HCCL_SUCCESS);
    RankTableInfo restoredTable(snapshot);
    EXPECT_EQ(restoredTable.logicSuperPodGroupIds, expectedGroups);
    RankGraphBuildResult recovered;
    ASSERT_EQ(
        bridge->recoverBuild(
            std::move(restoredTable), built.topoInfo, RANK_2, built.rankGraph->IsVirtualGroupsEnabled(), "", recovered),
        HCCL_SUCCESS);
    ASSERT_NE(recovered.rankGraph, nullptr);
    EXPECT_TRUE(recovered.rankGraph->GetPaths(TARGET_LAYER, RANK_2, RANK_0).empty());
    EXPECT_FALSE(recovered.rankGraph->GetPaths(TARGET_LAYER, RANK_2, RANK_3).empty());
}

TEST_F(ApplyLogicSuperPodIdTest, Ut_LogicSuperPod_When_ServerMissingSplitLayer_Expect_SharedLowerLayerBlocked)
{
    auto table = MakeSuperpodTable({{RANK_0, "A"}, {RANK_1, "A"}, {RANK_2, "B"}, {RANK_3, "B"}});
    table.SetSource(RankTableSource::ROOTINFO);
    auto topo = MakeSuperpodTopo(false);
    constexpr u32 interLayer = 3;
    for (auto& rank : table.ranks) {
        if (rank.rankId >= RANK_2) {
            rank.rankLevelInfos.pop_back(); // server1不注册L2，L1仍与server0共享。
        }
        rank.rankLevelInfos.push_back(test::MakeRankLevel(
            interLayer, "inter_cluster", NetType::CLOS,
            {test::MakeAddress("198.51.100." + std::to_string(rank.rankId + 1), {"3/0"})}));
        topo.edges[interLayer].push_back(
            test::MakeEdge(interLayer, LinkType::PEER2NET, rank.localId, {"3/0"}, 0, {}, LinkProtocol::ROCE));
    }
    WriteTopoFile(topoPath_, topo);
    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(bridge, nullptr);

    RankGraphBuildResult baseline;
    // recoverBuild 按右值消费入参，而 table 后续仍需用于 buildFromRankTable，故传入拷贝
    ASSERT_EQ(bridge->recoverBuild(RankTableInfo(table), topo, RANK_0, false, "", baseline), HCCL_SUCCESS);
    ASSERT_NE(baseline.rankGraph, nullptr);
    ASSERT_FALSE(baseline.rankGraph->GetPaths(TARGET_LAYER, RANK_0, RANK_2).empty());
    const auto interPaths = baseline.rankGraph->GetPaths(interLayer, RANK_0, RANK_2);
    ASSERT_FALSE(interPaths.empty());
    PhyTopo::GetInstance()->Clear();

    RankGraphBuildResult built;
    ASSERT_EQ(bridge->buildFromRankTable(table, topoPath_, RANK_0, "", built), HCCL_SUCCESS);
    ASSERT_NE(built.rankGraph, nullptr);
    ASSERT_EQ(built.rankTableInfo.logicSuperPodSplitLayer, INTER_LAYER);
    const auto peer = built.rankGraph->GetPeer(RANK_2);
    ASSERT_NE(peer, nullptr);
    ASSERT_EQ(peer->GetNetInstance(INTER_LAYER), nullptr);
    EXPECT_EQ(built.rankTableInfo.logicSuperPodGroupIds.at(RANK_2), "B");
    EXPECT_TRUE(built.rankGraph->GetPaths(TARGET_LAYER, RANK_0, RANK_2).empty());
    EXPECT_TRUE(built.rankGraph->GetPaths(TARGET_LAYER, RANK_2, RANK_0).empty());
    EXPECT_FALSE(built.rankGraph->GetPaths(TARGET_LAYER, RANK_0, RANK_1).empty());
    EXPECT_EQ(built.rankGraph->GetPaths(interLayer, RANK_0, RANK_2).size(), interPaths.size());
}

// 仅在目标层追加逻辑ID，其他层实例ID保持
TEST_F(RewriteSuperPodTableTest, Ut_RewriteTable_When_SplitTwoGroups_Expect_OnlyTargetLayerAppended)
{
    RankTableInfo table = MakeSuperpodTable({{RANK_0, "A"}, {RANK_1, "A"}, {RANK_2, "B"}, {RANK_3, "B"}});
    const RankTableInfo before = table;

    EXPECT_EQ(RewriteSuperPodNetInstId(table, TARGET_LAYER), HCCL_SUCCESS);
    EXPECT_EQ(GetInstId(table, RANK_0, TARGET_LAYER), TARGET_INST_ID + "_A");
    EXPECT_EQ(GetInstId(table, RANK_2, TARGET_LAYER), TARGET_INST_ID + "_B");
    // 其他层原值
    EXPECT_EQ(GetInstId(table, RANK_0, 0), before.ranks[0].rankLevelInfos[0].netInstId);
    EXPECT_EQ(GetInstId(table, RANK_0, INTER_LAYER), "cluster");
}

TEST_F(RewriteSuperPodTableTest, Ut_RewriteTable_When_RanksShareMapping_Expect_LogEachMappingOnce)
{
    auto table = MakeSuperpodTable({{RANK_0, "A"}, {RANK_1, "A"}, {RANK_2, "B"}, {RANK_3, "B"}});
    extern s32 log_level_get_stub();
    extern void log_level_set_stub(s32 logLevel);
    const s32 originalLogLevel = log_level_get_stub();
    log_level_set_stub(DLOG_INFO);
    testing::internal::CaptureStdout();
    const HcclResult ret = RewriteSuperPodNetInstId(table, TARGET_LAYER);
    const std::string output = testing::internal::GetCapturedStdout();
    log_level_set_stub(originalLogLevel);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    for (const std::string group : {"A", "B"}) {
        const std::string message = "netLayer[1] originNetInstId[" + TARGET_INST_ID + "] newNetInstId[" + TARGET_INST_ID
                                    + "_" + group + "] logicSuperPodId[" + group + "]";
        const size_t first = output.find(message);
        ASSERT_NE(first, std::string::npos);
        EXPECT_EQ(output.find(message, first + message.size()), std::string::npos);
    }
    EXPECT_EQ(output.find("rankId["), std::string::npos);
}

// 追加后超过1024字节返回PARA，不截断且不留下部分更新
TEST_F(RewriteSuperPodTableTest, Ut_RewriteTable_When_AppendedIdExceeds1024_Expect_ParaWithoutPartialUpdate)
{
    const std::string longId(MAX_VALUE_NETID - 1, 'a'); // 1023 + "_" + "A" = 1025
    RankTableInfo table = MakeSuperpodTable({{RANK_0, "A"}, {RANK_1, "A"}, {RANK_2, "B"}, {RANK_3, "B"}});
    // 最后一个rank超长，前面的rank已生成新ID，但也不能提前写回。
    for (auto& level : table.ranks.back().rankLevelInfos) {
        if (level.netLayer == TARGET_LAYER) {
            level.netInstId = longId;
        }
    }
    const RankTableInfo before = table;

    EXPECT_EQ(RewriteSuperPodNetInstId(table, TARGET_LAYER), HCCL_E_PARA);
    for (size_t i = 0; i < table.ranks.size(); ++i) {
        for (size_t j = 0; j < table.ranks[i].rankLevelInfos.size(); ++j) {
            EXPECT_EQ(table.ranks[i].rankLevelInfos[j].netInstId, before.ranks[i].rankLevelInfos[j].netInstId);
        }
        EXPECT_EQ(table.ranks[i].logicSuperPodId, before.ranks[i].logicSuperPodId);
    }
    EXPECT_EQ(table.logicSuperPodSplitLayer, before.logicSuperPodSplitLayer);
    EXPECT_EQ(table.logicSuperPodGroupIds, before.logicSuperPodGroupIds);
}

TEST_F(RewriteSuperPodTableTest, Ut_FindSuperPodLayer_When_InterProtocolUbTp_Expect_PreviousRegisteredLayer)
{
    const auto table = MakeSuperpodTable();
    BinaryStream exchanged;
    ASSERT_EQ(table.GetBinStream(true, exchanged), HCCL_SUCCESS);
    RankTableInfo receivedTable(exchanged);
    auto topo = MakeSuperpodTopo(false);
    for (auto& edge : topo.edges[INTER_LAYER]) {
        edge.protocols = {LinkProtocol::UB_TP};
    }
    u32 physicalSuperPodLayer = MAX_NET_LAYER;

    ASSERT_EQ(FindSuperPodLayer(receivedTable, topo, physicalSuperPodLayer), HCCL_SUCCESS);
    EXPECT_EQ(physicalSuperPodLayer, TARGET_LAYER);
}

TEST_F(RewriteSuperPodTableTest, Ut_FindSuperPodLayer_When_OnlyPeerBMatches_Expect_RequirePeer2Peer)
{
    const auto table = MakeSuperpodTable();
    auto topo = test::MakeTopoInfo(
        {2, 99}, {
                     test::MakeEdge(INTER_LAYER, LinkType::PEER2PEER, 99, {"2/0"}, 2, {"2/0"}, LinkProtocol::ROCE),
                 });
    u32 physicalSuperPodLayer = MAX_NET_LAYER;

    ASSERT_EQ(FindSuperPodLayer(table, topo, physicalSuperPodLayer), HCCL_SUCCESS);
    EXPECT_EQ(physicalSuperPodLayer, TARGET_LAYER);

    topo.edges[INTER_LAYER].front().linkType = LinkType::PEER2NET;
    ASSERT_EQ(FindSuperPodLayer(table, topo, physicalSuperPodLayer), HCCL_SUCCESS);
    EXPECT_EQ(physicalSuperPodLayer, INTER_LAYER);
}

TEST_F(RewriteSuperPodTableTest, Ut_FindSuperPodLayer_When_PortMatchesDifferentLayer_Expect_NoInterLayer)
{
    const auto table = MakeSuperpodTable();
    const auto topo = test::MakeTopoInfo(
        {0}, {
                 test::MakeEdge(TARGET_LAYER, LinkType::PEER2NET, 0, {"2/0"}, 0, {}, LinkProtocol::ROCE),
             });
    u32 physicalSuperPodLayer = MAX_NET_LAYER;

    ASSERT_EQ(FindSuperPodLayer(table, topo, physicalSuperPodLayer), HCCL_SUCCESS);
    EXPECT_EQ(physicalSuperPodLayer, INTER_LAYER);
}

TEST_F(RewriteSuperPodTableTest, Ut_FindSuperPodLayer_When_AsymmetricLayerGap_Expect_RegisteredPredecessor)
{
    auto table = MakeSuperpodTable();
    for (auto& rank : table.ranks) {
        if (rank.rankId == RANK_0) {
            rank.rankLevelInfos.erase(rank.rankLevelInfos.begin() + 1);
        } else {
            rank.rankLevelInfos[1].rankAddrs.clear();
            rank.rankLevelInfos[1].portAddrMap.clear();
        }
        rank.rankLevelInfos.back().netLayer = 3;
    }
    const auto topo = test::MakeTopoInfo(
        {0}, {
                 test::MakeEdge(3, LinkType::PEER2NET, 0, {"2/0"}, 0, {}, LinkProtocol::ROCE),
             });
    u32 physicalSuperPodLayer = MAX_NET_LAYER;

    ASSERT_EQ(FindSuperPodLayer(table, topo, physicalSuperPodLayer), HCCL_SUCCESS);
    EXPECT_EQ(physicalSuperPodLayer, TARGET_LAYER);
}

TEST_F(RewriteSuperPodTableTest, Ut_FindSuperPodLayer_When_LowestRegisteredLayerIsInter_Expect_NotSupport)
{
    auto table = MakeSuperpodTable();
    for (auto& rank : table.ranks) {
        rank.rankLevelInfos.erase(rank.rankLevelInfos.begin(), rank.rankLevelInfos.begin() + 2);
    }
    const auto topo = MakeSuperpodTopo();
    u32 physicalSuperPodLayer = MAX_NET_LAYER;

    EXPECT_EQ(FindSuperPodLayer(table, topo, physicalSuperPodLayer), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(physicalSuperPodLayer, MAX_NET_LAYER);
}

// A/B分组后按层成员、大小、链路边界全部由一次建图给出
TEST_F(TableRewriteBuildTest, Ut_Build_When_SplitTwoGroups_Expect_GroupedQueriesAndLinkBoundary)
{
    RankTableInfo table = MakeSuperpodTable({{RANK_0, "A"}, {RANK_1, "A"}, {RANK_2, "B"}, {RANK_3, "B"}});
    hccl::TopoInfo topo = MakeSuperpodTopo();

    u32 physicalSuperPodLayer = 99;
    ASSERT_EQ(FindSuperPodLayer(table, topo, physicalSuperPodLayer), HCCL_SUCCESS);
    ASSERT_EQ(physicalSuperPodLayer, TARGET_LAYER);
    ASSERT_EQ(RewriteSuperPodNetInstId(table, physicalSuperPodLayer), HCCL_SUCCESS);

    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(bridge, nullptr);
    RankGraphBuildResult result;
    ASSERT_EQ(bridge->recoverBuild(std::move(table), topo, RANK_0, false, "", result), HCCL_SUCCESS);
    auto graph = result.rankGraph;
    ASSERT_NE(graph, nullptr);

    // 按层查询逻辑组成员，rank0在目标层的实例成员恰为本组{0,1}
    std::vector<u32> rankList;
    u32 rankNum = 0;
    EXPECT_EQ(graph->GetLocalInstRanks(TARGET_LAYER, rankList, rankNum), HCCL_SUCCESS);
    EXPECT_EQ(rankNum, 2u);
    EXPECT_EQ(rankList, std::vector<u32>({0, 1}));

    // 按拓扑实例查询逻辑组成员，topoInst 0 的登记随建图落入本组实例
    std::vector<u32> topoInstRanks;
    u32 topoInstRankNum = 0;
    EXPECT_EQ(graph->GetRanksByTopoInst(TARGET_LAYER, 0, topoInstRanks, topoInstRankNum), HCCL_SUCCESS);
    EXPECT_EQ(topoInstRankNum, 2u);
    EXPECT_EQ(topoInstRanks, std::vector<u32>({0, 1}));

    // 查询逻辑实例大小列表，目标层实例大小为{2,2}
    std::vector<u32> instSizes;
    u32 listSize = 0;
    EXPECT_EQ(graph->GetNetInstanceList(TARGET_LAYER, instSizes, listSize), HCCL_SUCCESS);
    ASSERT_EQ(listSize, 2u);
    std::sort(instSizes.begin(), instSizes.end());
    EXPECT_EQ(instSizes[0], 2u);
    EXPECT_EQ(instSizes[1], 2u);

    // 实例索引：原ID消失，新ID存在
    EXPECT_EQ(graph->GetNetInstanceByNetInstId(TARGET_LAYER, TARGET_INST_ID), nullptr);
    EXPECT_NE(graph->GetNetInstanceByNetInstId(TARGET_LAYER, TARGET_INST_ID + "_A"), nullptr);
    EXPECT_NE(graph->GetNetInstanceByNetInstId(TARGET_LAYER, TARGET_INST_ID + "_B"), nullptr);

    // 同组保留原有超内链路
    EXPECT_FALSE(graph->GetPaths(TARGET_LAYER, RANK_0, RANK_1).empty());
    // 跨组无可用候选返回0条（目标层属于不同实例）
    EXPECT_TRUE(graph->GetPaths(TARGET_LAYER, RANK_0, RANK_2).empty());
    // 跨组返回可用跨超链路（跨超层共享保留）
    EXPECT_FALSE(graph->GetPaths(INTER_LAYER, RANK_0, RANK_2).empty());
}

// 未配置变量时，实例改写不改变原表，建图保持原有分组
TEST_F(TableRewriteBuildTest, Ut_Build_When_NoLogicIdConfigured_Expect_OriginalGrouping)
{
    RankTableInfo table = MakeSuperpodTable(); // 无任何logicSuperPodId
    hccl::TopoInfo topo = MakeSuperpodTopo();

    u32 physicalSuperPodLayer = 99;
    ASSERT_EQ(FindSuperPodLayer(table, topo, physicalSuperPodLayer), HCCL_SUCCESS);
    ASSERT_EQ(physicalSuperPodLayer, TARGET_LAYER);
    EXPECT_EQ(RewriteSuperPodNetInstId(table, physicalSuperPodLayer), HCCL_SUCCESS); // 空配置no-op

    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(bridge, nullptr);
    RankGraphBuildResult result;
    ASSERT_EQ(bridge->recoverBuild(std::move(table), topo, RANK_0, false, "", result), HCCL_SUCCESS);
    auto graph = result.rankGraph;
    ASSERT_NE(graph, nullptr);
    // 原ID单实例4个rank，无任何后缀实例
    const NetInstance* orig = graph->GetNetInstanceByNetInstId(TARGET_LAYER, TARGET_INST_ID);
    ASSERT_NE(orig, nullptr);
    EXPECT_EQ(orig->GetRankSize(), 4u);
    EXPECT_EQ(graph->GetNetInstanceByNetInstId(TARGET_LAYER, TARGET_INST_ID + "_A"), nullptr);
    // 未拆分组：任意两rank目标层超内链路均保留
    EXPECT_FALSE(graph->GetPaths(TARGET_LAYER, RANK_0, RANK_2).empty());
}

// 双物理超节点分别追加A/B，跨物理对只走跨超层
TEST_F(TableRewriteBuildTest, Ut_Build_When_TwoPhysicalSuperpods_Expect_BothRewritten)
{
    std::vector<hccl::NewRankInfo> ranks;
    for (u32 rankId = 0; rankId < 4; ++rankId) {
        const std::string ip = "192.0.2." + std::to_string(10 + rankId);
        std::vector<RankLevelInfo> levels = {
            test::MakeRankLevel(
                0, rankId < 2 ? "srv_0" : "srv_1", NetType::TOPO_FILE_DESC, {test::MakeAddress(ip, {"0/0"})}),
            test::MakeRankLevel(
                TARGET_LAYER, rankId < 2 ? "sp_0_idx_0" : "sp_1_idx_0", NetType::CLOS,
                {test::MakeAddress(ip, {"1/0"})}),
            test::MakeRankLevel(INTER_LAYER, "cluster", NetType::CLOS, {test::MakeAddress(ip, {"2/0"})}),
        };
        hccl::NewRankInfo rank = test::MakeRankInfo(rankId, rankId, rankId, levels);
        rank.logicSuperPodId = (rankId < 2) ? "A" : "B";
        ranks.push_back(rank);
    }
    RankTableInfo table = test::MakeRankTable(ranks);
    hccl::TopoInfo topo = MakeSuperpodTopo();

    u32 physicalSuperPodLayer = 99;
    ASSERT_EQ(FindSuperPodLayer(table, topo, physicalSuperPodLayer), HCCL_SUCCESS);
    ASSERT_EQ(physicalSuperPodLayer, TARGET_LAYER);
    ASSERT_EQ(RewriteSuperPodNetInstId(table, physicalSuperPodLayer), HCCL_SUCCESS);

    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(bridge, nullptr);
    RankGraphBuildResult result;
    ASSERT_EQ(bridge->recoverBuild(std::move(table), topo, RANK_0, false, "", result), HCCL_SUCCESS);
    auto graph = result.rankGraph;
    ASSERT_NE(graph, nullptr);
    EXPECT_NE(graph->GetNetInstanceByNetInstId(TARGET_LAYER, "sp_0_idx_0_A"), nullptr);
    EXPECT_NE(graph->GetNetInstanceByNetInstId(TARGET_LAYER, "sp_1_idx_0_B"), nullptr);
    EXPECT_EQ(graph->GetNetInstanceByNetInstId(TARGET_LAYER, "sp_0_idx_0"), nullptr);
    EXPECT_EQ(graph->GetNetInstanceByNetInstId(TARGET_LAYER, "sp_1_idx_0"), nullptr);

    std::vector<u32> instSizes;
    u32 listSize = 0;
    EXPECT_EQ(graph->GetNetInstanceList(TARGET_LAYER, instSizes, listSize), HCCL_SUCCESS);
    ASSERT_EQ(listSize, 2u);
    std::sort(instSizes.begin(), instSizes.end());
    EXPECT_EQ(instSizes[0], 2u);
    EXPECT_EQ(instSizes[1], 2u);

    EXPECT_FALSE(graph->GetPaths(TARGET_LAYER, RANK_0, RANK_1).empty()); // 同物理同组超内保留
    EXPECT_TRUE(graph->GetPaths(TARGET_LAYER, RANK_0, RANK_2).empty());  // 跨物理目标层0条
    EXPECT_FALSE(graph->GetPaths(INTER_LAYER, RANK_0, RANK_2).empty());  // 跨超层可用
}

// 已配置但topo文件不可读时，返回错误且不改写原表
TEST_F(ApplyLogicSuperPodIdTest, Ut_Apply_When_TopoFileMissing_Expect_ErrorPropagated)
{
    RankTableInfo table = MakeSuperpodTable({{RANK_0, "A"}, {RANK_1, "A"}});
    const RankTableInfo before = table;

    ASSERT_EQ(std::remove(topoPath_.c_str()), 0);
    EXPECT_EQ(ApplyLogicSuperPodId(table, topoPath_), HCCL_E_PARA);
    EXPECT_EQ(table.ranks[0].rankLevelInfos[1].netInstId, before.ranks[0].rankLevelInfos[1].netInstId);
    EXPECT_FALSE(table.ranks[0].logicSuperPodId.empty()); // 未消费
    EXPECT_EQ(table.logicSuperPodSplitLayer, before.logicSuperPodSplitLayer);
}

TEST_F(RankGraphBuilderTest, Ut_BuildAndRecover_When_SourceChanges_Expect_GroupingPolicyAndPhysicalTopology)
{
    std::vector<hccl::NewRankInfo> ranks;
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
    MOCKER_CPP(&hccl::PhyTopoBuilder::Build).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::PhyTopoBuilder::GetTopoInfo).stubs().will(returnValue(std::make_shared<hccl::TopoInfo>(topo)));
    for (u32 mode = 0; mode < 4; ++mode) {
        const bool enabled = (mode % 2 != 0);
        PhyTopo::GetInstance()->Clear();
        RankGraphBuilder builder;
        std::unique_ptr<RankGraph> graph;
        if (mode < 2) {
            graph = builder.RecoverBuild(table, topo, 0, enabled);
        } else {
            ASSERT_EQ(hccl::PhyTopoBuilder::GetInstance().RecoverBuild(topo), HCCL_SUCCESS);
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

namespace {
// vnic 缓存键与实现保持一致：(localDeviceId << 32) | targetDeviceId
constexpr std::uint64_t MakeVnicCacheKey(u32 localDeviceId, u32 targetDeviceId)
{
    return (static_cast<std::uint64_t>(localDeviceId) << 32) | targetDeviceId;
}

constexpr u32 STUB_VNIC_IP_NUM = 8;
const char* const STUB_VNIC_IPS[STUB_VNIC_IP_NUM]
    = {"192.168.200.0", "192.168.200.1", "192.168.200.2", "192.168.200.3",
       "192.168.200.4", "192.168.200.5", "192.168.200.6", "192.168.200.7"};

// 可成功返回的 vnic 查询桩，按 deviceId 返回不同的 vnic IP
void StubVnicIpQueryByDevice(u32 phyId, Hccl::DeviceIdType deviceIdType, u32 deviceId, IpAddress& vnicIp)
{
    (void)phyId;
    (void)deviceIdType;
    vnicIp = IpAddress(STUB_VNIC_IPS[deviceId % STUB_VNIC_IP_NUM]);
}

// deviceId==1 时抛出 NetworkApiException；mockcpp 在 mock 边界会把非 mockcpp 异常统一改写为
// mockcpp::Exception（what() 保留原始描述），故用例按 mockcpp::Exception 断言，
// 生产侧 catch(HcclException) 的"查询失败→跳过该对建链"降级路径由 HLT/ST 覆盖
void StubVnicIpQueryFailOnDevice1(u32 phyId, Hccl::DeviceIdType deviceIdType, u32 deviceId, IpAddress& vnicIp)
{
    (void)phyId;
    (void)deviceIdType;
    if (deviceId == 1U) {
        throw NetworkApiException("ut vnic query fail");
    }
    vnicIp = IpAddress(STUB_VNIC_IPS[deviceId % STUB_VNIC_IP_NUM]);
}
} // namespace

TEST_F(RankGraphBuilderTest, Ut_ConstructConnInterface_When_PcieMultiPort_Expect_VnicQueryOnce)
{
    const IpAddress vnicIp("192.168.100.2");
    MOCKER(HrtRaSocketGetVnicIpInfos).expects(once()).with(eq(0U), mockcpp::any(), eq(1U), outBound(vnicIp));
    auto phyIface = std::make_shared<PhyTopo::ConnInterface>(
        std::set<std::string>{"d2h", "d2h2"}, AddrPosition::DEVICE, LinkType::PEER2NET,
        std::set<LinkProtocol>{LinkProtocol::PCIE});
    auto ifaces = ConstructConnIFromPhyTopoConnIAndPortMap(
        phyIface, {}, Hccl::TopoType::CLOS, 7, ConnIfaceBuildCtx{0U, LinkType::PEER2NET, 1U});
    ASSERT_EQ(1U, ifaces.size());
    EXPECT_EQ(vnicIp, ifaces.front()->GetAddr());
    EXPECT_EQ((std::set<std::string>{"d2h"}), ifaces.front()->GetPorts());
}

TEST_F(RankGraphBuilderTest, Ut_Build_When_PcieFallbackMultiRank_Expect_VnicQueryOncePerDevice)
{
    PhyTopo::GetInstance()->Clear();
    const std::vector<IpAddress> vnicIps = {
        IpAddress("192.168.200.0"), IpAddress("192.168.200.1"), IpAddress("192.168.200.2"), IpAddress("192.168.200.3")};
    for (u32 deviceId = 0; deviceId < vnicIps.size(); ++deviceId) {
        MOCKER(HrtRaSocketGetVnicIpInfos)
            .expects(once())
            .with(eq(0U), eq(Hccl::DeviceIdType::DEVICE_ID_TYPE_PHY_ID), eq(deviceId), outBound(vnicIps[deviceId]));
    }
    RankTableInfo rankTable = test::MakeRankTable4pForBuilder();
    // 模拟rank_info_detect_client插入兜底level0时写入的pcie_fallback显式标记（本rank即rank0）
    rankTable.ranks[0].rankLevelInfos[0].pcieFallback = true;
    RankGraphBuilder builder;
    auto graph = builder.RecoverBuild(rankTable, test::MakeFourPeerBuilderTopo(), 0);
    ASSERT_NE(nullptr, graph);
    EXPECT_EQ(true, graph->IsLevel0PcieFallback());
    // 兜底链路统一注册为单个MESH_1D topoInst，覆盖inner域全部rank
    auto* layer0 = graph->GetNetInstanceByRankId(0, 0);
    ASSERT_NE(nullptr, layer0);
    std::vector<u32> topoRanks;
    u32 rankNum = 0;
    ASSERT_EQ(HCCL_SUCCESS, layer0->GetRanksByTopoInst(0, topoRanks, rankNum));
    EXPECT_EQ(4U, rankNum);
    EXPECT_EQ((std::vector<u32>{0, 1, 2, 3}), topoRanks);
}

TEST_F(RankGraphBuilderTest, Ut_GetVnicIp_When_CacheHit_Expect_NoBottomQuery)
{
    MOCKER(HrtRaSocketGetVnicIpInfos).expects(never());
    RankGraphBuilder builder;
    builder.vnicIpCache_[MakeVnicCacheKey(0U, 1U)] = IpAddress("192.168.100.2");
    EXPECT_EQ(IpAddress("192.168.100.2"), builder.GetVnicIp(0, 1));
    EXPECT_EQ(1U, builder.vnicIpCache_.size());
}

TEST_F(RankGraphBuilderTest, Ut_GetVnicIp_When_QueryFail_Expect_NotCachedAndRetryable)
{
    MOCKER(HrtRaSocketGetVnicIpInfos).stubs().will(throws(NetworkApiException("ut")));
    RankGraphBuilder builder;
    EXPECT_THROW(builder.GetVnicIp(0, 1), mockcpp::Exception);
    EXPECT_TRUE(builder.vnicIpCache_.empty());
    EXPECT_THROW(builder.GetVnicIp(0, 1), mockcpp::Exception);
    EXPECT_TRUE(builder.vnicIpCache_.empty());
}

TEST_F(RankGraphBuilderTest, Ut_ConstructConnInterface_When_CachePassed_Expect_QueryOnceAndCached)
{
    const IpAddress vnicIp("192.168.100.2");
    MOCKER(HrtRaSocketGetVnicIpInfos).expects(once()).with(eq(0U), mockcpp::any(), eq(1U), outBound(vnicIp));
    auto phyIface = std::make_shared<PhyTopo::ConnInterface>(
        std::set<std::string>{"d2h", "d2h2"}, AddrPosition::DEVICE, LinkType::PEER2NET,
        std::set<LinkProtocol>{LinkProtocol::PCIE});
    std::map<std::uint64_t, IpAddress> vnicIpCache;
    auto ifaces = ConstructConnIFromPhyTopoConnIAndPortMap(
        phyIface, {}, Hccl::TopoType::CLOS, 7, ConnIfaceBuildCtx{0U, LinkType::PEER2NET, 1U, &vnicIpCache});
    ASSERT_EQ(1U, ifaces.size());
    ASSERT_EQ(1U, vnicIpCache.size());
    EXPECT_EQ(vnicIp, vnicIpCache.at(MakeVnicCacheKey(0U, 1U)));

    // 第二次调用应命中缓存，不再查询底层
    auto ifacesAgain = ConstructConnIFromPhyTopoConnIAndPortMap(
        phyIface, {}, Hccl::TopoType::CLOS, 7, ConnIfaceBuildCtx{0U, LinkType::PEER2NET, 1U, &vnicIpCache});
    ASSERT_EQ(1U, ifacesAgain.size());
    EXPECT_EQ(vnicIp, ifacesAgain.front()->GetAddr());
}

TEST_F(RankGraphBuilderTest, Ut_RecoverBuild_When_StaleVnicCache_Expect_CacheCleared)
{
    MOCKER(HrtRaSocketGetVnicIpInfos).stubs().will(invoke(StubVnicIpQueryByDevice));
    RankTableInfo rankTable = test::MakeRankTable4pForBuilder();
    rankTable.ranks[0].rankLevelInfos[0].pcieFallback = true;
    RankGraphBuilder builder;
    builder.vnicIpCache_[MakeVnicCacheKey(0U, 999U)] = IpAddress("10.0.0.1"); // 模拟 builder 复用前的残留缓存

    auto graph = builder.RecoverBuild(rankTable, test::MakeFourPeerBuilderTopo(), 0);
    ASSERT_NE(nullptr, graph);
    EXPECT_EQ(builder.vnicIpCache_.count(MakeVnicCacheKey(0U, 999U)), 0U); // 构建入口清缓存，残留项不应保留
}

TEST_F(RankGraphBuilderTest, Ut_BuildPcieFallback_When_VnicQueryFail_Expect_ExceptionAndNotCached)
{
    MOCKER(HrtRaSocketGetVnicIpInfos).stubs().will(invoke(StubVnicIpQueryFailOnDevice1));
    RankTableInfo rankTable = test::MakeRankTable4pForBuilder();
    rankTable.ranks[0].rankLevelInfos[0].pcieFallback = true;
    RankGraphBuilder builder;

    // mock 边界把 NetworkApiException 改写为 mockcpp::Exception，异常会穿透 RecoverBuild
    EXPECT_THROW(builder.RecoverBuild(rankTable, test::MakeFourPeerBuilderTopo(), 0), mockcpp::Exception);
    EXPECT_EQ(builder.vnicIpCache_.count(MakeVnicCacheKey(0U, 1U)), 0U); // 失败的 deviceId 不写入缓存
}

TEST_F(RankGraphBuilderTest, Ut_ConstructConnInterface_When_PortMissing_Expect_SkipNoOutOfRange)
{
    const std::set<std::string> phyPorts = {"portMissing"};
    const std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    auto phyIface
        = std::make_shared<PhyTopo::ConnInterface>(phyPorts, AddrPosition::DEVICE, LinkType::PEER2PEER, protocols);
    const std::map<std::string, std::vector<IpAddress>> portAddrMap = {{"otherPort", {IpAddress("192.168.1.1")}}};

    std::vector<std::shared_ptr<NetInstance::ConnInterface>> ifaces;
    EXPECT_NO_THROW(
        ifaces = ConstructConnIFromPhyTopoConnIAndPortMap(
            phyIface, portAddrMap, Hccl::TopoType::MESH_1D, 0, ConnIfaceBuildCtx{0U}));
    EXPECT_TRUE(ifaces.empty());
}

TEST_F(RankGraphBuilderTest, Ut_GetPortAddrMapLayer0_When_Called_Expect_ConstRef)
{
    auto peer = std::make_shared<NetInstance::Peer>(0, 0, 0, 0);
    static_assert(
        std::is_same<
            decltype(peer->GetPortAddrMapLayer0()), const std::map<std::string, std::vector<IpAddress>>&>::value,
        "GetPortAddrMapLayer0 should return const reference");
    const auto& portAddrMap = peer->GetPortAddrMapLayer0();
    EXPECT_TRUE(portAddrMap.empty());
}

TEST_F(RankGraphBuilderTest, Ut_GetPortAddrMapLayer0_When_PopulatedAndMissingPort_Expect_RefNoCopy)
{
    auto peer = std::make_shared<NetInstance::Peer>(0, 0, 0, 0);
    std::map<std::string, std::vector<IpAddress>> portAddrMap
        = {{"0/0", {IpAddress("192.168.1.1")}}, {"0/1", {IpAddress("192.168.1.2")}}};
    peer->SetPortPortAddrMapLayer0(portAddrMap);

    const auto& first = peer->GetPortAddrMapLayer0();
    const auto& second = peer->GetPortAddrMapLayer0();
    EXPECT_EQ(&first, &second); // 返回同一成员对象，无深拷贝
    EXPECT_EQ(2U, first.size());
    EXPECT_EQ(IpAddress("192.168.1.1"), first.at("0/0").at(0));

    auto it = first.find("nonexistent");
    EXPECT_EQ(first.end(), it); // 缺失端口判空，不越界
}

TEST_F(RankGraphBuilderTest, Ut_ConstructConnInterface_When_EmptyProtocols_Expect_SkipNoLink)
{
    const IpAddress addr("192.168.1.1");
    const std::map<std::string, std::vector<IpAddress>> portAddrMap = {{"port0", {addr}}};
    auto phyIface = std::make_shared<PhyTopo::ConnInterface>(
        std::set<std::string>{"port0"}, AddrPosition::DEVICE, LinkType::PEER2PEER, std::set<LinkProtocol>{});

    std::vector<std::shared_ptr<NetInstance::ConnInterface>> ifaces;
    EXPECT_NO_THROW(
        ifaces = ConstructConnIFromPhyTopoConnIAndPortMap(
            phyIface, portAddrMap, Hccl::TopoType::MESH_1D, 0, ConnIfaceBuildCtx{0U}));
    // 空协议集不建链：不产生空协议 ConnInterface，下游 processLinks/GetFirstLinkProtocol 不会解引用空 begin()
    EXPECT_TRUE(ifaces.empty());
}

TEST_F(RankGraphBuilderTest, Ut_ConstructConnInterface_When_EmptyProtocolsAndPorts_Expect_EmptyNoCrash)
{
    auto phyIface = std::make_shared<PhyTopo::ConnInterface>(
        std::set<std::string>{}, AddrPosition::DEVICE, LinkType::PEER2PEER, std::set<LinkProtocol>{});

    std::vector<std::shared_ptr<NetInstance::ConnInterface>> ifaces;
    EXPECT_NO_THROW(
        ifaces
        = ConstructConnIFromPhyTopoConnIAndPortMap(phyIface, {}, Hccl::TopoType::MESH_1D, 0, ConnIfaceBuildCtx{0U}));
    EXPECT_TRUE(ifaces.empty());
}

TEST_F(RankGraphBuilderTest, Ut_RecoverBuild_When_RvalueRankTable_Expect_BuildSuccess)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder builder;
    RankTableInfo rankTable = test::MakeRankTable2p();
    ASSERT_FALSE(rankTable.ranks.empty());

    auto graph = builder.RecoverBuild(std::move(rankTable), test::MakeTwoPeerClosTopo(), 0);
    ASSERT_NE(nullptr, graph);
    EXPECT_EQ(2U, graph->GetRankSize());
}

TEST_F(RankGraphBuilderTest, Ut_RecoverBuild_When_LvalueRankTable_Expect_SourceIntact)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder builder;
    RankTableInfo rankTable = test::MakeRankTable2p();
    const size_t rankNum = rankTable.ranks.size();
    ASSERT_NE(0U, rankNum);

    auto graph = builder.RecoverBuild(rankTable, test::MakeTwoPeerClosTopo(), 0);
    ASSERT_NE(nullptr, graph);
    EXPECT_EQ(rankNum, rankTable.ranks.size()); // 左值拷贝，源不变
}

TEST_F(RankGraphBuilderTest, Ut_BuildIpIndexMap_When_DuplicateAddrs_Expect_IpIndexContiguous)
{
    auto level0 = test::MakeRankLevel(
        0, "inner", NetType::TOPO_FILE_DESC,
        {test::MakeAddress("192.168.1.1", {"0/0"}), test::MakeAddress("192.168.1.1", {"0/1"}),
         test::MakeAddress("192.168.1.2", {"0/0"})});
    auto level1 = test::MakeRankLevel(1, "clos", NetType::CLOS, {test::MakeAddress("192.168.1.3", {"0/0"})});
    RankTableInfo rankTable = test::MakeRankTable({test::MakeRankInfo(0, 0, 0, {level0, level1})});

    RankGraphBuilder builder;
    builder.rankTable_ = std::make_unique<RankTableInfo>(rankTable);
    builder.rankGraph_ = std::make_unique<RankGraph>(0);
    builder.BuildIpIndexMap();

    std::set<uint32_t> indices;
    for (const char* ip : {"192.168.1.1", "192.168.1.2", "192.168.1.3"}) {
        uint32_t idx = 0;
        ASSERT_EQ(HCCL_SUCCESS, builder.rankGraph_->GetIpIndex(0, IpAddress(ip), idx));
        indices.insert(idx);
    }
    EXPECT_EQ((std::set<uint32_t>{0U, 1U, 2U}), indices);
}

TEST_F(RankGraphBuilderTest, Ut_BuildIpIndexMap_When_RankHasNoAddr_Expect_NotFound)
{
    RankTableInfo rankTable = test::MakeRankTable({test::MakeRankInfo(0, 0, 0, {})});
    RankGraphBuilder builder;
    builder.rankTable_ = std::make_unique<RankTableInfo>(rankTable);
    builder.rankGraph_ = std::make_unique<RankGraph>(0);
    builder.BuildIpIndexMap();

    uint32_t idx = 5U;
    EXPECT_EQ(HCCL_E_NOT_FOUND, builder.rankGraph_->GetIpIndex(0, IpAddress("192.168.1.1"), idx));
}

TEST_F(RankGraphBuilderTest, Ut_BuildIpIndexMap_When_CompareWithSetImpl_Expect_SameResult)
{
    auto defaultAddr = test::MakeAddress("192.168.1.9", {"0/2"});
    defaultAddr.addr = Hccl::IpAddress(); // 默认地址：实现与参考实现均应跳过
    auto level0 = test::MakeRankLevel(
        0, "inner", NetType::TOPO_FILE_DESC,
        {test::MakeAddress("192.168.1.3", {"0/0"}), test::MakeAddress("192.168.1.1", {"0/0"}),
         test::MakeAddress("192.168.1.1", {"0/1"}), defaultAddr});
    auto level1 = test::MakeRankLevel(1, "clos", NetType::CLOS, {test::MakeAddress("192.168.1.2", {"0/0"})});
    auto rank0 = test::MakeRankInfo(0, 0, 0, {level0, level1});
    auto rank1 = test::MakeRankInfo(
        1, 1, 1,
        {test::MakeRankLevel(0, "inner", NetType::TOPO_FILE_DESC, {test::MakeAddress("192.168.2.1", {"0/0"})})});
    RankTableInfo rankTable = test::MakeRankTable({rank0, rank1});

    // 参考实现：std::set 去重排序
    std::map<std::pair<RankId, IpAddress>, uint32_t> expected;
    for (const auto& rankInfo : rankTable.ranks) {
        std::set<IpAddress> sortedAddrs;
        for (const auto& levelInfo : rankInfo.rankLevelInfos) {
            for (const auto& addrInfo : levelInfo.rankAddrs) {
                if (!(addrInfo.addr == IpAddress())) {
                    sortedAddrs.insert(addrInfo.addr);
                }
            }
        }
        uint32_t i = 0;
        for (const auto& addr : sortedAddrs) {
            expected[{rankInfo.rankId, addr}] = i++;
        }
    }

    RankGraphBuilder builder;
    builder.rankTable_ = std::make_unique<RankTableInfo>(rankTable);
    builder.rankGraph_ = std::make_unique<RankGraph>(0);
    builder.BuildIpIndexMap();

    // 默认地址不应进入映射表
    uint32_t defaultIdx = 42U;
    EXPECT_EQ(HCCL_E_NOT_FOUND, builder.rankGraph_->GetIpIndex(0, Hccl::IpAddress(), defaultIdx));
    EXPECT_EQ(42U, defaultIdx);

    for (const auto& kv : expected) {
        uint32_t idx = 0;
        ASSERT_EQ(HCCL_SUCCESS, builder.rankGraph_->GetIpIndex(kv.first.first, kv.first.second, idx));
        EXPECT_EQ(kv.second, idx);
    }
    uint32_t idx = 0;
    EXPECT_EQ(HCCL_E_NOT_FOUND, builder.rankGraph_->GetIpIndex(0, IpAddress("192.168.9.9"), idx));
}

namespace {

std::shared_ptr<NetInstance::Peer> FindPeerByLocalId(NetInstance* netInst, LocalId localId)
{
    for (const auto& it : netInst->GetPeers()) {
        if (it.second->GetLocalId() == localId) {
            return it.second;
        }
    }
    return nullptr;
}

std::shared_ptr<NetInstance::Peer>
FindPeerLinkedToReplaced(const UpdaterFor64Plus1& updater, NetInstance* netInst, LocalId replacedLocalId)
{
    auto phyGraph = PhyTopo::GetInstance()->GetTopoGraph(0);
    if (phyGraph == nullptr) {
        return nullptr;
    }
    for (const auto& it : netInst->GetPeers()) {
        const auto localId = it.second->GetLocalId();
        if (localId == BACKUP_LOCAL_ID) {
            continue;
        }
        if (!updater.IsSameX(localId, replacedLocalId) && !updater.IsSameY(localId, replacedLocalId)) {
            continue;
        }
        if (!phyGraph->HasEdge(PhyTopo::Peer::GetId(localId), PhyTopo::Peer::GetId(replacedLocalId))) {
            continue;
        }
        return it.second;
    }
    return nullptr;
}

// MakeRankTable2x2(true) 的 backup localId=512 与 topo 中的 64 不一致，此处归一为 64 保证两者对齐
RankTableInfo MakeRankTable2x2WithTopoBackupLocalId()
{
    RankTableInfo rankTable = test::MakeRankTable2x2(true);
    for (auto& rank : rankTable.ranks) {
        if (rank.localId == BACKUP_LOCAL_ID) {
            rank.localId = 64U;
            rank.deviceId = 64U;
        }
    }
    return rankTable;
}

} // namespace

TEST_F(RankGraphBuilderTest, Ut_AddPeer2BackupLinks_When_BackupPortMissing_Expect_E_Para)
{
    RankTableInfo rankTable = MakeRankTable2x2WithTopoBackupLocalId();
    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(rankTable, test::MakeTwoByTwoPlusBackupTopo(), 0);
    ASSERT_NE(rankGraph, nullptr);
    NetInstance* netInst = rankGraph->GetNetInstanceByNetInstId(0, "az0-rack0");
    ASSERT_NE(netInst, nullptr);
    ASSERT_NE(PhyTopo::GetInstance()->GetTopoGraph(0), nullptr);

    UpdaterFor64Plus1 updater;
    constexpr LocalId replacedLocalId = 1U;
    auto backupPeer = FindPeerByLocalId(netInst, 64U);
    auto peer = FindPeerLinkedToReplaced(updater, netInst, replacedLocalId);
    ASSERT_NE(backupPeer, nullptr);
    ASSERT_NE(peer, nullptr);

    backupPeer->SetPortPortAddrMapLayer0({});
    EXPECT_EQ(updater.AddPeer2BackupLinks(peer, backupPeer, replacedLocalId, netInst, &rankTable), HCCL_E_PARA);
    // 旧实现 map[port][0] 缺键会静默插入；新实现必须保持表为空
    EXPECT_TRUE(backupPeer->GetPortAddrMapLayer0().empty());
}

TEST_F(RankGraphBuilderTest, Ut_AddPeer2BackupLinks_When_PeerPortMissing_Expect_E_Para)
{
    RankTableInfo rankTable = MakeRankTable2x2WithTopoBackupLocalId();
    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(rankTable, test::MakeTwoByTwoPlusBackupTopo(), 0);
    ASSERT_NE(rankGraph, nullptr);
    NetInstance* netInst = rankGraph->GetNetInstanceByNetInstId(0, "az0-rack0");
    ASSERT_NE(netInst, nullptr);
    ASSERT_NE(PhyTopo::GetInstance()->GetTopoGraph(0), nullptr);

    UpdaterFor64Plus1 updater;
    constexpr LocalId replacedLocalId = 1U;
    auto backupPeer = FindPeerByLocalId(netInst, 64U);
    auto peer = FindPeerLinkedToReplaced(updater, netInst, replacedLocalId);
    ASSERT_NE(backupPeer, nullptr);
    ASSERT_NE(peer, nullptr);

    peer->SetPortPortAddrMapLayer0({});
    EXPECT_EQ(updater.AddPeer2BackupLinks(peer, backupPeer, replacedLocalId, netInst, &rankTable), HCCL_E_PARA);
    EXPECT_TRUE(peer->GetPortAddrMapLayer0().empty());
}
