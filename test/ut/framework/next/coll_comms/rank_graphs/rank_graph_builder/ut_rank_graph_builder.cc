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
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>

#define private public
#define protected public

#include "detour_service.h"
#include "internal_exception.h"
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

using namespace Hccl;

class RankGraphBuilderTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankGraphBuilderTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RankGraphBuilderTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in RankGraphBuilderTest SetUP" << std::endl;
        setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
        GlobalMockObject::reset();
        PhyTopo::GetInstance()->Clear();
    }

    virtual void TearDown()
    {
        PhyTopo::GetInstance()->Clear();
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

    auto ifaces
        = ConstructConnIFromPhyTopoConnIAndPortMap(phyIface, portAddrMap, TopoType::CLOS, 0, 0, LinkType::PEER2NET);

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

    auto ifaces = ConstructConnIFromPhyTopoConnIAndPortMap(phyIface, portAddrMap, TopoType::MESH_1D, 0, 0);

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
        AddressInfo address;
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
    auto graph = std::make_shared<Graph<PhyTopo::Node, PhyTopo::Link>>();
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
                    eq(deviceIds[myRank]), eq(DeviceIdType::DEVICE_ID_TYPE_PHY_ID), eq(deviceIds[rank]),
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
        TopoType topoType{TopoType::MESH_1D};
        ASSERT_EQ(HCCL_SUCCESS, layer0->GetTopoType(7, topoType));
        EXPECT_EQ(TopoType::CLOS, topoType);
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
            EXPECT_EQ(TopoType::CLOS, ifaces.front()->GetTopoType());
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
    auto ifaces = ConstructConnIFromPhyTopoConnIAndPortMap(phyIface, {}, TopoType::CLOS, 7, 0, LinkType::PEER2NET, 1);
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
    ASSERT_EQ(bridge->buildFromRankTable(table, topoPath_, RANK_0, built), HCCL_SUCCESS);
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
    ASSERT_EQ(bridge->buildFromRankTable(table, topoPath_, RANK_0, built), HCCL_SUCCESS);
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
    ASSERT_EQ(bridge->buildFromRankTable(table, topoPath_, RANK_0, built), HCCL_SUCCESS);
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
            restoredTable, built.topoInfo, RANK_0, built.rankGraph->IsVirtualGroupsEnabled(), recovered),
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
    ASSERT_EQ(bridge->buildFromRankTable(table, topoPath_, RANK_0, built), HCCL_SUCCESS);
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
    ASSERT_EQ(bridge->recoverBuild(table, MakeSuperpodTopo(false), RANK_0, false, baseline), HCCL_SUCCESS);
    ASSERT_NE(baseline.rankGraph, nullptr);
    ASSERT_FALSE(baseline.rankGraph->GetPaths(TARGET_LAYER, RANK_0, RANK_2).empty());
    PhyTopo::GetInstance()->Clear();

    RankGraphBuildResult built;
    ASSERT_EQ(bridge->buildFromRankTable(table, topoPath_, RANK_0, built), HCCL_SUCCESS);
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
            restoredTable, built.topoInfo, RANK_2, built.rankGraph->IsVirtualGroupsEnabled(), recovered),
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
    ASSERT_EQ(bridge->recoverBuild(table, topo, RANK_0, false, baseline), HCCL_SUCCESS);
    ASSERT_NE(baseline.rankGraph, nullptr);
    ASSERT_FALSE(baseline.rankGraph->GetPaths(TARGET_LAYER, RANK_0, RANK_2).empty());
    const auto interPaths = baseline.rankGraph->GetPaths(interLayer, RANK_0, RANK_2);
    ASSERT_FALSE(interPaths.empty());
    PhyTopo::GetInstance()->Clear();

    RankGraphBuildResult built;
    ASSERT_EQ(bridge->buildFromRankTable(table, topoPath_, RANK_0, built), HCCL_SUCCESS);
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
    ASSERT_EQ(bridge->recoverBuild(table, topo, RANK_0, false, result), HCCL_SUCCESS);
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
    ASSERT_EQ(bridge->recoverBuild(table, topo, RANK_0, false, result), HCCL_SUCCESS);
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
    ASSERT_EQ(bridge->recoverBuild(table, topo, RANK_0, false, result), HCCL_SUCCESS);
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
