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
#include "orion_adapter_hccp.h"

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
    std::vector<NewRankInfo> ranks;
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
    EXPECT_EQ(rankGraphBuilder.BuildRankGraph(), HcclResult::HCCL_SUCCESS);
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
    MOCKER_CPP(&PhyTopoBuilder::Build).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
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
    TopoInfo topoInfo;
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

    JsonParser parser;
    RankTableInfo rankTableInfo;
    TopoInfo topoInfo;
    ASSERT_EQ(HCCL_SUCCESS, parser.ParseString(rankTableString, rankTableInfo));
    ASSERT_EQ(HCCL_SUCCESS, parser.ParseString(topoString, topoInfo));

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
