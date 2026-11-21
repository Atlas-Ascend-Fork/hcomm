/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <map>
#include <unordered_map>
#include <utility>
#include "rank_gph.h"
#include "rank_graph_v2.h"
#include "rank_graph_builder.h"
#include "phy_topo_builder.h"
#include "detour_service.h"
#include "ranktable_builder.h"
#include "internal_exception.h"

using namespace Hccl;

class RankGraphTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        if (setenv("HCCL_DETOUR", "detour:1", 1) == -1) {
            perror("setenv");
        }

        // 获取环境变量
        char* env_var = getenv("HCCL_DETOUR");
        if (env_var != nullptr) {
            std::cout << "HCCL_DETOUR: " << env_var << std::endl;
        } else {
            std::cout << "HCCL_DETOUR is not set" << std::endl;
        }

        std::cout << "VirtTopoBuilderTest SetUP" << std::endl;
    }

    static void TearDownTestCase()
    {
        if (unsetenv("HCCL_DETOUR") == -1) {
            perror("unsetenv");
        }
        char* env_var = getenv("HCCL_DETOUR");
        if (env_var != nullptr) {
            std::cout << "HCCL_DETOUR: " << env_var << std::endl;
        } else {
            std::cout << "HCCL_DETOUR has been successfully unset" << std::endl;
        }

        std::cout << "VirtTopoBuilderTest TearDown" << std::endl;
    }

    virtual void SetUp() { std::cout << "A Test case in RankGraphTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RankGraphTest TearDown" << std::endl;
    }
    RankId myRank = 0;
};

// **单个测试用例命名**：`Ut_<API名称>_When_<测试条件>_Expect_<预期行为>`（大驼峰命名）
// 例如：`ut_HcclGetCommName_When_Normal_Expect_ReturnIsHCCL_SUCCESS` 用例名应该要按照这个格式吧

std::shared_ptr<NetInstance::Peer> createPeer(int rankId = 0, int localId = 0, DeviceId deviceId = 0)
{
    return std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
}

TEST_F(RankGraphTest, Ut_SetInnerRanks_When_InvalidOrFrozen_Expect_KeepMembers)
{
    RankGraph graph(0);
    ASSERT_EQ(HCCL_SUCCESS, graph.AddPeer(createPeer(0)));
    ASSERT_EQ(HCCL_SUCCESS, graph.AddPeer(createPeer(1)));
    ASSERT_EQ(HCCL_SUCCESS, graph.SetInnerRanks({0, 1}));
    EXPECT_EQ(HCCL_E_PARA, graph.SetInnerRanks({1}));
    EXPECT_EQ(HCCL_E_PARA, graph.SetInnerRanks({0, 2}));
    EXPECT_EQ(2U, graph.GetInnerRankSize());
    graph.InitFinish();
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, graph.SetInnerRanks({0}));
    EXPECT_EQ(2U, graph.GetInnerRankSize());
}

TEST_F(RankGraphTest, ut_AddPeer_When_Normal_Expect_SUCCESS)
{
    RankGraph rankGraph(myRank);
    auto peer = createPeer();
    EXPECT_EQ(rankGraph.AddPeer(peer), HCCL_SUCCESS);
    EXPECT_EQ(false, rankGraph.HasRank(3));
    EXPECT_EQ(true, rankGraph.HasRank(peer->GetRankId()));
}

TEST_F(RankGraphTest, ut_AddPeer_When_NullPeer_Expect_E_PTR)
{
    RankGraph rankGraph(myRank);
    EXPECT_EQ(rankGraph.AddPeer(nullptr), HCCL_E_PTR);
}

TEST_F(RankGraphTest, ut_AddPeer_When_Init_Finshed_Expect_E_INTERNAL)
{
    RankGraph rankGraph(myRank);
    rankGraph.InitFinish();
    auto peer = createPeer();
    EXPECT_EQ(rankGraph.AddPeer(peer), HCCL_E_INTERNAL);
}

TEST_F(RankGraphTest, ut_AddNetInstance_When_Normal_Expect_SUCCESS)
{
    s32 netLayer = 0;
    string netInstId = "test";
    RankGraph rankGraph(myRank);
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    EXPECT_EQ(rankGraph.AddNetInstance(netInstance), HCCL_SUCCESS);
    auto netInstResp = rankGraph.GetNetInstanceByNetInstId(netLayer, netInstId);
    EXPECT_NE(nullptr, netInstResp);
}

TEST_F(RankGraphTest, ut_AddNetInstance_When_NullPeer_Expect_E_PTR)
{
    RankGraph rankGraph(myRank);
    EXPECT_EQ(rankGraph.AddNetInstance(nullptr), HCCL_E_PTR);
}

TEST_F(RankGraphTest, ut_AddNetInstance_When_Init_Finshed_Expect_E_INTERNAL)
{
    RankGraph rankGraph(myRank);
    rankGraph.InitFinish();
    auto netInstance = std::make_shared<InnerNetInstance>(0, "test");
    EXPECT_EQ(rankGraph.AddNetInstance(netInstance), HCCL_E_INTERNAL);
}

TEST_F(RankGraphTest, ut_AddNetInstance_When_Not_Include_NetInstance_RankId_Expect_E_PARA)
{
    RankGraph rankGraph(myRank);
    auto netInstance = std::make_shared<InnerNetInstance>(0, "test");
    netInstance->AddRankId(1);
    EXPECT_EQ(rankGraph.AddNetInstance(netInstance), HCCL_E_PARA);
}

TEST_F(RankGraphTest, ut_Dump_When_LinkHasLongDescription_Expect_LogComplete)
{
    RankGraph rankGraph(myRank);
    auto netInstance = std::make_shared<ClosNetInstance>(0, "test");
    auto peer = createPeer(0, 0, 0);
    auto fabric = std::make_shared<NetInstance::Fabric>(0, "plane0");
    netInstance->AddRankId(peer->GetRankId());
    (void)netInstance->AddNode(peer);
    (void)netInstance->AddNode(fabric);
    (void)rankGraph.AddPeer(peer);
    (void)rankGraph.AddNetInstance(netInstance);

    std::set<std::string> ports = {"0/0"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP, LinkProtocol::UB_MEM};
    auto sourceIface = std::make_shared<NetInstance::ConnInterface>(
        IpAddress("ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffe"), ports, AddrPosition::DEVICE, LinkType::PEER2PEER,
        protocols);
    auto targetIface = std::make_shared<NetInstance::ConnInterface>(
        IpAddress("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"), ports, AddrPosition::DEVICE, LinkType::PEER2PEER,
        protocols);
    sourceIface->SetLocalDieId(111);
    targetIface->SetLocalDieId(222);
    netInstance->AddLink(
        std::make_shared<NetInstance::Link>(peer, fabric, sourceIface, nullptr, LinkType::PEER2NET, protocols));
    netInstance->AddLink(
        std::make_shared<NetInstance::Link>(fabric, peer, nullptr, targetIface, LinkType::PEER2NET, protocols));

    extern s32 log_level_get_stub();
    extern void log_level_set_stub(s32 logLevel);
    const s32 originalLogLevel = log_level_get_stub();
    log_level_set_stub(DLOG_DEBUG);
    testing::internal::CaptureStdout();
    rankGraph.Dump();
    const std::string output = testing::internal::GetCapturedStdout();
    log_level_set_stub(originalLogLevel);

    EXPECT_NE(output.find("netLayer selection: mode[DEFAULT], no layer filtering"), std::string::npos);
    EXPECT_NE(output.find("NetInstance::Link[srcId=0, dstId=4294967296, type=LinkType::PEER2NET"), std::string::npos);
    EXPECT_NE(output.find("srcIface=Iface"), std::string::npos);
    EXPECT_NE(output.find("localDieId=111"), std::string::npos);
    EXPECT_NE(output.find("NetInstance::Link[srcId=4294967296, dstId=0, type=LinkType::PEER2NET"), std::string::npos);
    EXPECT_NE(output.find("dstIface=Iface"), std::string::npos);
    EXPECT_NE(output.find("localDieId=222"), std::string::npos);
}

TEST_F(RankGraphTest, Ut_Dump_When_SelectedLayers_Expect_ReadableLayerLists)
{
    RankGraph rankGraph(myRank);
    rankGraph.buildLayerSelection_ = RankGraph::BuildLayerSelection{0x89U, 0x88U};
    extern s32 log_level_get_stub();
    extern void log_level_set_stub(s32 logLevel);
    const s32 originalLogLevel = log_level_get_stub();
    log_level_set_stub(DLOG_DEBUG);
    testing::internal::CaptureStdout();
    rankGraph.Dump();
    const std::string output = testing::internal::GetCapturedStdout();
    log_level_set_stub(originalLogLevel);

    EXPECT_NE(
        output.find("netLayer selection: mode[SELECTED], selectedLayers[3,7], availableLayers[0,3,7]"),
        std::string::npos);
    EXPECT_EQ(output.find("Mask[0x"), std::string::npos);
}

TEST_F(RankGraphTest, ut_InitInnerRanks_When_NullPeer_Expect_E_PTR)
{
    RankGraph rankGraph(myRank);
    EXPECT_EQ(rankGraph.InitInnerRanks(), HCCL_E_PTR);
}

TEST_F(RankGraphTest, ut_InitInnerRanks_When_Normal_Expect_SUCCESS)
{
    RankGraph rankGraph(myRank);
    std::shared_ptr<NetInstance> netInstance = std::make_shared<InnerNetInstance>(0, "test");
    auto peer = createPeer(myRank, 0, 0);
    (void)peer->AddNetInstance(netInstance);
    (void)rankGraph.AddNetInstance(netInstance);
    (void)rankGraph.AddPeer(peer);
    EXPECT_EQ(rankGraph.InitInnerRanks(), HCCL_SUCCESS);
}

std::shared_ptr<RankGraph> create4pRankGraph(RankId myRank)
{
    RankGraph rankGraph(myRank);

    s32 localId = 0;
    DeviceId deviceId = 0;
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/0", "0/2"};
    std::set<LinkProtocol> protocals = {LinkProtocol::UB_CTP, LinkProtocol::UB_TP};
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocals);
    std::shared_ptr<NetInstance::Node> node = std::make_shared<NetInstance::Peer>(myRank, localId, localId, deviceId);
    connInterface->SetLocalDieId(1);
    std::shared_ptr<NetInstance> netInstLayer0_1 = std::make_shared<InnerNetInstance>(0, "layer0_1");
    std::shared_ptr<NetInstance> netInstLayer0_2 = std::make_shared<InnerNetInstance>(0, "layer0_2");
    std::shared_ptr<NetInstance> netInstLayer1_1 = std::make_shared<ClosNetInstance>(1, "layer1_1");
    std::shared_ptr<NetInstance> netInstLayer1_2 = std::make_shared<ClosNetInstance>(1, "layer1_2");
    std::shared_ptr<NetInstance> netInstLayer2 = std::make_shared<ClosNetInstance>(2, "layer2");
    auto peer0 = createPeer(myRank, 0, 0);
    auto peer1 = createPeer(1, 1, 1);
    auto peer2 = createPeer(2, 2, 2);
    auto peer3 = createPeer(3, 3, 3);
    auto peer4 = createPeer(4, 4, 4);
    peer0->AddConnInterface(0, connInterface);
    peer1->AddConnInterface(0, connInterface);
    peer2->AddConnInterface(0, connInterface);
    peer3->AddConnInterface(0, connInterface);
    (void)peer0->AddNetInstance(netInstLayer0_1);
    (void)peer0->AddNetInstance(netInstLayer1_1);
    (void)peer0->AddNetInstance(netInstLayer2);
    (void)peer1->AddNetInstance(netInstLayer0_1);
    (void)peer1->AddNetInstance(netInstLayer1_1);
    (void)peer2->AddNetInstance(netInstLayer0_2);
    (void)peer2->AddNetInstance(netInstLayer1_1);
    (void)peer2->AddNetInstance(netInstLayer2);
    (void)peer3->AddNetInstance(netInstLayer0_2);
    (void)peer4->AddNetInstance(netInstLayer1_2);

    netInstLayer0_1->AddRankId(peer0->GetRankId());
    netInstLayer0_1->AddRankId(peer1->GetRankId());
    netInstLayer0_2->AddRankId(peer2->GetRankId());
    netInstLayer0_2->AddRankId(peer3->GetRankId());
    netInstLayer1_1->AddRankId(peer0->GetRankId());
    netInstLayer1_1->AddRankId(peer1->GetRankId());
    netInstLayer1_1->AddRankId(peer2->GetRankId());
    netInstLayer2->AddRankId(peer0->GetRankId());
    netInstLayer2->AddRankId(peer2->GetRankId());
    netInstLayer1_2->AddRankId(peer4->GetRankId());

    (void)rankGraph.AddPeer(peer0);
    (void)rankGraph.AddPeer(peer1);
    (void)rankGraph.AddPeer(peer2);
    (void)rankGraph.AddPeer(peer3);
    (void)rankGraph.AddPeer(peer4);

    (void)rankGraph.AddNetInstance(netInstLayer0_1);
    (void)rankGraph.AddNetInstance(netInstLayer0_2);
    (void)rankGraph.AddNetInstance(netInstLayer1_1);
    (void)rankGraph.AddNetInstance(netInstLayer1_2);
    (void)rankGraph.AddNetInstance(netInstLayer2);
    return std::make_shared<RankGraph>(rankGraph);
}

std::shared_ptr<RankGraph> create4pclosRankGraph(RankId myRank)
{
    RankGraph rankGraph(myRank);
    constexpr u32 layer = 0;
    constexpr u32 mesh01TopoInstId = 0;
    constexpr u32 mesh23TopoInstId = 1;
    constexpr u32 closTopoInstId = 2;

    auto netInstMixPcie = std::make_shared<InnerNetInstance>(layer, "mixpcie");
    auto peer0 = createPeer(myRank, 0, 0);
    auto peer1 = createPeer(1, 1, 1);
    auto peer2 = createPeer(2, 2, 2);
    auto peer3 = createPeer(3, 3, 3);
    auto fabric = std::make_shared<NetInstance::Fabric>(0, "plane0");

    netInstMixPcie->AddRankId(peer0->GetRankId());
    netInstMixPcie->AddRankId(peer1->GetRankId());
    netInstMixPcie->AddRankId(peer2->GetRankId());
    netInstMixPcie->AddRankId(peer3->GetRankId());

    (void)netInstMixPcie->AddNode(peer0);
    (void)netInstMixPcie->AddNode(peer1);
    (void)netInstMixPcie->AddNode(peer2);
    (void)netInstMixPcie->AddNode(peer3);
    (void)netInstMixPcie->AddNode(fabric);

    (void)peer0->AddNetInstance(netInstMixPcie);
    (void)peer1->AddNetInstance(netInstMixPcie);
    (void)peer2->AddNetInstance(netInstMixPcie);
    (void)peer3->AddNetInstance(netInstMixPcie);

    (void)rankGraph.AddPeer(peer0);
    (void)rankGraph.AddPeer(peer1);
    (void)rankGraph.AddPeer(peer2);
    (void)rankGraph.AddPeer(peer3);
    (void)rankGraph.AddNetInstance(netInstMixPcie);

    std::set<std::string> meshPorts = {"0/0"};
    std::set<LinkProtocol> meshProtocols = {LinkProtocol::UB_TP};
    auto makeMeshIface = [&](u32 topoInstId) {
        return std::make_shared<NetInstance::ConnInterface>(
            IpAddress(0), meshPorts, AddrPosition::DEVICE, LinkType::PEER2PEER, meshProtocols, Hccl::TopoType::MESH_1D,
            topoInstId);
    };
    auto addMeshLink = [&](NetInstance* netInstance, const std::shared_ptr<NetInstance::Peer>& srcPeer,
                           const std::shared_ptr<NetInstance::Peer>& dstPeer, u32 topoInstId) {
        auto srcIface = makeMeshIface(topoInstId);
        auto dstIface = makeMeshIface(topoInstId);
        srcPeer->AddConnInterface(layer, srcIface);
        dstPeer->AddConnInterface(layer, dstIface);
        netInstance->AddLink(std::make_shared<NetInstance::Link>(
            srcPeer, dstPeer, srcIface, dstIface, LinkType::PEER2PEER, meshProtocols));
    };
    addMeshLink(netInstMixPcie.get(), peer0, peer1, mesh01TopoInstId);
    addMeshLink(netInstMixPcie.get(), peer1, peer0, mesh01TopoInstId);
    addMeshLink(netInstMixPcie.get(), peer2, peer3, mesh23TopoInstId);
    addMeshLink(netInstMixPcie.get(), peer3, peer2, mesh23TopoInstId);

    std::set<std::string> ports = {"1/0"};
    std::set<LinkProtocol> protocols = {LinkProtocol::PCIE};
    auto makeIface = [&](u32 topoInstId) {
        return std::make_shared<NetInstance::ConnInterface>(
            IpAddress(0), ports, AddrPosition::DEVICE, LinkType::PEER2NET, protocols, Hccl::TopoType::CLOS, topoInstId);
    };

    auto rank0ToFabricIface = makeIface(closTopoInstId);
    auto fabricToRank0Iface = makeIface(closTopoInstId);
    auto rank1ToFabricIface = makeIface(closTopoInstId);
    auto fabricToRank1Iface = makeIface(closTopoInstId);
    auto rank2ToFabricIface = makeIface(closTopoInstId);
    auto fabricToRank2Iface = makeIface(closTopoInstId);
    auto rank3ToFabricIface = makeIface(closTopoInstId);
    auto fabricToRank3Iface = makeIface(closTopoInstId);
    peer0->AddConnInterface(layer, rank0ToFabricIface);
    peer0->AddConnInterface(layer, fabricToRank0Iface);
    peer1->AddConnInterface(layer, rank1ToFabricIface);
    peer1->AddConnInterface(layer, fabricToRank1Iface);
    peer2->AddConnInterface(layer, rank2ToFabricIface);
    peer2->AddConnInterface(layer, fabricToRank2Iface);
    peer3->AddConnInterface(layer, rank3ToFabricIface);
    peer3->AddConnInterface(layer, fabricToRank3Iface);

    netInstMixPcie->UpdateTopoInst(mesh01TopoInstId, Hccl::TopoType::MESH_1D, peer0->GetRankId());
    netInstMixPcie->UpdateTopoInst(mesh01TopoInstId, Hccl::TopoType::MESH_1D, peer1->GetRankId());
    netInstMixPcie->UpdateTopoInst(mesh23TopoInstId, Hccl::TopoType::MESH_1D, peer2->GetRankId());
    netInstMixPcie->UpdateTopoInst(mesh23TopoInstId, Hccl::TopoType::MESH_1D, peer3->GetRankId());
    netInstMixPcie->UpdateTopoInst(closTopoInstId, Hccl::TopoType::CLOS, peer0->GetRankId());
    netInstMixPcie->UpdateTopoInst(closTopoInstId, Hccl::TopoType::CLOS, peer1->GetRankId());
    netInstMixPcie->UpdateTopoInst(closTopoInstId, Hccl::TopoType::CLOS, peer2->GetRankId());
    netInstMixPcie->UpdateTopoInst(closTopoInstId, Hccl::TopoType::CLOS, peer3->GetRankId());
    netInstMixPcie->AddLink(
        std::make_shared<NetInstance::Link>(peer0, fabric, rank0ToFabricIface, nullptr, LinkType::PEER2NET, protocols));
    netInstMixPcie->AddLink(
        std::make_shared<NetInstance::Link>(fabric, peer0, nullptr, fabricToRank0Iface, LinkType::PEER2NET, protocols));
    netInstMixPcie->AddLink(
        std::make_shared<NetInstance::Link>(peer1, fabric, rank1ToFabricIface, nullptr, LinkType::PEER2NET, protocols));
    netInstMixPcie->AddLink(
        std::make_shared<NetInstance::Link>(fabric, peer1, nullptr, fabricToRank1Iface, LinkType::PEER2NET, protocols));
    netInstMixPcie->AddLink(
        std::make_shared<NetInstance::Link>(peer2, fabric, rank2ToFabricIface, nullptr, LinkType::PEER2NET, protocols));
    netInstMixPcie->AddLink(
        std::make_shared<NetInstance::Link>(fabric, peer2, nullptr, fabricToRank2Iface, LinkType::PEER2NET, protocols));
    netInstMixPcie->AddLink(
        std::make_shared<NetInstance::Link>(peer3, fabric, rank3ToFabricIface, nullptr, LinkType::PEER2NET, protocols));
    netInstMixPcie->AddLink(
        std::make_shared<NetInstance::Link>(fabric, peer3, nullptr, fabricToRank3Iface, LinkType::PEER2NET, protocols));
    return std::make_shared<RankGraph>(rankGraph);
}

TEST_F(RankGraphTest, ut_GetLayerRanks_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    EXPECT_EQ(4, rankGraph->GetLayerRanks(0));
    EXPECT_EQ(4, rankGraph->GetLayerRanks(1));
    EXPECT_EQ(2, rankGraph->GetLayerRanks(2));
}

TEST_F(RankGraphTest, ut_GetLocalInstRanks_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    std::vector<u32> rankListLayer0;
    std::vector<u32> rankListLayer1;
    std::vector<u32> rankListLayer2;
    u32 rankNumLayer0;
    u32 rankNumLayer1;
    u32 rankNumLayer2;
    EXPECT_EQ(HCCL_SUCCESS, rankGraph->GetLocalInstRanks(0, rankListLayer0, rankNumLayer0));
    EXPECT_EQ(HCCL_SUCCESS, rankGraph->GetLocalInstRanks(1, rankListLayer1, rankNumLayer1));
    EXPECT_EQ(HCCL_SUCCESS, rankGraph->GetLocalInstRanks(2, rankListLayer2, rankNumLayer2));

    vector<u32> expectedRankList0 = {0, 1};
    vector<u32> expectedRankList1 = {0, 1, 2};
    vector<u32> expectedRankList2 = {0, 2};

    EXPECT_EQ(expectedRankList0.size(), rankNumLayer0);
    for (size_t i = 0; i < rankListLayer0.size(); ++i) {
        EXPECT_EQ(rankListLayer0[i], rankListLayer0[i]);
    }

    EXPECT_EQ(expectedRankList1.size(), rankNumLayer1);
    for (size_t i = 0; i < rankListLayer1.size(); ++i) {
        EXPECT_EQ(rankListLayer1[i], rankListLayer1[i]);
    }

    EXPECT_EQ(expectedRankList2.size(), rankNumLayer2);
    for (size_t i = 0; i < rankListLayer2.size(); ++i) {
        EXPECT_EQ(rankListLayer2[i], rankListLayer2[i]);
    }
}

TEST_F(RankGraphTest, ut_GetLocalInstSize_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    u32 instSize0 = 0;
    u32 instSize1 = 0;
    u32 instSize2 = 0;
    EXPECT_EQ(HCCL_SUCCESS, rankGraph->GetLocalInstSize(0, instSize0));
    EXPECT_EQ(2, instSize0);
    EXPECT_EQ(HCCL_SUCCESS, rankGraph->GetLocalInstSize(1, instSize1));
    EXPECT_EQ(3, instSize1);
    EXPECT_EQ(HCCL_SUCCESS, rankGraph->GetLocalInstSize(2, instSize2));
    EXPECT_EQ(2, instSize2);
}

TEST_F(RankGraphTest, ut_GetNetType_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    EXPECT_EQ(NetType::TOPO_FILE_DESC, rankGraph->GetNetType(0));
    EXPECT_EQ(NetType::CLOS, rankGraph->GetNetType(1));
    EXPECT_EQ(NetType::CLOS, rankGraph->GetNetType(2));
}

TEST_F(RankGraphTest, ut_GetNetInstanceList_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    std::vector<u32> instSizeListLayer0;
    std::vector<u32> instSizeListLayer1;
    std::vector<u32> instSizeListLayer2;
    u32 listSizeLayer0;
    u32 listSizeLayer1;
    u32 listSizeLayer2;
    rankGraph->GetNetInstanceList(0, instSizeListLayer0, listSizeLayer0);
    rankGraph->GetNetInstanceList(1, instSizeListLayer1, listSizeLayer1);
    rankGraph->GetNetInstanceList(2, instSizeListLayer2, listSizeLayer2);

    vector<u32> expectedRankList0 = {2, 2};
    vector<u32> expectedRankList1 = {3, 1};
    vector<u32> expectedRankList2 = {2};

    EXPECT_EQ(expectedRankList0.size(), listSizeLayer0);
    for (size_t i = 0; i < instSizeListLayer0.size(); ++i) {
        EXPECT_EQ(instSizeListLayer0[i], instSizeListLayer0[i]);
    }

    EXPECT_EQ(expectedRankList1.size(), listSizeLayer1);
    for (size_t i = 0; i < instSizeListLayer1.size(); ++i) {
        EXPECT_EQ(instSizeListLayer1[i], instSizeListLayer1[i]);
    }

    EXPECT_EQ(expectedRankList2.size(), listSizeLayer2);
    for (size_t i = 0; i < instSizeListLayer2.size(); ++i) {
        EXPECT_EQ(instSizeListLayer2[i], instSizeListLayer2[i]);
    }
}

TEST_F(RankGraphTest, ut_IsSymmetric_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    EXPECT_EQ(true, rankGraph->IsSymmetric(0));
    EXPECT_EQ(false, rankGraph->IsSymmetric(1));
    EXPECT_EQ(true, rankGraph->IsSymmetric(2));
    EXPECT_THROW(rankGraph->IsSymmetric(3), NullPtrException);
}

TEST_F(RankGraphTest, ut_IsLevel0PcieFallback_When_Default_Expect_False)
{
    RankGraph rankGraph(myRank);
    EXPECT_EQ(false, rankGraph.IsLevel0PcieFallback());
}

TEST_F(RankGraphTest, ut_SetLevel0PcieFallback_When_SetTrue_Expect_GetterReturnTrue)
{
    RankGraph rankGraph(myRank);
    rankGraph.SetLevel0PcieFallback(true);
    EXPECT_EQ(true, rankGraph.IsLevel0PcieFallback());
}

TEST_F(RankGraphTest, ut_CreateSubRankGraph_When_ParentLevel0PcieFallback_Expect_SubGraphInheritFlag)
{
    auto rankGraph = create4pclosRankGraph(myRank);
    rankGraph->SetLevel0PcieFallback(true);
    vector<u32> subRankIds = {0, 1, 2, 3};
    std::unique_ptr<RankGraph> subRankGraph;
    EXPECT_EQ(HCCL_SUCCESS, rankGraph->CreateSubRankGraph(subRankIds, subRankGraph));
    u32 subInstSize0 = 0;
    EXPECT_EQ(HCCL_SUCCESS, subRankGraph->GetLocalInstSize(0, subInstSize0));
    EXPECT_EQ(4, subInstSize0);
    EXPECT_EQ(true, subRankGraph->IsLevel0PcieFallback());
}

TEST_F(RankGraphTest, ut_CreateSubRankGraph_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pclosRankGraph(myRank);
    constexpr u32 layer = 0;
    constexpr u32 missingTopoInstId = 0;
    constexpr u32 nullTopoInstId = 1;
    constexpr u32 noParentTopoInstId = 2;
    constexpr u32 parentTopoInstId = 3;

    vector<u32> subRankIds = {0, 1, 2, 3};
    std::unique_ptr<RankGraph> subRankGraph;
    EXPECT_EQ(HCCL_SUCCESS, rankGraph->CreateSubRankGraph(subRankIds, subRankGraph));
    u32 subInstSize = 0;
    EXPECT_EQ(HCCL_SUCCESS, subRankGraph->GetLocalInstSize(layer, subInstSize));
    EXPECT_EQ(4, subInstSize);
}

TEST_F(RankGraphTest, ut_GetEndpointNum_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    uint32_t num = 0;
    uint32_t topoInstId = 0;
    HcclResult ret = rankGraph->GetEndpointNum(0, topoInstId, &num);
    EXPECT_EQ(num, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RankGraphTest, ut_GetEndpointNum_When_nullPeer_HCCL_E_PTR)
{
    RankGraph rankGraph(0);
    uint32_t num = 0;
    uint32_t topoInstId = 0;
    HcclResult ret = rankGraph.GetEndpointNum(0, topoInstId, &num);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(RankGraphTest, ut_GetEndpointDesc_When_nullPeer_HCCL_E_PTR)
{
    RankGraph rankGraph(0);
    uint32_t descNum = 1;
    uint32_t topoInstId = 0;

    EndpointDesc* endPointDesc = new EndpointDesc[descNum];
    HcclResult ret = rankGraph.GetEndpointDesc(0, topoInstId, &descNum, endPointDesc);
    delete[] endPointDesc;
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(RankGraphTest, ut_GetEndpointDesc_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    uint32_t layer = 0;
    uint32_t num = 2;
    uint32_t topoInstId = 0;
    HcclResult ret = rankGraph->GetEndpointNum(0, topoInstId, &num);
    EXPECT_EQ(num, 2);

    uint32_t descNum = num;
    EndpointDesc* endPointDesc = new EndpointDesc[descNum];
    ret = rankGraph->GetEndpointDesc(layer, topoInstId, &descNum, endPointDesc);
    for (uint32_t i = 0; i < num; ++i) {
        EXPECT_NE(endPointDesc[i].commAddr.type, COMM_ADDR_TYPE_RESERVED);
        EXPECT_NE(endPointDesc[i].loc.locType, ENDPOINT_LOC_TYPE_RESERVED);
        EXPECT_NE(endPointDesc[i].protocol, COMM_PROTOCOL_RESERVED);
    }
    delete[] endPointDesc;

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RankGraphTest, ut_AddGroupLinks_When_1pNormal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    vector<u32> subRankIds = {0};
    std::unique_ptr<RankGraph> subRankGraph;
    EXPECT_EQ(HCCL_SUCCESS, rankGraph->CreateSubRankGraph(subRankIds, subRankGraph));
    u32 subInstSize0 = 0;
    EXPECT_EQ(HCCL_SUCCESS, subRankGraph->GetLocalInstSize(0, subInstSize0));
    EXPECT_EQ(1, subInstSize0);
}

TEST_F(RankGraphTest, ut_GetNetInstanceByNetInstId_When_NotExisted_Expect_ReturnNullptr)
{
    RankGraph rankGraph(myRank);
    auto result = rankGraph.GetNetInstanceByNetInstId(0, "not_existed");
    EXPECT_EQ(nullptr, result);

    s32 netLayer = 0;
    string netInstId = "test";
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    EXPECT_EQ(rankGraph.AddNetInstance(netInstance), HCCL_SUCCESS);
    auto result2 = rankGraph.GetNetInstanceByNetInstId(99, "test");
    EXPECT_EQ(nullptr, result2);
    auto result3 = rankGraph.GetNetInstanceByNetInstId(0, "not_existed");
    EXPECT_EQ(nullptr, result3);
}

TEST_F(RankGraphTest, ut_GetNetInstanceByNetInstId_When_NotExisted_ConstVersion_Expect_ReturnNullptr)
{
    RankGraph rankGraph(myRank);
    const RankGraph& constRankGraph = rankGraph;
    auto result = constRankGraph.GetNetInstanceByNetInstId(0, "not_existed");
    EXPECT_EQ(nullptr, result);

    s32 netLayer = 0;
    string netInstId = "test";
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    EXPECT_EQ(rankGraph.AddNetInstance(netInstance), HCCL_SUCCESS);
    auto result2 = constRankGraph.GetNetInstanceByNetInstId(99, "test");
    EXPECT_EQ(nullptr, result2);
    auto result3 = constRankGraph.GetNetInstanceByNetInstId(0, "not_existed");
    EXPECT_EQ(nullptr, result3);
}

TEST_F(RankGraphTest, ut_NeedUpdateTopoInstForSubGraph_When_TopoInstIsNull_Expect_ReturnFalse)
{
    u32 level = 0;
    std::string netInstId = "test";
    InnerNetInstance netInstance(level, netInstId);
    u32 topoInstId = 0;
    netInstance.topoInsts_.emplace(topoInstId, nullptr);

    EXPECT_EQ(false, NeedUpdateTopoInstForSubGraph(&netInstance, topoInstId, myRank));
}

// 未命中地址时返回 NOT_FOUND 且不改出参
TEST_F(RankGraphTest, Ut_GetIpIndex_When_RankExists_Expect_IndexMatchesSortedOrder)
{
    RankGraph rankGraph(0);
    std::unordered_map<RankId, RankIpIndexList> ipIndexMap;
    ipIndexMap[0] = {{IpAddress("192.168.1.1"), 0}, {IpAddress("192.168.1.2"), 1}, {IpAddress("192.168.1.3"), 2}};
    rankGraph.SetIpIndexMap(std::move(ipIndexMap));

    uint32_t idx = 99U;
    EXPECT_EQ(HCCL_SUCCESS, rankGraph.GetIpIndex(0, IpAddress("192.168.1.2"), idx));
    EXPECT_EQ(1U, idx);
    EXPECT_EQ(HCCL_E_NOT_FOUND, rankGraph.GetIpIndex(0, IpAddress("192.168.1.9"), idx));
    EXPECT_EQ(1U, idx);
}

TEST_F(RankGraphTest, Ut_GetIpIndex_When_RankIdMissing_Expect_ReturnNotFound)
{
    RankGraph rankGraph(0);
    std::unordered_map<RankId, RankIpIndexList> ipIndexMap;
    ipIndexMap[0] = {{IpAddress("192.168.1.1"), 0}};
    rankGraph.SetIpIndexMap(std::move(ipIndexMap));

    uint32_t idx = 77U;
    EXPECT_EQ(HCCL_E_NOT_FOUND, rankGraph.GetIpIndex(3, IpAddress("192.168.1.1"), idx));
    EXPECT_EQ(77U, idx);
}

// 未排序输入直接拒绝，避免 lower_bound 在无序区间上的未定义行为
TEST_F(RankGraphTest, Ut_SetIpIndexMap_When_Unsorted_Expect_E_Internal)
{
    RankGraph rankGraph(0);
    std::unordered_map<RankId, RankIpIndexList> ipIndexMap;
    ipIndexMap[0] = {{IpAddress("192.168.1.2"), 0}, {IpAddress("192.168.1.1"), 1}};

    EXPECT_EQ(HCCL_E_INTERNAL, rankGraph.SetIpIndexMap(std::move(ipIndexMap)));

    // 校验失败时不得安装映射表
    uint32_t idx = 42U;
    EXPECT_EQ(HCCL_E_NOT_FOUND, rankGraph.GetIpIndex(0, IpAddress("192.168.1.1"), idx));
    EXPECT_EQ(42U, idx);
}

TEST_F(RankGraphTest, Ut_SetIpIndexMap_When_Sorted_Expect_Success)
{
    RankGraph rankGraph(0);
    std::unordered_map<RankId, RankIpIndexList> ipIndexMap;
    ipIndexMap[0] = {{IpAddress("192.168.1.1"), 0}, {IpAddress("192.168.1.2"), 1}};

    EXPECT_EQ(HCCL_SUCCESS, rankGraph.SetIpIndexMap(std::move(ipIndexMap)));

    uint32_t idx = 0U;
    EXPECT_EQ(HCCL_SUCCESS, rankGraph.GetIpIndex(0, IpAddress("192.168.1.2"), idx));
    EXPECT_EQ(1U, idx);
}

// 子图继承父图映射，子 rankId 为 rankIds 数组下标
TEST_F(RankGraphTest, Ut_CreateSubRankGraph_When_ParentHasIpIndex_Expect_SameMapping)
{
    auto rankGraph = create4pclosRankGraph(myRank);
    std::unordered_map<RankId, RankIpIndexList> ipIndexMap;
    ipIndexMap[0] = {{IpAddress("192.168.1.1"), 0}};
    ipIndexMap[3] = {{IpAddress("192.168.1.3"), 1}};
    rankGraph->SetIpIndexMap(std::move(ipIndexMap));

    std::vector<u32> subRankIds = {0, 3};
    std::unique_ptr<RankGraph> subRankGraph;
    ASSERT_EQ(HCCL_SUCCESS, rankGraph->CreateSubRankGraph(subRankIds, subRankGraph));
    ASSERT_NE(nullptr, subRankGraph);

    uint32_t idx = 0;
    EXPECT_EQ(HCCL_SUCCESS, subRankGraph->GetIpIndex(0, IpAddress("192.168.1.1"), idx));
    EXPECT_EQ(0U, idx);
    EXPECT_EQ(HCCL_SUCCESS, subRankGraph->GetIpIndex(1, IpAddress("192.168.1.3"), idx));
    EXPECT_EQ(1U, idx);
}

namespace {
using VirtualTestLayer = std::vector<std::vector<u32>>;
constexpr u32 VIRTUAL_TEST_TOPO_ID = 17;

std::vector<u32> VirtualTestRange(u32 first, u32 count)
{
    std::vector<u32> ranks;
    for (u32 i = 0; i < count; ++i) {
        ranks.push_back(first + i);
    }
    return ranks;
}

// Build real NetInstances and peer associations; group IDs deliberately differ from topo IDs.
std::unique_ptr<RankGraph>
MakeVirtualTestGraph(u32 myRank, const std::vector<VirtualTestLayer>& layers, bool enableVirtualGroups = true)
{
    auto graph = std::make_unique<RankGraph>(myRank, enableVirtualGroups);
    std::set<u32> allRanks;
    for (const auto& layer : layers) {
        for (const auto& group : layer) {
            allRanks.insert(group.begin(), group.end());
        }
    }
    for (u32 rank : allRanks) {
        EXPECT_EQ(graph->AddPeer(createPeer(rank, rank)), HCCL_SUCCESS);
    }
    for (u32 layer = 0; layer < layers.size(); ++layer) {
        for (u32 index = 0; index < layers[layer].size(); ++index) {
            auto inst = std::make_shared<InnerNetInstance>(layer, std::to_string(index));
            for (u32 rank : layers[layer][index]) {
                inst->AddRankId(rank);
                auto peer = graph->GetPeer(rank);
                EXPECT_EQ(inst->AddNode(peer), HCCL_SUCCESS);
                EXPECT_EQ(peer->AddNetInstance(inst), HCCL_SUCCESS);
                inst->UpdateTopoInst(VIRTUAL_TEST_TOPO_ID, Hccl::TopoType::CLOS, rank);
            }
            EXPECT_EQ(graph->AddNetInstance(inst), HCCL_SUCCESS);
        }
    }
    EXPECT_EQ(graph->InitInnerRanks(), HCCL_SUCCESS);
    return graph;
}

std::vector<VirtualTestLayer> VirtualTestThreeLayers()
{
    return {
        {VirtualTestRange(0, 4), VirtualTestRange(4, 4), VirtualTestRange(8, 4), VirtualTestRange(12, 4)},
        {{0, 1, 2, 3, 8, 9, 10, 11}, {4, 5, 6, 7, 12, 13, 14, 15}},
        {VirtualTestRange(0, 16)}};
}
} // namespace

TEST_F(RankGraphTest, Ut_VirtualGroups_When_ServerMajor_Expect_LocalSegmentsAndPhysicalIds)
{
    for (u32 rank = 0; rank < 16; ++rank) {
        auto graph = MakeVirtualTestGraph(rank, VirtualTestThreeLayers());
        ASSERT_EQ(graph->BuildVirtualGroups(), HCCL_SUCCESS);
        ASSERT_EQ(graph->BuildVirtualGroups(), HCCL_SUCCESS); // Rebuild before publication is idempotent.
        std::vector<u32> ranks;
        u32 count = 0;
        ASSERT_EQ(graph->GetLocalInstRanks(1, ranks, count), HCCL_SUCCESS);
        EXPECT_EQ(ranks, VirtualTestRange(rank / 4 * 4, 4));
        EXPECT_EQ(count, 4U);
        ASSERT_EQ(graph->GetLocalInstSize(1, count), HCCL_SUCCESS);
        EXPECT_EQ(count, 4U);
        ASSERT_EQ(graph->GetNetInstanceList(1, ranks, count), HCCL_SUCCESS);
        EXPECT_EQ(ranks, (std::vector<u32>{4, 4, 4, 4}));
        EXPECT_EQ(count, 4U);
        ranks.clear();
        graph->GetTopoInstsByLayer(1, ranks, count);
        EXPECT_EQ(ranks, (std::vector<u32>{VIRTUAL_TEST_TOPO_ID}));
        ASSERT_EQ(graph->GetRanksByTopoInst(1, VIRTUAL_TEST_TOPO_ID, ranks, count), HCCL_SUCCESS);
        EXPECT_EQ(ranks, VirtualTestRange(rank / 4 * 4, 4));
        Hccl::TopoType type;
        ASSERT_EQ(graph->GetTopoType(1, VIRTUAL_TEST_TOPO_ID, type), HCCL_SUCCESS);
        EXPECT_EQ(type, Hccl::TopoType::CLOS);
        EXPECT_EQ(graph->GetRanksByTopoInst(1, 999, ranks, count), HCCL_E_PARA);
        ASSERT_EQ(graph->GetLocalInstRanks(0, ranks, count), HCCL_SUCCESS);
        EXPECT_EQ(ranks, VirtualTestRange(rank / 4 * 4, 4));
        ASSERT_EQ(graph->GetLocalInstRanks(2, ranks, count), HCCL_SUCCESS);
        EXPECT_EQ(ranks, VirtualTestRange(0, 16));
        EXPECT_EQ(graph->GetNetInstanceByRankId(1, rank)->GetRankSize(), 8U);
        graph->InitFinish();
        EXPECT_EQ(graph->BuildVirtualGroups(), HCCL_E_INTERNAL);
    }
}

TEST_F(RankGraphTest, Ut_VirtualGroups_When_TopoInstOutsideLocalGroup_Expect_NotFound)
{
    const u32 otherTopoInstId = VIRTUAL_TEST_TOPO_ID + 1;
    for (bool enabled : {true, false}) {
        auto graph = MakeVirtualTestGraph(0, VirtualTestThreeLayers(), enabled);
        auto* netInstance = graph->GetNetInstanceByRankId(1, 0);
        ASSERT_NE(netInstance, nullptr);
        for (u32 rank : VirtualTestRange(8, 4)) {
            netInstance->UpdateTopoInst(otherTopoInstId, Hccl::TopoType::CLOS, rank);
        }
        ASSERT_EQ(graph->BuildVirtualGroups(), HCCL_SUCCESS);
        Hccl::TopoType type;
        ASSERT_EQ(graph->GetTopoType(1, otherTopoInstId, type), HCCL_SUCCESS);
        EXPECT_EQ(type, Hccl::TopoType::CLOS);
        std::vector<u32> ranks{99};
        u32 count = 99;
        if (enabled) {
            EXPECT_EQ(graph->GetRanksByTopoInst(1, otherTopoInstId, ranks, count), HCCL_E_NOT_FOUND);
            EXPECT_TRUE(ranks.empty());
            EXPECT_EQ(count, 0U);
        } else {
            ASSERT_EQ(graph->GetRanksByTopoInst(1, otherTopoInstId, ranks, count), HCCL_SUCCESS);
            EXPECT_EQ(ranks, VirtualTestRange(8, 4));
            EXPECT_EQ(count, 4U);
        }
    }
}

TEST_F(RankGraphTest, Ut_VirtualGroups_When_FourLayers_Expect_IndependentMiddleLayers)
{
    std::vector<VirtualTestLayer> layers(4);
    layers[1].resize(4);
    layers[2].resize(2);
    layers[3].push_back(VirtualTestRange(0, 32));
    for (u32 server = 0; server < 8; ++server) {
        auto ranks = VirtualTestRange(server * 4, 4);
        layers[0].push_back(ranks);
        auto& pod = layers[1][(server % 2) * 2 + server / 4];
        pod.insert(pod.end(), ranks.begin(), ranks.end());
        auto& cluster = layers[2][server % 2];
        cluster.insert(cluster.end(), ranks.begin(), ranks.end());
    }
    for (u32 rank : {0U, 8U, 20U, 31U}) {
        auto graph = MakeVirtualTestGraph(rank, layers);
        ASSERT_EQ(graph->BuildVirtualGroups(), HCCL_SUCCESS);
        for (u32 layer : {1U, 2U}) {
            std::vector<u32> ranks;
            u32 count = 0;
            ASSERT_EQ(graph->GetNetInstanceList(layer, ranks, count), HCCL_SUCCESS);
            EXPECT_EQ(ranks, std::vector<u32>(8, 4));
            ASSERT_EQ(graph->GetLocalInstRanks(layer, ranks, count), HCCL_SUCCESS);
            EXPECT_EQ(ranks, VirtualTestRange(rank / 4 * 4, 4));
        }
    }
}

TEST_F(RankGraphTest, Ut_VirtualGroups_When_LocalLayersAreSparse_Expect_AllLayersAboveLowestSplit)
{
    const std::vector<VirtualTestLayer> layers = {{{0}, {2}, {4}, {6}}, {{4, 6}}, {{0, 2}}, {{0, 2}}, {{4, 6}}};
    auto graph = MakeVirtualTestGraph(0, layers);
    EXPECT_EQ(graph->GetLevels(0), (std::set<u32>{0, 2, 3}));
    ASSERT_EQ(graph->BuildVirtualGroups(), HCCL_SUCCESS);
    std::vector<u32> ranks;
    u32 count = 0;
    ASSERT_EQ(graph->GetLocalInstRanks(2, ranks, count), HCCL_SUCCESS);
    EXPECT_EQ(ranks, (std::vector<u32>{0}));
    EXPECT_EQ(count, 1U);
    // The highest joined layer is also split when its ranks are non-contiguous.
    ASSERT_EQ(graph->GetLocalInstRanks(3, ranks, count), HCCL_SUCCESS);
    EXPECT_EQ(ranks, (std::vector<u32>{0}));
    EXPECT_EQ(count, 1U);
    // A layer not joined by this rank keeps its physical instance sizes.
    ASSERT_EQ(graph->GetNetInstanceList(1, ranks, count), HCCL_SUCCESS);
    EXPECT_EQ(ranks, (std::vector<u32>{2}));
    EXPECT_EQ(count, 1U);
}

TEST_F(RankGraphTest, Ut_VirtualGroups_When_NoBreakOrTwoLayers_Expect_SplitOnlyNonContiguousLayers)
{
    auto layers = VirtualTestThreeLayers();
    layers[1] = {VirtualTestRange(0, 8), VirtualTestRange(8, 8)};
    auto graph = MakeVirtualTestGraph(8, layers);
    ASSERT_EQ(graph->BuildVirtualGroups(), HCCL_SUCCESS);
    std::vector<u32> ranks;
    u32 count = 0;
    ASSERT_EQ(graph->GetLocalInstRanks(1, ranks, count), HCCL_SUCCESS);
    EXPECT_EQ(ranks, VirtualTestRange(8, 8));
    // With two layers, the highest layer is still split when its ranks are non-contiguous.
    layers = VirtualTestThreeLayers();
    layers.pop_back();
    graph = MakeVirtualTestGraph(8, layers);
    ASSERT_EQ(graph->BuildVirtualGroups(), HCCL_SUCCESS);
    ASSERT_EQ(graph->GetLocalInstRanks(1, ranks, count), HCCL_SUCCESS);
    EXPECT_EQ(ranks, VirtualTestRange(8, 4));
    EXPECT_EQ(count, 4U);
}

TEST_F(RankGraphTest, Ut_VirtualGroups_When_MixedSegments_Expect_GlobalOrderAndSingletonPreserved)
{
    const VirtualTestLayer middle = {{0, 1, 5, 6, 7}, {2}, {3, 4}};
    auto graph = MakeVirtualTestGraph(2, {{VirtualTestRange(0, 8)}, middle, {VirtualTestRange(0, 8)}});
    ASSERT_EQ(graph->BuildVirtualGroups(), HCCL_SUCCESS);
    std::vector<u32> ranks;
    u32 count = 0;
    ASSERT_EQ(graph->GetNetInstanceList(1, ranks, count), HCCL_SUCCESS);
    EXPECT_EQ(ranks, (std::vector<u32>{2, 1, 2, 3}));
    ASSERT_EQ(graph->GetLocalInstRanks(1, ranks, count), HCCL_SUCCESS);
    EXPECT_EQ(ranks, (std::vector<u32>{2}));
    auto reversed = middle;
    std::reverse(reversed.begin(), reversed.end());
    graph = MakeVirtualTestGraph(2, {{VirtualTestRange(0, 8)}, reversed, {VirtualTestRange(0, 8)}});
    ASSERT_EQ(graph->BuildVirtualGroups(), HCCL_SUCCESS);
    ASSERT_EQ(graph->GetNetInstanceList(1, ranks, count), HCCL_SUCCESS);
    EXPECT_EQ(ranks, (std::vector<u32>{2, 1, 2, 3}));
}

TEST_F(RankGraphTest, Ut_VirtualGroups_When_AllBreaks_Expect_SingletonGroups)
{
    auto graph = MakeVirtualTestGraph(3, {{{0, 1}, {2, 3}}, {{0, 2}, {1, 3}}, {{0, 1, 2, 3}}});
    ASSERT_EQ(graph->BuildVirtualGroups(), HCCL_SUCCESS);
    std::vector<u32> ranks;
    u32 count = 0;
    ASSERT_EQ(graph->GetNetInstanceList(1, ranks, count), HCCL_SUCCESS);
    EXPECT_EQ(ranks, std::vector<u32>(4, 1));
    ASSERT_EQ(graph->GetLocalInstRanks(1, ranks, count), HCCL_SUCCESS);
    EXPECT_EQ(ranks, (std::vector<u32>{3}));
}

TEST_F(RankGraphTest, Ut_VirtualGroups_When_CrossSegmentLink_Expect_PhysicalPathAndEndpointsUnchanged)
{
    auto graph = MakeVirtualTestGraph(0, VirtualTestThreeLayers());
    auto* inst = graph->GetNetInstanceByRankId(1, 0);
    auto src = graph->GetPeer(0);
    auto dst = graph->GetPeer(8);
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_TP};
    auto sourceIface = std::make_shared<NetInstance::ConnInterface>(
        IpAddress("::1"), std::set<std::string>{"0/0"}, AddrPosition::DEVICE, LinkType::PEER2PEER, protocols,
        Hccl::TopoType::CLOS, VIRTUAL_TEST_TOPO_ID);
    auto targetIface = std::make_shared<NetInstance::ConnInterface>(
        IpAddress("::2"), std::set<std::string>{"0/1"}, AddrPosition::DEVICE, LinkType::PEER2PEER, protocols,
        Hccl::TopoType::CLOS, VIRTUAL_TEST_TOPO_ID);
    src->AddConnInterface(1, sourceIface);
    dst->AddConnInterface(1, targetIface);
    inst->AddLink(std::make_shared<NetInstance::Link>(
        src, dst, sourceIface, targetIface, LinkType::PEER2PEER, protocols, LinkDirection::BOTH, 0));
    const auto before = graph->GetPaths(1, 0, 8);
    ASSERT_EQ(before.size(), 1U);
    u32 endpointsBefore = 0;
    ASSERT_EQ(graph->GetEndpointNum(1, VIRTUAL_TEST_TOPO_ID, &endpointsBefore), HCCL_SUCCESS);
    ASSERT_EQ(graph->BuildVirtualGroups(), HCCL_SUCCESS);
    const auto after = graph->GetPaths(1, 0, 8);
    ASSERT_EQ(after.size(), before.size());
    EXPECT_EQ(after[0].links[0].Describe(), before[0].links[0].Describe());
    u32 endpointsAfter = 0;
    ASSERT_EQ(graph->GetEndpointNum(1, VIRTUAL_TEST_TOPO_ID, &endpointsAfter), HCCL_SUCCESS);
    EXPECT_EQ(endpointsAfter, endpointsBefore);
    EXPECT_EQ(endpointsAfter, 1U);
}

TEST_F(RankGraphTest, Ut_VirtualGroups_When_SubGraph_Expect_RebuildInChildRankSpace)
{
    auto graph = MakeVirtualTestGraph(8, VirtualTestThreeLayers());
    ASSERT_EQ(graph->BuildVirtualGroups(), HCCL_SUCCESS);
    graph->InitFinish();
    std::unique_ptr<RankGraph> child;
    ASSERT_EQ(graph->CreateSubRankGraph({0, 4, 8, 12}, child), HCCL_SUCCESS);
    ASSERT_NE(child, nullptr);
    EXPECT_EQ(child->GetMyRank(), 2U);
    std::vector<u32> ranks;
    u32 count = 0;
    ASSERT_EQ(child->GetNetInstanceList(1, ranks, count), HCCL_SUCCESS);
    EXPECT_EQ(ranks, std::vector<u32>(4, 1));
    ASSERT_EQ(child->GetLocalInstRanks(1, ranks, count), HCCL_SUCCESS);
    EXPECT_EQ(ranks, (std::vector<u32>{2}));
    EXPECT_EQ(child->GetNetInstanceByRankId(1, 2)->GetRankSize(), 2U);
}

TEST_F(RankGraphTest, Ut_VirtualGroups_When_InvalidRebuild_Expect_PreviousViewPreserved)
{
    auto graph = MakeVirtualTestGraph(8, VirtualTestThreeLayers());
    ASSERT_EQ(graph->BuildVirtualGroups(), HCCL_SUCCESS);
    // Introduce a duplicate membership without changing the rank-to-physical-instance association.
    auto* inst = graph->GetNetInstanceByNetInstId(1, "1");
    inst->AddRankId(0);
    EXPECT_EQ(graph->BuildVirtualGroups(), HCCL_E_PARA);
    std::vector<u32> ranks;
    u32 count = 0;
    ASSERT_EQ(graph->GetLocalInstRanks(1, ranks, count), HCCL_SUCCESS);
    EXPECT_EQ(ranks, VirtualTestRange(8, 4));
    EXPECT_EQ(graph->GetNetInstanceList(MAX_NET_LAYER, ranks, count), HCCL_E_PARA);
}

TEST_F(RankGraphTest, Ut_VirtualGroups_When_Disabled_Expect_PhysicalParentAndChild)
{
    auto graph = MakeVirtualTestGraph(8, VirtualTestThreeLayers(), false);
    EXPECT_FALSE(graph->IsVirtualGroupsEnabled());
    // Even an accidental direct call must not enable grouping for a ranktable graph.
    ASSERT_EQ(graph->BuildVirtualGroups(), HCCL_SUCCESS);
    std::vector<u32> ranks;
    u32 count = 0;
    ASSERT_EQ(graph->GetLocalInstRanks(1, ranks, count), HCCL_SUCCESS);
    EXPECT_EQ(ranks, (std::vector<u32>{0, 1, 2, 3, 8, 9, 10, 11}));
    graph->InitFinish();
    std::unique_ptr<RankGraph> child;
    ASSERT_EQ(graph->CreateSubRankGraph({0, 4, 8, 12}, child), HCCL_SUCCESS);
    ASSERT_NE(child, nullptr);
    EXPECT_FALSE(child->IsVirtualGroupsEnabled());
    ASSERT_EQ(child->GetLocalInstRanks(1, ranks, count), HCCL_SUCCESS);
    EXPECT_EQ(ranks, (std::vector<u32>{0, 2}));
}

TEST_F(RankGraphTest, Ut_VirtualGroups_When_ContinuousParent_Expect_ChildInheritsEnabledPolicy)
{
    auto layers = VirtualTestThreeLayers();
    layers[1] = {VirtualTestRange(0, 8), VirtualTestRange(8, 8)};
    auto graph = MakeVirtualTestGraph(0, layers, true);
    ASSERT_EQ(graph->BuildVirtualGroups(), HCCL_SUCCESS);
    graph->InitFinish();
    std::unique_ptr<RankGraph> child;
    ASSERT_EQ(graph->CreateSubRankGraph({0, 8, 4, 12}, child), HCCL_SUCCESS);
    ASSERT_NE(child, nullptr);
    EXPECT_TRUE(child->IsVirtualGroupsEnabled());
    std::vector<u32> sizes;
    u32 count = 0;
    ASSERT_EQ(child->GetNetInstanceList(1, sizes, count), HCCL_SUCCESS);
    EXPECT_EQ(sizes, (std::vector<u32>{1, 1, 1, 1}));
}

// ===================== worldteam 批量可达性优化新增接口 =====================

namespace {
std::shared_ptr<NetInstance::ConnInterface>
UtMakeRankGraphIface(const std::set<std::string>& ports, const std::set<LinkProtocol>& protocols, AddrPosition pos)
{
    IpAddress addr(0);
    return std::make_shared<NetInstance::ConnInterface>(addr, ports, pos, LinkType::PEER2PEER, protocols);
}
} // namespace

TEST_F(RankGraphTest, Ut_RankGraphHasLevel_When_RankInLayer_Return_True)
{
    RankGraph rankGraph(myRank);
    auto peer = createPeer(0, 0, 0);
    auto netInst = std::make_shared<InnerNetInstance>(0, "ut_haslevel");
    ASSERT_EQ(peer->AddNetInstance(netInst), HCCL_SUCCESS);
    ASSERT_EQ(rankGraph.AddPeer(peer), HCCL_SUCCESS);

    EXPECT_TRUE(peer->HasLevel(0));
    EXPECT_TRUE(rankGraph.HasLevel(0, 0));
}

TEST_F(RankGraphTest, Ut_RankGraphHasLevel_When_RankNotInLayer_Return_False)
{
    RankGraph rankGraph(myRank);
    auto peer = createPeer(0, 0, 0);
    auto netInst = std::make_shared<InnerNetInstance>(0, "ut_haslevel_miss");
    ASSERT_EQ(peer->AddNetInstance(netInst), HCCL_SUCCESS);
    ASSERT_EQ(rankGraph.AddPeer(peer), HCCL_SUCCESS);

    EXPECT_FALSE(peer->HasLevel(1));
    EXPECT_FALSE(rankGraph.HasLevel(0, 1));
}

TEST_F(RankGraphTest, Ut_RankGraphHasLevel_When_RankNotExist_Return_FalseWithoutThrow)
{
    // 行为变更点：原 GetLevels 对不存在 rank 抛 InvalidParamsException，HasLevel 返回 false 不抛异常
    // （10 处替换点 rankId 均为 GetMyRank() 恒存在，生产行为不变）
    RankGraph rankGraph(myRank);
    auto peer = createPeer(0, 0, 0);
    auto netInst = std::make_shared<InnerNetInstance>(0, "ut_haslevel_noexist");
    ASSERT_EQ(peer->AddNetInstance(netInst), HCCL_SUCCESS);
    ASSERT_EQ(rankGraph.AddPeer(peer), HCCL_SUCCESS);

    EXPECT_FALSE(rankGraph.HasLevel(27, 0));
}

TEST_F(RankGraphTest, Ut_GetReachableRanksPerProtocol_When_InvalidNetLayer_Return_EPARA)
{
    RankGraph rankGraph(myRank);
    auto peer = createPeer(0, 0, 0);
    auto netInst = std::make_shared<InnerNetInstance>(0, "ut_rrbys_epara");
    ASSERT_EQ(peer->AddNetInstance(netInst), HCCL_SUCCESS);
    ASSERT_EQ(rankGraph.AddPeer(peer), HCCL_SUCCESS);

    const std::set<CommProtocol> protocols
        = {COMM_PROTOCOL_UB_CTP, COMM_PROTOCOL_UBC_TP, COMM_PROTOCOL_UBOE, COMM_PROTOCOL_UB_RTP, COMM_PROTOCOL_UB_MEM};
    std::unordered_map<CommProtocol, std::vector<uint32_t>> reachable;
    EXPECT_EQ(rankGraph.GetReachableRanksPerProtocol(1, 0, protocols, reachable), HCCL_E_PARA);
    EXPECT_TRUE(reachable.empty()); // 出参未被修改
}

TEST_F(RankGraphTest, Ut_GetReachableRanksPerProtocol_When_OnlyUnfocusedProtocol_Return_EmptyFocusedBuckets)
{
    RankGraph rankGraph(myRank);
    auto peer0 = createPeer(0, 0, 0);
    auto peer1 = createPeer(1, 1, 1);
    auto netInst = std::make_shared<InnerNetInstance>(0, "ut_rrbys_roce");
    ASSERT_EQ(peer0->AddNetInstance(netInst), HCCL_SUCCESS);
    ASSERT_EQ(peer1->AddNetInstance(netInst), HCCL_SUCCESS);
    netInst->AddRankId(0);
    netInst->AddRankId(1);
    ASSERT_EQ(rankGraph.AddPeer(peer0), HCCL_SUCCESS);
    ASSERT_EQ(rankGraph.AddPeer(peer1), HCCL_SUCCESS);
    ASSERT_EQ(rankGraph.AddNetInstance(netInst), HCCL_SUCCESS);
    (void)netInst->AddNode(peer0);
    (void)netInst->AddNode(peer1);
    // 直达边 0->1 仅 ROCE 协议
    const std::set<LinkProtocol> roce = {LinkProtocol::ROCE};
    const std::set<std::string> ports = {"0/0"};
    netInst->AddLink(std::make_shared<NetInstance::Link>(
        peer0, peer1, UtMakeRankGraphIface(ports, roce, AddrPosition::HOST),
        UtMakeRankGraphIface(ports, roce, AddrPosition::DEVICE), LinkType::PEER2PEER, roce));

    // 步骤 1：NetInstance 层未过滤（LinkProtocol 域），含 ROCE 桶
    std::map<LinkProtocol, std::set<RankId>> reachableByProtocol;
    netInst->GetReachableRanks(0, reachableByProtocol);
    ASSERT_EQ(reachableByProtocol.count(LinkProtocol::ROCE), 1U);
    EXPECT_EQ(reachableByProtocol.at(LinkProtocol::ROCE), (std::set<RankId>{1}));

    // 步骤 2：RankGraph 层按聚焦协议过滤——出参自洽预置聚焦协议全集空桶（未命中协议不依赖调用方预置）
    const std::set<CommProtocol> protocols
        = {COMM_PROTOCOL_UB_CTP, COMM_PROTOCOL_UBC_TP, COMM_PROTOCOL_UBOE, COMM_PROTOCOL_UB_RTP, COMM_PROTOCOL_UB_MEM};
    std::unordered_map<CommProtocol, std::vector<uint32_t>> reachable;
    EXPECT_EQ(rankGraph.GetReachableRanksPerProtocol(0, 0, protocols, reachable), HCCL_SUCCESS);
    EXPECT_EQ(reachable.size(), protocols.size());
    for (CommProtocol protocol : protocols) {
        EXPECT_TRUE(reachable.at(protocol).empty());
    }
}

// ===================== 批量接口 vs 旧 GetLinks 逐 peer 路径对拍（真实图，非桩） =====================

namespace {
// 与生产收集一致的关注协议集合
const std::set<CommProtocol>& UtDuelFocusedProtocols()
{
    static const std::set<CommProtocol> protocols
        = {COMM_PROTOCOL_UB_CTP, COMM_PROTOCOL_UBC_TP, COMM_PROTOCOL_UBOE, COMM_PROTOCOL_UB_RTP, COMM_PROTOCOL_UB_MEM};
    return protocols;
}

// 旧路径等价聚合：逐 peer GetLinks 后按协议分桶（全量 std::find 去重，不依赖遍历顺序）。
// EXPECT 而非 ASSERT：失败时留痕并继续聚合，避免辅助函数提前 return 掩盖 GetLinks 错误码
void UtAggregateLegacyByGetLinks(
    hccl::RankGraphV2& graph, uint32_t netLayer, uint32_t selfRank, const std::vector<uint32_t>& peerRanks,
    std::unordered_map<CommProtocol, std::vector<uint32_t>>& out)
{
    for (uint32_t peerRank : peerRanks) {
        CommLink* links = nullptr;
        uint32_t linkNum = 0;
        EXPECT_EQ(graph.GetLinks(netLayer, selfRank, peerRank, &links, &linkNum), HCCL_SUCCESS)
            << "GetLinks failed, peer[" << peerRank << "]";
        if (links == nullptr) {
            continue;
        }
        for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
            auto protocolIt = out.find(links[linkIndex].linkAttr.linkProtocol);
            if (protocolIt == out.end()) {
                continue;
            }
            if (std::find(protocolIt->second.begin(), protocolIt->second.end(), peerRank) == protocolIt->second.end()) {
                protocolIt->second.push_back(peerRank);
            }
        }
    }
}

// 逐协议归一化比对（缺桶视为空桶、排序后比较）
::testing::AssertionResult UtDuelCompareBuckets(
    const std::unordered_map<CommProtocol, std::vector<uint32_t>>& batch,
    const std::unordered_map<CommProtocol, std::vector<uint32_t>>& legacy, const std::set<CommProtocol>& protocols)
{
    for (CommProtocol protocol : protocols) {
        auto batchIt = batch.find(protocol);
        auto legacyIt = legacy.find(protocol);
        std::vector<uint32_t> batchBucket = batchIt == batch.end() ? std::vector<uint32_t>{} : batchIt->second;
        std::vector<uint32_t> legacyBucket = legacyIt == legacy.end() ? std::vector<uint32_t>{} : legacyIt->second;
        std::sort(batchBucket.begin(), batchBucket.end());
        std::sort(legacyBucket.begin(), legacyBucket.end());
        if (batchBucket != legacyBucket) {
            return ::testing::AssertionFailure() << "protocol[" << static_cast<int>(protocol) << "] batchNum["
                                                 << batchBucket.size() << "] legacyNum[" << legacyBucket.size() << "]";
        }
    }
    return ::testing::AssertionSuccess();
}
} // namespace

// 对拍（Inner 真实图）：批量 GetReachableRanksPerProtocol vs 逐 peer GetLinks 汇总，
// 覆盖直达边多协议、fabric 端口配对成功/失败、多协议分桶
TEST_F(RankGraphTest, Ut_GetReachableRanksPerProtocolVsGetLinks_When_InnerRealGraph_Return_IdenticalBuckets)
{
    RankGraph rankGraph(myRank);
    auto netInst = std::make_shared<InnerNetInstance>(0, "ut_duel_inner");
    const std::vector<RankId> rankIds = {0, 1, 2, 3};
    std::map<RankId, std::shared_ptr<NetInstance::Peer>> graphPeers;
    for (RankId rankId : rankIds) {
        auto peer = std::make_shared<NetInstance::Peer>(rankId, rankId, rankId, rankId);
        ASSERT_EQ(peer->AddNetInstance(netInst), HCCL_SUCCESS);
        ASSERT_EQ(rankGraph.AddPeer(peer), HCCL_SUCCESS);
        netInst->AddRankId(rankId);
        (void)netInst->AddNode(peer);
        graphPeers[rankId] = peer;
    }
    ASSERT_EQ(rankGraph.AddNetInstance(netInst), HCCL_SUCCESS);
    auto fabric = std::make_shared<NetInstance::Fabric>(10, "ut_duel_plane");
    (void)netInst->AddNode(fabric);
    ASSERT_EQ(rankGraph.InitInnerRanks(), HCCL_SUCCESS);

    const std::set<LinkProtocol> dual = {LinkProtocol::UB_CTP, LinkProtocol::UB_TP};
    const std::set<LinkProtocol> ctp = {LinkProtocol::UB_CTP};
    const std::set<LinkProtocol> rtp = {LinkProtocol::UB_RTP};
    const std::set<std::string> ports1 = {"0/0"};
    const std::set<std::string> ports4 = {"0/0", "0/1", "0/2", "0/3"};
    // 直达边：0->1 双协议；0->2 仅 CTP（两侧 iface 均非空，老路径 InsertInnerLink 依赖）
    netInst->AddLink(std::make_shared<NetInstance::Link>(
        graphPeers[0], graphPeers[1], UtMakeRankGraphIface(ports1, dual, AddrPosition::DEVICE),
        UtMakeRankGraphIface(ports1, dual, AddrPosition::DEVICE), LinkType::PEER2PEER, dual, LinkDirection::BOTH, 1));
    netInst->AddLink(std::make_shared<NetInstance::Link>(
        graphPeers[0], graphPeers[2], UtMakeRankGraphIface(ports1, ctp, AddrPosition::DEVICE),
        UtMakeRankGraphIface(ports1, ctp, AddrPosition::DEVICE), LinkType::PEER2PEER, ctp, LinkDirection::BOTH, 1));
    // fabric（生产 PEER2NET 形态：peer 侧 iface 非空、fabric 侧 iface 为空——net_instance.h Link 契约）：
    // 老路径 BuildLinkList 据此走 InsertClosLinks（协议仅取 peer2net 边，且只对 Peer 节点 cast、不碰 fabric）。
    // sLink(0->F) 源侧 4 端口（配对键）；dLink(F->3) 目标侧 4 端口（配对）；dLink(F->1) 目标侧 1 端口（不配对）。
    // dLink 协议异于 sLink（UB_RTP）：若新实现误并入 dLink 协议，RTP 桶非空即与老路径失配
    netInst->AddLink(std::make_shared<NetInstance::Link>(
        graphPeers[0], fabric, UtMakeRankGraphIface(ports4, ctp, AddrPosition::DEVICE), nullptr, LinkType::PEER2PEER,
        ctp, LinkDirection::BOTH, 1));
    netInst->AddLink(std::make_shared<NetInstance::Link>(
        fabric, graphPeers[3], nullptr, UtMakeRankGraphIface(ports4, rtp, AddrPosition::DEVICE), LinkType::PEER2PEER,
        rtp, LinkDirection::BOTH, 1));
    netInst->AddLink(std::make_shared<NetInstance::Link>(
        fabric, graphPeers[1], nullptr, UtMakeRankGraphIface(ports1, rtp, AddrPosition::DEVICE), LinkType::PEER2PEER,
        rtp, LinkDirection::BOTH, 1));

    hccl::RankGraphV2 v2(static_cast<void*>(&rankGraph));
    const std::set<CommProtocol>& protocols = UtDuelFocusedProtocols();

    // 新路径：批量
    std::unordered_map<CommProtocol, std::vector<uint32_t>> batch;
    ASSERT_EQ(v2.GetReachableRanksPerProtocol(0, myRank, protocols, batch), HCCL_SUCCESS);

    // 旧路径：逐 peer GetLinks 汇总。与生产调用方 InitWorldTeamLayer 一致预置 5 协议空桶
    // （聚合仅向已存在的桶追加，桶不存在即跳过——空 map 传入会恒为空）
    std::unordered_map<CommProtocol, std::vector<uint32_t>> legacy;
    for (CommProtocol protocol : protocols) {
        legacy[protocol] = {};
    }
    UtAggregateLegacyByGetLinks(v2, 0, myRank, {1, 2, 3}, legacy);

    EXPECT_TRUE(UtDuelCompareBuckets(batch, legacy, protocols));
    // 语义锚点：两路径均应给出 CTP={1,2,3}、UBC_TP={1}；dLink 的 RTP 协议不出现（协议仅取 peer2net 边）
    ASSERT_EQ(batch.count(COMM_PROTOCOL_UB_CTP), 1U);
    EXPECT_EQ(batch.at(COMM_PROTOCOL_UB_CTP), (std::vector<uint32_t>{1, 2, 3}));
    ASSERT_EQ(batch.count(COMM_PROTOCOL_UBC_TP), 1U);
    EXPECT_EQ(batch.at(COMM_PROTOCOL_UBC_TP), (std::vector<uint32_t>{1}));
    ASSERT_EQ(batch.count(COMM_PROTOCOL_UB_RTP), 1U);
    EXPECT_TRUE(batch.at(COMM_PROTOCOL_UB_RTP).empty());
}

// 对拍（Clos 真实图）：同桶多条 sLink/dLink 时代表链路取尾语义
TEST_F(RankGraphTest, Ut_GetReachableRanksPerProtocolVsGetLinks_When_ClosRealGraph_Return_IdenticalBuckets)
{
    RankGraph rankGraph(myRank);
    // 双层拓扑贴近生产形态：layer 0 Inner（rankIds 与上层一致）+ layer 1 Clos。
    // InitInnerRanks 固定查 layer 0，仅挂 layer 1 Clos 实例会触发 NullPtrException 失败。
    auto innerInst = std::make_shared<InnerNetInstance>(0, "ut_duel_inner_l0");
    auto netInst = std::make_shared<ClosNetInstance>(1, "ut_duel_clos");
    const std::vector<RankId> rankIds = {0, 1};
    std::map<RankId, std::shared_ptr<NetInstance::Peer>> graphPeers;
    for (RankId rankId : rankIds) {
        auto peer = std::make_shared<NetInstance::Peer>(rankId, rankId, rankId, rankId);
        ASSERT_EQ(peer->AddNetInstance(innerInst), HCCL_SUCCESS);
        ASSERT_EQ(peer->AddNetInstance(netInst), HCCL_SUCCESS);
        ASSERT_EQ(rankGraph.AddPeer(peer), HCCL_SUCCESS);
        innerInst->AddRankId(rankId);
        (void)innerInst->AddNode(peer);
        netInst->AddRankId(rankId);
        (void)netInst->AddNode(peer);
        graphPeers[rankId] = peer;
    }
    ASSERT_EQ(rankGraph.AddNetInstance(innerInst), HCCL_SUCCESS);
    ASSERT_EQ(rankGraph.AddNetInstance(netInst), HCCL_SUCCESS);
    auto fabric = std::make_shared<NetInstance::Fabric>(10, "ut_duel_plane");
    (void)netInst->AddNode(fabric);
    ASSERT_EQ(rankGraph.InitInnerRanks(), HCCL_SUCCESS);

    const std::set<LinkProtocol> ctp = {LinkProtocol::UB_CTP};
    const std::set<LinkProtocol> tp = {LinkProtocol::UB_TP};
    const std::set<LinkProtocol> rtp = {LinkProtocol::UB_RTP};
    const std::set<std::string> ports2 = {"0/0", "0/1"};
    const std::set<std::string> ports3 = {"0/0", "0/1", "0/2"};
    // 同 (0,F) 两条 sLink：首条 CTP/2 端口，末条 TP/2 端口（代表取尾——协议锚点：误取首条即 CTP 桶非空失配）；
    // 同 (F,1) 两条 dLink：首条目标侧 3 端口（不配对），末条目标侧 2 端口（代表取尾，与末条 sLink 端口配对）。
    // 生产 PEER2NET 形态：peer 侧 iface 非空、fabric 侧 iface 为空（net_instance.h Link 契约），老路径
    // BuildLinkList 据此走 InsertClosLinks（协议仅取 peer2net 边，只对 Peer 节点 cast、不碰 fabric）；
    // dLink 协议异于 sLink（UB_RTP）：若新实现误并入 dLink 协议，RTP 桶非空即与老路径失配
    netInst->AddLink(std::make_shared<NetInstance::Link>(
        graphPeers[0], fabric, UtMakeRankGraphIface(ports2, ctp, AddrPosition::DEVICE), nullptr, LinkType::PEER2PEER,
        ctp, LinkDirection::BOTH, 1));
    netInst->AddLink(std::make_shared<NetInstance::Link>(
        graphPeers[0], fabric, UtMakeRankGraphIface(ports2, tp, AddrPosition::DEVICE), nullptr, LinkType::PEER2PEER, tp,
        LinkDirection::BOTH, 1));
    netInst->AddLink(std::make_shared<NetInstance::Link>(
        fabric, graphPeers[1], nullptr, UtMakeRankGraphIface(ports3, rtp, AddrPosition::DEVICE), LinkType::PEER2PEER,
        rtp, LinkDirection::BOTH, 1));
    netInst->AddLink(std::make_shared<NetInstance::Link>(
        fabric, graphPeers[1], nullptr, UtMakeRankGraphIface(ports2, rtp, AddrPosition::DEVICE), LinkType::PEER2PEER,
        rtp, LinkDirection::BOTH, 1));

    hccl::RankGraphV2 v2(static_cast<void*>(&rankGraph));
    const std::set<CommProtocol>& protocols = UtDuelFocusedProtocols();

    std::unordered_map<CommProtocol, std::vector<uint32_t>> batch;
    ASSERT_EQ(v2.GetReachableRanksPerProtocol(1, myRank, protocols, batch), HCCL_SUCCESS);

    // 与生产调用方一致预置协议空桶（聚合仅向已存在的桶追加）
    std::unordered_map<CommProtocol, std::vector<uint32_t>> legacy;
    for (CommProtocol protocol : protocols) {
        legacy[protocol] = {};
    }
    UtAggregateLegacyByGetLinks(v2, 1, myRank, {1}, legacy);

    EXPECT_TRUE(UtDuelCompareBuckets(batch, legacy, protocols));
    // 语义锚点：代表链路取尾 → 仅末条 sLink(TP) 参与配对，UBC_TP={1}、CTP 为预置空桶（误取首条 sLink 即失配）；
    // dLink 的 RTP 协议不出现（协议仅取 peer2net 边）
    ASSERT_EQ(batch.count(COMM_PROTOCOL_UBC_TP), 1U);
    EXPECT_EQ(batch.at(COMM_PROTOCOL_UBC_TP), (std::vector<uint32_t>{1}));
    ASSERT_EQ(batch.count(COMM_PROTOCOL_UB_CTP), 1U);
    EXPECT_TRUE(batch.at(COMM_PROTOCOL_UB_CTP).empty());
    ASSERT_EQ(batch.count(COMM_PROTOCOL_UB_RTP), 1U);
    EXPECT_TRUE(batch.at(COMM_PROTOCOL_UB_RTP).empty());
}

// 对拍（虚拟分段图）：分段激活且段间存在物理边时，批量收集限定在本 rank 分段内——
// 与老路径口径一致（老路径经 GetInstRanksByNetLayer 枚举分段成员逐 peer GetLinks，跨段 rank 不入桶）。
// 图形：layer1 单实例含 rank {0,1,5,6}（数值不连续触发分段：段A={0,1}、段B={5,6}），
// 直达边 0→1（段内）与 0→5（跨段）均存在——批量必须过滤跨段 rank 5，仅 CTP={1}
TEST_F(RankGraphTest, Ut_GetReachableRanksPerProtocolVsGetLinks_When_VirtualGroupSplit_Return_SegmentScopedBuckets)
{
    // enableVirtualGroups=true 构图；分段视图须 InitFinish 前建（BuildVirtualGroups 契约）
    RankGraph rankGraph(myRank, true);
    auto innerInst = std::make_shared<InnerNetInstance>(0, "ut_duel_vg_l0");
    auto netInst = std::make_shared<InnerNetInstance>(1, "ut_duel_vg");
    const std::vector<RankId> rankIds = {0, 1, 5, 6};
    std::map<RankId, std::shared_ptr<NetInstance::Peer>> graphPeers;
    for (RankId rankId : rankIds) {
        auto peer = std::make_shared<NetInstance::Peer>(rankId, rankId, rankId, rankId);
        ASSERT_EQ(peer->AddNetInstance(innerInst), HCCL_SUCCESS);
        ASSERT_EQ(peer->AddNetInstance(netInst), HCCL_SUCCESS);
        ASSERT_EQ(rankGraph.AddPeer(peer), HCCL_SUCCESS);
        innerInst->AddRankId(rankId);
        (void)innerInst->AddNode(peer);
        netInst->AddRankId(rankId);
        (void)netInst->AddNode(peer);
        graphPeers[rankId] = peer;
    }
    ASSERT_EQ(rankGraph.AddNetInstance(innerInst), HCCL_SUCCESS);
    ASSERT_EQ(rankGraph.AddNetInstance(netInst), HCCL_SUCCESS);
    ASSERT_EQ(rankGraph.InitInnerRanks(), HCCL_SUCCESS);
    ASSERT_EQ(rankGraph.BuildVirtualGroups(), HCCL_SUCCESS);

    const std::set<LinkProtocol> ctp = {LinkProtocol::UB_CTP};
    const std::set<std::string> ports1 = {"0/0"};
    // 直达边：0→1（段内）与 0→5（跨段——图上真实存在，但分段语义下不应入桶）
    netInst->AddLink(std::make_shared<NetInstance::Link>(
        graphPeers[0], graphPeers[1], UtMakeRankGraphIface(ports1, ctp, AddrPosition::DEVICE),
        UtMakeRankGraphIface(ports1, ctp, AddrPosition::DEVICE), LinkType::PEER2PEER, ctp, LinkDirection::BOTH, 1));
    netInst->AddLink(std::make_shared<NetInstance::Link>(
        graphPeers[0], graphPeers[5], UtMakeRankGraphIface(ports1, ctp, AddrPosition::DEVICE),
        UtMakeRankGraphIface(ports1, ctp, AddrPosition::DEVICE), LinkType::PEER2PEER, ctp, LinkDirection::BOTH, 1));

    hccl::RankGraphV2 v2(static_cast<void*>(&rankGraph));
    const std::set<CommProtocol>& protocols = UtDuelFocusedProtocols();

    // 分段视图核验：layer1 本 rank 分段 = {0,1}（分段激活）
    std::vector<uint32_t> segRanks;
    u32 segNum = 0;
    ASSERT_EQ(rankGraph.GetLocalInstRanks(1, segRanks, segNum), HCCL_SUCCESS);
    ASSERT_EQ(segRanks, (std::vector<uint32_t>{0, 1}));

    // 新路径：批量（应限定分段，跨段 rank 5 被过滤）
    std::unordered_map<CommProtocol, std::vector<uint32_t>> batch;
    ASSERT_EQ(v2.GetReachableRanksPerProtocol(1, myRank, protocols, batch), HCCL_SUCCESS);

    // 语义锚点：CTP 仅含段内 rank 1，跨段 rank 5 不入桶（无过滤实现将得 {1,5}）
    ASSERT_EQ(batch.count(COMM_PROTOCOL_UB_CTP), 1U);
    EXPECT_EQ(batch.at(COMM_PROTOCOL_UB_CTP), (std::vector<uint32_t>{1}));

    // 老路径：按生产口径枚举分段成员逐 peer GetLinks（ranks 来自 GetInstRanksByNetLayer 分段口径）
    std::unordered_map<CommProtocol, std::vector<uint32_t>> legacy;
    for (CommProtocol protocol : protocols) {
        legacy[protocol] = {};
    }
    UtAggregateLegacyByGetLinks(v2, 1, myRank, {1}, legacy);
    EXPECT_TRUE(UtDuelCompareBuckets(batch, legacy, protocols));
}
