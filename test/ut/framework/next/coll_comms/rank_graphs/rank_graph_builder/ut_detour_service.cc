/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <set>
#include <unordered_map>
#include <vector>

#include "detour_rules.h"
#include "detour_service.h"
#include "net_instance.h"
#include "phy_topo_builder.h"
#include "rank_graph_test_data_builder.h"

namespace Hccl {
void SetDetourTable2P(
    const std::set<u32>& tableIdSet,
    std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>>& detourTable);
}

using namespace Hccl;

class DetourServiceTest : public testing::Test {};

TEST_F(DetourServiceTest, Ut_SetDetourTable4P_When_MatchedTableIds_Expect_SelectExpectedTable)
{
    std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>> detourTable;

    SetDetourTable4P({0, 1, 2, 3}, detourTable);
    EXPECT_EQ(GetDetour4PTable0123(), detourTable);

    SetDetourTable4P({4, 5, 6, 7}, detourTable);
    EXPECT_EQ(GetDetour4PTable4567(), detourTable);

    SetDetourTable4P({0, 2, 4, 6}, detourTable);
    EXPECT_EQ(GetDetour4PTable0246(), detourTable);

    SetDetourTable4P({1, 3, 5, 7}, detourTable);
    EXPECT_EQ(GetDetour4PTable1357(), detourTable);
}

TEST_F(DetourServiceTest, Ut_SetDetourTable2P_When_MatchedTableIds_Expect_SelectExpectedTable)
{
    std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>> detourTable;

    SetDetourTable2P({0, 1}, detourTable);
    EXPECT_EQ(GetDetour2PTable01(), detourTable);

    SetDetourTable2P({0, 4}, detourTable);
    EXPECT_EQ(GetDetour2PTable04(), detourTable);
}

namespace {
DetourData
MakeDetourData(const std::shared_ptr<NetInstance::Peer>& srcPeer, const std::shared_ptr<NetInstance::Peer>& dstPeer)
{
    DetourData data;
    data.srcPhyPeerId = hccl::PhyTopo::Peer::GetId(0);
    data.detourPhyPeerId = hccl::PhyTopo::Peer::GetId(8);
    data.dstPhyPeerId = hccl::PhyTopo::Peer::GetId(1);
    data.srcNetInstPeer = srcPeer;
    data.dstNetInstPeer = dstPeer;
    return data;
}
} // namespace

class DetourLinkPortTest : public testing::Test {
protected:
    void SetUp() override { hccl::PhyTopo::GetInstance()->Clear(); }

    void TearDown() override { hccl::PhyTopo::GetInstance()->Clear(); }

    void RecoverDetourTopo()
    {
        const auto topoInfo = test::MakeTopoInfo(
            {0, 1, 8}, {
                           test::MakeEdge(0, LinkType::PEER2PEER, 0, {"0/0"}, 8, {"1/0"}),
                           test::MakeEdge(0, LinkType::PEER2PEER, 8, {"1/1"}, 1, {"0/1"}),
                       });
        ASSERT_EQ(hccl::PhyTopoBuilder::GetInstance().RecoverBuild(topoInfo), HCCL_SUCCESS);
    }
};

TEST_F(DetourLinkPortTest, Ut_AddDetourLink_When_SrcPortMissing_Expect_E_Para)
{
    RecoverDetourTopo();
    auto srcPeer = std::make_shared<NetInstance::Peer>(RankId(0), LocalId(0), LocalId(0), DeviceId(0));
    auto dstPeer = std::make_shared<NetInstance::Peer>(RankId(1), LocalId(1), LocalId(1), DeviceId(1));
    dstPeer->SetPortPortAddrMapLayer0({{"0/1", {IpAddress("10.0.0.2")}}});
    InnerNetInstance netInst(0, "az0-rack0");

    EXPECT_EQ(
        AddDetourLink(
            &netInst, MakeDetourData(srcPeer, dstPeer), hccl::PhyTopo::GetInstance()->GetTopoGraph(0), nullptr),
        HCCL_E_PARA);
    EXPECT_TRUE(srcPeer->GetPortAddrMapLayer0().empty());
}

TEST_F(DetourLinkPortTest, Ut_AddDetourLink_When_SrcPortAddrEmpty_Expect_E_Para)
{
    RecoverDetourTopo();
    auto srcPeer = std::make_shared<NetInstance::Peer>(RankId(0), LocalId(0), LocalId(0), DeviceId(0));
    auto dstPeer = std::make_shared<NetInstance::Peer>(RankId(1), LocalId(1), LocalId(1), DeviceId(1));
    srcPeer->SetPortPortAddrMapLayer0({{"0/0", {}}});
    dstPeer->SetPortPortAddrMapLayer0({{"0/1", {IpAddress("10.0.0.2")}}});
    InnerNetInstance netInst(0, "az0-rack0");

    EXPECT_EQ(
        AddDetourLink(
            &netInst, MakeDetourData(srcPeer, dstPeer), hccl::PhyTopo::GetInstance()->GetTopoGraph(0), nullptr),
        HCCL_E_PARA);
    EXPECT_EQ(srcPeer->GetPortAddrMapLayer0().size(), 1U);
    EXPECT_TRUE(srcPeer->GetPortAddrMapLayer0().begin()->second.empty());
}

TEST_F(DetourLinkPortTest, Ut_AddDetourLink_When_DstPortMissing_Expect_E_Para)
{
    RecoverDetourTopo();
    auto srcPeer = std::make_shared<NetInstance::Peer>(RankId(0), LocalId(0), LocalId(0), DeviceId(0));
    auto dstPeer = std::make_shared<NetInstance::Peer>(RankId(1), LocalId(1), LocalId(1), DeviceId(1));
    srcPeer->SetPortPortAddrMapLayer0({{"0/0", {IpAddress("10.0.0.1")}}});
    InnerNetInstance netInst(0, "az0-rack0");

    EXPECT_EQ(
        AddDetourLink(
            &netInst, MakeDetourData(srcPeer, dstPeer), hccl::PhyTopo::GetInstance()->GetTopoGraph(0), nullptr),
        HCCL_E_PARA);
    EXPECT_TRUE(dstPeer->GetPortAddrMapLayer0().empty());
    EXPECT_EQ(srcPeer->GetPortAddrMapLayer0().size(), 1U);
}

TEST_F(DetourLinkPortTest, Ut_AddDetourLink_When_DstPortAddrEmpty_Expect_E_Para)
{
    RecoverDetourTopo();
    auto srcPeer = std::make_shared<NetInstance::Peer>(RankId(0), LocalId(0), LocalId(0), DeviceId(0));
    auto dstPeer = std::make_shared<NetInstance::Peer>(RankId(1), LocalId(1), LocalId(1), DeviceId(1));
    srcPeer->SetPortPortAddrMapLayer0({{"0/0", {IpAddress("10.0.0.1")}}});
    dstPeer->SetPortPortAddrMapLayer0({{"0/1", {}}});
    InnerNetInstance netInst(0, "az0-rack0");

    EXPECT_EQ(
        AddDetourLink(
            &netInst, MakeDetourData(srcPeer, dstPeer), hccl::PhyTopo::GetInstance()->GetTopoGraph(0), nullptr),
        HCCL_E_PARA);
    EXPECT_EQ(dstPeer->GetPortAddrMapLayer0().size(), 1U);
    EXPECT_TRUE(dstPeer->GetPortAddrMapLayer0().begin()->second.empty());
    EXPECT_EQ(srcPeer->GetPortAddrMapLayer0().size(), 1U);
}

TEST_F(DetourLinkPortTest, Ut_AddDetourLink_When_PeerNullptr_Expect_E_Ptr)
{
    RecoverDetourTopo();
    auto dstPeer = std::make_shared<NetInstance::Peer>(RankId(1), LocalId(1), LocalId(1), DeviceId(1));
    dstPeer->SetPortPortAddrMapLayer0({{"0/1", {IpAddress("10.0.0.2")}}});
    InnerNetInstance netInst(0, "az0-rack0");

    EXPECT_EQ(
        AddDetourLink(
            &netInst, MakeDetourData(nullptr, dstPeer), hccl::PhyTopo::GetInstance()->GetTopoGraph(0), nullptr),
        HCCL_E_PTR);
    EXPECT_EQ(
        AddDetourLink(
            &netInst, MakeDetourData(dstPeer, nullptr), hccl::PhyTopo::GetInstance()->GetTopoGraph(0), nullptr),
        HCCL_E_PTR);
}
