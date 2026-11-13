/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "rank_gph.h"
#include "rank_graph_interface.h"

using namespace Hccl;

namespace {
constexpr RankId MY_RANK = 0;
constexpr RankId REMOTE_RANK = 1;
constexpr u32 INNER_LAYER = 0;
constexpr u32 OUTER_LAYER = 1;

std::shared_ptr<NetInstance::Peer> CreatePeer(RankId rankId, LocalId localId, DeviceId deviceId)
{
    return std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
}
} // namespace

class RankGraphGetNetLayersTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankGraphGetNetLayersTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "RankGraphGetNetLayersTest TearDown" << std::endl; }
    virtual void SetUp() {}
    virtual void TearDown() { GlobalMockObject::verify(); }
};

// 场景：非对称组网，本rank(MY_RANK)只挂 INNER_LAYER，远端 rank 挂 OUTER_LAYER，
//       全通信域存在两层非空 NetInstance，但本rank只属于其中一层
// 预期：netLayerNum 等于本rank的层集合大小（1），而不是全图非空层数（2）；
//       netLayers 仅返回本rank所在的 INNER_LAYER，调用方按 netLayerNum 遍历不越界
TEST_F(RankGraphGetNetLayersTest, Ut_GetNetLayers_When_AsymmetricTopology_Expect_CountEqualsMyRankLevelSize)
{
    RankGraph rankGraph(MY_RANK);
    auto myPeer = CreatePeer(MY_RANK, 0, 0);
    auto remotePeer = CreatePeer(REMOTE_RANK, 1, 1);
    (void)rankGraph.AddPeer(myPeer);
    (void)rankGraph.AddPeer(remotePeer);

    auto innerInst = std::make_shared<InnerNetInstance>(INNER_LAYER, "inner");
    auto outerInst = std::make_shared<InnerNetInstance>(OUTER_LAYER, "outer");
    innerInst->AddRankId(MY_RANK);
    outerInst->AddRankId(REMOTE_RANK);
    (void)myPeer->AddNetInstance(innerInst);
    (void)remotePeer->AddNetInstance(outerInst);
    (void)rankGraph.AddNetInstance(innerInst);
    (void)rankGraph.AddNetInstance(outerInst);

    // 前置校验：非对称前提成立，全图非空层数(2)大于本rank层数(1)
    EXPECT_EQ(rankGraph.GetLevelNum(), 2u);
    EXPECT_EQ(rankGraph.GetLevels(MY_RANK).size(), 1u);

    IRankGraph iRankGraph(static_cast<void*>(&rankGraph));
    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum = 0;
    HcclResult ret = iRankGraph.GetNetLayers(&netLayers, &netLayerNum);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(netLayerNum, 1u);
    ASSERT_NE(netLayers, nullptr);
    EXPECT_EQ(netLayers[0], INNER_LAYER);
}

// 场景：对称组网，本rank与远端 rank 同属 INNER_LAYER 和 OUTER_LAYER 两层
// 预期：netLayerNum 等于本rank层集合大小（2），netLayers=[0,1]
TEST_F(RankGraphGetNetLayersTest, Ut_GetNetLayers_When_SymmetricTwoLayers_Expect_CountEqualsMyRankLevelSize)
{
    RankGraph rankGraph(MY_RANK);
    auto myPeer = CreatePeer(MY_RANK, 0, 0);
    auto remotePeer = CreatePeer(REMOTE_RANK, 1, 1);
    (void)rankGraph.AddPeer(myPeer);
    (void)rankGraph.AddPeer(remotePeer);

    auto innerInst = std::make_shared<InnerNetInstance>(INNER_LAYER, "inner");
    auto outerInst = std::make_shared<InnerNetInstance>(OUTER_LAYER, "outer");
    innerInst->AddRankId(MY_RANK);
    innerInst->AddRankId(REMOTE_RANK);
    outerInst->AddRankId(MY_RANK);
    outerInst->AddRankId(REMOTE_RANK);
    (void)myPeer->AddNetInstance(innerInst);
    (void)myPeer->AddNetInstance(outerInst);
    (void)remotePeer->AddNetInstance(innerInst);
    (void)remotePeer->AddNetInstance(outerInst);
    (void)rankGraph.AddNetInstance(innerInst);
    (void)rankGraph.AddNetInstance(outerInst);

    // 前置校验：对称场景下全图非空层数与本rank层数一致
    EXPECT_EQ(rankGraph.GetLevelNum(), 2u);
    EXPECT_EQ(rankGraph.GetLevels(MY_RANK).size(), 2u);

    IRankGraph iRankGraph(static_cast<void*>(&rankGraph));
    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum = 0;
    HcclResult ret = iRankGraph.GetNetLayers(&netLayers, &netLayerNum);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(netLayerNum, 2u);
    ASSERT_NE(netLayers, nullptr);
    EXPECT_EQ(netLayers[0], INNER_LAYER);
    EXPECT_EQ(netLayers[1], OUTER_LAYER);
}
