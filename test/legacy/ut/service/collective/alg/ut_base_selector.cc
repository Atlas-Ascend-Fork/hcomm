/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include "base_selector.h"
#include "rank_gph.h"

using namespace Hccl;

namespace {
class TestBaseSelector : public BaseSelector {
public:
    using BaseSelector::ExtractNetLayerDetails;
    using BaseSelector::ExtractTopoDetails;
    using BaseSelector::TopoInfo;

    SelectorStatus Select(const CollAlgOperator&, CollAlgParams&, std::string&) override
    {
        return SelectorStatus::NOT_MATCH;
    }
};

class BaseSelectorTest : public testing::Test {
protected:
    void SetUp() override
    {
        for (RankId rank = 0; rank < 4; ++rank) {
            rankGraph_.AddPeer(std::make_shared<NetInstance::Peer>(rank, rank, rank, rank));
        }
        selector_.SetVirtualTopo(&rankGraph_).SetMyRank(0).SetRankSize(4);
    }

    void TearDown() override { GlobalMockObject::verify(); }

    void AddNetInstance(u32 layer, const std::string& name, const std::vector<RankId>& ranks)
    {
        auto instance = std::make_shared<ClosNetInstance>(layer, name);
        for (auto rank : ranks) {
            instance->AddRankId(rank);
            instance->UpdateTopoInst(0, TopoType::CLOS, rank);
            rankGraph_.GetPeer(rank)->AddNetInstance(instance);
        }
        rankGraph_.AddNetInstance(instance);
    }

    void CheckTwoLayerDetails(u32 topLayer)
    {
        AddNetInstance(0, "local", {0, 1});
        AddNetInstance(0, "remote", {2, 3});
        AddNetInstance(topLayer, "all", {0, 1, 2, 3});

        TestBaseSelector::TopoInfo topoInfo{};
        ASSERT_EQ(selector_.ExtractNetLayerDetails(topoInfo), HCCL_SUCCESS);
        EXPECT_EQ(topoInfo.levelNum, 2);
        const auto& details = topoInfo.netLayerDetails;
        EXPECT_EQ(details.netLayerNum, 2);
        EXPECT_EQ(details.netLayers, (std::set<u32>{0, topLayer}));
        ASSERT_EQ(details.netInstNumOfLayer.size(), topLayer + 1);
        ASSERT_EQ(details.instSizeListOfLayer.size(), topLayer + 1);
        ASSERT_EQ(details.localNetInsSizeOfLayer.size(), topLayer + 1);
        EXPECT_EQ(details.netInstNumOfLayer[0], 2);
        EXPECT_EQ(details.netInstNumOfLayer[topLayer], 1);
        EXPECT_EQ(details.instSizeListOfLayer[0], (std::vector<u32>{2, 2}));
        EXPECT_EQ(details.instSizeListOfLayer[topLayer], (std::vector<u32>{4}));
        EXPECT_EQ(details.localNetInsSizeOfLayer[0], 2);
        EXPECT_EQ(details.localNetInsSizeOfLayer[topLayer], 4);

        ASSERT_EQ(selector_.ExtractTopoDetails(topoInfo), HCCL_SUCCESS);
        ASSERT_EQ(topoInfo.topoInstDetailsOfLayer.size(), topLayer + 1);
        const auto& localTopo = topoInfo.topoInstDetailsOfLayer[0];
        EXPECT_EQ(localTopo.topoInstNum, 1);
        EXPECT_EQ(localTopo.sizeOfTopo, (std::vector<u32>{2}));
        EXPECT_EQ(localTopo.ranksInTopo, (std::vector<std::vector<u32>>{{0, 1}}));
        const auto& topTopo = topoInfo.topoInstDetailsOfLayer[topLayer];
        EXPECT_EQ(topTopo.topoInstNum, 1);
        EXPECT_EQ(topTopo.sizeOfTopo, (std::vector<u32>{4}));
        EXPECT_EQ(topTopo.typeOfTopo, (std::vector<TopoType>{TopoType::CLOS}));
        EXPECT_EQ(topTopo.ranksInTopo, (std::vector<std::vector<u32>>{{0, 1, 2, 3}}));
        for (u32 layer = 1; layer < topLayer; ++layer) {
            EXPECT_EQ(details.netInstNumOfLayer[layer], 0);
            EXPECT_TRUE(topoInfo.topoInstDetailsOfLayer[layer].ranksInTopo.empty());
        }
    }

    RankGraph rankGraph_{0, true};
    TestBaseSelector selector_;
};

TEST_F(BaseSelectorTest, ExtractContinuousLayers) { CheckTwoLayerDetails(1); }

TEST_F(BaseSelectorTest, ExtractSparseLayers) { CheckTwoLayerDetails(3); }

TEST_F(BaseSelectorTest, ExtractHighestSupportedLayer) { CheckTwoLayerDetails(7); }

TEST_F(BaseSelectorTest, SkipTopoInstancesOutsideLocalVirtualGroup)
{
    AddNetInstance(0, "local", {0, 1});
    AddNetInstance(0, "remote", {2, 3});
    AddNetInstance(1, "even", {0, 2});
    AddNetInstance(1, "odd", {1, 3});
    AddNetInstance(2, "all", {0, 1, 2, 3});
    auto* instance = rankGraph_.GetNetInstanceByRankId(1, 0);
    ASSERT_NE(instance, nullptr);
    instance->UpdateTopoInst(1, TopoType::MESH_1D, 2);
    instance->UpdateTopoInst(2, TopoType::CLOS, 0);

    TestBaseSelector::TopoInfo topoInfo{};
    ASSERT_EQ(selector_.ExtractNetLayerDetails(topoInfo), HCCL_SUCCESS);
    ASSERT_EQ(selector_.ExtractTopoDetails(topoInfo), HCCL_SUCCESS);
    ASSERT_EQ(topoInfo.topoInstDetailsOfLayer[1].topoInstNum, 3U);
    const auto physicalTopoTypes = topoInfo.topoInstDetailsOfLayer[1].typeOfTopo;
    EXPECT_EQ(topoInfo.topoInstDetailsOfLayer[1].rankNumForTopoType.at(TopoType::MESH_1D), (std::vector<u32>{1}));

    ASSERT_EQ(rankGraph_.BuildVirtualGroups(), HCCL_SUCCESS);
    std::vector<u32> ranks;
    u32 count = 0;
    ASSERT_EQ(rankGraph_.GetRanksByTopoInst(1, 1, ranks, count), HCCL_E_NOT_FOUND);
    // Reuse populated output to verify both filtering and repeated extraction discard stale entries.
    for (u32 attempt = 0; attempt < 2; ++attempt) {
        ASSERT_EQ(selector_.ExtractTopoDetails(topoInfo), HCCL_SUCCESS);
        const auto& details = topoInfo.topoInstDetailsOfLayer[1];
        EXPECT_EQ(details.topoInstNum, 3U);
        EXPECT_EQ(details.typeOfTopo, physicalTopoTypes);
        ASSERT_EQ(details.sizeOfTopo.size(), 3U);
        ASSERT_EQ(details.ranksInTopo.size(), 3U);
        for (size_t i = 0; i < physicalTopoTypes.size(); ++i) {
            const bool outsideLocalGroup = physicalTopoTypes[i] == TopoType::MESH_1D;
            EXPECT_EQ(details.sizeOfTopo[i], outsideLocalGroup ? 0U : 1U);
            EXPECT_EQ(details.ranksInTopo[i], outsideLocalGroup ? std::vector<u32>{} : std::vector<u32>{0});
        }
        ASSERT_EQ(details.rankNumForTopoType.size(), 1U);
        EXPECT_EQ(details.rankNumForTopoType.at(TopoType::CLOS), (std::vector<u32>{1, 1}));
        EXPECT_EQ(topoInfo.topoInstDetailsOfLayer[0].ranksInTopo, (std::vector<std::vector<u32>>{{0, 1}}));
        EXPECT_EQ(topoInfo.topoInstDetailsOfLayer[2].ranksInTopo, (std::vector<std::vector<u32>>{{0, 1, 2, 3}}));
    }
}

TEST_F(BaseSelectorTest, PropagateTopoRanksQueryErrors)
{
    AddNetInstance(0, "all", {0, 1, 2, 3});
    TestBaseSelector::TopoInfo topoInfo{};
    ASSERT_EQ(selector_.ExtractNetLayerDetails(topoInfo), HCCL_SUCCESS);
    MOCKER_CPP(&RankGraph::GetRanksByTopoInst).stubs().will(returnValue(HCCL_E_PARA));
    EXPECT_EQ(selector_.ExtractTopoDetails(topoInfo), HCCL_E_PARA);
}

TEST_F(BaseSelectorTest, StopCountingAtFirstLayerCoveringAllRanks)
{
    AddNetInstance(0, "local", {0, 1, 2, 3});
    AddNetInstance(3, "all", {0, 1, 2, 3});
    TestBaseSelector::TopoInfo topoInfo{};

    ASSERT_EQ(selector_.ExtractNetLayerDetails(topoInfo), HCCL_SUCCESS);
    EXPECT_EQ(topoInfo.levelNum, 1);
    EXPECT_EQ(topoInfo.netLayerDetails.netLayerNum, 2);
    ASSERT_EQ(topoInfo.netLayerDetails.localNetInsSizeOfLayer.size(), 4);
    EXPECT_EQ(topoInfo.netLayerDetails.localNetInsSizeOfLayer[3], 4);
}

TEST_F(BaseSelectorTest, RejectLayersWithoutInstanceCoveringAllRanks)
{
    AddNetInstance(0, "local", {0, 1});
    AddNetInstance(0, "remote", {2, 3});
    AddNetInstance(3, "local", {0, 1});
    AddNetInstance(3, "remote", {2, 3});
    TestBaseSelector::TopoInfo topoInfo{};

    EXPECT_EQ(selector_.ExtractNetLayerDetails(topoInfo), HCCL_E_INTERNAL);
    EXPECT_EQ(topoInfo.levelNum, 0);
}

TEST_F(BaseSelectorTest, RejectEmptyLayers)
{
    TestBaseSelector::TopoInfo topoInfo{};
    EXPECT_EQ(selector_.ExtractNetLayerDetails(topoInfo), HCCL_E_INTERNAL);
    EXPECT_EQ(selector_.ExtractTopoDetails(topoInfo), HCCL_E_INTERNAL);
}

TEST_F(BaseSelectorTest, RejectNullRankGraph)
{
    selector_.SetVirtualTopo(nullptr);
    TestBaseSelector::TopoInfo topoInfo{};
    EXPECT_EQ(selector_.ExtractNetLayerDetails(topoInfo), HCCL_E_PTR);
    EXPECT_EQ(selector_.ExtractTopoDetails(topoInfo), HCCL_E_PTR);
}

TEST_F(BaseSelectorTest, RejectMismatchedRankSize)
{
    AddNetInstance(0, "all", {0, 1, 2, 3});
    selector_.SetRankSize(3);
    TestBaseSelector::TopoInfo topoInfo{};
    EXPECT_EQ(selector_.ExtractNetLayerDetails(topoInfo), HCCL_E_PARA);
}
} // namespace
