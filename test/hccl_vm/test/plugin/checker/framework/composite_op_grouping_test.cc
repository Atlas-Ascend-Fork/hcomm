/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "composite_op_grouping.h"
#include "runtime_state/db_sim_communicator.h"

namespace {

std::map<uint64_t, std::pair<std::string, uint64_t>> g_commIdentities;

sim::operation::CompositeOpDetail
MakeOp(uint32_t id, uint32_t deviceId, uint32_t rankId, uint64_t commId, uint32_t opIter = 0)
{
    sim::operation::CompositeOpDetail op{};
    op.deviceId = deviceId;
    op.rankId = rankId;
    op.commId = commId;
    op.detail.id = id;
    op.detail.deviceId = deviceId;
    op.detail.rankId = rankId;
    op.detail.commId = commId;
    op.detail.opIter = opIter;
    return op;
}

} // namespace

namespace sim::runtime {

bool GetCommunicatorIdentity(uint64_t commId, std::string& commName, uint64_t& commHash)
{
    const auto iter = g_commIdentities.find(commId);
    if (iter == g_commIdentities.end()) {
        commName.clear();
        commHash = 0;
        return false;
    }
    commName = iter->second.first;
    commHash = iter->second.second;
    return true;
}

} // namespace sim::runtime

TEST(CompositeOpGroupingTest, SeparatesInterleavedCommunicatorDomains)
{
    g_commIdentities = {{11, {"comm_a", 0}}, {12, {"comm_a", 0}}, {21, {"comm_b", 0}}, {22, {"comm_b", 0}}};

    // 两个独立通信域允许在不同 rank 上以不同顺序提交。
    std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>> compositeData{
        {0, {MakeOp(1, 10, 0, 11), MakeOp(2, 10, 0, 21)}}, {1, {MakeOp(3, 20, 1, 22), MakeOp(4, 20, 1, 12)}}};

    std::vector<HcclSim::CompositeOpGroup> groups;
    ASSERT_EQ(HcclSim::GroupCompositeOpDetails(compositeData, groups), HCCL_SUCCESS);
    ASSERT_EQ(groups.size(), 2U);

    ASSERT_EQ(groups[0].size(), 2U);
    EXPECT_EQ(groups[0].at(10).commId, 11U);
    EXPECT_EQ(groups[0].at(20).commId, 12U);

    ASSERT_EQ(groups[1].size(), 2U);
    EXPECT_EQ(groups[1].at(10).commId, 21U);
    EXPECT_EQ(groups[1].at(20).commId, 22U);
}

TEST(CompositeOpGroupingTest, UsesCommNameHashAndOpIterAsOperatorIdentity)
{
    g_commIdentities = {{11, {"comm_a", 0}}, {12, {"comm_a", 0}}, {21, {"comm_b", 0}}};

    std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>> compositeData{
        {0, {MakeOp(1, 10, 0, 11, 0), MakeOp(2, 10, 0, 11, 1)}},
        {1, {MakeOp(3, 20, 1, 12, 0), MakeOp(4, 20, 1, 12, 1), MakeOp(5, 30, 1, 21, 0)}}};

    std::vector<HcclSim::CompositeOpGroup> groups;
    ASSERT_EQ(HcclSim::GroupCompositeOpDetails(compositeData, groups), HCCL_SUCCESS);
    ASSERT_EQ(groups.size(), 3U);

    EXPECT_EQ(groups[0].at(10).detail.opIter, 0U);
    EXPECT_EQ(groups[0].at(20).detail.opIter, 0U);
    EXPECT_EQ(groups[1].at(10).detail.opIter, 1U);
    EXPECT_EQ(groups[1].at(20).detail.opIter, 1U);
    EXPECT_EQ(groups[2].at(30).commId, 21U);
    EXPECT_EQ(groups[2].at(30).detail.opIter, 0U);
}

TEST(CompositeOpGroupingTest, SeparatesSameNameCommunicatorsWithDifferentHashes)
{
    g_commIdentities
        = {{11, {"sub_comm", 100}}, {12, {"sub_comm", 100}}, {21, {"sub_comm", 200}}, {22, {"sub_comm", 200}}};

    std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>> compositeData{
        {0, {MakeOp(1, 10, 0, 11, 0), MakeOp(2, 30, 0, 21, 0)}},
        {1, {MakeOp(3, 20, 1, 12, 0), MakeOp(4, 40, 1, 22, 0)}}};

    std::vector<HcclSim::CompositeOpGroup> groups;
    ASSERT_EQ(HcclSim::GroupCompositeOpDetails(compositeData, groups), HCCL_SUCCESS);
    ASSERT_EQ(groups.size(), 2U);

    EXPECT_EQ(groups[0].at(10).commId, 11U);
    EXPECT_EQ(groups[0].at(20).commId, 12U);
    EXPECT_EQ(groups[1].at(30).commId, 21U);
    EXPECT_EQ(groups[1].at(40).commId, 22U);
}
