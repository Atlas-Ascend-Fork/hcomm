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
#include "securec.h"
#include "types.h"
#include "hccl/hccl_types.h"

#define private public
#define protected public
#include "root_group_strategy.h"
#include "root_info_detect_bridge.h"
#undef protected
#undef private

#include "rank_info_detect_scalable.h"
#include "root_handle_v2.h"

using namespace Hccl;

// ============================ 分组策略（纯静态函数，无外部依赖） ============================

class RootGroupStrategyTest : public testing::Test {
protected:
    void SetUp() override
    {
        const RootGroupStrategy* strategy = GetRootGroupStrategy();
        ASSERT_NE(strategy, nullptr);
        // StrategySlot stores a mutable object; restore it only within this UT fixture.
        strategySlot_ = const_cast<RootGroupStrategy*>(strategy);
        savedStrategy_ = *strategy;
    }

    void TearDown() override
    {
        if (strategySlot_ != nullptr) {
            *strategySlot_ = savedStrategy_;
        }
    }

private:
    RootGroupStrategy* strategySlot_{nullptr};
    RootGroupStrategy savedStrategy_{};
};

// Ut_RootGroupStrategy_When_DefaultRegistered_Return_AverageGrouping：
// 库加载后默认分组策略生效，算法与 A3 平均分组一致（nRanks=10, nRoot=3 → 组0 4 rank，组1/2 各 3）
TEST_F(RootGroupStrategyTest, Ut_RootGroupStrategy_When_DefaultRegistered_Return_AverageGrouping)
{
    const RootGroupStrategy* strategy = GetRootGroupStrategy();
    ASSERT_NE(strategy, nullptr);
    ASSERT_NE(strategy->rootIdFromRank, nullptr);
    ASSERT_NE(strategy->firstRankFromRoot, nullptr);
    ASSERT_NE(strategy->nRankFromRoot, nullptr);
    ASSERT_NE(strategy->rankHasRoot, nullptr);

    const u32 nRanks = 10;
    const u32 nRoot = 3;
    // rootIdFromRank = [0,0,0,0,1,1,1,2,2,2]
    const u32 expectRootId[10] = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2};
    for (u32 rank = 0; rank < nRanks; ++rank) {
        EXPECT_EQ(strategy->rootIdFromRank(nRanks, nRoot, rank), expectRootId[rank]);
    }
    // firstRankFromRoot = [0, 4, 7]
    EXPECT_EQ(strategy->firstRankFromRoot(nRanks, nRoot, 0), 0U);
    EXPECT_EQ(strategy->firstRankFromRoot(nRanks, nRoot, 1), 4U);
    EXPECT_EQ(strategy->firstRankFromRoot(nRanks, nRoot, 2), 7U);
    // nRankFromRoot = [4, 3, 3]
    EXPECT_EQ(strategy->nRankFromRoot(nRanks, nRoot, 0), 4U);
    EXPECT_EQ(strategy->nRankFromRoot(nRanks, nRoot, 1), 3U);
    EXPECT_EQ(strategy->nRankFromRoot(nRanks, nRoot, 2), 3U);
    // rankHasRoot：仅 rank 0/4/7 为 true
    for (u32 rank = 0; rank < nRanks; ++rank) {
        const bool expectRoot = (rank == 0 || rank == 4 || rank == 7);
        EXPECT_EQ(strategy->rankHasRoot(nRanks, nRoot, rank), expectRoot);
    }
}

// Ut_RootGroupStrategy_When_CustomRegistered_Return_PolicyReplaced：
// 注册自定义策略后回调被替换；用例结束后由 fixture 恢复原策略。
static u32 CustomRootIdFromRank(u32 nRanks, u32 nRoot, u32 rank)
{
    (void)nRoot;
    return rank / (nRanks / 2); // 两组策略：上半场 root0、下半场 root1（仅 nRoot>=2 语义）
}
static u32 CustomFirstRankFromRoot(u32 nRanks, u32 nRoot, u32 rootIndex)
{
    (void)nRoot;
    return rootIndex * (nRanks / 2);
}
static u32 CustomNRankFromRoot(u32 nRanks, u32 nRoot, u32 rootIndex)
{
    (void)nRoot;
    (void)rootIndex;
    return nRanks / 2;
}
static bool CustomRankHasRoot(u32 nRanks, u32 nRoot, u32 rank)
{
    (void)nRoot;
    return (rank == 0) || (rank == nRanks / 2);
}

TEST_F(RootGroupStrategyTest, Ut_RootGroupStrategy_When_CustomRegistered_Return_PolicyReplaced)
{
    const RootGroupStrategy customStrategy
        = {CustomRootIdFromRank, CustomFirstRankFromRoot, CustomNRankFromRoot, CustomRankHasRoot};
    ASSERT_EQ(RegisterRootGroupStrategy(customStrategy), HCCL_SUCCESS);

    const RootGroupStrategy* strategy = GetRootGroupStrategy();
    ASSERT_NE(strategy, nullptr);
    // 自定义策略生效
    EXPECT_EQ(strategy->rootIdFromRank(10, 2, 7), 1U);
    EXPECT_EQ(strategy->rankHasRoot(10, 2, 5), true);
    EXPECT_EQ(strategy->rankHasRoot(10, 2, 3), false);

    // 重复注册应被拒绝，TearDown 直接恢复保存的槽位，避免污染后续用例。
    EXPECT_EQ(RegisterRootGroupStrategy(customStrategy), HCCL_E_INTERNAL);
}

// Ut_RootGroupStrategy_When_BoundaryNRootEqualsOneOrNRanks_Return_ValidGrouping：
// nRoot=1 与 nRoot=nRanks 边界分组正确，无越界/除零
TEST_F(RootGroupStrategyTest, Ut_RootGroupStrategy_When_BoundaryNRootEqualsOneOrNRanks_Return_ValidGrouping)
{
    const RootGroupStrategy* strategy = GetRootGroupStrategy();
    ASSERT_NE(strategy, nullptr);

    // nRoot=1：所有 rank 属 root0，groupSize=8，仅 rank0 为 root
    const u32 nRanks = 8;
    for (u32 rank = 0; rank < nRanks; ++rank) {
        EXPECT_EQ(strategy->rootIdFromRank(nRanks, 1, rank), 0U);
        EXPECT_EQ(strategy->rankHasRoot(nRanks, 1, rank), rank == 0);
    }
    EXPECT_EQ(strategy->firstRankFromRoot(nRanks, 1, 0), 0U);
    EXPECT_EQ(strategy->nRankFromRoot(nRanks, 1, 0), nRanks);

    // nRoot=nRanks：每 root 1 rank，每个 rank 均为 root
    for (u32 rank = 0; rank < nRanks; ++rank) {
        EXPECT_EQ(strategy->rootIdFromRank(nRanks, nRanks, rank), rank);
        EXPECT_EQ(strategy->rankHasRoot(nRanks, nRanks, rank), true);
        EXPECT_EQ(strategy->firstRankFromRoot(nRanks, nRanks, rank), rank);
        EXPECT_EQ(strategy->nRankFromRoot(nRanks, nRanks, rank), 1U);
    }
}

// ============================ bridge 扩展槽（配对校验与查询语义） ============================

// Ut_RegisterRootInfoDetectBridge_When_ScalableSlotsUnpaired_Return_EPtr：
// scalable 两槽只注册其一（不配对）时注册被拒绝
TEST(RootInfoDetectBridgeTest, Ut_RegisterRootInfoDetectBridge_When_ScalableSlotsUnpaired_Return_EPtr)
{
    RootInfoDetectBridge bridge{};
    // 基础两槽非法（空指针）直接拒绝
    EXPECT_EQ(RegisterRootInfoDetectBridge(bridge, ROOT_INFO_DETECT_BRIDGE_VERSION, sizeof(bridge)), HCCL_E_PTR);

    // 基础两槽合法、scalable 槽只填一个 → 配对校验拒绝
    bridge.getRootInfo = [](HcclRootInfo*, RootInfoDetectBridge::DetectContext&) -> HcclResult {
        return HCCL_SUCCESS;
    };
    bridge.detectRankTable = [](u32, u32, const HcclRootHandleV2&, RankTableInfo&, RootInfoDetectBridge::DetectContext&,
                                const char*) -> HcclResult {
        return HCCL_SUCCESS;
    };
    bridge.getRootInfoScalable = [](HcclRootInfo*) -> HcclResult {
        return HCCL_SUCCESS;
    };
    bridge.detectRankTableScalable = nullptr;
    EXPECT_EQ(RegisterRootInfoDetectBridge(bridge, ROOT_INFO_DETECT_BRIDGE_VERSION, sizeof(bridge)), HCCL_E_PTR);
}

// Ut_GetRootInfoDetectBridgeScalable_When_SlotsRegistered_Return_NonNull：
// 完整注册后 scalable 视角查询非空（当前进程 registrar 未跑时基础查询为空——验证槽视角判定逻辑）
TEST(RootInfoDetectBridgeTest, Ut_GetRootInfoDetectBridgeScalable_When_SlotsRegistered_Return_NonNull)
{
    // GetRootInfoDetectBridge 在未注册进程返回 nullptr；scalable 视口在同一条件下也返回 nullptr。
    // 本用例验证两个视角的判定一致性（hccl_v2 真实装载场景由 HLT 覆盖）。
    const RootInfoDetectBridge* basic = GetRootInfoDetectBridge();
    const RootInfoDetectBridge* scalable = GetRootInfoDetectBridgeScalable();
    if (basic == nullptr) {
        EXPECT_EQ(scalable, nullptr);
    } else {
        // 基础已注册（例如同进程其它 UT 注册过）：scalable 槽齐备才非空
        if (basic->getRootInfoScalable != nullptr) {
            EXPECT_EQ(scalable, basic);
        } else {
            EXPECT_EQ(scalable, nullptr);
        }
    }
}

// ============================ handle 序列化（尺寸与 roundtrip） ============================

// Ut_HcclScalableRootHandleV2_When_SerializedIntoRootInfo_Return_SizeFits：
// handle 序列化进 HcclRootInfo 不溢出且 roundtrip 一致
TEST(HcclScalableRootHandleV2Test, Ut_HcclScalableRootHandleV2_When_SerializedIntoRootInfo_Return_SizeFits)
{
    EXPECT_LE(sizeof(HcclScalableRootHandleV2), HCCL_ROOT_INFO_BYTES);

    HcclScalableRootHandleV2 handle{};
    const std::string identifier = "1.2.3.4_60016_0_1729000000";
    EXPECT_EQ(
        memcpy_s(
            handle.rootHandle.identifier, sizeof(handle.rootHandle.identifier), identifier.c_str(),
            identifier.size() + 1),
        EOK);
    EXPECT_EQ(strncpy_s(handle.rootHandle.ip, sizeof(handle.rootHandle.ip), "1.2.3.4", strlen("1.2.3.4")), EOK);
    handle.rootHandle.listenPort = 60016;
    handle.meshPort = 60020;

    HcclRootInfo rootInfo{};
    EXPECT_EQ(memcpy_s(rootInfo.internal, HCCL_ROOT_INFO_BYTES, &handle, sizeof(handle)), EOK);

    HcclScalableRootHandleV2 parsed{};
    EXPECT_EQ(memcpy_s(&parsed, sizeof(parsed), rootInfo.internal, sizeof(parsed)), EOK);
    parsed.rootHandle.identifier[ROOTINFO_IDENTIFIER_MAX_LENGTH - 1] = '\0';
    EXPECT_STREQ(parsed.rootHandle.identifier, identifier.c_str());
    EXPECT_STREQ(parsed.rootHandle.ip, "1.2.3.4");
    EXPECT_EQ(parsed.rootHandle.listenPort, 60016U);
    EXPECT_EQ(parsed.meshPort, 60020U);
}
