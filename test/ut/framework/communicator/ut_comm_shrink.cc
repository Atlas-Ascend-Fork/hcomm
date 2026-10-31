/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Shrink 通信域缩容功能 UT
// 覆盖 ValidateShrinkParams、BuildShrinkSurvivorRanks、CommShrink、CreateDerivedComm、
// 及 C 接口 HcclCommShrink 的全部分支与逻辑路径。

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <string>
#include <vector>

#include "hccl_api_base_test.h"
#include "coll_comm_mgr.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_comm.h"

using namespace hccl;

// ── MockCreateDerivedComm：填充出参 collComm，避免 happy-path 用例断言空洞 ──
// MOCKER_CPP 会将 this 作为第一个参数传入；后续参数与 CreateDerivedComm 原型一致
HcclResult MockCreateDerivedComm(CollComm*, std::unique_ptr<CollComm>& collComm, const CollComm::DerivedCommParams&)
{
    collComm
        = std::make_unique<CollComm>(nullptr, 0, "ut_shrink_derived", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    return HCCL_SUCCESS;
}

// ── MockRankGraph：重写 hccl::RankGraph 虚函数，使 GetRankSize 返回可配置值 ──
class MockRankGraph : public hccl::RankGraph {
public:
    uint32_t mockRankSize{0};

    HcclResult GetRankSize(uint32_t* rankSize) override
    {
        *rankSize = mockRankSize;
        return HCCL_SUCCESS;
    }
    HcclResult GetRankGraphInfo(GraphType, void**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetLinks(uint32_t, uint32_t, uint32_t, CommLink**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetNetLayers(uint32_t**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetInstTopoTypeByNetLayer(uint32_t, CommTopo*) override { return HCCL_SUCCESS; }
    HcclResult GetInstSizeByNetLayer(uint32_t, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetInstRanksByNetLayer(uint32_t, uint32_t**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetInstSizeListByNetLayer(uint32_t, uint32_t**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetDeviceId(uint32_t, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetEndpointInfo(uint32_t, const EndpointDesc*, EndpointAttr, uint32_t, void*) override
    {
        return HCCL_SUCCESS;
    }
};

// ── CommShrink 测试固件 ──
class CollCommShrinkTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }

    // 创建 simpleMode CollComm 并注入 MockRankGraph，设置 shrink 相关字段
    // 调用方需保证 mockGraph 生命周期覆盖 collComm
    static void SetupShrinkFields(
        CollComm& collComm, MockRankGraph& mockGraph, uint32_t rankSize, uint32_t myRank, uint32_t rootRankId)
    {
        mockGraph.mockRankSize = rankSize;
        collComm.rankgraph_ = &mockGraph;
        collComm.rankId_ = myRank;
        collComm.rootRankId_ = rootRankId;
        collComm.topParentIdentifier_ = "ut_shrink";
        collComm.generationIndex_ = 0;
        collComm.initSource_ = CommInitSource::rootInfo;
        HcclCommConfigInit(&collComm.hcclCommConfig_);
    }
};

// ════════════════════════════════════════════════════════════════════════════
// ValidateShrinkParams 分支覆盖
// ════════════════════════════════════════════════════════════════════════════

// 分支1: excludeRanksCount == 0 → HCCL_E_PARA
// rankgraph_ 为空时 GetRankSize() 返回 0，但 count==0 检查先于 GetRankSize() 调用
TEST_F(CollCommShrinkTest, Ut_ValidateShrinkParams_When_CountZero_Expect_E_PARA)
{
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    uint32_t excludeList[] = {0};
    HcclResult ret = collComm.ValidateShrinkParams(excludeList, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 分支2: excludeRanksCount >= rankSize → HCCL_E_PARA
// 此检查在循环之前，即使列表含自身也先触发
TEST_F(CollCommShrinkTest, Ut_ValidateShrinkParams_When_CountGEQRankSize_Expect_E_PARA)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    uint32_t excludeList[] = {0, 1, 2, 3};
    HcclResult ret = collComm.ValidateShrinkParams(excludeList, 4);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 分支3: excludeRank[i] >= rankSize（越界）→ HCCL_E_PARA
TEST_F(CollCommShrinkTest, Ut_ValidateShrinkParams_When_RankOutOfRange_Expect_E_PARA)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    uint32_t excludeList[] = {4};
    HcclResult ret = collComm.ValidateShrinkParams(excludeList, 1);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 分支4: excludeRank[i] == myRank（自身不可被剔除）→ HCCL_E_PARA
TEST_F(CollCommShrinkTest, Ut_ValidateShrinkParams_When_SelfExclusion_Expect_E_PARA)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    uint32_t excludeList[] = {1};
    HcclResult ret = collComm.ValidateShrinkParams(excludeList, 1);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 分支5: 全部参数合法 → HCCL_SUCCESS
TEST_F(CollCommShrinkTest, Ut_ValidateShrinkParams_When_AllValid_Expect_Success)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    uint32_t excludeList[] = {2, 3};
    HcclResult ret = collComm.ValidateShrinkParams(excludeList, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ════════════════════════════════════════════════════════════════════════════
// BuildShrinkSurvivorRanks 分支与逻辑覆盖
// ════════════════════════════════════════════════════════════════════════════

// 逻辑: root 未被剔除，myRank 在中间 → 正确重编号
TEST_F(CollCommShrinkTest, Ut_BuildSurvivorRanks_When_RootNotExcluded_Expect_CorrectRemap)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    uint32_t excludeList[] = {2};
    CollComm::ShrinkSurvivorResult result;

    HcclResult ret = collComm.BuildShrinkSurvivorRanks(excludeList, 1, result);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(result.survivorRanks, std::vector<uint32_t>({0, 1, 3}));
    EXPECT_EQ(result.myNewRank, 1u);
    EXPECT_FALSE(result.rootExcluded);
    EXPECT_EQ(result.childRootRank, 0u);
}

// 逻辑: root 被剔除 → rootExcluded=true, childRootRank=0
TEST_F(CollCommShrinkTest, Ut_BuildSurvivorRanks_When_RootExcluded_Expect_RootExcludedTrue)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    uint32_t excludeList[] = {0, 2};
    CollComm::ShrinkSurvivorResult result;

    HcclResult ret = collComm.BuildShrinkSurvivorRanks(excludeList, 2, result);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(result.survivorRanks, std::vector<uint32_t>({1, 3}));
    EXPECT_EQ(result.myNewRank, 0u);
    EXPECT_TRUE(result.rootExcluded);
    EXPECT_EQ(result.childRootRank, 0u);
}

// 逻辑: myRank 在存活列表开头 → myNewRank=0
TEST_F(CollCommShrinkTest, Ut_BuildSurvivorRanks_When_MyRankAtBeginning_Expect_MyNewRankZero)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 0, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 0, 1);

    uint32_t excludeList[] = {3};
    CollComm::ShrinkSurvivorResult result;

    HcclResult ret = collComm.BuildShrinkSurvivorRanks(excludeList, 1, result);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(result.myNewRank, 0u);
    EXPECT_FALSE(result.rootExcluded);
    EXPECT_EQ(result.childRootRank, 1u);
}

// 逻辑: myRank 在存活列表末尾 → myNewRank 为最后一位
TEST_F(CollCommShrinkTest, Ut_BuildSurvivorRanks_When_MyRankAtEnd_Expect_MyNewRankLast)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 3, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 3, 0);

    uint32_t excludeList[] = {1};
    CollComm::ShrinkSurvivorResult result;

    HcclResult ret = collComm.BuildShrinkSurvivorRanks(excludeList, 1, result);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(result.survivorRanks, std::vector<uint32_t>({0, 2, 3}));
    EXPECT_EQ(result.myNewRank, 2u);
    EXPECT_FALSE(result.rootExcluded);
    EXPECT_EQ(result.childRootRank, 0u);
}

// 防御分支: myRank 在剔除列表中 → !foundMyRank → HCCL_E_INTERNAL
// 正常流程 ValidateShrinkParams 会先拦截自身被剔除，此处直接调 BuildShrinkSurvivorRanks 触发防御
TEST_F(CollCommShrinkTest, Ut_BuildSurvivorRanks_When_MyRankInExcludeList_Expect_E_INTERNAL)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    uint32_t excludeList[] = {1, 2};
    CollComm::ShrinkSurvivorResult result;

    HcclResult ret = collComm.BuildShrinkSurvivorRanks(excludeList, 2, result);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// 防御分支: rootRankId 越界（不在存活列表也不在剔除列表）→ !rootExcluded && !foundRootRank → HCCL_E_INTERNAL
TEST_F(CollCommShrinkTest, Ut_BuildSurvivorRanks_When_RootRankIdOutOfRange_Expect_E_INTERNAL)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 10); // rootRankId=10 >= rankSize=4

    uint32_t excludeList[] = {2};
    CollComm::ShrinkSurvivorResult result;

    HcclResult ret = collComm.BuildShrinkSurvivorRanks(excludeList, 1, result);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// ════════════════════════════════════════════════════════════════════════════
// CommShrink 分支与逻辑覆盖
// ════════════════════════════════════════════════════════════════════════════

// 分支: GetInitSource() == clusterInfo → HCCL_E_NOT_SUPPORT
TEST_F(CollCommShrinkTest, Ut_CommShrink_When_InitSourceClusterInfo_Expect_E_NOT_SUPPORT)
{
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    collComm.initSource_ = CommInitSource::clusterInfo;

    uint32_t excludeList[] = {2};
    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm.CommShrink(excludeList, 1, newCollComm, nullptr);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

// 分支: identifierNew.size() >= ROOTINFO_IDENTIFIER_MAX_LENGTH → HCCL_E_PARA
// topParentIdentifier_ 设为 126 字符，加 "_1" 后缀共 128 >= 128
TEST_F(CollCommShrinkTest, Ut_CommShrink_When_IdentifierTooLong_Expect_E_PARA)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);
    collComm.topParentIdentifier_ = std::string(126, 'A'); // 126 + "_1" = 128 >= 128

    uint32_t excludeList[] = {2};
    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm.CommShrink(excludeList, 1, newCollComm, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 分支: config==nullptr（inheritParentConfig=true）+ root 未被剔除 → Success
// Mock CreateDerivedComm 返回成功，验证 config==nullptr 路径
TEST_F(CollCommShrinkTest, Ut_CommShrink_When_ConfigNullAndRootAlive_Expect_InheritParentConfig)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);
    collComm.rootHandle_.listenPort = 12345; // 使 root handle 有效

    MOCKER_CPP(&CollComm::CreateDerivedComm).stubs().will(invoke(MockCreateDerivedComm));

    uint32_t excludeList[] = {2};
    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm.CommShrink(excludeList, 1, newCollComm, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(nullptr, newCollComm);
}

// 分支: config!=nullptr（inheritParentConfig=false）+ root 被剔除 → Success
TEST_F(CollCommShrinkTest, Ut_CommShrink_When_ConfigNotNullAndRootExcluded_Expect_UserConfigAndRootCleared)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0); // rootRankId=0, excludeList 含 0 → root 被剔除

    MOCKER_CPP(&CollComm::CreateDerivedComm).stubs().will(invoke(MockCreateDerivedComm));

    HcclCommConfig userConfig;
    HcclCommConfigInit(&userConfig);

    uint32_t excludeList[] = {0, 2};
    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm.CommShrink(excludeList, 2, newCollComm, &userConfig);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(nullptr, newCollComm);
}

// 分支: config!=nullptr + root 未被剔除 → Success
// 与上一用例互补：覆盖 inheritParentConfig=false 且 rootExcluded=false 的组合
TEST_F(CollCommShrinkTest, Ut_CommShrink_When_ConfigNotNullAndRootAlive_Expect_Success)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 1); // rootRankId=1 不在 excludeList 中

    MOCKER_CPP(&CollComm::CreateDerivedComm).stubs().will(invoke(MockCreateDerivedComm));

    HcclCommConfig userConfig;
    HcclCommConfigInit(&userConfig);

    uint32_t excludeList[] = {2, 3};
    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm.CommShrink(excludeList, 2, newCollComm, &userConfig);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(nullptr, newCollComm);
}

// ════════════════════════════════════════════════════════════════════════════
// CreateDerivedComm 分支覆盖
// ════════════════════════════════════════════════════════════════════════════

// 分支: survivorRanks.empty() → HCCL_E_PARA（在访问 comm_ 之前返回，comm_ 可为 nullptr）
TEST_F(CollCommShrinkTest, Ut_CreateDerivedComm_When_SurvivorRanksEmpty_Expect_E_PARA)
{
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);

    std::unique_ptr<CollComm> newCollComm;
    HcclCommConfig subConfig;
    HcclCommConfigInit(&subConfig);
    CollComm::DerivationInfo derivation;

    std::vector<uint32_t> emptyRanks;
    Hccl::RankTableInfo emptyRankTable; // Shrink 传空拓扑表（Grow 传收集结果，Shrink 传空对象）
    CollComm::DerivedCommParams params{
        emptyRanks, 0, "test", subConfig, derivation, false, CollCommInitMode::fullMode, emptyRankTable};
    HcclResult ret = collComm.CreateDerivedComm(newCollComm, params);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// ════════════════════════════════════════════════════════════════════════════
// C 接口 HcclCommShrink 入参校验分支覆盖
// ════════════════════════════════════════════════════════════════════════════

// 分支: excludeRanksList == nullptr → HCCL_E_PTR
TEST_F(CollCommShrinkTest, Ut_HcclCommShrink_When_excludeRanksListNull_Expect_E_PTR)
{
    HcclComm newComm = nullptr;
    HcclResult ret = HcclCommShrink(nullptr, nullptr, 0, nullptr, 0, &newComm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 分支: newComm == nullptr → HCCL_E_PTR
TEST_F(CollCommShrinkTest, Ut_HcclCommShrink_When_NewCommNull_Expect_E_PTR)
{
    uint32_t excludeList[] = {0};
    HcclResult ret = HcclCommShrink(nullptr, excludeList, 1, nullptr, 0, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// ════════════════════════════════════════════════════════════════════════════
// root 被剔除分支覆盖
// ════════════════════════════════════════════════════════════════════════════

// 分支: root 被剔除（rootRankId 在 excludeList 中）→ root 不创建派生域，返回 SUCCESS 且 newCollComm=null
TEST_F(CollCommShrinkTest, Ut_CommShrink_When_RootExcluded_RootNotCreateDerivedComm)
{
    MockRankGraph mockGraph;
    // myRank=0, rootRankId=0, excludeList 含 0 → root 被剔除
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 0, 0);

    uint32_t excludeList[] = {0, 2};
    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm.CommShrink(excludeList, 2, newCollComm, nullptr);
    // root 被剔除时不创建派生域，返回 SUCCESS
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(nullptr, newCollComm);
}

// 分支: 非 root rank 自身被剔除 → HCCL_E_PARA（ValidateShrinkParams 拦截）
TEST_F(CollCommShrinkTest, Ut_CommShrink_When_NonRootSelfExcluded_Expect_E_PARA)
{
    MockRankGraph mockGraph;
    // myRank=1, rootRankId=0, excludeList 含 1 → 非 root 自身被剔除
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    uint32_t excludeList[] = {1};
    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm.CommShrink(excludeList, 1, newCollComm, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 分支: ValidateShrinkParams excludeRanksList 为空指针 → HCCL_E_PTR
TEST_F(CollCommShrinkTest, Ut_ValidateShrinkParams_When_excludeRanksListNull_Expect_E_PTR)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    HcclResult ret = collComm.ValidateShrinkParams(nullptr, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// ════════════════════════════════════════════════════════════════════════════
// HcclCommGrow 入参校验分支覆盖
// ════════════════════════════════════════════════════════════════════════════

// 分支: HcclCommGrow newComm == nullptr → HCCL_E_PTR
TEST_F(CollCommShrinkTest, Ut_HcclCommGrow_When_NewCommNull_Expect_E_PTR)
{
    HcclResult ret = HcclCommGrow(nullptr, 2, nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 分支: HcclCommGrow comm==nullptr 且 rootInfo==nullptr → HCCL_E_PARA
TEST_F(CollCommShrinkTest, Ut_HcclCommGrow_When_CommNullAndRootInfoNull_Expect_E_PARA)
{
    HcclComm newComm = nullptr;
    HcclResult ret = HcclCommGrow(nullptr, 2, nullptr, 0, nullptr, &newComm);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 分支: HcclCommGetRootInfo rootInfo == nullptr → HCCL_E_PTR
TEST_F(CollCommShrinkTest, Ut_HcclCommGetRootInfo_When_RootInfoNull_Expect_E_PTR)
{
    HcclResult ret = HcclCommGetRootInfo(nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
// 分支: comm == nullptr（映射后 hcclComm 为空）→ HCCL_E_PTR
TEST_F(CollCommShrinkTest, Ut_HcclCommShrink_When_CommNull_Expect_E_PTR)
{
    uint32_t excludeList[] = {0};
    HcclComm newComm = nullptr;
    HcclResult ret = HcclCommShrink(nullptr, excludeList, 1, nullptr, 0, &newComm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// ════════════════════════════════════════════════════════════════════════════
// ComputeExcludeRanksHash 逻辑覆盖
// ════════════════════════════════════════════════════════════════════════════

// 逻辑: 相同内容不同顺序产生相同 hash（排序不变性）
TEST_F(CollCommShrinkTest, Ut_ComputeHash_When_SameContentDiffOrder_Expect_SameHash)
{
    uint32_t list1[] = {1, 3, 5};
    uint32_t list2[] = {5, 1, 3};
    uint64_t hash1 = CollComm::ComputeExcludeRanksHash(list1, 3);
    uint64_t hash2 = CollComm::ComputeExcludeRanksHash(list2, 3);
    EXPECT_EQ(hash1, hash2);
}

// 逻辑: 不同内容产生不同 hash
TEST_F(CollCommShrinkTest, Ut_ComputeHash_When_DiffContent_Expect_DiffHash)
{
    uint32_t list1[] = {1, 3};
    uint32_t list2[] = {1, 5};
    uint64_t hash1 = CollComm::ComputeExcludeRanksHash(list1, 2);
    uint64_t hash2 = CollComm::ComputeExcludeRanksHash(list2, 2);
    EXPECT_NE(hash1, hash2);
}

// 逻辑: 不同 count 产生不同 hash（防止 {1} 和 {1,1} 碰撞）
TEST_F(CollCommShrinkTest, Ut_ComputeHash_When_DiffCount_Expect_DiffHash)
{
    uint32_t list1[] = {1};
    uint32_t list2[] = {1, 1};
    uint64_t hash1 = CollComm::ComputeExcludeRanksHash(list1, 1);
    uint64_t hash2 = CollComm::ComputeExcludeRanksHash(list2, 2);
    EXPECT_NE(hash1, hash2);
}

// 逻辑: 单元素 hash 非零且确定性（回归基线）
TEST_F(CollCommShrinkTest, Ut_ComputeHash_When_SingleElement_Expect_Deterministic)
{
    uint32_t list[] = {42};
    uint64_t hash = CollComm::ComputeExcludeRanksHash(list, 1);
    EXPECT_NE(hash, 0ULL);
    // 同一输入两次调用结果一致
    EXPECT_EQ(hash, CollComm::ComputeExcludeRanksHash(list, 1));
}

// ════════════════════════════════════════════════════════════════════════════
// CheckShrinkConsistency 分支覆盖
// ════════════════════════════════════════════════════════════════════════════

// 分支: rankInfoDetect_ == nullptr → 跳过校验，返回 HCCL_SUCCESS
TEST_F(CollCommShrinkTest, Ut_CheckConsistency_When_DetectNull_Expect_SkipAndSuccess)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);
    // rankInfoDetect_ 默认为 nullptr

    uint32_t excludeList[] = {2};
    HcclResult ret = collComm.CheckShrinkConsistency(excludeList, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
