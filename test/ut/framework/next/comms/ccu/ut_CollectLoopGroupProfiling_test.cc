/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#define private public
#define protected public
#include "ccu_kernel.h"
#include "ccu_rep_loopgroup_bundle_v1.h"
#include "ccu_rep_base_v1.h"
#include "ccu_rep_loadarg_v1.h"
#include "ccu_ins_generator_v1.h"
#undef private
#undef protected

using namespace hcomm;
using namespace hcomm::CcuRep;

namespace {
std::shared_ptr<CcuRepLoopGroupBundle> MakeLoopGroupBundle(uint16_t instrId)
{
    auto bundle = std::make_shared<CcuRepLoopGroupBundle>(nullptr, Variable(nullptr), Variable(nullptr));
    bundle->instrId = instrId;
    return bundle;
}
} // namespace

class CollectLoopGroupProfilingTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    void FillContainers(
        CcuKernel& kernel, size_t repN, size_t goN, size_t profN, const GroupInfo& infoTemplate = GroupInfo{})
    {
        auto& lgProfInfo = kernel.GetLGProfilingInfo();
        lgProfInfo.lgProfilingReps.assign(repN, nullptr);
        lgProfInfo.ccuProfilingInfos.assign(profN, CcuProfilingInfo{});
        kernel.profiling_.groupOpSizeInfo_.assign(goN, infoTemplate);
    }

    void SetRep(CcuKernel& kernel, size_t idx, uint16_t instrId)
    {
        kernel.GetLGProfilingInfo().lgProfilingReps[idx] = MakeLoopGroupBundle(instrId);
    }
};

// 决策点#2(空): safeSize=0，循环不进入
TEST_F(CollectLoopGroupProfilingTest, Ut_CollectLGProf_When_AllContainersEmpty_Expect_SuccessNoOutput)
{
    CcuKernel kernel;
    FillContainers(kernel, 0, 0, 0);
    std::vector<uint64_t> taskArgs = {0};
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo;
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo;
    EXPECT_EQ(
        kernel.GetCcuProfilingInfo(taskArgs.data(), taskArgs.size(), ccuStaticInfo, ccuDetailInfo),
        HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(kernel.GetAllCcuProfilingInfo().empty());
}

// 决策点#1(mismatch): repSize=4 但 goSizeNum=2、profSize=2 → safeSize=2 != repSize=4，触发 warning 分支
TEST_F(CollectLoopGroupProfilingTest, Ut_CollectLGProf_When_SizeMismatch_Expect_SafeSizeTruncate)
{
    CcuKernel kernel;
    FillContainers(kernel, 4, 2, 2);
    std::vector<uint64_t> taskArgs = {0};
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo;
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo;
    EXPECT_EQ(kernel.GetCcuProfilingInfo(taskArgs.data(), 0, ccuStaticInfo, ccuDetailInfo), HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(kernel.GetAllCcuProfilingInfo().empty());
}

// 决策点#1(match): 三容器等长，safeSize==repSize，不触发 warning
TEST_F(CollectLoopGroupProfilingTest, Ut_CollectLGProf_When_SizesEqual_Expect_NoTruncate)
{
    CcuKernel kernel;
    FillContainers(kernel, 2, 2, 2);
    std::vector<uint64_t> taskArgs = {0};
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo;
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo;
    EXPECT_EQ(kernel.GetCcuProfilingInfo(taskArgs.data(), 0, ccuStaticInfo, ccuDetailInfo), HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(kernel.GetAllCcuProfilingInfo().empty());
}

// 决策点#3(map空): varId2ArgIndexMap 为空，循环体提前返回
TEST_F(CollectLoopGroupProfilingTest, Ut_CollectLGProf_When_VarIdMapEmpty_Expect_SkipAndSuccess)
{
    CcuKernel kernel;
    FillContainers(kernel, 2, 2, 2);
    std::vector<uint64_t> taskArgs = {0};
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo;
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo;
    EXPECT_EQ(
        kernel.GetCcuProfilingInfo(taskArgs.data(), taskArgs.size(), ccuStaticInfo, ccuDetailInfo),
        HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(kernel.GetAllCcuProfilingInfo().empty());
}

// 决策点#4(true): loopParam!=0 → dataSize 赋值 + push
TEST_F(CollectLoopGroupProfilingTest, Ut_CollectLGProf_When_LoopParamNonZero_Expect_DataSizePushed)
{
    CcuKernel kernel;
    constexpr uint16_t loopParamId = 100;
    constexpr uint16_t parallelParamId = 101;
    constexpr uint16_t residualId = 102;
    GroupInfo info{loopParamId, parallelParamId, residualId};
    FillContainers(kernel, 2, 2, 2, info);
    SetRep(kernel, 0, 0x1234);
    kernel.profiling_.moConfig_.loopCount = 2;
    kernel.profiling_.moConfig_.memSlice = 3;

    CcuInsGeneratorV1 insGen;
    Variable loopVar(nullptr);
    loopVar.Reset(loopParamId);
    auto loopLoadArg = std::make_shared<CcuRep::CcuRepLoadArg>(&insGen, loopVar, 0, 0);
    kernel.GetLGProfilingInfo().loadRep2ArgIdxMap[loopLoadArg] = 0;

    Variable parallelVar(nullptr);
    parallelVar.Reset(parallelParamId);
    auto parallelLoadArg = std::make_shared<CcuRep::CcuRepLoadArg>(&insGen, parallelVar, 0, 0);
    kernel.GetLGProfilingInfo().loadRep2ArgIdxMap[parallelLoadArg] = 1;

    Variable residualVar(nullptr);
    residualVar.Reset(residualId);
    auto residualLoadArg = std::make_shared<CcuRep::CcuRepLoadArg>(&insGen, residualVar, 0, 0);
    kernel.GetLGProfilingInfo().loadRep2ArgIdxMap[residualLoadArg] = 2;

    std::vector<uint64_t> taskArgs = {5, 0, 0};
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo;
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo;
    EXPECT_EQ(
        kernel.GetCcuProfilingInfo(taskArgs.data(), taskArgs.size(), ccuStaticInfo, ccuDetailInfo),
        HcclResult::HCCL_SUCCESS);

    ASSERT_NE(ccuDetailInfo, nullptr);
    EXPECT_EQ(ccuDetailInfo->size(), 1U);
    EXPECT_EQ(ccuDetailInfo->at(0).dataSize, 5ULL * 2ULL * 3ULL);
    EXPECT_EQ(ccuDetailInfo->at(0).instrId, 0x1234U);
}

// 决策点#4(false): loopParam=0 → 不进入赋值分支，不 push
TEST_F(CollectLoopGroupProfilingTest, Ut_CollectLGProf_When_LoopParamZero_Expect_NoPush)
{
    CcuKernel kernel;
    constexpr uint16_t loopParamId = 100;
    constexpr uint16_t parallelParamId = 101;
    constexpr uint16_t residualId = 102;
    GroupInfo info{loopParamId, parallelParamId, residualId};
    FillContainers(kernel, 2, 2, 2, info);
    SetRep(kernel, 0, 0x1234);

    CcuInsGeneratorV1 insGen;
    Variable loopVar(nullptr);
    loopVar.Reset(loopParamId);
    auto loopLoadArg = std::make_shared<CcuRep::CcuRepLoadArg>(&insGen, loopVar, 0, 0);
    kernel.GetLGProfilingInfo().loadRep2ArgIdxMap[loopLoadArg] = 0;

    Variable parallelVar(nullptr);
    parallelVar.Reset(parallelParamId);
    auto parallelLoadArg = std::make_shared<CcuRep::CcuRepLoadArg>(&insGen, parallelVar, 0, 0);
    kernel.GetLGProfilingInfo().loadRep2ArgIdxMap[parallelLoadArg] = 1;

    Variable residualVar(nullptr);
    residualVar.Reset(residualId);
    auto residualLoadArg = std::make_shared<CcuRep::CcuRepLoadArg>(&insGen, residualVar, 0, 0);
    kernel.GetLGProfilingInfo().loadRep2ArgIdxMap[residualLoadArg] = 2;

    std::vector<uint64_t> taskArgs = {0, 0, 0};
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo;
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo;
    EXPECT_EQ(
        kernel.GetCcuProfilingInfo(taskArgs.data(), taskArgs.size(), ccuStaticInfo, ccuDetailInfo),
        HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ccuDetailInfo, nullptr);
}

// 决策点#5(true)+#6(false): parallelParam!=0 且 i+1<repSize(配对 rep 存在) → push
TEST_F(CollectLoopGroupProfilingTest, Ut_CollectLGProf_When_ParallelNonZero_PairedRep_Expect_Push)
{
    CcuKernel kernel;
    constexpr uint16_t loopParamId = 100;
    constexpr uint16_t parallelParamId = 101;
    constexpr uint16_t residualId = 102;
    GroupInfo info{loopParamId, parallelParamId, residualId};
    FillContainers(kernel, 2, 2, 2, info);
    SetRep(kernel, 1, 0x5678);
    kernel.profiling_.moConfig_.memSlice = 4;

    CcuInsGeneratorV1 insGen;
    Variable loopVar(nullptr);
    loopVar.Reset(loopParamId);
    auto loopLoadArg = std::make_shared<CcuRep::CcuRepLoadArg>(&insGen, loopVar, 0, 0);
    kernel.GetLGProfilingInfo().loadRep2ArgIdxMap[loopLoadArg] = 0;

    Variable parallelVar(nullptr);
    parallelVar.Reset(parallelParamId);
    auto parallelLoadArg = std::make_shared<CcuRep::CcuRepLoadArg>(&insGen, parallelVar, 0, 0);
    kernel.GetLGProfilingInfo().loadRep2ArgIdxMap[parallelLoadArg] = 1;

    Variable residualVar(nullptr);
    residualVar.Reset(residualId);
    auto residualLoadArg = std::make_shared<CcuRep::CcuRepLoadArg>(&insGen, residualVar, 0, 0);
    kernel.GetLGProfilingInfo().loadRep2ArgIdxMap[residualLoadArg] = 2;

    uint64_t parallelParam = static_cast<uint64_t>(3) << 55;
    std::vector<uint64_t> taskArgs = {0, parallelParam, 7};
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo;
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo;
    EXPECT_EQ(
        kernel.GetCcuProfilingInfo(taskArgs.data(), taskArgs.size(), ccuStaticInfo, ccuDetailInfo),
        HcclResult::HCCL_SUCCESS);

    ASSERT_NE(ccuDetailInfo, nullptr);
    EXPECT_EQ(ccuDetailInfo->size(), 1U);
    EXPECT_EQ(ccuDetailInfo->at(0).dataSize, 3ULL * 4ULL + 7ULL);
    EXPECT_EQ(ccuDetailInfo->at(0).instrId, 0x5678U);
}

// 决策点#5(true)+#6(true): parallelParam!=0 但 repSize 为奇数 → i+1>=repSize → 跳过不 push
TEST_F(CollectLoopGroupProfilingTest, Ut_CollectLGProf_When_ParallelNonZero_NoPairedRep_Expect_Skip)
{
    CcuKernel kernel;
    constexpr uint16_t loopParamId = 100;
    constexpr uint16_t parallelParamId = 101;
    constexpr uint16_t residualId = 102;
    GroupInfo info{loopParamId, parallelParamId, residualId};
    FillContainers(kernel, 1, 1, 1, info);
    kernel.profiling_.moConfig_.memSlice = 4;

    CcuInsGeneratorV1 insGen;
    Variable loopVar(nullptr);
    loopVar.Reset(loopParamId);
    auto loopLoadArg = std::make_shared<CcuRep::CcuRepLoadArg>(&insGen, loopVar, 0, 0);
    kernel.GetLGProfilingInfo().loadRep2ArgIdxMap[loopLoadArg] = 0;

    Variable parallelVar(nullptr);
    parallelVar.Reset(parallelParamId);
    auto parallelLoadArg = std::make_shared<CcuRep::CcuRepLoadArg>(&insGen, parallelVar, 0, 0);
    kernel.GetLGProfilingInfo().loadRep2ArgIdxMap[parallelLoadArg] = 1;

    Variable residualVar(nullptr);
    residualVar.Reset(residualId);
    auto residualLoadArg = std::make_shared<CcuRep::CcuRepLoadArg>(&insGen, residualVar, 0, 0);
    kernel.GetLGProfilingInfo().loadRep2ArgIdxMap[residualLoadArg] = 2;

    uint64_t parallelParam = static_cast<uint64_t>(3) << 55;
    std::vector<uint64_t> taskArgs = {0, parallelParam, 7};
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo;
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo;
    EXPECT_EQ(
        kernel.GetCcuProfilingInfo(taskArgs.data(), taskArgs.size(), ccuStaticInfo, ccuDetailInfo),
        HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ccuDetailInfo, nullptr);
}

// 验证 GetCcuProfilingInfo 稳态缓存分支
TEST_F(CollectLoopGroupProfilingTest, Ut_GetCcuProfilingInfo_SteadyState_ReuseCache)
{
    CcuKernel kernel;
    constexpr uint16_t loopParamId = 100;
    constexpr uint16_t parallelParamId = 101;
    constexpr uint16_t residualId = 102;
    GroupInfo info{loopParamId, parallelParamId, residualId};

    CcuProfilingInfo sqeProf;
    sqeProf.type = static_cast<uint8_t>(CcuProfilinType::CCU_TASK_PROFILING);
    sqeProf.name = "UnitTestSqe";
    sqeProf.itemId = 1;
    kernel.GetProfilingInfo().push_back(sqeProf);

    FillContainers(kernel, 2, 2, 2, info);
    SetRep(kernel, 0, 0x1234);
    kernel.profiling_.moConfig_.loopCount = 2;
    kernel.profiling_.moConfig_.memSlice = 3;

    CcuInsGeneratorV1 insGen;
    Variable loopVar(nullptr);
    loopVar.Reset(loopParamId);
    auto loopLoadArg = std::make_shared<CcuRep::CcuRepLoadArg>(&insGen, loopVar, 0, 0);
    kernel.GetLGProfilingInfo().loadRep2ArgIdxMap[loopLoadArg] = 0;

    Variable parallelVar(nullptr);
    parallelVar.Reset(parallelParamId);
    auto parallelLoadArg = std::make_shared<CcuRep::CcuRepLoadArg>(&insGen, parallelVar, 0, 0);
    kernel.GetLGProfilingInfo().loadRep2ArgIdxMap[parallelLoadArg] = 1;

    Variable residualVar(nullptr);
    residualVar.Reset(residualId);
    auto residualLoadArg = std::make_shared<CcuRep::CcuRepLoadArg>(&insGen, residualVar, 0, 0);
    kernel.GetLGProfilingInfo().loadRep2ArgIdxMap[residualLoadArg] = 2;

    std::vector<uint64_t> taskArgs = {5, 0, 0};

    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo1;
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo1;
    EXPECT_EQ(
        kernel.GetCcuProfilingInfo(taskArgs.data(), taskArgs.size(), ccuStaticInfo1, ccuDetailInfo1),
        HcclResult::HCCL_SUCCESS);

    auto& cache1 = kernel.profiling_.cache_;
    EXPECT_EQ(cache1.sqeWaitCkeSize, 1U);
    EXPECT_TRUE(cache1.varIdMapsCached);
    EXPECT_TRUE(cache1.groupArgIndexCached);
    ASSERT_NE(ccuStaticInfo1, nullptr);
    EXPECT_EQ(ccuStaticInfo1->size(), 1U);
    ASSERT_NE(ccuDetailInfo1, nullptr);
    EXPECT_EQ(ccuDetailInfo1->size(), 1U);
    EXPECT_EQ(ccuDetailInfo1->at(0).dataSize, 5ULL * 2ULL * 3ULL);

    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo2;
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo2;
    EXPECT_EQ(
        kernel.GetCcuProfilingInfo(taskArgs.data(), taskArgs.size(), ccuStaticInfo2, ccuDetailInfo2),
        HcclResult::HCCL_SUCCESS);

    auto& cache2 = kernel.profiling_.cache_;
    EXPECT_EQ(cache2.sqeWaitCkeSize, 1U);
    EXPECT_EQ(ccuStaticInfo2.get(), ccuStaticInfo1.get());
    ASSERT_NE(ccuDetailInfo2, nullptr);
    EXPECT_EQ(ccuDetailInfo2->size(), ccuDetailInfo1->size());
    EXPECT_EQ(ccuDetailInfo2->at(0).dataSize, ccuDetailInfo1->at(0).dataSize);
    EXPECT_EQ(kernel.GetAllCcuProfilingInfo().size(), 2U);
}

TEST_F(CollectLoopGroupProfilingTest, Ut_GetCcuProfilingInfo_When_SqeProfilingExists_Expect_ccuStaticInfoNotNull)
{
    CcuKernel kernel;
    CcuProfilingInfo sqeProf;
    sqeProf.type = static_cast<uint8_t>(CcuProfilinType::CCU_TASK_PROFILING);
    sqeProf.name = "UnitTestSqe";
    sqeProf.itemId = 1;
    sqeProf.dieId = 1;
    kernel.GetProfilingInfo().push_back(sqeProf);

    std::vector<uint64_t> taskArgs = {0};
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo;
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo;
    EXPECT_EQ(
        kernel.GetCcuProfilingInfo(taskArgs.data(), taskArgs.size(), ccuStaticInfo, ccuDetailInfo),
        HcclResult::HCCL_SUCCESS);

    ASSERT_NE(ccuStaticInfo, nullptr);
    EXPECT_EQ(ccuStaticInfo->size(), 1U);
    EXPECT_EQ((*ccuStaticInfo)[0].type, static_cast<uint8_t>(CcuProfilinType::CCU_TASK_PROFILING));
    EXPECT_EQ((*ccuStaticInfo)[0].name, "UnitTestSqe");
    EXPECT_EQ((*ccuStaticInfo)[0].dieId, 1);
    EXPECT_EQ(kernel.profiling_.cache_.sqeWaitCkeBuilt, true);
    EXPECT_EQ(kernel.profiling_.cache_.sqeWaitCkeSize, 1U);
    EXPECT_EQ(ccuDetailInfo, nullptr);
}

TEST_F(CollectLoopGroupProfilingTest, Ut_GetCcuProfilingInfo_When_NoSqeProfiling_Expect_ccuStaticInfoNull)
{
    CcuKernel kernel;

    std::vector<uint64_t> taskArgs = {0};
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo;
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo;
    EXPECT_EQ(
        kernel.GetCcuProfilingInfo(taskArgs.data(), taskArgs.size(), ccuStaticInfo, ccuDetailInfo),
        HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(ccuStaticInfo, nullptr);
    EXPECT_EQ(kernel.profiling_.cache_.sqeWaitCkeBuilt, true);
    EXPECT_EQ(kernel.profiling_.cache_.sqeWaitCkeSize, 0U);
    EXPECT_EQ(ccuDetailInfo, nullptr);
}
