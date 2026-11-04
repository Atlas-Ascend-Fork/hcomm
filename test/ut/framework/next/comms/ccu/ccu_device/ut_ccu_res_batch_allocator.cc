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
#include <mockcpp/mockcpp.hpp>

#include "src/base_comm/resources/ccu/ccu_device/ccu_res_batch_allocator.h"
#include "ccu_comp.h"
#include "hccl_common.h"

#define private public
#define protected public

using namespace hcomm;

static void InitTestMissionReq(MissionReq& missionReq, MissionReqType reqType)
{
    missionReq.reqType = reqType;
    missionReq.req[0] = 0;
    missionReq.req[1] = 0;
}

class CcuResBatchAllocatorTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    static void TearDownTestCase()
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    void SetUp() override
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    void TearDown() override
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
};

TEST_F(CcuResBatchAllocatorTest, Ut_Alloc_When_ReqTypeNotDefault_Expect_ForceSetToDefault)
{
    hcomm::CcuResBatchAllocator allocat{};
    uintptr_t handleKey;
    MissionReq missionReq;
    MissionResInfo missionInfos;

    // 调用公共函数
    InitTestMissionReq(missionReq, MissionReqType::COMM_ENGINE_RESERVED);

    auto ret = allocat.missionMgr_.Alloc(handleKey, missionReq, missionInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuResBatchAllocatorTest, Ut_Alloc_When_ReqNumIs0_Expect_ReturnSuccessDirectly)
{
    hcomm::CcuResBatchAllocator allocat{};
    uintptr_t handleKey;
    MissionReq missionReq;
    MissionResInfo missionInfos;

    // 调用公共函数
    InitTestMissionReq(missionReq, MissionReqType::FUSION_MULTIPLE_DIE);

    auto ret = allocat.missionMgr_.Alloc(handleKey, missionReq, missionInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_NoAllocations_Expect_MaxGapEqualsTotalCapacity)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags_[0] = true;
    allocator.userDevId_ = 0;

    allocator.resStrategies_[0].loopNum = 8;
    allocator.maxResBlockNums_.loopNum = 4; // poolSize = 4 * 8 = 32
    allocator.resBlocks_[0][ResType::LOOP] = {
        {0, 0, 8, 0, false},
        {1, 8, 8, 0, false},
        {2, 16, 8, 0, false},
        {3, 24, 8, 0, false},
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::LOOP, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(remainNum, 32u) << "no allocations -> max gap = pool size 32";
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_WithAllocations_Expect_MaxContiguousGap)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags_[0] = true;
    allocator.userDevId_ = 0;

    allocator.maxResBlockNums_.loopNum = 8; // poolSize = 8 * 8 = 64
    allocator.resBlocks_[0][ResType::LOOP] = {
        {0, 0, 8, 0, true},   // allocated
        {1, 8, 8, 0, true},   // allocated
        {2, 16, 8, 0, false}, // free
        {3, 24, 8, 0, true},  // allocated
        {4, 32, 8, 0, false}, // free
        {5, 40, 8, 0, false}, // free
        {6, 48, 8, 0, false}, // free
        {7, 56, 8, 0, false}, // free
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::LOOP, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // blocks 4-7 连续空闲: 4 × 8 = 32
    EXPECT_EQ(remainNum, 32u);
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_AllAllocated_Expect_Zero)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags_[0] = true;
    allocator.userDevId_ = 0;

    allocator.maxResBlockNums_.loopNum = 4; // poolSize = 4 * 8 = 32
    allocator.resBlocks_[0][ResType::LOOP] = {
        {0, 0, 8, 0, true},
        {1, 8, 8, 0, true},
        {2, 16, 8, 0, true},
        {3, 24, 8, 0, true},
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::LOOP, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(remainNum, 0u) << "all allocated -> zero remaining";
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_HoleAtStart_Expect_InitialGap)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags_[0] = true;
    allocator.userDevId_ = 0;

    allocator.maxResBlockNums_.loopNum = 6;
    allocator.resBlocks_[0][ResType::LOOP] = {
        {0, 0, 8, 0, true},   {1, 8, 8, 0, true},   {2, 16, 8, 0, false},
        {3, 24, 8, 0, false}, {4, 32, 8, 0, false}, {5, 40, 8, 0, false},
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::LOOP, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 4 free blocks × 8 = 32
    EXPECT_EQ(remainNum, 32u);
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_HoleAtEnd_Expect_TailGap)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags_[0] = true;
    allocator.userDevId_ = 0;

    allocator.maxResBlockNums_.loopNum = 6;
    allocator.resBlocks_[0][ResType::LOOP] = {
        {0, 0, 8, 0, false},  {1, 8, 8, 0, false}, {2, 16, 8, 0, false},
        {3, 24, 8, 0, false}, {4, 32, 8, 0, true}, {5, 40, 8, 0, true},
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::LOOP, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 4 free blocks × 8 = 32
    EXPECT_EQ(remainNum, 32u);
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_PreReservedMSBlocks_Expect_MaxGapReducedByReserved)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags_[0] = true;
    allocator.userDevId_ = 0;

    allocator.resStrategies_[0].msNum = 64;
    allocator.maxResBlockNums_.msNum = 4; // poolSize = 4 * 64 = 256

    // Block0: 预预留(allocated), Block1: 空闲, Block2: 已分配, Block3: 空闲
    allocator.resBlocks_[0][ResType::MS] = {
        {0, 0, 64, 0, true},
        {1, 64, 64, 0, false},
        {2, 128, 64, 0, true},
        {3, 192, 64, 0, false},
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::MS, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 最大连续空闲: 单 block = 64
    EXPECT_EQ(remainNum, 64u) << "pre-reserved + allocated block leave max gap 64";
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_PoolStartsAtNonZero_Expect_GapWithinPoolOnly)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags_[0] = true;
    allocator.userDevId_ = 0;

    // 模拟 mission 预分配池 [100, 356), 4 blocks of 64
    allocator.missionMgr_.blocks_ = {
        {0, 100, 64, 0, true},  // 已分配
        {1, 164, 64, 0, false}, // 空闲
        {2, 228, 64, 0, true},  // 已分配
        {3, 292, 64, 0, false}, // 空闲
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::MISSION, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 最大连续空闲: 单 block = 64
    EXPECT_EQ(remainNum, 64u) << "max gap 64 within pool only";
}

TEST_F(CcuResBatchAllocatorTest, Ut_GetConsecutiveRemainSize_When_PartialAlloc_Expect_MaxGap)
{
    CcuResIdAllocator allocator(100);
    allocator.resInfos_ = {{0, 20}, {70, 20}};

    EXPECT_EQ(allocator.GetConsecutiveRemainSize(), 50u);
}

TEST_F(CcuResBatchAllocatorTest, Ut_GetConsecutiveRemainSize_When_Full_Expect_Zero)
{
    CcuResIdAllocator allocator(50);
    std::vector<ResInfo> resInfos;
    allocator.Alloc(50, true, resInfos);

    EXPECT_EQ(allocator.GetConsecutiveRemainSize(), 0u);
}

namespace {
int g_cascAllocCallCount = 0;
int g_cascConfirmCallCount = 0;
int g_cascResetCallCount = 0;
int g_cascReleaseCallCount = 0;

constexpr uint32_t CASC_CNT_START_ID = 12288;
constexpr uint32_t WISH_CNT_XN_NUM = 1022;
// BuildCntXnBlock 按 CCU 资源空间规格计算用户态地址，UT 不加载真实规格，此处给出固定地址
constexpr uint64_t MOCK_RESOURCE_ADDR = 0xE7FFBF800000ULL;
constexpr uint64_t MOCK_XN_ADDR = 0x10000ULL;

CcuResult MockCascAllocSuccess(const int32_t, const uint8_t, const uint32_t num, ResInfo& resInfo)
{
    g_cascAllocCallCount++;
    resInfo = ResInfo(CASC_CNT_START_ID, num);
    return CcuResult::CCU_SUCCESS;
}

CcuResult MockCascAllocUnavail(const int32_t, const uint8_t, const uint32_t, ResInfo&)
{
    g_cascAllocCallCount++;
    return CcuResult::CCU_E_UNAVAIL;
}

// 用于覆盖多 die 部分失败：die0 申请成功，die1 私有池耗尽
CcuResult MockCascAllocUnavailOnDie1(const int32_t, const uint8_t dieId, const uint32_t num, ResInfo& resInfo)
{
    g_cascAllocCallCount++;
    if (dieId == 1) {
        return CcuResult::CCU_E_UNAVAIL;
    }
    resInfo = ResInfo(CASC_CNT_START_ID, num);
    return CcuResult::CCU_SUCCESS;
}

CcuResult MockCascConfirmSuccess(
    const int32_t, const uint8_t, const std::pair<uint32_t, uint32_t>&, const uint32_t, const uint32_t,
    uint32_t& blockIdx)
{
    g_cascConfirmCallCount++;
    blockIdx = 0;
    return CcuResult::CCU_SUCCESS;
}

CcuResult MockCascConfirmUnavail(
    const int32_t, const uint8_t, const std::pair<uint32_t, uint32_t>&, const uint32_t, const uint32_t,
    uint32_t& blockIdx)
{
    g_cascConfirmCallCount++;
    blockIdx = 0;
    return CcuResult::CCU_E_UNAVAIL;
}

CcuResult MockCascReset(const int32_t, const uint8_t, const uint32_t)
{
    g_cascResetCallCount++;
    return CcuResult::CCU_SUCCESS;
}

CcuResult MockCascRelease(const int32_t, const uint8_t, const ResInfo&)
{
    g_cascReleaseCallCount++;
    return CcuResult::CCU_SUCCESS;
}

void MockCcuResAddrSpecs(uint64_t& resourceAddr, uint64_t& xnAddr)
{
    resourceAddr = MOCK_RESOURCE_ADDR;
    xnAddr = MOCK_XN_ADDR;
    MOCKER_CPP(&CcuResSpecifications::GetResourceAddr)
        .stubs()
        .with(mockcpp::any(), outBound(resourceAddr))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuResSpecifications::GetXnOffsetCcumAddrById)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(xnAddr))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
}

// 基地址为 0 时 BuildCntXnBlock 拒绝组装块，用于覆盖申请成功后的回滚分支
void MockCcuResAddrZero(uint64_t& resourceAddr)
{
    resourceAddr = 0;
    MOCKER_CPP(&CcuResSpecifications::GetResourceAddr)
        .stubs()
        .with(mockcpp::any(), outBound(resourceAddr))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
}
} // namespace

TEST_F(CcuResBatchAllocatorTest, Ut_AllocCascCntRes_When_ReqNumWithinLimit_Expect_BlocksRecorded)
{
    g_cascAllocCallCount = 0;
    g_cascConfirmCallCount = 0;
    uint64_t resourceAddr = 0;
    uint64_t xnAddr = 0;
    MockCcuResAddrSpecs(resourceAddr, xnAddr);
    MOCKER(CcuAllocCntXn).stubs().will(invoke(MockCascAllocSuccess));
    MOCKER(CcuConfirmCntXns).stubs().will(invoke(MockCascConfirmSuccess));

    CcuResBatchAllocator allocator;
    allocator.userDevId_ = 0;
    CcuResReq resReq{};
    resReq.cascCntReq[0] = 2;
    auto resRepoPtr = std::make_unique<CcuResRepository>();

    EXPECT_EQ(allocator.AllocCascCntRes(resReq, resRepoPtr), HCCL_SUCCESS);
    EXPECT_EQ(g_cascAllocCallCount, 2);
    EXPECT_EQ(g_cascConfirmCallCount, 2);
    ASSERT_EQ(resRepoPtr->cascCntBlock[0].size(), 2U);
    EXPECT_NE(resRepoPtr->cascCntBlock[0][0].wishCntXnsMem.first, nullptr);
    EXPECT_EQ(resRepoPtr->cascCntBlock[0][0].resInfo.startId, CASC_CNT_START_ID);
    EXPECT_EQ(resRepoPtr->cascCntBlock[0][0].totalCntXn, CASC_CNT_START_ID + WISH_CNT_XN_NUM);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocCascCntRes_When_ReqNumExceedsLimit_Expect_UNAVAIL)
{
    g_cascAllocCallCount = 0;
    MOCKER(CcuAllocCntXn).stubs().will(invoke(MockCascAllocSuccess));

    CcuResBatchAllocator allocator;
    allocator.userDevId_ = 0;
    CcuResReq resReq{};
    resReq.cascCntReq[0] = CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM + 1;
    auto resRepoPtr = std::make_unique<CcuResRepository>();

    EXPECT_EQ(allocator.AllocCascCntRes(resReq, resRepoPtr), HCCL_E_UNAVAIL);
    EXPECT_EQ(g_cascAllocCallCount, 0);
    EXPECT_EQ(resRepoPtr->cascCntBlock[0].size(), 0U);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocCascCntRes_When_AllocUnavail_Expect_UNAVAIL)
{
    g_cascAllocCallCount = 0;
    MOCKER(CcuAllocCntXn).stubs().will(invoke(MockCascAllocUnavail));

    CcuResBatchAllocator allocator;
    allocator.userDevId_ = 0;
    CcuResReq resReq{};
    resReq.cascCntReq[0] = 1;
    auto resRepoPtr = std::make_unique<CcuResRepository>();

    EXPECT_EQ(allocator.AllocCascCntRes(resReq, resRepoPtr), HCCL_E_UNAVAIL);
    EXPECT_EQ(resRepoPtr->cascCntBlock[0].size(), 0U);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocCascCntRes_When_MultiDiePartiallyFail_Expect_DieSucceededBlocksReleased)
{
    g_cascAllocCallCount = 0;
    g_cascConfirmCallCount = 0;
    g_cascResetCallCount = 0;
    g_cascReleaseCallCount = 0;
    uint64_t resourceAddr = 0;
    uint64_t xnAddr = 0;
    MockCcuResAddrSpecs(resourceAddr, xnAddr);
    MOCKER(CcuAllocCntXn).stubs().will(invoke(MockCascAllocUnavailOnDie1));
    MOCKER(CcuConfirmCntXns).stubs().will(invoke(MockCascConfirmSuccess));
    MOCKER(CcuResetTotalCntXn).stubs().will(invoke(MockCascReset));
    MOCKER(CcuReleaseCntXn).stubs().will(invoke(MockCascRelease));

    CcuResBatchAllocator allocator;
    allocator.userDevId_ = 0;
    CcuResReq resReq{};
    resReq.cascCntReq[0] = 2;
    resReq.cascCntReq[1] = 1;
    auto resRepoPtr = std::make_unique<CcuResRepository>();

    // die0 已落 2 个块后 die1 申请失败：本层只返回失败，已落块由调用方
    // AllocResHandle 的临时资源释放路径（ReleaseResource -> ReleaseCascCntRes）回收
    EXPECT_EQ(allocator.AllocCascCntRes(resReq, resRepoPtr), HCCL_E_UNAVAIL);
    EXPECT_EQ(g_cascAllocCallCount, 3);
    EXPECT_EQ(g_cascConfirmCallCount, 2);
    ASSERT_EQ(resRepoPtr->cascCntBlock[0].size(), 2U);
    EXPECT_EQ(resRepoPtr->cascCntBlock[1].size(), 0U);

    EXPECT_EQ(allocator.ReleaseResource(resRepoPtr), HCCL_SUCCESS);
    EXPECT_EQ(g_cascResetCallCount, 2);
    EXPECT_EQ(g_cascReleaseCallCount, 2);
    EXPECT_EQ(resRepoPtr->cascCntBlock[0].size(), 0U);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocCascCntRes_When_ConfirmUnavail_Expect_RollbackAndUNAVAIL)
{
    g_cascAllocCallCount = 0;
    g_cascConfirmCallCount = 0;
    g_cascResetCallCount = 0;
    g_cascReleaseCallCount = 0;
    MOCKER(CcuAllocCntXn).stubs().will(invoke(MockCascAllocSuccess));
    MOCKER(CcuConfirmCntXns).stubs().will(invoke(MockCascConfirmUnavail));
    MOCKER(CcuResetTotalCntXn).stubs().will(invoke(MockCascReset));
    MOCKER(CcuReleaseCntXn).stubs().will(invoke(MockCascRelease));

    CcuResBatchAllocator allocator;
    allocator.userDevId_ = 0;
    CcuResReq resReq{};
    resReq.cascCntReq[0] = 1;
    auto resRepoPtr = std::make_unique<CcuResRepository>();

    EXPECT_EQ(allocator.AllocCascCntRes(resReq, resRepoPtr), HCCL_E_UNAVAIL);
    EXPECT_EQ(g_cascAllocCallCount, 1);
    EXPECT_EQ(g_cascConfirmCallCount, 1);
    // 申请成功但确认失败：total 比较寄存器由 CcuConfirmCntXns 内部回滚，
    // 本层只回收 cnt xn，且不落仓库
    EXPECT_EQ(g_cascResetCallCount, 0);
    EXPECT_EQ(g_cascReleaseCallCount, 1);
    EXPECT_EQ(resRepoPtr->cascCntBlock[0].size(), 0U);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocCascCntRes_When_BuildBlockFail_Expect_RollbackAndINTERNAL)
{
    g_cascAllocCallCount = 0;
    g_cascConfirmCallCount = 0;
    g_cascResetCallCount = 0;
    g_cascReleaseCallCount = 0;
    uint64_t resourceAddr = 0;
    MockCcuResAddrZero(resourceAddr);
    MOCKER(CcuAllocCntXn).stubs().will(invoke(MockCascAllocSuccess));
    MOCKER(CcuConfirmCntXns).stubs().will(invoke(MockCascConfirmSuccess));
    MOCKER(CcuResetTotalCntXn).stubs().will(invoke(MockCascReset));
    MOCKER(CcuReleaseCntXn).stubs().will(invoke(MockCascRelease));

    CcuResBatchAllocator allocator;
    allocator.userDevId_ = 0;
    CcuResReq resReq{};
    resReq.cascCntReq[0] = 1;
    auto resRepoPtr = std::make_unique<CcuResRepository>();

    EXPECT_EQ(allocator.AllocCascCntRes(resReq, resRepoPtr), HCCL_E_INTERNAL);
    EXPECT_EQ(g_cascAllocCallCount, 1);
    EXPECT_EQ(g_cascConfirmCallCount, 1);
    // 块组装失败，需回滚 total 比较寄存器与 cnt xn，且不落仓库
    EXPECT_EQ(g_cascResetCallCount, 1);
    EXPECT_EQ(g_cascReleaseCallCount, 1);
    EXPECT_EQ(resRepoPtr->cascCntBlock[0].size(), 0U);
}

TEST_F(CcuResBatchAllocatorTest, Ut_ReleaseCascCntRes_When_BlocksExist_Expect_ReleasedAndCleared)
{
    g_cascResetCallCount = 0;
    g_cascReleaseCallCount = 0;
    MOCKER(CcuResetTotalCntXn).stubs().will(invoke(MockCascReset));
    MOCKER(CcuReleaseCntXn).stubs().will(invoke(MockCascRelease));

    CcuResBatchAllocator allocator;
    allocator.userDevId_ = 0;
    auto resRepoPtr = std::make_unique<CcuResRepository>();
    resRepoPtr->cascCntBlock[0].emplace_back();
    resRepoPtr->cascCntBlock[0].emplace_back();

    allocator.ReleaseCascCntRes(resRepoPtr);
    EXPECT_EQ(g_cascResetCallCount, 2);
    EXPECT_EQ(g_cascReleaseCallCount, 2);
    EXPECT_EQ(resRepoPtr->cascCntBlock[0].size(), 0U);
}

// Deinit 需与正常释放同口径, 否则 total 比较寄存器占用标志会泄漏到下次 Init
TEST_F(CcuResBatchAllocatorTest, Ut_Deinit_When_HandleMapNotEmpty_Expect_CascCntResReleasedAndCleared)
{
    g_cascResetCallCount = 0;
    g_cascReleaseCallCount = 0;
    uint64_t resourceAddr = 0;
    uint64_t xnAddr = 0;
    MockCcuResAddrSpecs(resourceAddr, xnAddr);
    MOCKER(CcuAllocCntXn).stubs().will(invoke(MockCascAllocSuccess));
    MOCKER(CcuConfirmCntXns).stubs().will(invoke(MockCascConfirmSuccess));
    MOCKER(CcuResetTotalCntXn).stubs().will(invoke(MockCascReset));
    MOCKER(CcuReleaseCntXn).stubs().will(invoke(MockCascRelease));

    CcuResBatchAllocator allocator;
    allocator.userDevId_ = 0;
    CcuResReq resReq{};
    resReq.cascCntReq[0] = 1;
    auto resRepoPtr = std::make_unique<CcuResRepository>();
    ASSERT_EQ(allocator.AllocCascCntRes(resReq, resRepoPtr), HCCL_SUCCESS);
    allocator.handleMap_[1] = std::move(resRepoPtr);

    EXPECT_EQ(allocator.Deinit(), HCCL_SUCCESS);
    EXPECT_EQ(g_cascResetCallCount, 1);
    EXPECT_EQ(g_cascReleaseCallCount, 1);
    EXPECT_TRUE(allocator.handleMap_.empty());
}
