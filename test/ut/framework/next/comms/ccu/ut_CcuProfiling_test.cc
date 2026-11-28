/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// CcuProfiling 的 kernel 名登记与 profiling 信息回填用例。
// 原本覆盖这两个接口的 ut_HcommCcuProfiling_test.cc 不参与构建, 故在本目标内补齐。
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "ccu_profiling.h"

namespace hcomm {
namespace {
    constexpr uint32_t TEST_MISSION_ID = 7;
    constexpr uint32_t TEST_INSTR_ID = 4321;
    constexpr size_t MAX_KERNEL_FUNC_NAME_LEN = 128;
    constexpr char DEFAULT_KERNEL_NAME[] = "CCU_KERNEL";

    class CcuProfilingTest : public testing::Test {
    protected:
        CcuRep::CcuRepContext repCtx_;
        CcuProfiling profiling_{repCtx_};
    };

    TEST_F(CcuProfilingTest, Ut_SetupProfilingInfo_When_NameNull_Expect_DefaultName)
    {
        EXPECT_EQ(profiling_.SetupProfilingInfo(nullptr), HcclResult::HCCL_SUCCESS);

        EXPECT_EQ(profiling_.GetName(), DEFAULT_KERNEL_NAME);
        ASSERT_EQ(repCtx_.GetProfilingInfo().size(), 1U);
        EXPECT_EQ(repCtx_.GetProfilingInfo()[0].name, DEFAULT_KERNEL_NAME);
        EXPECT_EQ(repCtx_.GetProfilingInfo()[0].type, static_cast<uint8_t>(CcuProfilinType::CCU_TASK_PROFILING));
    }

    TEST_F(CcuProfilingTest, Ut_SetupProfilingInfo_When_NameEmpty_Expect_DefaultName)
    {
        EXPECT_EQ(profiling_.SetupProfilingInfo(""), HcclResult::HCCL_SUCCESS);

        EXPECT_EQ(profiling_.GetName(), DEFAULT_KERNEL_NAME);
        ASSERT_EQ(repCtx_.GetProfilingInfo().size(), 1U);
        EXPECT_EQ(repCtx_.GetProfilingInfo()[0].name, DEFAULT_KERNEL_NAME);
    }

    TEST_F(CcuProfilingTest, Ut_SetupProfilingInfo_When_NameNormal_Expect_NameRecorded)
    {
        const std::string kernelName = "Ut_AllReduceKernel";
        EXPECT_EQ(profiling_.SetupProfilingInfo(kernelName.c_str()), HcclResult::HCCL_SUCCESS);

        EXPECT_EQ(profiling_.GetName(), kernelName);
        ASSERT_EQ(repCtx_.GetProfilingInfo().size(), 1U);
        EXPECT_EQ(repCtx_.GetProfilingInfo()[0].name, kernelName);
    }

    TEST_F(CcuProfilingTest, Ut_SetupProfilingInfo_When_NameTooLong_Expect_Truncated)
    {
        const std::string kernelName(MAX_KERNEL_FUNC_NAME_LEN + 1, 'a');
        EXPECT_EQ(profiling_.SetupProfilingInfo(kernelName.c_str()), HcclResult::HCCL_SUCCESS);

        EXPECT_EQ(profiling_.GetName(), kernelName.substr(0, MAX_KERNEL_FUNC_NAME_LEN));
        ASSERT_EQ(repCtx_.GetProfilingInfo().size(), 1U);
        EXPECT_EQ(repCtx_.GetProfilingInfo()[0].name.size(), MAX_KERNEL_FUNC_NAME_LEN);
    }

    TEST_F(CcuProfilingTest, Ut_UpdateProfilingDieId_When_DieSelected_Expect_DieIdRefreshed)
    {
        constexpr uint32_t targetDieId = 1;
        EXPECT_EQ(profiling_.SetupProfilingInfo("Ut_DieKernel"), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(profiling_.UpdateProfilingDieId(targetDieId), HcclResult::HCCL_SUCCESS);

        ASSERT_EQ(repCtx_.GetProfilingInfo().size(), 1U);
        EXPECT_EQ(repCtx_.GetProfilingInfo()[0].dieId, targetDieId);
    }

    TEST_F(CcuProfilingTest, Ut_GetCcuProfilingInfo_When_CacheEmpty_Expect_Empty)
    {
        std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo;
        std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo;
        EXPECT_EQ(
            profiling_.GetCcuProfilingInfo(nullptr, 0, ccuStaticInfo, ccuDetailInfo, TEST_INSTR_ID),
            HcclResult::HCCL_SUCCESS);

        EXPECT_EQ(ccuStaticInfo, nullptr);
        EXPECT_EQ(ccuDetailInfo, nullptr);
        EXPECT_TRUE(profiling_.GetAllCcuProfilingInfo().empty());
    }

    TEST_F(CcuProfilingTest, Ut_GetCcuProfilingInfo_When_SqeCached_Expect_MissionAndInstrIdFilled)
    {
        const std::string kernelName = "Ut_SqeKernel";
        repCtx_.SetMissionId(TEST_MISSION_ID);
        EXPECT_EQ(profiling_.SetupProfilingInfo(kernelName.c_str()), HcclResult::HCCL_SUCCESS);

        std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo;
        std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo;
        EXPECT_EQ(
            profiling_.GetCcuProfilingInfo(nullptr, 0, ccuStaticInfo, ccuDetailInfo, TEST_INSTR_ID),
            HcclResult::HCCL_SUCCESS);

        const auto& out = profiling_.GetAllCcuProfilingInfo();
        ASSERT_EQ(out.size(), 1U);
        EXPECT_EQ(out[0].name, kernelName);
        EXPECT_EQ(out[0].missionId, TEST_MISSION_ID);
        EXPECT_EQ(out[0].instrId, TEST_INSTR_ID);
    }

    // 连续两次获取不应累积: 第二次 launch 复用缓存并截断尾部, 结果条数与首次一致。
    TEST_F(CcuProfilingTest, Ut_GetCcuProfilingInfo_When_CalledTwice_Expect_NoAccumulation)
    {
        EXPECT_EQ(profiling_.SetupProfilingInfo("Ut_RepeatKernel"), HcclResult::HCCL_SUCCESS);

        std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo;
        std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo;
        EXPECT_EQ(
            profiling_.GetCcuProfilingInfo(nullptr, 0, ccuStaticInfo, ccuDetailInfo, TEST_INSTR_ID),
            HcclResult::HCCL_SUCCESS);
        const size_t firstSize = profiling_.GetAllCcuProfilingInfo().size();
        EXPECT_EQ(
            profiling_.GetCcuProfilingInfo(nullptr, 0, ccuStaticInfo, ccuDetailInfo, TEST_INSTR_ID),
            HcclResult::HCCL_SUCCESS);

        EXPECT_EQ(firstSize, 1U);
        EXPECT_EQ(profiling_.GetAllCcuProfilingInfo().size(), firstSize);
    }

    TEST_F(CcuProfilingTest, Ut_AddProfilingInfo_When_NoLoopGroupRep_Expect_RepListEmpty)
    {
        const ChannelHandle channels[1]{0};
        constexpr uint32_t channelNum = 0;
        EXPECT_EQ(
            profiling_.AddProfilingInfo(
                channels, channelNum, HcclDataType::HCCL_DATA_TYPE_FP32, HcclDataType::HCCL_DATA_TYPE_FP32,
                HcclReduceOp::HCCL_REDUCE_SUM, "Ut_LoopGroupOp"),
            HcclResult::HCCL_SUCCESS);

        EXPECT_EQ(repCtx_.GetLGProfilingInfo().ccuProfilingInfos.size(), 1U);
        EXPECT_TRUE(repCtx_.GetLGProfilingInfo().lgProfilingReps.empty());
    }

    TEST_F(CcuProfilingTest, Ut_AddProfilingInfo_When_ChannelNumExceedsMax_Expect_ParaAndNoCache)
    {
        const ChannelHandle channels[CCU_MAX_CHANNEL_NUM + 1]{};
        constexpr uint32_t channelNum = CCU_MAX_CHANNEL_NUM + 1;
        EXPECT_EQ(
            profiling_.AddProfilingInfo(
                channels, channelNum, HcclDataType::HCCL_DATA_TYPE_FP32, HcclDataType::HCCL_DATA_TYPE_FP32,
                HcclReduceOp::HCCL_REDUCE_SUM, "Ut_LoopGroupOp"),
            HcclResult::HCCL_E_PARA);

        EXPECT_TRUE(repCtx_.GetLGProfilingInfo().ccuProfilingInfos.empty());
    }

    // 取满上限的 channelNum 须越过上限校验, 失败只应来自后续的通道查询。
    TEST_F(CcuProfilingTest, Ut_AddProfilingInfo_When_ChannelNumAtMax_Expect_NotPara)
    {
        const ChannelHandle channels[CCU_MAX_CHANNEL_NUM]{};
        constexpr uint32_t channelNum = CCU_MAX_CHANNEL_NUM;
        EXPECT_NE(
            profiling_.AddProfilingInfo(
                channels, channelNum, HcclDataType::HCCL_DATA_TYPE_FP32, HcclDataType::HCCL_DATA_TYPE_FP32,
                HcclReduceOp::HCCL_REDUCE_SUM, "Ut_LoopGroupOp"),
            HcclResult::HCCL_E_PARA);
    }

    // 空 rep 需在取 StartInstrId 之前被拦下, 故同时注入 waitcke 缓存与空 rep。
    TEST_F(CcuProfilingTest, Ut_GetCcuProfilingInfo_When_WaitCkeRepNull_Expect_PtrError)
    {
        CcuProfilingInfo waitCkeInfo{};
        waitCkeInfo.type = static_cast<uint8_t>(CcuProfilinType::CCU_WAITCKE_PROFILING);
        repCtx_.GetProfilingInfo().push_back(waitCkeInfo);
        repCtx_.waitCkeProfilingReps.push_back(nullptr);

        std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo;
        std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo;
        EXPECT_EQ(
            profiling_.GetCcuProfilingInfo(nullptr, 0, ccuStaticInfo, ccuDetailInfo, TEST_INSTR_ID),
            HcclResult::HCCL_E_PTR);
    }
} // namespace
} // namespace hcomm
