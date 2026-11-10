/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include <vector>
#include <memory>
#include <algorithm>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <unordered_set>
#define private public
#define protected public
#include "rtsq_a5.h"
#include "binary_stream.h"
#include "config_plf_log_v2.h"
#include "env_config_v2.h"
#include "sqe.h"
#include "ascend_hal.h"
#include "drv_api_exception.h"
#include "rtsq_base.h"
#include "internal_exception.h"
#include "aicpu_ts_thread.h"
#include "stream_lite.h"
#include "sqe_build_a5.h"
#include "adapter_rts_common.h"
#undef protected
#undef private

using namespace Hccl;
class RtsqA5Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RtsqA5 tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RtsqA5 tests tear down." << std::endl; }

    virtual void SetUp()
    {
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqDepth)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(static_cast<u32>(AC_SQE_MAX_CNT)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(1)));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();

        std::cout << "A Test case in RtsqA5 SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        SetPlfDebugConfigValue(0);
        // 重置SQE profiling置位开关，防止用例异常路径(THROW)泄漏状态污染后续用例
        Hccl::SetSqeProfilingEnabled(false);
        GlobalMockObject::verify();
        std::cout << "A Test case in RtsqA5 TearDown" << std::endl;
    }

    u32 fakedevPhyId = 0;
    u32 fakeStreamId = 1;
    u32 fakeSqId = 2;
    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
};

TEST_F(RtsqA5Test, CCoreWithoutDfxStillSubmitsSqe)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId, false);
    rtsq.taskId_ = 0x80000400U;
    rtsq.taskIdEnd_ = 0x80000800U;
    const auto taskId = rtsq.GetTaskId();
    rtsq.CCoreNotifyWait(0x1000, 0x2000, false);
    rtsq.CCoreNotifyRecord(0x3000, 0x2000);
    EXPECT_EQ(rtsq.GetTaskId(), taskId + 2);
}

class IsRtsqQueueSpaceSufficientTest : public RtsqA5Test {
protected:
    void SetUp() override
    {
        RtsqA5Test::SetUp();
        // 初始化测试环境
        pendingSqeCnt = 5; // 假设pendingSqeCnt为5
    }
    u32 pendingSqeCnt; // 模拟pendingSqeCnt的值
};

TEST_F(IsRtsqQueueSpaceSufficientTest, Ut_IsRtsqQueueSpaceSufficient_When_HeadEqualTail_ExpectFalse)
{
    // 准备测试数据
    RtsqA5 fakeRtsqA5(fakedevPhyId, fakeStreamId, fakeSqId);
    fakeRtsqA5.pendingSqeCnt = pendingSqeCnt;
    fakeRtsqA5.sqDepth_ = pendingSqeCnt + 1; // 设置availableSpace为pendingSqeCnt + 1
    fakeRtsqA5.sqHead_ = 0;                  // 设置sqHead_与sqTail为0，使GetTailToHeadDist返回sqDepth
    fakeRtsqA5.sqTail_ = 0;                  // 以上四个变量控制GetTailToHeadDist方法的输出

    // 模拟QuerySqHead方法
    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(fakeRtsqA5.sqHead_));

    // 执行测试
    bool result = fakeRtsqA5.IsRtsqQueueSpaceSufficient();

    // 验证结果
    EXPECT_FALSE(result);
}

TEST_F(IsRtsqQueueSpaceSufficientTest, Ut_IsRtsqQueueSpaceSufficient_When_HeadEqualTail_ExpectTrue)
{
    // 准备测试数据
    RtsqA5 fakeRtsqA5(fakedevPhyId, fakeStreamId, fakeSqId);
    fakeRtsqA5.pendingSqeCnt = pendingSqeCnt;
    fakeRtsqA5.sqDepth_ = pendingSqeCnt + 2; // 设置availableSpace为pendingSqeCnt + 2
    fakeRtsqA5.sqHead_ = 0;                  // 设置sqHead_与sqTail为0，使GetTailToHeadDist返回sqDepth
    fakeRtsqA5.sqTail_ = 0;                  // 以上四个变量控制GetTailToHeadDist方法的输出

    // 模拟QuerySqHead方法
    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(fakeRtsqA5.sqHead_));

    // 执行测试
    bool result = fakeRtsqA5.IsRtsqQueueSpaceSufficient();

    // 验证结果
    EXPECT_TRUE(result);
}

// 复核分支经RefreshSqHead同步提交解回绕基准, 锚点不更新会滞后整圈
TEST_F(IsRtsqQueueSpaceSufficientTest, Ut_IsRtsqQueueSpaceSufficient_RecheckCommitsBaseSqHead)
{
    RtsqA5 fakeRtsqA5(fakedevPhyId, fakeStreamId, fakeSqId);
    fakeRtsqA5.sqDepth_ = 8;
    fakeRtsqA5.sqHead_ = 3; // 缓存估计: dist(2→3)=1
    fakeRtsqA5.sqTail_ = 2;
    fakeRtsqA5.baseSqHead_ = 3;   // 锚点与缓存head对齐
    fakeRtsqA5.pendingSqeCnt = 1; // 第一级估计1 ≤ 2, 走复核分支

    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(7u));

    bool result = fakeRtsqA5.IsRtsqQueueSpaceSufficient();

    EXPECT_TRUE(result);                   // 复核后dist(2→7)=5 > 2
    EXPECT_EQ(7u, fakeRtsqA5.sqHead_);     // head缓存刷新
    EXPECT_EQ(7u, fakeRtsqA5.baseSqHead_); // 解回绕基准同步提交(3+dist(3→7)=7)
}

TEST_F(RtsqA5Test, launch_task_no_loop_back)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.LaunchTask(); // 没有SQE ，直接返回

    u32 oldTail = 0;
    u32 oldHead = 0;
    rtsq.sqTail_ = oldTail;
    rtsq.sqHead_ = oldHead;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;

    rtsq.RefreshInfo();
    u32 newTail = (rtsq.sqTail_ + rtsq.pendingSqeCnt) % rtsq.sqDepth_;
    rtsq.LaunchTask();

    EXPECT_EQ(rtsq.sqTail_, newTail);
}

TEST_F(RtsqA5Test, launch_task_with_loop_back)
{
    memset_s(mockSq, sizeof(mockSq), 0, sizeof(mockSq));
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 oldTail = AC_SQE_MAX_CNT - 1;
    u32 oldHead = AC_SQE_MAX_CNT - 2;
    rtsq.sqTail_ = oldTail;
    rtsq.sqHead_ = oldHead;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(oldHead));
    rtsq.RefreshInfo();
    rtsq.RefreshInfo();
    u32 newTail = (rtsq.sqTail_ + rtsq.pendingSqeCnt) % rtsq.sqDepth_;
    rtsq.LaunchTask();

    EXPECT_EQ(rtsq.sqTail_, newTail);
}

TEST_F(RtsqA5Test, launch_task_should_dump_generated_sqes_when_task_debug_enabled)
{
    SetPlfDebugConfigValue(PLF_TASK);
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(AC_SQE_MAX_CNT - 1));

    rtsq.NotifyWait(0);
    rtsq.SdmaCopy(0x100, 0x200, 0x300, 0x400);
    rtsq.UbDbSend(UbJettyLiteId(18, 18, 18), 0, 0, nullptr);

    EXPECT_NO_THROW(rtsq.LaunchTask());
}

TEST_F(RtsqA5Test, notify_wait)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 fakeNotifyId = 0;
    rtsq.NotifyWait(fakeNotifyId);
}

TEST_F(RtsqA5Test, notify_wait_with_timeout)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 fakeNotifyId = 0;
    u32 fakeTimeout = 68;
    rtsq.NotifyWait(fakeNotifyId, fakeTimeout);
}

TEST_F(RtsqA5Test, notify_record_local)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 fakeNotifyId = 0;
    rtsq.NotifyRecordLoc(fakeNotifyId);
}

TEST_F(RtsqA5Test, cnt_1ton_notify_wait)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 fakeNotifyId = 0;
    u32 fakeValue = 1;
    rtsq.Cnt1toNNotifyWait(fakeNotifyId, fakeValue);
}

TEST_F(RtsqA5Test, cnt_1ton_notify_record)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 fakeNotifyId = 0;
    u32 fakeValue = 1;
    rtsq.Cnt1toNNotifyRecord(fakeNotifyId, fakeValue);
}

TEST_F(RtsqA5Test, cnt_nto1_notify_wait)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 fakeNotifyId = 0;
    u32 fakeValue = 1;
    rtsq.CntNto1NotifyWait(fakeNotifyId, fakeValue);
}

TEST_F(RtsqA5Test, cnt_nto1_notify_record)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 fakeNotifyId = 0;
    u32 fakeValue = 1;
    rtsq.CntNto1NotifyRecord(fakeNotifyId, fakeValue);
}

TEST_F(RtsqA5Test, sdma_copy)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u64 srcAddr = 0x100;
    u64 dstAddr = 0x200;
    u32 size = 0x300;
    u32 partId = 0x400;
    rtsq.SdmaCopy(srcAddr, dstAddr, size, partId);
}

TEST_F(RtsqA5Test, sdma_reduce)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u64 srcAddr = 0x100;
    u64 dstAddr = 0x200;
    u32 size = 0x300;
    u32 partId = 0x400;
    ReduceIn reduceIn(DataType::INT8, ReduceOp::MAX);
    rtsq.SdmaReduce(srcAddr, dstAddr, size, partId, reduceIn);
}

TEST_F(RtsqA5Test, sdma_reduce_failed)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u64 srcAddr = 0x100;
    u64 dstAddr = 0x200;
    u32 size = 0x300;
    u32 partId = 0x400;
    ReduceIn reduceIn(DataType::UINT8, ReduceOp::MAX);

    EXPECT_THROW(rtsq.SdmaReduce(srcAddr, dstAddr, size, partId, reduceIn), Hccl::InternalException);
}

TEST_F(RtsqA5Test, ub_db_send)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    u32 dieId = 0;
    u32 funcId = 0;
    UbJettyLiteId jettyId(18, 18, 18);
    u32 piVal = 0;
    rtsq.UbDbSend(jettyId, piVal, 0, nullptr);
}

TEST_F(RtsqA5Test, ub_direct_send)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 dieId = 0;
    u32 funcId = 0;
    UbJettyLiteId jettyId(18, 18, 18);
    u32 dwqeSize = 128;
    u8 dwqe[128]{0};
    rtsq.UbDirectSend(jettyId, dwqeSize, dwqe);
}

TEST_F(RtsqA5Test, ub_write_value)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u64 dbAddr = 0;
    u32 piVal = 0;
    rtsq.UbWriteValue(dbAddr, piVal);
}

TEST_F(RtsqA5Test, query_sq_status_by_type)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    GlobalMockObject::reset();
    MOCKER(halSqCqQuery).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(rtsq.QuerySqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_CQE_STATUS), DrvApiException);
}

TEST_F(RtsqA5Test, query_sq_base_addr)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    GlobalMockObject::reset();
    MOCKER(halSqCqQuery).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(rtsq.QuerySqBaseAddr(), DrvApiException);
}

TEST_F(RtsqA5Test, config_sq_status_by_type)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    GlobalMockObject::reset();
    MOCKER(halSqCqConfig).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(rtsq.ConfigSqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_TAIL, 1), DrvApiException);
}

TEST_F(RtsqA5Test, Ut_CopyLocBufToSq_E_INTERNAL)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(2));
    rtsq.sqTail_ = 6;
    rtsq.sqDepth_ = 16;
    rtsq.pendingSqeCnt = 8;
    MOCKER(memcpy_sOptAsm).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_EQ(rtsq.CopySqeBufToSq(rtsq.locBuf), HCCL_E_INTERNAL);

    rtsq.pendingSqeCnt = 11;
    MOCKER(memcpy_sOptAsm).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_EQ(rtsq.CopySqeBufToSq(rtsq.locBuf), HCCL_E_INTERNAL);

    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(10));
    rtsq.sqTail_ = 1;
    rtsq.sqDepth_ = 16;
    rtsq.pendingSqeCnt = 8;
    MOCKER(memcpy_sOptAsm).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_EQ(rtsq.CopySqeBufToSq(rtsq.locBuf), HCCL_E_INTERNAL);
}

TEST_F(RtsqA5Test, Ut_TryLaunchTask_When_PendingSqeCnt_IsZero_Expect_NoAction)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 0;

    rtsq.TryLaunchTask();

    EXPECT_EQ(rtsq.pendingSqeCnt, 0);
}

TEST_F(RtsqA5Test, Ut_TryLaunchTask_When_HasEnoughSpace_Expect_LaunchTask)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    rtsq.sqTail_ = 0;
    rtsq.sqHead_ = 0;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;
    rtsq.pendingSqeCnt = 5;

    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(10));

    rtsq.TryLaunchTask();

    EXPECT_EQ(rtsq.pendingSqeCnt, 0);
}

TEST_F(RtsqA5Test, UT_CheckLaunchTaskStatus_CallbackReturnSuspending_Expect_PendingSqeCntResetAndNoThrow)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqTail_ = 8;
    rtsq.sqHead_ = 16;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;
    rtsq.pendingSqeCnt = 16;
    rtsq.sqFullTimeout_ = 10;

    auto startTime = std::chrono::steady_clock::now();
    auto curTime = startTime + std::chrono::seconds(5);
    rtsq.checkExecStatusCallback_ = [](bool isTimeout) {
        return HCCL_E_SUSPENDING;
    };

    EXPECT_NO_THROW(rtsq.CheckLaunchTaskStatus(startTime, curTime, "Rtsq full"));
    EXPECT_EQ(rtsq.pendingSqeCnt, 0);
}

TEST_F(RtsqA5Test, UT_CheckLaunchTaskStatus_CallbackReturnSuspendingWithTimeout_Expect_NoThrow)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqTail_ = 8;
    rtsq.sqHead_ = 16;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;
    rtsq.pendingSqeCnt = 16;
    rtsq.sqFullTimeout_ = 1;

    auto startTime = std::chrono::steady_clock::now() - std::chrono::seconds(10);
    auto curTime = std::chrono::steady_clock::now();
    rtsq.checkExecStatusCallback_ = [](bool isTimeout) {
        return HCCL_E_SUSPENDING;
    };

    EXPECT_NO_THROW(rtsq.CheckLaunchTaskStatus(startTime, curTime, "Rtsq full"));
    EXPECT_EQ(rtsq.pendingSqeCnt, 0);
}

TEST_F(RtsqA5Test, UT_CheckLaunchTaskStatus_CallbackReturnSuspendingZeroPending_Expect_NormalReturn)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 0;
    rtsq.sqFullTimeout_ = 10;

    auto startTime = std::chrono::steady_clock::now();
    auto curTime = startTime + std::chrono::seconds(5);
    rtsq.checkExecStatusCallback_ = [](bool isTimeout) {
        return HCCL_E_SUSPENDING;
    };

    EXPECT_NO_THROW(rtsq.CheckLaunchTaskStatus(startTime, curTime, "Rtsq full"));
    EXPECT_EQ(rtsq.pendingSqeCnt, 0);
}

TEST_F(RtsqA5Test, UT_CheckLaunchTaskStatus_CallbackReturnErrorWithNoTimeout_Expect_ThrowException)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 16;
    rtsq.sqFullTimeout_ = 10;

    auto startTime = std::chrono::steady_clock::now();
    auto curTime = startTime + std::chrono::seconds(5);
    rtsq.checkExecStatusCallback_ = [](bool isTimeout) {
        return HCCL_E_INTERNAL;
    };

    EXPECT_THROW(rtsq.CheckLaunchTaskStatus(startTime, curTime, "Rtsq full"), Hccl::InternalException);
}

TEST_F(RtsqA5Test, UT_CheckLaunchTaskStatus_CallbackNullWithTimeout_Expect_ThrowException)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 16;
    rtsq.sqFullTimeout_ = 1;
    rtsq.checkExecStatusCallback_ = nullptr;

    auto startTime = std::chrono::steady_clock::now() - std::chrono::seconds(10);
    auto curTime = std::chrono::steady_clock::now();

    EXPECT_THROW(rtsq.CheckLaunchTaskStatus(startTime, curTime, "Rtsq full"), Hccl::InternalException);
}

// ===================== RefreshSqeHeaderTaskField Tests =====================

TEST_F(RtsqA5Test, Ut_RefreshSqeHeaderTaskField_SetsTaskFields)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.taskId_ = 0x00010002U;

    std::vector<uint8_t> sqeBuf(AC_SQE_SIZE, 0);
    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeBuf.data());
    header->rtStreamId = 0;
    header->taskId = 0;

    rtsq.RefreshSqeHeaderTaskField(header);

    EXPECT_EQ(header->rtStreamId, static_cast<uint16_t>(0x0002));
    EXPECT_EQ(header->taskId, static_cast<uint16_t>(0x0001));
}

TEST_F(RtsqA5Test, Ut_RefreshSqeHeaderTaskField_IncrementsTaskId)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    u32 initialTaskId = rtsq.taskId_;

    std::vector<uint8_t> sqeBuf(AC_SQE_SIZE, 0);
    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeBuf.data());

    rtsq.RefreshSqeHeaderTaskField(header);

    EXPECT_EQ(rtsq.taskId_, initialTaskId + 1);
}

TEST_F(RtsqA5Test, Ut_RefreshSqeHeaderTaskField_MultipleCalls_TaskIdIncrementsEachTime)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    u32 initialTaskId = rtsq.taskId_;

    std::vector<uint8_t> sqeBuf(AC_SQE_SIZE, 0);
    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeBuf.data());

    rtsq.RefreshSqeHeaderTaskField(header);
    u32 taskIdAfterFirst = rtsq.taskId_;

    rtsq.RefreshSqeHeaderTaskField(header);
    u32 taskIdAfterSecond = rtsq.taskId_;

    EXPECT_EQ(taskIdAfterFirst, initialTaskId + 1);
    EXPECT_EQ(taskIdAfterSecond, initialTaskId + 2);
}

TEST_F(RtsqA5Test, Ut_RefreshSqeHeaderTaskField_PreservesOtherHeaderFields)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.taskId_ = 0;

    std::vector<uint8_t> sqeBuf(AC_SQE_SIZE, 0);
    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeBuf.data());
    header->type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    header->wrCqe = 1;
    header->numBlocks = 0xABCD;

    rtsq.RefreshSqeHeaderTaskField(header);

    EXPECT_EQ(header->type, static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT));
    EXPECT_EQ(header->wrCqe, 1U);
    EXPECT_EQ(header->numBlocks, static_cast<uint16_t>(0xABCD));
}

// ===================== LaunchNewTask Tests =====================

TEST_F(RtsqA5Test, Ut_LaunchNewTask_PendingSqeCntNotZero_Expect_E_Internal)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 5;
    rtsq.sqTail_ = 0;
    rtsq.sqHead_ = 0;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;

    std::vector<uint8_t> sqeArray(AC_SQE_SIZE * 2, 0);
    EXPECT_EQ(rtsq.LaunchNewTask(sqeArray.data(), 2), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(RtsqA5Test, Ut_LaunchNewTask_Success_UpdatesTailAndResetsPendingSqeCnt)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 0;
    rtsq.sqTail_ = 0;
    rtsq.sqHead_ = 0;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;

    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(static_cast<u32>(0)));

    std::vector<uint8_t> sqeArray(AC_SQE_SIZE * 3, 0);
    u32 sqeCount = 3;
    u32 expectedTail = (rtsq.sqTail_ + sqeCount) % rtsq.sqDepth_;

    EXPECT_EQ(rtsq.LaunchNewTask(sqeArray.data(), sqeCount), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(rtsq.sqTail_, expectedTail);
    EXPECT_EQ(rtsq.pendingSqeCnt, 0U);
}

TEST_F(RtsqA5Test, Ut_LaunchNewTask_Success_CopiesSqeToSqBuffer)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 0;
    rtsq.sqTail_ = 0;
    rtsq.sqHead_ = 0;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;

    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(static_cast<u32>(0)));

    std::vector<uint8_t> sqeArray(AC_SQE_SIZE * 2, 0);
    auto* header0 = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeArray.data());
    header0->type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    auto* header1 = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeArray.data() + AC_SQE_SIZE);
    header1->type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);

    EXPECT_EQ(rtsq.LaunchNewTask(sqeArray.data(), 2), HcclResult::HCCL_SUCCESS);

    auto* sqHeader0 = reinterpret_cast<Rt91095StarsSqeHeader*>(mockSq);
    auto* sqHeader1 = reinterpret_cast<Rt91095StarsSqeHeader*>(mockSq + AC_SQE_SIZE);
    EXPECT_EQ(sqHeader0->type, static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT));
    EXPECT_EQ(sqHeader1->type, static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA));
}

TEST_F(RtsqA5Test, Ut_LaunchNewTask_WithLoopback_Success)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 0;
    rtsq.sqTail_ = AC_SQE_MAX_CNT - 1; // 255
    rtsq.sqHead_ = 4;                  // availableSpace = 256 - (255 - 4) = 5 > 3
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;

    std::vector<uint8_t> sqeArray(AC_SQE_SIZE * 3, 0);
    u32 sqeCount = 3;
    u32 expectedTail = (rtsq.sqTail_ + sqeCount) % rtsq.sqDepth_;

    EXPECT_EQ(rtsq.LaunchNewTask(sqeArray.data(), sqeCount), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(rtsq.sqTail_, expectedTail);
    EXPECT_EQ(rtsq.pendingSqeCnt, 0U);
}

TEST_F(RtsqA5Test, Ut_LaunchNewTask_SingleSqe_Success)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 0;
    rtsq.sqTail_ = 0;
    rtsq.sqHead_ = 0;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;

    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(static_cast<u32>(0)));

    std::vector<uint8_t> sqeArray(AC_SQE_SIZE, 0);
    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeArray.data());
    header->type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);

    EXPECT_EQ(rtsq.LaunchNewTask(sqeArray.data(), 1), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(rtsq.sqTail_, 1U);
    EXPECT_EQ(rtsq.pendingSqeCnt, 0U);

    auto* sqHeader = reinterpret_cast<Rt91095StarsSqeHeader*>(mockSq);
    EXPECT_EQ(sqHeader->type, static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD));
}

TEST_F(RtsqA5Test, Ut_SqeProfilingDisabled_SqeProfStaysZero)
{
    Hccl::SetSqeProfilingEnabled(false);

    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.NotifyWait(0);

    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(rtsq.locBuf);
    EXPECT_EQ(header->sqeProf, 0U);
}

TEST_F(RtsqA5Test, Ut_SqeProfilingEnabled_SqeProfSetToOne)
{
    Hccl::SetSqeProfilingEnabled(true);

    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.NotifyWait(0);

    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(rtsq.locBuf);
    EXPECT_EQ(header->sqeProf, 1U);

    Hccl::SetSqeProfilingEnabled(false);
}

TEST_F(RtsqA5Test, Ut_NotifyWait_WithProfiling_SqeProfSetToOne)
{
    Hccl::SetSqeProfilingEnabled(true);

    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.NotifyWait(0);

    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(rtsq.locBuf);
    EXPECT_EQ(header->sqeProf, 1U);

    Hccl::SetSqeProfilingEnabled(false);
}

TEST_F(RtsqA5Test, Ut_SdmaCopy_WithProfiling_SqeProfSetToOne)
{
    Hccl::SetSqeProfilingEnabled(true);

    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.SdmaCopy(0x100, 0x200, 0x300, 0x400);

    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(rtsq.locBuf);
    EXPECT_EQ(header->sqeProf, 1U);

    Hccl::SetSqeProfilingEnabled(false);
}

TEST_F(RtsqA5Test, Ut_NotifyWait_WithoutProfiling_SqeProfStaysZero)
{
    Hccl::SetSqeProfilingEnabled(false);

    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.NotifyWait(0);

    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(rtsq.locBuf);
    EXPECT_EQ(header->sqeProf, 0U);
}

TEST_F(RtsqA5Test, Ut_SqeProfiling_ToggleOffThenBuild_SqeProfStaysZero)
{
    // 开关推送关闭后，后续构建的SQE不再置位
    Hccl::SetSqeProfilingEnabled(true);
    Hccl::SetSqeProfilingEnabled(false);

    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.NotifyWait(0);

    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(rtsq.locBuf);
    EXPECT_EQ(header->sqeProf, 0U);
}

TEST_F(RtsqA5Test, Ut_NotifyWait_EnabledOnOtherThread_SqeProfStaysZero)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    // 开关为线程级变量：其他线程置1不影响本线程，本线程未经报到口开启时读初始值0
    std::thread toggleThread([]() {
        Hccl::SetSqeProfilingEnabled(true);
    });
    toggleThread.join();

    std::thread buildThread([&rtsq]() {
        rtsq.NotifyWait(0);
    });
    buildThread.join();

    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(rtsq.locBuf);
    EXPECT_EQ(header->sqeProf, 0U);

    // 同线程内开启后构建，置位生效：第二次NotifyWait写入locBuf第二槽位，须取新槽位header断言
    Hccl::SetSqeProfilingEnabled(true);
    rtsq.NotifyWait(0);
    auto* secondHeader = reinterpret_cast<Rt91095StarsSqeHeader*>(rtsq.locBuf + RTSQ_SQE_SIZE);
    EXPECT_EQ(secondHeader->sqeProf, 1U);

    Hccl::SetSqeProfilingEnabled(false);
}

TEST_F(RtsqA5Test, Ut_InlineBuildFunctions_WithProfiling_SqeProfSetToOne)
{
    Hccl::SetSqeProfilingEnabled(true);
    std::vector<uint8_t> sqeBuf(128, 0);
    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeBuf.data());
    auto resetAndBuild = [&sqeBuf, header](auto buildFunc) {
        std::fill(sqeBuf.begin(), sqeBuf.end(), 0);
        buildFunc();
        return header->sqeProf;
    };

    // 内联Build函数10个（.h）
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeNotifyWait(1, 1, 1, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeNotifyWait(1, 1, 1, 68, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeNotifyRecord(1, 1, 1, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeCnt1toNNotifyRecord(1, 1, 1, 8, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeCnt1toNNotifyWait(1, 1, 1, 8, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeCntNto1NotifyRecord(1, 1, 1, 8, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeCntNto1NotifyWait(1, 1, 1, 8, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeSdmaCopy(1, 1, 0x100, 0x200, 0x30, 0, 0, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeUbDbSend(1, 1, Hccl::UbJettyLiteId(0, 1, 2), 3, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeP2pWriteValue(1, 1, 0x1000, 0x5a, sqeBuf.data());
              }));

    Hccl::SetSqeProfilingEnabled(false);
}

TEST_F(RtsqA5Test, Ut_NonInlineBuild_WithProfiling_SqeProfSetToOne)
{
    Hccl::SetSqeProfilingEnabled(true);
    // COND类型SQE约68字节，buffer开128字节覆盖全部SQE类型
    std::vector<uint8_t> sqeBuf(128, 0);
    auto* header1 = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeBuf.data());
    auto resetAndBuild = [&sqeBuf, header1](auto buildFunc) {
        std::fill(sqeBuf.begin(), sqeBuf.end(), 0);
        buildFunc();
        return header1->sqeProf;
    };

    // 非内联Build函数3个（.cc），last=true/false分别覆盖clear与nop分支
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeCCoreNotifyWait(1, 1, 0x2000, 0x2004, true, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeCCoreNotifyWait(1, 1, 0x2000, 0x2004, false, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeCCoreNotifyRecord(1, 1, 0x3000, 0x3004, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeRdmaDbSend(1, 1, 0x4000, 0x1234, sqeBuf.data());
              }));

    Hccl::SetSqeProfilingEnabled(false);
}

TEST_F(RtsqA5Test, Ut_NonInlineBuild_WithoutProfiling_SqeProfStaysZero)
{
    Hccl::SetSqeProfilingEnabled(false);
    std::vector<uint8_t> sqeBuf(128, 0);
    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeBuf.data());

    Hccl::BuildA5SqeCCoreNotifyWait(1, 1, 0x2000, 0x2004, true, sqeBuf.data());
    EXPECT_EQ(header->sqeProf, 0U);

    std::fill(sqeBuf.begin(), sqeBuf.end(), 0);
    Hccl::BuildA5SqeCCoreNotifyRecord(1, 1, 0x3000, 0x3004, sqeBuf.data());
    EXPECT_EQ(header->sqeProf, 0U);

    std::fill(sqeBuf.begin(), sqeBuf.end(), 0);
    Hccl::BuildA5SqeRdmaDbSend(1, 1, 0x4000, 0x1234, sqeBuf.data());
    EXPECT_EQ(header->sqeProf, 0U);
}

/* ---------- RefreshSqHead（解回绕） ---------- */

TEST_F(RtsqA5Test, Ut_RefreshSqHead_NormalAdvance_ExpectBaseUpdated)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.baseSqHead_ = 4;

    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(7u));

    rtsq.RefreshSqHead();
    EXPECT_EQ(7u, rtsq.sqHead_);
    EXPECT_EQ(7u, rtsq.baseSqHead_); // 刷新即提交(4+dist(4→7)=7)
}

TEST_F(RtsqA5Test, Ut_RefreshSqHead_SqHeadWraparound_ExpectBaseUnwrapped)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.baseSqHead_ = 7;

    // 硬件head从7绕回2, 解回绕后base = 7 + 3 = 10(越过回绕点, 不回退)
    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(2u));

    rtsq.RefreshSqHead();
    EXPECT_EQ(2u, rtsq.sqHead_);
    EXPECT_EQ(10u, rtsq.baseSqHead_); // 刷新即提交
}

TEST_F(RtsqA5Test, Ut_RefreshSqHead_ReachMaxSize_ExpectWrapToStart)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.baseSqHeadMaxSize_ = 16; // 测试用小上限
    rtsq.baseSqHead_ = 15;        // 15%8=7, 硬件head绕回0, 前进1格越过上限

    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(0u));

    rtsq.RefreshSqHead();
    EXPECT_EQ(0u, rtsq.sqHead_);
    EXPECT_EQ(0u, rtsq.baseSqHead_); // 15+1=16 回绕到0
}

/* ---------- UbDbSend (seqIdx + DbSendSlotMeta) ---------- */

TEST_F(RtsqA5Test, UbDbSend_ExpectAllRecordedInOrder)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    UbJettyLiteId jettyId(18, 18, 18);
    UbTransportLiteImpl* fakeKey = reinterpret_cast<UbTransportLiteImpl*>(0x5678);

    rtsq.UbDbSend(jettyId, 7, 0, nullptr);
    rtsq.UbDbSend(jettyId, 8, 1, fakeKey);
    rtsq.UbDbSend(jettyId, 9, 2, nullptr);
    rtsq.UbDbSend(jettyId, 10, 3, fakeKey);

    // null与有效transport均记录, seqIdx/piValue按调用顺序
    EXPECT_EQ(4u, rtsq.dbSendTail_);
    u32 slot0 = 0 % rtsq.dbSendDepth_;
    u32 slot1 = 1 % rtsq.dbSendDepth_;
    u32 slot2 = 2 % rtsq.dbSendDepth_;
    u32 slot3 = 3 % rtsq.dbSendDepth_;
    EXPECT_EQ(nullptr, rtsq.dbSendSlots_[slot0].transport);
    EXPECT_EQ(0u, rtsq.dbSendSlots_[slot0].seqIdx);
    EXPECT_EQ(7u, rtsq.dbSendSlots_[slot0].piValue);
    EXPECT_EQ(fakeKey, rtsq.dbSendSlots_[slot1].transport);
    EXPECT_EQ(1u, rtsq.dbSendSlots_[slot1].seqIdx);
    EXPECT_EQ(8u, rtsq.dbSendSlots_[slot1].piValue);
    EXPECT_EQ(nullptr, rtsq.dbSendSlots_[slot2].transport);
    EXPECT_EQ(2u, rtsq.dbSendSlots_[slot2].seqIdx);
    EXPECT_EQ(9u, rtsq.dbSendSlots_[slot2].piValue);
    EXPECT_EQ(fakeKey, rtsq.dbSendSlots_[slot3].transport);
    EXPECT_EQ(3u, rtsq.dbSendSlots_[slot3].seqIdx);
    EXPECT_EQ(10u, rtsq.dbSendSlots_[slot3].piValue);
}

// 环满+SUSPENDING: 跳过记录视同成功, SQE照常构建入批次(挂起不抛异常, 批次由挂起/恢复流程善后)
TEST_F(RtsqA5Test, Ut_UbDbSend_DbSendSlotsFull_Suspending_ExpectSkipRecordOnly)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.dbSendSlots_.assign(16, DbSendSlotMeta{});
    rtsq.dbSendHead_ = 0;
    rtsq.dbSendTail_ = 15; // nextTail回绕到0 == dbSendHead_, 环满
    rtsq.sqFullTimeout_ = 10;
    rtsq.checkExecStatusCallback_ = [](bool isTimeout) {
        return HCCL_E_SUSPENDING;
    };
    u32 taskIdBefore = rtsq.taskId_;

    EXPECT_NO_THROW(rtsq.UbDbSend(UbJettyLiteId(18, 18, 18), 1, 0, nullptr));
    EXPECT_EQ(15u, rtsq.dbSendTail_);  // 跳过记录, 不推进tail
    EXPECT_EQ(1u, rtsq.pendingSqeCnt); // SQE照常构建: CheckLaunchTaskStatus清0后RefreshInfo重新计入
    EXPECT_EQ(taskIdBefore + 1, rtsq.taskId_);
}

// 环满+通信域异常(非SUSPENDING): RecordDbSendSlot收口为错误码, UbDbSend经CHK_RET_THROW上抛InternalException
// 并在构建SQE前中止; 若记录失败被静默吞掉继续下发无记录SQE, 该SQE的CI将永久丢失
TEST_F(RtsqA5Test, Ut_UbDbSend_DbSendSlotsFull_ExecError_ExpectThrowAndNoSqeBuilt)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.dbSendSlots_.assign(16, DbSendSlotMeta{});
    rtsq.dbSendHead_ = 0;
    rtsq.dbSendTail_ = 15; // nextTail回绕到0 == dbSendHead_, 环满
    rtsq.sqFullTimeout_ = 10;
    rtsq.checkExecStatusCallback_ = [](bool isTimeout) {
        return HCCL_E_INTERNAL;
    };
    u32 taskIdBefore = rtsq.taskId_;

    EXPECT_THROW(rtsq.UbDbSend(UbJettyLiteId(18, 18, 18), 1, 0, nullptr), Hccl::InternalException);
    EXPECT_EQ(15u, rtsq.dbSendTail_);                    // 记录未写入, tail不推进
    EXPECT_EQ(0u, rtsq.pendingSqeCnt);                   // SQE未构建入批次: 异常在BuildA5SqeUbDbSend前中止
    EXPECT_EQ(taskIdBefore, rtsq.taskId_);               // taskId未消耗
    EXPECT_EQ(nullptr, rtsq.dbSendSlots_[15].transport); // 槽位未被写入
}

TEST_F(RtsqA5Test, Ut_RecordDbSendSlot_RecordsWrapSlotIdxAndAdvancesTail)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.dbSendSlots_.assign(16, DbSendSlotMeta{});
    rtsq.sqTail_ = 2;
    rtsq.sqTailTotal_ = 2; // 绝对tail与环内tail对齐(未回绕)
    rtsq.dbSendHead_ = 0;
    rtsq.dbSendTail_ = 0;
    UbTransportLiteImpl* key = reinterpret_cast<UbTransportLiteImpl*>(0x9999);

    EXPECT_EQ(HCCL_SUCCESS, rtsq.RecordDbSendSlot(key, 3, 5, 100)); // wrapSlotIdx = 绝对tail(2) + 批内偏移(3) = 5

    EXPECT_EQ(1u, rtsq.dbSendTail_);
    EXPECT_EQ(key, rtsq.dbSendSlots_[0].transport);
    EXPECT_EQ(5u, rtsq.dbSendSlots_[0].wrapSlotIdx);
    EXPECT_EQ(5u, rtsq.dbSendSlots_[0].seqIdx);
    EXPECT_EQ(100u, rtsq.dbSendSlots_[0].piValue);
}

// 环满+通信域异常(非SUSPENDING): 内部经TRY_CATCH_RETURN收口, 以错误码返回不再上抛
TEST_F(RtsqA5Test, Ut_RecordDbSendSlot_DbSendSlotsFull_ExecError_Expect_ReturnErrorCode)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.dbSendSlots_.assign(16, DbSendSlotMeta{});
    rtsq.dbSendHead_ = 0;
    rtsq.dbSendTail_ = 15;
    rtsq.sqFullTimeout_ = 10;
    rtsq.checkExecStatusCallback_ = [](bool isTimeout) {
        return HCCL_E_INTERNAL;
    };

    EXPECT_EQ(HCCL_E_INTERNAL, rtsq.RecordDbSendSlot(nullptr, 0, 0, 0)); // InternalException收口为错误码
    EXPECT_EQ(15u, rtsq.dbSendTail_);                                    // 记录未写入
}

// 环满+SUSPENDING: 跳过记录视同成功
TEST_F(RtsqA5Test, Ut_RecordDbSendSlot_DbSendSlotsFull_Suspending_ExpectSkipRecord)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.dbSendSlots_.assign(16, DbSendSlotMeta{});
    rtsq.dbSendHead_ = 0;
    rtsq.dbSendTail_ = 15;
    rtsq.sqFullTimeout_ = 10;
    rtsq.checkExecStatusCallback_ = [](bool isTimeout) {
        return HCCL_E_SUSPENDING;
    };

    EXPECT_EQ(HCCL_SUCCESS, rtsq.RecordDbSendSlot(nullptr, 0, 0, 0)); // 挂起跳过记录, 视同成功
    EXPECT_EQ(15u, rtsq.dbSendTail_);
    EXPECT_EQ(0u, rtsq.pendingSqeCnt); // SUSPENDING时清空挂起SQE
}

// 环满等待至后台出队腾出空位: 记录成功写入回绕槽位
TEST_F(RtsqA5Test, Ut_RecordDbSendSlot_DbSendSlotsFull_WaitUntilDrained_ExpectRecordSuccess)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.dbSendSlots_.assign(16, DbSendSlotMeta{});
    rtsq.sqTail_ = 2;
    rtsq.sqTailTotal_ = 2; // 绝对tail与环内tail对齐(未回绕)
    rtsq.dbSendHead_ = 0;
    rtsq.dbSendTail_ = 15;
    rtsq.sqFullTimeout_ = 10;
    UbTransportLiteImpl* key = reinterpret_cast<UbTransportLiteImpl*>(0x9999);
    // 模拟一轮等待后后台线程出队腾出空位
    rtsq.checkOpExecStatusCallback_ = [&rtsq]() {
        rtsq.dbSendHead_ = 5;
    };

    EXPECT_EQ(HCCL_SUCCESS, rtsq.RecordDbSendSlot(key, 3, 5, 100)); // wrapSlotIdx = 2+3 = 5, 等到空位记录成功

    EXPECT_EQ(0u, rtsq.dbSendTail_); // 从15回绕到0
    EXPECT_EQ(key, rtsq.dbSendSlots_[15].transport);
    EXPECT_EQ(5u, rtsq.dbSendSlots_[15].wrapSlotIdx);
    EXPECT_EQ(5u, rtsq.dbSendSlots_[15].seqIdx);
    EXPECT_EQ(100u, rtsq.dbSendSlots_[15].piValue);
}

/* ---------- PollCompletion ---------- */

TEST_F(RtsqA5Test, PollCompletion_ExpectCorrectConsume)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.lastHead_ = 4;
    rtsq.baseSqHead_ = 4; // 发送侧解回绕基准与出队进度对齐
    rtsq.dbSendSlots_.assign(16, DbSendSlotMeta{});
    UbTransportLiteImpl* key1 = reinterpret_cast<UbTransportLiteImpl*>(0x1111);
    UbTransportLiteImpl* key2 = reinterpret_cast<UbTransportLiteImpl*>(0x2222);

    // 空数组: 无可消费slot, no-op
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().will(returnValue(4u));
    EXPECT_NO_THROW(rtsq.PollCompletion());
    EXPECT_EQ(0u, rtsq.dbSendHead_);
    EXPECT_EQ(4u, rtsq.lastHead_);
    GlobalMockObject::reset();

    // head无变化: 不消费
    rtsq.dbSendTail_ = 1;
    rtsq.dbSendSlots_[0] = {key1, 4, 0, 100};
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().will(returnValue(4u));
    rtsq.PollCompletion();
    EXPECT_EQ(0u, rtsq.dbSendHead_);
    EXPECT_EQ(4u, rtsq.lastHead_);
    GlobalMockObject::reset();

    // 部分消费: newHead只到5, 消费slot0后停止, slot1保留
    rtsq.dbSendTail_ = 2;
    rtsq.dbSendSlots_[1] = {key1, 6, 1, 101};
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().will(returnValue(5u));
    rtsq.PollCompletion();
    EXPECT_EQ(1u, rtsq.dbSendHead_);
    EXPECT_EQ(5u, rtsq.lastHead_);
    GlobalMockObject::reset();

    // 跳过非DbSend slot + 同transport连续累积: 5无记录被跳过, 6匹配slot1(同key1)被消费
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().will(returnValue(7u));
    rtsq.PollCompletion();
    EXPECT_EQ(2u, rtsq.dbSendHead_);
    EXPECT_EQ(7u, rtsq.lastHead_);
    GlobalMockObject::reset();

    // 回绕 + 切换transport: 7匹配slot2(key2), lastHead在双倍深度空间推进到8(越过sqDepth_回绕点)
    rtsq.dbSendTail_ = 3;
    rtsq.dbSendSlots_[2] = {key2, 7, 2, 102};
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().will(returnValue(0u));
    rtsq.PollCompletion();
    EXPECT_EQ(3u, rtsq.dbSendHead_);
    EXPECT_EQ(8u, rtsq.lastHead_);
    GlobalMockObject::reset();
}

// 两轮poll之间head随提交累积多圈(此处2圈), 硬件head数值与lastHead_再次撞值
// baseSqHead_解回绕后newHead=base+2圈, 全部出队
TEST_F(RtsqA5Test, PollCompletion_FullLapAdvance_ExpectAllSlotsConsumed)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.dbSendSlots_.assign(16, DbSendSlotMeta{});
    UbTransportLiteImpl* key = reinterpret_cast<UbTransportLiteImpl*>(0x3333);

    // 上一轮poll后出队进度lastHead_=4, 发送侧随提交将baseSqHead_累积2圈到20(4+2*8)
    rtsq.lastHead_ = 4;
    rtsq.baseSqHead_ = 20;
    // 硬件head当前= 20 % 8 = 4, 数值上与lastHead_相同
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().will(returnValue(4u));

    // 8条记录覆盖绝对序号12~19, 环内槽位依次为4,5,6,7,0,1,2,3, 跨两个回绕点(绝对8/16)
    rtsq.dbSendTail_ = 8;
    rtsq.dbSendSlots_[0] = {key, 12, 0, 200};
    rtsq.dbSendSlots_[1] = {key, 13, 1, 201};
    rtsq.dbSendSlots_[2] = {key, 14, 2, 202};
    rtsq.dbSendSlots_[3] = {key, 15, 3, 203};
    rtsq.dbSendSlots_[4] = {key, 16, 4, 204};
    rtsq.dbSendSlots_[5] = {key, 17, 5, 205};
    rtsq.dbSendSlots_[6] = {key, 18, 6, 206};
    rtsq.dbSendSlots_[7] = {key, 19, 7, 207};

    rtsq.PollCompletion();

    EXPECT_EQ(8u, rtsq.dbSendHead_); // 2圈8条全部出队
    EXPECT_EQ(20u, rtsq.lastHead_);  // head追踪空间内推进2圈(4→20)
    GlobalMockObject::reset();
}

// 追踪空间回绕点: lastHead_行进至baseSqHeadMaxSize_边界, newHead回绕到小值(生产环境baseSqHeadMaxSize_约42.9亿,
// SQE以微秒级速率下发, 长稳运行必然撞上回绕点); 守卫须把跨回绕差值判为前进(回退量为巨大值, 不得误判为锚点滞后),
// walk经(lastHead+1==baseSqHeadMaxSize_)分支回绕推进, 跨回绕点记录正常出队
TEST_F(RtsqA5Test, PollCompletion_TrackingSpaceWraparound_ExpectWrappedAdvanceAndDequeue)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.baseSqHeadMaxSize_ = 16; // 测试用小上限, 等价模拟追踪空间回绕点(16即回绕到0)
    rtsq.dbSendSlots_.assign(16, DbSendSlotMeta{});
    UbTransportLiteImpl* key = reinterpret_cast<UbTransportLiteImpl*>(0x8888);

    // 上轮poll进度lastHead_=13(邻近回绕点); 锚点已推进至15, 硬件head环位2:
    // basePos=7, step=3, newHead=15+3=18回绕到2(数值上小于lastHead_, 实为跨回绕点前进)
    rtsq.lastHead_ = 13;
    rtsq.baseSqHead_ = 15;
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().will(returnValue(2u));

    // 5条记录跨回绕点: 绝对序号13,14,15 | 0,1(越过baseSqHeadMaxSize_)
    rtsq.dbSendTail_ = 5;
    rtsq.dbSendSlots_[0] = {key, 13, 0, 700};
    rtsq.dbSendSlots_[1] = {key, 14, 1, 701};
    rtsq.dbSendSlots_[2] = {key, 15, 2, 702};
    rtsq.dbSendSlots_[3] = {key, 0, 3, 703};
    rtsq.dbSendSlots_[4] = {key, 1, 4, 704};

    rtsq.PollCompletion();

    EXPECT_EQ(5u, rtsq.dbSendHead_); // 跨回绕点5条全部出队
    EXPECT_EQ(2u, rtsq.lastHead_);   // lastHead_经15回绕到0推进到2(守卫误判回退时将停在13)
    GlobalMockObject::reset();
}

// DbSend SQE已记录但未提交(目标槽位ring 4, 绝对序号12),
// 轮询时head仅完成该槽位上一圈占用者(位置4), 旧取余匹配(4%8==4)会误判完成提前出队;
// 新精确匹配(12!=4)不出队, 待SQE真正完成(位置12)才出队
TEST_F(RtsqA5Test, PollCompletion_PendingSqeNotPrematurelyDequeued)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.dbSendSlots_.assign(16, DbSendSlotMeta{});
    UbTransportLiteImpl* key = reinterpret_cast<UbTransportLiteImpl*>(0x4444);

    // 上一轮poll进度lastHead_=4; tail已整圈回绕(绝对序号12, 目标槽位ring 4)
    rtsq.lastHead_ = 4;
    rtsq.baseSqHead_ = 4;
    rtsq.sqTailTotal_ = 12;
    // 已记录未提交的DbSend: 绝对序号12
    rtsq.dbSendTail_ = 1;
    rtsq.dbSendSlots_[0] = {key, 12, 0, 300};

    // head前进到5: 位置4是目标槽位上一圈占用者的完成位置, 不得误匹配未提交的记录
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().will(returnValue(5u));
    rtsq.PollCompletion();
    EXPECT_EQ(0u, rtsq.dbSendHead_); // 未提前出队
    EXPECT_EQ(5u, rtsq.lastHead_);   // 非DbSend位置正常跳过
    GlobalMockObject::reset();

    // 发送侧解回绕基准推进到12, SQE提交并真正完成后head到13: 位置12精确匹配出队
    rtsq.baseSqHead_ = 12;
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().will(returnValue(5u));
    rtsq.PollCompletion();
    EXPECT_EQ(1u, rtsq.dbSendHead_); // 真正完成后出队
    EXPECT_EQ(13u, rtsq.lastHead_);
    GlobalMockObject::reset();
}

// head解回绕走ReadUnwrappedSqHead只读函数; sqHead_缓存仅由数据面RefreshSqHead刷新
TEST_F(RtsqA5Test, PollCompletion_NotModifySqHeadMember)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.dbSendSlots_.assign(16, DbSendSlotMeta{});
    UbTransportLiteImpl* key = reinterpret_cast<UbTransportLiteImpl*>(0x5555);

    rtsq.lastHead_ = 4;
    rtsq.baseSqHead_ = 4;
    rtsq.dbSendTail_ = 1;
    rtsq.dbSendSlots_[0] = {key, 4, 0, 400};

    rtsq.sqHead_ = 4; // 成员哨兵: 修复前PollCompletion经RefreshSqHead会将其改写为硬件查询值6
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().will(returnValue(6u));
    rtsq.PollCompletion();

    EXPECT_EQ(4u, rtsq.sqHead_);     // sqHead_未被后台线程触碰
    EXPECT_EQ(1u, rtsq.dbSendHead_); // 正常出队slot0(位置4)
    EXPECT_EQ(6u, rtsq.lastHead_);   // 解回绕head正常推进到6
    GlobalMockObject::reset();
}

// baseSqHead_滞后整圈时step回绕使newHead回退到lastHead_之前(回退量<sqDepth),
// 回退守卫命中则跳过本轮出队(记录保留), 避免walk以环空为唯一出口一次性扫荡整环误出队在途DbSend
TEST_F(RtsqA5Test, PollCompletion_AnchorLagRegression_ExpectSkipAndKeepRecords)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.dbSendSlots_.assign(16, DbSendSlotMeta{});
    UbTransportLiteImpl* key = reinterpret_cast<UbTransportLiteImpl*>(0x6666);

    // 锚点滞后: baseSqHead_=4未随进度提交, 上轮poll进度lastHead_=11(超出锚点窗口[4,12)),
    // 两条记录位于真实空间绝对序号12/13(在途未完成)
    rtsq.lastHead_ = 11;
    rtsq.baseSqHead_ = 4;
    rtsq.dbSendTail_ = 2;
    rtsq.dbSendSlots_[0] = {key, 12, 0, 600};
    rtsq.dbSendSlots_[1] = {key, 13, 1, 601};

    // 硬件head越过锚点环位圈边界: step回绕, newHead=4+1=5 < lastHead_=11, 回退量6 < sqDepth
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().will(returnValue(5u));
    rtsq.PollCompletion();
    EXPECT_EQ(0u, rtsq.dbSendHead_); // 守卫生效: 不出队(修复前12/13被提前出队)
    EXPECT_EQ(11u, rtsq.lastHead_);  // 出队进度不动
    EXPECT_EQ(2u, rtsq.dbSendTail_); // 记录保留
    EXPECT_EQ(12u, rtsq.dbSendSlots_[0].wrapSlotIdx);
    EXPECT_EQ(13u, rtsq.dbSendSlots_[1].wrapSlotIdx);
    GlobalMockObject::reset();

    // 锚点重新提交后(baseSqHead_=14): newHead=14+7=21在lastHead_前方, 回退量为巨值, 守卫不误伤,
    // walk [11,21) 正常出队12/13
    rtsq.baseSqHead_ = 14;
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().will(returnValue(5u));
    rtsq.PollCompletion();
    EXPECT_EQ(2u, rtsq.dbSendHead_); // 正常出队
    EXPECT_EQ(21u, rtsq.lastHead_);  // 出队进度推进到newHead
    GlobalMockObject::reset();
}

// 回绕点+锚点滞后叠加: 滞后锚点窗口[12,20)跨baseSqHeadMaxSize_边界, newHead伪回退到lastHead_之前且数值上更大,
// 回退量必须经lastHead_+baseSqHeadMaxSize_-newHead在追踪空间内计算(直接u32相减下溢为巨大值会漏判守卫,
// 误出队在途DbSend并腐化lastHead_), 守卫命中跳过本轮保留记录; 锚点恢复重锚后正常出队
TEST_F(RtsqA5Test, PollCompletion_WraparoundAnchorLag_ExpectGuardSkipThenRecover)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.baseSqHeadMaxSize_ = 16; // 测试用小上限, 等价模拟追踪空间回绕点16≡0
    rtsq.dbSendSlots_.assign(16, DbSendSlotMeta{});
    UbTransportLiteImpl* key = reinterpret_cast<UbTransportLiteImpl*>(0x9999);

    // poll进度lastHead_=2(已越过回绕点), 在途记录位于2,3; 锚点滞后至12(窗口跨回绕点),
    // 硬件head环位6被滞后锚点映射为伪newHead=14(数值上>lastHead_, 实为回退4)
    rtsq.lastHead_ = 2;
    rtsq.baseSqHead_ = 12;
    rtsq.dbSendTail_ = 2;
    rtsq.dbSendSlots_[0] = {key, 2, 0, 800};
    rtsq.dbSendSlots_[1] = {key, 3, 1, 801};
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().will(returnValue(6u));

    rtsq.PollCompletion();
    EXPECT_EQ(0u, rtsq.dbSendHead_); // 守卫生效: 不出队(直接u32相减漏判时会误出队2/3并腐化lastHead_)
    EXPECT_EQ(2u, rtsq.lastHead_);   // 出队进度不动
    EXPECT_EQ(2u, rtsq.dbSendTail_); // 记录保留
    GlobalMockObject::reset();

    // 锚点恢复重锚到回绕点之后(baseSqHead_=4): newHead=6在lastHead_前方, 回退量为巨值, 守卫不误伤,
    // walk [2,6) 正常出队2/3
    rtsq.baseSqHead_ = 4;
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().will(returnValue(6u));
    rtsq.PollCompletion();
    EXPECT_EQ(2u, rtsq.dbSendHead_); // 正常出队
    EXPECT_EQ(6u, rtsq.lastHead_);   // 出队进度推进到newHead
    GlobalMockObject::reset();
}

/* ---------- Reset ---------- */

TEST_F(RtsqA5Test, Reset_ClearsBackpressureState)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqDepth_ = 8;
    rtsq.dbSendDepth_ = 16;
    rtsq.lastHead_ = 5;
    rtsq.dbSendHead_ = 0;
    rtsq.dbSendTail_ = 1;
    rtsq.dbSendSlots_.assign(16, DbSendSlotMeta{});
    rtsq.dbSendSlots_[0] = {reinterpret_cast<UbTransportLiteImpl*>(0x7777), 5, 0, 800};

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().will(returnValue(3u));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().will(returnValue(3u));
    MOCKER_CPP(&RtsqBase::QuerySqDepth).stubs().will(returnValue(8u));
    MOCKER_CPP(&RtsqBase::QuerySqBaseAddr).stubs().will(returnValue(0u));

    rtsq.Reset(true);

    EXPECT_EQ(0u, rtsq.dbSendHead_);
    EXPECT_EQ(0u, rtsq.dbSendTail_);
    EXPECT_EQ(3u, rtsq.lastHead_);
    EXPECT_EQ(3u, rtsq.baseSqHead_);
    // Reset后绝对tail与baseSqHead_对齐(重置时head==tail, SQ为空)
    EXPECT_EQ(3u, rtsq.sqTailTotal_);
    // Reset按当前sqDepth_重算8倍深度; 用相对断言, 不依赖QuerySqDepth桩与SetUp桩的抢占顺序
    EXPECT_EQ(rtsq.sqDepth_ << 3, rtsq.dbSendDepth_);
    // 同步重算追踪空间上限: u32内sqDepth_最大整数倍再减一个sqDepth_余量
    EXPECT_EQ((UINT32_MAX / rtsq.sqDepth_) * rtsq.sqDepth_ - rtsq.sqDepth_, rtsq.baseSqHeadMaxSize_);
    GlobalMockObject::reset();
}

// ===================== SQE Transport Context Propagation Tests =====================
// 目的: 校验 RefreshInfo 快照 pendingSqe* 后立即复位 current*, 同批次内未显式设置上下文的
//       后续 SQE 获得默认值 (DFX_INVALID_U64/LINK_ONCHIP/DFX_TRANSPORT_TYPE_LOCAL),
//       覆盖方案2 修复的同批次内残留窗口.

TEST_F(RtsqA5Test, RefreshInfo_ResetsCurrentTransportContextAfterSnapshot)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    const u64 nonDefaultHandle = 0x1234ABCDEF5678ULL;
    const u8 nonDefaultLinkType = static_cast<u8>(DfxLinkTypeVal::LINK_ROCE);
    const u8 nonDefaultTransportType = static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE);

    rtsq.SetCurrentTransportContext(nonDefaultHandle, nonDefaultLinkType, nonDefaultTransportType);
    rtsq.NotifyWait(0);

    EXPECT_EQ(rtsq.pendingSqeTransports_[0], nonDefaultHandle);
    EXPECT_EQ(rtsq.pendingSqeLinkTypes_[0], nonDefaultLinkType);
    EXPECT_EQ(rtsq.pendingSqeTransportTypes_[0], nonDefaultTransportType);
    EXPECT_EQ(rtsq.currentTransport_, DFX_INVALID_U64);
    EXPECT_EQ(rtsq.currentLinkType_, static_cast<u8>(DfxLinkTypeVal::LINK_ONCHIP));
    EXPECT_EQ(rtsq.currentTransportType_, static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL));
}

TEST_F(RtsqA5Test, RefreshInfo_LocalTaskAfterTransportTaskGetsDefaultContext)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    const u64 nonDefaultHandle = 0x1234ABCDEF5678ULL;
    const u8 nonDefaultLinkType = static_cast<u8>(DfxLinkTypeVal::LINK_ROCE);
    const u8 nonDefaultTransportType = static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE);

    rtsq.SetCurrentTransportContext(nonDefaultHandle, nonDefaultLinkType, nonDefaultTransportType);
    rtsq.NotifyWait(0);
    rtsq.SdmaCopy(0x100, 0x200, 0x300, 0);

    EXPECT_EQ(rtsq.pendingSqeTransports_[0], nonDefaultHandle);
    EXPECT_EQ(rtsq.pendingSqeLinkTypes_[0], nonDefaultLinkType);
    EXPECT_EQ(rtsq.pendingSqeTransportTypes_[0], nonDefaultTransportType);
    EXPECT_EQ(rtsq.pendingSqeTransports_[1], DFX_INVALID_U64);
    EXPECT_EQ(rtsq.pendingSqeLinkTypes_[1], static_cast<u8>(DfxLinkTypeVal::LINK_ONCHIP));
    EXPECT_EQ(rtsq.pendingSqeTransportTypes_[1], static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL));
}

TEST_F(RtsqA5Test, DefaultTransportTypeIsLocal)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    EXPECT_EQ(rtsq.currentTransport_, DFX_INVALID_U64);
    EXPECT_EQ(rtsq.currentLinkType_, static_cast<u8>(DfxLinkTypeVal::LINK_ONCHIP));
    EXPECT_EQ(rtsq.currentTransportType_, static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL));
}
