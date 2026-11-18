/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include <memory>
#include <cstring>

#define private public
#define protected public

// 阻止旧版 profiling_handler.h / profiling_reporter.h 被重复包含
// （stub 链接旧版，新版 dfx_profiling_handler.h 定义了相同的 struct/const）
#define HCCL_PROFILING_HANDLER_H
#define HCCL_PROFILING_REPORTER_H

#include "dfx_dlprof_function.h"
#include "dfx_profiling_handler.h"
#include "dfx_profiling_reporter.h"
#include "mirror_task_manager.h"
#undef private
#undef protected

#include "task_info.h"
#include "task_param.h"
#include "circular_queue.h"
#include "vector_queue.h"
// SetLaunchKernelMode/IsLaunchKernelMode：UT链接hccl_llt内真实workflow.cc，用例可直接操控生产代码
// 读取的GE下发开关（进程级全局）
#include "workflow_pub.h"

using namespace Hccl;

// ===========================================================================
// DfxDlProfFunction UT
// ===========================================================================

class DfxDlProfFunctionTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& inst = DfxDlProfFunction::GetInstance();
        inst.initializedFlag_.store(false, std::memory_order_release);
        inst.handle_ = nullptr;
    }
    void TearDown() override { GlobalMockObject::verify(); }

    DfxDlProfFunction& inst_ = DfxDlProfFunction::GetInstance();
};

TEST_F(DfxDlProfFunctionTest, Ut_StubInit_Expect_StubFuncsNonNull)
{
    EXPECT_TRUE(inst_.dlMsprofRegisterCallback);
    EXPECT_TRUE(inst_.dlMsprofRegTypeInfo);
    EXPECT_TRUE(inst_.dlMsprofReportApi);
    EXPECT_TRUE(inst_.dlMsprofReportCompactInfo);
    EXPECT_TRUE(inst_.dlMsprofReportAdditionalInfo);
    EXPECT_TRUE(inst_.dlMsprofStr2Id);
    EXPECT_TRUE(inst_.dlMsprofSysCycleTime);
}

TEST_F(DfxDlProfFunctionTest, Ut_Init_When_AlreadyInit_Expect_Success)
{
    inst_.initializedFlag_.store(true, std::memory_order_release);
    EXPECT_EQ(inst_.DfxDlProfFunctionInit(), HCCL_SUCCESS);
}

TEST_F(DfxDlProfFunctionTest, Ut_Init_When_NotInit_NoLib_Expect_NoThrow)
{
    inst_.handle_ = nullptr;
    EXPECT_NO_THROW(inst_.DfxDlProfFunctionInit());
}

TEST_F(DfxDlProfFunctionTest, Ut_StubFuncs_Expect_NonNull)
{
    EXPECT_TRUE(inst_.dlMsprofSysCycleTime);
    EXPECT_TRUE(inst_.dlMsprofStr2Id);
    EXPECT_TRUE(inst_.dlMsprofReportApi);
}

// ===========================================================================
// DfxProfilingHandler UT
// ===========================================================================

class DfxProfilingHandlerTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& h = DfxProfilingHandler::GetInstance();
        h.initializedFlag_.store(false);
        h.enableHostApi_.store(false);
        h.enableHcclNode_.store(false);
        h.enableHcclL0_.store(false);
        h.enableHcclL1_.store(false);
        h.cachedAlgTypeHashId_.store(0);
        h.cacheTaskInfos_.clear();
        while (!h.cachedTaskApiInfo_.empty()) {
            h.cachedTaskApiInfo_.pop();
        }
        while (!h.cachedAclApiInfo_.empty()) {
            h.cachedAclApiInfo_.pop();
        }
        while (!h.cacheHcclOpInfo_.empty()) {
            h.cacheHcclOpInfo_.pop();
        }
        while (!h.cacheHcclAdditionInfo_.empty()) {
            h.cacheHcclAdditionInfo_.pop();
        }
    }
    void TearDown() override { GlobalMockObject::verify(); }

    DfxProfilingHandler& handler_ = DfxProfilingHandler::GetInstance();
};

TEST_F(DfxProfilingHandlerTest, Ut_Init_When_AlreadyInit_Expect_Success)
{
    handler_.initializedFlag_.store(true);
    EXPECT_EQ(handler_.Init(), HCCL_SUCCESS);
}

TEST_F(DfxProfilingHandlerTest, Ut_GetProfHashId_When_NameNull_Expect_Invalid)
{
    EXPECT_EQ(handler_.GetProfHashId(nullptr, 10), 0xFFFFFFFFFFFFFFFFULL);
}

TEST_F(DfxProfilingHandlerTest, Ut_GetProfHashId_When_LenZero_Expect_Invalid)
{
    EXPECT_EQ(handler_.GetProfHashId("test", 0), 0xFFFFFFFFFFFFFFFFULL);
}

TEST_F(DfxProfilingHandlerTest, Ut_GetProfHashId_When_ValidName_Expect_StubReturn0)
{
    EXPECT_EQ(handler_.GetProfHashId("allreduce", 9), 0ULL);
}

TEST_F(DfxProfilingHandlerTest, Ut_GetTaskTypeValue_When_DpuTask_Expect_DpuInfo)
{
    EXPECT_EQ(
        handler_.GetTaskTypeValue(TaskParamType::TASK_DPU_INLINE_WRITE),
        static_cast<uint32_t>(ProfTaskType::TASK_DPU_HCCL_INFO));
}

TEST_F(DfxProfilingHandlerTest, Ut_GetTaskTypeValue_When_OtherTask_Expect_HcclInfo)
{
    EXPECT_EQ(handler_.GetTaskTypeValue(TaskParamType::TASK_SDMA), static_cast<uint32_t>(ProfTaskType::TASK_HCCL_INFO));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskApi_When_DpuTask_Expect_EarlyReturn)
{
    EXPECT_NO_THROW(handler_.ReportHcclTaskApi(TaskParamType::TASK_DPU_KERNEL, 100, 200, true, false, false));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskApi_When_AicpuKernel_Expect_EarlyReturn)
{
    EXPECT_NO_THROW(handler_.ReportHcclTaskApi(TaskParamType::TASK_AICPU_KERNEL, 100, 200, true, false, false));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskApi_When_CachedReq_Expect_Pushed)
{
    handler_.enableHcclNode_.store(false);
    size_t before = handler_.cachedTaskApiInfo_.size();
    EXPECT_NO_THROW(handler_.ReportHcclTaskApi(TaskParamType::TASK_SDMA, 100, 200, true, true, false));
    EXPECT_EQ(handler_.cachedTaskApiInfo_.size(), before + 1);
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskApi_When_NodeDisabled_Expect_Return)
{
    handler_.enableHcclNode_.store(false);
    handler_.enableHcclL1_.store(true);
    EXPECT_NO_THROW(handler_.ReportHcclTaskApi(TaskParamType::TASK_SDMA, 100, 200, true, false, false));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskApi_When_NodeEnabled_L1Disabled_IgnoreFalse_Expect_Return)
{
    handler_.enableHcclNode_.store(true);
    handler_.enableHcclL1_.store(false);
    EXPECT_NO_THROW(handler_.ReportHcclTaskApi(TaskParamType::TASK_SDMA, 100, 200, true, false, false));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskApi_When_NodeDisabled_L1Enabled_IgnoreTrue_Expect_Return)
{
    handler_.enableHcclNode_.store(false);
    handler_.enableHcclL1_.store(true);
    EXPECT_NO_THROW(handler_.ReportHcclTaskApi(TaskParamType::TASK_SDMA, 100, 200, true, false, true));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskDetails_When_L1Off_NotCached_Expect_Return)
{
    handler_.enableHcclL1_.store(false);
    TaskInfo taskInfo(0, 0, 0, {}, nullptr, false);
    EXPECT_NO_THROW(handler_.ReportHcclTaskDetails(taskInfo, false));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskDetails_When_DfxOpInfoNull_Expect_Return)
{
    handler_.enableHcclL1_.store(true);
    TaskInfo taskInfo(0, 0, 0, {}, nullptr, false);
    EXPECT_NO_THROW(handler_.ReportHcclTaskDetails(taskInfo, true));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskDetails_When_CommNull_Expect_Return)
{
    handler_.enableHcclL1_.store(true);
    auto opInfo = std::make_shared<DfxOpInfo>();
    opInfo->comm_ = nullptr;
    TaskInfo taskInfo(0, 0, 0, {}, opInfo, false);
    EXPECT_NO_THROW(handler_.ReportHcclTaskDetails(taskInfo, true));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskDetailsBatch_When_Empty_Expect_Return)
{
    std::vector<TaskInfo*> tasks;
    EXPECT_NO_THROW(handler_.ReportHcclTaskDetailsBatch(tasks, false));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskDetailsBatch_When_DfxOpInfoNull_Expect_Return)
{
    handler_.enableHcclL1_.store(true);
    TaskInfo taskInfo(0, 0, 0, {}, nullptr, false);
    std::vector<TaskInfo*> tasks{&taskInfo};
    EXPECT_NO_THROW(handler_.ReportHcclTaskDetailsBatch(tasks, true));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskDetailsBatch_When_CommNull_Expect_Return)
{
    handler_.enableHcclL1_.store(true);
    auto opInfo = std::make_shared<DfxOpInfo>();
    opInfo->comm_ = nullptr;
    TaskInfo taskInfo(0, 0, 0, {}, opInfo, false);
    std::vector<TaskInfo*> tasks{&taskInfo};
    EXPECT_NO_THROW(handler_.ReportHcclTaskDetailsBatch(tasks, true));
}

TEST_F(DfxProfilingHandlerTest, Ut_CommandHandle_When_DataNull_Expect_Para)
{
    EXPECT_EQ(handler_.CommandHandle(0, nullptr, 100), HCCL_E_PARA);
}

TEST_F(DfxProfilingHandlerTest, Ut_CommandHandle_When_LenTooSmall_Expect_Para)
{
    rtProfCommandHandle_t data{};
    EXPECT_EQ(handler_.CommandHandle(0, &data, 1), HCCL_E_PARA);
}

TEST_F(DfxProfilingHandlerTest, Ut_CommandHandle_When_BadRtType_Expect_Para)
{
    rtProfCommandHandle_t data{};
    EXPECT_EQ(handler_.CommandHandle(999, &data, sizeof(data)), HCCL_E_PARA);
}

TEST_F(DfxProfilingHandlerTest, Ut_CommandHandle_When_Start_Expect_Success)
{
    rtProfCommandHandle_t data{};
    data.type = PROF_COMMANDHANDLE_TYPE_START;
    EXPECT_EQ(
        handler_.CommandHandle(static_cast<uint32_t>(rtProfCtrlType_t::RT_PROF_CTRL_SWITCH), &data, sizeof(data)),
        HcclResult::HCCL_SUCCESS);
}

TEST_F(DfxProfilingHandlerTest, Ut_CommandHandle_When_Stop_Expect_Success)
{
    rtProfCommandHandle_t data{};
    data.type = PROF_COMMANDHANDLE_TYPE_STOP;
    EXPECT_EQ(
        handler_.CommandHandle(static_cast<uint32_t>(rtProfCtrlType_t::RT_PROF_CTRL_SWITCH), &data, sizeof(data)),
        HcclResult::HCCL_SUCCESS);
}

TEST_F(DfxProfilingHandlerTest, Ut_CommandHandle_When_Default_Expect_Success)
{
    rtProfCommandHandle_t data{};
    data.type = static_cast<decltype(data.type)>(999);
    EXPECT_EQ(
        handler_.CommandHandle(static_cast<uint32_t>(rtProfCtrlType_t::RT_PROF_CTRL_SWITCH), &data, sizeof(data)),
        HcclResult::HCCL_SUCCESS);
}

TEST_F(DfxProfilingHandlerTest, Ut_GetHostApiState_Expect_MatchFlag)
{
    handler_.enableHostApi_.store(true);
    EXPECT_TRUE(handler_.GetHostApiState());
    handler_.enableHostApi_.store(false);
    EXPECT_FALSE(handler_.GetHostApiState());
}

TEST_F(DfxProfilingHandlerTest, Ut_GetHcclNodeState_Expect_MatchFlag)
{
    handler_.enableHcclNode_.store(true);
    EXPECT_TRUE(handler_.GetHcclNodeState());
}

TEST_F(DfxProfilingHandlerTest, Ut_GetHcclL0State_Expect_MatchFlag)
{
    handler_.enableHcclL0_.store(true);
    EXPECT_TRUE(handler_.GetHcclL0State());
}

TEST_F(DfxProfilingHandlerTest, Ut_GetHcclL1State_Expect_MatchFlag)
{
    handler_.enableHcclL1_.store(true);
    EXPECT_TRUE(handler_.GetHcclL1State());
}

TEST_F(DfxProfilingHandlerTest, Ut_GetCachedAlgTypeHashId_Expect_MatchStored)
{
    handler_.cachedAlgTypeHashId_.store(12345);
    EXPECT_EQ(handler_.GetCachedAlgTypeHashId(), 12345ULL);
}

// ===========================================================================
// DfxProfilingReporter UT
// ===========================================================================

namespace {
// 打桩记录DfxProfilingHandler::ReportHostApi的opType，供GE下发场景的ReportOp行为断言。
// mockcpp约定：invoke桩成员函数时首参为self（this指针），后接成员函数原参数；
// OpType为MAKE_ENUM生成的枚举类（默认构造为INVALID，不能从整型static_cast）
OpType g_reportHostApiLastOpType;

HcclResult StubRecordReportHostApi(
    DfxProfilingHandler* self, OpType opType, uint64_t beginTime, uint64_t endTime, bool cachedReq, bool isAiCpu)
{
    (void)self;
    (void)beginTime;
    (void)endTime;
    (void)cachedReq;
    (void)isAiCpu;
    g_reportHostApiLastOpType = opType;
    return HCCL_SUCCESS;
}

void ResetReportHostApiRecord() { g_reportHostApiLastOpType = OpType(); }

// 打桩计数DfxProfilingHandler::ReportHcclTaskApi调用次数，供ReportAllTasks遍历行为断言
int g_reportTaskApiCallCount = 0;

HcclResult StubCountReportTaskApi(
    DfxProfilingHandler* self, TaskParamType taskType, uint64_t beginTime, uint64_t endTime, bool isMasterStream,
    bool cachedReq, bool ignoreLevel, uint32_t threadId)
{
    (void)self;
    (void)taskType;
    (void)beginTime;
    (void)endTime;
    (void)isMasterStream;
    (void)cachedReq;
    (void)ignoreLevel;
    (void)threadId;
    g_reportTaskApiCallCount++;
    return HCCL_SUCCESS;
}
} // namespace

class DfxProfilingReporterTest : public testing::Test {
protected:
    void SetUp() override
    {
        handlerPtr_ = &DfxProfilingHandler::GetInstance();
        mgr_ = std::make_unique<MirrorTaskManager>(0, &GlobalMirrorTasks::Instance(), false);
        reporter_ = std::make_unique<DfxProfilingReporter>(mgr_.get(), handlerPtr_);
        SetLaunchKernelMode(false); // GE下发开关为进程级全局，用例间复位，防跨用例/跨文件泄漏
        g_reportTaskApiCallCount = 0;
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
        SetLaunchKernelMode(false);
    }

    DfxProfilingHandler* handlerPtr_ = nullptr;
    std::unique_ptr<MirrorTaskManager> mgr_;
    std::unique_ptr<DfxProfilingReporter> reporter_;
    std::vector<std::unique_ptr<TaskInfoQueue>> queues_;

    // 直接向MirrorTaskManager注册一个（可为空的）本地队列，绕过AddTaskInfo的建队逻辑，
    // 以便独立构造"队列存在但为空/半填槽"等异常形态。
    // 注意：不可经GlobalMirrorTasks::CreateQueue建队——UT桩(llt_next_orion_stub.cc)忽略
    // devId/streamId，恒返回同一个static队列，会造成用例间状态串扰
    TaskInfoQueue& RegisterQueue(u32 streamId, QueueType type)
    {
        if (type == QueueType::Circular_Queue) {
            queues_.push_back(std::make_unique<CircularQueue<std::unique_ptr<TaskInfo>>>(MAX_CIRCULAR_QUEUE_LENGTH));
        } else {
            queues_.push_back(std::make_unique<VectorQueue<std::unique_ptr<TaskInfo>>>());
        }
        TaskInfoQueue& queue = *queues_.back();
        mgr_->streamQueues_[streamId] = MirrorStreamQueueEntry{&queue, type, 0};
        return queue;
    }

    static std::unique_ptr<TaskInfo> MakeTask(u32 streamId, u32 taskId)
    {
        TaskParam param{}; // 匿名union成员非平凡导致默认构造被删除，须聚合初始化
        param.taskType = TaskParamType::TASK_SDMA;
        param.beginTime = 100;
        param.endTime = 200;
        param.tid = 42;
        return std::make_unique<TaskInfo>(streamId, taskId, 0, param, nullptr, true);
    }

    bool HasCursor(u32 streamId) const
    {
        auto it = reporter_->curLastPoses_.find(streamId);
        return it != reporter_->curLastPoses_.end() && it->second != nullptr;
    }
};

TEST_F(DfxProfilingReporterTest, Ut_Init_When_AlreadyInit_Expect_Success)
{
    reporter_->initializedFlag_ = true;
    EXPECT_EQ(reporter_->Init(), HCCL_SUCCESS);
}

TEST_F(DfxProfilingReporterTest, Ut_Init_When_NullMgr_Expect_NoThrow)
{
    DfxProfilingReporter rep(nullptr, handlerPtr_);
    EXPECT_NO_THROW(rep.Init());
}

TEST_F(DfxProfilingReporterTest, Ut_Init_When_NullHandler_Expect_NoThrow)
{
    DfxProfilingReporter rep(mgr_.get(), nullptr);
    EXPECT_NO_THROW(rep.Init());
}

TEST_F(DfxProfilingReporterTest, Ut_ReportOp_When_OpInfoNull_Expect_Success)
{
    EXPECT_EQ(reporter_->ReportOp(1000, false, true), HcclResult::HCCL_SUCCESS);
}

TEST_F(DfxProfilingReporterTest, Ut_ReportOp_When_AicpuEngine_Expect_Success)
{
    auto opInfo = std::make_shared<DfxOpInfo>();
    opInfo->op_.opType = OpType::ALLREDUCE;
    opInfo->engine = CommEngine::COMM_ENGINE_AICPU;
    mgr_->SetCurrDfxOpInfo(opInfo);
    EXPECT_EQ(reporter_->ReportOp(1000, false, true), HcclResult::HCCL_SUCCESS);
}

TEST_F(DfxProfilingReporterTest, Ut_ReportOp_When_OtherEngine_Expect_Success)
{
    auto opInfo = std::make_shared<DfxOpInfo>();
    opInfo->op_.opType = OpType::ALLREDUCE;
    opInfo->engine = CommEngine::COMM_ENGINE_CPU_TS;
    mgr_->SetCurrDfxOpInfo(opInfo);
    EXPECT_EQ(reporter_->ReportOp(1000, false, true), HcclResult::HCCL_SUCCESS);
}

TEST_F(DfxProfilingReporterTest, Ut_ReportOp_When_OpBasedTrue_Expect_Success)
{
    auto opInfo = std::make_shared<DfxOpInfo>();
    opInfo->op_.opType = OpType::ALLREDUCE;
    opInfo->engine = CommEngine::COMM_ENGINE_CPU_TS;
    mgr_->SetCurrDfxOpInfo(opInfo);
    EXPECT_EQ(reporter_->ReportOp(1000, false, true), HcclResult::HCCL_SUCCESS);
}

TEST_F(DfxProfilingReporterTest, Ut_ReportOp_When_OpBasedFalse_Expect_Success)
{
    auto opInfo = std::make_shared<DfxOpInfo>();
    opInfo->op_.opType = OpType::ALLREDUCE;
    opInfo->engine = CommEngine::COMM_ENGINE_CPU_TS;
    mgr_->SetCurrDfxOpInfo(opInfo);
    EXPECT_EQ(reporter_->ReportOp(1000, false, false), HcclResult::HCCL_SUCCESS);
}

// GE下发场景(LaunchKernelMode=true)下协商算子(NEGOTIATIONOP)仍按非GE的opbase方式
// 上报HostApi。先SetLaunchKernelMode(true)复现GE场景，再打桩断言ReportHostApi被调用恰一次且
// opType正确——若删除isNegotiationOp分支（回归），GE下协商算子不再上报，本用例即失败
TEST_F(DfxProfilingReporterTest, Ut_ReportOp_When_NegotiationOpUnderLaunchKernelMode_Expect_HostApiReported)
{
    SetLaunchKernelMode(true); // 复现GE下发场景
    auto opInfo = std::make_shared<DfxOpInfo>();
    opInfo->op_.opType = OpType::ALLREDUCE;
    opInfo->op_.opMode = OpMode::NEGOTIATIONOP;
    opInfo->engine = CommEngine::COMM_ENGINE_CPU_TS;
    mgr_->SetCurrDfxOpInfo(opInfo);

    ResetReportHostApiRecord();
    MOCKER_CPP(&DfxProfilingHandler::ReportHostApi).expects(once()).will(invoke(StubRecordReportHostApi));
    EXPECT_EQ(reporter_->ReportOp(1000, false, true), HcclResult::HCCL_SUCCESS);
    GlobalMockObject::verify();                                  // 恰好调用一次（零次或多次均失败）
    EXPECT_TRUE(g_reportHostApiLastOpType == OpType::ALLREDUCE); // opType透传正确
}

// 对照组：GE下发场景下普通算子(OPBASE模式)不上报HostApi（GE侧自行上报，HCCL不重复上报）——
// 与上一用例仅opMode不同，隔离"协商算子例外"这一唯一变量
TEST_F(DfxProfilingReporterTest, Ut_ReportOp_When_OpbaseOpUnderLaunchKernelMode_Expect_HostApiNotReported)
{
    SetLaunchKernelMode(true); // 复现GE下发场景
    auto opInfo = std::make_shared<DfxOpInfo>();
    opInfo->op_.opType = OpType::ALLREDUCE;
    opInfo->op_.opMode = OpMode::OPBASE; // 对照组：普通算子（非协商）
    opInfo->engine = CommEngine::COMM_ENGINE_CPU_TS;
    mgr_->SetCurrDfxOpInfo(opInfo);

    // GE下普通算子不得经HCCL上报HostApi：never()断言零调用，被误调时mockcpp在调用现场即失败
    MOCKER_CPP(&DfxProfilingHandler::ReportHostApi).expects(never());
    EXPECT_EQ(reporter_->ReportOp(1000, false, true), HcclResult::HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(DfxProfilingReporterTest, Ut_UpdateProfStat_When_SameState_Expect_NoChange)
{
    bool state = handlerPtr_->GetHcclL1State();
    reporter_->enableHcclL1_.store(state);
    EXPECT_NO_THROW(reporter_->UpdateProfStat());
    EXPECT_EQ(reporter_->enableHcclL1_.load(), state);
}

TEST_F(DfxProfilingReporterTest, Ut_UpdateProfStat_When_StateChanged_Expect_NoThrow)
{
    reporter_->enableHcclL1_.store(false);
    handlerPtr_->enableHcclL1_.store(true);
    EXPECT_NO_THROW(reporter_->UpdateProfStat());
}

TEST_F(DfxProfilingReporterTest, Ut_ReportAllTasksLog_When_CircularQueueEmpty_Expect_NoThrow)
{
    constexpr u32 streamId = 9101;
    TaskInfoQueue& queue = RegisterQueue(streamId, QueueType::Circular_Queue);
    ASSERT_TRUE(queue.IsEmpty());

    MOCKER_CPP(&DfxProfilingHandler::ReportHcclTaskApi).expects(never());
    EXPECT_NO_THROW(reporter_->ReportAllTasksLog());
    GlobalMockObject::verify();
}

// 日志路径空游标：按无游标处理直接跳过，不得解引用空shared_ptr游标
TEST_F(DfxProfilingReporterTest, Ut_ReportAllTasksLog_When_CursorNullptr_Expect_NoThrow)
{
    constexpr u32 streamId = 9111;
    TaskInfoQueue& queue = RegisterQueue(streamId, QueueType::Vector_Queue);
    queue.Append(MakeTask(streamId, 1));
    reporter_->curLastPoses_[streamId] = nullptr; // 模拟异常残留的空游标

    MOCKER_CPP(&DfxProfilingHandler::ReportHcclTaskApi).expects(never());
    EXPECT_NO_THROW(reporter_->ReportAllTasksLog());
    GlobalMockObject::verify();
}

// 日志路径半填槽：pastLastPos游标之后的空槽不得触发Describe空指针调用
TEST_F(DfxProfilingReporterTest, Ut_ReportAllTasksLog_When_MidSlotNullptr_Expect_SkipNullTask)
{
    constexpr u32 streamId = 9112;
    TaskInfoQueue& queue = RegisterQueue(streamId, QueueType::Vector_Queue);
    queue.Append(MakeTask(streamId, 1));
    (void)queue.GetAndUpdate(); // 游标之后构造size_+1且槽内容为nullptr的半填槽
    ASSERT_EQ(queue.Size(), 2U);

    reporter_->curLastPoses_[streamId] = queue.Begin(); // 游标指向首槽，pastLastPos后即遇到空槽
    MOCKER_CPP(&DfxProfilingHandler::ReportHcclTaskApi).expects(never());
    EXPECT_NO_THROW(reporter_->ReportAllTasksLog());
    GlobalMockObject::verify();
}

// 空VectorQueue且残留旧游标时，ReportAllTasks应跳过该流并清理游标（而非解引用Begin()触发越界）
TEST_F(DfxProfilingReporterTest, Ut_ReportAllTasks_When_QueueEmpty_Expect_SuccessAndCursorErased)
{
    constexpr u32 streamId = 9102;
    TaskInfoQueue& queue = RegisterQueue(streamId, QueueType::Vector_Queue);
    ASSERT_TRUE(queue.IsEmpty());

    reporter_->curLastPoses_[streamId] = queue.Begin(); // 模拟上一轮残留游标
    MOCKER_CPP(&DfxProfilingHandler::ReportHcclTaskApi).expects(never());
    EXPECT_EQ(reporter_->ReportAllTasks(false), HCCL_SUCCESS);
    GlobalMockObject::verify();

    // 空队列游标必须被erase，而非被更新为空队列的Tail()（归一化end()游标）
    EXPECT_TRUE(reporter_->curLastPoses_.find(streamId) == reporter_->curLastPoses_.end());
}

// 空CircularQueue（无游标）下ReportAllTasks应直接跳过，不得触发迭代器越界
TEST_F(DfxProfilingReporterTest, Ut_ReportAllTasks_When_CircularQueueEmpty_Expect_Success)
{
    constexpr u32 streamId = 9103;
    TaskInfoQueue& queue = RegisterQueue(streamId, QueueType::Circular_Queue);
    ASSERT_TRUE(queue.IsEmpty());

    MOCKER_CPP(&DfxProfilingHandler::ReportHcclTaskApi).expects(never());
    EXPECT_EQ(reporter_->ReportAllTasks(false), HCCL_SUCCESS);
    GlobalMockObject::verify();
    EXPECT_FALSE(HasCursor(streamId));
}

// 正常任务上报：首个任务经first-report路径上报一次，重复调用不重复上报（增量语义）
TEST_F(DfxProfilingReporterTest, Ut_ReportAllTasks_When_TaskValid_Expect_ReportOnceAndCursorSaved)
{
    constexpr u32 streamId = 9104;
    TaskInfoQueue& queue = RegisterQueue(streamId, QueueType::Vector_Queue);
    queue.Append(MakeTask(streamId, 1));
    ASSERT_EQ(queue.Size(), 1U);

    MOCKER_CPP(&DfxProfilingHandler::ReportHcclTaskApi).expects(exactly(1)).will(invoke(StubCountReportTaskApi));
    EXPECT_EQ(reporter_->ReportAllTasks(false), HCCL_SUCCESS);
    EXPECT_EQ(g_reportTaskApiCallCount, 1);
    EXPECT_TRUE(HasCursor(streamId)); // 游标推进到Tail

    // 队列无新增任务时再次上报，不产生新的ReportHcclTaskApi调用
    EXPECT_EQ(reporter_->ReportAllTasks(false), HCCL_SUCCESS);
    EXPECT_EQ(g_reportTaskApiCallCount, 1);
    GlobalMockObject::verify();
}

// 增量上报：新任务入队后仅上报新增部分
TEST_F(DfxProfilingReporterTest, Ut_ReportAllTasks_When_TaskAppended_Expect_IncrementalReport)
{
    constexpr u32 streamId = 9105;
    TaskInfoQueue& queue = RegisterQueue(streamId, QueueType::Vector_Queue);
    queue.Append(MakeTask(streamId, 1));

    MOCKER_CPP(&DfxProfilingHandler::ReportHcclTaskApi).expects(exactly(2)).will(invoke(StubCountReportTaskApi));
    EXPECT_EQ(reporter_->ReportAllTasks(false), HCCL_SUCCESS);
    EXPECT_EQ(g_reportTaskApiCallCount, 1);

    queue.Append(MakeTask(streamId, 2));
    EXPECT_EQ(reporter_->ReportAllTasks(false), HCCL_SUCCESS);
    EXPECT_EQ(g_reportTaskApiCallCount, 2); // 第二轮仅上报新增的task2
    EXPECT_TRUE(HasCursor(streamId));
    GlobalMockObject::verify();
}

// 半填槽：首槽有效、中间槽unique_ptr为空（写入窗口期状态），遍历应跳过空槽而非空指针解引用
TEST_F(DfxProfilingReporterTest, Ut_ReportAllTasks_When_MidSlotNullptr_Expect_SkipNullTask)
{
    constexpr u32 streamId = 9106;
    TaskInfoQueue& queue = RegisterQueue(streamId, QueueType::Vector_Queue);
    queue.Append(MakeTask(streamId, 1));
    (void)queue.GetAndUpdate(); // 占用下一槽但不赋值，构造size_+1且槽内容为nullptr的半填槽
    ASSERT_EQ(queue.Size(), 2U);

    MOCKER_CPP(&DfxProfilingHandler::ReportHcclTaskApi).expects(exactly(1)).will(invoke(StubCountReportTaskApi));
    EXPECT_EQ(reporter_->ReportAllTasks(false), HCCL_SUCCESS);
    EXPECT_EQ(g_reportTaskApiCallCount, 1); // 仅task1被上报，nullptr槽被跳过
    EXPECT_TRUE(HasCursor(streamId));
    GlobalMockObject::verify();
}

// 游标为nullptr（异常残留）时不应解引用游标，应重置为当前Tail后跳过
TEST_F(DfxProfilingReporterTest, Ut_ReportAllTasks_When_CursorNullptr_Expect_ResetToTail)
{
    constexpr u32 streamId = 9107;
    TaskInfoQueue& queue = RegisterQueue(streamId, QueueType::Vector_Queue);
    queue.Append(MakeTask(streamId, 1));
    reporter_->curLastPoses_[streamId] = nullptr; // 模拟异常残留的空游标

    MOCKER_CPP(&DfxProfilingHandler::ReportHcclTaskApi).expects(never());
    EXPECT_EQ(reporter_->ReportAllTasks(false), HCCL_SUCCESS);
    GlobalMockObject::verify();
    EXPECT_TRUE(HasCursor(streamId)); // 游标被重置为Tail，不再是nullptr
}

// 首槽unique_ptr为空（半填槽在队首）：整条流跳过，不进入遍历
TEST_F(DfxProfilingReporterTest, Ut_ReportAllTasks_When_FirstSlotNullptr_Expect_SkipStream)
{
    constexpr u32 streamId = 9108;
    TaskInfoQueue& queue = RegisterQueue(streamId, QueueType::Vector_Queue);
    (void)queue.GetAndUpdate(); // 首槽即为nullptr
    ASSERT_EQ(queue.Size(), 1U);

    MOCKER_CPP(&DfxProfilingHandler::ReportHcclTaskApi).expects(never());
    EXPECT_EQ(reporter_->ReportAllTasks(false), HCCL_SUCCESS);
    GlobalMockObject::verify();
    EXPECT_FALSE(HasCursor(streamId));
}
