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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <stdexcept>
#include <string>
#define private public
#define protected public
#include "profiling_reporter.h"
#include "communicator_impl.h"
#include "profiling_handler.h"
#include "circular_queue.h"

#undef private
#undef protected

using namespace Hccl;

class ProfilingReporterTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ProfilingReporterTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "ProfilingReporterTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in ProfilingReporterTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in ProfilingReporterTest TearDown" << std::endl;
    }
};

// 测试ProfilingReporter类接口
TEST_F(ProfilingReporterTest, Call_profilingReporter_api_test)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    for (auto& taskMap : globalMirrorTasks.taskMaps_) {
        taskMap.clear();
    }
    // 初始化TaskParam
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_NOTIFY_RECORD,
           .beginTime = 0,
           .endTime = 0,
           .taskPara = {.Notify = {.notifyID = 123, .value = 456}}};

    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo = std::make_shared<std::vector<CcuProfilingInfo>>();
    for (int i = 0; i < 3; ++i) {
        CcuProfilingInfo info;
        info.name = "StubTask" + std::to_string(i);
        info.type = i % 2; // 循环使用不同的类型
        info.dieId = i;
        info.missionId = i + 1;
        info.instrId = i + 2;
        info.reduceOpType = i + 3;
        info.inputDataType = i + 4;
        info.outputDataType = i + 5;
        info.dataSize = (i + 1) * 1024;
        info.ckeId = i + 6;
        info.mask = i + 7;
        for (int j = 0; j < CCU_MAX_CHANNEL_NUM; ++j) {
            info.channelId[j] = j;
            info.remoteRankId[j] = j + 1;
        }
        ccuDetailInfo->push_back(info);
    }
    taskParam.ccuDetailInfo = std::move(ccuDetailInfo);
    // 初始化dfxOpInfo
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    CommunicatorImpl comm;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    dfxOpInfo->comm_ = &comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    auto taskInfo1 = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
    auto taskInfo2 = std::make_unique<TaskInfo>(0, 1, 1, taskParam, dfxOpInfo);
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo1));
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo2));

    ProfilingReporter profilingReporter(&mirrorTaskManager, &ProfilingHandler::GetInstance());
    profilingReporter.Init();
    profilingReporter.ReportOp(0, true, true);
    profilingReporter.ReportAllTasks(true);
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    handler.enableHcclL1_ = true;
    profilingReporter.UpdateProfStat();
}

TEST_F(ProfilingReporterTest, Ut_ReportOp_When_OpInfoNullptr_Expect_ReturnNormally)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    for (auto& taskMap : globalMirrorTasks.taskMaps_) {
        taskMap.clear();
    }
    ProfilingReporter profilingReporter(&mirrorTaskManager, &ProfilingHandler::GetInstance());
    profilingReporter.Init();
    EXPECT_NO_THROW(profilingReporter.ReportOp(0, true, true));
}

TEST_F(ProfilingReporterTest, Ut_ReportOp_When_CommImpNullptr_Expect_ReturnNormally)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    for (auto& taskMap : globalMirrorTasks.taskMaps_) {
        taskMap.clear();
    }
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    dfxOpInfo->comm_ = nullptr;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_NOTIFY_RECORD,
           .beginTime = 0,
           .endTime = 0,
           .taskPara = {.Notify = {.notifyID = 123, .value = 456}}};
    std::unique_ptr<TaskInfo> taskInfo = std::make_unique<TaskInfo>(0, 0, 0, taskParam, dfxOpInfo);
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo));
    ProfilingReporter profilingReporter(&mirrorTaskManager, &ProfilingHandler::GetInstance());
    profilingReporter.Init();
    EXPECT_NO_THROW(profilingReporter.ReportOp(0, true, true));
}

TEST_F(ProfilingReporterTest, Ut_SetCurrDfxOpInfo_When_OpTypeFoundInMap_Expect_OpTypeAndTagSet)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    for (auto& taskMap : globalMirrorTasks.taskMaps_) {
        taskMap.clear();
    }

    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    dfxOpInfo->op_.oldOpType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    dfxOpInfo->op_.oldReduceOp = HCCL_REDUCE_SUM;
    dfxOpInfo->op_.oldDataType = HCCL_DATA_TYPE_FP32;

    ProfilingReporter profilingReporter(&mirrorTaskManager, &ProfilingHandler::GetInstance());
    profilingReporter.SetCurrDfxOpInfo(dfxOpInfo);

    EXPECT_EQ(dfxOpInfo->op_.opType, OpType::ALLREDUCE);
    EXPECT_EQ(dfxOpInfo->tag_, "OpType::ALLREDUCE");
    EXPECT_EQ(dfxOpInfo->op_.reduceOp, ReduceOp::SUM);
    EXPECT_EQ(dfxOpInfo->op_.dataType, DataType::FP32);
    EXPECT_EQ(mirrorTaskManager.GetCurrDfxOpInfo(), dfxOpInfo);
}

TEST_F(ProfilingReporterTest, Ut_SetCurrDfxOpInfo_When_OpTypeNotFoundInMap_Expect_OpTypeAndTagUnchanged)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    for (auto& taskMap : globalMirrorTasks.taskMaps_) {
        taskMap.clear();
    }

    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    dfxOpInfo->op_.oldOpType = HcclCMDType::HCCL_CMD_INVALID;
    dfxOpInfo->op_.opType = OpType::ALLGATHER;
    dfxOpInfo->tag_ = "preset_tag";
    dfxOpInfo->op_.oldReduceOp = HCCL_REDUCE_SUM;
    dfxOpInfo->op_.oldDataType = HCCL_DATA_TYPE_FP32;

    ProfilingReporter profilingReporter(&mirrorTaskManager, &ProfilingHandler::GetInstance());
    profilingReporter.SetCurrDfxOpInfo(dfxOpInfo);

    EXPECT_EQ(dfxOpInfo->op_.opType, OpType::ALLGATHER);
    EXPECT_EQ(dfxOpInfo->tag_, "preset_tag");
    EXPECT_EQ(dfxOpInfo->op_.reduceOp, ReduceOp::SUM);
    EXPECT_EQ(dfxOpInfo->op_.dataType, DataType::FP32);
    EXPECT_EQ(mirrorTaskManager.GetCurrDfxOpInfo(), dfxOpInfo);
}

namespace {
TaskParam MakeTaskParam()
{
    return {
        .taskType = TaskParamType::TASK_NOTIFY_RECORD,
        .beginTime = 0,
        .endTime = 0,
        .taskPara = {.Notify = {.notifyID = 123, .value = 456}}};
}

std::shared_ptr<DfxOpInfo> MakeDfxOpInfo()
{
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    return dfxOpInfo;
}
} // namespace

// 空队列下上报不能解引用归一化 end()，且队列重新有任务后游标需从 Begin() 重新锚定
TEST_F(ProfilingReporterTest, Ut_ReportAllTasks_When_QueueEmptyThenRefilled_Expect_Reanchor)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    for (auto& taskMap : globalMirrorTasks.taskMaps_) {
        taskMap.clear();
    }
    constexpr u32 streamId = 8;
    // 构造一个尚未写入任务的空队列
    TaskInfoQueue& emptyQueue = globalMirrorTasks.CreateQueue(0, streamId, QueueType::Circular_Queue);
    mirrorTaskManager.streamQueues_[streamId] = MirrorStreamQueueEntry{&emptyQueue, QueueType::Circular_Queue, 0};

    ProfilingReporter profilingReporter(&mirrorTaskManager, &ProfilingHandler::GetInstance());
    profilingReporter.Init();
    auto& lastPoses = ProfilingReporter::allLastPoses_[profilingReporter.deviceLogicId_];
    lastPoses.clear();

    // 空队列上报：不保存归一化 end() 游标，也不应崩溃
    EXPECT_NO_THROW(profilingReporter.ReportAllTasks(true));
    EXPECT_EQ(lastPoses.count(streamId), 0U);
    EXPECT_TRUE(profilingReporter.taskInfoBatch_.empty());

    std::shared_ptr<DfxOpInfo> dfxOpInfo = MakeDfxOpInfo();
    TaskParam taskParam = MakeTaskParam();

    // 队列重新有任务：游标从 Begin() 重新锚定，3 个任务全部上报且不重复
    for (u32 i = 0; i < 3; ++i) {
        auto taskInfo = std::make_unique<TaskInfo>(streamId, i, 0, taskParam, dfxOpInfo);
        mirrorTaskManager.AddTaskInfo(std::move(taskInfo));
    }
    EXPECT_NO_THROW(profilingReporter.ReportAllTasks(true));
    ASSERT_EQ(lastPoses.count(streamId), 1U);
    EXPECT_TRUE(*lastPoses[streamId] == *emptyQueue.Tail());
    EXPECT_EQ(profilingReporter.taskInfoBatch_.size(), 3U);

    // 再次增量上报：只上报新增的 2 个任务，不重复也不丢失
    for (u32 i = 3; i < 5; ++i) {
        auto taskInfo = std::make_unique<TaskInfo>(streamId, i, 0, taskParam, dfxOpInfo);
        mirrorTaskManager.AddTaskInfo(std::move(taskInfo));
    }
    EXPECT_NO_THROW(profilingReporter.ReportAllTasks(true));
    EXPECT_TRUE(*lastPoses[streamId] == *emptyQueue.Tail());
    EXPECT_EQ(profilingReporter.taskInfoBatch_.size(), 2U);
}

// 环形队列写满后继续追加（head 推进、旧槽位被复写）时，游标行走需在上界内正常停止
TEST_F(ProfilingReporterTest, Ut_ReportAllTasks_When_CircularQueueWraps_Expect_NoOutOfRange)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    for (auto& taskMap : globalMirrorTasks.taskMaps_) {
        taskMap.clear();
    }
    constexpr u32 streamId = 9;
    constexpr size_t capacity = 3;
    CircularQueue<std::unique_ptr<TaskInfo>> queue(capacity);
    mirrorTaskManager.streamQueues_[streamId] = MirrorStreamQueueEntry{&queue, QueueType::Circular_Queue, 0};

    ProfilingReporter profilingReporter(&mirrorTaskManager, &ProfilingHandler::GetInstance());
    profilingReporter.Init();
    auto& lastPoses = ProfilingReporter::allLastPoses_[profilingReporter.deviceLogicId_];
    lastPoses.clear();

    std::shared_ptr<DfxOpInfo> dfxOpInfo = MakeDfxOpInfo();
    TaskParam taskParam = MakeTaskParam();

    // 每追加一个任务上报一次，跨越一整圈以上，覆盖 head 推进/槽位复写后的游标行走
    for (u32 i = 0; i < capacity * 2 + 1; ++i) {
        auto taskInfo = std::make_unique<TaskInfo>(streamId, i, 0, taskParam, dfxOpInfo);
        EXPECT_NO_THROW(mirrorTaskManager.AddTaskInfo(std::move(taskInfo)));
        EXPECT_NO_THROW(profilingReporter.ReportAllTasks(true));
    }
    EXPECT_EQ(lastPoses.count(streamId), 1U);
}
