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
#include <string>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include "prof_sal_lite.h"
#define private public
#define protected public
#include "dfx/device/dfx_profiling_handler_lite_pub.h"
#include "dfx/device/dfx_res_pub.h"
#undef private
#undef protected

using namespace Hccl;

namespace aicpu {
enum DfxStatusT : uint8_t { AICPU_ERROR_NONE = 0, AICPU_ERROR_FAILED = 1 };
DfxStatusT __attribute__((weak)) GetTaskAndStreamId(uint64_t& taskId, uint32_t& streamId) { return AICPU_ERROR_NONE; }
} // namespace aicpu

class DfxProfilingHandlerLiteTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DfxProfilingHandlerLiteTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "DfxProfilingHandlerLiteTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in DfxProfilingHandlerLiteTest SetUP" << std::endl;
        handler_.enableHcclL0_ = false;
        handler_.enableHcclL1_ = false;
        handler_.initializedFlag_ = false;
        handler_.cachedAlgTypeHashId_ = 0;
        handler_.taskTypeHashCache_.clear();
        handler_.opTypeHashCache_.clear();
        handler_.algTypeHashCache_.clear();
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in DfxProfilingHandlerLiteTest TearDown" << std::endl;
    }

    DfxProfilingHandlerLite& handler_ = DfxProfilingHandlerLite::GetInstance();
};

static void PrepareHandlerInit(DfxProfilingHandlerLite& handler)
{
    handler.initializedFlag_ = false;
    handler.Init();
}

static DfxCommContext MakeDefaultCtx() { return {nullptr, DFX_INVALID_U64, INVALID_U32, 0}; }

TEST_F(DfxProfilingHandlerLiteTest, Ut_CCoreDetails_PreserveIdsAndDoNotDecodeNotify)
{
    const auto savedHash = handler_.getProfHashId_;
    handler_.getProfHashId_ = [](const char* name, size_t len) -> uint64_t {
        const std::string type(name, len);
        return type == "CCore_Wait" ? 101U : (type == "CCore_Record" ? 102U : 1U);
    };
    handler_.InitHashCaches();
    handler_.getProfHashId_ = savedHash;
    DfxDfxOpInfo opInfo{};
    opInfo.dataType = 1;
    for (const auto type : {TASK_CCORE_NOTIFY_WAIT, TASK_CCORE_NOTIFY_RECORD}) {
        DfxTaskInfo task{};
        task.taskType = type;
        task.sqId = 7;
        task.taskId = 0x80000400U;
        task.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
        task.linkType = LINK_ONCHIP;
        task.transportType = DFX_TRANSPORT_TYPE_LOCAL;
        // A CONDITION must never inherit an unrelated hardware Notify ID.
        task.taskPara.Notify.notifyId = 1;
        ASSERT_TRUE(task.IsTaskTypeValid());
        MsprofAicpuHcclTaskInfo detail{};
        handler_.GetTaskDetailInfosFromDfxTaskInfo(&task, detail, MakeDefaultCtx());
        EXPECT_EQ(detail.itemId, type == TASK_CCORE_NOTIFY_WAIT ? 101U : 102U);
        EXPECT_EQ(detail.streamId, 7U);
        EXPECT_EQ(detail.taskId, 0x80000400U);
        EXPECT_EQ(detail.notifyID, DFX_INVALID_U64);
        EXPECT_EQ(detail.dataSize, 0U);
        EXPECT_EQ(detail.transportType, DFX_TRANSPORT_TYPE_LOCAL);
    }
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetInstance_Expect_ReturnSameInstance)
{
    DfxProfilingHandlerLite& inst1 = DfxProfilingHandlerLite::GetInstance();
    DfxProfilingHandlerLite& inst2 = DfxProfilingHandlerLite::GetInstance();
    EXPECT_EQ(&inst1, &inst2);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_Init_When_NotInit_Expect_ReturnSuccess)
{
    handler_.initializedFlag_ = false;
    HcclResult ret = handler_.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(handler_.initializedFlag_, true);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_Init_When_AlreadyInit_Expect_ReturnSuccessDirectly)
{
    handler_.initializedFlag_ = true;
    HcclResult ret = handler_.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportHcclOpInfo_When_L0Off_Expect_EarlyReturn)
{
    handler_.enableHcclL0_ = false;
    DfxDfxOpInfo opInfo{};
    EXPECT_NO_THROW(handler_.ReportHcclOpInfo(opInfo, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportHcclOpInfo_When_L0On_Expect_RunToEnd)
{
    PrepareHandlerInit(handler_);
    handler_.enableHcclL0_ = true;
    DfxCommContext ctx = MakeDefaultCtx();
    ctx.groupName = 999;
    ctx.rankSize = 4;
    DfxDfxOpInfo opInfo{};
    opInfo.count = 100;
    opInfo.dataType = 1;
    EXPECT_NO_THROW(handler_.ReportHcclOpInfo(opInfo, ctx));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportMainStreamTask_When_L0Off_Expect_EarlyReturn)
{
    handler_.enableHcclL0_ = false;
    DfxFlagTaskInfo flagTaskInfo{};
    EXPECT_NO_THROW(handler_.ReportMainStreamTask(flagTaskInfo));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportMainStreamTask_When_L0On_Expect_RunToEnd)
{
    PrepareHandlerInit(handler_);
    handler_.enableHcclL0_ = true;
    DfxFlagTaskInfo flagTaskInfo{};
    flagTaskInfo.taskId = 0x00010002;
    flagTaskInfo.type = DfxMainStreamTaskType::TAIL;
    EXPECT_NO_THROW(handler_.ReportMainStreamTask(flagTaskInfo));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportAdditionInfo_When_Normal_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    MsprofAdditionalInfo reporterData{};
    EXPECT_NO_THROW(handler_.ReportAdditionInfo(reporterData));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillBatchReporterData_When_Normal_Expect_ReturnTrue)
{
    PrepareHandlerInit(handler_);
    MsprofAicpuHcclTaskInfo taskInfos[2] = {};
    MsprofAdditionalInfo addInfo{};
    bool ret = handler_.FillBatchReporterData(1, taskInfos, addInfo);
    EXPECT_EQ(ret, true);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillBatchReporterData_When_ZeroBatch_Expect_ReturnTrue)
{
    MsprofAicpuHcclTaskInfo taskInfos[1] = {};
    MsprofAdditionalInfo addInfo{};
    bool ret = handler_.FillBatchReporterData(0, taskInfos, addInfo);
    (void)ret;
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportBatchAddInfo_When_NotLast_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    MsprofAicpuHcclTaskInfo taskInfos[2] = {};
    MsprofAdditionalInfo addInfoVec[4] = {};
    uint32_t addInfoIndx = 0;
    bool ret = handler_.ReportBatchAddInfo(1, taskInfos, addInfoVec, addInfoIndx, 4, false);
    (void)ret;
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportBatchAddInfo_When_IsLast_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    MsprofAicpuHcclTaskInfo taskInfos[2] = {};
    MsprofAdditionalInfo addInfoVec[4] = {};
    uint32_t addInfoIndx = 0;
    bool ret = handler_.ReportBatchAddInfo(1, taskInfos, addInfoVec, addInfoIndx, 4, true);
    (void)ret;
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_UpdateProfSwitch_Expect_NoThrow)
{
    EXPECT_NO_THROW(handler_.UpdateProfSwitch());
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_IsProfOn_When_UnknownFeature_Expect_False)
{
    EXPECT_EQ(handler_.IsProfOn(0xFFFF), false);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_IsProfSwitchOn_When_L1Off_Expect_FalseAndFlagSet)
{
    handler_.enableHcclL1_ = true;
    bool ret = handler_.IsProfSwitchOn(DfxProfilingLevel::L1);
    EXPECT_EQ(ret, false);
    EXPECT_EQ(handler_.enableHcclL1_, false);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_SetProL0On_Expect_FlagSet)
{
    handler_.SetProL0On(true);
    EXPECT_EQ(handler_.enableHcclL0_, true);
    handler_.SetProL0On(false);
    EXPECT_EQ(handler_.enableHcclL0_, false);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_SetProL1On_Expect_FlagSet)
{
    handler_.SetProL1On(true);
    EXPECT_EQ(handler_.enableHcclL1_, true);
    handler_.SetProL1On(false);
    EXPECT_EQ(handler_.enableHcclL1_, false);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetProfHashId_When_NullName_Expect_ReturnInvalid)
{
    uint64_t hashId = handler_.GetProfHashId(nullptr, 10);
    EXPECT_EQ(hashId, DFX_INVALID_U64);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetProfHashId_When_ZeroLen_Expect_ReturnInvalid)
{
    uint64_t hashId = handler_.GetProfHashId("test", 0);
    EXPECT_EQ(hashId, DFX_INVALID_U64);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetCachedAlgTypeHashId_When_CacheEmpty_Expect_ReturnInvalid)
{
    handler_.cachedAlgTypeHashId_ = DFX_INVALID_U64;
    EXPECT_EQ(handler_.GetCachedAlgTypeHashId(), DFX_INVALID_U64);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetCachedAlgTypeHashId_When_CacheHasValue_Expect_ReturnValue)
{
    handler_.cachedAlgTypeHashId_ = 55555;
    EXPECT_EQ(handler_.GetCachedAlgTypeHashId(), 55555u);
}

static void FillDfxTaskInfoForType(Hccl::DfxTaskInfo& taskInfo, u8 taskType)
{
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    taskInfo.channelHandle = DFX_INVALID_U64;
    taskInfo.taskType = taskType;
    taskInfo.linkType = 0;
    taskInfo.sqId = 0;
    taskInfo.taskId = 0;
    taskInfo.transportType = 0;
    taskInfo.taskPara.ubDma.notifyId = INVALID_U32;
    taskInfo.taskPara.ubDma.srcAddr = 0;
    taskInfo.taskPara.ubDma.dstAddr = 0;
    taskInfo.taskPara.ubDma.size = 0;
    taskInfo.taskPara.Reduce.notifyId = INVALID_U32;
    taskInfo.taskPara.Reduce.srcAddr = 0;
    taskInfo.taskPara.Reduce.dstAddr = 0;
    taskInfo.taskPara.Reduce.size = 0;
    taskInfo.taskPara.Reduce.reduceOp = 0;
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_Sdma_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    DfxCommContext ctx = MakeDefaultCtx();
    ctx.groupName = 100;
    ctx.localRank = 0;
    ctx.rankSize = 8;
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA));
    taskInfo.linkType = 2;
    taskInfo.sqId = 1;
    taskInfo.taskId = 10;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, ctx));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_Rdma_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_RDMA));
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_ReduceInline_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_REDUCE_INLINE));
    taskInfo.linkType = 1;
    taskInfo.taskPara.Reduce.notifyId = 5;
    taskInfo.taskPara.Reduce.srcAddr = 0x1000;
    taskInfo.taskPara.Reduce.dstAddr = 0x2000;
    taskInfo.taskPara.Reduce.size = 256;
    taskInfo.taskPara.Reduce.reduceOp = 2;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_UbDma_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB));
    taskInfo.linkType = 3;
    taskInfo.taskPara.ubDma.notifyId = 20;
    taskInfo.taskPara.ubDma.srcAddr = 0x3000;
    taskInfo.taskPara.ubDma.dstAddr = 0x4000;
    taskInfo.taskPara.ubDma.size = 1024;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_UbInlineWrite_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB_INLINE_WRITE));
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_UbReduceInline_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB_REDUCE_INLINE));
    taskInfo.taskPara.Reduce.reduceOp = 3;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_WriteWithNotify_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY));
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_WriteReduceWithNotify_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY));
    taskInfo.taskPara.Reduce.reduceOp = 1;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_NotifyRecord_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_RECORD));
    taskInfo.linkType = 0;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_NotifyWait_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT));
    taskInfo.linkType = 0;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_InvalidType_Expect_EarlyReturn)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(200));
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_WithDfxOpInfo_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA));
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.opType = static_cast<u8>(OpTypeVal::OP_TYPE_ALLREDUCE);
    opInfo.dataType = 1;
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    std::unordered_map<u64, u32> rankMap;
    rankMap[0x5678] = 3;
    DfxCommContext ctx = MakeDefaultCtx();
    ctx.channelRemoteRankIdMap = &rankMap;
    taskInfo.channelHandle = 0x5678;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, ctx));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportStreamTaskDetailsLog_When_EmptyQueue_Expect_NoThrow)
{
    TaskInfoCircularQueue queue;
    EXPECT_NO_THROW(handler_.ReportStreamTaskDetailsLog(queue));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportStreamTaskDetailsLog_When_NonEmptyQueue_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    TaskInfoCircularQueue queue;
    Hccl::DfxTaskInfo* slot = static_cast<Hccl::DfxTaskInfo*>(queue.NextSlot());
    if (slot != nullptr) {
        FillDfxTaskInfoForType(*slot, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA));
    }
    EXPECT_NO_THROW(handler_.ReportStreamTaskDetailsLog(queue));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportStreamTaskDetails_When_EmptyQueue_Expect_EarlyReturn)
{
    TaskInfoCircularQueue queue;
    EXPECT_NO_THROW(handler_.ReportStreamTaskDetails(queue, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportStreamTaskDetails_When_NonEmptyQueue_Expect_RunToEnd)
{
    PrepareHandlerInit(handler_);
    DfxCommContext ctx = MakeDefaultCtx();
    ctx.groupName = 100;
    ctx.localRank = 0;
    ctx.rankSize = 8;
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA));
    taskInfo.linkType = 1;
    taskInfo.sqId = 0;
    taskInfo.taskId = 1;
    Hccl::TaskInfoCircularQueue queue;
    Hccl::DfxTaskInfo* slot = static_cast<Hccl::DfxTaskInfo*>(queue.NextSlot());
    if (slot != nullptr) {
        *slot = taskInfo;
    }
    EXPECT_NO_THROW(handler_.ReportStreamTaskDetails(queue, ctx));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportStreamTaskDetails_When_BatchReport_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    DfxCommContext ctx = MakeDefaultCtx();
    ctx.groupName = 100;
    ctx.localRank = 0;
    ctx.rankSize = 8;
    Hccl::TaskInfoCircularQueue queue;
    for (int i = 0; i < 3; i++) {
        Hccl::DfxTaskInfo* slot = static_cast<Hccl::DfxTaskInfo*>(queue.NextSlot());
        if (slot != nullptr) {
            FillDfxTaskInfoForType(*slot, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA));
            slot->linkType = 1;
            slot->sqId = static_cast<u32>(i);
            slot->taskId = static_cast<u32>(i);
        }
    }
    EXPECT_NO_THROW(handler_.ReportStreamTaskDetails(queue, ctx));
}

namespace {
std::vector<MsprofAdditionalInfo> capturedReports;
std::vector<uint32_t> capturedReportLengths;
uint32_t failReportCall = 0;

int32_t CaptureTaskReports(uint32_t flag, const void* data, uint32_t length)
{
    EXPECT_EQ(flag, 1U);
    EXPECT_EQ(length % sizeof(MsprofAdditionalInfo), 0U);
    const auto* records = static_cast<const MsprofAdditionalInfo*>(data);
    capturedReports.insert(capturedReports.end(), records, records + length / sizeof(MsprofAdditionalInfo));
    capturedReportLengths.push_back(length);
    return capturedReportLengths.size() == failReportCall ? 1 : 0;
}
} // namespace

class DfxProfilingBatchBufferTest : public DfxProfilingHandlerLiteTest {
protected:
    void SetUp() override
    {
        DfxProfilingHandlerLiteTest::SetUp();
        savedReport_ = handler_.reportAdditionalInfo_;
        savedBatchReport_ = handler_.reportBatchAdditionalInfo_;
        handler_.reportAdditionalInfo_ = CaptureTaskReports;
        handler_.reportBatchAdditionalInfo_ = CaptureTaskReports;
        ResetCapture();
    }

    void TearDown() override
    {
        handler_.reportAdditionalInfo_ = savedReport_;
        handler_.reportBatchAdditionalInfo_ = savedBatchReport_;
        ResetCapture();
        DfxProfilingHandlerLiteTest::TearDown();
    }

    void ResetCapture()
    {
        capturedReports.clear();
        capturedReportLengths.clear();
        failReportCall = 0;
    }

    DfxCommContext CompactCtx() const
    {
        auto ctx = MakeDefaultCtx();
        ctx.compactReportOpInfo = &opInfo_;
        return ctx;
    }

    void FillQueue(TaskInfoCircularQueue& queue, uint32_t count, bool wrap)
    {
        if (wrap) {
            for (uint32_t i = 0; i < queue.GetCapacity() - 1; ++i) {
                queue.NextSlot();
            }
            queue.MarkAllRead();
        }
        for (uint32_t i = 0; i < count; ++i) {
            auto* task = static_cast<DfxTaskInfo*>(queue.NextSlot());
            *task = DfxTaskInfo{};
            task->taskType
                = (i % 3 == 0) ? TASK_CCORE_NOTIFY_WAIT : ((i % 3 == 1) ? TASK_UB : TASK_CCORE_NOTIFY_RECORD);
            task->taskId = 0x80000400U + i;
            task->sqId = 7U;
            task->dfxOpInfo = reinterpret_cast<u64>(&opInfo_);
            task->linkType = LINK_ONCHIP;
            task->transportType = DFX_TRANSPORT_TYPE_LOCAL;
            if (task->taskType == TASK_UB) {
                task->taskPara.ubDma.srcAddr = 0x1000U + i;
                task->taskPara.ubDma.dstAddr = 0x2000U + i;
                task->taskPara.ubDma.size = 128U + i;
            }
        }
    }

    std::vector<MsprofAdditionalInfo> ExpectedReports(TaskInfoCircularQueue& queue)
    {
        std::vector<MsprofAdditionalInfo> expected;
        for (uint32_t i = 0; i < queue.GetCount(); i += 2) {
            MsprofAicpuHcclTaskInfo details[2] = {};
            const uint32_t count = std::min(2U, static_cast<uint32_t>(queue.GetCount()) - i);
            for (uint32_t j = 0; j < count; ++j) {
                auto* task = queue.GetSlot((queue.GetBegin() + i + j) % queue.GetCapacity());
                handler_.GetTaskDetailInfosFromDfxTaskInfo(task, details[j], MakeDefaultCtx());
            }
            MsprofAdditionalInfo record{};
            record.level = MSPROF_REPORT_AICPU_LEVEL;
            record.type = MSPROF_REPORT_AICPU_MC2_BATCH_HCCL_INFO;
            record.threadId = SalGetTidLite();
            record.dataLen = count * sizeof(MsprofAicpuHcclTaskInfo);
            EXPECT_EQ(memcpy_s(record.data, sizeof(record.data), details, record.dataLen), 0);
            expected.push_back(record);
        }
        return expected;
    }

    void ExpectSameReports(const std::vector<MsprofAdditionalInfo>& expected, bool compact = true)
    {
        ASSERT_EQ(capturedReports.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            // Legacy buffers may retain bytes outside dataLen after a full batch; these are not task data.
            const size_t length
                = compact ? sizeof(MsprofAdditionalInfo) : offsetof(MsprofAdditionalInfo, data) + expected[i].dataLen;
            EXPECT_EQ(std::memcmp(&capturedReports[i], &expected[i], length), 0) << "report index " << i;
        }
    }

    DfxProfilingHandlerLite::ReportAdditionalInfoHandle savedReport_ = nullptr;
    DfxProfilingHandlerLite::ReportBatchAdditionalInfoHandle savedBatchReport_ = nullptr;
    DfxDfxOpInfo opInfo_{};
};

TEST_F(DfxProfilingBatchBufferTest, BatchSizesPreservePacketsAndCallCounts)
{
    for (const bool compact : {false, true}) {
        for (const uint32_t count : {0U, 1U, 2U, 31U, 32U, 33U, 1023U, 1024U, 1025U, 2048U, 2176U}) {
            SCOPED_TRACE(count);
            TaskInfoCircularQueue queue;
            FillQueue(queue, count, false);
            const auto expected = ExpectedReports(queue);
            ResetCapture();
            const auto ctx = compact ? CompactCtx() : MakeDefaultCtx();
            EXPECT_EQ(handler_.CanUseCompactReport(queue, ctx), compact && count != 0);
            handler_.ReportStreamTaskDetails(queue, ctx);
            ExpectSameReports(expected, compact);
            ASSERT_EQ(capturedReportLengths.size(), (expected.size() + 511) / 512);
            for (size_t i = 0; i < capturedReportLengths.size(); ++i) {
                EXPECT_EQ(
                    capturedReportLengths[i],
                    std::min(size_t{512}, expected.size() - i * 512) * sizeof(MsprofAdditionalInfo));
            }
            EXPECT_EQ(queue.GetCount(), count);
        }
    }
}

TEST_F(DfxProfilingBatchBufferTest, WrappedQueueAndRepeatedReportsPreserveOrder)
{
    TaskInfoCircularQueue queue;
    FillQueue(queue, 33, true);
    const auto expected = ExpectedReports(queue);
    handler_.ReportStreamTaskDetails(queue, CompactCtx());
    ExpectSameReports(expected);
    ResetCapture();
    handler_.ReportStreamTaskDetails(queue, CompactCtx());
    ExpectSameReports(expected);
}

TEST_F(DfxProfilingBatchBufferTest, SingleReportFallbackPreservesPackets)
{
    handler_.reportBatchAdditionalInfo_ = nullptr;
    for (const bool compact : {false, true}) {
        for (const uint32_t count : {0U, 1U, 2U, 31U, 32U, 33U, 1025U}) {
            SCOPED_TRACE(count);
            TaskInfoCircularQueue queue;
            FillQueue(queue, count, false);
            const auto expected = ExpectedReports(queue);
            ResetCapture();
            handler_.ReportStreamTaskDetails(queue, compact ? CompactCtx() : MakeDefaultCtx());
            ExpectSameReports(expected);
            EXPECT_EQ(capturedReportLengths.size(), expected.size());
            for (const auto length : capturedReportLengths) {
                EXPECT_EQ(length, sizeof(MsprofAdditionalInfo));
            }
        }
    }
}

TEST_F(DfxProfilingBatchBufferTest, BatchReportFailureStopsAtFailedBatch)
{
    TaskInfoCircularQueue queue;
    FillQueue(queue, 1025, false);
    auto expected = ExpectedReports(queue);
    expected.resize(512);
    failReportCall = 1;
    handler_.ReportStreamTaskDetails(queue, CompactCtx());
    ExpectSameReports(expected);
    EXPECT_EQ(capturedReportLengths.size(), 1U);
    EXPECT_EQ(queue.GetCount(), 1025U);
}

TEST_F(DfxProfilingBatchBufferTest, SingleReportFailureRetainsExistingContinueBehavior)
{
    handler_.reportBatchAdditionalInfo_ = nullptr;
    TaskInfoCircularQueue queue;
    FillQueue(queue, 3, false);
    const auto expected = ExpectedReports(queue);
    failReportCall = 1;
    handler_.ReportStreamTaskDetails(queue, CompactCtx());
    ExpectSameReports(expected);
    EXPECT_EQ(capturedReportLengths.size(), 2U);
}

TEST_F(DfxProfilingBatchBufferTest, OnlyCurrentOperationUsesCompactBuffers)
{
    TaskInfoCircularQueue queue;
    FillQueue(queue, 3, true);
    const auto ctx = CompactCtx();
    EXPECT_TRUE(handler_.CanUseCompactReport(queue, ctx));
    EXPECT_FALSE(handler_.CanUseCompactReport(queue, MakeDefaultCtx()));
    auto* task = queue.GetSlot((queue.GetBegin() + 1) % queue.GetCapacity());
    DfxDfxOpInfo otherOp{};
    task->dfxOpInfo = reinterpret_cast<u64>(&otherOp);
    EXPECT_FALSE(handler_.CanUseCompactReport(queue, ctx));
    task->dfxOpInfo = 0;
    EXPECT_FALSE(handler_.CanUseCompactReport(queue, ctx));
    task->dfxOpInfo = reinterpret_cast<u64>(&opInfo_);
    // A sub-thread does not need a CONDITION of its own.
    for (u16 i = 0; i < queue.GetCount(); ++i) {
        queue.GetSlot((queue.GetBegin() + i) % queue.GetCapacity())->taskType = TASK_UB;
    }
    EXPECT_TRUE(handler_.CanUseCompactReport(queue, ctx));
}

TEST_F(DfxProfilingBatchBufferTest, DispatcherSelectsLegacyOrCompactImplementation)
{
    TaskInfoCircularQueue queue;
    FillQueue(queue, 2, false);
    MOCKER_CPP(&DfxProfilingHandlerLite::ReportStreamTaskDetailsLegacy).expects(exactly(2));
    MOCKER_CPP(&DfxProfilingHandlerLite::ReportStreamTaskDetailsCompact).expects(once());
    handler_.ReportStreamTaskDetails(queue, MakeDefaultCtx());
    handler_.ReportStreamTaskDetails(queue, CompactCtx());
    queue.GetSlot(queue.GetBegin())->dfxOpInfo = 0;
    handler_.ReportStreamTaskDetails(queue, CompactCtx());
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillReduceInlineDetail_When_CalledViaGetDetail_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_REDUCE_INLINE));
    taskInfo.linkType = 1;
    taskInfo.taskPara.Reduce.notifyId = 10;
    taskInfo.taskPara.Reduce.srcAddr = 0x1000;
    taskInfo.taskPara.Reduce.dstAddr = 0x2000;
    taskInfo.taskPara.Reduce.size = 512;
    taskInfo.taskPara.Reduce.reduceOp = 1;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillSdmaRdmaDetail_When_CalledViaGetDetail_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA));
    taskInfo.linkType = 2;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillUbDmaDetail_When_ReduceInline_CalledViaGetDetail_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB_REDUCE_INLINE));
    taskInfo.linkType = 3;
    taskInfo.taskPara.ubDma.notifyId = 20;
    taskInfo.taskPara.ubDma.srcAddr = 0x3000;
    taskInfo.taskPara.ubDma.dstAddr = 0x4000;
    taskInfo.taskPara.ubDma.size = 1024;
    taskInfo.taskPara.Reduce.reduceOp = 5;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillNotifyDetail_When_CalledViaGetDetail_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_RECORD));
    taskInfo.linkType = 4;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillDefaultDetail_When_CalledViaGetDetail_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_CCU));
    taskInfo.linkType = 0;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillCclTagAndRemoteRank_When_DfxOpInfoInvalid_Expect_NoThrow)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    taskInfo.channelHandle = DFX_INVALID_U64;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.FillCclTagAndRemoteRank(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillCclTagAndRemoteRank_When_MapNullptr_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.opType = 0;
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    taskInfo.channelHandle = 0x1234;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.FillCclTagAndRemoteRank(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillCclTagAndRemoteRank_When_ValidMap_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.opType = static_cast<u8>(OpTypeVal::OP_TYPE_ALLREDUCE);
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    std::unordered_map<u64, u32> rankMap;
    rankMap[0x5678] = 3;
    DfxCommContext ctx = MakeDefaultCtx();
    ctx.channelRemoteRankIdMap = &rankMap;
    taskInfo.channelHandle = 0x5678;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.FillCclTagAndRemoteRank(&taskInfo, taskDetailsInfos, ctx));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillCommonTailFields_When_DfxOpInfoInvalid_Expect_NoThrow)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    taskInfo.sqId = 1;
    taskInfo.taskId = 2;
    taskInfo.channelHandle = DFX_INVALID_U64;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.FillCommonTailFields(&taskInfo, taskDetailsInfos));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillCommonTailFields_When_RemoteRankInvalid_Expect_NoThrow)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    taskInfo.sqId = 1;
    taskInfo.taskId = 2;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    taskDetailsInfos.remoteRank = Hccl::DFX_INVALID_RANKID;
    EXPECT_NO_THROW(handler_.FillCommonTailFields(&taskInfo, taskDetailsInfos));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillCommonTailFields_When_WithDfxOpInfo_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.dataType = 2;
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    taskInfo.sqId = 5;
    taskInfo.taskId = 10;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.FillCommonTailFields(&taskInfo, taskDetailsInfos));
}
