/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include "dfx/device/dfx_res_pub.h"
#include "../../../hccl_api_base_test.h"
#include "local_notify_impl.h"
#include "llt_hccl_stub_rank_graph.h"
#define private public
#define protected public
#include "rtsq_a5.h"
#include "stream_lite.h"
#include "sqe.h"
#include "dfx/device/dfx_profiling_handler_lite_pub.h"
#include "dfx_profiling_reporter_lite.h"
#include "hcclCommDfxLite.h"
#undef protected
#undef private

namespace aicpu {
// Model the AICPU framework's full-width task ID allocation without loading its runtime.
void GetSqeId(const uint32_t num, uint32_t& start, uint32_t& end)
{
    static uint32_t next = 0x80000400U;
    start = next;
    next += num;
    end = next;
}
} // namespace aicpu

// Keep device RTSQ submissions batched without loading the AICPU dispatch runtime.
bool IsBatchLaunchMode() { return true; }
uint32_t GetSqFullTimeOut() { return 30U; }
HcclResult HandleDispatchAllStreams() { return HCCL_SUCCESS; }

namespace CCoreDfxTest {
using namespace Hccl;
class CCoreDfxRtsqTest : public testing::Test {
protected:
    void SetUp() override
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
    }

    void TearDown() override { GlobalMockObject::verify(); }
    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
};

class CCoreDfxOwner : public hccl::AicpuTsThread {
public:
    explicit CCoreDfxOwner(StreamLite* stream) : hccl::AicpuTsThread(std::string()), stream_(stream) {}
    void* GetStreamLitePtr() const override { return stream_; }

private:
    StreamLite* stream_;
};

TEST_F(CCoreDfxRtsqTest, CCoreMarksCurrentOpBeforeFullQueueReport)
{
    hccl::HcclCommDfxLite dfx;
    dfx.opInfoQueue_ = new DfxOpInfoCircularQueue();
    dfx.queueInitialized_ = true;
    DfxDfxOpInfo op{};
    op.opType = HCCL_CMD_ALLGATHER;
    op.dataType = HCCL_DATA_TYPE_FP16;
    ASSERT_EQ(dfx.SetCurrDfxOpInfo(&op), HCCL_SUCCESS);
    const auto* current = static_cast<const DfxDfxOpInfo*>(dfx.GetLatestDfxOpInfo());
    auto stream = std::make_unique<StreamLite>(126, 7, 0, 4);
    stream->SetGetLatestDfxOpInfoCallback([&]() {
        return current;
    });
    uint32_t reports = 0;
    stream->SetReportStreamTaskCallback([&](TaskInfoCircularQueue* queue) {
        ++reports;
        EXPECT_EQ(dfx.GetDfxCommContext().compactReportOpInfo, current);
        queue->MarkAllRead();
    });
    auto* queue = stream->GetTaskInfos();
    for (u32 i = 0; i < queue->GetCapacity(); ++i) {
        auto* task = static_cast<DfxTaskInfo*>(queue->NextSlot());
        *task = DfxTaskInfo{};
        task->dfxOpInfo = reinterpret_cast<u64>(current);
    }
    CCoreDfxOwner thread(stream.get());
    RtsqA5 rtsq(0, 126, 7, false);
    rtsq.aicpuTsThreadPtr_ = &thread;
    rtsq.taskId_ = 0x80000400U;
    rtsq.taskIdEnd_ = 0x80000800U;
    const auto taskId = rtsq.GetTaskId();
    rtsq.CCoreNotifyWait(0x1000, 0x2000, true);
    rtsq.CCoreNotifyRecord(0x3000, 0x2000);
    EXPECT_EQ(reports, 1U);
    ASSERT_EQ(queue->GetCount(), 2U);
    const auto* wait = queue->GetSlot(queue->GetBegin());
    EXPECT_EQ(wait->taskId, taskId);
    EXPECT_EQ(wait->sqId, 7);
    EXPECT_EQ(wait->taskType, TASK_CCORE_NOTIFY_WAIT);
    const auto* record = queue->GetSlot((queue->GetBegin() + 1) % queue->GetCapacity());
    EXPECT_EQ(record->taskId, taskId + 1);
    EXPECT_EQ(record->taskType, TASK_CCORE_NOTIFY_RECORD);
}

TEST_F(CCoreDfxRtsqTest, ccore_dfx_records_full_task_id_and_physical_sq)
{
    StreamLite stream(126, 7, 0, 0);
    CCoreDfxOwner owner(&stream);
    DfxDfxOpInfo opInfo{};
    stream.SetReportStreamTaskCallback([](TaskInfoCircularQueue* queue) {
        queue->MarkAllRead();
    });
    stream.SetGetLatestDfxOpInfoCallback([&]() -> const void* {
        return &opInfo;
    });
    auto* rtsq = static_cast<RtsqA5*>(stream.GetRtsq());
    ASSERT_EQ(rtsq->SetAicpuTsThreadPtr(&owner), HCCL_SUCCESS);

    const u32 waitId = rtsq->GetTaskId();
    EXPECT_GT(waitId, 0xFFFFU);
    rtsq->CCoreNotifyWait(0x1000, 0x2000, true);
    const u32 recordId = rtsq->GetTaskId();
    rtsq->CCoreNotifyRecord(0x3000, 0x2000);
    auto* queue = stream.GetTaskInfos();
    ASSERT_EQ(queue->GetCount(), 2U);
    auto* wait = queue->GetSlot(queue->GetBegin());
    auto* record = queue->GetSlot((queue->GetBegin() + 1) % queue->GetCapacity());
    EXPECT_EQ(wait->taskType, TASK_CCORE_NOTIFY_WAIT);
    EXPECT_EQ(record->taskType, TASK_CCORE_NOTIFY_RECORD);
    EXPECT_EQ(wait->taskId, waitId);
    EXPECT_EQ(record->taskId, recordId);
    EXPECT_EQ(wait->sqId, 7U);
    EXPECT_EQ(record->sqId, 7U);
    EXPECT_EQ(wait->dfxOpInfo, reinterpret_cast<u64>(&opInfo));
    EXPECT_EQ(wait->channelHandle, DFX_INVALID_U64);
    EXPECT_EQ(wait->transportType, DFX_TRANSPORT_TYPE_LOCAL);
    // CONDITION leaves the raw union zeroed; the reporter must not decode it as a Notify ID.
    EXPECT_EQ(wait->taskPara.Notify.notifyId, 0U);
    EXPECT_EQ(rtsq->GetPendingSqeCnt(), 2U);
}

TEST_F(CCoreDfxRtsqTest, ccore_dfx_skips_missing_owner_stream_callback_or_op_info)
{
    StreamLite stream(126, 7, 0, 0);
    auto* rtsq = static_cast<RtsqA5*>(stream.GetRtsq());
    rtsq->CCoreNotifyWait(0x1000, 0x2000, false);
    CCoreDfxOwner nullOwner(nullptr);
    ASSERT_EQ(rtsq->SetAicpuTsThreadPtr(&nullOwner), HCCL_SUCCESS);
    rtsq->CCoreNotifyRecord(0x3000, 0x2000);
    CCoreDfxOwner owner(&stream);
    ASSERT_EQ(rtsq->SetAicpuTsThreadPtr(&owner), HCCL_SUCCESS);
    rtsq->CCoreNotifyWait(0x1000, 0x2000, false);
    stream.SetReportStreamTaskCallback([](TaskInfoCircularQueue* queue) {
        queue->MarkAllRead();
    });
    rtsq->CCoreNotifyRecord(0x3000, 0x2000);
    EXPECT_TRUE(stream.GetTaskInfos()->IsEmpty());
    EXPECT_EQ(rtsq->GetPendingSqeCnt(), 4U);
}

TEST_F(CCoreDfxRtsqTest, ccore_dfx_flushes_full_queue_before_reusing_slot)
{
    StreamLite stream(126, 7, 0, 0);
    CCoreDfxOwner owner(&stream);
    DfxDfxOpInfo opInfo{};
    auto* queue = stream.GetTaskInfos();
    for (u32 i = 0; i < queue->GetCapacity(); ++i) {
        auto* slot = static_cast<DfxTaskInfo*>(queue->NextSlot());
        *slot = DfxTaskInfo{};
        slot->channelHandle = 0x1234;
        slot->taskPara.ubDma.size = 0x5678;
    }
    u32 reports = 0;
    stream.SetReportStreamTaskCallback([&](TaskInfoCircularQueue* tasks) {
        EXPECT_TRUE(tasks->IsFull());
        ++reports;
        tasks->MarkAllRead();
    });
    stream.SetGetLatestDfxOpInfoCallback([&]() -> const void* {
        return &opInfo;
    });
    auto* rtsq = static_cast<RtsqA5*>(stream.GetRtsq());
    ASSERT_EQ(rtsq->SetAicpuTsThreadPtr(&owner), HCCL_SUCCESS);
    rtsq->CCoreNotifyRecord(0x3000, 0x2000);
    EXPECT_EQ(reports, 1U);
    ASSERT_EQ(queue->GetCount(), 1U);
    auto* slot = queue->GetSlot(queue->GetBegin());
    EXPECT_EQ(slot->taskType, TASK_CCORE_NOTIFY_RECORD);
    EXPECT_EQ(slot->channelHandle, DFX_INVALID_U64);
    EXPECT_EQ(slot->taskPara.ubDma.size, 0U);
}

TEST_F(CCoreDfxRtsqTest, ccore_dfx_keeps_original_id_when_refresh_auto_launches)
{
    StreamLite stream(126, 7, 0, 0);
    CCoreDfxOwner owner(&stream);
    DfxDfxOpInfo opInfo{};
    stream.SetReportStreamTaskCallback([](TaskInfoCircularQueue* queue) {
        queue->MarkAllRead();
    });
    stream.SetGetLatestDfxOpInfoCallback([&]() -> const void* {
        return &opInfo;
    });
    auto* rtsq = static_cast<RtsqA5*>(stream.GetRtsq());
    ASSERT_EQ(rtsq->SetAicpuTsThreadPtr(&owner), HCCL_SUCCESS);
    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(AC_SQE_MAX_CNT - 1));
    for (u32 i = 0; i + 1 < PER_LAUNCH_SQE_CNT; ++i) {
        rtsq->NotifyWait(0);
    }
    const u32 taskId = rtsq->GetTaskId();
    rtsq->CCoreNotifyWait(0x1000, 0x2000, true);
    EXPECT_EQ(rtsq->GetPendingSqeCnt(), 0U);
    ASSERT_EQ(stream.GetTaskInfos()->GetCount(), 1U);
    EXPECT_EQ(stream.GetTaskInfos()->GetSlot(0)->taskId, taskId);
}
TEST_F(CCoreDfxRtsqTest, CompactContextDoesNotReportWhenProfilingOff)
{
    auto& handler = DfxProfilingHandlerLite::GetInstance();
    handler.enableHcclL1_ = false;
    DfxProfilingReporterLite reporter(&handler);
    DfxDfxOpInfo op{};
    DfxCommContext ctx{};
    ctx.compactReportOpInfo = &op;
    TaskInfoCircularQueue queue;
    auto* task = static_cast<DfxTaskInfo*>(queue.NextSlot());
    *task = DfxTaskInfo{};
    task->dfxOpInfo = reinterpret_cast<u64>(&op);
    MOCKER_CPP(&DfxProfilingHandlerLite::ReportStreamTaskDetails).expects(never());
    reporter.ReportStreamTask(&queue, ctx);
    EXPECT_TRUE(queue.IsEmpty());
}

TEST_F(CCoreDfxRtsqTest, ccore_dfx_l1_off_consumes_without_reporting)
{
    auto& handler_ = DfxProfilingHandlerLite::GetInstance();
    DfxProfilingReporterLite reporter(&handler_);
    handler_.enableHcclL1_ = false;
    TaskInfoCircularQueue queue;
    for (const auto type : {TASK_CCORE_NOTIFY_WAIT, TASK_CCORE_NOTIFY_RECORD}) {
        auto* slot = static_cast<DfxTaskInfo*>(queue.NextSlot());
        *slot = DfxTaskInfo{};
        slot->taskType = type;
    }
    MOCKER_CPP(&DfxProfilingHandlerLite::ReportStreamTaskDetails).expects(never());
    reporter.ReportStreamTask(&queue, DfxCommContext{});
    EXPECT_TRUE(queue.IsEmpty());
}

} // namespace CCoreDfxTest
