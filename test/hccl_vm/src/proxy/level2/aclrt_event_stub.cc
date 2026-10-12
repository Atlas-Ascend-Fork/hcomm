/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "EVENT_STUB"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <unistd.h>
#include <utility>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "sim_log.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "storage/table_access.h"
#include "graph_capture.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

aclError aclrtCreateEventWithFlag(aclrtEvent* event, uint32_t flag)
{
    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = sim::runtime::Db::GetById<sim::runtime::Context>(runner.current_ctx_id);
    if (!currCtx.ok()) {
        // not find
        HCCL_VM_ERROR("ctx not found:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    auto& crrCtxId = currCtx.value->id;

    auto now = std::chrono::system_clock::now();
    std::time_t timeNow = std::chrono::system_clock::to_time_t(now);

    sim::runtime::Event tmp{};
    tmp.create_ctx_id = crrCtxId;
    tmp.event_flag = flag;
    tmp.status = ACL_EVENT_RECORDED_STATUS_COMPLETE;
    tmp.created_time = static_cast<uint64_t>(timeNow);
    tmp.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::Event>(tmp);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto res = inserted.value->value;

    *event = (aclrtEvent)res;
    HCCL_VM_INFO("id:{:d}", res);
    return ACL_SUCCESS;
}

aclError aclrtCreateEventExWithFlag(aclrtEvent* event, uint32_t flag) { return aclrtCreateEventWithFlag(event, flag); }

aclError aclrtCreateEvent(aclrtEvent* event) { return aclrtCreateEventWithFlag(event, 0); }

aclError aclrtDestroyEvent(aclrtEvent event)
{
    uint64_t eventId = (uint32_t)(uintptr_t)event;
    HCCL_VM_INFO("id:{:d}", eventId);
    // 销毁句柄只清理活动索引，模型中已经记录的版本及已加入的流仍然保留。
    ForgetCaptureEvent(eventId);
    (void)sim::runtime::Db::Delete<sim::runtime::Event>(HcclSim::Storage::Eq(&sim::runtime::Event::id, eventId));
    return ACL_SUCCESS;
}

aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream)
{
    uint64_t streamIdx = (uint64_t)(uintptr_t)stream;
    uint64_t eventIdx = (uint32_t)(uintptr_t)event;

    auto currEvent = sim::runtime::Db::GetById<sim::runtime::Event>(eventIdx);
    if (!currEvent.ok()) {
        // not find
        HCCL_VM_ERROR("event not found:{:d}", eventIdx);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto currStm = sim::runtime::Db::GetById<sim::runtime::Stream>(streamIdx);
    if (!currStm.ok()) {
        // not find
        HCCL_VM_ERROR("stream not found:{:d}", streamIdx);
        return ACL_ERROR_INVALID_PARAM;
    }

    // 只记录 Event 的采集关联，不生成 Task，也不把 Event 转成 Notify。
    // 同一句柄再次 Record 会建立新版本，后续 Wait 关联到最近一次 Record。
    const auto captureResult = RecordCaptureEvent(
        streamIdx, eventIdx, currEvent->create_ctx_id, (currEvent->event_flag & ACL_EVENT_EXTERNAL) != 0);
    if (captureResult == CaptureEventResult::ERROR) {
        return ACL_ERROR_INVALID_PARAM;
    }
    (void)sim::runtime::Db::Update<sim::runtime::Event>(
        HcclSim::Storage::Eq(&sim::runtime::Event::id, eventIdx),
        HcclSim::Storage::Set(&sim::runtime::Event::status, ACL_EVENT_RECORDED_STATUS_NOT_READY));

    return ACL_SUCCESS;
}

aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream)
{
    (void)stream;
    uint64_t eventIdx = (uint32_t)(uintptr_t)event;
    auto currEvent = sim::runtime::Db::GetById<sim::runtime::Event>(eventIdx);
    if (!currEvent.ok()) {
        HCCL_VM_ERROR("event not found:{:d}", eventIdx);
        return ACL_ERROR_INVALID_PARAM;
    }
    (void)sim::runtime::Db::Update<sim::runtime::Event>(
        HcclSim::Storage::Eq(&sim::runtime::Event::id, eventIdx),
        HcclSim::Storage::Set(&sim::runtime::Event::status, ACL_EVENT_RECORDED_STATUS_COMPLETE));
    return ACL_SUCCESS;
}

aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus* status)
{
    uint64_t eventId = (uint32_t)(uintptr_t)event;
    auto res = sim::runtime::Db::GetById<sim::runtime::Event>(eventId);
    if (!res.ok()) {
        HCCL_VM_ERROR("event not found:{:d}", eventId);
        return ACL_ERROR_INVALID_PARAM;
    }

    *status = (aclrtEventRecordedStatus)(res.value->status);
    return ACL_SUCCESS;
}

aclError aclrtQueryEventWaitStatus(aclrtEvent event, aclrtEventWaitStatus* status)
{
    (void)event;
    (void)status;
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeEvent(aclrtEvent event)
{
    (void)event;
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeEventWithTimeout(aclrtEvent event, int32_t timeout)
{
    (void)event;
    (void)timeout;
    return ACL_SUCCESS;
}

aclError aclrtEventElapsedTime(float* ms, aclrtEvent startEvent, aclrtEvent endEvent)
{
    (void)startEvent;
    (void)endEvent;
    *ms = 1;
    return ACL_SUCCESS;
}

aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event)
{
    const uint64_t streamId = (uint64_t)(uintptr_t)stream;
    const uint64_t eventId = (uint32_t)(uintptr_t)event;
    const auto currEvent = sim::runtime::Db::GetById<sim::runtime::Event>(eventId);
    const auto currStream = sim::runtime::Db::GetById<sim::runtime::Stream>(streamId);
    if (!currEvent.ok()) {
        HCCL_VM_ERROR("Invalid Event handle for WaitEvent, event={}", eventId);
        return ACL_ERROR_INVALID_PARAM;
    }
    // PTA 在 capture 流 Record、通信主流 Wait；此处传播模型归属，
    // 让通信主流后续真正的 Notify Record/Wait 进入既有 action 录制分支。
    // Wait 不消耗 Record 关联，因此一次 Event Record 可供多条流加入。
    // 未捕获路径沿用空桩行为（包括 nullptr 默认流）；若 Event 属于活动图，
    // 无有效上下文的等待流不能加入，交由关联检查返回错误。
    const uint64_t contextId = currStream.ok() ? currStream->ctx_id : 0;
    if (WaitCaptureEvent(streamId, eventId, contextId, (currEvent->event_flag & ACL_EVENT_EXTERNAL) != 0)
        == CaptureEventResult::ERROR) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

aclError aclrtSetOpWaitTimeout(uint32_t timeout)
{
    (void)timeout;
    return ACL_SUCCESS;
}

aclError aclrtEventGetTimestamp(aclrtEvent event, uint64_t* timestamp)
{
    (void)event;
    (void)timestamp;
    return ACL_SUCCESS;
}

aclError aclrtGetEventId(aclrtEvent event, uint32_t* eventId)
{
    *eventId = (uint32_t)(uintptr_t)event;
    (void)eventId;
    return ACL_SUCCESS;
}

aclError aclrtGetEventAvailNum(uint32_t* eventCount)
{
    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (runner.current_ctx_id == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = sim::runtime::Db::GetById<sim::runtime::Context>(runner.current_ctx_id);
    if (!currCtx.ok()) {
        // not find
        HCCL_VM_ERROR("ctx not found:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    auto& devId = currCtx.value->device_id;

    auto device = sim::runtime::Db::GetById<sim::runtime::Device>(devId);
    if (!device.ok()) {
        HCCL_VM_ERROR("device not found:{:d}", devId);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (memcmp(device.value->soc_version, "A3", strlen("A3") == 0)) {
        *eventCount = 65535;
    }

    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
