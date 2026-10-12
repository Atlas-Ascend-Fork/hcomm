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
#define HCCL_VM_MODULE "MODEL_STUB"

#include <cstdint>
#include <utility>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "operation_data/operation_data_ops.h"
#include "graph_capture.h"
#include "runtime/base.h"
#include "sim_log.h"

// rt_external_base.h 只导出 RT_ERROR_NONE(=0)；真实 RT_ERROR_INVALID_VALUE 在 runtime 内部头,
// 这里用非 0 值表示失败即可, hlt 只判 rtError_t != RT_ERROR_NONE
static constexpr rtError_t kRtErrorInvalidValue = 1;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// 图模式 rtModel* 采集-重放桩
// 句柄约定与 stream 桩一致：句柄 = 进程内自增 id（非 DB id，model 不入库）。
// 类型区分对齐 runtime：rtModelCreate 直建 NORMAL（flag 必须 0，同 runtime rtsModelCreate 校验）；
// CAPTURE 模型由 aclmdlRICaptureBegin 内部创建（见 aclrt_graph_capture_stub.cc），
// 对齐 runtime StreamBeginCapture 内部 ModelCreate(RT_MODEL_CAPTURE_MODEL)。
// rtModelExecute = 遍历采集期录制的 actions 重放 + 末尾等待 boundStreams。

rtError_t rtModelCreate(rtModel_t* mdl, uint32_t flag)
{
    // 对齐 runtime：rtsModelCreate 校验 flag==0；rtModelCreate 的 flag 在 ApiImpl 中亦未参与建型
    if (mdl == nullptr || flag != 0U) {
        return kRtErrorInvalidValue;
    }
    const uint64_t modelId = CreateModelRecord(RtModelType::RT_MODEL_NORMAL);
    if (modelId == 0) {
        return kRtErrorInvalidValue;
    }
    *mdl = (rtModel_t)(uintptr_t)modelId;
    HCCL_VM_INFO("rtModelCreate, model={}", modelId);
    return RT_ERROR_NONE;
}

rtError_t rtModelBindStream(rtModel_t mdl, rtStream_t stm, uint32_t flag)
{
    uint64_t modelId = (uint64_t)(uintptr_t)mdl;
    uint64_t streamId = (uint64_t)(uintptr_t)stm;
    if (modelId == 0 || streamId == 0) {
        return kRtErrorInvalidValue;
    }
    std::lock_guard<std::mutex> lock(g_modelMutex);
    auto it = g_models.find(modelId);
    if (it == g_models.end()) {
        return kRtErrorInvalidValue;
    }
    auto bindIt = g_modelByStream.find(streamId);
    if (bindIt != g_modelByStream.end() && bindIt->second != modelId) {
        HCCL_VM_ERROR(
            "stream[{}] already bound to model[{}], cannot bind to model[{}]", streamId, bindIt->second, modelId);
        return kRtErrorInvalidValue;
    }

    ModelRecord& rec = it->second;
    if (rec.modelType == RtModelType::RT_MODEL_CAPTURE_MODEL) {
        const auto stream = sim::runtime::Db::GetById<sim::runtime::Stream>(streamId);
        if (!stream.ok() || stream->ctx_id == 0 || (rec.phase != ModelPhase::NEW && rec.phase != ModelPhase::CAPTURING)
            || rec.captureInvalidated || (rec.contextId != 0 && rec.contextId != stream->ctx_id)) {
            return kRtErrorInvalidValue;
        }
        rec.contextId = stream->ctx_id;
    }
    rec.boundStreams.insert(streamId);
    g_modelByStream[streamId] = modelId;
    if (flag == 0) { // RT_HEAD_STREAM
        rec.headStreamId = streamId;
    }
    rec.phase = ModelPhase::CAPTURING;
    HCCL_VM_INFO("rtModelBindStream, model={}, stream={}, flag={}", modelId, streamId, flag);
    return RT_ERROR_NONE;
}

rtError_t rtStreamAddToModel(rtStream_t stm, rtModel_t captureMdl)
{
    uint64_t modelId = (uint64_t)(uintptr_t)captureMdl;
    uint64_t streamId = (uint64_t)(uintptr_t)stm;
    if (modelId == 0 || streamId == 0) {
        return kRtErrorInvalidValue;
    }
    std::lock_guard<std::mutex> lock(g_modelMutex);
    auto it = g_models.find(modelId);
    if (it == g_models.end()) {
        return kRtErrorInvalidValue;
    }
    if (it->second.modelType == RtModelType::RT_MODEL_CAPTURE_MODEL) {
        const auto stream = sim::runtime::Db::GetById<sim::runtime::Stream>(streamId);
        if (!stream.ok() || it->second.phase != ModelPhase::CAPTURING || it->second.captureInvalidated
            || stream->ctx_id != it->second.contextId) {
            return kRtErrorInvalidValue;
        }
    }
    auto bindIt = g_modelByStream.find(streamId);
    if (bindIt != g_modelByStream.end() && bindIt->second != modelId) {
        HCCL_VM_ERROR(
            "stream[{}] already bound to model[{}], cannot add to model[{}]", streamId, bindIt->second, modelId);
        return kRtErrorInvalidValue;
    }
    it->second.boundStreams.insert(streamId);
    g_modelByStream[streamId] = modelId;
    HCCL_VM_INFO("rtStreamAddToModel, stream={}, model={}", streamId, modelId);
    return RT_ERROR_NONE;
}

rtError_t rtEndGraph(rtModel_t mdl, rtStream_t stm)
{
    uint64_t modelId = (uint64_t)(uintptr_t)mdl;
    std::lock_guard<std::mutex> lock(g_modelMutex);
    auto it = g_models.find(modelId);
    if (it == g_models.end()) {
        return kRtErrorInvalidValue;
    }
    if (it->second.modelType == RtModelType::RT_MODEL_CAPTURE_MODEL) {
        if (it->second.phase != ModelPhase::CAPTURING || it->second.headStreamId != (uint64_t)(uintptr_t)stm) {
            HCCL_VM_ERROR("CaptureEnd must use the capture head stream, model={}", modelId);
            return kRtErrorInvalidValue;
        }
        // Capture 结束后，原流可以继续普通下发或采集另一个图；重放仍使用模型快照。
        ReleaseCaptureBindings(it->second);
        if (it->second.captureInvalidated) {
            it->second.phase = ModelPhase::SEALED;
            return kRtErrorInvalidValue;
        }
    }
    it->second.phase = ModelPhase::SEALED;
    HCCL_VM_INFO("rtEndGraph, model={}", modelId);
    return RT_ERROR_NONE;
}

rtError_t rtEndGraphEx(rtModel_t mdl, rtStream_t stm, uint32_t flags)
{
    (void)flags;
    return rtEndGraph(mdl, stm);
}

rtError_t rtModelLoadComplete(rtModel_t mdl)
{
    uint64_t modelId = (uint64_t)(uintptr_t)mdl;
    std::lock_guard<std::mutex> lock(g_modelMutex);
    auto it = g_models.find(modelId);
    if (it == g_models.end()) {
        return kRtErrorInvalidValue;
    }
    if (it->second.modelType == RtModelType::RT_MODEL_CAPTURE_MODEL
        && (it->second.phase != ModelPhase::SEALED || it->second.captureInvalidated)) {
        return kRtErrorInvalidValue;
    }
    it->second.phase = ModelPhase::LOADED;
    HCCL_VM_INFO("rtModelLoadComplete, model={}", modelId);
    return RT_ERROR_NONE;
}

rtError_t rtModelExecute(rtModel_t mdl, rtStream_t stm, uint32_t flag)
{
    (void)flag;
    uint64_t modelId = (uint64_t)(uintptr_t)mdl;
    uint64_t mainStreamId = (uint64_t)(uintptr_t)stm; // 需求2：入参流 = 重放触发流(真实主流)
    if (modelId == 0) {
        HCCL_VM_ERROR("rtModelExecute, invalid model handle");
        return kRtErrorInvalidValue;
    }

    // 锁内只取 headStreamId 快照，重放动作在锁外执行（ReplayModelActions 自行加锁）
    uint64_t subStreamId = 0;
    {
        std::lock_guard<std::mutex> lock(g_modelMutex);
        auto it = g_models.find(modelId);
        if (it == g_models.end()) {
            HCCL_VM_ERROR("rtModelExecute, model={} not found", modelId);
            return kRtErrorInvalidValue;
        }
        if (it->second.modelType == RtModelType::RT_MODEL_CAPTURE_MODEL
            && (it->second.phase != ModelPhase::LOADED || it->second.captureInvalidated)) {
            return kRtErrorInvalidValue;
        }
        subStreamId = it->second.headStreamId; // 采集流 master_stream
        // aclGraph 的 START_SUB/END_SUB 绑定采集期算子真实主流；普通模型保持 head stream 语义。
        if (it->second.modelType == RtModelType::RT_MODEL_CAPTURE_MODEL && it->second.capturedOpStreamId != 0) {
            subStreamId = it->second.capturedOpStreamId;
        }
    }
    uint32_t execSeq = g_nextModelExecSeq++; // 同一次 rtModelExecute 的 4 任务共享 execSeq

    ModelReplayContext replay;
    if (!PrepareModelReplay(modelId, replay)) {
        HCCL_VM_ERROR("rtModelExecute, model={} prepare replay opDetail failed", modelId);
        return kRtErrorInvalidValue;
    }

    // 空 action（数据量为 0 的空算子等）无动作可重放：不插入 MODEL_EXEC 边界任务，直接成功返回，
    // 避免 checker 对孤立边界任务做模型执行配对校验时报错。
    if (replay.actions.empty()) {
        HCCL_VM_INFO("rtModelExecute, model={} has no actions, skip MODEL_EXEC boundary tasks", modelId);
        return RT_ERROR_NONE;
    }

    // 重放前：主流 Start + 从流 StartSub
    InsertModelExecTask(execSeq, MODEL_EXEC_START, modelId, mainStreamId, subStreamId);
    InsertModelExecTask(execSeq, MODEL_EXEC_START_SUB, modelId, subStreamId, mainStreamId);

    // 重放（现有逻辑：action 重发到 action.streamId=采集流，落在 StartSub/EndSub 之间）
    int ret = ReplayModelActions(modelId, replay);

    sim::operation::g_currOpDetailId = replay.boundaryOpDetailId;
    // 重放后：从流 EndSub + 主流 End
    InsertModelExecTask(execSeq, MODEL_EXEC_END_SUB, modelId, subStreamId, mainStreamId);
    InsertModelExecTask(execSeq, MODEL_EXEC_END, modelId, mainStreamId, subStreamId);

    HCCL_VM_INFO(
        "rtModelExecute, model={}, execSeq={}, mainStream={}, subStream={}, ret={}", modelId, execSeq, mainStreamId,
        subStreamId, ret);
    return ret == 0 ? RT_ERROR_NONE : kRtErrorInvalidValue;
}

rtError_t rtModelExecuteSync(rtModel_t mdl, rtStream_t stm, uint32_t flag, int32_t timeout)
{
    (void)timeout;
    return rtModelExecute(mdl, stm, flag);
}

rtError_t rtModelUnbindStream(rtModel_t mdl, rtStream_t stm)
{
    uint64_t modelId = (uint64_t)(uintptr_t)mdl;
    uint64_t streamId = (uint64_t)(uintptr_t)stm;
    std::lock_guard<std::mutex> lock(g_modelMutex);
    auto it = g_models.find(modelId);
    if (it == g_models.end()) {
        return kRtErrorInvalidValue;
    }
    it->second.boundStreams.erase(streamId);
    auto binding = g_modelByStream.find(streamId);
    if (binding != g_modelByStream.end() && binding->second == modelId) {
        g_modelByStream.erase(binding);
    }
    HCCL_VM_INFO("rtModelUnbindStream, model={}, stream={}", modelId, streamId);
    return RT_ERROR_NONE;
}

rtError_t rtModelDestroy(rtModel_t mdl)
{
    uint64_t modelId = (uint64_t)(uintptr_t)mdl;
    std::lock_guard<std::mutex> lock(g_modelMutex);
    auto it = g_models.find(modelId);
    if (it == g_models.end()) {
        return kRtErrorInvalidValue;
    }
    ReleaseCaptureBindings(it->second);
    it->second.phase = ModelPhase::DESTROYED;
    g_models.erase(it);
    HCCL_VM_INFO("rtModelDestroy, model={}", modelId);
    return RT_ERROR_NONE;
}

// ── rtEvent* 空桩 ──────────────────────────────────────────────
// hlt 图模式 hcomRunTest 在 rtModelLoadComplete 之后、rtModelExecute 重放之前,
// 直接调用 rtEventSynchronize(end_event) 同步 (hlt_hccl_base_test.cc:1449)。

rtError_t rtEventSynchronize(rtEvent_t evt)
{
    (void)evt;
    return RT_ERROR_NONE;
}

rtError_t rtEventSynchronizeWithTimeout(rtEvent_t evt, const int32_t timeout)
{
    (void)evt;
    (void)timeout;
    return RT_ERROR_NONE;
}

rtError_t rtEventElapsedTime(float32_t* timeInterval, rtEvent_t startEvent, rtEvent_t endEvent)
{
    (void)startEvent;
    (void)endEvent;
    if (timeInterval != nullptr) {
        *timeInterval = 0.0f;
    }
    return RT_ERROR_NONE;
}

// rts* 别名(新命名) 与 rt* 一一对应, 内部直接转发
rtError_t rtsModelCreate(rtModel_t* mdl, uint32_t flag) { return rtModelCreate(mdl, flag); }

rtError_t rtsModelBindStream(rtModel_t mdl, rtStream_t stm, uint32_t flag) { return rtModelBindStream(mdl, stm, flag); }

rtError_t rtsEndGraph(rtModel_t mdl, rtStream_t stm) { return rtEndGraph(mdl, stm); }

rtError_t rtsModelLoadComplete(rtModel_t mdl, void* reserve)
{
    (void)reserve;
    return rtModelLoadComplete(mdl);
}

rtError_t rtsModelExecute(rtModel_t mdl, int32_t timeout)
{
    (void)timeout;
    return rtModelExecute(mdl, nullptr, 0U);
}

rtError_t rtsModelUnbindStream(rtModel_t mdl, rtStream_t stm) { return rtModelUnbindStream(mdl, stm); }

rtError_t rtsModelDestroy(rtModel_t mdl) { return rtModelDestroy(mdl); }

#ifdef __cplusplus
}
#endif // __cplusplus
