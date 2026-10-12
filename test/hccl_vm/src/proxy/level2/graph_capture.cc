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
#define HCCL_VM_MODULE "GRAPH_CAPTURE"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>

#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "graph_capture.h"
#include "sim_log.h"
#include "runtime_state/sim_models.h"
#include "storage/table_access.h"
#include "store_sim_store_pub.h"

// 进程内单例：执行真源（不建 DB model 表）
std::unordered_map<uint64_t, ModelRecord> g_models;
std::unordered_map<uint64_t, uint64_t> g_modelByStream;
uint64_t g_nextModelId = 1;
uint32_t g_nextModelExecSeq = 0;

static void RecordCapturedOpStream(ModelRecord& model, uint32_t opDetailId)
{
    if (model.modelType != RtModelType::RT_MODEL_CAPTURE_MODEL || model.capturedOpStreamId != 0 || opDetailId == 0) {
        return;
    }
    sim::operation::OpDetailTab detail{};
    if (sim::operation::QueryOpDetailById(opDetailId, detail) == 0 && detail.streamId != 0) {
        model.capturedOpStreamId = detail.streamId;
        HCCL_VM_INFO(
            "record aclGraph operator stream, opDetailId={}, streamId={}", opDetailId, model.capturedOpStreamId);
    }
}

// 三方并发（torch 主线程 / HCCL 调用线程 / PTA taskqueue worker）访问保护。
// 使用约定：锁内只做 map 查改与数据拷贝，禁止持锁执行 kernel 重发/等待等重操作。
std::mutex g_modelMutex;

namespace {

struct CaptureEventBinding {
    uint64_t modelId;
    std::size_t recordIndex;
};

// 仅保存 Event 最近一次 Record 的活动关联；历史版本由 ModelRecord::captureEvents 保留。
// 与 g_models 一样由 g_modelMutex 保护，不把 Event 句柄寿命当作图依赖的寿命。
std::unordered_map<uint64_t, CaptureEventBinding> g_captureEvents;

// 调用方持有 g_modelMutex；GE 显式建图不参与 Event 的隐式入图传播。
ModelRecord* FindCapturingModel(uint64_t streamId)
{
    auto binding = g_modelByStream.find(streamId);
    if (binding == g_modelByStream.end()) {
        return nullptr;
    }
    auto model = g_models.find(binding->second);
    if (model == g_models.end() || model->second.phase != ModelPhase::CAPTURING
        || model->second.modelType != RtModelType::RT_MODEL_CAPTURE_MODEL) {
        return nullptr;
    }
    return &model->second;
}

// 重放期克隆采集期 opDetail + opMemInfo，生成 opIter 递增的新记录，返回新 opDetailId；失败回退为采集期 id。
uint32_t CloneOpDetailForReplay(uint32_t origOpDetailId)
{
    sim::operation::OpDetailTab orig{};
    if (sim::operation::QueryOpDetailById(origOpDetailId, orig) != 0) {
        HCCL_VM_WARN("clone opDetail failed for origOpDetailId={}, fallback to orig", origOpDetailId);
        return origOpDetailId;
    }

    // 重放 clone 必须携带采集期 opMemInfo，否则 checker 按 opIter 重组 CompositeOpDetail 时查不到 mem。
    sim::operation::OpMemInfoTab origMem{};
    const bool hasMem = sim::operation::QueryOpMemInfoByOpDetailId(origOpDetailId, origMem) == 0;

    orig.id = 0; // 交给 DB 自增新 id
    if (sim::operation::InsertOpDetail(orig) != 0) {
        HCCL_VM_WARN("InsertOpDetail failed for replay clone, fallback to orig opDetailId={}", origOpDetailId);
        return origOpDetailId;
    }
    const uint32_t newOpDetailId = sim::operation::g_currOpDetailId;

    if (hasMem) {
        origMem.id = 0;
        // InsertOpMem 会把 opDetailId 覆盖为 g_currOpDetailId（= newOpDetailId），自动关联新 opDetail。
        if (sim::operation::InsertOpMem(origMem) != 0) {
            HCCL_VM_WARN("InsertOpMem failed for replay clone, opDetailId={}", newOpDetailId);
        }
    }

    HCCL_VM_INFO(
        "cloned opDetail(mem={}) for replay, orig={}, new={}, opIter={}", hasMem, origOpDetailId, newOpDetailId,
        orig.opIter);
    return newOpDetailId;
}

} // namespace

extern "C" {

uint64_t CreateModelRecord(RtModelType type)
{
    std::lock_guard<std::mutex> lock(g_modelMutex);
    const uint64_t modelId = g_nextModelId++;
    ModelRecord rec;
    rec.modelId = modelId;
    rec.modelType = type;
    rec.phase = ModelPhase::NEW;
    g_models[modelId] = std::move(rec);
    HCCL_VM_INFO("create model record, model={}, type={}", modelId, static_cast<uint32_t>(type));
    return modelId;
}

bool IsCapturingStream(uint64_t streamId)
{
    std::lock_guard<std::mutex> lock(g_modelMutex);
    auto it = g_modelByStream.find(streamId);
    if (it == g_modelByStream.end()) {
        return false;
    }
    auto mIt = g_models.find(it->second);
    return mIt != g_models.end() && mIt->second.phase == ModelPhase::CAPTURING;
}

bool QueryCapturingModel(uint64_t streamId, void** modelRiOut, bool* invalidated)
{
    if (modelRiOut == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_modelMutex);
    auto it = g_modelByStream.find(streamId);
    if (it == g_modelByStream.end()) {
        return false;
    }
    auto mIt = g_models.find(it->second);
    // 仅 CAPTURE 会话对外暴露 stream-capture 语义（ACTIVE）；GE 显式建图（NORMAL）
    // 在真实 CANN 语义下 CaptureGetInfo 恒为 NONE，此处保持一致
    if (mIt == g_models.end() || mIt->second.phase != ModelPhase::CAPTURING
        || mIt->second.modelType != RtModelType::RT_MODEL_CAPTURE_MODEL) {
        return false;
    }
    *modelRiOut = reinterpret_cast<void*>(static_cast<uintptr_t>(mIt->second.modelId));
    if (invalidated != nullptr) {
        *invalidated = mIt->second.captureInvalidated;
    }
    return true;
}

CaptureEventResult RecordCaptureEvent(uint64_t streamId, uint64_t eventId, uint64_t contextId, bool external)
{
    std::lock_guard<std::mutex> lock(g_modelMutex);
    // External Event 允许图外参与，不能借它把其他流隐式捕获进来。
    // 本阶段仅处理普通 Event 的模型归属，不改变 External Event 的原有桩行为。
    if (external) {
        return CaptureEventResult::NOT_CAPTURED;
    }
    ModelRecord* model = FindCapturingModel(streamId);
    auto previous = g_captureEvents.find(eventId);
    if (previous != g_captureEvents.end() && (model == nullptr || previous->second.modelId != model->modelId)) {
        auto previousModel = g_models.find(previous->second.modelId);
        if (previousModel != g_models.end() && previousModel->second.phase == ModelPhase::CAPTURING) {
            previousModel->second.captureInvalidated = true;
            if (model != nullptr) {
                model->captureInvalidated = true;
            }
            HCCL_VM_ERROR("Event Record crosses capture models, event={}, stream={}", eventId, streamId);
            return CaptureEventResult::ERROR;
        }
    }
    g_captureEvents.erase(eventId);
    if (model == nullptr) {
        return CaptureEventResult::NOT_CAPTURED;
    }
    if (model->captureInvalidated || model->contextId != contextId) {
        model->captureInvalidated = true;
        HCCL_VM_ERROR(
            "Invalid or unsupported capture Event Record, event={}, stream={}, external={}", eventId, streamId,
            external);
        return CaptureEventResult::ERROR;
    }

    CaptureEventRecord record;
    record.eventId = eventId;
    record.recordStreamId = streamId;
    const std::size_t recordIndex = model->captureEvents.size();
    model->captureEvents.push_back(std::move(record));
    g_captureEvents[eventId] = {model->modelId, recordIndex};
    HCCL_VM_INFO(
        "capture Event Record association, model={}, event={}, recordIndex={}, stream={}", model->modelId, eventId,
        recordIndex, streamId);
    return CaptureEventResult::CAPTURED;
}

CaptureEventResult WaitCaptureEvent(uint64_t streamId, uint64_t eventId, uint64_t contextId, bool external)
{
    std::lock_guard<std::mutex> lock(g_modelMutex);
    if (external) {
        return CaptureEventResult::NOT_CAPTURED;
    }
    ModelRecord* waitingModel = FindCapturingModel(streamId);
    auto event = g_captureEvents.find(eventId);
    if (event == g_captureEvents.end()) {
        // 本阶段不模拟 Event 执行：没有活动 Record 来源时不推断模型，保留原桩行为。
        return CaptureEventResult::NOT_CAPTURED;
    }
    auto modelIt = g_models.find(event->second.modelId);
    if (modelIt == g_models.end() || modelIt->second.phase != ModelPhase::CAPTURING) {
        return CaptureEventResult::ERROR;
    }
    ModelRecord& model = modelIt->second;
    const auto binding = g_modelByStream.find(streamId);
    if (model.captureInvalidated || streamId == 0 || contextId == 0 || model.contextId != contextId
        || (binding != g_modelByStream.end() && binding->second != model.modelId)) {
        model.captureInvalidated = true;
        if (waitingModel != nullptr) {
            waitingModel->captureInvalidated = true;
        }
        HCCL_VM_ERROR("Capture Event Wait conflict, event={}, stream={}, model={}", eventId, streamId, model.modelId);
        return CaptureEventResult::ERROR;
    }

    const std::size_t recordIndex = event->second.recordIndex;
    // Wait 不消耗 Record；同一次 Record 可使任意多个同上下文流加入同一模型。
    model.captureEvents[recordIndex].waitStreams.insert(streamId);
    model.boundStreams.insert(streamId);
    g_modelByStream[streamId] = model.modelId;
    HCCL_VM_INFO(
        "capture Event Wait association, model={}, event={}, recordIndex={}, stream={}", model.modelId, eventId,
        recordIndex, streamId);
    return CaptureEventResult::CAPTURED;
}

void ForgetCaptureEvent(uint64_t eventId)
{
    std::lock_guard<std::mutex> lock(g_modelMutex);
    g_captureEvents.erase(eventId);
}

void ReleaseCaptureBindings(const ModelRecord& model)
{
    for (uint64_t streamId : model.boundStreams) {
        auto binding = g_modelByStream.find(streamId);
        // 老图销毁时，原流可能已经属于另一个活动图。
        if (binding != g_modelByStream.end() && binding->second == model.modelId) {
            g_modelByStream.erase(binding);
        }
    }
    for (auto event = g_captureEvents.begin(); event != g_captureEvents.end();) {
        if (event->second.modelId == model.modelId) {
            event = g_captureEvents.erase(event);
        } else {
            ++event;
        }
    }
}

bool IsModelOfType(uint64_t modelId, RtModelType type)
{
    std::lock_guard<std::mutex> lock(g_modelMutex);
    auto it = g_models.find(modelId);
    return it != g_models.end() && it->second.modelType == type;
}

bool RecordAicpuKernelAction(
    uint64_t streamId, const char* kernelName, const char* soName, const uint8_t* argsBytes, uint32_t argsSize,
    uint64_t commId, uint32_t opDetailId)
{
    GraphAction action;
    action.type = GraphActionType::AICPU_KERNEL;
    action.streamId = streamId;
    action.kernelName = kernelName != nullptr ? kernelName : "";
    action.soName = soName != nullptr ? soName : "";
    if (argsBytes != nullptr && argsSize > 0) {
        action.argsBytes.assign(argsBytes, argsBytes + argsSize);
    }
    action.commId = commId;
    action.opDetailId = opDetailId;

    std::lock_guard<std::mutex> lock(g_modelMutex);
    auto it = g_modelByStream.find(streamId);
    if (it == g_modelByStream.end()) {
        HCCL_VM_WARN("record AicpuKernelAction but stream[{}] not bound", streamId);
        return false;
    }
    auto mIt = g_models.find(it->second);
    if (mIt == g_models.end() || mIt->second.phase != ModelPhase::CAPTURING || mIt->second.captureInvalidated) {
        HCCL_VM_WARN("record AicpuKernelAction but model[{}] not valid for capture", it->second);
        return false;
    }

    mIt->second.actions.push_back(std::move(action));

    // 首个 kernel action 的 opDetailId 作为采集期基准，重放时据此 clone 出 opIter 递增的新 opDetail。
    if (mIt->second.origOpDetailId == 0 && opDetailId != 0) {
        mIt->second.origOpDetailId = opDetailId;
    }
    RecordCapturedOpStream(mIt->second, opDetailId);

    HCCL_VM_INFO(
        "record AicpuKernelAction, stream[{}], kernel[{}], args[{}B], commId[{}], opDetailId[{}]", streamId,
        mIt->second.actions.back().kernelName, argsSize, commId, opDetailId);
    return true;
}

void RecordUnknownKernelAction(uint64_t streamId)
{
    std::lock_guard<std::mutex> lock(g_modelMutex);
    auto it = g_modelByStream.find(streamId);
    if (it == g_modelByStream.end()) {
        return;
    }
    auto mIt = g_models.find(it->second);
    if (mIt == g_models.end() || mIt->second.phase != ModelPhase::CAPTURING) {
        return;
    }
    GraphAction action;
    action.type = GraphActionType::UNKNOWN_KERNEL;
    action.streamId = streamId;
    mIt->second.actions.push_back(std::move(action));
    HCCL_VM_WARN(
        "record UnknownKernelAction, stream[{}] (PTA op with unresolved funcHandle, replay skips it)", streamId);
}

bool RecordThreadTaskAction(uint64_t streamId, const HcclTaskMetaData& taskMeta)
{
    std::lock_guard<std::mutex> lock(g_modelMutex);
    auto it = g_modelByStream.find(streamId);
    if (it == g_modelByStream.end()) {
        HCCL_VM_WARN("record thread-task but stream[{}] not bound", streamId);
        return false;
    }
    auto mIt = g_models.find(it->second);
    if (mIt == g_models.end() || mIt->second.phase != ModelPhase::CAPTURING || mIt->second.captureInvalidated) {
        HCCL_VM_WARN("record thread-task but model[{}] not valid for capture", it->second);
        return false;
    }

    GraphAction action;
    switch (taskMeta.taskType) {
        case HccLTaskMetaType::NOTIFY_RECORD:
            action.type = GraphActionType::NOTIFY_RECORD;
            break;
        case HccLTaskMetaType::NOTIFY_WAIT:
            action.type = GraphActionType::NOTIFY_WAIT;
            break;
        default:
            HCCL_VM_WARN(
                "record thread-task unsupported taskType[{}], stream[{}]", static_cast<int>(taskMeta.taskType),
                streamId);
            return false;
    }
    action.streamId = streamId;
    action.opDetailId = sim::operation::g_currOpDetailId;
    action.commId = taskMeta.commId;
    action.taskMeta = taskMeta; // 完整拷贝(含 streamId)，重放 InsertTaskToCollection 沿用该流
    mIt->second.actions.push_back(std::move(action));

    HCCL_VM_INFO(
        "record thread-task type[{}], stream[{}], notifyId[{}]", static_cast<int>(taskMeta.taskType), streamId,
        taskMeta.taskData.notify.notifyId);
    return true;
}

void RecordCcuLaunchAction(uint64_t streamId, const HcclTaskMetaData& taskMeta, uint32_t opDetailId)
{
    std::lock_guard<std::mutex> lock(g_modelMutex);
    auto it = g_modelByStream.find(streamId);
    if (it == g_modelByStream.end()) {
        HCCL_VM_WARN("record CcuLaunchAction but stream[{}] not bound", streamId);
        return;
    }
    auto mIt = g_models.find(it->second);
    if (mIt == g_models.end() || mIt->second.phase != ModelPhase::CAPTURING) {
        HCCL_VM_WARN("record CcuLaunchAction but model[{}] not CAPTURING", it->second);
        return;
    }

    GraphAction action;
    action.type = GraphActionType::CCU_LAUNCH;
    action.streamId = streamId;
    action.taskMeta = taskMeta; // 完整拷贝(含 streamId/commId/rankId/ccu 参数)，重放 InsertTaskToCollection 沿用该流
    action.opDetailId = opDetailId;
    mIt->second.actions.push_back(std::move(action));

    // CCU 模式无 AICPU KERNEL action，首个 CCU launch 的 opDetailId 作为采集期基准，
    // 供 PrepareReplayOpDetail 在 loop 重放时 clone 出 opIter 递增的新 opDetail。
    if (mIt->second.origOpDetailId == 0 && opDetailId != 0) {
        mIt->second.origOpDetailId = opDetailId;
    }
    RecordCapturedOpStream(mIt->second, opDetailId);

    HCCL_VM_INFO(
        "record CcuLaunchAction, stream[{}], dieId[{}], missionId[{}], instStartId[{}], instCnt[{}],"
        " opDetailId[{}]",
        streamId, taskMeta.taskData.ccu.dieId, taskMeta.taskData.ccu.missionId, taskMeta.taskData.ccu.instStartId,
        taskMeta.taskData.ccu.instCnt, opDetailId);
}

void RecordAivKernelAction(
    uint64_t streamId, const char* kernelName, const char* soName, uint32_t numBlocks, const uint8_t* hostArgsBytes,
    uint32_t hostArgsSize, uint64_t commId, uint32_t rankId, uint32_t opDetailId)
{
    std::lock_guard<std::mutex> lock(g_modelMutex);
    auto it = g_modelByStream.find(streamId);
    if (it == g_modelByStream.end()) {
        HCCL_VM_WARN("record AivKernelAction but stream[{}] not bound", streamId);
        return;
    }
    auto mIt = g_models.find(it->second);
    if (mIt == g_models.end() || mIt->second.phase != ModelPhase::CAPTURING) {
        HCCL_VM_WARN("record AivKernelAction but model[{}] not CAPTURING", it->second);
        return;
    }

    GraphAction action;
    action.type = GraphActionType::AIV_KERNEL;
    action.streamId = streamId;
    action.kernelName = kernelName != nullptr ? kernelName : "";
    action.soName = soName != nullptr ? soName : "";
    action.aivNumBlocks = numBlocks;
    action.aivRankId = rankId;
    action.commId = commId;
    action.opDetailId = opDetailId;
    // AivHostLaunchArgs 为 POD（标量 + ExtraArgs 数组 + 指针），整块字节深拷贝即可；
    // 其中的 buffersIn/input/output 等指针值是 device 虚拟地址，模型重放期复用同一地址空间仍有效。
    if (hostArgsBytes != nullptr && hostArgsSize > 0) {
        action.aivHostArgs.assign(hostArgsBytes, hostArgsBytes + hostArgsSize);
    }
    mIt->second.actions.push_back(std::move(action));

    // AIV 模式无 AICPU KERNEL action，首个 AIV launch 的 opDetailId 作为采集期基准，
    // 供 PrepareReplayOpDetail 在 loop 重放时 clone 出 opIter 递增的新 opDetail。
    if (mIt->second.origOpDetailId == 0 && opDetailId != 0) {
        mIt->second.origOpDetailId = opDetailId;
    }
    RecordCapturedOpStream(mIt->second, opDetailId);

    HCCL_VM_INFO(
        "record AivKernelAction, stream[{}], kernel[{}], numBlocks[{}], hostArgs[{}B],"
        " commId[{}], rankId[{}], opDetailId[{}]",
        streamId, mIt->second.actions.back().kernelName, numBlocks, hostArgsSize, commId, rankId, opDetailId);
}

void WaitStreamTasksDone(uint64_t streamId)
{
    auto pluginRows = sim::runtime::Db::GetByPred<sim::runtime::Plugin>(
        HcclSim::Storage::Eq(&sim::runtime::Plugin::tag, std::string("runner")));
    if (!pluginRows.ok() || !pluginRows.value.has_value() || pluginRows.value->empty()) {
        // Runner 未启动：无消费方，直接返回
        return;
    }

    while (true) {
        uint64_t totalTasks = 0;
        uint64_t doneTasks = 0;
        if (sim::operation::QueryOpTaskCompletionByStreamId(streamId, totalTasks, doneTasks) != 0) {
            HCCL_VM_ERROR("QueryOpTasksByStreamId fail, streamId={}", streamId);
            return;
        }
        HCCL_VM_DEBUG("Waiting for stream[{}]... {}/{}", streamId, doneTasks, totalTasks);
        if (doneTasks == totalTasks) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void InsertModelExecTask(uint32_t execSeq, uint8_t role, uint64_t modelId, uint64_t streamId, uint64_t peerStreamId)
{
    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::MODEL_EXEC;
    taskMetaData.commId = 0;
    taskMetaData.deviceId = sim::runtime::GetCurrDeviceId();
    taskMetaData.rankId = UINT32_MAX;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.modelExec.execSeq = execSeq;
    taskMetaData.taskData.modelExec.role = role;
    taskMetaData.taskData.modelExec.modelId = modelId;
    taskMetaData.taskData.modelExec.peerStreamId = peerStreamId;

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("InsertTaskToCollection MODEL_EXEC fail, role={}, streamId={}", role, streamId);
        return;
    }
    HCCL_VM_INFO(
        "add MODEL_EXEC task, execSeq={}, role={}, model={}, streamId={}, peerStreamId={}", execSeq, role, modelId,
        streamId, peerStreamId);
}

bool PrepareModelReplay(uint64_t modelId, ModelReplayContext& context)
{
    uint32_t replayCount = 0;
    context = ModelReplayContext{};
    {
        std::lock_guard<std::mutex> lock(g_modelMutex);
        auto it = g_models.find(modelId);
        if (it == g_models.end()) {
            return false;
        }
        context.actions = it->second.actions;
        context.streams = it->second.boundStreams;
        context.streams.insert(it->second.headStreamId);
        replayCount = it->second.replayCount;
    }
    for (const auto& action : context.actions) {
        // GE 解绑只结束采集归属，不应丢失已录制任务的执行流。
        context.streams.insert(action.streamId);
        if (action.type == GraphActionType::UNKNOWN_KERNEL) {
            continue;
        }
        if (action.opDetailId == 0) {
            HCCL_VM_ERROR("model[{}] action has no captured operator identity", modelId);
            return false;
        }
        if (context.opDetailIds.count(action.opDetailId) == 0) {
            const uint32_t replayId = replayCount == 0 ? action.opDetailId : CloneOpDetailForReplay(action.opDetailId);
            context.opDetailIds.emplace(action.opDetailId, replayId);
            if (context.boundaryOpDetailId == 0) {
                context.boundaryOpDetailId = replayId;
            }
        }
    }
    if (context.actions.empty()) {
        // 兼容数据量为0等空算子场景：无任何 action 可重放，仅告警，不再回放边界任务。
        HCCL_VM_WARN("model[{}] has no captured action (empty/no-op graph), skip replay boundary tasks", modelId);
    } else {
        sim::operation::g_currOpDetailId = context.boundaryOpDetailId;
    }
    return true;
}

int ReplayModelActions(uint64_t modelId, const ModelReplayContext& context)
{
    for (const auto& action : context.actions) {
        const auto identity = context.opDetailIds.find(action.opDetailId);
        const uint32_t replayOpDetailId = identity == context.opDetailIds.end() ? 0 : identity->second;
        if (action.type != GraphActionType::UNKNOWN_KERNEL) {
            sim::operation::g_currOpDetailId = replayOpDetailId;
        }
        switch (action.type) {
            case GraphActionType::AICPU_KERNEL:
                // 一期：重发 EXEC_KERNEL，device 进程重新跑 kernel，重新产数据面任务
                LaunchAicpuKernelRaw(
                    action.kernelName.c_str(), action.soName.c_str(),
                    action.argsBytes.empty() ? nullptr : action.argsBytes.data(),
                    static_cast<uint32_t>(action.argsBytes.size()), replayOpDetailId);
                break;
            case GraphActionType::AIV_KERNEL: {
                // 重放 = 重新走 AIV 完整下发路径（与 aclrtLaunchKernelWithHostArgs 正常下发共用
                // LaunchAivKernelRaw）：launchIdx 重自增 + VirtualExecuteAivKernel 重新跑 x86 AIV stub
                // + 重插 AIV_GRAPH，关联本次重放的 opDetail；runner 的 TaskAivGraph 据此重新跑 AIV 仿真。
                sim::operation::g_currOpDetailId = replayOpDetailId;
                LaunchAivKernelRaw(
                    action.kernelName.c_str(), action.soName.c_str(), action.aivNumBlocks,
                    action.aivHostArgs.empty() ? nullptr : action.aivHostArgs.data(),
                    static_cast<uint32_t>(action.aivHostArgs.size()), action.commId, action.aivRankId, action.streamId,
                    replayOpDetailId);
                HCCL_VM_INFO(
                    "replay AIV_KERNEL, stream[{}], kernel[{}], numBlocks[{}]", action.streamId, action.kernelName,
                    action.aivNumBlocks);
                break;
            }
            case GraphActionType::UNKNOWN_KERNEL:
                // aclGraph：PTA 算子（funcHandle 不可解析）仅作占位，重放跳过（流程穿刺豁免项）
                HCCL_VM_WARN("replay UNKNOWN_KERNEL skipped, stream[{}]", action.streamId);
                break;
            case GraphActionType::NOTIFY_RECORD: {
                // 二期：重放 = 重插 NOTIFY_RECORD（taskMeta），关联本次重放的 opDetail
                HcclTaskMetaData taskMeta = action.taskMeta;
                sim::operation::g_currOpDetailId = replayOpDetailId;
                uint32_t index{0};
                auto insRet = InsertTaskToCollection(&taskMeta, &index);
                if (insRet != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
                    HCCL_VM_ERROR("replay NOTIFY_RECORD insert fail, stream[{}]", action.streamId);
                }
                HCCL_VM_INFO(
                    "replay NOTIFY_RECORD, stream[{}], notifyId[{}]", action.streamId,
                    taskMeta.taskData.notify.notifyId);
                break;
            }
            case GraphActionType::NOTIFY_WAIT: {
                // 二期：重放 = reset notify + 重插 NOTIFY_WAIT（taskMeta）
                uint64_t notifyId = action.taskMeta.taskData.notify.notifyId;
                (void)sim::runtime::Db::Update<sim::runtime::Notify>(
                    HcclSim::Storage::Eq(&sim::runtime::Notify::id, notifyId),
                    HcclSim::Storage::Set(&sim::runtime::Notify::value, 0));

                HcclTaskMetaData taskMeta = action.taskMeta;
                sim::operation::g_currOpDetailId = replayOpDetailId;
                uint32_t index{0};
                auto insRet = InsertTaskToCollection(&taskMeta, &index);
                if (insRet != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
                    HCCL_VM_ERROR("replay NOTIFY_WAIT insert fail, stream[{}]", action.streamId);
                }
                HCCL_VM_INFO(
                    "replay NOTIFY_WAIT, stream[{}], notifyId[{}]", action.streamId, taskMeta.taskData.notify.notifyId);
                break;
            }
            case GraphActionType::CCU_LAUNCH: {
                // 重放 = 重新走 CCU 完整下发路径（与 rtCCULaunch 正常下发共用 LaunchCcuKernelRaw）：
                // index 重分配 + GenCaModelCcuToml + sqe dump + 重插 CCU_GRAPH，关联本次重放的 opDetail；
                // runner 的 TaskCcuGraph 据此重新跑 CCU 仿真、重新产数据面行为。
                sim::operation::g_currOpDetailId = replayOpDetailId;
                LaunchCcuKernelRaw(action.taskMeta);
                HCCL_VM_INFO(
                    "replay CCU_LAUNCH, stream[{}], dieId[{}], missionId[{}], instStartId[{}], instCnt[{}]",
                    action.streamId, action.taskMeta.taskData.ccu.dieId, action.taskMeta.taskData.ccu.missionId,
                    action.taskMeta.taskData.ccu.instStartId, action.taskMeta.taskData.ccu.instCnt);
                break;
            }
            case GraphActionType::MEM_CPY:
            case GraphActionType::REDUCE:
            case GraphActionType::SYNC_STREAM:
                // 二期：InsertTaskToCollection(&action.taskMeta, &idx)
                HCCL_VM_WARN(
                    "replay thread-task type[{}] not implemented yet, stream[{}]", static_cast<int>(action.type),
                    action.streamId);
                break;
            case GraphActionType::EVENT_RECORD:
            case GraphActionType::EVENT_WAIT:
                // 二期：sim::Event 状态标记
                HCCL_VM_WARN(
                    "replay event type[{}] not implemented yet, stream[{}]", static_cast<int>(action.type),
                    action.streamId);
                break;
            default:
                HCCL_VM_WARN("replay unknown action type[{}]", static_cast<int>(action.type));
                break;
        }
    }

    // 末尾按 boundStreams 逐流等待（AICPU 下从流无任务，秒回；master_stream 有任务则等到完成）
    for (uint64_t streamId : context.streams) {
        WaitStreamTasksDone(streamId);
    }
    {
        std::lock_guard<std::mutex> lock(g_modelMutex);
        auto it = g_models.find(modelId);
        if (it != g_models.end()) {
            it->second.replayCount++;
            HCCL_VM_INFO(
                "replayed model[{}] done, replayCount={}, opDetailId={}", modelId, it->second.replayCount,
                context.boundaryOpDetailId);
        }
    }
    return 0;
}

} // extern "C"
