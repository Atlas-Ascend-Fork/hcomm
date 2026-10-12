/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LEVEL2_GRAPH_CAPTURE_H
#define LEVEL2_GRAPH_CAPTURE_H

#include <cstdint>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "sim_common_defs.h" // HcclTaskMetaData / HccLTaskMetaType

// ============================================================================
// 图模式采集-重放数据模型
// 执行真源只有进程内 ModelRecord（g_models/g_modelByStream），不建 DB model 表；
// 重放产物 = 重新走 launch 桩 -> 新 task 进 sim::Task 集合 -> runner/checker/insight 照常消费。
//
// 一期(P0)只实现 AICPU_KERNEL 的录制与重放；其余 action 类型在此一次性铺好，
// 供二期（AIV/CCU + thread 任务 notify/event/memcpy/reduce/sync）直接扩展。
//
// aclGraph（stream-capture）接入：对齐 runtime Model::modelType_ 的类型区分
// （runtime core/inc/model/model.hpp:38）——rtModel* 显式建图 = NORMAL，
// aclmdlRICaptureBegin 内部创建 = CAPTURE_MODEL；对外语义（CaptureGetInfo）仅对
// CAPTURE 会话暴露 ACTIVE，GE 路径保持 NONE（与真实 CANN 语义一致）。
// ============================================================================

// 对齐 runtime ModelType（model.hpp:38）：两种模型创建入口与对外语义不同
enum class RtModelType : uint8_t {
    RT_MODEL_NORMAL = 0,        // GE/hlt 显式建图：rtModelCreate 直建
    RT_MODEL_CAPTURE_MODEL = 1, // aclGraph stream-capture：aclmdlRICaptureBegin 内部建
};

enum class ModelPhase : uint8_t {
    NEW,       // rtModelCreate / rtStreamBeginCapture 后
    CAPTURING, // 首个流注册后，采集任务
    SEALED,    // rtEndGraph 后，停止采集
    LOADED,    // rtModelLoadComplete 后
    DESTROYED, // rtModelDestroy 后
};

// rtModelExecute 两级重放边界任务角色（写入 HcclTaskMetaData.taskData.modelExec.role，
// checker 据此配对建模：主流 Start -> 从流 StartSub、从流 EndSub -> 主流 End）。
enum ModelExecRole : uint8_t {
    MODEL_EXEC_START = 0,     // 主流(resources_str.stream) 重放开始
    MODEL_EXEC_START_SUB = 1, // 从流(master_stream) 重放开始
    MODEL_EXEC_END_SUB = 2,   // 从流(master_stream) 重放结束
    MODEL_EXEC_END = 3,       // 主流(resources_str.stream) 重放结束
};

// action 类型：一期 AICPU + 二期 launch 类(AIV/CCU) + 二期 thread 任务(NOTIFY/MEM_CPY/REDUCE/SYNC) + event
enum class GraphActionType : uint8_t {
    AICPU_KERNEL, // 一期：重放 = 重发 EXEC_KERNEL（device 重新跑 kernel）
    AIV_KERNEL,   // 二期：重放 = VirtualExecuteAivKernel + 重插 AIV_GRAPH（launchIdx 重自增）
    UNKNOWN_KERNEL, // aclGraph：funcHandle 无法解析的算子（如 PTA RNG fill），仅录占位，重放告警跳过
    CCU_LAUNCH,    // 二期：重放 = 重新走 CCU 下发路径（index 重分配 + toml + dump + 重插 CCU_GRAPH）
    NOTIFY_RECORD, // 二期 thread：重放 = 重插 NOTIFY_RECORD（taskMeta）
    NOTIFY_WAIT,   // 二期 thread：重放 = reset notify + 重插 NOTIFY_WAIT（taskMeta）
    MEM_CPY,       // 二期 thread：重放 = 重插 MEM_CPY（taskMeta）
    REDUCE,        // 二期 thread：重放 = 重插 REDUCE（taskMeta）
    SYNC_STREAM,   // 二期 thread：重放 = 重插 SYNC_STREAM（taskMeta，syncIdx 重算）
    EVENT_RECORD,  // 预留：Event 执行语义尚未实现，当前仅记录采集归属，不生成 action/task
    EVENT_WAIT,    // 预留：Event 一对多依赖不能直接当作二值 Notify 消费
};

// 每个动作在采集期固化的原料；重放期只读它，不读 thread_local/全局临时量。
struct GraphAction {
    GraphActionType type = GraphActionType::AICPU_KERNEL;
    uint64_t streamId = 0; // 提交到哪条流（重放沿用，忽略 rtModelExecute 入参流）

    // ---- 一期 AICPU_KERNEL：kernel 原料，重放重发 EXEC_KERNEL ----
    // kernelName/soName 对 AICPU_KERNEL 与 AIV_KERNEL 通用
    std::string kernelName;
    std::string soName;
    std::vector<uint8_t> argsBytes; // AICPU：OpParam + varMem 深拷贝（不引用采集期栈地址）
    uint64_t commId = 0;
    uint32_t opDetailId = 0;

    // ---- 二期 AIV_KERNEL：host 侧重算所需原料 ----
    uint32_t aivNumBlocks = 0;
    uint32_t aivRankId = UINT32_MAX;
    std::vector<uint8_t> aivHostArgs; // AivHostLaunchArgs 深拷贝

    // ---- 二期 CCU_LAUNCH / NOTIFY_RECORD / NOTIFY_WAIT / MEM_CPY / REDUCE / SYNC_STREAM ----
    // 统一存完整 taskMeta（自带 taskType + commId/rankId/deviceId/streamId + taskData），
    // 重放 = InsertTaskToCollection(&taskMeta, &idx)。
    HcclTaskMetaData taskMeta;

    // ---- 预留 EVENT_RECORD / EVENT_WAIT ----
    uint64_t eventId = 0;
};

// 一次 Event Record 的采集关联。每次重录追加一个版本，多个 Wait 共享该版本，
// 不消费/复位此记录；仅用于流归属传播，不表示 Event 已执行或生成了时间戳。
struct CaptureEventRecord {
    uint64_t eventId = 0;
    uint64_t recordStreamId = 0;
    std::set<uint64_t> waitStreams;
};

// 进程内单例：执行真源（不持久化，不建 DB 表）
struct ModelRecord {
    uint64_t modelId = 0; // 句柄 = 进程内自增 id（(rtModel_t)modelId）
    ModelPhase phase = ModelPhase::NEW;
    RtModelType modelType = RtModelType::RT_MODEL_NORMAL; // 对齐 runtime Model::modelType_
    uint64_t headStreamId = 0;                            // flag=0 绑定的头流（master_stream）
    uint64_t contextId = 0; // CAPTURE 会话所属上下文，禁止 Event 把其他上下文的流并入
    bool captureInvalidated = false;
    std::vector<CaptureEventRecord> captureEvents; // 保留各次 Record/Wait 关联，独立于任务 actions
    std::set<uint64_t> boundStreams;               // 头流 + Event Wait 隐式加入流 + rtStreamAddToModel 从流
    std::vector<GraphAction> actions;              // 只收"提交到已注册流"的动作
    uint32_t origOpDetailId = 0;     // 采集期固化 opDetailId（首个 AICPU_KERNEL action 的 opDetailId）
    uint64_t capturedOpStreamId = 0; // aclGraph 采集期 opDetails.streamId（算子真实主流）
    uint32_t replayCount = 0; // 已重放次数：0 复用采集期 id(opIter=0)，>=1 clone 出 opIter 递增的新 id
};

// 每次执行独立快照；采集模板不随回放改写。
struct ModelReplayContext {
    std::vector<GraphAction> actions;
    std::set<uint64_t> streams;
    std::unordered_map<uint32_t, uint32_t> opDetailIds;
    uint32_t boundaryOpDetailId = 0;
};

extern std::unordered_map<uint64_t, ModelRecord> g_models;     // modelId -> ModelRecord
extern std::unordered_map<uint64_t, uint64_t> g_modelByStream; // 活动采集/GE 绑定；CAPTURE 结束后释放
extern uint64_t g_nextModelId;                                 // 自增句柄
extern uint32_t g_nextModelExecSeq; // rtModelExecute 自增 execSeq（同一次 4 任务共享）
// g_models/g_modelByStream 的进程内互斥保护：torch 主线程（aclmdlRI* 入口）、
// HCCL 调用线程（launch 采集 / rtStreamAddToModel）、PTA taskqueue worker
// （WithHostArgs 采集）三方并发访问。所有访问必须经 g_modelMutex。
extern std::mutex g_modelMutex;

enum class CaptureEventResult : uint8_t {
    NOT_CAPTURED,
    CAPTURED,
    ERROR,
};

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// ── 模型生命周期（rtModel* 显式建图 与 aclmdlRI* stream-capture 共用） ─────────

// 按类型创建 ModelRecord（对齐 runtime Context::ModelCreate 的工厂语义：
// type=CAPTURE_MODEL 时行为等价 new CaptureModel()）。返回 modelId，0=失败。
uint64_t CreateModelRecord(RtModelType type);

// 采集态判定：stream 是否已注册进某个 CAPTURING 的 model（launch 桩采集分支的唯一判据）
bool IsCapturingStream(uint64_t streamId);

// 查询 stream 所属的 CAPTURE 会话（aclmdlRICaptureGetInfo 专用）：
// 命中（modelType==CAPTURE_MODEL 且 phase==CAPTURING）返回 true，并输出模型及失效状态。
bool QueryCapturingModel(uint64_t streamId, void** modelRiOut, bool* invalidated = nullptr);

// 仅维护普通 Event 的活动 Record/Wait 关联；不会产生 GraphAction 或 HcclTaskMetaData。
// 无采集关联及 External Event 返回 NOT_CAPTURED，由桩保留原有执行行为。
CaptureEventResult RecordCaptureEvent(uint64_t streamId, uint64_t eventId, uint64_t contextId, bool external);
CaptureEventResult WaitCaptureEvent(uint64_t streamId, uint64_t eventId, uint64_t contextId, bool external);
void ForgetCaptureEvent(uint64_t eventId);

// 调用方持有 g_modelMutex。结束活动归属但保留 actions/events，允许原流采集另一个图。
void ReleaseCaptureBindings(const ModelRecord& model);

// model 类型校验（aclmdlRI* / rtModel* 入口的防御性分支，镜像 runtime GetModelType() 检查）
bool IsModelOfType(uint64_t modelId, RtModelType type);

// AICPU 采集：把一次 kernel launch 录成 AicpuKernelAction（IsCapturing 时调用，不执行）
bool RecordAicpuKernelAction(
    uint64_t streamId, const char* kernelName, const char* soName, const uint8_t* argsBytes, uint32_t argsSize,
    uint64_t commId, uint32_t opDetailId);

// thread 任务采集：把 host 侧同步类任务(NOTIFY_RECORD/NOTIFY_WAIT 等)录成对应 GraphAction，
// 重放期由 ReplayModelActions 重插 taskMeta；采集态不立即进 DB，归属本次重放的 opDetail。
// Event 的跨流执行依赖尚未建模，不能将 DB 插入顺序解释为跨流先后关系。
bool RecordThreadTaskAction(uint64_t streamId, const HcclTaskMetaData& taskMeta);

// UNKNOWN 采集：funcHandle 无法解析的算子（aclGraph 捕获区内 PTA 算子，如 RNG fill）。
// 仅录占位保证流程穿刺，重放告警跳过；正确性不在一期承诺范围。
void RecordUnknownKernelAction(uint64_t streamId);

// CCU 采集：把一次 CCU SQE launch 录成 CCU_LAUNCH action（完整 taskMeta + 当前 opDetailId），
// 采集态不立即进 DB，重放期由 ReplayModelActions 在 START_SUB/END_SUB 之间重插并关联本次重放 opDetail。
// 与 AICPU_KERNEL 同粒度：首个 CCU launch 的 opDetailId 作为采集期基准，供 loop 重放 clone 出递增 opIter。
void RecordCcuLaunchAction(uint64_t streamId, const HcclTaskMetaData& taskMeta, uint32_t opDetailId);

// AIV 采集：把一次 AIV kernel launch 录成 AIV_KERNEL action（kernelName/soName/numBlocks + AivHostLaunchArgs
// 深拷贝 + commId/rankId/opDetailId），采集态不执行 kernel 也不插 AIV_GRAPH，
// 重放期由 ReplayModelActions 调 LaunchAivKernelRaw 重新虚拟执行并重插 AIV_GRAPH。
// hostArgsBytes 是采集期规范化后的 AivHostLaunchArgs 的字节拷贝（POD，重放期按同大小反序列化）。
void RecordAivKernelAction(
    uint64_t streamId, const char* kernelName, const char* soName, uint32_t numBlocks, const uint8_t* hostArgsBytes,
    uint32_t hostArgsSize, uint64_t commId, uint32_t rankId, uint32_t opDetailId);

// AICPU 重发核心：把原料重发为 EXEC_KERNEL（采集期立即执行 + 重放期共用）
void LaunchAicpuKernelRaw(
    const char* kernelName, const char* soName, const uint8_t* argsBytes, uint32_t argsSize, uint32_t opDetailId);

// CCU 重发核心：把录制的 CCU_GRAPH 原料重新走完整下发路径（index 重分配 + GenCaModelCcuToml +
// 写 sqe dump + InsertTaskToCollection），runner 重新跑 CCU 仿真，重新产数据面行为。
// 与 AICPU LaunchAicpuKernelRaw 对齐：正常下发(rtCCULaunch)与重放(ReplayModelActions)共用该核心。
void LaunchCcuKernelRaw(const HcclTaskMetaData& taskMeta);

// AIV 重发核心：把录制的 AIV_KERNEL 原料重新走完整下发路径（launchIdx 重自增 + VirtualExecuteAivKernel
// 重新跑 x86 AIV stub + InsertTaskToCollection 重插 AIV_GRAPH），关联本次重放的 opDetail。
// 与正常下发 aclrtLaunchKernelWithHostArgs 对齐：正常下发与重放共用同一虚拟执行核心。
void LaunchAivKernelRaw(
    const char* kernelName, const char* soName, uint32_t numBlocks, const uint8_t* hostArgsBytes, uint32_t hostArgsSize,
    uint64_t commId, uint32_t rankId, uint64_t streamId, uint32_t opDetailId);

// 插入一个 MODEL_EXEC 边界任务（需求2）：streamId 决定落在主流还是从流
void InsertModelExecTask(uint32_t execSeq, uint8_t role, uint64_t modelId, uint64_t streamId, uint64_t peerStreamId);

// 快照并逐算子准备回放 ID；首次复用，后续分别克隆。
bool PrepareModelReplay(uint64_t modelId, ModelReplayContext& context);
int ReplayModelActions(uint64_t modelId, const ModelReplayContext& context);

// 等待 stream 上所有 task 完成（runner 未启动时秒回）
void WaitStreamTasksDone(uint64_t streamId);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // LEVEL2_GRAPH_CAPTURE_H
