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
#define HCCL_VM_MODULE "ACLGRAPH_STUB"

#include <cstdint>

#include "acl/acl_base.h"
#include "acl/acl_base_rt.h"
#include "acl/acl_rt.h"
#include "graph_capture.h"
#include "runtime/base.h"
#include "sim_log.h"

// ============================================================================
// aclGraph（stream-capture / aclmdlRI）入口拦截桩 —— 第一阶段（算子级录制）
//
// 对齐 runtime 的类型区分（runtime core/inc/model/model.hpp:38 的 ModelType）：
//   - CAPTURE 会话仅由 aclmdlRICaptureBegin 内部创建（等价 runtime
//     StreamBeginCapture 里 ModelCreate(RT_MODEL_CAPTURE_MODEL)，context_aclgraph.cc:325）；
//   - rtModelCreate 直建的 NORMAL 会话不受本文件影响（GE/hlt 路径零回归，
//     CaptureGetInfo 对 NORMAL 会话保持 0916_G 现状 NONE）。
//
// 一期范围（评审定版）：
//   - 算子级录制：捕获流上的 launch（WithConfig/WithHostArgs）录为 action；
//   - Event 仅记录采集关联：普通 Event 的 Wait 将等待流并入 Record 所属模型，
//     不生成 Event Task、不转成 Notify，不模拟时间戳/置位/复位等执行语义；
//   - 数值正确性不做承诺（UNKNOWN 算子仅占位，重放告警跳过）。
//
// 句柄约定：aclmdlRI = (aclmdlRI)(uintptr_t)modelId，与 rtModel* 桩同域，
// rtStreamAddToModel（HCCL CaptureSlaveStreams 从流并入）因此直接可用。
// ============================================================================

namespace {

constexpr aclError kAclInvalidParam = ACL_ERROR_INVALID_PARAM;
constexpr aclError kAclInternalError = ACL_ERROR_INTERNAL_ERROR;

} // namespace

// 转调 aclrt_model_stub.cc 导出的 rtModel* 桩（跨 TU 需要与定义侧一致的 extern "C" 声明；
// 这些符号同时是桩库导出的拦截符号，真实 runtime 的同名实现被符号优先级覆盖）。
extern "C" {
rtError_t rtModelBindStream(rtModel_t mdl, rtStream_t stm, uint32_t flag);
rtError_t rtEndGraph(rtModel_t mdl, rtStream_t stm);
rtError_t rtModelLoadComplete(rtModel_t mdl);
rtError_t rtModelExecute(rtModel_t mdl, rtStream_t stm, uint32_t flag);
rtError_t rtModelDestroy(rtModel_t mdl);
}

extern "C" {

// 开始捕获：内部创建 CAPTURE 会话并注册捕获流为头流。
// 对齐 runtime：capture model 不由用户创建，等价 StreamBeginCapture 内部的
// ModelCreate(RT_MODEL_CAPTURE_MODEL) + ModelBindStream(RT_HEAD_STREAM)。
aclError aclmdlRICaptureBegin(aclrtStream stream, aclmdlRICaptureMode mode)
{
    if (stream == nullptr) {
        HCCL_VM_ERROR("aclmdlRICaptureBegin with null stream");
        return kAclInvalidParam;
    }
    const uint64_t streamId = (uint64_t)(uintptr_t)stream;
    const uint64_t modelId = CreateModelRecord(RtModelType::RT_MODEL_CAPTURE_MODEL);
    if (modelId == 0) {
        return kAclInternalError;
    }
    // flag=0 对应 RT_HEAD_STREAM：捕获流即 master_stream（重放时的从流）
    const rtError_t ret = rtModelBindStream((rtModel_t)(uintptr_t)modelId, stream, 0U);
    if (ret != RT_ERROR_NONE) {
        // 流已被其他会话绑定（重复 capture 同一流），回滚会话
        HCCL_VM_ERROR(
            "aclmdlRICaptureBegin bind stream[{}] failed, ret={}, rollback model={}", streamId, static_cast<int>(ret),
            modelId);
        (void)rtModelDestroy((rtModel_t)(uintptr_t)modelId);
        return kAclInvalidParam;
    }
    HCCL_VM_INFO(
        "aclmdlRICaptureBegin stream[{}] mode[{}] model[{}], capture session started", streamId, static_cast<int>(mode),
        modelId);
    return ACL_SUCCESS;
}

// 捕获状态查询：仅 CAPTURE 会话在 CAPTURING 期返回 ACTIVE；
// NORMAL（GE 显式建图）会话恒 NONE，与真实 CANN 语义一致（rtStreamGetCaptureInfo
// 的状态只由 BeginCapture/EndCapture 维护，显式建图不触碰）。
aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus* status, aclmdlRI* modelRI)
{
    // GLOBAL 是捕获模式，不意味着所有流自动属于同一个图。
    // PTA 的通信流通过 Event Wait 传播归属；查询与实际任务录制统一按流判定。
    void* ri = nullptr;
    bool invalidated = false;
    const bool active = QueryCapturingModel((uint64_t)(uintptr_t)stream, &ri, &invalidated);
    if (status != nullptr) {
        *status = !active ?
                      ACL_MODEL_RI_CAPTURE_STATUS_NONE :
                      (invalidated ? ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED : ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE);
    }
    if (modelRI != nullptr) {
        *modelRI = active ? static_cast<aclmdlRI>(ri) : nullptr;
    }
    return ACL_SUCCESS;
}

// 结束捕获：收口 CAPTURE 会话（等价 rtEndGraph + rtModelLoadComplete），
// 经 modelRI 归还句柄，后续 ExecuteAsync/GetId/Destroy 均基于该句柄。
aclError aclmdlRICaptureEnd(aclrtStream stream, aclmdlRI* modelRI)
{
    if (stream == nullptr || modelRI == nullptr) {
        HCCL_VM_ERROR("aclmdlRICaptureEnd with null input");
        return kAclInvalidParam;
    }
    const uint64_t streamId = (uint64_t)(uintptr_t)stream;
    void* ri = nullptr;
    if (!QueryCapturingModel(streamId, &ri)) {
        // 对齐真实语义 RT_ERROR_STREAM_NOT_CAPTURED（107029）
        HCCL_VM_WARN("aclmdlRICaptureEnd on stream[{}] without active capture", streamId);
        *modelRI = nullptr;
        return kAclInvalidParam;
    }
    const uint64_t modelId = (uint64_t)(uintptr_t)ri;
    *modelRI = nullptr;
    if (rtEndGraph((rtModel_t)(uintptr_t)modelId, stream) != RT_ERROR_NONE
        || rtModelLoadComplete((rtModel_t)(uintptr_t)modelId) != RT_ERROR_NONE) {
        HCCL_VM_ERROR("aclmdlRICaptureEnd failed, stream={}, model={}", streamId, modelId);
        return kAclInternalError;
    }
    *modelRI = static_cast<aclmdlRI>(ri);
    HCCL_VM_INFO("aclmdlRICaptureEnd stream[{}] model[{}], session sealed", streamId, modelId);
    return ACL_SUCCESS;
}

// 重放：转调 rtModelExecute（MODEL_EXEC 4 边界任务 + ReplayModelActions 均复用 GE 一期实现）。
// torch_npu 允许在任意流重放（含默认流），入参流即重放触发流（mainStreamId）。
aclError aclmdlRIExecuteAsync(aclmdlRI modelRI, aclrtStream stream)
{
    if (modelRI == nullptr) {
        HCCL_VM_ERROR("aclmdlRIExecuteAsync with null modelRI");
        return kAclInvalidParam;
    }
    const uint64_t modelId = (uint64_t)(uintptr_t)modelRI;
    if (!IsModelOfType(modelId, RtModelType::RT_MODEL_CAPTURE_MODEL)) {
        HCCL_VM_ERROR("aclmdlRIExecuteAsync on non-capture model[{}]", modelId);
        return kAclInvalidParam;
    }
    const rtError_t ret = rtModelExecute((rtModel_t)(uintptr_t)modelId, stream, 0U);
    if (ret != RT_ERROR_NONE) {
        HCCL_VM_ERROR("aclmdlRIExecuteAsync model[{}] replay failed, rtRet={}", modelId, static_cast<int>(ret));
        return kAclInternalError;
    }
    return ACL_SUCCESS;
}

// 销毁 CAPTURE 会话（torch_npu NPUGraph::reset）。仅允许销毁本路径创建的会话，
// 防止句柄误用误删 GE 显式建图的 NORMAL 模型。
aclError aclmdlRIDestroy(aclmdlRI modelRI)
{
    if (modelRI == nullptr) {
        HCCL_VM_ERROR("aclmdlRIDestroy with null modelRI");
        return kAclInvalidParam;
    }
    const uint64_t modelId = (uint64_t)(uintptr_t)modelRI;
    if (!IsModelOfType(modelId, RtModelType::RT_MODEL_CAPTURE_MODEL)) {
        HCCL_VM_ERROR("aclmdlRIDestroy on non-capture model[{}]", modelId);
        return kAclInvalidParam;
    }
    const rtError_t ret = rtModelDestroy((rtModel_t)(uintptr_t)modelId);
    if (ret != RT_ERROR_NONE) {
        HCCL_VM_ERROR("aclmdlRIDestroy model[{}] failed, rtRet={}", modelId, static_cast<int>(ret));
        return kAclInternalError;
    }
    HCCL_VM_INFO("aclmdlRIDestroy model[{}]", modelId);
    return ACL_SUCCESS;
}

// 会话 id：runtime 语义即 Model::Id_()；桩句柄本身就是自增 modelId，直接截断返回。
aclError aclmdlRIGetId(aclmdlRI modelRI, uint32_t* modelRIId)
{
    if (modelRI == nullptr || modelRIId == nullptr) {
        HCCL_VM_ERROR("aclmdlRIGetId with null input");
        return kAclInvalidParam;
    }
    *modelRIId = (uint32_t)(uintptr_t)modelRI;
    HCCL_VM_DEBUG("aclmdlRIGetId model[{}] -> id[{}]", (uint64_t)(uintptr_t)modelRI, *modelRIId);
    return ACL_SUCCESS;
}

// 捕获线程模式交换：桩不做线程级 capture mode 管理（GLOBAL/RELAXED 的差异
// 只影响真实 runtime 对危险 API 的拦截面），恒成功放行。
// 依赖点：HCCL 捕获中同步 memcpy/memset 的包装（adapter_acl.cc）在捕获检测
// 命中后会调用本接口，返回失败会直接打断 HCCL 下发，因此必须放行。
aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode* mode)
{
    (void)mode;
    return ACL_SUCCESS;
}

} // extern "C"
