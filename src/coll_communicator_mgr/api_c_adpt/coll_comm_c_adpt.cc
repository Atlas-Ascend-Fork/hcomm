/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_c_adpt.h"
#include "flush_manager.h"
#include "dpu_comm_dfx.h"
#include "coll_comm_mgr.h"
#include "op_base.h"
#include "dfx/dfx_profiling_handler_pub.h"
#include "dfx/prof_utils_pub.h"
#include "exception_handler.h"
#include "adapter_prof.h"
#include "common/dev_type_utils_pub.h"

using namespace hccl;

/**
 * @note 职责：集合通信的通信域管理的C接口的C到C++适配
 */
HcclResult HcclCommGetStatus(const char* commId, HcclCommStatus* status)
{
    CHK_PTR_NULL(commId);
    CHK_PTR_NULL(status);
    HcclComm comm = nullptr;
    // 公共流程, 需要同时获取单算子和图模式的通信域
    CHK_RET(HcomGetCommHandleByGroup(commId, &comm));
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("%s hcclComm is nullptr, commId[%s]", __func__, commId), HCCL_E_PTR);
    return static_cast<hccl::hcclComm*>(comm)->GetCommStatus(*status);
}

int32_t HcommAcquireComm(const char* commId)
{
    CHK_PTR_NULL(commId);
    std::shared_ptr<hccl::hcclComm> hcclComm;
    HcclGetCommHandle(commId, hcclComm);
    CHK_PRT_RET(hcclComm == nullptr, HCCL_ERROR("%s hcclComm is null, commId[%s]", __func__, commId), HCCL_E_PTR);
    CHK_RET(hcclComm->SetCommDispatcherCtx());
    return HCCL_SUCCESS;
}

int32_t HcommReleaseComm(const char* commId)
{
    CHK_PTR_NULL(commId);
    HCCL_INFO("%s not support, commId[%s], do nothing", __func__, commId);
    return HCCL_SUCCESS;
}

int32_t HcommFenceOnThread(ThreadHandle thread)
{
    HCCL_INFO("[%s] START. thread[0x%llx].", __func__, thread);
    (void)thread;
    // flush 与 FlushManager 同处 hcomm.so，直接调用，不再经 hccl_v2 的 HcommFlushV2 跨 so 往返
    HcclResult ret = Hccl::FlushManager::GetInstance().Flush();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] FAIL. thread[0x%llx].", __func__, thread), ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
int32_t HcommFlush() { return HcommFenceOnThread(0); }
#ifdef __cplusplus
}
#endif // __cplusplus

// HcclDfxOpInfo（C 边界结构）转 DfxOpInfo
static std::shared_ptr<Hccl::DfxOpInfo> ConvertToDfxOpInfo(const HcclDfxOpInfo& dfxOpInfo)
{
    auto dfxOpInfoOnce = std::make_shared<Hccl::DfxOpInfo>();
    dfxOpInfoOnce->op_.opMode = static_cast<Hccl::OpMode::Value>(dfxOpInfo.opMode);
    dfxOpInfoOnce->op_.oldOpType = dfxOpInfo.opType;     // 存A3的类型
    dfxOpInfoOnce->op_.oldReduceOp = dfxOpInfo.reduceOp; // 存A3的类型
    dfxOpInfoOnce->op_.oldDataType = dfxOpInfo.dataType; // 存A3的类型

    dfxOpInfoOnce->op_.dataCount = dfxOpInfo.dataCount;
    dfxOpInfoOnce->op_.root = dfxOpInfo.root;

    dfxOpInfoOnce->op_.inputAddr = dfxOpInfo.inputMemAddr;
    dfxOpInfoOnce->op_.inputSize = dfxOpInfo.inputMemSize;
    dfxOpInfoOnce->op_.outputAddr = dfxOpInfo.outputMemAddr;
    dfxOpInfoOnce->op_.outputSize = dfxOpInfo.outputMemSize;

    dfxOpInfoOnce->algTag_ = dfxOpInfo.algTag;
    dfxOpInfoOnce->beginTime_ = dfxOpInfo.beginTime;
    dfxOpInfoOnce->cpuWaitAicpuNotifyId_ = dfxOpInfo.cpuWaitAicpuNotifyId;

    return dfxOpInfoOnce;
}

HcclResult HcclDfxRegOpInfoByCommId(char* commId, void* hcclDfxOpInfo)
{
    EXCEPTION_HANDLE_BEGIN
    CHK_PTR_NULL(commId);
    HcclComm commHandle = nullptr;
    CHK_RET(HcomGetCommHandleByGroup(commId, &commHandle));
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(commHandle);
    CHK_PRT_RET(hcclComm == nullptr, HCCL_ERROR("%s hcclComm is null, commId[%s]", __func__, commId), HCCL_E_PTR);
    CHK_PRT_RET(hcclDfxOpInfo == nullptr, HCCL_ERROR("[%s] hcclDfxOpInfo is null", __func__), HCCL_E_PTR);
    HcclDfxOpInfo* dfxOpInfo = static_cast<HcclDfxOpInfo*>(hcclDfxOpInfo);
    CHK_PTR_NULL(dfxOpInfo);
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (!hcclComm->IsCommunicatorV2() && devType == DevType::DEV_TYPE_910B) {
        return HCCL_SUCCESS;
    }
    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_ERROR("[%s]comm is NOT_SUPPORT", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);

    dfxOpInfo->beginTime = hrtMsprofSysCycleTime();

    // HcclDfxOpInfo转为DfxOpInfo
    auto dfxOpInfoOnce = ConvertToDfxOpInfo(*dfxOpInfo);
    CHK_SMART_PTR_NULL(dfxOpInfoOnce);
    dfxOpInfoOnce->comm_ = static_cast<void*>(collComm);
    dfxOpInfoOnce->isIndop_ = true;
    dfxOpInfoOnce->groupName_ = collComm->GetCommId();
    dfxOpInfoOnce->opIndex_ = collComm->UpdateIndex();
    dfxOpInfoOnce->rankSize_ = collComm->GetRankSize();
    // 单算子模式，暂时覆盖opTag
    dfxOpInfoOnce->op_.opTag = collComm->GetCommId();
    dfxOpInfoOnce->op_.myRank = static_cast<Hccl::RankId>(collComm->GetMyRankId());
    dfxOpInfoOnce->engine = dfxOpInfo->engine;
    HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
    CHK_PTR_NULL(hcclCommDfx);
    CHK_RET(hcclCommDfx->UpdateProfStat());
    CHK_RET(hcclCommDfx->SetCurrDfxOpInfo(dfxOpInfoOnce));
    bool isOpBase = dfxOpInfoOnce->op_.opMode == Hccl::OpMode::OPBASE
                    || dfxOpInfoOnce->op_.opMode == Hccl::OpMode::ACLGRAPH
                    || dfxOpInfoOnce->op_.opMode == Hccl::OpMode::NEGOTIATIONOP;
    if (hcclCommDfx->GetDpuCommDfx() != nullptr && hcclCommDfx->GetDpuCommDfx()->IsDpuOpInfoEnabled()) {
        hcclCommDfx->GetDpuCommDfx()->SetDpuTaskOpInfo(dfxOpInfoOnce);
    }
    bool isCached
        = dfxOpInfoOnce->op_.opMode == Hccl::OpMode::OFFLOAD || dfxOpInfoOnce->op_.opMode == Hccl::OpMode::ACLGRAPH;
    Hccl::DfxProfilingHandler::GetInstance().SetOpModeFlags(isOpBase, isCached);
    HCCL_INFO(
        "[%s] Register DfxOpInfo success, opMode[%d], isOpBase[%d], isCached[%d], DfxOpInfo: %s", __func__,
        dfxOpInfoOnce->op_.opMode, isOpBase, isCached, dfxOpInfoOnce->Describe().c_str());
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult HcclProfilingReportOp(HcclComm comm, uint64_t beginTime)
{
    HCCL_INFO("[%s] START, comm[%p].", __func__, comm);
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_PTR_NULL(hcclComm);
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_910B && !hcclComm->IsCommunicatorV2()) {
        return HCCL_SUCCESS;
    }
    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_ERROR("[%s] comm is NOT_SUPPORT", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
    CHK_PTR_NULL(hcclCommDfx);
    HCCL_INFO(
        "[%s] Report All Tasks Info, comm[%p], hcclCommDfx[%p] GetMirrorTaskManager[%p].", __func__, comm, hcclCommDfx,
        hcclCommDfx->GetMirrorTaskManager());
    auto* mirrorTaskMgr = hcclCommDfx->GetMirrorTaskManager();
    CHK_PTR_NULL(mirrorTaskMgr);
    if (mirrorTaskMgr->GetCurrDfxOpInfo() == nullptr) {
        HCCL_INFO("[%s] commId[%s] currDfxOpInfo is null, skip report.", __func__, collComm->GetCommId().c_str());
        return HCCL_SUCCESS;
    }
    // 单算子模式暂时默认true
    bool isOpBaseMode = false;
    bool isCached = false;
    CHK_RET(hcclCommDfx->GetOpModeFlags(isOpBaseMode, isCached));
    CHK_RET(hcclCommDfx->ReportAllTasks(isCached));
    CHK_RET(hcclCommDfx->ReportOp(beginTime, isCached, isOpBaseMode));
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcclReportAicpuKernel(HcclComm comm, uint64_t beginTime, char* kernelName)
{
    HCCL_INFO("[%s] START, comm[%p].", __func__, comm);
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(kernelName == nullptr, HCCL_ERROR("[%s] kernelName is null", __func__), HCCL_E_PTR);
    // 填入remoteRankId
    auto hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_PTR_NULL(hcclComm);
    if (!hcclComm->IsCommunicatorV2()) {
        return HCCL_SUCCESS;
    }
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
    CHK_PTR_NULL(hcclCommDfx);

    auto* mirrorTaskMgr = hcclCommDfx->GetMirrorTaskManager();
    CHK_PTR_NULL(mirrorTaskMgr);
    if (mirrorTaskMgr->GetCurrDfxOpInfo() == nullptr) {
        HCCL_INFO("[%s] commId[%s] currDfxOpInfo is null, skip report.", __func__, collComm->GetCommId().c_str());
        return HCCL_SUCCESS;
    }

    std::string kernelNameStr(kernelName);
    uint32_t threadId = SalGetTid();
    bool isOpBaseMode = false;
    bool isCached = false;
    CHK_RET(hcclCommDfx->GetOpModeFlags(isOpBaseMode, isCached));
    CHK_RET(hcclCommDfx->ReportKernel(beginTime, collComm->GetCommId(), kernelNameStr, threadId, isCached));

    Hccl::TaskParam taskParam{};
    taskParam.beginTime = beginTime;
    taskParam.taskType = Hccl::TaskParamType::TASK_AICPU_KERNEL;
    taskParam.endTime = hcomm::GetProfCycleTime();
    uint32_t taskId = INVALID_UINT;
    uint32_t streamId = INVALID_UINT;
    CHK_RET(hrtGetTaskIdAndStreamID(taskId, streamId));
    HCCL_INFO("[%s] taskId[%u], streamId[%u].", __func__, taskId, streamId);
    hcclCommDfx->GetDpuCommDfx()->SetAicpuTaskIdAndStreamId(taskId, streamId);
    CHK_RET(hcclCommDfx->AddTaskInfoCallback(streamId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO("[HcclReportAicpuKernel] HcclReportAicpuKernel success");
    return HCCL_SUCCESS;
}

extern HcclResult HcclReportAivKernel(HcclComm comm, uint64_t beginTime)
{
    HCCL_INFO("[%s] START, comm[%p].", __func__, comm);
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    auto hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_PTR_NULL(hcclComm);
    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_ERROR("[%s] comm is not supported", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
    CHK_PTR_NULL(hcclCommDfx);

    Hccl::TaskParam taskParam{};
    taskParam.beginTime = beginTime;
    taskParam.taskType = Hccl::TaskParamType::TASK_AIV;
    taskParam.endTime = hcomm::GetProfCycleTime();
    taskParam.isMaster = true;
    uint32_t taskId = INVALID_UINT;
    uint32_t streamId = INVALID_UINT;
    CHK_RET(hrtGetTaskIdAndStreamID(taskId, streamId));
    CHK_RET(hcclCommDfx->AddTaskInfoCallback(streamId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO("[HcclReportAivKernel] HcclReportAivKernel success");
    return HCCL_SUCCESS;
}

uint64_t HcommGetProfilingSysCycleTime()
{
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    if (!isGe950.has_value()) {
        HCCL_WARNING("[%s] get device type failed, return 0.", __func__);
        return 0;
    }
    if (!*isGe950) {
        return hrtMsprofSysCycleTime();
    }
    return hcomm::GetProfCycleTime();
}
