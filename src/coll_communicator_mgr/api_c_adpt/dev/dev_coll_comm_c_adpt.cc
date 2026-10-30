/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_comm.h"
#include "coll_comm_aicpu_mgr.h"
#include "common/device/aicpu_indop_env_pub.h"
#include "comm_engine/thread/device/aicpu_ts_primitives_c_adpt_pub.h"
#include "comm_engine/thread/device/device_thread_pub.h"
#include "aicpu_hccl_process.h"
#include "sqe_build_a5.h"
#include "common/cast_utils_pub.h"
#include "common/dev_type_utils_pub.h"
#include "aicpu_task_cache_manager.h"

using namespace hccl;

HcclResult HcclCommGetStatus(const char* commId, HcclCommStatus* status)
{
    CHK_PTR_NULL(commId);
    CHK_PTR_NULL(status);
    *status = HcclCommStatus::HCCL_COMM_STATUS_READY;
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    // deviceType 保留用于 910B 分支与日志；谓词取数失败须显式返回，不得 value_or(false) 误报 READY
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (*isGe950) {
        CollCommAicpu* collCommAicpu = CollCommAicpuMgr::GetInstance().GetCurrentComm(commId);
        CHK_PRT_RET(!collCommAicpu, HCCL_ERROR("%s AicpuGetComm is null, commId[%s]", __func__, commId), HCCL_E_PTR);
        *status = collCommAicpu->GetCommmStatus();
    } else if (deviceType == DevType::DEV_TYPE_910B) {
        HCCL_INFO("[%s] deviceType[%d] comm status ready", __func__, deviceType);
        *status = HcclCommStatus::HCCL_COMM_STATUS_READY;
    } else {
        HCCL_ERROR("[%s] deviceType[%d] is not support", __func__, deviceType);
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

inline bool GetProfilingEnable()
{
    return Hccl::DfxProfilingHandlerLite::GetInstance().GetProfL0State()
           || Hccl::DfxProfilingHandlerLite::GetInstance().GetProfL1State();
}

namespace {
// 刷新SQE profiling置位开关：L1开启且设备为960(A6)时推送1，否则推送0。
// 须先于HcclDfxRegOpInfoByCommId全部提前return执行，否则开关从开到关后无法刷回0，
// SQE将永久错误置位，故收敛在唯一入口先行调用。
void RefreshSqeProfilingState()
{
    bool isSqeProfEnabled = false;
    if (Hccl::DfxProfilingHandlerLite::GetInstance().GetProfL1State()) {
        isSqeProfEnabled = hcomm::EqualTo960().value_or(false);
    }
    Hccl::SetSqeProfilingEnabled(isSqeProfEnabled);
}
} // namespace

HcclResult HcclDfxRegOpInfoByCommId(char* commId, void* hcclDfxOpInfo)
{
    RefreshSqeProfilingState();

    if (!GetProfilingEnable() && !hcomm::GetTaskExceptionEnable()) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(commId);
    CHK_PTR_NULL(hcclDfxOpInfo);

    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType == DevType::DEV_TYPE_910B) {
        HCCL_INFO("[%s] is not supported, commId[%s], devType[%d]", __func__, commId, deviceType);
        return HCCL_SUCCESS;
    }

    HcclDfxOpInfo* aicpuDfxInfo = ReinterpretAs<HcclDfxOpInfo*>(hcclDfxOpInfo);
    CHK_RET(static_cast<HcclResult>(HcommThreadGetNotifyId(
        aicpuDfxInfo->cpuTsThread, aicpuDfxInfo->cpuWaitAicpuNotifyIdx, &aicpuDfxInfo->cpuWaitAicpuNotifyId)));
    CollCommAicpu* currentComm = CollCommAicpuMgr::GetInstance().GetCurrentComm();
    CHK_PTR_NULL(currentComm);
    CHK_RET(currentComm->InitDfxOpInfo(aicpuDfxInfo));

    return HCCL_SUCCESS;
}

namespace {
HcclResult GetChannelRemoteRankIdForTaskCache(ChannelHandle channelHandle, u32& remoteRankId)
{
    // 依赖算子展开线程的thread_local currentComm_，须在同一线程的HcommAcquireComm与HcommReleaseComm之间调用。
    HCCL_DEBUG(
        "[%s] query current comm for channelHandle[0x%llx]", __func__, static_cast<unsigned long long>(channelHandle));
    CollCommAicpu* currentComm = CollCommAicpuMgr::GetInstance().GetCurrentComm();
    CHK_PTR_NULL(currentComm);
    // RunAicpuCommRecordChannels 已按解包后的 device transport handle 登记对端 rank。
    auto* dfxLite = currentComm->GetHcclCommDfxLite();
    CHK_PTR_NULL(dfxLite);
    const u32 rankId = dfxLite->GetChannelRemoteRankId(channelHandle);
    CHK_PRT_RET(
        rankId == INVALID_UINT,
        HCCL_ERROR(
            "[%s] remote rank not found for channelHandle[0x%llx]", __func__,
            static_cast<unsigned long long>(channelHandle)),
        HCCL_E_NOT_FOUND);
    remoteRankId = rankId;
    return HCCL_SUCCESS;
}

} // namespace

int32_t HcommAcquireComm(const char* commId)
{
    CHK_PTR_NULL(commId);
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    HCCL_INFO("[%s]comId[%s], devType[%d]", __func__, commId, deviceType);
    if (!hcomm::EqualToOrGreaterThan950().value_or(false)) {
        HcclCommAicpu* hcclComm = AicpuHcclProcess::AicpuGetCommbyGroup(commId);
        CHK_PRT_RET(!hcclComm, HCCL_ERROR("%s AicpuGetCommbyGroup is null, commId[%s]", __func__, commId), HCCL_E_PTR);
        CHK_RET(hcclComm->SetDispatcherCtxOnThread());
    } else {
        CollCommAicpu* hcclComm = CollCommAicpuMgr::GetInstance().AcquireCommForUse(commId);
        CHK_PRT_RET(!hcclComm, HCCL_ERROR("%s AcquireCommForUse is null, commId[%s]", __func__, commId), HCCL_E_PTR);
        hcomm::AicpuTaskCacheManager::RegisterGetChannelRemoteRankIdCallback(GetChannelRemoteRankIdForTaskCache);
    }
    return HCCL_SUCCESS;
}

int32_t HcommReleaseComm(const char* commId)
{
    CHK_PTR_NULL(commId);
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    HCCL_INFO("[%s]comId[%s], devType[%d]", __func__, commId, deviceType);
    if (!hcomm::EqualToOrGreaterThan950().value_or(false)) {
        AicpuHcclProcess::AicpuReleaseCommbyGroup(commId);
    } else {
        CollCommAicpuMgr::GetInstance().ReleaseComm(commId);
    }
    return HCCL_SUCCESS;
}

HcclResult HcommProfilingReportDeviceOp(const char* groupname)
{
    if (!GetProfilingEnable()) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(groupname);

    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (!(*isGe950)) {
        return HCCL_SUCCESS;
    }

    CollCommAicpu* currentComm = CollCommAicpuMgr::GetInstance().GetCurrentComm();
    CHK_PTR_NULL(currentComm);
    CHK_RET(currentComm->ProfilingReportDeviceOp());
    return HCCL_SUCCESS;
}

HcclResult HcommProfilingReportKernelStartTask(uint64_t thread, const char* groupname)
{
    if (!GetProfilingEnable()) {
        return HCCL_SUCCESS;
    }

    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (!(*isGe950)) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(groupname);
    CollCommAicpu* currentComm = CollCommAicpuMgr::GetInstance().GetCurrentComm();
    CHK_PTR_NULL(currentComm);
    CHK_RET(currentComm->UpdateTask());
    void* streamLitePtr = nullptr;
    CHK_RET(static_cast<HcclResult>(HcommThreadGetStreamLite(thread, &streamLitePtr)));
    CHK_PTR_NULL(streamLitePtr);
    auto* const streamLite = static_cast<Hccl::StreamLite*>(streamLitePtr);
    Hccl::DfxFlagTaskInfo flagTaskInfo;
    auto* rtsq = streamLite->GetRtsq();
    CHK_PRT_RET(rtsq == nullptr, HCCL_ERROR("[%s] rtsq is null", __func__), HCCL_E_PTR);
    flagTaskInfo.taskId = rtsq->GetTaskId();
    flagTaskInfo.type = Hccl::DfxMainStreamTaskType::HEAD;
    Hccl::DfxProfilingHandlerLite::GetInstance().ReportMainStreamTask(flagTaskInfo);
    HCCL_INFO("[%s] END, thread [%llu], groupname[%s], taskId[%u].", __func__, thread, groupname, flagTaskInfo.taskId);
    return HCCL_SUCCESS;
}

HcclResult HcommProfilingReportKernelEndTask(uint64_t thread, const char* groupname)
{
    if (!GetProfilingEnable()) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(groupname);
    HCCL_INFO("[%s] START. thread [%llu], groupname[%s].", __func__, thread, groupname);

    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (!(*isGe950)) {
        return HCCL_SUCCESS;
    }

    void* streamLitePtr = nullptr;
    CHK_PRT_RET(
        HcommThreadGetStreamLite(thread, &streamLitePtr) != HCOMM_SUCCESS || streamLitePtr == nullptr,
        HCCL_ERROR("[%s] streamLitePtr is null", __func__), HCCL_E_PTR);
    auto* const streamLite = static_cast<Hccl::StreamLite*>(streamLitePtr);
    // FlagTaskInfo Report
    Hccl::DfxFlagTaskInfo flagTaskInfo;
    flagTaskInfo.type = Hccl::DfxMainStreamTaskType::TAIL;
    auto* rtsq = streamLite->GetRtsq();
    CHK_PRT_RET(rtsq == nullptr, HCCL_ERROR("[%s] rtsq is null", __func__), HCCL_E_PTR);
    uint16_t streamId = 0;
    uint16_t taskId = 0;
    HcclResult ret = rtsq->GetLastStreamIdAndTaskId(streamId, taskId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s] GetLastStreamIdAndTaskId fail, ret[%d], sqId[%u].", __func__, ret, streamLite->GetSqId()),
        ret);
    constexpr uint32_t UINT16_BIT_WIDTH = 16U;
    flagTaskInfo.taskId = (static_cast<uint32_t>(taskId) << UINT16_BIT_WIDTH) | static_cast<uint32_t>(streamId);

    Hccl::DfxProfilingHandlerLite::GetInstance().ReportMainStreamTask(flagTaskInfo);
    return HCCL_SUCCESS;
}
