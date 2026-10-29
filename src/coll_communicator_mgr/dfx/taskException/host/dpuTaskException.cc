/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dpuTaskException.h"
#include <memory>
#include "log.h"
#include "coll_comm.h"
#include "coll_comm_mgr.h"
#include "hccl_comm_pub.h"
#include "orion_adapter_hccp.h"
#include "hcomm_adapter_hccp.h"
#include <adapter_error_manager_pub.h>
#include "hccl_types.h"
#include "hcclCommTaskException.h"
#include "dpu_kernel_entrance.h"
#include "hccl_log_keywords.h"
#include "task_info.h"
#include "reduce_op.h"
#include "data_type.h"
#include "global_mirror_tasks.h"

namespace hcomm {

using namespace std;

constexpr uint32_t TASK_CONTEXT_SIZE = 50;
constexpr uint32_t TASK_CONTEXT_INFO_SIZE = LOG_TMPBUF_SIZE - 50;

std::string AicpuGetAndPrintClusterMonitorErr(const rtExceptionInfo* exceptionInfo);

HcclResult DpuTaskException::ProcessDpuException(const rtExceptionInfo_t* exceptionInfo, const TaskExceptionHost& host)
{
    std::lock_guard<std::mutex> lock(GetSerMapMutex());
    uint16_t dpuErrCode = 0;
    std::string commId;
    uint8_t* taskExpPtr = nullptr;
    CHK_RET(GetDpuExceptionInfo(exceptionInfo, dpuErrCode, commId, taskExpPtr));

    HCCL_ERROR("[DpuTaskException][ProcessDpuException] Task from HCCL run failed.");

    CHK_RET(ReportDpuException(exceptionInfo, dpuErrCode, commId, taskExpPtr, host));

    return ClearDpuExceptionInfo(exceptionInfo, commId);
}

HcclResult DpuTaskException::GetDpuExceptionInfo(
    const rtExceptionInfo_t* exceptionInfo, uint16_t& dpuErrCode, std::string& commId, uint8_t*& taskExpPtr)
{
    void* expPtr = nullptr;
    if (!FindDpuExceptionByDevice(exceptionInfo->deviceid, commId, expPtr, dpuErrCode)) {
        return HCCL_E_NOT_FOUND;
    }
    taskExpPtr = reinterpret_cast<uint8_t*>(expPtr);
    if (dpuErrCode == 0 || taskExpPtr == nullptr) {
        HCCL_ERROR(
            "[%s] dpu exception info invalid, dpuErrCode[%u], taskExpPtr[%p].", __func__, dpuErrCode, taskExpPtr);
        return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

HcclResult DpuTaskException::ReportDpuException(
    const rtExceptionInfo_t* exceptionInfo, uint16_t dpuErrCode, const std::string& commId, uint8_t* taskExpPtr,
    const TaskExceptionHost& host)
{
    u32 infoStreamId = 0;
    u32 infoTaskId = 0;
    {
        // 排布：|stopflag[1]|hcclret[2]|dstret[2]|opIndex[4]|taskId[4]|streamId[4]|reserved[3]|
        uint8_t* infoPtr = taskExpPtr + sizeof(uint8_t) + sizeof(uint16_t) + sizeof(uint16_t) + sizeof(u32);
        errno_t ret = memcpy_s(&infoTaskId, sizeof(u32), infoPtr, sizeof(u32));
        CHK_PRT_RET(
            ret != EOK, HCCL_ERROR("[%s] memcpy_s get dpu taskId failed, ret[%d].", __func__, ret), HCCL_E_INTERNAL);
        ret = memcpy_s(&infoStreamId, sizeof(u32), infoPtr + sizeof(u32), sizeof(u32));
        CHK_PRT_RET(
            ret != EOK, HCCL_ERROR("[%s] memcpy_s get dpu streamId failed, ret[%d].", __func__, ret), HCCL_E_INTERNAL);
    }
    Hccl::TaskInfo* lastDpuTask = nullptr;
    if (infoTaskId != 0 || infoStreamId != 0) {
        HcclResult findRet = Hccl::GlobalMirrorTasks::Instance().FindTaskInfo(
            exceptionInfo->deviceid, infoStreamId, infoTaskId, lastDpuTask);
        if (findRet != HCCL_SUCCESS || lastDpuTask == nullptr) {
            HCCL_ERROR(
                "[DpuTaskException][%s] FindTaskInfo fail, deviceId[%u], streamId[%u], taskId[%u], ret[%d], "
                "dpuErrCode[%u], commId[%s].",
                __func__, exceptionInfo->deviceid, StoredToReportStreamId(commId, infoStreamId), infoTaskId, findRet,
                dpuErrCode, commId.c_str());
            return HCCL_SUCCESS;
        }
    }

    if (lastDpuTask != nullptr) {
        HandleDpuErrorReport(exceptionInfo, *lastDpuTask, dpuErrCode, commId, host);
    } else {
        HCCL_ERROR(
            "[DpuTaskException][%s] errorCode[%d], devId[%u], commId[%s] (task info not found in queue)", __func__,
            dpuErrCode, exceptionInfo->deviceid, commId.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult DpuTaskException::ClearDpuExceptionInfo(const rtExceptionInfo_t* exceptionInfo, const std::string& commId)
{
    void* taskExpPtr = FindTaskExpMem(commId, exceptionInfo->deviceid);
    if (taskExpPtr == nullptr) {
        HCCL_ERROR(
            "[%s] commId[%s] deviceId[%u] not found during memset, may have been destroyed.", __func__, commId.c_str(),
            exceptionInfo->deviceid);
        return HCCL_E_NOT_FOUND;
    }
    constexpr size_t CLEAR_OFFSET = sizeof(uint8_t) + sizeof(uint16_t);
    constexpr size_t CLEAR_SIZE
        = sizeof(uint16_t) + sizeof(u32) + sizeof(u32) + sizeof(u32); // dstRet + opIndex + taskId + streamId
    errno_t ret = memset_s(static_cast<uint8_t*>(taskExpPtr) + CLEAR_OFFSET, CLEAR_SIZE, 0, CLEAR_SIZE);
    CHK_PRT_RET(
        ret != EOK, HCCL_ERROR("[%s] memset_s clean dpu taskexception failed, ret[%d].", __func__, ret),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

void DpuTaskException::HandleDpuErrorReport(
    const rtExceptionInfo_t* exceptionInfo, const Hccl::TaskInfo& taskInfo, uint16_t hcclRet, const std::string& commId,
    const TaskExceptionHost& host)
{
    std::string stageErrInfo = "[TaskExec][RunFailed][DPU]";
    HCCL_ERROR(
        "%sTask from HCCL run failed, errorCode[%u], commId[%s].", stageErrInfo.c_str(), hcclRet, commId.c_str());
    HCCL_ERROR(
        "%sTask run failed, base information is deviceID:[%u], streamID(sqId):[%u], taskID(sqeId):[%u], taskType:[%s].",
        stageErrInfo.c_str(), exceptionInfo->deviceid, taskInfo.taskParam_.reportStreamId, taskInfo.taskId_,
        taskInfo.taskParam_.taskType.Describe().c_str());
    HCCL_ERROR("%sTask run failed, para information is %s.", stageErrInfo.c_str(), taskInfo.GetParaInfo().c_str());
    HCCL_ERROR(
        "%sTask run failed, groupRank information is %s.", stageErrInfo.c_str(), GetGroupRankInfo(taskInfo).c_str());
    PrintDpuTaskContextInfo(exceptionInfo->deviceid, taskInfo, stageErrInfo);
    std::string clusterMonitorErrMsg = AicpuGetAndPrintClusterMonitorErr(exceptionInfo);
    if (host.ShouldReportError()) {
        std::string baseInfo = Hccl::StringFormat(
            "streamID(sqId):[%u], taskID(sqeId):[%u], taskType:[%s]", taskInfo.taskParam_.reportStreamId,
            taskInfo.taskId_, taskInfo.taskParam_.taskType.Describe().c_str());
        RPT_INPUT_ERR(
            true, "EI0002",
            std::vector<std::string>({"remote_rankid", "base_information", "task_information", "group_rank_content"}),
            std::vector<std::string>(
                {std::to_string(taskInfo.remoteRank_), baseInfo, (taskInfo.GetParaInfo() + clusterMonitorErrMsg), ""}));
    } else {
        HCCL_WARNING(
            "[DpuTaskException] EI0002 already reported on device[%u], skip duplicate.", exceptionInfo->deviceid);
    }
}

void DpuTaskException::PrintDpuTaskContextInfo(
    uint32_t deviceId, const Hccl::TaskInfo& taskInfo, const std::string& stageErrInfo)
{
    Hccl::TaskInfoQueue* queue = nullptr;
    try {
        queue = Hccl::GlobalMirrorTasks::Instance().GetQueue(deviceId, taskInfo.streamId_);
    } catch (Hccl::HcclException& e) {
        HCCL_ERROR(
            "Exception task queue not found. deviceId[%u], streamId[%u].", deviceId,
            taskInfo.taskParam_.reportStreamId);
        return;
    }

    if (queue == nullptr) {
        HCCL_ERROR(
            "Exception task queue not found. deviceId[%u], streamId[%u].", deviceId,
            taskInfo.taskParam_.reportStreamId);
        return;
    }

    const u32 taskId = taskInfo.taskId_;
    auto func = [taskId](const unique_ptr<Hccl::TaskInfo>& task) {
        return task->taskId_ == taskId;
    };
    auto taskIterPtr = queue->Find(func);
    if (taskIterPtr == nullptr || *taskIterPtr == *queue->End()) {
        HCCL_ERROR(
            "Exception task queue not found. deviceId[%u], streamId[%u], taskId[%u].", deviceId,
            taskInfo.taskParam_.reportStreamId, taskId);
        return;
    }

    vector<Hccl::TaskInfo*> taskContext{};
    for (uint32_t i = 0; i < TASK_CONTEXT_SIZE; ++i) {
        taskContext.emplace_back((**taskIterPtr).get());
        if (*taskIterPtr == *queue->Begin()) {
            break;
        }
        --(*taskIterPtr);
    }

    HCCL_ERROR(
        "%s Task run failed, context sequence before error task is "
        "[DpuNotifyWait:DPU_NW(rank,id), DpuInlineWrite:DPU_IW(rank,id), DpuWriteWithNotify:DPU_WWN(rank,id), "
        "DpuChannelFence:DPU_CF(rank), DpuThreadFence:DPU_TF(rank)]:",
        stageErrInfo.c_str());

    std::string taskContextInfo = "";
    Hccl::TaskInfo* lastTask = nullptr;
    for (size_t i = 0; i < taskContext.size(); ++i) {
        if (taskContext[i] == nullptr) {
            continue;
        }
        if (lastTask == nullptr) {
            lastTask = taskContext[i];
        }
        std::string conciseInfo = taskContext[i]->GetConciseBaseInfo();
        conciseInfo += ",";
        u32 lastOpIndex = GetOpIndex(lastTask);
        u32 curOpIndex = GetOpIndex(taskContext[i]);
        bool overSize = (taskContextInfo.size() + conciseInfo.size()) >= TASK_CONTEXT_INFO_SIZE;
        if (overSize || (lastOpIndex != curOpIndex)) {
            HCCL_ERROR(
                "%sTask run failed, opData information is %s.", stageErrInfo.c_str(),
                GetDpuIndopDataInfo(lastTask).c_str());
            HCCL_ERROR("%s task sequence is OP(%u): %s", stageErrInfo.c_str(), lastOpIndex, taskContextInfo.c_str());
            taskContextInfo = "";
            lastTask = taskContext[i];
        }
        taskContextInfo += conciseInfo;
    }
    if (!taskContextInfo.empty() && lastTask != nullptr) {
        u32 lastOpIndex = GetOpIndex(lastTask);
        HCCL_ERROR(
            "%sTask run failed, opData information is %s.", stageErrInfo.c_str(),
            GetDpuIndopDataInfo(lastTask).c_str());
        HCCL_ERROR("%s task sequence is OP(%u): %s", stageErrInfo.c_str(), lastOpIndex, taskContextInfo.c_str());
    }
    HCCL_ERROR("%s task sequence end.", stageErrInfo.c_str());
}

std::string DpuTaskException::GetGroupRankInfo(const Hccl::TaskInfo& taskInfo)
{
    if (taskInfo.dfxOpInfo_ == nullptr || taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_ERROR("[DpuTaskException][%s]TaskInfo communicator is nullptr.", __func__);
        return "";
    }

    hccl::CollComm* communicator = static_cast<hccl::CollComm*>(taskInfo.dfxOpInfo_->comm_);
    return Hccl::StringFormat(
        "group:[%s], rankSize[%u], rankId[%u]", communicator->GetCommId().c_str(), communicator->GetRankSize(),
        communicator->GetMyRankId());
}

u32 DpuTaskException::GetOpIndex(const Hccl::TaskInfo* taskInfo)
{
    if (taskInfo == nullptr || taskInfo->dfxOpInfo_ == nullptr) {
        return UINT32_MAX;
    }
    return taskInfo->dfxOpInfo_->opIndex_;
}

u32 DpuTaskException::StoredToReportStreamId(const std::string& commId, u32 stored)
{
    HcclComm commHandle = nullptr;
    if (HcomGetCommHandleByGroup(commId.c_str(), &commHandle) != HCCL_SUCCESS || commHandle == nullptr) {
        return stored;
    }
    auto* hcclComm = static_cast<hccl::hcclComm*>(commHandle);
    auto* collComm = hcclComm->GetCollComm();
    if (collComm == nullptr || collComm->GetHcclCommDfx() == nullptr
        || collComm->GetHcclCommDfx()->GetDpuCommDfx() == nullptr) {
        return stored;
    }
    return stored - collComm->GetHcclCommDfx()->GetDpuCommDfx()->GetChannelStreamIdBase();
}

std::string DpuTaskException::GetDpuIndopDataInfo(const Hccl::TaskInfo* taskInfo)
{
    if (taskInfo == nullptr || taskInfo->dfxOpInfo_ == nullptr) {
        HCCL_ERROR(
            "[TaskInfo][GetIndopDataInfo]dfxOpInfo is nullptr, no op info registered, return default value. "
            "streamId(sqId)[%u], taskId(sqeId)[%u].",
            taskInfo == nullptr ? 0u : taskInfo->taskParam_.reportStreamId,
            taskInfo == nullptr ? 0u : taskInfo->taskId_);
        return "";
    }
    const auto& opInfo = taskInfo->dfxOpInfo_;
    Hccl::ReduceOp reduceOp = opInfo->op_.reduceOp;
    auto reduceOpIt = Hccl::REDUCE_OP_MAP.find(static_cast<HcclReduceOp>(opInfo->op_.oldReduceOp));
    if (reduceOpIt != Hccl::REDUCE_OP_MAP.end()) {
        reduceOp = reduceOpIt->second;
    }
    Hccl::DataType dataType = opInfo->op_.dataType;
    auto datatypeIt = Hccl::DATA_TYPE_MAP.find(static_cast<HcclDataType>(opInfo->op_.oldDataType));
    if (datatypeIt != Hccl::DATA_TYPE_MAP.end()) {
        dataType = datatypeIt->second;
    }
    return Hccl::StringFormat(
        "opIndex[%u], algTag[%s], count[%llu], reduceType[%s], dataType[%s], "
        "input: ptr[0x%llx] size[%llu], output: ptr[0x%llx] size[%llu]",
        opInfo->opIndex_, opInfo->algTag_.c_str(), opInfo->op_.dataCount, reduceOp.Describe().c_str(),
        dataType.Describe().c_str(), opInfo->op_.inputAddr, opInfo->op_.inputSize, opInfo->op_.outputAddr,
        opInfo->op_.outputSize);
}
} // namespace hcomm
