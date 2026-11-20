/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ndaTaskException.h"
#include "hcclCommTaskException.h"
#include "log.h"
#include "coll_comm_mgr.h"
#include "adapter_error_manager_pub.h"

namespace hcomm {

using namespace std;

HcclResult NdaTaskException::ProcessNdaException(const rtExceptionInfo_t* exceptionInfo, TaskExceptionHost& host)
{
    if (exceptionInfo == nullptr) {
        HCCL_ERROR("[NdaTaskException][%s] exceptionInfo is nullptr.", __func__);
        return HCCL_E_PARA;
    }

    // AICPU 报文按 comm 只写一次且 host 侧不清零，兜底路径 ProcessException 已用 hasAicpuReport_
    // 做一次性消费；NDA 先于兜底路径执行，这里同样拒绝重复认领旧报文，否则后续同设备异常
    // 会把旧报文的 rankId/EID 当作当前异常重发 EI0018（RPT_INPUT_ERR 无去重）
    if (host.hasAicpuReport_) {
        HCCL_RUN_INFO(
            "[NdaTaskException][%s] aicpu report already consumed, fall back to legacy path, deviceid[%u].", __func__,
            exceptionInfo->deviceid);
        return HCCL_E_NOT_FOUND;
    }

    // 1. 遍历通信域快照，从 KFC D2H 读 aicpu 写的 ErrorMessageReport，按 linkType 分流
    Hccl::ErrorMessageReport errMsg;
    std::string commId;
    HcclResult readRet = ReadNdaErrorMessage(exceptionInfo, errMsg, commId);
    if (readRet != HCCL_SUCCESS) {
        // 非 RoCE 链路或无 aicpu 上报，交回兜底路径
        return HCCL_E_NOT_FOUND;
    }
    // 报文已被本路径认领：置位与兜底路径共用的一次性消费标记，
    // 后续同设备异常不再重复读取/上报这条不会更新的旧报文
    host.hasAicpuReport_ = true;

    HCCL_RUN_WARNING(
        "[NdaTaskException][%s] NDA exception captured, commId[%s], taskType[%s], linkType[%s].", __func__,
        commId.c_str(), errMsg.taskType.Describe().c_str(), errMsg.linkType.Describe().c_str());

    // 2. 用 streamId / taskId 去 GlobalMirrorTasks 找 TaskInfo（与 NDR 共用同一张表）
    // 注意：errMsg.streamId 是 AICPU 侧写入的 SQ id（GenerateErrorMessageReport 中 taskInfo.sqId），
    // 不是镜像任务表的 key；镜像表以 host 下发时的运行时 streamId 为 key，与兜底链路
    // ProcessHcclTaskException 一致，这里必须用 exceptionInfo->streamid 查表
    Hccl::TaskInfo* task = nullptr;
    HcclResult ret = Hccl::GlobalMirrorTasks::Instance().FindTaskInfo(
        exceptionInfo->deviceid, exceptionInfo->streamid, exceptionInfo->taskid, task);
    CHK_PRT_RET(
        ret == HCCL_E_NOT_FOUND,
        HCCL_RUN_WARNING(
            "[%s]FindTaskInfo not found, deviceid[%u] streamid[%u] taskid[%u].", __func__, exceptionInfo->deviceid,
            exceptionInfo->streamid, exceptionInfo->taskid),
        HCCL_E_NOT_FOUND);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS || task == nullptr,
        HCCL_ERROR(
            "[%s]FindTaskInfo fail, ret[%d], deviceid[%u] streamid[%u] taskid[%u].", __func__, ret,
            exceptionInfo->deviceid, exceptionInfo->streamid, exceptionInfo->taskid),
        HCCL_E_INTERNAL);

    // 3. 拼接阶段标识：[TaskExec][RunFailed][NDA]
    auto stageErrInfo = "[" + LOG_KEYWORDS_TASK_EXEC + "][" + LOG_KEYWORDS_RUN_FAILED + "][" + LOG_KEYWORDS_NDA + "]";

    // 4. 分级打印 + 任务上下文序列
    PrintNdaErrorLog(exceptionInfo, *task, stageErrInfo, host);

    // 5. RoCE 错误码上报（EI0002 超时 / EI0018 其余）
    ReportNdaError(*task, errMsg, exceptionInfo, stageErrInfo, host);

    return HCCL_SUCCESS;
}

HcclResult NdaTaskException::ReadNdaErrorMessage(
    const rtExceptionInfo_t* exceptionInfo, Hccl::ErrorMessageReport& errMsg, std::string& commId)
{
    // 快照只取 commId：遍历期间通信域可能被注销，故不在锁外解引用任何 CollComm。
    // 反查、设备校验与 KFC D2H 报文读取统一由 CollCommMgr::GetAicpuTaskException 在锁内完成，
    // 报文按值拷贝返回，读取期间 CollComm 由 CollCommMgr 的 mutex_ 保证存活。
    hccl::CollCommMgr& commMgr = hccl::CollCommMgr::GetInstance();
    std::vector<std::string> commIds = commMgr.SnapshotCollComms();

    for (const auto& curCommId : commIds) {
        Hccl::ErrorMessageReport msg;
        if (commMgr.GetAicpuTaskException(curCommId, static_cast<s32>(exceptionInfo->deviceid), msg) != HCCL_SUCCESS) {
            continue;
        }

        // 该通信域无 aicpu 上报
        if (strlen(msg.tag) == 0) {
            continue;
        }

        // 入口处按 linkType 一次性分流：仅接管 RoCE 链路，其余交回兜底
        if (msg.linkType != Hccl::DfxLinkType::ROCE && msg.linkType != Hccl::DfxLinkType::STANDARD_ROCE) {
            HCCL_INFO(
                "[NdaTaskException][%s] linkType[%s] not RoCE, fall back to legacy path.", __func__,
                msg.linkType.Describe().c_str());
            continue;
        }

        errMsg = msg;
        commId = curCommId;
        return HCCL_SUCCESS;
    }
    return HCCL_E_NOT_FOUND;
}

void NdaTaskException::PrintNdaTaskContextInfo(
    const rtExceptionInfo_t* exceptionInfo, const Hccl::TaskInfo& task, const std::string& stageErrInfo,
    TaskExceptionHost& host)
{
    // 复用 TaskExceptionHost 的私有能力打印任务前序上下文（前 50 个 task）
    host.PrintTaskContextInfo(exceptionInfo->deviceid, task.streamId_, task.taskId_, stageErrInfo);
}

void NdaTaskException::PrintNdaErrorLog(
    const rtExceptionInfo_t* exceptionInfo, const Hccl::TaskInfo& task, const std::string& stageErrInfo,
    TaskExceptionHost& host)
{
    HCCL_ERROR("%s Task from HCCL run failed.", stageErrInfo.c_str());
    HCCL_ERROR("%s Task run failed, base information is %s.", stageErrInfo.c_str(), task.GetIndopBaseInfo().c_str());
    HCCL_ERROR("%s Task run failed, para information is %s.", stageErrInfo.c_str(), task.GetParaInfo().c_str());
    HCCL_ERROR(
        "%s Task run failed, base information (legacy) is %s.", stageErrInfo.c_str(), task.GetBaseInfo().c_str());

    // 任务前序上下文序列
    PrintNdaTaskContextInfo(exceptionInfo, task, stageErrInfo, host);
}

void NdaTaskException::ReportNdaError(
    const Hccl::TaskInfo& task, const Hccl::ErrorMessageReport& errMsg, const rtExceptionInfo_t* exceptionInfo,
    const std::string& stageErrInfo, TaskExceptionHost& host)
{
    // 超时用 EI0002，其余 RoCE 异常用 EI0018 并改文案
    if (errMsg.taskType == Hccl::TaskParamType::TASK_NOTIFY_WAIT) {
        if (host.ShouldReportError()) {
            HCCL_ERROR("[NdaTaskException][%s] EI0002 report.", __func__);
            RPT_INPUT_ERR(
                true, "EI0002",
                std::vector<std::string>(
                    {"remote_rankid", "base_information", "task_information", "group_rank_content"}),
                std::vector<std::string>(
                    {std::to_string(task.remoteRank_), task.GetIndopBaseInfo().c_str(), task.GetParaInfo().c_str(),
                     ""}));
        } else {
            HCCL_WARNING(
                "[NdaTaskException] EI0002 already reported on device[%u], skip duplicate.", exceptionInfo->deviceid);
        }
        return;
    }

    // 其余 RoCE 异常：复用 TaskExceptionHost::ReportEI0018Error（内部已含 local/remote serverId、deviceId、IP 文案）
    // 该函数会解引用 dfxOpInfo_->comm_，而镜像任务表中允许存在未登记算子信息的任务（兜底链路同样有该保护），
    // 此处需前置判空，避免在 RTS 异常回调线程里解引用空指针
    CHK_PRT_RET(
        task.dfxOpInfo_ == nullptr,
        HCCL_ERROR(
            "[NdaTaskException][%s] dfxOpInfo is nullptr, skip EI0018 report, deviceid[%u].", __func__,
            exceptionInfo->deviceid), );
    host.ReportEI0018Error(task, errMsg, stageErrInfo);
}

} // namespace hcomm
