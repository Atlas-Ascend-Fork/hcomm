/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream_task_monitor.h"
#include "dfx/device/dfx_res_pub.h"
#include "stream_lite.h"
#include "coll_comm_aicpu_mgr.h"
#include "comm_engine/thread/device/device_thread_pub.h"
#include "log.h"
#include <shared_mutex>

namespace hcomm {

constexpr u32 TASK_ID_SHIFT_BITS = 16;
constexpr u64 MS_TO_US = 1000; // ms 转 us

StreamTaskMonitor& StreamTaskMonitor::GetInstance()
{
    static StreamTaskMonitor instance;
    return instance;
}

void StreamTaskMonitor::SetInterval(u32 interval)
{
    taskMonitorInterval_ = interval;
    stopCall_ = false;
    HCCL_INFO("[StreamTaskMonitor]SetInterval[%u] ms, stopCall_ reset", interval);
}

void StreamTaskMonitor::SetTaskExceptionEnable(bool enable)
{
    taskExceptionEnable_ = enable;
    HCCL_INFO("[StreamTaskMonitor]SetTaskExceptionEnable[%d]", enable);
}

void StreamTaskMonitor::Init(u32 devId)
{
    devId_ = devId;
    HCCL_INFO("[StreamTaskMonitor]Init success, devId[%u]", devId);
}

void StreamTaskMonitor::Call()
{
    if (stopCall_ || IsNoNeedMonitor() || devId_ == INVALID_UINT) {
        return;
    }
    HcclResult ret = MonitorAllComms();
    if (ret != HCCL_SUCCESS) {
        stopCall_ = true;
        HCCL_ERROR(
            "[StreamTaskMonitor]MonitorAllComms fail, monitor permanently stopped. "
            "ret[%d]. To recover, restart the process.",
            ret);
    }
}

void StreamTaskMonitor::OnCommDestroy(CollCommAicpu* aicpuComm)
{
    if (aicpuComm == nullptr) {
        HCCL_ERROR("[StreamTaskMonitor]OnCommDestroy fail, aicpuComm is nullptr");
        return;
    }
    auto* engineResMgr = aicpuComm->GetCommEngineResMgr();
    if (engineResMgr == nullptr) {
        HCCL_ERROR(
            "[StreamTaskMonitor]OnCommDestroy fail, engineResMgr is nullptr, comm[%s]",
            aicpuComm->GetIdentifier().c_str());
        return;
    }
    // 清理销毁通信域的监控数据
    std::shared_lock<std::shared_mutex> threadRwlock(engineResMgr->GetThreadMutex());
    const std::vector<ThreadHandle>& threads = engineResMgr->GetAllThread();
    std::string erasedStreamIds;
    for (auto& thread : threads) {
        void* streamLitePtr = nullptr;
        if (HcommThreadGetStreamLite(thread, &streamLitePtr) != HCOMM_SUCCESS) {
            HCCL_ERROR(
                "[StreamTaskMonitor]OnCommDestroy skip, thread[0x%llx] GetStreamLite fail, comm[%s]",
                static_cast<unsigned long long>(thread), aicpuComm->GetIdentifier().c_str());
            continue;
        }
        Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(streamLitePtr);
        if (streamLite == nullptr) {
            HCCL_ERROR(
                "[StreamTaskMonitor]OnCommDestroy skip, streamLite is nullptr, comm[%s]",
                aicpuComm->GetIdentifier().c_str());
            continue;
        }
        u32 sqId = streamLite->GetSqId();
        streamTaskMonitor_.erase(sqId);
        if (!erasedStreamIds.empty()) {
            erasedStreamIds += ",";
        }
        erasedStreamIds += std::to_string(streamLite->GetId());
    }
    HCCL_RUN_INFO(
        "[StreamTaskMonitor]OnCommDestroy success, comm[%s], streamId[%s]", aicpuComm->GetIdentifier().c_str(),
        erasedStreamIds.c_str());
}

bool StreamTaskMonitor::IsNoNeedMonitor() const { return taskMonitorInterval_ == 0; }

HcclResult StreamTaskMonitor::MonitorAllComms()
{
    std::shared_lock<std::shared_mutex> rwlock(CollCommAicpuMgr::GetInstance().GetMutex());

    std::vector<std::pair<std::string, CollCommAicpu*>> aicpuCommInfo;
    CHK_RET(CollCommAicpuMgr::GetInstance().GetAllComms(aicpuCommInfo));

    for (auto& commInfo : aicpuCommInfo) {
        CollCommAicpu* aicpuComm = commInfo.second;
        if (aicpuComm == nullptr) {
            HCCL_ERROR("[StreamTaskMonitor]MonitorAllComms skip, comm[%s] is nullptr", commInfo.first.c_str());
            continue;
        }
        HcclResult mRet = MonitorComm(aicpuComm);
        if (mRet != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[StreamTaskMonitor]MonitorComm fail, ret[%d], comm[%s]", mRet, aicpuComm->GetIdentifier().c_str());
            return mRet;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult StreamTaskMonitor::MonitorComm(CollCommAicpu* aicpuComm)
{
    // 单通信域内部操作：状态校验与判空均在此处统一处理
    HcclCommStatus commStatus = aicpuComm->GetCommmStatus();
    if ((commStatus == HcclCommStatus::HCCL_COMM_STATUS_INVALID)
        || (commStatus == HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING)) {
        return HCCL_SUCCESS;
    }
    auto* engineResMgr = aicpuComm->GetCommEngineResMgr();
    CHK_PTR_NULL(engineResMgr);
    std::shared_lock<std::shared_mutex> threadRwlock(engineResMgr->GetThreadMutex());
    const std::vector<ThreadHandle>& threads = engineResMgr->GetAllThread();
    for (auto& thread : threads) {
        void* streamLitePtr = nullptr;
        if (HcommThreadGetStreamLite(thread, &streamLitePtr) != HCOMM_SUCCESS) {
            continue;
        }
        Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(streamLitePtr);
        HcclResult mRet = MonitorStream(aicpuComm, streamLite);
        if (mRet != HCCL_SUCCESS) {
            return mRet;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult StreamTaskMonitor::MonitorStream(CollCommAicpu* aicpuComm, Hccl::StreamLite* streamLite)
{
    // streamLite 判空移入此处统一处理，避免上层重复校验
    if (streamLite == nullptr) {
        HCCL_ERROR(
            "[StreamTaskMonitor]MonitorStream skip, streamLite is nullptr, comm[%s]",
            aicpuComm->GetIdentifier().c_str());
        return HCCL_SUCCESS;
    }
    u32 sqId = streamLite->GetSqId();

    // 1. 查询 SQ 状态
    u32 sqHead = 0;
    u32 sqTail = 0;
    HcclResult qRet = QuerySqStatus(devId_, sqId, sqHead, sqTail);
    if (qRet != HCCL_SUCCESS) {
        HCCL_ERROR("[StreamTaskMonitor]QuerySqStatus fail, ret[%d], sqId[%u]", qRet, sqId);
        return qRet;
    }

    auto monitorIt = streamTaskMonitor_.find(sqId);

    auto curTime = std::chrono::steady_clock::now();

    // 2. 队列为空，清理记录并跳过
    if (sqHead == sqTail) {
        if (monitorIt != streamTaskMonitor_.end()) {
            streamTaskMonitor_.erase(monitorIt);
        }
        return HCCL_SUCCESS;
    }

    // 3. 读取 SQE header 字段（streamId/taskId/sqeType/notifyId 一次读取）
    auto* rtsq = streamLite->GetRtsq();
    if (rtsq == nullptr) {
        HCCL_ERROR("[StreamTaskMonitor]GetRtsq return nullptr, ret[%d], sqId[%u]", HCCL_E_PTR, sqId);
        return HCCL_E_PTR;
    }
    uint16_t streamId = 0;
    uint16_t taskId = 0;
    u8 sqeType = 0;
    u32 notifyId = 0;
    HcclResult ret = rtsq->GetSqeHeaderFieldsBySqIdx(sqHead, streamId, taskId, sqeType, notifyId);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[StreamTaskMonitor]GetSqeHeaderFieldsBySqIdx fail, ret[%d], sqId[%u] sqHead[%u]", ret, sqId, sqHead);
        return ret;
    }
    const u32 sqeId = GetSqeId(taskId, streamId);

    // 4. 首次记录基线时间
    if (monitorIt == streamTaskMonitor_.end()) {
        streamTaskMonitor_.insert(std::make_pair(sqId, MonitorTaskInfo{curTime, sqHead, sqeId, sqeType}));
        return HCCL_SUCCESS;
    }

    auto& monitorInfo = monitorIt->second;

    // 5. 状态变化时刷新基线
    if (IsNeedRefreshMonitorData(monitorInfo, sqeId, sqHead, sqeType)) {
        monitorInfo.time = curTime;
        monitorInfo.sqHead = sqHead;
        monitorInfo.sqeId = sqeId;
        monitorInfo.sqeType = sqeType;
        return HCCL_SUCCESS;
    }

    // 6. 耗时判定
    auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(curTime - monitorInfo.time).count();
    if (static_cast<u64>(elapsedUs) < static_cast<u64>(taskMonitorInterval_) * MS_TO_US) {
        return HCCL_SUCCESS;
    }

    // 7. 打印监控日志
    if (taskExceptionEnable_) {
        Hccl::DfxTaskInfo* taskInfo
            = HcclCommTaskExceptionLite::GetInstance().FindDfxTaskInfoNoLock(aicpuComm, sqId, sqeId);
        if (taskInfo == nullptr) {
            // notify 类型 SQE（尤其纯 notify_wait）在 DfxTaskInfo 队列中可能无记录，属正常情况，
            // 不应返回错误码导致 Call() 置 stopCall_ 永久停止监控，故此处 return SUCCESS 跳过本轮。
            // 告警说明队头 task 确实卡住（耗时已超阈值），但本监控无法区分卡在本卡还是跨卡对端，
            // 需结合 task_exception/cqe 日志进一步定位（sqeId/notifyId 可辅助关联）。
            HCCL_RUN_INFO(
                "[StreamTaskMonitor]DfxTaskInfo not found, task may be stuck. sqId[%u] sqeId[%u] sqeType[%u] "
                "notifyId[%u] elapsed[%llu us] threshold[%u ms]. Cannot determine local/cross-card cause.",
                sqId, sqeId, sqeType, notifyId, static_cast<u64>(elapsedUs), taskMonitorInterval_);
            monitorInfo.time = curTime;
            return HCCL_SUCCESS;
        }
        u32 remoteRank = HcclCommTaskExceptionLite::GetInstance().GetRemoteRankId(*taskInfo);
        // 卡内 notify wait（remoteRank 为 INVALID）为正常等待，跳过打印仅重置计时
        if (IsIntraCardNotifyWait(taskInfo->taskType, remoteRank)) {
            monitorInfo.time = curTime;
            return HCCL_SUCCESS;
        }
        std::string taskName = HcclCommTaskExceptionLite::GetInstance().GetConciseTaskName(*taskInfo);
        PrintMonitorLog(sqId, streamLite, sqHead, sqTail, static_cast<u64>(elapsedUs), taskName);
    } else {
        // task_exception 未开启，无 remoteRank 无法区分卡内/跨卡 notify_wait，按 SQE type 整体跳过 notify_wait
        constexpr u8 SQE_NOTIFY_WAIT = static_cast<u8>(Hccl::Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
        if (sqeType == SQE_NOTIFY_WAIT) {
            monitorInfo.time = curTime;
            return HCCL_SUCCESS;
        }
        PrintBasicMonitorLog(sqId, streamLite, sqHead, sqTail, static_cast<u64>(elapsedUs), sqeType, sqeId, notifyId);
    }

    monitorInfo.time = curTime;

    return HCCL_SUCCESS;
}

bool StreamTaskMonitor::IsNeedRefreshMonitorData(const MonitorTaskInfo& info, u32 sqeId, u32 sqHead, u8 sqeType) const
{
    // 队列为空 (sqHead == sqTail) 已在 MonitorStream 调用前提前返回，此处无需重复判定
    // 仅检查 sqeId/sqHead/sqeType 是否变化即可
    return (info.sqeId != sqeId) || (info.sqHead != sqHead) || (info.sqeType != sqeType);
}

bool StreamTaskMonitor::IsIntraCardNotifyWait(u8 sqeType, u32 remoteRank) const
{
    constexpr u8 NOTIFY_WAIT_TYPE = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT);
    return (sqeType == NOTIFY_WAIT_TYPE) && (remoteRank == INVALID_UINT);
}

u32 StreamTaskMonitor::GetSqeId(uint16_t taskId, uint16_t streamId) const
{
    return (static_cast<u32>(taskId) << TASK_ID_SHIFT_BITS) | static_cast<u32>(streamId);
}

void StreamTaskMonitor::PrintMonitorLog(
    u32 sqId, Hccl::StreamLite* streamLite, u32 sqHead, u32 sqTail, u64 elapsedUs, const std::string& taskName) const
{
    HCCL_RUN_INFO(
        "[StreamTaskMonitor]prof monitor streamId:%u, sqid:%u, head:%u, tail:%u, elapsed[%llu us] threshold[%u ms], %s",
        streamLite->GetId(), sqId, sqHead, sqTail, elapsedUs, taskMonitorInterval_, taskName.c_str());
}

void StreamTaskMonitor::PrintBasicMonitorLog(
    u32 sqId, Hccl::StreamLite* streamLite, u32 sqHead, u32 sqTail, u64 elapsedUs, u8 sqeType, u32 sqeId,
    u32 notifyId) const
{
    HCCL_RUN_INFO(
        "[StreamTaskMonitor]prof monitor streamId:%u, sqid:%u, head:%u, tail:%u, elapsed[%llu us] threshold[%u ms], "
        "sqeType:%u, sqeId:%u, notifyId:%u, task_exception:off",
        streamLite->GetId(), sqId, sqHead, sqTail, elapsedUs, taskMonitorInterval_, sqeType, sqeId, notifyId);
}

} // namespace hcomm
