/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dpu_comm_dfx.h"
#include "dfx/prof_utils_pub.h"
#include "coll_comm_mgr.h"
#include "dfx/channel_profiling_adpt_pub.h"

namespace hccl {

std::mutex DpuCommDfx::taskIdMutex_;
std::unordered_map<u32, u32> DpuCommDfx::streamIdToTaskId_;
std::atomic<u32> DpuCommDfx::g_nextChannelStreamIdBase{DpuCommDfx::CHANNEL_STREAM_ID_BASE};
DpuCommDfx::DpuCommDfx() {}

DpuCommDfx::~DpuCommDfx() { dpuCallback_ = nullptr; }

void DpuCommDfx::Init(
    u32 deviceId, const std::string& commTag, EnqueueTaskFunc enqueueFunc, GetCurrDfxOpInfoFunc getCurrOpInfoFunc)
{
    deviceId_ = deviceId;
    commTag_ = commTag;
    enqueueTaskFunc_ = std::move(enqueueFunc);
    getCurrDfxOpInfoFunc_ = std::move(getCurrOpInfoFunc);
    channelStreamIdBase_ = g_nextChannelStreamIdBase.fetch_add(CHANNEL_STREAM_ID_RANGE, std::memory_order_relaxed);
    nextChannelStreamId_ = channelStreamIdBase_;
    dpuCallback_ = [this](const Hccl::TaskParam& taskParam, u64 handle) {
        return this->AddDpuTaskInfoCallback(taskParam, handle);
    };
}

HcclResult DpuCommDfx::AddDpuTaskInfoCallback(const Hccl::TaskParam& taskParam, u64 handle)
{
    auto dpuOpInfo = GetDpuTaskOpInfo();
    if (taskParam.taskType == Hccl::TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY) {
        if (HandleWNotifyCache(taskParam, handle, dpuOpInfo)) {
            return HCCL_SUCCESS;
        }
    }

    if (taskParam.taskType == Hccl::TaskParamType::TASK_DPU_CHANNEL_FENCE
        || taskParam.taskType == Hccl::TaskParamType::TASK_DPU_CHANNEL_DRAIN) {
        return HandleFenceOrDrain(taskParam, handle, dpuOpInfo);
    }

    u32 streamId;
    if (taskParam.taskType == Hccl::TaskParamType::TASK_DPU_KERNEL) {
        streamId = dpuStreamId_;
    } else {
        streamId = GetDpuChannelStreamId(handle);
    }
    u32 taskId = GetTaskId(dpuStreamId_);
    Hccl::TaskParam localTaskParam = taskParam;
    localTaskParam.aicpuTaskId = aicpuTaskId_;
    localTaskParam.npuDevId = deviceId_;
    if (taskParam.taskType != Hccl::TaskParamType::TASK_DPU_KERNEL) {
        dpuTaskInfo_ = {taskId, streamId};
    }
    HCCL_INFO(
        "[%s] streamId[%u], taskId[%u], aicpuTaskId[%llu], npuDevId[%u].", __func__, streamId, taskId,
        localTaskParam.aicpuTaskId, static_cast<u32>(localTaskParam.npuDevId));
    return enqueueTaskFunc_(streamId, taskId, localTaskParam, handle, dpuOpInfo);
}

bool DpuCommDfx::HandleWNotifyCache(
    const Hccl::TaskParam& taskParam, u64 handle, std::shared_ptr<Hccl::DfxOpInfo> dpuOpInfo)
{
    std::shared_ptr<Hccl::TaskParam> failedTaskParam;
    bool needEnqueueFailed = false;
    bool cached = false;
    bool needFlushCached = false;
    std::shared_ptr<Hccl::TaskParam> cachedTaskParam;
    u64 cachedTotalSize = 0;
    u32 cachedCount = 0;
    u64 cachedTotalSizeLog = 0;
    u64 failTime = 0;
    CacheWNotify(
        taskParam, handle, failedTaskParam, needEnqueueFailed, cached, needFlushCached, cachedTaskParam,
        cachedTotalSize, cachedCount, cachedTotalSizeLog, failTime);
    if (cached) {
        HCCL_INFO(
            "[%s] cached W+Notify, handle[0x%llx], count[%u], totalSize[%llu].", __func__, handle, cachedCount,
            cachedTotalSizeLog);
        return true;
    }
    if (needEnqueueFailed) {
        HcclResult enqueueRet = EnqueueWNotifyFailed(
            handle, dpuOpInfo, needFlushCached, needEnqueueFailed, failTime, cachedTaskParam, cachedTotalSize,
            failedTaskParam);
        if (enqueueRet != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] EnqueueWNotifyFailed failed, ret[%d].", __func__, enqueueRet);
        }
        return true;
    }
    HCCL_INFO("[%s] handle[0x%llx] not found in pendingWriteInfos_, fall through to default.", __func__, handle);
    return false;
}

void DpuCommDfx::CacheWNotify(
    const Hccl::TaskParam& taskParam, u64 handle, std::shared_ptr<Hccl::TaskParam>& failedTaskParam,
    bool& needEnqueueFailed, bool& cached, bool& needFlushCached, std::shared_ptr<Hccl::TaskParam>& cachedTaskParam,
    u64& cachedTotalSize, u32& cachedCount, u64& cachedTotalSizeLog, u64& failTime)
{
    auto it = pendingWriteInfos_.find(handle);
    if (it == pendingWriteInfos_.end()) {
        return;
    }
    auto& info = it->second;
    if (taskParam.isFailed) {
        failTime = hcomm::GetProfCycleTime();
        failedTaskParam = std::make_shared<Hccl::TaskParam>(taskParam);
        failedTaskParam->endTime = failTime;
        failedTaskParam->aicpuTaskId = aicpuTaskId_;
        failedTaskParam->npuDevId = deviceId_;
        if (info.valid) {
            cachedTaskParam = info.taskParam;
            cachedTotalSize = info.totalSize;
            needFlushCached = true;
        }
        info.valid = false;
        info.count = 0;
        info.totalSize = 0;
        needEnqueueFailed = true;
    } else if (info.count == 0) {
        info.taskParam = std::make_shared<Hccl::TaskParam>(taskParam);
        info.totalSize = taskParam.taskPara.DMA.size;
        info.count = 1;
        info.valid = true;
        cached = true;
        cachedCount = info.count;
        cachedTotalSizeLog = info.totalSize;
    } else {
        if (taskParam.beginTime < info.taskParam->beginTime) {
            info.taskParam->beginTime = taskParam.beginTime;
        }
        info.totalSize += taskParam.taskPara.DMA.size;
        info.count++;
        cached = true;
        cachedCount = info.count;
        cachedTotalSizeLog = info.totalSize;
    }
}

HcclResult DpuCommDfx::EnqueueWNotifyFailed(
    u64 handle, std::shared_ptr<Hccl::DfxOpInfo> dpuOpInfo, bool needFlushCached, bool needEnqueueFailed, u64 failTime,
    std::shared_ptr<Hccl::TaskParam> cachedTaskParam, u64 cachedTotalSize,
    std::shared_ptr<Hccl::TaskParam> failedTaskParam)
{
    u32 failedStreamId = 0;
    if (needFlushCached || needEnqueueFailed) {
        failedStreamId = GetDpuChannelStreamId(handle);
    }
    if (needFlushCached) {
        cachedTaskParam->taskType = Hccl::TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY;
        cachedTaskParam->taskPara.DMA.size = cachedTotalSize;
        cachedTaskParam->endTime = failTime;
        cachedTaskParam->aicpuTaskId = aicpuTaskId_;
        cachedTaskParam->npuDevId = deviceId_;
        u32 taskId = GetTaskId(dpuStreamId_);
        dpuTaskInfo_ = {taskId, failedStreamId};
        HCCL_INFO(
            "[%s] flush cached W+Notify before failed enqueue, handle[0x%llx], taskId[%u].", __func__, handle, taskId);
        (void)enqueueTaskFunc_(failedStreamId, taskId, *cachedTaskParam, handle, dpuOpInfo);
    }
    if (needEnqueueFailed) {
        u32 taskId = GetTaskId(dpuStreamId_);
        dpuTaskInfo_ = {taskId, failedStreamId};
        HCCL_INFO("[%s] W+Notify failed, direct enqueue, handle[0x%llx], taskId[%u].", __func__, handle, taskId);
        return enqueueTaskFunc_(failedStreamId, taskId, *failedTaskParam, handle, dpuOpInfo);
    }
    return HCCL_SUCCESS;
}

HcclResult
DpuCommDfx::HandleFenceOrDrain(const Hccl::TaskParam& taskParam, u64 handle, std::shared_ptr<Hccl::DfxOpInfo> dpuOpInfo)
{
    std::shared_ptr<Hccl::TaskParam> writeTaskParam;
    u64 totalSize = 0;
    u32 count = 0;
    bool needEnqueue = false;
    u32 fenceStreamId = 0;
    auto it = pendingWriteInfos_.find(handle);
    if (it != pendingWriteInfos_.end() && it->second.valid) {
        auto& info = it->second;
        writeTaskParam = std::make_shared<Hccl::TaskParam>(*info.taskParam);
        totalSize = info.totalSize;
        count = info.count;
        info.valid = false;
        info.count = 0;
        info.totalSize = 0;
        needEnqueue = true;
        auto sidIt = channelStreamIdMap_.find(handle);
        fenceStreamId = (sidIt != channelStreamIdMap_.end()) ? sidIt->second - channelStreamIdBase_ : dpuStreamId_;
    }
    if (needEnqueue) {
        writeTaskParam->taskType = Hccl::TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY;
        writeTaskParam->taskPara.DMA.size = totalSize;
        writeTaskParam->endTime = taskParam.endTime;
        writeTaskParam->aicpuTaskId = aicpuTaskId_;
        writeTaskParam->npuDevId = deviceId_;

        u32 taskId = GetTaskId(dpuStreamId_);
        dpuTaskInfo_ = {taskId, fenceStreamId};
        HCCL_INFO(
            "[%s] fence->write enqueued, handle[0x%llx], taskId[%u], count[%u], beginTime[%llu], endTime[%llu], "
            "totalSize[%llu].",
            __func__, handle, taskId, count, writeTaskParam->beginTime, writeTaskParam->endTime, totalSize);
        return enqueueTaskFunc_(fenceStreamId, taskId, *writeTaskParam, handle, dpuOpInfo);
    }
    HCCL_INFO("[%s] no pending W+Notify for handle[0x%llx], skip fence.", __func__, handle);
    return HCCL_SUCCESS;
}

void DpuCommDfx::InitPendingWriteInfo(u64 handle)
{
    dpuOpInfoEnabled_.store(true, std::memory_order_relaxed);
    pendingWriteInfos_[handle] = PendingWriteInfo{};
    if (channelStreamIdMap_.find(handle) == channelStreamIdMap_.end()) {
        channelStreamIdMap_[handle] = nextChannelStreamId_++;
    }
}

u32 DpuCommDfx::GetDpuChannelStreamId(u64 handle)
{
    auto it = channelStreamIdMap_.find(handle);
    if (it != channelStreamIdMap_.end()) {
        u32 reportStreamId = it->second - channelStreamIdBase_;
        HCCL_INFO(
            "[%s] commTag[%s] channel stream id for handle[0x%llx]: stored[%u], report[%u]", __func__, commTag_.c_str(),
            handle, it->second, reportStreamId);
        return reportStreamId;
    }
    HCCL_ERROR(
        "[%s] commTag[%s] handle[0x%llx] not found, fallback to dpuStreamId_[%u].", __func__, commTag_.c_str(), handle,
        dpuStreamId_);
    return dpuStreamId_;
}

void DpuCommDfx::SetDpuTaskOpInfo(std::shared_ptr<Hccl::DfxOpInfo> dfxOpInfo)
{
    if (dfxOpInfo == nullptr) {
        HCCL_ERROR("[%s] dfxOpInfo is nullptr, skip set dpu task op info.", __func__);
        return;
    }
    std::lock_guard<std::mutex> lock(opInfoRingMutex_);
    opInfoRing_[dfxOpInfo->opIndex_ % DPU_OP_INFO_RING_CAPACITY] = dfxOpInfo;
}

std::shared_ptr<Hccl::DfxOpInfo> DpuCommDfx::GetDpuTaskOpInfo()
{
    // 收编后异常区随 DPU 线程走 ThreadMgr 链，不再跨 so 读 g_taskExpMemMap；经 commTag_(=commId) 反查。
    // expRef 须活到本函数用完地址：keepAlive 是异常区所在实体的保活引用
    DpuTaskExpMemRef expRef;
    (void)CollCommMgr::GetInstance().GetDpuTaskExpMem(commTag_, expRef);
    void* taskexpShmem = expRef.addr;
    if (taskexpShmem == nullptr) {
        HCCL_WARNING(
            "[%s] commTag[%s] deviceId[%u] not found in taskExpMem, fallback to GetCurrDfxOpInfo", __func__,
            commTag_.c_str(), deviceId_);
        return getCurrDfxOpInfoFunc_();
    }
    constexpr u32 OPINDEX_OFFSET = 5;
    u32 opIndex = 0;
    auto ret
        = memcpy_s(&opIndex, sizeof(opIndex), static_cast<uint8_t*>(taskexpShmem) + OPINDEX_OFFSET, sizeof(opIndex));
    if (ret != EOK) {
        HCCL_ERROR("[%s] memcpy_s failed, fallback to GetCurrDfxOpInfo, ret[%d]", __func__, ret);
        return getCurrDfxOpInfoFunc_();
    }
    std::lock_guard<std::mutex> lock(opInfoRingMutex_);
    auto& opInfo = opInfoRing_[opIndex % DPU_OP_INFO_RING_CAPACITY];
    if (opInfo == nullptr) {
        HCCL_WARNING(
            "[%s] opInfoRing_[%u] is nullptr, fallback to GetCurrDfxOpInfo", __func__,
            opIndex % DPU_OP_INFO_RING_CAPACITY);
        return getCurrDfxOpInfoFunc_();
    }
    return opInfo;
}

void DpuCommDfx::SetDpuStreamId(u32 dpuStreamId) { dpuStreamId_ = dpuStreamId; }

u32 DpuCommDfx::GetTaskId(u32 streamId)
{
    std::lock_guard<std::mutex> lock(taskIdMutex_);
    auto& taskIdRef = streamIdToTaskId_[streamId];
    constexpr u32 TASK_ID_MODULO = 65536;
    taskIdRef = (taskIdRef + 1) % TASK_ID_MODULO;
    u32 retTaskId = taskIdRef;
    return retTaskId;
}

} // namespace hccl
