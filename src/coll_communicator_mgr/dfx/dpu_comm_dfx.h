/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_COMM_DFX_H
#define DPU_COMM_DFX_H

#include <memory>
#include <unordered_map>
#include <mutex>
#include <array>
#include <atomic>
#include <functional>
#include "hccl_common.h"
#include "common.h"
#include "hcclCommOp.h"

namespace hccl {

class DpuCommDfx {
public:
    using EnqueueTaskFunc = std::function<HcclResult(
        u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle,
        std::shared_ptr<Hccl::DfxOpInfo> opInfo)>;
    using GetCurrDfxOpInfoFunc = std::function<std::shared_ptr<Hccl::DfxOpInfo>()>;

    DpuCommDfx();
    ~DpuCommDfx();

    void
    Init(u32 deviceId, const std::string& commTag, EnqueueTaskFunc enqueueFunc, GetCurrDfxOpInfoFunc getCurrOpInfoFunc);

    HcclResult AddDpuTaskInfoCallback(const Hccl::TaskParam& taskParam, u64 handle);
    std::function<HcclResult(const Hccl::TaskParam&, u64)> GetDpuCallback() const { return dpuCallback_; }
    void SetDpuStreamId(u32 dpuStreamId);
    void SetAicpuTaskIdAndStreamId(u32 taskId, u32 streamId)
    {
        aicpuTaskId_ = taskId;
        aicpuStreamId_ = streamId;
    }
    void InitPendingWriteInfo(u64 handle);
    void SetDpuTaskOpInfo(std::shared_ptr<Hccl::DfxOpInfo> dfxOpInfo);
    bool IsDpuOpInfoEnabled() const { return dpuOpInfoEnabled_.load(std::memory_order_relaxed); }
    static u32 GetTaskId(u32 streamId);

    struct DpuTaskInfoEntry {
        u32 taskId{0};
        u32 streamId{0};
    };
    const DpuTaskInfoEntry& GetDpuTaskInfo() const { return dpuTaskInfo_; }

private:
    u32 GetDpuChannelStreamId(u64 handle);
    bool HandleWNotifyCache(const Hccl::TaskParam& taskParam, u64 handle, std::shared_ptr<Hccl::DfxOpInfo> dpuOpInfo);
    void CacheWNotify(
        const Hccl::TaskParam& taskParam, u64 handle, std::shared_ptr<Hccl::TaskParam>& failedTaskParam,
        bool& needEnqueueFailed, bool& cached, bool& needFlushCached, std::shared_ptr<Hccl::TaskParam>& cachedTaskParam,
        u64& cachedTotalSize, u32& cachedCount, u64& cachedTotalSizeLog, u64& failTime);
    HcclResult EnqueueWNotifyFailed(
        u64 handle, std::shared_ptr<Hccl::DfxOpInfo> dpuOpInfo, bool needFlushCached, bool needEnqueueFailed,
        u64 failTime, std::shared_ptr<Hccl::TaskParam> cachedTaskParam, u64 cachedTotalSize,
        std::shared_ptr<Hccl::TaskParam> failedTaskParam);
    HcclResult
    HandleFenceOrDrain(const Hccl::TaskParam& taskParam, u64 handle, std::shared_ptr<Hccl::DfxOpInfo> dpuOpInfo);
    std::shared_ptr<Hccl::DfxOpInfo> GetDpuTaskOpInfo();

    EnqueueTaskFunc enqueueTaskFunc_{};
    GetCurrDfxOpInfoFunc getCurrDfxOpInfoFunc_{};
    u32 deviceId_{0};
    std::string commTag_{};
    u32 aicpuTaskId_{INVALID_UINT};
    u32 aicpuStreamId_{INVALID_UINT};

    struct PendingWriteInfo {
        std::shared_ptr<Hccl::TaskParam> taskParam;
        u64 totalSize{0};
        u32 count{0};
        bool valid{false};
    };
    std::unordered_map<u64, PendingWriteInfo> pendingWriteInfos_;
    std::unordered_map<u64, u32> channelStreamIdMap_;
    static constexpr u32 CHANNEL_STREAM_ID_BASE = 0x80000000;
    static constexpr u32 CHANNEL_STREAM_ID_RANGE = 0x1000;
    u32 channelStreamIdBase_{CHANNEL_STREAM_ID_BASE};
    u32 nextChannelStreamId_{channelStreamIdBase_};
    DpuTaskInfoEntry dpuTaskInfo_{};
    u32 dpuStreamId_{0};
    std::function<HcclResult(const Hccl::TaskParam&, u64)> dpuCallback_{};
    static constexpr u32 DPU_OP_INFO_RING_CAPACITY = 2048;
    std::array<std::shared_ptr<Hccl::DfxOpInfo>, DPU_OP_INFO_RING_CAPACITY> opInfoRing_{};
    std::mutex opInfoRingMutex_;
    std::atomic<bool> dpuOpInfoEnabled_{false};
    static std::mutex taskIdMutex_;
    static std::unordered_map<u32, u32> streamIdToTaskId_;
    static std::atomic<u32> g_nextChannelStreamIdBase;
};

} // namespace hccl

#endif
