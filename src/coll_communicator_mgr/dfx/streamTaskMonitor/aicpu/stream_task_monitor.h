/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_STREAM_TASK_MONITOR_H
#define HCCL_STREAM_TASK_MONITOR_H

#include "daemon_func.h"
#include "dfx/res_pub.h"
#include "coll_comm_aicpu.h"
#include "aicpu_hccl_sqcq.h"
#include "hcclCommTaskExceptionLite.h"

#include <chrono>
#include <unordered_map>
#include <string>

namespace hcomm {

struct MonitorTaskInfo {
    std::chrono::steady_clock::time_point time;
    u32 sqHead{0};
    u32 sqeId{0};
    u8 sqeType{0};
};

class StreamTaskMonitor : public Hccl::DaemonFunc {
public:
    static StreamTaskMonitor& GetInstance();
    void SetInterval(u32 interval);
    void SetTaskExceptionEnable(bool enable);
    void Init(u32 devId);
    void Call() override;
    void OnCommDestroy(CollCommAicpu* aicpuComm);

private:
    StreamTaskMonitor() = default;
    ~StreamTaskMonitor() override = default;

    bool IsNoNeedMonitor() const;
    HcclResult MonitorAllComms();
    HcclResult MonitorComm(CollCommAicpu* aicpuComm);
    HcclResult MonitorStream(CollCommAicpu* aicpuComm, Hccl::StreamLite* streamLite);

    bool IsNeedRefreshMonitorData(const MonitorTaskInfo& info, u32 sqeId, u32 sqHead, u8 sqeType) const;
    bool IsIntraCardNotifyWait(u8 sqeType, u32 remoteRank) const;

    // 基础日志（无 taskName/remoteRank）
    void PrintBasicMonitorLog(
        u32 sqId, Hccl::StreamLite* streamLite, u32 sqHead, u32 sqTail, u64 elapsedUs, u8 sqeType, u32 sqeId,
        u32 notifyId) const;
    // 增强日志（含 taskName/remoteRank，taskName 内已含 notifyId）
    void PrintMonitorLog(
        u32 sqId, Hccl::StreamLite* streamLite, u32 sqHead, u32 sqTail, u64 elapsedUs,
        const std::string& taskName) const;

    u32 GetSqeId(uint16_t taskId, uint16_t streamId) const;

private:
    bool stopCall_{false};
    uint32_t taskMonitorInterval_{0};
    u32 devId_{INVALID_UINT};
    bool taskExceptionEnable_{false};
    std::unordered_map<u32, MonitorTaskInfo> streamTaskMonitor_; // key: sqId
};

} // namespace hcomm

#endif // HCCL_STREAM_TASK_MONITOR_H
