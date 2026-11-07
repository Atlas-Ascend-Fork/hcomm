/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_THREAD_LAUNCH_MGR_H
#define AICPU_THREAD_LAUNCH_MGR_H

#include "comm_engine/aicpu_launch_manager_pub.h"
#include "thread.h"

namespace hccl {

class CpuThread;

struct ThreadKernelLaunchConfig {
    std::string commId;       // 通信ID
    aclrtBinHandle binHandle; // 自定义二进制句柄
    std::string kernelName;   // 核函数名称
    bool needDeviceInfo;      // 是否需要设备信息
    uint32_t timeoutSec;      // 超时时间（秒）
    bool needProfiling;       // 是否需要性能分析
    bool isSupplementNotify;  // 是否是补充notify kernel

    ThreadKernelLaunchConfig(
        const std::string& cid, aclrtBinHandle binHandle, const std::string& name, bool needDev, uint32_t timeout,
        bool profiling, bool isSupplementNotify)
        : commId(cid),
          binHandle(binHandle),
          kernelName(name),
          needDeviceInfo(needDev),
          timeoutSec(timeout),
          needProfiling(profiling),
          isSupplementNotify(isSupplementNotify)
    {}
};

class AicpuThreadLaunchMgr {
public:
    static HcclResult ThreadKernelLaunchForComm(
        std::vector<std::shared_ptr<Thread>>& newThreads, const std::string& commId,
        std::unique_ptr<ThreadHandle[]>& aicpuHandle, aclrtBinHandle binHandle);
    static HcclResult ThreadKernelLaunchForBase(
        std::vector<std::shared_ptr<Thread>>& newThreads, std::unique_ptr<ThreadHandle[]>& aicpuHandle,
        aclrtBinHandle binHandle);
    static HcclResult SupplementNotifyKernelLaunch(
        std::vector<std::shared_ptr<Thread>>& newThreads, const std::string& commId,
        std::unique_ptr<ThreadHandle[]>& aicpuHandle, aclrtBinHandle binHandle);
    // 队列线程拿到 device 句柄后投递 SQ/CQ 名片：名片不落地，device 侧守护就看不到这对队列
    static HcclResult CpuQueueNamecardLaunch(
        std::vector<std::shared_ptr<Thread>>& newThreads, const ThreadHandle* threadHandles, const std::string& commId,
        aclrtBinHandle binHandle);
    // 队列线程补配 notify 后，device 侧实体仍指着旧地址，须把新地址与数量同步过去
    static HcclResult
    CpuNotifyAddrUpdateLaunch(CpuThread& cpuThread, ThreadHandle threadHandle, const std::string& commId);
    static HcclResult
    ThreadKernelLaunchDestroy(ThreadHandle* threadHandles, uint32_t listNum, aclrtBinHandle binHandle);

private:
    static HcclResult ThreadKernelLaunchImpl(
        std::vector<std::shared_ptr<Thread>>& newThreads, std::unique_ptr<ThreadHandle[]>& aicpuHandle,
        const ThreadKernelLaunchConfig& config);
};

} // namespace hccl
#endif // AICPU_THREAD_LAUNCH_MGR_H
