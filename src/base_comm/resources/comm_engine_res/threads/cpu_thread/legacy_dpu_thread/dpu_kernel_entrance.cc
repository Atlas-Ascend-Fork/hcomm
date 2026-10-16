/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dpu_kernel_entrance.h"
#include <atomic>
#include "log.h"
#include "acl/acl_rt.h"

using namespace Hccl;

namespace {
// DPU 执行超时时间(秒)：由宿主库在 launch 前经 SetDpuExecTimeout 下发，未下发时取默认值。
// 与 EnvRtsConfig 的 HCCL_EXEC_TIMEOUT 默认值(NOTIFY_DEFAULT_WAIT_TIME)保持一致。
std::atomic<uint32_t> g_dpuExecTimeoutSec{Hccl::DPU_EXEC_TIMEOUT_DEFAULT_S};
} // namespace

extern "C" {
__attribute__((visibility("default"))) void SetDpuExecTimeout(uint32_t timeoutSec)
{
    g_dpuExecTimeoutSec.store(timeoutSec, std::memory_order_relaxed);
}

__attribute__((visibility("default"))) uint32_t RunDpuRpcSrvLaunch(const uint64_t args)
{
    HCCL_INFO("[%s] Launch Dpu Kernel: 0x%lx", __func__, args);
    if (args == 0) {
        HCCL_ERROR("[%s] args is null.", __func__);
        return HCCL_E_PARA;
    }

    if (reinterpret_cast<uint64_t>(args) + sizeof(DpuKernelLaunchParam) < reinterpret_cast<uint64_t>(args)) {
        HCCL_ERROR("[%s] Invalid args address.", __func__);
        return HCCL_E_PARA;
    }
    // 解析参数信息：收编后参数块只带宿主侧已构造好的 TaskService 裸指针 + deviceId
    DpuKernelLaunchParam* params = reinterpret_cast<DpuKernelLaunchParam*>(args);

    HCCL_RUN_INFO(
        "[%s] DpuKernelLaunchParam{taskService:%p; devId:%u}", __func__, params->taskService, params->deviceId);

    if (params->taskService == nullptr) {
        HCCL_ERROR("[%s] taskService is nullptr.", __func__);
        return HCCL_E_PARA;
    }

    aclError ret = aclrtSetDevice(params->deviceId);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[%s] set device fail. DeviceId: %u.", __func__, params->deviceId);
        return HCCL_E_RUNTIME;
    }

    // Run：实例归宿主侧持有，kernel 侧只取指针下发任务轮询
    HCCL_INFO("[%s] start to TaskRun", __func__);
    return params->taskService->TaskRun();
}
}

namespace Hccl {
uint32_t GetDpuExecTimeout() { return g_dpuExecTimeoutSec.load(std::memory_order_relaxed); }
} // namespace Hccl
