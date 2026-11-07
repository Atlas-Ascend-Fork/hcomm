/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 本文件只编进 ccl_dpu（kernel 侧）。入口名进 ccl_dpu.ini 生成 json、符号进 ccl_dpu.map 白名单，
// 三处缺一都只在 Release 运行期报「函数解析失败」，可用 nm/json 自检兜底
#include "cpu_queue_kernel_entrance.h"
#include "cpu_task_pipe.h"
#include "log.h"
#include "acl/acl_rt.h"

using namespace hccl;

extern "C" {
__attribute__((visibility("default"))) uint32_t RunDpuRpcSrvLaunchNew(const uint64_t args)
{
    HCCL_INFO("[%s] Launch Dpu Kernel: 0x%lx", __func__, args);
    if (args == 0) {
        HCCL_ERROR("[%s] args is null.", __func__);
        return HCCL_E_PARA;
    }
    // 溢出预检（与 RunDpuRpcSrvLaunch 一致）：args 由 host 传入，非法地址只能在此拦截
    if (reinterpret_cast<uint64_t>(args) + sizeof(DpuQueueLaunchParam) < reinterpret_cast<uint64_t>(args)) {
        HCCL_ERROR("[%s] Invalid args address.", __func__);
        return HCCL_E_PARA;
    }
    // 未对齐解引用属未定义行为
    if ((args & 0x7U) != 0U) {
        HCCL_ERROR("[%s] args is not 8-byte aligned: 0x%lx", __func__, args);
        return HCCL_E_PARA;
    }
    DpuQueueLaunchParam* params = reinterpret_cast<DpuQueueLaunchParam*>(args);
    HCCL_RUN_INFO("[%s] DpuQueueLaunchParam{taskPipe:%p; devId:%u}", __func__, params->taskPipe, params->deviceId);

    if (params->taskPipe == nullptr) {
        HCCL_ERROR("[%s] taskPipe is nullptr.", __func__);
        return HCCL_E_PARA;
    }

    aclError ret = aclrtSetDevice(params->deviceId);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[%s] set device fail. DeviceId: %u.", __func__, params->deviceId);
        return HCCL_E_RUNTIME;
    }

    // 实例归host侧持有，kernel 侧只取指针进入 SQ 轮询
    HCCL_INFO("[%s] start to poll SQ", __func__);
    return params->taskPipe->Run();
}
}
