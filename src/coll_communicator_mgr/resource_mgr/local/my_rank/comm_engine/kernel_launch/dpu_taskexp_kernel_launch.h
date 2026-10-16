/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_TASKEXP_KERNEL_LAUNCH_H
#define DPU_TASKEXP_KERNEL_LAUNCH_H

#include <cstdint>
#include <string>
#include "hccl/hccl_res.h"

namespace hccl {

/**
 * @brief 下发 `HcclDpuTaskexpShmemRestore` kernel：把本通信域 taskexception 共享内存地址登记到
 *        AICPU 侧 commId→VA 登记表（sink 在 dfx/taskException），并顺带清零该块内存。
 *        仅在 950（有 V2 通信域、AICPU 侧登记表存在）下调用；拉起失败须由调用方回滚线程。
 * @param[in] binHandle 本通信域的 kernel 二进制句柄（ccl_kernel.json），随通信域初始化下传
 * @param[in] commId 通信域 id，device 侧以它登记
 * @param[in] devId 设备逻辑 id
 * @param[in] taskexpVa taskexception 共享内存地址（DPUTASKEXCEPTION）
 * @param[in] memorySize taskexpVa 的有效字节数
 */
HcclResult LaunchDpuTaskexpShmemRestoreKernel(
    aclrtBinHandle binHandle, const std::string& commId, uint32_t devId, void* taskexpVa, uint64_t memorySize);

} // namespace hccl

#endif // DPU_TASKEXP_KERNEL_LAUNCH_H
