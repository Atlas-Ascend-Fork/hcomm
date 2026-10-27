/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_LAUNCH_H
#define CCU_LAUNCH_H

#include "hccl_types.h"
#include "ccu_types.h"
#include "hccl_res.h"

#ifndef HCOMM_WEAK_SYMBOL
#define HCOMM_WEAK_SYMBOL __attribute__((weak))
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

extern CcuResult HcommCcuKernelRegisterStart(CcuInsHandle insHandle) HCOMM_WEAK_SYMBOL;

extern CcuResult HcommCcuKernelRegister(
    CcuInsHandle insHandle, uint32_t dieId, const char* kernelFuncName, const void* kernelFunc, const void** kernelArgs,
    uint32_t argNum, CcuKernelHandle* kernelHandle) HCOMM_WEAK_SYMBOL;

extern CcuResult HcommCcuKernelRegisterEnd(CcuInsHandle insHandle) HCOMM_WEAK_SYMBOL;

extern CcuResult HcommCcuGetTaskArgsNum(CcuKernelHandle kernelHandle, uint32_t* taskArgsNum);

// 按名字查询命名 func 的入口 pc。契约：
// 1) pc 为绝对指令 id，须在 HcommCcuKernelRegisterEnd（翻译完成）之后调用，否则返回 CCU_E_INTERNAL；
// 2) funcName 只在 kernelHandle 所属 kernel 内唯一，无全局命名空间。不同 kernel 里的同名 func
//    彼此无关，拼装 taskArgs 时传错 kernelHandle 会查到另一个 func 的 pc 且不报错，由调用方保证配对；
// 3) 后端优化回填入口 pc 失败时返回 CCU_E_INTERNAL，不会发布过期 pc。
// 并发约束：跨 kernel 调用的传参/返回地址寄存器按 die 静态分配，同一 die 上同时只能有一条跨 kernel
// 调用链在执行；若两条调用链在同一 die 上并发，传参寄存器会被互相覆写。该约束当前不做运行期检测，
// 需由调用方在调度层保证（例如同 die 上的调用方 kernel 串行 Launch）。
extern CcuResult HcommCcuFunctionGet(CcuKernelHandle kernelHandle, const char* funcName, uint64_t* pc);

extern CcuResult HcommCcuKernelLaunch(
    ThreadHandle threadHandle, CcuKernelHandle kernelHandle, const void* taskArgs, uint32_t argNum) HCOMM_WEAK_SYMBOL;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // CCU_LAUNCH_H
