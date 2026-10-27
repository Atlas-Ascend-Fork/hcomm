/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_THREAD_C_ADPT_PUB_H
#define HCOMM_THREAD_C_ADPT_PUB_H

#include <stdint.h>
#include "hcomm_res_defs.h"

// C 兼容：与 hccl/rt runtime 的 typedef void* rtStream_t 同义（去掉 C++ 头 hccl/base.h 依赖）。
#ifndef HCOMM_RT_STREAM_T_DEFINED
#define HCOMM_RT_STREAM_T_DEFINED
typedef void* rtStream_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief Allocate thread(s) with the target engine using explicit per-thread config.
 */
HcommResult HcommThreadAllocWithConfig(
    CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads);

/**
 * @brief Allocate thread(s) bound to a communication domain using explicit per-thread config.
 */
HcommResult HcommThreadAllocWithCommConfig(
    CommEngine engine, const char* commId, uint32_t threadNum, ThreadType type, const ThreadConfig* config,
    ThreadHandle* threads);

/**
 * @brief Acquire a thread associated with an existing stream and a set of notifies (borrowed, host-side).
 */
HcommResult HcommThreadAcquireByNotify(rtStream_t stream, void** notifys, uint32_t notifyNum, ThreadHandle* thread);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCOMM_THREAD_C_ADPT_PUB_H
