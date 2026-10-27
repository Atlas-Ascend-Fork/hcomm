/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Thread interfaces shared by coll_comm / base_comm (C++ linkage, internal use only, NOT exported).
// Implementations live in primitives/api_c_adpt/hcomm_thread_c_adpt.cc. The C-ABI thread interfaces
// (HcommThreadAllocWithConfig / HcommThreadAllocWithCommConfig / HcommThreadAcquireByNotify) are in
// thread_c_adpt_pub.h; the device-side (AICPU) thread interfaces are declared in device/device_thread_pub.h.
#ifndef HCOMM_THREAD_PUB_H
#define HCOMM_THREAD_PUB_H

#include <stddef.h>
#include "hcomm_res_defs.h"
#include "common/result_defs_pub.h"

/**
 * @brief Batch supplement thread notify resources; for AICPU engine it also triggers device kernel launch.
 */
HcommResult
HcommThreadSupplementNotify(const ThreadHandle* handles, uint32_t threadNum, const uint32_t* supplementNotifyNums);

/**
 * @brief Query the current notify count of a single thread.
 */
HcommResult HcommThreadGetNotifyNum(ThreadHandle thread, uint32_t* notifyNum);

/**
 * @brief Reset all local notify trigger states of a thread (idempotent).
 */
HcommResult HcommThreadResetNotifies(ThreadHandle thread);

/**
 * @brief Export thread handles to the target engine (CPU/CCU via device->host map; AICPU via kernel launch).
 */
HcommResult HcommThreadExportToCommEngine(
    const ThreadHandle* handles, const char* commId, uint32_t threadNum, CommEngine dstEngine,
    ThreadHandle* outHandles);

#endif // HCOMM_THREAD_PUB_H
