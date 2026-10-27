/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_THREAD_DFX_PUB_H
#define HCOMM_THREAD_DFX_PUB_H

#include <functional>
#include "hcomm_res_defs.h"
#include "hccl_common.h"
#include "task_param.h"
#include "comm_engine/thread/device/dfx/dfx_circular_queue_pub.h"

HcommResult
HcommThreadRegisterDfx(ThreadHandle thread, std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback);
HcommResult HcommThreadRegisterCheckExecStatus(ThreadHandle thread, std::function<HcclResult(bool)> callback);
HcommResult HcommNewThreadRegisterDfx(ThreadHandle thread, std::function<void(Hccl::TaskInfoCircularQueue*)> callback);
HcommResult HcommNewThreadRegisterGetLatestDfxOpInfo(ThreadHandle thread, std::function<const void*()> callback);

#endif // HCOMM_THREAD_DFX_PUB_H
