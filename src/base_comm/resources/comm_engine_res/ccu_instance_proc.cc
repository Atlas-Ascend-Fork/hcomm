/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_instance_proc.h"
#include "log.h"

namespace hcomm {

// 引擎族收尾（幂等，initializedFlag_ 守护）：graceful 执行（清 insMap_ 触发 ~CcuInstance 触设备链）；
// RESET 跳过并告警（README R5）
void CcuInstanceProc::DeInit(ExitProfile profile)
{
    if (!ccuInstanceMgr_.IsInitialized()) {
        return;
    }
    if (profile == ExitProfile::EXIT_PROFILE_RESET) {
        HCCL_WARNING(
            "[CcuInstanceProc][%s] RESET profile: skip CCU instance deinit (device may be unavailable), "
            "residual cleanup deferred to destructor",
            __func__);
        return;
    }
    (void)ccuInstanceMgr_.Deinit();
}

// 越界 id 回落的 reserve 兜底实例（语义见 README R9）：静态对象，在 BaseCommMgr 之后回收
CcuVarEventResMgr CcuInstanceProc::reserveCcuVarEventResMgr_{};
CcuKernelMgr CcuInstanceProc::reserveCcuKernelMgr_{};
CcuInstanceMgr CcuInstanceProc::reserveCcuInstanceMgr_{};

} // namespace hcomm
