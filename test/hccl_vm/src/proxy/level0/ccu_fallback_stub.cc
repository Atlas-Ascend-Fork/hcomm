/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "CCU_FALLBACK_STUB"

#include <cstdint>

#include "hccl/hccl_types.h"
#include "sim_log.h"

// 覆盖 hccl/src/ops/op_common/ccu_fallback.cc 中两个 __attribute__((weak)) C 接口，
// 仿真中直接短路返回成功，跳过真实协商/回退逻辑。
// 接口声明见 hccl/src/ops/op_common/inc/ccu_fallback_c.h。
extern "C" {

HcclResult CheckCcuResNegotiationC(HcclComm comm, const void* param, bool localResAvailable)
{
    HCCL_VM_INFO("this func is empty");
    if (!localResAvailable) {
        return HcclResult::HCCL_E_UNAVAIL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CheckCcuParamAndFallbackC(HcclComm comm, void* param, void** topoInfo, char* algNameBuf, uint32_t algNameBufLen)
{
    HCCL_VM_INFO("this func is empty");
    return HcclResult::HCCL_SUCCESS;
}

} // extern "C"
