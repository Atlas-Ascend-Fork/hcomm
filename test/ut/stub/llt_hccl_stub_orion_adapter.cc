/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "hccl/hccl_types.h"
#include "acl/acl_rt.h"
#include "orion_adapter_hal.h"

extern "C" {

// Stub implementation for HrtSetXpuDevice
HcclResult HrtSetXpuDevice(uint32_t devType, const uint32_t devId)
{
    (void)devType;
    (void)devId;
    return HCCL_SUCCESS;
}

// Stub implementation for HrtResetXpuDevice
HcclResult HrtResetXpuDevice(uint32_t devType, const uint32_t devId)
{
    (void)devType;
    (void)devId;
    return HCCL_SUCCESS;
}
}

namespace Hccl {
// Stub implementation for HrtHal*：仅提供符号定义，供 dpu_shmem_mgr 链接。
// 实际 UT 用例经 mockcpp 的 MOCKER 拦截（outBound 回填 connectType=UB），此定义体不执行。

HcclResult HrtHalGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t& value)
{
    (void)devId;
    (void)moduleType;
    (void)infoType;
    value = 0; // 默认 PCIE；测试经 mock 拦截时回填 UB
    return HCCL_SUCCESS;
}

HcclResult HrtHalHostRegister(void* srcPtr, uint64_t size, uint32_t flag, int32_t deviceLogicId, void** dstPtr)
{
    (void)srcPtr;
    (void)size;
    (void)flag;
    (void)deviceLogicId;
    if (dstPtr != nullptr) {
        *dstPtr = srcPtr;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtHalHostUnregister(void* ptr, int32_t deviceLogicId)
{
    (void)ptr;
    (void)deviceLogicId;
    return HCCL_SUCCESS;
}
} // namespace Hccl
