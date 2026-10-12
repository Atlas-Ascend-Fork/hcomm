/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define HCCL_VM_MODULE "HCOMM_DEV_STUB"

#include <cstdint>
#include <dlfcn.h>
#include "sim_log.h"
#include "hccl_device_pub.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int32_t HcommSendRequest(uint64_t handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId)
{
    HCCL_VM_INFO(
        "[HcommSendRequest], handle={}, msgTag={}, src={:p}, sizeByte={}, msgId_ptr={:p}", handle,
        (msgTag != nullptr) ? msgTag : "nullptr", src, sizeByte, (void*)msgId);

    using RealFunc
        = int32_t (*)(uint64_t handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId);

    // 获取原始真实函数地址
    const auto realFn = reinterpret_cast<RealFunc>(dlsym(RTLD_NEXT, __func__));
    if (realFn == nullptr) {
        const char* err = dlerror();
        HCCL_VM_ERROR("HcommSendRequest dlsym failed, err={}", err ? err : "unknown");
        return -1;
    }

    // HostDPU模式，在此记录device与dpu交互时的streamId
    uint32_t deviceId = GetCurDeviceKey();
    uint32_t sqId = GetLastQuerySqId();
    SetDpuStreamId(deviceId, sqId);
    HCCL_VM_INFO("Set DPU streamId, deviceId={} streamId={}", deviceId, sqId);

    // 转发调用真实底层函数
    int32_t ret = realFn(handle, msgTag, src, sizeByte, msgId);

    return ret;
}

#ifdef __cplusplus
}
#endif // __cplusplus
