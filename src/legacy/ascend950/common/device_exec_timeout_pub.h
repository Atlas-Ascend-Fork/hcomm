/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCOMM_DEVICE_EXEC_TIMEOUT_PUB_H
#define HCOMM_DEVICE_EXEC_TIMEOUT_PUB_H

#include <atomic>
#include <cstdint>
#include "common/device/plf_resource_runtime_pub.h"

namespace Hccl {
// 资源层不再直接依赖 CommunicatorImplLiteMgr：
// 上层注册超时读取回调，资源层只读取。
using DeviceExecTimeoutGetter = PlfDeviceExecTimeoutGetter;
#ifdef CCL_KERNEL_AICPU
inline void SetDeviceExecTimeoutGetter(DeviceExecTimeoutGetter getter) { HcclSetPlfDeviceExecTimeoutGetter(getter); }
inline uint32_t GetDeviceExecTimeout() { return HcclGetPlfDeviceExecTimeout(); }
#else
inline std::atomic<DeviceExecTimeoutGetter>& DeviceExecTimeoutGetterRef()
{
    static std::atomic<DeviceExecTimeoutGetter> getter{nullptr};
    return getter;
}
inline void SetDeviceExecTimeoutGetter(DeviceExecTimeoutGetter getter)
{
    DeviceExecTimeoutGetterRef().store(getter, std::memory_order_release);
}
inline uint32_t GetDeviceExecTimeout()
{
    const auto getter = DeviceExecTimeoutGetterRef().load(std::memory_order_acquire);
    return getter == nullptr ? PLF_DEVICE_EXEC_TIMEOUT_DEFAULT : getter();
}
#endif
} // namespace Hccl
#endif
