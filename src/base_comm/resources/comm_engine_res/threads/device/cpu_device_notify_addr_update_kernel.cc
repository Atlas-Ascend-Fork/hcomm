/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cpu_device_notify_addr_update_kernel.h"
#include "dpu_sync_layout.h"
#include "cpu_device_thread_registry.h"
#include "thread_entity_utils.h"
#include "log.h"

namespace {
constexpr uint32_t CPU_NOTIFY_ADDR_UPDATE_OK = 0;
constexpr uint32_t CPU_NOTIFY_ADDR_UPDATE_FAIL = 1;
} // namespace

extern "C" __attribute__((visibility("default"))) uint32_t RunCpuThreadNotifyAddrUpdate(void* args)
{
    if (args == nullptr) {
        HCCL_ERROR("[RunCpuThreadNotifyAddrUpdate] args is null.");
        return CPU_NOTIFY_ADDR_UPDATE_FAIL;
    }
    const auto* param = reinterpret_cast<const hccl::CpuThreadNotifyAddrUpdateParam*>(args);
    if (param->threadHandle == 0 || param->notifyNum == 0 || param->notifyHostAddr == 0 || param->notifyDevAddr == 0) {
        HCCL_ERROR(
            "[RunCpuThreadNotifyAddrUpdate] invalid param, threadHandle[0x%llx], notifyNum[%u], "
            "notifyHostAddr[0x%llx], notifyDevAddr[0x%llx]",
            param->threadHandle, param->notifyNum, param->notifyHostAddr, param->notifyDevAddr);
        return CPU_NOTIFY_ADDR_UPDATE_FAIL;
    }

    auto threadHandle = param->threadHandle;
    HcclResult unwrapRet = hccl::UnwrapThreadHandle(threadHandle);
    if (unwrapRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[RunCpuThreadNotifyAddrUpdate] unwrap thread handle failed, handle[0x%llx], ret[0x%x]",
            param->threadHandle, unwrapRet);
        return CPU_NOTIFY_ADDR_UPDATE_FAIL;
    }
    std::shared_ptr<hccl::CpuDeviceThread> thread = FindCpuDeviceThread(threadHandle);
    if (thread == nullptr) {
        HCCL_ERROR("[RunCpuThreadNotifyAddrUpdate] dpu thread not found, handle[0x%llx]", param->threadHandle);
        return CPU_NOTIFY_ADDR_UPDATE_FAIL;
    }
    thread->UpdateNotifyInfo(param->notifyNum, param->notifyHostAddr, param->notifyDevAddr);
    return CPU_NOTIFY_ADDR_UPDATE_OK;
}
