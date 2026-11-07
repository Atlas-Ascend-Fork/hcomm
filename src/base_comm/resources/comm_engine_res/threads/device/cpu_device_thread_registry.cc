/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cpu_device_thread_registry.h"
#include <mutex>
#include <unordered_map>
#include "log.h"

namespace {
std::unordered_map<ThreadHandle, std::shared_ptr<hccl::CpuDeviceThread>> g_cpuDeviceThreadMap;
std::mutex g_cpuDeviceThreadMapMutex;
} // namespace

void RegisterCpuDeviceThread(ThreadHandle handle, std::shared_ptr<hccl::CpuDeviceThread> entity)
{
    if (handle == 0 || entity == nullptr) {
        HCCL_ERROR("[RegisterCpuDeviceThread] invalid arg, handle[0x%llx], entity[%p]", handle, entity.get());
        return;
    }
    std::lock_guard<std::mutex> lock(g_cpuDeviceThreadMapMutex);
    auto it = g_cpuDeviceThreadMap.find(handle);
    if (it == g_cpuDeviceThreadMap.end()) {
        g_cpuDeviceThreadMap.insert({handle, entity});
        return;
    }
    HCCL_ERROR(
        "[RegisterCpuDeviceThread] handle already registered, handle[0x%llx], old entity[%p], new entity[%p]", handle,
        it->second.get(), entity.get());
}

std::shared_ptr<hccl::CpuDeviceThread> FindCpuDeviceThread(ThreadHandle handle)
{
    std::lock_guard<std::mutex> lock(g_cpuDeviceThreadMapMutex);
    auto it = g_cpuDeviceThreadMap.find(handle);
    return it == g_cpuDeviceThreadMap.end() ? nullptr : it->second;
}

bool EraseCpuDeviceThread(ThreadHandle handle)
{
    std::lock_guard<std::mutex> lock(g_cpuDeviceThreadMapMutex);
    return g_cpuDeviceThreadMap.erase(handle) != 0;
}

void ClearCpuDeviceThread()
{
    std::lock_guard<std::mutex> lock(g_cpuDeviceThreadMapMutex);
    g_cpuDeviceThreadMap.clear();
}
