/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define HCCL_VM_MODULE "DPU_KERNEL_MGR_STUB"

#include "sim_dpu_kernel_lib_mgr.h"
#include <cstdlib>
#include <dlfcn.h>
#include <thread>
#include <vector>
#include "sim_log.h"
#include "sim_common_defs.h"
#include "sim_common_api.h"

namespace sim {

DpuKernelLibManager::DpuKernelLibManager() = default;

DpuKernelLibManager& DpuKernelLibManager::GetInstance()
{
    static DpuKernelLibManager instance;
    return instance;
}

std::string DpuKernelLibManager::ResolveLibPath(const std::string& soName) const
{
    const char* ascendHomePath = std::getenv("ASCEND_HOME_PATH");
    if (ascendHomePath == nullptr) {
        HCCL_VM_ERROR("ASCEND_HOME_PATH not set, cannot resolve DPU kernel path");
        return "";
    }
    return std::string(ascendHomePath) + "/opp/built-in/op_impl/dpu/" + soName;
}

DpuKernelFunc DpuKernelLibManager::GetOrLoadFunc(const std::string& soName, const std::string& kernelName)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    auto symIt = m_symbolCache.find(kernelName);
    if (symIt != m_symbolCache.end()) {
        return symIt->second;
    }

    // 查找或 dlopen so
    void* libHandle = nullptr;
    auto handleIt = m_soHandles.find(soName);
    if (handleIt != m_soHandles.end()) {
        libHandle = handleIt->second;
    } else {
        std::string libPath = ResolveLibPath(soName);
        if (libPath.empty()) {
            return nullptr;
        }
        libHandle = dlopen(libPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (libHandle == nullptr) {
            HCCL_VM_ERROR("dlopen {} failed: {}", libPath, dlerror());
            return nullptr;
        }
        m_soHandles[soName] = libHandle;
        HCCL_VM_INFO("loaded DPU so: {}", libPath);
    }

    void* fn = dlsym(libHandle, kernelName.c_str());
    if (fn == nullptr) {
        HCCL_VM_ERROR("dlsym {} in {} failed: {}", kernelName, soName, dlerror());
        return nullptr;
    }

    DpuKernelFunc func = reinterpret_cast<DpuKernelFunc>(fn);
    m_symbolCache[kernelName] = func;
    HCCL_VM_INFO("resolved DPU kernel symbol: {} in {}", kernelName, soName);
    return func;
}

void DpuKernelLibManager::Launch(const std::string& kernelName, DpuKernelFunc fn, void* hostArgs, size_t argsSize)
{
    constexpr size_t maxDpuKernelArgsSize = 4096;
    if (argsSize > maxDpuKernelArgsSize || (argsSize > 0 && hostArgs == nullptr)) {
        HCCL_VM_ERROR("invalid DPU kernel args: kernel={}, hostArgs={:p}, argsSize={}", kernelName, hostArgs, argsSize);
        return;
    }

    // 对齐真实语义：launch 返回前完成参数拷贝，规避 detach 线程读到悬垂栈指针。
    std::vector<uint8_t> argsCopy;
    if (argsSize > 0) {
        const uint8_t* bytes = static_cast<const uint8_t*>(hostArgs);
        argsCopy.assign(bytes, bytes + argsSize);
    }
    HCCL_VM_INFO(
        "Launching DPU kernel {} in new thread, hostArgs={:p} (argsValue=0x{:x}, argsSize={})", kernelName, hostArgs,
        reinterpret_cast<uintptr_t>(argsCopy.data()), argsSize);

    std::thread([fn, kernelName, argsCopy = std::move(argsCopy)]() {
        HCCL_VM_INFO("DPU kernel thread started: {}", kernelName);
        uint32_t result = fn(reinterpret_cast<uint64_t>(argsCopy.data()));
        HCCL_VM_INFO("DPU kernel thread finished: {} result={}", kernelName, result);
    }).detach();

    HCCL_VM_INFO("DPU kernel {} launched successfully", kernelName);
}

void DpuKernelLibManager::Cleanup()
{
    std::lock_guard<std::mutex> lock(m_mutex);

    HCCL_VM_INFO("cleanup: {} SOs, {} symbols", m_soHandles.size(), m_symbolCache.size());

    if (!m_symbolCache.empty() || !m_soHandles.empty()) {
        HCCL_VM_WARN("detached DPU worker threads may still be running; dlclose may unload in-use code");
    }

    m_symbolCache.clear();

    for (auto& [name, handle] : m_soHandles) {
        if (handle != nullptr) {
            dlclose(handle);
            HCCL_VM_INFO("dlclose({})", name);
        }
    }
    m_soHandles.clear();
}

DpuKernelLibManager::~DpuKernelLibManager() { Cleanup(); }

} // namespace sim
