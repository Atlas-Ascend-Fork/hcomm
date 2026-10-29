/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dpu_kernel_entrance.h"
#include <atomic>
#include "log.h"
#include "acl/acl_rt.h"

using namespace Hccl;
using u64 = unsigned long long;
std::unordered_map<std::string, std::unordered_map<uint32_t, std::unique_ptr<Hccl::TaskService>>> g_taskServiceMap;
std::unordered_map<std::string, std::unordered_map<uint32_t, void*>> g_taskExpMemMap;
std::mutex g_serMapMutex;
namespace {
// DPU执行超时时间(秒)，由宿主库通过SetDpuExecTimeout下发，未下发时保持默认值
std::atomic<uint32_t> g_dpuExecTimeoutSec{DPU_EXEC_TIMEOUT_DEFAULT_S};
} // namespace

extern "C" void SetDpuExecTimeout(uint32_t timeoutSec)
{
    g_dpuExecTimeoutSec.store(timeoutSec, std::memory_order_relaxed);
}

namespace Hccl {
uint32_t GetDpuExecTimeout() { return g_dpuExecTimeoutSec.load(std::memory_order_relaxed); }
} // namespace Hccl

std::mutex& GetSerMapMutex() { return g_serMapMutex; }

void RegisterTaskService(const std::string& commId, uint32_t deviceId, std::unique_ptr<Hccl::TaskService> taskService)
{
    g_taskServiceMap[commId][deviceId] = std::move(taskService);
}

void EraseTaskService(const std::string& commId, uint32_t deviceId)
{
    auto outerIt = g_taskServiceMap.find(commId);
    if (outerIt != g_taskServiceMap.end()) {
        outerIt->second.erase(deviceId);
        if (outerIt->second.empty()) {
            g_taskServiceMap.erase(commId);
        }
    }
}

Hccl::TaskService* FindTaskService(const std::string& commId, uint32_t deviceId)
{
    auto outerIt = g_taskServiceMap.find(commId);
    if (outerIt == g_taskServiceMap.end()) {
        return nullptr;
    }
    auto innerIt = outerIt->second.find(deviceId);
    if (innerIt == outerIt->second.end() || innerIt->second == nullptr) {
        return nullptr;
    }
    return innerIt->second.get();
}

size_t GetTaskServiceMapSize() { return g_taskServiceMap.size(); }

void RegisterTaskExpMem(const std::string& commId, uint32_t deviceId, void* taskExpMem)
{
    g_taskExpMemMap[commId][deviceId] = taskExpMem;
}

void EraseTaskExpMem(const std::string& commId, uint32_t deviceId)
{
    auto outerIt = g_taskExpMemMap.find(commId);
    if (outerIt != g_taskExpMemMap.end()) {
        outerIt->second.erase(deviceId);
        if (outerIt->second.empty()) {
            g_taskExpMemMap.erase(commId);
        }
    }
}

void* FindTaskExpMem(const std::string& commId, uint32_t deviceId)
{
    auto outerIt = g_taskExpMemMap.find(commId);
    if (outerIt == g_taskExpMemMap.end()) {
        return nullptr;
    }
    auto innerIt = outerIt->second.find(deviceId);
    if (innerIt == outerIt->second.end() || innerIt->second == nullptr) {
        return nullptr;
    }
    return innerIt->second;
}

bool FindDpuExceptionByDevice(uint32_t deviceId, std::string& commId, void*& taskExpPtr, uint16_t& hcclRet)
{
    for (const auto& pairMap : g_taskExpMemMap) {
        auto innerIt = pairMap.second.find(deviceId);
        if (innerIt == pairMap.second.end() || innerIt->second == nullptr) {
            continue;
        }
        taskExpPtr = innerIt->second;
        auto ret = memcpy_s(
            &hcclRet, sizeof(uint16_t), static_cast<uint8_t*>(taskExpPtr) + sizeof(uint8_t) + sizeof(uint16_t),
            sizeof(uint16_t));
        if (ret != EOK) {
            HCCL_ERROR("[FindDpuExceptionByDevice] memcpy_s failed, ret[%d].", ret);
            return false;
        }
        if (hcclRet != 0) {
            commId = pairMap.first;
            return true;
        }
    }
    return false;
}
extern "C" {
__attribute__((visibility("default"))) uint32_t RunDpuRpcSrvLaunch(const uint64_t args)
{
    HCCL_INFO("[%s] Launch Dpu Kernel: 0x%lx", __func__, args);
    if (args == 0) {
        HCCL_ERROR("[%s] args is null.", __func__);
        return HCCL_E_PARA;
    }

    if (reinterpret_cast<uint64_t>(args) + sizeof(DpuKernelLaunchParam) < reinterpret_cast<uint64_t>(args)) {
        HCCL_ERROR("[%s] Invalid args address.", __func__);
        return HCCL_E_PARA;
    }
    // 解析参数信息
    DpuKernelLaunchParam* params = reinterpret_cast<DpuKernelLaunchParam*>(args);

    HCCL_RUN_INFO(
        "[%s] DpuKernelLaunchParam{commId:%s; memorySize:%lu; deviceMem:%p; hostMem:%p, taskExpMem:%p; devId:%u}",
        __func__, params->commId.c_str(), params->memorySize, params->deviceMem, params->hostMem, params->taskExpMem,
        params->deviceId);

    if (params->memorySize == 0) {
        HCCL_ERROR("[%s] memorySize is 0.", __func__);
        return HCCL_E_PARA;
    }
    if (params->deviceMem == nullptr || params->hostMem == nullptr || params->taskExpMem == nullptr) {
        HCCL_ERROR(
            "[%s] deviceMem[%p] or hostMem[%p] or taskExpMem[%p] is nullptr.", __func__, params->deviceMem,
            params->hostMem, params->taskExpMem);
        return HCCL_E_PARA;
    }

    // 实例化TaskService
    std::unique_ptr<Hccl::TaskService> taskService = std::make_unique<Hccl::TaskService>(
        params->deviceMem, params->memorySize, params->hostMem, params->memorySize, params->commId, params->deviceId);

    aclError ret = aclrtSetDevice(params->deviceId);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[%s] set device fail. DeviceId: %u.", __func__, params->deviceId);
        return HCCL_E_RUNTIME;
    }

    // 设置到通信域中保存 map<commId, map<devid, TaskService>>与map<commId, map<devid,taskExpMem>>
    HCCL_INFO("[%s] save TaskService", __func__);
    Hccl::TaskService* svcPtr = nullptr;
    {
        std::lock_guard<std::mutex> lock(GetSerMapMutex());
        RegisterTaskService(params->commId, params->deviceId, std::move(taskService));
        RegisterTaskExpMem(params->commId, params->deviceId, params->taskExpMem);
        svcPtr = FindTaskService(params->commId, params->deviceId);
    }

    // Run
    HCCL_INFO("[%s] start to TaskRun", __func__);
    HcclResult hcclRet = svcPtr->TaskRun();
    if (hcclRet != HCCL_SUCCESS) {
        uint8_t newFlag = TASK_TERMINATE_RESPONSE;
        errno_t cpyRet = memcpy_s(static_cast<uint8_t*>(params->deviceMem), sizeof(newFlag), &newFlag, sizeof(newFlag));
        if (cpyRet != EOK) {
            HCCL_ERROR("set exit flag failed: %d", cpyRet);
            return HCCL_E_INTERNAL;
        }
    }

    // End
    return hcclRet;
}
}
