/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_INTERFACE_H
#define DPU_INTERFACE_H

#include <cstdint>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>
#include "task_service.h"

extern std::mutex g_serMapMutex;
extern std::unordered_map<std::string, std::unordered_map<uint32_t, std::unique_ptr<Hccl::TaskService>>>
    g_taskServiceMap;
extern std::unordered_map<std::string, std::unordered_map<uint32_t, void*>> g_taskExpMemMap;

std::mutex& GetSerMapMutex();
void RegisterTaskService(const std::string& commId, uint32_t deviceId, std::unique_ptr<Hccl::TaskService> taskService);
void EraseTaskService(const std::string& commId, uint32_t deviceId);
Hccl::TaskService* FindTaskService(const std::string& commId, uint32_t deviceId);
size_t GetTaskServiceMapSize();
void RegisterTaskExpMem(const std::string& commId, uint32_t deviceId, void* taskExpMem);
void EraseTaskExpMem(const std::string& commId, uint32_t deviceId);
void* FindTaskExpMem(const std::string& commId, uint32_t deviceId);
bool FindDpuExceptionByDevice(uint32_t deviceId, std::string& commId, void*& taskExpPtr, uint16_t& hcclRet);

extern "C" {
__attribute__((visibility("default"))) uint32_t RunDpuRpcSrvLaunch(const uint64_t args);
__attribute__((visibility("default"))) void SetDpuExecTimeout(uint32_t timeoutSec);
}

namespace Hccl {
constexpr uint8_t TASK_TERMINATE_RESPONSE = 3;

// DPU执行超时默认值(秒)，须与EnvRtsConfig中HCCL_EXEC_TIMEOUT默认值(NOTIFY_DEFAULT_WAIT_TIME)保持一致
constexpr uint32_t DPU_EXEC_TIMEOUT_DEFAULT_S = 1836U;

// DPU执行超时时间(秒)由宿主库在launch前通过SetDpuExecTimeout下发，未下发时取默认值
uint32_t GetDpuExecTimeout();

struct DpuKernelLaunchParam {
    u64 memorySize;
    void* deviceMem;
    void* hostMem;
    uint32_t deviceId;
    std::string commId;
    void* taskExpMem;
};

} // namespace Hccl

#endif // DPU_INTERFACE_H
