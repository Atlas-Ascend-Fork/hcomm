/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <string>

std::unordered_map<std::string, std::unordered_map<uint32_t, void*>> g_taskExpMemMap;

namespace {
std::unordered_map<std::string, void*> s_taskExpDevMemMap;
std::mutex s_taskExpDevMemMapMutex;
} // namespace

void RegisterTaskExpDevMem(const std::string& commId, void* taskExpDevMem)
{
    std::lock_guard<std::mutex> lock(s_taskExpDevMemMapMutex);
    s_taskExpDevMemMap[commId] = taskExpDevMem;
}

void* FindTaskExpDevMem(const std::string& commId)
{
    std::lock_guard<std::mutex> lock(s_taskExpDevMemMapMutex);
    auto it = s_taskExpDevMemMap.find(commId);
    return (it != s_taskExpDevMemMap.end()) ? it->second : nullptr;
}

void EraseTaskExpDevMem(const std::string& commId)
{
    std::lock_guard<std::mutex> lock(s_taskExpDevMemMapMutex);
    s_taskExpDevMemMap.erase(commId);
}

void ClearTaskExpDevMem()
{
    std::lock_guard<std::mutex> lock(s_taskExpDevMemMapMutex);
    s_taskExpDevMemMap.clear();
}
