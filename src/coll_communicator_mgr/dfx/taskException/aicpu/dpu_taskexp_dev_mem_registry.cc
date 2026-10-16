/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dpu_taskexp_dev_mem_registry.h"
#include "log.h"

void DpuTaskExpDevMemRegistry::Register(const std::string& commId, void* taskExpDevMem)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(commId);
    if (it == map_.end()) {
        map_.insert({commId, taskExpDevMem});
        return;
    }
    // 登记已推迟到首次 HcclTaskRegister，同名通信域可能出现「旧的还没擦除、新的已登记」；
    // 旧地址随共享内存释放即失效，滞留会被设备侧 DFX 当成本次有效地址写入，故由新地址接管。
    HCCL_WARNING("taskExpDevMem[%p] update old[%p], key is commId[%s]", taskExpDevMem, it->second, commId.c_str());
    it->second = taskExpDevMem;
}

void* DpuTaskExpDevMemRegistry::Find(const std::string& commId)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(commId);
    return (it != map_.end()) ? it->second : nullptr;
}

void DpuTaskExpDevMemRegistry::Erase(const std::string& commId, void* taskExpDevMem)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(commId);
    // 同名 commId 复用重建后新登记已换新地址，迟到的旧销毁命令只能擦自己登记的旧地址，不得误删新登记
    if (it != map_.end() && it->second == taskExpDevMem) {
        map_.erase(it);
    }
}

void DpuTaskExpDevMemRegistry::Clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    map_.clear();
}
