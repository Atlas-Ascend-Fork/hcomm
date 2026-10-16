/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_TASKEXP_DEV_MEM_REGISTRY_H
#define DPU_TASKEXP_DEV_MEM_REGISTRY_H

#include <mutex>
#include <string>
#include <unordered_map>

// 设备侧 taskexception 共享内存 VA 登记表（旧方案 DPU taskexception，随该机制下线一并删除）。
// 内核入口按 commId 登记，通信域侧按 commId 查询/清理；独立于 commMap，因登记帧可能先于通信域注册帧。
// 收编原 aicpu_taskexp_registry 的全局函数与 map，由 CollCommAicpuMgr 持有。
// 存活契约：表内是宿主侧 LegacyDpuThread 持有的共享内存的 device VA，所有权不在本表；
// 条目随设备侧通信域对象销毁（CollCommAicpuMgr::DestroyComm）摘除，故 Find 结果只在同 commId 的
// 通信域仍存活期间有效，调用方不得把结果缓存到通信域生命周期之外。
class DpuTaskExpDevMemRegistry {
public:
    void Register(const std::string& commId, void* taskExpDevMem);

    /**
     * @brief 按 commId 取 taskexception 共享内存 device VA
     * @return 命中返回 VA，未登记返回 nullptr（仅表示尚未登记，不代表该通信域无 taskexception 内存）
     * @note 返回值不携带所有权与保活，见类头存活契约
     */
    void* Find(const std::string& commId);

    /**
     * @brief 按地址条件摘除：同名 commId 复用重建后新登记已换新地址时，迟到的旧销毁不得误删新登记
     */
    void Erase(const std::string& commId, void* taskExpDevMem);
    void Clear();

private:
    std::mutex mutex_;
    std::unordered_map<std::string, void*> map_;
};

#endif // DPU_TASKEXP_DEV_MEM_REGISTRY_H
