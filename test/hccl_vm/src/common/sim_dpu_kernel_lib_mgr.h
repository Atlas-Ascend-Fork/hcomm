/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_DPU_KERNEL_LIB_MGR_H
#define SIM_DPU_KERNEL_LIB_MGR_H

#include <cstdint>
#include <map>
#include <mutex>
#include <string>

using DpuKernelFunc = uint32_t (*)(uint64_t args);

namespace sim {

// DPU kernel 动态库管理单例
class DpuKernelLibManager {
public:
    static DpuKernelLibManager& GetInstance();

    DpuKernelFunc GetOrLoadFunc(const std::string& soName, const std::string& kernelName);

    void Launch(const std::string& kernelName, DpuKernelFunc fn, void* hostArgs);

    void Cleanup();

private:
    DpuKernelLibManager();
    ~DpuKernelLibManager();

    DpuKernelLibManager(const DpuKernelLibManager&) = delete;
    DpuKernelLibManager& operator=(const DpuKernelLibManager&) = delete;

    std::string ResolveLibPath(const std::string& soName) const;

    std::mutex m_mutex;
    std::map<std::string, void*> m_soHandles;           // soName -> dlopen handle
    std::map<std::string, DpuKernelFunc> m_symbolCache; // kernelName -> fn
};

} // namespace sim

#endif // SIM_DPU_KERNEL_LIB_MGR_H
