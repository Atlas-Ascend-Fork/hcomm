/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLSIM_STORE_PUB_H
#define HCCLSIM_STORE_PUB_H

#include <cstdint>
#include <memory>

#include "sim_common_defs.h"
#include "runtime_state/sim_models.h"

// Virtual Runtime
struct VmPtrReleaser {
    sim::runtime::PhyMemBlock phyMem;
    void operator()(void* ptr) const;
};

using VmUniquePtr = std::unique_ptr<void, VmPtrReleaser>;

HcclSim::HcclVmResult GetAddrByOffset(uint64_t offset, VmUniquePtr& addrPtr);

/**
 * @brief 地址解析（按 device 版本）：在常规虚拟内存块映射之外，额外支持 CCU 缓冲(MS) 窗口。
 *
 * CCU 北向（level1 劫持）会把 CCU 片上 scratch 编码成合成地址 `0x8000000 + cursor`
 * （见 ccu_kernel_sim.cc / task_meta_translator_v3.cc）。runner 需要真实可读写内存才能搬运，
 * 故此处按 device 提供该窗口的进程本地后备内存。
 * @param offset   设备地址/偏移（真实 device 地址或 CCU MS 窗口合成地址）。
 * @param deviceId 该地址所属 device（MS 窗口按 device 隔离各 rank 的 scratch）。
 * @param addrPtr  输出：解析到的可读写指针。
 * @return 成功返回 HCCL_SIM_SUCCESS。
 */
HcclSim::HcclVmResult GetAddrByOffset(uint64_t offset, uint32_t deviceId, VmUniquePtr& addrPtr);

HcclSim::HcclVmResult InsertTaskToCollection(HcclTaskMetaData* task, uint32_t* index);

#endif
