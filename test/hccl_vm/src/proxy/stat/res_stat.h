/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RES_STAT_H
#define RES_STAT_H

#include <atomic>
#include <cstdint>

#include "sim_common_defs.h"

namespace HcclSim {

struct MemResStat {
    std::atomic<uint64_t> allocDevBytes{0};  // 申请的device内存数
    std::atomic<uint64_t> freeDevBytes{0};   // 释放的device内存数
    std::atomic<uint64_t> allocHostBytes{0}; // 申请的host内存数
    std::atomic<uint64_t> freeHostBytes{0};  // 释放的host内存数
    std::atomic<uint64_t> allocVirBytes{0};  // 申请的虚拟内存数
    std::atomic<uint64_t> freeVirBytes{0};   // 释放的虚拟内存数
    std::atomic<uint64_t> allocPhyBytes{0};  // 申请的物理内存数
    std::atomic<uint64_t> freePhyBytes{0};   // 释放的物理内存数
};

struct JettyResStat {
    std::atomic<uint32_t> allocCount{0}; // 申请的数量
    std::atomic<uint32_t> freeCount{0};  // 释放的数量
};

struct QpResStat {
    std::atomic<uint32_t> allocCount{0}; // 申请的数量
    std::atomic<uint32_t> freeCount{0};  // 释放的数量
};

struct NotifyResStat {
    std::atomic<uint32_t> allocCount{0}; // 申请的数量
    std::atomic<uint32_t> freeCount{0};  // 释放的数量
};

struct AIVCoreResStat {
    std::atomic<uint32_t> usedCoreMax{0}; // 最大使用AIV核的数量
};

struct SimResStat {
    MemResStat mem;
    JettyResStat jetty;
    QpResStat qp;
    NotifyResStat notify;
    AIVCoreResStat aivCore;
};

// 按 deviceId 取统计条目（不存在则创建）；返回引用进程生命周期内稳定
SimResStat& GetResStat(uint64_t deviceId);

// 遍历所有已统计 device，各写一个
// <installRoot>/data/stat/<podId>_<serverId>_<boardId>_<deviceId>_res_<时间戳>.csv
// (boardId: POD形态4层拓扑下的board序号; server形态恒为0)
HcclVmResult DumpResStatToCsv();

} // namespace HcclSim

#define HCCL_VM_STAT_MEM_ALLOC_DEV(devId, bytes)         \
    (::HcclSim::GetResStat(static_cast<uint64_t>(devId)) \
         .mem.allocDevBytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed))
#define HCCL_VM_STAT_MEM_FREE_DEV(devId, bytes)          \
    (::HcclSim::GetResStat(static_cast<uint64_t>(devId)) \
         .mem.freeDevBytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed))
#define HCCL_VM_STAT_MEM_ALLOC_HOST(devId, bytes)        \
    (::HcclSim::GetResStat(static_cast<uint64_t>(devId)) \
         .mem.allocHostBytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed))
#define HCCL_VM_STAT_MEM_FREE_HOST(devId, bytes)         \
    (::HcclSim::GetResStat(static_cast<uint64_t>(devId)) \
         .mem.freeHostBytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed))

#define HCCL_VM_STAT_MEM_ALLOC_VIR(devId, bytes)         \
    (::HcclSim::GetResStat(static_cast<uint64_t>(devId)) \
         .mem.allocVirBytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed))
#define HCCL_VM_STAT_MEM_FREE_VIR(devId, bytes)          \
    (::HcclSim::GetResStat(static_cast<uint64_t>(devId)) \
         .mem.freeVirBytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed))
#define HCCL_VM_STAT_MEM_ALLOC_PHY(devId, bytes)         \
    (::HcclSim::GetResStat(static_cast<uint64_t>(devId)) \
         .mem.allocPhyBytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed))
#define HCCL_VM_STAT_MEM_FREE_PHY(devId, bytes)          \
    (::HcclSim::GetResStat(static_cast<uint64_t>(devId)) \
         .mem.freePhyBytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed))

#define HCCL_VM_STAT_JETTY_ALLOC(devId) \
    (::HcclSim::GetResStat(static_cast<uint64_t>(devId)).jetty.allocCount.fetch_add(1U, std::memory_order_relaxed))
#define HCCL_VM_STAT_JETTY_FREE(devId) \
    (::HcclSim::GetResStat(static_cast<uint64_t>(devId)).jetty.freeCount.fetch_add(1U, std::memory_order_relaxed))

#define HCCL_VM_STAT_QP_ALLOC(devId) \
    (::HcclSim::GetResStat(static_cast<uint64_t>(devId)).qp.allocCount.fetch_add(1U, std::memory_order_relaxed))
#define HCCL_VM_STAT_QP_FREE(devId) \
    (::HcclSim::GetResStat(static_cast<uint64_t>(devId)).qp.freeCount.fetch_add(1U, std::memory_order_relaxed))

#define HCCL_VM_STAT_NOTIFY_ALLOC(devId) \
    (::HcclSim::GetResStat(static_cast<uint64_t>(devId)).notify.allocCount.fetch_add(1U, std::memory_order_relaxed))
#define HCCL_VM_STAT_NOTIFY_FREE(devId) \
    (::HcclSim::GetResStat(static_cast<uint64_t>(devId)).notify.freeCount.fetch_add(1U, std::memory_order_relaxed))

// 最大值语义：CAS循环更新，不是累加
#define HCCL_VM_STAT_AIV_CORE_USED(devId, n)                                                                        \
    do {                                                                                                            \
        auto& aivUsedMax = ::HcclSim::GetResStat(static_cast<uint64_t>(devId)).aivCore.usedCoreMax;                 \
        uint32_t curVal = aivUsedMax.load(std::memory_order_relaxed);                                               \
        while (curVal < static_cast<uint32_t>(n)                                                                    \
               && !aivUsedMax.compare_exchange_weak(curVal, static_cast<uint32_t>(n), std::memory_order_relaxed)) { \
        }                                                                                                           \
    } while (0)

#endif // RES_STAT_H
