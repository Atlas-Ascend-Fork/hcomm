/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "store_sim_device_memory_manager.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <thread>
#include <unistd.h>

#include "sim_log.h"
#include "store_sim_memory_manager.h"
#include "store_sim_comm_pool_policy.h"
#include "runtime_state/sim_models.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "store_sim_run_mode.h"
#include "device_spec/device_spec_registry.h"
#include "device_spec/rank_address_space.h"

namespace sim {
// 静态成员变量初始化
DeviceMemoryManager* DeviceMemoryManager::s_instance = nullptr;
std::mutex DeviceMemoryManager::s_instanceLock;

// 获取单例实例
DeviceMemoryManager& DeviceMemoryManager::GetInstance()
{
    if (s_instance == nullptr) {
        std::lock_guard<std::mutex> lock(s_instanceLock);
        if (s_instance == nullptr) {
            s_instance = new DeviceMemoryManager();
        }
    }
    return *s_instance;
}

DeviceMemoryManager::DeviceMemoryManager() {}

DeviceMemoryManager::~DeviceMemoryManager() {}

const devspec::AddrSpace* DeviceMemoryManager::ResolveAddrSpace(uint64_t devPhyId)
{
    // 调用方已持有 m_virSizeLock
    auto cached = m_addrSpaceCache.find(devPhyId);
    if (cached != m_addrSpaceCache.end()) {
        return cached->second;
    }

    auto& reg = devspec::DeviceSpecRegistry::GetInstance();
    if (reg.Size() == 0) {
        // 注册表尚未加载：尝试从 install root 读 config/device/*.yaml（不依赖 DB）
        (void)reg.LoadAll();
    }

    const devspec::AddrSpace* addr = nullptr;

    // ① DB: devPhyId 是 Device 表主键（GetCurrDeviceId 返回 Context.device_id）
    if (reg.Size() != 0) {
        auto dev = sim::runtime::Db::GetById<sim::runtime::Device>(devPhyId);
        if (dev.ok() && dev.value->soc_version[0] != '\0') {
            const devspec::DeviceSpec* spec = reg.FindBySocVersion(dev.value->soc_version);
            if (spec != nullptr) {
                addr = &spec->addr;
            }
        }
    }

    // ② 兜底：所有已加载 spec 的 address_space 逐字段相同（A5/A6 芯片无关，验收 A14），
    //    故 DB 未就绪时任选一份也能得到正确的分段布局。
    if (addr == nullptr) {
        addr = reg.CommonAddrSpace();
    }

    if (addr != nullptr) {
        m_addrSpaceCache[devPhyId] = addr;
    }
    return addr;
}

void* DeviceMemoryManager::AllocVirMem(uint64_t devPhyId, size_t size)
{
    // ── G2 size 合法性 ────────────────────────────────────────────────────
    // ★ 修 latent bug B3：size==0 时原实现返回非空指针并插入 start_end_ptr == start_ptr
    //   的零长度 VirtualMemBlock，范围查询 Le&&Gt 永不命中 → 静默无效块。
    const uint64_t alignedSize = (static_cast<uint64_t>(size) + 7ULL) & ~7ULL;
    if (alignedSize == 0) {
        HCCL_VM_ERROR("alloc vir mem failed: dev:{:d} invalid size {:d}", devPhyId, size);
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(m_virSizeLock);

    // ── G0 devPhyId == 0：既有 id 空间不一致，保持旧行为 + 告警（见 O14）──────
    //   aclrtReserveMemAddress(:972) / :444 / :1040 传的是 Device.physical_id
    //   （0 起，服务器内首卡合法为 0）；而 aclrtMalloc(:277) 传的是 GetCurrDeviceId()
    //   = Context.device_id（DB 主键，1 起）。两者是**不同 id 空间**却共用同一窗口序列，
    //   这是改造前就存在的缺陷（latent bug B1 的具体表现）。
    //   此处**不改变既有行为**（(0-1) 的 uint64 回绕与改造前逐位一致），仅告警；
    //   id 空间的统一属 S2-2/O14 范围，不在 S4 内顺带修改。
    const bool devKeyIsZero = (devPhyId == 0);
    if (devKeyIsZero && m_degradedWarned.find(devPhyId) == m_degradedWarned.end()) {
        m_degradedWarned[devPhyId] = true;
        HCCL_VM_WARN("dev:0 passed to AllocVirMem: physical_id(0-based) and Device PK(1-based) are two "
                     "different id spaces sharing one VA window sequence (see O14); keeping legacy "
                     "wrap-around behavior for this call");
    }

    const devspec::AddrSpace* addrSpace = devKeyIsZero ? nullptr : ResolveAddrSpace(devPhyId);
    if (addrSpace == nullptr || addrSpace->userMemSegs.empty() || addrSpace->rankStride == 0) {
        // ── 降级路径：无 spec 可用，或 devPhyId == 0 ───────────────────────
        // 退回改造前的线性 bump 行为（VA 逐位一致），但仍加 rank 窗口上界检查
        // （严格优于改造前：改造前完全无上界）。
        // ⚠ 此路径**不跳过 CCU 窗口**；正常情况下 spec 总能从 config/device/*.yaml 加载。
        if (!devKeyIsZero && m_degradedWarned.find(devPhyId) == m_degradedWarned.end()) {
            m_degradedWarned[devPhyId] = true;
            HCCL_VM_WARN(
                "dev:{:d} device spec unavailable, AllocVirMem falls back to linear bump WITHOUT "
                "ccu-hole skipping; check config/device/*.yaml is installed",
                devPhyId);
        }
        constexpr uint64_t kFallbackBase = 0x8000000000;    // = yaml address_space.rank_base
        constexpr uint64_t kFallbackStride = 0x10000000000; // = yaml address_space.rank_stride
        auto& cursor = m_deviceMemCursors[devPhyId];
        if (cursor.empty()) {
            cursor.assign(1, 0);
        }
        const uint64_t used = cursor[0];
        if (used > kFallbackStride || alignedSize > kFallbackStride - used) {
            HCCL_VM_ERROR(
                "alloc vir mem FAILED(degraded): dev:{:d} req:{:#x} exceeds rank window {:#x} "
                "(used {:#x})",
                devPhyId, alignedSize, kFallbackStride, used);
            return nullptr;
        }
        // (devPhyId - 1) 在 devPhyId==0 时按 uint64 回绕，与改造前完全一致
        const uint64_t va = kFallbackBase + (devPhyId - 1) * kFallbackStride + used;
        cursor[0] = used + alignedSize;
        HCCL_VM_INFO("dev:{:d} alloc vir mem:{:#x} size:{} (degraded linear)", devPhyId, va, alignedSize);
        return reinterpret_cast<void*>(va);
    }

    const uint64_t rankIdx = devPhyId - 1; // 口径不变（D5）：rankIdx = Device.physical_id - 1

    // ★ 无 rank 槽位上界：集群 rank 数由拓扑决定，模拟器不设人为上限（原 G1 守卫已删除）。
    //   rankIdx 的唯一硬约束是 VA 计算不得溢出 uint64，由下方 RankBase() 的溢出防护兜底
    //   （溢出返回 0 → 本次分配失败）；47-bit 位宽只是诊断量（R4），不阻断分配。
    const devspec::RankAddressSpace ras(*addrSpace);

    auto& cursors = m_deviceMemCursors[devPhyId];
    if (cursors.size() != ras.UserMemSegCount()) {
        cursors.assign(ras.UserMemSegCount(), 0ULL);
    }

    // ── G3 分段 first-fit：块必须完整落在单个 user_mem 段内（保证不踩 CCU）──
    const uint64_t rankBase = ras.RankBase(rankIdx);
    if (rankBase == 0) {
        HCCL_VM_ERROR(
            "alloc vir mem failed: dev:{:d} rankIdx:{} rank window base overflow (rank_stride={:#x})", devPhyId,
            rankIdx, ras.RankStride());
        return nullptr;
    }
    const uint64_t windowTop = rankBase + ras.RankWindowSize();
    for (size_t i = 0; i < ras.UserMemSegCount(); ++i) {
        const uint64_t segOff = ras.UserMemSegOffset(i);
        const uint64_t segSize = ras.UserMemSegSize(i);
        const uint64_t cur = cursors[i];
        if (cur > segSize || alignedSize > segSize - cur) {
            continue; // 本段放不下 → 试下一段（浪费量严格 < alignedSize）
        }
        const uint64_t va = rankBase + segOff + cur;

        // ── G4 不变量断言（防御式，正常路径不应触发）─────────────────────
        uint32_t hitRank = 0;
        uint32_t hitDie = 0;
        if (va < rankBase || alignedSize > windowTop - va || ras.IsInCcuWindow(va, hitRank, hitDie)
            || ras.IsInCcuWindow(va + alignedSize - 1, hitRank, hitDie)) {
            HCCL_VM_ERROR(
                "alloc vir mem INTERNAL ERROR: dev:{:d} va={:#x} size={:#x} violates "
                "rank-window/ccu invariant (rankBase={:#x} windowTop={:#x})",
                devPhyId, va, alignedSize, rankBase, windowTop);
            return nullptr;
        }

        cursors[i] = cur + alignedSize;
        HCCL_VM_INFO(
            "dev:{:d} rankIdx:{} seg:{} alloc vir mem:{:#x} size:{} (seg used {:#x}/{:#x})", devPhyId, rankIdx, i, va,
            alignedSize, cursors[i], segSize);
        return reinterpret_cast<void*>(va);
    }

    // ── G5 耗尽：VA 超出 device 的 1 TiB 上限，必须报错而非静默回绕 ────────
    uint64_t usedTotal = 0;
    uint64_t capTotal = 0;
    for (size_t i = 0; i < ras.UserMemSegCount(); ++i) {
        usedTotal += cursors[i];
        capTotal += ras.UserMemSegSize(i);
    }
    HCCL_VM_ERROR(
        "alloc vir mem FAILED: dev:{:d} rankIdx:{} req:{:#x} exceeds device address budget. "
        "rank window [{:#x},{:#x}) = {:d} GiB; ccu reserved {:d} GiB; "
        "user_mem total {:#x} ({:d} GiB), used {:#x} ({:d} GiB), all {:d} segment(s) full",
        devPhyId, rankIdx, alignedSize, rankBase, windowTop, ras.RankWindowSize() >> 30, ras.CcuWindowSize() >> 30,
        capTotal, capTotal >> 30, usedTotal, usedTotal >> 30, ras.UserMemSegCount());
    return nullptr;
}

// 释放虚拟内存：**有意不回收 VA 水位**，详见头文件注释与设计文档 §4.5.7.5 / D17 / O13。
void DeviceMemoryManager::FreeVirMem(uint64_t devPhyId, void* virAddr)
{
    HCCL_VM_INFO("dev:{:d} free vir mem:{:p} (VA 不回收，见 D17)", devPhyId, virAddr);
}

std::vector<uint64_t> DeviceMemoryManager::GetUsedPerSegment(uint64_t devPhyId)
{
    std::lock_guard<std::mutex> lock(m_virSizeLock);
    auto it = m_deviceMemCursors.find(devPhyId);
    return (it == m_deviceMemCursors.end()) ? std::vector<uint64_t>() : it->second;
}

// GLOBAL 对称 VA 段基址（bit63），与 bit62 UB 堆段、device<<48 普通段隔离。
static constexpr uint64_t kGlobalSymVaBase = 1ULL << 63;
// 对称 VA 分配粒度（真实 aclrtMemGetAllocationGranularity(RECOMMENDED)=2MiB）。
static constexpr uint64_t kSymVaAllocGranularity = 2ULL * 1024ULL * 1024ULL;

void* DeviceMemoryManager::AllocGlobalVirMem(size_t size)
{
    std::lock_guard<std::mutex> lock(m_globalVirMemLock);
    uint64_t slots = (static_cast<uint64_t>(size) + kSymVaAllocGranularity - 1) / kSymVaAllocGranularity;
    uint64_t virAddr = kGlobalSymVaBase + m_globalVirMemCursor * kSymVaAllocGranularity;
    m_globalVirMemCursor += (slots == 0 ? 1 : slots);
    HCCL_VM_INFO("alloc global sym vir mem:{:p}, size:{:d}", reinterpret_cast<void*>(virAddr), size);
    return reinterpret_cast<void*>(virAddr);
}

// UB 对称堆【全局固定】基址（bit62，跨 rank 同址、不含 device 前缀）。
static constexpr uint64_t kSymHeapGlobalBase = 1ULL << 62;
// 真实 UB 对称堆单成员步距（hcomm 默认 16GB）。
static constexpr uint64_t kSymHeapStride = 16ULL * 1024ULL * 1024ULL * 1024ULL;

void* DeviceMemoryManager::AllocSymHeapVirMem(size_t size)
{
    std::lock_guard<std::mutex> lock(m_symHeapLock);
    uint32_t slot = m_symHeapCount++;
    // 单堆跨度按 reserve 请求的实际尺寸对齐到成员步距（16GB）取整。VA 只发号不占物理内存，
    // 多堆按各自实际跨度依次上移，变长堆之间不会重叠。
    const uint64_t heapSpan = (static_cast<uint64_t>(size) + kSymHeapStride - 1) / kSymHeapStride * kSymHeapStride;
    uint64_t virAddr = kSymHeapGlobalBase + m_symHeapOffset;
    m_symHeapOffset += heapSpan;
    HCCL_VM_INFO(
        "alloc symmetric heap base:{:p}, slot:{:d}, span:{:d}", reinterpret_cast<void*>(virAddr), slot, heapSpan);
    return reinterpret_cast<void*>(virAddr);
}

// 分配物理内存：大块（仅校验模式开）引流到复用区，否则按名独立分配
void* DeviceMemoryManager::AllocPhyMem(const char* name, uint64_t deviceId, size_t size)
{
    if (name == nullptr) {
        return nullptr;
    }
    HCCL_VM_INFO("dev:{:d} alloc phy mem:{}, size:{}", deviceId, name, size);
    bool checkOnlyMode = IsCheckOnlyMode();
    // 超过复用区上界直接报错，不回退真实分配（避免巨块吃满 /dev/shm）。
    if (CommPoolPolicy::ExceedsCeiling(size, checkOnlyMode)) {
        HCCL_VM_ERROR(
            "dev:{:d} alloc phy mem:{} size:{} exceeds pool ceiling:{}, reject", deviceId, name, size,
            CommPoolPolicy::kPoolSize);
        return nullptr;
    }
    if (CommPoolPolicy::ShouldRedirect(size, checkOnlyMode)) {
        return MemoryManager::GetInstance().AcquireMemByName(CommPoolPolicy::kPoolName);
    }
    return MemoryManager::GetInstance().AllocMemByName(name, size);
}

// 释放物理内存（非复用区）：是否在复用区由调用方按 size 判断后分流，这里只做真正的释放。
void DeviceMemoryManager::FreePhyMem(const char* name, uint64_t deviceId)
{
    if (name == nullptr) {
        return;
    }
    HCCL_VM_INFO("dev:{:d} free phy mem:{}", deviceId, name);
    MemoryManager::GetInstance().FreeMemByName(name);
}

// 获取物理内存：大块（仅校验模式开）引流到复用区，否则按名独立分配
void* DeviceMemoryManager::AcquirePhyMem(const char* name, uint64_t deviceId, size_t size)
{
    if (name == nullptr) {
        return nullptr;
    }
    HCCL_VM_INFO("dev:{:d} acquire phy mem:{}, size:{}", deviceId, name, size);
    bool checkOnlyMode = IsCheckOnlyMode();
    // 超过复用区上界直接报错，不回退真实分配。
    if (CommPoolPolicy::ExceedsCeiling(size, checkOnlyMode)) {
        HCCL_VM_ERROR(
            "dev:{:d} acquire phy mem:{} size:{} exceeds pool ceiling:{}, reject", deviceId, name, size,
            CommPoolPolicy::kPoolSize);
        return nullptr;
    }
    if (CommPoolPolicy::ShouldRedirect(size, checkOnlyMode)) {
        return MemoryManager::GetInstance().AcquireMemByName(CommPoolPolicy::kPoolName);
    }
    return MemoryManager::GetInstance().AcquireMemByName(name);
}

// 释放物理内存（非复用区）：是否在复用区由调用方按 size 判断后分流，这里只做真正的释放。
int DeviceMemoryManager::ReleasePhyMem(const char* name, uint64_t deviceId)
{
    if (name == nullptr) {
        return 0;
    }
    HCCL_VM_INFO("dev:{:d} release phy mem:{}", deviceId, name);
    MemoryManager::GetInstance().ReleaseMemByName(name);
    return 0;
}

// 缓存设备地址到host地址；deviceKey==0 说明调用线程无 device 上下文，
// 打 ERROR 并拒绝写入，避免 key=0 槽被多 rank 共享造成静默串数据。
void DeviceMemoryManager::MapDevPtrHostPtr(uint64_t deviceKey, void* devPtr, void* hostPtr)
{
    if (deviceKey == 0) {
        HCCL_VM_ERROR("MapDevPtrHostPtr with deviceKey=0, devPtr:{:p}, hostPtr:{:p}", devPtr, hostPtr);
        return;
    }
    std::lock_guard<std::mutex> lock(m_devPtrHostPtrLock);
    m_devPtrHostPtrMap[{deviceKey, devPtr}] = hostPtr;
}

// 去缓存设备地址到host地址；deviceKey==0 打 ERROR 并拒绝操作（语义同 MapDevPtrHostPtr）。
void DeviceMemoryManager::UnmapDevPtrHostPtr(uint64_t deviceKey, void* devPtr)
{
    if (deviceKey == 0) {
        HCCL_VM_ERROR("UnmapDevPtrHostPtr with deviceKey=0, devPtr:{:p}", devPtr);
        return;
    }
    std::lock_guard<std::mutex> lock(m_devPtrHostPtrLock);
    m_devPtrHostPtrMap.erase({deviceKey, devPtr});
}

// 获取host地址；deviceKey==0 打 ERROR 并返回 nullptr（调用方走获取/报错兜底，不静默给错地址）。
void* DeviceMemoryManager::GetHostPtrByDevPtr(uint64_t deviceKey, void* devPtr)
{
    if (deviceKey == 0) {
        HCCL_VM_ERROR("GetHostPtrByDevPtr with deviceKey=0, devPtr:{:p}", devPtr);
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(m_devPtrHostPtrLock);
    auto it = m_devPtrHostPtrMap.find({deviceKey, devPtr});
    if (it != m_devPtrHostPtrMap.end()) {
        return it->second;
    }
    return nullptr;
}
} // namespace sim
