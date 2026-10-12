/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_CAPACITY_LIMITS_H
#define SIM_CAPACITY_LIMITS_H

#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace HcclSim {
constexpr uint32_t HCCL_VM_MAX_RANKS = 1024;
// 每 RaContext 的通道 jetty 数上界 = 超节点内跨服务器对端数（每对端一条
// mode=USER_CTL_NORMAL jetty），随拓扑取 (服务器数×卡数−本服务器卡数)：
// 116_8/2_16_8=120、2_32_8=248、1_128_8=1016。上界不超过板卡规模上限
// （MAX_RANKS−1），故直接派生自 HCCL_VM_MAX_RANKS，不独立设数。
// 注意：该常量进入 ChannelData 定长数组与 dump 文件步长（sizeof 烤入格式），
// 调整即文件格式变更，全部读方（含仓外工具）须以新头重编。
constexpr uint32_t HCCL_VM_MAX_CHANNEL_JETTIES = HCCL_VM_MAX_RANKS;

inline uint32_t GetMaxRanks()
{
    static const uint32_t cached = []() {
        const char* env = std::getenv("HCCL_VM_MAX_RANKS");
        if (env == nullptr || env[0] == '\0') {
            return HCCL_VM_MAX_RANKS;
        }
        char* end = nullptr;
        unsigned long v = std::strtoul(env, &end, 10);
        if (end == env || v == 0 || v > UINT32_MAX) {
            return HCCL_VM_MAX_RANKS;
        }
        return static_cast<uint32_t>(v);
    }();
    return cached;
}

constexpr uint32_t HCCL_VM_DEVICE_NOTIFY_ID_BITS = 17;
constexpr uint32_t HCCL_VM_DEVICE_NOTIFY_ID_CAPACITY = 1U << HCCL_VM_DEVICE_NOTIFY_ID_BITS;
constexpr uint32_t HCCL_VM_DEVICE_NOTIFY_ID_MASK = HCCL_VM_DEVICE_NOTIFY_ID_CAPACITY - 1;
constexpr uint64_t HCCL_VM_INVALID_NOTIFY_TASK_ID = UINT64_MAX;

inline uint64_t MakeNotifyTaskId(uint32_t rankId, uint32_t deviceNotifyId)
{
    if (rankId >= GetMaxRanks() || deviceNotifyId >= HCCL_VM_DEVICE_NOTIFY_ID_CAPACITY) {
        return HCCL_VM_INVALID_NOTIFY_TASK_ID;
    }
    return (static_cast<uint64_t>(rankId) << HCCL_VM_DEVICE_NOTIFY_ID_BITS) | deviceNotifyId;
}

constexpr uint32_t GetDeviceNotifyId(uint64_t notifyTaskId)
{
    return static_cast<uint32_t>(notifyTaskId) & HCCL_VM_DEVICE_NOTIFY_ID_MASK;
}

inline bool IsValidNotifyTaskId(uint64_t notifyTaskId)
{
    return notifyTaskId != HCCL_VM_INVALID_NOTIFY_TASK_ID
           && (notifyTaskId >> HCCL_VM_DEVICE_NOTIFY_ID_BITS) < GetMaxRanks();
}

constexpr uint32_t GetNotifyTaskRankId(uint64_t notifyTaskId)
{
    return static_cast<uint32_t>(notifyTaskId >> HCCL_VM_DEVICE_NOTIFY_ID_BITS);
}

// ==================== AICPU Jetty 编号合同 ====================
// 硬件合同（hccp_ctx.h JettyMode）：各模式 jetty 编号空间按硬件分配、天然不相交——
//   URMA_NORMAL(0) ∈ [0,1023]；CCU(2/4) ∈ [1024,1151]；CACHE_LOCK_DWQE(1) ∈ [1216,5311]；
//   USER_CTL_NORMAL(3) ∈ [5312,9407]（每设备 4096 个，AICPU 数据面 SQ jetty）。
// 模拟器对 USER_CTL_NORMAL 模式按 rank 进程内分配真实硬件编号（其余模式沿用 caller
// 下发的编号）。UBDMA SQE 的 jettyId1 字段为 16 位，可编码全部模式范围；数据库行主键
// 无界增长、禁止作为编号下发（256p 实测行主键 65536~65792 经 16 位字段回绕为 0~256，
// 导致数据面 SQE 整批丢弃，见 redis256p-b4）。
constexpr uint32_t HCCL_VM_AICPU_JETTY_ID_BITS = 16;
constexpr uint32_t HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN = 5312;
constexpr uint32_t HCCL_VM_AICPU_USER_CTL_JETTY_ID_END = 9407;
constexpr uint32_t HCCL_VM_AICPU_JETTY_CAPACITY_PER_RANK
    = HCCL_VM_AICPU_USER_CTL_JETTY_ID_END - HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN + 1;
// RaJetty.mode 字段值（对齐 hccp_ctx.h JettyMode::JETTY_MODE_USER_CTL_NORMAL）
constexpr uint32_t HCCL_VM_JETTY_MODE_USER_CTL_NORMAL = 3;

// 编号合法性校验：入参在窄化转换前使用（uint64_t 直接判界，不做先转后验）。
constexpr bool IsAicpuJettyHwIdInRange(uint64_t jettyId)
{
    return jettyId >= HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN && jettyId <= HCCL_VM_AICPU_USER_CTL_JETTY_ID_END;
}

// 进程内单调预留（CAS 有界）：编号 = BEGIN + 局部序号，进程生命周期内不复用、不在
// 通信域销毁时清零（旧 SQE 可能仍引用原编号）；耗尽（4096）后不递增、不回绕，连续
// 调用稳定失败。创建失败的编号视为已消耗（不回收，避免编号复活后被新行复用）。
inline bool TryReserveAicpuJettyHwId(std::atomic<uint32_t>& localSeq, uint32_t& reservedHwId)
{
    uint32_t expected = localSeq.load(std::memory_order_relaxed);
    while (true) {
        if (expected >= HCCL_VM_AICPU_JETTY_CAPACITY_PER_RANK) {
            return false; // 耗尽：计数停留在上限，不再增长
        }
        if (localSeq.compare_exchange_weak(expected, expected + 1, std::memory_order_relaxed)) {
            reservedHwId = HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN + expected;
            return true;
        }
        // CAS 失败时 expected 已被刷新为最新值，重试
    }
}

static_assert(
    HCCL_VM_AICPU_USER_CTL_JETTY_ID_END < (1U << HCCL_VM_AICPU_JETTY_ID_BITS),
    "AICPU Jetty hardware IDs must fit the 16-bit SQE jettyId1 field");
static_assert(
    HCCL_VM_AICPU_JETTY_CAPACITY_PER_RANK >= HCCL_VM_MAX_RANKS + 1,
    "per-rank AICPU Jetty capacity must cover a full mesh plus the local jetty");
} // namespace HcclSim

#endif
