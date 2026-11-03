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
#include <mutex>
#include <unordered_map>

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

// UB 多 channel（jetty）数的默认值：1 表示一个 channel 一个 jetty（不使能多 jetty），
// 与 hccl::GetEnvUbConfig().GetUbMultiChannelNum() 的默认口径一致。
constexpr uint32_t HCCL_VM_UB_MULTI_CHANNEL_NUM_DEFAULT = 1;

// 解析 HCCL_UB_MULTI_CHANNEL_NUM（一个 channel 上并行的 jetty 数）。
// 未设置/空串/非十进制/带多余字符/0/溢出 uint32 一律视为非法：out 置默认值并返回
// false，是否告警由调用方决定（本头不引入日志依赖，口径同上方 GetMaxRanks）。
inline bool ParseUbMultiChannelNum(uint32_t& out)
{
    out = HCCL_VM_UB_MULTI_CHANNEL_NUM_DEFAULT;
    const char* env = std::getenv("HCCL_UB_MULTI_CHANNEL_NUM");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    char* end = nullptr;
    const unsigned long v = std::strtoul(env, &end, 10);
    if (end == env || v == 0 || v > UINT32_MAX) {
        return false;
    }
    while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r') {
        ++end;
    }
    if (*end != '\0') {
        return false;
    }
    out = static_cast<uint32_t>(v);
    return true;
}

// HCCL_UB_MULTI_CHANNEL_NUM 的取值（未设置/非法回落默认值 1）。
// 不做进程内缓存：调用点在算子落库与通信域配置查询路径（非热路径），
// 且免缓存便于 UT 直接注入环境变量。
inline uint32_t GetUbMultiChannelNum()
{
    uint32_t channelNum = HCCL_VM_UB_MULTI_CHANNEL_NUM_DEFAULT;
    (void)ParseUbMultiChannelNum(channelNum);
    return channelNum;
}

// 等待远端记录（对端通信域 / HcclBuffer / HcclMem / 反向 channel / CCU 交换值）
// 就绪的默认总超时（毫秒）：与 level1 各等待桩改造前的 10s 口径一致。
// 千卡规模下对端进程就绪时延显著变大，10s 预算不足会误超时，可经 HCCL 官方建链超时
// 环境变量 HCCL_CONNECT_TIMEOUT（单位秒，取值范围 [120, 7200]，默认 120s）按规模调大，
// 与现网部署文档（mariadb README 1024p 推荐配置）使用同一变量，按秒解析后换算为毫秒。
constexpr uint32_t HCCL_VM_CONNECT_TIMEOUT_DEFAULT_MS = 10000;

/**
 * @brief 解析 HCCL_CONNECT_TIMEOUT（等待远端记录就绪的总超时，输出毫秒）。
 *
 * HCCL_CONNECT_TIMEOUT 为 HCCL 官方建链超时环境变量，单位秒；本函数按秒解析后
 * 乘以 1000 换算为毫秒，乘后溢出 uint32（秒值 > UINT32_MAX/1000）视为非法。
 *
 * @param[out] out 输出：生效超时（毫秒）；未设置或非法时置 HCCL_VM_CONNECT_TIMEOUT_DEFAULT_MS。
 * @return true=取到合法配置值；false=未设置/空串/非十进制/带多余字符/0/溢出 uint32，
 *         是否告警由调用方决定（本头不引入日志依赖，口径同 ParseUbMultiChannelNum）。
 */
inline bool ParseConnectTimeoutMs(uint32_t& out)
{
    out = HCCL_VM_CONNECT_TIMEOUT_DEFAULT_MS;
    const char* env = std::getenv("HCCL_CONNECT_TIMEOUT");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    char* end = nullptr;
    const unsigned long v = std::strtoul(env, &end, 10);
    if (end == env || v == 0 || v > UINT32_MAX / 1000) {
        return false;
    }
    while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r') {
        ++end;
    }
    if (*end != '\0') {
        return false;
    }
    out = static_cast<uint32_t>(v) * 1000U;
    return true;
}

// HCCL_CONNECT_TIMEOUT 的取值：未设置/非法回落默认 10000ms；合法值小于默认下限（10s）时
// 提升到 10s——等待窗口只增不减，避免误配置把窗口压小导致千卡场景误超时。
// 不做进程内缓存：调用点在通信域/channel 初始化路径（非热路径），且免缓存便于 UT 直接注入环境变量。
inline uint32_t GetConnectTimeoutMs()
{
    uint32_t timeoutMs = HCCL_VM_CONNECT_TIMEOUT_DEFAULT_MS;
    (void)ParseConnectTimeoutMs(timeoutMs);
    return timeoutMs < HCCL_VM_CONNECT_TIMEOUT_DEFAULT_MS ? HCCL_VM_CONNECT_TIMEOUT_DEFAULT_MS : timeoutMs;
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

// 事件级 notify 任务 id：按方向 (srcRank, dstRank) 的第 k 次 notify 事件编码。
// 跨 rank 的 record/wait 分别在各自进程中按同一方向独立计数，第 k 个天然配对；
// 该编码不依赖反向 channel / 槽位上的 Notify 行解析结果，避免两侧解析不一致导致错配。
// 高位 tag(0xD) 区分事件 id 空间，避免与 MakeNotifyTaskId(rank<<17|id) 冲突。
inline uint64_t MakeNotifyEventTaskId(uint32_t srcRank, uint32_t dstRank)
{
    static std::mutex eventMutex;
    static std::unordered_map<uint64_t, uint32_t> eventSeq;
    const uint64_t key = (static_cast<uint64_t>(srcRank) << 32) | static_cast<uint64_t>(dstRank);
    uint32_t seq = 0;
    {
        std::lock_guard<std::mutex> lock(eventMutex);
        seq = eventSeq[key]++;
    }
    return 0xD000000000000000ULL | (static_cast<uint64_t>(srcRank & 0xFFFFU) << 40)
           | (static_cast<uint64_t>(dstRank & 0xFFFFU) << 24) | static_cast<uint64_t>(seq & 0xFFFFFFU);
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
