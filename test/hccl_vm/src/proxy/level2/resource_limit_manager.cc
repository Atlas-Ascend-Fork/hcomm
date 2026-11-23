/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "RES_LIMIT_MGR"

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unistd.h>

#include "sim_log.h"
#include "sim_capacity_limits.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "device_spec/device_spec_types.h"
#include "resource_limit_manager.h"

namespace HcclSim {

ResourceLimitManager& ResourceLimitManager::GetInstance()
{
    // Meyers singleton：C++11 保证局部 static 初始化的线程安全。
    static ResourceLimitManager instance;
    return instance;
}

bool ResourceLimitManager::AcquireLocked(
    std::unordered_map<uint64_t, uint32_t>& counter, uint64_t deviceKey, uint32_t count, uint32_t cap,
    const char* logTag, const char* resName)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    uint32_t& current = counter[deviceKey];
    const uint32_t used = current + count;
    current = used;
    // cap==0 表示该资源不限制；yaml 侧已校验各计数资源上限非 0。
    if (cap > 0 && used > cap) {
        current -= count; // 回滚，保持存活计数不被越界申请污染
        HCCL_VM_ERROR(
            "{} acquire rejected(over limit): name={}, deviceKey={}, tried={}, live={}, cap={}", logTag, resName,
            deviceKey, used, current, cap);
        return false;
    }
    HCCL_VM_INFO(
        "{} acquire +{}: name={}, deviceKey={}, live={}, cap={}", logTag, count, resName, deviceKey, current, cap);
    return true;
}

void ResourceLimitManager::ReleaseLocked(
    std::unordered_map<uint64_t, uint32_t>& counter, uint64_t deviceKey, uint32_t count, const char* logTag,
    const char* resName)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = counter.find(deviceKey);
    if (it == counter.end() || it->second < count) {
        HCCL_VM_ERROR(
            "{} release rejected(underflow): name={}, deviceKey={}, release={}", logTag, resName, deviceKey, count);
        return;
    }
    it->second -= count;
    HCCL_VM_INFO("{} release -{}: name={}, deviceKey={}, live={}", logTag, count, resName, deviceKey, it->second);
}

// ───────────────────────────── Stream ─────────────────────────────
bool ResourceLimitManager::TryAcquireStream(uint64_t deviceKey)
{
    const auto* spec = sim::runtime::GetDeviceSpec(deviceKey);
    if (spec == nullptr) {
        HCCL_VM_ERROR("get device spec failed, deviceKey={}", deviceKey);
        return false;
    }
    return AcquireLocked(m_streamUsed, deviceKey, 1, spec->deviceRes.streamNum, "[stream-account]", "stream");
}

void ResourceLimitManager::ReleaseStream(uint64_t deviceKey)
{
    ReleaseLocked(m_streamUsed, deviceKey, 1, "[stream-account]", "stream");
}

void ResourceLimitManager::ReleaseStreamById(uint64_t streamId)
{
    // Stream 表无 device_id 字段，需经 ctx_id 二级查 Context；默认流创建侧未计数，跳过避免 underflow。
    const auto record = sim::runtime::Db::GetById<sim::runtime::Stream>(streamId);
    if (!record.ok()) {
        return; // 记录不存在（已删或未登记），无计数可回减。
    }
    if (record->is_primary_default != 0) {
        return;
    }
    const auto context = sim::runtime::Db::GetById<sim::runtime::Context>(record->ctx_id);
    if (!context.ok()) {
        HCCL_VM_ERROR(
            "[stream-account] context not found, skip release: streamId={}, ctxId={}", streamId, record->ctx_id);
        return;
    }
    ReleaseStream(context->device_id);
}

// ───────────────────────────── Notify ─────────────────────────────
bool ResourceLimitManager::TryAcquireNotify(uint64_t deviceKey)
{
    const auto* spec = sim::runtime::GetDeviceSpec(deviceKey);
    if (spec == nullptr) {
        HCCL_VM_ERROR("get device spec failed, deviceKey={}", deviceKey);
        return false;
    }
    return AcquireLocked(m_notifyUsed, deviceKey, 1, spec->deviceRes.notifyNum, "[notify-account]", "notify");
}

void ResourceLimitManager::ReleaseNotify(uint64_t deviceKey)
{
    ReleaseLocked(m_notifyUsed, deviceKey, 1, "[notify-account]", "notify");
}

// ─────────── UB Jetty（AICPU mode3）：yaml 存活配额 + [5312,9407] 硬件编号合并发放 ───────────
bool ResourceLimitManager::TryAcquireUbJetty(uint64_t deviceKey, uint32_t& reservedHwId)
{
    const auto* spec = sim::runtime::GetDeviceSpec(deviceKey);
    if (spec == nullptr) {
        HCCL_VM_ERROR("get device spec failed, deviceKey={}", deviceKey);
        return false;
    }
    // 与原 hccp_stub.cc 两段式串联语义一致：先 yaml 存活配额 +1 判定（越界回滚、不消耗编号），
    // 再消耗硬件编号；编号池耗尽须回滚刚占用的存活配额，避免 quota 泄漏。
    if (!AcquireLocked(
            m_ubJettyUsed, deviceKey, 1, spec->deviceRes.ubJetty.normalPoolSize, "[hccp-account]", "ub_jetty")) {
        return false;
    }
    if (!TryReserveAicpuJettyHwId(m_aicpuJettyHwIdSeq, reservedHwId)) {
        // AICPU jetty 4096 硬件编号池耗尽（单调不回绕，见 sim_capacity_limits.h）：回滚前置占用的
        // 存活配额；调用方对齐真机资源耗尽返回 RMA_ENOSPC(228103)。
        ReleaseLocked(m_ubJettyUsed, deviceKey, 1, "[hccp-account]", "ub_jetty");
        HCCL_VM_ERROR(
            "[hccp-account] aicpu jetty hw id exhausted, capacity[{}], owner pid[{}]",
            HCCL_VM_AICPU_JETTY_CAPACITY_PER_RANK, getpid());
        return false;
    }
    return true;
}

void ResourceLimitManager::ReleaseUbJetty(uint64_t deviceKey)
{
    ReleaseLocked(m_ubJettyUsed, deviceKey, 1, "[hccp-account]", "ub_jetty");
}

// ───────────────────────────── AIV Vector Core（纯值校验，无计数）─────────────────────────────
bool ResourceLimitManager::CheckAivCoreLimit(uint64_t deviceKey, uint32_t requested, bool rejectIfSpecMissing)
{
    const auto* spec = sim::runtime::GetDeviceSpec(deviceKey);
    if (spec == nullptr) {
        if (rejectIfSpecMissing) {
            // kernel 下发路径：替闭源驱动做硬拦截，spec 缺失即拒绝。
            HCCL_VM_ERROR(
                "[capacity] aiv_core spec not found: deviceKey={}, requested={}, spec#compute.vector_core_num",
                deviceKey, requested);
            return false;
        }
        // SetDeviceResLimit 路径：取不到当前设备/spec 时不新增阻断，交由调用方既有逻辑处理。
        return true;
    }
    if (requested > spec->vectorCoreNum) {
        HCCL_VM_ERROR(
            "[capacity] aiv_core exhausted: deviceKey={}, requested={}, cap={}, spec={}#compute.vector_core_num",
            deviceKey, requested, spec->vectorCoreNum, spec->sourcePath);
        return false;
    }
    return true;
}

} // namespace HcclSim
