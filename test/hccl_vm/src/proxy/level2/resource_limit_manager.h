/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RESOURCE_LIMIT_MANAGER_H
#define RESOURCE_LIMIT_MANAGER_H

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace HcclSim {

/**
 * @brief Level2 设备资源配额统一管理（单例）：收敛 Stream / Notify / UB Jetty 计数型资源
 * 与 AIV 值比较型校验，上限均来自 yaml（sim::runtime::GetDeviceSpec），不写死。计数型语义：
 * 锁内 Add(+count) 后 Check(>cap)，越界回滚返回 false；Release(-count) 带 underflow 保护，
 * 按 deviceKey 独立计数；AIV 仅做请求值 vs vector_core_num 纯值比较。
 */
class ResourceLimitManager {
public:
    static ResourceLimitManager& GetInstance();

    ResourceLimitManager(const ResourceLimitManager&) = delete;
    ResourceLimitManager& operator=(const ResourceLimitManager&) = delete;

    // ── 计数型资源：申请（Add 后 Check，越界回滚返回 false）/ 释放（带 underflow 保护）──
    bool TryAcquireStream(uint64_t deviceKey);
    void ReleaseStream(uint64_t deviceKey);
    // 仅持有资源句柄（DB 行主键）时的释放：内部反查所属 deviceKey 后回减，
    // 把"查表定位设备"的样板收敛在本类，stub 只需一行调用。
    void ReleaseStreamById(uint64_t streamId);

    bool TryAcquireNotify(uint64_t deviceKey);
    void ReleaseNotify(uint64_t deviceKey);

    // UB Jetty（AICPU mode3）合并发放：yaml 存活配额（Add+Check，越界回滚）与 AICPU 硬件编号
    // （TryReserveAicpuJettyHwId，[5312,9407] CAS 单调发号）一步完成；reservedHwId 仅在返回
    // true 时有效。编号耗尽回滚刚占用的存活配额，两种失败统一由调用方对齐 RMA_ENOSPC。
    bool TryAcquireUbJetty(uint64_t deviceKey, uint32_t& reservedHwId);
    void ReleaseUbJetty(uint64_t deviceKey);

    /**
     * @brief AIV Vector Core 纯值校验（无计数）。
     * @param deviceKey              当前设备主键（一般取 g_cur_device_key）。
     * @param requested              请求核数（res limit 的 value 或 kernel 的 numBlocks）。
     * @param rejectIfSpecMissing    true：spec 取不到即拒绝（kernel 下发路径，替驱动硬拦截）；
     *                               false：spec 取不到则放行（SetDeviceResLimit 路径，维持原容错）。
     * @return true 未超限；false 超限（或按参数约定 spec 缺失）。
     */
    bool CheckAivCoreLimit(uint64_t deviceKey, uint32_t requested, bool rejectIfSpecMissing);

private:
    ResourceLimitManager() = default;
    ~ResourceLimitManager() = default;

    // 通用计数型申请：调用方已解析 cap；锁内 Add(count) → Check(used>cap) → 越界回滚。
    bool AcquireLocked(
        std::unordered_map<uint64_t, uint32_t>& counter, uint64_t deviceKey, uint32_t count, uint32_t cap,
        const char* logTag, const char* resName);
    // 通用计数型释放：underflow 保护后 -count。
    void ReleaseLocked(
        std::unordered_map<uint64_t, uint32_t>& counter, uint64_t deviceKey, uint32_t count, const char* logTag,
        const char* resName);

    std::mutex m_mutex;
    std::unordered_map<uint64_t, uint32_t> m_streamUsed;
    std::unordered_map<uint64_t, uint32_t> m_notifyUsed;
    std::unordered_map<uint64_t, uint32_t> m_ubJettyUsed;
    // AICPU jetty 硬件编号局部序号（进程内单调、CAS 有界、耗尽不回绕；语义见
    // sim_capacity_limits.h TryReserveAicpuJettyHwId，原 hccp_stub.cc 静态变量收敛至此）。
    std::atomic<uint32_t> m_aicpuJettyHwIdSeq{0};
};

} // namespace HcclSim

// ───────────────────────────── 一行调用宏（风格对齐 res_stat.h）─────────────────────────────
// 计数型：申请返回 bool（false=超配额已回滚），释放无返回。
#define HCCL_VM_ACQUIRE_STREAM(deviceKey) (::HcclSim::ResourceLimitManager::GetInstance().TryAcquireStream(deviceKey))
#define HCCL_VM_RELEASE_STREAM(deviceKey) (::HcclSim::ResourceLimitManager::GetInstance().ReleaseStream(deviceKey))
// 销毁侧：只给资源行主键，反查与回减全部在 manager 内完成，stub 一行即可。
#define HCCL_VM_RELEASE_STREAM_BY_ID(streamId) \
    (::HcclSim::ResourceLimitManager::GetInstance().ReleaseStreamById(streamId))

#define HCCL_VM_ACQUIRE_NOTIFY(deviceKey) (::HcclSim::ResourceLimitManager::GetInstance().TryAcquireNotify(deviceKey))
#define HCCL_VM_RELEASE_NOTIFY(deviceKey) (::HcclSim::ResourceLimitManager::GetInstance().ReleaseNotify(deviceKey))

// 出参 reservedHwId：AICPU 模式返回 [5312,9407] 硬件编号；返回 false 时无效。
#define HCCL_VM_ACQUIRE_UB_JETTY(deviceKey, reservedHwId) \
    (::HcclSim::ResourceLimitManager::GetInstance().TryAcquireUbJetty((deviceKey), (reservedHwId)))
#define HCCL_VM_RELEASE_UB_JETTY(deviceKey) (::HcclSim::ResourceLimitManager::GetInstance().ReleaseUbJetty(deviceKey))

// AIV：内部取当前设备 g_cur_device_key；两个宏分别对应 kernel 下发（缺 spec 硬拒）
// 与 SetDeviceResLimit（缺 spec 放行）两处语义。
#define HCCL_VM_CHECK_AIV_CORE_CAPACITY(numBlocks)                     \
    (::HcclSim::ResourceLimitManager::GetInstance().CheckAivCoreLimit( \
        sim::runtime::g_cur_device_key, (numBlocks), true))
#define HCCL_VM_CHECK_AIV_RES_LIMIT(value) \
    (::HcclSim::ResourceLimitManager::GetInstance().CheckAivCoreLimit(sim::runtime::g_cur_device_key, (value), false))

#endif // RESOURCE_LIMIT_MANAGER_H
