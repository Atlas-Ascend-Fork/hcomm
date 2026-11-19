/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_base_resolver.h"

#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "device_spec/ccu_address_space.h"
#include "device_spec/device_spec_registry.h"
#include "device_spec/device_spec_types.h"
#include "device_spec/rank_address_space.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/table_access.h"
#include "sim_log.h"

namespace HcclSim {

std::map<uint32_t, std::vector<uint64_t>> ResolveAllCcuBaseAddrs()
{
    std::map<uint32_t, std::vector<uint64_t>> result;
    const auto devices
        = sim::runtime::Db::GetByPred<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>());
    if (!devices.ok() || !devices.value.has_value()) {
        HCCL_VM_ERROR("ResolveAllCcuBaseAddrs: cannot read device table; ccu base addr table stays empty");
        return result;
    }

    // Ccu 表整表一次读入并建 (device_id, die_id) → resource_addr 索引。
    // ★ 不再逐 (device, die) 调 GetCcuResourceBaseAddr：本函数在 checker 启动时无条件执行
    //   （main.cc 的 InitCcuInfo，早于任务加载，无 CCU 任务也会跑），逐卡逐 die 查询会让
    //   读库次数随卡数线性增长（双 die：128 卡 385 次、1024 卡 3073 次），而全部进程共用
    //   同一个 hccl_sim.db。改为 2 次全表扫后与卡数无关。
    std::map<std::pair<uint64_t, uint32_t>, uint64_t> ccuAddrById;
    const auto ccus = sim::runtime::Db::GetByPred<sim::runtime::Ccu>(HcclSim::Storage::All<sim::runtime::Ccu>());
    if (ccus.ok() && ccus.value.has_value()) {
        for (const auto& ccu : *ccus.value) {
            if (ccu.resource_addr != 0) {
                ccuAddrById[{ccu.device_id, static_cast<uint32_t>(ccu.die_id)}] = ccu.resource_addr;
            }
        }
    } else {
        HCCL_VM_WARN("ResolveAllCcuBaseAddrs: cannot read ccu table, fall back to device spec for all devices");
    }

    // 规格来自 DeviceSpecRegistry（yaml，进程级常驻只读）：Device 行已在上手，
    // 直接按 soc_version 匹配，不再逐卡回查 Device 表。
    auto& reg = sim::devspec::DeviceSpecRegistry::GetInstance();
    if (reg.Size() == 0) {
        (void)reg.LoadAll(); // 只读 config/device/*.yaml，不依赖 DB
    }

    for (const auto& device : *devices.value) {
        const std::string socVer(device.soc_version);
        const auto* spec = socVer.empty() ? nullptr : reg.FindBySocVersion(socVer);
        if (spec == nullptr || spec->dieNum == 0) {
            HCCL_VM_ERROR(
                "ResolveAllCcuBaseAddrs: device spec/die_num unavailable, deviceId={}, soc_version=\"{}\"（{}）",
                device.id, socVer, reg.LastError());
            continue;
        }
        // yaml 回退路径（DB 无 Ccu 行或 resource_addr 为 0，如 mock-comm，§4.8.7 / O9）
        // 与 GetCcuResourceBaseAddr 同源：ccuDieBase(rankIdx, dieId)。
        uint32_t rankIdx = 0;
        const bool rankOk = sim::devspec::RankAddressSpace::RankIdxOfDeviceKey(device.id, rankIdx);
        const sim::devspec::RankAddressSpace ras(spec->addr);

        std::vector<uint64_t> bases(spec->dieNum, 0);
        for (uint32_t dieId = 0; dieId < spec->dieNum; ++dieId) {
            uint64_t base = 0;
            const auto it = ccuAddrById.find({device.id, dieId});
            if (it != ccuAddrById.end()) {
                base = it->second;
            } else if (rankOk) {
                base = ras.CcuDieBase(rankIdx, dieId);
            }
            if (base == 0) {
                HCCL_VM_ERROR(
                    "ResolveAllCcuBaseAddrs: resolve ccu base addr failed, deviceId={} dieId={}（写入 "
                    "0，不回退假地址）",
                    device.id, dieId);
                base = 0;
            }
            bases[dieId] = base;
        }
        result[static_cast<uint32_t>(device.id)] = std::move(bases);
    }
    HCCL_VM_INFO("ResolveAllCcuBaseAddrs: ccu base addr table filled, deviceCount={}", result.size());
    return result;
}

const sim::devspec::CcuAddressSpace* GetCcuAddrSpaceByDeviceId(uint32_t deviceId)
{
    // 进程级缓存：DeviceSpec 由 DeviceSpecRegistry 常驻（加载后只读），CcuAddressSpace
    // 仅持其引用，可安全长期缓存。checker 多线程翻译时经 mutex 保护懒加载。
    static std::mutex cacheMutex;
    static std::map<uint32_t, std::unique_ptr<sim::devspec::CcuAddressSpace>> cache;
    std::lock_guard<std::mutex> lock(cacheMutex);
    const auto cached = cache.find(deviceId);
    if (cached != cache.end()) {
        return cached->second.get();
    }
    const auto* spec = sim::runtime::GetDeviceSpec(static_cast<uint64_t>(deviceId));
    if (spec == nullptr) {
        HCCL_VM_ERROR("GetCcuAddrSpaceByDeviceId: resolve device spec failed, deviceId={}", deviceId);
        return nullptr;
    }
    auto space = std::make_unique<sim::devspec::CcuAddressSpace>(*spec);
    const auto* raw = space.get();
    cache.emplace(deviceId, std::move(space));
    return raw;
}

std::vector<DevMappedVirRange> ResolveAllDevMappedRanges()
{
    std::vector<DevMappedVirRange> ranges;
    const auto rows = sim::runtime::Db::GetByPred<sim::runtime::VirtualMemBlock>(
        HcclSim::Storage::All<sim::runtime::VirtualMemBlock>());
    if (!rows.ok() || !rows.value.has_value()) {
        HCCL_VM_WARN("ResolveAllDevMappedRanges: cannot read VirtualMemBlock, dev-mapped translation disabled");
        return ranges;
    }

    ranges.reserve(rows.value->size());
    for (const auto& row : *rows.value) {
        if (row.src_type != static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV) || row.is_dev_access == 0
            || row.dev_mapped_ptr == 0 || row.start_ptr == 0 || row.size == 0) {
            continue;
        }
        // is_freed 行保留：释放路径只打标不删行，迟到解析仍需命中（见 VirtualMemBlock.is_freed 注释）。
        ranges.push_back(DevMappedVirRange{row.dev_mapped_ptr, row.start_ptr, row.size, row.device_id});
    }
    HCCL_VM_INFO("ResolveAllDevMappedRanges: loaded {} dev-mapped ranges", ranges.size());
    return ranges;
}

} // namespace HcclSim
