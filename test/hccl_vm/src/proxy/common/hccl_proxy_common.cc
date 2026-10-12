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
#define HCCL_VM_MODULE "PROXY_STUB"

#include <cstdint>
#include "hccl_proxy_common.h"
#include "sim_log.h"
#include "storage/table_access.h"
#include "runtime_state/db_sim_runner_ops.h"
#include <fstream>
#include <nlohmann_json/json.hpp>

namespace sim {
// 三态合同见头文件声明：命中 OK+true / 确定未命中 OK+false / 查询无法完成原样传播。
HcclSim::Storage::DbResult<bool> IsDeviceAddress(void* addr)
{
    uint64_t devPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(addr));
    // 设备地址包含查询:下界 dev_mapped_ptr <= addr、上界派生列 dev_mapped_end_ptr > addr,全部下推。
    auto virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, devPtr),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, devPtr),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV)));
    if (virMemRes.code == HcclSim::Storage::DbCode::NOT_FOUND) {
        // 查询完成且确定未命中：host 地址（NOT_FOUND 是合法业务结果，不是错误）
        return {HcclSim::Storage::DbCode::OK, false, 0, {}};
    }
    if (!virMemRes.ok()) {
        // 查询无法完成：保留原错误码与诊断向上传播（不吞错、不冒充命中/未命中；
        // 失败结果不携带值载荷，避免未检查 ok() 的调用方把 false 误读为 host 地址）
        return {virMemRes.code, std::nullopt, virMemRes.affectedRows, virMemRes.diagnostic};
    }

    return {HcclSim::Storage::DbCode::OK, true, 0, {}};
}

bool ParseKernelJson(const std::string& jsonPath, std::map<std::string, std::string>& out)
{
    // 校验文件后缀是否为.json
    std::string suffix = ".json";
    if (jsonPath.size() < suffix.size() || jsonPath.substr(jsonPath.size() - suffix.size()) != suffix) {
        // 也会加载AIV的.o文件
        HCCL_VM_WARN("path is not json file: {}", jsonPath);
        return true;
    }

    std::ifstream ifs(jsonPath);
    if (!ifs.is_open()) {
        HCCL_VM_ERROR("open failed: {}", jsonPath);
        return false;
    }

    auto j = nlohmann::json::parse(ifs, nullptr, false);
    if (j.is_discarded()) {
        HCCL_VM_ERROR("parse failed: {}", jsonPath);
        return false;
    }

    for (auto& [key, val] : j.items()) {
        if (!val.contains("opInfo") || !val["opInfo"].contains("functionName") || !val["opInfo"].contains("kernelSo")) {
            HCCL_VM_ERROR("skip entry '{}', missing required fields.", key);
            continue;
        }

        std::string funcName = val["opInfo"]["functionName"].get<std::string>();
        std::string kernelSo = val["opInfo"]["kernelSo"].get<std::string>();
        out[funcName] = kernelSo;
    }

    HCCL_VM_INFO("loaded {} kernels from {}", out.size(), jsonPath);
    return true;
}

// Host侧的的转换，地址为当前进程申请根据device_id匹配
uint64_t GetVirPtrByDevPtr(uint64_t devAddr)
{
    uint64_t virAddr = TryGetVirPtrByDevPtr(devAddr);
    if (virAddr == 0) {
        HCCL_VM_ERROR("cannot find virMemRes by devAddr[{}]", devAddr);
    }
    return virAddr;
}

uint64_t TryGetVirPtrByDevPtr(uint64_t devAddr)
{
    uint64_t deviceKey = sim::runtime::GetCurrDeviceKey();
    auto virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Ne(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, uint64_t{0}),
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, devAddr),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, devAddr),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceKey)));
    if (!virMemRes.ok()) {
        return 0;
    }

    uint64_t diff = devAddr - virMemRes.value->dev_mapped_ptr;
    return virMemRes.value->start_ptr + diff;
}

uint64_t TransRemoteAddrToVirtualByRank(uint64_t devAddr, uint32_t rankId)
{
    auto virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, devAddr),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, devAddr),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::rank_id, rankId)));
    if (!virMemRes.ok()) {
        HCCL_VM_ERROR("cannot find virMemRes by devAddr[{}]", devAddr);
        return 0;
    }

    uint64_t diff = devAddr - virMemRes.value->dev_mapped_ptr;
    return virMemRes.value->start_ptr + diff;
}

uint64_t TransRemoteAddrToVirtualByDeviceId(uint64_t devAddr, uint32_t deviceId)
{
    auto virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, devAddr),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, devAddr),
        HcclSim::Storage::Eq(
            &sim::runtime::VirtualMemBlock::src_type, static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV)),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceId)));
    if (!virMemRes.ok()) {
        HCCL_VM_ERROR("cannot find remote VirtualMemBlock by devAddr={}, deviceId={}", devAddr, deviceId);
        return 0;
    }

    return virMemRes.value->start_ptr + (devAddr - virMemRes.value->dev_mapped_ptr);
}

void* GetDevMapperAddrByHostPtr(void* hostPtr)
{
    uint64_t hostAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(hostPtr));
    uint64_t deviceKey = sim::runtime::GetCurrDeviceKey();
    // host 地址包含查询：下界 host_ptr <= addr 下推；上界 host_ptr+size 无派生列，
    // 取回满足下界的结果后在代码侧完成上界过滤（语义与原 GetOneByPred 一致）。
    auto virMemRes = sim::runtime::Db::GetByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::host_ptr, hostAddr),
        HcclSim::Storage::Eq(
            &sim::runtime::VirtualMemBlock::src_type, static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV)),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceKey)));
    if (!virMemRes.ok() || !virMemRes.value.has_value()) {
        HCCL_VM_ERROR("cannot find virMemRes by hostAddr[0x{:x}]", hostAddr);
        return nullptr;
    }
    for (const auto& virMem : *virMemRes.value) {
        if (hostAddr >= virMem.host_ptr + virMem.size) {
            continue;
        }
        uint64_t offset = hostAddr - virMem.host_ptr;
        uint64_t devMappedAddr = reinterpret_cast<uint64_t>(virMem.dev_mapped_ptr) + offset;
        HCCL_VM_INFO("hostAddr=0x{:x}, devMappedAddr=0x{:x}", hostAddr, devMappedAddr);

        return reinterpret_cast<void*>(static_cast<uintptr_t>(devMappedAddr));
    }
    HCCL_VM_ERROR("cannot find virMemRes by hostAddr[0x{:x}]", hostAddr);
    return nullptr;
}

bool IsAICPUExpMode()
{
    const char* expanEnv = std::getenv("HCCL_OP_EXPANSION_MODE");
    return (expanEnv != nullptr) && (std::string(expanEnv) == "AI_CPU");
}

} // namespace sim
