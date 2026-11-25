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
#include "sim_common_defs.h"
#include "storage/table_access.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include <fstream>
#include <nlohmann_json/json.hpp>

namespace sim {
// 三态合同见头文件声明：命中 OK+true / 确定未命中 OK+false / 查询无法完成原样传播。
// P5：分类只需要存在性——改为字段投影（id）+ LIMIT 1，取消无业务必要的首行
// 排序（旧行为 GetOneByPred 固定 id ASC 取首行，行选择对存在性判定无意义）。
// 条件保持不变：src_type 等值 + dev_mapped 区间包含，不加 is_freed 谓词——
// 释放过的设备地址仍按设备地址分类（分类未命中≠host，不得据此走裸 memcpy）。
HcclSim::Storage::DbResult<bool> IsDeviceAddress(void* addr)
{
    uint64_t devPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(addr));
    // 设备地址包含查询:下界 dev_mapped_ptr <= addr、上界派生列 dev_mapped_end_ptr > addr,全部下推。
    const auto virMemRes = HcclSim::Storage::Select<sim::runtime::VirtualMemBlock>(
        HcclSim::Storage::And(
            HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, devPtr),
            HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, devPtr),
            HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV)),
        HcclSim::Storage::Fields(&sim::runtime::VirtualMemBlock::id), HcclSim::Storage::Limit(1U));
    if (!virMemRes.ok()) {
        // 查询无法完成：保留原错误码与诊断向上传播（不吞错、不冒充命中/未命中；
        // 失败结果不携带值载荷，避免未检查 ok() 的调用方把 false 误读为 host 地址）
        return {virMemRes.code, std::nullopt, virMemRes.affectedRows, virMemRes.diagnostic};
    }
    if (!virMemRes.value.has_value()) {
        // 合同外状态（成功结果必带值载荷）：按查询无法完成传播，不冒充命中/未命中
        return {HcclSim::Storage::DbCode::INVALID_STATE, std::nullopt, 0, "select returned ok without payload"};
    }
    // 查询完成：命中行存在 = 设备地址；空结果（含旧行为的 NOT_FOUND 业务语义）
    // = 确定不是设备地址。两类"无行"与后端错误不混淆。
    return {HcclSim::Storage::DbCode::OK, !virMemRes.value->empty(), 0, {}};
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
    // 翻译族选择规则（合同见 hccl_device_pub.cc FindDevMappedVirMemByDeviceNormalAndCheckOnly）：
    // 任务发射期（RDMA/D2D/算子参数偏移）按同步合同引用存活缓冲，活动行优先，
    // 同址复用后新任务命中新分配；活动行未命中（NOT_FOUND）才回退原始无谓词查询，
    // 保留迟到解析与取消 purge 前 free→下次分配窗口的原有行为；引擎错误不吞。
    auto virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Ne(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, uint64_t{0}),
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, devAddr),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, devAddr),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceKey),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::is_freed, uint64_t{0})));
    if (!virMemRes.ok() && virMemRes.code == HcclSim::Storage::DbCode::NOT_FOUND) {
        virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
            HcclSim::Storage::Ne(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, uint64_t{0}),
            HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, devAddr),
            HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, devAddr),
            HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
            HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceKey)));
    }
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
    // 翻译族选择规则（合同见 hccl_device_pub.cc FindDevMappedVirMemByDeviceNormalAndCheckOnly）：
    // RDMA/远端任务发射按同步合同引用存活缓冲，活动行优先；活动行未命中
    // （NOT_FOUND）才回退原始无谓词查询，保留迟到解析与既有窗口行为；引擎错误不吞。
    auto virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, devAddr),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, devAddr),
        HcclSim::Storage::Eq(
            &sim::runtime::VirtualMemBlock::src_type, static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV)),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceId),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::is_freed, uint64_t{0})));
    if (!virMemRes.ok() && virMemRes.code == HcclSim::Storage::DbCode::NOT_FOUND) {
        virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
            HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, devAddr),
            HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, devAddr),
            HcclSim::Storage::Eq(
                &sim::runtime::VirtualMemBlock::src_type, static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV)),
            HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceId)));
    }
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

// 优先取线程上下文 g_cur_comm_key 填充 commId 并解析 rankId;
// 未设置(如 worker 线程为 0)时, 回退按 deviceId 取首个 Communicator 成员行补齐.
void FillTaskCommInfo(HcclTaskMetaData& task, uint32_t deviceId)
{
    task.commId = sim::runtime::g_cur_comm_key;
    if (task.commId != 0) {
        if (!sim::runtime::GetCommRankByDeviceId(task.commId, deviceId, task.rankId)) {
            HCCL_VM_ERROR("dpu task cannot resolve rankId, commId={}, deviceId={}", task.commId, deviceId);
        }
        return;
    }

    auto members = sim::runtime::Db::GetByPred<sim::runtime::Communicator>(
        HcclSim::Storage::Eq(&sim::runtime::Communicator::device_id, deviceId));
    if (!members.ok() || !members.value.has_value() || members->empty()) {
        HCCL_VM_ERROR("dpu task has no communicator, deviceId={}", deviceId);
        return;
    }
    task.commId = members->front().id;
    task.rankId = members->front().rank_id;
}
