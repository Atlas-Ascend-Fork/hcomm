/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_task_sequential_execute_v2.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <execinfo.h>
#include <limits>
#include <set>
#include <unistd.h>
#include <unordered_set>

#include "sim_common_defs.h"
#include "sim_log.h"
#include "hccl_task_thread.h"
#include "device_resource_manager.h"
#include "ccu_resource_manager.h"
#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/sim_models.h"

using namespace HcclSim;

namespace VirtualRunTime {

namespace {
    // CCU资源基址与proxy侧SetCcuResourceBasicInfo保持一致: die0/die1
    const std::vector<uint64_t> CCU_RESOURCE_BASE_ADDR = {0x123123123, 0x456456456};
} // namespace

// 按DB变更签名增量刷新CCU资源：
// 1. ccuInstr表每次微码加载插入一行、ccuChannels表每次通道配置插入一行，行数作为轻量变更签名；
// 2. 参与CCU的设备集合 = 微码设备 ∪ 通道两端设备，资源按deviceId(Device表主键)索引，
//    多通信域下同一设备的rankId随通信域变化，硬件资源归属物理设备；
// 3. 仅为新设备创建资源(保留已有设备的XN/GSA/CKE/模拟器状态)，通道映射与微码空间整体刷新(幂等)；
HcclVmResult SequentialExecutorV2::EnsureCcuResource()
{
    uint32_t instrLoadCnt = 0;
    uint32_t channelCnt = 0;
    if (sim::operation::QueryCcuResourceMetaCount(instrLoadCnt, channelCnt) != 0) {
        HCCL_VM_ERROR("QueryCcuResourceMetaCount failed");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    if (instrLoadCnt == 0) {
        // 尚无CCU微码下发，无需初始化
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }
    if (ccuResourceInited_ && instrLoadCnt == lastInstrLoadCnt_ && channelCnt == lastChannelCnt_) {
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    std::vector<sim::operation::CcuInstrResTab> instrRes{};
    std::vector<sim::operation::CcuChannelTab> channels{};
    if (loader_.GetInstrResInfo(instrRes) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("GetInstrResInfo failed");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    if (loader_.GetCcuChannelInfo(channels) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("GetCcuChannelInfo failed");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    std::set<uint32_t> deviceIds{};
    uint32_t maxDeviceId = 0;
    for (const auto& instr : instrRes) {
        deviceIds.insert(instr.deviceId);
        maxDeviceId = std::max(maxDeviceId, instr.deviceId);
    }
    for (const auto& channel : channels) {
        deviceIds.insert(channel.srcDeviceId);
        deviceIds.insert(channel.dstDeviceId);
        maxDeviceId = std::max(maxDeviceId, std::max(channel.srcDeviceId, channel.dstDeviceId));
    }
    if (deviceIds.empty()) {
        HCCL_VM_WARN("No CCU device found, skip resource init");
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }
    if (maxDeviceId > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        HCCL_VM_ERROR("CCU device id {} exceeds resource manager index range", maxDeviceId);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 根据设备soc版本确定CCU指令集版本(950:V1, 960:V2)
    RunnerCcuVersion ccuVersion{RunnerCcuVersion::CCU_INVALID};
    for (uint32_t deviceId : deviceIds) {
        auto device = sim::runtime::Db::GetById<sim::runtime::Device>(deviceId);
        if (!device.ok()) {
            HCCL_VM_ERROR("Device {:d} not found in runner db", deviceId);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        RunnerCcuVersion ver{RunnerCcuVersion::CCU_INVALID};
        if (strncmp(device->soc_version, "Ascend950", strlen("Ascend950")) == 0) {
            ver = RunnerCcuVersion::CCU_V1;
        } else if (strncmp(device->soc_version, "Ascend960", strlen("Ascend960")) == 0) {
            ver = RunnerCcuVersion::CCU_V2;
        } else {
            HCCL_VM_ERROR("Unknown soc version: {} for device {:d}", device->soc_version, deviceId);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        if (ccuVersion == RunnerCcuVersion::CCU_INVALID) {
            ccuVersion = ver;
        } else if (ccuVersion != ver) {
            HCCL_VM_ERROR(
                "Mixed ccu version: {} and {:s}", RunnerCcuVersionToString(ccuVersion), RunnerCcuVersionToString(ver));
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
    }

    auto& ccuResMgr = CcuResourceManager::GetInstance();
    const uint32_t capacity = maxDeviceId + 1;

    // 仅为新设备创建资源；Init内部按需扩容到capacity并以deviceId为索引
    for (uint32_t deviceId : deviceIds) {
        if (ccuResMgr.HasResource(static_cast<int>(deviceId))) {
            continue;
        }
        ccuResMgr.Init(static_cast<int>(deviceId), capacity, ccuVersion, CCU_RESOURCE_BASE_ADDR);
    }

    // 通道映射刷新: channelId → 远端CcuInfo(远端deviceId, dieId)，按源设备分组
    std::map<uint32_t, RankChannelInfo> channelMapByDevice{};
    for (const auto& channel : channels) {
        if (channel.srcDieId >= HcclSim::DIE_NUM || channel.dstDieId >= HcclSim::DIE_NUM
            || channel.channelId >= SimCcuV1::MAX_CCU_CHANNEL_NUM) {
            HCCL_VM_ERROR(
                "Invalid channel info: srcDevice={:d}, srcDie={:d}, dstDevice={:d}, dstDie={:d}, channelId={:d}",
                channel.srcDeviceId, channel.srcDieId, channel.dstDeviceId, channel.dstDieId, channel.channelId);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        CcuInfo rmtInfo;
        rmtInfo.rankId = static_cast<int>(channel.dstDeviceId);
        rmtInfo.dieId = static_cast<int>(channel.dstDieId);
        channelMapByDevice[channel.srcDeviceId][channel.srcDieId][channel.channelId] = rmtInfo;
    }
    for (auto& entry : channelMapByDevice) {
        ccuResMgr.InitChannelInfo(static_cast<int>(entry.first), entry.second);
    }

    // 微码指令空间刷新: ccuInstrRes按(deviceId, dieId)一行，整体覆盖该die的指令空间
    for (const auto& instr : instrRes) {
        if (instr.dieId >= HcclSim::DIE_NUM) {
            HCCL_VM_ERROR("Invalid instr res dieId: device={:d}, die={:d}", instr.deviceId, instr.dieId);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        if (instr.instrCount > HcclSim::CCU_INSTRUCTION_NUM) {
            HCCL_VM_ERROR(
                "Invalid instr count: device={:d}, die={:d}, count={:d}, capacity={:d}", instr.deviceId, instr.dieId,
                instr.instrCount, HcclSim::CCU_INSTRUCTION_NUM);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        CcuInstrData ccuInstr;
        ccuInstr.instrCnt = static_cast<uint16_t>(instr.instrCount);
        ccuInstr.instrData.reserve(instr.instrCount);
        for (uint32_t i = 0; i < instr.instrCount; ++i) {
            hcomm::CcuRep::CcuInstr tmp{};
            std::memcpy(&tmp, instr.instrSpace[i], sizeof(hcomm::CcuRep::CcuInstr));
            ccuInstr.instrData.push_back(tmp);
        }
        ccuResMgr.InitInstrInfo(static_cast<int>(instr.deviceId), instr.dieId, ccuInstr);
    }

    lastInstrLoadCnt_ = instrLoadCnt;
    lastChannelCnt_ = channelCnt;
    ccuResourceInited_ = true;
    HCCL_VM_INFO(
        "CCU resources refreshed: version={}, devices={}, channels={}, instrRes={}",
        RunnerCcuVersionToString(ccuVersion), deviceIds.size(), channels.size(), instrRes.size());
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult SequentialExecutorV2::Execute()
{
    HcclVmResult ccuRet = EnsureCcuResource(); // todo 放在构造函数?
    if (ccuRet != HcclVmResult::HCCL_SIM_SUCCESS) {
        return ccuRet;
    }

    std::vector<sim::operation::OpTaskTab> tasks{};
    if (loader_.LoadAllOpTasks(tasks, true) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("LoadAllOpTasks failed");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    std::vector<sim::operation::OpTaskTab> streamHeadTasks;
    std::unordered_set<uint64_t> seenStreams;
    for (const auto& task : tasks) {
        if (!seenStreams.insert(task.streamId).second) {
            continue;
        }
        streamHeadTasks.emplace_back(task);
    }
    HCCL_VM_DEBUG("current task num {}, head task num {}", tasks.size(), streamHeadTasks.size());

    for (const auto& task : streamHeadTasks) {
        HcclTaskMetaData taskMeta;
        std::memcpy(&taskMeta, task.optaskMeta.data(), sizeof(HcclTaskMetaData));
        HCCL_VM_DEBUG(
            "Executing Task: id={}, pid={}, opDetailId={}, taskMeta=[{:s}]", task.id, task.pid, task.opDetailId,
            taskMeta.Describe());
        auto ret = ExecuteOneTask(taskMeta);

        if (ret == HcclVmResult::HCCL_SIM_VRT_HOLD_CMD) {
            // 无法完整执行当前Task，让出Executor，继续执行其他Task（可恢复，保留未完成态）
            HCCL_VM_DEBUG("Hold Task: id={}, pid={}", task.id, task.pid);
            continue;
        }

        // 与宿主侧的契约：任务必须到达终态（isDone），宿主侧 aclrtSynchronizeStream 才可能
        // 等到全部完成。硬失败时若把任务留在"未完成"态返回，则剩余任务永不变 isDone，
        // 宿主会无限打印 "Waiting for stream... N/M" 死等。故此处对硬失败也刷新 isDone（终态），
        // 并继续处理其它任务；失败本身由 ERROR 日志与最终数据校验体现。
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR(
                "Task Execute failed, ret={}, mark done to unblock waiters, taskMeta=[{:s}]", static_cast<int>(ret),
                taskMeta.Describe());
        }
        if (loader_.FinishOpTask(task) != HcclResult::HCCL_SUCCESS) {
            HCCL_VM_ERROR("FinishOpTask failed");
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        if (ret == HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_INFO("Task Finished: id={}, pid={}", task.id, task.pid);
        }
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult SequentialExecutorV2::ExecuteOneTask(HcclTaskMetaData& task)
{
    switch (task.taskType) {
        case HccLTaskMetaType::NOTIFY_WAIT:
            return TaskNotifyWait(task);
        case HccLTaskMetaType::NOTIFY_RECORD:
            return TaskNotifyRecord(task);
        case HccLTaskMetaType::REDUCE:
            return TaskReduce(task);
        case HccLTaskMetaType::MEM_CPY:
            return TaskMemcpy(task);
        case HccLTaskMetaType::CCU_GRAPH:
            return TaskCcuGraph(task);
        case HccLTaskMetaType::AIV_GRAPH:
            return TaskAivGraph(task);
        case HccLTaskMetaType::SYNC_STREAM:
            return HcclVmResult::HCCL_SIM_SUCCESS;
        case HccLTaskMetaType::MODEL_EXEC:
            return HcclVmResult::HCCL_SIM_SUCCESS;
        default:
            HCCL_VM_ERROR("Unsupported taskType: {:s}", task.TaskTypeName());
            break;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult SequentialExecutorV2::TaskAivGraph(const HcclTaskMetaData& task)
{
    const uint64_t deviceId = task.deviceId;
    const uint64_t streamId = task.streamId;
    const uint64_t launchIndex = task.taskData.aiv.launchIdx;
    auto aivGraphExecutor = GetAivGraphExecutor(deviceId, launchIndex);
    if (!aivGraphExecutor->Init()) {
        HCCL_VM_ERROR(
            "Failed to init AivGraphExecutor, deviceId={}, streamId={}, launchIndex={}", deviceId, streamId,
            launchIndex);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    auto ret = aivGraphExecutor->Execute();
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS && ret != HcclVmResult::HCCL_SIM_VRT_HOLD_CMD) {
        HCCL_VM_ERROR(
            "AivGraphExecutor failed, deviceId={}, streamId={}, launchIndex={}, ret={}", deviceId, streamId,
            launchIndex, static_cast<int>(ret));
    }
    if (ret == HcclVmResult::HCCL_SIM_SUCCESS) {
        RemoveAivGraphExecutor(deviceId, launchIndex);
    }
    return ret;
}

std::shared_ptr<AivGraphExecutor> SequentialExecutorV2::GetAivGraphExecutor(uint64_t deviceId, uint64_t launchIdx)
{
    const auto key = std::make_pair(deviceId, launchIdx);
    auto iter = aivExecutorMap_.find(key);
    if (iter == aivExecutorMap_.end()) {
        iter = aivExecutorMap_.emplace(key, std::make_shared<AivGraphExecutor>(deviceId, launchIdx)).first;
    }
    return iter->second;
}

void SequentialExecutorV2::RemoveAivGraphExecutor(uint64_t deviceId, uint64_t launchIdx)
{
    const auto key = std::make_pair(deviceId, launchIdx);
    auto iter = aivExecutorMap_.find(key);
    if (iter != aivExecutorMap_.end()) {
        aivExecutorMap_.erase(iter);
    }
}

} // namespace VirtualRunTime
