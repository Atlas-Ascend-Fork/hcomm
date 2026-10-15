/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_graph_executor.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "ai_core_stub.h"
#include "aiv_task_snapshot_loader.h"
#include "hccl_types.h"
#include "sim_log.h"
#include "sim_common_defs.h"
#include "store_sim_store_pub.h"

using HcclSim::HcclVmResult;

static void AppendPipeTasksToQueue(
    const std::vector<std::shared_ptr<AivSim::AivTask>>& pipeTasks,
    std::queue<std::shared_ptr<AivSim::AivTask>>& taskQueue, uint32_t& maxTaskId, uint32_t& maxEventId,
    uint32_t& maxSyncRound)
{
    for (const auto& task : pipeTasks) {
        maxTaskId = std::max(maxTaskId, task->GetTaskId());
        if (task->GetTaskType() == AivSim::AivTaskType::SET_FLAG) {
            auto setFlagTask = std::dynamic_pointer_cast<AivSim::AivTaskSetFlag>(task);
            maxEventId = std::max(maxEventId, static_cast<uint32_t>(setFlagTask->GetEventId()));
        }
        if (task->GetTaskType() == AivSim::AivTaskType::WAIT_FLAG) {
            auto waitFlagTask = std::dynamic_pointer_cast<AivSim::AivTaskWaitFlag>(task);
            maxEventId = std::max(maxEventId, static_cast<uint32_t>(waitFlagTask->GetEventId()));
        }
        if (task->GetTaskType() == AivSim::AivTaskType::SYNC_ALL) {
            auto syncAllTask = std::dynamic_pointer_cast<AivSim::AivTaskSyncAll>(task);
            maxSyncRound = std::max(maxSyncRound, syncAllTask->GetSyncRound());
        }
        taskQueue.push(task);
    }
}

AivBlock::AivBlock(uint32_t blockIdx, size_t maxEventId) : blockIdx_(blockIdx)
{
    for (size_t i = 0; i <= maxEventId; ++i) {
        events_.push_back(false);
    }
}

bool AivGraphExecutor::Init()
{
    if (isInitialized_) {
        return true;
    }

    AivRuntimeTaskSnapshot taskSnapshot;
    std::string errorMessage;
    if (!AivTaskSnapshotLoader::LoadRuntimeTasks(deviceId_, launchIdx_, taskSnapshot, &errorMessage)) {
        HCCL_VM_ERROR(
            "failed to load aiv task snapshot, deviceId={}, launchIdx={}, reason={}", deviceId_, launchIdx_,
            errorMessage);
        return false;
    }

    aivBlocks_.clear();
    pipeBarrierRegisters_.clear();
    syncAllRegisters_.clear();
    aivTaskQueues_.clear();
    isInitialized_ = false;

    uint32_t maxTaskId = 0;
    uint32_t maxSyncRound = 0;
    for (const auto& block : taskSnapshot.blocks) {
        uint32_t maxEventId = 0;

        aivTaskQueues_.emplace_back();
        AppendPipeTasksToQueue(block.scalarTasks, aivTaskQueues_.back(), maxTaskId, maxEventId, maxSyncRound);

        aivTaskQueues_.emplace_back();
        AppendPipeTasksToQueue(block.mte2Tasks, aivTaskQueues_.back(), maxTaskId, maxEventId, maxSyncRound);

        aivTaskQueues_.emplace_back();
        AppendPipeTasksToQueue(block.mte3Tasks, aivTaskQueues_.back(), maxTaskId, maxEventId, maxSyncRound);

        HCCL_VM_TRACE(
            "blockIdx={}, scalarTaskCount={}, mte2TaskCount={}, mte3TaskCount={}, maxEventId={}", block.blockIdx,
            block.scalarTasks.size(), block.mte2Tasks.size(), block.mte3Tasks.size(), maxEventId);
        aivBlocks_.emplace_back(std::make_unique<AivBlock>(block.blockIdx, maxEventId));
    }

    for (uint32_t i = 0; i <= maxTaskId; ++i) {
        pipeBarrierRegisters_.push_back(false);
    }
    size_t pipeSize = aivBlocks_.size() * AivSim::AivCore::GetPipeNum();
    for (uint32_t i = 0; i <= maxSyncRound; ++i) {
        syncAllRegisters_.emplace_back(pipeSize, false);
    }

    HCCL_VM_DEBUG(
        "AivGraphExecutor init success: deviceId={}, launchIdx={}, blockNum={}, taskQueNum={}, pipeBarrierRegNum={}, "
        "syncAllRegNum={}",
        deviceId_, launchIdx_, aivBlocks_.size(), aivTaskQueues_.size(), pipeBarrierRegisters_.size(),
        syncAllRegisters_.size());

    isInitialized_ = true;
    return true;
}

bool AivGraphExecutor::HasTask() const
{
    for (const auto& queue : aivTaskQueues_) {
        if (!queue.empty()) {
            return true;
        }
    }
    return false;
}

HcclVmResult AivGraphExecutor::Execute()
{
    while (HasTask()) {
        // 每轮从头到尾遍历全部TaskQueue
        bool progressed = false; // 记录本轮是否有任务执行完毕
        for (auto& queue : aivTaskQueues_) {
            if (queue.empty()) {
                continue;
            }

            std::shared_ptr<AivSim::AivTask> task = queue.front();
            auto ret = ExecuteTask(task);

            if (ret == HcclVmResult::HCCL_SIM_SUCCESS) {
                // Task执行完毕，从Queue中移除
                queue.pop();
                progressed = true;
            } else if (ret == HcclVmResult::HCCL_SIM_VRT_CONTINUE_CMD) {
                // Task暂时无法完成，需要先执行此AivGraph中的其他AivTask
                HCCL_VM_TRACE("AivTask not finish, taskId={}", task->GetTaskId());
            } else if (ret == HcclVmResult::HCCL_SIM_VRT_HOLD_CMD) {
                // Task暂时无法完成，可能需要其他AivGraph配合，但此AivGraph中的其他AivTask可能还可以完成，先执行其他AivTask
                HCCL_VM_TRACE("AivTask hold, taskId={}", task->GetTaskId());
            } else {
                // Task执行错误，返回错误码
                HCCL_VM_ERROR(
                    "AivTask execute failed, taskId={} ret={}", task->GetTaskId(), static_cast<uint32_t>(ret));
                return ret;
            }
        }

        if (!progressed) {
            // 此AivGraph每个TaskQueue都无法往下执行，需要让出Executor，让其他AivGraph先执行
            HCCL_VM_DEBUG("AivGraph Hold, deviceId={} launchIdx={}", deviceId_, launchIdx_);
            return HcclVmResult::HCCL_SIM_VRT_HOLD_CMD;
        }
    }
    // AivGraph的所有Task执行完毕
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTask> task)
{
    HCCL_VM_TRACE("Executing Task, taskId={}", task->GetTaskId());
    switch (task->GetTaskType()) {
        case AivSim::AivTaskType::MEM_COPY:
            return ExecuteTask(std::dynamic_pointer_cast<AivSim::AivTaskMemCopy>(task));
        case AivSim::AivTaskType::REDUCE:
            return ExecuteTask(std::dynamic_pointer_cast<AivSim::AivTaskReduce>(task));
        case AivSim::AivTaskType::WAIT_FLAG:
            return ExecuteTask(std::dynamic_pointer_cast<AivSim::AivTaskWaitFlag>(task));
        case AivSim::AivTaskType::SET_FLAG:
            return ExecuteTask(std::dynamic_pointer_cast<AivSim::AivTaskSetFlag>(task));
        case AivSim::AivTaskType::PIPE_BARRIER:
            return ExecuteTask(std::dynamic_pointer_cast<AivSim::AivTaskPipeBarrier>(task));
        case AivSim::AivTaskType::SYNC_ALL:
            return ExecuteTask(std::dynamic_pointer_cast<AivSim::AivTaskSyncAll>(task));
        case AivSim::AivTaskType::SEND_FLAG:
            return ExecuteTask(std::dynamic_pointer_cast<AivSim::AivTaskSendFlag>(task));
        case AivSim::AivTaskType::RECV_FLAG:
            return ExecuteTask(std::dynamic_pointer_cast<AivSim::AivTaskRecvFlag>(task));
        default:
            HCCL_VM_ERROR("Execute failed, unsupported taskType={:d}", static_cast<uint16_t>(task->GetTaskType()));
            return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTaskMemCopy> task)
{
    HCCL_VM_DEBUG("{}", task->Describe());

    if (task->GetSrc().GetSize() != task->GetDst().GetSize()) {
        HCCL_VM_ERROR("task invalid, len not match, taskId={:d}", task->GetTaskId());
        return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }
    const size_t len = task->GetSrc().GetSize();

    VmUniquePtr src;
    if (GetAddrByOffset(task->GetSrc().GetVirtualAddr(), src) != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR(
            "Transfer virtual-addr to addr failed, AivTaskId={} VirtualAddr={:#x}", task->GetTaskId(),
            task->GetSrc().GetVirtualAddr());
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    VmUniquePtr dst;
    if (GetAddrByOffset(task->GetDst().GetVirtualAddr(), dst) != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR(
            "Transfer virtual-addr to addr failed, AivTaskId={} VirtualAddr={:#x}", task->GetTaskId(),
            task->GetDst().GetVirtualAddr());
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    std::memcpy(dst.get(), src.get(), len);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTaskReduce> task)
{
    HCCL_VM_DEBUG("{}", task->Describe());

    if (task->GetSrc().GetSize() != task->GetDst().GetSize()) {
        HCCL_VM_ERROR("task invalid, len not match, taskId={:d}", task->GetTaskId());
        return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }
    const size_t len = task->GetSrc().GetSize();

    VmUniquePtr src;
    if (GetAddrByOffset(task->GetSrc().GetVirtualAddr(), src) != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR(
            "Transfer virtual-addr to addr failed, AivTaskId={} VirtualAddr={:#x}", task->GetTaskId(),
            task->GetSrc().GetVirtualAddr());
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    VmUniquePtr dst;
    if (GetAddrByOffset(task->GetDst().GetVirtualAddr(), dst) != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR(
            "Transfer virtual-addr to addr failed, AivTaskId={} VirtualAddr={:#x}", task->GetTaskId(),
            task->GetDst().GetVirtualAddr());
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    switch (static_cast<HcclDataType>(task->GetDataType())) {
        case HcclDataType::HCCL_DATA_TYPE_HIF8:
            return Reduce<AscendC::half>(src.get(), dst.get(), len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_INT16:
            return Reduce<int16_t>(src.get(), dst.get(), len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_UINT16:
            return Reduce<uint16_t>(src.get(), dst.get(), len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_FP32:
            return Reduce<float>(src.get(), dst.get(), len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_INT32:
            return Reduce<int32_t>(src.get(), dst.get(), len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_UINT32:
            return Reduce<uint32_t>(src.get(), dst.get(), len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_INT8:
            return Reduce<int8_t>(src.get(), dst.get(), len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_UINT8:
            return Reduce<uint8_t>(src.get(), dst.get(), len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_BFP16:
            return Reduce<AscendC::bfloat16_t>(src.get(), dst.get(), len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_INT64:
            return Reduce<int64_t>(src.get(), dst.get(), len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_UINT64:
            return Reduce<uint64_t>(src.get(), dst.get(), len, task->GetReduceOp());
        default:
            HCCL_VM_ERROR(
                "Reduce DataType not supported, taskId={:d} dataType={:d}", task->GetTaskId(), task->GetDataType());
            return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }
}

template <typename T>
HcclVmResult AivGraphExecutor::Reduce(void* src, void* dst, size_t len, uint32_t reduceOp)
{
    if (len % sizeof(T) != 0) {
        HCCL_VM_ERROR("reduce length invalid, len={:d}", len);
        return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }
    const size_t count = len / sizeof(T);

    T* srcArr = static_cast<T*>(src);
    T* dstArr = static_cast<T*>(dst);

    if (reduceOp == static_cast<uint32_t>(AivSim::ReduceOp::REDUCE_SUM)) {
        for (size_t i = 0; i < count; ++i) {
            dstArr[i] += srcArr[i];
        }
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    if (reduceOp == static_cast<uint32_t>(AivSim::ReduceOp::REDUCE_MAX)) {
        for (size_t i = 0; i < count; ++i) {
            dstArr[i] = std::max(srcArr[i], dstArr[i]);
        }
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    if (reduceOp == static_cast<uint32_t>(AivSim::ReduceOp::REDUCE_MIN)) {
        for (size_t i = 0; i < count; ++i) {
            dstArr[i] = std::min(srcArr[i], dstArr[i]);
        }
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    HCCL_VM_ERROR("ReduceOp not supported, reduceOp={:d}", reduceOp);
    return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTaskSetFlag> task)
{
    if (task->GetBlockId() >= aivBlocks_.size()) {
        HCCL_VM_ERROR(
            "SetFlag task blockIdx out of range, taskId={}, blockIdx={}, aivBlocksSize={}", task->GetTaskId(),
            task->GetBlockId(), aivBlocks_.size());
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    auto& events = aivBlocks_[task->GetBlockId()]->GetEvents();
    events[task->GetEventId()] = true;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTaskWaitFlag> task)
{
    if (task->GetBlockId() >= aivBlocks_.size()) {
        HCCL_VM_ERROR(
            "WaitFlag task blockIdx out of range, taskId={}, blockIdx={}, aivBlocksSize={}", task->GetTaskId(),
            task->GetBlockId(), aivBlocks_.size());
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    auto& events = aivBlocks_[task->GetBlockId()]->GetEvents();
    if (events[task->GetEventId()]) {
        events[task->GetEventId()] = false;
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }
    return HcclVmResult::HCCL_SIM_VRT_CONTINUE_CMD;
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTaskPipeBarrier> task)
{
    pipeBarrierRegisters_[task->GetTaskId()] = true;
    bool pass = true;
    for (const auto& groupTask : task->GetBarrierGroup()) {
        if (!pipeBarrierRegisters_[groupTask->GetTaskId()]) {
            pass = false;
            break;
        }
    }
    return pass ? HcclVmResult::HCCL_SIM_SUCCESS : HcclVmResult::HCCL_SIM_VRT_CONTINUE_CMD;
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTaskSyncAll> task)
{
    if (task->GetBlockId() >= aivBlocks_.size()) {
        HCCL_VM_ERROR(
            "SyncAll task blockIdx out of range, taskId={}, blockIdx={}, aivBlocksSize={}", task->GetTaskId(),
            task->GetBlockId(), aivBlocks_.size());
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    size_t curRound = task->GetSyncRound();
    size_t registerIdx = task->GetBlockId() * AivSim::AivCore::GetPipeNum() + static_cast<size_t>(task->GetCurPipe());
    syncAllRegisters_[curRound][registerIdx] = true;
    HCCL_VM_TRACE(
        "Executing SyncAll, taskId={}, syncRound={}, registerIdx={}", task->GetTaskId(), curRound, registerIdx);

    bool pass = true;
    for (bool val : syncAllRegisters_[curRound]) {
        if (!val) {
            pass = false;
            break;
        }
    }
    return pass ? HcclVmResult::HCCL_SIM_SUCCESS : HcclVmResult::HCCL_SIM_VRT_CONTINUE_CMD;
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTaskSendFlag> task)
{
    HCCL_VM_DEBUG("{}", task->Describe());

    VmUniquePtr flagBuffer;
    if (GetAddrByOffset(task->GetFlagBuffer().GetVirtualAddr(), flagBuffer) != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR(
            "Transfer virtual-addr to addr failed, AivTaskId={} VirtualAddr={:#x}", task->GetTaskId(),
            task->GetFlagBuffer().GetVirtualAddr());
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    AivSim::flag_t* flagPtr = reinterpret_cast<AivSim::flag_t*>(flagBuffer.get());
    *flagPtr = task->GetFlagValue();
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTaskRecvFlag> task)
{
    HCCL_VM_DEBUG("{}", task->Describe());

    VmUniquePtr flagBuffer;
    if (GetAddrByOffset(task->GetFlagBuffer().GetVirtualAddr(), flagBuffer) != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR(
            "Transfer virtual-addr to addr failed, AivTaskId={} VirtualAddr={:#x}", task->GetTaskId(),
            task->GetFlagBuffer().GetVirtualAddr());
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    AivSim::flag_t* flagPtr = reinterpret_cast<AivSim::flag_t*>(flagBuffer.get());
    AivSim::flag_t curFlagValue = *flagPtr;

    if (curFlagValue == task->GetFlagValue()) {
        return HcclVmResult::HCCL_SIM_SUCCESS;
    } else {
        return HcclVmResult::HCCL_SIM_VRT_HOLD_CMD;
    }
}

void AivGraphExecutor::ShowCurrentTaskStatus()
{
    std::stringstream ss;
    ss << "curDevice=" << deviceId_;
    ss << ", TaskStatus(queueIdx, taskId)={";
    for (size_t i = 0; i < aivTaskQueues_.size(); ++i) {
        if (i != 0) {
            ss << ",";
        }
        ss << "[" << i << ",";
        if (aivTaskQueues_[i].empty()) {
            ss << "empty";
        } else {
            ss << aivTaskQueues_[i].front()->GetTaskId();
        }
        ss << "]";
    }
    ss << "}";
    HCCL_VM_DEBUG("{}", ss.str());
}
