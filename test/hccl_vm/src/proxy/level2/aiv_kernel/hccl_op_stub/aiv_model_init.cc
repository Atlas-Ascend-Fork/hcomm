/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_model_init.h"

#include <array>
#include <cstdint>
#include <string>

#include "aiv_task_json.h"
#include "aiv_db_runtime.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/table_access.h"
#include "sim_log.h"
#include "sim_common_defs.h"

using sim::runtime::g_cur_comm_key;
using sim::runtime::g_cur_device_key;

namespace {
template <size_t N>
std::string ReadCommunicatorName(const char (&name)[N])
{
    size_t length = 0;
    while (length < N && name[length] != '\0') {
        ++length;
    }
    return std::string(name, length);
}

bool InitRankDeviceIds(AivSim::AivKernelExecutor& executor, uint32_t rankSize, uint64_t commId)
{
    if (commId == 0) {
        HCCL_VM_ERROR("cannot initialize AIV rank/device mapping without commId");
        return false;
    }
    const auto currentComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!currentComm.ok()) {
        HCCL_VM_ERROR("failed to resolve AIV communicator by commId={}", commId);
        return false;
    }
    const std::string commName = ReadCommunicatorName(currentComm->comm_id);
    const uint64_t commHash = currentComm->comm_hash;
    if (currentComm->rank_id != executor.GetRankId() || currentComm->rank_size != rankSize
        || executor.GetRankSize() != rankSize || commName.empty()) {
        HCCL_VM_ERROR(
            "invalid current AIV communicator, commId={}, commName={}, dbRank={}, executorRank={}, "
            "dbRankSize={}, rankSize={}",
            commId, commName, currentComm->rank_id, executor.GetRankId(), currentComm->rank_size, rankSize);
        return false;
    }

    if (executor.GetCommId() == commId) {
        if (executor.GetCommName() != commName) {
            HCCL_VM_ERROR(
                "AIV communicator name changed for commId={}, oldName={}, newName={}", commId, executor.GetCommName(),
                commName);
            return false;
        }
        for (uint32_t rankId = 0; rankId < rankSize; ++rankId) {
            if (executor.GetDeviceId(rankId) == UINT32_MAX) {
                HCCL_VM_ERROR("incomplete persistent AIV rank device mapping, commId={}, rankId={}", commId, rankId);
                return false;
            }
        }
        return true;
    }

    std::vector<AivSim::DeviceId> deviceIds(rankSize, UINT32_MAX);

    for (uint32_t rankId = 0; rankId < rankSize; ++rankId) {
        const auto rankCommResult = sim::runtime::Db::GetOneByPred<sim::runtime::Communicator>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Communicator::rank_id, rankId),
            HcclSim::Storage::Eq(&sim::runtime::Communicator::comm_hash, commHash),
            HcclSim::Storage::Eq(&sim::runtime::Communicator::comm_id, commName)));
        if (!rankCommResult.ok()) {
            HCCL_VM_ERROR(
                "failed to resolve rank communicator, commId={}, commName={}, rankId={}", commId, commName, rankId);
            return false;
        }
        const sim::runtime::Communicator& rankComm = *rankCommResult;
        const std::string rankCommName = ReadCommunicatorName(rankComm.comm_id);
        if (rankCommName != commName || rankComm.comm_hash != commHash || rankComm.rank_size != rankSize
            || rankComm.rank_id != rankId || rankComm.device_id >= UINT32_MAX) {
            HCCL_VM_ERROR(
                "invalid AIV rank device mapping, commId={}, commName={}, resolvedCommName={}, "
                "rankSize={}, resolvedRankSize={}, requestedRank={}, resolvedRank={}, deviceId={}",
                commId, commName, rankCommName, rankSize, rankComm.rank_size, rankId, rankComm.rank_id,
                rankComm.device_id);
            return false;
        }
        deviceIds[rankId] = static_cast<AivSim::DeviceId>(rankComm.device_id);
    }

    // Publish the new communication-domain identity and its complete device map
    // only after every rank has been resolved successfully.
    if (!executor.SetCommIdentity(commId, commName)) {
        return false;
    }
    for (uint32_t rankId = 0; rankId < rankSize; ++rankId) {
        if (!executor.SetDeviceId(rankId, deviceIds[rankId])) {
            HCCL_VM_ERROR(
                "failed to publish AIV rank device mapping, rankId={}, deviceId={}", rankId, deviceIds[rankId]);
            return false;
        }
    }
    return true;
}

uint64_t TransDevMappedAddrToVirtual(uint64_t devMappedAddr, uint32_t deviceId)
{
    if (devMappedAddr == 0) {
        return 0;
    }
    const auto virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, devMappedAddr),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, devMappedAddr),
        HcclSim::Storage::Eq(
            &sim::runtime::VirtualMemBlock::src_type, static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV)),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceId)));
    if (!virMemRes.ok()) {
        HCCL_VM_ERROR("cannot find AIV virtual memory, devMappedAddr=0x{:x}, deviceId={}", devMappedAddr, deviceId);
        return 0;
    }
    return virMemRes->start_ptr + (devMappedAddr - virMemRes->dev_mapped_ptr);
}
} // namespace

bool aiv_env_init(
    uint64_t commId, uint32_t rankId, size_t blockNum, const void* buffIn, uint32_t rankSize, uint64_t input,
    uint64_t inputSize, uint64_t output, uint64_t outputSize, uint64_t inputGlobalOffsetBase,
    uint64_t outputGlobalOffsetBase, uint64_t cclBufferSize, uint64_t aivCommInfoSize, AivSim::AivOpParam opParam)
{
    AscendC::block_num = blockNum;
    auto& executor = AivSim::AivKernelExecutor::GetInstance();
    executor.Init(rankId, blockNum, rankSize);
    if (!InitRankDeviceIds(executor, rankSize, commId)) {
        return false;
    }
    if (!AivSim::InitAivUB(executor, blockNum)) {
        HCCL_VM_ERROR("failed to initialize persistent AIV hardware UB buffers, rankId={}", rankId);
        return false;
    }
    executor.SetCurOp(opParam);
    const std::string inputMemDesc = AivSim::Mem{input, inputSize}.Describe();
    const std::string outputMemDesc = AivSim::Mem{output, outputSize}.Describe();
    HCCL_VM_DEBUG("begin");
    HCCL_VM_DEBUG("rankId={}, blockNum={}, buffIn={:p}, rankSize={}", rankId, blockNum, buffIn, rankSize);
    HCCL_VM_DEBUG(
        "input={}, output={}, "
        "inputGlobalOffsetBase={}, outputGlobalOffsetBase={}, cclBufferSize={}, aivCommInfoSize={}",
        inputMemDesc, outputMemDesc, inputGlobalOffsetBase, outputGlobalOffsetBase, cclBufferSize, aivCommInfoSize);
    HCCL_VM_DEBUG(
        "curOp: dataType={}, len={}, reduceOp={}, root={}, sliceId={}, inputStride={}, "
        "outputStride={}, kernelName={}",
        opParam.dataType, opParam.len, opParam.reduceOp, opParam.root, opParam.sliceId, opParam.inputStride,
        opParam.outputStride, opParam.kernelName);

    HCCL_VM_DEBUG("SetIoBuffer");
    executor.SetIoBuffer(input, inputSize, output, outputSize, inputGlobalOffsetBase, outputGlobalOffsetBase);

    if (buffIn == nullptr) {
        HCCL_VM_DEBUG("buffIn is null, skip SetCommBuffer initialization");
        return true;
    }

    const auto* cclBufferTable = static_cast<const uint64_t*>(buffIn);
    const auto* aivCommInfoTable = reinterpret_cast<const uint64_t*>(
        static_cast<const uint8_t*>(buffIn) + AivCommInfoLayout::GM_OUT_TABLE_OFFSET);
    for (uint32_t i = 0; i < rankSize; ++i) {
        const uint32_t deviceId = executor.GetDeviceId(i);
        const uint64_t cclBuffer = TransDevMappedAddrToVirtual(cclBufferTable[i], deviceId);
        const uint64_t aivCommInfoBuffer = TransDevMappedAddrToVirtual(aivCommInfoTable[i], deviceId);
        if ((cclBufferTable[i] != 0 && cclBuffer == 0) || (aivCommInfoTable[i] != 0 && aivCommInfoBuffer == 0)) {
            HCCL_VM_ERROR("Trans ccl or aivcomminfo dev_mapped_addr to vir addr fail");
            return false;
        }
        const std::string cclMemDesc = AivSim::Mem{cclBuffer, cclBufferSize}.Describe();
        const std::string aivCommInfoMemDesc = AivSim::Mem{aivCommInfoBuffer, aivCommInfoSize}.Describe();
        HCCL_VM_DEBUG("SetCommBuffer rank={}, cclBuffer={}, aivCommInfoBuffer={}", i, cclMemDesc, aivCommInfoMemDesc);
        executor.SetCommBuffer(i, cclBuffer, cclBufferSize, aivCommInfoBuffer, aivCommInfoSize);
    }
    return true;
}

void aiv_set_block_idx(int64_t blockIdx) { AscendC::block_idx = blockIdx; }

void aiv_dump_tasks(uint32_t launchIndex)
{
    AivSim::AivKernelExecutor::GetInstance().DumpAllTasks();
    AivSim::DumpExecutorToJsonFile(AivSim::AivKernelExecutor::GetInstance(), launchIndex);
}
