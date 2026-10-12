/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ai_core_stub.h"

#include <algorithm>
#include <cstdint>

#include "sim_log.h"

namespace AivSim {
AivKernelExecutor::AivKernelExecutor() {}

AivKernelExecutor& AivKernelExecutor::GetInstance()
{
    static AivKernelExecutor instance;
    return instance;
}

void AivKernelExecutor::Init(RankId rankId, size_t blockNum, uint32_t rankSize)
{
    Reset();
    if (rankSize > HcclSim::GetMaxRanks()) {
        HCCL_VM_ERROR("Init rankSize={} exceeds max ranks={}", rankSize, HcclSim::GetMaxRanks());
        return;
    }
    cclBuffer_.resize(rankSize);
    aivCommInfoBuffer_.resize(rankSize);
    rankDeviceIds_.resize(rankSize, UINT32_MAX);
    rankId_ = rankId;
    rankSize_ = rankSize;
    for (uint32_t i = 0; i < blockNum; ++i) {
        aivCores_.emplace_back(std::make_shared<AivCore>(i));
    }
}

void AivKernelExecutor::Reset()
{
    aivCores_.clear();
    taskIdGen_.store(0);
    rankId_ = UINT32_MAX;
    rankSize_ = 0;
    curOp_ = {};
    inBuffer_ = {};
    outBuffer_ = {};
    inputGlobalOffsetBase_ = 0;
    outputGlobalOffsetBase_ = 0;
    std::fill(cclBuffer_.begin(), cclBuffer_.end(), Mem{});
    std::fill(aivCommInfoBuffer_.begin(), aivCommInfoBuffer_.end(), Mem{});
}

bool AivKernelExecutor::SetCommIdentity(uint64_t commId, const std::string& commName)
{
    if (commId == 0 || commName.empty()) {
        HCCL_VM_ERROR("invalid AIV communicator identity, commId={}, commName={}", commId, commName);
        return false;
    }
    if (commId_ == commId) {
        return commName_ == commName;
    }
    commId_ = commId;
    commName_ = commName;
    std::fill(rankDeviceIds_.begin(), rankDeviceIds_.end(), UINT32_MAX);
    return true;
}

bool AivKernelExecutor::SetDeviceId(RankId rankId, DeviceId deviceId)
{
    if (rankId >= rankSize_ || deviceId == UINT32_MAX) {
        HCCL_VM_ERROR("SetDeviceId invalid mapping, rankId={}, deviceId={}, rankSize={}", rankId, deviceId, rankSize_);
        return false;
    }
    rankDeviceIds_[rankId] = deviceId;
    return true;
}

uint64_t AivKernelExecutor::ResolveDataSliceAddress(const AivDataSlice& slice, RankId rankId, BlockId blockId) const
{
    Mem buffer{};
    switch (slice.GetType()) {
        case AivBufferType::INPUT:
            buffer = inBuffer_;
            break;
        case AivBufferType::OUTPUT:
            buffer = outBuffer_;
            break;
        case AivBufferType::CCL:
            buffer = GetCclBuffer(rankId);
            break;
        case AivBufferType::UB:
            buffer = GetUbBuffer(blockId);
            break;
        case AivBufferType::AIV_COMM:
            buffer = GetAivCommInfoBuffer(rankId);
            break;
        default:
            return 0;
    }
    if (buffer.addr == 0 || slice.GetOffset() > buffer.size || slice.GetSize() > buffer.size - slice.GetOffset()) {
        return 0;
    }
    return buffer.addr + slice.GetOffset();
}

Mem AivKernelExecutor::GetUbBuffer(BlockId blockId) const
{
    return blockId < ubBufferInfos_.size() ? ubBufferInfos_[blockId] : Mem{};
}

bool AivKernelExecutor::AddUbBuffer(const Mem& buffer)
{
    if (buffer.addr == 0 || buffer.size == 0) {
        return false;
    }
    ubBufferInfos_.push_back(buffer);
    return true;
}

std::shared_ptr<AivCore> AivKernelExecutor::GetAivCore(int64_t blockIdx)
{
    return (blockIdx >= 0 && blockIdx < aivCores_.size()) ? aivCores_[blockIdx] : nullptr;
}

void AivKernelExecutor::SetIoBuffer(
    uint64_t inBuffer, uint64_t inBufferSize, uint64_t outBuffer, uint64_t outBufferSize,
    uint64_t inputGlobalOffsetBase, uint64_t outputGlobalOffsetBase)
{
    inBuffer_ = {inBuffer, inBufferSize};
    outBuffer_ = {outBuffer, outBufferSize};
    inputGlobalOffsetBase_ = inputGlobalOffsetBase;
    outputGlobalOffsetBase_ = outputGlobalOffsetBase;
}

void AivKernelExecutor::SetCommBuffer(
    RankId rankId, uint64_t cclBuffer, uint64_t cclBufferSize, uint64_t aivCommInfoBuffer,
    uint64_t aivCommInfoBufferSize)
{
    if (rankId >= cclBuffer_.size()) {
        HCCL_VM_WARN("SetCommBuffer rankId={} out of range", rankId);
        return;
    }
    cclBuffer_[rankId] = {cclBuffer, cclBufferSize};
    aivCommInfoBuffer_[rankId] = {aivCommInfoBuffer, aivCommInfoBufferSize};
}

bool AivBufferContains(const Mem& buffer, uint64_t addr, uint64_t size)
{
    if (buffer.addr == 0 || buffer.size == 0 || addr < buffer.addr) {
        return false;
    }

    const uint64_t offset = addr - buffer.addr;
    if (offset >= buffer.size) {
        return false;
    }

    return size <= (buffer.size - offset);
}

bool AivBufferMatch(
    uint64_t addr, uint64_t size, const Mem& buffer, AivBufferType type, RankId rank, AivDataSlice& slice,
    RankId* matchedRank)
{
    if (!AivBufferContains(buffer, addr, size)) {
        return false;
    }

    slice = AivDataSlice(type, addr - buffer.addr, size);
    if (matchedRank != nullptr) {
        *matchedRank = rank;
    }
    return true;
}

AivDataSlice
AivKernelExecutor::ResolveGlobalDataSlice(uint64_t addr, uint64_t size, RankId* rankId, BlockId ubBlockId) const
{
    if (rankId != nullptr) {
        *rankId = UINT32_MAX;
    }

    AivDataSlice matchedSlice{};
    if (ubBlockId != UINT32_MAX
        && AivBufferMatch(addr, size, GetUbBuffer(ubBlockId), AivBufferType::UB, rankId_, matchedSlice, rankId)) {
        matchedSlice.SetDeviceId(GetDeviceId(rankId_));
        matchedSlice.SetVirtualAddr(addr);
        return matchedSlice;
    }
    if (AivBufferMatch(addr, size, inBuffer_, AivBufferType::INPUT, rankId_, matchedSlice, rankId)) {
        matchedSlice.SetDeviceId(GetDeviceId(rankId_));
        matchedSlice.SetVirtualAddr(addr);
        return matchedSlice;
    }
    if (AivBufferMatch(addr, size, outBuffer_, AivBufferType::OUTPUT, rankId_, matchedSlice, rankId)) {
        matchedSlice.SetDeviceId(GetDeviceId(rankId_));
        matchedSlice.SetVirtualAddr(addr);
        return matchedSlice;
    }
    for (RankId i = 0; i < rankSize_; ++i) {
        if (AivBufferMatch(addr, size, cclBuffer_[i], AivBufferType::CCL, i, matchedSlice, rankId)) {
            matchedSlice.SetDeviceId(GetDeviceId(i));
            matchedSlice.SetVirtualAddr(addr);
            return matchedSlice;
        }
        if (AivBufferMatch(addr, size, aivCommInfoBuffer_[i], AivBufferType::AIV_COMM, i, matchedSlice, rankId)) {
            matchedSlice.SetDeviceId(GetDeviceId(i));
            matchedSlice.SetVirtualAddr(addr);
            return matchedSlice;
        }
    }

    return {};
}

void AivKernelExecutor::DumpAllTasks() const
{
    HCCL_VM_DEBUG("");
    HCCL_VM_DEBUG("[rankId={}]", GetRankId());

    for (const auto& core : aivCores_) {
        core->DumpAllTasks();
    }
}

void AivCore::AppendScalar(const std::shared_ptr<AivTask>& task)
{
    auto& executor = AivKernelExecutor::GetInstance();
    task->SetRankId(executor.GetRankId());
    task->SetDeviceId(executor.GetDeviceId(executor.GetRankId()));
    task->SetCommId(executor.GetCommId());
    task->SetBlockId(blockIdx_);
    task->SetTaskId(executor.NextTaskId());
    task->SetCurPipe(AscendC::pipe_t::PIPE_S);
    pipeScalar_.push_back(task);
}

void AivCore::AppendMTE2(const std::shared_ptr<AivTask>& task)
{
    auto& executor = AivKernelExecutor::GetInstance();
    task->SetRankId(executor.GetRankId());
    task->SetDeviceId(executor.GetDeviceId(executor.GetRankId()));
    task->SetCommId(executor.GetCommId());
    task->SetBlockId(blockIdx_);
    task->SetTaskId(executor.NextTaskId());
    task->SetCurPipe(AscendC::pipe_t::PIPE_MTE2);
    pipeMTE2_.push_back(task);
}

void AivCore::AppendMTE3(const std::shared_ptr<AivTask>& task)
{
    auto& executor = AivKernelExecutor::GetInstance();
    task->SetRankId(executor.GetRankId());
    task->SetDeviceId(executor.GetDeviceId(executor.GetRankId()));
    task->SetCommId(executor.GetCommId());
    task->SetBlockId(blockIdx_);
    task->SetTaskId(executor.NextTaskId());
    task->SetCurPipe(AscendC::pipe_t::PIPE_MTE3);
    pipeMTE3_.push_back(task);
}

void AivCore::DumpAllTasks() const
{
    HCCL_VM_DEBUG("[blockIdx={:d}]", blockIdx_);

    HCCL_VM_DEBUG("[SCALAR] ");
    for (const auto& task : pipeScalar_) {
        HCCL_VM_DEBUG("    {:s}", task->Describe());
    }

    HCCL_VM_DEBUG("[MTE2] ");
    for (const auto& task : pipeMTE2_) {
        HCCL_VM_DEBUG("    {:s}", task->Describe());
    }

    HCCL_VM_DEBUG("[MTE3] ");
    for (const auto& task : pipeMTE3_) {
        HCCL_VM_DEBUG("    {:s}", task->Describe());
    }
}
} // namespace AivSim
