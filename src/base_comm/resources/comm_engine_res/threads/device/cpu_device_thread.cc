/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cpu_device_thread.h"
#include <memory>
#include <utility>
#include "common/cast_utils_pub.h"
#include "cpu_device_thread_registry.h"
#include "log.h"

namespace hccl {

CpuDeviceThread::CpuDeviceThread(const std::string& commId, int32_t deviceLogicId, uint32_t deviceType)
    : commId_(commId),
      deviceLogicId_(deviceLogicId),
      deviceType_(deviceType)
{
    // 本线程不持有设备流资源：构造即置标记，L2/L3 的 DFX/Cache 注册按此标志统一拦截跳过
    SetFakeDeviceRes();
}

LocalNotify* CpuDeviceThread::GetNotify([[maybe_unused]] uint32_t index) const { return nullptr; }

HcclResult CpuDeviceThread::SupplementNotify([[maybe_unused]] uint32_t notifyNum) { return HCCL_SUCCESS; }

bool CpuDeviceThread::IsDeviceA5() const
{
    return deviceType_ == static_cast<uint32_t>(DevType::DEV_TYPE_950)
           || deviceType_ == static_cast<uint32_t>(DevType::DEV_TYPE_960);
}

HcclResult CpuDeviceThread::LocalNotifyRecord([[maybe_unused]] uint32_t notifyId) const
{
    HCCL_ERROR("[CpuDeviceThread][%s] not supported on cpu device thread", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CpuDeviceThread::LocalNotifyWait([[maybe_unused]] uint32_t notifyId) const
{
    HCCL_ERROR("[CpuDeviceThread][%s] not supported on cpu device thread", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CpuDeviceThread::LocalNotifyRecord(ThreadHandle dstThread, uint32_t dstNotifyIdx) const
{
    Thread* dst = ReinterpretAs<Thread*>(dstThread);
    CHK_PTR_NULL(dst);
    uint64_t notifyHostAddr = 0;
    HcclResult ret = dst->GetNotifyHostShmAddr(dstNotifyIdx, notifyHostAddr);
    if (ret == HCCL_SUCCESS) {
        DpuSqe sqe = {};
        sqe.opKind = DPU_OP_THREAD_NOTIFY_RECORD_TO_DPU;
        sqe.u.threadNotifyRecordToDpu.addr = notifyHostAddr;
        uint32_t taskId = 0;
        return GetSqSubmitter().Submit(sqe, taskId);
    }
    if (ret != HCCL_E_NOT_SUPPORT) {
        return ret;
    }
    uint32_t notifyId = 0;
    CHK_RET(dst->GetNotifyId(dstNotifyIdx, notifyId));

    DpuSqe sqe = {};
    sqe.opKind = DPU_OP_THREAD_NOTIFY_RECORD_TO_NPU;
    sqe.u.threadNotifyRecordToNpu.notifyId = notifyId;
    uint32_t taskId = 0;
    return GetSqSubmitter().Submit(sqe, taskId);
}

HcclResult CpuDeviceThread::GetNotifyDevShmAddr(uint32_t notifyIdx, uint64_t& shmAddr) const
{
    const uint32_t notifyNum = notifyNum_.load(std::memory_order_acquire);
    if (notifyIdx >= notifyNum) {
        HCCL_ERROR("[CpuDeviceThread][%s] notifyIdx[%u] out of range, capacity[%u]", __func__, notifyIdx, notifyNum);
        return HCCL_E_PARA;
    }
    uint8_t* const devAddr = notifyDevAddr_.load(std::memory_order_relaxed);
    if (devAddr == nullptr) {
        HCCL_ERROR("[CpuDeviceThread][%s] notify shmem is not set", __func__);
        return HCCL_E_INTERNAL;
    }
    shmAddr = reinterpret_cast<uint64_t>(devAddr + notifyIdx);
    return HCCL_SUCCESS;
}

HcclResult CpuDeviceThread::GetNotifyHostShmAddr(uint32_t notifyIdx, uint64_t& shmAddr) const
{
    const uint32_t notifyNum = notifyNum_.load(std::memory_order_acquire);
    if (notifyIdx >= notifyNum) {
        HCCL_ERROR("[CpuDeviceThread][%s] notifyIdx[%u] out of range, capacity[%u]", __func__, notifyIdx, notifyNum);
        return HCCL_E_PARA;
    }
    uint8_t* const hostAddr = notifyHostAddr_.load(std::memory_order_relaxed);
    if (hostAddr == nullptr) {
        HCCL_ERROR("[CpuDeviceThread][%s] notify shmem is not set", __func__);
        return HCCL_E_INTERNAL;
    }
    shmAddr = reinterpret_cast<uint64_t>(hostAddr + notifyIdx);
    return HCCL_SUCCESS;
}

HcclResult CpuDeviceThread::LocalNotifyWait(uint32_t notifyIdx, uint32_t timeOut) const
{
    const uint32_t notifyNum = notifyNum_.load(std::memory_order_acquire);
    if (notifyIdx >= notifyNum) {
        HCCL_ERROR("[CpuDeviceThread][%s] notifyIdx[%u] out of range, capacity[%u]", __func__, notifyIdx, notifyNum);
        return HCCL_E_PARA;
    }

    DpuSqe sqe = {};
    sqe.opKind = DPU_OP_THREAD_NOTIFY_WAIT;
    sqe.u.threadNotifyWait.notifyIdx = notifyIdx;
    sqe.u.threadNotifyWait.timeOut = timeOut;
    uint32_t taskId = 0;
    CHK_RET(GetSqSubmitter().Submit(sqe, taskId));
    return HCCL_SUCCESS;
}

HcclResult CpuDeviceThread::LocalChannelNotifyWait(ChannelHandle channel, uint32_t notifyIdx, uint32_t timeout) const
{
    DpuSqe sqe = {};
    sqe.opKind = DPU_OP_CHANNEL_NOTIFY_WAIT;
    sqe.u.channelNotifyWait.channelHandle = channel;
    sqe.u.channelNotifyWait.notifyIdx = notifyIdx;
    sqe.u.channelNotifyWait.timeOut = timeout;
    uint32_t taskId = 0;
    CHK_RET(GetSqSubmitter().Submit(sqe, taskId));
    return HCCL_SUCCESS;
}

HcclResult CpuDeviceThread::LocalChannelNotifyRecord(ChannelHandle channel, uint32_t notifyIdx) const
{
    DpuSqe sqe = {};
    sqe.opKind = DPU_OP_CHANNEL_NOTIFY_RECORD;
    sqe.u.channelNotifyRecord.channelHandle = channel;
    sqe.u.channelNotifyRecord.notifyIdx = notifyIdx;
    uint32_t taskId = 0;
    CHK_RET(GetSqSubmitter().Submit(sqe, taskId));
    return HCCL_SUCCESS;
}

HcclResult CpuDeviceThread::LocalChannelWriteWithNotifyNbi(
    ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t notifyIdx) const
{
    DpuSqe sqe = {};
    sqe.opKind = DPU_OP_CHANNEL_WRITE_WITH_NOTIFY_NBI;
    sqe.u.channelWriteWithNotify.channelHandle = channel;
    sqe.u.channelWriteWithNotify.dstAddr = reinterpret_cast<uint64_t>(dst);
    sqe.u.channelWriteWithNotify.srcAddr = reinterpret_cast<uint64_t>(src);
    sqe.u.channelWriteWithNotify.len = len;
    sqe.u.channelWriteWithNotify.notifyIdx = notifyIdx;
    uint32_t taskId = 0;
    CHK_RET(GetSqSubmitter().Submit(sqe, taskId));
    return HCCL_SUCCESS;
}

HcclResult CpuDeviceThread::LocalChannelDrain(ChannelHandle channel) const
{
    DpuSqe sqe = {};
    sqe.opKind = DPU_OP_CHANNEL_DRAIN;
    sqe.u.channelDrainOrFence.channelHandle = channel;
    uint32_t taskId = 0;
    CHK_RET(GetSqSubmitter().Submit(sqe, taskId));
    return HCCL_SUCCESS;
}

HcclResult CpuDeviceThread::LocalChannelFence(ChannelHandle channel) const
{
    DpuSqe sqe = {};
    sqe.opKind = DPU_OP_CHANNEL_FENCE;
    sqe.u.channelDrainOrFence.channelHandle = channel;
    uint32_t taskId = 0;
    CHK_RET(GetSqSubmitter().Submit(sqe, taskId));
    return HCCL_SUCCESS;
}

HcclResult CpuDeviceThread::LocalCommFence() const
{
    DpuSqe sqe = {};
    sqe.opKind = DPU_OP_COMM_FENCE;
    uint32_t taskId = 0;
    CHK_RET(GetSqSubmitter().Submit(sqe, taskId));
    return HCCL_SUCCESS;
}

HcclResult CpuDeviceThread::LocalCopy(void* dst, const void* src, uint64_t sizeByte) const
{
    DpuSqe sqe = {};
    sqe.opKind = DPU_OP_LOCAL_COPY;
    sqe.u.localCopy.dst = reinterpret_cast<uint64_t>(dst);
    sqe.u.localCopy.src = reinterpret_cast<uint64_t>(src);
    sqe.u.localCopy.len = sizeByte;
    uint32_t taskId = 0;
    CHK_RET(GetSqSubmitter().Submit(sqe, taskId));
    return HCCL_SUCCESS;
}

HcclResult CpuDeviceThread::LocalReduce(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t sizeByte,
    [[maybe_unused]] HcommDataType dataType, [[maybe_unused]] HcommReduceOp op) const
{
    HCCL_ERROR("[CpuDeviceThread][%s] not supported on cpu device thread", __func__);
    return HCCL_E_NOT_SUPPORT;
}

void CpuDeviceThread::SetQueueNamecard(
    const QueueDescriptor& sqDesc, const QueueDescriptor& cqDesc, uint32_t notifyNum, uint64_t notifyHostAddr,
    uint64_t notifyDevAddr)
{
    // SQ 描述只有 Init 会校验；初始化失败不能让名片进入「已就绪」，否则提交恒失败且无恢复路径
    HcclResult ret = sqSubmitter_.Init(sqDesc);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CpuDeviceThread][%s] comm[%s] init sq submitter failed, ret[%d]", __func__, commId_.c_str(), ret);
        return;
    }
    notifyHostAddr_.store(reinterpret_cast<uint8_t*>(notifyHostAddr), std::memory_order_relaxed);
    notifyDevAddr_.store(reinterpret_cast<uint8_t*>(notifyDevAddr), std::memory_order_relaxed);
    notifyNum_.store(notifyNum, std::memory_order_release);
    cqDesc_ = cqDesc;
    // 置 ready 前 cqDesc_ 与 notify 字段都已写完，读侧 acquire 到 true 即可用
    namecardReady_.store(true, std::memory_order_release);
    HCCL_INFO(
        "[CpuDeviceThread][%s] comm[%s] sqBase[0x%llx] cqBase[0x%llx]", __func__, commId_.c_str(),
        static_cast<unsigned long long>(sqDesc.base), static_cast<unsigned long long>(cqDesc.base));
}

void CpuDeviceThread::UpdateNotifyInfo(uint32_t notifyNum, uint64_t notifyHostAddr, uint64_t notifyDevAddr)
{
    notifyHostAddr_.store(reinterpret_cast<uint8_t*>(notifyHostAddr), std::memory_order_relaxed);
    notifyDevAddr_.store(reinterpret_cast<uint8_t*>(notifyDevAddr), std::memory_order_relaxed);
    // 地址先写、数量最后写；读侧看到新数量才取地址
    notifyNum_.store(notifyNum, std::memory_order_release);
    HCCL_INFO(
        "[CpuDeviceThread][%s] comm[%s] notify shmem updated, hostAddr[0x%llx] devAddr[0x%llx] notifyNum[%u]", __func__,
        commId_.c_str(), static_cast<unsigned long long>(notifyHostAddr),
        static_cast<unsigned long long>(notifyDevAddr), notifyNum);
}

HcclResult MakeCpuDeviceThread(
    ThreadType threadType, const char* commId, int32_t deviceLogicId, uint32_t deviceType, bool& isCpuDeviceThread,
    std::shared_ptr<hccl::Thread>& outThread)
{
    outThread = nullptr;
    isCpuDeviceThread = false;
    if (threadType != THREAD_TYPE_CPU) {
        return HCCL_SUCCESS; // 非 DPU 线程：调用方走 AicpuTsThread 原路
    }

    std::string commIdStr(commId == nullptr ? "" : commId);
    std::shared_ptr<CpuDeviceThread> entity;
    EXCEPTION_CATCH(
        (entity = std::make_shared<CpuDeviceThread>(commIdStr, deviceLogicId, deviceType)), return HCCL_E_PTR);
    if (entity == nullptr) {
        return HCCL_E_PTR;
    }
    entity->SetThreadType(threadType);
    RegisterCpuDeviceThread(reinterpret_cast<ThreadHandle>(entity.get()), entity);
    isCpuDeviceThread = true;
    outThread = std::move(entity);
    HCCL_INFO(
        "[MakeCpuDeviceThread] cpu device thread created, comm[%s], handle[0x%llx], deviceLogicId[%d]",
        commIdStr.c_str(), static_cast<unsigned long long>(reinterpret_cast<ThreadHandle>(outThread.get())),
        deviceLogicId);
    return HCCL_SUCCESS;
}

} // namespace hccl
