/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cpu_task_pipe.h"
#include <chrono>
#include <thread>
#include "dpu_queue_copy.h"
#include "flush_manager.h"
#include "log.h"
#include "securec.h"
#include "driver/ascend_hal.h"
#include "hcomm_primitives.h"
#include "acl/acl_rt.h"

namespace hccl {

namespace {
    constexpr uint32_t DPU_QUEUE_WAIT_INTERVAL_US = 1;
} // namespace

HcclResult CpuTaskPipe::Init(const DpuSyncLayout& layout)
{
    if (layout.sqMeta == 0 || layout.sqElem == 0 || layout.cqMeta == 0 || layout.cqElem == 0) {
        HCCL_ERROR("[CpuTaskPipe][%s] layout is null", __func__);
        return HCCL_E_PARA;
    }
    if (layout.sqDepth == 0 || (layout.sqDepth & (layout.sqDepth - 1U)) != 0 || layout.cqDepth == 0
        || layout.elemSize != DPU_SQE_SIZE) {
        HCCL_ERROR(
            "[CpuTaskPipe][%s] invalid layout: sqDepth[%u] cqDepth[%u] elemSize[%u]", __func__, layout.sqDepth,
            layout.cqDepth, layout.elemSize);
        return HCCL_E_PARA;
    }

    // DpuShmemMgr::Alloc 只清零首字节，游标与算子序号必须在此显式清零，否则是上一轮的垃圾值
    auto zero = [](uint64_t addr, uint64_t bytes) -> HcclResult {
        if (memset_s(reinterpret_cast<void*>(addr), bytes, 0, bytes) != EOK) {
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    };
    CHK_RET(zero(layout.sqMeta, sizeof(DpuSqMeta)));
    CHK_RET(zero(layout.cqMeta, sizeof(DpuCqMeta)));

    layout_ = layout;

    DpuSqMeta* sqMeta = SqMeta();
    DpuCqMeta* cqMeta = CqMeta();
    sqMeta->depth = layout.sqDepth;
    sqMeta->elemSize = layout.elemSize;
    cqMeta->depth = layout.cqDepth;
    cqMeta->elemSize = layout.elemSize;

    inited_ = true;
    HCCL_INFO(
        "[CpuTaskPipe][%s] sqMeta[0x%llx] sqElem[0x%llx] cqMeta[0x%llx] cqElem[0x%llx] sqDepth[%u] cqDepth[%u]",
        __func__, static_cast<unsigned long long>(layout.sqMeta), static_cast<unsigned long long>(layout.sqElem),
        static_cast<unsigned long long>(layout.cqMeta), static_cast<unsigned long long>(layout.cqElem), layout.sqDepth,
        layout.cqDepth);
    return HCCL_SUCCESS;
}

HcclResult CpuTaskPipe::Run()
{
    if (!inited_) {
        HCCL_ERROR("[CpuTaskPipe][%s] pipe is not inited", __func__);
        return HCCL_E_INTERNAL;
    }

    runExited_.store(false, std::memory_order_release);
    const HcclResult ret = RunLoop();
    // 循环未必等到终止请求才退出（如 SQE 执行失败），那时没人回写 destroyFlag_，退出状态只能在这里记
    runExited_.store(true, std::memory_order_release);
    return ret;
}

HcclResult CpuTaskPipe::RunLoop()
{
    HCCL_RUN_INFO("[CpuTaskPipe][%s] start polling SQ", __func__);
    for (;;) {
        // 1. 终止请求每轮最先查：host 置 true 请退出，kernel 写回 false 表示已退出
        if (destroyFlag_.load(std::memory_order_acquire)) {
            destroyFlag_.store(false, std::memory_order_release);
            HCCL_RUN_INFO("[CpuTaskPipe][%s] recv destroy request, exit", __func__);
            return HCCL_SUCCESS;
        }

        // 2. 复位请求：清空 SQ 并重置游标后继续轮询，不退出
        if (resetFlag_.load(std::memory_order_acquire)) {
            CHK_RET(ResetSq());
            resetFlag_.store(false, std::memory_order_release);
            continue;
        }

        const uint32_t pi = ReadU32(&SqMeta()->pi);
        const uint32_t ci = ReadU32(&SqMeta()->ci);
        if (pi == ci) {
            std::this_thread::sleep_for(std::chrono::microseconds(DPU_QUEUE_WAIT_INTERVAL_US));
            continue;
        }

        DpuSqe sqe{};
        if (!CopyQueueElem(sqe, *SqeAt(ci % layout_.sqDepth))) {
            HCCL_ERROR("[CpuTaskPipe][%s] read sqe failed, ci[%u]", __func__, ci);
            return HCCL_E_INTERNAL;
        }
        HcclResult ret = ExecuteByOpKind(sqe);
        if (ret != HCCL_SUCCESS) {
            // 任何失败都要回执：CQ 是队列唯一的失败通知通道，漏掉这一格上层只能看到后续操作超时
            DpuCqe cqe{};
            cqe.hcclErrCode = static_cast<int32_t>(ret);
            cqe.taskId = sqe.taskId;
            cqe.streamId = dfxStreamId_;
            (void)PushCqe(cqe);
            return ret;
        }
        WriteU32(&SqMeta()->ci, ci + 1U);
    }
    return HCCL_SUCCESS;
}

HcclResult CpuTaskPipe::ResetSq()
{
    // SQ 元素区与 pi/ci 仍落共享内存（生产侧是 device 侧 AICPU，跨地址空间），故照旧走共享内存写
    const uint64_t sqElemBytes = static_cast<uint64_t>(layout_.sqDepth) * layout_.elemSize;
    if (memset_s(reinterpret_cast<void*>(layout_.sqElem), sqElemBytes, 0, sqElemBytes) != EOK) {
        HCCL_ERROR("[CpuTaskPipe][%s] clear SQ elem failed", __func__);
        return HCCL_E_INTERNAL;
    }
    WriteU32(&SqMeta()->pi, 0);
    WriteU32(&SqMeta()->ci, 0);
    HCCL_INFO("[CpuTaskPipe][%s] SQ cleared, pi/ci reset", __func__);
    return HCCL_SUCCESS;
}

HcclResult CpuTaskPipe::PushCqe(const DpuCqe& cqe)
{
    if (!inited_) {
        return HCCL_E_INTERNAL;
    }
    const uint32_t pi = ReadU32(&CqMeta()->pi);
    const uint32_t ci = ReadU32(&CqMeta()->ci);
    // CQ 满则丢弃本事件
    if (pi - ci >= layout_.cqDepth) {
        HCCL_WARNING("[CpuTaskPipe][%s] cq full, drop cqe of taskId[%u]", __func__, cqe.taskId);
        return HCCL_SUCCESS;
    }
    if (!CopyQueueElem(*CqeAt(pi % layout_.cqDepth), cqe)) {
        HCCL_ERROR("[CpuTaskPipe][%s] write cqe failed, pi[%u]", __func__, pi);
        return HCCL_E_INTERNAL;
    }
    WriteU32(&CqMeta()->pi, pi + 1U);
    return HCCL_SUCCESS;
}

HcclResult CpuTaskPipe::ExecuteByOpKind(const DpuSqe& sqe)
{
    switch (sqe.opKind) {
        case DPU_OP_THREAD_NOTIFY_WAIT:
            return ExecuteThreadNotifyWait(sqe.u.threadNotifyWait);
        case DPU_OP_THREAD_NOTIFY_RECORD_TO_DPU:
            return ExecuteThreadNotifyRecordToDpu(sqe.u.threadNotifyRecordToDpu);
        case DPU_OP_THREAD_NOTIFY_RECORD_TO_NPU:
            return ExecuteThreadNotifyRecordToNpu(sqe.u.threadNotifyRecordToNpu);
        case DPU_OP_CHANNEL_NOTIFY_WAIT:
            return ExecuteChannelNotifyWait(sqe.u.channelNotifyWait);
        case DPU_OP_CHANNEL_NOTIFY_RECORD:
            return ExecuteChannelNotifyRecord(sqe.u.channelNotifyRecord);
        case DPU_OP_CHANNEL_WRITE_WITH_NOTIFY_NBI:
            return ExecuteChannelWriteWithNotifyNbi(sqe.u.channelWriteWithNotify);
        case DPU_OP_CHANNEL_DRAIN:
            return ExecuteChannelDrain(sqe.u.channelDrainOrFence);
        case DPU_OP_CHANNEL_FENCE:
            return ExecuteChannelFence(sqe.u.channelDrainOrFence);
        case DPU_OP_COMM_FENCE:
            return ExecuteFence();
        case DPU_OP_LOCAL_COPY:
            return ExecuteLocalCopy(sqe.u.localCopy);
        default:
            HCCL_WARNING(
                "[CpuTaskPipe][%s] opKind[0x%x] taskId[%u] is not supported in control plane", __func__, sqe.opKind,
                sqe.taskId);
            return HCCL_E_NOT_SUPPORT;
    }
}

HcclResult CpuTaskPipe::ExecuteThreadNotifyWait(const DpuThreadNotifyWaitInfo& info)
{
    if (notifyAddr_ == nullptr) {
        HCCL_ERROR("[CpuTaskPipe][%s] notify shmem is not set", __func__);
        return HCCL_E_INTERNAL;
    }

    if (info.notifyIdx >= notifyNum_) {
        HCCL_ERROR("[CpuTaskPipe][%s] notifyIdx[%u] out of range, capacity[%u]", __func__, info.notifyIdx, notifyNum_);
        return HCCL_E_PARA;
    }
    uint8_t* const notifyAddr = notifyAddr_ + info.notifyIdx;
    const auto timeStart = std::chrono::steady_clock::now();
    while (true) {
        if (ReadU8(notifyAddr) == 1U) {
            WriteU8(notifyAddr, 0U);
            return HCCL_SUCCESS;
        }
        if (destroyFlag_.load(std::memory_order_acquire)) {
            HCCL_RUN_INFO(
                "[CpuTaskPipe][%s] recv destroy request during notify wait, notifyIdx[%u]", __func__, info.notifyIdx);
            return HCCL_SUCCESS;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(DPU_QUEUE_WAIT_INTERVAL_US));

        const auto elapsed
            = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timeStart);
        if (elapsed >= std::chrono::seconds(info.timeOut)) {
            HCCL_ERROR(
                "[CpuTaskPipe][%s] timeout, notifyIdx[%u], timeout[%u]s", __func__, info.notifyIdx, info.timeOut);
            return HCCL_E_TIMEOUT;
        }
    }
}

HcclResult CpuTaskPipe::ExecuteThreadNotifyRecordToDpu(const DpuThreadNotifyRecordToDpuInfo& info)
{
    if (notifyAddr_ == nullptr) {
        HCCL_ERROR("[CpuTaskPipe][%s] notify shmem is not set", __func__);
        return HCCL_E_INTERNAL;
    }
    const uintptr_t dst = static_cast<uintptr_t>(info.addr);
    const uintptr_t base = reinterpret_cast<uintptr_t>(notifyAddr_);
    if (dst < base || dst - base >= notifyNum_) {
        HCCL_ERROR(
            "[CpuTaskPipe][%s] notify addr[0x%llx] out of range, base[0x%llx] capacity[%u]", __func__,
            static_cast<unsigned long long>(info.addr), static_cast<unsigned long long>(base), notifyNum_);
        return HCCL_E_PARA;
    }
    WriteU8(reinterpret_cast<uint8_t*>(info.addr), 1U);
    return HCCL_SUCCESS;
}

HcclResult CpuTaskPipe::ExecuteThreadNotifyRecordToNpu(const DpuThreadNotifyRecordToNpuInfo& info)
{
    struct halResourceIdInputInfo in {};
    in.type = DRV_NOTIFY_ID;
    in.tsId = 0;
    in.resourceId = info.notifyId;
    struct halResourceConfigInfo para {};
    para.prop = DRV_ID_RECORD;
    para.value[0] = 1;
    drvError_t ret = halResourceConfig(devId_, &in, &para);
    if (ret != DRV_ERROR_NONE) {
        HCCL_ERROR(
            "[CpuTaskPipe][%s] notify record to npu failed, devId[%u], notifyId[%u], ret[%d]", __func__, devId_,
            info.notifyId, ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult CpuTaskPipe::ExecuteChannelNotifyWait(const DpuChannelNotifyWaitInfo& info)
{
    const int32_t ret = HcommChannelNotifyWaitOnThread(0, info.channelHandle, info.notifyIdx, info.timeOut);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CpuTaskPipe][%s] channel notify wait failed, ret[%d]", __func__, ret);
        return static_cast<HcclResult>(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult CpuTaskPipe::ExecuteChannelNotifyRecord(const DpuChannelNotifyRecordInfo& info)
{
    const int32_t ret = HcommChannelNotifyRecordOnThread(0, info.channelHandle, info.notifyIdx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CpuTaskPipe][%s] channel notify record failed, ret[%d]", __func__, ret);
        return static_cast<HcclResult>(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult CpuTaskPipe::ExecuteChannelWriteWithNotifyNbi(const DpuChannelWriteWithNotifyInfo& info)
{
    const int32_t ret = HcommWriteWithNotifyNbiOnThread(
        0, info.channelHandle, reinterpret_cast<void*>(info.dstAddr), reinterpret_cast<const void*>(info.srcAddr),
        info.len, info.notifyIdx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CpuTaskPipe][%s] channel write with notify nbi failed, ret[%d]", __func__, ret);
        return static_cast<HcclResult>(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult CpuTaskPipe::ExecuteChannelDrain(const DpuChannelDrainOrFenceInfo& info)
{
    const int32_t ret = HcommChannelDrainOnThread(0, info.channelHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CpuTaskPipe][%s] channel drain failed, ret[%d]", __func__, ret);
        return static_cast<HcclResult>(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult CpuTaskPipe::ExecuteChannelFence(const DpuChannelDrainOrFenceInfo& info)
{
    const int32_t ret = HcommChannelFenceOnThread(0, info.channelHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CpuTaskPipe][%s] channel fence failed, ret[%d]", __func__, ret);
        return static_cast<HcclResult>(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult CpuTaskPipe::ExecuteFence()
{
    const HcclResult ret = Hccl::FlushManager::GetInstance().Flush();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CpuTaskPipe][%s] fence failed, ret[%d]", __func__, ret);
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult CpuTaskPipe::ExecuteLocalCopy(const DpuLocalCopyInfo& info)
{
    aclError ret = aclrtMemcpy(
        reinterpret_cast<void*>(info.dst), info.len, reinterpret_cast<const void*>(info.src), info.len,
        ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[CpuTaskPipe][%s] local copy failed, ret[%d], dst[0x%llx], src[0x%llx], len[%llu].", __func__, ret,
            info.dst, info.src, info.len);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

uint8_t CpuTaskPipe::ReadU8(const uint8_t* src) { return __atomic_load_n(src, __ATOMIC_ACQUIRE); }

void CpuTaskPipe::WriteU8(uint8_t* dst, uint8_t value) { __atomic_store_n(dst, value, __ATOMIC_RELEASE); }

uint32_t CpuTaskPipe::ReadU32(const uint32_t* src) { return __atomic_load_n(src, __ATOMIC_ACQUIRE); }

void CpuTaskPipe::WriteU32(uint32_t* dst, uint32_t value) { __atomic_store_n(dst, value, __ATOMIC_RELEASE); }

} // namespace hccl
