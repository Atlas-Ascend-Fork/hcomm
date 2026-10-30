/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "thread_entity_resources.h"
#include "comm_engine/aicpu_comm_def_pub.h"

#ifndef CCL_KERNEL_AICPU
#include <cstddef>
#include <utility>
#include "adapter_rts_common.h"

namespace hccl {
namespace {
    constexpr size_t RTSQ_OFFSET = sizeof(HcommThreadEntity);
    constexpr size_t RTSQ_SQE_SIZE = 64;
    constexpr size_t RTSQ_SQE_BATCH_SIZE = 128;
    // SQE 缓存起点按单个 SQE 大小对齐。
    constexpr size_t SQE_OFFSET
        = (RTSQ_OFFSET + sizeof(HcommThreadSqContext) + RTSQ_SQE_SIZE - 1) / RTSQ_SQE_SIZE * RTSQ_SQE_SIZE;
    constexpr size_t FIXED_POD_SIZE = SQE_OFFSET + RTSQ_SQE_SIZE * RTSQ_SQE_BATCH_SIZE;
    HcclResult AllocNotifyBuffer(uint32_t notifyNum, DeviceMem& mem)
    {
        CHK_PRT_RET(
            notifyNum > HCOMM_THREAD_NOTIFY_MAX_NUM, HCCL_ERROR("[%s] too many notifies[%u]", __func__, notifyNum),
            HCCL_E_PARA);
        if (notifyNum == 0) {
            return HCCL_SUCCESS;
        }
        const size_t size = notifyNum * sizeof(RegedNotifyEntity);
        mem = DeviceMem::alloc(size);
        CHK_PTR_NULL(mem.ptr());
        return hrtMemSet(mem.ptr(), size, size);
    }
} // namespace

HcclResult ThreadEntityResources::Init(const HcommDevInfo& devInfo, uint32_t notifyNum, bool initPodFields)
{
    CHK_PRT_RET(fixedMem_.ptr() != nullptr, HCCL_ERROR("[%s] already initialized", __func__), HCCL_E_PARA);
    DeviceMem fixedMem = DeviceMem::alloc(initPodFields ? FIXED_POD_SIZE : sizeof(HcommThreadEntity));
    CHK_PTR_NULL(fixedMem.ptr());
    CHK_RET(hrtMemSet(fixedMem.ptr(), fixedMem.size(), fixedMem.size()));
    DeviceMem notifyMem;
    HcommThreadEntity entity{};
    entity.abiHeader = {HCOMM_THREAD_ENTITY_VERSION, HCOMM_THREAD_ENTITY_MAGIC_WORD, sizeof(HcommThreadEntity), 0};
    if (initPodFields) {
        auto* base = static_cast<uint8_t*>(fixedMem.ptr());
        entity.devInfo = devInfo;
        entity.sqContextAddr = reinterpret_cast<HcommThreadSqContext*>(base + RTSQ_OFFSET);
        HcommThreadSqContext sq{};
        sq.type = 0U;
        sq.contextInfo.RtsqContext.localSqAddr = reinterpret_cast<uintptr_t>(base + SQE_OFFSET);
        sq.contextInfo.RtsqContext.localSqDepth = RTSQ_SQE_BATCH_SIZE;
        CHK_RET(hrtMemSyncCopy(
            entity.sqContextAddr, sizeof(sq), &sq, sizeof(sq), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        CHK_RET(AllocNotifyBuffer(notifyNum, notifyMem));
        entity.localNotifyNum = notifyNum;
        entity.localNotifyAddr = static_cast<RegedNotifyEntity*>(notifyMem.ptr());
    }
    CHK_RET(hrtMemSyncCopy(
        fixedMem.ptr(), sizeof(entity), &entity, sizeof(entity), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    fixedMem_ = std::move(fixedMem);
    notifyMem_ = std::move(notifyMem);
    return HCCL_SUCCESS;
}

HcclResult ThreadEntityResources::UpdateNotifyBuffer(uint32_t notifyNum)
{
    CHK_PTR_NULL(fixedMem_.ptr());
    // A3 不使用 Notify POD；相同或较小需求复用现有容量，不缩减已发布的资源。
    if (fixedMem_.size() == sizeof(HcommThreadEntity) || notifyNum <= notifyMem_.size() / sizeof(RegedNotifyEntity)) {
        return HCCL_SUCCESS;
    }
    DeviceMem notifyMem;
    CHK_RET(AllocNotifyBuffer(notifyNum, notifyMem));
    if (notifyMem_.ptr() != nullptr) {
        CHK_RET(hrtMemSyncCopy(
            notifyMem.ptr(), notifyMem.size(), notifyMem_.ptr(), notifyMem_.size(),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE));
    }
    auto* entity = static_cast<uint8_t*>(fixedMem_.ptr());
    RegedNotifyEntity* localNotifyAddr = static_cast<RegedNotifyEntity*>(notifyMem.ptr());
    CHK_RET(hrtMemSyncCopy(
        entity + offsetof(HcommThreadEntity, localNotifyAddr), sizeof(localNotifyAddr), &localNotifyAddr,
        sizeof(localNotifyAddr), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    // localNotifyNum 是发布点：新地址先发布，中间态最多按旧数量访问新数组前缀，不会越界。
    CHK_RET(hrtMemSyncCopy(
        entity + offsetof(HcommThreadEntity, localNotifyNum), sizeof(notifyNum), &notifyNum, sizeof(notifyNum),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    // DeviceMem 移动赋值不释放原内存，先移交旧数组给局部对象，在成功发布后析构释放。
    DeviceMem oldNotifyMem = std::move(notifyMem_);
    notifyMem_ = std::move(notifyMem);
    return HCCL_SUCCESS;
}
} // namespace hccl
#endif // CCL_KERNEL_AICPU
