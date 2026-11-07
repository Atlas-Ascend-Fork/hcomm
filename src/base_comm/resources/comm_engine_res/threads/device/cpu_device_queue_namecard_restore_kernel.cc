/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cpu_device_queue_namecard_restore_kernel.h"
#include "dpu_sync_layout.h"
#include "cpu_device_thread_registry.h"
#include "thread_entity_utils.h"
#include "log.h"

namespace {
constexpr uint32_t DPU_QUEUE_NAMECARD_RESTORE_OK = 0;
constexpr uint32_t DPU_QUEUE_NAMECARD_RESTORE_FAIL = 1;
} // namespace

extern "C" __attribute__((visibility("default"))) uint32_t RunDpuQueueNamecardRestore(void* args)
{
    if (args == nullptr) {
        HCCL_ERROR("[RunDpuQueueNamecardRestore] args is null.");
        return DPU_QUEUE_NAMECARD_RESTORE_FAIL;
    }
    const auto* param = reinterpret_cast<const hccl::DpuQueueNamecardParam*>(args);
    // CQ 深度恒为单槽，depth 不合契约会让下游 ci % depth 寻址越界
    if (param->cqDesc.base == 0 || param->cqDesc.piAddr == 0 || param->cqDesc.ciAddr == 0
        || param->cqDesc.elemSize != hccl::DPU_CQE_SIZE || param->cqDesc.depth != hccl::DPU_CQ_DEPTH) {
        HCCL_ERROR(
            "[RunDpuQueueNamecardRestore] invalid cq desc, base[0x%llx] piAddr[0x%llx] ciAddr[0x%llx] "
            "elemSize[%u] depth[%u]",
            param->cqDesc.base, param->cqDesc.piAddr, param->cqDesc.ciAddr, param->cqDesc.elemSize,
            param->cqDesc.depth);
        return DPU_QUEUE_NAMECARD_RESTORE_FAIL;
    }
    // SQ 描述同样要完整：生产侧按 depth 取模寻址，depth 非 2 的幂会算到半区之外
    if (param->sqDesc.base == 0 || param->sqDesc.piAddr == 0 || param->sqDesc.ciAddr == 0
        || param->sqDesc.elemSize != hccl::DPU_SQE_SIZE || param->sqDesc.depth == 0
        || (param->sqDesc.depth & (param->sqDesc.depth - 1U)) != 0) {
        HCCL_ERROR(
            "[RunDpuQueueNamecardRestore] invalid sq desc, base[0x%llx] piAddr[0x%llx] ciAddr[0x%llx] "
            "elemSize[%u] depth[%u]",
            param->sqDesc.base, param->sqDesc.piAddr, param->sqDesc.ciAddr, param->sqDesc.elemSize,
            param->sqDesc.depth);
        return DPU_QUEUE_NAMECARD_RESTORE_FAIL;
    }

    // 名片只落到线程实体上：handle 为 0 就没有落点，CQ 与 SQ 都会丢失
    if (param->threadHandle == 0) {
        HCCL_ERROR("[RunDpuQueueNamecardRestore] thread handle is null, namecard has no destination.");
        return DPU_QUEUE_NAMECARD_RESTORE_FAIL;
    }
    auto threadHandle = param->threadHandle;
    HcclResult unwrapRet = hccl::UnwrapThreadHandle(threadHandle);
    if (unwrapRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[RunDpuQueueNamecardRestore] unwrap thread handle failed, handle[0x%llx], ret[0x%x]", param->threadHandle,
            unwrapRet);
        return DPU_QUEUE_NAMECARD_RESTORE_FAIL;
    }
    std::shared_ptr<hccl::CpuDeviceThread> entity = FindCpuDeviceThread(threadHandle);
    if (entity == nullptr) {
        // 实体未登记：线程销毁与投递竞争时的正常可能，不上报为错误，只是名片落空
        HCCL_WARNING("[RunDpuQueueNamecardRestore] dpu thread entity not found, handle[0x%llx]", param->threadHandle);
    } else {
        entity->SetQueueNamecard(
            param->sqDesc, param->cqDesc, param->notifyNum, param->notifyHostAddr, param->notifyDevAddr);
        HCCL_INFO(
            "[RunDpuQueueNamecardRestore] handle[0x%llx], sqBase[0x%llx], cqBase[0x%llx], depth[%u], elemSize[%u], "
            "piAddr[0x%llx], ciAddr[0x%llx]",
            param->threadHandle, param->sqDesc.base, param->cqDesc.base, param->cqDesc.depth, param->cqDesc.elemSize,
            param->cqDesc.piAddr, param->cqDesc.ciAddr);
    }
    return DPU_QUEUE_NAMECARD_RESTORE_OK;
}
