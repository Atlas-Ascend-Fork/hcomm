/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_mem_reged_mem_mgr.h"
#include <algorithm>
#include "log.h"

namespace hcomm {

UbMemRegedMemMgr::UbMemRegedMemMgr() { localIpcRmaBufferMgr_ = std::make_unique<LocalIpcRmaBufferMgr>(); }

HcclResult UbMemRegedMemMgr::RegisterMemory(const HcommMem* mem, const char* memTag, void** memHandle)
{
    HCCL_INFO("[%s] Begin", __func__);
    CHK_PTR_NULL(mem);
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(localIpcRmaBufferMgr_);
    std::lock_guard<std::mutex> lock(memMtx_);
    return RegisterMemoryImpl(
        *mem, memTag, memHandle, localIpcRmaBufferMgr_, allRegisteredBuffers_, &handlesRecords_, "UbMemRegedMemMgr",
        [&](auto& bufPtr, auto& parent) {
            return std::make_shared<Hccl::LocalIpcRmaBuffer>(bufPtr, *parent);
        },
        [&](auto& bufPtr) {
            return std::make_shared<Hccl::LocalIpcRmaBuffer>(bufPtr);
        });
}

HcclResult UbMemRegedMemMgr::UnregisterMemory(void* memHandle)
{
    HCCL_INFO("[%s] Begin", __func__);
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(localIpcRmaBufferMgr_);
    std::lock_guard<std::mutex> lock(memMtx_);
    return UnregisterMemoryImpl(
        memHandle, localIpcRmaBufferMgr_, allRegisteredBuffers_, &handlesRecords_,
        [](auto* b) {
            return b->GetIpcPtr();
        },
        [](auto a, auto b) {
            return a == b;
        });
}

HcclResult UbMemRegedMemMgr::MemoryExport(
    [[maybe_unused]] const EndpointDesc& endpointDesc, [[maybe_unused]] void* memHandle,
    [[maybe_unused]] void** memDesc, [[maybe_unused]] uint32_t* memDescLen)
{
    HCCL_INFO("UbMemRegedMemMgr MemoryExport is not supported.");
    return HCCL_SUCCESS;
}

HcclResult UbMemRegedMemMgr::GetAllMemHandles(void** memHandles, uint32_t* memHandleNum)
{
    HCCL_INFO("[%s] Begin", __func__);
    // 返回 mgr 内部存储裸指针，仅可同步立即消费，不得保存或跨越后续注册/注销使用
    std::lock_guard<std::mutex> lock(memMtx_);
    CHK_PTR_NULL(memHandles);
    CHK_PTR_NULL(memHandleNum);
    *memHandleNum = static_cast<uint32_t>(handlesRecords_.size());
    *memHandles = handlesRecords_.empty() ? nullptr : static_cast<void*>(handlesRecords_.data());
    HCCL_INFO("[UbMemRegedMemMgr][GetAllMemHandles] memHandleNum[%u]", *memHandleNum);
    return HCCL_SUCCESS;
}
} // namespace hcomm
