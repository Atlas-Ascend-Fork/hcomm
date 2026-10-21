/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "resources/endpoints/endpoint.h"
#include "hccl_common.h"
#include "ub_reged_mem_mgr.h"
#include <algorithm>
#include <unistd.h>
#include "log.h"
#include "hccl/hccl_res.h"
#include "hccl_mem_v2.h"
#include "exchange_ub_buffer_dto.h"
#include "local_ub_rma_buffer_manager.h"
#include "local_ub_rma_buffer.h"

namespace hcomm {

UbRegedMemMgr::UbRegedMemMgr(RdmaHandle rdmaHandle) : rdmaHandle_(rdmaHandle)
{
    localUbRmaBufferMgr_ = std::make_unique<LocalUbRmaBufferMgr>();
}

HcclResult UbRegedMemMgr::RegisterMemory(const HcommMem* mem, const char* memTag, void** memHandle)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(mem);
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(localUbRmaBufferMgr_);
    std::lock_guard<std::mutex> lock(memMtx_);
    return RegisterMemoryImpl(
        *mem, memTag, memHandle, localUbRmaBufferMgr_, allRegisteredBuffers_, &handlesRecords_, "UbRegedMemMgr",
        [&](auto& bufPtr, auto& parent) {
            return std::make_shared<Hccl::LocalUbRmaBuffer>(bufPtr, rdmaHandle_, *parent);
        },
        [&](auto& bufPtr) {
            return std::make_shared<Hccl::LocalUbRmaBuffer>(bufPtr, rdmaHandle_);
        });
}

HcclResult UbRegedMemMgr::UnregisterMemory(void* memHandle)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(localUbRmaBufferMgr_);
    std::lock_guard<std::mutex> lock(memMtx_);
    return UnregisterMemoryImpl(
        memHandle, localUbRmaBufferMgr_, allRegisteredBuffers_, &handlesRecords_,
        [](auto* b) {
            return b->GetMemRegOutParam();
        },
        [](const void* lhs, const void* rhs) {
            return Hccl::LocalUbRmaBuffer::IsSameMemRegOutParam(lhs, rhs);
        });
}

HcclResult UbRegedMemMgr::GetMemDesc(const EndpointDesc endpointDesc, Hccl::LocalUbRmaBuffer* localUbRmaBuffer) const
{
    auto dto = localUbRmaBuffer->GetExchangeDto();
    CHK_SMART_PTR_NULL(dto);
    std::vector<char> tempLocalMemDesc;
    CHK_RET(BuildMemDesc(*dto, endpointDesc, tempLocalMemDesc));
    localUbRmaBuffer->Desc = std::move(tempLocalMemDesc);
    return HCCL_SUCCESS;
}

HcclResult
UbRegedMemMgr::MemoryExport(const EndpointDesc& endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);

    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(memDesc);
    CHK_PTR_NULL(memDescLen);
    std::lock_guard<std::mutex> lock(memMtx_);

    Hccl::LocalUbRmaBuffer* localUbRmaBuffer = nullptr;
    CHK_RET(ValidateMemExportHandle(memHandle, allRegisteredBuffers_, localUbRmaBuffer));

    // 获取序列化信息
    CHK_RET(GetMemDesc(endpointDesc, localUbRmaBuffer));

    *memDescLen = static_cast<uint32_t>(localUbRmaBuffer->Desc.size());
    *memDesc = static_cast<void*>(localUbRmaBuffer->Desc.data());

    return HCCL_SUCCESS;
}

HcclResult UbRegedMemMgr::GetAllMemHandles(void** memHandles, uint32_t* memHandleNum)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    // 返回 mgr 内部存储裸指针，仅可同步立即消费，不得保存或跨越后续注册/注销使用
    std::lock_guard<std::mutex> lock(memMtx_);
    CHK_PTR_NULL(memHandles);
    CHK_PTR_NULL(memHandleNum);
    *memHandleNum = static_cast<uint32_t>(handlesRecords_.size());
    *memHandles = handlesRecords_.empty() ? nullptr : static_cast<void*>(handlesRecords_.data());
    HCCL_INFO("[UbRegedMemMgr][GetAllMemHandles] memHandleNum[%u]", *memHandleNum);
    return HCCL_SUCCESS;
}

HcclResult UbRegedMemMgr::FindHandleRecord(void* memHandle, std::shared_ptr<Hccl::LocalUbRmaBuffer>* out) const
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(out);
    std::lock_guard<std::mutex> lock(memMtx_);
    auto it = std::find_if(handlesRecords_.begin(), handlesRecords_.end(), [memHandle](const auto& record) {
        return record.get() == memHandle;
    });
    CHK_PRT_RET(
        it == handlesRecords_.end(), HCCL_ERROR("[UbRegedMemMgr][%s] memHandle[%p] not found", __func__, memHandle),
        HCCL_E_NOT_FOUND);
    *out = *it;
    return HCCL_SUCCESS;
}

} // namespace hcomm
