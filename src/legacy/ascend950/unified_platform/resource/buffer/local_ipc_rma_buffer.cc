/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/cast_utils_pub.h"
#include "internal_exception.h"
#include "local_ipc_rma_buffer_v2.h"

#include "rma_buffer.h"
#include "exchange_ipc_buffer_dto.h"

namespace Hccl {

LocalIpcRmaBuffer::LocalIpcRmaBuffer(std::shared_ptr<Buffer> buf) : LocalRmaBuffer(buf, RmaType::IPC)
{
    CHK_RET_THROW(
        InternalException, "[LocalIpcRmaBuffer] HrtDevMemAlignWithPage failed",
        HrtDevMemAlignWithPage(ReinterpretAs<void*>(buf->GetAddr()), buf->GetSize(), ipcPtr, ipcSize, ipcOffset));
    CHK_RET_THROW(
        InternalException, "[LocalIpcRmaBuffer] HrtIpcSetMemoryName failed",
        HrtIpcSetMemoryName(ipcPtr, name, ipcSize, RTS_IPC_MEM_NAME_LEN));
}

LocalIpcRmaBuffer::LocalIpcRmaBuffer(std::shared_ptr<Buffer> buf, const LocalIpcRmaBuffer& parent)
    : LocalRmaBuffer(buf, RmaType::IPC, true)
{
    (void)memcpy_s(name, RTS_IPC_MEM_NAME_LEN, parent.name, RTS_IPC_MEM_NAME_LEN);
    ipcPtr = parent.ipcPtr;
    ipcOffset = parent.ipcOffset;
    ipcSize = parent.ipcSize;
    HCCL_INFO(
        "[LocalIpcRmaBuffer] alias constructor, parent name[%s] ipcPtr[%p] ipcOffset[%llu] ipcSize[%llu] buf=%s", name,
        ipcPtr, static_cast<unsigned long long>(ipcOffset), static_cast<unsigned long long>(ipcSize),
        buf->Describe().c_str());
}

LocalIpcRmaBuffer::~LocalIpcRmaBuffer()
{
    if (!isAlias_) {
        HcclResult ret = HrtIpcDestroyMemoryName(name);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[LocalIpcRmaBuffer] HrtIpcDestroyMemoryName failed, ret[%d].", ret);
        }
    }
}

string LocalIpcRmaBuffer::Describe() const
{
    return StringFormat(
        "LocalIpcRmaBuffer[buf=%s, ipcPtr=0x%llx, ipcOffset=0x%llx, ipcSize=0x%llx, name=%s]", buf->Describe().c_str(),
        ReinterpretAs<uintptr_t>(ipcPtr), ipcOffset, ipcSize, name);
}

std::unique_ptr<Serializable> LocalIpcRmaBuffer::GetExchangeDto()
{
    s32 tgid = 0;
    if (HrtDeviceGetBareTgid(tgid) != HCCL_SUCCESS) {
        THROW<InternalException>("HrtDeviceGetBareTgid failed");
    }
    std::unique_ptr<ExchangeIpcBufferDto> dto = make_unique<ExchangeIpcBufferDto>(
        buf->GetAddr(), buf->GetSize(), ipcOffset, static_cast<u32>(tgid), buf->GetMemInfo().c_str(),
        buf->GetMemType());
    (void)memcpy_s(dto->name, RTS_IPC_MEM_NAME_LEN, name, RTS_IPC_MEM_NAME_LEN);
    return std::unique_ptr<Serializable>(dto.release());
}

void LocalIpcRmaBuffer::Grant(u32 pid)
{
    s32 tgid = 0;
    if (HrtDeviceGetBareTgid(tgid) != HCCL_SUCCESS) {
        THROW<InternalException>("HrtDeviceGetBareTgid failed");
    }
    u32 myPid = static_cast<u32>(tgid);
    if (pid == myPid) {
        HCCL_INFO("pid is equal to myPid, do need to use HrtIpcSetMemoryPid grant, pid==myPid=%u", pid);
        return;
    }
    CHK_RET_THROW(
        InternalException, "[Grant] HrtIpcSetMemoryPid failed", HrtIpcSetMemoryPid(name, static_cast<s32>(pid)));
}

} // namespace Hccl
