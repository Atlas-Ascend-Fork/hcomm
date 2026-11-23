/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ipc_remote_notify.h"
#include "internal_exception.h"
#include "exchange_ipc_notify_dto.h"
#include "dev_capability.h"
namespace Hccl {

IpcRemoteNotify::IpcRemoteNotify() : BaseRemoteNotify(RmaType::IPC) {}

IpcRemoteNotify::~IpcRemoteNotify()
{
    if (handleOwned) {
        HcclResult ret = HrtNotifyDestroy(handle);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[IpcRemoteNotify] HrtNotifyDestroy failed, handle[%p], ret[%d].", handle, ret);
        }
    }
}

IpcRemoteNotify::IpcRemoteNotify(const Serializable& rmtDto) : BaseRemoteNotify(RmaType::IPC)
{
    auto dto = dynamic_cast<const ExchangeIpcNotifyDto&>(rmtDto);
    rmtPid = dto.pid;
    handleAddr = dto.handleAddr;
    id = dto.id;
    devUsed = dto.devUsed;
    rmtDevPhyId = dto.devPhyId;
    (void)memcpy_s(name, RTS_IPC_MEM_NAME_LEN, dto.name, RTS_IPC_MEM_NAME_LEN);

    // OpenIpc
    s32 tgid = 0;
    if (HrtDeviceGetBareTgid(tgid) != HCCL_SUCCESS) {
        THROW<InternalException>("HrtDeviceGetBareTgid failed");
    }
    u32 myPid = static_cast<u32>(tgid);
    if (rmtPid == myPid) {
        handle = reinterpret_cast<void*>(handleAddr);
    } else {
        if (devUsed) {
            RtNotify_t notify = nullptr;
            if (HrtIpcOpenNotifyWithFlag(name, RT_NOTIFY_FLAG_DOWNLOAD_TO_DEV, notify) != HCCL_SUCCESS) {
                THROW<InternalException>("HrtIpcOpenNotifyWithFlag failed");
            }
            handle = notify;
        } else {
            RtNotify_t notify = nullptr;
            if (HrtIpcOpenNotify(name, notify) != HCCL_SUCCESS) {
                THROW<InternalException>("HrtIpcOpenNotify failed");
            }
            handle = notify;
        }
        handleOwned = true;
    }
    u64 addrOut = 0;
    if (HrtNotifyGetAddr(handle, addrOut) != HCCL_SUCCESS) {
        if (handleOwned) {
            HcclResult ret = HrtNotifyDestroy(handle);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[IpcRemoteNotify] HrtNotifyDestroy failed, handle[%p], ret[%d].", handle, ret);
            }
            handleOwned = false;
        }
        THROW<InternalException>("HrtNotifyGetAddr failed");
    }
    addr = addrOut;
    size = DevCapability::GetInstance().GetNotifySize();
    HCCL_INFO(
        "IpcRemoteNotify[name=%s, handleAddr=0x%llx, id=%u, rmtPid=%u, rmtDevPhyId=%u, devUsed=%d, addr=0x%llx, "
        "handle=%p]",
        name, handleAddr, id, rmtPid, rmtDevPhyId, devUsed, addr, handle);
}

void IpcRemoteNotify::Post(const Stream& stream) const
{
    CHK_RET_THROW(
        InternalException, "[IpcRemoteNotify::Post] HrtNotifyRecord failed", HrtNotifyRecord(handle, stream.GetPtr()));
}

string IpcRemoteNotify::Describe() const
{
    return StringFormat(
        "IpcRemoteNotify[name=%s, handleAddr=0x%llx, id=%u, rmtPid=%u, rmtDevPhyId=%u, devUsed=%d, "
        "handle=%p, addr=0x%llx, size=%llu]",
        name, handleAddr, id, rmtPid, rmtDevPhyId, devUsed, handle, addr, size);
}

} // namespace Hccl
