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
#include "rts_notify_v2.h"
#include "log.h"
#include "dev_capability.h"
#include "binary_stream.h"
namespace Hccl {

RtsNotify::RtsNotify(bool devUsed) : devPhyId(0), devUsed(devUsed)
{
    s32 deviceID = 0;
    if (HrtGetDevice(deviceID) != HCCL_SUCCESS) {
        THROW<InternalException>("[RtsNotify] HrtGetDevice failed");
    }
    DevId devPhyIdOut = 0;
    if (HrtGetDevicePhyIdByUserDevId(deviceID, devPhyIdOut) != HCCL_SUCCESS) {
        THROW<InternalException>("[RtsNotify] HrtGetDevicePhyIdByUserDevId failed");
    }
    devPhyId = static_cast<u32>(devPhyIdOut);

    if (devUsed) {
        aclrtNotify notify = nullptr;
        if (HrtNotifyCreateWithFlag(static_cast<u32>(deviceID), ACL_NOTIFY_DEVICE_USE_ONLY, notify) != HCCL_SUCCESS) {
            THROW<InternalException>("[RtsNotify] HrtNotifyCreateWithFlag failed");
        }
        handle = notify;
    } else {
        aclrtNotify notify = nullptr;
        if (HrtNotifyCreate(deviceID, notify) != HCCL_SUCCESS) {
            THROW<InternalException>("[RtsNotify] HrtNotifyCreate failed");
        }
        handle = notify;
    }

    if (HrtGetNotifyID(handle, id) != HCCL_SUCCESS) {
        THROW<InternalException>("[RtsNotify] HrtGetNotifyID failed");
    }
}

RtsNotify::~RtsNotify()
{
    HcclResult ret = HrtNotifyDestroy(handle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[RtsNotify] HrtNotifyDestroy failed, handle[%p], ret[%d].", handle, ret);
    }
}

std::string RtsNotify::SetIpcName() const
{
    char ipcName[RTS_IPC_MEM_NAME_LEN] = {0};
    CHK_RET_THROW(
        InternalException, "[RtsNotify::SetIpcName] HrtIpcSetNotifyName failed",
        HrtIpcSetNotifyName(handle, ipcName, RTS_IPC_MEM_NAME_LEN));
    return ipcName;
}

void RtsNotify::SetIpcPid(s32 pid) const
{
    CHK_RET_THROW(
        InternalException, "[RtsNotify::SetIpcPid] HrtSetIpcNotifyPid failed", HrtSetIpcNotifyPid(handle, pid));
}

u32 RtsNotify::GetId() const { return id; }

u64 RtsNotify::GetOffset() const
{
    u32 offset = 0;
    CHK_RET_THROW(
        InternalException, "[RtsNotify::GetOffset] HrtNotifyGetOffset failed", HrtNotifyGetOffset(handle, offset));
    return static_cast<u64>(offset);
}

u64 RtsNotify::GetHandleAddr() const { return ReinterpretAs<u64>(handle); }

u32 RtsNotify::GetSize() const { return DevCapability::GetInstance().GetNotifySize(); }

bool RtsNotify::IsDevUsed() const { return devUsed; }

u32 RtsNotify::GetDevPhyId() const { return devPhyId; }

std::vector<char> RtsNotify::GetUniqueId() const
{
    BinaryStream binaryStream;
    binaryStream << id;
    binaryStream << devPhyId;
    HCCL_INFO("[RtsNotify][GetUniqueId]id[%u] devPhyId[%u]", id, devPhyId);
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

void RtsNotify::Wait(const Stream& stream, u32 timeout) const
{
    CHK_RET_THROW(
        InternalException, "[RtsNotify::Wait] HrtNotifyWaitWithTimeOut failed",
        HrtNotifyWaitWithTimeOut(handle, stream.GetPtr(), timeout));
}

void RtsNotify::Post(const Stream& stream) const
{
    CHK_RET_THROW(
        InternalException, "[RtsNotify::Post] HrtNotifyRecord failed", HrtNotifyRecord(handle, stream.GetPtr()));
}

string RtsNotify::Describe() const
{
    return StringFormat("RtsNotify[devUsed=%d, id=%u, handle=%p]", devUsed, id, handle);
}

} // namespace Hccl
