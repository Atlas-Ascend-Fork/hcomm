/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rts_1ton_cnt_notify.h"
#include "exception_util.h"
#include "internal_exception.h"
#include "task.h"

namespace Hccl {

Rts1ToNCntNotify::Rts1ToNCntNotify() : deviceId(0), devPhyId(0), handle(nullptr), id(0)
{
    s32 devId = 0;
    if (HrtGetDevice(devId) != HCCL_SUCCESS) {
        THROW<InternalException>("[Rts1ToNCntNotify] HrtGetDevice failed");
    }
    deviceId = static_cast<u32>(devId);
    DevId devPhyIdOut = 0;
    if (HrtGetDevicePhyIdByUserDevId(devId, devPhyIdOut) != HCCL_SUCCESS) {
        THROW<InternalException>("[Rts1ToNCntNotify] HrtGetDevicePhyIdByUserDevId failed");
    }
    devPhyId = static_cast<u32>(devPhyIdOut);
    aclrtCntNotify cntHandle = nullptr;
    if (HrtCntNotifyCreate(deviceId, cntHandle) != HCCL_SUCCESS) {
        THROW<InternalException>("[Rts1ToNCntNotify] HrtCntNotifyCreate failed");
    }
    handle = cntHandle;
    if (HrtGetCntNotifyId(handle, id) != HCCL_SUCCESS) {
        THROW<InternalException>("[Rts1ToNCntNotify] HrtGetCntNotifyId failed");
    }
}

Rts1ToNCntNotify::~Rts1ToNCntNotify()
{
    HcclResult ret = HrtCntNotifyDestroy(handle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Rts1ToNCntNotify] HrtCntNotifyDestroy failed, ret=%d", ret);
    }
}

std::unique_ptr<BaseTask> Rts1ToNCntNotify::WaitBits(u32 bitValue)
{
    return std::make_unique<TaskWaitBits>(this, bitValue);
}

std::unique_ptr<BaseTask> Rts1ToNCntNotify::PostValue(u32 value)
{
    return std::make_unique<TaskPostValue>(this, value);
}

void Rts1ToNCntNotify::WaitBits(u32 bitValue, u32 timeout, const Stream& stream) const
{
    CHK_RET_THROW(
        InternalException, "[Rts1ToNCntNotify::WaitBits] HrtCntNotifyWaitWithTimeOut failed",
        HrtCntNotifyWaitWithTimeOut(handle, stream.GetPtr(), HrtCntNotifyWaitMode::BITMAP, bitValue, timeout));
}

void Rts1ToNCntNotify::PostValue(u32 value, const aclrtStream& rtStream) const
{
    CHK_RET_THROW(
        InternalException, "[Rts1ToNCntNotify::PostValue] HrtCntNotifyRecord failed",
        HrtCntNotifyRecord(handle, rtStream, HrtCntNotifyRecordMode::STORE, value));
}

void Rts1ToNCntNotify::PostValue(u32 value, const Stream& stream) const { PostValue(value, stream.GetPtr()); }

std::string Rts1ToNCntNotify::Describe() const
{
    return StringFormat("Rts1ToNCntNotify[handle=%p, id=%d]", handle, id);
}

std::vector<char> Rts1ToNCntNotify::GetUniqueId() const
{
    BinaryStream binaryStream;
    binaryStream << id;
    binaryStream << devPhyId;
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

} // namespace Hccl
