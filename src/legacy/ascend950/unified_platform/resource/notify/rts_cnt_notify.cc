/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rts_cnt_notify.h"
#include "exception_util.h"
#include "internal_exception.h"
#include "task.h"
#include "dev_capability.h"

namespace Hccl {

RtsCntNotify::RtsCntNotify() : deviceId(0), devPhyId(0), handle(nullptr), id(0)
{
    s32 devId = 0;
    if (HrtGetDevice(devId) != HCCL_SUCCESS) {
        THROW<InternalException>("[RtsCntNotify] HrtGetDevice failed");
    }
    deviceId = static_cast<u32>(devId);
    DevId devPhyIdOut = 0;
    if (HrtGetDevicePhyIdByUserDevId(devId, devPhyIdOut) != HCCL_SUCCESS) {
        THROW<InternalException>("[RtsCntNotify] HrtGetDevicePhyIdByUserDevId failed");
    }
    devPhyId = static_cast<u32>(devPhyIdOut);
    aclrtCntNotify cntHandle = nullptr;
    if (HrtCntNotifyCreate(deviceId, cntHandle) != HCCL_SUCCESS) {
        THROW<InternalException>("[RtsCntNotify] HrtCntNotifyCreate failed");
    }
    handle = cntHandle;
    if (HrtGetCntNotifyId(handle, id) != HCCL_SUCCESS) {
        THROW<InternalException>("[RtsCntNotify] HrtGetCntNotifyId failed");
    }
    HrtDevResInfo devResInfo;
    devResInfo.dieId = 0;
    devResInfo.procType = HrtDevResProcType::PROCESS_HCCP;
    devResInfo.resType = HrtDevResType::RES_TYPE_STARS_CNT_NOTIFY_BIT_WR;
    devResInfo.resId = id;
    devResInfo.flag = 0;
    HrtDevResAddrInfo resAddrInfo;
    if (HrtGetDevResAddress(devResInfo, resAddrInfo) != HCCL_SUCCESS) {
        THROW<InternalException>("[RtsCntNotify] HrtGetDevResAddress failed");
    }
    addr = resAddrInfo.address;
    size = DevCapability::GetInstance().GetNotifySize();
}

RtsCntNotify::~RtsCntNotify()
{
    HrtDevResInfo devResInfo;
    devResInfo.dieId = 0;
    devResInfo.procType = HrtDevResProcType::PROCESS_HCCP;
    devResInfo.resType = HrtDevResType::RES_TYPE_STARS_CNT_NOTIFY_BIT_WR;
    devResInfo.resId = id;
    devResInfo.flag = 0;
    HcclResult ret = HrtReleaseDevResAddress(devResInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[RtsCntNotify] HrtReleaseDevResAddress failed, ret=%d", ret);
    }
    ret = HrtCntNotifyDestroy(handle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[RtsCntNotify] HrtCntNotifyDestroy failed, ret=%d", ret);
    }
}

std::unique_ptr<BaseTask> RtsCntNotify::PostBits(u32 bitValue)
{
    return std::make_unique<TaskPostBits>(this, bitValue);
}

std::unique_ptr<BaseTask> RtsCntNotify::WaitValue(u32 value) { return std::make_unique<TaskWaitValue>(this, value); }

void RtsCntNotify::PostBits(u32 bitValue, const Stream& stream) const
{
    CHK_RET_THROW(
        InternalException, "[RtsCntNotify::PostBits] HrtCntNotifyRecord failed",
        HrtCntNotifyRecord(handle, stream.GetPtr(), HrtCntNotifyRecordMode::WRITE_BIT, bitValue));
}

void RtsCntNotify::WaitValue(u32 value, u32 timeout, const aclrtStream& rtStream) const
{
    CHK_RET_THROW(
        InternalException, "[RtsCntNotify::WaitValue] HrtCntNotifyWaitWithTimeOut failed",
        HrtCntNotifyWaitWithTimeOut(handle, rtStream, HrtCntNotifyWaitMode::EQUAL, value, timeout));
}

void RtsCntNotify::WaitValue(u32 value, u32 timeout, const Stream& stream) const
{
    WaitValue(value, timeout, stream.GetPtr());
}

std::string RtsCntNotify::Describe() const
{
    return StringFormat("RtsCntNotify[handle=%p, id=%u, addr=0x%llx, size=%u]", handle, id, addr, size);
}

std::vector<char> RtsCntNotify::GetUniqueId() const
{
    BinaryStream binaryStream;
    binaryStream << id;
    binaryStream << devPhyId;
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

} // namespace Hccl
