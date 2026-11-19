/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream.h"
#include "runtime_api_exception.h"
#include "log.h"
#include "exception_util.h"
#include "internal_exception.h"
#include "binary_stream.h"
namespace Hccl {

Stream::Stream(aclrtStream ptr, bool isMaster) : ptr(ptr), selfOwned(false), isMaster_(isMaster)
{
    s32 streamId = 0;
    CHK_RET_THROW(InternalException, "[Stream] HrtGetStreamId failed", HrtGetStreamId(ptr, streamId));
    id = static_cast<u32>(streamId);
    InitDevPhyId();
}

Stream::Stream(bool deviceUsed, bool isMaster) : selfOwned(true), devUsed(deviceUsed), isMaster_(isMaster)
{
    try {
        if (deviceUsed) {
            aclrtStream streamPtr = nullptr;
            if (HrtStreamCreateWithFlags(HCCL_STREAM_PRIORITY_HIGH, ACL_STREAM_DEVICE_USE_ONLY, streamPtr)
                != HCCL_SUCCESS) {
                throw RuntimeApiException("HrtStreamCreateWithFlags failed");
            }
            ptr = streamPtr;
            u32 sqIdOut = 0;
            if (HrtStreamGetSqId(ptr, sqIdOut) != HCCL_SUCCESS) {
                throw RuntimeApiException("HrtStreamGetSqId failed");
            }
            sqId = sqIdOut;
            u32 cqIdOut = 0;
            if (HrtStreamGetCqId(ptr, cqIdOut) != HCCL_SUCCESS) {
                throw RuntimeApiException("HrtStreamGetCqId failed");
            }
            cqId = cqIdOut;
        } else {
            aclrtStream streamPtr = nullptr;
            if (HrtStreamCreateWithFlags(
                    HCCL_STREAM_PRIORITY_HIGH, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC, streamPtr)
                != HCCL_SUCCESS) {
                throw RuntimeApiException("HrtStreamCreateWithFlags failed");
            }
            ptr = streamPtr;
            if (HrtStreamSetMode(ptr, STREAM_MODE_STOP_ON_FAILURE) != HCCL_SUCCESS) {
                throw RuntimeApiException("HrtStreamSetMode failed");
            }
        }
        s32 streamId = 0;
        if (HrtGetStreamId(ptr, streamId) != HCCL_SUCCESS) {
            throw RuntimeApiException("HrtGetStreamId failed");
        }
        id = static_cast<u32>(streamId);
        InitDevPhyId();
    } catch (HcclException& e) {
        HCCL_ERROR("Stream init failed: %s", e.what());
        HcclResult ret = HrtStreamDestroy(ptr);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[Stream] HrtStreamDestroy failed during cleanup, ret[%d]", ret);
        }
        throw;
    }
}

Stream::~Stream()
{
    try {
        if (selfOwned) {
            HcclResult ret = HrtStreamDestroy(ptr);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[~Stream] HrtStreamDestroy failed, ret[%d], streamId[%u]", ret, id);
            }
        }
    } catch (HcclException& e) {
        HCCL_ERROR("%s", e.what());
    } catch (std::exception& e) {
        HCCL_ERROR("%s", e.what());
    } catch (...) {
        HCCL_ERROR("Unknown Error occurs when destruct stream %d", id);
    }
}

void Stream::SetStmMode(u64 stmMode) { mode = stmMode; }

aclrtStream Stream::GetPtr() const { return ptr; }

u32 Stream::GetId() const { return id; }

u32 Stream::GetSqId() const { return sqId; }

bool Stream::IsMaster() const { return isMaster_; }

bool Stream::IsSelfOwned() const { return selfOwned; }

u64 Stream::GetMode() const { return mode; }

std::vector<char> Stream::GetUniqueId() const
{
    std::vector<char> result;

    BinaryStream binaryStream;
    binaryStream << id;
    binaryStream << sqId;
    binaryStream << devPhyId;
    binaryStream << cqId;
    binaryStream.Dump(result);
    HCCL_INFO("Stream::GetUniqueId:%s:data=%s", Describe().c_str(), Bytes2hex(result.data(), result.size()).c_str());
    return result;
}

std::string Stream::Describe() const
{
    return StringFormat(
        "Stream[ptr=%p, id=%u, sqId=%u, selfOwned=%u, devUsed=%d, devPhyId=%u]", ptr, id, sqId, selfOwned, devUsed,
        devPhyId);
}

void Stream::InitDevPhyId()
{
    s32 devId = 0;
    CHK_RET_THROW(InternalException, "[Stream::InitDevPhyId] HrtGetDevice failed", HrtGetDevice(devId));
    DevId devPhyIdOut = 0;
    CHK_RET_THROW(
        InternalException, "[Stream::InitDevPhyId] HrtGetDevicePhyIdByUserDevId failed",
        HrtGetDevicePhyIdByUserDevId(devId, devPhyIdOut));
    devPhyId = static_cast<u32>(devPhyIdOut);
}

} // namespace Hccl
