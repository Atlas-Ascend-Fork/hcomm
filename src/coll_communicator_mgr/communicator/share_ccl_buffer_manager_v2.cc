/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "share_ccl_buffer_manager_v2.h"
#include "hccl_common_v2.h"
#include "log.h"
#include "orion_adapter_rts.h"
#include "exception_util.h"
#include "invalid_params_exception.h"

namespace Hccl {

ShareCclBufferManager& ShareCclBufferManager::GetInstance() { return GetInstance(static_cast<DevId>(HrtGetDevice())); }

ShareCclBufferManager& ShareCclBufferManager::GetInstance(DevId deviceLogicId)
{
    static ShareCclBufferManager instances[MAX_MODULE_DEVICE_NUM] = {};
    // 非法device不做静默回退，避免跨device复用同一片共享buffer
    if (deviceLogicId >= MAX_MODULE_DEVICE_NUM) {
        HCCL_ERROR("[ShareCclBufferManager][%s] invalid deviceLogicId[%u].", __func__, deviceLogicId);
        THROW<InvalidParamsException>(
            std::string("[ShareCclBufferManager] invalid deviceLogicId, please check the current device."));
    }
    return instances[deviceLogicId];
}

HcclResult ShareCclBufferManager::GetShareCclBuffer(
    const std::string& bufferName, u64 bufferSize, std::shared_ptr<DevBuffer>& cclBuffer)
{
    std::lock_guard<std::mutex> lock(lock_);
    auto iter = shareBufferMap_.find(bufferName);
    if (iter != shareBufferMap_.end()) {
        // cclbuffer引用计数+1
        cclBuffer = iter->second.lock();
        if (cclBuffer != nullptr) {
            if (bufferSizeMap_[bufferName] != bufferSize) {
                HCCL_ERROR(
                    "[ShareCclBufferManager][%s] bufferName[%s] size mismatch, init size[%llu], current size[%llu].",
                    __func__, bufferName.c_str(), bufferSizeMap_[bufferName], bufferSize);
                cclBuffer = nullptr;
                return HCCL_E_PARA;
            }
            HCCL_RUN_INFO(
                "[ShareCclBufferManager][%s] reuse share cclBuffer, bufferName[%s], addr[%llu], size[%llu], "
                "useCount[%ld].",
                __func__, bufferName.c_str(), static_cast<unsigned long long>(cclBuffer->GetAddr()), bufferSize,
                cclBuffer.use_count());
            return HCCL_SUCCESS;
        }
    }

    std::shared_ptr<DevBuffer> newBuffer;
    TRY_CATCH_RETURN(newBuffer = DevBuffer::CreateHugePageBuf(bufferSize););
    shareBufferMap_[bufferName] = newBuffer;
    bufferSizeMap_[bufferName] = bufferSize;
    cclBuffer = std::move(newBuffer);
    HCCL_RUN_INFO(
        "[ShareCclBufferManager][%s] create share cclBuffer, bufferName[%s], addr[%llu], size[%llu].", __func__,
        bufferName.c_str(), static_cast<unsigned long long>(cclBuffer->GetAddr()), bufferSize);
    return HCCL_SUCCESS;
}

HcclResult
ShareCclBufferManager::CheckCclBufferConflict(const std::string& commId, const std::string& bufferName, s32 streamId)
{
    if (bufferName.empty()) {
        return HCCL_SUCCESS;
    }

    std::lock_guard<std::mutex> lock(lock_);
    auto& commStreamMap = streamMap_[bufferName];
    // 仅不同通信域之间校验Stream一致性，同一通信域切换Stream由用户自行保证串行
    for (const auto& iter : commStreamMap) {
        if (iter.first != commId && iter.second != streamId) {
            HCCL_ERROR(
                "[ShareCclBufferManager][%s] bufferName[%s] is used by comm[%s] on streamId[%d], but comm[%s] uses "
                "streamId[%d]. Communicators sharing the same cclBuffer must be executed on the same stream.",
                __func__, bufferName.c_str(), iter.first.c_str(), iter.second, commId.c_str(), streamId);
            return HCCL_E_PARA;
        }
    }

    // 只在新增绑定或切换Stream时打印日志
    auto binding = commStreamMap.find(commId);
    if (binding == commStreamMap.end() || binding->second != streamId) {
        commStreamMap[commId] = streamId;
        HCCL_RUN_INFO(
            "[ShareCclBufferManager][%s] comm[%s] bind bufferName[%s] to streamId[%d].", __func__, commId.c_str(),
            bufferName.c_str(), streamId);
    }
    return HCCL_SUCCESS;
}

void ShareCclBufferManager::ReleaseCclBufferStreamBinding(const std::string& commId, const std::string& bufferName)
{
    if (bufferName.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(lock_);
    auto iter = streamMap_.find(bufferName);
    if (iter == streamMap_.end()) {
        return;
    }
    iter->second.erase(commId);
    if (iter->second.empty()) {
        streamMap_.erase(iter);
    }
}

} // namespace Hccl
