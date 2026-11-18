/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stuck_detect_counter_pool.h"

#include <cstdint>

#include "adapter_rts_common.h"
#include "common/cast_utils_pub.h"
#include "log.h"

namespace hcomm {
namespace {
    constexpr int32_t ADD_ONE_INIT_VALUE = 1; // addOne 的初始常量

    uint64_t ToU64(const void* ptr) { return static_cast<uint64_t>(ReinterpretAs<uintptr_t>(ptr)); }
} // namespace

StuckDetectCounterPool::StuckDetectCounterPool(int32_t deviceLogicId) : deviceLogicId_(deviceLogicId) {}

StuckDetectCounterPool::~StuckDetectCounterPool() { DeInitLocked(); }

bool StuckDetectCounterPool::IsReadyLocked() const { return isInitialized_ && poolMem_ != nullptr; }

HcclResult StuckDetectCounterPool::InitLocked()
{
    // 已初始化时直接复用；否则 poolMem_ 可能是上次初始化失败留下的残留内存，先释放再重建。
    if (poolMem_ != nullptr) {
        if (isInitialized_) {
            return HCCL_SUCCESS;
        }
        HcclResult freeRet = hrtFree(poolMem_);
        if (freeRet != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] free stale pool failed, deviceLogicId[%d], ret[%d].", __func__, deviceLogicId_, freeRet);
            return freeRet;
        }
        poolMem_ = nullptr;
    }
    isInitialized_ = false;

    // 预留最大 slot 数，避免 FreeCounter 回收 slot 时扩容。
    EXCEPTION_CATCH(freeSlotList_.reserve(MAX_SLOT_NUM), return HCCL_E_MEMORY);

    HcclResult ret = hrtMalloc(&poolMem_, POOL_SIZE_BYTES);
    if (ret != HCCL_SUCCESS || poolMem_ == nullptr) {
        poolMem_ = nullptr;
        HCCL_WARNING("[%s] hrtMalloc failed, deviceLogicId[%d], ret[%d].", __func__, deviceLogicId_, ret);
        return ret == HCCL_SUCCESS ? HCCL_E_MEMORY : ret;
    }

    // 整池清零，保证新分配 slot 的 head/tail 初值为 0。
    ret = hrtMemSet(poolMem_, POOL_SIZE_BYTES, POOL_SIZE_BYTES);
    if (ret != HCCL_SUCCESS) {
        HcclResult freeRet = hrtFree(poolMem_);
        if (freeRet != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[%s] hrtMemSet failed and hrtFree failed, deviceLogicId[%d], "
                "ret[%d], freeRet[%d].",
                __func__, deviceLogicId_, ret, freeRet);
            return freeRet;
        }
        poolMem_ = nullptr;
        HCCL_WARNING("[%s] hrtMemSet failed, deviceLogicId[%d], ret[%d].", __func__, deviceLogicId_, ret);
        return ret;
    }

    nextSlotIdx_ = 0;
    freeSlotList_.clear();
    isInitialized_ = true;

    HCCL_RUN_INFO("[%s] alloc pool success, deviceLogicId[%d].", __func__, deviceLogicId_);
    return HCCL_SUCCESS;
}

char* StuckDetectCounterPool::GetSlotBaseLocked(uint32_t slotIdx) const
{
    return static_cast<char*>(poolMem_) + slotIdx * StuckDetectCounterSlot::SLOT_SIZE;
}

HcclResult StuckDetectCounterPool::InitSlotLocked(uint32_t slotIdx)
{
    // slot 可能被复用，分配时必须同时重置 head/tail，并恢复 addOne 常量。
    int32_t initValue[4] = {0, 0, ADD_ONE_INIT_VALUE, 0};
    void* slotBase = GetSlotBaseLocked(slotIdx);
    HcclResult ret = hrtMemSyncCopy(
        slotBase, StuckDetectCounterSlot::SLOT_SIZE, initValue, StuckDetectCounterSlot::SLOT_SIZE,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    if (ret != HCCL_SUCCESS) {
        freeSlotList_.push_back(slotIdx);
        HCCL_WARNING("[%s] init slot failed, slot[%u], ret[%d].", __func__, slotIdx, ret);
        return ret;
    }
    return HCCL_SUCCESS;
}

void StuckDetectCounterPool::FillCounterAddressLocked(uint32_t slotIdx, StuckDetectCounterAddress& address) const
{
    // slot 固定 16B，head/tail/addOne 在同一 slot 内，按固定偏移计算设备地址。
    char* slotBase = GetSlotBaseLocked(slotIdx);
    address.headAddr = ToU64(slotBase + StuckDetectCounterSlot::HEAD_OFFSET);
    address.tailAddr = ToU64(slotBase + StuckDetectCounterSlot::TAIL_OFFSET);
    address.addOneAddr = ToU64(slotBase + StuckDetectCounterSlot::ADD_ONE_OFFSET);
}

HcclResult StuckDetectCounterPool::AllocateSlotLocked(uint32_t& slotIdx, StuckDetectCounterAddress& address)
{
    if (!IsReadyLocked()) {
        HcclResult ret = InitLocked();
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
    }

    // 优先复用已释放 slot，避免通信域反复创建时 pool 水位持续上涨。
    if (!freeSlotList_.empty()) {
        slotIdx = freeSlotList_.back();
        freeSlotList_.pop_back();
    } else if (nextSlotIdx_ < MAX_SLOT_NUM) {
        slotIdx = nextSlotIdx_++;
    } else {
        HCCL_WARNING("[%s] pool exhausted, allocated[%u], max[%u].", __func__, nextSlotIdx_, MAX_SLOT_NUM);
        return HCCL_E_MEMORY;
    }

    HcclResult ret = InitSlotLocked(slotIdx);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    FillCounterAddressLocked(slotIdx, address);
    return HCCL_SUCCESS;
}

void StuckDetectCounterPool::FreeSlotLocked(uint32_t slotIdx) { freeSlotList_.push_back(slotIdx); }

HcclResult StuckDetectCounterPool::ResetSlotLocked(uint32_t slotIdx)
{
    if (!IsReadyLocked()) {
        return HCCL_E_PTR;
    }

    // addOne 是只读常量，重置时只需要清 head/tail。
    int32_t zeroValue[2] = {0, 0};
    void* slotBase = GetSlotBaseLocked(slotIdx);
    HcclResult ret = hrtMemSyncCopy(
        slotBase, sizeof(zeroValue), zeroValue, sizeof(zeroValue),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[%s] reset head/tail failed, slot[%u], ret[%d].", __func__, slotIdx, ret);
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult StuckDetectCounterPool::GetCounterAddressLocked(uint32_t slotIdx, StuckDetectCounterAddress& address) const
{
    if (!IsReadyLocked()) {
        return HCCL_E_PTR;
    }
    FillCounterAddressLocked(slotIdx, address);
    return HCCL_SUCCESS;
}

HcclResult StuckDetectCounterPool::CopyLatestCountersLocked(std::vector<int32_t>& counterBuf) const
{
    if (!IsReadyLocked()) {
        return HCCL_E_PTR;
    }

    // 一次 D2H 拷贝整个 pool，后续按 slotIdx 从计数缓存中读取各通信域计数。
    EXCEPTION_CATCH(counterBuf.resize(POOL_SIZE_BYTES / sizeof(int32_t)), return HCCL_E_MEMORY);
    HcclResult ret = hrtMemSyncCopy(
        counterBuf.data(), POOL_SIZE_BYTES, poolMem_, POOL_SIZE_BYTES,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[%s] D2H copy failed, ret[%d].", __func__, ret);
        return ret;
    }
    return HCCL_SUCCESS;
}

void StuckDetectCounterPool::DeInitLocked()
{
    if (poolMem_ != nullptr) {
        HcclResult ret = hrtFree(poolMem_);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] hrtFree failed, ret[%d].", __func__, ret);
        }
        poolMem_ = nullptr;
    }

    nextSlotIdx_ = 0;
    freeSlotList_.clear();
    isInitialized_ = false;
}

} // namespace hcomm
