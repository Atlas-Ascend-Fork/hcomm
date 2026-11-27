/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file stuck_detect_counter_pool.h
 * @brief 卡住检测计数池设备内存和 slot 管理。
 */

#ifndef STUCK_DETECT_COUNTER_POOL_H
#define STUCK_DETECT_COUNTER_POOL_H

#include <cstdint>
#include <vector>

#include "hccl/hccl_types.h"
#include "stuck_detect_counter_types.h"

namespace hcomm {

/**
 * @brief 管理计数池设备内存和 slot 分配。
 *
 * 只负责设备内存、slot 分配/回收、地址计算和 D2H 计数读取，不感知 commId 和卡住状态。
 * 所有 Locked 接口要求调用方持有 StuckDetectCounterManager::managerMutex_。
 */
class StuckDetectCounterPool {
public:
    static constexpr uint32_t POOL_SIZE_BYTES = 128 * 1024; // 计数池大小（字节）
    static constexpr uint32_t MAX_SLOT_NUM = POOL_SIZE_BYTES / StuckDetectCounterSlot::SLOT_SIZE; // 最大 slot 数

    explicit StuckDetectCounterPool(int32_t deviceLogicId);
    ~StuckDetectCounterPool();

    StuckDetectCounterPool(const StuckDetectCounterPool&) = delete;
    StuckDetectCounterPool& operator=(const StuckDetectCounterPool&) = delete;

    // 计数池是否已初始化。
    bool IsReadyLocked() const;

    // 分配或复用 slot，返回 slot 下标和设备地址。
    HcclResult AllocateSlotLocked(uint32_t& slotIdx, StuckDetectCounterAddress& address);

    // 回收 slot。
    void FreeSlotLocked(uint32_t slotIdx);

    // 清零 slot 的 head/tail。
    HcclResult ResetSlotLocked(uint32_t slotIdx) const;

    // 获取已有 slot 的设备地址。
    HcclResult GetCounterAddressLocked(uint32_t slotIdx, StuckDetectCounterAddress& address) const;

    // D2H 拷贝整个计数池。
    HcclResult CopyLatestCountersLocked(std::vector<int32_t>& counterBuf) const;

    // 释放设备内存并清空 host 侧存储状态；除析构路径外，调用方需持有 managerMutex_。
    void DeInitLocked();

private:
    // 初始化设备内存。
    HcclResult InitLocked();

    // 初始化 slot 的 head/tail/addOne。
    HcclResult InitSlotLocked(uint32_t slotIdx);

    // 获取 slot 在设备内存中的起始地址。
    uint8_t* GetSlotBaseLocked(uint32_t slotIdx) const;

    // 根据 slot 下标填充设备地址。
    void FillCounterAddressLocked(uint32_t slotIdx, StuckDetectCounterAddress& address) const;

    int32_t deviceLogicId_{-1};
    void* poolMem_{nullptr};
    bool isInitialized_{false};
    std::vector<uint32_t> freeSlotList_;
    uint32_t nextSlotIdx_{0};
};

} // namespace hcomm

#endif // STUCK_DETECT_COUNTER_POOL_H
