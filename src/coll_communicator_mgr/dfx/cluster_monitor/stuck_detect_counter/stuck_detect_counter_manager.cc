/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stuck_detect_counter_manager.h"

#include <cstdint>
#include <mutex>
#include <utility>

#include "log.h"

namespace hcomm {

namespace {
    // 卡住日志打印的退避系数：每打印一次，下一轮打印阈值翻倍，避免持续卡住时刷日志。
    constexpr uint32_t PRINT_ROUND_BACKOFF = 2;
} // namespace

// ---------- 生命周期 ----------

StuckDetectCounterManager::StuckDetectCounterManager(int32_t deviceLogicId) : counterPool_(deviceLogicId) {}

StuckDetectCounterManager::~StuckDetectCounterManager() { DeInit(); }

void StuckDetectCounterManager::DeInit()
{
    // 打印可能因内存分配失败抛异常；析构路径不能向外抛，失败后继续释放资源。
    EXCEPTION_CATCH(PrintActiveCommCounters(), HCCL_WARNING("[%s] print active counters failed.", __func__));

    std::unique_lock<std::shared_mutex> lock(managerMutex_);
    counterPool_.DeInitLocked();
    commId2CounterInfo_.clear();
}

// ---------- 通信域计数 ----------

HcclResult StuckDetectCounterManager::AllocCounter(const std::string& commId, StuckDetectCounterAddress& address)
{
    // 常见路径：通信域已分配过，共享锁下直接返回地址，避免阻塞检测线程。
    {
        std::shared_lock<std::shared_mutex> readLock(managerMutex_);
        auto iter = commId2CounterInfo_.find(commId);
        if (counterPool_.IsReadyLocked() && iter != commId2CounterInfo_.end()) {
            return counterPool_.GetCounterAddressLocked(iter->second.slotIdx, address);
        }
    }

    // 未命中时升级为唯一锁，完成 pool 初始化和 slot 分配。
    std::unique_lock<std::shared_mutex> lock(managerMutex_);
    auto iter = commId2CounterInfo_.find(commId);
    if (counterPool_.IsReadyLocked() && iter != commId2CounterInfo_.end()) {
        return counterPool_.GetCounterAddressLocked(iter->second.slotIdx, address);
    }

    // pool 未就绪时，host 侧旧状态已经失效，先清空再重新分配。
    if (!counterPool_.IsReadyLocked()) {
        commId2CounterInfo_.clear();
    }

    uint32_t slotIdx = 0;
    HcclResult ret = counterPool_.AllocateSlotLocked(slotIdx, address);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    // 唯一锁内已确认通信域不存在，直接插入计数信息。
    commId2CounterInfo_.try_emplace(
        commId, StuckDetectCounterInfo{commId, 0, 0, StuckDetectCommStatus::NEVER_ADVANCED, slotIdx});

    HCCL_RUN_INFO("[%s] commId[%s] slot[%u] allocated.", __func__, commId.c_str(), slotIdx);
    return HCCL_SUCCESS;
}

HcclResult StuckDetectCounterManager::FreeCounter(const std::string& commId)
{
    // 锁内只做状态迁移和 slot 回收；计数打印移到锁外，避免日志阻塞控制面。
    StuckDetectCounterInfo counterInfo;
    {
        std::unique_lock<std::shared_mutex> lock(managerMutex_);
        auto iter = commId2CounterInfo_.find(commId);
        if (iter == commId2CounterInfo_.end()) {
            return HCCL_SUCCESS;
        }

        counterInfo = std::move(iter->second);
        counterPool_.FreeSlotLocked(counterInfo.slotIdx);
        commId2CounterInfo_.erase(iter);
    }
    PrintCommCounter(counterInfo);
    HCCL_DEBUG("[%s] commId[%s] counter slot recycled.", __func__, commId.c_str());
    return HCCL_SUCCESS;
}

HcclResult StuckDetectCounterManager::ResetCounter(const std::string& commId)
{
    // addOne 是只读常量，重置时只需清 head/tail 和 host 侧状态。
    std::unique_lock<std::shared_mutex> lock(managerMutex_);
    auto iter = commId2CounterInfo_.find(commId);
    if (iter == commId2CounterInfo_.end()) {
        return HCCL_SUCCESS;
    }

    HcclResult ret = counterPool_.ResetSlotLocked(iter->second.slotIdx);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[%s] reset slot failed, commId[%s], ret[%d].", __func__, commId.c_str(), ret);
        return ret;
    }

    iter->second.head = 0;
    iter->second.tail = 0;
    iter->second.status = StuckDetectCommStatus::NEVER_ADVANCED;
    HCCL_RUN_INFO("[%s] commId[%s] counter reset.", __func__, commId.c_str());
    return HCCL_SUCCESS;
}

// ---------- 卡住检测 ----------

HcclResult StuckDetectCounterManager::UpdateCommStatusLocked(
    StuckDetectCounterInfo& counterInfo, const std::vector<int32_t>& counterBuf) const
{
    if (counterInfo.slotIdx >= StuckDetectCounterPool::MAX_SLOT_NUM) {
        HCCL_WARNING(
            "[%s] invalid slotIdx[%u], commId[%s].", __func__, counterInfo.slotIdx, counterInfo.commId.c_str());
        return HCCL_E_INTERNAL;
    }

    const std::size_t base
        = static_cast<std::size_t>(counterInfo.slotIdx) * StuckDetectCounterSlot::SLOT_SIZE / sizeof(int32_t);
    const std::size_t headIndex = base + StuckDetectCounterSlot::HEAD_OFFSET / sizeof(int32_t);
    const std::size_t tailIndex = base + StuckDetectCounterSlot::TAIL_OFFSET / sizeof(int32_t);
    if (headIndex >= counterBuf.size() || tailIndex >= counterBuf.size()) {
        HCCL_WARNING(
            "[%s] counter buffer too small, commId[%s], slotIdx[%u], size[%zu], headIndex[%zu], tailIndex[%zu].",
            __func__, counterInfo.commId.c_str(), counterInfo.slotIdx, counterBuf.size(), headIndex, tailIndex);
        return HCCL_E_INTERNAL;
    }

    const int32_t newHead = counterBuf[headIndex];
    const int32_t newTail = counterBuf[tailIndex];

    const int32_t prevHead = counterInfo.head;
    const int32_t prevTail = counterInfo.tail;

    counterInfo.head = newHead;
    counterInfo.tail = newTail;

    switch (counterInfo.status) {
        case StuckDetectCommStatus::NEVER_ADVANCED:
            // 首次出现非零计数进入 ACTIVE；持续全零不参与判定。
            if (newHead != 0 || newTail != 0) {
                counterInfo.status = StuckDetectCommStatus::ACTIVE;
            }
            break;
        case StuckDetectCommStatus::ACTIVE:
        case StuckDetectCommStatus::FROZEN: {
            // head 不推进，且前后两轮都没有在途算子时判定冻结。
            const bool isFrozen = (newHead == prevHead) && (newHead == prevTail) && (newHead == newTail);
            counterInfo.status = isFrozen ? StuckDetectCommStatus::FROZEN : StuckDetectCommStatus::ACTIVE;
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult StuckDetectCounterManager::CheckCounterStuckLocked(bool& isStuck)
{
    isStuck = false;

    if (!counterPool_.IsReadyLocked() || commId2CounterInfo_.empty()) {
        return HCCL_SUCCESS;
    }

    std::vector<int32_t> counterBuf;
    HcclResult ret = counterPool_.CopyLatestCountersLocked(counterBuf);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[%s] copy counters failed, ret[%d].", __func__, ret);
        return ret;
    }

    bool hasFrozenComm = false;
    bool hasActiveComm = false;

    for (auto& item : commId2CounterInfo_) {
        StuckDetectCounterInfo& counterInfo = item.second;
        ret = UpdateCommStatusLocked(counterInfo, counterBuf);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[%s] update comm status failed, commId[%s], ret[%d].", __func__, counterInfo.commId.c_str(), ret);
            return ret;
        }

        if (counterInfo.status == StuckDetectCommStatus::FROZEN) {
            hasFrozenComm = true;
        } else if (counterInfo.status == StuckDetectCommStatus::ACTIVE) {
            hasActiveComm = true;
        }
    }

    // 进程级 STUCK 条件：1. 至少存在一个 FROZEN 通信域；2. 不存在 ACTIVE 通信域。
    // 全部通信域为 NEVER_ADVANCED 时返回 false，避免初始化阶段算子编译耗时太长出现误报。
    isStuck = hasFrozenComm && !hasActiveComm;
    return HCCL_SUCCESS;
}

HcclResult StuckDetectCounterManager::CheckProcessStuck(uint32_t reportThreshold, bool& needReportStuck)
{
    needReportStuck = false;
    if (reportThreshold == 0) {
        HCCL_WARNING("[%s] invalid report threshold[0].", __func__);
        return HCCL_E_PARA;
    }

    bool needPrint = false;
    {
        std::unique_lock<std::shared_mutex> lock(managerMutex_);

        bool isStuck = false;
        HcclResult ret = CheckCounterStuckLocked(isStuck);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] check counter stuck failed, ret[%d].", __func__, ret);
            return ret;
        }
        if (!isStuck) {
            counterStuckRounds_ = 0;
            processStuckRounds_ = 0;
            nextPrintRound_ = 1;
            return HCCL_SUCCESS;
        }

        ++counterStuckRounds_;
        if (counterStuckRounds_ >= reportThreshold) {
            needReportStuck = true;
            ++processStuckRounds_;
            if (processStuckRounds_ >= nextPrintRound_) {
                needPrint = true;
                nextPrintRound_ *= PRINT_ROUND_BACKOFF;
            }
        } else {
            needReportStuck = false;
            processStuckRounds_ = 0;
            nextPrintRound_ = 1;
            return HCCL_SUCCESS;
        }
    }

    if (needPrint) {
        EXCEPTION_CATCH(PrintActiveCommCounters(), HCCL_WARNING("[%s] print active counters failed.", __func__));
    }
    return HCCL_SUCCESS;
}

// ---------- 状态打印与清理 ----------

void StuckDetectCounterManager::PrintActiveCommCounters() const
{
    // 锁内只拷贝状态，锁外打印，避免日志 I/O 拖长控制面持锁时间。
    std::vector<StuckDetectCounterInfo> counters;
    {
        std::shared_lock<std::shared_mutex> lock(managerMutex_);
        counters.reserve(commId2CounterInfo_.size());
        for (const auto& item : commId2CounterInfo_) {
            counters.push_back(item.second);
        }
    }

    for (const auto& counterInfo : counters) {
        PrintCommCounter(counterInfo);
    }
}

void StuckDetectCounterManager::PrintCommCounter(const StuckDetectCounterInfo& counterInfo) const
{
    HCCL_RUN_INFO(
        "[%s] commId[%s], head[%d], tail[%d], status[%s].", __func__, counterInfo.commId.c_str(), counterInfo.head,
        counterInfo.tail, CommStatusToString(counterInfo.status));
}

} // namespace hcomm
