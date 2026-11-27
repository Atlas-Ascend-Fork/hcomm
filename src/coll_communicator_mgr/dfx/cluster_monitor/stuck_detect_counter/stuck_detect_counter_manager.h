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
 * @file stuck_detect_counter_manager.h
 * @brief 通信域卡住检测管理器。
 */

#ifndef STUCK_DETECT_COUNTER_MANAGER_H
#define STUCK_DETECT_COUNTER_MANAGER_H

#include <cstdint>
#include <map>
#include <shared_mutex>
#include <string>
#include <vector>

#include "hccl/hccl_types.h"
#include "stuck_detect_counter_pool.h"
#include "stuck_detect_counter_types.h"

namespace hcomm {

/**
 * @brief 卡住检测管理器。
 *
 * 管理通信域状态和卡住检测状态机；设备内存、slot 分配和 D2H 计数读取由 Pool 负责。
 * managerMutex_ 同时保护通信域状态和 Pool 的访问。
 */
class StuckDetectCounterManager {
public:
    // ---------- 生命周期 ----------
    // 初始化指定设备的计数管理器。
    explicit StuckDetectCounterManager(int32_t deviceLogicId);
    // 打印仍存活的通信域，并释放计数资源。
    ~StuckDetectCounterManager();

    StuckDetectCounterManager(const StuckDetectCounterManager&) = delete;
    StuckDetectCounterManager& operator=(const StuckDetectCounterManager&) = delete;

    void DeInit();

    // ---------- 通信域计数 ----------
    // 为通信域分配计数 slot，返回设备地址。
    HcclResult AllocCounter(const std::string& commId, StuckDetectCounterAddress& address);
    // 回收通信域计数 slot，并打印最终计数。
    HcclResult FreeCounter(const std::string& commId);
    // 重置通信域 head/tail 计数和卡住状态。
    HcclResult ResetCounter(const std::string& commId);

    // ---------- 卡住检测 ----------
    // 判断进程是否达到卡住上报阈值，并处理打印退避。
    HcclResult CheckProcessStuck(uint32_t reportThreshold, bool& needReportStuck);

private:
    // 在持有 managerMutex_ 的情况下更新各通信域状态并判断是否卡住。
    HcclResult CheckCounterStuckLocked(bool& isStuck);
    // 打印所有未析构通信域的计数和状态。
    void PrintActiveCommCounters() const;
    // 根据最新计数更新单个通信域的卡住状态。
    HcclResult
    UpdateCommStatusLocked(StuckDetectCounterInfo& counterInfo, const std::vector<int32_t>& counterBuf) const;
    // 在持锁状态下刷新单个通信域的计数，仅 D2H 失败时返回错误码。
    HcclResult RefreshCommCounterLocked(StuckDetectCounterInfo& counterInfo) const;
    // 打印单个通信域的计数和状态。
    void PrintCommCounter(const StuckDetectCounterInfo& counterInfo) const;

    // ---------- 数据成员 ----------
    // 设备内存和 slot 管理
    StuckDetectCounterPool counterPool_;

    // 并发保护
    mutable std::shared_mutex managerMutex_;

    // 通信域状态
    std::map<std::string, StuckDetectCounterInfo> commId2CounterInfo_; // commId -> 计数信息

    // 进程级卡住检测状态
    uint32_t counterStuckRounds_{0};
    uint32_t processStuckRounds_{0};
    uint32_t nextPrintRound_{1};
};

} // namespace hcomm

#endif // STUCK_DETECT_COUNTER_MANAGER_H
