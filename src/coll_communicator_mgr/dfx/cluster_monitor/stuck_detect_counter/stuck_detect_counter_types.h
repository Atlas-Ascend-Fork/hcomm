/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STUCK_DETECT_COUNTER_TYPES_H
#define STUCK_DETECT_COUNTER_TYPES_H

#include <cstdint>
#include <string>

namespace hcomm {

// 单个通信域的卡住状态。
enum class StuckDetectCommStatus {
    // 计数尚未出现非零推进。
    NEVER_ADVANCED,
    // 计数仍在推进。
    ACTIVE,
    // head/tail 出现过非0且当前未推进。
    FROZEN,
};

inline const char* CommStatusToString(StuckDetectCommStatus status)
{
    switch (status) {
        case StuckDetectCommStatus::NEVER_ADVANCED:
            return "NEVER_ADVANCED";
        case StuckDetectCommStatus::ACTIVE:
            return "ACTIVE";
        case StuckDetectCommStatus::FROZEN:
            return "FROZEN";
        default:
            return "UNKNOWN";
    }
}

// ---------- slot 布局 ----------
/**
 * @brief 单个通信域的计数器 slot 布局。
 *
 * slot 固定 16B，head/tail 位于同一 slot，简化地址计算：
 * slotBase + {0,4,8,12}。
 */
struct StuckDetectCounterSlot {
    /// slot 总大小，单位字节。
    static constexpr uint32_t SLOT_SIZE = 16;
    /// head 计数器在 slot 内的偏移。
    static constexpr uint32_t HEAD_OFFSET = 0;
    /// tail 计数器在 slot 内的偏移。
    static constexpr uint32_t TAIL_OFFSET = 4;
    /// 常量 1 在 slot 内的偏移，作为 reduce 的源操作数。
    static constexpr uint32_t ADD_ONE_OFFSET = 8;
    /// 保留字段，保持 16B 对齐。
    static constexpr uint32_t RESERVED_OFFSET = 12;
};

// ---------- 地址信息 ----------
/**
 * @brief 单个通信域计数器地址信息。
 *
 * 地址均为 device 地址。
 */
struct StuckDetectCounterAddress {
    /// head 计数器 device 地址。
    uint64_t headAddr = 0;
    /// tail 计数器 device 地址。
    uint64_t tailAddr = 0;
    /// 常量 1 的 device 地址。
    uint64_t addOneAddr = 0;
};

// ---------- 计数信息 ----------
/**
 * @brief 单个通信域的计数信息和状态。
 */
struct StuckDetectCounterInfo {
    /// 通信域标识，仅用于日志打印；状态索引使用 map key。
    std::string commId;
    /// head 计数。
    int32_t head = 0;
    /// tail 计数。
    int32_t tail = 0;
    /// 当前通信域的卡住判定状态。
    StuckDetectCommStatus status{StuckDetectCommStatus::NEVER_ADVANCED};
    /// 计数池内部 slot 序号，不打印。
    uint32_t slotIdx = 0;
};

} // namespace hcomm

#endif // STUCK_DETECT_COUNTER_TYPES_H
