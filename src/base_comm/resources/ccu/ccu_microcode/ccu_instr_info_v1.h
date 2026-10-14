/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_INSTR_INFO_H
#define CCU_INSTR_INFO_H

#include <vector>
#include "ccu_microcode_v1.h"

namespace hcomm {
namespace CcuRep {

    struct CcuInstrInfo {
        std::vector<CcuInstr> instrVec;
        uint16_t startInstrId{0};
        uint16_t instrCount{0};
        uint16_t missionStartInstrId{0};
        uint16_t missionInstrCount{0};
        // 是否经过后端优化 (V2). 由 MicrocodeOptimizer::Optimize 置真, 用于 DFX 区分:
        //   * false (未经优化): optToRawLocalIndex 为空属预期, 坐标直通;
        //   * true  (已优化) 但 optToRawLocalIndex 为空: 映射异常丢失 (不变式被破坏),
        //     DFX 侧需告警而非静默当作直通。
        bool backendOptimized{false};
        // 后端优化 (V2) 产出的"优化后本地下标 -> 优化前本地下标"映射:
        // optToRawLocalIndex[i] 表示优化后第 i 条指令来源于优化前的第几条 (本地下标, 相对 startInstrId);
        // -1 表示该条为优化过程新插入的 NOP (无优化前来源). 为空表示未经后端优化 (V1 直通 / 优化关闭),
        // 此时优化后坐标即优化前坐标. 供 DFX 把硬件 currentIns (优化后坐标) 翻译回 Rep 区间 (优化前坐标).
        std::vector<int32_t> optToRawLocalIndex{};
    };

}; // namespace CcuRep
}; // namespace hcomm

#endif // _CCU_INSTR_INFO_H
