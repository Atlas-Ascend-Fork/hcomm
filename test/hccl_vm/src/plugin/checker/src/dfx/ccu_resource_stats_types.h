/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Description: CCU 资源统计数据结构定义（dfx）。
 * 统计粒度：算子每次调用（OpKey） x device（phyDevId） x die（定位 CCU）。
 * 口径（详见 docs/ccu_resource_statistics_design.md）：
 *   - 寄存器/channel/mission：按 id 集合去重，同一资源被多条指令反复操作只计一次；
 *   - loop：按 LoopGroupKey 幂等记账，loopNum = n + (n - o) * m（静态展开实例数）；
 *   - 指令空间：主口径为本 op SQE 指令区间并集（opInstrIds），
 *     instrResidentNum 为 run 级快照参考值（CcuInstrResTab.instrCount）。
 */

#ifndef CHECKER_DFX_CCU_RESOURCE_STATS_TYPES_H
#define CHECKER_DFX_CCU_RESOURCE_STATS_TYPES_H

#include <cstdint>
#include <map>
#include <set>
#include <string>

namespace HcclSim {
namespace CcuResourceStats {

    // 算子一次调用标识，字段与 sim::operation::OpExecutionKey 一致（dfx 自持副本，避免引入依赖）
    struct OpKey {
        std::string commName;
        uint64_t commHash{0};
        uint32_t opIter{0};

        bool operator<(const OpKey& other) const
        {
            if (commName != other.commName) {
                return commName < other.commName;
            }
            if (opIter != other.opIter) {
                return opIter < other.opIter;
            }
            return commHash < other.commHash;
        }

        bool operator==(const OpKey& other) const
        {
            return commName == other.commName && commHash == other.commHash && opIter == other.opIter;
        }
    };

    // 算子展示信息（CSV 分节头附带，来源：CheckerParam + OpDetailTab.algorithmName）
    struct OpInfo {
        std::string collectiveType = "NA"; // 算子类型（HcclCmdTypeToString）
        std::string algName = "NA";        // 算法名（algorithmName 回填，空则 NA）
        std::string dataType = "NA";       // 数据类型（HcclDataTypeToString）
        uint64_t elementCount{0};          // 元素个数（dataCount）
        uint64_t inputBytes{0};            // 输入数据量（rank0 视角，CalcInputOutputSize）
        uint64_t outputBytes{0};           // 输出数据量（rank0 视角）
        std::string reduceType = "NA"; // 仅 reduce 类算子（ALLREDUCE/REDUCE/REDUCE_SCATTER/REDUCE_SCATTER_V），其余 NA
        uint32_t rankSize{0};          // 通信域 rank 数
    };

    // LoopGroup 幂等记账 key：指令空间 + SQE 参数确定后展开结果确定，
    // 同一 (mission, SQE, LoopGroup 指令) 重复展开不重复计数
    struct LoopGroupKey {
        uint16_t missionId{0};
        uint32_t instStartId{0};      // SQE 标识：同一 LoopGroup 指令可被不同 args 的 SQE 引用
        uint32_t loopGroupInstrId{0}; // LoopGroup 指令在指令空间中的绝对 id

        bool operator<(const LoopGroupKey& other) const
        {
            if (missionId != other.missionId) {
                return missionId < other.missionId;
            }
            if (instStartId != other.instStartId) {
                return instStartId < other.instStartId;
            }
            return loopGroupInstrId < other.loopGroupInstrId;
        }
    };

    struct LoopGroupRecord {
        uint32_t templateNum{0}; // n = loopInsCnt（loop 模板个数）
        uint32_t expandedNum{0}; // n + (n - o) * m（静态展开实例数）
    };

    // 单个 (op, device, die) 上的 CCU 资源统计
    struct CcuResStats {
        // ---- channel ----
        std::set<uint32_t> channelIds;    // 引用口径（主）：翻译期实际引用的 channelId
        std::set<uint32_t> newChannelIds; // 差分口径（参考）：本 op 新引入的 channelId

        // ---- 指令空间 ----
        uint32_t instrResidentNum{0};  // CcuInstrResTab.instrCount（run 级快照，参考值）
        std::set<uint32_t> opInstrIds; // 本 op SQE 指令区间并集（instructionNum 主口径）

        // ---- 寄存器（写入时记账，按 id 去重；A6 先地址->id 后入账）----
        std::set<uint16_t> xnIds;  // xnNum = size()
        uint16_t xnMaxId{0};       // 峰值水位（参考，暂不进 CSV）
        bool xnUsed{false};        // 区分"未使用"与"仅使用 id=0"
        std::set<uint16_t> gsaIds; // 仅 A5；A6 报告输出 NA
        std::set<uint16_t> ckeIds; // 以 ckeId 为粒度，不带 mask/bit 信息
        std::set<uint16_t> msIds;

        // ---- loop（幂等记账）----
        std::map<LoopGroupKey, LoopGroupRecord> loopGroups;

        // ---- mission ----
        std::set<uint16_t> missionIds;

        uint32_t LoopGroupNum() const { return static_cast<uint32_t>(loopGroups.size()); }

        uint32_t LoopTemplateNum() const
        {
            uint32_t total = 0;
            for (const auto& entry : loopGroups) {
                total += entry.second.templateNum;
            }
            return total;
        }

        uint32_t LoopNum() const
        {
            uint32_t total = 0;
            for (const auto& entry : loopGroups) {
                total += entry.second.expandedNum;
            }
            return total;
        }
    };

    using DieStatsMap = std::map<uint32_t /*dieId*/, CcuResStats>;
    using DevStatsMap = std::map<uint32_t /*deviceId(=phyDevId)*/, DieStatsMap>;
    using OpStatsMap = std::map<OpKey, DevStatsMap>;

    // channel 表快照：deviceId -> dieId -> channelId 集合（差分参考口径用）
    using ChannelTableSnapshot = std::map<uint32_t, std::map<uint32_t, std::set<uint32_t>>>;

} // namespace CcuResourceStats
} // namespace HcclSim

#endif // CHECKER_DFX_CCU_RESOURCE_STATS_TYPES_H
