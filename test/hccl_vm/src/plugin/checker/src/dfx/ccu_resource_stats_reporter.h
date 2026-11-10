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
 * Description: CCU 资源统计 CSV 报告器（dfx）。
 * 从 Collector 提取 per-op 统计数据，生成 16 列 CSV（见 docs/ccu_resource_statistics_design.md 4.2）：
 *   superPodId, serverId, phyDevId, dieId, commName, commHash, opIter,
 *   channelNum, instructionNum, loopTemplateNum, loopNum,
 *   xnNum, gsaNum, ckeNum, msNum, missionNum
 * 本模块不依赖 Db/拓扑解析（拓扑标签由调用方构建传入，缺失时输出 -1）。
 */

#ifndef CHECKER_DFX_CCU_RESOURCE_STATS_REPORTER_H
#define CHECKER_DFX_CCU_RESOURCE_STATS_REPORTER_H

#include <cstdint>
#include <map>
#include <string>

#include "ccu_resource_stats_types.h"
#include "hccl_types.h"

namespace HcclSim {
namespace CcuResourceStats {

    // phyDevId 的拓扑标签；解析失败或缺失时保持 -1
    struct TopoLabel {
        int64_t superPodId{-1};
        int64_t serverId{-1};
        // boardId: POD形态(4层拓扑)下的board序号; server形态恒为0。
        // POD形态下phyDevId只在board内唯一, 必须结合boardId才能定位到具体卡
        int64_t boardId{-1};
    };
    using TopoLabelMap = std::map<uint32_t /*phyDevId*/, TopoLabel>;

    // 生成 CSV 文本：按算子分节显示（每节 = '#' 算子标识/信息注释行 + 列头 + 该算子各 (dev,die) 数据行，
    // 节间空行分隔）。op 节按 (opIter, commName, commHash) 排序，节内行按
    // (superPodId, serverId, phyDevId, dieId) 排序。opInfos 缺失的算子信息字段输出 NA。
    // gsaApplicable=false（A6 设备无 GSA）时 gsaNum 列输出 NA。纯函数，供单测使用。
    std::string BuildCcuResourceStatsCsv(
        const OpStatsMap& data, const std::map<OpKey, OpInfo>& opInfos, const TopoLabelMap& topoLabels,
        bool gsaApplicable);

    // 生成报告并写盘（自动创建父目录），输出 summary 日志。
    // 采集数据为空时仍写出空报告文件并打 WARN。
    HcclResult
    GenerateCcuResourceStatsReport(const std::string& csvPath, bool gsaApplicable, const TopoLabelMap& topoLabels);

} // namespace CcuResourceStats
} // namespace HcclSim

#endif // CHECKER_DFX_CCU_RESOURCE_STATS_REPORTER_H
