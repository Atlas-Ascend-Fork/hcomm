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
 * Description: CCU 资源统计拓扑标签解析（dfx）。
 * 从 DB 读取 TopoMetaConfig 表项并解析拓扑（yaml / ranktable.json 双模式），
 * 构建 phyDevId → (superPodId, serverId) 反查表。
 * 独立成文件以隔离 runtime_state/Db 头文件依赖（reporter 保持纯函数可单测）。
 */

#ifndef CHECKER_DFX_CCU_RESOURCE_STATS_TOPO_H
#define CHECKER_DFX_CCU_RESOURCE_STATS_TOPO_H

#include "ccu_resource_stats_reporter.h"

namespace HcclSim {
namespace CcuResourceStats {

    // 解析 phyDevId → (superPodId, serverId) 拓扑标签。
    // 任一环节失败（表项缺失/文件不可达/解析失败）返回空表并打 WARN，
    // 报告端 superPodId/serverId 兜底输出 -1，不阻塞报告生成。
    TopoLabelMap BuildCcuStatsTopoLabels();

} // namespace CcuResourceStats
} // namespace HcclSim

#endif // CHECKER_DFX_CCU_RESOURCE_STATS_TOPO_H
