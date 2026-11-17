/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_resource_stats_topo.h"

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "sim_log.h"

namespace HcclSim {
namespace CcuResourceStats {

    TopoLabelMap BuildCcuStatsTopoLabels()
    {
        TopoLabelMap labels;
        auto topoConfigResult = sim::runtime::Db::GetOneByPred<sim::runtime::TopoMetaConfig>(
            HcclSim::Storage::All<sim::runtime::TopoMetaConfig>());
        if (!topoConfigResult.ok()) {
            HCCL_VM_WARN("[CCU_RESOURCE_STATS] TopoMetaConfig not found (run mock-comm first); "
                         "superPodId/serverId fall back to -1");
            return labels;
        }
        const std::string topoFileName(
            topoConfigResult->file_name, ::strnlen(topoConfigResult->file_name, sizeof(topoConfigResult->file_name)));
        sim::runtime::CommConfigData commConfig;
        if (!sim::runtime::ParseCommConfigData(topoFileName, nullptr, commConfig)) {
            HCCL_VM_WARN(
                "[CCU_RESOURCE_STATS] failed to parse topo config, fileName={}; superPodId/serverId fall back to -1",
                topoFileName);
            return labels;
        }
        // 主路径：Device 表 id（= checker 的 deviceId）→ server_id → CommConfigServer.serverKey。
        // 注意 deviceId 与 physical_id 口径不同（实测 id=1..4 对应 physical_id=0..3），
        // 不能用 deviceId 直接匹配 phyIds，否则标签错位/缺失。
        // POD形态(4层拓扑)下一行Server即一个board, 故标签需带boardId才能定位到具体卡
        std::map<uint64_t, TopoLabel> labelByServerKey;
        for (const sim::runtime::CommConfigServer& server : commConfig.servers) {
            labelByServerKey[server.serverKey] = TopoLabel{server.podId, server.serverId, server.boardId};
        }
        auto devicesResult
            = sim::runtime::Db::GetByPred<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>());
        for (const sim::runtime::Device& device : devicesResult.value_or(std::vector<sim::runtime::Device>{})) {
            const auto labelIt = labelByServerKey.find(device.server_id);
            if (labelIt != labelByServerKey.end()) {
                labels[static_cast<uint32_t>(device.id)] = labelIt->second;
            }
        }

        // 兜底路径：phyIds 直接匹配（deviceId == phyDevId 的部署场景）；emplace 不覆盖主路径结果
        for (const sim::runtime::CommConfigServer& server : commConfig.servers) {
            for (const uint32_t phyId : server.phyIds) {
                labels.emplace(phyId, TopoLabel{server.podId, server.serverId, server.boardId});
            }
        }
        HCCL_VM_INFO("[CCU_RESOURCE_STATS] topo labels built, fileName={}, labelCount={}", topoFileName, labels.size());
        return labels;
    }

} // namespace CcuResourceStats
} // namespace HcclSim
