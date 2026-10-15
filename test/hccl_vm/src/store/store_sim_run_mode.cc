/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "store_sim_run_mode.h"
#include "sim_log.h"

#include "storage/table_access.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/sim_models.h"

namespace sim {
// 文件级缓存变量，供 ResetRunModeCache 重置
static bool g_cachedCheckOnly = false;
static bool g_cachedEstimate = false;
static bool g_cacheInitialized = false;

bool ProbeCheckOnlyMode()
{
    // 运行模式单行表:空条件单行查询(取首行)。
    auto selected = sim::runtime::Db::GetOneByPred<sim::runtime::RunModeConfig>(
        HcclSim::Storage::All<sim::runtime::RunModeConfig>());
    // 精确匹配 1: mode=2(estimate) 不得被判为 check-only，否则 aclrt 内存重定向策略会误生效
    return selected.ok() && selected->mode == RUN_MODE_CHECK_ONLY;
}

bool IsCheckOnlyMode()
{
    if (!g_cacheInitialized) {
        g_cachedCheckOnly = ProbeCheckOnlyMode();
        g_cachedEstimate = ProbeEstimateMode();
        g_cacheInitialized = true;
    }
    return g_cachedCheckOnly;
}

bool ProbeEstimateMode()
{
    auto selected = sim::runtime::Db::GetOneByPred<sim::runtime::RunModeConfig>(
        HcclSim::Storage::All<sim::runtime::RunModeConfig>());
    return selected.ok() && selected->mode == RUN_MODE_ESTIMATE;
}

bool IsEstimateMode()
{
    if (!g_cacheInitialized) {
        g_cachedCheckOnly = ProbeCheckOnlyMode();
        g_cachedEstimate = ProbeEstimateMode();
        g_cacheInitialized = true;
    }
    return g_cachedEstimate;
}

void ResetRunModeCache()
{
    g_cachedCheckOnly = ProbeCheckOnlyMode();
    g_cachedEstimate = ProbeEstimateMode();
    g_cacheInitialized = true;
    HCCL_VM_INFO("Run mode cache reset: check-only={}, estimate={}", g_cachedCheckOnly, g_cachedEstimate);
}

const char* ModeName(uint8_t mode)
{
    switch (mode) {
        case RUN_MODE_CHECK_ONLY:
            return "check-only";
        case RUN_MODE_ESTIMATE:
            return "estimate";
        case RUN_MODE_NORMAL:
        default:
            return "normal";
    }
}

std::string GetRunModePrompt(uint8_t mode)
{
    switch (mode) {
        case RUN_MODE_CHECK_ONLY:
            return "(chk-only)$> ";
        case RUN_MODE_ESTIMATE:
            return "(estimate)$> ";
        case RUN_MODE_NORMAL:
        default:
            return "(hvm)$> ";
    }
}

std::string GetRunModePrompt()
{
    auto selected = sim::runtime::Db::GetOneByPred<sim::runtime::RunModeConfig>(
        HcclSim::Storage::All<sim::runtime::RunModeConfig>());
    // 表为空视为 normal
    return GetRunModePrompt(selected.ok() ? selected->mode : RUN_MODE_NORMAL);
}
} // namespace sim
