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

#include "storage/table_access.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/sim_models.h"

namespace sim {
bool ProbeCheckOnlyMode()
{
    // 运行模式单行表:空条件单行查询(取首行)。
    auto selected = sim::runtime::Db::GetOneByPred<sim::runtime::RunModeConfig>(
        HcclSim::Storage::All<sim::runtime::RunModeConfig>());
    return selected.ok() && selected->mode != 0;
}

bool IsCheckOnlyMode()
{
    // 函数局部静态变量初始化线程安全，进程内只读一次并缓存。
    static bool cached = ProbeCheckOnlyMode();
    return cached;
}
} // namespace sim
