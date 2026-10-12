/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TASK_SEQUENTIAL_EXECUTE_V2_H
#define HCCL_TASK_SEQUENTIAL_EXECUTE_V2_H

#include <map>
#include <string>

#include "sim_common_defs.h"
#include "sim_loader.h"
#include "operation_data/operation_data_types.h"
#include "aiv_graph_executor.h"

using namespace HcclSim;

namespace VirtualRunTime {

class SequentialExecutorV2 {
public:
    explicit SequentialExecutorV2(loader::Loader& loader) : loader_(loader) {}
    ~SequentialExecutorV2() = default;

    HcclVmResult Execute();

private:
    HcclVmResult ExecuteOneTask(HcclTaskMetaData& task);

    // AIV
    HcclVmResult TaskAivGraph(const HcclTaskMetaData& task);
    std::shared_ptr<AivGraphExecutor> GetAivGraphExecutor(uint64_t deviceId, uint64_t launchIdx);
    void RemoveAivGraphExecutor(uint64_t deviceId, uint64_t launchIdx);

    // CCU
    HcclVmResult EnsureCcuResource(); // 按DB变更签名增量刷新CCU资源(设备资源/通道映射/微码指令空间)

private:
    loader::Loader& loader_;

    // <{deviceId, launchIdx}, AivGraphExecutor> 每个AivGraphExecutor对应一个AivGraphTask节点, 因为在顺序执行引擎下,
    // AivGraphTask节点常常不能一次执行完成, 需要记录执行状态并让出
    std::map<std::pair<uint64_t, uint64_t>, std::shared_ptr<AivGraphExecutor>> aivExecutorMap_{};

    bool ccuResourceInited_{false};
    uint32_t lastInstrLoadCnt_{0};
    uint32_t lastChannelCnt_{0};
};

} // namespace VirtualRunTime
#endif
