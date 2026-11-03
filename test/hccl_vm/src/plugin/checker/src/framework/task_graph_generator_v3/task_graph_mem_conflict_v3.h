/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_MEM_CONFLICT_V3_H
#define CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_MEM_CONFLICT_V3_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hccl_types.h"
#include "task_def_v3.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    struct MemConflictCheckStats {
        size_t nodeCount{0};
        size_t dataTaskNodeCount{0};
        size_t processedDataTaskNodeCount{0};
        size_t accessIntervalCount{0};
        size_t memoryBucketCount{0};
        size_t overlapCandidatePairCount{0};
        size_t overlapCandidateTaskNodeCount{0};
        size_t sameStreamPrunedPairCount{0}; /**< 同队列有序剪枝掉的候选对数(免可达判定) */
        size_t duplicatePairPrunedCount{0};  /**< 按无序节点对去重删掉的重复候选对数 */
        size_t orderedCandidatePairCount{0};
        size_t parallelCandidatePairCount{0};
    };

    /**
     * @brief 校验数据搬运任务的内存并发冲突。
     * @param start             输入：主图起始节点。
     * @param queueEntryFlags   输入：队列任务身份表,按下标 nodeId 取非 0 表示该节点是任务队列中的
     *                          原始任务(构图时相邻队列任务连了顺序边);nullptr 表示不可用,跳过
     *                          同队列剪枝。CCU/AIV 展开产生的内部节点不在队列中,不会被剪枝。
     * @param stats             输出：校验统计信息(可为 nullptr)。
     * @return 成功返回 HCCL_SUCCESS,发现并发冲突返回 HCCL_E_MEMORY。
     */
    HcclResult CheckDataMoveTaskMemConflict(
        const TaskNode* start, const std::vector<uint8_t>* queueEntryFlags, MemConflictCheckStats* stats = nullptr);
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif // CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_MEM_CONFLICT_V3_H
