/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_REACHABILITY_SHARD_V3_H
#define CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_REACHABILITY_SHARD_V3_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "hccl_types.h"
#include "task_def_v3.h"
#include "task_graph_reachability_v3.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {

    /**
     * @brief 可达闭包按卡分块构建的节点数阈值。
     *
     * 主图可达节点总数超过该阈值时，内存冲突检查不再走全图一次性闭包，
     * 而是按内存归属卡拆成 全部内存节点×单卡宽度 的分片矩阵逐卡构建、
     * 即算即查即释放，降低峰值内存。
     */
    // constexpr size_t REACHABILITY_SHARD_NODE_THRESHOLD = 10000U;
    constexpr size_t REACHABILITY_SHARD_NODE_THRESHOLD = 1U;

    /**
     * @brief 按卡分片校验启用多线程并行的卡数阈值。
     *
     * 参与校验的内存归属卡数超过该值时，分片闭包构建与候选对判定按卡
     * 多线程并行执行，线程数 = min(空余线程数, 卡数, 工作线程数上限)
     * 动态决定（空余线程数见 task_graph_mem_conflict_v3.cc 的
     * GetIdleThreadCount）；不超过该值时保持串行逐卡，避免小规模场景
     * 引入线程创建开销。各卡分片相互独立、零交互，结果按卡序合并，
     * 结论与串行一致。
     */
    constexpr size_t SHARDED_REACHABILITY_PARALLEL_RANK_THRESHOLD = 1U;

    /**
     * @brief 按卡分片校验多线程并行的工作线程数上限。
     *
     * 并行时线程数 = min(空余线程数, 卡数, 该上限)（主线程同样参与）。
     * 上限固定为 8：数百卡大图场景限制并发构建分片矩阵的内存峰值与
     * 线程调度开销，避免空载主机上一次创建过多线程挤占其他流程。
     */
    constexpr size_t SHARDED_REACHABILITY_MAX_WORKER_THREADS = 8U;

    /**
     * @brief 大图分块可达闭包构建器。
     *
     * 全图 BFS 收集、Kahn 拓扑排序与数据搬运节点编号只执行一次，随后构建
     * 内存节点诱导压缩图（只在数据搬运节点之间保留边，非内存同步节点折叠进
     * 压缩边，构建全程迭代无递归）。之后可按目标节点子集反复构建闭包：
     * 每个分片的临时矩阵为 全部内存节点×目标节点数（行数不随非内存同步
     * 节点数增长），输出闭包（复用 ReachabilityClosure）只包含目标节点
     * 之间的可达性，查询走老文件已有的 IsReachable()。
     *
     * 线程安全：Init 完成后内部上下文只读（压缩图 CSR 同样只读），
     * BuildClosureForNodes 为 const 方法且不修改共享状态，可安全地按卡
     * 多线程并发调用；卡数超过 SHARDED_REACHABILITY_PARALLEL_RANK_THRESHOLD
     * 时内存冲突检查即按此方式并行。
     *
     * 使用示例：
     * @code
     *   ReachabilityShardBuilder builder;
     *   ret = builder.Init(mainStart);
     *   ReachabilityClosure closure;
     *   ret = builder.BuildClosureForNodes({nodeId1, nodeId2}, closure);
     *   ret = IsReachable(closure, nodeId1, nodeId2, isReachable);
     * @endcode
     */
    class ReachabilityShardBuilder {
    public:
        ReachabilityShardBuilder();
        ~ReachabilityShardBuilder();
        ReachabilityShardBuilder(const ReachabilityShardBuilder&) = delete;
        ReachabilityShardBuilder& operator=(const ReachabilityShardBuilder&) = delete;

        /**
         * @brief 构建全图上下文（BFS 收集可达节点、Kahn 拓扑排序、数据搬运节点编号）。
         * @param start 输入：主图起始节点。
         * @return 成功返回 HCCL_SUCCESS，图结构非法返回相应错误码。
         * @note 重复调用会丢弃上一次结果并重新构建；不要与其他方法并发调用。
         */
        HcclResult Init(const TaskNode* start);

        /**
         * @brief 获取主图可达的任务节点总数（不含起始节点）。
         * @return 节点总数；尚未初始化时返回 0。
         */
        size_t GetNodeCount() const;

        /**
         * @brief 为目标节点子集构建分片可达闭包。
         * @param targetNodeIds 输入：目标节点 ID 列表，允许重复，必须均为数据搬运任务节点。
         * @param closure       输出：目标节点之间的可达闭包，单个分片用完即弃以释放内存。
         * @return 成功返回 HCCL_SUCCESS，未初始化或目标节点非法返回相应错误码。
         * @note 目标节点按 nodeId 升序去重后顺序编号，保证分片闭包的列顺序确定；
         *       Init 后可多线程并发调用本方法（内部状态只读）。
         */
        HcclResult BuildClosureForNodes(const std::vector<NodeId>& targetNodeIds, ReachabilityClosure& closure) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_; /**< pimpl 实现指针，屏蔽内部上下文结构 */
    };

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif // CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_REACHABILITY_SHARD_V3_H
