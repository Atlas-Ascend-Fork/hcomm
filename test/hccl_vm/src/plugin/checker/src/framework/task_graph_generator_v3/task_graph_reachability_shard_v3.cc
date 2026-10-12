/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_graph_reachability_shard_v3.h"

#include <cstdint>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <utility>

#include "sim_log.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        constexpr size_t BITS_PER_WORD = 64U;
        constexpr uint32_t INVALID_MEM_INDEX = std::numeric_limits<uint32_t>::max();

        void SetBit(BitRow& row, size_t idx) { row[idx / BITS_PER_WORD] |= (1ULL << (idx % BITS_PER_WORD)); }

        bool IsDataMoveTask(const TaskNode* node)
        {
            return node != nullptr
                   && (node->GetType() == TaskType::TRANS_MEM || node->GetType() == TaskType::BATCH_TRANS_MEM
                       || node->GetType() == TaskType::REDUCE || node->GetType() == TaskType::BATCH_REDUCE);
        }

        bool IsMainGraphStartNode(const TaskNode* node)
        {
            if (node == nullptr || node->GetType() != TaskType::START || node->GetNodeId() != MAIN_START_NODE_ID) {
                return false;
            }

            const auto* start = dynamic_cast<const TaskStart*>(node);
            return start != nullptr && start->GetBoundaryType() == BoundaryType::MAIN_GRAPH;
        }

        void ClearShardClosure(ReachabilityClosure& closure)
        {
            closure.matrix.clear();
            closure.dataIndexByNodeId.clear();
            closure.dataNodeIdByIndex.clear();
        }

        // 分块构建的内部只读上下文：BFS 可达集合 + 全图拓扑序 + 数据搬运节点索引。
        // Init 完成后不再修改，是按卡并行构建分片闭包的基础。
        struct ShardTaskNodes {
            std::vector<const TaskNode*> nodes;
            std::map<const TaskNode*, size_t> indexByNode;
            std::map<NodeId, size_t> indexByNodeId;
        };

        // 内存节点诱导压缩图：只在内存节点（数据搬运节点）之间保留边，边语义为
        // "u 到 v 存在路径且路径中间节点全部为非内存节点"。分片传播时临时矩阵
        // 只给内存节点分配行（而非全图节点），这是降内存的关键。
        // 逆拓扑序保证父节点合并孩子集合时孩子集合必已算好，全程迭代无递归。
        struct CompressedMemGraph {
            std::vector<const TaskNode*> memTopoNodes; // 内存节点按全图拓扑序排列
            std::vector<uint32_t> memIndexByNode; // 全图行号 -> 压缩行号（非内存节点为 INVALID_MEM_INDEX）
            std::vector<uint32_t> memSuccOffsets; // CSR 偏移，size = memNodeCount + 1
            std::vector<uint32_t> memSuccTargets; // CSR 目标数组（值为压缩行号，列内有序）
        };

        struct ShardBuildContext {
            ShardTaskNodes reachable;
            std::map<const TaskNode*, size_t> dataIndexByNode;
            std::vector<const TaskNode*> taskTopoNodes;
            CompressedMemGraph memGraph;
        };

        HcclResult AddShardTaskNode(const TaskNode* node, ShardTaskNodes& reachable)
        {
            if (node == nullptr) {
                return HCCL_E_PTR;
            }
            if (reachable.nodes.size() >= MAX_NODE_COUNT) {
                return HCCL_E_MEMORY;
            }

            const NodeId nodeId = node->GetNodeId();
            if (nodeId < 0) {
                HCCL_VM_ERROR(
                    "{} Task node id is invalid, nodeId={}, node={}",
                    MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), nodeId, node->Describe());
                return HCCL_E_PARA;
            }

            const size_t nodeIndex = reachable.nodes.size();
            const auto idResult = reachable.indexByNodeId.insert(std::make_pair(nodeId, nodeIndex));
            if (!idResult.second) {
                const size_t previousIndex = idResult.first->second;
                const TaskNode* previousNode
                    = (previousIndex < reachable.nodes.size()) ? reachable.nodes[previousIndex] : nullptr;
                HCCL_VM_ERROR(
                    "{} Two different nodes share the same task node id, nodeId={}, "
                    "firstNode={}, secondNode={}",
                    MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), nodeId,
                    (previousNode == nullptr) ? "null" : previousNode->Describe(), node->Describe());
                return HCCL_E_PARA;
            }

            const auto nodeResult = reachable.indexByNode.insert(std::make_pair(node, nodeIndex));
            if (!nodeResult.second) {
                HCCL_VM_ERROR(
                    "{} The same task node object was inserted more than once, node={}",
                    MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), node->Describe());
                return HCCL_E_INTERNAL;
            }

            reachable.nodes.push_back(node);
            return HCCL_SUCCESS;
        }

        HcclResult GetShardChildIndex(
            const TaskNode* parent, const TaskNode* child, const ShardTaskNodes& reachable, size_t& childIndex)
        {
            if (parent == nullptr || child == nullptr) {
                HCCL_VM_ERROR(
                    "{} Graph structure is broken because one node is null, parent={}, child={}",
                    MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID),
                    parent == nullptr ? "null" : parent->Describe(), child == nullptr ? "null" : child->Describe());
                return HCCL_E_PTR;
            }

            const auto iter = reachable.indexByNode.find(child);
            if (iter == reachable.indexByNode.end()) {
                HCCL_VM_ERROR(
                    "{} One graph edge is invalid, so sharded reachability analysis cannot "
                    "continue, parent={}, child={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID), parent->Describe(), child->Describe());
                return HCCL_E_PARA;
            }

            childIndex = iter->second;
            return HCCL_SUCCESS;
        }

        HcclResult BuildCompressedMemGraph(ShardBuildContext& context);

        /**
         * @brief 构建全图上下文：BFS 收集可达节点并求全图拓扑序，为数据搬运节点编号并构建压缩图。
         *
         * @param start   输入：主图起始节点。
         * @param context 输出：分块构建的内部只读上下文。
         * @return 成功返回 HCCL_SUCCESS，图结构非法（非 DAG、空指针、重复 nodeId）返回相应错误码。
         */
        HcclResult BuildShardContext(const TaskNode* start, ShardBuildContext& context)
        {
            context = ShardBuildContext();
            if (!IsMainGraphStartNode(start)) {
                HCCL_VM_ERROR(
                    "{} Sharded reachability analysis cannot start because the main start node "
                    "is invalid, mainStartNode={}",
                    MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID),
                    start == nullptr ? "null" : start->Describe());
                return HCCL_E_PARA;
            }

            // 【核心步骤 1】BFS 收集主图可达的全部任务节点。
            std::queue<const TaskNode*> nodeQue;
            std::set<const TaskNode*> visited;
            nodeQue.push(start);
            visited.insert(start);
            while (!nodeQue.empty()) {
                const TaskNode* node = nodeQue.front();
                nodeQue.pop();
                if (node != start) {
                    HcclResult ret = AddShardTaskNode(node, context.reachable);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                }

                for (const TaskNode* childNode : node->GetChildren()) {
                    if (childNode == nullptr) {
                        HCCL_VM_ERROR(
                            "{} Graph structure is broken because one child node is null, "
                            "parent={}",
                            MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), node->Describe());
                        return HCCL_E_PTR;
                    }
                    if (visited.count(childNode) != 0) {
                        continue;
                    }
                    visited.insert(childNode);
                    nodeQue.push(childNode);
                }
            }

            // 【核心步骤 2】Kahn 入度消减求全图拓扑序（起始节点不计入），同时校验图是完整 DAG。
            const size_t nodeCount = context.reachable.nodes.size();
            std::vector<uint32_t> indegree(nodeCount, 0);
            auto addChildrenIndegree = [&context](const TaskNode* parent, std::vector<uint32_t>& indeg) -> HcclResult {
                for (const TaskNode* childNode : parent->GetChildren()) {
                    size_t childIndex = 0;
                    HcclResult ret = GetShardChildIndex(parent, childNode, context.reachable, childIndex);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                    ++indeg[childIndex];
                }
                return HCCL_SUCCESS;
            };

            HcclResult ret = addChildrenIndegree(start, indegree);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            for (const TaskNode* node : context.reachable.nodes) {
                ret = addChildrenIndegree(node, indegree);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }

            std::queue<const TaskNode*> topoQue;
            topoQue.push(start);
            context.taskTopoNodes.reserve(nodeCount);
            while (!topoQue.empty()) {
                const TaskNode* node = topoQue.front();
                topoQue.pop();
                if (node != start) {
                    context.taskTopoNodes.push_back(node);
                }

                for (const TaskNode* childNode : node->GetChildren()) {
                    size_t childIndex = 0;
                    ret = GetShardChildIndex(node, childNode, context.reachable, childIndex);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                    if (indegree[childIndex] == 0) {
                        return HCCL_E_PARA;
                    }
                    if (--indegree[childIndex] == 0) {
                        topoQue.push(childNode);
                    }
                }
            }
            if (context.taskTopoNodes.size() != nodeCount) {
                HCCL_VM_ERROR(
                    "{} This V3 graph is not a complete DAG from the main start node, "
                    "topoSize={}, expectedTopoSize={}, mainStartNodeId={}",
                    MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), context.taskTopoNodes.size(), nodeCount,
                    MAIN_START_NODE_ID);
                return HCCL_E_INTERNAL;
            }

            // 【核心步骤 3】为数据搬运节点（内存节点）建立索引，供分片目标校验与闭包抽取使用。
            for (const TaskNode* node : context.reachable.nodes) {
                if (!IsDataMoveTask(node)) {
                    continue;
                }
                const size_t dataIdx = context.dataIndexByNode.size();
                context.dataIndexByNode[node] = dataIdx;
            }

            // 【核心步骤 4】构建内存节点诱导压缩图，分片传播只在内存节点行上进行。
            return BuildCompressedMemGraph(context);
        }

        /**
         * @brief 构建内存节点诱导压缩图（非递归，三趟迭代）。
         *
         * 压缩图只在内存节点（数据搬运节点）之间保留边：u->v 有边当且仅当原图上
         * u 到 v 存在路径且路径中间节点全部为非内存节点。之后分片闭包传播只需给
         * 内存节点分配临时矩阵行（而非全图节点行），这是降内存的关键。
         *
         * 全程迭代无递归（递归深度等于最长链长度，大图下会栈溢出）：
         * 第 1 趟（正拓扑序）给内存节点顺序编号，压缩行号即拓扑序号；
         * 第 2 趟（逆拓扑序）逐节点把（内存孩子 ∪ 非内存孩子的已算好切片）并入
         *     扁平 arena，时间戳数组 O(1) 去重——处理任意节点时其全部孩子必已
         *     算好（拓扑序保证），结构记录完全替代递归栈；
         * 第 3 趟只保留内存节点的 arena 切片，压实为最终 CSR，非内存节点的临时
         *     集合随 arena 一起释放。
         *
         * @param context 输入/输出：全图 BFS/拓扑上下文，成功时 context.memGraph 被填充。
         * @return 成功返回 HCCL_SUCCESS，图结构非法返回相应错误码。
         */
        HcclResult BuildCompressedMemGraph(ShardBuildContext& context)
        {
            CompressedMemGraph& memGraph = context.memGraph;
            const size_t nodeCount = context.reachable.nodes.size();
            memGraph.memTopoNodes.clear();
            memGraph.memIndexByNode.assign(nodeCount, INVALID_MEM_INDEX);
            memGraph.memSuccOffsets.clear();
            memGraph.memSuccTargets.clear();

            // 【第 1 趟】正拓扑序给内存节点编号（压缩行号 = 编号），memRowByMemIndex 记录其全图行号。
            std::vector<uint32_t> memRowByMemIndex;
            for (const TaskNode* node : context.taskTopoNodes) {
                if (!IsDataMoveTask(node)) {
                    continue;
                }
                const auto nodeIter = context.reachable.indexByNode.find(node);
                if (nodeIter == context.reachable.indexByNode.end()) {
                    HCCL_VM_ERROR(
                        "{} This mem node is missing its topological index, node={}",
                        MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), node->Describe());
                    return HCCL_E_INTERNAL;
                }
                memGraph.memIndexByNode[nodeIter->second] = static_cast<uint32_t>(memGraph.memTopoNodes.size());
                memGraph.memTopoNodes.push_back(node);
                memRowByMemIndex.push_back(static_cast<uint32_t>(nodeIter->second));
            }
            const size_t memNodeCount = memGraph.memTopoNodes.size();

            // 【第 2 趟】逆拓扑序填 arena：succBegin/succEnd 按全图行号索引，切片即该节点的 memSucc 集合。
            // markStamp 用单调递增的轮次戳做 O(1) 去重（轮次值域远小于 UINT32_MAX，不会与初值冲突）。
            std::vector<uint32_t> succArena;
            std::vector<size_t> succBegin(nodeCount, 0);
            std::vector<size_t> succEnd(nodeCount, 0);
            std::vector<uint32_t> markStamp(memNodeCount, INVALID_MEM_INDEX);
            uint32_t round = 0;
            for (auto iter = context.taskTopoNodes.rbegin(); iter != context.taskTopoNodes.rend(); ++iter) {
                const TaskNode* node = *iter;
                const auto nodeIter = context.reachable.indexByNode.find(node);
                if (nodeIter == context.reachable.indexByNode.end()) {
                    HCCL_VM_ERROR(
                        "{} This node is missing its topological index, node={}",
                        MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), node->Describe());
                    return HCCL_E_INTERNAL;
                }
                const size_t nodeIndex = nodeIter->second;
                const uint32_t stamp = round++;
                succBegin[nodeIndex] = succArena.size();
                for (const TaskNode* childNode : node->GetChildren()) {
                    size_t childIndex = 0;
                    HcclResult ret = GetShardChildIndex(node, childNode, context.reachable, childIndex);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                    const uint32_t childMemIndex = memGraph.memIndexByNode[childIndex];
                    if (childMemIndex != INVALID_MEM_INDEX) {
                        // 内存孩子：直达压缩边。
                        if (markStamp[childMemIndex] != stamp) {
                            markStamp[childMemIndex] = stamp;
                            succArena.push_back(childMemIndex);
                        }
                        continue;
                    }
                    // 非内存孩子：并入其已算好的 memSucc 切片（逆拓扑序保证此时切片已填好）。
                    for (size_t pos = succBegin[childIndex]; pos < succEnd[childIndex]; ++pos) {
                        const uint32_t targetMemIndex = succArena[pos];
                        if (markStamp[targetMemIndex] != stamp) {
                            markStamp[targetMemIndex] = stamp;
                            succArena.push_back(targetMemIndex);
                        }
                    }
                }
                succEnd[nodeIndex] = succArena.size();
            }

            // 【第 3 趟】只保留内存节点的切片，前缀和压实为最终 CSR；arena 等临时结构随作用域释放。
            memGraph.memSuccOffsets.assign(memNodeCount + 1U, 0);
            size_t compressedEdgeCount = 0;
            for (size_t memIndex = 0; memIndex < memNodeCount; ++memIndex) {
                const size_t row = memRowByMemIndex[memIndex];
                compressedEdgeCount += succEnd[row] - succBegin[row];
            }
            memGraph.memSuccTargets.resize(compressedEdgeCount);
            size_t fillPos = 0;
            for (size_t memIndex = 0; memIndex < memNodeCount; ++memIndex) {
                memGraph.memSuccOffsets[memIndex] = fillPos;
                const size_t row = memRowByMemIndex[memIndex];
                for (size_t pos = succBegin[row]; pos < succEnd[row]; ++pos) {
                    memGraph.memSuccTargets[fillPos++] = succArena[pos];
                }
            }
            memGraph.memSuccOffsets[memNodeCount] = fillPos;

            HCCL_VM_INFO(
                "Compressed mem graph built, graphNodeCount={}, memNodeCount={}, "
                "compressedEdgeCount={}, compressedEdgePerMemNode={}",
                nodeCount, memNodeCount, compressedEdgeCount,
                (memNodeCount == 0U) ? 0.0 :
                                       (static_cast<double>(compressedEdgeCount) / static_cast<double>(memNodeCount)));
            return HCCL_SUCCESS;
        }

        /**
         * @brief 按目标节点子集构建分片可达闭包（内存节点压缩图上传播）。
         *
         * 逆拓扑序做按位 OR 传播：先生成 全部内存节点×targetCount 的临时矩阵，
         * 沿压缩图 CSR 边传播后只保留目标节点的行，得到目标节点之间的可达闭包。
         * 非内存节点（notify/wait、set_flag/wait_flag 等同步节点）不分配行，其
         * 传播作用已在压缩图构建时折叠进压缩边，跨卡先后关系不漏判。
         * 临时矩阵行数由全图节点数降为内存节点数，这是降内存的关键。
         *
         * @param context           输入：全图 BFS/拓扑上下文（含压缩图）。
         * @param targetIndexByNode 输入：目标节点到分片列编号的映射。
         * @param closure           输出：目标节点之间的可达闭包。
         * @return 成功返回 HCCL_SUCCESS，图结构非法返回相应错误码。
         */
        HcclResult BuildTargetedReachabilityClosure(
            const ShardBuildContext& context, const std::map<const TaskNode*, size_t>& targetIndexByNode,
            ReachabilityClosure& closure)
        {
            ClearShardClosure(closure);
            const size_t targetCount = targetIndexByNode.size();
            if (targetCount == 0U) {
                return HCCL_SUCCESS;
            }
            const CompressedMemGraph& memGraph = context.memGraph;
            const size_t memNodeCount = memGraph.memTopoNodes.size();
            const size_t wordCount = (targetCount + BITS_PER_WORD - 1U) / BITS_PER_WORD;

            // 【核心步骤 1】目标列查找表：压缩行号 -> 分片列号。
            // 目标节点必为内存节点（BuildClosureForNodes 已校验），此处逐个建立映射并防御性复查，
            // 传播内层即可 O(1) 查列，避免逐压缩边查 map。
            std::vector<uint32_t> targetColByMemIndex(memNodeCount, INVALID_MEM_INDEX);
            for (const auto& targetItem : targetIndexByNode) {
                const TaskNode* node = targetItem.first;
                const auto nodeIter = context.reachable.indexByNode.find(node);
                if (nodeIter == context.reachable.indexByNode.end()
                    || nodeIter->second >= memGraph.memIndexByNode.size()) {
                    HCCL_VM_ERROR(
                        "{} This target node is missing its reachability index, node={}",
                        MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), node->Describe());
                    return HCCL_E_INTERNAL;
                }
                const uint32_t memIndex = memGraph.memIndexByNode[nodeIter->second];
                if (memIndex == INVALID_MEM_INDEX || memIndex >= memNodeCount) {
                    HCCL_VM_ERROR(
                        "{} This target node is not a mem node in the compressed graph, node={}",
                        MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), node->Describe());
                    return HCCL_E_INTERNAL;
                }
                targetColByMemIndex[memIndex] = static_cast<uint32_t>(targetItem.second);
            }

            // 【核心步骤 2】分配临时矩阵：全部内存节点×目标节点数 的位图。
            // 矩阵生成日志：每次调用即生成一次分块矩阵，行数=内存节点数，列数=本片目标节点数，
            // 与调用侧的 shardIndex/shard plan 日志配合，可直接核对生成次数与每次规模。
            std::vector<BitRow> reachableTargetByMemNode(memNodeCount, BitRow(wordCount, 0));
            HCCL_VM_INFO(
                "Shard reachability matrix generated, matrixRowCount={}, matrixColCount={}, "
                "wordCount={}",
                memNodeCount, targetCount, wordCount);

            // 【核心步骤 3】逆压缩拓扑序（压缩行号即拓扑序号）单趟迭代传播：
            // m 的行 = 所有压缩后继行的按位 OR ∪ 后继自身（若后继是目标节点）。
            // 后继必为严格后代，压缩行号必更大，倒序遍历保证其行已算好，无递归。
            for (size_t revPos = memNodeCount; revPos > 0U; --revPos) {
                const size_t memIndex = revPos - 1U;
                BitRow& row = reachableTargetByMemNode[memIndex];
                const size_t edgeEnd = memGraph.memSuccOffsets[memIndex + 1U];
                for (size_t edgePos = memGraph.memSuccOffsets[memIndex]; edgePos < edgeEnd; ++edgePos) {
                    const uint32_t succMemIndex = memGraph.memSuccTargets[edgePos];
                    // 压缩后继本身是目标节点：直达，置位 m 行中该目标节点的列。
                    const uint32_t succCol = targetColByMemIndex[succMemIndex];
                    if (succCol != INVALID_MEM_INDEX) {
                        SetBit(row, succCol);
                    }
                    // 后继可达的目标节点 m 也全部可达：按位 OR 合并后继行（64bit/word 位并行）。
                    const BitRow& succRow = reachableTargetByMemNode[succMemIndex];
                    for (size_t wordIndex = 0; wordIndex < wordCount; ++wordIndex) {
                        row[wordIndex] |= succRow[wordIndex];
                    }
                }
            }

            // 【核心步骤 4】只保留目标节点的行，得到 目标节点×目标节点 的闭包并建立 nodeId 映射。
            closure.dataNodeIdByIndex.assign(targetCount, INVALID_NODE_ID);
            closure.matrix.assign(targetCount, BitRow(wordCount, 0));
            for (const auto& targetItem : targetIndexByNode) {
                const TaskNode* node = targetItem.first;
                const size_t targetIndex = targetItem.second;
                const auto nodeIter = context.reachable.indexByNode.find(node);
                if (nodeIter == context.reachable.indexByNode.end() || targetIndex >= closure.matrix.size()
                    || nodeIter->second >= memGraph.memIndexByNode.size()) {
                    HCCL_VM_ERROR(
                        "{} This target node is missing its reachability index, node={}",
                        MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), node->Describe());
                    return HCCL_E_INTERNAL;
                }
                const uint32_t memIndex = memGraph.memIndexByNode[nodeIter->second];
                if (memIndex == INVALID_MEM_INDEX || memIndex >= memNodeCount) {
                    HCCL_VM_ERROR(
                        "{} This target node is not a mem node in the compressed graph, node={}",
                        MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), node->Describe());
                    return HCCL_E_INTERNAL;
                }
                const NodeId nodeId = node->GetNodeId();
                closure.dataIndexByNodeId[nodeId] = targetIndex;
                closure.dataNodeIdByIndex[targetIndex] = nodeId;
                closure.matrix[targetIndex] = reachableTargetByMemNode[memIndex];
            }

            return HCCL_SUCCESS;
        }
    } // namespace

    /** @brief 构建器的 pimpl 内部上下文，持有全图 BFS/拓扑结果，Init 后只读。 */
    struct ReachabilityShardBuilder::Impl {
        ShardBuildContext context;
        bool initialized{false};
    };

    ReachabilityShardBuilder::ReachabilityShardBuilder() : impl_(new Impl()) {}

    ReachabilityShardBuilder::~ReachabilityShardBuilder() = default;

    // 接口说明见 task_graph_reachability_shard_v3.h 中的类声明。
    HcclResult ReachabilityShardBuilder::Init(const TaskNode* start)
    {
        impl_->initialized = false;
        HcclResult ret = BuildShardContext(start, impl_->context);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        impl_->initialized = true;
        return HCCL_SUCCESS;
    }

    size_t ReachabilityShardBuilder::GetNodeCount() const
    {
        return impl_->initialized ? impl_->context.reachable.nodes.size() : 0U;
    }

    HcclResult ReachabilityShardBuilder::BuildClosureForNodes(
        const std::vector<NodeId>& targetNodeIds, ReachabilityClosure& closure) const
    {
        ClearShardClosure(closure);
        if (!impl_->initialized) {
            HCCL_VM_ERROR(
                "{} Sharded reachability closure builder has not been initialized yet",
                MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID));
            return HCCL_E_PARA;
        }

        // 【核心步骤 1】目标节点按 nodeId 升序去重后顺序编号，保证分片闭包的列顺序确定。
        const std::set<NodeId> targetNodeIdSet(targetNodeIds.begin(), targetNodeIds.end());
        std::map<const TaskNode*, size_t> targetIndexByNode;

        // 【核心步骤 2】逐个校验目标节点：必须在主图可达集合内，且必须是数据搬运任务节点。
        for (const NodeId nodeId : targetNodeIdSet) {
            const auto nodeIndexIter = impl_->context.reachable.indexByNodeId.find(nodeId);
            if (nodeIndexIter == impl_->context.reachable.indexByNodeId.end()
                || nodeIndexIter->second >= impl_->context.reachable.nodes.size()) {
                HCCL_VM_ERROR(
                    "{} Sharded reachability found one target node outside the main graph, "
                    "targetNodeId={}",
                    MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), nodeId);
                return HCCL_E_PARA;
            }
            const TaskNode* node = impl_->context.reachable.nodes[nodeIndexIter->second];
            if (node == nullptr || impl_->context.dataIndexByNode.count(node) == 0U) {
                HCCL_VM_ERROR(
                    "{} Sharded reachability only covers data-move task nodes, but one "
                    "target node is not, targetNodeId={}",
                    MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), nodeId);
                return HCCL_E_PARA;
            }
            const size_t dataIdx = targetIndexByNode.size();
            targetIndexByNode[node] = dataIdx;
        }

        // 【核心步骤 3】复用已构建的全图上下文，生成该目标子集的分片可达闭包。
        return BuildTargetedReachabilityClosure(impl_->context, targetIndexByNode, closure);
    }
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
