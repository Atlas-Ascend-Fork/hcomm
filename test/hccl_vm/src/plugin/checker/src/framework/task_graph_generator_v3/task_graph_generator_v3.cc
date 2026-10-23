/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_graph_generator_v3.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <utility>

#include "aiv_graph_generator_v3/aiv_task_transform_v3.h"
#include "ccu_graph_generator_v3/ccu_task_transform_v3.h"
#include "sim_log.h"
#include "storage_manager.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        struct LocalRecordKey {
            CommId commId{INVALID_COMM_ID};
            uint32_t opIter{0};
            bool operator<(const LocalRecordKey& rhs) const
            {
                if (commId != rhs.commId) {
                    return commId < rhs.commId;
                }
                return opIter < rhs.opIter;
            }
        };
        using SeenLocalRecords = std::map<LocalRecordKey, std::vector<NodeId>>;
        using SeenInterRankRecords = std::map<DeviceId, std::map<DeviceId, std::vector<NodeId>>>;

        std::string NodeIdsToString(const std::vector<TaskNode*>& nodes)
        {
            std::ostringstream os;
            os << "[";
            bool first = true;
            for (const TaskNode* node : nodes) {
                if (!first) {
                    os << ",";
                }
                first = false;
                os << ((node == nullptr) ? "null" : std::to_string(node->GetNodeId()));
            }
            os << "]";
            return os.str();
        }

        bool IsLocalNotify(const AicpuNotify& notify) { return notify.recordDeviceId == notify.waitDeviceId; }

        bool IsInterRankNotify(const AicpuNotify& notify) { return notify.recordDeviceId != notify.waitDeviceId; }

        const AicpuNotify* GetRecordNotify(const TaskNode* node)
        {
            if (node == nullptr || node->GetType() != TaskType::RECORD) {
                return nullptr;
            }
            const auto* record = dynamic_cast<const TaskRecordAICPU*>(node);
            return (record == nullptr) ? nullptr : &record->GetNotify();
        }

        const AicpuNotify* GetWaitNotify(const TaskNode* node)
        {
            if (node == nullptr || node->GetType() != TaskType::WAIT) {
                return nullptr;
            }
            const auto* wait = dynamic_cast<const TaskWaitAICPU*>(node);
            return (wait == nullptr) ? nullptr : &wait->GetNotify();
        }

        bool IsNotifyIdPeer(const AicpuNotify& recordNotify, const AicpuNotify& waitNotify)
        {
            return recordNotify.notifyId == waitNotify.notifyId;
        }

        bool StreamHasAivGraph(const TaskGraphGeneratorV3* graph, const std::vector<NodeId>& stream)
        {
            if (graph == nullptr) {
                return false;
            }
            return std::any_of(stream.begin(), stream.end(), [graph](NodeId nodeId) {
                const TaskNode* node = graph->GetNode(nodeId);
                return node != nullptr && node->GetType() == TaskType::AIV_GRAPH;
            });
        }

        std::set<StreamId> GetMainStreamIds(
            const std::vector<std::unique_ptr<TaskNode>>& nodes, DeviceId deviceId,
            const RankNodeQueues& rankTaskQueues)
        {
            std::set<StreamId> mainStreamIds;
            for (const auto& stream : rankTaskQueues) {
                for (const NodeId nodeId : stream) {
                    if (nodeId < 0 || static_cast<size_t>(nodeId) >= nodes.size() || nodes[nodeId] == nullptr) {
                        continue;
                    }
                    const TaskPosition& position = nodes[nodeId]->GetPosition();
                    if (position.deviceId != deviceId) {
                        continue;
                    }
                    if (position.mainStreamId != INVALID_STREAM_ID) {
                        mainStreamIds.insert(position.mainStreamId);
                    }
                    // Event dependencies are represented by MODEL_EXEC boundaries. Keep the
                    // replay trigger stream reachable even when opDetails names another stream.
                    if (nodes[nodeId]->GetType() == TaskType::MODEL_EXEC_START
                        && position.streamId != INVALID_STREAM_ID) {
                        mainStreamIds.insert(position.streamId);
                    }
                }
            }
            // Keep manually constructed/legacy task data usable. Production task nodes
            // always carry the operator stream from opDetails.streamId.
            if (mainStreamIds.empty()) {
                mainStreamIds.insert(0);
            }
            return mainStreamIds;
        }

        struct SyncStreamGroupKey {
            CommId commId{INVALID_COMM_ID};
            uint32_t opIter{0};

            bool operator<(const SyncStreamGroupKey& rhs) const
            {
                if (commId != rhs.commId) {
                    return commId < rhs.commId;
                }
                return opIter < rhs.opIter;
            }
        };

        bool AddEdgeOnce(TaskNode* parentNode, TaskNode* childNode)
        {
            if (parentNode == nullptr || childNode == nullptr || parentNode == childNode) {
                return false;
            }
            if (!parentNode->AddChild(childNode)) {
                return false;
            }
            (void)childNode->AddParent(parentNode);
            return true;
        }
    } // namespace

    bool TaskGraphGeneratorV3::IsValidNodeId(NodeId nodeId) const
    {
        return nodeId >= 0 && static_cast<size_t>(nodeId) < nodes_.size();
    }

    bool TaskGraphGeneratorV3::IsMainStartNodeId(NodeId nodeId) const
    {
        return mainStart_ != nullptr && nodeId == mainStartNodeId_;
    }

    StorageManager& TaskGraphGeneratorV3::GetStorageManager() const
    {
        return storage_ == nullptr ? StorageManager::GetInstance() : *storage_;
    }

    TaskNode* TaskGraphGeneratorV3::GetNode(NodeId nodeId)
    {
        if (IsMainStartNodeId(nodeId)) {
            return mainStart_.get();
        }
        if (!IsValidNodeId(nodeId)) {
            return nullptr;
        }
        return nodes_[static_cast<size_t>(nodeId)].get();
    }

    const TaskNode* TaskGraphGeneratorV3::GetNode(NodeId nodeId) const
    {
        if (IsMainStartNodeId(nodeId)) {
            return mainStart_.get();
        }
        if (!IsValidNodeId(nodeId)) {
            return nullptr;
        }
        return nodes_[static_cast<size_t>(nodeId)].get();
    }

    HcclResult TaskGraphGeneratorV3::AddEdge(NodeId parentNodeId, NodeId childNodeId)
    {
        TaskNode* childNode = GetNode(childNodeId);
        if (childNode == nullptr || IsMainStartNodeId(childNodeId)) {
            return HCCL_E_PTR;
        }

        TaskNode* parentNode = GetNode(parentNodeId);
        if (parentNode == nullptr) {
            return HCCL_E_PTR;
        }

        parentNode->AddChild(childNode);
        childNode->AddParent(parentNode);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::RemoveEdge(NodeId parentNodeId, NodeId childNodeId)
    {
        TaskNode* parentNode = GetNode(parentNodeId);
        TaskNode* childNode = GetNode(childNodeId);
        if (parentNode == nullptr || childNode == nullptr) {
            HCCL_VM_ERROR(
                "{} Failed to remove one graph edge because the parent or child node does not "
                "exist, parentNodeId={}, childNodeId={}, parentNode={}, childNode={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID), parentNodeId, childNodeId,
                parentNode == nullptr ? "null" : parentNode->Describe(),
                childNode == nullptr ? "null" : childNode->Describe());
            return HCCL_E_PARA;
        }

        (void)parentNode->RemoveChild(childNode);
        (void)childNode->RemoveParent(parentNode);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::CompactSyncNodes(SyncCompactStats* stats)
    {
        SyncCompactStats localStats;

        std::map<SyncStreamGroupKey, std::vector<NodeId>> syncGroups;
        for (const auto& node : nodes_) {
            if (node == nullptr || node->GetType() != TaskType::SYNC_STREAM) {
                continue;
            }
            const TaskPosition& position = node->GetPosition();
            syncGroups[{position.commId, position.opIter}].push_back(node->GetNodeId());
            ++localStats.syncNodeCount;
        }
        localStats.syncGroupCount = syncGroups.size();

        std::set<NodeId> removedNodeIds;
        for (const auto& groupEntry : syncGroups) {
            const std::vector<NodeId>& group = groupEntry.second;
            if (group.size() <= 1) {
                continue;
            }
            ++localStats.mergedGroupCount;

            NodeId unifiedNodeId = group.front();
            DeviceId unifiedDeviceId = GetNode(unifiedNodeId)->GetPosition().deviceId;
            for (const NodeId nodeId : group) {
                const DeviceId deviceId = GetNode(nodeId)->GetPosition().deviceId;
                if (deviceId < unifiedDeviceId || (deviceId == unifiedDeviceId && nodeId < unifiedNodeId)) {
                    unifiedNodeId = nodeId;
                    unifiedDeviceId = deviceId;
                }
            }
            TaskNode* unifiedNode = GetNode(unifiedNodeId);
            if (unifiedNode == nullptr) {
                return HCCL_E_PTR;
            }

            for (const NodeId removedNodeId : group) {
                if (removedNodeId == unifiedNodeId) {
                    continue;
                }
                TaskNode* removedNode = GetNode(removedNodeId);
                if (removedNode == nullptr) {
                    return HCCL_E_PTR;
                }

                const std::vector<TaskNode*> parents = removedNode->GetParents();
                const std::vector<TaskNode*> children = removedNode->GetChildren();
                for (TaskNode* parentNode : parents) {
                    if (parentNode == nullptr) {
                        return HCCL_E_PTR;
                    }
                    (void)parentNode->RemoveChild(removedNode);
                    (void)removedNode->RemoveParent(parentNode);
                    if (parentNode != unifiedNode && AddEdgeOnce(parentNode, unifiedNode)) {
                        ++localStats.rewiredEdgeCount;
                    }
                }
                for (TaskNode* childNode : children) {
                    if (childNode == nullptr) {
                        return HCCL_E_PTR;
                    }
                    (void)removedNode->RemoveChild(childNode);
                    (void)childNode->RemoveParent(removedNode);
                    if (childNode != unifiedNode && AddEdgeOnce(unifiedNode, childNode)) {
                        ++localStats.rewiredEdgeCount;
                    }
                }
                for (TaskNode* parentNode : parents) {
                    for (TaskNode* childNode : children) {
                        if (parentNode == unifiedNode || childNode == unifiedNode) {
                            continue;
                        }
                        if (AddEdgeOnce(parentNode, childNode)) {
                            ++localStats.bypassEdgeCount;
                        }
                    }
                }

                const TaskPosition& position = removedNode->GetPosition();
                const auto queueIter = taskQueues_.find(position.deviceId);
                if (queueIter == taskQueues_.end() || position.streamId >= queueIter->second.size()) {
                    HCCL_VM_ERROR(
                        "{} Failed to compact sync-stream nodes because the task queue of the "
                        "removed node is missing, removedNodeId={}, deviceId={}, streamId={}",
                        MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID), removedNodeId, position.deviceId,
                        position.streamId);
                    return HCCL_E_PARA;
                }
                auto& stream = queueIter->second[position.streamId];
                stream.erase(std::remove(stream.begin(), stream.end(), removedNodeId), stream.end());
                (void)removedNodeIds.insert(removedNodeId);
                ++localStats.removedNodeCount;
            }
        }

        if (removedNodeIds.empty()) {
            if (stats != nullptr) {
                *stats = localStats;
            }
            HCCL_VM_DEBUG(
                "No cross-rank sync-stream nodes need compaction, syncNodeCount={}, "
                "syncGroupCount={}",
                localStats.syncNodeCount, localStats.syncGroupCount);
            return HCCL_SUCCESS;
        }

        std::vector<NodeId> remappedNodeIds(nodes_.size(), INVALID_NODE_ID);
        std::vector<std::unique_ptr<TaskNode>> compactedNodes;
        compactedNodes.reserve(nodes_.size() - removedNodeIds.size());
        for (auto& node : nodes_) {
            if (node == nullptr) {
                return HCCL_E_PTR;
            }
            const NodeId oldNodeId = node->GetNodeId();
            if (oldNodeId < 0 || static_cast<size_t>(oldNodeId) >= remappedNodeIds.size()) {
                return HCCL_E_PARA;
            }
            if (removedNodeIds.count(oldNodeId) != 0) {
                continue;
            }
            const NodeId newNodeId = static_cast<NodeId>(compactedNodes.size());
            node->SetNodeId(newNodeId);
            remappedNodeIds[oldNodeId] = newNodeId;
            compactedNodes.push_back(std::move(node));
        }
        nodes_ = std::move(compactedNodes);

        for (auto& rankEntry : taskQueues_) {
            for (auto& stream : rankEntry.second) {
                std::vector<NodeId> remappedStream;
                remappedStream.reserve(stream.size());
                for (const NodeId nodeId : stream) {
                    if (nodeId < 0 || static_cast<size_t>(nodeId) >= remappedNodeIds.size()
                        || remappedNodeIds[nodeId] == INVALID_NODE_ID) {
                        HCCL_VM_ERROR(
                            "{} Failed to compact sync-stream nodes because one task queue entry "
                            "is invalid, deviceId={}, nodeId={}",
                            MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID), rankEntry.first, nodeId);
                        return HCCL_E_PARA;
                    }
                    remappedStream.push_back(remappedNodeIds[nodeId]);
                }
                stream = std::move(remappedStream);
            }
        }

        if (stats != nullptr) {
            *stats = localStats;
        }
        HCCL_VM_INFO(
            "Compacted cross-rank sync-stream nodes, syncNodeCount={}, syncGroupCount={}, "
            "mergedGroupCount={}, removedNodeCount={}, rewiredEdgeCount={}, bypassEdgeCount={}, nodeCount={}",
            localStats.syncNodeCount, localStats.syncGroupCount, localStats.mergedGroupCount,
            localStats.removedNodeCount, localStats.rewiredEdgeCount, localStats.bypassEdgeCount, nodes_.size());
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::AppendGeneratedNode(
        std::unique_ptr<TaskNode> node, const TaskPosition& position, NodeId& nodeId)
    {
        if (node == nullptr) {
            return HCCL_E_MEMORY;
        }
        if (nodes_.size() >= MAX_NODE_COUNT) {
            return HCCL_E_MEMORY;
        }

        nodeId = static_cast<NodeId>(nodes_.size());
        node->SetNodeId(nodeId);
        node->SetPosition(position);
        nodes_.emplace_back(std::move(node));
        return HCCL_SUCCESS;
    }

    bool TaskGraphGeneratorV3::HasPath(NodeId fromNodeId, NodeId toNodeId) const
    {
        if (fromNodeId == toNodeId) {
            return true;
        }
        const TaskNode* fromNode = GetNode(fromNodeId);
        const TaskNode* toNode = GetNode(toNodeId);
        if (fromNode == nullptr || toNode == nullptr) {
            return false;
        }

        std::vector<const TaskNode*> nodeStack;
        std::vector<uint8_t> visitedNodes(nodes_.size(), 0);
        bool mainStartVisited = false;
        auto markVisited = [this, &visitedNodes, &mainStartVisited](const TaskNode* node) {
            if (node == nullptr) {
                return false;
            }

            const NodeId nodeId = node->GetNodeId();
            if (IsMainStartNodeId(nodeId)) {
                if (mainStartVisited) {
                    return false;
                }
                mainStartVisited = true;
                return true;
            }

            if (!IsValidNodeId(nodeId)) {
                return false;
            }

            const size_t index = static_cast<size_t>(nodeId);
            if (visitedNodes[index] != 0) {
                return false;
            }

            visitedNodes[index] = 1;
            return true;
        };

        nodeStack.push_back(fromNode);
        (void)markVisited(fromNode);
        while (!nodeStack.empty()) {
            const TaskNode* currNode = nodeStack.back();
            nodeStack.pop_back();
            for (const TaskNode* childNode : currNode->GetChildren()) {
                if (childNode == nullptr) {
                    continue;
                }
                if (childNode == toNode) {
                    return true;
                }
                if (markVisited(childNode)) {
                    nodeStack.push_back(childNode);
                }
            }
        }
        return false;
    }

    const std::vector<TaskNode*>& TaskGraphGeneratorV3::GetParents(NodeId nodeId) const
    {
        static const std::vector<TaskNode*> EMPTY_NODES;
        const TaskNode* node = GetNode(nodeId);
        return (node == nullptr) ? EMPTY_NODES : node->GetParents();
    }

    const std::vector<TaskNode*>& TaskGraphGeneratorV3::GetChildren(NodeId nodeId) const
    {
        static const std::vector<TaskNode*> EMPTY_NODES;
        const TaskNode* node = GetNode(nodeId);
        return (node == nullptr) ? EMPTY_NODES : node->GetChildren();
    }

    size_t TaskGraphGeneratorV3::CountEdges() const
    {
        size_t edgeCount = (mainStart_ == nullptr) ? 0U : mainStart_->GetChildren().size();
        for (const auto& nodeOwner : nodes_) {
            if (nodeOwner != nullptr) {
                edgeCount += nodeOwner->GetChildren().size();
            }
        }
        return edgeCount;
    }

    HcclResult TaskGraphGeneratorV3::PushNextNode(
        const RankNodeQueues& rankTaskQueues, NodeId currNodeId, std::vector<NodeId>& nodeQue) const
    {
        const TaskNode* node = GetNode(currNodeId);
        if (node == nullptr) {
            return HCCL_E_PTR;
        }

        const StreamId streamId = node->GetPosition().streamId;
        if (streamId >= rankTaskQueues.size()) {
            return HCCL_E_PARA;
        }

        const auto& stream = rankTaskQueues[streamId];
        const auto iter = std::find(stream.begin(), stream.end(), currNodeId);
        if (iter == stream.end()) {
            return HCCL_E_PARA;
        }

        const auto nextIter = iter + 1;
        if (nextIter != stream.end()) {
            nodeQue.push_back(*nextIter);
        }
        return HCCL_SUCCESS;
    }

    bool TaskGraphGeneratorV3::IsExecutable(NodeId nodeId, const std::vector<uint8_t>& execFlags) const
    {
        if (!IsValidNodeId(nodeId) || static_cast<size_t>(nodeId) >= execFlags.size()) {
            return false;
        }

        for (const TaskNode* parentNode : GetParents(nodeId)) {
            if (parentNode == nullptr) {
                return false;
            }
            const NodeId parentNodeId = parentNode->GetNodeId();
            if (IsMainStartNodeId(parentNodeId)) {
                continue;
            }
            if (!IsValidNodeId(parentNodeId) || static_cast<size_t>(parentNodeId) >= execFlags.size()
                || execFlags[static_cast<size_t>(parentNodeId)] == 0) {
                return false;
            }
        }
        return true;
    }

    HcclResult TaskGraphGeneratorV3::ExecuteNode(
        NodeId nodeId, std::vector<NodeId>& graphNodeQue, std::vector<uint8_t>& execFlags,
        std::vector<uint8_t>& traverseFlags) const
    {
        if (IsMainStartNodeId(nodeId)) {
            for (const TaskNode* childNode : GetChildren(nodeId)) {
                if (childNode == nullptr) {
                    return HCCL_E_PTR;
                }
                const NodeId childNodeId = childNode->GetNodeId();
                if (!IsValidNodeId(childNodeId) || static_cast<size_t>(childNodeId) >= traverseFlags.size()) {
                    return HCCL_E_PTR;
                }
                if (traverseFlags[static_cast<size_t>(childNodeId)] == 0) {
                    traverseFlags[static_cast<size_t>(childNodeId)] = 1;
                    graphNodeQue.push_back(childNodeId);
                }
            }
            return HCCL_SUCCESS;
        }

        if (!IsValidNodeId(nodeId) || static_cast<size_t>(nodeId) >= execFlags.size()
            || static_cast<size_t>(nodeId) >= traverseFlags.size()) {
            return HCCL_E_PTR;
        }

        execFlags[static_cast<size_t>(nodeId)] = 1;
        for (const TaskNode* childNode : GetChildren(nodeId)) {
            if (childNode == nullptr) {
                return HCCL_E_PTR;
            }
            const NodeId childNodeId = childNode->GetNodeId();
            if (!IsValidNodeId(childNodeId) || static_cast<size_t>(childNodeId) >= traverseFlags.size()) {
                return HCCL_E_PTR;
            }
            if (traverseFlags[static_cast<size_t>(childNodeId)] == 0) {
                traverseFlags[static_cast<size_t>(childNodeId)] = 1;
                graphNodeQue.push_back(childNodeId);
            }
        }
        return HCCL_SUCCESS;
    }

    void TaskGraphGeneratorV3::Reset()
    {
        mainStartNodeId_ = INVALID_NODE_ID;
        mainStart_.reset();
        taskQueues_.clear();
        nodes_.clear();
        hasAiv_ = false;
        hasCcu_ = false;
        hasModelExec_ = false;
        aivExpandStats_ = AivExpandStats{};
        ccuExpandStats_ = CcuExpandStats{};
        g_checkerAivUbBufferSize = 0;
        g_checkerAivCommInfoSize = 0;
    }

    HcclResult TaskGraphGeneratorV3::GenGraph(
        std::vector<std::unique_ptr<TaskNode>> translatedNodes, AllRankNodeQueues translatedTaskQueues)
    {
        Reset();
        HCCL_VM_INFO(
            "Start building the CheckerV3 graph from translated nodes, rankCount={}, nodeCount={}",
            translatedTaskQueues.size(), translatedNodes.size());
        if (translatedNodes.empty() || translatedTaskQueues.empty()) {
            HCCL_VM_ERROR(
                "{} Checker get empty task queue, please check if the HCCL-VM end normally, rankCount={}, nodeCount={}",
                MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), translatedTaskQueues.size(),
                translatedNodes.size());
            return HCCL_E_PARA;
        }
        nodes_ = std::move(translatedNodes);
        taskQueues_ = std::move(translatedTaskQueues);
        hasAiv_ = false;
        hasCcu_ = false;
        hasModelExec_ = false;
        for (const auto& node : nodes_) {
            if (node == nullptr) {
                continue;
            }
            const TaskType type = node->GetType();
            if (type == TaskType::AIV_GRAPH) {
                hasAiv_ = true;
            } else if (type == TaskType::CCU_GRAPH) {
                hasCcu_ = true;
            } else if (type >= TaskType::MODEL_EXEC_START && type <= TaskType::MODEL_EXEC_END) {
                hasModelExec_ = true;
            }
        }

        HcclResult ret = CreateMainStartNode();
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        return BuildDagEdges();
    }

    HcclResult TaskGraphGeneratorV3::CreateMainStartNode()
    {
        auto mainStart = std::make_unique<TaskStart>(BoundaryType::MAIN_GRAPH);
        TaskPosition startPosition;
        mainStart->SetNodeId(MAIN_START_NODE_ID);
        mainStart->SetPosition(startPosition);
        mainStartNodeId_ = MAIN_START_NODE_ID;
        mainStart_ = std::move(mainStart);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::BuildDagEdges()
    {
        HCCL_VM_INFO(
            "Start building graph edges, rankCount={}, nodeCount={}, mainStartNodeId={}", taskQueues_.size(),
            nodes_.size(), mainStartNodeId_);

        HcclResult ret = HCCL_SUCCESS;
        for (const auto& rankEntry : taskQueues_) {
            ret = GenGraph4Rank(rankEntry.first, rankEntry.second);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
        }

        // MODEL_EXEC 菱形跨流边必须在流内顺序边(GenGraph4Rank)之后,notify 边之前补齐, 使 notify
        // 依赖在正确的模型边界内建立.
        if (hasModelExec_) {
            ret = AddModelExecEdges();
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
        }

        for (const auto& rankEntry : taskQueues_) {
            ret = AddLocalNotifyEdges(rankEntry.first, rankEntry.second);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
        }

        ret = AddInterRankNotifyEdges();
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        if (hasAiv_) {
            ret = ExpandAivSubGraphs();
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
        }

        if (hasCcu_) {
            ret = ExpandCcuSubGraphs();
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
        }

        HCCL_VM_INFO(
            "Finished building graph edges, nodeCount={}, edgeCount={}, mainStartNodeId={}", nodes_.size(),
            CountEdges(), mainStartNodeId_);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::ExpandAivSubGraphs()
    {
        const size_t originalNodeCount = nodes_.size();
        AivExpandStats stats;
        StorageManager& storage = GetStorageManager();
        std::vector<TaskAivGraph*> aivGraphs;

        for (size_t nodeIndex = 0; nodeIndex < originalNodeCount; ++nodeIndex) {
            TaskNode* node = nodes_[nodeIndex].get();
            if (node == nullptr || node->GetType() != TaskType::AIV_GRAPH) {
                continue;
            }

            auto* aivGraph = dynamic_cast<TaskAivGraph*>(node);
            if (aivGraph == nullptr) {
                HCCL_VM_ERROR(
                    "{} One node expected to be an AIV subgraph entry is actually another node "
                    "type, nodeId={}, node={}",
                    MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR),
                    node == nullptr ? std::string("null") : std::to_string(node->GetNodeId()),
                    node == nullptr ? "node=null" : node->Describe());
                return HCCL_E_INTERNAL;
            }
            aivGraphs.push_back(aivGraph);
        }

        AivGraphsGenerateInputV3 input;
        input.graph = this;
        input.aivGraphs = aivGraphs;
        input.storage = &storage;
        AivGraphGenerateOutputV3 result;
        HcclResult ret = ExpandAivGraphsV3(input, result);
        if (ret != HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "{} Failed to expand AIV Graph nodes, "
                "aivGraphCount={}, ret={}",
                MakeErrorCodeText(ErrorCode::GRAPH_TRANSLATE_FAILED), aivGraphs.size(), static_cast<uint32_t>(ret));
            return ret;
        }

        stats.graphCount = aivGraphs.size();
        stats.internalNodeCount = result.internalNodeCount;
        stats.setWaitEdgeCount = result.setWaitEdgeCount;
        stats.pipeBarrierMergeCount = result.pipeBarrierMergeCount;
        stats.syncAllMergeCount = result.syncAllMergeCount;
        stats.sendRecvEdgeCount = result.sendRecvEdgeCount;
        stats.taskJsonTotalTaskCount = result.taskJsonTotalTaskCount;
        stats.dagNodeCountBeforeCpGmMerge = result.dagNodeCountBeforeCpGmMerge;
        stats.dagNodeCountAfterCpGmMerge = result.dagNodeCountAfterCpGmMerge;
        stats.cpGmLoopMergeCount = result.cpGmLoopMergeCount;
        stats.cpGmMergedIterationCount = result.cpGmMergedIterationCount;
        stats.cpGmMergedOriginalNodeCount = result.cpGmMergedOriginalNodeCount;
        stats.cpGmGeneratedNodeCount = result.cpGmGeneratedNodeCount;
        stats.cpGmInactiveNodeCount = result.cpGmInactiveNodeCount;
        stats.totalExpandNs = result.expandNs;
        aivExpandStats_ = stats;
        HCCL_VM_INFO(
            "Finished expanding AIV Graph nodes:\n"
            "  graph: aivGraphCount={}, internalNodeCount={}\n"
            "  edges: setWaitEdgeCount={}, pipeBarrierMergeCount={}, syncAllMergeCount={}, sendRecvEdgeCount={}\n"
            "  tasks: taskJsonTotalTaskCount={}, dagNodeCountBeforeCpGmMerge={}, dagNodeCountAfterCpGmMerge={}\n"
            "  merge: cpGmLoopMergeCount={}, cpGmMergedIterationCount={}, cpGmMergedOriginalNodeCount={}\n"
            "  nodes: cpGmGeneratedNodeCount={}, cpGmInactiveNodeCount={}\n"
            "  buf: ubBufferSize={}, aivCommInfoSize={}\n"
            "  time: expandTotalMs={}",
            stats.graphCount, stats.internalNodeCount, stats.setWaitEdgeCount, stats.pipeBarrierMergeCount,
            stats.syncAllMergeCount, stats.sendRecvEdgeCount, stats.taskJsonTotalTaskCount,
            stats.dagNodeCountBeforeCpGmMerge, stats.dagNodeCountAfterCpGmMerge, stats.cpGmLoopMergeCount,
            stats.cpGmMergedIterationCount, stats.cpGmMergedOriginalNodeCount, stats.cpGmGeneratedNodeCount,
            stats.cpGmInactiveNodeCount, g_checkerAivUbBufferSize, g_checkerAivCommInfoSize,
            stats.totalExpandNs / 1000000ULL);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::ExpandCcuSubGraphs()
    {
        const size_t originalNodeCount = nodes_.size();
        CcuExpandStats stats;
        StorageManager& storage = GetStorageManager();
        std::vector<TaskCcuGraph*> ccuGraphs;

        for (size_t nodeIndex = 0; nodeIndex < originalNodeCount; ++nodeIndex) {
            TaskNode* node = nodes_[nodeIndex].get();
            if (node == nullptr || node->GetType() != TaskType::CCU_GRAPH) {
                continue;
            }

            auto* ccuGraph = dynamic_cast<TaskCcuGraph*>(node);
            if (ccuGraph == nullptr) {
                HCCL_VM_ERROR(
                    "{} One node expected to be a CCU subgraph entry is actually another node "
                    "type, nodeId={}, node={}",
                    MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR),
                    node == nullptr ? std::string("null") : std::to_string(node->GetNodeId()),
                    node == nullptr ? "node=null" : node->Describe());
                return HCCL_E_INTERNAL;
            }

            ccuGraphs.push_back(ccuGraph);
        }

        CcuGraphsGenerateInputV3 input;
        input.graph = this;
        input.ccuGraphs = ccuGraphs;
        input.storage = &storage;
        CcuGraphGenerateOutputV3 result;
        HcclResult ret = ExpandCcuGraphsV3(input, result);
        if (ret != HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "{} Failed to expand CCU Graph nodes, "
                "ccuGraphCount={}, ret={}",
                MakeErrorCodeText(ErrorCode::GRAPH_TRANSLATE_FAILED), ccuGraphs.size(), static_cast<uint32_t>(ret));
            return ret;
        }
        stats.graphCount = ccuGraphs.size();
        stats.internalNodeCount = result.internalNodeCount;
        stats.recordWaitEdgeCount = result.recordWaitEdgeCount;
        stats.totalExpandNs = result.expandNs;

        ccuExpandStats_ = stats;
        HCCL_VM_INFO(
            "Finished expanding CCU Graph nodes, ccuGraphCount={}, internalNodeCount={}, "
            "recordWaitEdgeCount={}, expandTotalMs={}",
            stats.graphCount, stats.internalNodeCount, stats.recordWaitEdgeCount, stats.totalExpandNs / 1000000ULL);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::ValidateTranslatedNodes() const
    {
        if (nodes_.size() >= MAX_NODE_COUNT) {
            return HCCL_E_MEMORY;
        }

        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (nodes_[i] == nullptr || nodes_[i]->GetNodeId() != static_cast<NodeId>(i)) {
                return HCCL_E_PARA;
            }
        }

        for (const auto& rankEntry : taskQueues_) {
            for (const auto& stream : rankEntry.second) {
                for (const NodeId nodeId : stream) {
                    if (!IsValidNodeId(nodeId)) {
                        return HCCL_E_PARA;
                    }
                }
            }
        }

        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::GenGraph4Rank(DeviceId deviceId, const RankNodeQueues& rankTaskQueues)
    {
        size_t nonEmptyStreamCount = 0;
        size_t taskNodeCount = 0;
        size_t startEdgeCount = 0;
        size_t streamOrderEdgeCount = 0;
        const std::set<StreamId> mainStreamIds = GetMainStreamIds(nodes_, deviceId, rankTaskQueues);
        // Connect operator and model trigger streams at their first task to preserve stream order.
        for (const StreamId mainStreamId : mainStreamIds) {
            if (mainStreamId >= rankTaskQueues.size() || rankTaskQueues[mainStreamId].empty()) {
                continue;
            }
            HcclResult ret = AddEdge(mainStartNodeId_, rankTaskQueues[mainStreamId].front());
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            ++startEdgeCount;
        }

        for (size_t streamIndex = 0; streamIndex < rankTaskQueues.size(); ++streamIndex) {
            const auto& stream = rankTaskQueues[streamIndex];
            if (!stream.empty()) {
                ++nonEmptyStreamCount;
            }
            taskNodeCount += stream.size();
            if (hasAiv_ && mainStreamIds.find(static_cast<StreamId>(streamIndex)) == mainStreamIds.end()
                && !stream.empty() && StreamHasAivGraph(this, stream)) {
                HcclResult ret = AddEdge(mainStartNodeId_, stream.front());
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ++startEdgeCount;
            }
            for (size_t i = 1; i < stream.size(); ++i) {
                HcclResult ret = AddEdge(stream[i - 1], stream[i]);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ++streamOrderEdgeCount;
            }
        }
        HCCL_VM_DEBUG(
            "Built per-rank skeleton edges, rankId={}, nonEmptyStreamCount={}, taskNodeCount={}, "
            "startEdgeCount={}, streamOrderEdgeCount={}",
            deviceId, nonEmptyStreamCount, taskNodeCount, startEdgeCount, streamOrderEdgeCount);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::AddLocalNotifyEdges(DeviceId deviceId, const RankNodeQueues& rankTaskQueues)
    {
        std::vector<NodeId> rankNodeQue;
        SeenLocalRecords seenLocalRecords;
        uint64_t unmatchedCnt = 0;
        size_t matchedEdgeCount = 0;

        const std::set<StreamId> mainStreamIds = GetMainStreamIds(nodes_, deviceId, rankTaskQueues);
        for (const StreamId mainStreamId : mainStreamIds) {
            if (mainStreamId < rankTaskQueues.size() && !rankTaskQueues[mainStreamId].empty()) {
                rankNodeQue.push_back(rankTaskQueues[mainStreamId].front());
            }
        }
        for (size_t streamIndex = 0; streamIndex < rankTaskQueues.size(); ++streamIndex) {
            if (mainStreamIds.find(static_cast<StreamId>(streamIndex)) == mainStreamIds.end()
                && !rankTaskQueues[streamIndex].empty()) {
                rankNodeQue.push_back(rankTaskQueues[streamIndex].front());
            }
        }

        while (!rankNodeQue.empty()) {
            if (unmatchedCnt >= rankNodeQue.size()) {
                const TaskNode* node = GetNode(rankNodeQue.front());
                HCCL_VM_ERROR(
                    "{} Local Record/Wait matching is stuck on this rank. Some Wait tasks are "
                    "still blocked, but no new local Record task can unblock them, rankId={}, firstBlockedWaitNode={}, "
                    "blockedWaitNodeCount={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_DEADLOCK), deviceId, node == nullptr ? "null" : node->Describe(),
                    rankNodeQue.size());
                return HCCL_E_INTERNAL;
            }

            const NodeId currNodeId = rankNodeQue.front();
            rankNodeQue.erase(rankNodeQue.begin());
            const TaskNode* currNode = GetNode(currNodeId);
            if (currNode == nullptr) {
                return HCCL_E_PTR;
            }

            const AicpuNotify* recordNotify = GetRecordNotify(currNode);
            if (recordNotify != nullptr && IsLocalNotify(*recordNotify)) {
                HcclResult ret = PushNextNode(rankTaskQueues, currNodeId, rankNodeQue);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                const TaskPosition& recordPosition = currNode->GetPosition();
                seenLocalRecords[LocalRecordKey{recordPosition.commId, recordPosition.opIter}].push_back(currNodeId);
                unmatchedCnt = 0;
                continue;
            }

            const AicpuNotify* waitNotify = GetWaitNotify(currNode);
            if (waitNotify != nullptr && IsLocalNotify(*waitNotify)) {
                const TaskPosition& waitPosition = currNode->GetPosition();
                bool matched = false;
                auto bucketIt = seenLocalRecords.find(LocalRecordKey{waitPosition.commId, waitPosition.opIter});
                if (bucketIt != seenLocalRecords.end()) {
                    auto recordIter = bucketIt->second.begin();
                    for (; recordIter != bucketIt->second.end(); ++recordIter) {
                        const AicpuNotify* seenRecordNotify = GetRecordNotify(GetNode(*recordIter));
                        if (seenRecordNotify != nullptr && IsNotifyIdPeer(*seenRecordNotify, *waitNotify)) {
                            const NodeId matchedRecordNodeId = *recordIter;
                            HcclResult ret = AddEdge(matchedRecordNodeId, currNodeId);
                            if (ret != HCCL_SUCCESS) {
                                return ret;
                            }
                            bucketIt->second.erase(recordIter);
                            ret = PushNextNode(rankTaskQueues, currNodeId, rankNodeQue);
                            if (ret != HCCL_SUCCESS) {
                                return ret;
                            }
                            HCCL_VM_DEBUG(
                                "Matched local Record/Wait edge, rankId={}, commId={}, opIter={}, "
                                "notifyId={}, recordNode={}, waitNode={}",
                                deviceId, waitPosition.commId, waitPosition.opIter, waitNotify->notifyId,
                                matchedRecordNodeId, currNodeId);
                            ++matchedEdgeCount;
                            unmatchedCnt = 0;
                            matched = true;
                            break;
                        }
                    }
                }

                if (!matched) {
                    rankNodeQue.push_back(currNodeId);
                    ++unmatchedCnt;
                }
                continue;
            }

            HcclResult ret = PushNextNode(rankTaskQueues, currNodeId, rankNodeQue);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            unmatchedCnt = 0;
        }

        for (const auto& bucket : seenLocalRecords) {
            if (bucket.second.empty()) {
                continue;
            }
            const TaskNode* node = GetNode(bucket.second.front());
            HCCL_VM_ERROR(
                "{} Found local Record tasks that were never consumed by any local Wait task, "
                "rankId={}, commId={}, opIter={}, firstUnconsumedRecordNode={}, unconsumedRecordCount={}",
                MakeErrorCodeText(ErrorCode::GRAPH_UNMATCHED), deviceId, bucket.first.commId, bucket.first.opIter,
                node == nullptr ? "node=null" : node->Describe(), bucket.second.size());
            return HCCL_E_INTERNAL;
        }

        HCCL_VM_DEBUG(
            "Finished matching local Record/Wait edges on one rank, rankId={}, "
            "matchedRecordWaitEdgeCount={}",
            deviceId, matchedEdgeCount);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::AddInterRankNotifyEdges()
    {
        SeenInterRankRecords seenInterRankRecords;
        std::vector<NodeId> graphNodeQue;
        std::vector<uint8_t> execFlags(nodes_.size(), 0);
        std::vector<uint8_t> traverseFlags(nodes_.size(), 0);
        uint64_t unmatchedCnt = 0;
        size_t matchedEdgeCount = 0;

        HcclResult ret = ExecuteNode(mainStartNodeId_, graphNodeQue, execFlags, traverseFlags);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        while (!graphNodeQue.empty()) {
            if (unmatchedCnt >= graphNodeQue.size()) {
                const TaskNode* node = GetNode(graphNodeQue.front());
                HCCL_VM_ERROR(
                    "{} Cross-rank Record/Wait matching is stuck. Some Wait tasks are still "
                    "blocked, but no new cross-rank Record task can unblock them, firstBlockedWaitNode={}, "
                    "blockedWaitNodeCount={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_DEADLOCK), node == nullptr ? "node=null" : node->Describe(),
                    graphNodeQue.size());
                return HCCL_E_INTERNAL;
            }

            const NodeId currNodeId = graphNodeQue.front();
            graphNodeQue.erase(graphNodeQue.begin());
            const TaskNode* currNode = GetNode(currNodeId);
            if (currNode == nullptr) {
                return HCCL_E_PTR;
            }

            if (!IsExecutable(currNodeId, execFlags)) {
                graphNodeQue.push_back(currNodeId);
                ++unmatchedCnt;
                continue;
            }

            const AicpuNotify* recordNotify = GetRecordNotify(currNode);
            if (recordNotify != nullptr && IsInterRankNotify(*recordNotify)) {
                seenInterRankRecords[recordNotify->recordDeviceId][recordNotify->waitDeviceId].push_back(currNodeId);
                ret = ExecuteNode(currNodeId, graphNodeQue, execFlags, traverseFlags);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                unmatchedCnt = 0;
                continue;
            }

            const AicpuNotify* waitNotify = GetWaitNotify(currNode);
            if (waitNotify != nullptr && IsInterRankNotify(*waitNotify)) {
                bool matched = false;
                auto rankIter = seenInterRankRecords.find(waitNotify->recordDeviceId);
                if (rankIter != seenInterRankRecords.end()) {
                    auto peerIter = rankIter->second.find(waitNotify->waitDeviceId);
                    if (peerIter != rankIter->second.end()) {
                        auto recordIter = peerIter->second.begin();
                        for (; recordIter != peerIter->second.end(); ++recordIter) {
                            const AicpuNotify* seenRecordNotify = GetRecordNotify(GetNode(*recordIter));
                            if (seenRecordNotify != nullptr && IsNotifyIdPeer(*seenRecordNotify, *waitNotify)) {
                                ret = AddEdge(*recordIter, currNodeId);
                                if (ret != HCCL_SUCCESS) {
                                    return ret;
                                }
                                peerIter->second.erase(recordIter);
                                ret = ExecuteNode(currNodeId, graphNodeQue, execFlags, traverseFlags);
                                if (ret != HCCL_SUCCESS) {
                                    return ret;
                                }
                                ++matchedEdgeCount;
                                unmatchedCnt = 0;
                                matched = true;
                                break;
                            }
                        }
                        if (matched) {
                            continue;
                        }
                    }
                }

                graphNodeQue.push_back(currNodeId);
                ++unmatchedCnt;
                continue;
            }

            ret = ExecuteNode(currNodeId, graphNodeQue, execFlags, traverseFlags);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            unmatchedCnt = 0;
        }

        for (const auto& rankEntry : seenInterRankRecords) {
            for (const auto& peerEntry : rankEntry.second) {
                if (!peerEntry.second.empty()) {
                    const TaskNode* node = GetNode(peerEntry.second.front());
                    const AicpuNotify* recordNotify = GetRecordNotify(node);
                    HCCL_VM_ERROR(
                        "{} Found cross-rank Record tasks that were never consumed by any matching "
                        "Wait task, recordDeviceId={}, waitDeviceId={}, notifyId={}, firstUnconsumedRecordNode={}, "
                        "unconsumedRecordCount={}",
                        MakeErrorCodeText(ErrorCode::GRAPH_UNMATCHED), rankEntry.first, peerEntry.first,
                        recordNotify == nullptr ? std::string("null") : std::to_string(recordNotify->notifyId),
                        node == nullptr ? "null" : node->Describe(), peerEntry.second.size());
                    return HCCL_E_INTERNAL;
                }
            }
        }

        HCCL_VM_DEBUG(
            "Finished matching cross-rank Record/Wait edges, "
            "matchedCrossRankRecordWaitEdgeCount={}",
            matchedEdgeCount);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::AddModelExecEdges()
    {
        //   按 execSeq 把同一次 rtModelExecute 的 4 个 MODEL_EXEC 节点配对，建两条跨流依赖边：
        //   主流 MODEL_EXEC_START  -> 从流 MODEL_EXEC_START_SUB（触发重放）
        //   从流 MODEL_EXEC_END_SUB -> 主流 MODEL_EXEC_END   （重放完成回传）
        //   （主流/从流的流内顺序边已由 GenGraph4Rank 建好，这里只补菱形两条跨流边。）
        //   execSeq 是 host 进程内自增，多卡多进程下各 rank 会从 0 开始重复，必须叠加 deviceId
        //   作为分组键，否则不同 rank 的同 execSeq 会互相覆盖、跨 rank 错配。
        struct ModelExecGroupKey {
            DeviceId deviceId{INVALID_DEVICE_ID};
            uint32_t execSeq{0};
            bool operator<(const ModelExecGroupKey& rhs) const
            {
                if (deviceId != rhs.deviceId) {
                    return deviceId < rhs.deviceId;
                }
                return execSeq < rhs.execSeq;
            }
        };
        std::map<ModelExecGroupKey, std::map<uint8_t, NodeId>> execSeqNodes;
        for (size_t nodeIndex = 0; nodeIndex < nodes_.size(); ++nodeIndex) {
            const TaskNode* node = nodes_[nodeIndex].get();
            if (node == nullptr) {
                continue;
            }
            uint8_t role = UINT8_MAX;
            switch (node->GetType()) {
                case TaskType::MODEL_EXEC_START:
                    role = 0;
                    break;
                case TaskType::MODEL_EXEC_START_SUB:
                    role = 1;
                    break;
                case TaskType::MODEL_EXEC_END_SUB:
                    role = 2;
                    break;
                case TaskType::MODEL_EXEC_END:
                    role = 3;
                    break;
                default:
                    continue;
            }
            const auto* modelExec = dynamic_cast<const TaskModelExec*>(node);
            if (modelExec == nullptr) {
                continue;
            }
            execSeqNodes[{node->GetPosition().deviceId, modelExec->GetExecSeq()}][role]
                = static_cast<NodeId>(nodeIndex);
        }

        size_t addedEdgeCount = 0;
        for (const auto& execSeqEntry : execSeqNodes) {
            const auto& roleNodes = execSeqEntry.second;
            auto startIt = roleNodes.find(0);
            auto startSubIt = roleNodes.find(1);
            auto endSubIt = roleNodes.find(2);
            auto endIt = roleNodes.find(3);
            if (startIt == roleNodes.end() || startSubIt == roleNodes.end() || endSubIt == roleNodes.end()
                || endIt == roleNodes.end()) {
                HCCL_VM_ERROR(
                    "Incomplete MODEL_EXEC task group, deviceId={}, execSeq={}, start={}, startSub={}, "
                    "endSub={}, end={}",
                    execSeqEntry.first.deviceId, execSeqEntry.first.execSeq, startIt != roleNodes.end(),
                    startSubIt != roleNodes.end(), endSubIt != roleNodes.end(), endIt != roleNodes.end());
                continue;
            }
            // 触发流已由 GenGraph4Rank 接入 mainStart。仍将 START_SUB 的现有前序依赖复制到
            // START，使触发端也遵守算子流的已有顺序约束。
            // 注意: 必须在添加 START -> START_SUB 之前取快照, 否则会把 START 自身当成 START_SUB 的父节点形成环.
            const TaskNode* startSubNode = GetNode(startSubIt->second);
            if (startSubNode == nullptr) {
                return HCCL_E_PTR;
            }
            const std::vector<TaskNode*> startSubParents = startSubNode->GetParents();
            for (const TaskNode* parentNode : startSubParents) {
                if (parentNode == nullptr || parentNode->GetNodeId() == startIt->second) {
                    continue;
                }
                HcclResult ret = AddEdge(parentNode->GetNodeId(), startIt->second);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }

            HcclResult ret = AddEdge(startIt->second, startSubIt->second);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            ret = AddEdge(endSubIt->second, endIt->second);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            addedEdgeCount += 2;
        }

        HCCL_VM_DEBUG(
            "Finished building MODEL_EXEC dependency edges, groupCount={}, addedEdgeCount={}", execSeqNodes.size(),
            addedEdgeCount);
        return HCCL_SUCCESS;
    }
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
