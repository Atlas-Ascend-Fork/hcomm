/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "big_graph_checker.h"

#include <cstring>
#include <algorithm>
#include <set>
#include <limits>
#include <utility>

#include "sim_log.h"
#include "task_graph_generator_v3/task_graph_single_task_check_v3.h"
#include "task_graph_generator_v3/task_graph_sync_conflict_v3.h"
#include "task_graph_generator_v3/task_meta_translator_v3.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace BigGraphCheckV3 {

    namespace {
        using V3Graph = TaskGraphGeneratorV3::TaskGraphGeneratorV3;
        using V3Node = TaskGraphGeneratorV3::TaskNode;
        using V3NodeId = TaskGraphGeneratorV3::NodeId;
        using V3RankNodeQueues = TaskGraphGeneratorV3::RankNodeQueues;

        HcclResult DecodeOpDetails(const sim::operation::OpDetailTab& detailTab, OpDetails& details)
        {
            if (detailTab.opDetail.size() < sizeof(OpDetails)) {
                HCCL_VM_ERROR(
                    "Operator detail payload is too small, rankId={}, actualSize={}, expectedSize={}", detailTab.rankId,
                    detailTab.opDetail.size(), sizeof(OpDetails));
                return HCCL_E_PARA;
            }
            std::memcpy(&details, detailTab.opDetail.data(), sizeof(OpDetails));
            return HCCL_SUCCESS;
        }

        HcclResult AppendTranslatedOperator(
            TaskGraphGeneratorV3::TaskMetaTranslatorV3& translator, std::vector<std::unique_ptr<V3Node>>& nodes,
            TaskGraphGeneratorV3::AllRankNodeQueues& queues)
        {
            std::vector<std::unique_ptr<V3Node>> localNodes = translator.TakeNodes();
            TaskGraphGeneratorV3::AllRankNodeQueues localQueues = translator.TakeTaskQueues();
            const size_t nodeOffset = nodes.size();
            if (nodeOffset > static_cast<size_t>(std::numeric_limits<V3NodeId>::max())
                || localNodes.size() > static_cast<size_t>(std::numeric_limits<V3NodeId>::max()) - nodeOffset) {
                return HCCL_E_MEMORY;
            }
            for (const auto& node : localNodes) {
                if (node == nullptr) {
                    return HCCL_E_PTR;
                }
            }

            for (auto& node : localNodes) {
                node->SetNodeId(static_cast<V3NodeId>(nodeOffset + static_cast<size_t>(node->GetNodeId())));
                nodes.emplace_back(std::move(node));
            }

            for (const auto& rankEntry : localQueues) {
                V3RankNodeQueues& target = queues[rankEntry.first];
                if (target.size() < rankEntry.second.size()) {
                    target.resize(rankEntry.second.size());
                }
                for (size_t streamIndex = 0; streamIndex < rankEntry.second.size(); ++streamIndex) {
                    auto& targetStream = target[streamIndex];
                    for (V3NodeId nodeId : rankEntry.second[streamIndex]) {
                        if (nodeId < 0 || static_cast<size_t>(nodeId) >= localNodes.size()) {
                            return HCCL_E_PARA;
                        }
                        targetStream.push_back(static_cast<V3NodeId>(nodeOffset + static_cast<size_t>(nodeId)));
                    }
                }
            }
            return HCCL_SUCCESS;
        }
    } // namespace

    HcclResult BigGraphCheckerV3::LoadOpData(loader::Loader& loader)
    {
        data_.Clear();
        HcclResult ret = loader.GetCcuChannelInfo(data_.channels);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        ret = loader.GetInstrResInfo(data_.instrRes);
        if (ret != HCCL_SUCCESS) {
            data_.Clear();
            return ret;
        }
        ret = loader.GetHalfRTTInfo(data_.halfRTT);
        if (ret != HCCL_SUCCESS) {
            data_.Clear();
            return ret;
        }
        std::vector<sim::operation::OpExecution> executions;
        ret = loader.LoadAllOpExecutions(executions);
        if (ret != HCCL_SUCCESS) {
            data_.Clear();
            return ret;
        }
        if (executions.size() > static_cast<size_t>(TaskGraphGeneratorV3::INVALID_OPERATOR_ID)) {
            HCCL_VM_ERROR("Too many operators in big graph, operatorCount={}", executions.size());
            data_.Clear();
            return HCCL_E_PARA;
        }

        data_.operators.reserve(executions.size());
        for (size_t operatorIndex = 0; operatorIndex < executions.size(); ++operatorIndex) {
            const sim::operation::OpExecution& execution = executions[operatorIndex];
            OpParam opParam;
            opParam.operatorId = static_cast<TaskGraphGeneratorV3::OperatorId>(operatorIndex);
            opParam.key = execution.key;
            opParam.ranks.reserve(execution.deviceRecords.size());
            for (const sim::operation::DeviceOpExecutionRecord& record : execution.deviceRecords) {
                OperatorRankData rankData;
                rankData.deviceId = record.deviceId;
                rankData.rankId = record.rankId;
                rankData.op.deviceId = record.deviceId;
                rankData.op.rankId = record.rankId;
                rankData.op.commId = record.detail.commId;
                rankData.op.detail = record.detail;
                rankData.op.memInfo = record.memInfo;
                rankData.op.tasks = record.tasks;
                rankData.taskMetas.reserve(record.tasks.size());
                for (const sim::operation::OpTaskTab& task : record.tasks) {
                    if (task.optaskMeta.size() < sizeof(HcclTaskMetaData)) {
                        HCCL_VM_ERROR(
                            "Cannot load operator task metadata because the payload is too small, "
                            "actualSize={}, expectedSize={}",
                            task.optaskMeta.size(), sizeof(HcclTaskMetaData));
                        data_.Clear();
                        return HCCL_E_PARA;
                    }
                    HcclTaskMetaData taskMeta;
                    std::memcpy(&taskMeta, task.optaskMeta.data(), sizeof(HcclTaskMetaData));
                    rankData.taskMetas.push_back(taskMeta);
                }
                opParam.ranks.push_back(std::move(rankData));
            }
            data_.operators.push_back(std::move(opParam));
        }
        HCCL_VM_INFO("Loaded all operator data for big graph, operatorCount={}", data_.operators.size());
        return HCCL_SUCCESS;
    }

    HcclResult BigGraphCheckerV3::TranslateTask()
    {
        if (data_.operators.empty()) {
            return HCCL_E_PARA;
        }

        storage_.Reset(false);
        translatedNodes_.clear();
        translatedTaskQueues_.clear();
        graph_.reset();

        // 仅含模型执行的设备需要跨算子恢复同流提交顺序。
        std::set<uint32_t> graphDevices;
        for (const auto& op : data_.operators) {
            for (const auto& rank : op.ranks) {
                for (const auto& meta : rank.taskMetas) {
                    if (meta.taskType == HccLTaskMetaType::MODEL_EXEC) {
                        graphDevices.insert(rank.deviceId);
                    }
                }
            }
        }
        // task ID 仅在 opTask_P_<pid> 表内有序，绝不跨进程比较。
        std::vector<std::pair<uint32_t, uint32_t>> nodeSubmissionOrder;

        for (const OpParam& opParam : data_.operators) {
            for (const OperatorRankData& rankData : opParam.ranks) {
                HcclResult ret = storage_.LoadHcclVmSynthesisData(
                    rankData.deviceId, rankData.op.detail.commId, opParam.key.commName, opParam.key.commHash,
                    opParam.key.opIter, rankData.rankId, rankData.op.memInfo, data_.channels, data_.halfRTT);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
        }
        HcclResult ret = storage_.LoadHcclVmInstrData(data_.instrRes);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        // 原始指令资源表（每行固定 ~1MB）已解码进 storage，解码后不再使用，立即释放避免全程驻留。
        data_.instrRes.clear();
        data_.instrRes.shrink_to_fit();

        for (OpParam& opParam : data_.operators) {
            storage_.BeginOpGroup(opParam.key.commName, opParam.key.commHash, opParam.key.opIter);
            for (const OperatorRankData& rankData : opParam.ranks) {
                OpDetails details{};
                ret = DecodeOpDetails(rankData.op.detail, details);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                sim::operation::OpDetailTab detailTab = rankData.op.detail;
                ret = storage_.Trans2CheckerParam(detailTab, details);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            ret = storage_.FinalizeOpGroup();
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            storage_.MergeAll2AllVSendCountMatrix();
            storage_.SaveCheckerParam(opParam.operatorId);

            std::vector<std::pair<uint32_t, uint32_t>> sourceOrder;
            std::vector<std::vector<HcclTaskMetaData>> allTaskMetas;
            allTaskMetas.reserve(opParam.ranks.size());
            for (OperatorRankData& rankData : opParam.ranks) {
                // taskMetas 后续不再使用，move 进 allTaskMetas，避免一次全量深拷贝。
                allTaskMetas.push_back(std::move(rankData.taskMetas));
                for (const auto& task : rankData.op.tasks) {
                    sourceOrder.emplace_back(task.pid, task.id);
                }
            }
            ret = storage_.LoadDecodedHcclVmTaskMetaData(allTaskMetas);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }

            TaskGraphGeneratorV3::TaskMetaTranslatorV3 translator;
            ret = translator.Translate(
                storage_, opParam.operatorId, opParam.key.commName, opParam.key.commHash, opParam.key.opIter);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            for (uint32_t sourceIndex : translator.GetNodeSourceIndices()) {
                if (sourceIndex >= sourceOrder.size()) {
                    return HCCL_E_PARA;
                }
                nodeSubmissionOrder.push_back(sourceOrder[sourceIndex]);
            }
            ret = AppendTranslatedOperator(translator, translatedNodes_, translatedTaskQueues_);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
        }

        for (auto& rank : translatedTaskQueues_) {
            if (graphDevices.count(rank.first) == 0) {
                continue;
            }
            for (auto& stream : rank.second) {
                // 不同进程保持原有槽位，只在同一表内恢复顺序。
                std::map<uint32_t, std::vector<V3NodeId>> byProcess;
                for (auto id : stream) {
                    byProcess[nodeSubmissionOrder.at(id).first].push_back(id);
                }
                for (auto& entry : byProcess) {
                    std::stable_sort(entry.second.begin(), entry.second.end(), [&](V3NodeId lhs, V3NodeId rhs) {
                        return nodeSubmissionOrder[lhs].second < nodeSubmissionOrder[rhs].second;
                    });
                }
                std::map<uint32_t, size_t> offsets;
                for (auto& id : stream) {
                    const uint32_t pid = nodeSubmissionOrder[id].first;
                    id = byProcess[pid][offsets[pid]++];
                }
            }
        }

        HCCL_VM_INFO(
            "Translated big graph tasks, operatorCount={}, nodeCount={}, rankCount={}", data_.operators.size(),
            translatedNodes_.size(), translatedTaskQueues_.size());
        return HCCL_SUCCESS;
    }

    HcclResult BigGraphCheckerV3::GenerateBigGraph()
    {
        if (translatedNodes_.empty() || translatedTaskQueues_.empty()) {
            HCCL_VM_ERROR(
                "{} Checker get empty task queue, please check if the HCCL-VM end normally, operatorCount={}, "
                "nodeCount={}, rankCount={}",
                MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), data_.operators.size(), translatedNodes_.size(),
                translatedTaskQueues_.size());
            return HCCL_E_PARA;
        }

        HcclResult ret = TaskGraphGeneratorV3::CheckSlaveTaskQueue(translatedNodes_, translatedTaskQueues_);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        auto graph = std::make_unique<V3Graph>();
        graph->SetStorageManager(&storage_);
        ret = graph->GenGraph(std::move(translatedNodes_), std::move(translatedTaskQueues_));
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        ret = graph->CompactSyncNodes();
        if (ret != HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "{} Failed to compact big-graph sync-stream nodes, ret={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID), static_cast<uint32_t>(ret));
            return ret;
        }
        graph_ = std::move(graph);
        HCCL_VM_INFO(
            "Generated big graph, nodeCount={}, rankCount={}", graph_->GetNodes().size(),
            graph_->GetTaskQueues().size());
        return HCCL_SUCCESS;
    }

    HcclResult BigGraphCheckerV3::SingleTaskCheck() { return HCCL_E_NOT_SUPPORT; }

    HcclResult BigGraphCheckerV3::SyncCheck()
    {
        if (graph_ == nullptr || graph_->GetMainStartNode() == nullptr) {
            HCCL_VM_ERROR(
                "{} Cannot run big graph sync-conflict check before the graph is generated",
                MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR));
            return HCCL_E_PARA;
        }

        TaskGraphGeneratorV3::SyncConflictCheckStats stats;
        const HcclResult ret = TaskGraphGeneratorV3::CheckSyncResourceConflict(graph_->GetMainStartNode(), &stats);
        HCCL_VM_INFO(
            "Big graph sync-conflict check finished, status={}, originalNodeCount={}, copiedNodeCount={}, "
            "copiedEdgeCount={}, resourceBucketCount={}, pairCount={}, checkedBucketCount={}, conflictCount={}",
            ret == HCCL_SUCCESS ? "success" : "failed", stats.originalNodeCount, stats.copiedNodeCount,
            stats.copiedEdgeCount, stats.resourceBucketCount, stats.pairCount, stats.checkedBucketCount,
            stats.conflictCount);
        return ret;
    }

    HcclResult BigGraphCheckerV3::MemConflictCheck() { return HCCL_E_NOT_SUPPORT; }

    HcclResult BigGraphCheckerV3::SemanticCheck() { return HCCL_E_NOT_SUPPORT; }

} // namespace BigGraphCheckV3
} // namespace HcclSim
