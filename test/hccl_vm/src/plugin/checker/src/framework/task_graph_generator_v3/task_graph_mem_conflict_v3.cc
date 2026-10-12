/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_graph_mem_conflict_v3.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "sim_log.h"
#include "task_graph_reachability_v3.h"
#include "task_graph_reachability_shard_v3.h"
#include "thread_affinity.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        enum class MemoryAccessKind : uint8_t {
            READ = 0,
            WRITE,
        };

        struct MemoryKey {
            DeviceId deviceId{INVALID_DEVICE_ID};
            MemType memType{MemType::INVALID};
            uint64_t aivUbIdx{0};

            bool operator<(const MemoryKey& rhs) const
            {
                if (deviceId != rhs.deviceId) {
                    return deviceId < rhs.deviceId;
                }
                if (memType != rhs.memType) {
                    return static_cast<uint32_t>(memType) < static_cast<uint32_t>(rhs.memType);
                }
                return aivUbIdx < rhs.aivUbIdx;
            }
        };

        struct MemoryAccess {
            MemoryKey key;
            uint64_t start{0};
            uint64_t end{0};
            MemoryAccessKind kind{MemoryAccessKind::READ};
            NodeId nodeId{INVALID_NODE_ID};
            const TaskNode* node{nullptr};
        };

        using TaskMemoryAccesses = std::vector<MemoryAccess>;
        using MemoryAccessBuckets = std::map<MemoryKey, TaskMemoryAccesses>;
        using MemConflictClock = std::chrono::steady_clock;

        constexpr uint64_t MEM_CONFLICT_NS_PER_MS = 1000000ULL;

        struct AccessStartLess {
            bool operator()(const MemoryAccess& lhs, const MemoryAccess& rhs) const
            {
                if (lhs.start != rhs.start) {
                    return lhs.start < rhs.start;
                }
                if (lhs.end != rhs.end) {
                    return lhs.end < rhs.end;
                }
                if (lhs.nodeId != rhs.nodeId) {
                    return lhs.nodeId < rhs.nodeId;
                }
                return static_cast<uint32_t>(lhs.kind) < static_cast<uint32_t>(rhs.kind);
            }
        };

        struct AccessExactLess {
            bool operator()(const MemoryAccess& lhs, const MemoryAccess& rhs) const
            {
                if (lhs.key.deviceId != rhs.key.deviceId) {
                    return lhs.key.deviceId < rhs.key.deviceId;
                }
                if (lhs.key.memType != rhs.key.memType) {
                    return static_cast<uint32_t>(lhs.key.memType) < static_cast<uint32_t>(rhs.key.memType);
                }
                if (lhs.key.aivUbIdx != rhs.key.aivUbIdx) {
                    return lhs.key.aivUbIdx < rhs.key.aivUbIdx;
                }
                if (lhs.start != rhs.start) {
                    return lhs.start < rhs.start;
                }
                if (lhs.end != rhs.end) {
                    return lhs.end < rhs.end;
                }
                if (lhs.kind != rhs.kind) {
                    return static_cast<uint32_t>(lhs.kind) < static_cast<uint32_t>(rhs.kind);
                }
                return lhs.nodeId < rhs.nodeId;
            }
        };

        struct AccessEndLess {
            bool operator()(const MemoryAccess* lhs, const MemoryAccess* rhs) const
            {
                if (lhs == rhs) {
                    return false;
                }
                if (lhs->end != rhs->end) {
                    return lhs->end < rhs->end;
                }
                if (lhs->start != rhs->start) {
                    return lhs->start < rhs->start;
                }
                if (lhs->nodeId != rhs->nodeId) {
                    return lhs->nodeId < rhs->nodeId;
                }
                if (lhs->kind != rhs->kind) {
                    return static_cast<uint32_t>(lhs->kind) < static_cast<uint32_t>(rhs->kind);
                }
                return std::less<const MemoryAccess*>()(lhs, rhs);
            }
        };

        using ActiveAccesses = std::multiset<const MemoryAccess*, AccessEndLess>;

        struct OverlapCandidateStatsCollector {
            std::unordered_set<NodeId> nodeIds;
            std::map<std::string, size_t> pairCountByBucket;
            std::map<std::string, size_t> pairCountByTaskTypePair;
        };

        struct ConflictCandidate {
            const MemoryAccess* current{nullptr};
            const MemoryAccess* history{nullptr};
        };

        using ConflictCandidates = std::vector<ConflictCandidate>;

        const char* AccessKindName(MemoryAccessKind kind) { return kind == MemoryAccessKind::WRITE ? "WRITE" : "READ"; }

        const char* TaskTypeName(TaskType taskType)
        {
            switch (taskType) {
                case TaskType::TRANS_MEM:
                    return "TRANS_MEM";
                case TaskType::BATCH_TRANS_MEM:
                    return "BATCH_TRANS_MEM";
                case TaskType::REDUCE:
                    return "REDUCE";
                case TaskType::BATCH_REDUCE:
                    return "BATCH_REDUCE";
                case TaskType::RECORD:
                    return "RECORD";
                case TaskType::WAIT:
                    return "WAIT";
                case TaskType::CCU_GRAPH:
                    return "CCU_GRAPH";
                case TaskType::AIV_GRAPH:
                    return "AIV_GRAPH";
                case TaskType::AIV_SET_FLAG:
                    return "AIV_SET_FLAG";
                case TaskType::AIV_WAIT_FLAG:
                    return "AIV_WAIT_FLAG";
                case TaskType::AIV_PIPE_BARRIER:
                    return "AIV_PIPE_BARRIER";
                case TaskType::AIV_SYNC_ALL:
                    return "AIV_SYNC_ALL";
                case TaskType::AIV_SEND_FLAG:
                    return "AIV_SEND_FLAG";
                case TaskType::AIV_RECV_FLAG:
                    return "AIV_RECV_FLAG";
                case TaskType::START:
                    return "START";
                case TaskType::END:
                    return "END";
                case TaskType::SYNC_STREAM:
                    return "SYNC_STREAM";
                default:
                    return "INVALID";
            }
        }

        uint64_t ElapsedMemConflictMs(MemConflictClock::time_point start, MemConflictClock::time_point end)
        {
            const auto elapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            return static_cast<uint64_t>(elapsedNs) / MEM_CONFLICT_NS_PER_MS;
        }

        /** @brief 进程内存信息（单位 KB），读取失败时保持 0。 */
        struct SelfStatusMemoryKb {
            size_t rssKb{0}; /**< 当前驻留物理内存（VmRSS）。 */
            size_t hwmKb{0}; /**< 历史峰值物理内存（VmHWM），单调递增。 */
        };

        /**
         * @brief 从 /proc/self/status 读取进程当前与峰值物理内存（VmRSS/VmHWM）。
         *
         * 读取失败（非 Linux 环境等）返回 0，不影响正常流程；多线程并发调用安全（只读系统文件）。
         *
         * @return 当前与峰值物理内存（KB）。
         */
        SelfStatusMemoryKb ReadSelfStatusMemoryKb()
        {
            SelfStatusMemoryKb memory;
            std::ifstream statusFile("/proc/self/status");
            std::string line;
            while (std::getline(statusFile, line)) {
                const size_t valueBegin = line.find_first_of("0123456789");
                if (valueBegin == std::string::npos) {
                    continue;
                }
                if (line.compare(0, 6, "VmRSS:") == 0) {
                    memory.rssKb = static_cast<size_t>(std::strtoull(line.c_str() + valueBegin, nullptr, 10));
                } else if (line.compare(0, 6, "VmHWM:") == 0) {
                    memory.hwmKb = static_cast<size_t>(std::strtoull(line.c_str() + valueBegin, nullptr, 10));
                }
            }
            return memory;
        }

        void LogMemConflictStageTime(const char* stage, const char* status, MemConflictClock::time_point start)
        {
            // rssKb 为阶段结束时的当前物理内存，hwmKb 为进程历史峰值；相邻阶段 hwmKb 的差值
            // 即该阶段新推高的峰值内存，可用于定位哪个阶段的峰值贡献最大。
            const SelfStatusMemoryKb memory = ReadSelfStatusMemoryKb();
            HCCL_VM_INFO(
                "Stage finished, stage={}, status={}, costMs={}, rssKb={}, hwmKb={}", stage, status,
                ElapsedMemConflictMs(start, MemConflictClock::now()), memory.rssKb, memory.hwmKb);
        }

        std::string DescribeCpuList(const std::vector<int>& cpuList)
        {
            std::ostringstream cpuText;
            for (size_t index = 0; index < cpuList.size(); ++index) {
                if (index != 0) {
                    cpuText << ",";
                }
                cpuText << cpuList[index];
            }
            return cpuText.str();
        }

        bool IsDataMoveTaskNode(const TaskNode* node)
        {
            return node != nullptr
                   && (node->GetType() == TaskType::TRANS_MEM || node->GetType() == TaskType::BATCH_TRANS_MEM
                       || node->GetType() == TaskType::REDUCE || node->GetType() == TaskType::BATCH_REDUCE);
        }

        std::string DescribeKey(const MemoryKey& key)
        {
            std::ostringstream os;
            os << "rank ";
            if (key.deviceId == INVALID_DEVICE_ID) {
                os << "invalid";
            } else {
                os << key.deviceId;
            }
            os << ", " << DescribeMemType(key.memType);
            if (key.memType == MemType::AIV_UB) {
                os << ",(aivBlockIndex=" << key.aivUbIdx << ")";
            }
            return os.str();
        }

        std::string DescribeAccess(const MemoryAccess& access)
        {
            std::ostringstream os;
            os << "    node " << access.nodeId
               << ", action=" << (access.kind == MemoryAccessKind::WRITE ? "write" : "read")
               << "\n    access range : [0x" << std::hex << access.start << ",0x" << access.end << ")" << std::dec;
            if (access.node != nullptr) {
                os << "\n    task         : " << access.node->Describe();
            }
            return os.str();
        }

        std::string MakeTaskTypePairKey(TaskType lhs, TaskType rhs)
        {
            std::string lhsName = TaskTypeName(lhs);
            std::string rhsName = TaskTypeName(rhs);
            if (lhsName > rhsName) {
                std::swap(lhsName, rhsName);
            }
            return lhsName + "<->" + rhsName;
        }

        void LogCountBreakdown(const char* prefix, const std::map<std::string, size_t>& counts)
        {
            (void)prefix;
            (void)counts;
        }

        HcclResult CollectReachableTaskNodes(const TaskNode* start, std::vector<const TaskNode*>& nodes)
        {
            nodes.clear();
            if (start == nullptr) {
                return HCCL_E_PTR;
            }

            // 先用 BFS 收集 start 可达的全部节点，后续所有统计/冲突分析都只在这个闭包里进行。
            std::queue<const TaskNode*> nodeQue;
            std::set<const TaskNode*> visited;
            nodeQue.push(start);
            visited.insert(start);
            while (!nodeQue.empty()) {
                const TaskNode* node = nodeQue.front();
                nodeQue.pop();
                if (node != start) {
                    nodes.push_back(node);
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
            return HCCL_SUCCESS;
        }

        size_t CountDataMoveTaskNodes(const std::vector<const TaskNode*>& nodes)
        {
            size_t count = 0;
            for (const TaskNode* node : nodes) {
                if (IsDataMoveTaskNode(node)) {
                    ++count;
                }
            }
            return count;
        }

        uint64_t MakeAivUbIdx(const TaskPosition& position)
        {
            if (position.launchIdx == std::numeric_limits<uint64_t>::max()
                || position.blockId == std::numeric_limits<uint32_t>::max()) {
                return 0;
            }
            return (position.launchIdx << 32U) | static_cast<uint64_t>(position.blockId);
        }

        uint64_t GetAivUbIdx(const MemSlice& slice, const TaskNode* node)
        {
            if (slice.memType != MemType::AIV_UB || node == nullptr) {
                return 0;
            }
            return MakeAivUbIdx(node->GetPosition());
        }

        HcclResult
        AddAccess(const MemSlice& slice, MemoryAccessKind kind, const TaskNode* node, TaskMemoryAccesses& accesses)
        {
            if (node == nullptr) {
                return HCCL_E_PTR;
            }
            if (node->GetNodeId() < 0) {
                HCCL_VM_ERROR(
                    "{} Data task node id is invalid, nodeId={}, task={}",
                    MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), node->GetNodeId(), node->Describe());
                return HCCL_E_PARA;
            }
            if (slice.len == 0) {
                return HCCL_SUCCESS;
            }
            if (slice.deviceId == INVALID_DEVICE_ID || slice.memType == MemType::INVALID) {
                HCCL_VM_ERROR(
                    "{} One memory slice is missing a valid rank or memory type, "
                    "task={}, deviceId={}, memType={}, offset=0x{:x}, length=0x{:x}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(),
                    slice.deviceId == INVALID_DEVICE_ID ? "invalid" : std::to_string(slice.deviceId),
                    DescribeMemType(slice.memType), slice.offset, slice.len);
                return HCCL_E_PARA;
            }
            MemoryAccess access;
            access.key.deviceId = slice.deviceId;
            if (!slice.HasMemoryAddress() || slice.len > std::numeric_limits<uint64_t>::max() - slice.MemoryBegin()) {
                HCCL_VM_ERROR(
                    "{} Memory slice address is missing or its end overflows, task={}, memType={}, "
                    "rawAddr=0x{:x}, offset=0x{:x}, length=0x{:x}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(),
                    DescribeMemType(slice.memType), slice.rawAddr, slice.offset, slice.len);
                return HCCL_E_PARA;
            }
            // 全局逻辑视图按物理地址合桶；MS 和 AIV 保持各自独立的地址空间。
            const bool isGlobal = IsGlobalPhysicalMemoryType(slice.memType);
            access.key.memType = isGlobal ? MemType::PHYSICAL_GLOBAL : slice.memType;
            access.key.aivUbIdx = isGlobal ? 0 : GetAivUbIdx(slice, node);
            access.start = slice.MemoryBegin();
            access.end = slice.MemoryEnd();
            access.kind = kind;
            access.nodeId = node->GetNodeId();
            access.node = node;
            accesses.push_back(access);
            return HCCL_SUCCESS;
        }

        HcclResult BuildTaskMemoryAccesses(const TaskNode* node, TaskMemoryAccesses& accesses)
        {
            accesses.clear();
            if (!IsDataMoveTaskNode(node)) {
                return HCCL_SUCCESS;
            }

            if (node->GetType() == TaskType::TRANS_MEM) {
                const auto* transMem = dynamic_cast<const TaskTransMem*>(node);
                if (transMem == nullptr) {
                    return HCCL_E_PTR;
                }
                HcclResult ret = AddAccess(transMem->GetSrc(), MemoryAccessKind::READ, node, accesses);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                return AddAccess(transMem->GetDst(), MemoryAccessKind::WRITE, node, accesses);
            }

            if (node->GetType() == TaskType::REDUCE) {
                const auto* reduce = dynamic_cast<const TaskReduce*>(node);
                if (reduce == nullptr) {
                    return HCCL_E_PTR;
                }
                const std::vector<MemSlice>& srcs = reduce->GetSrcs();
                if (srcs.empty()) {
                    HCCL_VM_ERROR(
                        "{} This reduce task has no source data, task={}",
                        MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe());
                    return HCCL_E_PARA;
                }
                for (size_t index = 0; index < srcs.size(); ++index) {
                    HcclResult ret = AddAccess(srcs[index], MemoryAccessKind::READ, node, accesses);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                }
                return AddAccess(reduce->GetDst(), MemoryAccessKind::WRITE, node, accesses);
            }

            if (node->GetType() == TaskType::BATCH_TRANS_MEM) {
                const auto* transMem = dynamic_cast<const TaskBatchTransMem*>(node);
                if (transMem == nullptr) {
                    return HCCL_E_PTR;
                }
                const auto& srcs = transMem->GetMergedSrcs();
                const auto& dsts = transMem->GetMergedDsts();
                for (const auto& src : srcs) {
                    HcclResult ret = AddAccess(src, MemoryAccessKind::READ, node, accesses);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                }
                for (const auto& dst : dsts) {
                    HcclResult ret = AddAccess(dst, MemoryAccessKind::WRITE, node, accesses);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                }
                return HCCL_SUCCESS;
            }

            if (node->GetType() == TaskType::BATCH_REDUCE) {
                const auto* reduce = dynamic_cast<const TaskBatchReduce*>(node);
                if (reduce == nullptr) {
                    return HCCL_E_PTR;
                }
                const auto& srcGroups = reduce->GetMergedSrcs();
                const auto& dsts = reduce->GetMergedDsts();
                if (srcGroups.empty() || srcGroups.size() != dsts.size()) {
                    HCCL_VM_ERROR(
                        "{} Batch reduce has different counts of source groups and target "
                        "memory slices, task={}, srcGroupCount={}, dstCount={}",
                        MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), srcGroups.size(),
                        dsts.size());
                    return HCCL_E_PARA;
                }
                for (size_t groupIndex = 0; groupIndex < srcGroups.size(); ++groupIndex) {
                    if (srcGroups[groupIndex].empty()) {
                        HCCL_VM_ERROR(
                            "{} One reduce group has no source data at all, task={}, "
                            "groupIndex={}",
                            MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), groupIndex);
                        return HCCL_E_PARA;
                    }
                    for (const auto& src : srcGroups[groupIndex]) {
                        HcclResult ret = AddAccess(src, MemoryAccessKind::READ, node, accesses);
                        if (ret != HCCL_SUCCESS) {
                            return ret;
                        }
                    }
                    HcclResult ret = AddAccess(dsts[groupIndex], MemoryAccessKind::WRITE, node, accesses);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                }
                return HCCL_SUCCESS;
            }

            return HCCL_SUCCESS;
        }

        void DedupTaskMemoryAccesses(TaskMemoryAccesses& accesses)
        {
            if (accesses.size() <= 1) {
                return;
            }

            // 同一task可能抽取出重复区间，先做去重，减少后续扫描线候选数量。
            std::sort(accesses.begin(), accesses.end(), AccessExactLess());
            accesses.erase(
                std::unique(
                    accesses.begin(), accesses.end(),
                    [](const MemoryAccess& lhs, const MemoryAccess& rhs) {
                        return lhs.key.deviceId == rhs.key.deviceId && lhs.key.memType == rhs.key.memType
                               && lhs.key.aivUbIdx == rhs.key.aivUbIdx && lhs.start == rhs.start && lhs.end == rhs.end
                               && lhs.kind == rhs.kind && lhs.nodeId == rhs.nodeId;
                    }),
                accesses.end());
        }

        bool HasWriteAccess(const MemoryAccess& lhs, const MemoryAccess& rhs)
        {
            return lhs.kind == MemoryAccessKind::WRITE || rhs.kind == MemoryAccessKind::WRITE;
        }

        void RefreshOverlapCandidateStats(const OverlapCandidateStatsCollector& collector, MemConflictCheckStats& stats)
        {
            stats.overlapCandidateTaskNodeCount = collector.nodeIds.size();
        }

        void RecordOverlapCandidateTaskNodes(NodeId lhs, NodeId rhs, OverlapCandidateStatsCollector& collector)
        {
            collector.nodeIds.insert(lhs);
            collector.nodeIds.insert(rhs);
        }

        HcclResult CanRunInParallel(const ReachabilityClosure& closure, NodeId lhs, NodeId rhs, bool& canRunInParallel)
        {
            canRunInParallel = false;
            if (lhs == rhs) {
                return HCCL_SUCCESS;
            }

            bool lhsReachRhs = false;
            HcclResult ret = IsReachable(closure, lhs, rhs, lhsReachRhs);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }

            bool rhsReachLhs = false;
            ret = IsReachable(closure, rhs, lhs, rhsReachLhs);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }

            canRunInParallel = !lhsReachRhs && !rhsReachLhs;
            return HCCL_SUCCESS;
        }

        HcclResult CollectConflictCandidate(
            const MemoryAccess& current, const MemoryAccess& history, bool& needReachabilityClosure,
            ConflictCandidates& candidates, MemConflictCheckStats& stats, OverlapCandidateStatsCollector& collector)
        {
            // 同一节点内部重叠不在这里判冲突；纯读读重叠也不会形成冲突。
            if (current.nodeId == history.nodeId || !HasWriteAccess(current, history)) {
                return HCCL_SUCCESS;
            }

            ++stats.overlapCandidatePairCount;
            RecordOverlapCandidateTaskNodes(current.nodeId, history.nodeId, collector);
            collector.pairCountByBucket[DescribeKey(current.key)]++;
            if (current.node != nullptr && history.node != nullptr) {
                collector
                    .pairCountByTaskTypePair[MakeTaskTypePairKey(current.node->GetType(), history.node->GetType())]++;
            }
            needReachabilityClosure = true;
            candidates.push_back(ConflictCandidate{&current, &history});
            return HCCL_SUCCESS;
        }

        void RemoveInactiveAccesses(ActiveAccesses& activeAccesses, uint64_t sweepStart)
        {
            while (!activeAccesses.empty()) {
                const MemoryAccess* access = *activeAccesses.begin();
                // 扫描线推进到 sweepStart 后，所有已经完全结束的区间都可以移出活动集合。
                if (access->end > sweepStart) {
                    return;
                }
                activeAccesses.erase(activeAccesses.begin());
            }
        }

        HcclResult CheckActiveAccesses(
            const MemoryAccess& current, const ActiveAccesses& activeAccesses, bool& needReachabilityClosure,
            ConflictCandidates& candidates, MemConflictCheckStats& stats, OverlapCandidateStatsCollector& collector)
        {
            for (const MemoryAccess* activeAccess : activeAccesses) {
                if (activeAccess == nullptr) {
                    return HCCL_E_PTR;
                }
                HcclResult ret = CollectConflictCandidate(
                    current, *activeAccess, needReachabilityClosure, candidates, stats, collector);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            return HCCL_SUCCESS;
        }

        HcclResult ScanMemoryBucket(
            TaskMemoryAccesses& bucketAccesses, bool& needReachabilityClosure, ConflictCandidates& candidates,
            MemConflictCheckStats& stats, OverlapCandidateStatsCollector& collector)
        {
            // 同一个 bucket 内 deviceId/memType 相同，只需要按区间起点做一次扫描线检查。
            std::sort(bucketAccesses.begin(), bucketAccesses.end(), AccessStartLess());

            ActiveAccesses activeReads;
            ActiveAccesses activeWrites;
            for (const MemoryAccess& current : bucketAccesses) {
                RemoveInactiveAccesses(activeReads, current.start);
                RemoveInactiveAccesses(activeWrites, current.start);

                if (current.kind == MemoryAccessKind::READ) {
                    // 读只需要和当前仍活跃的写比较。
                    HcclResult ret = CheckActiveAccesses(
                        current, activeWrites, needReachabilityClosure, candidates, stats, collector);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                    activeReads.insert(&current);
                    continue;
                }

                // 写需要同时检查读/写两个集合，因为读写、写写都可能冲突。
                HcclResult ret
                    = CheckActiveAccesses(current, activeReads, needReachabilityClosure, candidates, stats, collector);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ret = CheckActiveAccesses(current, activeWrites, needReachabilityClosure, candidates, stats, collector);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                activeWrites.insert(&current);
            }
            return HCCL_SUCCESS;
        }

        HcclResult ScanMemoryBuckets(
            MemoryAccessBuckets& accessBuckets, bool& needReachabilityClosure, ConflictCandidates& candidates,
            MemConflictCheckStats& stats, OverlapCandidateStatsCollector& collector)
        {
            for (auto& bucketItem : accessBuckets) {
                HcclResult ret
                    = ScanMemoryBucket(bucketItem.second, needReachabilityClosure, candidates, stats, collector);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            return HCCL_SUCCESS;
        }

        template <typename ReachabilityClosureT>
        HcclResult CheckCollectedConflictCandidates(
            const ConflictCandidates& candidates, const ReachabilityClosureT& closure, MemConflictCheckStats& stats)
        {
            for (const ConflictCandidate& candidate : candidates) {
                if (candidate.current == nullptr || candidate.history == nullptr) {
                    return HCCL_E_PTR;
                }

                bool canRunInParallel = false;
                HcclResult ret
                    = CanRunInParallel(closure, candidate.current->nodeId, candidate.history->nodeId, canRunInParallel);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                // 若两个任务在图上已有先后约束，则即使地址重叠也不是并发冲突。
                if (!canRunInParallel) {
                    ++stats.orderedCandidatePairCount;
                    continue;
                }

                ++stats.parallelCandidatePairCount;
                const uint64_t overlapStart = std::max(candidate.current->start, candidate.history->start);
                const uint64_t overlapEnd = std::min(candidate.current->end, candidate.history->end);
                HCCL_VM_ERROR(
                    "{} Two tasks may access the same memory range in parallel, and at least one "
                    "access is a write.\n  Conflict memory : {}\n  Overlap range    : [0x{:x},0x{:x})\n"
                    "  Conflict task 1:\n{}\n  Conflict task 2:\n{}",
                    MakeErrorCodeText(ErrorCode::MEMCONFLICT_DETECTED), DescribeKey(candidate.current->key),
                    overlapStart, overlapEnd, DescribeAccess(*candidate.current), DescribeAccess(*candidate.history));
                return HCCL_E_MEMORY;
            }
            return HCCL_SUCCESS;
        }

        HcclResult CheckCandidatesWithDenseReachability(
            const TaskNode* start, const ConflictCandidates& candidates, MemConflictCheckStats& stats)
        {
            const auto closureStart = MemConflictClock::now();
            ReachabilityClosure closure;
            // 只有扫描线阶段确实发现候选重叠时，才构建可达闭包做精确判定。
            HcclResult ret = GenReachabilityClosure(start, closure);
            LogMemConflictStageTime(
                "GenReachabilityClosure(Dense)", ret == HCCL_SUCCESS ? "success" : "failed", closureStart);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }

            const auto checkStart = MemConflictClock::now();
            ret = CheckCollectedConflictCandidates(candidates, closure, stats);
            LogMemConflictStageTime(
                "CheckCollectedConflictCandidates(Dense)", ret == HCCL_SUCCESS ? "success" : "failed", checkStart);
            return ret;
        }

        /**
         * @brief 估算当前主机的空余线程数，用于动态决定按卡并行校验的线程数。
         *
         * 硬件线程数减去系统 1 分钟平均负载（向上取整，保守估计已被占用的线程数），
         * 剩余即空余线程数；硬件并发度未知或负载不可读时退化为硬件线程数。
         * 结果恒不小于 1，保证极端情况下也能继续逐卡推进。
         *
         * @return 空余线程数，恒不小于 1。
         */
        size_t GetIdleThreadCount()
        {
            const unsigned int hardwareThreadCount = std::thread::hardware_concurrency();
            if (hardwareThreadCount == 0U) {
                return 1U;
            }
            std::ifstream loadavgFile("/proc/loadavg");
            double loadAverage = 0.0;
            loadavgFile >> loadAverage;
            if (!loadavgFile || loadAverage <= 0.0) {
                return static_cast<size_t>(hardwareThreadCount);
            }
            const double busyThreadCount = std::ceil(loadAverage);
            if (busyThreadCount >= static_cast<double>(hardwareThreadCount)) {
                return 1U;
            }
            return static_cast<size_t>(hardwareThreadCount) - static_cast<size_t>(busyThreadCount);
        }

        /** @brief 单卡分片校验的固化输入：卡号、目标节点列表与该卡候选对，串行/并行路径共用。 */
        struct ShardWorkItem {
            DeviceId deviceId{INVALID_DEVICE_ID};
            std::vector<NodeId> targetNodeIds;
            const ConflictCandidates* candidates{nullptr};
        };

        /** @brief 单卡分片校验的独立结果：各卡写入自己的槽位，避免多线程直接累加共享统计。 */
        struct ShardRankResult {
            HcclResult ret{HCCL_SUCCESS}; /**< 本卡校验返回码，HCCL_E_MEMORY 表示发现并发冲突。 */
            bool processed{false}; /**< 本卡是否真正执行过分片校验（失败早退时部分卡会被跳过）。 */
            MemConflictCheckStats stats{}; /**< 本卡统计，最终按卡序合并进总统计。 */
        };

        /**
         * @brief 处理单张内存归属卡：构建该卡分片闭包 -> 判定候选对，结果写入独立槽位。
         *
         * 串行与多线程并行路径共用的单卡处理单元；builder 在 Init 后内部状态只读，
         * 可被多线程并发调用（见 task_graph_reachability_shard_v3.h 说明）。
         *
         * @param builder    输入：已完成 Init 的分块闭包构建器（只读，线程安全）。
         * @param shardIndex 输入：当前卡序号（1 起始），仅用于日志。
         * @param shardCount 输入：参与校验的总卡数，仅用于日志。
         * @param item       输入：该卡的卡号、目标节点列表与候选对。
         * @param result     输出：本卡校验返回码与统计信息。
         */
        void ProcessShardRank(
            const ReachabilityShardBuilder& builder, size_t shardIndex, size_t shardCount, const ShardWorkItem& item,
            ShardRankResult& result)
        {
            // 每次生成日志：shardIndex 即当前是第几次生成分块矩阵，targetNodeCount 为本片的列数，
            // 行数恒为内存节点数（见上面的 shard plan 日志），两者相乘即本片矩阵规模。
            HCCL_VM_INFO(
                "Mem conflict reachability shard begin, shardIndex={}/{}, deviceId={}, "
                "targetNodeCount={}, candidatePairCount={}",
                shardIndex, shardCount, item.deviceId, item.targetNodeIds.size(), item.candidates->size());

            // 只为本卡候选节点构建分片闭包：临时矩阵宽度 = 本卡候选节点数。
            const auto closureStart = MemConflictClock::now();
            ReachabilityClosure closure;
            HcclResult ret = builder.BuildClosureForNodes(item.targetNodeIds, closure);
            LogMemConflictStageTime(
                "GenReachabilityClosure(Sharded)", ret == HCCL_SUCCESS ? "success" : "failed", closureStart);
            result.processed = true;
            if (ret != HCCL_SUCCESS) {
                result.ret = ret;
                return;
            }

            // 用该卡闭包判定候选对：双向均不可达才判并发冲突，已有先后约束则放过。
            const auto checkStart = MemConflictClock::now();
            ret = CheckCollectedConflictCandidates(*item.candidates, closure, result.stats);
            LogMemConflictStageTime(
                "CheckCollectedConflictCandidates(Sharded)", ret == HCCL_SUCCESS ? "success" : "failed", checkStart);
            result.ret = ret;
            // 本卡闭包随作用域自动析构，分片矩阵不跨卡累计，峰值内存恒为单卡宽度。
        }

        /**
         * @brief 按内存归属卡分块构建可达闭包，校验扫描线收集到的重叠候选对（大图路径）。
         *
         * 分块策略见 task_graph_reachability_shard_v3：校验一张卡时其他卡不参与，
         * 每张卡只构建 全部内存节点×该卡候选节点 的分片可达矩阵（内存节点诱导
         * 压缩图上传播，非内存同步节点不占行），判定完该卡候选后立即析构释放，
         * 避免构建全量闭包。候选对判定与报错复用本文件已有的
         * CheckCollectedConflictCandidates，分块构建由 ReachabilityShardBuilder 提供。
         *
         * 执行模式：卡数不超过 SHARDED_REACHABILITY_PARALLEL_RANK_THRESHOLD 时串行逐卡；
         * 超过阈值时按卡并行，线程数 = min(空余线程数, 卡数, 线程数上限) 动态决定
         * （上限 SHARDED_REACHABILITY_MAX_WORKER_THREADS=8，主线程同样参与），
         * 工作线程通过原子计数动态领卡保证负载均衡，任一卡失败后其余线程尽快收工。
         * 独占机器场景可设置环境变量 HCCL_VM_BIND_CORE=1，并行线程按“优先不同物理核”
         * 的顺序绑核，减少线程迁移与缓存失效；开关未开启、无可用 CPU 或绑定失败时
         * 自动退化为普通多线程，校验结论不受影响。
         * 各卡分片相互独立、零交互，结果按卡序合并，错误码取卡序最前的失败，
         * 与串行逐卡的结论保持一致。
         *
         * @param start      输入：主图起始节点。
         * @param candidates 输入：扫描线阶段收集的重叠候选对（每个候选两个访问的内存归属卡相同）。
         * @param stats      输出：内存冲突检查统计信息。
         * @return 成功返回 HCCL_SUCCESS；发现并发冲突返回 HCCL_E_MEMORY，以卡序最前的冲突为准。
         */
        HcclResult CheckCandidatesWithShardedReachability(
            const TaskNode* start, const ConflictCandidates& candidates, MemConflictCheckStats& stats)
        {
            const auto builderStart = MemConflictClock::now();
            // 入口日志：确认大图分支确实进入分块流程（nodeCount > 阈值时才会走到这里）；
            // 内存信息作为基线，后续各阶段日志的 hwmKb 与其相减即该阶段的峰值内存增量。
            const SelfStatusMemoryKb beginMemory = ReadSelfStatusMemoryKb();
            HCCL_VM_INFO(
                "Mem conflict sharded reachability path begin, candidatePairCount={}, "
                "nodeThreshold={}, rssKb={}, hwmKb={}",
                candidates.size(), REACHABILITY_SHARD_NODE_THRESHOLD, beginMemory.rssKb, beginMemory.hwmKb);

            // 【核心步骤 1】按内存归属卡（候选访问的 deviceId）把候选对分组，
            // 并收集每张卡涉及的候选节点集合（后续分片矩阵的目标列）。
            // 同一候选对的两个访问必然属于同一张卡（桶键相同），按 current 即可归组。
            std::map<DeviceId, ConflictCandidates> candidatesByDevice;
            std::map<DeviceId, std::set<NodeId>> targetNodeIdSetByDevice;
            for (const ConflictCandidate& candidate : candidates) {
                if (candidate.current == nullptr || candidate.history == nullptr) {
                    return HCCL_E_PTR;
                }
                const DeviceId deviceId = candidate.current->key.deviceId;
                candidatesByDevice[deviceId].push_back(candidate);
                targetNodeIdSetByDevice[deviceId].insert(candidate.current->nodeId);
                targetNodeIdSetByDevice[deviceId].insert(candidate.history->nodeId);
            }

            // 【核心步骤 2】全图 BFS/拓扑上下文只构建一次，供后续各卡分片复用；
            // Init 后内部状态只读，是各卡（含并行）并发构建分片闭包的基础。
            ReachabilityShardBuilder builder;
            HcclResult ret = builder.Init(start);
            LogMemConflictStageTime(
                "ReachabilityShardBuilderInit(Sharded)", ret == HCCL_SUCCESS ? "success" : "failed", builderStart);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            // 计划日志：matrixGenerateCount 即本次校验分块矩阵将要生成的次数（有候选对的卡数），
            // graphNodeCount 为全图节点数（每次生成矩阵的行数），提前打印便于与实际生成次数核对。
            HCCL_VM_INFO(
                "Mem conflict shard plan, graphNodeCount={}, matrixGenerateCount={}", builder.GetNodeCount(),
                candidatesByDevice.size());

            // 【核心步骤 3】把每张卡的输入（卡号、目标节点、候选对）固化成数组，串行/并行路径共用。
            std::vector<ShardWorkItem> workItems;
            workItems.reserve(candidatesByDevice.size());
            for (const auto& deviceItem : candidatesByDevice) {
                const auto targetSetIter = targetNodeIdSetByDevice.find(deviceItem.first);
                if (targetSetIter == targetNodeIdSetByDevice.end()) {
                    return HCCL_E_INTERNAL;
                }
                ShardWorkItem item;
                item.deviceId = deviceItem.first;
                item.targetNodeIds.assign(targetSetIter->second.begin(), targetSetIter->second.end());
                item.candidates = &deviceItem.second;
                workItems.push_back(item);
            }
            const size_t shardCount = workItems.size();
            std::vector<ShardRankResult> shardResults(shardCount);

            // 【核心步骤 4】决定执行模式：卡数超过并行阈值且有空余线程时按卡并行，
            // 线程数 = min(空余线程数, 卡数, 线程数上限)（主线程参与干活，只额外创建
            // threadCount-1 个线程）；否则保持串行逐卡，避免小规模场景引入线程创建开销。
            size_t workerThreadCount = 1U;
            size_t idleThreadCount = 1U;
            if (shardCount > SHARDED_REACHABILITY_PARALLEL_RANK_THRESHOLD) {
                idleThreadCount = GetIdleThreadCount();
                workerThreadCount
                    = std::min(std::min(idleThreadCount, shardCount), SHARDED_REACHABILITY_MAX_WORKER_THREADS);
            }
            // 线程数日志：workerThreadCount 为参与干活的线程总数（含主线程），实际额外创建线程数为其减一，
            // 硬件线程数与空余线程数用于核对线程数是如何动态决定的。
            HCCL_VM_INFO(
                "Mem conflict sharded reachability thread plan, rankCount={}, workerThreadCount={}, "
                "createdThreadCount={}, hardwareThreadCount={}, idleThreadCount={}, parallelRankThreshold={}, "
                "maxWorkerThreadCount={}",
                shardCount, workerThreadCount, workerThreadCount - 1U, std::thread::hardware_concurrency(),
                idleThreadCount, SHARDED_REACHABILITY_PARALLEL_RANK_THRESHOLD, SHARDED_REACHABILITY_MAX_WORKER_THREADS);

            if (workerThreadCount > 1U) {
                // 并行模式：原子计数动态领卡，先完成的线程继续领下一张卡，天然负载均衡；
                // 任一卡失败即置位退出标志，其余线程完成在途卡后尽快收工（各线程只写自己的结果槽位）。
                // 独占机器场景（HCCL_VM_BIND_CORE=1）额外按“优先不同物理核”的顺序给各线程绑核；
                // 开关未开启、无可用 CPU 或绑定失败时退化为普通多线程，不影响校验流程与结论。
                std::vector<int> bindCpuList;
                const bool coreBindEnabled = IsThreadCoreBindEnabled();
                if (coreBindEnabled) {
                    bindCpuList = BuildPreferredBindCpuList();
                }
                const auto bindCpuForSlot = [&bindCpuList](size_t slotIndex) -> int {
                    return bindCpuList.empty() ? -1 : bindCpuList[slotIndex % bindCpuList.size()];
                };
                if (!bindCpuList.empty()) {
                    // 绑核计划日志：bindCpuCount 即可绑定的 CPU 数（物理核优先排序），
                    // preferredCpuList 前若干个即各线程槽位实际使用的 CPU。
                    HCCL_VM_INFO(
                        "Mem conflict sharded reachability core bind plan, workerThreadCount={}, "
                        "bindCpuCount={}, preferredCpuList=[{}]",
                        workerThreadCount, bindCpuList.size(), DescribeCpuList(bindCpuList));
                } else if (coreBindEnabled) {
                    HCCL_VM_WARN("Core binding is enabled but no bindable cpu was found, sharded reachability "
                                 "runs without core binding");
                }
                std::atomic<size_t> nextShardIndex(0);
                std::atomic<bool> hasFailure(false);
                auto workerFunc = [&builder, &workItems, &shardResults, &nextShardIndex, &hasFailure, shardCount]() {
                    while (!hasFailure.load(std::memory_order_relaxed)) {
                        const size_t shardIndex = nextShardIndex.fetch_add(1U, std::memory_order_relaxed);
                        if (shardIndex >= shardCount) {
                            return;
                        }
                        ProcessShardRank(
                            builder, shardIndex + 1U, shardCount, workItems[shardIndex], shardResults[shardIndex]);
                        if (shardResults[shardIndex].ret != HCCL_SUCCESS) {
                            hasFailure.store(true, std::memory_order_relaxed);
                        }
                    }
                };
                std::vector<std::thread> workers;
                workers.reserve(workerThreadCount - 1U);
                for (size_t threadIndex = 1U; threadIndex < workerThreadCount; ++threadIndex) {
                    const int bindCpu = bindCpuForSlot(threadIndex);
                    workers.emplace_back([bindCpu, &workerFunc]() {
                        // 工作线程退出即销毁，绑定无需恢复；绑定失败只告警，不影响校验。
                        BindCurrentThreadToCpu(bindCpu);
                        workerFunc();
                    });
                }
                // 主线程同样参与领卡干活；绑核场景下临时绑定 0 号槽位的 CPU，作用域结束自动恢复原亲和性。
                ScopedThreadCpuBinder mainThreadBinder(bindCpuForSlot(0U));
                workerFunc();
                for (std::thread& worker : workers) {
                    worker.join();
                }
            } else {
                // 串行模式：逐卡处理，失败即停（与历史行为一致）。
                for (size_t shardIndex = 0; shardIndex < shardCount; ++shardIndex) {
                    ProcessShardRank(
                        builder, shardIndex + 1U, shardCount, workItems[shardIndex], shardResults[shardIndex]);
                    if (shardResults[shardIndex].ret != HCCL_SUCCESS) {
                        break;
                    }
                }
            }

            // 【核心步骤 5】按卡序合并结果：统计累加所有已处理卡，错误码取卡序最前的失败，
            // 与串行逐卡的结论保持一致；全部成功时才打印收尾汇总日志。
            HcclResult firstFailedRet = HCCL_SUCCESS;
            size_t processedShardCount = 0;
            for (const ShardRankResult& shardResult : shardResults) {
                if (!shardResult.processed) {
                    continue;
                }
                ++processedShardCount;
                stats.orderedCandidatePairCount += shardResult.stats.orderedCandidatePairCount;
                stats.parallelCandidatePairCount += shardResult.stats.parallelCandidatePairCount;
                if (shardResult.ret != HCCL_SUCCESS && firstFailedRet == HCCL_SUCCESS) {
                    firstFailedRet = shardResult.ret;
                }
            }
            if (firstFailedRet != HCCL_SUCCESS) {
                return firstFailedRet;
            }
            // 汇总日志：matrixGenerateCount 即分块矩阵实际生成的次数（真正执行过分片校验的卡数），
            // 与入口的 shard plan 日志、循环内的 shardIndex 相互印证，确认没有漏生成或多生成；
            // hwmKb 与入口日志的 hwmKb 相减即整个分块校验路径的峰值内存增量。
            const SelfStatusMemoryKb finishedMemory = ReadSelfStatusMemoryKb();
            HCCL_VM_INFO(
                "Mem conflict sharded reachability path finished, matrixGenerateCount={}, "
                "candidatePairCount={}, rssKb={}, hwmKb={}",
                processedShardCount, candidates.size(), finishedMemory.rssKb, finishedMemory.hwmKb);
            return HCCL_SUCCESS;
        }

        HcclResult CollectMemoryAccessBuckets(
            const std::vector<const TaskNode*>& nodes, MemoryAccessBuckets& accessBuckets, MemConflictCheckStats& stats)
        {
            accessBuckets.clear();
            std::map<std::string, size_t> accessCountByTaskType;
            std::map<std::string, size_t> accessCountByTaskTypeAndKind;
            std::map<std::string, size_t> accessCountByMemType;
            std::map<std::string, size_t> accessCountByMemTypeAndKind;
            std::map<std::string, size_t> accessCountByBucket;
            for (const TaskNode* node : nodes) {
                if (!IsDataMoveTaskNode(node)) {
                    continue;
                }

                TaskMemoryAccesses currentAccesses;
                HcclResult ret = BuildTaskMemoryAccesses(node, currentAccesses);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                DedupTaskMemoryAccesses(currentAccesses);

                for (const MemoryAccess& access : currentAccesses) {
                    accessBuckets[access.key].push_back(access);
                    ++stats.accessIntervalCount;
                    accessCountByTaskType[TaskTypeName(node->GetType())]++;
                    accessCountByTaskTypeAndKind
                        [std::string(TaskTypeName(node->GetType())) + ":" + AccessKindName(access.kind)]++;
                    accessCountByMemType[DescribeMemType(access.key.memType)]++;
                    accessCountByMemTypeAndKind
                        [std::string(DescribeMemType(access.key.memType)) + ":" + AccessKindName(access.kind)]++;
                    accessCountByBucket[DescribeKey(access.key)]++;
                }
                ++stats.processedDataTaskNodeCount;
            }

            stats.memoryBucketCount = accessBuckets.size();
            LogCountBreakdown("AccessCountByTaskType", accessCountByTaskType);
            LogCountBreakdown("AccessCountByTaskTypeAndKind", accessCountByTaskTypeAndKind);
            LogCountBreakdown("AccessCountByMemType", accessCountByMemType);
            LogCountBreakdown("AccessCountByMemTypeAndKind", accessCountByMemTypeAndKind);
            LogCountBreakdown("AccessCountByBucket", accessCountByBucket);
            return HCCL_SUCCESS;
        }
    } // namespace

    HcclResult CheckDataMoveTaskMemConflict(const TaskNode* start, MemConflictCheckStats* stats)
    {
        const auto scanLineStart = MemConflictClock::now();
        std::vector<const TaskNode*> nodes;
        // 整体流程：
        // 1. BFS 收集可达节点；
        // 2. 为数据搬运类节点抽取读写区间并按内存桶归类；
        // 3. 在每个桶内扫描重叠候选；
        // 4. 只有真正存在前后无序关系时才计算可达闭包做精确判定。
        HcclResult ret = CollectReachableTaskNodes(start, nodes);
        if (ret != HCCL_SUCCESS) {
            LogMemConflictStageTime("CollectReachableTaskNodesToScanMemoryBuckets", "failed", scanLineStart);
            return ret;
        }

        MemConflictCheckStats localStats;
        localStats.nodeCount = nodes.size() + 1U;
        localStats.dataTaskNodeCount = CountDataMoveTaskNodes(nodes);

        MemoryAccessBuckets accessBuckets;
        ret = CollectMemoryAccessBuckets(nodes, accessBuckets, localStats);
        if (ret != HCCL_SUCCESS) {
            LogMemConflictStageTime("CollectReachableTaskNodesToScanMemoryBuckets", "failed", scanLineStart);
            return ret;
        }

        bool needReachabilityClosure = false;
        ConflictCandidates candidates;
        OverlapCandidateStatsCollector overlapCandidateStats;
        ret = ScanMemoryBuckets(accessBuckets, needReachabilityClosure, candidates, localStats, overlapCandidateStats);
        RefreshOverlapCandidateStats(overlapCandidateStats, localStats);
        LogCountBreakdown("OverlapCandidatePairsByBucket", overlapCandidateStats.pairCountByBucket);
        LogCountBreakdown("OverlapCandidatePairsByTaskTypePair", overlapCandidateStats.pairCountByTaskTypePair);
        LogMemConflictStageTime(
            "CollectReachableTaskNodesToScanMemoryBuckets", ret == HCCL_SUCCESS ? "success" : "failed", scanLineStart);
        if (ret != HCCL_SUCCESS) {
            if (stats != nullptr) {
                *stats = localStats;
            }
            return ret;
        }

        if (needReachabilityClosure) {
            if (localStats.nodeCount > REACHABILITY_SHARD_NODE_THRESHOLD) {
                // 节点总数超过阈值：可达闭包按内存归属卡分块构建（卡数超过并行阈值时按卡并行），降低峰值内存。
                ret = CheckCandidatesWithShardedReachability(start, candidates, localStats);
            } else {
                ret = CheckCandidatesWithDenseReachability(start, candidates, localStats);
                // ret = CheckCandidatesWithCRoaringReachability(start, candidates, localStats);
            }
            if (ret != HCCL_SUCCESS) {
                if (stats != nullptr) {
                    *stats = localStats;
                }
                return ret;
            }
        } else {
            LogMemConflictStageTime("GenReachabilityClosure", "skipped", MemConflictClock::now());
        }

        if (stats != nullptr) {
            *stats = localStats;
        }
        return HCCL_SUCCESS;
    }
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
