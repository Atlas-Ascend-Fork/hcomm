/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#include "task_graph_mem_conflict_v3.h"
#include "task_graph_reachability_v3.h"
#include "task_graph_reachability_shard_v3.h"
#include "thread_affinity.h"

namespace HcclSim {
namespace {
    using namespace TaskGraphGeneratorV3;

    // 测试用环境变量守卫：构造时设置、析构时恢复原值，避免用例间泄漏。
    class ScopedEnvGuard {
    public:
        ScopedEnvGuard(const char* name, const char* value) : name_(name)
        {
            const char* originalValue = std::getenv(name);
            hadOriginalValue_ = (originalValue != nullptr);
            if (hadOriginalValue_) {
                originalValue_ = originalValue;
            }
            (void)::setenv(name, value, 1);
        }

        ~ScopedEnvGuard()
        {
            if (hadOriginalValue_) {
                (void)::setenv(name_, originalValue_.c_str(), 1);
            } else {
                (void)::unsetenv(name_);
            }
        }

        ScopedEnvGuard(const ScopedEnvGuard&) = delete;
        ScopedEnvGuard& operator=(const ScopedEnvGuard&) = delete;

    private:
        const char* name_;
        std::string originalValue_;
        bool hadOriginalValue_{false};
    };

    class MemConflictTest : public testing::Test {
    protected:
        void SetUp() override { start_ = AddMainStart(); }

        template <typename T>
        T* AddNode(std::unique_ptr<T> node)
        {
            T* raw = node.get();
            raw->SetNodeId(static_cast<NodeId>(nodes_.size()));
            nodes_.emplace_back(std::move(node));
            return raw;
        }

        TaskStart* AddMainStart()
        {
            auto node = std::make_unique<TaskStart>(BoundaryType::MAIN_GRAPH);
            node->SetNodeId(MAIN_START_NODE_ID);
            TaskStart* raw = node.get();
            nodes_.emplace_back(std::move(node));
            return raw;
        }

        TaskTransMem* AddTransMem(RankId dstRank, uint64_t dstOffset, uint64_t len)
        {
            MemSlice src;
            src.rankId = 9U;
            src.deviceId = 9U;
            src.memType = MemType::INPUT;
            src.offset = 0x10000U;
            src.rawAddr = 0x10000U;
            src.len = len;
            MemSlice dst;
            dst.rankId = dstRank;
            dst.deviceId = dstRank;
            dst.memType = MemType::CCL;
            dst.offset = dstOffset;
            dst.rawAddr = 0x20000U + dstOffset;
            dst.len = len;
            return AddNode(std::make_unique<TaskTransMem>(src, dst));
        }

        // 非数据搬运节点，只用于把大图撑过 REACHABILITY_SHARD_NODE_THRESHOLD。
        TaskRecordAICPU* AddFiller(uint32_t notifyId)
        {
            AicpuNotify notify;
            notify.recordDeviceId = 0;
            notify.waitDeviceId = 0;
            notify.notifyId = notifyId;
            return AddNode(std::make_unique<TaskRecordAICPU>(notify, ProtocolType::SDMA));
        }

        // mainStart -> filler* -> tail，返回链尾节点。
        TaskNode* AddFillerChain(TaskNode* from, size_t fillerCount)
        {
            TaskNode* prev = from;
            for (size_t index = 0; index < fillerCount; ++index) {
                TaskRecordAICPU* filler = AddFiller(static_cast<uint32_t>(index));
                prev->AddChild(filler);
                prev = filler;
            }
            return prev;
        }

        TaskStart* start_ = nullptr;
        std::vector<std::unique_ptr<TaskNode>> nodes_;
    };

    // INPUT 与 OUTPUT 同址时必须识别并发写冲突，不能再按逻辑类型隔离。
    TEST_F(MemConflictTest, AbsoluteAliasesShareConflictBucket)
    {
        MemSlice src;
        src.deviceId = 9;
        src.memType = MemType::INPUT;
        src.rawAddr = 0x10000;
        src.len = 64;
        MemSlice dst = src;
        dst.deviceId = 0;
        dst.rawAddr = 0x20000;
        auto* first = AddNode(std::make_unique<TaskTransMem>(src, dst));
        dst.memType = MemType::OUTPUT;
        auto* second = AddNode(std::make_unique<TaskTransMem>(src, dst));
        start_->AddChild(first);
        start_->AddChild(second);
        EXPECT_EQ(CheckDataMoveTaskMemConflict(start_), HCCL_E_MEMORY);
    }

    // MS 的原始地址可为空，槽位 0 合法；同一槽位的并发写仍须报冲突。
    TEST_F(MemConflictTest, MsSlotZeroUsesIndependentAddressSpace)
    {
        MemSlice src;
        src.deviceId = 9;
        src.memType = MemType::INPUT;
        src.rawAddr = 0x10000;
        src.len = 64;
        MemSlice dst;
        dst.deviceId = 0;
        dst.memType = MemType::MS_CCU;
        dst.len = 64;
        auto* first = AddNode(std::make_unique<TaskTransMem>(src, dst));
        auto* second = AddNode(std::make_unique<TaskTransMem>(src, dst));
        start_->AddChild(first);
        start_->AddChild(second);
        EXPECT_EQ(CheckDataMoveTaskMemConflict(start_), HCCL_E_MEMORY);
    }

    // 小图（节点总数不超过阈值）走全量稠密闭包：两个并行写互相重叠 -> 报冲突。
    TEST_F(MemConflictTest, DensePathDetectsParallelWriteConflict)
    {
        TaskTransMem* taskA = AddTransMem(0U, 0x0U, 0x100U);
        TaskTransMem* taskB = AddTransMem(0U, 0x80U, 0x100U);
        start_->AddChild(taskA);
        start_->AddChild(taskB);

        MemConflictCheckStats stats;
        EXPECT_EQ(CheckDataMoveTaskMemConflict(start_, &stats), HCCL_E_MEMORY);
        EXPECT_EQ(stats.nodeCount, 3U);
        EXPECT_EQ(stats.parallelCandidatePairCount, 1U);
    }

    // 小图：图上已有先后约束时，写重叠不算并发冲突。
    TEST_F(MemConflictTest, DensePathAllowsOrderedWrites)
    {
        TaskTransMem* taskA = AddTransMem(0U, 0x0U, 0x100U);
        TaskTransMem* taskB = AddTransMem(0U, 0x80U, 0x100U);
        start_->AddChild(taskA);
        taskA->AddChild(taskB);

        MemConflictCheckStats stats;
        EXPECT_EQ(CheckDataMoveTaskMemConflict(start_, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.orderedCandidatePairCount, 1U);
    }

    // 大图（节点总数超过阈值）走按卡分片闭包：并行写互相重叠 -> 报冲突。
    TEST_F(MemConflictTest, ShardedPathDetectsParallelWriteConflict)
    {
        TaskNode* tail = AddFillerChain(start_, REACHABILITY_SHARD_NODE_THRESHOLD - 1U);
        TaskTransMem* taskA = AddTransMem(0U, 0x0U, 0x100U);
        TaskTransMem* taskB = AddTransMem(0U, 0x80U, 0x100U);
        tail->AddChild(taskA);
        tail->AddChild(taskB);

        MemConflictCheckStats stats;
        EXPECT_EQ(CheckDataMoveTaskMemConflict(start_, &stats), HCCL_E_MEMORY);
        EXPECT_GT(stats.nodeCount, REACHABILITY_SHARD_NODE_THRESHOLD);
        EXPECT_EQ(stats.parallelCandidatePairCount, 1U);
    }

    // 填充节点数饱和计算：阈值较小时（如 1）避免无符号下溢导致 filler 链长度爆炸。
    size_t SaturatedFillerCount(size_t need)
    {
        return (REACHABILITY_SHARD_NODE_THRESHOLD > need) ? (REACHABILITY_SHARD_NODE_THRESHOLD - need) : 0U;
    }

    // 大图：分片路径下已有先后约束的写重叠不算冲突。
    TEST_F(MemConflictTest, ShardedPathAllowsOrderedWrites)
    {
        TaskNode* tail = AddFillerChain(start_, SaturatedFillerCount(2U));
        TaskTransMem* taskA = AddTransMem(0U, 0x0U, 0x100U);
        TaskTransMem* taskB = AddTransMem(0U, 0x80U, 0x100U);
        tail->AddChild(taskA);
        taskA->AddChild(taskB);

        MemConflictCheckStats stats;
        EXPECT_EQ(CheckDataMoveTaskMemConflict(start_, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.orderedCandidatePairCount, 1U);
    }

    // 大图：不同卡各自独立分片校验，0 卡有序、1 卡并行冲突 -> 仍报冲突。
    TEST_F(MemConflictTest, ShardedPathChecksEachRankIndependently)
    {
        TaskNode* tail = AddFillerChain(start_, SaturatedFillerCount(4U));
        TaskTransMem* orderedA = AddTransMem(0U, 0x0U, 0x100U);
        TaskTransMem* orderedB = AddTransMem(0U, 0x80U, 0x100U);
        tail->AddChild(orderedA);
        orderedA->AddChild(orderedB);
        TaskTransMem* parallelC = AddTransMem(1U, 0x0U, 0x100U);
        TaskTransMem* parallelD = AddTransMem(1U, 0x80U, 0x100U);
        tail->AddChild(parallelC);
        tail->AddChild(parallelD);

        MemConflictCheckStats stats;
        EXPECT_EQ(CheckDataMoveTaskMemConflict(start_, &stats), HCCL_E_MEMORY);
        EXPECT_EQ(stats.orderedCandidatePairCount, 1U);
        EXPECT_EQ(stats.parallelCandidatePairCount, 1U);
    }

    // 卡数超过并行阈值（并行路径）：每张卡内两个写有序，全部卡成功 -> 统计按卡序完整合并。
    TEST_F(MemConflictTest, ParallelShardedPathMergesOrderedStatsFromAllRanks)
    {
        const size_t rankCount = SHARDED_REACHABILITY_PARALLEL_RANK_THRESHOLD + 1U;
        for (size_t rankId = 0; rankId < rankCount; ++rankId) {
            TaskTransMem* taskA = AddTransMem(static_cast<RankId>(rankId), 0x0U, 0x100U);
            TaskTransMem* taskB = AddTransMem(static_cast<RankId>(rankId), 0x80U, 0x100U);
            start_->AddChild(taskA);
            taskA->AddChild(taskB);
        }

        MemConflictCheckStats stats;
        EXPECT_EQ(CheckDataMoveTaskMemConflict(start_, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.orderedCandidatePairCount, rankCount);
        EXPECT_EQ(stats.parallelCandidatePairCount, 0U);
    }

    // 卡数超过并行阈值（并行路径）：每张卡内两个并行写互相重叠 -> 仍报冲突。
    // 失败早退下实际处理几张卡不确定，只校验返回码与至少发现一个冲突。
    TEST_F(MemConflictTest, ParallelShardedPathDetectsParallelWriteConflict)
    {
        const size_t rankCount = SHARDED_REACHABILITY_PARALLEL_RANK_THRESHOLD + 1U;
        for (size_t rankId = 0; rankId < rankCount; ++rankId) {
            TaskTransMem* taskA = AddTransMem(static_cast<RankId>(rankId), 0x0U, 0x100U);
            TaskTransMem* taskB = AddTransMem(static_cast<RankId>(rankId), 0x80U, 0x100U);
            start_->AddChild(taskA);
            start_->AddChild(taskB);
        }

        MemConflictCheckStats stats;
        EXPECT_EQ(CheckDataMoveTaskMemConflict(start_, &stats), HCCL_E_MEMORY);
        EXPECT_GE(stats.parallelCandidatePairCount, 1U);
        EXPECT_LE(stats.parallelCandidatePairCount, rankCount);
    }

    // 绑核开关开启（独占机器场景）时，并行分片路径结论与不绑核一致：绑核只影响线程调度。
    // 单核或满载环境退化为串行路径时结论同样保持一致。
    TEST_F(MemConflictTest, ParallelShardedPathResultUnchangedWithCoreBindingEnabled)
    {
        ScopedEnvGuard bindGuard(THREAD_CORE_BIND_ENV, "1");
        const size_t rankCount = SHARDED_REACHABILITY_PARALLEL_RANK_THRESHOLD + 1U;
        for (size_t rankId = 0; rankId < rankCount; ++rankId) {
            TaskTransMem* taskA = AddTransMem(static_cast<RankId>(rankId), 0x0U, 0x100U);
            TaskTransMem* taskB = AddTransMem(static_cast<RankId>(rankId), 0x80U, 0x100U);
            start_->AddChild(taskA);
            taskA->AddChild(taskB);
        }

        MemConflictCheckStats stats;
        EXPECT_EQ(CheckDataMoveTaskMemConflict(start_, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.orderedCandidatePairCount, rankCount);
        EXPECT_EQ(stats.parallelCandidatePairCount, 0U);
    }

    // 分片闭包与全量闭包在相同数据节点对上结果一致；非数据搬运节点不允许作为分片目标。
    TEST_F(MemConflictTest, ShardedClosureMatchesDenseClosure)
    {
        TaskTransMem* task0 = AddTransMem(0U, 0x0U, 0x10U);
        TaskTransMem* task1 = AddTransMem(0U, 0x100U, 0x10U);
        TaskTransMem* task2 = AddTransMem(0U, 0x200U, 0x10U);
        TaskTransMem* task3 = AddTransMem(0U, 0x300U, 0x10U);
        TaskTransMem* task4 = AddTransMem(1U, 0x400U, 0x10U);
        TaskRecordAICPU* filler = AddFiller(0);
        start_->AddChild(task0);
        task0->AddChild(task1);
        task0->AddChild(task2);
        task1->AddChild(task3);
        task2->AddChild(task3);
        task3->AddChild(task4);
        start_->AddChild(filler);

        ReachabilityClosure denseClosure;
        ASSERT_EQ(GenReachabilityClosure(start_, denseClosure), HCCL_SUCCESS);

        ReachabilityShardBuilder builder;
        ASSERT_EQ(builder.Init(start_), HCCL_SUCCESS);
        EXPECT_EQ(builder.GetNodeCount(), nodes_.size() - 1U);

        const std::vector<NodeId> dataNodeIds
            = {task0->GetNodeId(), task1->GetNodeId(), task2->GetNodeId(), task3->GetNodeId(), task4->GetNodeId()};
        ReachabilityClosure shardedClosure;
        ASSERT_EQ(builder.BuildClosureForNodes(dataNodeIds, shardedClosure), HCCL_SUCCESS);
        EXPECT_EQ(shardedClosure.dataIndexByNodeId.size(), dataNodeIds.size());

        for (const NodeId fromNodeId : dataNodeIds) {
            for (const NodeId toNodeId : dataNodeIds) {
                bool denseReachable = false;
                bool shardedReachable = false;
                ASSERT_EQ(IsReachable(denseClosure, fromNodeId, toNodeId, denseReachable), HCCL_SUCCESS);
                ASSERT_EQ(IsReachable(shardedClosure, fromNodeId, toNodeId, shardedReachable), HCCL_SUCCESS);
                EXPECT_EQ(shardedReachable, denseReachable) << "from=" << fromNodeId << ", to=" << toNodeId;
            }
        }
        EXPECT_EQ(shardedClosure.matrix.size(), denseClosure.matrix.size());

        // 非数据搬运节点（filler）不允许作为分片目标。
        ReachabilityClosure invalidClosure;
        EXPECT_NE(builder.BuildClosureForNodes({filler->GetNodeId()}, invalidClosure), HCCL_SUCCESS);
    }

    // 通用对拍：分片闭包（内存节点压缩图路径）与全量闭包在给定数据节点集上逐对结果必须一致。
    void ExpectShardedClosureMatchesDense(const TaskNode* start, const std::vector<NodeId>& dataNodeIds)
    {
        ReachabilityClosure denseClosure;
        ASSERT_EQ(GenReachabilityClosure(start, denseClosure), HCCL_SUCCESS);

        ReachabilityShardBuilder builder;
        ASSERT_EQ(builder.Init(start), HCCL_SUCCESS);

        ReachabilityClosure shardedClosure;
        ASSERT_EQ(builder.BuildClosureForNodes(dataNodeIds, shardedClosure), HCCL_SUCCESS);
        ASSERT_EQ(shardedClosure.matrix.size(), dataNodeIds.size());

        for (const NodeId fromNodeId : dataNodeIds) {
            for (const NodeId toNodeId : dataNodeIds) {
                bool denseReachable = false;
                bool shardedReachable = false;
                ASSERT_EQ(IsReachable(denseClosure, fromNodeId, toNodeId, denseReachable), HCCL_SUCCESS);
                ASSERT_EQ(IsReachable(shardedClosure, fromNodeId, toNodeId, shardedReachable), HCCL_SUCCESS);
                EXPECT_EQ(shardedReachable, denseReachable) << "from=" << fromNodeId << ", to=" << toNodeId;
            }
        }
    }

    // 压缩图：A(内存) 的孩子为 B(非内存) 与 C(内存)，B 的孩子为 D、E(内存)——
    // 求 A 的行必须等价于直接使用 C、D、E（B 折叠进压缩边），与全量闭包逐对一致。
    TEST_F(MemConflictTest, ShardedClosureFoldsNonMemDiamondIntermediate)
    {
        TaskTransMem* taskA = AddTransMem(0U, 0x0U, 0x10U);
        TaskRecordAICPU* fillerB = AddFiller(100U);
        TaskTransMem* taskC = AddTransMem(0U, 0x100U, 0x10U);
        TaskTransMem* taskD = AddTransMem(0U, 0x200U, 0x10U);
        TaskTransMem* taskE = AddTransMem(0U, 0x300U, 0x10U);
        TaskTransMem* taskF = AddTransMem(0U, 0x400U, 0x10U);
        start_->AddChild(taskA);
        start_->AddChild(taskF);
        taskA->AddChild(fillerB);
        taskA->AddChild(taskC);
        fillerB->AddChild(taskD);
        fillerB->AddChild(taskE);

        const std::vector<NodeId> dataNodeIds
            = {taskA->GetNodeId(), taskC->GetNodeId(), taskD->GetNodeId(), taskE->GetNodeId(), taskF->GetNodeId()};
        ExpectShardedClosureMatchesDense(start_, dataNodeIds);

        // A 经非内存 B 中转到达 D/E、直达 C 的语义必须在压缩后保留；无关叶子 F 不可达。
        ReachabilityShardBuilder builder;
        ASSERT_EQ(builder.Init(start_), HCCL_SUCCESS);
        ReachabilityClosure closure;
        ASSERT_EQ(builder.BuildClosureForNodes(dataNodeIds, closure), HCCL_SUCCESS);
        const std::vector<std::pair<TaskTransMem*, TaskTransMem*>> expectedReachable
            = {{taskA, taskC}, {taskA, taskD}, {taskA, taskE}};
        for (const auto& pair : expectedReachable) {
            bool reachable = false;
            ASSERT_EQ(IsReachable(closure, pair.first->GetNodeId(), pair.second->GetNodeId(), reachable), HCCL_SUCCESS);
            EXPECT_TRUE(reachable) << "from=" << pair.first->GetNodeId() << ", to=" << pair.second->GetNodeId();
        }
        const std::vector<std::pair<TaskTransMem*, TaskTransMem*>> expectedUnreachable
            = {{taskA, taskF}, {taskD, taskE}, {taskE, taskD}, {taskF, taskA}};
        for (const auto& pair : expectedUnreachable) {
            bool reachable = false;
            ASSERT_EQ(IsReachable(closure, pair.first->GetNodeId(), pair.second->GetNodeId(), reachable), HCCL_SUCCESS);
            EXPECT_FALSE(reachable) << "from=" << pair.first->GetNodeId() << ", to=" << pair.second->GetNodeId();
        }
    }

    // 压缩图：两个内存节点共享同一个非内存中转 X（X 再到两个内存节点）——
    // 共享中转不得漏传（两父均可达 X 的下游）也不得串扰（两父之间互相不可达）。
    TEST_F(MemConflictTest, ShardedClosureHandlesSharedNonMemIntermediate)
    {
        TaskTransMem* taskM1 = AddTransMem(0U, 0x0U, 0x10U);
        TaskTransMem* taskM2 = AddTransMem(0U, 0x100U, 0x10U);
        TaskRecordAICPU* fillerX = AddFiller(200U);
        TaskTransMem* taskM3 = AddTransMem(0U, 0x200U, 0x10U);
        TaskTransMem* taskM4 = AddTransMem(0U, 0x300U, 0x10U);
        start_->AddChild(taskM1);
        start_->AddChild(taskM2);
        taskM1->AddChild(fillerX);
        taskM2->AddChild(fillerX);
        fillerX->AddChild(taskM3);
        fillerX->AddChild(taskM4);

        const std::vector<NodeId> dataNodeIds
            = {taskM1->GetNodeId(), taskM2->GetNodeId(), taskM3->GetNodeId(), taskM4->GetNodeId()};
        ExpectShardedClosureMatchesDense(start_, dataNodeIds);

        ReachabilityShardBuilder builder;
        ASSERT_EQ(builder.Init(start_), HCCL_SUCCESS);
        ReachabilityClosure closure;
        ASSERT_EQ(builder.BuildClosureForNodes(dataNodeIds, closure), HCCL_SUCCESS);
        bool m1ToM3 = false;
        bool m1ToM4 = false;
        bool m2ToM3 = false;
        bool m2ToM4 = false;
        bool m1ToM2 = false;
        ASSERT_EQ(IsReachable(closure, taskM1->GetNodeId(), taskM3->GetNodeId(), m1ToM3), HCCL_SUCCESS);
        ASSERT_EQ(IsReachable(closure, taskM1->GetNodeId(), taskM4->GetNodeId(), m1ToM4), HCCL_SUCCESS);
        ASSERT_EQ(IsReachable(closure, taskM2->GetNodeId(), taskM3->GetNodeId(), m2ToM3), HCCL_SUCCESS);
        ASSERT_EQ(IsReachable(closure, taskM2->GetNodeId(), taskM4->GetNodeId(), m2ToM4), HCCL_SUCCESS);
        ASSERT_EQ(IsReachable(closure, taskM1->GetNodeId(), taskM2->GetNodeId(), m1ToM2), HCCL_SUCCESS);
        EXPECT_TRUE(m1ToM3);
        EXPECT_TRUE(m1ToM4);
        EXPECT_TRUE(m2ToM3);
        EXPECT_TRUE(m2ToM4);
        EXPECT_FALSE(m1ToM2);
    }

    // 压缩图：内存节点间隔多层非内存长链——压缩边必须正确跨链，与全量闭包逐对一致。
    TEST_F(MemConflictTest, ShardedClosureCrossesDeepNonMemChain)
    {
        const size_t chainLen = 64U;
        TaskTransMem* taskA = AddTransMem(0U, 0x0U, 0x10U);
        start_->AddChild(taskA);
        TaskNode* mid = AddFillerChain(taskA, chainLen);
        TaskTransMem* taskB = AddTransMem(0U, 0x100U, 0x10U);
        mid->AddChild(taskB);
        TaskNode* tail = AddFillerChain(taskB, chainLen);
        TaskTransMem* taskC = AddTransMem(0U, 0x200U, 0x10U);
        tail->AddChild(taskC);

        const std::vector<NodeId> dataNodeIds = {taskA->GetNodeId(), taskB->GetNodeId(), taskC->GetNodeId()};
        ExpectShardedClosureMatchesDense(start_, dataNodeIds);

        ReachabilityShardBuilder builder;
        ASSERT_EQ(builder.Init(start_), HCCL_SUCCESS);
        ReachabilityClosure closure;
        ASSERT_EQ(builder.BuildClosureForNodes(dataNodeIds, closure), HCCL_SUCCESS);
        bool aToB = false;
        bool aToC = false;
        bool bToC = false;
        bool cToA = false;
        ASSERT_EQ(IsReachable(closure, taskA->GetNodeId(), taskB->GetNodeId(), aToB), HCCL_SUCCESS);
        ASSERT_EQ(IsReachable(closure, taskA->GetNodeId(), taskC->GetNodeId(), aToC), HCCL_SUCCESS);
        ASSERT_EQ(IsReachable(closure, taskB->GetNodeId(), taskC->GetNodeId(), bToC), HCCL_SUCCESS);
        ASSERT_EQ(IsReachable(closure, taskC->GetNodeId(), taskA->GetNodeId(), cToA), HCCL_SUCCESS);
        EXPECT_TRUE(aToB);
        EXPECT_TRUE(aToC);
        EXPECT_TRUE(bToC);
        EXPECT_FALSE(cToA);
    }

    // 压缩图：单目标分片（叶子内存节点，memSucc 为空）——矩阵退化为 1 行全 0，自查询为不可达。
    TEST_F(MemConflictTest, ShardedClosureSingleLeafTargetHasEmptyRow)
    {
        TaskNode* tail = AddFillerChain(start_, REACHABILITY_SHARD_NODE_THRESHOLD);
        TaskTransMem* taskA = AddTransMem(0U, 0x0U, 0x10U);
        tail->AddChild(taskA);

        ReachabilityShardBuilder builder;
        ASSERT_EQ(builder.Init(start_), HCCL_SUCCESS);
        ReachabilityClosure closure;
        ASSERT_EQ(builder.BuildClosureForNodes({taskA->GetNodeId()}, closure), HCCL_SUCCESS);
        ASSERT_EQ(closure.matrix.size(), 1U);
        ASSERT_EQ(closure.matrix[0].size(), 1U);
        EXPECT_EQ(closure.matrix[0][0], 0U);

        bool selfReachable = true;
        ASSERT_EQ(IsReachable(closure, taskA->GetNodeId(), taskA->GetNodeId(), selfReachable), HCCL_SUCCESS);
        EXPECT_FALSE(selfReachable);
    }

    // 绑核开关：仅 1/true/on（忽略大小写）开启，其余取值关闭。
    TEST(ThreadAffinityTest, CoreBindSwitchFollowsEnvironmentValue)
    {
        {
            ScopedEnvGuard guard(THREAD_CORE_BIND_ENV, "0");
            EXPECT_FALSE(IsThreadCoreBindEnabled());
        }
        {
            ScopedEnvGuard guard(THREAD_CORE_BIND_ENV, "off");
            EXPECT_FALSE(IsThreadCoreBindEnabled());
        }
        {
            ScopedEnvGuard guard(THREAD_CORE_BIND_ENV, "1");
            EXPECT_TRUE(IsThreadCoreBindEnabled());
        }
        {
            ScopedEnvGuard guard(THREAD_CORE_BIND_ENV, "TRUE");
            EXPECT_TRUE(IsThreadCoreBindEnabled());
        }
        {
            ScopedEnvGuard guard(THREAD_CORE_BIND_ENV, "On");
            EXPECT_TRUE(IsThreadCoreBindEnabled());
        }
        {
            ScopedEnvGuard guard(THREAD_CORE_BIND_ENV, "0");
            EXPECT_FALSE(IsThreadCoreBindEnabled());
        }
    }

    // thread_siblings_list 解析：取列表中最小 CPU 号，非法内容返回 -1。
    TEST(ThreadAffinityTest, ParseSiblingListMinCpuHandlesCommonFormats)
    {
        EXPECT_EQ(ParseSiblingListMinCpu("0"), 0);
        EXPECT_EQ(ParseSiblingListMinCpu("0,1"), 0);
        EXPECT_EQ(ParseSiblingListMinCpu("1,0"), 0);
        EXPECT_EQ(ParseSiblingListMinCpu("0-3,8"), 0);
        EXPECT_EQ(ParseSiblingListMinCpu("8,10-11"), 8);
        EXPECT_EQ(ParseSiblingListMinCpu("4-6"), 4);
        EXPECT_EQ(ParseSiblingListMinCpu(""), -1);
        EXPECT_EQ(ParseSiblingListMinCpu("abc"), -1);
        EXPECT_EQ(ParseSiblingListMinCpu("abc,2"), 2);
    }

#ifdef __linux__
    // 优选绑核列表：元素无重复、都在进程亲和集内，且同一物理核（SMT 组）的首个成员排在兄弟之前。
    TEST(ThreadAffinityTest, PreferredBindCpuListMatchesProcessAffinity)
    {
        cpu_set_t allowedSet;
        CPU_ZERO(&allowedSet);
        ASSERT_EQ(sched_getaffinity(0, sizeof(allowedSet), &allowedSet), 0);

        const std::vector<int> cpuList = BuildPreferredBindCpuList();
        ASSERT_FALSE(cpuList.empty());

        const std::set<int> uniqueCpus(cpuList.begin(), cpuList.end());
        EXPECT_EQ(uniqueCpus.size(), cpuList.size());
        for (const int cpuId : cpuList) {
            EXPECT_NE(CPU_ISSET(cpuId, &allowedSet), 0);
        }

        std::map<int, size_t> firstPosByGroupKey;
        for (size_t index = 0; index < cpuList.size(); ++index) {
            const int groupKey = GetSiblingGroupKey(cpuList[index]);
            const auto iter = firstPosByGroupKey.find(groupKey);
            if (iter == firstPosByGroupKey.end()) {
                firstPosByGroupKey.emplace(groupKey, index);
            } else {
                EXPECT_LT(iter->second, index);
            }
        }
    }

    // 主线程 RAII 绑核器：绑定后只允许目标 CPU，析构后恢复原亲和性（主线程还要跑后续阶段）。
    TEST(ThreadAffinityTest, ScopedBinderBindsAndRestoresAffinity)
    {
        cpu_set_t beforeSet;
        CPU_ZERO(&beforeSet);
        ASSERT_EQ(pthread_getaffinity_np(pthread_self(), sizeof(beforeSet), &beforeSet), 0);

        const std::vector<int> cpuList = BuildPreferredBindCpuList();
        ASSERT_FALSE(cpuList.empty());
        const int targetCpu = cpuList.front();
        {
            ScopedThreadCpuBinder binder(targetCpu);
            EXPECT_TRUE(binder.IsBound());
            cpu_set_t duringSet;
            CPU_ZERO(&duringSet);
            ASSERT_EQ(pthread_getaffinity_np(pthread_self(), sizeof(duringSet), &duringSet), 0);
            EXPECT_NE(CPU_ISSET(targetCpu, &duringSet), 0);
        }

        cpu_set_t afterSet;
        CPU_ZERO(&afterSet);
        ASSERT_EQ(pthread_getaffinity_np(pthread_self(), sizeof(afterSet), &afterSet), 0);
        EXPECT_NE(CPU_EQUAL(&beforeSet, &afterSet), 0);
    }

    // 无效 CPU 号（负数）不绑定、不报错。
    TEST(ThreadAffinityTest, BindCurrentThreadRejectsInvalidCpu)
    {
        EXPECT_FALSE(BindCurrentThreadToCpu(-1));
        ScopedThreadCpuBinder invalidBinder(-1);
        EXPECT_FALSE(invalidBinder.IsBound());
    }
#endif
} // namespace
} // namespace HcclSim
