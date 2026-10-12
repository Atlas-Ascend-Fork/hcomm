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
#include <memory>
#include <random>
#include <set>
#include <vector>

#include "task_graph_reachability_shard_v3.h"
#include "task_graph_reachability_v3.h"

namespace HcclSim {
namespace {
    using namespace TaskGraphGeneratorV3;

    // 压力测试：随机 DAG + 各种真实图结构（多父、菱形、孤儿子图、高入度、深链、宽扇出），
    // 验证 BuildShardContext/BuildTargetedReachabilityClosure 的一致性。
    class ShardReachabilityStressTest : public testing::Test {
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
            src.rankId = dstRank;
            src.deviceId = dstRank;
            src.memType = MemType::INPUT;
            src.offset = 0x10000U;
            src.len = len;
            MemSlice dst;
            dst.rankId = dstRank;
            dst.deviceId = dstRank;
            dst.memType = MemType::CCL;
            dst.offset = dstOffset;
            dst.len = len;
            return AddNode(std::make_unique<TaskTransMem>(src, dst));
        }

        TaskRecordAICPU* AddRecord(uint32_t notifyId)
        {
            AicpuNotify notify;
            notify.recordDeviceId = 0;
            notify.waitDeviceId = 0;
            notify.notifyId = notifyId;
            return AddNode(std::make_unique<TaskRecordAICPU>(notify, ProtocolType::SDMA));
        }

        TaskStart* start_ = nullptr;
        std::vector<std::unique_ptr<TaskNode>> nodes_;
    };

    // 随机 DAG 压力：任意形状（多父/菱形/链/扇出），验证 Init + BuildClosureForNodes 不报错
    // 且与稠密闭包结果一致。
    TEST_F(ShardReachabilityStressTest, RandomDagShardMatchesDense)
    {
        for (uint32_t round = 0; round < 200; ++round) {
            std::mt19937 rng(round);
            const size_t nodeCount = 5 + rng() % 120;

            // 生成随机 DAG：节点 i 可以指向 j > i。
            std::vector<TaskTransMem*> tasks;
            tasks.reserve(nodeCount);
            for (size_t i = 0; i < nodeCount; ++i) {
                tasks.push_back(AddTransMem(rng() % 4, rng() % 0x1000U, 0x100U));
            }
            // start -> 第一个节点
            start_->AddChild(tasks[0]);
            for (size_t i = 0; i + 1 < nodeCount; ++i) {
                // 每个节点至少连向下一个（保证全图可达）
                tasks[i]->AddChild(tasks[i + 1]);
                // 随机前向边
                const size_t extraEdges = rng() % 3;
                for (size_t e = 0; e < extraEdges; ++e) {
                    const size_t j = i + 1 + rng() % (nodeCount - i - 1);
                    tasks[i]->AddChild(tasks[j]);
                }
            }

            ReachabilityShardBuilder builder;
            ASSERT_EQ(builder.Init(start_), HCCL_SUCCESS) << "round=" << round;

            // 全部数据节点作为目标
            std::vector<NodeId> allIds;
            for (const TaskTransMem* task : tasks) {
                allIds.push_back(task->GetNodeId());
            }
            ReachabilityClosure shardedClosure;
            ASSERT_EQ(builder.BuildClosureForNodes(allIds, shardedClosure), HCCL_SUCCESS) << "round=" << round;

            // 与稠密闭包对比
            ReachabilityClosure denseClosure;
            ASSERT_EQ(GenReachabilityClosure(start_, denseClosure), HCCL_SUCCESS) << "round=" << round;
            for (const NodeId fromId : allIds) {
                for (const NodeId toId : allIds) {
                    bool denseReachable = false;
                    bool shardedReachable = false;
                    ASSERT_EQ(IsReachable(denseClosure, fromId, toId, denseReachable), HCCL_SUCCESS);
                    ASSERT_EQ(IsReachable(shardedClosure, fromId, toId, shardedReachable), HCCL_SUCCESS);
                    ASSERT_EQ(shardedReachable, denseReachable)
                        << "round=" << round << ", from=" << fromId << ", to=" << toId;
                }
            }

            // 随机子集目标也要一致（多次分片）
            for (uint32_t shardRound = 0; shardRound < 5; ++shardRound) {
                std::vector<NodeId> subsetIds;
                for (const NodeId id : allIds) {
                    if (rng() % 2 == 0) {
                        subsetIds.push_back(id);
                    }
                }
                ReachabilityClosure subsetClosure;
                ASSERT_EQ(builder.BuildClosureForNodes(subsetIds, subsetClosure), HCCL_SUCCESS)
                    << "round=" << round << ", shardRound=" << shardRound;
            }

            // 重建下一轮之前清空图
            nodes_.clear();
            start_ = AddMainStart();
        }
    }

    // 孤儿子图边指向可达节点：notify 边 record(孤儿)->wait(可达)，Kahn/BFS 一致性。
    TEST_F(ShardReachabilityStressTest, OrphanEdgeIntoReachableSet)
    {
        // 主链：start -> A -> B
        TaskTransMem* taskA = AddTransMem(0U, 0x0U, 0x100U);
        TaskTransMem* taskB = AddTransMem(0U, 0x80U, 0x100U);
        start_->AddChild(taskA);
        taskA->AddChild(taskB);
        // 孤儿 record（不从 start 可达）指向 taskB
        TaskRecordAICPU* orphanRecord = AddRecord(7);
        orphanRecord->AddChild(taskB);

        ReachabilityShardBuilder builder;
        EXPECT_EQ(builder.Init(start_), HCCL_SUCCESS);
        ReachabilityClosure closure;
        const std::vector<NodeId> closureIds = {taskA->GetNodeId(), taskB->GetNodeId()};
        EXPECT_EQ(builder.BuildClosureForNodes(closureIds, closure), HCCL_SUCCESS);
    }

    // start 的孩子同时也是普通节点的孩子（多父）。
    TEST_F(ShardReachabilityStressTest, ChildOfStartAndRegularNode)
    {
        TaskTransMem* taskA = AddTransMem(0U, 0x0U, 0x100U);
        TaskTransMem* taskB = AddTransMem(0U, 0x80U, 0x100U);
        TaskTransMem* taskC = AddTransMem(0U, 0x100U, 0x100U);
        start_->AddChild(taskA);
        start_->AddChild(taskC); // C 既是 start 的孩子
        taskA->AddChild(taskB);
        taskB->AddChild(taskC); // 也是 B 的孩子（多父）

        ReachabilityShardBuilder builder;
        EXPECT_EQ(builder.Init(start_), HCCL_SUCCESS);
        ReachabilityClosure closure;
        EXPECT_EQ(
            builder.BuildClosureForNodes({taskA->GetNodeId(), taskB->GetNodeId(), taskC->GetNodeId()}, closure),
            HCCL_SUCCESS);
        bool reachable = false;
        EXPECT_EQ(IsReachable(closure, taskA->GetNodeId(), taskC->GetNodeId(), reachable), HCCL_SUCCESS);
        EXPECT_TRUE(reachable);
    }

    // 深链 + 高入度 + 宽扇出。
    TEST_F(ShardReachabilityStressTest, DeepChainHighIndegreeWideFanout)
    {
        const size_t chainLen = 2000;
        std::vector<TaskTransMem*> chain;
        for (size_t i = 0; i < chainLen; ++i) {
            chain.push_back(AddTransMem(0U, static_cast<uint64_t>(i) * 0x100U, 0x100U));
        }
        start_->AddChild(chain[0]);
        for (size_t i = 0; i + 1 < chainLen; ++i) {
            chain[i]->AddChild(chain[i + 1]);
        }
        // 宽扇出：chain[0] -> 后半段全部
        for (size_t i = chainLen / 2; i < chainLen; ++i) {
            chain[0]->AddChild(chain[i]);
        }
        // 高入度：所有前 100 个节点 -> chain 尾
        for (size_t i = 0; i < 100 && i < chainLen; ++i) {
            chain[i]->AddChild(chain[chainLen - 1]);
        }

        ReachabilityShardBuilder builder;
        ASSERT_EQ(builder.Init(start_), HCCL_SUCCESS);
        std::vector<NodeId> ids;
        for (const TaskTransMem* task : chain) {
            ids.push_back(task->GetNodeId());
        }
        ReachabilityClosure closure;
        ASSERT_EQ(builder.BuildClosureForNodes(ids, closure), HCCL_SUCCESS);
    }
} // namespace
} // namespace HcclSim
