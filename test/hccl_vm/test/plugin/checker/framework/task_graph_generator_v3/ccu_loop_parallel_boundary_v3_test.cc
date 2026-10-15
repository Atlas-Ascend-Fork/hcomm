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

#include <algorithm>
#include <deque>
#include <memory>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

// The loop transformation helpers are local to this translation unit. Include their implementation so tests
// execute production build/queue-assignment/rewire operations without adding a public test-only API.
#include "ccu_graph_generator_v3/ccu_task_transform_v3.cc"

namespace HcclSim {
namespace {
    using namespace TaskGraphGeneratorV3;
    using Graph = HcclSim::TaskGraphGeneratorV3::TaskGraphGeneratorV3;

    struct LoopNodes {
        TaskStart* start;
        TaskRecordCCU* body;
        TaskEnd* end;

        LoopInfoV3 Info() const { return {start, end}; }
    };

    class CcuLoopParallelBoundaryTest : public testing::Test {
    protected:
        template <typename T>
        T* AddNode(std::unique_ptr<T> node, QueueId queueId)
        {
            T* raw = node.get();
            TaskPosition position;
            position.queueId = queueId;
            NodeId nodeId = INVALID_NODE_ID;
            EXPECT_EQ(graph_.AppendGeneratedNode(std::move(node), position, nodeId), HCCL_SUCCESS);
            return raw;
        }

        TaskStart* AddStart(BoundaryType type, QueueId queueId)
        {
            return AddNode(std::make_unique<TaskStart>(type), queueId);
        }

        TaskEnd* AddEnd(BoundaryType type, QueueId queueId)
        {
            return AddNode(std::make_unique<TaskEnd>(type), queueId);
        }

        LoopNodes AddLoop(QueueId queueId)
        {
            LoopNodes loop{
                AddStart(BoundaryType::LOOP, queueId), AddNode(std::make_unique<TaskRecordCCU>(CcuNotify{}), queueId),
                AddEnd(BoundaryType::LOOP, queueId)};
            AddEdge(loop.start, loop.body);
            AddEdge(loop.body, loop.end);
            for (TaskNode* node : std::vector<TaskNode*>{loop.start, loop.body, loop.end}) {
                CcuTraceInfo trace;
                trace.valid = true;
                trace.position = node->GetPosition();
                trace.taskLoc = node->GetLocation();
                trace.queueId = queueId;
                trace.loopStartNodeId = loop.start->GetNodeId();
                node->SetCcuTrace(std::move(trace));
            }
            return loop;
        }

        void AddEdge(TaskNode* parent, TaskNode* child)
        {
            ASSERT_EQ(graph_.AddEdge(parent->GetNodeId(), child->GetNodeId()), HCCL_SUCCESS);
        }

        static void ExpectNodes(const std::vector<TaskNode*>& actual, const std::vector<TaskNode*>& expected)
        {
            EXPECT_EQ(actual.size(), expected.size());
            EXPECT_EQ(
                std::set<TaskNode*>(actual.begin(), actual.end()),
                std::set<TaskNode*>(expected.begin(), expected.end()));
        }

        static void ExpectEdge(TaskNode* parent, TaskNode* child, bool exists = true)
        {
            EXPECT_EQ(HasNodeEdgeV3(parent, child), exists);
            const auto& parents = child->GetParents();
            EXPECT_EQ(std::find(parents.begin(), parents.end(), parent) != parents.end(), exists);
        }

        static void ExpectLoopQueue(const LoopNodes& loop, QueueId queueId)
        {
            for (TaskNode* node : std::vector<TaskNode*>{loop.start, loop.body, loop.end}) {
                EXPECT_EQ(node->GetLocation().queueId, queueId);
                ASSERT_TRUE(node->HasCcuTrace());
                EXPECT_EQ(node->GetCcuTrace().queueId, queueId);
                EXPECT_EQ(node->GetCcuTrace().position.queueId, queueId);
                EXPECT_EQ(node->GetCcuTrace().taskLoc.queueId, queueId);
                EXPECT_EQ(node->GetCcuTrace().loopStartNodeId, loop.start->GetNodeId());
            }
            ExpectEdge(loop.start, loop.body);
            ExpectEdge(loop.body, loop.end);
        }

        using GraphSnapshot = std::vector<std::tuple<QueueId, std::vector<TaskNode*>, std::vector<TaskNode*>>>;
        GraphSnapshot Snapshot() const
        {
            GraphSnapshot result;
            for (const auto& node : graph_.GetNodes()) {
                result.emplace_back(node->GetLocation().queueId, node->GetParents(), node->GetChildren());
            }
            return result;
        }

        void TearDown() override
        {
            // Every scenario must leave reciprocal, duplicate-free edges and an acyclic graph.
            std::vector<size_t> indegrees(graph_.GetNodes().size());
            std::deque<TaskNode*> ready;
            for (const auto& node : graph_.GetNodes()) {
                const auto& parents = node->GetParents();
                const auto& children = node->GetChildren();
                EXPECT_EQ(std::set<TaskNode*>(parents.begin(), parents.end()).size(), parents.size());
                EXPECT_EQ(std::set<TaskNode*>(children.begin(), children.end()).size(), children.size());
                for (TaskNode* parent : parents) {
                    ExpectEdge(parent, node.get());
                }
                for (TaskNode* child : children) {
                    ExpectEdge(node.get(), child);
                }
                indegrees[node->GetNodeId()] = parents.size();
                if (parents.empty()) {
                    ready.push_back(node.get());
                }
            }
            size_t visited = 0;
            while (!ready.empty()) {
                TaskNode* node = ready.front();
                ready.pop_front();
                ++visited;
                for (TaskNode* child : node->GetChildren()) {
                    if (--indegrees[child->GetNodeId()] == 0) {
                        ready.push_back(child);
                    }
                }
            }
            EXPECT_EQ(visited, graph_.GetNodes().size());
        }

        Graph graph_;
    };

    // Queue head, queue tail, whole queue, and ordinary middle placement with 1/2/3 loop units.
    class CcuLoopParallelPlacementTest :
        public CcuLoopParallelBoundaryTest,
        public testing::WithParamInterface<std::tuple<bool, bool, int>> {};

    TEST_P(CcuLoopParallelPlacementTest, RealBoundariesFanOutAndJoinAllUnits)
    {
        const bool hasQueueBefore = std::get<0>(GetParam());
        const bool hasQueueAfter = std::get<1>(GetParam());
        const int unitCount = std::get<2>(GetParam());
        TaskStart* head = AddStart(BoundaryType::CCU_SUB_GRAPH, INVALID_QUEUE_ID);
        TaskNode* before = head;
        if (hasQueueBefore) {
            before = AddNode(std::make_unique<TaskRecordCCU>(CcuNotify{}), 0);
            AddEdge(head, before);
        }
        std::vector<LoopNodes> loops;
        std::vector<LoopInfoV3> infos;
        TaskNode* previous = before;
        for (int index = 0; index < unitCount; ++index) {
            loops.push_back(AddLoop(0));
            infos.push_back(loops.back().Info());
            AddEdge(previous, loops.back().start);
            previous = loops.back().end;
        }
        TaskNode* after = nullptr;
        if (hasQueueAfter) {
            after = AddNode(std::make_unique<TaskWaitCCU>(CcuNotify{}), 0);
            AddEdge(previous, after);
            previous = after;
        }
        TaskEnd* end = AddEnd(BoundaryType::CCU_SUB_GRAPH, INVALID_QUEUE_ID);
        AddEdge(previous, end);
        if (after == nullptr) {
            after = end;
        }

        LoopParallelGroupV3 group;
        ASSERT_EQ(BuildLoopParallelGroupV3(infos, group), HCCL_SUCCESS);
        ASSERT_EQ(group.units.size(), loops.size());
        ExpectNodes(group.beforeNodes, {before});
        ExpectNodes(group.afterNodes, {after});
        QueueId nextQueueId = 1;
        AssignLoopQueueIdsV3(group, nextQueueId);
        ASSERT_EQ(RewireLoopParallelGroupV3(graph_, group), HCCL_SUCCESS);
        EXPECT_EQ(nextQueueId, static_cast<QueueId>(unitCount + 1));

        std::vector<TaskNode*> expectedAfterParents{before};
        std::set<TaskNode*> completed{before};
        for (size_t index = 0; index < loops.size(); ++index) {
            ExpectNodes(loops[index].start->GetParents(), {before});
            ExpectNodes(loops[index].end->GetChildren(), {after});
            ExpectLoopQueue(loops[index], static_cast<QueueId>(index + 1));
            expectedAfterParents.push_back(loops[index].end);
            completed.insert(loops[index].end);
            for (size_t other = 0; other < loops.size(); ++other) {
                if (other != index) {
                    EXPECT_FALSE(graph_.HasPath(loops[index].end->GetNodeId(), loops[other].start->GetNodeId()));
                }
            }
        }
        ExpectNodes(after->GetParents(), expectedAfterParents);
        EXPECT_TRUE(AreParentsCompleted(after, completed));
        for (const auto& loop : loops) {
            auto incomplete = completed;
            incomplete.erase(loop.end);
            EXPECT_FALSE(AreParentsCompleted(after, incomplete));
        }
        EXPECT_EQ(before->GetLocation().queueId, hasQueueBefore ? 0U : INVALID_QUEUE_ID);
        EXPECT_EQ(after->GetLocation().queueId, hasQueueAfter ? 0U : INVALID_QUEUE_ID);
        EXPECT_EQ(head->GetLocation().queueId, INVALID_QUEUE_ID);
        EXPECT_EQ(end->GetLocation().queueId, INVALID_QUEUE_ID);
        if (hasQueueBefore) {
            ExpectEdge(head, before);
        }
        if (hasQueueAfter) {
            ExpectEdge(after, end);
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        QueuePositions, CcuLoopParallelPlacementTest,
        testing::Combine(testing::Bool(), testing::Bool(), testing::Values(1, 2, 3)));

    TEST_F(CcuLoopParallelBoundaryTest, ExternalDependenciesAreRetainedAlongsideQueueBoundaries)
    {
        TaskNode* queueBefore = AddNode(std::make_unique<TaskRecordCCU>(CcuNotify{}), 0);
        TaskNode* otherBefore = AddNode(std::make_unique<TaskRecordCCU>(CcuNotify{}), 1);
        const auto first = AddLoop(0);
        const auto second = AddLoop(0);
        TaskNode* queueAfter = AddNode(std::make_unique<TaskWaitCCU>(CcuNotify{}), 0);
        TaskNode* otherAfter = AddNode(std::make_unique<TaskWaitCCU>(CcuNotify{}), 1);
        AddEdge(queueBefore, first.start);
        AddEdge(otherBefore, first.start);
        AddEdge(first.end, second.start);
        AddEdge(second.end, queueAfter);
        AddEdge(second.end, otherAfter);

        LoopParallelGroupV3 group;
        ASSERT_EQ(BuildLoopParallelGroupV3({first.Info(), second.Info()}, group), HCCL_SUCCESS);
        ExpectNodes(group.beforeNodes, {queueBefore, otherBefore});
        ExpectNodes(group.afterNodes, {queueAfter, otherAfter});
        QueueId nextQueueId = 2;
        AssignLoopQueueIdsV3(group, nextQueueId);
        ASSERT_EQ(RewireLoopParallelGroupV3(graph_, group), HCCL_SUCCESS);
        for (const auto& loop : {first, second}) {
            ExpectNodes(loop.start->GetParents(), {queueBefore, otherBefore});
            ExpectNodes(loop.end->GetChildren(), {queueAfter, otherAfter});
            EXPECT_FALSE(AreParentsCompleted(loop.start, {queueBefore}));
            EXPECT_FALSE(AreParentsCompleted(loop.start, {otherBefore}));
            EXPECT_TRUE(AreParentsCompleted(loop.start, {queueBefore, otherBefore}));
        }
        for (TaskNode* after : {queueAfter, otherAfter}) {
            ExpectNodes(after->GetParents(), {queueBefore, otherBefore, first.end, second.end});
            EXPECT_FALSE(AreParentsCompleted(after, {queueBefore, otherBefore, second.end}));
            EXPECT_FALSE(AreParentsCompleted(after, {queueBefore, otherBefore, first.end}));
            EXPECT_TRUE(AreParentsCompleted(after, {queueBefore, otherBefore, first.end, second.end}));
        }
        ExpectLoopQueue(first, 2);
        ExpectLoopQueue(second, 3);
        EXPECT_EQ(otherBefore->GetLocation().queueId, 1U);
        EXPECT_EQ(otherAfter->GetLocation().queueId, 1U);
    }

    TEST_F(CcuLoopParallelBoundaryTest, LaterGroupReadsActualRewiredParentsAndExtraDependency)
    {
        TaskStart* head = AddStart(BoundaryType::CCU_SUB_GRAPH, INVALID_QUEUE_ID);
        TaskNode* extraBefore = AddNode(std::make_unique<TaskRecordCCU>(CcuNotify{}), 1);
        const auto first = AddLoop(0);
        const auto second = AddLoop(0);
        const auto third = AddLoop(0);
        const auto fourth = AddLoop(0);
        TaskEnd* end = AddEnd(BoundaryType::CCU_SUB_GRAPH, INVALID_QUEUE_ID);
        AddEdge(head, extraBefore);
        AddEdge(head, first.start);
        AddEdge(first.end, second.start);
        AddEdge(second.end, third.start);
        AddEdge(extraBefore, third.start);
        AddEdge(third.end, fourth.start);
        AddEdge(fourth.end, end);

        LoopParallelGroupV3 firstGroup;
        ASSERT_EQ(BuildLoopParallelGroupV3({first.Info(), second.Info()}, firstGroup), HCCL_SUCCESS);
        QueueId nextQueueId = 2;
        AssignLoopQueueIdsV3(firstGroup, nextQueueId);
        ASSERT_EQ(RewireLoopParallelGroupV3(graph_, firstGroup), HCCL_SUCCESS);

        LoopParallelGroupV3 laterGroup;
        ASSERT_EQ(BuildLoopParallelGroupV3({third.Info(), fourth.Info()}, laterGroup), HCCL_SUCCESS);
        ExpectNodes(laterGroup.beforeNodes, {head, first.end, second.end, extraBefore});
        ExpectNodes(laterGroup.afterNodes, {end});
        AssignLoopQueueIdsV3(laterGroup, nextQueueId);
        ASSERT_EQ(RewireLoopParallelGroupV3(graph_, laterGroup), HCCL_SUCCESS);
        for (const auto& loop : {third, fourth}) {
            ExpectNodes(loop.start->GetParents(), {head, first.end, second.end, extraBefore});
            EXPECT_FALSE(AreParentsCompleted(loop.start, {head, first.end, second.end}));
            EXPECT_FALSE(AreParentsCompleted(loop.start, {head, extraBefore, first.end}));
            EXPECT_FALSE(AreParentsCompleted(loop.start, {head, extraBefore, second.end}));
            EXPECT_TRUE(AreParentsCompleted(loop.start, {head, extraBefore, first.end, second.end}));
        }
        ExpectNodes(end->GetParents(), {head, extraBefore, first.end, second.end, third.end, fourth.end});
        EXPECT_FALSE(graph_.HasPath(first.end->GetNodeId(), second.start->GetNodeId()));
        EXPECT_FALSE(graph_.HasPath(third.end->GetNodeId(), fourth.start->GetNodeId()));
        ExpectLoopQueue(first, 2);
        ExpectLoopQueue(second, 3);
        ExpectLoopQueue(third, 4);
        ExpectLoopQueue(fourth, 5);
        EXPECT_EQ(nextQueueId, 6U);
        EXPECT_EQ(extraBefore->GetLocation().queueId, 1U);
    }

    TEST_F(CcuLoopParallelBoundaryTest, BoundariesCanBelongOnlyToOtherValidQueues)
    {
        TaskNode* before = AddNode(std::make_unique<TaskRecordCCU>(CcuNotify{}), 1);
        const auto first = AddLoop(0);
        const auto second = AddLoop(0);
        TaskNode* after = AddNode(std::make_unique<TaskWaitCCU>(CcuNotify{}), 2);
        AddEdge(before, first.start);
        AddEdge(first.end, second.start);
        AddEdge(second.end, after);

        LoopParallelGroupV3 group;
        ASSERT_EQ(BuildLoopParallelGroupV3({first.Info(), second.Info()}, group), HCCL_SUCCESS);
        ExpectNodes(group.beforeNodes, {before});
        ExpectNodes(group.afterNodes, {after});
        QueueId nextQueueId = 3;
        AssignLoopQueueIdsV3(group, nextQueueId);
        ASSERT_EQ(RewireLoopParallelGroupV3(graph_, group), HCCL_SUCCESS);
        ExpectNodes(first.start->GetParents(), {before});
        ExpectNodes(second.start->GetParents(), {before});
        ExpectNodes(after->GetParents(), {before, first.end, second.end});
        ExpectLoopQueue(first, 3);
        ExpectLoopQueue(second, 4);
        EXPECT_EQ(before->GetLocation().queueId, 1U);
        EXPECT_EQ(after->GetLocation().queueId, 2U);
    }

    TEST_F(CcuLoopParallelBoundaryTest, ApplyProcessesConsecutiveGroupsAndRebuildsQueueTails)
    {
        TaskStart* head = AddStart(BoundaryType::CCU_SUB_GRAPH, INVALID_QUEUE_ID);
        TaskNode* extraBefore = AddNode(std::make_unique<TaskRecordCCU>(CcuNotify{}), 1);
        const auto first = AddLoop(0);
        const auto second = AddLoop(0);
        const auto third = AddLoop(0);
        const auto fourth = AddLoop(0);
        TaskEnd* end = AddEnd(BoundaryType::CCU_SUB_GRAPH, INVALID_QUEUE_ID);
        AddEdge(head, extraBefore);
        AddEdge(head, first.start);
        AddEdge(first.end, second.start);
        AddEdge(second.end, third.start);
        AddEdge(extraBefore, third.start);
        AddEdge(third.end, fourth.start);
        AddEdge(fourth.end, end);

        TaskCcuGraph ccuGraph(CcuSubGraphDesc{});
        CcuGraphStateV3 state(graph_, ccuGraph);
        state.queueNum_ = 2;
        state.ccuHeadTaskNode = head;
        state.ccuSubGraphEndNode = end;
        state.loopGroupInfo_ = {{}, {first.Info(), second.Info()}, {third.Info(), fourth.Info()}};
        ASSERT_EQ(ApplyCcuQueueParallelizationV3(state), HCCL_SUCCESS);
        EXPECT_EQ(state.queueNum_, 6U);
        ASSERT_EQ(state.tailNodes.size(), 6U);
        EXPECT_EQ(state.tailNodes[0], nullptr);
        EXPECT_EQ(state.tailNodes[1], extraBefore);
        EXPECT_EQ(state.tailNodes[2], first.end);
        EXPECT_EQ(state.tailNodes[3], second.end);
        EXPECT_EQ(state.tailNodes[4], third.end);
        EXPECT_EQ(state.tailNodes[5], fourth.end);
        ExpectNodes(first.start->GetParents(), {head});
        ExpectNodes(second.start->GetParents(), {head});
        ExpectNodes(third.start->GetParents(), {head, extraBefore, first.end, second.end});
        ExpectNodes(fourth.start->GetParents(), {head, extraBefore, first.end, second.end});
        ExpectNodes(end->GetParents(), {head, extraBefore, first.end, second.end, third.end, fourth.end});
        ExpectLoopQueue(first, 2);
        ExpectLoopQueue(second, 3);
        ExpectLoopQueue(third, 4);
        ExpectLoopQueue(fourth, 5);
    }

    TEST_F(CcuLoopParallelBoundaryTest, DifferentOriginalQueuesShareEndAndKeepDistinctUnitQueues)
    {
        TaskStart* head = AddStart(BoundaryType::CCU_SUB_GRAPH, INVALID_QUEUE_ID);
        const auto first = AddLoop(0);
        const auto second = AddLoop(0);
        const auto third = AddLoop(1);
        const auto fourth = AddLoop(1);
        TaskEnd* end = AddEnd(BoundaryType::CCU_SUB_GRAPH, INVALID_QUEUE_ID);
        AddEdge(head, first.start);
        AddEdge(first.end, second.start);
        AddEdge(second.end, end);
        AddEdge(head, third.start);
        AddEdge(third.end, fourth.start);
        AddEdge(fourth.end, end);

        QueueId nextQueueId = 2;
        for (const auto& infos :
             {std::vector<LoopInfoV3>{first.Info(), second.Info()},
              std::vector<LoopInfoV3>{third.Info(), fourth.Info()}}) {
            LoopParallelGroupV3 group;
            ASSERT_EQ(BuildLoopParallelGroupV3(infos, group), HCCL_SUCCESS);
            AssignLoopQueueIdsV3(group, nextQueueId);
            ASSERT_EQ(RewireLoopParallelGroupV3(graph_, group), HCCL_SUCCESS);
        }
        ExpectNodes(end->GetParents(), {head, first.end, second.end, third.end, fourth.end});
        const std::set<TaskNode*> completed{head, first.end, second.end, third.end, fourth.end};
        for (const auto& loop : {first, second, third, fourth}) {
            auto incomplete = completed;
            incomplete.erase(loop.end);
            EXPECT_FALSE(AreParentsCompleted(end, incomplete));
        }
        EXPECT_TRUE(AreParentsCompleted(end, completed));
        ExpectLoopQueue(first, 2);
        ExpectLoopQueue(second, 3);
        ExpectLoopQueue(third, 4);
        ExpectLoopQueue(fourth, 5);
        EXPECT_EQ(end->GetLocation().queueId, INVALID_QUEUE_ID);
    }

    TEST_F(CcuLoopParallelBoundaryTest, DependenciesAttachedInsideUnitsStayAttachedToTheirOriginalNodes)
    {
        TaskStart* head = AddStart(BoundaryType::CCU_SUB_GRAPH, INVALID_QUEUE_ID);
        TaskNode* externalBefore = AddNode(std::make_unique<TaskRecordCCU>(CcuNotify{}), 1);
        const auto first = AddLoop(0);
        const auto second = AddLoop(0);
        TaskNode* externalAfter = AddNode(std::make_unique<TaskWaitCCU>(CcuNotify{}), 1);
        TaskEnd* end = AddEnd(BoundaryType::CCU_SUB_GRAPH, INVALID_QUEUE_ID);
        AddEdge(head, first.start);
        AddEdge(first.end, second.start);
        AddEdge(second.end, end);
        AddEdge(head, externalBefore);
        AddEdge(externalBefore, first.body);
        AddEdge(first.body, externalAfter);
        AddEdge(externalAfter, end);

        LoopParallelGroupV3 group;
        ASSERT_EQ(BuildLoopParallelGroupV3({first.Info(), second.Info()}, group), HCCL_SUCCESS);
        ExpectNodes(group.beforeNodes, {head});
        ExpectNodes(group.afterNodes, {end});
        ASSERT_EQ(group.units.size(), 2U);
        EXPECT_EQ(group.units[0].nodes, (std::set<TaskNode*>{first.start, first.body, first.end}));
        EXPECT_EQ(group.units[1].nodes, (std::set<TaskNode*>{second.start, second.body, second.end}));
        QueueId nextQueueId = 2;
        AssignLoopQueueIdsV3(group, nextQueueId);
        ASSERT_EQ(RewireLoopParallelGroupV3(graph_, group), HCCL_SUCCESS);
        ExpectNodes(first.body->GetParents(), {first.start, externalBefore});
        ExpectNodes(first.body->GetChildren(), {first.end, externalAfter});
        ExpectNodes(second.start->GetParents(), {head});
        ExpectNodes(end->GetParents(), {head, first.end, second.end, externalAfter});
        ExpectLoopQueue(first, 2);
        ExpectLoopQueue(second, 3);
        EXPECT_EQ(externalBefore->GetLocation().queueId, 1U);
        EXPECT_EQ(externalAfter->GetLocation().queueId, 1U);
    }

    class CcuLoopParallelMissingBoundaryTest :
        public CcuLoopParallelBoundaryTest,
        public testing::WithParamInterface<std::tuple<bool, bool>> {};

    TEST_P(CcuLoopParallelMissingBoundaryTest, RejectsMissingRealBoundaryWithoutMutatingGraph)
    {
        const auto loop = AddLoop(0);
        if (std::get<0>(GetParam())) {
            AddEdge(AddStart(BoundaryType::CCU_SUB_GRAPH, INVALID_QUEUE_ID), loop.start);
        }
        if (std::get<1>(GetParam())) {
            AddEdge(loop.end, AddEnd(BoundaryType::CCU_SUB_GRAPH, INVALID_QUEUE_ID));
        }
        const auto original = Snapshot();
        LoopParallelGroupV3 group;
        EXPECT_EQ(BuildLoopParallelGroupV3({loop.Info()}, group), HCCL_E_INTERNAL);
        EXPECT_EQ(Snapshot(), original);
        TaskCcuGraph ccuGraph(CcuSubGraphDesc{});
        CcuGraphStateV3 state(graph_, ccuGraph);
        state.queueNum_ = 1;
        state.loopGroupInfo_ = {{loop.Info()}};
        EXPECT_EQ(ApplyCcuQueueParallelizationV3(state), HCCL_E_INTERNAL);
        EXPECT_EQ(state.queueNum_, 1U);
        EXPECT_EQ(Snapshot(), original);
        ExpectLoopQueue(loop, 0);
    }

    INSTANTIATE_TEST_SUITE_P(
        MalformedGraph, CcuLoopParallelMissingBoundaryTest,
        testing::Values(std::make_tuple(false, true), std::make_tuple(true, false), std::make_tuple(false, false)));

    TEST_F(CcuLoopParallelBoundaryTest, EmptyGroupClearsOutputWithoutMutatingGraph)
    {
        LoopParallelGroupV3 group;
        group.originalQueueId = 0;
        group.units.emplace_back();
        group.beforeNodes.push_back(AddStart(BoundaryType::CCU_SUB_GRAPH, INVALID_QUEUE_ID));
        const auto original = Snapshot();
        ASSERT_EQ(BuildLoopParallelGroupV3({}, group), HCCL_SUCCESS);
        EXPECT_TRUE(group.units.empty());
        EXPECT_TRUE(group.beforeNodes.empty());
        EXPECT_TRUE(group.afterNodes.empty());
        EXPECT_EQ(Snapshot(), original);
    }

    TEST_F(CcuLoopParallelBoundaryTest, InvalidQueueIsSkippedWithoutMutatingGraph)
    {
        const auto loop = AddLoop(INVALID_QUEUE_ID);
        const auto original = Snapshot();
        LoopParallelGroupV3 group;
        ASSERT_EQ(BuildLoopParallelGroupV3({loop.Info()}, group), HCCL_SUCCESS);
        EXPECT_TRUE(group.units.empty());
        EXPECT_EQ(Snapshot(), original);
    }
} // namespace
} // namespace HcclSim
