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

#include <cstring>
#include <string>
#include <unistd.h>

#include "big_graph_check/big_graph_checker.h"
#include "operation_data/operation_data_ops.h"

namespace sync_stream_test {
bool ResetDatabase(const std::string& runner, const std::string& opdata);
void CleanDatabase(const std::string& runner, const std::string& opdata);
uint64_t SeedCommunicator(uint32_t deviceId);
} // namespace sync_stream_test

namespace {
using namespace sim::operation;
using namespace HcclSim::TaskGraphGeneratorV3;

class SyncStreamBigGraphTest : public testing::Test {
protected:
    void SetUp() override
    {
        const std::string prefix = "/tmp/checker_sync_stream_" + std::to_string(getpid());
        runner_ = prefix + "_runner.db";
        opdata_ = prefix + "_opdata.db";
        ASSERT_TRUE(sync_stream_test::ResetDatabase(runner_, opdata_));
        RestoreOpContext({});
        commId_ = sync_stream_test::SeedCommunicator(1);
        ASSERT_NE(commId_, 0U);
    }

    void TearDown() override { sync_stream_test::CleanDatabase(runner_, opdata_); }

    void AddOp(uint64_t stream, uint64_t input, uint64_t output, OpDetailTab& detail)
    {
        OpDetails params{};
        params.opType = HCCL_CMD_BROADCAST;
        params.dataType = HCCL_DATA_TYPE_FP32;
        params.opV1.count = 4;
        detail.pid = static_cast<uint32_t>(getpid());
        detail.deviceId = 1;
        detail.rankSize = 1;
        detail.commId = commId_;
        detail.streamId = stream;
        const auto* bytes = reinterpret_cast<const uint8_t*>(&params);
        detail.opDetail.assign(bytes, bytes + sizeof(params));
        OpMemInfoTab mem{};
        mem.inputAddr = input;
        mem.outputAddr = output;
        mem.inputSize = mem.outputSize = 64;
        ASSERT_EQ(InsertOpDetailAndMem(detail, mem), 0);
    }

    void AddTask(const HcclTaskMetaData& meta, OpTaskTab& task)
    {
        task.deviceId = meta.deviceId;
        task.streamId = meta.streamId;
        task.taskType = static_cast<uint32_t>(meta.taskType);
        const auto* bytes = reinterpret_cast<const uint8_t*>(&meta);
        task.optaskMeta.assign(bytes, bytes + sizeof(meta));
        ASSERT_EQ(InsertOpTask(task), 0);
    }

    void AddCopy(uint64_t stream, uint64_t input, uint64_t output)
    {
        HcclTaskMetaData meta{};
        meta.taskType = HccLTaskMetaType::MEM_CPY;
        meta.commId = commId_;
        meta.deviceId = 1;
        meta.rankId = 0;
        meta.streamId = stream;
        meta.taskData.transMem.srcDeviceId = meta.taskData.transMem.dstDeviceId = 1;
        meta.taskData.transMem.srcOffset = input;
        meta.taskData.transMem.dstOffset = output;
        meta.taskData.transMem.len = 16;
        OpTaskTab task{};
        AddTask(meta, task);
    }

    void AddSync(uint64_t stream, uint32_t expectedOwner)
    {
        HcclTaskMetaData meta{};
        meta.taskType = HccLTaskMetaType::SYNC_STREAM;
        meta.deviceId = 1;
        meta.streamId = stream;
        OpTaskTab task{};
        AddTask(meta, task);
        ASSERT_EQ(task.opDetailId, expectedOwner);
    }

    void BuildGraph()
    {
        ASSERT_EQ(checker_.LoadOpData(loader_), HCCL_SUCCESS);
        ASSERT_EQ(checker_.TranslateTask(), HCCL_SUCCESS);
        ASSERT_EQ(checker_.GenerateBigGraph(), HCCL_SUCCESS);
        ASSERT_EQ(checker_.SyncCheck(), HCCL_SUCCESS);
    }

    NodeId FindNode(TaskType type, StreamId stream, uint32_t opIter) const
    {
        for (const auto& node : checker_.GetGraph()->GetNodes()) {
            if (node != nullptr && node->GetType() == type && node->GetPosition().streamId == stream
                && node->GetPosition().opIter == opIter) {
                return node->GetNodeId();
            }
        }
        return INVALID_NODE_ID;
    }

    void CheckNestedWarmup(StreamId warmupStream);

    std::string runner_;
    std::string opdata_;
    uint64_t commId_{0};
    loader::Loader loader_;
    HcclSim::BigGraphCheckV3::BigGraphCheckerV3 checker_;
};

void SyncStreamBigGraphTest::CheckNestedWarmup(StreamId warmupStream)
{
    OpDetailTab outer{};
    ASSERT_NO_FATAL_FAILURE(AddOp(0, 0x100000, 0x200000, outer));
    const auto outerContext = SaveOpContext();
    OpDetailTab inner{};
    ASSERT_NO_FATAL_FAILURE(AddOp(warmupStream, 0x300000, 0x400000, inner));
    ASSERT_NO_FATAL_FAILURE(AddCopy(warmupStream, 0x300000, 0x400000));
    ASSERT_EQ(AssignOpEndSeq(inner.id), 0);
    RestoreOpContext(outerContext);
    ASSERT_NO_FATAL_FAILURE(AddSync(warmupStream, inner.id));
    ASSERT_NO_FATAL_FAILURE(AddCopy(0, 0x100000, 0x200000));
    ASSERT_EQ(AssignOpEndSeq(outer.id), 0);
    RestoreOpContext({});
    ASSERT_NO_FATAL_FAILURE(AddSync(0, outer.id));

    ASSERT_NO_FATAL_FAILURE(BuildGraph());
    const auto* graph = checker_.GetGraph();
    const auto warmupCopy = FindNode(TaskType::TRANS_MEM, warmupStream, inner.opIter);
    const auto warmupSync = FindNode(TaskType::SYNC_STREAM, warmupStream, inner.opIter);
    const auto outerCopy = FindNode(TaskType::TRANS_MEM, 0, outer.opIter);
    const auto outerSync = FindNode(TaskType::SYNC_STREAM, 0, outer.opIter);
    ASSERT_NE(warmupCopy, INVALID_NODE_ID);
    ASSERT_NE(warmupSync, INVALID_NODE_ID);
    ASSERT_NE(outerCopy, INVALID_NODE_ID);
    ASSERT_NE(outerSync, INVALID_NODE_ID);
    EXPECT_NE(warmupSync, outerSync);
    EXPECT_TRUE(graph->HasPath(warmupCopy, warmupSync));
    EXPECT_TRUE(graph->HasPath(outerCopy, outerSync));
    EXPECT_FALSE(graph->HasPath(outerSync, outerCopy));
}

TEST_F(SyncStreamBigGraphTest, WarmupAndOuterSyncRemainDistinctAndFollowTheirTasks)
{
    ASSERT_NO_FATAL_FAILURE(CheckNestedWarmup(2));
}

TEST_F(SyncStreamBigGraphTest, SameStreamNestedWarmupSyncFollowsOuterCompletion)
{
    ASSERT_NO_FATAL_FAILURE(CheckNestedWarmup(0));
}

TEST_F(SyncStreamBigGraphTest, NewTasklessOperatorOnReusedMainStreamKeepsItsSync)
{
    OpDetailTab first{};
    ASSERT_NO_FATAL_FAILURE(AddOp(0, 0x100000, 0x200000, first));
    ASSERT_NO_FATAL_FAILURE(AddCopy(0, 0x100000, 0x200000));
    ASSERT_EQ(AssignOpEndSeq(first.id), 0);
    RestoreOpContext({});
    ASSERT_NO_FATAL_FAILURE(AddSync(0, first.id));

    OpDetailTab next{};
    ASSERT_NO_FATAL_FAILURE(AddOp(0, 0x300000, 0x400000, next));
    ASSERT_EQ(AssignOpEndSeq(next.id), 0);
    RestoreOpContext({});
    ASSERT_NO_FATAL_FAILURE(AddSync(0, next.id));

    ASSERT_NO_FATAL_FAILURE(BuildGraph());
    const auto* graph = checker_.GetGraph();
    const auto& queue = graph->GetTaskQueues().at(1).at(0);
    ASSERT_EQ(queue.size(), 3U);
    EXPECT_EQ(graph->GetNode(queue[0])->GetType(), TaskType::TRANS_MEM);
    EXPECT_EQ(graph->GetNode(queue[1])->GetType(), TaskType::SYNC_STREAM);
    EXPECT_EQ(graph->GetNode(queue[2])->GetType(), TaskType::SYNC_STREAM);
    EXPECT_TRUE(graph->HasPath(queue[0], queue[2]));
    EXPECT_FALSE(graph->HasPath(queue[2], queue[0]));
}
} // namespace
