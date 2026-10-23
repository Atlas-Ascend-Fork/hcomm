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
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime_state/db_sim_communicator.h"
#include "storage_manager.h"
#include "task_graph_semantic_check_v3.h"
#include "reduce_order_check_v3.h"

namespace {
std::vector<sim::runtime::CommunicatorMemberInfo> g_semanticTestMembers;
}

// 不链接 runtime_state、不打开数据库，仅解析通信域身份信息。
namespace sim::runtime {
bool GetCommunicatorMembers(uint64_t commId, std::vector<CommunicatorMemberInfo>& members)
{
    members.clear();
    const auto member = std::find_if(
        g_semanticTestMembers.begin(), g_semanticTestMembers.end(), [commId](const CommunicatorMemberInfo& value) {
            return value.memberId == commId;
        });
    if (member == g_semanticTestMembers.end()) {
        return false;
    }
    members = g_semanticTestMembers;
    return true;
}
} // namespace sim::runtime

namespace HcclSim {
namespace {
    namespace v3 = TaskGraphGeneratorV3;

    class SemanticEnvGuard {
    public:
        explicit SemanticEnvGuard(const char* name) : name_(name)
        {
            const char* value = std::getenv(name);
            hadValue_ = value != nullptr;
            if (hadValue_) {
                value_ = value;
            }
            (void)::unsetenv(name);
        }
        ~SemanticEnvGuard()
        {
            if (hadValue_) {
                (void)::setenv(name_.c_str(), value_.c_str(), 1);
            } else {
                (void)::unsetenv(name_.c_str());
            }
        }

    private:
        std::string name_;
        std::string value_;
        bool hadValue_{false};
    };

    enum class TreeShape { BALANCED, LEFT_DEEP, REVERSE_ROOT, REVERSE_FIRST_PAIR };

    class TaskGraphSemanticCheckV3Test : public testing::Test {
    protected:
        static constexpr uint64_t kCommHash = 73;
        static constexpr uint64_t kFirstCommId = 1000;
        static constexpr v3::OperatorId kOperatorId = 7;

        void SetUp() override
        {
            storage_.Reset();
            g_semanticTestMembers.clear();
            env_.emplace_back(std::make_unique<SemanticEnvGuard>("HCCL_DETERMINISTIC"));
        }

        void TearDown() override
        {
            nodes_.clear();
            storage_.Reset();
            g_semanticTestMembers.clear();
            env_.clear();
        }

        uint32_t Device(uint32_t rank) const
        {
            // 物理设备号刻意与通信域 rank 号及其排序不同。
            static constexpr uint32_t devices[] = {17, 3, 29, 11};
            return devices[rank];
        }

        uint64_t InputBase(uint32_t rank) const { return 0x100000ULL + Device(rank) * 0x10000ULL; }
        uint64_t OutputBase(uint32_t rank) const
        {
            return inplace_ ? InputBase(rank) : 0x500000ULL + Device(rank) * 0x10000ULL;
        }
        uint64_t ScratchBase(uint32_t rank) const { return 0x900000ULL + Device(rank) * 0x10000ULL; }

        void Init(
            HcclCMDType command = HCCL_CMD_ALLREDUCE, uint32_t ranks = 4, uint64_t count = 8, bool enabled = true,
            bool inplace = false, HcclDataType dataType = HCCL_DATA_TYPE_FP32, HcclReduceOp op = HCCL_REDUCE_SUM)
        {
            nodes_.clear();
            storage_.Reset();
            g_semanticTestMembers.clear();
            command_ = command;
            rankSize_ = ranks;
            count_ = count;
            const uint64_t width = dataType == HCCL_DATA_TYPE_FP64                                     ? 8U :
                                   dataType == HCCL_DATA_TYPE_FP16 || dataType == HCCL_DATA_TYPE_BFP16 ? 2U :
                                                                                                         4U;
            bytes_ = count * width;
            inplace_ = inplace;
            dataType_ = dataType;
            op_ = op;
            ASSERT_EQ(::setenv("HCCL_DETERMINISTIC", enabled ? "STRICT" : "FALSE", 1), 0);
            storage_.BeginOpGroup("semantic-v3-ut", kCommHash, 0);
            for (uint32_t rank = 0; rank < ranks; ++rank) {
                g_semanticTestMembers.push_back({kFirstCommId + rank, Device(rank), rank});
            }
            std::vector<sim::operation::CcuChannelTab> channels;
            std::vector<sim::operation::HalfRTTTab> halfRtt;
            for (uint32_t rank = 0; rank < ranks; ++rank) {
                sim::operation::OpDetailTab row{};
                row.deviceId = Device(rank);
                row.rankId = rank;
                row.commId = kFirstCommId + rank;
                row.rankSize = ranks;
                ::OpDetails detail{};
                detail.opType = static_cast<uint16_t>(command);
                detail.dataType = static_cast<uint16_t>(dataType);
                detail.reduceType = static_cast<uint16_t>(op_);
                detail.opV1.count = count;
                ASSERT_EQ(storage_.Trans2CheckerParam(row, detail), HCCL_SUCCESS);
                sim::operation::OpMemInfoTab memory{};
                memory.inputAddr = InputBase(rank);
                memory.inputSize = bytes_ * (command == HCCL_CMD_REDUCE_SCATTER ? ranks : 1U);
                memory.outputAddr = OutputBase(rank);
                memory.outputSize = bytes_;
                memory.cclAddr = ScratchBase(rank);
                memory.cclSize = 0x2000;
                ASSERT_EQ(
                    storage_.LoadHcclVmSynthesisData(
                        Device(rank), row.commId, "semantic-v3-ut", kCommHash, 0, rank, memory, channels, halfRtt),
                    HCCL_SUCCESS);
            }
            ASSERT_EQ(storage_.FinalizeOpGroup(), HCCL_SUCCESS);
            storage_.SaveCheckerParam(kOperatorId);
            start_ = Add(std::make_unique<v3::TaskStart>(v3::BoundaryType::MAIN_GRAPH), 0);
            start_->SetNodeId(v3::MAIN_START_NODE_ID);
            tail_ = start_;
        }

        template <typename T>
        T* Add(std::unique_ptr<T> node, uint32_t owner)
        {
            T* result = node.get();
            result->SetNodeId(static_cast<v3::NodeId>(nodes_.size()));
            v3::TaskPosition position;
            position.operatorId = kOperatorId;
            position.commId = kFirstCommId + owner;
            position.commHash = kCommHash;
            position.deviceId = Device(owner);
            position.rankId = owner;
            position.streamId = 0;
            position.mainStreamId = 0;
            position.queueId = 0;
            result->SetPosition(position);
            nodes_.push_back(std::move(node));
            return result;
        }

        void Edge(v3::TaskNode* parent, v3::TaskNode* child)
        {
            ASSERT_TRUE(parent->AddChild(child));
            ASSERT_TRUE(child->AddParent(parent));
        }

        template <typename T>
        T* Append(std::unique_ptr<T> node, uint32_t owner)
        {
            T* result = Add(std::move(node), owner);
            Edge(tail_, result);
            tail_ = result;
            return result;
        }

        v3::MemSlice Slice(uint32_t rank, v3::MemType type, uint64_t offset, uint64_t len) const
        {
            v3::MemSlice result;
            result.deviceId = Device(rank);
            result.rankId = rank;
            result.memType = type;
            result.offset = offset;
            result.len = len;
            const uint64_t base = type == v3::MemType::INPUT  ? InputBase(rank) :
                                  type == v3::MemType::OUTPUT ? OutputBase(rank) :
                                                                ScratchBase(rank);
            result.rawAddr = base + offset;
            return result;
        }

        v3::MemSlice Input(uint32_t rank, uint32_t owner, uint64_t offset, uint64_t len) const
        {
            return Slice(
                rank, v3::MemType::INPUT, (command_ == HCCL_CMD_REDUCE_SCATTER ? owner * bytes_ : 0) + offset, len);
        }

        void Copy(const v3::MemSlice& src, const v3::MemSlice& dst, uint32_t owner)
        {
            Append(std::make_unique<v3::TaskTransMem>(src, dst), owner);
        }

        void Reduce(
            const v3::MemSlice& src, const v3::MemSlice& dst, uint32_t owner,
            HcclDataType dataType = HCCL_DATA_TYPE_RESERVED)
        {
            const HcclDataType actualType = dataType == HCCL_DATA_TYPE_RESERVED ? dataType_ : dataType;
            Append(
                std::make_unique<v3::TaskReduce>(src, dst, static_cast<uint8_t>(actualType), static_cast<uint8_t>(op_)),
                owner);
        }

        v3::MemSlice BuildTree(uint32_t owner, uint64_t offset, uint64_t len, TreeShape shape)
        {
            auto a = Slice(owner, v3::MemType::CCL, offset, len);
            if (rankSize_ < 4 || shape == TreeShape::LEFT_DEEP) {
                Copy(Input(0, owner, offset, len), a, owner);
                for (uint32_t rank = 1; rank < rankSize_; ++rank) {
                    Reduce(Input(rank, owner, offset, len), a, owner, dataType_);
                }
                return a;
            }
            auto b = Slice(owner, v3::MemType::CCL, 0x200 + offset, len);
            const bool reversePair = shape == TreeShape::REVERSE_FIRST_PAIR;
            Copy(Input(reversePair ? 2 : 0, owner, offset, len), a, owner);
            Reduce(Input(reversePair ? 0 : 2, owner, offset, len), a, owner, dataType_);
            Copy(Input(1, owner, offset, len), b, owner);
            Reduce(Input(3, owner, offset, len), b, owner, dataType_);
            if (shape == TreeShape::REVERSE_ROOT) {
                Reduce(a, b, owner, dataType_);
                return b;
            }
            Reduce(b, a, owner, dataType_);
            return a;
        }

        void WriteOwner(uint32_t owner, TreeShape shape = TreeShape::BALANCED)
        {
            Copy(BuildTree(owner, 0, bytes_, shape), Slice(owner, v3::MemType::OUTPUT, 0, bytes_), owner);
        }

        void WriteAll(TreeShape shape = TreeShape::BALANCED)
        {
            for (uint32_t owner = 0; owner < rankSize_; ++owner) {
                WriteOwner(owner, shape);
            }
        }

        void BatchCopy(const std::vector<v3::MemSlice>& srcs, const std::vector<v3::MemSlice>& dsts, uint32_t owner)
        {
            auto node = std::make_unique<v3::TaskBatchTransMem>();
            node->SetSrcMemSlices(srcs);
            node->SetDstMemSlices(dsts);
            Append(std::move(node), owner);
        }

        void BatchReduce(const std::vector<v3::MemSlice>& srcs, const std::vector<v3::MemSlice>& dsts, uint32_t owner)
        {
            auto node
                = std::make_unique<v3::TaskBatchReduce>(static_cast<uint8_t>(dataType_), static_cast<uint8_t>(op_));
            for (const auto& src : srcs) {
                node->AddSrcMemSlice({src});
            }
            node->SetDstMemSlices(dsts);
            Append(std::move(node), owner);
        }

        void WriteBatchedOwner(uint32_t owner)
        {
            const uint64_t half = bytes_ / 2;
            std::vector<v3::MemSlice> a;
            std::vector<v3::MemSlice> b;
            std::vector<v3::MemSlice> outputs;
            for (uint64_t offset : {uint64_t{0}, half}) {
                a.push_back(Slice(owner, v3::MemType::CCL, offset, half));
                b.push_back(Slice(owner, v3::MemType::CCL, 0x200 + offset, half));
                outputs.push_back(Slice(owner, v3::MemType::OUTPUT, offset, half));
            }
            const auto inputs = [&](uint32_t rank) {
                return std::vector<v3::MemSlice>{Input(rank, owner, 0, half), Input(rank, owner, half, half)};
            };
            BatchCopy(inputs(0), a, owner);
            BatchReduce(inputs(2), a, owner);
            BatchCopy(inputs(1), b, owner);
            BatchReduce(inputs(3), b, owner);
            BatchReduce(b, a, owner);
            BatchCopy(a, outputs, owner);
        }

        void ExpectPass()
        {
            v3::SemanticCheckStats stats;
            EXPECT_EQ(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
            EXPECT_EQ(stats.reduceOrder.status, v3::ReduceOrderStatus::PASS) << stats.reduceOrder.detail;
            EXPECT_GT(stats.reduceOrder.expressionNodes, 0U);
            if (count_ != 0) {
                EXPECT_GE(stats.reduceOrder.outputSpans, rankSize_);
            }
        }

        void ExpectFailure(v3::ReduceOrderStatus expected)
        {
            v3::SemanticCheckStats stats;
            EXPECT_NE(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
            EXPECT_EQ(stats.reduceOrder.status, expected) << stats.reduceOrder.detail;
        }

        StorageManager& storage_ = StorageManager::GetInstance();
        std::vector<std::unique_ptr<SemanticEnvGuard>> env_;
        std::vector<std::unique_ptr<v3::TaskNode>> nodes_;
        v3::TaskStart* start_{nullptr};
        v3::TaskNode* tail_{nullptr};
        HcclCMDType command_{HCCL_CMD_ALLREDUCE};
        HcclDataType dataType_{HCCL_DATA_TYPE_FP32};
        HcclReduceOp op_{HCCL_REDUCE_SUM};
        uint32_t rankSize_{4};
        uint64_t count_{8};
        uint64_t bytes_{32};
        bool inplace_{false};
    };

    TEST_F(TaskGraphSemanticCheckV3Test, FloatingPointSumAndProdMatrixPreservesTreeForArAndRs)
    {
        for (HcclCMDType command : {HCCL_CMD_ALLREDUCE, HCCL_CMD_REDUCE_SCATTER}) {
            for (HcclDataType type :
                 {HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_BFP16, HCCL_DATA_TYPE_FP64}) {
                for (HcclReduceOp op : {HCCL_REDUCE_SUM, HCCL_REDUCE_PROD}) {
                    SCOPED_TRACE(testing::Message() << "command=" << command << ", dtype=" << type << ", op=" << op);
                    Init(command, 4, 8, true, false, type, op);
                    ASSERT_FALSE(HasFatalFailure());
                    WriteAll();
                    ExpectPass();
                    Init(command, 4, 8, true, false, type, op);
                    ASSERT_FALSE(HasFatalFailure());
                    for (uint32_t owner = 0; owner < rankSize_; ++owner) {
                        WriteBatchedOwner(owner);
                    }
                    ExpectPass();
                }
            }
        }
    }

    TEST_F(TaskGraphSemanticCheckV3Test, AllReduceTreeUsesGlobalDeviceIds)
    {
        Init();
        WriteAll();
        v3::SemanticCheckStats stats;
        ASSERT_EQ(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.reduceOrder.outputSpans, 4U);
        // PASS 的 detail 里给出参考树的求值顺序；叶子是全局设备号，与 rank 排序不同。
        EXPECT_NE(stats.reduceOrder.detail.find("[17,29,3,11]"), std::string::npos);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, ReduceScatterStrictTreeUsesOwnerInputOffset)
    {
        Init(HCCL_CMD_REDUCE_SCATTER);
        WriteAll();
        ExpectPass();
    }

    TEST_F(TaskGraphSemanticCheckV3Test, ReduceScatterRejectsCorrectTreeAtWrongOwnerInputCoordinate)
    {
        Init(HCCL_CMD_REDUCE_SCATTER);
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            auto first = Slice(owner, v3::MemType::CCL, 0, bytes_);
            auto second = Slice(owner, v3::MemType::CCL, 0x200, bytes_);
            // 每个 owner 都错误地读取 0 号分片；仅看有序叶子树仍然合法。
            Copy(Slice(0, v3::MemType::INPUT, 0, bytes_), first, owner);
            Reduce(Slice(2, v3::MemType::INPUT, 0, bytes_), first, owner);
            Copy(Slice(1, v3::MemType::INPUT, 0, bytes_), second, owner);
            Reduce(Slice(3, v3::MemType::INPUT, 0, bytes_), second, owner);
            Reduce(second, first, owner);
            Copy(first, Slice(owner, v3::MemType::OUTPUT, 0, bytes_), owner);
        }
        v3::SemanticCheckStats stats;
        EXPECT_NE(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
        EXPECT_NE(stats.reduceOrder.status, v3::ReduceOrderStatus::PASS);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, ReduceTaskAddressOverflowCannotProduceStrictPass)
    {
        Init();
        auto src = Input(1, 0, 0, 8);
        src.rawAddr = UINT64_MAX - 3;
        auto dst = Slice(0, v3::MemType::CCL, 0, 8);
        Copy(Input(0, 0, 0, 8), dst, 0);
        Reduce(src, dst, 0);
        ExpectFailure(v3::ReduceOrderStatus::INCOMPLETE);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, OperatorElementCountOverflowCannotProduceStrictPass)
    {
        Init();
        sim::operation::OpDetailTab row{};
        row.deviceId = Device(0);
        row.rankSize = rankSize_;
        row.commId = kFirstCommId;
        ::OpDetails detail{};
        detail.opType = static_cast<uint16_t>(HCCL_CMD_ALLREDUCE);
        detail.dataType = static_cast<uint16_t>(HCCL_DATA_TYPE_FP32);
        detail.reduceType = static_cast<uint16_t>(HCCL_REDUCE_SUM);
        detail.opV1.count = UINT64_MAX / 4U + 1U;
        ASSERT_EQ(storage_.Trans2CheckerParam(row, detail), HCCL_SUCCESS);
        storage_.SaveCheckerParam(kOperatorId);
        ExpectFailure(v3::ReduceOrderStatus::INCOMPLETE);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, EnabledCaseRejectsUnmappedSourceDevice)
    {
        Init();
        auto src = Input(1, 0, 0, bytes_);
        src.deviceId = 100;
        auto dst = Slice(0, v3::MemType::CCL, 0, bytes_);
        Copy(Input(0, 0, 0, bytes_), dst, 0);
        Reduce(src, dst, 0);
        v3::SemanticCheckStats stats;
        EXPECT_NE(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
        EXPECT_NE(stats.reduceOrder.status, v3::ReduceOrderStatus::PASS);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, FragmentedTasksProduceWholeOutputSemanticBlocks)
    {
        for (HcclCMDType command : {HCCL_CMD_ALLREDUCE, HCCL_CMD_REDUCE_SCATTER}) {
            Init(command, 4, 32);
            for (uint32_t owner = 0; owner < rankSize_; ++owner) {
                // 逐元素构造，人为引入大量算法/任务切片边界。
                for (uint64_t offset = 0; offset < bytes_; offset += 4) {
                    const auto result = BuildTree(owner, offset, 4, TreeShape::BALANCED);
                    Copy(result, Slice(owner, v3::MemType::OUTPUT, offset, 4), owner);
                }
            }
            v3::SemanticCheckStats stats;
            ASSERT_EQ(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
            EXPECT_EQ(stats.reduceOrder.status, v3::ReduceOrderStatus::PASS);
            EXPECT_EQ(stats.reduceOrder.outputSpans, rankSize_);
        }
    }

    TEST_F(TaskGraphSemanticCheckV3Test, UniformAlternativeTreePassesConsistencyCheck)
    {
        Init();
        WriteAll(TreeShape::LEFT_DEEP);
        ExpectPass();
    }

    TEST_F(TaskGraphSemanticCheckV3Test, EqualLeafSetsDoNotHideReassociationAcrossRanks)
    {
        Init();
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            WriteOwner(owner, owner == 3 ? TreeShape::LEFT_DEEP : TreeShape::BALANCED);
        }
        ExpectFailure(v3::ReduceOrderStatus::MISMATCH);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, DestinationAndSourceRolesAreNotCommuted)
    {
        Init();
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            WriteOwner(owner, owner == 1 ? TreeShape::REVERSE_ROOT : TreeShape::BALANCED);
        }
        ExpectFailure(v3::ReduceOrderStatus::MISMATCH);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, InternalOperandDirectionIsNotCommuted)
    {
        Init();
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            WriteOwner(owner, owner == 2 ? TreeShape::REVERSE_FIRST_PAIR : TreeShape::BALANCED);
        }
        ExpectFailure(v3::ReduceOrderStatus::MISMATCH);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, DifferentTreesInTwoOutputHalvesAreRejected)
    {
        Init(HCCL_CMD_ALLREDUCE, 4, 16);
        const uint64_t half = bytes_ / 2;
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            auto first = BuildTree(owner, 0, half, TreeShape::BALANCED);
            Copy(first, Slice(owner, v3::MemType::OUTPUT, 0, half), owner);
            auto second = BuildTree(owner, half, half, TreeShape::LEFT_DEEP);
            Copy(second, Slice(owner, v3::MemType::OUTPUT, half, half), owner);
        }
        ExpectFailure(v3::ReduceOrderStatus::MISMATCH);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, ReduceScatterDifferentOwnerTreesAreRejected)
    {
        Init(HCCL_CMD_REDUCE_SCATTER);
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            WriteOwner(owner, owner == 3 ? TreeShape::LEFT_DEEP : TreeShape::BALANCED);
        }
        ExpectFailure(v3::ReduceOrderStatus::MISMATCH);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, CopySplitAndAdjacentMergePreserveTree)
    {
        Init();
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            const auto result = BuildTree(owner, 0, bytes_, TreeShape::BALANCED);
            auto temporary = Slice(owner, v3::MemType::CCL, 0x400, bytes_);
            Copy(result, temporary, owner);
            for (uint64_t offset : {uint64_t{0}, bytes_ / 2}) {
                Copy(
                    Slice(owner, v3::MemType::CCL, 0x400 + offset, bytes_ / 2),
                    Slice(owner, v3::MemType::OUTPUT, offset, bytes_ / 2), owner);
            }
        }
        ExpectPass();
    }

    TEST_F(TaskGraphSemanticCheckV3Test, InplaceOutputAliasesInputWithoutChangingTree)
    {
        Init(HCCL_CMD_ALLREDUCE, 4, 8, true, true);
        std::vector<v3::MemSlice> results;
        // 先快照所有归约的源，再覆盖任何 rank 的原始输入。
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            results.push_back(BuildTree(owner, 0, bytes_, TreeShape::BALANCED));
        }
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            Copy(results[owner], Slice(owner, v3::MemType::OUTPUT, 0, bytes_), owner);
        }
        ExpectPass();
    }

    TEST_F(TaskGraphSemanticCheckV3Test, BatchReduceAndBatchCopyPreserveStrictTree)
    {
        Init(HCCL_CMD_ALLREDUCE, 4, 16);
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            WriteBatchedOwner(owner);
        }
        ExpectPass();
    }

    TEST_F(TaskGraphSemanticCheckV3Test, ReduceScatterBatchTasksRetainOwnerCoordinate)
    {
        Init(HCCL_CMD_REDUCE_SCATTER, 4, 16);
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            WriteBatchedOwner(owner);
        }
        ExpectPass();
    }

    TEST_F(TaskGraphSemanticCheckV3Test, BatchCopyReadsAllSourcesBeforeWritingAliasedDestinations)
    {
        Init();
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            auto first = Slice(owner, v3::MemType::CCL, 0, bytes_);
            auto second = Slice(owner, v3::MemType::CCL, 0x200, bytes_);
            Copy(Input(1, owner, 0, bytes_), first, owner);
            BatchCopy({Input(0, owner, 0, bytes_), first}, {first, second}, owner);
            Reduce(second, first, owner);
            Reduce(Input(2, owner, 0, bytes_), first, owner);
            Reduce(Input(3, owner, 0, bytes_), first, owner);
            Copy(first, Slice(owner, v3::MemType::OUTPUT, 0, bytes_), owner);
        }
        ExpectPass();
    }

    TEST_F(TaskGraphSemanticCheckV3Test, BatchReduceReadsAllSourcesBeforeWritingAliasedDestinations)
    {
        Init();
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            auto first = Slice(owner, v3::MemType::CCL, 0, bytes_);
            auto second = Slice(owner, v3::MemType::CCL, 0x200, bytes_);
            Copy(Input(0, owner, 0, bytes_), first, owner);
            Copy(Input(1, owner, 0, bytes_), second, owner);
            BatchReduce({second, first}, {first, second}, owner);
            Reduce(Input(2, owner, 0, bytes_), first, owner);
            Reduce(Input(3, owner, 0, bytes_), first, owner);
            Copy(first, Slice(owner, v3::MemType::OUTPUT, 0, bytes_), owner);
        }
        ExpectPass();
    }

    TEST_F(TaskGraphSemanticCheckV3Test, MultiSourceReduceIsIncompleteRatherThanAnInventedFold)
    {
        Init();
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            auto result = Slice(owner, v3::MemType::CCL, 0, bytes_);
            Copy(Input(0, owner, 0, bytes_), result, owner);
            std::vector<v3::MemSlice> sources;
            for (uint32_t rank = 1; rank < rankSize_; ++rank) {
                sources.push_back(Input(rank, owner, 0, bytes_));
            }
            Append(
                std::make_unique<v3::TaskReduce>(
                    sources, result, static_cast<uint8_t>(dataType_), static_cast<uint8_t>(op_)),
                owner);
            Copy(result, Slice(owner, v3::MemType::OUTPUT, 0, bytes_), owner);
        }
        ExpectFailure(v3::ReduceOrderStatus::INCOMPLETE);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, BatchReduceGroupsWithSameDestinationAreIncomplete)
    {
        Init();
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            auto result = Slice(owner, v3::MemType::CCL, 0, bytes_);
            Copy(Input(0, owner, 0, bytes_), result, owner);
            BatchReduce({Input(1, owner, 0, bytes_), Input(2, owner, 0, bytes_)}, {result, result}, owner);
            Reduce(Input(3, owner, 0, bytes_), result, owner);
            Copy(result, Slice(owner, v3::MemType::OUTPUT, 0, bytes_), owner);
        }
        // 旧的集合并集口径能放过，但两个 batch 组之间没有编码先后关系。
        ExpectFailure(v3::ReduceOrderStatus::INCOMPLETE);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, BatchReduceGroupWithMultipleSourcesIsIncomplete)
    {
        Init();
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            auto result = Slice(owner, v3::MemType::CCL, 0, bytes_);
            Copy(Input(0, owner, 0, bytes_), result, owner);
            auto batch
                = std::make_unique<v3::TaskBatchReduce>(static_cast<uint8_t>(dataType_), static_cast<uint8_t>(op_));
            batch->AddSrcMemSlice({Input(1, owner, 0, bytes_), Input(2, owner, 0, bytes_)});
            batch->AddDstMemSlice(result);
            Append(std::move(batch), owner);
            Reduce(Input(3, owner, 0, bytes_), result, owner);
            Copy(result, Slice(owner, v3::MemType::OUTPUT, 0, bytes_), owner);
        }
        ExpectFailure(v3::ReduceOrderStatus::INCOMPLETE);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, BatchCopyGroupsWithSameDestinationAreIncomplete)
    {
        Init();
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            auto result = Slice(owner, v3::MemType::CCL, 0, bytes_);
            BatchCopy({Input(0, owner, 0, bytes_), Input(1, owner, 0, bytes_)}, {result, result}, owner);
            // 若按顺序解释，只是保留 rank 1，最终集合依然合法。
            for (uint32_t rank : {0U, 2U, 3U}) {
                Reduce(Input(rank, owner, 0, bytes_), result, owner);
            }
            Copy(result, Slice(owner, v3::MemType::OUTPUT, 0, bytes_), owner);
        }
        ExpectFailure(v3::ReduceOrderStatus::INCOMPLETE);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, MissingCopySourceCannotProduceStrictPass)
    {
        Init();
        Copy(Slice(0, v3::MemType::CCL, 0x1000, bytes_), Slice(0, v3::MemType::CCL, 0x1200, bytes_), 0);
        // 旧的终局溯源校验能放过：缺源的临时区不是输出。
        WriteAll();
        ExpectFailure(v3::ReduceOrderStatus::INCOMPLETE);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, TaskDataTypeCannotDisagreeWithOperatorType)
    {
        Init();
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            auto result = Slice(owner, v3::MemType::CCL, 0, bytes_);
            Copy(Input(0, owner, 0, bytes_), result, owner);
            Reduce(Input(1, owner, 0, bytes_), result, owner, HCCL_DATA_TYPE_FP16);
            Reduce(Input(2, owner, 0, bytes_), result, owner);
            Reduce(Input(3, owner, 0, bytes_), result, owner);
            Copy(result, Slice(owner, v3::MemType::OUTPUT, 0, bytes_), owner);
        }
        ExpectFailure(v3::ReduceOrderStatus::MISMATCH);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, MixedReduceOperationsCannotProduceStrictPass)
    {
        Init();
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            auto result = Slice(owner, v3::MemType::CCL, 0, bytes_);
            Copy(Input(0, owner, 0, bytes_), result, owner);
            Append(
                std::make_unique<v3::TaskReduce>(
                    Input(1, owner, 0, bytes_), result, static_cast<uint8_t>(dataType_),
                    static_cast<uint8_t>(owner == 0 ? HCCL_REDUCE_MAX : HCCL_REDUCE_SUM)),
                owner);
            for (uint32_t rank : {2U, 3U}) {
                Append(
                    std::make_unique<v3::TaskReduce>(
                        Input(rank, owner, 0, bytes_), result, static_cast<uint8_t>(dataType_),
                        static_cast<uint8_t>(owner == 0 ? HCCL_REDUCE_MAX : HCCL_REDUCE_SUM)),
                    owner);
            }
            Copy(result, Slice(owner, v3::MemType::OUTPUT, 0, bytes_), owner);
        }
        v3::SemanticCheckStats stats;
        EXPECT_NE(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
        EXPECT_NE(stats.reduceOrder.status, v3::ReduceOrderStatus::PASS);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, NonStrictEnvironmentRetainsLegacyLeafSetBehaviorForArAndRs)
    {
        for (HcclCMDType command : {HCCL_CMD_ALLREDUCE, HCCL_CMD_REDUCE_SCATTER}) {
            for (const char* value : {static_cast<const char*>(nullptr), "FALSE", "TRUE", "2", ""}) {
                SCOPED_TRACE(
                    testing::Message() << "command=" << command << ", env=" << (value == nullptr ? "<unset>" : value));
                Init(command, 4, 8, false);
                if (value == nullptr) {
                    ASSERT_EQ(::unsetenv("HCCL_DETERMINISTIC"), 0);
                } else {
                    ASSERT_EQ(::setenv("HCCL_DETERMINISTIC", value, 1), 0);
                }
                for (uint32_t owner = 0; owner < rankSize_; ++owner) {
                    WriteOwner(owner, owner == 1 ? TreeShape::LEFT_DEEP : TreeShape::BALANCED);
                }
                v3::SemanticCheckStats stats;
                EXPECT_EQ(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
                EXPECT_EQ(stats.reduceOrder.status, v3::ReduceOrderStatus::DISABLED);
                EXPECT_EQ(stats.reduceOrder.expressionNodes, 0U);
            }
        }
    }

    TEST_F(TaskGraphSemanticCheckV3Test, StrictEnvironmentIsCaseInsensitiveForArAndRs)
    {
        for (HcclCMDType command : {HCCL_CMD_ALLREDUCE, HCCL_CMD_REDUCE_SCATTER}) {
            for (const char* value : {"STRICT", "strict", "StRiCt"}) {
                SCOPED_TRACE(testing::Message() << "command=" << command << ", env=" << value);
                Init(command, 4, 8, false);
                ASSERT_EQ(::setenv("HCCL_DETERMINISTIC", value, 1), 0);
                WriteAll();
                ExpectPass();
            }
        }
    }

    TEST_F(TaskGraphSemanticCheckV3Test, EnvironmentChangedAfterInitCanEnableOrderCheck)
    {
        for (HcclCMDType command : {HCCL_CMD_ALLREDUCE, HCCL_CMD_REDUCE_SCATTER}) {
            SCOPED_TRACE(static_cast<int>(command));
            Init(command, 4, 8, false);
            for (uint32_t owner = 0; owner < rankSize_; ++owner) {
                WriteOwner(owner, owner == 1 ? TreeShape::LEFT_DEEP : TreeShape::BALANCED);
            }
            v3::SemanticCheckStats stats;
            ASSERT_EQ(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
            EXPECT_EQ(stats.reduceOrder.status, v3::ReduceOrderStatus::DISABLED);
            EXPECT_EQ(stats.reduceOrder.expressionNodes, 0U);
            // StorageManager 的初始化与上一次校验都不会缓存本进程的环境开关。
            ASSERT_EQ(::setenv("HCCL_DETERMINISTIC", "STRICT", 1), 0);
            ExpectFailure(v3::ReduceOrderStatus::MISMATCH);
        }
    }

    TEST_F(TaskGraphSemanticCheckV3Test, EnvironmentChangedAfterInitCanDisableOrderCheck)
    {
        for (HcclCMDType command : {HCCL_CMD_ALLREDUCE, HCCL_CMD_REDUCE_SCATTER}) {
            SCOPED_TRACE(static_cast<int>(command));
            Init(command);
            for (uint32_t owner = 0; owner < rankSize_; ++owner) {
                WriteOwner(owner, owner == 1 ? TreeShape::LEFT_DEEP : TreeShape::BALANCED);
            }
            ExpectFailure(v3::ReduceOrderStatus::MISMATCH);
            ASSERT_EQ(::setenv("HCCL_DETERMINISTIC", "FALSE", 1), 0);
            v3::SemanticCheckStats stats;
            EXPECT_EQ(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
            EXPECT_EQ(stats.reduceOrder.status, v3::ReduceOrderStatus::DISABLED);
            EXPECT_EQ(stats.reduceOrder.expressionNodes, 0U);
        }
    }

    TEST_F(TaskGraphSemanticCheckV3Test, OldOperatorExtensionCannotOverrideCheckerEnvironment)
    {
        for (HcclCMDType command : {HCCL_CMD_ALLREDUCE, HCCL_CMD_REDUCE_SCATTER}) {
            for (bool enabled : {false, true}) {
                SCOPED_TRACE(testing::Message() << "command=" << command << ", enabled=" << enabled);
                Init(command, 4, 8, enabled);
                for (uint32_t rank = 0; rank < rankSize_; ++rank) {
                    sim::operation::OpDetailTab row{};
                    row.deviceId = Device(rank);
                    row.rankId = rank;
                    row.commId = kFirstCommId + rank;
                    row.rankSize = rankSize_;
                    // 刻意构造非法的历史快照：检查器不允许解析这个扩展字段。
                    row.opExtInfo = {'B', 'C', 'C', 'F', 0xff, static_cast<uint8_t>(rank)};
                    ::OpDetails detail{};
                    detail.opType = static_cast<uint16_t>(command);
                    detail.dataType = static_cast<uint16_t>(dataType_);
                    detail.reduceType = static_cast<uint16_t>(op_);
                    detail.opV1.count = count_;
                    ASSERT_EQ(storage_.Trans2CheckerParam(row, detail), HCCL_SUCCESS);
                }
                storage_.SaveCheckerParam(kOperatorId);
                for (uint32_t owner = 0; owner < rankSize_; ++owner) {
                    WriteOwner(owner, !enabled && owner == 1 ? TreeShape::LEFT_DEEP : TreeShape::BALANCED);
                }
                if (enabled) {
                    ExpectPass();
                } else {
                    v3::SemanticCheckStats stats;
                    EXPECT_EQ(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
                    EXPECT_EQ(stats.reduceOrder.status, v3::ReduceOrderStatus::DISABLED);
                    EXPECT_EQ(stats.reduceOrder.expressionNodes, 0U);
                }
            }
        }
    }

    TEST_F(TaskGraphSemanticCheckV3Test, NonAllReduceOrReduceScatterDoesNotAllocateTrees)
    {
        Init(HCCL_CMD_REDUCE, 4, 8, true);
        WriteOwner(0);
        v3::SemanticCheckStats stats;
        EXPECT_EQ(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.reduceOrder.status, v3::ReduceOrderStatus::DISABLED);
        EXPECT_EQ(stats.reduceOrder.expressionNodes, 0U);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, IntegerStrictReductionHasExplicitApplicability)
    {
        for (HcclCMDType command : {HCCL_CMD_ALLREDUCE, HCCL_CMD_REDUCE_SCATTER}) {
            SCOPED_TRACE(static_cast<int>(command));
            Init(command, 4, 8, true, false, HCCL_DATA_TYPE_INT32);
            WriteAll();
            v3::SemanticCheckStats stats;
            EXPECT_EQ(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
            // 请求了 strict 但整型算子不在契约内：DISABLED 且给出原因，而不是静默跳过。
            EXPECT_EQ(stats.reduceOrder.status, v3::ReduceOrderStatus::DISABLED);
            EXPECT_FALSE(stats.reduceOrder.detail.empty());
            EXPECT_EQ(stats.reduceOrder.expressionNodes, 0U);
        }
    }

    TEST_F(TaskGraphSemanticCheckV3Test, MaxAndMinStrictReductionHaveExplicitApplicability)
    {
        for (HcclCMDType command : {HCCL_CMD_ALLREDUCE, HCCL_CMD_REDUCE_SCATTER}) {
            for (HcclReduceOp op : {HCCL_REDUCE_MAX, HCCL_REDUCE_MIN}) {
                SCOPED_TRACE(testing::Message() << "command=" << command << ", op=" << op);
                Init(command, 4, 8, true, false, HCCL_DATA_TYPE_FP32, op);
                WriteAll();
                v3::SemanticCheckStats stats;
                EXPECT_EQ(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
                EXPECT_EQ(stats.reduceOrder.status, v3::ReduceOrderStatus::DISABLED);
                EXPECT_FALSE(stats.reduceOrder.detail.empty());
                EXPECT_EQ(stats.reduceOrder.expressionNodes, 0U);
            }
        }
    }

    TEST_F(TaskGraphSemanticCheckV3Test, NodeIdsAndIndependentReadyOrderDoNotDefineTree)
    {
        Init();
        for (uint32_t owner : {3U, 1U, 0U, 2U}) {
            tail_ = start_;
            WriteOwner(owner);
        }
        for (size_t index = 1; index < nodes_.size(); ++index) {
            nodes_[index]->SetNodeId(static_cast<v3::NodeId>(10000 - index * 7));
        }
        ExpectPass();
    }

    TEST_F(TaskGraphSemanticCheckV3Test, SingleDeviceCopyIsNotApplicable)
    {
        Init(HCCL_CMD_ALLREDUCE, 1);
        WriteAll();
        v3::SemanticCheckStats stats;
        EXPECT_EQ(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.reduceOrder.status, v3::ReduceOrderStatus::DISABLED);
        EXPECT_FALSE(stats.reduceOrder.detail.empty());
        EXPECT_EQ(stats.reduceOrder.expressionNodes, 0U);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, TwoDevicesAreNotApplicable)
    {
        Init(HCCL_CMD_ALLREDUCE, 2);
        WriteAll();
        v3::SemanticCheckStats stats;
        EXPECT_EQ(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.reduceOrder.status, v3::ReduceOrderStatus::DISABLED);
        EXPECT_FALSE(stats.reduceOrder.detail.empty());
        EXPECT_EQ(stats.reduceOrder.expressionNodes, 0U);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, TwoRankReduceScatterSelfThenPeerIsNotFalseMismatch)
    {
        Init(HCCL_CMD_REDUCE_SCATTER, 2);
        for (uint32_t owner = 0; owner < rankSize_; ++owner) {
            auto result = Slice(owner, v3::MemType::CCL, 0, bytes_);
            Copy(Input(owner, owner, 0, bytes_), result, owner);
            Reduce(Input(1U - owner, owner, 0, bytes_), result, owner);
            Copy(result, Slice(owner, v3::MemType::OUTPUT, 0, bytes_), owner);
        }
        v3::SemanticCheckStats stats;
        EXPECT_EQ(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.reduceOrder.status, v3::ReduceOrderStatus::DISABLED);
        EXPECT_FALSE(stats.reduceOrder.detail.empty());
        EXPECT_EQ(stats.reduceOrder.expressionNodes, 0U);
    }

    TEST_F(TaskGraphSemanticCheckV3Test, ZeroCountDoesNotReadOrReduceUnregisteredInput)
    {
        Init(HCCL_CMD_ALLREDUCE, 4, 0);
        v3::SemanticCheckStats stats;
        EXPECT_EQ(v3::CheckSemantics(start_, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.reduceOrder.status, v3::ReduceOrderStatus::PASS) << stats.reduceOrder.detail;
        EXPECT_EQ(stats.reduceOrder.outputSpans, 0U);
    }

} // namespace
} // namespace HcclSim
