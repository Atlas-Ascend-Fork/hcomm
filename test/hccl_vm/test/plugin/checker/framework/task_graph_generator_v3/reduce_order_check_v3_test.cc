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
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "reduce_order_check_v3.h"
#include "sim_log.h"

#include <spdlog/sinks/ostream_sink.h>

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        using Span = ReduceOrderOutputSpan;
        constexpr HcclDataType DTYPE = HCCL_DATA_TYPE_FP32;
        constexpr HcclReduceOp OP = HCCL_REDUCE_SUM;

        class ReduceOrderCheckTest : public testing::Test {
        protected:
            void SetUp() override
            {
                previousLogger_ = g_logger;
                auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(log_);
                logger_ = std::make_shared<spdlog::logger>("reduce-order-test", sink);
                logger_->set_pattern("%v");
                logger_->set_level(spdlog::level::info);
                g_logger = logger_.get();
            }

            void TearDown() override { g_logger = previousLogger_; }

            ReduceOrderExprId Pair(ReduceOrderExprId dst, ReduceOrderExprId src)
            {
                return arena_.Reduce(OP, DTYPE, dst, src);
            }

            ReduceOrderExprId Tree(const std::vector<uint32_t>& devices)
            {
                EXPECT_EQ(devices.size(), 4U);
                return Pair(
                    Pair(arena_.Leaf(devices[0]), arena_.Leaf(devices[2])),
                    Pair(arena_.Leaf(devices[1]), arena_.Leaf(devices[3])));
            }

            ReduceOrderExprId Tree() { return Tree(devices_); }

            ReduceOrderExprId Chain()
            {
                auto root = arena_.Leaf(devices_.front());
                for (size_t i = 1; i < devices_.size(); ++i) {
                    root = Pair(root, arena_.Leaf(devices_[i]));
                }
                return root;
            }

            std::vector<Span> Uniform(ReduceOrderExprId root, uint64_t count = 8)
            {
                std::vector<Span> spans;
                for (uint32_t device : devices_) {
                    spans.push_back({device, 0, count, root});
                }
                return spans;
            }

            // 与真实调用方一致：spans 已按设备分组、组内按元素偏移有序。
            HcclResult Check(std::vector<Span>& spans)
            {
                return CheckOutputReduceOrder(devices_, DTYPE, OP, arena_, spans, report_);
            }

            size_t LogCount(const std::string& text) const
            {
                size_t count = 0;
                size_t pos = 0;
                const auto log = log_.str();
                while ((pos = log.find(text, pos)) != std::string::npos) {
                    ++count;
                    pos += text.size();
                }
                return count;
            }

            ReduceOrderArena arena_;
            const std::vector<uint32_t> devices_{17, 3, 29, 11};
            ReduceOrderReport report_;
            std::ostringstream log_;
            spdlog::logger* previousLogger_{nullptr};
            std::shared_ptr<spdlog::logger> logger_;
        };

        TEST_F(ReduceOrderCheckTest, UniformGlobalDeviceTreesPass)
        {
            auto spans = Uniform(Tree());
            EXPECT_EQ(Check(spans), HCCL_SUCCESS);
            EXPECT_EQ(report_.status, ReduceOrderStatus::PASS);
            EXPECT_EQ(report_.comparedSpans, 4U);
            // PASS 的 detail 里给出参考树的求值顺序（叶子设备号序列），不打印完整树。
            EXPECT_NE(report_.detail.find("[17,29,3,11]"), std::string::npos);
            EXPECT_EQ(LogCount("ReduceOrder output,"), 4U);
            EXPECT_NE(
                log_.str().find("deviceId=17, elements=[0,8), byteOffset=0, byteCount=32, consistent=YES"),
                std::string::npos);
            EXPECT_NE(log_.str().find("order=[17,29,3,11]"), std::string::npos);
        }

        TEST_F(ReduceOrderCheckTest, UniformAlternativeAssociationPassesWithoutReferenceOracle)
        {
            auto spans = Uniform(Chain());
            EXPECT_EQ(Check(spans), HCCL_SUCCESS);
            EXPECT_EQ(report_.status, ReduceOrderStatus::PASS);
        }

        TEST_F(ReduceOrderCheckTest, ChangedAssociationAcrossDevicesIsMismatchAndEveryBlockIsLogged)
        {
            auto spans = Uniform(Tree());
            spans[2].expr = Chain();
            EXPECT_EQ(Check(spans), HCCL_E_PARA);
            EXPECT_EQ(report_.status, ReduceOrderStatus::MISMATCH);
            EXPECT_EQ(report_.comparedSpans, 4U);
            EXPECT_NE(report_.detail.find("deviceId=29"), std::string::npos);
            EXPECT_NE(report_.detail.find("expected="), std::string::npos);
            EXPECT_NE(report_.detail.find("actual="), std::string::npos);
            EXPECT_EQ(LogCount("ReduceOrder output,"), 4U);
            EXPECT_EQ(LogCount("byteCount=32, consistent=NO"), 1U);
            EXPECT_EQ(LogCount("ReduceOrder output check failed, status=MISMATCH"), 1U);
            EXPECT_NE(log_.str().find("deviceId=11, elements=[0,8)"), std::string::npos);
        }

        TEST_F(ReduceOrderCheckTest, ReversingRootOperandsIsMismatch)
        {
            const auto left = Pair(arena_.Leaf(17), arena_.Leaf(29));
            const auto right = Pair(arena_.Leaf(3), arena_.Leaf(11));
            auto spans = Uniform(Pair(left, right));
            spans.back().expr = Pair(right, left);
            EXPECT_EQ(Check(spans), HCCL_E_PARA);
            EXPECT_EQ(report_.status, ReduceOrderStatus::MISMATCH);
        }

        TEST_F(ReduceOrderCheckTest, ReversingInternalOperandsIsMismatch)
        {
            const auto right = Pair(arena_.Leaf(3), arena_.Leaf(11));
            auto spans = Uniform(Pair(Pair(arena_.Leaf(17), arena_.Leaf(29)), right));
            spans.back().expr = Pair(Pair(arena_.Leaf(29), arena_.Leaf(17)), right);
            EXPECT_EQ(Check(spans), HCCL_E_PARA);
            EXPECT_EQ(report_.status, ReduceOrderStatus::MISMATCH);
        }

        TEST_F(ReduceOrderCheckTest, MissingSourceDeviceIsIncomplete)
        {
            auto spans = Uniform(Pair(Pair(arena_.Leaf(17), arena_.Leaf(3)), arena_.Leaf(29)));
            EXPECT_EQ(Check(spans), HCCL_E_PARA);
            EXPECT_EQ(report_.status, ReduceOrderStatus::INCOMPLETE);
            EXPECT_EQ(LogCount("byteCount=32, consistent=INVALID"), 4U);
        }

        TEST_F(ReduceOrderCheckTest, ForeignSourceDeviceIsIncomplete)
        {
            auto spans = Uniform(Tree({17, 3, 29, 99}));
            EXPECT_EQ(Check(spans), HCCL_E_PARA);
            EXPECT_EQ(report_.status, ReduceOrderStatus::INCOMPLETE);
            EXPECT_NE(report_.detail.find("foreign or duplicate"), std::string::npos);
        }

        TEST_F(ReduceOrderCheckTest, DuplicateSourceDeviceIsIncomplete)
        {
            auto spans = Uniform(Tree({17, 3, 29, 17}));
            EXPECT_EQ(Check(spans), HCCL_E_PARA);
            EXPECT_EQ(report_.status, ReduceOrderStatus::INCOMPLETE);
        }

        TEST_F(ReduceOrderCheckTest, SharedSubtreeIsExpandedAndCannotHideRepeatedOperands)
        {
            auto subtree = Pair(arena_.Leaf(17), arena_.Leaf(3));
            auto spans = Uniform(Pair(subtree, subtree));
            EXPECT_EQ(Check(spans), HCCL_E_PARA);
            EXPECT_EQ(report_.status, ReduceOrderStatus::INCOMPLETE);
        }

        TEST_F(ReduceOrderCheckTest, OversizedOperandTreeHitsExpansionBound)
        {
            auto spans = Uniform(Pair(Tree(), Tree()));
            EXPECT_EQ(Check(spans), HCCL_E_PARA);
            EXPECT_EQ(report_.status, ReduceOrderStatus::INCOMPLETE);
        }

        TEST_F(ReduceOrderCheckTest, WrongActualOperationOrDatatypeIsMismatch)
        {
            const auto left = Pair(arena_.Leaf(17), arena_.Leaf(29));
            const auto right = Pair(arena_.Leaf(3), arena_.Leaf(11));
            for (bool wrongOp : {true, false}) {
                auto root = arena_.Reduce(
                    wrongOp ? HCCL_REDUCE_PROD : OP, wrongOp ? DTYPE : HCCL_DATA_TYPE_FP16, left, right);
                auto spans = Uniform(root);
                EXPECT_EQ(Check(spans), HCCL_E_PARA);
                EXPECT_EQ(report_.status, ReduceOrderStatus::MISMATCH);
            }
            EXPECT_EQ(LogCount("byteCount=32, consistent=INVALID"), 8U);
        }

        TEST_F(ReduceOrderCheckTest, UnknownAndOutOfRangeExpressionIdsAreIncomplete)
        {
            for (auto id : {INVALID_REDUCE_ORDER_EXPR, std::numeric_limits<ReduceOrderExprId>::max()}) {
                auto spans = Uniform(id);
                EXPECT_EQ(Check(spans), HCCL_E_PARA);
                EXPECT_EQ(report_.status, ReduceOrderStatus::INCOMPLETE);
                EXPECT_EQ(report_.comparedSpans, 0U);
            }
            EXPECT_EQ(LogCount("byteCount=32, consistent=INVALID"), 8U);
        }

        TEST_F(ReduceOrderCheckTest, FirstValidBlockProvidesReferenceWhenFirstDeviceIsInvalid)
        {
            auto spans = Uniform(Tree());
            spans.front().expr = INVALID_REDUCE_ORDER_EXPR;
            spans.back().expr = Chain();
            EXPECT_EQ(Check(spans), HCCL_E_PARA);
            // 首个块无法建模（INCOMPLETE，优先级更高），参考树退到第二个块；
            // 末块与参考树不一致的 MISMATCH 仍逐块诊断，但不覆盖首条 INCOMPLETE。
            EXPECT_EQ(report_.status, ReduceOrderStatus::INCOMPLETE);
            EXPECT_NE(report_.detail.find("deviceId=17"), std::string::npos);
            EXPECT_EQ(report_.comparedSpans, 3U);
            EXPECT_EQ(LogCount("byteCount=32, consistent=INVALID"), 1U);
            EXPECT_EQ(LogCount("byteCount=32, consistent=YES"), 2U);
            EXPECT_EQ(LogCount("byteCount=32, consistent=NO"), 1U);
        }

        TEST_F(ReduceOrderCheckTest, IncompleteWinsOverMismatch)
        {
            auto spans = Uniform(Tree());
            spans[1].expr = Chain();                   // 与参考树不一致 → MISMATCH
            spans[3].expr = INVALID_REDUCE_ORDER_EXPR; // 无法建模 → INCOMPLETE（优先级更高，覆盖前者）
            EXPECT_EQ(Check(spans), HCCL_E_PARA);
            EXPECT_EQ(report_.status, ReduceOrderStatus::INCOMPLETE);
            EXPECT_NE(report_.detail.find("deviceId=11"), std::string::npos);
            EXPECT_EQ(LogCount("ReduceOrder output,"), 4U);
        }

        TEST_F(ReduceOrderCheckTest, DifferentPartitionsBecomeOneMaximalLoggedBlockPerDevice)
        {
            const auto root = Tree();
            std::vector<Span> spans{{17, 0, 4, root}, {17, 4, 4, root}, {3, 0, 8, root}, {29, 0, 3, root},
                                    {29, 3, 5, root}, {11, 0, 1, root}, {11, 1, 7, root}};
            EXPECT_EQ(Check(spans), HCCL_SUCCESS);
            ASSERT_EQ(spans.size(), 4U);
            for (size_t i = 0; i < spans.size(); ++i) {
                EXPECT_EQ(spans[i].begin, 0U);
                EXPECT_EQ(spans[i].count, 8U);
                EXPECT_EQ(spans[i].expr, root);
            }
            EXPECT_EQ(report_.outputSpans, 4U);
            EXPECT_EQ(LogCount("ReduceOrder output,"), 4U);
            EXPECT_EQ(LogCount("elements=[0,8), byteOffset=0, byteCount=32, consistent=YES"), 4U);
            EXPECT_EQ(log_.str().find("elements=[0,4)"), std::string::npos);
        }

        TEST_F(ReduceOrderCheckTest, DifferentTreesWithinOneOutputKeepBoundaryAndFail)
        {
            const auto root = Tree();
            auto spans = Uniform(root);
            spans.front() = {17, 0, 4, root};
            spans.insert(spans.begin() + 1, {17, 4, 4, Chain()});
            EXPECT_EQ(Check(spans), HCCL_E_PARA);
            EXPECT_EQ(report_.status, ReduceOrderStatus::MISMATCH);
            ASSERT_EQ(spans.size(), 5U);
            EXPECT_EQ(spans[0].count, 4U);
            EXPECT_EQ(spans[1].begin, 4U);
            EXPECT_EQ(report_.outputSpans, 5U);
            EXPECT_NE(log_.str().find("elements=[4,8), byteOffset=16, byteCount=16, consistent=NO"), std::string::npos);
        }

        TEST_F(ReduceOrderCheckTest, InvalidAdjacentFragmentsAreAlsoCoalesced)
        {
            auto spans = Uniform(INVALID_REDUCE_ORDER_EXPR);
            spans.front() = {17, 0, 4, INVALID_REDUCE_ORDER_EXPR};
            spans.insert(spans.begin() + 1, {17, 4, 4, INVALID_REDUCE_ORDER_EXPR});
            EXPECT_EQ(Check(spans), HCCL_E_PARA);
            EXPECT_EQ(report_.status, ReduceOrderStatus::INCOMPLETE);
            EXPECT_EQ(spans.size(), 4U);
            EXPECT_EQ(LogCount("ReduceOrder output,"), 4U);
        }

        TEST_F(ReduceOrderCheckTest, MoreThanSixteenBlocksAreAllCheckedAndLastMismatchIsLogged)
        {
            const auto a = Tree();
            const auto b = Chain();
            std::vector<Span> spans;
            for (uint32_t device : devices_) {
                for (uint64_t i = 0; i < 20; ++i) {
                    spans.push_back({device, i, 1, i % 2 == 0 ? a : b});
                }
            }
            EXPECT_EQ(Check(spans), HCCL_E_PARA);
            EXPECT_EQ(report_.status, ReduceOrderStatus::MISMATCH);
            EXPECT_EQ(report_.outputSpans, 80U);
            EXPECT_EQ(report_.comparedSpans, 80U);
            EXPECT_EQ(LogCount("ReduceOrder output,"), 80U);
            EXPECT_NE(
                log_.str().find("deviceId=11, elements=[19,20), byteOffset=76, byteCount=4, consistent=NO"),
                std::string::npos);
        }

        TEST_F(ReduceOrderCheckTest, FirstMismatchDiagnosticIsNotOverwrittenByLaterMismatches)
        {
            auto spans = Uniform(Tree());
            spans[1].expr = Chain();
            spans[3].expr = Chain();
            EXPECT_EQ(Check(spans), HCCL_E_PARA);
            EXPECT_NE(report_.detail.find("deviceId=3"), std::string::npos);
            EXPECT_EQ(LogCount("byteCount=32, consistent=NO"), 2U);
        }

        TEST_F(ReduceOrderCheckTest, HugeElementCountReportsByteOverflowWithoutWrapping)
        {
            auto spans = Uniform(Tree(), UINT64_MAX);
            EXPECT_EQ(Check(spans), HCCL_SUCCESS);
            EXPECT_EQ(report_.comparedSpans, 4U);
            EXPECT_EQ(report_.outputSpans, 4U);
            EXPECT_EQ(LogCount("byteCount=OVERFLOW"), 4U);
        }

        TEST_F(ReduceOrderCheckTest, EmptyOutputPassesWithoutExpressions)
        {
            std::vector<Span> spans;
            EXPECT_EQ(Check(spans), HCCL_SUCCESS);
            EXPECT_EQ(report_.status, ReduceOrderStatus::PASS);
            EXPECT_EQ(report_.outputSpans, 0U);
            EXPECT_EQ(report_.comparedSpans, 0U);
        }

        TEST_F(ReduceOrderCheckTest, LongTreeIsLoggedAsLeafOrderOnOneLine)
        {
            std::vector<uint32_t> devices;
            for (uint32_t i = 0; i < 128; ++i) {
                devices.push_back(10000 + i);
            }
            auto root = arena_.Leaf(devices.front());
            for (size_t i = 1; i < devices.size(); ++i) {
                root = Pair(root, arena_.Leaf(devices[i]));
            }
            std::vector<Span> spans;
            for (uint32_t device : devices) {
                spans.push_back({device, 0, 1, root});
            }
            EXPECT_EQ(CheckOutputReduceOrder(devices, DTYPE, OP, arena_, spans, report_), HCCL_SUCCESS);
            EXPECT_EQ(report_.outputSpans, 128U);
            EXPECT_EQ(LogCount("ReduceOrder output,"), 128U);
            // 128 个叶子的树只打一行求值顺序，不打印完整树、不分块。
            EXPECT_NE(log_.str().find("order=[10000,10001"), std::string::npos);
            EXPECT_NE(log_.str().find(",10127]"), std::string::npos);
            EXPECT_EQ(log_.str().find("chunk="), std::string::npos);
            EXPECT_EQ(log_.str().find("R["), std::string::npos);
            EXPECT_EQ(log_.str().find("..."), std::string::npos);
        }

        TEST_F(ReduceOrderCheckTest, StatusDescriptionsRemainStable)
        {
            for (const auto status :
                 {ReduceOrderStatus::DISABLED, ReduceOrderStatus::PASS, ReduceOrderStatus::MISMATCH,
                  ReduceOrderStatus::INCOMPLETE}) {
                EXPECT_STRNE(DescribeReduceOrderStatus(status), "INVALID");
            }
            EXPECT_STREQ(DescribeReduceOrderStatus(static_cast<ReduceOrderStatus>(255)), "INVALID");
        }
    } // namespace
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
