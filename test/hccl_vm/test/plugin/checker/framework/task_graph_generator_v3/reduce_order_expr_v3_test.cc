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
#include <limits>
#include <string>

#include "reduce_order_check_v3.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        TEST(ReduceOrderArenaTest, DefaultExpressionIsInvalidAndInitialized)
        {
            const ReduceOrderExpr expr;
            EXPECT_EQ(expr.kind, ReduceOrderExpr::Kind::INVALID);
            EXPECT_EQ(expr.deviceId, 0U);
            EXPECT_EQ(expr.op, HCCL_REDUCE_RESERVED);
            EXPECT_EQ(expr.dataType, HCCL_DATA_TYPE_RESERVED);
            EXPECT_EQ(expr.dst, INVALID_REDUCE_ORDER_EXPR);
            EXPECT_EQ(expr.src, INVALID_REDUCE_ORDER_EXPR);
        }

        TEST(ReduceOrderArenaTest, InternsLeavesAndCountsWithoutDummy)
        {
            ReduceOrderArena arena;
            EXPECT_EQ(arena.NodeCount(), 0U);
            EXPECT_EQ(arena.Get(INVALID_REDUCE_ORDER_EXPR), nullptr);
            const auto first = arena.Leaf(7);
            ASSERT_NE(first, INVALID_REDUCE_ORDER_EXPR);
            EXPECT_EQ(arena.Leaf(7), first);
            EXPECT_EQ(arena.NodeCount(), 1U);
            const auto second = arena.Leaf(3);
            EXPECT_NE(first, second);
            EXPECT_EQ(arena.NodeCount(), 2U);
            const auto* expr = arena.Get(first);
            ASSERT_NE(expr, nullptr);
            EXPECT_EQ(expr->kind, ReduceOrderExpr::Kind::LEAF);
            EXPECT_EQ(expr->deviceId, 7U);
            EXPECT_EQ(expr->dst, INVALID_REDUCE_ORDER_EXPR);
            EXPECT_EQ(expr->src, INVALID_REDUCE_ORDER_EXPR);
            EXPECT_EQ(arena.Describe(first), "device(7)");
            const auto largestDevice = arena.Leaf(std::numeric_limits<uint32_t>::max());
            EXPECT_EQ(arena.Describe(largestDevice), "device(4294967295)");
        }

        TEST(ReduceOrderArenaTest, InternsReductionsByTheExactOrderedKey)
        {
            ReduceOrderArena arena;
            const auto dst = arena.Leaf(0);
            const auto src = arena.Leaf(1);
            const auto sum = arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, dst, src);
            EXPECT_EQ(arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, dst, src), sum);
            EXPECT_EQ(arena.NodeCount(), 3U);
            const auto* expr = arena.Get(sum);
            ASSERT_NE(expr, nullptr);
            EXPECT_EQ(expr->kind, ReduceOrderExpr::Kind::REDUCE);
            EXPECT_EQ(expr->op, HCCL_REDUCE_SUM);
            EXPECT_EQ(expr->dataType, HCCL_DATA_TYPE_FP32);
            EXPECT_EQ(expr->dst, dst);
            EXPECT_EQ(expr->src, src);
            EXPECT_EQ(arena.Describe(sum), "R[0,4](device(0),device(1))");
            EXPECT_NE(arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, src, dst), sum);
            EXPECT_NE(arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP16, dst, src), sum);
            EXPECT_NE(arena.Reduce(HCCL_REDUCE_PROD, HCCL_DATA_TYPE_FP32, dst, src), sum);
            EXPECT_EQ(arena.NodeCount(), 6U);
        }

        TEST(ReduceOrderArenaTest, PreservesAssociationAndDoesNotFlattenOrDeduplicateChildren)
        {
            ReduceOrderArena arena;
            const auto a = arena.Leaf(0);
            const auto b = arena.Leaf(1);
            const auto c = arena.Leaf(2);
            const auto ab = arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, a, b);
            const auto bc = arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, b, c);
            const auto left = arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, ab, c);
            const auto right = arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, a, bc);
            EXPECT_NE(left, right);
            EXPECT_EQ(arena.Describe(left), "R[0,4](R[0,4](device(0),device(1)),device(2))");
            EXPECT_EQ(arena.Describe(right), "R[0,4](device(0),R[0,4](device(1),device(2)))");
            const auto repeated = arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, a, a);
            EXPECT_NE(repeated, a);
            EXPECT_EQ(arena.Describe(repeated), "R[0,4](device(0),device(0))");
        }

        TEST(ReduceOrderArenaTest, RejectsInvalidAndUnknownChildrenWithoutAddingNodes)
        {
            ReduceOrderArena arena;
            const auto valid = arena.Leaf(0);
            const auto unknown = std::numeric_limits<ReduceOrderExprId>::max();
            EXPECT_EQ(arena.Get(unknown), nullptr);
            EXPECT_EQ(arena.Get(valid + 1), nullptr);
            EXPECT_EQ(arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, 0, valid), 0U);
            EXPECT_EQ(arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, valid, 0), 0U);
            EXPECT_EQ(arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, unknown, valid), 0U);
            EXPECT_EQ(arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, valid, unknown), 0U);
            EXPECT_EQ(arena.NodeCount(), 1U);
            EXPECT_EQ(arena.Describe(0), "INVALID");
            EXPECT_EQ(arena.Describe(unknown), "INVALID");
        }

        TEST(ReduceOrderArenaTest, KeepsReservedOperationAndDatatypeAsExactValues)
        {
            ReduceOrderArena arena;
            const auto a = arena.Leaf(4);
            const auto b = arena.Leaf(8);
            const auto reserved = arena.Reduce(HCCL_REDUCE_RESERVED, HCCL_DATA_TYPE_RESERVED, a, b);
            ASSERT_NE(reserved, INVALID_REDUCE_ORDER_EXPR);
            EXPECT_EQ(arena.Describe(reserved), "R[255,255](device(4),device(8))");
            EXPECT_EQ(arena.Reduce(HCCL_REDUCE_RESERVED, HCCL_DATA_TYPE_RESERVED, a, b), reserved);
        }

        TEST(ReduceOrderArenaTest, RetainsNoncontiguousGlobalDeviceIdentity)
        {
            ReduceOrderArena arena;
            const auto device17 = arena.Leaf(17);
            const auto device3 = arena.Leaf(3);
            const auto device29 = arena.Leaf(29);
            EXPECT_EQ(arena.Leaf(17), device17);
            EXPECT_NE(device17, device3);
            EXPECT_NE(device3, device29);
            ASSERT_NE(arena.Get(device17), nullptr);
            EXPECT_EQ(arena.Get(device17)->deviceId, 17U);
            ASSERT_NE(arena.Get(device29), nullptr);
            EXPECT_EQ(arena.Get(device29)->deviceId, 29U);
            const auto root = arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, device17, device3);
            EXPECT_EQ(arena.Describe(root), "R[0,4](device(17),device(3))");
            EXPECT_NE(root, arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, device3, device17));
        }

        TEST(ReduceOrderArenaTest, LeafOrderFollowsDstBeforeSrcEvaluationOrder)
        {
            ReduceOrderArena arena;
            const auto a = arena.Leaf(17);
            const auto b = arena.Leaf(29);
            const auto c = arena.Leaf(3);
            const auto d = arena.Leaf(11);
            const auto left = arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, a, b);
            const auto right = arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, c, d);
            const auto root = arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, left, right);
            EXPECT_EQ(arena.DescribeLeafOrder(root), "[17,29,3,11]");
            const auto reversed = arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, right, left);
            EXPECT_EQ(arena.DescribeLeafOrder(reversed), "[3,11,17,29]");
            EXPECT_EQ(arena.DescribeLeafOrder(a), "[17]");
            EXPECT_EQ(arena.DescribeLeafOrder(INVALID_REDUCE_ORDER_EXPR), "[?]");
            // 共享子树按展开展开：同一设备出现两次也原样呈现。
            const auto shared = arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, left, left);
            EXPECT_EQ(arena.DescribeLeafOrder(shared), "[17,29,17,29]");
        }

        TEST(ReduceOrderArenaTest, LeafOrderStopsAtLeafBudget)
        {
            ReduceOrderArena arena;
            auto root = arena.Leaf(9);
            for (size_t i = 0; i < 128; ++i) {
                root = arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, root, root);
            }
            // 指数展开的 DAG 只有 129 个节点；叶子序按预算截断，不会真的展开 2^128 个叶子。
            EXPECT_EQ(arena.NodeCount(), 129U);
            EXPECT_EQ(arena.DescribeLeafOrder(root, 8), "[9,9,9,9,9,9,9,9,...]");
        }

        TEST(ReduceOrderArenaTest, DeepChainDescriptionIsIterative)
        {
            ReduceOrderArena arena;
            const auto leaf = arena.Leaf(0);
            auto root = leaf;
            constexpr size_t DEPTH = 50000;
            for (size_t i = 0; i < DEPTH; ++i) {
                root = arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, root, leaf);
            }
            EXPECT_EQ(arena.NodeCount(), DEPTH + 1);
            // 沿左深链整体展开，覆盖显式栈路径（不递归，深链不爆栈）。
            const auto fullDescription = arena.Describe(root);
            EXPECT_EQ(fullDescription.size(), DEPTH * 18 + 9);
            EXPECT_EQ(fullDescription.substr(fullDescription.size() - 11), ",device(0))");
            // 同一棵树的叶子序按预算截断，长度可控。
            const auto leafOrder = arena.DescribeLeafOrder(root);
            EXPECT_LT(leafOrder.size(), 4096U);
            EXPECT_NE(leafOrder.find(",...]"), std::string::npos);
        }

        TEST(ReduceOrderArenaTest, CompleteDescriptionRetainsEveryDistinctDeviceLeaf)
        {
            ReduceOrderArena arena;
            auto root = arena.Leaf(17);
            std::string expected = "device(17)";
            constexpr uint32_t LEAF_COUNT = 128;
            for (uint32_t index = 1; index < LEAF_COUNT; ++index) {
                const uint32_t deviceId = 17 + index * 11;
                root = arena.Reduce(HCCL_REDUCE_SUM, HCCL_DATA_TYPE_FP32, root, arena.Leaf(deviceId));
                expected = "R[0,4](" + expected + ",device(" + std::to_string(deviceId) + "))";
            }
            EXPECT_EQ(arena.NodeCount(), LEAF_COUNT * 2 - 1);
            EXPECT_EQ(arena.Describe(root), expected);
        }

    } // namespace
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
