/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_REDUCE_ORDER_CHECK_V3_H
#define CHECKER_REDUCE_ORDER_CHECK_V3_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "hccl_types.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    // 规约顺序表达式：记录语义运行中的叶子、部分和与完整输出树。
    using ReduceOrderExprId = uint32_t;
    constexpr ReduceOrderExprId INVALID_REDUCE_ORDER_EXPR = 0;

    struct ReduceOrderExpr {
        enum class Kind : uint8_t { INVALID, LEAF, REDUCE };

        Kind kind{Kind::INVALID};
        uint32_t deviceId{0};
        HcclReduceOp op{HCCL_REDUCE_RESERVED};
        HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
        ReduceOrderExprId dst{INVALID_REDUCE_ORDER_EXPR};
        ReduceOrderExprId src{INVALID_REDUCE_ORDER_EXPR};
    };

    // ID 仅在本 arena 内有效。相等性保留结合方式与 dst/src 的操作数顺序；
    // 不把任何算子当作可交换或可结合。
    class ReduceOrderArena {
    public:
        ReduceOrderExprId Leaf(uint32_t deviceId);
        ReduceOrderExprId Reduce(HcclReduceOp op, HcclDataType dtype, ReduceOrderExprId dst, ReduceOrderExprId src);

        // 非法/未知 ID 返回 nullptr。后续插入可能使指针失效。
        const ReduceOrderExpr* Get(ReduceOrderExprId id) const;
        size_t NodeCount() const;

        // 完整打印表达式结构（含结合方式），仅供调试与测试：叶子打印 device(<id>)，
        // op/dtype 打印数值（与枚举名表解耦），非法 ID 打印 INVALID。
        // 日志不打印完整树（见 DescribeLeafOrder）。
        std::string Describe(ReduceOrderExprId id) const;

        // 按求值顺序（先 dst 后 src）展开叶子设备号，形如 "[17,29,3,11]"；日志只用它
        // 打印规约顺序。maxLeaves 是展开预算（防非法 DAG 指数膨胀），超出后末尾追加
        // ",..."；非法节点记为 "?"。
        std::string DescribeLeafOrder(ReduceOrderExprId id, size_t maxLeaves = 1024) const;

    private:
        using Key = std::tuple<
            ReduceOrderExpr::Kind, uint32_t, HcclReduceOp, HcclDataType, ReduceOrderExprId, ReduceOrderExprId>;

        ReduceOrderExprId Intern(const ReduceOrderExpr& expr);

        // 0 号是占位 dummy，永远不是合法表达式。
        std::vector<ReduceOrderExpr> nodes_{ReduceOrderExpr{}};
        std::map<Key, ReduceOrderExprId> interned_;
    };

    // 输出一致性校验：按完整树结构比较各输出区间的规约顺序。
    // DISABLED：未开启（含"请求了 strict 但算子不在契约内"，此时 detail 说明原因）；
    // INCOMPLETE：无法完成校验（覆盖/坐标/映射前提不成立，或出现无法建模的形态），与 MISMATCH 一样按失败处理。
    enum class ReduceOrderStatus : uint8_t { DISABLED, PASS, MISMATCH, INCOMPLETE };
    const char* DescribeReduceOrderStatus(ReduceOrderStatus status);

    struct ReduceOrderOutputSpan {
        uint32_t deviceId{0};
        uint64_t begin{0}; // 输出元素偏移（不是地址，也不是通信域内 rank 号）
        uint64_t count{0};
        ReduceOrderExprId expr{INVALID_REDUCE_ORDER_EXPR};
    };

    struct ReduceOrderReport {
        ReduceOrderStatus status{ReduceOrderStatus::DISABLED};
        uint64_t comparedSpans{0};
        uint64_t outputSpans{0};
        uint64_t expressionNodes{0};
        std::string detail;
    };

    // 仅在调用方开启校验时调用。
    // devices 是全量的全局设备号集合（不假设 rank 编号）；首个树合法的块的表达式作为参考树。
    // 调用方（CheckFinalReduceOrder）保证：spans 已按设备分组、组内按元素偏移有序、且完整覆盖输出；
    // 本函数负责把相邻同树区间合并成极大块、校验树形状、并逐块比较与打日志（失败也不提前退出）。
    HcclResult CheckOutputReduceOrder(
        const std::vector<uint32_t>& devices, HcclDataType dataType, HcclReduceOp reduceOp,
        const ReduceOrderArena& arena, std::vector<ReduceOrderOutputSpan>& spans, ReduceOrderReport& report);
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
#endif
