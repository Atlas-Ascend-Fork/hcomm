/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_order_check_v3.h"

#include <limits>
#include <map>
#include <set>
#include <utility>

#include "checker_def.h"
#include "sim_log.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    ReduceOrderExprId ReduceOrderArena::Intern(const ReduceOrderExpr& expr)
    {
        const Key key{expr.kind, expr.deviceId, expr.op, expr.dataType, expr.dst, expr.src};
        const auto found = interned_.find(key);
        if (found != interned_.end()) {
            return found->second;
        }
        const auto id = static_cast<ReduceOrderExprId>(nodes_.size());
        nodes_.push_back(expr);
        interned_.emplace(key, id);
        return id;
    }

    ReduceOrderExprId ReduceOrderArena::Leaf(uint32_t deviceId)
    {
        ReduceOrderExpr expr;
        expr.kind = ReduceOrderExpr::Kind::LEAF;
        expr.deviceId = deviceId;
        return Intern(expr);
    }

    ReduceOrderExprId
    ReduceOrderArena::Reduce(HcclReduceOp op, HcclDataType dtype, ReduceOrderExprId dst, ReduceOrderExprId src)
    {
        if (Get(dst) == nullptr || Get(src) == nullptr) {
            return INVALID_REDUCE_ORDER_EXPR;
        }
        ReduceOrderExpr expr;
        expr.kind = ReduceOrderExpr::Kind::REDUCE;
        expr.op = op;
        expr.dataType = dtype;
        expr.dst = dst;
        expr.src = src;
        return Intern(expr);
    }

    const ReduceOrderExpr* ReduceOrderArena::Get(ReduceOrderExprId id) const
    {
        if (id == INVALID_REDUCE_ORDER_EXPR || static_cast<size_t>(id) >= nodes_.size()) {
            return nullptr;
        }
        return &nodes_[id];
    }

    size_t ReduceOrderArena::NodeCount() const { return nodes_.empty() ? 0 : nodes_.size() - 1; }

    std::string ReduceOrderArena::Describe(ReduceOrderExprId id) const
    {
        struct Frame {
            ReduceOrderExprId id;
            uint8_t stage{0};
        };
        std::vector<Frame> stack{{id}};
        std::string result;
        // 每个栈帧依次输出：前缀、左子树、逗号、右子树、后缀；显式栈避免深链递归。
        while (!stack.empty()) {
            Frame& frame = stack.back();
            const auto* expr = Get(frame.id);
            if (expr == nullptr) {
                result += "INVALID";
                stack.pop_back();
            } else if (expr->kind == ReduceOrderExpr::Kind::LEAF) {
                result += "device(" + std::to_string(expr->deviceId) + ")";
                stack.pop_back();
            } else if (frame.stage == 0) {
                result += "R[" + std::to_string(static_cast<int>(expr->op)) + ","
                          + std::to_string(static_cast<int>(expr->dataType)) + "](";
                frame.stage = 1;
                stack.push_back({expr->dst});
            } else if (frame.stage == 1) {
                result += ",";
                frame.stage = 2;
                stack.push_back({expr->src});
            } else {
                result += ")";
                stack.pop_back();
            }
        }
        return result;
    }

    std::string ReduceOrderArena::DescribeLeafOrder(ReduceOrderExprId id, size_t maxLeaves) const
    {
        std::vector<ReduceOrderExprId> stack{id};
        std::string result = "[";
        size_t leaves = 0;
        bool first = true;
        const auto append = [&](const std::string& text) {
            if (!first) {
                result += ",";
            }
            result += text;
            first = false;
            ++leaves;
        };
        while (!stack.empty()) {
            if (leaves >= maxLeaves) {
                result += ",...";
                break;
            }
            const auto cur = stack.back();
            stack.pop_back();
            const auto* node = Get(cur);
            if (node == nullptr || node->kind == ReduceOrderExpr::Kind::INVALID) {
                append("?");
            } else if (node->kind == ReduceOrderExpr::Kind::LEAF) {
                append(std::to_string(node->deviceId));
            } else {
                // 栈是 LIFO：先压 src 再压 dst，弹出顺序与求值顺序（先 dst）一致。
                stack.push_back(node->src);
                stack.push_back(node->dst);
            }
        }
        result += "]";
        return result;
    }

    namespace {
        struct TreeResult {
            ReduceOrderStatus status{ReduceOrderStatus::PASS};
            std::string detail;
        };

        // 同严重级保留首条诊断（便于定位最靠前的异常）；INCOMPLETE（前提不成立）优先于 MISMATCH。
        void RecordFailure(TreeResult& result, ReduceOrderStatus status, const std::string& detail)
        {
            if (result.status == ReduceOrderStatus::PASS
                || (status == ReduceOrderStatus::INCOMPLETE && result.status != ReduceOrderStatus::INCOMPLETE)) {
                result.status = status;
                result.detail = detail;
            }
        }

        // 校验单棵树的形状：叶子必须恰好覆盖全部参与设备（不多、不少、不重复），
        // 归约节点的 op/dtype 必须与算子一致，展开后的操作数个数不得超过上界 2N-1。
        // 按展开后的操作数计数而不是唯一节点数：hash-consing 共享子树时，重复引用会重复贡献叶子。
        TreeResult ValidateTree(
            ReduceOrderExprId root, const ReduceOrderArena& arena, const std::set<uint32_t>& devices,
            HcclDataType dtype, HcclReduceOp op)
        {
            TreeResult result;
            std::vector<ReduceOrderExprId> pending{root};
            std::set<uint32_t> leaves;
            uint64_t visited = 0;
            const uint64_t maxNodes = static_cast<uint64_t>(devices.size()) * 2U - 1U;
            while (!pending.empty()) {
                if (++visited > maxNodes) {
                    RecordFailure(result, ReduceOrderStatus::INCOMPLETE, "tree contains too many operands");
                    break;
                }
                const auto id = pending.back();
                pending.pop_back();
                const auto* node = arena.Get(id);
                if (node == nullptr) {
                    RecordFailure(result, ReduceOrderStatus::INCOMPLETE, "tree contains an unknown reduction order");
                    continue;
                }
                if (node->kind == ReduceOrderExpr::Kind::LEAF) {
                    if (devices.count(node->deviceId) == 0 || !leaves.insert(node->deviceId).second) {
                        RecordFailure(
                            result, ReduceOrderStatus::INCOMPLETE,
                            "tree contains a foreign or duplicate source deviceId");
                    }
                } else if (node->kind == ReduceOrderExpr::Kind::REDUCE) {
                    if (node->op != op || node->dataType != dtype) {
                        RecordFailure(
                            result, ReduceOrderStatus::MISMATCH, "a reduce node has an unexpected op or dtype");
                    }
                    pending.push_back(node->src);
                    pending.push_back(node->dst);
                } else {
                    RecordFailure(result, ReduceOrderStatus::INCOMPLETE, "tree contains an invalid node kind");
                }
            }
            if (leaves != devices) {
                RecordFailure(result, ReduceOrderStatus::INCOMPLETE, "tree does not cover all source deviceIds");
            }
            return result;
        }

        std::string ByteValue(uint64_t elements, uint64_t width)
        {
            if (width == 0) {
                return "UNAVAILABLE";
            }
            if (elements > std::numeric_limits<uint64_t>::max() / width) {
                return "OVERFLOW";
            }
            return std::to_string(elements * width);
        }

        void LogBlock(
            const ReduceOrderOutputSpan& span, const char* consistency, const ReduceOrderArena& arena,
            HcclDataType dtype)
        {
            const uint64_t width = dtype < HCCL_DATA_TYPE_RESERVED ? CHECK_SIZE_TABLE[dtype] : 0;
            // 每个极大块一行：区间定位 + 判定结论 + 求值顺序（叶子设备号序列）。
            HCCL_VM_INFO(
                "ReduceOrder output, deviceId={}, elements=[{},{}), byteOffset={}, byteCount={}, "
                "consistent={}, order={}",
                span.deviceId, span.begin, span.begin + span.count, ByteValue(span.begin, width),
                ByteValue(span.count, width), consistency, arena.DescribeLeafOrder(span.expr));
        }
    } // namespace

    const char* DescribeReduceOrderStatus(ReduceOrderStatus status)
    {
        switch (status) {
            case ReduceOrderStatus::DISABLED:
                return "DISABLED";
            case ReduceOrderStatus::PASS:
                return "PASS";
            case ReduceOrderStatus::MISMATCH:
                return "MISMATCH";
            case ReduceOrderStatus::INCOMPLETE:
                return "INCOMPLETE";
        }
        return "INVALID";
    }

    HcclResult CheckOutputReduceOrder(
        const std::vector<uint32_t>& devices, HcclDataType dataType, HcclReduceOp reduceOp,
        const ReduceOrderArena& arena, std::vector<ReduceOrderOutputSpan>& spans, ReduceOrderReport& report)
    {
        report = ReduceOrderReport{};
        report.expressionNodes = arena.NodeCount();

        // 把相邻且同树的区间合并成极大块：既减少比较次数，也让日志按块输出。
        // spans 的有序性与完整覆盖由调用方保证（见头文件注释）。
        std::vector<ReduceOrderOutputSpan> merged;
        for (const auto& span : spans) {
            if (!merged.empty() && merged.back().deviceId == span.deviceId && merged.back().expr == span.expr
                && merged.back().begin + merged.back().count == span.begin) {
                merged.back().count += span.count;
            } else {
                merged.push_back(span);
            }
        }
        spans = std::move(merged);
        report.outputSpans = spans.size();

        // 每棵唯一的树只校验一次形状；参考树取第一个合法块的表达式。
        // spans 顺序确定（设备序 + 元素偏移），因此参考树是确定的。
        const std::set<uint32_t> participants(devices.begin(), devices.end());
        std::map<ReduceOrderExprId, TreeResult> trees;
        ReduceOrderExprId reference = INVALID_REDUCE_ORDER_EXPR;
        for (const auto& span : spans) {
            auto found = trees.find(span.expr);
            if (found == trees.end()) {
                found
                    = trees.emplace(span.expr, ValidateTree(span.expr, arena, participants, dataType, reduceOp)).first;
            }
            if (found->second.status == ReduceOrderStatus::PASS && reference == INVALID_REDUCE_ORDER_EXPR) {
                reference = span.expr;
            }
        }

        // 逐块判定：合法块与参考树比较（ID 相等即结构相等），非法块直接记失败。
        // 不在首个失败处提前退出：每个极大块都要诊断并落日志。
        TreeResult failure;
        for (const auto& span : spans) {
            const auto& tree = trees.at(span.expr);
            const bool valid = tree.status == ReduceOrderStatus::PASS;
            const char* consistency = "INVALID";
            if (valid) {
                ++report.comparedSpans;
                consistency = span.expr == reference ? "YES" : "NO";
                if (span.expr != reference) {
                    RecordFailure(
                        failure, ReduceOrderStatus::MISMATCH,
                        "deviceId=" + std::to_string(span.deviceId) + ", outputElement=" + std::to_string(span.begin)
                            + ", expected=expr#" + std::to_string(reference) + arena.DescribeLeafOrder(reference)
                            + ", actual=expr#" + std::to_string(span.expr) + arena.DescribeLeafOrder(span.expr));
                }
            } else {
                RecordFailure(
                    failure, tree.status,
                    "deviceId=" + std::to_string(span.deviceId) + ", outputElement=" + std::to_string(span.begin) + ": "
                        + tree.detail);
            }
            LogBlock(span, consistency, arena, dataType);
        }
        if (failure.status != ReduceOrderStatus::PASS) {
            report.status = failure.status;
            report.detail = failure.detail;
            HCCL_VM_ERROR(
                "ReduceOrder output check failed, status={}, detail={}", DescribeReduceOrderStatus(report.status),
                report.detail);
            return HCCL_E_PARA;
        }
        report.status = ReduceOrderStatus::PASS;
        report.detail = spans.empty() ? "empty output; no reductions to compare" :
                                        "all output role trees agree; reference=" + arena.DescribeLeafOrder(reference)
                                            + "; numerical bitwise equality is not verified";
        return HCCL_SUCCESS;
    }
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
