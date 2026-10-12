/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * See the License in the specific file packaged with this Software for details.
 */

/**
 * @file query_spec.h
 * @brief 存储抽象层的基础值类型与唯一条件/更新数据形态。
 *
 * 【设计边界】
 * 本头文件只保留后端无关的"值词汇"与条件/更新的唯一拥有形态：
 * Compare/ConditionJoin/ChangeKind 为连续枚举（后端以 constexpr 数组按
 * 下标解释，不保存查询历史）；ConditionClause 平铺承载括号语义，
 * ConditionData 是公开入口构造的唯一条件载荷；UpdateData 是唯一更新
 * 载荷。构造完成后中间层只传 const 引用，后端直接消费同一对象，
 * 不再归一化成第二份条件/更新 DTO。
 *
 * 【时间/空间影响】
 * DbScalar 为 std::variant 值对象；string/blob 载荷只在公开入口
 * （Eq/Ne/.../In/Set/Increment）构造最终 ConditionData/UpdateData 时
 * 拥有化一次，组合条件只 move/splice，不复制 DbScalar。
 */
#ifndef HCCL_VM_STORAGE_QUERY_SPEC_H
#define HCCL_VM_STORAGE_QUERY_SPEC_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace HcclSim::Storage {
/// 事务内同表批量插入的单批行数上限（T1）：有界批量——超限由调用方按固定
/// 窗口拆分调用，引擎不积攒无界批次
inline constexpr std::size_t kInsertBatchMaxRows = 256;

/// 类型化主键包装器
template <typename T>
struct PrimaryKey {
    uint64_t value{0};
};

/// 数据库标量值通用容器
using DbScalar = std::variant<int64_t, uint64_t, double, bool, std::string, std::vector<uint8_t>>;

/// 比较运算符（连续枚举；COUNT 为数组长度哨兵，不是比较方式）
enum class Compare : uint8_t { EQUAL, NOT_EQUAL, LESS, LESS_OR_EQUAL, GREATER, GREATER_OR_EQUAL, PREFIX, IN, COUNT };

/// 条件子句与前一子句的连接方式（连续枚举；COUNT 为数组长度哨兵）
enum class ConditionJoin : uint8_t { NONE, AND, OR, COUNT };

/// 更新变更方式（连续枚举；COUNT 为数组长度哨兵）
enum class ChangeKind : uint8_t { SET, INCREMENT, COUNT };

/// 内部排序方向（逻辑 IR 使用；业务表达层使用 query_expression.h 的 SortDirection）
enum class Order : uint8_t { ASCENDING, DESCENDING };

/// 单表清理模式
enum class ClearMode : uint8_t {
    KEEP_SEQUENCE,     ///< 清空数据并保留自增序列号
    RESET_SEQUENCE,    ///< 清空数据并重置序列号为 0
    DROP_DYNAMIC_SCOPE ///< 销毁 PID 动态表结构
};

/// 未映射成员的无效字段编号（成员指针查表失败的返回值）
static constexpr uint16_t kUnknownFieldId = 0xFFFF;

/// 条件值：单值（内联 DbScalar）或 IN 多值（vector）；空 vector 表示恒假 IN
using ConditionValue = std::variant<DbScalar, std::vector<DbScalar>>;

/// 平铺条件子句：字段 + 比较方式 + 与前一子句的连接 + 括号边界 + 值。
/// 括号语义由 openGroupsBefore/closeGroupsAfter 承载（组合条件的子组整体
/// 加括号），不构造条件树和独立操作符 token。
struct ConditionClause {
    uint16_t fieldId{kUnknownFieldId};
    Compare compare{Compare::EQUAL};
    ConditionJoin joinWithPrevious{ConditionJoin::NONE};
    uint16_t openGroupsBefore{0};
    uint16_t closeGroupsAfter{0};
    ConditionValue value{DbScalar{uint64_t{0}}};
};

/// 唯一条件载荷：无条件（All）/单谓词（内联一个子句，不分配 vector）/
/// 组合（最终子句 vector）。公开入口构造后全程只读传递。
using ConditionData = std::variant<std::monostate, ConditionClause, std::vector<ConditionClause>>;

/// 单个更新变更项：字段 + 变更方式 + 值（SET 新值或 INCREMENT 增量）
struct LogicalChange {
    uint16_t fieldId{kUnknownFieldId};
    ChangeKind kind{ChangeKind::SET};
    DbScalar value{uint64_t{0}};
};

/// 唯一更新载荷（按调用方参数顺序；fieldId 决定列，kind 决定 SET/原子加）
using UpdateData = std::vector<LogicalChange>;

/// 查询选项：排序键（按传入顺序的复合排序）与数量限制
struct QueryOptions {
    std::vector<std::pair<uint16_t, Order>> sorts;
    std::optional<uint64_t> limit;
    /// 字段投影（空 = 全行读取）。非空时行读取只取列出的字段编号集合，
    /// 未列字段物化为记录默认值；条件/排序引用的字段不要求被投影包含
    /// （过滤/排序由数据库端独立完成，其合法性由各后端的条件/排序渲染
    /// 各自校验）。用于只需要少数字段的查询避免整行大 BLOB 回传（如
    /// opDetails 执行索引只取 6 个小字段）。
    std::vector<uint16_t> projectedFields;
};

/// 条件计数聚合结果：total 为 where 集合行数，matched 为其中再满足 match
/// 的行数。两值来自同一条聚合语句（引擎单语句实现），不存在两次读取间的
/// 并发漂移；空集合为 OK + {0, 0}，matched <= total。
struct CountMatchResult {
    uint64_t total{0};
    uint64_t matched{0};
};

/**
 * @brief 表作用域说明（用于区分静态公共表与 PID 独占表）。
 */
struct TableScope {
    std::optional<uint64_t> processId; ///< 若有值表示该进程专属的动态 PID 表

    static TableScope Static() { return {}; }
    static TableScope Process(uint64_t pid) { return {pid}; }

    bool operator==(const TableScope& other) const noexcept { return processId == other.processId; }
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_QUERY_SPEC_H
