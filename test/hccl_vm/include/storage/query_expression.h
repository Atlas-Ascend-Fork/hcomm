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
 * @file query_expression.h
 * @brief 后端无关的类型安全数据库表达式：条件、排序、数量限制与更新赋值。
 *
 * 【调用流程】
 *   业务代码 ──Eq/Ne/Lt/Le/Gt/Ge/In/StartsWith/And/Or──> Condition<T>（唯一 ConditionData）
 *        │                                                  │ OrderBy/Limit ──> QueryOptions
 *        │                                                  │ Set/Increment ──> LogicalChange
 *        ▼ Table<T>::Select/Count/Update/Delete
 *   引擎直接消费同一 ConditionData/UpdateData（const 引用传递，无第二份 DTO）
 *
 * 【设计约束】
 * 1. 表达式全部为短生命周期值对象：无虚函数层次、无共享所有权图；条件在
 *    数据库端编译执行，绝不退化为读全量后内存过滤。
 * 2. 成员指针（&T::member）到字段编号的映射由 RecordSchema<T>::FieldIdOf 提供，
 *    业务源文件不出现手写字符串字段名或数字编号。
 * 3. 值只在公开入口构造最终 ConditionData/UpdateData 时拥有化一次：单谓词内联
 *    一个子句，不分配子句 vector；And/Or 先计算最终子句数并精确 reserve，
 *    再 move/splice 子条件，不复制 DbScalar。例外（拥有式接口的语义要求，
 *    不是实现可删的复制）：以左值 Condition/Assignment 调用 And/Or/Update 时，
 *    调用方保留原对象，入口按独立所有权复制一份载荷；右值临时链全程 move。
 * 4. 条件值按成员类别编码为 DbScalar：无符号整数成员收整型/枚举值，有符号
 *    成员收整型/枚举值，char 数组成员收字符串值（NUL 前字节承载语义），
 *    uint8_t 数组成员收字节序列。
 */
#ifndef HCCL_VM_STORAGE_QUERY_EXPRESSION_H
#define HCCL_VM_STORAGE_QUERY_EXPRESSION_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "storage/query_spec.h"

namespace HcclSim::Storage {
/// RecordSchema 前置声明：表达式构造时按特化解析成员指针（见 record_contract.h）
template <typename T>
struct RecordSchema;

/// 公开排序方向（业务表达层；内部使用 Order）
enum class SortDirection : uint8_t { ASC, DESC };

/// 排序子句（由 OrderBy 生成）
template <typename T>
struct SortClause {
    uint16_t fieldId{0};
    SortDirection direction{SortDirection::ASC};
};

/// 数量限制子句（由 Limit 生成）
struct LimitClause {
    uint64_t limit{0};
};

/// 字段投影子句（由 Fields 生成）：Select 行读取只取列出的字段
struct ProjectionClause {
    std::vector<uint16_t> fieldIds;
};

/// 更新赋值子句（由 Set/Increment 生成的公开参数对象；Table 收集时直接转换
/// 为唯一 UpdateData 元素，不建中间容器）
template <typename T>
struct Assignment {
    uint16_t fieldId{0};
    bool increment{false};
    DbScalar value{uint64_t{0}};
};

/**
 * @brief 类型化查询条件（唯一 ConditionData 持有者）。
 * @tparam T 目标表记录类型
 *
 * 无条件（All）为 monostate；单谓词内联一个子句（不分配 vector）；
 * 组合条件拥有一个最终子句 vector，括号语义由子句的 openGroupsBefore/
 * closeGroupsAfter 与 joinWithPrevious 承载，不构造条件树。
 */
template <typename T>
struct Condition {
    ConditionData data{};
};

namespace expression_detail {
    /// 成员类别：决定值到数据库标量的编码规则
    enum class MemberCategory : uint8_t {
        UNSIGNED_INTEGER, ///< 无符号整数成员 → uint64_t
        SIGNED_INTEGER,   ///< 有符号整数成员 → int64_t
        BOOLEAN,          ///< bool 成员 → bool
        REAL,             ///< 浮点成员 → double
        TEXT,             ///< 字符串成员（char 数组/std::string）→ string
        BYTES,            ///< 字节序列成员（uint8_t 数组/vector）→ 字节
        TEXT_OR_BYTES,    ///< 字节数组成员：TXT/BLB 语义由 schema 声明决定，按值形态编码
        UNSUPPORTED       ///< 其余成员类型不允许出现在表达式中
    };

    /// 按成员类型归类（char 数组先于字节数组判定，避免 char[] 落入字节分支）
    template <typename M>
    constexpr MemberCategory CategoryOfMember()
    {
        if constexpr (std::is_same_v<M, bool>) {
            return MemberCategory::BOOLEAN;
        } else if constexpr (std::is_integral_v<M> && std::is_unsigned_v<M>) {
            return MemberCategory::UNSIGNED_INTEGER;
        } else if constexpr (std::is_integral_v<M>) {
            return MemberCategory::SIGNED_INTEGER;
        } else if constexpr (std::is_floating_point_v<M>) {
            return MemberCategory::REAL;
        } else if constexpr (std::is_same_v<M, std::string>) {
            return MemberCategory::TEXT;
        } else if constexpr (std::is_array_v<M> && std::is_same_v<std::remove_extent_t<M>, char>) {
            return MemberCategory::TEXT;
        } else if constexpr (std::is_same_v<M, std::vector<uint8_t>>) {
            return MemberCategory::BYTES;
        } else if constexpr (std::is_array_v<M> && std::is_same_v<std::remove_all_extents_t<M>, uint8_t>) {
            // uint8_t 数组的 TXT/BLB 归属由 schema 声明决定（如 IpcNotify::name_or_key 声明为 TXT），
            // 表达式按值形态编码：字符串值 → 文本语义，字节序列值 → BLOB 语义；
            // 列级校验（TEXT 列拒绝字节值 / PREFIX 要求 TEXT 列）在引擎层运行期把关。
            return MemberCategory::TEXT_OR_BYTES;
        } else {
            return MemberCategory::UNSUPPORTED;
        }
    }

    /// 按成员类别把业务值编码为数据库标量；类别与值不兼容在编译期拒绝
    template <MemberCategory category, typename V>
    DbScalar EncodeMemberValue(const V& value)
    {
        if constexpr (category == MemberCategory::UNSIGNED_INTEGER) {
            static_assert(
                std::is_integral_v<V> || std::is_enum_v<V>, "unsigned integer members accept integral or enum values");
            return DbScalar{static_cast<uint64_t>(value)};
        } else if constexpr (category == MemberCategory::SIGNED_INTEGER) {
            static_assert(
                std::is_integral_v<V> || std::is_enum_v<V>, "signed integer members accept integral or enum values");
            return DbScalar{static_cast<int64_t>(value)};
        } else if constexpr (category == MemberCategory::BOOLEAN) {
            static_assert(std::is_convertible_v<V, bool>, "boolean members accept boolean values");
            return DbScalar{static_cast<bool>(value)};
        } else if constexpr (category == MemberCategory::REAL) {
            static_assert(std::is_arithmetic_v<V>, "real members accept numeric values");
            return DbScalar{static_cast<double>(value)};
        } else if constexpr (category == MemberCategory::TEXT) {
            static_assert(std::is_convertible_v<const V&, std::string_view>, "text members accept string values");
            return DbScalar{std::string(std::string_view(value))};
        } else if constexpr (category == MemberCategory::BYTES) {
            static_assert(
                std::is_constructible_v<std::vector<uint8_t>, const V&>, "bytes members accept byte sequences");
            return DbScalar{std::vector<uint8_t>(value)};
        } else if constexpr (category == MemberCategory::TEXT_OR_BYTES) {
            if constexpr (std::is_convertible_v<const V&, std::string_view>) {
                return DbScalar{std::string(std::string_view(value))};
            } else {
                static_assert(
                    std::is_constructible_v<std::vector<uint8_t>, const V&>,
                    "byte array members accept string or byte sequence values");
                return DbScalar{std::vector<uint8_t>(value)};
            }
        } else {
            static_assert(category != MemberCategory::UNSUPPORTED, "member type is not a supported database field");
            return DbScalar{uint64_t{0}};
        }
    }

    /// 成员指针到逻辑字段编号（经 RecordSchema 合同解析）
    template <typename T, typename M>
    uint16_t FieldIdOfMember(M T::*member)
    {
        return RecordSchema<T>::template FieldIdOf<M>(member);
    }

    /// 增量专用编码：**仅无符号字段**的整型增量按值自身符号类别编码、保留负号。
    /// 【缺陷修复】无符号字段收有符号负增量（如 Increment(&Context::ref_cnt,
    /// int64_t{-1})）此前经 EncodeMemberValue 的无符号类别 static_cast<uint64_t>
    /// 折叠为 UINT64_MAX，MariaDB 生成 field + 18446744073709551615 并以
    /// errno 1690 拒绝合法减法。修复后负增量以 int64 语义到达各后端，由后端以
    /// field - magnitude 的单语句原子形式完成。有符号值 → int64（符号保真）；
    /// 无符号值 → uint64（构造上非负）。有符号/REAL 字段沿用 EncodeMemberValue
    /// 既有编码（正增量与 Set/过滤条件的类型规则不变）。
    template <typename M, typename V>
    DbScalar EncodeIncrementValue(const V& value)
    {
        constexpr auto category = CategoryOfMember<M>();
        static_assert(
            category == MemberCategory::UNSIGNED_INTEGER || category == MemberCategory::SIGNED_INTEGER
                || category == MemberCategory::REAL,
            "increment requires a numeric field");
        if constexpr (category != MemberCategory::UNSIGNED_INTEGER) {
            return EncodeMemberValue<category, V>(value);
        } else if constexpr (std::is_enum_v<V>) {
            using Underlying = std::underlying_type_t<V>;
            return EncodeIncrementValue<M>(static_cast<Underlying>(value));
        } else if constexpr (std::is_integral_v<V> && std::is_signed_v<V>) {
            return DbScalar{static_cast<int64_t>(value)};
        } else if constexpr (std::is_integral_v<V>) {
            return DbScalar{static_cast<uint64_t>(value)};
        } else {
            static_assert(!sizeof(V), "increment accepts integral or enum values on numeric fields");
            return DbScalar{uint64_t{0}};
        }
    }

    /// 平铺子句数量（无条件=0、单谓词=1、组合=vector 大小）
    inline std::size_t ClauseCountOf(const ConditionData& data)
    {
        if (const auto* clauses = std::get_if<std::vector<ConditionClause>>(&data); clauses != nullptr) {
            return clauses->size();
        }
        return std::holds_alternative<std::monostate>(data) ? 0U : 1U;
    }

    /// 把一个子条件并入平铺结果：All 消解跳过；单谓词直接并入；多子句组合整体
    /// 加括号并入（括号计数累加，嵌套语义原样保留）。不复制任何 DbScalar。
    template <typename T>
    void AppendConditionChild(std::vector<ConditionClause>& out, ConditionJoin join, Condition<T>&& child)
    {
        if (std::holds_alternative<std::monostate>(child.data)) {
            return; // 嵌套 All 子条件消解
        }
        const bool first = out.empty();
        if (auto* clause = std::get_if<ConditionClause>(&child.data); clause != nullptr) {
            clause->joinWithPrevious = first ? ConditionJoin::NONE : join;
            out.push_back(std::move(*clause));
            return;
        }
        auto& clauses = std::get<std::vector<ConditionClause>>(child.data);
        if (clauses.empty()) {
            return;
        }
        if (clauses.size() == 1U) {
            // 单子句组合不保留括号（与单子组合展平语义一致）
            clauses.front().joinWithPrevious = first ? ConditionJoin::NONE : join;
            out.push_back(std::move(clauses.front()));
            return;
        }
        clauses.front().joinWithPrevious = first ? ConditionJoin::NONE : join;
        clauses.front().openGroupsBefore += 1U;
        clauses.back().closeGroupsAfter += 1U;
        out.insert(out.end(), std::make_move_iterator(clauses.begin()), std::make_move_iterator(clauses.end()));
    }

    /// 组合条件构造：先计算最终子句数，再精确 reserve 并 move/splice。
    /// 全部子条件为 All 时结果为 All；仅一条存活子句时内联单谓词形态。
    template <typename T, ConditionJoin Join, typename... Children>
    Condition<T> MakeConditionGroup(Children&&... children)
    {
        const std::size_t total = (ClauseCountOf(children.data) + ... + 0U);
        Condition<T> group;
        if (total == 0U) {
            return group;
        }
        if (total == 1U) {
            Condition<T>* const candidates[] = {&children...};
            for (Condition<T>* child : candidates) {
                if (auto* clause = std::get_if<ConditionClause>(&child->data); clause != nullptr) {
                    group.data = std::move(*clause);
                    return group;
                }
                if (auto* clauses = std::get_if<std::vector<ConditionClause>>(&child->data);
                    clauses != nullptr && clauses->size() == 1U) {
                    group.data = std::move(clauses->front());
                    return group;
                }
            }
            return group; // 不可达：total==1 时必存在单一存活子句
        }
        std::vector<ConditionClause> flat;
        flat.reserve(total);
        (AppendConditionChild(flat, Join, std::move(children)), ...);
        group.data = std::move(flat);
        return group;
    }
} // namespace expression_detail

// ---- 条件工厂：单个成员指针 + 值 ----

/// 等值条件
template <typename T, typename M, typename V>
Condition<T> Eq(M T::*member, const V& value)
{
    Condition<T> condition;
    condition.data = ConditionClause{
        expression_detail::FieldIdOfMember(member),
        Compare::EQUAL,
        ConditionJoin::NONE,
        0,
        0,
        ConditionValue{expression_detail::EncodeMemberValue<expression_detail::CategoryOfMember<M>()>(value)}};
    return condition;
}

/// 不等条件
template <typename T, typename M, typename V>
Condition<T> Ne(M T::*member, const V& value)
{
    Condition<T> condition;
    condition.data = ConditionClause{
        expression_detail::FieldIdOfMember(member),
        Compare::NOT_EQUAL,
        ConditionJoin::NONE,
        0,
        0,
        ConditionValue{expression_detail::EncodeMemberValue<expression_detail::CategoryOfMember<M>()>(value)}};
    return condition;
}

/// 小于条件
template <typename T, typename M, typename V>
Condition<T> Lt(M T::*member, const V& value)
{
    Condition<T> condition;
    condition.data = ConditionClause{
        expression_detail::FieldIdOfMember(member),
        Compare::LESS,
        ConditionJoin::NONE,
        0,
        0,
        ConditionValue{expression_detail::EncodeMemberValue<expression_detail::CategoryOfMember<M>()>(value)}};
    return condition;
}

/// 小于等于条件
template <typename T, typename M, typename V>
Condition<T> Le(M T::*member, const V& value)
{
    Condition<T> condition;
    condition.data = ConditionClause{
        expression_detail::FieldIdOfMember(member),
        Compare::LESS_OR_EQUAL,
        ConditionJoin::NONE,
        0,
        0,
        ConditionValue{expression_detail::EncodeMemberValue<expression_detail::CategoryOfMember<M>()>(value)}};
    return condition;
}

/// 大于条件
template <typename T, typename M, typename V>
Condition<T> Gt(M T::*member, const V& value)
{
    Condition<T> condition;
    condition.data = ConditionClause{
        expression_detail::FieldIdOfMember(member),
        Compare::GREATER,
        ConditionJoin::NONE,
        0,
        0,
        ConditionValue{expression_detail::EncodeMemberValue<expression_detail::CategoryOfMember<M>()>(value)}};
    return condition;
}

/// 大于等于条件
template <typename T, typename M, typename V>
Condition<T> Ge(M T::*member, const V& value)
{
    Condition<T> condition;
    condition.data = ConditionClause{
        expression_detail::FieldIdOfMember(member),
        Compare::GREATER_OR_EQUAL,
        ConditionJoin::NONE,
        0,
        0,
        ConditionValue{expression_detail::EncodeMemberValue<expression_detail::CategoryOfMember<M>()>(value)}};
    return condition;
}

/// 集合包含条件（IN）：值集合为空时条件恒假（匹配 0 行），不生成非法 SQL。
/// 输入集合按 const 引用直读（左值集合不再经按值参数整集复制），载荷仅在
/// 构造最终 vector<DbScalar> 时逐项拥有化一次；右值临时集合调用语法不变。
template <typename T, typename M, typename V>
Condition<T> In(M T::*member, const std::vector<V>& values)
{
    Condition<T> condition;
    ConditionClause clause;
    clause.fieldId = expression_detail::FieldIdOfMember(member);
    clause.compare = Compare::IN;
    std::vector<DbScalar> encoded;
    encoded.reserve(values.size());
    for (const auto& value : values) {
        encoded.push_back(expression_detail::EncodeMemberValue<expression_detail::CategoryOfMember<M>(), V>(value));
    }
    clause.value = ConditionValue{std::move(encoded)};
    condition.data = std::move(clause);
    return condition;
}

/// 前缀匹配条件（StartsWith）：文本语义字段可用；字节数组成员按 schema 声明
/// 由引擎层运行期校验（PREFIX 要求 TEXT 列）
template <typename T, typename M, typename V>
Condition<T> StartsWith(M T::*member, const V& value)
{
    static_assert(
        expression_detail::CategoryOfMember<M>() == expression_detail::MemberCategory::TEXT
            || expression_detail::CategoryOfMember<M>() == expression_detail::MemberCategory::TEXT_OR_BYTES,
        "prefix comparison requires a text field");
    Condition<T> condition;
    condition.data = ConditionClause{
        expression_detail::FieldIdOfMember(member),
        Compare::PREFIX,
        ConditionJoin::NONE,
        0,
        0,
        ConditionValue{expression_detail::EncodeMemberValue<expression_detail::CategoryOfMember<M>()>(value)}};
    return condition;
}

/// 匹配全部行（All）：仅用于 Select/Count；Update/Delete 的无条件形态在合同层被拒绝
template <typename T>
Condition<T> All()
{
    return Condition<T>{};
}

/// AND 组合（至少两个子条件；嵌套组合带括号语义）
template <typename T, typename... Rest>
Condition<T> And(Condition<T> first, Condition<T> second, Rest... rest)
{
    return expression_detail::MakeConditionGroup<T, ConditionJoin::AND>(
        std::move(first), std::move(second), std::move(rest)...);
}

/// OR 组合（至少两个子条件；嵌套组合带括号语义）
template <typename T, typename... Rest>
Condition<T> Or(Condition<T> first, Condition<T> second, Rest... rest)
{
    return expression_detail::MakeConditionGroup<T, ConditionJoin::OR>(
        std::move(first), std::move(second), std::move(rest)...);
}

// ---- 查询子句工厂 ----

/// 排序子句（direction 缺省为 ASC；只有调用 OrderBy 才启用排序）
template <typename T, typename M>
SortClause<T> OrderBy(M T::*member, SortDirection direction = SortDirection::ASC)
{
    return SortClause<T>{expression_detail::FieldIdOfMember(member), direction};
}

/// 数量限制子句
inline LimitClause Limit(uint64_t limit) { return LimitClause{limit}; }

/// 字段投影子句：只收窄行的读取范围（未列字段物化为默认值）。条件与排序
/// 引用的字段不要求被投影包含（过滤/排序由数据库端独立完成）。与
/// OrderBy/Limit 同级使用；Select/SelectOne 均可携带（SelectOne 固定
/// limit=1，投影只影响列集合）
template <typename T, typename... Members>
ProjectionClause Fields(Members T::*... members)
{
    static_assert(sizeof...(members) > 0U, "projection requires at least one field");
    return ProjectionClause{std::vector<uint16_t>{expression_detail::FieldIdOfMember(members)...}};
}

// ---- 更新赋值工厂 ----

/// 字段赋值变更（公开参数对象；Table 收集时直接构造最终 UpdateData 元素）
template <typename T, typename M, typename V>
Assignment<T> Set(M T::*member, const V& value)
{
    return Assignment<T>{
        expression_detail::FieldIdOfMember(member), false,
        expression_detail::EncodeMemberValue<expression_detail::CategoryOfMember<M>(), V>(value)};
}

/// 字段原子增量变更（数据库单语句完成，非读后写）。增量值经
/// EncodeIncrementValue 按自身符号类别编码：无符号字段收负增量不再
/// 折叠为 UINT64_MAX（errno 1690 根因），合法减法由各后端单语句完成。
template <typename T, typename M, typename V>
Assignment<T> Increment(M T::*member, const V& value)
{
    static_assert(
        expression_detail::CategoryOfMember<M>() == expression_detail::MemberCategory::UNSIGNED_INTEGER
            || expression_detail::CategoryOfMember<M>() == expression_detail::MemberCategory::SIGNED_INTEGER
            || expression_detail::CategoryOfMember<M>() == expression_detail::MemberCategory::REAL,
        "increment requires a numeric field");
    return Assignment<T>{
        expression_detail::FieldIdOfMember(member), true, expression_detail::EncodeIncrementValue<M, V>(value)};
}
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_QUERY_EXPRESSION_H
