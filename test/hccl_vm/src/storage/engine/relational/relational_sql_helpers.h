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
 * See the License in the root of the software repository for the full text of the License.
 */

// 文件职责：各 SQL 后端引擎共用的关系处理辅助
// ——后端无关的遍历、校验与投影解析；单子句 SQL 方言、原生绑定与标识符引用
// 留在各后端。全部函数无状态：直接消费 ConditionData/UpdateData/CatalogTableMeta
// 与字段引用，把文本追加进调用方的同一份最终 SQL 字符串，不建立第二套
// 值列表、查询 AST 或中间行表示，也不自行创建表、获取连接或重试。
// 共享遍历不得改变单值/IN/空 IN、括号、占位符计数与参数绑定顺序。
#ifndef HCCL_VM_STORAGE_ENGINE_RELATIONAL_RELATIONAL_SQL_HELPERS_H
#define HCCL_VM_STORAGE_ENGINE_RELATIONAL_RELATIONAL_SQL_HELPERS_H

#include <cstdint>
#include <string>
#include <vector>

#include "storage/database_result.h"
#include "storage/query_spec.h"
#include "storage/storage_catalog.h"

namespace HcclSim::Storage {
namespace Relational {

    /// PREFIX 条件的 LIKE 模式：转义 %/_/\ 并追加前缀通配符（绑定值，不进语句
    /// 文本）。SQL 中 ESCAPE 的写法及绑定所有权由各后端生成。
    std::string EscapePrefixPattern(const std::string& value);

    /// 检查逻辑值类型与目录列类型一致（INTEGER 列接受 int64/uint64/bool，
    /// REAL 列拒绝 string/bytes，STRING/BYTES 列要求同型载荷）。不统一服务器
    /// 数值表示或错误码——服务器侧域语义留在各后端。
    DbResult<void> CheckColumnValue(const Ddl::FieldDesc& field, const DbScalar& value);

    /// 动态 scope 物理表名 <table>_P_<pid>；静态表直用逻辑表名。缺 process
    /// scope 的动态表明确失败。原生标识符长度策略不在此合并。
    DbResult<std::string> PhysicalTableNameOf(const Ddl::CatalogTableMeta& meta, const TableScope& scope);

    /// 后端方言的标识符追加函数（SQLite 双引号 / MariaDB 反引号；追加进调用方
    /// 的同一份最终 SQL 文本）
    using AppendIdentifierFn = DbResult<void> (*)(std::string&, std::string_view);

    /// INSERT 的列清单与占位符清单：按目录声明顺序把 " (c1, c2, ...) VALUES
    /// (?, ?, ...)" 追加进最终 SQL。自增零值处理、生成主键与执行留在后端。
    DbResult<void> AppendInsertColumnListAndValues(
        std::string& sql, const Ddl::CatalogTableMeta& meta, AppendIdentifierFn appendIdentifier);

    /// SELECT 投影解析（小型、无状态）。空投影 = 全列：fieldIndices 覆盖全部
    /// 字段（列序 = 声明序 = fieldId），columnOfField 保持空——空映射即恒等
    /// 映射的约定，SELECT 列清单、结果列绑定与 ReadField 必须一致理解，不能
    /// 把空映射当零列。非空投影校验字段编号合法并按给定顺序去重，
    /// columnOfField[fieldId] = 输出列下标（-1 = 未投影，物化保持记录默认值）。
    /// WHERE/ORDER BY 引用字段不要求被投影包含（过滤/排序由数据库端独立
    /// 完成），其合法性由各后端的条件/排序渲染各自校验。
    DbResult<void> ResolveProjection(
        const Ddl::CatalogTableMeta& meta, const QueryOptions& options, std::vector<std::size_t>& fieldIndices,
        std::vector<int32_t>& columnOfField);

    /// 条件文本遍历（第一遍）：平铺子句按序渲染进调用方的最终 SQL（output），
    /// 连接词先于括号，括号语义由 openGroupsBefore/closeGroupsAfter 承载，与
    /// 占位符/绑定值顺序严格一致——本遍历与 BindWhereValues 按完全相同的顺序
    /// 行走。单子句 SQL 由后端方言回调 appendClause(clause) 生成（回调向同一份
    /// output 追加；uint64 高半区展开、LIKE ESCAPE 写法、错误返回时序等差异
    /// 全部留在回调内）。
    template <typename AppendClauseFn>
    DbResult<void> AppendWhereText(const ConditionData& where, std::string& output, AppendClauseFn&& appendClause)
    {
        if (const auto* clauses = std::get_if<std::vector<ConditionClause>>(&where); clauses != nullptr) {
            for (std::size_t index = 0; index < clauses->size(); ++index) {
                const ConditionClause& clause = (*clauses)[index];
                if (index > 0) {
                    output += clause.joinWithPrevious == ConditionJoin::OR ? " OR " : " AND ";
                }
                for (std::uint16_t open = 0; open < clause.openGroupsBefore; ++open) {
                    output += "(";
                }
                const auto appended = appendClause(clause);
                if (!appended.ok()) {
                    return appended;
                }
                for (std::uint16_t close = 0; close < clause.closeGroupsAfter; ++close) {
                    output += ")";
                }
            }
            return {DbCode::OK, 0, {}};
        }
        if (const auto* clause = std::get_if<ConditionClause>(&where); clause != nullptr) {
            return appendClause(*clause);
        }
        return {DbCode::OK, 0, {}};
    }

    /// 条件绑定遍历（第二遍）：按与 AppendWhereText 完全相同的顺序逐子句调用
    /// 后端方言回调 bindClause(clause)（原生绑定调用与临时值生命周期留在回调内；
    /// 空不产生绑定值）。
    template <typename BindClauseFn>
    DbResult<void> BindWhereValues(const ConditionData& where, BindClauseFn&& bindClause)
    {
        if (const auto* clauses = std::get_if<std::vector<ConditionClause>>(&where); clauses != nullptr) {
            for (const auto& clause : *clauses) {
                const auto bound = bindClause(clause);
                if (!bound.ok()) {
                    return bound;
                }
            }
            return {DbCode::OK, 0, {}};
        }
        if (const auto* clause = std::get_if<ConditionClause>(&where); clause != nullptr) {
            return bindClause(*clause);
        }
        return {DbCode::OK, 0, {}};
    }

    /// CountMatches 聚合语句的公共 SQL 结构（只写一份，两关系后端复用）：
    ///   SELECT COUNT(*), COUNT(CASE WHEN (<match>) THEN 1 END) FROM <物理表> [WHERE <where>]
    /// - COUNT(expression) 只统计非 NULL 表达式，CASE 非匹配分支保持 NULL、
    ///   匹配分支返回 1，因此第二列为集合内匹配数；空集合两列均为 0；
    /// - All(match)（monostate）渲染为恒真表达式 "1 = 1"（matched == total，
    ///   不产生占位符）；All(where) 省略 WHERE；
    /// - 占位符顺序固定 match 在前、where 在后：appendCondition 按传入引用的
    ///   placeholder 计数递增编号（调用方初始化为 1）；绑定遍历必须按相同顺序
    ///   行走（先 BindWhereValues(match) 再 BindWhereValues(where)）；
    /// - appendIdentifier 为后端方言标识符引用；appendCondition 为后端方言的
    ///   条件文本渲染（签名与各后端文件内 AppendWhereText 一致，uint64 高半区
    ///   展开、LIKE ESCAPE 等差异全部留在回调内）。
    template <typename AppendConditionFn>
    DbResult<void> AppendCountMatchesAggregate(
        std::string& sql, const Ddl::CatalogTableMeta& meta, std::string_view physicalTable, const ConditionData& where,
        const ConditionData& match, std::size_t& placeholder, AppendIdentifierFn appendIdentifier,
        AppendConditionFn&& appendCondition)
    {
        sql += "SELECT COUNT(*), COUNT(CASE WHEN (";
        if (std::holds_alternative<std::monostate>(match)) {
            sql += "1 = 1";
        } else {
            const auto matchAppended = appendCondition(match, meta, placeholder, sql);
            if (!matchAppended.ok()) {
                return matchAppended;
            }
        }
        sql += ") THEN 1 END) FROM ";
        const auto tableAppended = appendIdentifier(sql, physicalTable);
        if (!tableAppended.ok()) {
            return tableAppended;
        }
        if (!std::holds_alternative<std::monostate>(where)) {
            sql += " WHERE ";
            const auto whereAppended = appendCondition(where, meta, placeholder, sql);
            if (!whereAppended.ok()) {
                return whereAppended;
            }
        }
        return {DbCode::OK, 0, {}};
    }

    /// ORDER BY 渲染遍历：复合排序键按传入顺序渲染，非主键排序追加主键
    /// tie-breaker（同值行按主键升序，可复现）。单字段排序表达式由后端方言
    /// 回调 appendSortColumn(sql, field, direction) 生成（direction 为
    /// " ASC"/" DESC"；SQLite 的 uint64 高半区双段排序展开留在回调内）；
    /// 主键补充列经 appendIdentifier 追加（方言标识符）。
    template <typename AppendSortColumnFn, typename AppendIdentifierFn>
    DbResult<void> AppendOrderBy(
        const Ddl::CatalogTableMeta& meta, const QueryOptions& options, std::string& sql,
        AppendSortColumnFn&& appendSortColumn, AppendIdentifierFn&& appendIdentifier)
    {
        if (options.sorts.empty()) {
            return {DbCode::OK, 0, {}};
        }
        const Ddl::FieldDesc* primaryKey = nullptr;
        for (std::size_t i = 0; i < meta.fieldCount; ++i) {
            if (meta.fields[i].primaryKey) {
                primaryKey = &meta.fields[i];
                break;
            }
        }
        bool sortHasPrimaryKey = false;
        bool firstSort = true;
        for (const auto& sort : options.sorts) {
            if (sort.first >= meta.fieldCount) {
                return {DbCode::INVALID_ARGUMENT, 0, "sort field is not mapped to a column"};
            }
            const Ddl::FieldDesc& field = meta.fields[sort.first];
            if (field.primaryKey) {
                sortHasPrimaryKey = true;
            }
            sql += firstSort ? " ORDER BY " : ", ";
            firstSort = false;
            const char* direction = sort.second == Order::ASCENDING ? " ASC" : " DESC";
            const auto appended = appendSortColumn(sql, field, direction);
            if (!appended.ok()) {
                return appended;
            }
        }
        if (primaryKey != nullptr && !sortHasPrimaryKey) {
            sql += ", ";
            const auto appended = appendIdentifier(sql, primaryKey->name);
            if (!appended.ok()) {
                return appended;
            }
            sql += " ASC";
        }
        return {DbCode::OK, 0, {}};
    }
} // namespace Relational
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_ENGINE_RELATIONAL_RELATIONAL_SQL_HELPERS_H
