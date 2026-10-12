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

// 文件职责：两个 SQL 后端共用关系处理辅助的实现（函数合同见同名头文件）。
#include "storage/engine/relational/relational_sql_helpers.h"

namespace HcclSim::Storage {
namespace Relational {

    std::string EscapePrefixPattern(const std::string& value)
    {
        std::string pattern;
        pattern.reserve(value.size() + 4);
        for (const char character : value) {
            if (character == '%' || character == '_' || character == '\\') {
                pattern += '\\';
            }
            pattern += character;
        }
        pattern += '%';
        return pattern;
    }

    DbResult<void> CheckColumnValue(const Ddl::FieldDesc& field, const DbScalar& value)
    {
        const bool isIntegerColumn = field.type == LogicalFieldType::SIGNED || field.type == LogicalFieldType::UNSIGNED
                                     || field.type == LogicalFieldType::BOOLEAN;
        const bool typeOk
            = (isIntegerColumn
               && (std::holds_alternative<int64_t>(value) || std::holds_alternative<uint64_t>(value)
                   || std::holds_alternative<bool>(value)))
              || (field.type == LogicalFieldType::REAL && !std::holds_alternative<std::string>(value)
                  && !std::holds_alternative<std::vector<uint8_t>>(value))
              || (field.type == LogicalFieldType::STRING && std::holds_alternative<std::string>(value))
              || (field.type == LogicalFieldType::BYTES && std::holds_alternative<std::vector<uint8_t>>(value));
        return typeOk ? DbResult<void>{DbCode::OK, 0, {}} :
                        DbResult<void>{
                            DbCode::INVALID_ARGUMENT, 0,
                            "logical value does not match the mapped column " + std::string{field.name}};
    }

    DbResult<std::string> PhysicalTableNameOf(const Ddl::CatalogTableMeta& meta, const TableScope& scope)
    {
        if (meta.scope == Ddl::ScopeKind::DynamicPid) {
            if (!scope.processId.has_value()) {
                return {
                    DbCode::INVALID_ARGUMENT, std::nullopt, 0,
                    "dynamic table requires a process scope: " + std::string{meta.tableId}};
            }
            return {
                DbCode::OK,
                std::optional<std::string>{std::string{meta.tableId} + "_P_" + std::to_string(*scope.processId)},
                0,
                {}};
        }
        return {DbCode::OK, std::optional<std::string>{std::string{meta.tableId}}, 0, {}};
    }

    DbResult<void> AppendInsertColumnListAndValues(
        std::string& sql, const Ddl::CatalogTableMeta& meta, AppendIdentifierFn appendIdentifier)
    {
        sql += " (";
        for (std::size_t i = 0; i < meta.fieldCount; ++i) {
            if (i > 0) {
                sql += ", ";
            }
            const auto appended = appendIdentifier(sql, meta.fields[i].name);
            if (!appended.ok()) {
                return appended;
            }
        }
        sql += ") VALUES (";
        for (std::size_t i = 0; i < meta.fieldCount; ++i) {
            sql += i == 0 ? "?" : ", ?";
        }
        sql += ")";
        return {DbCode::OK, 0, {}};
    }

    DbResult<void> ResolveProjection(
        const Ddl::CatalogTableMeta& meta, const QueryOptions& options, std::vector<std::size_t>& fieldIndices,
        std::vector<int32_t>& columnOfField)
    {
        // 空投影 = 全列：fieldIndices 覆盖全部字段（列序 = 声明序 = fieldId），
        // columnOfField 保持空（空映射即恒等映射；结果列绑定与 ReadField 按
        // 同一约定理解，不把空映射当零列）。
        if (options.projectedFields.empty()) {
            fieldIndices.resize(meta.fieldCount);
            for (std::size_t i = 0; i < meta.fieldCount; ++i) {
                fieldIndices[i] = i;
            }
            columnOfField.clear();
            return {DbCode::OK, 0, {}};
        }
        columnOfField.assign(meta.fieldCount, -1);
        fieldIndices.clear();
        for (const uint16_t fieldId : options.projectedFields) {
            if (fieldId >= meta.fieldCount) {
                return {
                    DbCode::INVALID_ARGUMENT, 0,
                    "projected field is not mapped to a column: " + std::to_string(fieldId)};
            }
            if (columnOfField[fieldId] < 0) {
                columnOfField[fieldId] = static_cast<int32_t>(fieldIndices.size());
                fieldIndices.push_back(fieldId);
            }
        }
        return {DbCode::OK, 0, {}};
    }
} // namespace Relational
} // namespace HcclSim::Storage
