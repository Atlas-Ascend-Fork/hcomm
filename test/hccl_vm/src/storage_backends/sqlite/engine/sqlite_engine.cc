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

// 文件职责：实现 SQLiteEngine 的全部能力：连接与 PRAGMA、静态 schema 单事务发布、
// 动态 PID 表惰性建表、每类 CRUD 的官方直通链（单次生成参数化 SQL → prepare_v2 →
// 按相同顺序直接 bind 原值 → step/column/changes/last_insert_rowid → finalize）、
// BEGIN IMMEDIATE/COMMIT/ROLLBACK 事务、sqlite3_backup 与错误码转换。
// 唯一锁：一把连接 mutex 覆盖同一 sqlite3 连接上的全部官方 API 调用。
// 单连接事务重叠合同：其他线程的即时 CRUD/第二笔 Begin 在活动事务期间经
// 条件变量有界等待（同线程立即失败防自锁，超时 RESOURCE_EXHAUSTED）。
// 原始 TEXT/BLOB 直接引用原 T/ConditionData/UpdateData 的存储并用 SQLITE_STATIC
// （源对象存活覆盖 step+finalize）；Derived 字段临时值与 PREFIX 转义模式是仅有的
// TRANSIENT 例外。不建跨调用语句缓存/manifest；批量 INSERT 一批内复用同一条
// 语句（一次 prepare，逐行 bind/step，行间 reset+clear_bindings），语句生命周期
// 止于单次调用，仍由局部 RAII finalize。
#include "storage_backends/sqlite/engine/sqlite_engine.h"

#include <sqlite3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "storage/engine/relational/relational_sql_helpers.h"

namespace HcclSim::Storage {
namespace {
    DbResult<void> SqliteError(DbCode code, const std::string& diagnostic) { return {code, 0, diagnostic}; }

    DbResult<void> InvalidArgument(const std::string& diagnostic) { return {DbCode::INVALID_ARGUMENT, 0, diagnostic}; }

    DbResult<void> InvalidState(const std::string& diagnostic) { return {DbCode::INVALID_STATE, 0, diagnostic}; }

    DbResult<std::string> FailText(DbCode code, const std::string& diagnostic)
    {
        return {code, std::nullopt, 0, diagnostic};
    }

    // SQLite 返回码到后端无关错误码的转换；未识别的返回码按 IO 错误处理。
    DbCode MapSqliteCode(int rc)
    {
        switch (rc & 0xFF) {
            case SQLITE_CONSTRAINT:
                return DbCode::CONFLICT;
            case SQLITE_BUSY:
            case SQLITE_LOCKED:
            case SQLITE_FULL:
            case SQLITE_NOMEM:
                return DbCode::RESOURCE_EXHAUSTED;
            case SQLITE_READONLY:
            case SQLITE_PERM:
                return DbCode::BACKEND_UNAVAILABLE;
            case SQLITE_IOERR:
            case SQLITE_CORRUPT:
            case SQLITE_NOTADB:
            case SQLITE_CANTOPEN:
            default:
                return DbCode::IO_ERROR;
        }
    }

    /// 标识符只允许字母数字与下划线（与基线渲染语义一致）
    DbResult<void> ValidateIdentifier(std::string_view name)
    {
        if (name.empty()) {
            return {DbCode::INVALID_ARGUMENT, 0, "identifier is empty"};
        }
        for (const char character : name) {
            const bool safe = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z')
                              || (character >= '0' && character <= '9') || character == '_';
            if (!safe) {
                return {DbCode::INVALID_ARGUMENT, 0, "identifier is unsafe: " + std::string{name}};
            }
        }
        return {DbCode::OK, 0, {}};
    }

    /// 双引号引用的拥有型标识符（DDL 渲染等冷路径；热路径用 AppendQuotedIdentifier
    /// 直接追加进最终 SQL，不产生第二份拥有字符串）
    DbResult<std::string> QuoteIdentifier(std::string_view name)
    {
        auto validated = ValidateIdentifier(name);
        if (!validated.ok()) {
            return {validated.code, std::nullopt, 0, validated.diagnostic};
        }
        return {DbCode::OK, std::optional<std::string>{"\"" + std::string{name} + "\""}, 0, {}};
    }

    /// 校验通过后把带双引号的标识符直接追加到最终 SQL 文本（单次调用内完成，
    /// 无第二份字符串、无跨调用状态）
    DbResult<void> AppendQuotedIdentifier(std::string& sql, std::string_view name)
    {
        auto validated = ValidateIdentifier(name);
        if (!validated.ok()) {
            return validated;
        }
        sql += '"';
        sql += name;
        sql += '"';
        return {DbCode::OK, 0, {}};
    }

    /// SQLite 声明类型（本文件唯一的物理类型转换点）
    const char* ColumnTypeSql(LogicalFieldType type)
    {
        switch (type) {
            case LogicalFieldType::SIGNED:
            case LogicalFieldType::UNSIGNED:
            case LogicalFieldType::BOOLEAN:
                return "INTEGER";
            case LogicalFieldType::REAL:
                return "REAL";
            case LogicalFieldType::STRING:
                return "TEXT";
            case LogicalFieldType::BYTES:
                return "BLOB";
        }
        return "BLOB";
    }

    bool IsIntegerColumn(LogicalFieldType type)
    {
        return type == LogicalFieldType::SIGNED || type == LogicalFieldType::UNSIGNED
               || type == LogicalFieldType::BOOLEAN;
    }

    /// 目录 uint64 UNSIGNED 列：SQLite INTEGER 按 signed 位型存储时，范围和排序
    /// 需要恢复无符号语义。窄 unsigned、SIGNED 与 BOOLEAN 均保持原生语义。
    bool IsUnsignedOrdering(const Ddl::FieldDesc& field)
    {
        return field.type == LogicalFieldType::UNSIGNED && field.byteWidth == sizeof(uint64_t);
    }

    // ---- 按声明宽度读取原始成员值（EmitFieldRef 的 data/bytes 解释）----

    int64_t ReadSignedValue(const void* data, size_t bytes)
    {
        switch (bytes) {
            case 1: {
                int8_t value = 0;
                std::memcpy(&value, data, sizeof(value));
                return value;
            }
            case 2: {
                int16_t value = 0;
                std::memcpy(&value, data, sizeof(value));
                return value;
            }
            case 4: {
                int32_t value = 0;
                std::memcpy(&value, data, sizeof(value));
                return value;
            }
            default: {
                int64_t value = 0;
                std::memcpy(&value, data, sizeof(value));
                return value;
            }
        }
    }

    uint64_t ReadUnsignedValue(const void* data, size_t bytes)
    {
        switch (bytes) {
            case 1: {
                uint8_t value = 0;
                std::memcpy(&value, data, sizeof(value));
                return value;
            }
            case 2: {
                uint16_t value = 0;
                std::memcpy(&value, data, sizeof(value));
                return value;
            }
            case 4: {
                uint32_t value = 0;
                std::memcpy(&value, data, sizeof(value));
                return value;
            }
            default: {
                uint64_t value = 0;
                std::memcpy(&value, data, sizeof(value));
                return value;
            }
        }
    }

    double ReadRealValue(const void* data, size_t bytes)
    {
        if (bytes == sizeof(float)) {
            float value = 0.0F;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        double value = 0.0;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }

    /// 自增主键"值 0 视为未指定"判定（整数/布尔按声明宽度读取）
    bool IsZeroIntegerValue(const Ddl::FieldDesc& field, const void* data, size_t bytes)
    {
        if (field.type == LogicalFieldType::BOOLEAN) {
            bool value = false;
            if (bytes >= sizeof(bool)) {
                std::memcpy(&value, data, sizeof(bool));
            }
            return !value;
        }
        if (field.type == LogicalFieldType::UNSIGNED) {
            return ReadUnsignedValue(data, bytes) == 0U;
        }
        if (field.type == LogicalFieldType::SIGNED) {
            return ReadSignedValue(data, bytes) == 0;
        }
        return false;
    }

    // ---- 绑定（原始引用 SQLITE_STATIC；Derived/PREFIX 临时值 TRANSIENT）----

    /// 把 EmitFieldRef 发射的原始成员值直接绑定到参数槽。transient 仅对 Derived
    /// 字段为 true（值来自目录 EmitDerived 的栈上临时 DbScalar，emit 返回后销毁）；
    /// 其余值直接引用原 T 成员，record 存活覆盖 step+finalize。
    DbResult<void> BindEmittedField(
        sqlite3* db, sqlite3_stmt* stmt, int index, const Ddl::FieldDesc& field, const void* data, size_t bytes,
        bool transient)
    {
        static_assert(sizeof(uint64_t) == sizeof(sqlite3_int64), "uint64_t and sqlite3_int64 must both be 64-bit wide");
        if (data == nullptr && bytes != 0) {
            return SqliteError(DbCode::INVALID_ARGUMENT, "borrowed insert value has null data");
        }
        sqlite3_destructor_type destructor = transient ? SQLITE_TRANSIENT : SQLITE_STATIC;
        int rc = SQLITE_MISUSE;
        switch (field.type) {
            case LogicalFieldType::SIGNED:
                rc = sqlite3_bind_int64(stmt, index, ReadSignedValue(data, bytes));
                break;
            case LogicalFieldType::UNSIGNED: {
                sqlite3_int64 bits = 0;
                const uint64_t value = ReadUnsignedValue(data, bytes);
                std::memcpy(&bits, &value, sizeof(bits)); // 64 位位型直存，最高位为 1 不截断
                rc = sqlite3_bind_int64(stmt, index, bits);
                break;
            }
            case LogicalFieldType::BOOLEAN:
                rc = sqlite3_bind_int(stmt, index, IsZeroIntegerValue(field, data, bytes) ? 0 : 1);
                break;
            case LogicalFieldType::REAL:
                rc = sqlite3_bind_double(stmt, index, ReadRealValue(data, bytes));
                break;
            case LogicalFieldType::STRING: {
                // 空文本绑零长度值而不是 NULL；data() 为空的调用方由目录保证非空。
                const char* text = bytes == 0 ? "" : static_cast<const char*>(data);
                rc = sqlite3_bind_text(stmt, index, text, static_cast<int>(bytes), destructor);
                break;
            }
            case LogicalFieldType::BYTES: {
                // 空 BLOB 绑零长度值而不是 NULL。
                const void* blob = bytes == 0 ? static_cast<const void*>("") : data;
                rc = sqlite3_bind_blob(stmt, index, blob, static_cast<int>(bytes), destructor);
                break;
            }
        }
        if (rc != SQLITE_OK) {
            return SqliteError(MapSqliteCode(rc), "sqlite bind failed: " + std::string(sqlite3_errmsg(db)));
        }
        return {DbCode::OK, 0, {}};
    }

    /// 绑定一个 DbScalar（WHERE/SET/LIMIT 值）。string/blob 直接引用唯一
    /// ConditionData/UpdateData 中的存储（SQLITE_STATIC，源对象存活覆盖 step+finalize）。
    DbResult<void> BindScalar(sqlite3* db, sqlite3_stmt* stmt, int index, const DbScalar& value)
    {
        static_assert(sizeof(uint64_t) == sizeof(sqlite3_int64), "uint64_t and sqlite3_int64 must both be 64-bit wide");
        int rc = SQLITE_MISUSE;
        if (const auto* number = std::get_if<int64_t>(&value); number != nullptr) {
            rc = sqlite3_bind_int64(stmt, index, *number);
        } else if (const auto* number = std::get_if<uint64_t>(&value); number != nullptr) {
            sqlite3_int64 bits = 0;
            std::memcpy(&bits, number, sizeof(bits));
            rc = sqlite3_bind_int64(stmt, index, bits);
        } else if (const auto* number = std::get_if<double>(&value); number != nullptr) {
            rc = sqlite3_bind_double(stmt, index, *number);
        } else if (const auto* flag = std::get_if<bool>(&value); flag != nullptr) {
            rc = sqlite3_bind_int(stmt, index, *flag ? 1 : 0);
        } else if (const auto* text = std::get_if<std::string>(&value); text != nullptr) {
            const char* data = text->empty() ? "" : text->data();
            rc = sqlite3_bind_text(stmt, index, data, static_cast<int>(text->size()), SQLITE_STATIC);
        } else if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&value); bytes != nullptr) {
            // data() 对空 vector 可能为空指针；传非空占位指针保证绑定零长度 BLOB 而不是 NULL。
            const void* data = bytes->empty() ? static_cast<const void*>("") : bytes->data();
            rc = sqlite3_bind_blob(stmt, index, data, static_cast<int>(bytes->size()), SQLITE_STATIC);
        }
        if (rc != SQLITE_OK) {
            return SqliteError(MapSqliteCode(rc), "sqlite bind failed: " + std::string(sqlite3_errmsg(db)));
        }
        return {DbCode::OK, 0, {}};
    }

    /// 比较符固定文本（constexpr 数组按连续枚举下标解释；IN/PREFIX 由专属形态展开）
    constexpr std::array<const char*, static_cast<std::size_t>(Compare::COUNT)> kCompareText
        = {"=", "<>", "<", "<=", ">", ">=", nullptr, nullptr};

    bool IsUnsignedRangeComparison(const Ddl::FieldDesc& field, const ConditionClause& clause)
    {
        if (!IsUnsignedOrdering(field)) {
            return false;
        }
        switch (clause.compare) {
            case Compare::LESS:
            case Compare::LESS_OR_EQUAL:
            case Compare::GREATER:
            case Compare::GREATER_OR_EQUAL:
                return std::holds_alternative<uint64_t>(std::get<DbScalar>(clause.value));
            default:
                return false;
        }
    }

    // ---- WHERE 生成（第一遍：扫描原条件生成参数化 SQL，同步完成校验）----

    DbResult<void> AppendClauseText(
        const ConditionClause& clause, const Ddl::CatalogTableMeta& meta, std::size_t& placeholder, std::string& output)
    {
        if (clause.fieldId >= meta.fieldCount) {
            return InvalidArgument("logical field is not mapped to a column: " + std::to_string(clause.fieldId));
        }
        const Ddl::FieldDesc& field = meta.fields[clause.fieldId];
        // 先行校验（与迁移前一致的失败时序：空 IN 等短路分支之前即拒绝不安全列名）
        const auto validated = ValidateIdentifier(field.name);
        if (!validated.ok()) {
            return validated;
        }
        const auto* single = std::get_if<DbScalar>(&clause.value);
        const auto* inValues = std::get_if<std::vector<DbScalar>>(&clause.value);
        const bool emptyIn = clause.compare == Compare::IN && inValues != nullptr && inValues->empty();
        if (emptyIn) {
            // 空 IN 渲染为常量假表达式：匹配 0 行，不产生占位符与绑定值。
            output += "1 = 0";
            return {DbCode::OK, 0, {}};
        }
        if (clause.compare == Compare::IN) {
            if (inValues == nullptr) {
                return InvalidArgument("predicate expects a single comparison value");
            }
            for (const auto& value : *inValues) {
                const auto checked = Relational::CheckColumnValue(field, value);
                if (!checked.ok()) {
                    return checked;
                }
            }
            const auto appended = AppendQuotedIdentifier(output, field.name);
            if (!appended.ok()) {
                return appended;
            }
            output += " IN (";
            for (std::size_t item = 0; item < inValues->size(); ++item) {
                output += item == 0 ? "?" : ", ?";
                ++placeholder;
            }
            output += ")";
            return {DbCode::OK, 0, {}};
        }
        if (inValues != nullptr) {
            return InvalidArgument("predicate expects a single comparison value");
        }
        if (single == nullptr) {
            return InvalidArgument("predicate has no comparison values");
        }
        if (clause.compare == Compare::PREFIX) {
            if (field.type != LogicalFieldType::STRING || !std::holds_alternative<std::string>(*single)) {
                return InvalidArgument("prefix comparison requires a text column and a string value");
            }
            const auto appended = AppendQuotedIdentifier(output, field.name);
            if (!appended.ok()) {
                return appended;
            }
            output += " LIKE ? ESCAPE '\\'";
            ++placeholder;
            return {DbCode::OK, 0, {}};
        }
        const auto checked = Relational::CheckColumnValue(field, *single);
        if (!checked.ok()) {
            return checked;
        }
        if (IsUnsignedRangeComparison(field, clause)) {
            // 无符号范围比较的值落在有符号高/低半区决定 SQL 展开；本次值决定形态，
            // 生成与绑定两次读取同一 ConditionData，决策一致。
            const bool highHalf
                = std::get<uint64_t>(*single) > static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
            const char* join = highHalf ? " OR " : " AND ";
            const char* joinInverted = highHalf ? " AND " : " OR ";
            const auto appendColumn = [&output, &field]() {
                return AppendQuotedIdentifier(output, field.name);
            };
            switch (clause.compare) {
                case Compare::LESS:
                    output += "(";
                    if (auto appended = appendColumn(); !appended.ok()) {
                        return appended;
                    }
                    output += " >= 0";
                    output += join;
                    if (auto appended = appendColumn(); !appended.ok()) {
                        return appended;
                    }
                    output += " < ?)";
                    break;
                case Compare::LESS_OR_EQUAL:
                    output += "(";
                    if (auto appended = appendColumn(); !appended.ok()) {
                        return appended;
                    }
                    output += " >= 0";
                    output += join;
                    if (auto appended = appendColumn(); !appended.ok()) {
                        return appended;
                    }
                    output += " <= ?)";
                    break;
                case Compare::GREATER:
                    output += "(";
                    if (auto appended = appendColumn(); !appended.ok()) {
                        return appended;
                    }
                    output += " < 0";
                    output += joinInverted;
                    if (auto appended = appendColumn(); !appended.ok()) {
                        return appended;
                    }
                    output += " > ?)";
                    break;
                case Compare::GREATER_OR_EQUAL:
                    output += "(";
                    if (auto appended = appendColumn(); !appended.ok()) {
                        return appended;
                    }
                    output += " < 0";
                    output += joinInverted;
                    if (auto appended = appendColumn(); !appended.ok()) {
                        return appended;
                    }
                    output += " >= ?)";
                    break;
                default:
                    return InvalidArgument("unsigned range comparison is not supported");
            }
            ++placeholder;
            return {DbCode::OK, 0, {}};
        }
        const char* compareText = kCompareText[static_cast<std::size_t>(clause.compare)];
        if (compareText == nullptr) {
            return InvalidArgument("compare kind is not supported");
        }
        const auto appended = AppendQuotedIdentifier(output, field.name);
        if (!appended.ok()) {
            return appended;
        }
        output += ' ';
        output += compareText;
        output += " ?";
        ++placeholder;
        return {DbCode::OK, 0, {}};
    }

    /// 条件文本（第一遍）：公共遍历（连接词/括号，见 relational_sql_helpers.h）
    /// 调用 SQLite 单子句方言函数；与占位符/绑定值顺序严格一致。
    DbResult<void> AppendWhereText(
        const ConditionData& where, const Ddl::CatalogTableMeta& meta, std::size_t& placeholder, std::string& output)
    {
        return Relational::AppendWhereText(
            where, output, [&meta, &placeholder, &output](const ConditionClause& clause) {
                return AppendClauseText(clause, meta, placeholder, output);
            });
    }

    // ---- WHERE 绑定（第二遍：按相同顺序直接 bind 原值）----

    DbResult<void> BindClauseValues(const ConditionClause& clause, sqlite3* db, sqlite3_stmt* stmt, int& nextIndex)
    {
        const auto* single = std::get_if<DbScalar>(&clause.value);
        const auto* inValues = std::get_if<std::vector<DbScalar>>(&clause.value);
        const bool emptyIn = clause.compare == Compare::IN && inValues != nullptr && inValues->empty();
        if (emptyIn) {
            return {DbCode::OK, 0, {}};
        }
        if (clause.compare == Compare::IN) {
            for (const auto& value : *inValues) {
                const auto bound = BindScalar(db, stmt, nextIndex, value);
                if (!bound.ok()) {
                    return bound;
                }
                ++nextIndex;
            }
            return {DbCode::OK, 0, {}};
        }
        if (clause.compare == Compare::PREFIX) {
            // 转义产生的临时字符串存活仅到本分支结束：必须以 SQLITE_TRANSIENT 绑定
            // （bind 返回前 SQLite 自行复制），不得经 BindScalar 的 STATIC 路径悬挂。
            const std::string pattern = Relational::EscapePrefixPattern(std::get<std::string>(*single));
            const char* data = pattern.empty() ? "" : pattern.data();
            const int rc = sqlite3_bind_text(stmt, nextIndex, data, static_cast<int>(pattern.size()), SQLITE_TRANSIENT);
            if (rc != SQLITE_OK) {
                return SqliteError(MapSqliteCode(rc), "sqlite bind failed: " + std::string(sqlite3_errmsg(db)));
            }
            ++nextIndex;
            return {DbCode::OK, 0, {}};
        }
        const auto bound = BindScalar(db, stmt, nextIndex, *single);
        if (!bound.ok()) {
            return bound;
        }
        ++nextIndex;
        return {DbCode::OK, 0, {}};
    }

    /// 条件绑定（第二遍）：公共遍历按相同顺序调用 SQLite 原生绑定（临时值生命
    /// 周期——PREFIX 转义模式的 TRANSIENT——留在 BindClauseValues 内）。
    DbResult<void> BindWhereValues(const ConditionData& where, sqlite3* db, sqlite3_stmt* stmt, int& nextIndex)
    {
        return Relational::BindWhereValues(where, [db, stmt, &nextIndex](const ConditionClause& clause) {
            return BindClauseValues(clause, db, stmt, nextIndex);
        });
    }

    /// SQLite 方言的单字段排序表达式（公共排序遍历的方言回调）：uint64 列以
    /// "(col < 0) DIR, col DIR" 恢复无符号语义（SQLite INTEGER 按有符号位型排序，
    /// 高半区会排在低半区之前）；其余列直出。direction 为 " ASC"/" DESC"。
    DbResult<void> AppendSqliteSortColumn(std::string& sql, const Ddl::FieldDesc& field, const char* direction)
    {
        if (IsUnsignedOrdering(field)) {
            sql += "(";
            const auto appended = AppendQuotedIdentifier(sql, field.name);
            if (!appended.ok()) {
                return appended;
            }
            sql += " < 0)";
            sql += direction;
            sql += ", ";
            const auto appendedAgain = AppendQuotedIdentifier(sql, field.name);
            if (!appendedAgain.ok()) {
                return appendedAgain;
            }
            sql += direction;
            return {DbCode::OK, 0, {}};
        }
        const auto appended = AppendQuotedIdentifier(sql, field.name);
        if (!appended.ok()) {
            return appended;
        }
        sql += direction;
        return {DbCode::OK, 0, {}};
    }

    // ---- SELECT 行源（NextRow/ReadFieldInto 直通 sqlite3_step/sqlite3_column）----

    struct SqliteRowCursor {
        sqlite3* db{nullptr};
        sqlite3_stmt* stmt{nullptr};
        const Ddl::CatalogTableMeta* meta{nullptr};
        int lastRc{SQLITE_DONE};
        /// 投影列映射：fieldId → 结果列下标。空 = 全列恒等映射（列序 = 声明序
        /// = fieldId，与空投影的 SELECT 全列清单一致理解）；-1 = 未投影（成员
        /// 保持记录默认值）。仅非空投影时由 Select 填充（全列热路径零新增分配）。
        std::vector<int32_t> columnOfField;

        static DbResult<bool> Next(void* opaque)
        {
            auto* self = static_cast<SqliteRowCursor*>(opaque);
            self->lastRc = sqlite3_step(self->stmt);
            if (self->lastRc == SQLITE_ROW) {
                return {DbCode::OK, true, 0, {}};
            }
            if (self->lastRc == SQLITE_DONE) {
                return {DbCode::OK, false, 0, {}};
            }
            return {
                MapSqliteCode(sqlite3_errcode(self->db)), std::nullopt, 0,
                "sqlite step failed: " + std::string(sqlite3_errmsg(self->db))};
        }

        /// 按 FieldDesc::type + byteWidth 把当前行列值直接写最终 T 成员。
        /// capacity 仅固定数组使用（BLOB 数组为有效长度约束后的容量）；
        /// NULL 列与未投影字段跳过对应成员（保持默认值）。
        static DbResult<void> ReadField(void* opaque, uint16_t fieldId, void* destination, std::size_t capacity)
        {
            auto* self = static_cast<SqliteRowCursor*>(opaque);
            if (fieldId >= self->meta->fieldCount) {
                return InvalidArgument("sqlite row field is unmapped");
            }
            int32_t column = 0;
            if (self->columnOfField.empty()) {
                // 全列直读：结果列序 = 声明顺序 = fieldId
                column = static_cast<int32_t>(fieldId);
            } else {
                column = self->columnOfField[fieldId];
                if (column < 0) {
                    return {DbCode::OK, 0, {}}; // 未投影字段保持记录默认值
                }
            }
            const Ddl::FieldDesc& field = self->meta->fields[fieldId];
            if (sqlite3_column_type(self->stmt, column) == SQLITE_NULL) {
                return {DbCode::OK, 0, {}};
            }
            switch (field.type) {
                case LogicalFieldType::SIGNED:
                case LogicalFieldType::UNSIGNED: {
                    const sqlite3_int64 value = sqlite3_column_int64(self->stmt, column);
                    switch (field.byteWidth) {
                        case 1: {
                            if (field.type == LogicalFieldType::SIGNED) {
                                const int8_t narrow = static_cast<int8_t>(value);
                                std::memcpy(destination, &narrow, sizeof(narrow));
                            } else {
                                const uint8_t narrow = static_cast<uint8_t>(value);
                                std::memcpy(destination, &narrow, sizeof(narrow));
                            }
                            break;
                        }
                        case 2: {
                            if (field.type == LogicalFieldType::SIGNED) {
                                const int16_t narrow = static_cast<int16_t>(value);
                                std::memcpy(destination, &narrow, sizeof(narrow));
                            } else {
                                const uint16_t narrow = static_cast<uint16_t>(value);
                                std::memcpy(destination, &narrow, sizeof(narrow));
                            }
                            break;
                        }
                        case 4: {
                            if (field.type == LogicalFieldType::SIGNED) {
                                const int32_t narrow = static_cast<int32_t>(value);
                                std::memcpy(destination, &narrow, sizeof(narrow));
                            } else {
                                const uint32_t narrow = static_cast<uint32_t>(value);
                                std::memcpy(destination, &narrow, sizeof(narrow));
                            }
                            break;
                        }
                        default: {
                            if (field.type == LogicalFieldType::SIGNED) {
                                std::memcpy(destination, &value, sizeof(value));
                            } else {
                                // UNSIGNED 64：按 64 位位型还原（最高位为 1 不经过符号收窄）
                                const uint64_t restored = static_cast<uint64_t>(value);
                                std::memcpy(destination, &restored, sizeof(restored));
                            }
                            break;
                        }
                    }
                    break;
                }
                case LogicalFieldType::BOOLEAN: {
                    const bool flag = sqlite3_column_int64(self->stmt, column) != 0;
                    std::memcpy(destination, &flag, sizeof(flag));
                    break;
                }
                case LogicalFieldType::REAL: {
                    if (field.byteWidth == sizeof(float)) {
                        const float narrow = static_cast<float>(sqlite3_column_double(self->stmt, column));
                        std::memcpy(destination, &narrow, sizeof(narrow));
                    } else {
                        const double value = sqlite3_column_double(self->stmt, column);
                        std::memcpy(destination, &value, sizeof(value));
                    }
                    break;
                }
                case LogicalFieldType::STRING: {
                    const unsigned char* text = sqlite3_column_text(self->stmt, column);
                    const int bytes = sqlite3_column_bytes(self->stmt, column);
                    if (field.byteWidth == 0) {
                        // std::string：SQLite 临时页到拥有型业务字段的唯一一次复制
                        auto* member = static_cast<std::string*>(destination);
                        if (text == nullptr) {
                            member->clear();
                        } else {
                            member->assign(reinterpret_cast<const char*>(text), static_cast<std::size_t>(bytes));
                        }
                    } else {
                        // 固定 char 数组：NUL 前字节承载语义；回写后显式补终止符，
                        // 目标结构体默认初始化的尾部未定字节不得泄漏进比较语义
                        const std::size_t bound = capacity > 0 ? capacity : field.byteWidth;
                        const std::size_t count = std::min(static_cast<std::size_t>(bytes), bound > 0 ? bound - 1 : 0);
                        if (count > 0 && text != nullptr) {
                            std::memcpy(destination, text, count);
                        }
                        if (bound > 0) {
                            static_cast<char*>(destination)[count] = '\0';
                        }
                    }
                    break;
                }
                case LogicalFieldType::BYTES: {
                    const void* blob = sqlite3_column_blob(self->stmt, column);
                    const int bytes = sqlite3_column_bytes(self->stmt, column);
                    if (field.byteWidth == 0) {
                        // std::vector<uint8_t>：唯一一次拥有化复制
                        auto* member = static_cast<std::vector<uint8_t>*>(destination);
                        if (blob == nullptr) {
                            member->clear();
                        } else {
                            const auto* begin = static_cast<const uint8_t*>(blob);
                            member->assign(begin, begin + bytes);
                        }
                    } else if (capacity > 0) {
                        // 固定字节数组：按有效长度约束后的容量截断
                        const std::size_t count = std::min(static_cast<std::size_t>(bytes), capacity);
                        if (count > 0 && blob != nullptr) {
                            std::memcpy(destination, blob, count);
                        }
                    } else if (static_cast<std::size_t>(bytes) == field.byteWidth && blob != nullptr) {
                        // 不透明定宽对象：仅字节数精确相等时整体回写
                        std::memcpy(destination, blob, field.byteWidth);
                    }
                    break;
                }
            }
            return {DbCode::OK, 0, {}};
        }
    };

    // ---- INSERT 字段绑定回调（唯一一次 visitRecord 内立即 bind）----

    struct InsertBindContext {
        sqlite3* db{nullptr};
        sqlite3_stmt* stmt{nullptr};
        const Ddl::CatalogTableMeta* meta{nullptr};

        static DbResult<void> Emit(void* opaque, uint16_t fieldId, const void* data, std::size_t bytes, bool isNull)
        {
            auto& self = *static_cast<InsertBindContext*>(opaque);
            if (fieldId >= self.meta->fieldCount) {
                return {DbCode::OK, 0, {}}; // 目录外字段忽略（防御；目录按声明顺序全量发射）
            }
            const Ddl::FieldDesc& field = self.meta->fields[fieldId];
            const int parameter = static_cast<int>(fieldId) + 1; // SQL 按声明顺序包含全部列
            if (isNull) {
                const int rc = sqlite3_bind_null(self.stmt, parameter);
                if (rc != SQLITE_OK) {
                    return SqliteError(
                        MapSqliteCode(rc), "sqlite bind failed: " + std::string(sqlite3_errmsg(self.db)));
                }
                return {DbCode::OK, 0, {}};
            }
            // 自增主键值 0 视为未指定：绑定 NULL 由 SQLite 生成（显式非零主键原样绑定）。
            if (field.primaryKey && Ddl::HasProp(field.props, Ddl::Prop::AutoIncrement)
                && IsZeroIntegerValue(field, data, bytes)) {
                const int rc = sqlite3_bind_null(self.stmt, parameter);
                if (rc != SQLITE_OK) {
                    return SqliteError(
                        MapSqliteCode(rc), "sqlite bind failed: " + std::string(sqlite3_errmsg(self.db)));
                }
                return {DbCode::OK, 0, {}};
            }
            // Derived 字段值来自目录 EmitDerived 的栈上临时 → TRANSIENT；
            // 其余直接引用原 T 成员 → STATIC（record 存活覆盖 step+finalize）。
            return BindEmittedField(
                self.db, self.stmt, parameter, field, data, bytes, Ddl::HasProp(field.props, Ddl::Prop::Derived));
        }
    };

    /// 单条/批量 INSERT 共用的固定列 SQL 构造：按声明顺序包含全部字段（自增主键
    /// 值 0 以 NULL 绑定由后端生成）。单次调用内直接生成最终 SQL：无 columns/values
    /// 中转字符串。单行 Insert 与 InsertMany 复用同一构造，不出现两份分叉实现。
    DbResult<std::string> BuildInsertSql(const Ddl::CatalogTableMeta& meta, std::string_view physicalTable)
    {
        if (meta.fieldCount == 0) {
            return FailText(DbCode::INVALID_ARGUMENT, "insert has no columns");
        }
        std::string sql;
        sql.reserve(48 + meta.fieldCount * 12);
        sql += "INSERT INTO ";
        const auto tableAppended = AppendQuotedIdentifier(sql, physicalTable);
        if (!tableAppended.ok()) {
            return FailText(tableAppended.code, tableAppended.diagnostic);
        }
        const auto columnsAppended = Relational::AppendInsertColumnListAndValues(sql, meta, &AppendQuotedIdentifier);
        if (!columnsAppended.ok()) {
            return FailText(columnsAppended.code, columnsAppended.diagnostic);
        }
        return {DbCode::OK, std::optional<std::string>{std::move(sql)}, 0, {}};
    }

    // ---- 无参 SQL 执行（DDL/事务边界）----

    DbResult<void> ExecuteSimple(sqlite3* db, const std::string& sql, const std::string& action)
    {
        char* message = nullptr;
        const int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &message);
        std::string diagnostic = message == nullptr ? std::string{} : std::string(message);
        if (message != nullptr) {
            sqlite3_free(message);
        }
        if (rc != SQLITE_OK) {
            return SqliteError(MapSqliteCode(rc), "sqlite " + action + " failed: " + diagnostic);
        }
        return {DbCode::OK, 0, {}};
    }

    // ---- 梯度延迟 busy handler（profile 决定是否安装；延迟梯度与基线一致）----

    int GradientBusyDelayMs(int retryCount)
    {
        static constexpr int kDelaysMs[] = {1, 2, 5, 10, 20, 50, 100, 200, 500};
        const int index = retryCount < 9 ? retryCount : 8;
        return kDelaysMs[index];
    }

    int GradientBusyHandler(void*, int retryCount)
    {
        const int delayMs = GradientBusyDelayMs(retryCount);
        std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
        return retryCount < 120;
    }

    // journal_mode 等连接建立 PRAGMA 在持有 writer lock 时可能直接返回 BUSY，
    // 不经 busy handler/busy_timeout。这里只补这一初始化动作的 SQLITE_BUSY：
    // LOCKED 仍按原错误返回，普通 DML/DDL 不经此函数。单次 PRAGMA 的总等待不能
    // 超过 profile 的时限，避免 SQLite 内部等待与外层等待累加。
    DbResult<void> ApplyConnectionPragma(sqlite3* db, const char* pragma, const SqliteConnectionProfile& profile)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(profile.busyTimeoutMs);
        for (int retryCount = 0;; ++retryCount) {
            auto applied = ExecuteSimple(db, pragma, "apply connection pragma");
            if (applied.ok()) {
                return applied;
            }
            if ((sqlite3_extended_errcode(db) & 0xFF) != SQLITE_BUSY) {
                return applied;
            }
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                return applied;
            }
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
            const int delayMs = profile.gradientBusyHandler ? GradientBusyDelayMs(retryCount) : 10;
            std::this_thread::sleep_for(std::min(std::chrono::milliseconds(delayMs), remaining));
            if (profile.gradientBusyHandler && retryCount >= 120) {
                return applied;
            }
        }
    }

    DbResult<std::string> ReadPragma(sqlite3* db, const char* name)
    {
        sqlite3_stmt* raw = nullptr;
        const std::string sql = "PRAGMA " + std::string(name);
        if (sqlite3_prepare_v2(db, sql.c_str(), -1, &raw, nullptr) != SQLITE_OK) {
            return {
                MapSqliteCode(sqlite3_errcode(db)), std::nullopt, 0,
                "sqlite prepare pragma check failed: " + std::string(sqlite3_errmsg(db))};
        }
        std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> statement(raw, sqlite3_finalize);
        if (sqlite3_step(statement.get()) != SQLITE_ROW) {
            return {
                MapSqliteCode(sqlite3_errcode(db)), std::nullopt, 0,
                "sqlite read pragma check failed: " + std::string(sqlite3_errmsg(db))};
        }
        const auto* value = sqlite3_column_text(statement.get(), 0);
        const int bytes = sqlite3_column_bytes(statement.get(), 0);
        return {
            DbCode::OK,
            std::optional<std::string>{
                value == nullptr ? std::string{} :
                                   std::string(reinterpret_cast<const char*>(value), static_cast<std::size_t>(bytes))},
            0,
            {}};
    }

    DbResult<void> VerifyConnectionPragmas(sqlite3* db, const SqliteConnectionProfile& profile, bool backupOnly)
    {
        const std::pair<const char*, std::string> localExpected[] = {
            {"foreign_keys", "1"},
            {"cache_size", "-10000"},
            {"temp_store", "2"},
            {"mmap_size", "268435456"},
        };
        const std::string autocheckpoint = std::to_string(profile.walAutocheckpoint);
        // 可写连接核验值与 Open 的可写 PRAGMA 集合一致：synchronous=0（OFF）与
        // secure_delete=0（OFF）为可重建数据的固定速度默认值（数据可丢弃重跑，
        // 不要求掉电可恢复性）；journal_mode/busy/外键/事务合同保持不变。
        const std::pair<const char*, std::string> writableExpected[] = {
            {"journal_mode", "wal"},
            {"synchronous", "0"},
            {"secure_delete", "0"},
            {"journal_size_limit", "67108864"},
            {"wal_autocheckpoint", autocheckpoint},
        };
        const auto verify = [db](const auto& expected) -> DbResult<void> {
            for (const auto& item : expected) {
                auto actual = ReadPragma(db, item.first);
                if (!actual.ok()) {
                    return {actual.code, actual.affectedRows, actual.diagnostic};
                }
                if (*actual.value != item.second) {
                    return SqliteError(
                        DbCode::IO_ERROR, "sqlite pragma did not take effect: " + std::string(item.first)
                                              + " expected=" + item.second + " actual=" + *actual.value);
                }
            }
            return {DbCode::OK, 0, {}};
        };
        auto local = verify(localExpected);
        if (!local.ok() || backupOnly) {
            return local;
        }
        return verify(writableExpected);
    }

    // 读取 sqlite_master 现有表/索引名集合（单条 SELECT 的单快照；WAL 读者不取写锁）。
    // 仅供 Prepare 的静态 schema 齐备性预检使用：期望名字与发布路径（meta.tableId +
    // DerivedIndexName）同源，不引入 manifest、结构缓存或跨进程协调状态。
    DbResult<std::set<std::string>> ExistingSchemaObjectNames(sqlite3* db)
    {
        sqlite3_stmt* raw = nullptr;
        if (sqlite3_prepare_v2(db, "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')", -1, &raw, nullptr)
            != SQLITE_OK) {
            return {
                MapSqliteCode(sqlite3_errcode(db)), std::nullopt, 0,
                "sqlite prepare schema snapshot failed: " + std::string(sqlite3_errmsg(db))};
        }
        std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> statement(raw, sqlite3_finalize);
        std::set<std::string> names;
        int rc = sqlite3_step(statement.get());
        while (rc == SQLITE_ROW) {
            const auto* value = sqlite3_column_text(statement.get(), 0);
            const int bytes = sqlite3_column_bytes(statement.get(), 0);
            if (value != nullptr && bytes > 0) {
                names.emplace(reinterpret_cast<const char*>(value), static_cast<std::size_t>(bytes));
            }
            rc = sqlite3_step(statement.get());
        }
        if (rc != SQLITE_DONE) {
            return {
                MapSqliteCode(rc), std::nullopt, 0,
                "sqlite read schema snapshot failed: " + std::string(sqlite3_errmsg(db))};
        }
        return {DbCode::OK, std::optional<std::set<std::string>>{std::move(names)}, 0, {}};
    }

    /// 目录字段名查找（入参可含 "表::成员" 前缀，按最后 "::" 之后的名字匹配本表字段）
    const Ddl::FieldDesc* FindField(const Ddl::CatalogTableMeta& meta, std::string_view name)
    {
        const auto sep = name.rfind("::");
        const std::string_view member = sep == std::string_view::npos ? name : name.substr(sep + 2);
        for (std::size_t i = 0; i < meta.fieldCount; ++i) {
            if (std::string_view{meta.fields[i].name} == member) {
                return &meta.fields[i];
            }
        }
        return nullptr;
    }

    // ---- DDL 渲染（直接遍历 CatalogTableMeta；外键目标只在启动建表时在本库目录解析）----

    DbResult<std::string> RenderCreateTable(
        const Ddl::DatabaseDescriptor& catalog, const Ddl::CatalogTableMeta& meta, const std::string& physicalTableName)
    {
        auto table = QuoteIdentifier(physicalTableName);
        if (!table.ok()) {
            return FailText(table.code, table.diagnostic);
        }
        std::string sql = "CREATE TABLE IF NOT EXISTS " + *table.value + " (";
        bool first = true;
        for (std::size_t i = 0; i < meta.fieldCount; ++i) {
            const Ddl::FieldDesc& field = meta.fields[i];
            auto name = QuoteIdentifier(field.name);
            if (!name.ok()) {
                return FailText(name.code, name.diagnostic);
            }
            sql += first ? "" : ", ";
            first = false;
            if (field.primaryKey) {
                sql += *name.value + " " + ColumnTypeSql(field.type) + " PRIMARY KEY";
                if (Ddl::HasProp(field.props, Ddl::Prop::AutoIncrement)) {
                    sql += " AUTOINCREMENT";
                }
                continue;
            }
            sql += *name.value + " " + ColumnTypeSql(field.type) + " NOT NULL";
        }
        for (std::size_t i = 0; i < meta.foreignKeyCount; ++i) {
            const Ddl::ForeignKeyDesc& foreignKey = meta.foreignKeys[i];
            const auto* localField = FindField(meta, foreignKey.localColumn);
            const Ddl::CatalogTableMeta* targetTable = nullptr;
            const Ddl::FieldDesc* targetField = nullptr;
            for (std::size_t t = 0; t < catalog.tableCount; ++t) {
                if (catalog.tables[t].tableId == Ddl::TargetTableName(foreignKey.target)) {
                    targetTable = &catalog.tables[t];
                    targetField = FindField(*targetTable, Ddl::TargetMemberName(foreignKey.target));
                    break;
                }
            }
            if (localField == nullptr || targetTable == nullptr || targetField == nullptr) {
                return FailText(
                    DbCode::INVALID_ARGUMENT,
                    "foreign key references an unmapped column or table: " + std::string{meta.tableId});
            }
            auto own = QuoteIdentifier(localField->name);
            auto targetTableId = QuoteIdentifier(targetTable->tableId);
            auto target = QuoteIdentifier(targetField->name);
            if (!own.ok()) {
                return FailText(own.code, own.diagnostic);
            }
            if (!targetTableId.ok()) {
                return FailText(targetTableId.code, targetTableId.diagnostic);
            }
            if (!target.ok()) {
                return FailText(target.code, target.diagnostic);
            }
            sql += ", FOREIGN KEY (" + *own.value + ") REFERENCES " + *targetTableId.value + " (" + *target.value + ")";
            if (foreignKey.cascadeDelete) {
                sql += " ON DELETE CASCADE";
            }
        }
        sql += ")";
        return {DbCode::OK, std::optional<std::string>{std::move(sql)}, 0, {}};
    }

    /// 物理索引名由表名与列名拼接（同一表共享首列的多条索引名字唯一）
    DbResult<std::string> DerivedIndexName(const Ddl::CatalogTableMeta& meta, const Ddl::IndexDesc& index)
    {
        std::string indexName = "idx_" + std::string{meta.tableId};
        for (std::size_t c = 0; c < index.columnCount; ++c) {
            const auto* field = FindField(meta, index.columns[c]);
            if (field == nullptr) {
                return FailText(
                    DbCode::INVALID_ARGUMENT, "index references an unmapped field: idx_" + std::string{meta.tableId});
            }
            indexName += "_" + std::string{field->name};
        }
        return {DbCode::OK, std::optional<std::string>{std::move(indexName)}, 0, {}};
    }

    DbResult<std::string> RenderCreateIndex(
        const Ddl::CatalogTableMeta& meta, const Ddl::IndexDesc& index, const std::string& indexName,
        const std::string& physicalTableName)
    {
        auto table = QuoteIdentifier(physicalTableName);
        auto name = QuoteIdentifier(indexName);
        if (!table.ok()) {
            return FailText(table.code, table.diagnostic);
        }
        if (!name.ok()) {
            return FailText(name.code, name.diagnostic);
        }
        std::string sql = "CREATE ";
        if (index.unique) {
            sql += "UNIQUE ";
        }
        sql += "INDEX IF NOT EXISTS " + *name.value + " ON " + *table.value + " (";
        bool first = true;
        for (std::size_t c = 0; c < index.columnCount; ++c) {
            const auto* field = FindField(meta, index.columns[c]);
            if (field == nullptr) {
                return FailText(DbCode::INVALID_ARGUMENT, "index references an unmapped field: " + indexName);
            }
            auto columnName = QuoteIdentifier(field->name);
            if (!columnName.ok()) {
                return FailText(columnName.code, columnName.diagnostic);
            }
            sql += first ? "" : ", ";
            first = false;
            sql += *columnName.value;
        }
        sql += ")";
        return {DbCode::OK, std::optional<std::string>{std::move(sql)}, 0, {}};
    }

    /// RAII finalize 守卫：每次调用一次 finalize，全路径覆盖。
    struct StatementFinalizer {
        sqlite3_stmt* stmt{nullptr};
        explicit StatementFinalizer(sqlite3_stmt* raw) : stmt(raw) {}
        ~StatementFinalizer()
        {
            if (stmt != nullptr) {
                (void)sqlite3_finalize(stmt);
            }
        }
        StatementFinalizer(const StatementFinalizer&) = delete;
        StatementFinalizer& operator=(const StatementFinalizer&) = delete;
    };

    /// 事务槽等待默认上限（项目既有排空口径 30 秒；私有常量，不新增配置面——
    /// 测试经 TestPeer 缩短，产品路径固定）
    constexpr std::chrono::milliseconds kTransactionWaitLimitMs{30000};
} // namespace

// ---- 引擎主体 ----

struct SQLiteEngine::Impl {
    Impl(std::string path, const Ddl::DatabaseDescriptor* descriptor, SqliteConnectionProfile connectionProfile)
        : databasePath(std::move(path)),
          catalog(descriptor),
          profile(connectionProfile)
    {}

    std::string databasePath;
    const Ddl::DatabaseDescriptor* catalog{nullptr};
    SqliteConnectionProfile profile{};
    sqlite3* db{nullptr};
    StorageOpenMode mode{StorageOpenMode::READ_WRITE_CREATE};
    std::string dataNamespace;
    enum class State : uint8_t { NOT_OPEN, OPENED, PREPARED, TRANSACTION_BUILDING, CLOSED };
    State state{State::NOT_OPEN};
    // 已成功创建的动态物理表名（唯一 DDL 生命周期状态；复用连接 mutex，不加锁）。
    // 事务回滚（含 COMMIT 失败后的回滚与 SQLite 自动回滚后的清理）会撤销事务内
    // 首次创建的动态表 DDL，集合随之在 RollbackPhysicalLocked 中整体失效；
    // 正常成功提交不清空。清空后下一次访问重新执行幂等 CREATE IF NOT EXISTS。
    std::unordered_set<std::string> ensuredDynamicTables;
    // 活动事务槽：同一连接同时只允许一个事务，以 binding 对象地址识别。
    // DML 失败不改变引擎状态（物理事务仍开启），失败回滚由 Transaction
    // 统一调用 Rollback 完成。
    const TransactionBinding* activeBinding{nullptr};
    // schema DDL 观察计数（仅启动/惰性建表路径；仅供测试固化结构合同）
    uint64_t staticDdlExecutions{0};
    uint64_t dynamicTableCreates{0};
    std::mutex mutex;
    // ---- 事务槽等待（单连接事务重叠合同）----
    // 活动事务与其他线程的即时 CRUD/第二笔 Begin 重叠时：同线程立即失败
    // （防自锁，错误文本不变），其他线程经条件变量有界等待事务结束。成功
    // 等待零日志；超时返回 RESOURCE_EXHAUSTED。
    std::condition_variable transactionIdle;
    std::thread::id transactionOwner{}; ///< 成功 Begin 的线程；离开事务时清空
    std::chrono::milliseconds transactionWaitLimit{kTransactionWaitLimitMs};

    /// 即时操作/第二事务的事务槽等待（持锁调用；wait_until 释放引擎 mutex）。
    /// 无活动事务、或当前线程即事务 owner（不得等待自己）时原样返回 OK，由
    /// 调用方的既有状态检查给出原错误；其他线程的活动事务期间等待其结束，
    /// 被唤醒后由调用方重新完整校验状态（可能已回到 PREPARED 或进入 CLOSED）。
    /// 超时返回 RESOURCE_EXHAUSTED（diagnostic 含后端、等待时长与
    /// transaction wait timed out）。
    DbResult<void> WaitForTransactionSlot(std::unique_lock<std::mutex>& lock)
    {
        if (state != State::TRANSACTION_BUILDING) {
            return {DbCode::OK, 0, {}};
        }
        if (transactionOwner == std::this_thread::get_id()) {
            return {DbCode::OK, 0, {}};
        }
        const auto startedAt = std::chrono::steady_clock::now();
        const bool idle = transactionIdle.wait_until(lock, startedAt + transactionWaitLimit, [this]() {
            return state != State::TRANSACTION_BUILDING;
        });
        if (idle) {
            return {DbCode::OK, 0, {}};
        }
        const auto waitedMs
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startedAt);
        return SqliteError(
            DbCode::RESOURCE_EXHAUSTED, "sqlite transaction wait timed out after " + std::to_string(waitedMs.count())
                                            + " ms: a transaction is building on this route");
    }

    /// 离开 TRANSACTION_BUILDING 的全部路径（Commit/Rollback/Close 与故障收尾）
    /// 在状态落定之后调用：清 owner 并唤醒全部等待者。
    void NotifyTransactionSlotReleasedLocked()
    {
        transactionOwner = {};
        transactionIdle.notify_all();
    }

    /// 动态 scope 表首次访问时按目录惰性建表（索引随物理表名唯一化）
    DbResult<void> EnsureDynamicTable(const Ddl::CatalogTableMeta& meta, const std::string& physicalTableName)
    {
        if (ensuredDynamicTables.count(physicalTableName) > 0) {
            return {DbCode::OK, 0, {}};
        }
        auto createTable = RenderCreateTable(*catalog, meta, physicalTableName);
        if (!createTable.ok()) {
            return {createTable.code, createTable.affectedRows, createTable.diagnostic};
        }
        auto created = ExecuteSimple(db, *createTable.value, "create dynamic table");
        if (!created.ok()) {
            return created;
        }
        for (std::size_t i = 0; i < meta.indexCount; ++i) {
            auto baseName = DerivedIndexName(meta, meta.indexes[i]);
            if (!baseName.ok()) {
                return {baseName.code, baseName.affectedRows, baseName.diagnostic};
            }
            // 物理索引名按物理表名派生，保证不同 pid scope 表的索引互不冲突。
            const std::string indexName = *baseName.value + "_" + physicalTableName;
            auto createIndex = RenderCreateIndex(meta, meta.indexes[i], indexName, physicalTableName);
            if (!createIndex.ok()) {
                return {createIndex.code, createIndex.affectedRows, createIndex.diagnostic};
            }
            auto indexed = ExecuteSimple(db, *createIndex.value, "create dynamic index");
            if (!indexed.ok()) {
                return indexed;
            }
        }
        ensuredDynamicTables.insert(physicalTableName);
        ++dynamicTableCreates;
        return {DbCode::OK, 0, {}};
    }

    /// 解析物理表名并按需惰性建表（CLEAR 不建表：表不存在时删行报错，与既有语义一致）
    DbResult<std::string>
    ResolvePhysicalTable(const Ddl::CatalogTableMeta& meta, const TableScope& scope, bool allowDynamicCreate)
    {
        auto physical = Relational::PhysicalTableNameOf(meta, scope);
        if (!physical.ok()) {
            return physical;
        }
        if (allowDynamicCreate && meta.scope == Ddl::ScopeKind::DynamicPid) {
            auto ensured = EnsureDynamicTable(meta, *physical.value);
            if (!ensured.ok()) {
                return {ensured.code, std::nullopt, ensured.affectedRows, ensured.diagnostic};
            }
        }
        return physical;
    }
};

SQLiteEngine::SQLiteEngine(
    std::string databasePath, const Ddl::DatabaseDescriptor* catalog, SqliteConnectionProfile profile)
    : impl_(std::make_unique<Impl>(std::move(databasePath), catalog, profile))
{}

// 析构只释放连接资源；不持久化、不重试、不做业务清理，也不隐式备份。
SQLiteEngine::~SQLiteEngine()
{
    if (impl_ != nullptr && impl_->db != nullptr) {
        (void)sqlite3_close(impl_->db);
        impl_->db = nullptr;
    }
}

// 按打开模式创建 SQLite 连接并应用连接级 PRAGMA（WAL、busy 策略、外键开关等）。
DbResult<void> SQLiteEngine::Open(const EngineOpenRequest& request)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->state == Impl::State::CLOSED) {
        return InvalidState("engine is closed");
    }
    if (impl_->state != Impl::State::NOT_OPEN) {
        return InvalidState("engine is already open");
    }
    if (impl_->databasePath.empty()) {
        return SqliteError(DbCode::INVALID_ARGUMENT, "sqlite database path is empty");
    }
    if (impl_->db != nullptr) {
        return SqliteError(DbCode::INVALID_STATE, "sqlite connection is already open");
    }
    int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_URI;
    if (request.mode == StorageOpenMode::READ_EXISTING) {
        flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_URI;
    } else if (request.mode == StorageOpenMode::BACKUP_ONLY) {
        flags = SQLITE_OPEN_READONLY | SQLITE_OPEN_URI;
    }
    sqlite3* db = nullptr;
    const int rc = sqlite3_open_v2(impl_->databasePath.c_str(), &db, flags, nullptr);
    if (rc != SQLITE_OK) {
        const std::string message = db == nullptr ? sqlite3_errstr(rc) : sqlite3_errmsg(db);
        if (db != nullptr) {
            sqlite3_close(db);
        }
        return SqliteError(
            DbCode::BACKEND_UNAVAILABLE, "failed to open sqlite database " + impl_->databasePath + ": " + message);
    }
    impl_->db = db;
    impl_->mode = request.mode;
    impl_->dataNamespace = request.dataNamespace;
    const bool backupOnly = request.mode == StorageOpenMode::BACKUP_ONLY;

    // BACKUP_ONLY 是只读句柄：不尝试 journal_mode、synchronous、journal_size_limit
    // 或 wal_autocheckpoint 等会改变库级/写入 profile 的项；仍设置本连接安全的读取
    // profile，并仅核验这些本连接生效值，不要求或修改源库的 journal_mode。
    // 可写连接（CREATE/READ_EXISTING）为可重建数据速度取向：synchronous=OFF
    // 与 secure_delete=OFF 是固定默认值（模拟器数据可丢弃重跑，不要求掉电
    // 可恢复性；WAL + NORMAL 本就不对大多数事务逐次同步，OFF 进一步省去
    // checkpoint/WAL 复用等同步工作）。备份目标连接单独在 BackupRelational
    // 中设置 synchronous=OFF（见该函数）。
    const std::string autocheckpoint = "PRAGMA wal_autocheckpoint=" + std::to_string(impl_->profile.walAutocheckpoint);
    const char* const writablePragmas[] = {
        "PRAGMA journal_mode=WAL",
        "PRAGMA synchronous=OFF",
        "PRAGMA secure_delete=OFF",
        "PRAGMA foreign_keys=ON",
        "PRAGMA cache_size=-10000",
        "PRAGMA temp_store=MEMORY",
        "PRAGMA journal_size_limit=67108864",
        autocheckpoint.c_str(),
        "PRAGMA mmap_size=268435456",
    };
    const char* const backupPragmas[] = {
        "PRAGMA foreign_keys=ON",
        "PRAGMA cache_size=-10000",
        "PRAGMA temp_store=MEMORY",
        "PRAGMA mmap_size=268435456",
    };
    const char* const* connectionPragmas = backupOnly ? backupPragmas : writablePragmas;
    const std::size_t pragmaCount = backupOnly ? sizeof(backupPragmas) / sizeof(*backupPragmas) :
                                                 sizeof(writablePragmas) / sizeof(*writablePragmas);
    for (std::size_t index = 0; index < pragmaCount; ++index) {
        auto applied = ApplyConnectionPragma(impl_->db, connectionPragmas[index], impl_->profile);
        if (!applied.ok()) {
            (void)sqlite3_close(impl_->db);
            impl_->db = nullptr;
            return applied;
        }
    }
    if (!backupOnly) {
        // 连接 PRAGMA 以 SQLite 默认无等待策略运行（上方 BUSY 循环只有一个时限）：
        // 该循环结束后才安装常规 DML busy 策略。
        const int busySetup = impl_->profile.gradientBusyHandler ?
                                  sqlite3_busy_handler(impl_->db, GradientBusyHandler, nullptr) :
                                  sqlite3_busy_timeout(impl_->db, impl_->profile.busyTimeoutMs);
        if (busySetup != SQLITE_OK) {
            const std::string diagnostic
                = "sqlite busy handler setup failed: " + std::string(sqlite3_errmsg(impl_->db));
            (void)sqlite3_close(impl_->db);
            impl_->db = nullptr;
            return SqliteError(DbCode::IO_ERROR, diagnostic);
        }
    }
    auto verified = VerifyConnectionPragmas(impl_->db, impl_->profile, backupOnly);
    if (!verified.ok()) {
        (void)sqlite3_close(impl_->db);
        impl_->db = nullptr;
        return verified;
    }
    impl_->state = Impl::State::OPENED;
    return {DbCode::OK, 0, {}};
}

// 静态 schema 单事务原子发布（owner 打开意图）：先以只读 sqlite_master 快照做
// 齐备性预检——已齐备直接进入 PREPARED（不取写锁不重放 DDL）；冷库/缺项才在
// BEGIN IMMEDIATE 内按目录顺序创建全部静态表/索引（CREATE IF NOT EXISTS）。
// consumer（READ_EXISTING）只读打开，不执行任何 DDL 与结构审计；缺失静态表由
// 首次 CRUD 的 SQLite 错误暴露。
DbResult<void> SQLiteEngine::Prepare()
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->state == Impl::State::NOT_OPEN || impl_->state == Impl::State::CLOSED) {
        return InvalidState("engine is not open");
    }
    if (impl_->state != Impl::State::OPENED) {
        return InvalidState("engine is already prepared");
    }
    if (impl_->mode == StorageOpenMode::BACKUP_ONLY || impl_->mode == StorageOpenMode::READ_EXISTING) {
        impl_->state = Impl::State::PREPARED;
        return {DbCode::OK, 0, {}};
    }
    if (impl_->db == nullptr) {
        return SqliteError(DbCode::INVALID_STATE, "sqlite connection is not open");
    }
    if (impl_->catalog == nullptr) {
        return SqliteError(DbCode::INVALID_STATE, "sqlite engine has no database catalog");
    }
    // SQLite 约束单列 INTEGER 主键（先于任何写事务校验全部表）。
    for (std::size_t t = 0; t < impl_->catalog->tableCount; ++t) {
        const auto& meta = impl_->catalog->tables[t];
        const Ddl::FieldDesc* primaryKey = nullptr;
        for (std::size_t f = 0; f < meta.fieldCount; ++f) {
            if (meta.fields[f].primaryKey) {
                if (primaryKey != nullptr) {
                    primaryKey = nullptr;
                    break;
                }
                primaryKey = &meta.fields[f];
            }
        }
        if (primaryKey == nullptr || !IsIntegerColumn(primaryKey->type)) {
            return SqliteError(
                DbCode::UNSUPPORTED_CAPABILITY,
                "sqlite requires a single INTEGER primary key column: " + std::string{meta.tableId});
        }
    }
    // 静态 schema 齐备性预检（只读 sqlite_master 快照，不取写锁）：目录声明的
    // 全部静态表名与索引名已齐备时，本次打开不重放 CREATE IF NOT EXISTS、不进入
    // BEGIN IMMEDIATE——已完成发布的库（其他进程已发布的共享库、加载记录位置的
    // 任务数据库）按既有 schema 直接使用，避免每个进程的重放序列化在写锁上。
    // 冷库/缺表/缺索引仍走下方原子的单事务发布（同一合同缺啥补啥）；期望名字与
    // 发布路径同源，不引入 manifest 或结构缓存；动态 PID 表仍由 EnsureDynamicTable
    // 惰性创建，不参与静态齐备性判定。
    {
        auto existing = ExistingSchemaObjectNames(impl_->db);
        if (!existing.ok()) {
            return {existing.code, existing.affectedRows, existing.diagnostic};
        }
        bool schemaComplete = true;
        for (std::size_t t = 0; schemaComplete && t < impl_->catalog->tableCount; ++t) {
            const auto& meta = impl_->catalog->tables[t];
            if (meta.scope == Ddl::ScopeKind::DynamicPid) {
                continue;
            }
            if (existing.value->count(std::string{meta.tableId}) == 0) {
                schemaComplete = false;
                break;
            }
            for (std::size_t i = 0; i < meta.indexCount; ++i) {
                auto indexName = DerivedIndexName(meta, meta.indexes[i]);
                if (!indexName.ok()) {
                    return {indexName.code, indexName.affectedRows, indexName.diagnostic};
                }
                if (existing.value->count(*indexName.value) == 0) {
                    schemaComplete = false;
                    break;
                }
            }
        }
        if (schemaComplete) {
            impl_->state = Impl::State::PREPARED;
            return {DbCode::OK, 0, {}};
        }
    }
    const auto rollbackWith = [this](DbResult<void> failure) {
        DbResult<void> rolledBack{DbCode::OK, 0, {}};
        if (failNextRollbackForTest_) {
            // 仅供故障注入测试（产品代码不得设置）：真实 WAL 连接上 ROLLBACK 失败
            // 不可构造，注入下一次失败以验证 rollback 失败路径的收尾合同。
            failNextRollbackForTest_ = false;
            rolledBack = SqliteError(DbCode::IO_ERROR, "sqlite rollback failed: fault injection for test");
        } else {
            rolledBack = ExecuteSimple(impl_->db, "ROLLBACK", "rollback");
        }
        if (!rolledBack.ok()) {
            // rollback 自身失败：事务的回滚状态不可证明，立即关闭连接（后续操作按
            // 未打开拒绝），诊断同时保留原始错误与 rollback 错误。
            const DbCode code = rolledBack.code;
            const std::string diagnostic = rolledBack.diagnostic;
            (void)CloseLocked();
            return SqliteError(
                failure.code, failure.diagnostic + "; rollback also failed and the sqlite connection was closed: "
                                  + diagnostic + " [original code " + std::to_string(static_cast<int>(code)) + "]");
        }
        return failure;
    };
    auto begun = ExecuteSimple(impl_->db, "BEGIN IMMEDIATE", "begin");
    if (!begun.ok()) {
        return begun;
    }
    for (std::size_t t = 0; t < impl_->catalog->tableCount; ++t) {
        const auto& meta = impl_->catalog->tables[t];
        if (meta.scope == Ddl::ScopeKind::DynamicPid) {
            continue; // 动态表按物理 scope 惰性创建
        }
        auto createTable = RenderCreateTable(*impl_->catalog, meta, std::string{meta.tableId});
        if (!createTable.ok()) {
            return rollbackWith({createTable.code, createTable.affectedRows, createTable.diagnostic});
        }
        auto created = ExecuteSimple(impl_->db, *createTable.value, "create table");
        if (!created.ok()) {
            return rollbackWith(created);
        }
        ++impl_->staticDdlExecutions;
        for (std::size_t i = 0; i < meta.indexCount; ++i) {
            auto indexName = DerivedIndexName(meta, meta.indexes[i]);
            if (!indexName.ok()) {
                return rollbackWith({indexName.code, indexName.affectedRows, indexName.diagnostic});
            }
            auto createIndex = RenderCreateIndex(meta, meta.indexes[i], *indexName.value, std::string{meta.tableId});
            if (!createIndex.ok()) {
                return rollbackWith({createIndex.code, createIndex.affectedRows, createIndex.diagnostic});
            }
            auto indexed = ExecuteSimple(impl_->db, *createIndex.value, "create index");
            if (!indexed.ok()) {
                return rollbackWith(indexed);
            }
            ++impl_->staticDdlExecutions;
        }
    }
    auto committed = ExecuteSimple(impl_->db, "COMMIT", "commit");
    if (!committed.ok()) {
        return rollbackWith(committed);
    }
    impl_->state = Impl::State::PREPARED;
    return {DbCode::OK, 0, {}};
}

// 即时/事务 INSERT 共用的唯一直通实现（调用前已持锁并完成状态校验）：
// meta 生成固定列 SQL 并 prepare，唯一一次 visitRecord 回调内立即绑定原 T
// 成员地址（SQLITE_STATIC；自增主键 0 → NULL 由 SQLite 生成）。
// 即时路径（transaction 为空）在活动事务属于其他线程时有界等待事务槽，
// 而不是立即 INVALID_STATE；同线程保持立即失败（防自锁）。
DbResult<uint64_t> SQLiteEngine::Insert(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* record, TransactionBinding* transaction)
{
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (transaction == nullptr) {
        auto waited = impl_->WaitForTransactionSlot(lock);
        if (!waited.ok()) {
            return {waited.code, std::nullopt, waited.affectedRows, waited.diagnostic};
        }
    }
    auto state = CheckDmlState(transaction);
    if (!state.ok()) {
        return {state.code, std::nullopt, state.affectedRows, state.diagnostic};
    }
    return InsertLocked(meta, scope, record);
}

DbResult<uint64_t>
SQLiteEngine::InsertLocked(const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* record)
{
    if (record == nullptr) {
        return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "insert requires a record payload"};
    }
    auto physical = impl_->ResolvePhysicalTable(meta, scope, true);
    if (!physical.ok()) {
        return {physical.code, std::nullopt, physical.affectedRows, physical.diagnostic};
    }
    auto sql = BuildInsertSql(meta, *physical.value);
    if (!sql.ok()) {
        return {sql.code, std::nullopt, sql.affectedRows, sql.diagnostic};
    }
    sqlite3_stmt* raw = nullptr;
    const int prepareRc = sqlite3_prepare_v2(impl_->db, sql.value->c_str(), -1, &raw, nullptr);
    if (prepareRc != SQLITE_OK) {
        return {
            MapSqliteCode(prepareRc), std::nullopt, 0,
            "sqlite prepare failed: " + std::string(sqlite3_errmsg(impl_->db)) + " [" + *sql.value + "]"};
    }
    StatementFinalizer finalizer(raw);
    InsertBindContext context{impl_->db, raw, &meta};
    const auto visited = meta.visitRecord(record, &context, InsertBindContext::Emit);
    if (!visited.ok()) {
        return {visited.code, std::nullopt, visited.affectedRows, visited.diagnostic};
    }
    const int stepRc = sqlite3_step(raw);
    if (stepRc != SQLITE_DONE) {
        return {
            MapSqliteCode(stepRc), std::nullopt, 0,
            "sqlite step failed: " + std::string(sqlite3_errmsg(impl_->db)) + " [" + *sql.value + "]"};
    }
    const uint64_t generatedId = static_cast<uint64_t>(sqlite3_last_insert_rowid(impl_->db));
    const uint64_t affectedRows = static_cast<uint64_t>(sqlite3_changes(impl_->db));
    return {DbCode::OK, generatedId, affectedRows, {}};
}

// 批量插入 override（T1 引擎侧）：入参合同、错误码与失败时序保持基类默认
// 逐行 Insert 的既有语义；一批内共享一次锁、一次物理表解析、一次 SQL 构造与
// 一次 prepare，逐行按原输入顺序 bind/step，行间 reset + clear_bindings 清理
// 参数（reset 不清参数），仍逐行读取 SQLite 实际生成主键（不推算）。SQLite
// 原生仍逐行执行——本实现只消除批内重复持锁/SQL 构造/prepare-finalize。
// 失败只返回错误（首个出错行的 0 基行号在 diagnostic），不回滚——物理回滚仍
// 由 Transaction 统一执行；语句经 RAII 在本次调用结束前 finalize，不缓存、
// 不跨调用保存任何语句状态。
DbResult<void> SQLiteEngine::InsertMany(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* const* records, std::size_t count,
    uint64_t* generatedIds, TransactionBinding* transaction)
{
    // 入参合同与基类默认实现逐条一致：count=0 平凡成功（不触碰指针/事务）；
    // 指针空或超限拒绝且零副作用。
    if (count == 0U) {
        return {DbCode::OK, 0, {}};
    }
    if (records == nullptr || generatedIds == nullptr || transaction == nullptr) {
        return InvalidArgument("insert-many requires records, an id array and an active transaction");
    }
    if (count > kInsertBatchMaxRows) {
        return InvalidArgument("insert-many batch exceeds kInsertBatchMaxRows");
    }
    // 整批一次持锁（默认实现经逐行 Insert 反复取放同一把锁）。
    std::lock_guard<std::mutex> lock(impl_->mutex);
    // 第 0 行的既有失败时序：状态失败、首行空指针、物理表解析/SQL/prepare 失败
    // 都对应默认实现第 0 行 Insert 的失败（含 " at row 0" 后缀）；首行空指针在
    // 解析物理表/生成 SQL/prepare 之前暴露，不提前建表。
    auto state = CheckDmlState(transaction);
    if (!state.ok()) {
        return {state.code, 0, state.diagnostic + " at row 0"};
    }
    if (records[0] == nullptr) {
        return InvalidArgument("insert requires a record payload at row 0");
    }
    auto physical = impl_->ResolvePhysicalTable(meta, scope, true);
    if (!physical.ok()) {
        return {physical.code, 0, physical.diagnostic + " at row 0"};
    }
    auto sql = BuildInsertSql(meta, *physical.value);
    if (!sql.ok()) {
        return {sql.code, 0, sql.diagnostic + " at row 0"};
    }
    sqlite3_stmt* raw = nullptr;
    const int prepareRc = sqlite3_prepare_v2(impl_->db, sql.value->c_str(), -1, &raw, nullptr);
    if (prepareRc != SQLITE_OK) {
        return {
            MapSqliteCode(prepareRc), 0,
            "sqlite prepare failed: " + std::string(sqlite3_errmsg(impl_->db)) + " [" + *sql.value + "] at row 0"};
    }
    StatementFinalizer finalizer(raw);
    InsertBindContext context{impl_->db, raw, &meta};
    for (std::size_t index = 0; index < count; ++index) {
        const void* record = records[index];
        if (record == nullptr) {
            return InvalidArgument("insert requires a record payload at row " + std::to_string(index));
        }
        const auto visited = meta.visitRecord(record, &context, InsertBindContext::Emit);
        if (!visited.ok()) {
            return {visited.code, 0, visited.diagnostic + " at row " + std::to_string(index)};
        }
        const int stepRc = sqlite3_step(raw);
        if (stepRc != SQLITE_DONE) {
            return {
                MapSqliteCode(stepRc), 0,
                "sqlite step failed: " + std::string(sqlite3_errmsg(impl_->db)) + " [" + *sql.value + "] at row "
                    + std::to_string(index)};
        }
        generatedIds[index] = static_cast<uint64_t>(sqlite3_last_insert_rowid(impl_->db));
        if (index + 1U >= count) {
            continue; // 末行：无下一行可清理，语句由 RAII finalize
        }
        // 结束第 index 行执行并为下一行清理参数。reset 不清参数——必须再
        // clear_bindings；两者失败都归属刚执行的第 index 行（不是尚未执行的
        // 下一行），失败即返回，不用清理错误覆盖原始执行结果。
        const int resetRc = sqlite3_reset(raw);
        if (resetRc != SQLITE_OK) {
            return {
                MapSqliteCode(resetRc), 0,
                "sqlite reset failed: " + std::string(sqlite3_errmsg(impl_->db)) + " at row " + std::to_string(index)};
        }
        const int clearRc = sqlite3_clear_bindings(raw);
        if (clearRc != SQLITE_OK) {
            return {
                MapSqliteCode(clearRc), 0,
                "sqlite clear bindings failed: " + std::string(sqlite3_errmsg(impl_->db)) + " at row "
                    + std::to_string(index)};
        }
    }
    return {DbCode::OK, count, {}};
}

// 即时 SELECT：两遍同序遍历（生成 SQL / 绑定原值）→ materializeRows 直写
// ResultSink 指向的最终 T/vector<T>。投影：空投影 = 全列直读（列序 = 声明序
// = fieldId，不建投影字段数组和列映射数组——全列热路径零新增分配）；非空
// 投影按给定顺序去重生成 SELECT 列清单，并建立 fieldId → 结果列下标映射
// （-1 = 未投影，物化保持记录默认值）；WHERE/ORDER BY 引用字段不要求被投影
// 包含（过滤/排序由数据库端独立完成，字段合法性在各自渲染中校验）。
DbResult<void> SQLiteEngine::Select(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where, const QueryOptions& options,
    ResultSink& sink)
{
    std::unique_lock<std::mutex> lock(impl_->mutex);
    // 即时读在活动事务属于其他线程时有界等待事务槽（同线程立即失败防自锁）
    auto waited = impl_->WaitForTransactionSlot(lock);
    if (!waited.ok()) {
        return waited;
    }
    auto state = CheckOperationState();
    if (!state.ok()) {
        return state;
    }
    if (meta.materializeRows == nullptr) {
        return {DbCode::INVALID_ARGUMENT, 0, "table has no row materializer"};
    }
    auto physical = impl_->ResolvePhysicalTable(meta, scope, true);
    if (!physical.ok()) {
        return {physical.code, physical.affectedRows, physical.diagnostic};
    }
    std::size_t placeholder = 1;
    std::vector<std::size_t> fieldIndices;
    std::vector<int32_t> columnOfField;
    const bool projected = !options.projectedFields.empty();
    if (projected) {
        const auto projection = Relational::ResolveProjection(meta, options, fieldIndices, columnOfField);
        if (!projection.ok()) {
            return projection;
        }
    }
    // 单次调用内直接生成最终 SQL：投影清单、WHERE、ORDER BY、LIMIT 全部按原顺序
    // 追加进同一份文本（无 columns/whereText 中转）。
    std::string sql;
    sql.reserve(48 + meta.fieldCount * 12);
    sql += "SELECT ";
    if (projected) {
        for (std::size_t i = 0; i < fieldIndices.size(); ++i) {
            if (i > 0) {
                sql += ", ";
            }
            const auto appended = AppendQuotedIdentifier(sql, meta.fields[fieldIndices[i]].name);
            if (!appended.ok()) {
                return {appended.code, appended.affectedRows, appended.diagnostic};
            }
        }
    } else {
        for (std::size_t i = 0; i < meta.fieldCount; ++i) {
            if (i > 0) {
                sql += ", ";
            }
            const auto appended = AppendQuotedIdentifier(sql, meta.fields[i].name);
            if (!appended.ok()) {
                return {appended.code, appended.affectedRows, appended.diagnostic};
            }
        }
    }
    sql += " FROM ";
    const auto tableAppended = AppendQuotedIdentifier(sql, *physical.value);
    if (!tableAppended.ok()) {
        return {tableAppended.code, tableAppended.affectedRows, tableAppended.diagnostic};
    }
    if (!std::holds_alternative<std::monostate>(where)) {
        sql += " WHERE ";
        const auto appended = AppendWhereText(where, meta, placeholder, sql);
        if (!appended.ok()) {
            return appended;
        }
    }
    const auto sortFailure
        = Relational::AppendOrderBy(meta, options, sql, &AppendSqliteSortColumn, &AppendQuotedIdentifier);
    if (!sortFailure.ok()) {
        return sortFailure;
    }
    if (options.limit.has_value()) {
        if (*options.limit == 1) {
            // SelectOne 固定上限 1 的字面量表达：SQLite 对 LIMIT 参数在每次执行前
            // 按绑定值重编译语句（SQLITE_STMTSTATUS_REPREPARE 实测每调用 +1，
            // 10k 次同形查询 +38.6% 原生耗时），字面量 1 消除该重编译。
            // 常量 1 不携带业务文本；非 1 的动态上限仍参数化。
            sql += " LIMIT 1";
        } else {
            sql += " LIMIT ?"; // limit 值参数化，不进文本字面量
            ++placeholder;
        }
    }
    sqlite3_stmt* raw = nullptr;
    const int prepareRc = sqlite3_prepare_v2(impl_->db, sql.c_str(), -1, &raw, nullptr);
    if (prepareRc != SQLITE_OK) {
        return {
            MapSqliteCode(prepareRc), 0,
            "sqlite prepare failed: " + std::string(sqlite3_errmsg(impl_->db)) + " [" + sql + "]"};
    }
    StatementFinalizer finalizer(raw);
    int nextIndex = 1;
    if (!std::holds_alternative<std::monostate>(where)) {
        const auto bound = BindWhereValues(where, impl_->db, raw, nextIndex);
        if (!bound.ok()) {
            return bound;
        }
    }
    if (options.limit.has_value() && *options.limit != 1) {
        const DbScalar limitValue{*options.limit};
        const auto bound = BindScalar(impl_->db, raw, nextIndex, limitValue);
        if (!bound.ok()) {
            return bound;
        }
        ++nextIndex;
    }
    SqliteRowCursor source{impl_->db, raw, &meta, SQLITE_DONE};
    source.columnOfField = std::move(columnOfField);
    return meta.materializeRows(sink, &source, SqliteRowCursor::Next, SqliteRowCursor::ReadField);
}

// 即时 COUNT：数据库端聚合，标量直接返回（不经 ResultSink）。
DbResult<uint64_t>
SQLiteEngine::Count(const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where)
{
    std::unique_lock<std::mutex> lock(impl_->mutex);
    auto waited = impl_->WaitForTransactionSlot(lock);
    if (!waited.ok()) {
        return {waited.code, std::nullopt, waited.affectedRows, waited.diagnostic};
    }
    auto state = CheckOperationState();
    if (!state.ok()) {
        return {state.code, std::nullopt, state.affectedRows, state.diagnostic};
    }
    auto physical = impl_->ResolvePhysicalTable(meta, scope, true);
    if (!physical.ok()) {
        return {physical.code, std::nullopt, physical.affectedRows, physical.diagnostic};
    }
    std::size_t placeholder = 1;
    std::string sql;
    sql.reserve(48);
    sql += "SELECT COUNT(*) FROM ";
    const auto tableAppended = AppendQuotedIdentifier(sql, *physical.value);
    if (!tableAppended.ok()) {
        return {tableAppended.code, std::nullopt, tableAppended.affectedRows, tableAppended.diagnostic};
    }
    if (!std::holds_alternative<std::monostate>(where)) {
        sql += " WHERE ";
        const auto appended = AppendWhereText(where, meta, placeholder, sql);
        if (!appended.ok()) {
            return {appended.code, std::nullopt, appended.affectedRows, appended.diagnostic};
        }
    }
    sqlite3_stmt* raw = nullptr;
    const int prepareRc = sqlite3_prepare_v2(impl_->db, sql.c_str(), -1, &raw, nullptr);
    if (prepareRc != SQLITE_OK) {
        return {
            MapSqliteCode(prepareRc), std::nullopt, 0,
            "sqlite prepare failed: " + std::string(sqlite3_errmsg(impl_->db)) + " [" + sql + "]"};
    }
    StatementFinalizer finalizer(raw);
    int nextIndex = 1;
    if (!std::holds_alternative<std::monostate>(where)) {
        const auto bound = BindWhereValues(where, impl_->db, raw, nextIndex);
        if (!bound.ok()) {
            return {bound.code, std::nullopt, bound.affectedRows, bound.diagnostic};
        }
    }
    const int stepRc = sqlite3_step(raw);
    if (stepRc != SQLITE_ROW) {
        return {
            MapSqliteCode(stepRc == SQLITE_DONE ? sqlite3_errcode(impl_->db) : stepRc), std::nullopt, 0,
            "sqlite step failed: " + std::string(sqlite3_errmsg(impl_->db)) + " [" + sql + "]"};
    }
    const uint64_t counted = static_cast<uint64_t>(sqlite3_column_int64(raw, 0));
    return {DbCode::OK, counted, 1, {}};
}

// 即时 CountMatches：单条聚合语句同时返回 where 集合总数与其中匹配 match 的
// 数量（公共 SQL 结构见 Relational::AppendCountMatchesAggregate；占位符顺序
// match 在前、where 在后，绑定遍历同序）。无故障成功调用只执行这一条语句。
DbResult<CountMatchResult> SQLiteEngine::CountMatches(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where, const ConditionData& match)
{
    std::unique_lock<std::mutex> lock(impl_->mutex);
    auto waited = impl_->WaitForTransactionSlot(lock);
    if (!waited.ok()) {
        return {waited.code, std::nullopt, waited.affectedRows, waited.diagnostic};
    }
    auto state = CheckOperationState();
    if (!state.ok()) {
        return {state.code, std::nullopt, state.affectedRows, state.diagnostic};
    }
    auto physical = impl_->ResolvePhysicalTable(meta, scope, true);
    if (!physical.ok()) {
        return {physical.code, std::nullopt, physical.affectedRows, physical.diagnostic};
    }
    std::size_t placeholder = 1;
    std::string sql;
    sql.reserve(64);
    const auto aggregated = Relational::AppendCountMatchesAggregate(
        sql, meta, *physical.value, where, match, placeholder, &AppendQuotedIdentifier, &AppendWhereText);
    if (!aggregated.ok()) {
        return {aggregated.code, std::nullopt, aggregated.affectedRows, aggregated.diagnostic};
    }
    sqlite3_stmt* raw = nullptr;
    const int prepareRc = sqlite3_prepare_v2(impl_->db, sql.c_str(), -1, &raw, nullptr);
    if (prepareRc != SQLITE_OK) {
        return {
            MapSqliteCode(prepareRc), std::nullopt, 0,
            "sqlite prepare failed: " + std::string(sqlite3_errmsg(impl_->db)) + " [" + sql + "]"};
    }
    StatementFinalizer finalizer(raw);
    int nextIndex = 1;
    if (!std::holds_alternative<std::monostate>(match)) {
        const auto matchBound = BindWhereValues(match, impl_->db, raw, nextIndex);
        if (!matchBound.ok()) {
            return {matchBound.code, std::nullopt, matchBound.affectedRows, matchBound.diagnostic};
        }
    }
    if (!std::holds_alternative<std::monostate>(where)) {
        const auto whereBound = BindWhereValues(where, impl_->db, raw, nextIndex);
        if (!whereBound.ok()) {
            return {whereBound.code, std::nullopt, whereBound.affectedRows, whereBound.diagnostic};
        }
    }
    const int stepRc = sqlite3_step(raw);
    if (stepRc != SQLITE_ROW) {
        return {
            MapSqliteCode(stepRc == SQLITE_DONE ? sqlite3_errcode(impl_->db) : stepRc), std::nullopt, 0,
            "sqlite step failed: " + std::string(sqlite3_errmsg(impl_->db)) + " [" + sql + "]"};
    }
    CountMatchResult counted;
    counted.total = static_cast<uint64_t>(sqlite3_column_int64(raw, 0));
    counted.matched = static_cast<uint64_t>(sqlite3_column_int64(raw, 1));
    return {DbCode::OK, counted, 1, {}};
}

// 即时/事务 UPDATE 共用的唯一直通实现（调用前已持锁并完成状态校验）：
// UpdateData 生成 SET 文本并绑定，随后绑定同一 ConditionData。
// 参数顺序固定为 UpdateData 在前、ConditionData 在后。
DbResult<uint64_t> SQLiteEngine::Update(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where, const UpdateData& update,
    TransactionBinding* transaction)
{
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (transaction == nullptr) {
        auto waited = impl_->WaitForTransactionSlot(lock);
        if (!waited.ok()) {
            return {waited.code, std::nullopt, waited.affectedRows, waited.diagnostic};
        }
    }
    auto state = CheckDmlState(transaction);
    if (!state.ok()) {
        return {state.code, std::nullopt, state.affectedRows, state.diagnostic};
    }
    return UpdateLocked(meta, scope, where, update);
}

DbResult<uint64_t> SQLiteEngine::UpdateLocked(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where, const UpdateData& update)
{
    if (update.empty()) {
        return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "update requires changes and a condition"};
    }
    if (std::holds_alternative<std::monostate>(where)) {
        return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "update requires changes and a condition"};
    }
    for (const auto& change : update) {
        if (change.fieldId >= meta.fieldCount) {
            return {
                DbCode::INVALID_ARGUMENT, std::nullopt, 0,
                "update field is not mapped to a column: " + std::to_string(change.fieldId)};
        }
        const Ddl::FieldDesc& field = meta.fields[change.fieldId];
        if (field.primaryKey) {
            return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "primary key column cannot be updated"};
        }
        if (change.kind == ChangeKind::INCREMENT && !IsIntegerColumn(field.type)
            && field.type != LogicalFieldType::REAL) {
            return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "increment requires a numeric column"};
        }
        const auto checked = Relational::CheckColumnValue(field, change.value);
        if (!checked.ok()) {
            return {checked.code, std::nullopt, checked.affectedRows, checked.diagnostic};
        }
    }
    // —— INCREMENT 域守卫计划（SET 文本与参数绑定共用同一份决策，两循环不
    // 各写一份域判断）：SQLite INTEGER 为有符号 64 位，无符号值以位型存取；
    // 算术越界时 SQLite 静默转 REAL（浮点化）或依赖位型回绕，都会掩盖非法
    // 上下溢。守卫用 CASE 把越界结果归 NULL——列声明 NOT NULL，NULL 赋值使
    // 单条 UPDATE 以 SQLITE_CONSTRAINT 原子失败：不回绕、不截断、不浮点化。
    // 域口径：无符号列的基值与结果须落在 [0, INT64_MAX]（位型域 [2^63,
    // 2^64-1] 的增量保守拒绝；MariaDB 支持无符号全域，此为 SQLite
    // 表示域差异）；有符号列按 int64 全域精确守卫；幅值超 INT64_MAX（含
    // INT64_MIN 增量）显式拒绝——位型绑定会翻转符号语义。
    // INT64_MAX 以十进制字面量内联（int64 可解析）；INT64_MIN 必须写成
    // (-9223372036854775807 - 1)：-9223372036854775808 字面量会按 REAL 解析。
    struct SetPlan {
        std::string rhs; ///< “列 = ”之后的表达式；? 与 binds 一一对应
        std::vector<DbScalar> binds;
    };
    std::vector<SetPlan> plans;
    plans.reserve(update.size());
    for (const auto& change : update) {
        const Ddl::FieldDesc& field = meta.fields[change.fieldId];
        std::string column;
        const auto quoted = AppendQuotedIdentifier(column, field.name);
        if (!quoted.ok()) {
            return {quoted.code, std::nullopt, quoted.affectedRows, quoted.diagnostic};
        }
        SetPlan plan;
        if (change.kind != ChangeKind::INCREMENT) {
            plan.rhs = "?";
            plan.binds.push_back(change.value);
            plans.push_back(std::move(plan));
            continue;
        }
        if (field.type == LogicalFieldType::UNSIGNED) {
            bool negative = false;
            uint64_t magnitude = 0;
            if (const auto* delta = std::get_if<int64_t>(&change.value); delta != nullptr) {
                negative = *delta < 0;
                // INT64_MIN 幅值经无符号回绕取得（禁止对 INT64_MIN 取负）；
                // 幅值 > INT64_MAX 在下方统一显式拒绝
                magnitude = negative ? (uint64_t{0} - static_cast<uint64_t>(*delta)) : static_cast<uint64_t>(*delta);
            } else if (const auto* delta = std::get_if<uint64_t>(&change.value); delta != nullptr) {
                magnitude = *delta;
            } else {
                return {
                    DbCode::INVALID_ARGUMENT, std::nullopt, 0,
                    "increment value type does not match the unsigned field: " + std::string{field.name}};
            }
            if (magnitude > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                return {
                    DbCode::INVALID_ARGUMENT, std::nullopt, 0,
                    "increment magnitude exceeds the sqlite integer domain: " + std::string{field.name}};
            }
            const DbScalar bound{static_cast<int64_t>(magnitude)};
            if (negative) {
                plan.rhs = "CASE WHEN " + column + " - ? >= 0 THEN " + column + " - ? ELSE NULL END";
            } else {
                plan.rhs = "CASE WHEN " + column + " >= 0 AND " + column
                           + " + ? <= 9223372036854775807"
                             " THEN "
                           + column + " + ? ELSE NULL END";
            }
            plan.binds.push_back(bound);
            plan.binds.push_back(bound);
        } else if (field.type == LogicalFieldType::SIGNED) {
            int64_t delta = 0;
            if (const auto* signedDelta = std::get_if<int64_t>(&change.value); signedDelta != nullptr) {
                delta = *signedDelta;
            } else if (const auto* unsignedDelta = std::get_if<uint64_t>(&change.value); unsignedDelta != nullptr) {
                if (*unsignedDelta > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                    return {
                        DbCode::INVALID_ARGUMENT, std::nullopt, 0,
                        "increment value exceeds the sqlite signed integer domain: " + std::string{field.name}};
                }
                delta = static_cast<int64_t>(*unsignedDelta);
            } else if (const auto* flag = std::get_if<bool>(&change.value); flag != nullptr) {
                delta = *flag ? 1 : 0;
            } else {
                return {
                    DbCode::INVALID_ARGUMENT, std::nullopt, 0,
                    "increment value type does not match the signed field: " + std::string{field.name}};
            }
            const DbScalar bound{delta};
            plan.rhs = "CASE WHEN NOT ((" + column + " > 0 AND ? > 0 AND " + column + " > 9223372036854775807 - ?) OR ("
                       + column + " < 0 AND ? < 0 AND " + column + " < (-9223372036854775807 - 1) - ?)) THEN " + column
                       + " + ? ELSE NULL END";
            // 五个 ?：上溢判定 2 + 下溢判定 2 + THEN 结果 1，绑定同一增量
            plan.binds.insert(plan.binds.end(), 5, bound);
        } else { // REAL：无整数边界，原生加法
            plan.rhs = column + " + ?";
            plan.binds.push_back(change.value);
        }
        plans.push_back(std::move(plan));
    }
    auto physical = impl_->ResolvePhysicalTable(meta, scope, true);
    if (!physical.ok()) {
        return {physical.code, std::nullopt, physical.affectedRows, physical.diagnostic};
    }
    std::size_t placeholder = 1;
    // 单次调用内直接生成最终 SQL（无 assignments/whereText 中转）。
    std::string sql;
    sql.reserve(64 + update.size() * 96);
    sql += "UPDATE ";
    const auto tableAppended = AppendQuotedIdentifier(sql, *physical.value);
    if (!tableAppended.ok()) {
        return {tableAppended.code, std::nullopt, tableAppended.affectedRows, tableAppended.diagnostic};
    }
    sql += " SET ";
    for (std::size_t i = 0; i < update.size(); ++i) {
        if (i > 0) {
            sql += ", ";
        }
        const auto column = AppendQuotedIdentifier(sql, meta.fields[update[i].fieldId].name);
        if (!column.ok()) {
            return {column.code, std::nullopt, column.affectedRows, column.diagnostic};
        }
        sql += " = ";
        sql += plans[i].rhs;
        placeholder += plans[i].binds.size();
    }
    sql += " WHERE ";
    const auto appended = AppendWhereText(where, meta, placeholder, sql);
    if (!appended.ok()) {
        return {appended.code, std::nullopt, appended.affectedRows, appended.diagnostic};
    }
    sqlite3_stmt* raw = nullptr;
    const int prepareRc = sqlite3_prepare_v2(impl_->db, sql.c_str(), -1, &raw, nullptr);
    if (prepareRc != SQLITE_OK) {
        return {
            MapSqliteCode(prepareRc), std::nullopt, 0,
            "sqlite prepare failed: " + std::string(sqlite3_errmsg(impl_->db)) + " [" + sql + "]"};
    }
    StatementFinalizer finalizer(raw);
    int nextIndex = 1;
    for (const auto& plan : plans) {
        for (const auto& value : plan.binds) {
            const auto bound = BindScalar(impl_->db, raw, nextIndex, value);
            if (!bound.ok()) {
                return {bound.code, std::nullopt, bound.affectedRows, bound.diagnostic};
            }
            ++nextIndex;
        }
    }
    const auto boundWhere = BindWhereValues(where, impl_->db, raw, nextIndex);
    if (!boundWhere.ok()) {
        return {boundWhere.code, std::nullopt, boundWhere.affectedRows, boundWhere.diagnostic};
    }
    const int stepRc = sqlite3_step(raw);
    if (stepRc != SQLITE_DONE) {
        return {
            MapSqliteCode(stepRc), std::nullopt, 0,
            "sqlite step failed: " + std::string(sqlite3_errmsg(impl_->db)) + " [" + sql + "]"};
    }
    return {
        DbCode::OK,
        static_cast<uint64_t>(sqlite3_changes(impl_->db)),
        static_cast<uint64_t>(sqlite3_changes(impl_->db)),
        {}};
}

// 即时/事务 DELETE 共用的唯一直通实现（调用前已持锁并完成状态校验）：
// 条件直推 WHERE，changes 即受影响行数。
DbResult<uint64_t> SQLiteEngine::Delete(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
    TransactionBinding* transaction)
{
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (transaction == nullptr) {
        auto waited = impl_->WaitForTransactionSlot(lock);
        if (!waited.ok()) {
            return {waited.code, std::nullopt, waited.affectedRows, waited.diagnostic};
        }
    }
    auto state = CheckDmlState(transaction);
    if (!state.ok()) {
        return {state.code, std::nullopt, state.affectedRows, state.diagnostic};
    }
    return DeleteLocked(meta, scope, where);
}

DbResult<uint64_t>
SQLiteEngine::DeleteLocked(const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where)
{
    if (std::holds_alternative<std::monostate>(where)) {
        return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "delete requires a primary key or conditions"};
    }
    auto physical = impl_->ResolvePhysicalTable(meta, scope, true);
    if (!physical.ok()) {
        return {physical.code, std::nullopt, physical.affectedRows, physical.diagnostic};
    }
    std::size_t placeholder = 1;
    std::string sql;
    sql.reserve(48);
    sql += "DELETE FROM ";
    const auto tableAppended = AppendQuotedIdentifier(sql, *physical.value);
    if (!tableAppended.ok()) {
        return {tableAppended.code, std::nullopt, tableAppended.affectedRows, tableAppended.diagnostic};
    }
    sql += " WHERE ";
    const auto appended = AppendWhereText(where, meta, placeholder, sql);
    if (!appended.ok()) {
        return {appended.code, std::nullopt, appended.affectedRows, appended.diagnostic};
    }
    sqlite3_stmt* raw = nullptr;
    const int prepareRc = sqlite3_prepare_v2(impl_->db, sql.c_str(), -1, &raw, nullptr);
    if (prepareRc != SQLITE_OK) {
        return {
            MapSqliteCode(prepareRc), std::nullopt, 0,
            "sqlite prepare failed: " + std::string(sqlite3_errmsg(impl_->db)) + " [" + sql + "]"};
    }
    StatementFinalizer finalizer(raw);
    int nextIndex = 1;
    const auto bound = BindWhereValues(where, impl_->db, raw, nextIndex);
    if (!bound.ok()) {
        return {bound.code, std::nullopt, bound.affectedRows, bound.diagnostic};
    }
    const int stepRc = sqlite3_step(raw);
    if (stepRc != SQLITE_DONE) {
        return {
            MapSqliteCode(stepRc), std::nullopt, 0,
            "sqlite step failed: " + std::string(sqlite3_errmsg(impl_->db)) + " [" + sql + "]"};
    }
    return {
        DbCode::OK,
        static_cast<uint64_t>(sqlite3_changes(impl_->db)),
        static_cast<uint64_t>(sqlite3_changes(impl_->db)),
        {}};
}

// 清空单表：KEEP/RESET_SEQUENCE 走 DELETE FROM；RESET 追加 sqlite_sequence
// 序列清理；DROP_DYNAMIC_SCOPE 枚举并销毁该基名的全部物理 scope 表。
DbResult<void> SQLiteEngine::Clear(const Ddl::CatalogTableMeta& meta, const TableScope& scope, ClearMode mode)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto state = CheckOperationState();
    if (!state.ok()) {
        return state;
    }
    if (mode == ClearMode::DROP_DYNAMIC_SCOPE && meta.scope != Ddl::ScopeKind::DynamicPid) {
        return InvalidArgument("drop-scope clear requires a dynamic table");
    }
    auto physical = impl_->ResolvePhysicalTable(meta, scope, false);
    if (!physical.ok()) {
        return {physical.code, physical.affectedRows, physical.diagnostic};
    }
    if (mode == ClearMode::DROP_DYNAMIC_SCOPE) {
        // 公开合同：销毁该动态表基名的全部物理 scope 表（<base>_P_<pid>）。
        // 逐个枚举 sqlite_master 后按物理表名 DROP；计数值即被销毁的物理 scope 表数。
        const std::string base{meta.tableId};
        // LIKE 中 '_' 是通配符：用 ESCAPE 转义后按字面量匹配基名后缀 _P_。
        const std::string listSql
            = "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE ? ESCAPE '\\' ORDER BY name";
        const std::string pattern = base + "\\_P\\_%";
        sqlite3_stmt* raw = nullptr;
        if (sqlite3_prepare_v2(impl_->db, listSql.c_str(), -1, &raw, nullptr) != SQLITE_OK) {
            return {
                MapSqliteCode(sqlite3_errcode(impl_->db)), 0,
                "sqlite prepare failed: " + std::string(sqlite3_errmsg(impl_->db))};
        }
        StatementFinalizer listFinalizer(raw);
        (void)sqlite3_bind_text(raw, 1, pattern.c_str(), -1, SQLITE_TRANSIENT);
        std::vector<std::string> physicalTables;
        while (sqlite3_step(raw) == SQLITE_ROW) {
            const unsigned char* name = sqlite3_column_text(raw, 0);
            if (name != nullptr) {
                physicalTables.emplace_back(reinterpret_cast<const char*>(name));
            }
        }
        uint64_t droppedScopes = 0;
        for (const auto& table : physicalTables) {
            auto dropped
                = ExecuteSimple(impl_->db, "DROP TABLE IF EXISTS \"" + table + "\"", "drop dynamic scope table");
            if (!dropped.ok()) {
                return dropped;
            }
            impl_->ensuredDynamicTables.erase(table);
            ++droppedScopes;
        }
        return {DbCode::OK, droppedScopes, {}};
    }
    auto table = QuoteIdentifier(*physical.value);
    if (!table.ok()) {
        return {table.code, table.affectedRows, table.diagnostic};
    }
    auto cleared = ExecuteSimple(impl_->db, "DELETE FROM " + *table.value, "clear table");
    if (!cleared.ok()) {
        return cleared;
    }
    if (mode == ClearMode::RESET_SEQUENCE) {
        // sqlite_sequence 不存在（尚无自增表）时按主线语义静默跳过序列清理。
        sqlite3_stmt* raw = nullptr;
        if (sqlite3_prepare_v2(impl_->db, "DELETE FROM sqlite_sequence WHERE name = ?", -1, &raw, nullptr)
            == SQLITE_OK) {
            StatementFinalizer sequenceFinalizer(raw);
            (void)sqlite3_bind_text(raw, 1, physical.value->c_str(), -1, SQLITE_TRANSIENT);
            (void)sqlite3_step(raw);
        }
    }
    return {DbCode::OK, cleared.affectedRows, {}};
}

// 列出当前库的用户表（sqlite_ 前缀系统表不对外暴露）。
DbResult<std::vector<std::string>> SQLiteEngine::ListTables()
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto state = CheckOperationState(/*allowBackupOnly*/ false);
    if (!state.ok()) {
        return {state.code, std::nullopt, state.affectedRows, state.diagnostic};
    }
    const std::string sql
        = "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name";
    sqlite3_stmt* raw = nullptr;
    if (sqlite3_prepare_v2(impl_->db, sql.c_str(), -1, &raw, nullptr) != SQLITE_OK) {
        return {
            MapSqliteCode(sqlite3_errcode(impl_->db)), std::nullopt, 0,
            "sqlite prepare failed: " + std::string(sqlite3_errmsg(impl_->db))};
    }
    StatementFinalizer finalizer(raw);
    std::vector<std::string> tables;
    while (true) {
        const int rc = sqlite3_step(raw);
        if (rc == SQLITE_ROW) {
            const unsigned char* text = sqlite3_column_text(raw, 0);
            const int size = sqlite3_column_bytes(raw, 0);
            tables.emplace_back(reinterpret_cast<const char*>(text), static_cast<std::size_t>(size));
            continue;
        }
        if (rc == SQLITE_DONE) {
            break;
        }
        return {MapSqliteCode(rc), std::nullopt, 0, "sqlite step failed: " + std::string(sqlite3_errmsg(impl_->db))};
    }
    return {DbCode::OK, std::optional<std::vector<std::string>>{std::move(tables)}, tables.size(), {}};
}

// ---- 物理事务边界（直达模型：DML 已即时执行，本层只处理 BEGIN/COMMIT/ROLLBACK）----

// 开启一个物理事务：BEGIN IMMEDIATE 后同一 binding 的 DML 直达执行。
// 只做连接状态与唯一活动事务校验，不解析表、不创建计划、不保存业务数据。
// 单连接上的第二笔事务在其他线程的活动事务期间串行等待（而非瞬态失败）；
// 同线程保持立即失败（防自锁）。
DbResult<void> SQLiteEngine::Begin(TransactionBinding& transaction)
{
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (impl_->state == Impl::State::NOT_OPEN || impl_->state == Impl::State::CLOSED) {
        return InvalidState("engine is not open");
    }
    if (impl_->state == Impl::State::OPENED) {
        return InvalidState("engine is not prepared");
    }
    if (impl_->state == Impl::State::TRANSACTION_BUILDING) {
        auto waited = impl_->WaitForTransactionSlot(lock);
        if (!waited.ok()) {
            return waited;
        }
        // 被唤醒后重新完整校验状态（不能假定仍为 PREPARED：可能已被关闭）
        if (impl_->state == Impl::State::NOT_OPEN || impl_->state == Impl::State::CLOSED) {
            return InvalidState("engine is not open");
        }
        if (impl_->state == Impl::State::OPENED) {
            return InvalidState("engine is not prepared");
        }
    }
    if (impl_->state != Impl::State::PREPARED) {
        return InvalidState("another transaction is active on this route");
    }
    if (impl_->mode == StorageOpenMode::BACKUP_ONLY) {
        return InvalidState("backup-only engine does not accept transactions");
    }
    if (impl_->db == nullptr) {
        return SqliteError(DbCode::INVALID_STATE, "sqlite connection is not open");
    }
    auto begun = ExecuteSimple(impl_->db, "BEGIN IMMEDIATE", "begin");
    if (!begun.ok()) {
        // BEGIN 失败不改变连接状态（仍为 PREPARED），活动槽保持为空。
        return begun;
    }
    impl_->activeBinding = &transaction;
    impl_->transactionOwner = std::this_thread::get_id();
    impl_->state = Impl::State::TRANSACTION_BUILDING;
    return {DbCode::OK, 0, {}};
}

// 提交当前物理事务：只执行 COMMIT，不解释/不重放任何 DML（结果由 Transaction
// 按执行顺序记录）。提交失败尝试 ROLLBACK 并清理活动槽；回滚也失败时沿用
// 故障合同关闭连接，不得复用不确定状态的连接。
DbResult<AtomicResult> SQLiteEngine::Commit(TransactionBinding& transaction)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->activeBinding != &transaction || impl_->state != Impl::State::TRANSACTION_BUILDING) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "no matching transaction is active on this route"};
    }
    auto committed = ExecuteSimple(impl_->db, "COMMIT", "commit");
    impl_->activeBinding = nullptr;
    impl_->state = Impl::State::PREPARED;
    impl_->NotifyTransactionSlotReleasedLocked(); // 事务槽已释放，唤醒等待者
    if (!committed.ok()) {
        auto rolled = RollbackPhysicalLocked();
        if (!rolled.ok()) {
            (void)CloseLocked();
            impl_->state = Impl::State::CLOSED;
            impl_->NotifyTransactionSlotReleasedLocked(); // 连接已关，等待者按 CLOSED 处理
            return {
                committed.code, std::nullopt, committed.affectedRows,
                committed.diagnostic + "; rollback after failed commit also failed: " + rolled.diagnostic
                    + " [connection closed]"};
        }
        return {committed.code, std::nullopt, committed.affectedRows, committed.diagnostic};
    }
    return {DbCode::OK, AtomicResult{}, committed.affectedRows, {}};
}

// 回滚当前物理事务：执行 ROLLBACK 并清理活动槽（显式回滚、析构兜底与
// Transaction 在 DML 失败后的统一回滚共用本入口）。回滚失败沿用故障合同
// 关闭连接，不得复用不确定状态的连接。
DbResult<void> SQLiteEngine::Rollback(TransactionBinding& transaction)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->activeBinding != &transaction || impl_->state != Impl::State::TRANSACTION_BUILDING) {
        return InvalidState("no matching transaction is active on this route");
    }
    auto rolled = RollbackPhysicalLocked();
    if (!rolled.ok()) {
        (void)CloseLocked();
        impl_->activeBinding = nullptr;
        impl_->state = Impl::State::CLOSED;
        impl_->NotifyTransactionSlotReleasedLocked(); // 连接已关，等待者按 CLOSED 处理
        return {rolled.code, 0, rolled.diagnostic + " [connection closed]"};
    }
    impl_->activeBinding = nullptr;
    impl_->state = Impl::State::PREPARED;
    impl_->NotifyTransactionSlotReleasedLocked(); // 事务槽已释放，唤醒等待者
    return {DbCode::OK, 0, {}};
}

// ROLLBACK 物理边界；对无活动事务幂等成功（"no transaction is active" 诊断，
// 覆盖 SQLite 已自动回滚后进入清理的情况）。两种成功结局下事务内容都已不在
// 库内：事务内首次创建的动态表 DDL 随之撤销，ensuredDynamicTables 同步失效，
// 否则后续访问跳过 DDL 出现 no such table。回滚失败沿用故障合同由调用方关闭
// 连接（CloseLocked 清空集合），本函数不吞掉不确定状态。
DbResult<void> SQLiteEngine::RollbackPhysicalLocked()
{
    if (impl_->db == nullptr) {
        return SqliteError(DbCode::INVALID_STATE, "sqlite connection is not open");
    }
    char* message = nullptr;
    const int rc = sqlite3_exec(impl_->db, "ROLLBACK", nullptr, nullptr, &message);
    const std::string diagnostic = message == nullptr ? std::string{} : std::string(message);
    if (message != nullptr) {
        sqlite3_free(message);
    }
    if (rc == SQLITE_OK || diagnostic.find("no transaction is active") != std::string::npos) {
        impl_->ensuredDynamicTables.clear();
        return {DbCode::OK, 0, {}};
    }
    return SqliteError(MapSqliteCode(rc), "sqlite rollback failed: " + diagnostic);
}

// 释放物理连接（持锁路径；Prepare 的 rollback 失败收尾复用）。
DbResult<void> SQLiteEngine::CloseLocked()
{
    if (impl_->db == nullptr) {
        return {DbCode::OK, 0, {}};
    }
    const int rc = sqlite3_close(impl_->db);
    impl_->db = nullptr;
    impl_->ensuredDynamicTables.clear();
    if (rc != SQLITE_OK) {
        return SqliteError(MapSqliteCode(rc), "sqlite close failed: " + std::string(sqlite3_errstr(rc)));
    }
    return {DbCode::OK, 0, {}};
}

// 关系型备份：生命周期与目标路径检查后经 sqlite3_backup 逐页拷贝。
DbResult<DatabaseBackupArtifact> SQLiteEngine::BackupRelational(const EngineBackupRequest& request)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->state == Impl::State::NOT_OPEN || impl_->state == Impl::State::CLOSED) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "engine is not open"};
    }
    if (impl_->state == Impl::State::OPENED) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "engine is not prepared"};
    }
    if (impl_->state == Impl::State::TRANSACTION_BUILDING) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "a transaction is building on this route"};
    }
    if (request.destinationPath.empty()) {
        return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "backup destination path is empty"};
    }
    if (impl_->db == nullptr) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "sqlite connection is not open"};
    }
    sqlite3* destination = nullptr;
    int rc = sqlite3_open(request.destinationPath.c_str(), &destination);
    if (rc != SQLITE_OK) {
        const std::string message = destination == nullptr ? sqlite3_errstr(rc) : sqlite3_errmsg(destination);
        if (destination != nullptr) {
            sqlite3_close(destination);
        }
        return {DbCode::IO_ERROR, std::nullopt, 0, "failed to open sqlite backup destination: " + message};
    }
    // 备份目标连接此前遗漏速度设置（使用连接默认 synchronous）。备份是可丢弃
    // 数据的整库复制：目标句柄单独显式 synchronous=OFF 并核验生效（不把源
    // 连接的整组 WAL/busy/foreign_keys 配置强灌给目标——目标在备份期间只有
    // sqlite3_backup 写入，不执行业务 DML）。设置或核验失败关闭句柄返回失败，
    // 不继续备份、不泄漏句柄；备份内容与结果合同不变（仍用 backup_init/step/
    // finish，成功后目标可重新打开读出完整结果）。
    {
        const auto applied = ApplyConnectionPragma(destination, "PRAGMA synchronous=OFF", impl_->profile);
        if (!applied.ok()) {
            const std::string diagnostic = "failed to set backup destination synchronous: " + applied.diagnostic;
            sqlite3_close(destination);
            return {applied.code, std::nullopt, 0, diagnostic};
        }
        const auto actual = ReadPragma(destination, "synchronous");
        if (!actual.ok() || *actual.value != "0") {
            const std::string diagnostic
                = "sqlite backup destination pragma did not take effect: synchronous"
                  + (actual.ok() ? (" expected=0 actual=" + *actual.value) : (" read failed: " + actual.diagnostic));
            sqlite3_close(destination);
            return {DbCode::IO_ERROR, std::nullopt, 0, diagnostic};
        }
    }
    sqlite3_backup* backup = sqlite3_backup_init(destination, "main", impl_->db, "main");
    if (backup == nullptr) {
        rc = sqlite3_errcode(destination);
        const DbCode code = MapSqliteCode(rc);
        const std::string diagnostic = "sqlite backup init failed: " + std::string(sqlite3_errmsg(destination));
        sqlite3_close(destination);
        return {code, std::nullopt, 0, diagnostic};
    }
    do {
        rc = sqlite3_backup_step(backup, -1);
    } while (rc == SQLITE_OK || rc == SQLITE_BUSY || rc == SQLITE_LOCKED);
    const int finishRc = sqlite3_backup_finish(backup);
    sqlite3_close(destination);
    const int finalRc = finishRc != SQLITE_OK ? finishRc : (rc == SQLITE_DONE ? SQLITE_OK : rc);
    if (finalRc != SQLITE_OK) {
        return {
            MapSqliteCode(finalRc), std::nullopt, 0, "sqlite backup failed: " + std::string(sqlite3_errstr(finalRc))};
    }
    return {DbCode::OK, DatabaseBackupArtifact{request.destinationPath}, 0, {}};
}

// 显式关闭：先丢弃未提交事务（不隐式提交、不隐式备份），再释放物理连接；Close 为终态。
DbResult<void> SQLiteEngine::Close()
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->state == Impl::State::NOT_OPEN) {
        return InvalidState("engine is not open");
    }
    if (impl_->state == Impl::State::CLOSED) {
        return InvalidState("engine is already closed");
    }
    if (impl_->state == Impl::State::TRANSACTION_BUILDING) {
        (void)RollbackPhysicalLocked();
    }
    auto closed = CloseLocked();
    if (!closed.ok()) {
        return closed;
    }
    impl_->activeBinding = nullptr;
    impl_->state = Impl::State::CLOSED;
    impl_->NotifyTransactionSlotReleasedLocked(); // 唤醒等待者按 CLOSED 有限返回
    return {DbCode::OK, 0, {}};
}

// 即时操作公共状态检查（持锁路径内联使用）。
DbResult<void> SQLiteEngine::CheckOperationState(bool allowBackupOnly)
{
    if (impl_->state == Impl::State::NOT_OPEN || impl_->state == Impl::State::CLOSED) {
        return InvalidState("engine is not open");
    }
    if (impl_->state == Impl::State::OPENED) {
        return InvalidState("engine is not prepared");
    }
    if (impl_->state == Impl::State::TRANSACTION_BUILDING) {
        return InvalidState("a transaction is building on this route");
    }
    if (!allowBackupOnly && impl_->mode == StorageOpenMode::BACKUP_ONLY) {
        return InvalidState("backup-only engine does not accept operations");
    }
    if (impl_->db == nullptr) {
        return SqliteError(DbCode::INVALID_STATE, "sqlite connection is not open");
    }
    return {DbCode::OK, 0, {}};
}

// DML 状态校验：transaction 为空要求 PREPARED（普通 DML 不得混入活动事务的
// 连接）；非空要求已 Begin 且 binding 匹配。DML 只执行并返回错误，失败回滚
// 由 Transaction 统一调用 Rollback 完成。
DbResult<void> SQLiteEngine::CheckDmlState(TransactionBinding* transaction)
{
    if (transaction == nullptr) {
        return CheckOperationState();
    }
    if (impl_->state == Impl::State::NOT_OPEN || impl_->state == Impl::State::CLOSED) {
        return InvalidState("engine is not open");
    }
    if (impl_->state == Impl::State::OPENED) {
        return InvalidState("engine is not prepared");
    }
    if (impl_->activeBinding != transaction || impl_->state != Impl::State::TRANSACTION_BUILDING) {
        return InvalidState("no matching transaction is active on this route");
    }
    if (impl_->db == nullptr) {
        return SqliteError(DbCode::INVALID_STATE, "sqlite connection is not open");
    }
    return {DbCode::OK, 0, {}};
}

// schema DDL 观察快照（仅供单元测试；DDL 路径计数，不在数据热路径）
SQLiteEngine::SchemaDdlSnapshot SQLiteEngine::SchemaDdlStatistics() const
{
    SchemaDdlSnapshot snapshot;
    snapshot.staticDdlExecutions = impl_->staticDdlExecutions;
    snapshot.dynamicTableCreates = impl_->dynamicTableCreates;
    return snapshot;
}

// 引擎实际连接上的 PRAGMA 读取（仅供单元测试；连接级 PRAGMA 的生效证明
// 必须来自同一连接的读数）
DbResult<std::string> SQLiteEngine::ReadConnectionPragmaForTest(const char* name) const
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->db == nullptr) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "sqlite connection is not open"};
    }
    return ReadPragma(impl_->db, name);
}

// 事务等待上限测试可见性（仅供测试 peer；持锁写入私有上限）
void SQLiteEngine::SetTransactionWaitLimitForTest(std::chrono::milliseconds limit)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->transactionWaitLimit = limit;
}
} // namespace HcclSim::Storage
