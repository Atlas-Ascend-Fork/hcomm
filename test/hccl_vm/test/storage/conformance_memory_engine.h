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
 * @file conformance_memory_engine.h
 * @brief 后端 conformance 的内存语义引擎（Fake 后端）。
 *
 * 按 IStorageEngine 显式能力合同实现关系语义（平铺条件/排序/limit/count/update/
 * delete/clear/自增/唯一/级联/事务），消费统一 StorageCatalog 的 CatalogTableMeta
 * 字段通道（visitRecord 原始值收集 / materializeRows 最终目标直填）；
 * 不包含任何 SQL、PRAGMA、文件或缓存行为。仅用于验证公共 conformance case 的
 * fixture 协议与复制合同的观察点；后端专有行为不在此实现。
 * 事务模型：Begin 登记活动 binding，DML 立即执行（与即时路径共用实现），
 * Commit/Rollback 只清理事务槽。
 */
#ifndef HCCL_VM_STORAGE_CONFORMANCE_MEMORY_ENGINE_H
#define HCCL_VM_STORAGE_CONFORMANCE_MEMORY_ENGINE_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "storage/engine/i_storage_engine.h"
#include "storage/internal/storage_runtime_impl.h"
#include "storage/storage_catalog.h"
#include "storage/storage_runtime.h"

namespace storage_conformance {

/// 测试内存引擎的模拟行（测试私有；生产中间行表示已随 Redis 后端移除，
/// CRUD 经 visitRecord/materializeRows 原始引用通道直通）：仅承载
/// fieldId → DbScalar 的字段值表，按目录声明顺序收集。
struct MemoryRow {
    std::vector<std::pair<uint16_t, HcclSim::Storage::DbScalar>> fields;
};

/// 内存语义引擎：每实例持有独立数据（fixture 每次会话新建实例，用例间零数据耦合）
class ConformanceMemoryEngine final : public HcclSim::Storage::IStorageEngine {
public:
    /// 工厂入口：绑定 conformance 目录（表元数据来自 CatalogTableMeta）
    static HcclSim::Storage::DbResult<std::unique_ptr<HcclSim::Storage::IStorageEngine>>
    Create(const HcclSim::Storage::Ddl::DatabaseDescriptor* catalog)
    {
        if (catalog == nullptr) {
            return {HcclSim::Storage::DbCode::INVALID_ARGUMENT, std::nullopt, 0, "catalog is null"};
        }
        return {
            HcclSim::Storage::DbCode::OK,
            std::optional<std::unique_ptr<HcclSim::Storage::IStorageEngine>>{
                std::unique_ptr<HcclSim::Storage::IStorageEngine>(new ConformanceMemoryEngine(catalog))},
            0,
            {}};
    }

    HcclSim::Storage::DbResult<void> Open(const HcclSim::Storage::EngineOpenRequest&) override { return Ok(); }

    HcclSim::Storage::DbResult<void> Prepare() override { return Ok(); }

    HcclSim::Storage::DbResult<uint64_t> Insert(
        const HcclSim::Storage::Ddl::CatalogTableMeta& meta, const HcclSim::Storage::TableScope& scope,
        const void* record, HcclSim::Storage::TransactionBinding* transaction) override
    {
        const auto state = CheckDmlState(transaction);
        if (!state.ok()) {
            return {state.code, std::nullopt, state.affectedRows, state.diagnostic};
        }
        lastInsertRecordAddress_ = record; // 复制合同观察点：即时/事务路径均零复制
        return ApplyInsert(meta, scope, record);
    }

    HcclSim::Storage::DbResult<void> Select(
        const HcclSim::Storage::Ddl::CatalogTableMeta& meta, const HcclSim::Storage::TableScope& scope,
        const HcclSim::Storage::ConditionData& where, const HcclSim::Storage::QueryOptions& options,
        HcclSim::Storage::ResultSink& sink) override
    {
        ++selectCallsByTable_[std::string{meta.tableId}]; // 查询次数观察点
        if (meta.materializeRows == nullptr) {
            return {HcclSim::Storage::DbCode::INVALID_ARGUMENT, 0, "materializer is missing"};
        }
        auto& table = TableOf(meta.tableId, scope);
        std::vector<MemoryRow> matched;
        for (const auto& row : table.rows) {
            if (ConditionMatches(where, row)) {
                matched.push_back(row);
            }
        }
        SortRows(matched, options.sorts);
        if (options.limit.has_value() && matched.size() > *options.limit) {
            matched.resize(static_cast<std::size_t>(*options.limit));
        }
        RowCursor cursor{std::move(matched)};
        cursor.meta = &meta;
        return meta.materializeRows(sink, &cursor, RowCursor::Next, RowCursor::ReadField);
    }

    HcclSim::Storage::DbResult<uint64_t> Count(
        const HcclSim::Storage::Ddl::CatalogTableMeta& meta, const HcclSim::Storage::TableScope& scope,
        const HcclSim::Storage::ConditionData& where) override
    {
        uint64_t count = 0;
        for (const auto& row : TableOf(meta.tableId, scope).rows) {
            if (ConditionMatches(where, row)) {
                ++count;
            }
        }
        return {HcclSim::Storage::DbCode::OK, count, 1, {}};
    }

    /// 条件计数聚合（内存语义实现）：在稳定表状态上直接累计——total 为
    /// where 匹配行数，matched 为 where 与 match 同时匹配行数。单次遍历，
    /// 两个计数天然来自同一快照（与真实后端单语句合同等价）。
    HcclSim::Storage::DbResult<HcclSim::Storage::CountMatchResult> CountMatches(
        const HcclSim::Storage::Ddl::CatalogTableMeta& meta, const HcclSim::Storage::TableScope& scope,
        const HcclSim::Storage::ConditionData& where, const HcclSim::Storage::ConditionData& match) override
    {
        HcclSim::Storage::CountMatchResult counted;
        for (const auto& row : TableOf(meta.tableId, scope).rows) {
            if (ConditionMatches(where, row)) {
                ++counted.total;
                if (ConditionMatches(match, row)) {
                    ++counted.matched;
                }
            }
        }
        return {HcclSim::Storage::DbCode::OK, counted, 1, {}};
    }

    HcclSim::Storage::DbResult<uint64_t> Update(
        const HcclSim::Storage::Ddl::CatalogTableMeta& meta, const HcclSim::Storage::TableScope& scope,
        const HcclSim::Storage::ConditionData& where, const HcclSim::Storage::UpdateData& update,
        HcclSim::Storage::TransactionBinding* transaction) override
    {
        const auto state = CheckDmlState(transaction);
        if (!state.ok()) {
            return {state.code, std::nullopt, state.affectedRows, state.diagnostic};
        }
        if (update.empty() || std::holds_alternative<std::monostate>(where)) {
            return {
                HcclSim::Storage::DbCode::INVALID_ARGUMENT, std::nullopt, 0, "update requires changes and a condition"};
        }
        for (const auto& change : update) {
            if (change.fieldId >= meta.fieldCount || meta.fields[change.fieldId].primaryKey) {
                return {
                    HcclSim::Storage::DbCode::INVALID_ARGUMENT, std::nullopt, 0,
                    "update field is not a writable declared column"};
            }
        }
        // 先计算后应用：INCREMENT 的域检查（非法上下溢/类型不匹配/缺行值）
        // 使整条 Update 失败且全部行值保持——对齐真实后端单语句原子语义，
        // 不产生部分更新。无符号字段接受负 int64 增量并按幅值减法（与
        // MariaDB/SQLite 修复后行为一致）；int64 加法显式溢出检查，
        // 不依赖回绕（UB）。幅值经无符号回绕取得，禁止对 INT64_MIN 取负。
        auto& rows = TableOf(meta.tableId, scope).rows;
        uint64_t affected = 0;
        std::vector<std::vector<std::pair<uint16_t, HcclSim::Storage::DbScalar>>> staged;
        staged.resize(rows.size());
        for (std::size_t rowIndex = 0; rowIndex < rows.size(); ++rowIndex) {
            auto& row = rows[rowIndex];
            if (!ConditionMatches(where, row)) {
                continue;
            }
            ++affected;
            for (const auto& change : update) {
                const HcclSim::Storage::DbScalar* current = nullptr;
                for (auto& field : row.fields) {
                    if (field.first == change.fieldId) {
                        current = &field.second;
                        break;
                    }
                }
                if (change.kind == HcclSim::Storage::ChangeKind::INCREMENT) {
                    if (current == nullptr) {
                        return {
                            HcclSim::Storage::DbCode::INVALID_ARGUMENT, std::nullopt, 0,
                            "increment requires the current row value"};
                    }
                    HcclSim::Storage::DbScalar next;
                    if (const auto* number = std::get_if<uint64_t>(current); number != nullptr) {
                        bool negative = false;
                        uint64_t magnitude = 0;
                        if (const auto* delta = std::get_if<int64_t>(&change.value); delta != nullptr) {
                            negative = *delta < 0;
                            magnitude = negative ? (uint64_t{0} - static_cast<uint64_t>(*delta)) :
                                                   static_cast<uint64_t>(*delta);
                        } else if (const auto* delta = std::get_if<uint64_t>(&change.value); delta != nullptr) {
                            magnitude = *delta;
                        } else {
                            return {
                                HcclSim::Storage::DbCode::INVALID_ARGUMENT, std::nullopt, 0,
                                "increment value type does not match the field"};
                        }
                        if (negative ? (*number < magnitude) : (magnitude > UINT64_MAX - *number)) {
                            return {
                                HcclSim::Storage::DbCode::INVALID_ARGUMENT, std::nullopt, 0,
                                "increment would drive the unsigned field out of range"};
                        }
                        next = HcclSim::Storage::DbScalar{negative ? (*number - magnitude) : (*number + magnitude)};
                    } else if (const auto* signedNumber = std::get_if<int64_t>(current); signedNumber != nullptr) {
                        const auto* delta = std::get_if<int64_t>(&change.value);
                        if (delta == nullptr) {
                            return {
                                HcclSim::Storage::DbCode::INVALID_ARGUMENT, std::nullopt, 0,
                                "increment value type does not match the field"};
                        }
                        if ((*delta > 0 && *signedNumber > INT64_MAX - *delta)
                            || (*delta < 0 && *signedNumber < INT64_MIN - *delta)) {
                            return {
                                HcclSim::Storage::DbCode::INVALID_ARGUMENT, std::nullopt, 0,
                                "increment would drive the signed field out of range"};
                        }
                        next = HcclSim::Storage::DbScalar{*signedNumber + *delta};
                    } else if (const auto* realNumber = std::get_if<double>(current); realNumber != nullptr) {
                        const auto* delta = std::get_if<double>(&change.value);
                        if (delta == nullptr) {
                            return {
                                HcclSim::Storage::DbCode::INVALID_ARGUMENT, std::nullopt, 0,
                                "increment value type does not match the field"};
                        }
                        next = HcclSim::Storage::DbScalar{*realNumber + *delta};
                    } else {
                        return {
                            HcclSim::Storage::DbCode::INVALID_ARGUMENT, std::nullopt, 0,
                            "increment requires a numeric field"};
                    }
                    staged[rowIndex].emplace_back(change.fieldId, next);
                } else {
                    staged[rowIndex].emplace_back(change.fieldId, change.value);
                }
            }
        }
        for (std::size_t rowIndex = 0; rowIndex < rows.size(); ++rowIndex) {
            for (auto& stagedField : staged[rowIndex]) {
                bool applied = false;
                for (auto& field : rows[rowIndex].fields) {
                    if (field.first == stagedField.first) {
                        field.second = stagedField.second;
                        applied = true;
                        break;
                    }
                }
                if (!applied) {
                    rows[rowIndex].fields.emplace_back(stagedField.first, stagedField.second);
                }
            }
        }
        return {HcclSim::Storage::DbCode::OK, affected, affected, {}};
    }

    HcclSim::Storage::DbResult<uint64_t> Delete(
        const HcclSim::Storage::Ddl::CatalogTableMeta& meta, const HcclSim::Storage::TableScope& scope,
        const HcclSim::Storage::ConditionData& where, HcclSim::Storage::TransactionBinding* transaction) override
    {
        const auto state = CheckDmlState(transaction);
        if (!state.ok()) {
            return {state.code, std::nullopt, state.affectedRows, state.diagnostic};
        }
        if (std::holds_alternative<std::monostate>(where)) {
            return {HcclSim::Storage::DbCode::INVALID_ARGUMENT, std::nullopt, 0, "delete requires a condition"};
        }
        auto& table = TableOf(meta.tableId, scope);
        std::vector<MemoryRow> removed;
        table.rows.erase(
            std::remove_if(
                table.rows.begin(), table.rows.end(),
                [&where, &removed](const MemoryRow& row) {
                    if (ConditionMatches(where, row)) {
                        removed.push_back(row);
                        return true;
                    }
                    return false;
                }),
            table.rows.end());
        for (const auto& parentRow : removed) {
            CascadeDelete(meta, parentRow, scope);
        }
        return {HcclSim::Storage::DbCode::OK, removed.size(), removed.size(), {}};
    }

    HcclSim::Storage::DbResult<void> Clear(
        const HcclSim::Storage::Ddl::CatalogTableMeta& meta, const HcclSim::Storage::TableScope& scope,
        HcclSim::Storage::ClearMode mode) override
    {
        if (mode == HcclSim::Storage::ClearMode::DROP_DYNAMIC_SCOPE
            && meta.scope != HcclSim::Storage::Ddl::ScopeKind::DynamicPid) {
            return {HcclSim::Storage::DbCode::INVALID_ARGUMENT, 0, "drop-scope clear requires a dynamic table"};
        }
        auto& table = TableOf(meta.tableId, scope);
        const std::size_t dropped = table.rows.size();
        table.rows.clear();
        if (mode == HcclSim::Storage::ClearMode::RESET_SEQUENCE) {
            table.nextId = 1;
        }
        return {HcclSim::Storage::DbCode::OK, dropped, {}};
    }

    HcclSim::Storage::DbResult<std::vector<std::string>> ListTables() override
    {
        std::vector<std::string> tables;
        for (std::size_t i = 0; i < catalog_->tableCount; ++i) {
            tables.emplace_back(catalog_->tables[i].tableId);
        }
        // 动态表按已访问 scope 物化 <table>_P_<pid> 物理表名
        for (const auto& entry : tables_) {
            const auto sep = entry.first.rfind('#');
            if (sep == std::string::npos) {
                continue;
            }
            const std::string suffix = entry.first.substr(sep + 1);
            if (suffix != "static") {
                tables.push_back(entry.first.substr(0, sep) + "_P_" + suffix);
            }
        }
        std::sort(tables.begin(), tables.end());
        return {
            HcclSim::Storage::DbCode::OK,
            std::optional<std::vector<std::string>>{std::move(tables)},
            tables.size(),
            {}};
    }

    HcclSim::Storage::DbResult<void> Begin(HcclSim::Storage::TransactionBinding& transaction) override
    {
        // 直达事务模型：开启一个物理事务槽（同一时刻仅一个 binding 活动），
        // 并快照当前表状态供 Rollback 恢复（DML 立即执行，回滚靠快照还原）。
        if (activeBinding_ != nullptr) {
            return {HcclSim::Storage::DbCode::INVALID_STATE, 0, "another transaction is active on this route"};
        }
        activeBinding_ = &transaction;
        snapshot_ = tables_;
        return Ok();
    }

    HcclSim::Storage::DbResult<HcclSim::Storage::AtomicResult>
    Commit(HcclSim::Storage::TransactionBinding& transaction) override
    {
        if (activeBinding_ != &transaction) {
            return {
                HcclSim::Storage::DbCode::INVALID_STATE, std::nullopt, 0,
                "no matching transaction is active on this route"};
        }
        activeBinding_ = nullptr;
        snapshot_.clear();
        // DML 已即时执行；AtomicResult 由 Transaction 层按执行顺序记录
        return {HcclSim::Storage::DbCode::OK, HcclSim::Storage::AtomicResult{}, 0, {}};
    }

    HcclSim::Storage::DbResult<void> Rollback(HcclSim::Storage::TransactionBinding& transaction) override
    {
        if (activeBinding_ != &transaction) {
            return {HcclSim::Storage::DbCode::INVALID_STATE, 0, "no matching transaction is active on this route"};
        }
        activeBinding_ = nullptr;
        tables_ = std::move(snapshot_);
        snapshot_.clear();
        return Ok();
    }

    HcclSim::Storage::DbResult<void> Close() override { return Ok(); }

    /// ---- 复制合同观察点（仅测试使用；产品代码不得引用） ----
    /// 最近一次即时 Insert 收到的记录地址（与原对象相同 = 全程零复制）
    const void* LastInsertRecordAddress() const noexcept { return lastInsertRecordAddress_; }

    /// ---- 查询次数观察点（仅测试使用；产品代码不得引用） ----
    /// 按逻辑表名累计 Select 调用次数（业务路径查询减量断言用）
    uint64_t SelectCallsOf(std::string_view tableId) const
    {
        const auto found = selectCallsByTable_.find(std::string{tableId});
        return found == selectCallsByTable_.end() ? 0U : found->second;
    }

private:
    using DbCode = HcclSim::Storage::DbCode;
    using DbScalar = HcclSim::Storage::DbScalar;
    using LogicalRecord = MemoryRow;

    struct TableData {
        std::vector<LogicalRecord> rows;
        uint64_t nextId{1};
    };

    explicit ConformanceMemoryEngine(const HcclSim::Storage::Ddl::DatabaseDescriptor* catalog) : catalog_(catalog) {}

    static HcclSim::Storage::DbResult<void> Ok() { return {DbCode::OK, 0, {}}; }

    /// 表数据键：逻辑表名 + scope（静态表忽略 PID，动态表按 PID 隔离）
    std::string TableKey(std::string_view tableId, const HcclSim::Storage::TableScope& scope) const
    {
        return std::string{tableId} + "#" + (scope.processId.has_value() ? std::to_string(*scope.processId) : "static");
    }

    TableData& TableOf(std::string_view tableId, const HcclSim::Storage::TableScope& scope)
    {
        return tables_[TableKey(tableId, scope)];
    }

    // ---- 原始字段值 → DbScalar（visitRecord 收集器；按 FieldDesc::type + 实际宽度）----

    static DbScalar
    ScalarFromRaw(const HcclSim::Storage::Ddl::FieldDesc& field, const void* data, std::size_t bytes, bool isNull)
    {
        (void)isNull;
        switch (field.type) {
            case HcclSim::Storage::LogicalFieldType::SIGNED: {
                int64_t value = 0;
                std::memcpy(&value, data, std::min(bytes, sizeof(value)));
                return DbScalar{value};
            }
            case HcclSim::Storage::LogicalFieldType::UNSIGNED: {
                uint64_t value = 0;
                std::memcpy(&value, data, std::min(bytes, sizeof(value)));
                return DbScalar{value};
            }
            case HcclSim::Storage::LogicalFieldType::BOOLEAN:
                return DbScalar{*static_cast<const uint8_t*>(data) != 0U};
            case HcclSim::Storage::LogicalFieldType::REAL:
                if (bytes == sizeof(float)) {
                    float value = 0.0F;
                    std::memcpy(&value, data, sizeof(value));
                    return DbScalar{static_cast<double>(value)};
                }
                {
                    double value = 0.0;
                    std::memcpy(&value, data, sizeof(value));
                    return DbScalar{value};
                }
            case HcclSim::Storage::LogicalFieldType::STRING:
                return DbScalar{std::string(static_cast<const char*>(data), bytes)};
            case HcclSim::Storage::LogicalFieldType::BYTES:
            default: {
                const auto* begin = static_cast<const uint8_t*>(data);
                return DbScalar{std::vector<uint8_t>(begin, begin + bytes)};
            }
        }
    }

    /// visitRecord 收集器：按声明顺序把原始字段值收进 LogicalRecord
    struct RecordCollector {
        const HcclSim::Storage::Ddl::CatalogTableMeta* meta{nullptr};
        LogicalRecord row;
        HcclSim::Storage::DbResult<void> status{DbCode::OK, 0, {}};

        static HcclSim::Storage::DbResult<void>
        Emit(void* opaque, uint16_t fieldId, const void* data, std::size_t bytes, bool isNull)
        {
            auto& self = *static_cast<RecordCollector*>(opaque);
            if (fieldId >= self.meta->fieldCount) {
                return {DbCode::INVALID_ARGUMENT, 0, "record emits an undeclared field"};
            }
            self.row.fields.emplace_back(fieldId, ScalarFromRaw(self.meta->fields[fieldId], data, bytes, isNull));
            return {DbCode::OK, 0, {}};
        }
    };

    // ---- 行游标（materializeRows 的 NextRow/ReadFieldInto 实现）----

    struct RowCursor {
        std::vector<LogicalRecord> rows;
        std::size_t cursor{0};
        const LogicalRecord* current{nullptr};
        const HcclSim::Storage::Ddl::CatalogTableMeta* meta{nullptr};

        static HcclSim::Storage::DbResult<bool> Next(void* opaque)
        {
            auto& self = *static_cast<RowCursor*>(opaque);
            while (self.cursor < self.rows.size()) {
                const LogicalRecord& row = self.rows[self.cursor++];
                self.current = &row;
                return {DbCode::OK, true, 0, {}};
            }
            self.current = nullptr;
            return {DbCode::OK, false, 0, {}};
        }

        static HcclSim::Storage::DbResult<void>
        ReadField(void* opaque, uint16_t fieldId, void* destination, std::size_t capacity)
        {
            auto& self = *static_cast<RowCursor*>(opaque);
            if (self.current == nullptr || self.meta == nullptr || fieldId >= self.meta->fieldCount) {
                return {HcclSim::Storage::DbCode::INVALID_STATE, 0, "row cursor has no current row"};
            }
            const DbScalar* value = nullptr;
            for (const auto& field : self.current->fields) {
                if (field.first == fieldId) {
                    value = &field.second;
                    break;
                }
            }
            if (value == nullptr) {
                return {DbCode::OK, 0, {}}; // 缺省字段保持目标默认值
            }
            const auto& field = self.meta->fields[fieldId];
            using Type = HcclSim::Storage::LogicalFieldType;
            switch (field.type) {
                case Type::SIGNED:
                case Type::UNSIGNED: {
                    std::uint64_t bits = 0;
                    if (const auto* number = std::get_if<int64_t>(value); number != nullptr) {
                        std::memcpy(&bits, number, sizeof(bits));
                    } else if (const auto* number = std::get_if<uint64_t>(value); number != nullptr) {
                        std::memcpy(&bits, number, sizeof(bits));
                    }
                    std::memcpy(
                        destination, &bits,
                        std::min(field.byteWidth != 0 ? field.byteWidth : sizeof(bits), sizeof(bits)));
                    return {DbCode::OK, 0, {}};
                }
                case Type::BOOLEAN: {
                    bool flag = false;
                    if (const auto* stored = std::get_if<bool>(value); stored != nullptr) {
                        flag = *stored;
                    } else if (const auto* number = std::get_if<int64_t>(value); number != nullptr) {
                        flag = *number != 0;
                    }
                    std::memcpy(destination, &flag, sizeof(flag));
                    return {DbCode::OK, 0, {}};
                }
                case Type::REAL: {
                    double number = 0.0;
                    if (const auto* stored = std::get_if<double>(value); stored != nullptr) {
                        number = *stored;
                    }
                    if (field.byteWidth == sizeof(float)) {
                        const float narrow = static_cast<float>(number);
                        std::memcpy(destination, &narrow, sizeof(narrow));
                    } else {
                        std::memcpy(destination, &number, sizeof(number));
                    }
                    return {DbCode::OK, 0, {}};
                }
                case Type::STRING: {
                    const auto* text = std::get_if<std::string>(value);
                    if (text == nullptr) {
                        return {DbCode::OK, 0, {}};
                    }
                    if (field.byteWidth == 0) { // std::string：一次拥有化
                        static_cast<std::string*>(destination)->assign(text->data(), text->size());
                        return {DbCode::OK, 0, {}};
                    }
                    const std::size_t bound = capacity != 0 ? capacity : field.byteWidth;
                    const std::size_t count = std::min(text->size(), bound > 0 ? bound - 1 : 0);
                    if (count > 0) {
                        std::memcpy(destination, text->data(), count);
                    }
                    if (bound > 0) {
                        static_cast<char*>(destination)[count] = '\0';
                    }
                    return {DbCode::OK, 0, {}};
                }
                case Type::BYTES:
                default: {
                    const auto* bytes = std::get_if<std::vector<uint8_t>>(value);
                    if (bytes == nullptr) {
                        return {DbCode::OK, 0, {}};
                    }
                    if (field.byteWidth == 0) { // std::vector<uint8_t>：一次拥有化
                        static_cast<std::vector<uint8_t>*>(destination)->assign(bytes->begin(), bytes->end());
                        return {DbCode::OK, 0, {}};
                    }
                    const std::size_t count = std::min(bytes->size(), capacity);
                    if (count > 0) {
                        std::memcpy(destination, bytes->data(), count);
                    }
                    return {DbCode::OK, 0, {}};
                }
            }
        }
    };

    // ---- DbScalar 比较（同字段同类型；IN/范围/前缀按合同） ----
    static bool ScalarEquals(const DbScalar& left, const DbScalar& right)
    {
        if (left.index() != right.index()) {
            return false;
        }
        if (const auto* a = std::get_if<uint64_t>(&left); a != nullptr) {
            return *a == std::get<uint64_t>(right);
        }
        if (const auto* a = std::get_if<int64_t>(&left); a != nullptr) {
            return *a == std::get<int64_t>(right);
        }
        if (const auto* a = std::get_if<double>(&left); a != nullptr) {
            return *a == std::get<double>(right);
        }
        if (const auto* a = std::get_if<bool>(&left); a != nullptr) {
            return *a == std::get<bool>(right);
        }
        if (const auto* a = std::get_if<std::string>(&left); a != nullptr) {
            return *a == std::get<std::string>(right);
        }
        if (const auto* a = std::get_if<std::vector<uint8_t>>(&left); a != nullptr) {
            return *a == std::get<std::vector<uint8_t>>(right);
        }
        return false;
    }

    static bool ScalarLess(const DbScalar& left, const DbScalar& right)
    {
        if (const auto* a = std::get_if<uint64_t>(&left); a != nullptr) {
            return *a < std::get<uint64_t>(right);
        }
        if (const auto* a = std::get_if<int64_t>(&left); a != nullptr) {
            return *a < std::get<int64_t>(right);
        }
        if (const auto* a = std::get_if<double>(&left); a != nullptr) {
            return *a < std::get<double>(right);
        }
        if (const auto* a = std::get_if<bool>(&left); a != nullptr) {
            return static_cast<int>(*a) < static_cast<int>(std::get<bool>(right));
        }
        if (const auto* a = std::get_if<std::string>(&left); a != nullptr) {
            return *a < std::get<std::string>(right);
        }
        return false;
    }

    static bool ClauseMatches(const HcclSim::Storage::ConditionClause& clause, const LogicalRecord& row)
    {
        std::optional<DbScalar> value;
        for (const auto& field : row.fields) {
            if (field.first == clause.fieldId) {
                value = field.second;
                break;
            }
        }
        if (!value.has_value()) {
            return false;
        }
        const auto* singleExpected = std::get_if<DbScalar>(&clause.value);
        const auto* inExpected = std::get_if<std::vector<DbScalar>>(&clause.value);
        const bool hasSingle = singleExpected != nullptr;
        switch (clause.compare) {
            case HcclSim::Storage::Compare::EQUAL:
                return hasSingle && ScalarEquals(*value, *singleExpected);
            case HcclSim::Storage::Compare::NOT_EQUAL:
                return !hasSingle || !ScalarEquals(*value, *singleExpected);
            case HcclSim::Storage::Compare::LESS:
                return hasSingle && ScalarLess(*value, *singleExpected);
            case HcclSim::Storage::Compare::LESS_OR_EQUAL:
                return hasSingle && (ScalarLess(*value, *singleExpected) || ScalarEquals(*value, *singleExpected));
            case HcclSim::Storage::Compare::GREATER:
                return hasSingle && ScalarLess(*singleExpected, *value);
            case HcclSim::Storage::Compare::GREATER_OR_EQUAL:
                return hasSingle && (ScalarLess(*singleExpected, *value) || ScalarEquals(*singleExpected, *value));
            case HcclSim::Storage::Compare::PREFIX: {
                const auto* text = std::get_if<std::string>(&*value);
                if (text == nullptr || !hasSingle) {
                    return false;
                }
                const auto* prefix = std::get_if<std::string>(singleExpected);
                return prefix != nullptr && text->rfind(*prefix, 0) == 0;
            }
            case HcclSim::Storage::Compare::IN: {
                if (inExpected == nullptr || inExpected->empty()) {
                    return false; // 空 IN 恒假（匹配 0 行）
                }
                for (const auto& candidate : *inExpected) {
                    if (ScalarEquals(*value, candidate)) {
                        return true;
                    }
                }
                return false;
            }
            default:
                return false;
        }
    }

    /// 平铺条件匹配：monostate = All；单子句直配；组合按兄弟导航（AND 全真/OR 任一真）
    static bool ConditionMatches(const HcclSim::Storage::ConditionData& where, const LogicalRecord& row)
    {
        if (const auto* clauses = std::get_if<std::vector<HcclSim::Storage::ConditionClause>>(&where);
            clauses != nullptr) {
            if (clauses->empty()) {
                return true;
            }
            return ClauseRangeMatches(*clauses, 0, clauses->size(), row);
        }
        if (const auto* clause = std::get_if<HcclSim::Storage::ConditionClause>(&where); clause != nullptr) {
            return ClauseMatches(*clause, row);
        }
        return true; // monostate = All
    }

    /// 平铺子句的同层兄弟子项（括号计数导航；与公共表达式构造的分组语义一致）
    struct SiblingRange {
        std::size_t begin{0};
        std::size_t end{0};
        bool group{false};
        HcclSim::Storage::ConditionJoin join{HcclSim::Storage::ConditionJoin::NONE};
    };

    static std::vector<SiblingRange>
    SplitSiblings(const std::vector<HcclSim::Storage::ConditionClause>& clauses, std::size_t begin, std::size_t end)
    {
        std::vector<SiblingRange> siblings;
        std::size_t i = begin;
        while (i < end) {
            const std::size_t start = i;
            std::size_t depth = 0;
            do {
                depth += clauses[i].openGroupsBefore;
                depth -= clauses[i].closeGroupsAfter;
                ++i;
            } while (depth > 0 && i < end);
            SiblingRange sibling;
            sibling.begin = start;
            sibling.end = i;
            sibling.group = i - start > 1U;
            sibling.join = clauses[start].joinWithPrevious;
            siblings.push_back(sibling);
        }
        return siblings;
    }

    static bool ClauseRangeMatches(
        const std::vector<HcclSim::Storage::ConditionClause>& clauses, std::size_t begin, std::size_t end,
        const LogicalRecord& row)
    {
        const auto siblings = SplitSiblings(clauses, begin, end);
        if (siblings.size() == 1U && !siblings.front().group) {
            return ClauseMatches(clauses[begin], row);
        }
        if (siblings.size() == 1U && siblings.front().group) {
            // 整段恰为一个分组（如 Or(And(a,b), And(c,d)) 的任一子组单独求值）：
            // 剥掉最外层括号（首子句 open-1 / 末子句 close-1）后按内层同层结构
            // 求值，否则对同一区间的切分结果不变，会无限递归。
            std::vector<HcclSim::Storage::ConditionClause> inner(clauses.begin() + begin, clauses.begin() + end);
            inner.front().openGroupsBefore -= 1U;
            inner.back().closeGroupsAfter -= 1U;
            return ClauseRangeMatches(inner, 0, inner.size(), row);
        }
        const HcclSim::Storage::ConditionJoin join
            = siblings.size() > 1U ? siblings[1].join : HcclSim::Storage::ConditionJoin::AND;
        if (join == HcclSim::Storage::ConditionJoin::AND) {
            for (const auto& sibling : siblings) {
                const bool matched = sibling.group ? ClauseRangeMatches(clauses, sibling.begin, sibling.end, row) :
                                                     ClauseMatches(clauses[sibling.begin], row);
                if (!matched) {
                    return false;
                }
            }
            return true;
        }
        for (const auto& sibling : siblings) {
            const bool matched = sibling.group ? ClauseRangeMatches(clauses, sibling.begin, sibling.end, row) :
                                                 ClauseMatches(clauses[sibling.begin], row);
            if (matched) {
                return true;
            }
        }
        return false;
    }

    /// 排序：多排序键按声明顺序稳定排序（相同键保持插入顺序）
    static void
    SortRows(std::vector<LogicalRecord>& rows, const std::vector<std::pair<uint16_t, HcclSim::Storage::Order>>& sorts)
    {
        if (sorts.empty()) {
            return;
        }
        std::stable_sort(rows.begin(), rows.end(), [&sorts](const LogicalRecord& left, const LogicalRecord& right) {
            for (const auto& sort : sorts) {
                std::optional<DbScalar> leftValue;
                std::optional<DbScalar> rightValue;
                for (const auto& field : left.fields) {
                    if (field.first == sort.first) {
                        leftValue = field.second;
                        break;
                    }
                }
                for (const auto& field : right.fields) {
                    if (field.first == sort.first) {
                        rightValue = field.second;
                        break;
                    }
                }
                if (!leftValue.has_value() || !rightValue.has_value()) {
                    return false;
                }
                if (ScalarEquals(*leftValue, *rightValue)) {
                    continue;
                }
                const bool ascending = sort.second == HcclSim::Storage::Order::ASCENDING;
                return ascending ? ScalarLess(*leftValue, *rightValue) : ScalarLess(*rightValue, *leftValue);
            }
            return false;
        });
    }

    /// 唯一约束检查：同 scope 表内唯一索引列组合已存在则拒绝
    HcclSim::Storage::DbResult<void>
    CheckUnique(const HcclSim::Storage::Ddl::CatalogTableMeta& meta, TableData& table, const LogicalRecord& row)
    {
        for (std::size_t i = 0; i < meta.indexCount; ++i) {
            const auto& index = meta.indexes[i];
            if (!index.unique) {
                continue;
            }
            std::vector<std::pair<uint16_t, DbScalar>> key;
            for (std::size_t c = 0; c < index.columnCount; ++c) {
                const uint16_t fieldId = FieldIdOfName(meta, index.columns[c]);
                for (const auto& field : row.fields) {
                    if (field.first == fieldId) {
                        key.emplace_back(fieldId, field.second);
                        break;
                    }
                }
            }
            for (const auto& existing : table.rows) {
                bool same = true;
                for (const auto& column : key) {
                    std::optional<DbScalar> value;
                    for (const auto& field : existing.fields) {
                        if (field.first == column.first) {
                            value = field.second;
                            break;
                        }
                    }
                    if (!value.has_value() || !ScalarEquals(*value, column.second)) {
                        same = false;
                        break;
                    }
                }
                if (same) {
                    return {DbCode::CONFLICT, 0, "unique constraint is violated"};
                }
            }
        }
        return Ok();
    }

    static uint16_t FieldIdOfName(const HcclSim::Storage::Ddl::CatalogTableMeta& meta, const char* name)
    {
        for (std::size_t i = 0; i < meta.fieldCount; ++i) {
            if (HcclSim::Storage::Ddl::TextEquals(meta.fields[i].name, name)) {
                return static_cast<uint16_t>(i);
            }
        }
        return HcclSim::Storage::kUnknownFieldId;
    }

    /// 级联删除：目录中 FK 指向本表的其他表，删除引用行
    void CascadeDelete(
        const HcclSim::Storage::Ddl::CatalogTableMeta& parent, const LogicalRecord& parentRow,
        const HcclSim::Storage::TableScope& scope)
    {
        uint64_t parentId = 0;
        {
            const uint16_t keyId = static_cast<uint16_t>(
                parent.primaryKeyCount > 0 ? FieldIdOfName(parent, parent.primaryKeyColumns[0]) : 0);
            for (const auto& field : parentRow.fields) {
                if (field.first == keyId) {
                    if (const auto* value = std::get_if<uint64_t>(&field.second); value != nullptr) {
                        parentId = *value;
                    }
                    break;
                }
            }
        }
        for (std::size_t t = 0; t < catalog_->tableCount; ++t) {
            const auto& table = catalog_->tables[t];
            for (std::size_t f = 0; f < table.foreignKeyCount; ++f) {
                const auto& foreignKey = table.foreignKeys[f];
                const auto targetName = HcclSim::Storage::Ddl::TargetTableName(foreignKey.target);
                if (targetName != parent.tableId || !foreignKey.cascadeDelete) {
                    continue;
                }
                const uint16_t localId = FieldIdOfName(table, foreignKey.localColumn);
                auto& childData = TableOf(table.tableId, scope);
                childData.rows.erase(
                    std::remove_if(
                        childData.rows.begin(), childData.rows.end(),
                        [localId, parentId](const LogicalRecord& child) {
                            for (const auto& field : child.fields) {
                                if (field.first == localId) {
                                    const auto* value = std::get_if<uint64_t>(&field.second);
                                    return value != nullptr && *value == parentId;
                                }
                            }
                            return false;
                        }),
                    childData.rows.end());
            }
        }
    }

    /// 即时 INSERT：visitRecord 收集原始字段值 → 自增分配 → 唯一检查 → 入表
    HcclSim::Storage::DbResult<uint64_t> ApplyInsert(
        const HcclSim::Storage::Ddl::CatalogTableMeta& meta, const HcclSim::Storage::TableScope& scope,
        const void* record)
    {
        if (record == nullptr || meta.visitRecord == nullptr) {
            return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "insert input is missing"};
        }
        RecordCollector collector;
        collector.meta = &meta;
        const auto visited = meta.visitRecord(record, &collector, RecordCollector::Emit);
        if (!visited.ok()) {
            return {visited.code, std::nullopt, visited.affectedRows, visited.diagnostic};
        }
        LogicalRecord row = std::move(collector.row);
        auto& table = TableOf(meta.tableId, scope);
        // 自增主键生成：目录声明 AutoIncrement 的字段由引擎分配并写回行数据。
        uint64_t generatedId = 0;
        for (std::size_t i = 0; i < meta.fieldCount; ++i) {
            if (HcclSim::Storage::Ddl::HasProp(meta.fields[i].props, HcclSim::Storage::Ddl::Prop::AutoIncrement)) {
                generatedId = table.nextId++;
                for (auto& field : row.fields) {
                    if (field.first == static_cast<uint16_t>(i)) {
                        field.second = DbScalar{generatedId};
                        break;
                    }
                }
            }
        }
        auto unique = CheckUnique(meta, table, row);
        if (!unique.ok()) {
            if (generatedId > 0) {
                --table.nextId; // 失败不消费序列
            }
            return {unique.code, std::nullopt, 0, unique.diagnostic};
        }
        table.rows.push_back(std::move(row));
        return {DbCode::OK, generatedId, 1, {}};
    }

    /// DML 事务槽状态校验（即时路径允许无事务；事务 DML 要求匹配的活动 binding）
    HcclSim::Storage::DbResult<void> CheckDmlState(HcclSim::Storage::TransactionBinding* transaction) const
    {
        if (transaction == nullptr) {
            return Ok();
        }
        if (activeBinding_ != transaction) {
            return {HcclSim::Storage::DbCode::INVALID_STATE, 0, "no matching transaction is active on this route"};
        }
        return Ok();
    }

    const HcclSim::Storage::Ddl::DatabaseDescriptor* catalog_;
    std::map<std::string, TableData> tables_;
    /// 事务开启时的表状态快照（Rollback 恢复用；测试引擎无 WAL，靠快照还原）
    std::map<std::string, TableData> snapshot_;
    const HcclSim::Storage::TransactionBinding* activeBinding_{nullptr};
    const void* lastInsertRecordAddress_{nullptr};

    /// 查询次数观察点：按逻辑表名累计 Select 调用（仅测试断言使用）
    std::map<std::string, uint64_t> selectCallsByTable_{};
};

} // namespace storage_conformance

#endif // HCCL_VM_STORAGE_CONFORMANCE_MEMORY_ENGINE_H
