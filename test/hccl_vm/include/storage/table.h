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
 * @file table.h
 * @brief 提供后端无关、强类型、Move-Only 的类型化表操作句柄（Table<T>）。
 *
 * 【Table<T> CRUD 数据流模型】
 *
 *   业务调用：table.Insert(record) / table.Select(表达式) / table.SelectOne(表达式)
 *              / table.Update(表达式, 变更)
 *       │
 *       ▼ 1. 表达式在公开入口构造唯一 ConditionData/UpdateData；字段编号经
 *          RecordSchema<T>::FieldIdOf 成员指针映射，未登记成员在入口拒绝
 *       │
 *       ▼ 2. 句柄创建时已完成一次解析：engine 指针 + 静态 meta 指针 + scope
 *          （RecordSchema<T>::DatabaseId() 定位库路由；会话/引擎/目录生命周期
 *           由控制块共享计数保证）
 *   engine_->Insert/Select/Count/Update/Delete/Clear(*meta_, scope_, ...)
 *       │
 *       ▼ 3. 析构时 RAII 自动释放
 *   ~Table() ──> control_->ReleaseTable() (activeTables--)
 *
 * 【业务表达边界】
 * 本句柄的全部公开方法只接受：记录对象、类型化条件表达式（Eq/And/Or/In/...）、
 * 排序/数量子句（OrderBy/Limit）与变更子句（Set/Increment）。
 * 查询规范、字段编号、数据库标量均为 storage 内部设施，不出现在业务源文件中。
 *
 * 【数据所有权合同】即时路径全程 const 引用直达引擎（无缓存/复制/重新编码）；
 * 事务 DML 同样以原始指针/const 引用直达引擎立即执行：事务层不保存任何
 * 业务载荷，生成主键由引擎返回后本层一次写回调用方 record。
 */
#ifndef HCCL_VM_STORAGE_TABLE_H
#define HCCL_VM_STORAGE_TABLE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "storage/database_result.h"
#include "storage/engine/i_storage_engine.h"
#include "storage/internal/session_control.h"
#include "storage/logical_operation.h"
#include "storage/query_expression.h"
#include "storage/query_spec.h"
#include "storage/record_contract.h"
#include "storage/transaction.h"

namespace HcclSim::Storage {
/**
 * @brief 强类型业务表操作句柄。
 * @tparam T 对应逻辑表 Schema 的 C++ 结构体类型
 */
template <typename T>
class Table {
public:
    Table(const Table&) = delete;
    Table& operator=(const Table&) = delete;
    // 移动构造必须整体移交句柄持有的全部状态（control_/engine_/meta_/scope_/active_）：
    // 一次性入口把句柄移入 optional/DbResult 返回链，遗漏任一指针会使操作
    // 回退到空引擎（假直达）。
    Table(Table&& other) noexcept
        : control_(std::move(other.control_)),
          engine_(std::exchange(other.engine_, nullptr)),
          meta_(std::exchange(other.meta_, nullptr)),
          scope_(other.scope_),
          active_(other.active_)
    {
        other.active_ = false;
    }
    Table& operator=(Table&&) = delete;

    /// 析构时自动递减控制块内的 activeTables 计数
    ~Table()
    {
        if (active_ && control_ != nullptr) {
            control_->ReleaseTable();
        }
    }

    /**
     * @brief 即时插入单条记录。
     * @param record 待插入的业务记录对象（自增主键将在插入成功后自动写回）
     * @return 成功返回新记录的主键（PrimaryKey<T>），失败返回错误码
     */
    DbResult<PrimaryKey<T>> Insert(T& record)
    {
        const auto inserted = engine_->Insert(*meta_, scope_, &record, nullptr);
        if (!inserted.ok()) {
            return {inserted.code, std::nullopt, inserted.affectedRows, inserted.diagnostic};
        }
        PrimaryKey<T> key{*inserted.value};
        const auto written = RecordCodec<T>::WritePrimaryKey(record, key);
        if (!written.ok()) {
            return {written.code, std::nullopt, written.affectedRows, written.diagnostic};
        }
        return {DbCode::OK, std::optional<PrimaryKey<T>>{key}, inserted.affectedRows, {}};
    }

    /**
     * @brief 在指定事务中插入单条记录（直达引擎立即执行）。
     * @param record 待插入记录（原始指针直传引擎，无任何复制）
     * @param transaction 目标事务句柄
     */
    DbResult<PrimaryKey<T>> Insert(T& record, Transaction& transaction)
    {
        const auto inserted = transaction.Insert(*meta_, scope_, engine_, &record);
        if (!inserted.ok()) {
            return {inserted.code, std::nullopt, inserted.affectedRows, inserted.diagnostic};
        }
        PrimaryKey<T> key{*inserted.value};
        const auto written = RecordCodec<T>::WritePrimaryKey(record, key);
        if (!written.ok()) {
            // 写回失败：同一 Transaction 失败负责人回滚熔断（此前该分支只返回
            // 错误未通知事务——本修复与批量写回失败共用 FailWrite）；错误转换
            // 为原返回类型
            const auto failed = transaction.FailWrite(written.code, written.diagnostic);
            return {failed.code, std::nullopt, failed.affectedRows, failed.diagnostic};
        }
        return {DbCode::OK, std::optional<PrimaryKey<T>>{key}, inserted.affectedRows, {}};
    }

    /**
     * @brief 在指定事务中批量插入同表记录（T1：自动无约束批次在引擎内一次
     * 取号与批存在性预检；显式/混合 ID、Unique/FK、跨 endpoint 等复杂批次由
     * 引擎退回既有逐行链，功能不裁剪）。
     *
     * 合同：
     * - 输入是连续的最终 T 数组：本层只在栈上保存指针与生成 ID（各至多
     *   kInsertBatchMaxRows 个），不复制 T；
     * - count=0 平凡成功（不取表、不 Begin）；事务已终态返回 INVALID_STATE
     *   （最外层先查公开 State()）；
     * - count>kInsertBatchMaxRows 或指针空 → INVALID_ARGUMENT（零网络写；
     *   超限由调用方按固定窗口拆分，不由引擎积攒）；
     * - 引擎整批成功后按输入顺序一次写回各记录主键；写回失败调用同一
     *   Transaction 失败负责人（FailWrite）整笔回滚熔断，不返回成功——本批
     *   已写回的 ID 不伪装恢复成未分配（与单行插入后回滚相同，ID 不代表已
     *   提交，不回收序列）。
     */
    DbResult<void> InsertMany(T* records, std::size_t count, Transaction& transaction)
    {
        if (count == 0U) {
            if (transaction.State() != TransactionState::BUILDING) {
                return {DbCode::INVALID_STATE, 0, "transaction is not building"};
            }
            return {DbCode::OK, 0, {}};
        }
        if (records == nullptr || count > kInsertBatchMaxRows) {
            return {DbCode::INVALID_ARGUMENT, 0, "insert-many requires records and at most kInsertBatchMaxRows rows"};
        }
        // 栈上指针/ID 数组：不复制业务载荷，指针只在本次同步调用期间存活
        std::array<const void*, kInsertBatchMaxRows> pointers{};
        std::array<uint64_t, kInsertBatchMaxRows> generatedIds{};
        for (std::size_t index = 0; index < count; ++index) {
            pointers[index] = &records[index];
        }
        const auto inserted
            = transaction.InsertMany(*meta_, scope_, engine_, pointers.data(), count, generatedIds.data());
        if (!inserted.ok()) {
            return {inserted.code, inserted.affectedRows, inserted.diagnostic};
        }
        // 引擎整批成功后按输入顺序写回主键（一次写回；失败由 Transaction 唯一
        // 回滚并熔断）
        for (std::size_t index = 0; index < count; ++index) {
            PrimaryKey<T> key{generatedIds[index]};
            const auto written = RecordCodec<T>::WritePrimaryKey(records[index], key);
            if (!written.ok()) {
                const auto failed = transaction.FailWrite(written.code, written.diagnostic);
                return {failed.code, failed.affectedRows, failed.diagnostic};
            }
        }
        return {DbCode::OK, count, {}};
    }

    /**
     * @brief 条件多行查询（过滤、排序与数量全部下推到数据库端）。
     *
     * @param where 类型化过滤条件（All<T>() 表示全量）
     * @param clauses 可选 OrderBy/Limit 子句；多个 OrderBy 按传入顺序成为复合排序键，
     *                不隐式附加排序或数量限制，依赖既有顺序的调用点必须显式 OrderBy；
     *                重复 Limit 被入口拒绝
     * @return 成功返回记录数组（无符合记录时返回 OK 及空数组；单行读取使用 SelectOne）
     */
    template <typename... Clauses>
    DbResult<std::vector<T>> Select(const Condition<T>& where, const Clauses&... clauses) const
    {
        QueryOptions options;
        const auto collected = CollectQueryClauses(options, clauses...);
        if (!collected.ok()) {
            return {collected.code, std::nullopt, collected.affectedRows, collected.diagnostic};
        }
        const auto sorted = ValidateSort(options);
        if (!sorted.ok()) {
            return {sorted.code, std::nullopt, sorted.affectedRows, sorted.diagnostic};
        }
        std::vector<T> records;
        auto sink = ResultSink::For(records);
        const auto result = engine_->Select(*meta_, scope_, where.data, options, sink);
        if (!result.ok()) {
            return {result.code, std::nullopt, result.affectedRows, result.diagnostic};
        }
        return {DbCode::OK, std::optional<std::vector<T>>{std::move(records)}, result.affectedRows, result.diagnostic};
    }

    /**
     * @brief 同表多作用域批量条件查询（每作用域独立结果，排序/数量独立应用）。
     *
     * 合同：
     * - scopes 为空时平凡成功（返回空外层 vector，不调用引擎）；
     * - 一次解析 options（子句归集与排序校验复用 Select 同一逻辑）；
     * - 外层结果先 resize(scopes.size()) 再建立各 ResultSink，地址稳定后
     *   交给引擎；调用期间不 resize 外层、不复制 T；
     * - 失败返回无 value，调用方不得消费部分结果。
     *
     * @param scopes 作用域列表（同表；动态 PID 表每作用域一个 PID）
     * @param where 类型化过滤条件（全部作用域共用同一份）
     * @param clauses 可选 OrderBy/Limit 子句（每作用域独立应用）
     */
    template <typename... Clauses>
    DbResult<std::vector<std::vector<T>>>
    SelectMany(const std::vector<TableScope>& scopes, const Condition<T>& where, const Clauses&... clauses) const
    {
        QueryOptions options;
        const auto collected = CollectQueryClauses(options, clauses...);
        if (!collected.ok()) {
            return {collected.code, std::nullopt, collected.affectedRows, collected.diagnostic};
        }
        const auto sorted = ValidateSort(options);
        if (!sorted.ok()) {
            return {sorted.code, std::nullopt, sorted.affectedRows, sorted.diagnostic};
        }
        if (scopes.empty()) {
            return {DbCode::OK, std::optional<std::vector<std::vector<T>>>{}, 0, {}};
        }
        // 外层先 resize 建立地址稳定的结果数组，再逐项建立写回槽；引擎调用
        // 期间外层不再扩容
        std::vector<std::vector<T>> results(scopes.size());
        std::vector<ResultSink> sinks;
        sinks.reserve(scopes.size());
        for (std::vector<T>& result : results) {
            sinks.push_back(ResultSink::For(result));
        }
        const auto selected = engine_->SelectMany(*meta_, scopes, where.data, options, sinks);
        if (!selected.ok()) {
            return {selected.code, std::nullopt, selected.affectedRows, selected.diagnostic};
        }
        return {
            DbCode::OK, std::optional<std::vector<std::vector<T>>>{std::move(results)}, selected.affectedRows,
            selected.diagnostic};
    }

    /**
     * @brief 条件单行查询（SELECT 的标量便利形态；过滤与排序下推数据库端）。
     *
     * 合同：
     * - 内部强制 limit=1，结果经 ResultSink::For(T&) 直接写最终对象；
     * - 只接受零个或多个 OrderBy 子句与至多一个 Fields 投影子句，Limit 子句
     *   在编译期拒绝；没有 OrderBy 时不承诺多个匹配行中命中哪一行；
     * - 未命中返回 DbCode::NOT_FOUND，其他后端错误原样传播；
     * - 主键查询只是 Eq(&T::id, id) 条件，不存在主键专用后端操作；
     * - 投影不改变 WHERE/ORDER BY 与结果选择：条件/排序引用字段可在投影外
     *   （过滤/排序由数据库端独立完成），未投影字段保持记录默认值。
     *
     * @param where 类型化过滤条件
     * @param clauses 可选 OrderBy 子句与至多一个 Fields 投影子句
     * @return 成功返回单条记录；未命中返回 NOT_FOUND
     */
    template <typename... Clauses>
    DbResult<T> SelectOne(const Condition<T>& where, const Clauses&... clauses) const
    {
        QueryOptions options;
        const auto collected = CollectOneRowClauses(options, clauses...);
        if (!collected.ok()) {
            return {collected.code, std::nullopt, collected.affectedRows, collected.diagnostic};
        }
        const auto sorted = ValidateSort(options);
        if (!sorted.ok()) {
            return {sorted.code, std::nullopt, sorted.affectedRows, sorted.diagnostic};
        }
        options.limit = 1;
        // 结果对象直接在待返回 DbResult 的 value 中构造（消除"局部 T 再转入
        // optional"的显式记录中转；具名结果返回不写 std::move，保留 NRVO 机会。
        // 未命中/失败由引擎返回 NOT_FOUND/错误码，走下方失败分支清空 value）。
        DbResult<T> result{DbCode::OK, std::nullopt, 0, {}};
        result.value.emplace();
        auto sink = ResultSink::For(*result.value);
        const auto selected = engine_->Select(*meta_, scope_, where.data, options, sink);
        if (!selected.ok()) {
            result.code = selected.code;
            result.affectedRows = selected.affectedRows;
            result.diagnostic = selected.diagnostic;
            result.value.reset();
            return result;
        }
        result.affectedRows = selected.affectedRows;
        result.diagnostic = selected.diagnostic;
        return result;
    }

    /**
     * @brief 条件计数（数据库端 COUNT 聚合，不加载行集后取 size）。
     * @param where 类型化过滤条件（All<T>() 表示全表计数）
     * @return 成功返回行数
     */
    DbResult<uint64_t> Count(const Condition<T>& where) const { return engine_->Count(*meta_, scope_, where.data); }

    /**
     * @brief 条件计数聚合（数据库端单语句同时返回 where 集合总数与其中匹配
     * match 的数量；不加载行集，不退化为两次 Count）。
     *
     * 合同：
     * - where 确定统计集合，match 只在该集合内匹配计数；All(match) 时
     *   matched == total；空集合返回 OK + {0, 0}；
     * - 未实现该能力的引擎返回 UNSUPPORTED_CAPABILITY（引擎默认实现，
     *   本层不掩盖、不降级）；
     * - 正常结果 matched <= total；失败返回失败码且无 value。
     *
     * @param where 类型化过滤条件（All<T>() 表示全表）
     * @param match 集合内匹配条件
     * @return 成功返回 CountMatchResult
     */
    DbResult<CountMatchResult> CountMatches(const Condition<T>& where, const Condition<T>& match) const
    {
        return engine_->CountMatches(*meta_, scope_, where.data, match.data);
    }

    /**
     * @brief 条件更新：把若干 Set/Increment 变更编译为数据库单语句更新。
     *
     * 变更列表中追加 Transaction& 时（事务句柄必须为已检查成功的左值），
     * 本调用改为把更新暂存到该事务；否则立即执行并返回受影响行数。
     *
     * @param where 目标过滤条件（不得为 All：无条件全表更新被安全拦截）
     * @param changes 若干 Set/Increment 变更子句，可尾随一个 Transaction&
     */
    template <typename ConditionArg, typename... Changes>
    std::conditional_t<
        std::disjunction_v<std::is_same<std::decay_t<Changes>, Transaction>...>, DbResult<void>, DbResult<uint64_t>>
    Update(ConditionArg&& where, Changes&&... changes)
    {
        static_assert(
            std::is_same_v<std::remove_cv_t<std::remove_reference_t<ConditionArg>>, Condition<T>>,
            "update requires a Condition<T> for this table");
        constexpr bool transactional = std::disjunction_v<std::is_same<std::decay_t<Changes>, Transaction>...>;
        Transaction* transaction = nullptr;
        // 只为唯一 UpdateData 精确 reserve 一次（按参数顺序直接转换写入，不建
        // 中间 Assignment 容器）
        constexpr std::size_t changeCount
            = (static_cast<std::size_t>(std::is_same_v<std::decay_t<Changes>, Assignment<T>>) + ... + 0U);
        UpdateData update;
        update.reserve(changeCount);
        (CollectUpdateClause(update, transaction, std::forward<Changes>(changes)), ...);
        if (std::holds_alternative<std::monostate>(where.data) || update.empty()) {
            return MakeUpdateFailure<transactional>(
                DbCode::INVALID_ARGUMENT, 0, "update requires a condition and at least one change");
        }
        for (const auto& change : update) {
            if (change.fieldId == kUnknownFieldId) {
                return MakeUpdateFailure<transactional>(DbCode::INVALID_ARGUMENT, 0, "update field is not declared");
            }
            if (IsPrimaryKeyField(change.fieldId)) {
                return MakeUpdateFailure<transactional>(
                    DbCode::INVALID_ARGUMENT, 0, "primary key field cannot be updated");
            }
        }
        if constexpr (transactional) {
            // 事务路径：条件/更新以 const 引用直达引擎立即执行（本层不复制）。
            const auto updated = transaction->Update(*meta_, scope_, engine_, where.data, update);
            if (!updated.ok()) {
                return {updated.code, updated.affectedRows, updated.diagnostic};
            }
            return {DbCode::OK, updated.affectedRows, {}};
        } else {
            // 即时路径：条件与更新全程 const 引用直达引擎。
            return engine_->Update(*meta_, scope_, where.data, update, nullptr);
        }
    }

    /**
     * @brief 条件删除。
     * @param where 目标过滤条件（不得为 All：无条件全表删除被安全拦截，全表清空使用 Clear）
     * @return 成功返回被删除的记录行数
     */
    template <typename ConditionArg>
    DbResult<uint64_t> Delete(ConditionArg&& where)
    {
        static_assert(
            std::is_same_v<std::remove_cv_t<std::remove_reference_t<ConditionArg>>, Condition<T>>,
            "delete requires a Condition<T> for this table");
        if (std::holds_alternative<std::monostate>(where.data)) {
            return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "delete requires a condition"};
        }
        return engine_->Delete(*meta_, scope_, where.data, nullptr);
    }

    /// 在指定事务中条件删除（直达引擎立即执行，条件以 const 引用传递）
    template <typename ConditionArg>
    DbResult<void> Delete(ConditionArg&& where, Transaction& transaction)
    {
        static_assert(
            std::is_same_v<std::remove_cv_t<std::remove_reference_t<ConditionArg>>, Condition<T>>,
            "delete requires a Condition<T> for this table");
        if (std::holds_alternative<std::monostate>(where.data)) {
            return {DbCode::INVALID_ARGUMENT, 0, "delete requires a condition"};
        }
        const auto removed = transaction.Delete(*meta_, scope_, engine_, where.data);
        if (!removed.ok()) {
            return {removed.code, removed.affectedRows, removed.diagnostic};
        }
        return {DbCode::OK, removed.affectedRows, {}};
    }

    /**
     * @brief 清空当前单表数据。
     * @param mode 清理模式（KEEP_SEQUENCE 保留序列号 / RESET_SEQUENCE 重置序列号 /
     *             DROP_DYNAMIC_SCOPE 销毁 PID 动态表）
     */
    DbResult<void> Clear(ClearMode mode) { return engine_->Clear(*meta_, scope_, mode); }

    /// ---- 内部构造（路由/引擎/在途计数已在构造前完成；产品业务代码不直接使用） ----
    /// 解析结果（engine + 静态 meta + scope）由会话一次性提供，本句柄的操作直达
    /// 引擎虚函数；句柄析构归还在途计数。
    Table(
        std::shared_ptr<SessionControlBlock> control, IStorageEngine* engine, const Ddl::CatalogTableMeta* meta,
        TableScope scope)
        : control_(std::move(control)),
          engine_(engine),
          meta_(meta),
          scope_(scope)
    {}

private:
    /// 按事务形态构造失败返回：即时更新与事务暂存两种返回类型的成员布局不同
    template <bool Transactional>
    static std::conditional_t<Transactional, DbResult<void>, DbResult<uint64_t>>
    MakeUpdateFailure(DbCode code, uint64_t affectedRows, std::string diagnostic)
    {
        if constexpr (Transactional) {
            return DbResult<void>{code, affectedRows, std::move(diagnostic)};
        } else {
            return DbResult<uint64_t>{code, std::nullopt, affectedRows, std::move(diagnostic)};
        }
    }

    /// 归集多行查询子句：OrderBy 追加排序键（多个按传入顺序复合），Limit 写入
    /// 数量限制，Fields 写入字段投影（至多一次）
    template <typename C>
    static DbResult<void> CollectQueryClause(QueryOptions& options, const C& clause)
    {
        if constexpr (std::is_same_v<C, SortClause<T>>) {
            options.sorts.emplace_back(
                clause.fieldId, clause.direction == SortDirection::ASC ? Order::ASCENDING : Order::DESCENDING);
            return {DbCode::OK, 0, {}};
        } else if constexpr (std::is_same_v<C, LimitClause>) {
            if (options.limit.has_value()) {
                return {DbCode::INVALID_ARGUMENT, 0, "limit clause must appear at most once"};
            }
            options.limit = clause.limit;
            return {DbCode::OK, 0, {}};
        } else if constexpr (std::is_same_v<C, ProjectionClause>) {
            if (!options.projectedFields.empty()) {
                return {DbCode::INVALID_ARGUMENT, 0, "projection clause must appear at most once"};
            }
            options.projectedFields = clause.fieldIds;
            return {DbCode::OK, 0, {}};
        } else {
            static_assert(!sizeof(C), "query clauses must come from OrderBy, Limit or Fields");
            return {DbCode::INVALID_ARGUMENT, 0, {}};
        }
    }

    /// 归集单行查询子句：接受 OrderBy 与至多一个 Fields 投影；Limit 在编译期
    /// 拒绝（SelectOne 内部固定 limit=1；重复 Fields 明确报错）
    template <typename C>
    static DbResult<void> CollectOneRowClause(QueryOptions& options, const C& clause)
    {
        if constexpr (std::is_same_v<C, SortClause<T>>) {
            options.sorts.emplace_back(
                clause.fieldId, clause.direction == SortDirection::ASC ? Order::ASCENDING : Order::DESCENDING);
            return {DbCode::OK, 0, {}};
        } else if constexpr (std::is_same_v<C, ProjectionClause>) {
            if (!options.projectedFields.empty()) {
                return {DbCode::INVALID_ARGUMENT, 0, "projection clause must appear at most once"};
            }
            options.projectedFields = clause.fieldIds;
            return {DbCode::OK, 0, {}};
        } else {
            static_assert(!sizeof(C), "SelectOne accepts OrderBy or Fields clauses only; use Select for Limit");
            return {DbCode::INVALID_ARGUMENT, 0, {}};
        }
    }

    /// 单行查询子句包（空包合法：无排序单行读取）
    static DbResult<void> CollectOneRowClauses(QueryOptions& options)
    {
        (void)options;
        return {DbCode::OK, 0, {}};
    }

    template <typename First, typename... Rest>
    static DbResult<void> CollectOneRowClauses(QueryOptions& options, const First& first, const Rest&... rest)
    {
        const auto collected = CollectOneRowClause(options, first);
        if (!collected.ok()) {
            return collected;
        }
        return CollectOneRowClauses(options, rest...);
    }

    /// 展开查询子句包（空包合法：全量无条件查询）
    static DbResult<void> CollectQueryClauses(QueryOptions& options)
    {
        (void)options;
        return {DbCode::OK, 0, {}};
    }

    template <typename First, typename... Rest>
    static DbResult<void> CollectQueryClauses(QueryOptions& options, const First& first, const Rest&... rest)
    {
        const auto collected = CollectQueryClause(options, first);
        if (!collected.ok()) {
            return collected;
        }
        return CollectQueryClauses(options, rest...);
    }

    /// 归集更新子句包中的变更赋值与事务句柄（每个 Assignment 直接转换为最终
    /// UpdateData 元素；右值 move DbScalar，不复制 owning 载荷）
    template <typename C>
    static void CollectUpdateClause(UpdateData& update, Transaction*& transaction, C&& clause)
    {
        if constexpr (std::is_same_v<std::decay_t<C>, Transaction>) {
            static_assert(std::is_lvalue_reference_v<C>, "transaction must be passed as an lvalue");
            transaction = &clause;
        } else if constexpr (std::is_same_v<std::decay_t<C>, Assignment<T>>) {
            update.push_back(LogicalChange{
                clause.fieldId, clause.increment ? ChangeKind::INCREMENT : ChangeKind::SET,
                std::forward<C>(clause).value});
        } else {
            static_assert(!sizeof(C), "update clauses must come from Set, Increment or a transaction handle");
        }
    }

    /// 排序字段必须已在该表目录中声明（通用合同：声明过的字段即可排序，无每表白名单）
    DbResult<void> ValidateSort(const QueryOptions& options) const
    {
        for (const auto& sort : options.sorts) {
            if (sort.first == kUnknownFieldId || sort.first >= meta_->fieldCount) {
                return {DbCode::INVALID_ARGUMENT, 0, "sort field is not declared"};
            }
        }
        return {DbCode::OK, 0, {}};
    }

    /// 判断字段编号是否为该表主键（主键由数据库生成，不允许业务更新）
    bool IsPrimaryKeyField(uint16_t fieldId) const
    {
        return fieldId < meta_->fieldCount && meta_->fields[fieldId].primaryKey;
    }

    std::shared_ptr<SessionControlBlock> control_;
    IStorageEngine* engine_{nullptr};
    const Ddl::CatalogTableMeta* meta_{nullptr};
    TableScope scope_{};
    bool active_{true};
    friend class StorageSession;
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_TABLE_H
