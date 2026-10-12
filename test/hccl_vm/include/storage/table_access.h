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
 * @file table_access.h
 * @brief 业务代码到存储层的单步类型化入口：AcquireTable<T>、BeginTransaction()
 *        与一次性原子操作入口。
 *
 * 【调用流程与生命周期】
 *
 *   业务入口
 *       │ AcquireTable<T>(scope)
 *       ▼
 *   ProcessStorageContext::AcquireReady() ──> StorageSessionLease（在途租约）
 *       │        （冷进程首次调用经已注册的应用级 bootstrap 惰性建立会话，
 *       │         等价迁移前惰性 singleton 的首次建连时机；single-flight）
 *       │ lease->Get().Table<T>(scope)
 *       ▼
 *   Table<T> 类型化句柄（持有 SessionControlBlock 共享计数，activeTables++；
 *       engine 指针 + 静态 meta + scope 一次解析）
 *       │ 租约在句柄构造完成后立即归还
 *       ▼
 *   业务执行 Insert/Select/Count/Update/Delete/Clear（直达引擎虚函数）
 *       │
 *       ▼ 句柄析构（activeTables--，归零时唤醒排空等待）
 *
 * 【生命周期约束】
 * 1. 业务不需要自行创建、安装或关闭会话；会话由进程装配方负责。冷进程首次原子
 *    访问经唯一 ready-session 入口自动建立会话；显式 close/remove 后不自动复活。
 * 2. 句柄在会话静止（quiesce）、排空或关闭阶段取表直接返回 INVALID_STATE，
 *    不抛异常、不静默忽略。
 * 3. 句柄持有期间，会话关闭/静止排空会等待 activeTables 归零；
 *    与租约持有期间的排空语义一致，不新增锁或第二个全局会话。
 * 4. 同一业务入口多次使用同一张表时只获取一次句柄并复用；
 *    多张表分别各取一次。
 *
 * 【两种等价入口形态】
 * 同一数据库原子能力有两种使用形态，业务按调用形态选择：
 * 1. 一次性入口（本文件下半部分 Insert/Select/Count/Update/Delete/Clear
 *    模板函数）：同一业务流程对某张表只执行一次操作时使用；每次调用内部只
 *    获取一次 Table<T>（engine/meta/scope 一次解析）并执行一次对应操作，
 *    随后立即释放句柄。
 * 2. Table 句柄入口（AcquireTable<T>）：同一流程对同一张表执行多次操作、
 *    需要显式持有表生命周期或参与事务时使用；只获取一次句柄并复用。
 * 两种形态都不绑定任何业务表或字段；后端选择只由注册和 mapping 决定，
 * 业务调用不变。
 *
 * 【事务入口】
 * BeginTransaction() 为唯一通用事务边界：业务不选择任何事务域或路由枚举，
 * 物理事务域在首个暂存 DML 时惰性绑定（Table<T> 提供 engine 指针）；
 * 事务析构时未提交则自动回滚。
 */
#ifndef HCCL_VM_STORAGE_TABLE_ACCESS_H
#define HCCL_VM_STORAGE_TABLE_ACCESS_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "storage/database_result.h"
#include "storage/internal/process_storage_context.h"
#include "storage/query_expression.h"
#include "storage/query_spec.h"
#include "storage/storage_session.h"
#include "storage/table.h"
#include "storage/transaction.h"

namespace HcclSim::Storage {
/**
 * @brief 单步获取进程全局会话上的类型化表句柄。
 * @tparam T 实现了 RecordContract 的 C++ 结构体类型
 * @param scope 表作用域（静态表传空，动态表传 PID）
 * @return 成功返回 Table<T> 句柄；未注册 bootstrap 的冷进程、会话静止或排空阶段
 *         返回错误码；冷进程首次调用经已注册 bootstrap 惰性建立会话
 */
template <typename T>
DbResult<Table<T>> AcquireTable(const TableScope& scope = {})
{
    auto lease = ProcessStorageContext::AcquireReady();
    if (!lease.ok()) {
        return {lease.code, std::nullopt, lease.affectedRows, lease.diagnostic};
    }
    return lease.value->Get().Table<T>(scope);
}

namespace detail {
    /// 一次性原子操作入口的内部取表：与 AcquireTable<T> 同一路径（一次会话租约内
    /// 完成库路由、引擎惰性打开与在途计数，句柄持有 engine/meta/scope 直达引擎）。
    template <typename T>
    DbResult<Table<T>> AcquireOperationTable(const TableScope& scope = {})
    {
        return AcquireTable<T>(scope);
    }
} // namespace detail

/**
 * @brief 在进程全局会话上开启通用事务。
 * @return 成功返回 Transaction 句柄；首个暂存 DML 时绑定物理事务域，
 *         跨物理域的后续 DML 在执行前返回 INVALID_ARGUMENT
 */
inline DbResult<Transaction> BeginTransaction()
{
    auto lease = ProcessStorageContext::AcquireReady();
    if (!lease.ok()) {
        return {lease.code, std::nullopt, lease.affectedRows, lease.diagnostic};
    }
    return lease.value->Get().BeginTransaction();
}

// ============================================================
// 一次性原子操作入口（Table 句柄入口的同能力第二种使用形态）
// ============================================================
// 适用：同一业务流程对某张表只执行一次操作。
// 不适用：同一流程多次操作同一张表或需要事务——继续使用 AcquireTable<T>()
// 句柄与 BeginTransaction()，避免重复取得表句柄、保持事务绑定。
// 合同：
// 1. 内部只取得一次 Table<T>（一次进程会话租约）并调用一次既有 Table<T> 操作；
// 2. 不缓存数据、不重试、不吞错误、不新增会话或后端调用，不改变路由；
// 3. 取表失败原样传播 code / affectedRows / diagnostic；
// 4. Table 操作失败原样返回对应 DbResult；
// 5. 静态表使用默认 scope；动态表以首个参数显式传入 TableScope，
//    置于条件/记录与可变子句之前，避免与 OrderBy/Limit/事务句柄产生歧义；
// 6. 事务重载复用现有 Transaction（Update/Delete 的事务句柄仍按 Table<T>
//    既有规则传入），不新增事务分组或业务路由。

/// 即时插入单条记录（静态 scope 表；自增主键写回 record）。
template <typename T>
DbResult<PrimaryKey<T>> Insert(T& record)
{
    auto table = detail::AcquireOperationTable<T>();
    if (!table.ok()) {
        return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
    }
    return table.value->Insert(record);
}

/// 即时插入单条记录（显式 scope：动态 PID 表传 TableScope::Process(pid)）。
template <typename T>
DbResult<PrimaryKey<T>> Insert(const TableScope& scope, T& record)
{
    auto table = detail::AcquireOperationTable<T>(scope);
    if (!table.ok()) {
        return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
    }
    return table.value->Insert(record);
}

/// 把记录插入暂存到指定事务（静态 scope 表）。
template <typename T>
DbResult<PrimaryKey<T>> Insert(T& record, Transaction& transaction)
{
    auto table = detail::AcquireOperationTable<T>();
    if (!table.ok()) {
        return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
    }
    return table.value->Insert(record, transaction);
}

/// 把记录插入暂存到指定事务（显式 scope：动态 PID 表）。
template <typename T>
DbResult<PrimaryKey<T>> Insert(const TableScope& scope, T& record, Transaction& transaction)
{
    auto table = detail::AcquireOperationTable<T>(scope);
    if (!table.ok()) {
        return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
    }
    return table.value->Insert(record, transaction);
}

/// 把同表多条记录批量插入暂存到指定事务（静态 scope 表；T1）。输入为连续
/// 最终 T 数组（count=0 在取表前平凡成功）；主键在引擎整批成功后按序写回。
template <typename T>
DbResult<void> InsertMany(T* records, std::size_t count, Transaction& transaction)
{
    if (count == 0U) {
        if (transaction.State() != TransactionState::BUILDING) {
            return {DbCode::INVALID_STATE, 0, "transaction is not building"};
        }
        return {DbCode::OK, 0, {}};
    }
    auto table = detail::AcquireOperationTable<T>();
    if (!table.ok()) {
        return {table.code, table.affectedRows, table.diagnostic};
    }
    return table.value->InsertMany(records, count, transaction);
}

/// 把同表多条记录批量插入暂存到指定事务（显式 scope：动态 PID 表；T1）。
template <typename T>
DbResult<void> InsertMany(const TableScope& scope, T* records, std::size_t count, Transaction& transaction)
{
    if (count == 0U) {
        if (transaction.State() != TransactionState::BUILDING) {
            return {DbCode::INVALID_STATE, 0, "transaction is not building"};
        }
        return {DbCode::OK, 0, {}};
    }
    auto table = detail::AcquireOperationTable<T>(scope);
    if (!table.ok()) {
        return {table.code, table.affectedRows, table.diagnostic};
    }
    return table.value->InsertMany(records, count, transaction);
}

/// 条件多行查询（静态 scope 表；过滤/排序/数量全部下推数据库端）。
/// 单行读取使用 SelectOne；多行 Select 不隐式附加排序或数量限制。
template <typename T, typename... Clauses>
DbResult<std::vector<T>> Select(const Condition<T>& where, const Clauses&... clauses)
{
    auto table = detail::AcquireOperationTable<T>();
    if (!table.ok()) {
        return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
    }
    return table.value->Select(where, clauses...);
}

/// 条件多行查询（显式 scope：动态 PID 表）。
template <typename T, typename... Clauses>
DbResult<std::vector<T>> Select(const TableScope& scope, const Condition<T>& where, const Clauses&... clauses)
{
    auto table = detail::AcquireOperationTable<T>(scope);
    if (!table.ok()) {
        return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
    }
    return table.value->Select(where, clauses...);
}

/// 条件单行查询（静态 scope 表；SELECT 标量形态，内部 limit=1）。
/// 未命中返回 NOT_FOUND；接受 OrderBy 与至多一个 Fields 子句，不分配单元素 vector。
template <typename T, typename... Clauses>
DbResult<T> SelectOne(const Condition<T>& where, const Clauses&... clauses)
{
    auto table = detail::AcquireOperationTable<T>();
    if (!table.ok()) {
        return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
    }
    return table.value->SelectOne(where, clauses...);
}

/// 条件单行查询（显式 scope：动态 PID 表）。
template <typename T, typename... Clauses>
DbResult<T> SelectOne(const TableScope& scope, const Condition<T>& where, const Clauses&... clauses)
{
    auto table = detail::AcquireOperationTable<T>(scope);
    if (!table.ok()) {
        return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
    }
    return table.value->SelectOne(where, clauses...);
}

/// 同表多作用域批量条件查询（一次性入口：一次取表覆盖整批访问）。
/// scopes 为空不取表、平凡成功；非空经现有 AcquireOperationTable<T>
/// （以 scopes.front() 建立会话）持有一个 Table 句柄并调用其 SelectMany。
/// 每作用域结果独立（排序/数量独立应用），失败返回无 value。
template <typename T, typename... Clauses>
DbResult<std::vector<std::vector<T>>>
SelectMany(const std::vector<TableScope>& scopes, const Condition<T>& where, const Clauses&... clauses)
{
    if (scopes.empty()) {
        return {DbCode::OK, std::optional<std::vector<std::vector<T>>>{}, 0, {}};
    }
    auto table = detail::AcquireOperationTable<T>(scopes.front());
    if (!table.ok()) {
        return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
    }
    return table.value->SelectMany(scopes, where, clauses...);
}

/// 条件计数（静态 scope 表；数据库端 COUNT 聚合，不加载行集）。
template <typename T>
DbResult<uint64_t> Count(const Condition<T>& where)
{
    auto table = detail::AcquireOperationTable<T>();
    if (!table.ok()) {
        return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
    }
    return table.value->Count(where);
}

/// 条件计数（显式 scope：动态 PID 表）。
template <typename T>
DbResult<uint64_t> Count(const TableScope& scope, const Condition<T>& where)
{
    auto table = detail::AcquireOperationTable<T>(scope);
    if (!table.ok()) {
        return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
    }
    return table.value->Count(where);
}

/// 条件计数聚合（静态 scope 表；单语句同时返回 where 集合总数与其中匹配
/// match 的数量，不加载行集）。合同见 Table<T>::CountMatches。
template <typename T>
DbResult<CountMatchResult> CountMatches(const Condition<T>& where, const Condition<T>& match)
{
    auto table = detail::AcquireOperationTable<T>();
    if (!table.ok()) {
        return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
    }
    return table.value->CountMatches(where, match);
}

/// 条件计数聚合（显式 scope：动态 PID 表）。
template <typename T>
DbResult<CountMatchResult> CountMatches(const TableScope& scope, const Condition<T>& where, const Condition<T>& match)
{
    auto table = detail::AcquireOperationTable<T>(scope);
    if (!table.ok()) {
        return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
    }
    return table.value->CountMatches(where, match);
}

/// 条件更新（静态 scope 表；Set/Increment 变更编译为数据库单语句更新）。
/// enable_if 约束首参为 Condition<T>：避免转发引用把显式 scope 重载的首参
/// 吸收为条件（TableScope 形态经下方 scope 重载进入）。
template <
    typename T, typename ConditionArg, typename... Changes,
    std::enable_if_t<std::is_same_v<std::remove_cv_t<std::remove_reference_t<ConditionArg>>, Condition<T>>, int> = 0>
std::conditional_t<
    std::disjunction_v<std::is_same<std::decay_t<Changes>, Transaction>...>, DbResult<void>, DbResult<uint64_t>>
Update(ConditionArg&& where, Changes&&... changes)
{
    constexpr bool transactional = std::disjunction_v<std::is_same<std::decay_t<Changes>, Transaction>...>;
    auto table = detail::AcquireOperationTable<T>();
    if (!table.ok()) {
        if constexpr (transactional) {
            return {table.code, table.affectedRows, table.diagnostic};
        } else {
            return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
        }
    }
    return table.value->Update(std::forward<ConditionArg>(where), std::forward<Changes>(changes)...);
}

/// 条件更新（显式 scope：动态 PID 表）。
template <typename T, typename ConditionArg, typename... Changes>
std::conditional_t<
    std::disjunction_v<std::is_same<std::decay_t<Changes>, Transaction>...>, DbResult<void>, DbResult<uint64_t>>
Update(const TableScope& scope, ConditionArg&& where, Changes&&... changes)
{
    constexpr bool transactional = std::disjunction_v<std::is_same<std::decay_t<Changes>, Transaction>...>;
    auto table = detail::AcquireOperationTable<T>(scope);
    if (!table.ok()) {
        if constexpr (transactional) {
            return {table.code, table.affectedRows, table.diagnostic};
        } else {
            return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
        }
    }
    return table.value->Update(std::forward<ConditionArg>(where), std::forward<Changes>(changes)...);
}

/// 条件删除（静态 scope 表；无条件全表删除被安全拦截）。
template <
    typename T, typename ConditionArg,
    std::enable_if_t<std::is_same_v<std::remove_cv_t<std::remove_reference_t<ConditionArg>>, Condition<T>>, int> = 0>
DbResult<uint64_t> Delete(ConditionArg&& where)
{
    auto table = detail::AcquireOperationTable<T>();
    if (!table.ok()) {
        return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
    }
    return table.value->Delete(std::forward<ConditionArg>(where));
}

/// 条件删除（显式 scope：动态 PID 表）。
template <typename T, typename ConditionArg>
DbResult<uint64_t> Delete(const TableScope& scope, ConditionArg&& where)
{
    auto table = detail::AcquireOperationTable<T>(scope);
    if (!table.ok()) {
        return {table.code, std::nullopt, table.affectedRows, table.diagnostic};
    }
    return table.value->Delete(std::forward<ConditionArg>(where));
}

/// 把条件删除暂存到指定事务（静态 scope 表）。
template <
    typename T, typename ConditionArg,
    std::enable_if_t<std::is_same_v<std::remove_cv_t<std::remove_reference_t<ConditionArg>>, Condition<T>>, int> = 0>
DbResult<void> Delete(ConditionArg&& where, Transaction& transaction)
{
    auto table = detail::AcquireOperationTable<T>();
    if (!table.ok()) {
        return {table.code, table.affectedRows, table.diagnostic};
    }
    return table.value->Delete(std::forward<ConditionArg>(where), transaction);
}

/// 把条件删除暂存到指定事务（显式 scope：动态 PID 表）。
template <typename T, typename ConditionArg>
DbResult<void> Delete(const TableScope& scope, ConditionArg&& where, Transaction& transaction)
{
    auto table = detail::AcquireOperationTable<T>(scope);
    if (!table.ok()) {
        return {table.code, table.affectedRows, table.diagnostic};
    }
    return table.value->Delete(std::forward<ConditionArg>(where), transaction);
}

/// 清空当前单表数据（静态 scope 表）。
template <typename T>
DbResult<void> Clear(ClearMode mode)
{
    auto table = detail::AcquireOperationTable<T>();
    if (!table.ok()) {
        return {table.code, table.affectedRows, table.diagnostic};
    }
    return table.value->Clear(mode);
}

/// 清空当前单表数据（显式 scope：动态 PID 表可传 DROP_DYNAMIC_SCOPE）。
template <typename T>
DbResult<void> Clear(const TableScope& scope, ClearMode mode)
{
    auto table = detail::AcquireOperationTable<T>(scope);
    if (!table.ok()) {
        return {table.code, table.affectedRows, table.diagnostic};
    }
    return table.value->Clear(mode);
}
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_TABLE_ACCESS_H
