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
 * NON-INFRINGEMENT. See the License in the specific file packaged with this Software for details.
 */

/**
 * @file transaction.h
 * @brief Move-Only 事务句柄（Transaction）与单向状态机模型定义。
 *
 * 【事务状态转移机模型 (Transaction State Machine)】
 *
 *                 ┌────────────────────────────────┐
 *                 │            BUILDING            │ <── BeginTransaction()
 *                 │  (DML 直达引擎 / 未提交状态)     │
 *                 └───┬────────────┬───────────┬───┘
 *                     │            │           │
 *        Commit() 成功│            │Commit()   │ Rollback() 或 析构自动回滚
 *                     ▼            │失败/异常  ▼
 *              ┌───────────┐       │     ┌─────────────┐
 *              │ COMMITTED │       │     │ ROLLED_BACK │
 *              └───────────┘       │     └─────────────┘
 *               [终态: 成功]       ▼      [终态: 已回滚]
 *                            ┌───────────┐
 *                            │  FAILED   │
 *                            └───────────┘
 *                             [终态: 熔断]
 *
 * @note 语义与约束规则：
 * 1. Move-Only：事务对象不可拷贝，在单线程/协程中独占流转；
 * 2. Single-Commit：Commit() 仅允许执行一次，终态后再次 Commit/Rollback 直接返回 INVALID_STATE；
 * 3. 析构安全：若处于 BUILDING 状态的事务句柄直接析构，析构函数自动执行安全 Rollback 兜底；
 * 4. 惰性绑定：通用事务在首条 DML 时记录实际 IStorageEngine*（由 Table<T> 提供）
 *    并调用 engine->Begin 开启物理事务，后续 DML 只比较 engine 指针（防止跨库/
 *    跨后端事务）；未发生任何 DML 的空事务提交/回滚为平凡成功，不触碰任何引擎；
 * 5. 直达执行：DML 以 const 引用/原始指针直达引擎立即执行；本层不保存任何
 *    record/ConditionData/UpdateData/SQL/statement 载荷，只按执行顺序记录
 *    AtomicItemResult；DML 失败统一调用一次 engine->Rollback 并熔断。
 */
#ifndef HCCL_VM_STORAGE_TRANSACTION_H
#define HCCL_VM_STORAGE_TRANSACTION_H

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "storage/database_result.h"
#include "storage/logical_operation.h"
#include "storage/query_spec.h"

namespace HcclSim::Storage {
class IStorageEngine;
class SessionControlBlock;
struct TransactionBinding;
namespace Ddl {
    struct CatalogTableMeta;
} // namespace Ddl

/// 事务生命周期状态枚举
enum class TransactionState : uint8_t {
    BUILDING,    ///< 正在执行事务 DML（未提交）
    COMMITTED,   ///< 事务已成功提交
    ROLLED_BACK, ///< 事务已显式或隐式回滚
    FAILED       ///< DML/提交失败或内部错误导致事务熔断
};

/**
 * @brief 事务句柄对象。
 */
class Transaction {
public:
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    Transaction(Transaction&&) noexcept;
    Transaction& operator=(Transaction&&) = delete;
    ~Transaction();

    /**
     * @brief 提交事务（物理 COMMIT；结果为按 DML 执行顺序记录的 AtomicResult）。
     * @return 成功返回 AtomicResult（包含各操作受影响行数/生成主键），终态不可重复提交
     */
    DbResult<AtomicResult> Commit();

    /**
     * @brief 显式回滚当前未提交事务。
     */
    DbResult<void> Rollback();

    /// 查询当前事务状态
    TransactionState State() const noexcept;

private:
    struct Impl;
    // engine 允许为空：通用事务在首条 DML 前未绑定物理引擎
    explicit Transaction(std::shared_ptr<SessionControlBlock> control);

    // ---- 以下四个入口仅由 Table<T> 调用：DML 直达引擎立即执行 ----
    // 首条 DML 经会话校验后绑定 engine 并开启物理事务；后续 DML 必须同一
    // engine。record/where/update/批量指针以 const 引用/原始指针传递，本层
    // 不复制；成功按序记录 AtomicItemResult，失败统一调用一次 engine->Rollback
    // 并进入 FAILED。
    DbResult<uint64_t>
    Insert(const Ddl::CatalogTableMeta& meta, const TableScope& scope, IStorageEngine* engine, const void* record);
    /// 批量插入入口（T1）：先 EnsureEngineBound 再引擎批量执行；成功按原输入
    /// 顺序追加 count 个 AtomicItemResult（每项 generatedId=对应ID、
    /// affectedRows=1——一批 N 行记 N 个逻辑结果项，不算一行）；失败调用
    /// 同一 FailWrite 回滚熔断
    DbResult<void> InsertMany(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, IStorageEngine* engine, const void* const* records,
        std::size_t count, uint64_t* generatedIds);
    DbResult<uint64_t> Update(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, IStorageEngine* engine, const ConditionData& where,
        const UpdateData& update);
    DbResult<uint64_t> Delete(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, IStorageEngine* engine, const ConditionData& where);
    /// BUILDING 状态与 engine 解析校验；首条 DML 绑定 engine 并 Begin 物理事务
    DbResult<void> EnsureEngineBound(IStorageEngine* engine);
    /// DML/写回失败的唯一负责人：若仍 BUILDING 且已绑定，只调用一次 engine
    /// Rollback，然后置 FAILED；返回原 DML/写回错误（Rollback 自身失败由引擎
    /// 既有故障关闭合同处置，不虚构 diagnostic 合并实现）。不复制错误清理链
    DbResult<void> FailWrite(DbCode code, std::string diagnostic);
    /// DML 成功按序记录结果；失败统一调用一次 engine->Rollback 并熔断（三个
    /// DML 入口共用）。generatedId 仅 INSERT 入口提供（引擎返回的生成主键）；
    /// UPDATE/DELETE 的引擎返回 value 是受影响行数，必须传 nullopt，不得误作
    /// 生成主键记录。
    DbResult<uint64_t> FinishWrite(DbResult<uint64_t>&& result, std::optional<uint64_t> generatedId);

    std::unique_ptr<Impl> impl_;
    friend class SessionControlBlock;
    template <typename T>
    friend class Table;
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_TRANSACTION_H
