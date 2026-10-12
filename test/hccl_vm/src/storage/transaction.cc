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
 * @file transaction.cc
 * @brief 实现 Move-Only Transaction 状态机、DML 直达引擎执行与原子 Commit/Rollback 语义。
 */
#include "storage/transaction.h"

#include <new>
#include <stdexcept>
#include <utility>

#include "storage/engine/i_storage_engine.h"
#include "storage/internal/session_control.h"

namespace HcclSim::Storage {
struct Transaction::Impl {
    std::shared_ptr<SessionControlBlock> control;
    IStorageEngine* engine{nullptr};
    TransactionBinding binding;
    TransactionState state{TransactionState::BUILDING};
    /// 按 DML 执行顺序记录的原子结果（Commit 时汇总为 AtomicResult 返回）
    std::vector<AtomicItemResult> results;
};

Transaction::Transaction(std::shared_ptr<SessionControlBlock> control)
    : impl_(std::make_unique<Impl>(
          Impl{std::move(control), nullptr, TransactionBinding{}, TransactionState::BUILDING, {}}))
{}
Transaction::Transaction(Transaction&&) noexcept = default;

/**
 * @brief 事务析构函数。
 * @note 若事务仍处于 BUILDING 未提交状态，析构时自动触发 Rollback 并归还会话 activeTransactions 计数；
 *       已 Commit/Rollback/Failed 的事务不重复执行物理操作。
 */
Transaction::~Transaction()
{
    if (impl_ != nullptr && impl_->state == TransactionState::BUILDING) {
        (void)Rollback();
    }
    if (impl_ != nullptr && impl_->control != nullptr) {
        impl_->control->ReleaseTransaction();
    }
}

/**
 * @brief BUILDING 状态与 engine 解析校验；首条 DML 绑定 engine 并开启物理事务。
 * @note 首条 DML 经会话控制块校验后记录物理引擎指针（Table<T> 直传）并调用
 *       engine->Begin；Begin 失败时引擎无活动物理事务，本事务直接熔断。
 *       后续 DML 只比较 engine 指针（防止跨库/跨后端事务）。
 */
DbResult<void> Transaction::EnsureEngineBound(IStorageEngine* engine)
{
    if (impl_ == nullptr || impl_->state != TransactionState::BUILDING) {
        return {DbCode::INVALID_STATE, 0, "transaction is not building"};
    }
    if (engine == nullptr) {
        return {DbCode::INVALID_ARGUMENT, 0, "transaction write requires a resolved engine"};
    }
    if (impl_->engine == nullptr) {
        auto accepted = impl_->control->AcceptTransactionEngine(engine);
        if (!accepted.ok()) {
            return accepted;
        }
        auto begun = engine->Begin(impl_->binding);
        if (!begun.ok()) {
            impl_->state = TransactionState::FAILED;
            return begun;
        }
        impl_->engine = engine;
    } else if (impl_->engine != engine) {
        return {DbCode::INVALID_ARGUMENT, 0, "operation is outside the transaction atomic domain"};
    }
    return {DbCode::OK, 0, {}};
}

/**
 * @brief DML/写回失败的唯一负责人：若仍 BUILDING 且已绑定，只调用一次
 *        engine->Rollback 回滚物理事务（引擎 DML/写回路径只执行并返回错误，
 *        不自行回滚），事务进入 FAILED；返回原错误。单行 DML（FinishWrite）
 *        与批量写回失败共用，不复制错误清理链。
 */
DbResult<void> Transaction::FailWrite(DbCode code, std::string diagnostic)
{
    if (impl_ != nullptr && impl_->state == TransactionState::BUILDING && impl_->engine != nullptr) {
        (void)impl_->engine->Rollback(impl_->binding);
    }
    if (impl_ != nullptr) {
        impl_->state = TransactionState::FAILED;
    }
    return {code, 0, std::move(diagnostic)};
}

/**
 * @brief DML 结果收尾：成功按序记录 AtomicItemResult（generatedId 仅 INSERT 携带，
 *        UPDATE/DELETE 传 nullopt——其引擎返回 value 是受影响行数，不是主键）；
 *        失败统一经 FailWrite 回滚熔断。
 */
DbResult<uint64_t> Transaction::FinishWrite(DbResult<uint64_t>&& result, std::optional<uint64_t> generatedId)
{
    if (!result.ok()) {
        (void)FailWrite(result.code, result.diagnostic);
        return std::move(result);
    }
    impl_->results.push_back(AtomicItemResult{std::move(generatedId), result.affectedRows});
    return std::move(result);
}

DbResult<uint64_t> Transaction::Insert(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, IStorageEngine* engine, const void* record)
{
    auto bound = EnsureEngineBound(engine);
    if (!bound.ok()) {
        return {bound.code, std::nullopt, bound.affectedRows, bound.diagnostic};
    }
    auto inserted = impl_->engine->Insert(meta, scope, record, &impl_->binding);
    // INSERT 的引擎返回 value 即生成主键（失败为 nullopt）：AtomicItemResult 中
    // generatedId 的唯一来源。
    auto generatedId = inserted.value;
    return FinishWrite(std::move(inserted), std::move(generatedId));
}

DbResult<void> Transaction::InsertMany(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, IStorageEngine* engine, const void* const* records,
    std::size_t count, uint64_t* generatedIds)
{
    auto bound = EnsureEngineBound(engine);
    if (!bound.ok()) {
        return {bound.code, bound.affectedRows, bound.diagnostic};
    }
    // 结果数组预分配（引擎执行前完成）：几何扩容——容量不足时至少翻倍、
    // 且不低于本批需求，避免每批 reserve(size+count) 恰好扩容造成的 O(N²)
    // 元素搬移。bad_alloc/length_error 经唯一 FailWrite(RESOURCE_EXHAUSTED)
    // 回滚熔断（本批引擎命令尚未执行；事务内前序 DML 由该回滚收尾）。
    // 预留成功后引擎执行与结果追加不再分配结果数组
    try {
        const std::size_t needed = impl_->results.size() + count;
        const std::size_t capacity = impl_->results.capacity();
        if (capacity < needed) {
            const std::size_t doubled = capacity <= impl_->results.max_size() / 2U ? capacity * 2U : needed;
            impl_->results.reserve(doubled > needed ? doubled : needed);
        }
    } catch (const std::bad_alloc&) {
        return FailWrite(DbCode::RESOURCE_EXHAUSTED, "transaction result allocation failed");
    } catch (const std::length_error&) {
        return FailWrite(DbCode::RESOURCE_EXHAUSTED, "transaction result allocation failed");
    }
    auto inserted = impl_->engine->InsertMany(meta, scope, records, count, generatedIds, &impl_->binding);
    if (!inserted.ok()) {
        return FailWrite(inserted.code, std::move(inserted.diagnostic));
    }
    // 成功按原输入顺序追加 count 个逻辑结果项（每项 generatedId=对应ID、
    // affectedRows=1）；容量已预留——追加不再分配。Commit 的结果长度按逻辑
    // 插入行数计算，不将一批算一行
    for (std::size_t index = 0; index < count; ++index) {
        impl_->results.push_back(AtomicItemResult{generatedIds[index], 1U});
    }
    return {DbCode::OK, count, {}};
}

DbResult<uint64_t> Transaction::Update(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, IStorageEngine* engine, const ConditionData& where,
    const UpdateData& update)
{
    auto bound = EnsureEngineBound(engine);
    if (!bound.ok()) {
        return {bound.code, std::nullopt, bound.affectedRows, bound.diagnostic};
    }
    return FinishWrite(impl_->engine->Update(meta, scope, where, update, &impl_->binding), std::nullopt);
}

DbResult<uint64_t> Transaction::Delete(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, IStorageEngine* engine, const ConditionData& where)
{
    auto bound = EnsureEngineBound(engine);
    if (!bound.ok()) {
        return {bound.code, std::nullopt, bound.affectedRows, bound.diagnostic};
    }
    return FinishWrite(impl_->engine->Delete(meta, scope, where, &impl_->binding), std::nullopt);
}

/**
 * @brief 提交事务（物理 COMMIT）。
 * @note Commit 仅允许执行一次，状态转移至 COMMITTED 或 FAILED。空事务（未绑定
 *       引擎）为平凡成功；结果为按 DML 执行顺序记录的 AtomicResult。
 */
DbResult<AtomicResult> Transaction::Commit()
{
    if (impl_ == nullptr || impl_->state != TransactionState::BUILDING) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "transaction is not building"};
    }
    if (impl_->engine == nullptr) {
        impl_->state = TransactionState::COMMITTED;
        return {DbCode::OK, AtomicResult{}, 0, {}};
    }
    auto committed = impl_->engine->Commit(impl_->binding);
    if (!committed.ok()) {
        impl_->state = TransactionState::FAILED;
        return {committed.code, std::nullopt, committed.affectedRows, committed.diagnostic};
    }
    impl_->state = TransactionState::COMMITTED;
    AtomicResult result;
    result.items = std::move(impl_->results);
    uint64_t affectedRows = 0;
    for (const auto& item : result.items) {
        affectedRows += item.affectedRows;
    }
    return {DbCode::OK, std::optional<AtomicResult>{std::move(result)}, affectedRows, {}};
}

/**
 * @brief 显式回滚事务。
 * @note 仅作用于 BUILDING 状态的事务，状态转移至 ROLLED_BACK 或 FAILED。
 *       未绑定引擎的空事务回滚为平凡成功。
 */
DbResult<void> Transaction::Rollback()
{
    if (impl_ == nullptr || impl_->state != TransactionState::BUILDING) {
        return {DbCode::INVALID_STATE, 0, "transaction is not building"};
    }
    if (impl_->engine == nullptr) {
        impl_->state = TransactionState::ROLLED_BACK;
        return {DbCode::OK, 0, {}};
    }
    auto result = impl_->engine->Rollback(impl_->binding);
    impl_->state = result.ok() ? TransactionState::ROLLED_BACK : TransactionState::FAILED;
    return result;
}

TransactionState Transaction::State() const noexcept
{
    return impl_ == nullptr ? TransactionState::FAILED : impl_->state;
}
} // namespace HcclSim::Storage
