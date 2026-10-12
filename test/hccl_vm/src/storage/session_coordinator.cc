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
 * @file session_coordinator.cc
 * @brief 实现 SessionCoordinator 对业务会话、Quiesce 闸门和具名 ResetPlan 的顺序编排。
 */
#include "storage/session_coordinator.h"

#include <optional>
#include <utility>

#include "storage/internal/process_storage_context.h"

namespace HcclSim::Storage {
/**
 * @brief 校验重置计划的合法性。
 * @note 必须具备非空 planId、至少包含一个数据库或外部步骤，且所有步骤的 ID 不为空。
 */
bool ResetPlan::Valid() const noexcept
{
    if (planId.empty() || (tableSteps.empty() && externalSteps.empty())) {
        return false;
    }
    for (const auto& step : tableSteps) {
        if (step.tableId.empty()) {
            return false;
        }
    }
    for (const auto& step : externalSteps) {
        if (step.resourceId.empty()) {
            return false;
        }
    }
    return true;
}

struct SessionCoordinator::Impl {
    explicit Impl(StorageSession value) : session(std::move(value)) {}
    StorageSession session;
    std::optional<QuiesceToken> token;
    bool installed{false};
    bool closed{false};
};

SessionCoordinator::SessionCoordinator(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SessionCoordinator::SessionCoordinator(SessionCoordinator&&) noexcept = default;
SessionCoordinator::~SessionCoordinator() = default;

/**
 * @brief 打开协调器内部持有的私有会话。
 * @note 在显式调用 Install() 成功前，该会话不会对外发布到进程全局上下文中。
 */
DbResult<SessionCoordinator>
SessionCoordinator::Open(const SessionStorageBinding& binding, const StorageOpenRequest& request)
{
    auto opened = StorageSession::Open(binding, request);
    if (!opened.ok()) {
        return {opened.code, std::nullopt, opened.affectedRows, opened.diagnostic};
    }
    SessionCoordinator coordinator(std::make_unique<Impl>(std::move(*opened.value)));
    return {DbCode::OK, std::optional<SessionCoordinator>{std::move(coordinator)}, 0, {}};
}

DbResult<void> SessionCoordinator::Install()
{
    if (impl_ == nullptr || impl_->closed || impl_->installed) {
        return {DbCode::INVALID_STATE, 0, "session coordinator cannot install"};
    }
    auto result = ProcessStorageContext::Install(impl_->session);
    if (result.ok()) {
        impl_->installed = true;
    }
    return result;
}

/**
 * @brief 从进程入口施加静止闸门。
 * @note 保存令牌副本作为后续 Reset 计划执行的 Epoch 基准。
 */
DbResult<QuiesceToken> SessionCoordinator::ApplyQuiesceGate(std::chrono::milliseconds timeout)
{
    if (impl_ == nullptr || impl_->closed || !impl_->installed || impl_->token.has_value()) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "session coordinator cannot quiesce"};
    }
    auto token = ProcessStorageContext::ApplyQuiesceGate(timeout);
    if (token.ok()) {
        impl_->token = *token.value;
    }
    return token;
}

/**
 * @brief 严格按步骤顺序执行 ResetPlan。
 * @note 数据库表清表失败立即终止返回；外部资源只加入待处理列表，由原 owner 继续清理。
 */
DbResult<ResetResult> SessionCoordinator::ExecuteReset(const ResetPlan& plan, const QuiesceToken& token)
{
    if (impl_ == nullptr || impl_->closed || !impl_->token.has_value() || !plan.Valid() || !token.Valid()
        || !token.SameEpoch(*impl_->token)) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "reset requires the active coordinator gate"};
    }
    ResetResult result;
    for (const auto& step : plan.tableSteps) {
        const auto cleared = impl_->session.ClearForReset(step.tableId, step.scope, step.mode, token);
        if (!cleared.ok()) {
            return {cleared.code, std::nullopt, cleared.affectedRows, cleared.diagnostic};
        }
        ++result.clearedTables;
    }
    for (const auto& step : plan.externalSteps) {
        result.pendingExternalResources.push_back(step.resourceId);
    }
    return {DbCode::OK, std::optional<ResetResult>{std::move(result)}, 0, {}};
}

DbResult<void> SessionCoordinator::ReleaseQuiesceGate(const QuiesceToken& token)
{
    if (impl_ == nullptr || impl_->closed || !impl_->token.has_value()) {
        return {DbCode::INVALID_STATE, 0, "session coordinator is not quiesced"};
    }
    auto result = ProcessStorageContext::ReleaseQuiesceGate(token);
    if (result.ok()) {
        impl_->token.reset();
    }
    return result;
}

/**
 * @brief 完整关闭协调器。
 * @note 顺序执行：释放 Gate -> 移除并排空进程会话 (RemoveAndDrain) -> 关闭 StorageSession。
 */
DbResult<void> SessionCoordinator::Close(std::chrono::milliseconds timeout)
{
    if (impl_ == nullptr || impl_->closed) {
        return {DbCode::INVALID_STATE, 0, "session coordinator is closed"};
    }
    if (impl_->token.has_value()) {
        auto released = ReleaseQuiesceGate(*impl_->token);
        if (!released.ok()) {
            return released;
        }
    }
    if (impl_->installed) {
        auto removed = ProcessStorageContext::RemoveAndDrain(timeout);
        if (!removed.ok()) {
            return removed;
        }
        impl_->installed = false;
    }
    auto closed = impl_->session.Close(timeout);
    if (closed.ok()) {
        impl_->closed = true;
    }
    return closed;
}
} // namespace HcclSim::Storage
