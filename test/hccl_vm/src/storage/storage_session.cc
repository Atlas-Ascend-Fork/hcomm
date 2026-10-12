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
 * @file storage_session.cc
 * @brief 实现 StorageSession 的打开方式、业务门面转发、显式关闭及 Fork 安全处理。
 */
#include "storage/storage_session.h"

#include <cstdio>
#include <exception>
#include <optional>
#include <utility>

#include "storage/internal/session_control.h"

namespace HcclSim::Storage {
namespace {
    [[noreturn]] void RejectInstalledSessionRelocation() noexcept
    {
        std::fputs("cannot move or destroy an installed StorageSession\n", stderr);
        std::fflush(stderr);
        std::terminate();
    }
} // namespace

StorageSession::StorageSession(std::shared_ptr<SessionControlBlock> control) : control_(std::move(control)) {}

/**
 * @brief 移动构造函数。
 * @note 已安装（Installed）会话的内存地址已发布到 ProcessStorageContext，禁止移动以防止悬挂指针。
 */
StorageSession::StorageSession(StorageSession&& other) noexcept
{
    if (other.IsInstalled()) {
        RejectInstalledSessionRelocation();
    }
    control_ = std::move(other.control_);
}

StorageSession::~StorageSession()
{
    if (IsInstalled()) {
        RejectInstalledSessionRelocation();
    }
}

/**
 * @brief 打开普通业务会话。
 * @note 明确拒绝 BACKUP_ONLY 模式，防止业务入口意外获得特权备份能力。
 */
DbResult<StorageSession> StorageSession::Open(const SessionStorageBinding& binding, const StorageOpenRequest& request)
{
    if (request.mode == StorageOpenMode::BACKUP_ONLY) {
        return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "BACKUP_ONLY requires OpenForBackup"};
    }
    auto control = SessionControlBlock::Create(binding, request);
    if (!control.ok()) {
        return {control.code, std::nullopt, control.affectedRows, control.diagnostic};
    }
    StorageSession session(std::move(*control.value));
    return {DbCode::OK, std::optional<StorageSession>{std::move(session)}, 0, {}};
}

/**
 * @brief 使用静止令牌打开仅备份会话（BACKUP_ONLY）。
 * @note 必须持有处于 active 状态且与 binding 严格匹配的 QuiesceToken。
 */
DbResult<StorageSession> StorageSession::OpenForBackup(
    const SessionStorageBinding& binding, const StorageOpenRequest& request, const QuiesceToken& token)
{
    if (request.mode != StorageOpenMode::BACKUP_ONLY || !token.Valid() || token.DataNamespace() != binding.dataNamespace
        || token.SessionId() != binding.sessionId) {
        return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "backup session requires a matching quiesce token"};
    }
    auto control = SessionControlBlock::Create(binding, request);
    if (!control.ok()) {
        return {control.code, std::nullopt, control.affectedRows, control.diagnostic};
    }
    StorageSession session(std::move(*control.value));
    return {DbCode::OK, std::optional<StorageSession>{std::move(session)}, 0, {}};
}

DbResult<std::vector<std::string>> StorageSession::ListTables(std::string_view databaseId)
{
    if (control_ == nullptr) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "storage session is empty"};
    }
    return control_->ListTables(databaseId);
}

DbResult<Transaction> StorageSession::BeginTransaction()
{
    if (control_ == nullptr) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "storage session is empty"};
    }
    return control_->BeginUnboundTransaction();
}

DbResult<QuiesceToken> StorageSession::ApplyQuiesceGate(std::chrono::milliseconds timeout)
{
    return control_ == nullptr ?
               DbResult<QuiesceToken>{DbCode::INVALID_STATE, std::nullopt, 0, "storage session is empty"} :
               control_->ApplyQuiesceGate(timeout);
}

DbResult<void> StorageSession::ReleaseQuiesceGate(const QuiesceToken& token)
{
    return control_ == nullptr ? DbResult<void>{DbCode::INVALID_STATE, 0, "storage session is empty"} :
                                 control_->ReleaseQuiesceGate(token);
}

/**
 * @brief 显式排空并关闭会话。
 * @note 已安装会话必须先通过 ProcessStorageContext::RemoveAndDrain 移除，否则返回 INVALID_STATE。
 */
DbResult<void> StorageSession::Close(std::chrono::milliseconds timeout)
{
    if (IsInstalled()) {
        return {DbCode::INVALID_STATE, 0, "installed storage session must be removed before close"};
    }
    return control_ == nullptr ? DbResult<void>{DbCode::INVALID_STATE, 0, "storage session is empty"} :
                                 control_->Close(timeout);
}

DbResult<BackupArtifact> StorageSession::Backup(const BackupRequest& request)
{
    if (control_ == nullptr) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "backup requires a BACKUP_ONLY session"};
    }
    return control_->Backup(request);
}

bool StorageSession::TryMarkInstalled() noexcept { return control_ != nullptr && control_->TryInstall(); }

void StorageSession::MarkRemoved() noexcept
{
    if (control_ != nullptr) {
        control_->MarkRemoved();
    }
}

bool StorageSession::IsInstalled() const noexcept { return control_ != nullptr && control_->Installed(); }

/**
 * @brief 私有重置入口（清表）。
 * @note 仅 SessionCoordinator 可见，普通业务表禁止直接调用。
 */
DbResult<void> StorageSession::ClearForReset(
    const std::string& tableId, const TableScope& scope, ClearMode mode, const QuiesceToken& token)
{
    return control_ == nullptr ? DbResult<void>{DbCode::INVALID_STATE, 0, "storage session is empty"} :
                                 control_->ExecuteReset(tableId, scope, mode, token);
}
} // namespace HcclSim::Storage
