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
 * @file storage_session.h
 * @brief 提供进程内数据库会话外观边界（StorageSession），管理表与事务生命周期。
 *
 * 【组件架构与所有权拓扑 (Component Hierarchy & Ownership)】
 *
 *   ┌───────────────────────────────────────────────────────────┐
 *   │             ProcessStorageContext (进程全局上下文)        │
 *   └─────────────────────────────┬─────────────────────────────┘
 *                                 │ 进程内唯一安装 (Install / Lease)
 *                                 ▼
 *   ┌───────────────────────────────────────────────────────────┐
 *   │                  StorageSession (会话外观)                │
 *   │  - 提供 Table<T>()、ListTables、BeginTransaction 门面     │
 *   └─────────────────────────────┬─────────────────────────────┘
 *                                 │ 共享持有 (std::shared_ptr)
 *                                 ▼
 *   ┌───────────────────────────────────────────────────────────┐
 *   │             SessionControlBlock (核心状态机与控制中枢)     │
 *   │  - 维护: activeTables, activeTransactions 在途计数        │
 *   │  - 路由: 每库 DatabaseRoute 槽（目录指针 + 引擎引用）      │
 *   └───┬─────────────────────────┬─────────────────────────┬───┘
 *       │ 借出 (Table<T>)        │ 开启 (Transaction)      │ 清理 (ExecuteReset)
 *       ▼                         ▼                         ▼
 *   Table<T> 句柄           Transaction 状态机         engine->Clear(...)
 *   (engine+meta+scope;     (activeTransactions++;     (受控 quiesce token 路径)
 *    activeTables++;        首个 Stage 绑定引擎)
 *    析构时 activeTables--)
 *
 * @note 析构与持久化原则：StorageSession 析构函数仅释放内存，不隐式执行 BGSAVE/AOF 同步
 *       或网络重试。所有数据持久化必须通过显式调用 Backup() 或事务 Commit() 来保证。
 */
#ifndef HCCL_VM_STORAGE_STORAGE_SESSION_H
#define HCCL_VM_STORAGE_STORAGE_SESSION_H

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "storage/backup_contract.h"
#include "storage/database_result.h"
#include "storage/engine_registration.h"
#include "storage/internal/session_control.h"
#include "storage/storage_binding.h"
#include "storage/table.h"
#include "storage/transaction.h"

namespace HcclSim::Storage {
/// 打开会话所需的配置请求
struct StorageOpenRequest {
    StorageOpenMode mode{StorageOpenMode::READ_WRITE_CREATE}; ///< 会话打开模式
    /// 静态 schema 打开意图：库引擎首次打开时映射为引擎打开模式
    /// （OwnerPrepare→READ_WRITE_CREATE 建库+DDL；ReadWriteExisting→READ_EXISTING
    ///  读写已存在库不执行 DDL；内部装配角色，不进入业务 API）。
    SchemaRole schemaRole{SchemaRole::OwnerPrepare};
    std::vector<EngineRegistration> engines; ///< 预装配的物理引擎实例/工厂（按逻辑库注册）
    class IStorageBackupProvider* backupProvider{nullptr}; ///< 备份驱动（仅 BACKUP_ONLY 需要）
};

class ProcessStorageContext;
class SessionControlBlock;
class SessionCoordinator;

/**
 * @brief 存储会话对象（Move-Only 资源句柄）。
 */
class StorageSession {
public:
    StorageSession(const StorageSession&) = delete;
    StorageSession& operator=(const StorageSession&) = delete;
    StorageSession(StorageSession&&) noexcept;
    StorageSession& operator=(StorageSession&&) = delete;
    ~StorageSession();

    /**
     * @brief 打开普通读写/只读业务会话。
     * @param binding 位置清单快照
     * @param request 引擎装配与表依赖请求
     * @return 成功返回可移动的 StorageSession，失败返回错误码
     */
    static DbResult<StorageSession> Open(const SessionStorageBinding& binding, const StorageOpenRequest& request);

    /**
     * @brief 使用有效静止令牌打开仅备份会话（BACKUP_ONLY）。
     * @note 仅允许 SessionCoordinator 或内部备份流程调用。
     */
    static DbResult<StorageSession>
    OpenForBackup(const SessionStorageBinding& binding, const StorageOpenRequest& request, const QuiesceToken& token);

    /**
     * @brief 获取类型化的表操作句柄。
     * @tparam T 实现了 RecordContract 的 C++ 结构体类型
     * @param scope 表作用域（静态表传空，动态表传 PID）
     * @return 成功返回拥有型 Table<T> 句柄（增加 activeTables 计数；持有
     *         engine 指针 + 静态 meta 指针 + scope，CRUD 直达引擎）
     */
    template <typename T>
    DbResult<::HcclSim::Storage::Table<T>> Table(const TableScope& scope = {});

    /// 列出指定逻辑库的用户表名（维护/兼容入口；库路由解析 + 引擎惰性打开）
    DbResult<std::vector<std::string>> ListTables(std::string_view databaseId);

    /**
     * @brief 开启通用事务（业务事务边界唯一入口）。
     * @return 成功返回 Transaction 句柄；首个暂存写时绑定物理引擎，
     *         后续跨库/跨后端写操作在执行前返回 INVALID_ARGUMENT；空事务提交/回滚为平凡成功。
     */
    DbResult<Transaction> BeginTransaction();

    /// 应用写入静止闸门，等待在途操作全部排空并生成 QuiesceToken
    DbResult<QuiesceToken> ApplyQuiesceGate(std::chrono::milliseconds timeout = std::chrono::seconds(30));

    /// 释放静止闸门，使 Token 失效并恢复正常业务读写
    DbResult<void> ReleaseQuiesceGate(const QuiesceToken& token);

    /// 执行显式物理快照备份（仅限 BACKUP_ONLY 模式）
    DbResult<BackupArtifact> Backup(const BackupRequest& request);

    /// 显式等待在途操作归零并安全关闭所有底层引擎
    DbResult<void> Close(std::chrono::milliseconds timeout = std::chrono::seconds(30));

private:
    explicit StorageSession(std::shared_ptr<SessionControlBlock> control);
    bool TryMarkInstalled() noexcept;
    void MarkRemoved() noexcept;
    bool IsInstalled() const noexcept;
    DbResult<void>
    ClearForReset(const std::string& tableId, const TableScope& scope, ClearMode mode, const QuiesceToken& token);
    std::shared_ptr<SessionControlBlock> control_;
    friend class ProcessStorageContext;
    friend class SessionCoordinator;
    /// 测试同伴（仅在测试目标中定义）：读取内部控制块真实同步状态，
    /// 不构成公开业务 API，生产代码不得依赖。
    friend struct StorageSessionTestPeer;
};
} // namespace HcclSim::Storage

// Table<T> 创建流程：先解析并占用库路由（engine 惰性打开 + 在途计数），失败时
// 不构造句柄。表 → 逻辑库归属由 RecordSchema<T>::DatabaseId() 编译期派生
// （静态目录唯一来源）；scope 类型由表目录声明校验（静态表无进程 scope、
// 动态 PID 表显式携带）；句柄持有 engine 指针 + 进程期稳定静态 meta + scope，
// CRUD 直达引擎虚函数，不回会话做表路由。
template <typename T>
HcclSim::Storage::DbResult<HcclSim::Storage::Table<T>> HcclSim::Storage::StorageSession::Table(const TableScope& scope)
{
    if (control_ == nullptr) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "storage session is empty"};
    }
    auto engine = control_->AcquireTable(RecordSchema<T>::DatabaseId(), RecordSchema<T>::ScopeKind(), scope);
    if (!engine.ok()) {
        return {engine.code, std::nullopt, engine.affectedRows, engine.diagnostic};
    }
    HcclSim::Storage::Table<T> table(control_, *engine.value, &RecordSchema<T>::Meta(), scope);
    return {DbCode::OK, std::optional<HcclSim::Storage::Table<T>>{std::move(table)}, 0, {}};
}

#endif // HCCL_VM_STORAGE_STORAGE_SESSION_H
