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
 * @file session_control.h
 * @brief 会话共享控制块（SessionControlBlock），维护在途计数、Quiesce 状态机与每库引擎路由。
 *
 * 【控制块核心状态机 (Session State Transitions)】
 *
 *              ┌─────────────────────────────────────────┐
 *              │                  OPEN                   │ <── Create() 初始化成功
 *              │ (activeTables++, activeTransactions++)  │
 *              └────────────────────┬────────────────────┘
 *                                   │
 *                 ApplyQuiesceGate()│ 拦截新表/新事务
 *                                   ▼
 *              ┌─────────────────────────────────────────┐
 *              │                QUIESCING                │
 *              │  (等待 activeTables/Tx 归零 -> Drained) │
 *              └────────┬───────────────────────┬────────┘
 *                       │                       │
 *   ReleaseQuiesceGate()│ 恢复业务              │ Close() 显式关闭
 *                       ▼                       ▼
 *              ┌─────────────────┐    ┌──────────────────┐
 *              │      OPEN       │    │      CLOSED      │ (引擎安全退出)
 *              └─────────────────┘    └──────────────────┘
 *
 * 【路由模型】会话只保存每逻辑库一个 DatabaseRoute 槽（目录指针 + 引擎引用）；
 * 表 → 逻辑库归属由 RecordSchema<T>::DatabaseId() 编译期派生，取表路由为
 * O(已配置库数)。Acquire 完成路由、引擎惰性打开与在途计数，产出
 * engine 指针（Table<T> 自行携带静态 meta 与 scope）；会话不保存逐表绑定、
 * 字段映射、SQL、查询条件或查询历史。
 */
#ifndef HCCL_VM_STORAGE_INTERNAL_SESSION_CONTROL_H
#define HCCL_VM_STORAGE_INTERNAL_SESSION_CONTROL_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "storage/backup_contract.h"
#include "storage/database_result.h"
#include "storage/engine_registration.h"
#include "storage/query_spec.h"
#include "storage/storage_binding.h"
#include "storage/storage_catalog.h"

namespace HcclSim::Storage {
class IStorageEngine;
class Transaction;
struct StorageOpenRequest;
namespace Ddl {
    struct CatalogTableMeta;
} // namespace Ddl

/// 会话内单个逻辑数据库的路由槽（每配置库一项）。
/// 静态目录是表定义唯一来源：本槽只保存本库的静态目录指针与引擎引用，
/// 不复制库内每张表的绑定/映射字符串（表 → 逻辑库归属由 Ddl::DatabaseOf<T>
/// 编译期派生，类型化取表路由 O(库数)）。
/// 引擎/物理连接按本库首次真实访问惰性创建（REQ-CAP-2：安装时只登记稳定
/// factory 配置，不构造引擎；single-flight：并发首访只执行一次创建+Open+Prepare，
/// 等待者共享同轮结果；打开期间持有在途计数，quiesce/drain 不提前放行）。
/// 打开/准备失败时丢弃本轮构造的实例（终态 Close 后的同一对象不得伪装成可重试），
/// 下次访问经 factory 重新构造。
struct SessionDatabaseRoute {
    std::string databaseId;                          ///< 逻辑库名（每库一份 owning，非每表）
    StorageBackend backend{};                        ///< 本库后端模块注册名（装配层解析后写入）
    const Ddl::DatabaseDescriptor* catalog{nullptr}; ///< 本库静态目录（只读指针，不复制）
    IStorageEngine* engine{nullptr};                 ///< 装配注册的预创建引擎（非拥有）
    /// 惰性构造工厂（稳定配置；创建/Open/Prepare 纳入每库 single-flight）
    StorageEngineFactory factory;
    /// 工厂形态下本路由槽持有的引擎实例（失败轮次后丢弃以支持真实重试）
    std::unique_ptr<IStorageEngine> ownedEngine;
    /// 本库 schema 打开意图覆盖（nullopt 时继承会话级角色；冷 bootstrap 按库判定）
    std::optional<SchemaRole> roleOverride;
    /// 预创建引擎形态（外部所有权）：Open 已成功但 Prepare 尚未完成。失败轮不把
    /// 外部所有权引擎终态 Close——下一轮跳过第二次 Open，只重试 Prepare。
    bool precreatedOpened{false};
    /// 预创建引擎形态的永久失败（Open 抛异常后对象状态不可知；非空时本路由直接
    /// 返回该稳定错误，不再触碰该引擎对象）。
    std::optional<DbResult<void>> permanentFailure;

    /// 引擎惰性打开状态机（锁为所在会话控制块 mutex）
    enum class OpenState : uint8_t { NotOpened, Opening, Opened };
    OpenState openState{OpenState::NotOpened};
    /// 已结束的打开轮次计数（等待者据代差识别自己等待的那一轮）
    uint64_t openGeneration{0};
    /// 最近一轮打开结果（全部本轮等待者消费前不得被下一轮覆盖）
    std::optional<DbResult<void>> lastOpenResult;
    /// 已登记为本轮等待者且尚未消费结果的线程数
    std::size_t openWaiters{0};
};

/**
 * @brief 会话内部控制中枢（引用计数共享对象）。
 * @note 内部封装互斥锁、条件变量与活跃对象计数器，协调 Table/Transaction 与底层引擎驱动。
 */
class SessionControlBlock final : public std::enable_shared_from_this<SessionControlBlock> {
public:
    /// 校验 binding 身份、装配库路由槽并创建控制块（引擎按需打开由首次访问触发）
    static DbResult<std::shared_ptr<SessionControlBlock>>
    Create(const SessionStorageBinding& binding, const StorageOpenRequest& request);
    ~SessionControlBlock();

    /// 解析逻辑库路由并增加在途 Table 计数（activeTables++），返回该库引擎。
    /// 类型化路径：databaseId 由 RecordSchema<T>::DatabaseId() 编译期提供，
    /// scopeKind 由表目录声明提供——静态表要求无进程 scope，动态 PID 表要求
    /// 显式进程 scope；静态 meta 与 scope 由 Table<T> 自行携带。
    DbResult<IStorageEngine*>
    AcquireTable(std::string_view databaseId, Ddl::ScopeKind scopeKind, const TableScope& scope);

    /// 归还在途 Table 计数（activeTables--，归零时触发 drained 通知）
    void ReleaseTable() noexcept;

    /// 仅允许持有当前 active quiesce token 的协调器按计划清表
    /// （按表名在本会话已配置库的静态目录内解析 meta 与引擎，冷路径允许线性查找）
    DbResult<void>
    ExecuteReset(const std::string& tableId, const TableScope& scope, ClearMode mode, const QuiesceToken& token);

    /// 列出指定逻辑库的用户表名（解析库路由并惰性打开引擎）
    DbResult<std::vector<std::string>> ListTables(std::string_view databaseId);

    /// 创建未绑定路由的通用事务：首个暂存 DML 时绑定物理引擎
    DbResult<Transaction> BeginUnboundTransaction();

    /// 为通用事务的首个暂存写校验会话状态并接受传入引擎（后续 Stage 只比较指针）
    DbResult<void> AcceptTransactionEngine(IStorageEngine* engine);

    /// 归还在途事务计数（activeTransactions--）
    void ReleaseTransaction() noexcept;

    /// 停止创建新 Table/Transaction，等待在途对象排空归零并生成带 generation 的静止令牌
    DbResult<QuiesceToken> ApplyQuiesceGate(std::chrono::milliseconds timeout);

    /// 解除静止状态，使当前 generation 的令牌失效并唤醒业务读写
    DbResult<void> ReleaseQuiesceGate(const QuiesceToken& token);

    /// 执行物理快照备份（仅限 BACKUP_ONLY 控制块）
    DbResult<BackupArtifact> Backup(const BackupRequest& request);

    /// 等待在途对象归零排空，按顺序关闭所有引擎
    DbResult<void> Close(std::chrono::milliseconds timeout);

    bool TryInstall() noexcept;
    void MarkRemoved() noexcept;
    bool Installed() const noexcept;
    bool BackupOnly() const noexcept;

    /// 会话同步状态观察点（只读真实状态，非运行期计数器；仅供测试精确编排并发时序：
    /// pendingOpenWaiters = 已登记为惰性打开等待者且尚未消费结果的线程数；
    /// sessionQuiescing = 静止闸门当前是否生效。生产代码不得依赖本观察点。）
    struct SyncStateForTest {
        std::size_t pendingOpenWaiters{0};
        bool sessionQuiescing{false};
    };
    SyncStateForTest SyncState() const noexcept;

private:
    struct Impl;
    explicit SessionControlBlock(std::unique_ptr<Impl> impl);
    /// 库引擎惰性打开（single-flight；调用前必须持有 impl_->mutex 的 unique_lock）：
    /// 打开者在标记 Opening 并登记在途计数后释放锁执行引擎 Open+Prepare（不阻塞其他库），
    /// 完成后重新加锁写结果并唤醒等待者；等待者按轮次代差共享同轮结果。
    /// underQuiesceToken：调用方持有本会话当前有效 quiesce token（受控 maintenance
    /// 路径）——quiescing 状态下仍允许打开（普通 CRUD 不得使用该特权）。
    DbResult<void> EnsureRouteOpen(
        std::unique_lock<std::mutex>& lock, SessionDatabaseRoute& route, bool keepInFlightCount,
        bool underQuiesceToken = false);
    std::unique_ptr<Impl> impl_;
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_INTERNAL_SESSION_CONTROL_H
