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

// 文件职责：定义 SQLite 特化引擎 SQLiteEngine。
// 本类直接实现 IRelationalEngine 全部显式能力方法：物理连接与 PRAGMA、
// 静态 schema 发布、每类 CRUD 的官方直通链（prepare/bind/step/column/finalize）、
// 动态 PID 表惰性建表、BEGIN IMMEDIATE/COMMIT/ROLLBACK 事务与 sqlite3_backup。
// 表与字段定义只消费 Table 传入的 CatalogTableMeta（静态目录唯一事实源）；
// 不建 tableId map、column map、adapter registry 或语句缓存。
// 本目录不识别任何业务数据库名——连接参数全部来自强类型 SqliteConnectionProfile。
#ifndef HCCL_VM_STORAGE_ENGINE_SQLITE_SQLITE_ENGINE_H
#define HCCL_VM_STORAGE_ENGINE_SQLITE_SQLITE_ENGINE_H

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "storage/database_result.h"
#include "storage/engine/i_relational_engine.h"
#include "storage/internal/engine_contract_types.h"
#include "storage/query_spec.h"
#include "storage/storage_catalog.h"

namespace HcclSim::Storage {
/// SQLite 连接 profile（强类型部署参数；composition 按库生成，本引擎只消费，
/// 不识别业务数据库名）
struct SqliteConnectionProfile {
    int busyTimeoutMs{10000};        ///< 连接建立期 PRAGMA 的 BUSY 等待时限
    bool gradientBusyHandler{false}; ///< true → 梯度延迟 busy handler（延迟阶梯有界重试）
    int walAutocheckpoint{200};      ///< WAL 自动检查点页数
};

/// SQLite 特化引擎：物理连接、参数化 SQL、动态表名、DDL、备份与错误码转换。
/// 唯一执行链：每类 CRUD 单次生成参数化 SQL → sqlite3_prepare_v2 → 按相同顺序
/// 直接 bind 原值（原始 TEXT/BLOB 用 SQLITE_STATIC）→ step/column（SELECT 经
/// catalog materializeRows 直写最终业务成员）→ finalize。批量 INSERT 在一批内
/// 复用同一条语句（一次 prepare、逐行 bind/step、行间 reset+clear_bindings），
/// 语句不跨调用缓存。
class SQLiteEngine final : public IRelationalEngine {
public:
    /// databasePath 是 SQLite 数据文件路径；catalog 为该逻辑库的静态目录
    /// （owner Prepare 遍历其全部静态表/索引；本引擎不复制目录数据）。
    SQLiteEngine(std::string databasePath, const Ddl::DatabaseDescriptor* catalog, SqliteConnectionProfile profile);
    // 析构只释放连接资源；不持久化、不重试、不做业务清理。
    ~SQLiteEngine() override;

    SQLiteEngine(const SQLiteEngine&) = delete;
    SQLiteEngine& operator=(const SQLiteEngine&) = delete;

    DbResult<void> Open(const EngineOpenRequest&) override;
    DbResult<void> Prepare() override;
    DbResult<uint64_t> Insert(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* record,
        TransactionBinding* transaction) override;
    /// 批量插入 override：一批只生成一次 SQL/prepare 一次，逐行 bind/step/reset
    /// （取真实生成主键；错误语义/回滚合同与基类默认逐行实现一致）
    DbResult<void> InsertMany(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* const* records, std::size_t count,
        uint64_t* generatedIds, TransactionBinding* transaction) override;
    DbResult<void> Select(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
        const QueryOptions& options, ResultSink& sink) override;
    DbResult<uint64_t>
    Count(const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where) override;
    DbResult<CountMatchResult> CountMatches(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
        const ConditionData& match) override;
    DbResult<uint64_t> Update(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
        const UpdateData& update, TransactionBinding* transaction) override;
    DbResult<uint64_t> Delete(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
        TransactionBinding* transaction) override;
    DbResult<void> Clear(const Ddl::CatalogTableMeta& meta, const TableScope& scope, ClearMode mode) override;
    DbResult<std::vector<std::string>> ListTables() override;
    DbResult<void> Begin(TransactionBinding& transaction) override;
    DbResult<AtomicResult> Commit(TransactionBinding& transaction) override;
    DbResult<void> Rollback(TransactionBinding& transaction) override;
    DbResult<void> Close() override;
    DbResult<DatabaseBackupArtifact> BackupRelational(const EngineBackupRequest&) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // 即时/事务 DML 公共状态检查（transaction 为空要求 PREPARED；非空要求
    // 已 Begin 且 binding 匹配；DML 只执行并返回错误，失败回滚由 Transaction
    // 统一调用 Rollback 完成）
    DbResult<void> CheckDmlState(TransactionBinding* transaction);
    // 只读/维护操作状态检查（要求 PREPARED，活动事务期间一律拒绝）
    DbResult<void> CheckOperationState(bool allowBackupOnly = true);
    // 唯一的 DML 直通实现（即时与事务路径共用；调用前已持锁并完成状态校验）
    DbResult<uint64_t> InsertLocked(const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* record);
    DbResult<uint64_t> UpdateLocked(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
        const UpdateData& update);
    DbResult<uint64_t>
    DeleteLocked(const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where);
    // 物理事务边界（持锁路径）
    DbResult<void> RollbackPhysicalLocked();
    DbResult<void> CloseLocked();

    // schema DDL 观察快照：静态 DDL（owner 建表/索引）与动态表惰性建表的累计执行
    // 次数；仅供单元测试固化结构合同，产品代码不得调用。
    struct SchemaDdlSnapshot {
        uint64_t staticDdlExecutions{0}; ///< 静态表/索引 DDL 执行累计（owner 路径）
        uint64_t dynamicTableCreates{0}; ///< 动态 scope 物理表惰性建表累计
    };
    SchemaDdlSnapshot SchemaDdlStatistics() const;

    // 引擎实际连接上的 PRAGMA 读取（测试可见性）：连接级 PRAGMA（如
    // synchronous/secure_delete）只有本连接上的读数才证明已生效，另开 CLI
    // 连接检查无效。仅供单元测试，产品代码不得调用。
    DbResult<std::string> ReadConnectionPragmaForTest(const char* name) const;
    // 事务等待上限测试可见性（仅测试 peer 经 friend 调用；产品代码不得依赖）：
    // 把私有等待上限缩短到毫秒级，验证超时路径的 RESOURCE_EXHAUSTED 合同。
    void SetTransactionWaitLimitForTest(std::chrono::milliseconds limit);

    // 仅供故障注入测试（产品代码不得设置）：真实 WAL 连接上 ROLLBACK 失败不可构造，
    // 置位后下一次 Prepare 失败路径的 ROLLBACK 按注入失败处理。
    bool failNextRollbackForTest_{false};
    // 语句缓存单元测试观察点：仅在测试文件中定义与使用，产品代码不得引用。
    friend struct SQLiteEngineTestPeer;
    // ROLLBACK 失败故障注入测试点（真实 WAL 连接上 ROLLBACK 失败不可构造）：仅在
    // 测试文件中定义与使用，产品代码不得引用。
    friend struct SQLiteRollbackFaultTestPeer;
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_ENGINE_SQLITE_SQLITE_ENGINE_H
