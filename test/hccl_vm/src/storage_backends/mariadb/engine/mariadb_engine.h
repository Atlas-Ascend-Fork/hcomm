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

// 文件职责：定义 MariaDB 特化引擎 MariaDbEngine（第二外部存储后端，与 SQLite
// 并存，互不共享代码）。
// 【选型理由】SQLite 跨进程单写者（文件锁 + BEGIN IMMEDIATE 全库串行）无法
// 支撑 512 rank 并发写；MariaDB InnoDB 原生 MVCC 提供跨进程并发读写与一致性
// 快照读。
// 【已知限制（v1）】进程内单条长连接（与 SQLite 进程内单句柄同构，仅串行化
// 本进程多线程）；跨 rank 并发由 MariaDB 服务端线程并行提供。若 128p 实测
// 显示进程内串行成为瓶颈，v2 再评估每引擎小连接池（本版明确禁止）。无自动
// 重连：连接死亡 → 语句缓存清空、引擎进入终态、错误如实上抛。
#ifndef HCCL_VM_STORAGE_ENGINE_MARIADB_MARIADB_ENGINE_H
#define HCCL_VM_STORAGE_ENGINE_MARIADB_MARIADB_ENGINE_H

#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "storage/database_result.h"
#include "storage/engine/i_relational_engine.h"
#include "storage_backends/mariadb/engine/mariadb_connection_profile.h"
#include "storage/internal/engine_contract_types.h"
#include "storage/query_spec.h"
#include "storage/storage_catalog.h"

namespace HcclSim::Storage {
/// MariaDB 特化引擎：一条长连接 + 语句缓存 LRU + 参数化 SQL 直通链（生成 SQL →
/// 缓存取 MYSQL_STMT → MYSQL_BIND 绑定原值 → execute/fetch → affected/insert_id）
/// → catalog materializeRows 直写最终业务成员。DDL/事务边界经 mysql_real_query。
/// 事务：START TRANSACTION/COMMIT/ROLLBACK；autocommit=1（非事务 DML 逐句提交，
// 对齐 SQLite 立即模式）。所有 SELECT 走默认 unbuffered 取行（禁止
/// CURSOR_TYPE_READ_ONLY 与 store_result 全量缓冲）。
class MariaDbEngine final : public IRelationalEngine {
public:
    /// catalog 为该逻辑库的静态目录（owner Prepare 遍历其全部静态表/索引；
    /// 本引擎不复制目录数据）；profile 含 endpoint URI 与三类超时（强类型部署
    /// 参数，由 composition 按部署配置构造，本引擎只消费不识别业务库名）。
    /// 物理 database 名由 Open 请求的 dataNamespace 与目录 databaseId 共同
    /// 派生（每进程两实例 = 两条连接 = 两个 database）。
    explicit MariaDbEngine(const Ddl::DatabaseDescriptor* catalog, MariadbConnectionProfile profile);
    /// 析构只释放连接资源（先清语句缓存再 mysql_close）；不持久化、不重试。
    ~MariaDbEngine() override;

    MariaDbEngine(const MariaDbEngine&) = delete;
    MariaDbEngine& operator=(const MariaDbEngine&) = delete;

    DbResult<void> Open(const EngineOpenRequest&) override;
    DbResult<void> Prepare() override;
    DbResult<uint64_t> Insert(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* record,
        TransactionBinding* transaction) override;
    /// 批量插入 override（T1 真批插）：合格批次（单列自增主键、无 Unique
    /// 约束/外键、全批主键 0/NULL、服务端 10.5+）走单条参数化多值
    /// INSERT ... RETURNING 主键（一次 execute，缩短事务占用窗口）；不满足
    /// 任一条件自动回退基类默认逐行实现（错误语义/首错行号/回滚合同逐字
    /// 一致，功能不裁剪）
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
    /// host 启动专用：临时连 server 级（不选库）DROP DATABASE IF EXISTS 物理库
    /// （hcclvm_<namespace>_<databaseId>），等价 SQLite 删库文件。仅 owner 调用。
    DbResult<void> Wipe(std::string_view dataNamespace) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // 即时/事务 DML 公共状态检查（transaction 为空要求 PREPARED；非空要求已
    // Begin 且 binding 匹配且事务未被死锁中止）
    DbResult<void> CheckDmlState(TransactionBinding* transaction);
    // 只读/维护操作状态检查（要求 PREPARED，活动事务期间一律拒绝）
    DbResult<void> CheckOperationState(bool allowBackupOnly = true);
    // InsertMany 快路径资格预检（静态；只读目录与调用方记录，不持锁、不碰
    // 连接状态）：目录形状 + 逐行记录形态（指针非空、visit 成功、主键 0/NULL）
    static bool
    InsertManyFastPathEligible(const Ddl::CatalogTableMeta& meta, const void* const* records, std::size_t count);
    // InsertMany 快路径主体（调用前已持锁、已通过 CheckDmlState 与服务端
    // 版本检查）：一次物理表解析/SQL 构造/prepare/绑定/execute，RETURNING
    // 结果逐行回填真实主键
    DbResult<void> InsertManyFastPathLocked(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* const* records, std::size_t count,
        uint64_t* generatedIds);
    // 唯一的 DML 直通实现（即时与事务路径共用；调用前已持锁并完成状态校验）
    DbResult<uint64_t> InsertLocked(const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* record);
    DbResult<uint64_t> UpdateLocked(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
        const UpdateData& update);
    DbResult<uint64_t>
    DeleteLocked(const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where);
    // Select/Count 主体（调用前已持锁）；allowRevive=false 表示已是复活重试轮
    DbResult<void> SelectLocked(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
        const QueryOptions& options, ResultSink& sink, bool allowRevive);
    DbResult<uint64_t> CountLocked(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where, bool allowRevive);
    DbResult<CountMatchResult> CountMatchesLocked(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
        const ConditionData& match, bool allowRevive);
    // 物理事务边界（持锁路径）
    DbResult<void> RollbackPhysicalLocked();
    DbResult<void> CloseLocked();
    // 备份单表数据行 dump（持锁路径；BackupRelational 的逐表子步骤）
    DbResult<void> DumpTableRowsLocked(std::ofstream& output, const std::string& tableName);
    // 连接死亡处置（v1 无自动重连）：清语句缓存、关闭连接、引擎进入终态。
    // 调用方负责把原始错误如实上抛。
    void AbandonConnectionLocked();

    // 语句缓存观察快照：仅供单元测试固化缓存合同，产品代码不得调用。
    struct StatementCacheSnapshot {
        uint64_t hits{0};    ///< 命中次数（省去的 prepare 网络往返）
        uint64_t misses{0};  ///< 未命中次数（实际执行 prepare）
        std::size_t size{0}; ///< 当前缓存语句数
    };
    StatementCacheSnapshot StatementCacheStatistics() const;

    // 测试可见性（仅测试 peer 经 friend 调用；产品代码不得依赖）：
    // - 缩短私有事务等待上限（毫秒级），验证超时路径的 RESOURCE_EXHAUSTED；
    // - 批插快路径（多值 INSERT ... RETURNING）执行计数——证明合格批次只
    //   执行一次批语句；
    // - 服务端版本 seam——模拟 <10.5 旧服务端验证自动回退逐行路径。
    void SetTransactionWaitLimitForTest(std::chrono::milliseconds limit);
    uint64_t BatchInsertExecutionsForTest() const;
    void SetServerVersionForTest(unsigned long version);

    // 语句缓存观察点：仅在测试文件中定义与使用，产品代码不得引用。
    friend struct MariadbEngineTestPeer;
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_ENGINE_MARIADB_MARIADB_ENGINE_H
