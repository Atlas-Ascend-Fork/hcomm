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

// 文件职责：SQLite 后端模块的 composition 实现（默认路由/引擎构造/清库 no-op/
// 专用备份连接/历史位置重配）。全部行为自迁移前公共 composition 原样迁入：
// - RunnerDB（hccl_sim.db）：busy_timeout=60000、梯度 BusyHandler、
//   wal_autocheckpoint=1000；OP_DATA（hccl_vm_data.db）：busy_timeout=10000、
//   wal_autocheckpoint=200（与迁移前基线逐库一致）；
// - 清库为 no-op：SQLite 部署的文件由 envInit 删除，模块不重复删除；
// - 备份：经通用 SQLite 工厂按当前路由创建独立引擎，BACKUP_ONLY 打开后执行
//   sqlite3_backup 在线快照（不占用业务会话）；
// - 历史位置恢复：重配到任务文件记录的路径（等价迁移前"换库/加载备份"语义）。
#include "storage_backends/sqlite/composition/sqlite_backend_module.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "storage/database_result.h"
#include "storage/engine/i_relational_engine.h"
#include "storage/internal/engine_contract_types.h"
#include "storage/storage_runtime.h"
#include "storage_backends/sqlite/engine/sqlite_backend_factory.h"

namespace sim::backends {
namespace {
    using HcclSim::Storage::DbCode;
    using HcclSim::Storage::DbResult;
    using HcclSim::Storage::LogicalDatabaseRuntimeConfig;
    using HcclSim::Storage::SchemaRole;
    using HcclSim::Storage::SqliteConnectionProfile;

    /// 两个业务逻辑库的 canonical 标识（与 include/runtime_state/db_sim_runner_ops.h
    /// 与 include/operation_data/operation_data_ops.h 的目录声明一致）——模块按名
    /// 装配默认路径与连接 profile，不包含业务头。
    constexpr std::string_view kRunnerDatabaseId = "RUNNER_DB";
    constexpr std::string_view kOpDataDatabaseId = "OP_DATA_DB";

    /// 按逻辑库构造强类型 SQLite 连接 profile（部署知识单点驻留本模块 composition：
    /// SQLite 引擎只消费 profile，不识别业务库名）。逐库对齐基线真实配置：
    /// - RunnerDB（hccl_sim.db，运行时模型库）：基线 sim_sqlite_table.h 连接设
    ///   wal_autocheckpoint=1000 并注册梯度 BusyHandler（1,2,5,10,20,50,100,200,500ms）。
    /// - OP_DATA（hccl_vm_data.db，任务/算子库）：基线 db_hccl_db_sqlite.cc 连接设
    ///   wal_autocheckpoint=200、busy_timeout=10000。
    SqliteConnectionProfile SqliteProfileOf(const std::string& databaseId)
    {
        if (databaseId == kRunnerDatabaseId) {
            return SqliteConnectionProfile{60000, true, 1000};
        }
        return SqliteConnectionProfile{10000, false, 200};
    }

    /// 安装根下的默认物理文件（与迁移前基线路径一致）。
    DbResult<LogicalDatabaseRuntimeConfig> MakeSqliteRoute(const BackendRouteRequest& request)
    {
        if (request.descriptor == nullptr) {
            return {
                DbCode::INVALID_ARGUMENT, std::nullopt, 0, "sqlite route request is missing the catalog descriptor"};
        }
        if (request.descriptor->databaseId == kRunnerDatabaseId) {
            return {
                DbCode::OK,
                std::optional<LogicalDatabaseRuntimeConfig>{LogicalDatabaseRuntimeConfig{
                    std::string{request.descriptor->databaseId}, kSqliteBackendName,
                    request.installRoot + "/data/hccl_sim.db", SchemaRole::OwnerPrepare}},
                0,
                {}};
        }
        if (request.descriptor->databaseId == kOpDataDatabaseId) {
            return {
                DbCode::OK,
                std::optional<LogicalDatabaseRuntimeConfig>{LogicalDatabaseRuntimeConfig{
                    std::string{request.descriptor->databaseId}, kSqliteBackendName,
                    request.installRoot + "/data/hccl_vm_data.db", SchemaRole::OwnerPrepare}},
                0,
                {}};
        }
        return {
            DbCode::INVALID_ARGUMENT, std::nullopt, 0,
            "sqlite module has no default route for database " + std::string{request.descriptor->databaseId}};
    }

    DbResult<std::unique_ptr<HcclSim::Storage::IStorageEngine>>
    CreateSqliteEngineFromRoute(const BackendEngineRequest& request)
    {
        if (request.descriptor == nullptr) {
            return {
                DbCode::INVALID_ARGUMENT, std::nullopt, 0, "sqlite engine request is missing the catalog descriptor"};
        }
        // 引擎装配全部由通用工厂从描述符派生，连接 profile 由本模块按库配置构造。
        return HcclSim::Storage::CreateSqliteEngine(
            *request.descriptor, request.route.endpoint, SqliteProfileOf(request.route.database));
    }

    DbResult<void> WipeSqliteDeployment(const BackendEngineRequest& request, const std::string& dataNamespace)
    {
        (void)request;
        (void)dataNamespace;
        // SQLite 部署无需处理：文件已由 envInit 删除（干净起点）；模块保持 no-op。
        return {DbCode::OK, 0, {}};
    }

    DbResult<HcclSim::Storage::BackupArtifact> BackupSqliteDatabase(
        const BackendEngineRequest& request, const std::string& dataNamespace,
        const HcclSim::Storage::BackupRequest& backupRequest)
    {
        (void)dataNamespace;
        if (request.descriptor == nullptr) {
            return {
                DbCode::INVALID_ARGUMENT, std::nullopt, 0, "sqlite backup request is missing the catalog descriptor"};
        }
        // 专用备份连接：经通用 SQLite 工厂按当前路由创建独立引擎，打开既有文件后执行
        // sqlite3_backup 在线快照（等价迁移前备份只读连接装配；不占用业务会话）。
        auto engine = HcclSim::Storage::CreateSqliteEngine(
            *request.descriptor, request.route.endpoint, SqliteProfileOf(request.route.database));
        if (!engine.ok()) {
            return {engine.code, std::nullopt, engine.affectedRows, engine.diagnostic};
        }
        auto& relational = static_cast<HcclSim::Storage::IRelationalEngine&>(**engine.value);
        HcclSim::Storage::EngineOpenRequest openRequest;
        openRequest.mode = HcclSim::Storage::StorageOpenMode::BACKUP_ONLY;
        auto opened = relational.Open(openRequest);
        if (!opened.ok()) {
            (void)relational.Close();
            return {opened.code, std::nullopt, opened.affectedRows, opened.diagnostic};
        }
        auto prepared = relational.Prepare();
        if (!prepared.ok()) {
            (void)relational.Close();
            return {prepared.code, std::nullopt, prepared.affectedRows, prepared.diagnostic};
        }
        auto backed = relational.BackupRelational(HcclSim::Storage::EngineBackupRequest{backupRequest.destinationPath});
        (void)relational.Close();
        if (!backed.ok()) {
            return {backed.code, std::nullopt, backed.affectedRows, backed.diagnostic};
        }
        return {DbCode::OK, HcclSim::Storage::BackupArtifact{backed.value->destinationPath}, 0, {}};
    }

    DbResult<std::optional<LogicalDatabaseRuntimeConfig>>
    ResolveSqliteRecordedLocation(const LogicalDatabaseRuntimeConfig& currentRoute, const std::string& recordedPath)
    {
        (void)currentRoute;
        // SQLite 部署：等价迁移前"换库/加载备份"语义——重配到任务文件记录的路径。
        return {
            DbCode::OK,
            std::optional<LogicalDatabaseRuntimeConfig>{
                LogicalDatabaseRuntimeConfig{currentRoute.database, kSqliteBackendName, recordedPath}},
            0,
            {}};
    }

    DbResult<BackendDeployment> ConfigureSqliteDeployment(const BackendConfigureRequest& request)
    {
        if (request.sectionPresent) {
            // SQLite 部署不取任何配置段：显式出现即拒绝（部署意图不明宁可装配失败）。
            return {
                DbCode::INVALID_ARGUMENT, std::nullopt, 0,
                "storage.sqlite section is not supported: the sqlite backend takes no configuration"};
        }
        BackendDeployment deployment;
        deployment.makeRoute = MakeSqliteRoute;
        deployment.createEngine = CreateSqliteEngineFromRoute;
        deployment.wipe = WipeSqliteDeployment;
        deployment.backup = BackupSqliteDatabase;
        deployment.resolveRecordedLocation = ResolveSqliteRecordedLocation;
        return {DbCode::OK, std::optional<BackendDeployment>{std::move(deployment)}, 0, {}};
    }
} // namespace

const BackendModule& SqliteBackendModule()
{
    static const BackendModule module = {kSqliteBackendName, true, ConfigureSqliteDeployment};
    return module;
}
} // namespace sim::backends
