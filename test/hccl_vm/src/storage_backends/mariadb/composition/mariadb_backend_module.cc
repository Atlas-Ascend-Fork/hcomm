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

// 文件职责：MariaDB 后端模块的 composition 实现。全部行为自迁移前公共
// composition 原样迁入（v1 单实例 K=1：部署 uri + 三类超时（秒）+ 线路协议压缩
// 开关，无清单概念；物理库名由引擎按 dataNamespace + databaseId 派生）：
// - 默认路由：endpoint 即部署 uri，endpoint/profile.uri 一致性由引擎工厂校验；
// - 清库：host-only 维护边界，逐逻辑库 transient 引擎 Wipe（不建业务会话）；
// - 备份：专用备份连接，BACKUP_ONLY + 一致性快照 dump（自产 .sql 文本）；
// - 历史位置恢复：保持部署路由（数据位置由部署配置唯一决定，与任务文件
//   记录路径无关），公共层不得把 MariaDB 部署重配成文件路径。
#include "storage_backends/mariadb/composition/mariadb_backend_module.h"

#include <memory>
#include <string>
#include <utility>

#include "storage/database_result.h"
#include "storage/engine/i_relational_engine.h"
#include "storage/internal/engine_contract_types.h"
#include "storage/storage_runtime.h"
#include "storage_backends/mariadb/composition/mariadb_deploy_section.h"
#include "storage_backends/mariadb/engine/mariadb_backend_factory.h"

namespace sim::backends {
namespace {
    using HcclSim::Storage::DbCode;
    using HcclSim::Storage::DbResult;
    using HcclSim::Storage::LogicalDatabaseRuntimeConfig;
    using HcclSim::Storage::MariadbConnectionProfile;
    using HcclSim::Storage::SchemaRole;

    constexpr char kProfileMissing[] = "mariadb deployment profile is missing: configure storage.mariadb.uri "
                                       "(backend=mariadb or an explicit mariadb section)";

    DbResult<LogicalDatabaseRuntimeConfig>
    MakeMariadbRoute(const BackendRouteRequest& request, const std::shared_ptr<const MariadbConnectionProfile>& profile)
    {
        if (request.descriptor == nullptr) {
            return {
                DbCode::INVALID_ARGUMENT, std::nullopt, 0, "mariadb route request is missing the catalog descriptor"};
        }
        if (profile == nullptr || profile->uri.empty()) {
            return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, kProfileMissing};
        }
        return {
            DbCode::OK,
            std::optional<LogicalDatabaseRuntimeConfig>{LogicalDatabaseRuntimeConfig{
                std::string{request.descriptor->databaseId}, kMariadbBackendName, profile->uri,
                SchemaRole::OwnerPrepare}},
            0,
            {}};
    }

    DbResult<std::unique_ptr<HcclSim::Storage::IStorageEngine>> CreateMariaDbEngineFromRoute(
        const BackendEngineRequest& request, const std::shared_ptr<const MariadbConnectionProfile>& profile)
    {
        if (request.descriptor == nullptr) {
            return {
                DbCode::INVALID_ARGUMENT, std::nullopt, 0, "mariadb engine request is missing the catalog descriptor"};
        }
        if (profile == nullptr || profile->uri.empty()) {
            return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, kProfileMissing};
        }
        // endpoint 与 profile.uri 的一致性由工厂校验（v1 单实例 K=1）。
        return HcclSim::Storage::CreateMariadbEngine(*request.descriptor, request.route, *profile);
    }

    DbResult<void> WipeMariadbDeployment(
        const BackendEngineRequest& request, const std::string& dataNamespace,
        const std::shared_ptr<const MariadbConnectionProfile>& profile)
    {
        if (request.descriptor == nullptr) {
            return {DbCode::INVALID_ARGUMENT, 0, "mariadb wipe request is missing the catalog descriptor"};
        }
        if (profile == nullptr || profile->uri.empty()) {
            return {DbCode::INVALID_ARGUMENT, 0, kProfileMissing};
        }
        // 逐逻辑库创建 transient 引擎调用其 Wipe()（不建立业务会话、不 Prepare）。
        auto engine = HcclSim::Storage::CreateMariadbEngine(
            *request.descriptor,
            LogicalDatabaseRuntimeConfig{
                request.route.database, kMariadbBackendName, profile->uri, SchemaRole::OwnerPrepare},
            *profile);
        if (!engine.ok()) {
            return {engine.code, engine.affectedRows, engine.diagnostic};
        }
        auto wiped = (*engine.value)->Wipe(dataNamespace);
        if (!wiped.ok()) {
            return {wiped.code, wiped.affectedRows, wiped.diagnostic};
        }
        return {DbCode::OK, 0, {}};
    }

    DbResult<HcclSim::Storage::BackupArtifact> BackupMariadbDatabase(
        const BackendEngineRequest& request, const std::string& dataNamespace,
        const HcclSim::Storage::BackupRequest& backupRequest,
        const std::shared_ptr<const MariadbConnectionProfile>& profile)
    {
        if (request.descriptor == nullptr) {
            return {
                DbCode::INVALID_ARGUMENT, std::nullopt, 0, "mariadb backup request is missing the catalog descriptor"};
        }
        if (profile == nullptr || profile->uri.empty()) {
            return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, kProfileMissing};
        }
        // 专用备份连接：经 MariaDB 工厂按部署 profile 创建独立引擎，BACKUP_ONLY
        // 打开后执行 START TRANSACTION WITH CONSISTENT SNAPSHOT 快照 dump（v1：
        // 自产 .sql 文本，mariadb 客户端可回放；不占用业务会话）。活动事务存在
        // 时引擎侧拒绝（CheckOperationState）。
        auto engine = HcclSim::Storage::CreateMariadbEngine(
            *request.descriptor,
            LogicalDatabaseRuntimeConfig{
                request.route.database, kMariadbBackendName, profile->uri, SchemaRole::OwnerPrepare},
            *profile);
        if (!engine.ok()) {
            return {engine.code, std::nullopt, engine.affectedRows, engine.diagnostic};
        }
        auto& relational = static_cast<HcclSim::Storage::IRelationalEngine&>(**engine.value);
        HcclSim::Storage::EngineOpenRequest openRequest;
        openRequest.mode = HcclSim::Storage::StorageOpenMode::BACKUP_ONLY;
        // 物理库名派生与业务会话一致（hcclvm_<namespace>_<databaseId>）：备份
        // 引擎必须命中会话所用的同一 database，namespace 不一致会 dump 到空库。
        openRequest.dataNamespace = dataNamespace;
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
    ResolveMariadbRecordedLocation(const LogicalDatabaseRuntimeConfig& currentRoute, const std::string& recordedPath)
    {
        (void)recordedPath;
        // MariaDB 部署保持路由：任务文件记录的 SQLite 路径只在文件部署语义下有意义
        // （换库/加载备份）；MariaDB 部署的数据位置由部署 uri + dataNamespace 派生
        // （HCCL_VM_STORAGE_CONFIG 唯一决定），与任务文件记录路径无关，不得回退到
        // 文件路径——Runner/Checker 的 op 数据查询必须继续走既有路由。
        // 成功不换库的合同形状：外层载荷必须存在，内层 optional 为空（见
        // backend_module.h；成功而缺外层载荷会被公共层按合同违规拒绝）。
        (void)currentRoute;
        return {DbCode::OK, std::optional<LogicalDatabaseRuntimeConfig>{}, 0, {}};
    }

    /// 按 profile（可为空=未配置）建立部署对象：全部回调就位，未配置时按需报
    /// 配置缺失错误，而不是空回调。
    BackendDeployment MakeMariadbDeployment(const std::shared_ptr<const MariadbConnectionProfile>& profile)
    {
        BackendDeployment deployment;
        deployment.makeRoute = [profile](const BackendRouteRequest& request) {
            return MakeMariadbRoute(request, profile);
        };
        deployment.createEngine = [profile](const BackendEngineRequest& request) {
            return CreateMariaDbEngineFromRoute(request, profile);
        };
        deployment.wipe = [profile](const BackendEngineRequest& request, const std::string& dataNamespace) {
            return WipeMariadbDeployment(request, dataNamespace, profile);
        };
        deployment.backup = [profile](
                                const BackendEngineRequest& request, const std::string& dataNamespace,
                                const HcclSim::Storage::BackupRequest& backupRequest) {
            return BackupMariadbDatabase(request, dataNamespace, backupRequest, profile);
        };
        deployment.resolveRecordedLocation = ResolveMariadbRecordedLocation;
        return deployment;
    }

    DbResult<BackendDeployment> ConfigureMariadbDeployment(const BackendConfigureRequest& request)
    {
        if (!request.sectionPresent) {
            if (request.selected) {
                return {
                    DbCode::INVALID_ARGUMENT, std::nullopt, 0, "storage.mariadb is required when backend is mariadb"};
            }
            // 未被选中且无配置段：缺少 profile 的部署对象（显式路由可命中，实际
            // 用于创建引擎/路由时返回配置缺失错误，不阻止默认后端正常启动）。
            return {DbCode::OK, std::optional<BackendDeployment>{MakeMariadbDeployment(nullptr)}, 0, {}};
        }
        auto profile = ParseMariadbDeploySection(request.section);
        if (!profile.ok()) {
            return {profile.code, std::nullopt, 0, profile.diagnostic};
        }
        // 已编入但未被选中的模块出现配置段同样完成校验并保留解析结果：
        // 显式路由（维护换库/测试装配）可继续选择该后端。
        return {
            DbCode::OK,
            std::optional<BackendDeployment>{
                MakeMariadbDeployment(std::make_shared<const MariadbConnectionProfile>(std::move(*profile.value)))},
            0,
            {}};
    }
} // namespace

const BackendModule& MariadbBackendModule()
{
    static const BackendModule module = {kMariadbBackendName, false, ConfigureMariadbDeployment};
    return module;
}
} // namespace sim::backends
