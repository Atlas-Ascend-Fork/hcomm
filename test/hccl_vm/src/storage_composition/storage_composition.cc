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

// 文件职责：模拟器存储装配实现（唯一 composition 实现文件，后端中立）。
// 已编入的后端模块经生成的注册清单（compiled_backends.cc，构建目录内）显式
// 接入：本文件只做模块发现结果的消费与分派——注册工厂、解析部署配置、按选中/
// 默认部署构造逻辑路由、按当前路由分派清库/备份/历史位置恢复。具体引擎、工厂、
// profile 与配置段规则全部驻留在 src/storage_backends/<模块名>/，公共层不出现
// 具体后端名，也不按后端名做条件分支。
// 两个业务目录在此登记；无参 EnsureSimulatorStorageSession() 同时是登记进
// ProcessStorageContext 的应用级 ready-session bootstrap：冷进程首次原子/业务
// API 调用经 AcquireReady() 惰性回调本函数建立默认会话（single-flight）；
// 已知初始化边界（start/run/device-arm/aclInit）与维护入口直接调用同一函数
// 提前 fail-fast，不是唯一保障。
#include "storage_composition/storage_composition.h"

#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include "sim_common_api.h"
#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/internal/engine_contract_types.h"
#include "storage/internal/process_storage_context.h"
#include "storage/internal/storage_runtime_impl.h"
#include "storage/storage_runtime.h"
#include "storage_composition/backend_module.h"
#include "storage_composition/compiled_backends.h"
#include "storage_composition/storage_deploy_config.h"

namespace sim {
namespace {
    /// 模拟器存储部署配置环境变量（主控/rank/plugin/device 子进程正常继承）。
    /// 只在本 composition bootstrap 读取；每个进程解析一次，形成不可变 runtime config。
    constexpr char kStorageConfigEnv[] = "HCCL_VM_STORAGE_CONFIG";

    /// 进程级部署配置快照（bootstrap 解析一次，进程生命周期内不可变）。
    /// selection.backend 是已解析的实际后端模块注册名；status 非空错误时
    /// selection 仍保留各模块的部署描述（显式路由可继续使用已编入后端）。
    struct StorageDeploySnapshot {
        HcclSim::Storage::StorageDeploySelection selection;
        HcclSim::Storage::DbResult<void> status{HcclSim::Storage::DbCode::OK, 0, {}};
    };

    /// 读取 HCCL_VM_STORAGE_CONFIG（函数局部 static：首个调用线程判定一次，之后
    /// 全部进程内访问复用同一不可变快照；不得静默回退到任何后端）。
    const StorageDeploySnapshot& DeployConfigSnapshot()
    {
        static const StorageDeploySnapshot snapshot = [] {
            using HcclSim::Storage::DbCode;
            const sim::backends::BackendModule* const* modules = sim::backends::CompiledBackendModules();
            const std::size_t moduleCount = sim::backends::CompiledBackendCount();
            StorageDeploySnapshot state;
            const char* configPath = std::getenv(kStorageConfigEnv);
            if (configPath != nullptr && configPath[0] != '\0') {
                auto parsed = HcclSim::Storage::ParseStorageDeployConfigFile(configPath);
                if (parsed.ok()) {
                    state.selection = std::move(*parsed.value);
                    return state; // 配置解析成功：部署描述已按模块建立
                }
                // 配置存在但解析失败：缓存失败结果（同进程重试保持同一失败）；
                // 仍建立无配置段的部署描述，显式路由不因部署配置错误而失去工厂。
                state.status
                    = {parsed.code, 0, std::string{kStorageConfigEnv} + "=" + configPath + ": " + parsed.diagnostic};
            } else {
                // 未设置配置：由唯一的默认候选提供默认部署；无默认候选（裁剪为
                // 无默认后端的模块组合）时明确失败，禁止按"只剩一个目录"自动猜测。
                std::size_t defaultCount = 0;
                for (std::size_t index = 0; index < moduleCount; ++index) {
                    if (modules[index]->providesDefault) {
                        ++defaultCount;
                        state.selection.backend = modules[index]->name;
                    }
                }
                if (defaultCount > 1) {
                    state.status
                        = {DbCode::INVALID_ARGUMENT, 0, "multiple default storage backend candidates are compiled in"};
                } else if (defaultCount == 0) {
                    state.status
                        = {DbCode::INVALID_ARGUMENT, 0,
                           std::string{"no default storage backend is compiled in; set "} + kStorageConfigEnv
                               + " to configure one explicitly"};
                    state.selection.backend.clear();
                }
            }
            // 无配置（或配置解析失败）路径：每个已编入模块按"无配置段"建立部署描述。
            for (std::size_t index = 0; index < moduleCount; ++index) {
                const sim::backends::BackendModule& module = *modules[index];
                sim::backends::BackendConfigureRequest request;
                request.selected = state.selection.backend == module.name;
                auto deployment = module.configure(request);
                if (!deployment.ok()) {
                    if (state.status.ok()) {
                        state.status = {deployment.code, 0, deployment.diagnostic};
                    }
                    continue;
                }
                state.selection.deployments[std::string{module.name}] = std::move(*deployment.value);
            }
            return state;
        }();
        return snapshot;
    }

    /// registration-only 静态注册器：把唯一 simulator bootstrap 函数指针登记进进程
    /// 会话注册表（只注册函数指针——不打开连接、不安装会话、不解析安装根路径）。
    /// 冷进程首次 Storage 原子 API / 业务兼容 API 调用由此经 ProcessStorageContext::
    /// AcquireReady() 惰性建立会话；本翻译单元因业务实现（operation_data 等）对
    /// sim::EnsureSimulatorStorageSession 的既有引用而必然进入实际链接产物。
    struct SimulatorStorageBootstrapRegistration {
        SimulatorStorageBootstrapRegistration()
        {
            (void)HcclSim::Storage::ProcessStorageContext::RegisterBootstrap(&EnsureSimulatorStorageSession);
        }
    };

    const SimulatorStorageBootstrapRegistration g_simulatorStorageBootstrapRegistration;

    /// 当前逻辑路由登记（默认路由 + 维护换库更新；备份等观测由此取得当前端点）。
    /// backend 为后端模块注册名；初值由选中/默认部署的 MakeRoute 得出——公共层
    /// 不写任何后端专属路径。
    struct RouteRecord {
        HcclSim::Storage::StorageBackend backend;
        std::string endpoint;
    };

    std::mutex& RouteMutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    std::map<std::string, RouteRecord>& CurrentRoutes()
    {
        static std::map<std::string, RouteRecord> routes = [] {
            std::map<std::string, RouteRecord> initial;
            const auto& deploy = DeployConfigSnapshot();
            if (!deploy.status.ok() || deploy.selection.backend.empty()) {
                return initial; // 部署配置失败/无默认候选：无初始路由，维护入口明确失败
            }
            const auto found = deploy.selection.deployments.find(deploy.selection.backend);
            if (found == deploy.selection.deployments.end() || !found->second.makeRoute) {
                return initial;
            }
            const std::string installRoot = InstallPath::GetHcclVmInstallAbsPath();
            const HcclSim::Storage::Ddl::DatabaseDescriptor* const databases[]
                = {&HcclSim::Storage::Ddl::kRunnerDatabase, &HcclSim::Storage::Ddl::kOpDataDatabase};
            for (const HcclSim::Storage::Ddl::DatabaseDescriptor* database : databases) {
                auto route = found->second.makeRoute({database, installRoot});
                if (!route.ok()) {
                    continue; // 该库无默认路由：不登记，维护入口按缺路由明确失败
                }
                initial[std::string{database->databaseId}] = {route.value->backend, route.value->endpoint};
            }
            return initial;
        }();
        return routes;
    }

    /// 逻辑库对应的业务目录描述符（未知库返回空）。
    const HcclSim::Storage::Ddl::DatabaseDescriptor* CatalogOf(const std::string& databaseId)
    {
        if (databaseId == HcclSim::Storage::Ddl::kRunnerDatabase.databaseId) {
            return &HcclSim::Storage::Ddl::kRunnerDatabase;
        }
        if (databaseId == HcclSim::Storage::Ddl::kOpDataDatabase.databaseId) {
            return &HcclSim::Storage::Ddl::kOpDataDatabase;
        }
        return nullptr;
    }

    /// 查找逻辑库当前路由（无登记返回空）。
    std::optional<RouteRecord> RouteOf(const std::string& databaseId)
    {
        std::lock_guard<std::mutex> lock(RouteMutex());
        const auto found = CurrentRoutes().find(databaseId);
        if (found == CurrentRoutes().end()) {
            return std::nullopt;
        }
        return found->second;
    }

    /// 查找后端部署快照（模块未编入/部署未建立返回空）。
    const sim::backends::BackendDeployment* DeploymentOf(const std::string& backend)
    {
        const auto& deploy = DeployConfigSnapshot();
        const auto found = deploy.selection.deployments.find(backend);
        if (found == deploy.selection.deployments.end()) {
            return nullptr;
        }
        return &found->second;
    }

    /// 登记当前路由（会话安装/维护换库成功后调用）。
    void RecordRoutes(const HcclSim::Storage::StorageRuntimeConfig& config)
    {
        std::lock_guard<std::mutex> lock(RouteMutex());
        for (const auto& database : config.databases) {
            CurrentRoutes()[database.database] = {database.backend, database.endpoint};
        }
    }
} // namespace

HcclSim::Storage::DbResult<void> RegisterSimulatorStorageBackends()
{
    auto& runtime = HcclSim::Storage::detail::StorageRuntimeImpl::Instance();
    // 每个已编入模块按名注册工厂。工厂闭包【按值捕获】模块 createEngine 回调
    // （回调自身按值/shared_ptr 持有所需 profile，见 backend_module.h）——不得
    // 把部署快照内 BackendDeployment 的指针捕获进工厂：部署快照是普通静态
    // 存储，进程退出时析构，而 StorageRuntime 与已安装会话遵循既有 no-destroy
    // 合同，退出消费者仍可能经未打开的惰性路由首次调用工厂。部署快照缺失
    // （模块配置失败）时工厂按需返回后端不可用错误，不静默换后端。
    const sim::backends::BackendModule* const* modules = sim::backends::CompiledBackendModules();
    for (std::size_t index = 0; index < sim::backends::CompiledBackendCount(); ++index) {
        const sim::backends::BackendModule& module = *modules[index];
        const std::string backendName{module.name};
        const sim::backends::BackendDeployment* deployment = DeploymentOf(backendName);
        decltype(sim::backends::BackendDeployment::createEngine) createEngine{};
        if (deployment != nullptr) {
            createEngine = deployment->createEngine;
        }
        runtime.RegisterEngineFactory(
            backendName,
            [createEngine](const HcclSim::Storage::LogicalDatabaseRuntimeConfig& config)
                -> HcclSim::Storage::DbResult<std::unique_ptr<HcclSim::Storage::IStorageEngine>> {
                // 按逻辑库路由到对应业务目录；引擎装配全部由模块部署从描述符派生。
                const auto* catalog = CatalogOf(config.database);
                if (catalog == nullptr) {
                    return {
                        HcclSim::Storage::DbCode::INVALID_ARGUMENT, std::nullopt, 0,
                        "database is not registered in the simulator catalog"};
                }
                if (!createEngine) {
                    return {
                        HcclSim::Storage::DbCode::BACKEND_UNAVAILABLE, std::nullopt, 0,
                        "backend deployment is not available"};
                }
                return createEngine({catalog, config});
            });
    }
    runtime.UseDatabase(HcclSim::Storage::Ddl::kRunnerDatabase);
    runtime.UseDatabase(HcclSim::Storage::Ddl::kOpDataDatabase);
    return {HcclSim::Storage::DbCode::OK, 0, {}};
}

HcclSim::Storage::DbResult<void> EnsureSimulatorStorageSession()
{
    // 应用级 ready-session bootstrap（同时登记进 ProcessStorageContext，供冷进程
    // 首次原子/业务 API 调用经 AcquireReady() single-flight 回调）。
    // 迁移前语义：进程会话已安装（含测试 peer/维护入口以显式路由安装）时，
    // 幂等成功，不强制改回默认路由；显式 close/remove 后允许本入口重建（维护
    // 路径），普通业务原子入口不会到达这里（REMOVED 阶段 AcquireReady 直接拒绝）。
    if (HcclSim::Storage::ProcessStorageContext::Acquire().ok()) {
        return {HcclSim::Storage::DbCode::OK, 0, {}};
    }
    // 部署配置（HCCL_VM_STORAGE_CONFIG）只在进程 bootstrap 解析一次：
    // 配置错误在此明确失败，禁止静默回退任何后端。
    const auto& deploy = DeployConfigSnapshot();
    if (!deploy.status.ok()) {
        return deploy.status;
    }
    if (deploy.selection.backend.empty()) {
        return {HcclSim::Storage::DbCode::INVALID_STATE, 0, "no storage backend is selected for the default session"};
    }
    const auto selected = deploy.selection.deployments.find(deploy.selection.backend);
    if (selected == deploy.selection.deployments.end() || !selected->second.makeRoute) {
        return {HcclSim::Storage::DbCode::BACKEND_UNAVAILABLE, 0, "selected backend deployment is not available"};
    }
    // 无参冷 bootstrap：两个默认逻辑库都按 OwnerPrepare 装配，路由端点由选中
    // 部署的 MakeRoute 给出（部署知识单点驻留各后端模块）。
    HcclSim::Storage::StorageRuntimeConfig config;
    const std::string installRoot = InstallPath::GetHcclVmInstallAbsPath();
    const HcclSim::Storage::Ddl::DatabaseDescriptor* const databases[]
        = {&HcclSim::Storage::Ddl::kRunnerDatabase, &HcclSim::Storage::Ddl::kOpDataDatabase};
    for (const HcclSim::Storage::Ddl::DatabaseDescriptor* database : databases) {
        auto route = selected->second.makeRoute({database, installRoot});
        if (!route.ok()) {
            return {route.code, 0, route.diagnostic};
        }
        config.databases.push_back(std::move(*route.value));
    }
    return EnsureSimulatorStorageSession(config);
}

HcclSim::Storage::DbResult<void> WipeSimulatorStorage()
{
    // 仅 host（hccl-vm start）的 envInit 调用：把外部持久化后端上一轮运行的
    // 数据整体清回全新态，提供等价删除 data/ 目录（含 db 文件）的干净起点。
    // 按当前逻辑路由分派到对应后端模块的 Wipe（host-only 维护边界；无需处理
    // 的后端模块返回成功 no-op）；不建立业务会话。
    const auto& deploy = DeployConfigSnapshot();
    if (!deploy.status.ok()) {
        return deploy.status;
    }
    const std::string dbIds[]
        = {std::string{HcclSim::Storage::Ddl::kRunnerDatabase.databaseId},
           std::string{HcclSim::Storage::Ddl::kOpDataDatabase.databaseId}};
    for (const auto& database : dbIds) {
        const auto* catalog = CatalogOf(database);
        if (catalog == nullptr) {
            continue;
        }
        const auto route = RouteOf(database);
        if (!route.has_value()) {
            return {HcclSim::Storage::DbCode::INVALID_ARGUMENT, 0, "database route endpoint is unknown"};
        }
        const auto* deployment = DeploymentOf(route->backend);
        if (deployment == nullptr || !deployment->wipe) {
            return {HcclSim::Storage::DbCode::BACKEND_UNAVAILABLE, 0, "backend deployment is not available for wipe"};
        }
        auto wiped = deployment->wipe(
            {catalog, HcclSim::Storage::LogicalDatabaseRuntimeConfig{database, route->backend, route->endpoint}},
            HcclSim::Storage::detail::kSessionDataNamespace);
        if (!wiped.ok()) {
            return wiped;
        }
    }
    return {HcclSim::Storage::DbCode::OK, 0, {}};
}

HcclSim::Storage::DbResult<void> EnsureSimulatorStorageSession(const HcclSim::Storage::StorageRuntimeConfig& config)
{
    auto registered = RegisterSimulatorStorageBackends();
    if (!registered.ok()) {
        return registered;
    }
    // 默认请求解析：backend 为空的路由在进入 Runtime 之前解析为实际后端模块名
    // （选中/默认部署）；空值不是已解析的合法后端。部署配置失败时无法解析，
    // 返回缓存的配置错误（全部显式路由不受影响）。
    HcclSim::Storage::StorageRuntimeConfig resolved = config;
    bool hasUnresolved = false;
    for (const auto& database : resolved.databases) {
        if (database.backend.empty()) {
            hasUnresolved = true;
        }
    }
    if (hasUnresolved) {
        const auto& deploy = DeployConfigSnapshot();
        if (!deploy.status.ok() || deploy.selection.backend.empty()) {
            return deploy.status.ok() ?
                       HcclSim::Storage::DbResult<void>{
                           HcclSim::Storage::DbCode::INVALID_STATE, 0,
                           "no storage backend is selected for the default request"} :
                       deploy.status;
        }
        for (auto& database : resolved.databases) {
            if (database.backend.empty()) {
                database.backend = deploy.selection.backend;
            }
        }
    }
    auto ensured = HcclSim::Storage::StorageRuntime::EnsureProcessSession(resolved);
    if (ensured.ok()) {
        // 记录当前逻辑路由（备份等维护观测由此取得端点；换库入口同步更新）。
        RecordRoutes(resolved);
    }
    return ensured;
}

HcclSim::Storage::DbResult<void>
ReconfigureSimulatorDatabase(const HcclSim::Storage::LogicalDatabaseRuntimeConfig& config)
{
    // 维护入口：空 backend 同样先解析为选中/默认后端模块名。
    HcclSim::Storage::LogicalDatabaseRuntimeConfig resolved = config;
    if (resolved.backend.empty()) {
        const auto& deploy = DeployConfigSnapshot();
        if (!deploy.status.ok() || deploy.selection.backend.empty()) {
            return deploy.status.ok() ?
                       HcclSim::Storage::DbResult<void>{
                           HcclSim::Storage::DbCode::INVALID_STATE, 0,
                           "no storage backend is selected for the default request"} :
                       deploy.status;
        }
        resolved.backend = deploy.selection.backend;
    }
    auto reconfigured = HcclSim::Storage::StorageRuntime::ReconfigureDatabase(resolved);
    if (!reconfigured.ok()) {
        return reconfigured;
    }
    std::lock_guard<std::mutex> lock(RouteMutex());
    CurrentRoutes()[resolved.database] = {resolved.backend, resolved.endpoint};
    return reconfigured;
}

HcclSim::Storage::DbResult<void> ReconfigureDatabaseToRecordedLocation(
    const HcclSim::Storage::LogicalDatabaseId& database, const std::string& recordedPath)
{
    // 部署形态差异由装配层按当前路由的后端模块裁定（模块 ResolveRecordedLocation
    // 回调）：文件部署后端重配到记录路径（换库/加载备份语义）；服务端部署后端
    // 保持部署路由（数据位置由部署配置唯一决定，与任务文件记录路径无关）。
    const auto route = RouteOf(database);
    if (!route.has_value()) {
        // 无当前路由（未知逻辑库/未安装且无默认部署）：走显式重配路径，保持
        // 运行时对未知逻辑库的原有错误。
        const auto& deploy = DeployConfigSnapshot();
        if (!deploy.status.ok() || deploy.selection.backend.empty()) {
            return deploy.status.ok() ?
                       HcclSim::Storage::DbResult<void>{
                           HcclSim::Storage::DbCode::INVALID_ARGUMENT, 0, "database route endpoint is unknown"} :
                       deploy.status;
        }
        return ReconfigureSimulatorDatabase({database, deploy.selection.backend, recordedPath});
    }
    const auto* deployment = DeploymentOf(route->backend);
    if (deployment == nullptr || !deployment->resolveRecordedLocation) {
        return {
            HcclSim::Storage::DbCode::BACKEND_UNAVAILABLE, 0,
            "backend deployment is not available for recorded location"};
    }
    auto resolved = deployment->resolveRecordedLocation(
        HcclSim::Storage::LogicalDatabaseRuntimeConfig{database, route->backend, route->endpoint}, recordedPath);
    if (!resolved.ok()) {
        return {resolved.code, 0, resolved.diagnostic};
    }
    if (!resolved.value.has_value()) {
        // 成功但缺少外层载荷：回调合同违规——成功裁定必须携带载荷，"不换库"
        // 由载荷内的空 optional 表达（见 backend_module.h）。明确报合同错误，
        // 不得当作保持路由，也不得对内层空 optional 继续解引用。
        return {
            HcclSim::Storage::DbCode::INVALID_STATE, 0,
            "backend resolveRecordedLocation succeeded without a relocation verdict"};
    }
    const std::optional<HcclSim::Storage::LogicalDatabaseRuntimeConfig>& relocation = *resolved.value;
    if (!relocation.has_value()) {
        return {HcclSim::Storage::DbCode::OK, 0, {}}; // 内层空：保持当前部署路由
    }
    return ReconfigureSimulatorDatabase(*relocation);
}

HcclSim::Storage::DbResult<HcclSim::Storage::BackupArtifact> BackupSimulatorDatabase(
    const HcclSim::Storage::LogicalDatabaseId& database, const HcclSim::Storage::BackupRequest& request)
{
    const auto* catalog = CatalogOf(database);
    if (catalog == nullptr) {
        return {
            HcclSim::Storage::DbCode::INVALID_ARGUMENT, std::nullopt, 0,
            "database is not registered in the simulator catalog"};
    }
    // 按当前逻辑路由的实际后端与端点分派（不能一律按进程默认后端分派）；
    // 默认部署路径可用于首次显式安装前的维护场景。
    const auto route = RouteOf(database);
    if (!route.has_value() || route->endpoint.empty()) {
        return {HcclSim::Storage::DbCode::INVALID_ARGUMENT, std::nullopt, 0, "database route endpoint is unknown"};
    }
    const auto* deployment = DeploymentOf(route->backend);
    if (deployment == nullptr || !deployment->backup) {
        return {
            HcclSim::Storage::DbCode::BACKEND_UNAVAILABLE, std::nullopt, 0,
            "backend deployment is not available for backup"};
    }
    return deployment->backup(
        {catalog, HcclSim::Storage::LogicalDatabaseRuntimeConfig{database, route->backend, route->endpoint}},
        HcclSim::Storage::detail::kSessionDataNamespace, request);
}
} // namespace sim
