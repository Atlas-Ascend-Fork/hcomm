/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：StorageRuntimeImpl 的中立装配实现。
// 会话安装：按 StorageRuntimeConfig 逐库创建引擎（未配置的库不建引擎），
// 由登记目录派生表绑定与 requiredTables，经 SessionCoordinator 打开并安装到进程上下文。
// 会话归属：实例与已安装会话为 no-destroy 进程生命周期（见 StorageRuntime::GetImpl），
// 存活至进程退出由 OS 回收；显式 owner（主控/device-arm/测试/换库）经
// CloseProcessSession 提前关闭。不依赖 aclFinalize/跨 DSO 静态析构顺序。
// 维护操作（换库/备份/静态清理/动态销毁）在会话静止语义下委托会话层执行。
#include "storage/internal/storage_runtime_impl.h"

#include <chrono>
#include <set>
#include <utility>

#include "storage/engine/i_storage_engine.h"
#include "storage/internal/process_storage_context.h"
#include "storage/query_spec.h"
#include "storage/session_coordinator.h"
#include "storage/storage_catalog.h"
#include "storage/storage_session.h"

namespace HcclSim::Storage::detail {
namespace {} // namespace

struct StorageRuntimeImpl::RuntimeState {
    StorageRuntimeConfig config;
    std::unique_ptr<SessionCoordinator> coordinator;
};

StorageRuntimeImpl& StorageRuntimeImpl::Instance() { return StorageRuntime::GetImpl(); }

void StorageRuntimeImpl::RegisterEngineFactory(StorageBackend backend, EngineFactory factory)
{
    std::lock_guard<std::mutex> lock(mutex_);
    factories_[backend] = std::move(factory);
}

void StorageRuntimeImpl::UseDatabase(const Ddl::DatabaseDescriptor& database)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto* existing : databases_) {
        if (existing->databaseId == database.databaseId) {
            return; // 同名目录重复登记幂等
        }
    }
    databases_.push_back(&database);
}

SessionStorageBinding StorageRuntimeImpl::BuildBindingLocked(const StorageRuntimeConfig& config) const
{
    SessionStorageBinding binding;
    binding.sessionId = "storage-runtime";
    binding.dataNamespace = kSessionDataNamespace;
    // 只登记逻辑库路由（每库一条，附静态目录只读指针）；逐表绑定不进入会话——
    // 表定义唯一来源是静态目录（表 → 逻辑库归属由 Ddl::DatabaseOf<T> 编译期派生）。
    for (const auto& databaseConfig : config.databases) {
        const auto* descriptor = FindDatabaseLocked(databaseConfig.database);
        if (descriptor == nullptr) {
            continue;
        }
        binding.databases.push_back({databaseConfig.database, databaseConfig.endpoint, descriptor});
    }
    return binding;
}

const Ddl::DatabaseDescriptor* StorageRuntimeImpl::FindDatabaseLocked(const std::string& databaseId) const
{
    for (const auto* descriptor : databases_) {
        if (descriptor->databaseId == databaseId) {
            return descriptor;
        }
    }
    return nullptr;
}

DbResult<void> StorageRuntimeImpl::InstallSessionLocked(const StorageRuntimeConfig& config)
{
    // 逐库校验：配置的库必须已登记目录且后端工厂可用，未配置的库不创建引擎。
    for (const auto& databaseConfig : config.databases) {
        if (FindDatabaseLocked(databaseConfig.database) == nullptr) {
            return {DbCode::INVALID_ARGUMENT, 0, "database is not registered in the catalog"};
        }
        const auto factory = factories_.find(databaseConfig.backend);
        if (factory == factories_.end() || !factory->second) {
            return {DbCode::BACKEND_UNAVAILABLE, 0, "backend factory is not registered"};
        }
    }

    auto binding = BuildBindingLocked(config);
    if (binding.databases.empty()) {
        return {DbCode::INVALID_ARGUMENT, 0, "configuration does not cover any registered database"};
    }

    // REQ-CAP-2：会话安装只登记每库惰性构造工厂（捕获已解析的后端工厂与部署配置），
    // 不创建引擎/连接；引擎创建、Open、Prepare 纳入会话路由槽的每库 single-flight
    // （首次真实访问构造；失败轮丢弃实例、下次访问重新构造）。
    StorageOpenRequest request;
    request.schemaRole = config.schemaRole;
    for (const auto& databaseConfig : config.databases) {
        EngineRegistration registration;
        registration.databaseId = databaseConfig.database;
        registration.backend = databaseConfig.backend;
        registration.factory
            = [backendFactory = factories_.at(databaseConfig.backend), database = databaseConfig.database,
               backend = databaseConfig.backend, endpoint = databaseConfig.endpoint]() {
                  return backendFactory(LogicalDatabaseRuntimeConfig{database, backend, endpoint});
              };
        registration.schemaRole = databaseConfig.schemaRole; // 每库 owner/consumer 判定
        request.engines.push_back(std::move(registration));
    }

    auto opened = SessionCoordinator::Open(binding, request);
    if (!opened.ok()) {
        return {opened.code, opened.affectedRows, opened.diagnostic};
    }
    auto installed = opened.value->Install();
    if (!installed.ok()) {
        return {installed.code, installed.affectedRows, installed.diagnostic};
    }
    state_ = std::make_unique<RuntimeState>();
    state_->config = config;
    state_->coordinator.reset(new SessionCoordinator(std::move(*opened.value)));
    return {DbCode::OK, 0, {}};
}

DbResult<void> StorageRuntimeImpl::EnsureProcessSession(const StorageRuntimeConfig& config)
{
    std::lock_guard<std::mutex> lock(mutex_);
    // fork 边界：进程上下文无已安装会话（子进程 ResetAfterForkInChild 后）时，
    // state_ 是父进程继承物。子进程放弃继承的会话/引擎/协调器整体所有权（release
    // 泄漏，由子进程退出/exec 回收）——不调用 coordinator/Session/Engine 的任何
    // 成员或析构：它们的锁可能由消失的父进程线程持有（fork 后永久锁死）。
    if (state_ != nullptr && !HcclSim::Storage::ProcessStorageContext::Acquire().ok()) {
        (void)state_.release();
        state_ = nullptr;
    }
    // 幂等：同配置已安装直接成功；配置变化重建（关闭旧会话再装新会话）。
    // 角色比较按每库有效角色（perDatabaseRole.value_or(sessionRole)）——有效角色
    // 改变必须重装，不得错误返回幂等 OK。
    if (state_ != nullptr && !config.databases.empty()) {
        bool same = state_->config.databases.size() == config.databases.size()
                    && state_->config.schemaRole == config.schemaRole;
        for (std::size_t i = 0; same && i < config.databases.size(); ++i) {
            const auto& left = state_->config.databases[i];
            const auto& right = config.databases[i];
            same = left.database == right.database && left.backend == right.backend && left.endpoint == right.endpoint
                   && left.schemaRole.value_or(state_->config.schemaRole)
                          == right.schemaRole.value_or(config.schemaRole);
        }
        if (same) {
            return {DbCode::OK, 0, {}};
        }
    }
    if (state_ != nullptr) {
        (void)state_->coordinator->Close();
        state_.reset();
    }
    // 会话安装成功即归属本进程直至进程退出（no-destroy process-lifetime，见
    // StorageRuntime::GetImpl 注释）：不注册任何退出析构 guard——退出兜底 guard
    // 构造晚于此处安装，跨 DSO 静态析构按登记逆序会先于仍需访问数据库的 HCCL
    // 退出消费者执行，顺序不可靠；显式 owner（主控/device-arm/测试/换库）仍经
    // CloseProcessSession 提前关闭。
    return InstallSessionLocked(config);
}

DbResult<void> StorageRuntimeImpl::CloseProcessSession()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == nullptr) {
        return {DbCode::INVALID_STATE, 0, "process session is not installed"};
    }
    // 卸载与关闭由协调器统一编排（释放静止闸门 -> RemoveAndDrain 排空 -> 关闭引擎），
    // 避免重复 RemoveAndDrain 把会话置入已移除状态。
    auto closed = state_->coordinator->Close();
    state_.reset();
    return closed;
}

DbResult<void> StorageRuntimeImpl::ReconfigureDatabase(const LogicalDatabaseRuntimeConfig& config)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == nullptr) {
        return {DbCode::INVALID_STATE, 0, "process session is not installed"};
    }
    StorageRuntimeConfig next = state_->config;
    bool replaced = false;
    for (auto& databaseConfig : next.databases) {
        if (databaseConfig.database == config.database) {
            databaseConfig = config;
            replaced = true;
        }
    }
    if (!replaced) {
        return {DbCode::INVALID_ARGUMENT, 0, "database is not part of the installed session"};
    }
    (void)state_->coordinator->Close();
    state_.reset();
    auto installed = InstallSessionLocked(next);
    if (!installed.ok()) {
        return installed;
    }
    return {DbCode::OK, 0, {}};
}

DbResult<BackupArtifact>
StorageRuntimeImpl::BackupDatabase(const LogicalDatabaseId& database, const BackupRequest& request)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == nullptr) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "process session is not installed"};
    }
    (void)database;
    // 备份走会话静止流程（备份提供者由引擎装配注入；此处为后端无关骨架，具体后端接入时由装配层提供）
    return {DbCode::UNSUPPORTED_CAPABILITY, std::nullopt, 0, "backup provider is not installed"};
}

DbResult<void> StorageRuntimeImpl::ClearStaticData(const LogicalDatabaseId& database)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == nullptr) {
        return {DbCode::INVALID_STATE, 0, "process session is not installed"};
    }
    const auto* descriptor = FindDatabaseLocked(database);
    if (descriptor == nullptr) {
        return {DbCode::INVALID_ARGUMENT, 0, "database is not registered in the catalog"};
    }
    auto token = state_->coordinator->ApplyQuiesceGate();
    if (!token.ok()) {
        return {token.code, token.affectedRows, token.diagnostic};
    }
    ResetPlan plan;
    plan.planId = "runtime-clear-static";
    for (std::size_t i = 0; i < descriptor->tableCount; ++i) {
        const auto& table = descriptor->tables[i];
        if (table.scope == Ddl::ScopeKind::DynamicPid) {
            continue; // 静态清理只覆盖静态表
        }
        ResetTableStep step;
        step.tableId = std::string{table.tableId};
        step.mode = ClearMode::KEEP_SEQUENCE;
        plan.tableSteps.push_back(std::move(step));
    }
    auto executed = state_->coordinator->ExecuteReset(plan, *token.value);
    (void)state_->coordinator->ReleaseQuiesceGate(*token.value);
    if (!executed.ok()) {
        return {executed.code, executed.affectedRows, executed.diagnostic};
    }
    return {DbCode::OK, 0, {}};
}

DbResult<uint64_t> StorageRuntimeImpl::DropDynamicData(const LogicalDatabaseId& database)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == nullptr) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "process session is not installed"};
    }
    const auto* descriptor = FindDatabaseLocked(database);
    if (descriptor == nullptr) {
        return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "database is not registered in the catalog"};
    }
    auto token = state_->coordinator->ApplyQuiesceGate();
    if (!token.ok()) {
        return {token.code, std::nullopt, token.affectedRows, token.diagnostic};
    }
    ResetPlan plan;
    plan.planId = "runtime-drop-dynamic";
    uint64_t dropped = 0;
    for (std::size_t i = 0; i < descriptor->tableCount; ++i) {
        const auto& table = descriptor->tables[i];
        if (table.scope != Ddl::ScopeKind::DynamicPid) {
            continue;
        }
        ResetTableStep step;
        step.tableId = std::string{table.tableId};
        // 动态表绑定按进程 scope 发布（基名路由 Process(0)）：重置步骤必须携带
        // 进程 scope 才能解析到表绑定；DROP_DYNAMIC_SCOPE 由引擎枚举销毁全部物理表。
        step.scope = TableScope::Process(0);
        step.mode = ClearMode::DROP_DYNAMIC_SCOPE;
        plan.tableSteps.push_back(std::move(step));
        ++dropped;
    }
    auto executed = state_->coordinator->ExecuteReset(plan, *token.value);
    (void)state_->coordinator->ReleaseQuiesceGate(*token.value);
    if (!executed.ok()) {
        return {executed.code, std::nullopt, executed.affectedRows, executed.diagnostic};
    }
    return {DbCode::OK, dropped, 0, {}};
}

void StorageRuntimeImpl::ResetForTest()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != nullptr) {
        (void)state_->coordinator->Close();
        state_.reset();
    }
    factories_.clear();
    databases_.clear();
}
} // namespace HcclSim::Storage::detail

namespace HcclSim::Storage {
detail::StorageRuntimeImpl& StorageRuntime::GetImpl()
{
    // no-destroy 进程生命周期单例（沿用基线 SimRunnerSqliteDB::Instance() 的惰性
    // new 语义；以函数局部静态指针的线程安全首次初始化替代基线的显式互斥锁）：
    // 实例、已安装会话与引擎连接存活至进程退出、由 OS 统一回收，不在 aclFinalize
    // 或普通静态析构中销毁。原因：aclFinalize 与跨 DSO 静态析构都不是数据库最后
    // 使用边界——HCCL 动态库的退出清理（Socket/Context/Device）仍会访问数据库；
    // 普通可析构静态对象既会在逆序析构中销毁 installed StorageSession 触发 C1
    // invariant（terminate/SIGABRT），其析构顺序也晚于更早登记的退出消费者而不可
    // 依赖。指针静态量本身析构平凡（不注册退出回调），首次初始化线程安全；每进程
    // 固定 O(1) 资源（一个实例 + 一个指针），不随表数/查询数/操作次数增长，不增加
    // 任何 CRUD 成本。真正具有完整所有权边界的主控、device-arm、测试与换库流程
    // 仍可显式调用 CloseProcessSession 提前关闭。
    static detail::StorageRuntimeImpl* const instance = new detail::StorageRuntimeImpl();
    return *instance;
}

DbResult<void> StorageRuntime::EnsureProcessSession(const StorageRuntimeConfig& config)
{
    return GetImpl().EnsureProcessSession(config);
}

DbResult<void> StorageRuntime::CloseProcessSession() { return GetImpl().CloseProcessSession(); }

DbResult<void> StorageRuntime::ReconfigureDatabase(const LogicalDatabaseRuntimeConfig& config)
{
    return GetImpl().ReconfigureDatabase(config);
}

DbResult<BackupArtifact> StorageRuntime::BackupDatabase(const LogicalDatabaseId& database, const BackupRequest& request)
{
    return GetImpl().BackupDatabase(database, request);
}

DbResult<void> StorageRuntime::ClearStaticData(const LogicalDatabaseId& database)
{
    return GetImpl().ClearStaticData(database);
}

DbResult<uint64_t> StorageRuntime::DropDynamicData(const LogicalDatabaseId& database)
{
    return GetImpl().DropDynamicData(database);
}
} // namespace HcclSim::Storage
