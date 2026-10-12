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
 * @file session_control.cc
 * @brief 实现会话共享状态机，集中管理活跃计数、Quiesce 静止拦截与每库引擎路由。
 *
 * 【静态路由模型】静态目录是表定义唯一来源：会话只保存每逻辑库一个 DatabaseRoute
 * 槽（目录指针 + 引擎引用），表 → 逻辑库归属由 Ddl::DatabaseOf<T> 编译期派生，
 * 类型化取表路由为 O(已配置库数)，不复制库内每张表的绑定/映射字符串，也不逐表
 * 线性扫描；仅无类型维护路径（ExecuteReset/ListTables）经库目录按名解析。
 */
#include "storage/internal/session_control.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "storage/engine/i_storage_engine.h"
#include "storage/engine_registration.h"
#include "storage/storage_session.h"
#include "storage/transaction.h"

namespace HcclSim::Storage {
struct SessionControlBlock::Impl {
    std::string sessionId;
    std::string dataNamespace;
    /// 逻辑库路由槽：每配置库一项（表绑定不进入会话，静态目录是唯一表定义来源）
    std::vector<SessionDatabaseRoute> routes;
    StorageOpenMode mode{StorageOpenMode::READ_WRITE_CREATE};
    SchemaRole schemaRole{SchemaRole::OwnerPrepare};
    class IStorageBackupProvider* backupProvider{nullptr};
    mutable std::mutex mutex;
    std::condition_variable drained;
    /// 库引擎惰性打开 single-flight 唤醒：一轮 Open 结束后唤醒全部等待线程
    std::condition_variable routeOpenDone;
    /// 上一轮打开结果被全部等待者消费后，唤醒等待发起下一轮的新调用者
    std::condition_variable routeOpenConsumed;
    size_t activeTables{0};
    size_t activeTransactions{0};
    std::atomic<bool> installed{false};
    /// 静止标志：写者持锁翻转，guard 直达执行路径以原子读检查（无锁）
    std::atomic<bool> quiescing{false};
    bool closed{false};
    uint64_t quiesceGeneration{0};
    std::shared_ptr<std::atomic<bool>> quiesceActive;
};

namespace {
    DbResult<void> InvalidState(const char* diagnostic) { return {DbCode::INVALID_STATE, 0, diagnostic}; }

    /// 在库静态目录内查表（无类型维护冷路径；类型化路由不经过本查找）。
    const Ddl::CatalogTableMeta* FindTableInCatalog(const Ddl::DatabaseDescriptor* catalog, std::string_view tableId)
    {
        if (catalog == nullptr) {
            return nullptr;
        }
        for (std::size_t i = 0; i < catalog->tableCount; ++i) {
            if (catalog->tables[i].tableId == tableId) {
                return &catalog->tables[i];
            }
        }
        return nullptr;
    }

    /// 按逻辑库查找路由槽（O(已配置库数)，与表数量无关）。
    const SessionDatabaseRoute* FindRoute(const std::vector<SessionDatabaseRoute>& routes, std::string_view databaseId)
    {
        for (const auto& route : routes) {
            if (std::string_view{route.databaseId} == databaseId) {
                return &route;
            }
        }
        return nullptr;
    }
} // namespace

SessionControlBlock::SessionControlBlock(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SessionControlBlock::~SessionControlBlock() = default;

/**
 * @brief 创建会话控制块。
 * @note 校验 binding 身份并把每配置库装配为一条 DatabaseRoute 槽（目录指针 + 引擎引用）；
 *       引擎的 Open/Prepare 由会话装配层（SessionCoordinator/StorageRuntimeImpl）完成，
 *       本层不打开物理连接。
 */
DbResult<std::shared_ptr<SessionControlBlock>>
SessionControlBlock::Create(const SessionStorageBinding& binding, const StorageOpenRequest& request)
{
    if (binding.sessionId.empty() || binding.dataNamespace.empty()) {
        return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "session binding identity is incomplete"};
    }
    if (request.engines.empty()) {
        return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "session requires at least one registered engine"};
    }
    for (const auto& registration : request.engines) {
        if ((registration.engine == nullptr && !registration.factory) || registration.databaseId.empty()) {
            return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "engine registration is incomplete"};
        }
    }
    auto impl = std::make_unique<Impl>();
    impl->sessionId = binding.sessionId;
    impl->dataNamespace = binding.dataNamespace;
    impl->mode = request.mode;
    impl->schemaRole = request.schemaRole;
    impl->backupProvider = request.backupProvider;
    for (const auto& registration : request.engines) {
        if (FindRoute(impl->routes, registration.databaseId) != nullptr) {
            continue; // 同库重复注册幂等（首个引擎生效）
        }
        SessionDatabaseRoute route;
        route.databaseId = registration.databaseId;
        route.backend = registration.backend;
        route.engine = registration.engine;
        route.factory = registration.factory;         // 惰性构造配置（安装不创建引擎/连接）
        route.roleOverride = registration.schemaRole; // 每库角色覆盖（冷 bootstrap 按库判定）
        impl->routes.push_back(std::move(route));
    }
    // 目录指针按库装配（静态只读数据；无类型维护路径经此解析表归属）。
    for (const auto& database : binding.databases) {
        auto* route = const_cast<SessionDatabaseRoute*>(FindRoute(impl->routes, database.databaseId));
        if (route != nullptr) {
            route->catalog = database.catalog;
        }
    }
    auto control = std::shared_ptr<SessionControlBlock>(new SessionControlBlock(std::move(impl)));
    return {DbCode::OK, std::optional<std::shared_ptr<SessionControlBlock>>{std::move(control)}, 0, {}};
}

/**
 * @brief 解析逻辑库路由并借出表句柄所需的引擎。
 * @note 类型化路径：databaseId 由 Ddl::DatabaseOf<T> 编译期提供（O(已配置库数) 槽查找）；
 *       scopeKind 由表目录声明提供——静态表要求无进程 scope，动态 PID 表要求显式
 *       进程 scope（静态请求不命中动态表，保持既有隔离语义）；解析成功递增
 *       activeTables，静态 meta 与 scope 由 Table<T> 自行携带（会话不解析表身份）。
 */
DbResult<IStorageEngine*>
SessionControlBlock::AcquireTable(std::string_view databaseId, Ddl::ScopeKind scopeKind, const TableScope& scope)
{
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (impl_->closed || impl_->quiescing.load(std::memory_order_acquire)
        || impl_->mode == StorageOpenMode::BACKUP_ONLY) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "session does not accept table operations"};
    }
    if (scopeKind == Ddl::ScopeKind::DynamicPid) {
        if (!scope.processId.has_value()) {
            // 动态表必须显式携带进程 scope（目录合同：静态请求不命中动态表）。
            return {DbCode::NOT_FOUND, std::nullopt, 0, "table binding is not published"};
        }
    } else if (scope.processId.has_value()) {
        // 静态表不接受进程 scope（既有精确 scope 匹配语义）。
        return {DbCode::NOT_FOUND, std::nullopt, 0, "table binding is not published"};
    }
    auto* route = const_cast<SessionDatabaseRoute*>(FindRoute(impl_->routes, databaseId));
    if (route == nullptr) {
        return {DbCode::BACKEND_UNAVAILABLE, std::nullopt, 0, "table route has no registered engine"};
    }
    // 每库首次真实访问惰性打开引擎（single-flight；Runner 与 OpData 互不阻塞）。
    // 打开线程在 Opening 临界区内登记在途计数，quiesce/drain 不会提前放行。
    auto opened = EnsureRouteOpen(lock, *route, true);
    if (!opened.ok()) {
        return {opened.code, std::nullopt, opened.affectedRows, opened.diagnostic};
    }
    return {DbCode::OK, std::optional<IStorageEngine*>{route->engine}, 0, {}};
}

DbResult<void> SessionControlBlock::EnsureRouteOpen(
    std::unique_lock<std::mutex>& lock, SessionDatabaseRoute& route, bool keepInFlightCount, bool underQuiesceToken)
{
    // 在途登记收尾（RAII 语义的本地状态）：等待者从被会话接受（登记为等待者）起即
    // 计入排空模型，每条失败退出路径精确归还；成功路径按 keepInFlightCount 决定
    // 是否把登记计数保留为本次取表的活跃计数（由 ReleaseTable 归还）。
    bool inFlightHeld = false;
    auto releaseInFlight = [this, &inFlightHeld]() {
        if (inFlightHeld) {
            --impl_->activeTables;
            inFlightHeld = false;
            if (impl_->activeTables == 0 && impl_->activeTransactions == 0) {
                impl_->drained.notify_all();
            }
        }
    };
    for (;;) {
        // 预创建引擎形态的永久失败：Open 抛异常后对象状态不可知，本路由直接返回
        // 同一稳定错误（不再触碰该引擎对象，不得伪装成可重试）。
        if (route.permanentFailure.has_value()) {
            releaseInFlight();
            return *route.permanentFailure;
        }
        if (route.openState == SessionDatabaseRoute::OpenState::Opened) {
            if (keepInFlightCount) {
                if (!inFlightHeld) {
                    ++impl_->activeTables; // 已打开快路径：补齐本次取表的活跃计数
                    inFlightHeld = true;
                }
                // 等待者的登记计数即本次取表的活跃计数（不重复累加）。
            } else {
                releaseInFlight(); // 调用方自行管理在途保护（事务/受控重置路径）
            }
            return {DbCode::OK, 0, {}};
        }
        if (route.openState == SessionDatabaseRoute::OpenState::Opening) {
            // 登记为本轮等待者：从被会话接受起计入在途（quiesce/drain 覆盖等待者）；
            // 本轮结束后精确消费本轮结果（不被下一轮重试污染）。
            ++route.openWaiters;
            if (keepInFlightCount && !inFlightHeld) {
                ++impl_->activeTables;
                inFlightHeld = true;
            }
            const uint64_t waitedGeneration = route.openGeneration;
            impl_->routeOpenDone.wait(lock, [&route, waitedGeneration, this]() {
                return route.openGeneration != waitedGeneration || impl_->closed;
            });
            // 防御：关闭后的空结果（正常路径等待者已被排空计数覆盖，先于关闭复位消费）。
            if (!route.lastOpenResult.has_value()) {
                if (route.openWaiters > 0) {
                    --route.openWaiters;
                }
                releaseInFlight();
                return InvalidState("session does not accept table operations");
            }
            const DbResult<void> roundResult = *route.lastOpenResult;
            if (route.openWaiters > 0) {
                --route.openWaiters;
            }
            if (route.openWaiters == 0) {
                // 本轮结果已被全部等待者消费：允许下一轮重试安全发布新结果。
                route.lastOpenResult.reset();
                impl_->routeOpenConsumed.notify_all();
            }
            if (!roundResult.ok()) {
                releaseInFlight();
                return roundResult;
            }
            // 唤醒后重新检查闸门/关闭状态：本轮成功但会话已进入静止或关闭——
            // 等待者不得取得 guard 在排空之后执行（被拒绝并归还在途计数）。
            if (impl_->closed || impl_->quiescing.load(std::memory_order_acquire)) {
                releaseInFlight();
                return InvalidState("session does not accept table operations");
            }
            continue; // 本轮成功：回顶部按已打开快路径处理（等待者的登记计数即活跃计数）
        }
        // 等待上一轮结果被全部等待者消费：新一轮不得覆盖仍被旧等待者依赖的结果
        // （与 AcquireReady 的 bootstrap 轮次隔离同一模型）。重试等待者从被会话接受
        // （入口闸门检查，同一锁持有段）起计入排空模型——quiesce 不得在其未计数时
        // 取得 token；唤醒后重查 closed/quiescing（受控 maintenance 持有效 token 的
        // 打开路径不因 quiescing 被拒绝）。
        if (route.openWaiters != 0) {
            if (keepInFlightCount && !inFlightHeld) {
                ++impl_->activeTables;
                inFlightHeld = true;
            }
            impl_->routeOpenConsumed.wait(lock, [&route, this]() {
                return route.openWaiters == 0 || impl_->closed;
            });
            if (impl_->closed || (!underQuiesceToken && impl_->quiescing.load(std::memory_order_acquire))) {
                releaseInFlight();
                return InvalidState("session does not accept table operations");
            }
            continue;
        }
        // 打开者：在途计数先行登记（打开期间 quiesce/drain 等待本线程返回）。
        if (!inFlightHeld) {
            ++impl_->activeTables;
            inFlightHeld = true;
        }
        route.openState = SessionDatabaseRoute::OpenState::Opening;
        lock.unlock();
        // 引擎实例解析 + Open + Prepare 的统一执行段：工厂形态在本轮构造全新实例
        // （上轮失败实例已丢弃）；预创建形态由外部管理者持有所有权（本会话不
        // delete/close 该对象——失败轮不终态关闭，重试轮跳过已成功的 Open）。
        // 字符串构造、工厂构造与引擎调用的任何异常都转为可诊断失败（不传播、
        // 不毒化 single-flight），由下方发布路径统一收尾唤醒。
        DbResult<void> opened{DbCode::BACKEND_UNAVAILABLE, 0, "table route has no registered engine"};
        IStorageEngine* engine = route.engine;
        const bool precreated = engine != nullptr; // 预创建形态（外部所有权）
        bool threwDuringOpen = false;
        // 本库 schema 打开意图：每库覆盖优先（冷 bootstrap 按库判定 owner/consumer），
        // 未覆盖时继承会话级角色。
        const SchemaRole schemaRole = route.roleOverride.value_or(impl_->schemaRole);
        try {
            if (engine == nullptr && route.factory) {
                auto created = route.factory();
                if (!created.ok()) {
                    opened = {created.code, 0, created.diagnostic};
                } else if (!created.value.has_value() || *created.value == nullptr) {
                    // factory 返回 OK 但没有引擎实例：有限可诊断失败（不解引用空值），
                    // 由下方发布路径唤醒同轮等待者；下一轮允许重试。
                    opened = {DbCode::BACKEND_UNAVAILABLE, 0, "engine factory returned no engine instance"};
                } else {
                    route.ownedEngine = std::move(*created.value);
                    engine = route.ownedEngine.get();
                }
            }
            if (engine != nullptr) {
                EngineOpenRequest openRequest;
                openRequest.mode = impl_->mode;
                openRequest.dataNamespace = impl_->dataNamespace;
                // schema 打开意图映射引擎模式：owner 建库+DDL；consumer 读写已存在库
                // （打开期校验静态 schema，不执行 DDL）。备份会话模式保持不变。
                if (impl_->mode != StorageOpenMode::BACKUP_ONLY) {
                    openRequest.mode = schemaRole == SchemaRole::ReadWriteExisting ? StorageOpenMode::READ_EXISTING :
                                                                                     StorageOpenMode::READ_WRITE_CREATE;
                }
                if (!precreated || !route.precreatedOpened) {
                    threwDuringOpen = true;
                    opened = engine->Open(openRequest);
                    threwDuringOpen = false;
                    if (opened.ok() && precreated) {
                        // 预创建形态：连接已打开（后续 Prepare 失败不关闭连接）。
                        route.precreatedOpened = true;
                    }
                } else {
                    // 预创建重试轮：连接已打开、仅 Prepare 未完成——跳过第二次 Open。
                    opened = {DbCode::OK, 0, {}};
                }
                if (opened.ok()) {
                    opened = engine->Prepare();
                    if (!opened.ok() && !precreated) {
                        // 工厂形态 Prepare 失败：实例按终态语义关闭并在发布时丢弃——
                        // 终态 Close 后的同一对象不得伪装成可重试。
                        // （预创建形态保留连接，下一轮只重试 Prepare。）
                        (void)engine->Close();
                    }
                }
            }
        } catch (const std::exception& ex) {
            opened = {DbCode::BACKEND_UNAVAILABLE, 0, std::string{"engine open threw an exception: "} + ex.what()};
            if (precreated && threwDuringOpen) {
                // 预创建引擎 Open 抛异常且状态不可知：进入永久失败（后续调用返回
                // 同一稳定错误，不再触碰该对象）。
                route.permanentFailure = opened;
            }
        } catch (...) {
            opened = {DbCode::BACKEND_UNAVAILABLE, 0, "engine open threw an unknown exception"};
            if (precreated && threwDuringOpen) {
                route.permanentFailure = opened;
            }
        }
        lock.lock();
        if (opened.ok() && route.engine == nullptr && route.ownedEngine != nullptr) {
            route.engine = route.ownedEngine.get(); // 工厂形态解析成功：路由槽记录已解析引擎
        }
        if (!opened.ok() && route.ownedEngine != nullptr) {
            // 失败轮丢弃工厂构造的实例（已按终态语义关闭或从未成功打开连接）；
            // 路由槽回到未解析状态，下次访问经 factory 重新构造（真实可重试）。
            route.ownedEngine.reset();
            route.engine = nullptr;
        }
        route.openState
            = opened.ok() ? SessionDatabaseRoute::OpenState::Opened : SessionDatabaseRoute::OpenState::NotOpened;
        ++route.openGeneration;
        route.lastOpenResult = opened;
        impl_->routeOpenDone.notify_all();
        if (!opened.ok()) {
            // 打开失败：归还本线程在途计数（失败结果已由本轮等待者共享；新调用可重试）。
            releaseInFlight();
            return opened;
        }
        if (!keepInFlightCount) {
            // 打开成功但调用方自行管理在途保护（事务路径由 activeTransactions 承担）：
            // 归还打开期间的临时计数。
            releaseInFlight();
        }
        // keepInFlightCount：打开者的计数保留为本次取表的活跃计数（由 ReleaseTable 归还）。
        return {DbCode::OK, 0, {}};
    }
}

void SessionControlBlock::ReleaseTable() noexcept
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->activeTables > 0) {
        --impl_->activeTables;
    }
    if (impl_->activeTables == 0 && impl_->activeTransactions == 0) {
        impl_->drained.notify_all();
    }
}

/**
 * @brief 执行受控重置（清表）。
 * @note 必须持有与当前会话同一 generation 的 active quiesce token。按表名在本会话
 *       已配置库的静态目录内解析 meta 与路由槽（冷路径，允许目录内线性查找），
 *       目标库引擎按首次访问惰性打开（受控 maintenance 在合法 quiesce token
 *       上下文执行——不因 quiescing 被拒绝，也不得由普通 CRUD 复用该特权）；
 *       打开成功后才读取 route.engine，不得在惰性打开前用空 engine 拒绝。
 */
DbResult<void> SessionControlBlock::ExecuteReset(
    const std::string& tableId, const TableScope& scope, ClearMode mode, const QuiesceToken& token)
{
    const Ddl::CatalogTableMeta* meta = nullptr;
    SessionDatabaseRoute* targetRoute = nullptr;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->closed || !impl_->quiescing.load(std::memory_order_acquire) || !token.Valid()
            || token.generation_ != impl_->quiesceGeneration || token.SessionId() != impl_->sessionId) {
            return InvalidState("reset requires the active session quiesce token");
        }
        for (auto& route : impl_->routes) {
            if (route.catalog == nullptr) {
                continue;
            }
            meta = FindTableInCatalog(route.catalog, tableId);
            if (meta == nullptr) {
                continue;
            }
            targetRoute = &route;
            break;
        }
        if (targetRoute == nullptr || meta == nullptr) {
            return {DbCode::NOT_FOUND, 0, "table binding is not published"};
        }
    }
    IStorageEngine* engine = nullptr;
    {
        // 受控 maintenance 打开路径：持有本会话当前有效 quiesce token（上方已校验），
        // 在 quiescing 状态下惰性打开目标库引擎（普通 CRUD 不传入该上下文）。
        std::unique_lock<std::mutex> lock(impl_->mutex);
        auto opened = EnsureRouteOpen(lock, *targetRoute, false, true);
        if (!opened.ok()) {
            return opened;
        }
        engine = targetRoute->engine; // 打开成功后读取；打开前不得以空 engine 拒绝
        if (engine == nullptr) {
            return {DbCode::BACKEND_UNAVAILABLE, 0, "reset route has no registered engine"};
        }
    }
    return engine->Clear(*meta, scope, mode);
}

/**
 * @brief 列出指定逻辑库的用户表名。
 * @note 解析库路由槽并惰性打开引擎后调用 ListTables；冷路径（维护/兼容入口）。
 */
DbResult<std::vector<std::string>> SessionControlBlock::ListTables(std::string_view databaseId)
{
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (impl_->closed || impl_->quiescing.load(std::memory_order_acquire)
        || impl_->mode == StorageOpenMode::BACKUP_ONLY) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "session does not accept table operations"};
    }
    auto* route = const_cast<SessionDatabaseRoute*>(FindRoute(impl_->routes, databaseId));
    if (route == nullptr) {
        return {DbCode::BACKEND_UNAVAILABLE, std::nullopt, 0, "table route has no registered engine"};
    }
    auto opened = EnsureRouteOpen(lock, *route, false);
    if (!opened.ok()) {
        return {opened.code, std::nullopt, opened.affectedRows, opened.diagnostic};
    }
    IStorageEngine* engine = route->engine;
    if (engine == nullptr) {
        return {DbCode::BACKEND_UNAVAILABLE, std::nullopt, 0, "table route has no registered engine"};
    }
    return engine->ListTables();
}

/**
 * @brief 开启未绑定路由的通用事务。
 * @note 递增 activeTransactions 活跃计数；物理引擎在首个暂存写时由 Table<T> 提供
 *       并经 AcceptTransactionEngine 校验记录，业务不需要感知事务域或路由等
 *       存储内部设施。
 */
DbResult<Transaction> SessionControlBlock::BeginUnboundTransaction()
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || impl_->quiescing.load(std::memory_order_acquire)
        || impl_->mode == StorageOpenMode::BACKUP_ONLY) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "session does not accept transactions"};
    }
    ++impl_->activeTransactions;
    Transaction transaction(shared_from_this());
    return {DbCode::OK, std::optional<Transaction>{std::move(transaction)}, 0, {}};
}

/**
 * @brief 为通用事务的首个暂存写校验会话状态并接受传入引擎。
 * @note 与取表入口使用相同的会话状态检查；引擎已由 Table<T> 在取表时解析并打开，
 *       本入口只做静止/关闭/仅备份拦截，不做路由查找（后续 Stage 只比较指针）。
 */
DbResult<void> SessionControlBlock::AcceptTransactionEngine(IStorageEngine* engine)
{
    if (engine == nullptr) {
        return {DbCode::INVALID_ARGUMENT, 0, "transaction stage requires a resolved engine"};
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || impl_->quiescing.load(std::memory_order_acquire)
        || impl_->mode == StorageOpenMode::BACKUP_ONLY) {
        return InvalidState("session does not accept transactions");
    }
    return {DbCode::OK, 0, {}};
}

void SessionControlBlock::ReleaseTransaction() noexcept
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->activeTransactions > 0) {
        --impl_->activeTransactions;
    }
    if (impl_->activeTables == 0 && impl_->activeTransactions == 0) {
        impl_->drained.notify_all();
    }
}

/**
 * @brief 应用写入静止闸门。
 * @note 设置 quiescing = true，等待 activeTables/activeTransactions 归零，生成 QuiesceToken。
 */
DbResult<QuiesceToken> SessionControlBlock::ApplyQuiesceGate(std::chrono::milliseconds timeout)
{
    static std::atomic<uint64_t> nextGeneration{1};
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (impl_->closed || impl_->quiescing.load(std::memory_order_acquire)) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "session is closed or already quiesced"};
    }
    impl_->quiescing.store(true, std::memory_order_release);
    impl_->quiesceGeneration = nextGeneration.fetch_add(1, std::memory_order_relaxed);
    impl_->quiesceActive = std::make_shared<std::atomic<bool>>(true);
    if (!impl_->drained.wait_for(lock, timeout, [this]() {
            return impl_->activeTables == 0 && impl_->activeTransactions == 0;
        })) {
        impl_->quiescing.store(false, std::memory_order_release);
        impl_->quiesceGeneration = 0;
        impl_->quiesceActive.reset();
        return {DbCode::RESOURCE_EXHAUSTED, std::nullopt, 0, "timed out applying quiesce gate"};
    }
    QuiesceToken token(impl_->quiesceGeneration, impl_->dataNamespace, impl_->sessionId, impl_->quiesceActive);
    return {DbCode::OK, std::optional<QuiesceToken>{std::move(token)}, 0, {}};
}

/**
 * @brief 解除静止闸门。
 * @note 校验 session、namespace、generation 全部匹配后，将 token 标志置为 false 并恢复业务读写。
 */
DbResult<void> SessionControlBlock::ReleaseQuiesceGate(const QuiesceToken& token)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || !impl_->quiescing || !token.Valid() || token.generation_ != impl_->quiesceGeneration
        || token.DataNamespace() != impl_->dataNamespace || token.SessionId() != impl_->sessionId) {
        return InvalidState("quiesce token does not match the active gate");
    }
    impl_->quiescing.store(false, std::memory_order_release);
    impl_->quiesceGeneration = 0;
    token.active_->store(false, std::memory_order_release);
    impl_->quiesceActive.reset();
    return {DbCode::OK, 0, {}};
}

/**
 * @brief 执行显式快照备份。
 * @note 严格要求 mode == BACKUP_ONLY，由注册的 provider 执行底层持久化。
 */
DbResult<BackupArtifact> SessionControlBlock::Backup(const BackupRequest& request)
{
    IStorageBackupProvider* provider = nullptr;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->closed || impl_->mode != StorageOpenMode::BACKUP_ONLY) {
            return {DbCode::INVALID_STATE, std::nullopt, 0, "backup requires an open BACKUP_ONLY session"};
        }
        provider = impl_->backupProvider;
    }
    if (provider == nullptr) {
        return {DbCode::UNSUPPORTED_CAPABILITY, std::nullopt, 0, "backup provider is not registered"};
    }
    return provider->Backup(request);
}

/**
 * @brief 排空在途对象并关闭所有引擎。
 * @note 逆序关闭引擎（经库路由槽），保留首个失败错误码，同时使 active token 失效。
 */
DbResult<void> SessionControlBlock::Close(std::chrono::milliseconds timeout)
{
    {
        std::unique_lock<std::mutex> lock(impl_->mutex);
        if (impl_->closed) {
            return InvalidState("session is closed");
        }
        const bool appliedByClose = !impl_->quiescing.load(std::memory_order_acquire);
        impl_->quiescing.store(true, std::memory_order_release);
        if (!impl_->drained.wait_for(lock, timeout, [this]() {
                return impl_->activeTables == 0 && impl_->activeTransactions == 0;
            })) {
            if (appliedByClose) {
                impl_->quiescing.store(false, std::memory_order_release);
            }
            return {DbCode::RESOURCE_EXHAUSTED, 0, "timed out closing storage session"};
        }
    }
    std::vector<IStorageEngine*> engines;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        engines.reserve(impl_->routes.size());
        for (auto& route : impl_->routes) {
            // 只关闭已实际打开的引擎：未访问过的库从未建立连接，Close 幂等跳过。
            // 预创建形态"连接已打开但 Prepare 未完成"（precreatedOpened）同样关闭一次；
            // 永久失败（Open 抛异常、状态不可知）的引擎不触碰。
            if (route.engine != nullptr
                && (route.openState == SessionDatabaseRoute::OpenState::Opened || route.precreatedOpened)) {
                engines.push_back(route.engine);
            }
        }
    }
    DbResult<void> first{DbCode::OK, 0, {}};
    for (auto engine = engines.rbegin(); engine != engines.rend(); ++engine) {
        auto result = (*engine)->Close();
        if (!result.ok() && first.ok()) {
            first = std::move(result);
        }
    }
    if (first.ok()) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->closed = true;
        for (auto& route : impl_->routes) {
            route.openState = SessionDatabaseRoute::OpenState::NotOpened;
            route.lastOpenResult.reset();
            route.openWaiters = 0;
            route.precreatedOpened = false;
            // 防御性代差推进 + 唤醒：正常路径下等待者已被在途计数排空覆盖、先于关闭
            // 复位消费完本轮结果；残留等待者（理论防御路径）据此醒来并被拒绝。
            ++route.openGeneration;
        }
        impl_->routeOpenDone.notify_all();
        impl_->routeOpenConsumed.notify_all();
        if (impl_->quiesceActive != nullptr) {
            impl_->quiesceActive->store(false, std::memory_order_release);
            impl_->quiesceActive.reset();
        }
    }
    return first;
}

bool SessionControlBlock::TryInstall() noexcept
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || impl_->quiescing.load(std::memory_order_acquire)
        || impl_->mode == StorageOpenMode::BACKUP_ONLY) {
        return false;
    }
    bool expected = false;
    return impl_->installed.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
}

void SessionControlBlock::MarkRemoved() noexcept { impl_->installed.store(false, std::memory_order_release); }

bool SessionControlBlock::Installed() const noexcept { return impl_->installed.load(std::memory_order_acquire); }

bool SessionControlBlock::BackupOnly() const noexcept
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->mode == StorageOpenMode::BACKUP_ONLY;
}

SessionControlBlock::SyncStateForTest SessionControlBlock::SyncState() const noexcept
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    SyncStateForTest state;
    for (const auto& route : impl_->routes) {
        state.pendingOpenWaiters += route.openWaiters;
    }
    state.sessionQuiescing = impl_->quiescing.load(std::memory_order_acquire);
    return state;
}
} // namespace HcclSim::Storage
