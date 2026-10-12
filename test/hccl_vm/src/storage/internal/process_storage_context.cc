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
 * @file process_storage_context.cc
 * @brief 进程唯一会话的安装、租约借还、Quiesce 闸门拦截、排空移除与 Fork 子进程重置实现。
 */
#include "storage/internal/process_storage_context.h"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include "storage/storage_session.h"

namespace HcclSim::Storage {
namespace {
    /// 进程级上下文状态机阶段
    enum class ContextPhase : uint8_t { NEVER_INSTALLED, INSTALLED, QUIESCING, QUIESCED, DRAINING, REMOVED };

    /// 进程全局状态结构
    struct ContextState {
        std::mutex mutex;
        std::condition_variable drained;
        /// single-flight 唤醒：bootstrap 一轮结束后（成功或失败）唤醒全部等待线程
        std::condition_variable bootstrapDone;
        /// 上一轮结果被全部等待者消费后，唤醒等待发起下一轮的新调用者
        std::condition_variable roundConsumed;
        StorageSession* session{nullptr};
        size_t activeLeases{0};
        ContextPhase phase{ContextPhase::NEVER_INSTALLED};
        /// 唯一执行 bootstrap 的线程标记；其余首次调用线程在 bootstrapDone 上等待
        bool bootstrapInProgress{false};
        /// 最近一轮 bootstrap 的结束结果（成功或失败）；在全部本轮等待者消费
        /// （roundWaiters 归零）之前不得被下一轮覆盖或清除
        std::optional<DbResult<void>> lastRoundResult;
        /// 已登记为最近一轮等待者且尚未消费本轮结果的线程数
        size_t roundWaiters{0};
        /// 已结束的 bootstrap 轮次计数：等待线程据代差识别自己等待的那一轮
        uint64_t bootstrapGeneration{0};
    };

    ContextState*& StatePointer()
    {
        // no-destroy 进程生命周期存储：ContextState 含 mutex/条件变量，进程退出阶段
        // 仍会被晚于 storage 侧任何析构 guard 执行的退出消费者（HCCL 动态库静态析构
        // 中的数据库访问）使用——普通可析构 function-local 静态对象会先于晚退出消费
        // 者析构，使用已析构的锁是未定义行为，且其析构顺序跨 DSO 不可依赖。惰性 new
        // 的指针静态量本身析构平凡（不注册退出回调）、首次初始化线程安全；每进程固定
        // O(1)（初态与 fork 子态各一个），由 OS 在进程退出时统一回收，不增加任何操作
        // 成本。显式 owner 仍可经 RemoveAndDrain/CloseProcessSession 提前排空。
        static ContextState* const initialState = new ContextState();
        static ContextState* state = initialState;
        return state;
    }

    ContextState& CurrentState() { return *StatePointer(); }

    ContextState& ForkChildState()
    {
        // fork 子进程独立状态同样 no-destroy（理由同 StatePointer；仅在
        // ResetAfterForkInChild 首次调用时按需构造，普通进程不分配）。
        static ContextState* const childState = new ContextState();
        return *childState;
    }

    /// 应用级 bootstrap 注册表：独立于 ContextState 存续——ResetAfterForkInChild 只重置
    /// 会话状态，不清除注册；fork 子进程因此继承注册信息（不继承父进程会话与连接）。
    std::atomic<ProcessSessionBootstrap>& BootstrapRegistry()
    {
        static std::atomic<ProcessSessionBootstrap> registered{nullptr};
        return registered;
    }
} // namespace

StorageSessionLease::StorageSessionLease(StorageSessionLease&& other) noexcept
    : session_(std::exchange(other.session_, nullptr))
{}

StorageSessionLease& StorageSessionLease::operator=(StorageSessionLease&& other) noexcept
{
    if (this != &other) {
        Release();
        session_ = std::exchange(other.session_, nullptr);
    }
    return *this;
}

StorageSessionLease::~StorageSessionLease() { Release(); }

StorageSession& StorageSessionLease::Get() const
{
    if (session_ == nullptr) {
        std::terminate();
    }
    return *session_;
}

void StorageSessionLease::Release()
{
    if (session_ != nullptr) {
        session_ = nullptr;
        ProcessStorageContext::ReleaseLease();
    }
}

/**
 * @brief 安装进程唯一业务会话。
 * @note 先将 StorageSession 标记为 installed，确保同一对象不会被重复安装或并发移动。
 */
DbResult<void> ProcessStorageContext::Install(StorageSession& session)
{
    auto& state = CurrentState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.phase == ContextPhase::INSTALLED || state.phase == ContextPhase::DRAINING) {
        return {DbCode::INVALID_STATE, 0, "a process storage session is already installed"};
    }
    if (!session.TryMarkInstalled()) {
        return {DbCode::INVALID_STATE, 0, "storage session is closed or already installed"};
    }
    state.session = &session;
    state.phase = ContextPhase::INSTALLED;
    return {DbCode::OK, 0, {}};
}

/**
 * @brief 获取进程会话租约。
 * @note 仅在 INSTALLED 阶段发放 lease；其他阶段（QUIESCING, DRAINING, REMOVED）统一拒绝。
 */
DbResult<StorageSessionLease> ProcessStorageContext::Acquire()
{
    auto& state = CurrentState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.phase != ContextPhase::INSTALLED || state.session == nullptr) {
        return {DbCode::INVALID_STATE, std::nullopt, 0, "process storage session is not installed"};
    }
    ++state.activeLeases;
    StorageSessionLease lease(state.session);
    return {DbCode::OK, std::optional<StorageSessionLease>{std::move(lease)}, 0, {}};
}

/**
 * @brief 登记唯一应用级会话 bootstrap。
 * @note composition 的 registration-only 静态注册器在模块载入时调用；轻量函数指针
 *       存储，无堆分配。唯一注册合同：null 按非法参数拒绝；注册表为空时以
 *       compare/exchange 原子安装（并发首个注册者只有一个成功）；已注册同一
 *       函数幂等成功；已注册不同函数明确拒绝（INVALID_STATE），原注册保持不变。
 */
DbResult<void> ProcessStorageContext::RegisterBootstrap(ProcessSessionBootstrap bootstrap)
{
    if (bootstrap == nullptr) {
        return {DbCode::INVALID_ARGUMENT, 0, "process session bootstrap is null"};
    }
    auto& registry = BootstrapRegistry();
    ProcessSessionBootstrap registered = registry.load(std::memory_order_acquire);
    for (;;) {
        if (registered == nullptr) {
            // 注册表为空：原子安装。CAS 失败侧把 registered 更新为当前值，
            // 重判同指针（幂等）或不同指针（拒绝）——并发注册不会静默互相覆盖。
            if (registry.compare_exchange_strong(
                    registered, bootstrap, std::memory_order_acq_rel, std::memory_order_acquire)) {
                return {DbCode::OK, 0, {}};
            }
            continue;
        }
        if (registered == bootstrap) {
            return {DbCode::OK, 0, {}};
        }
        return {DbCode::INVALID_STATE, 0, "a different process session bootstrap is already registered"};
    }
}

/**
 * @brief 业务数据库访问的唯一 ready-session 入口。
 * @note 与 Acquire() 的区别：NEVER_INSTALLED 阶段（真实冷进程或 fork 子进程重置后）
 *       经已注册的应用级 bootstrap 惰性建立会话，等价迁移前惰性 singleton 的首次
 *       建连时机；其余非 INSTALLED 阶段语义与 Acquire() 一致（直接拒绝）。
 *       并发首次访问 single-flight：仅一个线程执行 bootstrap，其余在 bootstrapDone
 *       上等待；同一轮次的全部等待者精确共享该轮结果（code/affectedRows/diagnostic），
 *       不被下一轮重试污染——下一轮引导者必须先等上一轮结果被全部等待者消费。
 *       bootstrap 抛异常被捕获并转为可诊断失败（不毒化 single-flight）；
 *       bootstrap 返回 OK 但未安装会话时有限失败（不无限循环重试）。
 *       bootstrap 最终回调 Install()（获取 state.mutex），因此不得持锁调用。
 */
DbResult<StorageSessionLease> ProcessStorageContext::AcquireReady()
{
    auto& state = CurrentState();
    std::unique_lock<std::mutex> lock(state.mutex);
    for (;;) {
        if (state.phase == ContextPhase::INSTALLED && state.session != nullptr) {
            ++state.activeLeases;
            StorageSessionLease lease(state.session);
            return {DbCode::OK, std::optional<StorageSessionLease>{std::move(lease)}, 0, {}};
        }
        if (state.phase != ContextPhase::NEVER_INSTALLED || state.session != nullptr) {
            // QUIESCING/QUIESCED/DRAINING/REMOVED：显式关闭/移除后不自动复活。
            return {DbCode::INVALID_STATE, std::nullopt, 0, "process storage session is not installed"};
        }
        if (state.bootstrapInProgress) {
            // 登记为本轮等待者：本轮结束后精确消费本轮结果。已登记的等待者
            // 保证 lastRoundResult 在自己消费前不被下一轮覆盖或清除。
            ++state.roundWaiters;
            const uint64_t waitedGeneration = state.bootstrapGeneration;
            state.bootstrapDone.wait(lock, [&state, waitedGeneration]() {
                return state.bootstrapGeneration != waitedGeneration;
            });
            const DbResult<void> roundResult = *state.lastRoundResult;
            --state.roundWaiters;
            if (state.roundWaiters == 0) {
                state.lastRoundResult.reset();
                state.roundConsumed.notify_all();
            }
            if (!roundResult.ok()) {
                return {roundResult.code, std::nullopt, roundResult.affectedRows, roundResult.diagnostic};
            }
            // 本轮成功：循环重新取租约（引导与重取锁之间被显式移除时按当前状态拒绝）。
            continue;
        }
        // 引导者路径：先等上一轮结果被全部等待者消费——不覆盖未消费的轮次结果。
        if (state.roundWaiters != 0) {
            state.roundConsumed.wait(lock, [&state]() {
                return state.roundWaiters == 0;
            });
            continue;
        }
        const ProcessSessionBootstrap bootstrap = BootstrapRegistry().load(std::memory_order_acquire);
        if (bootstrap == nullptr) {
            return {DbCode::INVALID_STATE, std::nullopt, 0, "process storage session is not installed"};
        }
        state.bootstrapInProgress = true;
        lock.unlock();
        // bootstrap 内部执行 string/vector/engine 构造，标准分配可能抛异常：
        // 任何异常都被捕获并转为可诊断失败，single-flight 状态始终收尾（不毒化）。
        DbResult<void> bootstrapped{DbCode::BACKEND_UNAVAILABLE, 0, "process session bootstrap failed"};
        try {
            bootstrapped = bootstrap();
        } catch (const std::exception& ex) {
            bootstrapped
                = {DbCode::BACKEND_UNAVAILABLE, 0,
                   std::string{"process session bootstrap threw an exception: "} + ex.what()};
        } catch (...) {
            bootstrapped = {DbCode::BACKEND_UNAVAILABLE, 0, "process session bootstrap threw an unknown exception"};
        }
        lock.lock();
        state.bootstrapInProgress = false;
        ++state.bootstrapGeneration;
        // 后置条件：bootstrap 返回 OK 但会话仍为 NEVER_INSTALLED（未安装）时有限失败，
        // 不得无限循环重复执行 bootstrap；该失败与普通 bootstrap 失败同规则（本轮
        // 等待者共享、后续新调用允许重试）。
        if (bootstrapped.ok() && state.phase == ContextPhase::NEVER_INSTALLED) {
            bootstrapped
                = {DbCode::INVALID_STATE, 0, "process session bootstrap succeeded without installing a session"};
        }
        state.lastRoundResult = bootstrapped;
        state.bootstrapDone.notify_all();
        if (!bootstrapped.ok()) {
            return {bootstrapped.code, std::nullopt, bootstrapped.affectedRows, bootstrapped.diagnostic};
        }
        // 成功：循环重新评估（bootstrap 与重取锁之间被显式移除时按当前状态拒绝）。
    }
}

/**
 * @brief 开启静止闸门。
 * @note 先将进程状态切到 QUIESCING 阻止新 Acquire，再由 StorageSession 排空在途 Table/Transaction。
 */
DbResult<QuiesceToken> ProcessStorageContext::ApplyQuiesceGate(std::chrono::milliseconds timeout)
{
    auto& state = CurrentState();
    StorageSession* session = nullptr;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.phase != ContextPhase::INSTALLED || state.session == nullptr) {
            return {DbCode::INVALID_STATE, std::nullopt, 0, "process storage session is not installed"};
        }
        state.phase = ContextPhase::QUIESCING;
        session = state.session;
    }
    auto token = session->ApplyQuiesceGate(timeout);
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.phase = token.ok() ? ContextPhase::QUIESCED : ContextPhase::INSTALLED;
    }
    return token;
}

/**
 * @brief 释放静止闸门。
 * @note 解除 QUIESCED 状态，恢复为 INSTALLED，重新允许业务 Acquire。
 */
DbResult<void> ProcessStorageContext::ReleaseQuiesceGate(const QuiesceToken& token)
{
    auto& state = CurrentState();
    StorageSession* session = nullptr;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.phase != ContextPhase::QUIESCED || state.session == nullptr) {
            return {DbCode::INVALID_STATE, 0, "process storage session is not quiesced"};
        }
        session = state.session;
    }
    auto result = session->ReleaseQuiesceGate(token);
    if (result.ok()) {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.phase = ContextPhase::INSTALLED;
    }
    return result;
}

/**
 * @brief 移除并排空进程会话。
 * @note DRAINING 阶段等待所有 lease 归还；超时则恢复 INSTALLED 避免留下中间状态。
 */
DbResult<void> ProcessStorageContext::RemoveAndDrain(std::chrono::milliseconds timeout)
{
    auto& state = CurrentState();
    std::unique_lock<std::mutex> lock(state.mutex);
    if (state.phase != ContextPhase::INSTALLED || state.session == nullptr) {
        return {DbCode::INVALID_STATE, 0, "process storage session is not installed"};
    }
    state.phase = ContextPhase::DRAINING;
    if (!state.drained.wait_for(lock, timeout, [&state]() {
            return state.activeLeases == 0;
        })) {
        state.phase = ContextPhase::INSTALLED;
        return {DbCode::RESOURCE_EXHAUSTED, 0, "timed out draining process storage leases"};
    }
    state.session->MarkRemoved();
    state.session = nullptr;
    state.phase = ContextPhase::REMOVED;
    return {DbCode::OK, 0, {}};
}

/**
 * @brief Fork 子进程重置（收窄合同：仅切换进程本地上下文状态）。
 * @note 真实产品 fork 合同：所有 fork child 立即 exec，exec 前不得访问 Storage API
 *       （主控 Bash、device 进程与插件进程均如此）；exec 失败必须 _exit，不得运行
 *       可能触发继承对象析构的普通 exit；新进程 exec 后依靠正常初始化与惰性
 *       bootstrap 建立自己的会话。本函数不承载"child 重置后继续使用数据库"的
 *       生产承诺——runtime/composition 的静态锁同样是父进程继承物，child 冷
 *       bootstrap 可能死锁。它只把进程上下文指针切换到独立静态状态（回
 *       NEVER_INSTALLED，bootstrap 注册保留）：fork child 不触碰任何继承
 *       Session/Coordinator/Engine/SQLite 连接或 mutex，继承对象整体放弃，
 *       由子进程退出/exec 回收（POSIX 文件锁不随 fork 继承，无锁泄漏风险）。
 *       冷进程测试以 fork + 本函数取得真实 NEVER_INSTALLED 载体，属测试车辆
 *       而非产品 fork 合同。
 */
void ProcessStorageContext::ResetAfterForkInChild() noexcept
{
    // 不读取、不触碰继承状态（含继承 Session 的 MarkRemoved——那是对继承对象的
    // 访问，且不能证明任何继承 lease/引擎不再可用）；只切换到子进程独立状态。
    auto& state = ForkChildState();
    state.session = nullptr;
    state.activeLeases = 0;
    state.phase = ContextPhase::NEVER_INSTALLED;
    // bootstrap 注册信息保留（子进程据此重建自己的会话）；单飞状态清零——
    // 子进程不存在父进程在途 bootstrap 线程，残留 inProgress 会永久阻塞首次访问；
    // 未消费的轮次结果与等待者计数一并清零（子进程无父进程等待者）。
    state.bootstrapInProgress = false;
    state.lastRoundResult.reset();
    state.roundWaiters = 0;
    ++state.bootstrapGeneration;
    StatePointer() = &state;
}

void ProcessStorageContext::ReleaseLease()
{
    auto& state = CurrentState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.activeLeases > 0) {
        --state.activeLeases;
    }
    if (state.phase == ContextPhase::DRAINING && state.activeLeases == 0) {
        state.drained.notify_all();
    }
}
} // namespace HcclSim::Storage
