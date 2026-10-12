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
 * @file process_storage_context.h
 * @brief 进程级存储上下文单例（ProcessStorageContext），管理会话安装、Lease 租约与 Fork 隔离。
 *
 * 【进程单例会话与租约生命周期模型 (Process Context & Lease Model)】
 *
 *                     ┌───────────────────────────────────────────────┐
 *                     │  ProcessStorageContext (当前进程唯一静态中枢) │
 *                     │  ┌─────────────────────────────────────────┐  │
 *                     │  │ 应用级 bootstrap 注册表（函数指针）    │  │
 *                     │  └─────────────────────────────────────────┘  │
 *                     └───────────────────────┬───────────────────────┘
 *                                             │ Install(session) 安装会话
 *                        AcquireReady() 冷启动 │ （bootstrap 幂等回调）
 *                                             ▼
 *                          StorageSession (受管业务会话)
 *                          ├── leaseCount (在途租约计数器)
 *                          └── quiesceGate (静止闸门拦截标志)
 *                                             │
 *                     ┌───────────────────────┴───────────────────────┐
 *                     │ Acquire()/AcquireReady() 成功借出              │
 *                     ▼                                               ▼
 *          StorageSessionLease A                           StorageSessionLease B
 *          (leaseCount++)                                  (leaseCount++)
 *          │                                               │
 *          └── 析构 / 归还 ────────────────────────────────┘ (leaseCount--)
 *                     │
 *                     ▼ 当 RemoveAndDrain() 被调用时
 *          等待 leaseCount 归零 ──> 从进程上下文卸载会话 ──> 允许安全 Close()
 *
 * 【ready-session 唯一入口模型】
 * Acquire() 保持严格语义：只读取已安装会话，未安装一律 INVALID_STATE（生命周期测试与
 * 维护代码用它观察真实状态）。业务数据库访问（table_access 原子入口）只经 AcquireReady()：
 * INSTALLED 直接取得租约；NEVER_INSTALLED 经已注册的应用级 bootstrap 一次建立会话
 * （single-flight：并发首次调用只允许一个线程执行 bootstrap，其余等待并共享结果）；
 * QUIESCING/QUIESCED/DRAINING/REMOVED 直接拒绝——显式 close/remove 后不自动复活。
 * bootstrap 注册表独立于会话状态存续：fork 子进程继承注册信息但不继承父进程会话与
 * 连接（子进程按下方 fork 合同 exec 后，新进程经正常初始化/惰性 bootstrap 建立会话）。
 *
 * 【Fork 多进程隔离约束（收窄为立即 exec 合同）】
 * 真实产品 fork 点（主控交互 Bash、device 子进程、插件进程）的子进程全部立即 exec：
 * fork child 在 exec 前不得访问 Storage API；exec 失败必须 _exit（不运行继承进程的
 * atexit/静态析构）；新进程 exec 后依靠正常进程初始化与惰性 bootstrap 建立自己的
 * StorageSession。ResetAfterForkInChild() 不承载"child 重置后继续使用数据库"的
 * 生产承诺：runtime/composition 的静态锁同样是父进程继承物，child 冷 bootstrap
 * 可能在被消失父进程线程持有的锁上永久死锁。
 */
#ifndef HCCL_VM_STORAGE_INTERNAL_PROCESS_STORAGE_CONTEXT_H
#define HCCL_VM_STORAGE_INTERNAL_PROCESS_STORAGE_CONTEXT_H

#include <chrono>

#include "storage/database_result.h"
#include "storage/logical_operation.h"
#include "storage/storage_binding.h"

namespace HcclSim::Storage {
class StorageSession;

/**
 * @brief 应用级会话 bootstrap 函数指针：按默认逻辑路由建立本进程会话（幂等）。
 * @note 由唯一 simulator composition 注册；只建立会话（最终回调 Install()），
 *       不得回调 AcquireReady()（自锁）。轻量函数指针形态：无堆分配、无线程、无连接。
 */
using ProcessSessionBootstrap = DbResult<void> (*)();

/**
 * @brief 进程会话短期使用租约（RAII 守卫对象）。
 * @note 租约持有期间保护底层会话不被 RemoveAndDrain 移除；析构时自动归还在途计数。
 */
class StorageSessionLease final {
public:
    StorageSessionLease(const StorageSessionLease&) = delete;
    StorageSessionLease& operator=(const StorageSessionLease&) = delete;
    StorageSessionLease(StorageSessionLease&&) noexcept;
    StorageSessionLease& operator=(StorageSessionLease&&) noexcept;
    ~StorageSessionLease();

    /// 获取租约保护的 StorageSession 引用（仅在当前 Lease 生命周期内有效）
    StorageSession& Get() const;

private:
    explicit StorageSessionLease(StorageSession* session) : session_(session) {}
    void Release();
    StorageSession* session_{nullptr};
    friend class ProcessStorageContext;
};

/**
 * @brief 进程全局存储上下文静态管理类。
 */
class ProcessStorageContext final {
public:
    ProcessStorageContext() = delete;

    /// 安装本进程唯一业务会话；重复安装或安装无效会话将返回错误
    static DbResult<void> Install(StorageSession& session);

    /// 获取进程级使用租约；在 quiesce、drain 或未安装阶段将直接拒绝并返回 INVALID_STATE
    static DbResult<StorageSessionLease> Acquire();

    /// 登记唯一应用级会话 bootstrap（composition 在模块首次数据库访问前完成登记）。
    /// 唯一注册合同：null 拒绝；注册表为空时原子安装；重复登记同一函数幂等成功；
    /// 登记另一个不同函数返回 INVALID_STATE 且原注册保持不变（不被覆盖）。
    /// 注册信息在 fork 子进程中仍然可用。
    static DbResult<void> RegisterBootstrap(ProcessSessionBootstrap bootstrap);

    /// 业务数据库访问的唯一 ready-session 入口：INSTALLED 直接取得租约；
    /// NEVER_INSTALLED 经已注册 bootstrap 一次建立会话（single-flight，并发首次调用
    /// 只执行一次 bootstrap，等待线程共享同一结果）；静止/排空/已移除阶段直接拒绝，
    /// 显式 close/remove 后不自动复活。
    static DbResult<StorageSessionLease> AcquireReady();

    /// 拦截新 Acquire 请求，等待现有在途租约与操作静止，返回 session+epoch 静止令牌
    static DbResult<QuiesceToken> ApplyQuiesceGate(std::chrono::milliseconds timeout = std::chrono::seconds(30));

    /// 释放与当前 gate 完全匹配的令牌，恢复业务正常 Acquire
    static DbResult<void> ReleaseQuiesceGate(const QuiesceToken& token);

    /// 等待在途 lease 归零排空，然后从进程全局上下文中安全移除会话
    static DbResult<void> RemoveAndDrain(std::chrono::milliseconds timeout = std::chrono::seconds(30));

    /// fork 子进程中调用：把进程上下文切换到独立状态（NEVER_INSTALLED，bootstrap
    /// 注册保留）。不触碰任何继承 Session/Coordinator/Engine/SQLite 连接或 mutex——
    /// 继承对象整体放弃，由子进程退出/exec 回收。不构成"child 可继续使用数据库"的
    /// 生产合同（见上方 fork 合同：child 必须 exec 后由新进程惰性 bootstrap）。
    static void ResetAfterForkInChild() noexcept;

private:
    friend class StorageSessionLease;
    static void ReleaseLease();
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_INTERNAL_PROCESS_STORAGE_CONTEXT_H
