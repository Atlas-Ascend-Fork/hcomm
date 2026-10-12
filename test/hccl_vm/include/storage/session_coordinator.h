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
 * @file session_coordinator.h
 * @brief 会话生命周期协调器（SessionCoordinator），统一编排会话安装、写入静止与具名重置计划。
 *
 * 【全局会话编排与重置时序 (Session Coordination & ResetPlan)】
 *
 *   Host 主控 / 编排层            SessionCoordinator               底层 StorageSession
 *          │                              │                                 │
 *   1. 准备重置                           │                                 │
 *      ApplyQuiesceGate() ───────────────>│ 开启静止闸门，等待在途操作排空  │
 *          │                              │────────────────────────────────>│ ApplyQuiesceGate()
 *          │<── 返回 QuiesceToken ────────┤                                 │
 *          │                              │                                 │
 *   2. 按计划执行清表 (ResetPlan)         │                                 │
 *      ExecuteReset(plan, token) ────────>│ 校验 Token 纪元与会话合法性     │
 *          │                              │ 逐表执行 ClearForReset() ──────>│ 派发 Clear 操作到引擎
 *          │                              │ 收集非数据库外部资源清单        │
 *          │<── 返回 ResetResult ─────────┤ (返回 pendingExternalResources) │
 *          │                              │                                 │
 *   3. 恢复业务服务                       │                                 │
 *      ReleaseQuiesceGate(token) ────────>│ 令牌失效，恢复正常读写 ────────>│ ReleaseQuiesceGate()
 *
 * @note 跨表原子性说明：ResetPlan 不承诺跨表分布式强一致事务；按步骤顺序执行，
 *       若某一表清理失败则立即锁存错误停止后续 DML，并由外部资源 owner 清理外部资源。
 */
#ifndef HCCL_VM_STORAGE_SESSION_COORDINATOR_H
#define HCCL_VM_STORAGE_SESSION_COORDINATOR_H

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "storage/database_result.h"
#include "storage/query_spec.h"
#include "storage/storage_session.h"

namespace HcclSim::Storage {
/// 重置计划中的单表清理步骤
struct ResetTableStep {
    std::string tableId;                      ///< 待清理的逻辑表名
    TableScope scope;                         ///< 作用域（Static / Dynamic PID）
    ClearMode mode{ClearMode::KEEP_SEQUENCE}; ///< 清理模式（保留序列号 / 重置序列号）
};

/// 重置计划中的外部资源清理步骤（如共享内存、文件描述符等非数据库资源）
struct ResetExternalStep {
    std::string resourceId; ///< 外部资源标识符
};

/**
 * @brief 具名重置计划（ResetPlan）。
 * @note 步骤顺序严格代表实际执行顺序；包含数据库表与数据库外资源。
 */
struct ResetPlan {
    std::string planId;                           ///< 计划唯一标识（如 "RESET_COMMAND"）
    std::vector<ResetTableStep> tableSteps;       ///< 数据库表清理步骤清单
    std::vector<ResetExternalStep> externalSteps; ///< 待外部模块处理的资源清单

    /// 校验计划有效性（必须有 planId、至少一个步骤，且步骤身份非空）
    bool Valid() const noexcept;
};

/// 重置执行结果报告
struct ResetResult {
    uint64_t clearedTables{0};                         ///< 成功清理的数据库表数量
    std::vector<std::string> pendingExternalResources; ///< 待宿主清理的外部资源清单
};

/**
 * @brief 会话协调器，统筹会话生命周期、Quiesce 状态机与重置计划编排。
 */
class SessionCoordinator final {
public:
    SessionCoordinator(const SessionCoordinator&) = delete;
    SessionCoordinator& operator=(const SessionCoordinator&) = delete;
    SessionCoordinator(SessionCoordinator&&) noexcept;
    SessionCoordinator& operator=(SessionCoordinator&&) = delete;
    ~SessionCoordinator();

    /**
     * @brief 创建并打开协调器内部持有的 StorageSession。
     * @note 此时会话尚未安装到进程全局上下文。
     */
    static DbResult<SessionCoordinator> Open(const SessionStorageBinding& binding, const StorageOpenRequest& request);

    /// 将内部会话安装为当前进程唯一业务会话（ProcessStorageContext::Install）
    DbResult<void> Install();

    /// 开启写入静止闸门并等待在途操作归零，返回当前协调器专属 active epoch token
    DbResult<QuiesceToken> ApplyQuiesceGate(std::chrono::milliseconds timeout = std::chrono::seconds(30));

    /// 校验 token 后按顺序执行 ResetPlan 中的数据库清表步骤
    DbResult<ResetResult> ExecuteReset(const ResetPlan& plan, const QuiesceToken& token);

    /// 释放静止闸门，使当前 Token 失效
    DbResult<void> ReleaseQuiesceGate(const QuiesceToken& token);

    /// 释放 gate、从进程上下文移除会话并显式关闭所有底层引擎
    DbResult<void> Close(std::chrono::milliseconds timeout = std::chrono::seconds(30));

private:
    struct Impl;
    explicit SessionCoordinator(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_SESSION_COORDINATOR_H
