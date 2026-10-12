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
 * @file backup_contract.h
 * @brief 定义后端无关的物理快照与持久化备份合同接口。
 *
 * 【写入静止与库级备份时序 (Quiesce & Backup Workflow)】
 *
 *   SessionCoordinator        SessionControlBlock              IStorageBackupProvider
 *          │                           │                                  │
 *   1. 准备备份                        │                                  │
 *      ApplyQuiesceGate() ────────────>│  quiescing = true (拦截后续写入) │
 *          │                           │  等待 activeTables == 0          │
 *          │                           │  等待 activeTransactions == 0    │
 *          │                           │  (Drain 排空在途操作)            │
 *          │<── 返回 QuiesceToken ─────┤                                  │
 *          │                           │                                  │
 *   2. 开启仅备份会话                  │                                  │
 *      OpenForBackup(..., token) ─────>│  校验 Token 有效性与 Generation  │
 *          │                           │                                  │
 *   3. 执行物理快照 (库/实例级)        │                                  │
 *      session.Backup(request) ───────>│  校验 mode == BACKUP_ONLY        │
 *          │                           │─────────────────────────────────>│ 执行 BGSAVE / .db 备份
 *          │                           │<── 返回 BackupArtifact ──────────│ 生成校验摘要 SHA-256
 *          │<── 返回 BackupArtifact ───│                                  │
 *          │                           │                                  │
 *   4. 恢复业务运行                    │                                  │
 *      ReleaseQuiesceGate(token) ─────>│  token.active = false (令牌失效) │
 *                                      │  quiescing = false (恢复读写)    │
 *
 * @note 备份粒度说明：本接口执行的是「实例/库级别（Database/Instance-Level）」的物理快照，
 *       SQLite 备份对应完整 .db 文件复制，MariaDB 备份对应一致性快照 dump。
 *       单表清理走 ClearForReset / ResetPlan，不经过此接口。
 */
#ifndef HCCL_VM_STORAGE_BACKUP_CONTRACT_H
#define HCCL_VM_STORAGE_BACKUP_CONTRACT_H

#include <cstdint>
#include <string>

#include "storage/database_result.h"

namespace HcclSim::Storage {
/// 备份意图区分（供底层引擎据此选择不同的收尾与持久化策略）
enum class BackupPurpose : uint8_t {
    Exit,       ///< 任务正常退出备份（收尾并生成最终可归档的权威快照）
    OnlineReset ///< 在线重置前快照（用于临时故障恢复或状态回退点）
};

/// 备份请求参数
struct BackupRequest {
    std::string destinationPath;                ///< 备份目标目录/文件路径
    BackupPurpose purpose{BackupPurpose::Exit}; ///< 备份意图
};

/// 备份生成的可校验产物元数据
struct BackupArtifact {
    std::string destinationPath; ///< 成功落盘的物理备份文件/目录路径
};

/**
 * @brief 存储后端备份提供者抽象接口。
 * @note 必须由具体存储引擎或外部备份驱动实现。
 */
class IStorageBackupProvider {
public:
    virtual ~IStorageBackupProvider() = default;

    /**
     * @brief 执行一次显式物理快照备份。
     * @param request 包含目标路径与备份意图的请求
     * @return 成功时返回包含产物路径的 BackupArtifact，失败时返回错误码与诊断信息
     */
    virtual DbResult<BackupArtifact> Backup(const BackupRequest& request) = 0;
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_BACKUP_CONTRACT_H
