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
 * @file i_relational_engine.h
 * @brief 关系型存储引擎专用扩展接口（IRelationalEngine），继承自 IStorageEngine。
 * @note 关系型数据库引擎继承本接口以支持库文件的专属备份方法。
 */
#ifndef HCCL_VM_STORAGE_ENGINE_I_RELATIONAL_ENGINE_H
#define HCCL_VM_STORAGE_ENGINE_I_RELATIONAL_ENGINE_H

#include "storage/engine/i_storage_engine.h"

namespace HcclSim::Storage {
/**
 * @brief 关系型存储引擎接口。
 */
class IRelationalEngine : public IStorageEngine {
public:
    /// 生成关系型数据库物理文件（如 .db 文件）的备份快照
    virtual DbResult<DatabaseBackupArtifact> BackupRelational(const EngineBackupRequest&) = 0;
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_ENGINE_I_RELATIONAL_ENGINE_H
