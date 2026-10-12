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
 * @file engine_contract_types.h
 * @brief 定义 IStorageEngine 虚函数表中传递的内部交互 DTO 数据结构。
 */
#ifndef HCCL_VM_STORAGE_INTERNAL_ENGINE_CONTRACT_TYPES_H
#define HCCL_VM_STORAGE_INTERNAL_ENGINE_CONTRACT_TYPES_H

#include <string>
#include <vector>

#include "storage/logical_operation.h"
#include "storage/query_spec.h"
#include "storage/storage_binding.h"
#include "storage/storage_catalog.h"

namespace HcclSim::Storage {
class IStorageEngine;

namespace Ddl {
    struct CatalogTableMeta;
} // namespace Ddl

/// 引擎打开请求参数包
struct EngineOpenRequest {
    StorageOpenMode mode{StorageOpenMode::READ_WRITE_CREATE};
    std::string dataNamespace;
};

/// 关系型数据库备份请求
struct EngineBackupRequest {
    std::string destinationPath;
};

/// 事务绑定身份：引擎以对象地址识别活动事务；不携带任何业务 DML 数据
/// （事务 DML 直达引擎立即执行，事务层与引擎都不保存载荷）。
struct TransactionBinding {};

/// 关系型数据库备份生成产物
struct DatabaseBackupArtifact {
    std::string destinationPath;
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_INTERNAL_ENGINE_CONTRACT_TYPES_H
