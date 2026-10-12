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

// 文件职责：SQLite 后端通用引擎工厂。
// 只依赖统一 StorageCatalog 的 DatabaseDescriptor、部署路径与强类型连接
// profile：从描述符直接创建一个未打开的 SQLiteEngine（表/字段/索引定义由
// 引擎按 Table 传入的 CatalogTableMeta 消费，不派生第二份 mapping）。
// profile 由 composition 按逻辑库配置构造——本工厂与引擎都不识别业务库名。
// 本工厂不包含业务头，也不提供 engine/manager getter；引擎实例由
// StorageRuntimeImpl 的工厂注册表创建并持有，业务装配层只把所属逻辑库的
// 描述符传进来。
#ifndef HCCL_VM_STORAGE_ENGINE_SQLITE_SQLITE_BACKEND_FACTORY_H
#define HCCL_VM_STORAGE_ENGINE_SQLITE_SQLITE_BACKEND_FACTORY_H

#include <memory>
#include <utility>

#include "storage/database_result.h"
#include "storage/engine/i_storage_engine.h"
#include "storage_backends/sqlite/engine/sqlite_engine.h"

namespace HcclSim::Storage {

/// 从统一目录描述符创建一个未打开的 SQLite 引擎（通用工厂）。
/// database 为该逻辑库的表目录（唯一事实源）；path 为 SQLite 文件路径；
/// profile 为 composition 按库生成的强类型连接参数。
/// 引擎的 Open/Prepare/Close 由会话层统一编排，工厂不持有实例。
inline DbResult<std::unique_ptr<IStorageEngine>> CreateSqliteEngine(
    const Ddl::DatabaseDescriptor& database, const std::string& path, const SqliteConnectionProfile& profile)
{
    return {
        DbCode::OK,
        std::optional<std::unique_ptr<IStorageEngine>>{std::make_unique<SQLiteEngine>(path, &database, profile)},
        0,
        {}};
}
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_ENGINE_SQLITE_SQLITE_BACKEND_FACTORY_H
