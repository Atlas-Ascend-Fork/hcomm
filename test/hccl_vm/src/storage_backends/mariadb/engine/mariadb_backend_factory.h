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

// 文件职责：MariaDB 后端通用引擎工厂（仿 sqlite_backend_factory.h）。
// 只依赖统一 StorageCatalog 的 DatabaseDescriptor 与强类型连接 profile
// （MariadbConnectionProfile：endpoint URI + 三类超时（秒）+ 线路协议压缩
// 开关）：从描述符直接创建一
// 个未打开的 MariaDbEngine（表/字段/索引定义由引擎按 Table 传入的
// CatalogTableMeta 消费，不派生第二份 mapping）。profile 由 composition 按
// 部署配置构造——本工厂与引擎都不识别业务库名。引擎实例由
// StorageRuntimeImpl 的工厂注册表创建并持有，业务装配层只把所属逻辑库的
// 描述符传进来。
#ifndef HCCL_VM_STORAGE_ENGINE_MARIADB_MARIADB_BACKEND_FACTORY_H
#define HCCL_VM_STORAGE_ENGINE_MARIADB_MARIADB_BACKEND_FACTORY_H

#include <memory>
#include <utility>

#include "storage/database_result.h"
#include "storage/engine/i_storage_engine.h"
#include "storage_backends/mariadb/engine/mariadb_connection_profile.h"
#include "storage_backends/mariadb/engine/mariadb_engine.h"
#include "storage/internal/engine_contract_types.h"
#include "storage/storage_catalog.h"
#include "storage/storage_runtime.h"

namespace HcclSim::Storage {

/// 从统一目录描述符创建一个未打开的 MariaDB 引擎（通用工厂）。
/// database 为该逻辑库的表目录（唯一事实源）；profile 携带 endpoint URI、
/// 三类超时（秒）与线路协议压缩开关（composition 按部署配置构造）。databaseConfig.endpoint 必须与
/// profile.uri 一致（v1 单实例 K=1，无 endpoint 清单选择；不一致 →
/// INVALID_ARGUMENT）。物理 database 名由
/// 引擎在 Open 期按 dataNamespace + databaseId 派生，工厂不参与。
/// 引擎的 Open/Prepare/Close 由会话层统一编排，工厂不持有实例。
inline DbResult<std::unique_ptr<IStorageEngine>> CreateMariadbEngine(
    const Ddl::DatabaseDescriptor& database, const LogicalDatabaseRuntimeConfig& databaseConfig,
    const MariadbConnectionProfile& profile)
{
    if (databaseConfig.endpoint != profile.uri) {
        return {
            DbCode::INVALID_ARGUMENT, std::nullopt, 0,
            "mariadb route endpoint does not match the deployed uri: " + databaseConfig.endpoint};
    }
    return {
        DbCode::OK,
        std::optional<std::unique_ptr<IStorageEngine>>{std::make_unique<MariaDbEngine>(&database, profile)},
        0,
        {}};
}
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_ENGINE_MARIADB_MARIADB_BACKEND_FACTORY_H
