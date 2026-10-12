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

// 文件职责：SQLite 后端模块的 composition 装配入口（模块注册/配置/五操作回调）。
// 部署知识单点驻留本模块：安装根下两个逻辑库的默认文件路径与逐库连接 profile。
// 本模块不包含业务头，也不被公共 composition 按名引用——公共层经
// storage_composition/backend_module.h 的中立合同与生成注册清单接入。
#ifndef HCCL_VM_STORAGE_BACKENDS_SQLITE_COMPOSITION_SQLITE_BACKEND_MODULE_H
#define HCCL_VM_STORAGE_BACKENDS_SQLITE_COMPOSITION_SQLITE_BACKEND_MODULE_H

#include "storage_composition/backend_module.h"

namespace sim::backends {
/// SQLite 后端模块注册名（= 模块目录名；部署配置 backend 取值/路由 backend 字段）
inline constexpr char kSqliteBackendName[] = "sqlite";

/// 模块描述单例（注册名 + 唯一无配置默认候选 + 纯配置函数）。
/// Configure 不打开任何数据库文件；注册/生成部署快照不触发真实数据库操作。
const BackendModule& SqliteBackendModule();
} // namespace sim::backends

#endif // HCCL_VM_STORAGE_BACKENDS_SQLITE_COMPOSITION_SQLITE_BACKEND_MODULE_H
