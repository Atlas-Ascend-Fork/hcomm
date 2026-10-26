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

// 文件职责：MariaDB 后端模块的 composition 装配入口（模块注册/配置/五操作回调）。
// 本模块不提供无配置默认值：未被选中且无配置段时建立缺少 profile 的部署对象，
// 被显式用于创建引擎/路由时返回配置缺失错误，不阻止默认后端正常启动。
#ifndef HCCL_VM_STORAGE_BACKENDS_MARIADB_COMPOSITION_MARIADB_BACKEND_MODULE_H
#define HCCL_VM_STORAGE_BACKENDS_MARIADB_COMPOSITION_MARIADB_BACKEND_MODULE_H

#include "storage_composition/backend_module.h"

namespace sim::backends {
/// MariaDB 后端模块注册名（= 模块目录名；部署配置 backend 取值/路由 backend 字段）
inline constexpr char kMariadbBackendName[] = "mariadb";

/// 模块描述单例（注册名 + 非默认候选 + 纯配置函数）。
/// Configure 不连接服务端；注册/生成部署快照不触发真实数据库操作。
const BackendModule& MariadbBackendModule();
} // namespace sim::backends

#endif // HCCL_VM_STORAGE_BACKENDS_MARIADB_COMPOSITION_MARIADB_BACKEND_MODULE_H
