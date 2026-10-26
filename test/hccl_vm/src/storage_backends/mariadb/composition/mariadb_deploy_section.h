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

// 文件职责：MariaDB 模块部署配置段（storage.mariadb）的解析与校验。
// 校验合同（URI 形态/超时范围/秒字段/旧毫秒字段拒绝/压缩开关）与引擎 Open 期的
// ParseMariadbUri 保持同一规则，非法值在解析期报错；全部规则自迁移前公共
// storage_deploy_config 原样迁入，模块测试经本入口直接复用同一套断言。
#ifndef HCCL_VM_STORAGE_BACKENDS_MARIADB_COMPOSITION_MARIADB_DEPLOY_SECTION_H
#define HCCL_VM_STORAGE_BACKENDS_MARIADB_COMPOSITION_MARIADB_DEPLOY_SECTION_H

#include <yaml-cpp/yaml.h>

#include "storage/database_result.h"
#include "storage_backends/mariadb/engine/mariadb_connection_profile.h"

namespace sim::backends {
/// 解析 storage.mariadb 配置段（非 map/缺 uri/非法 URI/旧超时字段/越界值 →
/// INVALID_ARGUMENT；错误诊断与迁移前逐字一致）。
HcclSim::Storage::DbResult<HcclSim::Storage::MariadbConnectionProfile>
ParseMariadbDeploySection(const YAML::Node& section);
} // namespace sim::backends

#endif // HCCL_VM_STORAGE_BACKENDS_MARIADB_COMPOSITION_MARIADB_DEPLOY_SECTION_H
