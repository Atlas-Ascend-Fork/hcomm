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

// 文件职责：模拟器存储部署配置（HCCL_VM_STORAGE_CONFIG）的中立类型与解析器。
// 配置文件路径来自环境变量（主控/rank/plugin/device 子进程正常继承）；每个进程
// 只在 composition bootstrap 时解析一次，形成不可变 runtime config，之后任何
// CRUD 不得重读 YAML。公共解析器只负责文件读取、YAML 外层、storage.backend
// 的模块名匹配与各后端模块配置段的分发校验：后端专属字段规则全部驻留在
// src/storage_backends/<模块名>/ 的模块 Configure 内。
// 裁剪规则：backend 值与配置段键都必须命中当前已编入的模块注册名——被移除
// 后端的取值/残留配置段一律解析期明确报错，不静默忽略，也不回退。
#ifndef HCCL_VM_SIMULATION_STORAGE_DEPLOY_CONFIG_H
#define HCCL_VM_SIMULATION_STORAGE_DEPLOY_CONFIG_H

#include <map>
#include <string>
#include <utility>

#include "storage/database_result.h"
#include "storage_composition/backend_module.h"

namespace HcclSim::Storage {
/// 部署配置解析结果：选中的后端模块注册名 + 每个已编入模块的部署快照
/// （含未被选中/未配置段的模块——显式路由仍可选择已编入的其他后端）。
struct StorageDeploySelection {
    std::string backend; ///< 选中的模块注册名（storage.backend 值）
    std::map<std::string, sim::backends::BackendDeployment> deployments; ///< 键为模块注册名
};

/// 解析部署配置文件（文件不存在/YAML 解析失败/未知后端/未知配置段/模块校验
/// 失败 → INVALID_ARGUMENT）
DbResult<StorageDeploySelection> ParseStorageDeployConfigFile(const std::string& path);

/// 解析部署配置文本（与文件版同一套校验；供测试与等价注入）
DbResult<StorageDeploySelection> ParseStorageDeployConfigText(const std::string& text);
} // namespace HcclSim::Storage

#endif // HCCL_VM_SIMULATION_STORAGE_DEPLOY_CONFIG_H
