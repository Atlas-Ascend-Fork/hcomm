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

// 文件职责：存储后端模块的私有装配合同（BackendModule / BackendDeployment）。
// 这是 composition 层内部合同，不进入业务公开 API：业务只感知
// storage_composition.h 的后端无关入口；公共层经本合同发现并分派已编入的
// 后端模块，不出现具体引擎/工厂/profile 类型，也不按后端名做条件分支。
// 后端专属实现（引擎、工厂、profile、部署解析）全部驻留在
// src/storage_backends/<模块名>/ 目录，随目录整体保留或裁剪。
//
// 【固定对象模型】只有两种对象，不另建注册/生命周期系统：
//   BackendModule      静态描述：注册名、是否为无配置默认候选、纯配置函数
//                       Configure（不打开文件、不连接服务端）
//   BackendDeployment  Configure 产出的只读快照：五个维护/装配操作的回调；
//                       后端 profile 由回调按值捕获（或 shared_ptr<const> 持有），
//                       不经 void*/std::any/跨模块 dynamic_cast 传递，
//                       公共头也不定义任何后端 profile 类型
#ifndef HCCL_VM_SIMULATION_STORAGE_COMPOSITION_BACKEND_MODULE_H
#define HCCL_VM_SIMULATION_STORAGE_COMPOSITION_BACKEND_MODULE_H

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <yaml-cpp/yaml.h>

#include "storage/backup_contract.h"
#include "storage/database_result.h"
#include "storage/storage_runtime.h"

namespace HcclSim::Storage {
class IStorageEngine;
struct BackupRequest;
struct BackupArtifact;

namespace Ddl {
    struct DatabaseDescriptor;
} // namespace Ddl
} // namespace HcclSim::Storage

namespace sim::backends {

/// Configure 输入（只读）：该模块在部署配置 storage: 段下的 YAML 子节点、
/// 是否为选中后端、配置段是否显式存在。
struct BackendConfigureRequest {
    YAML::Node section;         ///< 模块同名子节点（sectionPresent=false 时为未定义节点）
    bool selected{false};       ///< storage.backend 是否选中本模块
    bool sectionPresent{false}; ///< 模块同名配置段是否显式存在
};

/// MakeRoute 输入（只读）：逻辑库目录描述符与安装根路径。
struct BackendRouteRequest {
    const HcclSim::Storage::Ddl::DatabaseDescriptor* descriptor{nullptr};
    std::string installRoot; ///< hccl-vm 安装根（绝对路径）
};

/// CreateEngine 输入（只读）：逻辑库目录描述符与已解析的显式路由。
struct BackendEngineRequest {
    const HcclSim::Storage::Ddl::DatabaseDescriptor* descriptor{nullptr};
    HcclSim::Storage::LogicalDatabaseRuntimeConfig route;
};

/// 配置后的只读部署快照：五个操作回调。未配置（缺少必需 profile）的部署
/// 也建立全部回调，被显式调用时返回配置缺失错误，而不是空回调。
struct BackendDeployment {
    /// 逻辑库默认路由：安装根下的物理端点（模块自有路径/URI 规则）。
    /// 返回完整 LogicalDatabaseRuntimeConfig（含 schema 打开意图）。
    std::function<HcclSim::Storage::DbResult<HcclSim::Storage::LogicalDatabaseRuntimeConfig>(
        const BackendRouteRequest&)>
        makeRoute;

    /// 按显式路由构造未打开的引擎（复用模块工厂；不提前 Open/Prepare）。
    std::function<HcclSim::Storage::DbResult<std::unique_ptr<HcclSim::Storage::IStorageEngine>>(
        const BackendEngineRequest&)>
        createEngine;

    /// 清库（host-only 维护边界）：按路由创建 transient 引擎清回全新态；
    /// 无需处理的后端返回成功（no-op）。
    std::function<HcclSim::Storage::DbResult<void>(const BackendEngineRequest&, const std::string& dataNamespace)> wipe;

    /// 备份：专用备份连接整体流程（独立引擎、BACKUP_ONLY、Open/Prepare/
    /// Backup/Close、错误传播），不占用业务会话。
    std::function<HcclSim::Storage::DbResult<HcclSim::Storage::BackupArtifact>(
        const BackendEngineRequest&, const std::string& dataNamespace, const HcclSim::Storage::BackupRequest& request)>
        backup;

    /// 历史位置恢复裁定：错误直接传播；成功时外层载荷必须存在（缺失即回调
    /// 合同违规，公共层按 INVALID_STATE 拒绝）。载荷内的 optional 有值表示
    /// 公共层执行现有重配（换库语义），为空表示保持当前路由（部署位置由
    /// 部署配置唯一决定）。
    std::function<HcclSim::Storage::DbResult<std::optional<HcclSim::Storage::LogicalDatabaseRuntimeConfig>>(
        const HcclSim::Storage::LogicalDatabaseRuntimeConfig& currentRoute, const std::string& recordedPath)>
        resolveRecordedLocation;
};

/// 后端模块静态描述（各模块 composition 提供单例；注册名与模块目录名一致）。
struct BackendModule {
    std::string name;            ///< 模块注册名（= src/storage_backends/<目录名>）
    bool providesDefault{false}; ///< 是否为无配置默认候选（默认候选须唯一）
    /// 纯配置函数：校验模块配置段并产出部署快照。不执行任何真实数据库操作。
    std::function<HcclSim::Storage::DbResult<BackendDeployment>(const BackendConfigureRequest&)> configure;
};
} // namespace sim::backends

#endif // HCCL_VM_SIMULATION_STORAGE_COMPOSITION_BACKEND_MODULE_H
