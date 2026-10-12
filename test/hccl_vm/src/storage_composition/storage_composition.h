/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：模拟器存储装配的唯一 bootstrap/composition 入口。
// 本头文件是业务侧接触存储装配的唯一入口：经生成的模块注册清单发现已编入的
// 后端模块（引擎/工厂/profile 全部驻留各模块目录），登记两个逻辑库目录
// （Runner/OperationData 的统一 StorageCatalog）、构造安装根下的默认逻辑路由并经
// 后端无关 StorageRuntime 安装进程会话。
// 普通 CRUD/生命周期调用方只包含本公开头，不接触具体 manager/engine/session lease。
#ifndef HCCL_VM_SIMULATION_STORAGE_COMPOSITION_H
#define HCCL_VM_SIMULATION_STORAGE_COMPOSITION_H

#include <string>

#include "storage/backup_contract.h"
#include "storage/database_result.h"
#include "storage/storage_runtime.h"

namespace sim {

/// 注册可用存储后端并登记两个逻辑库目录（幂等）。
/// 只经 StorageRuntime 的后端无关合同；未配置的逻辑库不创建引擎。
HcclSim::Storage::DbResult<void> RegisterSimulatorStorageBackends();

/// host 启动专用（envInit 调用，owner-only）：把持久化后端上一轮运行的数据整体
/// 清回全新态——等价文件后端在 envInit 里删 data/ 目录（含 db 文件）的干净起点。
/// 按当前逻辑路由分派到对应后端模块的 Wipe（无需处理的后端返回成功 no-op；
/// 不建立业务会话）。rank/checker/消费者进程不经 envInit，绝不触发本入口。
HcclSim::Storage::DbResult<void> WipeSimulatorStorage();

/// 按安装根下的默认逻辑路由安装进程存储会话（幂等）。fork 合同：所有产品 fork
/// child 立即 exec，exec 后的新进程经正常初始化/惰性 bootstrap 建立自己的会话。
/// 迁移前语义：业务首次访问数据库前完成建连与全部表结构初始化。
HcclSim::Storage::DbResult<void> EnsureSimulatorStorageSession();

/// 按显式逻辑路由安装进程存储会话（测试 peer/部署装配用；记录当前路由供备份观测）。
HcclSim::Storage::DbResult<void> EnsureSimulatorStorageSession(const HcclSim::Storage::StorageRuntimeConfig& config);

/// 维护入口：切换一个逻辑库的物理实例并重装会话（记录当前路由供备份等观测）。
/// 等价迁移前换库（加载备份文件）语义；只有维护入口构造 RuntimeConfig。
HcclSim::Storage::DbResult<void>
ReconfigureSimulatorDatabase(const HcclSim::Storage::LogicalDatabaseRuntimeConfig& config);

/// 维护入口：把一个逻辑库对齐到主进程在任务文件中记录的数据位置（业务只
/// 表达维护意图，不感知部署后端类型；部署形态差异由装配层裁定）。
/// recordedPath 为任务文件记录的路径形态字段：文件后端部署下重配到该路径
/// （等价迁移前"换库/加载备份"语义）；服务端后端部署下数据位置由部署配置
/// 决定（HCCL_VM_STORAGE_CONFIG 经环境继承），路径字段无意义——保持部署路由
/// 不切换（主控/rank/插件进程读写同一部署实例）。部署形态差异由各后端模块的
/// 回调裁定，公共层不按后端名分支。
HcclSim::Storage::DbResult<void> ReconfigureDatabaseToRecordedLocation(
    const HcclSim::Storage::LogicalDatabaseId& database, const std::string& recordedPath);

/// 维护入口：生成一个逻辑库的物理备份（后端模块的专用备份连接流程，与业务会话
/// 独立；等价迁移前备份只读连接装配，不接触业务 manager/engine）。
HcclSim::Storage::DbResult<HcclSim::Storage::BackupArtifact> BackupSimulatorDatabase(
    const HcclSim::Storage::LogicalDatabaseId& database, const HcclSim::Storage::BackupRequest& request);
} // namespace sim

#endif // HCCL_VM_SIMULATION_STORAGE_COMPOSITION_H
