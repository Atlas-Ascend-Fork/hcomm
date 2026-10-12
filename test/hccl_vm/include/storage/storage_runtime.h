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
 * @file storage_runtime.h
 * @brief 后端中立进程存储生命周期与维护合同（StorageRuntime，PImpl 隐藏装配细节）。
 *
 * 【装配边界】
 *   业务：Storage::Insert/SelectOne/Select/Count/Update/Delete/Clear、AcquireTable、BeginTransaction
 *     -> 进程全局 StorageRuntime（本类，公开面全部后端无关）
 *          -> detail::StorageRuntimeImpl（公开头仅前置声明，实现私有）
 *               -> 后端工厂注册表（各后端模块经 composition 注册，键为模块注册名）
 *                    -> IStorageEngine -> 各后端引擎（构造与控制面私有）
 *
 * 普通 CRUD 不读取 runtime 配置；只有 main/启动装配和维护入口构造 StorageRuntimeConfig。
 * 业务与公共 record contract 不出现具体 manager/engine/mapping 类型；
 * 不存在具体后端 getter（GetSQLiteEngine/RunnerEngine 等逃逸接口一律不提供）。
 */
#ifndef HCCL_VM_STORAGE_STORAGE_RUNTIME_H
#define HCCL_VM_STORAGE_STORAGE_RUNTIME_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "storage/backup_contract.h"
#include "storage/database_result.h"
#include "storage/storage_binding.h"

namespace HcclSim::Storage {
namespace detail {
    class StorageRuntimeImpl;
}

/// 逻辑数据库标识（canonical 逻辑库名，如 "RUNNER_DB"/"OP_DATA_DB"；由 StorageCatalog 的 Database 声明）
using LogicalDatabaseId = std::string;

/// 单个逻辑数据库的部署路由（后端模块注册名 + 部署端点；文件路径或服务 URI 由
/// 各后端模块定义）。backend 为空表示尚未解析的默认请求，必须先经装配层解析。
struct LogicalDatabaseRuntimeConfig {
    LogicalDatabaseId database;
    StorageBackend backend{};
    std::string endpoint;
    /// 本库 schema 打开意图覆盖：nullopt 继承 StorageRuntimeConfig.schemaRole
    /// （会话全局角色）；显式值才覆盖全局。三字段聚合初始化保持兼容（继承全局）。
    /// 冷 bootstrap 对已存在文件按 ReadWriteExisting、缺失文件按 OwnerPrepare 兜底时
    /// 显式传入；普通消费者进程以全局 ReadWriteExisting 装配、逐库不指定。
    std::optional<SchemaRole> schemaRole{std::nullopt};
};

/// 进程存储装配配置：每个逻辑库一条路由；未出现的库不在本进程打开
struct StorageRuntimeConfig {
    std::vector<LogicalDatabaseRuntimeConfig> databases;
    /// 静态 schema 打开意图（默认 owner：显式装配边界建库；普通消费者进程
    /// 由 composition 以 ReadWriteExisting 装配，不重复执行 DDL）
    SchemaRole schemaRole{SchemaRole::OwnerPrepare};
};

/**
 * @brief 进程存储生命周期与维护的唯一公开装配入口（PImpl；进程全局单实例）。
 *
 * 合同：
 * - EnsureProcessSession 幂等安装进程唯一逻辑会话；按配置惰性创建实际需要的
 *   逻辑路由（未登记的库不建引擎）；fork 子进程检测到安装进程不一致时重建；
 * - CloseProcessSession 排空在途操作并统一关闭（正常退出路径唯一入口）；
 * - ReconfigureDatabase 切换一个逻辑库的物理实例（换库/加载备份语义）；
 * - BackupDatabase/ClearStaticData/DropDynamicData 只接收逻辑库与后端无关请求；
 * - 具体后端对象的创建、注册与持有全部在 StorageRuntimeImpl 内部，公开面无后端类型。
 */
class StorageRuntime final {
public:
    friend class detail::StorageRuntimeImpl;
    /// 幂等安装进程全局逻辑会话（检测 fork 后 PID 变化并在子进程重建）
    static DbResult<void> EnsureProcessSession(const StorageRuntimeConfig& config);

    /// 显式排空并关闭进程存储会话（重复关闭按无效状态拒绝）
    static DbResult<void> CloseProcessSession();

    /// 切换一个逻辑数据库的物理实例并重建会话（维护入口）
    static DbResult<void> ReconfigureDatabase(const LogicalDatabaseRuntimeConfig& config);

    /// 生成一个逻辑数据库的物理备份（静止闸门 + 备份会话；维护入口）
    static DbResult<BackupArtifact> BackupDatabase(const LogicalDatabaseId& database, const BackupRequest& request);

    /// 清空一个逻辑数据库的全部静态表数据（保留动态表；维护入口）
    static DbResult<void> ClearStaticData(const LogicalDatabaseId& database);

    /// 销毁一个逻辑数据库的全部动态 scope 表；返回被销毁的动态表数量（维护入口）
    static DbResult<uint64_t> DropDynamicData(const LogicalDatabaseId& database);

private:
    StorageRuntime() = delete;
    static detail::StorageRuntimeImpl& GetImpl();
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_STORAGE_RUNTIME_H
