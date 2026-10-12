/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：StorageRuntime 的中立装配实现（StorageRuntimeImpl）。
// 持有后端工厂注册表与逻辑库目录登记，按 StorageRuntimeConfig 惰性创建实际需要的
// 逻辑路由（未登记的库不建引擎），通过 SessionCoordinator 安装进程唯一会话。
// 本文件属于 storage 库内部实现头，不进入公开 include/storage；具体后端
// （SQLite/MariaDB）由 composition 在工厂注册表中登记，本层不出现具体引擎类型。
#ifndef HCCL_VM_STORAGE_INTERNAL_STORAGE_RUNTIME_IMPL_H
#define HCCL_VM_STORAGE_INTERNAL_STORAGE_RUNTIME_IMPL_H

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "storage/database_result.h"
#include "storage/engine_registration.h"
#include "storage/storage_binding.h"
#include "storage/storage_catalog.h"
#include "storage/storage_runtime.h"

namespace HcclSim::Storage {
class IStorageEngine;
class SessionCoordinator;
class StorageSession;

namespace detail {

    /// 后端引擎工厂：按逻辑库路由创建一个未打开的引擎实例（由具体后端提交注册）
    using EngineFactory = std::function<DbResult<std::unique_ptr<IStorageEngine>>(const LogicalDatabaseRuntimeConfig&)>;

    /// 会话数据命名空间（BuildBindingLocked 发布的会话清单字段值；后端私有
    /// 部署路由（如 MariaDB 物理 database 名派生）必须使用同一常量，不得另造 namespace）。
    inline constexpr char kSessionDataNamespace[] = "hccl-vm";

    /**
     * @brief 进程存储装配实现：后端工厂注册表 + 逻辑库目录 + 会话编排。
     *
     * 唯一实例由 StorageRuntime::GetImpl 持有；全部公开维护操作经 StorageRuntime 转发，
     * 本类不暴露具体后端类型，也不向业务提供 engine getter。
     */
    class StorageRuntimeImpl final {
    public:
        /// 进程唯一实例（与 StorageRuntime::GetImpl 同一对象；供后端装配与测试 peer 注册工厂/目录）
        static StorageRuntimeImpl& Instance();

        /// 注册一个后端工厂（同一后端重复注册覆盖旧值；测试与后端装配期调用）
        void RegisterEngineFactory(StorageBackend backend, EngineFactory factory);

        /// 登记一个逻辑数据库的只读目录（表归属/scope 由 catalog 描述符给出；可多次登记）
        void UseDatabase(const Ddl::DatabaseDescriptor& database);

        /// 幂等安装进程会话：按配置创建引擎与表绑定，惰性创建实际需要的路由
        DbResult<void> EnsureProcessSession(const StorageRuntimeConfig& config);

        /// 排空并关闭进程会话
        DbResult<void> CloseProcessSession();

        /// 切换一个逻辑库的物理实例并重建会话
        DbResult<void> ReconfigureDatabase(const LogicalDatabaseRuntimeConfig& config);

        /// 逻辑库备份（静止闸门 + 备份会话流程由会话层编排）
        DbResult<BackupArtifact> BackupDatabase(const LogicalDatabaseId& database, const BackupRequest& request);

        /// 清空逻辑库全部静态表数据
        DbResult<void> ClearStaticData(const LogicalDatabaseId& database);

        /// 销毁逻辑库全部动态 scope 表，返回销毁数量
        DbResult<uint64_t> DropDynamicData(const LogicalDatabaseId& database);

        /// 测试与重置入口：清空全部登记（工厂与目录）
        void ResetForTest();

    private:
        struct RuntimeState;

        DbResult<void> InstallSessionLocked(const StorageRuntimeConfig& config);
        /// 由登记目录派生会话位置清单（每库路由槽）
        SessionStorageBinding BuildBindingLocked(const StorageRuntimeConfig& config) const;
        /// 查找登记目录中的逻辑库
        const Ddl::DatabaseDescriptor* FindDatabaseLocked(const std::string& databaseId) const;

        std::mutex mutex_;
        std::map<StorageBackend, EngineFactory> factories_;
        std::vector<const Ddl::DatabaseDescriptor*> databases_;
        /// 已安装会话的运行态（协调器 + 当前配置）
        std::unique_ptr<RuntimeState> state_;
    };
} // namespace detail
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_INTERNAL_STORAGE_RUNTIME_IMPL_H
