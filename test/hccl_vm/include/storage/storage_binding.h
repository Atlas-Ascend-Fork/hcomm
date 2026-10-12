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

/**
 * @file storage_binding.h
 * @brief 定义会话级位置清单、打开模式与静止令牌。
 *
 * 【位置清单与路由解析数据流】
 *
 *   1. [Host 启动期] ──> SessionCoordinator 校验 Placement 配置
 *                          │
 *                          ▼ 发布全局只读清单
 *                   SessionStorageBinding (会话位置清单)
 *                   ├── sessionId       : 会话标识 (UUID)
 *                   ├── dataNamespace   : 数据命名空间
 *                   └── databases[]     : 逻辑库清单（目录指针 + 物理路径）
 *                          │
 *                          ▼ [消费进程打开会话]
 *   2. [消费进程]   ──> 每库路由槽（SessionDatabaseRoute）
 *                          │
 *                          ▼ [业务发起 Table<T> 访问]
 *                   RecordSchema<T>::DatabaseId() 编译期定位库路由槽
 *                          │
 *                          ▼ 解析产出
 *                   engine 指针 + 静态 meta + scope（Table<T> 直接持有）
 *
 * 会话不保存逐表绑定、字段映射、SQL、查询条件或查询历史；
 * 表定义唯一来源是静态目录（CatalogTableMeta）。
 */
#ifndef HCCL_VM_STORAGE_STORAGE_BINDING_H
#define HCCL_VM_STORAGE_STORAGE_BINDING_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "storage/query_spec.h"

namespace HcclSim::Storage::Ddl {
struct DatabaseDescriptor; // 前置声明：会话库绑定只持静态目录只读指针（定义见 storage_catalog.h）
} // namespace HcclSim::Storage::Ddl

namespace HcclSim::Storage {
/// 存储后端标识：后端模块注册名（如编译入树的 sqlite/mariadb 模块目录名）。
/// 公共层不枚举具体后端——装配层在进入 Runtime 之前把默认请求解析为实际
/// 模块名；空值不是已解析的合法后端（Runtime 工厂表按名查找，空名必失败）。
using StorageBackend = std::string;

/// 存储会话打开模式
enum class StorageOpenMode : uint8_t {
    READ_WRITE_CREATE, ///< 正常读写并按需建表（schema owner 打开意图）
    READ_EXISTING, ///< 读写已存在的库文件（普通 consumer：不建文件、不执行静态 DDL，缺库/缺表/列漂移明确失败）
    BACKUP_ONLY    ///< 仅备份模式（仅允许执行 Backup 操作）
};

/// 进程对静态 schema 的打开意图（内部装配角色，不进入业务 API）：
/// OwnerPrepare——本进程负责创建/迁移数据库文件并执行全部静态表/索引 DDL
/// （启动/预置数据的主控与显式维护边界）；ReadWriteExisting——普通消费者
/// （rank/plugin/device 等），读写已存在的库文件，不执行静态 DDL。
enum class SchemaRole : uint8_t { OwnerPrepare, ReadWriteExisting };

/**
 * @brief 写入静止令牌（QuiesceToken）。
 *
 * 【生命周期与失效模型】
 *   ApplyQuiesceGate() ──> 排空在途操作 ──> 生成 QuiesceToken(generation, active=true)
 *          │
 *          ├── 传递给 OpenForBackup() ──> 创建仅备份会话
 *          ├── 传递给 ExecuteReset()   ──> 执行受控的表重置 (ResetPlan)
 *          │
 *   ReleaseQuiesceGate() ──> active->store(false) ──> 所有持有该令牌的副本立即全局失效！
 */
class QuiesceToken {
public:
    QuiesceToken() = default;

    /// 校验令牌当前是否仍然有效（必须处于 active 纪元且 generation != 0）
    bool Valid() const noexcept
    {
        return generation_ != 0 && active_ != nullptr && active_->load(std::memory_order_acquire);
    }

    /// 获取令牌所属的数据命名空间
    const std::string& DataNamespace() const noexcept { return dataNamespace_; }

    /// 获取令牌所属的会话 ID
    const std::string& SessionId() const noexcept { return sessionId_; }

    /// 判断两个令牌是否来自同一会话的同一次静止纪元（Same Epoch）
    bool SameEpoch(const QuiesceToken& other) const noexcept
    {
        return generation_ != 0 && generation_ == other.generation_ && sessionId_ == other.sessionId_
               && active_ != nullptr && active_ == other.active_;
    }

private:
    QuiesceToken(
        uint64_t generation, std::string dataNamespace, std::string sessionId,
        std::shared_ptr<std::atomic<bool>> active)
        : generation_(generation),
          dataNamespace_(std::move(dataNamespace)),
          sessionId_(std::move(sessionId)),
          active_(std::move(active))
    {}
    uint64_t generation_{0};                    ///< 静止纪元自增编号
    std::string dataNamespace_;                 ///< 数据命名空间
    std::string sessionId_;                     ///< 绑定的会话 ID
    std::shared_ptr<std::atomic<bool>> active_; ///< 共享失效控制标志
    friend class SessionControlBlock;
};

/**
 * @brief 物理数据库绑定信息。
 */
struct StorageDatabaseBinding {
    std::string databaseId;   ///< 逻辑库名
    std::string databasePath; ///< 数据库物理路径/地址
    /// 本库静态目录（只读指针，由装配层从登记目录提供；会话按库引用，不复制表清单）
    const Ddl::DatabaseDescriptor* catalog{nullptr};
};

/**
 * @brief SessionCoordinator 发布的会话级只读位置清单。
 */
struct SessionStorageBinding {
    std::string sessionId;                         ///< 会话唯一标识
    std::string dataNamespace;                     ///< 数据命名空间
    std::vector<StorageDatabaseBinding> databases; ///< 包含的物理库清单
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_STORAGE_BINDING_H
