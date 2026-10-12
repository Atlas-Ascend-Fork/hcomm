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
 * @file engine_registration.h
 * @brief 存储引擎实例与逻辑库的装配绑定接口。
 *
 * 【装配模型】每个逻辑数据库一条注册（databaseId 定位）；引擎实例或惰性
 * 构造工厂二选一。会话按每库路由槽持有注册结果，首次真实访问时惰性创建
 * 引擎并纳入 single-flight 生命周期。
 */
#ifndef HCCL_VM_STORAGE_ENGINE_REGISTRATION_H
#define HCCL_VM_STORAGE_ENGINE_REGISTRATION_H

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "storage/database_result.h"
#include "storage/storage_binding.h"

namespace HcclSim::Storage {
class IStorageEngine;

/// 引擎惰性构造工厂（REQ-CAP-2）：按路由稳定配置创建一个未打开的引擎实例。
/// 会话安装只登记工厂，不创建引擎/连接；引擎创建、Open、Prepare 纳入同一每库
/// single-flight 生命周期——首次真实访问时构造，打开/准备失败丢弃实例，下次访问
/// 重新构造（终态 Close 后的同一对象不得伪装成可重试）。
using StorageEngineFactory = std::function<DbResult<std::unique_ptr<IStorageEngine>>()>;

/**
 * @brief 引擎注册装配结构体。
 *
 * @note 两种装配形态：
 *       - engine 非空：外部生命周期管理者（如测试装配）持有的预创建实例，
 *         本结构仅作为非拥有型引用传递（Non-owning pointer）；
 *       - factory 非空：惰性构造配置，首次真实访问时由会话路由槽调用并持有
 *         创建出的实例（拥有型；失败轮次后丢弃可重建）。
 */
struct EngineRegistration {
    std::string databaseId;          ///< 本注册覆盖的逻辑库名
    StorageBackend backend{};        ///< 本库后端模块注册名（空值未经装配层解析）
    IStorageEngine* engine{nullptr}; ///< 预创建引擎实例（非拥有；与 factory 二选一）
    StorageEngineFactory factory;    ///< 惰性构造工厂（与 engine 二选一）
    /// 本库 schema 打开意图覆盖（nullopt 时继承会话级 schemaRole；runtime 装配
    /// 按每库配置填入——冷 bootstrap 对每个逻辑数据库分别判定 owner/consumer）。
    std::optional<SchemaRole> schemaRole;
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_ENGINE_REGISTRATION_H
