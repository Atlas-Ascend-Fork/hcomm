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
 * See the License in the root of the software repository for the full text of the License.
 */

/**
 * @file record_contract.h
 * @brief 定义业务结构体与存储目录之间的 Schema 合同 Traits。
 *
 * 【表契约双 Traits（由 StorageCatalog 统一生成）】
 *
 *   1. RecordSchema<T> : 声明 TableId/DatabaseId/ScopeKind/SchemaVersion、进程期
 *      稳定的静态表元信息 Meta()（CatalogTableMeta，来自同一 TableDdl<T> 静态
 *      定义）及成员指针到字段编号的映射
 *   2. RecordCodec<T>  : 主键读写（自增 id 写回与主键值读取；生产 CRUD 的记录
 *      数据经 Catalog 回调（visitRecord/materializeRows）原始引用直通，不存在
 *      类型擦除的中间行表示）
 *
 * 特化由 storage_catalog.h 的目录登记机制自动生成（一个 Table entry 派生全部产物），
 * 业务侧不手写字段清单。
 *
 * 【成员指针到字段编号的映射】
 * 业务表达式（Eq/OrderBy/Set 等）以 `&T::member` 成员指针声明目标字段；
 * RecordSchema<T>::FieldIdOf(member) 负责把成员指针解析为字段编号
 *（即声明顺序下标），业务源文件因此不出现手写字符串字段名或数字编号。
 * 未登记成员返回 kUnknownFieldId。
 */
#ifndef HCCL_VM_STORAGE_RECORD_CONTRACT_H
#define HCCL_VM_STORAGE_RECORD_CONTRACT_H

#include <cstdint>
#include <string_view>

#include "storage/database_result.h"
#include "storage/query_expression.h"
#include "storage/query_spec.h"

namespace HcclSim::Storage::Ddl {
enum class ScopeKind : uint8_t; // 定义见 storage_catalog.h（表 scope 类型）
struct CatalogTableMeta;        // 定义见 storage_catalog.h（进程期静态表元信息）
} // namespace HcclSim::Storage::Ddl

namespace HcclSim::Storage {
/// 逻辑字段数据类型
enum class LogicalFieldType : uint8_t { SIGNED, UNSIGNED, REAL, BOOLEAN, STRING, BYTES };

/**
 * @brief 结构体 T 的逻辑 Schema 特化 Traits 模板。
 */
template <typename T>
struct RecordSchema {
    static std::string_view TableId();
    /// 表 → 逻辑库归属（编译期静态数据；会话静态路由的单一来源）
    static std::string_view DatabaseId();
    /// 表 scope 类型（静态公共表 / PID 动态表；目录声明）
    static Ddl::ScopeKind ScopeKind();
    static uint32_t SchemaVersion();
    /// 进程期稳定的静态表元信息（来自同一 TableDdl<T> 静态定义，不构造动态
    /// 字段 vector、不复制字段和索引）
    static const Ddl::CatalogTableMeta& Meta();
    /// 成员指针到字段编号的映射；未登记成员返回 kUnknownFieldId
    template <typename M>
    static uint16_t FieldIdOf(M T::*member);
};

/**
 * @brief 结构体 T 的主键读写 Codec 特化 Traits 模板。
 */
template <typename T>
struct RecordCodec {
    static PrimaryKey<T> ReadPrimaryKey(const T& record);
    static DbResult<void> WritePrimaryKey(T& record, PrimaryKey<T> key);
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_RECORD_CONTRACT_H
