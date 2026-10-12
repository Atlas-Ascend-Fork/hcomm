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
 * @file i_storage_engine.h
 * @brief 各存储后端引擎必须共同实现的显式能力接口。
 *
 * 【存储引擎抽象生命周期模型】
 *
 *   1. 装配阶段 : Open(openRequest) ──> Prepare() 预编译脚本/校验 Schema
 *   2. 读写阶段 : 每次业务操作只调用一个对应虚函数（Insert/Select/Count/
 *                Update/Delete/Clear/ListTables），无运行期 kind 分派
 *   3. 事务阶段 : Begin(tx) ──> 带 tx 指针的 DML ──> Commit(tx) / Rollback(tx)
 *   4. 收尾阶段 : Close() 显式关闭物理连接/释放资源
 *
 * 【数据所有权合同】公开业务 API 完成唯一参数对象构造后，本接口的全部方法
 * 只接收 const 引用/原始指针/静态 meta 指针/标量；SELECT 经 ResultSink
 * 直接写最终业务目标；INSERT 返回生成主键，COUNT/UPDATE/DELETE 返回整数。
 * 事务 DML 与普通 DML 共用同一实现：transaction 为空表示即时执行，非空
 * 表示属于当前已 Begin 的活动事务（引擎按 binding 地址校验）。
 */
#ifndef HCCL_VM_STORAGE_ENGINE_I_STORAGE_ENGINE_H
#define HCCL_VM_STORAGE_ENGINE_I_STORAGE_ENGINE_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "storage/database_result.h"
#include "storage/internal/engine_contract_types.h"
#include "storage/logical_operation.h"
#include "storage/query_spec.h"
#include "storage/storage_catalog.h"

namespace HcclSim::Storage {
namespace Ddl {
    struct CatalogTableMeta;
} // namespace Ddl

/**
 * @brief 存储引擎抽象基类（Pure Virtual Interface）。
 */
class IStorageEngine {
public:
    virtual ~IStorageEngine() = default;

    /// 打开当前物理路由对应的存储实例单元
    virtual DbResult<void> Open(const EngineOpenRequest&) = 0;

    /// 在会话发布前完成后端所需的结构准备（owner 建表/索引，consumer 只读打开）
    virtual DbResult<void> Prepare() = 0;

    /// 插入单条记录；返回生成主键（自增主键写回由 Table<T> 完成）。
    /// transaction == nullptr 为即时执行；非空必须属于当前已 Begin 的活动事务。
    virtual DbResult<uint64_t> Insert(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* record, TransactionBinding* transaction)
        = 0;

    /// 事务内同表批量插入（T1）：records 为同表最终业务记录的指针数组（本层
    /// 不复制业务载荷）；成功后 generatedIds 按输入顺序填各生成主键并返回
    /// affectedRows=count，失败输出不可消费（diagnostic 含本批首个错误的
    /// 0 基行号）。只接受已开启事务（transaction 非空）；count=0 平凡成功。
    /// 默认实现按原顺序逐条调用本引擎 Insert（同一事务 binding，不经
    /// Table/Transaction 重复记账），失败即返回——回滚仅由 Transaction 统一
    /// 执行。
    virtual DbResult<void> InsertMany(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* const* records, std::size_t count,
        uint64_t* generatedIds, TransactionBinding* transaction)
    {
        if (count == 0U) {
            return {DbCode::OK, 0, {}};
        }
        if (records == nullptr || generatedIds == nullptr || transaction == nullptr) {
            return {DbCode::INVALID_ARGUMENT, 0, "insert-many requires records, an id array and an active transaction"};
        }
        if (count > kInsertBatchMaxRows) {
            return {DbCode::INVALID_ARGUMENT, 0, "insert-many batch exceeds kInsertBatchMaxRows"};
        }
        for (std::size_t index = 0; index < count; ++index) {
            const auto inserted = Insert(meta, scope, records[index], transaction);
            if (!inserted.ok()) {
                return {inserted.code, inserted.affectedRows, inserted.diagnostic + " at row " + std::to_string(index)};
            }
            generatedIds[index] = *inserted.value;
        }
        return {DbCode::OK, count, {}};
    }

    /// 条件查询：过滤/排序/数量下推数据库端，结果经 materializeRows 直接写
    /// ResultSink 所指的最终业务对象/容器
    virtual DbResult<void> Select(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
        const QueryOptions& options, ResultSink& sink)
        = 0;

    /// 同表多作用域批量条件查询（Q2 唯一新增公共存储接口）：scopes 与 sinks
    /// 一一对应，第 i 个作用域的结果写入 sinks[i]（每作用域独立应用排序与
    /// 数量限制）。默认实现按原顺序逐作用域调用 Select——SQLite/memory 等
    /// 后端继承该循环，不新增第二套查询实现；遇错立即返回，调用方不得消费
    /// 部分结果。空列表平凡成功
    virtual DbResult<void> SelectMany(
        const Ddl::CatalogTableMeta& meta, const std::vector<TableScope>& scopes, const ConditionData& where,
        const QueryOptions& options, std::vector<ResultSink>& sinks)
    {
        if (scopes.size() != sinks.size()) {
            return {DbCode::INVALID_ARGUMENT, 0, "select-many requires matching scope and sink counts"};
        }
        for (std::size_t index = 0; index < scopes.size(); ++index) {
            const auto selected = Select(meta, scopes[index], where, options, sinks[index]);
            if (!selected.ok()) {
                return selected;
            }
        }
        return {DbCode::OK, 0, {}};
    }

    /// 条件计数（数据库端 COUNT 聚合）
    virtual DbResult<uint64_t>
    Count(const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where) = 0;

    /// 条件计数聚合：一条聚合语句同时返回 where 集合总数与其中匹配 match 的
    /// 行数（CountMatchResult）。两个计数必须来自同一语句——先 Count(total)
    /// 再 Count(matched) 会引入并发插入/完成更新导致的两次读取漂移，禁止。
    /// 未实现该能力的引擎保持默认实现：明确返回 UNSUPPORTED_CAPABILITY，
    /// 不退化为两次 Count，也不经 Select 加载行集后扫描。SQLite/MariaDB
    /// 必须覆盖本方法。
    virtual DbResult<CountMatchResult> CountMatches(
        [[maybe_unused]] const Ddl::CatalogTableMeta& meta, [[maybe_unused]] const TableScope& scope,
        [[maybe_unused]] const ConditionData& where, [[maybe_unused]] const ConditionData& match)
    {
        return {
            DbCode::UNSUPPORTED_CAPABILITY, std::nullopt, 0, "count-matches aggregate is not supported by this engine"};
    }

    /// 条件更新：单语句 UPDATE SET ... WHERE ...；参数顺序固定为 UpdateData
    /// 在前、ConditionData 在后。transaction 语义同 Insert。
    virtual DbResult<uint64_t> Update(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
        const UpdateData& update, TransactionBinding* transaction)
        = 0;

    /// 条件删除；返回被删除行数。transaction 语义同 Insert。
    virtual DbResult<uint64_t> Delete(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
        TransactionBinding* transaction)
        = 0;

    /// 清空单表（KEEP_SEQUENCE / RESET_SEQUENCE / DROP_DYNAMIC_SCOPE）
    virtual DbResult<void> Clear(const Ddl::CatalogTableMeta& meta, const TableScope& scope, ClearMode mode) = 0;

    /// 列出当前库的用户表名
    virtual DbResult<std::vector<std::string>> ListTables() = 0;

    /// 在当前连接上开启一个物理事务（同一时刻只允许一个活动 binding）
    virtual DbResult<void> Begin(TransactionBinding& transaction) = 0;

    /// 提交当前活动物理事务（只处理物理边界，不解释/不重放 DML）
    virtual DbResult<AtomicResult> Commit(TransactionBinding& transaction) = 0;

    /// 回滚当前活动物理事务
    virtual DbResult<void> Rollback(TransactionBinding& transaction) = 0;

    /// 显式关闭引擎连接并释放物理资源
    virtual DbResult<void> Close() = 0;

    /// host 启动专用（envInit 调用）：把本后端持久化数据整体清回全新态——等价
    /// 于 SQLite 的"删库文件"，为外部后端（MariaDB）提供同样的干净起点。
    /// 仅 owner（host 的 `hccl-vm start`）在建立业务会话前调用一次；rank/checker/
    /// 普通消费者进程绝不调用（它们以 READ_EXISTING 连接既有库）。dataNamespace
    /// 用于派生物理库/键命名空间，与 Open 请求保持同一取值。未实现/不需要的
    /// 后端走默认实现（返回 UNSUPPORTED_CAPABILITY，由调用方按需忽略或告警）。
    virtual DbResult<void> Wipe([[maybe_unused]] std::string_view dataNamespace)
    {
        return {DbCode::UNSUPPORTED_CAPABILITY, 0, "storage backend wipe is not supported"};
    }
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_ENGINE_I_STORAGE_ENGINE_H
