/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：RunnerDB 回归测试的存储会话初始化辅助。
// 等价迁移前测试的 SetDbPath + CleanUpDb 模式：每个用例关闭进程会话、清理数据库文件后重新安装。
// 测试作为装配 root：经唯一 simulator bootstrap 注册后端/目录，再按测试私有路径安装
// 后端无关 StorageRuntime 进程会话（不接触具体 manager/engine）。
#ifndef RUNNERDB_STORAGE_TEST_HELPER_H
#define RUNNERDB_STORAGE_TEST_HELPER_H

#include <cstdio>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "storage_composition/storage_composition.h"
#include "storage/query_expression.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/storage_runtime.h"
#include "storage/table_access.h"

namespace runnerdb_test {
/// 关闭进程存储会话并删除测试数据库文件（主库/WAL/SHM）。
inline void CleanUpDatabases(const std::string& runnerPath, const std::string& opDataPath)
{
    (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
    for (const std::string* path : {&runnerPath, &opDataPath}) {
        std::remove(path->c_str());
        std::remove((*path + "-wal").c_str());
        std::remove((*path + "-shm").c_str());
    }
}

/// 以干净文件重建进程存储会话（等价迁移前 SetDbPath + ClearAll 的测试隔离语义）。
inline bool ResetTestSession(const std::string& runnerPath, const std::string& opDataPath)
{
    CleanUpDatabases(runnerPath, opDataPath);
    // 测试 peer 装配：经唯一 bootstrap 按测试私有路由安装会话
    // （同时登记当前路由，备份等维护观测由此取得端点）。
    HcclSim::Storage::StorageRuntimeConfig config;
    config.databases.push_back({std::string{"RUNNER_DB"}, "sqlite", runnerPath});
    config.databases.push_back({std::string{"OP_DATA_DB"}, "sqlite", opDataPath});
    return sim::EnsureSimulatorStorageSession(config).ok();
}

// 测试只通过公共 StorageSession/Table 合同准备数据；这些辅助函数不属于产品业务层。
template <typename T>
uint64_t InsertRecord(T& record)
{
    auto table = HcclSim::Storage::AcquireTable<T>();
    if (!table.ok()) {
        return 0;
    }
    record.id = 0;
    auto inserted = table->Insert(record);
    return inserted.ok() && inserted.value.has_value() ? inserted.value->value : 0;
}

// 按目录合同推导主键字段的条件（测试辅助，不手写数字编号）
template <typename T>
HcclSim::Storage::Condition<T> PrimaryKeyCondition(uint64_t id)
{
    HcclSim::Storage::Condition<T> condition;
    const auto& meta = HcclSim::Storage::RecordSchema<T>::Meta();
    for (std::size_t fieldId = 0; fieldId < meta.fieldCount; ++fieldId) {
        if (meta.fields[fieldId].primaryKey) {
            condition.data = HcclSim::Storage::ConditionClause{
                static_cast<uint16_t>(fieldId),
                HcclSim::Storage::Compare::EQUAL,
                HcclSim::Storage::ConditionJoin::NONE,
                0,
                0,
                HcclSim::Storage::ConditionValue{HcclSim::Storage::DbScalar{id}}};
            return condition;
        }
    }
    return condition;
}

template <typename T>
std::optional<T> SelectRecord(uint64_t id)
{
    auto table = HcclSim::Storage::AcquireTable<T>();
    if (!table.ok()) {
        return std::nullopt;
    }
    auto selected = table->SelectOne(PrimaryKeyCondition<T>(id));
    if (!selected.ok() || !selected.value.has_value()) {
        return std::nullopt;
    }
    return std::move(*selected.value);
}

template <typename T, typename... Clauses>
std::vector<T> SelectRecords(HcclSim::Storage::Condition<T> where, const Clauses&... clauses)
{
    auto table = HcclSim::Storage::AcquireTable<T>();
    if (!table.ok()) {
        return {};
    }
    auto selected = table->Select(where, clauses...);
    return selected.ok() ? std::move(*selected) : std::vector<T>{};
}

template <typename T, typename... Clauses>
std::pair<T, bool> SelectFirstRecord(HcclSim::Storage::Condition<T> where, const Clauses&... clauses)
{
    std::pair<T, bool> result{};
    auto table = HcclSim::Storage::AcquireTable<T>();
    if (!table.ok()) {
        return result;
    }
    auto selected = table->SelectOne(where, clauses...);
    if (selected.ok() && selected.value.has_value()) {
        result.first = std::move(*selected.value);
        result.second = true;
    }
    return result;
}

// 全量读取不再隐含主键顺序：依赖顺序的用例显式传入 OrderBy。
template <typename T>
std::vector<T> SelectAllRecords()
{
    return SelectRecords<T>(HcclSim::Storage::All<T>());
}

template <typename T>
bool UpdateRecord(uint64_t id, HcclSim::Storage::Assignment<T> change)
{
    auto table = HcclSim::Storage::AcquireTable<T>();
    if (!table.ok()) {
        return false;
    }
    auto updated = table->Update(PrimaryKeyCondition<T>(id), change);
    return updated.ok() && updated.value.has_value() && *updated.value > 0;
}

template <typename T>
bool DeleteRecord(uint64_t id)
{
    auto table = HcclSim::Storage::AcquireTable<T>();
    if (!table.ok()) {
        return false;
    }
    auto deleted = table->Delete(PrimaryKeyCondition<T>(id));
    return deleted.ok() && deleted.value.has_value() && *deleted.value > 0;
}

template <typename T>
bool ClearRecords()
{
    auto table = HcclSim::Storage::AcquireTable<T>();
    return table.ok() && table->Clear(HcclSim::Storage::ClearMode::RESET_SEQUENCE).ok();
}
} // namespace runnerdb_test

#endif
