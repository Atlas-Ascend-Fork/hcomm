/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：验证 simulator composition（唯一 bootstrap）的会话装配与生命周期边界。
// 装配只经后端无关 StorageRuntime：注册后端/登记两个业务目录后，按配置惰性创建
// 实际需要的逻辑路由；未配置的库不建引擎；关闭/未知库走公开合同错误路径。
#include <gtest/gtest.h>

#include <cstdio>
#include <string>

#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage_composition/storage_composition.h"
#include "storage/storage_runtime.h"
#include "storage/table_access.h"
#include "storage_backends/sqlite/composition/sqlite_backend_module.h"
using sim::backends::kSqliteBackendName;

namespace {
using HcclSim::Storage::DbCode;
using HcclSim::Storage::StorageBackend;
using HcclSim::Storage::StorageRuntime;
using HcclSim::Storage::StorageRuntimeConfig;

const char* kRunnerPath = "/tmp/test_storage_composition_runner.db";
const char* kOpDataPath = "/tmp/test_storage_composition_opdata.db";

void RemoveDatabaseFiles(const char* path)
{
    std::remove(path);
    std::remove((std::string{path} + "-wal").c_str());
    std::remove((std::string{path} + "-shm").c_str());
}

StorageRuntimeConfig TwoDatabaseConfig()
{
    return StorageRuntimeConfig{
        {{std::string{"RUNNER_DB"}, kSqliteBackendName, std::string{kRunnerPath}},
         {std::string{"OP_DATA_DB"}, kSqliteBackendName, std::string{kOpDataPath}}}};
}

class StorageCompositionTest : public testing::Test {
protected:
    void TearDown() override
    {
        (void)StorageRuntime::CloseProcessSession();
        RemoveDatabaseFiles(kRunnerPath);
        RemoveDatabaseFiles(kOpDataPath);
    }
};

// 双库安装：两个业务目录登记后，配置的库各自创建引擎并可完成公开原子 CRUD。
TEST_F(StorageCompositionTest, EnsureProcessSessionInstallsBothDatabases)
{
    RemoveDatabaseFiles(kRunnerPath);
    RemoveDatabaseFiles(kOpDataPath);
    auto registered = sim::RegisterSimulatorStorageBackends();
    ASSERT_TRUE(registered.ok()) << registered.diagnostic;
    auto ensured = StorageRuntime::EnsureProcessSession(TwoDatabaseConfig());
    ASSERT_TRUE(ensured.ok()) << ensured.diagnostic;

    sim::runtime::Device device;
    device.server_id = 1;
    device.physical_id = 2;
    ASSERT_TRUE(HcclSim::Storage::Insert(device).ok());
    const auto one
        = HcclSim::Storage::SelectOne<sim::runtime::Device>(HcclSim::Storage::Eq(&sim::runtime::Device::id, device.id));
    ASSERT_TRUE(one.ok());
    EXPECT_EQ(one.value->server_id, 1U);

    sim::operation::SyncRecordTab record;
    record.pid = 100;
    record.rankId = 1;
    ASSERT_TRUE(HcclSim::Storage::Insert(record).ok());
    const auto syncOne = HcclSim::Storage::SelectOne<sim::operation::SyncRecordTab>(
        HcclSim::Storage::Eq(&sim::operation::SyncRecordTab::id, record.id));
    ASSERT_TRUE(syncOne.ok());
    EXPECT_EQ(syncOne.value->pid, 100U);
}

// 惰性路由：只配置 RUNNER_DB 时未配置的库不建引擎，OpDB 表不可用。
TEST_F(StorageCompositionTest, UnconfiguredDatabaseDoesNotCreateRoute)
{
    RemoveDatabaseFiles(kRunnerPath);
    RemoveDatabaseFiles(kOpDataPath);
    auto registered = sim::RegisterSimulatorStorageBackends();
    ASSERT_TRUE(registered.ok());
    const StorageRuntimeConfig runnerOnly{{{std::string{"RUNNER_DB"}, kSqliteBackendName, std::string{kRunnerPath}}}};
    auto ensured = StorageRuntime::EnsureProcessSession(runnerOnly);
    ASSERT_TRUE(ensured.ok()) << ensured.diagnostic;

    // 配置内的库正常工作。
    sim::runtime::Host host;
    host.server_id = 5;
    ASSERT_TRUE(HcclSim::Storage::Insert(host).ok());
    // 未配置的库不创建路由：OpDB 表取得失败。
    auto opTable = HcclSim::Storage::AcquireTable<sim::operation::SyncRecordTab>();
    EXPECT_FALSE(opTable.ok());
}

// 未登记的库出现在配置中：安装期按目录校验拒绝。
TEST_F(StorageCompositionTest, UnknownDatabaseIsRejected)
{
    auto registered = sim::RegisterSimulatorStorageBackends();
    ASSERT_TRUE(registered.ok());
    const StorageRuntimeConfig unknown{
        {{std::string{"UNKNOWN_DB"}, kSqliteBackendName, std::string{"/tmp/unknown.db"}}}};
    auto ensured = StorageRuntime::EnsureProcessSession(unknown);
    ASSERT_FALSE(ensured.ok());
    EXPECT_EQ(ensured.code, DbCode::INVALID_ARGUMENT);
}

// 未安装会话时关闭：公开合同按无效状态拒绝。
TEST_F(StorageCompositionTest, CloseWithoutSessionIsInvalidState)
{
    (void)StorageRuntime::CloseProcessSession();
    auto closed = StorageRuntime::CloseProcessSession();
    ASSERT_FALSE(closed.ok());
    EXPECT_EQ(closed.code, DbCode::INVALID_STATE);
}

// 会话已关闭后引擎级操作按无效状态拒绝（进程退出统一关闭的唯一入口语义）。
TEST_F(StorageCompositionTest, ClosedSessionRejectsBusinessAccess)
{
    RemoveDatabaseFiles(kRunnerPath);
    RemoveDatabaseFiles(kOpDataPath);
    auto registered = sim::RegisterSimulatorStorageBackends();
    ASSERT_TRUE(registered.ok());
    ASSERT_TRUE(StorageRuntime::EnsureProcessSession(TwoDatabaseConfig()).ok());
    ASSERT_TRUE(StorageRuntime::CloseProcessSession().ok());
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::Device>();
    EXPECT_FALSE(table.ok());
}
} // namespace
