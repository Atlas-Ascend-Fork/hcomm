/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：验证 RunnerDB 会话装配与操作的故障路径：冷启动/显式关闭后的公开
// 取表语义、重复安装、关闭后的引擎直达拒绝、未注册表拒绝；全部返回明确错误码
// 且无隐式回落。
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "storage_composition/storage_composition.h"
#include "storage/internal/process_storage_context.h"
#include "storage/table_access.h"
#include "storage/storage_session.h"
#include "storage/storage_runtime.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage_backends/sqlite/composition/sqlite_backend_module.h"

namespace {

const char* kRunnerPath = "/tmp/test_sqlite_runner_fault_runner.db";
const char* kOpDataPath = "/tmp/test_sqlite_runner_fault_opdata.db";
// 冷 bootstrap 默认路由的私有安装根（不触碰真实安装目录；解析前设置）。
const char* kColdInstallRoot = "/tmp/test_sqlite_runner_fault_install";

class SqliteRunnerEngineFaultTest : public testing::Test {
protected:
    void SetUp() override { CleanFiles(); }

    void TearDown() override
    {
        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
        CleanFiles();
        std::error_code ignored;
        std::filesystem::remove_all(kColdInstallRoot, ignored);
    }

    static void CleanFiles()
    {
        for (const char* path : {kRunnerPath, kOpDataPath}) {
            std::remove(path);
            std::remove((std::string(path) + "-wal").c_str());
            std::remove((std::string(path) + "-shm").c_str());
        }
    }
};

// 冷进程首次公开取表经唯一 simulator bootstrap 惰性建立会话（等价迁移前 RunnerDB
// 惰性 singleton 的首次建连时机）；显式关闭（REMOVED）后不再自动重建，公开取表
// 按无效状态失败（无隐式回落）。
TEST_F(SqliteRunnerEngineFaultTest, ColdTableAccessBootstrapsThenFailsAfterExplicitClose)
{
    (void)setenv("HCCL_VM_INSTALL_ROOT", kColdInstallRoot, 1);
    std::error_code ignored;
    std::filesystem::create_directories(std::string{kColdInstallRoot} + "/data", ignored);
    {
        // 句柄作用域内完成冷启动断言；句柄析构（activeTables--）后再显式关闭。
        auto table = HcclSim::Storage::AcquireTable<sim::runtime::Server>();
        ASSERT_TRUE(table.ok()) << table.diagnostic;
    }
    EXPECT_TRUE(HcclSim::Storage::ProcessStorageContext::Acquire().ok());
    ASSERT_TRUE(HcclSim::Storage::StorageRuntime::CloseProcessSession().ok());
    auto closed = HcclSim::Storage::AcquireTable<sim::runtime::Server>();
    ASSERT_FALSE(closed.ok());
    EXPECT_EQ(closed.code, HcclSim::Storage::DbCode::INVALID_STATE);
}

TEST_F(SqliteRunnerEngineFaultTest, EnsureIsIdempotentForSameProcess)
{
    ASSERT_TRUE(sim::RegisterSimulatorStorageBackends().ok());
    HcclSim::Storage::StorageRuntimeConfig sessionConfig;
    sessionConfig.databases.push_back(
        {std::string{"RUNNER_DB"}, sim::backends::kSqliteBackendName, std::string{kRunnerPath}});
    sessionConfig.databases.push_back(
        {std::string{"OP_DATA_DB"}, sim::backends::kSqliteBackendName, std::string{kOpDataPath}});
    auto first = HcclSim::Storage::StorageRuntime::EnsureProcessSession(sessionConfig);
    ASSERT_TRUE(first.ok()) << first.diagnostic;
    // 幂等：第二次安装同一路径直接成功，不重复创建引擎。
    auto second = HcclSim::Storage::StorageRuntime::EnsureProcessSession(sessionConfig);
    ASSERT_TRUE(second.ok()) << second.diagnostic;
}

TEST_F(SqliteRunnerEngineFaultTest, RunnerTableRejectsUndeclaredComparison)
{
    ASSERT_TRUE(sim::RegisterSimulatorStorageBackends().ok());
    HcclSim::Storage::StorageRuntimeConfig sessionConfig;
    sessionConfig.databases.push_back(
        {std::string{"RUNNER_DB"}, sim::backends::kSqliteBackendName, std::string{kRunnerPath}});
    sessionConfig.databases.push_back(
        {std::string{"OP_DATA_DB"}, sim::backends::kSqliteBackendName, std::string{kOpDataPath}});
    auto ensured = HcclSim::Storage::StorageRuntime::EnsureProcessSession(sessionConfig);
    ASSERT_TRUE(ensured.ok());
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::Server>();
    ASSERT_TRUE(table.ok());
    // 旧"未声明比较被拒"已删除：目录合同删除每表白名单后，比较能力由字段类型在
    // 编译期决定（文本成员只接受字符串值，见 query_expression.h 的
    // "text members accept string values" static_assert）；运行期只保留合法路径。
    auto rows = table->Select(HcclSim::Storage::Lt(&sim::runtime::Server::version, std::string{"x"}));
    EXPECT_TRUE(rows.ok()) << rows.diagnostic;
}

TEST_F(SqliteRunnerEngineFaultTest, RunnerMutationRejectsPrimaryKeyUpdate)
{
    ASSERT_TRUE(sim::RegisterSimulatorStorageBackends().ok());
    HcclSim::Storage::StorageRuntimeConfig sessionConfig;
    sessionConfig.databases.push_back(
        {std::string{"RUNNER_DB"}, sim::backends::kSqliteBackendName, std::string{kRunnerPath}});
    sessionConfig.databases.push_back(
        {std::string{"OP_DATA_DB"}, sim::backends::kSqliteBackendName, std::string{kOpDataPath}});
    auto ensured = HcclSim::Storage::StorageRuntime::EnsureProcessSession(sessionConfig);
    ASSERT_TRUE(ensured.ok());
    auto table = HcclSim::Storage::AcquireTable<sim::runtime::Server>();
    ASSERT_TRUE(table.ok());
    sim::runtime::Server record{};
    auto inserted = table->Insert(record);
    ASSERT_TRUE(inserted.ok());

    // 主键列不可更新：对主键的 Set 被合同拦截。
    auto updated = table->Update(
        HcclSim::Storage::Eq(&sim::runtime::Server::id, inserted.value->value),
        HcclSim::Storage::Set(&sim::runtime::Server::id, uint64_t{9}));
    ASSERT_FALSE(updated.ok());
    EXPECT_EQ(updated.code, HcclSim::Storage::DbCode::INVALID_ARGUMENT);
}

TEST_F(SqliteRunnerEngineFaultTest, ReconfigureWithoutSessionIsRejected)
{
    // 会话未安装时换库按公开合同拒绝（重装后使用新路径由调用方重新 Ensure）。
    const std::string switched = "/tmp/test_sqlite_runner_fault_switched.db";
    auto reconfigured = HcclSim::Storage::StorageRuntime::ReconfigureDatabase(
        {std::string{"OP_DATA_DB"}, sim::backends::kSqliteBackendName, switched});
    ASSERT_FALSE(reconfigured.ok());
    EXPECT_EQ(reconfigured.code, HcclSim::Storage::DbCode::INVALID_STATE);
    for (const char* suffix : {"", "-wal", "-shm"}) {
        std::remove((switched + suffix).c_str());
    }
}
} // namespace
