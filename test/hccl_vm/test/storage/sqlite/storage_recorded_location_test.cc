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

// 文件职责：历史位置恢复（ReconfigureDatabaseToRecordedLocation）的回调合同
// 回归（评审问题二）。四类裁定形状全部经【真实】公共 composition 消费者与
// 部署快照机制验证：错误传播、成功不换库（外层载荷存在 + 内层空）、成功
// 换库（内层有值）、成功但缺外层载荷（合同违规）。
// 可控裁定来自链接期以【真实 sqlite 模块 + 测试自有 fake 模块】重定义生成的
// compiled_backends 清单（归档成员因符号已定义不被拉入）——公共层经真实
// CompiledBackendModules/部署快照/消费者运行，不是复制消费者代码分支。
// 真实路由保持/重配经回调收到的当前路由捕获断言；另含真实 sqlite 模块的
// 端到端换库用例（记录路径成为活跃库文件）。
#include <sys/stat.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "runtime_state/db_sim_runner_ops.h"
#include "storage_composition/backend_module.h"
#include "storage_composition/compiled_backends.h"
#include "storage_composition/storage_composition.h"
#include "storage/storage_runtime.h"
#include "storage/table_access.h"
#include "storage_backends/sqlite/composition/sqlite_backend_module.h"
using sim::backends::kSqliteBackendName;

namespace {
constexpr char kFakeBackendName[] = "recorded_location_fake";
constexpr char kFakeEndpoint[] = "fake-endpoint-original";

/// fake 模块可控状态：每个用例设置下一次 resolveRecordedLocation 的裁定形状。
enum class Verdict { kError, kOuterMissing, kKeepRoute, kRelocate };
Verdict g_verdict = Verdict::kKeepRoute;
HcclSim::Storage::DbCode g_errorCode = HcclSim::Storage::DbCode::IO_ERROR;
std::string g_relocateEndpoint;
HcclSim::Storage::LogicalDatabaseRuntimeConfig g_capturedRoute{};

HcclSim::Storage::DbResult<sim::backends::BackendDeployment>
ConfigureFakeModule(const sim::backends::BackendConfigureRequest&)
{
    using HcclSim::Storage::DbCode;
    sim::backends::BackendDeployment deployment;
    // fake 模块不提供默认路由/可用引擎：显式路由安装与历史位置裁定不需要
    // 打开引擎（惰性），其余维护回调给出明确错误。
    deployment.makeRoute = [](const sim::backends::BackendRouteRequest&) {
        return HcclSim::Storage::DbResult<HcclSim::Storage::LogicalDatabaseRuntimeConfig>{
            DbCode::UNSUPPORTED_CAPABILITY, std::nullopt, 0, "fake backend provides no default route"};
    };
    deployment.createEngine = [](const sim::backends::BackendEngineRequest&) {
        return HcclSim::Storage::DbResult<std::unique_ptr<HcclSim::Storage::IStorageEngine>>{
            DbCode::BACKEND_UNAVAILABLE, std::nullopt, 0, "fake backend engine is not available"};
    };
    deployment.wipe = [](const sim::backends::BackendEngineRequest&, const std::string&) {
        return HcclSim::Storage::DbResult<void>{DbCode::OK, 0, {}};
    };
    deployment.backup
        = [](const sim::backends::BackendEngineRequest&, const std::string&, const HcclSim::Storage::BackupRequest&) {
              return HcclSim::Storage::DbResult<HcclSim::Storage::BackupArtifact>{
                  DbCode::UNSUPPORTED_CAPABILITY, std::nullopt, 0, "fake backend has no backup"};
          };
    deployment.resolveRecordedLocation
        = [](const HcclSim::Storage::LogicalDatabaseRuntimeConfig& currentRoute, const std::string& recordedPath) {
              (void)recordedPath;
              // 捕获消费者传入的当前路由：用例据此断言真实路由被保持/重配
              g_capturedRoute = currentRoute;
              using Result = HcclSim::Storage::DbResult<std::optional<HcclSim::Storage::LogicalDatabaseRuntimeConfig>>;
              switch (g_verdict) {
                  case Verdict::kError:
                      return Result{g_errorCode, std::nullopt, 0, "fake recorded-location failure"};
                  case Verdict::kOuterMissing:
                      return Result{HcclSim::Storage::DbCode::OK, std::nullopt, 0, {}};
                  case Verdict::kKeepRoute:
                      return Result{
                          HcclSim::Storage::DbCode::OK,
                          std::optional<HcclSim::Storage::LogicalDatabaseRuntimeConfig>{},
                          0,
                          {}};
                  case Verdict::kRelocate:
                      break;
              }
              return Result{
                  HcclSim::Storage::DbCode::OK,
                  std::optional<HcclSim::Storage::LogicalDatabaseRuntimeConfig>{
                      HcclSim::Storage::LogicalDatabaseRuntimeConfig{
                          currentRoute.database, kFakeBackendName, g_relocateEndpoint}},
                  0,
                  {}};
          };
    return {DbCode::OK, std::optional<sim::backends::BackendDeployment>{std::move(deployment)}, 0, {}};
}

const sim::backends::BackendModule kFakeModule{kFakeBackendName, false, ConfigureFakeModule};

/// 链接期重定义生成的已编入模块清单：真实 sqlite 模块 + 测试自有 fake 模块
/// （按注册名排序，与收集器约定一致）。归档内 compiled_backends.cc.o 因符号
/// 已定义不被拉入；公共 composition 全部经真实生成入口发现本清单。
const sim::backends::BackendModule* const kTestModules[] = {&kFakeModule, &sim::backends::SqliteBackendModule()};
} // namespace

namespace sim::backends {
const BackendModule* const* CompiledBackendModules() { return kTestModules; }

std::size_t CompiledBackendCount() { return sizeof(kTestModules) / sizeof(kTestModules[0]); }
} // namespace sim::backends

namespace {
using HcclSim::Storage::DbCode;
using HcclSim::Storage::StorageRuntime;
using HcclSim::Storage::StorageRuntimeConfig;

void RemoveDatabaseFiles(const std::string& path)
{
    std::remove(path.c_str());
    std::remove((path + "-wal").c_str());
    std::remove((path + "-shm").c_str());
}

bool FileExists(const std::string& path)
{
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 && st.st_size > 0;
}

/// 显式 fake 路由会话（惰性：不触表则不调 fake createEngine）。
class RecordedLocationTest : public testing::Test {
protected:
    void SetUp() override
    {
        unsetenv("HCCL_VM_STORAGE_CONFIG");
        g_capturedRoute = {};
        StorageRuntimeConfig config;
        config.databases.push_back({"RUNNER_DB", kFakeBackendName, std::string{kFakeEndpoint}});
        auto ensured = sim::EnsureSimulatorStorageSession(config);
        ASSERT_TRUE(ensured.ok()) << ensured.diagnostic;
    }
    void TearDown() override { (void)StorageRuntime::CloseProcessSession(); }
};

// 错误传播：模块裁定失败 → 错误码/诊断原样传播，实际路由保持不变。
TEST_F(RecordedLocationTest, ErrorVerdictPropagatesAndKeepsRoute)
{
    g_verdict = Verdict::kError;
    g_errorCode = DbCode::IO_ERROR;
    const auto result = sim::ReconfigureDatabaseToRecordedLocation("RUNNER_DB", "/tmp/unused-recorded.db");
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.code, DbCode::IO_ERROR);
    EXPECT_NE(result.diagnostic.find("fake recorded-location failure"), std::string::npos);

    g_verdict = Verdict::kKeepRoute;
    ASSERT_TRUE(sim::ReconfigureDatabaseToRecordedLocation("RUNNER_DB", "/tmp/unused-recorded.db").ok());
    EXPECT_EQ(g_capturedRoute.backend, kFakeBackendName);
    EXPECT_EQ(g_capturedRoute.endpoint, kFakeEndpoint);
}

// 成功不换库（内层空）：返回 OK，实际路由保持。
TEST_F(RecordedLocationTest, KeepRouteVerdictSucceedsWithoutRelocation)
{
    g_verdict = Verdict::kKeepRoute;
    const auto result = sim::ReconfigureDatabaseToRecordedLocation("RUNNER_DB", "/tmp/unused-recorded.db");
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(result.code, DbCode::OK);
    EXPECT_EQ(g_capturedRoute.backend, kFakeBackendName);
    EXPECT_EQ(g_capturedRoute.endpoint, kFakeEndpoint);
}

// 成功但缺外层载荷：回调合同违规 → INVALID_STATE 明确报错（不得当作不换库），
// 实际路由保持。
TEST_F(RecordedLocationTest, OuterMissingVerdictIsContractErrorAndKeepsRoute)
{
    g_verdict = Verdict::kOuterMissing;
    const auto result = sim::ReconfigureDatabaseToRecordedLocation("RUNNER_DB", "/tmp/unused-recorded.db");
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.code, DbCode::INVALID_STATE);
    EXPECT_NE(result.diagnostic.find("relocation verdict"), std::string::npos);

    g_verdict = Verdict::kKeepRoute;
    ASSERT_TRUE(sim::ReconfigureDatabaseToRecordedLocation("RUNNER_DB", "/tmp/unused-recorded.db").ok());
    EXPECT_EQ(g_capturedRoute.endpoint, kFakeEndpoint);
}

// 成功换库（内层有值）：公共层实际重配，路由更新为裁定端点。
TEST_F(RecordedLocationTest, RelocateVerdictReconfiguresRoute)
{
    g_verdict = Verdict::kRelocate;
    g_relocateEndpoint = "fake-endpoint-relocated";
    const auto result = sim::ReconfigureDatabaseToRecordedLocation("RUNNER_DB", "/tmp/unused-recorded.db");
    EXPECT_TRUE(result.ok());

    g_verdict = Verdict::kKeepRoute;
    ASSERT_TRUE(sim::ReconfigureDatabaseToRecordedLocation("RUNNER_DB", "/tmp/unused-recorded.db").ok());
    EXPECT_EQ(g_capturedRoute.backend, kFakeBackendName);
    EXPECT_EQ(g_capturedRoute.endpoint, "fake-endpoint-relocated");
}

// 真实 sqlite 模块端到端：记录路径裁定 → 实际换库，记录路径文件成为活跃库
// （换库后仅新插入的行可见——若路由未切换则两行都在旧库可见）。
TEST(RealSqliteRecordedLocationTest, RecordedPathBecomesActiveDatabase)
{
    const std::string dbA = "/tmp/test_recorded_location_a.db";
    const std::string dbB = "/tmp/test_recorded_location_b.db";
    RemoveDatabaseFiles(dbA);
    RemoveDatabaseFiles(dbB);
    unsetenv("HCCL_VM_STORAGE_CONFIG");

    StorageRuntimeConfig config;
    config.databases.push_back({"RUNNER_DB", std::string{kSqliteBackendName}, dbA});
    ASSERT_TRUE(sim::EnsureSimulatorStorageSession(config).ok());

    sim::runtime::Device first;
    first.server_id = 1;
    first.physical_id = 1;
    ASSERT_TRUE(HcclSim::Storage::Insert(first).ok());
    ASSERT_TRUE(FileExists(dbA)) << "route A should be the active database";

    const auto relocated = sim::ReconfigureDatabaseToRecordedLocation("RUNNER_DB", dbB);
    ASSERT_TRUE(relocated.ok()) << relocated.diagnostic;

    sim::runtime::Device second;
    second.server_id = 2;
    second.physical_id = 2;
    ASSERT_TRUE(HcclSim::Storage::Insert(second).ok());
    ASSERT_TRUE(FileExists(dbB)) << "recorded path should become the active database";
    // 换库生效：活跃库（B）只含换库后插入的一行
    const auto count = HcclSim::Storage::Count<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>());
    ASSERT_TRUE(count.ok());
    EXPECT_EQ(*count.value, 1U);

    (void)StorageRuntime::CloseProcessSession();
    RemoveDatabaseFiles(dbA);
    RemoveDatabaseFiles(dbB);
}
} // namespace
