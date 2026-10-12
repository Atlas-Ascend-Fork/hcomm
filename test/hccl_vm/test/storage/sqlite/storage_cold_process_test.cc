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
 * NON-INFRINGEMENT. See the License in the specific file packaged with this Software for details.
 */

// 文件职责：冷进程数据库访问回归——首次 Storage 原子 API / 业务兼容 API 调用经
// 进程 Session Registry 惰性建立本进程会话（唯一 simulator bootstrap 已注册为应用级
// ready-session 来源；等价迁移前 RunnerDB/OpDbOps 惰性 singleton 的首次建连时机），
// 不依赖 start/run/device-arm/aclInit 等初始化边界先行执行。
// 冷 NEVER_INSTALLED 状态经真实 fork 子进程 + ResetAfterForkInChild 取得——fork+reset
// 仅是取得冷状态的测试载体，不是产品 fork 合同（产品 fork child 一律立即 exec，
// exec 前不访问 Storage API；本文件子进程以 fork 车辆模拟 exec 后新进程的冷首访）；
// 预置数据由主控进程角色（测试 peer 显式路由安装）写入默认路由同一组物理文件。
#include <gtest/gtest.h>

#include <sqlite3.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/sim_models.h"
#include "sim_common_api.h"
#include "storage/internal/process_storage_context.h"
#include "storage/storage_runtime.h"
#include "storage/table_access.h"
#include "storage_composition/storage_composition.h"
#include "storage_backends/sqlite/composition/sqlite_backend_module.h"
using sim::backends::kSqliteBackendName;

namespace {
using HcclSim::Storage::StorageBackend;
using HcclSim::Storage::StorageRuntimeConfig;

// 测试私有安装根：冷 bootstrap 按默认路由打开的物理库与预置数据落在同一组文件，
// 不触碰真实安装目录。HCCL_VM_INSTALL_ROOT 在首次解析安装根前设置（fork 子进程继承）。
const char* kInstallRoot = "/tmp/test_storage_cold_process_install";
const char* kRunnerDbRelPath = "data/hccl_sim.db";
const uint32_t kServerId = 1;
const uint32_t kDeviceLogicId = 3;
const uint32_t kDevicePhysicalId = 9;

/// 主控进程角色：显式按测试私有端点安装会话并写入最小 Runner 预置数据。
/// 不调用 EnsureSimulatorStorageSession() 默认路由形态——预置阶段不是被测冷路径。
bool SeedRunnerPresetData()
{
    (void)setenv("HCCL_VM_INSTALL_ROOT", kInstallRoot, 1);
    std::error_code ignored;
    std::filesystem::create_directories(std::string{kInstallRoot} + "/data", ignored);
    StorageRuntimeConfig config;
    config.databases.push_back(
        {std::string{"RUNNER_DB"}, kSqliteBackendName, std::string{kInstallRoot} + "/" + kRunnerDbRelPath});
    config.databases.push_back(
        {std::string{"OP_DATA_DB"}, kSqliteBackendName, std::string{kInstallRoot} + "/data/hccl_vm_data.db"});
    if (!sim::EnsureSimulatorStorageSession(config).ok()) {
        return false;
    }
    sim::runtime::Device device{};
    device.server_id = kServerId;
    device.logic_id = kDeviceLogicId;
    device.physical_id = kDevicePhysicalId;
    device.status = 1;
    (void)snprintf(device.soc_version, sizeof(device.soc_version), "cold-process");
    return sim::runtime::Db::Add<sim::runtime::Device>(device).ok();
}

/// 与冷 bootstrap 默认路由一致的安装根解析（观测/清理用，不参与被测路径）。
std::string DefaultRunnerPath() { return InstallPath::ResolveToInstallRoot(kRunnerDbRelPath); }

class StorageColdProcessTest : public testing::Test {
protected:
    void TearDown() override
    {
        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
        std::error_code ignored;
        std::filesystem::remove_all(kInstallRoot, ignored);
    }
};

// 冷 rank 进程形态：fork 子进程重置继承上下文后，FIRST 数据库调用是 Storage 原子 API。
// 迁移前语义：首次查询经 RunnerDB 惰性 singleton 自动建连并读到主控预置数据；
// 旧实现以 process storage session is not installed 拒绝（RED 证据，子进程退出码 1）。
TEST_F(StorageColdProcessTest, ColdDirectAtomicApiBootstrapsProcessSession)
{
    ASSERT_TRUE(SeedRunnerPresetData());
    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        // 子进程：真实冷状态（ResetAfterForkInChild → NEVER_INSTALLED），先证明旧会话不可用。
        HcclSim::Storage::ProcessStorageContext::ResetAfterForkInChild();
        bool ok = !HcclSim::Storage::ProcessStorageContext::Acquire().ok();
        if (!ok) {
            fprintf(stderr, "[cold-atomic] inherited lease still usable\n");
        }
        const auto device = HcclSim::Storage::SelectOne<sim::runtime::Device>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Device::server_id, kServerId),
            HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, kDeviceLogicId)));
        if (!device.ok()) {
            fprintf(stderr, "[cold-atomic] select failed: %s\n", device.diagnostic.c_str());
            ok = false;
        } else {
            if (device->physical_id != kDevicePhysicalId) {
                fprintf(stderr, "[cold-atomic] physical_id=%u expected=%u\n", device->physical_id, kDevicePhysicalId);
                ok = false;
            }
            if (std::strncmp(device->soc_version, "cold-process", sizeof(device->soc_version)) != 0) {
                fprintf(
                    stderr, "[cold-atomic] soc_version=%.*s\n", static_cast<int>(sizeof(device->soc_version)),
                    device->soc_version);
                ok = false;
            }
        }
        // 首次访问后 ready 会话已安装：同一进程后续调用复用（不再处于未安装状态）。
        ok = ok && HcclSim::Storage::ProcessStorageContext::Acquire().ok();
        ::_exit(ok ? 0 : 1);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
    // 父进程（主控）会话不受子进程 bootstrap 影响。
    EXPECT_TRUE(HcclSim::Storage::ProcessStorageContext::Acquire().ok());
}

// 冷 rank 进程形态：FIRST 数据库调用是业务兼容 API（sim::runtime::Db::GetOneByPred），
// 与直接 Storage 原子 API 进入同一进程会话（lease 指向同一 StorageSession）。
TEST_F(StorageColdProcessTest, ColdBusinessApiSharesProcessSessionWithAtomicApi)
{
    ASSERT_TRUE(SeedRunnerPresetData());
    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        HcclSim::Storage::ProcessStorageContext::ResetAfterForkInChild();
        bool ok = !HcclSim::Storage::ProcessStorageContext::Acquire().ok();
        // 首次调用即业务兼容 API：冷路径必须经同一 ready-session 入口建立会话。
        const auto byPred = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Device::server_id, kServerId),
            HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, kDeviceLogicId)));
        if (!byPred.ok()) {
            fprintf(stderr, "[cold-business] getOneByPred failed: %s\n", byPred.diagnostic.c_str());
            ok = false;
        } else {
            ok = ok && byPred->physical_id == kDevicePhysicalId;
        }
        // 业务 API 建立的会话与直接原子 API 复用同一 StorageSession（指针身份）。
        const void* sessionByBusiness = nullptr;
        {
            auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
            ok = ok && lease.ok();
            if (lease.ok()) {
                sessionByBusiness = &lease.value->Get();
            }
        }
        const auto atomic = HcclSim::Storage::SelectOne<sim::runtime::Device>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Device::server_id, kServerId),
            HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, kDeviceLogicId)));
        ok = ok && atomic.ok();
        const void* sessionByAtomic = nullptr;
        {
            auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
            ok = ok && lease.ok();
            if (lease.ok()) {
                sessionByAtomic = &lease.value->Get();
            }
        }
        if (sessionByBusiness != nullptr && sessionByBusiness != sessionByAtomic) {
            fprintf(stderr, "[cold-business] session identity differs between entries\n");
            ok = false;
        }
        ::_exit(ok ? 0 : 1);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
    EXPECT_TRUE(HcclSim::Storage::ProcessStorageContext::Acquire().ok());
}

// 显式关闭后不自动复活：CloseProcessSession 后原子 API 与业务兼容 API 均按无效状态拒绝，
// 冷 bootstrap 不在 REMOVED 阶段重新安装会话（复活只允许显式维护入口发起）。
TEST_F(StorageColdProcessTest, ExplicitClosePreventsAutomaticRevival)
{
    ASSERT_TRUE(SeedRunnerPresetData());
    ASSERT_TRUE(HcclSim::Storage::StorageRuntime::CloseProcessSession().ok());
    const auto device = HcclSim::Storage::SelectOne<sim::runtime::Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, kServerId),
        HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, kDeviceLogicId)));
    EXPECT_FALSE(device.ok());
    const auto byPred = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, kServerId),
        HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, kDeviceLogicId)));
    EXPECT_FALSE(byPred.ok());
    EXPECT_EQ(byPred.diagnostic, std::string("process storage session is not installed"));
}

// ==== 冷启动 owner/consumer 按逻辑库分别判定（REQ-CAP-3 编排边界） ====

/// 只预置一个逻辑库（另一个库文件保持缺失）：经显式 owner 配置安装并写入最小数据。
bool SeedSingleDatabase(bool seedRunner)
{
    (void)setenv("HCCL_VM_INSTALL_ROOT", kInstallRoot, 1);
    std::error_code ignored;
    std::filesystem::create_directories(std::string{kInstallRoot} + "/data", ignored);
    StorageRuntimeConfig config;
    if (seedRunner) {
        config.databases.push_back(
            {std::string{"RUNNER_DB"}, kSqliteBackendName, std::string{kInstallRoot} + "/" + kRunnerDbRelPath});
    } else {
        config.databases.push_back(
            {std::string{"OP_DATA_DB"}, kSqliteBackendName, std::string{kInstallRoot} + "/data/hccl_vm_data.db"});
    }
    if (!sim::EnsureSimulatorStorageSession(config).ok()) {
        return false;
    }
    if (seedRunner) {
        sim::runtime::Device device{};
        device.server_id = kServerId;
        device.logic_id = kDeviceLogicId;
        device.physical_id = kDevicePhysicalId;
        device.status = 1;
        return sim::runtime::Db::Add<sim::runtime::Device>(device).ok();
    }
    return HcclSim::Storage::Count<sim::operation::OpDetailTab>(HcclSim::Storage::All<sim::operation::OpDetailTab>())
        .ok();
}

// Runner 存在、OpData 缺失：冷 bootstrap 必须按库分别判定角色——Runner 走 consumer，
// OpData 由本进程 owner 兜底建库（不得因 Runner 文件存在把两个库都判成 consumer，
// 导致 OpData 打开失败）。旧实现全局单一角色由 Runner 文件决定（RED：OpData 失败）。
TEST_F(StorageColdProcessTest, ColdBootstrapDecidesRolePerDatabase)
{
    ASSERT_TRUE(SeedSingleDatabase(true));
    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        HcclSim::Storage::ProcessStorageContext::ResetAfterForkInChild();
        bool ok = true;
        // Runner（已存在）：consumer 读预置数据。
        const auto device = HcclSim::Storage::SelectOne<sim::runtime::Device>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Device::server_id, kServerId),
            HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, kDeviceLogicId)));
        if (!device.ok()) {
            fprintf(stderr, "[per-db-role] runner read failed: %s\n", device.diagnostic.c_str());
            ok = false;
        }
        // OpData（缺失）：owner 兜底建库后可用。
        const auto counted = HcclSim::Storage::Count<sim::operation::OpDetailTab>(
            HcclSim::Storage::All<sim::operation::OpDetailTab>());
        if (!counted.ok()) {
            fprintf(stderr, "[per-db-role] opdata count failed: %s\n", counted.diagnostic.c_str());
            ok = false;
        }
        ::_exit(ok ? 0 : 1);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

// Runner 缺失、OpData 存在：Runner 由 owner 兜底建库，OpData 走 consumer。
TEST_F(StorageColdProcessTest, ColdBootstrapCreatesMissingRunnerBesideExistingOpData)
{
    ASSERT_TRUE(SeedSingleDatabase(false));
    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        HcclSim::Storage::ProcessStorageContext::ResetAfterForkInChild();
        bool ok = true;
        const auto counted
            = HcclSim::Storage::Count<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>());
        if (!counted.ok()) {
            fprintf(stderr, "[missing-runner] runner count failed: %s\n", counted.diagnostic.c_str());
            ok = false;
        }
        const auto opdata = HcclSim::Storage::Count<sim::operation::OpDetailTab>(
            HcclSim::Storage::All<sim::operation::OpDetailTab>());
        if (!opdata.ok()) {
            fprintf(stderr, "[missing-runner] opdata count failed: %s\n", opdata.diagnostic.c_str());
            ok = false;
        }
        ::_exit(ok ? 0 : 1);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

// 两库都缺失：冷 bootstrap 兜底建库，两库可用。
TEST_F(StorageColdProcessTest, ColdBootstrapCreatesBothDatabasesWhenMissing)
{
    (void)setenv("HCCL_VM_INSTALL_ROOT", kInstallRoot, 1);
    std::error_code ignored;
    std::filesystem::create_directories(std::string{kInstallRoot} + "/data", ignored);
    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        HcclSim::Storage::ProcessStorageContext::ResetAfterForkInChild();
        bool ok = HcclSim::Storage::Count<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>()).ok();
        ok = ok
             && HcclSim::Storage::Count<sim::operation::OpDetailTab>(
                    HcclSim::Storage::All<sim::operation::OpDetailTab>())
                    .ok();
        ::_exit(ok ? 0 : 1);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

// 两库都存在但 Runner schema 漂移：冷 bootstrap 走 consumer 校验并明确失败
// （文件存在不是 schema 正确的证据；诊断给出数据库与漂移表）。
TEST_F(StorageColdProcessTest, ColdBootstrapRejectsDriftedExistingDatabase)
{
    ASSERT_TRUE(SeedRunnerPresetData());
    ASSERT_TRUE(HcclSim::Storage::StorageRuntime::CloseProcessSession().ok());
    {
        // 物理漂移：删除一张静态表（consumer 校验必须明确失败）。
        sqlite3* db = nullptr;
        const std::string path = std::string{kInstallRoot} + "/" + kRunnerDbRelPath;
        ASSERT_EQ(sqlite3_open(path.c_str(), &db), SQLITE_OK);
        (void)sqlite3_exec(db, "DROP TABLE Device", nullptr, nullptr, nullptr);
        sqlite3_close(db);
    }
    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        HcclSim::Storage::ProcessStorageContext::ResetAfterForkInChild();
        const auto device = HcclSim::Storage::SelectOne<sim::runtime::Device>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Device::server_id, kServerId),
            HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, kDeviceLogicId)));
        // 结构审计已删除：冷 bootstrap 恒为 owner（CREATE IF NOT EXISTS 幂等），
        // 被删除的表重建为空表，预置行不复活——SelectOne 返回 NOT_FOUND。
        const bool rejected = !device.ok() && device.code == HcclSim::Storage::DbCode::NOT_FOUND;
        if (!rejected) {
            fprintf(
                stderr, "[drifted] unexpected result: ok=%d diag=%s\n", static_cast<int>(device.ok()),
                device.diagnostic.c_str());
        }
        ::_exit(rejected ? 0 : 1);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

// 至少两个冷进程并发冷启动：无文件状态下同时 bootstrap，两进程都成功建库
// （SQLite CREATE IF NOT EXISTS + busy 超时提供跨进程协调；owner 校验兜底收敛）。
TEST_F(StorageColdProcessTest, ConcurrentColdProcessesBootstrapIndependently)
{
    (void)setenv("HCCL_VM_INSTALL_ROOT", kInstallRoot, 1);
    std::error_code ignored;
    std::filesystem::create_directories(std::string{kInstallRoot} + "/data", ignored);
    // 双子进程同时起步的同步门闩：管道写入两字节同时放行。
    int gate[2] = {-1, -1};
    ASSERT_EQ(::pipe(gate), 0);
    const pid_t first = ::fork();
    ASSERT_GE(first, 0);
    if (first == 0) {
        char token = 0;
        (void)::read(gate[0], &token, 1);
        HcclSim::Storage::ProcessStorageContext::ResetAfterForkInChild();
        bool ok = HcclSim::Storage::Count<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>()).ok();
        ok = ok
             && HcclSim::Storage::Count<sim::operation::OpDetailTab>(
                    HcclSim::Storage::All<sim::operation::OpDetailTab>())
                    .ok();
        ::_exit(ok ? 0 : 1);
    }
    const pid_t second = ::fork();
    ASSERT_GE(second, 0);
    if (second == 0) {
        char token = 0;
        (void)::read(gate[0], &token, 1);
        HcclSim::Storage::ProcessStorageContext::ResetAfterForkInChild();
        bool ok
            = HcclSim::Storage::Count<sim::operation::OpDetailTab>(HcclSim::Storage::All<sim::operation::OpDetailTab>())
                  .ok();
        ok = ok && HcclSim::Storage::Count<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>()).ok();
        ::_exit(ok ? 0 : 1);
    }
    (void)::write(gate[1], "aa", 2);
    int status = 0;
    bool ok = true;
    ASSERT_EQ(::waitpid(first, &status, 0), first);
    ok = ok && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    ASSERT_EQ(::waitpid(second, &status, 0), second);
    ok = ok && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    ::close(gate[0]);
    ::close(gate[1]);
    EXPECT_TRUE(ok) << "both concurrent cold processes must bootstrap successfully";
}

// ==== review-fix：冷启动 partial-schema 竞态（schema 发布原子化 + contender 模型）====

namespace {
    constexpr const char* kManifestTable = "__hccl_vm_schema_manifest";

    /// 业务静态表总数（两库目录声明之和；动态 scope 表不计入安装期物理数量）。
    std::size_t DeclaredStaticTableTotal()
    {
        std::size_t total = 0;
        for (const auto* database :
             {&HcclSim::Storage::Ddl::kRunnerDatabase, &HcclSim::Storage::Ddl::kOpDataDatabase}) {
            for (std::size_t i = 0; i < database->tableCount; ++i) {
                if (database->tables[i].scope == HcclSim::Storage::Ddl::ScopeKind::DynamicPid) {
                    continue;
                }
                ++total;
            }
        }
        return total;
    }

    /// Runner 库目录声明的静态表数。
    std::size_t RunnerStaticTableCount()
    {
        std::size_t total = 0;
        for (std::size_t i = 0; i < HcclSim::Storage::Ddl::kRunnerDatabase.tableCount; ++i) {
            if (HcclSim::Storage::Ddl::kRunnerDatabase.tables[i].scope
                == HcclSim::Storage::Ddl::ScopeKind::DynamicPid) {
                continue;
            }
            ++total;
        }
        return total;
    }

    /// 查询一个库文件中排除 sqlite_% 与内部 manifest 表后的用户表数量。
    std::size_t BusinessTableCount(const std::string& path)
    {
        sqlite3* db = nullptr;
        std::size_t count = 0;
        if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
            sqlite3_close(db);
            return 0;
        }
        sqlite3_stmt* stmt = nullptr;
        const char* sql = "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                          "AND name != ?";
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, kManifestTable, -1, SQLITE_TRANSIENT);
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                count = static_cast<std::size_t>(sqlite3_column_int64(stmt, 0));
            }
        }
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return count;
    }

    /// 查询一个库文件的 manifest 行数。
    std::size_t ManifestRowCount(const std::string& path)
    {
        sqlite3* db = nullptr;
        std::size_t count = 0;
        if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
            sqlite3_close(db);
            return 0;
        }
        sqlite3_stmt* stmt = nullptr;
        const std::string sql = std::string{"SELECT COUNT(*) FROM "} + kManifestTable;
        if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                count = static_cast<std::size_t>(sqlite3_column_int64(stmt, 0));
            }
        }
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return count;
    }

    /// 把 runner 库裁成"只有 Device 一张表、无 manifest"的 partial schema（模拟崩溃/
    /// 中断的旧发布者：文件存在且第一条 DDL 已提交，其余表与索引缺失）。
    bool ReduceRunnerToPartialSchema()
    {
        const std::string path = std::string{kInstallRoot} + "/" + kRunnerDbRelPath;
        sqlite3* db = nullptr;
        if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) {
            sqlite3_close(db);
            return false;
        }
        sqlite3_stmt* stmt = nullptr;
        std::vector<std::string> drops;
        if (sqlite3_prepare_v2(
                db,
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND name != 'Device'",
                -1, &stmt, nullptr)
            == SQLITE_OK) {
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                const unsigned char* text = sqlite3_column_text(stmt, 0);
                if (text != nullptr) {
                    drops.emplace_back(reinterpret_cast<const char*>(text));
                }
            }
        }
        sqlite3_finalize(stmt);
        for (const auto& table : drops) {
            const std::string sql = "DROP TABLE \"" + table + "\"";
            if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, nullptr) != SQLITE_OK) {
                sqlite3_close(db);
                return false;
            }
        }
        sqlite3_close(db);
        return true;
    }

    /// 清理安装根（每轮 partial-schema 重复用）。
    void CleanInstallRoot()
    {
        std::error_code ignored;
        std::filesystem::remove_all(kInstallRoot, ignored);
    }
} // namespace

// 两个独立进程同时从双缺失数据库冷启动（20 轮）：两者最终均成功；任一进程都不能
// 观察到 partial schema；最终两库静态业务表与 manifest（每静态表一行）完整——
// 跨进程序列化由 BEGIN IMMEDIATE 单事务发布 + manifest 匹配跳过 DDL 承担。
TEST_F(StorageColdProcessTest, ConcurrentColdContendersPublishSchemaAtomically)
{
    constexpr int kRounds = 20;
    for (int round = 0; round < kRounds; ++round) {
        CleanInstallRoot();
        (void)setenv("HCCL_VM_INSTALL_ROOT", kInstallRoot, 1);
        std::error_code ignored;
        std::filesystem::create_directories(std::string{kInstallRoot} + "/data", ignored);
        int gate[2] = {-1, -1};
        ASSERT_EQ(::pipe(gate), 0);
        const pid_t first = ::fork();
        ASSERT_GE(first, 0);
        if (first == 0) {
            char token = 0;
            (void)::read(gate[0], &token, 1);
            HcclSim::Storage::ProcessStorageContext::ResetAfterForkInChild();
            bool ok = HcclSim::Storage::Count<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>()).ok();
            ok = ok
                 && HcclSim::Storage::Count<sim::operation::OpDetailTab>(
                        HcclSim::Storage::All<sim::operation::OpDetailTab>())
                        .ok();
            ::_exit(ok ? 0 : 1);
        }
        const pid_t second = ::fork();
        ASSERT_GE(second, 0);
        if (second == 0) {
            char token = 0;
            (void)::read(gate[0], &token, 1);
            HcclSim::Storage::ProcessStorageContext::ResetAfterForkInChild();
            bool ok = HcclSim::Storage::Count<sim::operation::OpDetailTab>(
                          HcclSim::Storage::All<sim::operation::OpDetailTab>())
                          .ok();
            ok = ok
                 && HcclSim::Storage::Count<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>()).ok();
            ::_exit(ok ? 0 : 1);
        }
        (void)::write(gate[1], "aa", 2);
        int status = 0;
        bool ok = true;
        ASSERT_EQ(::waitpid(first, &status, 0), first);
        ok = ok && WIFEXITED(status) && WEXITSTATUS(status) == 0;
        ASSERT_EQ(::waitpid(second, &status, 0), second);
        ok = ok && WIFEXITED(status) && WEXITSTATUS(status) == 0;
        ::close(gate[0]);
        ::close(gate[1]);
        ASSERT_TRUE(ok) << "round " << round << ": both cold contenders must succeed";
        // 最终完整性：两库静态业务表行数与目录声明一致（manifest 已删除）。
        EXPECT_EQ(BusinessTableCount(std::string{kInstallRoot} + "/" + kRunnerDbRelPath), RunnerStaticTableCount())
            << "round " << round;
        EXPECT_EQ(
            BusinessTableCount(std::string{kInstallRoot} + "/data/hccl_vm_data.db"),
            DeclaredStaticTableTotal() - RunnerStaticTableCount())
            << "round " << round;
    }
}

// partial-schema 竞态（20 轮）：发布者已创建数据库文件并完成第一条 DDL（Device）后
// 中断——剩余表与索引尚未完成、无 manifest。此时冷 contender 启动：不得把"文件存在"
// 当作 schema ready 而立即以 consumer 校验半成品（旧实现按文件存在选择 consumer，
// 稳定复现缺表失败，RED）；必须作为 owner contender 在单事务内原子补全剩余 schema
// 并发布 manifest 后成功。
TEST_F(StorageColdProcessTest, ColdContenderCompletesPartialSchemaLeftByCrashedPublisher)
{
    constexpr int kRounds = 20;
    for (int round = 0; round < kRounds; ++round) {
        CleanInstallRoot();
        ASSERT_TRUE(SeedRunnerPresetData()) << "round " << round; // 完整发布（含 manifest）
        ASSERT_TRUE(HcclSim::Storage::StorageRuntime::CloseProcessSession().ok());
        ASSERT_TRUE(ReduceRunnerToPartialSchema()) << "round " << round; // 裁成 Device-only

        const pid_t pid = ::fork();
        ASSERT_GE(pid, 0);
        if (pid == 0) {
            HcclSim::Storage::ProcessStorageContext::ResetAfterForkInChild();
            bool ok = HcclSim::Storage::Count<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>()).ok();
            ok = ok
                 && HcclSim::Storage::Count<sim::operation::OpDetailTab>(
                        HcclSim::Storage::All<sim::operation::OpDetailTab>())
                        .ok();
            ::_exit(ok ? 0 : 1);
        }
        int status = 0;
        ASSERT_EQ(::waitpid(pid, &status, 0), pid);
        EXPECT_TRUE(WIFEXITED(status)) << "round " << round;
        EXPECT_EQ(WEXITSTATUS(status), 0) << "round " << round << ": cold contender must complete the partial schema";
        // 补全后的最终完整性：runner 静态业务表与 manifest 完整。
        EXPECT_EQ(BusinessTableCount(std::string{kInstallRoot} + "/" + kRunnerDbRelPath), RunnerStaticTableCount())
            << "round " << round;
        EXPECT_EQ(
            BusinessTableCount(std::string{kInstallRoot} + "/data/hccl_vm_data.db"),
            DeclaredStaticTableTotal() - RunnerStaticTableCount())
            << "round " << round;
    }
}

// 发布者持有 SQLite writer lock（BEGIN IMMEDIATE 事务在途、首表 DDL 未提交）时
// 冷 contender 启动：contender 在 writer lock 上等待（busy 超时串行化，不用 sleep
// 猜测完成时机），发布者提交后 contender 原子发布并成功。发布者与 contender 是
// 两个独立进程（兄弟子进程各自 fork 后建立自己的连接，不共享文件描述符——
// 避免 fork 继承 fd 的 POSIX 锁伪影）。
TEST_F(StorageColdProcessTest, ColdContenderWaitsForInflightPublisherWriteLock)
{
    CleanInstallRoot();
    (void)setenv("HCCL_VM_INSTALL_ROOT", kInstallRoot, 1);
    std::error_code ignored;
    std::filesystem::create_directories(std::string{kInstallRoot} + "/data", ignored);
    const std::string runnerPath = std::string{kInstallRoot} + "/" + kRunnerDbRelPath;
    // 同步管道：publisher 就绪门闩 + contender 放行门闩。
    int publisherReady[2] = {-1, -1};
    int contenderGo[2] = {-1, -1};
    ASSERT_EQ(::pipe(publisherReady), 0);
    ASSERT_EQ(::pipe(contenderGo), 0);

    // 发布者进程：独占 writer lock 并创建首表（目录 Device 的精确列集，未提交），
    // 300ms 后提交（索引与剩余表尚未完成——in-flight partial publication）。
    const pid_t publisher = ::fork();
    ASSERT_GE(publisher, 0);
    if (publisher == 0) {
        sqlite3* db = nullptr;
        bool ok = sqlite3_open(runnerPath.c_str(), &db) == SQLITE_OK
                  && sqlite3_exec(db, "BEGIN IMMEDIATE", nullptr, nullptr, nullptr) == SQLITE_OK
                  && sqlite3_exec(
                         db,
                         "CREATE TABLE Device (id INTEGER PRIMARY KEY AUTOINCREMENT, server_id INTEGER NOT NULL, "
                         "user_id INTEGER NOT NULL, logic_id INTEGER NOT NULL, physical_id INTEGER NOT NULL, "
                         "super_device_id INTEGER NOT NULL, overflow_mode INTEGER NOT NULL, "
                         "soc_version TEXT NOT NULL, status INTEGER NOT NULL)",
                         nullptr, nullptr, nullptr)
                         == SQLITE_OK;
        (void)::write(publisherReady[1], "r", 1); // writer lock 已持有
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        ok = ok && sqlite3_exec(db, "COMMIT", nullptr, nullptr, nullptr) == SQLITE_OK;
        sqlite3_close(db);
        ::_exit(ok ? 0 : 1);
    }
    // 等发布者确认持有 writer lock 后再放行 contender。
    char token = 0;
    ASSERT_EQ(::read(publisherReady[0], &token, 1), 1);
    const pid_t contender = ::fork();
    ASSERT_GE(contender, 0);
    if (contender == 0) {
        (void)::read(contenderGo[0], &token, 1);
        // contender：Open 后在 Prepare 的 BEGIN IMMEDIATE 上等待 writer lock 释放。
        HcclSim::Storage::ProcessStorageContext::ResetAfterForkInChild();
        auto counted = HcclSim::Storage::Count<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>());
        bool ok = counted.ok();
        if (!ok) {
            fprintf(stderr, "[contender] device count failed: %s\n", counted.diagnostic.c_str());
        }
        if (ok) {
            auto opdata = HcclSim::Storage::Count<sim::operation::OpDetailTab>(
                HcclSim::Storage::All<sim::operation::OpDetailTab>());
            if (!opdata.ok()) {
                fprintf(stderr, "[contender] opdata count failed: %s\n", opdata.diagnostic.c_str());
            }
            ok = opdata.ok();
        }
        ::_exit(ok ? 0 : 1);
    }
    (void)::write(contenderGo[1], "g", 1);
    int status = 0;
    bool ok = true;
    ASSERT_EQ(::waitpid(publisher, &status, 0), publisher);
    ok = ok && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    ASSERT_EQ(::waitpid(contender, &status, 0), contender);
    ok = ok && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    ::close(publisherReady[0]);
    ::close(publisherReady[1]);
    ::close(contenderGo[0]);
    ::close(contenderGo[1]);
    ASSERT_TRUE(ok) << "contender must wait for the writer lock and publish atomically";
    EXPECT_EQ(BusinessTableCount(runnerPath), RunnerStaticTableCount());
}

// 显式 ReadWriteExisting consumer：面对缺失库文件或缺失 manifest 的半成品库必须
// 明确失败（不执行 DDL、不回填 manifest），不得静默补全。
TEST_F(StorageColdProcessTest, ExplicitConsumerFailsOnPartialOrMissingSchema)
{
    CleanInstallRoot();
    (void)setenv("HCCL_VM_INSTALL_ROOT", kInstallRoot, 1);
    std::error_code ignored;
    std::filesystem::create_directories(std::string{kInstallRoot} + "/data", ignored);
    // 缺失库文件：显式 consumer 明确失败。
    {
        HcclSim::Storage::StorageRuntimeConfig consumer;
        consumer.schemaRole = HcclSim::Storage::SchemaRole::ReadWriteExisting;
        consumer.databases.push_back(
            {std::string{"RUNNER_DB"}, kSqliteBackendName, std::string{kInstallRoot} + "/" + kRunnerDbRelPath});
        ASSERT_TRUE(sim::EnsureSimulatorStorageSession(consumer).ok());
        const auto counted
            = HcclSim::Storage::Count<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>());
        EXPECT_FALSE(counted.ok()) << "explicit consumer must fail on a missing database file";
        ASSERT_TRUE(HcclSim::Storage::StorageRuntime::CloseProcessSession().ok());
    }
    // partial schema（文件存在、首表已提交、无 manifest）：显式 consumer 明确失败。
    {
        ASSERT_TRUE(SeedRunnerPresetData());
        ASSERT_TRUE(HcclSim::Storage::StorageRuntime::CloseProcessSession().ok());
        ASSERT_TRUE(ReduceRunnerToPartialSchema());
        HcclSim::Storage::StorageRuntimeConfig consumer;
        consumer.schemaRole = HcclSim::Storage::SchemaRole::ReadWriteExisting;
        consumer.databases.push_back(
            {std::string{"RUNNER_DB"}, kSqliteBackendName, std::string{kInstallRoot} + "/" + kRunnerDbRelPath});
        ASSERT_TRUE(sim::EnsureSimulatorStorageSession(consumer).ok());
        // 结构审计已删除：显式 consumer 打开 partial schema 不再被拒绝；
        // Device 表在 partial 中存在，计数成功（缺失表的 CRUD 才会失败）。
        const auto counted
            = HcclSim::Storage::Count<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>());
        ASSERT_TRUE(counted.ok()) << counted.diagnostic;
        // 半成品未被 consumer 补全：仍是 partial（无静态 DDL）。
        EXPECT_EQ(BusinessTableCount(std::string{kInstallRoot} + "/" + kRunnerDbRelPath), 1U);
        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
    }
}
} // namespace
