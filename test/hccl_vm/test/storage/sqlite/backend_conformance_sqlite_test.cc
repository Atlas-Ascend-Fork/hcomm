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
 * @file backend_conformance_sqlite_test.cc
 * @brief 用 SQLiteEngine 实例化全部公共 backend conformance case。
 *
 * fixture 协议：临时 SQLite 文件 + 统一 StorageCatalog 的 conformance 目录 +
 * 强类型连接 profile 经 SQLite 通用工厂创建引擎，并经 StorageRuntime 工厂注册
 * 装配进程会话。公共 case 与期望全部来自 backend_conformance.h 单一来源，
 * 不为 SQLite 修改；SQLite 专有行为（SQL 渲染、PRAGMA、文件备份、fault
 * injection）留在 test_sqlite_engine* 私有测试。
 */
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "backend_conformance.h"
#include "storage_backends/sqlite/engine/sqlite_backend_factory.h"
#include "storage_backends/sqlite/engine/sqlite_engine.h"
#include "storage/internal/storage_runtime_impl.h"
#include "storage/storage_runtime.h"
#include "storage_backends/sqlite/composition/sqlite_backend_module.h"
using sim::backends::kSqliteBackendName;

namespace storage_conformance {

namespace {
    /// 每个用例一个全新临时 SQLite 文件（用例结束删除；失败保留供诊断）
    std::string& ActiveDatabasePath()
    {
        static std::string path;
        return path;
    }

    std::string NewTempDatabasePath()
    {
        static uint64_t counter = 0;
        return ::testing::TempDir() + "backend_conformance_sqlite_" + std::to_string(++counter) + ".db";
    }
} // namespace

/// fixture 会话安装：SQLite 通用引擎工厂（目录 + 路径 + profile）+ 进程会话装配
bool OpenConformanceSession()
{
    ActiveDatabasePath() = NewTempDatabasePath();
    auto& runtime = HcclSim::Storage::detail::StorageRuntimeImpl::Instance();
    runtime.ResetForTest();
    runtime.RegisterEngineFactory(
        sim::backends::kSqliteBackendName,
        [](const HcclSim::Storage::LogicalDatabaseRuntimeConfig& config)
            -> HcclSim::Storage::DbResult<std::unique_ptr<HcclSim::Storage::IStorageEngine>> {
            // 引擎装配的唯一入口：表定义全部由统一目录派生（单一事实源），
            // 连接参数来自强类型 profile（工厂与引擎都不识别业务库名）。
            return HcclSim::Storage::CreateSqliteEngine(
                kConformanceDatabase, config.endpoint, HcclSim::Storage::SqliteConnectionProfile{10000, false, 200});
        });
    runtime.UseDatabase(kConformanceDatabase);
    const HcclSim::Storage::StorageRuntimeConfig config{
        {{"CONFORMANCE_DB", sim::backends::kSqliteBackendName, ActiveDatabasePath()}}};
    const auto ensured = HcclSim::Storage::StorageRuntime::EnsureProcessSession(config);
    return ensured.ok();
}

/// fixture 会话关闭：排空并关闭进程会话，清理装配登记与临时文件
bool CloseConformanceSession()
{
    const auto closed = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    HcclSim::Storage::detail::StorageRuntimeImpl::Instance().ResetForTest();
    if (!ActiveDatabasePath().empty()) {
        (void)std::remove(ActiveDatabasePath().c_str());
        ActiveDatabasePath().clear();
    }
    return closed.ok();
}
} // namespace storage_conformance

HCCL_VM_ALL_CONFORMANCE_CASES()
// 并发原子减法（真实后端专用清单；Fake 引擎无跨线程锁不实例化）
HCCL_VM_CONCURRENT_INCREMENT_CONFORMANCE_CASES()

// 通用工厂装配合同：同一 CreateSqliteEngine 实现可从任意目录描述符创建引擎，
// 不识别业务库名（不存在按库名分支的引擎创建）。canonical conformance 目录在此被
// 登记为任意命名的逻辑库并完整走一遍会话安装 + 公开原子 Insert/SelectOne；
// 若工厂含业务库名 if-chain，该装配与读写往返将失败。
namespace {
TEST(SqliteBackendFactoryTest, AssemblesEngineFromAnyCatalogDescriptor)
{
    using namespace HcclSim::Storage;
    using storage_conformance::Device;

    // 任意命名的逻辑库描述符：复用 conformance 目录的表闭包，仅替换逻辑库名。
    const Ddl::DatabaseDescriptor renamedDatabase{
        std::string_view{"ARBITRARY_LOGICAL_DB"}, Ddl::kConformanceDatabase.tables,
        Ddl::kConformanceDatabase.tableCount};

    const std::string databasePath = ::testing::TempDir() + "sqlite_backend_factory_arbitrary.db";
    auto& runtime = HcclSim::Storage::detail::StorageRuntimeImpl::Instance();
    runtime.ResetForTest();
    runtime.RegisterEngineFactory(
        kSqliteBackendName,
        [renamedDatabase](const LogicalDatabaseRuntimeConfig& config) -> DbResult<std::unique_ptr<IStorageEngine>> {
            return HcclSim::Storage::CreateSqliteEngine(
                renamedDatabase, config.endpoint, HcclSim::Storage::SqliteConnectionProfile{10000, false, 200});
        });
    // 会话装配按表类型的编译期归属（CONFORMANCE_DB）登记原名描述符；工厂内部仍从
    // 任意命名的描述符副本派生 mapping/适配器——若工厂含业务库名 if-chain，
    // 该装配与读写往返将失败。
    runtime.UseDatabase(Ddl::kConformanceDatabase);
    const StorageRuntimeConfig config{{{std::string{"CONFORMANCE_DB"}, kSqliteBackendName, databasePath}}};
    ASSERT_TRUE(StorageRuntime::EnsureProcessSession(config).ok());

    // 端到端：公开原子接口经该引擎完成自增主键写入与单行读回。
    Device device;
    device.server_id = 3;
    device.status = 5;
    ASSERT_TRUE(Insert(device).ok());
    EXPECT_GT(device.id, 0U);
    const auto one = SelectOne<Device>(Eq(&Device::id, device.id));
    ASSERT_TRUE(one.ok());
    ASSERT_TRUE(one.value.has_value());
    EXPECT_EQ(one.value->server_id, 3U);
    EXPECT_EQ(one.value->status, 5U);

    ASSERT_TRUE(StorageRuntime::CloseProcessSession().ok());
    runtime.ResetForTest();
    (void)std::remove(databasePath.c_str());
}

// 工厂直连合同：CreateSqliteEngine 以描述符为全部输入产出引擎实例，
// 不经过任何全局状态（同一描述符可创建多个独立引擎）。
TEST(SqliteBackendFactoryTest, CreatesEngineDirectlyFromDescriptor)
{
    using namespace HcclSim::Storage;
    const std::string firstPath = ::testing::TempDir() + "sqlite_backend_factory_direct_a.db";
    const std::string secondPath = ::testing::TempDir() + "sqlite_backend_factory_direct_b.db";
    const LogicalDatabaseRuntimeConfig first{"CONFORMANCE_DB", kSqliteBackendName, firstPath};
    const LogicalDatabaseRuntimeConfig second{"CONFORMANCE_DB", kSqliteBackendName, secondPath};
    const auto firstEngine = HcclSim::Storage::CreateSqliteEngine(
        Ddl::kConformanceDatabase, first.endpoint, HcclSim::Storage::SqliteConnectionProfile{10000, false, 200});
    const auto secondEngine = HcclSim::Storage::CreateSqliteEngine(
        Ddl::kConformanceDatabase, second.endpoint, HcclSim::Storage::SqliteConnectionProfile{10000, false, 200});
    ASSERT_TRUE(firstEngine.ok());
    ASSERT_TRUE(secondEngine.ok());
    ASSERT_TRUE(firstEngine.value.has_value());
    ASSERT_TRUE(secondEngine.value.has_value());
    EXPECT_NE(firstEngine.value->get(), secondEngine.value->get());
    (void)std::remove(firstPath.c_str());
    (void)std::remove(secondPath.c_str());
}
} // namespace
