/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：验证 SQLITE_BASELINE 会话的表路由解析：73 类表的数据库归属、后端、
// mapping 绑定与动态 scope 语义；未发布 scope 与未发布表被明确拒绝。
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "storage_composition/storage_composition.h"
#include "storage/storage_session.h"
#include "storage/table_access.h"
#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage_backends/sqlite/composition/sqlite_backend_module.h"

namespace {
using HcclSim::Storage::TableScope;

const char* kRunnerPath = "/tmp/test_sqlite_routing_runner.db";
const char* kOpDataPath = "/tmp/test_sqlite_routing_opdata.db";

class StorageRoutingTest : public testing::Test {
protected:
    void SetUp() override
    {
        CleanFiles();
        ASSERT_TRUE(sim::RegisterSimulatorStorageBackends().ok());
        HcclSim::Storage::StorageRuntimeConfig sessionConfig;
        sessionConfig.databases.push_back(
            {std::string{"RUNNER_DB"}, sim::backends::kSqliteBackendName, std::string{kRunnerPath}});
        sessionConfig.databases.push_back(
            {std::string{"OP_DATA_DB"}, sim::backends::kSqliteBackendName, std::string{kOpDataPath}});
        auto ensured = HcclSim::Storage::StorageRuntime::EnsureProcessSession(sessionConfig);
        ASSERT_TRUE(ensured.ok()) << ensured.diagnostic;
    }

    void TearDown() override
    {
        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
        CleanFiles();
    }

    static void CleanFiles()
    {
        for (const char* path : {kRunnerPath, kOpDataPath}) {
            std::remove(path);
            std::remove((std::string(path) + "-wal").c_str());
            std::remove((std::string(path) + "-shm").c_str());
        }
    }

    /// 经会话表句柄读写一张 Runner 表并核对归属（全表扫描 + 主键读取）。
    void ExerciseRunnerTable()
    {
        auto table = HcclSim::Storage::AcquireTable<sim::runtime::Device>();
        ASSERT_TRUE(table.ok()) << table.diagnostic;
        sim::runtime::Device record{};
        record.server_id = 7;
        record.logic_id = 3;
        auto inserted = table->Insert(record);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
        auto found = table->SelectOne(HcclSim::Storage::Eq(&sim::runtime::Device::id, inserted->value));
        ASSERT_TRUE(found.ok()) << found.diagnostic;
        EXPECT_EQ(found.value->server_id, 7U);
        auto deleted = table->Delete(HcclSim::Storage::Eq(&sim::runtime::Device::id, inserted->value));
        ASSERT_TRUE(deleted.ok());
    }
};

TEST_F(StorageRoutingTest, AllOpDbTablesResolveThroughPublicEntry)
{
    // 公开一次性入口解析全部 OpDB 静态表（路由按目录数据库归属解析）。
    EXPECT_TRUE(HcclSim::Storage::AcquireTable<sim::operation::OpDetailTab>().ok());
    EXPECT_TRUE(HcclSim::Storage::AcquireTable<sim::operation::OpMemInfoTab>().ok());
    EXPECT_TRUE(HcclSim::Storage::AcquireTable<sim::operation::SyncRecordTab>().ok());
    EXPECT_TRUE(HcclSim::Storage::AcquireTable<sim::operation::CcuChannelTab>().ok());
    EXPECT_TRUE(HcclSim::Storage::AcquireTable<sim::operation::JettyMapTab>().ok());
    EXPECT_TRUE(HcclSim::Storage::AcquireTable<sim::operation::CcuInstrResTab>().ok());
    EXPECT_TRUE(HcclSim::Storage::AcquireTable<sim::operation::CcuInstrTab>().ok());
    EXPECT_TRUE(HcclSim::Storage::AcquireTable<sim::operation::HalfRTTTab>().ok());
}

TEST_F(StorageRoutingTest, RunnerTablesResolveThroughTemplateEntrypoints) { ExerciseRunnerTable(); }

TEST_F(StorageRoutingTest, OpTaskResolvesForAnyProcessScope)
{
    // 动态 PID 表按请求 scope 适配（物理表惰性创建）；进程自身 scope 可解析。
    const uint64_t self = static_cast<uint64_t>(::getpid());
    auto tableHandle = HcclSim::Storage::AcquireTable<sim::operation::OpTaskTab>(TableScope::Process(self));
    ASSERT_TRUE(tableHandle.ok()) << tableHandle.diagnostic;
}

TEST_F(StorageRoutingTest, DynamicTableRejectsStaticScope)
{
    // 目录合同：静态请求不命中动态表（保持隔离）；动态表必须显式携带进程 scope。
    auto tableHandle = HcclSim::Storage::AcquireTable<sim::operation::OpTaskTab>(TableScope::Static());
    ASSERT_FALSE(tableHandle.ok());
    EXPECT_EQ(tableHandle.code, HcclSim::Storage::DbCode::NOT_FOUND);
}

TEST_F(StorageRoutingTest, ReconfigureOpDataDatabaseSwitchesFile)
{
    // 换库重建后旧库数据不可见、新库可用（迁移前 SetDbConfig 语义）。
    {
        auto table = HcclSim::Storage::AcquireTable<sim::operation::SyncRecordTab>();
        ASSERT_TRUE(table.ok()) << table.diagnostic;
        sim::operation::SyncRecordTab record{};
        record.pid = 9;
        record.rankId = 9;
        record.rankSize = 1;
        record.syncIter = 1;
        record.streamId = 1;
        record.status = 0;
        auto inserted = table->Insert(record);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    }
    const std::string newPath = "/tmp/test_sqlite_routing_opdata_switched.db";
    auto reconfigured = HcclSim::Storage::StorageRuntime::ReconfigureDatabase(
        {std::string{"OP_DATA_DB"}, sim::backends::kSqliteBackendName, newPath});
    ASSERT_TRUE(reconfigured.ok()) << reconfigured.diagnostic;
    {
        auto table = HcclSim::Storage::AcquireTable<sim::operation::SyncRecordTab>();
        ASSERT_TRUE(table.ok());
        auto rows = table->Select(HcclSim::Storage::All<sim::operation::SyncRecordTab>());
        ASSERT_TRUE(rows.ok()) << rows.diagnostic;
        EXPECT_TRUE(rows.value->empty());
        sim::operation::SyncRecordTab record{};
        record.pid = 9;
        record.rankId = 9;
        record.rankSize = 1;
        record.syncIter = 1;
        record.streamId = 1;
        record.status = 0;
        auto inserted = table->Insert(record);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
        auto all = table->Select(HcclSim::Storage::All<sim::operation::SyncRecordTab>());
        ASSERT_TRUE(all.ok());
        EXPECT_EQ(all.value->size(), 1U);
    }
    for (const char* suffix : {"", "-wal", "-shm"}) {
        std::remove((newPath + suffix).c_str());
    }
}
// ==== 静态路由合同（REQ-CAP-1）：目录是表定义唯一来源，Session 不复制表元数据 ====

/// 断言单个表类型的编译期数据库归属（EXPECT 展开为语句，不能直接进折叠表达式）。
template <typename Table>
void ExpectOneDatabaseOf(std::string_view databaseId)
{
    EXPECT_EQ(HcclSim::Storage::Ddl::DatabaseOf<Table>::kDatabaseId, databaseId);
}

/// 断言一组表类型的编译期数据库归属。
template <typename... Tables>
void ExpectDatabaseOf(std::string_view databaseId)
{
    (ExpectOneDatabaseOf<Tables>(databaseId), ...);
}

TEST_F(StorageRoutingTest, AllRunnerTablesBelongToRunnerDatabase)
{
    // Runner 64 表全部由编译期目录派生归属 RUNNER_DB（不随会话或运行时清单变化；
    // RaSocket 死表已于 2026-09-17 删除）。
    using namespace sim::runtime;
    ExpectDatabaseOf<
        Server, Device, Host, Context, Stream, Task, Event, EventSyncTask, Runner, RunModeConfig, ComputeDie,
        TaskSchedulerDevice, DeviceConnection, DeviceStatus, Link, Port, EndPoint, EndPointPair, EndPointPortMapping,
        VirtualMemBlock, PhyMemBlock, FdMemRecord, FdMemWhiteList, IpcMemRecord, IpcMemWhiteList, IpcNotify,
        IpcNotifyVistorList, MemoryLayout, Notify, NotifyRecordTask, NotifyWaitTask, RaChan, RaContext, RaCr, RaCQ,
        RaCQE, RaDevice, RaJetty, RaJfc, RaLmem, RaMR, RaQP, RaRmem, RaSocketPair, RaTlv, RaTokenId, RaTp, Ccu,
        CcuChannel, SimModelData, Communicator, CommunicatorDestroySync, TopoMetaConfig, Plugin, HcclBuffer, HcclThread,
        HcclChannel, HcommEndpoint, HcommMemReg, HcclEngineCtx, HcclMem, DpuDeviceInfo, DpuPendingNotify>(
        std::string_view{"RUNNER_DB"});
}

TEST_F(StorageRoutingTest, AllOpDataTablesBelongToOpDataDatabase)
{
    // OpData 9 表全部由编译期目录派生归属 OP_DATA_DB。
    using namespace sim::operation;
    ExpectDatabaseOf<
        OpDetailTab, OpMemInfoTab, SyncRecordTab, CcuChannelTab, JettyMapTab, CcuInstrResTab, CcuInstrTab, OpTaskTab,
        HalfRTTTab>(std::string_view{"OP_DATA_DB"});
}

TEST_F(StorageRoutingTest, UnconfiguredDatabaseFailsExplicitly)
{
    // 未配置的逻辑库路由明确失败：表在目录中登记但会话未配置该库时不得静默通过。
    HcclSim::Storage::StorageRuntimeConfig sessionConfig;
    sessionConfig.databases.push_back(
        {std::string{"RUNNER_DB"}, sim::backends::kSqliteBackendName, std::string{kRunnerPath}});
    auto reconfigured = HcclSim::Storage::StorageRuntime::EnsureProcessSession(sessionConfig);
    ASSERT_TRUE(reconfigured.ok()) << reconfigured.diagnostic;
    auto opdataTable = HcclSim::Storage::AcquireTable<sim::operation::SyncRecordTab>();
    ASSERT_FALSE(opdataTable.ok());
    EXPECT_EQ(opdataTable.code, HcclSim::Storage::DbCode::BACKEND_UNAVAILABLE);
}
} // namespace
