/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：SQLITE_BASELINE 物理清单与执行计划验证。
// 物理表/索引全部由统一 StorageCatalog 派生（canonical 表名/列名），期望清单
// 从两个业务目录描述符推导，不维护第二份手写表/索引清单；每个固定索引的
// 代表查询用真实 SQLite EXPLAIN QUERY PLAN 断言 SEARCH + USING INDEX，
// 不出现 SCAN 或 TEMP B-TREE，也不写死自动生成的物理索引名。
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include <sqlite3.h>

#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage_composition/storage_composition.h"
#include "storage/storage_runtime.h"
#include "storage/table_access.h"
#include "storage_backends/sqlite/composition/sqlite_backend_module.h"

namespace {
const char* kRunnerPath = "/tmp/test_baseline_inventory_runner.db";
const char* kOpDataPath = "/tmp/test_baseline_inventory_opdata.db";
const char* kSchemaManifestTable = "__hccl_vm_schema_manifest";

/// 查询 sqlite_master 指定 type 的全部名称（排除 sqlite_ 系统对象与引擎内部
/// schema manifest 簿记表——后者不是业务表）。
std::vector<std::string> QueryMaster(const std::string& path, const char* type)
{
    sqlite3* db = nullptr;
    std::vector<std::string> names;
    if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        return names;
    }
    sqlite3_stmt* stmt = nullptr;
    const std::string sql = std::string{"SELECT name FROM sqlite_master WHERE type = ? AND name NOT LIKE "}
                            + "'sqlite_%' AND name != ? ORDER BY name";
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, type, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, kSchemaManifestTable, -1, SQLITE_TRANSIENT);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const unsigned char* text = sqlite3_column_text(stmt, 0);
            if (text != nullptr) {
                names.emplace_back(reinterpret_cast<const char*>(text));
            }
        }
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return names;
}

/// 运行 EXPLAIN QUERY PLAN 并拼接全部计划行。
std::string ExplainQueryPlan(const std::string& path, const std::string& sql)
{
    sqlite3* db = nullptr;
    std::string plan;
    if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        return "open-failed";
    }
    sqlite3_stmt* stmt = nullptr;
    const std::string explain = "EXPLAIN QUERY PLAN " + sql;
    if (sqlite3_prepare_v2(db, explain.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const unsigned char* text = sqlite3_column_text(stmt, 3);
            if (text != nullptr) {
                plan += reinterpret_cast<const char*>(text);
                plan += "\n";
            }
        }
    } else {
        plan = "prepare-failed: ";
        plan += sqlite3_errmsg(db);
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return plan;
}

/// 按列序查找 VirtualMemBlock 的物理索引名：PRAGMA index_list/index_info 的
/// 语义匹配，不耦合 CREATE 语句文本格式，也不写死自动生成的物理索引名。
/// 返回空串 = 无该列序的索引。
std::string VmbIndexNameByColumns(const std::string& path, const std::vector<std::string>& columns)
{
    sqlite3* db = nullptr;
    std::string found;
    if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        return found;
    }
    sqlite3_stmt* list = nullptr;
    if (sqlite3_prepare_v2(db, "PRAGMA index_list('VirtualMemBlock')", -1, &list, nullptr) == SQLITE_OK) {
        while (found.empty() && sqlite3_step(list) == SQLITE_ROW) {
            const unsigned char* nameText = sqlite3_column_text(list, 1);
            if (nameText == nullptr) {
                continue;
            }
            const std::string name{reinterpret_cast<const char*>(nameText)};
            sqlite3_stmt* info = nullptr;
            if (sqlite3_prepare_v2(db, ("PRAGMA index_info(" + name + ")").c_str(), -1, &info, nullptr) != SQLITE_OK) {
                continue;
            }
            std::vector<std::string> actual;
            while (sqlite3_step(info) == SQLITE_ROW) {
                const unsigned char* columnText = sqlite3_column_text(info, 2);
                if (columnText != nullptr) {
                    actual.emplace_back(reinterpret_cast<const char*>(columnText));
                }
            }
            sqlite3_finalize(info);
            if (actual == columns) {
                found = name;
            }
        }
    }
    sqlite3_finalize(list);
    sqlite3_close(db);
    return found;
}

/// 直接执行一条写侧 SQL（DROP INDEX 等；空返回 = 成功）。
std::string ExecRawSql(const std::string& path, const std::string& sql)
{
    sqlite3* db = nullptr;
    if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READWRITE, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        return "open-failed";
    }
    char* error = nullptr;
    const int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &error);
    std::string message;
    if (rc != SQLITE_OK) {
        message = error != nullptr ? error : "exec-failed";
    }
    sqlite3_free(error);
    sqlite3_close(db);
    return message;
}

/// 单个固定索引代表查询的计划断言：SEARCH + USING（COVERING）INDEX + 期望列组合，
/// 不允许 SCAN 或 TEMP B-TREE；不写死自动生成的物理索引名。
void ExpectIndexSearch(const std::string& path, const std::string& sql, const std::string& expectedColumns)
{
    const std::string plan = ExplainQueryPlan(path, sql);
    ASSERT_NE(plan.find("SEARCH"), std::string::npos) << sql << "\nplan: " << plan;
    // 索引访问形态：普通索引或覆盖索引（COVERING）都满足无 SCAN/无临时排序要求。
    const bool usesIndex
        = plan.find("USING INDEX") != std::string::npos || plan.find("USING COVERING INDEX") != std::string::npos;
    EXPECT_TRUE(usesIndex) << sql << "\nplan: " << plan;
    if (!expectedColumns.empty()) {
        EXPECT_NE(plan.find(expectedColumns), std::string::npos) << sql << "\nplan: " << plan;
    }
    EXPECT_EQ(plan.find("SCAN"), std::string::npos) << sql << "\nplan: " << plan;
    EXPECT_EQ(plan.find("TEMP B-TREE"), std::string::npos) << sql << "\nplan: " << plan;
}

/// 目录中声明的静态表索引总数（动态 scope 表的索引随物理 scope 表惰性创建，
/// 不计入安装期的物理索引数量）。
std::size_t DeclaredStaticIndexCount(const HcclSim::Storage::Ddl::DatabaseDescriptor& database)
{
    std::size_t count = 0;
    for (std::size_t i = 0; i < database.tableCount; ++i) {
        if (database.tables[i].scope == HcclSim::Storage::Ddl::ScopeKind::DynamicPid) {
            continue;
        }
        count += database.tables[i].indexCount;
    }
    return count;
}

class SqliteBaselineInventoryTest : public testing::Test {
protected:
    void SetUp() override
    {
        CleanFiles();
        auto registered = sim::RegisterSimulatorStorageBackends();
        ASSERT_TRUE(registered.ok()) << registered.diagnostic;
        const HcclSim::Storage::StorageRuntimeConfig config{
            {{std::string{"RUNNER_DB"}, sim::backends::kSqliteBackendName, std::string{kRunnerPath}},
             {std::string{"OP_DATA_DB"}, sim::backends::kSqliteBackendName, std::string{kOpDataPath}}}};
        auto ensured = HcclSim::Storage::StorageRuntime::EnsureProcessSession(config);
        ASSERT_TRUE(ensured.ok()) << ensured.diagnostic;
        // schema owner（OwnerPrepare 装配）的静态 DDL 在每库首次真实访问时物化：
        // 本测试断言安装后的物理 schema，故对两库各执行一次访问触发建库。
        ASSERT_TRUE(HcclSim::Storage::AcquireTable<sim::runtime::Device>().ok());
        ASSERT_TRUE(HcclSim::Storage::AcquireTable<sim::operation::OpDetailTab>().ok());
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
};

// Runner 物理表集合与业务目录逐项一致（canonical 表名，无历史别名）。
TEST_F(SqliteBaselineInventoryTest, RunnerTablesMatchCatalogExactly)
{
    const auto tables = QueryMaster(kRunnerPath, "table");
    ASSERT_EQ(tables.size(), HcclSim::Storage::Ddl::kRunnerDatabase.tableCount);
    for (std::size_t i = 0; i < HcclSim::Storage::Ddl::kRunnerDatabase.tableCount; ++i) {
        EXPECT_NE(
            std::find(tables.begin(), tables.end(), HcclSim::Storage::Ddl::kRunnerDatabase.tables[i].tableId),
            tables.end())
            << HcclSim::Storage::Ddl::kRunnerDatabase.tables[i].tableId;
    }
    // 历史物理别名不再存在（canonical 名为唯一标准）。
    EXPECT_EQ(std::find(tables.begin(), tables.end(), "VirMem"), tables.end());
}

// OpData 静态物理表（动态 opTask 表按 scope 惰性创建，不在安装期出现）。
TEST_F(SqliteBaselineInventoryTest, OpDataStaticTablesMatchCatalog)
{
    const auto tables = QueryMaster(kOpDataPath, "table");
    std::size_t staticCount = 0;
    for (std::size_t i = 0; i < HcclSim::Storage::Ddl::kOpDataDatabase.tableCount; ++i) {
        if (HcclSim::Storage::Ddl::kOpDataDatabase.tables[i].scope == HcclSim::Storage::Ddl::ScopeKind::Static) {
            ++staticCount;
            EXPECT_NE(
                std::find(tables.begin(), tables.end(), HcclSim::Storage::Ddl::kOpDataDatabase.tables[i].tableId),
                tables.end())
                << HcclSim::Storage::Ddl::kOpDataDatabase.tables[i].tableId;
        }
    }
    EXPECT_EQ(tables.size(), staticCount);
    EXPECT_EQ(std::find(tables.begin(), tables.end(), "opDetails"), tables.end()); // 旧别名不存在
}

// 物理索引数量与目录声明一致（无第二份索引清单：期望值由 catalog 派生）。
TEST_F(SqliteBaselineInventoryTest, PhysicalIndexCountMatchesCatalog)
{
    const auto runnerIndexes = QueryMaster(kRunnerPath, "index");
    EXPECT_EQ(runnerIndexes.size(), DeclaredStaticIndexCount(HcclSim::Storage::Ddl::kRunnerDatabase));
    const auto opDataIndexes = QueryMaster(kOpDataPath, "index");
    EXPECT_EQ(opDataIndexes.size(), DeclaredStaticIndexCount(HcclSim::Storage::Ddl::kOpDataDatabase));
}

// 2026-09-20 VMB purge 复合索引：列序 (device_id, is_freed) 的物理索引存在，
// 且派生名与命名公式一致（idx_<tableId>_<col1>_<col2>；与 MariaDB 版
// DerivedIndexName 同构——两后端共用同一 DDL 派生命名，不硬编码第二份清单）
TEST_F(SqliteBaselineInventoryTest, VmbPurgeIndexPresentWithDerivedName)
{
    const std::string name = VmbIndexNameByColumns(kRunnerPath, {"device_id", "is_freed"});
    EXPECT_EQ(name, "idx_VirtualMemBlock_device_id_is_freed") << "purge 索引缺失或派生名与公式不符";
}

// 固定索引的代表查询计划：全部 SEARCH + USING INDEX，无 SCAN/TEMP B-TREE。
// 覆盖规范 8.6.2 的全部固定索引（22 个新增/修正项 + 保留项），按列组合断言。
TEST_F(SqliteBaselineInventoryTest, FixedIndexQueriesUseIndexSearch)
{
    const std::string runner{kRunnerPath};
    const std::string opdata{kOpDataPath};
    // 动态表按 scope 惰性创建：计划断言前先写入一条记录建出 OpTaskTab_P_1。
    sim::operation::OpTaskTab setupTask{};
    setupTask.opDetailId = 1;
    setupTask.optaskMeta = {0x01};
    ASSERT_TRUE(
        HcclSim::Storage::Insert<sim::operation::OpTaskTab>(HcclSim::Storage::TableScope::Process(1), setupTask).ok());
    // ---- Runner 固定索引 ----
    ExpectIndexSearch(
        runner, "SELECT * FROM Device WHERE server_id = 1 AND physical_id = 2", "server_id=? AND physical_id=?");
    ExpectIndexSearch(runner, "SELECT * FROM Device WHERE physical_id = 2", "physical_id=?");
    ExpectIndexSearch(
        runner,
        "SELECT * FROM RaSocketPair WHERE client_id = 1 AND server_id = 2 AND tag_hash = 3 "
        "AND port = 4",
        "client_id=? AND server_id=? AND tag_hash=? AND port=?");
    ExpectIndexSearch(
        runner,
        "SELECT * FROM RaSocketPair WHERE client_id = 1 AND server_id = 2 AND tag_hash = 3 "
        "AND buf_status = 0",
        "client_id=? AND server_id=? AND tag_hash=? AND buf_status=?");
    ExpectIndexSearch(
        runner,
        "SELECT * FROM VirtualMemBlock WHERE device_id = 1 AND src_type = 0 AND "
        "dev_mapped_ptr >= 4096",
        "device_id=? AND src_type=? AND dev_mapped_ptr");
    ExpectIndexSearch(
        runner,
        "SELECT * FROM VirtualMemBlock WHERE rank_id = 1 AND src_type = 0 AND "
        "dev_mapped_ptr >= 4096",
        "rank_id=? AND src_type=? AND dev_mapped_ptr");
    ExpectIndexSearch(
        runner,
        "SELECT * FROM VirtualMemBlock WHERE device_id = 1 AND src_type = 0 AND "
        "start_ptr = 4096",
        "device_id=? AND src_type=? AND start_ptr=?");
    ExpectIndexSearch(
        runner, "SELECT * FROM VirtualMemBlock WHERE src_type = 0 AND start_ptr = 4096", "src_type=? AND start_ptr=?");
    ExpectIndexSearch(runner, "SELECT * FROM VirtualMemBlock WHERE dev_mapped_ptr >= 4096", "dev_mapped_ptr");
    ExpectIndexSearch(runner, "SELECT * FROM VirtualMemBlock WHERE start_ptr >= 4096", "start_ptr");
    // 2026-09-13 B 轮新增联合索引（src_type, dev_mapped_ptr, dev_mapped_end_ptr）：
    // IsDeviceAddress 包含查询形状（无 device_id/rank_id 前导）的代表计划
    ExpectIndexSearch(
        runner, "SELECT * FROM VirtualMemBlock WHERE src_type = 0 AND dev_mapped_ptr >= 4096",
        "src_type=? AND dev_mapped_ptr");
    ExpectIndexSearch(
        runner,
        "SELECT * FROM VirtualMemBlock WHERE src_type = 0 AND dev_mapped_ptr <= 6144 "
        "AND dev_mapped_end_ptr > 4096",
        "src_type=? AND dev_mapped_ptr");
    // 2026-09-20 VMB purge 复合索引（device_id, is_freed）：释放标记行的插入前
    // 物理清除（aclrt_memory_stub.cc 的 Delete And(Eq(device_id), Eq(is_freed,1))）
    // 的 DELETE 计划必须走该复合索引（SEARCH，不允许全表 SCAN）
    ExpectIndexSearch(
        runner, "DELETE FROM VirtualMemBlock WHERE device_id = 1 AND is_freed = 1", "device_id=? AND is_freed=?");
    ExpectIndexSearch(runner, "SELECT * FROM Server WHERE pod_id = 1 AND server_id = 2", "pod_id=? AND server_id=?");
    ExpectIndexSearch(runner, "SELECT * FROM Host WHERE server_id = 2", "server_id=?");
    ExpectIndexSearch(
        runner, "SELECT * FROM DeviceConnection WHERE src_dev_id = 1 AND dst_dev_id = 2",
        "src_dev_id=? AND dst_dev_id=?");
    ExpectIndexSearch(
        runner, "SELECT * FROM Stream WHERE ctx_id = 1 AND is_primary_default = 1",
        "ctx_id=? AND is_primary_default=?");
    ExpectIndexSearch(runner, "SELECT COUNT(*) FROM Stream WHERE ctx_id = 1", "ctx_id=?");
    ExpectIndexSearch(
        runner, "SELECT * FROM TaskSchedulerDevice WHERE device_id = 1 AND type = 2", "device_id=? AND type=?");
    ExpectIndexSearch(runner, "SELECT * FROM ComputeDie WHERE ts_id = 1 AND type = 2", "ts_id=? AND type=?");
    ExpectIndexSearch(runner, "SELECT * FROM Ccu WHERE device_id = 1 AND die_id = 2", "device_id=? AND die_id=?");
    ExpectIndexSearch(
        runner, "SELECT * FROM EndPointPair WHERE local_enpoint_id = 1 AND remote_enpoint_id = 2",
        "local_enpoint_id=? AND remote_enpoint_id=?");
    // ---- OpData 固定索引 ----
    ExpectIndexSearch(
        opdata, "SELECT * FROM OpDetailTab WHERE commId = 1 AND deviceId = 2 ORDER BY id", "commId=? AND deviceId=?");
    ExpectIndexSearch(opdata, "SELECT * FROM OpDetailTab WHERE pid = 1 AND syncIter = 2", "pid=? AND syncIter=?");
    ExpectIndexSearch(opdata, "SELECT * FROM OpDetailTab WHERE pid = 1", "pid=?");
    // syncIter 单列索引：runner 每 sync 装载与 checker 装载的
    // Select<OpDetailTab>(Eq(syncIter)) 形状（(pid,syncIter) 左前缀 pid 用不上）
    ExpectIndexSearch(opdata, "SELECT * FROM OpDetailTab WHERE syncIter = 2", "syncIter=?");
    ExpectIndexSearch(
        opdata, "SELECT * FROM CcuInstrResTab WHERE deviceId = 1 AND dieId = 2", "deviceId=? AND dieId=?");
    ExpectIndexSearch(opdata, "SELECT * FROM OpTaskTab_P_1 WHERE opDetailId = 1 ORDER BY id", "opDetailId=?");
    ExpectIndexSearch(opdata, "SELECT * FROM OpMemInfoTab WHERE opDetailId = 1", "opDetailId=?");
    ExpectIndexSearch(opdata, "SELECT * FROM SyncRecordTab WHERE status = 0", "status=?");
}

// A1（2026-09-24）：OpTask (streamId, isDone) 覆盖索引——任务完成度聚合
// （QueryOpTaskCompletionByStreamId / CountMatches：streamId 过滤 + isDone
// 匹配计数，不加载任务行）只消费索引列，计划必须是 SEARCH + COVERING INDEX；
// 只按 isDone 过滤的查询（QueryAllOpTasks filterDone 形态）仍走独立 isDone 索引。
TEST_F(SqliteBaselineInventoryTest, OpTaskCompletionAggregateUsesCoveringStreamIndex)
{
    const std::string opdata{kOpDataPath};
    // 动态表按 scope 惰性创建：多 stream、混合完成态的代表性数据
    const auto seedTask = [](uint64_t streamId, bool done) {
        sim::operation::OpTaskTab task{};
        task.streamId = streamId;
        task.isDone = done;
        task.optaskMeta = {0x01, 0x02};
        const auto inserted
            = HcclSim::Storage::Insert<sim::operation::OpTaskTab>(HcclSim::Storage::TableScope::Process(1), task);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    };
    seedTask(11U, false);
    seedTask(11U, true);
    seedTask(12U, true);

    // 生产聚合 SQL 形态（AppendCountMatchesAggregate 的渲染结果）
    const std::string aggregate = "SELECT COUNT(*), COUNT(CASE WHEN (isDone <> 0) THEN 1 END) "
                                  "FROM OpTaskTab_P_1 WHERE streamId = 11";
    ExpectIndexSearch(opdata, aggregate, "streamId=?");
    // 覆盖读取证明：聚合只消费 streamId/isDone 两列，不回表取任务行
    const std::string plan = ExplainQueryPlan(opdata, aggregate);
    EXPECT_NE(plan.find("USING COVERING INDEX"), std::string::npos) << plan;
    // 独立 isDone 索引仍在且服务只按完成态过滤的查询
    ExpectIndexSearch(opdata, "SELECT * FROM OpTaskTab_P_1 WHERE isDone = 0", "isDone=?");
}

// A2（2026-09-24）：OpDetail (deviceId, id) 索引——设备维度最新算子定位
// （QueryLatestOpDetailStreamByDeviceId：deviceId 过滤 + id 倒序取首条）经该
// 索引等值定位 + 倒序取首行（无 TEMP B-TREE 排序）；id 在索引内，streamId 回表。
TEST_F(SqliteBaselineInventoryTest, LatestOpDetailStreamByDeviceUsesDeviceIdIndex)
{
    const std::string opdata{kOpDataPath};
    for (uint32_t device = 0; device < 4U; ++device) {
        sim::operation::OpDetailTab detail{};
        detail.pid = 1;
        detail.deviceId = device;
        detail.rankId = 0;
        detail.streamId = 900U + device;
        detail.opDetail = {0x01};
        const auto inserted = HcclSim::Storage::Insert<sim::operation::OpDetailTab>(detail);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    }
    ExpectIndexSearch(
        opdata, "SELECT id, streamId FROM OpDetailTab WHERE deviceId = 2 ORDER BY id DESC LIMIT 1", "deviceId=?");
}

// 动态 opTask 表：插入按 scope 惰性建表，销毁动态数据后物理表消失。
TEST_F(SqliteBaselineInventoryTest, OpTaskDynamicTableLifecycle)
{
    sim::operation::OpTaskTab task{};
    task.opDetailId = 1;
    task.optaskMeta = {0x11, 0x22};
    ASSERT_TRUE(
        HcclSim::Storage::Insert<sim::operation::OpTaskTab>(HcclSim::Storage::TableScope::Process(111), task).ok());
    const auto tables = QueryMaster(kOpDataPath, "table");
    EXPECT_NE(std::find(tables.begin(), tables.end(), "OpTaskTab_P_111"), tables.end());

    auto dropped = HcclSim::Storage::StorageRuntime::DropDynamicData("OP_DATA_DB");
    ASSERT_TRUE(dropped.ok()) << dropped.diagnostic;
    EXPECT_EQ(*dropped.value, 1U);
    const auto after = QueryMaster(kOpDataPath, "table");
    EXPECT_EQ(std::find(after.begin(), after.end(), "OpTaskTab_P_111"), after.end());
}

// 新增普通联合索引（src_type, dev_mapped_ptr, dev_mapped_end_ptr——区间包含
// 查询形状的代表索引）的旧库补建合同：旧二进制发布的库文件缺该索引（等价于
// 直接 DROP INDEX 模拟），重新打开后齐备性预检发现缺失 → 原子单事务
// CREATE INDEX IF NOT EXISTS 补建并由 SQLite 对全部既有行回填；补建后计划
// 选中该索引，包含查询命中既有行且结果正确。
TEST_F(SqliteBaselineInventoryTest, NewCompositeIndexBackfilledOnReopen)
{
    using sim::runtime::VirtualMemBlock;
    const std::vector<std::string> kNewIndexColumns{"src_type", "dev_mapped_ptr", "dev_mapped_end_ptr"};

    // 现场形状造数：相邻块 A/B + 覆盖 A 末尾的重叠块 C（主键升序 limit 1 取 A）
    const auto addVmb = [](uint64_t ptr, uint64_t size) {
        VirtualMemBlock row{};
        row.start_ptr = ptr;
        row.dev_mapped_ptr = ptr;
        row.size = size;
        row.device_id = 7U;
        row.rank_id = 7U;
        row.src_type = static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV);
        auto added = sim::runtime::Db::Add(row);
        EXPECT_TRUE(added.ok()) << added.diagnostic;
        return row.id;
    };
    constexpr uint64_t kAddrBase = 0x0010000000000000ULL;
    const uint64_t blockA = addVmb(kAddrBase + 0x19903981ULL, 0x1000ULL);
    const uint64_t blockB = addVmb(kAddrBase + 0x19904981ULL, 0x1000ULL);
    const uint64_t blockC = addVmb(kAddrBase + 0x19904800ULL, 0x200ULL);
    ASSERT_LT(blockA, blockC);
    constexpr uint64_t kProbe = kAddrBase + 0x19904979ULL; // A 末 8 字节，被 A/C 覆盖

    // 关闭会话，模拟旧二进制发布的库：直接删除新索引
    const auto closed = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    ASSERT_TRUE(closed.ok()) << closed.diagnostic;
    const std::string original = VmbIndexNameByColumns(kRunnerPath, kNewIndexColumns);
    ASSERT_FALSE(original.empty()) << "新联合索引未在初始 schema 中登记";
    const std::string dropError = ExecRawSql(kRunnerPath, "DROP INDEX " + original);
    ASSERT_TRUE(dropError.empty()) << dropError;
    EXPECT_TRUE(VmbIndexNameByColumns(kRunnerPath, kNewIndexColumns).empty())
        << "DROP INDEX 未生效，后续断言不能证明补建";

    // 重新打开并首次访问：owner Prepare 的齐备性预检发现缺索引 → 补建 + 回填
    const HcclSim::Storage::StorageRuntimeConfig config{
        {{std::string{"RUNNER_DB"}, sim::backends::kSqliteBackendName, std::string{kRunnerPath}},
         {std::string{"OP_DATA_DB"}, sim::backends::kSqliteBackendName, std::string{kOpDataPath}}}};
    const auto reopened = HcclSim::Storage::StorageRuntime::EnsureProcessSession(config);
    ASSERT_TRUE(reopened.ok()) << reopened.diagnostic;
    const auto found = sim::runtime::Db::GetOneByPred<VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&VirtualMemBlock::dev_mapped_ptr, kProbe),
        HcclSim::Storage::Gt(&VirtualMemBlock::dev_mapped_end_ptr, kProbe),
        HcclSim::Storage::Eq(&VirtualMemBlock::src_type, static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV))));
    ASSERT_TRUE(found.ok()) << found.diagnostic;
    ASSERT_TRUE(found.value.has_value());
    EXPECT_EQ((*found.value).id, blockA) << "主键升序 limit 1：A/C 重叠时必须取 A";

    // 补建完整：索引回到 sqlite_master，且计划实际选中它
    const std::string restored = VmbIndexNameByColumns(kRunnerPath, kNewIndexColumns);
    EXPECT_FALSE(restored.empty()) << "重新打开后新联合索引未被补建";
    ExpectIndexSearch(
        kRunnerPath,
        "SELECT * FROM VirtualMemBlock WHERE src_type = 0 AND dev_mapped_ptr <= " + std::to_string(kProbe)
            + " AND dev_mapped_end_ptr > " + std::to_string(kProbe),
        "src_type=? AND dev_mapped_ptr");
}
} // namespace
