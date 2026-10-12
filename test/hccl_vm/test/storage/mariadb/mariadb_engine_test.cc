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
 * @file mariadb_engine_test.cc
 * @brief MariaDbEngine 引擎级单元测试（直连真实 mysqld；不经会话装配层）。
 *
 * fixture 协议：外部真实 MariaDB Server 的 endpoint 由环境变量
 * HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 提供（mariadb://user:password@host:port）：
 *   - 未设置 → 每个用例明确 GTEST_SKIP（本地无 mysqld 时不伪造通过）；
 *   - 已设置但不可用/不可达 → 用例必须失败（不得静默跳过）。
 * 用例隔离：每个用例用唯一 dataNamespace（引擎按 namespace 派生物理 database，
 * 全新 database 天然零数据耦合；per-run 实例数据目录用后即弃）。
 */
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>

#include <mysql.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>

#include "gtest/gtest.h"

#include "storage/database_result.h"
#include "storage_backends/mariadb/engine/mariadb_engine.h"
#include "storage_backends/mariadb/engine/mariadb_statement_cache.h"
#include "storage/internal/engine_contract_types.h"
#include "storage/query_spec.h"
#include "storage/storage_catalog.h"
#include "storage_composition/storage_deploy_config.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage_backends/mariadb/composition/mariadb_deploy_section.h"
#include "storage_backends/mariadb/composition/mariadb_backend_module.h"

namespace mariadb_engine_test {
namespace {
    /// 真实 MariaDB endpoint 环境变量（由测试环境提供；per-run mysqld）
    constexpr char kMariadbEndpointEnv[] = "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT";

    bool EndpointConfigured()
    {
        const char* endpoint = std::getenv(kMariadbEndpointEnv);
        return endpoint != nullptr && endpoint[0] != '\0';
    }

    HcclSim::Storage::MariadbConnectionProfile TestProfile()
    {
        HcclSim::Storage::MariadbConnectionProfile profile;
        profile.uri = std::string{std::getenv(kMariadbEndpointEnv)};
        profile.connectionEstablishmentTimeoutS = 3;
        profile.packetReadTimeoutS = 30;
        profile.packetWriteTimeoutS = 30;
        return profile;
    }

    /// 指定线路协议压缩的测试 profile（其余同 TestProfile）
    HcclSim::Storage::MariadbConnectionProfile TestProfileWithCompression(bool compression)
    {
        auto profile = TestProfile();
        profile.wireProtocolCompression = compression;
        return profile;
    }
} // namespace

// ---- 测试目录：六类型全字段表 + 定长 Text/Blob 数组表 + 外键子表 ----

struct AllTypesRow {
    uint64_t id{0};
    int64_t signed64{0};
    int32_t signed32{0};
    uint16_t unsigned16{0};
    double realDouble{0.0};
    float realFloat{0.0F};
    bool flag{false};
    std::string text;
    std::vector<uint8_t> blob;
    uint64_t unsigned64{0};
};

struct FixedArrayRow {
    uint64_t id{0};
    char name[16]{};
    uint8_t bytes[12]{};
    uint32_t owner{0};
};

struct ChildOfAllTypes {
    uint64_t id{0};
    uint64_t parentId{0};
    uint32_t amount{0};
};

/// 动态 PID scope 表（物理表 <table>_P_<pid>，两 pid 互不可见）
struct DynRow {
    uint64_t id{0};
    uint64_t owner{0};
    uint32_t seq{0};
};

/// 区间形状联合索引合同 fixture：镜像 VirtualMemBlock 查找形状（device/src_type 等
/// 值前缀 + start ≤ addr < end 半开区间），声明普通四列联合索引（末两列为
/// 互异 uint64 字段——FORCE INDEX 匹配的端点类型合同）
struct IntervalRow {
    uint64_t id{0};
    uint64_t device_id{0};
    uint32_t src_type{0};
    uint64_t start_ptr{0};
    uint64_t end_ptr{0};
};
} // namespace mariadb_engine_test

using namespace HcclSim::Storage::Ddl;
#define Field HCCL_VM_DDL_FIELD
#define Version HCCL_VM_DDL_VERSION
#define PrimaryKey HCCL_VM_DDL_PRIMARY_KEY
#define Index HCCL_VM_DDL_INDEX
#define ForeignKey HCCL_VM_DDL_FOREIGN_KEY
#define Scope Ddl::ScopeKind
#define Text Ddl::Text
#define Blob Ddl::Blob
#define AutoIncrement Ddl::AutoIncrement
#define OnDelete Ddl::OnDelete

#define HCCL_VM_MARIADB_TEST_DATABASE(Table)                                                                           \
    Table(                                                                                                             \
        mariadb_engine_test::AllTypesRow, (Version(1), Scope::Static, Field(id, AutoIncrement), Field(signed64),       \
                                           Field(signed32), Field(unsigned16), Field(realDouble), Field(realFloat),    \
                                           Field(flag), Field(text), Field(blob), Field(unsigned64), PrimaryKey(id)))  \
        Table(                                                                                                         \
            mariadb_engine_test::FixedArrayRow,                                                                        \
            (Version(1), Scope::Static, Field(id, AutoIncrement), Field(name, Text), Field(bytes, Blob), Field(owner), \
             PrimaryKey(id), Index(owner)))                                                                            \
            Table(                                                                                                     \
                mariadb_engine_test::ChildOfAllTypes,                                                                  \
                (Version(1), Scope::Static, Field(id, AutoIncrement), Field(parentId), Field(amount), PrimaryKey(id),  \
                 ForeignKey(parentId, mariadb_engine_test::AllTypesRow::id, OnDelete::Cascade)))                       \
                Table(                                                                                                 \
                    mariadb_engine_test::DynRow, (Version(1), Scope::DynamicPid, Field(id, AutoIncrement),             \
                                                  Field(owner), Field(seq), PrimaryKey(id)))                           \
                    Table(                                                                                             \
                        mariadb_engine_test::IntervalRow,                                                              \
                        (Version(1), Scope::Static, Field(id, AutoIncrement), Field(device_id), Field(src_type),       \
                         Field(start_ptr), Field(end_ptr), PrimaryKey(id),                                             \
                         Index(device_id, src_type, start_ptr, end_ptr)))

#define HCCL_VM_DDL_CURRENT_DATABASE "MARIADB_TEST_DB"
HCCL_VM_DEFINE_DATABASE(kMariadbTestDatabase, MARIADB_TEST_DB, HCCL_VM_MARIADB_TEST_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE
#undef Field
#undef Version
#undef PrimaryKey
#undef Index
#undef ForeignKey
#undef Scope
#undef Text
#undef Blob
#undef AutoIncrement
#undef OnDelete

namespace mariadb_engine_test {
using namespace HcclSim::Storage;

/// 测试观察点：语句缓存命中计数（仅本文件使用；产品代码不得引用。
/// 与 SQLite 测试同构——friend 声明的名字必须落在 HcclSim::Storage 命名空间）
namespace {
    /// 打开一个指向唯一 namespace 的引擎（READ_WRITE_CREATE；全新 database）。
    /// 装配失败记录在 ok/diagnostic（构造器内不得 ASSERT 提前 return），用例
    /// 先断言 ready 再使用。
    struct EngineFixture {
        std::unique_ptr<MariaDbEngine> engine;
        std::string dataNamespace;
        bool ready{false};
        std::string diagnostic;

        /// namespace 追加进程 pid：物理 database 跨测试运行持久存在，唯一化保证
        /// 同服务器上的重复运行零数据耦合（per-run mysqld 实例整体用后即弃）
        /// profile 缺省 TestProfile()；需要固定线路行为的用例显式传入
        /// （如行流中断线用例依赖压缩协议的 CPU 开销拖慢行流，见该用例注释）
        explicit EngineFixture(std::string ns, HcclSim::Storage::MariadbConnectionProfile profile = TestProfile())
            : dataNamespace(ns + "_p" + std::to_string(static_cast<long>(::getpid())))
        {
            engine = std::make_unique<MariaDbEngine>(&kMariadbTestDatabase, profile);
            EngineOpenRequest request;
            request.mode = StorageOpenMode::READ_WRITE_CREATE;
            request.dataNamespace = dataNamespace;
            const auto opened = engine->Open(request);
            if (!opened.ok()) {
                diagnostic = opened.diagnostic;
                return;
            }
            const auto prepared = engine->Prepare();
            if (!prepared.ok()) {
                diagnostic = prepared.diagnostic;
                return;
            }
            ready = true;
        }
        ~EngineFixture()
        {
            if (engine != nullptr && ready) {
                (void)engine->Close();
            }
        }
    };

    const Ddl::CatalogTableMeta& MetaOf(const std::string_view tableId)
    {
        for (std::size_t i = 0; i < kMariadbTestDatabase.tableCount; ++i) {
            if (kMariadbTestDatabase.tables[i].tableId == tableId) {
                return kMariadbTestDatabase.tables[i];
            }
        }
        ADD_FAILURE() << "test table missing from catalog: " << tableId;
        static const Ddl::CatalogTableMeta kEmpty{};
        return kEmpty;
    }
} // namespace
} // namespace mariadb_engine_test

namespace HcclSim::Storage {
/// 测试观察点：经引擎公共统计合同读取缓存计数（镜像 SQLite 测试的
/// SchemaDdlStatistics 消费方式；产品代码不得引用本别名）
struct MariadbEngineTestPeer {
    static uint64_t CacheHits(const MariaDbEngine& engine) { return engine.StatementCacheStatistics().hits; }
    static uint64_t CacheMisses(const MariaDbEngine& engine) { return engine.StatementCacheStatistics().misses; }
    static uint64_t CacheSize(const MariaDbEngine& engine) { return engine.StatementCacheStatistics().size; }
};
} // namespace HcclSim::Storage

namespace mariadb_engine_test {

// M0 smoke：Open（建库建连接）+ Prepare（静态 DDL：3 张表 + 索引 + 外键）+
// Insert（六类型全字段 + 自增回填）+ Select（vector 全行读回逐字段比对）+
// Count（条件计数）+ ListTables 由 M1 落码后补齐。
TEST(MariadbEngineSmoke, OpenPrepareInsertSelectCountRoundtrip)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "真实 mysqld smoke 需要外部 endpoint";
    }
    EngineFixture fixture{"m0smoke_roundtrip"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    AllTypesRow row{};
    row.signed64 = -42;
    row.signed32 = -7;
    row.unsigned16 = 60000;
    row.realDouble = 3.25;
    row.realFloat = -0.5F;
    row.flag = true;
    row.text = "hccl-mariadb-smoke 你好";
    row.blob = {0x00, 0x01, 0xFE, 0xFF, 0x7F, 0x80};
    const auto inserted = fixture.engine->Insert(meta, TableScope::Static(), &row, nullptr);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    ASSERT_GT(*inserted.value, 0U);
    // 自增主键写回由 Table<T> 层完成；引擎级测试手动镜像该语义
    row.id = *inserted.value;

    AllTypesRow second{};
    second.signed64 = std::numeric_limits<int64_t>::min();
    second.signed32 = std::numeric_limits<int32_t>::max();
    second.unsigned16 = 0;
    second.realDouble = -1.5e300;
    second.realFloat = 1.5e-30F;
    second.flag = false;
    second.text = "";
    second.blob = {};
    const auto insertedSecond = fixture.engine->Insert(meta, TableScope::Static(), &second, nullptr);
    ASSERT_TRUE(insertedSecond.ok()) << insertedSecond.diagnostic;
    ASSERT_GT(*insertedSecond.value, row.id);
    second.id = *insertedSecond.value;

    const auto where = ConditionData{ConditionClause{
        static_cast<uint16_t>(0), Compare::EQUAL, ConditionJoin::NONE, 0, 0,
        ConditionValue{DbScalar{uint64_t{row.id}}}}};
    std::vector<AllTypesRow> rows;
    auto sink = ResultSink::For(rows);
    const auto selected = fixture.engine->Select(meta, TableScope::Static(), where, QueryOptions{}, sink);
    ASSERT_TRUE(selected.ok()) << selected.diagnostic;
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].id, row.id);
    EXPECT_EQ(rows[0].signed64, row.signed64);
    EXPECT_EQ(rows[0].signed32, row.signed32);
    EXPECT_EQ(rows[0].unsigned16, row.unsigned16);
    EXPECT_DOUBLE_EQ(rows[0].realDouble, 3.25);
    EXPECT_FLOAT_EQ(rows[0].realFloat, -0.5F);
    EXPECT_TRUE(rows[0].flag);
    EXPECT_EQ(rows[0].text, row.text);
    EXPECT_EQ(rows[0].blob, row.blob);

    const auto counted = fixture.engine->Count(meta, TableScope::Static(), where);
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    EXPECT_EQ(*counted.value, 1U);

    const auto countedAll = fixture.engine->Count(meta, TableScope::Static(), ConditionData{});
    ASSERT_TRUE(countedAll.ok()) << countedAll.diagnostic;
    EXPECT_EQ(*countedAll.value, 2U);
}

// M0 smoke：2MB BLOB 与 MEDIUMTEXT 往返（LONGBLOB/MEDIUMTEXT 列 + fetch_column
// 按需拉取路径 + max_allowed_packet 形状）
TEST(MariadbEngineSmoke, LargeBlobAndTextRoundtrip)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m0smoke_largeblob"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    AllTypesRow row{};
    row.text.assign(256 * 1024, 'x');
    row.blob.resize(2U * 1024U * 1024U + 13U);
    for (std::size_t i = 0; i < row.blob.size(); ++i) {
        row.blob[i] = static_cast<uint8_t>(i % 251U);
    }
    const auto inserted = fixture.engine->Insert(meta, TableScope::Static(), &row, nullptr);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    ASSERT_NE(*inserted.value, 0U);
    row.id = *inserted.value; // 镜像 Table<T> 的自增主键写回

    std::vector<AllTypesRow> rows;
    auto sink = ResultSink::For(rows);
    const auto where = ConditionData{ConditionClause{
        static_cast<uint16_t>(0), Compare::EQUAL, ConditionJoin::NONE, 0, 0,
        ConditionValue{DbScalar{uint64_t{row.id}}}}};
    const auto selected = fixture.engine->Select(meta, TableScope::Static(), where, QueryOptions{}, sink);
    ASSERT_TRUE(selected.ok()) << selected.diagnostic;
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].text.size(), row.text.size());
    EXPECT_EQ(rows[0].text, row.text);
    EXPECT_EQ(rows[0].blob.size(), row.blob.size());
    EXPECT_EQ(rows[0].blob, row.blob);
}

// M0 smoke：定长 char[16] Text 与 uint8_t[12] Blob 数组往返（NUL 截断与
// 有效字节语义）
TEST(MariadbEngineSmoke, FixedArrayTextAndBlobRoundtrip)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m0smoke_fixedarray"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("FixedArrayRow");

    FixedArrayRow row{};
    std::memcpy(row.name, "pod-0001", 8); // NUL 前字节承载语义
    for (uint8_t i = 0; i < 12U; ++i) {
        row.bytes[i] = static_cast<uint8_t>(0xF0U + i);
    }
    row.owner = 7;
    const auto inserted = fixture.engine->Insert(meta, TableScope::Static(), &row, nullptr);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    ASSERT_NE(*inserted.value, 0U);
    row.id = *inserted.value; // 镜像 Table<T> 的自增主键写回

    std::vector<FixedArrayRow> rows;
    auto sink = ResultSink::For(rows);
    const auto where = ConditionData{ConditionClause{
        static_cast<uint16_t>(3), Compare::EQUAL, ConditionJoin::NONE, 0, 0, ConditionValue{DbScalar{uint64_t{7}}}}};
    const auto selected = fixture.engine->Select(meta, TableScope::Static(), where, QueryOptions{}, sink);
    ASSERT_TRUE(selected.ok()) << selected.diagnostic;
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_STREQ(rows[0].name, "pod-0001");
    EXPECT_EQ(std::memcmp(rows[0].bytes, row.bytes, sizeof(row.bytes)), 0);
}

// M0 smoke：语句缓存命中计数（同形 INSERT/SELECT 第二次执行命中缓存，
// 省去 prepare 的两次网络往返）
TEST(MariadbEngineSmoke, StatementCacheHitsOnRepeatedShape)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m0smoke_cachecount"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    const auto insertRow = [&](AllTypesRow& row) {
        row.text = "cache";
        auto result = fixture.engine->Insert(meta, TableScope::Static(), &row, nullptr);
        if (result.ok()) {
            row.id = *result.value; // 镜像 Table<T> 的自增主键写回
        }
        return result;
    };
    AllTypesRow first{};
    ASSERT_TRUE(insertRow(first).ok());
    const uint64_t missesAfterFirst = MariadbEngineTestPeer::CacheMisses(*fixture.engine);
    const uint64_t hitsAfterFirst = MariadbEngineTestPeer::CacheHits(*fixture.engine);
    EXPECT_GE(missesAfterFirst, 1U);

    AllTypesRow second{};
    ASSERT_TRUE(insertRow(second).ok());
    AllTypesRow third{};
    ASSERT_TRUE(insertRow(third).ok());
    EXPECT_EQ(MariadbEngineTestPeer::CacheMisses(*fixture.engine), missesAfterFirst) << "同形 INSERT 不得新增 prepare";
    EXPECT_GT(MariadbEngineTestPeer::CacheHits(*fixture.engine), hitsAfterFirst) << "同形 INSERT 第二次起必须命中缓存";

    const auto where = ConditionData{ConditionClause{
        static_cast<uint16_t>(0), Compare::GREATER_OR_EQUAL, ConditionJoin::NONE, 0, 0,
        ConditionValue{DbScalar{uint64_t{first.id}}}}};
    std::vector<AllTypesRow> rows;
    auto sink = ResultSink::For(rows);
    ASSERT_TRUE(fixture.engine->Select(meta, TableScope::Static(), where, QueryOptions{}, sink).ok());
    const uint64_t selectMisses = MariadbEngineTestPeer::CacheMisses(*fixture.engine);
    std::vector<AllTypesRow> again;
    auto secondSink = ResultSink::For(again);
    ASSERT_TRUE(fixture.engine->Select(meta, TableScope::Static(), where, QueryOptions{}, secondSink).ok());
    EXPECT_EQ(MariadbEngineTestPeer::CacheMisses(*fixture.engine), selectMisses) << "同形 SELECT 不得新增 prepare";
    EXPECT_EQ(again.size(), rows.size());
}

// M0 smoke：外键表 Insert（约束创建成功即 DDL 冒烟）+ Prepare 齐备性预检
// （第二次 Prepare 在同连接上幂等拒绝，重开引擎走预检路径不重放 DDL——
// 观察计数经 M1 后补齐，此处以重开成功为准）
TEST(MariadbEngineSmoke, ForeignKeyTableAndReopen)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    std::string reopenNamespace;
    {
        EngineFixture fixture{"m0smoke_fk"};
        // 重开必须命中 fixture 的同一物理 database（namespace 带 _p<pid> 后缀）
        reopenNamespace = fixture.dataNamespace;
        ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
        // 先插父行（外键目标必须存在——与 SQLite foreign_keys=ON 同一约束语义）
        const auto& parentMeta = MetaOf("AllTypesRow");
        AllTypesRow parent{};
        parent.text = "fk-parent";
        const auto parentInserted = fixture.engine->Insert(parentMeta, TableScope::Static(), &parent, nullptr);
        ASSERT_TRUE(parentInserted.ok()) << parentInserted.diagnostic;
        parent.id = *parentInserted.value;

        const auto& meta = MetaOf("ChildOfAllTypes");
        ChildOfAllTypes child{};
        child.parentId = parent.id;
        child.amount = 5;
        const auto inserted = fixture.engine->Insert(meta, TableScope::Static(), &child, nullptr);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
        ASSERT_NE(*inserted.value, 0U);
    }
    // 重开同一 namespace：READ_WRITE_CREATE + 齐备 schema → Prepare 直接成功
    // （不重放 DDL；数据仍在——物理 database 是持久的，清理由 per-run 实例承担）
    {
        auto engine = std::make_unique<MariaDbEngine>(&kMariadbTestDatabase, TestProfile());
        EngineOpenRequest request;
        request.mode = StorageOpenMode::READ_WRITE_CREATE;
        request.dataNamespace = reopenNamespace;
        ASSERT_TRUE(engine->Open(request).ok());
        ASSERT_TRUE(engine->Prepare().ok());
        const auto& meta = MetaOf("ChildOfAllTypes");
        const auto counted = engine->Count(meta, TableScope::Static(), ConditionData{});
        ASSERT_TRUE(counted.ok()) << counted.diagnostic;
        EXPECT_EQ(*counted.value, 1U);
        ASSERT_TRUE(engine->Close().ok());
    }
}

// ---- M1：Update/Delete/Clear/ListTables/事务/死锁/并发/备份 ----

namespace {
    /// 条件子句构造快捷方式（fieldId/compare/值；连接与括号按需追加）
    ConditionClause Clause(
        uint16_t fieldId, Compare compare, ConditionValue value, ConditionJoin join = ConditionJoin::NONE,
        uint16_t open = 0, uint16_t close = 0)
    {
        return ConditionClause{fieldId, compare, join, open, close, std::move(value)};
    }

    /// 插入一行已知 signed64/text 的 AllTypesRow 并回填 id（事务内传入 binding）
    uint64_t InsertTyped(
        MariaDbEngine& engine, const Ddl::CatalogTableMeta& meta, int64_t signedValue, std::string text,
        TransactionBinding* transaction = nullptr)
    {
        AllTypesRow row{};
        row.signed64 = signedValue;
        row.text = std::move(text);
        auto inserted = engine.Insert(meta, TableScope::Static(), &row, transaction);
        EXPECT_TRUE(inserted.ok()) << inserted.diagnostic;
        return inserted.ok() ? *inserted.value : 0U;
    }

    /// 按条件取 id 集合
    std::set<uint64_t> SelectIds(
        MariaDbEngine& engine, const Ddl::CatalogTableMeta& meta, const ConditionData& where,
        const QueryOptions& options = QueryOptions{})
    {
        std::vector<AllTypesRow> rows;
        auto sink = ResultSink::For(rows);
        const auto selected = engine.Select(meta, TableScope::Static(), where, options, sink);
        EXPECT_TRUE(selected.ok()) << selected.diagnostic;
        std::set<uint64_t> ids;
        for (const auto& row : rows) {
            ids.insert(row.id);
        }
        return ids;
    }
} // namespace

// Compare 全集 + AND/OR/括号分组 + 空 IN（六类 Compare 逐种 1:1 用例）
TEST(MariadbEngineSemantics, CompareMatrixAndGrouping)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m1smoke_compare"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");
    constexpr uint16_t kSigned = 1; // signed64
    constexpr uint16_t kText = 7;   // text

    const std::vector<int64_t> values = {10, 20, 30, 40, 50};
    const std::vector<std::string> texts = {"alpha", "beta", "gamma", "alphabet", "Beta"};
    std::set<uint64_t> allIds;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto id = InsertTyped(*fixture.engine, meta, values[i], texts[i]);
        ASSERT_NE(id, 0U);
        allIds.insert(id);
    }

    EXPECT_EQ(
        SelectIds(
            *fixture.engine, meta,
            ConditionData{Clause(kSigned, Compare::EQUAL, ConditionValue{DbScalar{int64_t{30}}})}),
        (std::set<uint64_t>{*std::next(allIds.begin(), 2)}));
    EXPECT_EQ(
        SelectIds(
            *fixture.engine, meta,
            ConditionData{Clause(kSigned, Compare::NOT_EQUAL, ConditionValue{DbScalar{int64_t{30}}})}),
        (std::set<uint64_t>{
            *allIds.begin(), *std::next(allIds.begin(), 1), *std::next(allIds.begin(), 3),
            *std::next(allIds.begin(), 4)}));
    EXPECT_EQ(
        SelectIds(
            *fixture.engine, meta,
            ConditionData{Clause(kSigned, Compare::LESS, ConditionValue{DbScalar{int64_t{30}}})}),
        (std::set<uint64_t>{*allIds.begin(), *std::next(allIds.begin(), 1)}));
    EXPECT_EQ(
        SelectIds(
            *fixture.engine, meta,
            ConditionData{Clause(kSigned, Compare::LESS_OR_EQUAL, ConditionValue{DbScalar{int64_t{30}}})}),
        (std::set<uint64_t>{*allIds.begin(), *std::next(allIds.begin(), 1), *std::next(allIds.begin(), 2)}));
    EXPECT_EQ(
        SelectIds(
            *fixture.engine, meta,
            ConditionData{Clause(kSigned, Compare::GREATER, ConditionValue{DbScalar{int64_t{30}}})}),
        (std::set<uint64_t>{*std::next(allIds.begin(), 3), *std::next(allIds.begin(), 4)}));
    EXPECT_EQ(
        SelectIds(
            *fixture.engine, meta,
            ConditionData{Clause(kSigned, Compare::GREATER_OR_EQUAL, ConditionValue{DbScalar{int64_t{30}}})}),
        (std::set<uint64_t>{
            *std::next(allIds.begin(), 2), *std::next(allIds.begin(), 3), *std::next(allIds.begin(), 4)}));
    // PREFIX：utf8mb4_bin 下大小写敏感（"alphabet" 命中 "alpha" 前缀，"Beta" 不命中）
    EXPECT_EQ(
        SelectIds(
            *fixture.engine, meta,
            ConditionData{Clause(kText, Compare::PREFIX, ConditionValue{DbScalar{std::string{"alpha"}}})}),
        (std::set<uint64_t>{*allIds.begin(), *std::next(allIds.begin(), 3)}));
    EXPECT_EQ(
        SelectIds(
            *fixture.engine, meta,
            ConditionData{Clause(kText, Compare::PREFIX, ConditionValue{DbScalar{std::string{"Be"}}})}),
        (std::set<uint64_t>{*std::next(allIds.begin(), 4)}));
    // IN 多值；空 IN 恒假（集合字面量提升到宏参数外，规避 EXPECT_EQ 宏参数
    // 内裸花括号逗号）
    const std::vector<DbScalar> inValues{DbScalar{int64_t{10}}, DbScalar{int64_t{30}}, DbScalar{int64_t{50}}};
    EXPECT_EQ(
        SelectIds(*fixture.engine, meta, ConditionData{Clause(kSigned, Compare::IN, ConditionValue{inValues})}),
        (std::set<uint64_t>{*allIds.begin(), *std::next(allIds.begin(), 2), *std::next(allIds.begin(), 4)}));
    const std::vector<DbScalar> emptyValues{};
    EXPECT_TRUE(
        SelectIds(*fixture.engine, meta, ConditionData{Clause(kSigned, Compare::IN, ConditionValue{emptyValues})})
            .empty());
    // AND（默认连接）
    const std::vector<ConditionClause> andClauses{
        Clause(kSigned, Compare::GREATER_OR_EQUAL, ConditionValue{DbScalar{int64_t{20}}}),
        Clause(kSigned, Compare::LESS, ConditionValue{DbScalar{int64_t{40}}})};
    EXPECT_EQ(
        SelectIds(*fixture.engine, meta, ConditionData{andClauses}),
        (std::set<uint64_t>{*std::next(allIds.begin(), 1), *std::next(allIds.begin(), 2)}));
    // 括号分组 + OR：(signed64 <= 20 OR signed64 >= 50) AND text = 'beta'
    // （'Beta' 因 utf8mb4_bin 不等于 'beta'，验证括号语义与二进制排序同时生效）
    const std::vector<ConditionClause> orClauses{
        Clause(kSigned, Compare::LESS_OR_EQUAL, ConditionValue{DbScalar{int64_t{20}}}, ConditionJoin::NONE, 1, 0),
        Clause(kSigned, Compare::GREATER_OR_EQUAL, ConditionValue{DbScalar{int64_t{50}}}, ConditionJoin::OR, 0, 1),
        Clause(kText, Compare::EQUAL, ConditionValue{DbScalar{std::string{"beta"}}}, ConditionJoin::AND)};
    EXPECT_EQ(
        SelectIds(*fixture.engine, meta, ConditionData{orClauses}),
        (std::set<uint64_t>{*std::next(allIds.begin(), 1)}));
}

// 无效 utf8 字节绑定前清洗（方言差异 #15）：业务层存在未初始化定长 char
// 数组（如 runtime::Server.version 从未赋值），SQLite 原样入库；MariaDB
// utf8mb4 严格模式拒收 errno 1366——引擎清洗无效字节为空格，有效前缀与
// 嵌入 NUL（合法 utf8）原样保留
TEST(MariadbEngineSemantics, InvalidUtf8TextIsSanitizedOnBind)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m1smoke_utf8san"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    AllTypesRow row{};
    row.signed64 = 99;
    row.text = "ok";
    row.text += '\xFF';
    row.text += '\xFE';
    row.text += '\x00';
    row.text += "tail";
    row.text += '\xC3'; // 孤立续段前导（缺后续字节）
    const auto inserted = fixture.engine->Insert(meta, TableScope::Static(), &row, nullptr);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;

    std::vector<AllTypesRow> rows;
    auto sink = ResultSink::For(rows);
    const auto selected = fixture.engine->Select(
        meta, TableScope::Static(), ConditionData{Clause(1, Compare::EQUAL, ConditionValue{DbScalar{int64_t{99}}})},
        QueryOptions{}, sink);
    ASSERT_TRUE(selected.ok()) << selected.diagnostic;
    ASSERT_EQ(rows.size(), 1U);
    const std::string expected = std::string("ok  \x00tail ", 10);
    EXPECT_EQ(rows[0].text, expected) << "无效字节须替换为空格，其余原样往返";
}

// utf8mb4_bin 等值比较大小写敏感（对齐 SQLite BINARY 语义）
TEST(MariadbEngineSemantics, Utf8mb4BinEqualityIsCaseSensitive)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m1smoke_case"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");
    constexpr uint16_t kText = 7;

    const uint64_t a = InsertTyped(*fixture.engine, meta, 1, "Abc");
    const uint64_t b = InsertTyped(*fixture.engine, meta, 2, "abc");
    const uint64_t c = InsertTyped(*fixture.engine, meta, 3, "ABc");
    ASSERT_NE(a, 0U);
    ASSERT_NE(b, 0U);
    ASSERT_NE(c, 0U);

    EXPECT_EQ(
        SelectIds(
            *fixture.engine, meta,
            ConditionData{Clause(kText, Compare::EQUAL, ConditionValue{DbScalar{std::string{"abc"}}})}),
        (std::set<uint64_t>{b}));
    EXPECT_EQ(
        SelectIds(
            *fixture.engine, meta,
            ConditionData{Clause(kText, Compare::EQUAL, ConditionValue{DbScalar{std::string{"Abc"}}})}),
        (std::set<uint64_t>{a}));
    EXPECT_EQ(
        SelectIds(
            *fixture.engine, meta,
            ConditionData{Clause(kText, Compare::EQUAL, ConditionValue{DbScalar{std::string{"ABC"}}})}),
        (std::set<uint64_t>{}));
}

// ORDER BY 方向/复合键/tie-breaker + LIMIT + 投影（未投影字段保持默认值）
TEST(MariadbEngineSemantics, OrderByLimitAndProjection)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m1smoke_order"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    // signed64 序列 5,3,3,1,2（两个 3 验证主键 tie-breaker）
    const std::vector<int64_t> values = {5, 3, 3, 1, 2};
    std::vector<uint64_t> ids;
    for (const int64_t value : values) {
        ids.push_back(InsertTyped(*fixture.engine, meta, value, "row"));
    }
    QueryOptions options;
    options.sorts.emplace_back(static_cast<uint16_t>(1), Order::ASCENDING);
    EXPECT_EQ(
        SelectIds(*fixture.engine, meta, ConditionData{}, options),
        (std::set<uint64_t>{ids.begin(), ids.end()})); // 集合语义（顺序另验）

    // 全行读回验证排序顺序：升序 + 同值按主键升序
    {
        std::vector<AllTypesRow> rows;
        auto sink = ResultSink::For(rows);
        const auto selected = fixture.engine->Select(meta, TableScope::Static(), ConditionData{}, options, sink);
        ASSERT_TRUE(selected.ok()) << selected.diagnostic;
        ASSERT_EQ(rows.size(), 5U);
        EXPECT_EQ(rows[0].signed64, 1);
        EXPECT_EQ(rows[1].signed64, 2);
        EXPECT_EQ(rows[2].signed64, 3);
        EXPECT_EQ(rows[3].signed64, 3);
        EXPECT_LT(rows[2].id, rows[3].id) << "同值行必须按主键升序 tie-break";
        EXPECT_EQ(rows[4].signed64, 5);
    }
    // 降序 + LIMIT 3
    {
        QueryOptions desc;
        desc.sorts.emplace_back(static_cast<uint16_t>(1), Order::DESCENDING);
        desc.limit = 3U;
        std::vector<AllTypesRow> rows;
        auto sink = ResultSink::For(rows);
        const auto selected = fixture.engine->Select(meta, TableScope::Static(), ConditionData{}, desc, sink);
        ASSERT_TRUE(selected.ok()) << selected.diagnostic;
        ASSERT_EQ(rows.size(), 3U);
        EXPECT_EQ(rows[0].signed64, 5);
        EXPECT_EQ(rows[1].signed64, 3);
        EXPECT_EQ(rows[2].signed64, 3);
        EXPECT_LT(rows[1].id, rows[2].id);
    }
    // 投影：只取 id+signed64，text 物化为默认空串；过滤/排序字段不要求被投影
    // 包含——WHERE/ORDER BY 独立生成，由数据库端执行过滤与排序
    {
        QueryOptions projected;
        projected.projectedFields = {0, 1};
        std::vector<AllTypesRow> rows;
        auto sink = ResultSink::For(rows);
        const auto selected = fixture.engine->Select(meta, TableScope::Static(), ConditionData{}, projected, sink);
        ASSERT_TRUE(selected.ok()) << selected.diagnostic;
        ASSERT_EQ(rows.size(), 5U);
        EXPECT_NE(rows[0].id, 0U);
        EXPECT_TRUE(rows[0].text.empty()) << "未投影字段必须保持默认值";
    }
    {
        // 条件引用未投影字段（signed64 不在投影中）：查询成功且过滤正确。
        QueryOptions projected;
        projected.projectedFields = {0, 7};
        std::vector<AllTypesRow> rows;
        auto sink = ResultSink::For(rows);
        const auto filtered = fixture.engine->Select(
            meta, TableScope::Static(), ConditionData{Clause(1, Compare::EQUAL, ConditionValue{DbScalar{int64_t{3}}})},
            projected, sink);
        ASSERT_TRUE(filtered.ok()) << filtered.diagnostic;
        ASSERT_EQ(rows.size(), 2U);
        EXPECT_EQ(rows[0].signed64, 0); // signed64 未投影，保持默认值
        EXPECT_LT(rows[0].id, rows[1].id);
    }
    {
        // 排序引用未投影字段：查询成功且排序正确（含主键 tie-breaker）。
        QueryOptions projected;
        projected.projectedFields = {0};
        projected.sorts.emplace_back(static_cast<uint16_t>(1), Order::ASCENDING);
        std::vector<AllTypesRow> rows;
        auto sink = ResultSink::For(rows);
        const auto sorted = fixture.engine->Select(meta, TableScope::Static(), ConditionData{}, projected, sink);
        ASSERT_TRUE(sorted.ok()) << sorted.diagnostic;
        ASSERT_EQ(rows.size(), 5U);
        EXPECT_EQ(rows[0].id, ids[3]); // signed64=1
        EXPECT_EQ(rows[4].id, ids[0]); // signed64=5
    }
    {
        // 非法投影列（超出目录字段数的编号 / 未映射编号）：明确报错。
        QueryOptions outOfRange;
        outOfRange.projectedFields = {static_cast<uint16_t>(meta.fieldCount)};
        std::vector<AllTypesRow> rows;
        auto sink = ResultSink::For(rows);
        const auto rejectedRange
            = fixture.engine->Select(meta, TableScope::Static(), ConditionData{}, outOfRange, sink);
        EXPECT_EQ(rejectedRange.code, DbCode::INVALID_ARGUMENT);

        QueryOptions unmapped;
        unmapped.projectedFields = {kUnknownFieldId};
        std::vector<AllTypesRow> unmappedRows;
        auto unmappedSink = ResultSink::For(unmappedRows);
        const auto rejectedUnmapped
            = fixture.engine->Select(meta, TableScope::Static(), ConditionData{}, unmapped, unmappedSink);
        EXPECT_EQ(rejectedUnmapped.code, DbCode::INVALID_ARGUMENT);
    }
}

// Update/Delete：affected_rows 语义 + INCREMENT 原子加 + 唯一冲突 CONFLICT
TEST(MariadbEngineSemantics, UpdateDeleteAndIncrement)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m1smoke_updatedelete"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");
    constexpr uint16_t kSigned = 1;

    InsertTyped(*fixture.engine, meta, 10, "a");
    InsertTyped(*fixture.engine, meta, 20, "b");
    InsertTyped(*fixture.engine, meta, 30, "c");

    // 条件更新：affected = 2，SET 新值生效
    const auto updated = fixture.engine->Update(
        meta, TableScope::Static(),
        ConditionData{Clause(kSigned, Compare::GREATER_OR_EQUAL, ConditionValue{DbScalar{int64_t{20}}})},
        UpdateData{LogicalChange{7, ChangeKind::SET, DbScalar{std::string{"touched"}}}}, nullptr);
    ASSERT_TRUE(updated.ok()) << updated.diagnostic;
    EXPECT_EQ(*updated.value, 2U);
    EXPECT_EQ(
        SelectIds(
            *fixture.engine, meta,
            ConditionData{Clause(7, Compare::EQUAL, ConditionValue{DbScalar{std::string{"touched"}}})})
            .size(),
        2U);

    // INCREMENT：signed64 += 5（单语句原子加）
    const auto incremented = fixture.engine->Update(
        meta, TableScope::Static(),
        ConditionData{Clause(kSigned, Compare::EQUAL, ConditionValue{DbScalar{int64_t{10}}})},
        UpdateData{LogicalChange{kSigned, ChangeKind::INCREMENT, DbScalar{int64_t{5}}}}, nullptr);
    ASSERT_TRUE(incremented.ok()) << incremented.diagnostic;
    EXPECT_EQ(*incremented.value, 1U);
    EXPECT_EQ(
        SelectIds(
            *fixture.engine, meta,
            ConditionData{Clause(kSigned, Compare::EQUAL, ConditionValue{DbScalar{int64_t{15}}})})
            .size(),
        1U);

    // 主键列不可更新；无 where 不可更新
    EXPECT_EQ(
        fixture.engine
            ->Update(
                meta, TableScope::Static(),
                ConditionData{Clause(0, Compare::EQUAL, ConditionValue{DbScalar{uint64_t{1}}})},
                UpdateData{LogicalChange{0, ChangeKind::SET, DbScalar{uint64_t{9}}}}, nullptr)
            .code,
        DbCode::INVALID_ARGUMENT);
    EXPECT_EQ(
        fixture.engine
            ->Update(
                meta, TableScope::Static(), ConditionData{},
                UpdateData{LogicalChange{7, ChangeKind::SET, DbScalar{std::string{"x"}}}}, nullptr)
            .code,
        DbCode::INVALID_ARGUMENT);

    // Delete：affected 行数 + 数据消失（INCREMENT 后值为 15/20/30，条件 < 20
    // 恰好命中被原子加过的那一行）
    const auto deleted = fixture.engine->Delete(
        meta, TableScope::Static(),
        ConditionData{Clause(kSigned, Compare::LESS, ConditionValue{DbScalar{int64_t{20}}})}, nullptr);
    ASSERT_TRUE(deleted.ok()) << deleted.diagnostic;
    EXPECT_EQ(*deleted.value, 1U);
    const auto counted = fixture.engine->Count(meta, TableScope::Static(), ConditionData{});
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    EXPECT_EQ(*counted.value, 2U);
}

// INCREMENT INT64_MIN 增量（无符号 64 列全域边界）：幅值 2^63 经
// 0 - uint64(INT64_MIN) 无符号回绕取得（不对 INT64_MIN 取负）；基值 2^63
// 减幅值 2^63 = 0 合法，基值 0 下溢为明确错误且数据保持。BIGINT UNSIGNED
// 原生承载幅值——与 SQLite（表示域限制，显式拒绝）的差异见交付说明。
TEST(MariadbEngineSemantics, IncrementInt64MinDeltaOnUnsignedColumn)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m1smoke_incmin"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");
    constexpr uint16_t kSigned = 1;
    constexpr uint16_t kUnsigned64 = 9;

    InsertTyped(*fixture.engine, meta, 10, "a");
    const ConditionData byRow{Clause(kSigned, Compare::EQUAL, ConditionValue{DbScalar{int64_t{10}}})};

    const auto based = fixture.engine->Update(
        meta, TableScope::Static(), byRow,
        UpdateData{LogicalChange{kUnsigned64, ChangeKind::SET, DbScalar{uint64_t{1} << 63}}}, nullptr);
    ASSERT_TRUE(based.ok()) << based.diagnostic;

    const auto decremented = fixture.engine->Update(
        meta, TableScope::Static(), byRow,
        UpdateData{LogicalChange{kUnsigned64, ChangeKind::INCREMENT, DbScalar{INT64_MIN}}}, nullptr);
    ASSERT_TRUE(decremented.ok()) << decremented.diagnostic;
    EXPECT_EQ(
        SelectIds(
            *fixture.engine, meta,
            ConditionData{Clause(kUnsigned64, Compare::EQUAL, ConditionValue{DbScalar{uint64_t{0}}})})
            .size(),
        1U);

    const auto underflow = fixture.engine->Update(
        meta, TableScope::Static(), byRow,
        UpdateData{LogicalChange{kUnsigned64, ChangeKind::INCREMENT, DbScalar{INT64_MIN}}}, nullptr);
    EXPECT_FALSE(underflow.ok());
    EXPECT_EQ(
        SelectIds(
            *fixture.engine, meta,
            ConditionData{Clause(kUnsigned64, Compare::EQUAL, ConditionValue{DbScalar{uint64_t{0}}})})
            .size(),
        1U);
}
// 【参数生命周期回归】BindScalar 对字符串/BLOB 借用数据指针（数值入槽
// 复制）：UPDATE 的 SET 参数必须直接绑定调用方持有的原始 change.value
// （UpdateData 存活覆盖 execute 与断线重试）；不得经按值返回 DbScalar 的
// 中间转换——临时副本在绑定表达式结束时销毁，字符串/BLOB 借用即成悬空
// 指针（静态确认的 use-after-free；修复前短串 SSO/堆块未复写时偶然通过
// 读回，不能作为无缺陷证据）。长串（>SSO 15 字符，堆分配）与非空 BLOB
// 提高读回暴露概率，逐字节比对。
TEST(MariadbEngineSemantics, UpdateSetStringAndBlobLifecycleRoundtrip)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m1smoke_setlife"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");
    constexpr uint16_t kSigned = 1;     // signed64
    constexpr uint16_t kText = 7;       // text
    constexpr uint16_t kBlob = 8;       // blob
    constexpr uint16_t kUnsigned64 = 9; // unsigned64
    const uint64_t rowId = InsertTyped(*fixture.engine, meta, 10, "a");

    const auto byRow = ConditionData{Clause(0, Compare::EQUAL, ConditionValue{DbScalar{rowId}})};
    const auto selectRow = [&]() -> AllTypesRow {
        std::vector<AllTypesRow> rows;
        auto sink = ResultSink::For(rows);
        const auto selected = fixture.engine->Select(meta, TableScope::Static(), byRow, QueryOptions{}, sink);
        EXPECT_TRUE(selected.ok()) << selected.diagnostic;
        return rows.empty() ? AllTypesRow{} : rows.front();
    };

    // 1) 长 ASCII 字符串（堆分配）SET：读回逐字节一致
    std::string longText(517, '\0');
    for (std::size_t i = 0; i < longText.size(); ++i) {
        longText[i] = static_cast<char>('a' + (i % 26));
    }
    {
        const auto set = fixture.engine->Update(
            meta, TableScope::Static(), byRow, UpdateData{LogicalChange{kText, ChangeKind::SET, DbScalar{longText}}},
            nullptr);
        ASSERT_TRUE(set.ok()) << set.diagnostic;
        const auto back = selectRow();
        EXPECT_EQ(back.text, longText);
    }

    // 2) 非空 BLOB SET：读回逐字节一致
    std::vector<uint8_t> blob(769);
    for (std::size_t i = 0; i < blob.size(); ++i) {
        blob[i] = static_cast<uint8_t>((i * 7U + 13U) & 0xFFU);
    }
    {
        const auto set = fixture.engine->Update(
            meta, TableScope::Static(), byRow, UpdateData{LogicalChange{kBlob, ChangeKind::SET, DbScalar{blob}}},
            nullptr);
        ASSERT_TRUE(set.ok()) << set.diagnostic;
        const auto back = selectRow();
        EXPECT_EQ(back.blob, blob);
    }

    // 3) 同一 UPDATE 混合：字符串 SET + BLOB SET + 无符号列负 INCREMENT +
    //    有符号列正 INCREMENT（四参数 + WHERE 参数；借用与数值归一化并存）
    ASSERT_TRUE(fixture.engine
                    ->Update(
                        meta, TableScope::Static(), byRow,
                        UpdateData{LogicalChange{kUnsigned64, ChangeKind::SET, DbScalar{uint64_t{10}}}}, nullptr)
                    .ok());
    std::string mixedText(64, '\0');
    for (std::size_t i = 0; i < mixedText.size(); ++i) {
        mixedText[i] = static_cast<char>('A' + (i % 23));
    }
    std::vector<uint8_t> mixedBlob(256);
    for (std::size_t i = 0; i < mixedBlob.size(); ++i) {
        mixedBlob[i] = static_cast<uint8_t>((i * 31U + 5U) & 0xFFU);
    }
    {
        const UpdateData mixed{
            LogicalChange{kText, ChangeKind::SET, DbScalar{mixedText}},
            LogicalChange{kBlob, ChangeKind::SET, DbScalar{mixedBlob}},
            LogicalChange{kUnsigned64, ChangeKind::INCREMENT, DbScalar{int64_t{-3}}},
            LogicalChange{kSigned, ChangeKind::INCREMENT, DbScalar{int64_t{5}}}};
        const auto updated = fixture.engine->Update(meta, TableScope::Static(), byRow, mixed, nullptr);
        ASSERT_TRUE(updated.ok()) << updated.diagnostic;
        const auto back = selectRow();
        EXPECT_EQ(back.text, mixedText);
        EXPECT_EQ(back.blob, mixedBlob);
        EXPECT_EQ(back.unsigned64, 7U);
        EXPECT_EQ(back.signed64, 15);
    }

    // 4) 多参数 WHERE（字符串条件）+ 多列 SET：参数顺序与绑定一一对应
    {
        const auto byText
            = ConditionData{Clause(kText, Compare::EQUAL, ConditionValue{DbScalar{std::string{mixedText}}})};
        std::string tail(96, '\0');
        for (std::size_t i = 0; i < tail.size(); ++i) {
            tail[i] = static_cast<char>('0' + (i % 10));
        }
        const auto multi = fixture.engine->Update(
            meta, TableScope::Static(), byText,
            UpdateData{
                LogicalChange{kText, ChangeKind::SET, DbScalar{std::string{"prefix-match-target-0123456789"} + tail}},
                LogicalChange{kSigned, ChangeKind::SET, DbScalar{int64_t{4242}}}},
            nullptr);
        ASSERT_TRUE(multi.ok()) << multi.diagnostic;
        const auto back = selectRow();
        EXPECT_EQ(back.text, std::string{"prefix-match-target-0123456789"} + tail);
        EXPECT_EQ(back.signed64, 4242);
    }

    // 5) PREFIX（StartsWith）长模式：转义产物必须由参数槽持有副本到 execute
    //    （ASan 实证旧路径借用局部 pattern 悬空；IpcNotify name_or_key 业务在用）
    {
        std::string keyBase(120, '\0');
        for (std::size_t i = 0; i < keyBase.size(); ++i) {
            keyBase[i] = static_cast<char>('a' + (i % 26));
        }
        const std::string stored = keyBase + "_tail";
        ASSERT_TRUE(fixture.engine
                        ->Update(
                            meta, TableScope::Static(), byRow,
                            UpdateData{LogicalChange{kText, ChangeKind::SET, DbScalar{stored}}}, nullptr)
                        .ok());
        const auto prefixMiss
            = HcclSim::Storage::StartsWith(&AllTypesRow::text, keyBase.substr(0, 100) + std::string{"%_never"}).data;
        std::vector<AllTypesRow> missRows;
        auto missSink = ResultSink::For(missRows);
        const auto miss = fixture.engine->Select(meta, TableScope::Static(), prefixMiss, QueryOptions{}, missSink);
        EXPECT_TRUE(miss.ok()) << miss.diagnostic;
        EXPECT_TRUE(missRows.empty());
        const auto prefixHit = HcclSim::Storage::StartsWith(&AllTypesRow::text, keyBase.substr(0, 100)).data;
        std::vector<AllTypesRow> hitRows;
        auto hitSink = ResultSink::For(hitRows);
        const auto hit = fixture.engine->Select(meta, TableScope::Static(), prefixHit, QueryOptions{}, hitSink);
        EXPECT_TRUE(hit.ok()) << hit.diagnostic;
        ASSERT_EQ(hitRows.size(), 1U);
        EXPECT_EQ(hitRows[0].text, stored);
    }
}

// —— 缺陷B：区间查询选择已声明 IntervalIndex 的四列联合索引 ——
namespace {
    struct AdminEndpoint {
        std::string host;
        std::string user;
        std::string password;
        unsigned int port{0};
    };

    bool ParseAdminEndpoint(const std::string& uri, AdminEndpoint& out)
    {
        const std::string scheme = "mariadb://";
        if (uri.rfind(scheme, 0) != 0) {
            return false;
        }
        const auto at = uri.find('@', scheme.size());
        const auto colon = uri.find(':', scheme.size());
        if (at == std::string::npos || colon == std::string::npos || colon > at) {
            return false;
        }
        out.user = uri.substr(scheme.size(), colon - scheme.size());
        out.password = uri.substr(colon + 1, at - colon - 1);
        const std::string hostPort = uri.substr(at + 1);
        const auto portColon = hostPort.rfind(':');
        if (portColon == std::string::npos) {
            return false;
        }
        out.host = hostPort.substr(0, portColon);
        out.port = static_cast<unsigned int>(std::stoul(hostPort.substr(portColon + 1)));
        return !out.host.empty() && out.port != 0;
    }

    /// 管理连接直执行（仅针对本目录 per-run 私有实例；只操作本 fixture 的索引）
    bool AdminQuery(const std::string& sql, std::string& error)
    {
        AdminEndpoint ep{};
        if (!ParseAdminEndpoint(std::string{std::getenv(kMariadbEndpointEnv)}, ep)) {
            error = "endpoint parse failed";
            return false;
        }
        MYSQL* admin = mysql_init(nullptr);
        if (admin == nullptr) {
            error = "mysql_init failed";
            return false;
        }
        if (mysql_real_connect(
                admin, ep.host.c_str(), ep.user.c_str(), ep.password.c_str(), nullptr, ep.port, nullptr, 0)
            == nullptr) {
            error = "connect failed: " + std::string{mysql_error(admin)};
            mysql_close(admin);
            return false;
        }
        const bool ok = mysql_query(admin, sql.c_str()) == 0;
        if (!ok) {
            error = "query failed: " + std::string{mysql_error(admin)};
        }
        mysql_close(admin);
        return ok;
    }
} // namespace

// 区间形状查询（device/src_type 等值前缀 + start ≤ addr < end 半开区间，ORDER BY id
// LIMIT 1——与 FindDevMappedVirMemByDevice 同形）在普通四列联合索引在场时结果正确
// （多设备隔离/边界/无命中/重叠取最小 id）；DROP 索引后匹配形状查询必须失败（证明
// 生成了 FORCE INDEX 且指向 DerivedIndexName 的真实索引名——匹配依据是普通联合
// 索引的列序、末两列互异 uint64 端点与严格谓词形状），非匹配形状（OR 连接/
// 缺等值前缀/反向区间/多余子句）不受影响仍走默认计划。
TEST(MariadbEngineSemantics, IntervalQueryCompositeIndexSelection)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m1smoke_interval"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("IntervalRow");
    constexpr uint16_t kId = 0;
    constexpr uint16_t kDev = 1;
    constexpr uint16_t kSrc = 2;
    constexpr uint16_t kStart = 3;
    constexpr uint16_t kEnd = 4;

    struct Spec {
        uint64_t device;
        uint32_t src;
        uint64_t start;
        uint64_t end;
    };
    const Spec specs[] = {
        {7, 1, 100, 200}, {7, 1, 150, 250}, {7, 1, 200, 300}, {7, 1, 1000, 1100}, {7, 2, 100, 200}, {8, 1, 100, 200},
    };
    std::vector<uint64_t> ids;
    for (const auto& spec : specs) {
        IntervalRow row{};
        row.device_id = spec.device;
        row.src_type = spec.src;
        row.start_ptr = spec.start;
        row.end_ptr = spec.end;
        const auto inserted = fixture.engine->Insert(meta, TableScope::Static(), &row, nullptr);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
        ids.push_back(*inserted.value);
    }

    const auto intervalWhere = [&](uint64_t device, uint32_t src, uint64_t addr) {
        return ConditionData{std::vector<ConditionClause>{
            Clause(kStart, Compare::LESS_OR_EQUAL, ConditionValue{DbScalar{addr}}),
            Clause(kEnd, Compare::GREATER, ConditionValue{DbScalar{addr}}, ConditionJoin::AND),
            Clause(kSrc, Compare::EQUAL, ConditionValue{DbScalar{uint64_t{src}}}, ConditionJoin::AND),
            Clause(kDev, Compare::EQUAL, ConditionValue{DbScalar{device}}, ConditionJoin::AND)}};
    };
    const auto selectFirst = [&](const ConditionData& where) -> std::vector<IntervalRow> {
        QueryOptions options;
        options.sorts.emplace_back(kId, Order::ASCENDING);
        options.limit = 1U;
        std::vector<IntervalRow> rows;
        auto sink = ResultSink::For(rows);
        const auto selected = fixture.engine->Select(meta, TableScope::Static(), where, options, sink);
        EXPECT_TRUE(selected.ok()) << selected.diagnostic;
        return rows;
    };

    // 重叠区间取最小 id（180 同时命中 [100,200) 与 [150,250)）
    auto rows = selectFirst(intervalWhere(7, 1, 180));
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].id, ids[0]);
    // 左闭右开边界（200 不属于 [100,200)，属于 [150,250) 与 [200,300)）
    rows = selectFirst(intervalWhere(7, 1, 200));
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].id, ids[1]);
    // 无命中
    rows = selectFirst(intervalWhere(7, 1, 99));
    EXPECT_TRUE(rows.empty());
    rows = selectFirst(intervalWhere(7, 1, 1500));
    EXPECT_TRUE(rows.empty());
    // 多行完整结果（无 limit 时不裁剪）
    {
        std::vector<IntervalRow> all;
        auto sink = ResultSink::For(all);
        const auto selected
            = fixture.engine->Select(meta, TableScope::Static(), intervalWhere(7, 1, 180), QueryOptions{}, sink);
        ASSERT_TRUE(selected.ok()) << selected.diagnostic;
        std::set<uint64_t> hit;
        for (const auto& row : all) {
            hit.insert(row.id);
        }
        EXPECT_EQ(hit, (std::set<uint64_t>{ids[0], ids[1]}));
    }
    // 设备与 src_type 隔离
    rows = selectFirst(intervalWhere(8, 1, 180));
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].id, ids[5]);
    rows = selectFirst(intervalWhere(7, 2, 180));
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].id, ids[4]);

    // —— FORCE INDEX 取证：DROP 联合索引后，匹配形状必须失败（语句引用了该
    // 索引），非匹配形状继续走默认计划不受影响。
    std::string fullName;
    {
        AdminEndpoint ep{};
        ASSERT_TRUE(ParseAdminEndpoint(std::string{std::getenv(kMariadbEndpointEnv)}, ep));
        MYSQL* admin = mysql_init(nullptr);
        ASSERT_NE(admin, nullptr);
        ASSERT_TRUE(
            mysql_real_connect(
                admin, ep.host.c_str(), ep.user.c_str(), ep.password.c_str(), nullptr, ep.port, nullptr, 0)
            != nullptr)
            << mysql_error(admin);
        const std::string findIndex = "SELECT CONCAT(TABLE_SCHEMA, '.', TABLE_NAME, '.', INDEX_NAME) "
                                      "FROM information_schema.statistics WHERE TABLE_SCHEMA LIKE 'hcclvm_"
                                      + fixture.dataNamespace + "_%' AND INDEX_NAME LIKE 'idx_IntervalRow%'";
        ASSERT_EQ(mysql_query(admin, findIndex.c_str()), 0) << mysql_error(admin);
        MYSQL_RES* res = mysql_store_result(admin);
        ASSERT_NE(res, nullptr);
        MYSQL_ROW indexRow = mysql_fetch_row(res);
        ASSERT_NE(indexRow, nullptr) << "interval index not found";
        ASSERT_NE(indexRow[0], nullptr);
        fullName = indexRow[0];
        mysql_free_result(res);
        mysql_close(admin);
    }
    const auto schemaEnd = fullName.find('.');
    const auto tableEnd = fullName.find('.', schemaEnd + 1);
    ASSERT_NE(schemaEnd, std::string::npos);
    ASSERT_NE(tableEnd, std::string::npos);
    const std::string dropSql = "DROP INDEX `" + fullName.substr(tableEnd + 1) + "` ON `"
                                + fullName.substr(0, schemaEnd) + "`.`"
                                + fullName.substr(schemaEnd + 1, tableEnd - schemaEnd - 1) + "`";
    std::string adminError;
    ASSERT_TRUE(AdminQuery(dropSql, adminError)) << adminError;

    std::vector<IntervalRow> forcedRows;
    auto forcedSink = ResultSink::For(forcedRows);
    const auto forced
        = fixture.engine->Select(meta, TableScope::Static(), intervalWhere(7, 1, 180), QueryOptions{}, forcedSink);
    EXPECT_FALSE(forced.ok()) << "matching shape must reference the interval index";

    std::vector<IntervalRow> prefixRows;
    auto prefixSink = ResultSink::For(prefixRows);
    const auto missingPrefix = fixture.engine->Select(
        meta, TableScope::Static(),
        ConditionData{std::vector<ConditionClause>{
            Clause(kStart, Compare::LESS_OR_EQUAL, ConditionValue{DbScalar{uint64_t{180}}}),
            Clause(kEnd, Compare::GREATER, ConditionValue{DbScalar{uint64_t{180}}}, ConditionJoin::AND),
            Clause(kSrc, Compare::EQUAL, ConditionValue{DbScalar{uint64_t{1}}}, ConditionJoin::AND)}},
        QueryOptions{}, prefixSink);
    EXPECT_TRUE(missingPrefix.ok()) << missingPrefix.diagnostic;

    std::vector<IntervalRow> reversedRows;
    auto reversedSink = ResultSink::For(reversedRows);
    const auto reversed = fixture.engine->Select(
        meta, TableScope::Static(),
        ConditionData{std::vector<ConditionClause>{
            Clause(kStart, Compare::GREATER, ConditionValue{DbScalar{uint64_t{180}}}),
            Clause(kEnd, Compare::LESS_OR_EQUAL, ConditionValue{DbScalar{uint64_t{180}}}, ConditionJoin::AND),
            Clause(kSrc, Compare::EQUAL, ConditionValue{DbScalar{uint64_t{1}}}, ConditionJoin::AND),
            Clause(kDev, Compare::EQUAL, ConditionValue{DbScalar{uint64_t{7}}}, ConditionJoin::AND)}},
        QueryOptions{}, reversedSink);
    EXPECT_TRUE(reversed.ok()) << reversed.diagnostic;

    std::vector<IntervalRow> extraRows;
    auto extraSink = ResultSink::For(extraRows);
    const auto extraClause = fixture.engine->Select(
        meta, TableScope::Static(),
        ConditionData{std::vector<ConditionClause>{
            Clause(kStart, Compare::LESS_OR_EQUAL, ConditionValue{DbScalar{uint64_t{180}}}),
            Clause(kEnd, Compare::GREATER, ConditionValue{DbScalar{uint64_t{180}}}, ConditionJoin::AND),
            Clause(kSrc, Compare::EQUAL, ConditionValue{DbScalar{uint64_t{1}}}, ConditionJoin::AND),
            Clause(kDev, Compare::EQUAL, ConditionValue{DbScalar{uint64_t{7}}}, ConditionJoin::AND),
            Clause(kId, Compare::GREATER, ConditionValue{DbScalar{uint64_t{0}}}, ConditionJoin::AND)}},
        QueryOptions{}, extraSink);
    EXPECT_TRUE(extraClause.ok()) << extraClause.diagnostic;

    // OR 连接的区间形状：即使全部子句按列匹配，OR 不构成联合索引的等值前缀 +
    // 半开区间形态，不得套用提示（走默认计划，结果正确）。
    std::vector<IntervalRow> orRows;
    auto orSink = ResultSink::For(orRows);
    const auto orJoined = fixture.engine->Select(
        meta, TableScope::Static(),
        ConditionData{std::vector<ConditionClause>{
            Clause(kStart, Compare::LESS_OR_EQUAL, ConditionValue{DbScalar{uint64_t{180}}}),
            Clause(kEnd, Compare::GREATER, ConditionValue{DbScalar{uint64_t{180}}}, ConditionJoin::OR),
            Clause(kSrc, Compare::EQUAL, ConditionValue{DbScalar{uint64_t{1}}}, ConditionJoin::AND),
            Clause(kDev, Compare::EQUAL, ConditionValue{DbScalar{uint64_t{7}}}, ConditionJoin::AND)}},
        QueryOptions{}, orSink);
    EXPECT_TRUE(orJoined.ok()) << orJoined.diagnostic;
}

// Clear 三模式 + 活动事务期间拒绝（镜像 SQLite L1804 语义）
TEST(MariadbEngineSemantics, ClearModesAndTransactionRejection)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m1smoke_clear"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    // KEEP_SEQUENCE：清数据保留自增计数器
    InsertTyped(*fixture.engine, meta, 1, "one");
    InsertTyped(*fixture.engine, meta, 2, "two");
    ASSERT_TRUE(fixture.engine->Clear(meta, TableScope::Static(), ClearMode::KEEP_SEQUENCE).ok());
    EXPECT_EQ(*fixture.engine->Count(meta, TableScope::Static(), ConditionData{}).value, 0U);
    const uint64_t kept = InsertTyped(*fixture.engine, meta, 3, "three");
    EXPECT_GT(kept, 2U) << "KEEP_SEQUENCE 后自增计数器不得回退";

    // RESET_SEQUENCE：TRUNCATE 重置计数器
    ASSERT_TRUE(fixture.engine->Clear(meta, TableScope::Static(), ClearMode::RESET_SEQUENCE).ok());
    EXPECT_EQ(*fixture.engine->Count(meta, TableScope::Static(), ConditionData{}).value, 0U);
    const uint64_t reset = InsertTyped(*fixture.engine, meta, 4, "four");
    EXPECT_EQ(reset, 1U) << "RESET_SEQUENCE 后首个自增 id 必须为 1";

    // 活动事务期间拒绝 Clear（INVALID_STATE）
    TransactionBinding transaction;
    ASSERT_TRUE(fixture.engine->Begin(transaction).ok());
    EXPECT_EQ(fixture.engine->Clear(meta, TableScope::Static(), ClearMode::KEEP_SEQUENCE).code, DbCode::INVALID_STATE);
    ASSERT_TRUE(fixture.engine->Rollback(transaction).ok());
}

// 动态 PID 表：两 pid 物理表互不可见 + DROP_DYNAMIC_SCOPE 销毁全部 scope 表
TEST(MariadbEngineSemantics, DynamicPidScopesIsolated)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m1smoke_dynpid"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("DynRow");

    const auto insertDyn = [&](uint64_t pid, uint64_t owner, uint32_t seq) {
        DynRow row{};
        row.owner = owner;
        row.seq = seq;
        const auto inserted = fixture.engine->Insert(meta, TableScope::Process(pid), &row, nullptr);
        EXPECT_TRUE(inserted.ok()) << inserted.diagnostic;
        return inserted.ok() ? *inserted.value : 0U;
    };
    ASSERT_NE(insertDyn(101, 101, 1), 0U);
    ASSERT_NE(insertDyn(101, 101, 2), 0U);
    ASSERT_NE(insertDyn(102, 102, 3), 0U);

    // 跨进程重入：第二个引擎实例（另一进程视角）首次触碰同一物理 scope 表
    // ——CREATE INDEX 撞已存在索引（MariaDB 无 IF NOT EXISTS 语法），引擎须
    // 按 1061 幂等容忍（镜像 SQLite 语义），否则 checker 读 rank 数据即败
    {
        auto second = std::make_unique<MariaDbEngine>(&kMariadbTestDatabase, TestProfile());
        EngineOpenRequest request;
        request.mode = StorageOpenMode::READ_WRITE_CREATE;
        request.dataNamespace = fixture.dataNamespace;
        ASSERT_TRUE(second->Open(request).ok());
        ASSERT_TRUE(second->Prepare().ok());
        DynRow extra{};
        extra.owner = 101;
        extra.seq = 9;
        const auto inserted = second->Insert(meta, TableScope::Process(101), &extra, nullptr);
        EXPECT_TRUE(inserted.ok()) << inserted.diagnostic;
        EXPECT_EQ(*second->Count(meta, TableScope::Process(101), ConditionData{}).value, 3U);
        ASSERT_TRUE(second->Close().ok());
    }

    const auto countOf = [&](uint64_t pid) {
        return *fixture.engine->Count(meta, TableScope::Process(pid), ConditionData{}).value;
    };
    EXPECT_EQ(countOf(101), 3U);
    EXPECT_EQ(countOf(102), 1U);

    // ListTables 可见两张物理表
    const auto listed = fixture.engine->ListTables();
    ASSERT_TRUE(listed.ok()) << listed.diagnostic;
    const std::set<std::string> names(listed.value->begin(), listed.value->end());
    EXPECT_TRUE(names.count("DynRow_P_101") > 0);
    EXPECT_TRUE(names.count("DynRow_P_102") > 0);
    EXPECT_TRUE(names.count("AllTypesRow") > 0);
    EXPECT_TRUE(names.count("FixedArrayRow") > 0);
    EXPECT_TRUE(names.count("ChildOfAllTypes") > 0);

    // DROP_DYNAMIC_SCOPE：销毁该基名的全部物理 scope 表（两 pid 一起）
    const auto dropped = fixture.engine->Clear(meta, TableScope::Process(101), ClearMode::DROP_DYNAMIC_SCOPE);
    ASSERT_TRUE(dropped.ok()) << dropped.diagnostic;
    EXPECT_EQ(dropped.affectedRows, 2U);
    const auto after = fixture.engine->ListTables();
    ASSERT_TRUE(after.ok()) << after.diagnostic;
    const std::set<std::string> afterNames(after.value->begin(), after.value->end());
    EXPECT_EQ(afterNames.count("DynRow_P_101"), 0U);
    EXPECT_EQ(afterNames.count("DynRow_P_102"), 0U);
}

// 引擎级事务：提交可见/回滚不可见/活动期拒绝普通 DML/重复 Begin 拒绝
TEST(MariadbEngineSemantics, TransactionCommitRollback)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m1smoke_transaction"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    // 回滚：两行插入全部消失
    {
        TransactionBinding transaction;
        ASSERT_TRUE(fixture.engine->Begin(transaction).ok());
        ASSERT_NE(InsertTyped(*fixture.engine, meta, 1, "tx-a", &transaction), 0U);
        ASSERT_NE(InsertTyped(*fixture.engine, meta, 2, "tx-b", &transaction), 0U);
        // 活动事务期间：普通 DML（transaction 为空）拒绝；重复 Begin 拒绝
        AllTypesRow plain{};
        EXPECT_EQ(fixture.engine->Insert(meta, TableScope::Static(), &plain, nullptr).code, DbCode::INVALID_STATE);
        TransactionBinding second;
        EXPECT_EQ(fixture.engine->Begin(second).code, DbCode::INVALID_STATE);
        ASSERT_TRUE(fixture.engine->Rollback(transaction).ok());
        EXPECT_EQ(*fixture.engine->Count(meta, TableScope::Static(), ConditionData{}).value, 0U);
    }
    // 提交：行可见
    {
        TransactionBinding transaction;
        ASSERT_TRUE(fixture.engine->Begin(transaction).ok());
        AllTypesRow row{};
        row.signed64 = 7;
        row.text = "committed";
        const auto inserted = fixture.engine->Insert(meta, TableScope::Static(), &row, &transaction);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
        const auto committed = fixture.engine->Commit(transaction);
        ASSERT_TRUE(committed.ok()) << committed.diagnostic;
        EXPECT_EQ(*fixture.engine->Count(meta, TableScope::Static(), ConditionData{}).value, 1U);
    }
    // 事务内 Update/Delete 参与同一原子域
    {
        TransactionBinding transaction;
        ASSERT_TRUE(fixture.engine->Begin(transaction).ok());
        const auto updated = fixture.engine->Update(
            meta, TableScope::Static(),
            ConditionData{Clause(7, Compare::EQUAL, ConditionValue{DbScalar{std::string{"committed"}}})},
            UpdateData{LogicalChange{1, ChangeKind::SET, DbScalar{int64_t{99}}}}, &transaction);
        ASSERT_TRUE(updated.ok()) << updated.diagnostic;
        ASSERT_TRUE(fixture.engine->Rollback(transaction).ok());
        EXPECT_EQ(
            SelectIds(
                *fixture.engine, meta, ConditionData{Clause(1, Compare::EQUAL, ConditionValue{DbScalar{int64_t{99}}})})
                .size(),
            0U);
    }
}

// 死锁中止语义：AB-BA 环 → InnoDB 回滚受害事务 → 受害方 binding 的后续
// DML 与 Commit 返回失败（不吞死锁错误），Rollback 后连接恢复可用
TEST(MariadbEngineSemantics, DeadlockAbortsVictimTransaction)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m1smoke_deadlock"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    // 第二条连接（同一 database；两实例=两连接，模拟跨进程并发）
    auto second = std::make_unique<MariaDbEngine>(&kMariadbTestDatabase, TestProfile());
    {
        EngineOpenRequest request;
        request.mode = StorageOpenMode::READ_WRITE_CREATE;
        request.dataNamespace = fixture.dataNamespace;
        ASSERT_TRUE(second->Open(request).ok());
        ASSERT_TRUE(second->Prepare().ok());
    }

    // 数据：5 行；A 加重自身 undo（锁 1/2/3 行），B 只锁第 4 行——
    // InnoDB 死锁受害选择按事务权重，B 权重小 → 确定性成为受害方
    for (int64_t value = 1; value <= 5; ++value) {
        ASSERT_NE(InsertTyped(*fixture.engine, meta, value, "dl"), 0U);
    }
    const auto setId = [&](int64_t value) {
        return ConditionData{Clause(1, Compare::EQUAL, ConditionValue{DbScalar{int64_t{value}}})};
    };
    const auto touchText = [&](const std::string& text) {
        return UpdateData{LogicalChange{7, ChangeKind::SET, DbScalar{text}}};
    };

    TransactionBinding bindingA;
    TransactionBinding bindingB;
    ASSERT_TRUE(fixture.engine->Begin(bindingA).ok());
    ASSERT_TRUE(second->Begin(bindingB).ok());
    for (int64_t value = 1; value <= 3; ++value) {
        ASSERT_TRUE(
            fixture.engine
                ->Update(meta, TableScope::Static(), setId(value), touchText("a" + std::to_string(value)), &bindingA)
                .ok());
    }
    ASSERT_TRUE(second->Update(meta, TableScope::Static(), setId(4), touchText("b4"), &bindingB).ok());

    // A 请求第 4 行（阻塞在 B 的锁上）；随后 B 请求第 1 行 → 死锁环。
    // 死锁判定瞬间完成（受害方事务被 InnoDB 即时回滚，锁随之释放），A 的
    // 阻塞更新随即返回——先 join 再断言，避免任何断言失败时线程未汇合而
    // 触发 std::terminate。
    std::atomic<bool> aWasVictim{false};
    std::thread blocked([&]() {
        const auto result = fixture.engine->Update(meta, TableScope::Static(), setId(4), touchText("a4"), &bindingA);
        if (!result.ok()) {
            aWasVictim.store(true, std::memory_order_release);
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    const auto deadlocked = second->Update(meta, TableScope::Static(), setId(1), touchText("b1"), &bindingB);
    blocked.join();

    if (!deadlocked.ok() && deadlocked.code == DbCode::RESOURCE_EXHAUSTED) {
        // B 为受害方（InnoDB 已回滚整个事务）：后续 DML 与 Commit 必须失败
        AllTypesRow probe{};
        EXPECT_EQ(second->Insert(meta, TableScope::Static(), &probe, &bindingB).code, DbCode::INVALID_STATE);
        const auto commitB = second->Commit(bindingB);
        EXPECT_EQ(commitB.code, DbCode::INVALID_STATE);
        // Rollback 后连接恢复可用（可再次 Begin）
        ASSERT_TRUE(second->Rollback(bindingB).ok());
        TransactionBinding fresh;
        EXPECT_TRUE(second->Begin(fresh).ok());
        EXPECT_TRUE(second->Rollback(fresh).ok());
    } else {
        // A 为受害方：B 的更新成功，A 的阻塞更新以 RESOURCE_EXHAUSTED 失败
        ASSERT_TRUE(deadlocked.ok()) << deadlocked.diagnostic;
        ASSERT_TRUE(second->Commit(bindingB).ok());
        EXPECT_TRUE(aWasVictim.load());
        AllTypesRow probe{};
        EXPECT_EQ(fixture.engine->Insert(meta, TableScope::Static(), &probe, &bindingA).code, DbCode::INVALID_STATE);
        EXPECT_EQ(fixture.engine->Commit(bindingA).code, DbCode::INVALID_STATE);
        ASSERT_TRUE(fixture.engine->Rollback(bindingA).ok());
        return;
    }
    // A 正常提交（其全部更新生效）
    ASSERT_TRUE(fixture.engine->Commit(bindingA).ok());
    EXPECT_EQ(
        SelectIds(
            *fixture.engine, meta,
            ConditionData{Clause(7, Compare::EQUAL, ConditionValue{DbScalar{std::string{"a1"}}})})
            .size(),
        1U);
    ASSERT_TRUE(second->Close().ok());
}

// 双实例并发写 smoke（MVCC/服务端线程并行）：两连接各 60 行并发插入，
// 终态 120 行且生成主键无重复
TEST(MariadbEngineSemantics, DualInstanceConcurrentWrites)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m1smoke_concurrent"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    auto second = std::make_unique<MariaDbEngine>(&kMariadbTestDatabase, TestProfile());
    {
        EngineOpenRequest request;
        request.mode = StorageOpenMode::READ_WRITE_CREATE;
        request.dataNamespace = fixture.dataNamespace;
        ASSERT_TRUE(second->Open(request).ok());
        ASSERT_TRUE(second->Prepare().ok());
    }
    constexpr int kPerWriter = 60;
    std::atomic<int> failures{0};
    const auto writer
        = [](MariaDbEngine* engine, const Ddl::CatalogTableMeta* tableMeta, int base, std::atomic<int>* failureCount) {
              for (int i = 0; i < kPerWriter; ++i) {
                  AllTypesRow row{};
                  row.signed64 = base + i;
                  row.text = "w" + std::to_string(base);
                  const auto inserted = engine->Insert(*tableMeta, TableScope::Static(), &row, nullptr);
                  if (!inserted.ok()) {
                      ++(*failureCount);
                  }
              }
          };
    std::thread left(writer, fixture.engine.get(), &meta, 1000, &failures);
    std::thread right(writer, second.get(), &meta, 2000, &failures);
    left.join();
    right.join();
    EXPECT_EQ(failures.load(), 0);
    const auto counted = fixture.engine->Count(meta, TableScope::Static(), ConditionData{});
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    EXPECT_EQ(*counted.value, 2U * kPerWriter);
    ASSERT_TRUE(second->Close().ok());
}

// BackupRelational：自产可回放 .sql（一致性快照 + 建表语句 + INSERT 字面量）；
// 产物含转义文本与二进制 hex 字面量，成功后 tmp 原子 rename
TEST(MariadbEngineSemantics, BackupRelationalProducesReplayableDump)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"m1smoke_backup"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    AllTypesRow row{};
    row.signed64 = -17;
    row.text = "it's a \\ 'quoted' \n text";
    row.blob = {0x00, 0x01, 0xFF, 0xFE, 0x7F, 0x80, 0x00};
    ASSERT_TRUE(fixture.engine->Insert(meta, TableScope::Static(), &row, nullptr).ok());
    ASSERT_NE(InsertTyped(*fixture.engine, meta, 8, "plain"), 0U);

    const std::string destination = "/tmp/mariadb_engine_test_backup_" + std::to_string(::getpid()) + ".sql";
    const auto backed = fixture.engine->BackupRelational(EngineBackupRequest{destination});
    ASSERT_TRUE(backed.ok()) << backed.diagnostic;
    EXPECT_EQ(backed.value->destinationPath, destination);

    std::ifstream input(destination, std::ios::in | std::ios::binary);
    ASSERT_TRUE(input.good());
    std::string dump((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    EXPECT_NE(dump.find("CREATE TABLE"), std::string::npos);
    EXPECT_NE(dump.find("INSERT INTO `AllTypesRow`"), std::string::npos);
    EXPECT_NE(dump.find("it\\'s a \\\\ \\'quoted\\' \\n text"), std::string::npos) << "文本必须按 SQL 转义规则落盘";
    EXPECT_NE(dump.find("0x0001fffe7f8000"), std::string::npos) << "二进制必须以小写 hex 字面量落盘（引擎 dump 合同）";
    EXPECT_NE(dump.find("SET FOREIGN_KEY_CHECKS = 1"), std::string::npos);
    (void)std::remove(destination.c_str());
}

// ---- 部署配置 mariadb 段解析合同（纯解析，无需 mysqld；storage_mariadb.yaml）----

/// 合法 mariadb 配置：uri 与三类超时（秒）+ 线路协议压缩全部解析。
/// 字段断言经模块段解析入口（ParseMariadbDeploySection，与 Configure 同一
/// 校验函数）；完整配置的外层行为（选中后端名）经公共中立解析器验证。
TEST(MariadbEngineConfig, ParsesValidMariadbConfig)
{
    const std::string section = R"YAML(uri: mariadb://hccl:secret@127.0.0.1:3306
connection_establishment_timeout_s: 5
packet_read_timeout_s: 120
packet_write_timeout_s: 130
wire_protocol_compression: false
)YAML";
    auto parsed = sim::backends::ParseMariadbDeploySection(YAML::Load(section));
    ASSERT_TRUE(parsed.ok()) << parsed.diagnostic;
    EXPECT_EQ(parsed.value->uri, "mariadb://hccl:secret@127.0.0.1:3306");
    EXPECT_EQ(parsed.value->connectionEstablishmentTimeoutS, 5U);
    EXPECT_EQ(parsed.value->packetReadTimeoutS, 120U);
    EXPECT_EQ(parsed.value->packetWriteTimeoutS, 130U);
    EXPECT_FALSE(parsed.value->wireProtocolCompression);

    // 完整配置外层：同一段嵌入 storage.mariadb 后选中后端名为 mariadb
    std::string indented;
    for (const char* line = section.c_str(); line != nullptr;) {
        const char* next = std::strchr(line, '\n');
        const std::size_t length = next != nullptr ? static_cast<std::size_t>(next - line) : std::strlen(line);
        if (length > 0) {
            indented += "    ";
            indented.append(line, length);
            indented += '\n';
        }
        line = next != nullptr ? next + 1 : nullptr;
    }
    const std::string config = "storage:\n  backend: mariadb\n  mariadb:\n" + indented;
    auto selected = HcclSim::Storage::ParseStorageDeployConfigText(config);
    ASSERT_TRUE(selected.ok()) << selected.diagnostic;
    EXPECT_EQ(selected.value->backend, std::string{sim::backends::kMariadbBackendName});
}

/// 超时缺省：默认 3/120/120 秒；压缩缺省 false（同机部署不压缩，远端链路
/// 显式 true）
TEST(MariadbEngineConfig, TimeoutsDefaultWhenOmitted)
{
    const std::string section = "uri: mariadb://hccl:secret@127.0.0.1:3306\n";
    auto parsed = sim::backends::ParseMariadbDeploySection(YAML::Load(section));
    ASSERT_TRUE(parsed.ok()) << parsed.diagnostic;
    EXPECT_EQ(parsed.value->connectionEstablishmentTimeoutS, 3U);
    EXPECT_EQ(parsed.value->packetReadTimeoutS, 120U);
    EXPECT_EQ(parsed.value->packetWriteTimeoutS, 120U);
    EXPECT_FALSE(parsed.value->wireProtocolCompression);
}

/// 旧毫秒字段已删除：出现即解析失败并提示新字段名（不能忽略旧字段后静默用
/// 缺省值）；新旧字段同时出现同样拒绝
TEST(MariadbEngineConfig, RejectsRemovedTimeoutFieldsWithHint)
{
    const std::vector<std::pair<std::string, std::string>> removedFields = {
        {"connect_timeout_ms", "connection_establishment_timeout_s"},
        {"read_timeout_ms", "packet_read_timeout_s"},
        {"write_timeout_ms", "packet_write_timeout_s"},
    };
    for (const auto& field : removedFields) {
        const std::string config = "storage:\n  backend: mariadb\n  mariadb:\n    uri: "
                                   "mariadb://hccl:secret@127.0.0.1:3306\n    "
                                   + field.first + ": 3000\n";
        auto parsed = HcclSim::Storage::ParseStorageDeployConfigText(config);
        EXPECT_FALSE(parsed.ok()) << field.first << " 必须被拒绝";
        if (!parsed.ok()) {
            EXPECT_NE(parsed.diagnostic.find(field.second), std::string::npos)
                << "诊断必须提示新字段名: " << parsed.diagnostic;
        }
    }
    // 新旧混用：旧字段出现即报错（不进新字段解析）
    const std::string mixed = R"YAML(storage:
  backend: mariadb
  mariadb:
    uri: mariadb://hccl:secret@127.0.0.1:3306
    connection_establishment_timeout_s: 3
    connect_timeout_ms: 3000
)YAML";
    EXPECT_FALSE(HcclSim::Storage::ParseStorageDeployConfigText(mixed).ok());
    // 旧字段显式 null 同样算出现（部署意图不明，拒绝而非视同缺省）
    const std::string nullOld = "storage:\n  backend: mariadb\n  mariadb:\n    uri: "
                                "mariadb://hccl:secret@127.0.0.1:3306\n    read_timeout_ms:\n";
    EXPECT_FALSE(HcclSim::Storage::ParseStorageDeployConfigText(nullOld).ok());
}

/// 线路协议压缩显式 true/false 均可解析
TEST(MariadbEngineConfig, WireProtocolCompressionParsesExplicitValues)
{
    const std::string enabled = "uri: mariadb://hccl:secret@127.0.0.1:3306\nwire_protocol_compression: true\n";
    auto parsedEnabled = sim::backends::ParseMariadbDeploySection(YAML::Load(enabled));
    ASSERT_TRUE(parsedEnabled.ok()) << parsedEnabled.diagnostic;
    EXPECT_TRUE(parsedEnabled.value->wireProtocolCompression);

    const std::string disabled = "uri: mariadb://hccl:secret@127.0.0.1:3306\nwire_protocol_compression: false\n";
    auto parsedDisabled = sim::backends::ParseMariadbDeploySection(YAML::Load(disabled));
    ASSERT_TRUE(parsedDisabled.ok()) << parsedDisabled.diagnostic;
    EXPECT_FALSE(parsedDisabled.value->wireProtocolCompression);
}

/// 线路协议压缩非法值（非布尔标量/整数/数组/map/显式 null）全部拒绝
TEST(MariadbEngineConfig, RejectsInvalidCompressionValues)
{
    const std::vector<std::string> invalidValues = {
        "not-a-bool", "1", "0", "[true]", "{a: 1}", "",
    };
    for (const std::string& value : invalidValues) {
        const std::string config = "storage:\n  backend: mariadb\n  mariadb:\n    uri: "
                                   "mariadb://hccl:secret@127.0.0.1:3306\n    wire_protocol_compression: "
                                   + value + "\n";
        auto parsed = HcclSim::Storage::ParseStorageDeployConfigText(config);
        EXPECT_FALSE(parsed.ok()) << "value=[" << value << "] 必须被拒绝";
        if (!parsed.ok()) {
            EXPECT_EQ(parsed.code, HcclSim::Storage::DbCode::INVALID_ARGUMENT) << "value=[" << value << "]";
        }
    }
}

/// 密码可含 '@'（最后一个 '@' 切分）与空密码；URI 原样保留
TEST(MariadbEngineConfig, UriAcceptsAtInPasswordAndEmptyPassword)
{
    const std::string withAt = "uri: mariadb://hccl:p@ss@127.0.0.1:3306\n";
    auto parsed = sim::backends::ParseMariadbDeploySection(YAML::Load(withAt));
    ASSERT_TRUE(parsed.ok()) << parsed.diagnostic;
    EXPECT_EQ(parsed.value->uri, "mariadb://hccl:p@ss@127.0.0.1:3306");

    const std::string emptyPassword = "uri: mariadb://hccl:@127.0.0.1:3306\n";
    auto parsedEmpty = sim::backends::ParseMariadbDeploySection(YAML::Load(emptyPassword));
    ASSERT_TRUE(parsedEmpty.ok()) << parsedEmpty.diagnostic;
}

/// backend=mariadb 但缺 mariadb 段：装配期失败（不静默回退 SQLite）
TEST(MariadbEngineConfig, RejectsMariadbBackendWithoutSection)
{
    const std::string config = "storage:\n  backend: mariadb\n";
    auto parsed = HcclSim::Storage::ParseStorageDeployConfigText(config);
    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.code, HcclSim::Storage::DbCode::INVALID_ARGUMENT);
    EXPECT_NE(parsed.diagnostic.find("storage.mariadb"), std::string::npos);
}

/// 非法 URI：非 mariadb:// 前缀、缺 userinfo、缺 host/port、IPv6 形态、
/// port 0/越界/非数字、多余段——全部装配期失败
TEST(MariadbEngineConfig, RejectsMalformedUris)
{
    const std::vector<std::string> invalidUris = {
        "mysql://hccl:secret@127.0.0.1:3306",    // 错误 scheme
        "mariadb://127.0.0.1:3306",              // 缺 user:password
        "mariadb://:secret@127.0.0.1:3306",      // 用户名空
        "mariadb://hccl:secret@127.0.0.1",       // 缺 port
        "mariadb://hccl:secret@:3306",           // host 空
        "mariadb://hccl:secret@127.0.0.1:0",     // port 0
        "mariadb://hccl:secret@127.0.0.1:65536", // port 越界
        "mariadb://hccl:secret@127.0.0.1:abc",   // port 非数字
        "mariadb://hccl:secret@[::1]:3306",      // IPv6 不支持
        "mariadb://hccl:secret@host:3306:extra", // 多余段（多冒号拒绝）
    };
    for (const std::string& uri : invalidUris) {
        const std::string config = "storage:\n  backend: mariadb\n  mariadb:\n    uri: " + uri + "\n";
        auto parsed = HcclSim::Storage::ParseStorageDeployConfigText(config);
        EXPECT_FALSE(parsed.ok()) << "uri=[" << uri << "] 必须被拒绝";
        if (!parsed.ok()) {
            EXPECT_EQ(parsed.code, HcclSim::Storage::DbCode::INVALID_ARGUMENT) << "uri=[" << uri << "]";
        }
    }
}

/// mariadb 超时非法（0/越界/非整数/类型不符）与 uri/段类型错误：装配期失败；
/// 边界值 1 与 3600 秒合法
TEST(MariadbEngineConfig, RejectsInvalidTimeoutsAndFieldTypes)
{
    const std::vector<std::string> invalidTimeouts = {"0", "3601", "-1", "abc", "1.5", "[1]", "{a: 1}"};
    for (const std::string& value : invalidTimeouts) {
        const std::string config = R"YAML(storage:
  backend: mariadb
  mariadb:
    uri: mariadb://hccl:secret@127.0.0.1:3306
    packet_read_timeout_s: )YAML" + value
                                   + "\n";
        auto parsed = HcclSim::Storage::ParseStorageDeployConfigText(config);
        EXPECT_FALSE(parsed.ok()) << "value=[" << value << "] 必须被拒绝";
        if (!parsed.ok()) {
            EXPECT_EQ(parsed.code, HcclSim::Storage::DbCode::INVALID_ARGUMENT) << "value=[" << value << "]";
        }
    }
    // 边界值 [1, 3600] 闭区间合法
    for (const std::string& value : {"1", "3600"}) {
        const std::string config = R"YAML(storage:
  backend: mariadb
  mariadb:
    uri: mariadb://hccl:secret@127.0.0.1:3306
    packet_read_timeout_s: )YAML" + value
                                   + "\n";
        auto parsed = HcclSim::Storage::ParseStorageDeployConfigText(config);
        EXPECT_TRUE(parsed.ok()) << "value=[" << value << "] 边界值必须被接受: " << parsed.diagnostic;
    }
    // uri 非标量 / mariadb 段非 map：装配期失败
    EXPECT_FALSE(
        HcclSim::Storage::ParseStorageDeployConfigText("storage:\n  backend: mariadb\n  mariadb:\n    uri: [a, b]\n")
            .ok());
    EXPECT_FALSE(HcclSim::Storage::ParseStorageDeployConfigText("storage:\n  backend: mariadb\n  mariadb: 7\n").ok());
}

/// 已移除的 Redis 后端必须明确报错：backend=redis 或残留 storage.redis 段
/// 都在解析期失败（不静默回退 SQLite/MariaDB）；backend=sqlite 混入坏
/// mariadb 段同样失败（不静默吞掉错误字段）
TEST(MariadbEngineConfig, RejectsRemovedRedisBackendAndSections)
{
    // backend=redis：后端已删除，明确报错。
    const std::string redisBackend = R"YAML(storage:
  backend: redis
  redis:
    endpoints:
      - id: 0
        uri: redis://127.0.0.1:6379
)YAML";
    auto parsedBackend = HcclSim::Storage::ParseStorageDeployConfigText(redisBackend);
    ASSERT_FALSE(parsedBackend.ok());
    EXPECT_EQ(parsedBackend.code, HcclSim::Storage::DbCode::INVALID_ARGUMENT);
    EXPECT_NE(parsedBackend.diagnostic.find("redis"), std::string::npos);

    // backend=mariadb 混入残留 storage.redis 段：同样明确报错。
    const std::string mariadbWithRedis = R"YAML(storage:
  backend: mariadb
  mariadb:
    uri: mariadb://hccl:secret@127.0.0.1:3306
  redis:
    endpoints:
      - id: 0
        uri: redis://127.0.0.1:6379
)YAML";
    auto parsedSection = HcclSim::Storage::ParseStorageDeployConfigText(mariadbWithRedis);
    ASSERT_FALSE(parsedSection.ok());
    EXPECT_EQ(parsedSection.code, HcclSim::Storage::DbCode::INVALID_ARGUMENT);
    EXPECT_NE(parsedSection.diagnostic.find("redis"), std::string::npos);

    // backend=sqlite 混入坏 mariadb 段：未选中的段仍需类型正确。
    EXPECT_FALSE(HcclSim::Storage::ParseStorageDeployConfigText(
                     "storage:\n  backend: sqlite\n  mariadb:\n    uri: not-a-valid-uri\n")
                     .ok());

    // backend=sqlite 正常解析：完整树（sqlite 模块在）下选中后端名为 sqlite；
    // 裁剪掉 sqlite 模块的源码树中按未知后端明确失败（裁剪版配置规则）。
    auto plain = HcclSim::Storage::ParseStorageDeployConfigText("storage:\n  backend: sqlite\n");
    if (plain.ok()) {
        EXPECT_EQ(plain.value->backend, std::string{"sqlite"});
    } else {
        EXPECT_NE(plain.diagnostic.find("unknown storage backend"), std::string::npos)
            << "裁剪树必须明确报未知后端: " << plain.diagnostic;
    }
}

// ---- 历史位置恢复裁定形状（回调合同回归）----
// MariaDB 成功不换库的合同形状：外层载荷必须存在（DbResult::value 有值）、
// 内层 optional 为空。成功而缺外层载荷会被公共层按回调合同违规拒绝
// （storage_composition.cc 的 ReconfigureDatabaseToRecordedLocation 消费者）。
TEST(MariadbRecordedLocationContract, SuccessKeepRouteCarriesOuterPayloadWithEmptyInner)
{
    const sim::backends::BackendModule& module = sim::backends::MariadbBackendModule();
    sim::backends::BackendConfigureRequest request; // 无配置段、未选中（不连接服务端）
    const auto configured = module.configure(request);
    ASSERT_TRUE(configured.ok()) << configured.diagnostic;
    ASSERT_TRUE(configured.value.has_value());
    const sim::backends::BackendDeployment& deployment = *configured.value;
    ASSERT_TRUE(deployment.resolveRecordedLocation);

    const HcclSim::Storage::LogicalDatabaseRuntimeConfig currentRoute{
        "RUNNER_DB", sim::backends::kMariadbBackendName, "mariadb://hccl:hccl-pass@127.0.0.1:13316"};
    const auto resolved = deployment.resolveRecordedLocation(currentRoute, "/tmp/legacy-recorded.sqlite.db");
    EXPECT_TRUE(resolved.ok());
    // 外层载荷存在 + 内层空 = 保持当前部署路由
    EXPECT_TRUE(resolved.value.has_value()) << "成功裁定必须携带外层载荷（回调合同）";
    EXPECT_FALSE((*resolved.value).has_value()) << "内层空 = 不换库（保持路由）";
}
// ---- 连接复活（errno 2006/2013 族有界重试）----
// 1024p swap 冻结实测形状：服务端网络超时掐断被冻结客户端的连接，客户端下
// 一笔语句收到 Server has gone away。本组用例在服务端 KILL 引擎连接复现：
// autocommit DML/SELECT 应透明重建连接并重试成功；事务路径不重试（失败按
// 合同交 Transaction 统一回滚）。

namespace {
    /// 解析 endpoint uri（mariadb://user:password@host:port）供测试自建管理连接
    struct ReviveTestEndpoint {
        std::string host;
        std::string user;
        std::string password;
        unsigned int port{0};
    };
    bool ParseReviveEndpoint(const std::string& uri, ReviveTestEndpoint& out)
    {
        const std::string scheme = "mariadb://";
        if (uri.rfind(scheme, 0) != 0) {
            return false;
        }
        const auto at = uri.find('@', scheme.size());
        const auto colon = uri.find(':', scheme.size());
        if (at == std::string::npos || colon == std::string::npos || colon > at) {
            return false;
        }
        out.user = uri.substr(scheme.size(), colon - scheme.size());
        out.password = uri.substr(colon + 1, at - colon - 1);
        const std::string hostPort = uri.substr(at + 1);
        const auto portColon = hostPort.rfind(':');
        if (portColon == std::string::npos) {
            return false;
        }
        out.host = hostPort.substr(0, portColon);
        out.port = static_cast<unsigned int>(std::stoul(hostPort.substr(portColon + 1)));
        return !out.host.empty() && out.port != 0;
    }

    /// 服务端 KILL 引擎当前连接：同账号下排除自身的连接即引擎连接（本用例期间
    /// 只有 fixture 引擎与本管理连接存活；gtest 顺序执行保证）。返回是否杀到。
    bool KillEngineConnectionsForRevive()
    {
        ReviveTestEndpoint ep{};
        if (!ParseReviveEndpoint(std::string{std::getenv(kMariadbEndpointEnv)}, ep)) {
            return false;
        }
        MYSQL* admin = mysql_init(nullptr);
        if (admin == nullptr) {
            return false;
        }
        if (mysql_real_connect(
                admin, ep.host.c_str(), ep.user.c_str(), ep.password.c_str(), nullptr, ep.port, nullptr, 0)
            == nullptr) {
            mysql_close(admin);
            return false;
        }
        const std::string query = "SELECT ID FROM information_schema.PROCESSLIST "
                                  "WHERE USER = '"
                                  + ep.user + "' AND ID <> CONNECTION_ID()";
        if (mysql_query(admin, query.c_str()) != 0) {
            mysql_close(admin);
            return false;
        }
        MYSQL_RES* res = mysql_store_result(admin);
        if (res == nullptr) {
            mysql_close(admin);
            return false;
        }
        std::vector<unsigned long> ids;
        for (MYSQL_ROW row = mysql_fetch_row(res); row != nullptr; row = mysql_fetch_row(res)) {
            if (row[0] != nullptr) {
                ids.push_back(std::stoul(row[0]));
            }
        }
        mysql_free_result(res);
        for (unsigned long id : ids) {
            const std::string kill = "KILL " + std::to_string(id);
            (void)mysql_query(admin, kill.c_str());
        }
        mysql_close(admin);
        return !ids.empty();
    }
} // namespace

TEST(MariadbEngineRevive, ServerSideKillRecoversForAutocommitDmlAndSelect)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "连接复活用例需要外部 mysqld endpoint";
    }
    EngineFixture fixture{"revive_autocommit"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    AllTypesRow before{};
    before.signed64 = 11;
    before.text = "before-kill";
    const auto insertedBefore = fixture.engine->Insert(meta, TableScope::Static(), &before, nullptr);
    ASSERT_TRUE(insertedBefore.ok()) << insertedBefore.diagnostic;

    ASSERT_TRUE(KillEngineConnectionsForRevive()) << "服务端 KILL 引擎连接失败";

    // KILL 后首笔 autocommit DML：应收到 2006 族并透明复活重试成功
    AllTypesRow after{};
    after.signed64 = 22;
    after.text = "after-kill";
    const auto insertedAfter = fixture.engine->Insert(meta, TableScope::Static(), &after, nullptr);
    ASSERT_TRUE(insertedAfter.ok()) << insertedAfter.diagnostic;
    ASSERT_GT(*insertedAfter.value, *insertedBefore.value);

    // 复活后的读路径：SELECT 与 COUNT 均正常
    const auto where = ConditionData{ConditionClause{
        static_cast<uint16_t>(0), Compare::EQUAL, ConditionJoin::NONE, 0, 0,
        ConditionValue{DbScalar{*insertedAfter.value}}}};
    std::vector<AllTypesRow> rows;
    auto sink = ResultSink::For(rows);
    const auto selected = fixture.engine->Select(meta, TableScope::Static(), where, QueryOptions{}, sink);
    ASSERT_TRUE(selected.ok()) << selected.diagnostic;
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].text, "after-kill");
    const auto counted = fixture.engine->Count(meta, TableScope::Static(), where);
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    EXPECT_EQ(*counted.value, 1U);

    // 读路径复活：再次 KILL 后的 SELECT 也应透明重试
    ASSERT_TRUE(KillEngineConnectionsForRevive());
    std::vector<AllTypesRow> rowsAfterKill;
    auto sinkAfterKill = ResultSink::For(rowsAfterKill);
    const auto selectedAfterKill
        = fixture.engine->Select(meta, TableScope::Static(), where, QueryOptions{}, sinkAfterKill);
    ASSERT_TRUE(selectedAfterKill.ok()) << selectedAfterKill.diagnostic;
    EXPECT_EQ(rowsAfterKill.size(), 1U);
}

TEST(MariadbEngineRevive, ServerSideKillInsideTransactionDoesNotRetry)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "连接复活用例需要外部 mysqld endpoint";
    }
    EngineFixture fixture{"revive_in_transaction"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    TransactionBinding transaction;
    ASSERT_TRUE(fixture.engine->Begin(transaction).ok());
    ASSERT_TRUE(KillEngineConnectionsForRevive());

    // 事务路径不复活重试：DML 失败如实返回（失败回滚由 Transaction 统一负责）
    AllTypesRow row{};
    row.signed64 = 33;
    row.text = "in-transaction-after-kill";
    const auto inserted = fixture.engine->Insert(meta, TableScope::Static(), &row, &transaction);
    EXPECT_FALSE(inserted.ok());
}

// ---- SELECT 错误处理专项（prepare 真实错误保存 / prepare 断线有界复活 /
// fetch-drain 清理不变量）。1024p 现场形状：VMB 单行 SELECT 在连接被服务端
// 掐断后失败，且错误信息丢失（连接级 errno 0）→ 引擎终态关闭 → 后续全部
// 存储操作 INVALID_STATE 级联。----

namespace {
    /// 管理连接上执行单条文本 SQL（每次独立连接；本文件全部用例只操作
    /// HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 指向的 per-run 私有实例；
    /// 失败时把服务端错误写进 error 出参供断言输出）
    bool AdminExecSql(const std::string& sql, std::string* error = nullptr)
    {
        ReviveTestEndpoint ep{};
        if (!ParseReviveEndpoint(std::string{std::getenv(kMariadbEndpointEnv)}, ep)) {
            return false;
        }
        MYSQL* admin = mysql_init(nullptr);
        if (admin == nullptr) {
            return false;
        }
        if (mysql_real_connect(
                admin, ep.host.c_str(), ep.user.c_str(), ep.password.c_str(), nullptr, ep.port, nullptr, 0)
            == nullptr) {
            if (error != nullptr) {
                *error = "connect failed: " + std::string{mysql_error(admin)};
            }
            mysql_close(admin);
            return false;
        }
        const bool ok = mysql_query(admin, sql.c_str()) == 0;
        if (!ok && error != nullptr) {
            *error = std::string{mysql_error(admin)};
        }
        mysql_close(admin);
        return ok;
    }

    /// 管理连接上执行文本查询并取回全部行（独立连接）
    bool AdminQueryRows(const std::string& sql, std::vector<std::vector<std::string>>& rows)
    {
        ReviveTestEndpoint ep{};
        if (!ParseReviveEndpoint(std::string{std::getenv(kMariadbEndpointEnv)}, ep)) {
            return false;
        }
        MYSQL* admin = mysql_init(nullptr);
        if (admin == nullptr) {
            return false;
        }
        if (mysql_real_connect(
                admin, ep.host.c_str(), ep.user.c_str(), ep.password.c_str(), nullptr, ep.port, nullptr, 0)
            == nullptr) {
            mysql_close(admin);
            return false;
        }
        if (mysql_query(admin, sql.c_str()) != 0) {
            mysql_close(admin);
            return false;
        }
        MYSQL_RES* result = mysql_store_result(admin);
        if (result == nullptr) {
            mysql_close(admin);
            return false;
        }
        rows.clear();
        for (MYSQL_ROW row = mysql_fetch_row(result); row != nullptr; row = mysql_fetch_row(result)) {
            std::vector<std::string> values(mysql_num_fields(result));
            for (unsigned int i = 0; i < values.size(); ++i) {
                values[i] = row[i] == nullptr ? std::string{} : std::string{row[i]};
            }
            rows.push_back(std::move(values));
        }
        mysql_free_result(result);
        mysql_close(admin);
        return true;
    }

    /// 长驻管理连接（LOCK TABLES 等需要跨语句持锁的用例；析构断连即释放锁）
    struct AdminSession {
        MYSQL* mysql{nullptr};
        AdminSession()
        {
            ReviveTestEndpoint ep{};
            if (!ParseReviveEndpoint(std::string{std::getenv(kMariadbEndpointEnv)}, ep)) {
                return;
            }
            mysql = mysql_init(nullptr);
            if (mysql == nullptr) {
                return;
            }
            if (mysql_real_connect(
                    mysql, ep.host.c_str(), ep.user.c_str(), ep.password.c_str(), nullptr, ep.port, nullptr, 0)
                == nullptr) {
                mysql_close(mysql);
                mysql = nullptr;
            }
        }
        ~AdminSession()
        {
            if (mysql != nullptr) {
                mysql_close(mysql);
            }
        }
        bool Ok() const { return mysql != nullptr; }
        bool Query(const std::string& sql) { return mysql != nullptr && mysql_query(mysql, sql.c_str()) == 0; }
    };

    /// 定位 fixture 引擎的物理 database（hcclvm_<ns>_<dbId>；经 information_schema
    /// 发现，不复制派生规则。同一实例上每个 fixture 各建一个含 AllTypesRow 的库，
    /// 按本 fixture 的唯一 namespace 定位）
    bool FindFixtureSchema(const std::string& dataNamespace, std::string& schema)
    {
        std::vector<std::vector<std::string>> rows;
        if (!AdminQueryRows(
                "SELECT DISTINCT TABLE_SCHEMA FROM information_schema.tables "
                "WHERE TABLE_SCHEMA LIKE 'hcclvm\\_%' AND TABLE_NAME = 'AllTypesRow'",
                rows)) {
            return false;
        }
        for (const auto& row : rows) {
            if (!row.empty() && row[0].find(dataNamespace) != std::string::npos) {
                schema = row[0];
                return true;
            }
        }
        return false;
    }

    /// 等待 pid 进程退出（SIGTERM 后的收敛确认；超时返回 false）
    bool WaitProcessGone(pid_t pid, int timeoutMs)
    {
        for (int waited = 0; waited < timeoutMs; waited += 50) {
            if (::kill(pid, 0) != 0 && errno == ESRCH) {
                return true;
            }
            ::usleep(50 * 1000);
        }
        return ::kill(pid, 0) != 0 && errno == ESRCH;
    }
} // namespace

// 2026-09-20 VMB purge 复合索引（真实 kRunnerDatabase 目录——DDL 唯一事实源在
// db_sim_runner_ops.h，本用例不重复声明表结构）：(device_id, is_freed) 物理索引
// 存在、列序正确、派生名与命名公式一致（idx_<tableId>_<col...>，与 SQLite 版
// DerivedIndexName 同构）；EXPLAIN DELETE 对 purge 谓词（两列等值）选中该索引
// 而非全表扫描；引擎侧 purge 形状 DELETE 行为不变（只删本设备标记行，在用行
// 与其他设备不受影响）。
TEST(MariadbEngineSemantics, VmbPurgeIndexDerivedNameAndDeletePlan)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "VMB purge 索引用例需要外部 mysqld endpoint";
    }
    const std::string ns = "vmbpurge_p" + std::to_string(static_cast<long>(::getpid()));
    MariaDbEngine engine{&kRunnerDatabase, TestProfile()};
    EngineOpenRequest request;
    request.mode = StorageOpenMode::READ_WRITE_CREATE;
    request.dataNamespace = ns;
    ASSERT_TRUE(engine.Open(request).ok());
    ASSERT_TRUE(engine.Prepare().ok());

    const Ddl::CatalogTableMeta* vmb = nullptr;
    for (std::size_t i = 0; i < kRunnerDatabase.tableCount; ++i) {
        if (std::string_view{kRunnerDatabase.tables[i].tableId} == std::string_view{"VirtualMemBlock"}) {
            vmb = &kRunnerDatabase.tables[i];
            break;
        }
    }
    ASSERT_NE(vmb, nullptr) << "VirtualMemBlock missing from kRunnerDatabase";

    // 造数：设备 7 大量在用行（is_freed=0）+ 3 行标记行（is_freed=1），设备 8
    // 各一行——行数足以让优化器对 purge 谓词选择索引而非小表全扫
    const auto addVmb = [&engine, vmb](uint64_t device, uint8_t freed, uint64_t seq) {
        sim::runtime::VirtualMemBlock row{};
        row.start_ptr = 0x0010000000000000ULL + seq * 0x2000ULL;
        row.dev_mapped_ptr = row.start_ptr;
        row.size = 0x1000ULL;
        row.device_id = device;
        row.rank_id = device;
        row.src_type = static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV);
        row.is_freed = freed;
        const auto added = engine.Insert(*vmb, TableScope::Static(), &row, nullptr);
        if (!added.ok()) {
            ADD_FAILURE() << added.diagnostic;
            return false;
        }
        return true;
    };
    for (uint64_t seq = 0; seq < 1500; ++seq) {
        ASSERT_TRUE(addVmb(7, 0, seq));
    }
    ASSERT_TRUE(addVmb(7, 1, 5000));
    ASSERT_TRUE(addVmb(7, 1, 5001));
    ASSERT_TRUE(addVmb(7, 1, 5002));
    ASSERT_TRUE(addVmb(8, 0, 5003));
    ASSERT_TRUE(addVmb(8, 1, 5004));

    // 定位本 namespace 的物理 database（经 information_schema 发现，不复制
    // DeriveDatabaseName 派生规则）
    std::string schema;
    {
        std::vector<std::vector<std::string>> rows;
        ASSERT_TRUE(AdminQueryRows(
            "SELECT DISTINCT TABLE_SCHEMA FROM information_schema.tables "
            "WHERE TABLE_SCHEMA LIKE 'hcclvm\\_%' AND TABLE_NAME = 'VirtualMemBlock'",
            rows));
        for (const auto& row : rows) {
            if (!row.empty() && row[0].find(ns) != std::string::npos) {
                schema = row[0];
                break;
            }
        }
    }
    ASSERT_FALSE(schema.empty()) << "physical database for " << ns << " not found";

    // 物理索引存在：列序 (device_id, is_freed)，派生名与公式一致
    {
        std::vector<std::vector<std::string>> rows;
        ASSERT_TRUE(AdminQueryRows(
            "SELECT SEQ_IN_INDEX, COLUMN_NAME FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = '"
                + schema
                + "' AND TABLE_NAME = 'VirtualMemBlock' "
                  "AND INDEX_NAME = 'idx_VirtualMemBlock_device_id_is_freed' "
                  "ORDER BY SEQ_IN_INDEX",
            rows));
        ASSERT_EQ(rows.size(), 2U) << "purge 索引缺失或列数不符";
        EXPECT_EQ(rows[0][1], "device_id");
        EXPECT_EQ(rows[1][1], "is_freed");
    }

    // EXPLAIN DELETE：purge 谓词（device_id 等值 + is_freed 等值）选中该索引
    // （EXPLAIN 列序：id,select_type,table,type,possible_keys,key,...）
    {
        std::vector<std::vector<std::string>> rows;
        ASSERT_TRUE(AdminQueryRows(
            "EXPLAIN DELETE FROM `" + schema
                + "`.`VirtualMemBlock` "
                  "WHERE device_id = 7 AND is_freed = 1",
            rows));
        ASSERT_EQ(rows.size(), 1U);
        EXPECT_EQ(rows[0][5], "idx_VirtualMemBlock_device_id_is_freed")
            << "EXPLAIN key=" << rows[0][5] << " possible_keys=" << rows[0][4];
    }

    // 引擎 purge 形状 DELETE（aclrt_memory_stub.cc 的谓词形态）：只删设备 7 的
    // 3 行标记行；设备 7 的 1500 行在用行与设备 8 的 2 行不受影响
    const auto purged = engine.Delete(
        *vmb, TableScope::Static(),
        And(Eq(&sim::runtime::VirtualMemBlock::device_id, uint64_t{7}),
            Eq(&sim::runtime::VirtualMemBlock::is_freed, uint64_t{1}))
            .data,
        nullptr);
    ASSERT_TRUE(purged.ok()) << purged.diagnostic;
    EXPECT_EQ(*purged.value, 3U);
    const auto device7
        = engine.Count(*vmb, TableScope::Static(), Eq(&sim::runtime::VirtualMemBlock::device_id, uint64_t{7}).data);
    ASSERT_TRUE(device7.ok()) << device7.diagnostic;
    EXPECT_EQ(*device7.value, 1500U);
    const auto device8
        = engine.Count(*vmb, TableScope::Static(), Eq(&sim::runtime::VirtualMemBlock::device_id, uint64_t{8}).data);
    ASSERT_TRUE(device8.ok()) << device8.diagnostic;
    EXPECT_EQ(*device8.value, 2U);

    ASSERT_TRUE(engine.Close().ok());
}

// 2026-09-21 Derived 端点回归（真实 kRunnerDatabase 目录——DDL 唯一事实源在
// db_sim_runner_ops.h，本用例不重复声明表结构）：VMB 五个区间形状联合索引的
// 末两列端点 start_end_ptr / dev_mapped_end_ptr 是 Derived 字段
// （RecordCodecHooks<VirtualMemBlock> 编码时按 base+size 计算入库，解码忽略），
// 端点类型合同只要求互异 uint64、不排除 Derived（BASE 的区间索引校验从未排除
// Derived 端点）。回归锚点：曾错误地在端点检查中加入"非 Derived"限制，使五个
// 索引全部失去 FORCE INDEX 提示——本用例对每个形状以"DROP 物理索引后匹配查询
// 必须失败"取证提示确实生成且指向该索引（修复前：提示缺失，DROP 后查询仍按
// 默认计划成功，本用例失败）。谓词形状对应业务调用点：1/2=FindDevMappedVirMem
// ByDevice（device/rank 前缀 + dev_mapped 区间）、3=按 device 的 start 区间、
// 4/5=IsDeviceAddress 形状（仅 src_type 前导的 start / dev_mapped 区间）。
TEST(MariadbEngineSemantics, VmbIntervalIndexSelectionDerivedEndpoints)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "VMB 区间索引用例需要外部 mysqld endpoint";
    }
    const std::string ns = "vmbinterval_p" + std::to_string(static_cast<long>(::getpid()));
    MariaDbEngine engine{&kRunnerDatabase, TestProfile()};
    EngineOpenRequest request;
    request.mode = StorageOpenMode::READ_WRITE_CREATE;
    request.dataNamespace = ns;
    ASSERT_TRUE(engine.Open(request).ok());
    ASSERT_TRUE(engine.Prepare().ok());

    const Ddl::CatalogTableMeta* vmb = nullptr;
    for (std::size_t i = 0; i < kRunnerDatabase.tableCount; ++i) {
        if (std::string_view{kRunnerDatabase.tables[i].tableId} == std::string_view{"VirtualMemBlock"}) {
            vmb = &kRunnerDatabase.tables[i];
            break;
        }
    }
    ASSERT_NE(vmb, nullptr) << "VirtualMemBlock missing from kRunnerDatabase";
    const auto& meta = *vmb;

    // fieldId 按名定位（不硬编码列序；字段表见 db_sim_runner_ops.h）
    const auto fieldIdOf = [&meta](std::string_view name) -> uint16_t {
        for (std::size_t f = 0; f < meta.fieldCount; ++f) {
            if (std::string_view{meta.fields[f].name} == name) {
                return static_cast<uint16_t>(f);
            }
        }
        ADD_FAILURE() << "VMB field missing: " << name;
        return 0;
    };
    const uint16_t kId = fieldIdOf("id");
    const uint16_t kStart = fieldIdOf("start_ptr");
    const uint16_t kDevMapped = fieldIdOf("dev_mapped_ptr");
    const uint16_t kDevice = fieldIdOf("device_id");
    const uint16_t kRank = fieldIdOf("rank_id");
    const uint16_t kSrcType = fieldIdOf("src_type");
    const uint16_t kStartEnd = fieldIdOf("start_end_ptr");
    const uint16_t kDevMappedEnd = fieldIdOf("dev_mapped_end_ptr");
    // 回归前提：两个区间端点必须带 Derived 属性——否则本用例退化为普通端点
    // 目录，抓不住"端点误排除 Derived"缺陷
    ASSERT_TRUE(Ddl::HasProp(meta.fields[kStartEnd].props, Ddl::Prop::Derived)) << "start_end_ptr must stay Derived";
    ASSERT_TRUE(Ddl::HasProp(meta.fields[kDevMappedEnd].props, Ddl::Prop::Derived))
        << "dev_mapped_end_ptr must stay Derived";

    // 造数：start 与 dev_mapped 基址分离（两组端点可区分）；重叠/边界衔接/
    // 孤立区间；src_type 与 device/rank 隔离。区间端点不手工赋值——由
    // RecordCodecHooks 按 base+size 计算入库，匹配查询的非空/边界结果即证明
    // 派生端点确实落库并参与谓词。
    struct Spec {
        uint64_t device;
        uint64_t rank;
        uint8_t src;
        uint64_t start;
        uint64_t devMapped;
        uint64_t size;
    };
    const Spec specs[] = {
        {7, 70, sim::runtime::VIR_MEM_TYPE_DEV, 0x1000, 0x5000, 0x100},
        {7, 70, sim::runtime::VIR_MEM_TYPE_DEV, 0x1080, 0x5080, 0x100},
        {7, 70, sim::runtime::VIR_MEM_TYPE_DEV, 0x1100, 0x5100, 0x100},
        {7, 70, sim::runtime::VIR_MEM_TYPE_DEV, 0x8000, 0x9000, 0x100},
        {7, 70, sim::runtime::VIR_MEM_TYPE_HOST, 0x1000, 0x5000, 0x100},
        {8, 80, sim::runtime::VIR_MEM_TYPE_DEV, 0x1000, 0x5000, 0x100},
    };
    std::vector<uint64_t> ids;
    for (const auto& spec : specs) {
        sim::runtime::VirtualMemBlock row{};
        row.start_ptr = spec.start;
        row.dev_mapped_ptr = spec.devMapped;
        row.size = spec.size;
        row.device_id = spec.device;
        row.rank_id = spec.rank;
        row.src_type = spec.src;
        const auto inserted = engine.Insert(meta, TableScope::Static(), &row, nullptr);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
        ids.push_back(*inserted.value);
    }

    // 匹配形状构造：前导等值列（按形状为 device/rank/src_type 或仅 src_type）+
    // start ≤ addr < end 半开区间（值参数化）
    const auto eqClause = [](uint16_t field, uint64_t value, ConditionJoin join = ConditionJoin::NONE) {
        return Clause(field, Compare::EQUAL, ConditionValue{DbScalar{value}}, join);
    };
    const auto intervalWhere
        = [&](std::vector<ConditionClause> leadingEquals, uint16_t startField, uint16_t endField, uint64_t addr) {
              std::vector<ConditionClause> clauses{std::move(leadingEquals)};
              clauses.push_back(
                  Clause(startField, Compare::LESS_OR_EQUAL, ConditionValue{DbScalar{addr}}, ConditionJoin::AND));
              clauses.push_back(Clause(endField, Compare::GREATER, ConditionValue{DbScalar{addr}}, ConditionJoin::AND));
              return ConditionData{std::move(clauses)};
          };
    const auto byDeviceDevMapped = [&](uint64_t device, uint8_t src, uint64_t addr) {
        return intervalWhere(
            {eqClause(kDevice, device), eqClause(kSrcType, src, ConditionJoin::AND)}, kDevMapped, kDevMappedEnd, addr);
    };
    const auto byRankDevMapped = [&](uint64_t rank, uint8_t src, uint64_t addr) {
        return intervalWhere(
            {eqClause(kRank, rank), eqClause(kSrcType, src, ConditionJoin::AND)}, kDevMapped, kDevMappedEnd, addr);
    };
    const auto byDeviceStart = [&](uint64_t device, uint8_t src, uint64_t addr) {
        return intervalWhere(
            {eqClause(kDevice, device), eqClause(kSrcType, src, ConditionJoin::AND)}, kStart, kStartEnd, addr);
    };
    const auto byStartOnly = [&](uint8_t src, uint64_t addr) {
        return intervalWhere({eqClause(kSrcType, src)}, kStart, kStartEnd, addr);
    };
    const auto byDevMappedOnly = [&](uint8_t src, uint64_t addr) {
        return intervalWhere({eqClause(kSrcType, src)}, kDevMapped, kDevMappedEnd, addr);
    };

    const auto selectIds = [&](const ConditionData& where) -> std::set<uint64_t> {
        QueryOptions options;
        options.sorts.emplace_back(kId, Order::ASCENDING);
        std::vector<sim::runtime::VirtualMemBlock> rows;
        auto sink = ResultSink::For(rows);
        const auto selected = engine.Select(meta, TableScope::Static(), where, options, sink);
        EXPECT_TRUE(selected.ok()) << selected.diagnostic;
        std::set<uint64_t> hit;
        for (const auto& row : rows) {
            hit.insert(row.id);
        }
        return hit;
    };

    // —— 形状 1：(device_id, src_type, dev_mapped_ptr, dev_mapped_end_ptr) ——
    // 重叠区间（0x5090 同时命中 [5000,5100) 与 [5080,5180)）
    EXPECT_EQ(
        selectIds(byDeviceDevMapped(7, sim::runtime::VIR_MEM_TYPE_DEV, 0x5090)), (std::set<uint64_t>{ids[0], ids[1]}));
    // 左闭右开边界（0x5100 不属于 [5000,5100)，属于 [5080,5180) 与 [5100,5200)）
    EXPECT_EQ(
        selectIds(byDeviceDevMapped(7, sim::runtime::VIR_MEM_TYPE_DEV, 0x5100)), (std::set<uint64_t>{ids[1], ids[2]}));
    // 无命中（区间外两侧）
    EXPECT_TRUE(selectIds(byDeviceDevMapped(7, sim::runtime::VIR_MEM_TYPE_DEV, 0x6800)).empty());
    EXPECT_TRUE(selectIds(byDeviceDevMapped(7, sim::runtime::VIR_MEM_TYPE_DEV, 0x4ff0)).empty());
    // src_type / device 隔离
    EXPECT_EQ(selectIds(byDeviceDevMapped(7, sim::runtime::VIR_MEM_TYPE_HOST, 0x5090)), (std::set<uint64_t>{ids[4]}));
    EXPECT_EQ(selectIds(byDeviceDevMapped(8, sim::runtime::VIR_MEM_TYPE_DEV, 0x5090)), (std::set<uint64_t>{ids[5]}));
    // 业务取首语义（重叠取最小 id；limit=1）
    {
        QueryOptions options;
        options.sorts.emplace_back(kId, Order::ASCENDING);
        options.limit = 1U;
        std::vector<sim::runtime::VirtualMemBlock> rows;
        auto sink = ResultSink::For(rows);
        const auto selected = engine.Select(
            meta, TableScope::Static(), byDeviceDevMapped(7, sim::runtime::VIR_MEM_TYPE_DEV, 0x5090), options, sink);
        ASSERT_TRUE(selected.ok()) << selected.diagnostic;
        ASSERT_EQ(rows.size(), 1U);
        EXPECT_EQ(rows[0].id, ids[0]);
    }

    // —— 形状 2：(rank_id, src_type, dev_mapped_ptr, dev_mapped_end_ptr) ——
    EXPECT_EQ(
        selectIds(byRankDevMapped(70, sim::runtime::VIR_MEM_TYPE_DEV, 0x5090)), (std::set<uint64_t>{ids[0], ids[1]}));
    EXPECT_EQ(selectIds(byRankDevMapped(80, sim::runtime::VIR_MEM_TYPE_DEV, 0x5090)), (std::set<uint64_t>{ids[5]}));
    EXPECT_EQ(
        selectIds(byRankDevMapped(70, sim::runtime::VIR_MEM_TYPE_DEV, 0x5100)), (std::set<uint64_t>{ids[1], ids[2]}));

    // —— 形状 3：(device_id, src_type, start_ptr, start_end_ptr) ——
    EXPECT_EQ(
        selectIds(byDeviceStart(7, sim::runtime::VIR_MEM_TYPE_DEV, 0x1090)), (std::set<uint64_t>{ids[0], ids[1]}));
    EXPECT_EQ(
        selectIds(byDeviceStart(7, sim::runtime::VIR_MEM_TYPE_DEV, 0x1100)), (std::set<uint64_t>{ids[1], ids[2]}));
    EXPECT_TRUE(selectIds(byDeviceStart(7, sim::runtime::VIR_MEM_TYPE_DEV, 0x6900)).empty());

    // —— 形状 4：(src_type, start_ptr, start_end_ptr)（无 device/rank 前导，
    // 跨设备命中 start 区间）
    EXPECT_EQ(
        selectIds(byStartOnly(sim::runtime::VIR_MEM_TYPE_DEV, 0x1090)), (std::set<uint64_t>{ids[0], ids[1], ids[5]}));
    EXPECT_TRUE(selectIds(byStartOnly(sim::runtime::VIR_MEM_TYPE_DEV, 0x6900)).empty());

    // —— 形状 5：(src_type, dev_mapped_ptr, dev_mapped_end_ptr)（IsDeviceAddress）
    EXPECT_EQ(
        selectIds(byDevMappedOnly(sim::runtime::VIR_MEM_TYPE_DEV, 0x5090)),
        (std::set<uint64_t>{ids[0], ids[1], ids[5]}));
    EXPECT_TRUE(selectIds(byDevMappedOnly(sim::runtime::VIR_MEM_TYPE_DEV, 0x6800)).empty());

    // —— FORCE INDEX 取证（每个形状）：经 information_schema 按列序定位该形状
    // 的物理索引（索引名经 64 字符截断+哈希，不硬编码），DROP 后匹配查询必须
    // 失败（提示引用了缺失索引）；逐形状独立取证。
    std::string schema;
    {
        std::vector<std::vector<std::string>> rows;
        ASSERT_TRUE(AdminQueryRows(
            "SELECT DISTINCT TABLE_SCHEMA FROM information_schema.tables "
            "WHERE TABLE_SCHEMA LIKE 'hcclvm\\_%' AND TABLE_NAME = 'VirtualMemBlock'",
            rows));
        for (const auto& row : rows) {
            if (!row.empty() && row[0].find(ns) != std::string::npos) {
                schema = row[0];
                break;
            }
        }
    }
    ASSERT_FALSE(schema.empty()) << "physical database for " << ns << " not found";

    const auto indexNameOfColumns = [&schema](const std::string& columnsCsv) -> std::string {
        std::vector<std::vector<std::string>> rows;
        if (!AdminQueryRows(
                "SELECT INDEX_NAME FROM information_schema.STATISTICS "
                "WHERE TABLE_SCHEMA = '"
                    + schema
                    + "' AND TABLE_NAME = 'VirtualMemBlock' "
                      "GROUP BY INDEX_NAME HAVING GROUP_CONCAT(COLUMN_NAME ORDER BY "
                      "SEQ_IN_INDEX SEPARATOR ',') = '"
                    + columnsCsv + "'",
                rows)) {
            return {};
        }
        return rows.size() == 1U ? rows[0][0] : std::string{};
    };
    const auto dropIndex = [&schema](const std::string& indexName) {
        std::string adminError;
        return AdminQuery("DROP INDEX `" + indexName + "` ON `" + schema + "`.`VirtualMemBlock`", adminError);
    };
    const auto selectOk = [&](const ConditionData& where) {
        std::vector<sim::runtime::VirtualMemBlock> rows;
        auto sink = ResultSink::For(rows);
        const auto selected = engine.Select(meta, TableScope::Static(), where, QueryOptions{}, sink);
        return selected.ok();
    };

    struct ShapeEvidence {
        const char* columnsCsv;
        ConditionData where;
    };
    const ShapeEvidence shapes[] = {
        {"device_id,src_type,dev_mapped_ptr,dev_mapped_end_ptr",
         byDeviceDevMapped(7, sim::runtime::VIR_MEM_TYPE_DEV, 0x5090)},
        {"rank_id,src_type,dev_mapped_ptr,dev_mapped_end_ptr",
         byRankDevMapped(70, sim::runtime::VIR_MEM_TYPE_DEV, 0x5090)},
        {"device_id,src_type,start_ptr,start_end_ptr", byDeviceStart(7, sim::runtime::VIR_MEM_TYPE_DEV, 0x1090)},
        {"src_type,start_ptr,start_end_ptr", byStartOnly(sim::runtime::VIR_MEM_TYPE_DEV, 0x1090)},
        {"src_type,dev_mapped_ptr,dev_mapped_end_ptr", byDevMappedOnly(sim::runtime::VIR_MEM_TYPE_DEV, 0x5090)},
    };
    for (const auto& shape : shapes) {
        const std::string indexName = indexNameOfColumns(shape.columnsCsv);
        ASSERT_FALSE(indexName.empty()) << "physical index not found for columns: " << shape.columnsCsv;
        ASSERT_TRUE(dropIndex(indexName)) << "failed to drop index " << indexName;
        EXPECT_FALSE(selectOk(shape.where)) << "matched shape must reference the dropped interval index (" << indexName
                                            << "); a success means no FORCE INDEX hint was generated";
    }

    // —— 非匹配形状（五索引已全部 DROP，仍必须走默认计划成功——证明这些形状
    // 从未生成提示）。注：与单索引 IntervalRow 目录不同，VMB 全目录下"缺前导
    // 等值"的 3 子句形态恰是形状 4/5 的合法匹配形状（获得各自提示），不能作为
    // 非匹配样例；非匹配语义由 IntervalQueryCompositeIndexSelection 覆盖。
    // OR 连接
    EXPECT_TRUE(selectOk(ConditionData{std::vector<ConditionClause>{
        eqClause(kDevice, 7), eqClause(kSrcType, sim::runtime::VIR_MEM_TYPE_DEV, ConditionJoin::AND),
        Clause(kDevMapped, Compare::LESS_OR_EQUAL, ConditionValue{DbScalar{uint64_t{0x5090}}}, ConditionJoin::AND),
        Clause(kDevMappedEnd, Compare::GREATER, ConditionValue{DbScalar{uint64_t{0x5090}}}, ConditionJoin::OR)}}));
    // 反向区间（start > addr 且 end ≤ addr）
    EXPECT_TRUE(selectOk(ConditionData{std::vector<ConditionClause>{
        eqClause(kDevice, 7), eqClause(kSrcType, sim::runtime::VIR_MEM_TYPE_DEV, ConditionJoin::AND),
        Clause(kDevMapped, Compare::GREATER, ConditionValue{DbScalar{uint64_t{0x5090}}}, ConditionJoin::AND),
        Clause(
            kDevMappedEnd, Compare::LESS_OR_EQUAL, ConditionValue{DbScalar{uint64_t{0x5090}}}, ConditionJoin::AND)}}));
    // 多余子句（5 子句超出全部索引列数）
    EXPECT_TRUE(selectOk(ConditionData{std::vector<ConditionClause>{
        eqClause(kDevice, 7), eqClause(kSrcType, sim::runtime::VIR_MEM_TYPE_DEV, ConditionJoin::AND),
        Clause(kDevMapped, Compare::LESS_OR_EQUAL, ConditionValue{DbScalar{uint64_t{0x5090}}}, ConditionJoin::AND),
        Clause(kDevMappedEnd, Compare::GREATER, ConditionValue{DbScalar{uint64_t{0x5090}}}, ConditionJoin::AND),
        Clause(kId, Compare::GREATER, ConditionValue{DbScalar{uint64_t{0}}}, ConditionJoin::AND)}}));

    ASSERT_TRUE(engine.Close().ok());
}

// ---- 线路协议压缩（wire_protocol_compression；Open/Reconnect/Wipe 三条建连
//      路径共用 ApplyMariadbConnectionOptions，经透明代理观测握手能力位）----

/// MySQL/MariaDB 线路协议里，客户端在握手响应包（客户端首包）声明能力位，
/// CLIENT_COMPRESS = 0x20；压缩从握手完成后才生效，首包恒为普通格式
/// （3 字节长度 + 1 字节序号，随后 4 字节小端能力位）。
class WireCompressionProbe {
public:
    WireCompressionProbe() = default;
    ~WireCompressionProbe() { Stop(); }
    WireCompressionProbe(const WireCompressionProbe&) = delete;
    WireCompressionProbe& operator=(const WireCompressionProbe&) = delete;

    bool Start(const std::string& upstreamHost, unsigned int upstreamPort)
    {
        listenFd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listenFd_ < 0) {
            return false;
        }
        int reuse = 1;
        (void)::setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in bindAddr{};
        bindAddr.sin_family = AF_INET;
        bindAddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        bindAddr.sin_port = 0; // 内核分配空闲端口
        if (::bind(listenFd_, reinterpret_cast<sockaddr*>(&bindAddr), sizeof(bindAddr)) != 0
            || ::listen(listenFd_, 8) != 0) {
            return false;
        }
        sockaddr_in bound{};
        socklen_t boundLen = sizeof(bound);
        if (::getsockname(listenFd_, reinterpret_cast<sockaddr*>(&bound), &boundLen) != 0) {
            return false;
        }
        port_ = ntohs(bound.sin_port);
        upstreamHost_ = upstreamHost;
        upstreamPort_ = upstreamPort;
        acceptor_ = std::thread([this] {
            AcceptLoop();
        });
        return true;
    }

    unsigned int Port() const { return port_; }

    /// 每个已见连接的握手能力位（连接建立序；客户端首包解析完成后追加）
    std::vector<uint32_t> ClientFlags() const
    {
        std::lock_guard<std::mutex> lock(flagsMutex_);
        return clientFlags_;
    }

    void Stop()
    {
        if (listenFd_ >= 0) {
            (void)::shutdown(listenFd_, SHUT_RDWR);
            (void)::close(listenFd_);
            listenFd_ = -1;
        }
        if (acceptor_.joinable()) {
            acceptor_.join();
        }
        std::vector<Relay> relays;
        {
            std::lock_guard<std::mutex> lock(relaysMutex_);
            relays = std::move(relays_); // Relay 含 thread 不可拷贝，整向量移动
        }
        for (Relay& relay : relays) {
            if (relay.downFd >= 0) {
                (void)::shutdown(relay.downFd, SHUT_RDWR);
            }
            if (relay.upFd >= 0) {
                (void)::shutdown(relay.upFd, SHUT_RDWR);
            }
        }
        for (Relay& relay : relays) {
            if (relay.upPump.joinable()) {
                relay.upPump.join();
            }
            if (relay.downPump.joinable()) {
                relay.downPump.join();
            }
            if (relay.downFd >= 0) {
                (void)::close(relay.downFd);
            }
            if (relay.upFd >= 0) {
                (void)::close(relay.upFd);
            }
        }
    }

private:
    struct Relay {
        int downFd{-1}; ///< 引擎（客户端）侧
        int upFd{-1};   ///< 真实服务端侧
        std::thread upPump;
        std::thread downPump;
    };

    void AcceptLoop()
    {
        while (true) {
            sockaddr_in peer{};
            socklen_t peerLen = sizeof(peer);
            const int downFd = ::accept(listenFd_, reinterpret_cast<sockaddr*>(&peer), &peerLen);
            if (downFd < 0) {
                return; // listen 套接字已关闭（Stop）
            }
            const int upFd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (upFd < 0) {
                (void)::close(downFd);
                continue;
            }
            sockaddr_in upstream{};
            upstream.sin_family = AF_INET;
            upstream.sin_port = htons(upstreamPort_);
            if (::inet_pton(AF_INET, upstreamHost_.c_str(), &upstream.sin_addr) != 1
                || ::connect(upFd, reinterpret_cast<sockaddr*>(&upstream), sizeof(upstream)) != 0) {
                (void)::close(downFd);
                (void)::close(upFd);
                continue;
            }
            Relay relay;
            relay.downFd = downFd;
            relay.upFd = upFd;
            relay.upPump = std::thread([this, downFd, upFd] {
                PumpUpstream(downFd, upFd);
            });
            relay.downPump = std::thread([this, downFd, upFd] {
                PumpDown(upFd, downFd);
            });
            {
                std::lock_guard<std::mutex> lock(relaysMutex_);
                relays_.push_back(std::move(relay));
            }
        }
    }

    /// 客户端 → 服务端：首个包是握手响应，解析能力位后再转发（不足 8 字节
    /// 继续累积，不提前转发保证解析原子性）
    void PumpUpstream(int fromFd, int toFd)
    {
        std::string pending;
        bool first = true;
        char buffer[8192];
        while (true) {
            const ssize_t received = ::recv(fromFd, buffer, sizeof(buffer), 0);
            if (received <= 0) {
                break;
            }
            if (first) {
                pending.append(buffer, static_cast<std::size_t>(received));
                if (pending.size() < 8U) {
                    continue;
                }
                const auto flagByte = [&pending](std::size_t index) {
                    return static_cast<uint32_t>(static_cast<unsigned char>(pending[index]));
                };
                const uint32_t flags = flagByte(4) | (flagByte(5) << 8U) | (flagByte(6) << 16U) | (flagByte(7) << 24U);
                {
                    std::lock_guard<std::mutex> lock(flagsMutex_);
                    clientFlags_.push_back(flags);
                }
                first = false;
                if (!SendAll(toFd, pending.data(), pending.size())) {
                    break;
                }
                continue;
            }
            if (!SendAll(toFd, buffer, static_cast<std::size_t>(received))) {
                break;
            }
        }
        (void)::shutdown(toFd, SHUT_RDWR);
    }

    /// 服务端 → 客户端：纯字节转发
    void PumpDown(int fromFd, int toFd)
    {
        char buffer[8192];
        while (true) {
            const ssize_t received = ::recv(fromFd, buffer, sizeof(buffer), 0);
            if (received <= 0) {
                break;
            }
            if (!SendAll(toFd, buffer, static_cast<std::size_t>(received))) {
                break;
            }
        }
        (void)::shutdown(toFd, SHUT_RDWR);
    }

    static bool SendAll(int fd, const char* data, std::size_t size)
    {
        std::size_t sent = 0;
        while (sent < size) {
            const ssize_t written = ::send(fd, data + sent, size - sent, MSG_NOSIGNAL);
            if (written <= 0) {
                return false;
            }
            sent += static_cast<std::size_t>(written);
        }
        return true;
    }

    int listenFd_{-1};
    unsigned int port_{0};
    std::string upstreamHost_;
    unsigned int upstreamPort_{0};
    std::thread acceptor_;
    mutable std::mutex flagsMutex_;
    std::vector<uint32_t> clientFlags_;
    std::mutex relaysMutex_;
    std::vector<Relay> relays_;
};

/// CLIENT_COMPRESS 能力位（MySQL/MariaDB 线路协议握手响应）
constexpr uint32_t kClientCompressFlag = 0x20U;

/// 等待代理记录到 count 条握手能力位（泵线程异步解析，建连完成后极快到达；
/// 50 秒上限防悬挂）
bool WaitForClientFlagCount(const WireCompressionProbe& probe, std::size_t count)
{
    for (int waitedMs = 0; waitedMs < 50000; waitedMs += 10) {
        if (probe.ClientFlags().size() >= count) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return probe.ClientFlags().size() >= count;
}

/// 压缩观测 fixture：引擎经本地透明代理连接真实 endpoint（uri 指向代理端口），
/// 其余字段同 TestProfile；dataNamespace 记录供 Wipe 用例复用
struct CompressionFixture {
    WireCompressionProbe probe;
    std::unique_ptr<MariaDbEngine> engine;
    std::string dataNamespace;
    bool ready{false};
    std::string diagnostic;

    CompressionFixture(bool compression, const std::string& ns)
    {
        AdminEndpoint ep{};
        if (!ParseAdminEndpoint(std::string{std::getenv(kMariadbEndpointEnv)}, ep)) {
            diagnostic = "endpoint parse failed";
            return;
        }
        if (!probe.Start(ep.host, ep.port)) {
            diagnostic = "wire probe start failed";
            return;
        }
        auto profile = TestProfileWithCompression(compression);
        profile.uri = "mariadb://" + ep.user + ":" + ep.password + "@127.0.0.1:" + std::to_string(probe.Port());
        dataNamespace = ns + "_p" + std::to_string(static_cast<long>(::getpid()));
        engine = std::make_unique<MariaDbEngine>(&kMariadbTestDatabase, profile);
        EngineOpenRequest request;
        request.mode = StorageOpenMode::READ_WRITE_CREATE;
        request.dataNamespace = dataNamespace;
        const auto opened = engine->Open(request);
        if (!opened.ok()) {
            diagnostic = opened.diagnostic;
            return;
        }
        const auto prepared = engine->Prepare();
        if (!prepared.ok()) {
            diagnostic = prepared.diagnostic;
            return;
        }
        ready = true;
    }
    ~CompressionFixture()
    {
        if (engine != nullptr && ready) {
            (void)engine->Close();
        }
    }
};

/// 压缩开/关两种 profile 的真实 CRUD：握手能力位与配置一致（true 声明
/// CLIENT_COMPRESS，false 不声明），长 ASCII 字符串与非空 BLOB 逐字节一致
TEST(MariadbEngineCompression, CrudRoundtripMatchesCompressionConfig)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "线路协议压缩用例需要外部 mysqld endpoint";
    }
    for (const bool compression : {true, false}) {
        CompressionFixture fixture{compression, compression ? "wirecomp_on" : "wirecomp_off"};
        ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
        const auto& meta = MetaOf("AllTypesRow");

        ASSERT_TRUE(WaitForClientFlagCount(fixture.probe, 1U)) << "代理未捕获 Open 握手";
        const auto flags = fixture.probe.ClientFlags();
        ASSERT_EQ(flags.size(), 1U) << "引擎应恰好建立一条连接（Open）";
        EXPECT_EQ((flags[0] & kClientCompressFlag) != 0U, compression)
            << "握手能力位与 wire_protocol_compression=" << compression << " 不符";

        // CRUD：长 ASCII 字符串（517B，越过 SSO）+ 非空 BLOB（769B）逐字节读回
        AllTypesRow row{};
        row.signed64 = -7;
        std::string text(517, 'x');
        text[3] = 'A';
        std::vector<uint8_t> blob(769);
        for (std::size_t i = 0; i < blob.size(); ++i) {
            blob[i] = static_cast<uint8_t>(i % 251U);
        }
        row.text = text;
        row.blob = blob;
        const auto inserted = fixture.engine->Insert(meta, TableScope::Static(), &row, nullptr);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
        row.id = *inserted.value;

        const auto byId = ConditionData{
            ConditionClause{0, Compare::EQUAL, ConditionJoin::NONE, 0, 0, ConditionValue{DbScalar{row.id}}}};
        std::vector<AllTypesRow> rows;
        auto sink = ResultSink::For(rows);
        ASSERT_TRUE(fixture.engine->Select(meta, TableScope::Static(), byId, QueryOptions{}, sink).ok());
        ASSERT_EQ(rows.size(), 1U);
        EXPECT_EQ(rows[0].text, text) << "压缩=" << compression << " 下字符串逐字节一致";
        EXPECT_EQ(rows[0].blob, blob) << "压缩=" << compression << " 下 BLOB 逐字节一致";
        EXPECT_EQ(rows[0].signed64, -7);

        // UPDATE 后读回 + DELETE 清零
        std::string updatedText = text;
        updatedText[100] = 'Z';
        ASSERT_TRUE(fixture.engine
                        ->Update(
                            meta, TableScope::Static(), byId,
                            UpdateData{LogicalChange{7, ChangeKind::SET, DbScalar{updatedText}}}, nullptr)
                        .ok());
        std::vector<AllTypesRow> updatedRows;
        auto updatedSink = ResultSink::For(updatedRows);
        ASSERT_TRUE(fixture.engine->Select(meta, TableScope::Static(), byId, QueryOptions{}, updatedSink).ok());
        ASSERT_EQ(updatedRows.size(), 1U);
        EXPECT_EQ(updatedRows[0].text, updatedText);
        ASSERT_TRUE(fixture.engine->Delete(meta, TableScope::Static(), byId, nullptr).ok());
        const auto counted = fixture.engine->Count(meta, TableScope::Static(), byId);
        ASSERT_TRUE(counted.ok());
        EXPECT_EQ(*counted.value, 0U);

        ASSERT_TRUE(fixture.engine->Close().ok());
        fixture.ready = false; // 析构不再重复 Close
    }
}

/// 重连保持压缩配置：compression=false 的引擎被服务端 KILL 后复活，第二条
/// 连接的握手能力位仍不声明 CLIENT_COMPRESS（重连不得恢复为硬编码启用）
TEST(MariadbEngineCompression, ReconnectKeepsCompressionConfig)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "重连压缩用例需要外部 mysqld endpoint";
    }
    CompressionFixture fixture{false, "wirecomp_reconnect"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    AllTypesRow first{};
    first.signed64 = 31;
    first.text = "before-kill";
    const auto insertedFirst = fixture.engine->Insert(meta, TableScope::Static(), &first, nullptr);
    ASSERT_TRUE(insertedFirst.ok()) << insertedFirst.diagnostic;
    ASSERT_TRUE(WaitForClientFlagCount(fixture.probe, 1U)) << "代理未捕获首条握手";

    ASSERT_TRUE(KillEngineConnectionsForRevive()) << "服务端 KILL 引擎连接失败";

    // KILL 后下一笔 DML 触发有界复活（经代理建立第二条连接）
    AllTypesRow second{};
    second.signed64 = 32;
    second.text = "after-kill";
    const auto insertedSecond = fixture.engine->Insert(meta, TableScope::Static(), &second, nullptr);
    ASSERT_TRUE(insertedSecond.ok()) << insertedSecond.diagnostic;

    ASSERT_TRUE(WaitForClientFlagCount(fixture.probe, 2U)) << "复活未经代理建立第二条连接";
    const auto flags = fixture.probe.ClientFlags();
    ASSERT_EQ(flags.size(), 2U) << "复活应经代理建立第二条连接";
    EXPECT_EQ(flags[0] & kClientCompressFlag, 0U);
    EXPECT_EQ(flags[1] & kClientCompressFlag, 0U) << "重连不得恢复为硬编码压缩";

    // 复活后读路径正常：两行都在
    std::vector<AllTypesRow> rows;
    auto sink = ResultSink::For(rows);
    ASSERT_TRUE(fixture.engine->Select(meta, TableScope::Static(), ConditionData{}, QueryOptions{}, sink).ok());
    EXPECT_EQ(rows.size(), 2U);
}

/// Wipe（DROP DATABASE 的独立短连接）与 Open 使用同一压缩配置：true 时
/// Wipe 连接同样声明 CLIENT_COMPRESS（统一前 Wipe 从不施加压缩——本用例
/// 在旧行为下失败）；Wipe 后物理 database 消失
TEST(MariadbEngineCompression, WipeAppliesSameCompressionOptions)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "Wipe 压缩用例需要外部 mysqld endpoint";
    }
    for (const bool compression : {true, false}) {
        CompressionFixture fixture{compression, compression ? "wirecomp_wipe_on" : "wirecomp_wipe_off"};
        ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
        const std::string ns = fixture.dataNamespace;
        ASSERT_TRUE(fixture.engine->Close().ok());
        fixture.ready = false;

        const auto wiped = fixture.engine->Wipe(ns);
        ASSERT_TRUE(wiped.ok()) << wiped.diagnostic;

        ASSERT_TRUE(WaitForClientFlagCount(fixture.probe, 2U)) << "代理未捕获 Wipe 握手";
        const auto flags = fixture.probe.ClientFlags();
        ASSERT_EQ(flags.size(), 2U) << "Open 与 Wipe 应各建立一条连接";
        EXPECT_EQ((flags[0] & kClientCompressFlag) != 0U, compression) << "Open 连接能力位与配置不符";
        EXPECT_EQ((flags[1] & kClientCompressFlag) != 0U, compression) << "Wipe 连接必须与 Open 使用同一压缩配置";

        // Wipe 实效：物理 database 已不存在
        std::vector<std::vector<std::string>> schemas;
        ASSERT_TRUE(AdminQueryRows(
            "SELECT SCHEMA_NAME FROM information_schema.schemata "
            "WHERE SCHEMA_NAME LIKE 'hcclvm\\_%'",
            schemas));
        const auto found = std::any_of(schemas.begin(), schemas.end(), [&ns](const std::vector<std::string>& row) {
            return !row.empty() && row[0].find(ns) != std::string::npos;
        });
        EXPECT_FALSE(found) << "Wipe 后物理 database 应已删除: " << ns;
    }
}

/// prepare 阶段断线（全新 SQL 文本 → 语句缓存未命中 → 死连接上 prepare）：
/// 应复用 DML 的有界复活语义——重连一次并重新 prepare/bind/execute 成功。
/// 修改前：SELECT prepare 无恢复 → 引擎终态关闭 → 本用例在首个断言失败。
TEST(MariadbEngineSelectRevive, PrepareStageDisconnectOnFreshSqlRevivesOnce)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "SELECT 复活用例需要外部 mysqld endpoint";
    }
    EngineFixture fixture{"select_prepare_revive"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    AllTypesRow first{};
    first.signed64 = 101;
    first.text = "prepare-revive";
    const auto insertedFirst = fixture.engine->Insert(meta, TableScope::Static(), &first, nullptr);
    ASSERT_TRUE(insertedFirst.ok()) << insertedFirst.diagnostic;
    AllTypesRow second{};
    second.signed64 = 202;
    second.text = "second-row";
    const auto insertedSecond = fixture.engine->Insert(meta, TableScope::Static(), &second, nullptr);
    ASSERT_TRUE(insertedSecond.ok()) << insertedSecond.diagnostic;

    // 预热形状 A（按 id 等值）：使后续"按 signed64"是全新 SQL 文本，
    // 断线后必然落在 prepare 阶段而非 execute 阶段
    const auto byId = ConditionData{ConditionClause{
        static_cast<uint16_t>(0), Compare::EQUAL, ConditionJoin::NONE, 0, 0,
        ConditionValue{DbScalar{*insertedFirst.value}}}};
    std::vector<AllTypesRow> warm;
    auto warmSink = ResultSink::For(warm);
    const auto warmed = fixture.engine->Select(meta, TableScope::Static(), byId, QueryOptions{}, warmSink);
    ASSERT_TRUE(warmed.ok()) << warmed.diagnostic;

    ASSERT_TRUE(KillEngineConnectionsForRevive()) << "服务端 KILL 引擎连接失败";

    // 新形状 SELECT（按 signed64 等值）：prepare 撞死连接 → 复活一次 → 成功
    const auto bySigned = ConditionData{ConditionClause{
        static_cast<uint16_t>(1), Compare::EQUAL, ConditionJoin::NONE, 0, 0, ConditionValue{DbScalar{int64_t{202}}}}};
    std::vector<AllTypesRow> rows;
    auto sink = ResultSink::For(rows);
    const auto selected = fixture.engine->Select(meta, TableScope::Static(), bySigned, QueryOptions{}, sink);
    ASSERT_TRUE(selected.ok()) << selected.diagnostic;
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].text, "second-row");

    // 新形状 COUNT（无 WHERE）：再次断线后同样在 prepare 阶段复活
    ASSERT_TRUE(KillEngineConnectionsForRevive()) << "服务端 KILL 引擎连接失败";
    const auto counted = fixture.engine->Count(meta, TableScope::Static(), ConditionData{});
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    EXPECT_EQ(*counted.value, 2U);
}

/// prepare 阶段服务端拒绝（DROP 物理表 → ER_NO_SUCH_TABLE 1146；服务端错误包
/// 携带在 stmt 上，连接级 errno 为 0——与 256p max_prepared_stmt_count 实证同形）：
/// 不得重试（非断线族），真实 errno/错误文本必须保留在诊断里。
/// 修改前：仅依赖连接级 mysql_errno → errno 0 / 空诊断 → 真实原因丢失。
TEST(MariadbEngineSelectRevive, PrepareStageServerRejectionKeepsRealError)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "SELECT prepare 拒绝用例需要外部 mysqld endpoint";
    }
    EngineFixture fixture{"select_prepare_reject"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    AllTypesRow row{};
    row.signed64 = 55;
    row.text = "before-drop";
    const auto inserted = fixture.engine->Insert(meta, TableScope::Static(), &row, nullptr);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;

    std::string schema;
    ASSERT_TRUE(FindFixtureSchema(fixture.dataNamespace, schema)) << "定位 fixture 物理 database 失败";
    // 先删外键子表再删目标表（ChildOfAllTypes.FK → AllTypesRow）
    std::string dropError;
    ASSERT_TRUE(AdminExecSql("DROP TABLE `" + schema + "`.`ChildOfAllTypes`", &dropError))
        << "DROP 子表失败: " << dropError;
    ASSERT_TRUE(AdminExecSql("DROP TABLE `" + schema + "`.`AllTypesRow`", &dropError))
        << "DROP 物理表失败: " << dropError;

    const auto byId = ConditionData{ConditionClause{
        static_cast<uint16_t>(0), Compare::EQUAL, ConditionJoin::NONE, 0, 0,
        ConditionValue{DbScalar{*inserted.value}}}};
    std::vector<AllTypesRow> rows;
    auto sink = ResultSink::For(rows);
    const auto selected = fixture.engine->Select(meta, TableScope::Static(), byId, QueryOptions{}, sink);
    EXPECT_FALSE(selected.ok());
    // 1146 → IO_ERROR（非断线族不映射 BACKEND_UNAVAILABLE、不触发复活）
    EXPECT_EQ(selected.code, HcclSim::Storage::DbCode::IO_ERROR);
    // 真实错误可见：errno 1146 与 "doesn't exist" 文本（修改前：errno 0 / 空诊断）
    EXPECT_NE(selected.diagnostic.find("1146"), std::string::npos) << selected.diagnostic;
    EXPECT_NE(selected.diagnostic.find("exist"), std::string::npos) << selected.diagnostic;
}

/// prepare 阶段断线且复活失败（私有实例已被本用例停掉）：重连失败的真实原因
/// 必须保留（CR_CONN_HOST_ERROR 族 → BACKEND_UNAVAILABLE + reconnect 诊断），
/// 引擎随后进入终态、后续操作干净失败。修改前：无复活尝试，诊断不含 reconnect。
/// 注意：本用例会停掉 HCCL_VM_MARIADB_TEST_SERVER_PIDFILE 指向的 per-run 私有
/// 实例（由 run_mariadb_private_tests.sh 单独第二轮调度），绝不触碰共享实例。
TEST(MariadbEngineSelectRevive, PrepareStageReviveFailureKeepsRealErrorKillsPrivateServer)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "复活失败用例需要外部 mysqld endpoint";
    }
    const char* pidfile = std::getenv("HCCL_VM_MARIADB_TEST_SERVER_PIDFILE");
    if (pidfile == nullptr || pidfile[0] == '\0') {
        GTEST_SKIP() << "HCCL_VM_MARIADB_TEST_SERVER_PIDFILE 未设置："
                        "复活失败用例需要 per-run 私有实例（run_mariadb_private_tests.sh）";
    }
    EngineFixture fixture{"revive_failure_private"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    AllTypesRow row{};
    row.signed64 = 88;
    row.text = "before-server-stop";
    const auto inserted = fixture.engine->Insert(meta, TableScope::Static(), &row, nullptr);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;

    // 预热按 id 形状
    const auto byId = ConditionData{ConditionClause{
        static_cast<uint16_t>(0), Compare::EQUAL, ConditionJoin::NONE, 0, 0,
        ConditionValue{DbScalar{*inserted.value}}}};
    std::vector<AllTypesRow> warm;
    auto warmSink = ResultSink::For(warm);
    const auto warmed = fixture.engine->Select(meta, TableScope::Static(), byId, QueryOptions{}, warmSink);
    ASSERT_TRUE(warmed.ok()) << warmed.diagnostic;

    ASSERT_TRUE(KillEngineConnectionsForRevive()) << "服务端 KILL 引擎连接失败";

    // 停掉本测试专用的私有实例（per-run，由 runner 创建）
    std::ifstream pidIn{pidfile};
    ASSERT_TRUE(pidIn.good()) << "pidfile 不可读: " << pidfile;
    long serverPid = 0;
    pidIn >> serverPid;
    ASSERT_GT(serverPid, 1) << "pidfile 内容异常";
    ASSERT_EQ(::kill(static_cast<pid_t>(serverPid), SIGTERM), 0) << "停止私有实例失败";
    ASSERT_TRUE(WaitProcessGone(static_cast<pid_t>(serverPid), 15000)) << "私有实例未退出";

    // 新形状 SELECT：prepare 断线 → 复活 → 重连失败 → 真实错误保留
    const auto bySigned = ConditionData{ConditionClause{
        static_cast<uint16_t>(1), Compare::EQUAL, ConditionJoin::NONE, 0, 0, ConditionValue{DbScalar{int64_t{88}}}}};
    std::vector<AllTypesRow> rows;
    auto sink = ResultSink::For(rows);
    const auto selected = fixture.engine->Select(meta, TableScope::Static(), bySigned, QueryOptions{}, sink);
    EXPECT_FALSE(selected.ok());
    EXPECT_EQ(selected.code, HcclSim::Storage::DbCode::BACKEND_UNAVAILABLE);
    EXPECT_NE(selected.diagnostic.find("reconnect"), std::string::npos) << selected.diagnostic;

    // 引擎终态：后续操作干净失败（INVALID_STATE），无悬空访问
    const auto counted = fixture.engine->Count(meta, TableScope::Static(), ConditionData{});
    EXPECT_FALSE(counted.ok());
    EXPECT_EQ(counted.code, HcclSim::Storage::DbCode::INVALID_STATE);
}

/// VMB 形状的单行查询语义不受修复影响：单对象 sink 命中返回首行；真无记录
/// 返回 NOT_FOUND（独立语义，不是 IO_ERROR）；多行命中时取首行后排空残余。
TEST(MariadbEngineSelectRevive, SingleRowHitTrueNotFoundAndDrainAfterFirstRow)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "单行语义用例需要外部 mysqld endpoint";
    }
    EngineFixture fixture{"single_row_semantics"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    AllTypesRow first{};
    first.signed64 = 7;
    first.text = "hit";
    const auto insertedFirst = fixture.engine->Insert(meta, TableScope::Static(), &first, nullptr);
    ASSERT_TRUE(insertedFirst.ok()) << insertedFirst.diagnostic;

    // 单对象 sink 命中
    AllTypesRow hit{};
    auto hitSink = ResultSink::For(hit);
    const auto byId = ConditionData{ConditionClause{
        static_cast<uint16_t>(0), Compare::EQUAL, ConditionJoin::NONE, 0, 0,
        ConditionValue{DbScalar{*insertedFirst.value}}}};
    const auto selected = fixture.engine->Select(meta, TableScope::Static(), byId, QueryOptions{}, hitSink);
    ASSERT_TRUE(selected.ok()) << selected.diagnostic;
    EXPECT_EQ(hit.signed64, 7);
    EXPECT_EQ(hit.text, "hit");

    // 单对象 sink 真无记录：NOT_FOUND（不是成功空对象，也不是 IO_ERROR）
    AllTypesRow miss{};
    auto missSink = ResultSink::For(miss);
    const auto byMissing = ConditionData{ConditionClause{
        static_cast<uint16_t>(0), Compare::EQUAL, ConditionJoin::NONE, 0, 0,
        ConditionValue{DbScalar{*insertedFirst.value + 4096U}}}};
    const auto missing = fixture.engine->Select(meta, TableScope::Static(), byMissing, QueryOptions{}, missSink);
    ASSERT_FALSE(missing.ok());
    EXPECT_EQ(missing.code, HcclSim::Storage::DbCode::NOT_FOUND);

    // 多行命中 + 单对象 sink：取首行后必须排空残余行（DrainCursorLocked 成功路径）
    AllTypesRow second{};
    second.signed64 = 7;
    second.text = "second-hit";
    const auto insertedSecond = fixture.engine->Insert(meta, TableScope::Static(), &second, nullptr);
    ASSERT_TRUE(insertedSecond.ok()) << insertedSecond.diagnostic;
    AllTypesRow firstOfTwo{};
    auto firstSink = ResultSink::For(firstOfTwo);
    const auto bySigned = ConditionData{ConditionClause{
        static_cast<uint16_t>(1), Compare::EQUAL, ConditionJoin::NONE, 0, 0, ConditionValue{DbScalar{int64_t{7}}}}};
    const auto two = fixture.engine->Select(meta, TableScope::Static(), bySigned, QueryOptions{}, firstSink);
    ASSERT_TRUE(two.ok()) << two.diagnostic;
    EXPECT_TRUE(firstOfTwo.id == *insertedFirst.value || firstOfTwo.id == *insertedSecond.value);
    EXPECT_EQ(firstOfTwo.signed64, 7);
}

/// execute 阶段非连接错误（表锁等待超时 ER_LOCK_WAIT_TIMEOUT 1205）：真实
/// errno/文本保留（1205 → RESOURCE_EXHAUSTED），不触发复活，连接保持可用。
TEST(MariadbEngineSelectRevive, ExecuteLockTimeoutKeepsRealErrorAndEngineUsable)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "锁等待用例需要外部 mysqld endpoint";
    }
    // 会话变量在连接建立时从 GLOBAL 继承：先压低全局再建 fixture 引擎连接；
    // 离开时还原全局默认（守卫不依赖断言路径）
    struct ResetGlobal {
        ~ResetGlobal() { (void)AdminExecSql("SET GLOBAL lock_wait_timeout = 86400"); }
    } resetGlobal;
    ASSERT_TRUE(AdminExecSql("SET GLOBAL lock_wait_timeout = 3"));

    EngineFixture fixture{"lock_timeout_exec"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    AllTypesRow row{};
    row.signed64 = 9;
    row.text = "locked-table";
    const auto inserted = fixture.engine->Insert(meta, TableScope::Static(), &row, nullptr);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;

    // 预热按 id 形状（语句入缓存）：后续失败落在 execute 而非 prepare
    const auto byId = ConditionData{ConditionClause{
        static_cast<uint16_t>(0), Compare::EQUAL, ConditionJoin::NONE, 0, 0,
        ConditionValue{DbScalar{*inserted.value}}}};
    std::vector<AllTypesRow> warm;
    auto warmSink = ResultSink::For(warm);
    const auto warmed = fixture.engine->Select(meta, TableScope::Static(), byId, QueryOptions{}, warmSink);
    ASSERT_TRUE(warmed.ok()) << warmed.diagnostic;

    std::string schema;
    ASSERT_TRUE(FindFixtureSchema(fixture.dataNamespace, schema)) << "定位 fixture 物理 database 失败";
    {
        AdminSession admin;
        ASSERT_TRUE(admin.Ok()) << "管理连接建立失败";
        ASSERT_TRUE(admin.Query("USE `" + schema + "`"));
        ASSERT_TRUE(admin.Query("LOCK TABLES `AllTypesRow` WRITE"));

        std::vector<AllTypesRow> rows;
        auto sink = ResultSink::For(rows);
        const auto selected = fixture.engine->Select(meta, TableScope::Static(), byId, QueryOptions{}, sink);
        EXPECT_FALSE(selected.ok());
        // 1205 → RESOURCE_EXHAUSTED；真实 errno 与文本必须可见
        EXPECT_EQ(selected.code, HcclSim::Storage::DbCode::RESOURCE_EXHAUSTED);
        EXPECT_NE(selected.diagnostic.find("1205"), std::string::npos) << selected.diagnostic;

        (void)admin.Query("UNLOCK TABLES");
    }

    // 非连接错误不放弃连接：解锁后同一引擎继续可用
    const auto recount = fixture.engine->Count(meta, TableScope::Static(), ConditionData{});
    ASSERT_TRUE(recount.ok()) << recount.diagnostic;
    EXPECT_EQ(*recount.value, 1U);
}

/// 行流中途断线（大 BLOB 多行 + 流式回读期间服务端 KILL）：必须失败（不得
/// 部分成功）、已收行无重复（无整批重放）、引擎进入终态后操作干净失败。
TEST(MariadbEngineSelectRevive, MidStreamDisconnectNoReplayAndCleanTerminalState)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "行流中断线用例需要外部 mysqld endpoint";
    }
    // 压缩协议固定显式 true：行流耗时前提依赖压缩协议对随机数据的 CPU 开销
    // （缺省压缩随部署默认变为 false 后，未压缩流在 loopback 上可跑赢 500ms
    // KILL，测不到中途断线；显式 true 固定本用例的原始前提，不受默认值影响）。
    EngineFixture fixture{"midstream_kill", TestProfileWithCompression(true)};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    // 60 行 × 2MB 随机（不可压缩）BLOB：协议压缩对随机数据无收益，unbuffered
    // 回读 wire 体积 ~120MB，行流耗时秒级且无法被内核 socket 缓冲整段吞下——
    // 500ms 的延迟 KILL 必然落在行流中途（可压缩数据会整段进缓冲、KILL 后仍
    // 从缓冲收完，测不到中途断线）。
    constexpr std::size_t kRows = 60U;
    constexpr std::size_t kBlobSize = 2U * 1024U * 1024U;
    std::mt19937_64 rng{12345U};
    std::vector<uint8_t> blob(kBlobSize);
    for (std::size_t i = 0; i < kBlobSize; ++i) {
        blob[i] = static_cast<uint8_t>(rng() & 0xFFU);
    }
    for (std::size_t i = 0; i < kRows; ++i) {
        AllTypesRow row{};
        row.signed64 = static_cast<int64_t>(i);
        row.blob = blob;
        const auto inserted = fixture.engine->Insert(meta, TableScope::Static(), &row, nullptr);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    }

    std::vector<AllTypesRow> received;
    auto sink = ResultSink::For(received);
    std::thread killer([] {
        std::this_thread::sleep_for(std::chrono::milliseconds{500});
        (void)KillEngineConnectionsForRevive();
    });
    const auto selected = fixture.engine->Select(meta, TableScope::Static(), ConditionData{}, QueryOptions{}, sink);
    killer.join();

    // 行流中断线：整笔失败，不得部分成功
    EXPECT_FALSE(selected.ok()) << selected.diagnostic;
    // 已收行 id 严格递增：无重复行（无整批重放/重放叠加）
    ASSERT_GT(received.size(), 0U) << "KILL 未落在行流中途（未收到任何行）";
    for (std::size_t i = 1; i < received.size(); ++i) {
        ASSERT_GT(received[i].id, received[i - 1].id);
    }
    // 断线族已放弃连接 → 引擎终态：后续操作干净失败（INVALID_STATE），无悬空访问
    const auto counted = fixture.engine->Count(meta, TableScope::Static(), ConditionData{});
    EXPECT_FALSE(counted.ok());
    EXPECT_EQ(counted.code, HcclSim::Storage::DbCode::INVALID_STATE);
}

} // namespace mariadb_engine_test

namespace mariadb_engine_test {
// ---- A4 合同：事务内断线后引擎终态一致性（不得把死连接继续当可用）----
// 事务内 DML 失败（不复活）→ 连接已死必须进入终态：Rollback/新 Begin/后续
// 查询全部 INVALID_STATE 干净失败，无悬空访问；对比非事务路径断线可复活继续
// 用（既有 ServerSideKillRecoversForAutocommitDmlAndSelect 覆盖）。
TEST(MariadbEngineSelectRevive, TransactionDisconnectTerminalStateConsistent)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "事务终态用例需要外部 mysqld endpoint";
    }
    EngineFixture fixture{"txn_terminal_state"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    AllTypesRow row{};
    row.signed64 = 66;
    row.text = "before-txn-kill";
    const auto inserted = fixture.engine->Insert(meta, TableScope::Static(), &row, nullptr);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;

    TransactionBinding binding;
    ASSERT_TRUE(fixture.engine->Begin(binding).ok());
    ASSERT_TRUE(KillEngineConnectionsForRevive()) << "服务端 KILL 引擎连接失败";

    // 事务内 DML：断线族失败如实返回（不复活重放）
    AllTypesRow inTxn{};
    inTxn.signed64 = 77;
    const auto txnInserted = fixture.engine->Insert(meta, TableScope::Static(), &inTxn, &binding);
    EXPECT_FALSE(txnInserted.ok());

    // 终态一致性：同一 binding 的 Rollback、新 Begin、普通查询全部干净失败
    const auto rolled = fixture.engine->Rollback(binding);
    EXPECT_FALSE(rolled.ok());
    EXPECT_EQ(rolled.code, HcclSim::Storage::DbCode::INVALID_STATE);
    TransactionBinding fresh;
    const auto begun = fixture.engine->Begin(fresh);
    EXPECT_FALSE(begun.ok());
    EXPECT_EQ(begun.code, HcclSim::Storage::DbCode::INVALID_STATE);
    const auto counted = fixture.engine->Count(meta, TableScope::Static(), ConditionData{});
    EXPECT_FALSE(counted.ok());
    EXPECT_EQ(counted.code, HcclSim::Storage::DbCode::INVALID_STATE);
}

// ---- §5.2 prepared 语句额度：真实错误保真、不重连（私有实例可调额度）----
TEST(MariadbEngineSelectRevive, PreparedStmtQuotaKeepsRealErrorNoRevive)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"stmt_quota"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    AllTypesRow row{};
    row.signed64 = 12;
    const auto inserted = fixture.engine->Insert(meta, TableScope::Static(), &row, nullptr);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;

    // 管理连接：读当前打开的预备语句数，把额度压到当前值（新增 prepare 必失败
    // 1461 ER_MAX_PREPARED_STMT_COUNT；离开时还原默认 16382）
    struct ResetQuota {
        ~ResetQuota() { (void)AdminExecSql("SET GLOBAL max_prepared_stmt_count = 16382"); }
    } resetQuota;
    std::vector<std::vector<std::string>> rows;
    ASSERT_TRUE(AdminQueryRows("SHOW GLOBAL STATUS LIKE 'Prepared_stmt_count'", rows));
    ASSERT_EQ(rows.size(), 1U);
    const auto current = std::stoul(rows[0][1]);
    ASSERT_TRUE(AdminExecSql("SET GLOBAL max_prepared_stmt_count = " + std::to_string(current)));

    // 全新 SQL 形状（按 signed64）撞额度：1461 → IO_ERROR，非断线族不复活
    const auto bySigned = ConditionData{ConditionClause{
        static_cast<uint16_t>(1), Compare::EQUAL, ConditionJoin::NONE, 0, 0, ConditionValue{DbScalar{int64_t{12}}}}};
    std::vector<AllTypesRow> out;
    auto sink = ResultSink::For(out);
    const auto selected = fixture.engine->Select(meta, TableScope::Static(), bySigned, QueryOptions{}, sink);
    EXPECT_FALSE(selected.ok());
    EXPECT_EQ(selected.code, HcclSim::Storage::DbCode::IO_ERROR);
    EXPECT_NE(selected.diagnostic.find("1461"), std::string::npos) << selected.diagnostic;
}

// ---- §5.3 初次连接/握手失败：connect 阶段与真实码可见，不伪装 NOT_FOUND ----
TEST(MariadbEngineSelectRevive, FirstOpenHandshakeFailureShowsConnectStageAndCode)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    // 指向必然拒连的端口（127.0.0.1:9 = discard 服务的经典关闭端口；本机无监听）
    HcclSim::Storage::MariadbConnectionProfile profile;
    profile.uri = "mariadb://hccl:hccl-pass@127.0.0.1:9";
    profile.connectionEstablishmentTimeoutS = 2;
    profile.packetReadTimeoutS = 2;
    profile.packetWriteTimeoutS = 2;
    MariaDbEngine engine{&kMariadbTestDatabase, profile};
    EngineOpenRequest request;
    request.mode = StorageOpenMode::READ_WRITE_CREATE;
    request.dataNamespace = "handshake_dead_port";
    const auto opened = engine.Open(request);
    ASSERT_FALSE(opened.ok());
    // 拒连/不可达 → BACKEND_UNAVAILABLE（2002/2003 族），不是 IO_ERROR/NOT_FOUND
    EXPECT_EQ(opened.code, HcclSim::Storage::DbCode::BACKEND_UNAVAILABLE);
    EXPECT_NE(opened.diagnostic.find("failed to connect to mariadb"), std::string::npos) << opened.diagnostic;
    EXPECT_NE(opened.diagnostic.find("mariadb errno 200"), std::string::npos) << opened.diagnostic;
}

// ---- §5.6 COUNT=0 与紧随查询成功（成功路径不变）----
TEST(MariadbEngineSelectRevive, CountZeroAndSubsequentQueryStillSucceed)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"count_zero_then_query"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    AllTypesRow row{};
    row.signed64 = 21;
    row.text = "count-zero";
    const auto inserted = fixture.engine->Insert(meta, TableScope::Static(), &row, nullptr);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;

    // 无匹配条件 COUNT = 0（成功语义，非错误）
    const auto byMissing = ConditionData{ConditionClause{
        static_cast<uint16_t>(1), Compare::EQUAL, ConditionJoin::NONE, 0, 0, ConditionValue{DbScalar{int64_t{9999}}}}};
    const auto counted = fixture.engine->Count(meta, TableScope::Static(), byMissing);
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    EXPECT_EQ(*counted.value, 0U);

    // 紧随其后正常查询成功（连接/语句缓存状态未被破坏）
    const auto bySigned = ConditionData{ConditionClause{
        static_cast<uint16_t>(1), Compare::EQUAL, ConditionJoin::NONE, 0, 0, ConditionValue{DbScalar{int64_t{21}}}}};
    std::vector<AllTypesRow> rows;
    auto sink = ResultSink::For(rows);
    const auto selected = fixture.engine->Select(meta, TableScope::Static(), bySigned, QueryOptions{}, sink);
    ASSERT_TRUE(selected.ok()) << selected.diagnostic;
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].text, "count-zero");
}

// ---- §5.6c CountMatches 单语句聚合：一次调用一个缓存语句、结果集排空、
// 活动事务期间沿 Count 合同拒绝 ----
TEST(MariadbEngineSelectRevive, CountMatchesAggregateDrainsAndKeepsConnectionUsable)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    EngineFixture fixture{"count_matches_aggregate"};
    ASSERT_TRUE(fixture.ready) << fixture.diagnostic;
    const auto& meta = MetaOf("AllTypesRow");

    // 三行：signed64=21 两行（text 不同，其中一行 unsigned16=1），signed64=42 一行。
    AllTypesRow first{};
    first.signed64 = 21;
    first.text = "cm-a";
    first.unsigned16 = 1;
    ASSERT_TRUE(fixture.engine->Insert(meta, TableScope::Static(), &first, nullptr).ok());
    AllTypesRow second{};
    second.signed64 = 21;
    second.text = "cm-b";
    ASSERT_TRUE(fixture.engine->Insert(meta, TableScope::Static(), &second, nullptr).ok());
    AllTypesRow third{};
    third.signed64 = 42;
    third.text = "cm-c";
    ASSERT_TRUE(fixture.engine->Insert(meta, TableScope::Static(), &third, nullptr).ok());

    // where（signed64=21，2 行）与 match（text="cm-a"，其中 1 行）使用不同参数。
    const auto where = ConditionData{ConditionClause{
        static_cast<uint16_t>(1), Compare::EQUAL, ConditionJoin::NONE, 0, 0, ConditionValue{DbScalar{int64_t{21}}}}};
    const auto match = ConditionData{ConditionClause{
        static_cast<uint16_t>(7), Compare::EQUAL, ConditionJoin::NONE, 0, 0,
        ConditionValue{DbScalar{std::string{"cm-a"}}}}};
    const auto counted = fixture.engine->CountMatches(meta, TableScope::Static(), where, match);
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    ASSERT_TRUE(counted.value.has_value());
    EXPECT_EQ(counted.value->total, 2U);
    EXPECT_EQ(counted.value->matched, 1U);
    EXPECT_EQ(counted.affectedRows, 1U);

    // 空集合：where 无命中 → OK + {0, 0}（不是失败）。
    const auto missing = ConditionData{ConditionClause{
        static_cast<uint16_t>(1), Compare::EQUAL, ConditionJoin::NONE, 0, 0, ConditionValue{DbScalar{int64_t{9999}}}}};
    const auto empty = fixture.engine->CountMatches(meta, TableScope::Static(), missing, match);
    ASSERT_TRUE(empty.ok()) << empty.diagnostic;
    EXPECT_EQ(empty.value->total, 0U);
    EXPECT_EQ(empty.value->matched, 0U);

    // 语句缓存观察：同一聚合语句第二次调用命中缓存（无第二次 prepare）。
    const uint64_t missesBefore = MariadbEngineTestPeer::CacheMisses(*fixture.engine);
    const auto again = fixture.engine->CountMatches(meta, TableScope::Static(), where, match);
    ASSERT_TRUE(again.ok()) << again.diagnostic;
    EXPECT_EQ(again.value->total, 2U);
    EXPECT_EQ(MariadbEngineTestPeer::CacheMisses(*fixture.engine), missesBefore) << "同形聚合语句应命中语句缓存";

    // 排空证据：聚合后同连接继续查询与更新成功（无 commands out of sync）。
    std::vector<AllTypesRow> rows;
    auto sink = ResultSink::For(rows);
    const auto selected = fixture.engine->Select(meta, TableScope::Static(), where, QueryOptions{}, sink);
    ASSERT_TRUE(selected.ok()) << selected.diagnostic;
    ASSERT_EQ(rows.size(), 2U);
    const auto touched = fixture.engine->Update(
        meta, TableScope::Static(),
        ConditionData{ConditionClause{
            static_cast<uint16_t>(7), Compare::EQUAL, ConditionJoin::NONE, 0, 0,
            ConditionValue{DbScalar{std::string{"cm-b"}}}}},
        UpdateData{LogicalChange{static_cast<uint16_t>(3), ChangeKind::SET, DbScalar{uint64_t{7}}}}, nullptr);
    ASSERT_TRUE(touched.ok()) << touched.diagnostic;

    // 活动事务期间拒绝（沿 Count/Select 只读操作既有合同）。
    TransactionBinding binding;
    ASSERT_TRUE(fixture.engine->Begin(binding).ok());
    const auto during = fixture.engine->CountMatches(meta, TableScope::Static(), where, match);
    EXPECT_EQ(during.code, HcclSim::Storage::DbCode::INVALID_STATE);
    ASSERT_TRUE(fixture.engine->Rollback(binding).ok());
}

} // namespace mariadb_engine_test
