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

// 文件职责：用目录登记的测试表直接验证 SQLiteEngine 的唯一执行链：
// 自增主键生成、类型/BLOB 字节一致、条件查询/排序/数量、更新、删除、清表、
// 序列重置、动态 scope 表、外键级联、事务与备份的端到端 SQLite 语义。
// 语句缓存/schema manifest 已删除——验证它们的用例随结构一并移除。
#include <gtest/gtest.h>

#include <sqlite3.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "storage_backends/sqlite/engine/sqlite_engine.h"
#include "storage/query_expression.h"
#include "storage/record_contract.h"
#include "storage/storage_catalog.h"

namespace HcclSim::Storage {
// DDL 统计测试观察点：经 friend 关系读取 SQLiteEngine 私有快照；
// 仅在本测试文件定义与使用，不是产品公共 API，产品代码不得引用。
struct SQLiteEngineTestPeer {
    static SQLiteEngine::SchemaDdlSnapshot Snapshot(const SQLiteEngine& engine) { return engine.SchemaDdlStatistics(); }
    static DbResult<std::string> ReadPragma(const SQLiteEngine& engine, const char* name)
    {
        return engine.ReadConnectionPragmaForTest(name);
    }
};
} // namespace HcclSim::Storage

// ---- 主测试目录：静态 metric 表（主键+唯一文本+整数/浮点/BLOB）、动态 task 表、
// 无自增主键的 plain 表（显式 0 是合法值）----
namespace sqlite_engine_test {
struct MetricTab {
    uint64_t id{0};
    std::string name;
    int64_t count{0};
    double score{0.0};
    std::vector<uint8_t> payload;
};
struct TaskTab {
    uint64_t id{0};
    std::string note;
};
struct PlainKeyTab {
    uint64_t id{0};
    uint64_t value{0};
};
} // namespace sqlite_engine_test

using namespace HcclSim::Storage::Ddl;
#define Field HCCL_VM_DDL_FIELD
#define Version HCCL_VM_DDL_VERSION
#define PrimaryKey HCCL_VM_DDL_PRIMARY_KEY
#define Unique HCCL_VM_DDL_UNIQUE
#define Index HCCL_VM_DDL_INDEX
#define ForeignKey HCCL_VM_DDL_FOREIGN_KEY
#define Scope Ddl::ScopeKind

#define SQLITE_ENGINE_TEST_DATABASE(Table)                                                                             \
    Table(                                                                                                             \
        sqlite_engine_test::MetricTab,                                                                                 \
        (Version(1), Scope::Static, Field(id, AutoIncrement), Field(name), Field(count), Field(score), Field(payload), \
         PrimaryKey(id), Unique(name), Index(count)))                                                                  \
        Table(                                                                                                         \
            sqlite_engine_test::TaskTab,                                                                               \
            (Version(1), Scope::DynamicPid, Field(id, AutoIncrement), Field(note), PrimaryKey(id)))                    \
            Table(                                                                                                     \
                sqlite_engine_test::PlainKeyTab, (Version(1), Scope::Static, Field(id), Field(value), PrimaryKey(id)))

#define HCCL_VM_DDL_CURRENT_DATABASE "SQLITE_ENGINE_TEST_DB"
HCCL_VM_DEFINE_DATABASE(kSqliteEngineTestDatabase, SQLITE_ENGINE_TEST_DB, SQLITE_ENGINE_TEST_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE

// ---- 外键目标列语义（目标列为非首字段，验证目录解析到真实成员编号）----
namespace fk_semantics_test {
struct FkParentTab {
    uint64_t id{0};
    uint64_t code{0};
};
struct FkChildTab {
    uint64_t id{0};
    uint64_t parent_code{0};
};
} // namespace fk_semantics_test

#define FK_SEMANTICS_DATABASE(Table)                                                                      \
    Table(                                                                                                \
        fk_semantics_test::FkParentTab,                                                                   \
        (Version(1), Scope::Static, Field(id, AutoIncrement), Field(code), PrimaryKey(id), Unique(code))) \
        Table(                                                                                            \
            fk_semantics_test::FkChildTab,                                                                \
            (Version(1), Scope::Static, Field(id, AutoIncrement), Field(parent_code), PrimaryKey(id),     \
             ForeignKey(parent_code, fk_semantics_test::FkParentTab::code, OnDelete::Cascade)))

#define HCCL_VM_DDL_CURRENT_DATABASE "FK_SEMANTICS_DB"
HCCL_VM_DEFINE_DATABASE(kFkSemanticsDatabase, FK_SEMANTICS_DB, FK_SEMANTICS_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE

// ---- uint64 宽列范围/排序语义 ----
namespace unsigned_mapping_test {
struct UnsignedRow {
    uint64_t id{0};
    uint64_t wideValue{0};
    uint32_t narrowValue{0};
    bool enabled{false};
    int64_t signedValue{0};
};
} // namespace unsigned_mapping_test

#define UNSIGNED_MAPPING_DATABASE(Table)                                                                            \
    Table(                                                                                                          \
        unsigned_mapping_test::UnsignedRow, (Version(1), Scope::Static, Field(id, AutoIncrement), Field(wideValue), \
                                             Field(narrowValue), Field(enabled), Field(signedValue), PrimaryKey(id)))

#define HCCL_VM_DDL_CURRENT_DATABASE "UNSIGNED_MAPPING_DB"
HCCL_VM_DEFINE_DATABASE(kUnsignedMappingDatabase, UNSIGNED_MAPPING_DB, UNSIGNED_MAPPING_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE

#undef Field
#undef Version
#undef PrimaryKey
#undef Unique
#undef Index
#undef ForeignKey
#undef Scope

namespace {
using namespace HcclSim::Storage;

constexpr SqliteConnectionProfile kTestProfile{10000, false, 200};

// 生成一次性临时数据库路径，并清理上次运行可能遗留的同名文件。
std::string TempDatabasePath(const std::string& tag)
{
    static std::atomic<uint64_t> counter{0};
    const std::string path
        = testing::TempDir() + "sqlite_engine_" + tag + "_" + std::to_string(counter.fetch_add(1)) + ".db";
    (void)std::remove(path.c_str());
    return path;
}

// 直接读 sqlite_master 中一张表的 CREATE TABLE 文本（独立只读连接，WAL 下并发安全）。
std::string ReadTableDdl(const std::string& databasePath, const std::string& tableName)
{
    sqlite3* raw = nullptr;
    if (sqlite3_open(databasePath.c_str(), &raw) != SQLITE_OK) {
        sqlite3_close(raw);
        return {};
    }
    sqlite3_stmt* stmt = nullptr;
    std::string ddl;
    const std::string sql = "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?";
    if (sqlite3_prepare_v2(raw, sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, tableName.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const unsigned char* text = sqlite3_column_text(stmt, 0);
            ddl = text == nullptr ? std::string{} : std::string(reinterpret_cast<const char*>(text));
        }
    }
    sqlite3_finalize(stmt);
    sqlite3_close(raw);
    return ddl;
}

// 主测试夹具：每个用例使用独立临时数据库与目录登记表。
class SQLiteEngineTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        databasePath = TempDatabasePath("metric");
        engine = std::make_unique<SQLiteEngine>(databasePath, &Ddl::kSqliteEngineTestDatabase, kTestProfile);
        EngineOpenRequest open;
        open.mode = StorageOpenMode::READ_WRITE_CREATE;
        const auto opened = engine->Open(open);
        ASSERT_TRUE(opened.ok()) << opened.diagnostic;
        const auto prepared = engine->Prepare();
        ASSERT_TRUE(prepared.ok()) << prepared.diagnostic;
    }

    using Metric = sqlite_engine_test::MetricTab;

    DbResult<uint64_t> InsertMetric(Metric& record)
    {
        return engine->Insert(RecordSchema<Metric>::Meta(), TableScope::Static(), &record, nullptr);
    }

    DbResult<std::vector<Metric>> SelectMetric(const Condition<Metric>& where, QueryOptions options = {})
    {
        std::vector<Metric> rows;
        auto sink = ResultSink::For(rows);
        const auto result
            = engine->Select(RecordSchema<Metric>::Meta(), TableScope::Static(), where.data, options, sink);
        if (!result.ok()) {
            return {result.code, std::nullopt, result.affectedRows, result.diagnostic};
        }
        return {DbCode::OK, std::optional<std::vector<Metric>>{std::move(rows)}, result.affectedRows, {}};
    }

    DbResult<Metric> SelectOneMetric(const Condition<Metric>& where)
    {
        Metric record;
        auto sink = ResultSink::For(record);
        const auto result = engine->Select(RecordSchema<Metric>::Meta(), TableScope::Static(), where.data, {}, sink);
        if (!result.ok()) {
            return {result.code, std::nullopt, result.affectedRows, result.diagnostic};
        }
        return {DbCode::OK, std::optional<Metric>{std::move(record)}, result.affectedRows, {}};
    }

    std::string databasePath;
    std::unique_ptr<SQLiteEngine> engine;
};

TEST_F(SQLiteEngineTest, InsertGeneratesSequentialIds)
{
    for (uint64_t expected = 1; expected <= 3U; ++expected) {
        Metric record;
        record.name = "m" + std::to_string(expected);
        const auto inserted = InsertMetric(record);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
        EXPECT_EQ(*inserted.value, expected);
    }
}

TEST_F(SQLiteEngineTest, InsertExplicitPrimaryKeyIsPreserved)
{
    Metric record;
    record.id = 42U;
    record.name = "explicit";
    const auto inserted = InsertMetric(record);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    EXPECT_EQ(*inserted.value, 42U);
    const auto rows = SelectMetric(Eq(&Metric::id, 42U)).value;
    ASSERT_EQ(rows->size(), 1U);
    EXPECT_EQ((*rows)[0].name, "explicit");
}

TEST(SQLiteEnginePrimaryKeySemanticsTest, PlainPrimaryKeyBindsExplicitZeroWithoutAutoincrement)
{
    using Plain = sqlite_engine_test::PlainKeyTab;
    const std::string path = TempDatabasePath("plain-key");
    SQLiteEngine engine(path, &Ddl::kSqliteEngineTestDatabase, kTestProfile);
    EngineOpenRequest open;
    open.mode = StorageOpenMode::READ_WRITE_CREATE;
    ASSERT_TRUE(engine.Open(open).ok());
    ASSERT_TRUE(engine.Prepare().ok());
    Plain record;
    record.id = 0U; // 普通主键：0 是显式值，不是自增占位
    record.value = 7U;
    const auto inserted = engine.Insert(RecordSchema<Plain>::Meta(), TableScope::Static(), &record, nullptr);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    EXPECT_EQ(*inserted.value, 0U);
    std::vector<Plain> rows;
    auto sink = ResultSink::For(rows);
    const auto selected
        = engine.Select(RecordSchema<Plain>::Meta(), TableScope::Static(), Eq(&Plain::id, 0U).data, {}, sink);
    ASSERT_TRUE(selected.ok()) << selected.diagnostic;
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].value, 7U);
}

// 可重建数据的速度默认值在引擎实际连接上生效：连接级 PRAGMA（synchronous/
// secure_delete）只有同一连接上的读数才证明生效（另开 CLI 连接检查无效）。
// 可写连接（CREATE 与 READ_EXISTING——后者在当前源码同样是可写连接）固定
// synchronous=0、secure_delete=0；WAL/外键/busy/事务合同保持不变；关闭后
// 新引擎重开仍生效（生产 VerifyConnectionPragmas 的严格核验不删除）。
namespace {
    void ExpectWritableSpeedPragmas(const SQLiteEngine& engine)
    {
        const std::pair<const char*, std::string> expected[] = {
            {"journal_mode", "wal"}, {"synchronous", "0"}, {"secure_delete", "0"},
            {"foreign_keys", "1"},   {"temp_store", "2"},  {"mmap_size", "268435456"},
        };
        for (const auto& item : expected) {
            const auto actual = SQLiteEngineTestPeer::ReadPragma(engine, item.first);
            ASSERT_TRUE(actual.ok()) << item.first << ": " << actual.diagnostic;
            EXPECT_EQ(*actual.value, item.second) << "PRAGMA " << item.first << " 未按速度默认值生效";
        }
    }
} // namespace

TEST(SQLiteEngineSpeedDefaultsTest, WritableConnectionSpeedPragmasTakeEffectAcrossReopen)
{
    const std::string path = TempDatabasePath("speed-defaults");
    {
        SQLiteEngine engine(path, &Ddl::kSqliteEngineTestDatabase, kTestProfile);
        EngineOpenRequest open;
        open.mode = StorageOpenMode::READ_WRITE_CREATE;
        ASSERT_TRUE(engine.Open(open).ok()) << "CREATE 打开即核验失败（VerifyConnectionPragmas）";
        ASSERT_TRUE(engine.Prepare().ok());
        ExpectWritableSpeedPragmas(engine);
        // 一次正常读写事务仍工作（速度默认值不破坏正常运行合同）。
        using Metric = sqlite_engine_test::MetricTab;
        Metric record;
        record.name = "speed";
        const auto inserted = engine.Insert(RecordSchema<Metric>::Meta(), TableScope::Static(), &record, nullptr);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    }
    {
        SQLiteEngine engine(path, &Ddl::kSqliteEngineTestDatabase, kTestProfile);
        EngineOpenRequest open;
        open.mode = StorageOpenMode::READ_EXISTING;
        ASSERT_TRUE(engine.Open(open).ok()) << "READ_EXISTING 打开即核验失败";
        ASSERT_TRUE(engine.Prepare().ok());
        ExpectWritableSpeedPragmas(engine);
        ASSERT_TRUE(engine.Close().ok());
    }
    {
        SQLiteEngine engine(path, &Ddl::kSqliteEngineTestDatabase, kTestProfile);
        EngineOpenRequest open;
        open.mode = StorageOpenMode::READ_WRITE_CREATE;
        ASSERT_TRUE(engine.Open(open).ok());
        ASSERT_TRUE(engine.Prepare().ok());
        ExpectWritableSpeedPragmas(engine);
        ASSERT_TRUE(engine.Close().ok());
    }
}

TEST_F(SQLiteEngineTest, AutoIncrementPrimaryKeyRendersAutoincrementInDdl)
{
    const std::string ddl = ReadTableDdl(databasePath, "MetricTab");
    EXPECT_NE(ddl.find("AUTOINCREMENT"), std::string::npos) << ddl;
    const std::string plainDdl = ReadTableDdl(databasePath, "PlainKeyTab");
    EXPECT_EQ(plainDdl.find("AUTOINCREMENT"), std::string::npos) << plainDdl;
}

TEST_F(SQLiteEngineTest, SelectByPrimaryKeyRoundTripsEveryColumnType)
{
    Metric record;
    record.name = "every-type";
    record.count = -12345;
    record.score = 3.25;
    record.payload = {0x00, 0x01, 0xFF, 0xFE, 0x7F};
    const auto inserted = InsertMetric(record);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    const auto rows = SelectMetric(Eq(&Metric::id, *inserted.value)).value;
    ASSERT_EQ(rows->size(), 1U);
    EXPECT_EQ((*rows)[0].name, "every-type");
    EXPECT_EQ((*rows)[0].count, -12345);
    EXPECT_DOUBLE_EQ((*rows)[0].score, 3.25);
    EXPECT_EQ((*rows)[0].payload, (std::vector<uint8_t>{0x00, 0x01, 0xFF, 0xFE, 0x7F}));
}

TEST_F(SQLiteEngineTest, EmptyBlobRoundTripsAsZeroLengthValue)
{
    Metric record;
    record.name = "empty-blob";
    record.payload = {};
    const auto inserted = InsertMetric(record);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    const auto rows = SelectMetric(Eq(&Metric::name, std::string("empty-blob"))).value;
    ASSERT_EQ(rows->size(), 1U);
    EXPECT_TRUE((*rows)[0].payload.empty());
}

TEST_F(SQLiteEngineTest, OneMiBBlobRoundTripsThroughImmediatePath)
{
    Metric record;
    record.name = "big";
    record.payload.resize(1024U * 1024U);
    for (std::size_t i = 0; i < record.payload.size(); ++i) {
        record.payload[i] = static_cast<uint8_t>(i & 0xFFU);
    }
    const auto inserted = InsertMetric(record);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    const auto rows = SelectMetric(Eq(&Metric::id, *inserted.value)).value;
    ASSERT_EQ(rows->size(), 1U);
    ASSERT_EQ((*rows)[0].payload.size(), 1024U * 1024U);
    EXPECT_EQ((*rows)[0].payload[12345], 12345U & 0xFFU);
    EXPECT_EQ((*rows)[0].payload.back(), (1024U * 1024U - 1U) & 0xFFU);
}

TEST_F(SQLiteEngineTest, PrefixComparisonMatchesTextColumnPrefix)
{
    Metric record;
    record.name = "prefixable";
    const auto inserted = InsertMetric(record);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    // 文本列前缀匹配：命中（非文本列前缀由公开表达式工厂在编译期拒绝）
    const auto textRows = SelectMetric(StartsWith(&Metric::name, std::string("prefix"))).value;
    ASSERT_EQ(textRows->size(), 1U);
    const auto missRows = SelectMetric(StartsWith(&Metric::name, std::string("other"))).value;
    EXPECT_TRUE(missRows->empty());
}

// PREFIX 转义模式串经 SQLITE_TRANSIENT 绑定（bind 时由 SQLite 复制，临时串
// 生命周期仅覆盖 bind 调用）：覆盖 SSO 短串、超过 SSO 的堆串、%/_/\ 字面量
// 语义（不得当作 LIKE 通配符）与多个 PREFIX 的组合条件。
TEST_F(SQLiteEngineTest, PrefixBindingSurvivesEscapedAndLongPatterns)
{
    const char* const names[] = {
        "p_short",                             // 短前缀目标（SSO 内）
        "long-prefix-target-0123456789abcdef", // 长前缀目标（超过 SSO 阈值）
        "literal%percent",                     // % 按字面量匹配的目标
        "literalXpercent",                     // 干扰行：仅当 % 被误当通配符才命中
        "under_score",                         // _ 按字面量匹配的目标
        "underXscore",                         // 干扰行：仅当 _ 被误当通配符才命中
        "back\\slash",                         // 反斜杠按字面量匹配的目标
        "backXslash",                          // 干扰行
    };
    for (const char* name : names) {
        Metric record;
        record.name = name;
        ASSERT_TRUE(InsertMetric(record).ok()) << name;
    }
    // 短前缀（SSO 内，绑定发生在模式串析构前）
    EXPECT_EQ(SelectMetric(StartsWith(&Metric::name, std::string("p_s"))).value->size(), 1U);
    // 长前缀（堆存储临时串）
    const std::string longPrefix = "long-prefix-target-0123456";
    ASSERT_GT(longPrefix.size(), 15U); // 超过 libstdc++ SSO 阈值，确保走堆存储
    EXPECT_EQ(SelectMetric(StartsWith(&Metric::name, longPrefix)).value->size(), 1U);
    // 含 % 的前缀按字面量匹配
    const auto percentRows = SelectMetric(StartsWith(&Metric::name, std::string("literal%"))).value;
    ASSERT_EQ(percentRows->size(), 1U);
    EXPECT_EQ((*percentRows)[0].name, "literal%percent");
    // 含 _ 的前缀按字面量匹配
    const auto underRows = SelectMetric(StartsWith(&Metric::name, std::string("under_"))).value;
    ASSERT_EQ(underRows->size(), 1U);
    EXPECT_EQ((*underRows)[0].name, "under_score");
    // 含反斜杠的前缀按字面量匹配
    const auto slashRows = SelectMetric(StartsWith(&Metric::name, std::string("back\\"))).value;
    ASSERT_EQ(slashRows->size(), 1U);
    EXPECT_EQ((*slashRows)[0].name, "back\\slash");
    // 多个 PREFIX 的组合条件：AND 同时命中同一行的两个前缀
    EXPECT_EQ(
        SelectMetric(
            And(StartsWith(&Metric::name, std::string("long-")), StartsWith(&Metric::name, std::string("long-prefix"))))
            .value->size(),
        1U);
    // 多个 PREFIX 的组合条件：OR 命中两行不同前缀（两个转义临时串先后绑定）
    EXPECT_EQ(
        SelectMetric(
            Or(StartsWith(&Metric::name, std::string("p_s")), StartsWith(&Metric::name, std::string("under_"))))
            .value->size(),
        2U);
}

TEST_F(SQLiteEngineTest, SelectByPrimaryKeyConditionMissingRowReturnsEmptySet)
{
    const auto rows = SelectMetric(Eq(&Metric::id, 999U));
    ASSERT_TRUE(rows.ok()) << rows.diagnostic;
    EXPECT_TRUE(rows.value->empty());
}

TEST_F(SQLiteEngineTest, SelectHonorsPredicatesSortLimitAndConnectives)
{
    for (uint64_t i = 1; i <= 6U; ++i) {
        Metric record;
        record.name = "row" + std::to_string(i);
        record.count = static_cast<int64_t>(i);
        record.score = static_cast<double>(i) * 1.5;
        ASSERT_TRUE(InsertMetric(record).ok());
    }
    // 单谓词
    EXPECT_EQ(SelectMetric(Eq(&Metric::count, int64_t{3})).value->size(), 1U);
    EXPECT_EQ(SelectMetric(Ne(&Metric::count, int64_t{3})).value->size(), 5U);
    EXPECT_EQ(SelectMetric(Lt(&Metric::count, int64_t{3})).value->size(), 2U);
    EXPECT_EQ(SelectMetric(Le(&Metric::count, int64_t{3})).value->size(), 3U);
    EXPECT_EQ(SelectMetric(Gt(&Metric::count, int64_t{3})).value->size(), 3U);
    EXPECT_EQ(SelectMetric(Ge(&Metric::count, int64_t{3})).value->size(), 4U);
    // IN
    EXPECT_EQ(SelectMetric(In(&Metric::count, std::vector<int64_t>{1, 3, 5})).value->size(), 3U);
    // 组合条件（括号语义）：(count<=2 OR count>=5) AND score>0
    const auto grouped = SelectMetric(
        And(Or(Le(&Metric::count, int64_t{2}), Ge(&Metric::count, int64_t{5})), Gt(&Metric::score, 0.0)));
    ASSERT_EQ(grouped.value->size(), 4U);
    // 嵌套组合：count=1 OR (count>=3 AND count<=4)
    const auto nested = SelectMetric(
        Or(Eq(&Metric::count, int64_t{1}), And(Ge(&Metric::count, int64_t{3}), Le(&Metric::count, int64_t{4}))));
    ASSERT_EQ(nested.value->size(), 3U);
    // 排序 + 数量
    QueryOptions options;
    options.sorts.emplace_back(RecordSchema<Metric>::template FieldIdOf(&Metric::count), Order::DESCENDING);
    options.limit = 2U;
    const auto topRows = SelectMetric(All<Metric>(), options).value;
    ASSERT_EQ(topRows->size(), 2U);
    EXPECT_EQ((*topRows)[0].count, 6);
    EXPECT_EQ((*topRows)[1].count, 5);
}

TEST_F(SQLiteEngineTest, SelectWithLimitOneReturnsFirstRowOrEmptySet)
{
    Metric record;
    record.name = "only";
    ASSERT_TRUE(InsertMetric(record).ok());
    QueryOptions options;
    options.limit = 1U;
    const auto rows = SelectMetric(All<Metric>(), options).value;
    ASSERT_EQ(rows->size(), 1U);
    options.limit = 0U; // Limit(0)：OK + 空结果（数量是绑定参数）
    const auto empty = SelectMetric(All<Metric>(), options);
    ASSERT_TRUE(empty.ok());
    EXPECT_TRUE(empty.value->empty());
}

TEST_F(SQLiteEngineTest, EmptyInPredicateMatchesNoRows)
{
    Metric record;
    record.name = "present";
    ASSERT_TRUE(InsertMetric(record).ok());
    const auto rows = SelectMetric(In(&Metric::count, std::vector<int64_t>{}));
    ASSERT_TRUE(rows.ok()) << rows.diagnostic;
    EXPECT_TRUE(rows.value->empty());
}

TEST_F(SQLiteEngineTest, CountAggregatesOnDatabaseSide)
{
    for (int64_t i = 0; i < 4; ++i) {
        Metric record;
        record.name = "c" + std::to_string(i);
        record.count = i;
        ASSERT_TRUE(InsertMetric(record).ok());
    }
    const auto all = engine->Count(RecordSchema<Metric>::Meta(), TableScope::Static(), All<Metric>().data);
    ASSERT_TRUE(all.ok()) << all.diagnostic;
    EXPECT_EQ(*all.value, 4U);
    const auto filtered
        = engine->Count(RecordSchema<Metric>::Meta(), TableScope::Static(), Ge(&Metric::count, int64_t{2}).data);
    ASSERT_TRUE(filtered.ok());
    EXPECT_EQ(*filtered.value, 2U);
}

// CountMatches 单语句聚合（引擎级）：where 与 match 不同参数不串位；All(match)
// 恒真；空集合 {0,0}；聚合后同连接继续查询成功（语句/连接状态未被破坏）。
TEST_F(SQLiteEngineTest, CountMatchesAggregatesBothCountersInOneStatement)
{
    for (int64_t i = 0; i < 4; ++i) {
        Metric record;
        record.name = "cm-" + std::to_string(i);
        record.count = i;
        record.score = i < 2 ? 1.0 : 2.0;
        ASSERT_TRUE(InsertMetric(record).ok());
    }
    const auto& meta = RecordSchema<Metric>::Meta();
    // where：count >= 1（3 行）；match：score = 2.0（其中 2 行）。
    const auto counted = engine->CountMatches(
        meta, TableScope::Static(), Ge(&Metric::count, int64_t{1}).data, Eq(&Metric::score, 2.0).data);
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    ASSERT_TRUE(counted.value.has_value());
    EXPECT_EQ(counted.value->total, 3U);
    EXPECT_EQ(counted.value->matched, 2U);
    EXPECT_EQ(counted.affectedRows, 1U);

    // All(match)：恒真，matched == total。
    const auto always
        = engine->CountMatches(meta, TableScope::Static(), Ge(&Metric::count, int64_t{1}).data, All<Metric>().data);
    ASSERT_TRUE(always.ok()) << always.diagnostic;
    EXPECT_EQ(always.value->total, 3U);
    EXPECT_EQ(always.value->matched, 3U);

    // 空集合：where 无命中 → OK + {0, 0}。
    const auto empty = engine->CountMatches(
        meta, TableScope::Static(), Eq(&Metric::count, int64_t{999}).data, Eq(&Metric::score, 2.0).data);
    ASSERT_TRUE(empty.ok()) << empty.diagnostic;
    EXPECT_EQ(empty.value->total, 0U);
    EXPECT_EQ(empty.value->matched, 0U);

    // 聚合后同连接查询成功。
    const auto rows = SelectMetric(Ge(&Metric::count, int64_t{1}));
    ASSERT_TRUE(rows.ok()) << rows.diagnostic;
    EXPECT_EQ(rows.value->size(), 3U);
}

TEST_F(SQLiteEngineTest, UpdateSetAndIncrementAffectMatchingRows)
{
    for (int64_t i = 1; i <= 3; ++i) {
        Metric record;
        record.name = "u" + std::to_string(i);
        record.count = i * 10;
        ASSERT_TRUE(InsertMetric(record).ok());
    }
    const UpdateData set{{RecordSchema<Metric>::template FieldIdOf(&Metric::score), ChangeKind::SET, DbScalar{9.5}}};
    const auto updated = engine->Update(
        RecordSchema<Metric>::Meta(), TableScope::Static(), Ge(&Metric::count, int64_t{20}).data, set, nullptr);
    ASSERT_TRUE(updated.ok()) << updated.diagnostic;
    EXPECT_EQ(*updated.value, 2U);
    const auto rows = SelectMetric(Eq(&Metric::score, 9.5)).value;
    ASSERT_EQ(rows->size(), 2U);

    const UpdateData increment{
        {RecordSchema<Metric>::template FieldIdOf(&Metric::count), ChangeKind::INCREMENT, DbScalar{int64_t{5}}}};
    const auto bumped = engine->Update(
        RecordSchema<Metric>::Meta(), TableScope::Static(), Eq(&Metric::name, std::string("u1")).data, increment,
        nullptr);
    ASSERT_TRUE(bumped.ok()) << bumped.diagnostic;
    EXPECT_EQ(*bumped.value, 1U);
    const auto one = SelectMetric(Eq(&Metric::name, std::string("u1"))).value;
    ASSERT_EQ(one->size(), 1U);
    EXPECT_EQ((*one)[0].count, 15);
}

TEST_F(SQLiteEngineTest, UpdateWithoutConditionIsRejected)
{
    Metric record;
    record.name = "guard";
    ASSERT_TRUE(InsertMetric(record).ok());
    const UpdateData set{{RecordSchema<Metric>::template FieldIdOf(&Metric::score), ChangeKind::SET, DbScalar{1.0}}};
    const auto rejected
        = engine->Update(RecordSchema<Metric>::Meta(), TableScope::Static(), All<Metric>().data, set, nullptr);
    EXPECT_EQ(rejected.code, DbCode::INVALID_ARGUMENT);
}

TEST_F(SQLiteEngineTest, DeleteByPrimaryKeyAndByPredicate)
{
    for (int64_t i = 1; i <= 4; ++i) {
        Metric record;
        record.name = "d" + std::to_string(i);
        record.count = i;
        ASSERT_TRUE(InsertMetric(record).ok());
    }
    const auto byKey = engine->Delete(
        RecordSchema<Metric>::Meta(), TableScope::Static(), Eq(&Metric::count, int64_t{1}).data, nullptr);
    ASSERT_TRUE(byKey.ok()) << byKey.diagnostic;
    EXPECT_EQ(*byKey.value, 1U);
    const auto byPredicate = engine->Delete(
        RecordSchema<Metric>::Meta(), TableScope::Static(), Gt(&Metric::count, int64_t{2}).data, nullptr);
    ASSERT_TRUE(byPredicate.ok());
    EXPECT_EQ(*byPredicate.value, 2U);
    const auto rest = SelectMetric(All<Metric>()).value;
    ASSERT_EQ(rest->size(), 1U);
    EXPECT_EQ((*rest)[0].count, 2);
}

TEST_F(SQLiteEngineTest, ClearKeepSequenceContinuesAutoincrement)
{
    Metric record;
    record.name = "first";
    const auto first = InsertMetric(record);
    ASSERT_TRUE(first.ok());
    const auto cleared = engine->Clear(RecordSchema<Metric>::Meta(), TableScope::Static(), ClearMode::KEEP_SEQUENCE);
    ASSERT_TRUE(cleared.ok()) << cleared.diagnostic;
    Metric second;
    second.name = "second";
    const auto next = InsertMetric(second);
    ASSERT_TRUE(next.ok());
    EXPECT_GT(*next.value, *first.value); // 序列继续
}

TEST_F(SQLiteEngineTest, ClearResetSequenceRestartsAutoincrement)
{
    Metric record;
    record.name = "first";
    ASSERT_TRUE(InsertMetric(record).ok());
    const auto cleared = engine->Clear(RecordSchema<Metric>::Meta(), TableScope::Static(), ClearMode::RESET_SEQUENCE);
    ASSERT_TRUE(cleared.ok()) << cleared.diagnostic;
    Metric second;
    second.name = "second";
    const auto next = InsertMetric(second);
    ASSERT_TRUE(next.ok());
    EXPECT_EQ(*next.value, 1U); // 序列重置为 0 后重新从 1 开始
}

TEST_F(SQLiteEngineTest, ListTablesHidesInternalTables)
{
    Metric record;
    record.name = "any";
    ASSERT_TRUE(InsertMetric(record).ok());
    const auto tables = engine->ListTables();
    ASSERT_TRUE(tables.ok()) << tables.diagnostic;
    const auto has = [&tables](const std::string& name) {
        return std::find(tables.value->begin(), tables.value->end(), name) != tables.value->end();
    };
    EXPECT_TRUE(has("MetricTab"));
    EXPECT_TRUE(has("PlainKeyTab"));
    // TaskTab 是 DynamicPid 表：物理表在首次访问才创建，此刻不存在（与基线语义一致）。
    EXPECT_FALSE(has("TaskTab"));
    for (const auto& table : *tables.value) {
        EXPECT_NE(table.compare(0, 7, "sqlite_"), 0) << table; // 无 sqlite_ 前缀表
    }
}

TEST_F(SQLiteEngineTest, DynamicScopedTablesAreIsolatedPerProcess)
{
    using Task = sqlite_engine_test::TaskTab;
    const auto& meta = RecordSchema<Task>::Meta();
    Task first;
    first.note = "p1";
    const auto r1 = engine->Insert(meta, TableScope::Process(1U), &first, nullptr);
    ASSERT_TRUE(r1.ok()) << r1.diagnostic;
    Task second;
    second.note = "p2";
    const auto r2 = engine->Insert(meta, TableScope::Process(2U), &second, nullptr);
    ASSERT_TRUE(r2.ok());
    EXPECT_EQ(*r1.value, 1U); // 各 scope 独立序列
    EXPECT_EQ(*r2.value, 1U);
    std::vector<Task> rows1;
    auto sink1 = ResultSink::For(rows1);
    ASSERT_TRUE(engine->Select(meta, TableScope::Process(1U), All<Task>().data, {}, sink1).ok());
    ASSERT_EQ(rows1.size(), 1U);
    EXPECT_EQ(rows1[0].note, "p1");
    std::vector<Task> rows2;
    auto sink2 = ResultSink::For(rows2);
    ASSERT_TRUE(engine->Select(meta, TableScope::Process(2U), All<Task>().data, {}, sink2).ok());
    ASSERT_EQ(rows2.size(), 1U);
    EXPECT_EQ(rows2[0].note, "p2");
    // 物理表名带 PID 后缀
    EXPECT_NE(ReadTableDdl(databasePath, "TaskTab_P_1"), std::string{});
    EXPECT_NE(ReadTableDdl(databasePath, "TaskTab_P_2"), std::string{});
}

TEST_F(SQLiteEngineTest, DropDynamicScopeRemovesEveryPhysicalTableOfBase)
{
    using Task = sqlite_engine_test::TaskTab;
    const auto& meta = RecordSchema<Task>::Meta();
    Task first;
    first.note = "p1";
    ASSERT_TRUE(engine->Insert(meta, TableScope::Process(1U), &first, nullptr).ok());
    Task second;
    second.note = "p2";
    ASSERT_TRUE(engine->Insert(meta, TableScope::Process(2U), &second, nullptr).ok());
    const auto dropped = engine->Clear(meta, TableScope::Process(0U), ClearMode::DROP_DYNAMIC_SCOPE);
    ASSERT_TRUE(dropped.ok()) << dropped.diagnostic;
    EXPECT_EQ(dropped.affectedRows, 2U); // 被销毁的物理 scope 表数
    EXPECT_EQ(ReadTableDdl(databasePath, "TaskTab_P_1"), std::string{});
    EXPECT_EQ(ReadTableDdl(databasePath, "TaskTab_P_2"), std::string{});
    const auto tables = engine->ListTables();
    ASSERT_TRUE(tables.ok());
    EXPECT_EQ(std::find(tables.value->begin(), tables.value->end(), std::string("TaskTab_P_1")), tables.value->end());
}

TEST_F(SQLiteEngineTest, UniqueIndexRejectsDuplicateValuesAsConflict)
{
    Metric record;
    record.name = "dup";
    ASSERT_TRUE(InsertMetric(record).ok());
    Metric duplicate;
    duplicate.name = "dup";
    const auto rejected = InsertMetric(duplicate);
    EXPECT_EQ(rejected.code, DbCode::CONFLICT);
}

TEST_F(SQLiteEngineTest, TransactionCommitsDirectWritesInOrder)
{
    using Metric = sqlite_engine_test::MetricTab;
    const auto& meta = RecordSchema<Metric>::Meta();
    TransactionBinding binding;
    ASSERT_TRUE(engine->Begin(binding).ok());
    Metric first;
    first.name = "tx1";
    first.count = 1;
    const auto inserted1 = engine->Insert(meta, TableScope::Static(), &first, &binding);
    ASSERT_TRUE(inserted1.ok()) << inserted1.diagnostic;
    EXPECT_EQ(*inserted1.value, 1U); // 生成主键在 Commit 前即可取得（sqlite3_last_insert_rowid）
    Metric second;
    second.name = "tx2";
    second.count = 2;
    ASSERT_TRUE(engine->Insert(meta, TableScope::Static(), &second, &binding).ok());
    const UpdateData bump{
        {RecordSchema<Metric>::template FieldIdOf(&Metric::count), ChangeKind::INCREMENT, DbScalar{int64_t{10}}}};
    ASSERT_TRUE(
        engine->Update(meta, TableScope::Static(), Eq(&Metric::name, std::string("tx1")).data, bump, &binding).ok());
    const auto committed = engine->Commit(binding);
    ASSERT_TRUE(committed.ok()) << committed.diagnostic;
    const auto rows = SelectMetric(Eq(&Metric::name, std::string("tx1"))).value;
    ASSERT_EQ(rows->size(), 1U);
    EXPECT_EQ((*rows)[0].count, 11); // 前序 INSERT 对后序 UPDATE 可见
    EXPECT_EQ(SelectMetric(All<Metric>()).value->size(), 2U);
}

TEST_F(SQLiteEngineTest, TransactionRollbackDiscardsDirectWrites)
{
    using Metric = sqlite_engine_test::MetricTab;
    const auto& meta = RecordSchema<Metric>::Meta();
    TransactionBinding binding;
    ASSERT_TRUE(engine->Begin(binding).ok());
    Metric record;
    record.name = "rolled";
    ASSERT_TRUE(engine->Insert(meta, TableScope::Static(), &record, &binding).ok());
    const auto rolled = engine->Rollback(binding);
    ASSERT_TRUE(rolled.ok()) << rolled.diagnostic;
    EXPECT_TRUE(SelectMetric(All<Metric>()).value->empty());
}

TEST_F(SQLiteEngineTest, TransactionDmlFailureLeavesRollbackToCaller)
{
    using Metric = sqlite_engine_test::MetricTab;
    const auto& meta = RecordSchema<Metric>::Meta();
    Metric existing;
    existing.name = "kept";
    ASSERT_TRUE(InsertMetric(existing).ok());
    TransactionBinding binding;
    ASSERT_TRUE(engine->Begin(binding).ok());
    Metric good;
    good.name = "good";
    ASSERT_TRUE(engine->Insert(meta, TableScope::Static(), &good, &binding).ok());
    Metric duplicate;
    duplicate.name = "kept"; // 唯一索引冲突 → 执行失败
    const auto failed = engine->Insert(meta, TableScope::Static(), &duplicate, &binding);
    EXPECT_EQ(failed.code, DbCode::CONFLICT);
    // DML 只执行并返回错误：引擎仍处于事务开启状态（物理事务未回滚），
    // 普通即时 DML 在活动事务期间被拒绝（不得混入同一连接）
    Metric extra;
    extra.name = "extra";
    EXPECT_EQ(engine->Insert(meta, TableScope::Static(), &extra, nullptr).code, DbCode::INVALID_STATE);
    // Transaction 层在失败后统一调用一次 Rollback：物理回滚并清理活动槽
    const auto rolled = engine->Rollback(binding);
    ASSERT_TRUE(rolled.ok()) << rolled.diagnostic;
    // 整体回滚：成功执行的 good 也没有落库；引擎恢复可用
    EXPECT_EQ(SelectMetric(All<Metric>()).value->size(), 1U);
    Metric after;
    after.name = "after";
    ASSERT_TRUE(engine->Insert(meta, TableScope::Static(), &after, nullptr).ok());
}

// 事务内首次创建的动态表随 ROLLBACK 撤销后，引擎的动态表存在性登记必须同步
// 失效：同 scope 的下一次访问重新执行幂等 DDL，不得因登记残留跳过建表出现
// no such table。失效发生在共同回滚清理路径（显式回滚/DML 失败回滚/COMMIT
// 失败后的回滚共用同一入口；COMMIT 失败在真实 WAL 连接上不可构造，机制由
// 同一函数覆盖）。正常成功提交不清空登记。
TEST_F(SQLiteEngineTest, TransactionRollbackInvalidatesFirstCreatedDynamicTable)
{
    using Task = sqlite_engine_test::TaskTab;
    const auto& meta = RecordSchema<Task>::Meta();
    TransactionBinding binding;
    ASSERT_TRUE(engine->Begin(binding).ok());
    Task record;
    record.note = "first-tx";
    // 事务内首次访问该 scope：CREATE TABLE/INDEX 在物理事务内执行并登记
    ASSERT_TRUE(engine->Insert(meta, TableScope::Process(9U), &record, &binding).ok());
    ASSERT_TRUE(engine->Rollback(binding).ok());
    // 首次建表已随回滚撤销（无独立提交者）
    EXPECT_EQ(ReadTableDdl(databasePath, "TaskTab_P_9"), std::string{});
    // 同 scope 再次访问：登记已失效，重新执行幂等 DDL 后写入成功
    Task retry;
    retry.note = "retry";
    const auto reinserted = engine->Insert(meta, TableScope::Process(9U), &retry, nullptr);
    ASSERT_TRUE(reinserted.ok()) << reinserted.diagnostic;
    EXPECT_EQ(*reinserted.value, 1U); // 新物理表的自增序列从头开始
    std::vector<Task> rows;
    auto sink = ResultSink::For(rows);
    ASSERT_TRUE(engine->Select(meta, TableScope::Process(9U), All<Task>().data, {}, sink).ok());
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].note, "retry");
    EXPECT_NE(ReadTableDdl(databasePath, "TaskTab_P_9"), std::string{});
}

// 已存在的动态表不受回滚失效影响：登记清空只针对"库内实际状态"的保守同步，
// 幂等 CREATE IF NOT EXISTS 对既有表为空操作；事务外的既有记录保持、后续访问正常。
TEST_F(SQLiteEngineTest, TransactionRollbackKeepsPreexistingDynamicTableRows)
{
    using Task = sqlite_engine_test::TaskTab;
    const auto& meta = RecordSchema<Task>::Meta();
    Task kept;
    kept.note = "kept";
    ASSERT_TRUE(engine->Insert(meta, TableScope::Process(5U), &kept, nullptr).ok());
    TransactionBinding binding;
    ASSERT_TRUE(engine->Begin(binding).ok());
    Task extra;
    extra.note = "extra";
    ASSERT_TRUE(engine->Insert(meta, TableScope::Process(5U), &extra, &binding).ok());
    ASSERT_TRUE(engine->Rollback(binding).ok());
    // 事务外写入的原记录保持；表可继续访问
    std::vector<Task> rows;
    auto sink = ResultSink::For(rows);
    ASSERT_TRUE(engine->Select(meta, TableScope::Process(5U), All<Task>().data, {}, sink).ok());
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].note, "kept");
    Task after;
    after.note = "after";
    ASSERT_TRUE(engine->Insert(meta, TableScope::Process(5U), &after, nullptr).ok());
    // 成功事务不清空登记：提交后既有 scope 的访问不重复建表（行为正确即可，
    // 本用例主要固定"清空不破坏既有表"）
    TransactionBinding committed;
    ASSERT_TRUE(engine->Begin(committed).ok());
    Task inTx;
    inTx.note = "committed";
    ASSERT_TRUE(engine->Insert(meta, TableScope::Process(5U), &inTx, &committed).ok());
    ASSERT_TRUE(engine->Commit(committed).ok());
    std::vector<Task> finalRows;
    auto finalSink = ResultSink::For(finalRows);
    ASSERT_TRUE(engine->Select(meta, TableScope::Process(5U), All<Task>().data, {}, finalSink).ok());
    EXPECT_EQ(finalRows.size(), 3U); // kept + after + committed
}

TEST_F(SQLiteEngineTest, BackupRestoresFullDatabaseOnFreshEngine)
{
    Metric record;
    record.name = "backed";
    record.payload = {1, 2, 3};
    const auto inserted = InsertMetric(record);
    ASSERT_TRUE(inserted.ok());
    const std::string backupPath = TempDatabasePath("backup-dest");
    auto& relational = static_cast<IRelationalEngine&>(*engine);
    const auto backed = relational.BackupRelational(EngineBackupRequest{backupPath});
    ASSERT_TRUE(backed.ok()) << backed.diagnostic;
    // 新引擎读备份文件
    SQLiteEngine restored(backupPath, &Ddl::kSqliteEngineTestDatabase, kTestProfile);
    EngineOpenRequest open;
    open.mode = StorageOpenMode::READ_EXISTING;
    ASSERT_TRUE(restored.Open(open).ok());
    ASSERT_TRUE(restored.Prepare().ok());
    std::vector<Metric> rows;
    auto sink = ResultSink::For(rows);
    ASSERT_TRUE(
        restored
            .Select(RecordSchema<Metric>::Meta(), TableScope::Static(), Eq(&Metric::id, *inserted.value).data, {}, sink)
            .ok());
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].name, "backed");
    EXPECT_EQ(rows[0].payload, (std::vector<uint8_t>{1, 2, 3}));
}

TEST_F(SQLiteEngineTest, CloseReleasesConnectionAndRejectsFurtherUse)
{
    const auto closed = engine->Close();
    ASSERT_TRUE(closed.ok()) << closed.diagnostic;
    Metric record;
    record.name = "after-close";
    EXPECT_EQ(
        engine->Insert(RecordSchema<Metric>::Meta(), TableScope::Static(), &record, nullptr).code,
        DbCode::INVALID_STATE);
    EXPECT_EQ(engine->Close().code, DbCode::INVALID_STATE);
}

// ---- consumer（READ_EXISTING）打开语义 ----
TEST(SQLiteSchemaRoleTest, ConsumerOpensExistingDatabaseWithoutStaticDdl)
{
    using Metric = sqlite_engine_test::MetricTab;
    const std::string path = TempDatabasePath("consumer-existing");
    {
        SQLiteEngine owner(path, &Ddl::kSqliteEngineTestDatabase, kTestProfile);
        EngineOpenRequest open;
        open.mode = StorageOpenMode::READ_WRITE_CREATE;
        ASSERT_TRUE(owner.Open(open).ok());
        ASSERT_TRUE(owner.Prepare().ok());
        Metric record;
        record.name = "seed";
        ASSERT_TRUE(owner.Insert(RecordSchema<Metric>::Meta(), TableScope::Static(), &record, nullptr).ok());
    }
    SQLiteEngine consumer(path, &Ddl::kSqliteEngineTestDatabase, kTestProfile);
    EngineOpenRequest open;
    open.mode = StorageOpenMode::READ_EXISTING;
    ASSERT_TRUE(consumer.Open(open).ok());
    ASSERT_TRUE(consumer.Prepare().ok());
    // consumer 不执行任何静态 DDL（Prepare 是纯状态推进）
    EXPECT_EQ(SQLiteEngineTestPeer::Snapshot(consumer).staticDdlExecutions, 0U);
    // 数据可见
    std::vector<Metric> rows;
    auto sink = ResultSink::For(rows);
    ASSERT_TRUE(consumer
                    .Select(
                        RecordSchema<Metric>::Meta(), TableScope::Static(), Eq(&Metric::name, std::string("seed")).data,
                        {}, sink)
                    .ok());
    EXPECT_EQ(rows.size(), 1U);
    // consumer 仍可惰性创建动态表
    using Task = sqlite_engine_test::TaskTab;
    Task task;
    task.note = "lazy";
    const auto inserted = consumer.Insert(RecordSchema<Task>::Meta(), TableScope::Process(7U), &task, nullptr);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
}

// ---- owner 重复打开已发布库：齐备性预检跳过静态 DDL ----
TEST(SQLiteSchemaRoleTest, OwnerReopensPublishedDatabaseWithoutReplayingStaticDdl)
{
    using Metric = sqlite_engine_test::MetricTab;
    const std::string path = TempDatabasePath("owner-reopen");
    {
        SQLiteEngine owner(path, &Ddl::kSqliteEngineTestDatabase, kTestProfile);
        EngineOpenRequest open;
        open.mode = StorageOpenMode::READ_WRITE_CREATE;
        ASSERT_TRUE(owner.Open(open).ok());
        ASSERT_TRUE(owner.Prepare().ok());
        // 冷库首开执行静态 DDL 发布
        EXPECT_GT(SQLiteEngineTestPeer::Snapshot(owner).staticDdlExecutions, 0U);
        Metric record;
        record.name = "seed";
        ASSERT_TRUE(owner.Insert(RecordSchema<Metric>::Meta(), TableScope::Static(), &record, nullptr).ok());
    }
    SQLiteEngine second(path, &Ddl::kSqliteEngineTestDatabase, kTestProfile);
    EngineOpenRequest open;
    open.mode = StorageOpenMode::READ_WRITE_CREATE;
    ASSERT_TRUE(second.Open(open).ok());
    ASSERT_TRUE(second.Prepare().ok());
    // schema 已齐备：预检命中，不重放任何静态 DDL
    EXPECT_EQ(SQLiteEngineTestPeer::Snapshot(second).staticDdlExecutions, 0U);
    // 既有数据可见，且可继续写入
    std::vector<Metric> rows;
    auto sink = ResultSink::For(rows);
    ASSERT_TRUE(second
                    .Select(
                        RecordSchema<Metric>::Meta(), TableScope::Static(), Eq(&Metric::name, std::string("seed")).data,
                        {}, sink)
                    .ok());
    EXPECT_EQ(rows.size(), 1U);
    Metric another;
    another.name = "second";
    ASSERT_TRUE(second.Insert(RecordSchema<Metric>::Meta(), TableScope::Static(), &another, nullptr).ok());
}

TEST(SQLiteSchemaRoleTest, ConsumerFailsExplicitlyWhenDatabaseFileMissing)
{
    const std::string path = TempDatabasePath("consumer-missing");
    SQLiteEngine consumer(path, &Ddl::kSqliteEngineTestDatabase, kTestProfile);
    EngineOpenRequest open;
    open.mode = StorageOpenMode::READ_EXISTING; // 不建文件
    const auto opened = consumer.Open(open);
    EXPECT_FALSE(opened.ok());
}

TEST(SQLiteSchemaRoleTest, OwnerPrepareIsIdempotentOnExistingSchema)
{
    const std::string path = TempDatabasePath("owner-idempotent");
    {
        SQLiteEngine owner(path, &Ddl::kSqliteEngineTestDatabase, kTestProfile);
        EngineOpenRequest open;
        open.mode = StorageOpenMode::READ_WRITE_CREATE;
        ASSERT_TRUE(owner.Open(open).ok());
        ASSERT_TRUE(owner.Prepare().ok());
    }
    // 第二个 owner 进程：CREATE IF NOT EXISTS 幂等补全，不报漂移
    SQLiteEngine second(path, &Ddl::kSqliteEngineTestDatabase, kTestProfile);
    EngineOpenRequest open;
    open.mode = StorageOpenMode::READ_WRITE_CREATE;
    ASSERT_TRUE(second.Open(open).ok());
    ASSERT_TRUE(second.Prepare().ok());
    using Metric = sqlite_engine_test::MetricTab;
    Metric record;
    record.name = "after-reopen";
    ASSERT_TRUE(second.Insert(RecordSchema<Metric>::Meta(), TableScope::Static(), &record, nullptr).ok());
}

// ---- uint64 宽列范围与排序 ----
TEST(SQLiteUnsignedIntegerTest, WideUnsignedRangesSortAcrossSignedHalves)
{
    using Row = unsigned_mapping_test::UnsignedRow;
    const std::string path = TempDatabasePath("unsigned-integer");
    SQLiteEngine engine(path, &Ddl::kUnsignedMappingDatabase, kTestProfile);
    EngineOpenRequest open;
    open.mode = StorageOpenMode::READ_WRITE_CREATE;
    ASSERT_TRUE(engine.Open(open).ok());
    ASSERT_TRUE(engine.Prepare().ok());

    const uint64_t low = 0;
    const uint64_t signedMax = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    const uint64_t high = signedMax + 1U;
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    const uint64_t values[] = {low, signedMax, high, maximum};
    for (uint64_t index = 0; index < 4U; ++index) {
        Row row;
        row.id = index + 1U;
        row.wideValue = values[index];
        row.narrowValue = 7U;
        row.enabled = false;
        row.signedValue = -7;
        ASSERT_TRUE(engine.Insert(RecordSchema<Row>::Meta(), TableScope::Static(), &row, nullptr).ok());
    }
    const auto selectIds = [&](const Condition<Row>& where, std::optional<Order> order = std::nullopt) {
        QueryOptions options;
        if (order.has_value()) {
            options.sorts.emplace_back(RecordSchema<Row>::template FieldIdOf(&Row::wideValue), *order);
        }
        std::vector<Row> rows;
        auto sink = ResultSink::For(rows);
        const auto result = engine.Select(RecordSchema<Row>::Meta(), TableScope::Static(), where.data, options, sink);
        EXPECT_TRUE(result.ok()) << result.diagnostic;
        std::vector<uint64_t> ids;
        for (const auto& row : rows) {
            ids.push_back(row.id);
        }
        std::sort(ids.begin(), ids.end()); // 范围语义只约束集合
        return ids;
    };
    EXPECT_EQ(selectIds(Lt(&Row::wideValue, signedMax)), (std::vector<uint64_t>{1}));
    EXPECT_EQ(selectIds(Lt(&Row::wideValue, high)), (std::vector<uint64_t>{1, 2}));
    EXPECT_EQ(selectIds(Le(&Row::wideValue, signedMax)), (std::vector<uint64_t>{1, 2}));
    EXPECT_EQ(selectIds(Le(&Row::wideValue, high)), (std::vector<uint64_t>{1, 2, 3}));
    EXPECT_EQ(selectIds(Gt(&Row::wideValue, signedMax)), (std::vector<uint64_t>{3, 4}));
    EXPECT_EQ(selectIds(Gt(&Row::wideValue, high)), (std::vector<uint64_t>{4}));
    EXPECT_EQ(selectIds(Ge(&Row::wideValue, signedMax)), (std::vector<uint64_t>{2, 3, 4}));
    EXPECT_EQ(selectIds(Ge(&Row::wideValue, high)), (std::vector<uint64_t>{3, 4}));
    EXPECT_EQ(selectIds(Eq(&Row::wideValue, low), Order::ASCENDING), (std::vector<uint64_t>{1}));
    EXPECT_EQ(selectIds(Eq(&Row::wideValue, maximum)), (std::vector<uint64_t>{4}));

    // 无符号排序：宽列升序跨半区正确
    QueryOptions options;
    options.sorts.emplace_back(RecordSchema<Row>::template FieldIdOf(&Row::wideValue), Order::ASCENDING);
    std::vector<Row> rows;
    auto sink = ResultSink::For(rows);
    ASSERT_TRUE(engine.Select(RecordSchema<Row>::Meta(), TableScope::Static(), All<Row>().data, options, sink).ok());
    ASSERT_EQ(rows.size(), 4U);
    EXPECT_EQ(rows[0].wideValue, low);
    EXPECT_EQ(rows[1].wideValue, signedMax);
    EXPECT_EQ(rows[2].wideValue, high);
    EXPECT_EQ(rows[3].wideValue, maximum);
    EXPECT_EQ(rows[3].narrowValue, 7U); // 同行其余列精确回填
}

// INCREMENT 域合同（SQLite 表示域边界）：SQLite INTEGER 为有符号 64 位，
// 无符号值按位型存取；算术越界若不设防会静默转 REAL（浮点化）或按位型回绕。
// 修复后幅值 > INT64_MAX（含 INT64_MIN 增量）显式 INVALID_ARGUMENT，
// 基值/结果越出 [0, INT64_MAX] 与有符号上下越界经 CASE 守卫以约束错误原子
// 拒绝——不回绕、不截断、不浮点化。位型域 [2^63, 2^64-1] 的增量保守拒绝
// （MariaDB 支持无符号全域，表示域差异见交付说明）。
TEST(SQLiteUnsignedIntegerTest, IncrementDomainGuardsOnUnsignedColumn)
{
    using Row = unsigned_mapping_test::UnsignedRow;
    const std::string path = TempDatabasePath("unsigned-increment");
    SQLiteEngine engine(path, &Ddl::kUnsignedMappingDatabase, kTestProfile);
    EngineOpenRequest open;
    open.mode = StorageOpenMode::READ_WRITE_CREATE;
    ASSERT_TRUE(engine.Open(open).ok());
    ASSERT_TRUE(engine.Prepare().ok());

    Row row;
    row.id = 1;
    row.wideValue = 2;
    row.signedValue = 0;
    ASSERT_TRUE(engine.Insert(RecordSchema<Row>::Meta(), TableScope::Static(), &row, nullptr).ok());

    const auto updateWide = [&](ChangeKind kind, const DbScalar& delta) {
        return engine.Update(
            RecordSchema<Row>::Meta(), TableScope::Static(), Eq(&Row::id, uint64_t{1}).data,
            UpdateData{LogicalChange{RecordSchema<Row>::template FieldIdOf(&Row::wideValue), kind, delta}}, nullptr);
    };
    const auto updateSigned = [&](ChangeKind kind, const DbScalar& delta) {
        return engine.Update(
            RecordSchema<Row>::Meta(), TableScope::Static(), Eq(&Row::id, uint64_t{1}).data,
            UpdateData{LogicalChange{RecordSchema<Row>::template FieldIdOf(&Row::signedValue), kind, delta}}, nullptr);
    };
    const auto readRow = [&]() {
        std::vector<Row> rows;
        auto sink = ResultSink::For(rows);
        const auto result = engine.Select(
            RecordSchema<Row>::Meta(), TableScope::Static(), Eq(&Row::id, uint64_t{1}).data, QueryOptions{}, sink);
        EXPECT_TRUE(result.ok()) << result.diagnostic;
        return rows.empty() ? Row{} : rows.front();
    };

    // 2-1=1、1+1=2、零增量幂等（与 MariaDB 一致的基础算术）
    ASSERT_TRUE(updateWide(ChangeKind::INCREMENT, DbScalar{int64_t{-1}}).ok());
    EXPECT_EQ(readRow().wideValue, 1U);
    ASSERT_TRUE(updateWide(ChangeKind::INCREMENT, DbScalar{uint64_t{1}}).ok());
    EXPECT_EQ(readRow().wideValue, 2U);
    ASSERT_TRUE(updateWide(ChangeKind::INCREMENT, DbScalar{uint64_t{0}}).ok());
    EXPECT_EQ(readRow().wideValue, 2U);

    // 零减一：明确错误 + 数据保持
    ASSERT_TRUE(updateWide(ChangeKind::SET, DbScalar{uint64_t{0}}).ok());
    EXPECT_FALSE(updateWide(ChangeKind::INCREMENT, DbScalar{int64_t{-1}}).ok());
    EXPECT_EQ(readRow().wideValue, 0U);

    // INT64_MIN 增量（幅值 2^63）：超出本后端可承载幅值域，显式拒绝
    const auto int64Min = updateWide(ChangeKind::INCREMENT, DbScalar{INT64_MIN});
    EXPECT_FALSE(int64Min.ok());
    EXPECT_EQ(int64Min.code, DbCode::INVALID_ARGUMENT);
    EXPECT_EQ(readRow().wideValue, 0U);

    // 位型基值（≥ 2^63）上的增量：保守拒绝，不依赖位型回绕
    ASSERT_TRUE(updateWide(ChangeKind::SET, DbScalar{uint64_t{1} << 63}).ok());
    EXPECT_EQ(readRow().wideValue, uint64_t{1} << 63);
    EXPECT_FALSE(updateWide(ChangeKind::INCREMENT, DbScalar{uint64_t{1}}).ok());
    EXPECT_EQ(readRow().wideValue, uint64_t{1} << 63);

    // 有符号列上下界：守卫拒绝（不浮点化）
    ASSERT_TRUE(updateSigned(ChangeKind::SET, DbScalar{INT64_MAX}).ok());
    EXPECT_FALSE(updateSigned(ChangeKind::INCREMENT, DbScalar{int64_t{1}}).ok());
    EXPECT_EQ(readRow().signedValue, INT64_MAX);
    ASSERT_TRUE(updateSigned(ChangeKind::SET, DbScalar{INT64_MIN}).ok());
    EXPECT_FALSE(updateSigned(ChangeKind::INCREMENT, DbScalar{int64_t{-1}}).ok());
    EXPECT_EQ(readRow().signedValue, INT64_MIN);
}

TEST_F(SQLiteEngineTest, TwoPidScopesIsolateRowsAndDropClearsBoth)
{
    using Task = sqlite_engine_test::TaskTab;
    const auto& meta = RecordSchema<Task>::Meta();
    for (uint64_t pid = 1U; pid <= 2U; ++pid) {
        Task task;
        task.note = "note" + std::to_string(pid);
        const auto inserted = engine->Insert(meta, TableScope::Process(pid), &task, nullptr);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
        EXPECT_EQ(*inserted.value, 1U); // 各 scope 独立序列
    }
    const auto count1 = engine->Count(meta, TableScope::Process(1U), All<Task>().data);
    ASSERT_TRUE(count1.ok());
    EXPECT_EQ(*count1.value, 1U);
    const auto dropped = engine->Clear(meta, TableScope::Process(0U), ClearMode::DROP_DYNAMIC_SCOPE);
    ASSERT_TRUE(dropped.ok()) << dropped.diagnostic;
    EXPECT_EQ(dropped.affectedRows, 2U);
}

// ---- 外键语义（独立目录：目标列为非首字段 + 级联删除）----
class SQLiteEngineForeignKeyTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        databasePath = TempDatabasePath("fk");
        engine = std::make_unique<SQLiteEngine>(databasePath, &Ddl::kFkSemanticsDatabase, kTestProfile);
        EngineOpenRequest open;
        open.mode = StorageOpenMode::READ_WRITE_CREATE;
        ASSERT_TRUE(engine->Open(open).ok());
        ASSERT_TRUE(engine->Prepare().ok());
    }

    using Parent = fk_semantics_test::FkParentTab;
    using Child = fk_semantics_test::FkChildTab;

    std::string databasePath;
    std::unique_ptr<SQLiteEngine> engine;
};

TEST_F(SQLiteEngineForeignKeyTest, ForeignKeysCascadeOnParentDelete)
{
    Parent parent;
    parent.code = 100U;
    ASSERT_TRUE(engine->Insert(RecordSchema<Parent>::Meta(), TableScope::Static(), &parent, nullptr).ok());
    for (uint64_t i = 0; i < 3U; ++i) {
        Child child;
        child.parent_code = 100U;
        ASSERT_TRUE(engine->Insert(RecordSchema<Child>::Meta(), TableScope::Static(), &child, nullptr).ok());
    }
    std::vector<Child> children;
    auto sink = ResultSink::For(children);
    ASSERT_TRUE(engine->Select(RecordSchema<Child>::Meta(), TableScope::Static(), All<Child>().data, {}, sink).ok());
    ASSERT_EQ(children.size(), 3U);
    // 删除父行：子行级联删除
    const auto deleted
        = engine->Delete(RecordSchema<Parent>::Meta(), TableScope::Static(), Eq(&Parent::code, 100U).data, nullptr);
    ASSERT_TRUE(deleted.ok()) << deleted.diagnostic;
    EXPECT_EQ(*deleted.value, 1U);
    std::vector<Child> after;
    auto afterSink = ResultSink::For(after);
    ASSERT_TRUE(
        engine->Select(RecordSchema<Child>::Meta(), TableScope::Static(), All<Child>().data, {}, afterSink).ok());
    EXPECT_TRUE(after.empty());
}

TEST_F(SQLiteEngineForeignKeyTest, ForeignKeyResolvesNonFirstTargetColumnFromCatalog)
{
    Parent parent;
    parent.code = 55U; // 目标列 code 是父表第二列（非主键、UNIQUE）
    ASSERT_TRUE(engine->Insert(RecordSchema<Parent>::Meta(), TableScope::Static(), &parent, nullptr).ok());
    Child valid;
    valid.parent_code = 55U;
    ASSERT_TRUE(engine->Insert(RecordSchema<Child>::Meta(), TableScope::Static(), &valid, nullptr).ok());
    // 引用不存在的父值：外键约束拒绝
    Child orphan;
    orphan.parent_code = 999U;
    const auto rejected = engine->Insert(RecordSchema<Child>::Meta(), TableScope::Static(), &orphan, nullptr);
    EXPECT_EQ(rejected.code, DbCode::CONFLICT);
}

// ---- 字段投影（引擎级）：SQLite 按 projectedFields 实际生成 SELECT 列清单，
// 空投影 = 全列直读（列序 = 声明序 = fieldId，无投影数组分配）；非法投影列
// （超出目录字段数的编号）明确报错。投影行为合同（未投影字段保持默认值/
// 过滤排序字段不要求投影）由 backend conformance 的 Projection 用例覆盖。
TEST(SQLiteEngineProjectionTest, ProjectsDeclaredFieldsAndRejectsIllegalColumn)
{
    using Row = sqlite_engine_test::MetricTab;
    const auto& meta = RecordSchema<Row>::Meta();
    const std::string path = TempDatabasePath("projection");
    SQLiteEngine engine(path, &Ddl::kSqliteEngineTestDatabase, kTestProfile);
    EngineOpenRequest open;
    open.mode = StorageOpenMode::READ_WRITE_CREATE;
    ASSERT_TRUE(engine.Open(open).ok());
    ASSERT_TRUE(engine.Prepare().ok());

    Row row;
    row.name = "metric";
    row.count = 42;
    row.score = 3.5;
    row.payload = {0x01, 0x02, 0x03};
    ASSERT_TRUE(engine.Insert(meta, TableScope::Static(), &row, nullptr).ok());
    row.id = 1U;

    // 非空投影：只读 name/count 两列，其余成员保持默认值。
    {
        QueryOptions options;
        options.projectedFields
            = {RecordSchema<Row>::template FieldIdOf(&Row::name), RecordSchema<Row>::template FieldIdOf(&Row::count)};
        std::vector<Row> rows;
        auto sink = ResultSink::For(rows);
        const auto selected = engine.Select(meta, TableScope::Static(), Eq(&Row::id, uint64_t{1}).data, options, sink);
        ASSERT_TRUE(selected.ok()) << selected.diagnostic;
        ASSERT_EQ(rows.size(), 1U);
        EXPECT_EQ(rows[0].name, "metric");
        EXPECT_EQ(rows[0].count, 42);
        EXPECT_EQ(rows[0].id, 0U);
        EXPECT_EQ(rows[0].score, 0.0);
        EXPECT_TRUE(rows[0].payload.empty());
    }

    // 空投影：全列直读，行完整等值。
    {
        std::vector<Row> rows;
        auto sink = ResultSink::For(rows);
        const auto selected
            = engine.Select(meta, TableScope::Static(), Eq(&Row::id, uint64_t{1}).data, QueryOptions{}, sink);
        ASSERT_TRUE(selected.ok()) << selected.diagnostic;
        ASSERT_EQ(rows.size(), 1U);
        EXPECT_EQ(rows[0].name, "metric");
        EXPECT_EQ(rows[0].count, 42);
        EXPECT_DOUBLE_EQ(rows[0].score, 3.5);
        EXPECT_EQ(rows[0].payload, row.payload);
    }

    // 非法投影列（未映射编号）：明确报错，不静默全列。
    {
        QueryOptions options;
        options.projectedFields = {kUnknownFieldId};
        std::vector<Row> rows;
        auto sink = ResultSink::For(rows);
        const auto selected = engine.Select(meta, TableScope::Static(), Eq(&Row::id, uint64_t{1}).data, options, sink);
        EXPECT_EQ(selected.code, DbCode::INVALID_ARGUMENT);
        EXPECT_TRUE(rows.empty());
    }
    {
        QueryOptions options;
        options.projectedFields = {static_cast<uint16_t>(meta.fieldCount)};
        std::vector<Row> rows;
        auto sink = ResultSink::For(rows);
        const auto selected = engine.Select(meta, TableScope::Static(), Eq(&Row::id, uint64_t{1}).data, options, sink);
        EXPECT_EQ(selected.code, DbCode::INVALID_ARGUMENT);
    }
}
} // namespace
