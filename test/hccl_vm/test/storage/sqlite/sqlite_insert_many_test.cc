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
 * See the License in the root of the software repository for the full text of the License.
 */

// 文件职责：验证 SQLiteEngine 批量插入 override（批内语句复用）的全部合同：
// 入参校验顺序与时序、逐行真实主键、显式/混合主键、TEXT/BLOB/高位 uint64/
// Derived 值批次往返、中途行空指针/Unique/FK 约束失败的行号归属与回滚、
// 失败后连接可用性，以及与基类默认逐行实现的等价对照；经链接期
// --wrap=sqlite3_prepare_v2 观察窗证明同表 N 行显式 prepare 次数 BASE=N、OPT=1。
// 包装只链接本测试目标，不改变产品链接选项；观察点不是产品计数器。
#include <gtest/gtest.h>

#include <sqlite3.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "storage_backends/sqlite/engine/sqlite_engine.h"
#include "storage/query_expression.h"
#include "storage/record_contract.h"
#include "storage/storage_catalog.h"

// ---- 链接期观察窗（--wrap=sqlite3_prepare_v2，仅本测试目标）：调用 real 函数并
// 只统计目标表的 INSERT prepare；DDL/PRAGMA/SELECT 等其他语句不计入 ----
extern "C" int
__real_sqlite3_prepare_v2(sqlite3* db, const char* zSql, int nByte, sqlite3_stmt** ppStmt, const char** pzTail);

namespace {
std::atomic<int> g_targetInsertPrepares{0};
std::string g_preparePrefix; ///< 非空时只统计以该前缀开头的语句；空 = 不统计
} // namespace

extern "C" int
__wrap_sqlite3_prepare_v2(sqlite3* db, const char* zSql, int nByte, sqlite3_stmt** ppStmt, const char** pzTail)
{
    if (!g_preparePrefix.empty() && zSql != nullptr
        && std::strncmp(zSql, g_preparePrefix.c_str(), g_preparePrefix.size()) == 0) {
        g_targetInsertPrepares.fetch_add(1);
    }
    return __real_sqlite3_prepare_v2(db, zSql, nByte, ppStmt, pzTail);
}

// ---- 测试目录：变体行（TEXT/BLOB/高位 uint64/REAL）、Unique 约束、FK 约束、
// Derived 派生列、动态 PID 表 ----
namespace sqlite_insert_many_test {
struct BatchRow {
    uint64_t id{0};
    std::string name;
    int64_t count{0};
    double score{0.0};
    std::vector<uint8_t> payload;
    uint64_t wide{0};
};
struct UniqueNameRow {
    uint64_t id{0};
    std::string name;
};
struct FkParentRow {
    uint64_t id{0};
    uint64_t code{0};
};
struct FkChildRow {
    uint64_t id{0};
    uint64_t parentCode{0};
};
struct DerivedRow {
    uint64_t id{0};
    uint64_t base{0};
    uint64_t size{0};
    uint64_t endPtr{0};
};
struct DynRow {
    uint64_t id{0};
    std::string note;
};
} // namespace sqlite_insert_many_test

using namespace HcclSim::Storage::Ddl;
#define Field HCCL_VM_DDL_FIELD
#define Version HCCL_VM_DDL_VERSION
#define PrimaryKey HCCL_VM_DDL_PRIMARY_KEY
#define Unique HCCL_VM_DDL_UNIQUE
#define Index HCCL_VM_DDL_INDEX
#define ForeignKey HCCL_VM_DDL_FOREIGN_KEY
#define Scope Ddl::ScopeKind
#define Derived Ddl::Derived

#define SQLITE_INSERT_MANY_TEST_DATABASE(Table)                                                                       \
    Table(                                                                                                            \
        sqlite_insert_many_test::BatchRow, (Version(1), Scope::Static, Field(id, AutoIncrement), Field(name),         \
                                            Field(count), Field(score), Field(payload), Field(wide), PrimaryKey(id))) \
        Table(                                                                                                        \
            sqlite_insert_many_test::UniqueNameRow,                                                                   \
            (Version(1), Scope::Static, Field(id, AutoIncrement), Field(name), PrimaryKey(id), Unique(name)))         \
            Table(                                                                                                    \
                sqlite_insert_many_test::FkParentRow,                                                                 \
                (Version(1), Scope::Static, Field(id, AutoIncrement), Field(code), PrimaryKey(id), Unique(code)))     \
                Table(                                                                                                \
                    sqlite_insert_many_test::FkChildRow,                                                              \
                    (Version(1), Scope::Static, Field(id, AutoIncrement), Field(parentCode), PrimaryKey(id),          \
                     ForeignKey(parentCode, sqlite_insert_many_test::FkParentRow::code, OnDelete::Cascade)))          \
                    Table(                                                                                            \
                        sqlite_insert_many_test::DerivedRow,                                                          \
                        (Version(1), Scope::Static, Field(id, AutoIncrement), Field(base), Field(size),               \
                         Field(endPtr, Derived), PrimaryKey(id)))                                                     \
                        Table(                                                                                        \
                            sqlite_insert_many_test::DynRow,                                                          \
                            (Version(1), Scope::DynamicPid, Field(id, AutoIncrement), Field(note), PrimaryKey(id)))

#define HCCL_VM_DDL_CURRENT_DATABASE "SQLITE_INSERT_MANY_TEST_DB"
HCCL_VM_DEFINE_DATABASE(kSqliteInsertManyTestDatabase, SQLITE_INSERT_MANY_TEST_DB, SQLITE_INSERT_MANY_TEST_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE

#undef Field
#undef Version
#undef PrimaryKey
#undef Unique
#undef Index
#undef ForeignKey
#undef Scope
#undef Derived

namespace HcclSim::Storage::Ddl {
// DerivedRow 派生列 hook：endPtr = base + size（编码入库；解码忽略，基址/大小权威）
template <>
struct RecordCodecHooks<sqlite_insert_many_test::DerivedRow> :
    DefaultRecordCodecHooks<sqlite_insert_many_test::DerivedRow> {
    static std::optional<DbScalar>
    EncodeDerivedField(const sqlite_insert_many_test::DerivedRow& record, uint16_t fieldId)
    {
        if (fieldId == 3U) {
            return DbScalar{record.base + record.size};
        }
        return std::nullopt;
    }

    static bool IgnoreDecodedField(uint16_t fieldId) { return fieldId == 3U; }
};
} // namespace HcclSim::Storage::Ddl

namespace {
using namespace HcclSim::Storage;
using BatchRow = sqlite_insert_many_test::BatchRow;

constexpr SqliteConnectionProfile kTestProfile{10000, false, 200};

std::string TempDatabasePath(const std::string& tag)
{
    static std::atomic<uint64_t> counter{0};
    const std::string path
        = testing::TempDir() + "sqlite_insert_many_" + tag + "_" + std::to_string(counter.fetch_add(1)) + ".db";
    (void)std::remove(path.c_str());
    return path;
}

/// 直接读目标表某行的单个整数列（独立只读连接；Derived 列解码忽略时的物理值核对）
std::optional<uint64_t>
ReadUint64Column(const std::string& databasePath, const std::string& table, const std::string& column, uint64_t id)
{
    sqlite3* raw = nullptr;
    if (sqlite3_open(databasePath.c_str(), &raw) != SQLITE_OK) {
        sqlite3_close(raw);
        return std::nullopt;
    }
    sqlite3_stmt* stmt = nullptr;
    std::optional<uint64_t> value;
    const std::string sql = "SELECT " + column + " FROM " + table + " WHERE id = ?";
    if (sqlite3_prepare_v2(raw, sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(id));
        if (sqlite3_step(stmt) == SQLITE_ROW && sqlite3_column_type(stmt, 0) != SQLITE_NULL) {
            value = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0)); // 位型还原（与引擎读回一致）
        }
    }
    sqlite3_finalize(stmt);
    sqlite3_close(raw);
    return value;
}

// ---- 夹具：每个用例独立临时库与引擎 ----
class SqliteInsertManyTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetEngine(); // 每次重置都取全新临时库（等价对照两侧互不携带状态）
        g_preparePrefix.clear();
    }

    void TearDown() override
    {
        if (engine != nullptr) {
            (void)engine->Close();
        }
        g_preparePrefix.clear();
    }

    void ResetEngine()
    {
        if (engine != nullptr) {
            (void)engine->Close();
        }
        databasePath = TempDatabasePath("engine"); // 全新库：重置即冷启动
        engine = std::make_unique<SQLiteEngine>(databasePath, &Ddl::kSqliteInsertManyTestDatabase, kTestProfile);
        EngineOpenRequest open;
        open.mode = StorageOpenMode::READ_WRITE_CREATE;
        const auto opened = engine->Open(open);
        ASSERT_TRUE(opened.ok()) << opened.diagnostic;
        const auto prepared = engine->Prepare();
        ASSERT_TRUE(prepared.ok()) << prepared.diagnostic;
    }

    const Ddl::CatalogTableMeta& Meta() const { return RecordSchema<BatchRow>::Meta(); }

    /// 行数组 → 原始指针视图（地址仅在本调用栈内使用，不复制业务载荷）
    template <typename T>
    static std::vector<const void*> PointerView(const std::vector<T>& rows)
    {
        std::vector<const void*> pointers;
        pointers.reserve(rows.size());
        for (const T& row : rows) {
            pointers.push_back(&row);
        }
        return pointers;
    }

    /// 观察窗：只统计目标表 INSERT 的 prepare 调用
    struct PrepareCounter {
        explicit PrepareCounter(const std::string& prefix)
        {
            g_targetInsertPrepares.store(0);
            g_preparePrefix = prefix;
        }
        ~PrepareCounter() { g_preparePrefix.clear(); }
        int value() const { return g_targetInsertPrepares.load(); }
    };

    /// 造 N 行可区分数据（第 i 行：name/计数/分数/payload/高位 wide 均按 i 变化）
    static void MakeRows(std::vector<BatchRow>& rows, std::size_t count)
    {
        rows.clear();
        for (std::size_t i = 0; i < count; ++i) {
            BatchRow row;
            row.name = "row-" + std::to_string(i);
            row.count = static_cast<int64_t>(i) * 7 - 3;
            row.score = 0.5 * static_cast<double>(i);
            row.payload = std::vector<uint8_t>(i % 13U, static_cast<uint8_t>(i & 0xFF));
            if (i % 4U == 0U) {
                row.wide = std::numeric_limits<uint64_t>::max() - i; // 高位 uint64
            } else {
                row.wide = static_cast<uint64_t>(i);
            }
            rows.push_back(std::move(row));
        }
    }

    static std::vector<const void*> PointersOf(const std::vector<BatchRow>& rows) { return PointerView(rows); }

    /// 全表回读（id 升序；只比较业务字段与 id 的对应关系）
    DbResult<std::vector<BatchRow>> SelectAllRows()
    {
        std::vector<BatchRow> rows;
        auto sink = ResultSink::For(rows);
        QueryOptions options;
        options.sorts.emplace_back(RecordSchema<BatchRow>::FieldIdOf(&BatchRow::id), Order::ASCENDING);
        const auto result = engine->Select(Meta(), TableScope::Static(), All<BatchRow>().data, options, sink);
        if (!result.ok()) {
            return {result.code, std::nullopt, result.affectedRows, result.diagnostic};
        }
        return {DbCode::OK, std::optional<std::vector<BatchRow>>{std::move(rows)}, result.affectedRows, {}};
    }

    std::string databasePath;
    std::unique_ptr<SQLiteEngine> engine;
};

// ---- 入参合同与既有时序 ----

TEST_F(SqliteInsertManyTest, CountZeroIsTrivialSuccessWithoutTouchingArguments)
{
    // 基类默认实现的第一检查：count=0 平凡成功（不校验指针/事务状态、零副作用）
    TransactionBinding binding;
    const auto empty = engine->InsertMany(Meta(), TableScope::Static(), nullptr, 0, nullptr, nullptr);
    EXPECT_TRUE(empty.ok());
    EXPECT_EQ(empty.affectedRows, 0U);
    // 未 Begin 的 binding 上 count=0 同样平凡成功（引擎层不检查事务终态）
    const auto zeroWithBinding = engine->InsertMany(Meta(), TableScope::Static(), nullptr, 0, nullptr, &binding);
    EXPECT_TRUE(zeroWithBinding.ok());
}

TEST_F(SqliteInsertManyTest, NullArgumentPointersAreRejectedBeforeAnyWork)
{
    std::vector<BatchRow> rows;
    MakeRows(rows, 3);
    auto pointers = PointersOf(rows);
    std::array<uint64_t, 3> ids{};
    TransactionBinding binding;
    ASSERT_TRUE(engine->Begin(binding).ok());

    const auto nullRecords = engine->InsertMany(Meta(), TableScope::Static(), nullptr, 3, ids.data(), &binding);
    EXPECT_EQ(nullRecords.code, DbCode::INVALID_ARGUMENT);
    const auto nullIds = engine->InsertMany(Meta(), TableScope::Static(), pointers.data(), 3, nullptr, &binding);
    EXPECT_EQ(nullIds.code, DbCode::INVALID_ARGUMENT);
    const auto nullTransaction
        = engine->InsertMany(Meta(), TableScope::Static(), pointers.data(), 3, ids.data(), nullptr);
    EXPECT_EQ(nullTransaction.code, DbCode::INVALID_ARGUMENT);
    (void)engine->Rollback(binding);
    // 全部拒绝路径零副作用：库内无行（读操作须在事务结束之后）
    const auto rowsAfter = SelectAllRows();
    ASSERT_TRUE(rowsAfter.ok());
    EXPECT_TRUE(rowsAfter.value->empty());
}

TEST_F(SqliteInsertManyTest, BatchAboveMaxRowsIsRejected)
{
    std::vector<BatchRow> rows;
    MakeRows(rows, kInsertBatchMaxRows + 1);
    auto pointers = PointersOf(rows);
    std::vector<uint64_t> ids(kInsertBatchMaxRows + 1, 0);
    TransactionBinding binding;
    ASSERT_TRUE(engine->Begin(binding).ok());
    const auto rejected = engine->InsertMany(
        Meta(), TableScope::Static(), pointers.data(), kInsertBatchMaxRows + 1, ids.data(), &binding);
    EXPECT_EQ(rejected.code, DbCode::INVALID_ARGUMENT);
    (void)engine->Rollback(binding);
}

TEST_F(SqliteInsertManyTest, NotBegunTransactionFailsWithRowZeroSuffix)
{
    std::vector<BatchRow> rows;
    MakeRows(rows, 2);
    auto pointers = PointersOf(rows);
    std::array<uint64_t, 2> ids{};
    TransactionBinding notBegun;
    const auto state = engine->InsertMany(Meta(), TableScope::Static(), pointers.data(), 2, ids.data(), &notBegun);
    EXPECT_EQ(state.code, DbCode::INVALID_STATE);
    EXPECT_NE(state.diagnostic.find(" at row 0"), std::string::npos) << state.diagnostic;
}

// ---- 主键与内容合同 ----

TEST_F(SqliteInsertManyTest, AutoKeysAreSequentialAndOrdered)
{
    constexpr std::size_t kRows = 16;
    std::vector<BatchRow> rows;
    MakeRows(rows, kRows);
    auto pointers = PointersOf(rows);
    std::array<uint64_t, kRows> ids{};
    TransactionBinding binding;
    ASSERT_TRUE(engine->Begin(binding).ok());
    const auto inserted
        = engine->InsertMany(Meta(), TableScope::Static(), pointers.data(), kRows, ids.data(), &binding);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    EXPECT_EQ(inserted.affectedRows, kRows);
    ASSERT_TRUE(engine->Commit(binding).ok());

    for (std::size_t i = 0; i < kRows; ++i) {
        EXPECT_EQ(ids[i], i + 1) << "row " << i; // 自增主键按输入顺序逐行真实生成
    }
    const auto stored = SelectAllRows();
    ASSERT_TRUE(stored.ok());
    ASSERT_EQ(stored.value->size(), kRows);
    for (std::size_t i = 0; i < kRows; ++i) {
        EXPECT_EQ((*stored.value)[i].name, rows[i].name);
        EXPECT_EQ((*stored.value)[i].count, rows[i].count);
        EXPECT_DOUBLE_EQ((*stored.value)[i].score, rows[i].score);
        EXPECT_EQ((*stored.value)[i].payload, rows[i].payload);
        EXPECT_EQ((*stored.value)[i].wide, rows[i].wide);
    }
}

TEST_F(SqliteInsertManyTest, BatchOfExactlyMaxRowsSucceeds)
{
    constexpr std::size_t kRows = kInsertBatchMaxRows;
    std::vector<BatchRow> rows;
    MakeRows(rows, kRows);
    auto pointers = PointersOf(rows);
    std::vector<uint64_t> ids(kRows, 0);
    TransactionBinding binding;
    ASSERT_TRUE(engine->Begin(binding).ok());
    const auto inserted
        = engine->InsertMany(Meta(), TableScope::Static(), pointers.data(), kRows, ids.data(), &binding);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    EXPECT_EQ(inserted.affectedRows, kRows);
    ASSERT_TRUE(engine->Commit(binding).ok());
    const auto stored = SelectAllRows();
    ASSERT_TRUE(stored.ok());
    EXPECT_EQ(stored.value->size(), kRows);
}

TEST_F(SqliteInsertManyTest, ExplicitAndMixedKeysKeepPerRowGeneratedIds)
{
    // 显式主键原样入库；其后自动行从显式值之后继续（SQLite AUTOINCREMENT 语义）。
    std::vector<BatchRow> rows;
    MakeRows(rows, 4);
    rows[0].id = 100;
    rows[2].id = 50;
    auto pointers = PointersOf(rows);
    std::array<uint64_t, 4> ids{};
    TransactionBinding binding;
    ASSERT_TRUE(engine->Begin(binding).ok());
    const auto inserted = engine->InsertMany(Meta(), TableScope::Static(), pointers.data(), 4, ids.data(), &binding);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    ASSERT_TRUE(engine->Commit(binding).ok());
    EXPECT_EQ(ids[0], 100U);
    EXPECT_EQ(ids[2], 50U);
    EXPECT_GT(ids[1], 0U);
    EXPECT_GT(ids[3], 100U);
    // 内容与真实库内 id 一一对应
    const auto stored = SelectAllRows();
    ASSERT_TRUE(stored.ok());
    ASSERT_EQ(stored.value->size(), 4U);
    for (std::size_t i = 0; i < 4; ++i) {
        const auto found = std::find_if(stored.value->begin(), stored.value->end(), [&](const BatchRow& row) {
            return row.id == ids[i];
        });
        ASSERT_NE(found, stored.value->end());
        EXPECT_EQ(found->name, rows[i].name);
        EXPECT_EQ(found->wide, rows[i].wide);
    }
}

TEST_F(SqliteInsertManyTest, TextBlobAndHighUint64VarietyRoundTrip)
{
    std::vector<BatchRow> rows;
    MakeRows(rows, 8);
    // 追加边界形态：空文本/长文本/空 BLOB/大 BLOB/极高位 uint64 交替
    rows[0].name = "";
    rows[0].payload = {};
    rows[1].name = std::string(4096, 'x');
    rows[1].payload = std::vector<uint8_t>(65536, 0xAB);
    rows[2].wide = std::numeric_limits<uint64_t>::max();
    rows[3].wide = 1ULL << 63;
    rows[4].payload = {0x00};
    rows[5].name = std::string("含中文与\t制表符\1控制字符");
    auto pointers = PointersOf(rows);
    std::array<uint64_t, 8> ids{};
    TransactionBinding binding;
    ASSERT_TRUE(engine->Begin(binding).ok());
    const auto inserted = engine->InsertMany(Meta(), TableScope::Static(), pointers.data(), 8, ids.data(), &binding);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    ASSERT_TRUE(engine->Commit(binding).ok());

    const auto stored = SelectAllRows();
    ASSERT_TRUE(stored.ok());
    ASSERT_EQ(stored.value->size(), 8U);
    for (std::size_t i = 0; i < 8; ++i) {
        const auto found = std::find_if(stored.value->begin(), stored.value->end(), [&](const BatchRow& row) {
            return row.id == ids[i];
        });
        ASSERT_NE(found, stored.value->end()) << "row " << i;
        EXPECT_EQ(found->name, rows[i].name) << "row " << i;
        EXPECT_EQ(found->payload, rows[i].payload) << "row " << i;
        EXPECT_EQ(found->wide, rows[i].wide) << "row " << i;
    }
}

TEST_F(SqliteInsertManyTest, DerivedFieldBindsTransientValuePerRow)
{
    using DerivedRow = sqlite_insert_many_test::DerivedRow;
    std::vector<DerivedRow> rows;
    for (std::size_t i = 0; i < 6; ++i) {
        DerivedRow row;
        row.base = 0x1000 * (i + 1);
        row.size = 0x40 + i;
        rows.push_back(row); // endPtr 由 hook 在发射时计算（SQLITE_TRANSIENT 复制）
    }
    std::vector<const void*> pointers = PointerView(rows);
    std::array<uint64_t, 6> ids{};
    TransactionBinding binding;
    ASSERT_TRUE(engine->Begin(binding).ok());
    const auto inserted = engine->InsertMany(
        RecordSchema<DerivedRow>::Meta(), TableScope::Static(), pointers.data(), 6, ids.data(), &binding);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    ASSERT_TRUE(engine->Commit(binding).ok());

    for (std::size_t i = 0; i < 6; ++i) {
        const auto stored = ReadUint64Column(databasePath, "DerivedRow", "endPtr", ids[i]);
        ASSERT_TRUE(stored.has_value()) << "row " << i;
        EXPECT_EQ(*stored, rows[i].base + rows[i].size) << "row " << i;
    }
}

// ---- 行内失败：行号归属、回滚、失败后可用性 ----

TEST_F(SqliteInsertManyTest, RowZeroNullRecordRejectsBeforeCreatingDynamicTable)
{
    using DynRow = sqlite_insert_many_test::DynRow;
    std::vector<DynRow> rows;
    rows.resize(3);
    std::vector<const void*> pointers{&rows[0], &rows[1], &rows[2]};
    pointers[0] = nullptr; // 第一行空指针
    std::array<uint64_t, 3> ids{};
    TransactionBinding binding;
    ASSERT_TRUE(engine->Begin(binding).ok());
    const auto failed = engine->InsertMany(
        RecordSchema<DynRow>::Meta(), TableScope::Process(4242), pointers.data(), 3, ids.data(), &binding);
    EXPECT_EQ(failed.code, DbCode::INVALID_ARGUMENT);
    EXPECT_NE(failed.diagnostic.find("insert requires a record payload at row 0"), std::string::npos)
        << failed.diagnostic;
    (void)engine->Rollback(binding);
    // 首行空指针在物理表解析之前暴露：动态表未被创建（与逐行 Insert 时序一致；
    // ListTables 须在事务结束之后调用）
    const auto tables = engine->ListTables();
    ASSERT_TRUE(tables.ok());
    EXPECT_EQ(std::count(tables.value->begin(), tables.value->end(), "DynRow_P_4242"), 0);
}

TEST_F(SqliteInsertManyTest, MidBatchNullRecordFailsAtOwnRowIndex)
{
    std::vector<BatchRow> rows;
    MakeRows(rows, 8);
    auto pointers = PointersOf(rows);
    pointers[3] = nullptr; // 第 3 行空指针（不提前预检全部行）
    std::array<uint64_t, 8> ids{};
    TransactionBinding binding;
    ASSERT_TRUE(engine->Begin(binding).ok());
    const auto failed = engine->InsertMany(Meta(), TableScope::Static(), pointers.data(), 8, ids.data(), &binding);
    EXPECT_EQ(failed.code, DbCode::INVALID_ARGUMENT);
    EXPECT_NE(failed.diagnostic.find("insert requires a record payload at row 3"), std::string::npos)
        << failed.diagnostic;
    ASSERT_TRUE(engine->Rollback(binding).ok());
    // 回滚后：无行残留、可开新事务并正常批量插入（未遗留语句/锁）
    const auto stored = SelectAllRows();
    ASSERT_TRUE(stored.ok());
    EXPECT_TRUE(stored.value->empty());

    std::vector<BatchRow> freshRows;
    MakeRows(freshRows, 4);
    auto freshPointers = PointersOf(freshRows);
    std::array<uint64_t, 4> freshIds{};
    TransactionBinding next;
    ASSERT_TRUE(engine->Begin(next).ok());
    const auto inserted
        = engine->InsertMany(Meta(), TableScope::Static(), freshPointers.data(), 4, freshIds.data(), &next);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    ASSERT_TRUE(engine->Commit(next).ok());
    EXPECT_EQ(freshIds[0], 1U); // 回滚撤销序列消耗后从 1 重新开始（WAL 回滚语义）
}

TEST_F(SqliteInsertManyTest, UniqueConstraintMidBatchFailsAtFirstBadRow)
{
    using UniqueNameRow = sqlite_insert_many_test::UniqueNameRow;
    std::vector<UniqueNameRow> rows;
    for (std::size_t i = 0; i < 5; ++i) {
        UniqueNameRow row;
        row.name = (i == 2) ? "dup" : "name-" + std::to_string(i);
        rows.push_back(row);
    }
    UniqueNameRow first;
    first.name = "dup"; // 库内既有行：批次第 2 行与其冲突
    TransactionBinding setup;
    ASSERT_TRUE(engine->Begin(setup).ok());
    const auto seeded = engine->Insert(RecordSchema<UniqueNameRow>::Meta(), TableScope::Static(), &first, &setup);
    ASSERT_TRUE(seeded.ok()) << seeded.diagnostic;
    ASSERT_TRUE(engine->Commit(setup).ok());

    std::vector<const void*> pointers = PointerView(rows);
    std::array<uint64_t, 5> ids{};
    TransactionBinding binding;
    ASSERT_TRUE(engine->Begin(binding).ok());
    const auto failed = engine->InsertMany(
        RecordSchema<UniqueNameRow>::Meta(), TableScope::Static(), pointers.data(), 5, ids.data(), &binding);
    EXPECT_EQ(failed.code, DbCode::CONFLICT);
    EXPECT_NE(failed.diagnostic.find(" at row 2"), std::string::npos) << failed.diagnostic;
    // 后续行不执行：第 3/4 行不落库；本批前两行随回滚撤销
    ASSERT_TRUE(engine->Rollback(binding).ok());
    std::vector<UniqueNameRow> remaining;
    auto sink = ResultSink::For(remaining);
    const auto selected = engine->Select(
        RecordSchema<UniqueNameRow>::Meta(), TableScope::Static(), All<UniqueNameRow>().data, {}, sink);
    ASSERT_TRUE(selected.ok());
    ASSERT_EQ(remaining.size(), 1U); // 只有 setup 阶段那行
    EXPECT_EQ(remaining[0].name, "dup");
    // 失败后可正常开启新事务
    TransactionBinding next;
    ASSERT_TRUE(engine->Begin(next).ok());
    UniqueNameRow another;
    another.name = "after-failure";
    const auto inserted = engine->Insert(RecordSchema<UniqueNameRow>::Meta(), TableScope::Static(), &another, &next);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    ASSERT_TRUE(engine->Commit(next).ok());
}

TEST_F(SqliteInsertManyTest, ForeignKeyMidBatchFailsAndRollsBack)
{
    using FkChildRow = sqlite_insert_many_test::FkChildRow;
    std::vector<FkChildRow> rows;
    for (std::size_t i = 0; i < 4; ++i) {
        FkChildRow row;
        row.parentCode = (i == 1) ? 9999U : 7U; // 第 1 行引用不存在的父行
        rows.push_back(row);
    }
    // 批次前的同事务成功写入也应被回滚（整笔回滚）
    sqlite_insert_many_test::FkParentRow parent;
    parent.code = 7U;
    TransactionBinding binding;
    ASSERT_TRUE(engine->Begin(binding).ok());
    const auto seeded = engine->Insert(
        RecordSchema<sqlite_insert_many_test::FkParentRow>::Meta(), TableScope::Static(), &parent, &binding);
    ASSERT_TRUE(seeded.ok()) << seeded.diagnostic;

    std::vector<const void*> pointers = PointerView(rows);
    std::array<uint64_t, 4> ids{};
    const auto failed = engine->InsertMany(
        RecordSchema<FkChildRow>::Meta(), TableScope::Static(), pointers.data(), 4, ids.data(), &binding);
    EXPECT_EQ(failed.code, DbCode::CONFLICT);
    EXPECT_NE(failed.diagnostic.find(" at row 1"), std::string::npos) << failed.diagnostic;
    // 引擎只返回错误；物理回滚由 Transaction 统一执行（本测试直接调用引擎，手动回滚）
    ASSERT_TRUE(engine->Rollback(binding).ok());
    std::vector<FkChildRow> children;
    auto childSink = ResultSink::For(children);
    ASSERT_TRUE(
        engine->Select(RecordSchema<FkChildRow>::Meta(), TableScope::Static(), All<FkChildRow>().data, {}, childSink)
            .ok());
    EXPECT_TRUE(children.empty());
    std::vector<sqlite_insert_many_test::FkParentRow> parents;
    auto parentSink = ResultSink::For(parents);
    ASSERT_TRUE(engine
                    ->Select(
                        RecordSchema<sqlite_insert_many_test::FkParentRow>::Meta(), TableScope::Static(),
                        All<sqlite_insert_many_test::FkParentRow>().data, {}, parentSink)
                    .ok());
    EXPECT_TRUE(parents.empty());
}

// ---- 与基类默认逐行实现的等价对照 ----

TEST_F(SqliteInsertManyTest, OverrideMatchesBaseDefaultImplementationOnSuccess)
{
    constexpr std::size_t kRows = 16;
    std::vector<BatchRow> rows;
    MakeRows(rows, kRows);
    auto pointers = PointersOf(rows);

    std::array<uint64_t, kRows> overrideIds{};
    TransactionBinding overrideBinding;
    ASSERT_TRUE(engine->Begin(overrideBinding).ok());
    const auto overrideResult = engine->InsertMany(
        Meta(), TableScope::Static(), pointers.data(), kRows, overrideIds.data(), &overrideBinding);
    ASSERT_TRUE(overrideResult.ok()) << overrideResult.diagnostic;
    ASSERT_TRUE(engine->Commit(overrideBinding).ok());
    const auto overrideStored = SelectAllRows();

    // 第二个全新引擎/库，同输入走基类默认实现（显式限定调用绕过 override）
    ResetEngine();
    std::array<uint64_t, kRows> baseIds{};
    TransactionBinding baseBinding;
    ASSERT_TRUE(engine->Begin(baseBinding).ok());
    const auto baseResult = engine->IStorageEngine::InsertMany(
        Meta(), TableScope::Static(), pointers.data(), kRows, baseIds.data(), &baseBinding);
    ASSERT_TRUE(baseResult.ok()) << baseResult.diagnostic;
    ASSERT_TRUE(engine->Commit(baseBinding).ok());
    const auto baseStored = SelectAllRows();

    EXPECT_EQ(overrideResult.affectedRows, baseResult.affectedRows);
    for (std::size_t i = 0; i < kRows; ++i) {
        EXPECT_EQ(overrideIds[i], baseIds[i]) << "row " << i;
    }
    ASSERT_EQ(overrideStored.value->size(), baseStored.value->size());
    for (std::size_t i = 0; i < kRows; ++i) {
        EXPECT_EQ((*overrideStored.value)[i].name, (*baseStored.value)[i].name);
        EXPECT_EQ((*overrideStored.value)[i].payload, (*baseStored.value)[i].payload);
        EXPECT_EQ((*overrideStored.value)[i].wide, (*baseStored.value)[i].wide);
    }
}

TEST_F(SqliteInsertManyTest, OverrideMatchesBaseDefaultImplementationOnFailure)
{
    using UniqueNameRow = sqlite_insert_many_test::UniqueNameRow;
    auto makeBatch = [] {
        std::vector<UniqueNameRow> rows;
        for (std::size_t i = 0; i < 5; ++i) {
            UniqueNameRow row;
            row.name = (i == 2) ? "dup" : "name-" + std::to_string(i);
            rows.push_back(row);
        }
        return rows;
    };
    auto seedDuplicate = [&](TransactionBinding& binding) {
        UniqueNameRow first;
        first.name = "dup";
        const auto seeded = engine->Insert(RecordSchema<UniqueNameRow>::Meta(), TableScope::Static(), &first, &binding);
        ASSERT_TRUE(seeded.ok()) << seeded.diagnostic;
    };

    std::vector<UniqueNameRow> rows = makeBatch();
    std::vector<const void*> pointers = PointerView(rows);
    std::array<uint64_t, 5> ids{};

    TransactionBinding overrideBinding;
    ASSERT_TRUE(engine->Begin(overrideBinding).ok());
    seedDuplicate(overrideBinding);
    const auto overrideFailed = engine->InsertMany(
        RecordSchema<UniqueNameRow>::Meta(), TableScope::Static(), pointers.data(), 5, ids.data(), &overrideBinding);
    ASSERT_TRUE(engine->Rollback(overrideBinding).ok());

    ResetEngine();
    TransactionBinding baseBinding;
    ASSERT_TRUE(engine->Begin(baseBinding).ok());
    seedDuplicate(baseBinding);
    const auto baseFailed = engine->IStorageEngine::InsertMany(
        RecordSchema<UniqueNameRow>::Meta(), TableScope::Static(), pointers.data(), 5, ids.data(), &baseBinding);
    ASSERT_TRUE(engine->Rollback(baseBinding).ok());

    EXPECT_EQ(overrideFailed.code, baseFailed.code);
    EXPECT_EQ(overrideFailed.affectedRows, baseFailed.affectedRows);
    EXPECT_EQ(overrideFailed.diagnostic, baseFailed.diagnostic);
}

// ---- 机制证据：--wrap 观察窗的 prepare 计数 ----

TEST_F(SqliteInsertManyTest, PrepareOncePerBatchOverrideAndOncePerRowBaseDefault)
{
    constexpr std::size_t kRows = 64;
    std::vector<BatchRow> rows;
    MakeRows(rows, kRows);
    auto pointers = PointersOf(rows);

    // OPT：override 一批一次显式 prepare（仅计目标表 INSERT）
    {
        PrepareCounter counter{R"(INSERT INTO "BatchRow")"};
        std::vector<uint64_t> ids(kRows, 0);
        TransactionBinding binding;
        ASSERT_TRUE(engine->Begin(binding).ok());
        const auto inserted
            = engine->InsertMany(Meta(), TableScope::Static(), pointers.data(), kRows, ids.data(), &binding);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
        ASSERT_TRUE(engine->Commit(binding).ok());
        EXPECT_EQ(counter.value(), 1) << "override 应一批只显式 prepare 一次";
        for (std::size_t i = 0; i < kRows; ++i) {
            EXPECT_EQ(ids[i], i + 1);
        }
    }
    // BASE：显式限定基类默认实现——每行一次 prepare
    ResetEngine();
    {
        PrepareCounter counter{R"(INSERT INTO "BatchRow")"};
        std::vector<uint64_t> ids(kRows, 0);
        TransactionBinding binding;
        ASSERT_TRUE(engine->Begin(binding).ok());
        const auto inserted = engine->IStorageEngine::InsertMany(
            Meta(), TableScope::Static(), pointers.data(), kRows, ids.data(), &binding);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
        ASSERT_TRUE(engine->Commit(binding).ok());
        EXPECT_EQ(counter.value(), static_cast<int>(kRows)) << "默认实现应逐行 prepare";
        for (std::size_t i = 0; i < kRows; ++i) {
            EXPECT_EQ(ids[i], i + 1);
        }
    }
}

TEST_F(SqliteInsertManyTest, FailedBatchLeavesNoPrepareForLaterRows)
{
    std::vector<BatchRow> rows;
    MakeRows(rows, 6);
    auto pointers = PointersOf(rows);
    pointers[5] = nullptr;
    std::array<uint64_t, 6> ids{};
    {
        PrepareCounter counter{R"(INSERT INTO "BatchRow")"};
        TransactionBinding binding;
        ASSERT_TRUE(engine->Begin(binding).ok());
        const auto failed = engine->InsertMany(Meta(), TableScope::Static(), pointers.data(), 6, ids.data(), &binding);
        EXPECT_EQ(failed.code, DbCode::INVALID_ARGUMENT);
        // 第 0-4 行已 prepare/执行（同一语句），第 5 行空指针不再触发新 prepare
        EXPECT_EQ(counter.value(), 1);
        ASSERT_TRUE(engine->Rollback(binding).ok());
    }
}
} // namespace
