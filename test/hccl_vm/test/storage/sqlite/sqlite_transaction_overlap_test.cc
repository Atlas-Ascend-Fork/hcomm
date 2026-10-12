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
 * See the License for details on how you should not use this file except in compliance with the License.
 */

// 文件职责：验证 SQLiteEngine 单连接上"活动事务与其他线程即时 CRUD 的重叠"
// 合同：
//   - 同线程在活动事务内发起即时操作（Insert/Select/Count/Begin/Clear）必须
//     立即 INVALID_STATE（不得等待，防自锁）；
//   - 其他线程的即时 Insert/Select/Count/Update/Delete 与第二笔 Begin 在活动
//     事务期间有界等待（条件变量，不轮询），事务 Commit/Rollback/关闭后按序
//     完成；
//   - 等待超时返回 RESOURCE_EXHAUSTED，diagnostic 携带 backend/等待时长/
//     transaction wait timed out；
//   - 引擎关闭唤醒全部等待者，有限返回（不永久挂起）。
// RED→GREEN：旧实现（重叠即 INVALID_STATE）在本文件的跨线程用例上失败；
// 同线程快速失败用例在两种实现上都通过（回归防线）。
// 等待上限经测试 peer 缩短（产品默认 30s）；旧引擎无该成员时编译期退化为
// 无操作（超时用例在旧实现下本就立即失败，不依赖缩短）。
#include <gtest/gtest.h>

#include <sqlite3.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "storage_backends/sqlite/engine/sqlite_engine.h"
#include "storage/storage_catalog.h"

namespace sqlite_transaction_overlap_test {
struct OverlapRow {
    uint64_t id{0};
    uint64_t value{0};
    std::string tag;
};
} // namespace sqlite_transaction_overlap_test

using namespace HcclSim::Storage::Ddl;
#define Field HCCL_VM_DDL_FIELD
#define Version HCCL_VM_DDL_VERSION
#define PrimaryKey HCCL_VM_DDL_PRIMARY_KEY
#define Scope Ddl::ScopeKind

#define SQLITE_TRANSACTION_OVERLAP_DATABASE(Table)   \
    Table(                                           \
        sqlite_transaction_overlap_test::OverlapRow, \
        (Version(1), Scope::Static, Field(id, AutoIncrement), Field(value), Field(tag), PrimaryKey(id)))

#define HCCL_VM_DDL_CURRENT_DATABASE "SQLITE_TX_OVERLAP_TEST_DB"
HCCL_VM_DEFINE_DATABASE(kSqliteTxOverlapTestDatabase, SQLITE_TX_OVERLAP_TEST_DB, SQLITE_TRANSACTION_OVERLAP_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE

#undef Field
#undef Version
#undef PrimaryKey
#undef Scope

namespace {
using namespace HcclSim::Storage;
using sqlite_transaction_overlap_test::OverlapRow;

constexpr SqliteConnectionProfile kTestProfile{10000, false, 200};

std::string TempDatabasePath(const std::string& tag)
{
    static std::atomic<uint64_t> counter{0};
    const std::string path
        = testing::TempDir() + "sqlite_tx_overlap_" + tag + "_" + std::to_string(counter.fetch_add(1)) + ".db";
    (void)std::remove(path.c_str());
    return path;
}

/// 异步引擎调用：entered 先于引擎调用建立 happens-before，主线程据此确认对端
/// 已进入调用（不靠裸 sleep 猜调度）；返回 (enteredFuture, resultFuture)。
/// 内层 lambda 必须 mutable：call 是按值捕获的可变 lambda，const operator()
/// 内调用会丢失限定符。
template <typename Call>
std::pair<std::future<void>, std::future<std::invoke_result_t<Call>>> LaunchAsync(Call call)
{
    auto entered = std::make_shared<std::promise<void>>();
    auto enteredFuture = entered->get_future();
    auto result = std::async(std::launch::async, [entered, call]() mutable -> std::invoke_result_t<Call> {
        entered->set_value();
        return call();
    });
    return {std::move(enteredFuture), std::move(result)};
}
} // namespace

namespace HcclSim::Storage {
/// 测试观察点（产品代码不得引用；本名仅落在本测试二进制内）。编译期探测
/// 定义在本 friend 结构内——对引擎私有测试入口的探测表达式在 friend 语境
/// 获得访问权；旧实现无该入口时退化为 false_type（peer 无操作，超时用例
/// 即 RED 证据，不会真实等待 30s）。
struct SQLiteEngineTestPeer {
    template <typename Engine, typename = void>
    struct HasTransactionWaitLimit : std::false_type {};

    template <typename Engine>
    struct HasTransactionWaitLimit<
        Engine, std::void_t<decltype(std::declval<Engine&>().SetTransactionWaitLimitForTest(
                    std::declval<std::chrono::milliseconds>()))>> : std::true_type {};

    /// 模板函数：if constexpr 的假分支在旧引擎（无该入口）下被丢弃，
    /// 不实例化调用
    template <typename Engine>
    static void SetTransactionWaitLimit(Engine& engine, std::chrono::milliseconds limit)
    {
        if constexpr (HasTransactionWaitLimit<Engine>::value) {
            engine.SetTransactionWaitLimitForTest(limit);
        }
    }
};
} // namespace HcclSim::Storage

namespace {
/// 夹具：每个用例独立临时库与引擎
class SqliteTransactionOverlapTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        databasePath = TempDatabasePath("engine");
        engine = std::make_unique<SQLiteEngine>(databasePath, &Ddl::kSqliteTxOverlapTestDatabase, kTestProfile);
        EngineOpenRequest open;
        open.mode = StorageOpenMode::READ_WRITE_CREATE;
        const auto opened = engine->Open(open);
        ASSERT_TRUE(opened.ok()) << opened.diagnostic;
        const auto prepared = engine->Prepare();
        ASSERT_TRUE(prepared.ok()) << prepared.diagnostic;
    }

    void TearDown() override
    {
        if (engine != nullptr) {
            (void)engine->Close();
        }
    }

    const Ddl::CatalogTableMeta& Meta() const { return RecordSchema<OverlapRow>::Meta(); }

    OverlapRow MakeRow(uint64_t value, const std::string& tag)
    {
        OverlapRow row;
        row.value = value;
        row.tag = tag;
        return row;
    }

    uint64_t CountAll()
    {
        const auto counted = engine->Count(Meta(), TableScope::Static(), ConditionData{});
        EXPECT_TRUE(counted.ok()) << counted.diagnostic;
        return counted.ok() ? *counted.value : UINT64_MAX;
    }

    std::string databasePath;
    std::unique_ptr<SQLiteEngine> engine;
};
} // namespace

// 同线程：活动事务内的即时操作立即 INVALID_STATE（不得等待 30s 防自锁）；
// 重复 Begin 同样立即失败。两种实现都必须保持（回归防线）。
TEST_F(SqliteTransactionOverlapTest, SameThreadImmediateOpsStillRejectedFast)
{
    TransactionBinding transaction;
    ASSERT_TRUE(engine->Begin(transaction).ok());

    const auto begunAt = std::chrono::steady_clock::now();
    OverlapRow plain = MakeRow(1, "plain");
    const auto inserted = engine->Insert(Meta(), TableScope::Static(), &plain, nullptr);
    EXPECT_EQ(inserted.code, DbCode::INVALID_STATE);
    EXPECT_NE(inserted.diagnostic.find("a transaction is building"), std::string::npos) << inserted.diagnostic;

    const auto counted = engine->Count(Meta(), TableScope::Static(), ConditionData{});
    EXPECT_EQ(counted.code, DbCode::INVALID_STATE);

    const auto cleared = engine->Clear(Meta(), TableScope::Static(), ClearMode::KEEP_SEQUENCE);
    EXPECT_EQ(cleared.code, DbCode::INVALID_STATE);

    TransactionBinding second;
    const auto begun = engine->Begin(second);
    EXPECT_EQ(begun.code, DbCode::INVALID_STATE);

    std::vector<OverlapRow> rows;
    auto sink = ResultSink::For(rows);
    const auto selected = engine->Select(Meta(), TableScope::Static(), ConditionData{}, QueryOptions{}, sink);
    EXPECT_EQ(selected.code, DbCode::INVALID_STATE);

    const auto elapsedMs
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begunAt).count();
    EXPECT_LT(elapsedMs, 2000) << "同线程拒绝必须立即返回，不得进入等待";

    ASSERT_TRUE(engine->Rollback(transaction).ok());
}

// 跨线程读：线程 A 开事务并保持；线程 B 的 Count 等待（未完成）；
// A Commit 后 B 成功完成
TEST_F(SqliteTransactionOverlapTest, CrossThreadSelectWaitsForCommit)
{
    TransactionBinding transaction;
    ASSERT_TRUE(engine->Begin(transaction).ok());
    OverlapRow inTx = MakeRow(11, "in-tx");
    const auto inserted = engine->Insert(Meta(), TableScope::Static(), &inTx, &transaction);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;

    auto launched = LaunchAsync([this]() {
        return engine->Count(Meta(), TableScope::Static(), ConditionData{});
    });
    ASSERT_EQ(launched.first.wait_for(std::chrono::seconds(5)), std::future_status::ready)
        << "等待线程应已进入引擎调用";
    // B 进入调用后仍未完成 ⇒ 正在等待事务槽位（旧实现立即返回 INVALID_STATE）
    ASSERT_EQ(launched.second.wait_for(std::chrono::milliseconds(300)), std::future_status::timeout)
        << "活动事务期间跨线程 Count 应等待而不是立即失败";

    ASSERT_TRUE(engine->Commit(transaction).ok());
    const auto counted = launched.second.get();
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    EXPECT_EQ(*counted.value, 1U);
}

// 跨线程写：A 回滚后 B 的即时 Insert 成功；最终数据只含 B 的行（顺序语义）
TEST_F(SqliteTransactionOverlapTest, CrossThreadInsertWaitsForRollback)
{
    TransactionBinding transaction;
    ASSERT_TRUE(engine->Begin(transaction).ok());
    OverlapRow inTx = MakeRow(22, "in-tx");
    ASSERT_TRUE(engine->Insert(Meta(), TableScope::Static(), &inTx, &transaction).ok());

    OverlapRow after = MakeRow(33, "after");
    auto launched = LaunchAsync([this, after]() mutable {
        return engine->Insert(Meta(), TableScope::Static(), &after, nullptr);
    });
    ASSERT_EQ(launched.first.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_EQ(launched.second.wait_for(std::chrono::milliseconds(300)), std::future_status::timeout)
        << "活动事务期间跨线程即时 Insert 应等待而不是立即失败";

    ASSERT_TRUE(engine->Rollback(transaction).ok());
    const auto inserted = launched.second.get();
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    ASSERT_GT(*inserted.value, 0U);

    EXPECT_EQ(CountAll(), 1U);
    std::vector<OverlapRow> rows;
    auto sink = ResultSink::For(rows);
    ASSERT_TRUE(engine->Select(Meta(), TableScope::Static(), ConditionData{}, QueryOptions{}, sink).ok());
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].tag, "after");
}

// 跨线程第二事务：B 的 Begin 等待 A 结束后成功开启并提交
TEST_F(SqliteTransactionOverlapTest, CrossThreadSecondBeginSerializes)
{
    TransactionBinding first;
    ASSERT_TRUE(engine->Begin(first).ok());
    OverlapRow inFirst = MakeRow(7, "first");
    ASSERT_TRUE(engine->Insert(Meta(), TableScope::Static(), &inFirst, &first).ok());

    auto launched = LaunchAsync([this]() {
        TransactionBinding second;
        const auto begun = engine->Begin(second);
        if (!begun.ok()) {
            return begun;
        }
        OverlapRow inSecond = MakeRow(8, "second");
        const auto inserted = engine->Insert(Meta(), TableScope::Static(), &inSecond, &second);
        if (!inserted.ok()) {
            (void)engine->Rollback(second);
            return DbResult<void>{inserted.code, 0, inserted.diagnostic};
        }
        const auto committed = engine->Commit(second);
        if (!committed.ok()) {
            return DbResult<void>{committed.code, 0, committed.diagnostic};
        }
        return DbResult<void>{DbCode::OK, 0, {}};
    });
    ASSERT_EQ(launched.first.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_EQ(launched.second.wait_for(std::chrono::milliseconds(300)), std::future_status::timeout)
        << "单连接上的第二笔事务应串行等待而不是立即失败";

    ASSERT_TRUE(engine->Commit(first).ok());
    const auto secondOutcome = launched.second.get();
    ASSERT_TRUE(secondOutcome.ok()) << secondOutcome.diagnostic;

    EXPECT_EQ(CountAll(), 2U);
}

// 等待超时：peer 缩短上限后，跨线程即时操作返回 RESOURCE_EXHAUSTED，
// diagnostic 携带 backend/等待时长/transaction wait timed out
TEST_F(SqliteTransactionOverlapTest, WaitTimeoutReturnsResourceExhausted)
{
    constexpr std::chrono::milliseconds kShortLimit{60};
    SQLiteEngineTestPeer::SetTransactionWaitLimit(*engine, kShortLimit);

    if (!SQLiteEngineTestPeer::HasTransactionWaitLimit<SQLiteEngine>::value) {
        // 旧引擎：无测试入口（peer 无操作），跨线程即时操作立即失败——
        // 本用例即为 RED 证据
        TransactionBinding legacy;
        ASSERT_TRUE(engine->Begin(legacy).ok());
        auto launched = LaunchAsync([this]() {
            return engine->Count(Meta(), TableScope::Static(), ConditionData{});
        });
        ASSERT_EQ(launched.first.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        const auto counted = launched.second.get();
        ASSERT_TRUE(engine->Rollback(legacy).ok());
        ADD_FAILURE() << "旧实现：活动事务期间跨线程即时 Count 未等待即返回，code="
                      << static_cast<uint32_t>(counted.code) << "（应为 RESOURCE_EXHAUSTED）";
        return;
    }

    TransactionBinding transaction;
    ASSERT_TRUE(engine->Begin(transaction).ok());

    const auto begunAt = std::chrono::steady_clock::now();
    auto launched = LaunchAsync([this]() {
        return engine->Count(Meta(), TableScope::Static(), ConditionData{});
    });
    ASSERT_EQ(launched.first.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    const auto counted = launched.second.get();
    const auto elapsedMs
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begunAt).count();

    ASSERT_EQ(counted.code, DbCode::RESOURCE_EXHAUSTED) << counted.diagnostic;
    EXPECT_NE(counted.diagnostic.find("transaction wait timed out"), std::string::npos) << counted.diagnostic;
    EXPECT_NE(counted.diagnostic.find("sqlite"), std::string::npos) << counted.diagnostic;
    EXPECT_GE(elapsedMs, 50) << "超时应发生在缩短后的等待上限附近";
    EXPECT_LE(elapsedMs, 5000) << "等待必须有界";

    ASSERT_TRUE(engine->Rollback(transaction).ok());
    EXPECT_EQ(CountAll(), 0U);
}

// 关闭唤醒：等待中的线程在引擎关闭后有限返回（不永久挂起）
TEST_F(SqliteTransactionOverlapTest, CloseWakesWaiters)
{
    TransactionBinding transaction;
    ASSERT_TRUE(engine->Begin(transaction).ok());

    auto launched = LaunchAsync([this]() {
        return engine->Count(Meta(), TableScope::Static(), ConditionData{});
    });
    ASSERT_EQ(launched.first.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_EQ(launched.second.wait_for(std::chrono::milliseconds(300)), std::future_status::timeout)
        << "关闭前等待线程应仍处于等待状态";

    ASSERT_TRUE(engine->Close().ok());
    const auto counted = launched.second.get();
    EXPECT_FALSE(counted.ok());
    EXPECT_EQ(counted.code, DbCode::INVALID_STATE) << counted.diagnostic;
}
