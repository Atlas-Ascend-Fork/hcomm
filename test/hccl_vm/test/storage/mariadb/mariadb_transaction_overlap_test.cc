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

// 文件职责：验证 MariaDbEngine 单连接上"活动事务与其他线程即时 CRUD 的重叠"
// 合同（与 SQLite 引擎同构的语义对齐）：
//   - 同线程在活动事务内发起即时操作（Insert/Select/Count/Begin）必须立即
//     INVALID_STATE（不得等待，防自锁）；
//   - 其他线程的即时 Insert/Select/Count 与第二笔 Begin 在活动事务期间有界
//     等待，事务结束后按序完成；
//   - 等待超时返回 RESOURCE_EXHAUSTED，diagnostic 携带 backend/等待时长/
//     transaction wait timed out；
//   - 引擎关闭/连接放弃唤醒全部等待者，有限返回。
// RED→GREEN：旧实现（重叠即 INVALID_STATE，13 环境跑批高频触发存储 API 失败）
// 在本文件的跨线程用例上失败。endpoint 由 HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT
// 提供（真实 mysqld）；等待上限经测试 peer 缩短（产品默认 30s）。
#include <gtest/gtest.h>

#include <mysql.h>

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "storage_backends/mariadb/engine/mariadb_engine.h"
#include "storage/storage_catalog.h"

namespace mariadb_tx_overlap_test {
struct OverlapRow {
    uint64_t id{0};
    uint64_t value{0};
    std::string tag;
};
} // namespace mariadb_tx_overlap_test

using namespace HcclSim::Storage::Ddl;
#define Field HCCL_VM_DDL_FIELD
#define Version HCCL_VM_DDL_VERSION
#define PrimaryKey HCCL_VM_DDL_PRIMARY_KEY
#define Scope Ddl::ScopeKind

#define MARIADB_TX_OVERLAP_DATABASE(Table)   \
    Table(                                   \
        mariadb_tx_overlap_test::OverlapRow, \
        (Version(1), Scope::Static, Field(id, AutoIncrement), Field(value), Field(tag), PrimaryKey(id)))

#define HCCL_VM_DDL_CURRENT_DATABASE "MARIADB_TX_OVERLAP_TEST_DB"
HCCL_VM_DEFINE_DATABASE(kMariadbTxOverlapTestDatabase, MARIADB_TX_OVERLAP_TEST_DB, MARIADB_TX_OVERLAP_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE

#undef Field
#undef Version
#undef PrimaryKey
#undef Scope

namespace {
using namespace HcclSim::Storage;
using mariadb_tx_overlap_test::OverlapRow;

/// 每用例唯一 namespace 计数：物理 database 在服务器上持久存在，同一进程
/// 内共享 namespace 会让先前用例的已提交行污染后续用例的计数断言——
/// 唯一化保证用例间零数据耦合（镜像既有 mariadb_engine_test 的
/// EngineFixture 每用例唯一 namespace 约定）
std::atomic<uint64_t> gNamespaceCounter{0};

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

/// 异步引擎调用：entered 先于引擎调用建立 happens-before；返回
/// (enteredFuture, resultFuture)。内层 lambda 必须 mutable：call 是按值捕获
/// 的可变 lambda，const operator() 内调用会丢失限定符。
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
struct MariadbEngineTestPeer {
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
/// 夹具：唯一 namespace 的全新 database + 真实 endpoint 引擎
class MariadbTransactionOverlapTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        if (!EndpointConfigured()) {
            GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置：真实 mysqld 需要外部 endpoint";
        }
        dataNamespace = "mtx_overlap_p" + std::to_string(static_cast<long>(::getpid())) + "_"
                        + std::to_string(gNamespaceCounter.fetch_add(1));
        engine = std::make_unique<MariaDbEngine>(&Ddl::kMariadbTxOverlapTestDatabase, TestProfile());
        EngineOpenRequest request;
        request.mode = StorageOpenMode::READ_WRITE_CREATE;
        request.dataNamespace = dataNamespace;
        const auto opened = engine->Open(request);
        ASSERT_TRUE(opened.ok()) << opened.diagnostic;
        const auto prepared = engine->Prepare();
        ASSERT_TRUE(prepared.ok()) << prepared.diagnostic;
        ready = true;
    }

    void TearDown() override
    {
        if (engine != nullptr && ready) {
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

    std::unique_ptr<MariaDbEngine> engine;
    std::string dataNamespace;
    bool ready{false};
};
} // namespace

// 同线程：活动事务内的即时操作立即 INVALID_STATE（不得等待防自锁）；重复
// Begin 同样立即失败。两种实现都必须保持（回归防线）
TEST_F(MariadbTransactionOverlapTest, SameThreadImmediateOpsStillRejectedFast)
{
    ASSERT_TRUE(ready);
    TransactionBinding transaction;
    ASSERT_TRUE(engine->Begin(transaction).ok());

    const auto begunAt = std::chrono::steady_clock::now();
    OverlapRow plain = MakeRow(1, "plain");
    const auto inserted = engine->Insert(Meta(), TableScope::Static(), &plain, nullptr);
    EXPECT_EQ(inserted.code, DbCode::INVALID_STATE);
    EXPECT_NE(inserted.diagnostic.find("a transaction is building"), std::string::npos) << inserted.diagnostic;

    const auto counted = engine->Count(Meta(), TableScope::Static(), ConditionData{});
    EXPECT_EQ(counted.code, DbCode::INVALID_STATE);

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

// 跨线程读：A 开事务并保持；B 的 Count 等待；A Commit 后 B 成功完成
TEST_F(MariadbTransactionOverlapTest, CrossThreadSelectWaitsForCommit)
{
    ASSERT_TRUE(ready);
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
    ASSERT_EQ(launched.second.wait_for(std::chrono::milliseconds(300)), std::future_status::timeout)
        << "活动事务期间跨线程 Count 应等待而不是立即失败";

    ASSERT_TRUE(engine->Commit(transaction).ok());
    const auto counted = launched.second.get();
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    EXPECT_EQ(*counted.value, 1U);
}

// 跨线程写：A Rollback 后 B 的即时 Insert 成功；最终数据只含 B 的行
TEST_F(MariadbTransactionOverlapTest, CrossThreadInsertWaitsForRollback)
{
    ASSERT_TRUE(ready);
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
TEST_F(MariadbTransactionOverlapTest, CrossThreadSecondBeginSerializes)
{
    ASSERT_TRUE(ready);
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
TEST_F(MariadbTransactionOverlapTest, WaitTimeoutReturnsResourceExhausted)
{
    ASSERT_TRUE(ready);
    constexpr std::chrono::milliseconds kShortLimit{60};
    MariadbEngineTestPeer::SetTransactionWaitLimit(*engine, kShortLimit);

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

    if (!MariadbEngineTestPeer::HasTransactionWaitLimit<MariaDbEngine>::value) {
        // 旧引擎：无测试入口（peer 无操作），跨线程即时操作立即失败——本用例
        // 即为 RED 证据
        ASSERT_TRUE(engine->Rollback(transaction).ok());
        ADD_FAILURE() << "旧实现：活动事务期间跨线程即时 Count 未等待即返回，code="
                      << static_cast<uint32_t>(counted.code) << "（应为 RESOURCE_EXHAUSTED）";
        return;
    }

    ASSERT_EQ(counted.code, DbCode::RESOURCE_EXHAUSTED) << counted.diagnostic;
    EXPECT_NE(counted.diagnostic.find("transaction wait timed out"), std::string::npos) << counted.diagnostic;
    EXPECT_NE(counted.diagnostic.find("mariadb"), std::string::npos) << counted.diagnostic;
    EXPECT_GE(elapsedMs, 50) << "超时应发生在缩短后的等待上限附近";
    EXPECT_LE(elapsedMs, 5000) << "等待必须有界";

    ASSERT_TRUE(engine->Rollback(transaction).ok());
    EXPECT_EQ(CountAll(), 0U);
}

// 关闭唤醒：等待中的线程在引擎关闭后有限返回（不永久挂起）
TEST_F(MariadbTransactionOverlapTest, CloseWakesWaiters)
{
    ASSERT_TRUE(ready);
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
