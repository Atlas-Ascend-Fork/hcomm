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
 * @file storage_contract_test.cc
 * @brief 存储层统一共享契约、四维路由、会话生命周期与业务表达式的合同单元测试。
 *
 * 【测试覆盖契约范围】
 * 1. DbResult 基础数据结构与安全箭头访问 (Code/Value/AffectedRows/Diagnostic)
 *    及 value_or 显式默认值访问（成功载荷 / 失败与无载荷默认值）
 * 2. IStorageEngine / IRelationalEngine 虚函数分发与备份调用链
 * 3. Table<T> / Transaction 的 Move-Only 强类型与单向状态机语义
 * 4. 类型安全条件表达式（成员指针映射、比较/IN/前缀、嵌套 And/Or 条件树）
 * 5. Select / Count / Update(Set/Increment) / Delete 的后端下推语义
 * 6. AcquireTable 单步类型化表获取与通用事务入口（惰性绑定物理事务域）
 * 7. 一次性原子操作入口：每次一次取表 + 一次操作的调度合同、取表失败与
 *    操作失败的 DbResult 原样传播、默认 scope 与显式动态 scope 解析
 * 8. TableBindingCatalog 四维坐标解析与非拥有型路由隔离
 * 9. ProcessStorageContext 进程级租约与多线程并发排空
 * 10. QuiesceToken 纪元控制与 BACKUP_ONLY 只读备份会话拦截
 * 11. SessionCoordinator 具名 ResetPlan 顺序编排与闸门校验
 * 12. Fork 合同（收窄为立即 exec）：产品 fork child 在 exec 前不访问 Storage API；
 *     ResetAfterForkInChild 只切换进程本地上下文状态，不触碰任何继承
 *     Session/Coordinator/Engine/SQLite 对象或 mutex（fork+reset 仅作为取得
 *     NEVER_INSTALLED 冷状态的测试载体，不是产品 fork 合同）
 */
#include <gtest/gtest.h>

#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "storage/database_result.h"
#include "storage/logical_operation.h"
#include "storage/query_expression.h"
#include "storage/query_spec.h"
#include "storage/record_contract.h"
#include "storage/storage_binding.h"
#include "storage/storage_catalog.h"
#include "storage/session_coordinator.h"
#include "storage/storage_session.h"
#include "storage/table.h"
#include "storage/table_access.h"
#include "storage/transaction.h"
#include "storage/engine/i_relational_engine.h"
#include "storage/engine/i_storage_engine.h"
#include "storage/internal/process_storage_context.h"
#include "storage/internal/storage_runtime_impl.h"
#include "storage/storage_runtime.h"
#include "session_test_peer.h"
#include "test_fake_backend.h"

struct ContractRecord {
    uint64_t id{0};
    int64_t state{0};
    char label[16]{};
    std::string tag;
};

/// 第二物理路由上的最小记录类型（验证事务跨物理域拒绝）
struct OpRecord {
    uint64_t id{0};
};

namespace HcclSim::Storage {
// 测试记录经统一 StorageCatalog 机制登记（一个 Table entry 派生 schema/codec/主键通道，
// 测试文件不再手写字段清单与编解码特化）。
} // namespace HcclSim::Storage

// ---- 合同测试记录的 DDL 区（全局作用域；短别名区段结束立即 #undef） ----
using namespace HcclSim::Storage::Ddl;
#define Field HCCL_VM_DDL_FIELD
#define Version HCCL_VM_DDL_VERSION
#define PrimaryKey HCCL_VM_DDL_PRIMARY_KEY
#define Index HCCL_VM_DDL_INDEX
#define Scope Ddl::ScopeKind

#define CONTRACT_TEST_DATABASE(Table)                                                                           \
    Table(                                                                                                      \
        ContractRecord, (Version(1), Scope::Static, Field(id, AutoIncrement), Field(state), Field(label, Text), \
                         Field(tag), PrimaryKey(id)))

#define HCCL_VM_DDL_CURRENT_DATABASE "CONTRACT_TEST_DB"
HCCL_VM_DEFINE_DATABASE(kContractTestDatabase, CONTRACT_TEST_DB, CONTRACT_TEST_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE

// 第二物理路由的静态测试库（跨物理域拒绝用）
#define CONTRACT_TEST_OP_DATABASE(Table) \
    Table(OpRecord, (Version(1), Scope::Static, Field(id, AutoIncrement), PrimaryKey(id)))
#define HCCL_VM_DDL_CURRENT_DATABASE "CONTRACT_TEST_OP_DB"
HCCL_VM_DEFINE_DATABASE(kContractTestOpDatabase, CONTRACT_TEST_OP_DB, CONTRACT_TEST_OP_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE

// 动态 PID scope 测试库
struct DynamicRecord {
    uint64_t id{0};
};
#define CONTRACT_TEST_DYNAMIC_DATABASE(Table) \
    Table(DynamicRecord, (Version(1), Scope::DynamicPid, Field(id, AutoIncrement), PrimaryKey(id)))
#define HCCL_VM_DDL_CURRENT_DATABASE "CONTRACT_TEST_DYNAMIC_DB"
HCCL_VM_DEFINE_DATABASE(kContractTestDynamicDatabase, CONTRACT_TEST_DYNAMIC_DB, CONTRACT_TEST_DYNAMIC_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE
#undef Field
#undef Version
#undef PrimaryKey
#undef Index
#undef Scope

namespace {
using namespace HcclSim::Storage;

using OpenSignature = DbResult<void> (IStorageEngine::*)(const EngineOpenRequest&);
using PrepareSignature = DbResult<void> (IStorageEngine::*)();
using EngineInsertSignature = DbResult<uint64_t> (IStorageEngine::*)(
    const Ddl::CatalogTableMeta&, const TableScope&, const void*, TransactionBinding*);
using EngineSelectSignature = DbResult<void> (IStorageEngine::*)(
    const Ddl::CatalogTableMeta&, const TableScope&, const ConditionData&, const QueryOptions&, ResultSink&);
using EngineCountSignature
    = DbResult<uint64_t> (IStorageEngine::*)(const Ddl::CatalogTableMeta&, const TableScope&, const ConditionData&);
using EngineUpdateSignature = DbResult<uint64_t> (IStorageEngine::*)(
    const Ddl::CatalogTableMeta&, const TableScope&, const ConditionData&, const UpdateData&, TransactionBinding*);
using EngineDeleteSignature = DbResult<uint64_t> (IStorageEngine::*)(
    const Ddl::CatalogTableMeta&, const TableScope&, const ConditionData&, TransactionBinding*);
using EngineClearSignature
    = DbResult<void> (IStorageEngine::*)(const Ddl::CatalogTableMeta&, const TableScope&, ClearMode);
using EngineListTablesSignature = DbResult<std::vector<std::string>> (IStorageEngine::*)();
using BeginSignature = DbResult<void> (IStorageEngine::*)(TransactionBinding&);
using CommitSignature = DbResult<AtomicResult> (IStorageEngine::*)(TransactionBinding&);
using RollbackSignature = DbResult<void> (IStorageEngine::*)(TransactionBinding&);
using CloseSignature = DbResult<void> (IStorageEngine::*)();
using RelationalBackupSignature = DbResult<DatabaseBackupArtifact> (IRelationalEngine::*)(const EngineBackupRequest&);

static_assert(std::has_virtual_destructor_v<IStorageEngine>);
static_assert(std::is_same_v<decltype(&IStorageEngine::Open), OpenSignature>);
static_assert(std::is_same_v<decltype(&IStorageEngine::Prepare), PrepareSignature>);
static_assert(std::is_same_v<decltype(&IStorageEngine::Insert), EngineInsertSignature>);
static_assert(std::is_same_v<decltype(&IStorageEngine::Select), EngineSelectSignature>);
static_assert(std::is_same_v<decltype(&IStorageEngine::Count), EngineCountSignature>);
static_assert(std::is_same_v<decltype(&IStorageEngine::Update), EngineUpdateSignature>);
static_assert(std::is_same_v<decltype(&IStorageEngine::Delete), EngineDeleteSignature>);
static_assert(std::is_same_v<decltype(&IStorageEngine::Clear), EngineClearSignature>);
static_assert(std::is_same_v<decltype(&IStorageEngine::ListTables), EngineListTablesSignature>);
static_assert(std::is_same_v<decltype(&IStorageEngine::Begin), BeginSignature>);
static_assert(std::is_same_v<decltype(&IStorageEngine::Commit), CommitSignature>);
static_assert(std::is_same_v<decltype(&IStorageEngine::Rollback), RollbackSignature>);
static_assert(std::is_same_v<decltype(&IStorageEngine::Close), CloseSignature>);
static_assert(std::is_base_of_v<IStorageEngine, IRelationalEngine>);
static_assert(std::is_same_v<decltype(&IRelationalEngine::BackupRelational), RelationalBackupSignature>);

static_assert(!std::is_copy_constructible_v<Table<int>>);
static_assert(!std::is_copy_assignable_v<Table<int>>);
static_assert(std::is_move_constructible_v<Table<int>>);
static_assert(!std::is_copy_constructible_v<Transaction>);
static_assert(!std::is_copy_assignable_v<Transaction>);
static_assert(std::is_move_constructible_v<Transaction>);
static_assert(!std::is_copy_constructible_v<StorageSession>);
static_assert(!std::is_copy_assignable_v<StorageSession>);
static_assert(std::is_move_constructible_v<StorageSession>);
static_assert(!std::is_move_assignable_v<StorageSession>);
static_assert(!std::is_copy_constructible_v<StorageSessionLease>);
static_assert(std::is_move_constructible_v<StorageSessionLease>);

using InsertSignature = DbResult<PrimaryKey<ContractRecord>> (Table<ContractRecord>::*)(ContractRecord&);
using InsertTxSignature
    = DbResult<PrimaryKey<ContractRecord>> (Table<ContractRecord>::*)(ContractRecord&, Transaction&);
using SelectSignature
    = DbResult<std::vector<ContractRecord>> (Table<ContractRecord>::*)(const Condition<ContractRecord>&) const;
using CountSignature = DbResult<uint64_t> (Table<ContractRecord>::*)(const Condition<ContractRecord>&) const;
using ClearSignature = DbResult<void> (Table<ContractRecord>::*)(ClearMode);

static_assert(std::is_same_v<decltype(static_cast<InsertSignature>(&Table<ContractRecord>::Insert)), InsertSignature>);
static_assert(
    std::is_same_v<decltype(static_cast<InsertTxSignature>(&Table<ContractRecord>::Insert)), InsertTxSignature>);
static_assert(std::is_same_v<decltype(static_cast<SelectSignature>(&Table<ContractRecord>::Select)), SelectSignature>);
static_assert(std::is_same_v<decltype(static_cast<CountSignature>(&Table<ContractRecord>::Count)), CountSignature>);
static_assert(std::is_same_v<decltype(&Table<ContractRecord>::Clear), ClearSignature>);
// Update/Delete 为转发模板（事务条件以 const 引用直达）；句柄字段与
// move-only 合同由上方所有权断言覆盖。

DbResult<void> OkVoid(uint64_t affectedRows = 0) { return {DbCode::OK, affectedRows, {}}; }

// 记录公共 IStorageEngine 每个能力的调用次数，并捕获唯一条件/更新载荷供表达式
// 合同断言（即时路径全程 const 引用直达引擎；观察字段为测试副本）。
class FakeStorageEngine : public IStorageEngine {
public:
    DbResult<void> Open(const EngineOpenRequest&) override
    {
        ++openCalls;
        return OkVoid();
    }

    DbResult<void> Prepare() override
    {
        ++prepareCalls;
        return OkVoid();
    }

    DbResult<uint64_t>
    Insert(const Ddl::CatalogTableMeta&, const TableScope&, const void* record, TransactionBinding*) override
    {
        ++executeCalls;
        ++insertCalls;
        lastInsertRecord = record; // 复制合同观察点：与原对象相同 = 全程零复制
        return {DbCode::OK, 7, 1, {}};
    }

    DbResult<void> Select(
        const Ddl::CatalogTableMeta&, const TableScope&, const ConditionData& where, const QueryOptions& options,
        ResultSink& sink) override
    {
        ++executeCalls;
        ++selectCalls;
        lastQuery = where;
        lastOptions = options;
        // 标量 sink（SelectOne）：同一 SELECT 通道，limit=1，未命中按合同返回 NOT_FOUND。
        if (ContractRecord* single = sink.Target<ContractRecord>()) {
            scalarSinkUsed = true;
            scalarSinkAddress = single;
            if (selectEmpty) {
                return {DbCode::NOT_FOUND, 0, "no matching row"};
            }
            *single = ContractRecord{11, 3};
            return OkVoid(1);
        }
        auto* rows = sink.Target<std::vector<ContractRecord>>();
        if (rows == nullptr) {
            return {DbCode::INVALID_ARGUMENT, 0, "sink type does not match"};
        }
        if (selectEmpty) {
            return OkVoid(0);
        }
        *rows = {ContractRecord{11, 3}, ContractRecord{12, 4}};
        // 引擎按合同执行下推的数量限制：Limit 是查询选项的一部分，由后端截断行集。
        if (options.limit.has_value() && *options.limit < rows->size()) {
            rows->resize(static_cast<size_t>(*options.limit));
        }
        return OkVoid(rows->size());
    }

    DbResult<uint64_t> Count(const Ddl::CatalogTableMeta&, const TableScope&, const ConditionData& where) override
    {
        ++executeCalls;
        ++countCalls;
        lastQuery = where;
        return {DbCode::OK, countResult, 1, {}};
    }

    DbResult<uint64_t> Update(
        const Ddl::CatalogTableMeta&, const TableScope&, const ConditionData& where, const UpdateData& update,
        TransactionBinding*) override
    {
        ++executeCalls;
        ++updateCalls;
        lastQuery = where;
        lastUpdate = update;
        sawMutation = true;
        return {DbCode::OK, 2, 0, {}};
    }

    DbResult<uint64_t>
    Delete(const Ddl::CatalogTableMeta&, const TableScope&, const ConditionData& where, TransactionBinding*) override
    {
        ++executeCalls;
        ++deleteCalls;
        lastQuery = where;
        sawDelete = true;
        return {DbCode::OK, 1, 0, {}};
    }

    DbResult<void> Clear(const Ddl::CatalogTableMeta&, const TableScope&, ClearMode mode) override
    {
        ++executeCalls;
        ++clearCalls;
        lastClearMode = mode;
        return OkVoid();
    }

    DbResult<std::vector<std::string>> ListTables() override
    {
        ++executeCalls;
        return {DbCode::OK, std::optional<std::vector<std::string>>{{"ContractRecord"}}, 1, {}};
    }

    DbResult<void> Begin(TransactionBinding& transaction) override
    {
        ++beginCalls;
        activeBinding = &transaction;
        return OkVoid();
    }

    DbResult<AtomicResult> Commit(TransactionBinding& transaction) override
    {
        ++commitCalls;
        if (&transaction != activeBinding) {
            return {DbCode::INVALID_STATE, std::nullopt, 0, "no matching transaction"};
        }
        activeBinding = nullptr;
        AtomicResult result;
        result.items.push_back({7, 1});
        return {DbCode::OK, std::move(result), 1, {}};
    }

    DbResult<void> Rollback(TransactionBinding& transaction) override
    {
        ++rollbackCalls;
        if (&transaction != activeBinding) {
            return {DbCode::INVALID_STATE, 0, "no matching transaction"};
        }
        activeBinding = nullptr;
        return OkVoid();
    }

    DbResult<void> Close() override
    {
        ++closeCalls;
        return OkVoid();
    }

    int openCalls = 0;
    int prepareCalls = 0;
    int executeCalls = 0;
    int insertCalls = 0;
    int selectCalls = 0;
    int countCalls = 0;
    int updateCalls = 0;
    int deleteCalls = 0;
    int clearCalls = 0;
    int beginCalls = 0;
    int commitCalls = 0;
    int rollbackCalls = 0;
    int closeCalls = 0;
    const void* lastInsertRecord{nullptr};
    const TransactionBinding* activeBinding{nullptr};
    ContractRecord* scalarSinkAddress{nullptr};
    ConditionData lastQuery{};
    QueryOptions lastOptions{};
    UpdateData lastUpdate{};
    ClearMode lastClearMode{ClearMode::KEEP_SEQUENCE};
    bool sawMutation = false;
    bool sawDelete = false;
    bool selectEmpty = false;
    bool scalarSinkUsed = false;
    uint64_t countResult = 5;
};

// 验证关系型薄接口仍通过公共引擎合同工作。
class FakeRelationalEngine final : public IRelationalEngine {
public:
    DbResult<void> Open(const EngineOpenRequest&) override { return OkVoid(); }
    DbResult<void> Prepare() override { return OkVoid(); }
    DbResult<uint64_t>
    Insert(const Ddl::CatalogTableMeta&, const TableScope&, const void*, TransactionBinding*) override
    {
        return {DbCode::OK, 1, 1, {}};
    }
    DbResult<void> Select(
        const Ddl::CatalogTableMeta&, const TableScope&, const ConditionData&, const QueryOptions&,
        ResultSink&) override
    {
        return OkVoid();
    }
    DbResult<uint64_t> Count(const Ddl::CatalogTableMeta&, const TableScope&, const ConditionData&) override
    {
        return {DbCode::OK, 0, 1, {}};
    }
    DbResult<uint64_t> Update(
        const Ddl::CatalogTableMeta&, const TableScope&, const ConditionData&, const UpdateData&,
        TransactionBinding*) override
    {
        return {DbCode::OK, 0, 0, {}};
    }
    DbResult<uint64_t>
    Delete(const Ddl::CatalogTableMeta&, const TableScope&, const ConditionData&, TransactionBinding*) override
    {
        return {DbCode::OK, 0, 0, {}};
    }
    DbResult<void> Clear(const Ddl::CatalogTableMeta&, const TableScope&, ClearMode) override { return OkVoid(); }
    DbResult<std::vector<std::string>> ListTables() override
    {
        return {DbCode::OK, std::optional<std::vector<std::string>>{}, 0, {}};
    }
    DbResult<void> Begin(TransactionBinding&) override { return OkVoid(); }
    DbResult<AtomicResult> Commit(TransactionBinding&) override { return {DbCode::OK, AtomicResult{}, 0, {}}; }
    DbResult<void> Rollback(TransactionBinding&) override { return OkVoid(); }
    DbResult<void> Close() override { return OkVoid(); }

    bool backupRelationalCalled = false;

    DbResult<DatabaseBackupArtifact> BackupRelational(const EngineBackupRequest&) override
    {
        backupRelationalCalled = true;
        return {DbCode::OK, DatabaseBackupArtifact{}, 0, {}};
    }
};

// 验证 BackupRequest 的目的和路径能无损传递到实际 provider。
class FakeBackupProvider final : public IStorageBackupProvider {
public:
    DbResult<BackupArtifact> Backup(const BackupRequest& request) override
    {
        ++calls;
        purpose = request.purpose;
        return {DbCode::OK, std::optional<BackupArtifact>{{request.destinationPath}}, 0, {}};
    }
    int calls{0};
    BackupPurpose purpose{BackupPurpose::Exit};
};

// 组装一个指向 FakeStorageEngine 的最小可用会话。
struct ContractSession {
    FakeStorageEngine engine;
    StorageOpenRequest request;
    std::optional<StorageSession> session;

    DbResult<void> Open(const std::string& sessionId, StorageOpenMode mode = StorageOpenMode::READ_WRITE_CREATE)
    {
        // 引擎按逻辑库注册（databaseId 与测试目录 kContractTestDatabase 一致）：
        // 表 → 逻辑库归属由 Ddl::DatabaseOf<T> 编译期派生，会话按库路由槽定位引擎。
        SessionStorageBinding binding;
        binding.sessionId = sessionId;
        binding.dataNamespace = "contract";
        request.mode = mode;
        request.engines = {EngineRegistration{"CONTRACT_TEST_DB", kTestFakeBackendName, &engine}};
        auto opened = StorageSession::Open(binding, request);
        if (!opened.ok()) {
            return {opened.code, opened.affectedRows, opened.diagnostic};
        }
        session.emplace(std::move(*opened.value));
        return {DbCode::OK, 0, {}};
    }
};

// DbResult 同时保留返回码、可选值、影响行数和诊断信息；箭头访问不吞错误。
TEST(StorageContractTest, DbResultCarriesCodeValueRowsAndDiagnostic)
{
    DbResult<int> valueResult{DbCode::OK, 7, 3, "value"};
    ASSERT_TRUE(valueResult.ok());
    ASSERT_TRUE(valueResult.value.has_value());
    EXPECT_EQ(*valueResult.value, 7);
    EXPECT_EQ(*valueResult, 7);
    EXPECT_EQ(*valueResult.operator->(), 7);
    EXPECT_EQ(valueResult.affectedRows, 3U);
    EXPECT_EQ(valueResult.diagnostic, "value");

    DbResult<int> failed{DbCode::NOT_FOUND, std::nullopt, 0, "missing"};
    EXPECT_FALSE(failed.ok());
    EXPECT_EQ(failed.code, DbCode::NOT_FOUND);
    EXPECT_FALSE(failed.value.has_value());
    EXPECT_EQ(failed.diagnostic, "missing");

    DbResult<void> voidResult{DbCode::NOT_FOUND, 0, "missing"};
    EXPECT_FALSE(voidResult.ok());
    EXPECT_EQ(voidResult.code, DbCode::NOT_FOUND);
    EXPECT_EQ(voidResult.diagnostic, "missing");
}

// 唯一 IStorageEngine vtable 的所有能力都能被统一调用。
TEST(StorageContractTest, CommonInterfaceDispatchesEveryCapability)
{
    FakeStorageEngine fake;
    IStorageEngine& engine = fake;
    const auto& meta = RecordSchema<ContractRecord>::Meta();
    ContractRecord record;
    std::vector<ContractRecord> rows;
    EngineOpenRequest openRequest;
    EXPECT_TRUE(engine.Open(openRequest).ok());
    EXPECT_TRUE(engine.Prepare().ok());
    EXPECT_TRUE(engine.Insert(meta, TableScope::Static(), &record, nullptr).ok());
    auto sink = ResultSink::For(rows);
    EXPECT_TRUE(engine.Select(meta, TableScope::Static(), All<ContractRecord>().data, {}, sink).ok());
    EXPECT_TRUE(engine.Count(meta, TableScope::Static(), All<ContractRecord>().data).ok());
    EXPECT_TRUE(engine
                    .Update(
                        meta, TableScope::Static(), Eq(&ContractRecord::state, int64_t{1}).data,
                        UpdateData{{1, ChangeKind::SET, DbScalar{int64_t{2}}}}, nullptr)
                    .ok());
    EXPECT_TRUE(engine.Delete(meta, TableScope::Static(), Eq(&ContractRecord::state, int64_t{2}).data, nullptr).ok());
    EXPECT_TRUE(engine.Clear(meta, TableScope::Static(), ClearMode::KEEP_SEQUENCE).ok());
    EXPECT_TRUE(engine.ListTables().ok());
    TransactionBinding transaction;
    EXPECT_TRUE(engine.Begin(transaction).ok());
    EXPECT_TRUE(engine.Insert(meta, TableScope::Static(), &record, &transaction).ok());
    EXPECT_TRUE(engine.Commit(transaction).ok());
    EXPECT_TRUE(engine.Begin(transaction).ok());
    EXPECT_TRUE(engine.Rollback(transaction).ok());
    EXPECT_TRUE(engine.Close().ok());

    EXPECT_EQ(fake.openCalls, 1);
    EXPECT_EQ(fake.prepareCalls, 1);
    EXPECT_EQ(fake.insertCalls, 2);
    EXPECT_EQ(fake.selectCalls, 1);
    EXPECT_EQ(fake.countCalls, 1);
    EXPECT_EQ(fake.updateCalls, 1);
    EXPECT_EQ(fake.deleteCalls, 1);
    EXPECT_EQ(fake.clearCalls, 1);
    EXPECT_EQ(fake.beginCalls, 2);
    EXPECT_EQ(fake.commitCalls, 1);
    EXPECT_EQ(fake.rollbackCalls, 1);
    EXPECT_EQ(fake.closeCalls, 1);
}

// 关系型备份扩展不复制或绕过公共 CRUD/事务接口。
TEST(StorageContractTest, RelationalBackupDelegatesThroughCommonContract)
{
    FakeRelationalEngine relational;
    EngineBackupRequest request;

    EXPECT_TRUE(relational.BackupRelational(request).ok());
    EXPECT_TRUE(relational.backupRelationalCalled);
}

// 编译期 static_assert 已证明 Table、Transaction、Session 和 lease 的所有权约束。
TEST(StorageContractTest, TableAndTransactionRemainMoveOnly) { SUCCEED(); }

// 类型安全表达式把成员指针映射为字段编号，值按成员类别编码为数据库标量；
// 单谓词内联一个 ConditionClause（不为叶子分配 vector）。
TEST(StorageContractTest, TypedExpressionsMapMembersAndEncodeValues)
{
    const auto equals = Eq(&ContractRecord::state, int64_t{3});
    const auto* equalsClause = std::get_if<ConditionClause>(&equals.data);
    ASSERT_NE(equalsClause, nullptr);
    EXPECT_EQ(equalsClause->fieldId, 1U);
    EXPECT_EQ(equalsClause->compare, Compare::EQUAL);
    const auto* equalsScalar = std::get_if<DbScalar>(&equalsClause->value);
    ASSERT_NE(equalsScalar, nullptr); // 单值内联（非 vector）
    ASSERT_NE(std::get_if<int64_t>(equalsScalar), nullptr);

    const auto notEqual = Ne(&ContractRecord::state, int64_t{3});
    EXPECT_EQ(std::get<ConditionClause>(notEqual.data).compare, Compare::NOT_EQUAL);
    const auto less = Lt(&ContractRecord::state, int64_t{3});
    EXPECT_EQ(std::get<ConditionClause>(less.data).compare, Compare::LESS);
    const auto lessOrEqual = Le(&ContractRecord::state, int64_t{3});
    EXPECT_EQ(std::get<ConditionClause>(lessOrEqual.data).compare, Compare::LESS_OR_EQUAL);
    const auto greater = Gt(&ContractRecord::state, int64_t{3});
    EXPECT_EQ(std::get<ConditionClause>(greater.data).compare, Compare::GREATER);
    const auto greaterOrEqual = Ge(&ContractRecord::state, int64_t{3});
    EXPECT_EQ(std::get<ConditionClause>(greaterOrEqual.data).compare, Compare::GREATER_OR_EQUAL);

    const auto unsignedEquals = Eq(&ContractRecord::id, uint64_t{9});
    const auto* unsignedClause = std::get_if<ConditionClause>(&unsignedEquals.data);
    ASSERT_NE(unsignedClause, nullptr);
    EXPECT_EQ(unsignedClause->fieldId, 0U);
    ASSERT_NE(std::get_if<uint64_t>(std::get_if<DbScalar>(&unsignedClause->value)), nullptr);

    const auto collection = In(&ContractRecord::id, std::vector<uint64_t>{1, 2, 3});
    const auto* collectionClause = std::get_if<ConditionClause>(&collection.data);
    ASSERT_NE(collectionClause, nullptr);
    EXPECT_EQ(collectionClause->compare, Compare::IN);
    ASSERT_EQ(std::get<std::vector<DbScalar>>(collectionClause->value).size(), 3U);

    // 空 In 是合法条件：叶子保留 IN 比较与 0 个绑定值，语义为恒假（匹配 0 行）。
    const auto emptyCollection = In(&ContractRecord::id, std::vector<uint64_t>{});
    const auto* emptyClause = std::get_if<ConditionClause>(&emptyCollection.data);
    ASSERT_NE(emptyClause, nullptr);
    EXPECT_EQ(emptyClause->compare, Compare::IN);
    EXPECT_TRUE(std::get<std::vector<DbScalar>>(emptyClause->value).empty());

    // 左值集合按 const 引用直读：条件持有唯一一份编码载荷，调用后原集合保持
    // 完整可复用（不为按值参数复制整集，也不消费调用方容器）。
    std::vector<uint64_t> reusableIds{1, 2, 3};
    const auto firstUse = In(&ContractRecord::id, reusableIds);
    const auto* firstClause = std::get_if<ConditionClause>(&firstUse.data);
    ASSERT_NE(firstClause, nullptr);
    ASSERT_EQ(std::get<std::vector<DbScalar>>(firstClause->value).size(), 3U);
    EXPECT_EQ(reusableIds.size(), 3U);
    const auto secondUse = In(&ContractRecord::id, reusableIds);
    const auto* secondClause = std::get_if<ConditionClause>(&secondUse.data);
    ASSERT_NE(secondClause, nullptr);
    ASSERT_EQ(std::get<std::vector<DbScalar>>(secondClause->value).size(), 3U);
    EXPECT_EQ(reusableIds.size(), 3U);

    const auto stringEquals = Eq(&ContractRecord::tag, std::string{"ready"});
    const auto* stringClause = std::get_if<ConditionClause>(&stringEquals.data);
    ASSERT_NE(stringClause, nullptr);
    EXPECT_EQ(stringClause->fieldId, 3U);
    ASSERT_NE(std::get_if<std::string>(std::get_if<DbScalar>(&stringClause->value)), nullptr);

    const auto prefix = StartsWith(&ContractRecord::label, std::string{"ready"});
    const auto* prefixClause = std::get_if<ConditionClause>(&prefix.data);
    ASSERT_NE(prefixClause, nullptr);
    EXPECT_EQ(prefixClause->compare, Compare::PREFIX);
    ASSERT_NE(std::get_if<std::string>(std::get_if<DbScalar>(&prefixClause->value)), nullptr);

    const auto everything = All<ContractRecord>();
    EXPECT_TRUE(std::holds_alternative<std::monostate>(everything.data));

    const auto sort = OrderBy(&ContractRecord::id, SortDirection::ASC);
    EXPECT_EQ(sort.fieldId, 0U);
    EXPECT_EQ(sort.direction, SortDirection::ASC);
    // OrderBy 方向参数缺省为 ASC：只有调用 OrderBy 才启用排序。
    const auto defaultSort = OrderBy(&ContractRecord::id);
    EXPECT_EQ(defaultSort.fieldId, 0U);
    EXPECT_EQ(defaultSort.direction, SortDirection::ASC);
    const auto limited = Limit(10U);
    EXPECT_EQ(limited.limit, 10U);

    const auto assigned = Set(&ContractRecord::state, int64_t{4});
    EXPECT_EQ(assigned.fieldId, 1U);
    EXPECT_FALSE(assigned.increment);
    const auto incremented = Increment(&ContractRecord::id, uint64_t{2});
    EXPECT_EQ(incremented.fieldId, 0U);
    EXPECT_TRUE(incremented.increment);
}

// And/Or 生成平铺子句：括号语义由 openGroupsBefore/closeGroupsAfter 承载。
TEST(StorageContractTest, NestedAndOrBuildsParenthesizedFlatClauses)
{
    const auto tree = Or(
        And(Eq(&ContractRecord::id, uint64_t{1}), Eq(&ContractRecord::state, int64_t{2})),
        And(Eq(&ContractRecord::id, uint64_t{3}), Eq(&ContractRecord::state, int64_t{4})));
    const auto* treeClauses = std::get_if<std::vector<ConditionClause>>(&tree.data);
    ASSERT_NE(treeClauses, nullptr);
    ASSERT_EQ(treeClauses->size(), 4U);
    // 第一个 AND 组：子句 0（首，无连接）+ 子句 1（AND 连接 + 组开括号）
    EXPECT_EQ((*treeClauses)[0].joinWithPrevious, ConditionJoin::NONE);
    EXPECT_EQ((*treeClauses)[0].openGroupsBefore, 1U);
    EXPECT_EQ((*treeClauses)[1].joinWithPrevious, ConditionJoin::AND);
    EXPECT_EQ((*treeClauses)[1].closeGroupsAfter, 1U);
    // 第二个 AND 组经 OR 连接
    EXPECT_EQ((*treeClauses)[2].joinWithPrevious, ConditionJoin::OR);
    EXPECT_EQ((*treeClauses)[2].openGroupsBefore, 1U);
    EXPECT_EQ((*treeClauses)[3].closeGroupsAfter, 1U);

    const auto flat = And(Eq(&ContractRecord::id, uint64_t{1}), Eq(&ContractRecord::state, int64_t{2}));
    const auto* flatClauses = std::get_if<std::vector<ConditionClause>>(&flat.data);
    ASSERT_NE(flatClauses, nullptr);
    ASSERT_EQ(flatClauses->size(), 2U);
    EXPECT_EQ((*flatClauses)[0].joinWithPrevious, ConditionJoin::NONE);
    EXPECT_EQ((*flatClauses)[1].joinWithPrevious, ConditionJoin::AND);

    const auto mixed = And(
        Or(Eq(&ContractRecord::id, uint64_t{1}), Eq(&ContractRecord::id, uint64_t{2})),
        Eq(&ContractRecord::state, int64_t{5}));
    const auto* mixedClauses = std::get_if<std::vector<ConditionClause>>(&mixed.data);
    ASSERT_NE(mixedClauses, nullptr);
    ASSERT_EQ(mixedClauses->size(), 3U);
    EXPECT_EQ((*mixedClauses)[0].openGroupsBefore, 1U);
    EXPECT_EQ((*mixedClauses)[1].joinWithPrevious, ConditionJoin::OR);
    EXPECT_EQ((*mixedClauses)[1].closeGroupsAfter, 1U);
    EXPECT_EQ((*mixedClauses)[2].joinWithPrevious, ConditionJoin::AND);

    // 嵌套消解：All 子条件被吸收；And(All, leaf) 退化为单谓词。
    const auto single = And(All<ContractRecord>(), Eq(&ContractRecord::state, int64_t{2}));
    EXPECT_NE(std::get_if<ConditionClause>(&single.data), nullptr);
    const auto none = And(All<ContractRecord>(), All<ContractRecord>());
    EXPECT_TRUE(std::holds_alternative<std::monostate>(none.data));
}

// 【已删除结构】TableBindingCatalog 与 mappingId 发布合同随逐表绑定目录删除：
// 会话只保存每逻辑库路由槽，表归属由 RecordSchema<T>::DatabaseId() 编译期派生。

// Session→Table/Transaction→Engine 的真实调用链只使用注册 route，并验证拥有型事务快照。
TEST(StorageContractTest, SessionTableTransactionAndContextUseOnlyRegisteredEngine)
{
    ContractSession holder;
    ASSERT_TRUE(holder.Open("contract-session").ok());
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());
    {
        auto lease = ProcessStorageContext::Acquire();
        ASSERT_TRUE(lease.ok());
        EXPECT_EQ(ProcessStorageContext::RemoveAndDrain(std::chrono::milliseconds(0)).code, DbCode::RESOURCE_EXHAUSTED);
        EXPECT_EQ(&lease.value->Get(), &*holder.session);
    }
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());

    {
        auto tableResult = holder.session->Table<ContractRecord>();
        ASSERT_TRUE(tableResult.ok());
        auto table = std::move(*tableResult.value);
        ContractRecord record;
        const auto inserted = table.Insert(record);
        ASSERT_TRUE(inserted.ok());
        EXPECT_EQ(inserted.value->value, 7U);

        auto transactionResult = holder.session->BeginTransaction();
        ASSERT_TRUE(transactionResult.ok());
        auto transaction = std::move(*transactionResult.value);
        EXPECT_TRUE(table.Insert(record, transaction).ok());
        // 事务 INSERT 的生成主键 = 引擎返回值（fake 固定返回 7），Commit 前已写回
        EXPECT_EQ(record.id, 7U);
        // 事务 INSERT 与即时路径共用零复制直达：引擎收到的是调用方原对象地址
        // （无 Stage 副本，无 ErasedRecordPtr）。
        EXPECT_EQ(holder.engine.lastInsertRecord, &record);
        record.id = 99;
        EXPECT_TRUE(transaction.Commit().ok());
        EXPECT_EQ(transaction.State(), TransactionState::COMMITTED);
        EXPECT_EQ(transaction.Commit().code, DbCode::INVALID_STATE);
    }
    EXPECT_TRUE(holder.session->Close().ok());
}

// quiesce 阻止业务入口，token 只允许同会话备份且 release 后全部副本失效。
TEST(StorageContractTest, QuiesceTokenControlsBackupOnlySessionAndCanBeReleased)
{
    FakeStorageEngine businessEngine;
    FakeStorageEngine backupEngine;
    FakeBackupProvider backupProvider;
    SessionStorageBinding binding;
    binding.sessionId = "backup-session";
    binding.dataNamespace = "backup";
    binding.databases.push_back({"CONTRACT_TEST_DB", "", &Ddl::kContractTestDatabase});
    StorageOpenRequest businessRequest;
    businessRequest.engines = {EngineRegistration{"CONTRACT_TEST_DB", kTestFakeBackendName, &businessEngine}};
    auto businessOpened = StorageSession::Open(binding, businessRequest);
    ASSERT_TRUE(businessOpened.ok());
    auto business = std::move(*businessOpened.value);
    ASSERT_TRUE(ProcessStorageContext::Install(business).ok());
    auto token = ProcessStorageContext::ApplyQuiesceGate();
    ASSERT_TRUE(token.ok());
    EXPECT_EQ(ProcessStorageContext::Acquire().code, DbCode::INVALID_STATE);
    EXPECT_EQ(AcquireTable<ContractRecord>().code, DbCode::INVALID_STATE);
    EXPECT_EQ(BeginTransaction().code, DbCode::INVALID_STATE);

    StorageOpenRequest backupRequest = businessRequest;
    backupRequest.mode = StorageOpenMode::BACKUP_ONLY;
    backupRequest.engines = {EngineRegistration{"CONTRACT_TEST_DB", kTestFakeBackendName, &backupEngine}};
    backupRequest.backupProvider = &backupProvider;
    auto otherBinding = binding;
    otherBinding.sessionId = "other-backup-session";
    EXPECT_EQ(StorageSession::OpenForBackup(otherBinding, backupRequest, *token.value).code, DbCode::INVALID_ARGUMENT);
    auto backupOpened = StorageSession::OpenForBackup(binding, backupRequest, *token.value);
    ASSERT_TRUE(backupOpened.ok());
    auto backup = std::move(*backupOpened.value);
    EXPECT_EQ(backup.Table<ContractRecord>().code, DbCode::INVALID_STATE);
    EXPECT_EQ(backup.BeginTransaction().code, DbCode::INVALID_STATE);
    const auto artifact = backup.Backup({"/tmp/contract-backup", BackupPurpose::OnlineReset});
    ASSERT_TRUE(artifact.ok());
    EXPECT_EQ(artifact.value->destinationPath, "/tmp/contract-backup");
    EXPECT_EQ(backupProvider.calls, 1);
    EXPECT_EQ(backupProvider.purpose, BackupPurpose::OnlineReset);
    EXPECT_TRUE(backup.Close().ok());

    EXPECT_TRUE(ProcessStorageContext::ReleaseQuiesceGate(*token.value).ok());
    EXPECT_FALSE(token.value->Valid());
    EXPECT_EQ(StorageSession::OpenForBackup(binding, backupRequest, *token.value).code, DbCode::INVALID_ARGUMENT);
    EXPECT_TRUE(business.Table<ContractRecord>().ok());
    EXPECT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(business.Close().ok());
}

// 具名 ResetPlan 只能由同一协调器在 active epoch gate 内执行。
TEST(StorageContractTest, SessionCoordinatorRunsNamedResetOnlyInsideItsActiveGate)
{
    FakeStorageEngine engine;
    SessionStorageBinding binding;
    binding.sessionId = "reset-session";
    binding.dataNamespace = "reset";
    binding.databases.push_back({"CONTRACT_TEST_DB", "", &Ddl::kContractTestDatabase});
    StorageOpenRequest request;
    request.engines = {EngineRegistration{"CONTRACT_TEST_DB", kTestFakeBackendName, &engine}};

    auto opened = SessionCoordinator::Open(binding, request);
    ASSERT_TRUE(opened.ok());
    auto coordinator = std::move(*opened.value);
    ASSERT_TRUE(coordinator.Install().ok());
    auto token = coordinator.ApplyQuiesceGate();
    ASSERT_TRUE(token.ok());
    EXPECT_EQ(ProcessStorageContext::Acquire().code, DbCode::INVALID_STATE);

    ResetPlan invalid{"", {{"ContractRecord", TableScope::Static(), ClearMode::RESET_SEQUENCE}}, {}};
    EXPECT_EQ(coordinator.ExecuteReset(invalid, *token.value).code, DbCode::INVALID_STATE);
    ResetPlan plan{
        "RESET_COMMAND", {{"ContractRecord", TableScope::Static(), ClearMode::RESET_SEQUENCE}}, {{"shared-memory"}}};
    const auto reset = coordinator.ExecuteReset(plan, *token.value);
    ASSERT_TRUE(reset.ok());
    EXPECT_EQ(reset.value->clearedTables, 1U);
    ASSERT_EQ(reset.value->pendingExternalResources.size(), 1U);
    EXPECT_EQ(reset.value->pendingExternalResources.front(), "shared-memory");
    EXPECT_EQ(engine.executeCalls, 1);

    EXPECT_TRUE(coordinator.ReleaseQuiesceGate(*token.value).ok());
    EXPECT_FALSE(token.value->Valid());
    EXPECT_EQ(coordinator.ExecuteReset(plan, *token.value).code, DbCode::INVALID_STATE);
    EXPECT_TRUE(coordinator.Close().ok());
}

// fork 子进程重置后不能继续取得父进程会话（真实 fork：child 只 reset、验证拒绝
// 后 _exit，不关闭/析构继承会话对象——继承对象整体放弃由退出回收；父进程会话
// 不受影响并按正常排空路径清理）。
TEST(StorageContractTest, ForkChildResetReplacesInheritedProcessContext)
{
    FakeStorageEngine engine;
    SessionStorageBinding binding;
    binding.sessionId = "fork-session";
    binding.dataNamespace = "fork";
    binding.databases.push_back({"CONTRACT_TEST_DB", "", &Ddl::kContractTestDatabase});
    StorageOpenRequest request;
    request.engines = {EngineRegistration{"CONTRACT_TEST_DB", kTestFakeBackendName, &engine}};
    auto opened = StorageSession::Open(binding, request);
    ASSERT_TRUE(opened.ok());
    auto session = std::move(*opened.value);
    ASSERT_TRUE(ProcessStorageContext::Install(session).ok());
    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        // 子进程：重置后不得再取得继承会话；子进程内不使用 ASSERT（失败会跳出
        // 测试函数），以布尔链收集结果后立即 _exit。
        ProcessStorageContext::ResetAfterForkInChild();
        bool ok = ProcessStorageContext::Acquire().code == DbCode::INVALID_STATE;
        ok = ok && AcquireTable<ContractRecord>().code == DbCode::INVALID_STATE;
        ::_exit(ok ? 0 : 1);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
    // 父进程：会话仍安装且可正常排空关闭（child 重置只作用于子进程本地状态）。
    EXPECT_TRUE(ProcessStorageContext::Acquire().ok());
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(session.Close().ok());
}

// AcquireTable 单步取得类型化表句柄：业务不再接触租约与 Session 门面链条。
// 句柄存活期间排空保护生效（activeTables 不归零则关闭等待），故先释放句柄再关闭会话。
TEST(StorageContractTest, AcquireTableProvidesSingleStepTypedTableAccess)
{
    ContractSession holder;
    ASSERT_TRUE(holder.Open("acquire-table-session").ok());
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());

    {
        auto table = AcquireTable<ContractRecord>();
        ASSERT_TRUE(table.ok());
        ContractRecord record;
        const auto inserted = table->Insert(record);
        ASSERT_TRUE(inserted.ok());
        EXPECT_EQ(record.id, 7U);
        EXPECT_EQ(holder.engine.executeCalls, 1);

        const auto selected = table->Select(Eq(&ContractRecord::id, uint64_t{7}), Limit(1U));
        ASSERT_TRUE(selected.ok());
        ASSERT_EQ(selected.value->size(), 1U);
        EXPECT_EQ(selected.value->front().id, 11U);
        EXPECT_EQ(holder.engine.selectCalls, 1);
    }

    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(holder.session->Close().ok());
}

// 条件查询把排序与数量下推为逻辑查询的一部分，全量查询不需要条件。
TEST(StorageContractTest, SelectPushesConditionSortAndLimitAsOneOperation)
{
    ContractSession holder;
    ASSERT_TRUE(holder.Open("select-pushdown-session").ok());
    {
        auto tableResult = holder.session->Table<ContractRecord>();
        ASSERT_TRUE(tableResult.ok());
        auto table = std::move(*tableResult.value);

        const auto rows = table.Select(
            And(Eq(&ContractRecord::state, int64_t{3}), In(&ContractRecord::id, std::vector<uint64_t>{1, 2})),
            OrderBy(&ContractRecord::id, SortDirection::DESC), Limit(10U));
        ASSERT_TRUE(rows.ok());
        ASSERT_EQ(rows.value->size(), 2U);
        EXPECT_EQ(holder.engine.selectCalls, 1);
        // 唯一 ConditionData：平铺两个子句（AND 连接），不构造条件树
        const auto* clauses = std::get_if<std::vector<ConditionClause>>(&holder.engine.lastQuery);
        ASSERT_NE(clauses, nullptr);
        ASSERT_EQ(clauses->size(), 2U);
        EXPECT_EQ((*clauses)[0].joinWithPrevious, ConditionJoin::NONE);
        EXPECT_EQ((*clauses)[1].joinWithPrevious, ConditionJoin::AND);
        EXPECT_EQ((*clauses)[1].compare, Compare::IN);
        // 唯一 QueryOptions：排序键与数量限制随同一次调用下推
        ASSERT_EQ(holder.engine.lastOptions.sorts.size(), 1U);
        EXPECT_EQ(holder.engine.lastOptions.sorts[0].first, 0U);
        EXPECT_EQ(holder.engine.lastOptions.sorts[0].second, Order::DESCENDING);
        ASSERT_TRUE(holder.engine.lastOptions.limit.has_value());
        EXPECT_EQ(*holder.engine.lastOptions.limit, 10U);

        const auto all = table.Select(All<ContractRecord>());
        ASSERT_TRUE(all.ok());
        EXPECT_TRUE(std::holds_alternative<std::monostate>(holder.engine.lastQuery));

        // 比较能力由字段类型与通用合同决定：有符号整数字段支持范围比较，无每表白名单。
        const auto ranged = table.Select(Lt(&ContractRecord::state, int64_t{1}));
        EXPECT_EQ(ranged.code, DbCode::OK);
    }
    EXPECT_TRUE(holder.session->Close().ok());
}

// SelectOne 标量单行合同：仍是唯一 SELECT kind，内部强制 limit=1，标量 sink 写回，
// 未命中 NOT_FOUND；显式 OrderBy 保留；主键查询只是 Eq 条件。
TEST(StorageContractTest, SelectOneIsSelectKindWithLimitOneAndScalarSink)
{
    ContractSession holder;
    ASSERT_TRUE(holder.Open("select-one-session").ok());
    {
        auto tableResult = holder.session->Table<ContractRecord>();
        ASSERT_TRUE(tableResult.ok());
        auto table = std::move(*tableResult.value);

        holder.engine.selectEmpty = false;
        ContractRecord directTarget;
        const auto one = table.SelectOne(Eq(&ContractRecord::id, uint64_t{11}));
        ASSERT_TRUE(one.ok());
        ASSERT_TRUE(one.value.has_value());
        EXPECT_EQ(one.value->state, 3);
        EXPECT_EQ(holder.engine.selectCalls, 1);
        ASSERT_TRUE(holder.engine.lastOptions.limit.has_value());
        EXPECT_EQ(*holder.engine.lastOptions.limit, 1U);
        // 标量 sink：结果直接写回最终 T 目标，不构造单元素 vector。
        EXPECT_TRUE(holder.engine.scalarSinkUsed);
        (void)directTarget;

        // 未命中：NOT_FOUND（后端错误原样传播，不伪造默认值）。
        holder.engine.selectEmpty = true;
        const auto missing = table.SelectOne(Eq(&ContractRecord::id, uint64_t{404}));
        EXPECT_EQ(missing.code, DbCode::NOT_FOUND);
        EXPECT_FALSE(missing.value.has_value());

        // 显式排序：SelectOne 的 OrderBy 进入排序键列表（DESC 保留）。
        holder.engine.selectEmpty = false;
        const auto ordered = table.SelectOne(
            Eq(&ContractRecord::state, int64_t{3}), OrderBy(&ContractRecord::id, SortDirection::DESC));
        ASSERT_TRUE(ordered.ok());
        ASSERT_EQ(holder.engine.lastOptions.sorts.size(), 1U);
        EXPECT_EQ(holder.engine.lastOptions.sorts[0].first, 0U);
        EXPECT_EQ(holder.engine.lastOptions.sorts[0].second, Order::DESCENDING);
        ASSERT_TRUE(holder.engine.lastOptions.limit.has_value());
        EXPECT_EQ(*holder.engine.lastOptions.limit, 1U);
    }
    EXPECT_TRUE(holder.session->Close().ok());
}

// SelectOne 的多排序键：多个 OrderBy 按传入顺序成为复合排序键。
TEST(StorageContractTest, SelectOneAcceptsCompositeOrderBy)
{
    ContractSession holder;
    ASSERT_TRUE(holder.Open("select-one-composite-order").ok());
    {
        auto tableResult = holder.session->Table<ContractRecord>();
        ASSERT_TRUE(tableResult.ok());
        auto table = std::move(*tableResult.value);
        const auto one = table.SelectOne(
            Eq(&ContractRecord::state, int64_t{3}), OrderBy(&ContractRecord::state, SortDirection::DESC),
            OrderBy(&ContractRecord::id, SortDirection::ASC));
        ASSERT_TRUE(one.ok());
        ASSERT_EQ(holder.engine.lastOptions.sorts.size(), 2U);
        EXPECT_EQ(holder.engine.lastOptions.sorts[0].first, 1U);
        EXPECT_EQ(holder.engine.lastOptions.sorts[0].second, Order::DESCENDING);
        EXPECT_EQ(holder.engine.lastOptions.sorts[1].first, 0U);
        EXPECT_EQ(holder.engine.lastOptions.sorts[1].second, Order::ASCENDING);
    }
    EXPECT_TRUE(holder.session->Close().ok());
}

// 一次性 SelectOne 入口：只 Acquire 一次、执行一次 Table 操作；支持显式动态 scope。
TEST(StorageContractTest, OneShotSelectOneEntryRunsSingleOperation)
{
    ContractSession holder;
    ASSERT_TRUE(holder.Open("select-one-oneshot").ok());
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());
    const auto one = SelectOne<ContractRecord>(Eq(&ContractRecord::id, uint64_t{11}));
    ASSERT_TRUE(one.ok());
    ASSERT_TRUE(one.value.has_value());
    EXPECT_EQ(one.value->state, 3);
    EXPECT_EQ(holder.engine.executeCalls, 1);
    EXPECT_TRUE(holder.engine.scalarSinkUsed);

    const auto scoped = SelectOne<ContractRecord>(TableScope::Static(), Eq(&ContractRecord::id, uint64_t{11}));
    ASSERT_TRUE(scoped.ok());
    EXPECT_EQ(scoped.value->state, 3);
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
}

// SelectOne 可调用形态合同：接受零个或多个 OrderBy；返回 DbResult<T> 标量形态。
// SelectOne 与 Limit 的组合由 Table<T>::CollectOneRowClause 的 static_assert 在编译期拒绝
// （负路径由验证命令单独证明：错误信息 "SelectOne accepts OrderBy clauses only; use Select for Limit"）。
TEST(StorageContractTest, SelectOneCallableShapeAndScalarResultType)
{
    static_assert(std::is_same_v<
                  decltype(std::declval<const Table<ContractRecord>&>().SelectOne(
                      std::declval<const Condition<ContractRecord>&>())),
                  DbResult<ContractRecord>>);
    static_assert(
        std::is_same_v<
            decltype(std::declval<const Table<ContractRecord>&>().SelectOne(
                std::declval<const Condition<ContractRecord>&>(), std::declval<const SortClause<ContractRecord>&>(),
                std::declval<const SortClause<ContractRecord>&>())),
            DbResult<ContractRecord>>);
    SUCCEED();
}

// 多行 Select 语义保持：Limit(0) 是 OK + 空行集；重复 Limit 入口拒绝；
// 多个 OrderBy 复合排序；单行读取不再需要 Limit(1) 样板。
TEST(StorageContractTest, SelectKeepsMultiRowSemanticsAndRejectsDuplicateLimit)
{
    ContractSession holder;
    ASSERT_TRUE(holder.Open("select-multi-row-semantics").ok());
    {
        auto tableResult = holder.session->Table<ContractRecord>();
        ASSERT_TRUE(tableResult.ok());
        auto table = std::move(*tableResult.value);

        const auto emptyLimit = table.Select(All<ContractRecord>(), Limit(0U));
        EXPECT_EQ(emptyLimit.code, DbCode::OK);
        ASSERT_TRUE(emptyLimit.value.has_value());
        EXPECT_TRUE(emptyLimit.value->empty());

        const auto duplicated = table.Select(All<ContractRecord>(), Limit(1U), Limit(2U));
        EXPECT_EQ(duplicated.code, DbCode::INVALID_ARGUMENT);

        const auto composite = table.Select(
            All<ContractRecord>(), OrderBy(&ContractRecord::state, SortDirection::DESC),
            OrderBy(&ContractRecord::id, SortDirection::ASC));
        ASSERT_TRUE(composite.ok());
        ASSERT_EQ(holder.engine.lastOptions.sorts.size(), 2U);
    }
    EXPECT_TRUE(holder.session->Close().ok());
}

// Count 是数据库端计数：不加载行集，条件同样下推。
TEST(StorageContractTest, CountIsEngineSideCounting)
{
    ContractSession holder;
    ASSERT_TRUE(holder.Open("count-session").ok());
    {
        auto tableResult = holder.session->Table<ContractRecord>();
        ASSERT_TRUE(tableResult.ok());
        auto table = std::move(*tableResult.value);

        holder.engine.countResult = 42;
        const auto count = table.Count(Eq(&ContractRecord::state, int64_t{3}));
        ASSERT_TRUE(count.ok());
        EXPECT_EQ(*count.value, 42U);
        EXPECT_EQ(holder.engine.countCalls, 1);
        EXPECT_NE(std::get_if<ConditionClause>(&holder.engine.lastQuery), nullptr);

        const auto total = table.Count(All<ContractRecord>());
        ASSERT_TRUE(total.ok());
        EXPECT_TRUE(std::holds_alternative<std::monostate>(holder.engine.lastQuery));
    }
    EXPECT_TRUE(holder.session->Close().ok());
}

// Set/Increment 变更编译为单语句更新；无条件更新被安全拦截。
TEST(StorageContractTest, UpdateCompilesSetAndIncrementIntoSingleStatement)
{
    ContractSession holder;
    ASSERT_TRUE(holder.Open("update-session").ok());
    {
        auto tableResult = holder.session->Table<ContractRecord>();
        ASSERT_TRUE(tableResult.ok());
        auto table = std::move(*tableResult.value);

        const auto changed
            = table.Update(Eq(&ContractRecord::id, uint64_t{9}), Set(&ContractRecord::state, int64_t{4}));
        ASSERT_TRUE(changed.ok());
        EXPECT_EQ(*changed.value, 2U); // fake 引擎固定返回 2 行受影响
        ASSERT_TRUE(holder.engine.sawMutation);
        EXPECT_NE(std::get_if<ConditionClause>(&holder.engine.lastQuery), nullptr);
        ASSERT_EQ(holder.engine.lastUpdate.size(), 1U);
        EXPECT_EQ(holder.engine.lastUpdate.front().kind, ChangeKind::SET);

        const auto incremented
            = table.Update(Eq(&ContractRecord::id, uint64_t{9}), Increment(&ContractRecord::state, int64_t{2}));
        ASSERT_TRUE(incremented.ok());
        EXPECT_EQ(holder.engine.lastUpdate.front().kind, ChangeKind::INCREMENT);

        const auto multi = table.Update(
            Eq(&ContractRecord::id, uint64_t{9}), Set(&ContractRecord::state, int64_t{4}),
            Set(&ContractRecord::state, int64_t{5}));
        ASSERT_TRUE(multi.ok());
        EXPECT_EQ(holder.engine.lastUpdate.size(), 2U);

        EXPECT_EQ(
            table.Update(All<ContractRecord>(), Set(&ContractRecord::state, int64_t{4})).code,
            DbCode::INVALID_ARGUMENT);
        EXPECT_EQ(
            table.Update(Eq(&ContractRecord::id, uint64_t{9}), Set(&ContractRecord::id, uint64_t{4})).code,
            DbCode::INVALID_ARGUMENT);
    }
    EXPECT_TRUE(holder.session->Close().ok());
}

// 条件删除与全表清空分离：无条件删除被拒绝，显式清空保留 ClearMode。
TEST(StorageContractTest, DeleteRequiresConditionAndClearKeepsMode)
{
    ContractSession holder;
    ASSERT_TRUE(holder.Open("delete-session").ok());
    {
        auto tableResult = holder.session->Table<ContractRecord>();
        ASSERT_TRUE(tableResult.ok());
        auto table = std::move(*tableResult.value);

        const auto deleted = table.Delete(Eq(&ContractRecord::id, uint64_t{9}));
        ASSERT_TRUE(deleted.ok());
        EXPECT_EQ(*deleted.value, 1U);
        ASSERT_TRUE(holder.engine.sawDelete);
        EXPECT_NE(std::get_if<ConditionClause>(&holder.engine.lastQuery), nullptr);

        EXPECT_EQ(table.Delete(All<ContractRecord>()).code, DbCode::INVALID_ARGUMENT);

        const auto cleared = table.Clear(ClearMode::KEEP_SEQUENCE);
        ASSERT_TRUE(cleared.ok());
        EXPECT_EQ(holder.engine.lastClearMode, ClearMode::KEEP_SEQUENCE);
    }
    EXPECT_TRUE(holder.session->Close().ok());
}

// 通用事务入口不需要业务枚举：首个暂存 DML 惰性绑定物理域，跨域操作在执行前被拒绝。
TEST(StorageContractTest, GenericTransactionLazilyBindsAndRejectsForeignDomain)
{
    FakeStorageEngine runnerEngine;
    FakeStorageEngine opEngine;
    SessionStorageBinding binding;
    binding.sessionId = "generic-transaction-session";
    binding.dataNamespace = "generic";
    binding.databases.push_back({"CONTRACT_TEST_DB", "", &Ddl::kContractTestDatabase});
    binding.databases.push_back({"CONTRACT_TEST_OP_DB", "", &Ddl::kContractTestOpDatabase});
    StorageOpenRequest request;
    request.engines
        = {EngineRegistration{"CONTRACT_TEST_DB", kTestFakeBackendName, &runnerEngine},
           EngineRegistration{"CONTRACT_TEST_OP_DB", kTestFakeBackendName, &opEngine}};
    auto opened = StorageSession::Open(binding, request);
    ASSERT_TRUE(opened.ok());
    auto session = std::move(*opened.value);
    ASSERT_TRUE(ProcessStorageContext::Install(session).ok());

    {
        auto transaction = BeginTransaction();
        ASSERT_TRUE(transaction.ok());
        EXPECT_EQ(transaction->State(), TransactionState::BUILDING);

        {
            auto deviceTable = AcquireTable<ContractRecord>();
            ASSERT_TRUE(deviceTable.ok());
            ContractRecord record;
            EXPECT_TRUE(deviceTable->Insert(record, *transaction).ok());
            EXPECT_EQ(runnerEngine.insertCalls, 1);
            EXPECT_EQ(opEngine.insertCalls, 0);

            // 跨物理域 DML 在执行前被拒绝：第二张表属于另一个物理引擎
            auto opTable = AcquireTable<OpRecord>();
            ASSERT_TRUE(opTable.ok());
            OpRecord opRecord;
            EXPECT_EQ(opTable->Insert(opRecord, *transaction).code, DbCode::INVALID_ARGUMENT);
            EXPECT_EQ(opEngine.insertCalls, 0);
            // 拒绝不改变事务状态：已暂存的同域操作仍可正常提交
            EXPECT_EQ(transaction->State(), TransactionState::BUILDING);
        }

        const auto committed = transaction->Commit();
        ASSERT_TRUE(committed.ok());
        EXPECT_EQ(runnerEngine.commitCalls, 1);
        EXPECT_EQ(transaction->State(), TransactionState::COMMITTED);
    }

    // 跨域拒绝后显式回滚：只触碰已绑定的物理域，另一域引擎全程未被接触
    {
        auto transaction = BeginTransaction();
        ASSERT_TRUE(transaction.ok());
        auto deviceTable = AcquireTable<ContractRecord>();
        ASSERT_TRUE(deviceTable.ok());
        ContractRecord record;
        EXPECT_TRUE(deviceTable->Insert(record, *transaction).ok());
        EXPECT_EQ(runnerEngine.insertCalls, 2);
        auto opTable = AcquireTable<OpRecord>();
        ASSERT_TRUE(opTable.ok());
        OpRecord opRecord;
        EXPECT_EQ(opTable->Insert(opRecord, *transaction).code, DbCode::INVALID_ARGUMENT);
        EXPECT_TRUE(transaction->Rollback().ok());
        EXPECT_EQ(transaction->State(), TransactionState::ROLLED_BACK);
        EXPECT_EQ(runnerEngine.rollbackCalls, 1);
        EXPECT_EQ(opEngine.insertCalls, 0);
        EXPECT_EQ(opEngine.commitCalls, 0);
        EXPECT_EQ(opEngine.rollbackCalls, 0);
    }

    EXPECT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(session.Close().ok());
}

// 未提交事务析构自动回滚；空事务提交为平凡成功。
TEST(StorageContractTest, UncommittedTransactionRollsBackOnDestruction)
{
    ContractSession holder;
    ASSERT_TRUE(holder.Open("rollback-session").ok());
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());

    {
        auto transaction = BeginTransaction();
        ASSERT_TRUE(transaction.ok());
        auto table = AcquireTable<ContractRecord>();
        ASSERT_TRUE(table.ok());
        ContractRecord record;
        EXPECT_TRUE(table->Insert(record, *transaction).ok());
        EXPECT_EQ(holder.engine.beginCalls, 1);
    }
    EXPECT_EQ(holder.engine.rollbackCalls, 1);
    EXPECT_EQ(holder.engine.commitCalls, 0);

    {
        auto empty = BeginTransaction();
        ASSERT_TRUE(empty.ok());
        const auto committed = empty->Commit();
        ASSERT_TRUE(committed.ok());
        EXPECT_TRUE(committed.value->items.empty());
    }
    EXPECT_EQ(holder.engine.commitCalls, 0);

    // 空事务显式回滚同样为平凡成功：不触碰物理引擎，状态进入终态后不可重复提交
    {
        auto empty = BeginTransaction();
        ASSERT_TRUE(empty.ok());
        const auto rolled = empty->Rollback();
        ASSERT_TRUE(rolled.ok());
        EXPECT_EQ(empty->State(), TransactionState::ROLLED_BACK);
        EXPECT_EQ(empty->Commit().code, DbCode::INVALID_STATE);
        EXPECT_EQ(empty->Rollback().code, DbCode::INVALID_STATE);
    }
    EXPECT_EQ(holder.engine.rollbackCalls, 1);
    EXPECT_EQ(holder.engine.commitCalls, 0);

    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(holder.session->Close().ok());
}

// 组装一个只含动态 PID scope 表的最小可用会话（验证一次性入口的显式 scope 解析）。
struct DynamicScopeSession {
    FakeStorageEngine engine;
    StorageOpenRequest request;
    std::optional<StorageSession> session;

    DbResult<void> Open(const std::string& sessionId)
    {
        SessionStorageBinding binding;
        binding.sessionId = sessionId;
        binding.dataNamespace = "contract-dynamic";
        binding.databases.push_back({"CONTRACT_TEST_DYNAMIC_DB", "", &Ddl::kContractTestDynamicDatabase});
        request.engines = {EngineRegistration{"CONTRACT_TEST_DYNAMIC_DB", kTestFakeBackendName, &engine}};
        auto opened = StorageSession::Open(binding, request);
        if (!opened.ok()) {
            return {opened.code, opened.affectedRows, opened.diagnostic};
        }
        session.emplace(std::move(*opened.value));
        return {DbCode::OK, 0, {}};
    }
};

// 一次性原子入口：每次调用恰好一次取表 + 一次引擎操作，全部七类能力可用，
// OrderBy/Limit 子句与事务重载与 Table 句柄入口行为一致。
TEST(StorageContractTest, OneShotEntryPointsRunEachAtomicOperationOnce)
{
    ContractSession holder;
    ASSERT_TRUE(holder.Open("one-shot-session").ok());
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());
    {
        ContractRecord record;
        const auto inserted = Insert<ContractRecord>(record);
        ASSERT_TRUE(inserted.ok());
        EXPECT_EQ(record.id, 7U);
        EXPECT_EQ(holder.engine.insertCalls, 1);

        const auto found = Select<ContractRecord>(Eq(&ContractRecord::id, uint64_t{7}), Limit(1U));
        ASSERT_TRUE(found.ok());
        ASSERT_EQ(found.value->size(), 1U);
        EXPECT_EQ(found.value->front().id, 11U);

        const auto ordered = Select<ContractRecord>(
            Eq(&ContractRecord::state, int64_t{3}), OrderBy(&ContractRecord::id, SortDirection::DESC), Limit(1U));
        ASSERT_TRUE(ordered.ok());
        ASSERT_EQ(ordered.value->size(), 1U);
        ASSERT_FALSE(holder.engine.lastOptions.sorts.empty());

        const auto rows = Select<ContractRecord>(
            All<ContractRecord>(), OrderBy(&ContractRecord::id, SortDirection::ASC), Limit(10U));
        ASSERT_TRUE(rows.ok());
        ASSERT_EQ(rows.value->size(), 2U);

        const auto count = Count<ContractRecord>(All<ContractRecord>());
        ASSERT_TRUE(count.ok());
        EXPECT_EQ(*count.value, 5U);
        EXPECT_EQ(holder.engine.countCalls, 1);

        const auto updated
            = Update<ContractRecord>(Eq(&ContractRecord::id, uint64_t{7}), Set(&ContractRecord::state, int64_t{9}));
        ASSERT_TRUE(updated.ok());
        EXPECT_EQ(*updated.value, 2U); // fake 引擎对任意匹配条件固定影响 2 行

        const auto removed = Delete<ContractRecord>(Eq(&ContractRecord::id, uint64_t{7}));
        ASSERT_TRUE(removed.ok());

        const auto cleared = Clear<ContractRecord>(ClearMode::KEEP_SEQUENCE);
        ASSERT_TRUE(cleared.ok());

        auto transaction = BeginTransaction();
        ASSERT_TRUE(transaction.ok());
        const auto stagedInsert = Insert<ContractRecord>(record, *transaction);
        ASSERT_TRUE(stagedInsert.ok());
        const auto stagedUpdate = Update<ContractRecord>(
            Eq(&ContractRecord::id, uint64_t{7}), Set(&ContractRecord::state, int64_t{9}), *transaction);
        ASSERT_TRUE(stagedUpdate.ok());
        const auto stagedDelete = Delete<ContractRecord>(Eq(&ContractRecord::id, uint64_t{7}), *transaction);
        ASSERT_TRUE(stagedDelete.ok());
        // 同一事务的三条 DML 只开启一次物理事务（首条 DML 惰性 Begin）；
        // 各类 DML 计数 = 前序一次性即时调用 + 本事务内各一条
        EXPECT_EQ(holder.engine.beginCalls, 1);
        EXPECT_EQ(holder.engine.insertCalls, 2);
        EXPECT_EQ(holder.engine.updateCalls, 2);
        EXPECT_EQ(holder.engine.deleteCalls, 2);
        EXPECT_TRUE(transaction->Commit().ok());
        ;
    }
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(holder.session->Close().ok());
}

// 一次性直达合同（REQ-CAP-5）：经正式一次性 API 的 Table 移动/optional 路径后，
// guard 移交的 directEngine_ 必须仍然有效——七类一次性操作的 Dispatch 直达已解析
// 引擎（sessionControlExecutes 增量为 0：不回 SessionControlBlock::Execute 二次
// 取会话锁做路由查找），引擎仍各执行恰好一次；显式 Table 句柄与事务语义保持
// 原样（继续经会话控制块派发，sessionControlExecutes 按操作数递增）。
TEST(StorageContractTest, OneShotDispatchReachesEngineDirectlyWithoutSessionReroute)
{
    ContractSession holder;
    ASSERT_TRUE(holder.Open("one-shot-direct-session").ok());
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());
    {
        const int engineExecutesBefore = holder.engine.executeCalls;

        // 七类一次性操作：每个操作恰好一个 operation guard，且 Dispatch 不经过
        // SessionControlBlock::Execute（直达已解析引擎）。
        ContractRecord record;
        ASSERT_TRUE(Insert<ContractRecord>(record).ok());
        ASSERT_TRUE(SelectOne<ContractRecord>(Eq(&ContractRecord::id, uint64_t{7})).ok());
        ASSERT_TRUE(Select<ContractRecord>(All<ContractRecord>(), Limit(1U)).ok());
        ASSERT_TRUE(Count<ContractRecord>(All<ContractRecord>()).ok());
        ASSERT_TRUE(
            Update<ContractRecord>(Eq(&ContractRecord::id, uint64_t{7}), Set(&ContractRecord::state, int64_t{9})).ok());
        ASSERT_TRUE(Delete<ContractRecord>(Eq(&ContractRecord::id, uint64_t{7})).ok());
        ASSERT_TRUE(Clear<ContractRecord>(ClearMode::KEEP_SEQUENCE).ok());

        // 直达合同核心断言：一次性 Dispatch 不回会话控制块二次路由；
        // 引擎仍然每个操作恰好执行一次（直达只是省掉二次路由，不省执行）。
        EXPECT_EQ(holder.engine.executeCalls - engineExecutesBefore, 7);

        // 额外 move construction（允许的全部移动形态）：一次性取得的句柄再移动一次，
        // directEngine_ 仍须有效（移动后 Dispatch 仍直达引擎）。
        auto acquired = HcclSim::Storage::detail::AcquireOperationTable<ContractRecord>();
        ASSERT_TRUE(acquired.ok());
        Table<ContractRecord> movedHandle{std::move(*acquired.value)};
        const int engineExecutesBeforeMove = holder.engine.executeCalls;
        ASSERT_TRUE(movedHandle.Count(All<ContractRecord>()).ok());
        EXPECT_EQ(holder.engine.executeCalls - engineExecutesBeforeMove, 1);

        // 显式 Table 句柄语义保持原样：Dispatch 经会话控制块派发（按操作数递增）。
        {
            auto table = AcquireTable<ContractRecord>();
            ASSERT_TRUE(table.ok());
            ASSERT_TRUE(table->Count(All<ContractRecord>()).ok());
            ASSERT_TRUE(table->Count(All<ContractRecord>()).ok());
        }

        // 事务形态的一次性入口：DML 直达引擎，不经会话控制块 Execute。
        {
            auto transaction = BeginTransaction();
            ASSERT_TRUE(transaction.ok());
            ASSERT_TRUE(Insert<ContractRecord>(record, *transaction).ok());
            ASSERT_TRUE(transaction->Commit().ok());
        }
    }
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(holder.session->Close().ok());
}

// 一次性入口的取表失败原样传播：无进程会话时错误码、影响行数与诊断一一透传。
TEST(StorageContractTest, OneShotEntriesPropagateTableAcquireFailure)
{
    const auto count = Count<ContractRecord>(All<ContractRecord>());
    EXPECT_EQ(count.code, DbCode::INVALID_STATE);
    EXPECT_EQ(count.affectedRows, 0U);
    EXPECT_FALSE(count.value.has_value());
    EXPECT_EQ(count.diagnostic, "process storage session is not installed");

    ContractRecord record;
    const auto inserted = Insert<ContractRecord>(record);
    EXPECT_EQ(inserted.code, DbCode::INVALID_STATE);
    EXPECT_EQ(inserted.affectedRows, 0U);
    EXPECT_EQ(inserted.diagnostic, "process storage session is not installed");

    const auto updated
        = Update<ContractRecord>(Eq(&ContractRecord::id, uint64_t{7}), Set(&ContractRecord::tag, std::string("x")));
    EXPECT_EQ(updated.code, DbCode::INVALID_STATE);
    EXPECT_EQ(updated.affectedRows, 0U);
    EXPECT_FALSE(updated.value.has_value());

    const auto deleted = Delete<ContractRecord>(Eq(&ContractRecord::id, uint64_t{7}));
    EXPECT_EQ(deleted.code, DbCode::INVALID_STATE);

    const auto cleared = Clear<ContractRecord>(ClearMode::KEEP_SEQUENCE);
    EXPECT_EQ(cleared.code, DbCode::INVALID_STATE);
    EXPECT_EQ(cleared.diagnostic, "process storage session is not installed");

    const auto rows = Select<ContractRecord>(All<ContractRecord>());
    EXPECT_EQ(rows.code, DbCode::INVALID_STATE);
}

// 会话内取表失败同样原样传播：静态会话上以动态 scope 取静态表返回 NOT_FOUND，
// 事务形态的一次性更新在取表失败时按 DbResult<void> 透传错误。
TEST(StorageContractTest, OneShotEntriesPropagateScopeAndTableResolutionFailure)
{
    ContractSession holder;
    ASSERT_TRUE(holder.Open("one-shot-scope-failure-session").ok());
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());
    {
        const auto mismatched = Count<ContractRecord>(TableScope::Process(11), All<ContractRecord>());
        EXPECT_EQ(mismatched.code, DbCode::NOT_FOUND);
        EXPECT_EQ(mismatched.affectedRows, 0U);
        EXPECT_FALSE(mismatched.value.has_value());

        auto transaction = BeginTransaction();
        ASSERT_TRUE(transaction.ok());
        // OpRecord 属第二逻辑库（CONTRACT_TEST_OP_DB），本会话未配置该库路由：
        // 静态路由按编译期归属定位库，库路由缺失明确失败（表可用性由库路由链路保证）。
        const auto outside
            = Update<OpRecord>(Eq(&OpRecord::id, uint64_t{1}), Set(&OpRecord::id, uint64_t{2}), *transaction);
        EXPECT_EQ(outside.code, DbCode::BACKEND_UNAVAILABLE);
        EXPECT_EQ(outside.diagnostic, "table route has no registered engine");
        EXPECT_FALSE(outside.ok());
    }
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(holder.session->Close().ok());
}

// 一次性入口的 Table 操作失败原样返回：非法条件与非法更新一律透传；
// 单行未命中不再是错误：Select + Limit(1) 返回 OK + 空行集。
TEST(StorageContractTest, OneShotEntriesPropagateOperationFailure)
{
    ContractSession holder;
    ASSERT_TRUE(holder.Open("one-shot-operation-failure-session").ok());
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());
    {
        holder.engine.selectEmpty = true;
        const auto missing = Select<ContractRecord>(Eq(&ContractRecord::id, uint64_t{404}), Limit(1U));
        EXPECT_EQ(missing.code, DbCode::OK);
        EXPECT_EQ(missing.affectedRows, 0U);
        ASSERT_TRUE(missing.value.has_value());
        EXPECT_TRUE(missing.value->empty());

        const auto unconditioned
            = Update<ContractRecord>(All<ContractRecord>(), Set(&ContractRecord::tag, std::string("x")));
        EXPECT_EQ(unconditioned.code, DbCode::INVALID_ARGUMENT);
        EXPECT_FALSE(unconditioned.value.has_value());

        const auto unconditionedDelete = Delete<ContractRecord>(All<ContractRecord>());
        EXPECT_EQ(unconditionedDelete.code, DbCode::INVALID_ARGUMENT);

        // 比较能力由字段类型与通用合同决定：范围比较是合法查询，无每表白名单拒绝。
        const auto ranged = Select<ContractRecord>(Lt(&ContractRecord::state, int64_t{1}));
        EXPECT_EQ(ranged.code, DbCode::OK);
    }
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(holder.session->Close().ok());
}

// 一次性入口接受显式动态 scope：按 PID 解析动态表，静态 scope 解析同一表失败。
TEST(StorageContractTest, OneShotEntriesAcceptExplicitDynamicScope)
{
    DynamicScopeSession holder;
    ASSERT_TRUE(holder.Open("one-shot-dynamic-scope-session").ok());
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());
    {
        holder.engine.countResult = 42;
        const auto count = Count<DynamicRecord>(TableScope::Process(11), All<DynamicRecord>());
        ASSERT_TRUE(count.ok());
        EXPECT_EQ(*count.value, 42U);
        EXPECT_EQ(holder.engine.countCalls, 1);

        DynamicRecord record;
        const auto inserted = Insert<DynamicRecord>(TableScope::Process(11), record);
        ASSERT_TRUE(inserted.ok());

        const auto staticScope = Count<DynamicRecord>(TableScope::Static(), All<DynamicRecord>());
        EXPECT_EQ(staticScope.code, DbCode::NOT_FOUND);
        EXPECT_FALSE(staticScope.value.has_value());
    }
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(holder.session->Close().ok());
}

// value_or 只做显式默认值访问：成功返回载荷，失败或无载荷返回调用方默认值；
// 箭头访问合同不受影响。
TEST(StorageContractTest, DbResultValueOrReturnsPayloadOrDefault)
{
    static_assert(std::is_same_v<decltype(std::declval<DbResult<int>&>().value_or(0)), int>);
    static_assert(std::is_same_v<decltype(std::declval<DbResult<int>&&>().value_or(0)), int>);

    DbResult<int> succeeded{DbCode::OK, 7, 3, "value"};
    EXPECT_EQ(succeeded.value_or(-1), 7);
    EXPECT_EQ(*succeeded, 7);
    EXPECT_EQ(succeeded.operator->() != nullptr, true);
    DbResult<int> implicitPayload{DbCode::OK, 7};
    EXPECT_EQ(implicitPayload.value_or(-1), 7);

    const DbResult<int> missing{DbCode::NOT_FOUND, std::nullopt, 0, "missing"};
    EXPECT_EQ(missing.value_or(-1), -1);
    EXPECT_EQ(missing.code, DbCode::NOT_FOUND);

    // OK 但无载荷同样返回默认值（与 std::optional::value_or 的判据一致）。
    const DbResult<int> empty{DbCode::OK};
    EXPECT_EQ(empty.value_or(9), 9);

    const DbResult<std::string> kept{DbCode::OK, std::string("payload")};
    EXPECT_EQ(std::move(kept).value_or(std::string("fallback")), "payload");
    const DbResult<std::string> dropped{DbCode::IO_ERROR, std::nullopt, 0, "io"};
    EXPECT_EQ(std::move(dropped).value_or(std::string("fallback")), "fallback");

    // 失败结果上调用 value_or 不终止进程，也不改变失败码本身。
    EXPECT_EQ(missing.ok(), false);
    EXPECT_EQ(missing.value_or(-1), -1);
}

// 失败结果的载荷不得经 value_or 泄漏：判据是 ok() 且有载荷，不是仅有载荷；
// 左值复制、右值移动与 move-only 形态同受该合同约束，调用 value_or 也不改变原失败码。
TEST(StorageContractTest, DbResultValueOrFallsBackWhenFailedResultCarriesPayload)
{
    // OK + payload：返回载荷。
    const DbResult<int> succeeded{DbCode::OK, 7, 3, "value"};
    EXPECT_EQ(succeeded.value_or(-1), 7);

    // OK + no payload：返回默认值。
    const DbResult<int> empty{DbCode::OK};
    EXPECT_EQ(empty.value_or(9), 9);

    // error + no payload：返回默认值。
    const DbResult<int> failedWithoutPayload{DbCode::NOT_FOUND, std::nullopt, 0, "missing"};
    EXPECT_EQ(failedWithoutPayload.value_or(-1), -1);

    // error + payload：必须返回默认值，不得返回载荷。
    const DbResult<int> failedWithPayload{DbCode::IO_ERROR, 7, 0, "partial"};
    EXPECT_EQ(failedWithPayload.value_or(-1), -1);

    // 右值 string 形态（非 const 对象命中移动重载）：error + payload 同样返回默认值。
    DbResult<std::string> failedString{DbCode::IO_ERROR, std::string("partial"), 0, "io"};
    EXPECT_EQ(std::move(failedString).value_or(std::string("fallback")), "fallback");

    // 右值 move-only 形态：error + payload 同样返回默认值。
    DbResult<std::unique_ptr<int>> failedMoveOnly{DbCode::IO_ERROR, std::make_unique<int>(7), 0, "partial"};
    auto moveOnlyFallback = std::move(failedMoveOnly).value_or(std::make_unique<int>(-1));
    ASSERT_TRUE(moveOnlyFallback != nullptr);
    EXPECT_EQ(*moveOnlyFallback, -1);

    // 调用 value_or 不改变原 DbCode。
    EXPECT_EQ(failedWithPayload.code, DbCode::IO_ERROR);
    EXPECT_EQ(failedWithPayload.ok(), false);
    EXPECT_EQ(failedWithPayload.value_or(-1), -1);
}

// ==== 进程 Session Registry（AcquireReady 惰性 ready-session 合同） ====

// 计数测试 bootstrap：安装一个进程生命周期内存活的静态 Fake 会话（RemoveAndDrain
// 后可重新安装）；调用计数供单飞/复用/不复活断言。
int& BootstrapRegistryCalls()
{
    static int calls = 0;
    return calls;
}

ContractSession& BootstrapContractHolder()
{
    static ContractSession holder;
    return holder;
}

DbResult<void> BootstrapRegistryContractSession()
{
    static bool opened = false;
    if (!opened) {
        auto openedResult = BootstrapContractHolder().Open("process-registry-bootstrap-session");
        if (!openedResult.ok()) {
            return openedResult;
        }
        opened = true;
    }
    ++BootstrapRegistryCalls();
    return ProcessStorageContext::Install(*BootstrapContractHolder().session);
}

// 单飞门闩测试 bootstrap：首次进入后保持 bootstrap 在途，直到主线程放行——
// 保证第二个线程确实处于等待窗口（无单飞实现会在该窗口内重复执行 bootstrap）。
struct ColdBootstrapGate {
    std::atomic<int> calls{0};
    std::atomic<bool> release{false};
};

ColdBootstrapGate& ColdGate()
{
    static ColdBootstrapGate gate;
    return gate;
}

DbResult<void> GatedBootstrapContractSession()
{
    ++ColdGate().calls;
    while (!ColdGate().release.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return BootstrapRegistryContractSession();
}

DbResult<void> GatedFailingBootstrapContractSession()
{
    ++ColdGate().calls;
    while (!ColdGate().release.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return {DbCode::BACKEND_UNAVAILABLE, 0, "contract bootstrap deliberately fails"};
}

// ==== RegisterBootstrap 唯一注册合同：单一 dispatcher 测试形态 ====
// 生产注册表在整个测试进程内只登记一个 ContractBootstrapDispatcher 函数指针；
// 需要不同 bootstrap 行为的用例经测试侧原子模式选择既有行为（含下文后置定义的
// 两轮失败/抛异常/OK 不安装三种），不再向生产注册表写入不同函数——注册语义由
// RegisterBootstrapRejectsDifferentFunctionAndKeepsOriginal 合同测试固定：
// 同指针幂等、不同指针 INVALID_STATE、被拒绝后原注册保持不变。
DbResult<void> TwoRoundFailingThenSucceedingBootstrap();
DbResult<void> GatedThrowingBootstrapContractSession();
DbResult<void> OkWithoutInstallBootstrapContractSession();

// 测试侧 dispatcher 模式（原子：用例先稳定模式再发起 bootstrap 调用）。
enum class ContractBootstrapMode : int {
    NormalInstall,                 // 正常安装（BootstrapRegistryContractSession）
    GatedSuccess,                  // 门闩保持后成功
    GatedFailure,                  // 门闩保持后失败
    TwoRoundFailingThenSucceeding, // 第一轮失败、第二轮成功
    Throwing,                      // 抛异常
    OkWithoutInstall,              // 返回 OK 但不 Install
};

std::atomic<ContractBootstrapMode>& ContractBootstrapModeRef()
{
    static std::atomic<ContractBootstrapMode> mode{ContractBootstrapMode::NormalInstall};
    return mode;
}

// 唯一登记进生产注册表的 bootstrap：按测试侧模式分发到既有行为。
DbResult<void> ContractBootstrapDispatcher()
{
    switch (ContractBootstrapModeRef().load()) {
        case ContractBootstrapMode::GatedSuccess:
            return GatedBootstrapContractSession();
        case ContractBootstrapMode::GatedFailure:
            return GatedFailingBootstrapContractSession();
        case ContractBootstrapMode::TwoRoundFailingThenSucceeding:
            return TwoRoundFailingThenSucceedingBootstrap();
        case ContractBootstrapMode::Throwing:
            return GatedThrowingBootstrapContractSession();
        case ContractBootstrapMode::OkWithoutInstall:
            return OkWithoutInstallBootstrapContractSession();
        case ContractBootstrapMode::NormalInstall:
            break;
    }
    return BootstrapRegistryContractSession();
}

// 合同测试的"不同函数"样本：若被 last-write-wins 静默覆盖安装，AcquireReady 将
// 执行本函数并返回可区分失败（StrayBootstrapCalls 供"从未执行"断言）。
int& StrayBootstrapCalls()
{
    static int calls = 0;
    return calls;
}

DbResult<void> StrayBootstrapMustNeverRun()
{
    ++StrayBootstrapCalls();
    return {DbCode::BACKEND_UNAVAILABLE, 0, "stray bootstrap must never run"};
}

// 冷进程首次原子访问：进程未安装会话时（真实冷状态以 ResetAfterForkInChild 取得），
// 唯一 ready-session 入口应经已注册的应用级 bootstrap 惰性建立本进程会话——等价
// 迁移前 RunnerDB/OpDbOps 惰性 singleton 的首次建连时机；旧实现在此直接以
// INVALID_STATE（process storage session is not installed）拒绝（RED 证据）。
TEST(StorageContractTest, ColdAtomicEntryBootstrapsProcessSessionOnFirstUse)
{
    ProcessStorageContext::ResetAfterForkInChild();
    EXPECT_EQ(ProcessStorageContext::Acquire().code, DbCode::INVALID_STATE);
    BootstrapRegistryCalls() = 0;
    ContractBootstrapModeRef().store(ContractBootstrapMode::NormalInstall);
    ASSERT_TRUE(ProcessStorageContext::RegisterBootstrap(&ContractBootstrapDispatcher).ok());
    const void* sessionOfFirstUse = nullptr;
    {
        auto table = AcquireTable<ContractRecord>();
        ASSERT_TRUE(table.ok()) << table.diagnostic;
        ContractRecord record;
        EXPECT_TRUE(table->Insert(record).ok());
        auto lease = ProcessStorageContext::Acquire();
        ASSERT_TRUE(lease.ok());
        sessionOfFirstUse = static_cast<const void*>(&lease.value->Get());
    }
    EXPECT_EQ(BootstrapRegistryCalls(), 1);

    // 重复原子调用复用同一会话与连接：不再触发 bootstrap、不重开引擎。
    for (int i = 0; i < 3; ++i) {
        const auto count = Count<ContractRecord>(All<ContractRecord>());
        EXPECT_TRUE(count.ok());
    }
    EXPECT_EQ(BootstrapRegistryCalls(), 1);
    EXPECT_EQ(BootstrapContractHolder().engine.openCalls, 1);
    {
        auto lease = ProcessStorageContext::Acquire();
        ASSERT_TRUE(lease.ok());
        EXPECT_EQ(static_cast<const void*>(&lease.value->Get()), sessionOfFirstUse);
    }
    EXPECT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
}

// 并发首次访问 single-flight：两个线程同时冷调用时只允许一个线程执行 bootstrap，
// 另一个等待其完成；两者取得同一会话且均成功，无重复 bootstrap、无死锁。
TEST(StorageContractTest, AcquireReadySingleFlightBootstrapsExactlyOnce)
{
    ProcessStorageContext::ResetAfterForkInChild();
    ColdGate().calls.store(0);
    ColdGate().release.store(false);
    ContractBootstrapModeRef().store(ContractBootstrapMode::GatedSuccess);
    ASSERT_TRUE(ProcessStorageContext::RegisterBootstrap(&ContractBootstrapDispatcher).ok());

    std::atomic<int> entered{0};
    std::mutex resultsMutex;
    std::vector<DbResult<uint64_t>> results;
    auto worker = [&entered, &resultsMutex, &results]() {
        ++entered;
        auto counted = Count<ContractRecord>(All<ContractRecord>());
        std::lock_guard<std::mutex> lock(resultsMutex);
        results.push_back(std::move(counted));
    };
    std::thread first(worker);
    std::thread second(worker);
    // 等待两个线程都到达调用边界且引导线程已进入 bootstrap。
    while (entered.load() < 2 || ColdGate().calls.load() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // 保持 bootstrap 在途，给第二个线程留出进入等待窗口的时间。
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ColdGate().release.store(true);
    first.join();
    second.join();
    ASSERT_EQ(results.size(), 2U);
    for (const auto& counted : results) {
        EXPECT_TRUE(counted.ok()) << counted.diagnostic;
    }
    EXPECT_EQ(ColdGate().calls.load(), 1);
    EXPECT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
}

// bootstrap 失败：并发等待线程与引导线程共享同一可诊断错误（同一 code 与诊断）；
// 本轮只执行一次 bootstrap；后续新调用允许重试（不把瞬时失败固化为永久拒绝）。
TEST(StorageContractTest, AcquireReadyPropagatesBootstrapFailureToConcurrentWaiters)
{
    ProcessStorageContext::ResetAfterForkInChild();
    BootstrapRegistryCalls() = 0;
    ColdGate().calls.store(0);
    ColdGate().release.store(false);
    ContractBootstrapModeRef().store(ContractBootstrapMode::GatedFailure);
    ASSERT_TRUE(ProcessStorageContext::RegisterBootstrap(&ContractBootstrapDispatcher).ok());

    std::atomic<int> entered{0};
    std::mutex resultsMutex;
    std::vector<DbResult<uint64_t>> results;
    auto worker = [&entered, &resultsMutex, &results]() {
        ++entered;
        auto counted = Count<ContractRecord>(All<ContractRecord>());
        std::lock_guard<std::mutex> lock(resultsMutex);
        results.push_back(std::move(counted));
    };
    std::thread first(worker);
    std::thread second(worker);
    while (entered.load() < 2 || ColdGate().calls.load() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ColdGate().release.store(true);
    first.join();
    second.join();
    ASSERT_EQ(results.size(), 2U);
    for (const auto& counted : results) {
        EXPECT_FALSE(counted.ok());
        EXPECT_EQ(counted.code, DbCode::BACKEND_UNAVAILABLE);
        EXPECT_EQ(counted.diagnostic, "contract bootstrap deliberately fails");
    }
    EXPECT_EQ(ColdGate().calls.load(), 1);
    // 后续新调用重试 bootstrap（再次失败返回同一可诊断错误，不固化永久拒绝）。
    const auto retried = Count<ContractRecord>(All<ContractRecord>());
    EXPECT_EQ(retried.code, DbCode::BACKEND_UNAVAILABLE);
    EXPECT_EQ(ColdGate().calls.load(), 2);
}

// 显式移除后不自动复活：RemoveAndDrain 后原子入口按无效状态拒绝，bootstrap 不再执行。
TEST(StorageContractTest, ExplicitClosePreventsAutomaticRevival)
{
    ProcessStorageContext::ResetAfterForkInChild();
    BootstrapRegistryCalls() = 0;
    ContractBootstrapModeRef().store(ContractBootstrapMode::NormalInstall);
    ASSERT_TRUE(ProcessStorageContext::RegisterBootstrap(&ContractBootstrapDispatcher).ok());
    ASSERT_TRUE(Count<ContractRecord>(All<ContractRecord>()).ok());
    EXPECT_EQ(BootstrapRegistryCalls(), 1);
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_EQ(Count<ContractRecord>(All<ContractRecord>()).code, DbCode::INVALID_STATE);
    EXPECT_EQ(BootstrapRegistryCalls(), 1);
}

// RegisterBootstrap 唯一注册合同：注册表为空时安装传入函数并返回 OK；再次注册
// 同一函数幂等成功；注册另一个不同函数必须明确拒绝（INVALID_STATE）且原注册
// 保持不变——AcquireReady 仍执行原 dispatcher 的正常安装路径，陌生函数从未执行。
// 旧 last-write-wins 实现：不同函数被静默覆盖安装且返回 OK，AcquireReady 执行
// 陌生函数（RED 证据）。
TEST(StorageContractTest, RegisterBootstrapRejectsDifferentFunctionAndKeepsOriginal)
{
    ProcessStorageContext::ResetAfterForkInChild();
    BootstrapRegistryCalls() = 0;
    StrayBootstrapCalls() = 0;
    ContractBootstrapModeRef().store(ContractBootstrapMode::NormalInstall);

    // 注册 dispatcher（注册表为空时安装，已注册时同指针幂等）：一律 OK。
    ASSERT_TRUE(ProcessStorageContext::RegisterBootstrap(&ContractBootstrapDispatcher).ok());
    // 再次注册同一 dispatcher：幂等成功。
    ASSERT_TRUE(ProcessStorageContext::RegisterBootstrap(&ContractBootstrapDispatcher).ok());

    // 注册另一个不同函数：必须明确拒绝，不得静默覆盖。
    EXPECT_EQ(ProcessStorageContext::RegisterBootstrap(&StrayBootstrapMustNeverRun).code, DbCode::INVALID_STATE);

    // 被拒绝的不同函数注册不得改变原注册：AcquireReady 仍执行原 dispatcher，
    // 而不是被替换成陌生函数（租约作用域内完成断言，归还后再排空）。
    {
        auto lease = ProcessStorageContext::AcquireReady();
        ASSERT_TRUE(lease.ok()) << lease.diagnostic;
        EXPECT_EQ(BootstrapRegistryCalls(), 1);
        EXPECT_EQ(StrayBootstrapCalls(), 0);
    }
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
}

// ==== 单值条件表示（REQ-CAP-6） ====

// 单值谓词内联持有 DbScalar（不为 values 分配 vector）；In 才拥有多值容器；
// All 无值。语义（条件树/括号/绑定顺序）由既有表达式测试覆盖。
TEST(StorageContractTest, SingleValuePredicatesHoldInlineScalarNotVector)
{
    // 单谓词内联一个 ConditionClause（variant 直接持有子句，不分配 vector）。
    const auto eq = Eq(&ContractRecord::id, uint64_t{7});
    const auto* eqClause = std::get_if<ConditionClause>(&eq.data);
    ASSERT_NE(eqClause, nullptr);
    EXPECT_TRUE(std::holds_alternative<DbScalar>(eqClause->value));
    EXPECT_FALSE(std::holds_alternative<std::vector<DbScalar>>(eqClause->value));
    const auto* eqNumber = std::get_if<uint64_t>(&std::get<DbScalar>(eqClause->value));
    ASSERT_NE(eqNumber, nullptr);
    EXPECT_EQ(*eqNumber, 7U);

    const auto starts = StartsWith(&ContractRecord::label, "abc");
    const auto* startsClause = std::get_if<ConditionClause>(&starts.data);
    ASSERT_NE(startsClause, nullptr);
    EXPECT_TRUE(std::holds_alternative<DbScalar>(startsClause->value));

    const auto any = All<ContractRecord>();
    EXPECT_TRUE(std::holds_alternative<std::monostate>(any.data));

    const auto in = In(&ContractRecord::id, std::vector<uint64_t>{1, 2, 3});
    const auto* inClause = std::get_if<ConditionClause>(&in.data);
    ASSERT_NE(inClause, nullptr);
    const auto* inValues = std::get_if<std::vector<DbScalar>>(&inClause->value);
    ASSERT_NE(inValues, nullptr);
    EXPECT_EQ(inValues->size(), 3U);

    const auto emptyIn = In(&ContractRecord::id, std::vector<uint64_t>{});
    const auto* emptyClause = std::get_if<ConditionClause>(&emptyIn.data);
    ASSERT_NE(emptyClause, nullptr);
    // 空 In 恒假：叶子保留 IN 比较与空多值（引擎渲染为常量假表达式）。
    EXPECT_EQ(emptyClause->compare, Compare::IN);
    EXPECT_TRUE(std::get<std::vector<DbScalar>>(emptyClause->value).empty());
}

// ==== 逻辑库 Engine/连接惰性创建（REQ-CAP-2） ====

// 门闩化 fake 引擎：Open 进入后在门闩上保持，用于构造并发首访 single-flight 窗口。
class GatedOpenEngine final : public FakeStorageEngine {
public:
    std::atomic<int> openAttempts{0};
    std::atomic<bool> releaseOpen{false};
    std::atomic<bool> failOpen{false};

    DbResult<void> Open(const EngineOpenRequest& request) override
    {
        ++openAttempts;
        while (!releaseOpen.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (failOpen.load()) {
            return {DbCode::BACKEND_UNAVAILABLE, 0, "engine open deliberately fails"};
        }
        return FakeStorageEngine::Open(request);
    }
};

// 两逻辑库会话装配（CONTRACT_TEST_DB + CONTRACT_TEST_OP_DB）。
struct LazyEngineSession {
    GatedOpenEngine runnerEngine;
    GatedOpenEngine opEngine;
    std::optional<StorageSession> session;

    void Open(const std::string& sessionId)
    {
        SessionStorageBinding binding;
        binding.sessionId = sessionId;
        binding.dataNamespace = "lazy-engine";
        binding.databases.push_back({"CONTRACT_TEST_DB", "", &Ddl::kContractTestDatabase});
        binding.databases.push_back({"CONTRACT_TEST_OP_DB", "", &Ddl::kContractTestOpDatabase});
        StorageOpenRequest request;
        request.engines
            = {EngineRegistration{"CONTRACT_TEST_DB", kTestFakeBackendName, &runnerEngine},
               EngineRegistration{"CONTRACT_TEST_OP_DB", kTestFakeBackendName, &opEngine}};
        auto opened = StorageSession::Open(binding, request);
        ASSERT_TRUE(opened.ok());
        session.emplace(std::move(*opened.value));
    }
};

// 会话安装完成时引擎/连接数为 0；每逻辑库首次真实访问才打开自己的引擎。
TEST(StorageContractTest, SessionInstallKeepsEnginesClosedUntilFirstUse)
{
    LazyEngineSession holder;
    holder.Open("lazy-engine-session");
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());
    EXPECT_EQ(holder.runnerEngine.openAttempts.load(), 0);
    EXPECT_EQ(holder.opEngine.openAttempts.load(), 0);

    holder.runnerEngine.releaseOpen.store(true);
    const auto counted = Count<ContractRecord>(All<ContractRecord>());
    EXPECT_TRUE(counted.ok());
    EXPECT_EQ(holder.runnerEngine.openAttempts.load(), 1);
    EXPECT_EQ(holder.opEngine.openAttempts.load(), 0);

    EXPECT_TRUE(Count<ContractRecord>(All<ContractRecord>()).ok());
    EXPECT_EQ(holder.runnerEngine.openAttempts.load(), 1);
    EXPECT_EQ(holder.opEngine.openAttempts.load(), 0);

    holder.opEngine.releaseOpen.store(true);
    EXPECT_TRUE(Count<OpRecord>(All<OpRecord>()).ok());
    EXPECT_EQ(holder.runnerEngine.openAttempts.load(), 1);
    EXPECT_EQ(holder.opEngine.openAttempts.load(), 1);

    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(holder.session->Close().ok());
}

// 并发首次访问同一逻辑库 single-flight：只有一个线程执行引擎 Open，
// 其余等待同一轮结果；另一逻辑库引擎保持关闭。
TEST(StorageContractTest, ConcurrentFirstAccessOpensDatabaseEngineExactlyOnce)
{
    LazyEngineSession holder;
    holder.Open("lazy-engine-concurrent");
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());

    constexpr int kWorkers = 8;
    std::atomic<int> entered{0};
    std::vector<DbResult<uint64_t>> results(kWorkers);
    std::vector<std::thread> workers;
    for (int i = 0; i < kWorkers; ++i) {
        workers.emplace_back([&entered, &results, i]() {
            ++entered;
            results[i] = Count<ContractRecord>(All<ContractRecord>());
        });
    }
    while (entered.load() < kWorkers || holder.runnerEngine.openAttempts.load() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    holder.runnerEngine.releaseOpen.store(true);
    for (auto& worker : workers) {
        worker.join();
    }
    for (const auto& counted : results) {
        EXPECT_TRUE(counted.ok());
    }
    EXPECT_EQ(holder.runnerEngine.openAttempts.load(), 1);
    EXPECT_EQ(holder.opEngine.openAttempts.load(), 0);
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(holder.session->Close().ok());
}

// 引擎 Open 失败：同一轮并发等待者共享同一错误；后续新调用允许重试（再次尝试打开）。
TEST(StorageContractTest, EngineOpenFailureSharedByWaitersAndRetryable)
{
    LazyEngineSession holder;
    holder.Open("lazy-engine-open-failure");
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());
    // 门闩保持：首个访问线程进入 Open 后被门闩挂住，其余并发线程登记为同轮等待者。
    holder.runnerEngine.failOpen.store(true);

    constexpr int kWaiters = 4;
    std::atomic<int> entered{0};
    std::vector<DbResult<uint64_t>> results(kWaiters);
    std::vector<std::thread> workers;
    for (int i = 0; i < kWaiters; ++i) {
        workers.emplace_back([&entered, &results, i]() {
            ++entered;
            results[i] = Count<ContractRecord>(All<ContractRecord>());
        });
    }
    while (entered.load() < kWaiters || holder.runnerEngine.openAttempts.load() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    holder.runnerEngine.releaseOpen.store(true);
    for (auto& worker : workers) {
        worker.join();
    }
    for (const auto& counted : results) {
        EXPECT_EQ(counted.code, DbCode::BACKEND_UNAVAILABLE);
        EXPECT_EQ(counted.diagnostic, "engine open deliberately fails");
    }
    EXPECT_EQ(holder.runnerEngine.openAttempts.load(), 1);

    // 后续新调用重试：再次执行打开（再次失败可诊断，不固化永久拒绝）。
    const auto retried = Count<ContractRecord>(All<ContractRecord>());
    EXPECT_EQ(retried.code, DbCode::BACKEND_UNAVAILABLE);
    EXPECT_EQ(holder.runnerEngine.openAttempts.load(), 2);
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(holder.session->Close().ok());
}

// ==== 库引擎惰性打开轮次隔离（失败轮次不被下一轮重试污染） ====

// 轮次隔离测试引擎：第 1 次 Open 在门闩上保持后返回明确的 BACKEND_UNAVAILABLE
// （诊断 A）；第 2 次起立即成功（无门闩）——重试调用者在旧等待者消费链中间
// 即可完成第二轮，构造轮次污染窗口。
class RoundGatedOpenEngine final : public FakeStorageEngine {
public:
    std::atomic<int> openAttempts{0};
    std::atomic<bool> releaseFirst{false};

    DbResult<void> Open(const EngineOpenRequest& request) override
    {
        const int attempt = ++openAttempts;
        if (attempt == 1) {
            while (!releaseFirst.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return {DbCode::BACKEND_UNAVAILABLE, 0, "round-one engine open failure"};
        }
        return FakeStorageEngine::Open(request);
    }
};

// 打开失败轮次隔离：第一轮打开失败并唤醒等待者后，新调用者在旧等待者读取结果前
// 发起第二轮并成功；第一轮的全部参与者（打开者 + 固定等待者 + 重试调用者的首次
// 调用）必须精确返回第一轮错误（BACKEND_UNAVAILABLE + 诊断 A），不被第二轮成功
// 污染；每轮 Open 精确执行一次、成功的第二轮 Prepare 精确一次；重试调用者可以
// 重试并最终成功；无死锁。
TEST(StorageContractTest, EngineOpenFailedRoundWaitersNotAbsorbedByNextOpenRound)
{
    RoundGatedOpenEngine engine;
    std::optional<StorageSession> session;
    {
        SessionStorageBinding binding;
        binding.sessionId = "round-isolation-session";
        binding.dataNamespace = "round-isolation";
        binding.databases.push_back({"CONTRACT_TEST_DB", "", &Ddl::kContractTestDatabase});
        StorageOpenRequest request;
        request.engines = {EngineRegistration{"CONTRACT_TEST_DB", kTestFakeBackendName, &engine}};
        auto opened = StorageSession::Open(binding, request);
        ASSERT_TRUE(opened.ok());
        session.emplace(std::move(*opened.value));
    }
    ASSERT_TRUE(ProcessStorageContext::Install(*session).ok());

    constexpr int kWaiters = 64;
    constexpr int kRetryCallers = 3;
    std::atomic<int> participantsEntered{0};
    std::mutex resultsMutex;
    // 第一轮参与者（打开者 + 固定等待者 + 重试调用者首次调用）的返回码与诊断。
    std::vector<std::pair<DbCode, std::string>> firstRoundResults;
    auto recordFirstRound = [&resultsMutex, &firstRoundResults](const DbResult<uint64_t>& counted) {
        std::lock_guard<std::mutex> lock(resultsMutex);
        firstRoundResults.emplace_back(counted.code, counted.diagnostic);
    };

    // 第一轮打开者。
    std::atomic<bool> leaderReturned{false};
    std::thread leader([&recordFirstRound, &leaderReturned]() {
        recordFirstRound(Count<ContractRecord>(All<ContractRecord>()));
        leaderReturned.store(true);
    });
    while (engine.openAttempts.load() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // 固定等待者：第一轮在途时进入等待。
    std::vector<std::thread> waiters;
    for (int i = 0; i < kWaiters; ++i) {
        waiters.emplace_back([&participantsEntered, &recordFirstRound]() {
            ++participantsEntered;
            recordFirstRound(Count<ContractRecord>(All<ContractRecord>()));
        });
    }
    // 失败后立即重试的调用者：首次调用同样进入第一轮等待；消费到第一轮失败后在
    // 同一 CPU 上紧邻重试（相对 futex 唤醒延迟有抢锁优势），确定性地在剩余等待者
    // 的消费链中间发起并完成第二轮——用于验证旧等待者的结果不被下一轮污染。
    std::atomic<int> retrySucceeded{0};
    std::vector<std::thread> retryCallers;
    for (int i = 0; i < kRetryCallers; ++i) {
        retryCallers.emplace_back([&participantsEntered, &recordFirstRound, &retrySucceeded]() {
            ++participantsEntered;
            recordFirstRound(Count<ContractRecord>(All<ContractRecord>())); // 首次调用进入第一轮
            for (int attempt = 0; attempt < 500; ++attempt) {
                auto counted = Count<ContractRecord>(All<ContractRecord>());
                if (counted.ok()) {
                    ++retrySucceeded;
                    return;
                }
            }
        });
    }
    while (participantsEntered.load() < kWaiters + kRetryCallers) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // 第一轮结束（失败）。
    engine.releaseFirst.store(true);
    while (!leaderReturned.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // 等待第二轮真实开始（某个重试调用者发起新一轮打开）。
    const auto secondRoundDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (engine.openAttempts.load() < 2 && std::chrono::steady_clock::now() < secondRoundDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(engine.openAttempts.load(), 2) << "a retry caller should start the second open round";

    leader.join();
    for (auto& waiter : waiters) {
        waiter.join();
    }
    for (auto& caller : retryCallers) {
        caller.join();
    }

    // 第一轮全部固定参与者（打开者 + 固定等待者 + 重试调用者的首次调用）精确返回
    // 第一轮错误，不被第二轮成功污染。
    ASSERT_EQ(firstRoundResults.size(), static_cast<std::size_t>(kWaiters + kRetryCallers + 1));
    for (const auto& result : firstRoundResults) {
        EXPECT_EQ(result.first, DbCode::BACKEND_UNAVAILABLE);
        EXPECT_EQ(result.second, "round-one engine open failure");
    }
    // 重试调用者最终观察第二轮结果（成功）。
    EXPECT_EQ(retrySucceeded.load(), kRetryCallers);
    // 每轮 Open 精确执行一次（失败轮与重试轮各一次）；成功轮 Prepare 精确一次。
    EXPECT_EQ(engine.openAttempts.load(), 2);
    EXPECT_EQ(engine.prepareCalls, 1);
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(session->Close().ok());
}

// 事务路径的惰性打开合同（真实合同已选定并记录）：表句柄获取就是首次数据库访问
// ——AcquireTable 经会话 single-flight 打开本库引擎（RED 探针已证明：句柄获取后
// openAttempts 即为 1，不是首个 Stage 才打开）；BeginTransaction 只登记事务在途
// 计数、不打开任何引擎；首个 Stage 仅解析事务物理域（表句柄已保证引擎打开）；
// 另一逻辑库引擎保持关闭。
TEST(StorageContractTest, TableHandleAcquisitionOpensItsDatabaseEngine)
{
    LazyEngineSession holder;
    holder.Open("lazy-engine-transaction");
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());
    holder.runnerEngine.releaseOpen.store(true);
    holder.opEngine.releaseOpen.store(true);

    {
        auto transaction = BeginTransaction();
        ASSERT_TRUE(transaction.ok());
        // 开启事务不打开引擎（无表访问、无库路由解析）。
        EXPECT_EQ(holder.runnerEngine.openAttempts.load(), 0);
        {
            // 表句柄获取即首次数据库访问：此处（任何 Stage 之前）引擎已被打开。
            auto table = AcquireTable<ContractRecord>();
            ASSERT_TRUE(table.ok());
            EXPECT_EQ(holder.runnerEngine.openAttempts.load(), 1)
                << "table handle acquisition is the first database access";
            ContractRecord record;
            // 首个 Stage 只解析事务物理域：不产生第二次打开。
            EXPECT_TRUE(table->Insert(record, *transaction).ok());
            EXPECT_EQ(holder.runnerEngine.openAttempts.load(), 1);
        }
        EXPECT_EQ(holder.opEngine.openAttempts.load(), 0);
        EXPECT_TRUE(transaction->Rollback().ok());
    } // 事务析构归还在途计数后才能排空关闭
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(holder.session->Close().ok());
}

// 在途打开参与排空：引擎打开期间 quiesce 等待打开线程的临时计数归还，
// 不提前放行、不死锁（打开完成后 quiesce 正常取得静止点）。
TEST(StorageContractTest, InFlightEngineOpenBlocksQuiesceUntilReleased)
{
    LazyEngineSession holder;
    holder.Open("lazy-engine-quiesce");
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());

    std::atomic<bool> countReturned{false};
    std::thread opener([&countReturned]() {
        // 与 quiesce 并发的一次性调用按既有 quiesce 语义可能被拒绝；
        // 本测试关注的是打开在途计数使 quiesce 等待，而非该调用的结果。
        (void)Count<ContractRecord>(All<ContractRecord>());
        countReturned.store(true);
    });
    while (holder.runnerEngine.openAttempts.load() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // 引擎打开在途：quiesce 必须等待（打开线程持有临时在途计数），不提前放行。
    std::atomic<bool> quiesced{false};
    std::thread quiescer([&quiesced]() {
        const auto token = ProcessStorageContext::ApplyQuiesceGate(std::chrono::seconds(10));
        quiesced.store(token.ok());
        if (token.ok()) {
            (void)ProcessStorageContext::ReleaseQuiesceGate(*token.value);
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_FALSE(quiesced.load()); // 打开未完成，quiesce 不得提前返回成功
    holder.runnerEngine.releaseOpen.store(true);
    quiescer.join();
    opener.join();
    EXPECT_TRUE(quiesced.load());
    EXPECT_TRUE(countReturned.load());
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(holder.session->Close().ok());
}

// ==== 惰性打开等待者与 quiesce/close 的排空竞态 ====

// 排空竞态构造引擎：Open 在门闩上保持（构造等待者登记与 quiesce 停泊窗口）；
// 可配置每轮返回成功或失败（失败轮的路由回到 NotOpened，可反复构造轮次）。
class OpenWaiterRaceEngine final : public FakeStorageEngine {
public:
    std::atomic<int> openAttempts{0};
    std::atomic<bool> releaseOpen{false};
    std::atomic<bool> failEveryOpen{false};
    std::atomic<bool> tokenIssued{false};
    std::atomic<bool> engineClosed{false};
    std::atomic<bool> sawExecuteAfterToken{false};
    std::atomic<bool> sawExecuteAfterClose{false};

    DbResult<void> Open(const EngineOpenRequest& request) override
    {
        ++openAttempts;
        while (!releaseOpen.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (failEveryOpen.load()) {
            return {DbCode::BACKEND_UNAVAILABLE, 0, "engine open deliberately fails"};
        }
        return FakeStorageEngine::Open(request);
    }

    DbResult<uint64_t> Insert(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* record,
        TransactionBinding* transaction) override
    {
        if (tokenIssued.load()) {
            sawExecuteAfterToken = true; // quiesce token 发出后不得再有 CRUD 进入引擎
        }
        if (engineClosed.load()) {
            sawExecuteAfterClose = true; // 会话关闭引擎后不得再有 CRUD 进入引擎
        }
        return FakeStorageEngine::Insert(meta, scope, record, transaction);
    }

    DbResult<void> Select(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
        const QueryOptions& options, ResultSink& sink) override
    {
        if (tokenIssued.load()) {
            sawExecuteAfterToken = true;
        }
        if (engineClosed.load()) {
            sawExecuteAfterClose = true;
        }
        return FakeStorageEngine::Select(meta, scope, where, options, sink);
    }

    DbResult<void> Close() override
    {
        engineClosed = true;
        return FakeStorageEngine::Close();
    }
};

// 排空竞态（quiesce 变体）：4 个等待者已被会话接受（登记进同一打开轮次）后应用
// 静止闸门；打开轮随后成功并唤醒等待者——等待者唤醒后必须重新检查闸门状态：
// 不得取得 guard 执行 CRUD（必须被拒绝）；quiesce 不得在这些已接受等待者安全结束
// 或被拒绝前返回（token 到手时不得仍有未决等待者）；token 发出后不得再有 CRUD
// 进入引擎。
TEST(StorageContractTest, QuiesceGateRejectsWokenLazyOpenWaiters)
{
    OpenWaiterRaceEngine engine;
    std::optional<StorageSession> session;
    {
        SessionStorageBinding binding;
        binding.sessionId = "quiesce-waiter-race-session";
        binding.dataNamespace = "quiesce-race";
        binding.databases.push_back({"CONTRACT_TEST_DB", "", &Ddl::kContractTestDatabase});
        StorageOpenRequest request;
        request.engines = {EngineRegistration{"CONTRACT_TEST_DB", kTestFakeBackendName, &engine}};
        auto opened = StorageSession::Open(binding, request);
        ASSERT_TRUE(opened.ok());
        session.emplace(std::move(*opened.value));
    }
    ASSERT_TRUE(ProcessStorageContext::Install(*session).ok());
    const auto pendingWaiters = [&session]() {
        return StorageSessionTestPeer::SyncState(*session).pendingOpenWaiters;
    };

    // 打开者：进入 Open 后被门闩挂住（持有在途计数，使 quiesce 停泊等待排空）。
    std::atomic<bool> leaderReturned{false};
    std::thread opener([&leaderReturned]() {
        (void)Count<ContractRecord>(All<ContractRecord>()); // 打开者自身可完成（在途计数覆盖）
        leaderReturned.store(true);
    });
    while (engine.openAttempts.load() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // 4 个已接受等待者：在打开轮在途时登记（闸门应用之前通过入口检查）。
    constexpr int kWaiters = 4;
    std::vector<DbResult<uint64_t>> waiterResults(kWaiters);
    std::atomic<int> waitersSettled{0};
    std::vector<std::thread> waiters;
    for (int i = 0; i < kWaiters; ++i) {
        waiters.emplace_back([&waiterResults, &waitersSettled, i]() {
            waiterResults[i] = Count<ContractRecord>(All<ContractRecord>());
            ++waitersSettled;
        });
    }
    while (pendingWaiters() < static_cast<std::size_t>(kWaiters)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 停泊的 quiesce：闸门应用后因打开者的在途计数而等待排空（等待者此时已接受）。
    std::atomic<bool> quiesceOk{false};
    std::atomic<std::size_t> waitersPendingAtToken{0};
    std::thread quiescer([&quiesceOk, &waitersPendingAtToken, &session, &engine]() {
        auto token = ProcessStorageContext::ApplyQuiesceGate(std::chrono::seconds(10));
        quiesceOk.store(token.ok());
        if (token.ok()) {
            // token 到手时不得仍有未决打开等待者（quiesce 不得在等待者结束/被拒绝前返回）。
            waitersPendingAtToken.store(StorageSessionTestPeer::SyncState(*session).pendingOpenWaiters);
            engine.tokenIssued.store(true); // token 发出时刻（此后不得再有 CRUD 进引擎）
            (void)ProcessStorageContext::ReleaseQuiesceGate(*token.value);
        }
    });
    // 确认闸门已应用（进程上下文进入 QUIESCING；探测不经会话取表路径，避免登记为等待者）。
    const auto gateDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (ProcessStorageContext::Acquire().code != DbCode::INVALID_STATE
           && std::chrono::steady_clock::now() < gateDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_NE(ProcessStorageContext::Acquire().code, DbCode::OK)
        << "quiesce gate must be applied while the open round is in flight";

    // 打开轮成功：发布结果并唤醒等待者（等待者唤醒时闸门已生效）。
    engine.releaseOpen.store(true);
    quiescer.join();
    for (auto& waiter : waiters) {
        waiter.join();
    }
    opener.join();

    EXPECT_TRUE(quiesceOk.load());
    EXPECT_EQ(waitersSettled.load(), kWaiters);
    // 唤醒后的等待者必须重新检查闸门：不得取得 guard 执行（被拒绝，非 OK）。
    for (int i = 0; i < kWaiters; ++i) {
        EXPECT_EQ(waiterResults[i].code, DbCode::INVALID_STATE)
            << "woken open waiter must be rejected by the quiesce gate (waiter " << i << ")";
    }
    // token 到手时不得仍有未决打开等待者。
    EXPECT_EQ(waitersPendingAtToken.load(), 0U);
    // token 发出后不得再有 CRUD 进入引擎。
    EXPECT_FALSE(engine.sawExecuteAfterToken.load());
    EXPECT_EQ(engine.openAttempts.load(), 1);
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(session->Close().ok());
}

// 排空竞态（close 变体）：同样的已接受等待者窗口下请求会话 Close——等待者唤醒后
// 必须重新检查闸门/关闭状态（被拒绝，不得取得 guard 执行）；Close 在已接受等待者
// 安全结束或被拒绝前不得返回；关闭引擎后不得再有 CRUD 进入引擎。
// （Close 合同要求会话先从进程上下文移除；本测试直接经 StorageSession 取表驱动，
// 与进程上下文路径共用同一 AcquireTable/EnsureRouteOpen 机制。）
TEST(StorageContractTest, CloseDrainsLazyOpenWaitersBeforeClosingEngines)
{
    OpenWaiterRaceEngine engine;
    std::optional<StorageSession> session;
    {
        SessionStorageBinding binding;
        binding.sessionId = "close-waiter-race-session";
        binding.dataNamespace = "close-race";
        binding.databases.push_back({"CONTRACT_TEST_DB", "", &Ddl::kContractTestDatabase});
        StorageOpenRequest request;
        request.engines = {EngineRegistration{"CONTRACT_TEST_DB", kTestFakeBackendName, &engine}};
        auto opened = StorageSession::Open(binding, request);
        ASSERT_TRUE(opened.ok());
        session.emplace(std::move(*opened.value));
    }
    const auto pendingWaiters = [&session]() {
        return StorageSessionTestPeer::SyncState(*session).pendingOpenWaiters;
    };

    std::atomic<bool> leaderReturned{false};
    std::thread opener([&leaderReturned, &session]() {
        auto table = session->Table<ContractRecord>();
        if (table.ok()) {
            (void)table.value->Count(All<ContractRecord>());
        }
        leaderReturned.store(true);
    });
    while (engine.openAttempts.load() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    constexpr int kWaiters = 4;
    std::vector<DbResult<uint64_t>> waiterResults(kWaiters);
    std::atomic<int> waitersSettled{0};
    std::vector<std::thread> waiters;
    for (int i = 0; i < kWaiters; ++i) {
        // 等待者经会话直取表（本测试不安装进程上下文：Close 合同要求会话先
        // 从进程上下文移除）；唤醒后取得表句柄即违反"重新检查闸门状态"的排空合同。
        waiters.emplace_back([&waiterResults, &waitersSettled, &session, i]() {
            auto table = session->Table<ContractRecord>();
            if (!table.ok()) {
                waiterResults[i] = {table.code, std::nullopt, table.affectedRows, table.diagnostic};
                ++waitersSettled;
                return;
            }
            const auto counted = table.value->Count(All<ContractRecord>());
            waiterResults[i] = counted;
            ++waitersSettled;
        });
    }
    while (pendingWaiters() < static_cast<std::size_t>(kWaiters)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 停泊的 Close：闸门应用后因打开者在途计数而等待排空。
    std::atomic<bool> closeReturned{false};
    std::atomic<std::size_t> waitersPendingAtClose{0};
    std::thread closer([&closeReturned, &waitersPendingAtClose, &session]() {
        const auto closed = session->Close(std::chrono::seconds(10));
        closeReturned.store(closed.ok());
        if (closed.ok()) {
            waitersPendingAtClose.store(StorageSessionTestPeer::SyncState(*session).pendingOpenWaiters);
        }
    });
    // 确认闸门已应用（Close 先置 quiescing 拦截新调用；探测只读会话观察点）。
    const auto gateDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!StorageSessionTestPeer::SyncState(*session).sessionQuiescing
           && std::chrono::steady_clock::now() < gateDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(StorageSessionTestPeer::SyncState(*session).sessionQuiescing)
        << "close must apply its drain gate while the open round is in flight";

    // 打开轮成功并唤醒等待者；Close 必须先排空已接受等待者再关闭引擎。
    engine.releaseOpen.store(true);
    closer.join();
    for (auto& waiter : waiters) {
        waiter.join();
    }
    opener.join();

    EXPECT_TRUE(closeReturned.load());
    EXPECT_EQ(waitersSettled.load(), kWaiters);
    for (int i = 0; i < kWaiters; ++i) {
        EXPECT_EQ(waiterResults[i].code, DbCode::INVALID_STATE)
            << "woken open waiter must be rejected by the closing session (waiter " << i << ")";
    }
    // Close 返回时不得仍有未决打开等待者。
    EXPECT_EQ(waitersPendingAtClose.load(), 0U);
    // 关闭引擎后不得再有 CRUD 进入引擎。
    EXPECT_FALSE(engine.sawExecuteAfterClose.load());
    EXPECT_EQ(engine.openAttempts.load(), 1);
}

// 排空竞态（失败轮计数窗口）：打开轮失败时，打开者在发布结果的同一临界区释放
// 在途计数——未计入在途的等待者在此窗口内仍被会话接受。停泊的 quiesce 在该窗口
// 取得 token 即违反"不得在已接受等待者安全结束或被拒绝前返回"。反复构造该窗口
// （50 轮），每轮断言 token 到手时不得仍有未决打开等待者。
TEST(StorageContractTest, QuiesceTokenNotIssuedWhileOpenWaitersPending)
{
    OpenWaiterRaceEngine engine;
    engine.failEveryOpen.store(true);
    std::optional<StorageSession> session;
    {
        SessionStorageBinding binding;
        binding.sessionId = "quiesce-token-window-session";
        binding.dataNamespace = "token-window";
        binding.databases.push_back({"CONTRACT_TEST_DB", "", &Ddl::kContractTestDatabase});
        StorageOpenRequest request;
        request.engines = {EngineRegistration{"CONTRACT_TEST_DB", kTestFakeBackendName, &engine}};
        auto opened = StorageSession::Open(binding, request);
        ASSERT_TRUE(opened.ok());
        session.emplace(std::move(*opened.value));
    }
    ASSERT_TRUE(ProcessStorageContext::Install(*session).ok());
    const auto pendingWaiters = [&session]() {
        return StorageSessionTestPeer::SyncState(*session).pendingOpenWaiters;
    };

    constexpr int kRounds = 50;
    constexpr int kWaiters = 4;
    for (int round = 0; round < kRounds; ++round) {
        engine.releaseOpen.store(false);
        // 打开者：进入 Open 后被门闩挂住（持有在途计数使 quiesce 停泊）。
        std::atomic<bool> leaderReturned{false};
        std::thread opener([&leaderReturned]() {
            (void)Count<ContractRecord>(All<ContractRecord>());
            leaderReturned.store(true);
        });
        while (engine.openAttempts.load() < round + 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // 已接受等待者登记进本轮。
        std::atomic<int> waitersSettled{0};
        std::vector<std::thread> waiters;
        for (int i = 0; i < kWaiters; ++i) {
            waiters.emplace_back([&waitersSettled]() {
                (void)Count<ContractRecord>(All<ContractRecord>());
                ++waitersSettled;
            });
        }
        while (pendingWaiters() < static_cast<std::size_t>(kWaiters)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        // 停泊的 quiesce：闸门因打开者在途计数而等待排空。
        std::atomic<bool> quiesceOk{false};
        std::atomic<std::size_t> waitersPendingAtToken{0};
        std::thread quiescer([&quiesceOk, &waitersPendingAtToken, &session]() {
            auto token = ProcessStorageContext::ApplyQuiesceGate(std::chrono::seconds(10));
            quiesceOk.store(token.ok());
            if (token.ok()) {
                waitersPendingAtToken.store(StorageSessionTestPeer::SyncState(*session).pendingOpenWaiters);
                (void)ProcessStorageContext::ReleaseQuiesceGate(*token.value);
            }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        // 失败轮：发布失败的同一临界区释放在途计数——停泊的 quiesce 与等待者竞争
        // 唤醒后的首把锁；等待者若未从被接受起计入在途，quiesce 将提前取得 token。
        engine.releaseOpen.store(true);
        quiescer.join();
        for (auto& waiter : waiters) {
            waiter.join();
        }
        opener.join();
        EXPECT_TRUE(quiesceOk.load()) << "round " << round;
        EXPECT_EQ(waitersSettled.load(), kWaiters) << "round " << round;
        // token 到手时不得仍有未决打开等待者（等待者从被接受起就必须被排空覆盖）。
        EXPECT_EQ(waitersPendingAtToken.load(), 0U) << "round " << round;
    }
    EXPECT_EQ(engine.openAttempts.load(), kRounds);
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(session->Close().ok());
}

// ==== 引擎/连接真正惰性创建（REQ-CAP-2）与失败可重试性 ====

// 计数生命周期引擎：按工厂每次构造新实例（实例编号）；Open 计数且 Close 为终态
// （终态后 Open 拒绝——与真实关系引擎合同一致）；Prepare 可注入前 N 次失败；
// Open/Prepare 可注入抛异常；Execute 记录最后执行的实例编号（fork 不复用观察点）。
class CountingLifecycleEngine final : public FakeStorageEngine {
public:
    struct Shared {
        std::atomic<int> constructions{0};
        std::atomic<int> openAttempts{0};
        std::atomic<int> prepareAttempts{0};
        std::atomic<int> closeCalls{0};
        std::atomic<int> prepareFailuresLeft{0};
        std::atomic<int> throwAtOpenCount{0};
        std::atomic<int> throwAtPrepareCount{0};
        std::atomic<int> lastExecutorId{0};
        /// 失败/抛异常轮的门闩：保持轮次在途，供等待者登记共享同一轮。
        std::atomic<bool> releaseRound{false};
    };

    explicit CountingLifecycleEngine(Shared& shared) : shared_(shared), id_(shared.constructions.load()) {}

    DbResult<void> Open(const EngineOpenRequest& request) override
    {
        if (terminated_) {
            return {DbCode::INVALID_STATE, 0, "engine is closed"};
        }
        ++shared_.openAttempts;
        if (shared_.throwAtOpenCount.load() > 0) {
            while (!shared_.releaseRound.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            --shared_.throwAtOpenCount;
            throw std::runtime_error("engine open exploded");
        }
        return FakeStorageEngine::Open(request);
    }

    DbResult<void> Prepare() override
    {
        const int attempt = ++shared_.prepareAttempts;
        if (shared_.throwAtPrepareCount.load() > 0) {
            while (!shared_.releaseRound.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            --shared_.throwAtPrepareCount;
            throw std::runtime_error("engine prepare exploded");
        }
        if (attempt <= shared_.prepareFailuresLeft.load()) {
            while (!shared_.releaseRound.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return {DbCode::BACKEND_UNAVAILABLE, 0, "prepare deliberately fails"};
        }
        return OkVoid();
    }

    DbResult<uint64_t> Insert(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* record,
        TransactionBinding* transaction) override
    {
        shared_.lastExecutorId.store(id_);
        return FakeStorageEngine::Insert(meta, scope, record, transaction);
    }

    DbResult<uint64_t>
    Count(const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where) override
    {
        shared_.lastExecutorId.store(id_);
        return FakeStorageEngine::Count(meta, scope, where);
    }

    DbResult<void> Select(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
        const QueryOptions& options, ResultSink& sink) override
    {
        shared_.lastExecutorId.store(id_);
        return FakeStorageEngine::Select(meta, scope, where, options, sink);
    }

    DbResult<void> Close() override
    {
        ++shared_.closeCalls;
        terminated_ = true;
        return OkVoid();
    }

    Shared& shared_;
    int id_{0};
    bool terminated_{false};
};

// 计数工厂装配：经 StorageRuntime 注册后端工厂（引擎构造真实计数），按显式配置
// 安装进程会话；工厂/目录登记用 ResetForTest 清理。
struct CountingRuntimeFixture {
    CountingLifecycleEngine::Shared state;
    StorageRuntimeConfig config;

    DbResult<void> Install()
    {
        auto& runtime = HcclSim::Storage::detail::StorageRuntimeImpl::Instance();
        runtime.RegisterEngineFactory(
            kTestFakeBackendName,
            [this](const LogicalDatabaseRuntimeConfig&) -> DbResult<std::unique_ptr<IStorageEngine>> {
                ++state.constructions;
                return {
                    DbCode::OK,
                    std::optional<std::unique_ptr<IStorageEngine>>{std::make_unique<CountingLifecycleEngine>(state)},
                    0,
                    {}};
            });
        runtime.UseDatabase(Ddl::kContractTestDatabase);
        runtime.UseDatabase(Ddl::kContractTestOpDatabase);
        if (config.databases.empty()) {
            config.databases.push_back({"CONTRACT_TEST_DB", kTestFakeBackendName, "counting-runner"});
            config.databases.push_back({"CONTRACT_TEST_OP_DB", kTestFakeBackendName, "counting-opdata"});
        }
        return StorageRuntime::EnsureProcessSession(config);
    }

    void Teardown()
    {
        (void)StorageRuntime::CloseProcessSession();
        HcclSim::Storage::detail::StorageRuntimeImpl::Instance().ResetForTest();
    }
};

// REQ-CAP-2：Session 安装不得创建引擎/连接（引擎构造真实计数为 0）；首次真实访问
// 才构造本库引擎；同库并发首访只构造一次；未访问的库在 Close 时不得被构造；
// fork 子进程不复用父进程 Engine。
TEST(StorageContractTest, RuntimeSessionDefersEngineConstructionUntilFirstUse)
{
    CountingRuntimeFixture fixture;
    ASSERT_TRUE(fixture.Install().ok());
    // 安装即构造（旧实现急切创建两个引擎）——本断言即 RED 证据点。
    EXPECT_EQ(fixture.state.constructions.load(), 0) << "session install must not construct engines";

    // 首次 Runner 访问：只构造 Runner 引擎。
    ASSERT_TRUE(Count<ContractRecord>(All<ContractRecord>()).ok());
    EXPECT_EQ(fixture.state.constructions.load(), 1);
    // 重复访问复用同一引擎实例。
    ASSERT_TRUE(Count<ContractRecord>(All<ContractRecord>()).ok());
    EXPECT_EQ(fixture.state.constructions.load(), 1);
    EXPECT_EQ(fixture.state.openAttempts.load(), 1);

    // 首次 OpData 访问：只构造 OpData 引擎。
    ASSERT_TRUE(Count<OpRecord>(All<OpRecord>()).ok());
    EXPECT_EQ(fixture.state.constructions.load(), 2);

    // 同库并发首访（换新会话装配，两库都未构造）：只构造一次。
    fixture.Teardown();
    CountingRuntimeFixture concurrent;
    ASSERT_TRUE(concurrent.Install().ok());
    constexpr int kWorkers = 8;
    std::vector<DbResult<uint64_t>> results(kWorkers);
    std::vector<std::thread> workers;
    for (int i = 0; i < kWorkers; ++i) {
        workers.emplace_back([&results, i]() {
            results[i] = Count<OpRecord>(All<OpRecord>());
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    for (const auto& counted : results) {
        EXPECT_TRUE(counted.ok()) << counted.diagnostic;
    }
    EXPECT_EQ(concurrent.state.constructions.load(), 1) << "concurrent first access must construct once";
    EXPECT_EQ(concurrent.state.openAttempts.load(), 1);

    // 只访问 Runner 的新会话：Close 时不构造未访问库（OpData）的引擎。
    fixture.Teardown();
    CountingRuntimeFixture closeOnly;
    ASSERT_TRUE(closeOnly.Install().ok());
    ASSERT_TRUE(Count<ContractRecord>(All<ContractRecord>()).ok());
    EXPECT_EQ(closeOnly.state.constructions.load(), 1);
    closeOnly.Teardown(); // CloseProcessSession 排空并关闭
    EXPECT_EQ(closeOnly.state.constructions.load(), 1) << "close must not construct unaccessed engines";
}

// Prepare 失败后的真实可重试性：第一轮 Open 成功、Prepare 注入失败（并发等待者共享
// 该失败）；修复外部条件后第二次访问必须重新构造引擎并成功（终态 Close 后的同一
// 对象不得伪装成可重试）；无永久 poisoned 状态。
TEST(StorageContractTest, PrepareFailureIsRetryableWithFreshEngine)
{
    CountingRuntimeFixture fixture;
    fixture.state.prepareFailuresLeft.store(1);
    ASSERT_TRUE(fixture.Install().ok());

    // 第一轮：Prepare 门闩保持轮次在途，4 个等待者登记共享本轮失败。
    std::atomic<int> waitersSettled{0};
    std::vector<DbResult<uint64_t>> waiterResults(4);
    std::thread first([]() {
        (void)Count<ContractRecord>(All<ContractRecord>());
    });
    std::vector<std::thread> waiters;
    for (int i = 0; i < 4; ++i) {
        waiters.emplace_back([&waiterResults, &waitersSettled, i]() {
            waiterResults[i] = Count<ContractRecord>(All<ContractRecord>());
            ++waitersSettled;
        });
    }
    while (waitersSettled.load() == 0 && fixture.state.prepareAttempts.load() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // 等待者全部登记后放行失败轮。
    {
        auto lease = ProcessStorageContext::Acquire();
        ASSERT_TRUE(lease.ok());
        while (StorageSessionTestPeer::SyncState(lease.value->Get()).pendingOpenWaiters < 4) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    fixture.state.releaseRound.store(true);
    first.join();
    for (auto& waiter : waiters) {
        waiter.join();
    }
    EXPECT_EQ(waitersSettled.load(), 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(waiterResults[i].code, DbCode::BACKEND_UNAVAILABLE);
        EXPECT_EQ(waiterResults[i].diagnostic, "prepare deliberately fails");
    }
    EXPECT_EQ(fixture.state.constructions.load(), 1);
    EXPECT_EQ(fixture.state.closeCalls.load(), 1) << "failed prepare must close the engine terminal state";

    // 修复外部条件（后续 Prepare 成功）后的重试：必须重新构造引擎并成功。
    const auto retried = Count<ContractRecord>(All<ContractRecord>());
    ASSERT_TRUE(retried.ok()) << retried.diagnostic;
    EXPECT_EQ(fixture.state.constructions.load(), 2) << "retry must construct a fresh engine instance";
    EXPECT_EQ(fixture.state.openAttempts.load(), 2);
    // 无永久 poisoned 状态：后续访问持续成功。
    EXPECT_TRUE(Count<ContractRecord>(All<ContractRecord>()).ok());
    EXPECT_EQ(fixture.state.constructions.load(), 2);
    ASSERT_TRUE(StorageRuntime::CloseProcessSession().ok());
    HcclSim::Storage::detail::StorageRuntimeImpl::Instance().ResetForTest();
}

// 引擎 Open 抛异常：异常必须被收尾——Opening 状态清除、在途计数归还、同轮等待者
// 唤醒并收到可诊断失败（不传播异常、不死锁）、后续可重试（条件修复后成功）。
TEST(StorageContractTest, ThrowingEngineOpenWakesWaitersAndStaysRetryable)
{
    CountingRuntimeFixture fixture;
    fixture.state.throwAtOpenCount.store(1);
    ASSERT_TRUE(fixture.Install().ok());

    std::atomic<int> waitersSettled{0};
    std::atomic<bool> leaderThrew{false};
    std::vector<DbResult<uint64_t>> waiterResults(4);
    std::thread leader([&leaderThrew]() {
        try {
            (void)Count<ContractRecord>(All<ContractRecord>());
        } catch (const std::exception&) {
            leaderThrew = true; // 旧实现：异常直接传播（RED 证据）
        }
    });
    std::vector<std::thread> waiters;
    for (int i = 0; i < 4; ++i) {
        waiters.emplace_back([&waiterResults, &waitersSettled, i]() {
            waiterResults[i] = Count<ContractRecord>(All<ContractRecord>());
            ++waitersSettled;
        });
    }
    // 等待者全部登记后放行抛异常轮。
    {
        auto lease = ProcessStorageContext::Acquire();
        ASSERT_TRUE(lease.ok());
        while (StorageSessionTestPeer::SyncState(lease.value->Get()).pendingOpenWaiters < 4) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    fixture.state.releaseRound.store(true);
    leader.join();
    // 等待者必须在时限内被唤醒并收到结果（detach：旧实现会永久阻塞等待者）。
    for (auto& waiter : waiters) {
        waiter.detach();
    }
    const auto settleDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (waitersSettled.load() < 4 && std::chrono::steady_clock::now() < settleDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    // 引擎异常必须被收尾转换（不传播到调用方）。
    EXPECT_FALSE(leaderThrew.load()) << "engine exception must not propagate out of the entry point";
    EXPECT_EQ(waitersSettled.load(), 4) << "throwing engine must wake all same-round waiters";
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(waiterResults[i].code, DbCode::BACKEND_UNAVAILABLE);
        EXPECT_NE(waiterResults[i].diagnostic.find("engine open exploded"), std::string::npos)
            << "exception must surface as a diagnosable failure";
    }
    // Opening 状态已清除、无残留未决等待者（后续可重试的前提）。
    {
        auto lease = ProcessStorageContext::Acquire();
        ASSERT_TRUE(lease.ok());
        EXPECT_EQ(StorageSessionTestPeer::SyncState(lease.value->Get()).pendingOpenWaiters, 0U);
    }
    // 条件修复后重试成功（重新构造引擎）。
    const auto retried = Count<ContractRecord>(All<ContractRecord>());
    ASSERT_TRUE(retried.ok()) << retried.diagnostic;
    EXPECT_EQ(fixture.state.constructions.load(), 2);
    ASSERT_TRUE(StorageRuntime::CloseProcessSession().ok());
    HcclSim::Storage::detail::StorageRuntimeImpl::Instance().ResetForTest();
}

// 引擎 Prepare 抛异常：同一收尾合同（状态清除、等待者唤醒、可诊断失败、可重试）。
TEST(StorageContractTest, ThrowingEnginePrepareWakesWaitersAndStaysRetryable)
{
    CountingRuntimeFixture fixture;
    fixture.state.throwAtPrepareCount.store(1);
    ASSERT_TRUE(fixture.Install().ok());

    std::atomic<int> waitersSettled{0};
    std::atomic<bool> leaderThrew{false};
    std::vector<DbResult<uint64_t>> waiterResults(4);
    std::thread leader([&leaderThrew]() {
        try {
            (void)Count<ContractRecord>(All<ContractRecord>());
        } catch (const std::exception&) {
            leaderThrew = true; // 旧实现：异常直接传播（RED 证据）
        }
    });
    std::vector<std::thread> waiters;
    for (int i = 0; i < 4; ++i) {
        waiters.emplace_back([&waiterResults, &waitersSettled, i]() {
            waiterResults[i] = Count<ContractRecord>(All<ContractRecord>());
            ++waitersSettled;
        });
    }
    {
        auto lease = ProcessStorageContext::Acquire();
        ASSERT_TRUE(lease.ok());
        while (StorageSessionTestPeer::SyncState(lease.value->Get()).pendingOpenWaiters < 4) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    fixture.state.releaseRound.store(true);
    leader.join();
    for (auto& waiter : waiters) {
        waiter.detach();
    }
    const auto settleDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (waitersSettled.load() < 4 && std::chrono::steady_clock::now() < settleDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_FALSE(leaderThrew.load()) << "engine exception must not propagate out of the entry point";
    EXPECT_EQ(waitersSettled.load(), 4) << "throwing engine must wake all same-round waiters";
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(waiterResults[i].code, DbCode::BACKEND_UNAVAILABLE);
        EXPECT_NE(waiterResults[i].diagnostic.find("engine prepare exploded"), std::string::npos);
    }
    {
        auto lease = ProcessStorageContext::Acquire();
        ASSERT_TRUE(lease.ok());
        EXPECT_EQ(StorageSessionTestPeer::SyncState(lease.value->Get()).pendingOpenWaiters, 0U);
    }
    const auto retried = Count<ContractRecord>(All<ContractRecord>());
    ASSERT_TRUE(retried.ok()) << retried.diagnostic;
    EXPECT_EQ(fixture.state.constructions.load(), 2);
    ASSERT_TRUE(StorageRuntime::CloseProcessSession().ok());
    HcclSim::Storage::detail::StorageRuntimeImpl::Instance().ResetForTest();
}

// ==== 生命周期加固：失败轮次隔离 / bootstrap 异常收尾 / OK-未安装后置条件 ====

// 两轮 bootstrap：第一轮确定性失败（可诊断错误），第二轮经标准注册会话成功。
// 用于构造"第一轮失败后新调用者立即竞争发起第二轮"的轮次污染窗口。
struct TwoRoundGate {
    std::atomic<int> calls{0};
    std::atomic<bool> releaseFirst{false};
    std::atomic<bool> releaseSecond{false};
};

TwoRoundGate& RoundGate()
{
    static TwoRoundGate gate;
    return gate;
}

DbResult<void> TwoRoundFailingThenSucceedingBootstrap()
{
    const int call = ++RoundGate().calls;
    if (call == 1) {
        while (!RoundGate().releaseFirst.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return {DbCode::BACKEND_UNAVAILABLE, 0, "first-round bootstrap failure"};
    }
    while (!RoundGate().releaseSecond.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return BootstrapRegistryContractSession();
}

// 失败轮次隔离：第一轮失败通知后，新调用者竞争发起第二轮重试；第一轮的全部
// 既有参与者（引导线程 + 已进入第一轮的等待者）必须精确返回第一轮错误，不被
// 下一轮重试污染成第二轮结果；每轮 bootstrap 精确执行一次；无死锁。
TEST(StorageContractTest, AcquireReadyFailedRoundWaitersNotAbsorbedByNextRound)
{
    ProcessStorageContext::ResetAfterForkInChild();
    RoundGate().calls.store(0);
    RoundGate().releaseFirst.store(false);
    RoundGate().releaseSecond.store(false);
    BootstrapRegistryCalls() = 0;
    ContractBootstrapModeRef().store(ContractBootstrapMode::TwoRoundFailingThenSucceeding);
    ASSERT_TRUE(ProcessStorageContext::RegisterBootstrap(&ContractBootstrapDispatcher).ok());

    // 第一轮参与者：引导线程 + 固定等待者 + 失败后立即重试的循环调用者。
    // 重试调用者与固定等待者一样在第一轮在途时进入（成为第一轮参与者），
    // 消费到第一轮失败后在同一 CPU 上紧邻重试（相对 futex 唤醒延迟有绝对
    // 抢锁优势），确定性地在剩余等待者的消费链中间发起第二轮重试——用于
    // 验证旧等待者的结果不被下一轮重试污染。
    constexpr int kWaiters = 16;
    constexpr int kRetryCallers = 3;
    std::atomic<int> participantsEntered{0};
    std::mutex resultsMutex;
    // 第一轮参与者（引导线程 + 等待者）的返回码与诊断。
    std::vector<std::pair<DbCode, std::string>> firstRoundResults;
    auto recordFirstRound = [&resultsMutex, &firstRoundResults](const DbResult<StorageSessionLease>& lease) {
        std::lock_guard<std::mutex> lock(resultsMutex);
        firstRoundResults.emplace_back(lease.code, lease.diagnostic);
    };

    // 第一轮引导线程。
    std::atomic<bool> leaderReturned{false};
    std::thread leader([&recordFirstRound, &leaderReturned]() {
        auto lease = ProcessStorageContext::AcquireReady();
        recordFirstRound(lease);
        leaderReturned.store(true);
    });
    while (RoundGate().calls.load() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // 固定等待者：在第一轮 bootstrap 在途时进入等待。
    std::vector<std::thread> waiters;
    for (int i = 0; i < kWaiters; ++i) {
        waiters.emplace_back([&participantsEntered, &recordFirstRound]() {
            ++participantsEntered;
            auto lease = ProcessStorageContext::AcquireReady();
            recordFirstRound(lease);
        });
    }
    // 失败后立即重试的循环调用者：第一次调用同样进入第一轮等待。
    std::atomic<int> retrySucceeded{0};
    std::vector<std::thread> retryCallers;
    for (int i = 0; i < kRetryCallers; ++i) {
        retryCallers.emplace_back([&participantsEntered, &retrySucceeded]() {
            ++participantsEntered;
            // 失败立即重试：其中一个调用者将在其余等待者消费链中间发起第二轮。
            for (int attempt = 0; attempt < 500; ++attempt) {
                auto lease = ProcessStorageContext::AcquireReady();
                if (lease.ok()) {
                    ++retrySucceeded;
                    return;
                }
            }
        });
    }
    while (participantsEntered.load() < kWaiters + kRetryCallers) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // 第一轮结束（失败）。
    RoundGate().releaseFirst.store(true);
    while (!leaderReturned.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // 等待第二轮真实开始（某个重试调用者发起新一轮 bootstrap）。
    const auto secondRoundDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (RoundGate().calls.load() < 2 && std::chrono::steady_clock::now() < secondRoundDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(RoundGate().calls.load(), 2) << "new caller should start the second round after first-round failure";
    // 第二轮保持一段在途窗口，让第一轮等待者的消费行为充分暴露（被卷入的实现
    // 会阻塞在第二轮等待上）。
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    RoundGate().releaseSecond.store(true);

    leader.join();
    for (auto& waiter : waiters) {
        waiter.join();
    }
    for (auto& caller : retryCallers) {
        caller.join();
    }

    // 第一轮全部固定参与者（引导线程 + 固定等待者）精确返回第一轮错误。
    ASSERT_EQ(firstRoundResults.size(), static_cast<std::size_t>(kWaiters + 1));
    for (const auto& result : firstRoundResults) {
        EXPECT_EQ(result.first, DbCode::BACKEND_UNAVAILABLE);
        EXPECT_EQ(result.second, "first-round bootstrap failure");
    }
    // 重试调用者最终观察第二轮结果（成功）。
    EXPECT_EQ(retrySucceeded.load(), kRetryCallers);
    // 每轮 bootstrap 精确执行一次（失败轮与重试轮各一次）。
    EXPECT_EQ(RoundGate().calls.load(), 2);
    EXPECT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
}

// 抛异常的 bootstrap：先在门闩上保持一轮在途，放行后抛出 std::runtime_error。
struct ThrowingGate {
    std::atomic<int> calls{0};
    std::atomic<bool> release{false};
};

ThrowingGate& ThrowGate()
{
    static ThrowingGate gate;
    return gate;
}

DbResult<void> GatedThrowingBootstrapContractSession()
{
    ++ThrowGate().calls;
    while (!ThrowGate().release.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("bootstrap exploded");
}

// bootstrap 抛异常不得永久毒化 single-flight：全部等待者必须被唤醒并收到可诊断
// 失败（现有 DbCode，含异常文本），bootstrapInProgress 必须恢复，后续新调用允许
// 重试；不得死锁、不得终止进程、不得吞掉已有 DbResult 错误。
TEST(StorageContractTest, ThrowingBootstrapWakesWaitersAndKeepsSingleFlightUsable)
{
    ProcessStorageContext::ResetAfterForkInChild();
    ThrowGate().calls.store(0);
    ThrowGate().release.store(false);
    ContractBootstrapModeRef().store(ContractBootstrapMode::Throwing);
    ASSERT_TRUE(ProcessStorageContext::RegisterBootstrap(&ContractBootstrapDispatcher).ok());

    constexpr int kWaiters = 4;
    std::atomic<int> waitersEntered{0};
    std::atomic<int> waitersReturned{0};
    std::atomic<bool> leaderReturned{false};
    std::mutex resultsMutex;
    std::vector<std::pair<DbCode, std::string>> results;
    auto recordResult = [&resultsMutex, &results](const DbResult<StorageSessionLease>& lease) {
        std::lock_guard<std::mutex> lock(resultsMutex);
        results.emplace_back(lease.code, lease.diagnostic);
    };
    auto caughtException = [&resultsMutex, &results](const char* what) {
        std::lock_guard<std::mutex> lock(resultsMutex);
        results.emplace_back(DbCode::IO_ERROR, what);
    };

    // 引导线程：修复后收到可诊断失败；旧实现异常直接传播（线程内捕获以便对比）。
    std::thread leader([&leaderReturned, &recordResult, &caughtException]() {
        try {
            auto lease = ProcessStorageContext::AcquireReady();
            recordResult(lease);
        } catch (const std::exception& ex) {
            caughtException(ex.what());
        }
        leaderReturned.store(true);
    });
    while (ThrowGate().calls.load() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // 第一轮等待者：bootstrap 在途时进入等待。
    std::vector<std::thread> waiters;
    for (int i = 0; i < kWaiters; ++i) {
        waiters.emplace_back([&waitersEntered, &waitersReturned, &recordResult, &caughtException]() {
            ++waitersEntered;
            try {
                auto lease = ProcessStorageContext::AcquireReady();
                recordResult(lease);
            } catch (const std::exception& ex) {
                caughtException(ex.what());
            }
            ++waitersReturned;
        });
    }
    while (waitersEntered.load() < kWaiters) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 放行：bootstrap 抛出异常。
    ThrowGate().release.store(true);

    // 全部第一轮参与者（引导线程 + 等待者）必须在时限内返回；旧实现会永久阻塞。
    const auto wakeDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while ((waitersReturned.load() < kWaiters || !leaderReturned.load())
           && std::chrono::steady_clock::now() < wakeDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_EQ(waitersReturned.load(), kWaiters) << "throwing bootstrap must wake all waiters";
    ASSERT_TRUE(leaderReturned.load()) << "throwing bootstrap must not block the bootstrap thread";
    leader.join();
    for (auto& waiter : waiters) {
        waiter.join();
    }

    // 全部参与者收到同一可诊断失败（不吞错误、不终止进程、不传播异常）。
    ASSERT_EQ(results.size(), static_cast<std::size_t>(kWaiters + 1));
    for (const auto& result : results) {
        EXPECT_EQ(result.first, DbCode::BACKEND_UNAVAILABLE);
        EXPECT_NE(result.second.find("bootstrap exploded"), std::string::npos);
    }
    EXPECT_EQ(ThrowGate().calls.load(), 1);

    // single-flight 未被毒化：后续新调用可以重试（再次执行 bootstrap，再次失败可诊断）。
    std::atomic<bool> retryReturned{false};
    std::thread retry([&retryReturned, &recordResult, &caughtException]() {
        try {
            auto lease = ProcessStorageContext::AcquireReady();
            recordResult(lease);
        } catch (const std::exception& ex) {
            caughtException(ex.what());
        }
        retryReturned.store(true);
    });
    const auto retryDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!retryReturned.load() && std::chrono::steady_clock::now() < retryDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_TRUE(retryReturned.load()) << "single-flight must stay usable after a throwing bootstrap";
    retry.join();
    ASSERT_EQ(results.size(), static_cast<std::size_t>(kWaiters + 2));
    const auto& retried = results.back();
    EXPECT_EQ(retried.first, DbCode::BACKEND_UNAVAILABLE);
    EXPECT_NE(retried.second.find("bootstrap exploded"), std::string::npos);
    EXPECT_EQ(ThrowGate().calls.load(), 2);
}

// 返回 OK 但不安装会话的 bootstrap：AcquireReady 不得无限循环重试 bootstrap，
// 必须返回有限失败且诊断明确指出 bootstrap 成功但未安装进程会话；后续新调用
// 与普通 bootstrap 失败保持同一重试规则。
struct OkNoInstallGate {
    std::atomic<int> calls{0};
};

OkNoInstallGate& OkNoInstallGateRef()
{
    static OkNoInstallGate gate;
    return gate;
}

DbResult<void> OkWithoutInstallBootstrapContractSession()
{
    ++OkNoInstallGateRef().calls;
    return {DbCode::OK, 0, {}};
}

TEST(StorageContractTest, BootstrapOkWithoutInstallFailsFinite)
{
    ProcessStorageContext::ResetAfterForkInChild();
    OkNoInstallGateRef().calls.store(0);
    ContractBootstrapModeRef().store(ContractBootstrapMode::OkWithoutInstall);
    ASSERT_TRUE(ProcessStorageContext::RegisterBootstrap(&ContractBootstrapDispatcher).ok());

    // 调用必须有限返回：旧实现会无限循环重复执行 bootstrap（忙旋）。
    std::atomic<bool> returned{false};
    std::mutex resultsMutex;
    DbResult<StorageSessionLease> first{DbCode::IO_ERROR, std::nullopt, 0, "not-run"};
    std::thread caller([&returned, &resultsMutex, &first]() {
        auto lease = ProcessStorageContext::AcquireReady();
        {
            std::lock_guard<std::mutex> lock(resultsMutex);
            first = std::move(lease);
        }
        returned.store(true);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!returned.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_TRUE(returned.load()) << "bootstrap returning OK without install must fail finite, not spin forever";
    caller.join();

    // 有限失败：INVALID_STATE + 诊断明确指出 bootstrap succeeded without installing。
    EXPECT_EQ(first.code, DbCode::INVALID_STATE);
    EXPECT_NE(first.diagnostic.find("without installing"), std::string::npos);
    // bootstrap 只执行一次（不得连续重复重试形成忙旋）。
    EXPECT_EQ(OkNoInstallGateRef().calls.load(), 1);

    // 后续新调用与普通 bootstrap 失败同一规则：允许重试（再执行一次、再次有限失败）。
    const auto second = ProcessStorageContext::AcquireReady();
    EXPECT_EQ(second.code, DbCode::INVALID_STATE);
    EXPECT_NE(second.diagnostic.find("without installing"), std::string::npos);
    EXPECT_EQ(OkNoInstallGateRef().calls.load(), 2);
}

// ==== review-fix：routeOpenConsumed 重试等待者越过 quiesce / 预创建引擎失败语义 /
// ==== factory 异常返回 / maintenance 惰性打开 / schemaRole 优先级 / fork child 重置锁安全 ====

// 问题1观察引擎：Open 计数并观察 tokenIssued（quiesce token 发出后不得再有 Open/CRUD）。
class RetryDrainRaceEngine final : public FakeStorageEngine {
public:
    std::atomic<int> openAttempts{0};
    std::atomic<bool> releaseOpen{false};
    std::atomic<bool> tokenIssued{false};
    std::atomic<bool> sawOpenAfterToken{false};
    std::atomic<bool> sawExecuteAfterToken{false};

    DbResult<void> Open(const EngineOpenRequest& request) override
    {
        ++openAttempts;
        if (tokenIssued.load()) {
            sawOpenAfterToken = true; // token 发出后不得再启动新一轮 Open
        }
        while (!releaseOpen.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return {DbCode::BACKEND_UNAVAILABLE, 0, "engine open deliberately fails"};
    }

    DbResult<uint64_t> Insert(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* record,
        TransactionBinding* transaction) override
    {
        if (tokenIssued.load()) {
            sawExecuteAfterToken = true; // token 发出后不得再有 CRUD 进入引擎
        }
        return FakeStorageEngine::Insert(meta, scope, record, transaction);
    }

    DbResult<void> Select(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
        const QueryOptions& options, ResultSink& sink) override
    {
        if (tokenIssued.load()) {
            sawExecuteAfterToken = true;
        }
        return FakeStorageEngine::Select(meta, scope, where, options, sink);
    }
};

/// 单库会话装配（预创建引擎或 factory 形态；供 review-fix 用例复用）。
struct ReviewFixSession {
    ReviewFixSession() = default;
    ReviewFixSession(const ReviewFixSession&) = delete;
    ReviewFixSession& operator=(const ReviewFixSession&) = delete;

    void OpenWithEngine(IStorageEngine& engine, const char* sessionId)
    {
        SessionStorageBinding binding;
        binding.sessionId = sessionId;
        binding.dataNamespace = "review-fix";
        binding.databases.push_back({"CONTRACT_TEST_DB", "", &Ddl::kContractTestDatabase});
        StorageOpenRequest request;
        request.engines = {EngineRegistration{"CONTRACT_TEST_DB", kTestFakeBackendName, &engine}};
        auto opened = StorageSession::Open(binding, request);
        ASSERT_TRUE(opened.ok());
        session.emplace(std::move(*opened.value));
    }

    std::optional<StorageSession> session;
};

// 问题1（quiesce 变体，20 轮压力）：第一轮 Open 失败发布后、固定等待者消费链进行中，
// 已被会话接受的重试调用者进入 routeOpenConsumed 等待（旧实现未把该等待者计入排空
// 模型）。随后停泊 quiesce：旧实现可在其未计数时取得 token，该调用者唤醒后只查
// closed 即继续——启动第二轮 Open（token 后 Open）。观察窗口为 token 保持期
// （gate 持有 50ms，期间不得有新 Open/CRUD；释放后的新 Open 属合法恢复不判违规）。
TEST(StorageContractTest, RetryCallerWaitingForRoundConsumptionJoinsDrainModel)
{
    RetryDrainRaceEngine engine;
    ReviewFixSession holder;
    holder.OpenWithEngine(engine, "retry-drain-race-session");
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());
    const auto pendingWaiters = [&holder]() {
        return StorageSessionTestPeer::SyncState(*holder.session).pendingOpenWaiters;
    };

    constexpr int kRounds = 20;
    constexpr int kWaiters = 64;
    constexpr int kRetryCallers = 2;
    for (int round = 0; round < kRounds; ++round) {
        engine.releaseOpen.store(false);
        engine.tokenIssued.store(false); // 观察标志按轮复位
        engine.sawOpenAfterToken.store(false);
        engine.sawExecuteAfterToken.store(false);
        std::atomic<bool> leaderReturned{false};
        std::thread leader([&leaderReturned]() {
            (void)Count<ContractRecord>(All<ContractRecord>());
            leaderReturned.store(true);
        });
        while (engine.openAttempts.load() < round + 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::atomic<int> settled{0};
        std::vector<std::thread> waiters;
        for (int i = 0; i < kWaiters; ++i) {
            waiters.emplace_back([&settled]() {
                (void)Count<ContractRecord>(All<ContractRecord>());
                ++settled;
            });
        }
        // 重试调用者：首次调用与固定等待者一起加入第一轮（入口闸门通过，已被会话
        // 接受）；消费失败后立即重试——在旧等待者消费链中间进入 routeOpenConsumed
        // 等待（排空模型必须覆盖的窗口）。
        std::vector<std::thread> retryCallers;
        for (int i = 0; i < kRetryCallers; ++i) {
            retryCallers.emplace_back([&settled]() {
                (void)Count<ContractRecord>(All<ContractRecord>()); // 加入第一轮
                ++settled;
                for (int attempt = 0; attempt < 50; ++attempt) {
                    const auto counted = Count<ContractRecord>(All<ContractRecord>());
                    if (counted.ok() || counted.code == DbCode::INVALID_STATE) {
                        return; // 成功或被闸门拒绝都视为有限终结
                    }
                }
            });
        }
        while (pendingWaiters() < static_cast<std::size_t>(kWaiters + kRetryCallers)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // 放行第一轮（失败发布；等待者开始消费，重试调用者随即停泊进
        // routeOpenConsumed）。1ms 后停泊 quiesce：位于消费链中间的重试等待窗口内。
        engine.releaseOpen.store(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::atomic<bool> quiesceOk{false};
        std::atomic<std::size_t> waitersPendingAtToken{0};
        std::atomic<int> openAttemptsAtToken{0};
        std::atomic<int> openAttemptsAtRelease{0};
        std::thread quiescer([&]() {
            auto token = ProcessStorageContext::ApplyQuiesceGate(std::chrono::seconds(10));
            quiesceOk.store(token.ok());
            if (token.ok()) {
                waitersPendingAtToken.store(StorageSessionTestPeer::SyncState(*holder.session).pendingOpenWaiters);
                openAttemptsAtToken.store(engine.openAttempts.load());
                engine.tokenIssued.store(true);
                // token 保持期：观察窗口（期间不得有新 Open/CRUD）。
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                openAttemptsAtRelease.store(engine.openAttempts.load());
                engine.tokenIssued.store(false); // 释放后恢复合法业务，停止观察
                (void)ProcessStorageContext::ReleaseQuiesceGate(*token.value);
            }
        });
        quiescer.join();
        for (auto& waiter : waiters) {
            waiter.join();
        }
        for (auto& caller : retryCallers) {
            caller.join();
        }
        leader.join();
        EXPECT_TRUE(quiesceOk.load()) << "round " << round;
        EXPECT_EQ(settled.load(), kWaiters + kRetryCallers) << "round " << round;
        EXPECT_EQ(waitersPendingAtToken.load(), 0U) << "round " << round;
        EXPECT_FALSE(engine.sawOpenAfterToken.load()) << "round " << round;
        EXPECT_FALSE(engine.sawExecuteAfterToken.load()) << "round " << round;
        EXPECT_EQ(openAttemptsAtRelease.load(), openAttemptsAtToken.load())
            << "round " << round << ": no new open may start while the quiesce token is held";
    }
    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(holder.session->Close().ok());
}

// 问题1（close 变体，单轮合同回归）：同样的重试等待者窗口下请求会话 Close——
// 重试等待者唤醒后必须被拒绝（closed/quiescing 重查），Close 排空后关闭引擎，
// 关闭后不得再有新 Open 进入引擎。
TEST(StorageContractTest, CloseRejectsRetryCallerWaitingForRoundConsumption)
{
    RetryDrainRaceEngine engine;
    ReviewFixSession holder;
    holder.OpenWithEngine(engine, "retry-drain-close-session");
    const auto pendingWaiters = [&holder]() {
        return StorageSessionTestPeer::SyncState(*holder.session).pendingOpenWaiters;
    };

    std::atomic<bool> leaderReturned{false};
    std::thread opener([&leaderReturned, &holder]() {
        auto table = holder.session->Table<ContractRecord>();
        if (table.ok()) {
            (void)table.value->Count(All<ContractRecord>());
        }
        leaderReturned.store(true);
    });
    while (engine.openAttempts.load() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::vector<std::thread> waiters;
    std::atomic<int> settled{0};
    for (int i = 0; i < 8; ++i) {
        waiters.emplace_back([&settled, &holder]() {
            (void)holder.session->Table<ContractRecord>();
            ++settled;
        });
    }
    std::vector<std::thread> retryCallers;
    for (int i = 0; i < 2; ++i) {
        retryCallers.emplace_back([&settled, &holder]() {
            auto first = holder.session->Table<ContractRecord>();
            ++settled;
            for (int attempt = 0; attempt < 50; ++attempt) {
                auto retried = holder.session->Table<ContractRecord>();
                if (retried.ok() || retried.code == DbCode::INVALID_STATE) {
                    return;
                }
            }
        });
    }
    while (pendingWaiters() < static_cast<std::size_t>(10)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::atomic<bool> closeReturned{false};
    std::thread closer([&closeReturned, &holder]() {
        const auto closed = holder.session->Close(std::chrono::seconds(10));
        closeReturned.store(closed.ok());
    });
    const auto gateDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!StorageSessionTestPeer::SyncState(*holder.session).sessionQuiescing
           && std::chrono::steady_clock::now() < gateDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    engine.releaseOpen.store(true);
    closer.join();
    for (auto& waiter : waiters) {
        waiter.join();
    }
    for (auto& caller : retryCallers) {
        caller.join();
    }
    opener.join();
    EXPECT_TRUE(closeReturned.load());
    EXPECT_EQ(settled.load(), 10);
    EXPECT_EQ(engine.openAttempts.load(), 1) << "no new open may start while the session is closing";
}

// 问题2观察引擎：构造计数 + 打开模式记录（schemaRole 优先级断言用）。
class ModeProbeEngine final : public FakeStorageEngine {
public:
    struct Shared {
        std::atomic<int> constructions{0};
        std::atomic<int> opens{0};
        std::atomic<int> lastOpenMode{0};
    };
    explicit ModeProbeEngine(Shared& shared) : shared_(shared) {}
    DbResult<void> Open(const EngineOpenRequest& request) override
    {
        ++shared_.opens;
        shared_.lastOpenMode.store(static_cast<int>(request.mode));
        return FakeStorageEngine::Open(request);
    }
    Shared& shared_;
};

/// 计数 runtime 装配（factory 形态构造计数；配置由用例自定义）。
struct ModeProbeRuntimeFixture {
    ModeProbeEngine::Shared state;
    StorageRuntimeConfig config;

    DbResult<void> Install()
    {
        auto& runtime = HcclSim::Storage::detail::StorageRuntimeImpl::Instance();
        runtime.RegisterEngineFactory(
            kTestFakeBackendName,
            [this](const LogicalDatabaseRuntimeConfig&) -> DbResult<std::unique_ptr<IStorageEngine>> {
                ++state.constructions;
                return {
                    DbCode::OK,
                    std::optional<std::unique_ptr<IStorageEngine>>{std::make_unique<ModeProbeEngine>(state)},
                    0,
                    {}};
            });
        runtime.UseDatabase(Ddl::kContractTestDatabase);
        runtime.UseDatabase(Ddl::kContractTestOpDatabase);
        runtime.UseDatabase(Ddl::kContractTestDynamicDatabase);
        return StorageRuntime::EnsureProcessSession(config);
    }

    void Teardown()
    {
        (void)StorageRuntime::CloseProcessSession();
        HcclSim::Storage::detail::StorageRuntimeImpl::Instance().ResetForTest();
    }
};

// 问题5：只设置全局 ReadWriteExisting、每库三字段初始化不指定角色——有效角色必须
// 继承全局 consumer（引擎以 READ_EXISTING 打开）。旧实现每库默认 OwnerPrepare
// 覆盖全局角色（RED：以 READ_WRITE_CREATE 打开）。
TEST(StorageContractTest, SchemaRoleSessionGlobalInheritedWhenDatabaseUnset)
{
    ModeProbeRuntimeFixture fixture;
    fixture.config.schemaRole = SchemaRole::ReadWriteExisting;
    fixture.config.databases.push_back({"CONTRACT_TEST_DB", kTestFakeBackendName, "role-runner"});
    fixture.config.databases.push_back({"CONTRACT_TEST_OP_DB", kTestFakeBackendName, "role-opdata"});
    ASSERT_TRUE(fixture.Install().ok());
    ASSERT_TRUE(Count<ContractRecord>(All<ContractRecord>()).ok());
    EXPECT_EQ(fixture.state.lastOpenMode.load(), static_cast<int>(StorageOpenMode::READ_EXISTING))
        << "unset per-database role must inherit the session-level ReadWriteExisting";
    fixture.Teardown();
}

// 问题5：每库显式 OwnerPrepare 覆盖全局 consumer（该库以 READ_WRITE_CREATE 打开）。
TEST(StorageContractTest, SchemaRolePerDatabaseOverridesSessionGlobal)
{
    ModeProbeRuntimeFixture fixture;
    fixture.config.schemaRole = SchemaRole::ReadWriteExisting;
    fixture.config.databases.push_back(
        {"CONTRACT_TEST_DB", kTestFakeBackendName, "role-runner", SchemaRole::OwnerPrepare});
    fixture.config.databases.push_back({"CONTRACT_TEST_OP_DB", kTestFakeBackendName, "role-opdata"});
    ASSERT_TRUE(fixture.Install().ok());
    ASSERT_TRUE(Count<ContractRecord>(All<ContractRecord>()).ok());
    EXPECT_EQ(fixture.state.lastOpenMode.load(), static_cast<int>(StorageOpenMode::READ_WRITE_CREATE))
        << "explicit per-database OwnerPrepare must override the session-level consumer role";
    fixture.Teardown();
}

// 问题5：仅角色变化（database/backend/endpoint 不变）不得错误返回幂等 OK——必须重装
// 会话并以新有效角色打开引擎。旧实现幂等比较忽略角色（RED：不重装、模式不变）。
TEST(StorageContractTest, SchemaRoleChangeReinstallsProcessSession)
{
    ModeProbeRuntimeFixture fixture;
    fixture.config.schemaRole = SchemaRole::ReadWriteExisting;
    fixture.config.databases.push_back({"CONTRACT_TEST_DB", kTestFakeBackendName, "role-runner"});
    ASSERT_TRUE(fixture.Install().ok());
    ASSERT_TRUE(Count<ContractRecord>(All<ContractRecord>()).ok());
    EXPECT_EQ(fixture.state.lastOpenMode.load(), static_cast<int>(StorageOpenMode::READ_EXISTING));
    const int constructionsBefore = fixture.state.constructions.load();

    // 同库同端点，仅全局角色变为 OwnerPrepare：有效角色改变必须重装。
    StorageRuntimeConfig ownerConfig;
    ownerConfig.schemaRole = SchemaRole::OwnerPrepare;
    ownerConfig.databases.push_back({"CONTRACT_TEST_DB", kTestFakeBackendName, "role-runner"});
    ASSERT_TRUE(StorageRuntime::EnsureProcessSession(ownerConfig).ok());
    ASSERT_TRUE(Count<ContractRecord>(All<ContractRecord>()).ok());
    EXPECT_GT(fixture.state.constructions.load(), constructionsBefore)
        << "effective role change must reinstall the process session";
    EXPECT_EQ(fixture.state.lastOpenMode.load(), static_cast<int>(StorageOpenMode::READ_WRITE_CREATE));
    fixture.Teardown();
}

// 问题5：相同有效角色重复安装保持幂等（不重建引擎）。
TEST(StorageContractTest, SchemaRoleSameEffectiveRoleStaysIdempotent)
{
    ModeProbeRuntimeFixture fixture;
    fixture.config.schemaRole = SchemaRole::ReadWriteExisting;
    fixture.config.databases.push_back({"CONTRACT_TEST_DB", kTestFakeBackendName, "role-runner"});
    ASSERT_TRUE(fixture.Install().ok());
    ASSERT_TRUE(Count<ContractRecord>(All<ContractRecord>()).ok());
    const int constructionsBefore = fixture.state.constructions.load();
    // 逐库显式写成与全局相同的角色：有效角色不变，幂等不重建。
    StorageRuntimeConfig explicitConfig;
    explicitConfig.schemaRole = SchemaRole::ReadWriteExisting;
    explicitConfig.databases.push_back(
        {"CONTRACT_TEST_DB", kTestFakeBackendName, "role-runner", SchemaRole::ReadWriteExisting});
    ASSERT_TRUE(StorageRuntime::EnsureProcessSession(explicitConfig).ok());
    EXPECT_EQ(fixture.state.constructions.load(), constructionsBefore)
        << "same effective role must stay idempotent without reinstall";
    ASSERT_TRUE(Count<ContractRecord>(All<ContractRecord>()).ok());
    fixture.Teardown();
}

// 问题2：未打开逻辑库的首次 maintenance。Session 安装后引擎构造为 0；不执行任何
// CRUD，直接 ClearStaticData——目标库应惰性构造/Open/Prepare 后成功，另一库不访问；
// maintenance 完成后 quiesce token 正确释放（再次 maintenance 立即成功，无残留
// active count）。旧实现 ExecuteReset 在惰性打开前用空 engine 拒绝（RED）。
TEST(StorageContractTest, FirstMaintenanceOnUnopenedDatabaseLazilyOpens)
{
    {
        ModeProbeRuntimeFixture fixture;
        fixture.config.databases.push_back({"CONTRACT_TEST_DB", kTestFakeBackendName, "maint-runner"});
        fixture.config.databases.push_back({"CONTRACT_TEST_OP_DB", kTestFakeBackendName, "maint-opdata"});
        ASSERT_TRUE(fixture.Install().ok());
        EXPECT_EQ(fixture.state.constructions.load(), 0) << "session install must not construct engines";

        const auto cleared = StorageRuntime::ClearStaticData("CONTRACT_TEST_DB");
        ASSERT_TRUE(cleared.ok()) << cleared.diagnostic;
        EXPECT_EQ(fixture.state.constructions.load(), 1) << "maintenance must lazily open only the target database";
        // token 释放/无残留 active count：再次 maintenance 立即成功。
        const auto clearedAgain = StorageRuntime::ClearStaticData("CONTRACT_TEST_DB");
        ASSERT_TRUE(clearedAgain.ok()) << clearedAgain.diagnostic;
        EXPECT_EQ(fixture.state.constructions.load(), 1);
        fixture.Teardown();
    }
    {
        // 新 Session 下直接 DropDynamicData：只打开动态表所在的逻辑库。
        ModeProbeRuntimeFixture fixture;
        fixture.config.databases.push_back({"CONTRACT_TEST_DB", kTestFakeBackendName, "maint-runner"});
        fixture.config.databases.push_back({"CONTRACT_TEST_DYNAMIC_DB", kTestFakeBackendName, "maint-dynamic"});
        ASSERT_TRUE(fixture.Install().ok());
        EXPECT_EQ(fixture.state.constructions.load(), 0);
        const auto dropped = StorageRuntime::DropDynamicData("CONTRACT_TEST_DYNAMIC_DB");
        ASSERT_TRUE(dropped.ok()) << dropped.diagnostic;
        EXPECT_EQ(*dropped.value, 1U);
        EXPECT_EQ(fixture.state.constructions.load(), 1) << "drop must open only the dynamic database";
        const auto droppedAgain = StorageRuntime::DropDynamicData("CONTRACT_TEST_DYNAMIC_DB");
        ASSERT_TRUE(droppedAgain.ok()) << droppedAgain.diagnostic;
        fixture.Teardown();
    }
}

// 问题6观察引擎（预创建形态）：Open 计数/Prepare 可注入失败与抛异常/Open 可注入抛异常。
class PrecreatedFailureEngine final : public FakeStorageEngine {
public:
    std::atomic<int> openCalls{0};
    std::atomic<int> prepareAttempts{0};
    std::atomic<int> closeCalls{0};
    std::atomic<int> prepareFailuresLeft{0};
    std::atomic<bool> throwAtOpen{false};

    DbResult<void> Open(const EngineOpenRequest& request) override
    {
        ++openCalls;
        if (throwAtOpen.load()) {
            throw std::runtime_error("precreated open exploded");
        }
        return FakeStorageEngine::Open(request);
    }

    DbResult<void> Prepare() override
    {
        const int attempt = ++prepareAttempts;
        if (attempt <= prepareFailuresLeft.load()) {
            return {DbCode::BACKEND_UNAVAILABLE, 0, "prepare deliberately fails"};
        }
        return OkVoid();
    }

    DbResult<void> Close() override
    {
        ++closeCalls;
        return OkVoid();
    }
};

// 问题6：预创建引擎 Open 成功、Prepare 首次失败——不得把外部所有权引擎终态 Close
// 后伪装成可重试；路由记录"连接已打开但未 Prepare"，下一轮跳过第二次 Open 只重试
// Prepare；会话最终 Close 时关闭一次。旧实现立即 Close（RED：closeCalls>0/Open 2 次）。
TEST(StorageContractTest, PrecreatedEnginePrepareFailureRetriesPrepareWithoutReopen)
{
    PrecreatedFailureEngine engine;
    engine.prepareFailuresLeft.store(1);
    ReviewFixSession holder;
    holder.OpenWithEngine(engine, "precreated-prepare-retry-session");
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());

    const auto first = Count<ContractRecord>(All<ContractRecord>());
    EXPECT_EQ(first.code, DbCode::BACKEND_UNAVAILABLE);
    EXPECT_EQ(engine.openCalls.load(), 1);
    EXPECT_EQ(engine.prepareAttempts.load(), 1);
    EXPECT_EQ(engine.closeCalls.load(), 0) << "prepare failure must not terminal-close a precreated engine";

    const auto retried = Count<ContractRecord>(All<ContractRecord>());
    EXPECT_TRUE(retried.ok()) << retried.diagnostic;
    EXPECT_EQ(engine.openCalls.load(), 1) << "retry round must skip the second Open";
    EXPECT_EQ(engine.prepareAttempts.load(), 2);
    EXPECT_EQ(engine.closeCalls.load(), 0);

    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(holder.session->Close().ok());
    EXPECT_EQ(engine.closeCalls.load(), 1) << "session close must close the precreated engine exactly once";
}

// 问题6：预创建引擎 Open 抛异常且状态不可知——返回稳定可诊断的不可重试错误，
// 后续调用不得触碰该对象（无第二次 Open）也不得返回误导性的"engine is closed"。
TEST(StorageContractTest, PrecreatedEngineOpenThrowIsPermanentlyFailed)
{
    PrecreatedFailureEngine engine;
    engine.throwAtOpen.store(true);
    ReviewFixSession holder;
    holder.OpenWithEngine(engine, "precreated-open-throw-session");
    ASSERT_TRUE(ProcessStorageContext::Install(*holder.session).ok());

    const auto first = Count<ContractRecord>(All<ContractRecord>());
    ASSERT_FALSE(first.ok());
    EXPECT_NE(first.diagnostic.find("engine open threw an exception"), std::string::npos);
    EXPECT_EQ(engine.openCalls.load(), 1);

    // 后续调用：同样的稳定失败（不重试 Open、不产生误导性 closed 诊断）。
    const auto second = Count<ContractRecord>(All<ContractRecord>());
    ASSERT_FALSE(second.ok());
    EXPECT_NE(second.diagnostic.find("engine open threw an exception"), std::string::npos)
        << "subsequent calls must return the same stable diagnosable failure";
    EXPECT_EQ(second.diagnostic.find("engine is closed"), std::string::npos);
    EXPECT_EQ(engine.openCalls.load(), 1) << "a thrown precreated Open must not be retried";

    ASSERT_TRUE(ProcessStorageContext::RemoveAndDrain().ok());
    EXPECT_TRUE(holder.session->Close().ok());
    EXPECT_EQ(engine.closeCalls.load(), 0) << "permanently failed engine must not be touched at close";
}

// 问题7：factory 返回 OK 但 optional 无值——必须转换为有限可诊断失败、唤醒同轮等待者、
// 归还在途计数、后续可按合同重试；不得解引用空 optional（旧实现 UB 挂死）。
TEST(StorageContractTest, FactoryOkWithoutEngineIsFiniteDiagnosableAndRetryable)
{
    ModeProbeRuntimeFixture fixture;
    std::atomic<int> factoryMode{0};         // 0：OK + 空 optional；1：真实引擎
    std::atomic<bool> factoryRelease{false}; // 首次 factory 调用门闩：持住打开轮供等待者登记
    auto& runtime = HcclSim::Storage::detail::StorageRuntimeImpl::Instance();
    runtime.RegisterEngineFactory(
        kTestFakeBackendName,
        [&factoryMode, &factoryRelease,
         &fixture](const LogicalDatabaseRuntimeConfig&) -> DbResult<std::unique_ptr<IStorageEngine>> {
            if (factoryMode.load() == 0) {
                while (!factoryRelease.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                return {DbCode::OK, std::nullopt, 0, {}};
            }
            ++fixture.state.constructions;
            return {
                DbCode::OK,
                std::optional<std::unique_ptr<IStorageEngine>>{std::make_unique<ModeProbeEngine>(fixture.state)},
                0,
                {}};
        });
    runtime.UseDatabase(Ddl::kContractTestDatabase);
    StorageRuntimeConfig config;
    config.databases.push_back({"CONTRACT_TEST_DB", kTestFakeBackendName, "factory-ok-empty"});
    ASSERT_TRUE(StorageRuntime::EnsureProcessSession(config).ok());

    std::atomic<int> settled{0};
    std::vector<DbResult<uint64_t>> results(3);
    std::thread leader([]() {
        (void)Count<ContractRecord>(All<ContractRecord>());
    });
    std::vector<std::thread> waiters;
    for (int i = 0; i < 3; ++i) {
        waiters.emplace_back([&results, &settled, i]() {
            results[i] = Count<ContractRecord>(All<ContractRecord>());
            ++settled;
        });
    }
    {
        // 打开轮在 factory 门闩上保持：等待者登记进同轮。
        auto lease = ProcessStorageContext::Acquire();
        ASSERT_TRUE(lease.ok());
        const auto registerDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (StorageSessionTestPeer::SyncState(lease.value->Get()).pendingOpenWaiters < 3
               && std::chrono::steady_clock::now() < registerDeadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ASSERT_EQ(StorageSessionTestPeer::SyncState(lease.value->Get()).pendingOpenWaiters, 3U);
    }
    factoryRelease.store(true); // 放行 factory：返回 OK + 空 optional
    leader.join();
    for (auto& waiter : waiters) {
        waiter.join();
    }
    EXPECT_EQ(settled.load(), 3) << "factory OK-without-value must wake all same-round waiters finitely";
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(results[i].code, DbCode::BACKEND_UNAVAILABLE) << results[i].diagnostic;
        EXPECT_NE(results[i].diagnostic.find("engine factory"), std::string::npos)
            << "empty optional must surface as a diagnosable factory failure";
    }
    {
        auto lease = ProcessStorageContext::Acquire();
        ASSERT_TRUE(lease.ok());
        EXPECT_EQ(StorageSessionTestPeer::SyncState(lease.value->Get()).pendingOpenWaiters, 0U);
    }
    // 修复 factory 后按合同重试成功（同一会话内重新构造引擎）。
    factoryMode.store(1);
    const auto retried = Count<ContractRecord>(All<ContractRecord>());
    ASSERT_TRUE(retried.ok()) << retried.diagnostic;
    EXPECT_EQ(fixture.state.constructions.load(), 1);
    fixture.Teardown();
}

// 问题7：factory 返回 OK + 空 unique_ptr——同样必须有限可诊断失败并可重试。
TEST(StorageContractTest, FactoryOkNullUniquePtrIsFiniteDiagnosableAndRetryable)
{
    ModeProbeRuntimeFixture fixture;
    std::atomic<int> factoryMode{0}; // 0：OK + 空 unique_ptr；1：真实引擎
    auto& runtime = HcclSim::Storage::detail::StorageRuntimeImpl::Instance();
    runtime.RegisterEngineFactory(
        kTestFakeBackendName,
        [&factoryMode, &fixture](const LogicalDatabaseRuntimeConfig&) -> DbResult<std::unique_ptr<IStorageEngine>> {
            if (factoryMode.load() == 0) {
                return {
                    DbCode::OK,
                    std::optional<std::unique_ptr<IStorageEngine>>{std::unique_ptr<IStorageEngine>{}},
                    0,
                    {}};
            }
            ++fixture.state.constructions;
            return {
                DbCode::OK,
                std::optional<std::unique_ptr<IStorageEngine>>{std::make_unique<ModeProbeEngine>(fixture.state)},
                0,
                {}};
        });
    runtime.UseDatabase(Ddl::kContractTestDatabase);
    StorageRuntimeConfig config;
    config.databases.push_back({"CONTRACT_TEST_DB", kTestFakeBackendName, "factory-ok-null"});
    ASSERT_TRUE(StorageRuntime::EnsureProcessSession(config).ok());

    const auto first = Count<ContractRecord>(All<ContractRecord>());
    ASSERT_FALSE(first.ok());
    EXPECT_NE(first.diagnostic.find("engine factory"), std::string::npos)
        << "null unique_ptr must surface as a diagnosable factory failure";
    {
        auto lease = ProcessStorageContext::Acquire();
        ASSERT_TRUE(lease.ok());
        EXPECT_EQ(StorageSessionTestPeer::SyncState(lease.value->Get()).pendingOpenWaiters, 0U);
    }
    factoryMode.store(1);
    const auto retried = Count<ContractRecord>(All<ContractRecord>());
    ASSERT_TRUE(retried.ok()) << retried.diagnostic;
    EXPECT_EQ(fixture.state.constructions.load(), 1);
    fixture.Teardown();
}

// 问题8观察引擎：操作互斥锁可被父进程线程持有（模拟父线程正在引擎操作中）；
// Close/析构会尝试获取该锁（旧 child reset 死锁源）。
class LockedCloseEngine final : public FakeStorageEngine {
public:
    struct Shared {
        std::atomic<int> constructions{0};
        std::atomic<int> closeCalls{0};
        std::atomic<int> lastExecutorId{0};
        std::atomic<LockedCloseEngine*> sessionEngine{nullptr}; // 会话持有的当前实例
    };
    explicit LockedCloseEngine(Shared& shared) : shared_(shared), id_(shared.constructions.load()) {}
    DbResult<uint64_t> Insert(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* record,
        TransactionBinding* transaction) override
    {
        std::lock_guard<std::mutex> lock(opMutex); // 模拟引擎操作互斥（继承后子进程不可获取）
        shared_.lastExecutorId.store(id_);
        return FakeStorageEngine::Insert(meta, scope, record, transaction);
    }

    DbResult<uint64_t>
    Count(const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where) override
    {
        std::lock_guard<std::mutex> lock(opMutex);
        shared_.lastExecutorId.store(id_);
        return FakeStorageEngine::Count(meta, scope, where);
    }

    DbResult<void> Select(
        const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
        const QueryOptions& options, ResultSink& sink) override
    {
        std::lock_guard<std::mutex> lock(opMutex);
        shared_.lastExecutorId.store(id_);
        return FakeStorageEngine::Select(meta, scope, where, options, sink);
    }
    DbResult<void> Close() override
    {
        std::lock_guard<std::mutex> lock(opMutex); // 旧实现 child reset 在此永久阻塞
        ++shared_.closeCalls;
        return OkVoid();
    }
    Shared& shared_;
    int id_{0};
    std::mutex opMutex;
};

// 问题8（收窄合同）：父进程线程持有会话引擎的操作锁时 fork——child 侧重置只切换
// 进程本地上下文状态，不得触碰任何可能由消失父线程持有的继承锁或继承对象，必须
// 有限完成并立即 _exit（child 不调用任何 Storage API：exec 前访问 Storage 不是
// 产品合同）；父进程引擎继续可用。旧实现 child 调用继承引擎 Close 永久阻塞
// （RED：waitpid 超时）。
TEST(StorageContractTest, ForkChildResetDoesNotTouchParentEngineLocks)
{
    LockedCloseEngine::Shared shared;
    StorageRuntimeConfig config;
    config.databases.push_back({"CONTRACT_TEST_DB", kTestFakeBackendName, "fork-lock-runner"});
    {
        auto& runtime = HcclSim::Storage::detail::StorageRuntimeImpl::Instance();
        runtime.RegisterEngineFactory(
            kTestFakeBackendName,
            [&shared](const LogicalDatabaseRuntimeConfig&) -> DbResult<std::unique_ptr<IStorageEngine>> {
                ++shared.constructions; // 先计数再构造（实例编号从 1 起）
                auto engine = std::make_unique<LockedCloseEngine>(shared);
                shared.sessionEngine.store(engine.get());
                return {DbCode::OK, std::optional<std::unique_ptr<IStorageEngine>>{std::move(engine)}, 0, {}};
            });
        runtime.UseDatabase(Ddl::kContractTestDatabase);
        ASSERT_TRUE(StorageRuntime::EnsureProcessSession(config).ok());
    }
    ASSERT_TRUE(Count<ContractRecord>(All<ContractRecord>()).ok());
    const int parentExecutor = shared.lastExecutorId.load();
    EXPECT_GE(parentExecutor, 1);

    // 父进程线程持有会话引擎的操作锁（引擎操作在途时 fork）。
    LockedCloseEngine* engine = shared.sessionEngine.load();
    ASSERT_NE(engine, nullptr);
    std::atomic<bool> lockHeld{false};
    std::thread lockHolder([engine, &lockHeld]() {
        std::lock_guard<std::mutex> lock(engine->opMutex);
        lockHeld.store(true);
        while (lockHeld.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    while (!lockHeld.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        // 子进程：reset 不得获取继承引擎锁或触碰继承对象；立即 _exit，不访问
        // Storage API（产品合同：fork child 在 exec 前不得访问 Storage API）。
        HcclSim::Storage::ProcessStorageContext::ResetAfterForkInChild();
        ::_exit(0);
    }
    // 父进程：child 必须有限退出（不得永久阻塞在被父线程持有的锁上）。
    int status = 0;
    bool reaped = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        const pid_t waited = ::waitpid(pid, &status, WNOHANG);
        if (waited == pid) {
            reaped = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!reaped) {
        (void)::kill(pid, SIGKILL);
        (void)::waitpid(pid, &status, 0);
    }
    // 先完成全部资源收尾（ASSERT 失败提前返回会泄漏持锁线程导致 terminate）。
    lockHeld.store(false);
    lockHolder.join();
    const bool childExitedOk = reaped && WIFEXITED(status) && WEXITSTATUS(status) == 0;

    EXPECT_TRUE(reaped) << "fork child reset must not block on inherited engine locks";
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
    if (!childExitedOk) {
        (void)StorageRuntime::CloseProcessSession();
        HcclSim::Storage::detail::StorageRuntimeImpl::Instance().ResetForTest();
        return;
    }

    // 父进程引擎继续可用（同一实例）。
    ASSERT_TRUE(Count<ContractRecord>(All<ContractRecord>()).ok());
    EXPECT_EQ(shared.lastExecutorId.load(), parentExecutor);
    (void)StorageRuntime::CloseProcessSession();
    HcclSim::Storage::detail::StorageRuntimeImpl::Instance().ResetForTest();
}

// 进程生命周期 Session 的正常退出合同（no-destroy owner）：子进程经 StorageRuntime
// 正式安装入口建立本进程会话，再以直接 Storage 原子入口使用它——模拟只直接使用
// 数据库原子接口、从不调用 aclFinalize/CloseProcessSession 的进程；随后走正常
// std::exit 退出路径（非 _exit，静态析构完整执行）。RuntimeImpl 与 ContextState 是
// no-destroy 进程生命周期对象：正常 std::exit 不会析构仍处于 installed 状态的
// StorageSession（不触发 cannot move or destroy an installed StorageSession
// invariant），进程正常退出、无 terminate/SIGABRT。不存在任何 ExitGuard——本测试
// 不声称退出时会调用 CloseProcessSession；显式关闭只属于真实 owner
// （主控/device-arm/测试/换库）。
TEST(StorageContractTest, ProcessLifetimeSessionExitsNormallyWithoutExplicitClose)
{
    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        // 子进程：独立冷状态（放弃继承会话/引擎），注册本进程引擎工厂与目录后经
        // StorageRuntime::EnsureProcessSession 安装，再以直接原子入口使用会话。
        HcclSim::Storage::ProcessStorageContext::ResetAfterForkInChild();
        auto& runtime = HcclSim::Storage::detail::StorageRuntimeImpl::Instance();
        runtime.RegisterEngineFactory(
            kTestFakeBackendName, [](const LogicalDatabaseRuntimeConfig&) -> DbResult<std::unique_ptr<IStorageEngine>> {
                return {
                    DbCode::OK,
                    std::optional<std::unique_ptr<IStorageEngine>>{std::make_unique<FakeStorageEngine>()},
                    0,
                    {}};
            });
        runtime.UseDatabase(Ddl::kContractTestDatabase);
        StorageRuntimeConfig config;
        config.databases.push_back({"CONTRACT_TEST_DB", kTestFakeBackendName, "exit-guard-runner"});
        if (!StorageRuntime::EnsureProcessSession(config).ok()) {
            ::_exit(2); // 安装失败：可区分的非零码，不落入正常退出断言
        }
        if (!Count<ContractRecord>(All<ContractRecord>()).ok()) {
            ::_exit(3); // 原子入口使用失败：同样以可区分的非零码退出
        }
        // 不显式 CloseProcessSession；正常退出路径（静态析构 + 进程 owner 兜底）。
        std::exit(0);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status)) << "child must exit normally instead of signal "
                                   << (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

// ==== 退出析构顺序回归（no-destroy 进程生命周期 owner 合同） ====

// 退出消费者的 pipe 写端（子进程内设置，父进程只读）：atexit 回调无上下文参数，
// 以进程内静态句柄传递测试结果通道。
int& ExitConsumerPipeWriteFd()
{
    static int fd = -1;
    return fd;
}

// 正常 std::exit 阶段执行的退出消费者：执行一次真实 Storage 原子读取——与真实
// rank 进程退出清理中 RaSocketListenStop 等 Db::GetById/GetOneByPred 相同的数据库
// 访问形态（不 mock StorageSession，不读布尔标志）。
void ExitOrderedStorageConsumer()
{
    const auto counted = Count<ContractRecord>(All<ContractRecord>());
    const char verdict = counted.ok() ? 0 : 1;
    if (ExitConsumerPipeWriteFd() >= 0) {
        (void)::write(ExitConsumerPipeWriteFd(), &verdict, 1);
    }
}

// 更早登记的退出消费者在会话安装后仍能于正常 std::exit 阶段访问数据库。真实 rank
// 进程中 HCCL DSO/全局静态对象先于 storage 会话构造（登记更早），C++ 退出按登记
// 逆序析构——它们在 storage 侧任何退出兜底之后才执行退出清理并访问数据库。本测试
// 以 atexit 登记模拟该顺序（atexit 回调与静态析构共享同一 LIFO 退出队列）：先登记
// 退出消费者、再经 StorageRuntime 正式安装入口安装会话并执行一次正常原子操作，
// 不显式 CloseProcessSession、正常 std::exit。旧实现 ProcessSessionExitGuard 在
// EnsureProcessSession 首次成功时才构造（登记晚于消费者）→ 析构先于消费者执行并经
// CloseProcessSession 关闭会话（REMOVED），消费者原子读取被拒（INVALID_STATE）——
// 证伪"function-local 退出 guard 是统一正常退出 owner"的假设（RED）。
TEST(StorageContractTest, EarlierRegisteredExitConsumerStillUsesStorageOnNormalExit)
{
    int fds[2] = {-1, -1};
    ASSERT_EQ(::pipe(fds), 0);
    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        ::close(fds[0]);
        // 子进程：独立冷状态（放弃继承会话/引擎），再按真实顺序先登记退出消费者。
        HcclSim::Storage::ProcessStorageContext::ResetAfterForkInChild();
        ExitConsumerPipeWriteFd() = fds[1];
        if (::atexit(ExitOrderedStorageConsumer) != 0) {
            ::_exit(4); // atexit 登记失败：可区分的非零码，不落入正常退出断言
        }
        auto& runtime = HcclSim::Storage::detail::StorageRuntimeImpl::Instance();
        runtime.RegisterEngineFactory(
            kTestFakeBackendName, [](const LogicalDatabaseRuntimeConfig&) -> DbResult<std::unique_ptr<IStorageEngine>> {
                return {
                    DbCode::OK,
                    std::optional<std::unique_ptr<IStorageEngine>>{std::make_unique<FakeStorageEngine>()},
                    0,
                    {}};
            });
        runtime.UseDatabase(Ddl::kContractTestDatabase);
        StorageRuntimeConfig config;
        config.databases.push_back({"CONTRACT_TEST_DB", kTestFakeBackendName, "exit-order-runner"});
        if (!StorageRuntime::EnsureProcessSession(config).ok()) {
            ::_exit(2); // 安装失败：可区分的非零码，不落入正常退出断言
        }
        if (!Count<ContractRecord>(All<ContractRecord>()).ok()) {
            ::_exit(3); // 原子入口使用失败：可区分的非零码
        }
        // 不显式 CloseProcessSession；正常退出路径（atexit 消费者与静态析构执行）。
        std::exit(0);
    }
    ::close(fds[1]);
    char verdict = -1;
    const ssize_t received = ::read(fds[0], &verdict, 1);
    ::close(fds[0]);
    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status)) << "child must exit normally instead of signal "
                                   << (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
    ASSERT_EQ(received, 1) << "exit consumer must report its database access outcome";
    EXPECT_EQ(verdict, 0) << "earlier-registered exit consumer must still access storage at "
                             "normal exit (session stays installed until process exit)";
}
} // namespace
