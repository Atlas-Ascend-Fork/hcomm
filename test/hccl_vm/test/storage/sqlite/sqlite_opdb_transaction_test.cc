/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：验证 OpDB 原子组事务语义：OpDetail+OpMem 同事务提交与主键引用传递、
// Sync 批量更新的整组原子性、Stage 失败回滚与事务中毒、事务期间即时操作被拒绝。
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "storage_composition/storage_composition.h"
#include "storage/table_access.h"
#include "storage/storage_session.h"
#include "operation_data/operation_data_ops.h"
#include "storage_backends/sqlite/composition/sqlite_backend_module.h"

namespace {

const char* kRunnerPath = "/tmp/test_sqlite_opdb_tx_runner.db";
const char* kOpDataPath = "/tmp/test_sqlite_opdb_tx_opdata.db";

class SqliteOpDbTransactionTest : public testing::Test {
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
};

TEST_F(SqliteOpDbTransactionTest, OpDetailAndMemCommitAtomicallyWithKeyPropagation)
{
    auto transaction = HcclSim::Storage::BeginTransaction();
    ASSERT_TRUE(transaction.ok()) << transaction.diagnostic;
    auto detailTable = HcclSim::Storage::AcquireTable<sim::operation::OpDetailTab>();
    ASSERT_TRUE(detailTable.ok());
    auto memTable = HcclSim::Storage::AcquireTable<sim::operation::OpMemInfoTab>();
    ASSERT_TRUE(memTable.ok());

    sim::operation::OpDetailTab detail{};
    detail.pid = 1;
    detail.rankId = 2;
    auto stagedDetail = detailTable->Insert(detail, *transaction.value);
    ASSERT_TRUE(stagedDetail.ok()) << stagedDetail.diagnostic;
    // 事务内主键引用：detail.id 已由暂存结果回填，供 OpMem 外键使用。
    EXPECT_GT(detail.id, 0U);

    sim::operation::OpMemInfoTab mem{};
    mem.opDetailId = detail.id;
    mem.inputAddr = 0x1111;
    auto stagedMem = memTable->Insert(mem, *transaction.value);
    ASSERT_TRUE(stagedMem.ok()) << stagedMem.diagnostic;

    auto committed = transaction.value->Commit();
    ASSERT_TRUE(committed.ok()) << committed.diagnostic;
    ASSERT_EQ(committed.value->items.size(), 2U);
    EXPECT_EQ(committed.value->items[0].generatedId, detail.id);
    EXPECT_EQ(committed.value->items[1].generatedId, mem.id);

    // 提交后两条记录均可读且外键一致。
    auto detailRow = detailTable->SelectOne(HcclSim::Storage::Eq(&sim::operation::OpDetailTab::id, detail.id));
    ASSERT_TRUE(detailRow.ok());
    auto memRow = memTable->SelectOne(HcclSim::Storage::Eq(&sim::operation::OpMemInfoTab::id, mem.id));
    ASSERT_TRUE(memRow.ok()) << "mem.id=" << mem.id << " detail.id=" << detail.id << " diag=" << memRow.diagnostic;
    EXPECT_EQ(memRow.value->opDetailId, detail.id);
}

TEST_F(SqliteOpDbTransactionTest, RollbackDiscardsStagedOperations)
{
    auto transaction = HcclSim::Storage::BeginTransaction();
    ASSERT_TRUE(transaction.ok());
    auto detailTable = HcclSim::Storage::AcquireTable<sim::operation::OpDetailTab>();
    ASSERT_TRUE(detailTable.ok());

    sim::operation::OpDetailTab detail{};
    detail.pid = 1;
    detail.rankId = 1;
    auto staged = detailTable->Insert(detail, *transaction.value);
    ASSERT_TRUE(staged.ok());
    EXPECT_GT(detail.id, 0U);

    auto rolled = transaction.value->Rollback();
    ASSERT_TRUE(rolled.ok()) << rolled.diagnostic;

    // 回滚后记录不可见，自增序列未被消费的行不落库。
    auto found = detailTable->SelectOne(HcclSim::Storage::Eq(&sim::operation::OpDetailTab::id, detail.id));
    ASSERT_FALSE(found.ok());
    EXPECT_EQ(found.code, HcclSim::Storage::DbCode::NOT_FOUND);
    auto rows = detailTable->Select(HcclSim::Storage::All<sim::operation::OpDetailTab>());
    ASSERT_TRUE(rows.ok());
    EXPECT_TRUE(rows.value->empty());
}

TEST_F(SqliteOpDbTransactionTest, SyncBatchUpdateCommitsAllOrNothing)
{
    auto table = HcclSim::Storage::AcquireTable<sim::operation::SyncRecordTab>();
    ASSERT_TRUE(table.ok());

    std::vector<HcclSim::Storage::PrimaryKey<sim::operation::SyncRecordTab>> keys;
    for (uint32_t index = 0; index < 3; ++index) {
        sim::operation::SyncRecordTab record{};
        record.pid = 1;
        record.rankId = index;
        record.rankSize = 1;
        record.syncIter = 1;
        record.streamId = index;
        record.status = 0;
        auto inserted = table->Insert(record);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
        keys.push_back(*inserted.value);
    }

    auto transaction = HcclSim::Storage::BeginTransaction();
    ASSERT_TRUE(transaction.ok());
    for (const auto& key : keys) {
        auto staged = table->Update(
            HcclSim::Storage::Eq(&sim::operation::SyncRecordTab::id, key.value),
            HcclSim::Storage::Set(&sim::operation::SyncRecordTab::status, uint64_t{2}), *transaction.value);
        ASSERT_TRUE(staged.ok()) << staged.diagnostic;
    }
    auto committed = transaction.value->Commit();
    ASSERT_TRUE(committed.ok()) << committed.diagnostic;
    EXPECT_EQ(committed.value->items.size(), 3U);

    auto rows = table->Select(HcclSim::Storage::All<sim::operation::SyncRecordTab>());
    ASSERT_TRUE(rows.ok());
    EXPECT_EQ(rows.value->size(), 3U);
}

TEST_F(SqliteOpDbTransactionTest, ImmediateOperationIsRejectedWhileTransactionBuilding)
{
    auto transaction = HcclSim::Storage::BeginTransaction();
    ASSERT_TRUE(transaction.ok());
    auto table = HcclSim::Storage::AcquireTable<sim::operation::OpDetailTab>();
    ASSERT_TRUE(table.ok());

    sim::operation::OpDetailTab detail{};
    auto staged = table->Insert(detail, *transaction.value);
    ASSERT_TRUE(staged.ok());

    // 事务构建期间引擎拒绝即时操作（迁移前 BEGIN IMMEDIATE 期间同连接不可并发的等价约束）。
    sim::operation::OpDetailTab immediate{};
    auto immediateResult = table->Insert(immediate);
    ASSERT_FALSE(immediateResult.ok());
    EXPECT_EQ(immediateResult.code, HcclSim::Storage::DbCode::INVALID_STATE);

    (void)transaction.value->Rollback();
}

TEST_F(SqliteOpDbTransactionTest, EmptyTransactionCommitsAndRollsBackTrivially)
{
    // 通用事务合同：未发生任何暂存的空事务提交/回滚为平凡成功，不触碰物理引擎。
    {
        auto transaction = HcclSim::Storage::BeginTransaction();
        ASSERT_TRUE(transaction.ok());
        auto committed = transaction.value->Commit();
        ASSERT_TRUE(committed.ok()) << committed.diagnostic;
        EXPECT_TRUE(committed.value->items.empty());
        EXPECT_EQ(transaction.value->State(), HcclSim::Storage::TransactionState::COMMITTED);
    }
    {
        auto transaction = HcclSim::Storage::BeginTransaction();
        ASSERT_TRUE(transaction.ok());
        auto rolled = transaction.value->Rollback();
        ASSERT_TRUE(rolled.ok()) << rolled.diagnostic;
        EXPECT_EQ(transaction.value->State(), HcclSim::Storage::TransactionState::ROLLED_BACK);
    }
}

TEST_F(SqliteOpDbTransactionTest, TransactionHandleRollsBackOnDestruction)
{
    {
        auto transaction = HcclSim::Storage::BeginTransaction();
        ASSERT_TRUE(transaction.ok());
        auto table = HcclSim::Storage::AcquireTable<sim::operation::OpDetailTab>();
        ASSERT_TRUE(table.ok());
        sim::operation::OpDetailTab detail{};
        detail.pid = 5;
        auto staged = table->Insert(detail, *transaction.value);
        ASSERT_TRUE(staged.ok());
        // 不提交：Transaction 析构自动回滚（BUILDING 状态安全兜底）。
    }
    auto table = HcclSim::Storage::AcquireTable<sim::operation::OpDetailTab>();
    ASSERT_TRUE(table.ok());
    auto rows = table->Select(HcclSim::Storage::All<sim::operation::OpDetailTab>());
    ASSERT_TRUE(rows.ok());
    EXPECT_TRUE(rows.value->empty());
}

// ---- 批量插入（Table/Transaction 公开 API）----

TEST_F(SqliteOpDbTransactionTest, InsertManyWritesBackIdsAndCommitsAtomically)
{
    auto detailTable = HcclSim::Storage::AcquireTable<sim::operation::OpDetailTab>();
    ASSERT_TRUE(detailTable.ok());
    auto transaction = HcclSim::Storage::BeginTransaction();
    ASSERT_TRUE(transaction.ok());

    std::vector<sim::operation::OpDetailTab> rows(16);
    for (uint32_t i = 0; i < rows.size(); ++i) {
        rows[i].pid = 1;
        rows[i].deviceId = i % 4U;
        rows[i].rankId = i;
        rows[i].commId = 100U;
        rows[i].streamId = 0x7000'0000'0000'0000ULL + i; // 高位 uint64
        rows[i].opDetail = std::vector<uint8_t>(i * 8U, 0x5AU);
        rows[i].opExtInfo = std::vector<uint8_t>(i % 5U, 0xC3U);
    }
    const auto inserted = detailTable->InsertMany(rows.data(), rows.size(), *transaction.value);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    EXPECT_EQ(inserted.affectedRows, rows.size());
    for (uint32_t i = 0; i < rows.size(); ++i) {
        EXPECT_GT(rows[i].id, 0U) << "row " << i;
        if (i > 0) {
            EXPECT_EQ(rows[i].id, rows[i - 1U].id + 1U) << "row " << i; // 生成主键按输入顺序写回
        }
    }
    const auto committed = transaction.value->Commit();
    ASSERT_TRUE(committed.ok()) << committed.diagnostic;
    EXPECT_EQ(committed.value->items.size(), rows.size()); // 一批 N 行计 N 个逻辑结果项

    const auto stored = detailTable->Select(HcclSim::Storage::All<sim::operation::OpDetailTab>());
    ASSERT_TRUE(stored.ok());
    ASSERT_EQ(stored.value->size(), rows.size());
    EXPECT_EQ((*stored.value)[7].opDetail, rows[7].opDetail);
    EXPECT_EQ((*stored.value)[7].streamId, rows[7].streamId);
}

TEST_F(SqliteOpDbTransactionTest, InsertManyConstraintFailureFailsTransactionAndRollsBack)
{
    auto detailTable = HcclSim::Storage::AcquireTable<sim::operation::OpDetailTab>();
    ASSERT_TRUE(detailTable.ok());
    auto memTable = HcclSim::Storage::AcquireTable<sim::operation::OpMemInfoTab>();
    ASSERT_TRUE(memTable.ok());
    auto transaction = HcclSim::Storage::BeginTransaction();
    ASSERT_TRUE(transaction.ok());

    // 同事务此前成功的单行写入：批次失败后必须随整笔回滚消失
    sim::operation::OpDetailTab detail{};
    detail.pid = 3;
    detail.deviceId = 9;
    const auto staged = detailTable->Insert(detail, *transaction.value);
    ASSERT_TRUE(staged.ok()) << staged.diagnostic;

    // mem 批次第 2 行 FK 违反（引用不存在的 opDetailId）
    std::vector<sim::operation::OpMemInfoTab> mems(4);
    for (uint32_t i = 0; i < mems.size(); ++i) {
        mems[i].opDetailId = (i == 2U) ? 999999U : detail.id;
        mems[i].inputAddr = 0x1000 + i;
    }
    const auto failed = memTable->InsertMany(mems.data(), mems.size(), *transaction.value);
    EXPECT_EQ(failed.code, HcclSim::Storage::DbCode::CONFLICT);
    EXPECT_NE(failed.diagnostic.find(" at row 2"), std::string::npos) << failed.diagnostic;
    // 事务熔断：状态 FAILED，不再接受操作
    EXPECT_EQ(transaction.value->State(), HcclSim::Storage::TransactionState::FAILED);
    sim::operation::OpDetailTab afterFailure{};
    const auto rejected = detailTable->Insert(afterFailure, *transaction.value);
    EXPECT_EQ(rejected.code, HcclSim::Storage::DbCode::INVALID_STATE);

    // 回滚生效：此前的 detail 写入与批次前两行都不在库内
    const auto details = detailTable->Select(HcclSim::Storage::All<sim::operation::OpDetailTab>());
    ASSERT_TRUE(details.ok());
    EXPECT_TRUE(details.value->empty());
    const auto memsLeft = memTable->Select(HcclSim::Storage::All<sim::operation::OpMemInfoTab>());
    ASSERT_TRUE(memsLeft.ok());
    EXPECT_TRUE(memsLeft.value->empty());

    // 失败后可正常开启新事务并操作（未遗留语句/锁）
    auto next = HcclSim::Storage::BeginTransaction();
    ASSERT_TRUE(next.ok());
    sim::operation::OpDetailTab fresh{};
    fresh.pid = 4;
    const auto freshInserted = detailTable->Insert(fresh, *next.value);
    ASSERT_TRUE(freshInserted.ok()) << freshInserted.diagnostic;
    ASSERT_TRUE(next.value->Commit().ok());
}

TEST_F(SqliteOpDbTransactionTest, InsertManyCountZeroAndOversizeContract)
{
    auto detailTable = HcclSim::Storage::AcquireTable<sim::operation::OpDetailTab>();
    ASSERT_TRUE(detailTable.ok());
    // count=0 且事务 BUILDING：平凡成功（不触碰引擎）
    {
        auto transaction = HcclSim::Storage::BeginTransaction();
        ASSERT_TRUE(transaction.ok());
        const auto empty = detailTable->InsertMany(nullptr, 0, *transaction.value);
        EXPECT_TRUE(empty.ok());
    }
    // count=0 且事务已终态：INVALID_STATE（Table 层最先检查事务终态）
    {
        auto transaction = HcclSim::Storage::BeginTransaction();
        ASSERT_TRUE(transaction.ok());
        ASSERT_TRUE(transaction.value->Commit().ok());
        const auto stale = detailTable->InsertMany(nullptr, 0, *transaction.value);
        EXPECT_EQ(stale.code, HcclSim::Storage::DbCode::INVALID_STATE);
    }
    // 超限批次：INVALID_ARGUMENT（不取引擎）
    {
        auto transaction = HcclSim::Storage::BeginTransaction();
        ASSERT_TRUE(transaction.ok());
        std::vector<sim::operation::OpDetailTab> rows(HcclSim::Storage::kInsertBatchMaxRows + 1U);
        const auto oversize = detailTable->InsertMany(rows.data(), rows.size(), *transaction.value);
        EXPECT_EQ(oversize.code, HcclSim::Storage::DbCode::INVALID_ARGUMENT);
    }
}
} // namespace
