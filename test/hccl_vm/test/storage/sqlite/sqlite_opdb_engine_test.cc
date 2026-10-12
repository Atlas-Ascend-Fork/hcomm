/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：验证 OpDB 记录在存储抽象层上的端到端语义：字段化插入与自增主键、
// BLOB 列（定长/计数前缀/整块指令空间）、条件查询与具名 JOIN 查询、字段更新。
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "storage_composition/storage_composition.h"
#include "storage/table_access.h"
#include "storage/storage_session.h"
#include "operation_data/operation_data_ops.h"
#include "storage_backends/sqlite/composition/sqlite_backend_module.h"

namespace {

const char* kRunnerPath = "/tmp/test_sqlite_opdb_engine_runner.db";
const char* kOpDataPath = "/tmp/test_sqlite_opdb_engine_opdata.db";

class SqliteOpDbEngineTest : public testing::Test {
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

    uint32_t firstDetailId_{0};
    uint32_t newestDetailId_{0};
};

TEST_F(SqliteOpDbEngineTest, InsertOpDetailAssignsIdAndRoundTripsAllColumns)
{
    auto table = HcclSim::Storage::AcquireTable<sim::operation::OpDetailTab>();
    ASSERT_TRUE(table.ok()) << table.diagnostic;

    sim::operation::OpDetailTab record{};
    record.pid = 100;
    record.rankId = 7;
    record.opIter = 3;
    record.syncIter = 2;
    record.streamId = 0x0ABBCCDD11223344ULL;
    record.root = 0;
    record.opExpansionMode = 1;
    record.devType = 2;
    record.rankSize = 8;
    record.srcRank = 0;
    record.dstRank = 7;
    record.opDetail = {0x01, 0x02, 0x03};
    record.opExtInfo = {0xAA, 0xBB};
    auto inserted = table->Insert(record);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    EXPECT_EQ(record.id, 1U);

    auto found = table->SelectOne(HcclSim::Storage::Eq(&sim::operation::OpDetailTab::id, inserted->value));
    ASSERT_TRUE(found.ok()) << found.diagnostic;
    EXPECT_EQ(found.value->pid, 100U);
    EXPECT_EQ(found.value->rankId, 7U);
    EXPECT_EQ(found.value->opIter, 3U);
    EXPECT_EQ(found.value->syncIter, 2U);
    EXPECT_EQ(found.value->streamId, 0x0ABBCCDD11223344ULL);
    EXPECT_EQ(found.value->opExpansionMode, 1U);
    EXPECT_EQ(found.value->devType, 2U);
    EXPECT_EQ(found.value->rankSize, 8U);
    EXPECT_EQ(found.value->srcRank, 0U);
    EXPECT_EQ(found.value->dstRank, 7U);
    EXPECT_EQ(found.value->opDetail, (std::vector<uint8_t>{0x01, 0x02, 0x03}));
    EXPECT_EQ(found.value->opExtInfo, (std::vector<uint8_t>{0xAA, 0xBB}));
}

// typed SQLite SELECT 经 RowSource 直接 materialize 时，零长 TEXT/BLOB 仍必须
// 形成目标对象的空拥有值；sqlite3 对零长 BLOB 可给出 nullptr 数据指针。
TEST_F(SqliteOpDbEngineTest, TypedSelectRoundTripsEmptyBlobWithoutBorrowingNull)
{
    auto table = HcclSim::Storage::AcquireTable<sim::operation::OpDetailTab>();
    ASSERT_TRUE(table.ok()) << table.diagnostic;
    sim::operation::OpDetailTab record{};
    record.pid = 7;
    record.rankId = 1;
    record.opDetail.clear();
    record.opExtInfo.clear();
    auto inserted = table->Insert(record);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    auto found = table->SelectOne(HcclSim::Storage::Eq(&sim::operation::OpDetailTab::id, inserted->value));
    ASSERT_TRUE(found.ok()) << found.diagnostic;
    EXPECT_TRUE(found.value->opDetail.empty());
    EXPECT_TRUE(found.value->opExtInfo.empty());
}

TEST_F(SqliteOpDbEngineTest, TypedInsertFailureDoesNotWriteBackGeneratedKeyOrRetainBindings)
{
    auto memoryTable = HcclSim::Storage::AcquireTable<sim::operation::OpMemInfoTab>();
    ASSERT_TRUE(memoryTable.ok()) << memoryTable.diagnostic;

    sim::operation::OpMemInfoTab rejected{};
    rejected.opDetailId = 999; // 不存在的外键目标，SQLite 必须拒绝本次即时 INSERT。
    const auto inserted = memoryTable->Insert(rejected);
    ASSERT_FALSE(inserted.ok());
    EXPECT_EQ(inserted.code, HcclSim::Storage::DbCode::CONFLICT);
    EXPECT_EQ(rejected.id, 0U);

    // 用一条带大 BLOB 的有效父记录建立 FK 目标；随后 OpMemInfo INSERT 必须复用前一条
    // 失败语句的相同 typed 形状，并把每个绑定值全部替换掉。
    auto detailTable = HcclSim::Storage::AcquireTable<sim::operation::OpDetailTab>();
    ASSERT_TRUE(detailTable.ok()) << detailTable.diagnostic;
    sim::operation::OpDetailTab parent{};
    parent.pid = 123;
    parent.rankId = 4;
    parent.opDetail.resize(1024U * 1024U);
    for (std::size_t index = 0; index < parent.opDetail.size(); ++index) {
        parent.opDetail[index] = static_cast<uint8_t>((index * 17U + 5U) & 0xFFU);
    }
    parent.opExtInfo = {0xCA, 0xFE, 0xBA, 0xBE};
    const auto parentInserted = detailTable->Insert(parent);
    ASSERT_TRUE(parentInserted.ok()) << parentInserted.diagnostic;
    ASSERT_NE(parent.id, 0U);

    sim::operation::OpMemInfoTab accepted{};
    accepted.opDetailId = parent.id;
    accepted.inputAddr = 0x1111222233334444ULL;
    accepted.inputSize = 0x1011121314151617ULL;
    accepted.outputAddr = 0x5555666677778888ULL;
    accepted.outputSize = 0x2122232425262728ULL;
    accepted.cclAddr = 0x9999AAAABBBBCCCCULL;
    accepted.cclSize = 0x3132333435363738ULL;
    const auto acceptedInserted = memoryTable->Insert(accepted);
    ASSERT_TRUE(acceptedInserted.ok()) << acceptedInserted.diagnostic;
    ASSERT_NE(accepted.id, 0U);

    const auto memory
        = memoryTable->SelectOne(HcclSim::Storage::Eq(&sim::operation::OpMemInfoTab::id, acceptedInserted->value));
    ASSERT_TRUE(memory.ok()) << memory.diagnostic;
    EXPECT_EQ(memory.value->opDetailId, parent.id);
    EXPECT_EQ(memory.value->inputAddr, accepted.inputAddr);
    EXPECT_EQ(memory.value->inputSize, accepted.inputSize);
    EXPECT_EQ(memory.value->outputAddr, accepted.outputAddr);
    EXPECT_EQ(memory.value->outputSize, accepted.outputSize);
    EXPECT_EQ(memory.value->cclAddr, accepted.cclAddr);
    EXPECT_EQ(memory.value->cclSize, accepted.cclSize);

    const auto loadedParent
        = detailTable->SelectOne(HcclSim::Storage::Eq(&sim::operation::OpDetailTab::id, parentInserted->value));
    ASSERT_TRUE(loadedParent.ok()) << loadedParent.diagnostic;
    EXPECT_EQ(loadedParent.value->pid, parent.pid);
    EXPECT_EQ(loadedParent.value->rankId, parent.rankId);
    EXPECT_EQ(loadedParent.value->opDetail, parent.opDetail);
    EXPECT_EQ(loadedParent.value->opExtInfo, parent.opExtInfo);
}

TEST_F(SqliteOpDbEngineTest, QueryOpDetailBySyncIterUsesDeclaredPredicate)
{
    auto table = HcclSim::Storage::AcquireTable<sim::operation::OpDetailTab>();
    ASSERT_TRUE(table.ok());
    for (uint32_t syncIter = 1; syncIter <= 2; ++syncIter) {
        for (uint32_t rank = 0; rank < 2; ++rank) {
            sim::operation::OpDetailTab record{};
            record.pid = 1;
            record.rankId = rank;
            record.syncIter = syncIter;
            auto inserted = table->Insert(record);
            ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
        }
    }
    auto rows = table->Select(HcclSim::Storage::Eq(&sim::operation::OpDetailTab::syncIter, uint64_t{1}));
    ASSERT_TRUE(rows.ok()) << rows.diagnostic;
    EXPECT_EQ(rows.value->size(), 2U);
    for (const auto& row : *rows.value) {
        EXPECT_EQ(row.syncIter, 1U);
    }
}

TEST_F(SqliteOpDbEngineTest, CcuChannelBlobRulesMatchLegacyBytes)
{
    auto table = HcclSim::Storage::AcquireTable<sim::operation::CcuChannelTab>();
    ASSERT_TRUE(table.ok()) << table.diagnostic;

    sim::operation::CcuChannelTab record{};
    record.channelId = 9;
    record.srcDieId = 1;
    record.dstDieId = 2;
    record.srcRankId = 3;
    record.dstRankId = 4;
    for (size_t index = 0; index < sizeof(record.leid); ++index) {
        record.leid[index] = static_cast<uint8_t>(index);
        record.reid[index] = static_cast<uint8_t>(0xFF - index);
    }
    record.protocol = 5;
    record.jettyNum = 3;
    for (uint32_t index = 0; index < record.jettyNum; ++index) {
        record.jettyId[index] = 0x1000 + index;
    }
    auto inserted = table->Insert(record);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;

    auto rows = table->Select(HcclSim::Storage::All<sim::operation::CcuChannelTab>());
    ASSERT_TRUE(rows.ok());
    ASSERT_EQ(rows.value->size(), 1U);
    const auto& loaded = rows.value->front();
    EXPECT_EQ(loaded.channelId, 9U);
    EXPECT_EQ(loaded.protocol, 5U);
    EXPECT_EQ(loaded.jettyNum, 3U);
    EXPECT_EQ(std::memcmp(loaded.leid, record.leid, sizeof(record.leid)), 0);
    EXPECT_EQ(std::memcmp(loaded.reid, record.reid, sizeof(record.reid)), 0);
    EXPECT_EQ(std::memcmp(loaded.jettyId, record.jettyId, sizeof(record.jettyId)), 0);
}

TEST_F(SqliteOpDbEngineTest, CcuInstrResStoresAndReloadsWholeInstructionSpace)
{
    auto table = HcclSim::Storage::AcquireTable<sim::operation::CcuInstrResTab>();
    ASSERT_TRUE(table.ok()) << table.diagnostic;

    sim::operation::CcuInstrResTab record{};
    record.deviceId = 4;
    record.dieId = 1;
    record.instrCount = 16;
    // 只填充头部若干指令字节，剩余为记录初值（迁移前整块 1MiB 落库语义）。
    for (size_t index = 0; index < 128; ++index) {
        record.instrSpace[0][index] = static_cast<uint8_t>(index * 3);
    }
    auto inserted = table->Insert(record);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;

    auto rows = table->Select(HcclSim::Storage::All<sim::operation::CcuInstrResTab>());
    ASSERT_TRUE(rows.ok()) << rows.diagnostic;
    ASSERT_EQ(rows.value->size(), 1U);
    const auto& loaded = rows.value->front();
    EXPECT_EQ(loaded.instrCount, 16U);
    EXPECT_EQ(std::memcmp(loaded.instrSpace, record.instrSpace, sizeof(record.instrSpace)), 0);
}

TEST_F(SqliteOpDbEngineTest, UpdateSyncRecordStatusByFieldMutation)
{
    auto table = HcclSim::Storage::AcquireTable<sim::operation::SyncRecordTab>();
    ASSERT_TRUE(table.ok());
    sim::operation::SyncRecordTab record{};
    record.pid = 1;
    record.rankId = 2;
    record.rankSize = 4;
    record.syncIter = 1;
    record.streamId = 9;
    record.status = 0;
    auto inserted = table->Insert(record);
    ASSERT_TRUE(inserted.ok());

    auto updated = table->Update(
        HcclSim::Storage::Eq(&sim::operation::SyncRecordTab::id, inserted.value->value),
        HcclSim::Storage::Set(&sim::operation::SyncRecordTab::status, uint64_t{2}));
    ASSERT_TRUE(updated.ok()) << updated.diagnostic;
    EXPECT_EQ(*updated.value, 1U);

    auto rows = table->Select(HcclSim::Storage::All<sim::operation::SyncRecordTab>());
    ASSERT_TRUE(rows.ok());
    EXPECT_EQ(rows.value->size(), 1U);
    EXPECT_EQ(rows.value->front().rankId, 2U);
}

TEST_F(SqliteOpDbEngineTest, CurrentMemByRankResolvesThroughPlainTableOperations)
{
    auto detailTable = HcclSim::Storage::AcquireTable<sim::operation::OpDetailTab>();
    ASSERT_TRUE(detailTable.ok());
    auto memTable = HcclSim::Storage::AcquireTable<sim::operation::OpMemInfoTab>();
    ASSERT_TRUE(memTable.ok());

    // syncIter=5、rankId=1 的两条 OpDetail，各挂一条 OpMemInfo；当前 detail 取最新一条。
    for (uint32_t index = 0; index < 2; ++index) {
        sim::operation::OpDetailTab detail{};
        detail.pid = 1;
        detail.rankId = 1;
        detail.syncIter = 5;
        auto detailKey = detailTable->Insert(detail);
        ASSERT_TRUE(detailKey.ok()) << detailKey.diagnostic;
        sim::operation::OpMemInfoTab mem{};
        mem.opDetailId = detail.id;
        mem.inputAddr = 0x1000 + index;
        mem.inputSize = 0x2000;
        mem.outputAddr = 0x3000 + index;
        mem.outputSize = 0x4000;
        mem.cclAddr = 0x5000 + index;
        mem.cclSize = 0x6000;
        auto memKey = memTable->Insert(mem);
        ASSERT_TRUE(memKey.ok()) << memKey.diagnostic;
        if (index == 0) {
            firstDetailId_ = detail.id;
        } else {
            newestDetailId_ = detail.id;
        }
    }

    // 业务查询形态：先验证当前 detail 属于该 rank，再按主键升序取其首条内存记录。
    const auto detail = detailTable->SelectOne(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::operation::OpDetailTab::id, newestDetailId_),
        HcclSim::Storage::Eq(&sim::operation::OpDetailTab::rankId, 1U)));
    ASSERT_TRUE(detail.ok()) << detail.diagnostic;
    const auto mem = memTable->SelectOne(
        HcclSim::Storage::Eq(&sim::operation::OpMemInfoTab::opDetailId, newestDetailId_),
        HcclSim::Storage::OrderBy(&sim::operation::OpMemInfoTab::id, HcclSim::Storage::SortDirection::ASC));
    ASSERT_TRUE(mem.ok()) << mem.diagnostic;
    EXPECT_EQ(mem.value->inputAddr, 0x1001U);
    EXPECT_EQ(mem.value->cclAddr, 0x5001U);
    EXPECT_EQ(mem.value->opDetailId, newestDetailId_);

    // detail 不存在：按空行集处理，不触碰 mem 查询。
    const auto missing = detailTable->SelectOne(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::operation::OpDetailTab::id, firstDetailId_ + 10000U),
        HcclSim::Storage::Eq(&sim::operation::OpDetailTab::rankId, 1U)));
    // detail 不存在：SelectOne 返回 NOT_FOUND，不触碰 mem 查询。
    ASSERT_FALSE(missing.ok());
    EXPECT_EQ(missing.code, HcclSim::Storage::DbCode::NOT_FOUND);

    // rank 不匹配：当前 detail 属于其他 rank 时同样按空行集拒绝。
    const auto foreignRank = detailTable->SelectOne(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::operation::OpDetailTab::id, newestDetailId_),
        HcclSim::Storage::Eq(&sim::operation::OpDetailTab::rankId, 2U)));
    // rank 不匹配：当前 detail 属于其他 rank 时未命中（NOT_FOUND）。
    ASSERT_FALSE(foreignRank.ok());
    EXPECT_EQ(foreignRank.code, HcclSim::Storage::DbCode::NOT_FOUND);
}

TEST_F(SqliteOpDbEngineTest, OpTaskDynamicTableInsertAndQueryByScope)
{
    // 动态 scope 表经公开原子接口按进程 scope 读写（pid 语义：任意 scope 显式指定）。
    sim::operation::OpTaskTab task{};
    task.opDetailId = 77;
    task.optaskMeta = {0x0A, 0x0B, 0x0C};
    auto inserted
        = HcclSim::Storage::Insert<sim::operation::OpTaskTab>(HcclSim::Storage::TableScope::Process(321), task);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    EXPECT_EQ(task.id, 1U);

    const auto rows = HcclSim::Storage::Select<sim::operation::OpTaskTab>(
        HcclSim::Storage::TableScope::Process(321),
        HcclSim::Storage::Eq(&sim::operation::OpTaskTab::opDetailId, uint64_t{77}),
        HcclSim::Storage::OrderBy(&sim::operation::OpTaskTab::id));
    ASSERT_TRUE(rows.ok()) << rows.diagnostic;
    ASSERT_EQ(rows.value->size(), 1U);
    EXPECT_EQ(rows.value->front().opDetailId, 77U);
    EXPECT_EQ(rows.value->front().optaskMeta, (std::vector<uint8_t>{0x0A, 0x0B, 0x0C}));
}
} // namespace
