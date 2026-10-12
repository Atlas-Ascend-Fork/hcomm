/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：sim::operation 业务操作入口在存储抽象层上的回归验证：
// 插入族、查询族、更新族、复合查询（动态表）与领域备份维护入口。
#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "operation_data/operation_data_ops.h"
#include "sim_common_api.h"
#include "sqlite/simulation_storage_test_helper.h"

namespace {
const std::string kRunnerPath = "/tmp/test_sqlite_opdb_ops_runner.db";
const std::string kOpDataPath = "/tmp/test_sqlite_opdb_ops_opdata.db";
const std::string kColdSwitchPath = "/tmp/test_sqlite_opdb_ops_cold_switch.db";

bool FileExists(const std::string& path)
{
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        return false;
    }
    std::fclose(file);
    return true;
}

void RemoveDatabaseFiles(const std::string& path)
{
    for (const char* suffix : {"", "-wal", "-shm"}) {
        std::remove((path + suffix).c_str());
    }
}

class SqliteOpDbOpsTest : public testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_TRUE(runnerdb_test::ResetTestSession(kRunnerPath, kOpDataPath));
        // 主线语义：InsertOpDetail/InsertOpDetailAndMem 分配 opIter 前必须解析通信域
        // 身份（(commName, commHash) 独立序列键，两侧进程天然对齐）；缺 Communicator
        // 行则拒绝插入。播种默认通信域供本用例组的 detail 插入使用。
        commId_ = SeedCommunicator("opdb_ops_comm");
        ASSERT_NE(commId_, 0U);
    }

    void TearDown() override { runnerdb_test::CleanUpDatabases(kRunnerPath, kOpDataPath); }

    /// 播种一个通信域成员行并返回其 id（主线 opIter 分配的身份来源）。
    static uint64_t SeedCommunicator(const char* name)
    {
        sim::runtime::Communicator comm{};
        std::strncpy(comm.comm_id, name, sizeof(comm.comm_id) - 1);
        comm.rank_size = 1;
        comm.rank_id = 0;
        comm.device_id = 1;
        return runnerdb_test::InsertRecord<sim::runtime::Communicator>(comm);
    }

    uint64_t commId_{0}; ///< SetUp 播种的默认通信域 id
};

TEST_F(SqliteOpDbOpsTest, EnsureProcessStorageSessionIsIdempotent)
{
    EXPECT_EQ(sim::operation::EnsureProcessStorageSession(), 0);
    EXPECT_EQ(sim::operation::EnsureProcessStorageSession(), 0);
}

TEST_F(SqliteOpDbOpsTest, InsertAndQueryOpDetailChain)
{
    sim::operation::OpDetailTab detail{};
    detail.pid = 100;
    detail.rankId = 2;
    detail.commId = commId_;
    detail.opDetail = {0x01, 0x02};
    detail.opExtInfo = {0x03};
    ASSERT_EQ(sim::operation::InsertOpDetail(detail), 0);
    EXPECT_GT(detail.id, 0U);
    EXPECT_EQ(detail.opIter, 0U);

    sim::operation::OpMemInfoTab mem{};
    mem.inputAddr = 0x100;
    mem.inputSize = 0x10;
    ASSERT_EQ(sim::operation::InsertOpMem(mem), 0);
    EXPECT_EQ(mem.opDetailId, detail.id);

    // 当前 OpMem 查询：以当前 detail 的 syncIter 定位目标 rank 的最新 detail 后取其 mem。
    sim::operation::OpMemInfoTab current{};
    ASSERT_EQ(sim::operation::QueryCurrentOpMemInfoByRank(2, current), 0);
    EXPECT_EQ(current.inputAddr, 0x100U);
    EXPECT_EQ(current.opDetailId, detail.id);
}

// 跨 rank 语义：当前 detail 属于 rank A 时，查询 rank B 返回同一 syncIter 下
// rank B 最新 detail 对应的 mem——不返回 rank A 的 mem，也不返回 rank B 在
// 其他 syncIter 下更新的 detail 的 mem。
TEST_F(SqliteOpDbOpsTest, QueryCurrentOpMemInfoByRankReturnsSameSyncIterLatestMemOfRequestedRank)
{
    // 本轮 sync：rank B 两条 detail，各带一条 mem。
    sim::operation::OpDetailTab detailB1{};
    detailB1.pid = 100;
    detailB1.rankId = 2;
    detailB1.commId = commId_;
    detailB1.opDetail = {0x01};
    ASSERT_EQ(sim::operation::InsertOpDetail(detailB1), 0);
    ASSERT_GT(detailB1.id, 0U);
    sim::operation::OpMemInfoTab memB1{};
    memB1.inputAddr = 0xB10;
    ASSERT_EQ(sim::operation::InsertOpMem(memB1), 0);
    EXPECT_EQ(memB1.opDetailId, detailB1.id);

    sim::operation::OpDetailTab detailB2{};
    detailB2.pid = 100;
    detailB2.rankId = 2;
    detailB2.commId = commId_;
    detailB2.opDetail = {0x02};
    ASSERT_EQ(sim::operation::InsertOpDetail(detailB2), 0);
    ASSERT_GT(detailB2.id, detailB1.id);
    sim::operation::OpMemInfoTab memB2{};
    memB2.inputAddr = 0xB20;
    ASSERT_EQ(sim::operation::InsertOpMem(memB2), 0);
    EXPECT_EQ(memB2.opDetailId, detailB2.id);

    // 当前 detail 属于 rank A（同一 sync 轮次内最后插入，成为当前 detail），带自己的 mem。
    sim::operation::OpDetailTab detailA{};
    detailA.pid = 100;
    detailA.rankId = 0;
    detailA.commId = commId_;
    detailA.opDetail = {0x03};
    ASSERT_EQ(sim::operation::InsertOpDetail(detailA), 0);
    ASSERT_EQ(detailA.syncIter, detailB2.syncIter);
    sim::operation::OpMemInfoTab memA{};
    memA.inputAddr = 0xA0;
    ASSERT_EQ(sim::operation::InsertOpMem(memA), 0);
    EXPECT_EQ(memA.opDetailId, detailA.id);

    // rank B 在另一 sync 轮次还有一条更新的 detail 与 mem；经表合同直接准备，
    // 保持当前 detail 指向不变。
    sim::operation::OpDetailTab detailB3{};
    detailB3.pid = 100;
    detailB3.rankId = 2;
    detailB3.syncIter = detailA.syncIter + 1U;
    detailB3.opDetail = {0x04};
    detailB3.id = static_cast<uint32_t>(runnerdb_test::InsertRecord(detailB3));
    ASSERT_GT(detailB3.id, detailB2.id);
    sim::operation::OpMemInfoTab memB3{};
    memB3.opDetailId = detailB3.id;
    memB3.inputAddr = 0xB30;
    ASSERT_GT(runnerdb_test::InsertRecord(memB3), 0U);

    // 查询 rank B：必须返回同 syncIter 下 id 最大的 detail（第二条）的 mem。
    sim::operation::OpMemInfoTab queried{};
    ASSERT_EQ(sim::operation::QueryCurrentOpMemInfoByRank(2, queried), 0);
    EXPECT_EQ(queried.opDetailId, detailB2.id);
    EXPECT_EQ(queried.inputAddr, 0xB20U);
}

TEST_F(SqliteOpDbOpsTest, InsertOpDetailAndMemIsAtomic)
{
    sim::operation::OpDetailTab detail{};
    detail.pid = 7;
    detail.rankId = 1;
    detail.commId = commId_;
    sim::operation::OpMemInfoTab mem{};
    mem.inputAddr = 0x200;
    ASSERT_EQ(sim::operation::InsertOpDetailAndMem(detail, mem), 0);
    EXPECT_GT(detail.id, 0U);
    EXPECT_EQ(mem.opDetailId, detail.id);

    std::vector<sim::operation::OpMemInfoTab> notQueried;
    (void)notQueried;
    // 事务提交后两条记录都落库。
    std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>> composite;
    ASSERT_EQ(sim::operation::QueryCompositeOpDetailBySyncIter(detail.syncIter, composite), 0);
    ASSERT_EQ(composite.size(), 1U);
    ASSERT_EQ(composite.count(1U), 1U);
    EXPECT_EQ(composite[1].front().detail.id, detail.id);
    EXPECT_EQ(composite[1].front().memInfo.inputAddr, 0x200U);
}

TEST_F(SqliteOpDbOpsTest, SyncRecordStatusFlow)
{
    sim::operation::SyncRecordTab record{};
    record.pid = 5;
    record.rankId = 6;
    record.rankSize = 8;
    record.streamId = 3;
    record.status = 0;
    ASSERT_EQ(sim::operation::InsertSyncRecord(record), 0);

    std::vector<sim::operation::SyncRecordTab> pending;
    ASSERT_EQ(sim::operation::QuerySyncRecordByStatus(0, pending), 0);
    ASSERT_EQ(pending.size(), 1U);
    EXPECT_EQ(pending.front().rankId, 6U);

    pending.front().status = 2;
    ASSERT_EQ(sim::operation::UpdateSyncRecordStatus(pending), 0);

    std::vector<sim::operation::SyncRecordTab> completed;
    ASSERT_EQ(sim::operation::QuerySyncRecordByStatus(2, completed), 0);
    ASSERT_EQ(completed.size(), 1U);
    std::vector<sim::operation::SyncRecordTab> all;
    ASSERT_EQ(sim::operation::QuerySyncRecordAll(all), 0);
    ASSERT_EQ(all.size(), 1U);
    EXPECT_EQ(all.front().status, 2U);
}

TEST_F(SqliteOpDbOpsTest, OpTaskInsertAndCompositeQueryIncludeTasks)
{
    sim::operation::OpDetailTab detail{};
    // 动态任务表按进程 pid 寻址：OpDetail 行的 pid 必须与任务写入进程一致。
    detail.pid = static_cast<uint32_t>(getpid());
    detail.rankId = 1;
    detail.commId = commId_;
    ASSERT_EQ(sim::operation::InsertOpDetail(detail), 0);

    sim::operation::OpTaskTab first{};
    first.streamId = 1;
    first.optaskMeta = {0x0A};
    ASSERT_EQ(sim::operation::InsertOpTask(first, false), 0);
    sim::operation::OpTaskTab second{};
    second.streamId = 0;
    second.optaskMeta = {0x0B};
    ASSERT_EQ(sim::operation::InsertOpTask(second, false), 0);

    std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>> composite;
    ASSERT_EQ(sim::operation::QueryCompositeOpDetailBySyncIter(detail.syncIter, composite), 0);
    ASSERT_EQ(composite.size(), 1U);
    const auto& tasks = composite[1].front().tasks;
    ASSERT_EQ(tasks.size(), 2U);
    // 按主键 id（插入顺序）返回；taskSeq 字段已随 CheckerL2 基线移除。
    EXPECT_EQ(tasks[0].optaskMeta, (std::vector<uint8_t>{0x0A}));
    EXPECT_EQ(tasks[1].optaskMeta, (std::vector<uint8_t>{0x0B}));
}

TEST_F(SqliteOpDbOpsTest, NewestOpDetailIdByPidAndExpansionMode)
{
    sim::operation::OpDetailTab detail{};
    detail.pid = 42;
    detail.rankId = 1;
    detail.opExpansionMode = 1;
    detail.commId = commId_;
    ASSERT_EQ(sim::operation::InsertOpDetail(detail), 0);

    uint32_t newestId = 0;
    ASSERT_EQ(sim::operation::QueryNewestOpDeatailIdByPid(42, newestId), 0);
    EXPECT_EQ(newestId, detail.id);

    EXPECT_EQ(sim::operation::QueryLatestOpExpansionMode(), 1);
    ASSERT_EQ(sim::operation::UpdateOpExpansionMode(2), 0);
    EXPECT_EQ(sim::operation::QueryLatestOpExpansionMode(), 2);
}

// 标量 OpDetail 查询（QueryCurrentOpMemInfoByRank / QueryCurrentOpMemInfo /
// QueryLatestOpExpansionMode / QueryNewestOpDeatailIdByPid / QueryOpDetailIdentity）
// 改为 Fields 投影读取后的返回值回归：投影只收窄行读取范围，不改变命中行与
// 返回值——首条（id 降序）语义、跨 rank 同 syncIter 语义、标量字段值逐一
// 与全行读取等价（MariaDB 生成路径的体积证据见 mariadb_opdb_projection_test）。
TEST_F(SqliteOpDbOpsTest, ScalarOpDetailQueriesKeepFullRowEquivalentValues)
{
    // rank 0 / commA / device 1：两条 detail（第二条为最新），各带 mem。
    // commA/commB 为播种通信域的真实 id（主线：InsertOpDetail 需可解析身份）。
    const uint64_t commA = SeedCommunicator("scalar_comm_a");
    const uint64_t commB = SeedCommunicator("scalar_comm_b");
    ASSERT_NE(commA, 0U);
    ASSERT_NE(commB, 0U);
    sim::operation::OpDetailTab detailR0Old{};
    detailR0Old.pid = 11;
    detailR0Old.deviceId = 1;
    detailR0Old.rankId = 0;
    detailR0Old.commId = commA;
    detailR0Old.opExpansionMode = 2;
    detailR0Old.opDetail = {0x01};
    detailR0Old.opExtInfo = {0x02};
    ASSERT_EQ(sim::operation::InsertOpDetail(detailR0Old), 0);
    sim::operation::OpMemInfoTab memR0Old{};
    memR0Old.inputAddr = 0xA000;
    ASSERT_EQ(sim::operation::InsertOpMem(memR0Old), 0);
    EXPECT_EQ(memR0Old.opDetailId, detailR0Old.id);

    sim::operation::OpDetailTab detailR0New{};
    detailR0New.pid = 11;
    detailR0New.deviceId = 1;
    detailR0New.rankId = 0;
    detailR0New.commId = commA;
    detailR0New.opExpansionMode = 1;
    detailR0New.opDetail = {0x03};
    detailR0New.opExtInfo = {0x04};
    ASSERT_EQ(sim::operation::InsertOpDetail(detailR0New), 0);
    sim::operation::OpMemInfoTab memR0New{};
    memR0New.inputAddr = 0xB000;
    ASSERT_EQ(sim::operation::InsertOpMem(memR0New), 0);
    EXPECT_EQ(memR0New.opDetailId, detailR0New.id);

    // rank 1 / comm 200 / device 2：当前 detail（最后插入），带 mem。
    sim::operation::OpDetailTab detailR1{};
    detailR1.pid = 12;
    detailR1.deviceId = 2;
    detailR1.rankId = 1;
    detailR1.commId = commB;
    detailR1.opExpansionMode = 0;
    detailR1.opDetail = {0x05};
    detailR1.opExtInfo = {0x06};
    ASSERT_EQ(sim::operation::InsertOpDetail(detailR1), 0);
    sim::operation::OpMemInfoTab memR1{};
    memR1.inputAddr = 0xC000;
    ASSERT_EQ(sim::operation::InsertOpMem(memR1), 0);
    EXPECT_EQ(memR1.opDetailId, detailR1.id);

    // QueryLatestOpExpansionMode：当前 detail（detailR1）的模式。
    EXPECT_EQ(sim::operation::QueryLatestOpExpansionMode(), 0);

    // QueryCurrentOpMemInfoByRank：当前 detail 的 syncIter 下，rank 0 的
    // 最新 detail 是 detailR0New（id 降序首条），mem 为 memR0New。
    sim::operation::OpMemInfoTab byRank{};
    ASSERT_EQ(sim::operation::QueryCurrentOpMemInfoByRank(0, byRank), 0);
    EXPECT_EQ(byRank.inputAddr, 0xB000U);
    EXPECT_EQ(byRank.opDetailId, detailR0New.id);

    // QueryCurrentOpMemInfo：commId+deviceId 命中最新 detail（detailR0New）。
    sim::operation::OpMemInfoTab byComm{};
    ASSERT_EQ(sim::operation::QueryCurrentOpMemInfo(commA, 1, byComm), 0);
    EXPECT_EQ(byComm.inputAddr, 0xB000U);
    EXPECT_EQ(byComm.opDetailId, detailR0New.id);

    // QueryNewestOpDeatailIdByPid：pid 11 的最新 detail。
    uint32_t newestId = 0;
    ASSERT_EQ(sim::operation::QueryNewestOpDeatailIdByPid(11, newestId), 0);
    EXPECT_EQ(newestId, detailR0New.id);

    // QueryOpDetailIdentity：三字段逐一等于写入值（含旧行点查）。
    uint64_t commId = 0;
    uint32_t rankId = 0;
    uint32_t deviceId = 0;
    ASSERT_EQ(sim::operation::QueryOpDetailIdentity(detailR0Old.id, commId, rankId, deviceId), 0);
    EXPECT_EQ(commId, commA);
    EXPECT_EQ(rankId, 0U);
    EXPECT_EQ(deviceId, 1U);
    ASSERT_EQ(sim::operation::QueryOpDetailIdentity(detailR1.id, commId, rankId, deviceId), 0);
    EXPECT_EQ(commId, commB);
    EXPECT_EQ(rankId, 1U);
    EXPECT_EQ(deviceId, 2U);
}

TEST_F(SqliteOpDbOpsTest, CcuChannelAndJettyMapRoundTrip)
{
    sim::operation::CcuChannelTab channel{};
    channel.channelId = 3;
    channel.srcDieId = 0;
    channel.dstDieId = 1;
    channel.srcRankId = 2;
    channel.dstRankId = 3;
    for (size_t index = 0; index < sizeof(channel.leid); ++index) {
        channel.leid[index] = static_cast<uint8_t>(index);
    }
    channel.protocol = 1;
    channel.jettyNum = 2;
    channel.jettyId[0] = 0x10;
    channel.jettyId[1] = 0x20;
    ASSERT_EQ(sim::operation::InsertCcuChannel(channel), 0);

    std::vector<sim::operation::CcuChannelTab> channels;
    ASSERT_EQ(sim::operation::QueryCcuChannelAll(channels), 0);
    ASSERT_EQ(channels.size(), 1U);
    EXPECT_EQ(channels.front().channelId, 3U);
    EXPECT_EQ(channels.front().jettyNum, 2U);
    EXPECT_EQ(channels.front().jettyId[1], 0x20U);
    EXPECT_EQ(std::memcmp(channels.front().leid, channel.leid, sizeof(channel.leid)), 0);

    sim::operation::JettyMapTab jetty{};
    jetty.srcDieId = 0;
    jetty.dstDieId = 1;
    jetty.srcRankId = 2;
    jetty.dstRankId = 3;
    jetty.protocol = 4;
    ASSERT_EQ(sim::operation::InsertJettyMap(jetty), 0);
    std::vector<sim::operation::JettyMapTab> jetties;
    ASSERT_EQ(sim::operation::QueryJettyMapAll(jetties), 0);
    ASSERT_EQ(jetties.size(), 1U);
    EXPECT_EQ(jetties.front().protocol, 4U);
}

TEST_F(SqliteOpDbOpsTest, UpdateAndInsertByCcuIdReadsModifiesWrites)
{
    uint64_t ccuId = 0;
    const uint8_t info1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    // 首次：插入新记录（main 2bc9076f5：新记录 instrCount 记 startId + instrCount）。
    ASSERT_EQ(sim::operation::UpdateAndInsertByCcuId(ccuId, 1, 2, 3, 8, 0, sizeof(info1), info1), 0);
    EXPECT_GT(ccuId, 0U);

    // 二次：命中已有记录做读改写（指令空间偏移写入；instrCount 取
    // max(已存指令数, startId + instrCount)，即 main 2bc9076f5 多批合并语义：
    // 允许指令 id 空隙，计数为绝对末端而非累加，后续批次不得使计数回退）。
    uint64_t sameId = 0;
    const uint8_t info2[4] = {0xAA, 0xBB, 0xCC, 0xDD};
    std::vector<uint8_t> mergedInstrSpace;
    uint32_t totalInstrCount = 0;
    ASSERT_EQ(
        sim::operation::UpdateAndInsertByCcuId(
            sameId, 1, 2, 3, 4, 16, sizeof(info2), info2, &mergedInstrSpace, &totalInstrCount),
        0);
    EXPECT_EQ(sameId, ccuId);

    std::vector<sim::operation::CcuInstrResTab> all;
    ASSERT_EQ(sim::operation::QueryCcuInstrResAll(all), 0);
    ASSERT_EQ(all.size(), 1U);
    // 首插 3+8=11；二次 max(11, 3+4)=11（不回退）。
    EXPECT_EQ(all.front().instrCount, 11U);
    EXPECT_EQ(totalInstrCount, 11U);
    ASSERT_EQ(mergedInstrSpace.size(), sizeof(all.front().instrSpace));
    EXPECT_EQ(all.front().instrSpace[0][0], 1U);
    // 指令空间按一维字节偏移写入：偏移 16 落在第 0 行的第 16 字节。
    EXPECT_EQ(all.front().instrSpace[0][16], 0xAAU);
    EXPECT_EQ(all.front().instrSpace[0][17], 0xBBU);
    EXPECT_EQ(all.front().instrSpace[0][18], 0xCCU);
    EXPECT_EQ(all.front().instrSpace[0][19], 0xDDU);
    // 出参为合并后的完整指令空间（与落库内容一致）。
    EXPECT_EQ(mergedInstrSpace[0], 1U);
    EXPECT_EQ(mergedInstrSpace[16], 0xAAU);
    EXPECT_EQ(mergedInstrSpace[19], 0xDDU);
}

TEST_F(SqliteOpDbOpsTest, UpdateAndInsertByCcuIdUsesByteOffsetAcrossInstrRows)
{
    // 指令空间按一维字节缓冲访问：偏移落在行中间（非行对齐）时字节必须精确落位，
    // 与迁移前一维 byte buffer 语义一致（行指针偏移会放大 32 倍，本用例可证伪）。
    uint64_t ccuId = 0;
    const uint32_t offset = 32 * 5 + 7;
    const uint8_t info[4] = {0x11, 0x22, 0x33, 0x44};
    ASSERT_EQ(sim::operation::UpdateAndInsertByCcuId(ccuId, 1, 2, 3, 4, offset, sizeof(info), info), 0);
    EXPECT_GT(ccuId, 0U);

    std::vector<sim::operation::CcuInstrResTab> all;
    ASSERT_EQ(sim::operation::QueryCcuInstrResAll(all), 0);
    ASSERT_EQ(all.size(), 1U);
    const auto& space = all.front().instrSpace;
    EXPECT_EQ(space[5][7], 0x11U);
    EXPECT_EQ(space[5][8], 0x22U);
    EXPECT_EQ(space[5][9], 0x33U);
    EXPECT_EQ(space[5][10], 0x44U);
    EXPECT_EQ(space[0][0], 0U);
    EXPECT_EQ(space[6][0], 0U);
}

TEST_F(SqliteOpDbOpsTest, UpdateAndInsertByCcuIdAcceptsExactTailBoundary)
{
    // 末端边界：offset + size 恰好等于指令空间总大小时合法。
    uint64_t ccuId = 0;
    constexpr uint32_t kTotal = 32 * 1024 * 32;
    const uint32_t tailOffset = kTotal - 4;
    const uint8_t tail[4] = {0xA1, 0xA2, 0xA3, 0xA4};
    ASSERT_EQ(sim::operation::UpdateAndInsertByCcuId(ccuId, 1, 2, 3, 4, tailOffset, sizeof(tail), tail), 0);

    std::vector<sim::operation::CcuInstrResTab> all;
    ASSERT_EQ(sim::operation::QueryCcuInstrResAll(all), 0);
    ASSERT_EQ(all.size(), 1U);
    const auto& space = all.front().instrSpace;
    EXPECT_EQ(space[32 * 1024 - 1][28], 0xA1U);
    EXPECT_EQ(space[32 * 1024 - 1][31], 0xA4U);
}

TEST_F(SqliteOpDbOpsTest, UpdateAndInsertByCcuIdRejectsOverflowingOffsets)
{
    uint64_t ccuId = 0;
    constexpr uint32_t kTotal = 32 * 1024 * 32;
    const uint8_t info[4] = {0x01, 0x02, 0x03, 0x04};

    // 普通越界：offset + size 超出空间。
    EXPECT_EQ(sim::operation::UpdateAndInsertByCcuId(ccuId, 1, 2, 3, 4, kTotal - 2, sizeof(info), info), -1);
    // 溢出保护：uint32 加法回绕后小于空间大小的组合必须被拒绝（可证伪无溢出检查）。
    EXPECT_EQ(sim::operation::UpdateAndInsertByCcuId(ccuId, 1, 2, 3, 4, 0xFFFFFFFFU, 2, info), -1);
    EXPECT_EQ(sim::operation::UpdateAndInsertByCcuId(ccuId, 1, 2, 3, 4, 2, 0xFFFFFFFFU, info), -1);
    EXPECT_EQ(ccuId, 0U);
}

TEST_F(SqliteOpDbOpsTest, SetDbConfigSwitchesOpDataFile)
{
    sim::operation::SyncRecordTab record{};
    record.pid = 1;
    record.rankId = 1;
    record.rankSize = 1;
    record.streamId = 1;
    record.status = 0;
    ASSERT_EQ(sim::operation::InsertSyncRecord(record), 0);

    sim::operation::DBConfig config;
    config.dbPath = "/tmp/test_sqlite_opdb_ops_switched.db";
    ASSERT_EQ(sim::operation::SetDbConfig(config), 0);

    std::vector<sim::operation::SyncRecordTab> out;
    ASSERT_EQ(sim::operation::QuerySyncRecordAll(out), 0);
    EXPECT_TRUE(out.empty());
    for (const char* suffix : {"", "-wal", "-shm"}) {
        std::remove((config.dbPath + suffix).c_str());
    }
}

// 冷启动回归：Runner/Checker 插件共享库首次调用 SetDbConfig 时，本进程 session 尚未安装
// （插件拥有自己的冷 storage runtime 状态）。迁移前语义：OpDbOps::Instance() 先构造
// singleton 建立默认连接，SetDbConfig 再关闭并连接 config.dbPath——冷路径必须自足。
TEST_F(SqliteOpDbOpsTest, SetDbConfigInstallsColdProcessSessionBeforeSwitch)
{
    RemoveDatabaseFiles(kColdSwitchPath);

    // 旧库哨兵：切换前写入旧测试库的记录，切换后查询不得再看到它。
    sim::operation::SyncRecordTab sentinel{};
    sentinel.pid = 1;
    sentinel.rankId = 1;
    sentinel.rankSize = 1;
    sentinel.streamId = 1;
    sentinel.status = 0;
    ASSERT_EQ(sim::operation::InsertSyncRecord(sentinel), 0);

    // 冷路径冷 Ensure 会按默认路由打开安装根下两个库：先记录既有文件，测试后只清理
    // 本次新建的文件（不触碰安装根下任何既有数据），并保证默认路由目录存在
    // （正式部署由安装布局保证 data/ 存在）。
    const std::string defaultRunner = InstallPath::ResolveToInstallRoot("data/hccl_sim.db");
    const std::string defaultOpData = InstallPath::ResolveToInstallRoot("data/hccl_vm_data.db");
    std::vector<std::string> filesCreatedByColdPath;
    for (const std::string& base : {defaultRunner, defaultOpData}) {
        for (const char* suffix : {"", "-wal", "-shm"}) {
            const std::string candidate = base + suffix;
            if (!FileExists(candidate)) {
                filesCreatedByColdPath.push_back(candidate);
            }
        }
    }
    ASSERT_FALSE(defaultRunner.empty());
    std::filesystem::create_directories(InstallPath::GetHcclVmInstallAbsPath() + "/data");

    // 1. 显式关闭当前进程会话；二次关闭被拒的公开诊断即冷状态与真实缺口签名。
    (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
    const auto cold = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    ASSERT_FALSE(cold.ok());
    ASSERT_EQ(cold.diagnostic, std::string("process session is not installed"));

    // 2. 无预装 session 状态下直接调用 SetDbConfig（旧实现在此处返回 -1）。
    sim::operation::DBConfig config;
    config.dbPath = kColdSwitchPath;
    ASSERT_EQ(sim::operation::SetDbConfig(config), 0);

    // 3. 真实类型化写入/查询：session 与目标路由同时有效；结果只含新记录——
    //    不是旧测试库（那里还有 rankId=1 哨兵记录），也不是默认安装库（那是空库）。
    sim::operation::SyncRecordTab afterSwitch{};
    afterSwitch.pid = 2;
    afterSwitch.rankId = 2;
    afterSwitch.rankSize = 1;
    afterSwitch.streamId = 2;
    afterSwitch.status = 0;
    ASSERT_EQ(sim::operation::InsertSyncRecord(afterSwitch), 0);
    std::vector<sim::operation::SyncRecordTab> rows;
    ASSERT_EQ(sim::operation::QuerySyncRecordAll(rows), 0);
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows.front().rankId, 2U);

    // 4. config.dbPath 对应的新物理库已建立。
    ASSERT_TRUE(FileExists(kColdSwitchPath));

    // 5. 清理本测试创建的 db/wal/shm（含冷路径默认路由新建文件）。
    RemoveDatabaseFiles(kColdSwitchPath);
    for (const std::string& path : filesCreatedByColdPath) {
        std::remove(path.c_str());
    }
}

TEST_F(SqliteOpDbOpsTest, OpDbOpsBackupProducesFile)
{
    sim::operation::SyncRecordTab record{};
    record.pid = 1;
    record.rankId = 1;
    record.rankSize = 1;
    record.streamId = 1;
    record.status = 0;
    ASSERT_EQ(sim::operation::InsertSyncRecord(record), 0);

    const std::string backupPath = "/tmp/test_sqlite_opdb_ops_backup.db";
    std::remove(backupPath.c_str());
    EXPECT_EQ(sim::operation::BackupOperationData(backupPath), HcclSim::HCCL_SIM_SUCCESS);
    // 备份产物非空。
    FILE* file = std::fopen(backupPath.c_str(), "rb");
    ASSERT_NE(file, nullptr);
    std::fseek(file, 0, SEEK_END);
    EXPECT_GT(std::ftell(file), 0);
    std::fclose(file);
    std::remove(backupPath.c_str());
    std::remove((backupPath + "-wal").c_str());
    std::remove((backupPath + "-shm").c_str());
}

// 新状态/摘要接口与旧完整扫描的逐项对照（同一稳定数据集）：
// 1) QueryOpTaskCompletionByStreamId 的 total/done 与 QueryOpTasksByStreamId +
//    遍历统计 isDone 完全一致（含空 stream 的 0/0 与跨 stream 不串扰）；
// 2) QueryLatestOpDetailStreamByDeviceId 与完整行接口返回同一 id/streamId，
//    空表设备同样 -1；
// 3) 旧完整接口仍读回大 BLOB（optaskMeta/opDetail）。
TEST_F(SqliteOpDbOpsTest, OpTaskCompletionAndStreamSummaryMatchFullScan)
{
    // 种子 detail：device 5 两行（不同 streamId，后者为最新），带大 BLOB。
    sim::operation::OpDetailTab older{};
    older.pid = 100;
    older.rankId = 0;
    older.deviceId = 5;
    older.commId = commId_;
    older.streamId = 700;
    older.opDetail.assign(256U * 1024U, 0xAB);
    older.opExtInfo.assign(64U * 1024U, 0xCD);
    ASSERT_EQ(sim::operation::InsertOpDetail(older), 0);
    sim::operation::OpDetailTab newer{};
    newer.pid = 100;
    newer.rankId = 0;
    newer.deviceId = 5;
    newer.commId = commId_;
    newer.streamId = 701;
    newer.opDetail.assign(256U * 1024U, 0xAB);
    newer.opExtInfo.assign(64U * 1024U, 0xCD);
    ASSERT_EQ(sim::operation::InsertOpDetail(newer), 0);
    ASSERT_GT(newer.id, older.id);

    // 种子任务：stream 7 三条（两条完成）、stream 8 一条（完成）、stream 9 空；
    // 每条带 128KiB optaskMeta（完整接口必须能读回）。
    const auto seedTask = [](uint64_t streamId, bool done) {
        sim::operation::OpTaskTab task{};
        task.streamId = streamId;
        task.optaskMeta.assign(128U * 1024U, 0xEE);
        EXPECT_EQ(sim::operation::InsertOpTask(task, false), 0);
        if (done) {
            EXPECT_EQ(sim::operation::FinishOpTask(task), 0);
        }
    };
    seedTask(7U, true);
    seedTask(7U, false);
    seedTask(7U, true);
    seedTask(8U, true);

    for (const uint64_t streamId : {7ULL, 8ULL, 9ULL}) {
        uint64_t total = 0;
        uint64_t done = 0;
        ASSERT_EQ(sim::operation::QueryOpTaskCompletionByStreamId(streamId, total, done), 0);
        std::vector<sim::operation::OpTaskTab> tasks;
        ASSERT_EQ(sim::operation::QueryOpTasksByStreamId(streamId, tasks), 0);
        uint64_t scannedDone = 0;
        for (const auto& task : tasks) {
            if (task.isDone) {
                ++scannedDone;
            }
            EXPECT_EQ(task.optaskMeta.size(), 128U * 1024U) << "完整接口必须读回 optaskMeta";
        }
        EXPECT_EQ(total, tasks.size()) << "stream " << streamId << " total 不一致";
        EXPECT_EQ(done, scannedDone) << "stream " << streamId << " done 不一致";
    }
    {
        uint64_t total = 99;
        uint64_t done = 99;
        ASSERT_EQ(sim::operation::QueryOpTaskCompletionByStreamId(7U, total, done), 0);
        EXPECT_EQ(total, 3U);
        EXPECT_EQ(done, 2U);
        ASSERT_EQ(sim::operation::QueryOpTaskCompletionByStreamId(8U, total, done), 0);
        EXPECT_EQ(total, 1U);
        EXPECT_EQ(done, 1U);
        ASSERT_EQ(sim::operation::QueryOpTaskCompletionByStreamId(9U, total, done), 0);
        EXPECT_EQ(total, 0U);
        EXPECT_EQ(done, 0U);
    }

    // 主流摘要与完整行接口同源：同 id/streamId；空表设备同样 -1。
    uint32_t summaryId = 0;
    uint64_t summaryStream = 0;
    ASSERT_EQ(sim::operation::QueryLatestOpDetailStreamByDeviceId(5U, summaryId, summaryStream), 0);
    EXPECT_EQ(summaryId, newer.id);
    EXPECT_EQ(summaryStream, 701U);
    sim::operation::OpDetailTab latestFull{};
    ASSERT_EQ(sim::operation::QueryLatestOpDetailByDeviceId(5U, latestFull), 0);
    EXPECT_EQ(latestFull.id, summaryId);
    EXPECT_EQ(latestFull.streamId, summaryStream);
    EXPECT_EQ(latestFull.opDetail.size(), 256U * 1024U) << "完整行接口必须读回 opDetail";
    EXPECT_EQ(latestFull.opExtInfo.size(), 64U * 1024U) << "完整行接口必须读回 opExtInfo";
    EXPECT_EQ(sim::operation::QueryLatestOpDetailStreamByDeviceId(999U, summaryId, summaryStream), -1);
}

// A1 索引形态补充：高位 uint64 streamId、0/全部完成状态、多 PID 动态表隔离
// （另一 pid scope 的同 streamId 任务不串入本进程统计）。
TEST_F(SqliteOpDbOpsTest, OpTaskCompletionHighStreamIdsAndScopeIsolation)
{
    constexpr uint64_t kHighStream = 0xFFFF'0000'0000'0011ULL;
    constexpr uint64_t kAllDoneStream = 0xFFFF'0000'0000'0022ULL;
    const auto seedTask = [](uint64_t streamId, bool done) {
        sim::operation::OpTaskTab task{};
        task.streamId = streamId;
        task.optaskMeta = {0x01};
        EXPECT_EQ(sim::operation::InsertOpTask(task, false), 0);
        if (done) {
            EXPECT_EQ(sim::operation::FinishOpTask(task), 0);
        }
    };
    seedTask(kHighStream, false); // 0/1 完成
    seedTask(kAllDoneStream, true);
    seedTask(kAllDoneStream, true); // 2/2 全部完成

    uint64_t total = 0;
    uint64_t done = 0;
    ASSERT_EQ(sim::operation::QueryOpTaskCompletionByStreamId(kHighStream, total, done), 0);
    EXPECT_EQ(total, 1U);
    EXPECT_EQ(done, 0U);
    ASSERT_EQ(sim::operation::QueryOpTaskCompletionByStreamId(kAllDoneStream, total, done), 0);
    EXPECT_EQ(total, 2U);
    EXPECT_EQ(done, 2U);

    // 多 PID 动态表隔离：另一 pid scope 的同 streamId 任务不影响本进程统计
    const uint64_t otherPid = 424242U;
    sim::operation::OpTaskTab foreign{};
    foreign.pid = static_cast<uint32_t>(otherPid);
    foreign.streamId = kHighStream;
    foreign.optaskMeta = {0x02};
    ASSERT_TRUE(
        HcclSim::Storage::Insert<sim::operation::OpTaskTab>(HcclSim::Storage::TableScope::Process(otherPid), foreign)
            .ok());
    const auto foreignCounted = HcclSim::Storage::CountMatches<sim::operation::OpTaskTab>(
        HcclSim::Storage::TableScope::Process(otherPid),
        HcclSim::Storage::Eq(&sim::operation::OpTaskTab::streamId, kHighStream),
        HcclSim::Storage::Ne(&sim::operation::OpTaskTab::isDone, false));
    ASSERT_TRUE(foreignCounted.ok()) << foreignCounted.diagnostic;
    ASSERT_TRUE(foreignCounted.value.has_value());
    EXPECT_EQ(foreignCounted.value->total, 1U);
    EXPECT_EQ(foreignCounted.value->matched, 0U);
    ASSERT_EQ(sim::operation::QueryOpTaskCompletionByStreamId(kHighStream, total, done), 0);
    EXPECT_EQ(total, 1U);
    EXPECT_EQ(done, 0U);
}

// A2 索引形态补充：多设备交错插入、目标设备较早已停止插入（其最大 id 非全局
// 最大）、目标设备不存在、streamId 更新后仍取该设备最大 id 记录的新值。
TEST_F(SqliteOpDbOpsTest, LatestOpDetailStreamInterleavedDevicesAndAfterMainStreamUpdate)
{
    const auto seedDetail = [&](uint32_t deviceId, uint64_t streamId) {
        sim::operation::OpDetailTab detail{};
        detail.pid = 100;
        detail.rankId = 0;
        detail.deviceId = deviceId;
        detail.commId = commId_;
        detail.streamId = streamId;
        detail.opDetail = {0x01};
        EXPECT_EQ(sim::operation::InsertOpDetail(detail), 0);
        return detail;
    };
    const auto a1 = seedDetail(3U, 300U);
    const auto b1 = seedDetail(8U, 800U);
    const auto c1 = seedDetail(5U, 500U);
    const auto b2 = seedDetail(8U, 801U);
    const auto c2 = seedDetail(5U, 501U);
    ASSERT_GT(b2.id, b1.id);
    ASSERT_GT(c2.id, c1.id);

    uint32_t id = 0;
    uint64_t stream = 0;
    // 设备 3 较早停止插入：必须跨过其后其他设备的多条记录取 a1
    ASSERT_EQ(sim::operation::QueryLatestOpDetailStreamByDeviceId(3U, id, stream), 0);
    EXPECT_EQ(id, a1.id);
    EXPECT_EQ(stream, 300U);
    ASSERT_EQ(sim::operation::QueryLatestOpDetailStreamByDeviceId(8U, id, stream), 0);
    EXPECT_EQ(id, b2.id);
    EXPECT_EQ(stream, 801U);
    ASSERT_EQ(sim::operation::QueryLatestOpDetailStreamByDeviceId(5U, id, stream), 0);
    EXPECT_EQ(id, c2.id);
    EXPECT_EQ(stream, 501U);
    EXPECT_EQ(sim::operation::QueryLatestOpDetailStreamByDeviceId(777U, id, stream), -1);

    // 主流更新后：仍取该设备最大 id 的记录，但 streamId 为新值
    ASSERT_EQ(sim::operation::UpdateOpDetailMainStream(c2.id, 999U), 0);
    ASSERT_EQ(sim::operation::QueryLatestOpDetailStreamByDeviceId(5U, id, stream), 0);
    EXPECT_EQ(id, c2.id);
    EXPECT_EQ(stream, 999U);
}
} // namespace
