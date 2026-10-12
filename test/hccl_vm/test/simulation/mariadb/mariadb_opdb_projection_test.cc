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
 * You may refer to the License for details. You should not use this file except in compliance with the License.
 * See the License in the root of the software repository for the full text of the License.
 */

// 文件职责：OpDetail 标量查询字段投影在 MariaDB SQL 生成路径上的回归。
// 五个标量入口（QueryCurrentOpMemInfoByRank / QueryCurrentOpMemInfo /
// QueryLatestOpExpansionMode / QueryNewestOpDeatailIdByPid / QueryOpDetailIdentity）
// 读取 opDetails 表时生成的 SELECT 列清单必须不含 opDetail/opExtInfo 大 BLOB
// 列（1024p 单行 extInfo ≈ 8MiB，标量点查拖全行是纯无效传输）。证据为
// per-run 私有实例的服务端 Bytes_sent 增量：
//   1) 负基线：同一批数据全行读取（无投影）搬运 ≥ 数 MiB——BLOB 真实在库
//      且会被整行拖回；
//   2) 五个业务函数连续调用两轮（覆盖语句缓存未命中与命中路径）：总搬运
//      < 512KiB——任一未投影的单行点查即会拖回 ≥ 1.5MiB；
//   3) 返回值与写入值逐一相同（投影不改变命中行与首条语义）。
// endpoint 由 HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 提供（mariadb://user:
// password@host:port；未设置 GTEST_SKIP，已设置但不可用必须失败——与
// backend conformance fixture 同一协议，实例由测试环境自备 per-run 私有实例）。
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

#include <mysql.h>

#include "gtest/gtest.h"

#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/database_result.h"
#include "storage_backends/mariadb/engine/mariadb_backend_factory.h"
#include "storage_backends/mariadb/engine/mariadb_engine.h"
#include "storage/internal/storage_runtime_impl.h"
#include "storage/storage_runtime.h"
#include "storage/table_access.h"
#include "storage_backends/mariadb/composition/mariadb_backend_module.h"

namespace {
constexpr char kMariadbEndpointEnv[] = "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT";

struct AdminEndpoint {
    std::string host;
    std::string user;
    std::string password;
    unsigned int port{0};
};

/// 解析 mariadb://user:password@host:port（与 mariadb_engine_test 同一形态）
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

/// 服务端全局状态计数（只读本 per-run 私有实例自身的计数；失败返回 -1）
long long ServerGlobalStatus(const AdminEndpoint& ep, const char* name)
{
    MYSQL* admin = mysql_init(nullptr);
    if (admin == nullptr) {
        return -1;
    }
    if (mysql_real_connect(admin, ep.host.c_str(), ep.user.c_str(), ep.password.c_str(), nullptr, ep.port, nullptr, 0)
        == nullptr) {
        mysql_close(admin);
        return -1;
    }
    const std::string sql = std::string{"SHOW GLOBAL STATUS LIKE '"} + name + "'";
    long long value = -1;
    if (mysql_query(admin, sql.c_str()) == 0) {
        MYSQL_RES* res = mysql_store_result(admin);
        if (res != nullptr) {
            MYSQL_ROW row = mysql_fetch_row(res);
            if (row != nullptr && row[1] != nullptr) {
                value = std::atoll(row[1]);
            }
            mysql_free_result(res);
        }
    }
    mysql_close(admin);
    return value;
}

/// 播种一个通信域成员行并返回其 id（主线 opIter 分配的身份来源；RUNNER_DB
/// 与 OP_DATA_DB 同实例装配，Communicator 经公共原子接口写入）。
uint64_t SeedCommunicator(const char* name)
{
    sim::runtime::Communicator comm{};
    std::strncpy(comm.comm_id, name, sizeof(comm.comm_id) - 1);
    comm.rank_size = 1;
    comm.rank_id = 0;
    comm.device_id = 1;
    const auto inserted = HcclSim::Storage::Insert<sim::runtime::Communicator>(comm);
    EXPECT_TRUE(inserted.ok()) << inserted.diagnostic;
    return inserted.ok() ? inserted.value->value : 0U;
}

/// OP_DATA_DB 以真实 MariaDB 引擎安装进程会话（镜像 backend conformance
/// fixture 的工厂注册协议；返回 false 表示 endpoint 已配置但不可用）。
bool OpenOpDataMariadbSession(const std::string& endpointUri)
{
    HcclSim::Storage::MariadbConnectionProfile profile;
    profile.uri = endpointUri;
    profile.connectionEstablishmentTimeoutS = 3;
    profile.packetReadTimeoutS = 30;
    profile.packetWriteTimeoutS = 30;
    // MariaDB 验证固定关闭线路协议压缩：Bytes_sent 增量才是裸列体积
    profile.wireProtocolCompression = false;

    auto& runtime = HcclSim::Storage::detail::StorageRuntimeImpl::Instance();
    runtime.ResetForTest();
    runtime.RegisterEngineFactory(
        sim::backends::kMariadbBackendName,
        [profile](const HcclSim::Storage::LogicalDatabaseRuntimeConfig& config)
            -> HcclSim::Storage::DbResult<std::unique_ptr<HcclSim::Storage::IStorageEngine>> {
            if (config.database == std::string{HcclSim::Storage::Ddl::kRunnerDatabase.databaseId}) {
                return HcclSim::Storage::CreateMariadbEngine(HcclSim::Storage::Ddl::kRunnerDatabase, config, profile);
            }
            return HcclSim::Storage::CreateMariadbEngine(HcclSim::Storage::Ddl::kOpDataDatabase, config, profile);
        });
    runtime.UseDatabase(HcclSim::Storage::Ddl::kOpDataDatabase);
    // RUNNER_DB 一并登记（Communicator 身份解析经其目录路由）
    runtime.UseDatabase(HcclSim::Storage::Ddl::kRunnerDatabase);
    const HcclSim::Storage::StorageRuntimeConfig sessionConfig{
        {{std::string{HcclSim::Storage::Ddl::kOpDataDatabase.databaseId}, sim::backends::kMariadbBackendName,
          endpointUri},
         {std::string{HcclSim::Storage::Ddl::kRunnerDatabase.databaseId}, sim::backends::kMariadbBackendName,
          endpointUri}}};
    const auto ensured = HcclSim::Storage::StorageRuntime::EnsureProcessSession(sessionConfig);
    if (!ensured.ok()) {
        ADD_FAILURE() << "session install failed: " << ensured.diagnostic;
        return false;
    }
    return true;
}
} // namespace

TEST(MariadbOpDetailScalarProjection, ScalarQueriesDoNotCarryOpDetailBlobs)
{
    const char* endpointEnv = std::getenv(kMariadbEndpointEnv);
    if (endpointEnv == nullptr || endpointEnv[0] == '\0') {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "MariaDB SQL 生成路径回归需要 per-run 私有实例";
    }
    const std::string endpointUri{endpointEnv};
    AdminEndpoint admin{};
    ASSERT_TRUE(ParseAdminEndpoint(endpointUri, admin));

    ASSERT_TRUE(OpenOpDataMariadbSession(endpointUri));

    using sim::operation::OpDetailTab;
    using sim::operation::OpMemInfoTab;
    // 真实 mysqld 是跨运行共享状态：用例前清空本用例涉及的静态表
    ASSERT_TRUE(HcclSim::Storage::Clear<OpDetailTab>(HcclSim::Storage::ClearMode::RESET_SEQUENCE).ok());
    ASSERT_TRUE(HcclSim::Storage::Clear<OpMemInfoTab>(HcclSim::Storage::ClearMode::RESET_SEQUENCE).ok());

    // 种子：3 行 detail，各带 512KiB opDetail + 1MiB opExtInfo（≈4.5MiB BLOB）。
    // 伪随机填充避免任何按内容缩减体积的路径干扰"整行拖回 ≥ 数 MiB"前提。
    std::vector<uint8_t> detailBlob(512U * 1024U);
    std::vector<uint8_t> extBlob(1024U * 1024U);
    {
        uint32_t state = 0x9E3779B9U;
        for (auto& byte : detailBlob) {
            state = state * 1664525U + 1013904223U;
            byte = static_cast<uint8_t>((state >> 16U) & 0xFFU);
        }
        for (auto& byte : extBlob) {
            state = state * 1664525U + 1013904223U;
            byte = static_cast<uint8_t>((state >> 16U) & 0xFFU);
        }
    }

    struct Seed {
        uint32_t pid;
        uint32_t deviceId;
        uint32_t rankId;
        uint64_t commId;
        uint8_t mode;
        uint64_t memAddr;
    };
    const uint64_t commA = SeedCommunicator("mariadb_proj_comm_a");
    const uint64_t commB = SeedCommunicator("mariadb_proj_comm_b");
    ASSERT_NE(commA, 0U);
    ASSERT_NE(commB, 0U);
    const Seed seeds[] = {
        {11, 1, 0, commA, 2, 0xA000}, // rank 0 旧 detail
        {11, 1, 0, commA, 1, 0xB000}, // rank 0 新 detail（同 comm/device 最新）
        {12, 2, 1, commB, 0, 0xC000}, // rank 1（当前 detail：最后插入）
    };
    std::vector<uint32_t> detailIds;
    for (const auto& seed : seeds) {
        OpDetailTab detail{};
        detail.pid = seed.pid;
        detail.deviceId = seed.deviceId;
        detail.rankId = seed.rankId;
        detail.commId = seed.commId;
        detail.opExpansionMode = seed.mode;
        detail.opDetail = detailBlob;
        detail.opExtInfo = extBlob;
        ASSERT_EQ(sim::operation::InsertOpDetail(detail), 0);
        OpMemInfoTab mem{};
        mem.inputAddr = seed.memAddr;
        ASSERT_EQ(sim::operation::InsertOpMem(mem), 0);
        EXPECT_EQ(mem.opDetailId, detail.id);
        detailIds.push_back(detail.id);
    }
    ASSERT_EQ(detailIds.size(), 3U);

    // ---- 1) 负基线：全行读取搬运 ≥3MiB（BLOB 真实在库且被整行拖回）----
    const long long baseline = ServerGlobalStatus(admin, "Bytes_sent");
    ASSERT_GE(baseline, 0);
    const auto fullRows = HcclSim::Storage::Select<OpDetailTab>(HcclSim::Storage::All<OpDetailTab>());
    ASSERT_TRUE(fullRows.ok()) << fullRows.diagnostic;
    ASSERT_EQ(fullRows.value->size(), 3U);
    for (const auto& row : *fullRows.value) {
        EXPECT_EQ(row.opDetail.size(), detailBlob.size());
        EXPECT_EQ(row.opExtInfo.size(), extBlob.size());
    }
    const long long fullDelta = ServerGlobalStatus(admin, "Bytes_sent") - baseline;
    std::cout << "[opdb-projection] full-row baseline moved " << (fullDelta / 1024) << "KiB" << std::endl;
    ASSERT_GE(fullDelta, 3LL * 1024LL * 1024LL);

    // ---- 2) 五个标量业务函数连续两轮：总搬运 <512KiB，返回值逐轮正确 ----
    const long long before = ServerGlobalStatus(admin, "Bytes_sent");
    ASSERT_GE(before, 0);
    for (int round = 0; round < 2; ++round) {
        // QueryLatestOpExpansionMode：当前 detail（rank 1 种子）的模式。
        EXPECT_EQ(sim::operation::QueryLatestOpExpansionMode(), 0);

        // QueryCurrentOpMemInfoByRank：当前 detail 的 syncIter 下 rank 0 的
        // 最新 detail（id 降序首条）的 mem。
        OpMemInfoTab byRank{};
        ASSERT_EQ(sim::operation::QueryCurrentOpMemInfoByRank(0, byRank), 0);
        EXPECT_EQ(byRank.inputAddr, 0xB000U);
        EXPECT_EQ(byRank.opDetailId, detailIds[1]);

        // QueryCurrentOpMemInfo：commId+deviceId 最新 detail 的 mem。
        OpMemInfoTab byComm{};
        ASSERT_EQ(sim::operation::QueryCurrentOpMemInfo(commA, 1, byComm), 0);
        EXPECT_EQ(byComm.inputAddr, 0xB000U);
        EXPECT_EQ(byComm.opDetailId, detailIds[1]);

        // QueryNewestOpDeatailIdByPid：pid 11 的最新 detail。
        uint32_t newestId = 0;
        ASSERT_EQ(sim::operation::QueryNewestOpDeatailIdByPid(11, newestId), 0);
        EXPECT_EQ(newestId, detailIds[1]);

        // QueryOpDetailIdentity：三字段逐一等于写入值。
        uint64_t commId = 0;
        uint32_t rankId = 0;
        uint32_t deviceId = 0;
        ASSERT_EQ(sim::operation::QueryOpDetailIdentity(detailIds[2], commId, rankId, deviceId), 0);
        EXPECT_EQ(commId, commB);
        EXPECT_EQ(rankId, 1U);
        EXPECT_EQ(deviceId, 2U);
    }
    const long long scalarDelta = ServerGlobalStatus(admin, "Bytes_sent") - before;
    std::cout << "[opdb-projection] scalar queries moved " << (scalarDelta / 1024) << "KiB in 2 rounds" << std::endl;
    ASSERT_LT(scalarDelta, 512LL * 1024LL);

    // 会话收尾：排空并关闭进程会话，清理装配登记（镜像 conformance fixture）
    const auto closed = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    HcclSim::Storage::detail::StorageRuntimeImpl::Instance().ResetForTest();
    ASSERT_TRUE(closed.ok());
}

// 新状态/摘要接口的 MariaDB SQL 路径回归（本任务公共读取优化）：
// 1) QueryOpTaskCompletionByStreamId 与 QueryOpTasksByStreamId + 遍历统计
//    isDone 在同一稳定数据集上逐项一致（含空 stream 0/0、跨 stream 不串扰）；
// 2) QueryLatestOpDetailStreamByDeviceId 与完整行接口同 id/streamId；
// 3) 服务端 Bytes_sent 证据：聚合/摘要路径连续多轮不搬运 optaskMeta/
//    opDetail/opExtInfo 大 BLOB（每次全量任务扫描搬运 ≥1.5MiB 对照）；
// 4) 旧完整接口仍读回大 BLOB；聚合语句后同连接继续查询/更新成功
//    （unbuffered 结果集已排空，无 commands out of sync）。
TEST(MariadbOpDetailScalarProjection, CompletionAggregateAndStreamSummaryDoNotCarryBlobs)
{
    const char* endpointEnv = std::getenv(kMariadbEndpointEnv);
    if (endpointEnv == nullptr || endpointEnv[0] == '\0') {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置："
                        "MariaDB SQL 生成路径回归需要 per-run 私有实例";
    }
    const std::string endpointUri{endpointEnv};
    AdminEndpoint admin{};
    ASSERT_TRUE(ParseAdminEndpoint(endpointUri, admin));

    ASSERT_TRUE(OpenOpDataMariadbSession(endpointUri));

    using sim::operation::OpDetailTab;
    using sim::operation::OpTaskTab;
    // 本用例涉及的静态表先清空（真实 mysqld 是跨用例共享状态）。动态
    // OpTaskTab_P_<pid> 表由本用例首次插入惰性创建（per-run 实例内无残留），
    // 不预先 Clear——表不存在时 Clear 会以 1146 失败。
    ASSERT_TRUE(HcclSim::Storage::Clear<OpDetailTab>(HcclSim::Storage::ClearMode::RESET_SEQUENCE).ok());

    // 种子：device 5 两行 detail（不同 streamId，后者最新），各带 512KiB BLOB；
    // stream 7 六条任务（四条完成）、stream 8 一条（完成）、stream 9 空，
    // 每条任务带 256KiB optaskMeta。
    const std::vector<uint8_t> detailBlob(512U * 1024U, 0xAB);
    const std::vector<uint8_t> taskBlob(256U * 1024U, 0xEE);
    const uint64_t commId = SeedCommunicator("mariadb_proj_comm_c");
    ASSERT_NE(commId, 0U);
    OpDetailTab older{};
    older.pid = 100;
    older.rankId = 0;
    older.deviceId = 5;
    older.commId = commId;
    older.streamId = 700;
    older.opDetail = detailBlob;
    older.opExtInfo = detailBlob;
    ASSERT_EQ(sim::operation::InsertOpDetail(older), 0);
    OpDetailTab newer{};
    newer.pid = 100;
    newer.rankId = 0;
    newer.deviceId = 5;
    newer.commId = commId;
    newer.streamId = 701;
    newer.opDetail = detailBlob;
    newer.opExtInfo = detailBlob;
    ASSERT_EQ(sim::operation::InsertOpDetail(newer), 0);
    ASSERT_GT(newer.id, older.id);

    const auto seedTask = [&taskBlob](uint64_t streamId, bool done) {
        OpTaskTab task{};
        task.streamId = streamId;
        task.optaskMeta = taskBlob;
        EXPECT_EQ(sim::operation::InsertOpTask(task, false), 0);
        if (done) {
            EXPECT_EQ(sim::operation::FinishOpTask(task), 0);
        }
    };
    seedTask(7U, true);
    seedTask(7U, false);
    seedTask(7U, true);
    seedTask(7U, true);
    seedTask(7U, false);
    seedTask(7U, true);
    seedTask(8U, true);

    // ---- 1) 新聚合与旧全量扫描逐项一致 ----
    for (const uint64_t streamId : {7ULL, 8ULL, 9ULL}) {
        uint64_t total = 0;
        uint64_t done = 0;
        ASSERT_EQ(sim::operation::QueryOpTaskCompletionByStreamId(streamId, total, done), 0);
        std::vector<OpTaskTab> tasks;
        ASSERT_EQ(sim::operation::QueryOpTasksByStreamId(streamId, tasks), 0);
        uint64_t scannedDone = 0;
        for (const auto& task : tasks) {
            if (task.isDone) {
                ++scannedDone;
            }
            EXPECT_EQ(task.optaskMeta.size(), taskBlob.size()) << "完整接口必须读回 optaskMeta";
        }
        EXPECT_EQ(total, tasks.size()) << "stream " << streamId << " total 不一致";
        EXPECT_EQ(done, scannedDone) << "stream " << streamId << " done 不一致";
    }
    {
        uint64_t total = 0;
        uint64_t done = 0;
        ASSERT_EQ(sim::operation::QueryOpTaskCompletionByStreamId(7U, total, done), 0);
        EXPECT_EQ(total, 6U);
        EXPECT_EQ(done, 4U);
        ASSERT_EQ(sim::operation::QueryOpTaskCompletionByStreamId(8U, total, done), 0);
        EXPECT_EQ(total, 1U);
        EXPECT_EQ(done, 1U);
        ASSERT_EQ(sim::operation::QueryOpTaskCompletionByStreamId(9U, total, done), 0);
        EXPECT_EQ(total, 0U);
        EXPECT_EQ(done, 0U);
    }

    // ---- 2) 主流摘要与完整行接口同源 ----
    uint32_t summaryId = 0;
    uint64_t summaryStream = 0;
    ASSERT_EQ(sim::operation::QueryLatestOpDetailStreamByDeviceId(5U, summaryId, summaryStream), 0);
    EXPECT_EQ(summaryId, newer.id);
    EXPECT_EQ(summaryStream, 701U);
    OpDetailTab latestFull{};
    ASSERT_EQ(sim::operation::QueryLatestOpDetailByDeviceId(5U, latestFull), 0);
    EXPECT_EQ(latestFull.id, summaryId);
    EXPECT_EQ(latestFull.streamId, summaryStream);
    EXPECT_EQ(latestFull.opDetail.size(), detailBlob.size());
    EXPECT_EQ(latestFull.opExtInfo.size(), detailBlob.size());
    EXPECT_EQ(sim::operation::QueryLatestOpDetailStreamByDeviceId(999U, summaryId, summaryStream), -1);

    // ---- 3) Bytes_sent 证据：聚合/摘要多轮不搬运大 BLOB ----
    // 负基线：一次全量任务扫描搬运 ≥1.5MiB（6 × 256KiB optaskMeta）。
    const long long fullBefore = ServerGlobalStatus(admin, "Bytes_sent");
    ASSERT_GE(fullBefore, 0);
    {
        std::vector<OpTaskTab> tasks;
        ASSERT_EQ(sim::operation::QueryOpTasksByStreamId(7U, tasks), 0);
        ASSERT_EQ(tasks.size(), 6U);
    }
    const long long fullDelta = ServerGlobalStatus(admin, "Bytes_sent") - fullBefore;
    std::cout << "[opdb-completion] full task scan moved " << (fullDelta / 1024) << "KiB" << std::endl;
    ASSERT_GE(fullDelta, 1536LL * 1024LL);

    // 聚合/摘要连续 10 轮（覆盖语句缓存未命中与命中路径）：总搬运 <512KiB
    // ——任一次把 256KiB optaskMeta 或 512KiB opDetail 拖回都会超限。
    const long long aggregateBefore = ServerGlobalStatus(admin, "Bytes_sent");
    ASSERT_GE(aggregateBefore, 0);
    for (int round = 0; round < 10; ++round) {
        uint64_t total = 0;
        uint64_t done = 0;
        ASSERT_EQ(sim::operation::QueryOpTaskCompletionByStreamId(7U, total, done), 0);
        EXPECT_EQ(total, 6U);
        EXPECT_EQ(done, 4U);
        ASSERT_EQ(sim::operation::QueryOpTaskCompletionByStreamId(9U, total, done), 0);
        EXPECT_EQ(total, 0U);
        uint32_t id = 0;
        uint64_t stream = 0;
        ASSERT_EQ(sim::operation::QueryLatestOpDetailStreamByDeviceId(5U, id, stream), 0);
        EXPECT_EQ(id, newer.id);
    }
    const long long aggregateDelta = ServerGlobalStatus(admin, "Bytes_sent") - aggregateBefore;
    std::cout << "[opdb-completion] aggregate+summary 10 rounds moved " << (aggregateDelta / 1024) << "KiB"
              << std::endl;
    ASSERT_LT(aggregateDelta, 512LL * 1024LL);

    // ---- 4) 聚合后同连接继续查询/更新成功（结果集已排空）----
    {
        OpTaskTab probe{};
        probe.streamId = 8U;
        probe.optaskMeta = {0x01};
        ASSERT_EQ(sim::operation::InsertOpTask(probe, false), 0);
        ASSERT_EQ(sim::operation::FinishOpTask(probe), 0);
        uint64_t total = 0;
        uint64_t done = 0;
        ASSERT_EQ(sim::operation::QueryOpTaskCompletionByStreamId(8U, total, done), 0);
        EXPECT_EQ(total, 2U);
        EXPECT_EQ(done, 2U);
        OpDetailTab full{};
        ASSERT_EQ(sim::operation::QueryLatestOpDetailByDeviceId(5U, full), 0);
        EXPECT_EQ(full.opDetail.size(), detailBlob.size());
    }

    // 会话收尾：排空并关闭进程会话，清理装配登记（镜像 conformance fixture）
    const auto closed = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    HcclSim::Storage::detail::StorageRuntimeImpl::Instance().ResetForTest();
    ASSERT_TRUE(closed.ok());
}
