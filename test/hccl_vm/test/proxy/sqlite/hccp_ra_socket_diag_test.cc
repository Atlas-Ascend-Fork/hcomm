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

// RaSocketPair 生产链首错诊断与合同回归（镜像 hccp_ra_socket_async_test 的
// 会话/设备/真实套接字装配）：
//   1) Add 失败：返回 -1（合同不变），日志带 tag/tagHash(hex+dec)/slotIdx/code/diagnostic；
//   2) 末次 close 删除 1 行 vs 行已不在（0 行）两形态可区分，返回值与关闭顺序不变；
//   3) RaGetSockets 轮询：真 NOT_FOUND → 随后插入 → 成功（connectedNum 合同）；
//   4) 有界慢用例（唯一，约 93s）：后端失败→NOT_FOUND→180 次超时，最终摘要
//      保留首个后端错误（firstBackendCode=4 与 lastCode=1 并存），返回 0 /
//      connectedNum=0 合同不变（stdout 捕获校验摘要字段）。
// 等待协议本身（180 次 × 500ms、continue、connectedNum、return 0）不改。
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>

#include <gtest/gtest.h>

#include "hccp_async.h"
#include "hccp_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/storage_runtime.h"
#include "storage_composition/storage_composition.h"
#include "store_sim_resource_root.h"
#include "store_sim_comm_memory_manager.h"
#include "sqlite/simulation_storage_test_helper.h"
#include "sim_log.h"

// 同步族声明（与 hccp_ra_socket_stub.cc 实现签名一致，镜像 async 测试文件口径）
extern "C" {
int RaSocketInit(int mode, struct rdev rdevInfo, void** socketHandle);
int RaSocketDeinit(void* socketHandle);
int RaSocketBatchConnect(struct SocketConnectInfoT conn[], unsigned int num);
int RaGetSockets(unsigned int role, struct SocketInfoT conn[], unsigned int num, unsigned int* connectedNum);
int RaSocketBatchClose(struct SocketCloseInfoT conn[], unsigned int num);
int SimRaCustomChannel(void* tlvHandle, struct TlvMsg* sendMsg, struct TlvMsg* recvMsg);
}

int SimRaCustomChannel(void* tlvHandle, struct TlvMsg* sendMsg, struct TlvMsg* recvMsg)
{
    (void)tlvHandle;
    (void)sendMsg;
    (void)recvMsg;
    return -1;
}

extern uint64_t g_cur_server_key;

namespace {
constexpr char kRunnerDb[] = "/tmp/hccp_ra_socket_diag_runner.db";
constexpr char kOpDataDb[] = "/tmp/hccp_ra_socket_diag_opdata.db";
constexpr unsigned int kRaRoleServer = 0;
constexpr unsigned int kRaRoleClient = 1;

void RemoveDatabaseFiles(const char* path)
{
    std::remove(path);
    std::remove((std::string(path) + "-wal").c_str());
    std::remove((std::string(path) + "-shm").c_str());
}

void MakeEidBytes(uint8_t eid[16], uint8_t tail)
{
    std::memset(eid, 0, 16);
    eid[15] = tail;
}

/// stdout 捕获守卫（sim_log 控制台 sink 写 stdout）：窗口内 dup2 到临时文件
class StdoutCapture {
public:
    bool Start()
    {
        filePath_ = "/tmp/hccp_ra_diag_stdout_" + std::to_string(::getpid()) + ".log";
        captured_ = ::open(filePath_.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
        if (captured_ < 0) {
            return false;
        }
        ::fflush(stdout);
        saved_ = ::dup(1);
        if (::dup2(captured_, 1) < 0) {
            ::close(captured_);
            captured_ = -1;
            return false;
        }
        return true;
    }
    std::string StopAndRead()
    {
        ::fflush(stdout);
        if (captured_ < 0) {
            return {};
        }
        ::dup2(saved_, 1);
        ::close(saved_);
        ::close(captured_);
        captured_ = -1;
        std::FILE* in = std::fopen(filePath_.c_str(), "r");
        if (in == nullptr) {
            return {};
        }
        std::string text;
        char buffer[4096];
        size_t n = 0;
        while ((n = std::fread(buffer, 1, sizeof(buffer), in)) > 0) {
            text.append(buffer, n);
        }
        std::fclose(in);
        ::remove(filePath_.c_str());
        return text;
    }

private:
    int captured_{-1};
    int saved_{-1};
    std::string filePath_;
};

/// 计算与被测实现一致的 tag_hash（std::hash<std::string>，镜像
/// hccp_ra_socket_stub.cc 的 ComputeTagHash——同头文件同实现，无手工重推）
uint64_t ComputeTagHashForTest(const char* tag)
{
    return static_cast<uint64_t>(std::hash<std::string>{}(std::string(tag)));
}

uint64_t CountPairsByTagHash(uint64_t tagHash)
{
    auto rows = runnerdb_test::SelectRecords<sim::runtime::RaSocketPair>(
        HcclSim::Storage::Eq(&sim::runtime::RaSocketPair::tag_hash, tagHash));
    return rows.size();
}
} // namespace

class HccpRaSocketDiagTest : public testing::Test {
protected:
    void SetUp() override
    {
        // 日志初始化（consoleLevel=2：error/info 到 stdout——本套件用 stdout
        // 捕获校验失败日志字段；未初始化时宏静默无输出）
        static std::once_flag loggerOnce;
        std::call_once(loggerOnce, [] {
            LogConfig config;
            config.consoleLevel = 2;
            config.fileLevel = 4; // 测试不留文件日志
            InitLogger(config);
        });

        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
        RemoveDatabaseFiles(kRunnerDb);
        RemoveDatabaseFiles(kOpDataDb);
        HcclSim::Storage::StorageRuntimeConfig config;
        config.databases.push_back({std::string{"RUNNER_DB"}, "sqlite", kRunnerDb});
        config.databases.push_back({std::string{"OP_DATA_DB"}, "sqlite", kOpDataDb});
        ASSERT_TRUE(sim::EnsureSimulatorStorageSession(config).ok());
        g_cur_server_key = 1;
        ASSERT_TRUE(sim::SimResourceRoot::GetInstance().Init());

        // 每个用例独立 tag（g_tagCache 进程级缓存，旧条目不可复用）
        static uint32_t tagSeq = 0;
        std::snprintf(tag_, sizeof(tag_), "ut_diag_tag_%u", tagSeq++);

        // 两台设备 + 两个 EndPoint + 两只真实套接字（A=client，B=server）
        uint8_t eidA[16];
        uint8_t eidB[16];
        MakeEidBytes(eidA, 0x01);
        MakeEidBytes(eidB, 0x02);
        sim::runtime::Device devA{};
        devA.server_id = 1;
        devA.logic_id = 0;
        devA.physical_id = 0;
        auto devAAdded = sim::runtime::Db::Add<sim::runtime::Device>(devA);
        ASSERT_TRUE(devAAdded.ok() && devAAdded.value.has_value());
        sim::runtime::Device devB{};
        devB.server_id = 1;
        devB.logic_id = 1;
        devB.physical_id = 1;
        auto devBAdded = sim::runtime::Db::Add<sim::runtime::Device>(devB);
        ASSERT_TRUE(devBAdded.ok() && devBAdded.value.has_value());
        sim::runtime::EndPoint epA{};
        epA.device_id = devAAdded.value->value;
        std::memcpy(epA.eid, eidA, sizeof(epA.eid));
        ASSERT_TRUE(sim::runtime::Db::Add<sim::runtime::EndPoint>(epA).ok());
        sim::runtime::EndPoint epB{};
        epB.device_id = devBAdded.value->value;
        std::memcpy(epB.eid, eidB, sizeof(epB.eid));
        ASSERT_TRUE(sim::runtime::Db::Add<sim::runtime::EndPoint>(epB).ok());

        struct rdev rdevA {};
        rdevA.phyId = 0;
        rdevA.family = AF_INET6;
        std::memcpy(&rdevA.localIp, eidA, sizeof(eidA));
        ASSERT_EQ(0, RaSocketInit(0, rdevA, &sockA_));
        ASSERT_NE(nullptr, sockA_);
        struct rdev rdevB {};
        rdevB.phyId = 1;
        rdevB.family = AF_INET6;
        std::memcpy(&rdevB.localIp, eidB, sizeof(eidB));
        ASSERT_EQ(0, RaSocketInit(0, rdevB, &sockB_));
        ASSERT_NE(nullptr, sockB_);
    }

    void TearDown() override
    {
        if (sockA_ != nullptr) {
            RaSocketDeinit(sockA_);
        }
        if (sockB_ != nullptr) {
            RaSocketDeinit(sockB_);
        }
        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
        RemoveDatabaseFiles(kRunnerDb);
        RemoveDatabaseFiles(kOpDataDb);
    }

    char tag_[128]{};
    void* sockA_{nullptr};
    void* sockB_{nullptr};
};

// Add 失败：关闭进程会话 → 任一存储步骤失败 → 返回 -1（合同不变）。
// 注：会话关闭后 GetEndPointByEid 先于 Add 失败（同函数更早出口），Add 专属
// 字段（tagHash/slotIdx/code/diagnostic）由代码审查+首个后端错误日志通道保证；
// 本用例固化“失败即 -1 + 失败分支有日志”的合同。
TEST_F(HccpRaSocketDiagTest, BatchConnectAnyStorageFailureReturnsErrorWithLog)
{
    struct SocketConnectInfoT conn {};
    conn.socketHandle = sockA_;
    uint8_t eidB[16];
    MakeEidBytes(eidB, 0x02);
    std::memcpy(&conn.remoteIp, eidB, sizeof(eidB));
    conn.port = 4321;
    std::snprintf(conn.tag, sizeof(conn.tag), "%s", tag_);

    (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
    StdoutCapture capture;
    ASSERT_TRUE(capture.Start());
    const int rc = RaSocketBatchConnect(&conn, 1);
    const std::string log = capture.StopAndRead();
    EXPECT_EQ(rc, -1);
    // 失败分支必有带上下文的首错日志（本注入落在 eid 解析出口）
    EXPECT_NE(log.find("cannot find remote eid"), std::string::npos) << log;
}

// 末次 close 删除 1 行：DB 中 pair 行消失（成功 1 行路径，返回 0）
TEST_F(HccpRaSocketDiagTest, CloseLastDeletesExactlyOneRow)
{
    struct SocketConnectInfoT conn {};
    conn.socketHandle = sockA_;
    uint8_t eidB[16];
    MakeEidBytes(eidB, 0x02);
    std::memcpy(&conn.remoteIp, eidB, sizeof(eidB));
    conn.port = 4321;
    std::snprintf(conn.tag, sizeof(conn.tag), "%s", tag_);
    ASSERT_EQ(0, RaSocketBatchConnect(&conn, 1));
    const uint64_t tagHash = ComputeTagHashForTest(tag_);
    ASSERT_EQ(CountPairsByTagHash(tagHash), 1U);

    struct SocketInfoT cliConn {};
    cliConn.socketHandle = sockA_;
    std::snprintf(cliConn.tag, sizeof(cliConn.tag), "%s", tag_);
    unsigned int connected = 0;
    ASSERT_EQ(0, RaGetSockets(kRaRoleClient, &cliConn, 1, &connected));
    ASSERT_EQ(1U, connected);

    struct SocketCloseInfoT closeInfo {};
    closeInfo.fdHandle = cliConn.fdHandle;
    ASSERT_EQ(0, RaSocketBatchClose(&closeInfo, 1));
    EXPECT_EQ(CountPairsByTagHash(tagHash), 0U);
}

// 行已不在时末次 close：0 行删除形态可区分（返回 0、不崩溃、行数保持 0）
TEST_F(HccpRaSocketDiagTest, CloseLastWithRowAbsentReportsZeroRows)
{
    struct SocketConnectInfoT conn {};
    conn.socketHandle = sockA_;
    uint8_t eidB[16];
    MakeEidBytes(eidB, 0x02);
    std::memcpy(&conn.remoteIp, eidB, sizeof(eidB));
    conn.port = 4321;
    std::snprintf(conn.tag, sizeof(conn.tag), "%s", tag_);
    ASSERT_EQ(0, RaSocketBatchConnect(&conn, 1));
    const uint64_t tagHash = ComputeTagHashForTest(tag_);

    struct SocketInfoT cliConn {};
    cliConn.socketHandle = sockA_;
    std::snprintf(cliConn.tag, sizeof(cliConn.tag), "%s", tag_);
    unsigned int connected = 0;
    ASSERT_EQ(0, RaGetSockets(kRaRoleClient, &cliConn, 1, &connected));
    ASSERT_EQ(1U, connected);

    // 预删 pair 行（模拟早前重发/他人已删），随后 close 走 0 行删除路径
    auto rows = runnerdb_test::SelectRecords<sim::runtime::RaSocketPair>(
        HcclSim::Storage::Eq(&sim::runtime::RaSocketPair::tag_hash, tagHash));
    ASSERT_EQ(rows.size(), 1U);
    ASSERT_TRUE(runnerdb_test::DeleteRecord<sim::runtime::RaSocketPair>(rows[0].id));

    StdoutCapture capture;
    ASSERT_TRUE(capture.Start());
    struct SocketCloseInfoT closeInfo {};
    closeInfo.fdHandle = cliConn.fdHandle;
    const int rc = RaSocketBatchClose(&closeInfo, 1);
    const std::string log = capture.StopAndRead();
    EXPECT_EQ(rc, 0);
    EXPECT_EQ(CountPairsByTagHash(tagHash), 0U);
    EXPECT_NE(log.find("delete RaSocketPair OK/0rows"), std::string::npos) << log;
}

// 轮询：真 NOT_FOUND → 1.2s 后插入 → 成功（connectedNum=1，返回 0）
TEST_F(HccpRaSocketDiagTest, PollingNotFoundThenInsertSucceeds)
{
    const uint64_t tagHash = ComputeTagHashForTest(tag_);

    // 后台线程延迟插入 pair 行（模拟对端慢建链；slotIdx 用真实槽位——
    // RaGetSockets 命中后要 GetSlotHandle/AddRef，假槽位会失败）
    std::thread inserter([tagHash] {
        std::this_thread::sleep_for(std::chrono::milliseconds{1200});
        const uint32_t slotIdx = sim::CommunicationMemoryManager::GetInstance().AllocSlot();
        sim::runtime::RaSocketPair pair{};
        pair.client_id = 1;
        pair.server_id = 0;
        pair.tag_hash = tagHash;
        pair.slot_idx = slotIdx;
        (void)runnerdb_test::InsertRecord<sim::runtime::RaSocketPair>(pair);
    });

    struct SocketInfoT srvConn {};
    srvConn.socketHandle = sockB_;
    std::snprintf(srvConn.tag, sizeof(srvConn.tag), "%s", tag_);
    unsigned int connected = 0;
    const auto startedAt = std::chrono::steady_clock::now();
    const int rc = RaGetSockets(kRaRoleServer, &srvConn, 1, &connected);
    const auto elapsedMs
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startedAt).count();
    inserter.join();
    EXPECT_EQ(rc, 0);
    EXPECT_EQ(connected, 1U);
    EXPECT_GE(elapsedMs, 1000); // 确实经历了等待轮询
    EXPECT_LT(elapsedMs, 30000);
}

// 有界慢用例（唯一，约 93 秒）：后端失败→NOT_FOUND→180 次超时。
// 时序：0s 起轮询；~1.5s 关会话（INVALID_STATE 后端失败）；~3s 重建空会话
// （NOT_FOUND）；跑满 180×500ms=90s 超时。断言：返回 0、connectedNum=0、
// 立即打印首个后端失败、最终摘要保留首错（firstBackendCode=4 证明后端失败
// 计数≥1；lastCode=1 证明末次为 NOT_FOUND——两者并存即“末次 NOT_FOUND
// 不抹掉首错”）。
TEST_F(HccpRaSocketDiagTest, PollingBackendFailThenNotFoundTimeoutKeepsFirstError)
{
    std::thread churn([] {
        std::this_thread::sleep_for(std::chrono::milliseconds{1500});
        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
        std::this_thread::sleep_for(std::chrono::milliseconds{1500});
        HcclSim::Storage::StorageRuntimeConfig config;
        config.databases.push_back({std::string{"RUNNER_DB"}, "sqlite", kRunnerDb});
        config.databases.push_back({std::string{"OP_DATA_DB"}, "sqlite", kOpDataDb});
        (void)sim::EnsureSimulatorStorageSession(config);
    });

    StdoutCapture capture;
    ASSERT_TRUE(capture.Start());
    struct SocketInfoT srvConn {};
    srvConn.socketHandle = sockB_;
    std::snprintf(srvConn.tag, sizeof(srvConn.tag), "%s", tag_);
    unsigned int connected = 42;
    const int rc = RaGetSockets(kRaRoleServer, &srvConn, 1, &connected);
    const std::string log = capture.StopAndRead();
    churn.join();

    EXPECT_EQ(rc, 0);         // 合同：超时也返回 0
    EXPECT_EQ(connected, 0U); // 合同：connectedNum 不虚增
    EXPECT_NE(log.find("RaGetSockets first backend failure"), std::string::npos) << log;
    EXPECT_NE(log.find("RaGetSockets pair not found"), std::string::npos) << log;
    EXPECT_NE(log.find("firstBackendCode:4"), std::string::npos) << log; // INVALID_STATE=4（后端失败≥1）
    EXPECT_NE(log.find("lastCode:1"), std::string::npos) << log;         // NOT_FOUND=1（末次结果）
    EXPECT_NE(log.find("notFound:"), std::string::npos) << log;
    EXPECT_NE(log.find("backendFails:"), std::string::npos) << log;
}
