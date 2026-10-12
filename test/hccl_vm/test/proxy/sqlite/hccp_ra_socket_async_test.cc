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
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

// RaSocket 异步请求合同回归（512p 忙等风暴修复）：
//   1) 空队列 RecvAsync 后查询必须得到 reqResult==SOCK_EAGAIN（hcomm
//      HrtRaGetAsyncReqResult → Socket::CheckRecvRequestResult 据此走
//      HandleSocketEAgain 1ms 睡眠重投）；旧实现"恒返回 0 且不填写 reqResult +
//      receivedSize 归零"被 hcomm 解释为 completed+0 字节 → recvLeftSize 永不
//      减少的忙等风暴（08 节点 512p 55 分钟无进展、372MB recvSize[0] 日志）。
//   2) 写入数据后重新提交：查询成功（reqResult==0）且 receivedSize/内容正确；
//   3) 并发 request 句柄独立，固定假句柄 0x12345678(=305419896) 必须消失；
//   4) request 查询即释放：重复查询/非法句柄/空指针明确报错（无 UAF 可观测路径）；
//   5) SendAsync 满槽（kSlotBufSize=4096）记 EAGAIN 而非 completed+0，
//      BatchCloseAsync/ListenStopAsync 同样登记真实 request。
#include <arpa/inet.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "hccp_async.h"
#include "hccp_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/storage_runtime.h"
#include "storage_composition/storage_composition.h"
#include "store_sim_resource_root.h"

// hccp 异步族由 hccp_async.h 声明；同步族（Init/Deinit/BatchConnect/RaGetSockets）
// 声明在 hcomm 侧另一头文件、未随 hccp 头导出，被测实现编译在本目标内，此处按
// 实现签名直接声明（与 hccp_ra_socket_stub.cc 定义一致）。
extern "C" {
int RaSocketInit(int mode, struct rdev rdevInfo, void** socketHandle);
int RaSocketDeinit(void* socketHandle);
int RaSocketBatchConnect(struct SocketConnectInfoT conn[], unsigned int num);
int RaGetSockets(unsigned int role, struct SocketInfoT conn[], unsigned int num, unsigned int* connectedNum);
// QP/TP 族异步生产者（hccp_stub.cc 实现，同合同：句柄经公共注册表）
int RaGetTpAttrAsync(void* ctxHandle, uint64_t tpHandle, uint32_t* attrBitmap, struct TpAttr* attr, void** reqHandle);
int RaCtxQpUnimportAsync(void* remQpHandle, void** reqHandle);
// 链接占位：hccp_stub.cc 引用 hccp_ccu_stub.cc 的 SimRaCustomChannel（本用例
// 不触达该路径；生产中由 hccl_proxy_level2 一并链接 ccu_stub）。
int SimRaCustomChannel(void* tlvHandle, struct TlvMsg* sendMsg, struct TlvMsg* recvMsg);
}

// 占位实现：仅满足链接，不参与任何用例路径。
int SimRaCustomChannel(void* tlvHandle, struct TlvMsg* sendMsg, struct TlvMsg* recvMsg)
{
    (void)tlvHandle;
    (void)sendMsg;
    (void)recvMsg;
    return -1;
}

// GetDeviceByPhysicalId 按 (physical_id, server_id=g_cur_server_key) 查询
extern uint64_t g_cur_server_key;

namespace {
constexpr char kRunnerDb[] = "/tmp/hccp_ra_socket_async_runner.db";
constexpr char kOpDataDb[] = "/tmp/hccp_ra_socket_async_opdata.db";
constexpr uint64_t kFakeLegacyHandle = 0x12345678ULL; // 旧实现共用假句柄（十进制 305419896）
constexpr unsigned int kRaRoleServer = 0;
constexpr unsigned int kRaRoleClient = 1;
constexpr size_t kCtrlMsgSize = 336; // 512p 现场控制消息长度

void RemoveDatabaseFiles(const char* path)
{
    std::remove(path);
    std::remove((std::string(path) + "-wal").c_str());
    std::remove((std::string(path) + "-shm").c_str());
}

/// 16 字节 EID（末字节区分设备），写入 EndPoint.eid 与 rdev/conn 的 IP 字段
void MakeEidBytes(uint8_t eid[16], uint8_t tail)
{
    std::memset(eid, 0, 16);
    eid[15] = tail;
}
} // namespace

class HccpRaSocketAsyncTest : public testing::Test {
protected:
    void SetUp() override
    {
        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
        RemoveDatabaseFiles(kRunnerDb);
        RemoveDatabaseFiles(kOpDataDb);
        HcclSim::Storage::StorageRuntimeConfig config;
        config.databases.push_back({std::string{"RUNNER_DB"}, "sqlite", kRunnerDb});
        config.databases.push_back({std::string{"OP_DATA_DB"}, "sqlite", kOpDataDb});
        ASSERT_TRUE(sim::EnsureSimulatorStorageSession(config).ok());

        // GetDeviceByPhysicalId 按 server_id=g_cur_server_key 过滤；槽位池
        // （ra_sock_pool_*）依赖资源根目录，二者都需先就位
        g_cur_server_key = 1;
        ASSERT_TRUE(sim::SimResourceRoot::GetInstance().Init());

        // 每个用例独立 tag：g_tagCache 为进程级缓存，旧条目随旧库失效，不可复用
        static uint32_t tagSeq = 0;
        snprintf(tag_, sizeof(tag_), "ut_async_tag_%u", tagSeq++);

        SetUpSocketPair();
    }

    void TearDown() override
    {
        // 走异步关闭：同时覆盖 BatchCloseAsync 的真实 request 登记与查询
        if (serverFd_ != nullptr) {
            CloseFdViaAsync(serverFd_);
        }
        if (clientFd_ != nullptr) {
            CloseFdViaAsync(clientFd_);
        }
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

    /// 建立一对已连接 fd：A(physical 0) 连接 B(physical 1)，A=client、B=server
    void SetUpSocketPair()
    {
        uint8_t eidA[16];
        uint8_t eidB[16];
        MakeEidBytes(eidA, 0x01);
        MakeEidBytes(eidB, 0x02);

        sim::runtime::Device devA{};
        devA.server_id = 1;
        devA.logic_id = 0;
        devA.physical_id = 0;
        devA.super_device_id = 0;
        auto devAAdded = sim::runtime::Db::Add<sim::runtime::Device>(devA);
        ASSERT_TRUE(devAAdded.ok() && devAAdded.value.has_value());
        sim::runtime::Device devB{};
        devB.server_id = 1;
        devB.logic_id = 1;
        devB.physical_id = 1;
        devB.super_device_id = 0;
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

        struct SocketConnectInfoT conn {};
        conn.socketHandle = sockA_;
        std::memcpy(&conn.remoteIp, eidB, sizeof(eidB));
        conn.port = 4321;
        snprintf(conn.tag, sizeof(conn.tag), "%s", tag_);
        ASSERT_EQ(0, RaSocketBatchConnect(&conn, 1));

        struct SocketInfoT srvConn {};
        srvConn.socketHandle = sockB_;
        snprintf(srvConn.tag, sizeof(srvConn.tag), "%s", tag_);
        unsigned int connected = 0;
        ASSERT_EQ(0, RaGetSockets(kRaRoleServer, &srvConn, 1, &connected));
        ASSERT_EQ(1, connected);
        serverFd_ = srvConn.fdHandle;

        struct SocketInfoT cliConn {};
        cliConn.socketHandle = sockA_;
        snprintf(cliConn.tag, sizeof(cliConn.tag), "%s", tag_);
        connected = 0;
        ASSERT_EQ(0, RaGetSockets(kRaRoleClient, &cliConn, 1, &connected));
        ASSERT_EQ(1, connected);
        clientFd_ = cliConn.fdHandle;
    }

    void CloseFdViaAsync(void* fdHandle)
    {
        struct SocketCloseInfoT closeInfo {};
        closeInfo.fdHandle = fdHandle;
        void* handle = nullptr;
        ASSERT_EQ(0, RaSocketBatchCloseAsync(&closeInfo, 1, &handle));
        ASSERT_NE(nullptr, handle);
        struct AsyncReqResult result {
            - 1, static_cast<unsigned int>(-1)
        };
        ASSERT_EQ(0, RaGetAsyncReqResult(handle, &result));
        ASSERT_EQ(0, result.reqResult);
    }

    char tag_[64]{};
    void* sockA_ = nullptr;
    void* sockB_ = nullptr;
    void* serverFd_ = nullptr;
    void* clientFd_ = nullptr;
};

/// 空队列：RecvAsync 提交成功后查询必须得到 SOCK_EAGAIN；receivedSize 不被污染
/// 为"completed+0"，句柄唯一且非旧固定假句柄。
TEST_F(HccpRaSocketAsyncTest, EmptyQueueRecvPollsEagainNotCompletedZero)
{
    uint8_t buf[kCtrlMsgSize];
    unsigned long long recvSize = 0xABCD; // 非零哨兵：EAGAIN 路径不得写 0 伪装完成
    void* handle = nullptr;
    ASSERT_EQ(0, RaSocketRecvAsync(clientFd_, buf, sizeof(buf), &recvSize, &handle));
    ASSERT_NE(nullptr, handle);
    EXPECT_NE(reinterpret_cast<void*>(kFakeLegacyHandle), handle);
    // 无数据时 receivedSize 保持调用方初值（提交侧不伪造 0 字节完成）
    EXPECT_EQ(0xABCDULL, recvSize);

    struct AsyncReqResult result {
        - 1, static_cast<unsigned int>(-1)
    };
    ASSERT_EQ(0, RaGetAsyncReqResult(handle, &result));
    EXPECT_EQ(SOCK_EAGAIN, result.reqResult); // 绝不能是 0（=completed）
    EXPECT_EQ(0xABCDULL, recvSize);           // 查询侧同样不得伪装
}

/// 服务端写入 336B 后：客户端重新提交收请求，查询成功且字节/内容真实。
TEST_F(HccpRaSocketAsyncTest, SendThenRecvDeliversRealPayload)
{
    uint8_t payload[kCtrlMsgSize];
    for (size_t i = 0; i < sizeof(payload); i++) {
        payload[i] = static_cast<uint8_t>(i * 7 + 3);
    }
    unsigned long long sentSize = 0;
    void* sendHandle = nullptr;
    ASSERT_EQ(0, RaSocketSendAsync(serverFd_, payload, sizeof(payload), &sentSize, &sendHandle));
    ASSERT_EQ(sizeof(payload), sentSize);
    struct AsyncReqResult sendResult {
        - 1, static_cast<unsigned int>(-1)
    };
    ASSERT_EQ(0, RaGetAsyncReqResult(sendHandle, &sendResult));
    EXPECT_EQ(0, sendResult.reqResult);

    uint8_t got[kCtrlMsgSize]{};
    unsigned long long recvSize = 0;
    void* recvHandle = nullptr;
    ASSERT_EQ(0, RaSocketRecvAsync(clientFd_, got, sizeof(got), &recvSize, &recvHandle));
    ASSERT_EQ(sizeof(got), recvSize);
    struct AsyncReqResult recvResult {
        - 1, static_cast<unsigned int>(-1)
    };
    ASSERT_EQ(0, RaGetAsyncReqResult(recvHandle, &recvResult));
    EXPECT_EQ(0, recvResult.reqResult);
    EXPECT_EQ(0, std::memcmp(got, payload, sizeof(payload)));
}

/// 满槽（kSlotBufSize=4096）后的 SendAsync：查询得到 SOCK_EAGAIN（睡眠重投），
/// 而不是 completed+0 字节造成的发送侧忙等。
TEST_F(HccpRaSocketAsyncTest, FullSlotSendPollsEagainNotCompletedZero)
{
    uint8_t full[4096]{};
    unsigned long long sentSize = 0;
    void* handle = nullptr;
    // 一次灌满槽位
    ASSERT_EQ(0, RaSocketSendAsync(serverFd_, full, sizeof(full), &sentSize, &handle));
    ASSERT_EQ(sizeof(full), sentSize);
    struct AsyncReqResult result {
        - 1, static_cast<unsigned int>(-1)
    };
    ASSERT_EQ(0, RaGetAsyncReqResult(handle, &result));
    EXPECT_EQ(0, result.reqResult);

    // 槽位已满：1 字节也发不出。同步层如实写 sentSize=0（真实实发字节），
    // 异步层查询必须给 SOCK_EAGAIN（睡眠重投），而不是 completed+0 的忙等。
    uint8_t one = 0x5A;
    sentSize = 0xFFFF;
    handle = nullptr;
    ASSERT_EQ(0, RaSocketSendAsync(serverFd_, &one, 1, &sentSize, &handle));
    EXPECT_EQ(0ULL, sentSize); // 同步层如实回写实发字节数
    result = {-1, static_cast<unsigned int>(-1)};
    ASSERT_EQ(0, RaGetAsyncReqResult(handle, &result));
    EXPECT_EQ(SOCK_EAGAIN, result.reqResult);
}

/// 并发 request：两个在途句柄互异、可乱序查询，各自一次性释放。
TEST_F(HccpRaSocketAsyncTest, ConcurrentRequestsHaveIndependentHandles)
{
    uint8_t buf[16];
    unsigned long long recvSize1 = 0;
    unsigned long long recvSize2 = 0;
    void* handle1 = nullptr;
    void* handle2 = nullptr;
    ASSERT_EQ(0, RaSocketRecvAsync(clientFd_, buf, sizeof(buf), &recvSize1, &handle1));
    ASSERT_EQ(0, RaSocketRecvAsync(clientFd_, buf, sizeof(buf), &recvSize2, &handle2));
    ASSERT_NE(handle1, handle2);
    EXPECT_NE(reinterpret_cast<void*>(kFakeLegacyHandle), handle1);
    EXPECT_NE(reinterpret_cast<void*>(kFakeLegacyHandle), handle2);

    // 乱序查询：句柄互不覆盖
    struct AsyncReqResult result2 {
        - 1, static_cast<unsigned int>(-1)
    };
    ASSERT_EQ(0, RaGetAsyncReqResult(handle2, &result2));
    EXPECT_EQ(SOCK_EAGAIN, result2.reqResult);
    struct AsyncReqResult result1 {
        - 1, static_cast<unsigned int>(-1)
    };
    ASSERT_EQ(0, RaGetAsyncReqResult(handle1, &result1));
    EXPECT_EQ(SOCK_EAGAIN, result1.reqResult);
}

/// request 查询即释放：重复查询、未登记句柄、空指针均明确报错。
TEST_F(HccpRaSocketAsyncTest, DoublePollAndInvalidParamsRejected)
{
    uint8_t buf[8];
    unsigned long long recvSize = 0;
    void* handle = nullptr;
    ASSERT_EQ(0, RaSocketRecvAsync(clientFd_, buf, sizeof(buf), &recvSize, &handle));
    struct AsyncReqResult result {
        - 1, static_cast<unsigned int>(-1)
    };
    ASSERT_EQ(0, RaGetAsyncReqResult(handle, &result));
    // 同一句柄二次查询：合同规定完成即释放，必须报错而非返回旧状态
    EXPECT_NE(0, RaGetAsyncReqResult(handle, &result));
    // 从未登记过的句柄
    uint64_t bogus = 0xDEADBEEF;
    EXPECT_NE(0, RaGetAsyncReqResult(reinterpret_cast<void*>(bogus), &result));
    // 空指针入参
    EXPECT_NE(0, RaGetAsyncReqResult(nullptr, &result));
    EXPECT_NE(0, RaGetAsyncReqResult(handle, nullptr));
    // 提交侧空出参
    EXPECT_NE(0, RaSocketRecvAsync(clientFd_, buf, sizeof(buf), &recvSize, nullptr));
    EXPECT_NE(0, RaSocketSendAsync(serverFd_, buf, 1, &recvSize, nullptr));
}

/// ListenStopAsync 同合同：真实句柄 + 查询成功（teardown 已覆盖 BatchCloseAsync）。
TEST_F(HccpRaSocketAsyncTest, ListenStopAsyncUsesRealRequest)
{
    struct SocketListenInfoT listenInfo {};
    listenInfo.socketHandle = sockB_;
    listenInfo.port = 4321;
    void* handle = nullptr;
    ASSERT_EQ(0, RaSocketListenStopAsync(&listenInfo, 1, &handle));
    ASSERT_NE(nullptr, handle);
    EXPECT_NE(reinterpret_cast<void*>(kFakeLegacyHandle), handle);
    struct AsyncReqResult result {
        - 1, static_cast<unsigned int>(-1)
    };
    ASSERT_EQ(0, RaGetAsyncReqResult(handle, &result));
    EXPECT_EQ(0, result.reqResult);
}

/// 同合同 QP/TP 族（hccp_stub.cc 生产者，含 CCU 建链路径的 jetty/QP 异步族）：
/// 必须经公共注册表返回真实唯一句柄，查询得到 COMPLETED——旧假句柄被
/// RaGetAsyncReqResult 识别为未知句柄报错，会中断 hcomm 通信域初始化
/// （256p 现场：CcuInitFeature 抛 NetworkApiException → ret 11）。
TEST_F(HccpRaSocketAsyncTest, TpAndQpFamilyAsyncUseRealRequests)
{
    struct TpAttr attr {};
    uint32_t attrBitmap = 0;
    void* handle = nullptr;
    ASSERT_EQ(0, RaGetTpAttrAsync(nullptr, 0x12345678, &attrBitmap, &attr, &handle));
    ASSERT_NE(nullptr, handle);
    EXPECT_NE(reinterpret_cast<void*>(kFakeLegacyHandle), handle);
    struct AsyncReqResult result {
        - 1, static_cast<unsigned int>(-1)
    };
    ASSERT_EQ(0, RaGetAsyncReqResult(handle, &result));
    EXPECT_EQ(0, result.reqResult);

    handle = nullptr;
    ASSERT_EQ(0, RaCtxQpUnimportAsync(nullptr, &handle));
    ASSERT_NE(nullptr, handle);
    EXPECT_NE(reinterpret_cast<void*>(kFakeLegacyHandle), handle);
    result = {-1, static_cast<unsigned int>(-1)};
    ASSERT_EQ(0, RaGetAsyncReqResult(handle, &result));
    EXPECT_EQ(0, result.reqResult);
}
