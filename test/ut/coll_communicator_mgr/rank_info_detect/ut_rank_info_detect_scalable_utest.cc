/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>
#include <thread>

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include "binary_stream.h"
#include "securec.h"
#include "types.h"

#define private public
#define protected public
#include "rank_info_detect_scalable.h"
#undef protected
#undef private

#include "rank_table_info.h"
#include "env_config/env_config_v2.h"
#include "root_group_strategy.h"
#include "root_handle_v2.h"

using namespace Hccl;

namespace {
// 构造最小 ScalableServerInfo（nRoot/rootIndex/groupSize 供调度与完成判定使用）。
ScalableServerInfo MakeServerInfo(u32 nRoot, u32 rootIndex, u32 groupSize)
{
    ScalableServerInfo info{};
    info.nRoot = nRoot;
    info.rootIndex = rootIndex;
    info.groupSize = groupSize;
    info.meshInfos.resize(nRoot);
    return info;
}
} // namespace

// Ut_SetScalableServerInfo_When_InjectedBeforeServerRun_Return_ThreadWoken：
// 注入同步：server 线程阻塞等待，注入后唤醒并读到完整信息
TEST(RankInfoDetectScalableSyncTest, Ut_SetScalableServerInfo_When_InjectedBeforeServerRun_Return_ThreadWoken)
{
    auto detect = std::make_shared<RankInfoDetectScalable>(0, 0, 0);
    ASSERT_NE(detect, nullptr);

    std::atomic<bool> woken{false};
    std::atomic<HcclResult> waitRet{HCCL_E_INTERNAL};
    std::thread waiter([&]() {
        waitRet.store(detect->WaitScalableInfoReady());
        woken.store(true);
    });

    // 确认等待方已阻塞（短睡眠让线程先进入等待）
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(woken.load());

    const ScalableServerInfo info = MakeServerInfo(4, 1, 5);
    EXPECT_EQ(detect->SetScalableServerInfo(info), HCCL_SUCCESS);

    waiter.join();
    EXPECT_TRUE(woken.load());
    EXPECT_EQ(waitRet.load(), HCCL_SUCCESS);
    // 注入信息可完整读回
    EXPECT_EQ(detect->scalableInfo_.nRoot, 4U);
    EXPECT_EQ(detect->scalableInfo_.rootIndex, 1U);
    EXPECT_EQ(detect->scalableInfo_.groupSize, 5U);
    EXPECT_EQ(detect->scalableInfo_.meshInfos.size(), 4U);
}

// Ut_SetScalableServerInfo_When_SecondWaitAfterInject_Return_Immediately：
// 注入后二次等待立即成功（谓词短路，不触发超时分支）。
// 说明：WaitScalableInfoReady 的 HCCL_E_TIMEOUT 分支暂无本目录 UT 覆盖，
//       由 HLT rootinfo_scalable_a5_003（root 缺席场景）承载数值验证。
TEST(RankInfoDetectScalableSyncTest, Ut_SetScalableServerInfo_When_SecondWaitAfterInject_Return_Immediately)
{
    auto detect = std::make_shared<RankInfoDetectScalable>(0, 0, 0);
    const ScalableServerInfo info = MakeServerInfo(2, 0, 2);
    EXPECT_EQ(detect->SetScalableServerInfo(info), HCCL_SUCCESS);
    // 已注入状态下的二次等待立即成功（谓词短路，不依赖超时）
    auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(detect->WaitScalableInfoReady(), HCCL_SUCCESS);
    auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 2000);
}

namespace {
std::vector<char> incoming;
std::vector<char> outgoing;
std::string ioOrder;

template <typename T>
void AppendValue(std::vector<char>& bytes, const T& value)
{
    const auto* begin = reinterpret_cast<const char*>(&value);
    bytes.insert(bytes.end(), begin, begin + sizeof(value));
}

std::vector<char> Frame(u32 source, const std::vector<char>& payload, const std::string& failedList = "")
{
    std::vector<char> bytes;
    AppendValue(bytes, source);
    AppendValue(bytes, static_cast<u64>(payload.size()));
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    AppendValue(bytes, static_cast<u64>(failedList.size()));
    bytes.insert(bytes.end(), failedList.begin(), failedList.end());
    return bytes;
}

// mockcpp 2.7-h6 passes a non-const C* as the first invoke argument,
// including for const member functions (ApiHookFunctor / getParameterWithThis).
bool CaptureSend(Socket*, const void* data, u32 size)
{
    ioOrder += 'S';
    const auto* bytes = static_cast<const char*>(data);
    outgoing.insert(outgoing.end(), bytes, bytes + size);
    return true;
}

bool SupplyRecv(Socket*, void* data, u32 size)
{
    ioOrder += 'R';
    if (incoming.size() < size) {
        return false;
    }
    if (memcpy_s(data, size, incoming.data(), size) != EOK) {
        return false;
    }
    incoming.erase(incoming.begin(), incoming.begin() + size);
    return true;
}

RankTableInfo MakePartial(u32 firstRank, u32 count)
{
    RankTableInfo table;
    table.version = "1.0";
    table.rankCount = count;
    for (u32 i = 0; i < count; ++i) {
        hccl::NewRankInfo rank;
        rank.rankId = firstRank + i;
        rank.rankLevelInfos.emplace_back();
        rank.controlPlane.addrType = AddrType::IPV4;
        rank.controlPlane.addr = IpAddress("127.0.0.1");
        rank.controlPlane.listenPort = 60000;
        table.ranks.push_back(rank);
    }
    return table;
}

std::atomic<bool> acceptExited{false};
HcclResult FinishAcceptAfterConnectFailure([[maybe_unused]] RankInfoDetectScalable*, [[maybe_unused]] u32 timeoutSec)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    acceptExited.store(true);
    return HCCL_SUCCESS;
}

// Accept 打桩序号：第 2 次（peer 1）返回失败模拟缺席，其余成功；记录每次收到的预算（验证按剩余对端数切片）
std::atomic<u32> acceptSeq{0};
std::vector<u32> acceptTimeouts;
HcclResult StubAcceptSeq(
    [[maybe_unused]] Socket* self, [[maybe_unused]] const std::string& tag, std::shared_ptr<Socket>& acceptedSocket,
    u32 timeout)
{
    acceptTimeouts.push_back(timeout);
    if (acceptSeq.fetch_add(1) == 1) {
        return HCCL_E_TIMEOUT;
    }
    acceptedSocket = std::make_shared<Socket>(
        nullptr, IpAddress(), 0, IpAddress(), "mesh-ut", SocketRole::SERVER, NicType::HOST_NIC_TYPE);
    return HCCL_SUCCESS;
}

// Accept 全部成功打桩：配合抛异常的 Recv 验证「已建链但交换失败」路径
std::atomic<u32> acceptAlwaysCount{0};
HcclResult StubAcceptAlwaysOk(
    [[maybe_unused]] Socket* self, const std::string& tag, std::shared_ptr<Socket>& acceptedSocket,
    [[maybe_unused]] u32 timeout)
{
    acceptAlwaysCount.fetch_add(1);
    acceptedSocket = std::make_shared<Socket>(
        nullptr, IpAddress(), 0, IpAddress(), tag, SocketRole::SERVER, NicType::HOST_NIC_TYPE);
    return HCCL_SUCCESS;
}

std::atomic<u32> closeCount{0};
void CountClose([[maybe_unused]] Socket* self) { closeCount.fetch_add(1); }

// GetStatus 打桩：第 1 次返回 TIMEOUT 模拟对端慢启动（组内缺员耗满超时、mesh 晚启动），其余建链成功；记录每次等待预算
std::atomic<u32> getStatusSeq{0};
std::vector<u32> getStatusTimeouts;
SocketStatus StubGetStatusTimeoutFirst([[maybe_unused]] Socket* self, u32 timeout)
{
    getStatusTimeouts.push_back(timeout);
    return getStatusSeq.fetch_add(1) == 0 ? SocketStatus::TIMEOUT : SocketStatus::OK;
}

// GetStatus 打桩：全部返回 TIMEOUT（不完成建链，预算断言不受交换阶段干扰）；记录每次等待预算
SocketStatus StubGetStatusAlwaysTimeout([[maybe_unused]] Socket* self, u32 timeout)
{
    getStatusTimeouts.push_back(timeout);
    return SocketStatus::TIMEOUT;
}

// GetStatus 打桩：第 1 次返回 INIT 模拟连接被明确拒绝（对端 root 进程不在、监听已关），其余建链成功
std::atomic<u32> getStatusRefusedSeq{0};
SocketStatus StubGetStatusRefusedFirst([[maybe_unused]] Socket* self, [[maybe_unused]] u32 timeout)
{
    return getStatusRefusedSeq.fetch_add(1) == 0 ? SocketStatus::INIT : SocketStatus::OK;
}

// GetStatus 打桩：第 1 次返回 CONNECTING 模拟对端慢启动（mesh 晚于其组内收集启动、建链未完成），其余建链成功
std::atomic<u32> getStatusConnectingSeq{0};
SocketStatus StubGetStatusConnectingFirst([[maybe_unused]] Socket* self, [[maybe_unused]] u32 timeout)
{
    return getStatusConnectingSeq.fetch_add(1) == 0 ? SocketStatus::CONNECTING : SocketStatus::OK;
}

// GetStatus 打桩：全部建链成功
SocketStatus StubGetStatusAlwaysOk([[maybe_unused]] Socket* self, [[maybe_unused]] u32 timeout)
{
    return SocketStatus::OK;
}
} // namespace

class RankInfoDetectMeshTest : public testing::Test {
protected:
    void SetUp() override
    {
        incoming.clear();
        outgoing.clear();
        ioOrder.clear();
        MOCKER_CPP(&Socket::Send).stubs().will(invoke(CaptureSend));
        MOCKER_CPP(&Socket::Recv).stubs().will(invoke(SupplyRecv));
    }
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(RankInfoDetectMeshTest, SingleRootKeepsLocalTableWithoutMesh)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(1, 0, 8)), HCCL_SUCCESS);
    RankTableInfo table;
    table.rankCount = 8;
    std::string failedList = "0-7";
    EXPECT_EQ(detect.RootMeshAllGather(table, failedList, 8, std::chrono::steady_clock::now()), HCCL_SUCCESS);
    EXPECT_EQ(table.rankCount, 8U);
    EXPECT_EQ(failedList, "0-7"); // 单 root 不改失败列表
    EXPECT_TRUE(detect.partials_.empty());
    EXPECT_TRUE(ioOrder.empty());
}

TEST_F(RankInfoDetectMeshTest, FullMeshExchangesOwnPartialWithEveryPeer)
{
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "mesh-ut", SocketRole::CLIENT, NicType::HOST_NIC_TYPE);
    for (u32 count : {2U, 4U, 8U}) {
        for (u32 self = 0; self < count; ++self) {
            RankInfoDetectScalable detect(0, 0, 0);
            ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(count, self, 1)), HCCL_SUCCESS);
            detect.partials_.resize(count);
            detect.peerFailedLists_.resize(count);
            const std::vector<char> local{static_cast<char>(self), 'L'};
            detect.partials_[self] = local;
            for (u32 peer = 0; peer < count; ++peer) {
                if (peer == self) {
                    continue;
                }
                const std::vector<char> remote{static_cast<char>(peer), 'P'};
                incoming = Frame(peer, remote);
                outgoing.clear();
                ioOrder.clear();
                ASSERT_EQ(detect.ExchangeMeshPartial(socket, peer, self < peer), HCCL_SUCCESS);
                EXPECT_EQ(outgoing, Frame(self, local));
                EXPECT_EQ(detect.partials_[peer], remote);
                EXPECT_TRUE(detect.peerFailedLists_[peer].empty()); // 空失败列表合法落槽
                EXPECT_TRUE(incoming.empty());
                // 收发各 4 段：rootIndex/长度/子表/失败列表长度（空列表无 payload）
                EXPECT_EQ(ioOrder, self < peer ? "SSSSRRRR" : "RRRRSSSS");
            }
            EXPECT_EQ(detect.meshRecvCount_.load(), count - 1);
        }
    }
}

// mesh 消息携带本组区间压缩失败列表（M2）：发送侧附加 [长度 u64][列表]，接收侧落槽 peerFailedLists_
TEST_F(RankInfoDetectMeshTest, MeshFrameCarriesCompressedFailedList)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 0, 1)), HCCL_SUCCESS);
    detect.partials_.resize(2);
    detect.peerFailedLists_.resize(2);
    detect.partials_[0] = {'L'};
    detect.localFailedList_ = "4-5,7";
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "mesh-ut", SocketRole::CLIENT, NicType::HOST_NIC_TYPE);
    incoming = Frame(1, {'P'}, "0-3");
    outgoing.clear();
    ioOrder.clear();
    ASSERT_EQ(detect.ExchangeMeshPartial(socket, 1, true), HCCL_SUCCESS);
    EXPECT_EQ(outgoing, Frame(0, {'L'}, "4-5,7"));
    EXPECT_EQ(detect.peerFailedLists_[1], "0-3");
    EXPECT_EQ(detect.partials_[1], std::vector<char>({'P'}));
    EXPECT_EQ(detect.meshRecvCount_.load(), 1U);
    // 收发各 5 段：rootIndex/长度/子表/失败列表长度/失败列表
    EXPECT_EQ(ioOrder, "SSSSSRRRRR");
}

TEST_F(RankInfoDetectMeshTest, RejectsWrongSourceAndInvalidLengths)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 0, 1)), HCCL_SUCCESS);
    detect.partials_.resize(2);
    detect.peerFailedLists_.resize(2);
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "mesh-ut", SocketRole::CLIENT, NicType::HOST_NIC_TYPE);
    incoming = Frame(0, {'x'});
    EXPECT_EQ(detect.RecvMeshPartial(socket, 1), HCCL_E_PARA);
    for (u64 size : {u64{0}, static_cast<u64>(MAX_BUFFER_LEN) + 1}) {
        incoming.clear();
        AppendValue(incoming, u32{1});
        AppendValue(incoming, size);
        EXPECT_EQ(detect.RecvMeshPartial(socket, 1), HCCL_E_PARA);
        EXPECT_TRUE(detect.partials_[1].empty());
        EXPECT_TRUE(detect.peerFailedLists_[1].empty());
    }
    // 失败列表长度非法（0 合法、超收紧后的独立上界非法）：PARA 且不落槽
    for (u64 failedLen : {static_cast<u64>(MAX_MESH_FAILED_LIST_LEN) + 1, static_cast<u64>(MAX_BUFFER_LEN) + 1}) {
        incoming.clear();
        AppendValue(incoming, u32{1});
        AppendValue(incoming, u64{1});
        incoming.push_back('x');
        AppendValue(incoming, failedLen);
        EXPECT_EQ(detect.RecvMeshPartial(socket, 1), HCCL_E_PARA);
        EXPECT_TRUE(detect.partials_[1].empty());
        EXPECT_TRUE(detect.peerFailedLists_[1].empty());
    }
    // 失败列表长度为 0：合法（该组无失败），子表与空列表一并原子落槽
    incoming = Frame(1, {'x'});
    EXPECT_EQ(detect.RecvMeshPartial(socket, 1), HCCL_SUCCESS);
    EXPECT_EQ(detect.partials_[1], std::vector<char>({'x'}));
    EXPECT_TRUE(detect.peerFailedLists_[1].empty());
}

TEST_F(RankInfoDetectMeshTest, RejectsDuplicatePartialAndTruncatedBody)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 0, 1)), HCCL_SUCCESS);
    detect.partials_.resize(2);
    detect.peerFailedLists_.resize(2);
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "mesh-ut", SocketRole::CLIENT, NicType::HOST_NIC_TYPE);
    // 子表体截断（长度声明 2 仅到 1 字节，且无失败列表段）：子表接收失败，NETWORK 且不落槽
    // （失败列表段的缺帧/截断走降级路径，由 DegradesToEmptyFailedListOnOldFrameOrTruncation 覆盖）
    incoming.clear();
    AppendValue(incoming, u32{1});
    AppendValue(incoming, u64{2});
    incoming.push_back('x');
    EXPECT_EQ(detect.RecvMeshPartial(socket, 1), HCCL_E_NETWORK);
    EXPECT_TRUE(detect.partials_[1].empty());
    EXPECT_TRUE(detect.peerFailedLists_[1].empty());
    incoming = Frame(1, {'x'}, "0-3");
    ASSERT_EQ(detect.RecvMeshPartial(socket, 1), HCCL_SUCCESS);
    EXPECT_EQ(detect.peerFailedLists_[1], "0-3");
    incoming = Frame(1, {'y'});
    EXPECT_EQ(detect.RecvMeshPartial(socket, 1), HCCL_E_PARA);
    EXPECT_EQ(detect.partials_[1], std::vector<char>({'x'}));
    EXPECT_EQ(detect.peerFailedLists_[1], "0-3");
}

// 失败列表段降级（混版/断连）：旧版本帧缺失败列表段、或列表体截断时，
// 子表已完整接收则保留落槽、失败列表降级为空（agent 侧 rankCount 校验兜底），不整体丢弃子表
TEST_F(RankInfoDetectMeshTest, DegradesToEmptyFailedListOnOldFrameOrTruncation)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 0, 1)), HCCL_SUCCESS);
    detect.partials_.resize(2);
    detect.peerFailedLists_.resize(2);
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "mesh-ut", SocketRole::CLIENT, NicType::HOST_NIC_TYPE);

    // 旧版本帧（仅 [rootIndex][子表长度][子表]，无失败列表段）：降级为空列表，子表保留
    incoming.clear();
    AppendValue(incoming, u32{1});
    AppendValue(incoming, u64{1});
    incoming.push_back('x');
    EXPECT_EQ(detect.RecvMeshPartial(socket, 1), HCCL_SUCCESS);
    EXPECT_EQ(detect.partials_[1], std::vector<char>({'x'}));
    EXPECT_TRUE(detect.peerFailedLists_[1].empty());

    // 失败列表体截断（长度声明 4 字节仅到 2 字节）：同降级为空列表，子表保留
    RankInfoDetectScalable detectTrunc(0, 0, 0);
    ASSERT_EQ(detectTrunc.SetScalableServerInfo(MakeServerInfo(2, 0, 1)), HCCL_SUCCESS);
    detectTrunc.partials_.resize(2);
    detectTrunc.peerFailedLists_.resize(2);
    incoming.clear();
    AppendValue(incoming, u32{1});
    AppendValue(incoming, u64{1});
    incoming.push_back('x');
    AppendValue(incoming, u64{4});
    incoming.insert(incoming.end(), {'a', 'b'}); // 仅 2/4 字节
    EXPECT_EQ(detectTrunc.RecvMeshPartial(socket, 1), HCCL_SUCCESS);
    EXPECT_EQ(detectTrunc.partials_[1], std::vector<char>({'x'}));
    EXPECT_TRUE(detectTrunc.peerFailedLists_[1].empty());

    // 失败列表长度字段部分截断（仅 7/8 字节，长度段接收失败）：同降级为空列表，子表保留
    RankInfoDetectScalable detectLenTrunc(0, 0, 0);
    ASSERT_EQ(detectLenTrunc.SetScalableServerInfo(MakeServerInfo(2, 0, 1)), HCCL_SUCCESS);
    detectLenTrunc.partials_.resize(2);
    detectLenTrunc.peerFailedLists_.resize(2);
    incoming = Frame(1, {'x', 'y'});
    incoming.pop_back(); // 掉 failedLen(u64) 的最高字节，子表完整可收
    EXPECT_EQ(detectLenTrunc.RecvMeshPartial(socket, 1), HCCL_SUCCESS);
    EXPECT_EQ(detectLenTrunc.partials_[1], std::vector<char>({'x', 'y'}));
    EXPECT_TRUE(detectLenTrunc.peerFailedLists_[1].empty());
}

TEST_F(RankInfoDetectMeshTest, ReceiveFailureStopsExchangeBeforeReply)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 1, 1)), HCCL_SUCCESS);
    detect.partials_.resize(2);
    detect.peerFailedLists_.resize(2);
    detect.partials_[1] = {'L'};
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "mesh-ut", SocketRole::SERVER, NicType::HOST_NIC_TYPE);
    // 无供应（对端不发数据）：阻塞收包失败，先收后发路径收包失败即止，不回发
    EXPECT_EQ(detect.ExchangeMeshPartial(socket, 0, false), HCCL_E_NETWORK);
    EXPECT_TRUE(outgoing.empty());
    EXPECT_EQ(detect.meshRecvCount_.load(), 0U);
}

// 空槽不再报错（M3）：合并仅含在场子表，缺席组区间按分组策略反推并入失败列表
TEST_F(RankInfoDetectMeshTest, MergeSkipsAbsentPartialAndInfersFailedRange)
{
    RankInfoDetectScalable detect(0, 0, 0);
    // groupSize=4 与 totalRankNum=8/nRoot=2 经默认策略推导的本组区间 [0,4) 自洽（反推一致性校验前提）
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 0, 4)), HCCL_SUCCESS);
    detect.totalRankNum_ = 8; // nRanks=8, nRoot=2：root1 组区间 [4,7]
    detect.partials_.resize(2);
    detect.peerFailedLists_.resize(2);
    detect.peerMeshStates_.assign(2, MeshPeerState::PENDING);
    detect.peerMeshStates_[1] = MeshPeerState::UNREACHABLE; // 建链未建立 → 可整组反推
    auto selfPartial = MakePartial(0, 1);
    BinaryStream stream;
    ASSERT_EQ(selfPartial.GetBinStream(true, stream), HCCL_SUCCESS);
    stream.Dump(detect.partials_[0]);
    detect.peerFailedLists_[0] = "0";
    RankTableInfo table;
    std::string failedList;
    ASSERT_EQ(detect.MergeMeshPartials(2, table, failedList), HCCL_SUCCESS);
    EXPECT_EQ(table.rankCount, 1U); // 缺员表：仅在场子表，agent 侧 rankCount 校验兜底
    EXPECT_EQ(table.ranks.size(), 1U);
    EXPECT_EQ(failedList, "0,4-7"); // 本组失败 + 缺席组反推区间
}

// 余数摊前分组（rmr≠0）的缺席反推边界：nRanks=10, nRoot=3 → root2 组区间 [7,9]
TEST_F(RankInfoDetectMeshTest, MergeInfersAbsentRangeWithRemainderGroups)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(3, 0, 4)), HCCL_SUCCESS);
    detect.totalRankNum_ = 10;
    detect.partials_.resize(3);
    detect.peerFailedLists_.resize(3);
    detect.peerMeshStates_.assign(3, MeshPeerState::PENDING);
    detect.peerMeshStates_[2] = MeshPeerState::UNREACHABLE; // root2 建链未建立 → 可反推
    for (u32 root : {0U, 1U}) {
        auto partial = MakePartial(root == 0 ? 0 : 4, root == 0 ? 4 : 3);
        BinaryStream stream;
        ASSERT_EQ(partial.GetBinStream(true, stream), HCCL_SUCCESS);
        stream.Dump(detect.partials_[root]);
        detect.peerFailedLists_[root] = root == 0 ? "2-3" : "";
    }
    RankTableInfo table;
    std::string failedList;
    ASSERT_EQ(detect.MergeMeshPartials(3, table, failedList), HCCL_SUCCESS);
    EXPECT_EQ(table.rankCount, 7U);
    EXPECT_EQ(failedList, "2-3,7-9"); // 在场列表按 rootIndex 顺序拼接 + 缺席反推
}

// 反推依据缺失（全局 nRanks 未上报）时不阻断：仅聚合在场失败列表
TEST_F(RankInfoDetectMeshTest, MergeSkipsInferenceWhenTotalRankNumMissing)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 0, 1)), HCCL_SUCCESS);
    detect.totalRankNum_ = 0;
    detect.partials_.resize(2);
    detect.peerFailedLists_.resize(2);
    detect.peerMeshStates_.assign(2, MeshPeerState::PENDING);
    detect.peerMeshStates_[1] = MeshPeerState::UNREACHABLE;
    auto selfPartial = MakePartial(0, 1);
    BinaryStream stream;
    ASSERT_EQ(selfPartial.GetBinStream(true, stream), HCCL_SUCCESS);
    stream.Dump(detect.partials_[0]);
    detect.peerFailedLists_[0] = "0";
    RankTableInfo table;
    std::string failedList;
    ASSERT_EQ(detect.MergeMeshPartials(2, table, failedList), HCCL_SUCCESS);
    EXPECT_EQ(failedList, "0");
}

// 反推一致性校验：agent 上报的 totalRankNum 与 Get 阶段本地推导的本组区间不一致（应用给各 rank
// 传了不同 nRanks/配置错位）时，按 totalRankNum 反推的组区间会与本组真实区间错位 → 放弃反推，
// 仅聚合在场失败列表（错误遗言比无遗言更危险）
TEST_F(RankInfoDetectMeshTest, MergeSkipsInferenceWhenTotalRankNumMismatchedWithLocalGroup)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 0, 4)), HCCL_SUCCESS);
    // 本地推导 nRanks=8（本组 [0,4)），agent 上报 12：nRankFromRoot(12,2,0)=6 ≠ groupSize=4，校验不过
    detect.totalRankNum_ = 12;
    detect.partials_.resize(2);
    detect.peerFailedLists_.resize(2);
    detect.peerMeshStates_.assign(2, MeshPeerState::PENDING);
    detect.peerMeshStates_[1] = MeshPeerState::UNREACHABLE;
    auto selfPartial = MakePartial(0, 4);
    BinaryStream stream;
    ASSERT_EQ(selfPartial.GetBinStream(true, stream), HCCL_SUCCESS);
    stream.Dump(detect.partials_[0]);
    detect.peerFailedLists_[0] = "1-2";
    RankTableInfo table;
    std::string failedList;
    ASSERT_EQ(detect.MergeMeshPartials(2, table, failedList), HCCL_SUCCESS);
    EXPECT_EQ(failedList, "1-2"); // 不反推缺席组区间（按 12 反推的 [6,11] 与本组真实区间错位）
    EXPECT_EQ(table.rankCount, 4U);
}

TEST_F(RankInfoDetectMeshTest, InvalidRootMetadataFailsBeforeConnections)
{
    RankInfoDetectScalable detect(0, 0, 0);
    RankTableInfo table;
    std::string failedList;
    EXPECT_EQ(detect.RootMeshAllGather(table, failedList, 8, std::chrono::steady_clock::now()), HCCL_E_PARA);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 2, 1)), HCCL_SUCCESS);
    EXPECT_EQ(detect.RootMeshAllGather(table, failedList, 8, std::chrono::steady_clock::now()), HCCL_E_PARA);
    // 重复注入拦截后同一对象不可二次注入，换新对象覆盖 meshInfos 缺失场景
    RankInfoDetectScalable detectIncomplete(0, 0, 0);
    auto info = MakeServerInfo(2, 0, 1);
    info.meshInfos.clear();
    ASSERT_EQ(detectIncomplete.SetScalableServerInfo(info), HCCL_SUCCESS);
    EXPECT_EQ(detectIncomplete.RootMeshAllGather(table, failedList, 8, std::chrono::steady_clock::now()), HCCL_E_PARA);
    EXPECT_TRUE(ioOrder.empty());
}

TEST_F(RankInfoDetectMeshTest, DuplicateInjectionRejected)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 0, 1)), HCCL_SUCCESS);
    EXPECT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 1, 1)), HCCL_E_INTERNAL);
}

TEST_F(RankInfoDetectMeshTest, MergeProducesOrderedGlobalTableForUnevenGroups)
{
    RankInfoDetectScalable detect(0, 0, 0);
    detect.partials_.resize(3);
    detect.peerFailedLists_.resize(3);
    u32 firstRank = 0;
    const std::vector<u32> sizes{4, 3, 3};
    const std::vector<std::string> failedLists{"0-1", "", "8"};
    for (u32 root = 0; root < sizes.size(); ++root) {
        auto partial = MakePartial(firstRank, sizes[root]);
        BinaryStream stream;
        // 与生产路径同款序列化（RootMeshAllGather 用 true 保留 localId），测试数据流经相同代码路径。
        ASSERT_EQ(partial.GetBinStream(true, stream), HCCL_SUCCESS);
        stream.Dump(detect.partials_[root]);
        detect.peerFailedLists_[root] = failedLists[root];
        firstRank += sizes[root];
    }
    RankTableInfo merged;
    std::string failedList;
    ASSERT_EQ(detect.MergeMeshPartials(3, merged, failedList), HCCL_SUCCESS);
    ASSERT_EQ(merged.ranks.size(), 10U);
    EXPECT_EQ(merged.rankCount, 10U);
    for (u32 i = 0; i < merged.ranks.size(); ++i) {
        EXPECT_EQ(merged.ranks[i].rankId, i);
    }
    // 各组失败列表按 rootIndex 顺序拼接（分组互斥无需去重），空列表跳过
    EXPECT_EQ(failedList, "0-1,8");
}

TEST_F(RankInfoDetectMeshTest, ConnectFailureJoinsAcceptWorkerBeforeReturning)
{
    RankInfoDetectScalable detect(0, 0, 0);
    auto info = MakeServerInfo(3, 1, 1);
    for (u32 i = 0; i < info.nRoot; ++i) {
        ASSERT_EQ(strcpy_s(info.meshInfos[i].ip, sizeof(info.meshInfos[i].ip), "127.0.0.1"), EOK);
        info.meshInfos[i].meshPort = 60000 + i;
    }
    ASSERT_EQ(detect.SetScalableServerInfo(info), HCCL_SUCCESS);
    detect.meshListenSocket_ = std::make_shared<Socket>(
        nullptr, IpAddress(), 60001, IpAddress(), "mesh-ut", SocketRole::SERVER, NicType::HOST_NIC_TYPE);
    acceptExited.store(false);
    MOCKER_CPP(&RankInfoDetectScalable::MeshConnectToLargerRoots).expects(once()).will(returnValue(HCCL_E_NETWORK));
    MOCKER_CPP(&RankInfoDetectScalable::MeshAcceptSmallerRoots)
        .expects(once())
        .will(invoke(FinishAcceptAfterConnectFailure));
    auto table = MakePartial(1, 1);
    std::string failedList = "1";
    // M3：mesh 连接失败不再中止（缺席对端由空槽体现，由 MergeMeshPartials 反推兜底）
    EXPECT_EQ(detect.RootMeshAllGather(table, failedList, 8, std::chrono::steady_clock::now()), HCCL_SUCCESS);
    EXPECT_TRUE(acceptExited.load());
    EXPECT_EQ(table.rankCount, 1U);
    EXPECT_FALSE(detect.partials_[1].empty()); // 自身槽位已填
    EXPECT_TRUE(detect.partials_[0].empty());  // 缺席对端空槽
    EXPECT_TRUE(detect.partials_[2].empty());
}

// mesh 交换失败容忍 + 全链路聚合（M3）：失败列表在 MergeMeshPartials 聚合出全局列表；
// 「明确拒绝（INIT）→ 分类 UNREACHABLE → 整组反推」全链路经生产代码驱动，无手工状态注入
TEST_F(RankInfoDetectMeshTest, MeshExchangeFailureStillMergesGlobalFailedList)
{
    RankInfoDetectScalable detect(0, 0, 0);
    auto info = MakeServerInfo(2, 0, 4); // 与 totalRankNum=8/nRoot=2 推导的本组区间 [0,4) 自洽
    for (u32 i = 0; i < info.nRoot; ++i) {
        ASSERT_EQ(strcpy_s(info.meshInfos[i].ip, sizeof(info.meshInfos[i].ip), "127.0.0.1"), EOK);
        info.meshInfos[i].meshPort = 60000 + i;
    }
    ASSERT_EQ(detect.SetScalableServerInfo(info), HCCL_SUCCESS);
    // MakeMeshSocket 以 rootHandle_.ip 构造本端 IpAddress，空值抛 Invalid Network Address
    // → 状态记 CONNECT_TIMEOUT（不可整组反推）；须置合法值使「明确拒绝」分类生效
    ASSERT_EQ(strcpy_s(detect.rootHandle_.ip, sizeof(detect.rootHandle_.ip), "127.0.0.1"), EOK);
    detect.meshListenSocket_ = std::make_shared<Socket>(
        nullptr, IpAddress(), 60000, IpAddress(), "mesh-ut", SocketRole::SERVER, NicType::HOST_NIC_TYPE);
    // 真实 MeshConnectToLargerRoots 循环：GetStatus 首次返回 INIT 模拟连接被明确拒绝
    // （等待超时（CONNECT_TIMEOUT）不反推——对端 mesh 晚于其组内收集启动，超时≠失联）
    getStatusRefusedSeq.store(0);
    MOCKER_CPP(&Socket::GetStatus).stubs().will(invoke(StubGetStatusRefusedFirst));
    MOCKER_CPP(&Socket::Close).stubs();

    auto table = MakePartial(0, 1);
    std::string failedList = "0";
    // nRanks=8, nRoot=2：root1 缺席（连接被明确拒绝），反推区间 [4,7]
    ASSERT_EQ(detect.RootMeshAllGather(table, failedList, 8, std::chrono::steady_clock::now()), HCCL_SUCCESS);
    // 生产路径分类：连接被明确拒绝（TCP_CONNECT）→ root1 判不可达，可整组反推
    EXPECT_EQ(detect.peerMeshStates_[1], MeshPeerState::UNREACHABLE);
    EXPECT_EQ(detect.localFailedList_, "0");
    EXPECT_EQ(detect.peerFailedLists_[0], "0");
    ASSERT_EQ(detect.MergeMeshPartials(2, table, failedList), HCCL_SUCCESS);
    EXPECT_EQ(failedList, "0,4-7");
    EXPECT_EQ(table.rankCount, 1U); // 缺员表
}

// 零连接（本组全失联）mesh 空帧传播：空表不再阻断 mesh——RootMeshAllGather 手工构造合法空帧
// （GetBinStream 对 ranks.size()==0 报 E_PARA），本组槽位非空、整组失败列表正常跨组传播；
// 接收侧 MergeMeshPartials 对空组帧（rankCount=0）合并为 rankCount+=0，聚合其失败列表
TEST_F(RankInfoDetectMeshTest, ZeroConnectedRootSpansFailedListViaEmptyFrame)
{
    RankInfoDetectScalable detect(0, 0, 0);
    auto info = MakeServerInfo(2, 0, 4); // 与 totalRankNum=8/nRoot=2 推导的本组区间 [0,4) 自洽
    for (u32 i = 0; i < info.nRoot; ++i) {
        ASSERT_EQ(strcpy_s(info.meshInfos[i].ip, sizeof(info.meshInfos[i].ip), "127.0.0.1"), EOK);
        info.meshInfos[i].meshPort = 60000 + i;
    }
    ASSERT_EQ(detect.SetScalableServerInfo(info), HCCL_SUCCESS);
    // MakeMeshSocket 以 rootHandle_.ip 构造本端 IpAddress，须置合法值（空值抛异常归本地失败）
    ASSERT_EQ(strcpy_s(detect.rootHandle_.ip, sizeof(detect.rootHandle_.ip), "127.0.0.1"), EOK);
    detect.meshListenSocket_ = std::make_shared<Socket>(
        nullptr, IpAddress(), 60000, IpAddress(), "mesh-ut", SocketRole::SERVER, NicType::HOST_NIC_TYPE);

    // root0（本端）零连接：空表 + 整组失败列表 [0,3]；GetStatus 全 OK 模拟 root1 mesh 建链成功并交换
    MOCKER_CPP(&Socket::GetStatus).stubs().will(invoke(StubGetStatusAlwaysOk));
    MOCKER_CPP(&Socket::Close).stubs();
    // root1 在场子表（rank 4）+ 其组失败列表为空
    RankTableInfo peerTable;
    peerTable.version = "2.0";
    peerTable.rankCount = 1;
    hccl::NewRankInfo rankInfo{};
    rankInfo.rankId = 4;
    rankInfo.rankLevelInfos.emplace_back(RankLevelInfo{});
    peerTable.ranks.emplace_back(rankInfo);
    BinaryStream peerStream;
    ASSERT_EQ(peerTable.GetBinStream(true, peerStream), HCCL_SUCCESS);
    std::vector<char> peerBytes;
    peerStream.Dump(peerBytes);
    incoming = Frame(1, peerBytes, "");
    outgoing.clear();

    RankTableInfo emptyTable; // 零连接：空表（ranks 空、rankCount=0）
    std::string failedList = "0-3";
    ASSERT_EQ(detect.RootMeshAllGather(emptyTable, failedList, 8, std::chrono::steady_clock::now()), HCCL_SUCCESS);
    // 本组槽位已填（手工空帧非空字节流），不再被视为缺席空槽
    EXPECT_FALSE(detect.partials_[0].empty());
    EXPECT_EQ(detect.localFailedList_, "0-3");
    // 发出的空帧携带整组失败列表（而非 GetBinStream 的 E_PARA）
    EXPECT_EQ(outgoing, Frame(0, detect.partials_[0], "0-3"));
    // root1 子表落槽
    EXPECT_EQ(detect.peerFailedLists_[1], "");
    ASSERT_EQ(detect.MergeMeshPartials(2, emptyTable, failedList), HCCL_SUCCESS);
    // 全局失败列表 = 零连接组整组区间（root0 经空帧传播）+ root1 无失败
    EXPECT_EQ(failedList, "0-3");
    // 合并表仅含 root1 的 rank4（root0 空组 rankCount+=0）
    EXPECT_EQ(emptyTable.rankCount, 1U);
    ASSERT_EQ(emptyTable.ranks.size(), 1U);
    EXPECT_EQ(emptyTable.ranks[0].rankId, 4U);
}

// SendMeshPartial 发送侧失败列表长度上界：超 MAX_MESH_FAILED_LIST_LEN 拒绝发送（防损坏帧/恶意帧透传）
TEST_F(RankInfoDetectMeshTest, SendMeshPartialRejectsOversizedFailedList)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 0, 1)), HCCL_SUCCESS);
    detect.partials_.push_back({'L'});
    detect.localFailedList_.assign(static_cast<size_t>(MAX_MESH_FAILED_LIST_LEN) + 1, '7');
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "mesh-ut", SocketRole::CLIENT, NicType::HOST_NIC_TYPE);
    EXPECT_EQ(detect.SendMeshPartial(socket), HcclResult::HCCL_E_PARA);
    EXPECT_TRUE(outgoing.empty());
}

// 全局失败列表聚合——多 root 顺序拼接（空列表跳过、无多余逗号、按 rootIndex 排序）
TEST_F(RankInfoDetectMeshTest, MergeAggregatesFailedListsInRootIndexOrder)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(4, 0, 1)), HCCL_SUCCESS);
    detect.partials_.resize(4);
    detect.peerFailedLists_.resize(4);
    const std::vector<std::string> failedLists{"1-2", "", "50", "100-105,200"};
    for (u32 root = 0; root < 4; ++root) {
        auto partial = MakePartial(root * 16, 1);
        BinaryStream stream;
        ASSERT_EQ(partial.GetBinStream(true, stream), HCCL_SUCCESS);
        stream.Dump(detect.partials_[root]);
        detect.peerFailedLists_[root] = failedLists[root];
    }
    RankTableInfo table;
    std::string failedList;
    ASSERT_EQ(detect.MergeMeshPartials(4, table, failedList), HCCL_SUCCESS);
    EXPECT_EQ(failedList, "1-2,50,100-105,200");
    EXPECT_EQ(table.rankCount, 4U);
}

// 缺席反推（rmr=0）——nRanks=64、nRoot=4，root2 缺席 → 整组区间 "32-47"，不逐 id 展开
TEST_F(RankInfoDetectMeshTest, MergeInfersAbsentRangeWhenRemainderZero)
{
    // root0→[0-15]、root1→[16-31]、root2→[32-47]、root3→[48-63]
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(4, 0, 16)), HCCL_SUCCESS);
    detect.totalRankNum_ = 64;
    detect.partials_.resize(4);
    detect.peerFailedLists_.resize(4);
    detect.peerMeshStates_.assign(4, MeshPeerState::PENDING);
    detect.peerMeshStates_[2] = MeshPeerState::UNREACHABLE; // root2 建链未建立 → 可反推
    for (u32 root : {0U, 1U, 3U}) {                         // root2 缺席
        auto partial = MakePartial(root * 16, 16);
        BinaryStream stream;
        ASSERT_EQ(partial.GetBinStream(true, stream), HCCL_SUCCESS);
        stream.Dump(detect.partials_[root]);
        detect.peerFailedLists_[root] = "";
    }
    RankTableInfo table;
    std::string failedList;
    ASSERT_EQ(detect.MergeMeshPartials(4, table, failedList), HCCL_SUCCESS);
    EXPECT_EQ(failedList, "32-47");
    EXPECT_EQ(table.rankCount, 48U); // 缺员表：仅 3 组在场
}

// 缺席反推（rmr≠0 余数摊前的分组边界）——nRanks=66、nRoot=4，
// root0 与 root3 缺席 → "0-16" 与 "50-65" 两段，区间边界与分组算法逐点一致
TEST_F(RankInfoDetectMeshTest, MergeInfersAbsentRangesAtRemainderBoundaries)
{
    // root0→[0-16]、root1→[17-33]、root2→[34-49]、root3→[50-65]；本组 root1 在场
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(4, 1, 17)), HCCL_SUCCESS);
    // 本组首 rank 由 Get 阶段本地推导，须与反推校验的 firstRankFromRoot(66,4,1)=17 一致
    detect.scalableInfo_.groupFirstRank = 17;
    detect.totalRankNum_ = 66;
    detect.partials_.resize(4);
    detect.peerFailedLists_.resize(4);
    detect.peerMeshStates_.assign(4, MeshPeerState::PENDING);
    detect.peerMeshStates_[0] = MeshPeerState::UNREACHABLE; // root0/root3 建链未建立 → 可反推
    detect.peerMeshStates_[3] = MeshPeerState::UNREACHABLE;
    auto selfPartial = MakePartial(17, 17);
    BinaryStream selfStream;
    ASSERT_EQ(selfPartial.GetBinStream(true, selfStream), HCCL_SUCCESS);
    selfStream.Dump(detect.partials_[1]);
    detect.peerFailedLists_[1] = "20"; // 本组失败示例
    auto peerPartial = MakePartial(34, 16);
    BinaryStream peerStream;
    ASSERT_EQ(peerPartial.GetBinStream(true, peerStream), HCCL_SUCCESS);
    peerStream.Dump(detect.partials_[2]);
    detect.peerFailedLists_[2] = "";
    RankTableInfo table;
    std::string failedList;
    ASSERT_EQ(detect.MergeMeshPartials(4, table, failedList), HCCL_SUCCESS);
    // 在场列表按 rootIndex 在前，缺席反推区间（root0、root3）随后追加
    EXPECT_EQ(failedList, "20,0-16,50-65");
    EXPECT_EQ(table.rankCount, 33U);
}

// 单个对端失败后继续尝试后续对端（每个 root 与所有其他 root 建立交换，失败仅记录不中止）：
// root 3 接受 0/1/2，peer 1 缺席 → peer 2 仍被尝试并完成交换
TEST_F(RankInfoDetectMeshTest, AcceptFailureContinuesRemainingPeers)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(4, 3, 1)), HCCL_SUCCESS);
    detect.partials_.resize(4);
    detect.peerFailedLists_.resize(4);
    detect.peerMeshStates_.assign(4, MeshPeerState::PENDING);
    detect.partials_[3] = {'L'};
    detect.meshListenSocket_ = std::make_shared<Socket>(
        nullptr, IpAddress(), 60003, IpAddress(), "mesh-ut", SocketRole::SERVER, NicType::HOST_NIC_TYPE);
    acceptSeq.store(0);
    acceptTimeouts.clear();
    MOCKER_CPP(&Socket::Accept).stubs().will(invoke(StubAcceptSeq));
    MOCKER_CPP(&Socket::Close).stubs();

    incoming = Frame(0, {'A'});
    {
        const std::vector<char> peerFrame = Frame(2, {'B'});
        incoming.insert(incoming.end(), peerFrame.begin(), peerFrame.end());
    }
    outgoing.clear();
    ioOrder.clear();

    EXPECT_EQ(detect.MeshAcceptSmallerRootsCore(60), HcclResult::HCCL_E_TIMEOUT); // 透出首个失败错误码
    // 预算按剩余对端数切片：60/3=20、剩余/2=30、最后一个对端独享剩余预算，缺席对端不独占全量预算
    EXPECT_EQ(acceptTimeouts, std::vector<u32>({20, 30, 60}));
    EXPECT_EQ(detect.partials_[0], std::vector<char>({'A'}));
    EXPECT_TRUE(detect.partials_[1].empty());                 // 超时对端空槽（状态未知，不反推）
    EXPECT_EQ(detect.partials_[2], std::vector<char>({'B'})); // peer 2 仍完成交换
    EXPECT_TRUE(detect.peerFailedLists_[1].empty());
    // accept 等待超时无法区分「明确缺席」与「晚到」（对端 mesh 晚于其组内收集启动）：
    // 记状态未知不反推；缺席对端的整组反推由其 connect 侧「明确拒绝」判定承载
    EXPECT_EQ(detect.peerMeshStates_[1], MeshPeerState::CONNECT_TIMEOUT);
    EXPECT_EQ(detect.meshRecvCount_.load(), 2U);
    EXPECT_EQ(ioOrder, "RRRRSSSSRRRRSSSS"); // 两次完整的先收后发交换
}

// 单对端交换失败就地隔离（返回码路径）：Recv 供应不足返回 false → RecvMeshPartial 返回 HCCL_E_NETWORK，
// 失败不跳出循环、不跳过后续对端，Close 由守卫在失败路径统一执行；
// 生产路径 Recv 超时/断连重抛的异常由同一段 per-peer EXCEPTION_CATCH 收敛为 HCCL_E_INTERNAL，失败隔离逻辑同源
// （mockcpp 桩不可抛异常：分发层行为未定义，故以返回码路径覆盖失败隔离语义）
TEST_F(RankInfoDetectMeshTest, AcceptExchangeFailureIsolatedPerPeerAndSocketClosed)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(3, 2, 1)), HCCL_SUCCESS);
    detect.partials_.resize(3);
    detect.peerFailedLists_.resize(3);
    detect.peerMeshStates_.assign(3, MeshPeerState::PENDING);
    detect.partials_[2] = {'L'};
    detect.meshListenSocket_ = std::make_shared<Socket>(
        nullptr, IpAddress(), 60002, IpAddress(), "mesh-ut", SocketRole::SERVER, NicType::HOST_NIC_TYPE);
    acceptAlwaysCount.store(0);
    MOCKER_CPP(&Socket::Accept).stubs().will(invoke(StubAcceptAlwaysOk));
    closeCount.store(0);
    MOCKER_CPP(&Socket::Close).stubs().will(invoke(CountClose));
    // Recv 沿用 SetUp 的 SupplyRecv：incoming 为空 → 供应不足返回 false → 交换失败（HCCL_E_NETWORK）

    EXPECT_EQ(detect.MeshAcceptSmallerRootsCore(60), HcclResult::HCCL_E_NETWORK);
    EXPECT_EQ(acceptAlwaysCount.load(), 2U); // peer 0 失败后 peer 1 仍被 Accept（不被整体跳过）
    EXPECT_EQ(closeCount.load(), 2U);        // 两个对端 socket 均在失败路径被守卫 Close
    EXPECT_TRUE(detect.partials_[0].empty());
    EXPECT_TRUE(detect.partials_[1].empty());
    EXPECT_EQ(detect.peerMeshStates_[0], MeshPeerState::EXCHANGE_FAILED); // 已建链交换失败 → 状态未知
    EXPECT_EQ(detect.peerMeshStates_[1], MeshPeerState::EXCHANGE_FAILED);
    EXPECT_EQ(detect.meshRecvCount_.load(), 0U);
}

// 合并中段失败（子表 rankCount 与 ranks.size 不匹配）输出保持调用方原值：失败列表与子表
// 一并原子提交，杜绝「失败列表已跨组聚合、子表仍是本组表」的半合并状态随广播下发
TEST_F(RankInfoDetectMeshTest, MergeFailureKeepsCallerOutputsConsistent)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(3, 0, 4)), HCCL_SUCCESS);
    detect.totalRankNum_ = 10;
    detect.partials_.resize(3);
    detect.peerFailedLists_.resize(3);
    detect.peerMeshStates_.assign(3, MeshPeerState::PENDING);
    detect.peerMeshStates_[2] = MeshPeerState::UNREACHABLE; // root2 缺席，反推区间 [7,9]
    // 本组（调用方原值）：表 = 4 rank，失败列表 = "1-2"
    auto selfPartial = MakePartial(0, 4);
    BinaryStream selfStream;
    ASSERT_EQ(selfPartial.GetBinStream(true, selfStream), HCCL_SUCCESS);
    selfStream.Dump(detect.partials_[0]);
    detect.peerFailedLists_[0] = "1-2";
    // root1 子表构造 rankCount 不匹配（声明 3 实际 2）→ 合并中段失败
    auto corruptPartial = MakePartial(4, 2);
    corruptPartial.rankCount = 3;
    BinaryStream corruptStream;
    ASSERT_EQ(corruptPartial.GetBinStream(true, corruptStream), HCCL_SUCCESS);
    corruptStream.Dump(detect.partials_[1]);

    RankTableInfo table = selfPartial; // 调用方原值：本组表
    std::string failedList = "1-2";    // 调用方原值：本组失败列表
    EXPECT_EQ(detect.MergeMeshPartials(3, table, failedList), HCCL_E_INTERNAL);
    // 原子性：两个输出均保持调用方原值，未被跨组内容污染
    EXPECT_EQ(failedList, "1-2");
    EXPECT_EQ(table.rankCount, 4U);
    ASSERT_EQ(table.ranks.size(), 4U);
    EXPECT_EQ(table.ranks[0].rankId, 0U);
}

// 空槽分类（反推依据细分）：仅 UNREACHABLE（连接被明确拒绝）整组反推；
// CONNECT_TIMEOUT（对端 mesh 晚于组内收集启动，超时≠失联）/EXCHANGE_FAILED/PENDING 状态未知不反推
TEST_F(RankInfoDetectMeshTest, MergeOnlyInfersUnreachablePeersNotUnknownOnes)
{
    // root0→[0-15]、root1→[16-31]、root2→[32-47]、root3→[48-63]
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(4, 0, 16)), HCCL_SUCCESS);
    detect.totalRankNum_ = 64;
    detect.partials_.resize(4);
    detect.peerFailedLists_.resize(4);
    detect.peerMeshStates_.assign(4, MeshPeerState::PENDING);
    detect.peerMeshStates_[1] = MeshPeerState::UNREACHABLE;     // 连接被明确拒绝：可反推
    detect.peerMeshStates_[2] = MeshPeerState::CONNECT_TIMEOUT; // 建链等待超时（慢启动/黑洞）：状态未知
    detect.peerMeshStates_[3] = MeshPeerState::EXCHANGE_FAILED; // 已建链交换未收齐：状态未知（PENDING 同路径）
    auto selfPartial = MakePartial(0, 16);
    BinaryStream stream;
    ASSERT_EQ(selfPartial.GetBinStream(true, stream), HCCL_SUCCESS);
    stream.Dump(detect.partials_[0]);
    detect.peerFailedLists_[0] = "5";
    RankTableInfo table;
    std::string failedList;
    ASSERT_EQ(detect.MergeMeshPartials(4, table, failedList), HCCL_SUCCESS);
    EXPECT_EQ(failedList, "5,16-31"); // 仅被明确拒绝的 root1 反推；root2/root3 不臆造整组失败
    EXPECT_EQ(table.rankCount, 16U);  // 缺员表：仅本组在场
}

// connect 侧预算切片 + 不可达分类：缺席对端只消耗自己的预算份额，后续存活对端仍完成交换。
// Connect 为虚函数不可 hook（mockcpp 对虚成员指针的 vtable 偏移编码不可 patch），
// 但 UT stub 的 Socket::Connect 本为空实现，无需 mock；GetStatus patch 覆盖 stub 默认返回以控制建链结果
TEST_F(RankInfoDetectMeshTest, ConnectSlicesBudgetPerPeerAndContinuesRemainingPeers)
{
    RankInfoDetectScalable detect(0, 0, 0);
    // meshInfos/rootHandle 须填合法地址：MakeMeshSocket 真实执行，空地址会被 IpAddress 判非法抛异常
    auto info = MakeServerInfo(4, 0, 1);
    for (u32 i = 0; i < info.nRoot; ++i) {
        ASSERT_EQ(strcpy_s(info.meshInfos[i].ip, sizeof(info.meshInfos[i].ip), "127.0.0.1"), EOK);
        info.meshInfos[i].meshPort = 60000 + i;
    }
    ASSERT_EQ(detect.SetScalableServerInfo(info), HCCL_SUCCESS);
    ASSERT_EQ(strcpy_s(detect.rootHandle_.ip, sizeof(detect.rootHandle_.ip), "127.0.0.1"), EOK);
    detect.partials_.resize(4);
    detect.peerFailedLists_.resize(4);
    detect.peerMeshStates_.assign(4, MeshPeerState::PENDING);
    detect.partials_[0] = {'L'};
    getStatusSeq.store(0);
    getStatusTimeouts.clear();
    MOCKER_CPP(&Socket::GetStatus).stubs().will(invoke(StubGetStatusTimeoutFirst));
    MOCKER_CPP(&Socket::Close).stubs();

    incoming = Frame(2, {'B'});
    {
        const std::vector<char> peerFrame = Frame(3, {'C'});
        incoming.insert(incoming.end(), peerFrame.begin(), peerFrame.end());
    }
    outgoing.clear();
    ioOrder.clear();

    EXPECT_EQ(detect.MeshConnectToLargerRoots(60), HcclResult::HCCL_E_TIMEOUT); // 透出首个失败错误码
    // 预算切片：60/3=20、剩余/2=30、最后一个对端独享剩余预算
    EXPECT_EQ(getStatusTimeouts, std::vector<u32>({20, 30, 60}));
    EXPECT_TRUE(detect.partials_[1].empty());                 // 超时对端空槽
    EXPECT_EQ(detect.partials_[2], std::vector<char>({'B'})); // 存活对端仍完成交换
    EXPECT_EQ(detect.partials_[3], std::vector<char>({'C'}));
    // 超时 ≠ 失联（对端 mesh 晚于其组内收集启动）：记状态未知，不整组反推
    EXPECT_EQ(detect.peerMeshStates_[1], MeshPeerState::CONNECT_TIMEOUT);
    EXPECT_EQ(detect.meshRecvCount_.load(), 2U);
}

// mesh 预算以 accept 起始时刻为锚收缩（检视缺陷：按阶段耗时扣减漏计 accept 后的收集耗时）：
// linkTimeOut=100、accept 起点 0s、mesh 入口 now()=10s → 剩余 90，扣 40 余量 → 总预算 50，
// nRoot=2 单对端独享全额（SliceMeshWaitSeconds 对 remainPeers=1 返回全部）；随后对端建链超时，
// 预算耗尽后退出（无后续对端）。兜底分支（剩余 ≤ 40 时退化 1s）见
// MeshBudgetFallsBackToOneSecondWhenRemainderBelowMargin
TEST_F(RankInfoDetectMeshTest, MeshBudgetShrinksByElapsedSinceAcceptStart)
{
    RankInfoDetectScalable detect(0, 0, 0);
    auto info = MakeServerInfo(2, 0, 1);
    for (u32 i = 0; i < info.nRoot; ++i) {
        ASSERT_EQ(strcpy_s(info.meshInfos[i].ip, sizeof(info.meshInfos[i].ip), "127.0.0.1"), EOK);
        info.meshInfos[i].meshPort = 60000 + i;
    }
    ASSERT_EQ(detect.SetScalableServerInfo(info), HCCL_SUCCESS);
    ASSERT_EQ(strcpy_s(detect.rootHandle_.ip, sizeof(detect.rootHandle_.ip), "127.0.0.1"), EOK);
    detect.partials_.resize(2);
    detect.peerFailedLists_.resize(2);
    detect.partials_[0] = {'L'};
    detect.meshListenSocket_ = std::make_shared<Socket>(
        nullptr, IpAddress(), 60000, IpAddress(), "mesh-ut", SocketRole::SERVER, NicType::HOST_NIC_TYPE);

    // linkTimeOut=100s
    static EnvSocketConfig fakeEnvSocketConfig;
    fakeEnvSocketConfig.linkTimeOut = CfgField<s32>{"HCCL_CONNECT_TIMEOUT", s32(100), Str2T<s32>};
    fakeEnvSocketConfig.linkTimeOut.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetSocketConfig).stubs().will(returnValue(fakeEnvSocketConfig));

    // 固定时钟序列：RootMeshAllGather 内多次 now()（预算计算/RemainingMeshWaitSeconds/切片），
    // 全部锚定同一时刻 10s（accept 起始 0s → elapsed=10）
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(10))));

    getStatusTimeouts.clear();
    MOCKER_CPP(&Socket::GetStatus).stubs().will(invoke(StubGetStatusAlwaysTimeout));
    MOCKER_CPP(&Socket::Close).stubs();
    incoming.clear();
    outgoing.clear();

    // accept 起始时刻 0s：mesh 预算 = 100 - 10(elapsed) - 40(余量) = 50
    const auto acceptStartTime = std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0));
    EXPECT_EQ(detect.RootMeshAllGather(RankTableInfo(), "", 8, acceptStartTime), HCCL_SUCCESS);
    // 单对端独享全额预算 50s（若未扣 elapsed 会得 60，漏计的收集耗时恰好暴露算式错误）
    EXPECT_EQ(getStatusTimeouts, std::vector<u32>({50}));
    EXPECT_EQ(detect.peerMeshStates_[1], MeshPeerState::CONNECT_TIMEOUT);
}

// mesh 预算 1s 兜底分支：剩余 ≤ SOCKET_LINK_EXTRA_TIMEOUT(40) 时退化 1s（缺员/零连接场景 accept 耗满
// linkTimeOut 后 elapsed ≥ linkTimeOut-40 的实际路径）——linkTimeOut=100、accept 起点 0s、
// now()=60s → 剩余 40 不大于余量 → timeoutSec=1，单对端以 1s 预等供建链超时
TEST_F(RankInfoDetectMeshTest, MeshBudgetFallsBackToOneSecondWhenRemainderBelowMargin)
{
    RankInfoDetectScalable detect(0, 0, 0);
    auto info = MakeServerInfo(2, 0, 1);
    for (u32 i = 0; i < info.nRoot; ++i) {
        ASSERT_EQ(strcpy_s(info.meshInfos[i].ip, sizeof(info.meshInfos[i].ip), "127.0.0.1"), EOK);
        info.meshInfos[i].meshPort = 60000 + i;
    }
    ASSERT_EQ(detect.SetScalableServerInfo(info), HCCL_SUCCESS);
    ASSERT_EQ(strcpy_s(detect.rootHandle_.ip, sizeof(detect.rootHandle_.ip), "127.0.0.1"), EOK);
    detect.partials_.resize(2);
    detect.peerFailedLists_.resize(2);
    detect.partials_[0] = {'L'};
    detect.meshListenSocket_ = std::make_shared<Socket>(
        nullptr, IpAddress(), 60000, IpAddress(), "mesh-ut", SocketRole::SERVER, NicType::HOST_NIC_TYPE);

    // linkTimeOut=100s
    static EnvSocketConfig fallbackEnvSocketConfig;
    fallbackEnvSocketConfig.linkTimeOut = CfgField<s32>{"HCCL_CONNECT_TIMEOUT", s32(100), Str2T<s32>};
    fallbackEnvSocketConfig.linkTimeOut.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetSocketConfig).stubs().will(returnValue(fallbackEnvSocketConfig));

    // 固定时钟 60s：剩余 100-60=40，不大于 40 余量 → 兜底 1s
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(60))));

    getStatusTimeouts.clear();
    MOCKER_CPP(&Socket::GetStatus).stubs().will(invoke(StubGetStatusAlwaysTimeout));
    MOCKER_CPP(&Socket::Close).stubs();
    incoming.clear();
    outgoing.clear();

    const auto acceptStartTime = std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0));
    EXPECT_EQ(detect.RootMeshAllGather(RankTableInfo(), "", 8, acceptStartTime), HCCL_SUCCESS);
    // 兜底预算 1s（若误走正常分支会得 0 或 100-60-40=0 的 max(…,1) 混淆——精确断言 1 区分）
    EXPECT_EQ(getStatusTimeouts, std::vector<u32>({1}));
    EXPECT_EQ(detect.peerMeshStates_[1], MeshPeerState::CONNECT_TIMEOUT);
}

// 连接被明确拒绝（GetStatus 返回 INIT，对端 root 进程不在）：记 UNREACHABLE 可整组反推，后续对端不受影响
TEST_F(RankInfoDetectMeshTest, ConnectRefusedPeerMarkedUnreachable)
{
    RankInfoDetectScalable detect(0, 0, 0);
    auto info = MakeServerInfo(4, 0, 1);
    for (u32 i = 0; i < info.nRoot; ++i) {
        ASSERT_EQ(strcpy_s(info.meshInfos[i].ip, sizeof(info.meshInfos[i].ip), "127.0.0.1"), EOK);
        info.meshInfos[i].meshPort = 60000 + i;
    }
    ASSERT_EQ(detect.SetScalableServerInfo(info), HCCL_SUCCESS);
    ASSERT_EQ(strcpy_s(detect.rootHandle_.ip, sizeof(detect.rootHandle_.ip), "127.0.0.1"), EOK);
    detect.partials_.resize(4);
    detect.peerFailedLists_.resize(4);
    detect.peerMeshStates_.assign(4, MeshPeerState::PENDING);
    detect.partials_[0] = {'L'};
    getStatusRefusedSeq.store(0);
    MOCKER_CPP(&Socket::GetStatus).stubs().will(invoke(StubGetStatusRefusedFirst));
    MOCKER_CPP(&Socket::Close).stubs();

    incoming = Frame(2, {'B'});
    {
        const std::vector<char> peerFrame = Frame(3, {'C'});
        incoming.insert(incoming.end(), peerFrame.begin(), peerFrame.end());
    }
    outgoing.clear();
    ioOrder.clear();

    EXPECT_EQ(detect.MeshConnectToLargerRoots(60), HcclResult::HCCL_E_TCP_CONNECT); // 透出首个失败错误码
    EXPECT_TRUE(detect.partials_[1].empty());                                       // 被拒对端空槽
    EXPECT_EQ(detect.partials_[2], std::vector<char>({'B'})); // 后续对端不受影响仍完成交换
    EXPECT_EQ(detect.partials_[3], std::vector<char>({'C'}));
    // 明确拒绝 = 对端 root 进程不在：唯一可整组反推的失败类别
    EXPECT_EQ(detect.peerMeshStates_[1], MeshPeerState::UNREACHABLE);
    EXPECT_EQ(detect.meshRecvCount_.load(), 2U);
}

// 建链仍进行中（GetStatus 返回 CONNECTING，对端 mesh 晚于其组内收集启动）：未获明确拒绝前
// 超时≠失联，记状态未知不反推（防慢启动对端组被错报全体失败），后续对端不受影响
TEST_F(RankInfoDetectMeshTest, ConnectStillConnectingPeerNotInferredAbsent)
{
    RankInfoDetectScalable detect(0, 0, 0);
    auto info = MakeServerInfo(4, 0, 1);
    for (u32 i = 0; i < info.nRoot; ++i) {
        ASSERT_EQ(strcpy_s(info.meshInfos[i].ip, sizeof(info.meshInfos[i].ip), "127.0.0.1"), EOK);
        info.meshInfos[i].meshPort = 60000 + i;
    }
    ASSERT_EQ(detect.SetScalableServerInfo(info), HCCL_SUCCESS);
    ASSERT_EQ(strcpy_s(detect.rootHandle_.ip, sizeof(detect.rootHandle_.ip), "127.0.0.1"), EOK);
    detect.partials_.resize(4);
    detect.peerFailedLists_.resize(4);
    detect.peerMeshStates_.assign(4, MeshPeerState::PENDING);
    detect.partials_[0] = {'L'};
    getStatusConnectingSeq.store(0);
    MOCKER_CPP(&Socket::GetStatus).stubs().will(invoke(StubGetStatusConnectingFirst));
    MOCKER_CPP(&Socket::Close).stubs();

    incoming = Frame(2, {'B'});
    {
        const std::vector<char> peerFrame = Frame(3, {'C'});
        incoming.insert(incoming.end(), peerFrame.begin(), peerFrame.end());
    }
    outgoing.clear();
    ioOrder.clear();

    EXPECT_EQ(detect.MeshConnectToLargerRoots(60), HcclResult::HCCL_E_TIMEOUT); // 透出首个失败错误码
    EXPECT_TRUE(detect.partials_[1].empty());                                   // 慢启动对端空槽
    EXPECT_EQ(detect.partials_[2], std::vector<char>({'B'})); // 后续对端不受影响仍完成交换
    EXPECT_EQ(detect.partials_[3], std::vector<char>({'C'}));
    // 仍在建链 ≠ 明确拒绝：状态未知，不整组反推
    EXPECT_EQ(detect.peerMeshStates_[1], MeshPeerState::CONNECT_TIMEOUT);
    EXPECT_EQ(detect.meshRecvCount_.load(), 2U);
}

// 建链准备本地失败（MakeMeshSocket 抛 InvalidParamsException，如 RA 句柄创建失败）就地隔离：
// 本地失败与对端无关，归状态未知不反推（生产路径地址已在 RootMeshAllGather 预校验），
// 不消耗预算、不上升为整体失败，后续对端仍完成交换
TEST_F(RankInfoDetectMeshTest, ConnectLocalFailureIsolatedPerPeer)
{
    RankInfoDetectScalable detect(0, 0, 0);
    auto info = MakeServerInfo(4, 0, 1);
    for (u32 i = 0; i < info.nRoot; ++i) {
        if (i == 1) {
            continue; // peer 1 地址保持默认全零：IpAddress 判非法抛 InvalidParamsException
        }
        ASSERT_EQ(strcpy_s(info.meshInfos[i].ip, sizeof(info.meshInfos[i].ip), "127.0.0.1"), EOK);
        info.meshInfos[i].meshPort = 60000 + i;
    }
    ASSERT_EQ(detect.SetScalableServerInfo(info), HCCL_SUCCESS);
    ASSERT_EQ(strcpy_s(detect.rootHandle_.ip, sizeof(detect.rootHandle_.ip), "127.0.0.1"), EOK);
    detect.partials_.resize(4);
    detect.peerFailedLists_.resize(4);
    detect.peerMeshStates_.assign(4, MeshPeerState::PENDING);
    detect.partials_[0] = {'L'};
    MOCKER_CPP(&Socket::GetStatus).stubs().will(invoke(StubGetStatusAlwaysOk));
    MOCKER_CPP(&Socket::Close).stubs();

    incoming = Frame(2, {'B'});
    {
        const std::vector<char> peerFrame = Frame(3, {'C'});
        incoming.insert(incoming.end(), peerFrame.begin(), peerFrame.end());
    }
    outgoing.clear();
    ioOrder.clear();

    // 首个失败码来自 peer 1 建链准备异常（HCCL_E_INTERNAL），整体不中止
    EXPECT_EQ(detect.MeshConnectToLargerRoots(60), HcclResult::HCCL_E_INTERNAL);
    // 建链准备残余异常均为本地失败，与「对端 root 进程不在」无关：状态未知不反推
    EXPECT_EQ(detect.peerMeshStates_[1], MeshPeerState::CONNECT_TIMEOUT);
    EXPECT_EQ(detect.partials_[2], std::vector<char>({'B'})); // 后续对端不受影响仍完成交换
    EXPECT_EQ(detect.partials_[3], std::vector<char>({'C'}));
    EXPECT_EQ(detect.meshRecvCount_.load(), 2U);
}

// u32 回绕守卫：自定义分组策略返回超界 nRank 使 first+nRank-1 回绕（first>last）时，跳过该非法段，
// 不污染 globalFailedList（agent 侧遇 first>last 会整体判解析失败，失败总数丢失）。
// RootGroupStrategy 成员为裸函数指针，带捕获 lambda 不可转换——trampoline + 文件级静态指针承载
// 被包装的默认回调（无捕获 lambda/自由函数可隐式转换）；用例结束经槽位指针还原默认策略，
// 不污染同二进制后续用例（含 --gtest_filter 单独执行场景）
namespace {
u32 (*g_wrappedNRankFromRoot)(u32, u32, u32) = nullptr;
u32 WrappedNRankFromRoot(u32 nRanks, u32 nRoot, u32 rootIndex)
{
    return rootIndex == 1 ? 0xFFFFFFFFU : g_wrappedNRankFromRoot(nRanks, nRoot, rootIndex);
}

// 策略槽位 RAII 还原：ASSERT 提前退出也会析构还原，避免自定义策略泄漏污染同二进制后续用例
struct RootGroupStrategyRestoreGuard {
    RootGroupStrategy saved_{};
    RootGroupStrategy* slot_{nullptr};
    explicit RootGroupStrategyRestoreGuard()
    {
        const RootGroupStrategy* slot = GetRootGroupStrategy();
        if (slot != nullptr) {
            saved_ = *slot;
            slot_ = const_cast<RootGroupStrategy*>(slot);
        }
    }
    ~RootGroupStrategyRestoreGuard()
    {
        if (slot_ != nullptr) {
            *slot_ = saved_; // 直接写回槽位（StrategySlot 静态对象地址），绕过防重复覆盖约束
        }
        g_wrappedNRankFromRoot = nullptr;
    }
};
} // namespace

TEST_F(RankInfoDetectMeshTest, MergeSkipsWrappedInferredRangeFromCustomStrategy)
{
    // RAII guard：任何退出路径（含 ASSERT 失败 return）都还原默认策略
    RootGroupStrategyRestoreGuard restoreGuard;
    ASSERT_NE(restoreGuard.slot_, nullptr);

    // 自定义策略：root1 的 nRank 返回超大值（UINT32_MAX）触发回绕，其余回调与默认策略一致
    g_wrappedNRankFromRoot = restoreGuard.slot_->nRankFromRoot;
    RootGroupStrategy custom = restoreGuard.saved_;
    custom.nRankFromRoot = &WrappedNRankFromRoot; // trampoline：裸函数指针，绕过带捕获 lambda 不可转换
    ASSERT_EQ(RegisterRootGroupStrategy(custom), HCCL_SUCCESS);

    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 0, 4)), HCCL_SUCCESS);
    detect.totalRankNum_ = 8;
    detect.partials_.resize(2);
    detect.peerFailedLists_.resize(2);
    detect.peerMeshStates_.assign(2, MeshPeerState::PENDING);
    detect.peerMeshStates_[1] = MeshPeerState::UNREACHABLE; // root1 缺席，反推段 first=4、nRank=UINT32_MAX 回绕
    auto selfPartial = MakePartial(0, 4);
    BinaryStream stream;
    ASSERT_EQ(selfPartial.GetBinStream(true, stream), HCCL_SUCCESS);
    stream.Dump(detect.partials_[0]);
    detect.peerFailedLists_[0] = "1-2";
    RankTableInfo table;
    std::string failedList;
    ASSERT_EQ(detect.MergeMeshPartials(2, table, failedList), HCCL_SUCCESS);
    // 回绕段 [4, 4+UINT32_MAX-1] 被守卫跳过，仅剩本组在场失败列表
    EXPECT_EQ(failedList, "1-2");
    EXPECT_EQ(table.rankCount, 4U);
}
