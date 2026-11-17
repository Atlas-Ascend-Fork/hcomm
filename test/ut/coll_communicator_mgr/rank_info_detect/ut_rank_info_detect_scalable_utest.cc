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

std::vector<char> Frame(u32 source, const std::vector<char>& payload)
{
    std::vector<char> bytes;
    AppendValue(bytes, source);
    AppendValue(bytes, static_cast<u64>(payload.size()));
    bytes.insert(bytes.end(), payload.begin(), payload.end());
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
    EXPECT_EQ(detect.RootMeshAllGather(table), HCCL_SUCCESS);
    EXPECT_EQ(table.rankCount, 8U);
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
                EXPECT_TRUE(incoming.empty());
                EXPECT_EQ(ioOrder, self < peer ? "SSSRRR" : "RRRSSS");
            }
            EXPECT_EQ(detect.meshRecvCount_.load(), count - 1);
        }
    }
}

TEST_F(RankInfoDetectMeshTest, RejectsWrongSourceAndInvalidLengths)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 0, 1)), HCCL_SUCCESS);
    detect.partials_.resize(2);
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "mesh-ut", SocketRole::CLIENT, NicType::HOST_NIC_TYPE);
    incoming = Frame(0, {'x'});
    EXPECT_EQ(detect.RecvMeshPartial(socket, 1), HCCL_E_PARA);
    for (u64 size : {u64{0}, static_cast<u64>(MAX_BUFFER_LEN) + 1}) {
        incoming.clear();
        AppendValue(incoming, u32{1});
        AppendValue(incoming, size);
        EXPECT_EQ(detect.RecvMeshPartial(socket, 1), HCCL_E_PARA);
        EXPECT_TRUE(detect.partials_[1].empty());
    }
}

TEST_F(RankInfoDetectMeshTest, RejectsDuplicatePartialAndTruncatedBody)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 0, 1)), HCCL_SUCCESS);
    detect.partials_.resize(2);
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "mesh-ut", SocketRole::CLIENT, NicType::HOST_NIC_TYPE);
    incoming = Frame(1, {'x', 'y'});
    incoming.pop_back();
    EXPECT_EQ(detect.RecvMeshPartial(socket, 1), HCCL_E_NETWORK);
    EXPECT_TRUE(detect.partials_[1].empty());
    incoming = Frame(1, {'x'});
    ASSERT_EQ(detect.RecvMeshPartial(socket, 1), HCCL_SUCCESS);
    incoming = Frame(1, {'y'});
    EXPECT_EQ(detect.RecvMeshPartial(socket, 1), HCCL_E_PARA);
    EXPECT_EQ(detect.partials_[1], std::vector<char>({'x'}));
}

TEST_F(RankInfoDetectMeshTest, ReceiveFailureStopsExchangeBeforeReply)
{
    RankInfoDetectScalable detect(0, 0, 0);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 1, 1)), HCCL_SUCCESS);
    detect.partials_.resize(2);
    detect.partials_[1] = {'L'};
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "mesh-ut", SocketRole::SERVER, NicType::HOST_NIC_TYPE);
    EXPECT_EQ(detect.ExchangeMeshPartial(socket, 0, false), HCCL_E_NETWORK);
    EXPECT_TRUE(outgoing.empty());
    EXPECT_EQ(detect.meshRecvCount_.load(), 0U);
}

TEST_F(RankInfoDetectMeshTest, MergeRejectsMissingPartialWithoutChangingOutput)
{
    RankInfoDetectScalable detect(0, 0, 0);
    detect.partials_.resize(2);
    RankTableInfo table;
    table.rankCount = 8;
    EXPECT_EQ(detect.MergeMeshPartials(2, table), HCCL_E_INTERNAL);
    EXPECT_EQ(table.rankCount, 8U);
}

TEST_F(RankInfoDetectMeshTest, InvalidRootMetadataFailsBeforeConnections)
{
    RankInfoDetectScalable detect(0, 0, 0);
    RankTableInfo table;
    EXPECT_EQ(detect.RootMeshAllGather(table), HCCL_E_PARA);
    ASSERT_EQ(detect.SetScalableServerInfo(MakeServerInfo(2, 2, 1)), HCCL_SUCCESS);
    EXPECT_EQ(detect.RootMeshAllGather(table), HCCL_E_PARA);
    // 重复注入拦截后同一对象不可二次注入，换新对象覆盖 meshInfos 缺失场景
    RankInfoDetectScalable detectIncomplete(0, 0, 0);
    auto info = MakeServerInfo(2, 0, 1);
    info.meshInfos.clear();
    ASSERT_EQ(detectIncomplete.SetScalableServerInfo(info), HCCL_SUCCESS);
    EXPECT_EQ(detectIncomplete.RootMeshAllGather(table), HCCL_E_PARA);
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
    u32 firstRank = 0;
    const std::vector<u32> sizes{4, 3, 3};
    for (u32 root = 0; root < sizes.size(); ++root) {
        auto partial = MakePartial(firstRank, sizes[root]);
        BinaryStream stream;
        // 与生产路径同款序列化（RootMeshAllGather 用 true 保留 localId），测试数据流经相同代码路径。
        ASSERT_EQ(partial.GetBinStream(true, stream), HCCL_SUCCESS);
        stream.Dump(detect.partials_[root]);
        firstRank += sizes[root];
    }
    RankTableInfo merged;
    ASSERT_EQ(detect.MergeMeshPartials(3, merged), HCCL_SUCCESS);
    ASSERT_EQ(merged.ranks.size(), 10U);
    EXPECT_EQ(merged.rankCount, 10U);
    for (u32 i = 0; i < merged.ranks.size(); ++i) {
        EXPECT_EQ(merged.ranks[i].rankId, i);
    }
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
    EXPECT_EQ(detect.RootMeshAllGather(table), HCCL_E_NETWORK);
    EXPECT_TRUE(acceptExited.load());
    EXPECT_EQ(table.rankCount, 1U);
}
