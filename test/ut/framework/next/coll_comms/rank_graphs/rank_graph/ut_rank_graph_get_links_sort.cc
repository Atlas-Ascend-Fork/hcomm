/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "rank_gph.h"
#include "rank_graph_interface.h"

using namespace Hccl;

namespace {
constexpr RankId MY_RANK = 0;
constexpr RankId SRC_RANK = 0;
constexpr RankId DST_RANK = 1;
// 使用 netLayer=0：InitInnerRanks 依赖 netLayer[0] 的 NetInstance 存在；
// 同时 Peer::AddNetInstance 会把该 layer 注册到 peer 的 levels，使 GetLevels 校验通过
constexpr u32 NET_LAYER = 0;

// 创建Peer辅助函数
std::shared_ptr<NetInstance::Peer> CreatePeer(RankId rankId, LocalId localId, DeviceId deviceId)
{
    return std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
}

// 按hop值查找link在linkList中的索引，避免顺序调整后用例语义漂移
int FindLinkIndexByHop(CommLink* linkList, uint32_t size, u32 hop)
{
    for (uint32_t i = 0; i < size; ++i) {
        if (linkList[i].linkAttr.hop == hop) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// 构造一条Clos链路（src->fabric->dst）并加入NetInstance
// srcPortNum控制源端点port口数量（单port物理口=1，多port pg口>1）
// hop用于标识该条链路，便于断言时按属性查找
void AddClosLink(
    NetInstance& netInst, const std::shared_ptr<NetInstance::Peer>& srcPeer,
    const std::shared_ptr<NetInstance::Peer>& dstPeer, const std::shared_ptr<NetInstance::Fabric>& fabric,
    u32 srcPortNum, u32 hop)
{
    std::set<std::string> ports;
    for (u32 i = 0; i < srcPortNum; ++i) {
        ports.insert(std::to_string(hop) + "/" + std::to_string(i));
    }
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_TP};
    auto srcIface = std::make_shared<NetInstance::ConnInterface>(
        IpAddress("ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffe"), ports, AddrPosition::DEVICE, LinkType::PEER2NET,
        protocols, TopoType::CLOS, hop);
    auto dstIface = std::make_shared<NetInstance::ConnInterface>(
        IpAddress("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"), ports, AddrPosition::DEVICE, LinkType::PEER2NET,
        protocols, TopoType::CLOS, hop);
    srcPeer->AddConnInterface(NET_LAYER, srcIface);
    dstPeer->AddConnInterface(NET_LAYER, dstIface);
    netInst.AddLink(std::make_shared<NetInstance::Link>(
        srcPeer, fabric, srcIface, nullptr, LinkType::PEER2NET, protocols, LinkDirection::BOTH, hop));
    netInst.AddLink(std::make_shared<NetInstance::Link>(
        fabric, dstPeer, nullptr, dstIface, LinkType::PEER2NET, protocols, LinkDirection::BOTH, hop));
}

// 构造一条P2P链路并加入NetInstance
void AddP2PLink(
    NetInstance& netInst, const std::shared_ptr<NetInstance::Peer>& srcPeer,
    const std::shared_ptr<NetInstance::Peer>& dstPeer, u32 hop)
{
    std::set<std::string> ports = {"p2p/0"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_TP};
    auto srcIface = std::make_shared<NetInstance::ConnInterface>(
        IpAddress("::1"), ports, AddrPosition::DEVICE, LinkType::PEER2PEER, protocols);
    auto dstIface = std::make_shared<NetInstance::ConnInterface>(
        IpAddress("::2"), ports, AddrPosition::DEVICE, LinkType::PEER2PEER, protocols);
    srcPeer->AddConnInterface(NET_LAYER, srcIface);
    dstPeer->AddConnInterface(NET_LAYER, dstIface);
    netInst.AddLink(std::make_shared<NetInstance::Link>(
        srcPeer, dstPeer, srcIface, dstIface, LinkType::PEER2PEER, protocols, LinkDirection::BOTH, hop));
}
} // namespace

class RankGraphGetLinksSortTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankGraphGetLinksSortTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "RankGraphGetLinksSortTest TearDown" << std::endl; }
    virtual void SetUp() {}
    virtual void TearDown() { GlobalMockObject::verify(); }
};

// 场景：混合拓扑（2条p2p + 4条clos，clos port数分别=1,3,2,2）
// 预期：p2p在前保持原序；clos按源端点port数降序（3>2>2>1），port数相同保持原相对顺序；
//       所有p2p链路索引 < 所有clos链路索引
TEST_F(RankGraphGetLinksSortTest, Ut_GetLinks_When_MixedP2PAndClos_Expect_P2PBeforeClosSortedByPortNumDesc)
{
    RankGraph rankGraph(MY_RANK);
    // InnerNetInstance::GetPaths 先返回 p2p（TraverseEdge srcPeerId→dstPeerId），
    // 再按 fabric 顺序返回 clos，p2p 在前 clos 在后不交替
    auto netInst = std::make_shared<InnerNetInstance>(NET_LAYER, "inner");
    auto srcPeer = CreatePeer(SRC_RANK, 0, 0);
    auto dstPeer = CreatePeer(DST_RANK, 1, 1);
    netInst->AddRankId(SRC_RANK);
    netInst->AddRankId(DST_RANK);
    (void)netInst->AddNode(srcPeer);
    (void)netInst->AddNode(dstPeer);
    (void)rankGraph.AddPeer(srcPeer);
    (void)rankGraph.AddPeer(dstPeer);

    // 2条p2p链路，hop=1,2，加入顺序即原相对顺序
    AddP2PLink(*netInst, srcPeer, dstPeer, 1); // p2pFirst: hop=1
    AddP2PLink(*netInst, srcPeer, dstPeer, 2); // p2pSecond: hop=2

    // 4个fabric对应4条clos链路，port数依次为1,3,2,2
    auto fabricA = std::make_shared<NetInstance::Fabric>(10, "planeA"); // port数1
    auto fabricB = std::make_shared<NetInstance::Fabric>(20, "planeB"); // port数3
    auto fabricC = std::make_shared<NetInstance::Fabric>(30, "planeC"); // port数2
    auto fabricD = std::make_shared<NetInstance::Fabric>(40, "planeD"); // port数2
    (void)netInst->AddNode(fabricA);
    (void)netInst->AddNode(fabricB);
    (void)netInst->AddNode(fabricC);
    (void)netInst->AddNode(fabricD);
    // hop值唯一标识链路，便于按属性查找断言
    AddClosLink(*netInst, srcPeer, dstPeer, fabricA, 1, 10); // closA: port=1
    AddClosLink(*netInst, srcPeer, dstPeer, fabricB, 3, 20); // closB: port=3
    AddClosLink(*netInst, srcPeer, dstPeer, fabricC, 2, 30); // closC: port=2
    AddClosLink(*netInst, srcPeer, dstPeer, fabricD, 2, 40); // closD: port=2

    // Peer::AddNetInstance 把 netLayer 注册到 peer 的 levels 和 netInsts_，
    // 使 GetLevels 校验通过、GetNetInstanceByRankId 能取到 netInst
    (void)srcPeer->AddNetInstance(netInst);
    (void)dstPeer->AddNetInstance(netInst);
    (void)rankGraph.AddNetInstance(netInst);
    ASSERT_EQ(rankGraph.InitInnerRanks(), HCCL_SUCCESS);

    IRankGraph iRankGraph(static_cast<void*>(&rankGraph));
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    HcclResult ret = iRankGraph.GetLinks(NET_LAYER, SRC_RANK, DST_RANK, &linkList, &listSize);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 2条p2p + 4条clos = 6条
    EXPECT_EQ(listSize, 6u);
    EXPECT_NE(linkList, nullptr);
    if (ret != HCCL_SUCCESS || linkList == nullptr) {
        return;
    }
    // p2p按hop查找：hop=1先加入，hop=2后加入
    int idxP2pFirst = FindLinkIndexByHop(linkList, listSize, 1);
    int idxP2pSecond = FindLinkIndexByHop(linkList, listSize, 2);
    // clos按hop查找：hop=10/20/30/40
    int idxA = FindLinkIndexByHop(linkList, listSize, 10);
    int idxB = FindLinkIndexByHop(linkList, listSize, 20);
    int idxC = FindLinkIndexByHop(linkList, listSize, 30);
    int idxD = FindLinkIndexByHop(linkList, listSize, 40);
    EXPECT_GE(idxP2pFirst, 0);
    EXPECT_GE(idxP2pSecond, 0);
    EXPECT_GE(idxA, 0);
    EXPECT_GE(idxB, 0);
    EXPECT_GE(idxC, 0);
    EXPECT_GE(idxD, 0);
    // p2p之间保持原相对顺序：hop=1 在 hop=2 前
    EXPECT_LT(idxP2pFirst, idxP2pSecond);
    // 所有p2p索引 < 所有clos索引
    EXPECT_LT(idxP2pSecond, idxB); // p2p最后一条 < clos第一条（clos降序首条是closB）
    // clos降序：closB(3) -> closC(2) -> closD(2) -> closA(1)
    EXPECT_LT(idxB, idxC); // port3 在 port2 前
    EXPECT_LT(idxC, idxD); // port2相同保持原序（closC在closD前）
    EXPECT_LT(idxD, idxA); // port2 在 port1 前
}

// 场景：纯clos拓扑（3条clos，port数=1,3,2），对应跳卡场景
// 预期：按源端点port数降序（3>2>1）
TEST_F(RankGraphGetLinksSortTest, Ut_GetLinks_When_OnlyClos_Expect_SortedByPortNumDesc)
{
    RankGraph rankGraph(MY_RANK);
    auto netInst = std::make_shared<ClosNetInstance>(NET_LAYER, "clos");
    auto srcPeer = CreatePeer(SRC_RANK, 0, 0);
    auto dstPeer = CreatePeer(DST_RANK, 1, 1);
    netInst->AddRankId(SRC_RANK);
    netInst->AddRankId(DST_RANK);
    (void)netInst->AddNode(srcPeer);
    (void)netInst->AddNode(dstPeer);
    (void)rankGraph.AddPeer(srcPeer);
    (void)rankGraph.AddPeer(dstPeer);

    auto fabricA = std::make_shared<NetInstance::Fabric>(10, "planeA");
    auto fabricB = std::make_shared<NetInstance::Fabric>(20, "planeB");
    auto fabricC = std::make_shared<NetInstance::Fabric>(30, "planeC");
    (void)netInst->AddNode(fabricA);
    (void)netInst->AddNode(fabricB);
    (void)netInst->AddNode(fabricC);
    AddClosLink(*netInst, srcPeer, dstPeer, fabricA, 1, 10); // port=1
    AddClosLink(*netInst, srcPeer, dstPeer, fabricB, 3, 20); // port=3
    AddClosLink(*netInst, srcPeer, dstPeer, fabricC, 2, 30); // port=2

    // Peer::AddNetInstance 把 netLayer 注册到 peer 的 levels 和 netInsts_，
    // 使 GetLevels 校验通过、GetNetInstanceByRankId 能取到 netInst
    (void)srcPeer->AddNetInstance(netInst);
    (void)dstPeer->AddNetInstance(netInst);
    (void)rankGraph.AddNetInstance(netInst);
    ASSERT_EQ(rankGraph.InitInnerRanks(), HCCL_SUCCESS);

    IRankGraph iRankGraph(static_cast<void*>(&rankGraph));
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    HcclResult ret = iRankGraph.GetLinks(NET_LAYER, SRC_RANK, DST_RANK, &linkList, &listSize);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 3u);
    if (ret != HCCL_SUCCESS || linkList == nullptr) {
        return;
    }
    // 预期降序：closB(3) -> closC(2) -> closA(1)
    int idxB = FindLinkIndexByHop(linkList, listSize, 20);
    int idxC = FindLinkIndexByHop(linkList, listSize, 30);
    int idxA = FindLinkIndexByHop(linkList, listSize, 10);
    EXPECT_LT(idxB, idxC);
    EXPECT_LT(idxC, idxA);
}

// 场景：2条clos port数相同，验证stable_sort保持原相对顺序
// 预期：原序在前者排序后仍在前
TEST_F(RankGraphGetLinksSortTest, Ut_GetLinks_When_ClosSamePortNum_Expect_StableOrder)
{
    RankGraph rankGraph(MY_RANK);
    auto netInst = std::make_shared<ClosNetInstance>(NET_LAYER, "clos");
    auto srcPeer = CreatePeer(SRC_RANK, 0, 0);
    auto dstPeer = CreatePeer(DST_RANK, 1, 1);
    netInst->AddRankId(SRC_RANK);
    netInst->AddRankId(DST_RANK);
    (void)netInst->AddNode(srcPeer);
    (void)netInst->AddNode(dstPeer);
    (void)rankGraph.AddPeer(srcPeer);
    (void)rankGraph.AddPeer(dstPeer);

    auto fabricFirst = std::make_shared<NetInstance::Fabric>(100, "planeFirst");
    auto fabricSecond = std::make_shared<NetInstance::Fabric>(200, "planeSecond");
    (void)netInst->AddNode(fabricFirst);
    (void)netInst->AddNode(fabricSecond);
    // 两条port数均为2的clos链路，fabricFirst先加入（GetPaths先返回）
    AddClosLink(*netInst, srcPeer, dstPeer, fabricFirst, 2, 100);
    AddClosLink(*netInst, srcPeer, dstPeer, fabricSecond, 2, 200);

    // Peer::AddNetInstance 把 netLayer 注册到 peer 的 levels 和 netInsts_，
    // 使 GetLevels 校验通过、GetNetInstanceByRankId 能取到 netInst
    (void)srcPeer->AddNetInstance(netInst);
    (void)dstPeer->AddNetInstance(netInst);
    (void)rankGraph.AddNetInstance(netInst);
    ASSERT_EQ(rankGraph.InitInnerRanks(), HCCL_SUCCESS);

    IRankGraph iRankGraph(static_cast<void*>(&rankGraph));
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    HcclResult ret = iRankGraph.GetLinks(NET_LAYER, SRC_RANK, DST_RANK, &linkList, &listSize);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 2u);
    if (ret != HCCL_SUCCESS || linkList == nullptr) {
        return;
    }
    // port数相同，stable_sort保持原序：hop=100 在 hop=200 前
    int idxFirst = FindLinkIndexByHop(linkList, listSize, 100);
    int idxSecond = FindLinkIndexByHop(linkList, listSize, 200);
    EXPECT_LT(idxFirst, idxSecond);
}

// 场景：netLayer不在levels中
// 预期：返回HCCL_E_PARA
TEST_F(RankGraphGetLinksSortTest, Ut_GetLinks_When_InvalidNetLayer_Expect_EPara)
{
    RankGraph rankGraph(MY_RANK);
    auto netInst = std::make_shared<ClosNetInstance>(NET_LAYER, "clos");
    auto srcPeer = CreatePeer(SRC_RANK, 0, 0);
    auto dstPeer = CreatePeer(DST_RANK, 1, 1);
    netInst->AddRankId(SRC_RANK);
    netInst->AddRankId(DST_RANK);
    (void)netInst->AddNode(srcPeer);
    (void)netInst->AddNode(dstPeer);
    (void)rankGraph.AddPeer(srcPeer);
    (void)rankGraph.AddPeer(dstPeer);
    // Peer::AddNetInstance 把 netLayer 注册到 peer 的 levels 和 netInsts_，
    // 使 GetLevels 校验通过、GetNetInstanceByRankId 能取到 netInst
    (void)srcPeer->AddNetInstance(netInst);
    (void)dstPeer->AddNetInstance(netInst);
    (void)rankGraph.AddNetInstance(netInst);
    ASSERT_EQ(rankGraph.InitInnerRanks(), HCCL_SUCCESS);

    IRankGraph iRankGraph(static_cast<void*>(&rankGraph));
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    // 使用一个不存在的netLayer
    HcclResult ret = iRankGraph.GetLinks(99, SRC_RANK, DST_RANK, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 场景：InsertClosLinks 中 CHK_PTR_NULL(peer2net)/CHK_PTR_NULL(net2peer) 防御分支覆盖说明
// 预期：无法通过 NetInstance 公共接口构造触发该分支的输入，故不补断言用例
// 说明：
//   1. InsertClosLinks 收到的 path 来自 GetPaths，而 ClosNetInstance::GetPaths（net_instance.cc:313）
//      与 InnerNetInstance::GetPaths（net_instance.cc:257）均在 src→fabric 与 fabric→dst 两条边
//      都非空时才构造 path，不会产生"只有 peer2net 没有 net2peer"的不完整 path。
//   2. 无论用 AddLink 如何构造拓扑，只要 GetPaths 不返回不完整 path，InsertClosLinks 的
//      CHK_PTR_NULL(peer2net)/CHK_PTR_NULL(net2peer) 就不会触发，返回码 HCCL_E_PTR 无测试兜底。
//   3. 该防御分支用于应对未来 GetPaths 实现放宽或异常输入，属上游已保证但本函数仍需自保的场景，
//      强行构造需绕过 GetPaths 直接调用 InsertClosLinks（private static 函数），不属行为契约测试范畴。
TEST_F(RankGraphGetLinksSortTest, Ut_GetLinks_DefensiveNullCheckInInsertClosLinks_NotTriggerableViaPublicAPI)
{
    // 仅添加 srcPeer→fabric 链路、缺少 fabric→dstPeer 链路，构造不完整 CLOS 拓扑
    RankGraph rankGraph(MY_RANK);
    auto netInst = std::make_shared<ClosNetInstance>(NET_LAYER, "clos");
    auto srcPeer = CreatePeer(SRC_RANK, 0, 0);
    auto dstPeer = CreatePeer(DST_RANK, 1, 1);
    netInst->AddRankId(SRC_RANK);
    netInst->AddRankId(DST_RANK);
    (void)netInst->AddNode(srcPeer);
    (void)netInst->AddNode(dstPeer);
    (void)rankGraph.AddPeer(srcPeer);
    (void)rankGraph.AddPeer(dstPeer);

    auto fabric = std::make_shared<NetInstance::Fabric>(10, "planeA");
    (void)netInst->AddNode(fabric);
    // 只添加 srcPeer→fabric 边，不添加 fabric→dstPeer 边
    std::set<std::string> ports = {"10/0"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_TP};
    auto srcIface = std::make_shared<NetInstance::ConnInterface>(
        IpAddress("ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffe"), ports, AddrPosition::DEVICE, LinkType::PEER2NET,
        protocols, TopoType::CLOS, 10);
    srcPeer->AddConnInterface(NET_LAYER, srcIface);
    netInst->AddLink(std::make_shared<NetInstance::Link>(
        srcPeer, fabric, srcIface, nullptr, LinkType::PEER2NET, protocols, LinkDirection::BOTH, 10));

    (void)srcPeer->AddNetInstance(netInst);
    (void)dstPeer->AddNetInstance(netInst);
    (void)rankGraph.AddNetInstance(netInst);
    ASSERT_EQ(rankGraph.InitInnerRanks(), HCCL_SUCCESS);

    IRankGraph iRankGraph(static_cast<void*>(&rankGraph));
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    // GetPaths 因 fabric→dstPeer 边缺失而不构造 path，GetLinks 返回成功但 listSize=0，
    // 不会进入 InsertClosLinks 的 CHK_PTR_NULL 分支（验证防御分支不可通过公共接口触发）
    HcclResult ret = iRankGraph.GetLinks(NET_LAYER, SRC_RANK, DST_RANK, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0u);
}
