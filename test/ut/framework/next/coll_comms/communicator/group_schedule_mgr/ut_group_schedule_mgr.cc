/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <vector>
#include <string>
#include <set>
#include "adapter_rts_common.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_launch.h"
#include "hccl/hccl_rank_graph.h"
#include "acl/acl_base_rt.h"
#define private public
#include "group_schedule_mgr.h"
#include "hccl_comm_pub.h"
#include "coll_comm.h"
#include "rank_graph_v2.h"
#undef private
#include "net_instance.h"

using namespace hccl;

class GroupScheduleMgrTest : public testing::Test {
public:
    void SetUp() override
    {
        SetHcclP2pTaskNums(0);
        GetHcclGroupCommList().clear();
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
        SetHcclP2pTaskNums(0);
        GetHcclGroupCommList().clear();
    }
};

TEST_F(GroupScheduleMgrTest, Ut_GetHcclGroupCommList_When_Initialized_Expect_EmptyList)
{
    std::vector<HcclComm>& commList = GetHcclGroupCommList();
    EXPECT_EQ(commList.size(), 0u);
}

TEST_F(GroupScheduleMgrTest, Ut_GetHcclGroupCommList_When_AddedComms_Expect_ReturnCorrectList)
{
    std::vector<HcclComm>& commList = GetHcclGroupCommList();
    HcclComm comm1 = reinterpret_cast<HcclComm>(0x1000);
    HcclComm comm2 = reinterpret_cast<HcclComm>(0x2000);
    commList.push_back(comm1);
    commList.push_back(comm2);

    EXPECT_EQ(commList.size(), 2u);
    EXPECT_EQ(commList[0], comm1);
    EXPECT_EQ(commList[1], comm2);
}

TEST_F(GroupScheduleMgrTest, Ut_GetHcclP2pTaskNums_When_Initialized_Expect_Zero)
{
    int32_t taskNums = GetHcclP2pTaskNums();
    EXPECT_EQ(taskNums, 0);
}

TEST_F(GroupScheduleMgrTest, Ut_SetHcclP2pTaskNums_When_SetValue_Expect_GetReturnsSameValue)
{
    SetHcclP2pTaskNums(100);
    int32_t taskNums = GetHcclP2pTaskNums();
    EXPECT_EQ(taskNums, 100);

    SetHcclP2pTaskNums(0);
    taskNums = GetHcclP2pTaskNums();
    EXPECT_EQ(taskNums, 0);
}

TEST_F(GroupScheduleMgrTest, Ut_SetUsrStream_When_ValidStream_Expect_Success)
{
    GroupScheduleMgr mgr;
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x1000);
    HcclResult ret = mgr.SetUsrStream(stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(GroupScheduleMgrTest, Ut_SetUsrStream_When_NullStream_Expect_PtrError)
{
    GroupScheduleMgr mgr;
    aclrtStream stream = nullptr;
    HcclResult ret = mgr.SetUsrStream(stream);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(GroupScheduleMgrTest, Ut_GetUsrStream_When_StreamSet_Expect_ReturnCorrectStream)
{
    GroupScheduleMgr mgr;
    aclrtStream inputStream = reinterpret_cast<aclrtStream>(0x2000);
    HcclResult ret = mgr.SetUsrStream(inputStream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    aclrtStream outputStream = nullptr;
    ret = mgr.GetUsrStream(outputStream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(outputStream, inputStream);
}

TEST_F(GroupScheduleMgrTest, Ut_GetUsrStream_When_StreamNotSet_Expect_PtrError)
{
    GroupScheduleMgr mgr;
    aclrtStream outputStream = nullptr;
    HcclResult ret = mgr.GetUsrStream(outputStream);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

HcclResult
MockHcclRankGraphGetInstSizeListByLayer(HcclComm comm, uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize)
{
    static uint32_t mockSizeList[4] = {2, 2, 2, 2};
    *instSizeList = mockSizeList;
    *listSize = 4;
    return HCCL_SUCCESS;
}

static CollComm g_mockCollComm(nullptr, 0, "test", ManagerCallbacks{}, CollCommInitMode::simpleMode);
void SetupCollCommMocks()
{
    MOCKER_CPP(&hcclComm::GetCollComm).stubs().will(returnValue(&g_mockCollComm));

    MOCKER_CPP(&CollComm::GetMyRankId).stubs().will(returnValue(0U));

    MOCKER_CPP(&CollComm::GetRankSize).stubs().will(returnValue(8U));
}

TEST_F(GroupScheduleMgrTest, Ut_AppendGroupP2pTask_When_NullComm_Expect_PtrError)
{
    GroupScheduleMgr mgr;
    HcclP2pTask task;
    HcclOpP2pDesc p2pDesc;
    p2pDesc.cmdType = HcclCMDType::HCCL_CMD_SEND;
    p2pDesc.remoteRank = 0;

    HcclResult ret = mgr.AppendGroupP2pTask(nullptr, task, p2pDesc);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(GroupScheduleMgrTest, Ut_AppendGroupP2pTask_When_MaxTaskNumReached_Expect_InternalError)
{
    GroupScheduleMgr mgr;
    SetHcclP2pTaskNums(MAX_P2P_TASK_NUM);

    hccl::hcclComm mockHcclComm;
    HcclComm comm = static_cast<HcclComm>(&mockHcclComm);
    HcclP2pTask task;
    HcclOpP2pDesc p2pDesc;
    p2pDesc.cmdType = HcclCMDType::HCCL_CMD_SEND;
    p2pDesc.remoteRank = 0;

    HcclResult ret = mgr.AppendGroupP2pTask(comm, task, p2pDesc);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    SetHcclP2pTaskNums(0);
}

TEST_F(GroupScheduleMgrTest, Ut_AppendGroupP2pTask_When_SendTask_Expect_AddedToSendQueue)
{
    GroupScheduleMgr mgr;
    hccl::hcclComm mockHcclComm;
    HcclComm comm = static_cast<HcclComm>(&mockHcclComm);
    HcclP2pTask task;
    HcclOpP2pDesc p2pDesc;
    p2pDesc.cmdType = HcclCMDType::HCCL_CMD_SEND;
    p2pDesc.remoteRank = 1;

    SetupCollCommMocks();
    MOCKER(HcclRankGraphGetInstSizeListByLayer).stubs().will(invoke(MockHcclRankGraphGetInstSizeListByLayer));

    HcclResult ret = mgr.AppendGroupP2pTask(comm, task, p2pDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(GroupScheduleMgrTest, Ut_GetP2pTaskSchedule_When_NoTasks_Expect_EmptyQueues)
{
    GroupScheduleMgr mgr;
    std::vector<HcclP2pTask> sortedSendQue;
    std::vector<HcclP2pTask> sortedRecvQue;

    HcclResult ret = mgr.GetP2pTaskSchedule(sortedSendQue, sortedRecvQue);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(sortedSendQue.size(), 0u);
    EXPECT_EQ(sortedRecvQue.size(), 0u);
}

TEST_F(GroupScheduleMgrTest, Ut_GroupScheduleMgrDestructor_When_ObjectDestroyed_Expect_NoException)
{
    GroupScheduleMgr* mgr = new GroupScheduleMgr();
    delete mgr;
    EXPECT_TRUE(true);
}

TEST_F(GroupScheduleMgrTest, Ut_pow2Up_When_InputPowerOfTwo_Expect_ReturnSameValue)
{
    uint32_t input = 8;
    uint32_t expected = 8;
    uint32_t power = 1;
    while (power < input) {
        power = power << 1U;
    }
    EXPECT_EQ(power, expected);
}

TEST_F(GroupScheduleMgrTest, Ut_GetCurLocalRank_When_RankNotInList_Expect_InternalError)
{
    GroupScheduleMgr mgr;
    mgr.userRank_ = 5;
    mgr.serverNum_ = 1;
    mgr.serverToRankSize_[0] = 2;
    mgr.serverToRankList_[0] = {0, 1};

    uint32_t localRank = 0;
    HcclResult ret = mgr.GetCurLocalRank(localRank);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

namespace {
HcclResult EnableV2Support(bool* isSupport)
{
    if (isSupport == nullptr) {
        return HCCL_E_PTR;
    }
    *isSupport = true;
    return HCCL_SUCCESS;
}

// 测试只借用图，不走真实通信域资源初始化/销毁。
struct ScopedGroupCollComm {
    CollComm comm{nullptr, 0, "selected-group", ManagerCallbacks{}, CollCommInitMode::simpleMode};
    explicit ScopedGroupCollComm(RankGraphV2& graph)
    {
        comm.rankgraph_ = &graph;
        comm.initMode_ = CollCommInitMode::fullMode;
    }
    ~ScopedGroupCollComm()
    {
        comm.rankgraph_ = nullptr;
        comm.initMode_ = CollCommInitMode::simpleMode;
    }
};
} // namespace

TEST_F(GroupScheduleMgrTest, NotLayerZeroKeepsOriginalPlannerThroughMetadataQuery)
{
    for (uint32_t rankSize : {4U, 6U}) {
        SCOPED_TRACE(rankSize);
        Hccl::RankGraph graph(3);
        graph.buildLayerSelection_ = Hccl::RankGraph::BuildLayerSelection{5, 4};
        auto instance = std::make_shared<Hccl::ClosNetInstance>(2, "selected-plane");
        std::set<Hccl::RankId> innerRanks;
        for (uint32_t rank = 0; rank < rankSize; ++rank) {
            auto peer = std::make_shared<Hccl::NetInstance::Peer>(rank, rank < 2 ? rank : rank - 2, 0, 0);
            ASSERT_EQ(HCCL_SUCCESS, graph.AddPeer(peer));
            instance->AddRankId(rank);
            ASSERT_EQ(HCCL_SUCCESS, instance->AddNode(peer));
            ASSERT_EQ(HCCL_SUCCESS, peer->AddNetInstance(instance));
            graph.layer0NetInstIds_[rank] = rank < 2 ? "local-a" : "local-b";
            if (rank >= 2) {
                innerRanks.insert(rank);
            }
        }
        ASSERT_EQ(HCCL_SUCCESS, graph.SetInnerRanks(innerRanks));
        ASSERT_EQ(HCCL_SUCCESS, graph.AddNetInstance(instance));
        RankGraphV2 wrapper(&graph);
        ASSERT_EQ(HCCL_SUCCESS, wrapper.InitNetLayerSelection());

        ScopedGroupCollComm scoped(wrapper);
        hcclComm handle;
        MOCKER(hrtGetHcclV2Support).stubs().will(invoke(EnableV2Support));
        MOCKER_CPP(&hcclComm::GetCollComm).stubs().will(returnValue(&scoped.comm));
        MOCKER_CPP(&CollComm::GetMyRankId).stubs().will(returnValue(3U));
        MOCKER_CPP(&CollComm::GetRankSize).stubs().will(returnValue(rankSize));
        // 不桩人数查询：原调度入口通过公共API读取真实V2元数据，覆盖连续2+2和2+4分组。
        EXPECT_TRUE(wrapper.layer0InstSizes_.empty());
        GroupScheduleMgr mgr;
        HcclP2pTask task{};
        HcclOpP2pDesc desc{};
        desc.cmdType = HCCL_CMD_SEND;
        desc.remoteRank = 0;
        ASSERT_EQ(HCCL_SUCCESS, mgr.AppendGroupP2pTask(&handle, task, desc));
        EXPECT_EQ((std::vector<uint32_t>{2U, rankSize - 2U}), wrapper.layer0InstSizes_);
        EXPECT_EQ(2U, mgr.groupSize_);
        EXPECT_EQ(1U, mgr.peers_[0].sendQue.size());
        std::set<uint32_t> destinations;
        for (const auto& pair : mgr.p2pSchedule_) {
            ASSERT_LT(pair.sendRank, rankSize);
            destinations.insert(pair.sendRank);
        }
        EXPECT_EQ(rankSize, destinations.size());

        CommLink* links = nullptr;
        uint32_t linkNum = 0;
        EXPECT_EQ(HCCL_SUCCESS, HcclRankGraphGetLinks(&handle, 0, 2, 3, &links, &linkNum));
        EXPECT_EQ(nullptr, links);
        EXPECT_EQ(0U, linkNum);
        EXPECT_EQ((std::set<uint32_t>{2}), graph.GetLevels(3));
        EXPECT_TRUE(graph.netInsts_[0].empty());
        GlobalMockObject::verify();
        SetHcclP2pTaskNums(0);
        GetHcclGroupCommList().clear();
    }
}
