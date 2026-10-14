/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <vector>
#include <string>
#include "hccl/hccl_types.h"
#include "hccl/hccl_comm.h"
#include "acl/acl_base_rt.h"
#include "hccl_group.h"
#define private public
#include "op_group_queue_mgr.h"
#include "hccl_comm_pub.h"
#include "coll_comm.h"
#undef private

using namespace hccl;

static const uint64_t TEST_ITEM_COUNT = 100;

// simpleMode的mock通信域，构造时自动创建groupQueueMgr实例
static CollComm g_mockCollComm(nullptr, 0, "test", ManagerCallbacks{}, CollCommInitMode::simpleMode);

// 捕获批量执行函数的回调参数
static uint32_t g_execCallCount = 0;
static uint32_t g_lastItemNum = 0;
static HcclComm g_lastExecComm = nullptr;
static aclrtStream g_lastExecStream = nullptr;
static std::vector<HcclSendRecvItem> g_lastExecItems;

HcclResult MockExecFuncSuccess(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
{
    g_execCallCount++;
    g_lastItemNum = itemNum;
    g_lastExecComm = comm;
    g_lastExecStream = stream;
    g_lastExecItems.assign(sendRecvInfo, sendRecvInfo + itemNum);
    return HCCL_SUCCESS;
}

HcclResult MockExecFuncFail(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
{
    g_execCallCount++;
    g_lastItemNum = itemNum;
    g_lastExecComm = comm;
    g_lastExecStream = stream;
    g_lastExecItems.assign(sendRecvInfo, sendRecvInfo + itemNum);
    return HCCL_E_INTERNAL;
}

// 构造P2P算子描述
static HcclOpDesc MakeOpDesc(HcclCMDType cmdType, uint32_t remoteRank, void* buffer)
{
    HcclOpDesc opDesc{};
    opDesc.opDescType = 1; // 1: P2P算子
    opDesc.p2p.buffer = buffer;
    opDesc.p2p.cmdType = cmdType;
    opDesc.p2p.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    opDesc.p2p.count = TEST_ITEM_COUNT;
    opDesc.p2p.remoteRank = remoteRank;
    return opDesc;
}

class OpGroupQueueMgrTest : public testing::Test {
public:
    void SetUp() override
    {
        OpGroupQueueMgr::SetGroupQueueTaskNums(0);
        OpGroupQueueMgr::GetGroupQueueCommList().clear();
        g_mockCollComm.groupQueueMgr.ClearGroupOpTask();
        hcclGroupDepth = 0;
        g_execCallCount = 0;
        g_lastItemNum = 0;
        g_lastExecComm = nullptr;
        g_lastExecStream = nullptr;
        g_lastExecItems.clear();
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
        OpGroupQueueMgr::SetGroupQueueTaskNums(0);
        OpGroupQueueMgr::GetGroupQueueCommList().clear();
        g_mockCollComm.groupQueueMgr.ClearGroupOpTask();
        hcclGroupDepth = 0;
    }
};

TEST_F(OpGroupQueueMgrTest, Ut_AppendGroupOpTask_When_ValidTask_Expect_Success)
{
    OpGroupQueueMgr mgr;
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x2000);
    HcclOpDesc opDesc = MakeOpDesc(HcclCMDType::HCCL_CMD_SEND, 1, reinterpret_cast<void*>(0x10000));

    HcclResult ret = mgr.AppendGroupOpTask(comm, opDesc, stream, reinterpret_cast<void*>(MockExecFuncSuccess));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(mgr.groupOpItems_.size(), 1u);
    EXPECT_EQ(OpGroupQueueMgr::GetGroupQueueTaskNums(), 1);
    EXPECT_EQ(OpGroupQueueMgr::GetGroupQueueCommList().size(), 1u);
    EXPECT_EQ(OpGroupQueueMgr::GetGroupQueueCommList()[0], comm);
}

TEST_F(OpGroupQueueMgrTest, Ut_AppendGroupOpTask_When_TaskNumsReachLimit_Expect_InternalError)
{
    OpGroupQueueMgr mgr;
    OpGroupQueueMgr::SetGroupQueueTaskNums(MAX_GROUP_OP_TASK_NUM);
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    HcclOpDesc opDesc = MakeOpDesc(HcclCMDType::HCCL_CMD_SEND, 1, reinterpret_cast<void*>(0x10000));

    HcclResult ret = mgr.AppendGroupOpTask(comm, opDesc, nullptr, reinterpret_cast<void*>(MockExecFuncSuccess));
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(mgr.groupOpItems_.size(), 0u);
    EXPECT_EQ(OpGroupQueueMgr::GetGroupQueueTaskNums(), MAX_GROUP_OP_TASK_NUM);
    EXPECT_EQ(OpGroupQueueMgr::GetGroupQueueCommList().size(), 0u);
}

TEST_F(OpGroupQueueMgrTest, Ut_LaunchGroupOpTask_When_EmptyQueue_Expect_Success)
{
    OpGroupQueueMgr mgr;
    HcclResult ret = mgr.LaunchGroupOpTask(reinterpret_cast<HcclComm>(0x1000));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_execCallCount, 0u);
}

TEST_F(OpGroupQueueMgrTest, Ut_LaunchGroupOpTask_When_DifferentFuncs_Expect_ParaErrorAndCleared)
{
    OpGroupQueueMgr mgr;
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x2000);
    HcclOpDesc sendDesc = MakeOpDesc(HcclCMDType::HCCL_CMD_SEND, 1, reinterpret_cast<void*>(0x10000));
    HcclOpDesc recvDesc = MakeOpDesc(HcclCMDType::HCCL_CMD_RECEIVE, 2, reinterpret_cast<void*>(0x20000));

    EXPECT_EQ(
        mgr.AppendGroupOpTask(comm, sendDesc, stream, reinterpret_cast<void*>(MockExecFuncSuccess)), HCCL_SUCCESS);
    EXPECT_EQ(mgr.AppendGroupOpTask(comm, recvDesc, stream, reinterpret_cast<void*>(MockExecFuncFail)), HCCL_SUCCESS);
    EXPECT_EQ(mgr.groupOpItems_.size(), 2u);

    HcclResult ret = mgr.LaunchGroupOpTask(comm);
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(mgr.groupOpItems_.size(), 0u);
    EXPECT_EQ(g_execCallCount, 0u);
}

TEST_F(OpGroupQueueMgrTest, Ut_LaunchGroupOpTask_When_NormalBatch_Expect_Success)
{
    OpGroupQueueMgr mgr;
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    aclrtStream firstStream = reinterpret_cast<aclrtStream>(0x2000);
    aclrtStream secondStream = reinterpret_cast<aclrtStream>(0x3000);
    void* sendBuf = reinterpret_cast<void*>(0x10000);
    void* recvBuf = reinterpret_cast<void*>(0x20000);
    HcclOpDesc sendDesc = MakeOpDesc(HcclCMDType::HCCL_CMD_SEND, 1, sendBuf);
    HcclOpDesc recvDesc = MakeOpDesc(HcclCMDType::HCCL_CMD_RECEIVE, 2, recvBuf);

    EXPECT_EQ(
        mgr.AppendGroupOpTask(comm, sendDesc, firstStream, reinterpret_cast<void*>(MockExecFuncSuccess)), HCCL_SUCCESS);
    EXPECT_EQ(
        mgr.AppendGroupOpTask(comm, recvDesc, secondStream, reinterpret_cast<void*>(MockExecFuncSuccess)),
        HCCL_SUCCESS);

    HcclResult ret = mgr.LaunchGroupOpTask(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_execCallCount, 1u);
    EXPECT_EQ(g_lastItemNum, 2u);
    EXPECT_EQ(g_lastExecComm, comm);
    EXPECT_EQ(g_lastExecStream, firstStream); // 以组内首条任务的stream为准
    ASSERT_EQ(g_lastExecItems.size(), 2u);
    EXPECT_EQ(g_lastExecItems[0].sendRecvType, HCCL_SEND);
    EXPECT_EQ(g_lastExecItems[0].buf, sendBuf);
    EXPECT_EQ(g_lastExecItems[0].count, TEST_ITEM_COUNT);
    EXPECT_EQ(g_lastExecItems[0].dataType, HCCL_DATA_TYPE_FP32);
    EXPECT_EQ(g_lastExecItems[0].remoteRank, 1u);
    EXPECT_EQ(g_lastExecItems[1].sendRecvType, HCCL_RECV);
    EXPECT_EQ(g_lastExecItems[1].buf, recvBuf);
    EXPECT_EQ(g_lastExecItems[1].count, TEST_ITEM_COUNT);
    EXPECT_EQ(g_lastExecItems[1].remoteRank, 2u);
    EXPECT_EQ(mgr.groupOpItems_.size(), 0u); // 执行后队列清空
}

TEST_F(OpGroupQueueMgrTest, Ut_LaunchGroupOpTask_When_ExecFuncFailed_Expect_ReturnErrorAndCleared)
{
    OpGroupQueueMgr mgr;
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x2000);
    HcclOpDesc opDesc = MakeOpDesc(HcclCMDType::HCCL_CMD_SEND, 1, reinterpret_cast<void*>(0x10000));

    EXPECT_EQ(mgr.AppendGroupOpTask(comm, opDesc, stream, reinterpret_cast<void*>(MockExecFuncFail)), HCCL_SUCCESS);

    HcclResult ret = mgr.LaunchGroupOpTask(comm);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(g_execCallCount, 1u);
    EXPECT_EQ(g_lastItemNum, 1u);
    EXPECT_EQ(mgr.groupOpItems_.size(), 0u); // 失败也清空队列
}

TEST_F(OpGroupQueueMgrTest, Ut_HcclGroupEnqueueOp_When_GroupNotEnabled_Expect_InternalError)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    hcclGroupDepth = 0;
    HcclOpDesc opDesc = MakeOpDesc(HcclCMDType::HCCL_CMD_SEND, 1, reinterpret_cast<void*>(0x10000));

    HcclResult ret = HcclGroupEnqueueOp(comm, &opDesc, nullptr, reinterpret_cast<void*>(MockExecFuncSuccess));
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(OpGroupQueueMgrTest, Ut_HcclGroupEnqueueOp_When_InvalidCmdType_Expect_ParaError)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    hcclGroupDepth = 1;
    HcclOpDesc opDesc = MakeOpDesc(HcclCMDType::HCCL_CMD_ALLGATHER, 1, reinterpret_cast<void*>(0x10000));

    HcclResult ret = HcclGroupEnqueueOp(comm, &opDesc, nullptr, reinterpret_cast<void*>(MockExecFuncSuccess));
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(OpGroupQueueMgrTest, Ut_HcclGroupEnqueueOp_When_CollCommNull_Expect_PtrError)
{
    hccl::hcclComm mockHcclComm;
    HcclComm comm = static_cast<HcclComm>(&mockHcclComm);
    hcclGroupDepth = 1;
    HcclOpDesc opDesc = MakeOpDesc(HcclCMDType::HCCL_CMD_SEND, 1, reinterpret_cast<void*>(0x10000));

    MOCKER_CPP(&hcclComm::GetCollComm).stubs().will(returnValue(static_cast<hccl::CollComm*>(nullptr)));

    HcclResult ret = HcclGroupEnqueueOp(comm, &opDesc, nullptr, reinterpret_cast<void*>(MockExecFuncSuccess));
    EXPECT_EQ(ret, HCCL_E_PTR);
    GlobalMockObject::verify();
}

TEST_F(OpGroupQueueMgrTest, Ut_HcclGroupEnqueueOp_When_ValidParams_Expect_Success)
{
    hccl::hcclComm mockHcclComm;
    HcclComm comm = static_cast<HcclComm>(&mockHcclComm);
    hcclGroupDepth = 1;
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x2000);
    HcclOpDesc opDesc = MakeOpDesc(HcclCMDType::HCCL_CMD_SEND, 1, reinterpret_cast<void*>(0x10000));

    MOCKER_CPP(&hcclComm::GetCollComm).stubs().will(returnValue(&g_mockCollComm));

    HcclResult ret = HcclGroupEnqueueOp(comm, &opDesc, stream, reinterpret_cast<void*>(MockExecFuncSuccess));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_mockCollComm.groupQueueMgr.groupOpItems_.size(), 1u);
    EXPECT_EQ(OpGroupQueueMgr::GetGroupQueueTaskNums(), 1);
    EXPECT_EQ(OpGroupQueueMgr::GetGroupQueueCommList().size(), 1u);
    GlobalMockObject::verify();
}

TEST_F(OpGroupQueueMgrTest, Ut_HcclGroupEnqueueOp_When_NullComm_Expect_PtrError)
{
    HcclOpDesc opDesc = MakeOpDesc(HcclCMDType::HCCL_CMD_SEND, 1, reinterpret_cast<void*>(0x10000));
    HcclResult ret = HcclGroupEnqueueOp(nullptr, &opDesc, nullptr, reinterpret_cast<void*>(MockExecFuncSuccess));
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(OpGroupQueueMgrTest, Ut_HcclGroupEnqueueOp_When_NullOpDesc_Expect_PtrError)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    HcclResult ret = HcclGroupEnqueueOp(comm, nullptr, nullptr, reinterpret_cast<void*>(MockExecFuncSuccess));
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(OpGroupQueueMgrTest, Ut_HcclGroupEnqueueOp_When_NullFunc_Expect_PtrError)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    HcclOpDesc opDesc = MakeOpDesc(HcclCMDType::HCCL_CMD_SEND, 1, reinterpret_cast<void*>(0x10000));
    HcclResult ret = HcclGroupEnqueueOp(comm, &opDesc, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
