/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include <memory>
#include <functional>
#define private public
#include "hcclCommDfx.h"
#include "dpu_comm_dfx.h"
#include "hcclCommProfiling.h"
#include "task_param.h"
#include "mirror_task_manager.h"
#include "hcclCommDfxLite.h"
using namespace hccl;

class HcclCommDfxTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclCommDfxTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "HcclCommDfxTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in HcclCommDfxTest SetUp" << std::endl;
        dfx_ = std::make_unique<HcclCommDfx>();
        EXPECT_EQ(dfx_->Init(0, "test_comm", 0), HCCL_SUCCESS);

        dfxLite_ = std::make_unique<HcclCommDfxLite>();
        EXPECT_EQ(dfxLite_->Init(0, "test_comm", 0, 0), HCCL_SUCCESS);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HcclCommDfxTest TearDown" << std::endl;
    }

    std::unique_ptr<HcclCommDfx> dfx_;
    std::unique_ptr<HcclCommDfxLite> dfxLite_;
};

// 测试 AddDpuTaskInfoCallback - 正常情况
TEST_F(HcclCommDfxTest, Ut_AddDpuTaskInfoCallback_When_Normal_Expect_ReturnSuccess)
{
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_DPU_KERNEL;
    u64 handle = 0xFFFFFFFFFFFFFFFF;
    u32 remoteRankId = 1;

    // 先建立 handle 到 remoteRankId 的映射
    HcclCommDfx::AddChannelRemoteRankId("test_comm", handle, remoteRankId);

    // 设置 dpuStreamId_ 和 aicpuTaskId_
    dfx_->GetDpuCommDfx()->SetDpuStreamId(100);
    dfx_->GetDpuCommDfx()->SetAicpuTaskIdAndStreamId(200, 300);

    HcclResult ret = dfx_->AddDpuTaskInfoCallback(taskParam, handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试 AddDpuTaskInfoCallback - 空 taskParam
TEST_F(HcclCommDfxTest, Ut_AddDpuTaskInfoCallback_When_EmptyTaskParam_Expect_ReturnSuccess)
{
    Hccl::TaskParam taskParam{};
    u64 handle = INVALID_U64;

    HcclResult ret = dfx_->AddDpuTaskInfoCallback(taskParam, handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试 GetTaskId - 首次调用返回 0
TEST_F(HcclCommDfxTest, Ut_GetTaskId_When_FirstCall_Expect_ReturnZero)
{
    u32 streamId = 123;
    u32 taskId = DpuCommDfx::GetTaskId(streamId);
    EXPECT_EQ(taskId, 1u);
}

// 测试 GetTaskId - 多次调用自增
TEST_F(HcclCommDfxTest, Ut_GetTaskId_When_MultipleCalls_Expect_Increment)
{
    u32 streamId = 456;

    u32 taskId1 = DpuCommDfx::GetTaskId(streamId);
    EXPECT_EQ(taskId1, 1u);

    u32 taskId2 = DpuCommDfx::GetTaskId(streamId);
    EXPECT_EQ(taskId2, 2u);

    u32 taskId3 = DpuCommDfx::GetTaskId(streamId);
    EXPECT_EQ(taskId3, 3u);
}

// 测试 GetTaskId - 超过 65535 回环到 0
TEST_F(HcclCommDfxTest, Ut_GetTaskId_When_ExceedsLimit_Expect_ReturnToZero)
{
    u32 streamId = 789;

    // 先设置到 65535
    for (int i = 0; i < 65536; i++) {
        DpuCommDfx::GetTaskId(streamId);
    }

    u32 taskId = DpuCommDfx::GetTaskId(streamId);
    EXPECT_EQ(taskId, 1u);
}

// 测试 SetDpuStreamId - 正常设置
TEST_F(HcclCommDfxTest, Ut_SetDpuStreamId_When_Normal_Expect_SetSuccess)
{
    u32 expectedStreamId = 999;
    dfx_->GetDpuCommDfx()->SetDpuStreamId(expectedStreamId);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->dpuStreamId_, expectedStreamId);
}

// 测试 SetDpuStreamId - 设置为 0
TEST_F(HcclCommDfxTest, Ut_SetDpuStreamId_When_Zero_Expect_SetSuccess)
{
    dfx_->GetDpuCommDfx()->SetDpuStreamId(0);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->dpuStreamId_, 0u);
}

// 测试 GetDpuCallback - 获取回调
TEST_F(HcclCommDfxTest, Ut_GetDpuCallback_When_Normal_Expect_ReturnValidCallback)
{
    auto callback = dfx_->GetDpuCommDfx()->GetDpuCallback();
    EXPECT_TRUE(callback != nullptr);
}

// 测试 GetDpuCallback - 回调可调用
TEST_F(HcclCommDfxTest, Ut_GetDpuCallback_When_CallCallback_Expect_ReturnSuccess)
{
    auto callback = dfx_->GetDpuCommDfx()->GetDpuCallback();
    Hccl::TaskParam taskParam{};
    u64 handle = 0xFFFFFFFFFFFFFFFF;
    u32 remoteRankId = 2;

    // 先建立 handle 到 remoteRankId 的映射
    HcclCommDfx::AddChannelRemoteRankId("test_comm", handle, remoteRankId);

    HcclResult ret = callback(taskParam, handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试 SetAicpuTaskIdAndStreamId - 正常设置
TEST_F(HcclCommDfxTest, Ut_SetAicpuTaskIdAndStreamId_When_Normal_Expect_SetSuccess)
{
    u32 taskId = 555;
    u32 streamId = 666;
    dfx_->GetDpuCommDfx()->SetAicpuTaskIdAndStreamId(taskId, streamId);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->aicpuTaskId_, taskId);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->aicpuStreamId_, streamId);
}

// 测试 SetAicpuTaskIdAndStreamId - 设置 INVALID_UINT
TEST_F(HcclCommDfxTest, Ut_SetAicpuTaskIdAndStreamId_When_InvalidValue_Expect_SetSuccess)
{
    dfx_->GetDpuCommDfx()->SetAicpuTaskIdAndStreamId(INVALID_UINT, INVALID_UINT);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->aicpuTaskId_, INVALID_UINT);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->aicpuStreamId_, INVALID_UINT);
}

// 测试不同 streamId 的 taskId 独立
TEST_F(HcclCommDfxTest, Ut_GetTaskId_When_DifferentStreamId_Expect_Independent)
{
    u32 streamId1 = 111;
    u32 streamId2 = 222;

    u32 taskId1 = DpuCommDfx::GetTaskId(streamId1);
    EXPECT_EQ(taskId1, 1u);

    u32 taskId2 = DpuCommDfx::GetTaskId(streamId2);
    EXPECT_EQ(taskId2, 1u);

    taskId1 = DpuCommDfx::GetTaskId(streamId1);
    EXPECT_EQ(taskId1, 2u);

    taskId2 = DpuCommDfx::GetTaskId(streamId2);
    EXPECT_EQ(taskId2, 2u);
}

TEST_F(HcclCommDfxTest, Ut_Add_Get_ChannelRemoteRankId)
{
    u64 channelHandle = 0x9527;
    u32 remoteRankId = 3;
    dfxLite_->AddChannelRemoteRankId(channelHandle, remoteRankId);

    EXPECT_EQ(dfxLite_->GetChannelRemoteRankId(channelHandle), remoteRankId);

    EXPECT_EQ(dfxLite_->GetChannelRemoteRankId(0x123), INVALID_UINT);
}

// 测试 GetOpModeFlags - OFFLOAD 模式返回 false
TEST_F(HcclCommDfxTest, Ut_GetOpModeFlags_When_OpModeIsOffload_Expect_ReturnFalse)
{
    dfx_->mirrorTaskManager_->opMode_ = Hccl::OpMode::OFFLOAD;

    bool isOpBase = true;
    bool isCached = false;
    HcclResult ret = dfx_->GetOpModeFlags(isOpBase, isCached);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(isOpBase);
    EXPECT_TRUE(isCached);
}

TEST_F(HcclCommDfxTest, Ut_GetOpModeFlags_When_OpModeIsAclgraph_Expect_BothTrue)
{
    dfx_->mirrorTaskManager_->opMode_ = Hccl::OpMode::ACLGRAPH;

    bool isOpBase = false;
    bool isCached = false;
    HcclResult ret = dfx_->GetOpModeFlags(isOpBase, isCached);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(isOpBase);
    EXPECT_TRUE(isCached);
}

// 测试 GetOpModeFlags - NEGOTIATIONOP 模式按非GE的opbase方式上报（isOpBase为true、isCached为false）
TEST_F(HcclCommDfxTest, Ut_GetOpModeFlags_When_OpModeIsNegotiationop_Expect_OpBaseTrueAndCachedFalse)
{
    dfx_->mirrorTaskManager_->opMode_ = Hccl::OpMode::NEGOTIATIONOP;

    bool isOpBase = false;
    bool isCached = true;
    HcclResult ret = dfx_->GetOpModeFlags(isOpBase, isCached);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(isOpBase);
    EXPECT_FALSE(isCached);
}

// 测试 GetChannelRemoteRankId（非Lite版）- 正常查找命中 shared_lock 读路径
TEST_F(HcclCommDfxTest, Ut_GetChannelRemoteRankId_When_Exist_Expect_ReturnSuccess)
{
    std::string commTag = "test_comm";
    u64 channelHandle = 0x8888;
    u32 remoteRankId = 7;

    // 先通过 AddChannelRemoteRankId 写入（覆盖 unique_lock 写路径）
    HcclCommDfx::AddChannelRemoteRankId(commTag, channelHandle, remoteRankId);

    // 查找已存在的 commTag+handle（覆盖 shared_lock 读路径）
    u32 result = INVALID_UINT;
    HcclResult ret = HcclCommDfx::GetChannelRemoteRankId(commTag, channelHandle, result);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(result, remoteRankId);

    // 清理静态 map，避免影响其他用例
    HcclCommDfx::channelRemoteRankId_.clear();
}

// 测试 GetChannelRemoteRankId - commTag 不存在
TEST_F(HcclCommDfxTest, Ut_GetChannelRemoteRankId_When_CommTagNotFound_Expect_ReturnParaError)
{
    u32 result = INVALID_UINT;
    HcclResult ret = HcclCommDfx::GetChannelRemoteRankId("non_exist_comm", 0x1234, result);
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(result, INVALID_UINT);
}

// 测试 GetChannelRemoteRankId - commTag 存在但 handle 不存在
TEST_F(HcclCommDfxTest, Ut_GetChannelRemoteRankId_When_HandleNotFound_Expect_ReturnParaError)
{
    std::string commTag = "test_comm_handle";
    HcclCommDfx::AddChannelRemoteRankId(commTag, 0x1111, 1);

    u32 result = INVALID_UINT;
    HcclResult ret = HcclCommDfx::GetChannelRemoteRankId(commTag, 0x2222, result);
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(result, INVALID_UINT);

    HcclCommDfx::channelRemoteRankId_.clear();
}

// 测试 AddTaskInfoCallback - CCU 批量查找路径，覆盖 shared_lock 读路径(L91)
TEST_F(HcclCommDfxTest, Ut_AddTaskInfoCallback_When_CcuTaskAndHandleExist_Expect_Success)
{
    std::string commTag = "test_comm";
    u64 channelHandle = 0x7777;
    u32 remoteRankId = 5;

    // 写入 channelRemoteRankId_ 映射
    HcclCommDfx::AddChannelRemoteRankId(commTag, channelHandle, remoteRankId);

    // 构造 CCU 类型 TaskParam
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_CCU;
    auto ccuDetailInfo = std::make_shared<std::vector<Hccl::CcuProfilingInfo>>();
    Hccl::CcuProfilingInfo profInfo{};
    profInfo.channelId[0] = 0; // 非 INVALID_VALUE_CHANNELID，进入查找
    profInfo.channelHandle[0] = channelHandle;
    ccuDetailInfo->push_back(profInfo);
    taskParam.ccuDetailInfo = ccuDetailInfo;

    // mock MirrorTaskManager
    auto opInfo = std::make_shared<Hccl::DfxOpInfo>();
    MOCKER_CPP(
        &Hccl::MirrorTaskManager::GetCurrDfxOpInfo,
        std::shared_ptr<Hccl::DfxOpInfo>(Hccl::MirrorTaskManager::*)() const)
        .stubs()
        .will(returnValue(opInfo));
    MOCKER_CPP(
        &Hccl::MirrorTaskManager::AddTaskInfo,
        HcclResult(Hccl::MirrorTaskManager::*)(
            u32, u32, u32, const Hccl::TaskParam&, std::shared_ptr<Hccl::DfxOpInfo>, bool, u32))
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = dfx_->AddTaskInfoCallback(1, 1, taskParam, channelHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
    HcclCommDfx::channelRemoteRankId_.clear();
}

// 测试 AddTaskInfoCallback - CCU 路径但 commTag 不存在，覆盖 shared_lock 读路径的提前返回
TEST_F(HcclCommDfxTest, Ut_AddTaskInfoCallback_When_CcuTaskAndCommTagNotFound_Expect_ParaError)
{
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_CCU;
    auto ccuDetailInfo = std::make_shared<std::vector<Hccl::CcuProfilingInfo>>();
    Hccl::CcuProfilingInfo profInfo{};
    profInfo.channelId[0] = 0;
    profInfo.channelHandle[0] = 0x9999;
    ccuDetailInfo->push_back(profInfo);
    taskParam.ccuDetailInfo = ccuDetailInfo;

    // 不添加任何 channelRemoteRankId_ 映射，commTag 查找失败
    HcclResult ret = dfx_->AddTaskInfoCallback(1, 1, taskParam, INVALID_U64);
    EXPECT_EQ(ret, HCCL_E_PARA);

    HcclCommDfx::channelRemoteRankId_.clear();
}

TEST_F(HcclCommDfxTest, Ut_ReportAllTasks_When_EmptyThreads_Expect_NoThrow)
{
    std::vector<hccl::Thread*> threads;
    EXPECT_NO_THROW(dfxLite_->ReportAllTasks(threads));
}

TEST_F(HcclCommDfxTest, Ut_GetLatestDfxOpInfo_When_QueueEmpty_Expect_ReturnNullptr)
{
    const void* result = dfxLite_->GetLatestDfxOpInfo();
    EXPECT_EQ(result, nullptr);
}

TEST_F(HcclCommDfxTest, Ut_GetLatestDfxOpInfo_When_SetOpInfo_Expect_ReturnLatest)
{
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.count = 100;
    opInfo.opType = 0;
    opInfo.dataType = 0;
    EXPECT_EQ(dfxLite_->SetCurrDfxOpInfo(&opInfo), HCCL_SUCCESS);
    const void* result = dfxLite_->GetLatestDfxOpInfo();
    EXPECT_NE(result, nullptr);
    const Hccl::DfxDfxOpInfo* retrieved = static_cast<const Hccl::DfxDfxOpInfo*>(result);
    EXPECT_EQ(retrieved->count, opInfo.count);
}

// ==================== AddDpuTaskInfoCallback 新增分支测试 ====================

// 决策点#1+#2+#3: W+Notify endTime==0（失败路径），handle 已预初始化
TEST_F(HcclCommDfxTest, Ut_AddDpuTaskInfoCallback_When_WNotifyFailedAndHandleExists_Expect_DirectEnqueue)
{
    u64 handle = 0x1000;
    dfx_->GetDpuCommDfx()->InitPendingWriteInfo(handle);
    dfx_->GetDpuCommDfx()->SetDpuStreamId(100);
    dfx_->GetDpuCommDfx()->SetAicpuTaskIdAndStreamId(200, 300);
    HcclCommDfx::AddChannelRemoteRankId("test_comm", handle, 1);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY;
    taskParam.isFailed = true;
    taskParam.endTime = 0;
    taskParam.taskPara.DMA.size = 1024;
    taskParam.taskPara.DMA.notifyID = 42;

    HcclResult ret = dfx_->AddDpuTaskInfoCallback(taskParam, handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->pendingWriteInfos_[handle].valid, false);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->pendingWriteInfos_[handle].count, 0u);

    HcclCommDfx::channelRemoteRankId_.clear();
}

// 决策点#1+#2+#4: W+Notify 首次缓存（endTime!=0, count==0）
TEST_F(HcclCommDfxTest, Ut_AddDpuTaskInfoCallback_When_WNotifyFirstCache_Expect_Cached)
{
    u64 handle = 0x2000;
    dfx_->GetDpuCommDfx()->InitPendingWriteInfo(handle);
    dfx_->GetDpuCommDfx()->SetDpuStreamId(100);
    dfx_->GetDpuCommDfx()->SetAicpuTaskIdAndStreamId(200, 300);
    HcclCommDfx::AddChannelRemoteRankId("test_comm", handle, 1);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY;
    taskParam.beginTime = 1000;
    taskParam.endTime = 2000;
    taskParam.taskPara.DMA.size = 512;
    taskParam.taskPara.DMA.notifyID = 10;

    HcclResult ret = dfx_->AddDpuTaskInfoCallback(taskParam, handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->pendingWriteInfos_[handle].valid, true);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->pendingWriteInfos_[handle].count, 1u);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->pendingWriteInfos_[handle].totalSize, 512u);

    HcclCommDfx::channelRemoteRankId_.clear();
}

// 决策点#1+#2+#4 else: W+Notify 二次缓存累加（count>0）
TEST_F(HcclCommDfxTest, Ut_AddDpuTaskInfoCallback_When_WNotifySecondCache_Expect_Accumulated)
{
    u64 handle = 0x3000;
    dfx_->GetDpuCommDfx()->InitPendingWriteInfo(handle);
    dfx_->GetDpuCommDfx()->SetDpuStreamId(100);
    dfx_->GetDpuCommDfx()->SetAicpuTaskIdAndStreamId(200, 300);
    HcclCommDfx::AddChannelRemoteRankId("test_comm", handle, 1);

    Hccl::TaskParam taskParam1{};
    taskParam1.taskType = Hccl::TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY;
    taskParam1.beginTime = 1000;
    taskParam1.endTime = 2000;
    taskParam1.taskPara.DMA.size = 512;
    taskParam1.taskPara.DMA.notifyID = 10;

    EXPECT_EQ(dfx_->AddDpuTaskInfoCallback(taskParam1, handle), HCCL_SUCCESS);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->pendingWriteInfos_[handle].count, 1u);

    Hccl::TaskParam taskParam2{};
    taskParam2.taskType = Hccl::TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY;
    taskParam2.beginTime = 900;
    taskParam2.endTime = 2100;
    taskParam2.taskPara.DMA.size = 256;
    taskParam2.taskPara.DMA.notifyID = 10;

    EXPECT_EQ(dfx_->AddDpuTaskInfoCallback(taskParam2, handle), HCCL_SUCCESS);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->pendingWriteInfos_[handle].count, 2u);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->pendingWriteInfos_[handle].totalSize, 768u);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->pendingWriteInfos_[handle].taskParam->beginTime, 900u);

    HcclCommDfx::channelRemoteRankId_.clear();
}

// 决策点#1+#2: W+Notify handle 未预初始化，走 default
TEST_F(HcclCommDfxTest, Ut_AddDpuTaskInfoCallback_When_WNotifyHandleNotInit_Expect_FallThroughDefault)
{
    u64 handle = 0x4000;
    dfx_->GetDpuCommDfx()->SetDpuStreamId(100);
    dfx_->GetDpuCommDfx()->SetAicpuTaskIdAndStreamId(200, 300);
    HcclCommDfx::AddChannelRemoteRankId("test_comm", handle, 1);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY;
    taskParam.endTime = 2000;
    taskParam.taskPara.DMA.size = 512;
    taskParam.taskPara.DMA.notifyID = 10;

    HcclResult ret = dfx_->AddDpuTaskInfoCallback(taskParam, handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommDfx::channelRemoteRankId_.clear();
}

// 决策点#5+#6: ChannelFence 有缓存，触发 flush
TEST_F(HcclCommDfxTest, Ut_AddDpuTaskInfoCallback_When_FenceWithPending_Expect_FlushEnqueue)
{
    u64 handle = 0x5000;
    dfx_->GetDpuCommDfx()->InitPendingWriteInfo(handle);
    dfx_->GetDpuCommDfx()->SetDpuStreamId(100);
    dfx_->GetDpuCommDfx()->SetAicpuTaskIdAndStreamId(200, 300);
    HcclCommDfx::AddChannelRemoteRankId("test_comm", handle, 1);

    Hccl::TaskParam wParam{};
    wParam.taskType = Hccl::TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY;
    wParam.beginTime = 1000;
    wParam.endTime = 2000;
    wParam.taskPara.DMA.size = 512;
    wParam.taskPara.DMA.notifyID = 10;
    EXPECT_EQ(dfx_->AddDpuTaskInfoCallback(wParam, handle), HCCL_SUCCESS);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->pendingWriteInfos_[handle].valid, true);

    Hccl::TaskParam fParam{};
    fParam.taskType = Hccl::TaskParamType::TASK_DPU_CHANNEL_FENCE;
    fParam.beginTime = 1500;
    fParam.endTime = 3000;
    fParam.taskPara.Notify.notifyID = INVALID_U64;
    fParam.taskPara.Notify.value = 1;

    HcclResult ret = dfx_->AddDpuTaskInfoCallback(fParam, handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->pendingWriteInfos_[handle].valid, false);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->pendingWriteInfos_[handle].count, 0u);

    HcclCommDfx::channelRemoteRankId_.clear();
}

// 决策点#5+#6 else: ChannelFence 无缓存，skip
TEST_F(HcclCommDfxTest, Ut_AddDpuTaskInfoCallback_When_FenceNoPending_Expect_Skip)
{
    u64 handle = 0x6000;
    dfx_->GetDpuCommDfx()->InitPendingWriteInfo(handle);
    dfx_->GetDpuCommDfx()->SetDpuStreamId(100);
    dfx_->GetDpuCommDfx()->SetAicpuTaskIdAndStreamId(200, 300);
    HcclCommDfx::AddChannelRemoteRankId("test_comm", handle, 1);

    Hccl::TaskParam fParam{};
    fParam.taskType = Hccl::TaskParamType::TASK_DPU_CHANNEL_FENCE;
    fParam.beginTime = 1500;
    fParam.endTime = 3000;
    fParam.taskPara.Notify.notifyID = INVALID_U64;
    fParam.taskPara.Notify.value = 1;

    HcclResult ret = dfx_->AddDpuTaskInfoCallback(fParam, handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommDfx::channelRemoteRankId_.clear();
}

// 决策点#7: TASK_DPU_KERNEL 不写 dpuTaskInfo_
TEST_F(HcclCommDfxTest, Ut_AddDpuTaskInfoCallback_When_DpuKernel_Expect_NoDpuTaskInfoUpdate)
{
    u64 handle = 0x7000;
    dfx_->GetDpuCommDfx()->InitPendingWriteInfo(handle);
    dfx_->GetDpuCommDfx()->SetDpuStreamId(100);
    dfx_->GetDpuCommDfx()->SetAicpuTaskIdAndStreamId(200, 300);
    HcclCommDfx::AddChannelRemoteRankId("test_comm", handle, 1);

    dfx_->GetDpuCommDfx()->dpuTaskInfo_ = {999, 888};

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_DPU_KERNEL;
    taskParam.beginTime = 1000;
    taskParam.endTime = 2000;
    taskParam.isMaster = true;

    HcclResult ret = dfx_->AddDpuTaskInfoCallback(taskParam, handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->dpuTaskInfo_.taskId, 999u);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->dpuTaskInfo_.streamId, 888u);

    HcclCommDfx::channelRemoteRankId_.clear();
}

// 决策点#7 else: 非 KERNEL 走 default 写 dpuTaskInfo_
TEST_F(HcclCommDfxTest, Ut_AddDpuTaskInfoCallback_When_NotifyWait_Expect_DpuTaskInfoUpdated)
{
    u64 handle = 0x8000;
    dfx_->GetDpuCommDfx()->InitPendingWriteInfo(handle);
    dfx_->GetDpuCommDfx()->SetDpuStreamId(100);
    dfx_->GetDpuCommDfx()->SetAicpuTaskIdAndStreamId(200, 300);
    HcclCommDfx::AddChannelRemoteRankId("test_comm", handle, 1);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_DPU_NOTIFY_WAIT;
    taskParam.beginTime = 1000;
    taskParam.endTime = 2000;
    taskParam.taskPara.Notify.notifyID = 5;

    HcclResult ret = dfx_->AddDpuTaskInfoCallback(taskParam, handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 expectedStreamId
        = dfx_->GetDpuCommDfx()->channelStreamIdMap_[handle] - dfx_->GetDpuCommDfx()->channelStreamIdBase_;
    EXPECT_EQ(dfx_->GetDpuCommDfx()->dpuTaskInfo_.streamId, expectedStreamId);

    HcclCommDfx::channelRemoteRankId_.clear();
}

// ==================== InitPendingWriteInfo 测试 ====================

// 决策点#9: 首次调用 InitPendingWriteInfo
TEST_F(HcclCommDfxTest, Ut_InitPendingWriteInfo_When_FirstCall_Expect_EntryCreated)
{
    u64 handle = 0xA001;
    dfx_->GetDpuCommDfx()->InitPendingWriteInfo(handle);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->pendingWriteInfos_.count(handle), 1u);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->pendingWriteInfos_[handle].valid, false);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->pendingWriteInfos_[handle].count, 0u);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->channelStreamIdMap_.count(handle), 1u);
}

// 决策点#9 else: 二次调用 InitPendingWriteInfo 不重复分配 streamId
TEST_F(HcclCommDfxTest, Ut_InitPendingWriteInfo_When_SecondCall_Expect_NoDuplicateStreamId)
{
    u64 handle = 0xA002;
    dfx_->GetDpuCommDfx()->InitPendingWriteInfo(handle);
    u32 firstStreamId = dfx_->GetDpuCommDfx()->channelStreamIdMap_[handle];

    dfx_->GetDpuCommDfx()->InitPendingWriteInfo(handle);
    u32 secondStreamId = dfx_->GetDpuCommDfx()->channelStreamIdMap_[handle];
    EXPECT_EQ(firstStreamId, secondStreamId);
}

// ==================== GetDpuChannelStreamId 测试 ====================

// 决策点#8: handle 已预初始化
TEST_F(HcclCommDfxTest, Ut_GetDpuChannelStreamId_When_HandleExists_Expect_ReturnMappedId)
{
    u64 handle = 0xB001;
    dfx_->GetDpuCommDfx()->InitPendingWriteInfo(handle);
    u32 expectedId = dfx_->GetDpuCommDfx()->channelStreamIdMap_[handle] - dfx_->GetDpuCommDfx()->channelStreamIdBase_;
    u32 actualId = dfx_->GetDpuCommDfx()->GetDpuChannelStreamId(handle);
    EXPECT_EQ(actualId, expectedId);
}

// 决策点#8 else: handle 未预初始化，fallback 到 dpuStreamId_
TEST_F(HcclCommDfxTest, Ut_GetDpuChannelStreamId_When_HandleNotExists_Expect_ReturnDpuStreamId)
{
    u64 handle = 0xB002;
    dfx_->GetDpuCommDfx()->SetDpuStreamId(555);
    u32 actualId = dfx_->GetDpuCommDfx()->GetDpuChannelStreamId(handle);
    EXPECT_EQ(actualId, 555u);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->channelStreamIdMap_.count(handle), 0u);
}

// ==================== GetDpuTaskInfo 测试 ====================

TEST_F(HcclCommDfxTest, Ut_GetDpuTaskInfo_When_Default_Expect_Zero)
{
    const auto& info = dfx_->GetDpuCommDfx()->GetDpuTaskInfo();
    EXPECT_EQ(info.taskId, 0u);
    EXPECT_EQ(info.streamId, 0u);
}

TEST_F(HcclCommDfxTest, Ut_GetDpuTaskInfo_When_AfterWNotifyCache_Expect_Updated)
{
    u64 handle = 0xC001;
    dfx_->GetDpuCommDfx()->InitPendingWriteInfo(handle);
    dfx_->GetDpuCommDfx()->SetDpuStreamId(100);
    dfx_->GetDpuCommDfx()->SetAicpuTaskIdAndStreamId(200, 300);
    HcclCommDfx::AddChannelRemoteRankId("test_comm", handle, 1);

    Hccl::TaskParam wParam{};
    wParam.taskType = Hccl::TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY;
    wParam.beginTime = 1000;
    wParam.endTime = 2000;
    wParam.taskPara.DMA.size = 512;
    wParam.taskPara.DMA.notifyID = 10;

    EXPECT_EQ(dfx_->AddDpuTaskInfoCallback(wParam, handle), HCCL_SUCCESS);

    Hccl::TaskParam fParam{};
    fParam.taskType = Hccl::TaskParamType::TASK_DPU_CHANNEL_FENCE;
    fParam.beginTime = 1500;
    fParam.endTime = 3000;
    fParam.taskPara.Notify.notifyID = INVALID_U64;
    fParam.taskPara.Notify.value = 1;

    EXPECT_EQ(dfx_->AddDpuTaskInfoCallback(fParam, handle), HCCL_SUCCESS);

    const auto& info = dfx_->GetDpuCommDfx()->GetDpuTaskInfo();
    EXPECT_NE(info.taskId, 0u);

    HcclCommDfx::channelRemoteRankId_.clear();
}

// ==================== IsDpuOpInfoEnabled 测试 ====================

TEST_F(HcclCommDfxTest, Ut_IsDpuOpInfoEnabled_When_Default_Expect_False)
{
    EXPECT_EQ(dfx_->GetDpuCommDfx()->IsDpuOpInfoEnabled(), false);
}

TEST_F(HcclCommDfxTest, Ut_IsDpuOpInfoEnabled_After_InitPendingWriteInfo_Expect_True)
{
    u64 handle = 0xE001;
    dfx_->GetDpuCommDfx()->InitPendingWriteInfo(handle);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->IsDpuOpInfoEnabled(), true);
}

// ==================== SetDpuTaskOpInfo 测试 ====================

TEST_F(HcclCommDfxTest, Ut_SetDpuTaskOpInfo_When_Nullptr_Expect_NoThrow)
{
    EXPECT_NO_THROW(dfx_->GetDpuCommDfx()->SetDpuTaskOpInfo(nullptr));
}

TEST_F(HcclCommDfxTest, Ut_SetDpuTaskOpInfo_When_DpuEnabled_Expect_RingWritten)
{
    u64 handle = 0xE002;
    dfx_->GetDpuCommDfx()->InitPendingWriteInfo(handle);
    EXPECT_EQ(dfx_->GetDpuCommDfx()->IsDpuOpInfoEnabled(), true);

    auto opInfo = std::make_shared<Hccl::DfxOpInfo>();
    opInfo->opIndex_ = 42;
    dfx_->GetDpuCommDfx()->SetDpuTaskOpInfo(opInfo);

    constexpr u32 DPU_OP_INFO_RING_CAPACITY = 2048;
    EXPECT_EQ(dfx_->GetDpuCommDfx()->opInfoRing_[42 % DPU_OP_INFO_RING_CAPACITY].get(), opInfo.get());
}

TEST_F(HcclCommDfxTest, Ut_SetCurrDfxOpInfo_When_DpuNotEnabled_Expect_RingEmpty)
{
    auto opInfo = std::make_shared<Hccl::DfxOpInfo>();
    opInfo->opIndex_ = 99;
    EXPECT_EQ(dfx_->SetCurrDfxOpInfo(opInfo), HCCL_SUCCESS);

    constexpr u32 DPU_OP_INFO_RING_CAPACITY = 2048;
    EXPECT_EQ(dfx_->GetDpuCommDfx()->opInfoRing_[99 % DPU_OP_INFO_RING_CAPACITY], nullptr);
}
