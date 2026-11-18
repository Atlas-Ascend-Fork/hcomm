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
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "hccl_comm_pub.h"
#define private public
#define protected public
#include "ndaTaskException.h"
#include "hcclCommTaskException.h"
#include "coll_comm_mgr.h"
#undef private
#undef protected
#include "hccl_types.h"

using namespace hcomm;
using namespace hccl;

namespace {
// 各用例使用互不相同的 deviceLogicId：TaskExceptionHost 按 device 维护单例，
// 复用 device 会让 ei0002Reported_ 等状态串到其他用例
constexpr s32 NDA_DEV_NULL_INFO = 40;
constexpr s32 NDA_DEV_RDMA_TASK = 41;
constexpr s32 NDA_DEV_NOTIFY_WAIT = 42;
constexpr s32 NDA_DEV_NO_COMM = 43;
constexpr s32 NDA_DEV_MISMATCH = 44;
constexpr s32 NDA_DEV_NO_TAG = 45;
constexpr s32 NDA_DEV_NON_ROCE = 46;
constexpr s32 NDA_DEV_ROCE = 47;
constexpr s32 NDA_DEV_STANDARD_ROCE = 48;
constexpr s32 NDA_DEV_MULTI_COMM = 49;
constexpr s32 NDA_DEV_TASK_NOT_FOUND = 24;
constexpr s32 NDA_DEV_DFX_OP_NULL = 25;
constexpr s32 NDA_DEV_NOTIFY_WAIT_DUP = 30;
constexpr s32 NDA_DEV_REPORT_CONSUMED = 50;

// 构造一条 aicpu 上报的 ErrorMessageReport（模拟 KFC D2H 读到的内容）
Hccl::ErrorMessageReport
MakeNdaReport(Hccl::DfxLinkType::Value linkType, Hccl::TaskParamType::Value taskType, s32 streamId, u32 taskId)
{
    Hccl::ErrorMessageReport msg{};
    (void)strncpy(msg.tag, "nda_test_tag", sizeof(msg.tag) - 1);
    msg.linkType = Hccl::DfxLinkType(linkType);
    msg.taskType = Hccl::TaskParamType(taskType);
    msg.streamId = streamId;
    msg.taskId = taskId;
    return msg;
}

hccl::CollComm& AddCollComm(const std::string& commId, s32 devLogicId)
{
    // 用池保活，避免 CollCommMgr 里残留悬垂指针；simpleMode 析构无外部依赖
    static std::vector<std::unique_ptr<hccl::CollComm>> commPool;
    commPool.emplace_back(new hccl::CollComm(nullptr, 0, commId, {}, hccl::CollCommInitMode::simpleMode));
    hccl::CollComm* comm = commPool.back().get();
    comm->deviceLogicId_ = devLogicId;
    CollCommMgr::GetInstance().allCollComms_[commId] = comm;
    return *comm;
}
} // namespace

class NdaTaskExceptionTest : public testing::Test {
protected:
    void SetUp() override { CollCommMgr::GetInstance().allCollComms_.clear(); }
    void TearDown() override
    {
        CollCommMgr::GetInstance().allCollComms_.clear();
        GlobalMockObject::verify();
    }
};

TEST_F(NdaTaskExceptionTest, Ut_ProcessNdaException_When_ExceptionInfoNull_Expect_Para)
{
    TaskExceptionHost* host = TaskExceptionHost::GetInstance(NDA_DEV_NULL_INFO);
    ASSERT_NE(host, nullptr);

    EXPECT_EQ(NdaTaskException::ProcessNdaException(nullptr, *host), HCCL_E_PARA);
}

TEST_F(NdaTaskExceptionTest, Ut_ReadNdaErrorMessage_When_NoComm_Expect_NotFound)
{
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<u32>(NDA_DEV_NO_COMM);
    Hccl::ErrorMessageReport errMsg{};
    std::string commId;

    EXPECT_EQ(NdaTaskException::ReadNdaErrorMessage(&exceptionInfo, errMsg, commId), HCCL_E_NOT_FOUND);
    EXPECT_EQ(commId, "");
}

TEST_F(NdaTaskExceptionTest, Ut_ReadNdaErrorMessage_When_DeviceMismatch_Expect_NotFound)
{
    AddCollComm("nda_dev_mismatch", NDA_DEV_MISMATCH + 1);

    // deviceLogicId 不匹配时不得读该通信域的 D2H 报文
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).expects(never());

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<u32>(NDA_DEV_MISMATCH);
    Hccl::ErrorMessageReport errMsg{};
    std::string commId;

    EXPECT_EQ(NdaTaskException::ReadNdaErrorMessage(&exceptionInfo, errMsg, commId), HCCL_E_NOT_FOUND);
}

TEST_F(NdaTaskExceptionTest, Ut_ReadNdaErrorMessage_When_TagEmpty_Expect_NotFound)
{
    AddCollComm("nda_no_tag", NDA_DEV_NO_TAG);

    // tag 为空表示该通信域没有 aicpu 上报
    Hccl::ErrorMessageReport emptyMsg{};
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).stubs().will(returnValue(emptyMsg));

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<u32>(NDA_DEV_NO_TAG);
    Hccl::ErrorMessageReport errMsg{};
    std::string commId;

    EXPECT_EQ(NdaTaskException::ReadNdaErrorMessage(&exceptionInfo, errMsg, commId), HCCL_E_NOT_FOUND);
}

TEST_F(NdaTaskExceptionTest, Ut_ReadNdaErrorMessage_When_NonRoceLinkType_Expect_NotFound)
{
    AddCollComm("nda_non_roce", NDA_DEV_NON_ROCE);

    // 非 RoCE 链路不是 NDA 的接管范围，交回 legacy 兜底
    Hccl::ErrorMessageReport msg = MakeNdaReport(Hccl::DfxLinkType::UB, Hccl::TaskParamType::TASK_RDMA, 0, 0);
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).stubs().will(returnValue(msg));

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<u32>(NDA_DEV_NON_ROCE);
    Hccl::ErrorMessageReport errMsg{};
    std::string commId;

    EXPECT_EQ(NdaTaskException::ReadNdaErrorMessage(&exceptionInfo, errMsg, commId), HCCL_E_NOT_FOUND);
}

TEST_F(NdaTaskExceptionTest, Ut_ReadNdaErrorMessage_When_RoceLinkType_Expect_Success)
{
    AddCollComm("nda_roce", NDA_DEV_ROCE);

    Hccl::ErrorMessageReport msg = MakeNdaReport(Hccl::DfxLinkType::ROCE, Hccl::TaskParamType::TASK_RDMA, 3, 7);
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).stubs().will(returnValue(msg));

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<u32>(NDA_DEV_ROCE);
    Hccl::ErrorMessageReport errMsg{};
    std::string commId;

    EXPECT_EQ(NdaTaskException::ReadNdaErrorMessage(&exceptionInfo, errMsg, commId), HCCL_SUCCESS);
    EXPECT_EQ(commId, "nda_roce");
    EXPECT_EQ(errMsg.taskId, 7u);
    EXPECT_EQ(errMsg.streamId, 3);
    EXPECT_EQ(errMsg.taskType, Hccl::TaskParamType::TASK_RDMA);
    EXPECT_EQ(errMsg.linkType, Hccl::DfxLinkType(Hccl::DfxLinkType::ROCE));
}

TEST_F(NdaTaskExceptionTest, Ut_ReadNdaErrorMessage_When_StandardRoceLinkType_Expect_Success)
{
    AddCollComm("nda_standard_roce", NDA_DEV_STANDARD_ROCE);

    // STANDARD_ROCE 与 ROCE 同属 NDA 接管范围
    Hccl::ErrorMessageReport msg
        = MakeNdaReport(Hccl::DfxLinkType::STANDARD_ROCE, Hccl::TaskParamType::TASK_RDMA, 1, 2);
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).stubs().will(returnValue(msg));

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<u32>(NDA_DEV_STANDARD_ROCE);
    Hccl::ErrorMessageReport errMsg{};
    std::string commId;

    EXPECT_EQ(NdaTaskException::ReadNdaErrorMessage(&exceptionInfo, errMsg, commId), HCCL_SUCCESS);
    EXPECT_EQ(commId, "nda_standard_roce");
    EXPECT_EQ(errMsg.linkType, Hccl::DfxLinkType(Hccl::DfxLinkType::STANDARD_ROCE));
}

TEST_F(NdaTaskExceptionTest, Ut_ReadNdaErrorMessage_When_MultiComm_Expect_SkipOtherDevice)
{
    // 其他设备的通信域在前/在后都不能影响结果（allCollComms_ 为 unordered_map，顺序不固定）
    AddCollComm("nda_other_dev", NDA_DEV_MULTI_COMM + 1);
    AddCollComm("nda_current_dev", NDA_DEV_MULTI_COMM);

    Hccl::ErrorMessageReport msg = MakeNdaReport(Hccl::DfxLinkType::ROCE, Hccl::TaskParamType::TASK_RDMA, 5, 9);
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).stubs().will(returnValue(msg));

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<u32>(NDA_DEV_MULTI_COMM);
    Hccl::ErrorMessageReport errMsg{};
    std::string commId;

    EXPECT_EQ(NdaTaskException::ReadNdaErrorMessage(&exceptionInfo, errMsg, commId), HCCL_SUCCESS);
    EXPECT_EQ(commId, "nda_current_dev");
}

TEST_F(NdaTaskExceptionTest, Ut_ProcessNdaException_When_TaskNotFound_Expect_NotFound)
{
    AddCollComm("nda_task_not_found", NDA_DEV_TASK_NOT_FOUND);

    Hccl::ErrorMessageReport msg = MakeNdaReport(Hccl::DfxLinkType::ROCE, Hccl::TaskParamType::TASK_RDMA, 0, 999);
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).stubs().will(returnValue(msg));

    TaskExceptionHost* host = TaskExceptionHost::GetInstance(NDA_DEV_TASK_NOT_FOUND);
    ASSERT_NE(host, nullptr);

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<u32>(NDA_DEV_TASK_NOT_FOUND);

    // UT 环境 GlobalMirrorTasks::FindTaskInfo 为空桩（返回 NOT_FOUND）
    EXPECT_EQ(NdaTaskException::ProcessNdaException(&exceptionInfo, *host), HCCL_E_NOT_FOUND);
}

TEST_F(NdaTaskExceptionTest, Ut_ProcessNdaException_When_NonRoceReport_Expect_NotFound)
{
    AddCollComm("nda_process_non_roce", NDA_DEV_NON_ROCE + 1);

    Hccl::ErrorMessageReport msg = MakeNdaReport(Hccl::DfxLinkType::HCCS, Hccl::TaskParamType::TASK_RDMA, 0, 0);
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).stubs().will(returnValue(msg));

    TaskExceptionHost* host = TaskExceptionHost::GetInstance(NDA_DEV_NON_ROCE + 1);
    ASSERT_NE(host, nullptr);

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<u32>(NDA_DEV_NON_ROCE + 1);

    EXPECT_EQ(NdaTaskException::ProcessNdaException(&exceptionInfo, *host), HCCL_E_NOT_FOUND);
}

TEST_F(NdaTaskExceptionTest, Ut_ProcessNdaException_When_RdmaTaskFound_Expect_Success)
{
    AddCollComm("nda_rdma_task", NDA_DEV_RDMA_TASK);

    Hccl::ErrorMessageReport msg = MakeNdaReport(Hccl::DfxLinkType::ROCE, Hccl::TaskParamType::TASK_RDMA, 2, 7);
    msg.rankId = 0;
    msg.remoteUserRank = 1;
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).stubs().will(returnValue(msg));
    // 非 notify-wait 的 RoCE 异常走 EI0018 上报
    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_RDMA;
    auto dfxOpInfo = std::make_shared<Hccl::DfxOpInfo>();
    dfxOpInfo->comm_ = nullptr; // ReportEI0018Error 内部对 nullptr 做了保护
    Hccl::TaskInfo taskInfo(2, 7, 1, taskParam, dfxOpInfo);
    MOCKER_CPP(&Hccl::GlobalMirrorTasks::FindTaskInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(&taskInfo))
        .will(returnValue(HCCL_SUCCESS));

    TaskExceptionHost* host = TaskExceptionHost::GetInstance(NDA_DEV_RDMA_TASK);
    ASSERT_NE(host, nullptr);

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<u32>(NDA_DEV_RDMA_TASK);

    EXPECT_EQ(NdaTaskException::ProcessNdaException(&exceptionInfo, *host), HCCL_SUCCESS);
}

TEST_F(NdaTaskExceptionTest, Ut_ProcessNdaException_When_DfxOpInfoNull_Expect_SkipEi0018Report)
{
    AddCollComm("nda_dfx_op_null", NDA_DEV_DFX_OP_NULL);

    Hccl::ErrorMessageReport msg = MakeNdaReport(Hccl::DfxLinkType::ROCE, Hccl::TaskParamType::TASK_RDMA, 2, 7);
    msg.remoteUserRank = 1;
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).stubs().will(returnValue(msg));
    // 任务未登记算子信息（dfxOpInfo_ 为空）时，EI0018 上报需跳过，不得解引用空指针
    MOCKER(RptInputErr).expects(never());

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_RDMA;
    Hccl::TaskInfo taskInfo(2, 7, 1, taskParam, nullptr);
    MOCKER_CPP(&Hccl::GlobalMirrorTasks::FindTaskInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(&taskInfo))
        .will(returnValue(HCCL_SUCCESS));

    TaskExceptionHost* host = TaskExceptionHost::GetInstance(NDA_DEV_DFX_OP_NULL);
    ASSERT_NE(host, nullptr);

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<u32>(NDA_DEV_DFX_OP_NULL);

    EXPECT_EQ(NdaTaskException::ProcessNdaException(&exceptionInfo, *host), HCCL_SUCCESS);
}

TEST_F(NdaTaskExceptionTest, Ut_ProcessNdaException_When_NotifyWaitNotReported_Expect_Ei0002Report)
{
    AddCollComm("nda_notify_wait", NDA_DEV_NOTIFY_WAIT);

    Hccl::ErrorMessageReport msg = MakeNdaReport(Hccl::DfxLinkType::ROCE, Hccl::TaskParamType::TASK_NOTIFY_WAIT, 2, 8);
    msg.remoteUserRank = 1;
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).stubs().will(returnValue(msg));
    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    Hccl::TaskInfo taskInfo(2, 8, 1, taskParam, nullptr);
    MOCKER_CPP(&Hccl::GlobalMirrorTasks::FindTaskInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(&taskInfo))
        .will(returnValue(HCCL_SUCCESS));

    TaskExceptionHost* host = TaskExceptionHost::GetInstance(NDA_DEV_NOTIFY_WAIT);
    ASSERT_NE(host, nullptr);
    host->ei0002Reported_ = false;

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<u32>(NDA_DEV_NOTIFY_WAIT);

    // notify wait 超时：ShouldReportError 返回 true，走 EI0002 上报
    EXPECT_EQ(NdaTaskException::ProcessNdaException(&exceptionInfo, *host), HCCL_SUCCESS);
    EXPECT_TRUE(host->ei0002Reported_);
}

TEST_F(NdaTaskExceptionTest, Ut_ProcessNdaException_When_NotifyWaitAlreadyReported_Expect_SkipDuplicate)
{
    AddCollComm("nda_notify_wait_dup", NDA_DEV_NOTIFY_WAIT_DUP);

    Hccl::ErrorMessageReport msg = MakeNdaReport(Hccl::DfxLinkType::ROCE, Hccl::TaskParamType::TASK_NOTIFY_WAIT, 2, 8);
    msg.remoteUserRank = 1;
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).stubs().will(returnValue(msg));
    // 同设备已上报过 EI0002，本次不得重复上报
    MOCKER(RptInputErr).expects(never());

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    Hccl::TaskInfo taskInfo(2, 8, 1, taskParam, nullptr);
    MOCKER_CPP(&Hccl::GlobalMirrorTasks::FindTaskInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(&taskInfo))
        .will(returnValue(HCCL_SUCCESS));

    TaskExceptionHost* host = TaskExceptionHost::GetInstance(NDA_DEV_NOTIFY_WAIT_DUP);
    ASSERT_NE(host, nullptr);
    host->ei0002Reported_ = true;

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<u32>(NDA_DEV_NOTIFY_WAIT_DUP);

    EXPECT_EQ(NdaTaskException::ProcessNdaException(&exceptionInfo, *host), HCCL_SUCCESS);
    EXPECT_TRUE(host->ei0002Reported_);
}

TEST_F(NdaTaskExceptionTest, Ut_ProcessNdaException_When_SecondExceptionOnSameDevice_Expect_NoRepeatReport)
{
    AddCollComm("nda_report_consumed", NDA_DEV_REPORT_CONSUMED);

    Hccl::ErrorMessageReport msg = MakeNdaReport(Hccl::DfxLinkType::ROCE, Hccl::TaskParamType::TASK_RDMA, 2, 7);
    msg.rankId = 0;
    msg.remoteUserRank = 1;
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).stubs().will(returnValue(msg));
    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_RDMA;
    auto dfxOpInfo = std::make_shared<Hccl::DfxOpInfo>();
    dfxOpInfo->comm_ = nullptr; // ReportEI0018Error 内部对 nullptr 做了保护
    Hccl::TaskInfo taskInfo(2, 7, 1, taskParam, dfxOpInfo);
    MOCKER_CPP(&Hccl::GlobalMirrorTasks::FindTaskInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(&taskInfo))
        .will(returnValue(HCCL_SUCCESS));

    TaskExceptionHost* host = TaskExceptionHost::GetInstance(NDA_DEV_REPORT_CONSUMED);
    ASSERT_NE(host, nullptr);
    host->hasAicpuReport_ = false;

    rtExceptionInfo_t firstException{};
    firstException.deviceid = static_cast<u32>(NDA_DEV_REPORT_CONSUMED);
    EXPECT_EQ(NdaTaskException::ProcessNdaException(&firstException, *host), HCCL_SUCCESS);
    EXPECT_TRUE(host->hasAicpuReport_.load());

    GlobalMockObject::verify();

    // 同设备第二个异常：报文已被一次性消费，不得再读旧报文，也不得重复上报（EI0018 无去重）
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).expects(never());
    MOCKER(RptInputErr).expects(never());
    MOCKER_CPP(&Hccl::GlobalMirrorTasks::FindTaskInfo).expects(never());

    rtExceptionInfo_t secondException{};
    secondException.deviceid = static_cast<u32>(NDA_DEV_REPORT_CONSUMED);
    secondException.streamid = 9;
    secondException.taskid = 11;
    EXPECT_EQ(NdaTaskException::ProcessNdaException(&secondException, *host), HCCL_E_NOT_FOUND);
}
