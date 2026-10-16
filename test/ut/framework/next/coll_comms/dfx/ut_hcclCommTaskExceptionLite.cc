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
#include "hccl_comm_pub.h"
#define private public
#include "hcclCommTaskExceptionLite.h"
#include "aicpu_ts_thread.h"
#include "hcclCommTaskException.h"
#undef private
#include "hcomm_task_scheduler_error.h"
#include "aicpu_indop_env.h"
#include "comm_engine_res_aicpu_mgr.h"
#include "adapter_hal_pub.h"
#include "dlhal_function_v2.h"
#include "rtsq_base.h"
#include "coll_comm_aicpu_mgr.h"
#include "dfx_profiling_handler_lite.h"
#include "adapter_error_manager_pub.h"
#include "task_info.h"
#include "hccp.h"
#include "hccp_ctx.h"

using namespace hccl;
using namespace hcomm;

constexpr u32 RT_UB_LOCAL_OPERATIOINERR = 0x2;
constexpr u32 RT_UB_REMOTE_OPERATIOINERR = 0x3;
constexpr u32 RT_UB_LINK_FAILEDERR = 0x5;

inline void InitCommEngineResMgr(CollCommAicpu& c)
{
    if (c.commEngineResMgr_ == nullptr) {
        c.commEngineResMgr_ = std::make_unique<CommEngineResAicpuMgr>(c.dfx_, [](bool) {
            return HCCL_SUCCESS;
        });
    }
}
class hcclCommTaskExceptionLiteTest : public testing::Test {
protected:
    virtual void SetUp() override
    {
        MOCKER(::getpid).stubs().will(returnValue(12345));
        MOCKER(HrtHalDrvQueryProcessHostPid).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(HrtHalDrvGetDevIDByLocalDevID).stubs().will(returnValue(HCCL_SUCCESS));
        Hccl::DlHalFunctionV2::GetInstance().dlHalEschedSubmitEvent
            = [](unsigned int, struct event_summary*) -> drvError_t {
            return DRV_ERROR_NONE;
        };
        Hccl::DlHalFunctionV2::GetInstance().dlHalDrvQueryProcessHostPid
            = [](int, unsigned int*, unsigned int* vfid, unsigned int* hostpid, unsigned int*) -> drvError_t {
            *vfid = 0;
            *hostpid = 12345;
            return DRV_ERROR_NONE;
        };
        HcclCommTaskExceptionLite::GetInstance().Init(0);
        HcclCommTaskExceptionLite::GetInstance().stopCall_ = false;
        HcclCommTaskExceptionLite::GetInstance().threadsPrinted_.clear();
        CollCommAicpuMgr::GetInstance().GetTaskExpDevMemRegistry().Clear();
    }

    virtual void TearDown() override
    {
        CollCommAicpuMgr::GetInstance().GetTaskExpDevMemRegistry().Clear();
        GlobalMockObject::verify();
    }

private:
    u32 notifyId = 1;
    u32 tsId = 2;
};

TEST_F(hcclCommTaskExceptionLiteTest, Ut_CCoreError_DoesNotDecodeHardwareNotify)
{
    for (const auto type : {Hccl::TASK_CCORE_NOTIFY_WAIT, Hccl::TASK_CCORE_NOTIFY_RECORD}) {
        Hccl::DfxTaskInfo task{};
        task.taskType = type;
        task.taskPara.Notify.notifyId = 1;
        Hccl::ErrorMessageReport error{};
        error.taskType = Hccl::TaskParamType(static_cast<Hccl::TaskParamType::Value>(task.taskType));
        rtLogicCqReport_t report{};
        HcclCommTaskExceptionLite::GetInstance().GenerateTaskErrMsg(task, error, report);
        EXPECT_EQ(error.taskType, Hccl::TaskParamType::INVALID);
        EXPECT_EQ(error.notifyId, INVALID_U32);
        EXPECT_EQ(error.notifyValue, INVALID_U32);
    }
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SwitchUBCqeErrCodeToTsErrCode_When_Normal_Expect_ReturnIsCorrect)
{
    uint16_t ret = HcclCommTaskExceptionLite::GetInstance().SwitchUBCqeErrCodeToTsErrCode(RT_UB_LOCAL_OPERATIOINERR);
    EXPECT_EQ(ret, TS_ERROR_HCCL_OP_UB_DDRC_FAILED);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchUBCqeErrCodeToTsErrCode(RT_UB_REMOTE_OPERATIOINERR);
    EXPECT_EQ(ret, TS_ERROR_HCCL_OP_UB_POISON_FAILED);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchUBCqeErrCodeToTsErrCode(RT_UB_LINK_FAILEDERR);
    EXPECT_EQ(ret, TS_ERROR_HCCL_OP_UB_LINK_FAILED);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchUBCqeErrCodeToTsErrCode(static_cast<u32>(123));
    EXPECT_EQ(ret, TS_ERROR_HCCL_OTHER_ERROR);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SwitchSdmaCqeErrCodeToTsErrCode_When_Normal_Expect_ReturnIsCorrect)
{
    uint16_t ret = HcclCommTaskExceptionLite::GetInstance().SwitchSdmaCqeErrCodeToTsErrCode(RT_SDMA_COMPERR);
    EXPECT_EQ(ret, TS_ERROR_SDMA_LINK_ERROR);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchSdmaCqeErrCodeToTsErrCode(RT_SDMA_COMPDATAERR);
    EXPECT_EQ(ret, TS_ERROR_SDMA_POISON_ERROR);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchSdmaCqeErrCodeToTsErrCode(RT_SDMA_DATAERR);
    EXPECT_EQ(ret, TS_ERROR_SDMA_DDRC_ERROR);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchSdmaCqeErrCodeToTsErrCode(static_cast<u32>(123));
    EXPECT_EQ(ret, TS_ERROR_HCCL_OTHER_ERROR);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SwitchSdmaCqeErrCodeToTsErrCode_taskexception_disable)
{
    hcomm::SetTaskExceptionEnable(false);
    rtLogicCqReport_t exceptionInfo;
    dfx::CqeStatus cqeStatus = dfx::CqeStatus::kDefault;
    std::vector<std::pair<std::string, CollCommAicpu*>> aicpuCommInfo;
    HcclResult ret
        = HcclCommTaskExceptionLite::GetInstance().ProcessCqe(nullptr, exceptionInfo, cqeStatus, aicpuCommInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hcomm::SetTaskExceptionEnable(true);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SendTaskExceptionByMBox_When_UBSqeType_Expect_ReturnHCCL_SUCCESS)
{
    rtLogicCqReport_t exceptionInfo;
    exceptionInfo.sqeType = 9;
    exceptionInfo.errorCode = RT_UB_LOCAL_OPERATIOINERR;

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().SendTaskExceptionByMBox(notifyId, tsId, exceptionInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SendTaskExceptionByMBox_When_SDMASqeType_Expect_ReturnHCCL_SUCCESS)
{
    rtLogicCqReport_t exceptionInfo;
    exceptionInfo.sqeType = 11;
    exceptionInfo.errorCode = RT_SDMA_COMPERR;

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().SendTaskExceptionByMBox(notifyId, tsId, exceptionInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SendTaskExceptionByMBox_When_OtherSqeType_Expect_ReturnHCCL_SUCCESS)
{
    rtLogicCqReport_t exceptionInfo;
    exceptionInfo.sqeType = 8;
    exceptionInfo.errorCode = 123;

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().SendTaskExceptionByMBox(notifyId, tsId, exceptionInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_PrintAllCommTaskException)
{
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));

    CommAicpuParam commAicpuParam;
    std::string commName = "taskException_test_group";
    strncpy(commAicpuParam.hcomId, commName.c_str(), HCOMID_MAX_SIZE - 1);
    EXPECT_EQ(CollCommAicpuMgr::GetInstance().InitComm(&commAicpuParam), HCCL_SUCCESS);

    // InitAicpuIndOp 被 mock，手动初始化 commEngineResMgr_ 以防 PrintAllCommTaskException 空指针
    std::vector<std::pair<std::string, CollCommAicpu*>> commInfo;
    CollCommAicpuMgr::GetInstance().GetAllComms(commInfo);
    for (auto& kv : commInfo) {
        if (kv.second->commEngineResMgr_ == nullptr) {
            kv.second->commEngineResMgr_ = std::make_unique<CommEngineResAicpuMgr>(kv.second->dfx_, [](bool) {
                return HCCL_SUCCESS;
            });
        }
    }

    EXPECT_EQ(hcomm::HcclCommTaskExceptionLite::GetInstance().PrintAllCommTaskException(), HCCL_SUCCESS);
    EXPECT_EQ(CollCommAicpuMgr::GetInstance().DestroyComm(commAicpuParam.hcomId), HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetGroupInfo_When_AicpuCommNullptr_Expect_ReturnEmpty)
{
    std::string result = HcclCommTaskExceptionLite::GetInstance().GetGroupInfo(nullptr);
    EXPECT_EQ(result, "");
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetGroupInfo_When_AicpuCommValid_Expect_ReturnGroupName)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_group_name";
    InitCommEngineResMgr(aicpuComm);
    std::string result = HcclCommTaskExceptionLite::GetInstance().GetGroupInfo(&aicpuComm);
    EXPECT_EQ(result, "group:[test_group_name], hcclUdi:[], rankSize:[0], localRank:[0]");
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetGroupInfo_When_AicpuCommValidWithUdi_Expect_ReturnGroupNameAndUdi)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_group_name";
    aicpuComm.udi_ = "test_udi_123";
    InitCommEngineResMgr(aicpuComm);
    std::string result = HcclCommTaskExceptionLite::GetInstance().GetGroupInfo(&aicpuComm);
    EXPECT_EQ(result, "group:[test_group_name], hcclUdi:[test_udi_123], rankSize:[0], localRank:[0]");
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetGroupRankInfo_When_HostCommValid_Expect_ReturnGroupNameAndUdi)
{
    const s32 testDeviceId = 63;
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    Hccl::TaskInfo taskInfo(0, 0, 1, taskParam, nullptr);
    taskInfo.dfxOpInfo_ = std::make_shared<Hccl::DfxOpInfo>();
    hccl::ManagerCallbacks callbacks;
    hccl::CollComm collComm(nullptr, 0, "test_group_name", callbacks);
    taskInfo.dfxOpInfo_->comm_ = &collComm;

    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);
    std::string result = handler->GetGroupRankInfo(taskInfo);
    EXPECT_TRUE(result.find("group:[test_group_name]") != std::string::npos);
    EXPECT_TRUE(result.find("hcclUdi:[]") != std::string::npos);

    collComm.GetCommConfig().SetConfigUdi(std::string("test_udi_123"));
    result = handler->GetGroupRankInfo(taskInfo);
    EXPECT_TRUE(result.find("hcclUdi:[test_udi_123]") != std::string::npos);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_CommIdNotInMap_Expect_ReturnSuccess)
{
    std::string testCommId = "dpuExpTest";
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_TaskexceptionVaNull_Expect_ReturnSuccess)
{
    std::string testCommId = "dpuExpTest";
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    aicpuComm.dfx_.SetTaskExpDevMem(nullptr);

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_StopFlagIsOne_Expect_ClearFlagAndSetNullptr)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 1;

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    aicpuComm.dfx_.SetTaskExpDevMem(shmem.data());

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(shmem[0], 0);
    EXPECT_EQ(aicpuComm.dfx_.IsTaskExpStopped(), true);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_ErrorFlagZero_Expect_ReturnSuccess)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 0;

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    aicpuComm.dfx_.SetTaskExpDevMem(shmem.data());

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(
    hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_ErrorFlagNonZeroAndDfxLiteNull_Expect_ReturnPtrNull)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 0;
    uint16_t errVal = 1;
    memcpy(shmem.data() + 1, &errVal, sizeof(uint16_t));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    aicpuComm.dfx_.SetTaskExpDevMem(shmem.data());

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(
    hcclCommTaskExceptionLiteTest,
    Ut_HandleDpuTaskexception_When_ErrorFlagNonZeroAndMirrorTaskMgrNull_Expect_ReturnPtrNull)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 0;
    uint16_t errVal = 1;
    memcpy(shmem.data() + 1, &errVal, sizeof(uint16_t));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    aicpuComm.dfx_ = HcclCommDfxLite();
    aicpuComm.dfx_.SetTaskExpDevMem(shmem.data());

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(
    hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_ErrorFlagNonZeroAndDfxOpInfoNull_Expect_ReturnPtrNull)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 0;
    uint16_t errVal = 1;
    memcpy(shmem.data() + 1, &errVal, sizeof(uint16_t));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    aicpuComm.dfx_.Init(0, testCommId, 0, 0);
    aicpuComm.dfx_.SetTaskExpDevMem(shmem.data());

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(
    hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_ErrorFlagNonZero_Expect_SendTaskExceptionAndClearFlag)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 0;
    uint16_t errVal = 1;
    memcpy(shmem.data() + 1, &errVal, sizeof(uint16_t));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    aicpuComm.dfx_.Init(0, testCommId, 0, 0);
    Hccl::DfxDfxOpInfo dfxOpInfo;
    dfxOpInfo.cpuWaitAicpuNotifyId = 10;
    aicpuComm.dfx_.SetCurrDfxOpInfo(&dfxOpInfo);
    aicpuComm.dfx_.SetTaskExpDevMem(shmem.data());

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint16_t clearedFlag = 0xFFFF;
    memcpy(&clearedFlag, shmem.data() + 1, sizeof(uint16_t));
    EXPECT_EQ(clearedFlag, 0);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_Call_ReturnHCCL_SUCCESS_When_CommStatusSuSpending)
{
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));

    CommAicpuParam commAicpuParam;
    std::string commName = "taskException_test_group";
    strncpy(commAicpuParam.hcomId, commName.c_str(), HCOMID_MAX_SIZE - 1);
    EXPECT_EQ(CollCommAicpuMgr::GetInstance().InitComm(&commAicpuParam), HCCL_SUCCESS);
    MOCKER_CPP(&CollCommAicpu::GetCommmStatus).stubs().will(returnValue(HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING));
    hcomm::HcclCommTaskExceptionLite::GetInstance().Call();
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetConciseTaskName_When_NotifyWait_Expect_ReturnName)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT);
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    std::string name = HcclCommTaskExceptionLite::GetInstance().GetConciseTaskName(taskInfo);
    EXPECT_FALSE(name.empty());
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetConciseTaskName_When_Sdma_Expect_ReturnName)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA);
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    std::string name = HcclCommTaskExceptionLite::GetInstance().GetConciseTaskName(taskInfo);
    EXPECT_FALSE(name.empty());
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetNotifyInfo_When_UbDma_Expect_ReturnNotifyId)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB_INLINE_WRITE);
    taskInfo.taskPara.ubDma.notifyId = 42;
    std::string info = HcclCommTaskExceptionLite::GetInstance().GetNotifyInfo(taskInfo);
    EXPECT_EQ(info, "42");
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetNotifyInfo_When_ReducWithNotify_Expect_ReturnNotifyId)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY);
    taskInfo.taskPara.Reduce.notifyId = 99;
    std::string info = HcclCommTaskExceptionLite::GetInstance().GetNotifyInfo(taskInfo);
    EXPECT_EQ(info, "99");
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetNotifyInfo_When_DefaultTask_Expect_ReturnSlash)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA);
    std::string info = HcclCommTaskExceptionLite::GetInstance().GetNotifyInfo(taskInfo);
    EXPECT_EQ(info, "/");
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetRemoteRankId_When_DfxOpInfoInvalid_Expect_ReturnInvalidRankId)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    u32 rankId = HcclCommTaskExceptionLite::GetInstance().GetRemoteRankId(taskInfo);
    EXPECT_EQ(rankId, Hccl::DFX_INVALID_RANKID);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetRemoteRankId_When_OpInfoNull_Expect_ReturnInvalidRankId)
{
    Hccl::DfxTaskInfo taskInfo{};
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.hcclCommDfxLite = nullptr;
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    u32 rankId = HcclCommTaskExceptionLite::GetInstance().GetRemoteRankId(taskInfo);
    EXPECT_EQ(rankId, Hccl::DFX_INVALID_RANKID);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_PrintTaskContextInfo_When_QueueNull_Expect_ReturnParaError)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_context";
    InitCommEngineResMgr(aicpuComm);
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().PrintTaskContextInfo(&aicpuComm, 0, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_CollectTaskContext_When_QueueNull_Expect_ReturnParaError)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_collect";
    InitCommEngineResMgr(aicpuComm);
    std::vector<Hccl::DfxTaskInfo*> taskContext;
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().CollectTaskContext(&aicpuComm, 0, 0, taskContext);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GenerateErrorMessageReport_When_DfxOpInfoInvalid_Expect_ReturnPtrError)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_gen_err";
    InitCommEngineResMgr(aicpuComm);
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    rtLogicCqReport_t exceptionInfo{};
    Hccl::ErrorMessageReport errMsgInfo{};
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().GenerateErrorMessageReport(
        &aicpuComm, taskInfo, exceptionInfo, errMsgInfo);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_ReportErrMsg_When_CommNull_Expect_ReturnPtrError)
{
    rtLogicCqReport_t exceptionInfo{};
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().ReportErrMsg(nullptr, exceptionInfo);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

class MockThreadForReportErr : public Thread {
public:
    MockThreadForReportErr(void* streamPtr) : streamPtr_(streamPtr) {}
    HcclResult Init() override { return HCCL_SUCCESS; }
    HcclResult DeInit() override { return HCCL_SUCCESS; }
    std::string& GetUniqueId() override { return uniqueId_; }
    uint32_t GetNotifyNum() const override { return 0; }
    LocalNotify* GetNotify(uint32_t index) const override { return nullptr; }
    HcclResult SupplementNotify(uint32_t notifyNum) override { return HCCL_SUCCESS; }
    bool IsDeviceA5() const override { return true; }
    Stream* GetStream() const override { return nullptr; }
    void* GetStreamLitePtr() const override { return streamPtr_; }
    void LaunchTask() const override {}
    void TryLaunchTask() const override {}
    HcclResult LocalNotifyRecord(uint32_t notifyId) const override { return HCCL_SUCCESS; }
    HcclResult LocalNotifyWait(uint32_t notifyId) const override { return HCCL_SUCCESS; }
    HcclResult LocalNotifyRecord(ThreadHandle dstThread, uint32_t dstNotifyIdx) const override { return HCCL_SUCCESS; }
    HcclResult LocalNotifyWait(uint32_t notifyIdx, uint32_t timeOut) const override { return HCCL_SUCCESS; }
    HcclResult LocalCopy(void* dst, const void* src, uint64_t sizeByte) const override { return HCCL_SUCCESS; }
    HcclResult LocalReduce(
        void* dst, const void* src, uint64_t sizeByte, HcommDataType dataType, HcommReduceOp reduceOp) const override
    {
        return HCCL_SUCCESS;
    }
    bool GetMaster() const override { return false; }
    void SetIsMaster(bool isMaster) override {}

private:
    void* streamPtr_;
    std::string uniqueId_{"mock_thread"};
};

TEST_F(hcclCommTaskExceptionLiteTest, Ut_ReportErrMsg_When_FindDfxTaskInfoNull_Expect_ReturnPtrError)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_report_err";
    InitCommEngineResMgr(aicpuComm);
    rtLogicCqReport_t exceptionInfo{};
    exceptionInfo.streamId = 0;
    exceptionInfo.sqId = 99;
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().ReportErrMsg(&aicpuComm, exceptionInfo);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_ReportErrMsg_When_DfxOpInfoInvalid_Expect_ReturnPtrError)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_report_err";
    InitCommEngineResMgr(aicpuComm);
    auto streamLite = std::make_shared<Hccl::StreamLite>(0, 0, 0, 0);
    Hccl::DfxTaskInfo* slot = static_cast<Hccl::DfxTaskInfo*>(streamLite->taskInfos_.NextSlot());
    ASSERT_NE(slot, nullptr);
    slot->taskId = (1U << 16) | 0U;
    slot->dfxOpInfo = DFX_INVALID_U64;
    auto thread = std::make_shared<MockThreadForReportErr>(streamLite.get());

    aicpuComm.GetCommEngineResMgr()->threadMgr_->threads_.push_back(thread);
    rtLogicCqReport_t exceptionInfo{};
    exceptionInfo.taskId = 1;
    exceptionInfo.streamId = 0;
    exceptionInfo.sqId = 0;
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().ReportErrMsg(&aicpuComm, exceptionInfo);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_ReportErrMsg_When_ErrorAlreadyReported_Expect_ReturnSuccess)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_report_err";
    InitCommEngineResMgr(aicpuComm);
    aicpuComm.isErrorReported_ = true;
    auto streamLite = std::make_shared<Hccl::StreamLite>(0, 0, 0, 0);
    Hccl::DfxTaskInfo* slot = static_cast<Hccl::DfxTaskInfo*>(streamLite->taskInfos_.NextSlot());
    ASSERT_NE(slot, nullptr);
    slot->taskId = (1U << 16) | 0U;
    Hccl::DfxDfxOpInfo opInfo{};
    slot->dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    auto thread = std::make_shared<MockThreadForReportErr>(streamLite.get());

    aicpuComm.GetCommEngineResMgr()->threadMgr_->threads_.push_back(thread);
    rtLogicCqReport_t exceptionInfo{};
    exceptionInfo.taskId = 1;
    exceptionInfo.streamId = 0;
    exceptionInfo.sqId = 0;
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().ReportErrMsg(&aicpuComm, exceptionInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_ReportErrMsg_When_QueueMarkedAllRead_Expect_StillFindTask)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_report_err";
    InitCommEngineResMgr(aicpuComm);
    aicpuComm.isErrorReported_ = true;
    auto streamLite = std::make_shared<Hccl::StreamLite>(0, 0, 0, 0);
    Hccl::DfxTaskInfo* slot = static_cast<Hccl::DfxTaskInfo*>(streamLite->taskInfos_.NextSlot());
    ASSERT_NE(slot, nullptr);
    slot->taskId = (1U << 16) | 0U;
    Hccl::DfxDfxOpInfo opInfo{};
    slot->dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    streamLite->taskInfos_.MarkAllRead();
    auto thread = std::make_shared<MockThreadForReportErr>(streamLite.get());

    aicpuComm.GetCommEngineResMgr()->threadMgr_->threads_.push_back(thread);
    rtLogicCqReport_t exceptionInfo{};
    exceptionInfo.taskId = 1;
    exceptionInfo.streamId = 0;
    exceptionInfo.sqId = 0;
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().ReportErrMsg(&aicpuComm, exceptionInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_CollectTaskContext_When_MarkAllRead_Expect_StopAtEmptySlot)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_collect_ctx";
    InitCommEngineResMgr(aicpuComm);
    auto streamLite = std::make_shared<Hccl::StreamLite>(0, 0, 0, 0);

    Hccl::DfxDfxOpInfo opInfo{};
    for (u32 i = 0; i < 3; i++) {
        Hccl::DfxTaskInfo* slot = static_cast<Hccl::DfxTaskInfo*>(streamLite->taskInfos_.NextSlot());
        ASSERT_NE(slot, nullptr);
        slot->taskId = ((i + 1U) << 16) | 0U;
        slot->dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    }
    streamLite->taskInfos_.MarkAllRead();

    auto thread = std::make_shared<MockThreadForReportErr>(streamLite.get());
    aicpuComm.GetCommEngineResMgr()->threadMgr_->threads_.push_back(thread);

    std::vector<Hccl::DfxTaskInfo*> taskContext;
    HcclResult ret
        = HcclCommTaskExceptionLite::GetInstance().CollectTaskContext(&aicpuComm, 0, (3U << 16) | 0U, taskContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(taskContext.size(), 3u);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GenerateErrorMessageReport_When_UbTask_Expect_JettyFilled)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_gen_err_jetty";
    InitCommEngineResMgr(aicpuComm);

    const u64 jettyHandle = 0x1234567890abULL;
    const u32 jettyId = 7;

    Hccl::DfxDfxOpInfo opInfo{};
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB);
    taskInfo.taskPara.ubDma.jettyHandle = jettyHandle;
    taskInfo.taskPara.ubDma.jettyId = jettyId;

    rtLogicCqReport_t exceptionInfo{};
    Hccl::ErrorMessageReport errMsgInfo{};
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().GenerateErrorMessageReport(
        &aicpuComm, taskInfo, exceptionInfo, errMsgInfo);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(errMsgInfo.jettyHandle, jettyHandle);
    EXPECT_EQ(errMsgInfo.jettyId, jettyId);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_HandleHostErrorReport_When_DuplicateReport_Expect_SkipSecond)
{
    const s32 testDeviceId = 62;
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<uint32_t>(testDeviceId);
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    Hccl::TaskInfo taskInfo(0, 0, 1, taskParam, nullptr);

    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);

    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));
    handler->HandleHostErrorReport(&exceptionInfo, taskInfo);
    GlobalMockObject::verify();

    MOCKER(RptInputErr).expects(never());
    handler->HandleHostErrorReport(&exceptionInfo, taskInfo);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_ReportErrorMsg_When_DuplicateReport_Expect_SkipSecond)
{
    const s32 testDeviceId = 61;
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<uint32_t>(testDeviceId);
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    Hccl::TaskInfo taskInfo(0, 0, 1, taskParam, nullptr);

    Hccl::ErrorMessageReport errorMessage{};

    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);

    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));
    handler->ReportErrorMsg(taskInfo, "", errorMessage, &exceptionInfo, "");
    GlobalMockObject::verify();

    MOCKER(RptInputErr).expects(never());
    handler->ReportErrorMsg(taskInfo, "", errorMessage, &exceptionInfo, "");
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_Register_When_CommRegisterMapEmpty_Expect_RegisterCallbackAndUnregisterLegacy)
{
    const s32 testDeviceId = 62;
    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);
    handler->CommRegisterMap_.clear();

    HcclResult ret = handler->Register(0xABCD);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(handler->CommRegisterMap_.count(0xABCD), 1u);

    handler->CommRegisterMap_.clear();
}

// ============ NotifyControlPlaneOnUbError 测试 ============

TEST_F(hcclCommTaskExceptionLiteTest, Ut_NotifyControlPlaneOnUbError_AllBranches)
{
    struct TestScene {
        uint16_t ubCqeStatus;
        bool capabilitySupported;
        RdmaHandle rdmaHandle;
        bool expectNotify;
        int32_t notifyRet;
        std::string sceneName;
    };
    const std::vector<TestScene> scenes = {
        {0, true, reinterpret_cast<RdmaHandle>(0x100), false, 0, "UbCqeStatusZero"},
        {1, false, reinterpret_cast<RdmaHandle>(0x100), false, 0, "CapabilityNotSupported"},
        {1, true, nullptr, false, 0, "RdmaHandleNull"},
        {1, true, reinterpret_cast<RdmaHandle>(0x200), true, 0, "NotifySuccess"},
        {1, true, reinterpret_cast<RdmaHandle>(0x300), true, -1, "NotifyFailNoCrash"},
    };

    const s32 testDeviceId = 0;
    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);

    for (const auto& scene : scenes) {
        SCOPED_TRACE("scene: " + scene.sceneName);
        Hccl::ErrorMessageReport errorMessage{};
        errorMessage.ubCqeStatus = scene.ubCqeStatus;
        errorMessage.tpn = 5;

        MOCKER(RaHasCapability)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(scene.capabilitySupported));
        if (scene.expectNotify) {
            MOCKER(RaCtxNotifyEvent)
                .expects(once())
                .with(mockcpp::any(), mockcpp::any())
                .will(returnValue(scene.notifyRet));
        } else {
            MOCKER(RaCtxNotifyEvent).expects(never());
        }

        EXPECT_NO_THROW(handler->NotifyControlPlaneOnUbError(0, scene.rdmaHandle, errorMessage));
        GlobalMockObject::verify();
    }
}

// ==================== 方案 A P0：主打印函数 CaptureStdout 用例 ====================
#include "llt_hccl_stub_pub.h"

struct PreparedCommWithTask {
    CollCommAicpu* aicpuComm;
    std::shared_ptr<Hccl::StreamLite> streamLite;
    std::shared_ptr<MockThreadForReportErr> thread;
    Hccl::DfxDfxOpInfo opInfo;
    u32 sqeId;
};

static PreparedCommWithTask PrepareCommWithOneTask(u32 taskType, u32 sqId = 0, u32 taskId = 1)
{
    PreparedCommWithTask p;
    p.aicpuComm = new CollCommAicpu();
    p.aicpuComm->identifier_ = "test_print_group";
    InitCommEngineResMgr(*p.aicpuComm);

    p.streamLite = std::make_shared<Hccl::StreamLite>(0, 0, 0, 0);
    Hccl::DfxTaskInfo* slot = static_cast<Hccl::DfxTaskInfo*>(p.streamLite->taskInfos_.NextSlot());
    slot->taskId = (static_cast<u32>(taskId) << 16) | sqId;
    slot->sqId = sqId;
    slot->taskType = static_cast<u8>(taskType);
    slot->dfxOpInfo = reinterpret_cast<u64>(&p.opInfo);

    p.opInfo.opIndex = 100;
    p.opInfo.count = 1024;
    p.opInfo.srcAddr = 0xA000;
    p.opInfo.dstAddr = 0xB000;
    p.opInfo.srcSize = 512;
    p.opInfo.dstSize = 512;
    p.opInfo.dataType = 7;
    strncpy(p.opInfo.algTag, "test_alg_tag", sizeof(p.opInfo.algTag) - 1);

    p.thread = std::make_shared<MockThreadForReportErr>(p.streamLite.get());
    p.aicpuComm->GetCommEngineResMgr()->threadMgr_->threads_.push_back(p.thread);

    p.sqeId = (static_cast<u32>(taskId) << 16) | sqId;
    return p;
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_PrintTaskExceptionBySqeId_When_Normal_Expect_LogBaseGroupOpData)
{
    auto p = PrepareCommWithOneTask(static_cast<u32>(Hccl::TaskParamTypeVal::TASK_UB), 0, 1);

    log_level_set_stub(DLOG_INFO);
    testing::internal::CaptureStdout();
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().PrintTaskExceptionBySqeId(p.aicpuComm, 0, p.sqeId);
    std::string output = testing::internal::GetCapturedStdout();
    log_level_set_stub(DLOG_ERROR);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(output.find("[TaskException][AICPU]base information"), std::string::npos);
    EXPECT_NE(output.find("taskType"), std::string::npos);
    EXPECT_NE(output.find("group information"), std::string::npos);
    EXPECT_NE(output.find("group:[test_print_group]"), std::string::npos);
    EXPECT_NE(output.find("opData information"), std::string::npos);
    EXPECT_NE(output.find("opIndex[100]"), std::string::npos);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_PrintTaskExceptionBySqeId_When_NotifyWait_Expect_PrintTaskContext)
{
    auto p = PrepareCommWithOneTask(static_cast<u32>(Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT), 0, 1);

    log_level_set_stub(DLOG_INFO);
    testing::internal::CaptureStdout();
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().PrintTaskExceptionBySqeId(p.aicpuComm, 0, p.sqeId);
    std::string output = testing::internal::GetCapturedStdout();
    log_level_set_stub(DLOG_ERROR);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(output.find("[TaskException][AICPU]base information"), std::string::npos);
    EXPECT_NE(output.find("group information"), std::string::npos);
    // NOTIFY_WAIT 走 PrintTaskContextInfo，内部按 opIndex 分组时也打印 opData
    // 关键验证：NOTIFY_WAIT 应出现 task sequence
    EXPECT_NE(output.find("task sequence"), std::string::npos);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_PrintOpDataInfo_When_DfxOpInfoInvalid_Expect_Unavailable)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = DFX_INVALID_U64;

    log_level_set_stub(DLOG_INFO);
    testing::internal::CaptureStdout();
    HcclCommTaskExceptionLite::GetInstance().PrintOpDataInfo(&taskInfo);
    std::string output = testing::internal::GetCapturedStdout();
    log_level_set_stub(DLOG_ERROR);

    EXPECT_NE(output.find("opData information is (dfxOpInfo unavailable)"), std::string::npos);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_PrintOpDataInfo_When_Normal_Expect_LogKeyFields)
{
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.opIndex = 200;
    opInfo.count = 2048;
    opInfo.srcAddr = 0x1000;
    opInfo.dstAddr = 0x2000;
    opInfo.srcSize = 256;
    opInfo.dstSize = 256;
    opInfo.dataType = 3;
    strncpy(opInfo.algTag, "op_test_tag", sizeof(opInfo.algTag) - 1);

    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);

    log_level_set_stub(DLOG_INFO);
    testing::internal::CaptureStdout();
    HcclCommTaskExceptionLite::GetInstance().PrintOpDataInfo(&taskInfo);
    std::string output = testing::internal::GetCapturedStdout();
    log_level_set_stub(DLOG_ERROR);

    EXPECT_NE(output.find("opIndex[200]"), std::string::npos);
    EXPECT_NE(output.find("algTag[op_test_tag]"), std::string::npos);
    EXPECT_NE(output.find("count[2048]"), std::string::npos);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetParaInfo_When_UbDma_Expect_CorrectString)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB);
    taskInfo.linkType = static_cast<u8>(Hccl::DfxLinkType::UB);
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    taskInfo.taskPara.ubDma.srcAddr = 0x100;
    taskInfo.taskPara.ubDma.dstAddr = 0x200;
    taskInfo.taskPara.ubDma.size = 0x400;
    taskInfo.taskPara.ubDma.notifyId = 42;

    std::string result = HcclCommTaskExceptionLite::GetInstance().GetParaInfo(taskInfo);
    EXPECT_NE(result.find("src:[0x100]"), std::string::npos);
    EXPECT_NE(result.find("notify id:[42]"), std::string::npos);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetParaInfo_When_Reduce_Expect_CorrectString)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB_REDUCE_INLINE);
    taskInfo.linkType = static_cast<u8>(Hccl::DfxLinkType::UB);
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    taskInfo.taskPara.Reduce.srcAddr = 0x300;
    taskInfo.taskPara.Reduce.dstAddr = 0x400;
    taskInfo.taskPara.Reduce.notifyId = 99;
    taskInfo.taskPara.Reduce.reduceOp = 1;

    std::string result = HcclCommTaskExceptionLite::GetInstance().GetParaInfo(taskInfo);
    EXPECT_NE(result.find("src:[0x300]"), std::string::npos);
    EXPECT_NE(result.find("notify id:[99]"), std::string::npos);
    EXPECT_NE(result.find("op:[1]"), std::string::npos);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetParaInfo_When_Sdma_Expect_CorrectString)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA);
    taskInfo.linkType = static_cast<u8>(Hccl::DfxLinkType::HCCS);
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    taskInfo.taskPara.Dma.srcAddr = 0x600;
    taskInfo.taskPara.Dma.dstAddr = 0x700;
    taskInfo.taskPara.Dma.size = 0x800;

    std::string result = HcclCommTaskExceptionLite::GetInstance().GetParaInfo(taskInfo);
    EXPECT_NE(result.find("src:[0x600]"), std::string::npos);
    EXPECT_EQ(result.find("notify id"), std::string::npos);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetParaInfo_When_NotifyWait_Expect_CorrectString)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT);
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    taskInfo.taskPara.Notify.notifyId = 7;

    std::string result = HcclCommTaskExceptionLite::GetInstance().GetParaInfo(taskInfo);
    EXPECT_NE(result.find("notify id:[7]"), std::string::npos);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetParaInfo_When_Default_Expect_NotMatch)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = 0xFF;
    taskInfo.dfxOpInfo = DFX_INVALID_U64;

    std::string result = HcclCommTaskExceptionLite::GetInstance().GetParaInfo(taskInfo);
    EXPECT_NE(result.find("not match"), std::string::npos);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetEidFromChannelHandle_When_Invalid_Expect_ErrorLog)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.channelHandle = DFX_INVALID_U64;
    taskInfo.sqId = 5;
    taskInfo.taskId = 10;
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB);

    Hccl::Eid locEid{};
    Hccl::Eid rmtEid{};

    log_level_set_stub(DLOG_INFO);
    testing::internal::CaptureStdout();
    HcclCommTaskExceptionLite::GetInstance().GetEidFromChannelHandle(taskInfo, locEid, rmtEid);
    std::string output = testing::internal::GetCapturedStdout();
    log_level_set_stub(DLOG_ERROR);

    EXPECT_NE(output.find("channelHandle is invalid"), std::string::npos);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_StopCall_When_True_Expect_CallReturnImmediately)
{
    HcclCommTaskExceptionLite::GetInstance().stopCall_ = true;

    log_level_set_stub(DLOG_INFO);
    testing::internal::CaptureStdout();
    HcclCommTaskExceptionLite::GetInstance().Call();
    std::string output = testing::internal::GetCapturedStdout();
    log_level_set_stub(DLOG_ERROR);

    EXPECT_TRUE(output.empty());
    HcclCommTaskExceptionLite::GetInstance().stopCall_ = false;
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_ThreadsPrinted_When_SameSqeId_Expect_SkipSecond)
{
    auto p = PrepareCommWithOneTask(static_cast<u32>(Hccl::TaskParamTypeVal::TASK_UB), 0, 1);

    log_level_set_stub(DLOG_INFO);
    testing::internal::CaptureStdout();
    HcclCommTaskExceptionLite::GetInstance().PrintTaskExceptionBySqeId(p.aicpuComm, 0, p.sqeId);
    std::string output1 = testing::internal::GetCapturedStdout();

    testing::internal::CaptureStdout();
    HcclCommTaskExceptionLite::GetInstance().PrintTaskExceptionBySqeId(p.aicpuComm, 0, p.sqeId);
    std::string output2 = testing::internal::GetCapturedStdout();
    log_level_set_stub(DLOG_ERROR);

    EXPECT_NE(output1.find("base information"), std::string::npos);
    EXPECT_NE(output2.find("has been printed, skip"), std::string::npos);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetRemoteRankId_When_DfxOpInfoInvalid_Expect_ErrorLog)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    taskInfo.sqId = 3;
    taskInfo.taskId = 7;

    log_level_set_stub(DLOG_INFO);
    testing::internal::CaptureStdout();
    u32 rank = HcclCommTaskExceptionLite::GetInstance().GetRemoteRankId(taskInfo);
    std::string output = testing::internal::GetCapturedStdout();
    log_level_set_stub(DLOG_ERROR);

    EXPECT_EQ(rank, Hccl::DFX_INVALID_RANKID);
    EXPECT_NE(output.find("is invalid"), std::string::npos);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetOpIndex_When_Nullptr_Expect_ErrorLog)
{
    log_level_set_stub(DLOG_INFO);
    testing::internal::CaptureStdout();
    u32 idx = HcclCommTaskExceptionLite::GetInstance().GetOpIndex(nullptr);
    std::string output = testing::internal::GetCapturedStdout();
    log_level_set_stub(DLOG_ERROR);

    EXPECT_EQ(idx, UINT32_MAX);
    EXPECT_NE(output.find("taskInfo is nullptr"), std::string::npos);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetOpIndex_When_DfxOpInfoInvalid_Expect_ErrorLog)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    taskInfo.sqId = 1;
    taskInfo.taskId = 2;

    log_level_set_stub(DLOG_INFO);
    testing::internal::CaptureStdout();
    u32 idx = HcclCommTaskExceptionLite::GetInstance().GetOpIndex(&taskInfo);
    std::string output = testing::internal::GetCapturedStdout();
    log_level_set_stub(DLOG_ERROR);

    EXPECT_EQ(idx, UINT32_MAX);
    EXPECT_NE(output.find("is invalid"), std::string::npos);
}

// ==================== 方案 C：关键函数双重验证用例 ====================

TEST_F(hcclCommTaskExceptionLiteTest, UtC_FillReduceErrMsg_When_Normal_Expect_AllFieldsCorrect)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB_REDUCE_INLINE);
    taskInfo.linkType = static_cast<u8>(Hccl::DfxLinkType::UB);
    taskInfo.taskPara.Reduce.reduceOp = 1;
    taskInfo.taskPara.Reduce.notifyId = 42;
    taskInfo.taskPara.Reduce.size = 4096;
    taskInfo.taskPara.Reduce.srcAddr = 0x1000;
    taskInfo.taskPara.Reduce.dstAddr = 0x2000;
    taskInfo.channelHandle = DFX_INVALID_U64;

    rtLogicCqReport_t exceptionInfo{};
    exceptionInfo.errorCode = 0x102;

    Hccl::ErrorMessageReport errMsg{};
    errMsg.taskType = Hccl::TaskParamType(static_cast<Hccl::TaskParamType::Value>(taskInfo.taskType));
    HcclCommTaskExceptionLite::GetInstance().GenerateTaskErrMsg(taskInfo, errMsg, exceptionInfo);

    EXPECT_EQ(errMsg.reduceType, 1u);
    EXPECT_EQ(errMsg.notifyId, 42u);
    EXPECT_EQ(errMsg.notifyValue, INVALID_U32);
    EXPECT_EQ(errMsg.ubCqeStatus, 0x02u);
    EXPECT_EQ(errMsg.size, 4096u);
    EXPECT_EQ(errMsg.taskSrcAddr, 0x1000u);
    EXPECT_EQ(errMsg.taskDstAddr, 0x2000u);
}

TEST_F(hcclCommTaskExceptionLiteTest, UtC_FillSdmaErrMsg_When_Normal_Expect_AllFieldsCorrect)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA);
    taskInfo.linkType = static_cast<u8>(Hccl::DfxLinkType::HCCS);
    taskInfo.taskPara.Dma.srcAddr = 0x5000;
    taskInfo.taskPara.Dma.dstAddr = 0x6000;
    taskInfo.taskPara.Dma.size = 8192;

    Hccl::ErrorMessageReport errMsg{};
    HcclCommTaskExceptionLite::GetInstance().GenerateTaskErrMsg(taskInfo, errMsg, rtLogicCqReport_t{});

    EXPECT_EQ(errMsg.taskSrcAddr, 0x5000u);
    EXPECT_EQ(errMsg.taskDstAddr, 0x6000u);
    EXPECT_EQ(errMsg.size, 8192u);
}

TEST_F(hcclCommTaskExceptionLiteTest, UtC_FillNotifyErrMsg_When_Normal_Expect_AllFieldsCorrect)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT);
    taskInfo.taskPara.Notify.notifyId = 88;

    Hccl::ErrorMessageReport errMsg{};
    HcclCommTaskExceptionLite::GetInstance().GenerateTaskErrMsg(taskInfo, errMsg, rtLogicCqReport_t{});

    EXPECT_EQ(errMsg.notifyId, 88u);
    EXPECT_EQ(errMsg.notifyValue, 1u);
}

// 方案 C：GenerateErrorMessageReport — TASK_UB 走 FillUbErrMsg（不设 notifyId/notifyValue）
TEST_F(hcclCommTaskExceptionLiteTest, UtC_GenerateErrorMessageReport_When_TaskUb_Expect_FillUbErrMsgFields)
{
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.opIndex = 100;
    opInfo.count = 1024;
    opInfo.srcAddr = 0xA000;
    opInfo.dstAddr = 0xB000;
    strncpy(opInfo.algTag, "ub_tag", sizeof(opInfo.algTag) - 1);

    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    taskInfo.sqId = 10;
    taskInfo.taskId = 20;
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB);
    taskInfo.linkType = static_cast<u8>(Hccl::DfxLinkType::UB);
    taskInfo.taskPara.ubDma.jettyHandle = 0x1234;
    taskInfo.taskPara.ubDma.jettyId = 7;
    taskInfo.taskPara.ubDma.tpn = 3;
    taskInfo.taskPara.ubDma.notifyId = 55;
    taskInfo.taskPara.ubDma.srcAddr = 0xC000;
    taskInfo.taskPara.ubDma.dstAddr = 0xD000;
    taskInfo.taskPara.ubDma.size = 512;
    taskInfo.channelHandle = DFX_INVALID_U64;

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_group";
    InitCommEngineResMgr(aicpuComm);

    rtLogicCqReport_t exceptionInfo{};
    exceptionInfo.errorType = 3;
    exceptionInfo.errorCode = 0x205;

    Hccl::ErrorMessageReport errMsg{};
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().GenerateErrorMessageReport(
        &aicpuComm, taskInfo, exceptionInfo, errMsg);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // FillUbErrMsg 填充的字段
    EXPECT_EQ(errMsg.ubCqeStatus, 0x05u);
    EXPECT_EQ(errMsg.size, 512u);
    EXPECT_EQ(errMsg.taskSrcAddr, 0xC000u);
    EXPECT_EQ(errMsg.taskDstAddr, 0xD000u);
    // FillUbErrMsg 不设 notifyId/notifyValue（保持默认值 0）
    EXPECT_EQ(errMsg.notifyId, 0u);
    EXPECT_EQ(errMsg.notifyValue, 0u);
    // 公共字段
    EXPECT_EQ(errMsg.streamId, 10);
    EXPECT_EQ(errMsg.opIndex, 100u);
    EXPECT_EQ(errMsg.jettyHandle, 0x1234u);
    EXPECT_STREQ(errMsg.tag, "ub_tag");
    EXPECT_EQ(errMsg.tag[strlen("ub_tag")], '\0');
    EXPECT_STREQ(errMsg.group, "test_group");
}

// 方案 C：GenerateErrorMessageReport — TASK_UB_INLINE_WRITE 走 FillDmaErrMsg（设 notifyId/notifyValue）
TEST_F(hcclCommTaskExceptionLiteTest, UtC_GenerateErrorMessageReport_When_TaskUbInlineWrite_Expect_FillDmaErrMsgFields)
{
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.opIndex = 100;
    opInfo.count = 1024;
    opInfo.srcAddr = 0xA000;
    opInfo.dstAddr = 0xB000;
    strncpy(opInfo.algTag, "dma_tag", sizeof(opInfo.algTag) - 1);

    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    taskInfo.sqId = 10;
    taskInfo.taskId = 20;
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB_INLINE_WRITE);
    taskInfo.linkType = static_cast<u8>(Hccl::DfxLinkType::UB);
    taskInfo.taskPara.ubDma.jettyHandle = 0x1234;
    taskInfo.taskPara.ubDma.jettyId = 7;
    taskInfo.taskPara.ubDma.tpn = 3;
    taskInfo.taskPara.ubDma.notifyId = 55;
    taskInfo.taskPara.ubDma.srcAddr = 0xC000;
    taskInfo.taskPara.ubDma.dstAddr = 0xD000;
    taskInfo.taskPara.ubDma.size = 512;
    taskInfo.channelHandle = DFX_INVALID_U64;

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_group";
    InitCommEngineResMgr(aicpuComm);

    rtLogicCqReport_t exceptionInfo{};
    exceptionInfo.errorType = 3;
    exceptionInfo.errorCode = 0x205;

    Hccl::ErrorMessageReport errMsg{};
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().GenerateErrorMessageReport(
        &aicpuComm, taskInfo, exceptionInfo, errMsg);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // FillDmaErrMsg 设置 notifyId/notifyValue（与 FillUbErrMsg 的关键差异）
    EXPECT_EQ(errMsg.notifyId, 55u);
    EXPECT_EQ(errMsg.notifyValue, 1u);
    // FillUbErrMsg 也被调用（FillDmaErrMsg 内部调 FillUbErrMsg）
    EXPECT_EQ(errMsg.ubCqeStatus, 0x05u);
    EXPECT_EQ(errMsg.size, 512u);
    EXPECT_EQ(errMsg.taskSrcAddr, 0xC000u);
    EXPECT_EQ(errMsg.taskDstAddr, 0xD000u);
    EXPECT_STREQ(errMsg.tag, "dma_tag");
    EXPECT_EQ(errMsg.tag[strlen("dma_tag")], '\0');
    EXPECT_STREQ(errMsg.group, "test_group");
}

TEST_F(hcclCommTaskExceptionLiteTest, UtC_GenerateErrorMessageReport_When_DfxOpInfoInvalid_Expect_ReturnPtrError)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_gen_err";
    InitCommEngineResMgr(aicpuComm);
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    rtLogicCqReport_t exceptionInfo{};
    Hccl::ErrorMessageReport errMsg{};
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().GenerateErrorMessageReport(
        &aicpuComm, taskInfo, exceptionInfo, errMsg);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// ==================== 端到端用例 ====================

TEST_F(hcclCommTaskExceptionLiteTest, UtE2E_CollectTaskContext_When_TaskAtBegin_Expect_FirstTaskCollected)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_e2e_begin";
    InitCommEngineResMgr(aicpuComm);

    auto streamLite = std::make_shared<Hccl::StreamLite>(0, 0, 0, 0);
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.opIndex = 1;
    Hccl::DfxTaskInfo* slot = static_cast<Hccl::DfxTaskInfo*>(streamLite->taskInfos_.NextSlot());
    ASSERT_NE(slot, nullptr);
    slot->taskId = (1U << 16) | 0U;
    slot->taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT);
    slot->dfxOpInfo = reinterpret_cast<u64>(&opInfo);

    auto thread = std::make_shared<MockThreadForReportErr>(streamLite.get());
    aicpuComm.GetCommEngineResMgr()->threadMgr_->threads_.push_back(thread);

    std::vector<Hccl::DfxTaskInfo*> taskContext;
    HcclResult ret
        = HcclCommTaskExceptionLite::GetInstance().CollectTaskContext(&aicpuComm, 0, (1U << 16) | 0U, taskContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(taskContext.size(), 1u);
}

TEST_F(hcclCommTaskExceptionLiteTest, UtE2E_CollectTaskContext_When_55Tasks_Expect_Max50Collected)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_e2e_55";
    InitCommEngineResMgr(aicpuComm);

    auto streamLite = std::make_shared<Hccl::StreamLite>(0, 0, 0, 0);
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.opIndex = 1;

    for (u32 i = 0; i < 55; i++) {
        Hccl::DfxTaskInfo* slot = static_cast<Hccl::DfxTaskInfo*>(streamLite->taskInfos_.NextSlot());
        ASSERT_NE(slot, nullptr);
        slot->taskId = ((i + 1U) << 16) | 0U;
        slot->taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT);
        slot->dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    }

    auto thread = std::make_shared<MockThreadForReportErr>(streamLite.get());
    aicpuComm.GetCommEngineResMgr()->threadMgr_->threads_.push_back(thread);

    u32 exceptionTaskId = (55U << 16) | 0U;
    std::vector<Hccl::DfxTaskInfo*> taskContext;
    HcclResult ret
        = HcclCommTaskExceptionLite::GetInstance().CollectTaskContext(&aicpuComm, 0, exceptionTaskId, taskContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_LE(taskContext.size(), 50u);
}

TEST_F(hcclCommTaskExceptionLiteTest, UtE2E_CollectTaskContext_When_QueueOverflow_Expect_Max50Collected)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_e2e_overflow";
    InitCommEngineResMgr(aicpuComm);

    auto streamLite = std::make_shared<Hccl::StreamLite>(0, 0, 0, 0);
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.opIndex = 1;

    constexpr u32 TOTAL_TASKS = Hccl::DFX_TASK_INFO_QUEUE_CAPACITY + 10;
    for (u32 i = 0; i < TOTAL_TASKS; i++) {
        Hccl::DfxTaskInfo* slot = static_cast<Hccl::DfxTaskInfo*>(streamLite->taskInfos_.NextSlot());
        ASSERT_NE(slot, nullptr);
        slot->taskId = ((i + 1U) << 16) | 0U;
        slot->taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT);
        slot->dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    }

    auto thread = std::make_shared<MockThreadForReportErr>(streamLite.get());
    aicpuComm.GetCommEngineResMgr()->threadMgr_->threads_.push_back(thread);

    u32 exceptionTaskId = (TOTAL_TASKS << 16) | 0U;
    std::vector<Hccl::DfxTaskInfo*> taskContext;
    HcclResult ret
        = HcclCommTaskExceptionLite::GetInstance().CollectTaskContext(&aicpuComm, 0, exceptionTaskId, taskContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_LE(taskContext.size(), 50u);
}

// ==================== Host 侧用例 ====================

// L-33: ProcessHcclTaskException comm 未注册 — 直接调用 ProcessException（跳过 FindTaskInfo），
// isIndop_=true，comm_=nullptr → 验证 "not exist" 安全返回路径（PR #3153 修复点）
// 因 stub 中 FindTaskInfo 默认返回 HCCL_E_NOT_FOUND 无法到达 comm 检查路径，
// 改为直接调用 ProcessException 中的 HandleHostErrorReport 路径验证 comm 未注册安全返回
TEST_F(hcclCommTaskExceptionLiteTest, Ut_ProcessHcclTaskException_When_CommNotRegistered_Expect_SafeReturn)
{
    const s32 testDeviceId = 57;
    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);
    handler->CommRegisterMap_.clear();
    handler->hasAicpuReport_ = false;

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<uint32_t>(testDeviceId);
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 1;

    // 构造 TaskInfo：isIndop_=true，comm_=nullptr
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    Hccl::TaskInfo taskInfo(0, 1, 1, taskParam, nullptr);
    taskInfo.dfxOpInfo_ = std::make_shared<Hccl::DfxOpInfo>();
    taskInfo.dfxOpInfo_->isIndop_ = true;
    taskInfo.dfxOpInfo_->comm_ = nullptr; // comm_ 为 nullptr

    // mock GetAicpuTaskException 返回 tag 为空（走 HandleHostErrorReport 路径）
    Hccl::ErrorMessageReport mockErrMsg{};
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).stubs().will(returnValue(mockErrMsg));
    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    log_level_set_stub(DLOG_INFO);
    testing::internal::CaptureStdout();
    // 直接调用 ProcessException（跳过 FindTaskInfo）
    EXPECT_NO_THROW(handler->ProcessException(&exceptionInfo, taskInfo));
    std::string output = testing::internal::GetCapturedStdout();
    log_level_set_stub(DLOG_ERROR);

    // PR #3153 修复点：验证安全返回不 core dump
    SUCCEED(); // 到这里就说明没 core dump

    GlobalMockObject::verify();
}

// 补充：FindTaskInfo 未找到时安全返回（stub 默认行为）
TEST_F(hcclCommTaskExceptionLiteTest, Ut_ProcessHcclTaskException_When_TaskNotFound_Expect_SafeReturn)
{
    const s32 testDeviceId = 50;
    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<uint32_t>(testDeviceId);
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 999;

    // stub 中 FindTaskInfo 默认返回 HCCL_E_NOT_FOUND
    log_level_set_stub(DLOG_INFO);
    testing::internal::CaptureStdout();
    EXPECT_NO_THROW(handler->ProcessHcclTaskException(&exceptionInfo));
    std::string output = testing::internal::GetCapturedStdout();
    log_level_set_stub(DLOG_ERROR);

    EXPECT_NE(output.find("not found"), std::string::npos);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_Register_When_FirstComm_Expect_InsertIntoMap)
{
    const s32 testDeviceId = 56;
    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);
    handler->CommRegisterMap_.clear();

    HcclResult ret = handler->Register(0xABCD);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(handler->CommRegisterMap_.count(0xABCD), 1u);

    handler->CommRegisterMap_.clear();
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_UnRegister_When_Registered_Expect_RemoveFromMap)
{
    const s32 testDeviceId = 55;
    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);
    handler->CommRegisterMap_.clear();
    handler->CommRegisterMap_.insert(0x1234);

    HcclResult ret = handler->UnRegister(0x1234);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(handler->CommRegisterMap_.count(0x1234), 0u);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_UnRegister_When_NotRegistered_Expect_SafeReturn)
{
    const s32 testDeviceId = 54;
    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);
    handler->CommRegisterMap_.clear();

    HcclResult ret = handler->UnRegister(0xFFFF);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, UtC_ProcessException_When_AicpuReported_Expect_HasAicpuReportSet)
{
    const s32 testDeviceId = 53;
    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);
    handler->hasAicpuReport_ = false;

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    Hccl::TaskInfo taskInfo(0, 0, 1, taskParam, nullptr);
    taskInfo.dfxOpInfo_ = std::make_shared<Hccl::DfxOpInfo>();
    hccl::CollComm collComm(nullptr, 0, "test_group", {});
    taskInfo.dfxOpInfo_->comm_ = &collComm;

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<uint32_t>(testDeviceId);

    Hccl::ErrorMessageReport mockErrMsg{};
    strncpy(mockErrMsg.tag, "aicpu_reported", sizeof(mockErrMsg.tag) - 1);
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).stubs().will(returnValue(mockErrMsg));
    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    handler->ProcessException(&exceptionInfo, taskInfo);

    EXPECT_TRUE(handler->hasAicpuReport_);
    GlobalMockObject::verify();
}

TEST_F(hcclCommTaskExceptionLiteTest, UtC_ProcessException_When_HostDetected_Expect_HasAicpuReportUnset)
{
    const s32 testDeviceId = 52;
    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);
    handler->hasAicpuReport_ = false;

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    Hccl::TaskInfo taskInfo(0, 0, 1, taskParam, nullptr);
    taskInfo.dfxOpInfo_ = std::make_shared<Hccl::DfxOpInfo>();
    hccl::CollComm collComm(nullptr, 0, "test_group", {});
    taskInfo.dfxOpInfo_->comm_ = &collComm;

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<uint32_t>(testDeviceId);

    Hccl::ErrorMessageReport mockErrMsg{};
    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).stubs().will(returnValue(mockErrMsg));
    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    handler->ProcessException(&exceptionInfo, taskInfo);

    EXPECT_FALSE(handler->hasAicpuReport_);
    GlobalMockObject::verify();
}

TEST_F(hcclCommTaskExceptionLiteTest, UtC_ProcessException_When_SecondCall_Expect_SkipGetAicpuTaskException)
{
    const s32 testDeviceId = 51;
    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);
    handler->hasAicpuReport_ = true;

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    Hccl::TaskInfo taskInfo(0, 0, 1, taskParam, nullptr);
    taskInfo.dfxOpInfo_ = std::make_shared<Hccl::DfxOpInfo>();
    hccl::CollComm collComm(nullptr, 0, "test_group", {});
    taskInfo.dfxOpInfo_->comm_ = &collComm;

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<uint32_t>(testDeviceId);

    MOCKER_CPP(&hccl::CollComm::GetAicpuTaskException).expects(never());
    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    handler->ProcessException(&exceptionInfo, taskInfo);
    GlobalMockObject::verify();
}

// ==================== remoteRankId / opIndex 字段值正确性验证 ====================

// GetRemoteRankId — channelHandle 为 DFX_INVALID_U64 时返回 DFX_INVALID_RANKID
TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetRemoteRankId_When_HandleInvalid_Expect_ReturnInvalidRankId)
{
    hccl::HcclCommDfxLite dfxLite;
    dfxLite.Init(0, "test_comm", 4, 0);

    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.hcclCommDfxLite = &dfxLite;

    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    taskInfo.channelHandle = DFX_INVALID_U64;

    u32 rank = HcclCommTaskExceptionLite::GetInstance().GetRemoteRankId(taskInfo);
    EXPECT_EQ(rank, Hccl::DFX_INVALID_RANKID);
}

// GetRemoteRankId — channelHandle 未注册时返回 DFX_INVALID_RANKID
TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetRemoteRankId_When_HandleNotFound_Expect_ReturnInvalidRankId)
{
    hccl::HcclCommDfxLite dfxLite;
    dfxLite.Init(0, "test_comm", 4, 0);

    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.hcclCommDfxLite = &dfxLite;

    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    taskInfo.channelHandle = 0x9999; // 未注册的 handle

    u32 rank = HcclCommTaskExceptionLite::GetInstance().GetRemoteRankId(taskInfo);
    EXPECT_EQ(rank, Hccl::DFX_INVALID_RANKID);
}

// GetRemoteRankId — 正常路径返回正确 rankId（方案 C 字段值断言）
TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetRemoteRankId_When_Normal_Expect_ReturnCorrectRankId)
{
    hccl::HcclCommDfxLite dfxLite;
    dfxLite.Init(0, "test_comm", 4, 0);

    // 注册 channelHandle → remoteRankId 映射
    const u64 testHandle = 0x9527;
    const u32 expectedRank = 3;
    dfxLite.AddChannelRemoteRankId(testHandle, expectedRank);

    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.hcclCommDfxLite = &dfxLite;

    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    taskInfo.channelHandle = testHandle;

    u32 rank = HcclCommTaskExceptionLite::GetInstance().GetRemoteRankId(taskInfo);
    EXPECT_EQ(rank, expectedRank); // 方案 C：断言返回值等于注册的 rankId
}

// GetOpIndex — 正常路径返回正确 opIndex（方案 C 字段值断言）
TEST_F(hcclCommTaskExceptionLiteTest, UtC_GetOpIndex_When_Normal_Expect_ReturnCorrectOpIndex)
{
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.opIndex = 200;

    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    taskInfo.sqId = 1;
    taskInfo.taskId = 2;

    u32 idx = HcclCommTaskExceptionLite::GetInstance().GetOpIndex(&taskInfo);
    EXPECT_EQ(idx, 200u); // 方案 C：断言返回值等于设置的 opIndex
}
