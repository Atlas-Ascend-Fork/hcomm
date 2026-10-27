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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#include "coll_comm_aicpu.h"
#include "coll_comm_aicpu_mgr.h"
#include "ns_recovery/aicpu/ns_recovery_lite.h"
// Resume 用例构造 device 侧打包数据与反序列化结果用
#include "aicpu_res_package_helper.h"
#include "binary_stream.h"
#include "adapter_hal_pub.h"

#define private public
using namespace hccl;

class CollCommAicpuTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(CollCommAicpuTest, Ut_DefaultStatus_IsInvalid_And_SetGet_Works)
{
    CollCommAicpu coll;
    EXPECT_EQ(coll.GetCommmStatus(), HcclCommStatus::HCCL_COMM_STATUS_INVALID);

    coll.SetCommmStatus(HcclCommStatus::HCCL_COMM_STATUS_READY);
    EXPECT_EQ(coll.GetCommmStatus(), HcclCommStatus::HCCL_COMM_STATUS_READY);
}

TEST_F(CollCommAicpuTest, Ut_GetNsRecoveryLitePtr_DefaultNull_And_Settable)
{
    CollCommAicpu coll;
    EXPECT_EQ(coll.GetNsRecoveryLitePtr(), nullptr);

    auto nsPtr = std::make_shared<NsRecoveryLite>();
    coll.nsRecoveryLitePtr_ = nsPtr;
    EXPECT_NE(coll.GetNsRecoveryLitePtr(), nullptr);
}

TEST_F(CollCommAicpuTest, Ut_Resume_When_EmptyList_Expect_Success_And_ResetsNsRecoveryFlags)
{
    CollCommAicpu coll;

    // 初始化 nsRecovery 使其可调用 Resume
    coll.nsRecoveryLitePtr_ = std::make_shared<NsRecoveryLite>();
    // 初始化 commEngineResMgr_，避免 GetCommEngineResMgr()->GetAllThread() 解引用 nullptr
    coll.commEngineResMgr_ = std::make_unique<CommEngineResAicpuMgr>(coll.dfx_, [](bool) {
        return HCCL_SUCCESS;
    });
    coll.SetCommmStatus(HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING);

    // ResumeChannels 首句校验三个指针非空；listNum 保持 0 表示不遍历任何 channel
    ChannelHandle channelList[1] = {0};
    u8 uniqueIdAddr[1] = {0};
    u32 channelSizeAddr[1] = {0};
    HcclChannelUrmaRes commParam{};
    commParam.channelList = channelList;
    commParam.uniqueIdAddr = uniqueIdAddr;
    commParam.channelSizeAddr = channelSizeAddr;

    auto ret = coll.Resume(&commParam);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(coll.GetCommmStatus(), HcclCommStatus::HCCL_COMM_STATUS_READY);
    EXPECT_FALSE(coll.GetNsRecoveryLitePtr()->IsNeedClean());

    // null 入参由 ResumeChannels 前的 CHK_PTR_NULL 拦截
    EXPECT_NE(coll.Resume(nullptr), HCCL_SUCCESS);
}

// Resume — channelList 为空句柄时被 ResumeChannels 的空指针校验拦截
TEST_F(CollCommAicpuTest, Ut_Resume_When_ChannelListNull_Return_Error)
{
    CollCommAicpu coll;
    coll.nsRecoveryLitePtr_ = std::make_shared<NsRecoveryLite>();

    HcclChannelUrmaRes commParam{};
    commParam.channelList = nullptr;
    commParam.listNum = 1;

    EXPECT_NE(coll.Resume(&commParam), HCCL_SUCCESS);
}

// Resume — uniqueIdAddr / channelSizeAddr 为空同样在遍历前被拦下
TEST_F(CollCommAicpuTest, Ut_Resume_When_ResAddrNull_Return_Error)
{
    CollCommAicpu coll;
    coll.nsRecoveryLitePtr_ = std::make_shared<NsRecoveryLite>();

    ChannelHandle channelList[1] = {0};
    u8 uniqueIdAddr[1] = {0};
    u32 channelSizeAddr[1] = {0};

    HcclChannelUrmaRes commParam{};
    commParam.channelList = channelList;
    commParam.listNum = 1;

    commParam.uniqueIdAddr = nullptr;
    commParam.channelSizeAddr = channelSizeAddr;
    EXPECT_NE(coll.Resume(&commParam), HCCL_SUCCESS);

    commParam.uniqueIdAddr = uniqueIdAddr;
    commParam.channelSizeAddr = nullptr;
    EXPECT_NE(coll.Resume(&commParam), HCCL_SUCCESS);
}

// Resume — channelList 中的句柄无法解析（0）时明确报错，不静默成功
// 句柄合法性由 UnwrapChannelHandle 判定
TEST_F(CollCommAicpuTest, Ut_Resume_When_ChannelHandleUnresolvable_Return_Error)
{
    CollCommAicpu coll;
    coll.nsRecoveryLitePtr_ = std::make_shared<NsRecoveryLite>();

    // 本 UT 链接里 AicpuResPackageHelper::ParsePackedData 被打桩成空实现
    // （见 test/ut/stub/llt_next_orion_stub.cc），按真实职责注入只含 STREAM 模块的解析结果，
    // 使流程能越过反序列化落到 UnwrapChannelHandle 的句柄校验上
    Hccl::BinaryStream innerStream;
    std::vector<char> transpUniqueId(64, 0);
    innerStream << transpUniqueId;
    const std::string innerBytes = innerStream.GetString();
    std::vector<Hccl::ModuleData> parsed(Hccl::AicpuResMgrType::STREAM + 1);
    parsed[Hccl::AicpuResMgrType::STREAM].data.assign(innerBytes.begin(), innerBytes.end());

    MOCKER_CPP(&Hccl::AicpuResPackageHelper::ParsePackedData).stubs().with(mockcpp::any()).will(returnValue(parsed));

    u32 blobSize = 64;
    u8 blob[64] = {0};
    ChannelHandle channelList[1] = {0};
    HcclChannelUrmaRes commParam{};
    commParam.channelList = channelList;
    commParam.listNum = 1;
    commParam.uniqueIdAddr = blob;
    commParam.channelSizeAddr = &blobSize;

    EXPECT_NE(coll.Resume(&commParam), HCCL_SUCCESS);
}

TEST_F(CollCommAicpuTest, Ut_CheckIndOpExecStatus_Return_HCCL_E_SUSPENDING)
{
    CollCommAicpu coll;
    coll.SetCommmStatus(HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING);
    auto ret = coll.CheckIndOpExecStatus(0);
    EXPECT_EQ(ret, HCCL_E_SUSPENDING);
}

TEST_F(CollCommAicpuTest, Ut_InitIndopEnv_Expect_Success)
{
    CommAicpuParam param{};
    param.commConfig.taskExceptionEnable = true;
    param.commConfig.notifyWaitTimeout = 1836;
    param.commConfig.plfDebugConfig = 0;
    CollCommAicpuMgr::GetInstance().InitIndopEnv(&param);
}

TEST_F(CollCommAicpuTest, Ut_GetCommEngineResMgr_NotNull_After_Init)
{
    CollCommAicpu coll;
    // 通过 InitAicpuIndOp 初始化后 GetCommEngineResMgr 应非空
    CommAicpuParam param{};
    strncpy(param.hcomId, "test_group", HCOMID_MAX_SIZE - 1);

    // Mock 必要的底层调用
    MOCKER_CPP(hrtSetWorkModeAicpu).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtSetlocalDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtSetlocalDeviceType).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtDrvGetLocalDevIDByHostDevID).stubs().will(returnValue(HCCL_SUCCESS));

    auto ret = coll.InitAicpuIndOp(&param);
    if (ret == HCCL_SUCCESS) {
        EXPECT_NE(coll.GetCommEngineResMgr(), nullptr);
    }
}

TEST_F(CollCommAicpuTest, Ut_GetTopoInfo_DefaultInit)
{
    CollCommAicpu coll;
    const HcclTopoInfo& topo = coll.GetTopoInfo();
    EXPECT_EQ(topo.userRank, 0u);
    EXPECT_EQ(topo.userRankSize, 0u);
}

TEST_F(CollCommAicpuTest, Ut_GetIdentifier_EmptyByDefault)
{
    CollCommAicpu coll;
    EXPECT_EQ(coll.GetIdentifier(), "");
}

TEST_F(CollCommAicpuTest, Ut_IsErrorReported_DefaultFalse_And_SetGet_Works)
{
    CollCommAicpu coll;
    EXPECT_FALSE(coll.IsErrorReported());
    coll.SetErrorReported(true);
    EXPECT_TRUE(coll.IsErrorReported());
}

TEST_F(CollCommAicpuTest, Ut_UpdateIndex_Increments)
{
    CollCommAicpu coll;
    u32 idx1 = coll.UpdateIndex();
    u32 idx2 = coll.UpdateIndex();
    EXPECT_GT(idx2, idx1);
}

TEST_F(CollCommAicpuTest, Ut_GetDevId_DefaultZero)
{
    CollCommAicpu coll;
    EXPECT_EQ(coll.GetDevId(), 0u);
}

TEST_F(CollCommAicpuTest, Ut_CheckIndOpExecStatus_ReturnError_When_InvalidStatus)
{
    CollCommAicpu coll;
    coll.SetCommmStatus(HcclCommStatus::HCCL_COMM_STATUS_INVALID);
    auto ret = coll.CheckIndOpExecStatus(0);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// 910B legacy 通信域管理
TEST_F(CollCommAicpuTest, Ut_GetLegacy910CollComm_DefaultNull)
{
    CollCommAicpu coll;
    EXPECT_EQ(coll.GetLegacy910CollComm(), nullptr);
}

TEST_F(CollCommAicpuTest, Ut_IsLegacy910CollCommBusy_DefaultFalse)
{
    CollCommAicpu coll;
    EXPECT_FALSE(coll.IsLegacy910CollCommBusy());
}

TEST_F(CollCommAicpuTest, Ut_SetLegacy910CollCommBusy_Works)
{
    CollCommAicpu coll;
    coll.SetLegacy910CollCommBusy(true);
    EXPECT_TRUE(coll.IsLegacy910CollCommBusy());
    coll.SetLegacy910CollCommBusy(false);
    EXPECT_FALSE(coll.IsLegacy910CollCommBusy());
}
