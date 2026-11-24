/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"
#include "coll_comm_mgr.h"

extern thread_local s32 g_hcclDeviceId;

HcclResult HcclSetConfigV2(HcclConfig config, HcclConfigValue configValue)
{
    (void)config;
    (void)configValue;
    return HCCL_SUCCESS;
}

static HcclResult HrtGetHcclV2SupportForUt(bool* isSupport)
{
    *isSupport = true;
    return HCCL_SUCCESS;
}

class OpBaseMiscTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();

        UT_USE_1SERVER_1RANK_AS_DEFAULT;
        // 将enableEntryLog默认返回为true
        MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
    }

    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(OpBaseMiscTest, Ut_HcclConfigGetInfo_When_CollCommIsNotInit_Expect_ReturnIsHCCL_E_PTR)
{
    UT_COMM_CREATE_DEFAULT(comm);
    void* info = nullptr;
    HcclOpExpansionMode opExpansionMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
    HcclResult ret = HcclConfigGetInfo(
        comm, HcclConfigType::HCCL_CONFIG_TYPE_OP_EXPANSION_MODE, sizeof(HcclOpExpansionMode), &opExpansionMode);
    EXPECT_EQ(ret, HCCL_E_PTR);
    Ut_Comm_Destroy(comm);
}

TEST_F(OpBaseMiscTest, Ut_HcclConfigGetInfo_When_CollCommIsNotInit_And_CfgTypeIsHCCL_ALGO_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);
    CollComm collComm(nullptr, 0, "ut_comm", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    MOCKER_CPP(&hcclComm::GetCollComm).stubs().will(returnValue(&collComm));
    char algoInfo[HCCL_COMM_ALGO_MAX_LENGTH] = {0};
    HcclResult ret
        = HcclConfigGetInfo(comm, HcclConfigType::HCCL_CONFIG_TYPE_HCCL_ALGO, HCCL_COMM_ALGO_MAX_LENGTH, algoInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    Ut_Comm_Destroy(comm);
}

TEST_F(OpBaseMiscTest, Ut_HcclConfigGetInfo_When_CfgTypeIsUB_MULTI_CHANNEL_NUM_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);
    CollComm collComm(nullptr, 0, "ut_comm", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    MOCKER_CPP(&hcclComm::GetCollComm).stubs().will(returnValue(&collComm));
    uint32_t multiChannelNum = 0;
    HcclResult ret = HcclConfigGetInfo(
        comm, HcclConfigType::HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM, sizeof(uint32_t), &multiChannelNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_GE(multiChannelNum, 1u);
    EXPECT_LE(multiChannelNum, 16u);
    Ut_Comm_Destroy(comm);
}

TEST_F(OpBaseMiscTest, Ut_HcclConfigGetInfo_When_CfgTypeIsAICPU_ORDER_LAUNCH_ENABLE_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);
    CollComm collComm(nullptr, 0, "ut_comm", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    MOCKER_CPP(&hcclComm::GetCollComm).stubs().will(returnValue(&collComm));
    uint32_t aicpuOrderLaunchEnable = 0xFFFFFFFFU;
    HcclResult ret = HcclConfigGetInfo(
        comm, HcclConfigType::HCCL_CONFIG_TYPE_AICPU_ORDER_LAUNCH_ENABLE, sizeof(uint32_t), &aicpuOrderLaunchEnable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 初值取合法域外哨兵，防住漏写info仅返回成功的回归；断言容忍UT环境中已显式配置该环境变量的情况
    EXPECT_LE(aicpuOrderLaunchEnable, 1u);
    Ut_Comm_Destroy(comm);
}

TEST_F(OpBaseMiscTest, Ut_HcclConfigGetInfo_When_CfgTypeIsOP_EXPANSION_MODE_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);
    CollComm collComm(nullptr, 0, "ut_comm", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    MOCKER_CPP(&hcclComm::GetCollComm).stubs().will(returnValue(&collComm));
    MyRank myRank(nullptr, 0, collComm.GetCommConfig(), ManagerCallbacks{}, nullptr, nullptr);
    MOCKER_CPP(&CollComm::GetMyRank).stubs().will(returnValue(&myRank));
    MOCKER_CPP(&MyRank::GetOpExpansionMode).stubs().will(returnValue(0u));
    HcclOpExpansionMode opExpansionMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
    HcclResult ret = HcclConfigGetInfo(
        comm, HcclConfigType::HCCL_CONFIG_TYPE_OP_EXPANSION_MODE, sizeof(HcclOpExpansionMode), &opExpansionMode);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    Ut_Comm_Destroy(comm);
}

// 正向映射: 内部值10(AIV_URMA_MODE)应返回HCCL_OP_EXPANSION_MODE_AIV_URMA, 防查询映射被误改
TEST_F(OpBaseMiscTest, Ut_HcclConfigGetInfo_When_OpExpansionModeIsAivUrmaMode_Expect_ReturnIsAivUrma)
{
    UT_COMM_CREATE_DEFAULT(comm);
    CollComm collComm(nullptr, 0, "ut_comm", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    MOCKER_CPP(&hcclComm::GetCollComm).stubs().will(returnValue(&collComm));
    MyRank myRank(nullptr, 0, collComm.GetCommConfig(), ManagerCallbacks{}, nullptr, nullptr);
    MOCKER_CPP(&CollComm::GetMyRank).stubs().will(returnValue(&myRank));
    MOCKER_CPP(&MyRank::GetOpExpansionMode).stubs().will(returnValue(10u));
    HcclOpExpansionMode opExpansionMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
    HcclResult ret = HcclConfigGetInfo(
        comm, HcclConfigType::HCCL_CONFIG_TYPE_OP_EXPANSION_MODE, sizeof(HcclOpExpansionMode), &opExpansionMode);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(opExpansionMode, HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AIV_URMA);
    Ut_Comm_Destroy(comm);
}

TEST_F(OpBaseMiscTest, Ut_HcclConfigGetInfo_When_DeterministicIsZeroOneTwo_Expect_ReturnConfiguredValue)
{
    UT_COMM_CREATE_DEFAULT(comm);
    CollComm collComm(nullptr, 0, "ut_comm", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    MOCKER_CPP(&hcclComm::GetCollComm).stubs().will(returnValue(&collComm));

    for (uint32_t expected = 0U; expected <= 2U; ++expected) {
        ASSERT_EQ(collComm.GetCommConfig().SetConfigDeterministic(static_cast<u8>(expected)), HCCL_SUCCESS);
        uint32_t deterministic = 3U;
        HcclResult ret = HcclConfigGetInfo(
            comm, HcclConfigType::HCCL_CONFIG_TYPE_DETERMINISTIC, sizeof(deterministic), &deterministic);

        EXPECT_EQ(ret, HCCL_SUCCESS);
        EXPECT_EQ(deterministic, expected);
    }
    Ut_Comm_Destroy(comm);
}

TEST_F(OpBaseMiscTest, Ut_HcclConfigGetInfo_When_DeterministicArgumentsInvalid_Expect_ReturnErrorAndKeepOutput)
{
    UT_COMM_CREATE_DEFAULT(comm);
    CollComm collComm(nullptr, 0, "ut_comm", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    ASSERT_EQ(collComm.GetCommConfig().SetConfigDeterministic(2U), HCCL_SUCCESS);
    MOCKER_CPP(&hcclComm::GetCollComm).stubs().will(returnValue(&collComm));
    uint32_t deterministic = 7U;

    HcclResult ret
        = HcclConfigGetInfo(comm, HcclConfigType::HCCL_CONFIG_TYPE_DETERMINISTIC, sizeof(uint16_t), &deterministic);

    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(deterministic, 7U);

    EXPECT_EQ(
        HcclConfigGetInfo(
            nullptr, HcclConfigType::HCCL_CONFIG_TYPE_DETERMINISTIC, sizeof(deterministic), &deterministic),
        HCCL_E_PTR);
    EXPECT_EQ(deterministic, 7U);
    EXPECT_EQ(
        HcclConfigGetInfo(comm, HcclConfigType::HCCL_CONFIG_TYPE_DETERMINISTIC, sizeof(deterministic), nullptr),
        HCCL_E_PTR);
    Ut_Comm_Destroy(comm);
}

TEST_F(OpBaseMiscTest, Ut_HcclSetConfig_When_A5CommExists_Expect_UpdateUnlessEnvironmentConfigured)
{
    MOCKER(hrtGetHcclV2Support).stubs().will(invoke(HrtGetHcclV2SupportForUt));

    g_hcclDeviceId = 0;
    auto hcclCommPtr = std::make_shared<hcclComm>();
    hcclCommPtr->collComm_
        = std::make_unique<CollComm>(nullptr, 0, "ut_deterministic", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    HcclOpInfoCtx& opBaseInfo = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    opBaseInfo.opGroup2CommMap.clear();
    opBaseInfo.opGroup2CommMap.emplace("ut_deterministic", hcclCommPtr);

    struct TestCase {
        const char* environmentValue;
        u8 expected;
    };
    const TestCase testCases[] = {
        {nullptr, 0U},
        {"strict", 2U},
        {"", 0U},
    };
    for (const auto& testCase : testCases) {
        if (testCase.environmentValue == nullptr) {
            unsetenv("HCCL_DETERMINISTIC");
        } else {
            setenv("HCCL_DETERMINISTIC", testCase.environmentValue, 1);
        }
        EXPECT_EQ(hcclCommPtr->collComm_->GetCommConfig().SetConfigDeterministic(2U), HCCL_SUCCESS);
        HcclConfigValue configValue{};
        configValue.value = 0;

        EXPECT_EQ(HcclSetConfig(HCCL_DETERMINISTIC, configValue), HCCL_SUCCESS);
        EXPECT_EQ(hcclCommPtr->collComm_->GetCommConfig().GetConfigDeterministic(), testCase.expected);
    }

    unsetenv("HCCL_DETERMINISTIC");
    opBaseInfo.opGroup2CommMap.clear();
}

TEST_F(OpBaseMiscTest, Ut_HcclConfigGetInfo_When_CfgTypeIsUDI_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);
    CollComm collComm(nullptr, 0, "ut_comm", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    MOCKER_CPP(&hcclComm::GetCollComm).stubs().will(returnValue(&collComm));
    (void)collComm.GetCommConfig().SetConfigUdi(std::string("test_udi"));
    char udiInfo[UDI_MAX_LENGTH] = {0};
    HcclResult ret = HcclConfigGetInfo(comm, HcclConfigType::HCCL_CONFIG_TYPE_UDI, UDI_MAX_LENGTH, udiInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(udiInfo, "test_udi");
    Ut_Comm_Destroy(comm);
}

TEST_F(OpBaseMiscTest, Ut_HcclConfigGetInfo_When_CfgTypeIsUDI_And_InfoLenTooSmall_Expect_ReturnIsHCCL_E_PARA)
{
    UT_COMM_CREATE_DEFAULT(comm);
    CollComm collComm(nullptr, 0, "ut_comm", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    MOCKER_CPP(&hcclComm::GetCollComm).stubs().will(returnValue(&collComm));
    char udiInfo[UDI_MAX_LENGTH] = {0};
    HcclResult ret = HcclConfigGetInfo(comm, HcclConfigType::HCCL_CONFIG_TYPE_UDI, 1, udiInfo);
    EXPECT_EQ(ret, HCCL_E_PARA);
    Ut_Comm_Destroy(comm);
}

TEST_F(OpBaseMiscTest, Ut_HcclCommSymWinGet_When_GetCommSymWinSucceeds_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);
    MOCKER_CPP(&hcclComm::IsCommunicatorV2).stubs().will(returnValue(false));
    MOCKER_CPP(&hcclComm::GetCommSymWin)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    u8 buffer = 0;
    HcclCommSymWindow winHandle = nullptr;
    size_t offset = 0;
    HcclResult ret = HcclCommSymWinGet(comm, &buffer, sizeof(buffer), &winHandle, &offset);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    Ut_Comm_Destroy(comm);
}
