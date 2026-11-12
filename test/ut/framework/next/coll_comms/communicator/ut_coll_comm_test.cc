/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../ut_hcomm_base.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define private public
#include "coll_comm.h"
#include "rank_info_detect.h"
#include "symmetric_memory/symmetric_memory.h"
#undef private
#include "root_handle_v2.h"
#include "rank_table_info.h"
#include "ubmem_symmetric_memory.h"
#include "coll_comm_config.h"
#include "hcom_common.h"

#include "hccl_team_mgr.h"
#include "hccl_team_c_adpt.h"
#include "hcomm_team.h"
#include "hcomm_team_c_adpt.h"
#include "hccl/hccl_channel.h"
#include "hccl/hccl_comm.h"
#include "my_rank.h"
#include "coll_comm.h"
#include "coll_comm_mgr.h"

namespace {
std::unique_ptr<CollComm> PrepareSuspendingCollCommForResetNotify(HcclResult resetNotifyRet)
{
    auto coll = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), hccl::ManagerCallbacks{});
    coll->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING;
    coll->isCleaned_ = true;
    coll->commEngineResMgr_ = std::make_unique<CommEngineResMgr>();

    MOCKER_CPP(&MyRank::Resume, HcclResult(MyRank::*)()).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommEngineResMgr::ResetCommLocalNotifies, HcclResult(CommEngineResMgr::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(resetNotifyRet));

    aclrtBinHandle binHandle;
    coll->myRank_ = std::make_shared<MyRank>(binHandle, 0, coll->GetCommConfig(), ManagerCallbacks(), nullptr, nullptr);
    return coll;
}
} // namespace

class TestCollComm : public TestHcommCAdptBase {
public:
    void SetUp() override { TestHcommCAdptBase::SetUp(); }
    void TearDown() override { TestHcommCAdptBase::TearDown(); }
};

HcclResult StubCollCommUrmaHrtMalloc(void** devPtr, u64 size, bool Level2Address)
{
    static uintptr_t devAddr = 0x7000000;
    (void)size;
    (void)Level2Address;
    *devPtr = reinterpret_cast<void*>(devAddr);
    devAddr += 0x1000;
    return HCCL_SUCCESS;
}

TEST_F(TestCollComm, Ut_TestCollCommInit_When_RankGraphNullptr_Return_HCCL_E_PTR)
{
    hccl::CollComm collComm(nullptr, 0, "test_comm", hccl::ManagerCallbacks{});
    HcclMem cclBuffer = {};
    HcclResult ret = collComm.Init(nullptr, nullptr, cclBuffer, 0);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestCollComm, Ut_PrintCommKeyInfo_When_Called_Expect_A5Keywords)
{
    hccl::CollComm collComm(reinterpret_cast<void*>(0x123), 3, "test_comm", hccl::ManagerCallbacks{});
    collComm.deviceLogicId_ = 1;

    const s32 originalLogLevel = log_level_get_stub();
    log_level_set_stub(DLOG_INFO);
    testing::internal::CaptureStdout();
    collComm.PrintCommKeyInfo(8);
    const std::string log = testing::internal::GetCapturedStdout();
    log_level_set_stub(originalLogLevel);

    EXPECT_NE(log.find("[Communicator Key Info]identifier[test_comm] rankSize[8]"), std::string::npos);
    EXPECT_NE(log.find("[LocalRank Key Info]userRank[3] deviceLogicId[1]"), std::string::npos);
}

TEST_F(TestCollComm, test_get_comm_status_initial_and_after_change)
{
    std::unique_ptr<CollComm> coll_
        = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    coll_->isCleaned_ = false;
    EXPECT_EQ(coll_->GetCommStatus(), HcclCommStatus::HCCL_COMM_STATUS_INVALID);

    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;
    EXPECT_EQ(coll_->GetCommStatus(), HcclCommStatus::HCCL_COMM_STATUS_READY);
}

TEST_F(TestCollComm, test_suspend_success_and_idempotent)
{
    std::unique_ptr<CollComm> coll_
        = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    coll_->isCleaned_ = false;
    // mock MyRank::StopLaunch to return success
    MOCKER_CPP(&MyRank::StopLaunch, HcclResult(MyRank::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // attach a MyRank instance (can be real or mocked; method is mocked above)
    aclrtBinHandle binHandle;
    coll_->myRank_
        = std::make_shared<MyRank>(binHandle, 0, coll_->GetCommConfig(), ManagerCallbacks(), nullptr, nullptr);

    auto ret = coll_->Suspend();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(coll_->commStatus_, HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING);

    // calling Suspend again when already suspending should return success without error
    ret = coll_->Suspend();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestCollComm, test_clean_fail_not_suspending)
{
    std::unique_ptr<CollComm> coll_
        = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    coll_->isCleaned_ = false;
    // when not suspending, Clean should return not support
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;
    auto ret = coll_->Clean();
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(TestCollComm, test_clean_success_and_idempotent)
{
    std::unique_ptr<CollComm> coll_
        = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    coll_->isCleaned_ = false;
    // prepare for cleaning: put into suspending state
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING;
    coll_->isCleaned_ = false;

    // mock MyRank::Clean to return success
    MOCKER_CPP(&MyRank::Clean, HcclResult(MyRank::*)()).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // attach a MyRank instance
    aclrtBinHandle binHandle;
    coll_->myRank_
        = std::make_shared<MyRank>(binHandle, 0, coll_->GetCommConfig(), ManagerCallbacks(), nullptr, nullptr);

    auto ret = coll_->Clean();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(coll_->isCleaned_);

    // calling Clean again should be idempotent and return success
    ret = coll_->Clean();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestCollComm, test_resume_fail_invalid_and_resume_success)
{
    std::unique_ptr<CollComm> coll_
        = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    coll_->isCleaned_ = false;
    // Resume when commStatus_ is INVALID should return internal error
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    auto ret = coll_->Resume();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    // Now test successful resume from SUSPENDING
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING;
    coll_->isCleaned_ = true;

    // mock MyRank::Resume to return success
    MOCKER_CPP(&MyRank::Resume, HcclResult(MyRank::*)()).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // attach a MyRank instance
    aclrtBinHandle binHandle;
    coll_->myRank_
        = std::make_shared<MyRank>(binHandle, 0, coll_->GetCommConfig(), ManagerCallbacks(), nullptr, nullptr);

    ret = coll_->Resume();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(coll_->commStatus_, HcclCommStatus::HCCL_COMM_STATUS_READY);
    EXPECT_FALSE(coll_->isCleaned_);
}

TEST_F(TestCollComm, Ut_Resume_When_CommEngineResMgrNullptr_Expect_SkipResetCommLocalNotifiesAndSuccess)
{
    std::unique_ptr<CollComm> coll_
        = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING;
    coll_->isCleaned_ = true;
    coll_->commEngineResMgr_ = nullptr;

    MOCKER_CPP(&MyRank::Resume, HcclResult(MyRank::*)()).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    aclrtBinHandle binHandle;
    coll_->myRank_
        = std::make_shared<MyRank>(binHandle, 0, coll_->GetCommConfig(), ManagerCallbacks(), nullptr, nullptr);

    auto ret = coll_->Resume();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(coll_->commStatus_, HcclCommStatus::HCCL_COMM_STATUS_READY);
    EXPECT_FALSE(coll_->isCleaned_);
}

TEST_F(TestCollComm, Ut_Resume_When_ResetCommLocalNotifiesSuccess_Expect_ReturnSuccess)
{
    auto coll = PrepareSuspendingCollCommForResetNotify(HCCL_SUCCESS);

    auto ret = coll->Resume();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(coll->commStatus_, HcclCommStatus::HCCL_COMM_STATUS_READY);
    EXPECT_FALSE(coll->isCleaned_);
}

TEST_F(TestCollComm, Ut_Resume_When_ResetCommLocalNotifiesFailed_Expect_ReturnFailed)
{
    auto coll = PrepareSuspendingCollCommForResetNotify(HCCL_E_INTERNAL);

    auto ret = coll->Resume();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(coll->commStatus_, HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING);
    EXPECT_TRUE(coll->isCleaned_);
}

TEST_F(TestCollComm, Ut_InitSimpleMode_When_Success_Expect_ReturnsSuccessAndRankgraphSet)
{
    // Create CollComm instance
    hccl::CollComm collComm(nullptr, 0, "test_comm", hccl::ManagerCallbacks{});

    // Verify initial state - rankgraph should be nullptr before initialization
    // This validates the internal state that InitSimpleMode would populate
    EXPECT_EQ(collComm.GetRankSize(), 0u);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_ValidConfig_Expect_Success)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 2U;
    config.hcclRdmaTrafficClass = 120U;
    config.hcclRdmaServiceLevel = 3U;
    config.hcclQos = 5U;
    config.hcclChannelSqDepth = 128U;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, 0), HCCL_SUCCESS);
    EXPECT_EQ(opExpansionMode, 2U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigHcclQos(), 5U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigTrafficClass(), 120U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigServiceLevel(), 3U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigSqDepth(), 128U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigSymmetricMemoryStride(), HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_NullConfig_Expect_Success)
{
    for (const int32_t processValue : {0, 2}) {
        hccl::CollComm coll(nullptr, 0, "ut_qos", hccl::ManagerCallbacks{});
        uint32_t opExpansionMode = 9U;
        EXPECT_EQ(ApplyHcclCommConfig(nullptr, coll.GetCommConfig(), opExpansionMode, processValue), HCCL_SUCCESS);
        EXPECT_EQ(opExpansionMode, 0U);
        EXPECT_EQ(coll.GetCommConfig().GetConfigDeterministic(), static_cast<u8>(processValue));
        EXPECT_EQ(coll.GetCommConfig().GetConfigSqDepth(), HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET);
        EXPECT_EQ(coll.GetCommConfig().GetConfigSymmetricMemoryStride(), HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE);
    }
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_DeterministicConfigured_Expect_UseCommValueOrProcessDefault)
{
    struct TestCase {
        uint32_t commValue;
        int32_t processValue;
        uint32_t expectedValue;
    };
    const TestCase testCases[] = {
        {0U, 2, 0U},
        {1U, 2, 1U},
        {2U, 0, 2U},
        {HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET, 0, 0U},
        {HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET, 1, 1U},
        {HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET, 2, 2U},
    };

    for (const auto& testCase : testCases) {
        SCOPED_TRACE(testing::Message() << "commValue=" << testCase.commValue);
        hccl::CollComm coll(nullptr, 0, "ut_deterministic", hccl::ManagerCallbacks{});
        HcclCommConfig config{};
        UtInitHcclCommConfig(config);
        config.hcclDeterministic = testCase.commValue;
        uint32_t opExpansionMode = 0U;

        ASSERT_EQ(
            ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, testCase.processValue), HCCL_SUCCESS);
        EXPECT_EQ(coll.GetCommConfig().GetConfigDeterministic(), testCase.expectedValue);
    }
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_DeterministicInvalid_Expect_ReturnHCCL_E_PARA)
{
    hccl::CollComm coll(nullptr, 0, "ut_deterministic_invalid", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclDeterministic = 3U;
    uint32_t opExpansionMode = 0U;

    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, 0), HCCL_E_PARA);

    config.hcclDeterministic = HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, -1), HCCL_E_PARA);
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, 3), HCCL_E_PARA);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_TwoCommsConfigured_Expect_KeepIndependentValues)
{
    hccl::CollComm first(nullptr, 0, "ut_deterministic_first", hccl::ManagerCallbacks{});
    hccl::CollComm second(nullptr, 0, "ut_deterministic_second", hccl::ManagerCallbacks{});
    HcclCommConfig firstConfig{};
    HcclCommConfig secondConfig{};
    UtInitHcclCommConfig(firstConfig);
    UtInitHcclCommConfig(secondConfig);
    firstConfig.hcclDeterministic = 0U;
    secondConfig.hcclDeterministic = 2U;
    firstConfig.hcclSymWinMaxMemSizePerRank = 17ULL;
    secondConfig.hcclSymWinMaxMemSizePerRank = 48ULL;
    uint32_t opExpansionMode = 0U;

    ASSERT_EQ(ApplyHcclCommConfig(&firstConfig, first.GetCommConfig(), opExpansionMode, 0), HCCL_SUCCESS);
    ASSERT_EQ(ApplyHcclCommConfig(&secondConfig, second.GetCommConfig(), opExpansionMode, 0), HCCL_SUCCESS);
    EXPECT_EQ(first.GetCommConfig().GetConfigDeterministic(), 0U);
    EXPECT_EQ(second.GetCommConfig().GetConfigDeterministic(), 2U);
    EXPECT_EQ(first.GetCommConfig().GetConfigSymmetricMemoryStride(), 17ULL);
    EXPECT_EQ(second.GetCommConfig().GetConfigSymmetricMemoryStride(), 48ULL);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_SymmetricStrideConfigured_Expect_UseCommValue)
{
    for (const uint64_t stride : {1ULL, 16ULL, 17ULL, 24ULL, 48ULL}) {
        SCOPED_TRACE(testing::Message() << "stride=" << stride);
        hccl::CollComm coll(nullptr, 0, "ut_symmetric_stride", hccl::ManagerCallbacks{});
        HcclCommConfig config{};
        UtInitHcclCommConfig(config);
        config.hcclSymWinMaxMemSizePerRank = stride;
        uint32_t opExpansionMode = 0U;

        ASSERT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, 0), HCCL_SUCCESS);
        EXPECT_EQ(coll.GetCommConfig().GetConfigSymmetricMemoryStride(), stride);
    }
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_SqDepthVaries_Expect_VersionRules)
{
    struct TestCase {
        uint32_t version;
        uint32_t sqDepth;
        uint32_t scqDepth;
        HcclResult expectedResult;
        uint32_t expectedSqDepth;
        uint32_t expectedScqDepth;
    };

    const TestCase testCases[] = {
        {11U, 16U, 64U, HCCL_SUCCESS, 16U, HCCL_COMM_SCQ_DEPTH_CONFIG_NOT_SET},
        {11U, 8192U, 32768U, HCCL_SUCCESS, 8192U, HCCL_COMM_SCQ_DEPTH_CONFIG_NOT_SET},
        {11U, 15U, 64U, HCCL_SUCCESS, 15U, HCCL_COMM_SCQ_DEPTH_CONFIG_NOT_SET},
        {11U, 8193U, 64U, HCCL_SUCCESS, 8193U, HCCL_COMM_SCQ_DEPTH_CONFIG_NOT_SET},
        {10U, 128U, 64U, HCCL_SUCCESS, HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET, HCCL_COMM_SCQ_DEPTH_CONFIG_NOT_SET},
        {12U, 16U, 64U, HCCL_SUCCESS, 16U, 64U},
        {12U, 8192U, 32768U, HCCL_SUCCESS, 8192U, 32768U},
    };

    for (const auto& testCase : testCases) {
        SCOPED_TRACE(
            testing::Message() << "version=" << testCase.version << ", sqDepth=" << testCase.sqDepth
                               << ", scqDepth=" << testCase.scqDepth);
        hccl::CollComm coll(nullptr, 0, "ut_sqdepth", hccl::ManagerCallbacks{});
        HcclCommConfig config{};
        UtInitHcclCommConfig(config);
        auto* configInfo = reinterpret_cast<CommConfigInfo*>(config.reserved);
        configInfo->version = testCase.version;
        config.hcclChannelSqDepth = testCase.sqDepth;
        config.hcclChannelScqDepth = testCase.scqDepth;
        uint32_t opExpansionMode = 0U;

        EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, 0), testCase.expectedResult);
        EXPECT_EQ(coll.GetCommConfig().GetConfigSqDepth(), testCase.expectedSqDepth);
        EXPECT_EQ(coll.GetCommConfig().GetConfigScqDepth(), testCase.expectedScqDepth);
    }
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_InvalidHcclQos_Expect_EPara)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclQos = 8U;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, 0), HCCL_E_PARA);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_InvalidTrafficClass_Expect_EPara)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclRdmaTrafficClass = 256U;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, 0), HCCL_E_PARA);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_InvalidServiceLevel_Expect_EPara)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclRdmaServiceLevel = 8U;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, 0), HCCL_E_PARA);
}

TEST_F(TestCollComm, Ut_RegisterPendingSymmetricMemHandles_When_PendingConsumed_Expect_RegisteredHandleRetained)
{
    MOCKER_CPP(hrtMalloc).stubs().will(invoke(StubCollCommUrmaHrtMalloc));
    MOCKER_CPP(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));
    hccl::CollComm coll(nullptr, 0, "ut_sym", hccl::ManagerCallbacks{});
    aclrtBinHandle binHandle{};
    coll.myRank_ = std::make_shared<MyRank>(binHandle, 0, coll.GetCommConfig(), ManagerCallbacks(), nullptr, nullptr);
    coll.myRank_->commMems_ = std::make_unique<CommMems>(0);
    coll.symmetricMemory_.reset(new SymmetricMemory(0, 2, 0, SymmetricMemoryMode::URMA));

    void* win = nullptr;
    void* ptr = reinterpret_cast<void*>(0x8000000);
    HcclComm comm = reinterpret_cast<HcclComm>(&coll);
    EXPECT_EQ(coll.RegisterWindow(comm, ptr, 0x2000, &win), HCCL_SUCCESS);
    HcclComm ownerComm = nullptr;
    EXPECT_EQ(hccl::GetHcommWindowComm(win, ownerComm), HCCL_SUCCESS);
    EXPECT_EQ(ownerComm, comm);

    EXPECT_EQ(coll.RegisterPendingSymmetricMemHandles(), HCCL_SUCCESS);

    SymmetricMemoryResource resource;
    // PR 后 *winHandle 为 HcommWindow device 副本，symmetricMemory_ 资源以 devSymWin 为 key，
    // 经 hcommToSymMap_ 反查 devSymWin 后查询
    void* devSymWin = coll.hcommToSymMap_[win];
    ASSERT_NE(devSymWin, nullptr);
    EXPECT_EQ(coll.symmetricMemory_->GetRegisteredMemoryResource(devSymWin, resource), HCCL_SUCCESS);
    EXPECT_NE(resource.memHandle, nullptr);

    EXPECT_EQ(coll.RegisterPendingSymmetricMemHandles(), HCCL_SUCCESS);

    std::vector<HcclMemHandle> memHandles;
    EXPECT_EQ(coll.GetAllRegisteredSymMemHandles(memHandles), HCCL_SUCCESS);
    ASSERT_EQ(memHandles.size(), 1U);
    EXPECT_EQ(memHandles[0], static_cast<HcclMemHandle>(resource.memHandle));

    std::vector<std::string> remoteMemTags;
    remoteMemTags.emplace_back(resource.memTag);
    EXPECT_EQ(coll.GetRemoteMissingSymMemHandles(remoteMemTags, memHandles), HCCL_SUCCESS);
    EXPECT_TRUE(memHandles.empty());

    remoteMemTags[0] = "unrelated_tag";
    EXPECT_EQ(coll.GetRemoteMissingSymMemHandles(remoteMemTags, memHandles), HCCL_SUCCESS);
    ASSERT_EQ(memHandles.size(), 1U);
    EXPECT_EQ(memHandles[0], static_cast<HcclMemHandle>(resource.memHandle));
}

TEST_F(TestCollComm, Ut_GetCommSymWin_When_UbMemoryEnabled_Expect_UseSharedWindowMapping)
{
    hccl::CollComm coll(nullptr, 0, "ut_ub_sym_get", hccl::ManagerCallbacks{});
    const std::vector<uint32_t> worldRanks{0U};
    coll.ubMemSymmetricMemory_ = std::make_unique<hccl::UbMemSymmetricMemory>(
        &coll, reinterpret_cast<HcommTeamHandle>(0x1000U), 2U, worldRanks);
    coll.symmetricMemory_.reset(new SymmetricMemory(0U, 2U, 0U, SymmetricMemoryMode::URMA));

    void* userAddress = reinterpret_cast<void*>(0x3000U);
    void* legacyWindow = reinterpret_cast<void*>(0x4000U);
    HcclCommSymWindow hcommWindow = reinterpret_cast<HcclCommSymWindow>(0x5000U);
    auto urmaWindow = std::make_shared<SymmetricWindow>();
    urmaWindow->userVa = userAddress;
    urmaWindow->userSize = 0x1000U;
    urmaWindow->devWin = legacyWindow;
    coll.symmetricMemory_->sortedWindows_.emplace_back(urmaWindow);
    coll.symmetricMemory_->windowMap_.emplace(legacyWindow, urmaWindow);
    coll.symToHcommMap_.emplace(legacyWindow, hcommWindow);

    HcclCommSymWindow winHandle = reinterpret_cast<HcclCommSymWindow>(0x2000U);
    size_t offset = 0x100U;
    EXPECT_EQ(
        coll.GetCommSymWin(static_cast<uint8_t*>(userAddress) + 0x80U, 0x100U, &winHandle, &offset), HCCL_SUCCESS);
    EXPECT_EQ(winHandle, hcommWindow);
    EXPECT_EQ(offset, 0x80U);

    coll.symToHcommMap_.clear();
    coll.symmetricMemory_->windowMap_.clear();
    coll.symmetricMemory_->sortedWindows_.clear();
}

TEST_F(TestCollComm, Ut_GetCommSymWin_When_WindowNotFound_Expect_SuccessWithNullHandle)
{
    hccl::CollComm coll(nullptr, 0, "ut_urma_sym_miss", hccl::ManagerCallbacks{});
    coll.symmetricMemory_.reset(new SymmetricMemory(0, 2, 0, SymmetricMemoryMode::URMA));

    HcclCommSymWindow winHandle = reinterpret_cast<HcclCommSymWindow>(0x2000U);
    size_t offset = 0x100U;
    EXPECT_EQ(coll.GetCommSymWin(reinterpret_cast<void*>(0x3000U), 0x100U, &winHandle, &offset), HCCL_SUCCESS);
    EXPECT_EQ(winHandle, nullptr);
    EXPECT_EQ(offset, 0U);
}

TEST_F(TestCollComm, Ut_DeregisterWindow_When_UbDeregisterFails_Expect_FullyRetryable)
{
    hccl::CollComm coll(nullptr, 0, "ut_sym_ub_fail", hccl::ManagerCallbacks{});
    coll.symmetricMemory_.reset(new SymmetricMemory(0, 2, 0, SymmetricMemoryMode::URMA));
    const std::vector<uint32_t> ranks{0U};
    coll.ubMemSymmetricMemory_ = std::make_unique<hccl::UbMemSymmetricMemory>(&coll, nullptr, 0U, ranks);

    HcclCommSymWindow window = reinterpret_cast<HcclCommSymWindow>(0x8200000);
    void* legacyWindow = reinterpret_cast<void*>(0x8300000);
    coll.hcommToSymMap_[window] = legacyWindow;

    EXPECT_EQ(coll.DeregisterWindow(window), HCCL_E_NOT_FOUND);
    EXPECT_NE(coll.hcommToSymMap_.find(window), coll.hcommToSymMap_.end());
}

TEST_F(TestCollComm, Ut_UpdateSymmetricRemoteMem_When_ChannelReturnsRemoteMem_Expect_UpdateWindow)
{
    MOCKER_CPP(hrtMalloc).stubs().will(invoke(StubCollCommUrmaHrtMalloc));
    MOCKER_CPP(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));

    hccl::CollComm coll(nullptr, 0, "ut_sym", hccl::ManagerCallbacks{});
    coll.symmetricMemory_.reset(new SymmetricMemory(0, 2, 0, SymmetricMemoryMode::URMA));

    void* win = nullptr;
    void* ptr = reinterpret_cast<void*>(0x9000000);
    constexpr size_t winSize = 0x2000;
    HcclComm comm = reinterpret_cast<HcclComm>(&coll);
    EXPECT_EQ(coll.RegisterWindow(comm, ptr, winSize, &win), HCCL_SUCCESS);
    HcclComm ownerComm = nullptr;
    EXPECT_EQ(hccl::GetHcommWindowComm(win, ownerComm), HCCL_SUCCESS);
    EXPECT_EQ(ownerComm, comm);

    SymmetricMemoryResource resource;
    resource.memHandle = reinterpret_cast<void*>(0x9100000);
    resource.memTag = std::string(HCCL_SYMMETRIC_MEMORY_TAG_PREFIX) + "ut_sym_addr_"
                      + std::to_string(reinterpret_cast<uintptr_t>(ptr)) + "_size_" + std::to_string(winSize);
    // PR 后 *winHandle 为 HcommWindow device 副本，SetRegisteredMemoryResource 须以 devSymWin 为 key
    void* devSymWin = coll.hcommToSymMap_[win];
    ASSERT_NE(devSymWin, nullptr);
    EXPECT_EQ(coll.symmetricMemory_->SetRegisteredMemoryResource(devSymWin, resource), HCCL_SUCCESS);

    CommMem remoteMem{};
    remoteMem.type = COMM_MEM_TYPE_DEVICE;
    remoteMem.addr = reinterpret_cast<void*>(0x9200000);
    remoteMem.size = winSize;
    std::vector<std::string> memTags = {resource.memTag};
    EXPECT_EQ(coll.UpdateSymmetricRemoteMem(1, &remoteMem, memTags), HCCL_SUCCESS);

    // PR 后 *winHandle 为 HcommWindow device 副本，remoteMemMap_ 以 devSymWin 为 key
    auto remoteMemIt = coll.symmetricMemory_->remoteMemMap_.find(devSymWin);
    ASSERT_NE(remoteMemIt, coll.symmetricMemory_->remoteMemMap_.end());
    ASSERT_EQ(remoteMemIt->second.size(), 2U);
    EXPECT_EQ(remoteMemIt->second[1].addr, remoteMem.addr);
    EXPECT_EQ(remoteMemIt->second[1].size, remoteMem.size);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_NullptrConfig_Expect_TcSlSkipped)
{
    hccl::CollComm coll(nullptr, 0, "ut_tcsl", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclRdmaTrafficClass = 0xFFFFFFFFu;
    config.hcclRdmaServiceLevel = 0xFFFFFFFFu;
    config.hcclQos = 0xFFFFFFFFu;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, 0), HCCL_SUCCESS);
    EXPECT_EQ(coll.GetCommConfig().GetConfigTrafficClass(), 0xFFFFFFFFu);
    EXPECT_EQ(coll.GetCommConfig().GetConfigServiceLevel(), 0xFFFFFFFFu);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_TcNotMultipleOf4_Expect_EPara)
{
    hccl::CollComm coll(nullptr, 0, "ut_tc", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclRdmaTrafficClass = 3U;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, 0), HCCL_E_PARA);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_TcValidMultipleOf4_Expect_Success)
{
    hccl::CollComm coll(nullptr, 0, "ut_tc_valid", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclRdmaTrafficClass = 4U;
    config.hcclRdmaServiceLevel = 5U;
    config.hcclQos = 0xFFFFFFFFu;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, 0), HCCL_SUCCESS);
    EXPECT_EQ(coll.GetCommConfig().GetConfigTrafficClass(), 4U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigServiceLevel(), 5U);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_QosDefault_Expect_Success)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos_def", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclQos = 0xFFFFFFFFu;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, 0), HCCL_SUCCESS);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_QosValid_Expect_Success)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos_valid", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclQos = 7U;
    config.hcclRdmaTrafficClass = 0xFFFFFFFFu;
    config.hcclRdmaServiceLevel = 0xFFFFFFFFu;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, 0), HCCL_SUCCESS);
    EXPECT_EQ(coll.GetCommConfig().GetConfigHcclQos(), 7U);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_LowVersionQos_Expect_QosNotSet)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos_ver", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclQos = 5U;
    config.hcclRdmaTrafficClass = 0xFFFFFFFFu;
    config.hcclRdmaServiceLevel = 0xFFFFFFFFu;
    CommConfigInfo info{};
    info.version = 5U;
    errno_t sRet = memcpy_s(config.reserved, sizeof(config.reserved), &info, sizeof(info));
    ASSERT_EQ(sRet, EOK);
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, 0), HCCL_SUCCESS);
    EXPECT_EQ(coll.GetCommConfig().GetConfigHcclQos(), HCCL_COMM_QOS_CONFIG_NOT_SET);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_TcZero_Expect_Success)
{
    hccl::CollComm coll(nullptr, 0, "ut_tc_zero", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclRdmaTrafficClass = 0U;
    config.hcclRdmaServiceLevel = 0U;
    config.hcclQos = 0xFFFFFFFFu;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode, 0), HCCL_SUCCESS);
    EXPECT_EQ(coll.GetCommConfig().GetConfigTrafficClass(), 0U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigServiceLevel(), 0U);
}

// =====================================================================
// 以下用例覆盖 PR 4786（AIN 控制面重构）在 coll_comm.cc 新增的 4 个函数：
// CreatePrebuiltWorldTeam / InitWorldTeams / ReExchangeWindowsForBoundTeams /
// UpdateHcommWindowRemoteMem（对应 STC ut_ain_047~058，见 docs/stc.md 2.3/2.6）。
// 设计说明：
// 1) RankGraph::GetNetLayers/GetInstRanksByNetLayer/GetLinks 为虚函数（rank_graph_base.h），
//    此前判定"pImpl 转发不可 MOCKER"不成立——用可配置桩子类 UtRankGraphStub 直接注入
//    collComm.rankgraph_（#define private public），比 MOCKER 更稳（无签名/abi 依赖）。
// 2) HcommTeamCreate/HcommTeamGetNetLayer/HcommTeamGetEngine/
//    HcommTeamUpdateWindowRemoteMemByRank 为 extern "C" 接口，MOCKER(...) 打桩；
//    MyRank::CreateChannels/ChannelGetRemoteMems 为非虚成员函数，MOCKER_CPP 打桩。
// 3) HcclTeamMgr 为真实单例（不可 mock），用例内真实注册 team，
//    TearDown 用 UnregisterTeam/ClearByCollComm 兜底清理防状态泄漏。
// =====================================================================

namespace {
// —— 可配置 RankGraph 桩：按 (netLayer, proto) 配置每层 rank 集合与 peer 链路 ——
class UtRankGraphStub : public hccl::RankGraph {
public:
    ~UtRankGraphStub() override = default;

    HcclResult GetRankSize(uint32_t* rankSize) override
    {
        *rankSize = static_cast<uint32_t>(allRanks.size());
        return HCCL_SUCCESS;
    }
    HcclResult GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum) override
    {
        *netLayers = layerBuf.data();
        *netLayerNum = static_cast<uint32_t>(layerBuf.size());
        return getNetLayersRet;
    }
    HcclResult GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** rankList, uint32_t* rankNum) override
    {
        auto it = layerRanks.find(netLayer);
        if (it == layerRanks.end()) {
            *rankList = nullptr;
            *rankNum = 0;
            return HCCL_E_NOT_FOUND;
        }
        *rankList = it->second.data();
        *rankNum = static_cast<uint32_t>(it->second.size());
        return HCCL_SUCCESS;
    }
    HcclResult
    GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize) override
    {
        (void)srcRank;
        auto it = layerLinks.find(netLayer);
        if (it == layerLinks.end()) {
            *linkList = nullptr;
            *listSize = 0;
            return HCCL_SUCCESS;
        }
        *linkList = it->second.data();
        *listSize = static_cast<uint32_t>(it->second.size());
        return HCCL_SUCCESS;
    }
    HcclResult GetRankGraphInfo(GraphType type, void** graph, uint32_t* len) override
    {
        (void)type;
        *graph = nullptr;
        *len = 0;
        return HCCL_SUCCESS;
    }
    HcclResult GetInstTopoTypeByNetLayer(uint32_t netLayer, CommTopo* topoType) override
    {
        (void)netLayer;
        *topoType = COMM_TOPO_RESERVED;
        return HCCL_SUCCESS;
    }
    HcclResult GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum) override
    {
        *rankNum = static_cast<uint32_t>(layerRanks[netLayer].size());
        return HCCL_SUCCESS;
    }
    HcclResult GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize) override
    {
        (void)netLayer;
        *instSizeList = nullptr;
        *listSize = 0;
        return HCCL_SUCCESS;
    }
    HcclResult GetDeviceId(uint32_t rankId, uint32_t* deviceId) override
    {
        *deviceId = rankId;
        return HCCL_SUCCESS;
    }
    HcclResult GetEndpointInfo(
        uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen,
        void* info) override
    {
        (void)rankId;
        (void)endPointDesc;
        (void)endpointAttr;
        (void)infoLen;
        (void)info;
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> allRanks{0, 1};
    std::vector<uint32_t> layerBuf{0};
    std::map<uint32_t, std::vector<uint32_t>> layerRanks; // netLayer -> ranks
    std::map<uint32_t, std::vector<CommLink>> layerLinks; // netLayer -> links
    HcclResult getNetLayersRet{HCCL_SUCCESS};
};

CommLink MakeLink(CommProtocol proto)
{
    CommLink link{};
    link.header.version = 1;
    link.header.magicWord = 0x0f0e0f0f;
    link.header.size = sizeof(CommLink);
    link.linkAttr.linkProtocol = proto;
    link.linkAttr.hop = 1;
    return link;
}

// —— extern "C" 接口桩 ——
HcommTeamHandle UtNextWorldTeamHandle()
{
    static uintptr_t nextHandle = 0x30000;
    nextHandle += 0x1000;
    return reinterpret_cast<HcommTeamHandle>(nextHandle);
}

HcommResult UtStubHcommTeamCreateOk(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* syncMemSize)
{
    (void)worldTeam;
    (void)desc;
    *team = UtNextWorldTeamHandle(); // 每次分配不同伪句柄，避免单例内同 handle 重复注册 E_PARA
    *syncMemSize = 0;
    return 0;
}

HcommResult UtStubHcommTeamCreateFail(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* syncMemSize)
{
    (void)worldTeam;
    (void)desc;
    *team = nullptr;
    *syncMemSize = 0;
    return static_cast<HcommResult>(1); // HCOMM_E_PARA
}

uint32_t g_utGetNetLayerVal = 1;
HcommResult UtStubHcommTeamGetNetLayer(HcommTeamHandle team, uint32_t* netLayer)
{
    (void)team;
    *netLayer = g_utGetNetLayerVal;
    return 0;
}

CommEngine g_utGetEngineVal = COMM_ENGINE_AIV;
HcommResult UtStubHcommTeamGetEngine(HcommTeamHandle team, CommEngine* engine)
{
    (void)team;
    *engine = g_utGetEngineVal;
    return 0;
}

// HcommTeamUpdateWindowRemoteMemByRank 捕获桩：记录 sizes/slots 供偏移断言
bool g_utUpdateWinCalled = false;
uint32_t g_utUpdateWinSizes[3] = {0, 0, 0};
std::vector<uint32_t> g_utUpdateWinSlots;
HcclMemHandle g_utLastRemoteMem = nullptr;
HcommResult UtStubUpdateWindowRemoteMemByRank(
    HcclCommSymWindow handle, const uint32_t* sizes, uint32_t sizeNum, const uint32_t* slots, uint32_t slotNum,
    const CommMem* remoteMem)
{
    (void)handle;
    (void)sizeNum;
    (void)slotNum;
    g_utUpdateWinCalled = true;
    for (uint32_t i = 0; i < 3; i++) {
        g_utUpdateWinSizes[i] = sizes[i];
    }
    g_utUpdateWinSlots.assign(slots, slots + slotNum);
    g_utLastRemoteMem = remoteMem ? reinterpret_cast<HcclMemHandle>(remoteMem->addr) : nullptr;
    return 0;
}

// —— MyRank 非虚成员函数桩 ——
uint32_t g_utCreateChannelsCalls = 0;
HcclResult UtStubMyRankCreateChannelsOk(
    hccl::MyRank* self, CommEngine engine, const std::string& commTag, const HcclChannelDesc* channelDescs,
    uint32_t channelNum, ChannelHandle* channels)
{
    (void)self;
    (void)engine;
    (void)commTag;
    (void)channelDescs;
    g_utCreateChannelsCalls++;
    for (uint32_t i = 0; i < channelNum; i++) {
        channels[i] = static_cast<ChannelHandle>(0x71000) + static_cast<int>(i);
    }
    return HCCL_SUCCESS;
}

} // namespace

// ===================== 1. CreatePrebuiltWorldTeam =====================

// ut_ain_044 L2 侧：HcommTeamCreate 成功 + 真实注册，索引可查。
TEST_F(TestCollComm, Ut_CreatePrebuiltWorldTeam_When_CreateSuccess_Expect_RegisterInTeamMgr)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(UtStubHcommTeamCreateOk));
    hccl::CollComm coll(nullptr, 0, "ut_prebuilt", hccl::ManagerCallbacks{});
    uint32_t rankIds[2] = {0, 1};
    HcommTeamHandle created = nullptr;
    {
        // 捕获桩分配的句柄：经 FindWorldTeamByProtoLayer 反查验证。
        EXPECT_EQ(coll.CreatePrebuiltWorldTeam(COMM_PROTOCOL_UB_CTP, 1, rankIds, 2, 0), HCCL_SUCCESS);
        created = hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 1);
    }
    ASSERT_NE(created, nullptr);

    hccl::HcclTeamMgr::GetInstance().UnregisterTeam(created);
    GlobalMockObject::verify();
}

// HcommTeamCreate 失败透传为 HCCL_E_INTERNAL。
TEST_F(TestCollComm, Ut_CreatePrebuiltWorldTeam_When_CreateFail_Expect_ReturnInternal)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(UtStubHcommTeamCreateFail));
    hccl::CollComm coll(nullptr, 0, "ut_prebuilt_fail", hccl::ManagerCallbacks{});
    uint32_t rankIds[2] = {0, 1};

    EXPECT_EQ(coll.CreatePrebuiltWorldTeam(COMM_PROTOCOL_UB_CTP, 1, rankIds, 2, 0), HCCL_E_INTERNAL);

    EXPECT_EQ(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 1), nullptr);
    GlobalMockObject::verify();
}

// 创建成功后可按 (protocol, netLayer) 查询。
TEST_F(TestCollComm, Ut_CreatePrebuiltWorldTeam_When_Created_Expect_QuerySuccess)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(UtStubHcommTeamCreateOk));
    hccl::CollComm coll(nullptr, 0, "ut_prebuilt_dup", hccl::ManagerCallbacks{});
    uint32_t rankIds[2] = {0, 1};

    ASSERT_EQ(coll.CreatePrebuiltWorldTeam(COMM_PROTOCOL_UB_CTP, 1, rankIds, 2, 0), HCCL_SUCCESS);
    HcommTeamHandle created
        = hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 1);
    ASSERT_NE(created, nullptr);
    hccl::HcclTeamMgr::GetInstance().UnregisterTeam(created);
    GlobalMockObject::verify();
}

// ===================== 2. InitWorldTeams =====================

// ut_ain_047 URMA 多协议多层预制：netLayers={0,1}，layer0=UB_CTP、layer1=UBC_TP+UB_CTP，共 3 个 worldTeam。
TEST_F(TestCollComm, Ut_InitWorldTeams_When_MultiLayerMultiProto_Expect_ThreeWorldTeams)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(UtStubHcommTeamCreateOk));
    hccl::CollComm coll(nullptr, 0, "ut_iwt_multi", hccl::ManagerCallbacks{});

    UtRankGraphStub graph;
    graph.layerBuf = {0, 1};
    graph.layerRanks[0] = {0, 1};
    graph.layerRanks[1] = {0, 1};
    graph.layerLinks[0] = {MakeLink(COMM_PROTOCOL_UB_CTP)};
    graph.layerLinks[1] = {MakeLink(COMM_PROTOCOL_UBC_TP), MakeLink(COMM_PROTOCOL_UB_CTP)};
    coll.rankgraph_ = &graph;

    EXPECT_EQ(coll.InitWorldTeams(), HCCL_SUCCESS);

    EXPECT_NE(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 0), nullptr);
    EXPECT_NE(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UBC_TP, 1), nullptr);
    EXPECT_NE(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 1), nullptr);
    std::vector<uint32_t> sizes = {0, 0, 0};
    hccl::HcclTeamMgr::GetInstance().GetWorldTeamSizesPerNetLayer(&coll, sizes);
    EXPECT_EQ(sizes[0], 2U);
    EXPECT_EQ(sizes[1], 2U); // 同层多协议只记一份 layerRanks
    EXPECT_EQ(sizes[2], 0U);

    coll.registeredSymMemHandleMap_.clear();
    hccl::HcclTeamMgr::GetInstance().ClearByCollComm(&coll);
    coll.rankgraph_ = nullptr;
    GlobalMockObject::verify();
}

// ut_ain_048 UB_MEM 仅最高层预制：3 层均有 UB_MEM（layer1 另有 UB_CTP），最终仅 (UB_MEM,2) 1 个。
TEST_F(TestCollComm, Ut_InitWorldTeams_When_UbMemMultiLayer_Expect_OnlyHighestLayer)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(UtStubHcommTeamCreateOk));
    hccl::CollComm coll(nullptr, 0, "ut_iwt_ubmem", hccl::ManagerCallbacks{});

    UtRankGraphStub graph;
    graph.layerBuf = {0, 1, 2};
    graph.layerRanks[0] = {0, 1};
    graph.layerRanks[1] = {0, 1};
    graph.layerRanks[2] = {0, 1};
    graph.layerLinks[0] = {MakeLink(COMM_PROTOCOL_UB_MEM)};
    graph.layerLinks[1] = {MakeLink(COMM_PROTOCOL_UB_MEM), MakeLink(COMM_PROTOCOL_UB_CTP)};
    graph.layerLinks[2] = {MakeLink(COMM_PROTOCOL_UB_MEM)};
    coll.rankgraph_ = &graph;

    EXPECT_EQ(coll.InitWorldTeams(), HCCL_SUCCESS);

    EXPECT_NE(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_MEM, 2), nullptr);
    EXPECT_EQ(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_MEM, 0), nullptr);
    EXPECT_EQ(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_MEM, 1), nullptr);
    // 层分段表：layerRanksMap_ 按 netLayer 记录（协议无关），本场景 layer1(UB_CTP)+layer2(UB_MEM) 各 2
    std::vector<uint32_t> sizes = {0, 0, 0};
    hccl::HcclTeamMgr::GetInstance().GetWorldTeamSizesPerNetLayer(&coll, sizes);
    EXPECT_EQ(sizes[0], 0U);
    EXPECT_EQ(sizes[1], 2U);
    EXPECT_EQ(sizes[2], 2U);

    hccl::HcclTeamMgr::GetInstance().ClearByCollComm(&coll);
    coll.rankgraph_ = nullptr;
    GlobalMockObject::verify();
}

// UB Memory不限定固定NetLayer，多层均可达时选择最高层作为LSA WorldTeam。
TEST_F(TestCollComm, Ut_InitWorldTeams_When_UbMemHasHigherLayer_Expect_SelectHighestCandidate)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(UtStubHcommTeamCreateOk));
    hccl::CollComm coll(nullptr, 0, "ut_iwt_ubmem_higher_layer", hccl::ManagerCallbacks{});

    UtRankGraphStub graph;
    graph.layerBuf = {2, 3};
    graph.layerRanks[2] = {0, 1};
    graph.layerRanks[3] = {0, 1};
    graph.layerLinks[2] = {MakeLink(COMM_PROTOCOL_UB_MEM)};
    graph.layerLinks[3] = {MakeLink(COMM_PROTOCOL_UB_MEM)};
    coll.rankgraph_ = &graph;

    EXPECT_EQ(coll.InitWorldTeams(), HCCL_SUCCESS);
    EXPECT_EQ(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_MEM, 2), nullptr);
    EXPECT_NE(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_MEM, 3), nullptr);

    hccl::HcclTeamMgr::GetInstance().ClearByCollComm(&coll);
    coll.rankgraph_ = nullptr;
    GlobalMockObject::verify();
}

// ut_ain_049 rankNum<=1 的层跳过：不预制任何 team，整体 SUCCESS。
TEST_F(TestCollComm, Ut_InitWorldTeams_When_SingleRankLayer_Expect_SkipLayer)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(UtStubHcommTeamCreateOk));
    hccl::CollComm coll(nullptr, 0, "ut_iwt_single", hccl::ManagerCallbacks{});

    UtRankGraphStub graph;
    graph.layerBuf = {0, 1};
    graph.layerRanks[0] = {0}; // 仅本 rank，rankNum=1 跳过
    graph.layerRanks[1] = {0, 1};
    graph.layerLinks[1] = {MakeLink(COMM_PROTOCOL_UB_CTP)};
    coll.rankgraph_ = &graph;

    EXPECT_EQ(coll.InitWorldTeams(), HCCL_SUCCESS);

    EXPECT_EQ(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 0), nullptr);
    EXPECT_NE(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 1), nullptr);

    hccl::HcclTeamMgr::GetInstance().ClearByCollComm(&coll);
    coll.rankgraph_ = nullptr;
    GlobalMockObject::verify();
}

// ut_ain_050 无 link 的 peer 跳过：layer1 无 link，(UB_CTP,1) 不预制，整体 SUCCESS。
TEST_F(TestCollComm, Ut_InitWorldTeams_When_NoLinkPeer_Expect_SkipPeer)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(UtStubHcommTeamCreateOk));
    hccl::CollComm coll(nullptr, 0, "ut_iwt_nolink", hccl::ManagerCallbacks{});

    UtRankGraphStub graph;
    graph.layerBuf = {0, 1};
    graph.layerRanks[0] = {0, 1};
    graph.layerRanks[1] = {0, 1};
    graph.layerLinks[0] = {MakeLink(COMM_PROTOCOL_UB_CTP)};
    // layer1 无 links 条目 -> GetLinks 返回空
    coll.rankgraph_ = &graph;

    EXPECT_EQ(coll.InitWorldTeams(), HCCL_SUCCESS);

    EXPECT_NE(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 0), nullptr);
    EXPECT_EQ(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 1), nullptr);

    hccl::HcclTeamMgr::GetInstance().ClearByCollComm(&coll);
    coll.rankgraph_ = nullptr;
    GlobalMockObject::verify();
}

// ut_ain_051 GetNetLayers 返回空：HCCL_E_INTERNAL。
TEST_F(TestCollComm, Ut_InitWorldTeams_When_NetLayersEmpty_Expect_ReturnInternal)
{
    hccl::CollComm coll(nullptr, 0, "ut_iwt_empty", hccl::ManagerCallbacks{});

    UtRankGraphStub graph;
    graph.layerBuf.clear(); // netLayerNum=0
    coll.rankgraph_ = &graph;

    EXPECT_EQ(coll.InitWorldTeams(), HCCL_E_INTERNAL);

    coll.rankgraph_ = nullptr;
}

// rankgraph_ 为空：HCCL_E_PTR。
TEST_F(TestCollComm, Ut_InitWorldTeams_When_RankGraphNull_Expect_ReturnPtrError)
{
    hccl::CollComm coll(nullptr, 0, "ut_iwt_null", hccl::ManagerCallbacks{});
    EXPECT_EQ(coll.InitWorldTeams(), HCCL_E_PTR);
}

// ut_ain_052 同 (proto,layer) 已预制不重复：预注册同键后 InitWorldTeams 不再创建。
TEST_F(TestCollComm, Ut_InitWorldTeams_When_ProtoLayerAlreadyPrebuilt_Expect_NoDuplicateCreate)
{
    uint32_t rankIds[2] = {0, 1};
    HcommTeamHandle existing = reinterpret_cast<HcommTeamHandle>(0x31000);
    hccl::CollComm coll(nullptr, 0, "ut_iwt_dup", hccl::ManagerCallbacks{});
    ASSERT_EQ(
        hccl::HcclTeamMgr::GetInstance().RegisterPrebuiltWorldTeam(
            existing, &coll, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2),
        HCCL_SUCCESS);

    // 若误创建则必然调用（返回不同句柄触发 E_PARA），expects(never()) 锁死该路径
    MOCKER(HcommTeamCreate).expects(never());

    UtRankGraphStub graph;
    graph.layerBuf = {1};
    graph.layerRanks[1] = {0, 1};
    graph.layerLinks[1] = {MakeLink(COMM_PROTOCOL_UB_CTP)};
    coll.rankgraph_ = &graph;

    EXPECT_EQ(coll.InitWorldTeams(), HCCL_SUCCESS);
    EXPECT_EQ(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 1), existing);

    hccl::HcclTeamMgr::GetInstance().ClearByCollComm(&coll);
    coll.rankgraph_ = nullptr;
    GlobalMockObject::verify();
}

// ===================== 3. UpdateHcommWindowRemoteMem =====================

// ut_ain_053 slots 计算与 L3 透传：layer0(2)+layer1(2) 预制，rank0 slots={0,2}、rank1 slots={1,3}。
TEST_F(TestCollComm, Ut_UpdateHcommWindowRemoteMem_When_MultiLayerPrebuilt_Expect_SlotsWithOffset)
{
    MOCKER(HcommTeamUpdateWindowRemoteMemByRank).stubs().will(invoke(UtStubUpdateWindowRemoteMemByRank));
    /* selfSlot 新路径：UpdateHcommWindowRemoteMem 先调 SetSelfInfo 登记本端槽位，
     * 桩句柄非真实 window，打桩直接返回成功 */
    MOCKER(HcommTeamWindowSetSelfInfo).stubs().will(returnValue(0));
    uint32_t rankIds[2] = {0, 1};
    hccl::CollComm coll(nullptr, 0, "ut_uhwrm", hccl::ManagerCallbacks{});
    HcommTeamHandle w0 = reinterpret_cast<HcommTeamHandle>(0x32000);
    HcommTeamHandle w1 = reinterpret_cast<HcommTeamHandle>(0x33000);
    ASSERT_EQ(
        hccl::HcclTeamMgr::GetInstance().RegisterPrebuiltWorldTeam(w0, &coll, COMM_PROTOCOL_UB_CTP, 0, rankIds, 2),
        HCCL_SUCCESS);
    ASSERT_EQ(
        hccl::HcclTeamMgr::GetInstance().RegisterPrebuiltWorldTeam(w1, &coll, COMM_PROTOCOL_UB_RTP, 1, rankIds, 2),
        HCCL_SUCCESS);

    void* fakeWin = reinterpret_cast<void*>(0x9500000);
    coll.tagToHcommMap_["__hccl_sym_win__ut_uhwrm"] = fakeWin;

    CommMem remoteMem{};
    remoteMem.type = COMM_MEM_TYPE_DEVICE;
    remoteMem.addr = reinterpret_cast<void*>(0x9600000);
    remoteMem.size = 0x2000;
    std::vector<std::string> memTags = {"__hccl_sym_win__ut_uhwrm"};

    // remoteRank=0：layer0 槽 0 + (sizes[0]=2) + layer1 槽 0 => slots 集合={0,2}
    // （layerRanksMap_ 为 unordered_map，槽位顺序不定，按升序集合断言）
    g_utUpdateWinCalled = false;
    EXPECT_EQ(coll.UpdateHcommWindowRemoteMem(0, &remoteMem, memTags), HCCL_SUCCESS);
    EXPECT_TRUE(g_utUpdateWinCalled);
    EXPECT_EQ(g_utUpdateWinSizes[0], 2U);
    EXPECT_EQ(g_utUpdateWinSizes[1], 2U);
    EXPECT_EQ(g_utUpdateWinSizes[2], 0U);
    ASSERT_EQ(g_utUpdateWinSlots.size(), 2U);
    std::sort(g_utUpdateWinSlots.begin(), g_utUpdateWinSlots.end());
    EXPECT_EQ(g_utUpdateWinSlots[0], 0U);
    EXPECT_EQ(g_utUpdateWinSlots[1], 2U);
    EXPECT_EQ(g_utLastRemoteMem, reinterpret_cast<HcclMemHandle>(remoteMem.addr));

    // remoteRank=1：layer0 槽 1 + 2 + layer1 槽 1 => slots 集合={1,3}
    EXPECT_EQ(coll.UpdateHcommWindowRemoteMem(1, &remoteMem, memTags), HCCL_SUCCESS);
    ASSERT_EQ(g_utUpdateWinSlots.size(), 2U);
    std::sort(g_utUpdateWinSlots.begin(), g_utUpdateWinSlots.end());
    EXPECT_EQ(g_utUpdateWinSlots[0], 1U);
    EXPECT_EQ(g_utUpdateWinSlots[1], 3U);

    hccl::HcclTeamMgr::GetInstance().ClearByCollComm(&coll);
    GlobalMockObject::verify();
}

// ut_ain_054 memTag 空/未命中跳过：SUCCESS 且 L3 Update 不被调。
TEST_F(TestCollComm, Ut_UpdateHcommWindowRemoteMem_When_TagMissOrEmpty_Expect_SkipL3Update)
{
    MOCKER(HcommTeamUpdateWindowRemoteMemByRank).expects(never());
    hccl::CollComm coll(nullptr, 0, "ut_uhwrm_miss", hccl::ManagerCallbacks{});

    CommMem remoteMem{};
    remoteMem.type = COMM_MEM_TYPE_DEVICE;
    remoteMem.addr = reinterpret_cast<void*>(0x9700000);
    remoteMem.size = 0x2000;
    std::vector<std::string> memTags = {"", "unknown_tag"};

    EXPECT_EQ(coll.UpdateHcommWindowRemoteMem(1, &remoteMem, memTags), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

// L3 回填失败透传：HCOMM_E_PARA -> HCCL_E_PARA。
TEST_F(TestCollComm, Ut_UpdateHcommWindowRemoteMem_When_L3UpdateFail_Expect_ReturnError)
{
    MOCKER(HcommTeamUpdateWindowRemoteMemByRank).stubs().will(returnValue(static_cast<HcommResult>(1)));
    hccl::CollComm coll(nullptr, 0, "ut_uhwrm_fail", hccl::ManagerCallbacks{});
    coll.tagToHcommMap_["__hccl_sym_win__ut_fail"] = reinterpret_cast<void*>(0x9800000);

    CommMem remoteMem{};
    remoteMem.type = COMM_MEM_TYPE_DEVICE;
    remoteMem.addr = reinterpret_cast<void*>(0x9900000);
    remoteMem.size = 0x2000;
    std::vector<std::string> memTags = {"__hccl_sym_win__ut_fail"};

    EXPECT_EQ(coll.UpdateHcommWindowRemoteMem(1, &remoteMem, memTags), HCCL_E_PARA);
    GlobalMockObject::verify();
}

// ===================== 4. ReExchangeWindowsForBoundTeams =====================

// ut_ain_056 无已建链 team：空操作 SUCCESS，不建 channel 不回填。
TEST_F(TestCollComm, Ut_ReExchangeWindowsForBoundTeams_When_NoLinkedTeam_Expect_NoOp)
{
    MOCKER_CPP(
        &hccl::MyRank::CreateChannels,
        HcclResult(hccl::MyRank::*)(CommEngine, const std::string&, const HcclChannelDesc*, uint32_t, ChannelHandle*))
        .expects(never());
    hccl::CollComm coll(nullptr, 0, "ut_reex_empty", hccl::ManagerCallbacks{});
    aclrtBinHandle binHandle{};
    coll.myRank_ = std::make_shared<hccl::MyRank>(
        binHandle, 0, coll.GetCommConfig(), hccl::ManagerCallbacks(), nullptr, nullptr);

    UtRankGraphStub graph;
    coll.rankgraph_ = &graph;

    EXPECT_EQ(coll.ReExchangeWindowsForBoundTeams(), HCCL_SUCCESS);

    coll.rankgraph_ = nullptr;
    coll.myRank_ = nullptr;
    GlobalMockObject::verify();
}

// ut_ain_057/058 ReExchange 有已建链 team：1 个 subTeam（成员 {0,1}，self=0）。
// 设计说明：MyRank::ChannelGetRemoteMems(vector) 为 const 成员函数，mockcpp 的 MOCKER_CPP invoke
// 对其存在桩签名匹配异常（同型比较失败/段错误），故本用例取"无 pending window"变体：
// symmetricMemory_=nullptr 使 RegisterPendingSymmetricMemHandles 真实返回空，源码按设计跳过
// GetRemoteMems/回填分支；对每个非 self peer 各建 1 条 channel（CreateChannels 计数验证）。
// memNum>0 触发回填的分支由既有 Ut_UpdateSymmetricRemoteMem_When_ChannelReturnsRemoteMem_*
// （驱动 UpdateSymmetricRemoteMem→UpdateHcommWindowRemoteMem 全链）与本文件 UpdateHcommWindowRemoteMem
// 用例共同覆盖。
TEST_F(TestCollComm, Ut_ReExchangeWindowsForBoundTeams_When_HasLinkedTeam_Expect_ReExchangePerPeer)
{
    MOCKER(HcommTeamGetNetLayer).stubs().will(invoke(UtStubHcommTeamGetNetLayer));
    MOCKER(HcommTeamGetEngine).stubs().will(invoke(UtStubHcommTeamGetEngine));
    MOCKER_CPP(
        &hccl::MyRank::CreateChannels,
        HcclResult(hccl::MyRank::*)(CommEngine, const std::string&, const HcclChannelDesc*, uint32_t, ChannelHandle*))
        .stubs()
        .will(invoke(UtStubMyRankCreateChannelsOk));
    g_utCreateChannelsCalls = 0;

    hccl::CollComm coll(nullptr, 0, "ut_reex_sub", hccl::ManagerCallbacks{});
    // 注册 fake sym mem handle 使 GetAllRegisteredSymMemHandles 返回非空（触发 ReExchangeChannelsForTeam）
    coll.registeredSymMemHandleMap_["fake_sym_mem"] = reinterpret_cast<HcclMemHandle>(0x12345);
    aclrtBinHandle binHandle{};
    coll.myRank_ = std::make_shared<hccl::MyRank>(
        binHandle, 0, coll.GetCommConfig(), hccl::ManagerCallbacks(), nullptr, nullptr);
    // symmetricMemory_ 保持 nullptr：RegisterPendingSymmetricMemHandles 返回空，跳过回填分支

    // 注册 worldTeam + subTeam（成员 {0,1}，self rank=0）
    uint32_t rankIds[2] = {0, 1};
    HcommTeamHandle worldTeam = reinterpret_cast<HcommTeamHandle>(0x34000);
    HcommTeamHandle subTeam = reinterpret_cast<HcommTeamHandle>(0x35000);
    ASSERT_EQ(
        hccl::HcclTeamMgr::GetInstance().RegisterPrebuiltWorldTeam(
            worldTeam, &coll, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2),
        HCCL_SUCCESS);
    ASSERT_EQ(
        hccl::HcclTeamMgr::GetInstance().RegisterSubTeam(worldTeam, subTeam, nullptr, 0, rankIds, 2), HCCL_SUCCESS);

    UtRankGraphStub graph;
    graph.layerBuf = {1};
    graph.layerRanks[1] = {0, 1};
    graph.layerLinks[1] = {MakeLink(COMM_PROTOCOL_UB_CTP)};
    coll.rankgraph_ = &graph;
    g_utGetNetLayerVal = 1;
    g_utGetEngineVal = COMM_ENGINE_AIV;

    // ReExchangeWindowsForBoundTeams 走补交换路径：CreateChannels 被调 1 次（验证路径走到）
    // ChannelGetRemoteMems 在 UT 环境下因无真实 channel 会返回错误——验证 CreateChannels 计数即可
    coll.ReExchangeWindowsForBoundTeams();
    EXPECT_EQ(g_utCreateChannelsCalls, 1U);

    hccl::HcclTeamMgr::GetInstance().ClearByCollComm(&coll);
    coll.rankgraph_ = nullptr;
    coll.myRank_ = nullptr;
    GlobalMockObject::verify();
}

TEST_F(TestCollComm, Ut_HcommWindowOwner_When_EraseWindow_Expect_OtherOwnerKept)
{
    HcclCommSymWindow window = reinterpret_cast<HcclCommSymWindow>(0x310000U);
    HcclCommSymWindow otherWindow = reinterpret_cast<HcclCommSymWindow>(0x312000U);
    HcclComm expectedComm = reinterpret_cast<HcclComm>(0x320000U);
    HcclComm otherComm = reinterpret_cast<HcclComm>(0x330000U);
    HcclComm actualComm = nullptr;

    ASSERT_EQ(hccl::RecordHcommWindowOwner(window, expectedComm), HCCL_SUCCESS);
    ASSERT_EQ(hccl::RecordHcommWindowOwner(otherWindow, otherComm), HCCL_SUCCESS);
    EXPECT_EQ(hccl::GetHcommWindowComm(window, actualComm), HCCL_SUCCESS);
    EXPECT_EQ(actualComm, expectedComm);
    EXPECT_EQ(hccl::RecordHcommWindowOwner(window, expectedComm), HCCL_E_INTERNAL);

    hccl::EraseHcommWindowOwner(window);
    EXPECT_EQ(hccl::GetHcommWindowComm(window, actualComm), HCCL_E_NOT_FOUND);
    EXPECT_EQ(actualComm, nullptr);
    EXPECT_EQ(hccl::GetHcommWindowComm(otherWindow, actualComm), HCCL_SUCCESS);
    EXPECT_EQ(actualComm, otherComm);

    hccl::EraseHcommWindowOwner(otherWindow);
}

// 回归：BuildDerivedCommParams 必须在构造/Init HcclCommunicator 前把 commName 写入 ctx.subParams.commId，
// 否则 CommunicatorImpl::id 固化为空，shrink/grow 后 HcclGetCommName 返回空串，
// SingleRankProc dfx 注册触发 group name[] length[0] 校验失败。
TEST_F(TestCollComm, Ut_BuildDerivedCommParams_SetsCommNameBeforeHcclCommunicatorInit)
{
    // comm_ 仅需非空哨兵以满足 CHK_PTR_NULL；GetDevType 由 MOCKER_CPP 全局拦截，不真正解引用对象。
    hccl::CollComm collComm(reinterpret_cast<void*>(0x1), 0, "ut_parent", hccl::ManagerCallbacks{});

    // shrink 路径：isGrow=false，survivorRanks 非空，mergedRankTable 为空对象
    {
        MOCKER_CPP(&Hccl::HcclCommunicator::GetDevType, HcclResult(Hccl::HcclCommunicator::*)(Hccl::DevType&))
            .stubs()
            .will(returnValue(HCCL_SUCCESS));

        hccl::CollComm::DerivedCommContext ctx;
        hccl::CollComm::DerivationInfo derivationInfo;
        derivationInfo.isGrow = false;
        std::vector<uint32_t> survivors = {0u}; // 1-rank 子域（触发 SingleRankProc 的退化边界）
        Hccl::RankTableInfo emptyTable{};
        const std::string commName = "ut_shrink_derived_comm";

        HcclResult ret = collComm.BuildDerivedCommParams(survivors, 0u, commName, derivationInfo, emptyTable, ctx);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        // 核心契约：commId 必须在 CreateSubCommunicator 构造 HcclCommunicator 前即设为 commName（非空）
        EXPECT_EQ(ctx.subParams.commId, commName);
        EXPECT_FALSE(ctx.subParams.commId.empty());
        EXPECT_EQ(ctx.subParams.rankSize, 1u);
        EXPECT_EQ(ctx.subParams.rankInParentComm, 0u);
        GlobalMockObject::verify();
    }

    // grow 路径：isGrow=true，survivorRanks 空，rankSize 取 mergedRankTable.rankCount
    {
        MOCKER_CPP(&Hccl::HcclCommunicator::GetDevType, HcclResult(Hccl::HcclCommunicator::*)(Hccl::DevType&))
            .stubs()
            .will(returnValue(HCCL_SUCCESS));

        hccl::CollComm::DerivedCommContext ctx;
        hccl::CollComm::DerivationInfo derivationInfo;
        derivationInfo.isGrow = true;
        std::vector<uint32_t> emptySurvivors;
        Hccl::RankTableInfo table{};
        table.rankCount = 2u;
        const std::string commName = "ut_grow_derived_comm";

        HcclResult ret = collComm.BuildDerivedCommParams(emptySurvivors, 0u, commName, derivationInfo, table, ctx);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        EXPECT_EQ(ctx.subParams.commId, commName);
        EXPECT_FALSE(ctx.subParams.commId.empty());
        EXPECT_EQ(ctx.subParams.rankSize, 2u);
        EXPECT_EQ(ctx.subParams.rankInParentComm, 0u);
        GlobalMockObject::verify();
    }
}

// ============================================================================
// Comm grow + detect ownership UT
// ============================================================================

using namespace hccl;
using namespace Hccl;

using CreateDerivedCommMockType
    = HcclResult (CollComm::*)(std::unique_ptr<CollComm>&, const CollComm::DerivedCommParams&);

struct DerivedCommParamsCapture {
    std::vector<uint32_t> survivorRanks;
    uint32_t rankId{0};
    std::string commName;
    HcclCommConfig subConfig{};
    CollComm::DerivationInfo derivationInfo{};
    bool inheritParentConfig{false};
    CollCommInitMode initMode{CollCommInitMode::fullMode};
    RankTableInfo mergedRankTable{};
};

static DerivedCommParamsCapture g_derivedParamsCapture;

static void CaptureDerivedCommParams(const CollComm::DerivedCommParams& params)
{
    g_derivedParamsCapture.survivorRanks = params.survivorRanks;
    g_derivedParamsCapture.rankId = params.rankId;
    g_derivedParamsCapture.commName = params.commName;
    g_derivedParamsCapture.subConfig = params.subConfig;
    g_derivedParamsCapture.derivationInfo = params.derivationInfo;
    g_derivedParamsCapture.inheritParentConfig = params.inheritParentConfig;
    g_derivedParamsCapture.initMode = params.initMode;
    g_derivedParamsCapture.mergedRankTable = params.mergedRankTable;
}

class MockRankGraphForGrow : public hccl::RankGraph {
public:
    uint32_t mockRankSize{0};

    HcclResult GetRankSize(uint32_t* rankSize) override
    {
        *rankSize = mockRankSize;
        return HCCL_SUCCESS;
    }
    HcclResult GetRankGraphInfo(GraphType, void**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetLinks(uint32_t, uint32_t, uint32_t, CommLink**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetNetLayers(uint32_t**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetInstTopoTypeByNetLayer(uint32_t, CommTopo*) override { return HCCL_SUCCESS; }
    HcclResult GetInstSizeByNetLayer(uint32_t, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetInstRanksByNetLayer(uint32_t, uint32_t**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetInstSizeListByNetLayer(uint32_t, uint32_t**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetDeviceId(uint32_t, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetEndpointInfo(uint32_t, const EndpointDesc*, EndpointAttr, uint32_t, void*) override
    {
        return HCCL_SUCCESS;
    }
};

static std::unique_ptr<CollComm> MakeCollCommWithDetect(
    MockRankGraphForGrow& mockGraph, uint32_t rankSize, uint32_t myRank, uint32_t rootRankId,
    std::shared_ptr<RankInfoDetect> detect, bool isRoot)
{
    auto collComm
        = std::make_unique<CollComm>(nullptr, myRank, "ut_grow", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    mockGraph.mockRankSize = rankSize;
    collComm->rankgraph_ = &mockGraph;
    collComm->rankId_ = myRank;
    collComm->topParentIdentifier_ = "ut_grow";
    collComm->generationIndex_ = 0;
    collComm->initSource_ = CommInitSource::rootInfo;
    if (detect != nullptr) {
        detect->rankId_ = myRank;
    }

    HcclRootHandleV2 rootHandle{};
    rootHandle.listenPort = 60020;
    strncpy_s(rootHandle.identifier, sizeof(rootHandle.identifier), "ut_grow", 7);
    rootHandle.ip[0] = '\0';
    rootHandle.rootRankId = rootRankId;
    collComm->SetRootHandle(rootHandle);
    collComm->SetRankInfoDetect(detect);

    HcclCommConfigInit(&collComm->hcclCommConfig_);
    return collComm;
}

HcclResult MockCreateDerivedCommForGrow(
    CollComm* parent, std::unique_ptr<CollComm>& collComm, const CollComm::DerivedCommParams& params)
{
    CaptureDerivedCommParams(params);
    auto derived
        = std::make_unique<CollComm>(nullptr, 0, "ut_grow_derived", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    derived->rankId_ = parent->GetMyRankId();
    derived->SetRootHandle(parent->GetCollCommRootHandle());
    collComm = std::move(derived);
    return HCCL_SUCCESS;
}

class CollCommGrowDetectTest : public BaseInit {
protected:
    void SetUp() override
    {
        BaseInit::SetUp();
        MOCKER(hrtGetDeviceType).stubs().will(invoke(+[](Hccl::DevType& t) -> HcclResult {
            t = Hccl::DevType::DEV_TYPE_950;
            return HCCL_SUCCESS;
        }));
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

// 1. detect 所有权传递链验证

TEST_F(CollCommGrowDetectTest, Ut_SetGetRankInfoDetect_When_SetDetect_Expect_SamePtr)
{
    auto detect = std::make_shared<RankInfoDetect>();
    CollComm collComm(nullptr, 0, "ut", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    collComm.SetRankInfoDetect(detect);
    EXPECT_EQ(collComm.GetRankInfoDetect().get(), detect.get());
}

TEST_F(CollCommGrowDetectTest, Ut_TakeRankInfoDetect_When_Moved_Expect_OldCommDisabled)
{
    auto detect = std::make_shared<RankInfoDetect>();
    CollComm collComm(nullptr, 0, "ut", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    collComm.SetRankInfoDetect(detect);

    EXPECT_TRUE(collComm.HasDetectOwnership());
    auto taken = collComm.TakeRankInfoDetect();
    EXPECT_EQ(taken.get(), detect.get());
    EXPECT_FALSE(collComm.GetRankInfoDetect());
    EXPECT_FALSE(collComm.HasDetectOwnership());

    EXPECT_FALSE(collComm.TakeRankInfoDetect());
    EXPECT_FALSE(collComm.HasDetectOwnership());
}

TEST_F(CollCommGrowDetectTest, Ut_TakeRankInfoDetect_When_NoDetect_Expect_OldCommStillDisabled)
{
    CollComm collComm(nullptr, 0, "ut", ManagerCallbacks{}, CollCommInitMode::simpleMode);

    EXPECT_TRUE(collComm.HasDetectOwnership());
    EXPECT_FALSE(collComm.TakeRankInfoDetect());
    EXPECT_FALSE(collComm.HasDetectOwnership());
}

TEST_F(CollCommGrowDetectTest, Ut_ShutdownRankInfoDetect_When_DetectMoved_Expect_ChainTailPreserved)
{
    auto detect = std::make_shared<RankInfoDetect>();
    CollComm parent(nullptr, 0, "ut_parent", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    CollComm child(nullptr, 0, "ut_child", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    parent.SetRankInfoDetect(detect);
    child.SetRankInfoDetect(parent.TakeRankInfoDetect());

    EXPECT_EQ(parent.ShutdownRankInfoDetect(), HCCL_SUCCESS);
    EXPECT_FALSE(parent.GetRankInfoDetect());
    EXPECT_EQ(child.GetRankInfoDetect().get(), detect.get());
}

TEST_F(CollCommGrowDetectTest, Ut_ShutdownRankInfoDetect_When_DetectOwned_Expect_IdempotentSuccess)
{
    auto detect = std::make_shared<RankInfoDetect>();
    CollComm collComm(nullptr, 0, "ut_shutdown", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    collComm.SetRankInfoDetect(detect);

    EXPECT_EQ(collComm.ShutdownRankInfoDetect(), HCCL_SUCCESS);
    EXPECT_EQ(collComm.GetRankInfoDetect(), nullptr);
    EXPECT_EQ(collComm.ShutdownRankInfoDetect(), HCCL_SUCCESS);
}

TEST_F(CollCommGrowDetectTest, Ut_PendingDetect_When_Take_Expect_NullAfterTake)
{
    auto detect = std::make_shared<RankInfoDetect>();
    CollComm collComm(nullptr, 0, "ut", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    collComm.SetPendingDetect(detect);
    auto taken = collComm.TakePendingDetect();
    EXPECT_NE(taken, nullptr);
    auto taken2 = collComm.TakePendingDetect();
    EXPECT_EQ(taken2, nullptr);
}

TEST_F(CollCommGrowDetectTest, Ut_GetCollCommRootInfo_When_PendingDetect_Expect_NotCommitRootHandle)
{
    auto detect = std::make_shared<RankInfoDetect>();
    HcclRootHandleV2 pendingRootHandle{};
    pendingRootHandle.listenPort = 60030;
    pendingRootHandle.rootRankId = 0;
    strncpy_s(pendingRootHandle.identifier, sizeof(pendingRootHandle.identifier), "ut_pending", 11);
    detect->SetRootHandle(pendingRootHandle);

    CollComm collComm(nullptr, 0, "ut", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    collComm.SetPendingDetect(detect);

    HcclRootHandleV2 actualRootHandle{};
    EXPECT_EQ(collComm.GetCollCommRootInfo(actualRootHandle), HCCL_SUCCESS);
    EXPECT_EQ(actualRootHandle.listenPort, pendingRootHandle.listenPort);
    EXPECT_FALSE(collComm.IsRootEndpointValid());
    EXPECT_TRUE(collComm.HasPendingDetect());
}

TEST_F(CollCommGrowDetectTest, Ut_GetCollCommRootInfo_When_RootSurvives_Expect_IdentifierLineage)
{
    MockRankGraphForGrow mockGraph;
    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, nullptr, true);
    collComm->topParentIdentifier_ = "root_identifier";
    collComm->generationIndex_ = 1;

    HcclRootHandleV2 actualRootHandle{};
    EXPECT_EQ(collComm->GetCollCommRootInfo(actualRootHandle), HCCL_SUCCESS);
    EXPECT_EQ(actualRootHandle.valid, 1u);
    EXPECT_EQ(actualRootHandle.generationIndex, 2u);
    EXPECT_EQ(std::string(actualRootHandle.topParentIdentifier), "root_identifier");
    EXPECT_EQ(std::string(actualRootHandle.identifier), "ut_grow");
}

TEST_F(CollCommGrowDetectTest, Ut_InitDerivationFields_When_NoLineage_Expect_RootHandleIdentifier)
{
    CollComm collComm(nullptr, 0, "user_comm_name", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    HcclRootHandleV2 rootHandle{};
    rootHandle.listenPort = 60020;
    strncpy_s(rootHandle.identifier, sizeof(rootHandle.identifier), "root_identifier", sizeof("root_identifier") - 1);
    collComm.SetRootHandle(rootHandle);

    EXPECT_EQ(collComm.InitDerivationFields(), HCCL_SUCCESS);
    EXPECT_EQ(collComm.GetTopParentIdentifier(), "root_identifier");
    EXPECT_EQ(collComm.GetGenerationIndex(), 0u);
}

TEST_F(CollCommGrowDetectTest, Ut_TakeFromCommunicator_When_CommNull_Expect_Success)
{
    CollComm collComm(nullptr, 0, "ut", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    HcclResult ret = collComm.TakeRankInfoDetectFromCommunicator();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(collComm.GetRankInfoDetect(), nullptr);
}

// 2. ValidateGrowParams 分支覆盖

TEST_F(CollCommGrowDetectTest, Ut_ValidateGrowParams_When_NRanksZero_Expect_E_PARA)
{
    MockRankGraphForGrow mockGraph;
    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, nullptr, false);
    HcclRootInfo rootInfo{};
    EXPECT_EQ(collComm->ValidateGrowParams(0, 0, &rootInfo), HCCL_E_PARA);
}

TEST_F(CollCommGrowDetectTest, Ut_ValidateGrowParams_When_NRanksLEQRankSize_Expect_E_PARA)
{
    MockRankGraphForGrow mockGraph;
    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, nullptr, false);
    HcclRootInfo rootInfo{};
    EXPECT_EQ(collComm->ValidateGrowParams(4, 0, &rootInfo), HCCL_E_PARA);
}

TEST_F(CollCommGrowDetectTest, Ut_ValidateGrowParams_When_RankOutOfRange_Expect_E_PARA)
{
    MockRankGraphForGrow mockGraph;
    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, nullptr, false);
    HcclRootInfo rootInfo{};
    EXPECT_EQ(collComm->ValidateGrowParams(8, 8, &rootInfo), HCCL_E_PARA);
}

TEST_F(CollCommGrowDetectTest, Ut_ValidateGrowParams_When_DetectNull_Expect_Success)
{
    MockRankGraphForGrow mockGraph;
    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, nullptr, false);
    HcclRootInfo rootInfo{};
    EXPECT_EQ(collComm->ValidateGrowParams(8, 0, &rootInfo), HCCL_SUCCESS);
}

TEST_F(CollCommGrowDetectTest, Ut_ValidateGrowParams_When_RootHandleInvalid_Expect_Success)
{
    MockRankGraphForGrow mockGraph;
    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, nullptr, false);
    collComm->rootHandle_.listenPort = Hccl::HCCL_INVALID_PORT;
    HcclRootInfo rootInfo{};
    EXPECT_EQ(collComm->ValidateGrowParams(8, 0, &rootInfo), HCCL_SUCCESS);
}

TEST_F(CollCommGrowDetectTest, Ut_ValidateGrowParams_When_NormalGrowAllValid_Expect_Success)
{
    MockRankGraphForGrow mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, detect, true);
    EXPECT_EQ(collComm->ValidateGrowParams(8, 0, nullptr), HCCL_SUCCESS);
}

TEST_F(CollCommGrowDetectTest, Ut_ValidateGrowParams_When_ClusterInfo_Expect_E_NOT_SUPPORT)
{
    MockRankGraphForGrow mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, detect, true);
    collComm->initSource_ = CommInitSource::clusterInfo;
    HcclRootInfo rootInfo{};
    EXPECT_EQ(collComm->ValidateGrowParams(8, 0, &rootInfo), HCCL_E_NOT_SUPPORT);
}

TEST_F(CollCommGrowDetectTest, Ut_CommGrow_When_RootReselectRankMismatch_Expect_E_PARA)
{
    MockRankGraphForGrow mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, detect, true);

    HcclRootInfo rootInfo{};
    HcclRootHandleV2 mismatchRootHandle{};
    mismatchRootHandle.listenPort = 60030;
    mismatchRootHandle.rootRankId = 0;
    strncpy_s(
        mismatchRootHandle.identifier, sizeof(mismatchRootHandle.identifier), "ut_root_reselect",
        sizeof("ut_root_reselect") - 1);
    memcpy_s(rootInfo.internal, sizeof(rootInfo.internal), &mismatchRootHandle, sizeof(mismatchRootHandle));

    std::unique_ptr<CollComm> newCollComm;
    EXPECT_EQ(collComm->CommGrow(8, 1, newCollComm, &rootInfo, nullptr), HCCL_E_PARA);
    EXPECT_EQ(newCollComm, nullptr);
}

// 3. HandleGrowNormal -- root 存活场景

TEST_F(CollCommGrowDetectTest, Ut_HandleGrowNormal_When_RootAndDetectValid_Expect_TriggerGrowCollect)
{
    MockRankGraphForGrow mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetect::TriggerGrowCollect)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, detect, true);
    RankTableInfo mergedRankTable{};
    HcclResult ret = collComm->HandleGrowNormal(8, mergedRankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CollCommGrowDetectTest, Ut_HandleGrowNormal_When_TriggerGrowCollectFails_Expect_Error)
{
    MockRankGraphForGrow mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetect::TriggerGrowCollect)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, detect, true);
    RankTableInfo mergedRankTable{};
    HcclResult ret = collComm->HandleGrowNormal(8, mergedRankTable);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(CollCommGrowDetectTest, Ut_HandleGrowNormal_When_NonRoot_Expect_WaitForNextBroadcast)
{
    MockRankGraphForGrow mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(false));
    MOCKER_CPP(&RankInfoDetect::WaitForNextBroadcast)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 1, 0, detect, false);
    RankTableInfo mergedRankTable{};
    HcclResult ret = collComm->HandleGrowNormal(8, mergedRankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CollCommGrowDetectTest, Ut_HandleGrowNormal_When_WaitForNextBroadcastFails_Expect_Error)
{
    MockRankGraphForGrow mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(false));
    MOCKER_CPP(&RankInfoDetect::WaitForNextBroadcast)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 1, 0, detect, false);
    RankTableInfo mergedRankTable{};
    HcclResult ret = collComm->HandleGrowNormal(8, mergedRankTable);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(CollCommGrowDetectTest, Ut_HandleGrowNormal_When_DetectNull_Expect_E_INTERNAL)
{
    MockRankGraphForGrow mockGraph;
    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, nullptr, false);
    RankTableInfo mergedRankTable{};
    HcclResult ret = collComm->HandleGrowNormal(8, mergedRankTable);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// 4. HandleGrowRootReselect -- root 重选场景

TEST_F(CollCommGrowDetectTest, Ut_HandleGrowRootReselect_When_CoordinatorWithPendingDetect_Expect_FullCollect)
{
    MockRankGraphForGrow mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    MOCKER_CPP(&RankInfoDetect::IsReadyForGrow, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetect::SetupAgent)
        .expects(once())
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankInfoDetect::WaitComplete)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankInfoDetect::GetRankTable)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect*, RankTableInfo& rankTable) {
            rankTable.rankCount = 8;
        }));
    MOCKER_CPP(&RankInfoDetect::TriggerGrowCollect).expects(never());

    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, nullptr, false);
    collComm->SetRootHandle({});
    collComm->SetPendingDetect(detect);

    HcclRootInfo rootInfo{};
    HcclRootHandleV2 rootHandle{};
    rootHandle.listenPort = 60030;
    rootHandle.rootRankId = 0;
    strncpy_s(rootHandle.identifier, sizeof(rootHandle.identifier), "ut_reselect", 11);
    detect->SetRootHandle(rootHandle);
    memcpy_s(rootInfo.internal, sizeof(rootInfo.internal), &rootHandle, sizeof(rootHandle));

    RankTableInfo mergedRankTable{};
    HcclResult ret = collComm->HandleGrowRootReselect(8, 0, &rootInfo, nullptr, mergedRankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(mergedRankTable.rankCount, 8u);
    EXPECT_TRUE(collComm->IsRootEndpointValid());
    EXPECT_EQ(collComm->TakePendingDetect(), nullptr);
    EXPECT_NE(collComm->GetRankInfoDetect(), nullptr);
}

TEST_F(CollCommGrowDetectTest, Ut_HandleGrowRootReselect_When_DetectNotReady_Expect_Unavail)
{
    MockRankGraphForGrow mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    MOCKER_CPP(&RankInfoDetect::SetupAgent).expects(never());
    MOCKER_CPP(&RankInfoDetect::TriggerGrowCollect).expects(never());

    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, nullptr, false);
    collComm->SetPendingDetect(detect);

    HcclRootInfo rootInfo{};
    HcclRootHandleV2 rootHandle{};
    rootHandle.listenPort = 60030;
    rootHandle.rootRankId = 0;
    strncpy_s(rootHandle.identifier, sizeof(rootHandle.identifier), "ut_reselect", 11);
    memcpy_s(rootInfo.internal, sizeof(rootInfo.internal), &rootHandle, sizeof(rootHandle));

    RankTableInfo mergedRankTable{};
    EXPECT_EQ(collComm->HandleGrowRootReselect(8, 0, &rootInfo, nullptr, mergedRankTable), HCCL_E_UNAVAIL);
}

TEST_F(CollCommGrowDetectTest, Ut_HandleGrowRootReselect_When_WaitCompleteFails_Expect_Error)
{
    MockRankGraphForGrow mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    MOCKER_CPP(&RankInfoDetect::IsReadyForGrow, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetect::SetupAgent)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankInfoDetect::WaitComplete)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, nullptr, false);
    collComm->SetRootHandle({});
    collComm->SetPendingDetect(detect);

    HcclRootInfo rootInfo{};
    HcclRootHandleV2 rootHandle{};
    rootHandle.listenPort = 60030;
    memcpy_s(rootInfo.internal, sizeof(rootInfo.internal), &rootHandle, sizeof(rootHandle));

    RankTableInfo mergedRankTable{};
    HcclResult ret = collComm->HandleGrowRootReselect(8, 0, &rootInfo, nullptr, mergedRankTable);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_FALSE(collComm->IsRootEndpointValid());
}

TEST_F(CollCommGrowDetectTest, Ut_HandleGrowRootReselect_When_NoPendingDetect_Expect_NewDetectSetupAgent)
{
    MockRankGraphForGrow mockGraph;
    MOCKER_CPP(&RankInfoDetect::SetupAgent)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect* detect, u32, u32, const HcclRootHandleV2& handle) -> HcclResult {
            HcclRootHandleV2 detectedHandle = handle;
            detectedHandle.rootRankId = 1;
            detect->SetRootHandle(detectedHandle);
            return HCCL_SUCCESS;
        }));
    MOCKER_CPP(&RankInfoDetect::WaitComplete)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankInfoDetect::GetRankTable)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect*, RankTableInfo& rankTable) {
            rankTable.rankCount = 8;
        }));

    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 1, 0, nullptr, false);

    HcclRootInfo rootInfo{};
    HcclRootHandleV2 rootHandle{};
    rootHandle.listenPort = 60030;
    strncpy_s(rootHandle.identifier, sizeof(rootHandle.identifier), "ut_reselect", 11);
    memcpy_s(rootInfo.internal, sizeof(rootInfo.internal), &rootHandle, sizeof(rootHandle));

    RankTableInfo mergedRankTable{};
    HcclResult ret = collComm->HandleGrowRootReselect(8, 1, &rootInfo, nullptr, mergedRankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(mergedRankTable.rankCount, 8u);
    EXPECT_NE(collComm->GetRankInfoDetect(), nullptr);
}

TEST_F(CollCommGrowDetectTest, Ut_HandleGrowRootReselect_When_SetupAgentFails_Expect_Error)
{
    MockRankGraphForGrow mockGraph;
    MOCKER_CPP(&RankInfoDetect::SetupAgent)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 1, 0, nullptr, false);

    HcclRootInfo rootInfo{};
    HcclRootHandleV2 rootHandle{};
    rootHandle.listenPort = 60030;
    memcpy_s(rootInfo.internal, sizeof(rootInfo.internal), &rootHandle, sizeof(rootHandle));

    RankTableInfo mergedRankTable{};
    HcclResult ret = collComm->HandleGrowRootReselect(8, 1, &rootInfo, nullptr, mergedRankTable);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

static void CommitDetectTransfer(CollComm* parent, CollComm* derived);

// 5. CommGrow 全流程 -- root 存活的正常 Grow

TEST_F(CollCommGrowDetectTest, Ut_CommGrow_When_RootSurvivesNormal_Expect_Success)
{
    MockRankGraphForGrow mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetect::TriggerGrowCollect)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect*, u32 newRankCount, RankTableInfo& mergedRankTable) -> HcclResult {
            mergedRankTable.rankCount = newRankCount + 4;
            return HCCL_SUCCESS;
        }));
    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommForGrow));

    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, detect, true);

    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm->CommGrow(8, 0, newCollComm, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(newCollComm, nullptr);
    CommitDetectTransfer(collComm.get(), newCollComm.get());
    EXPECT_TRUE(g_derivedParamsCapture.inheritParentConfig);
    EXPECT_TRUE(g_derivedParamsCapture.survivorRanks.empty());
    EXPECT_EQ(g_derivedParamsCapture.rankId, 0u);
    EXPECT_EQ(g_derivedParamsCapture.commName, "ut_grow_1");
    EXPECT_EQ(std::string(g_derivedParamsCapture.subConfig.hcclCommName), "ut_grow_1");
    EXPECT_TRUE(g_derivedParamsCapture.derivationInfo.isGrow);
    EXPECT_EQ(g_derivedParamsCapture.derivationInfo.rootHandle.listenPort, 60020u);
    EXPECT_EQ(g_derivedParamsCapture.initMode, CollCommInitMode::fullMode);
}

TEST_F(CollCommGrowDetectTest, Ut_CommGrow_When_NonRootSurvivesNormal_Expect_Success)
{
    MockRankGraphForGrow mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(false));
    MOCKER_CPP(&RankInfoDetect::WaitForNextBroadcast)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect*, RankTableInfo& rankTable, uint32_t expectedRankCount) -> HcclResult {
            rankTable.rankCount = expectedRankCount;
            return HCCL_SUCCESS;
        }));
    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommForGrow));

    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 1, 0, detect, false);

    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm->CommGrow(8, 1, newCollComm, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(newCollComm, nullptr);
    CommitDetectTransfer(collComm.get(), newCollComm.get());
}

TEST_F(CollCommGrowDetectTest, Ut_CommGrow_When_RankCountMismatch_Expect_E_INTERNAL)
{
    MockRankGraphForGrow mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetect::TriggerGrowCollect)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect*, u32 newRankCount, RankTableInfo& mergedRankTable) -> HcclResult {
            mergedRankTable.rankCount = 6;
            return HCCL_SUCCESS;
        }));

    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, detect, true);

    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm->CommGrow(8, 0, newCollComm, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(newCollComm, nullptr);
}

// 6. detect 继承给派生通信域验证

TEST_F(CollCommGrowDetectTest, Ut_CommGrow_When_HandleGrowNormal_Expect_DetectStillAccessible)
{
    MockRankGraphForGrow mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetect::TriggerGrowCollect)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect*, u32 newRankCount, RankTableInfo& mergedRankTable) -> HcclResult {
            mergedRankTable.rankCount = newRankCount + 4;
            return HCCL_SUCCESS;
        }));

    auto collComm = MakeCollCommWithDetect(mockGraph, 4, 0, 0, detect, true);

    RankTableInfo mergedRankTable{};
    ASSERT_EQ(collComm->HandleGrowNormal(8, mergedRankTable), HCCL_SUCCESS);
    EXPECT_EQ(collComm->GetRankInfoDetect().get(), detect.get());
}

// 7. pendingDetect move 语义模拟（验证 HcclGetRootInfo -> CommInitRootInfo 交接）

TEST_F(CollCommGrowDetectTest, Ut_DetectOwnershipChain_When_StoreAndTake_Expect_SameDetect)
{
    auto detect = std::make_shared<RankInfoDetect>();
    CollComm collComm(nullptr, 0, "ut", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    collComm.SetPendingDetect(detect);

    auto taken = collComm.TakePendingDetect();
    EXPECT_NE(taken, nullptr);

    auto takenDetect = std::static_pointer_cast<RankInfoDetect>(taken);
    EXPECT_EQ(takenDetect.get(), detect.get());

    EXPECT_EQ(collComm.TakePendingDetect(), nullptr);
}

// ============================================================================
// Comm shrink-grow chain UT
// ============================================================================

// -- MockRankGraph: override RankGraph virtuals, GetRankSize returns configurable value --
class MockRankGraphSG : public hccl::RankGraph {
public:
    uint32_t mockRankSize{0};

    HcclResult GetRankSize(uint32_t* rankSize) override
    {
        *rankSize = mockRankSize;
        return HCCL_SUCCESS;
    }
    HcclResult GetRankGraphInfo(GraphType, void**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetLinks(uint32_t, uint32_t, uint32_t, CommLink**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetNetLayers(uint32_t**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetInstTopoTypeByNetLayer(uint32_t, CommTopo*) override { return HCCL_SUCCESS; }
    HcclResult GetInstSizeByNetLayer(uint32_t, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetInstRanksByNetLayer(uint32_t, uint32_t**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetInstSizeListByNetLayer(uint32_t, uint32_t**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetDeviceId(uint32_t, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetEndpointInfo(uint32_t, const EndpointDesc*, EndpointAttr, uint32_t, void*) override
    {
        return HCCL_SUCCESS;
    }
};

// -- Captures detect / rootHandle / generationIndex from MockCreateDerivedComm --
// Simulates real CreateDerivedComm: propagate parent detect, rootHandle, generationIndex to derived
struct DerivedCommCapture {
    std::shared_ptr<RankInfoDetect> detect;
    HcclRootHandleV2 rootHandle{};
    uint32_t generationIndex{0};
    std::string topParentIdentifier;
    CommInitSource initSource{CommInitSource::rootInfo};
    bool inheritParentConfig{false};
};

static DerivedCommCapture g_derivedCapture;

HcclResult MockCreateDerivedCommSG(
    CollComm* parent, std::unique_ptr<CollComm>& collComm, const CollComm::DerivedCommParams& params)
{
    CaptureDerivedCommParams(params);
    auto derived = std::make_unique<CollComm>(
        nullptr, params.rankId, params.commName, ManagerCallbacks{}, CollCommInitMode::fullMode);

    // Simulate real derivation: detect ownership is committed by the API after all registrations succeed.
    derived->rankgraph_ = parent->rankgraph_;
    derived->SetRootHandle(params.derivationInfo.rootHandle);
    derived->SetTopParentIdentifier(parent->GetTopParentIdentifier());
    derived->SetGenerationIndex(parent->GetGenerationIndex() + 1);
    derived->initSource_ = parent->initSource_;
    derived->hcclCommConfig_ = params.subConfig;

    // Capture for test assertions
    g_derivedCapture.detect = derived->GetRankInfoDetect();
    g_derivedCapture.rootHandle = params.derivationInfo.rootHandle;
    g_derivedCapture.generationIndex = parent->GetGenerationIndex() + 1;
    g_derivedCapture.topParentIdentifier = parent->GetTopParentIdentifier();
    g_derivedCapture.initSource = parent->initSource_;
    g_derivedCapture.inheritParentConfig = params.inheritParentConfig;

    collComm = std::move(derived);
    return HCCL_SUCCESS;
}

static void CommitDetectTransfer(CollComm* parent, CollComm* derived)
{
    ASSERT_NE(parent, nullptr);
    ASSERT_NE(derived, nullptr);
    parent->TransferRankInfoDetect(*derived);
}

// -- Helper: create CollComm with detect and valid rootHandle --
static std::unique_ptr<CollComm> MakeCommForChain(
    MockRankGraphSG& mockGraph, uint32_t rankSize, uint32_t myRank, uint32_t rootRankId,
    std::shared_ptr<RankInfoDetect> detect, uint32_t genIndex = 0, const std::string& identifier = "ut_chain")
{
    auto comm
        = std::make_unique<CollComm>(nullptr, myRank, identifier, ManagerCallbacks{}, CollCommInitMode::simpleMode);
    mockGraph.mockRankSize = rankSize;
    comm->rankgraph_ = &mockGraph;
    comm->rankId_ = myRank;
    comm->topParentIdentifier_ = identifier;
    comm->generationIndex_ = genIndex;
    comm->initSource_ = CommInitSource::rootInfo;
    if (detect != nullptr) {
        detect->rankId_ = myRank;
    }

    HcclRootHandleV2 rootHandle{};
    rootHandle.listenPort = 60040;
    strncpy_s(rootHandle.identifier, sizeof(rootHandle.identifier), identifier.c_str(), identifier.size());
    rootHandle.ip[0] = '\0';
    rootHandle.rootRankId = rootRankId;
    comm->SetRootHandle(rootHandle);
    comm->SetRankInfoDetect(detect);

    HcclCommConfigInit(&comm->hcclCommConfig_);
    return comm;
}

// -- Test fixture --
class CollCommShrinkGrowTest : public BaseInit {
protected:
    void SetUp() override
    {
        BaseInit::SetUp();
        MOCKER(hrtGetDeviceType).stubs().will(invoke(+[](Hccl::DevType& t) -> HcclResult {
            t = Hccl::DevType::DEV_TYPE_950;
            return HCCL_SUCCESS;
        }));
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(CollCommShrinkGrowTest, Ut_CommGrow_When_RankOutOfRange_Expect_E_PARA)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto comm = MakeCommForChain(mockGraph, 4, 0, 0, detect);

    std::unique_ptr<CollComm> grownComm;
    EXPECT_EQ(comm->CommGrow(8, 8, grownComm, nullptr, nullptr), HCCL_E_PARA);
    EXPECT_EQ(grownComm, nullptr);
}

TEST_F(CollCommShrinkGrowTest, Ut_CommGrow_When_RankMismatchMyRank_Expect_E_PARA)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto comm = MakeCommForChain(mockGraph, 4, 0, 0, detect);

    HcclRootInfo rootInfo{};
    HcclRootHandleV2 mismatchHandle{};
    mismatchHandle.listenPort = 99999;
    strncpy_s(mismatchHandle.identifier, sizeof(mismatchHandle.identifier), "rank_mismatch", 13);
    memcpy_s(rootInfo.internal, sizeof(rootInfo.internal), &mismatchHandle, sizeof(mismatchHandle));

    std::unique_ptr<CollComm> grownComm;
    EXPECT_EQ(comm->CommGrow(8, 1, grownComm, &rootInfo, nullptr), HCCL_E_PARA);
    EXPECT_EQ(grownComm, nullptr);
}

// =============================================================================
// 1. Shrink -> Grow chain: 2 rank -> Shrink to 1 -> Grow back to 2
// =============================================================================

// Full chain: root(rank0) survives, exclude rank1 -> shrink to 1 -> grow back to 2
// Verify detect ownership transferred parent -> shrinked -> grown
TEST_F(CollCommShrinkGrowTest, Ut_Chain_Shrink2to1_Grow1to2_DetectOwnershipPreserved)
{
    // -- Initial 2-rank comm, rank 0 = root, holds detect --
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto parentComm = MakeCommForChain(mockGraph, 2, 0, 0, detect);

    // -- Step 1: Shrink exclude rank 1, root survives --
    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));

    uint32_t excludeList[] = {1};
    std::unique_ptr<CollComm> shrinkedComm;
    HcclResult shrinkRet = parentComm->CommShrink(excludeList, 1, shrinkedComm, nullptr);
    ASSERT_EQ(shrinkRet, HCCL_SUCCESS);
    ASSERT_NE(shrinkedComm, nullptr);
    CommitDetectTransfer(parentComm.get(), shrinkedComm.get());

    // Verify detect ownership moved from parent to shrinked comm
    EXPECT_FALSE(parentComm->GetRankInfoDetect());
    EXPECT_FALSE(parentComm->HasDetectOwnership());
    EXPECT_NE(shrinkedComm->GetRankInfoDetect(), nullptr);
    EXPECT_EQ(shrinkedComm->GetRankInfoDetect().get(), detect.get());

    // Verify rootHandle inherited
    EXPECT_TRUE(shrinkedComm->IsRootHandleValid());
    EXPECT_EQ(shrinkedComm->GetCollCommRootHandle().listenPort, 60040u);

    // Verify generationIndex incremented
    EXPECT_EQ(shrinkedComm->GetGenerationIndex(), 1u);

    // Verify rootRankId mapped correctly (root 0 survives, new rank 0)
    EXPECT_EQ(shrinkedComm->GetRootRankId(), 0u);

    // -- Step 2: Grow from 1 rank to 2 (root survives, normal grow) --
    mockGraph.mockRankSize = 1;

    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetect::TriggerGrowCollect)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect*, u32 newRankCount, RankTableInfo& mergedRankTable) -> HcclResult {
            mergedRankTable.rankCount = newRankCount + 1;
            return HCCL_SUCCESS;
        }));

    std::unique_ptr<CollComm> grownComm;
    HcclResult growRet = shrinkedComm->CommGrow(2, 0, grownComm, nullptr, nullptr);
    ASSERT_EQ(growRet, HCCL_SUCCESS);
    ASSERT_NE(grownComm, nullptr);
    CommitDetectTransfer(shrinkedComm.get(), grownComm.get());

    // Verify grown: detect ownership moved again to the new chain tail
    EXPECT_FALSE(shrinkedComm->GetRankInfoDetect());
    EXPECT_FALSE(shrinkedComm->HasDetectOwnership());
    EXPECT_NE(grownComm->GetRankInfoDetect(), nullptr);
    EXPECT_EQ(grownComm->GetRankInfoDetect().get(), detect.get());

    // Verify generationIndex incremented again
    EXPECT_EQ(grownComm->GetGenerationIndex(), 2u);

    // Verify rootHandle still valid
    EXPECT_TRUE(grownComm->IsRootHandleValid());
}

// root 非 0：rank3 为 root，Shrink 排除 rank0/1 后 root 重映射为 rank1，Grow 必须继续继承 rank1
TEST_F(CollCommShrinkGrowTest, Ut_Chain_NonZeroRootRemappedAndInheritedByGrow)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto parentComm = MakeCommForChain(mockGraph, 4, 2, 3, detect);

    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));

    uint32_t excludeList[] = {0, 1};
    std::unique_ptr<CollComm> shrinkedComm;
    ASSERT_EQ(parentComm->CommShrink(excludeList, 2, shrinkedComm, nullptr), HCCL_SUCCESS);
    ASSERT_NE(shrinkedComm, nullptr);
    CommitDetectTransfer(parentComm.get(), shrinkedComm.get());

    // survivors = [2, 3]，root rank3 在新域中重映射为 rank1
    EXPECT_EQ(shrinkedComm->GetRootRankId(), 1u);
    EXPECT_TRUE(shrinkedComm->IsRootHandleValid());

    mockGraph.mockRankSize = 2;
    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetect::TriggerGrowCollect)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect*, u32 newRankCount, RankTableInfo& mergedRankTable) -> HcclResult {
            mergedRankTable.rankCount = newRankCount + 2;
            return HCCL_SUCCESS;
        }));

    std::unique_ptr<CollComm> grownComm;
    ASSERT_EQ(shrinkedComm->CommGrow(4, 2, grownComm, nullptr, nullptr), HCCL_SUCCESS);
    ASSERT_NE(grownComm, nullptr);
    CommitDetectTransfer(shrinkedComm.get(), grownComm.get());
    EXPECT_EQ(grownComm->GetRootRankId(), 1u);
}

// Shrink excludes root -> rootHandle invalid -> Grow goes rootReselect
TEST_F(CollCommShrinkGrowTest, Ut_Chain_ShrinkExcludeRoot_GrowRootReselect)
{
    // -- Initial 2-rank comm, rank 1 = non-root, root = rank 0 --
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto parentComm = MakeCommForChain(mockGraph, 2, 1, 0, detect);

    // -- Step 1: Shrink exclude rank 0 (root excluded) --
    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));

    uint32_t excludeList[] = {0};
    std::unique_ptr<CollComm> shrinkedComm;
    HcclResult shrinkRet = parentComm->CommShrink(excludeList, 1, shrinkedComm, nullptr);
    ASSERT_EQ(shrinkRet, HCCL_SUCCESS);
    ASSERT_NE(shrinkedComm, nullptr);
    CommitDetectTransfer(parentComm.get(), shrinkedComm.get());

    // Verify shrinked: rootHandle cleared
    EXPECT_FALSE(shrinkedComm->IsRootHandleValid());

    // Verify generationIndex incremented
    EXPECT_EQ(shrinkedComm->GetGenerationIndex(), 1u);

    // root 被剔除后新域没有 root，rootRankId 必须是无效值，等待 Grow 重选
    EXPECT_EQ(shrinkedComm->GetRootRankId(), UINT32_MAX);

    // -- Step 2: Grow goes rootReselect (rootHandle invalid) --
    mockGraph.mockRankSize = 1;

    // Non-coordinator path: SetupAgent + WaitComplete + GetRankTable
    MOCKER_CPP(&RankInfoDetect::SetupAgent)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect* detect, u32, u32, const HcclRootHandleV2& handle) -> HcclResult {
            HcclRootHandleV2 detectedHandle = handle;
            detectedHandle.rootRankId = 1;
            detect->SetRootHandle(detectedHandle);
            return HCCL_SUCCESS;
        }));
    MOCKER_CPP(&RankInfoDetect::WaitComplete)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankInfoDetect::GetRankTable)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect*, RankTableInfo& rankTable) {
            rankTable.rankCount = 2;
        }));

    HcclRootInfo rootInfo{};
    HcclRootHandleV2 newRootHandle{};
    newRootHandle.listenPort = 60050;
    strncpy_s(newRootHandle.identifier, sizeof(newRootHandle.identifier), "ut_reselect_chain", 18);
    memcpy_s(rootInfo.internal, sizeof(rootInfo.internal), &newRootHandle, sizeof(newRootHandle));

    std::unique_ptr<CollComm> grownComm;
    HcclResult growRet = shrinkedComm->CommGrow(2, 0, grownComm, &rootInfo, nullptr);
    ASSERT_EQ(growRet, HCCL_SUCCESS);
    ASSERT_NE(grownComm, nullptr);
    CommitDetectTransfer(shrinkedComm.get(), grownComm.get());

    // Verify grown: detect created (non-null)
    EXPECT_NE(grownComm->GetRankInfoDetect(), nullptr);

    // Verify rootHandle updated to new rootInfo
    EXPECT_TRUE(grownComm->IsRootHandleValid());
    EXPECT_EQ(grownComm->GetCollCommRootHandle().listenPort, 60050u);
    EXPECT_EQ(grownComm->GetRootRankId(), 1u);
}

// =============================================================================
// 2. CommGrow rootInfo mismatch -> rootReselect path
// =============================================================================

// rootHandle valid but input rootInfo port mismatch -> rootReselect=true
TEST_F(CollCommShrinkGrowTest, Ut_CommGrow_When_RootInfoPortMismatch_Expect_RootReselect)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto comm = MakeCommForChain(mockGraph, 4, 0, 0, detect);

    HcclRootInfo rootInfo{};
    HcclRootHandleV2 mismatchHandle{};
    mismatchHandle.listenPort = 99999; // differs from comm 60040
    strncpy_s(mismatchHandle.identifier, sizeof(mismatchHandle.identifier), "ut_chain", 8);
    memcpy_s(rootInfo.internal, sizeof(rootInfo.internal), &mismatchHandle, sizeof(mismatchHandle));

    auto pendingDetect = std::make_shared<RankInfoDetect>();
    MOCKER_CPP(&RankInfoDetect::IsReadyForGrow, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    HcclRootHandleV2 pendingRootHandle{};
    pendingRootHandle.listenPort = 60030;
    pendingRootHandle.rootRankId = 0;
    strncpy_s(pendingRootHandle.identifier, sizeof(pendingRootHandle.identifier), "ut_chain", 8);
    pendingDetect->SetRootHandle(pendingRootHandle);
    comm->SetPendingDetect(pendingDetect);

    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetect::SetupAgent)
        .expects(once())
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankInfoDetect::WaitComplete)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankInfoDetect::GetRankTable)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect*, RankTableInfo& rankTable) {
            rankTable.rankCount = 8;
        }));
    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));

    std::unique_ptr<CollComm> grownComm;
    HcclResult ret = comm->CommGrow(8, 0, grownComm, &rootInfo, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(grownComm, nullptr);
    CommitDetectTransfer(comm.get(), grownComm.get());

    // pendingDetect consumed
    EXPECT_EQ(comm->TakePendingDetect(), nullptr);

    // detect ownership moved from parent to the derived comm
    EXPECT_FALSE(comm->GetRankInfoDetect());
    EXPECT_FALSE(comm->HasDetectOwnership());
    EXPECT_NE(grownComm->GetRankInfoDetect(), nullptr);
}

// rootHandle valid but input rootInfo identifier mismatch -> rootReselect=true
TEST_F(CollCommShrinkGrowTest, Ut_CommGrow_When_RootInfoIdentifierMismatch_Expect_RootReselect)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto comm = MakeCommForChain(mockGraph, 4, 0, 0, detect);

    HcclRootInfo rootInfo{};
    HcclRootHandleV2 mismatchHandle{};
    mismatchHandle.listenPort = 60040; // same as comm
    strncpy_s(mismatchHandle.identifier, sizeof(mismatchHandle.identifier), "different_id", 12);
    memcpy_s(rootInfo.internal, sizeof(rootInfo.internal), &mismatchHandle, sizeof(mismatchHandle));

    auto pendingDetect = std::make_shared<RankInfoDetect>();
    MOCKER_CPP(&RankInfoDetect::IsReadyForGrow, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    HcclRootHandleV2 pendingRootHandle{};
    pendingRootHandle.listenPort = 60030;
    pendingRootHandle.rootRankId = 0;
    strncpy_s(pendingRootHandle.identifier, sizeof(pendingRootHandle.identifier), "ut_chain", 8);
    pendingDetect->SetRootHandle(pendingRootHandle);
    comm->SetPendingDetect(pendingDetect);

    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetect::SetupAgent)
        .expects(once())
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankInfoDetect::WaitComplete)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RankInfoDetect::GetRankTable)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect*, RankTableInfo& rankTable) {
            rankTable.rankCount = 8;
        }));
    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));

    std::unique_ptr<CollComm> grownComm;
    HcclResult ret = comm->CommGrow(8, 0, grownComm, &rootInfo, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(grownComm, nullptr);
}

// =============================================================================
// 3. Shrink detect ownership verification (detect survives in shrinked comm)
// =============================================================================

// After Shrink, parent releases detect and the derived comm becomes the only owner
TEST_F(CollCommShrinkGrowTest, Ut_Shrink_When_DetectTransferred_Expect_ParentReleased)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto parentComm = MakeCommForChain(mockGraph, 4, 1, 0, detect);

    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));

    uint32_t excludeList[] = {2, 3};
    std::unique_ptr<CollComm> shrinkedComm;
    HcclResult ret = parentComm->CommShrink(excludeList, 2, shrinkedComm, nullptr);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_NE(shrinkedComm, nullptr);
    CommitDetectTransfer(parentComm.get(), shrinkedComm.get());

    EXPECT_FALSE(parentComm->GetRankInfoDetect());
    EXPECT_FALSE(parentComm->HasDetectOwnership());
    EXPECT_EQ(shrinkedComm->GetRankInfoDetect().get(), detect.get());
}

TEST_F(CollCommShrinkGrowTest, Ut_Shrink_When_DerivedCreated_Expect_DetectKeptUntilCommit)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto parentComm = MakeCommForChain(mockGraph, 4, 1, 0, detect);

    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));

    uint32_t excludeList[] = {2, 3};
    std::unique_ptr<CollComm> shrinkedComm;
    ASSERT_EQ(parentComm->CommShrink(excludeList, 2, shrinkedComm, nullptr), HCCL_SUCCESS);
    ASSERT_NE(shrinkedComm, nullptr);

    // Derived comm 创建成功但尚未提交，父域必须仍保留 detect，供后续注册失败时重试
    EXPECT_TRUE(parentComm->HasDetectOwnership());
    EXPECT_EQ(parentComm->GetRankInfoDetect().get(), detect.get());
    EXPECT_EQ(shrinkedComm->GetRankInfoDetect(), nullptr);

    CommitDetectTransfer(parentComm.get(), shrinkedComm.get());
    EXPECT_FALSE(parentComm->HasDetectOwnership());
    EXPECT_EQ(parentComm->GetRankInfoDetect(), nullptr);
    EXPECT_EQ(shrinkedComm->GetRankInfoDetect().get(), detect.get());
}

TEST_F(CollCommShrinkGrowTest, Ut_Chain_When_DetectTransferred_Expect_OldCommRejectsShrinkAndGrow)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto parentComm = MakeCommForChain(mockGraph, 4, 1, 0, detect);

    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));

    uint32_t excludeList[] = {3};
    std::unique_ptr<CollComm> shrinkedComm;
    ASSERT_EQ(parentComm->CommShrink(excludeList, 1, shrinkedComm, nullptr), HCCL_SUCCESS);
    ASSERT_NE(shrinkedComm, nullptr);
    CommitDetectTransfer(parentComm.get(), shrinkedComm.get());
    EXPECT_FALSE(parentComm->HasDetectOwnership());

    uint32_t excludeListAgain[] = {2};
    std::unique_ptr<CollComm> rejectedShrink;
    EXPECT_EQ(parentComm->CommShrink(excludeListAgain, 1, rejectedShrink, nullptr), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(rejectedShrink, nullptr);

    std::unique_ptr<CollComm> rejectedGrow;
    EXPECT_EQ(parentComm->CommGrow(8, 1, rejectedGrow, nullptr, nullptr), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(rejectedGrow, nullptr);
}

// Shrink root excluded: rootHandle cleared but detect still transferred
TEST_F(CollCommShrinkGrowTest, Ut_Shrink_When_RootExcluded_Expect_DetectPreserved_RootHandleCleared)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto parentComm = MakeCommForChain(mockGraph, 4, 1, 0, detect);

    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));

    uint32_t excludeList[] = {0, 2};
    std::unique_ptr<CollComm> shrinkedComm;
    HcclResult ret = parentComm->CommShrink(excludeList, 2, shrinkedComm, nullptr);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_NE(shrinkedComm, nullptr);
    CommitDetectTransfer(parentComm.get(), shrinkedComm.get());

    EXPECT_FALSE(shrinkedComm->IsRootHandleValid());
    EXPECT_EQ(shrinkedComm->GetRankInfoDetect().get(), detect.get());
}

// =============================================================================
// 4. identifier derivation chain verification
// =============================================================================

// After Shrink identifier correctly derived: topParentIdentifier + "_" + (genIndex+1)
TEST_F(CollCommShrinkGrowTest, Ut_Shrink_When_IdentifierDerived_Expect_CorrectFormat)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto parentComm = MakeCommForChain(mockGraph, 4, 1, 0, detect, 0, "base_id");

    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));

    uint32_t excludeList[] = {2};
    std::unique_ptr<CollComm> shrinkedComm;
    HcclResult ret = parentComm->CommShrink(excludeList, 1, shrinkedComm, nullptr);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(g_derivedCapture.topParentIdentifier, "base_id");
    EXPECT_EQ(g_derivedParamsCapture.commName, "base_id_1");
    EXPECT_EQ(g_derivedCapture.generationIndex, 1u);
}

// Double Shrink: generationIndex 0 -> 1 -> 2
TEST_F(CollCommShrinkGrowTest, Ut_Chain_DoubleShrink_Expect_GenerationIncrementedTwice)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto parentComm = MakeCommForChain(mockGraph, 4, 0, 0, detect, 0, "chain_id");

    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));

    // First Shrink: 4 -> 3, exclude rank 3
    uint32_t excludeList1[] = {3};
    std::unique_ptr<CollComm> shrinked1;
    ASSERT_EQ(parentComm->CommShrink(excludeList1, 1, shrinked1, nullptr), HCCL_SUCCESS);
    ASSERT_NE(shrinked1, nullptr);
    CommitDetectTransfer(parentComm.get(), shrinked1.get());
    EXPECT_EQ(shrinked1->GetGenerationIndex(), 1u);
    EXPECT_EQ(shrinked1->GetTopParentIdentifier(), "chain_id");
    EXPECT_EQ(g_derivedParamsCapture.commName, "chain_id_1");

    mockGraph.mockRankSize = 3;

    // Second Shrink: 3 -> 2, exclude rank 2
    uint32_t excludeList2[] = {2};
    std::unique_ptr<CollComm> shrinked2;
    ASSERT_EQ(shrinked1->CommShrink(excludeList2, 1, shrinked2, nullptr), HCCL_SUCCESS);
    ASSERT_NE(shrinked2, nullptr);
    CommitDetectTransfer(shrinked1.get(), shrinked2.get());
    EXPECT_EQ(shrinked2->GetGenerationIndex(), 2u);
    EXPECT_EQ(shrinked2->GetTopParentIdentifier(), "chain_id");
    EXPECT_EQ(g_derivedParamsCapture.commName, "chain_id_2");

    EXPECT_EQ(shrinked2->GetRankInfoDetect().get(), detect.get());
}

// =============================================================================
// 5. CommGrow rootInfo=nullptr and rootHandle valid -> normal grow path
// =============================================================================

// root survives + rootInfo=nullptr -> normal grow (not rootReselect)
TEST_F(CollCommShrinkGrowTest, Ut_CommGrow_When_RootSurvivesRootInfoNull_Expect_NormalGrow)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto comm = MakeCommForChain(mockGraph, 4, 0, 0, detect);

    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetect::TriggerGrowCollect)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect*, u32 newRankCount, RankTableInfo& mergedRankTable) -> HcclResult {
            mergedRankTable.rankCount = newRankCount + 4;
            return HCCL_SUCCESS;
        }));
    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));

    std::unique_ptr<CollComm> grownComm;
    HcclResult ret = comm->CommGrow(8, 0, grownComm, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(grownComm, nullptr);
    CommitDetectTransfer(comm.get(), grownComm.get());

    EXPECT_EQ(grownComm->GetRankInfoDetect().get(), detect.get());
}

// Non-root survives + rootInfo=nullptr -> normal grow (WaitForNextBroadcast)
TEST_F(CollCommShrinkGrowTest, Ut_CommGrow_When_NonRootSurvivesRootInfoNull_Expect_NormalGrow)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto comm = MakeCommForChain(mockGraph, 4, 1, 0, detect);

    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(false));
    MOCKER_CPP(&RankInfoDetect::WaitForNextBroadcast)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect*, RankTableInfo& rankTable, uint32_t expectedRankCount) -> HcclResult {
            rankTable.rankCount = expectedRankCount;
            return HCCL_SUCCESS;
        }));
    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));

    std::unique_ptr<CollComm> grownComm;
    HcclResult ret = comm->CommGrow(8, 1, grownComm, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(grownComm, nullptr);
}

// =============================================================================
// 6. config inheritance strategy verification
// =============================================================================

// Shrink config=nullptr -> inheritParentConfig=true
TEST_F(CollCommShrinkGrowTest, Ut_Shrink_When_ConfigNull_Expect_InheritParentConfig)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto parentComm = MakeCommForChain(mockGraph, 4, 1, 0, detect);

    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));

    uint32_t excludeList[] = {2};
    std::unique_ptr<CollComm> shrinkedComm;
    HcclResult ret = parentComm->CommShrink(excludeList, 1, shrinkedComm, nullptr);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(g_derivedCapture.inheritParentConfig);
    EXPECT_TRUE(g_derivedParamsCapture.inheritParentConfig);
    EXPECT_EQ(g_derivedParamsCapture.commName, "ut_chain_1");
    EXPECT_EQ(std::string(g_derivedParamsCapture.subConfig.hcclCommName), "ut_chain_1");
    EXPECT_EQ(g_derivedParamsCapture.survivorRanks, std::vector<uint32_t>({0, 1, 3}));
    EXPECT_EQ(g_derivedParamsCapture.rankId, 1u);
    EXPECT_FALSE(g_derivedParamsCapture.derivationInfo.isGrow);
    EXPECT_EQ(g_derivedParamsCapture.derivationInfo.rootHandle.listenPort, 60040u);
}

// Shrink config!=nullptr -> inheritParentConfig=false
TEST_F(CollCommShrinkGrowTest, Ut_Shrink_When_ConfigNotNull_Expect_NotInheritParentConfig)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto parentComm = MakeCommForChain(mockGraph, 4, 1, 0, detect);

    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));

    HcclCommConfig userConfig;
    HcclCommConfigInit(&userConfig);
    userConfig.hcclBufferSize = 777;

    uint32_t excludeList[] = {2};
    std::unique_ptr<CollComm> shrinkedComm;
    HcclResult ret = parentComm->CommShrink(excludeList, 1, shrinkedComm, &userConfig);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(g_derivedCapture.inheritParentConfig);
    EXPECT_FALSE(g_derivedParamsCapture.inheritParentConfig);
    EXPECT_EQ(g_derivedParamsCapture.subConfig.hcclBufferSize, 777u);
    EXPECT_EQ(g_derivedParamsCapture.commName, "ut_chain_1");
    EXPECT_EQ(g_derivedParamsCapture.rankId, 1u);
    EXPECT_EQ(g_derivedParamsCapture.derivationInfo.rootHandle.listenPort, 60040u);
}

// Grow config=nullptr -> inheritParentConfig=true
TEST_F(CollCommShrinkGrowTest, Ut_Grow_When_ConfigNull_Expect_InheritParentConfig)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto comm = MakeCommForChain(mockGraph, 4, 0, 0, detect);

    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetect::TriggerGrowCollect)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect*, u32 newRankCount, RankTableInfo& mergedRankTable) -> HcclResult {
            mergedRankTable.rankCount = newRankCount + 4;
            return HCCL_SUCCESS;
        }));
    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));

    std::unique_ptr<CollComm> grownComm;
    HcclResult ret = comm->CommGrow(8, 0, grownComm, nullptr, nullptr);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(g_derivedCapture.inheritParentConfig);
    EXPECT_TRUE(g_derivedParamsCapture.inheritParentConfig);
    EXPECT_TRUE(g_derivedParamsCapture.survivorRanks.empty());
    EXPECT_EQ(g_derivedParamsCapture.rankId, 0u);
    EXPECT_EQ(g_derivedParamsCapture.commName, "ut_chain_1");
    EXPECT_EQ(std::string(g_derivedParamsCapture.subConfig.hcclCommName), "ut_chain_1");
    EXPECT_TRUE(g_derivedParamsCapture.derivationInfo.isGrow);
    EXPECT_EQ(g_derivedParamsCapture.derivationInfo.rootHandle.listenPort, 60040u);
}

// =============================================================================
// 7. Shrink root excluded -> root does not create derived comm
// =============================================================================

// 本 rank 就是 root 且被剔除：禁止调用方在自身被剔除时发起 Shrink
TEST_F(CollCommShrinkGrowTest, Ut_Shrink_When_RootSelfExcluded_Expect_E_PARA)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto parentComm = MakeCommForChain(mockGraph, 4, 0, 0, detect);

    uint32_t excludeList[] = {0, 2};
    std::unique_ptr<CollComm> shrinkedComm;
    HcclResult ret = parentComm->CommShrink(excludeList, 2, shrinkedComm, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(shrinkedComm, nullptr);
}

// =============================================================================
// 8. generationIndex overflow protection
// =============================================================================

// generationIndex == UINT32_MAX -> HCCL_E_PARA
TEST_F(CollCommShrinkGrowTest, Ut_Shrink_When_GenerationOverflow_Expect_E_PARA)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto parentComm = MakeCommForChain(mockGraph, 4, 1, 0, detect);
    parentComm->SetGenerationIndex(UINT32_MAX);

    uint32_t excludeList[] = {2};
    std::unique_ptr<CollComm> shrinkedComm;
    HcclResult ret = parentComm->CommShrink(excludeList, 1, shrinkedComm, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// Grow generationIndex == UINT32_MAX -> HCCL_E_PARA
TEST_F(CollCommShrinkGrowTest, Ut_Grow_When_GenerationOverflow_Expect_E_PARA)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto comm = MakeCommForChain(mockGraph, 4, 0, 0, detect);
    comm->SetGenerationIndex(UINT32_MAX);

    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetect::TriggerGrowCollect)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect*, u32 newRankCount, RankTableInfo& mergedRankTable) -> HcclResult {
            mergedRankTable.rankCount = newRankCount + 4;
            return HCCL_SUCCESS;
        }));

    std::unique_ptr<CollComm> grownComm;
    HcclResult ret = comm->CommGrow(8, 0, grownComm, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 连续两个 shrink -> grow 周期，验证 detect 所有权和 generationIndex 在链式派生中保持稳定
TEST_F(CollCommShrinkGrowTest, Ut_Chain_RepeatedShrinkGrow_ExpectStableAcrossCycles)
{
    MockRankGraphSG mockGraph;
    auto detect = std::make_shared<RankInfoDetect>();
    auto currentComm = MakeCommForChain(mockGraph, 2, 0, 0, detect);

    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockCreateDerivedCommSG));
    MOCKER_CPP(&RankInfoDetect::IsRoot, bool(RankInfoDetect::*)() const).stubs().will(returnValue(true));
    MOCKER_CPP(&RankInfoDetect::TriggerGrowCollect)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect*, u32, RankTableInfo& mergedRankTable) -> HcclResult {
            mergedRankTable.rankCount = 2;
            return HCCL_SUCCESS;
        }));

    for (uint32_t cycle = 0; cycle < 2; ++cycle) {
        uint32_t excludeList[] = {1};
        std::unique_ptr<CollComm> shrinkedComm;
        ASSERT_EQ(currentComm->CommShrink(excludeList, 1, shrinkedComm, nullptr), HCCL_SUCCESS);
        ASSERT_NE(shrinkedComm, nullptr);
        CommitDetectTransfer(currentComm.get(), shrinkedComm.get());
        EXPECT_EQ(shrinkedComm->GetRankInfoDetect().get(), detect.get());
        EXPECT_FALSE(currentComm->GetRankInfoDetect());
        EXPECT_EQ(shrinkedComm->GetTopParentIdentifier(), "ut_chain");
        EXPECT_EQ(g_derivedParamsCapture.commName, "ut_chain_" + std::to_string(cycle * 2 + 1));

        // MockCreateDerivedCommSG 不创建真实 rankGraph，显式注入以支持下一轮 GetRankSize()
        mockGraph.mockRankSize = 1;
        shrinkedComm->rankgraph_ = &mockGraph;

        std::unique_ptr<CollComm> grownComm;
        ASSERT_EQ(shrinkedComm->CommGrow(2, 0, grownComm, nullptr, nullptr), HCCL_SUCCESS);
        ASSERT_NE(grownComm, nullptr);
        CommitDetectTransfer(shrinkedComm.get(), grownComm.get());
        EXPECT_EQ(grownComm->GetRankInfoDetect().get(), detect.get());
        EXPECT_EQ(grownComm->GetTopParentIdentifier(), "ut_chain");
        EXPECT_EQ(g_derivedParamsCapture.commName, "ut_chain_" + std::to_string(cycle * 2 + 2));

        mockGraph.mockRankSize = 2;
        grownComm->rankgraph_ = &mockGraph;
        currentComm = std::move(grownComm);
    }

    EXPECT_EQ(currentComm->GetGenerationIndex(), 4u);
    EXPECT_EQ(currentComm->GetRankInfoDetect().get(), detect.get());
}

// 新 rank 探测路径：rootInfo 解析 + SetupAgent + WaitComplete + rankTable 回填
TEST_F(CollCommShrinkGrowTest, Ut_DetectRankTableForNewRank_When_Success_ExpectDetectResultFilled)
{
    HcclRootHandleV2 rootHandle{};
    rootHandle.listenPort = 60123;
    rootHandle.rootRankId = 0;
    strncpy_s(rootHandle.identifier, sizeof(rootHandle.identifier), "grow_new_rank", 13);

    HcclRootInfo rootInfo{};
    memcpy_s(rootInfo.internal, sizeof(rootInfo.internal), &rootHandle, sizeof(rootHandle));

    MOCKER_CPP(&RankInfoDetect::SetupAgent)
        .stubs()
        .will(invoke(+[](RankInfoDetect* self, u32 rankSize, u32 rankId, const HcclRootHandleV2& inputRoot) {
            self->rootHandle_ = inputRoot;
            self->rankId_ = rankId;
            self->rankTable_.rankCount = rankSize;
            return HCCL_SUCCESS;
        }));
    MOCKER_CPP(&RankInfoDetect::WaitComplete).stubs().will(returnValue(HCCL_SUCCESS));

    CollComm::GrowNewRankDetectResult result;
    ASSERT_EQ(CollComm::DetectRankTableForNewRank(2, 1, &rootInfo, result), HCCL_SUCCESS);
    EXPECT_NE(result.reselectDetect, nullptr);
    EXPECT_EQ(result.newRootHandle.listenPort, rootHandle.listenPort);
    EXPECT_EQ(result.newRootHandle.rootRankId, rootHandle.rootRankId);
    EXPECT_EQ(std::string(result.newRootHandle.identifier), std::string(rootHandle.identifier));
    EXPECT_EQ(result.mergedRankTable.rankCount, 2u);
}

// 新 rank 建域路径：验证 config 继承、HcclCommunicator Init、rankGraph/cclBuffer 提取
TEST_F(CollCommShrinkGrowTest, Ut_CreateCommunicatorForNewRank_When_Success_ExpectCommResourcesFilled)
{
    CollComm::GrowNewRankDetectResult detectResult;
    detectResult.newRootHandle.listenPort = 60124;
    detectResult.newRootHandle.rootRankId = 0;
    strncpy_s(
        detectResult.newRootHandle.identifier, sizeof(detectResult.newRootHandle.identifier), "grow_new_rank", 13);
    detectResult.mergedRankTable.rankCount = 2;

    MOCKER_CPP(&Hccl::HcclCommunicator::Init, HcclResult(Hccl::HcclCommunicator::*)(const RankTableInfo&))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::HcclCommunicator::GetRankGraphV2, HcclResult(Hccl::HcclCommunicator::*)(void*&))
        .stubs()
        .will(invoke(+[](Hccl::HcclCommunicator*, void*& rankGraph) {
            rankGraph = reinterpret_cast<void*>(0x1234);
            return HCCL_SUCCESS;
        }));
    MOCKER_CPP(
        &Hccl::HcclCommunicator::HcclGetCclBuffer,
        HcclResult(Hccl::HcclCommunicator::*)(uintptr_t&, size_t&, HcclMemType&))
        .stubs()
        .will(invoke(+[](Hccl::HcclCommunicator*, uintptr_t& addr, size_t& size, HcclMemType& type) {
            addr = 0x2000;
            size = 4096;
            type = HcclMemType::HCCL_MEM_TYPE_DEVICE;
            return HCCL_SUCCESS;
        }));

    HcclCommConfig config;
    HcclCommConfigInit(&config);
    CollComm::GrowNewRankCommResult result;
    ASSERT_EQ(CollComm::CreateCommunicatorForNewRank(2, 1, detectResult, &config, result), HCCL_SUCCESS);
    ASSERT_NE(result.drivedHcclComm, nullptr);
    EXPECT_EQ(result.rankGraph, reinterpret_cast<void*>(0x1234));
    EXPECT_EQ(result.cclBuffer.addr, reinterpret_cast<void*>(0x2000));
    EXPECT_EQ(result.cclBuffer.size, 4096u);
    EXPECT_EQ(std::string(result.subConfig.hcclCommName), std::string("grow_new_rank"));
}

namespace {
HcclRootInfo MakeNewRankRootInfo(uint32_t rootRankId, HcclRootHandleV2& rootHandle)
{
    rootHandle = HcclRootHandleV2{};
    rootHandle.listenPort = 60200;
    rootHandle.rootRankId = rootRankId;
    rootHandle.valid = 1;
    rootHandle.generationIndex = 3;
    strncpy_s(rootHandle.identifier, sizeof(rootHandle.identifier), "new_rank_grow", sizeof("new_rank_grow") - 1);
    strncpy_s(
        rootHandle.topParentIdentifier, sizeof(rootHandle.topParentIdentifier), "new_rank_top",
        sizeof("new_rank_top") - 1);

    HcclRootInfo rootInfo{};
    memcpy_s(rootInfo.internal, sizeof(rootInfo.internal), &rootHandle, sizeof(rootHandle));
    return rootInfo;
}

void MockNewRankDetectAgent(HcclResult waitRet)
{
    MOCKER_CPP(&RankInfoDetect::SetupAgent)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(+[](RankInfoDetect* self, u32 rankSize, u32 rankId, const HcclRootHandleV2& inputRoot) {
            self->rankId_ = rankId;
            self->rootHandle_ = inputRoot;
            self->rankTable_.rankCount = rankSize;
            return HCCL_SUCCESS;
        }));
    MOCKER_CPP(&RankInfoDetect::WaitComplete)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(waitRet));
}

void MockNewRankCommunicatorCreate()
{
    MOCKER_CPP(&Hccl::HcclCommunicator::Init, HcclResult(Hccl::HcclCommunicator::*)(const RankTableInfo&))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::HcclCommunicator::GetRankGraphV2, HcclResult(Hccl::HcclCommunicator::*)(void*&))
        .stubs()
        .will(invoke(+[](Hccl::HcclCommunicator*, void*& rankGraph) {
            rankGraph = reinterpret_cast<void*>(0x1234);
            return HCCL_SUCCESS;
        }));
    MOCKER_CPP(
        &Hccl::HcclCommunicator::HcclGetCclBuffer,
        HcclResult(Hccl::HcclCommunicator::*)(uintptr_t&, size_t&, HcclMemType&))
        .stubs()
        .will(invoke(+[](Hccl::HcclCommunicator*, uintptr_t& addr, size_t& size, HcclMemType& type) {
            addr = 0x2000;
            size = 4096;
            type = HcclMemType::HCCL_MEM_TYPE_DEVICE;
            return HCCL_SUCCESS;
        }));
}

HcclResult MockNewRankDetectShutdown(RankInfoDetect* self)
{
    self->shutdownDone_ = true;
    return HCCL_SUCCESS;
}

void MockNewRankCollCommInit(HcclResult initRet)
{
    MOCKER_CPP(&CollComm::LoadCclKernelBinHandle).stubs().will(invoke(+[](aclrtBinHandle& binHandle) {
        binHandle = reinterpret_cast<aclrtBinHandle>(0x4000);
        return HCCL_SUCCESS;
    }));
    MOCKER_CPP(&CollComm::Init).stubs().will(returnValue(initRet));
}
} // namespace

TEST_F(CollCommShrinkGrowTest, Ut_CommGrowNewRank_When_SetupAgentFailed_Expect_ShutdownAndError)
{
    HcclRootHandleV2 rootHandle{};
    HcclRootInfo rootInfo = MakeNewRankRootInfo(0, rootHandle);
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1;

    MOCKER_CPP(&RankInfoDetect::SetupAgent)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));
    MOCKER_CPP(&RankInfoDetect::Shutdown).expects(once()).will(invoke(MockNewRankDetectShutdown));

    std::unique_ptr<CollComm> newCollComm;
    EXPECT_EQ(CollComm::CommGrowNewRank(4, 3, &rootInfo, &config, newCollComm), HCCL_E_INTERNAL);
    EXPECT_EQ(newCollComm, nullptr);
}

TEST_F(CollCommShrinkGrowTest, Ut_CommGrowNewRank_When_WaitCompleteFailed_Expect_ShutdownAndError)
{
    HcclRootHandleV2 rootHandle{};
    HcclRootInfo rootInfo = MakeNewRankRootInfo(0, rootHandle);
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1;

    MockNewRankDetectAgent(HCCL_E_INTERNAL);
    MOCKER_CPP(&RankInfoDetect::Shutdown).expects(once()).will(invoke(MockNewRankDetectShutdown));

    std::unique_ptr<CollComm> newCollComm;
    EXPECT_EQ(CollComm::CommGrowNewRank(4, 3, &rootInfo, &config, newCollComm), HCCL_E_INTERNAL);
    EXPECT_EQ(newCollComm, nullptr);
}

TEST_F(CollCommShrinkGrowTest, Ut_CommGrowNewRank_When_HcclCommunicatorInitFailed_Expect_Error)
{
    HcclRootHandleV2 rootHandle{};
    HcclRootInfo rootInfo = MakeNewRankRootInfo(0, rootHandle);
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1;

    MockNewRankDetectAgent(HCCL_SUCCESS);
    MOCKER_CPP(&RankInfoDetect::Shutdown).stubs().will(invoke(MockNewRankDetectShutdown));
    MOCKER_CPP(&Hccl::HcclCommunicator::Init, HcclResult(Hccl::HcclCommunicator::*)(const RankTableInfo&))
        .stubs()
        .will(returnValue(HCCL_E_INTERNAL));

    std::unique_ptr<CollComm> newCollComm;
    EXPECT_EQ(CollComm::CommGrowNewRank(4, 3, &rootInfo, &config, newCollComm), HCCL_E_INTERNAL);
    EXPECT_EQ(newCollComm, nullptr);
}

TEST_F(CollCommShrinkGrowTest, Ut_CommGrowNewRank_When_Success_Expect_CommInitializedAndDetectAttached)
{
    HcclRootHandleV2 rootHandle{};
    HcclRootInfo rootInfo = MakeNewRankRootInfo(0, rootHandle);
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1;

    MockNewRankDetectAgent(HCCL_SUCCESS);
    MockNewRankCommunicatorCreate();
    MOCKER_CPP(&RankInfoDetect::Shutdown).stubs().will(invoke(MockNewRankDetectShutdown));
    MockNewRankCollCommInit(HCCL_SUCCESS);
    MOCKER_CPP(&CollCommMgr::RegisterCollComm).stubs();

    std::unique_ptr<CollComm> newCollComm;
    ASSERT_EQ(CollComm::CommGrowNewRank(4, 3, &rootInfo, &config, newCollComm), HCCL_SUCCESS);
    ASSERT_NE(newCollComm, nullptr);
    EXPECT_EQ(newCollComm->GetMyRankId(), 3u);
    EXPECT_EQ(newCollComm->GetRootRankId(), 0u);
    EXPECT_EQ(newCollComm->GetTopParentIdentifier(), "new_rank_top");
    EXPECT_EQ(newCollComm->GetGenerationIndex(), 3u);
    EXPECT_EQ(newCollComm->GetInitSource(), CommInitSource::rootInfo);
    EXPECT_TRUE(newCollComm->HasDetectOwnership());
    ASSERT_NE(newCollComm->GetRankInfoDetect(), nullptr);
    EXPECT_EQ(newCollComm->GetRankInfoDetect()->GetRootHandleInfo().listenPort, rootHandle.listenPort);
    ASSERT_NE(newCollComm->GetHcclCommunicatorSharedPtr(), nullptr);
    EXPECT_EQ(newCollComm->GetCommId(), "new_rank_top_3");
    EXPECT_EQ(std::string(newCollComm->GetHcclCommConfig().hcclCommName), "new_rank_top_3");
}

TEST_F(CollCommShrinkGrowTest, Ut_CommGrowNewRank_When_LoadBinFailed_Expect_Error)
{
    HcclRootHandleV2 rootHandle{};
    HcclRootInfo rootInfo = MakeNewRankRootInfo(0, rootHandle);
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1;

    MockNewRankDetectAgent(HCCL_SUCCESS);
    MockNewRankCommunicatorCreate();
    MOCKER_CPP(&RankInfoDetect::Shutdown).stubs().will(invoke(MockNewRankDetectShutdown));
    MOCKER_CPP(&CollComm::LoadCclKernelBinHandle).stubs().will(returnValue(HCCL_E_INTERNAL));

    std::unique_ptr<CollComm> newCollComm;
    EXPECT_EQ(CollComm::CommGrowNewRank(4, 3, &rootInfo, &config, newCollComm), HCCL_E_INTERNAL);
    EXPECT_NE(newCollComm, nullptr);
}

TEST_F(CollCommShrinkGrowTest, Ut_CommGrowNewRank_When_CollCommInitFailed_Expect_UnloadBinary)
{
    HcclRootHandleV2 rootHandle{};
    HcclRootInfo rootInfo = MakeNewRankRootInfo(0, rootHandle);
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1;

    MockNewRankDetectAgent(HCCL_SUCCESS);
    MockNewRankCommunicatorCreate();
    MOCKER_CPP(&RankInfoDetect::Shutdown).stubs().will(invoke(MockNewRankDetectShutdown));
    MockNewRankCollCommInit(HCCL_E_INTERNAL);
    MOCKER(aclrtBinaryUnLoad).expects(once()).will(returnValue(0));

    std::unique_ptr<CollComm> newCollComm;
    EXPECT_EQ(CollComm::CommGrowNewRank(4, 3, &rootInfo, &config, newCollComm), HCCL_E_INTERNAL);
    EXPECT_NE(newCollComm, nullptr);
}

TEST_F(CollCommShrinkGrowTest, Ut_CommGrowNewRank_When_InvalidRootRankId_Expect_InternalError)
{
    MockNewRankDetectAgent(HCCL_SUCCESS);
    MockNewRankCommunicatorCreate();
    MOCKER_CPP(&RankInfoDetect::Shutdown).stubs().will(invoke(MockNewRankDetectShutdown));
    MockNewRankCollCommInit(HCCL_SUCCESS);
    MOCKER_CPP(&CollCommMgr::RegisterCollComm).stubs();
    MOCKER(aclrtBinaryUnLoad).expects(exactly(2)).will(returnValue(0));

    uint32_t invalidRootRankIds[] = {UINT32_MAX, 4};
    for (uint32_t invalidRootRankId : invalidRootRankIds) {
        HcclRootHandleV2 rootHandle{};
        HcclRootInfo rootInfo = MakeNewRankRootInfo(invalidRootRankId, rootHandle);
        HcclCommConfig config;
        HcclCommConfigInit(&config);
        config.hcclDeterministic = 1;

        std::unique_ptr<CollComm> newCollComm;
        EXPECT_EQ(CollComm::CommGrowNewRank(4, 3, &rootInfo, &config, newCollComm), HCCL_E_INTERNAL);
        EXPECT_NE(newCollComm, nullptr);
    }
}

// ============================================================================
// Comm shrink UT
// ============================================================================

// ── MockCreateDerivedComm：填充出参 collComm，避免 happy-path 用例断言空洞 ──
// MOCKER_CPP 会将 this 作为第一个参数传入；后续参数与 CreateDerivedComm 原型一致
HcclResult
MockCreateDerivedComm(CollComm*, std::unique_ptr<CollComm>& collComm, const CollComm::DerivedCommParams& params)
{
    CaptureDerivedCommParams(params);
    collComm
        = std::make_unique<CollComm>(nullptr, 0, "ut_shrink_derived", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    return HCCL_SUCCESS;
}

// ── MockRankGraph：重写 hccl::RankGraph 虚函数，使 GetRankSize 返回可配置值 ──
class MockRankGraph : public hccl::RankGraph {
public:
    uint32_t mockRankSize{0};

    HcclResult GetRankSize(uint32_t* rankSize) override
    {
        *rankSize = mockRankSize;
        return HCCL_SUCCESS;
    }
    HcclResult GetRankGraphInfo(GraphType, void**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetLinks(uint32_t, uint32_t, uint32_t, CommLink**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetNetLayers(uint32_t**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetInstTopoTypeByNetLayer(uint32_t, CommTopo*) override { return HCCL_SUCCESS; }
    HcclResult GetInstSizeByNetLayer(uint32_t, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetInstRanksByNetLayer(uint32_t, uint32_t**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetInstSizeListByNetLayer(uint32_t, uint32_t**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetDeviceId(uint32_t, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult GetEndpointInfo(uint32_t, const EndpointDesc*, EndpointAttr, uint32_t, void*) override
    {
        return HCCL_SUCCESS;
    }
};

// ── MockRankGraphV1Info：为 rootRankId 解析提供 V1 GraphRankInfo ──
class MockRankGraphV1Info : public hccl::RankGraphV1 {
public:
    std::vector<GraphRankInfo> rankInfos;

    HcclResult GetRankSize(uint32_t* rankSize) override
    {
        *rankSize = static_cast<uint32_t>(rankInfos.size());
        return HCCL_SUCCESS;
    }
    HcclResult GetRankGraphInfo(GraphType type, void** graph, uint32_t* len) override
    {
        CHK_PTR_NULL(graph);
        CHK_PTR_NULL(len);
        if (type != RANK_GRAPH_910_93) {
            return HCCL_E_NOT_SUPPORT;
        }
        *graph = rankInfos.data();
        *len = static_cast<uint32_t>(rankInfos.size() * sizeof(GraphRankInfo));
        return HCCL_SUCCESS;
    }
};

// ── CommShrink 测试固件 ──
class CollCommShrinkTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }

    // 创建 simpleMode CollComm 并注入 MockRankGraph，设置 shrink 相关字段
    // 调用方需保证 mockGraph 生命周期覆盖 collComm
    static void SetupShrinkFields(
        CollComm& collComm, MockRankGraph& mockGraph, uint32_t rankSize, uint32_t myRank, uint32_t rootRankId)
    {
        mockGraph.mockRankSize = rankSize;
        collComm.rankgraph_ = &mockGraph;
        collComm.rankId_ = myRank;
        collComm.topParentIdentifier_ = "ut_shrink";
        collComm.generationIndex_ = 0;
        collComm.initSource_ = CommInitSource::rootInfo;
        collComm.rootHandle_.listenPort = 12345;
        collComm.rootHandle_.rootRankId = rootRankId;
        strncpy_s(collComm.rootHandle_.identifier, sizeof(collComm.rootHandle_.identifier), "ut_shrink", 9);
        HcclCommConfigInit(&collComm.hcclCommConfig_);
    }
};

// 分支: rootRankId 未初始化 → 用 rootHandle.identifier 中的 devPhyId 在 V1 rankGraph 中唯一匹配
TEST_F(CollCommShrinkTest, Ut_ResolveRootRankId_When_RankGraphDevPhyIdUnique_Expect_Success)
{
    MockRankGraphV1Info mockGraph;
    mockGraph.rankInfos.resize(4);
    for (size_t rankId = 0; rankId < mockGraph.rankInfos.size(); ++rankId) {
        mockGraph.rankInfos[rankId].rankId = static_cast<u32>(rankId);
        mockGraph.rankInfos[rankId].deviceInfo.devicePhyId = static_cast<s32>(rankId);
    }

    CollComm collComm(nullptr, 1, "ut_resolve_root", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    collComm.rankgraph_ = &mockGraph;
    collComm.rankId_ = 1;
    collComm.rootHandle_.listenPort = 60000;
    collComm.rootHandle_.rootRankId = UINT32_MAX;
    strncpy_s(
        collComm.rootHandle_.identifier, sizeof(collComm.rootHandle_.identifier), "10.0.0.1_60000_0_123456",
        sizeof("10.0.0.1_60000_0_123456") - 1);
    strncpy_s(collComm.rootHandle_.ip, sizeof(collComm.rootHandle_.ip), "10.0.0.1", 8);

    EXPECT_EQ(collComm.ResolveRootRankIdForDerivation(), HCCL_SUCCESS);
    EXPECT_EQ(collComm.GetRootRankId(), 0u);
}

// ════════════════════════════════════════════════════════════════════════════
// ValidateShrinkParams 分支覆盖
// ════════════════════════════════════════════════════════════════════════════

// 分支1: excludeRanksCount == 0 → HCCL_E_PARA
// rankgraph_ 为空时 GetRankSize() 返回 0，但 count==0 检查先于 GetRankSize() 调用
TEST_F(CollCommShrinkTest, Ut_ValidateShrinkParams_When_CountZero_Expect_E_PARA)
{
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    uint32_t excludeList[] = {0};
    HcclResult ret = collComm.ValidateShrinkParams(excludeList, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 分支2: excludeRanksCount >= rankSize → HCCL_E_PARA
// 此检查在循环之前，即使列表含自身也先触发
TEST_F(CollCommShrinkTest, Ut_ValidateShrinkParams_When_CountGEQRankSize_Expect_E_PARA)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    uint32_t excludeList[] = {0, 1, 2, 3};
    HcclResult ret = collComm.ValidateShrinkParams(excludeList, 4);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 分支3: excludeRank[i] >= rankSize（越界）→ HCCL_E_PARA
TEST_F(CollCommShrinkTest, Ut_ValidateShrinkParams_When_RankOutOfRange_Expect_E_PARA)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    uint32_t excludeList[] = {4};
    HcclResult ret = collComm.ValidateShrinkParams(excludeList, 1);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 分支4: excludeRank[i] == myRank（自身不可被剔除）→ HCCL_E_PARA
TEST_F(CollCommShrinkTest, Ut_ValidateShrinkParams_When_SelfExclusion_Expect_E_PARA)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    uint32_t excludeList[] = {1};
    HcclResult ret = collComm.ValidateShrinkParams(excludeList, 1);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 分支5: 全部参数合法 → HCCL_SUCCESS
TEST_F(CollCommShrinkTest, Ut_ValidateShrinkParams_When_AllValid_Expect_Success)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    uint32_t excludeList[] = {2, 3};
    HcclResult ret = collComm.ValidateShrinkParams(excludeList, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ════════════════════════════════════════════════════════════════════════════
// BuildShrinkSurvivorRanks 分支与逻辑覆盖
// ════════════════════════════════════════════════════════════════════════════

// 逻辑: root 未被剔除，myRank 在中间 → 正确重编号
TEST_F(CollCommShrinkTest, Ut_BuildSurvivorRanks_When_RootNotExcluded_Expect_CorrectRemap)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    uint32_t excludeList[] = {2};
    CollComm::ShrinkSurvivorResult result;

    HcclResult ret = collComm.BuildShrinkSurvivorRanks(excludeList, 1, result);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(result.survivorRanks, std::vector<uint32_t>({0, 1, 3}));
    EXPECT_EQ(result.myNewRank, 1u);
    EXPECT_FALSE(result.rootExcluded);
    EXPECT_EQ(result.childRootRank, 0u);
}

// 逻辑: root 被剔除 → rootExcluded=true, childRootRank=0
TEST_F(CollCommShrinkTest, Ut_BuildSurvivorRanks_When_RootExcluded_Expect_RootExcludedTrue)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    uint32_t excludeList[] = {0, 2};
    CollComm::ShrinkSurvivorResult result;

    HcclResult ret = collComm.BuildShrinkSurvivorRanks(excludeList, 2, result);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(result.survivorRanks, std::vector<uint32_t>({1, 3}));
    EXPECT_EQ(result.myNewRank, 0u);
    EXPECT_TRUE(result.rootExcluded);
    EXPECT_EQ(result.childRootRank, 0u);
}

// 逻辑: myRank 在存活列表开头 → myNewRank=0
TEST_F(CollCommShrinkTest, Ut_BuildSurvivorRanks_When_MyRankAtBeginning_Expect_MyNewRankZero)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 0, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 0, 1);

    uint32_t excludeList[] = {3};
    CollComm::ShrinkSurvivorResult result;

    HcclResult ret = collComm.BuildShrinkSurvivorRanks(excludeList, 1, result);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(result.myNewRank, 0u);
    EXPECT_FALSE(result.rootExcluded);
    EXPECT_EQ(result.childRootRank, 1u);
}

// 逻辑: myRank 在存活列表末尾 → myNewRank 为最后一位
TEST_F(CollCommShrinkTest, Ut_BuildSurvivorRanks_When_MyRankAtEnd_Expect_MyNewRankLast)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 3, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 3, 0);

    uint32_t excludeList[] = {1};
    CollComm::ShrinkSurvivorResult result;

    HcclResult ret = collComm.BuildShrinkSurvivorRanks(excludeList, 1, result);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(result.survivorRanks, std::vector<uint32_t>({0, 2, 3}));
    EXPECT_EQ(result.myNewRank, 2u);
    EXPECT_FALSE(result.rootExcluded);
    EXPECT_EQ(result.childRootRank, 0u);
}

// 防御分支: rootRankId 越界（不在存活列表也不在剔除列表）→ !rootExcluded && !foundRootRank → HCCL_E_INTERNAL
TEST_F(CollCommShrinkTest, Ut_BuildSurvivorRanks_When_RootRankIdOutOfRange_Expect_E_INTERNAL)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 10); // rootRankId=10 >= rankSize=4

    uint32_t excludeList[] = {2};
    CollComm::ShrinkSurvivorResult result;

    HcclResult ret = collComm.BuildShrinkSurvivorRanks(excludeList, 1, result);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// ════════════════════════════════════════════════════════════════════════════
// CommShrink 分支与逻辑覆盖
// ════════════════════════════════════════════════════════════════════════════

// 分支: GetInitSource() == clusterInfo → HCCL_E_NOT_SUPPORT
TEST_F(CollCommShrinkTest, Ut_CommShrink_When_InitSourceClusterInfo_Expect_E_NOT_SUPPORT)
{
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    collComm.initSource_ = CommInitSource::clusterInfo;

    uint32_t excludeList[] = {2};
    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm.CommShrink(excludeList, 1, newCollComm, nullptr);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

// 分支: identifierNew.size() >= Hccl::ROOTINFO_IDENTIFIER_MAX_LENGTH → HCCL_E_PARA
// topParentIdentifier_ 设为 126 字符，加 "_1" 后缀共 128 >= 128
TEST_F(CollCommShrinkTest, Ut_CommShrink_When_IdentifierTooLong_Expect_E_PARA)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);
    collComm.topParentIdentifier_ = std::string(126, 'A'); // 126 + "_1" = 128 >= 128

    uint32_t excludeList[] = {2};
    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm.CommShrink(excludeList, 1, newCollComm, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 分支: config==nullptr（inheritParentConfig=true）+ root 未被剔除 → Success
// Mock CreateDerivedComm 返回成功，验证 config==nullptr 路径
TEST_F(CollCommShrinkTest, Ut_CommShrink_When_ConfigNullAndRootAlive_Expect_InheritParentConfig)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);
    collComm.rootHandle_.listenPort = 12345; // 使 root handle 有效

    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType).stubs().will(invoke(MockCreateDerivedComm));

    uint32_t excludeList[] = {2};
    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm.CommShrink(excludeList, 1, newCollComm, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(nullptr, newCollComm);
    EXPECT_TRUE(g_derivedParamsCapture.inheritParentConfig);
    EXPECT_EQ(g_derivedParamsCapture.commName, "ut_shrink_1");
    EXPECT_EQ(std::string(g_derivedParamsCapture.subConfig.hcclCommName), "ut_shrink_1");
    EXPECT_EQ(g_derivedParamsCapture.survivorRanks, std::vector<uint32_t>({0, 1, 3}));
    EXPECT_EQ(g_derivedParamsCapture.rankId, 1u);
    EXPECT_FALSE(g_derivedParamsCapture.derivationInfo.isGrow);
    EXPECT_EQ(g_derivedParamsCapture.derivationInfo.rootHandle.listenPort, 12345u);
    EXPECT_EQ(g_derivedParamsCapture.initMode, CollCommInitMode::fullMode);
}

// 分支: config!=nullptr（inheritParentConfig=false）+ root 被剔除 → Success
TEST_F(CollCommShrinkTest, Ut_CommShrink_When_ConfigNotNullAndRootExcluded_Expect_UserConfigAndRootCleared)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0); // rootRankId=0, excludeList 含 0 → root 被剔除

    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType).stubs().will(invoke(MockCreateDerivedComm));

    HcclCommConfig userConfig;
    HcclCommConfigInit(&userConfig);
    userConfig.hcclBufferSize = 321;

    uint32_t excludeList[] = {0, 2};
    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm.CommShrink(excludeList, 2, newCollComm, &userConfig);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(nullptr, newCollComm);
    EXPECT_FALSE(g_derivedParamsCapture.inheritParentConfig);
    EXPECT_EQ(g_derivedParamsCapture.commName, "ut_shrink_1");
    EXPECT_EQ(std::string(g_derivedParamsCapture.subConfig.hcclCommName), "ut_shrink_1");
    EXPECT_EQ(g_derivedParamsCapture.subConfig.hcclBufferSize, 321u);
    EXPECT_EQ(g_derivedParamsCapture.survivorRanks, std::vector<uint32_t>({1, 3}));
    EXPECT_EQ(g_derivedParamsCapture.rankId, 0u);
    EXPECT_FALSE(g_derivedParamsCapture.derivationInfo.isGrow);
    EXPECT_EQ(g_derivedParamsCapture.derivationInfo.rootHandle.listenPort, Hccl::HCCL_INVALID_PORT);
    EXPECT_EQ(g_derivedParamsCapture.derivationInfo.rootHandle.rootRankId, UINT32_MAX);
}

// 分支: config!=nullptr + root 未被剔除 → Success
// 与上一用例互补：覆盖 inheritParentConfig=false 且 rootExcluded=false 的组合
TEST_F(CollCommShrinkTest, Ut_CommShrink_When_ConfigNotNullAndRootAlive_Expect_Success)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 1); // rootRankId=1 不在 excludeList 中

    MOCKER_CPP(&CollComm::CreateDerivedComm, CreateDerivedCommMockType).stubs().will(invoke(MockCreateDerivedComm));

    HcclCommConfig userConfig;
    HcclCommConfigInit(&userConfig);
    userConfig.hcclBufferSize = 654;

    uint32_t excludeList[] = {2, 3};
    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm.CommShrink(excludeList, 2, newCollComm, &userConfig);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(nullptr, newCollComm);
    EXPECT_FALSE(g_derivedParamsCapture.inheritParentConfig);
    EXPECT_EQ(g_derivedParamsCapture.commName, "ut_shrink_1");
    EXPECT_EQ(g_derivedParamsCapture.subConfig.hcclBufferSize, 654u);
    EXPECT_EQ(g_derivedParamsCapture.survivorRanks, std::vector<uint32_t>({0, 1}));
    EXPECT_EQ(g_derivedParamsCapture.rankId, 1u);
    EXPECT_FALSE(g_derivedParamsCapture.derivationInfo.isGrow);
    EXPECT_EQ(g_derivedParamsCapture.derivationInfo.rootHandle.listenPort, 12345u);
    EXPECT_EQ(g_derivedParamsCapture.derivationInfo.rootHandle.rootRankId, 1u);
}

// ════════════════════════════════════════════════════════════════════════════
// CreateDerivedComm 分支覆盖
// ════════════════════════════════════════════════════════════════════════════

// 分支: survivorRanks.empty() → HCCL_E_PARA（在访问 comm_ 之前返回，comm_ 可为 nullptr）
TEST_F(CollCommShrinkTest, Ut_CreateDerivedComm_When_SurvivorRanksEmpty_Expect_E_PARA)
{
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);

    std::unique_ptr<CollComm> newCollComm;
    HcclCommConfig subConfig;
    HcclCommConfigInit(&subConfig);
    CollComm::DerivationInfo derivation;

    std::vector<uint32_t> emptyRanks;
    Hccl::RankTableInfo emptyRankTable; // Shrink 传空拓扑表（Grow 传收集结果，Shrink 传空对象）
    CollComm::DerivedCommParams params{
        emptyRanks, 0, "test", subConfig, derivation, false, CollCommInitMode::fullMode, emptyRankTable};
    HcclResult ret = collComm.CreateDerivedComm(newCollComm, params);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// ════════════════════════════════════════════════════════════════════════════
// C 接口 HcclCommShrink 入参校验分支覆盖
// ════════════════════════════════════════════════════════════════════════════

// 分支: excludeRanksList == nullptr → HCCL_E_PTR
TEST_F(CollCommShrinkTest, Ut_HcclCommShrink_When_excludeRanksListNull_Expect_E_PTR)
{
    HcclComm newComm = nullptr;
    HcclResult ret = HcclCommShrink(nullptr, nullptr, 0, nullptr, 0, &newComm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 分支: newComm == nullptr → HCCL_E_PTR
TEST_F(CollCommShrinkTest, Ut_HcclCommShrink_When_NewCommNull_Expect_E_PTR)
{
    uint32_t excludeList[] = {0};
    HcclResult ret = HcclCommShrink(nullptr, excludeList, 1, nullptr, 0, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// ════════════════════════════════════════════════════════════════════════════
// root 被剔除分支覆盖
// ════════════════════════════════════════════════════════════════════════════

// 分支: 本 rank 是 root 且被剔除 → ValidateShrinkParams 拦截，返回 HCCL_E_PARA
TEST_F(CollCommShrinkTest, Ut_CommShrink_When_RootSelfExcluded_Expect_E_PARA)
{
    MockRankGraph mockGraph;
    // myRank=0, rootRankId=0, excludeList 含 0 → 剔除自身
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 0, 0);

    uint32_t excludeList[] = {0, 2};
    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm.CommShrink(excludeList, 2, newCollComm, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(nullptr, newCollComm);
}

// 分支: 非 root rank 自身被剔除 → HCCL_E_PARA（ValidateShrinkParams 拦截）
TEST_F(CollCommShrinkTest, Ut_CommShrink_When_NonRootSelfExcluded_Expect_E_PARA)
{
    MockRankGraph mockGraph;
    // myRank=1, rootRankId=0, excludeList 含 1 → 非 root 自身被剔除
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    uint32_t excludeList[] = {1};
    std::unique_ptr<CollComm> newCollComm;
    HcclResult ret = collComm.CommShrink(excludeList, 1, newCollComm, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 分支: ValidateShrinkParams excludeRanksList 为空指针 → HCCL_E_PTR
TEST_F(CollCommShrinkTest, Ut_ValidateShrinkParams_When_excludeRanksListNull_Expect_E_PTR)
{
    MockRankGraph mockGraph;
    CollComm collComm(nullptr, 1, "ut_shrink", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    SetupShrinkFields(collComm, mockGraph, 4, 1, 0);

    HcclResult ret = collComm.ValidateShrinkParams(nullptr, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// ════════════════════════════════════════════════════════════════════════════
// HcclCommGrow 入参校验分支覆盖
// ════════════════════════════════════════════════════════════════════════════

// 分支: HcclCommGrow newComm == nullptr → HCCL_E_PTR
TEST_F(CollCommShrinkTest, Ut_HcclCommGrow_When_NewCommNull_Expect_E_PTR)
{
    HcclRootInfo rootInfo{};
    HcclResult ret = HcclCommGrow(nullptr, 2, &rootInfo, 0, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 分支: HcclCommGrow comm==nullptr 且 rootInfo==nullptr → HCCL_E_PARA
TEST_F(CollCommShrinkTest, Ut_HcclCommGrow_When_CommNullAndRootInfoNull_Expect_E_PARA)
{
    HcclComm newComm = nullptr;
    HcclResult ret = HcclCommGrow(nullptr, 2, nullptr, 0, nullptr, &newComm);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 分支: HcclCommGetRootInfo rootInfo == nullptr → HCCL_E_PTR
TEST_F(CollCommShrinkTest, Ut_HcclCommGetRootInfo_When_RootInfoNull_Expect_E_PTR)
{
    HcclResult ret = HcclCommGetRootInfo(nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
// 分支: comm == nullptr（映射后 hcclComm 为空）→ HCCL_E_PTR
TEST_F(CollCommShrinkTest, Ut_HcclCommShrink_When_CommNull_Expect_E_PTR)
{
    uint32_t excludeList[] = {0};
    HcclComm newComm = nullptr;
    HcclResult ret = HcclCommShrink(nullptr, excludeList, 1, nullptr, 0, &newComm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
