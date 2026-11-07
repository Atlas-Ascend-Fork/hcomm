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
#include "mockcpp/mockcpp.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>

#define private public
#include "hccl_comm_pub.h"
#include "coll_comm.h"
#include "my_rank.h"
#include "ccu_instance_mgr.h"
#undef private
#include "base_comm_node_accessors.h"

#include "ccu/ccu_res.h"
#include "hccl/hccl_ccu_res.h"
#include "comm_engine/ccu/ccu_device_res_pub.h"
#include "ccu_instance.h"

using namespace hccl;
using namespace hcomm;

class HcclCommAssignCcuInsTest : public testing::Test {
protected:
    void TearDown() override
    {
        GetCcuInstanceMgr(0).Deinit();
        GlobalMockObject::verify();
    }

    HcclComm BuildComm(
        bool isCommunicatorV2 = true, bool hasCollComm = true, bool hasMyRank = true,
        CcuInsHandle currentAssignedInsHandle = 0)
    {
        comm_ = std::make_unique<hcclComm>(0, 0, "assign_ccu_ut");
        // hcclComm::IsCommunicatorV2() 已改为实时调 hcomm::EqualToOrGreaterThan950()，
        // 不再读成员 devType_，故此处须桩住 hrtGetDeviceType 才能驱动 V2 真假；
        // outBound 按值拷贝，局部量出作用域后桩仍有效。
        DevType stubDevType = isCommunicatorV2 ? DevType::DEV_TYPE_950 : DevType::DEV_TYPE_COUNT;
        MOCKER(hrtGetDeviceType).stubs().with(outBound(stubDevType)).will(returnValue(HCCL_SUCCESS));

        if (!hasCollComm) {
            return static_cast<HcclComm>(comm_.get());
        }

        comm_->collComm_
            = std::make_unique<CollComm>(nullptr, 0, "assign_ccu_ut", ManagerCallbacks{}, CollCommInitMode::simpleMode);
        comm_->collComm_->deviceLogicId_ = 0;
        if (!hasMyRank) {
            return static_cast<HcclComm>(comm_.get());
        }

        myRank_ = std::make_shared<MyRank>(nullptr, 0, comm_->collComm_->config_, ManagerCallbacks{}, nullptr, nullptr);
        // 语义变更：HcclCommAssignCcuIns 改写 assignedCcuInsHandle_（不再写 ccuInsHandle_）
        myRank_->assignedCcuInsHandle_ = currentAssignedInsHandle;
        comm_->collComm_->myRank_ = myRank_;
        return static_cast<HcclComm>(comm_.get());
    }

    CcuInsHandle MockCcuInstanceFound(CcuInsHandle insHandle = validInsHandle_)
    {
        auto& mgr = GetCcuInstanceMgr(0);
        mgr.initializedFlag_ = true;
        auto instance = std::make_unique<CcuInstance>();
        instance->SetHandle(insHandle);
        instance->snapshot_ = std::make_unique<HcommCcuInstanceEntity>();
        instance->snapshot_->ccuInsKey = insHandle;
        instance->snapshotReady_ = true;
        const auto insEntityAddress = reinterpret_cast<CcuInsHandle>(instance->snapshot_.get());
        mgr.entityAddressMap_.emplace(insEntityAddress, insHandle);
        mgr.insMap_.emplace(insHandle, std::move(instance));
        return insEntityAddress;
    }

    static constexpr CcuInsHandle validInsHandle_ = 100;
    std::unique_ptr<hcclComm> comm_;
    std::shared_ptr<MyRank> myRank_;
};

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommAssignCcuIns_When_CommHasNoCcuInstance_Expect_ReturnIsHCCL_SUCCESS)
{
    constexpr CcuInsHandle insHandle = validInsHandle_;
    HcclComm comm = BuildComm(true, true, true, 0);
    const auto entityHandle = MockCcuInstanceFound();

    HcclResult ret = HcclCommAssignCcuIns(comm, entityHandle);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(myRank_->GetAssignedCcuInstance(), entityHandle);
    EXPECT_EQ(myRank_->GetCcuInstance(), 0); // ccuInsHandle_ 不被 Assign 改写
}

TEST_F(
    HcclCommAssignCcuInsTest, Ut_HcclCommAssignCcuIns_When_AssignSuccessAndCommDestroyed_Expect_DestroyCcuInstanceOnce)
{
    constexpr CcuInsHandle insHandle = validInsHandle_;
    HcclComm comm = BuildComm(true, true, true, 0);
    std::weak_ptr<MyRank> myRankWeak = myRank_;
    const auto entityHandle = MockCcuInstanceFound();
    MOCKER(HcommCcuInsDestroy).expects(once()).with(entityHandle).will(returnValue(CcuResult::CCU_SUCCESS));

    HcclResult ret = HcclCommAssignCcuIns(comm, entityHandle);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(myRank_->GetAssignedCcuInstance(), entityHandle);
    myRank_.reset();
    comm_.reset();
    EXPECT_TRUE(myRankWeak.expired());
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommAssignCcuIns_When_CommIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    constexpr CcuInsHandle insHandle = 1;

    HcclResult ret = HcclCommAssignCcuIns(nullptr, insHandle);

    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommAssignCcuIns_When_InsHandleIsZero_Expect_ReturnIsHCCL_E_PARA)
{
    HcclComm comm = BuildComm();

    HcclResult ret = HcclCommAssignCcuIns(comm, 0);

    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(myRank_->GetAssignedCcuInstance(), 0);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommAssignCcuIns_When_CommIsNotV2_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    constexpr CcuInsHandle insHandle = 1;
    HcclComm comm = BuildComm(false);

    HcclResult ret = HcclCommAssignCcuIns(comm, insHandle);

    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(myRank_->GetAssignedCcuInstance(), 0);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommAssignCcuIns_When_CollCommIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    constexpr CcuInsHandle insHandle = 1;
    HcclComm comm = BuildComm(true, false);

    HcclResult ret = HcclCommAssignCcuIns(comm, insHandle);

    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommAssignCcuIns_When_MyRankIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    constexpr CcuInsHandle insHandle = 1;
    HcclComm comm = BuildComm(true, true, false);

    HcclResult ret = HcclCommAssignCcuIns(comm, insHandle);

    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommAssignCcuIns_When_InsHandleNotFound_Expect_ReturnIsHCCL_E_NOT_FOUND)
{
    constexpr CcuInsHandle insHandle = validInsHandle_;
    HcclComm comm = BuildComm(true, true, true, 0);
    GetCcuInstanceMgr(0).Deinit();

    HcclResult ret = HcclCommAssignCcuIns(comm, insHandle);

    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    EXPECT_EQ(myRank_->GetAssignedCcuInstance(), 0);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommAssignCcuIns_When_CommHasDifferentCcuInstance_Expect_ReturnIsHCCL_E_PARA)
{
    const auto oldEntityHandle = MockCcuInstanceFound(validInsHandle_);
    const auto newEntityHandle = MockCcuInstanceFound(200);
    HcclComm comm = BuildComm(true, true, true, oldEntityHandle);

    HcclResult ret = HcclCommAssignCcuIns(comm, newEntityHandle);

    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(myRank_->GetAssignedCcuInstance(), oldEntityHandle);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommAssignCcuIns_When_CommHasSameCcuInstance_Expect_ReturnIsHCCL_E_PARA)
{
    const auto entityHandle = MockCcuInstanceFound();
    HcclComm comm = BuildComm(true, true, true, entityHandle);

    HcclResult ret = HcclCommAssignCcuIns(comm, entityHandle);

    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(myRank_->GetAssignedCcuInstance(), entityHandle);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommAssignCcuIns_When_InternalExceptionOccurs_Expect_ReturnIsHCCL_E_INTERNAL)
{
    constexpr CcuInsHandle insHandle = validInsHandle_;
    HcclComm comm = BuildComm(true, true, true, 0);
    const auto entityHandle = MockCcuInstanceFound();

    CcuInsHandle ccuInsKey = 0;
    MOCKER_CPP(&CcuInstanceMgr::GetCcuInsKey)
        .expects(once())
        .with(eq(entityHandle), outBound(ccuInsKey))
        .will(throws(std::logic_error("test exception")));

    HcclResult ret = HcclCommAssignCcuIns(comm, entityHandle);

    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(myRank_->GetAssignedCcuInstance(), 0);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommAssignCcuIns_When_TwoThreadsAssignConcurrently_Expect_OnlyOneSucceeds)
{
    constexpr CcuInsHandle firstInsHandle = validInsHandle_;
    constexpr CcuInsHandle secondInsHandle = 200;
    HcclComm comm = BuildComm(true, true, true, 0);
    const auto firstEntityHandle = MockCcuInstanceFound(firstInsHandle);
    const auto secondEntityHandle = MockCcuInstanceFound(secondInsHandle);

    std::atomic<uint32_t> readyCount{0};
    std::atomic<bool> start{false};
    HcclResult firstRet = HCCL_E_INTERNAL;
    HcclResult secondRet = HCCL_E_INTERNAL;
    auto assign = [&](CcuInsHandle insHandle, HcclResult& ret) {
        readyCount.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        ret = HcclCommAssignCcuIns(comm, insHandle);
    };

    std::thread firstThread(assign, firstEntityHandle, std::ref(firstRet));
    std::thread secondThread(assign, secondEntityHandle, std::ref(secondRet));
    while (readyCount.load(std::memory_order_acquire) != 2) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    firstThread.join();
    secondThread.join();

    const bool firstSucceeded = firstRet == HCCL_SUCCESS && secondRet == HCCL_E_PARA;
    const bool secondSucceeded = firstRet == HCCL_E_PARA && secondRet == HCCL_SUCCESS;
    EXPECT_TRUE(firstSucceeded || secondSucceeded);
    EXPECT_EQ(myRank_->GetAssignedCcuInstance(), firstSucceeded ? firstEntityHandle : secondEntityHandle);
}

// ============================================================================================
// HcclCommQueryCcuIns 单元测试（阶段2新语义：查/建通信域自有的 ccuInsHandle_，旧方式）
// 对应业务代码：src/coll_communicator_mgr/api_c_adpt/coll_comm_ccu_c_adpt.cc
// ============================================================================================

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryCcuIns_When_SelfHandleExists_Expect_ReturnSuccessAndNoCreate)
{
    HcclComm comm = BuildComm(true, true, true, 0);
    const auto selfHandle = MockCcuInstanceFound();
    myRank_->ccuInsHandle_ = selfHandle; // 通信域自有 handle 已存在
    // 按需创建不应被调用
    MOCKER(HcommCcuInsCreateLegacy).expects(never());

    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 0;
    HcclResult ret = HcclCommQueryCcuIns(comm, insHandles, &insNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(insHandles[0], selfHandle);
    EXPECT_EQ(insNum, 1U);
    EXPECT_EQ(myRank_->GetCcuInstance(), selfHandle); // 未被改写
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryCcuIns_When_SelfHandleZero_And_ModeValid_Expect_CreateAndWriteback)
{
    HcclComm comm = BuildComm(true, true, true, 0);
    myRank_->ccuInsHandle_ = 0;
    myRank_->opExpansionMode_ = 6; // CCU_SCHED_MODE 映射为 CCU_SCHED
    CcuInsHandle createdHandle = MockCcuInstanceFound();
    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 0;
    // mock 按需创建：通过 outBoundP 把 createdHandle 拷贝到出参指针 *ccuInsHandle，并返回成功
    MOCKER(HcommCcuInsCreateLegacy)
        .expects(once())
        .with(mockcpp::any(), mockcpp::outBoundP(&createdHandle, sizeof(CcuInsHandle)))
        .will(returnValue(CcuResult::CCU_SUCCESS));

    HcclResult ret = HcclCommQueryCcuIns(comm, insHandles, &insNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(insHandles[0], createdHandle);
    EXPECT_EQ(insNum, 1U);
    EXPECT_EQ(myRank_->GetCcuInstance(), createdHandle); // 已回写
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryCcuIns_When_SelfHandleZero_And_ModeUnused_Expect_ReturnUnavail)
{
    HcclComm comm = BuildComm(true, true, true, 0);
    myRank_->ccuInsHandle_ = 0;
    myRank_->opExpansionMode_ = 0; // DEFAULT_MODE 映射为 CCU_UNUSED
    MOCKER(HcommCcuInsCreateLegacy).expects(never());

    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 0;
    HcclResult ret = HcclCommQueryCcuIns(comm, insHandles, &insNum);

    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
    EXPECT_EQ(insHandles[0], static_cast<CcuInsHandle>(0)); // 出参未修改
    EXPECT_EQ(insNum, 0U);
    EXPECT_EQ(myRank_->GetCcuInstance(), 0); // 未创建
}

TEST_F(
    HcclCommAssignCcuInsTest,
    Ut_HcclCommQueryCcuIns_When_SelfHandleZero_And_CreateFails_Expect_ReturnErrorAndNoWriteback)
{
    HcclComm comm = BuildComm(true, true, true, 0);
    myRank_->ccuInsHandle_ = 0;
    myRank_->opExpansionMode_ = 5; // CCU_MS_MODE 映射为 CCU_MS
    MOCKER(HcommCcuInsCreateLegacy).expects(once()).will(returnValue(CcuResult::CCU_E_DRV_BUSY));

    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 0;
    HcclResult ret = HcclCommQueryCcuIns(comm, insHandles, &insNum);

    EXPECT_EQ(ret, static_cast<HcclResult>(CcuResult::CCU_E_DRV_BUSY));
    EXPECT_EQ(myRank_->GetCcuInstance(), 0); // 失败不回写
}

// CCU_MS 资源不足时降级到 CCU_SCHED 重试，CCU_SCHED 创建成功 → 返回成功并回写降级后创建的 handle
TEST_F(
    HcclCommAssignCcuInsTest, Ut_HcclCommQueryCcuIns_When_CcuMsUnavail_And_FallbackCcuSchedSuccess_Expect_ReturnSuccess)
{
    HcclComm comm = BuildComm(true, true, true, 0);
    myRank_->ccuInsHandle_ = 0;
    myRank_->opExpansionMode_ = 5; // CCU_MS_MODE 映射为 CCU_MS
    CcuInsHandle createdHandle = MockCcuInstanceFound();
    // 首次 CCU_MS 返回资源不足，降级到 CCU_SCHED 重试一次成功。
    // 用 eq(CCU_MS)/eq(CCU_SCHED) 区分入参，使 mockcpp 按参数精确匹配两次调用
    // （多次 expects(once()) 无 with 时会误匹配首条，导致第二次调用报错）
    MOCKER(HcommCcuInsCreateLegacy)
        .expects(once())
        .with(mockcpp::eq(CcuInstanceType::CCU_MS), mockcpp::any())
        .will(returnValue(CcuResult::CCU_E_UNAVAIL));
    MOCKER(HcommCcuInsCreateLegacy)
        .expects(once())
        .with(mockcpp::eq(CcuInstanceType::CCU_SCHED), mockcpp::outBoundP(&createdHandle, sizeof(CcuInsHandle)))
        .will(returnValue(CcuResult::CCU_SUCCESS));

    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 0;
    HcclResult ret = HcclCommQueryCcuIns(comm, insHandles, &insNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(insHandles[0], createdHandle);
    EXPECT_EQ(insNum, 1U);
    EXPECT_EQ(myRank_->GetCcuInstance(), createdHandle); // 降级后创建的 handle 已回写
}

// CCU_MS 资源不足降级到 CCU_SCHED 仍资源不足 → 返回 HCCL_E_UNAVAIL，不创建、不回写
TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryCcuIns_When_CcuMsAndSchedBothUnavail_Expect_ReturnUnavail)
{
    HcclComm comm = BuildComm(true, true, true, 0);
    myRank_->ccuInsHandle_ = 0;
    myRank_->opExpansionMode_ = 5; // CCU_MS_MODE 映射为 CCU_MS
    // 首次 CCU_MS 资源不足，降级 CCU_SCHED 仍资源不足（共调用两次）。
    // 用 eq(CCU_MS)/eq(CCU_SCHED) 区分入参，避免两次 expects(once()) 互相误匹配
    MOCKER(HcommCcuInsCreateLegacy)
        .expects(once())
        .with(mockcpp::eq(CcuInstanceType::CCU_MS), mockcpp::any())
        .will(returnValue(CcuResult::CCU_E_UNAVAIL));
    MOCKER(HcommCcuInsCreateLegacy)
        .expects(once())
        .with(mockcpp::eq(CcuInstanceType::CCU_SCHED), mockcpp::any())
        .will(returnValue(CcuResult::CCU_E_UNAVAIL));

    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 0;
    HcclResult ret = HcclCommQueryCcuIns(comm, insHandles, &insNum);

    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
    EXPECT_EQ(insHandles[0], static_cast<CcuInsHandle>(0)); // 出参未修改
    EXPECT_EQ(insNum, 0U);
    EXPECT_EQ(myRank_->GetCcuInstance(), 0); // 失败不回写
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryCcuIns_When_CommIsNotV2_Expect_ReturnNotSupport)
{
    HcclComm comm = BuildComm(false, true, true, 0);
    MOCKER(HcommCcuInsCreateLegacy).expects(never());

    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 0;
    HcclResult ret = HcclCommQueryCcuIns(comm, insHandles, &insNum);

    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryCcuIns_When_CommIsNull_Expect_ReturnPtr)
{
    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 0;
    EXPECT_EQ(HcclCommQueryCcuIns(nullptr, insHandles, &insNum), HCCL_E_PTR);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryCcuIns_When_InsHandlesIsNull_Expect_ReturnPtr)
{
    HcclComm comm = BuildComm();
    uint32_t insNum = 0;
    EXPECT_EQ(HcclCommQueryCcuIns(comm, nullptr, &insNum), HCCL_E_PTR);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryCcuIns_When_InsNumIsNull_Expect_ReturnPtr)
{
    HcclComm comm = BuildComm();
    CcuInsHandle insHandles[1] = {0};
    EXPECT_EQ(HcclCommQueryCcuIns(comm, insHandles, nullptr), HCCL_E_PTR);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryCcuIns_When_RepeatQuery_Expect_IdempotentAndNoSecondCreate)
{
    HcclComm comm = BuildComm(true, true, true, 0);
    myRank_->ccuInsHandle_ = 0;
    myRank_->opExpansionMode_ = 6; // CCU_SCHED
    CcuInsHandle createdHandle = MockCcuInstanceFound();
    MOCKER(HcommCcuInsCreateLegacy)
        .expects(once()) // 仅首次创建
        .with(mockcpp::any(), mockcpp::outBoundP(&createdHandle, sizeof(CcuInsHandle)))
        .will(returnValue(CcuResult::CCU_SUCCESS));

    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 0;
    EXPECT_EQ(HcclCommQueryCcuIns(comm, insHandles, &insNum), HCCL_SUCCESS);
    EXPECT_EQ(insHandles[0], createdHandle);
    // 第二次 Query：handle 已存在，不应再创建
    CcuInsHandle insHandles2[1] = {0};
    uint32_t insNum2 = 0;
    EXPECT_EQ(HcclCommQueryCcuIns(comm, insHandles2, &insNum2), HCCL_SUCCESS);
    EXPECT_EQ(insHandles2[0], createdHandle);
    EXPECT_EQ(myRank_->GetCcuInstance(), createdHandle);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryCcuIns_When_InstanceExists_Expect_ReturnStoredHandleAndNoCreate)
{
    HcclComm comm = BuildComm(true, true, true, 0);
    const auto selfHandle = MockCcuInstanceFound();
    myRank_->ccuInsHandle_ = selfHandle;
    MOCKER(HcommCcuInsCreateLegacy).expects(never());

    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 0;
    HcclResult firstRet = HcclCommQueryCcuIns(comm, insHandles, &insNum);

    CcuInsHandle insHandles2[1] = {0};
    uint32_t insNum2 = 0;
    HcclResult secondRet = HcclCommQueryCcuIns(comm, insHandles2, &insNum2);

    EXPECT_EQ(firstRet, HCCL_SUCCESS);
    EXPECT_EQ(secondRet, HCCL_SUCCESS);
    EXPECT_EQ(insNum, 1U);
    EXPECT_EQ(insNum2, 1U);
    EXPECT_NE(insHandles[0], 0U);
    EXPECT_EQ(insHandles2[0], insHandles[0]);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryAssignedCcuIns_When_InstanceExists_Expect_UseAssignedInstance)
{
    const auto assignedHandle = MockCcuInstanceFound();
    HcclComm comm = BuildComm(true, true, true, assignedHandle);

    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 0;
    HcclResult ret = HcclCommQueryAssignedCcuIns(comm, insHandles, &insNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(insNum, 1U);
    EXPECT_EQ(myRank_->GetCcuInstance(), 0U);
    EXPECT_EQ(insHandles[0], assignedHandle);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryCcuIns_When_InstanceDestroyed_Expect_ReturnNotFound)
{
    HcclComm comm = BuildComm(true, true, true, 0);
    const auto entityHandle = MockCcuInstanceFound();
    myRank_->ccuInsHandle_ = entityHandle;
    auto& mgr = GetCcuInstanceMgr(0);
    CcuInsHandle ccuInsKey = 0;
    ASSERT_EQ(mgr.GetCcuInsKey(entityHandle, ccuInsKey), CcuResult::CCU_SUCCESS);
    ASSERT_EQ(mgr.Destroy(ccuInsKey), CcuResult::CCU_SUCCESS);

    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 1;
    EXPECT_EQ(HcclCommQueryCcuIns(comm, insHandles, &insNum), HCCL_E_NOT_FOUND);
    EXPECT_EQ(insNum, 0U);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryAssignedCcuIns_When_InstanceDestroyed_Expect_ReturnNotFound)
{
    HcclComm comm = BuildComm(true, true, true, 0);
    const auto entityHandle = MockCcuInstanceFound();
    myRank_->assignedCcuInsHandle_ = entityHandle;
    auto& mgr = GetCcuInstanceMgr(0);
    CcuInsHandle ccuInsKey = 0;
    ASSERT_EQ(mgr.GetCcuInsKey(entityHandle, ccuInsKey), CcuResult::CCU_SUCCESS);
    ASSERT_EQ(mgr.Destroy(ccuInsKey), CcuResult::CCU_SUCCESS);

    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 1;
    EXPECT_EQ(HcclCommQueryAssignedCcuIns(comm, insHandles, &insNum), HCCL_E_NOT_FOUND);
    EXPECT_EQ(insNum, 0U);
}

// ============================================================================================
// HcclCommQueryAssignedCcuIns 单元测试（新增接口：查询新方式绑定的 assignedCcuInsHandle_）
// ============================================================================================

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryAssignedCcuIns_When_Bound_Expect_ReturnSuccess)
{
    const auto assignedHandle = MockCcuInstanceFound();
    HcclComm comm = BuildComm(true, true, true, assignedHandle);
    // 不应创建任何 instance
    MOCKER(HcommCcuInsCreateLegacy).expects(never());

    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 0;
    HcclResult ret = HcclCommQueryAssignedCcuIns(comm, insHandles, &insNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(insHandles[0], assignedHandle);
    EXPECT_EQ(insNum, 1U);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryAssignedCcuIns_When_NotBound_Expect_ReturnUnavail)
{
    HcclComm comm = BuildComm(true, true, true, 0);   // assignedCcuInsHandle_ = 0
    MOCKER(HcommCcuInsCreateLegacy).expects(never()); // 不创建

    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 0;
    HcclResult ret = HcclCommQueryAssignedCcuIns(comm, insHandles, &insNum);

    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
    EXPECT_EQ(insHandles[0], static_cast<CcuInsHandle>(0)); // 出参未修改
    EXPECT_EQ(insNum, 0U);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryAssignedCcuIns_When_CommIsNotV2_Expect_ReturnNotSupport)
{
    HcclComm comm = BuildComm(false, true, true, 0xEF);
    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 0;
    EXPECT_EQ(HcclCommQueryAssignedCcuIns(comm, insHandles, &insNum), HCCL_E_NOT_SUPPORT);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryAssignedCcuIns_When_CommIsNull_Expect_ReturnPtr)
{
    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 0;
    EXPECT_EQ(HcclCommQueryAssignedCcuIns(nullptr, insHandles, &insNum), HCCL_E_PTR);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryAssignedCcuIns_When_InsHandlesIsNull_Expect_ReturnPtr)
{
    HcclComm comm = BuildComm();
    uint32_t insNum = 0;
    EXPECT_EQ(HcclCommQueryAssignedCcuIns(comm, nullptr, &insNum), HCCL_E_PTR);
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryAssignedCcuIns_When_InsNumIsNull_Expect_ReturnPtr)
{
    HcclComm comm = BuildComm();
    CcuInsHandle insHandles[1] = {0};
    EXPECT_EQ(HcclCommQueryAssignedCcuIns(comm, insHandles, nullptr), HCCL_E_PTR);
}

// ============================================================================================
// HcclCommAssignCcuIns 语义变更补充：写 assignedCcuInsHandle_，ccuInsHandle_ 不变
// ============================================================================================

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommAssignCcuIns_When_Success_Expect_CcuInsHandleUnchanged)
{
    constexpr CcuInsHandle selfHandle = 0xA;
    constexpr CcuInsHandle assignedHandle = 0xB;
    HcclComm comm = BuildComm(true, true, true, 0);
    myRank_->ccuInsHandle_ = selfHandle; // 通信域自有 handle
    const auto assignedEntityHandle = MockCcuInstanceFound(assignedHandle);

    HcclResult ret = HcclCommAssignCcuIns(comm, assignedEntityHandle);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(myRank_->GetAssignedCcuInstance(), assignedEntityHandle);
    EXPECT_EQ(myRank_->GetCcuInstance(), selfHandle); // 自有 handle 不被改写
}

TEST_F(HcclCommAssignCcuInsTest, Ut_HcclCommQueryAssignedCcuIns_When_CalledBeforeAssign_Expect_ReturnUnavail)
{
    // 组合：未 Assign 直接 QueryAssigned
    HcclComm comm = BuildComm(true, true, true, 0);
    CcuInsHandle insHandles[1] = {0};
    uint32_t insNum = 0;
    EXPECT_EQ(HcclCommQueryAssignedCcuIns(comm, insHandles, &insNum), HCCL_E_UNAVAIL);
}
