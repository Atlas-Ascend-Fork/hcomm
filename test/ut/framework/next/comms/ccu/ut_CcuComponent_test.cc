/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"

#include <iostream>

#define private public
#define protected public

#include "ccu_comp.h"

#undef protected
#undef private

#include "mocks/ccu_device_mock_utils.h"

class CcuComponentTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        // 将enableEntryLog默认返回为true
        MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
    }
    void TearDown() override
    {
        // MockCcuResourcesDefault 会 Init 单例；在 mock 仍有效时 Deinit，避免进程退出时静态析构触发 bad_alloc
        const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
        (void)hcomm::CcuComponent::GetInstance(devLogicId).Deinit();
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

protected:
};

TEST_F(CcuComponentTest, Ut_CcuComponent_Init_When_Mock_Is_Fine_Expect_Return_Ok)
{
    constexpr int32_t fakeDevLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MOCKER(hrtGetDevicePhyIdByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDevLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MockCcuNetworkDeviceDefault(fakeDevLogicId); // 先处理网络设备，再初始化ccu
    EXPECT_EQ(MockCcuResourcesDefault(fakeDevLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);

    hcomm::CcuComponent ccuComponent{};
    ccuComponent.userDevId_ = fakeDevLogicId;

    EXPECT_EQ(ccuComponent.Init(), HcclResult::HCCL_SUCCESS);
}

// Init 需清空上次运行残留的 totalCntXn 占用标记, 否则该 die 的级联计数器能力在重新 Init 后仍不可用
TEST_F(CcuComponentTest, Ut_CcuComponent_Init_When_StaleTotalCntXnFlags_Expect_Cleared)
{
    constexpr int32_t fakeDevLogicId = MAX_MODULE_DEVICE_NUM - 1;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MOCKER(hrtGetDevicePhyIdByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDevLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MockCcuNetworkDeviceDefault(fakeDevLogicId);
    EXPECT_EQ(MockCcuResourcesDefault(fakeDevLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);

    hcomm::CcuComponent ccuComponent{};
    ccuComponent.userDevId_ = fakeDevLogicId;
    for (uint8_t dieId = 0; dieId < hcomm::CCU_MAX_IODIE_NUM; dieId++) {
        for (uint32_t index = 0; index < hcomm::CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM; index++) {
            ccuComponent.usedTotalCntXnFlags_[dieId][index] = true;
        }
    }

    EXPECT_EQ(ccuComponent.Init(), HcclResult::HCCL_SUCCESS);
    for (uint8_t dieId = 0; dieId < hcomm::CCU_MAX_IODIE_NUM; dieId++) {
        for (uint32_t index = 0; index < hcomm::CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM; index++) {
            EXPECT_FALSE(ccuComponent.usedTotalCntXnFlags_[dieId][index]);
        }
    }
}

TEST_F(CcuComponentTest, Ut_CcuComponent_CleanDieCkes_InvalidDieId)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    const hcomm::CcuVersion ccuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(devLogicId);
    MockCcuResourcesDefault(devLogicId, ccuVersion);

    hcomm::CcuComponent ccuComponent{};
    ccuComponent.userDevId_ = devLogicId;

    // 使用非法的 dieId（等于上限），期望参数错误返回
    const uint8_t badDieId = static_cast<uint8_t>(hcomm::CCU_MAX_IODIE_NUM);
    EXPECT_EQ(ccuComponent.CleanDieCkes(badDieId), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuComponentTest, Ut_CcuComponent_CleanDieCkes_DieDisabled_NoOp)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    const hcomm::CcuVersion ccuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(devLogicId);
    MockCcuResourcesDefault(devLogicId, ccuVersion);

    hcomm::CcuComponent ccuComponent{};
    ccuComponent.userDevId_ = devLogicId;

    // 确保 die 被标记为不可用，CleanDieCkes 应直接返回成功且不调用外部驱动
    ccuComponent.dieEnableFlags_.fill(false);
    EXPECT_EQ(ccuComponent.CleanDieCkes(0), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuComponentTest, Ut_CcuComponent_SetTaskKill_Transitions)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    const hcomm::CcuVersion ccuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(devLogicId);
    MockCcuResourcesDefault(devLogicId, ccuVersion);

    hcomm::CcuComponent ccuComponent{};
    ccuComponent.userDevId_ = devLogicId;

    // 保证所有 die 为不可用，以避免 SetProcess 内部调用真实驱动函数
    ccuComponent.dieEnableFlags_.fill(false);

    // 初始态置为 INVALID，首次调用应将状态置为 TASK_KILL
    ccuComponent.status = hcomm::CcuComponent::CcuTaskKillStatus::INVALID;
    EXPECT_EQ(ccuComponent.SetTaskKill(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ccuComponent.status, hcomm::CcuComponent::CcuTaskKillStatus::TASK_KILL);

    // 再次调用，因已处于 TASK_KILL，应直接返回成功且不改变状态
    EXPECT_EQ(ccuComponent.SetTaskKill(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ccuComponent.status, hcomm::CcuComponent::CcuTaskKillStatus::TASK_KILL);

    // 调用 SetTaskKillDone，应将状态恢复为 INIT
    EXPECT_EQ(ccuComponent.SetTaskKillDone(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ccuComponent.status, hcomm::CcuComponent::CcuTaskKillStatus::INIT);

    // 调用 CleanTaskKillState（const 方法），仅验证返回值
    EXPECT_EQ(ccuComponent.CleanTaskKillState(), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuComponentTest, Ut_CcuComponent_GetLoopTpAttr_Cached_ReturnsCachedValue)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    hcomm::CcuComponent ccuComponent{};
    ccuComponent.userDevId_ = devLogicId;

    hcomm::TpAttrInfo cachedTpAttrInfo{};
    cachedTpAttrInfo.tpAttr.at = 2;
    cachedTpAttrInfo.tpAttr.retryTimesInit = 1;
    ccuComponent.tpAttrInfoMap_[0] = cachedTpAttrInfo;

    CommAddr commAddr{};
    hcomm::TpAttrInfo result{};
    EXPECT_EQ(ccuComponent.GetLoopTpAttr(0, commAddr, result), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(result.tpAttr.at, 2);
    EXPECT_EQ(result.tpAttr.retryTimesInit, 1);

    GlobalMockObject::verify();
}

TEST_F(CcuComponentTest, Ut_CcuComponent_GetLoopTpAttr_TpInfoNotFound_ReturnsError)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    hcomm::CcuComponent ccuComponent{};
    ccuComponent.userDevId_ = devLogicId;
    ccuComponent.devPhyId_ = 0;

    CommAddr commAddr{};
    hcomm::TpAttrInfo result{};
    EXPECT_EQ(ccuComponent.GetLoopTpAttr(0, commAddr, result), HcclResult::HCCL_E_NOT_FOUND);

    GlobalMockObject::verify();
}

// cnt xn 块布局不变式：wishCntXn 连续段之后紧跟 1 个 totalCntXn 与 1 个 expectedCntXn。
// 布局错位会把相邻块的寄存器配成本块的计数器，必须在触及硬件前拒绝
TEST_F(CcuComponentTest, Ut_CcuComponent_ConfirmCntXns_When_LayoutInvalid_Expect_PARA)
{
    constexpr int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    hcomm::CcuComponent ccuComponent{};
    ccuComponent.userDevId_ = devLogicId;
    ccuComponent.ccuVersion_ = hcomm::CcuVersion::CCU_V2;

    uint32_t blockIdx = 0;
    // totalCntXn 未紧跟 wishCntXn 段末尾
    EXPECT_EQ(ccuComponent.ConfirmCntXns(0, {0, 1021}, 1030, 1031, blockIdx), HcclResult::HCCL_E_PARA);
    EXPECT_EQ(blockIdx, hcomm::CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM);
    // expectedCntXn 未紧跟 totalCntXn
    EXPECT_EQ(ccuComponent.ConfirmCntXns(0, {0, 1021}, 1022, 1030, blockIdx), HcclResult::HCCL_E_PARA);
    // wishCntXn 区间反向
    EXPECT_EQ(ccuComponent.ConfirmCntXns(0, {1021, 0}, 1022, 1023, blockIdx), HcclResult::HCCL_E_PARA);

    GlobalMockObject::verify();
}
