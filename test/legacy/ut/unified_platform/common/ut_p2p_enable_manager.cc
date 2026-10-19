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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "p2p_enable_manager.h"
#include "orion_adapter_rts.h"
#include "driver/ascend_hal.h"

using namespace Hccl;

class P2PEnableManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "P2PEnableManagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "P2PEnableManagerTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in P2PEnableManagerTest SetUp" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in P2PEnableManager TearDown" << std::endl;
    }
};

TEST_F(P2PEnableManagerTest, should_successfully_add_device_pairs_to_set_after_calling_enablep2p)
{
    // Given
    auto devicePairs = P2PEnableManager::GetInstance().GetSet();

    // then
    // 注：期望值有时可能需要调整
    // 原因：P2PEnableManager是单例，容易受其他UT用例的影响，目前RmaConnManager相关UT会影响
    EXPECT_EQ(devicePairs.size(), 0);
}

TEST_F(P2PEnableManagerTest, enable_p2p_success_and_disable_p2p_success)
{
    MOCKER(HrtEnableP2P).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtDisableP2P).stubs().will(returnValue(HCCL_SUCCESS));

    std::vector<u32> enableP2PDevices_;
    for (u32 i = 1; i < 4; i++) {
        enableP2PDevices_.emplace_back(i);
    }
    HcclResult ret = P2PEnableManager::GetInstance().EnableP2P(enableP2PDevices_);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = P2PEnableManager::GetInstance().DisableP2P(1U, enableP2PDevices_);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(P2PEnableManagerTest, disable_p2p_success_when_not_enable)
{
    std::vector<u32> enableP2PDevices_;
    for (u32 i = 1; i < 4; i++) {
        enableP2PDevices_.emplace_back(i);
    }
    HcclResult ret = P2PEnableManager::GetInstance().DisableP2P(1U, enableP2PDevices_);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(P2PEnableManagerTest, enable_p2p_failed_when_hrt_enable_p2p_failed)
{
    constexpr u32 localDeviceLogicID = 5U;
    constexpr u32 remoteDevicePhysicID = 50U;
    MOCKER(HrtGetDevice).stubs().will(returnValue(static_cast<s32>(localDeviceLogicID)));
    MOCKER(HrtEnableP2P).stubs().will(returnValue(HCCL_E_RUNTIME));

    // 驱动下发失败：EnableP2P 返回失败，且未持有引用计数
    std::vector<u32> remoteDevices{remoteDevicePhysicID};
    HcclResult ret = P2PEnableManager::GetInstance().EnableP2P(remoteDevices);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);

    // enable 未成功执行时 wait 直接报错，isEnabled 保持 false
    bool isEnabled = true;
    ret = P2PEnableManager::GetInstance().WaitP2PEnabled(localDeviceLogicID, remoteDevicePhysicID, isEnabled);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_FALSE(isEnabled);
}

TEST_F(P2PEnableManagerTest, wait_p2p_enabled_success_when_driver_not_enabled)
{
    constexpr u32 localDeviceLogicID = 5U;
    constexpr u32 remoteDevicePhysicID = 50U;
    MOCKER(HrtGetDevice).stubs().will(returnValue(static_cast<s32>(localDeviceLogicID)));
    MOCKER(HrtEnableP2P).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtDisableP2P).stubs().will(returnValue(HCCL_SUCCESS));
    uint32_t drvStatus = DRV_P2P_STATUS_DISABLE;
    MOCKER(HrtGetP2PStatus)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&drvStatus, sizeof(drvStatus)))
        .will(returnValue(HCCL_SUCCESS));

    std::vector<u32> remoteDevices{remoteDevicePhysicID};
    EXPECT_EQ(P2PEnableManager::GetInstance().EnableP2P(remoteDevices), HCCL_SUCCESS);

    // 驱动侧尚未使能：单次查询返回 SUCCESS 且 isEnabled=false，由调用方下次轮询
    bool isEnabled = true;
    HcclResult ret
        = P2PEnableManager::GetInstance().WaitP2PEnabled(localDeviceLogicID, remoteDevicePhysicID, isEnabled);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(isEnabled);

    // 清理单例状态，保证用例间独立
    EXPECT_EQ(P2PEnableManager::GetInstance().DisableP2P(localDeviceLogicID, remoteDevices), HCCL_SUCCESS);
}

TEST_F(P2PEnableManagerTest, wait_p2p_enabled_success_when_driver_enabled_and_cache_hit)
{
    constexpr u32 localDeviceLogicID = 5U;
    constexpr u32 remoteDevicePhysicID = 50U;
    MOCKER(HrtGetDevice).stubs().will(returnValue(static_cast<s32>(localDeviceLogicID)));
    MOCKER(HrtEnableP2P).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtDisableP2P).stubs().will(returnValue(HCCL_SUCCESS));
    uint32_t drvStatus = DRV_P2P_STATUS_ENABLE;
    // 限定驱动查询恰好一次：首次查询确认使能并写入缓存后，再次 wait 应命中缓存不再查驱动
    MOCKER(HrtGetP2PStatus)
        .expects(once())
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&drvStatus, sizeof(drvStatus)))
        .will(returnValue(HCCL_SUCCESS));

    std::vector<u32> remoteDevices{remoteDevicePhysicID};
    EXPECT_EQ(P2PEnableManager::GetInstance().EnableP2P(remoteDevices), HCCL_SUCCESS);

    // 首次查询：驱动侧已使能，返回 SUCCESS 且 isEnabled=true
    bool isEnabled = false;
    HcclResult ret
        = P2PEnableManager::GetInstance().WaitP2PEnabled(localDeviceLogicID, remoteDevicePhysicID, isEnabled);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(isEnabled);

    // 第二次调用命中缓存直接返回（HrtGetP2PStatus 恰好调用一次由 TearDown 的 verify 校验）
    isEnabled = false;
    ret = P2PEnableManager::GetInstance().WaitP2PEnabled(localDeviceLogicID, remoteDevicePhysicID, isEnabled);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(isEnabled);

    EXPECT_EQ(P2PEnableManager::GetInstance().DisableP2P(localDeviceLogicID, remoteDevices), HCCL_SUCCESS);
}

TEST_F(P2PEnableManagerTest, wait_p2p_enabled_failed_when_hrt_get_p2p_status_failed)
{
    constexpr u32 localDeviceLogicID = 5U;
    constexpr u32 remoteDevicePhysicID = 50U;
    MOCKER(HrtGetDevice).stubs().will(returnValue(static_cast<s32>(localDeviceLogicID)));
    MOCKER(HrtEnableP2P).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtDisableP2P).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtGetP2PStatus)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_RUNTIME));

    std::vector<u32> remoteDevices{remoteDevicePhysicID};
    EXPECT_EQ(P2PEnableManager::GetInstance().EnableP2P(remoteDevices), HCCL_SUCCESS);

    // 驱动状态查询失败：返回错误码且 isEnabled=false
    bool isEnabled = true;
    HcclResult ret
        = P2PEnableManager::GetInstance().WaitP2PEnabled(localDeviceLogicID, remoteDevicePhysicID, isEnabled);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
    EXPECT_FALSE(isEnabled);

    EXPECT_EQ(P2PEnableManager::GetInstance().DisableP2P(localDeviceLogicID, remoteDevices), HCCL_SUCCESS);
}
