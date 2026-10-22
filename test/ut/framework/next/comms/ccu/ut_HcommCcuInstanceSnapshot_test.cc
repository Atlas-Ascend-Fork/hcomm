/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#include <array>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#define private public
#include "ccu_instance_mgr.h"
#undef private
#include "ccu_dev_mgr_imp.h"

using namespace hcomm;

namespace {

CcuResult StubDieEnableFirstDieOnly(int32_t, uint8_t dieId, bool& enableFlag)
{
    enableFlag = dieId == 0;
    return dieId == 1 ? CcuResult::CCU_E_INTERNAL : CcuResult::CCU_SUCCESS;
}

} // namespace

class HcommCcuInstanceSnapshotTest : public testing::Test {
protected:
    void SetUp() override
    {
        GlobalMockObject::reset();
        MOCKER_CPP(&CcuInstance::InitByInsType).stubs().will(returnValue(CCU_SUCCESS));
    }

    void TearDown() override
    {
        for (const auto& instance : instances_) {
            EXPECT_EQ(CcuInstanceMgr::GetInstance(instance.first).Destroy(instance.second), CCU_SUCCESS);
        }
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }

    HcommCcuInstanceEntity Create(int32_t deviceId)
    {
        CcuInsHandle key = 0;
        auto& mgr = CcuInstanceMgr::GetInstance(deviceId);
        EXPECT_EQ(mgr.CreateByInsType(CcuInstanceType::CCU_SCHED, key), CCU_SUCCESS);
        instances_.emplace_back(deviceId, key);
        mgr.Get(key)->userDevId_ = deviceId;
        HcommCcuInstanceEntity handle{};
        HcommCcuInstanceEntityInit(&handle);
        handle.ccuInsKey = key;
        return handle;
    }

    // die-enable 查询单独 mock：mockcpp 按注册顺序首个匹配生效，若混在 MockSnapshotMetadata 中，
    // 用例自定义的 die-enable stub 须依赖注册顺序才能覆盖同名 stub；拆出后由各用例显式控制，语义清晰。
    void MockDieEnableAll()
    {
        const bool enabled = true;
        MOCKER(CcuGetDieEnableInfo)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), outBound(enabled))
            .will(returnValue(CCU_SUCCESS));
    }

    void MockSnapshotMetadata()
    {
        const CcuVersion version = CcuVersion::CCU_V1;
        MOCKER(CcuDevMgrImp::GetCcuVersion)
            .expects(once())
            .with(mockcpp::any(), outBound(version))
            .will(returnValue(HCCL_SUCCESS));
        MOCKER(CcuDevMgrImp::GetMissionKey).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(CcuDevMgrImp::GetXnBaseAddr).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(CcuDevMgrImp::GetLoopChannelId).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(CcuDevMgrImp::GetCcuResourceSpaceBufInfo).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(CcuDevMgrImp::GetCcuResourceSpaceTokenInfo).stubs().will(returnValue(HCCL_SUCCESS));
    }

    std::vector<std::pair<int32_t, CcuInsHandle>> instances_;
};

TEST_F(HcommCcuInstanceSnapshotTest, CreateLeavesSnapshotUninitializedAndQueryReusesSnapshot)
{
    MockDieEnableAll();
    MockSnapshotMetadata();
    auto handle = Create(0);
    auto* instance = CcuInstanceMgr::GetInstance(0).Get(handle.ccuInsKey);
    ASSERT_NE(instance, nullptr);
    EXPECT_EQ(instance->GetHcommCcuInstanceEntity(), nullptr);
    instance->resPack_ = std::make_unique<CcuResPack>();
    instance->resPack_->GetCcuResRepo().xn[0].push_back({64, 8});

    ASSERT_EQ(CcuInstanceMgr::GetInstance(0).QueryInstanceEntity(handle), CCU_SUCCESS);
    ASSERT_NE(handle.ccuDieReses[0].resTypes[HCOMM_CCU_BATCH_RES_XN].resRangeNum, 0U);
    const auto& xn = handle.ccuDieReses[0].resTypes[HCOMM_CCU_BATCH_RES_XN];
    EXPECT_EQ(xn.resRangeNum, 1U);
    EXPECT_EQ(xn.resRanges[0].startId, 64U);
    EXPECT_EQ(xn.resRanges[0].count, 8U);
    handle.ccuDieReses[0].resTypes[HCOMM_CCU_BATCH_RES_XN].resRangeNum = 0;
    ASSERT_EQ(CcuInstanceMgr::GetInstance(0).QueryInstanceEntity(handle), CCU_SUCCESS);
    EXPECT_EQ(handle.ccuDieReses[0].resTypes[HCOMM_CCU_BATCH_RES_XN].resRangeNum, 1U);
    EXPECT_EQ(handle.ccuDieReses[0].resTypes[HCOMM_CCU_BATCH_RES_XN].resRanges[0].startId, 64U);
}

TEST_F(HcommCcuInstanceSnapshotTest, FailedInitializationCanBeRetried)
{
    auto handle = Create(0);
    auto* instance = CcuInstanceMgr::GetInstance(0).Get(handle.ccuInsKey);
    instance->resPack_ = std::make_unique<CcuResPack>();
    instance->resPack_->GetCcuResRepo().xn[0].push_back({64, 8});

    // 本轮仅注册自定义 die-enable stub（die0 启用、die1 返回 CCU_E_INTERNAL），模拟首轮构建失败
    MOCKER(CcuGetDieEnableInfo).stubs().will(invoke(StubDieEnableFirstDieOnly));
    MockSnapshotMetadata();
    EXPECT_EQ(CcuInstanceMgr::GetInstance(0).QueryInstanceEntity(handle), CCU_E_INTERNAL);
    EXPECT_EQ(handle.ccuDieReses[0].resTypes[HCOMM_CCU_BATCH_RES_XN].resRangeNum, 0U);
    EXPECT_EQ(instance->snapshotRanges_[0][HCOMM_CCU_BATCH_RES_XN].size(), 1U);
    EXPECT_FALSE(instance->snapshotReady_);

    GlobalMockObject::verify();
    GlobalMockObject::reset();
    instance->resPack_->GetCcuResRepo().xn[0].clear();
    instance->resPack_->GetCcuResRepo().xn[0].push_back({64, 8});
    MockDieEnableAll();
    MockSnapshotMetadata();
    EXPECT_EQ(CcuInstanceMgr::GetInstance(0).QueryInstanceEntity(handle), CCU_SUCCESS);
    EXPECT_EQ(handle.ccuDieReses[0].resTypes[HCOMM_CCU_BATCH_RES_XN].resRangeNum, 1U);
    EXPECT_EQ(handle.ccuDieReses[0].resTypes[HCOMM_CCU_BATCH_RES_XN].resRanges[0].startId, 64U);
    EXPECT_EQ(handle.ccuDieReses[0].resTypes[HCOMM_CCU_BATCH_RES_XN].resRanges[0].count, 8U);
}

TEST_F(HcommCcuInstanceSnapshotTest, RejectsInvalidArguments)
{
    HcommCcuInstanceEntity handle{};
    HcommCcuInstanceEntityInit(&handle);
    handle.ccuInsKey = 1;
    auto* nonExistInstance = CcuInstanceMgr::GetInstance(0).Get(handle.ccuInsKey);
    EXPECT_EQ(nonExistInstance, nullptr);
    EXPECT_EQ(CcuInstanceMgr::GetInstance(0).QueryInstanceEntity(handle), CCU_E_NOT_FOUND);
    handle.header.version = 0;
    EXPECT_EQ(CcuInstanceMgr::GetInstance(0).QueryInstanceEntity(handle), CCU_E_PARA);
    HcommCcuInstanceEntityInit(&handle);
    handle.header.magicWord = 0;
    EXPECT_EQ(CcuInstanceMgr::GetInstance(0).QueryInstanceEntity(handle), CCU_E_PARA);
    HcommCcuInstanceEntityInit(&handle);
    handle.header.size = 0;
    EXPECT_EQ(CcuInstanceMgr::GetInstance(0).QueryInstanceEntity(handle), CCU_E_PARA);
    HcommCcuInstanceEntityInit(&handle);
    handle.header.reserved = 1;
    EXPECT_EQ(CcuInstanceMgr::GetInstance(0).QueryInstanceEntity(handle), CCU_E_PARA);
    HcommCcuInstanceEntityInit(&handle);
    handle.ccuInsKey = 0;
    EXPECT_EQ(CcuInstanceMgr::GetInstance(0).QueryInstanceEntity(handle), CCU_E_PARA);
}

TEST_F(HcommCcuInstanceSnapshotTest, RejectsDestroyedKeyAndClearsBorrowedPointer)
{
    auto handle = Create(0);
    ASSERT_EQ(CcuInstanceMgr::GetInstance(0).Destroy(handle.ccuInsKey), CCU_SUCCESS);
    instances_.clear();
    EXPECT_EQ(CcuInstanceMgr::GetInstance(0).QueryInstanceEntity(handle), CCU_E_NOT_FOUND);
    EXPECT_EQ(handle.ccuDieReses[0].resTypes[HCOMM_CCU_BATCH_RES_XN].resRangeNum, 0U);
}

TEST_F(HcommCcuInstanceSnapshotTest, LooksUpKeyOnRequestedDevice)
{
    auto handle = Create(0);
    EXPECT_EQ(CcuInstanceMgr::GetInstance(1).QueryInstanceEntity(handle), CCU_E_NOT_FOUND);
    EXPECT_EQ(handle.ccuDieReses[0].resTypes[HCOMM_CCU_BATCH_RES_XN].resRangeNum, 0U);
}
