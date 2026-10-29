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
#include "hccl_group.h"
#include "hcom_common.h"
#include <thread>
#include <vector>

namespace {
enum class DestroyEvent { PRE, DESTROY, POST };

struct DestroyCallbackState {
    HcclComm comm = nullptr;
    HcclComm commV2 = nullptr;
    HcclOpInfoCtx* opInfo = nullptr;
    std::string group;
    std::vector<DestroyEvent> events;
    HcclResult preResult = HCCL_SUCCESS;
    HcclResult destroyResult = HCCL_SUCCESS;
    bool postLockAvailable = false;
    bool postGroupRemoved = false;
};

DestroyCallbackState* g_destroyState = nullptr;
constexpr const char* CALLBACK_NAME = "ut_group_destroy_state_notify";

HcclResult IgnoreDestroyCallback(HcclComm, HcclCommStatePhase, void*) { return HCCL_SUCCESS; }

HcclResult EnableV2Support(bool* isSupport)
{
    if (isSupport == nullptr) {
        return HCCL_E_PTR;
    }
    *isSupport = true;
    return HCCL_SUCCESS;
}

HcclResult RecordDestroyCallback(HcclComm comm, HcclCommStatePhase phase, void* args)
{
    auto* state = static_cast<DestroyCallbackState*>(args);
    EXPECT_EQ(comm, state->comm);
    if (phase == HcclCommStatePhase::HCCL_COMM_STATE_PHASE_DESTROY_PRE) {
        state->events.push_back(DestroyEvent::PRE);
        return state->preResult;
    }
    EXPECT_EQ(phase, HcclCommStatePhase::HCCL_COMM_STATE_PHASE_DESTROY_POST);
    state->events.push_back(DestroyEvent::POST);
    // 从另一线程探测同一把锁，避免回归到持锁通知时让整个测试自死锁。
    std::thread probe([state]() {
        std::unique_lock<std::mutex> lock(state->opInfo->opGroupMapMutex, std::try_to_lock);
        state->postLockAvailable = lock.owns_lock();
        if (lock.owns_lock()) {
            state->postGroupRemoved = state->opInfo->opGroup2CommMap.count(state->group) == 0;
        }
    });
    probe.join();
    return HCCL_SUCCESS;
}

HcclResult RecordDestroyV2(HcclComm comm)
{
    EXPECT_EQ(comm, g_destroyState->commV2);
    g_destroyState->events.push_back(DestroyEvent::DESTROY);
    return g_destroyState->destroyResult;
}
} // namespace

class HcclCommDestroyGroupTest : public testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_EQ(hcclGroupDepth, 0);
        MOCKER(hrtSetDevice).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(HcclDeviceRefresh).stubs().with(outBound(0)).will(returnValue(HCCL_SUCCESS));
        // 仅返回成功不会打开 V2，必须同时设置输出参数。
        MOCKER(hrtGetHcclV2Support).stubs().will(invoke(EnableV2Support));
        MOCKER(HcclGetDeviceId).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(0));
        MOCKER_CPP(&hccl::hcclComm::GetCommunicatorV2).stubs().will(returnValue(static_cast<HcclComm>(&commV2Token_)));
        MOCKER(HcclCommDestroyV2).stubs().will(invoke(RecordDestroyV2));

        state_.group = "ut_group_destroy_state_notify";
        owner_ = std::make_shared<hccl::hcclComm>(0, 0, state_.group);
        state_.comm = owner_.get();
        state_.commV2 = &commV2Token_;
        auto& manager = hccl::CollCommMgr::GetInstance();
        state_.opInfo = &manager.opHcomInfos_[0];
        // 本用例只使用管理表，跳过设备资源初始化，并在 TearDown 恢复原状态。
        wasBaseCommInited_ = manager.baseCommInited_[0];
        wasUsed_ = state_.opInfo->isUsed;
        manager.baseCommInited_[0] = true;
        state_.opInfo->isUsed = true;
        {
            std::lock_guard<std::mutex> lock(state_.opInfo->opGroupMapMutex);
            ASSERT_EQ(state_.opInfo->opGroup2CommMap.count(state_.group), 0U);
            state_.opInfo->opGroup2CommMap.emplace(state_.group, owner_);
        }
        g_destroyState = &state_;
        ASSERT_EQ(HcclCommRegCommStateCallback(CALLBACK_NAME, RecordDestroyCallback, &state_), HCCL_SUCCESS);
    }

    void TearDown() override
    {
        // 即使用例提前断言失败，也先收齐已入队任务，再使回调上下文失效。
        while (hcclGroupDepth > 0) {
            (void)HcclGroupEnd();
        }
        // 注册表是进程级且没有注销接口，用无上下文回调覆盖，避免污染后续用例。
        EXPECT_EQ(HcclCommRegCommStateCallback(CALLBACK_NAME, IgnoreDestroyCallback, nullptr), HCCL_SUCCESS);
        if (state_.opInfo != nullptr) {
            std::lock_guard<std::mutex> lock(state_.opInfo->opGroupMapMutex);
            state_.opInfo->opGroup2CommMap.erase(state_.group);
            state_.opInfo->isUsed = wasUsed_;
            hccl::CollCommMgr::GetInstance().baseCommInited_[0] = wasBaseCommInited_;
        }
        owner_.reset();
        g_destroyState = nullptr;
        GlobalMockObject::verify();
    }

    void EnqueueDestroy()
    {
        ASSERT_EQ(HcclGroupStart(), HCCL_SUCCESS);
        ASSERT_EQ(HcclCommDestroy(state_.comm), HCCL_SUCCESS);
        EXPECT_TRUE(state_.events.empty());
    }

    DestroyCallbackState state_;
    std::shared_ptr<hccl::hcclComm> owner_;
    char commV2Token_ = 0;
    bool wasBaseCommInited_ = false;
    bool wasUsed_ = false;
};

TEST_F(HcclCommDestroyGroupTest, Ut_GroupDestroy_WhenSuccess_ExpectPreDestroyPostAndUnlockedPost)
{
    EnqueueDestroy();
    ASSERT_EQ(HcclGroupEnd(), HCCL_SUCCESS);
    const std::vector<DestroyEvent> expected{DestroyEvent::PRE, DestroyEvent::DESTROY, DestroyEvent::POST};
    EXPECT_EQ(state_.events, expected);
    EXPECT_TRUE(state_.postLockAvailable);
    EXPECT_TRUE(state_.postGroupRemoved);
}

TEST_F(HcclCommDestroyGroupTest, Ut_GroupDestroy_WhenDestroyFails_ExpectPreWithoutPost)
{
    state_.destroyResult = HCCL_E_AGAIN;
    EnqueueDestroy();
    EXPECT_EQ(HcclGroupEnd(), HCCL_E_AGAIN);
    const std::vector<DestroyEvent> expected{DestroyEvent::PRE, DestroyEvent::DESTROY};
    EXPECT_EQ(state_.events, expected);
    EXPECT_EQ(state_.opInfo->opGroup2CommMap.count(state_.group), 1U);
}

TEST_F(HcclCommDestroyGroupTest, Ut_GroupDestroy_WhenPreFails_ExpectNoDestroyOrPost)
{
    state_.preResult = HCCL_E_INTERNAL;
    EnqueueDestroy();
    EXPECT_EQ(HcclGroupEnd(), HCCL_E_INTERNAL);
    const std::vector<DestroyEvent> expected{DestroyEvent::PRE};
    EXPECT_EQ(state_.events, expected);
    EXPECT_EQ(state_.opInfo->opGroup2CommMap.count(state_.group), 1U);
}
