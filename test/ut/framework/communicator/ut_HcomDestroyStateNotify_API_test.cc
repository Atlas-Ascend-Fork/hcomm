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
#include "hcom_common.h"
#include "hcom_common_v2.h"
#include "hcom_private.h"
#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

namespace {
constexpr const char* CALLBACK_NAME = "ut_hcom_destroy_state_notify";
constexpr const char* GROUP_NAME = "ut_hcom_notify_group";
enum class Event { PRE, DESTROY, POST };

struct CommState {
    HcclComm handle = nullptr;
    std::weak_ptr<hccl::hcclComm> owner;
    std::string name;
};

struct CallbackState {
    HcomInfo* context = nullptr;
    std::vector<CommState> comms;
    std::vector<std::pair<HcclComm, Event>> events;
    HcclResult preResult = HCCL_SUCCESS;
    HcclResult postResult = HCCL_SUCCESS;
    HcclResult destroyResult = HCCL_SUCCESS;
    bool innerGroupExists = true;
    std::function<void()> onPre;
};

CallbackState* g_state = nullptr;

HcclResult GetHcomTestDevice(s32* device)
{
    *device = 0;
    return HCCL_SUCCESS;
}

HcclResult EnableHcomV2(bool* supported)
{
    *supported = true;
    return HCCL_SUCCESS;
}

HcclResult IgnoreHcomCallback(HcclComm, HcclCommStatePhase, void*) { return HCCL_SUCCESS; }

HcclResult RecordHcomCallback(HcclComm comm, HcclCommStatePhase phase, void* args)
{
    auto& state = *static_cast<CallbackState*>(args);
    auto iter = std::find_if(state.comms.begin(), state.comms.end(), [comm](const CommState& item) {
        return item.handle == comm;
    });
    EXPECT_NE(iter, state.comms.end());
    if (iter == state.comms.end()) {
        return HCCL_E_PARA;
    }
    bool isPre = phase == HCCL_COMM_STATE_PHASE_DESTROY_PRE;
    EXPECT_FALSE(iter->owner.expired());
    if (!iter->owner.expired()) {
        EXPECT_EQ(static_cast<hccl::hcclComm*>(comm)->GetIdentifier(), iter->name);
    }
    state.events.emplace_back(comm, isPre ? Event::PRE : Event::POST);
    // 跨线程探测管理锁，既校验锁外通知，也避免回归时用例自死锁。
    bool locksAvailable = false;
    std::thread probe([&]() {
        std::unique_lock<std::mutex> groupLock(state.context->groupParamsLock, std::try_to_lock);
        locksAvailable = groupLock.owns_lock();
    });
    probe.join();
    EXPECT_TRUE(locksAvailable);
    if (!isPre) {
        EXPECT_EQ(phase, HCCL_COMM_STATE_PHASE_DESTROY_POST);
        EXPECT_EQ(state.context->hcomGroupMap.count(GROUP_NAME), 0U);
        return state.postResult;
    }
    if (state.onPre) {
        state.onPre();
    }
    return state.preResult;
}

HcclResult RecordGroupDestroy(const std::string& group)
{
    EXPECT_EQ(group, GROUP_NAME);
    g_state->events.emplace_back(nullptr, Event::DESTROY);
    for (const auto& comm : g_state->comms) {
        EXPECT_FALSE(comm.owner.expired());
    }
    if (g_state->destroyResult != HCCL_SUCCESS) {
        return g_state->destroyResult;
    }
    if (!g_state->innerGroupExists) {
        return HCCL_E_PARA;
    }
    g_state->innerGroupExists = false;
    return HCCL_SUCCESS;
}

} // namespace

class HcomDestroyStateNotifyTest : public testing::Test {
protected:
    void SetUp() override
    {
        MOCKER(hrtGetDevice).stubs().will(invoke(GetHcomTestDevice));
        for (u32 i = 0; i <= MAX_MODULE_DEVICE_NUM; ++i) {
            auto& context = HcomGetCtxHomInfoById(i);
            backups_.push_back({&context, std::move(context.pComm), std::move(context.hcomGroupMap)});
            context.hcomGroupMap.clear();
        }
        state_.context = &HcomGetCtxHomInfo();
        g_state = &state_;
        wasAutoTune_ = HcomGetCtxAutoTuneMode();
        HcomGetCtxAutoTuneMode() = false;
        MOCKER(hrtGetHcclV2Support).stubs().will(invoke(EnableHcomV2));
        MOCKER(HcomDestroyGroupImplV2).stubs().will(invoke(RecordGroupDestroy));
        ASSERT_EQ(HcclCommRegCommStateCallback(CALLBACK_NAME, RecordHcomCallback, &state_), HCCL_SUCCESS);
    }

    void TearDown() override
    {
        EXPECT_EQ(HcclCommRegCommStateCallback(CALLBACK_NAME, IgnoreHcomCallback, nullptr), HCCL_SUCCESS);
        for (auto& backup : backups_) {
            backup.context->pComm = std::move(backup.world);
            backup.context->hcomGroupMap = std::move(backup.groups);
        }
        HcomGetCtxAutoTuneMode() = wasAutoTune_;
        g_state = nullptr;
        GlobalMockObject::verify();
    }

    void AddGroup(const std::string& name)
    {
        auto comm = std::make_shared<hccl::hcclComm>(0, 0, name);
        state_.comms.push_back({static_cast<HcclComm>(comm.get()), comm, name});
        HcclGroupParams params{};
        params.pSubComm = comm;
        state_.context->hcomGroupMap.emplace(name, params);
    }

    void ExpectSuccessfulSequence()
    {
        EXPECT_FALSE(state_.innerGroupExists);
        ASSERT_EQ(state_.events.size(), state_.comms.size() * 2 + 1);
        EXPECT_EQ(state_.events[state_.comms.size()], std::make_pair(static_cast<HcclComm>(nullptr), Event::DESTROY));
        for (const auto& comm : state_.comms) {
            EXPECT_TRUE(comm.owner.expired());
            EXPECT_EQ(
                std::count(state_.events.begin(), state_.events.end(), std::make_pair(comm.handle, Event::PRE)), 1);
            EXPECT_EQ(
                std::count(state_.events.begin(), state_.events.end(), std::make_pair(comm.handle, Event::POST)), 1);
        }
        for (size_t i = 0; i < state_.comms.size(); ++i) {
            EXPECT_EQ(state_.events[i].second, Event::PRE);
            EXPECT_EQ(state_.events[i + state_.comms.size() + 1].second, Event::POST);
        }
    }

    CallbackState state_;
    struct ContextBackup {
        HcomInfo* context;
        HcclCommPtr world;
        std::unordered_map<std::string, HcclGroupParams> groups;
    };
    std::vector<ContextBackup> backups_;
    bool wasAutoTune_ = false;
};

TEST_F(HcomDestroyStateNotifyTest, GroupSuccessNotifiesOutsideLocksWithLiveHandle)
{
    AddGroup(GROUP_NAME);
    ASSERT_EQ(HcomDestroyGroup(GROUP_NAME), HCCL_SUCCESS);
    ExpectSuccessfulSequence();
}

TEST_F(HcomDestroyStateNotifyTest, GroupPreFailurePreservesGroupAndAllowsRetry)
{
    AddGroup(GROUP_NAME);
    state_.preResult = HCCL_E_INTERNAL;
    EXPECT_EQ(HcomDestroyGroup(GROUP_NAME), HCCL_E_INTERNAL);
    ASSERT_EQ(state_.events.size(), 1U);
    EXPECT_EQ(state_.events[0].second, Event::PRE);
    EXPECT_EQ(state_.context->hcomGroupMap.count(GROUP_NAME), 1U);
    state_.preResult = HCCL_SUCCESS;
    state_.events.clear();
    ASSERT_EQ(HcomDestroyGroup(GROUP_NAME), HCCL_SUCCESS);
    ExpectSuccessfulSequence();
}

TEST_F(HcomDestroyStateNotifyTest, GroupDestroyFailureHasNoPost)
{
    AddGroup(GROUP_NAME);
    state_.destroyResult = HCCL_E_AGAIN;
    EXPECT_EQ(HcomDestroyGroup(GROUP_NAME), HCCL_E_AGAIN);
    ASSERT_EQ(state_.events.size(), 2U);
    EXPECT_EQ(state_.events.back().second, Event::DESTROY);
    EXPECT_EQ(state_.context->hcomGroupMap.count(GROUP_NAME), 1U);
    EXPECT_FALSE(state_.comms[0].owner.expired());
}

TEST_F(HcomDestroyStateNotifyTest, MissingGroupAttemptsInnerCleanupWithoutNotify)
{
    state_.innerGroupExists = false;
    EXPECT_EQ(HcomDestroyGroup(GROUP_NAME), HCCL_E_PARA);
    const std::vector<std::pair<HcclComm, Event>> expected{{nullptr, Event::DESTROY}};
    EXPECT_EQ(state_.events, expected);
    EXPECT_TRUE(state_.context->hcomGroupMap.empty());
}

TEST_F(HcomDestroyStateNotifyTest, MissingOuterRecordCleansOrphanedInnerGroup)
{
    ASSERT_TRUE(state_.innerGroupExists);
    EXPECT_EQ(HcomDestroyGroup(GROUP_NAME), HCCL_E_PARA);
    EXPECT_FALSE(state_.innerGroupExists);
    EXPECT_TRUE(state_.context->hcomGroupMap.empty());
    const std::vector<std::pair<HcclComm, Event>> expected{{nullptr, Event::DESTROY}};
    EXPECT_EQ(state_.events, expected);
}

TEST_F(HcomDestroyStateNotifyTest, MissingOuterRecordPropagatesCleanupFailureAndAllowsRetry)
{
    state_.destroyResult = HCCL_E_INTERNAL;
    EXPECT_EQ(HcomDestroyGroup(GROUP_NAME), HCCL_E_INTERNAL);
    EXPECT_TRUE(state_.innerGroupExists);
    EXPECT_TRUE(state_.context->hcomGroupMap.empty());
    const std::vector<std::pair<HcclComm, Event>> expected{{nullptr, Event::DESTROY}};
    EXPECT_EQ(state_.events, expected);
    state_.destroyResult = HCCL_SUCCESS;
    state_.events.clear();
    EXPECT_EQ(HcomDestroyGroup(GROUP_NAME), HCCL_E_PARA);
    EXPECT_FALSE(state_.innerGroupExists);
    EXPECT_EQ(state_.events, expected);
}

TEST_F(HcomDestroyStateNotifyTest, GroupRemovedDuringPreDoesNotDestroyAgain)
{
    AddGroup(GROUP_NAME);
    state_.onPre = [this]() {
        std::lock_guard<std::mutex> lock(state_.context->groupParamsLock);
        state_.context->hcomGroupMap.erase(GROUP_NAME);
    };
    EXPECT_EQ(HcomDestroyGroup(GROUP_NAME), HCCL_E_AGAIN);
    const std::vector<std::pair<HcclComm, Event>> expected{{state_.comms[0].handle, Event::PRE}};
    EXPECT_EQ(state_.events, expected);
    EXPECT_TRUE(state_.context->hcomGroupMap.empty());
    EXPECT_TRUE(state_.comms[0].owner.expired());
}

TEST_F(HcomDestroyStateNotifyTest, GroupReplacedDuringPrePreservesReplacement)
{
    AddGroup(GROUP_NAME);
    auto replacement = std::make_shared<hccl::hcclComm>(0, 0, GROUP_NAME);
    state_.onPre = [this, replacement]() {
        std::lock_guard<std::mutex> lock(state_.context->groupParamsLock);
        state_.context->hcomGroupMap.at(GROUP_NAME).pSubComm = replacement;
    };
    EXPECT_EQ(HcomDestroyGroup(GROUP_NAME), HCCL_E_AGAIN);
    const std::vector<std::pair<HcclComm, Event>> expected{{state_.comms[0].handle, Event::PRE}};
    EXPECT_EQ(state_.events, expected);
    EXPECT_EQ(state_.context->hcomGroupMap.at(GROUP_NAME).pSubComm, replacement);
    EXPECT_TRUE(state_.comms[0].owner.expired());
}

TEST_F(HcomDestroyStateNotifyTest, GroupPostFailureDoesNotFailCompletedDestroy)
{
    AddGroup(GROUP_NAME);
    state_.postResult = HCCL_E_INTERNAL;
    EXPECT_EQ(HcomDestroyGroup(GROUP_NAME), HCCL_SUCCESS);
    ExpectSuccessfulSequence();
    EXPECT_TRUE(state_.context->hcomGroupMap.empty());
}

TEST_F(HcomDestroyStateNotifyTest, NullSubCommDestroysAndErasesWithoutNotify)
{
    state_.context->hcomGroupMap.emplace(GROUP_NAME, HcclGroupParams{});
    EXPECT_EQ(HcomDestroyGroup(GROUP_NAME), HCCL_SUCCESS);
    EXPECT_TRUE(state_.context->hcomGroupMap.empty());
    const std::vector<std::pair<HcclComm, Event>> expected{{nullptr, Event::DESTROY}};
    EXPECT_EQ(state_.events, expected);
}

TEST_F(HcomDestroyStateNotifyTest, NullSubCommDestroyFailurePreservesRecord)
{
    state_.context->hcomGroupMap.emplace(GROUP_NAME, HcclGroupParams{});
    state_.destroyResult = HCCL_E_INTERNAL;
    EXPECT_EQ(HcomDestroyGroup(GROUP_NAME), HCCL_E_INTERNAL);
    EXPECT_EQ(state_.context->hcomGroupMap.count(GROUP_NAME), 1U);
    const std::vector<std::pair<HcclComm, Event>> expected{{nullptr, Event::DESTROY}};
    EXPECT_EQ(state_.events, expected);
}
