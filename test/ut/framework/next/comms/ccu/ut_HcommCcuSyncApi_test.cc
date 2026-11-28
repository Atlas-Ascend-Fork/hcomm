/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 同步类 C 接口(Event / LocalNotify)的端到端用例:
// 在注册流程中逐个调用 C 接口, 校验返回值及每次调用新追加的 rep 类型。
// 依赖 ChannelHandle 的远端同步接口由 Ut_CcuSyncApi_When_RemoteOps_* 用例建链后覆盖正常路径。
#include "ut_HcommCcuControlApi_common.h"

#include "ccu_log.h"
#include "ccu_primitives_impl.h"
#include "ccu_rep_loc_record_event.h"
#include "ccu_rep_loc_wait_event.h"
#include "ccu_rep_loc_wait_notify.h"
#include "ccu_rep_record_shared_notify.h"

namespace {
namespace CcuRep = hcomm::CcuRep;
using hcomm::CcuRep::CcuRepType;

constexpr uint16_t SYNC_MASK = 1;
constexpr uint16_t NOTIFY_MASK = 2;

// 核对 rep 上的 event Id、mask 以及 tag 对应的 notify Id。
CcuResult SyncLocalOpsKernel(CcuKernelArg arg)
{
    (void)arg;
    CcuVariableHandle var{};
    CcuEventHandle event{};
    EXPECT_EQ(CcuVariableAlloc(&var), OK);
    EXPECT_EQ(CcuEventAlloc(&event), OK);
    CcuRepCursor reps;
    auto* kernel = reps.Kernel();
    const uint16_t eventId = EventIdOf(kernel, event);
    constexpr char notifyTag[] = "ut_sync_tag";

    EXPECT_EQ(CcuEventRecord(event, SYNC_MASK), OK);
    if (auto record = TakeRep<CcuRep::CcuRepLocRecordEvent>(reps, CcuRepType::LOC_RECORD_EVENT)) {
        EXPECT_EQ(record->GetEventId(), eventId);
        EXPECT_EQ(record->GetMask(), SYNC_MASK);
    }

    EXPECT_EQ(CcuEventWait(event, NOTIFY_MASK), OK);
    if (auto wait = TakeRep<CcuRep::CcuRepLocWaitEvent>(reps, CcuRepType::LOC_WAIT_EVENT)) {
        EXPECT_EQ(wait->GetEventId(), eventId);
        EXPECT_EQ(wait->GetMask(), NOTIFY_MASK);
    }

    EXPECT_EQ(CcuLocalNotifyRecord(notifyTag, SYNC_MASK), OK);
    if (auto notifyRecord = TakeRep<CcuRep::CcuRepRecordSharedNotify>(reps, CcuRepType::RECORD_SHARED_NOTIFY)) {
        const auto& imported = kernel->GetImportedRes().sharedNotifies;
        const auto it = imported.find(notifyTag);
        EXPECT_NE(it, imported.end());
        if (it != imported.end()) {
            EXPECT_EQ(notifyRecord->GetNotifyId(), it->second.Id());
        }
        EXPECT_EQ(notifyRecord->GetMask(), SYNC_MASK);
    }

    EXPECT_EQ(CcuLocalNotifyWait(notifyTag, NOTIFY_MASK), OK);
    if (auto notifyWait = TakeRep<CcuRep::CcuRepLocWaitNotify>(reps, CcuRepType::LOC_WAIT_NOTIFY)) {
        const auto& exported = kernel->GetExportedRes().sharedNotifies;
        const auto it = exported.find(notifyTag);
        EXPECT_NE(it, exported.end());
        if (it != exported.end()) {
            EXPECT_EQ(notifyWait->GetNotifyId(), it->second.Id());
        }
        EXPECT_EQ(notifyWait->GetMask(), NOTIFY_MASK);
    }
    return OK;
}

CcuResult SyncInvalidHandleKernel(CcuKernelArg arg)
{
    (void)arg;
    CcuVariableHandle var{};
    CcuEventHandle event{};
    EXPECT_EQ(CcuVariableAlloc(&var), OK);
    EXPECT_EQ(CcuEventAlloc(&event), OK);
    CcuRepCursor reps;

    EXPECT_EQ(CcuEventRecord(INVALID_HANDLE, SYNC_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuEventWait(INVALID_HANDLE, SYNC_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuWriteVariableWithNotify(INVALID_HANDLE, INVALID_HANDLE, 0, 0, SYNC_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuCascCntClear(INVALID_HANDLE, INVALID_HANDLE, SYNC_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuNotifyWait(INVALID_HANDLE, 0, SYNC_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_TRUE(reps.Kernel()->GetChannels().empty());
    return OK;
}

// 远端同步接口的正常路径: 通道由用例建链后经 kernelArg 传入。
// notifyIdx 与 mask 都是整型, 实参写串编译期发现不了, 故核对 rep 上的通道、信号下标与 mask。
CcuResult SyncRemoteOpsKernel(CcuKernelArg arg)
{
    const auto channel = static_cast<CcuChannelKernelArg*>(arg)->channel;
    constexpr uint32_t remoteNotifyIdx = 3;
    constexpr uint32_t localNotifyIdx = 2;
    constexpr uint32_t remoteVarIdx = 0; // 对端 xn 下标, 取 0 以适配建链 mock 分配到的资源数
    CcuVariableHandle var{};
    EXPECT_EQ(CcuVariableAlloc(&var), OK);

    uint32_t channelId = UINT32_MAX;
    EXPECT_EQ(CcuChannelIdGet(channel, &channelId), OK);
    EXPECT_NE(channelId, UINT32_MAX);

    CcuRepCursor reps;
    const uint16_t varId = VariableIdOf(reps.Kernel(), var);
    EXPECT_EQ(CcuNotifyRecord(channel, remoteNotifyIdx, SYNC_MASK), OK);
    if (auto post = TakeRep<CcuRep::CcuRepRemPostSem>(reps, CcuRepType::REM_POST_SEM)) {
        EXPECT_EQ(post->GetChannel(), channel);
        EXPECT_EQ(post->GetSemIndex(), remoteNotifyIdx);
        EXPECT_EQ(post->GetMask(), SYNC_MASK);
    }

    EXPECT_EQ(CcuNotifyWait(channel, localNotifyIdx, SYNC_MASK), OK);
    if (auto wait = TakeRep<CcuRep::CcuRepRemWaitSem>(reps, CcuRepType::REM_WAIT_SEM)) {
        EXPECT_EQ(wait->GetChannel(), channel);
        EXPECT_EQ(wait->GetSemIndex(), localNotifyIdx);
        EXPECT_EQ(wait->GetMask(), SYNC_MASK);
    }

    EXPECT_EQ(CcuWriteVariableWithNotify(channel, var, remoteVarIdx, remoteNotifyIdx, SYNC_MASK), OK);
    if (auto postVar = TakeRep<CcuRep::CcuRepRemPostVar>(reps, CcuRepType::REM_POST_VAR)) {
        EXPECT_EQ(postVar->GetChannel(), channel);
        EXPECT_EQ(postVar->GetParam().Id(), varId);
        EXPECT_EQ(postVar->GetParamIndex(), remoteVarIdx);
        EXPECT_EQ(postVar->GetSemIndex(), remoteNotifyIdx);
        EXPECT_EQ(postVar->GetMask(), SYNC_MASK);
    }
    return OK;
}
} // namespace

CCU_FUNC_KERNEL_V2_TEST(Ut_CcuSyncApi_When_LocalOps_Expect_MatchingRep, SyncLocalOpsKernel, true)
CCU_FUNC_KERNEL_V2_TEST(Ut_CcuSyncApi_When_InvalidHandle_Expect_NotFoundAndNoRep, SyncInvalidHandleKernel, true)

TEST_F(HcommCcuControlApiTest, Ut_CcuSyncApi_When_RemoteOps_Expect_MatchingRepOperands)
{
    RunKernelWithChannel("SyncRemoteOpsKernel", SyncRemoteOpsKernel);
}
