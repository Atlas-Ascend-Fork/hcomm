/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_sync.h"

#include "ccu_rep_v1.h"
#include "ccu_log.h"
#include "config_plf_log_v2.h"

namespace hcomm {

using Hccl::PLF_DATA_OP;

// mask 由调用方独立传入，不再从 CompletedEvent 上读取
static HcclResult RecordEvent(CcuKernel& kernel, CcuRep::CompletedEvent event, uint32_t mask);
static HcclResult WaitEvent(CcuKernel& kernel, CcuRep::CompletedEvent event, uint32_t mask);

/*========== Event信号同步类 相关接口 ==========*/
CcuResult ccu_sync::EventRecord(CcuKernel& kernel, CcuEventHandle eventHandle, uint32_t mask)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[EventRecord] eventHandle=%llu, mask=%u", eventHandle, mask);
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveEvent(eventHandle, &event));
    // 复用已有的 RecordEvent 实现（内部 Append CcuRepLocRecordEvent）
    CCU_CHK_RET(RecordEvent(kernel, *event, mask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_sync::EventWait(CcuKernel& kernel, CcuEventHandle eventHandle, uint32_t mask)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[EventWait] eventHandle=%llu, mask=%u", eventHandle, mask);
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveEvent(eventHandle, &event));
    // 复用已有的 WaitEvent 实现（内部 Append CcuRepLocWaitEvent）
    CCU_CHK_RET(WaitEvent(kernel, *event, mask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_sync::LocalNotifyRecord(CcuKernel& kernel, const char* notifyTag, const uint32_t mask)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[LocalNotifyRecord] tag=%s, mask=%u", (notifyTag ? notifyTag : "null"), mask);
    if (notifyTag == nullptr) {
        HCCL_ERROR("[CcuKernel][%s] notifyTag is nullptr, please check.", __func__);
        return CcuResult::CCU_E_PTR;
    }
    if (kernel.CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR("[CcuKernel][%s] is not supported in loop block, please check.", __func__);
        return kernel.LatchBodyErrorForRepBuild(CcuResult::CCU_E_NOT_SUPPORT);
    }

    const std::string tagKey(notifyTag);

    auto& sharedNotifies = kernel.GetImportedRes().sharedNotifies;
    if (sharedNotifies.find(tagKey) == sharedNotifies.end()) {
        CcuRep::LocalNotify localNotify;
        sharedNotifies.insert({tagKey, localNotify});
    }

    kernel.Append(
        std::make_shared<CcuRep::CcuRepRecordSharedNotify>(kernel.GetInsGenerator(), sharedNotifies.at(tagKey), mask));

    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_sync::LocalNotifyWait(CcuKernel& kernel, const char* notifyTag, const uint32_t mask)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[LocalNotifyWait] tag=%s, mask=%u", (notifyTag ? notifyTag : "null"), mask);
    if (notifyTag == nullptr) {
        HCCL_ERROR("[CcuKernel][%s] notifyTag is nullptr, please check.", __func__);
        return CcuResult::CCU_E_PTR;
    }

    const std::string tagKey(notifyTag);

    auto& sharedNotifies = kernel.GetExportedRes().sharedNotifies;
    if (sharedNotifies.find(tagKey) == sharedNotifies.end()) {
        CcuRep::LocalNotify notify = kernel.CreateLocalNotifyForRepBuild();
        kernel.GetExportedRes().sharedNotifies.insert({tagKey, notify});
    }

    bool isProfiling = kernel.CurrentBlock()->Type() != CcuRep::CcuRepType::LOOP_BLOCK;
    kernel.Append(std::make_shared<CcuRep::CcuRepLocWaitNotify>(
        kernel.GetInsGenerator(), kernel.GetExportedRes().sharedNotifies.at(tagKey), mask, isProfiling));
    return CcuResult::CCU_SUCCESS;
}

CcuResult
ccu_sync::NotifyRecord(CcuKernel& kernel, const ChannelHandle channel, uint32_t remoteNotifyIdx, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[NotifyRecord] channel=%llu, remoteNotifyIdx=%u, mask=%u", channel, remoteNotifyIdx, mask);
    if (kernel.CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR("[%s] NotifyRecord is not allowed inside a ccu::Loop body", __func__);
        return kernel.LatchBodyErrorForRepBuild(CcuResult::CCU_E_NOT_SUPPORT);
    }
    kernel.AddChannelForRepBuild(channel);
    kernel.Append(std::make_shared<CcuRep::CcuRepRemPostSem>(kernel.GetInsGenerator(), channel, remoteNotifyIdx, mask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_sync::NotifyWait(CcuKernel& kernel, const ChannelHandle channel, uint32_t localNotifyIdx, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[NotifyWait] channel=%llu, localNotifyIdx=%u, mask=%u", channel, localNotifyIdx, mask);
    bool isProfiling = kernel.CurrentBlock()->Type() != CcuRep::CcuRepType::LOOP_BLOCK;
    if (isProfiling) {
        CCU_CHK_RET(static_cast<HcclResult>(kernel.AddProfiling(channel, "NotifyWait", localNotifyIdx, mask)));
    }
    kernel.AddChannelForRepBuild(channel);
    kernel.Append(std::make_shared<CcuRep::CcuRepRemWaitSem>(
        kernel.GetInsGenerator(), channel, localNotifyIdx, mask, isProfiling));
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_sync::CascCntWait(CcuKernel& kernel, HcommCcuCascCntHandle cntHandle, uint64_t tgtValue)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[CascCntWait] cntHandle=%llu, tgtValue=%llu", cntHandle, tgtValue);
    kernel.AddCascCntHandle(cntHandle);
    auto rep = std::make_shared<CcuRep::CcuRepCascCntWait>(kernel.GetInsGenerator(), cntHandle, tgtValue);
    kernel.Append(rep);
    return CcuResult::CCU_SUCCESS;
}
CcuResult
ccu_sync::CascCntClear(CcuKernel& kernel, HcommCcuCascCntHandle cntHandle, CcuEventHandle eventHandle, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[CascCntClear] cntHandle=%llu, eventHandle=%llu, mask=%u", cntHandle, eventHandle, mask);
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveEvent(eventHandle, &event));
    kernel.AddCascCntHandle(cntHandle);
    auto rep = std::make_shared<CcuRep::CcuRepCascCntClear>(kernel.GetInsGenerator(), cntHandle, *event, mask);
    kernel.Append(rep);
    kernel.SetDependencyInfo(event->Id(), mask, rep);
    return CcuResult::CCU_SUCCESS;
}
CcuResult ccu_sync::WriteVariableWithNotify(
    CcuKernel& kernel, const ChannelHandle channel, CcuVariableHandle varHandle, uint32_t remoteVarIdx,
    uint32_t remoteNotifyIdx, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[WriteVariableWithNotify] channel=%llu, varHandle=%llu, remoteVarIdx=%u,"
        " remoteNotifyIdx=%u, mask=%u",
        channel, varHandle, remoteVarIdx, remoteNotifyIdx, mask);
    if (kernel.CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR("[%s] WriteVariableWithNotify is not allowed inside a ccu::Loop body", __func__);
        return kernel.LatchBodyErrorForRepBuild(CcuResult::CCU_E_NOT_SUPPORT);
    }
    CcuRep::Variable* var{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &var));
    kernel.AddChannelForRepBuild(channel);
    kernel.Append(std::make_shared<CcuRep::CcuRepRemPostVar>(
        kernel.GetInsGenerator(), *var, channel, remoteVarIdx, remoteNotifyIdx, mask));
    return CcuResult::CCU_SUCCESS;
}

static HcclResult RecordEvent(CcuKernel& kernel, CcuRep::CompletedEvent event, uint32_t mask)
{
    if (kernel.CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR("[CcuKernel][%s] is not supported in loop block, please check.", __func__);
        kernel.LatchBodyErrorForRepBuild(CcuResult::CCU_E_NOT_SUPPORT);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    auto rep = std::make_shared<CcuRep::CcuRepLocRecordEvent>(kernel.GetInsGenerator(), event, mask);
    kernel.Append(rep);
    kernel.SetDependencyInfo(event.Id(), mask, rep);
    return HCCL_SUCCESS;
}

static HcclResult WaitEvent(CcuKernel& kernel, CcuRep::CompletedEvent event, uint32_t mask)
{
    bool isProfiling = kernel.CurrentBlock()->Type() != CcuRep::CcuRepType::LOOP_BLOCK;
    auto rep = std::make_shared<CcuRep::CcuRepLocWaitEvent>(kernel.GetInsGenerator(), event, mask, isProfiling);
    if (isProfiling) {
        CHK_RET(static_cast<HcclResult>(kernel.AddProfiling("WaitEvent", rep->GetMask())));
    }
    rep->SetDependencyInfo(kernel.GetDependencyInfo(event.Id()));
    kernel.EraseDependencyInfo(event.Id());
    kernel.Append(rep);
    return HCCL_SUCCESS;
}

} // namespace hcomm
