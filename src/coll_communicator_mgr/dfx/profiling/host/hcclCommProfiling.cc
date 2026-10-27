/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcclCommProfiling.h"
#include "dfx_profiling_reporter.h"
#include "dfx/dfx_profiling_handler_pub.h"
#include "dfx/prof_utils_pub.h"
namespace hccl {

HcclResult HcclCommProfiling::ReportKernel(
    uint64_t beginTime, const std::string& commTag, const std::string& kernelName, uint32_t threadId,
    bool cachedReq) const
{
    u64 endTime = hcomm::GetProfCycleTime();
    uint64_t cmdItemId = hcomm::ProfStr2Id(kernelName.c_str(), kernelName.length());
    CHK_RET(Hccl::DfxProfilingHandler::GetInstance().ReportNodeApi(beginTime, endTime, cmdItemId, threadId, cachedReq));
    CHK_RET(Hccl::DfxProfilingHandler::GetInstance().ReportNodeBasicInfo(endTime, cmdItemId, threadId, cachedReq));
    HCCL_INFO(
        "[HcclCommProfiling][ReportKernel] beginTime [%llu] endTime[%llu] kernelName[%s] commTag[%s] threadId[%u]",
        beginTime, endTime, kernelName.c_str(), commTag.c_str(), threadId);
    return HCCL_SUCCESS;
}

HcclCommProfiling::HcclCommProfiling(u32 deviceId, Hccl::MirrorTaskManager* mirrorTaskManager)
    : mirrorTaskManager_(mirrorTaskManager)
{
    (void)deviceId;
}

HcclResult HcclCommProfiling::Init()
{
    if (initializedFlag_) {
        return HCCL_SUCCESS;
    }
    CHK_RET(Hccl::DfxProfilingHandler::GetInstance().Init());
    profilingReporter_
        = std::make_unique<Hccl::DfxProfilingReporter>(mirrorTaskManager_, &Hccl::DfxProfilingHandler::GetInstance());
    CHK_RET(profilingReporter_->Init());
    initializedFlag_ = true;
    return HCCL_SUCCESS;
}

// HcclCommProfiling任务上报
HcclResult HcclCommProfiling::ReportAllTasks(bool cachedReq) const
{
    CHK_RET(profilingReporter_->ReportAllTasks(cachedReq));
    return HCCL_SUCCESS;
}

// HcclCommProfiling::ReportOp实现
HcclResult HcclCommProfiling::ReportOp(uint64_t beginTime, bool cachedReq, bool isOpBase)
{
    CHK_RET(profilingReporter_->ReportOp(beginTime, cachedReq, isOpBase));
    return HCCL_SUCCESS;
}

HcclResult HcclCommProfiling::ReportMc2CommInfo(const Mc2CommInfo& mc2CommInfo)
{
    CHK_RET(profilingReporter_->CallReportMc2CommInfo(
        mc2CommInfo.FreeStreamId, mc2CommInfo.streamsId, mc2CommInfo.groupname, mc2CommInfo.myRankId,
        mc2CommInfo.rankSize, mc2CommInfo.parentRankId));
    return HCCL_SUCCESS;
}

void HcclCommProfiling::SetCurrDfxOpInfo(std::shared_ptr<Hccl::DfxOpInfo> dfxOpInfo)
{
    profilingReporter_->SetCurrDfxOpInfo(dfxOpInfo);
}

// HcclCommProfiling::UpdateProfStat实现
void HcclCommProfiling::UpdateProfStat() { profilingReporter_->UpdateProfStat(); }
Hccl::MirrorTaskManager* HcclCommProfiling::GetMirrorTaskManager() const { return mirrorTaskManager_; }
} // namespace hccl
