/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_rtsq_poll_completion_daemon.h"
#include "comm_engine/thread/device/device_thread_pub.h"
#include "log.h"
#include "stream_lite.h"

namespace hcomm {

CollRtsqPollCompletionDaemon& CollRtsqPollCompletionDaemon::GetInstance()
{
    static CollRtsqPollCompletionDaemon instance;
    return instance;
}

void CollRtsqPollCompletionDaemon::Call()
{
    std::shared_lock<std::shared_mutex> rwlock(CollCommAicpuMgr::GetInstance().GetMutex());

    std::vector<std::pair<std::string, CollCommAicpu*>> aicpuCommInfo;
    HcclResult ret = CollCommAicpuMgr::GetInstance().GetAllComms(aicpuCommInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CollRtsqPoll][BackGround] GetAllComms failed, errNo[0x%016llx]", ret);
        return;
    }

    for (auto& commInfo : aicpuCommInfo) {
        CollCommAicpu* aicpuComm = commInfo.second;
        if (aicpuComm == nullptr) {
            continue;
        }
        HcclCommStatus status = aicpuComm->GetCommmStatus();
        if (status == HcclCommStatus::HCCL_COMM_STATUS_INVALID
            || status == HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING) {
            continue;
        }

        CommEngineResAicpuMgr* resMgr = aicpuComm->GetCommEngineResMgr();
        if (resMgr == nullptr) {
            continue;
        }
        std::shared_lock<std::shared_mutex> threadRwlock(resMgr->GetThreadMutex());
        const std::vector<ThreadHandle>& threads = resMgr->GetAllThread();
        for (auto thread : threads) {
            if (thread == 0) {
                continue;
            }
            void* streamLitePtr = nullptr;
            if (HcommThreadGetStreamLite(thread, &streamLitePtr) != HCOMM_SUCCESS) {
                continue;
            }
            Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(streamLitePtr);
            if (streamLite == nullptr) {
                continue;
            }
            Hccl::RtsqBase* rtsq = streamLite->GetRtsq();
            if (rtsq == nullptr) {
                continue;
            }
            rtsq->PollCompletion();
        }
    }
}

} // namespace hcomm
