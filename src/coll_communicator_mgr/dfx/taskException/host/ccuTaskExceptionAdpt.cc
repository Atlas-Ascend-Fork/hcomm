/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// L2 (coll_comm) glue for the CCU task exception decoder. The decoder lives in
// base_comm and queries the communication domain through callbacks; the L2 types
// (CollComm / HcclCommunicator / RankGraph) stay on this side of the boundary.

#include "comm_engine/ccu/ccu_task_exception_pub.h"
#include "coll_comm.h"
#include "hccl_communicator.h"
#include "rank_gph.h"
#include "hcclCommDfx.h"
#include "dfx/res_pub.h"
#include "log.h"

namespace hcomm {
namespace {

    void GetGroupRankInfoImpl(void* comm, std::string& commId, std::string& udi, u32& rankSize, u32& rankId)
    {
        auto* collComm = static_cast<hccl::CollComm*>(comm);
        if (collComm == nullptr) {
            HCCL_ERROR("[CcuTaskExceptionAdpt][GetGroupRankInfoImpl] comm is nullptr.");
            return;
        }
        commId = collComm->GetCommId();
        udi = collComm->GetCommConfig().GetConfigUdi();
        rankSize = collComm->GetRankSize();
        rankId = collComm->GetMyRankId();
    }

    Hccl::RankGraph* GetRankGraphByComm(void* comm)
    {
        auto* collComm = static_cast<hccl::CollComm*>(comm);
        if (collComm == nullptr) {
            HCCL_ERROR("[CcuTaskExceptionAdpt][GetRankGraphByComm] comm is nullptr.");
            return nullptr;
        }
        Hccl::HcclCommunicator* commV2 = static_cast<Hccl::HcclCommunicator*>(collComm->GetCommunicatorV2());
        if (commV2 == nullptr) {
            HCCL_ERROR("[CcuTaskExceptionAdpt][GetRankGraphByComm] commV2 is nullptr.");
            return nullptr;
        }
        void* rankGraph = nullptr;
        HcclResult ret = commV2->GetRankGraphV2(rankGraph);
        if (ret != HCCL_SUCCESS || rankGraph == nullptr) {
            HCCL_ERROR("[CcuTaskExceptionAdpt][GetRankGraphByComm] GetRankGraphV2 failed, ret[%d].", ret);
            return nullptr;
        }
        return static_cast<Hccl::RankGraph*>(rankGraph);
    }

    void GetCcuCqeErrInfoByRankIdImpl(void* comm, u32 rankId, u32& remoteLocalId, std::string& netInstanceId)
    {
        if (rankId == Hccl::DFX_INVALID_RANKID) {
            HCCL_ERROR("[CcuTaskExceptionAdpt][GetCcuCqeErrInfoByRankIdImpl] rankId is invalid.");
            return;
        }
        Hccl::RankGraph* rankGraph = GetRankGraphByComm(comm);
        if (rankGraph == nullptr) {
            return;
        }
        remoteLocalId = rankGraph->GetLocalId(rankId);
        const Hccl::NetInstance* netInstance = rankGraph->GetNetInstanceByRankId(0, rankId);
        if (netInstance == nullptr) {
            HCCL_ERROR(
                "[CcuTaskExceptionAdpt][GetCcuCqeErrInfoByRankIdImpl] netInstance is nullptr, rankId[%u].", rankId);
            return;
        }
        netInstanceId = netInstance->GetNetInstId();
    }

    HcclResult GetChannelRemoteRankIdImpl(void* comm, u64 channelHandle, u32& remoteRank)
    {
        auto* collComm = static_cast<hccl::CollComm*>(comm);
        if (collComm == nullptr) {
            HCCL_ERROR("[CcuTaskExceptionAdpt][GetChannelRemoteRankIdImpl] comm is nullptr.");
            return HCCL_E_PTR;
        }
        return hccl::HcclCommDfx::GetChannelRemoteRankId(collComm->GetCommId(), channelHandle, remoteRank);
    }

    struct CcuTaskExceptionCallbacksRegister {
        CcuTaskExceptionCallbacksRegister()
        {
            CcuTaskExceptionCallbacks callbacks{};
            callbacks.getGroupRankInfo = &GetGroupRankInfoImpl;
            callbacks.getCcuCqeErrInfoByRankId = &GetCcuCqeErrInfoByRankIdImpl;
            callbacks.getChannelRemoteRankId = &GetChannelRemoteRankIdImpl;
            RegisterCcuTaskExceptionCallbacks(callbacks);
        }
    };
    const CcuTaskExceptionCallbacksRegister g_ccuTaskExceptionCallbacksRegister;
} // namespace
} // namespace hcomm
