/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_ub_mem_channel.h"
#include "endpoint.h"
#include "hcomm_c_adpt.h"
#include "base_comm_node_accessors.h"

namespace hcomm {

AivUbMemChannel::AivUbMemChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc, CommEngine engine)
    : Channel(endpointHandle, channelDesc, engine, HcommChannelKind::AIV_UB_MEM, "AivUbMemChannel")
{}

AivUbMemChannel::~AivUbMemChannel() { StopListen(); }

HcclResult AivUbMemChannel::ParseInputParam()
{
    CHK_RET(Channel::ParseInputParam());

    CHK_RET(CollectLocalRmaBuffers(local_.commonRes.bufferVec));

    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::StartListen()
{
    CHK_RET(Channel::StartListen());
    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::BuildSocket()
{
    CHK_RET(Channel::BuildSocket());
    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::BuildConnection()
{
    CHK_RET(Channel::BuildConnection());
    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::BuildNotify()
{
    CHK_RET(Channel::BuildNotify());
    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::BuildBuffer()
{
    CHK_RET(Channel::BuildBuffer());
    EXCEPTION_CATCH(
        transport_ = std::make_unique<AivUbMemTransport>(local_.commonRes.bufferVec, local_.socket, channelDesc_),
        return HCCL_E_PTR);
    CHK_PTR_NULL(transport_);
    CHK_RET(transport_->Init());
    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::GetNotifyNum([[maybe_unused]] uint32_t* notifyNum) const
{
    HCCL_INFO("[%s::%s] AivUbMemChannel GetNotifyNum is not supported.", channelClassName_, __func__);
    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    return transport_->GetRemoteMems(memNum, remoteMem, memInfos);
}

ChannelStatus AivUbMemChannel::GetStatus()
{
    return TransportStatusToChannelStatus(transport_->GetStatus(), local_.ep, GetChannelDesc());
}

HcclResult AivUbMemChannel::UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum)
{
    return transport_->UpdateMemInfo(memHandles, memHandleNum);
}

HcclResult AivUbMemChannel::Clean()
{
    // 该模式当前不支持N秒快恢
    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::Resume()
{
    // 该模式当前不支持N秒快恢
    return HCCL_SUCCESS;
}

void AivUbMemChannel::StopListen() { (void)HcommEndpointStopListen(endpointHandle_, local_.serverPort); }

} // namespace hcomm
