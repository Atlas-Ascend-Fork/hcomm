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
#include "../sockets/socket_mgr.h"
#include "orion_adpt_utils.h"
#include "hcomm_c_adpt.h"

namespace hcomm {

constexpr uint16_t DEFAULT_LISTENING_PORT = 60001;

AivUbMemChannel::AivUbMemChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc)
    : endpointHandle_(endpointHandle),
      channelDesc_(channelDesc)
{}

AivUbMemChannel::~AivUbMemChannel()
{
    PutSocketIfNeeded();
    StopListen();
}

void AivUbMemChannel::PutSocketIfNeeded()
{
    if (socket_ == nullptr) {
        return;
    }
    if (socketConfig_ == nullptr) {
        socket_ = nullptr;
        return;
    }
    if (channelDesc_.socket == nullptr) {
        (void)SocketMgr::GetInstance(devicePhyId_).DestroySocket(*socketConfig_);
    }
    socket_ = nullptr;
}

HcclResult AivUbMemChannel::ParseInputParam()
{
    Endpoint* localEpPtr = static_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(localEpPtr);
    localEp_ = localEpPtr->GetEndpointDesc();
    devicePhyId_ = localEp_.loc.device.devPhyId;
    remoteEp_ = channelDesc_.remoteEndpoint;
    socket_ = static_cast<Hccl::Socket*>(channelDesc_.socket);
    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::BuildTransport()
{
    EXCEPTION_CATCH(transport_ = std::make_unique<AivUbMemTransport>(socket_, channelDesc_), return HCCL_E_PTR);
    CHK_PTR_NULL(transport_);
    CHK_RET(transport_->Init());
    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::StartListen()
{
    if (channelDesc_.role != HCOMM_SOCKET_ROLE_SERVER) {
        return HCCL_SUCCESS;
    }

    uint16_t port = channelDesc_.port;
    HCCL_INFO("[AivUbMemChannel::%s] Start. EndpointHandle[%p], port[%u]", __func__, endpointHandle_, port);
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO("[AivUbMemChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
    }
    CHK_RET(static_cast<HcclResult>(HcommEndpointStartListen(endpointHandle_, port, nullptr)));
    HCCL_INFO("[AivUbMemChannel::%s] SUCCESS. port[%u].", __func__, port);
    serverListened_ = true;
    return HCCL_SUCCESS;
}

void AivUbMemChannel::StopListen()
{
    if (channelDesc_.role != HCOMM_SOCKET_ROLE_SERVER || !serverListened_) {
        return;
    }
    uint16_t port = channelDesc_.port == 0 ? DEFAULT_LISTENING_PORT : channelDesc_.port;
    (void)HcommEndpointStopListen(endpointHandle_, port);
    serverListened_ = false;
}

HcclResult AivUbMemChannel::BuildSocket()
{
    if (socket_ != nullptr) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[AivUbMemChannel][%s] socket ptr is NULL, rebuildSocket", __func__);

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
    HCCL_INFO("[AivUbMemChannel][%s] built linkData: %s", __func__, linkData.Describe().c_str());
    uint16_t port = channelDesc_.port;
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO("[AivUbMemChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
    }
    std::string socketTag
        = (channelDesc_.channelName != nullptr) ? std::string(channelDesc_.channelName) : "AUTOMATIC_SOCKET_TAG";
    bool useRawTag = (channelDesc_.channelName != nullptr && channelDesc_.channelName[0] != '\0');
    if (channelDesc_.role == HCOMM_SOCKET_ROLE_RESERVED) {
        EXCEPTION_CATCH(
            socketConfigHolder_ = std::make_unique<Hccl::SocketConfig>(linkData, port, socketTag), return HCCL_E_PTR);
        socketConfigHolder_->noRankId = true;
    } else {
        bool isServer = (channelDesc_.role == HCOMM_SOCKET_ROLE_SERVER);
        EXCEPTION_CATCH(
            socketConfigHolder_ = std::make_unique<Hccl::SocketConfig>(linkData, port, socketTag, isServer),
            return HCCL_E_PTR);
    }
    if (useRawTag) {
        socketConfigHolder_->SetHccpTag(socketTag);
    }
    socketConfig_ = socketConfigHolder_.get();
    CHK_RET(SocketMgr::GetInstance(devicePhyId_).GetSocket(*socketConfigHolder_, socket_));
    HCCL_INFO("[AivUbMemChannel][%s] SUCCESS. port[%u].", __func__, port);
    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::Init()
{
    CHK_RET(ParseInputParam());
    CHK_RET(StartListen());
    CHK_RET(BuildSocket());
    CHK_RET(BuildTransport());
    return HCCL_SUCCESS;
}

ChannelStatus AivUbMemChannel::GetStatus()
{
    return TransportStatusToChannelStatus(transport_->GetStatus(), localEp_, GetChannelDesc());
}

HcclResult AivUbMemChannel::GetNotifyNum([[maybe_unused]] uint32_t* notifyNum) const
{
    HCCL_INFO("AivUbMemChannel GetNotifyNum is not supported.");
    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    return transport_->GetRemoteMems(memNum, remoteMem, memInfos);
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

HcclResult AivUbMemChannel::NotifyRecord([[maybe_unused]] const uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AivUbMemChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AivUbMemChannel::NotifyWait([[maybe_unused]] const uint32_t localNotifyIdx, [[maybe_unused]] const uint32_t timeout)
{
    HCCL_INFO("[AivUbMemChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AivUbMemChannel::WriteWithNotify(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] const uint64_t len,
    [[maybe_unused]] uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AivUbMemChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AivUbMemChannel::Write([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[AivUbMemChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AivUbMemChannel::Read([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[AivUbMemChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AivUbMemChannel::ChannelFence()
{
    HCCL_INFO("[AivUbMemChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}
} // namespace hcomm
