/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/cast_utils_pub.h"
#include "aicpu_ts_uboe_channel.h"
#include "orion_adpt_utils.h"
#include "base_comm_node_accessors.h"
#include "endpoint.h"

// Orion
#include "adapter_rts_common.h"
#include "topo_common_types.h"
#include "rdma_handle_manager.h"

namespace hcomm {

AicpuTsUboeChannel::AicpuTsUboeChannel(
    EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc, CommEngine engine)
    : AicpuTsUboeUbRtpChannelHelper(
          endpointHandle, channelDesc, engine, HcommChannelKind::AICPU_TS_UBOE, "AicpuTsUboeChannel")
{
    type_ = static_cast<u32>(Hccl::TransportType::UBoE);
}

AicpuTsUboeChannel::~AicpuTsUboeChannel() = default;

ChannelStatus AicpuTsUboeChannel::GetStatus()
{
    if (channelStatus == ChannelStatus::READY) {
        return channelStatus;
    }
    if (channelStatus == ChannelStatus::INIT)
        uboeStatus = UboeStatus::INIT;

    if (!IsSocketReady())
        return channelStatus;

    HcclResult ret = ProcessUboeState();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] ProcessUboeState failed, ret=%d", __func__, ret);
        channelStatus = ChannelStatus::FAILED;
    }

    return channelStatus;
}

HcclResult AicpuTsUboeChannel::ParseInputParam()
{
    CHK_RET(AicpuTsUboeUbRtpChannelHelper::ParseInputParam());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::StartListen()
{
    CHK_RET(AicpuTsUboeUbRtpChannelHelper::StartListen());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::BuildSocket()
{
    CHK_RET(AicpuTsUboeUbRtpChannelHelper::BuildSocket());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::BuildConnection()
{
    CHK_RET(AicpuTsUboeUbRtpChannelHelper::BuildConnection());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::BuildNotify()
{
    CHK_RET(AicpuTsUboeUbRtpChannelHelper::BuildNotify());
    return HCCL_SUCCESS;
}
HcclResult AicpuTsUboeChannel::BuildBuffer()
{
    CHK_RET(AicpuTsUboeUbRtpChannelHelper::BuildBuffer());
    CHK_RET(BuildDrainResource());
    /*
        HccpRaGetDevBaseAttr
        获取urma read/write 单个wr的最大传输数据大小
        调用前,rdmaHandle_要在ParseInputParam中被赋值好,之后BuildConnection会使用获取的属性
        uboe的BuildConnection不再Init里面执行，Init之后会有单独流程建链
    */
    return HccpRaGetDevBaseAttr(local_.rdmaHandle, &devBaseAttr_);
}

HcclResult AicpuTsUboeChannel::Clean()
{
    local_.commonRes.connVec.clear();
    connections_.clear();

    rmtNotifyVec_.clear();
    locBufferVec_.clear();

    recvData_.clear();
    recvFinishMsg_.clear();
    recvEidData_.clear();
    sendData_.clear();
    sendFinishMsg_.clear();
    sendEidData_.clear();

    bufferNum_ = 0;
    connNum_ = 0;
    recvDataSize_ = 0;
    local_.ip = Hccl::IpAddress();
    remote_.ip = Hccl::IpAddress();

    {
        std::lock_guard<std::mutex> lock(remote_.memCache.mutex);
        rmtBufferVec_.clear();
        remote_.memCache.Invalidate();
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::Resume()
{
    channelStatus = ChannelStatus::INIT;
    uboeStatus = UboeStatus::INIT;
    HCCL_INFO("[%s::%s] reset uboe state machine to rebuild connection.", channelClassName_, __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::BuildStateMachineConn()
{
    UbConnBuildContext ctx;
    CHK_RET(PrepareUbConnBuildContext(local_.ep, remote_.ep, channelDesc_, ctx));
    CHK_RET(CheckUbSqDepth(ctx, devBaseAttr_));
    CHK_RET(CheckUbScqDepth(ctx, HrtUbJfcMode::STARS_POLL));

    Hccl::OpMode opMode = Hccl::OpMode::OPBASE;
    bool devUsed = true; // aicpu 为 true
    HCCL_INFO(
        "[AicpuTsUboeChannel::%s] LinkProtocol[%s], locIpv4Addr[%s], rmtIpv4Addr[%s]", __func__,
        ctx.protocol.Describe().c_str(), ctx.locAddr.Describe().c_str(), ctx.rmtAddr.Describe().c_str());
    HCCL_INFO(
        "[AicpuTsUboeChannel::%s] local_.ip[%s], remote_.ip[%s]", __func__, local_.ip.Describe().c_str(),
        remote_.ip.Describe().c_str());

    // UBOE 协议对应 HCOMM_TA_RTP_UBOE_TIMEOUT，由 base_comm 从环境变量获取后传入
    u8 taTimeOut = 0;
    uint32_t taTimeOutValue = 0;
    CHK_RET(hcomm::GetConfigMgr().GetRdmaConfig().GetTaRtpUboeTimeOut(taTimeOutValue));
    taTimeOut = static_cast<u8>(taTimeOutValue);
    std::unique_ptr<Hccl::DevUbConnection> ubConn = std::make_unique<Hccl::DevUbUboeConnection>(
        local_.rdmaHandle, local_.ip, remote_.ip, opMode, devUsed, Hccl::HrtUbJfcMode::STARS_POLL, ctx.locAddr,
        ctx.rmtAddr, ctx.qosPre, taTimeOut, COMM_ENGINE_AICPU_TS, ctx.sqDepth, ctx.scqDepth);
    CHK_SMART_PTR_NULL(ubConn);

    if (devBaseAttr_.maxReadSize == 0 || devBaseAttr_.maxWriteSize == 0) {
        HCCL_ERROR(
            "[%s::%s] maxReadSize[%u] or maxWriteSize[%u] must not be zero", channelClassName_, __func__,
            devBaseAttr_.maxReadSize, devBaseAttr_.maxWriteSize);
        return HCCL_E_PARA;
    }
    ubConn->SetMaxReadSize(devBaseAttr_.maxReadSize);
    ubConn->SetMaxWriteSize(devBaseAttr_.maxWriteSize);
    HCCL_INFO(
        "[%s::%s] maxReadSize[%u], maxWriteSize[%u]", channelClassName_, __func__, devBaseAttr_.maxReadSize,
        devBaseAttr_.maxWriteSize);

    local_.commonRes.connVec.clear();
    local_.commonRes.connVec.emplace_back(ubConn.get());
    connections_.clear();
    connections_.push_back(std::move(ubConn));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::SendFinish()
{
    HCCL_INFO("[%s::%s] start send Finish Msg [%s]", channelClassName_, __func__, FINISH_MSG);
    sendFinishMsg_ = std::vector<char>(FINISH_MSG, FINISH_MSG + FINISH_MSG_SIZE);
    TRY_CATCH_RETURN(local_.socket->SendAsync(sendFinishMsg_.data(), FINISH_MSG_SIZE););
    HCCL_INFO("[%s::%s] end send Finish Msg [%s]", channelClassName_, __func__, FINISH_MSG);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::RecvFinish()
{
    recvFinishMsg_.resize(FINISH_MSG_SIZE);
    HCCL_INFO("[%s::%s] start recv Finish Msg [%s]", channelClassName_, __func__, FINISH_MSG);
    TRY_CATCH_RETURN(local_.socket->RecvAsync(ReinterpretAs<u8*>(recvFinishMsg_.data()), FINISH_MSG_SIZE););
    HCCL_INFO("[%s::%s] end recv Finish Msg [%s]", channelClassName_, __func__, FINISH_MSG);
    return HCCL_SUCCESS;
}

void AicpuTsUboeChannel::EidPack()
{
    Hccl::IpAddress locIpv4Addr;
    CommAddrToIpAddress(local_.ep.commAddr, locIpv4Addr);
    Hccl::RdmaHandleManager::GetInstance().GetEidByIpv4Addr(locIpv4Addr, local_.ip);
    sendEidData_ = local_.ip.GetUniqueId();
    HCCL_INFO(
        "[AicpuTsUboeChannel::%s] locIpv4Addr[%s], local_.ip[%s], sendEidData_ size[%u]", __func__,
        locIpv4Addr.Describe().c_str(), local_.ip.Describe().c_str(), sendEidData_.size());
}

HcclResult AicpuTsUboeChannel::SendEidData()
{
    EidPack();
    TRY_CATCH_RETURN(local_.socket->SendAsync(sendEidData_.data(), sendEidData_.size()););
    HCCL_INFO("[AicpuTsUboeChannel::%s] send eid data, size=%llu", __func__, sendEidData_.size());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::RecvEidData()
{
    recvEidData_.resize(sendEidData_.size());
    TRY_CATCH_RETURN(local_.socket->RecvAsync(ReinterpretAs<u8*>(recvEidData_.data()), recvEidData_.size()););
    HCCL_INFO("[AicpuTsUboeChannel::%s] recv eid data, size=%llu", __func__, recvEidData_.size());
    return HCCL_SUCCESS;
}

void AicpuTsUboeChannel::RecvEidDataProcess() { RmtEidUnpackProc(remote_.ip); }

void AicpuTsUboeChannel::RmtEidUnpackProc(Hccl::IpAddress& rmtAddr)
{
    Hccl::IpAddress rmtEidAddr(recvEidData_);
    rmtAddr = rmtEidAddr;
    HCCL_INFO("[AicpuTsUboeChannel::%s] rmtAddr[%s]", __func__, rmtAddr.Describe().c_str());
}

HcclResult AicpuTsUboeChannel::HandleProcessData()
{
    bool needFinish = false;
    CHK_RET(RecvDataProcess(needFinish));
    if (needFinish) {
        uboeStatus = UboeStatus::SEND_FIN;
    } else {
        channelStatus = ChannelStatus::READY;
        uboeStatus = UboeStatus::READY;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::ProcessUboeState()
{
    auto SetState = [this](UboeStatus next, ChannelStatus ch) {
        this->uboeStatus = next;
        this->channelStatus = ch;
    };

    switch (uboeStatus) {
        case UboeStatus::INIT:
            SetState(UboeStatus::SEND_EID, ChannelStatus::SOCKET_OK);
            break;
        case UboeStatus::SEND_EID:
            CHK_RET(SendEidData());
            SetState(UboeStatus::RECV_EID, channelStatus);
            break;
        case UboeStatus::RECV_EID:
            CHK_RET(RecvEidData());
            SetState(UboeStatus::PROCESS_EID_DATA, channelStatus);
            break;
        case UboeStatus::PROCESS_EID_DATA:
            RecvEidDataProcess();
            SetState(UboeStatus::BUILD_CONN, channelStatus);
            break;
        case UboeStatus::BUILD_CONN:
            CHK_RET(BuildConn());
            SetState(UboeStatus::SEND_SIZE, channelStatus);
            break;
        case UboeStatus::SEND_SIZE: {
            bool isReady = false;
            CHK_RET(IsResReady(isReady));
            if (isReady) {
                CHK_RET(SendDataSize());
                SetState(UboeStatus::RECV_SIZE, channelStatus);
            }
        } break;
        case UboeStatus::RECV_SIZE:
            CHK_RET(RecvDataSize());
            SetState(local_.isRecvFirst ? UboeStatus::RECV_DATA : UboeStatus::SEND_DATA, channelStatus);
            break;
        case UboeStatus::SEND_DATA:
            CHK_RET(SendExchangeData());
            SetState(local_.isRecvFirst ? UboeStatus::PROCESS_DATA : UboeStatus::RECV_DATA, channelStatus);
            break;
        case UboeStatus::RECV_DATA:
            CHK_RET(RecvExchangeData());
            SetState(local_.isRecvFirst ? UboeStatus::SEND_DATA : UboeStatus::PROCESS_DATA, channelStatus);
            break;
        case UboeStatus::PROCESS_DATA:
            CHK_RET(HandleProcessData());
            break;
        case UboeStatus::SEND_FIN:
            if (IsConnsReady()) {
                CHK_RET(SendFinish());
                SetState(UboeStatus::RECV_FIN, channelStatus);
            }
            break;
        case UboeStatus::RECV_FIN:
            CHK_RET(RecvFinish());
            SetState(UboeStatus::SET_READY, channelStatus);
            break;
        case UboeStatus::SET_READY:
            channelStatus = ChannelStatus::READY;
            SetState(UboeStatus::READY, ChannelStatus::READY);
            break;
        default:
            break;
    }
    return HCCL_SUCCESS;
}

} // namespace hcomm
