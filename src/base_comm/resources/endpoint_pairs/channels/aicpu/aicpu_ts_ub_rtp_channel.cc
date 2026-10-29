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
#include "aicpu_ts_ub_rtp_channel.h"
#include "orion_adpt_utils.h"
#include "base_comm_node_accessors.h"
#include "endpoint.h"

// Orion
#include "topo_common_types.h"

namespace hcomm {

HcclResult AicpuTsUbRtpChannel::ParseInputParam()
{
    CHK_RET(AicpuTsUboeUbRtpChannelHelper::ParseInputParam());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUbRtpChannel::StartListen()
{
    CHK_RET(AicpuTsUboeUbRtpChannelHelper::StartListen());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUbRtpChannel::BuildSocket()
{
    CHK_RET(AicpuTsUboeUbRtpChannelHelper::BuildSocket());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUbRtpChannel::BuildConnection()
{
    CHK_RET(AicpuTsUboeUbRtpChannelHelper::BuildConnection());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUbRtpChannel::BuildNotify()
{
    CHK_RET(AicpuTsUboeUbRtpChannelHelper::BuildNotify());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUbRtpChannel::BuildBuffer()
{
    CHK_RET(AicpuTsUboeUbRtpChannelHelper::BuildBuffer());
    /*
        HccpRaGetDevBaseAttr
        获取urma read/write 单个wr的最大传输数据大小
        调用前,rdmaHandle_要在ParseInputParam中被赋值好,之后BuildConnection会使用获取的属性
        ub_rtp的BuildConnection不再Init里面执行，Init之后会有单独流程建链
    */
    return HccpRaGetDevBaseAttr(local_.rdmaHandle, &devBaseAttr_);
}

ChannelStatus AicpuTsUbRtpChannel::GetStatus()
{
    if (channelStatus == ChannelStatus::READY) {
        return channelStatus;
    }
    if (channelStatus == ChannelStatus::INIT)
        ubRtpStatus = UbRtpStatus::INIT;

    if (!IsSocketReady())
        return channelStatus;

    HcclResult ret = ProcessUbRtpState();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] ProcessUbRtpState failed, ret=%d", __func__, ret);
        channelStatus = ChannelStatus::FAILED;
    }

    return channelStatus;
}

HcclResult AicpuTsUbRtpChannel::Clean()
{
    HCCL_INFO(
        "[%s::%s] start, channelStatus[%u], ubRtpStatus[%u]", channelClassName_, __func__,
        static_cast<u32>(channelStatus), static_cast<u32>(ubRtpStatus));
    local_.commonRes.connVec.clear();
    connections_.clear();

    rmtNotifyVec_.clear();
    locBufferVec_.clear();

    recvData_.clear();
    recvFinishMsg_.clear();
    sendData_.clear();
    sendFinishMsg_.clear();

    bufferNum_ = 0;
    connNum_ = 0;
    recvDataSize_ = 0;

    {
        std::lock_guard<std::mutex> lock(remote_.memCache.mutex);
        rmtBufferVec_.clear();
        remote_.memCache.Invalidate();
    }

    channelStatus = ChannelStatus::INIT;
    ubRtpStatus = UbRtpStatus::INIT;

    HCCL_INFO("[%s::%s] done.", channelClassName_, __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUbRtpChannel::Resume()
{
    HCCL_INFO(
        "[%s::%s] start, channelStatus[%u], ubRtpStatus[%u]", channelClassName_, __func__,
        static_cast<u32>(channelStatus), static_cast<u32>(ubRtpStatus));
    channelStatus = ChannelStatus::INIT;
    ubRtpStatus = UbRtpStatus::INIT;
    HCCL_INFO("[%s::%s] done.", channelClassName_, __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUbRtpChannel::BuildStateMachineConn()
{
    UbConnBuildContext ctx;
    CHK_RET(PrepareUbConnBuildContext(local_.ep, remote_.ep, channelDesc_, ctx));
    CHK_RET(CheckUbSqDepth(ctx, devBaseAttr_));
    CHK_RET(CheckUbScqDepth(ctx, HrtUbJfcMode::STARS_POLL));

    Hccl::OpMode opMode = Hccl::OpMode::OPBASE;
    bool devUsed = true; // aicpu 为 true
    HCCL_INFO(
        "[%s::%s] LinkProtocol[%s], local_.ip[%s], remote_.ip[%s], qos[%u]", channelClassName_, __func__,
        ctx.protocol.Describe().c_str(), local_.ip.Describe().c_str(), remote_.ip.Describe().c_str(),
        static_cast<unsigned int>(ctx.qosPre));

    // UB_RTP 协议对应 HCOMM_TA_RTP_UB_TIMEOUT，由 base_comm 从环境变量获取后传入
    u8 taTimeOut = 0;
    uint32_t taTimeOutValue = 0;
    CHK_RET(hcomm::GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(taTimeOutValue));
    taTimeOut = static_cast<u8>(taTimeOutValue);
    std::unique_ptr<Hccl::DevUbConnection> ubConn = std::make_unique<Hccl::DevUbRtpConnection>(
        local_.rdmaHandle, local_.ip, remote_.ip, opMode, devUsed, Hccl::HrtUbJfcMode::STARS_POLL, local_.ip,
        remote_.ip, ctx.qosPre, taTimeOut, COMM_ENGINE_AICPU_TS, ctx.sqDepth, ctx.scqDepth);
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

HcclResult AicpuTsUbRtpChannel::SendFinish()
{
    HCCL_INFO("[%s::%s] start send Finish Msg [%s]", channelClassName_, __func__, UB_RTP_FINISH_MSG);
    sendFinishMsg_ = std::vector<char>(UB_RTP_FINISH_MSG, UB_RTP_FINISH_MSG + FINISH_MSG_SIZE);
    TRY_CATCH_RETURN(local_.socket->SendAsync(sendFinishMsg_.data(), FINISH_MSG_SIZE););
    HCCL_INFO("[%s::%s] end send Finish Msg [%s]", channelClassName_, __func__, UB_RTP_FINISH_MSG);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUbRtpChannel::RecvFinish()
{
    recvFinishMsg_.resize(FINISH_MSG_SIZE);
    HCCL_INFO("[%s::%s] start recv Finish Msg [%s]", channelClassName_, __func__, UB_RTP_FINISH_MSG);
    TRY_CATCH_RETURN(local_.socket->RecvAsync(ReinterpretAs<u8*>(recvFinishMsg_.data()), FINISH_MSG_SIZE););
    HCCL_INFO("[%s::%s] end recv Finish Msg [%s]", channelClassName_, __func__, UB_RTP_FINISH_MSG);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUbRtpChannel::ProcessUbRtpState()
{
    auto SetState = [this](UbRtpStatus next, ChannelStatus ch) {
        ubRtpStatus = next;
        channelStatus = ch;
    };

    switch (ubRtpStatus) {
        case UbRtpStatus::INIT:
            SetState(UbRtpStatus::BUILD_CONN, channelStatus);
            break;
        case UbRtpStatus::BUILD_CONN:
            CHK_RET(BuildConn());
            SetState(UbRtpStatus::SEND_SIZE, channelStatus);
            break;
        case UbRtpStatus::SEND_SIZE: {
            bool isReady = false;
            CHK_RET(IsResReady(isReady));
            if (isReady) {
                CHK_RET(SendDataSize());
                SetState(UbRtpStatus::RECV_SIZE, channelStatus);
            }
        } break;
        case UbRtpStatus::RECV_SIZE:
        case UbRtpStatus::SEND_DATA:
        case UbRtpStatus::RECV_DATA:
        case UbRtpStatus::PROCESS_DATA:
            CHK_RET(ProcessUbRtpDataState());
            break;
        case UbRtpStatus::SEND_FIN:
            if (IsConnsReady()) {
                CHK_RET(SendFinish());
                SetState(UbRtpStatus::RECV_FIN, channelStatus);
            }
            break;
        case UbRtpStatus::RECV_FIN:
            CHK_RET(RecvFinish());
            SetState(UbRtpStatus::SET_READY, channelStatus);
            break;
        case UbRtpStatus::SET_READY:
            channelStatus = ChannelStatus::READY;
            SetState(UbRtpStatus::READY, ChannelStatus::READY);
            break;
        default:
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUbRtpChannel::ProcessUbRtpDataState()
{
    switch (ubRtpStatus) {
        case UbRtpStatus::RECV_SIZE:
            CHK_RET(RecvDataSize());
            ubRtpStatus = local_.isRecvFirst ? UbRtpStatus::RECV_DATA : UbRtpStatus::SEND_DATA;
            break;
        case UbRtpStatus::SEND_DATA:
            CHK_RET(SendExchangeData());
            ubRtpStatus = local_.isRecvFirst ? UbRtpStatus::PROCESS_DATA : UbRtpStatus::RECV_DATA;
            break;
        case UbRtpStatus::RECV_DATA:
            CHK_RET(RecvExchangeData());
            ubRtpStatus = local_.isRecvFirst ? UbRtpStatus::SEND_DATA : UbRtpStatus::PROCESS_DATA;
            break;
        case UbRtpStatus::PROCESS_DATA: {
            bool needFinish = false;
            CHK_RET(RecvDataProcess(needFinish));
            if (needFinish) {
                ubRtpStatus = UbRtpStatus::SEND_FIN;
            } else {
                channelStatus = ChannelStatus::READY;
                ubRtpStatus = UbRtpStatus::READY;
            }
            break;
        }
        default:
            break;
    }
    return HCCL_SUCCESS;
}

} // namespace hcomm
