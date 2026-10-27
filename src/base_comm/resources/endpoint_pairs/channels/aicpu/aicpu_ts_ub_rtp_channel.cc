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
#include "hcomm_res_mgr.h"
#include "endpoint.h"

// Orion
#include "topo_common_types.h"

namespace hcomm {

HcclResult AicpuTsUbRtpChannel::Init()
{
    s32 devLogicId;
    CHK_RET(hrtGetDevice(&devLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(devLogicId), devicePhyId_));
    CHK_RET(ParseInputParam());

    // UB_RTP 直接从 EID type CommAddr 获取地址，不做 IP→EID 转换
    CHK_RET(CommAddrToIpAddress(localEp_.commAddr, locAddr_));
    CHK_RET(CommAddrToIpAddress(remoteEp_.commAddr, rmtAddr_));
    HCCL_INFO(
        "[AicpuTsUbRtpChannel][%s] locAddr_[%s], rmtAddr_[%s]", __func__, locAddr_.Describe().c_str(),
        rmtAddr_.Describe().c_str());

    CHK_RET(BuildSocket());
    CHK_RET(BuildNotify());
    /*
        HccpRaGetDevBaseAttr
        获取urma read/write 单个wr的最大传输数据大小
        调用前,rdmaHandle_要在ParseInputParam中被赋值好,之后BuildConnection会使用获取的属性
        ub_rtp的BuildConnection不再Init里面执行，Init之后会有单独流程建链
    */
    CHK_RET(HccpRaGetDevBaseAttr(rdmaHandle_, &devBaseAttr_));

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUbRtpChannel::BuildConnection()
{
    UbConnBuildContext ctx;
    CHK_RET(PrepareUbConnBuildContext(localEp_, remoteEp_, channelDesc_, ctx));
    CHK_RET(CheckUbSqDepth(ctx, devBaseAttr_));
    CHK_RET(CheckUbScqDepth(ctx, HrtUbJfcMode::STARS_POLL));

    Hccl::OpMode opMode = Hccl::OpMode::OPBASE;
    bool devUsed = true; // aicpu 为 true
    // UB_RTP 的 locAddr_/rmtAddr_ 已经是 EID-based IpAddress，无需额外转换
    HCCL_INFO(
        "[AicpuTsUbRtpChannel][%s] LinkProtocol[%s], locAddr_[%s], rmtAddr_[%s], qos[%u]", __func__,
        ctx.protocol.Describe().c_str(), locAddr_.Describe().c_str(), rmtAddr_.Describe().c_str(),
        static_cast<unsigned int>(ctx.qosPre));

    // UB_RTP 使用 DevUbRtpConnection，locAddr_/rmtAddr_ 作为 EID 地址；qos 与 UBOE 一致来自 channelDesc_
    // UB_RTP 协议对应 HCOMM_TA_RTP_UB_TIMEOUT，由 base_comm 从环境变量获取后传入
    u8 taTimeOut = 0;
    uint32_t taTimeOutValue = 0;
    CHK_RET(hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(taTimeOutValue));
    taTimeOut = static_cast<u8>(taTimeOutValue);
    std::unique_ptr<Hccl::DevUbConnection> ubConn = std::make_unique<Hccl::DevUbRtpConnection>(
        rdmaHandle_, locAddr_, rmtAddr_, opMode, devUsed, Hccl::HrtUbJfcMode::STARS_POLL, locAddr_, rmtAddr_,
        ctx.qosPre, taTimeOut, COMM_ENGINE_AICPU_TS, ctx.sqDepth, ctx.scqDepth);
    CHK_SMART_PTR_NULL(ubConn);

    if (devBaseAttr_.maxReadSize == 0 || devBaseAttr_.maxWriteSize == 0) {
        HCCL_ERROR(
            "[AicpuTsUbRtpChannel][%s] maxReadSize[%u] or maxWriteSize[%u] must not be zero", __func__,
            devBaseAttr_.maxReadSize, devBaseAttr_.maxWriteSize);
        return HCCL_E_PARA;
    }
    ubConn->SetMaxReadSize(devBaseAttr_.maxReadSize);
    ubConn->SetMaxWriteSize(devBaseAttr_.maxWriteSize);
    HCCL_INFO(
        "[AicpuTsUbRtpChannel][%s] maxReadSize[%u], maxWriteSize[%u]", __func__, devBaseAttr_.maxReadSize,
        devBaseAttr_.maxWriteSize);

    commonRes_.connVec.clear();
    commonRes_.connVec.emplace_back(ubConn.get());
    connections_.clear();
    connections_.push_back(std::move(ubConn));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUbRtpChannel::SendFinish()
{
    HCCL_INFO("start send Finish Msg [%s]", UB_RTP_FINISH_MSG);
    sendFinishMsg_ = std::vector<char>(UB_RTP_FINISH_MSG, UB_RTP_FINISH_MSG + FINISH_MSG_SIZE);
    TRY_CATCH_RETURN(socket_->SendAsync(sendFinishMsg_.data(), FINISH_MSG_SIZE););
    HCCL_INFO("end send Finish Msg [%s]", UB_RTP_FINISH_MSG);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUbRtpChannel::RecvFinish()
{
    recvFinishMsg_.resize(FINISH_MSG_SIZE);
    HCCL_INFO("start recv Finish Msg [%s]", UB_RTP_FINISH_MSG);
    TRY_CATCH_RETURN(socket_->RecvAsync(ReinterpretAs<u8*>(recvFinishMsg_.data()), FINISH_MSG_SIZE););
    HCCL_INFO("end recv Finish Msg [%s]", UB_RTP_FINISH_MSG);
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
            ubRtpStatus = isRecvFirst_ ? UbRtpStatus::RECV_DATA : UbRtpStatus::SEND_DATA;
            break;
        case UbRtpStatus::SEND_DATA:
            CHK_RET(SendExchangeData());
            ubRtpStatus = isRecvFirst_ ? UbRtpStatus::PROCESS_DATA : UbRtpStatus::RECV_DATA;
            break;
        case UbRtpStatus::RECV_DATA:
            CHK_RET(RecvExchangeData());
            ubRtpStatus = isRecvFirst_ ? UbRtpStatus::SEND_DATA : UbRtpStatus::PROCESS_DATA;
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
        "[AicpuTsUbRtpChannel][%s] start, channelStatus[%u], ubRtpStatus[%u]", __func__,
        static_cast<u32>(channelStatus), static_cast<u32>(ubRtpStatus));
    commonRes_.connVec.clear();
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
        std::lock_guard<std::mutex> lock(remoteMemsMutex_);
        rmtBufferVec_.clear();
        cacheValid_ = false;
        remoteUserMems_.clear();
        memInfoCopies_.clear();
        memInfoPointers_.clear();
    }

    channelStatus = ChannelStatus::INIT;
    ubRtpStatus = UbRtpStatus::INIT;

    HCCL_INFO("[AicpuTsUbRtpChannel][%s] done.", __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUbRtpChannel::Resume()
{
    HCCL_INFO(
        "[AicpuTsUbRtpChannel][%s] start, channelStatus[%u], ubRtpStatus[%u]", __func__,
        static_cast<u32>(channelStatus), static_cast<u32>(ubRtpStatus));
    channelStatus = ChannelStatus::INIT;
    ubRtpStatus = UbRtpStatus::INIT;
    HCCL_INFO("[AicpuTsUbRtpChannel][%s] done.", __func__);
    return HCCL_SUCCESS;
}

} // namespace hcomm
