/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/cast_utils_pub.h"
#include "aicpu_ts_urma_channel.h"
#include "endpoint.h"
#include "../../sockets/socket_mgr.h"
#include "base_comm_node_accessors.h"
#include "orion_adpt_utils.h"
#include "hcomm_c_adpt.h"
#include "config_log.h"
#include "config_plf_log_v2.h"
#include "endpoint.h"

// Orion
#include "adapter_rts_common.h"
#include "topo_common_types.h"
#include "virtual_topo.h"
#include "aicpu_res_package_helper.h"
#include "makebufs_helper.h"

namespace hcomm {
using Hccl::PLF_CHANNEL;

HcclResult SetModuleDataName(Hccl::ModuleData& module, const std::string& name)
{
    int ret = strcpy_s(module.name, sizeof(module.name), name.c_str());
    if (ret != 0) {
        HCCL_ERROR("[AicpuTsUrmaChannel::%s] strcpy_s name %s failed", __func__, name.c_str());
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

AicpuTsUrmaChannel::AicpuTsUrmaChannel(
    EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc, CommEngine engine)
    : Channel(endpointHandle, channelDesc, engine, HcommChannelKind::AICPU_TS_URMA, "AicpuTsUrmaChannel")
{}

AicpuTsUrmaChannel::~AicpuTsUrmaChannel() {}

HcclResult AicpuTsUrmaChannel::ParseInputParam()
{
    CHK_RET(Channel::ParseInputParam());

    CHK_RET(CollectLocalRmaBuffers(local_.commonRes.bufferVec));

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::StartListen()
{
    CHK_RET(Channel::StartListen());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildSocket()
{
    CHK_RET(Channel::BuildSocket());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildConnection()
{
    CHK_RET(Channel::BuildConnection());
    if (!devBaseAttrReady_) {
        CHK_RET(BuildAttr());
        /*
            HccpRaGetDevBaseAttr
            获取urma read/write 单个wr的最大传输数据大小
            调用前,rdmaHandle_要在ParseInputParam中被赋值好,之后BuildConnection会使用获取的属性
        */
        CHK_RET(HccpRaGetDevBaseAttr(local_.rdmaHandle, &devBaseAttr_));
        devBaseAttrReady_ = true;
    }

    UbConnBuildContext ctx;
    CHK_RET(PrepareUbConnBuildContext(local_.ep, remote_.ep, channelDesc_, ctx));
    CHK_RET(CheckUbSqDepth(ctx, devBaseAttr_));
    CHK_RET(CheckUbScqDepth(ctx, HrtUbJfcMode::STARS_POLL));

    Hccl::OpMode opMode = Hccl::OpMode::OPBASE;
    bool devUsed = true; // aicpu 为 true
    // UB_CTP → HCOMM_TA_CTP_UB_TIMEOUT，UB_TP → HCOMM_TA_RTP_UB_TIMEOUT
    u8 taTimeOut = 0;
    uint32_t taTimeOutValue = 0;
    if (ctx.protocol == Hccl::LinkProtocol::UB_CTP) {
        CHK_RET(hcomm::GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(taTimeOutValue));
    } else {
        CHK_RET(hcomm::GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(taTimeOutValue));
    }
    taTimeOut = static_cast<u8>(taTimeOutValue);
    std::unique_ptr<Hccl::DevUbConnection> ubConn = nullptr;
    switch (ctx.protocol) {
        case Hccl::LinkProtocol::UB_TP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::DevUbTpConnection>(
                    local_.rdmaHandle, ctx.locAddr, ctx.rmtAddr, opMode, devUsed, Hccl::HrtUbJfcMode::STARS_POLL,
                    Hccl::IpAddress(), Hccl::IpAddress(), ctx.qosPre, taTimeOut, COMM_ENGINE_AICPU_TS, ctx.sqDepth,
                    ctx.scqDepth),
                return HCCL_E_PTR);
            break;
        case Hccl::LinkProtocol::UB_CTP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::DevUbCtpConnection>(
                    local_.rdmaHandle, ctx.locAddr, ctx.rmtAddr, opMode, devUsed, Hccl::HrtUbJfcMode::STARS_POLL,
                    Hccl::IpAddress(), Hccl::IpAddress(), ctx.qosPre, taTimeOut, COMM_ENGINE_AICPU_TS, ctx.sqDepth,
                    ctx.scqDepth),
                return HCCL_E_PTR);
            break;
        default:
            HCCL_ERROR("[%s::%s] No LinkProtocol to match", channelClassName_, __func__);
            break;
    }
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
    PLF_CONFIG_INFO(
        PLF_CHANNEL, "[%s::%s] build DevUbConnection, protocol[%s] sqDepth[%u].", channelClassName_, __func__,
        ctx.protocol.Describe().c_str(), ctx.sqDepth);
    return HCCL_SUCCESS;
}

std::vector<char> AicpuTsUrmaChannel::GetMemUpdatePack() { return memTransport_->GetMemUpdatePack(); }

HcclResult AicpuTsUrmaChannel::BuildNotify()
{
    CHK_RET(Channel::BuildNotify());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildBuffer()
{
    CHK_RET(Channel::BuildBuffer());
    Hccl::BaseMemTransport::LocCntNotifyRes locCntNotifyRes{};
    locCntNotifyRes.vec.clear();
    locCntNotifyRes.desc.clear();
    const Hccl::Socket& socket = *local_.socket;

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(local_.ep, remote_.ep, linkData));

    bool isRecvFirst = socket.GetRole() == Hccl::SocketRole::CLIENT ? true : false;

    // make_unique / make_shared / release 包一层抛异常的宏
    EXCEPTION_CATCH(
        memTransport_ = std::make_unique<Hccl::UbMemTransport>(
            local_.commonRes, local_.attr, linkData, socket, local_.rdmaHandle, locCntNotifyRes, isRecvFirst),
        return HCCL_E_PTR);
    PLF_CONFIG_INFO(
        PLF_CHANNEL, "[%s::%s] create UbMemTransport, socket[%s], linkData[%s].", channelClassName_, __func__,
        local_.socket->Describe().c_str(), linkData.Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::GetNotifyNum(uint32_t* notifyNum) const
{
    *notifyNum = this->local_.notifyNum;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    return memTransport_->GetRemoteMems(memNum, remoteMem, memInfos);
}

ChannelStatus AicpuTsUrmaChannel::GetStatus()
{
    ChannelStatus out = TransportStatusToChannelStatus(memTransport_->GetStatus(), local_.ep, GetChannelDesc());

    if (isFirstPrintChannelInfo_ && out == ChannelStatus::READY) {
        std::string channelInfo = "create channel info:channel handle[";
        channelInfo.append(std::to_string(ReinterpretAs<uint64_t>(this)));
        channelInfo.append("] ");
        HcclResult ret = memTransport_->Describe(channelInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[%s::%s] Describe channel info failed, ret=%d", channelClassName_, __func__, ret);
            out = ChannelStatus::FAILED;
        } else {
            channelInfo.append(" TA[RM]"); // 目前TA只支持RM
            HCCL_CONFIG_DEBUG(hccl::HCCL_RES, "%s", channelInfo.c_str());
        }
        isFirstPrintChannelInfo_ = false;
    }
    return out;
}

HcclResult AicpuTsUrmaChannel::UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum)
{
    std::vector<Hccl::LocalRmaBuffer*> bufferVecTemp;
    CHK_RET(MakeRmaBufferVecFromMemHandles(memHandles, memHandleNum, bufferVecTemp, "AicpuTsUrmaChannel"));
    CHK_RET(memTransport_->UpdateMemInfo(bufferVecTemp));
    local_.commonRes.bufferVec.insert(local_.commonRes.bufferVec.end(), bufferVecTemp.begin(), bufferVecTemp.end());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::Clean()
{
    PLF_CONFIG_INFO(PLF_CHANNEL, "[%s::%s] clean channel resource.", channelClassName_, __func__);
    memTransport_.reset();
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::Resume()
{
    CHK_RET(BuildSocket());
    CHK_RET(BuildConnection());
    CHK_RET(ResetLocalNotifies());
    CHK_RET(BuildBuffer());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::H2DResPack(std::vector<char>& buffer)
{
    CHK_RET(PackOpData(buffer));
    HCCL_INFO(
        "[%s::%s] Pack Buffer data[%p], Pack Buffer size[%zu].", channelClassName_, __func__, buffer.data(),
        buffer.size());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildAttr()
{
    local_.attr.devicePhyId = local_.ep.loc.device.devPhyId;
    local_.attr.opMode = Hccl::OpMode::OPBASE;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::PackOpData(std::vector<char>& data)
{
    std::vector<Hccl::ModuleData> dataVec;
    dataVec.resize(Hccl::AicpuResMgrType::__COUNT__);

    Hccl::AicpuResMgrType resType = Hccl::AicpuResMgrType::STREAM;
    CHK_RET(SetModuleDataName(dataVec[resType], "UbMemTransport"));

    std::vector<char> result;
    Hccl::BinaryStream binaryStream;
    std::vector<char> uniqueIdV2;
    CHK_RET(memTransport_->GetUniqueIdV2(uniqueIdV2));
    binaryStream << uniqueIdV2;

    binaryStream.Dump(result);

    dataVec[resType].data = result;

    Hccl::AicpuResPackageHelper helper;
    data = helper.GetPackedData(dataVec);

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::ResetLocalNotifies()
{
    for (size_t i = 0; i < local_.notifies.size(); ++i) {
        if (local_.notifies[i] == nullptr) {
            continue;
        }
        Hccl::RtsNotify* rtsNotify = local_.notifies[i]->GetNotify();
        if (rtsNotify == nullptr) {
            continue;
        }
        HcclRtNotify rtNotify = ReinterpretAs<HcclRtNotify>(rtsNotify->GetHandleAddr());
        if (rtNotify == nullptr) {
            continue;
        }
        HcclResult ret = hrtNotifyReset(rtNotify);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s::%s] hrtNotifyReset failed, channel[%p], notifyIdx[%zu], "
                "notifyId[%u], ret[0x%016llx]",
                channelClassName_, __func__, this, i, rtsNotify->GetId(), HCCL_ERROR_CODE(ret)),
            ret);
        HCCL_INFO(
            "[%s::%s] reset notify success, channel[%p], notifyIdx[%zu], "
            "notifyId[%u], notifyPtr[%p]",
            channelClassName_, __func__, this, i, rtsNotify->GetId(), rtNotify);
    }
    return HCCL_SUCCESS;
}

} // namespace hcomm
