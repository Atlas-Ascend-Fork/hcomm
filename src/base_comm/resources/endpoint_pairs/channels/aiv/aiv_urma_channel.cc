/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_urma_channel.h"
#include "endpoint.h"
#include "../sockets/socket_mgr.h"
#include "base_comm_node_accessors.h"
#include "comm_queue_context/jetty_context.h"
#include "orion_adpt_utils.h"
#include "shared_jetty_channel_helper.h"

#include "hcomm_c_adpt.h"
#include "config_plf_log_v2.h"

// Orion
#include "topo_common_types.h"
#include "virtual_topo.h"
#include "orion_adapter_hccp.h"
#include "adapter_rts.h"
#include "acl/acl_rt.h"

#include <cstdint>
#include <cstdlib>

namespace hcomm {
using Hccl::PLF_CHANNEL;

AivUrmaChannel::AivUrmaChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc, CommEngine engine)
    : Channel(endpointHandle, channelDesc, engine, HcommChannelKind::AIV_URMA, "AivUrmaChannel")
{}

AivUrmaChannel::~AivUrmaChannel() { aivChannelHelper_.ReleaseDeviceEntitySlab(); }

HcclResult AivUrmaChannel::ParseInputParam()
{
    CHK_RET(Channel::ParseInputParam());

    CHK_RET(CollectLocalRmaBuffers(local_.commonRes.bufferVec));

    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::StartListen()
{
    CHK_RET(Channel::StartListen());
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::BuildSocket()
{
    CHK_RET(Channel::BuildSocket());
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::BuildConnection()
{
    CHK_RET(Channel::BuildConnection());
    CHK_RET(BuildAttr());
    CHK_RET(HccpRaGetDevBaseAttr(local_.rdmaHandle, &devBaseAttr_));

    UbConnBuildContext ctx;
    CHK_RET(PrepareUbConnBuildContext(local_.ep, remote_.ep, channelDesc_, ctx));
    CHK_RET(CheckUbSqDepth(ctx, devBaseAttr_));
    CHK_RET(CheckUbScqDepth(ctx, HrtUbJfcMode::USER_CTL));

    /* 需求边界：scqDepth 仅在独占 JFC 场景（CreateUbConnectionByProtocol 走 SELF_CREATE）生效；
     * 共享 jetty 模式（EXTERNAL_INJECT）复用同 Endpoint 已创建的 jetty，不创建独占 JFC，
     * scqDepth 的"独占创建"语义不适用，深度由 SetSharedJettyFields 的 sDepth 统一决定。
     * 但 BuildConnection 统一从 ctx 取 scqDepth 并统一经 CheckUbScqDepth 校验（无效值一律拒绝），
     * 有效值在共享 jetty 下被静默忽略，故此处显式告警提示用户，避免"校验通过却不生效"的困惑。 */
    if (IsSharedQueue() && ctx.scqDepth != UB_SCQ_DEPTH_NOT_SET && ctx.scqDepth != 0) {
        HCCL_WARNING(
            "[%s::%s] scqDepth[%u] is not effective in shared jetty mode, "
            "depth is determined by SetSharedJettyFields.",
            channelClassName_, __func__, ctx.scqDepth);
    }

    // 共享 jetty 模式：主 connection 构造时传 EXTERNAL_INJECT，跳过建 JFC/jetty，等 SetSharedJettyFields 填充
    auto jettyMode = IsSharedQueue() ? Hccl::DevUbConnection::JettyMode::EXTERNAL_INJECT :
                                       Hccl::DevUbConnection::JettyMode::SELF_CREATE;
    std::unique_ptr<Hccl::DevUbConnection> ubConn = nullptr;
    CHK_RET(CreateUbConnectionByProtocol(ctx, ubConn, jettyMode));
    CHK_SMART_PTR_NULL(ubConn);

    // 共享 jetty 模式：复用同 Endpoint 下已创建的 jetty。
    // 必须在 push_back(move(ubConn)) 之前调用：AcquireSharedJetty 失败时 ubConn 仍为局部变量，
    // 函数返回时自动析构，不会在 connections_/connVec 中残留不完整 connection。
    if (IsSharedQueue()) {
        CHK_RET(AcquireSharedJettyInBuildConnection(ctx, ubConn.get()));
    }

    local_.commonRes.connVec.clear();
    connections_.clear();
    local_.commonRes.connVec.emplace_back(ubConn.get());
    connections_.push_back(std::move(ubConn));
    PLF_CONFIG_INFO(
        PLF_CHANNEL, "[%s::%s] build DevUbConnection, protocol[%s] sqDepth[%u] sharedJetty[%d].", channelClassName_,
        __func__, ctx.protocol.Describe().c_str(), ctx.sqDepth, static_cast<int>(IsSharedQueue()));

    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::BuildNotify()
{
    CHK_RET(Channel::BuildNotify());
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::BuildBuffer()
{
    CHK_RET(Channel::BuildBuffer());
    const Hccl::Socket& socket = *local_.socket;

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(local_.ep, remote_.ep, linkData));

    // make_unique / make_shared / release 包一层抛异常的宏
    EXCEPTION_CATCH(
        transport_ = std::make_unique<Hccl::AivUrmaTransport>(
            local_.commonRes, local_.attr, linkData, socket, local_.rdmaHandle), // 这里区分是否是优先recv
        return HCCL_E_PTR);
    PLF_CONFIG_INFO(
        PLF_CHANNEL, "[%s::%s] create AivUrmaTransport, socket[%s], linkData[%s].", channelClassName_, __func__,
        local_.socket->Describe().c_str(), linkData.Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::GetNotifyNum([[maybe_unused]] uint32_t* notifyNum) const
{
    HCCL_INFO("[%s::%s] AivUrmaChannel GetNotifyNum is not supported.", channelClassName_, __func__);
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    return transport_->GetRemoteMems(memNum, remoteMem, memInfos);
}

ChannelStatus AivUrmaChannel::GetStatus()
{
    Hccl::TransportStatus transportStatus = transport_->GetStatus();
    ChannelStatus out = ChannelStatus::INIT;
    switch (transportStatus) {
        case Hccl::TransportStatus::INIT:
            out = ChannelStatus::INIT;
            break;
        case Hccl::TransportStatus::SOCKET_OK:
            out = ChannelStatus::SOCKET_OK;
            break;
        case Hccl::TransportStatus::SOCKET_TIMEOUT:
            out = ChannelStatus::SOCKET_TIMEOUT;
            break;
        case Hccl::TransportStatus::READY:
            out = ChannelStatus::READY;
            break;
        default:
            HCCL_ERROR("[%s::%s] Invalid TransportStatus[%d]", channelClassName_, __func__, transportStatus);
            out = ChannelStatus::FAILED;
            break;
    }
    return out;
}

HcclResult AivUrmaChannel::UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum)
{
    CHK_PTR_NULL(transport_.get());
    return transport_->UpdateMemInfo(memHandles, memHandleNum);
}

HcclResult AivUrmaChannel::Clean()
{
    PLF_CONFIG_INFO(PLF_CHANNEL, "[%s::%s] clean channel resource.", channelClassName_, __func__);
    aivChannelHelper_.ReleaseDeviceEntitySlab();
    transport_.reset();
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::Resume()
{
    CHK_RET(BuildConnection());
    CHK_RET(BuildBuffer());
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::BuildAttr()
{
    local_.attr.devicePhyId = local_.ep.loc.device.devPhyId;
    local_.attr.opMode = Hccl::OpMode::OPBASE;
    local_.attr.opAcceState = Hccl::AcceleratorState::AIV;
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::CreateUbConnectionByProtocol(
    const UbConnBuildContext& ctx, std::unique_ptr<Hccl::DevUbConnection>& ubConn,
    Hccl::DevUbConnection::JettyMode jettyMode)
{
    Hccl::OpMode opMode = Hccl::OpMode::OPBASE;
    bool devUsed = true;
    Hccl::HrtUbJfcMode jfcMode = Hccl::HrtUbJfcMode::USER_CTL;
    // UB_CTP → HCOMM_TA_CTP_UB_TIMEOUT，UB_TP/UB_RTP → HCOMM_TA_RTP_UB_TIMEOUT
    u8 taTimeOut = 0;
    uint32_t taTimeOutValue = 0;
    if (ctx.protocol == Hccl::LinkProtocol::UB_CTP) {
        CHK_RET(hcomm::GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(taTimeOutValue));
    } else {
        CHK_RET(hcomm::GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(taTimeOutValue));
    }
    taTimeOut = static_cast<u8>(taTimeOutValue);
    switch (ctx.protocol) {
        case Hccl::LinkProtocol::UB_TP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::DevUbTpConnection>(
                    local_.rdmaHandle, ctx.locAddr, ctx.rmtAddr, opMode, devUsed, jfcMode, Hccl::IpAddress(),
                    Hccl::IpAddress(), ctx.qosPre, taTimeOut, COMM_ENGINE_AIV, ctx.sqDepth, ctx.scqDepth, jettyMode),
                return HCCL_E_PTR);
            break;
        case Hccl::LinkProtocol::UB_CTP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::DevUbCtpConnection>(
                    local_.rdmaHandle, ctx.locAddr, ctx.rmtAddr, opMode, devUsed, jfcMode, Hccl::IpAddress(),
                    Hccl::IpAddress(), ctx.qosPre, taTimeOut, COMM_ENGINE_AIV, ctx.sqDepth, ctx.scqDepth, jettyMode),
                return HCCL_E_PTR);
            break;
        case Hccl::LinkProtocol::UB_RTP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::DevUbRtpConnection>(
                    local_.rdmaHandle, ctx.locAddr, ctx.rmtAddr, opMode, devUsed, jfcMode, ctx.locAddr, ctx.rmtAddr,
                    ctx.qosPre, taTimeOut, COMM_ENGINE_AIV, ctx.sqDepth, ctx.scqDepth, jettyMode),
                return HCCL_E_PTR);
            break;
        default:
            HCCL_ERROR("[%s::%s] No LinkProtocol to match", channelClassName_, __func__);
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult
AivUrmaChannel::AcquireSharedJettyInBuildConnection(const UbConnBuildContext& ctx, Hccl::DevUbConnection* connection)
{
    // 共享 jetty 模式：复用同 Endpoint 下已创建的 jetty
    Endpoint* endpoint = static_cast<Endpoint*>(endpointHandle_);
    // UB_CTP → HCOMM_TA_CTP_UB_TIMEOUT，UB_TP/UB_RTP → HCOMM_TA_RTP_UB_TIMEOUT
    u8 taTimeOut = 0;
    uint32_t taTimeOutValue = 0;
    if (ctx.protocol == Hccl::LinkProtocol::UB_CTP) {
        CHK_RET(hcomm::GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(taTimeOutValue));
    } else {
        CHK_RET(hcomm::GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(taTimeOutValue));
    }
    taTimeOut = static_cast<u8>(taTimeOutValue);
    auto tempFactory = [rdmaHandle = local_.rdmaHandle, &ctxLoc = ctx.locAddr, &ctxRmt = ctx.rmtAddr,
                        qosPre = ctx.qosPre, protocol = ctx.protocol, sqDepth = ctx.sqDepth,
                        taTimeOut = taTimeOut]() -> std::unique_ptr<Hccl::DevUbConnection> {
        return CreateSharedJettyConnection(rdmaHandle, ctxLoc, ctxRmt, qosPre, protocol, taTimeOut, sqDepth);
    };
    JettyContext::Ctx sharedCtx{};
    CHK_RET(hcomm::AcquireSharedJettyForChannel(endpoint, connection, tempFactory, sharedCtx));
    // 保存共享 PI/CI 指针，供 BuildChannelEntityToDevice 绑给 transport
    sharedSqPiPtr_ = sharedCtx.sqPiPtr;
    sharedSqCiPtr_ = sharedCtx.sqCiPtr;
    sharedCqPiPtr_ = sharedCtx.cqPiPtr;
    sharedCqCiPtr_ = sharedCtx.cqCiPtr;
    return HCCL_SUCCESS;
}

std::unique_ptr<Hccl::DevUbConnection> AivUrmaChannel::CreateSharedJettyConnection(
    Hccl::RdmaHandle rdmaHandle, const Hccl::IpAddress& locAddr, const Hccl::IpAddress& rmtAddr, u8 qosPre,
    Hccl::LinkProtocol protocol, u8 taTimeOut, u32 sqDepth)
{
    switch (protocol) {
        case Hccl::LinkProtocol::UB_TP: {
            std::unique_ptr<Hccl::DevUbConnection> conn;
            EXCEPTION_CATCH(
                conn = std::make_unique<Hccl::DevUbTpConnection>(
                    rdmaHandle, locAddr, rmtAddr, Hccl::OpMode::OPBASE, true, Hccl::HrtUbJfcMode::USER_CTL,
                    Hccl::IpAddress(), Hccl::IpAddress(), qosPre, taTimeOut, COMM_ENGINE_AIV, sqDepth),
                return nullptr);
            return conn;
        }
        case Hccl::LinkProtocol::UB_CTP: {
            std::unique_ptr<Hccl::DevUbConnection> conn;
            EXCEPTION_CATCH(
                conn = std::make_unique<Hccl::DevUbCtpConnection>(
                    rdmaHandle, locAddr, rmtAddr, Hccl::OpMode::OPBASE, true, Hccl::HrtUbJfcMode::USER_CTL,
                    Hccl::IpAddress(), Hccl::IpAddress(), qosPre, taTimeOut, COMM_ENGINE_AIV, sqDepth),
                return nullptr);
            return conn;
        }
        case Hccl::LinkProtocol::UB_RTP: {
            std::unique_ptr<Hccl::DevUbConnection> conn;
            EXCEPTION_CATCH(
                conn = std::make_unique<Hccl::DevUbRtpConnection>(
                    rdmaHandle, locAddr, rmtAddr, Hccl::OpMode::OPBASE, true, Hccl::HrtUbJfcMode::USER_CTL, locAddr,
                    rmtAddr, qosPre, taTimeOut, COMM_ENGINE_AIV, sqDepth),
                return nullptr);
            return conn;
        }
        default:
            HCCL_ERROR(
                "[AivUrmaChannel::%s] unsupported protocol[%s], return nullptr.", __func__,
                protocol.Describe().c_str());
            return nullptr;
    }
}

HcclResult AivUrmaChannel::BuildChannelEntityToDevice(void** devChannelPtr)
{
    return GetChannelEntity()->BuildChannelEntityToDevice(devChannelPtr);
}

} // namespace hcomm
