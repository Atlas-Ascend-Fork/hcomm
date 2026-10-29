/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>

#include "log.h"
#include "channel.h"
#include "config_plf_log_v2.h"
#include "common/orion_adpt_utils_pub.h"
#include "orion_adpt_utils.h"
#include "hcomm_c_adpt.h"
#include "endpoint.h"
#include "adapter_rts_common.h"
#include "../../endpoints/net_dev/global_net_dev_manager.h"
#include "channel_config.h"
#include "common/cast_utils_pub.h"
#include "makebufs_helper.h"
#include "exception_handler.h"
#include "./aicpu/aicpu_ts_urma_channel.h"
#include "common/comm_engine_utils_pub.h"
#include "protocol_utils.h"

#include "./aicpu/aicpu_ts_p2p_channel.h"
#include "./aicpu/aicpu_ts_uboe_channel.h"
#include "./aicpu/aicpu_ts_ub_rtp_channel.h"
#include "./aicpu/aicpu_ts_roce_dev_loc_channel.h"
#include "./host/host_cpu_roce_channel.h"
#include "./host/host_cpu_urma_channel.h"
#include "./ccu/ccu_urma_channel.h"
#include "./aiv/aiv_ub_mem_channel.h"
#include "./aiv/aiv_urma_channel.h"
#include "./aicpu/aicpu_ts_hccs_channel.h"
#include "./aicpu/roce_host_loc/aicpu_ts_roce_host_loc_channel.h"
#include "./aiv/aiv_roce_host_loc_channel.h"
#include "common/dev_type_utils_pub.h"
#include "../sockets/socket_mgr.h"
#include "../../../../../src/legacy/ascend950/unified_platform/resource/notify/ub_local_notify.h"
#include "../../../../../src/legacy/ascend950/unified_platform/resource/notify/ipc_local_notify.h"
#include "base_comm_node_accessors.h"

namespace hcomm {
using Hccl::PLF_CHANNEL;

Channel::Channel(
    EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc, CommEngine engine, HcommChannelKind channelKind,
    const char* channelClassName)
    : channelKind_(channelKind),
      engine_(engine),
      endpointHandle_(endpointHandle),
      channelDesc_(channelDesc),
      channelClassName_(channelClassName)
{
    switch (channelKind_) {
        case HcommChannelKind::AICPU_TS_URMA:
            traits_.socketBuildMode = (channelDesc_.role == HCOMM_SOCKET_ROLE_RESERVED) ?
                                          SocketBuildMode::LISTEN_AND_BORROW :
                                          SocketBuildMode::BORROW;
            traits_.listenPortType = Hccl::ConnectProtoType::UB;
            traits_.notifyBuildMode = NotifyBuildMode::BUILD_LOCAL;
            traits_.notifyType = LocalNotifyType::UB;
            break;
        case HcommChannelKind::AICPU_TS_HCCS:
            traits_.ipMode = EndpointIpMode::FROM_VNIC;
            traits_.defaultListenPort = HCCS_DEFAULT_LISTENING_PORT;
            break;
        case HcommChannelKind::CPU_ROCE:
        case HcommChannelKind::AICPU_TS_ROCE_HOST_LOC:
        case HcommChannelKind::AIV_ROCE:
            traits_.socketBuildMode = SocketBuildMode::BORROW;
            break;
        case HcommChannelKind::AICPU_TS_ROCE_DEV_LOC:
            traits_.ipMode = EndpointIpMode::FROM_COMM_ADDR;
            traits_.defaultListenPort = ROCE_V1_DEFAULT_LISTENING_PORT;
            break;
        case HcommChannelKind::AIV_URMA:
            traits_.socketBuildMode = SocketBuildMode::BORROW;
            traits_.reservedNoRankId = true;
            break;
        case HcommChannelKind::CPU_URMA:
            traits_.socketBuildMode = SocketBuildMode::BORROW;
            traits_.reservedSocketAddrTag = true;
            traits_.memFromDescAllowed = false;
            break;
        case HcommChannelKind::AICPU_TS_UBOE:
            traits_.socketBuildMode = SocketBuildMode::LISTEN_AND_BORROW;
            traits_.listenPortType = Hccl::ConnectProtoType::UB;
            traits_.notifyBuildMode = NotifyBuildMode::BUILD_LOCAL;
            traits_.notifyType = LocalNotifyType::UB;
            break;
        case HcommChannelKind::AICPU_TS_UB_RTP:
            traits_.ipMode = EndpointIpMode::FROM_COMM_ADDR;
            traits_.socketBuildMode = SocketBuildMode::LISTEN_AND_BORROW;
            traits_.listenPortType = Hccl::ConnectProtoType::UB;
            traits_.notifyBuildMode = NotifyBuildMode::BUILD_LOCAL;
            traits_.notifyType = LocalNotifyType::UB;
            break;
        case HcommChannelKind::AICPU_TS_PCIE:
            traits_.socketBuildMode = SocketBuildMode::LISTEN_AND_BORROW;
            traits_.notifyBuildMode = NotifyBuildMode::BUILD_LOCAL;
            traits_.notifyType = LocalNotifyType::IPC;
            break;
        case HcommChannelKind::AIV_UB_MEM:
            traits_.socketBuildMode = SocketBuildMode::BORROW;
            traits_.reservedNoRankId = true;
            break;
        case HcommChannelKind::CCU_URMA:
        case HcommChannelKind::INVALID:
        default:
            break;
    }
}

Channel::~Channel() { PutSocketIfNeeded(); }

HcclResult Channel::Init()
{
    EXCEPTION_HANDLE_BEGIN
    HCCL_INFO("[%s::%s] start init", channelClassName_, __func__);

    s32 devLogicId;
    CHK_RET(hrtGetDevice(&devLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(devLogicId), local_.devicePhyId));

    CHK_RET(ParseInputParam());
    CHK_RET(StartListen());
    CHK_RET(BuildSocket());
    CHK_RET(BuildConnection());
    CHK_RET(BuildNotify());
    CHK_RET(BuildBuffer());
    EXCEPTION_HANDLE_END
    HCCL_INFO("[%s::%s] finish init", channelClassName_, __func__);
    return HCCL_SUCCESS;
}

HcclResult Channel::ParseInputParam()
{
    CHK_PTR_NULL(endpointHandle_);

    local_.epPtr = static_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(local_.epPtr);

    local_.ep = local_.epPtr->GetEndpointDesc();
    remote_.ep = channelDesc_.remoteEndpoint;
    local_.notifyNum = channelDesc_.notifyNum;
    local_.serverPort = channelDesc_.port != 0 ? channelDesc_.port : traits_.defaultListenPort;
    if (channelDesc_.port == 0) {
        HCCL_INFO(
            "[%s::%s] channelDesc port is 0, use default port [%u]", channelClassName_, __func__, local_.serverPort);
    }

    local_.socket = reinterpret_cast<Hccl::Socket*>(channelDesc_.socket);
    local_.rdmaHandle = local_.epPtr->GetRdmaHandle();

    CHK_RET(GetEndpointIp(local_.ep, local_.ip));
    CHK_RET(GetEndpointIp(remote_.ep, remote_.ip));

    HCCL_INFO("[%s::%s] localProtocol[%d]", channelClassName_, __func__, local_.ep.protocol);

    HCCL_INFO(
        "[%s::%s] local devPhyId [%u] ip[%s] remote devPhyId[%u] ip[%s], "
        "serverPort[%u]",
        channelClassName_, __func__, local_.ep.loc.device.devPhyId, local_.ip.GetIpStr().c_str(),
        remote_.ep.loc.device.devPhyId, remote_.ip.GetIpStr().c_str(), local_.serverPort);

    CHK_RET(DecideSocketRole());
    if (traits_.socketBuildMode == SocketBuildMode::LISTEN_AND_BORROW) {
        local_.isSocketServer = false;
    }

    return HCCL_SUCCESS;
}

HcclResult Channel::DecideSocketRole()
{
    if (channelDesc_.role == HCOMM_SOCKET_ROLE_SERVER) {
        local_.isSocketServer = true;
    } else if (channelDesc_.role == HCOMM_SOCKET_ROLE_CLIENT) {
        local_.isSocketServer = false;
    }
    return HCCL_SUCCESS;
}

HcclResult Channel::CollectLocalRmaBuffers(std::vector<Hccl::LocalRmaBuffer*>& bufferVec)
{
    bufferVec.clear();
    if (channelDesc_.exchangeAllMems) {
        HCCL_INFO("[%s::%s] exchangeAllMems == True. Get memHandles from endpoint.", channelClassName_, __func__);
        std::shared_ptr<Hccl::LocalRmaBuffer>* memHandles = nullptr;
        uint32_t memHandleNum = 0;
        CHK_RET(static_cast<HcclResult>(
            HcommMemGetAllMemHandles(endpointHandle_, ReinterpretAs<void**>(&memHandles), &memHandleNum)));
        HCCL_INFO("[%s::%s] Got memHandleNum[%u].", channelClassName_, __func__, memHandleNum);
        for (uint32_t i = 0; i < memHandleNum; ++i) {
            std::shared_ptr<Hccl::LocalRmaBuffer>& localRmaBuffer = memHandles[i];
            CHK_SMART_PTR_NULL(localRmaBuffer);
            auto buf = localRmaBuffer->GetBuf();
            CHK_PTR_NULL(buf);
            HCCL_INFO(
                "[%s::%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], memType[%d], memInfo[%s].",
                channelClassName_, __func__, i, static_cast<unsigned long long>(localRmaBuffer->GetAddr()),
                static_cast<unsigned long long>(localRmaBuffer->GetSize()), static_cast<int>(buf->GetMemType()),
                buf->GetMemInfo().c_str());
            bufferVec.push_back(localRmaBuffer.get());
        }
    } else if (traits_.memFromDescAllowed) {
        HCCL_INFO("[%s::%s] exchangeAllMems == false. Get memHandles from channelDesc.", channelClassName_, __func__);
        CHK_RET(MakeRmaBufferVecFromMemHandles(
            channelDesc_.memHandles, channelDesc_.memHandleNum, bufferVec, channelClassName_));
    } else {
        HCCL_WARNING("[%s::%s] exchangeAllMems is false.", channelClassName_, __func__);
    }
    return HCCL_SUCCESS;
}

HcclResult Channel::GetEndpointIp(const EndpointDesc& ep, Hccl::IpAddress& ip)
{
    switch (traits_.ipMode) {
        case EndpointIpMode::FROM_VNIC:
            return GetFirstIpByPhyId(ep.loc.device.devPhyId, ep.loc.device.superDevId, ip);
        case EndpointIpMode::FROM_COMM_ADDR:
            return CommAddrToIpAddress(ep.commAddr, ip);
        case EndpointIpMode::NONE:
        default:
            return HCCL_SUCCESS;
    }
}

HcclResult Channel::GetFirstIpByPhyId(uint32_t devicePhyId, uint32_t superDevId, Hccl::IpAddress& ip) const
{
    hccl::HcclIpAddress vnicIp{};
    CHK_RET(hccl::GlobalNetDevMgr::GetDeviceVnicIP(devicePhyId, superDevId, vnicIp));
    ip = HcclIpAddressToIpAddress(vnicIp);
    HCCL_INFO(
        "[%s::%s] devicePhyId[%u] superDevId[%u] vnicIp[%s]", channelClassName_, __func__, devicePhyId, superDevId,
        ip.GetIpStr().c_str());
    return HCCL_SUCCESS;
}

HcclResult Channel::StartListen()
{
    if (local_.isSocketServer == true) {
        HCCL_INFO(
            "[%s::%s] Start. EndpointHandle[%p], port[%u]", channelClassName_, __func__, endpointHandle_,
            local_.serverPort);

        CHK_RET(static_cast<HcclResult>(HcommEndpointStartListen(endpointHandle_, local_.serverPort, nullptr)));
        HCCL_INFO("[%s::%s] SUCCESS. port[%u].", channelClassName_, __func__, local_.serverPort);
    }

    if (local_.socket == nullptr && traits_.socketBuildMode == SocketBuildMode::LISTEN_AND_BORROW) {
        Hccl::IpAddress ipaddr{};
        CHK_RET(CommAddrToIpAddress(local_.ep.commAddr, ipaddr));
        Hccl::PortData localPort = Hccl::PortData(
            static_cast<Hccl::RankId>(local_.ep.loc.device.devPhyId), Hccl::DevNetPortType(traits_.listenPortType), 0,
            ipaddr);
        Hccl::SocketHandle socketHandle
            = Hccl::SocketHandleManager::GetInstance().Create(local_.ep.loc.device.devPhyId, localPort);
        EXCEPTION_CATCH(
            local_.serverSocket = std::make_unique<Hccl::Socket>(
                socketHandle, ipaddr, traits_.defaultListenPort, ipaddr, "server", Hccl::SocketRole::SERVER,
                Hccl::NicType::DEVICE_NIC_TYPE),
            return HCCL_E_PARA);
        HCCL_INFO(
            "[%s::%s] listen_socket_info[%s]", channelClassName_, __func__, local_.serverSocket->Describe().c_str());
        EXCEPTION_CATCH(local_.serverSocket->Listen(), return HCCL_E_INTERNAL);
    }

    return HCCL_SUCCESS;
}

HcclResult Channel::BuildSocket()
{
    if (local_.socket != nullptr) {
        return HCCL_SUCCESS;
    }
    if (traits_.socketBuildMode == SocketBuildMode::NONE) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[%s::%s] socket ptr is NULL, rebuildSocket", channelClassName_, __func__);

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(local_.ep, remote_.ep, linkData));
    HCCL_INFO("[%s::%s] built linkData: %s", channelClassName_, __func__, linkData.Describe().c_str());
    std::string socketTag
        = (channelDesc_.channelName != nullptr) ? std::string(channelDesc_.channelName) : "AUTOMATIC_SOCKET_TAG";

    if (traits_.socketBuildMode == SocketBuildMode::LISTEN_AND_BORROW) {
        local_.SaveSocketConfig(Hccl::SocketConfig(linkData, socketTag, true));
    } else if (channelDesc_.role != HCOMM_SOCKET_ROLE_RESERVED) {
        local_.SaveSocketConfig(
            Hccl::SocketConfig(linkData, local_.serverPort, socketTag, channelDesc_.role == HCOMM_SOCKET_ROLE_SERVER));
    } else if (traits_.reservedSocketAddrTag) {
        local_.SaveSocketConfig(Hccl::SocketConfig(linkData, socketTag, true));
    } else {
        Hccl::SocketConfig socketConfig(linkData, local_.serverPort, socketTag);
        socketConfig.noRankId = traits_.reservedNoRankId;
        local_.SaveSocketConfig(socketConfig);
    }
    if (channelDesc_.channelName != nullptr && channelDesc_.channelName[0] != '\0') {
        local_.socketConfigHolder->SetHccpTag(socketTag);
    }
    CHK_RET(GetSocketMgr(local_.devicePhyId).GetSocket(*local_.socketConfig, local_.socket));

    local_.isRecvFirst = (local_.socket != nullptr) && (local_.socket->GetRole() == Hccl::SocketRole::CLIENT);
    HCCL_INFO("[%s::%s] SUCCESS. port[%u].", channelClassName_, __func__, local_.serverPort);
    return HCCL_SUCCESS;
}

HcclResult Channel::BuildConnection() { return HCCL_SUCCESS; }

std::unique_ptr<Hccl::BaseLocalNotify> Channel::CreateLocalNotify(bool devUsed) const
{
    switch (traits_.notifyType) {
        case LocalNotifyType::UB:
            return std::make_unique<Hccl::UbLocalNotify>(local_.rdmaHandle, devUsed);
        case LocalNotifyType::IPC:
            return std::make_unique<Hccl::IpcLocalNotify>(devUsed);
        case LocalNotifyType::NONE:
        default:
            return nullptr;
    }
}

HcclResult Channel::BuildNotify()
{
    if (traits_.notifyBuildMode == NotifyBuildMode::NONE) {
        return HCCL_SUCCESS;
    }

    local_.notifies.clear();
    local_.commonRes.notifyVec.clear();
    bool devUsed = true;
    for (uint32_t i = 0; i < local_.notifyNum; ++i) {
        std::unique_ptr<Hccl::BaseLocalNotify> notifyPtr;
        EXCEPTION_CATCH(notifyPtr = CreateLocalNotify(devUsed), return HCCL_E_PTR);
        if (notifyPtr == nullptr) {
            HCCL_ERROR("[%s::%s] create local notify failed, index[%u].", channelClassName_, __func__, i);
            return HCCL_E_PTR;
        }
        local_.commonRes.notifyVec.push_back(notifyPtr.get());
        local_.notifies.push_back(std::move(notifyPtr));
    }
    PLF_CONFIG_INFO(
        PLF_CHANNEL, "[%s::%s] create notify, notifyNum[%u].", channelClassName_, __func__, local_.notifyNum);
    return HCCL_SUCCESS;
}
HcclResult Channel::BuildBuffer() { return HCCL_SUCCESS; }

HcclResult Channel::UpdateMemInfo([[maybe_unused]] HcommMemHandle* memHandles, [[maybe_unused]] uint32_t memHandleNum)
{
    HCCL_WARNING("[%s::%s] not support.", channelClassName_, __func__);
    return HCCL_SUCCESS;
}

HcclResult Channel::Serialize(std::shared_ptr<hccl::DeviceMem>& out)
{
    out.reset();
    return HCCL_E_NOT_SUPPORT;
}

HcclResult Channel::NotifyRecord([[maybe_unused]] const uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[%s::%s] not supported yet.", channelClassName_, __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult Channel::NotifyWait([[maybe_unused]] const uint32_t localNotifyIdx, [[maybe_unused]] const uint32_t timeout)
{
    HCCL_INFO("[%s::%s] not supported yet.", channelClassName_, __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult Channel::WriteWithNotify(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] const uint64_t len,
    [[maybe_unused]] uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[%s::%s] not supported yet.", channelClassName_, __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult Channel::Write([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[%s::%s] not supported yet.", channelClassName_, __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult Channel::Read([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[%s::%s] not supported yet.", channelClassName_, __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult Channel::ChannelFence()
{
    HCCL_INFO("[%s::%s] not supported yet.", channelClassName_, __func__);
    return HCCL_E_NOT_SUPPORT;
}

void Channel::PutSocketIfNeeded()
{
    if (local_.socket == nullptr) {
        return;
    }
    if (local_.socketConfig != nullptr && channelDesc_.socket == nullptr) {
        (void)GetSocketMgr(local_.devicePhyId).DestroySocket(*local_.socketConfig);
    }
    local_.socket = nullptr;
}

ChannelStatus Channel::TransportStatusToChannelStatus(
    Hccl::TransportStatus ts, const EndpointDesc& localEp, const HcommChannelDesc& channelDesc)
{
    const EndpointDesc& remoteEp = channelDesc.remoteEndpoint;

    if (ts != lastTransportStatus_) {
        lastTransportStatus_ = ts;
        if ((Hccl::GetPlfDebugConfigValue() & PLF_CHANNEL) != 0) {
            Hccl::IpAddress localAddr{};
            std::string localEid = "invalid";
            if (CommAddrToIpAddress(localEp.commAddr, localAddr) == HCCL_SUCCESS) {
                localEid = localAddr.Describe();
            }

            Hccl::IpAddress remoteAddr{};
            std::string remoteEid = "invalid";
            if (CommAddrToIpAddress(remoteEp.commAddr, remoteAddr) == HCCL_SUCCESS) {
                remoteEid = remoteAddr.Describe();
            }

            const char* socketTag = channelDesc.channelName != nullptr ? channelDesc.channelName : "anonymous";

            PLF_CONFIG_INFO(
                PLF_CHANNEL, "[%s::%s] status[%d], protocol[%s], localEid[%s], remoteEid[%s], socketTag[%s].",
                channelClassName_, __func__, static_cast<int>(ts),
                GetEnumToString(GetCommProtocolStrMap(), remoteEp.protocol).c_str(), localEid.c_str(),
                remoteEid.c_str(), socketTag);
        }
    }
    switch (ts) {
        case Hccl::TransportStatus::INIT:
            return ChannelStatus::INIT;
        case Hccl::TransportStatus::SOCKET_OK:
            return ChannelStatus::SOCKET_OK;
        case Hccl::TransportStatus::SOCKET_TIMEOUT:
            return ChannelStatus::SOCKET_TIMEOUT;
        case Hccl::TransportStatus::READY:
            return ChannelStatus::READY;
        default:
            HCCL_ERROR("[%s::%s] Invalid TransportStatus[%d]", channelClassName_, __func__, ts);
            return ChannelStatus::FAILED;
    }
}

HcclResult Channel::CreateChannel(
    EndpointHandle endpointHandle, CommEngine engine, const HcommChannelDesc& channelDesc, HcommChannelConfig config,
    std::shared_ptr<Channel>& out)
{
    // deviceType 仅用于错误日志；原站点已 CHK_RET 校验，失败态按 value_or(false) 兜底
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    std::shared_ptr<Channel> uniqueChannelPtr;
    switch (engine) {
        case COMM_ENGINE_CPU:
            if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_ROCE) {
                EXCEPTION_CATCH(
                    uniqueChannelPtr = std::make_unique<HostCpuRoceChannel>(endpointHandle, channelDesc, engine),
                    return HCCL_E_PARA);
                break;
            }
            if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_CTP
                || channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_TP) {
                EXCEPTION_CATCH(
                    uniqueChannelPtr = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc, engine),
                    return HCCL_E_PARA);
                break;
            }
            HCCL_ERROR(
                "[Channel::%s] Engine[COMM_ENGINE_CPU] not support Protocol[%d]", __func__,
                channelDesc.remoteEndpoint.protocol);
            return HCCL_E_NOT_SUPPORT;
        case COMM_ENGINE_CPU_TS:
            HCCL_ERROR("[Channel::%s] CommEngine[COMM_ENGINE_CPU_TS] not support", __func__);
            return HCCL_E_NOT_SUPPORT;
        case COMM_ENGINE_AICPU:
        case COMM_ENGINE_AICPU_TS:
            if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBOE) {
                uniqueChannelPtr.reset(new (std::nothrow) AicpuTsUboeChannel(endpointHandle, channelDesc, engine));
            } else if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_RTP) {
                if (!hcomm::EqualToOrGreaterThan950().value_or(false)) {
                    HCCL_ERROR(
                        "[Channel::%s] UB_RTP protocol only support DEV_TYPE_950/960, current deviceType=%d", __func__,
                        static_cast<int>(deviceType));
                    return HCCL_E_NOT_SUPPORT;
                }
                uniqueChannelPtr.reset(new (std::nothrow) AicpuTsUbRtpChannel(endpointHandle, channelDesc, engine));
            } else if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_PCIE) {
                uniqueChannelPtr.reset(new (std::nothrow) AicpuTsP2pChannel(endpointHandle, channelDesc, engine));
            } else if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_ROCE) {
                if (hcomm::EqualToOrGreaterThan950().value_or(false)) {
                    uniqueChannelPtr = std::make_unique<AicpuTsRoceHostLocChannel>(endpointHandle, channelDesc, engine);
                } else {
                    uniqueChannelPtr = std::make_unique<AicpuTsRoceDevLocChannel>(endpointHandle, channelDesc, engine);
                }
            } else if (
                channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_CTP
                || channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_TP) {
                uniqueChannelPtr.reset(new (std::nothrow) AicpuTsUrmaChannel(endpointHandle, channelDesc, engine));
            } else if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_HCCS) {
                uniqueChannelPtr.reset(new (std::nothrow) AicpuTsHccsChannel(endpointHandle, channelDesc, engine));
            } else {
                HCCL_ERROR(
                    "[Channel::%s] invalid protocol for engine[%s], protocol[%s]", __func__,
                    GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),
                    GetEnumToString(GetCommProtocolStrMap(), channelDesc.remoteEndpoint.protocol).c_str());
                return HCCL_E_NOT_SUPPORT;
            }
            break;
        case COMM_ENGINE_AIV:
            if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_ROCE
                && hcomm::EqualToOrGreaterThan950().value_or(false)) {
                uniqueChannelPtr = std::make_unique<AivRoceHostLocChannel>(endpointHandle, channelDesc, engine);
            } else if (
                channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_CTP
                || channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_TP
                || channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_RTP) {
                if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_RTP
                    && !hcomm::EqualToOrGreaterThan950().value_or(false)) {
                    HCCL_ERROR(
                        "[Channel::%s] UB_RTP protocol only support DEV_TYPE_950/960, current deviceType=%d", __func__,
                        static_cast<int>(deviceType));
                    return HCCL_E_NOT_SUPPORT;
                }
                uniqueChannelPtr.reset(new (std::nothrow) AivUrmaChannel(endpointHandle, channelDesc, engine));
            } else {
                uniqueChannelPtr.reset(new (std::nothrow) AivUbMemChannel(endpointHandle, channelDesc, engine));
            }
            break;
        case COMM_ENGINE_CCU:
            uniqueChannelPtr.reset(new (std::nothrow) CcuUrmaChannel(endpointHandle, channelDesc, engine));
            break;
        default:
            HCCL_ERROR("[Channel::%s] invalid type of CommEngine", __func__);
            return HCCL_E_NOT_FOUND;
    }
    CHK_PTR_NULL(uniqueChannelPtr);
    bool isSharedQueue = false;
    if (config != nullptr) {
        auto* cfg = static_cast<HcommChannelConfigData*>(config);
        isSharedQueue = cfg->isSharedQueue;
    }
    uniqueChannelPtr->SetSharedQueue(isSharedQueue);
    CHK_RET_UNAVAIL(uniqueChannelPtr->Init());
    out = std::move(uniqueChannelPtr);
    return HCCL_SUCCESS;
}

} // namespace hcomm
