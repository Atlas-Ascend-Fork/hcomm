/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ts_p2p_channel.h"
#include "endpoint.h"
#include "../../sockets/socket_mgr.h"
#include "base_comm_node_accessors.h"
#include "orion_adpt_utils.h"
#include "hcomm_c_adpt.h"
#include "adapter_rts_common.h"

#include "virtual_topo.h"
#include "p2p_connection.h"

namespace hcomm {

AicpuTsP2pChannel::AicpuTsP2pChannel(
    EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc, CommEngine engine)
    : Channel(endpointHandle, channelDesc, engine, HcommChannelKind::AICPU_TS_PCIE, "AicpuTsP2pChannel")
{}

AicpuTsP2pChannel::~AicpuTsP2pChannel() {}

HcclResult AicpuTsP2pChannel::ParseInputParam()
{
    CHK_RET(Channel::ParseInputParam());

    CHK_RET(CollectLocalRmaBuffers(local_.commonRes.bufferVec));

    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::StartListen()
{
    CHK_RET(Channel::StartListen());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::BuildSocket()
{
    CHK_RET(Channel::BuildSocket());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::BuildConnection()
{
    CHK_RET(Channel::BuildConnection());
    CHK_RET(BuildAttr());
    std::unique_ptr<Hccl::P2PConnection> p2pConn = nullptr;
    std::string connTag = "P2P_CHANNEL_" + std::to_string(local_.ep.loc.device.devPhyId);

    EXCEPTION_CATCH(p2pConn = std::make_unique<Hccl::P2PConnection>(local_.socket, connTag), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(p2pConn);

    local_.commonRes.connVec.clear();
    local_.commonRes.connVec.emplace_back(p2pConn.get());
    connections_.clear();
    connections_.push_back(std::move(p2pConn));

    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::BuildNotify()
{
    CHK_RET(Channel::BuildNotify());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::BuildBuffer()
{
    CHK_RET(Channel::BuildBuffer());
    const Hccl::Socket& socket = *local_.socket;

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(local_.ep, remote_.ep, linkData));

    EXCEPTION_CATCH(
        memTransport_ = std::make_unique<Hccl::P2PTransport>(local_.commonRes, local_.attr, linkData, socket),
        return HCCL_E_PTR);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::GetNotifyNum(uint32_t* notifyNum) const
{
    *notifyNum = this->local_.notifyNum;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    return memTransport_->GetRemoteMems(memNum, remoteMem, memInfos);
}

ChannelStatus AicpuTsP2pChannel::GetStatus()
{
    ChannelStatus out = TransportStatusToChannelStatus(memTransport_->GetStatus(), local_.ep, GetChannelDesc());
    return out;
}

HcclResult
AicpuTsP2pChannel::UpdateMemInfo([[maybe_unused]] HcommMemHandle* memHandles, [[maybe_unused]] uint32_t memHandleNum)
{
    HCCL_WARNING("[%s::%s] P2PTransport does not support UpdateMemInfo.", channelClassName_, __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::Clean()
{
    memTransport_.reset();
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::Resume()
{
    CHK_RET(BuildSocket());
    CHK_RET(BuildConnection());
    CHK_RET(BuildBuffer());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::H2DResPack(std::vector<char>& buffer)
{
    CHK_RET(PackOpData(buffer));
    HCCL_INFO(
        "[%s::%s] Pack Buffer data[%p], Pack Buffer size[%zu].", channelClassName_, __func__, buffer.data(),
        buffer.size());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::BuildAttr()
{
    local_.attr.devicePhyId = local_.ep.loc.device.devPhyId;
    local_.attr.opMode = Hccl::OpMode::OPBASE;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::PackOpData(std::vector<char>& data)
{
    std::vector<Hccl::ModuleData> dataVec;
    dataVec.resize(Hccl::AicpuResMgrType::__COUNT__);

    Hccl::AicpuResMgrType resType = Hccl::AicpuResMgrType::STREAM;
    CHK_RET(SetModuleDataName(dataVec[resType], "P2PTransport"));

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

HcclResult AicpuTsP2pChannel::SetModuleDataName(Hccl::ModuleData& module, const std::string& name) const
{
    int ret = strcpy_s(module.name, sizeof(module.name), name.c_str());
    if (ret != 0) {
        HCCL_ERROR("[%s::%s] strcpy_s name %s failed", channelClassName_, __func__, name.c_str());
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

} // namespace hcomm
