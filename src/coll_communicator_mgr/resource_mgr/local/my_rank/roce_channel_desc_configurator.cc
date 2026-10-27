/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "roce_channel_desc_configurator.h"

#include "adapter_rts_common.h"
#include "coll_comm_mgr.h"
#include "env_config/env_config_v2.h"
#include "common/orion_adpt_utils_pub.h"

namespace hccl {

RoceChannelDescConfigurator::RoceChannelDescConfigurator(uint32_t channelNum) : srcPortBuffers_(channelNum) {}

void RoceChannelDescConfigurator::FillPortsFromHostMultiQpConfig(std::vector<uint16_t>& ports)
{
    ports.clear();
    s32 deviceLogicId = INVALID_INT;
    uint32_t devicePhyId = INVALID_UINT;
    if ((hrtGetDevice(&deviceLogicId) != HCCL_SUCCESS)
        || (hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(deviceLogicId), devicePhyId, false) != HCCL_SUCCESS)) {
        HCCL_WARNING("[%s] get device phy id failed, fall back to MultiQpSrcPort.cfg", __func__);
        return;
    }

    const auto& hostMultiQpConfig = CollCommMgr::GetInstance().GetConfigMgr().GetHostMultiQpConfig();
    const auto* configuredPorts = hostMultiQpConfig.GetUdpPorts(devicePhyId);
    if (configuredPorts != nullptr) {
        ports = *configuredPorts;
        return;
    }
    HCCL_INFO("[%s] no matching host multi qp UDP ports for phyId[%u]", __func__, devicePhyId);
}

HcclResult RoceChannelDescConfigurator::FillPortsFromMultiQpSrcPortConfig(
    const HcclChannelDesc& hcclDesc, std::vector<uint16_t>& ports)
{
    const auto& qpSrcPortConfig = Hccl::EnvConfig::GetInstance().GetRdmaConfig().GetMultiQpSrcPortConfig();
    if (!qpSrcPortConfig.IsAvailable()) {
        HCCL_INFO(
            "[%s] skip: multiQpSrcPortConfig not available (env HCCL_RDMA_QP_PORT_CONFIG_PATH unset or "
            "MultiQpSrcPort.cfg empty)",
            __func__);
        return HCCL_SUCCESS;
    }

    Hccl::IpAddress localIp;
    Hccl::IpAddress remoteIp;
    HcclResult localRet = hcomm::CommAddrToIpAddress(hcclDesc.localEndpoint.commAddr, localIp);
    HcclResult remoteRet = hcomm::CommAddrToIpAddress(hcclDesc.remoteEndpoint.commAddr, remoteIp);
    CHK_PRT_RET(
        localRet != HCCL_SUCCESS || remoteRet != HCCL_SUCCESS,
        HCCL_ERROR("[%s] CommAddrToIpAddress failed: localRet[%d] remoteRet[%d]", __func__, localRet, remoteRet),
        HCCL_E_INTERNAL);
    ports = Hccl::GetMultiQpSrcPortsByIpPair(qpSrcPortConfig, localIp, remoteIp);
    if (ports.empty()) {
        HCCL_INFO(
            "[%s] skip: no matching ports for localIp[%s] remoteIp[%s]", __func__, localIp.GetIpStr().c_str(),
            remoteIp.GetIpStr().c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult
RoceChannelDescConfigurator::ResolveRoceSrcPorts(const HcclChannelDesc& hcclDesc, std::vector<uint16_t>& ports)
{
    FillPortsFromHostMultiQpConfig(ports);
    if (!ports.empty()) {
        return HCCL_SUCCESS;
    }

    return FillPortsFromMultiQpSrcPortConfig(hcclDesc, ports);
}

HcclResult RoceChannelDescConfigurator::FillRoceSrcPortList(
    const HcclChannelDesc& hcclDesc, uint32_t channelIndex, HcommChannelDesc& hcommDesc)
{
    hcommDesc.roceAttr.srcPortList = nullptr;
    if (hcclDesc.localEndpoint.loc.locType != ENDPOINT_LOC_TYPE_HOST
        || hcommDesc.remoteEndpoint.protocol != COMM_PROTOCOL_ROCE || hcommDesc.exchangeAllMems) {
        HCCL_INFO(
            "[%s] skip: localLocType[%d] protocol[%d] exchangeAllMems[%d]", __func__,
            hcclDesc.localEndpoint.loc.locType, hcommDesc.remoteEndpoint.protocol, hcommDesc.exchangeAllMems);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        channelIndex >= srcPortBuffers_.size(),
        HCCL_ERROR(
            "[%s] channelIndex[%u] is out of range, channelNum[%zu]", __func__, channelIndex, srcPortBuffers_.size()),
        HCCL_E_PARA);

    std::vector<uint16_t> ports;
    CHK_RET(ResolveRoceSrcPorts(hcclDesc, ports));
    if (ports.empty()) {
        return HCCL_SUCCESS;
    }
    const uint32_t queueNum = hcommDesc.roceAttr.queueNum;
    if (ports.size() != static_cast<std::size_t>(queueNum)) {
        HCCL_RUN_WARNING(
            "[%s] UDP source port count[%zu] does not match queueNum[%u], ports will be truncated or reused "
            "cyclically",
            __func__, ports.size(), queueNum);
    }
    auto& srcPortBuffer = srcPortBuffers_[channelIndex];
    srcPortBuffer.resize(queueNum);
    for (uint32_t index = 0; index < queueNum; ++index) {
        srcPortBuffer[index] = ports[index % ports.size()];
    }
    hcommDesc.roceAttr.srcPortList = srcPortBuffer.data();
    HCCL_INFO("[%s] success: queueNum[%u] portCount[%zu]", __func__, queueNum, ports.size());
    return HCCL_SUCCESS;
}

} // namespace hccl
