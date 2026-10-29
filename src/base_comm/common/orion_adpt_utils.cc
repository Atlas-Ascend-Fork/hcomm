/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "orion_adpt_utils.h"

// Orion
#include "adapter_rts_common.h"
#include "orion_adapter_hccp.h"
#include "tp_manager.h"
#include "topo_common_types.h"
#include "virtual_topo.h"
#include "hcomm_c_adpt.h"
#include "hcomm_adapter_hccp.h"

namespace hcomm {

const char* CommAddrTypeToStr(CommAddrType type)
{
    switch (type) {
        case COMM_ADDR_TYPE_IP_V4:
            return "COMM_ADDR_TYPE_IP_V4";
        case COMM_ADDR_TYPE_IP_V6:
            return "COMM_ADDR_TYPE_IP_V6";
        case COMM_ADDR_TYPE_EID:
            return "COMM_ADDR_TYPE_EID";
        case COMM_ADDR_TYPE_ID:
            return "COMM_ADDR_TYPE_ID";
        case COMM_ADDR_TYPE_RESERVED:
            return "COMM_ADDR_TYPE_RESERVED";
        default:
            return "UNKNOWN_COMM_ADDR_TYPE";
    }
}

HcclResult CommAddrToIpAddress(const CommAddr& commAddr, Hccl::IpAddress& ipAddr)
{
    if (commAddr.type != COMM_ADDR_TYPE_IP_V4 && commAddr.type != COMM_ADDR_TYPE_IP_V6
        && commAddr.type != COMM_ADDR_TYPE_EID) {
        if (commAddr.type == COMM_ADDR_TYPE_ID || commAddr.type == COMM_ADDR_TYPE_RESERVED) {
            HCCL_WARNING(
                "[%s] is unsuccessful, comm address type[%d][%s] is not supported.", __func__, commAddr.type,
                CommAddrTypeToStr(commAddr.type));
        } else {
            HCCL_WARNING(
                "[%s] is unsuccessful, comm address type[%d][%s] is not valid.", __func__, commAddr.type,
                CommAddrTypeToStr(commAddr.type));
        }
        return HCCL_E_NOT_SUPPORT;
    }

    Hccl::BinaryAddr binAddr;
    int32_t family = AF_INET6;
    if (commAddr.type == COMM_ADDR_TYPE_IP_V4) {
        binAddr.addr = commAddr.addr;
        int32_t family = AF_INET;
        ipAddr = Hccl::IpAddress(binAddr, family);
        return HCCL_SUCCESS;
    }

    if (commAddr.type == COMM_ADDR_TYPE_EID) {
        Hccl::Eid inputEid;
        s32 sret = memcpy_s(inputEid.raw, Hccl::URMA_EID_LEN, commAddr.eid, Hccl::URMA_EID_LEN);
        CHK_PRT_RET(sret != EOK, HCCL_ERROR("memcpy failed, errno[%d]", sret), HCCL_E_MEMORY);
        ipAddr = Hccl::IpAddress(inputEid);
        return HCCL_SUCCESS;
    }

    binAddr.addr6 = commAddr.addr6;
    ipAddr = Hccl::IpAddress(binAddr, family);
    return HCCL_SUCCESS;
}

HcclResult IpAddressToCommAddr(const Hccl::IpAddress& ipAddr, CommAddr& commAddr)
{
    int32_t family = ipAddr.GetFamily();
    const auto& binAddr = ipAddr.GetBinaryAddress();

    if (family == AF_INET) {
        commAddr.addr = binAddr.addr;
        commAddr.type = COMM_ADDR_TYPE_IP_V4;
        return HcclResult::HCCL_SUCCESS;
    }

    commAddr.addr6 = binAddr.addr6;
    commAddr.type = COMM_ADDR_TYPE_IP_V6;
    return HcclResult::HCCL_SUCCESS;
}

Hccl::IpAddress HcclIpAddressToIpAddress(const hccl::HcclIpAddress& v1)
{
    Hccl::BinaryAddr bin{};
    const hccl::HcclInAddr src = v1.GetBinaryAddress();
    if (v1.GetFamily() == AF_INET6) {
        bin.addr6 = src.addr6;
    } else {
        bin.addr = src.addr;
    }
    return Hccl::IpAddress(bin, v1.GetFamily(), v1.GetScopeID());
}

hccl::HcclIpAddress IpAddressToHcclIpAddress(const Hccl::IpAddress& v2)
{
    hccl::HcclInAddr bin{};
    const Hccl::BinaryAddr src = v2.GetBinaryAddress();
    if (v2.GetFamily() == AF_INET6) {
        bin.addr6 = src.addr6;
    } else {
        bin.addr = src.addr;
    }
    hccl::HcclIpAddress out(v2.GetFamily(), bin);
    (void)out.SetScopeID(v2.GetScopeID());
    return out;
}

HcclResult CommProtocolToLinkProtocol(CommProtocol commProtocol, Hccl::LinkProtocol& linkProtocol)
{
    switch (commProtocol) {
        case COMM_PROTOCOL_UB_CTP:
            linkProtocol = Hccl::LinkProtocol::UB_CTP;
            break;
        case COMM_PROTOCOL_UBC_TP:
            linkProtocol = Hccl::LinkProtocol::UB_TP;
            break;
        case COMM_PROTOCOL_ROCE:
            linkProtocol = Hccl::LinkProtocol::ROCE;
            break;
        case COMM_PROTOCOL_HCCS:
            linkProtocol = Hccl::LinkProtocol::HCCS;
            break;
        case COMM_PROTOCOL_UB_MEM:
            linkProtocol = Hccl::LinkProtocol::UB_MEM;
            break;
        case COMM_PROTOCOL_PCIE:
            linkProtocol = Hccl::LinkProtocol::PCIE;
            break;
        case COMM_PROTOCOL_UBOE:
            linkProtocol = Hccl::LinkProtocol::UBOE;
            break;
        case COMM_PROTOCOL_UB_RTP:
            linkProtocol = Hccl::LinkProtocol::UB_RTP;
            break;
        default:
            HCCL_ERROR("[%s] Invalid CommProtocol[%u]", __func__, commProtocol);
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CommAddrTypeToHcclAddressType(CommAddrType commAddrType, HcclAddressType& hcclAddressType)
{
    switch (commAddrType) {
        case COMM_ADDR_TYPE_IP_V4:
            hcclAddressType = HCCL_ADDR_TYPE_IP_V4;
            break;
        case COMM_ADDR_TYPE_IP_V6:
            hcclAddressType = HCCL_ADDR_TYPE_IP_V6;
            break;
        default:
            HCCL_ERROR("[%s] Invalid CommAddrType[%u]", __func__, commAddrType);
            return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

Hccl::LinkData BuildDefaultLinkData()
{
    Hccl::PortDeploymentType portDeploymentType = Hccl::PortDeploymentType::HOST_NET;
    Hccl::LinkProtocol linkProtocol = Hccl::LinkProtocol::ROCE;
    Hccl::IpAddress locAddr;
    Hccl::IpAddress rmtAddr;
    uint32_t locDevPhyId = 0;
    uint32_t rmtDevPhyId = 0;
    return Hccl::LinkData(portDeploymentType, linkProtocol, locDevPhyId, rmtDevPhyId, locAddr, rmtAddr);
}

static HcclResult
EndpointLocTypeToPortDeploymentType(const EndpointLocType locType, Hccl::PortDeploymentType& deployType)
{
    switch (locType) {
        case EndpointLocType::ENDPOINT_LOC_TYPE_HOST:
            deployType = Hccl::PortDeploymentType::HOST_NET;
            break;
        case EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE:
            deployType = Hccl::PortDeploymentType::DEV_NET;
            break;
        default:
            HCCL_ERROR("[%s] unknown type of EndpointLocType[%d]", __func__, locType);
            return HcclResult::HCCL_E_PARA;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
EndpointDescPairToLinkData(const EndpointDesc& locEp, const EndpointDesc& rmtEp, Hccl::LinkData& linkData, u32 reuseIdx)
{
    Hccl::PortDeploymentType portDeploymentType = Hccl::PortDeploymentType::INVALID;
    CHK_RET(EndpointLocTypeToPortDeploymentType(locEp.loc.locType, portDeploymentType));

    Hccl::LinkProtocol linkProtocol = Hccl::LinkProtocol::INVALID;
    CHK_RET(CommProtocolToLinkProtocol(locEp.protocol, linkProtocol));

    Hccl::IpAddress locAddr{};
    Hccl::IpAddress rmtAddr{};
    CHK_RET(CommAddrToIpAddress(locEp.commAddr, locAddr));
    CHK_RET(CommAddrToIpAddress(rmtEp.commAddr, rmtAddr));

    uint32_t locDevPhyId = locEp.loc.device.devPhyId;
    uint32_t rmtDevPhyId = rmtEp.loc.device.devPhyId;

    // 开源开放架构下comms层级不感知通信域层级的rank信息
    // 当前复用orion数据结构故使用devId替换
    // 注意：devicePhyId/remoteDevicePhyId 也需显式传入，否则 remoteDeviceId_ 取默认值 0，
    // 会导致按需 enable p2p 时对端物理设备ID错误
    linkData = Hccl::LinkData(
        portDeploymentType, linkProtocol, locDevPhyId, rmtDevPhyId, locAddr, rmtAddr, locDevPhyId, rmtDevPhyId,
        reuseIdx);

    return HCCL_SUCCESS;
}

HcclResult EndpointDescPairToLinkDataWithRankIds(
    const uint32_t myRank, const uint32_t rmtRank, const EndpointDesc& locEp, const EndpointDesc& rmtEp,
    Hccl::LinkData& linkData, uint32_t devicePhyId, uint32_t remoteDevicePhyId, u32 reuseIdx)
{
    Hccl::PortDeploymentType portDeploymentType = Hccl::PortDeploymentType::INVALID;
    CHK_RET(EndpointLocTypeToPortDeploymentType(locEp.loc.locType, portDeploymentType));

    Hccl::LinkProtocol linkProtocol = Hccl::LinkProtocol::INVALID;
    CHK_RET(CommProtocolToLinkProtocol(locEp.protocol, linkProtocol));

    Hccl::IpAddress locAddr{};
    Hccl::IpAddress rmtAddr{};
    CHK_RET(CommAddrToIpAddress(locEp.commAddr, locAddr));
    CHK_RET(CommAddrToIpAddress(rmtEp.commAddr, rmtAddr));

    // 临时方案，为支持开源开放与orion通信域混跑，复用orion数据结构，添加rank信息
    linkData = Hccl::LinkData(
        portDeploymentType, linkProtocol, myRank, rmtRank, locAddr, rmtAddr, devicePhyId, remoteDevicePhyId, reuseIdx);
    linkData.UpdateIpAddrWithPCIE();

    return HCCL_SUCCESS;
}

HcclResult PrepareUbConnBuildContext(
    const EndpointDesc& locEp, const EndpointDesc& rmtEp, const HcommChannelDesc& channelDesc, UbConnBuildContext& ctx)
{
    CHK_RET(CommProtocolToLinkProtocol(locEp.protocol, ctx.protocol));
    CHK_RET(CommAddrToIpAddress(locEp.commAddr, ctx.locAddr));
    CHK_RET(CommAddrToIpAddress(rmtEp.commAddr, ctx.rmtAddr));
    CHK_RET(hrtGetDevice(&ctx.deviceLogicId));
    Hccl::TpManager::GetInstance(ctx.deviceLogicId).Init();
    if (channelDesc.qos > 7U) {
        HCCL_WARNING(
            "[PrepareUbConnBuildContext] channelQos[%u] is not valid, expect [0, 7], use default qos[%u].",
            channelDesc.qos, Hccl::kRaUbGetTpInfoParamDefaultQos);
        ctx.qosPre = static_cast<u8>(Hccl::kRaUbGetTpInfoParamDefaultQos);
    } else {
        ctx.qosPre = static_cast<u8>(channelDesc.qos);
    }
    ctx.sqDepth = channelDesc.ubAttr.sqDepth;
    ctx.scqDepth = channelDesc.ubAttr.scqDepth;
    return HCCL_SUCCESS;
}

HcclResult CheckUbSqDepth(const UbConnBuildContext& ctx, const DevBaseAttr& devBaseAttr)
{
    if (ctx.sqDepth == UB_SQ_DEPTH_NOT_SET || ctx.sqDepth == 0) {
        return HCCL_SUCCESS;
    }
    if (ctx.sqDepth < UB_SQ_DEPTH_MIN || ctx.sqDepth > devBaseAttr.sqMaxDepth) {
        HCCL_ERROR(
            "[%s] invalid ubAttr.sqDepth[%u], aligned range is [%u, %u] (aligned to power-of-two before compared).",
            __func__, ctx.sqDepth, UB_SQ_DEPTH_MIN, devBaseAttr.sqMaxDepth);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CheckUbScqDepth(const UbConnBuildContext& ctx, HrtUbJfcMode jfcMode)
{
    if (ctx.scqDepth == UB_SCQ_DEPTH_NOT_SET || ctx.scqDepth == 0) {
        return HCCL_SUCCESS;
    }
    u32 scqMaxDepth = 0;
    if (jfcMode == HrtUbJfcMode::NORMAL || jfcMode == HrtUbJfcMode::USER_CTL) {
        scqMaxDepth = UB_SCQ_DEPTH_MAX_NORMAL;
    } else if (jfcMode == HrtUbJfcMode::STARS_POLL) {
        scqMaxDepth = UB_SCQ_DEPTH_MAX_STARS_POLL;
    } else {
        HCCL_ERROR("[%s] invalid jfcMode for scqDepth check.", __func__);
        return HCCL_E_PARA;
    }
    if (ctx.scqDepth < UB_SCQ_DEPTH_MIN || ctx.scqDepth > scqMaxDepth) {
        HCCL_ERROR(
            "[%s] invalid ubAttr.scqDepth[%u], expect range is [%u, %u].", __func__, ctx.scqDepth, UB_SCQ_DEPTH_MIN,
            scqMaxDepth);
        return HCCL_E_PARA;
    }
    // 非硬件轮询模式（NORMAL/USER_CTL）校验 SQ 深度不大于 CQ 深度，避免 CQ overflow
    if (jfcMode != HrtUbJfcMode::STARS_POLL && ctx.sqDepth != UB_SQ_DEPTH_NOT_SET && ctx.sqDepth != 0
        && ctx.sqDepth > ctx.scqDepth) {
        HCCL_ERROR(
            "[%s] ubAttr.sqDepth[%u] > scqDepth[%u], may cause CQ overflow.", __func__, ctx.sqDepth, ctx.scqDepth);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

} // namespace hcomm
