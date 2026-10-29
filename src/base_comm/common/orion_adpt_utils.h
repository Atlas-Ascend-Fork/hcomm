/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ORION_ADPT_UTILS_H
#define ORION_ADPT_UTILS_H

#include "hccl_net_dev_defs.h"
#include "hccp_ctx.h"
#include "hcomm_adapter_hccp.h"
#include "common/orion_adpt_utils_pub.h"
#include "hccl_ip_address.h"

// Orion
#include "virtual_topo.h"

namespace hcomm {

HcclResult IpAddressToCommAddr(const Hccl::IpAddress& ipAddr, CommAddr& commAddr);

constexpr u32 UB_SQ_DEPTH_MIN = 16U;
constexpr u32 UB_SCQ_DEPTH_MIN = 64U;
constexpr u32 UB_SCQ_DEPTH_MAX_NORMAL = 32768U;     // Host UB (normal 模式)
constexpr u32 UB_SCQ_DEPTH_MAX_STARS_POLL = 16384U; // AICPU UB (stars poll 模式)
constexpr u32 UB_SQ_DEPTH_NOT_SET = 0xFFFFFFFFU;    // sqDepth 哨兵值，表示使用默认值
constexpr u32 UB_SCQ_DEPTH_NOT_SET = 0xFFFFFFFFU;   // scqDepth 哨兵值，表示使用默认值

HcclResult CommAddrToIpAddress(const CommAddr& commAddr, Hccl::IpAddress& ipAddr);
HcclResult IpAddressToCommAddr(const Hccl::IpAddress& ipAddr, CommAddr& commAddr);
Hccl::IpAddress HcclIpAddressToIpAddress(const hccl::HcclIpAddress& v1);
hccl::HcclIpAddress IpAddressToHcclIpAddress(const Hccl::IpAddress& v2);
HcclResult CommProtocolToLinkProtocol(CommProtocol commProtocol, Hccl::LinkProtocol& linkProtocol);
HcclResult CommAddrTypeToHcclAddressType(CommAddrType commAddrType, HcclAddressType& hcclAddressType);
Hccl::LinkData BuildDefaultLinkData();
HcclResult EndpointDescPairToLinkData(
    const EndpointDesc& locEp, const EndpointDesc& rmtEp, Hccl::LinkData& linkData, u32 reuseIdx = 0);
HcclResult EndpointDescPairToLinkDataWithRankIds(
    const uint32_t myRank, const uint32_t rmtRank, const EndpointDesc& locEp, const EndpointDesc& rmtEp,
    Hccl::LinkData& linkData, uint32_t devicePhyId, uint32_t remoteDevicePhyId, u32 reuseIdx = 0);

struct UbConnBuildContext {
    Hccl::LinkProtocol protocol{};
    Hccl::IpAddress locAddr{};
    Hccl::IpAddress rmtAddr{};
    s32 deviceLogicId{0};
    u8 qosPre{0};
    u32 sqDepth{0xFFFFFFFFU};
    u32 scqDepth{0xFFFFFFFFU};
};

HcclResult PrepareUbConnBuildContext(
    const EndpointDesc& locEp, const EndpointDesc& rmtEp, const HcommChannelDesc& channelDesc, UbConnBuildContext& ctx);

HcclResult CheckUbSqDepth(const UbConnBuildContext& ctx, const DevBaseAttr& devBaseAttr);

HcclResult CheckUbScqDepth(const UbConnBuildContext& ctx, HrtUbJfcMode jfcMode);

} // namespace hcomm

#endif // ORION_ADPT_UTILS_H
