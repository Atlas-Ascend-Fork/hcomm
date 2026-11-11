/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LINK_DATA_ADAPTER_H
#define LINK_DATA_ADAPTER_H

#include <cstdint>
#include <memory>
#include <set>

#include "virtual_topo.h"
#include "port.h"
#include "net_instance.h"
#include "log.h"

namespace Hccl {

// 控制面辅助：用于 RankGraph 构建/打包阶段的 Path/ConnInterface 转换。
// 转换必须留在 coll 侧，避免 base_comm 反向依赖 coll；资源面运行时不使用该文件。
// 不能放回 virtual_topo.h/port.h：这两个头仍被 base_comm 引用，放回会把 net_instance.h 重新带入 base_comm。
// Path -> LinkData 的字段中间态，仅 coll 侧使用，避免在 legacy 目录新增类型/接口。
struct LinkDataFromPathFields {
    PortDeploymentType type{};
    LinkProtocol linkProtocol{};
    RankId localRankId{0};
    RankId remoteRankId{0};
    IpAddress localAddr;
    IpAddress remoteAddr;
    DeviceId localDeviceId{UINT32_MAX};
    DeviceId remoteDeviceId{UINT32_MAX};
    u32 localDieId{0};
    u32 hop{0};
    u8 portGroupSize{1};
    bool fullmesh{false};
    LinkDirection direction{};
};

inline LinkData BuildInvalidLinkData(LinkDirection direction)
{
    LinkDataFromPathFields invalidFields;
    invalidFields.direction = direction;
    LinkData linkData(
        invalidFields.type, invalidFields.linkProtocol, invalidFields.localRankId, invalidFields.remoteRankId,
        invalidFields.localAddr, invalidFields.remoteAddr, invalidFields.localDeviceId, invalidFields.remoteDeviceId);
    linkData.SetHop(invalidFields.hop);
    linkData.SetDirection(invalidFields.direction);
    linkData.SetLocalDieId(invalidFields.localDieId);
    linkData.SetPortGroupSize(invalidFields.portGroupSize);
    linkData.SetFullmesh(invalidFields.fullmesh);
    linkData.UpdateIpAddrWithPCIE();
    return linkData;
}

inline std::shared_ptr<NetInstance::Peer> GetLinkPeer(const std::shared_ptr<NetInstance::Node>& node)
{
    return std::dynamic_pointer_cast<NetInstance::Peer>(node);
}

inline LinkProtocol GetFirstLinkProtocol(const std::set<LinkProtocol>& linkProtocols) { return *linkProtocols.begin(); }

// NetInstance::Path -> LinkData 的转换放在 coll 侧，避免 base_comm 编译期反向依赖 coll。
inline LinkData MakeLinkData(const NetInstance::Path& path)
{
    constexpr u32 MAX_LINK_PATH_NUM = 2;
    if (path.links.empty() || path.links.size() > MAX_LINK_PATH_NUM) {
        HCCL_ERROR("[LinkData][Constructor]path.links.size()[%u] is invalid", static_cast<u32>(path.links.size()));
        return BuildInvalidLinkData(path.direction);
    }

    const bool isSingleLink = path.links.size() == 1;
    const auto& srcLink = path.links[0];
    const auto& targetLink = path.links[isSingleLink ? 0 : 1];
    auto srcPeer = GetLinkPeer(srcLink.GetSourceNode());
    auto targetPeer = GetLinkPeer(targetLink.GetTargetNode());
    auto srcConnIface = srcLink.GetSourceIface();
    auto targetConnIface = targetLink.GetTargetIface();

    LinkDataFromPathFields f;
    f.linkProtocol = GetFirstLinkProtocol(srcLink.GetLinkProtocols());
    f.type = AddrPos2PortDeploymentType(srcConnIface->GetPos(), f.linkProtocol);
    f.localRankId = srcPeer->GetRankId();
    f.remoteRankId = targetPeer->GetRankId();
    f.localDeviceId = srcPeer->GetDeviceId();
    f.remoteDeviceId = targetPeer->GetDeviceId();
    f.localAddr = srcConnIface->GetAddr();
    f.remoteAddr = targetConnIface->GetAddr();
    f.localDieId = srcConnIface->GetLocalDieId();
    f.hop = srcLink.GetHop();
    if (isSingleLink) {
        f.portGroupSize = 1;
        f.fullmesh = true; // 单链路场景，标识为fullmesh
    } else {
        const auto srcPortNum = srcConnIface->GetPorts().size();
        const auto tgtPortNum = targetConnIface->GetPorts().size();
        f.portGroupSize = static_cast<u8>(srcPortNum);
        if (srcPortNum != tgtPortNum) {
            HCCL_WARNING(
                "[LinkData][Constructor]srcConnIface.portGroupSize[%u] "
                "is not equal to targetConnIface.portGroupSize[%u]",
                static_cast<u32>(srcPortNum), static_cast<u32>(tgtPortNum));
        }
        f.fullmesh = false; // 多链路场景，非fullmesh
    }
    f.direction = path.direction;

    LinkData linkData(
        f.type, f.linkProtocol, f.localRankId, f.remoteRankId, f.localAddr, f.remoteAddr, f.localDeviceId,
        f.remoteDeviceId);
    linkData.SetHop(f.hop);
    linkData.SetDirection(f.direction);
    linkData.SetLocalDieId(f.localDieId);
    linkData.SetPortGroupSize(f.portGroupSize);
    linkData.SetFullmesh(f.fullmesh);
    linkData.UpdateIpAddrWithPCIE();
    return linkData;
}

// NetInstance::ConnInterface -> PortData 的转换放在 coll 侧。
inline PortData MakePortData(RankId rankId, const NetInstance::ConnInterface& connIface)
{
    const auto linkProtocols = connIface.GetLinkProtocols();
    const auto linkProtocol = GetFirstLinkProtocol(linkProtocols);
    return PortData(
        rankId, AddrPos2PortDeploymentType(connIface.GetPos(), linkProtocol), LinkProtocol2LinkProtoType(linkProtocol),
        0, connIface.GetAddr());
}

} // namespace Hccl

#endif // LINK_DATA_ADAPTER_H
