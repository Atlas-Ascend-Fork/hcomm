/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_VIRTUAL_TOPO_H
#define HCCLV2_VIRTUAL_TOPO_H

#include <cassert>
#include <map>
#include <set>
#include <cstdint>
#include "port.h"
#include "iterator.h"
#include "dev_type.h"
#include "log.h"
namespace Hccl {

using namespace std;

MAKE_ENUM(PeerType, CPU, NPU)

class LinkData {
public:
    // 构造函数不对外开发，LinkData只能由Link生成
    LinkData(BasePortType portType, RankId localRankId, RankId remoteRankId, u32 localPortId, u32 remotePortId)
        : type(portType.GetType()),
          linkProtocol_(ConnProto2LinkProtocol(portType.GetProto())),
          localRankId_(localRankId),
          remoteRankId_(remoteRankId),
          localPortId_(localPortId),
          remotePortId_(remotePortId) {};
    LinkData(
        PortDeploymentType portDeploymentType, LinkProtocol linkProtocol, RankId localRankId, RankId remoteRankId,
        IpAddress localAddr, IpAddress remoteAddr, u32 devicePhyId = 0, u32 remoteDevicePhyId = 0, u32 reuseIdx = 0)
        : type(portDeploymentType),
          linkProtocol_(linkProtocol),
          localRankId_(localRankId),
          remoteRankId_(remoteRankId),
          localAddr_(localAddr),
          remoteAddr_(remoteAddr),
          localDeviceId_(devicePhyId),
          remoteDeviceId_(remoteDevicePhyId),
          reuseIdx_(reuseIdx) {};

    explicit LinkData(vector<char>& data);

    std::vector<char> GetUniqueId() const;

    bool operator==(const LinkData& rhs) const
    {
        return type == rhs.type && linkProtocol_ == rhs.linkProtocol_ && localRankId_ == rhs.localRankId_
               && remoteRankId_ == rhs.remoteRankId_ && localAddr_ == rhs.localAddr_ && remoteAddr_ == rhs.remoteAddr_
               && hop == rhs.hop && direction == rhs.direction && portGroupSize == rhs.portGroupSize
               && fullmesh == rhs.fullmesh && reuseIdx_ == rhs.reuseIdx_;
    }

    bool operator!=(const LinkData& rhs) const { return !(rhs == *this); }

    bool operator<(const LinkData& rhs) const
    {
        if (type < rhs.type) {
            return true;
        }
        if (rhs.type < type) {
            return false;
        }
        if (linkProtocol_ < rhs.linkProtocol_) {
            return true;
        }
        if (rhs.linkProtocol_ < linkProtocol_) {
            return false;
        }
        if (localRankId_ < rhs.localRankId_) {
            return true;
        }
        if (rhs.localRankId_ < localRankId_) {
            return false;
        }
        if (remoteRankId_ < rhs.remoteRankId_) {
            return true;
        }
        if (rhs.remoteRankId_ < remoteRankId_) {
            return false;
        }
        if (localAddr_ < rhs.localAddr_) {
            return true;
        }
        if (rhs.localAddr_ < localAddr_) {
            return false;
        }
        if (remoteAddr_ < rhs.remoteAddr_) {
            return true;
        }
        if (rhs.remoteAddr_ < remoteAddr_) {
            return false;
        }
        if (hop < rhs.hop) {
            return true;
        }
        if (rhs.hop < hop) {
            return false;
        }
        if (direction < rhs.direction) {
            return true;
        }
        if (rhs.direction < direction) {
            return false;
        }
        if (rhs.portGroupSize < portGroupSize) {
            return false;
        }
        if (fullmesh == false && rhs.fullmesh == true) {
            return true;
        }
        if (rhs.fullmesh == false && fullmesh == true) {
            return false;
        }
        if (localPortId_ < rhs.localPortId_) {
            return true;
        }
        if (rhs.localPortId_ < localPortId_) {
            return false;
        }
        if (reuseIdx_ < rhs.reuseIdx_) {
            return true;
        }
        if (rhs.reuseIdx_ < reuseIdx_) {
            return false;
        }
        return remotePortId_ < rhs.remotePortId_;
    }

    string Describe() const
    {
        return StringFormat(
            "LinkData:type=%s, protocol=%s, localRankId=%d, localAddr=%s, remoteRankId=%d, "
            "remoteAddr=%s, reuseIdx=%u",
            type.Describe().c_str(), linkProtocol_.Describe().c_str(), localRankId_, localAddr_.Describe().c_str(),
            remoteRankId_, remoteAddr_.Describe().c_str(), reuseIdx_);
    };

    PortData GetLocalPort() const
    {
        return {localRankId_, type, LinkProtocol2LinkProtoType(linkProtocol_), localPortId_, localAddr_};
    };

    PortData GetRemotePort() const
    {
        return {remoteRankId_, type, LinkProtocol2LinkProtoType(linkProtocol_), remotePortId_, remoteAddr_};
    };

    bool IsSymetric(const LinkData& rhs) const
    {
        return (type == rhs.type) && (linkProtocol_ == rhs.linkProtocol_) && (localRankId_ == rhs.remoteRankId_)
               && (remoteRankId_ == rhs.localRankId_) && (localAddr_ == rhs.remoteAddr_)
               && (remoteAddr_ == rhs.localAddr_) && (hop == rhs.hop) && (direction == rhs.direction);
    };

    const PortDeploymentType& GetType() const { return type; };

    const LinkProtocol& GetLinkProtocol() const { return linkProtocol_; }

    u32 GetHop() const { return hop; }

    LinkDirection GetDirection() const { return direction; }

    RankId GetLocalRankId() const { return localRankId_; };

    RankId GetRemoteRankId() const { return remoteRankId_; };

    DeviceId GetRemoteDeviceId() const { return remoteDeviceId_; };

    u32 GetLocalPortId() const { return localPortId_; };

    u32 GetRemotePortId() const { return remotePortId_; };

    const IpAddress& GetLocalAddr() const { return localAddr_; };

    const IpAddress& GetRemoteAddr() const { return remoteAddr_; };

    void SetRemoteAddr(const IpAddress& remoteAddr) { remoteAddr_ = remoteAddr; }
    // LinkData 构造迁移到 coll 侧的临时桥接，legacy 目录日落后随 setter 一并删除。
    void SetHop(u32 value) { hop = value; }
    void SetDirection(LinkDirection value) { direction = value; }
    void SetLocalDieId(u32 value) { localDieId_ = value; }
    void SetPortGroupSize(u8 value) { portGroupSize = value; }
    void SetFullmesh(bool value) { fullmesh = value; }

    u32 GetLocalDieId() const { return localDieId_; };

    u8 GetPortGroupSize() const { return portGroupSize; };

    bool Readable() const { return readable; };

    bool Writable() const { return writable; };
    void UpdateIpAddrWithPCIE();

    bool GetFullmesh() const { return fullmesh; };

    std::string GetReuseIdx() const { return std::to_string(reuseIdx_); };

    u8 GetLocalIpIndex() const { return localIpIndex_; }
    u8 GetRemoteIpIndex() const { return remoteIpIndex_; }
    void SetLocalIpIndex(u8 idx) { localIpIndex_ = idx; }
    void SetRemoteIpIndex(u8 idx) { remoteIpIndex_ = idx; }

private:
    PortDeploymentType type;
    LinkProtocol linkProtocol_;
    RankId localRankId_{0};
    RankId remoteRankId_{0};
    u32 localPortId_{0};
    u32 remotePortId_{0};
    IpAddress localAddr_;
    IpAddress remoteAddr_;
    bool readable{true};
    bool writable{true};
    u32 hop{0};
    LinkDirection direction{LinkDirection::BOTH};
    u32 localDieId_{};
    u8 portGroupSize{1};
    DeviceId localDeviceId_{UINT32_MAX};
    DeviceId remoteDeviceId_{UINT32_MAX};
    bool fullmesh{false}; // 标识是否为全互联单链路场景
    u32 reuseIdx_{0};     // socket复用idx，加在socket建链tag后面
    u8 localIpIndex_{0};  // IP逻辑编号，用于hccpTag拼接替代IP字符串
    u8 remoteIpIndex_{0};
};
} // namespace Hccl

namespace std {

template <>
class hash<Hccl::LinkData> {
public:
    size_t operator()(const Hccl::LinkData& linkData) const
    {
        auto typeHash = hash<uint8_t>{}(linkData.GetType());
        auto linkProtoHash = hash<uint8_t>{}(linkData.GetLinkProtocol());
        auto localRankIdHash = hash<Hccl::RankId>{}(linkData.GetLocalRankId());
        auto remoteRankIdHash = hash<Hccl::RankId>{}(linkData.GetRemoteRankId());
        auto localPortIdHash = hash<u32>{}(linkData.GetLocalPortId());
        auto remotePortIdHash = hash<u32>{}(linkData.GetRemotePortId());
        auto localAddrHash = hash<Hccl::IpAddress>{}(linkData.GetLocalAddr());
        auto remoteAddrHash = hash<Hccl::IpAddress>{}(linkData.GetRemoteAddr());
        auto portGrpSizeHash = hash<uint8_t>{}(linkData.GetPortGroupSize());
        auto fullmeshHash = hash<bool>{}(linkData.GetFullmesh());

        return Hccl::HashCombine(
            {typeHash, linkProtoHash, localRankIdHash, remoteRankIdHash, localPortIdHash, remotePortIdHash,
             localAddrHash, remoteAddrHash, portGrpSizeHash, fullmeshHash});
    }
};
} // namespace std

#endif // HCCLV2_VIRTUAL_TOPO_H
