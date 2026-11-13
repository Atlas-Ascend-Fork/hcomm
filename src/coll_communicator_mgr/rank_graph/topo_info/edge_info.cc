/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "edge_info.h"
#include "log.h"
#include "string_util.h"
#include "invalid_params_exception.h"
#include "exception_util.h"

namespace Hccl {

const unordered_map<string, LinkProtocol> EdgeInfo::strToLinkProtocol = (unordered_map<string, LinkProtocol>{
    {"UB_CTP", LinkProtocol::UB_CTP},
    {"UB_TP", LinkProtocol::UB_TP},
    {"ROCE", LinkProtocol::ROCE},
    {"HCCS", LinkProtocol::HCCS},
    {"PCIE", LinkProtocol::PCIE},
    {"TCP", LinkProtocol::TCP},
    {"UB_MEM", LinkProtocol::UB_MEM},
    {"UBOE", LinkProtocol::UBOE},
    {"UB_RTP", LinkProtocol::UB_RTP},
    {"UBG", LinkProtocol::UB_RTP}});

const unordered_map<std::string, TopoType> EdgeInfo::strToTopoType = (unordered_map<string, TopoType>{
    {"CLOS", TopoType::CLOS},
    {"1DMESH", TopoType::MESH_1D},
    {"2DMESH", TopoType::MESH_2D},
    {"A3_SERVER", TopoType::A3_SERVER},
    {"A2_AX_SERVER", TopoType::A2_AX_SERVER}});

const unordered_map<string, LinkType> EdgeInfo::strToLinkType
    = (unordered_map<string, LinkType>{{"PEER2PEER", LinkType::PEER2PEER}, {"PEER2NET", LinkType::PEER2NET}});

const unordered_map<string, AddrPosition> EdgeInfo::strToAddrPosition
    = (unordered_map<string, AddrPosition>{{"DEVICE", AddrPosition::DEVICE}, {"HOST", AddrPosition::HOST}});

bool EdgeInfo::operator==(const EdgeInfo& other) const
{
    return netLayer == other.netLayer && linkType == other.linkType && protocols == other.protocols
           && topoType == other.topoType && topoInstId == other.topoInstId && CompareEndpoints(other)
           && position == other.position;
}

// 比较EndpointA和B
bool EdgeInfo::CompareEndpoints(const EdgeInfo& other) const
{
    // 无论什么情况，A=other.A && B=other.B时，可视为相同的连接关系
    if (localA == other.localA && localB == other.localB && localAPorts == other.localAPorts
        && localBPorts == other.localBPorts) {
        return true;
    }

    // 当连接类型是PEER2PEER时，A=other.B && B=other.A时，可视为相同的连接关系
    if (linkType == other.linkType && linkType == LinkType::PEER2PEER && localA == other.localB
        && localAPorts == other.localBPorts && localB == other.localA && localBPorts == other.localAPorts) {
        return true;
    }

    return false;
}

LinkProtocol EdgeInfo::GetLinkProtocol(string str) const
{
    CHK_PRT_THROW(
        strToLinkProtocol.count(str) == 0,
        HCCL_ERROR("[EdgeInfo::%s] string['%s'] is not type of LinkProtocol.", __func__, str.c_str()),
        InvalidParamsException, "string is not type of LinkProtocol");
    return strToLinkProtocol.at(str);
}

TopoType EdgeInfo::GetTopoType(std::string topoTypeStr) const
{
    if (topoTypeStr.empty()) {
        return TopoType::CLOS; // 不填写时，取默认值
    }
    CHK_PRT_THROW(
        strToTopoType.count(topoTypeStr) == 0,
        HCCL_ERROR("[EdgeInfo::%s] string['%s'] is not type of TopoType.", __func__, topoTypeStr.c_str()),
        InvalidParamsException, "string is not type of TopoType");
    return strToTopoType.at(topoTypeStr);
}

LinkType EdgeInfo::GetLinkType(std::string linkTypeStr) const
{
    CHK_PRT_THROW(
        strToLinkType.count(linkTypeStr) == 0,
        HCCL_ERROR("[EdgeInfo::%s] string['%s'] is not type of LinkType.", __func__, linkTypeStr.c_str()),
        InvalidParamsException, "string is not type of LinkType");
    return strToLinkType.at(linkTypeStr);
}

AddrPosition EdgeInfo::GetAddrPosition(string str) const
{
    if (str.empty()) {
        HCCL_WARNING("[EdgeInfo::%s] position is null, [default]position=DEVICE", __func__);
        return AddrPosition::DEVICE; //  默认取值为DEVICE
    }
    CHK_PRT_THROW(
        strToAddrPosition.count(str) == 0,
        HCCL_ERROR("[EdgeInfo::%s] string['%s'] is not type of AddrPosition.", __func__, str.c_str()),
        InvalidParamsException, "string is not type of AddrPosition");
    return strToAddrPosition.at(str);
}

std::string EdgeInfo::Describe() const
{
    stringstream protocolStr;
    protocolStr << "[";
    for (auto it = protocols.begin(); it != protocols.end(); ++it) {
        if (it != protocols.begin()) {
            protocolStr << ", ";
        }
        protocolStr << it->Describe();
    }
    protocolStr << "]";

    string localAPortsStr = DescribePorts(localAPorts);
    string localBPortsStr = DescribePorts(localBPorts);

    std::string description = "EdgeInfo{";
    description += StringFormat("netLayer=%u", netLayer);
    description += StringFormat("topoType=%s", topoType.Describe().c_str());
    description += StringFormat(", topoInstanceId=%u", topoInstId);
    description += StringFormat(", protocols=%s", protocolStr.str().c_str());
    description += StringFormat(", linkType=%s", linkType.Describe().c_str());
    description += StringFormat(", localA=%u", localA);
    description += StringFormat(", localAPortsStr=%s", localAPortsStr.c_str());
    description += StringFormat(", localB=%u", localB);
    description += StringFormat(", localBPortsStr=%s", localBPortsStr.c_str());
    description += StringFormat(", position=%s", position.Describe().c_str());
    description += "}";
    return description;
}

std::string EdgeInfo::DescribePorts(std::set<std::string> ports) const
{
    stringstream portsStr;
    portsStr << "[";
    for (auto it = ports.begin(); it != ports.end(); ++it) {
        if (it != ports.begin()) {
            portsStr << ", ";
        }
        portsStr << *it;
    }
    portsStr << "]";
    return portsStr.str();
}

void EdgeInfo::GetBinStream(BinaryStream& binaryStream) const
{
    binaryStream << netLayer << static_cast<u32>(linkType) << static_cast<u32>(topoType) << topoInstId;
    binaryStream << protocols.size();
    for (LinkProtocol protocol : protocols) {
        binaryStream << static_cast<u32>(protocol);
    }

    binaryStream << localA << localB;

    binaryStream << localAPorts.size();
    for (string port : localAPorts) {
        binaryStream << port;
    }

    binaryStream << localBPorts.size();
    for (string port : localBPorts) {
        binaryStream << port;
    }

    binaryStream << static_cast<u32>(position);
}

EdgeInfo::EdgeInfo(BinaryStream& binaryStream)
{
    binaryStream >> netLayer;
    u32 linkTypeTmp;
    binaryStream >> linkTypeTmp;
    linkType = static_cast<LinkType::Value>(linkTypeTmp);
    u32 topoTypeTmp;
    binaryStream >> topoTypeTmp;
    topoType = static_cast<TopoType::Value>(topoTypeTmp);
    binaryStream >> topoInstId;
    size_t protocolSize;
    binaryStream >> protocolSize;
    protocols.clear();
    for (size_t i = 0; i < protocolSize; i++) {
        u32 protocolTmp;
        binaryStream >> protocolTmp;
        LinkProtocol protocol = static_cast<LinkProtocol::Value>(protocolTmp);
        protocols.emplace(protocol);
    }

    binaryStream >> localA >> localB;

    size_t localAPortsSize;
    binaryStream >> localAPortsSize;
    localAPorts.clear();
    for (size_t i = 0; i < localAPortsSize; i++) {
        string port;
        binaryStream >> port;
        localAPorts.emplace(port);
    }

    size_t localBPortsSize;
    binaryStream >> localBPortsSize;
    localBPorts.clear();
    for (size_t i = 0; i < localBPortsSize; i++) {
        string port;
        binaryStream >> port;
        localBPorts.emplace(port);
    }

    u32 positionTmp;
    binaryStream >> positionTmp;
    position = static_cast<AddrPosition::Value>(positionTmp);
}

} // namespace Hccl
