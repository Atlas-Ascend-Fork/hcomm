/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "phy_topo.h"
#include "exception_util.h"
#include "internal_exception.h"
namespace hccl {

std::unique_ptr<PhyTopo>& PhyTopo::GetInstance()
{
    static std::unique_ptr<PhyTopo> topo = std::make_unique<PhyTopo>();
    return topo;
}

void PhyTopo::InitFinish() { initFlag = true; }

bool PhyTopo::IsInitFinished() const { return initFlag; }

void PhyTopo::Clear()
{
    topos.clear();
    initFlag = false;
}

HcclResult PhyTopo::AddTopoGraph(const u32 netLayer, std::shared_ptr<hccl::Graph<PhyTopo::Node, PhyTopo::Link>> topo)
{
    if (!initFlag) {
        topos[netLayer] = topo;
        HCCL_DEBUG("[PhyTopo]add topo success, netLayer [%u], topo size is [%zu]", netLayer, topos.size());
    } else {
        HCCL_ERROR("[%s] PhyTopo AddTopoGraph fail. PhyTopo has been initialized and cannot be changed.", __func__);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

std::shared_ptr<hccl::Graph<PhyTopo::Node, PhyTopo::Link>> PhyTopo::GetTopoGraph(const u32 netLayer) const
{
    if (topos.find(netLayer) == topos.end()) {
        return nullptr;
    }
    return topos.at(netLayer);
}

bool PhyTopo::IsNetLayerExisted(const u32 netLayer) const
{
    if (topos.find(netLayer) == topos.end()) {
        return false;
    }
    return true;
}

void PhyTopo::Dump() const
{
    HCCL_DEBUG("PhyTopo Dump:");
    for (auto& pair : topos) {
        HCCL_DEBUG("netLayer[%u]:", pair.first);
        HCCL_DEBUG("nodes:");
        std::set<Hccl::NodeId> nodeIds{};
        pair.second->TraverseNode([&](Hccl::NodeId nodeId, std::shared_ptr<PhyTopo::Node> node) {
            nodeIds.insert(nodeId);
            HCCL_DEBUG("%s", node->Describe().c_str());
        });
        HCCL_DEBUG("links:");
        for (Hccl::NodeId nodeId : nodeIds) {
            pair.second->TraverseEdge(nodeId, [&](std::shared_ptr<Link> link) {
                HCCL_DEBUG("%s", link->Describe().c_str());
            });
        }
    }
}

PhyTopo::ConnInterface::ConnInterface(
    const std::set<std::string> inputPorts, const Hccl::AddrPosition inputPos, const Hccl::LinkType inputLinkType,
    const std::set<Hccl::LinkProtocol> inputLinkProtocols)
    : ports(inputPorts),
      pos(inputPos),
      linkType(inputLinkType),
      linkProtocols(inputLinkProtocols)
{}

std::set<std::string> PhyTopo::ConnInterface::GetPorts() const { return ports; }

Hccl::AddrPosition PhyTopo::ConnInterface::GetPos() const { return pos; }

Hccl::LinkType PhyTopo::ConnInterface::GetLinkType() const { return linkType; }

std::set<Hccl::LinkProtocol> PhyTopo::ConnInterface::GetLinkProtocols() const { return linkProtocols; }

std::string PhyTopo::ConnInterface::Describe() const
{
    std::string portsStr;
    for (auto it = ports.begin(); it != ports.end(); ++it) {
        if (it != ports.begin()) {
            portsStr += ",";
        }
        portsStr += *it;
    }
    std::string protocolStr;
    for (auto it = linkProtocols.begin(); it != linkProtocols.end(); ++it) {
        if (!protocolStr.empty()) {
            protocolStr += ", ";
        }
        protocolStr += it->Describe();
    }

    return Hccl::StringFormat(
        "ConnInterface[ports={%s}, pos=%s, protocols={%s}, linkType=%s]", portsStr.c_str(), pos.Describe().c_str(),
        protocolStr.c_str(), linkType.Describe().c_str());
}

bool PhyTopo::ConnInterface::operator==(const ConnInterface& rhs) const
{
    return ports == rhs.ports && pos == rhs.pos && linkType == rhs.linkType && linkProtocols == rhs.linkProtocols;
}

bool PhyTopo::ConnInterface::operator!=(const ConnInterface& rhs) const { return !(rhs == *this); }

PhyTopo::Node::Node(const PhyTopo::Node::NodeType inputType) : type(inputType) {}

PhyTopo::Node::NodeType PhyTopo::Node::GetType() const { return type; }

void PhyTopo::Node::AddConnInterface(const std::shared_ptr<PhyTopo::ConnInterface>& interface)
{
    for (const auto& iface : interfaces) {
        if (*iface == *interface) {
            HCCL_WARNING("[PhyTopo][Node][AddConnInterface] %s has existed.", interface->Describe().c_str());
            return;
        }
    }
    interfaces.emplace_back(interface);
}

PhyTopo::Node::IfaceIterator PhyTopo::Node::IterIfaces() const { return PhyTopo::Node::IfaceIterator(interfaces); }

std::string PhyTopo::Node::Describe() const { return "PhyTopo::Node[]"; }

PhyTopo::Peer::Peer(const Hccl::LocalId localId) : Node(PhyTopo::Node::NodeType::PEER), localId(localId) {}

Hccl::LocalId PhyTopo::Peer::GetLocalId() const { return localId; }

Hccl::NodeId PhyTopo::Peer::GetId(const Hccl::LocalId localId) { return static_cast<Hccl::NodeId>(localId); }

std::string PhyTopo::Peer::Describe() const { return Hccl::StringFormat("PhyTopo::Peer[localId=%u]", localId); }

PhyTopo::Fabric::Fabric() : Node(PhyTopo::Node::NodeType::FABRIC) {}

constexpr s32 FABRIC_ID_OFFSET = 32;
Hccl::NodeId PhyTopo::Fabric::GetId()
{
    Hccl::NodeId res = static_cast<Hccl::NodeId>(0);
    res |= (1ULL << FABRIC_ID_OFFSET);
    return res;
}

std::string PhyTopo::Fabric::Describe() const { return Hccl::StringFormat("PhyTopo::Fabric[NodeId=%llu]", GetId()); }

PhyTopo::Link::Link(
    std::shared_ptr<PhyTopo::Node> inputSource, std::shared_ptr<PhyTopo::Node> inputTarget,
    const LinkAttributes& properties, const Hccl::TopoType inputTopoType, const u32 inputTopoInstId)
    : sourceIface(nullptr),
      targetIface(nullptr),
      source(inputSource),
      target(inputTarget),
      linkProtocols(properties.protocols),
      linkType(properties.linktype),
      direction(Hccl::LinkDirection::BOTH),
      topoType(inputTopoType),
      topoInstId(inputTopoInstId),
      hop{1}
{}

void PhyTopo::Link::SetSourceIface(std::shared_ptr<PhyTopo::ConnInterface> inputSourceIface)
{
    sourceIface = inputSourceIface;
}

void PhyTopo::Link::SetTargetIface(std::shared_ptr<PhyTopo::ConnInterface> inputTargetIface)
{
    targetIface = inputTargetIface;
}

Hccl::LinkType PhyTopo::Link::GetType() const { return linkType; }

std::set<Hccl::LinkProtocol> PhyTopo::Link::GetLinkProtocols() const { return linkProtocols; }

Hccl::LinkDirection PhyTopo::Link::GetLinkDirection() const { return direction; }

Hccl::TopoType PhyTopo::Link::GetTopoType() const { return topoType; }

u32 PhyTopo::Link::GetTopoInstId() const { return topoInstId; }

u32 PhyTopo::Link::GetHop() const { return hop; }

std::shared_ptr<PhyTopo::ConnInterface> PhyTopo::Link::GetSourceIFace() { return sourceIface; }

std::shared_ptr<PhyTopo::ConnInterface> PhyTopo::Link::GetTargetIFace() { return targetIface; }

std::shared_ptr<PhyTopo::Node> PhyTopo::Link::GetSourceNode() { return source; }

std::shared_ptr<PhyTopo::Node> PhyTopo::Link::GetTargetNode() { return target; }

std::string PhyTopo::Link::Describe() const
{
    std::stringstream iFace;
    if (sourceIface != nullptr) {
        iFace << ", sourceIface=" << sourceIface->Describe();
    }
    if (targetIface != nullptr) {
        iFace << ", targetIface=" << targetIface->Describe();
    }

    // 将 linkProtocol 转换为字符串
    std::string protocolStr;
    for (auto it = linkProtocols.begin(); it != linkProtocols.end(); ++it) {
        if (!protocolStr.empty()) {
            protocolStr += ", ";
        }
        protocolStr += it->Describe();
    }

    return Hccl::StringFormat(
        "PhyTopo::Link[type=%s, protocol=%s, source=%s, target=%s%s, topoInstId=%u, topoType=%d]",
        linkType.Describe().c_str(), protocolStr.c_str(), source->Describe().c_str(), target->Describe().c_str(),
        iFace.str().c_str(), topoInstId, topoType);
}
} // namespace hccl
