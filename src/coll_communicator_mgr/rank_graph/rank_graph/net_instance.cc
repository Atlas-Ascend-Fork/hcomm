/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "net_instance.h"
#include "exception_util.h"

namespace Hccl {

using namespace std;

NetInstance::NetInstance(const u32 netLayer, const string& netInstId, const NetType netType)
{
    this->netLayer = netLayer;
    this->netInstId = netInstId;
    this->netType = netType;
}

u32 NetInstance::GetNetLayer() const { return netLayer; }

string NetInstance::GetNetInstId() const { return netInstId; }

NetType NetInstance::GetNetType() const { return netType; }

set<RankId> NetInstance::GetRankIds() const { return rankIds; }

u32 NetInstance::GetRankSize() const { return rankIds.size(); }

bool NetInstance::HasNode(const NodeId nodeId) const { return vGraph.HasNode(nodeId); }

const std::unordered_map<RankId, std::shared_ptr<NetInstance::Peer>>& NetInstance::GetPeers() const { return peers; }

const std::vector<std::shared_ptr<NetInstance::Fabric>>& NetInstance::GetFabrics() const { return fabrics; }

Graph<NetInstance::Node, NetInstance::Link>& NetInstance::GetGraph() { return vGraph; }

void NetInstance::AddRankId(const RankId rankId)
{
    rankIds.insert(rankId);
    HCCL_DEBUG("[NetInstance::AddRankId] add rank id [%d] to %s", rankId, this->Describe().c_str());
}

HcclResult NetInstance::AddNode(const shared_ptr<Node>& node)
{
    NetInstance::Node::NodeType nodeType = node->GetType();
    if (nodeType == NetInstance::Node::NodeType::PEER) {
        CHK_RET(AddPeer(dynamic_pointer_cast<NetInstance::Peer>(node)));
    } else if (nodeType == NetInstance::Node::NodeType::FABRIC) {
        CHK_RET(AddFabric(dynamic_pointer_cast<NetInstance::Fabric>(node)));
    } else {
        HCCL_ERROR(
            "[%s] failed to add %s to %s, only PEER or FABRIC type node can be added.", __func__,
            node->Describe().c_str(), this->Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult NetInstance::AddPeer(const shared_ptr<Peer>& peer)
{
    if (netLayer == 0 && localIdsMap.find(peer->GetLocalId()) != localIdsMap.end()) {
        HCCL_ERROR(
            "[%s] when netLayer is 0, local id[%u] is repeat. rank id [%u], netInstId[%s]", __func__,
            peer->GetLocalId(), peer->GetRankId(), netInstId.c_str());
        return HCCL_E_PARA;
    }
    localIdsMap.insert({peer->GetLocalId(), peer->GetRankId()});

    peers[peer->GetRankId()] = peer;
    vGraph.AddNode(peer->GetNodeId(), peer);

    HCCL_DEBUG("[NetInstance::AddPeer] add %s to %s", peer->Describe().c_str(), this->Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult NetInstance::AddFabric(const shared_ptr<NetInstance::Fabric>& fabric)
{
    if (netType != NetType::CLOS && netType != NetType::TOPO_FILE_DESC) {
        HCCL_ERROR(
            "[%s] failed to add %s to %s, only CLOS type NetInstance can add Fabrics.", __func__,
            fabric->Describe().c_str(), this->Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }

    NodeId fabricId = fabric->GetNodeId();
    fabrics.emplace_back(fabric);
    vGraph.AddNode(fabricId, fabric);

    HCCL_DEBUG("[NetInstance::AddFabric] add %s to %s", fabric->Describe().c_str(), this->Describe().c_str());
    return HCCL_SUCCESS;
}

void NetInstance::AddLink(const shared_ptr<NetInstance::Link>& link)
{
    NodeId srcNodeId = link->GetSourceNode()->GetNodeId();
    NodeId dstNodeId = link->GetTargetNode()->GetNodeId();

    bool hasLink = false;
    vGraph.TraverseEdge(srcNodeId, dstNodeId, [&](shared_ptr<NetInstance::Link> edge) {
        if (*edge == *link) {
            hasLink = true;
            return;
        }
    });

    if (hasLink) {
        HCCL_WARNING(
            "[NetInstance::AddLink] unsuccessful in adding %s to %s, "
            "the fabric group already has the same link.",
            link->Describe().c_str(), this->Describe().c_str());
        return;
    }

    vGraph.AddEdge(srcNodeId, dstNodeId, link);

    HCCL_DEBUG("[NetInstance::AddLink] add %s to %s", link->Describe().c_str(), this->Describe().c_str());
}

void NetInstance::DeleteLink(const NodeId srcNodeId, const NodeId dstNodeId)
{
    HCCL_RUN_INFO("[NetInstance::DeleteLink] delete %lu -> %lu", srcNodeId, dstNodeId);
    vGraph.DeleteEdge(srcNodeId, dstNodeId);
    vGraph.DeleteEdge(dstNodeId, srcNodeId);
}

void NetInstance::UpdateTopoInst(u32 topoInstId, TopoType topoType, RankId rankId)
{
    auto it = topoInsts_.find(topoInstId);
    if (it != topoInsts_.end()) {
        TopoInstance& existingInst = *it->second;
        existingInst.ranks.insert(rankId);
    } else {
        // 创建新的TopoInstance
        TopoInstance newInst;
        newInst.topoInstId = topoInstId;
        newInst.topoType = topoType;
        newInst.ranks.insert(rankId);
        topoInsts_.emplace(topoInstId, std::make_shared<TopoInstance>(std::move(newInst)));
    }
}

void NetInstance::GetTopoInstsByLayer(std::vector<u32>& topoInsts, u32& topoInstNum) const
{
    for (const auto& entry : topoInsts_) {
        topoInsts.push_back(entry.first);
    }

    topoInstNum = static_cast<u32>(topoInsts.size());
}

HcclResult NetInstance::GetTopoType(const u32 topoInstId, TopoType& topoType) const
{
    auto it = topoInsts_.find(topoInstId);
    if (it != topoInsts_.end()) {
        const std::shared_ptr<TopoInstance>& topoInstPtr = it->second;
        topoType = topoInstPtr->topoType;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[NetInstance::GetTopoType] Failed to find TopoInstance with ID: %u", topoInstId);
    return HCCL_E_PARA;
}

HcclResult NetInstance::GetRanksByTopoInst(const u32 topoInstId, std::vector<u32>& ranks, u32& rankNum) const
{
    auto it = topoInsts_.find(topoInstId);
    if (it != topoInsts_.end()) {
        const std::shared_ptr<TopoInstance>& topoInstPtr = it->second;
        ranks.assign(topoInstPtr->ranks.begin(), topoInstPtr->ranks.end());
        rankNum = static_cast<u32>(ranks.size());
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[NetInstance::GetRanksByTopoInst] Failed to find ranks with ID: %u", topoInstId);
    return HCCL_E_PARA;
}

string NetInstance::Describe() const
{
    return StringFormat(
        "NetInstance[ID=%s, Level=%u, FabType=%s, RankIds_Size=%zu]", netInstId.c_str(), netLayer,
        netType.Describe().c_str(), rankIds.size());
}

bool CheckPortGroupSize(u32 netLayer, NetInstance::Link& srcLink, NetInstance::Link& dstLink)
{
    auto srcConnIface = srcLink.GetSourceIface();
    auto targetConnIface = dstLink.GetTargetIface();
    auto srcPortGroupSize = static_cast<u8>(srcConnIface->GetPortNum());
    auto tgtPortGroupSize = static_cast<u8>(targetConnIface->GetPortNum());
    if (srcPortGroupSize != tgtPortGroupSize) {
        auto srcPeer = srcLink.GetSourceNode();
        auto targetPeer = dstLink.GetTargetNode();
        auto localAddr = srcConnIface->GetAddr();
        auto remoteAddr = targetConnIface->GetAddr();
        auto localRankId = std::dynamic_pointer_cast<NetInstance::Peer>(srcPeer)->GetRankId();
        auto remoteRankId = std::dynamic_pointer_cast<NetInstance::Peer>(targetPeer)->GetRankId();
        HCCL_WARNING(
            "[GetPaths][CheckPortGroupSize] portGroupSize is not equal => src[%u], target[%u]. "
            "LocatedInfo: NetLayer[%u], localRank[%u], rmtRank[%u], localAddr[%s], rmtAddr[%s]",
            srcPortGroupSize, tgtPortGroupSize, netLayer, localRankId, remoteRankId, localAddr.Describe().c_str(),
            remoteAddr.Describe().c_str());
        return false;
    }
    return true;
}

vector<NetInstance::Path> InnerNetInstance::GetPaths(const RankId srcRankId, const RankId dstRankId) const
{
    vector<NetInstance::Path> paths;
    if (peers.count(srcRankId) == 0 || peers.count(dstRankId) == 0) {
        HCCL_WARNING(
            "[InnerNetInstance::GetPaths] srcRankId[%d] or dstRankId[%d] not exist in netInstance, "
            "netLayer[%u], netInstId[%s]",
            srcRankId, dstRankId, netLayer, netInstId.c_str());
        return paths;
    }
    NodeId srcPeerId = peers.at(srcRankId)->GetNodeId();
    NodeId dstPeerId = peers.at(dstRankId)->GetNodeId();
    // 1. 获取边
    vGraph.TraverseEdge(srcPeerId, dstPeerId, [&](shared_ptr<NetInstance::Link> edge) {
        NetInstance::Path path;
        path.links = {*edge};
        path.direction = edge->GetLinkDirection();
        paths.emplace_back(path);
        HCCL_DEBUG(
            "[InnerNetInstance::GetPaths] netLayer[%u], from src[%s] to dst[%s] get path.", netLayer,
            peers.at(srcRankId)->Describe().c_str(), peers.at(dstRankId)->Describe().c_str());
        HCCL_DEBUG(
            "[InnerNetInstance::GetPaths] netLayer[%u], srcRankId[%u], dstRankId[%u], path[%s]", netLayer, srcRankId,
            dstRankId, path.links[0].Describe().c_str());
    });

    // 2. 通过 fabric 的路径
    for (auto& fabric : fabrics) {
        NodeId fabricId = fabric->GetNodeId();

        // 所有 src -> fabric 的链路
        vector<NetInstance::Link> srcToFabricLinks;
        vGraph.TraverseEdge(srcPeerId, fabricId, [&](shared_ptr<NetInstance::Link> edge) {
            srcToFabricLinks.push_back(*edge);
            return;
        });

        // 所有 fabric -> dst 的链路
        vector<NetInstance::Link> fabricToDstLinks;
        vGraph.TraverseEdge(fabricId, dstPeerId, [&](shared_ptr<NetInstance::Link> edge) {
            fabricToDstLinks.push_back(*edge);
            return;
        });

        if (!srcToFabricLinks.empty() && !fabricToDstLinks.empty()) {
            for (auto& srcLink : srcToFabricLinks) {
                for (auto& dstLink : fabricToDstLinks) {
                    if (!CheckPortGroupSize(netLayer, srcLink, dstLink)) {
                        continue;
                    }
                    NetInstance::Path path;
                    path.links = {srcLink, dstLink};
                    paths.emplace_back(path);
                }
            }
        } else {
            HCCL_WARNING(
                "[NetInstance::GetPaths] netLayer[%u], srcRankId[%d], dstRankId[%d], netInstId[%s], "
                "from src[%s] to dst[%s] link via fabric[%s] not found.",
                netLayer, srcRankId, dstRankId, netInstId.c_str(), peers.at(srcRankId)->Describe().c_str(),
                peers.at(dstRankId)->Describe().c_str(), fabric->Describe().c_str());
        }
    }

    return paths;
}

namespace {
    // NodeId 位编码：低 32 位为 rankId/fabricId，高 32 位为节点类型标记（Peer 为 0、Fabric 为 1，
    // 见 Peer/Fabric::GenerateNodeId）；编码若调整须同步修改本常量与两处 GenerateNodeId
    constexpr u32 NODE_TYPE_MARK_SHIFT = 32;
    bool IsPeerNodeId(NodeId nodeId) { return (nodeId >> NODE_TYPE_MARK_SHIFT) == 0; }
} // namespace

void InnerNetInstance::GetReachableRanks(
    const RankId srcRankId, std::map<LinkProtocol, std::set<RankId>>& reachableByProtocol) const
{
    if (peers.count(srcRankId) == 0) {
        return; // 与 GetPaths 的存在性检查等价
    }
    NodeId srcPeerId = peers.at(srcRankId)->GetNodeId();
    // 1. 直达边；2. fabric 路径（协议仅取 peer2net 边）
    CollectDirectEdgeReachableRanks(srcRankId, srcPeerId, reachableByProtocol);
    CollectFabricReachableRanks(srcRankId, srcPeerId, reachableByProtocol);
}

// 直达边批量收集：src 出边中目标为 Peer 的边，并入该边协议全集
void InnerNetInstance::CollectDirectEdgeReachableRanks(
    const RankId srcRankId, const NodeId srcPeerId, std::map<LinkProtocol, std::set<RankId>>& reachableByProtocol) const
{
    const auto* outEdges = vGraph.GetOutEdges(srcPeerId);
    if (outEdges != nullptr) {
        for (const auto& [dstNodeId, edgeVec] : *outEdges) {
            if (!IsPeerNodeId(dstNodeId) || peers.count(static_cast<RankId>(dstNodeId)) == 0) {
                continue; // fabric 节点或未知节点，走 fabric 路径
            }
            RankId dstRank = static_cast<RankId>(dstNodeId);
            if (dstRank == srcRankId) {
                continue; // 防御：旧路径从不收集自己（外层跳过 self）
            }
            for (const auto& edge : edgeVec) {
                for (LinkProtocol proto : edge->GetLinkProtocols()) {
                    reachableByProtocol[proto].insert(dstRank);
                }
            }
        }
    }
}

void InnerNetInstance::CollectFabricReachableRanks(
    const RankId srcRankId, const NodeId srcPeerId, std::map<LinkProtocol, std::set<RankId>>& reachableByProtocol) const
{
    for (const auto& fabric : fabrics) {
        NodeId fabricNodeId = fabric->GetNodeId();
        const auto* sLinks = vGraph.FindEdges(srcPeerId, fabricNodeId);
        if (sLinks == nullptr || sLinks->empty()) {
            continue;
        }
        // 预聚合 F->peer 边：dLink 目标侧端口数 -> 可达 rank 集
        std::unordered_map<u32, std::set<RankId>> peersByPortNum;
        const auto* fabOut = vGraph.GetOutEdges(fabricNodeId);
        if (fabOut != nullptr) {
            for (const auto& [dstNodeId, edgeVec] : *fabOut) {
                if (!IsPeerNodeId(dstNodeId) || peers.count(static_cast<RankId>(dstNodeId)) == 0) {
                    continue;
                }
                RankId dstRank = static_cast<RankId>(dstNodeId);
                if (dstRank == srcRankId) {
                    continue;
                }
                for (const auto& dLink : edgeVec) {
                    const auto tgtIface = dLink->GetTargetIface();
                    if (tgtIface == nullptr) {
                        continue;
                    }
                    peersByPortNum[tgtIface->GetPortNum()].insert(dstRank);
                }
            }
        }
        // 配对命中的 sLink：其协议桶并入该端口数下全部可达 rank（Inner 取全部 sLink×dLink 组合并集）
        for (const auto& sLink : *sLinks) {
            const auto srcIface = sLink->GetSourceIface();
            if (srcIface == nullptr) {
                continue;
            }
            auto it = peersByPortNum.find(srcIface->GetPortNum());
            if (it == peersByPortNum.end()) {
                continue;
            }
            for (LinkProtocol proto : sLink->GetLinkProtocols()) {
                auto& protoBucket = reachableByProtocol[proto];
                for (RankId dstRank : it->second) {
                    protoBucket.insert(dstRank);
                }
            }
        }
    }
}

const std::unordered_map<u32, std::vector<std::shared_ptr<NetInstance::ConnInterface>>>
NetInstance::Node::GetInterfacesMap() const
{
    return interfacesMap_;
}

vector<NetInstance::Path> ClosNetInstance::GetPaths(const RankId srcRankId, const RankId dstRankId) const
{
    vector<NetInstance::Path> paths;
    if (peers.count(srcRankId) == 0 || peers.count(dstRankId) == 0) {
        HCCL_WARNING(
            "[ClosNetInstance::GetPaths] srcRankId[%u] or dstRankId[%u] not exist in netInstance, "
            "netLayer[%u], netInstId[%s].",
            srcRankId, dstRankId, netLayer, netInstId.c_str());
        return paths;
    }
    NodeId srcPeerId = peers.at(srcRankId)->GetNodeId();
    NodeId dstPeerId = peers.at(dstRankId)->GetNodeId();
    for (auto& fabric : fabrics) {
        NodeId fabricId = fabric->GetNodeId();

        NetInstance::Link srcToFabricLink;
        vGraph.TraverseEdge(srcPeerId, fabricId, [&](shared_ptr<NetInstance::Link> edge) {
            srcToFabricLink = *edge;
            return;
        });

        NetInstance::Link fabricToDstLink;
        vGraph.TraverseEdge(fabricId, dstPeerId, [&](shared_ptr<NetInstance::Link> edge) {
            fabricToDstLink = *edge;
            return;
        });

        if (!srcToFabricLink.IsEmpty() && !fabricToDstLink.IsEmpty()) {
            if (!CheckPortGroupSize(netLayer, srcToFabricLink, fabricToDstLink)) {
                continue;
            }
            NetInstance::Path path;
            path.links = {srcToFabricLink, fabricToDstLink};
            paths.emplace_back(path);
        } else {
            HCCL_WARNING(
                "[ClosNetInstance::GetPaths] from src[%s] to dst[%s] link by fabric[%s] not found.",
                peers.at(srcRankId)->Describe().c_str(), peers.at(dstRankId)->Describe().c_str(),
                fabric->Describe().c_str());
        }
    }

    return paths;
}

void ClosNetInstance::GetReachableRanks(
    const RankId srcRankId, std::map<LinkProtocol, std::set<RankId>>& reachableByProtocol) const
{
    if (peers.count(srcRankId) == 0) {
        return;
    }
    NodeId srcPeerId = peers.at(srcRankId)->GetNodeId();
    for (const auto& fabric : fabrics) {
        NodeId fabricNodeId = fabric->GetNodeId();
        // 代表 sLink：桶内最后一条（TraverseEdge 覆盖式赋值取尾的等价实现）
        const auto* sLinks = vGraph.FindEdges(srcPeerId, fabricNodeId);
        if (sLinks == nullptr || sLinks->empty()) {
            continue;
        }
        const auto& sLink = sLinks->back();
        const auto srcIface = sLink->GetSourceIface();
        if (srcIface == nullptr) {
            continue;
        }
        const u32 sPortNum = srcIface->GetPortNum();
        // 对 F 的每个出边 peer：代表 dLink = 桶内最后一条；端口配对则并入 sLink 协议
        const auto* fabOut = vGraph.GetOutEdges(fabricNodeId);
        if (fabOut == nullptr) {
            continue;
        }
        for (const auto& [dstNodeId, edgeVec] : *fabOut) {
            if (!IsPeerNodeId(dstNodeId) || peers.count(static_cast<RankId>(dstNodeId)) == 0 || edgeVec.empty()) {
                continue;
            }
            RankId dstRank = static_cast<RankId>(dstNodeId);
            if (dstRank == srcRankId) {
                continue;
            }
            const auto& dLink = edgeVec.back(); // 代表 dLink：桶内最后一条
            const auto tgtIface = dLink->GetTargetIface();
            if (tgtIface == nullptr || tgtIface->GetPortNum() != sPortNum) {
                continue;
            }
            for (LinkProtocol proto : sLink->GetLinkProtocols()) {
                reachableByProtocol[proto].insert(dstRank);
            }
        }
    }
}

void NetInstance::Node::AddConnInterface(u32 layer, const shared_ptr<NetInstance::ConnInterface>& interface)
{
    auto& interfacesVec = interfacesMap_[layer];
    for (const auto& iface : interfacesVec) {
        if (*iface == *interface) {
            HCCL_WARNING(
                "[NetInstance][Node][AddConnInterface] interface addr[%s] has existed.",
                interface->GetAddr().Describe().c_str());
            return;
        }
    }

    interfacesVec.emplace_back(interface);
}

void NetInstance::Node::AddConnInterfaces(
    u32 layer, const std::vector<std::shared_ptr<NetInstance::ConnInterface>>& interfaces)
{
    if (interfaces.empty()) {
        return;
    }
    for (auto interface : interfaces) {
        AddConnInterface(layer, interface);
    }
}

NetInstance::Node::NodeType NetInstance::Node::GetType() const { return type_; }

std::vector<std::shared_ptr<NetInstance::ConnInterface>> NetInstance::Node::GetIfacesByLayer(u32 layer) const
{
    auto it = interfacesMap_.find(layer);
    if (it == interfacesMap_.end()) {
        HCCL_WARNING("[NetInstance][Node][GetIfacesByLayer] netLayer[%u] not exist.", layer);
        return std::vector<std::shared_ptr<NetInstance::ConnInterface>>{};
    }
    return it->second;
}

std::vector<std::shared_ptr<NetInstance::ConnInterface>> NetInstance::Node::GetIfaces() const
{
    std::vector<std::shared_ptr<NetInstance::ConnInterface>> ifaces;
    for (auto layerIfacesPair : interfacesMap_) {
        for (auto iface : layerIfacesPair.second) {
            ifaces.emplace_back(iface);
        }
    }
    return ifaces;
}

void NetInstance::Node::SetEndpointToIface(
    const CommAddr& commAddr, CommProtocol protocol, const std::shared_ptr<NetInstance::ConnInterface>& iface)
{
    endpointToIfaceMap_[std::make_pair(commAddr, protocol)] = iface;
}

const std::unordered_map<std::pair<CommAddr, CommProtocol>, std::shared_ptr<NetInstance::ConnInterface>>
NetInstance::Node::GetEndpointToIfaceMap() const
{
    return endpointToIfaceMap_;
}

NodeId NetInstance::Node::GetNodeId() const { return nodeId_; }

LocalId NetInstance::Peer::GetLocalId() const { return localId_; }

LocalId NetInstance::Peer::GetReplacedLocalId() const { return replacedLocalId_; }

DeviceId NetInstance::Peer::GetDeviceId() const { return deviceId_; }

u32 NetInstance::Peer::GetDevicePort() const { return devicePort_; }

u32 NetInstance::Peer::GetHostPort() const { return hostPort_; }

RankId NetInstance::Peer::GetRankId() const { return rankId_; }

set<u32> NetInstance::Peer::GetLevels() const { return netLayers_; }

bool NetInstance::Peer::HasLevel(u32 netLayer) const { return netLayers_.count(netLayer) > 0; }

const NetInstance* NetInstance::Peer::GetNetInstance(u32 netLayer) const
{
    if (netLayer >= netInsts_.size() || netInsts_.at(netLayer) == nullptr) {
        HCCL_WARNING("[NetInstance][Peer][GetNetInstance] netLayer[%u] not exist.", netLayer);
        return nullptr;
    }
    return netInsts_[netLayer];
}

NodeId NetInstance::Peer::GenerateNodeId(RankId rankId)
{
    return (static_cast<u64>(rankId) | static_cast<u64>(0) << 32); // 第32位为0 + rankId
}

string NetInstance::Peer::Describe() const
{
    return StringFormat(
        "NetInstance::Peer[rankId=%d, localId=%u, NodeId=%llu, netLayers_size=%zu]", rankId_, localId_, nodeId_,
        netLayers_.size());
}

HcclResult NetInstance::Peer::AddNetInstance(const std::shared_ptr<NetInstance>& netInst)
{
    u32 netLayer = netInst->GetNetLayer();
    if (netLayer >= netInsts_.size()) {
        netInsts_.resize(netLayer + 1);
    }

    if (netInsts_[netLayer] != nullptr) {
        HCCL_ERROR("[%s] rankId[%d] netLayer[%u] NetInstance has existed", __func__, rankId_, netLayer);
        return HCCL_E_PARA;
    }
    netInsts_[netLayer] = netInst.get();
    netLayers_.insert(netInst->GetNetLayer());
    return HCCL_SUCCESS;
}

void NetInstance::Peer::SetPortPortAddrMapLayer0(std::map<std::string, std::vector<IpAddress>> portAddrMap)
{
    portAddrMapLayer0_ = std::move(portAddrMap);
}

std::map<std::string, std::vector<IpAddress>> NetInstance::Peer::GetPortAddrMapLayer0() const
{
    return portAddrMapLayer0_;
}

PlaneId NetInstance::Fabric::GetPlaneId() const { return planeId_; }

NodeId NetInstance::Fabric::GenerateNodeId(FabricId fabricId) const
{
    return (static_cast<u64>(fabricId) | static_cast<u64>(1) << 32); // 第32位为1 + netplaneId
}

string NetInstance::Fabric::Describe() const
{
    return StringFormat("NetInstance::Fabric[netplaneId=%s, FabricNodeId=%llu]", planeId_.c_str(), nodeId_);
}

LinkType NetInstance::Link::GetType() const { return type_; }

const std::set<LinkProtocol>& NetInstance::Link::GetLinkProtocols() const { return linkProtocols_; }

LinkDirection NetInstance::Link::GetLinkDirection() const { return direction_; }

u32 NetInstance::Link::GetHop() const { return hop_; }

shared_ptr<NetInstance::Node> NetInstance::Link::GetSourceNode() const { return source_; }

shared_ptr<NetInstance::Node> NetInstance::Link::GetTargetNode() const { return target_; }

shared_ptr<NetInstance::ConnInterface> NetInstance::Link::GetSourceIface() const { return sourceIface_; }

shared_ptr<NetInstance::ConnInterface> NetInstance::Link::GetTargetIface() const { return targetIface_; }

string NetInstance::Link::Describe() const
{
    stringstream iFace;
    if (sourceIface_ != nullptr) {
        iFace << ", srcIface=" << sourceIface_->Describe();
    }
    if (targetIface_ != nullptr) {
        iFace << ", dstIface=" << targetIface_->Describe();
    }
    std::stringstream linkProtocolsStr;
    for (auto protocol : linkProtocols_) {
        if (!linkProtocolsStr.str().empty()) {
            linkProtocolsStr << ", ";
        }
        linkProtocolsStr << protocol;
    }
    return StringFormat(
        "NetInstance::Link[srcId=%llu, dstId=%llu, type=%s, hop=%u, dir=%s, proto=%s%s]", source_->GetNodeId(),
        target_->GetNodeId(), type_.Describe().c_str(), hop_, direction_.Describe().c_str(),
        linkProtocolsStr.str().c_str(), iFace.str().c_str());
}

bool NetInstance::Link::IsEmpty() const { return (source_ == nullptr) && (target_ == nullptr); }

bool NetInstance::Link::operator==(const NetInstance::Link& rhs) const
{
    return source_->GetNodeId() == rhs.source_->GetNodeId() && target_->GetNodeId() == rhs.target_->GetNodeId()
           && sourceIface_ == rhs.sourceIface_ && targetIface_ == rhs.targetIface_ && type_ == rhs.type_
           && linkProtocols_ == rhs.linkProtocols_ && direction_ == rhs.direction_ && hop_ == rhs.hop_;
}

bool NetInstance::Link::operator!=(const NetInstance::Link& rhs) const { return !(rhs == *this); }

IpAddress NetInstance::ConnInterface::GetAddr() const { return addr; }

std::set<string> NetInstance::ConnInterface::GetPorts() const { return ports; }

AddrPosition NetInstance::ConnInterface::GetPos() const { return pos; }

LinkType NetInstance::ConnInterface::GetLinkType() const { return linkType; }

std::set<LinkProtocol> NetInstance::ConnInterface::GetLinkProtocols() const { return linkProtocols; }

void NetInstance::ConnInterface::SetLocalDieId(u32 dieId) { localDieId_ = dieId; }

u32 NetInstance::ConnInterface::GetLocalDieId() const { return localDieId_; }

TopoType NetInstance::ConnInterface::GetTopoType() const { return topoType; }

u32 NetInstance::ConnInterface::GetTopoInstId() const { return topoInstId; }

std::string NetInstance::ConnInterface::Describe() const
{
    std::string portsStr;
    for (auto it = ports.begin(); it != ports.end(); ++it) {
        if (it != ports.begin()) {
            portsStr += ",";
        }
        portsStr += *it;
    }
    return StringFormat(
        "Iface[addr=%s, pos=%s, topoInstId=%u, topoType=%d, localDieId=%u, ports={%s}]", addr.Describe().c_str(),
        pos.Describe().c_str(), topoInstId, topoType, localDieId_, portsStr.c_str());
}

bool NetInstance::ConnInterface::operator==(const NetInstance::ConnInterface& rhs) const
{
    return addr == rhs.addr && pos == rhs.pos && linkType == rhs.linkType && linkProtocols == rhs.linkProtocols
           && ports == rhs.ports && topoInstId == rhs.topoInstId && topoType == rhs.topoType;
}

bool NetInstance::ConnInterface::operator!=(const NetInstance::ConnInterface& rhs) const { return !(rhs == *this); }

} // namespace Hccl
