/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANK_GRAPH_H
#define RANK_GRAPH_H

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "log.h"
#include "net_instance.h"
#include "rank_table_info.h"
#include "types.h"
#include "hccl_res.h"

namespace Hccl {

using Level2Id2NetInst = std::vector<std::unordered_map<std::string, std::shared_ptr<NetInstance>>>;
using RankId2PeerMap = std::unordered_map<RankId, std::shared_ptr<NetInstance::Peer>>;
constexpr u32 MAX_NET_LAYER = 8;

class RankGraph {
public:
    explicit RankGraph(RankId myRank) : netInsts_(MAX_NET_LAYER), myRank_(myRank) {}
    friend class VirtualTopoStub;

    // 修改接口
    HcclResult AddPeer(const std::shared_ptr<NetInstance::Peer>& peer);
    HcclResult AddNetInstance(const std::shared_ptr<NetInstance>& netInst);
    HcclResult InitInnerRanks();
    void InitFinish();

    // 查询接口
    bool HasRank(RankId rankId) const;
    u32 GetRankSize() const;
    u32 GetInnerRankSize() const;
    RankId GetMyRank() const;
    LocalId GetLocalId(RankId rankId) const;
    LocalId GetReplacedLocalId(RankId rankId) const;
    std::set<u32> GetLevels(RankId rankId) const;
    u32 GetLevelNum() const;
    const NetInstance* GetNetInstanceByNetInstId(u32 netLayer, const std::string& netInstId) const;
    NetInstance* GetNetInstanceByNetInstId(u32 netLayer, const std::string& netInstId);
    const NetInstance* GetNetInstanceByRankId(u32 netLayer, RankId rankId) const;
    NetInstance* GetNetInstanceByRankId(u32 netLayer, RankId rankId);
    const std::shared_ptr<NetInstance::Peer> GetPeer(RankId rankId) const;
    std::vector<NetInstance::Path> GetPaths(u32 netLayer, RankId sRankId, RankId dRankId) const;
    u32 GetLayerRanks(const u32 netLayer) const; // 获取myRank在指定netLayer包含的rank总数
    HcclResult GetLocalInstRanks(const u32 netLayer, vector<u32>& rankList, u32& rankNum)
        const; // 查询myRank在该netLayer下所在的netInstance中的所有ranks列表及总数
    HcclResult
    GetLocalInstSize(const u32 netLayer, u32& rankSize) const; // 查询myRank在该netLayer下所在的netInstance中的ranks总数
    const NetType GetNetType(const u32 netLayer) const; //  查询netLayer的NetType
    HcclResult GetNetInstanceList(const u32 netLayer, vector<u32>& instSizeList, u32& listSize)
        const; // 给定netLayer，查询RankGraph在该netLayer分为多少NetInstance，以及每个NetInstance的size
    bool IsSymmetric(const u32 netLayer) const; // 给定netLayer，查询RankGraph在该netLayer是否是对称的
    // 无UB盖板兜底场景：本rank的level0为pcie fallback层（由RankGraphBuilder按rank table显式标记写入）
    bool IsLevel0PcieFallback() const { return level0PcieFallback_; }
    void SetLevel0PcieFallback(bool flag) { level0PcieFallback_ = flag; }

    void GetTopoInstsByLayer(const u32 netLayer, std::vector<u32>& topoInsts, u32& topoInstNum) const;
    HcclResult GetTopoType(const u32 netLayer, const u32 topoInstId, TopoType& topoType) const;
    HcclResult
    GetRanksByTopoInst(const u32 netLayer, const u32 topoInstId, std::vector<u32>& ranks, u32& rankNum) const;

    HcclResult GetEndpointNum(uint32_t layer, uint32_t topoInstId, uint32_t* num) const;
    HcclResult
    GetEndpointDesc(uint32_t layer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc) const;
    HcclResult GetEndpointInfo(
        uint32_t rankId, const EndpointDesc* endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen,
        void* info) const;

    // 按 addr 和 rankId 查询 ipIndex（打平所有 netLayer 去重排序后的下标）
    HcclResult GetIpIndex(RankId rankId, const IpAddress& ipAddr, uint32_t& ipIndex) const;
    // 设置 ipIndex 映射表（由 RankGraphBuilder 在构建完成后统一设置）
    void SetIpIndexMap(std::unordered_map<RankId, std::map<IpAddress, uint32_t>>&& ipIndexMap)
    {
        ipIndexMap_ = std::move(ipIndexMap);
    }

    // 创建子虚拟拓扑
    HcclResult CreateSubRankGraph(
        const std::vector<u32>& rankIds, std::unique_ptr<RankGraph>& subRankGraph) const; // 外部接口传入类型为u32
    // 打包接口
    std::vector<char> GetPackedData(const std::vector<std::pair<u32, RankId>>& netLayerRankPairs) const;
    void Dump() const;

private:
    RankId2PeerMap peers_;      // <rankId, Peer>
    Level2Id2NetInst netInsts_; // <netLayer, netInstId, group>
    std::set<RankId> innerRanks_;
    RankId myRank_;
    bool initFlag_{false};
    bool level0PcieFallback_{false}; // 本rank的level0是否为无UB兜底层
    std::unordered_map<RankId, std::map<IpAddress, uint32_t>>
        ipIndexMap_; // ipIndex 映射表，由 RankGraphBuilder 统一设置

    HcclResult CreateSubNetInstances(
        const std::vector<RankId> rankIds, Level2Id2NetInst& subNetInstances, RankId2PeerMap& peers,
        RankGraph* subRankGraph) const;
    HcclResult AddSubPeers(const std::vector<RankId>& rankIds, RankGraph* subRankGraph, RankId2PeerMap& peers) const;
    HcclResult AddSubLinks(
        const std::vector<RankId>& rankIds, RankId2PeerMap& peers, Level2Id2NetInst& subNetInsts,
        RankId parentMyRank) const;
};

CommProtocol LinkProtocolToCommProtocol(const LinkProtocol& linkProtocol);

std::shared_ptr<NetInstance> GetOrCreateNetInstance(
    u32 netLayer, const string& netInstId, NetType type, Level2Id2NetInst& netInsts, RankGraph* rankGraph);
RankId GetSubRankId(const vector<RankId>& rankIds, RankId rank);

HcclResult GetNewNodeInfo(
    u32 layer, RankId newRankId, const NetInstance::Link& oldLink, shared_ptr<NetInstance>& newNetInstance,
    RankId2PeerMap& tmpPeers, shared_ptr<NetInstance::Node>& newNode, shared_ptr<NetInstance::ConnInterface>& newIface,
    bool isSource);

HcclResult AddNewLink(
    u32 layer, const NetInstance::Link& oldLink, RankId srcNewRankId, RankId dstNewRankId,
    shared_ptr<NetInstance>& newNetInstance, RankId2PeerMap& tmpPeers, const NetInstance* oldNetInstance,
    RankId parentMyRank);

HcclResult AddGroupLinks(
    const vector<RankId>& rankIds, const NetInstance* oldNetInstance, shared_ptr<NetInstance>& newNetInstance,
    RankId2PeerMap& tmpPeers, RankId parentMyRank);

bool NeedUpdateTopoInstForSubGraph(const NetInstance* oldNetInstance, u32 topoInstId, RankId parentMyRank);

HcclResult GetCommAddr(CommAddr& commAddr, const IpAddress& ipAddr);

EndpointLocType AddrPositionToEndpointLoc(AddrPosition pos);

} // namespace Hccl

#endif // RANK_GRAPH_H
