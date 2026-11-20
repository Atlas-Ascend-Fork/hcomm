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
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "log.h"
#include "net_instance.h"
#include "rank_table_info.h"
#include "types.h"
#include "hccl_res.h"

namespace hccl {
class TopoInfo;
} // namespace hccl

namespace Hccl {

using Level2Id2NetInst = std::vector<std::unordered_map<std::string, std::shared_ptr<NetInstance>>>;
using RankId2PeerMap = std::unordered_map<RankId, std::shared_ptr<NetInstance::Peer>>;
constexpr u32 MAX_NET_LAYER = 8;
static_assert(MAX_NET_LAYER == MAX_VALUE_NETLAYER + 1, "RankGraph layer count must match RankTable layer range");

class RankGraph {
public:
    struct BuildLayerSelection {
        u32 availableLayers{0};
        u32 selectedLayers{0};
    };
    // 仅由建图器记录；查询层据此识别真实裁剪图，不再对裁剪结果重新解释排除表达式。
    const std::optional<BuildLayerSelection>& GetBuildLayerSelection() const { return buildLayerSelection_; }

    explicit RankGraph(RankId myRank, bool enableVirtualGroups = false)
        : netInsts_(MAX_NET_LAYER),
          myRank_(myRank),
          enableVirtualGroups_(enableVirtualGroups)
    {}
    friend class VirtualTopoStub;

    // 修改接口
    HcclResult AddPeer(const std::shared_ptr<NetInstance::Peer>& peer);
    HcclResult AddNetInstance(const std::shared_ptr<NetInstance>& netInst);
    HcclResult InitInnerRanks();
    // 本地成员独立于通信层保存，支持不含 layer 0 的图。
    HcclResult SetInnerRanks(const std::set<RankId>& ranks);
    void InitFinish();
    // Build the immutable grouping view before InitFinish when enabled; physical topology is unchanged.
    HcclResult BuildVirtualGroups();

    // 查询接口
    bool IsVirtualGroupsEnabled() const { return enableVirtualGroups_; }
    bool HasRank(RankId rankId) const;
    u32 GetRankSize() const;
    u32 GetInnerRankSize() const;
    // 本地归属不代表可通信层；裁掉 layer 0 后仍用于区分本机与跨机资源。
    const std::set<RankId>& GetInnerRanks() const { return innerRanks_; }
    RankId GetMyRank() const;
    LocalId GetLocalId(RankId rankId) const;
    // 查询原 layer 0 的 netInstanceId；仅表示归属，不授权 layer 0 通信。
    HcclResult GetLayer0NetInstanceId(RankId rankId, std::string& netInstId) const;
    // 仅统计保留的 layer 0 身份元数据，按各实例最小 rank 排列人数。
    HcclResult GetLayer0InstSizes(std::vector<u32>& instSizes) const;
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
    // 查询myRank在该netLayer下所在虚拟分段组的ranks列表及总数；未建立虚拟分组视图时返回其netInstance成员。
    HcclResult GetLocalInstRanks(const u32 netLayer, vector<u32>& rankList, u32& rankNum) const;
    // 查询myRank在该netLayer下所在虚拟分段组的rank总数；未建立虚拟分组视图时返回其netInstance成员总数。
    HcclResult GetLocalInstSize(const u32 netLayer, u32& rankSize) const;
    const NetType GetNetType(const u32 netLayer) const; //  查询netLayer的NetType
    HcclResult GetNetInstanceList(const u32 netLayer, vector<u32>& instSizeList, u32& listSize)
        const; // 给定netLayer，查询RankGraph在该netLayer分为多少NetInstance，以及每个NetInstance的size
    bool IsSymmetric(const u32 netLayer) const; // 给定netLayer，查询RankGraph在该netLayer是否是对称的
    // 无UB盖板兜底场景：本rank的level0为pcie fallback层（由RankGraphBuilder按rank table显式标记写入）
    bool IsLevel0PcieFallback() const { return level0PcieFallback_; }
    void SetLevel0PcieFallback(bool flag) { level0PcieFallback_ = flag; }

    void GetTopoInstsByLayer(const u32 netLayer, std::vector<u32>& topoInsts, u32& topoInstNum) const;
    HcclResult GetTopoType(const u32 netLayer, const u32 topoInstId, TopoType& topoType) const;
    // Returns HCCL_E_NOT_FOUND when the topology instance has no ranks in the local virtual group.
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

    // 未指定选层时继承父域策略；显式选层只能保留父图已有层，不重新编号。
    HcclResult CreateSubRankGraph(
        const std::vector<u32>& rankIds, std::unique_ptr<RankGraph>& subRankGraph,
        std::optional<u32> selectedLayers = std::nullopt) const;
    // 打包接口
    std::vector<char> GetPackedData(const std::vector<std::pair<u32, RankId>>& netLayerRankPairs) const;
    void Dump() const;

    // 逻辑超节点拆分标记：建图入口在改写成功后调用，记录拆分层号与各rank的逻辑ID分组映射。
    void SetLogicSuperPodSplit(u32 logicSuperPodSplitLayer, const std::map<RankId, std::string>& logicSuperPodGroupIds)
    {
        logicSuperPodSplitLayer_ = logicSuperPodSplitLayer;
        logicSuperPodGroupIds_ = logicSuperPodGroupIds;
        logicSuperPodSplitEnabled_ = true;
    }

private:
    struct LayerVirtualGroups {
        std::vector<std::vector<u32>> groups;
        std::unordered_map<RankId, u32> rankToGroup;
    };
    std::unordered_map<u32, LayerVirtualGroups> virtualGroups_;
    HcclResult BuildLayerVirtualGroups(u32 netLayer, LayerVirtualGroups& view, bool& hasSplit) const;
    HcclResult FindLocalVirtualGroup(u32 netLayer, const std::vector<u32>*& ranks) const;

    friend class RankGraphBuilder;
    RankId2PeerMap peers_;      // <rankId, Peer>
    Level2Id2NetInst netInsts_; // <netLayer, netInstId, group>
    std::set<RankId> innerRanks_;
    RankId myRank_;
    bool initFlag_{false};
    bool enableVirtualGroups_{false};            // Inherited by subgraphs and persisted for recovery.
    bool level0PcieFallback_{false};             // 本rank的level0是否为无UB兜底层
    u32 logicSuperPodSplitLayer_{MAX_NET_LAYER}; // 逻辑超节点拆分生效的目标层，MAX_NET_LAYER表示未生效
    bool logicSuperPodSplitEnabled_{false};      // 是否已启用逻辑超节点拆分
    std::map<RankId, std::string> logicSuperPodGroupIds_; // rankId→逻辑ID，SetLogicSuperPodSplit时拷入
    std::unordered_map<RankId, std::map<IpAddress, uint32_t>>
        ipIndexMap_; // ipIndex 映射表，由 RankGraphBuilder 统一设置
    std::optional<BuildLayerSelection> buildLayerSelection_;
    std::unordered_map<RankId, std::string> layer0NetInstIds_;
    // 保存原 Peer 的层号和接口供再次选层；不访问其 NetInstance，也不作为子图接口公开。
    std::unordered_map<RankId, std::shared_ptr<const NetInstance::Peer>> endpointSourcePeers_;

    HcclResult CreateSubNetInstances(
        const std::vector<RankId> rankIds, Level2Id2NetInst& subNetInstances, RankId2PeerMap& peers,
        RankGraph* subRankGraph) const;
    HcclResult AddSubPeers(const std::vector<RankId>& rankIds, RankGraph* subRankGraph, RankId2PeerMap& peers) const;
    // 使用已确定的选层结果构建子图，复制成员归属并按保留层处理端点映射。
    HcclResult CreateSubRankGraphImpl(
        const std::vector<u32>& rankIds, std::unique_ptr<RankGraph>& subRankGraph,
        const std::optional<BuildLayerSelection>& selection) const;
    HcclResult AddSubLinks(
        const std::vector<RankId>& rankIds, RankId2PeerMap& peers, Level2Id2NetInst& subNetInsts,
        RankId parentMyRank) const;
};

CommProtocol LinkProtocolToCommProtocol(const LinkProtocol& linkProtocol);

// 物理超节点层取首个跨超协议匹配层的前一个已注册层；未匹配时取最高已注册层。
HcclResult
FindSuperPodLayer(const RankTableInfo& rankTable, const hccl::TopoInfo& topoInfo, u32& physicalSuperPodLayer);

// 在目标层实例ID后追加"_<逻辑ID>"，全部校验通过后统一写入。
HcclResult RewriteSuperPodNetInstId(RankTableInfo& rankTable, u32 physicalSuperPodLayer);

// 建图前查找物理超节点层，并按逻辑ID改写该层实例ID。
HcclResult ApplyLogicSuperPodId(RankTableInfo& rankTable, const std::string& topoPath);

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
