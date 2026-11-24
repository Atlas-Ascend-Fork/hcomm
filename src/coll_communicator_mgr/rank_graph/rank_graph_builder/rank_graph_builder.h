/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANK_GRAPH_BUILDER_H
#define RANK_GRAPH_BUILDER_H

#include <cstdint>
#include <memory>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "rank_gph.h"
#include "rank_table_info.h"
#include "phy_topo.h"
#include "types.h"
#include "topo_common_types.h"
#include "topo_info.h"
#include "updater_for_64_plus_1.h"

namespace Hccl {

class RankGraphBuilder {
public:
    using LayerSelector = std::function<HcclResult(u32 availableLayers, u32& effectiveLayers)>;
    // 默认保留 RankTable 实际提供的层；显式策略在读取候选层后解析，共用同一构图流程。
    explicit RankGraphBuilder(const LayerSelector& selector = {}) : layerSelector_(selector) {}

    // External ranktable input retains the physical grouping view.
    std::unique_ptr<RankGraph> Build(const std::string& ranktableM, const std::string& topoPath, RankId myRank);
    // RootInfoDetect input enables continuous virtual groups.
    std::unique_ptr<RankGraph> Build(const RankTableInfo& ranktable, const std::string& topoPath, RankId myRank);
    std::unique_ptr<RankTableInfo> GetRankTableInfo();
    std::shared_ptr<hccl::TopoInfo> GetTopoInfo();
    std::unique_ptr<RankGraph> RecoverBuild(
        RankTableInfo rankTableInfo, const hccl::TopoInfo& topoInfo, RankId myRank, bool enableVirtualGroups = false);
    HcclResult SetEndpointDesc();

private:
    std::unique_ptr<RankTableInfo> rankTable_;
    std::unique_ptr<RankGraph> rankGraph_;
    RankId2PeerMap peers_;
    Level2Id2NetInst tempNetInsts_;
    RankId myRank_;
    // 无UB兜底标记：本rank的level0为pcie fallback层（ranktable中pcie_fallback字段），
    // BuildFromRankTable时记录，供建链/topoInst注册路径替代实例名前缀匹配
    bool level0PcieFallback_{false};
    std::shared_ptr<hccl::TopoInfo> topoInfo_;
    // 350L拆层形态: 非对称mesh组时为true, AddTopoDescFabricInfo据此过滤graph(0)中的clos边,
    // 维持"level0纯mesh"视图; 对称形态在RestoreSymmetricLevel0View中已还原单层, 无需过滤
    bool filterClosInLevel0_{false};
    UpdaterFor64Plus1 updaterFor64Plus1_{};
    std::optional<u32> selectedLayers_;
    u32 availableLayers_{0};
    LayerSelector layerSelector_;

    // vnic查询缓存<(localDeviceId, targetDeviceId), ip>：构建期单线程无需加锁；
    // 仅缓存查询成功结果，失败沿原异常路径传播保持可重试语义
    std::map<std::uint64_t, IpAddress> vnicIpCache_{};

    // 判断该真实层号是否需要建图；未配置选层时沿用默认构图。
    bool IsLayerSelected(u32 netLayer) const;
    // 根据拓扑还原后本 rank 的候选层解析策略并保存掩码；不检查算子可达性。
    HcclResult CheckSelectedLayers();
    // 从保留的本地归属初始化 innerRanks，不补回被禁用的 layer 0 通信边。
    HcclResult InitInnerRanksFromLocalGroups();

    IpAddress GetVnicIp(u32 localDeviceId, u32 targetDeviceId);
    HcclResult CheckMyRankInRankTable() const;
    HcclResult CheckNetLayerFromPhyTopo(const u32 netLayer) const;
    HcclResult BuildRankGraph(bool enableVirtualGroups);
    HcclResult BuildIpIndexMap();
    HcclResult BuildFromRankTable();
    void RestoreSymmetricLevel0View();
    HcclResult BuildPeer2PeerLinks();
    HcclResult BuildPcieFallbackLinks(NetInstance* innerNetInstance);
    HcclResult AddFabricInfo(u32 netLayer);
    void AddPeer2NetLink(
        const u32 netLayer, const std::string& netInstId, RankId rankId, const hccl::AddressInfo& addrInfo,
        const std::shared_ptr<NetInstance::Fabric>& fabNode, const std::vector<std::shared_ptr<PhyTopo::Link>>& links);
    HcclResult AddTopoDescFabricInfo();
    HcclResult UpdateTopoInstForMyRankOnly();
    const RankLevelInfo& GetRankLevelInfoByNetLayer(const hccl::NewRankInfo& rankInfo, u32 netLayer) const;
    std::shared_ptr<NetInstance> GetNetInstance(const RankLevelInfo& levelInfo);
    std::shared_ptr<NetInstance> CreateNetInstance(const RankLevelInfo& levelInfo);
};

std::map<PlaneId, FabricId> GetFabricsFromAddrInfo(const std::vector<hccl::AddressInfo>& rankAddrs);
std::vector<std::shared_ptr<PhyTopo::Link>> GetPeer2NetPhyLinks(u32 netLayer, LocalId localId);
// 建链上下文：与物理边/peer数据分离，避免自由函数参数膨胀
struct ConnIfaceBuildCtx {
    u32 localDeviceId{0};
    std::optional<LinkType> linkTypeOverride{std::nullopt};
    std::optional<u32> targetDeviceId{std::nullopt};
    std::map<std::uint64_t, IpAddress>* vnicIpCache{nullptr};
};

std::vector<std::shared_ptr<NetInstance::ConnInterface>> ConstructConnIFromPhyTopoConnIAndPortMap(
    std::shared_ptr<PhyTopo::ConnInterface> phyConnIFace,
    const std::map<std::string, std::vector<IpAddress>>& portAddrMap, const TopoType topoType, const u32 topoInstId,
    const ConnIfaceBuildCtx& ctx);
std::vector<std::shared_ptr<NetInstance::Link>> ConstructLinks(
    std::shared_ptr<NetInstance::Peer> srcPeer, std::shared_ptr<NetInstance::Peer> dstPeer,
    std::vector<std::shared_ptr<NetInstance::ConnInterface>> sourceIfaces,
    std::vector<std::shared_ptr<NetInstance::ConnInterface>> targetIfaces, std::shared_ptr<PhyTopo::Link> phyLink);
std::vector<std::shared_ptr<PhyTopo::Link>> GetPeer2PeerPhyLinks(
    std::shared_ptr<hccl::Graph<PhyTopo::Node, PhyTopo::Link>> phyTopoGraph, LocalId srcLocalId, LocalId dstLocalId);
} // namespace Hccl

#endif // RANK_GRAPH_BUILDER_H
