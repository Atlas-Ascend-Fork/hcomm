/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_graph_builder_bridge.h"

#include <algorithm>
#include <exception>
#include <new>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hccl_exception.h"
#include "net_layer_config.h"
#include "log.h"
#include "rank_graph_builder.h"
#include "phy_topo_builder.h"
#include "topo_info.h"

namespace Hccl {
namespace {
    using InterSuperPodPortsByLocalId = std::unordered_map<u32, std::unordered_set<std::string>>;

    bool IsInterSuperPodProtocol(const std::set<LinkProtocol>& protocols)
    {
        return protocols.count(LinkProtocol::ROCE) != 0 || protocols.count(LinkProtocol::UB_RTP) != 0
               || protocols.count(LinkProtocol::UBOE) != 0 || protocols.count(LinkProtocol::UB_TP) != 0;
    }

    // 收集指定层的跨超端口，返回localId→端口集合，供各rank复用。
    InterSuperPodPortsByLocalId CollectInterSuperPodPorts(const hccl::TopoInfo& topoInfo, u32 layer)
    {
        InterSuperPodPortsByLocalId portsByLocalId;
        const auto edgesIt = topoInfo.edges.find(layer);
        if (edgesIt == topoInfo.edges.end()) {
            return portsByLocalId;
        }
        for (const auto& edge : edgesIt->second) {
            // 只收集使用跨超协议的连接。
            if (!IsInterSuperPodProtocol(edge.protocols)) {
                continue;
            }
            auto& sourcePorts = portsByLocalId[edge.localA];
            sourcePorts.insert(edge.localAPorts.begin(), edge.localAPorts.end());
            // PEER2PEER的B端也是设备；PEER2NET只收集A端设备的端口。
            if (edge.linkType == LinkType::PEER2PEER) {
                auto& targetPorts = portsByLocalId[edge.localB];
                targetPorts.insert(edge.localBPorts.begin(), edge.localBPorts.end());
            }
        }
        return portsByLocalId;
    }

    // 判断当前rank在指定层的端口是否与该设备的跨超端口有交集。
    bool RankHasInterSuperPodPort(
        const hccl::NewRankInfo& rankInfo, u32 layer, const InterSuperPodPortsByLocalId& portsByLocalId)
    {
        const auto localPorts = portsByLocalId.find(rankInfo.localId);
        if (localPorts == portsByLocalId.end()) {
            return false;
        }

        const auto level = std::find_if(
            rankInfo.rankLevelInfos.begin(), rankInfo.rankLevelInfos.end(), [layer](const RankLevelInfo& info) {
                return info.netLayer == layer;
            });
        if (level == rankInfo.rankLevelInfos.end()) {
            return false;
        }

        for (const auto& portAddr : level->portAddrMap) {
            if (localPorts->second.count(portAddr.first) != 0) {
                return true;
            }
        }
        return false;
    }

    // 收集目标层追加逻辑ID后的新实例ID并检查长度。
    HcclResult CollectNetInstIdUpdates(
        RankTableInfo& rankTable, u32 physicalSuperPodLayer, std::vector<NetInstIdUpdate>& netInstIdUpdateList)
    {
        for (auto& rankInfo : rankTable.ranks) {
            if (rankInfo.logicSuperPodId.empty()) {
                continue;
            }
            for (auto& level : rankInfo.rankLevelInfos) {
                if (level.netLayer != physicalSuperPodLayer) {
                    continue;
                }
                std::string newNetInstId;
                const HcclResult ret = RankTableInfo::BuildLogicSuperPodNetInstId(
                    level.netInstId, rankInfo.logicSuperPodId, newNetInstId);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[%s] build logical netInstId failed, netLayer[%u] rankId[%u] ret[%d]", __func__,
                        physicalSuperPodLayer, rankInfo.rankId, ret),
                    ret);
                netInstIdUpdateList.push_back({&level.netInstId, &rankInfo.logicSuperPodId, std::move(newNetInstId)});
            }
        }
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult FindSuperPodLayer(const RankTableInfo& rankTable, const hccl::TopoInfo& topoInfo, u32& physicalSuperPodLayer)
{
    std::set<u32> registeredLayers;
    for (const auto& rankInfo : rankTable.ranks) {
        for (const auto& level : rankInfo.rankLevelInfos) {
            registeredLayers.insert(level.netLayer);
        }
    }
    if (registeredLayers.empty()) {
        // 所有rank都没有网络层配置，无法查找物理超节点层。
        HCCL_RUN_INFO("[%s] no net layer registered in rank table", __func__);
        return HCCL_E_NOT_FOUND;
    }

    // 保存上一次遍历的层号，初始值MAX_NET_LAYER表示还没遍历任何层。
    u32 previousRegisteredLayer = MAX_NET_LAYER;
    for (u32 layer : registeredLayers) {
        const auto portsByLocalId = CollectInterSuperPodPorts(topoInfo, layer);
        if (portsByLocalId.empty()) {
            previousRegisteredLayer = layer;
            continue;
        }
        bool hasInterSuperPodPort = false;
        for (const auto& rankInfo : rankTable.ranks) {
            if (RankHasInterSuperPodPort(rankInfo, layer, portsByLocalId)) {
                hasInterSuperPodPort = true;
                break;
            }
        }
        if (hasInterSuperPodPort) {
            if (previousRegisteredLayer == MAX_NET_LAYER) {
                // 最底层就是跨超网络，没有更低的层可作为物理超节点层。
                HCCL_RUN_INFO("[%s] inter layer[%u] is the lowest layer, no layer below it", __func__, layer);
                return HCCL_E_NOT_SUPPORT;
            }
            physicalSuperPodLayer = previousRegisteredLayer;
            HCCL_INFO("[%s] inter layer[%u], physical superpod layer[%u]", __func__, layer, physicalSuperPodLayer);
            return HCCL_SUCCESS; // 首个命中即终止
        }
        previousRegisteredLayer = layer;
    }
    // 没有找到跨超网络，使用最高的网络层作为物理超节点层。
    physicalSuperPodLayer = previousRegisteredLayer;
    HCCL_INFO("[%s] no inter layer found, take highest layer[%u]", __func__, physicalSuperPodLayer);
    return HCCL_SUCCESS;
}

HcclResult RewriteSuperPodNetInstId(RankTableInfo& rankTable, u32 physicalSuperPodLayer)
{
    bool hasLogicId = false;
    for (const auto& rankInfo : rankTable.ranks) {
        if (!rankInfo.logicSuperPodId.empty()) {
            hasLogicId = true;
            break;
        }
    }
    if (!hasLogicId) {
        return HCCL_SUCCESS;
    }

    std::vector<NetInstIdUpdate> netInstIdUpdateList;
    CHK_RET(CollectNetInstIdUpdates(rankTable, physicalSuperPodLayer, netInstIdUpdateList));

    std::set<std::pair<std::string, std::string>> loggedNetInstIds;
    // 统一写入已校验的新ID。
    for (auto& update : netInstIdUpdateList) {
        const std::string originNetInstId = std::move(*update.netInstId);
        *update.netInstId = std::move(update.newNetInstId);
        if (loggedNetInstIds.emplace(originNetInstId, *update.netInstId).second) {
            HCCL_INFO(
                "[%s] netLayer[%u] originNetInstId[%s] newNetInstId[%s] logicSuperPodId[%s]", __func__,
                physicalSuperPodLayer, originNetInstId.c_str(), update.netInstId->c_str(),
                update.logicSuperPodId->c_str());
        }
    }
    for (auto& rankInfo : rankTable.ranks) {
        // 未配置逻辑ID的rank也记录为空，确保它与已配置rank之间不会返回超内链路。
        rankTable.logicSuperPodGroupIds[rankInfo.rankId] = rankInfo.logicSuperPodId;
        // 清空已使用的逻辑ID，避免恢复建图时给实例ID重复加后缀。
        rankInfo.logicSuperPodId.clear();
    }
    rankTable.logicSuperPodSplitLayer = physicalSuperPodLayer;
    return HCCL_SUCCESS;
}

// 找到物理超节点层，并在该层实例ID后追加逻辑ID。
HcclResult SetLogicSuperPodId(RankTableInfo& rankTable, const std::string& topoPath)
{
    bool hasLogicId = false;
    for (const auto& rankInfo : rankTable.ranks) {
        if (!rankInfo.logicSuperPodId.empty()) {
            hasLogicId = true;
            break;
        }
    }
    if (!hasLogicId) {
        return HCCL_SUCCESS;
    }

    CHK_RET(hccl::PhyTopoBuilder::GetInstance().Build(topoPath));
    std::shared_ptr<hccl::TopoInfo> topoInfo = hccl::PhyTopoBuilder::GetInstance().GetTopoInfo();
    if (topoInfo == nullptr) {
        HCCL_ERROR("[%s] GetTopoInfo returns nullptr, topoPath[%s]", __func__, topoPath.c_str());
        return HCCL_E_PTR;
    }

    u32 physicalSuperPodLayer = MAX_NET_LAYER;
    HcclResult ret = FindSuperPodLayer(rankTable, *topoInfo, physicalSuperPodLayer);
    if (ret == HCCL_E_NOT_SUPPORT) {
        // 最底层就是跨超网络，无法确定物理超节点层，保留原表。
        HCCL_WARNING("[%s] inter layer is the lowest layer, keep original table and skip rewrite", __func__);
        return HCCL_SUCCESS;
    }
    if (ret == HCCL_E_NOT_FOUND) {
        // 所有rank都没有网络层配置，保留原表。
        HCCL_WARNING("[%s] no net layer registered in table, keep original table and skip rewrite", __func__);
        return HCCL_SUCCESS;
    }
    ret = RewriteSuperPodNetInstId(rankTable, physicalSuperPodLayer);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] RewriteSuperPodNetInstId failed, ret[%d]", __func__, ret), ret);
    return HCCL_SUCCESS;
}

namespace {

    // RankGraph 的实现和分配均位于 hcomm，跨 SO 持有时也必须回到 hcomm 执行析构。
    void DestroyRankGraphImpl(RankGraph* rankGraph) { delete rankGraph; }

    HcclResult AdoptRankGraphImpl(std::unique_ptr<RankGraph> rankGraph, std::shared_ptr<RankGraph>& sharedRankGraph)
    {
        if (rankGraph == nullptr) {
            HCCL_ERROR("[%s] input rankGraph is nullptr.", __func__);
            return HCCL_E_PTR;
        }

        RankGraph* rawRankGraph = rankGraph.release();
        try {
            // shared_ptr 构造失败时会自行调用 deleter，禁止在 catch 中再次释放 rawRankGraph。
            std::shared_ptr<RankGraph> adoptedRankGraph(rawRankGraph, DestroyRankGraphImpl);
            sharedRankGraph = std::move(adoptedRankGraph);
        } catch (const std::bad_alloc& e) {
            HCCL_ERROR("[%s] create shared RankGraph failed: %s", __func__, e.what());
            return HCCL_E_MEMORY;
        }
        return HCCL_SUCCESS;
    }

    HcclResult FillBuildResult(
        RankGraphBuilder& rankGraphBuilder, std::unique_ptr<RankGraph> rankGraph, RankGraphBuildResult& result)
    {
        // 在栈上 builder 析构前转移完整结果，保证 RankGraph、RankTableInfo 与 hccl::TopoInfo 来自同一次构建。
        std::unique_ptr<RankTableInfo> rankTableInfo = rankGraphBuilder.GetRankTableInfo();
        std::shared_ptr<hccl::TopoInfo> topoInfo = rankGraphBuilder.GetTopoInfo();
        if (rankGraph == nullptr || rankTableInfo == nullptr || topoInfo == nullptr) {
            HCCL_ERROR("[%s] rank graph build result is invalid.", __func__);
            return HCCL_E_INTERNAL;
        }

        // 正常建图和快照恢复共用此出口，交付前设置拆分标记（层号+分组映射），确保查询时的跨逻辑组链路限制生效。
        if (rankTableInfo->logicSuperPodSplitLayer < MAX_NET_LAYER) {
            rankGraph->SetLogicSuperPodSplit(
                rankTableInfo->logicSuperPodSplitLayer, rankTableInfo->logicSuperPodGroupIds);
        }

        CHK_RET(AdoptRankGraphImpl(std::move(rankGraph), result.rankGraph));
        result.rankTableInfo = std::move(*rankTableInfo);
        result.topoInfo = *topoInfo;
        return HCCL_SUCCESS;
    }

    // 默认和显式选层共用一次构造、构图和结果发布，仅策略不同。
    template <typename BuildFunc>
    HcclResult
    RunBuild(const char* funcName, BuildFunc&& buildFunc, RankGraphBuildResult& result, const char* netLayerExpression)
    {
        try {
            CHK_PTR_NULL(netLayerExpression);
            hccl::NetLayerSelection selection;
            // 空串保持默认构图；不读取通信域的其他配置。
            if (netLayerExpression[0] != '\0') {
                CHK_RET(hccl::ReadNetLayerSelection(netLayerExpression, selection));
            }
            RankGraphBuilder::LayerSelector selector;
            if (selection.GetMode() != hccl::NetLayerSelectionMode::DEFAULT) {
                // 按值保存策略，在读取真实候选层后解析，不重新读取环境。
                selector = [selection](u32 available, u32& effective) {
                    return selection.Resolve(available, effective);
                };
            }
            RankGraphBuilder builder(selector);
            return FillBuildResult(builder, buildFunc(builder), result);
        } catch (const HcclException& e) {
            HCCL_ERROR("[%s] failed: %s", funcName, e.what());
            return e.GetErrorCode();
        } catch (const std::bad_alloc& e) {
            HCCL_ERROR("[%s] failed: %s", funcName, e.what());
            return HCCL_E_MEMORY;
        } catch (const std::exception& e) {
            HCCL_ERROR("[%s] failed: %s", funcName, e.what());
            return HCCL_E_INTERNAL;
        } catch (...) {
            HCCL_ERROR("[%s] failed: unknown exception", funcName);
            return HCCL_E_INTERNAL;
        }
    }

    // 三个窄入口分别承接字符串初始化、RootInfoDetect 结果初始化和故障恢复，
    HcclResult BuildFromStringImpl(
        const std::string& rankTable, const std::string& topoPath, RankId myRank, const char* netLayerExpression,
        RankGraphBuildResult& result)
    {
        return RunBuild(
            __func__,
            [&](RankGraphBuilder& rankGraphBuilder) {
                return rankGraphBuilder.Build(rankTable, topoPath, myRank);
            },
            result, netLayerExpression);
    }

    std::unique_ptr<RankGraph> BuildRankGraphWithLogicSuperPodId(
        RankGraphBuilder& rankGraphBuilder, const RankTableInfo& rankTable, const std::string& topoPath, RankId myRank,
        HcclResult& logicRet)
    {
        RankTableInfo inputTable = rankTable;
        logicRet = SetLogicSuperPodId(inputTable, topoPath);
        if (logicRet != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] SetLogicSuperPodId failed, myRank[%d], ret[%d]", __func__, myRank, logicRet);
            return nullptr;
        }
        return rankGraphBuilder.Build(inputTable, topoPath, myRank);
    }

    HcclResult BuildFromRankTableImpl(
        const RankTableInfo& rankTable, const std::string& topoPath, RankId myRank, const char* netLayerExpression,
        RankGraphBuildResult& result)
    {
        bool needsLogicSuperPodRewrite = false;
        if (rankTable.GetSource() == RankTableSource::ROOTINFO) {
            for (const auto& rankInfo : rankTable.ranks) {
                if (!rankInfo.logicSuperPodId.empty()) {
                    needsLogicSuperPodRewrite = true;
                    break;
                }
            }
        }
        HcclResult logicRet = HCCL_SUCCESS;
        HcclResult ret = RunBuild(
            __func__,
            [&rankTable, &topoPath, myRank, needsLogicSuperPodRewrite, &logicRet](RankGraphBuilder& rankGraphBuilder) {
                if (needsLogicSuperPodRewrite) {
                    return BuildRankGraphWithLogicSuperPodId(rankGraphBuilder, rankTable, topoPath, myRank, logicRet);
                }
                return rankGraphBuilder.Build(rankTable, topoPath, myRank);
            },
            result, netLayerExpression);
        return (logicRet != HCCL_SUCCESS) ? logicRet : ret;
    }

    HcclResult RecoverBuildImpl(
        RankTableInfo&& rankTable, const hccl::TopoInfo& topoInfo, RankId myRank, bool enableVirtualGroups,
        const char* netLayerExpression, RankGraphBuildResult& result)
    {
        return RunBuild(
            __func__,
            [&](RankGraphBuilder& rankGraphBuilder) {
                return rankGraphBuilder.RecoverBuild(std::move(rankTable), topoInfo, myRank, enableVirtualGroups);
            },
            result, netLayerExpression);
    }

    HcclResult CreateSubGraphImpl(
        const RankGraph& parent, const std::vector<u32>& rankIds, const char* netLayerExpression,
        std::unique_ptr<RankGraph>& result)
    {
        try {
            CHK_PTR_NULL(netLayerExpression);
            if (netLayerExpression[0] == '\0') {
                return parent.CreateSubRankGraph(rankIds, result);
            }
            u32 available = 0;
            for (u32 layer : parent.GetLevels(parent.GetMyRank())) {
                if (layer >= MAX_NET_LAYER) {
                    HCCL_ERROR(
                        "[%s] parent rank[%d] has invalid netLayer[%u], expected less than[%u]", __func__,
                        parent.GetMyRank(), layer, MAX_NET_LAYER);
                    return HCCL_E_PARA;
                }
                available |= 1U << layer;
            }
            const auto& built = parent.GetBuildLayerSelection();
            const u32 allowed = built ? built->selectedLayers : available;
            u32 selected = 0;
            CHK_RET(hccl::NetLayerSelection::ResolveForSubComm(netLayerExpression, allowed, available, selected));
            return parent.CreateSubRankGraph(rankIds, result, selected);
        } catch (const HcclException& e) {
            HCCL_ERROR("[%s] failed: %s", __func__, e.what());
            return e.GetErrorCode();
        } catch (const std::bad_alloc& e) {
            HCCL_ERROR("[%s] allocate subgraph for rank count[%zu] failed: %s", __func__, rankIds.size(), e.what());
            return HCCL_E_MEMORY;
        } catch (...) {
            HCCL_ERROR("[%s] failed to build subgraph", __func__);
            return HCCL_E_INTERNAL;
        }
    }

    // 默认和显式选层共用原注册表，只传递构图所需的选层字段。
    const RankGraphBuilderBridge RANK_GRAPH_BUILDER_BRIDGE = {
        BuildFromStringImpl, BuildFromRankTableImpl, RecoverBuildImpl, AdoptRankGraphImpl, CreateSubGraphImpl,
    };

    // hcomm 装载时向 hccl_v2 发布窄回调接口，legacy 调用方无需链接 RankGraphBuilder 的具体实现。
    struct RankGraphBuilderBridgeRegistrar {
        RankGraphBuilderBridgeRegistrar()
        {
            HcclResult ret = RegisterRankGraphBuilderBridge(
                RANK_GRAPH_BUILDER_BRIDGE, RANK_GRAPH_BUILDER_BRIDGE_VERSION, sizeof(RANK_GRAPH_BUILDER_BRIDGE));
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[%s] register RankGraphBuilder bridge failed, ret[%d].", __func__, static_cast<int>(ret));
            }
        }
    };

    RankGraphBuilderBridgeRegistrar g_rankGraphBuilderBridgeRegistrar;

} // namespace
} // namespace Hccl
