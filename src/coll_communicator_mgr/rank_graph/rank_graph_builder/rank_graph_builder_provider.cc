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
        // 按设备localId查找跨超端口，避免不同设备的同名端口误匹配。
        const auto localPorts = portsByLocalId.find(rankInfo.localId);
        if (localPorts == portsByLocalId.end()) {
            return false;
        }
        // 找到当前rank在指定层的配置；未配置该层则不匹配。
        const auto level = std::find_if(
            rankInfo.rankLevelInfos.begin(), rankInfo.rankLevelInfos.end(), [layer](const RankLevelInfo& info) {
                return info.netLayer == layer;
            });
        if (level == rankInfo.rankLevelInfos.end()) {
            return false;
        }
        // RankTable JSON解析和二进制反序列化都会从rank_addr_list构建portAddrMap，这里直接遍历其端口key。
        for (const auto& portAddr : level->portAddrMap) {
            if (localPorts->second.count(portAddr.first) != 0) {
                return true;
            }
        }
        return false;
    }

    struct NetInstIdUpdate {
        std::string* netInstId;
        const std::string* logicId;
        std::string newNetInstId;
    };

    // 只收集待更新项，全部校验通过前不修改原表。
    HcclResult CollectNetInstIdUpdates(
        RankTableInfo& rankTable, u32 physicalSuperPodLayer, std::vector<NetInstIdUpdate>& netInstIdUpdates)
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
                netInstIdUpdates.push_back({&level.netInstId, &rankInfo.logicSuperPodId, std::move(newNetInstId)});
            }
        }
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult FindSuperPodLayer(const RankTableInfo& rankTable, const hccl::TopoInfo& topoInfo, u32& physicalSuperPodLayer)
{
    std::set<u32> registeredLayers; // 全表已注册层，set升序即实际层序
    for (const auto& rankInfo : rankTable.ranks) {
        for (const auto& level : rankInfo.rankLevelInfos) {
            registeredLayers.insert(level.netLayer);
        }
    }
    if (registeredLayers.empty()) {
        // 全表无已注册层：外层分别告警后跳过
        HCCL_DEBUG("[%s] no net layer registered in rank table", __func__);
        return HCCL_E_NOT_FOUND;
    }

    // 记录前一个已注册的网络层，MAX_NET_LAYER表示尚无前一层。
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
                // 最低已注册层即命中，其下没有网络层：按不支持返回，由外层告警跳过
                HCCL_DEBUG("[%s] inter layer[%u] is the lowest layer, no layer below it", __func__, layer);
                return HCCL_E_NOT_SUPPORT;
            }
            physicalSuperPodLayer = previousRegisteredLayer;
            HCCL_INFO("[%s] inter layer[%u], physical superpod layer[%u]", __func__, layer, physicalSuperPodLayer);
            return HCCL_SUCCESS; // 首个命中即终止
        }
        previousRegisteredLayer = layer; // 记录当前层，供下一轮使用；没有命中跨超端口的层也保留。
    }
    // 未找到跨超层时，前一个已注册层就是遍历到的最高层。
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
        return HCCL_SUCCESS; // 空配置：保持原表
    }

    std::vector<NetInstIdUpdate> netInstIdUpdates;
    CHK_RET(CollectNetInstIdUpdates(rankTable, physicalSuperPodLayer, netInstIdUpdates));

    std::set<std::pair<std::string, std::string>> loggedNetInstIds;
    // 统一写入已校验的新ID。
    for (auto& update : netInstIdUpdates) {
        const std::string originNetInstId = std::move(*update.netInstId);
        *update.netInstId = std::move(update.newNetInstId);
        if (loggedNetInstIds.emplace(originNetInstId, *update.netInstId).second) {
            HCCL_INFO(
                "[%s] netLayer[%u] originNetInstId[%s] newNetInstId[%s] logicSuperPodId[%s]", __func__,
                physicalSuperPodLayer, originNetInstId.c_str(), update.netInstId->c_str(), update.logicId->c_str());
        }
    }
    for (auto& rankInfo : rankTable.ranks) {
        // 未配置的rank也登记空ID，避免与已配置组查询时绕过跨组限制。
        rankTable.logicSuperPodGroupIds[rankInfo.rankId] = rankInfo.logicSuperPodId;
        // 清空已消费的配置，恢复重建不二次追加。
        rankInfo.logicSuperPodId.clear();
    }
    rankTable.logicSuperPodSplitLayer = physicalSuperPodLayer;
    return HCCL_SUCCESS;
}

// 建图前查找物理超节点层，并按逻辑ID改写该层实例ID。
HcclResult ApplyLogicSuperPodId(RankTableInfo& rankTable, const std::string& topoPath)
{
    bool hasLogicId = false;
    for (const auto& rankInfo : rankTable.ranks) {
        if (!rankInfo.logicSuperPodId.empty()) {
            hasLogicId = true;
            break;
        }
    }
    if (!hasLogicId) {
        return HCCL_SUCCESS; // 全部未配置：跳过，不触topo解析
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
        // 首个跨超协议层是最低层、其下没有网络层：告警跳过
        HCCL_WARNING("[%s] inter layer is the lowest layer, keep original table and skip rewrite", __func__);
        return HCCL_SUCCESS; // 外层告警跳过：不改表按原表建图
    }
    if (ret == HCCL_E_NOT_FOUND) {
        // 全表无已注册层：单独告警跳过，坏表由后续建图以各自错误报出
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

    // BuildRankGraph 是三个 provider 入口共享的构建骨架，BuildFunc 表示各入口不同的构建动作：
    template <typename BuildFunc>
    HcclResult BuildRankGraph(BuildFunc&& buildFunc, RankGraphBuildResult& result)
    {
        RankGraphBuilder rankGraphBuilder;
        return FillBuildResult(rankGraphBuilder, buildFunc(rankGraphBuilder), result);
    }

    template <typename BuildFunc>
    HcclResult RunBuild(const char* funcName, BuildFunc&& buildFunc, RankGraphBuildResult& result)
    {
        try {
            return BuildRankGraph(buildFunc, result);
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
        const std::string& rankTable, const std::string& topoPath, RankId myRank, RankGraphBuildResult& result)
    {
        return RunBuild(
            __func__,
            [&](RankGraphBuilder& rankGraphBuilder) {
                return rankGraphBuilder.Build(rankTable, topoPath, myRank);
            },
            result);
    }

    HcclResult BuildFromRankTableImpl(
        const RankTableInfo& rankTable, const std::string& topoPath, RankId myRank, RankGraphBuildResult& result)
    {
        // 逻辑超节点：表汇聚后、Build之前集中查层并改写全表目标层实例ID，改写后的表一次建图。
        // 拆分层号随RankTable保存，在FillBuildResult中设置图的拆分标记，让GetPaths过滤跨组的超内链路。
        HcclResult logicRet = HCCL_SUCCESS;
        HcclResult ret = RunBuild(
            __func__,
            [&](RankGraphBuilder& rankGraphBuilder) -> std::unique_ptr<RankGraph> {
                const bool needsRewrite
                    = rankTable.GetSource() == RankTableSource::ROOTINFO
                      && std::any_of(
                          rankTable.ranks.begin(), rankTable.ranks.end(), [](const hccl::NewRankInfo& rankInfo) {
                              return !rankInfo.logicSuperPodId.empty();
                          });
                if (!needsRewrite) {
                    return rankGraphBuilder.Build(rankTable, topoPath, myRank);
                }
                RankTableInfo inputTable = rankTable;
                logicRet = ApplyLogicSuperPodId(inputTable, topoPath);
                if (logicRet != HCCL_SUCCESS) {
                    return nullptr;
                }
                return rankGraphBuilder.Build(inputTable, topoPath, myRank);
            },
            result);
        return (logicRet != HCCL_SUCCESS) ? logicRet : ret;
    }

    HcclResult RecoverBuildImpl(
        const RankTableInfo& rankTable, const hccl::TopoInfo& topoInfo, RankId myRank, bool enableVirtualGroups,
        RankGraphBuildResult& result)
    {
        return RunBuild(
            __func__,
            [&](RankGraphBuilder& rankGraphBuilder) {
                return rankGraphBuilder.RecoverBuild(rankTable, topoInfo, myRank, enableVirtualGroups);
            },
            result);
    }

    // 将 bridge 的四个字段绑定到本文件中的 provider 实现；调用函数指针等价于调用对应的 Impl 函数。
    const RankGraphBuilderBridge RANK_GRAPH_BUILDER_BRIDGE = {
        BuildFromStringImpl,
        BuildFromRankTableImpl,
        RecoverBuildImpl,
        AdoptRankGraphImpl,
    };

    // hcomm 装载时向 hccl_v2 发布窄回调接口，legacy 调用方无需链接 RankGraphBuilder 的具体实现。
    struct RankGraphBuilderBridgeRegistrar {
        RankGraphBuilderBridgeRegistrar()
        {
            HcclResult ret = RegisterRankGraphBuilderBridge(RANK_GRAPH_BUILDER_BRIDGE);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[%s] register RankGraphBuilder bridge failed, ret[%d].", __func__, static_cast<int>(ret));
            }
        }
    };

    RankGraphBuilderBridgeRegistrar g_rankGraphBuilderBridgeRegistrar;

} // namespace
} // namespace Hccl
