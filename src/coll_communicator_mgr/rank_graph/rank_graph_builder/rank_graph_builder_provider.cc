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
#include <unordered_set>

#include "hccl_exception.h"
#include "log.h"
#include "rank_graph_builder.h"

namespace Hccl {
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
        // 在栈上 builder 析构前转移完整结果，保证 RankGraph、RankTableInfo 与 TopoInfo 来自同一次构建。
        std::unique_ptr<RankTableInfo> rankTableInfo = rankGraphBuilder.GetRankTableInfo();
        std::shared_ptr<TopoInfo> topoInfo = rankGraphBuilder.GetTopoInfo();
        if (rankGraph == nullptr || rankTableInfo == nullptr || topoInfo == nullptr) {
            HCCL_ERROR("[%s] rank graph build result is invalid.", __func__);
            return HCCL_E_INTERNAL;
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

    using InterSuperPodPortsByLocalId = std::unordered_map<u32, std::unordered_set<std::string>>;

    bool IsInterSuperPodProtocol(const std::set<LinkProtocol>& protocols)
    {
        return protocols.count(LinkProtocol::ROCE) != 0 || protocols.count(LinkProtocol::UB_RTP) != 0
               || protocols.count(LinkProtocol::UBOE) != 0 || protocols.count(LinkProtocol::UB_TP) != 0;
    }

    // 收集指定层的跨超端口，返回localId→端口集合，供各rank复用。
    InterSuperPodPortsByLocalId CollectInterSuperPodPorts(const TopoInfo& topoInfo, u32 layer)
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
    bool
    RankHasInterSuperPodPort(const NewRankInfo& rankInfo, u32 layer, const InterSuperPodPortsByLocalId& portsByLocalId)
    {
        // 按设备localId查找跨超端口，避免不同设备的同名端口匹配。
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
        // rank table JSON解析和二进制反序列化都会以rank_addr_list构建portAddrMap，这里直接遍历其端口key。
        for (const auto& portAddr : level->portAddrMap) {
            if (localPorts->second.count(portAddr.first) != 0) {
                return true;
            }
        }
        return false;
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
        return RunBuild(
            __func__,
            [&](RankGraphBuilder& rankGraphBuilder) {
                return rankGraphBuilder.Build(rankTable, topoPath, myRank);
            },
            result);
    }

    HcclResult RecoverBuildImpl(
        const RankTableInfo& rankTable, const TopoInfo& topoInfo, RankId myRank, bool enableVirtualGroups,
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

// 判定物理超节点层：虚拟分组分段上界的来源。
HcclResult FindSuperPodLayer(const RankTableInfo& rankTable, const TopoInfo& topoInfo, u32& physicalSuperPodLayer)
{
    std::set<u32> registeredLayers; // 全表已注册层，set升序即实际层序。
    for (const auto& rankInfo : rankTable.ranks) {
        for (const auto& level : rankInfo.rankLevelInfos) {
            registeredLayers.insert(level.netLayer);
        }
    }
    if (registeredLayers.empty()) {
        // 全表无已注册层：由外层告警后跳过。
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
                // 最低已注册层即命中，其下没有网络层：按不支持返回，由外层告警跳过。
                HCCL_DEBUG("[%s] inter layer[%u] is the lowest layer, no layer below it", __func__, layer);
                return HCCL_E_NOT_SUPPORT;
            }
            physicalSuperPodLayer = previousRegisteredLayer;
            HCCL_INFO("[%s] inter layer[%u], physical superpod layer[%u]", __func__, layer, physicalSuperPodLayer);
            return HCCL_SUCCESS; // 首个命中即终止。
        }
        previousRegisteredLayer = layer; // 记录当前层，供下一轮使用；没有命中跨超端口的层也保留。
    }
    // 未遇到跨超层时，前一个已注册层就是遍历到的最高层。
    physicalSuperPodLayer = previousRegisteredLayer;
    HCCL_INFO("[%s] no inter layer found, take highest layer[%u]", __func__, physicalSuperPodLayer);
    return HCCL_SUCCESS;
}
} // namespace Hccl
