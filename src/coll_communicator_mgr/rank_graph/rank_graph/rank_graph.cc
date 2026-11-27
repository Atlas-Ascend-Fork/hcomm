/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <iterator>
#include <unordered_map>
#include "virtual_topo.h"
#include "string_util.h"
#include "binary_stream.h"
#include "exception_util.h"
#include "topo_common_types.h"
#include "null_ptr_exception.h"
#include "internal_exception.h"
#include "not_support_exception.h"
#include "rank_gph.h"
#include "link_data_adapter.h"
#include "net_layer_config.h"

namespace Hccl {

using namespace std;

string RankIds2Str(const set<RankId>& rankIds);

HcclResult RankGraph::AddPeer(const shared_ptr<NetInstance::Peer>& peer)
{
    if (!initFlag_) {
        if (peer == nullptr) {
            HCCL_ERROR("[%s] peer is nullptr.", __func__);
            return HCCL_E_PTR;
        }

        RankId rankId = peer->GetRankId();
        peers_[rankId] = peer;
    } else {
        HCCL_ERROR("[%s] RankGraph AddPeer fail, rankGraph has been initialized.", __func__);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult RankGraph::AddNetInstance(const shared_ptr<NetInstance>& netInst)
{
    if (netInst == nullptr) {
        HCCL_ERROR("[%s] netInst is nullptr.", __func__);
        return HCCL_E_PTR;
    }
    if (!initFlag_) {
        for (const auto rankId : netInst->GetRankIds()) {
            if (!HasRank(rankId)) {
                HCCL_ERROR("[%s] Non-innerRank[%d] exists netInst.", __func__, rankId);
                return HCCL_E_PARA;
            }
        }

        auto res = netInsts_[netInst->GetNetLayer()].emplace(netInst->GetNetInstId(), netInst);
        if (!res.second) {
            HCCL_WARNING(
                "[RankGraph][AddNetInstance] netLayer[%u] netInstId[%s] is existed.", netInst->GetNetLayer(),
                netInst->GetNetInstId().c_str());
        }
    } else {
        HCCL_ERROR("[%s] RankGraph AddNetInstance fail, rankGraph has been initialized.", __func__);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult RankGraph::InitInnerRanks()
{
    const NetInstance* innerInstance = nullptr;
    TRY_CATCH_RETURN(innerInstance = GetNetInstanceByRankId(0, myRank_););
    if (innerInstance == nullptr) {
        HCCL_ERROR("[%s] myRank[%d] netLayer[0] netInst is not existed.", __func__, myRank_);
        return HCCL_E_PTR;
    }

    innerRanks_ = innerInstance->GetRankIds();
    return HCCL_SUCCESS;
}

HcclResult RankGraph::SetInnerRanks(const std::set<RankId>& ranks)
{
    CHK_PRT_RET(
        initFlag_,
        HCCL_ERROR(
            "[RankGraph][%s] cannot change inner ranks after initialization, myRank[%d], innerRanks[%s], "
            "requestedRanks[%s]",
            __func__, myRank_, RankIds2Str(innerRanks_).c_str(), RankIds2Str(ranks).c_str()),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(
        ranks.count(myRank_) == 0, HCCL_ERROR("[RankGraph] local rank is absent from inner ranks"), HCCL_E_PARA);
    for (const auto rank : ranks) {
        CHK_PRT_RET(!HasRank(rank), HCCL_ERROR("[RankGraph] unknown inner rank[%d]", rank), HCCL_E_PARA);
    }
    innerRanks_ = ranks;
    return HCCL_SUCCESS;
}

void RankGraph::InitFinish() { initFlag_ = true; }

HcclResult RankGraph::BuildLayerVirtualGroups(u32 netLayer, LayerVirtualGroups& view, bool& hasSplit) const
{
    hasSplit = false;
    for (const auto& entry : netInsts_[netLayer]) {
        CHK_PTR_NULL(entry.second);
        const auto ranks = entry.second->GetRankIds();
        if (ranks.empty()) {
            continue;
        }
        std::vector<u32> segment;
        for (RankId rank : ranks) {
            if (!segment.empty() && static_cast<u64>(rank) != static_cast<u64>(segment.back()) + 1) {
                HCCL_RUN_WARNING(
                    "[BuildVirtualGroups] non-contiguous ranks trigger virtual grouping, myRank[%d], layer[%u], "
                    "netInstId[%s], segment[%u..%u], nextRank[%d]",
                    myRank_, netLayer, entry.first.c_str(), segment.front(), segment.back(), rank);
                view.groups.push_back(std::move(segment));
                segment.clear();
                hasSplit = true;
            }
            segment.push_back(static_cast<u32>(rank));
        }
        view.groups.push_back(std::move(segment));
    }
    if (!hasSplit) {
        return HCCL_SUCCESS;
    }
    // Sorting instances alone is insufficient: their segments may interleave.
    std::sort(view.groups.begin(), view.groups.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.front() < rhs.front();
    });
    for (u32 index = 0; index < view.groups.size(); ++index) {
        const auto& group = view.groups[index];
        for (u32 rank : group) {
            if (!HasRank(rank) || !view.rankToGroup.emplace(rank, index).second) {
                HCCL_ERROR("[BuildVirtualGroups] invalid or duplicate rank[%u] layer[%u]", rank, netLayer);
                return HCCL_E_PARA;
            }
        }
        HCCL_INFO(
            "[BuildVirtualGroups] layer[%u] _HCCLSPLIT_ group[%u] ranks[%u..%u]", netLayer, index, group.front(),
            group.back());
    }
    return HCCL_SUCCESS;
}

HcclResult RankGraph::BuildVirtualGroups()
{
    // Once published, interface caches and concurrent readers require an immutable view.
    if (initFlag_) {
        HCCL_ERROR("[BuildVirtualGroups] rank graph is already initialized");
        return HCCL_E_INTERNAL;
    }
    if (!enableVirtualGroups_) {
        return HCCL_SUCCESS;
    }
    std::set<u32> localLevels;
    TRY_CATCH_RETURN(localLevels = GetLevels(myRank_););
    const std::vector<u32> levels(localLevels.begin(), localLevels.end());
    std::unordered_map<u32, LayerVirtualGroups> newViews;
    // 当前策略：跳过当前 rank 加入的最低层级；分段上界为物理超节点层（层值），levels 升序，超过上界即停止切分。
    for (size_t i = 1; i < levels.size(); ++i) {
        if (levels[i] > physicalSuperPodLayer_) {
            break;
        }
        LayerVirtualGroups view;
        bool hasSplit = false;
        CHK_RET(BuildLayerVirtualGroups(levels[i], view, hasSplit));
        if (hasSplit) {
            newViews.emplace(levels[i], std::move(view));
        }
    }
    virtualGroups_.swap(newViews);
    return HCCL_SUCCESS;
}

HcclResult RankGraph::FindLocalVirtualGroup(u32 netLayer, const std::vector<u32>*& ranks) const
{
    ranks = nullptr;
    const auto layer = virtualGroups_.find(netLayer);
    if (layer == virtualGroups_.end()) {
        return HCCL_SUCCESS;
    }
    const auto group = layer->second.rankToGroup.find(myRank_);
    if (group == layer->second.rankToGroup.end() || group->second >= layer->second.groups.size()) {
        HCCL_ERROR("[FindLocalVirtualGroup] missing rank[%d] on layer[%u]", myRank_, netLayer);
        return HCCL_E_PARA;
    }
    ranks = &layer->second.groups[group->second];
    return HCCL_SUCCESS;
}

bool RankGraph::HasRank(RankId rankId) const
{
    if (peers_.find(rankId) == peers_.end()) {
        HCCL_DEBUG("[RankGraph][HasRank] rankId[%d] is not existed", rankId);
        return false;
    }
    return true;
}

u32 RankGraph::GetRankSize() const { return peers_.size(); }

u32 RankGraph::GetInnerRankSize() const { return innerRanks_.size(); }

RankId RankGraph::GetMyRank() const { return myRank_; }

LocalId RankGraph::GetLocalId(RankId rankId) const
{
    if (!HasRank(rankId)) {
        THROW<InvalidParamsException>(StringFormat("[RankGraph][GetLocalId] rankId[%d] is not existed.", rankId));
    }

    return peers_.at(rankId)->GetLocalId();
}

LocalId RankGraph::GetReplacedLocalId(RankId rankId) const
{
    if (!HasRank(rankId)) {
        THROW<InvalidParamsException>(StringFormat("[RankGraph][GetLocalId] rankId[%d] is not existed.", rankId));
    }

    return peers_.at(rankId)->GetReplacedLocalId();
}

HcclResult RankGraph::GetLayer0NetInstanceId(RankId rankId, std::string& netInstId) const
{
    // 返回原 layer 0 的 netInstanceId（本地归属），不代表 layer 0 仍可用于通信。
    netInstId.clear();
    if (!HasRank(rankId)) {
        HCCL_WARNING("[RankGraph][GetLayer0NetInstanceId] rank[%d] does not exist", rankId);
        return HCCL_E_NOT_FOUND;
    }
    auto it = layer0NetInstIds_.find(rankId);
    if (it != layer0NetInstIds_.end()) {
        netInstId = it->second;
        return HCCL_SUCCESS;
    }
    // 兼容直接组装的完整图；仅查询身份，不创建或恢复 layer 0 通信边。
    const auto* localInstance = GetNetInstanceByRankId(0, rankId);
    if (localInstance == nullptr) {
        HCCL_DEBUG(
            "[RankGraph][GetLayer0NetInstanceId] no layer 0 identity metadata or netInstance, myRank[%d], rank[%d]",
            myRank_, rankId);
        return HCCL_E_NOT_FOUND;
    }
    netInstId = localInstance->GetNetInstId();
    return HCCL_SUCCESS;
}

HcclResult RankGraph::GetLayer0InstSizes(std::vector<u32>& instSizes) const
{
    CHK_PRT_RET(
        layer0NetInstIds_.empty(), HCCL_ERROR("[RankGraph] layer 0 identity metadata is empty, rank[%d]", myRank_),
        HCCL_E_NOT_FOUND);
    // 一次遍历身份表计数，不逐 rank 查询实例，也不复制所有实例的成员列表。
    std::unordered_map<std::string, std::pair<RankId, u32>> instanceSizes;
    for (const auto& entry : layer0NetInstIds_) {
        auto inserted = instanceSizes.emplace(entry.second, std::make_pair(entry.first, 0U));
        auto& size = inserted.first->second;
        size.first = std::min(size.first, entry.first);
        ++size.second;
    }
    std::map<RankId, u32> orderedSizes;
    for (const auto& instance : instanceSizes) {
        orderedSizes.emplace(instance.second.first, instance.second.second);
    }
    instSizes.clear();
    for (const auto& instance : orderedSizes) {
        instSizes.push_back(instance.second);
    }
    return HCCL_SUCCESS;
}

set<u32> RankGraph::GetLevels(RankId rankId) const
{
    if (!HasRank(rankId)) {
        THROW<InvalidParamsException>(StringFormat("[RankGraph][GetLevels] rankId[%d] is not existed.", rankId));
    }

    return peers_.at(rankId)->GetLevels();
}

bool RankGraph::HasLevel(RankId rankId, u32 netLayer) const
{
    return HasRank(rankId) && peers_.at(rankId)->HasLevel(netLayer);
}

u32 RankGraph::GetLevelNum() const
{
    u32 validLevelNum{0};
    for (const auto& netInst : netInsts_) {
        if (!netInst.empty()) {
            validLevelNum++;
        }
    }
    HCCL_INFO("[RankGraph][%s] validLevelNum[%u]", __func__, validLevelNum);
    return validLevelNum;
}

const NetInstance* RankGraph::GetNetInstanceByNetInstId(u32 netLayer, const string& netInstId) const
{
    // 不存在netInst, 则返回空
    if (netLayer >= netInsts_.size() || netInsts_.at(netLayer).count(netInstId) == 0) {
        HCCL_WARNING(
            "[RankGraph][GetNetInstance] NetInstance netLayer[%u] netInstId[%s] is not existed.", netLayer,
            netInstId.c_str());
        return nullptr;
    }
    return netInsts_.at(netLayer).at(netInstId).get();
}

NetInstance* RankGraph::GetNetInstanceByNetInstId(u32 netLayer, const std::string& netInstId)
{
    // 不存在netInst, 则返回空
    if (netLayer >= netInsts_.size() || netInsts_.at(netLayer).count(netInstId) == 0) {
        HCCL_WARNING(
            "[RankGraph][GetNetInstance] NetInstance netLayer[%u] netInstId[%s] is not existed.", netLayer,
            netInstId.c_str());
        return nullptr;
    }
    return netInsts_.at(netLayer).at(netInstId).get();
}

const NetInstance* RankGraph::GetNetInstanceByRankId(u32 netLayer, RankId rankId) const
{
    // 不存在netInst, 则返回空
    if (!HasRank(rankId)) {
        HCCL_WARNING(
            "[RankGraph][GetNetInstance] NetInstance rankId[%d] netLayer[%u] is not existed.", rankId, netLayer);
        return nullptr;
    }
    return peers_.at(rankId)->GetNetInstance(netLayer);
}

NetInstance* RankGraph::GetNetInstanceByRankId(u32 netLayer, RankId rankId)
{
    if (!HasRank(rankId)) {
        HCCL_WARNING(
            "[RankGraph][GetNetInstance] NetInstance rankId[%d] netLayer[%u] is not existed.", rankId, netLayer);
        return nullptr;
    }
    const NetInstance* constInstance = peers_.at(rankId)->GetNetInstance(netLayer);
    if (constInstance == nullptr) {
        THROW<NullPtrException>(StringFormat("[RankGraph][GetGroup]GetNetInstance(rankId, netLayer) is nullptr"));
    }
    string netInstId = constInstance->GetNetInstId();
    return netInsts_.at(netLayer).at(netInstId).get();
}

const shared_ptr<NetInstance::Peer> RankGraph::GetPeer(RankId rankId) const
{
    if (!HasRank(rankId)) {
        HCCL_WARNING("[RankGraph][GetPeer] rankId[%d] is not existed.", rankId);
        return nullptr;
    }

    return peers_.at(rankId);
}

vector<NetInstance::Path> RankGraph::GetPaths(u32 netLayer, RankId sRankId, RankId dRankId) const
{
    // 跨逻辑组的rank对在拆分层及其下层不得返回超内链路。
    if (logicSuperPodSplitEnabled_ && netLayer <= logicSuperPodSplitLayer_) {
        auto sGroup = logicSuperPodGroupIds_.find(sRankId);
        auto dGroup = logicSuperPodGroupIds_.find(dRankId);
        if (sGroup != logicSuperPodGroupIds_.end() && dGroup != logicSuperPodGroupIds_.end()
            && sGroup->second != dGroup->second) {
            HCCL_DEBUG(
                "[RankGraph][GetPaths] cross logic superpod, netLayer[%u] sRankId[%d] dRankId[%d]", netLayer, sRankId,
                dRankId);
            return {};
        }
    }
    // 若sRank和dRank均不在innerGroup里，则返回空
    if (innerRanks_.count(sRankId) == 0 && innerRanks_.count(dRankId) == 0) {
        HCCL_WARNING("[RankGraph][GetPaths] sRankId[%d] and dRankId[%d] are not in innerRanks", sRankId, dRankId);
        return {};
    }

    vector<NetInstance::Path> paths;
    auto netInst = GetNetInstanceByRankId(netLayer, sRankId);
    if (netInst == nullptr) {
        HCCL_WARNING("[RankGraph][GetPaths] netLayer[%u] sRankId[%d] netInst is not existed.", netLayer, sRankId);
        return {};
    }
    if (!netInst->HasNode(NetInstance::Peer::GenerateNodeId(dRankId))) {
        HCCL_WARNING(
            "[RankGraph][GetPaths] netLayer[%u] sRankId[%d] netInst has no dRankId[%d].", netLayer, sRankId, dRankId);
        return {};
    }

    paths = netInst->GetPaths(sRankId, dRankId);
    if (paths.size() == 0) {
        HCCL_WARNING(
            "[RankGraph][GetPaths] netLayer[%u] sRankId[%d] dRankId[%d] netInst has no path.", netLayer, sRankId,
            dRankId);
        return paths;
    }

    HCCL_DEBUG(
        "[RankGraph][GetPaths] netLayer[%u] sRankId[%d] dRankId[%d] pathsize[%u].", netLayer, sRankId, dRankId,
        paths.size());
    return paths;
}

u32 RankGraph::GetLayerRanks(const u32 netLayer) const
{
    u32 layerRankSize = 0;
    if (netInsts_.at(netLayer).size() == 0) {
        HCCL_WARNING("[RankGraph][GetLayerRanks] RankGraph has no netInstance on netLayer %u", netLayer);
        return 0;
    }
    for (const auto& netInst : netInsts_.at(netLayer)) {
        layerRankSize += netInst.second->GetRankSize();
    }
    return layerRankSize;
}

HcclResult RankGraph::GetLocalInstRanks(const u32 netLayer, vector<u32>& rankList, u32& rankNum) const
{
    const std::vector<u32>* group = nullptr;
    CHK_RET(FindLocalVirtualGroup(netLayer, group));
    if (group != nullptr) {
        rankList = *group;
        rankNum = static_cast<u32>(rankList.size());
        return HCCL_SUCCESS;
    }
    const NetInstance* netInstance = GetNetInstanceByRankId(netLayer, myRank_);
    if (netInstance == nullptr) {
        HCCL_ERROR("[%s] myRank %u has no netInstance on layer %u", __func__, myRank_, netLayer);
        return HCCL_E_PTR;
    }
    set<RankId> rankSet = netInstance->GetRankIds();
    rankList.clear();
    for (const RankId& rank : rankSet) {
        rankList.push_back(static_cast<u32>(rank));
    }
    rankNum = rankSet.size();
    return HCCL_SUCCESS;
}

HcclResult RankGraph::GetReachableRanksPerProtocol(
    u32 netLayer, RankId selfRank, const std::set<CommProtocol>& focusedProtocols,
    std::unordered_map<CommProtocol, std::vector<uint32_t>>& reachableRanks) const
{
    if (!HasLevel(selfRank, netLayer)) {
        HCCL_ERROR("[RankGraph][%s] netLayer[%u] is invalid for rank[%u]", __func__, netLayer, selfRank);
        return HCCL_E_PARA;
    }
    const NetInstance* netInstance = GetNetInstanceByRankId(netLayer, selfRank);
    CHK_PRT_RET(
        netInstance == nullptr, HCCL_ERROR("[%s] rank %u has no netInstance on layer %u", __func__, selfRank, netLayer),
        HCCL_E_PTR);

    // 口径对齐老路径：虚拟分段视图存在时，收集域限定在本 rank 所在分段内（老逐 peer 路径经
    // GetInstRanksByNetLayer→GetLocalInstRanks 枚举分段成员；无分段视图时不过滤，行为不变）
    const std::vector<u32>* virtualGroup = nullptr;
    if (selfRank == myRank_) {
        CHK_RET(FindLocalVirtualGroup(netLayer, virtualGroup));
    }

    std::map<LinkProtocol, std::set<RankId>> reachableByProtocol; // 未过滤（LinkProtocol 域）
    netInstance->GetReachableRanks(selfRank, reachableByProtocol);
    // 出参自洽：先按聚焦协议预置空桶再填充，未命中协议的桶不依赖调用方预置
    for (CommProtocol protocol : focusedProtocols) {
        reachableRanks[protocol] = {};
    }
    // 1:1 映射过滤：聚焦协议外的桶丢弃（ROCE/HCCS/PCIE 零物化成本跳过）
    for (const auto& [linkProto, rankSet] : reachableByProtocol) {
        CommProtocol commProto = LinkProtocolToCommProtocol(linkProto);
        if (focusedProtocols.count(commProto) == 0) {
            continue;
        }
        if (virtualGroup != nullptr) {
            // 分段交集过滤（rankSet 与分段各自升序，双指针求交保持升序去重）
            std::vector<uint32_t> filtered;
            filtered.reserve(rankSet.size());
            auto groupIt = virtualGroup->begin();
            for (RankId rank : rankSet) {
                while (groupIt != virtualGroup->end() && *groupIt < static_cast<uint32_t>(rank)) {
                    ++groupIt;
                }
                if (groupIt != virtualGroup->end() && *groupIt == static_cast<uint32_t>(rank)) {
                    filtered.push_back(static_cast<uint32_t>(rank));
                }
            }
            reachableRanks[commProto] = std::move(filtered);
        } else {
            reachableRanks[commProto] = std::vector<uint32_t>(rankSet.begin(), rankSet.end()); // set 天然升序去重
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RankGraph::GetLocalInstSize(const u32 netLayer, u32& rankSize) const
{
    const std::vector<u32>* group = nullptr;
    CHK_RET(FindLocalVirtualGroup(netLayer, group));
    if (group != nullptr) {
        rankSize = static_cast<u32>(group->size());
        return HCCL_SUCCESS;
    }
    const NetInstance* netInstance = GetNetInstanceByRankId(netLayer, myRank_);
    if (netInstance == nullptr) {
        HCCL_ERROR("[%s] myRank %u has no netInstance on layer %u", __func__, myRank_, netLayer);
        return HCCL_E_PTR;
    }
    rankSize = netInstance->GetRankSize();
    return HCCL_SUCCESS;
}

const NetType RankGraph::GetNetType(const u32 netLayer) const
{
    const NetInstance* netInstance = GetNetInstanceByRankId(netLayer, myRank_);
    if (netInstance == nullptr) {
        THROW<NullPtrException>(
            StringFormat("[RankGraph][GetLocalInstSize] myRank %u has no netInstance on layer %u", myRank_, netLayer));
    }
    return netInstance->GetNetType();
}

HcclResult RankGraph::GetNetInstanceList(const u32 netLayer, vector<u32>& instSizeList, u32& listSize) const
{
    instSizeList.clear();
    listSize = 0;
    const auto view = virtualGroups_.find(netLayer);
    if (view != virtualGroups_.end()) {
        for (const auto& group : view->second.groups) {
            instSizeList.push_back(static_cast<u32>(group.size()));
        }
        listSize = static_cast<u32>(instSizeList.size());
        return HCCL_SUCCESS;
    }
    if (netLayer >= netInsts_.size()) {
        return HCCL_E_PARA;
    }
    if (netInsts_.at(netLayer).size() == 0) {
        HCCL_WARNING("[RankGraph][GetNetInstanceList] RankGraph has no net instance on layer %u", netLayer);
        return HCCL_E_PARA;
    }
    for (const auto& netInst : netInsts_.at(netLayer)) {
        instSizeList.push_back(netInst.second->GetRankSize());
    }
    listSize = instSizeList.size();
    return HCCL_SUCCESS;
}

void RankGraph::GetTopoInstsByLayer(const u32 netLayer, std::vector<u32>& topoInsts, u32& topoInstNum) const
{
    auto* netInstance = GetNetInstanceByRankId(netLayer, myRank_);
    netInstance->GetTopoInstsByLayer(topoInsts, topoInstNum);
}

HcclResult RankGraph::GetTopoType(const u32 netLayer, const u32 topoInstId, TopoType& topoType) const
{
    auto* netInstance = GetNetInstanceByRankId(netLayer, myRank_);
    CHK_PRT_RET(
        netInstance == nullptr,
        HCCL_ERROR(
            "[RankGraph::GetTopoType] netInstance is nullptr, myRank[%d], netLayer[%u], "
            "topoInstId[%u]",
            myRank_, netLayer, topoInstId),
        HCCL_E_PTR);

    auto ret = netInstance->GetTopoType(topoInstId, topoType);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] Failed to GetTopoType, myRank[%d], netLayer[%u], netInstId[%s], topoInstId[%u], "
            "ret[%d]",
            __func__, myRank_, netLayer, netInstance->GetNetInstId().c_str(), topoInstId, ret),
        ret);
    return HCCL_SUCCESS;
}

HcclResult
RankGraph::GetRanksByTopoInst(const u32 netLayer, const u32 topoInstId, std::vector<u32>& ranks, u32& rankNum) const
{
    auto* netInstance = GetNetInstanceByRankId(netLayer, myRank_);
    CHK_PRT_RET(
        netInstance == nullptr,
        HCCL_ERROR(
            "[RankGraph::GetRanksByTopoInst] netInstance is nullptr, myRank[%d], netLayer[%u], "
            "topoInstId[%u]",
            myRank_, netLayer, topoInstId),
        HCCL_E_PTR);

    auto ret = netInstance->GetRanksByTopoInst(topoInstId, ranks, rankNum);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] Failed to GetRanksByTopoInst, myRank[%d], netLayer[%u], netInstId[%s], "
            "topoInstId[%u], ret[%d]",
            __func__, myRank_, netLayer, netInstance->GetNetInstId().c_str(), topoInstId, ret),
        ret);
    // Keep the physical topoInstId valid for type/endpoint queries. Only project membership.
    const std::vector<u32>* group = nullptr;
    CHK_RET(FindLocalVirtualGroup(netLayer, group));
    if (group != nullptr) {
        std::vector<u32> filtered;
        std::set_intersection(ranks.begin(), ranks.end(), group->begin(), group->end(), std::back_inserter(filtered));
        ranks.swap(filtered);
        rankNum = static_cast<u32>(ranks.size());
        CHK_PRT_RET(
            ranks.empty(),
            HCCL_ERROR(
                "[RankGraph::GetRanksByTopoInst] no ranks in local virtual group, myRank[%d], netLayer[%u], "
                "topoInstId[%u]",
                myRank_, netLayer, topoInstId),
            HCCL_E_NOT_FOUND);
    }
    return HCCL_SUCCESS;
}

HcclResult RankGraph::GetEndpointNum(uint32_t layer, uint32_t topoInstId, uint32_t* num) const
{
    auto peer = GetPeer(myRank_);
    if (peer == nullptr) {
        HCCL_ERROR("[RankGraph::GetEndpointNum] Peer is nullptr at netLayer [%u]", layer);
        return HCCL_E_PTR;
    }
    auto ifacesVec = peer->GetIfacesByLayer(layer);
    uint32_t sum = 0;
    for (auto& iface : ifacesVec) {
        if (iface->GetTopoInstId() == topoInstId) {
            sum += iface->GetLinkProtocols().size();
        }
    }
    *num = sum;
    return HCCL_SUCCESS;
}

HcclResult RankGraph::GetIpIndex(RankId rankId, const IpAddress& ipAddr, uint32_t& ipIndex) const
{
    auto rankIt = ipIndexMap_.find(rankId);
    if (rankIt == ipIndexMap_.end()) {
        HCCL_WARNING("[RankGraph::GetIpIndex] rankId [%u] not in ipIndexMap", rankId);
        return HCCL_E_NOT_FOUND;
    }
    const auto& entries = rankIt->second;
    auto addrIt = std::lower_bound(
        entries.begin(), entries.end(), ipAddr, [](const std::pair<IpAddress, uint32_t>& entry, const IpAddress& addr) {
            return entry.first < addr;
        });
    if (addrIt == entries.end() || ipAddr < addrIt->first) {
        HCCL_WARNING("[RankGraph::GetIpIndex] ipAddr[%s] not found for rankId [%u]", ipAddr.GetIpStr().c_str(), rankId);
        return HCCL_E_NOT_FOUND;
    }
    ipIndex = addrIt->second;
    return HCCL_SUCCESS;
}

HcclResult GetCommAddr(CommAddr& commAddr, const IpAddress& ipAddr)
{
    s32 family = ipAddr.GetFamily();
    if (family == AF_INET) {
        string addr = ipAddr.GetIpStr();
        if (ipAddr.IsEID(addr)) {
            commAddr.type = COMM_ADDR_TYPE_EID;
            const auto& eid = ipAddr.GetEid();
            for (u32 i = 0; i < URMA_EID_LEN && i < sizeof(commAddr.eid); i++) {
                commAddr.eid[i] = eid.raw[i];
            }
        } else {
            commAddr.type = COMM_ADDR_TYPE_IP_V4;
            commAddr.addr = ipAddr.GetBinaryAddress().addr;
        }
    } else if (family == AF_INET6) {
        commAddr.type = COMM_ADDR_TYPE_IP_V6;
        commAddr.addr6 = ipAddr.GetBinaryAddress().addr6;
    } else {
        HCCL_ERROR("invalid commAddrType");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

EndpointLocType AddrPositionToEndpointLoc(AddrPosition pos)
{
    switch (pos) {
        case AddrPosition::HOST:
            return ENDPOINT_LOC_TYPE_HOST;
        case AddrPosition::DEVICE:
            return ENDPOINT_LOC_TYPE_DEVICE;
        default:
            return ENDPOINT_LOC_TYPE_RESERVED;
    }
}

HcclResult
RankGraph::GetEndpointDesc(uint32_t layer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc) const
{
    auto peer = GetPeer(myRank_);
    CHK_PTR_NULL(peer);

    auto ifacesVec = peer->GetIfacesByLayer(layer);
    uint32_t count = 0;

    // 找到 topoInstId 匹配的 iface
    for (const auto& iface : ifacesVec) {
        if (iface->GetTopoInstId() != topoInstId) {
            continue;
        }

        // 对该 iface，从 endpointToIfaceMap_ 中找出所有匹配的 EndpointDesc
        // 一个 iface 可能对应多个 protocol（即多个 EndpointDesc）
        const auto& endpointMap = peer->GetEndpointToIfaceMap();
        for (const auto& entry : endpointMap) {
            std::pair<CommAddr, CommProtocol> endpoint = entry.first;
            const std::shared_ptr<NetInstance::ConnInterface>& mappedIface = entry.second;

            if (mappedIface == nullptr || *mappedIface != *iface) {
                continue;
            }

            // 检查输出缓冲区是否足够
            if (count >= *descNum) {
                HCCL_ERROR(
                    "[RankGraph::GetEndpointDesc] endpointDesc array too small: "
                    "need %u, given %u",
                    count + 1, *descNum);
                return HCCL_E_PARA;
            }

            endpointDesc[count].commAddr = endpoint.first;
            endpointDesc[count].protocol = endpoint.second;
            endpointDesc[count].loc.locType = AddrPositionToEndpointLoc(iface->GetPos());

            HCCL_INFO(
                "[RankGraph::GetEndpointDesc] local type is %d, protocol %d", endpointDesc[count].loc.locType,
                endpointDesc[count].protocol);
            count++;
        }
    }

    *descNum = count;
    return HCCL_SUCCESS;
}

HcclResult RankGraph::GetEndpointInfo(
    uint32_t rankId, const EndpointDesc* endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info) const
{
    if (endpointDesc == nullptr || info == nullptr) {
        HCCL_ERROR("[GetEndpointInfo] Invalid parameter");
        return HCCL_E_PTR;
    }
    const std::shared_ptr<NetInstance::Peer> peer = GetPeer(rankId);
    if (peer == nullptr) {
        HCCL_ERROR("[RankGraph::GetEndpointInfo] Peer is nullptr for rankId [%u]", rankId);
        return HCCL_E_PTR;
    }

    // 查找接口
    auto key = std::make_pair(endpointDesc->commAddr, endpointDesc->protocol);
    const auto& endpointToIfaceMap = peer->GetEndpointToIfaceMap();
    auto it = endpointToIfaceMap.find(key);
    if (it == endpointToIfaceMap.end()) {
        HCCL_ERROR("[GetEndpointInfo] No matching interface found");
        return HCCL_E_NOT_FOUND;
    }
    const auto& iface = it->second;
    // 填充信息
    switch (endpointAttr) {
        case ENDPOINT_ATTR_BW_COEFF: {
            if (infoLen != sizeof(EndpointAttrBwCoeff)) {
                HCCL_ERROR(
                    "[GetEndpointInfo] Size mismatch: expected %zu, actual %u", sizeof(EndpointAttrBwCoeff), infoLen);
                return HCCL_E_PARA;
            }
            *(static_cast<EndpointAttrBwCoeff*>(info)) = iface->GetPortNum();
            break;
        }
        case ENDPOINT_ATTR_DIE_ID: {
            if (infoLen != sizeof(EndpointAttrDieId)) {
                HCCL_ERROR(
                    "[GetEndpointInfo] Size mismatch: expected %zu, actual %u", sizeof(EndpointAttrDieId), infoLen);
                return HCCL_E_PARA;
            }
            *(static_cast<EndpointAttrDieId*>(info)) = iface->GetLocalDieId();
            HCCL_INFO("GetEndpointInfo rankId[%u] iface[%s]", rankId, iface->Describe().c_str());
            break;
        }
        case ENDPOINT_ATTR_LOCATION: {
            if (infoLen != sizeof(EndpointAttrLocation)) {
                HCCL_ERROR(
                    "[GetEndpointInfo] Size mismatch: expected %zu, actual %u", sizeof(EndpointAttrLocation), infoLen);
                return HCCL_E_PARA;
            }
            *(static_cast<EndpointAttrLocation*>(info)) = iface->GetPos();
            break;
        }
        default: {
            HCCL_ERROR("[GetEndpointInfo] Invalid endpointAttr [%d]", endpointAttr);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

bool RankGraph::IsSymmetric(const u32 netLayer) const
{
    if (netInsts_.at(netLayer).size() == 0) {
        THROW<NullPtrException>(
            StringFormat("[RankGraph][IsSymmetric] RankGraph no netInstance on net layer %u", netLayer));
    }
    std::unordered_set<u32> rankSize;
    for (const auto& netInst : netInsts_.at(netLayer)) {
        rankSize.insert(netInst.second->GetRankSize());
    }
    return rankSize.size() == 1;
}

std::shared_ptr<NetInstance> GetOrCreateNetInstance(
    u32 netLayer, const string& netInstId, NetType type, Level2Id2NetInst& netInsts, RankGraph* rankGraph)
{
    std::shared_ptr<NetInstance> netInstance;

    // 若netLayer和netInstId对应netInstance没创建则创建
    if (netInsts[netLayer].count(netInstId) == 0) {
        if (type == NetType::TOPO_FILE_DESC) {
            netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
        } else if (type == NetType::CLOS) {
            netInstance = std::make_shared<ClosNetInstance>(netLayer, netInstId);
        }
        netInsts[netLayer][netInstId] = netInstance;
        CHK_RET_THROW(
            InternalException, "[GetOrCreateNetInstance] AddNetInstance failed",
            rankGraph->AddNetInstance(netInstance));

        HCCL_DEBUG(
            "[CreateNetInstance] create netInstance success, netLayer[%u] netInstId[%s] type[%s].", netLayer,
            netInstId.c_str(), type.Describe().c_str());
    } else {
        // 若netInstance存在, type不一致则报错
        NetType curType = netInsts[netLayer][netInstId]->GetNetType();
        if (curType != type) {
            HCCL_WARNING(
                "[CreateNetInstance]FabType [%s] and [%s] no match", curType.Describe().c_str(),
                type.Describe().c_str());
            return nullptr;
        }
        // 若netInstance存在, type一致则直接获取
        netInstance = netInsts[netLayer][netInstId];
    }
    return netInstance;
}

HcclResult GetNewNodeInfo(
    u32 layer, RankId newRankId, const NetInstance::Link& oldLink, shared_ptr<NetInstance>& newNetInstance,
    RankId2PeerMap& tmpPeers, shared_ptr<NetInstance::Node>& newNode, shared_ptr<NetInstance::ConnInterface>& newIface,
    bool isSource)
{
    shared_ptr<NetInstance::Node> oldNode;
    shared_ptr<NetInstance::ConnInterface> oldIface;
    if (isSource) {
        oldNode = oldLink.GetSourceNode();
        oldIface = oldLink.GetSourceIface();
    } else {
        oldNode = oldLink.GetTargetNode();
        oldIface = oldLink.GetTargetIface();
    }

    NetInstance::Node::NodeType type = oldNode->GetType();
    if (type == NetInstance::Node::NodeType::PEER) {
        newIface = oldIface;
        newNode = tmpPeers.at(newRankId);
        tmpPeers.at(newRankId)->AddConnInterface(layer, newIface);
    } else if (type == NetInstance::Node::NodeType::FABRIC) {
        newIface = nullptr;
        newNode = oldNode;
        if (!newNetInstance->HasNode(newNode->GetNodeId())) {
            CHK_RET(newNetInstance->AddNode(newNode));
        }
    } else {
        HCCL_ERROR(
            "[%s] newRankId[%d] oldLink Node isSource[%d] type[%s] is not supported.", __func__, newRankId, isSource,
            type.Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

bool NeedUpdateTopoInstForSubGraph(const NetInstance* oldNetInstance, u32 topoInstId, RankId parentMyRank)
{
    auto topoInstIter = oldNetInstance->topoInsts_.find(topoInstId);
    if (topoInstIter == oldNetInstance->topoInsts_.end()) {
        HCCL_DEBUG(
            "[SubRankGraph][NeedUpdateTopoInstForSubGraph] skip topoInstId[%u], parentMyRank[%d], "
            "not found in parent netInstId[%s]",
            topoInstId, parentMyRank, oldNetInstance->GetNetInstId().c_str());
        return false;
    }
    if (topoInstIter->second == nullptr) {
        HCCL_WARNING(
            "[SubRankGraph][NeedUpdateTopoInstForSubGraph] skip topoInstId[%u], parentMyRank[%u], "
            "topoInst is null in parent netInstId[%s]",
            topoInstId, parentMyRank, oldNetInstance->GetNetInstId().c_str());
        return false;
    }
    return topoInstIter->second->ranks.count(parentMyRank) > 0;
}

HcclResult AddNewLink(
    u32 layer, const NetInstance::Link& oldLink, RankId srcNewRankId, RankId dstNewRankId,
    shared_ptr<NetInstance>& newNetInstance, RankId2PeerMap& tmpPeers, const NetInstance* oldNetInstance,
    RankId parentMyRank)
{
    if (oldLink.GetHop() > 1 && oldLink.GetType() != LinkType::PEER2NET) {
        return HCCL_SUCCESS;
    }

    shared_ptr<NetInstance::ConnInterface> newSourceIface;
    shared_ptr<NetInstance::ConnInterface> newTargetIface;
    shared_ptr<NetInstance::Node> newSourceNode;
    shared_ptr<NetInstance::Node> newTargetNode;
    CHK_RET(
        GetNewNodeInfo(layer, srcNewRankId, oldLink, newNetInstance, tmpPeers, newSourceNode, newSourceIface, true));
    CHK_RET(
        GetNewNodeInfo(layer, dstNewRankId, oldLink, newNetInstance, tmpPeers, newTargetNode, newTargetIface, false));
    shared_ptr<NetInstance::Link> link = make_shared<NetInstance::Link>(
        newSourceNode, newTargetNode, newSourceIface, newTargetIface, oldLink.GetType(), oldLink.GetLinkProtocols(),
        oldLink.GetLinkDirection(), oldLink.GetHop());

    newNetInstance->AddLink(link);
    if (newSourceIface != nullptr) {
        u32 sourceTopoInstId = newSourceIface->GetTopoInstId();
        TopoType sourceTopoType = newSourceIface->GetTopoType();
        if (NeedUpdateTopoInstForSubGraph(oldNetInstance, sourceTopoInstId, parentMyRank)) {
            newNetInstance->UpdateTopoInst(sourceTopoInstId, sourceTopoType, srcNewRankId);
        }
    }
    if (newTargetIface != nullptr) {
        u32 targetTopoInstId = newTargetIface->GetTopoInstId();
        TopoType targetTopoType = newTargetIface->GetTopoType();
        if (NeedUpdateTopoInstForSubGraph(oldNetInstance, targetTopoInstId, parentMyRank)) {
            newNetInstance->UpdateTopoInst(targetTopoInstId, targetTopoType, dstNewRankId);
        }
    }

    for (const auto& pair : newNetInstance->topoInsts_) {
        uint32_t topoInstId = pair.first;
        if (pair.second == nullptr) {
            HCCL_ERROR("[%s] topoInstId %u has no TopoInst", __func__, topoInstId);
            return HCCL_E_PTR;
        }
        auto topoType = pair.second->topoType;
        if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
            HCCL_DEBUG("[SubRankGraph] topoInstId[%u] topoType[%d]", topoInstId, topoType);
        }
    }
    HCCL_DEBUG(
        "[RankGraph][AddNewLink] srcNewRankId[%d] dstNewRankId[%d] newLink[%s]", srcNewRankId, dstNewRankId,
        link->Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult AddGroupLinks(
    const vector<RankId>& rankIds, const NetInstance* oldNetInstance, shared_ptr<NetInstance>& newNetInstance,
    RankId2PeerMap& tmpPeers, RankId parentMyRank)
{
    set<RankId> newRankIds = newNetInstance->GetRankIds();
    u32 layer = newNetInstance->GetNetLayer();
    if (oldNetInstance == nullptr) {
        HCCL_ERROR("[%s] oldNetInstance is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (newRankIds.size() == 1) {
        RankId singleId = *newRankIds.begin();
        newNetInstance->UpdateTopoInst(0, TopoType::MESH_1D, singleId);
        return HCCL_SUCCESS;
    }
    for (RankId srcRankId : newRankIds) {
        for (RankId dstRankId : newRankIds) {
            if (srcRankId == dstRankId) {
                continue;
            }
            vector<NetInstance::Path> oldPaths = oldNetInstance->GetPaths(rankIds[srcRankId], rankIds[dstRankId]);
            for (auto& oldPath : oldPaths) {
                for (auto& oldLink : oldPath.links) {
                    CHK_RET(AddNewLink(
                        layer, oldLink, srcRankId, dstRankId, newNetInstance, tmpPeers, oldNetInstance, parentMyRank));
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

namespace {
    // 子域收窄 hcclUseNetLayer 时，只保留允许层的地址/协议到接口映射。
    HcclResult CopySelectedEndpointMap(const NetInstance::Peer& source, NetInstance::Peer& target, u32 selectedLayers)
    {
        for (u32 layer : source.GetLevels()) {
            if (layer >= MAX_NET_LAYER) {
                HCCL_ERROR(
                    "[RankGraph][%s] invalid endpoint layer[%u], supportedRange[0,%u], sourceRank[%d], subRank[%d], "
                    "selectedLayers[%s]",
                    __func__, layer, MAX_NET_LAYER - 1, source.GetRankId(), target.GetRankId(),
                    hccl::NetLayerSelection::FormatLayers(selectedLayers).c_str());
                return HCCL_E_PARA;
            }
            if ((selectedLayers & (1U << layer)) == 0) {
                continue;
            }
            const auto ifaces = source.GetIfacesByLayer(layer);
            for (const auto& iface : ifaces) {
                CHK_PTR_NULL(iface);
                CommAddr address{};
                CHK_RET(GetCommAddr(address, iface->GetAddr()));
                for (const auto& protocol : iface->GetLinkProtocols()) {
                    target.SetEndpointToIface(address, LinkProtocolToCommProtocol(protocol), iface);
                }
            }
        }
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult
RankGraph::AddSubPeers(const std::vector<RankId>& rankIds, RankGraph* subRankGraph, RankId2PeerMap& peers) const
{
    // 遍历rankIds将索引作为子虚拟拓扑的rankId构造subPeer并添加到subRankGraph
    s32 rankSize = rankIds.size();
    for (RankId subRankId = 0; subRankId < rankSize; ++subRankId) {
        RankId rankId = rankIds[subRankId];
        shared_ptr<NetInstance::Peer> oldPeer = GetPeer(rankId);
        LocalId localId = oldPeer->GetLocalId();
        LocalId replacedLocalId = oldPeer->GetReplacedLocalId();
        DeviceId deviceId = oldPeer->GetDeviceId();
        u32 devicePort = oldPeer->GetDevicePort();
        u32 hostPort = oldPeer->GetHostPort();
        shared_ptr<NetInstance::Peer> subPeer
            = make_shared<NetInstance::Peer>(subRankId, localId, replacedLocalId, deviceId, devicePort, hostPort);
        CHK_RET(subRankGraph->AddPeer(subPeer));
        std::string layer0NetInstId;
        const auto layer0IdentityRet = GetLayer0NetInstanceId(rankId, layer0NetInstId);
        if (layer0IdentityRet == HCCL_SUCCESS) {
            subRankGraph->layer0NetInstIds_.emplace(subRankId, layer0NetInstId);
        } else {
            HCCL_WARNING(
                "[RankGraph][AddSubPeers] layer 0 identity unavailable, parentRank[%d], subRank[%d], ret[%d]; "
                "continue building without copying layer 0 identity",
                rankId, subRankId, layer0IdentityRet);
        }
        peers.emplace(subRankId, subPeer);
        // 保留接口来源供嵌套子域再次选层；单 rank 子图的按层接口列表可能为空。
        const auto source = endpointSourcePeers_.find(rankId);
        const std::shared_ptr<const NetInstance::Peer> endpointSource
            = source == endpointSourcePeers_.end() ? oldPeer : source->second;
        subRankGraph->endpointSourcePeers_.emplace(subRankId, endpointSource);
        const auto& selection = subRankGraph->buildLayerSelection_;
        const auto parentLayers = oldPeer->GetLevels();
        // 默认、全选及等层继承均复制原映射，仅移除父图现有层时重建。
        const bool narrowed
            = selection.has_value() && std::any_of(parentLayers.begin(), parentLayers.end(), [&selection](u32 layer) {
                  return layer < MAX_NET_LAYER && (selection->selectedLayers & (1U << layer)) == 0;
              });
        if (narrowed) {
            // 仅按保留层重建端点映射，避免同地址/协议指向禁用层；接口列表仍由原子图构建流程填充。
            CHK_RET(CopySelectedEndpointMap(*endpointSource, *subPeer, selection->selectedLayers));
        } else {
            const auto& oldEndpointMap = oldPeer->GetEndpointToIfaceMap();
            for (const auto& entry : oldEndpointMap) {
                subPeer->SetEndpointToIface(entry.first.first, entry.first.second, entry.second);
            }
        }
        HCCL_DEBUG(
            "[RankGraph][AddSubPeers] oldRankId[%d] subPeer[%s] add success.", rankId, subPeer->Describe().c_str());
    }
    return HCCL_SUCCESS;
}

RankId GetSubRankId(const vector<RankId>& rankIds, RankId rank)
{
    RankId subRank;

    // rankIds中查找rank, 数组索引即为subMyRank
    auto iter = find(begin(rankIds), end(rankIds), rank);
    if (iter != end(rankIds)) {
        subRank = distance(begin(rankIds), iter);
    } else {
        THROW<InvalidParamsException>(StringFormat("[RankGraph][CreateSubVirtTopo] rankIds has no rank[%d].", rank));
    }

    HCCL_DEBUG("[GetSubRank] rank[%d] subRank[%d].", rank, subRank);
    return subRank;
}

/**
 * 1. 创建子NetInstance
 * 2. peer添加对应netInstance
 */
HcclResult RankGraph::CreateSubNetInstances(
    const std::vector<RankId> rankIds, Level2Id2NetInst& subNetInstances, RankId2PeerMap& peers,
    RankGraph* subRankGraph) const
{
    RankId rankSize = rankIds.size();
    for (RankId subRankId = 0; subRankId < rankSize; ++subRankId) {
        set<u32> curLevels;
        TRY_CATCH_RETURN(curLevels = GetLevels(rankIds[subRankId]););
        for (u32 netLayer : curLevels) {
            CHK_PRT_RET(
                netLayer >= MAX_NET_LAYER,
                HCCL_ERROR("[%s] rank[%d] invalid netLayer[%u]", __func__, rankIds[subRankId], netLayer), HCCL_E_PARA);
            if (subRankGraph->buildLayerSelection_.has_value()
                && (subRankGraph->buildLayerSelection_->selectedLayers & (1U << netLayer)) == 0) {
                continue;
            }
            const NetInstance* oldNetInstance = GetNetInstanceByRankId(netLayer, rankIds[subRankId]);
            if (oldNetInstance == nullptr) {
                HCCL_ERROR("[%s] oldNetInstance is nullptr", __func__);
                return HCCL_E_PTR;
            }
            NetType netType = oldNetInstance->GetNetType();
            string netInstId = oldNetInstance->GetNetInstId();
            shared_ptr<NetInstance> subNetInstance = nullptr;
            TRY_CATCH_RETURN(subNetInstance
                             = GetOrCreateNetInstance(netLayer, netInstId, netType, subNetInstances, subRankGraph););
            if (subNetInstance == nullptr) {
                HCCL_ERROR("[%s] subNetInstance is nullptr", __func__);
                return HCCL_E_PTR;
            }

            shared_ptr<NetInstance::Peer> subPeer = peers.at(subRankId);
            subNetInstance->AddRankId(subRankId);
            CHK_RET(subNetInstance->AddNode(subPeer));
            CHK_RET(subPeer->AddNetInstance(subNetInstance));
            HCCL_DEBUG(
                "[RankGraph][CreateSubNetInstances] subNetInstance subRankId[%d] subType[%s] subNetInstId[%s]",
                subRankId, netType.Describe().c_str(), netInstId.c_str());
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RankGraph::AddSubLinks(
    const std::vector<RankId>& rankIds, RankId2PeerMap& peers, Level2Id2NetInst& subNetInsts, RankId parentMyRank) const
{
    // 遍历subNetInstances，对每一个NetInstance插入Links
    for (u32 netLayer = 0; netLayer < subNetInsts.size(); ++netLayer) {
        for (auto& curNetInstance : subNetInsts[netLayer]) {
            const NetInstance* oldNetInstance = GetNetInstanceByNetInstId(netLayer, curNetInstance.first);
            CHK_RET(AddGroupLinks(rankIds, oldNetInstance, curNetInstance.second, peers, parentMyRank));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RankGraph::CreateSubRankGraph(
    const std::vector<u32>& rankIds, std::unique_ptr<RankGraph>& subRankGraph, std::optional<u32> selectedLayers) const
{
    auto selection = buildLayerSelection_;
    if (selectedLayers.has_value()) {
        u32 availableLayers = 0;
        for (u32 layer : GetLevels(myRank_)) {
            CHK_PRT_RET(
                layer >= MAX_NET_LAYER, HCCL_ERROR("[%s] invalid parent layer[%u]", __func__, layer), HCCL_E_PARA);
            availableLayers |= 1U << layer;
        }
        CHK_PRT_RET(
            *selectedLayers == 0 || (*selectedLayers & ~availableLayers) != 0,
            HCCL_ERROR(
                "[%s] invalid child netLayers[%s], parent availableLayers[%s]", __func__,
                hccl::NetLayerSelection::FormatLayers(*selectedLayers).c_str(),
                hccl::NetLayerSelection::FormatLayers(availableLayers).c_str()),
            HCCL_E_PARA);
        selection = BuildLayerSelection{availableLayers, *selectedLayers};
    }
    // 继承和显式选层均成功后再替换输出，失败保留调用方原结果。
    std::unique_ptr<RankGraph> prepared;
    CHK_RET(CreateSubRankGraphImpl(rankIds, prepared, selection));
    subRankGraph = std::move(prepared);
    return HCCL_SUCCESS;
}

HcclResult RankGraph::CreateSubRankGraphImpl(
    const std::vector<u32>& rankIds, std::unique_ptr<RankGraph>& subRankGraph,
    const std::optional<BuildLayerSelection>& selection) const
{
    // 参数类型转换
    const vector<RankId> subRankIds(rankIds.begin(), rankIds.end());

    // 参数检查, 若rankIds中存在当前virtualTopo不存在的rankId, 抛异
    for (const auto rankId : subRankIds) {
        if (!HasRank(rankId)) {
            HCCL_ERROR("[%s] rankId[%d] is not existed.", __func__, rankId);
            return HCCL_E_PARA;
        }
    }

    // step1: 创建subRankGraph
    RankId subMyRankId;
    TRY_CATCH_RETURN(subMyRankId = GetSubRankId(subRankIds, myRank_););
    subRankGraph = make_unique<RankGraph>(subMyRankId, enableVirtualGroups_);
    // 子图只投影父图已有资源，保留建图选层来源，不能退回未裁剪的查询视图。
    subRankGraph->buildLayerSelection_ = selection;
    subRankGraph->SetLevel0PcieFallback(level0PcieFallback_); // 无UB兜底标记随子图继承
    if (logicSuperPodSplitEnabled_) {
        // 子图rankId是rankIds数组下标，需按父图映射重编
        std::map<RankId, std::string> subLogicSuperPodGroupIds;
        for (RankId subId = 0; subId < static_cast<RankId>(subRankIds.size()); ++subId) {
            auto it = logicSuperPodGroupIds_.find(subRankIds[subId]);
            if (it != logicSuperPodGroupIds_.end()) {
                subLogicSuperPodGroupIds[subId] = it->second;
            }
        }
        subRankGraph->SetLogicSuperPodSplit(logicSuperPodSplitLayer_, subLogicSuperPodGroupIds);
    }

    // 继承父图 ipIndexMap_，子图 rankId 是 rankIds 数组下标，需做映射
    std::unordered_map<RankId, RankIpIndexList> subIpIndexMap;
    for (RankId subRankId = 0; subRankId < static_cast<RankId>(subRankIds.size()); ++subRankId) {
        auto it = ipIndexMap_.find(subRankIds[subRankId]);
        if (it != ipIndexMap_.end()) {
            subIpIndexMap[subRankId] = it->second;
        }
    }
    CHK_RET(subRankGraph->SetIpIndexMap(std::move(subIpIndexMap)));

    // step2: subRankGraph添加subPeers
    RankId2PeerMap peers; // 保存Peer指针以便后续执行Add操作
    CHK_RET(AddSubPeers(subRankIds, subRankGraph.get(), peers));

    // step3: 构造subNetInstances, NetInstance添加RankId和Peer, Peer添加NetInstance
    Level2Id2NetInst subNetInstances(MAX_NET_LAYER); // 保存NetInstance指针以便后续执行Add操作
    CHK_RET(CreateSubNetInstances(subRankIds, subNetInstances, peers, subRankGraph.get()));

    // step4: subNetInstances添加Links, Peer添加ConnIfaces
    CHK_RET(AddSubLinks(subRankIds, peers, subNetInstances, myRank_));

    // step5: 设置innerRanks
    if (subRankGraph->GetLevels(subMyRankId).count(0) != 0) {
        // 保留 layer 0 时沿用原分组计算；禁用时仅继承归属，不补回通信层。
        CHK_RET(subRankGraph->InitInnerRanks());
    } else {
        // 无 layer 0 时按父域本地成员投影，并转换成子域 rankId。
        std::set<RankId> subInnerRanks;
        for (RankId rank = 0; rank < static_cast<RankId>(subRankIds.size()); ++rank) {
            if (innerRanks_.count(subRankIds[rank]) != 0) {
                subInnerRanks.insert(rank);
            }
        }
        CHK_RET(subRankGraph->SetInnerRanks(subInnerRanks));
    }
    // step6: 构造完成；子图无 rankTable/topoInfo，不重算物理超节点层，继承父图分段上界。
    if (enableVirtualGroups_) {
        subRankGraph->physicalSuperPodLayer_ = physicalSuperPodLayer_;
        CHK_RET(subRankGraph->BuildVirtualGroups());
    }
    subRankGraph->InitFinish();

    if (selection.has_value()) {
        HCCL_RUN_INFO(
            "[RankGraph][%s] subgraph build success, parentRank[%d], subRank[%d], rankSize[%zu], "
            "selectedLayers[%s], availableLayers[%s], innerRankSize[%zu]",
            __func__, myRank_, subMyRankId, rankIds.size(),
            hccl::NetLayerSelection::FormatLayers(selection->selectedLayers).c_str(),
            hccl::NetLayerSelection::FormatLayers(selection->availableLayers).c_str(),
            subRankGraph->GetInnerRanks().size());
    } else {
        HCCL_INFO("[subRankGraph] Build success!");
    }
    // 优化：超大图全量Dump构造Describe大字符串，日志级别不满足时直接跳过
    if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
        subRankGraph->Dump();
    }
    return HCCL_SUCCESS;
}

std::vector<char> RankGraph::GetPackedData(const std::vector<std::pair<u32, RankId>>& netLayerRankPairs) const
{
    std::vector<u32> numVec;
    std::vector<LinkData> links;
    u32 netLayerRankPairsNum = netLayerRankPairs.size();

    HCCL_DEBUG("netLayerRankPairs Num=%u", netLayerRankPairsNum);

    for (const auto& it : netLayerRankPairs) {
        auto paths = GetPaths(it.first, myRank_, it.second);
        numVec.push_back(paths.size());
        HCCL_DEBUG("RankGraph::GetPackedData: netLayer=%u, srcRank=%u, dstRank=%u", it.first, myRank_, it.second);
        for (const auto& path : paths) {
            links.emplace_back(MakeLinkData(path));
            HCCL_DEBUG("RankGraph::GetPackedData: %s", links.back().Describe().c_str());
        }
    }
    if (links.empty()) {
        HCCL_WARNING("[RankGraph][GetPackedData] connected links are empty");
    }

    std::vector<char> result;
    BinaryStream binaryStream;
    u32 linkSize = links.size();
    binaryStream << netLayerRankPairsNum;
    binaryStream << linkSize;
    HCCL_DEBUG("netLayerRankPairsNum=%u, linkSize=%u", netLayerRankPairsNum, linkSize);
    u32 idx = 0;
    for (const auto& it : netLayerRankPairs) {
        binaryStream << it.first;
        binaryStream << it.second;
        binaryStream << numVec[idx];
        HCCL_DEBUG("netLayer=%u, RankId=%u, num=%u", it.first, it.second, numVec[idx]);
        idx++;
    }

    for (const auto& link : links) {
        binaryStream << link.GetUniqueId();
    }
    binaryStream.Dump(result);

    return result;
}

string RankIds2Str(const set<RankId>& rankIds)
{
    stringstream ranks;
    for (auto it = rankIds.begin(); it != rankIds.end(); ++it) {
        if (it != rankIds.begin()) {
            ranks << ", ";
        }
        ranks << *it;
    }
    return ranks.str();
}

void RankGraph::Dump() const
{
    HCCL_DEBUG("RankGraph Dump:");
    HCCL_DEBUG("myRank: %d", myRank_);
    if (buildLayerSelection_.has_value()) {
        HCCL_DEBUG(
            "netLayer selection: mode[SELECTED], selectedLayers[%s], availableLayers[%s]",
            hccl::NetLayerSelection::FormatLayers(buildLayerSelection_->selectedLayers).c_str(),
            hccl::NetLayerSelection::FormatLayers(buildLayerSelection_->availableLayers).c_str());
    } else {
        HCCL_DEBUG("netLayer selection: mode[DEFAULT], no layer filtering");
    }
    HCCL_DEBUG("innerRanks: [%s]", RankIds2Str(innerRanks_).c_str());
    HCCL_DEBUG("peers:");
    for (const auto& peer : peers_) {
        HCCL_DEBUG("%s", peer.second->Describe().c_str());
    }
    HCCL_DEBUG("netInsts:");
    for (uint32_t i = 0; i < netInsts_.size(); ++i) {
        for (auto& netInst : netInsts_[i]) {
            HCCL_DEBUG("[netLayer=%u, netInstId=%s]", i, netInst.first.c_str());
            HCCL_DEBUG("%s", netInst.second->Describe().c_str());
            HCCL_DEBUG("rankIds: [%s]", RankIds2Str(netInst.second->GetRankIds()).c_str());
            HCCL_DEBUG("peers:");
            for (const auto& peerMapIt : netInst.second->GetPeers()) {
                HCCL_DEBUG("%s", peerMapIt.second->Describe().c_str());
            }
            HCCL_DEBUG("fabrics:");
            for (const auto& fabric : netInst.second->GetFabrics()) {
                HCCL_DEBUG("%s", fabric->Describe().c_str());
            }
            set<NodeId> nodeIds{};
            HCCL_DEBUG("Graph Nodes:");
            netInst.second->GetGraph().TraverseNode([&](shared_ptr<NetInstance::Node> node) {
                nodeIds.insert(node->GetNodeId());
                HCCL_DEBUG("%s", node->Describe().c_str());
            });
            HCCL_DEBUG("Graph Links:");
            for (NodeId nodeId : nodeIds) {
                netInst.second->GetGraph().TraverseEdge(nodeId, [&](shared_ptr<NetInstance::Link> link) {
                    HCCL_DEBUG("%s", link->Describe().c_str());
                });
            }
        }
    }
}
CommProtocol LinkProtocolToCommProtocol(const LinkProtocol& linkProtocol)
{
    constexpr std::pair<LinkProtocol, CommProtocol> protocolPairs[]
        = {{LinkProtocol::UB_CTP, COMM_PROTOCOL_UB_CTP}, {LinkProtocol::UB_TP, COMM_PROTOCOL_UBC_TP},
           {LinkProtocol::ROCE, COMM_PROTOCOL_ROCE},     {LinkProtocol::HCCS, COMM_PROTOCOL_HCCS},
           {LinkProtocol::PCIE, COMM_PROTOCOL_PCIE},     {LinkProtocol::UB_MEM, COMM_PROTOCOL_UB_MEM},
           {LinkProtocol::UBOE, COMM_PROTOCOL_UBOE},     {LinkProtocol::UB_RTP, COMM_PROTOCOL_UB_RTP}};

    for (const auto& p : protocolPairs) {
        if (p.first == linkProtocol) {
            return p.second;
        }
    }

    return COMM_PROTOCOL_RESERVED;
}
} // namespace Hccl
