/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_graph_v2.h"
#include "net_layer_config.h"

namespace hccl {

RankGraphV2::RankGraphV2() {}

RankGraphV2::~RankGraphV2() {}

RankGraphV2::RankGraphV2(void* rankGraphPtr) : rankGraph_(static_cast<Hccl::RankGraph*>(rankGraphPtr))
{
    pImpl = std::make_unique<Hccl::IRankGraph>(rankGraphPtr);
}

bool RankGraphV2::IsLevel0PcieFallback() const { return rankGraph_ != nullptr && rankGraph_->IsLevel0PcieFallback(); }

HcclResult RankGraphV2::InitNetLayerSelection()
{
    CHK_PTR_NULL(pImpl);
    CHK_PTR_NULL(rankGraph_);
    const auto& built = rankGraph_->GetBuildLayerSelection();
    if (!built.has_value()) {
        return HCCL_SUCCESS;
    }
    const auto rankId = rankGraph_->GetMyRank();
    CHK_PRT_RET(
        !rankGraph_->HasRank(rankId), HCCL_ERROR("[RankGraphV2] local rank[%d] is absent", rankId), HCCL_E_PARA);
    // 使用本rank层集合的实际长度，不混用全图层数与本rank层数组。
    const auto layers = rankGraph_->GetLevels(rankId);
    uint32_t actualLayers = 0;
    for (const auto layer : layers) {
        CHK_PRT_RET(
            layer >= Hccl::MAX_NET_LAYER,
            HCCL_ERROR(
                "[RankGraphV2][%s] invalid graph layer[%u], rank[%d], supported layer count[%u]", __func__, layer,
                rankId, Hccl::MAX_NET_LAYER),
            HCCL_E_PARA);
        actualLayers |= 1U << layer;
    }
    CHK_PRT_RET(
        (actualLayers & ~built->selectedLayers) != 0,
        HCCL_ERROR(
            "[RankGraphV2][%s] graph contains unselected layers, rank[%d], actualLayers[%s], "
            "selectedLayers[%s], availableLayers[%s]",
            __func__, rankId, NetLayerSelection::FormatLayers(actualLayers).c_str(),
            NetLayerSelection::FormatLayers(built->selectedLayers).c_str(),
            NetLayerSelection::FormatLayers(built->availableLayers).c_str()),
        HCCL_E_INTERNAL);
    // 仅保存查询权限；层列表复用 IRankGraph 对实际图的缓存，不补回子图缺失层。
    selectedLayerMask_ = actualLayers;
    hasNetLayerSelection_ = true;
    HCCL_INFO("[RankGraphV2] selected netLayers[%s]", NetLayerSelection::FormatLayers(selectedLayerMask_).c_str());
    return HCCL_SUCCESS;
}

HcclResult RankGraphV2::CheckNetLayer(uint32_t netLayer) const
{
    if (hasNetLayerSelection_ && (netLayer >= Hccl::MAX_NET_LAYER || (selectedLayerMask_ & (1U << netLayer)) == 0)) {
        HCCL_ERROR(
            "[RankGraphV2] netLayer[%u] is outside selectedLayers[%s]", netLayer,
            NetLayerSelection::FormatLayers(selectedLayerMask_).c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

bool RankGraphV2::IsLayer0MetadataQuery(uint32_t netLayer) const
{
    if (netLayer != 0 || !hasNetLayerSelection_ || (selectedLayerMask_ & 1U) != 0 || rankGraph_ == nullptr
        || !rankGraph_->GetBuildLayerSelection().has_value()) {
        return false;
    }
    // 子域可用层已裁剪，按实际保留的身份元数据判断，不补回 layer 0 通信层。
    std::string netInstId;
    return rankGraph_->GetLayer0NetInstanceId(rankGraph_->GetMyRank(), netInstId) == HCCL_SUCCESS;
}

HcclResult RankGraphV2::PrepareLayer0Metadata()
{
    std::call_once(layer0MetadataOnce_, [this] {
        layer0MetadataResult_ = rankGraph_->GetLayer0InstSizes(layer0InstSizes_);
        if (layer0MetadataResult_ != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[RankGraphV2][PrepareLayer0Metadata] layer 0 group sizes unavailable, rank[%d], ret[%d]",
                rankGraph_->GetMyRank(), layer0MetadataResult_);
            return;
        }
        // 构图已保存包含本 rank 的本地成员，不再从所有实例成员中重新查找。
        const auto& innerRanks = rankGraph_->GetInnerRanks();
        layer0LocalRanks_.assign(innerRanks.begin(), innerRanks.end());
        HCCL_DEBUG(
            "[RankGraphV2] retained layer 0 grouping metadata only, selectedLayers[%s]",
            NetLayerSelection::FormatLayers(selectedLayerMask_).c_str());
    });
    return layer0MetadataResult_;
}

HcclResult RankGraphV2::GetIpIndex(uint32_t rankId, const Hccl::IpAddress& ipAddr, uint32_t& ipIndex) const
{
    CHK_PTR_NULL(rankGraph_);
    return rankGraph_->GetIpIndex(rankId, ipAddr, ipIndex);
}

HcclResult RankGraphV2::GetReachableRanksPerProtocol(
    uint32_t netLayer, uint32_t selfRank, const std::set<CommProtocol>& protocols,
    std::unordered_map<CommProtocol, std::vector<uint32_t>>& reachableRanks) const
{
    CHK_PTR_NULL(rankGraph_);
    return rankGraph_->GetReachableRanksPerProtocol(netLayer, selfRank, protocols, reachableRanks);
}

HcclResult RankGraphV2::GetRankSize(uint32_t* rankSize) { return pImpl->GetRankSize(rankSize); }

HcclResult RankGraphV2::GetDevicePort(const uint32_t rank, uint32_t* devPort)
{
    CHK_RET(pImpl->GetDevicePort(rank, devPort));
    return HCCL_SUCCESS;
}

HcclResult RankGraphV2::GetListenPort(const uint32_t rank, uint32_t* listenPort, EndpointLocType locType)
{
    if (locType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE) {
        CHK_RET(pImpl->GetDevicePort(rank, listenPort));
    } else if (locType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
        CHK_RET(pImpl->GetHostPort(rank, listenPort));
    } else {
        HCCL_ERROR("[%s] Invalid locType[%d] for rank[%u]", __func__, locType, rank);
        return HcclResult::HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult RankGraphV2::GetNetInstId(const uint32_t rank, std::string& netInstId)
{
    // 机器标识不随通信选层变化，复用保留的 layer 0 身份元数据。
    CHK_PTR_NULL(rankGraph_);
    return rankGraph_->GetLayer0NetInstanceId(rank, netInstId);
}

HcclResult RankGraphV2::GetRankId(uint32_t* rank) { return pImpl->GetRankId(rank); }

HcclResult
RankGraphV2::GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize)
{
    if (IsLayer0MetadataQuery(netLayer)) {
        CHK_PTR_NULL(linkList);
        CHK_PTR_NULL(listSize);
        CHK_PRT_RET(
            !rankGraph_->HasRank(srcRank) || !rankGraph_->HasRank(dstRank),
            HCCL_ERROR("[RankGraphV2] invalid ranks[%u, %u] in disabled layer 0 query", srcRank, dstRank), HCCL_E_PARA);
        // 兼容链路统计查询，但不返回被禁用层的任何可建链端点。
        *linkList = nullptr;
        *listSize = 0;
        return HCCL_SUCCESS;
    }
    CHK_RET(CheckNetLayer(netLayer));
    return pImpl->GetLinks(netLayer, srcRank, dstRank, linkList, listSize);
}

HcclResult RankGraphV2::GetRankGraphInfo([[maybe_unused]] GraphType type, void** graph, uint32_t* len)
{
    return pImpl->GetRankGraphInfo(graph, len);
}

HcclResult RankGraphV2::GetDeviceId(uint32_t rankId, uint32_t* deviceId)
{
    return pImpl->GetDeviceId(rankId, deviceId);
}

HcclResult RankGraphV2::GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum)
{
    return pImpl->GetNetLayers(netLayers, netLayerNum);
}

HcclResult RankGraphV2::GetInstTopoTypeByNetLayer(uint32_t netLayer, CommTopo* topoType)
{
    CHK_RET(CheckNetLayer(netLayer));
    return pImpl->GetInstTopoTypeByNetLayer(netLayer, topoType);
}

HcclResult RankGraphV2::GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum)
{
    if (IsLayer0MetadataQuery(netLayer)) {
        CHK_PTR_NULL(rankNum);
        CHK_RET(PrepareLayer0Metadata());
        *rankNum = static_cast<uint32_t>(layer0LocalRanks_.size());
        return HCCL_SUCCESS;
    }
    CHK_RET(CheckNetLayer(netLayer));
    return pImpl->GetInstSizeByNetLayer(netLayer, rankNum);
}

HcclResult RankGraphV2::GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** rankList, uint32_t* rankNum)
{
    if (IsLayer0MetadataQuery(netLayer)) {
        CHK_PTR_NULL(rankList);
        CHK_PTR_NULL(rankNum);
        CHK_RET(PrepareLayer0Metadata());
        *rankList = layer0LocalRanks_.data();
        *rankNum = static_cast<uint32_t>(layer0LocalRanks_.size());
        return HCCL_SUCCESS;
    }
    CHK_RET(CheckNetLayer(netLayer));
    return pImpl->GetInstRanksByNetLayer(netLayer, rankList, rankNum);
}

HcclResult RankGraphV2::GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize)
{
    if (IsLayer0MetadataQuery(netLayer)) {
        CHK_PTR_NULL(instSizeList);
        CHK_PTR_NULL(listSize);
        CHK_RET(PrepareLayer0Metadata());
        *instSizeList = layer0InstSizes_.data();
        *listSize = static_cast<uint32_t>(layer0InstSizes_.size());
        return HCCL_SUCCESS;
    }
    CHK_RET(CheckNetLayer(netLayer));
    return pImpl->GetInstSizeListByNetLayer(netLayer, instSizeList, listSize);
}

HcclResult RankGraphV2::GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum)
{
    CHK_RET(CheckNetLayer(netLayer));
    return pImpl->GetTopoInstsByLayer(netLayer, topoInsts, topoInstNum);
}

HcclResult RankGraphV2::GetTopoType(const uint32_t netLayer, const uint32_t topoInstId, CommTopo* topoType)
{
    CHK_RET(CheckNetLayer(netLayer));
    return pImpl->GetTopoType(netLayer, topoInstId, topoType);
}

HcclResult
RankGraphV2::GetRanksByTopoInst(const uint32_t netLayer, const uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum)
{
    CHK_RET(CheckNetLayer(netLayer));
    return pImpl->GetRanksByTopoInst(netLayer, topoInstId, ranks, rankNum);
}

HcclResult RankGraphV2::GetEndpointNum(uint32_t netLayer, uint32_t topoInstId, uint32_t* num)
{
    CHK_RET(CheckNetLayer(netLayer));
    return pImpl->GetEndpointNum(netLayer, topoInstId, num);
}

HcclResult
RankGraphV2::GetEndpointDesc(uint32_t netLayer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc)
{
    CHK_RET(CheckNetLayer(netLayer));
    return pImpl->GetEndpointDesc(netLayer, topoInstId, descNum, endpointDesc);
}

HcclResult RankGraphV2::GetEndpointInfo(
    uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info)
{
    return pImpl->GetEndpointInfo(rankId, endPointDesc, endpointAttr, infoLen, info);
}

}; // namespace hccl
