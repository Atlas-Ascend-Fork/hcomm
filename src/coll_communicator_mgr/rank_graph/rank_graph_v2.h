/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_RANK_GRAPH_V2_H
#define HCOMM_RANK_GRAPH_V2_H
#include <mutex>
#include "rank_graph_base.h"
#include "rank_graph_interface.h"
namespace hccl {

class RankGraphV2 : public RankGraph {
public:
    RankGraphV2();
    RankGraphV2(void* rankGraphPtr);
    ~RankGraphV2() override;

    // 仅在发布给查询方前调用一次；不改原图、不读环境，失败保持原视图。
    // 使用图中保存的选层结果，校验实际层未超出允许范围；完整成员连通性由算法检查。
    HcclResult InitNetLayerSelection();

    HcclResult GetRankId(uint32_t* rank) override;
    HcclResult GetRankSize(uint32_t* rankSize) override;
    HcclResult
    GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize) override;
    HcclResult GetRankGraphInfo(GraphType type, void** graph, uint32_t* len) override;
    HcclResult GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum) override;
    HcclResult GetInstTopoTypeByNetLayer(uint32_t netLayer, CommTopo* topoType) override;
    HcclResult GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum) override;
    HcclResult GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** rankList, uint32_t* rankNum) override;
    HcclResult GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize) override;
    HcclResult GetDevicePort(const uint32_t rank, uint32_t* devPort) override;
    HcclResult GetListenPort(const uint32_t rank, uint32_t* listenPort, EndpointLocType locType) override;
    HcclResult GetNetInstId(const uint32_t rank, std::string& netInstId);
    HcclResult GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum);
    HcclResult GetTopoType(const uint32_t netLayer, const uint32_t topoInstId, CommTopo* topoType);
    HcclResult
    GetRanksByTopoInst(const uint32_t netLayer, const uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum);
    HcclResult GetEndpointNum(uint32_t netLayer, uint32_t topoInstId, uint32_t* num);
    HcclResult GetEndpointDesc(uint32_t netLayer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc);
    HcclResult GetEndpointInfo(
        uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen,
        void* info) override;
    HcclResult GetDeviceId(uint32_t rankId, uint32_t* deviceId) override;
    bool IsLevel0PcieFallback() const override; // 无UB盖板兜底场景：本rank的level0为pcie fallback层
    HcclResult GetIpIndex(uint32_t rankId, const Hccl::IpAddress& ipAddr, uint32_t& ipIndex) const override;

private:
    // 选层图拒绝查询未选择的层；默认图沿用原接口检查。
    HcclResult CheckNetLayer(uint32_t netLayer) const;
    // 仅在禁用 layer 0 且保留本地身份时允许分组兼容查询，不授权通信链路。
    bool IsLayer0MetadataQuery(uint32_t netLayer) const;
    // 一次性缓存本地成员及各组人数，同时缓存失败结果供后续查询复用。
    HcclResult PrepareLayer0Metadata();
    // 仅缓存服务器分组，借用数组初始化后不再修改，不包含端点或通信边。
    std::once_flag layer0MetadataOnce_;
    HcclResult layer0MetadataResult_{HCCL_E_NOT_FOUND};
    std::vector<uint32_t> layer0InstSizes_;
    std::vector<uint32_t> layer0LocalRanks_;
    bool hasNetLayerSelection_{false};
    uint32_t selectedLayerMask_{0};
    std::unique_ptr<Hccl::IRankGraph> pImpl;
    Hccl::RankGraph* rankGraph_{nullptr}; // 由void*构造时转换而来，供标志类查询直读，绕开IRankGraph
};
} // namespace hccl
#endif
