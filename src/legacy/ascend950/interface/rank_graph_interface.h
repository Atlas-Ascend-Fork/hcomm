/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANK_GRAPH_INTERFACE_H
#define RANK_GRAPH_INTERFACE_H

#include <map>
#include <shared_mutex>
#include <tuple>
#include <utility>
#include <vector>
#include "hccl_rank_graph.h"
#include "hccl_types.h"
#include "rank_gph.h"

namespace Hccl {

class IRankGraph {
public:
    IRankGraph(void* rankGraphPtr) : rankGraphPtr_(rankGraphPtr) {}

    virtual ~IRankGraph() = default;

    HcclResult GetRankId(uint32_t* rank);
    HcclResult GetRankSize(uint32_t* rankSize);
    HcclResult GetDevicePort(const uint32_t rank, uint32_t* devPort);
    HcclResult GetHostPort(const uint32_t rank, uint32_t* hostPort);
    HcclResult GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize);
    HcclResult GetRankGraphInfo(void** graph, uint32_t* len);
    HcclResult GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum);
    HcclResult GetInstTopoTypeByNetLayer(uint32_t netLayer, CommTopo* topoType);
    HcclResult GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum);
    HcclResult GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** rankList, uint32_t* rankNum);
    HcclResult GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize);
    HcclResult GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum);
    HcclResult GetTopoType(const uint32_t netLayer, const uint32_t topoInstId, CommTopo* topoType);
    HcclResult
    GetRanksByTopoInst(const uint32_t netLayer, const uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum);
    HcclResult GetEndpointNum(uint32_t netLayer, uint32_t topoInstId, uint32_t* num);
    HcclResult GetEndpointDesc(uint32_t netLayer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc);
    HcclResult GetEndpointInfo(
        uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info);
    HcclResult GetDeviceId(uint32_t rankId, uint32_t* deviceId);

private:
    // 六接口查询结果按入参缓存：条目写入后不再修改，返回指针在通信域销毁前始终有效。
    // cacheMutex_只保护缓存表的查找/插入（读锁快路径、写锁慢路径），不覆盖借用期。
    // v2路径RankGraph对象在通信域生命周期内不变（CollComm::rankGraphOwner_只初始化一次），故本层不提供缓存失效入口。
    struct IdListCacheEntry {
        std::vector<uint32_t> values;
        uint32_t count{0};
    };
    struct LinkListCacheEntry {
        std::vector<CommLink> values;
        uint32_t count{0};
    };

    void* rankGraphPtr_;
    mutable std::shared_mutex cacheMutex_;
    // netLayers只依赖通信域，其余缓存按入参区分
    bool netLayersCached_{false};
    IdListCacheEntry netLayersCache_;
    std::map<uint32_t, IdListCacheEntry> instRankListCache_; // key: netLayer
    std::map<uint32_t, IdListCacheEntry> instSizeListCache_; // key: netLayer
    std::map<uint32_t, IdListCacheEntry> topoInstsCache_;    // key: netLayer
    // key: netLayer+srcRank+dstRank
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, LinkListCacheEntry> linkListCache_;
    // key: netLayer+topoInstId
    std::map<std::pair<uint32_t, uint32_t>, IdListCacheEntry> ranksByTopoInstCache_;
};

} // namespace Hccl

#endif // RANK_GRAPH_INTERFACE_H
