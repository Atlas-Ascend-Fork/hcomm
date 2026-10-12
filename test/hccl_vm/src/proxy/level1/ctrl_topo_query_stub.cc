/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Description: 控制面拓扑信息查询打桩函数（北向劫持）
 *              - HcclRankGraphGetLayers、HcclRankGraphGetLinks等拓扑查询接口作为 Checker 的输入。
 *              与南向劫持底层运行时接口的区别：这些接口位于HCCL调用HComm的北向边界，
 *              根据通信域、rank_table.json和topo.json返回结构化拓扑信息。
 * Create: 2026-07-20
 */

#define HCCL_VM_MODULE "TOPO_QUERY_STUB"

#include <vector>

#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "hccl/hccl_rank_graph.h"
#include "level1_proxy_common.h"
#include "sim_log.h"

namespace {

// 从通信域数据库取得当前rank，供每个桩函数按步骤调用。
HcclResult GetCurrentRank(HcclComm comm, uint64_t& commId, uint32_t& rankId)
{
    commId = reinterpret_cast<uint64_t>(comm);
    auto communicator = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!communicator.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }
    rankId = static_cast<uint32_t>(communicator->rank_id);
    return HCCL_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclRankGraphGetLayers(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum)
{
    // 步骤1：检查通信域和输出参数。
    if (comm == nullptr || netLayers == nullptr || netLayerNum == nullptr) {
        HCCL_VM_ERROR(
            "{}: input pointer is nullptr, comm={:p}, netLayers={:p}, netLayerNum={:p}", __func__, comm,
            static_cast<void*>(netLayers), static_cast<void*>(netLayerNum));
        return HCCL_E_PTR;
    }
    // 步骤2：查询通信域数据库，取得当前rank。
    uint64_t commId = 0;
    uint32_t rankId = 0;
    HcclResult result = GetCurrentRank(comm, commId, rankId);
    if (result != HCCL_SUCCESS) {
        return result;
    }
    // 步骤3：把单例当作数据黑盒，查询当前rank拥有的网络层。
    static std::vector<uint32_t> resultLayers;
    sim::RankTableQueryStatus status = sim::RankTable::Instance().GetNetLayers(rankId, resultLayers);
    if (status != sim::RankTableQueryStatus::SUCCESS) {
        HCCL_VM_ERROR(
            "{}: failed to get network layers for rank {:d}, status={:d}", __func__, rankId, static_cast<int>(status));
        return HCCL_E_INTERNAL;
    }
    // 步骤4：返回函数内静态数组，保证函数返回后地址仍有效。
    *netLayers = resultLayers.data();
    *netLayerNum = static_cast<uint32_t>(resultLayers.size());
    HCCL_VM_INFO("{} success, commId={:d}, rankId={:d}, netLayerNum={:d}", __func__, commId, rankId, *netLayerNum);
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetRanksByLayer(HcclComm comm, uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum)
{
    // 步骤1：检查输入和输出地址。
    if (comm == nullptr || ranks == nullptr || rankNum == nullptr) {
        HCCL_VM_ERROR(
            "{}: input pointer is nullptr, comm={:p}, ranks={:p}, rankNum={:p}", __func__, comm,
            static_cast<void*>(ranks), static_cast<void*>(rankNum));
        return HCCL_E_PTR;
    }
    // 步骤2：取得当前通信域的rank。
    uint64_t commId = 0;
    uint32_t rankId = 0;
    HcclResult result = GetCurrentRank(comm, commId, rankId);
    if (result != HCCL_SUCCESS) {
        return result;
    }
    // 步骤3：向单例查询指定层中的rank列表。
    static std::vector<uint32_t> resultRanks;
    sim::RankTableQueryStatus status = sim::RankTable::Instance().GetRanksByLayer(rankId, netLayer, resultRanks);
    if (status == sim::RankTableQueryStatus::LAYER_NOT_FOUND) {
        HCCL_VM_ERROR("{}: network layer {:d} was not found for rank {:d}", __func__, netLayer, rankId);
        return HCCL_E_PARA;
    }
    if (status != sim::RankTableQueryStatus::SUCCESS) {
        HCCL_VM_ERROR(
            "{}: failed to query ranks, rankId={:d}, netLayer={:d}, status={:d}", __func__, rankId, netLayer,
            static_cast<int>(status));
        return HCCL_E_INTERNAL;
    }
    // 步骤4：填写输出数组和数量。
    *ranks = resultRanks.data();
    *rankNum = static_cast<uint32_t>(resultRanks.size());
    HCCL_VM_INFO(
        "{} success, commId={:d}, rankId={:d}, "
        "netLayer={:d}, rankNum={:d}",
        __func__, commId, rankId, netLayer, *rankNum);
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetRankSizeByLayer(HcclComm comm, uint32_t netLayer, uint32_t* rankNum)
{
    // 步骤1：检查参数。
    if (comm == nullptr || rankNum == nullptr) {
        HCCL_VM_ERROR(
            "{}: input pointer is nullptr, comm={:p}, rankNum={:p}", __func__, comm, static_cast<void*>(rankNum));
        return HCCL_E_PTR;
    }
    // 步骤2：取得当前rank。
    uint64_t commId = 0;
    uint32_t rankId = 0;
    HcclResult result = GetCurrentRank(comm, commId, rankId);
    if (result != HCCL_SUCCESS) {
        return result;
    }
    // 步骤3：向单例查询网络实例中的rank数量。
    sim::RankTableQueryStatus status = sim::RankTable::Instance().GetRankSizeByLayer(rankId, netLayer, *rankNum);
    if (status == sim::RankTableQueryStatus::LAYER_NOT_FOUND) {
        HCCL_VM_ERROR("{}: network layer {:d} was not found for rank {:d}", __func__, netLayer, rankId);
        return HCCL_E_PARA;
    }
    if (status != sim::RankTableQueryStatus::SUCCESS) {
        HCCL_VM_ERROR(
            "{}: failed to query rank size, rankId={:d}, netLayer={:d}, status={:d}", __func__, rankId, netLayer,
            static_cast<int>(status));
        return HCCL_E_INTERNAL;
    }
    // 步骤4：记录查询结果。
    HCCL_VM_INFO(
        "{} success, commId={:d}, rankId={:d}, "
        "netLayer={:d}, rankNum={:d}",
        __func__, commId, rankId, netLayer, *rankNum);
    return HCCL_SUCCESS;
}

HcclResult
HcclRankGraphGetInstSizeListByLayer(HcclComm comm, uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize)
{
    // 步骤1：检查参数。
    if (comm == nullptr || instSizeList == nullptr || listSize == nullptr) {
        HCCL_VM_ERROR(
            "{}: input pointer is nullptr, comm={:p}, instSizeList={:p}, listSize={:p}", __func__, comm,
            static_cast<void*>(instSizeList), static_cast<void*>(listSize));
        return HCCL_E_PTR;
    }
    // 步骤2：取得当前rank。
    uint64_t commId = 0;
    uint32_t rankId = 0;
    HcclResult result = GetCurrentRank(comm, commId, rankId);
    if (result != HCCL_SUCCESS) {
        return result;
    }
    // 步骤3：向单例查询各网络实例的大小。
    static std::vector<uint32_t> resultSizes;
    sim::RankTableQueryStatus status = sim::RankTable::Instance().GetInstSizeListByLayer(rankId, netLayer, resultSizes);
    if (status == sim::RankTableQueryStatus::LAYER_NOT_FOUND) {
        HCCL_VM_ERROR("{}: network layer {:d} was not found for rank {:d}", __func__, netLayer, rankId);
        return HCCL_E_PARA;
    }
    if (status != sim::RankTableQueryStatus::SUCCESS) {
        HCCL_VM_ERROR(
            "{}: failed to query instance sizes, rankId={:d}, netLayer={:d}, status={:d}", __func__, rankId, netLayer,
            static_cast<int>(status));
        return HCCL_E_INTERNAL;
    }
    // 步骤4：返回函数内静态数组。
    *instSizeList = resultSizes.data();
    *listSize = static_cast<uint32_t>(resultSizes.size());
    HCCL_VM_INFO(
        "{} success, commId={:d}, rankId={:d}, "
        "netLayer={:d}, listSize={:d}",
        __func__, commId, rankId, netLayer, *listSize);
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetTopoTypeByLayer(HcclComm comm, uint32_t netLayer, CommTopo* topoType)
{
    // 步骤1：检查参数。
    if (comm == nullptr || topoType == nullptr) {
        HCCL_VM_ERROR(
            "{}: input pointer is nullptr, comm={:p}, topoType={:p}", __func__, comm, static_cast<void*>(topoType));
        return HCCL_E_PTR;
    }
    // 步骤2：取得当前rank。
    uint64_t commId = 0;
    uint32_t rankId = 0;
    HcclResult result = GetCurrentRank(comm, commId, rankId);
    if (result != HCCL_SUCCESS) {
        return result;
    }
    // 步骤3：向单例查询已经转换好的HComm拓扑类型。
    sim::RankTableQueryStatus status = sim::RankTable::Instance().GetCommTopoTypeByLayer(rankId, netLayer, *topoType);
    if (status == sim::RankTableQueryStatus::LAYER_NOT_FOUND) {
        HCCL_VM_ERROR("{}: network layer {:d} was not found for rank {:d}", __func__, netLayer, rankId);
        return HCCL_E_PARA;
    }
    if (status == sim::RankTableQueryStatus::UNSUPPORTED_TYPE) {
        HCCL_VM_ERROR("{}: unsupported topology type, rankId={:d}, netLayer={:d}", __func__, rankId, netLayer);
        return HCCL_E_PARA;
    }
    if (status != sim::RankTableQueryStatus::SUCCESS) {
        HCCL_VM_ERROR(
            "{}: failed to query topology type, rankId={:d}, netLayer={:d}, status={:d}", __func__, rankId, netLayer,
            static_cast<int>(status));
        return HCCL_E_INTERNAL;
    }
    // 步骤4：单例已经完成格式转换，桩函数只记录并返回结果。
    HCCL_VM_INFO(
        "{} success, commId={:d}, rankId={:d}, "
        "netLayer={:d}, topoType={:d}",
        __func__, commId, rankId, netLayer, static_cast<int>(*topoType));
    return HCCL_SUCCESS;
}

HcclResult
HcclRankGraphGetTopoInstsByLayer(HcclComm comm, uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum)
{
    // 步骤1：检查参数。
    if (comm == nullptr || topoInsts == nullptr || topoInstNum == nullptr) {
        HCCL_VM_ERROR(
            "{}: input pointer is nullptr, comm={:p}, topoInsts={:p}, topoInstNum={:p}", __func__, comm,
            static_cast<void*>(topoInsts), static_cast<void*>(topoInstNum));
        return HCCL_E_PTR;
    }
    // 步骤2：取得当前rank。
    uint64_t commId = 0;
    uint32_t rankId = 0;
    HcclResult result = GetCurrentRank(comm, commId, rankId);
    if (result != HCCL_SUCCESS) {
        return result;
    }
    // 步骤3：向单例查询拓扑实例ID。
    static std::vector<uint32_t> resultInstances;
    sim::RankTableQueryStatus status
        = sim::RankTable::Instance().GetTopoInstsByLayer(rankId, netLayer, resultInstances);
    if (status == sim::RankTableQueryStatus::LAYER_NOT_FOUND || status == sim::RankTableQueryStatus::UNSUPPORTED_TYPE) {
        HCCL_VM_ERROR(
            "{}: invalid topology layer, rankId={:d}, netLayer={:d}, status={:d}", __func__, rankId, netLayer,
            static_cast<int>(status));
        return HCCL_E_PARA;
    }
    if (status != sim::RankTableQueryStatus::SUCCESS) {
        HCCL_VM_ERROR(
            "{}: failed to query topology instances, rankId={:d}, netLayer={:d}, status={:d}", __func__, rankId,
            netLayer, static_cast<int>(status));
        return HCCL_E_INTERNAL;
    }
    // 步骤4：返回函数内静态数组。
    *topoInsts = resultInstances.data();
    *topoInstNum = static_cast<uint32_t>(resultInstances.size());
    HCCL_VM_INFO(
        "{} success, commId={:d}, rankId={:d}, "
        "netLayer={:d}, topoInstNum={:d}",
        __func__, commId, rankId, netLayer, *topoInstNum);
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetTopoType(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType)
{
    // 步骤1：检查参数。
    if (comm == nullptr || topoType == nullptr) {
        HCCL_VM_ERROR(
            "{}: input pointer is nullptr, comm={:p}, topoType={:p}", __func__, comm, static_cast<void*>(topoType));
        return HCCL_E_PTR;
    }
    // 步骤2：取得当前rank。
    uint64_t commId = 0;
    uint32_t rankId = 0;
    HcclResult result = GetCurrentRank(comm, commId, rankId);
    if (result != HCCL_SUCCESS) {
        return result;
    }
    // 步骤3：向单例查询已经转换好的HComm拓扑类型。
    sim::RankTableQueryStatus status
        = sim::RankTable::Instance().GetCommTopoType(rankId, netLayer, topoInstId, *topoType);
    if (status == sim::RankTableQueryStatus::LAYER_NOT_FOUND
        || status == sim::RankTableQueryStatus::TOPO_INSTANCE_NOT_FOUND
        || status == sim::RankTableQueryStatus::UNSUPPORTED_TYPE) {
        HCCL_VM_ERROR(
            "{}: invalid topology instance, rankId={:d}, netLayer={:d}, "
            "topoInstId={:d}, status={:d}",
            __func__, rankId, netLayer, topoInstId, static_cast<int>(status));
        return HCCL_E_PARA;
    }
    if (status != sim::RankTableQueryStatus::SUCCESS) {
        HCCL_VM_ERROR(
            "{}: failed to query topology type, rankId={:d}, netLayer={:d}, "
            "topoInstId={:d}, status={:d}",
            __func__, rankId, netLayer, topoInstId, static_cast<int>(status));
        return HCCL_E_INTERNAL;
    }
    // 步骤4：单例已经完成格式转换，桩函数只记录并返回结果。
    HCCL_VM_INFO(
        "{} success, commId={:d}, rankId={:d}, "
        "netLayer={:d}, topoInstId={:d}, topoType={:d}",
        __func__, commId, rankId, netLayer, topoInstId, static_cast<int>(*topoType));
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetLinks(
    HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** links, uint32_t* linkNum)
{
    // 步骤1：检查参数。
    if (comm == nullptr || links == nullptr || linkNum == nullptr) {
        HCCL_VM_ERROR(
            "{}: input pointer is nullptr, comm={:p}, links={:p}, linkNum={:p}", __func__, comm,
            static_cast<void*>(links), static_cast<void*>(linkNum));
        return HCCL_E_PTR;
    }
    // 步骤2：取得当前rank并验证通信域。
    uint64_t commId = 0;
    uint32_t rankId = 0;
    HcclResult result = GetCurrentRank(comm, commId, rankId);
    if (result != HCCL_SUCCESS) {
        return result;
    }
    // 步骤3：向单例查询已经转换好的HComm链路数据。
    static std::vector<CommLink> resultLinks;
    sim::RankTableQueryStatus status
        = sim::RankTable::Instance().GetCommLinks(rankId, netLayer, srcRank, dstRank, resultLinks);
    if (status == sim::RankTableQueryStatus::LAYER_NOT_FOUND) {
        HCCL_VM_ERROR("{}: network layer {:d} was not found for rank {:d}", __func__, netLayer, rankId);
        return HCCL_E_PARA;
    }
    if (status != sim::RankTableQueryStatus::SUCCESS) {
        HCCL_VM_ERROR(
            "{}: failed to query links, netLayer={:d}, srcRank={:d}, "
            "dstRank={:d}, status={:d}",
            __func__, netLayer, srcRank, dstRank, static_cast<int>(status));
        return HCCL_E_INTERNAL;
    }
    // 步骤4：返回函数内静态链路数组；转换细节由单例隐藏。
    *links = resultLinks.data();
    *linkNum = static_cast<uint32_t>(resultLinks.size());
    HCCL_VM_INFO(
        "{} success, commId={:d}, rankId={:d}, netLayer={:d}, "
        "srcRank={:d}, dstRank={:d}, linkNum={:d}",
        __func__, commId, rankId, netLayer, srcRank, dstRank, *linkNum);
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetEndpointNum(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* num)
{
    // 步骤1：检查参数。
    if (comm == nullptr || num == nullptr) {
        HCCL_VM_ERROR("{}: input pointer is nullptr, comm={:p}, num={:p}", __func__, comm, static_cast<void*>(num));
        return HCCL_E_PTR;
    }
    // 步骤2：取得当前rank。
    uint64_t commId = 0;
    uint32_t rankId = 0;
    HcclResult result = GetCurrentRank(comm, commId, rankId);
    if (result != HCCL_SUCCESS) {
        return result;
    }
    // 步骤3：向单例查询端点数量。
    sim::RankTableQueryStatus status = sim::RankTable::Instance().GetEndpointNum(rankId, layer, topoInstId, *num);
    if (status == sim::RankTableQueryStatus::LAYER_NOT_FOUND) {
        HCCL_VM_ERROR("{}: network layer {:d} was not found for rank {:d}", __func__, layer, rankId);
        return HCCL_E_PARA;
    }
    if (status != sim::RankTableQueryStatus::SUCCESS) {
        HCCL_VM_ERROR(
            "{}: failed to query endpoint number, rankId={:d}, layer={:d}, "
            "topoInstId={:d}, status={:d}",
            __func__, rankId, layer, topoInstId, static_cast<int>(status));
        return HCCL_E_INTERNAL;
    }
    // 步骤4：记录最终数量。
    HCCL_VM_INFO(
        "{} success, commId={:d}, rankId={:d}, "
        "netLayer={:d}, topoInstId={:d}, endpointNum={:d}",
        __func__, commId, rankId, layer, topoInstId, *num);
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetRanksByTopoInst(
    HcclComm comm, uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum)
{
    // 步骤1：检查参数。
    if (comm == nullptr || ranks == nullptr || rankNum == nullptr) {
        HCCL_VM_ERROR(
            "{}: input pointer is nullptr, comm={:p}, ranks={:p}, rankNum={:p}", __func__, comm,
            static_cast<void*>(ranks), static_cast<void*>(rankNum));
        return HCCL_E_PTR;
    }
    // 步骤2：取得当前rank。
    uint64_t commId = 0;
    uint32_t rankId = 0;
    HcclResult result = GetCurrentRank(comm, commId, rankId);
    if (result != HCCL_SUCCESS) {
        return result;
    }
    // 步骤3：向单例查询拓扑实例包含的rank。
    static std::vector<uint32_t> resultRanks;
    sim::RankTableQueryStatus status
        = sim::RankTable::Instance().GetRanksByTopoInst(rankId, netLayer, topoInstId, resultRanks);
    if (status == sim::RankTableQueryStatus::LAYER_NOT_FOUND
        || status == sim::RankTableQueryStatus::TOPO_INSTANCE_NOT_FOUND) {
        HCCL_VM_ERROR(
            "{}: invalid topology instance, rankId={:d}, netLayer={:d}, "
            "topoInstId={:d}, status={:d}",
            __func__, rankId, netLayer, topoInstId, static_cast<int>(status));
        return HCCL_E_PARA;
    }
    if (status != sim::RankTableQueryStatus::SUCCESS) {
        HCCL_VM_ERROR(
            "{}: failed to query ranks by topology instance, rankId={:d}, "
            "netLayer={:d}, topoInstId={:d}, status={:d}",
            __func__, rankId, netLayer, topoInstId, static_cast<int>(status));
        return HCCL_E_INTERNAL;
    }
    // 步骤4：返回函数内静态数组。
    *ranks = resultRanks.data();
    *rankNum = static_cast<uint32_t>(resultRanks.size());
    HCCL_VM_INFO(
        "{} success, commId={:d}, rankId={:d}, "
        "netLayer={:d}, topoInstId={:d}, rankNum={:d}",
        __func__, commId, rankId, netLayer, topoInstId, *rankNum);
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetEndpointDesc(
    HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc)
{
    // 步骤1：检查参数；descNum同时表示调用者提供的容量。
    if (comm == nullptr || descNum == nullptr || endpointDesc == nullptr) {
        HCCL_VM_ERROR(
            "{}: input pointer is nullptr, comm={:p}, descNum={:p}, endpointDesc={:p}", __func__, comm,
            static_cast<void*>(descNum), static_cast<void*>(endpointDesc));
        return HCCL_E_PTR;
    }
    // 步骤2：取得当前rank。
    uint64_t commId = 0;
    uint32_t rankId = 0;
    HcclResult result = GetCurrentRank(comm, commId, rankId);
    if (result != HCCL_SUCCESS) {
        return result;
    }
    // 步骤3：向单例查询已经转换好的HComm端点列表。
    std::vector<EndpointDesc> endpointDescList;
    sim::RankTableQueryStatus status
        = sim::RankTable::Instance().GetCommEndpointDescList(rankId, layer, topoInstId, endpointDescList);
    if (status == sim::RankTableQueryStatus::LAYER_NOT_FOUND) {
        HCCL_VM_ERROR("{}: network layer {:d} was not found for rank {:d}", __func__, layer, rankId);
        return HCCL_E_PARA;
    }
    if (status != sim::RankTableQueryStatus::SUCCESS) {
        HCCL_VM_ERROR(
            "{}: failed to query endpoints, rankId={:d}, layer={:d}, "
            "topoInstId={:d}, status={:d}",
            __func__, rankId, layer, topoInstId, static_cast<int>(status));
        return HCCL_E_INTERNAL;
    }
    // 步骤4：检查调用者数组容量。
    if (endpointDescList.size() > *descNum) {
        HCCL_VM_ERROR(
            "{}: endpointDesc array is too small, required={:d}, provided={:d}", __func__, endpointDescList.size(),
            *descNum);
        return HCCL_E_PARA;
    }
    // 步骤5：单例已经完成转换，桩函数只复制到调用者数组。
    for (size_t index = 0; index < endpointDescList.size(); ++index) {
        endpointDesc[index] = endpointDescList[index];
    }
    // 步骤6：返回实际写入数量。
    *descNum = static_cast<uint32_t>(endpointDescList.size());
    HCCL_VM_INFO(
        "{} success, commId={:d}, rankId={:d}, "
        "netLayer={:d}, topoInstId={:d}, descNum={:d}",
        __func__, commId, rankId, layer, topoInstId, *descNum);
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetEndpointInfo(
    HcclComm comm, uint32_t rankId, const EndpointDesc* endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen,
    void* info)
{
    // 步骤1：检查参数。
    if (comm == nullptr || endpointDesc == nullptr || info == nullptr) {
        HCCL_VM_ERROR(
            "{}: input pointer is nullptr, comm={:p}, endpointDesc={:p}, info={:p}", __func__, comm,
            static_cast<const void*>(endpointDesc), info);
        return HCCL_E_PTR;
    }
    // 步骤2：验证通信域存在；该接口使用入参rankId查询目标rank。
    uint64_t commId = 0;
    uint32_t currentRank = 0;
    HcclResult result = GetCurrentRank(comm, commId, currentRank);
    if (result != HCCL_SUCCESS) {
        return result;
    }
    (void)currentRank;
    // 步骤3：把HComm端点和目标属性交给单例；单例完成转换、查询和属性填写。
    sim::RankTableQueryStatus status
        = sim::RankTable::Instance().GetCommEndpointInfo(rankId, *endpointDesc, endpointAttr, infoLen, info);
    if (status == sim::RankTableQueryStatus::RANK_NOT_FOUND) {
        HCCL_VM_ERROR("{}: rank {:d} was not found", __func__, rankId);
        return HCCL_E_PTR;
    }
    if (status == sim::RankTableQueryStatus::ENDPOINT_NOT_FOUND) {
        HCCL_VM_ERROR("{}: endpoint was not found for rank {:d}", __func__, rankId);
        return HCCL_E_NOT_FOUND;
    }
    if (status == sim::RankTableQueryStatus::INVALID_PARAMETER) {
        HCCL_VM_ERROR(
            "{}: invalid endpoint address, protocol, attribute or info length, "
            "rankId={:d}, endpointAttr={:d}, infoLen={:d}",
            __func__, rankId, static_cast<int>(endpointAttr), infoLen);
        return HCCL_E_PARA;
    }
    if (status != sim::RankTableQueryStatus::SUCCESS) {
        HCCL_VM_ERROR(
            "{}: failed to query endpoint info, rankId={:d}, status={:d}", __func__, rankId, static_cast<int>(status));
        return HCCL_E_INTERNAL;
    }
    // 步骤4：属性已经由单例写入info，桩函数记录调用结果。
    HCCL_VM_INFO(
        "{} success, commId={:d}, rankId={:d}, "
        "endpointAttr={:d}",
        __func__, commId, rankId, static_cast<int>(endpointAttr));
    return HCCL_SUCCESS;
}

HcclResult HcclGetHeterogMode(HcclComm comm, HcclHeterogMode* mode)
{
    // 北向劫持场景为单一芯片类型（Ascend950）的同构组网，返回 HOMOGENEOUS，
    // 放行 Scatter 等算子的单芯片类型校验；A5 无混合组网。
    if (comm == nullptr || mode == nullptr) {
        HCCL_VM_ERROR("{}: comm or mode is nullptr", __func__);
        return HCCL_E_PTR;
    }
    *mode = HcclHeterogMode::HCCL_HETEROG_MODE_HOMOGENEOUS;
    uint64_t commId = reinterpret_cast<uint64_t>(comm);
    HCCL_VM_INFO("{} success, commId={:d}, mode=HOMOGENEOUS", __func__, commId);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif
