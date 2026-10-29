/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_info.h"
#include "json_parser.h"
#include "invalid_params_exception.h"
#include "exception_util.h"
#include "adapter_error_manager_pub.h"
#include "common/hccl_log_keywords_pub.h"
#include "rank_table_report_macro.h"

namespace Hccl {

using namespace std;

HcclResult TopoInfo::Deserialize(const nlohmann::json& topoInfoJson)
{
    std::string msgVersion = "[TopoInfo::Deserialize] error occurs when parser object of propName \"version\"";
    std::string msgPc = "[TopoInfo::Deserialize] error occurs when parser object of propName \"peer_count\"";
    std::string msgEc = "[TopoInfo::Deserialize] error occurs when parser object of propName \"edge_count\"";
    TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT_TOPO(
        InvalidParamsException, msgVersion, (version = GetJsonProperty(topoInfoJson, "version")), topoInfoJson,
        "version", "non-empty string"));
    TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT_TOPO(
        InvalidParamsException, msgPc, (peerCount = GetJsonPropertyUInt(topoInfoJson, "peer_count")), topoInfoJson,
        "peer_count", "0 ~ UINT32_MAX"));
    TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT_TOPO(
        InvalidParamsException, msgEc, (edgeCount = GetJsonPropertyUInt(topoInfoJson, "edge_count")), topoInfoJson,
        "edge_count", "0 ~ UINT32_MAX"));

    if (version != "2.0") {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({version, "version", "2.0"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] version[%s] is not \"2.0\"", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), version.c_str());
        return HCCL_E_PARA;
    }

    if (peerCount == 0 || peerCount > MAX_PEER_COUNT) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(peerCount), "peer_count", "[1, " + std::to_string(MAX_PEER_COUNT) + "]"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] peer_count[%u] is out of range [1, %u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), peerCount, MAX_PEER_COUNT);
        return HCCL_E_PARA;
    }

    if (edgeCount == 0) {
        HCCL_WARNING("[TopoInfo::%s]: edge_count is zero", __func__);
    }

    CHK_RET(DeserializePeers(topoInfoJson));
    CHK_RET(DeserializeEdges(topoInfoJson));
    return HCCL_SUCCESS;
}

HcclResult TopoInfo::DeserializePeers(const nlohmann::json& topoInfoJson)
{
    nlohmann::json peerJsons;
    HcclResult ret = GetJsonPropertyList(topoInfoJson, "peer_list", peerJsons);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(topoInfoJson, "peer_list", "array", RankTableSource::RANKTABLE);
        return ret;
    }
    for (auto& peerJson : peerJsons) {
        PeerInfo peer;
        CHK_RET(peer.Deserialize(peerJson));

        if (idSet.count(peer.localId) > 0) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({std::to_string(peer.localId), "local_id", "unique local_id"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] duplicate localId[%u] in peers, idSet.count[%zu]",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), peer.localId, idSet.count(peer.localId));
            return HCCL_E_PARA;
        }

        peers.emplace_back(peer);
        idSet.insert(peer.localId);
    }

    if (peerCount != peers.size()) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({std::to_string(peerCount), "peer_count", std::to_string(peers.size())}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] peer_count[%u] is inconsistent with peer_list size[%zu]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), peerCount, peers.size());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfo::VerifyEdges(EdgeInfo& edge)
{
    bool localAExist = idSet.count(edge.localA) > 0;
    bool localBExist = idSet.count(edge.localB) > 0;
    if (!localAExist || !localBExist) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(edge.localA) + "/" + std::to_string(edge.localB), "local_a/local_b",
                 "existing local_id in peers"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] endpoint localId[%u] or [%u] is not exist in peers[%zu], "
            "idSet.count(localA)[%zu], idSet.count(localB)[%zu]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), edge.localA, edge.localB, idSet.size(), idSet.count(edge.localA),
            idSet.count(edge.localB));
        return HCCL_E_PARA;
    }
    //  检查edge.netLayer这一层级是否存在，不存在则初始化
    if (edges.find(edge.netLayer) == edges.end()) {
        edges[edge.netLayer] = vector<EdgeInfo>();
    }
    //  判断edge.netLayer该层级是否存在重复edge
    if (find(edges[edge.netLayer].begin(), edges[edge.netLayer].end(), edge) != edges[edge.netLayer].end()) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(edge.localA) + "-" + std::to_string(edge.localB), "edge", "unique edge"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] duplicate edges: netLayer[%u], linkType[%s], topoType[%s], topoInstId[%u], "
            "localA[%u], localB[%u], existingEdgeCount[%zu]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), edge.netLayer, edge.linkType.Describe().c_str(),
            edge.topoType.Describe().c_str(), edge.topoInstId, edge.localA, edge.localB, edges[edge.netLayer].size());
        return HCCL_E_PARA;
    }

    edges[edge.netLayer].emplace_back(edge);
    return HCCL_SUCCESS;
}

HcclResult TopoInfo::DeserializeEdges(const nlohmann::json& topoInfoJson)
{
    nlohmann::json edgeJsons;
    HcclResult ret = GetJsonPropertyList(topoInfoJson, "edge_list", edgeJsons);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(topoInfoJson, "edge_list", "array", RankTableSource::RANKTABLE);
        return ret;
    }
    if (edgeJsons.empty()) {
        if (edgeCount != 0) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({std::to_string(edgeCount), "edge_count", "0 or matching edge_list size"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] edge_count[%u] is inconsistent with empty edge_list",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), edgeCount);
            return HCCL_E_PARA;
        } else {
            HCCL_WARNING("[TopoInfo::%s] edge count is zero", __func__);
            return HCCL_SUCCESS;
        }
    }
    for (auto& edgeJson : edgeJsons) {
        EdgeInfo edge;
        CHK_RET(edge.Deserialize(edgeJson));
        CHK_RET(VerifyEdges(edge));
    }

    size_t sumEdge = 0;
    for (const auto& entry : edges) {
        sumEdge += entry.second.size();
    }
    if (sumEdge != edgeCount) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({std::to_string(edgeCount), "edge_count", std::to_string(sumEdge)}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] edge_count[%u] is inconsistent with edge_list size[%zu]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), edgeCount, sumEdge);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

string TopoInfo::Describe() const
{
    string description = "TopoInfo{";
    description += StringFormat("version=%s", version.c_str());
    description += StringFormat(", peer_count=%u", peerCount);
    description += StringFormat(", edge_count=%u", edgeCount);
    description += StringFormat(", Peers.size=%u", peers.size());
    description += StringFormat(", Edges.size=%u", edges.size());
    description += "}";
    return description;
}

void TopoInfo::Dump() const
{
    HCCL_DEBUG("TopoInfo Dump:");
    HCCL_DEBUG("%s", Describe().c_str());
    HCCL_DEBUG("peers:");
    for (const auto& peer : peers) {
        HCCL_DEBUG("%s", peer.Describe().c_str());
    }
    HCCL_DEBUG("edges:");
    for (const auto& itor : edges) {
        HCCL_DEBUG("netLayer[%u]:", itor.first);
        for (const auto& edge : itor.second) {
            HCCL_DEBUG("    %s", edge.Describe().c_str());
        }
    }
}

TopoInfo::TopoInfo(BinaryStream& binaryStream)
{
    binaryStream >> version >> peerCount >> edgeCount;
    size_t peersSize = 0;
    binaryStream >> peersSize;
    for (u32 i = 0; i < peersSize; i++) {
        PeerInfo peer(binaryStream);
        peers.emplace_back(peer);
    }
    size_t edgesSize = 0;
    binaryStream >> edgesSize;

    HCCL_INFO(
        "[TopoInfo] version is [%s], peerCount is [%u], edgeCount is [%u], peers size is [%u], edges size is [%u]",
        version.c_str(), peerCount, edgeCount, peers.size(), edgesSize);
    for (u32 i = 0; i < edgesSize; i++) {
        u32 edgeInfoIndex = 0;
        binaryStream >> edgeInfoIndex; // key
        size_t edgeSize = 0;
        binaryStream >> edgeSize;
        HCCL_INFO("[TopoInfo] edges key is [%u], value size is [%u]", edgeInfoIndex, edgeSize);
        for (u32 j = 0; j < edgeSize; j++) { // value
            EdgeInfo edge(binaryStream);
            edges[edgeInfoIndex].emplace_back(edge);
        }
    }
}

void TopoInfo::GetBinStream(BinaryStream& binaryStream) const
{
    binaryStream << version << peerCount << edgeCount;
    binaryStream << peers.size();
    for (auto& it : peers) {
        it.GetBinStream(binaryStream);
    }
    binaryStream << edges.size();
    HCCL_INFO(
        "[TopoInfo::GetBinStream] version is [%s], peerCount is [%u], edgeCount is [%u], peers size is [%u], edges "
        "size is [%u]",
        version.c_str(), peerCount, edgeCount, peers.size(), edges.size());
    for (auto& it : edges) {
        binaryStream << it.first;
        binaryStream << it.second.size();
        HCCL_INFO("[TopoInfo::GetBinStream] edges key is [%u], value size is [%u]", it.first, it.second.size());
        for (auto& edge : it.second) {
            edge.GetBinStream(binaryStream);
        }
    }
}

} // namespace Hccl
