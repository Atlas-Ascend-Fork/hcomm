/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_info_reader.h"

#include <string>
#include "rank_table_report_macro.h"

namespace Hccl {

HcclResult TopoInfoReader::Read(const std::string& jsonString, TopoInfo& out) const
{
    if (jsonString.empty()) {
        HCCL_ERROR("[%s] json string length is zero", __func__);
        return HCCL_E_PARA;
    }
    try {
        const nlohmann::json json = nlohmann::json::parse(jsonString);
        return ReadJson(json, out);
    } catch (nlohmann::json::exception& e) {
        HCCL_ERROR("[%s] parse json failed: %s", __func__, e.what());
        return HCCL_E_PARA;
    }
}

HcclResult TopoInfoReader::ReadJson(const nlohmann::json& topoInfoJson, TopoInfo& out) const
{
    out = TopoInfo{};

    HcclResult ret = GetJsonProperty(topoInfoJson, "version", true, out.version);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(topoInfoJson, "version", "non-empty string", RankTableSource::RANKTABLE);
        return ret;
    }
    ret = GetJsonPropertyUInt(topoInfoJson, "peer_count", true, 0, out.peerCount);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(topoInfoJson, "peer_count", "0 ~ UINT32_MAX", RankTableSource::RANKTABLE);
        return ret;
    }
    ret = GetJsonPropertyUInt(topoInfoJson, "edge_count", true, 0, out.edgeCount);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(topoInfoJson, "edge_count", "0 ~ UINT32_MAX", RankTableSource::RANKTABLE);
        return ret;
    }

    nlohmann::json peerJsons;
    ret = GetJsonPropertyList(topoInfoJson, "peer_list", peerJsons);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(topoInfoJson, "peer_list", "array", RankTableSource::RANKTABLE);
        return ret;
    }
    for (auto& peerJson : peerJsons) {
        PeerInfo peer;
        CHK_RET(ReadPeer(peerJson, peer));
        out.peers.emplace_back(std::move(peer));
    }

    nlohmann::json edgeJsons;
    ret = GetJsonPropertyList(topoInfoJson, "edge_list", edgeJsons);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(topoInfoJson, "edge_list", "array", RankTableSource::RANKTABLE);
        return ret;
    }
    for (auto& edgeJson : edgeJsons) {
        EdgeInfo edge;
        CHK_RET(ReadEdge(edgeJson, edge));
        out.edges[edge.netLayer].emplace_back(std::move(edge));
    }
    return HCCL_SUCCESS;
}

} // namespace Hccl
