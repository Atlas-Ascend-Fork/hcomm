/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TOPO_INFO_READER_H
#define HCCLV2_TOPO_INFO_READER_H

#include <set>
#include <string>
#include "json/common/json_parse_util.h"
#include "topo_info.h"
#include "peer_info.h"
#include "edge_info.h"

namespace Hccl {

// topo 域的 DOM Reader。peer/edge 的解析作为本类成员，不再单独暴露 Reader 类。
class TopoInfoReader {
public:
    HcclResult Read(const std::string& json, TopoInfo& out) const;
    HcclResult ReadJson(const nlohmann::json& json, TopoInfo& out) const;

    // 子结构入口：供本类内部复用，也供单测直接验证。
    HcclResult ReadPeer(const nlohmann::json& json, PeerInfo& out) const;
    HcclResult ReadEdge(const nlohmann::json& json, EdgeInfo& out) const;

private:
    HcclResult ReadProtocol(const nlohmann::json& edgeInfoJson, EdgeInfo& out) const;
    HcclResult ReadEndpoint(const nlohmann::json& edgeInfoJson, EdgeInfo& out) const;
    HcclResult
    ReadPort(const nlohmann::json& edgeInfoJson, const std::string& propName, std::set<std::string>& ports) const;
};

template <>
inline HcclResult ReadJson<TopoInfo>(const nlohmann::json& json, TopoInfo& info)
{
    TopoInfoReader reader;
    HcclResult ret = reader.ReadJson(json, info);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    return info.Check();
}

template <>
inline HcclResult ReadJson<PeerInfo>(const nlohmann::json& json, PeerInfo& info)
{
    TopoInfoReader reader;
    return reader.ReadPeer(json, info);
}

template <>
inline HcclResult ReadJson<EdgeInfo>(const nlohmann::json& json, EdgeInfo& info)
{
    TopoInfoReader reader;
    return reader.ReadEdge(json, info);
}

} // namespace Hccl

#endif // HCCLV2_TOPO_INFO_READER_H
