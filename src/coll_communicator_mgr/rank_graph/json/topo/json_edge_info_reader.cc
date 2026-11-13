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
#include <vector>
#include "adapter_error_manager_pub.h"
#include "common/hccl_log_keywords_pub.h"
#include "exception_util.h"
#include "invalid_params_exception.h"
#include "rank_table_report_macro.h"

namespace Hccl {

HcclResult TopoInfoReader::ReadEdge(const nlohmann::json& edgeInfoJson, EdgeInfo& out) const
{
    out = EdgeInfo{};
    out.localA = 0;
    out.localB = 0;
    out.localAPorts.clear();
    out.localBPorts.clear();

    HcclResult ret = GetJsonPropertyUInt(edgeInfoJson, "net_layer", true, 0, out.netLayer);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(edgeInfoJson, "net_layer", "0 ~ UINT32_MAX", RankTableSource::RANKTABLE);
        return ret;
    }
    if (out.netLayer > MAX_VALUE_LEVEL) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(out.netLayer), "net_layer", "[0, " + std::to_string(MAX_VALUE_LEVEL) + "]"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] net_layer[%u] is out of range [0, %u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), out.netLayer,
            MAX_VALUE_LEVEL);
        return HCCL_E_PARA;
    }

    CHK_RET(ReadProtocol(edgeInfoJson, out));

    if (edgeInfoJson.contains("topo_type")) {
        std::string topoTypeStr;
        ret = GetJsonProperty(edgeInfoJson, "topo_type", true, topoTypeStr);
        if (ret != HCCL_SUCCESS) {
            ReportJsonConfigError(edgeInfoJson, "topo_type", "non-empty string", RankTableSource::RANKTABLE);
            return ret;
        }
        TRY_CATCH_RETURN(out.topoType = out.GetTopoType(topoTypeStr););
    } else {
        HCCL_WARNING("[TopoInfoReader::%s] topo_type not found, [default]topo_type=TopoType::CLOS", __func__);
        out.topoType = TopoType::CLOS;
    }

    if (edgeInfoJson.contains("topo_instance_id")) {
        ret = GetJsonPropertyUInt(edgeInfoJson, "topo_instance_id", true, 0, out.topoInstId);
        if (ret != HCCL_SUCCESS) {
            ReportJsonConfigError(edgeInfoJson, "topo_instance_id", "0 ~ UINT32_MAX", RankTableSource::RANKTABLE);
            return ret;
        }
    } else {
        HCCL_WARNING("[TopoInfoReader::%s] topo_instance_id not found, [default]topo_instance_id=0", __func__);
        out.topoInstId = 0;
    }

    CHK_RET(ReadEndpoint(edgeInfoJson, out));
    return HCCL_SUCCESS;
}

HcclResult TopoInfoReader::ReadProtocol(const nlohmann::json& edgeInfoJson, EdgeInfo& out) const
{
    nlohmann::json jsonProtocols;
    HcclResult ret = GetJsonPropertyList(edgeInfoJson, "protocols", jsonProtocols);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(edgeInfoJson, "protocols", "array", RankTableSource::RANKTABLE);
        return ret;
    }
    for (auto& protocolEle : jsonProtocols) {
        auto protocolStr = protocolEle.get<std::string>();
        LinkProtocol protocol;
        TRY_CATCH_RETURN(protocol = out.GetLinkProtocol(protocolStr););
        if (out.protocols.count(protocol) == 0) {
            out.protocols.emplace(protocol);
        } else {
            HCCL_WARNING("[TopoInfoReader::%s] repeat member[%s] in \"protocols\"", __func__, protocolStr.c_str());
        }
    }

    if (out.protocols.empty()) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({"empty", "protocols", "non-empty array"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] protocols is empty", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoReader::ReadEndpoint(const nlohmann::json& edgeInfoJson, EdgeInfo& out) const
{
    std::string linkTypeStr;
    HcclResult ret = GetJsonProperty(edgeInfoJson, "link_type", true, linkTypeStr);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(edgeInfoJson, "link_type", "non-empty string", RankTableSource::RANKTABLE);
        return ret;
    }
    TRY_CATCH_RETURN(out.linkType = out.GetLinkType(linkTypeStr););

    ret = GetJsonPropertyUInt(edgeInfoJson, "local_a", true, 0, out.localA);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(edgeInfoJson, "local_a", "0 ~ UINT32_MAX", RankTableSource::RANKTABLE);
        return ret;
    }

    CHK_RET(ReadPort(edgeInfoJson, "local_a_ports", out.localAPorts));
    if (out.localAPorts.empty()) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({"empty", "local_a_ports", "non-empty array"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] local_a_ports is empty", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    if (out.linkType == LinkType::PEER2PEER) {
        ret = GetJsonPropertyUInt(edgeInfoJson, "local_b", true, 0, out.localB);
        if (ret != HCCL_SUCCESS) {
            ReportJsonConfigError(edgeInfoJson, "local_b", "0 ~ UINT32_MAX", RankTableSource::RANKTABLE);
            return ret;
        }

        if (out.localA == out.localB) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({std::to_string(out.localA), "local_a/local_b", "different local_id"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] local_a[%u] and local_b[%u] can not be the same",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), out.localA, out.localB);
            return HCCL_E_PARA;
        }

        CHK_RET(ReadPort(edgeInfoJson, "local_b_ports", out.localBPorts));
        if (out.localBPorts.empty()) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({"empty", "local_b_ports", "non-empty array for PEER2PEER"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] local_b_ports is empty when PEER2PEER", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA));
            return HCCL_E_PARA;
        }
    } else {
        if (edgeInfoJson.contains("local_b") || edgeInfoJson.contains("local_b_ports")) {
            HCCL_WARNING("[TopoInfoReader::%s] local_b and local_b_ports are not needed when PEER2NET", __func__);
        }
    }

    if (edgeInfoJson.contains("position")) {
        std::string positionStr;
        ret = GetJsonProperty(edgeInfoJson, "position", true, positionStr);
        if (ret != HCCL_SUCCESS) {
            ReportJsonConfigError(edgeInfoJson, "position", "non-empty string", RankTableSource::RANKTABLE);
            return ret;
        }
        TRY_CATCH_RETURN(out.position = out.GetAddrPosition(positionStr););
    } else {
        HCCL_WARNING("[TopoInfoReader::%s] position not found, [default]position=DEVICE", __func__);
        out.position = AddrPosition::DEVICE;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoReader::ReadPort(
    const nlohmann::json& edgeInfoJson, const std::string& propName, std::set<std::string>& ports) const
{
    nlohmann::json jsonPorts;
    HcclResult ret = GetJsonPropertyList(edgeInfoJson, propName.c_str(), jsonPorts);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(edgeInfoJson, propName.c_str(), "array", RankTableSource::RANKTABLE);
        return ret;
    }
    if (jsonPorts.empty() || jsonPorts.size() > MAX_PORTS_SIZE) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(jsonPorts.size()), propName, "[1, " + std::to_string(MAX_PORTS_SIZE) + "]"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] ports[%s] size[%zu] out of range [1, %u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), propName.c_str(),
            jsonPorts.size(), MAX_PORTS_SIZE);
        return HCCL_E_PARA;
    }
    for (auto& portEle : jsonPorts) {
        std::string port = portEle.get<std::string>();
        if (!port.empty() && port.size() <= PORT_MAX_LENGTH) {
            if (ports.count(port) == 0) {
                ports.emplace(port);
            } else {
                HCCL_WARNING("[TopoInfoReader::%s] Repeat port:[%s]", __func__, port.c_str());
            }
        } else {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({port, "port", "length [1, " + std::to_string(PORT_MAX_LENGTH) + "]"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] invalid port[%s], length[%zu] out of range [1, %u]",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), port.c_str(), port.size(), PORT_MAX_LENGTH);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

} // namespace Hccl
