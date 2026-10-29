/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "edge_info.h"
#include "json_parser.h"
#include "invalid_params_exception.h"
#include "exception_util.h"
#include "adapter_error_manager_pub.h"
#include "common/hccl_log_keywords_pub.h"
#include "rank_table_report_macro.h"

namespace Hccl {

const unordered_map<string, LinkProtocol> EdgeInfo::strToLinkProtocol = (unordered_map<string, LinkProtocol>{
    {"UB_CTP", LinkProtocol::UB_CTP},
    {"UB_TP", LinkProtocol::UB_TP},
    {"ROCE", LinkProtocol::ROCE},
    {"HCCS", LinkProtocol::HCCS},
    {"PCIE", LinkProtocol::PCIE},
    {"TCP", LinkProtocol::TCP},
    {"UB_MEM", LinkProtocol::UB_MEM},
    {"UBOE", LinkProtocol::UBOE},
    {"UB_RTP", LinkProtocol::UB_RTP},
    {"UBG", LinkProtocol::UB_RTP}});

const unordered_map<std::string, TopoType> EdgeInfo::strToTopoType = (unordered_map<string, TopoType>{
    {"CLOS", TopoType::CLOS},
    {"1DMESH", TopoType::MESH_1D},
    {"2DMESH", TopoType::MESH_2D},
    {"A3_SERVER", TopoType::A3_SERVER},
    {"A2_AX_SERVER", TopoType::A2_AX_SERVER}});

const unordered_map<string, LinkType> EdgeInfo::strToLinkType
    = (unordered_map<string, LinkType>{{"PEER2PEER", LinkType::PEER2PEER}, {"PEER2NET", LinkType::PEER2NET}});

const unordered_map<string, AddrPosition> EdgeInfo::strToAddrPosition
    = (unordered_map<string, AddrPosition>{{"DEVICE", AddrPosition::DEVICE}, {"HOST", AddrPosition::HOST}});

HcclResult EdgeInfo::Deserialize(const nlohmann::json& edgeInfoJson)
{
    std::string msgNetLayer = "[EdgeInfo::Deserialize] error occurs when parser object of propName \"net_layer\"";
    TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT_TOPO(
        InvalidParamsException, msgNetLayer, (netLayer = GetJsonPropertyUInt(edgeInfoJson, "net_layer")), edgeInfoJson,
        "net_layer", "0 ~ UINT32_MAX"));
    if (netLayer > MAX_VALUE_LEVEL) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(netLayer), "net_layer", "[0, " + std::to_string(MAX_VALUE_LEVEL) + "]"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] net_layer[%u] is out of range [0, %u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), netLayer, MAX_VALUE_LEVEL);
        return HCCL_E_PARA;
    }

    CHK_RET(DeserializeProtocol(edgeInfoJson));

    if (edgeInfoJson.contains("topo_type")) {
        std::string topoTypeStr;
        std::string msgtopoType = "[EdgeInfo::Deserialize] error occurs when parser object of propName \"topo_type\"";
        TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT_TOPO(
            InvalidParamsException, msgtopoType, (topoTypeStr = GetJsonProperty(edgeInfoJson, "topo_type")),
            edgeInfoJson, "topo_type", "non-empty string"));
        TRY_CATCH_RETURN(topoType = GetTopoType(topoTypeStr););
    } else {
        HCCL_WARNING("[EdgeInfo::%s] topo_type not found, [default]topo_type=TopoType::CLOS", __func__);
        topoType = TopoType::CLOS; // topo_type字段不存在时，取默认值CLOS
    }

    std::string msgtopoInstIdType
        = "[EdgeInfo::Deserialize] error occurs when parser object of propName \"topo_instance_id\"";
    if (edgeInfoJson.contains("topo_instance_id")) {
        TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT_TOPO(
            InvalidParamsException, msgtopoInstIdType,
            (topoInstId = GetJsonPropertyUInt(edgeInfoJson, "topo_instance_id")), edgeInfoJson, "topo_instance_id",
            "0 ~ UINT32_MAX"));
    } else {
        HCCL_WARNING("[EdgeInfo::%s] topo_instance_id not found, [default]topo_instance_id=0", __func__);
        topoInstId = 0;
    }

    // 解析localA 和 localB
    CHK_RET(DeserializeEndpoint(edgeInfoJson));
    return HCCL_SUCCESS;
}

HcclResult EdgeInfo::DeserializeProtocol(const nlohmann::json& edgeInfoJson)
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
        TRY_CATCH_RETURN(protocol = GetLinkProtocol(protocolStr););
        if (protocols.count(protocol) == 0) {
            protocols.emplace(protocol);
        } else {
            HCCL_WARNING("[EdgeInfo::%s] repeat member[%s] in \"protocols\"", __func__, protocolStr.c_str());
        }
    }

    if (protocols.empty()) {
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

HcclResult EdgeInfo::DeserializeEndpoint(const nlohmann::json& edgeInfoJson)
{
    std::string linkTypeStr;
    std::string msglinkType
        = "[EdgeInfo::DeserializeEndpoint] error occurs when parser object of propName \"link_type\"";
    TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT_TOPO(
        InvalidParamsException, msglinkType, (linkTypeStr = GetJsonProperty(edgeInfoJson, "link_type")), edgeInfoJson,
        "link_type", "non-empty string"));
    TRY_CATCH_RETURN(linkType = GetLinkType(linkTypeStr););

    std::string msgLocalA = "[EdgeInfo::DeserializeEndpoint] error occurs when parser object of propName \"local_a\"";
    TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT_TOPO(
        InvalidParamsException, msgLocalA, (localA = GetJsonPropertyUInt(edgeInfoJson, "local_a")), edgeInfoJson,
        "local_a", "0 ~ UINT32_MAX"));

    CHK_RET(DeserializePort(edgeInfoJson, "local_a_ports", localAPorts));
    if (localAPorts.empty()) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({"empty", "local_a_ports", "non-empty array"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] local_a_ports is empty", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    if (linkType == LinkType::PEER2PEER) {
        std::string msgLocalB
            = "[EdgeInfo::DeserializeEndpoint] error occurs when parser object of propName \"local_b\"";
        TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT_TOPO(
            InvalidParamsException, msgLocalB, (localB = GetJsonPropertyUInt(edgeInfoJson, "local_b")), edgeInfoJson,
            "local_b", "0 ~ UINT32_MAX"));

        if (localA == localB) { // localA 和 localB 不能是同一个点
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({std::to_string(localA), "local_a/local_b", "different local_id"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] local_a[%u] and local_b[%u] can not be the same",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), localA, localB);
            return HCCL_E_PARA;
        }

        CHK_RET(DeserializePort(edgeInfoJson, "local_b_ports", localBPorts));
        if (localBPorts.empty()) {
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
            HCCL_WARNING("[EdgeInfo::%s] local_b and local_b_ports are not needed when PEER2NET", __func__);
        }
    }

    if (edgeInfoJson.contains("position")) {
        string positionStr;
        std::string msgPosition
            = "[EdgeInfo::DeserializeEndpoint] error occurs when parser object of propName \"position\"";
        TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT_TOPO(
            InvalidParamsException, msgPosition, (positionStr = GetJsonProperty(edgeInfoJson, "position")),
            edgeInfoJson, "position", "non-empty string"));
        TRY_CATCH_RETURN(position = GetAddrPosition(positionStr););
    } else {
        HCCL_WARNING("[EdgeInfo::%s] position not found, [default]position=DEVICE", __func__);
        position = AddrPosition::DEVICE;
    }
    return HCCL_SUCCESS;
}

HcclResult
EdgeInfo::DeserializePort(const nlohmann::json& edgeInfoJson, std::string propName, std::set<std::string>& ports) const
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
        string port = portEle.get<string>();
        if (!port.empty() && port.size() <= PORT_MAX_LENGTH) {
            if (ports.count(port) == 0) {
                ports.emplace(port);
            } else {
                HCCL_WARNING("[EdgeInfo::%s] Repeat port:[%s]", __func__, port.c_str());
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

bool EdgeInfo::operator==(const EdgeInfo& other) const
{
    return netLayer == other.netLayer && linkType == other.linkType && protocols == other.protocols
           && topoType == other.topoType && topoInstId == other.topoInstId && CompareEndpoints(other)
           && position == other.position;
}

// 比较EndpointA和B
bool EdgeInfo::CompareEndpoints(const EdgeInfo& other) const
{
    // 无论什么情况，A=other.A && B=other.B时，可视为相同的连接关系
    if (localA == other.localA && localB == other.localB && localAPorts == other.localAPorts
        && localBPorts == other.localBPorts) {
        return true;
    }

    // 当连接类型是PEER2PEER时，A=other.B && B=other.A时，可视为相同的连接关系
    if (linkType == other.linkType && linkType == LinkType::PEER2PEER && localA == other.localB
        && localAPorts == other.localBPorts && localB == other.localA && localBPorts == other.localAPorts) {
        return true;
    }

    return false;
}

LinkProtocol EdgeInfo::GetLinkProtocol(string str) const
{
    CHK_PRT_THROW(
        strToLinkProtocol.count(str) == 0,
        HCCL_ERROR("[EdgeInfo::%s] string['%s'] is not type of LinkProtocol.", __func__, str.c_str()),
        InvalidParamsException, "string is not type of LinkProtocol");
    return strToLinkProtocol.at(str);
}

TopoType EdgeInfo::GetTopoType(std::string topoTypeStr) const
{
    if (topoTypeStr.empty()) {
        return TopoType::CLOS; // 不填写时，取默认值
    }
    CHK_PRT_THROW(
        strToTopoType.count(topoTypeStr) == 0,
        HCCL_ERROR("[EdgeInfo::%s] string['%s'] is not type of TopoType.", __func__, topoTypeStr.c_str()),
        InvalidParamsException, "string is not type of TopoType");
    return strToTopoType.at(topoTypeStr);
}

LinkType EdgeInfo::GetLinkType(std::string linkTypeStr) const
{
    CHK_PRT_THROW(
        strToLinkType.count(linkTypeStr) == 0,
        HCCL_ERROR("[EdgeInfo::%s] string['%s'] is not type of LinkType.", __func__, linkTypeStr.c_str()),
        InvalidParamsException, "string is not type of LinkType");
    return strToLinkType.at(linkTypeStr);
}

AddrPosition EdgeInfo::GetAddrPosition(string str) const
{
    if (str.empty()) {
        HCCL_WARNING("[EdgeInfo::%s] position is null, [default]position=DEVICE", __func__);
        return AddrPosition::DEVICE; //  默认取值为DEVICE
    }
    CHK_PRT_THROW(
        strToAddrPosition.count(str) == 0,
        HCCL_ERROR("[EdgeInfo::%s] string['%s'] is not type of AddrPosition.", __func__, str.c_str()),
        InvalidParamsException, "string is not type of AddrPosition");
    return strToAddrPosition.at(str);
}

std::string EdgeInfo::Describe() const
{
    stringstream protocolStr;
    protocolStr << "[";
    for (auto it = protocols.begin(); it != protocols.end(); ++it) {
        if (it != protocols.begin()) {
            protocolStr << ", ";
        }
        protocolStr << it->Describe();
    }
    protocolStr << "]";

    string localAPortsStr = DescribePorts(localAPorts);
    string localBPortsStr = DescribePorts(localBPorts);

    std::string description = "EdgeInfo{";
    description += StringFormat("netLayer=%u", netLayer);
    description += StringFormat("topoType=%s", topoType.Describe().c_str());
    description += StringFormat(", topoInstanceId=%u", topoInstId);
    description += StringFormat(", protocols=%s", protocolStr.str().c_str());
    description += StringFormat(", linkType=%s", linkType.Describe().c_str());
    description += StringFormat(", localA=%u", localA);
    description += StringFormat(", localAPortsStr=%s", localAPortsStr.c_str());
    description += StringFormat(", localB=%u", localB);
    description += StringFormat(", localBPortsStr=%s", localBPortsStr.c_str());
    description += StringFormat(", position=%s", position.Describe().c_str());
    description += "}";
    return description;
}

std::string EdgeInfo::DescribePorts(std::set<std::string> ports) const
{
    stringstream portsStr;
    portsStr << "[";
    for (auto it = ports.begin(); it != ports.end(); ++it) {
        if (it != ports.begin()) {
            portsStr << ", ";
        }
        portsStr << *it;
    }
    portsStr << "]";
    return portsStr.str();
}

void EdgeInfo::GetBinStream(BinaryStream& binaryStream) const
{
    binaryStream << netLayer << static_cast<u32>(linkType) << static_cast<u32>(topoType) << topoInstId;
    binaryStream << protocols.size();
    for (LinkProtocol protocol : protocols) {
        binaryStream << static_cast<u32>(protocol);
    }

    binaryStream << localA << localB;

    binaryStream << localAPorts.size();
    for (string port : localAPorts) {
        binaryStream << port;
    }

    binaryStream << localBPorts.size();
    for (string port : localBPorts) {
        binaryStream << port;
    }

    binaryStream << static_cast<u32>(position);
}

EdgeInfo::EdgeInfo(BinaryStream& binaryStream)
{
    binaryStream >> netLayer;
    u32 linkTypeTmp;
    binaryStream >> linkTypeTmp;
    linkType = static_cast<LinkType::Value>(linkTypeTmp);
    u32 topoTypeTmp;
    binaryStream >> topoTypeTmp;
    topoType = static_cast<TopoType::Value>(topoTypeTmp);
    binaryStream >> topoInstId;
    size_t protocolSize;
    binaryStream >> protocolSize;
    protocols.clear();
    for (size_t i = 0; i < protocolSize; i++) {
        u32 protocolTmp;
        binaryStream >> protocolTmp;
        LinkProtocol protocol = static_cast<LinkProtocol::Value>(protocolTmp);
        protocols.emplace(protocol);
    }

    binaryStream >> localA >> localB;

    size_t localAPortsSize;
    binaryStream >> localAPortsSize;
    localAPorts.clear();
    for (size_t i = 0; i < localAPortsSize; i++) {
        string port;
        binaryStream >> port;
        localAPorts.emplace(port);
    }

    size_t localBPortsSize;
    binaryStream >> localBPortsSize;
    localBPorts.clear();
    for (size_t i = 0; i < localBPortsSize; i++) {
        string port;
        binaryStream >> port;
        localBPorts.emplace(port);
    }

    u32 positionTmp;
    binaryStream >> positionTmp;
    position = static_cast<AddrPosition::Value>(positionTmp);
}

} // namespace Hccl
