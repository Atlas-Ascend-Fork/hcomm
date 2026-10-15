/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define HCCL_VM_MODULE "PROXY_COMMON"

#include "level1_proxy_common.h"

#include "acl/acl_rt.h"
#include <arpa/inet.h>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>

#include <nlohmann/json.hpp>

#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "sim_log.h"

using json = nlohmann::json;

namespace sim {

uint32_t GetCurrRankId()
{
    // Level1 has one communicator per current device. This is intentionally a
    // single-domain lookup and does not use level2's HcclComm handle mapping.
    const uint64_t deviceId = sim::runtime::GetCurrDeviceKey();
    if (deviceId == 0) {
        HCCL_VM_ERROR("cannot resolve current rank: no current level1 device");
        return UINT32_MAX;
    }
    const auto communicators = sim::runtime::Db::GetByPred<sim::runtime::Communicator>(
        HcclSim::Storage::Eq(&sim::runtime::Communicator::device_id, deviceId));
    if (!communicators.ok() || communicators->size() != 1) {
        HCCL_VM_ERROR(
            "cannot resolve current rank for level1 deviceId={}, communicators={}", deviceId,
            communicators.ok() ? communicators->size() : 0U);
        return UINT32_MAX;
    }
    return communicators->front().rank_id;
}

namespace {

    bool ParseTopoType(const std::string& typeName, RankTableTopoType& topoType)
    {
        if (typeName == "CLOS") {
            topoType = RankTableTopoType::CLOS;
        } else if (typeName == "1DMESH") {
            topoType = RankTableTopoType::MESH_1D;
        } else if (typeName == "2DMESH") {
            topoType = RankTableTopoType::MESH_2D;
        } else if (typeName == "A3_SERVER") {
            topoType = RankTableTopoType::A3_SERVER;
        } else if (typeName == "A2_AX_SERVER") {
            topoType = RankTableTopoType::A2_AX_SERVER;
        } else if (typeName == "TOPO_FILE_DESC") {
            topoType = RankTableTopoType::CUSTOM;
        } else {
            return false;
        }
        return true;
    }

    std::string GetSiblingTopoPath(const std::string& rankTablePath)
    {
        std::string::size_type slashPos = rankTablePath.find_last_of('/');
        if (slashPos == std::string::npos) {
            return "topo.json";
        }
        return rankTablePath.substr(0, slashPos + 1) + "topo.json";
    }

    bool ParseUniqueStringList(const json& object, const char* fieldName, std::vector<std::string>& values)
    {
        values.clear();
        if (!object.contains(fieldName) || !object[fieldName].is_array()) {
            return false;
        }

        for (const auto& value : object[fieldName]) {
            if (!value.is_string()) {
                return false;
            }
            std::string text = value.get<std::string>();
            if (text.empty()) {
                return false;
            }
            values.push_back(std::move(text));
        }
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        return !values.empty();
    }

    bool IsSupportedProtocol(const std::string& protocol)
    {
        static const std::set<std::string> supportedProtocols
            = {"UB_CTP", "UB_TP", "UBG", "ROCE", "HCCS", "PCIE", "TCP", "UB_MEM", "UBOE"};
        return supportedProtocols.find(protocol) != supportedProtocols.end();
    }

    bool HasCommonPort(const std::vector<std::string>& lhs, const std::vector<std::string>& rhs)
    {
        return std::any_of(lhs.begin(), lhs.end(), [&rhs](const std::string& port) {
            return std::find(rhs.begin(), rhs.end(), port) != rhs.end();
        });
    }

    std::vector<std::string> GetCommonPorts(const std::vector<std::string>& lhs, const std::vector<std::string>& rhs)
    {
        std::vector<std::string> commonPorts;
        std::set_intersection(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(commonPorts));
        return commonPorts;
    }

    int GetProtocolOrder(const std::string& protocol)
    {
        static const std::vector<std::string> protocolOrder
            = {"UB_CTP", "UB_TP", "ROCE", "HCCS", "TCP", "UB_MEM", "PCIE", "UBOE", "UBG"};
        auto protocolIt = std::find(protocolOrder.begin(), protocolOrder.end(), protocol);
        return static_cast<int>(std::distance(protocolOrder.begin(), protocolIt));
    }

    bool UsesPcieVirtualInterface(const TopoEdgeInfo& edge)
    {
        if (edge.protocols.empty()) {
            return false;
        }
        auto firstProtocol = std::min_element(
            edge.protocols.begin(), edge.protocols.end(), [](const std::string& lhs, const std::string& rhs) {
                return GetProtocolOrder(lhs) < GetProtocolOrder(rhs);
            });
        return *firstProtocol == "PCIE";
    }

} // namespace

RankTable::RankTable() = default;

RankTable& RankTable::Instance()
{
    static RankTable s_instance;
    return s_instance;
}

bool RankTable::HexCharToValue(char character, uint8_t& value)
{
    if (character >= '0' && character <= '9') {
        value = static_cast<uint8_t>(character - '0');
        return true;
    }
    character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    if (character >= 'a' && character <= 'f') {
        value = static_cast<uint8_t>(character - 'a' + 10);
        return true;
    }
    return false;
}

HcclResult RankTable::SetCommAddress(const RankEndpointInfo& endpoint, CommAddr& commAddr)
{
    std::memset(&commAddr, 0, sizeof(commAddr));
    if (endpoint.addressType == "EID") {
        if (endpoint.address.size() != COMM_ADDR_EID_LEN * 2U) {
            return HCCL_E_INTERNAL;
        }
        commAddr.type = COMM_ADDR_TYPE_EID;
        for (uint32_t index = 0; index < COMM_ADDR_EID_LEN; ++index) {
            uint8_t high = 0;
            uint8_t low = 0;
            if (!HexCharToValue(endpoint.address[index * 2U], high)
                || !HexCharToValue(endpoint.address[index * 2U + 1U], low)) {
                return HCCL_E_INTERNAL;
            }
            commAddr.eid[index] = static_cast<uint8_t>((high << 4U) | low);
        }
        return HCCL_SUCCESS;
    }
    if (endpoint.addressType == "IPV4") {
        commAddr.type = COMM_ADDR_TYPE_IP_V4;
        return inet_pton(AF_INET, endpoint.address.c_str(), &commAddr.addr) == 1 ? HCCL_SUCCESS : HCCL_E_INTERNAL;
    }
    if (endpoint.addressType == "IPV6") {
        commAddr.type = COMM_ADDR_TYPE_IP_V6;
        return inet_pton(AF_INET6, endpoint.address.c_str(), &commAddr.addr6) == 1 ? HCCL_SUCCESS : HCCL_E_INTERNAL;
    }
    if (endpoint.addressType == "ID") {
        commAddr.type = COMM_ADDR_TYPE_ID;
        try {
            commAddr.id = static_cast<uint32_t>(std::stoul(endpoint.address));
        } catch (const std::exception&) {
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }
    return HCCL_E_INTERNAL;
}

CommProtocol RankTable::GetCommProtocol(const std::string& protocol)
{
    if (protocol == "UB_CTP") {
        return COMM_PROTOCOL_UBC_CTP;
    }
    if (protocol == "UB_TP") {
        return COMM_PROTOCOL_UBC_TP;
    }
    if (protocol == "ROCE") {
        return COMM_PROTOCOL_ROCE;
    }
    if (protocol == "HCCS") {
        return COMM_PROTOCOL_HCCS;
    }
    if (protocol == "PCIE") {
        return COMM_PROTOCOL_PCIE;
    }
    if (protocol == "UB_MEM") {
        return COMM_PROTOCOL_UB_MEM;
    }
    if (protocol == "UBG") {
        // UBG 为兼容旧协议名（hcomm_res_defs.h: COMM_PROTOCOL_UBG = COMM_PROTOCOL_UB_RTP）
        return COMM_PROTOCOL_UBG;
    }
    if (protocol == "UBOE") {
        return COMM_PROTOCOL_UBOE;
    }
    return COMM_PROTOCOL_RESERVED;
}

HcclResult RankTable::SetEndpointDesc(const RankEndpointInfo& endpoint, EndpointDesc& endpointDesc)
{
    if (EndpointDescInit(&endpointDesc, 1) != HCCL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    HcclResult result = SetCommAddress(endpoint, endpointDesc.commAddr);
    if (result != HCCL_SUCCESS) {
        return result;
    }
    endpointDesc.protocol = GetCommProtocol(endpoint.protocol);
    endpointDesc.loc.locType
        = endpoint.position == TopoAddressPosition::DEVICE ? ENDPOINT_LOC_TYPE_DEVICE : ENDPOINT_LOC_TYPE_HOST;
    return HCCL_SUCCESS;
}

std::string RankTable::GetProtocolName(CommProtocol protocol)
{
    switch (protocol) {
        case COMM_PROTOCOL_UBC_CTP:
            return "UB_CTP";
        case COMM_PROTOCOL_UBC_TP:
            return "UB_TP";
        case COMM_PROTOCOL_ROCE:
            return "ROCE";
        case COMM_PROTOCOL_HCCS:
            return "HCCS";
        case COMM_PROTOCOL_PCIE:
            return "PCIE";
        case COMM_PROTOCOL_UB_MEM:
            return "UB_MEM";
        case COMM_PROTOCOL_UBG:
            return "UBG";
        case COMM_PROTOCOL_UBOE:
            return "UBOE";
        case COMM_PROTOCOL_RESERVED:
            return "TCP";
        default:
            return "";
    }
}

HcclResult RankTable::GetCommAddressText(const CommAddr& commAddr, std::string& addressType, std::string& address)
{
    char addressBuffer[INET6_ADDRSTRLEN] = {};
    switch (commAddr.type) {
        case COMM_ADDR_TYPE_IP_V4:
            addressType = "IPV4";
            if (inet_ntop(AF_INET, &commAddr.addr, addressBuffer, sizeof(addressBuffer)) == nullptr) {
                return HCCL_E_INTERNAL;
            }
            address = addressBuffer;
            return HCCL_SUCCESS;
        case COMM_ADDR_TYPE_IP_V6:
            addressType = "IPV6";
            if (inet_ntop(AF_INET6, &commAddr.addr6, addressBuffer, sizeof(addressBuffer)) == nullptr) {
                return HCCL_E_INTERNAL;
            }
            address = addressBuffer;
            return HCCL_SUCCESS;
        case COMM_ADDR_TYPE_EID: {
            static const char hexDigits[] = "0123456789abcdef";
            addressType = "EID";
            address.resize(COMM_ADDR_EID_LEN * 2U);
            for (uint32_t index = 0; index < COMM_ADDR_EID_LEN; ++index) {
                address[index * 2U] = hexDigits[commAddr.eid[index] >> 4U];
                address[index * 2U + 1U] = hexDigits[commAddr.eid[index] & 0x0fU];
            }
            return HCCL_SUCCESS;
        }
        case COMM_ADDR_TYPE_ID:
            addressType = "ID";
            address = std::to_string(commAddr.id);
            return HCCL_SUCCESS;
        default:
            return HCCL_E_PARA;
    }
}

RankTableQueryStatus RankTable::ConvertTopoType(RankTableTopoType internalType, bool allowCustom, CommTopo& topoType)
{
    switch (internalType) {
        case RankTableTopoType::CLOS:
            topoType = COMM_TOPO_CLOS;
            return RankTableQueryStatus::SUCCESS;
        case RankTableTopoType::MESH_1D:
            topoType = COMM_TOPO_1DMESH;
            return RankTableQueryStatus::SUCCESS;
        case RankTableTopoType::A3_SERVER:
            topoType = COMM_TOPO_910_93;
            return RankTableQueryStatus::SUCCESS;
        case RankTableTopoType::A2_AX_SERVER:
            topoType = COMM_TOPO_A2AXSERVER;
            return RankTableQueryStatus::SUCCESS;
        case RankTableTopoType::CUSTOM:
            if (allowCustom) {
                topoType = COMM_TOPO_CUSTOM;
                return RankTableQueryStatus::SUCCESS;
            }
            return RankTableQueryStatus::UNSUPPORTED_TYPE;
        case RankTableTopoType::MESH_2D:
        default:
            return RankTableQueryStatus::UNSUPPORTED_TYPE;
    }
}

RankTableQueryStatus RankTable::GetCommTopoTypeByLayer(uint32_t rankId, uint32_t netLayer, CommTopo& topoType)
{
    RankTableTopoType internalType;
    RankTableQueryStatus status = GetTopoTypeByLayer(rankId, netLayer, internalType);
    if (status != RankTableQueryStatus::SUCCESS) {
        return status;
    }
    return ConvertTopoType(internalType, true, topoType);
}

RankTableQueryStatus
RankTable::GetCommTopoType(uint32_t rankId, uint32_t netLayer, uint32_t topoInstanceId, CommTopo& topoType)
{
    RankTableTopoType internalType;
    RankTableQueryStatus status = GetTopoType(rankId, netLayer, topoInstanceId, internalType);
    if (status != RankTableQueryStatus::SUCCESS) {
        return status;
    }
    return ConvertTopoType(internalType, false, topoType);
}

RankTableQueryStatus RankTable::GetCommLinks(
    uint32_t rankId, uint32_t netLayer, uint32_t sourceRank, uint32_t destinationRank, std::vector<CommLink>& links)
{
    links.clear();
    std::vector<RankLinkInfo> internalLinks;
    RankTableQueryStatus status = GetLinks(rankId, netLayer, sourceRank, destinationRank, internalLinks);
    if (status != RankTableQueryStatus::SUCCESS) {
        return status;
    }

    links.reserve(internalLinks.size());
    for (const auto& internalLink : internalLinks) {
        CommLink commLink{};
        if (CommLinkInit(&commLink, 1) != HCCL_SUCCESS
            || SetEndpointDesc(internalLink.sourceEndpoint, commLink.srcEndpointDesc) != HCCL_SUCCESS
            || SetEndpointDesc(internalLink.destinationEndpoint, commLink.dstEndpointDesc) != HCCL_SUCCESS) {
            links.clear();
            return RankTableQueryStatus::INVALID_DATA;
        }

        CommProtocol protocol = GetCommProtocol(internalLink.protocol);
        commLink.linkAttr.linkProtocol = protocol;
        commLink.linkAttr.hop = internalLink.hop;
        commLink.srcEndpointDesc.protocol = protocol;
        commLink.dstEndpointDesc.protocol = protocol;
        if (commLink.srcEndpointDesc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
            commLink.srcEndpointDesc.loc.device.devPhyId = internalLink.sourceEndpoint.deviceId;
        }
        if (commLink.dstEndpointDesc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
            commLink.dstEndpointDesc.loc.device.devPhyId = internalLink.destinationEndpoint.deviceId;
        }
        links.push_back(std::move(commLink));
    }
    return RankTableQueryStatus::SUCCESS;
}

RankTableQueryStatus RankTable::GetCommEndpointDescList(
    uint32_t rankId, uint32_t netLayer, uint32_t topoInstanceId, std::vector<EndpointDesc>& endpointDescList)
{
    endpointDescList.clear();
    std::vector<RankEndpointInfo> internalEndpoints;
    RankTableQueryStatus status = GetEndpointList(rankId, netLayer, topoInstanceId, internalEndpoints);
    if (status != RankTableQueryStatus::SUCCESS) {
        return status;
    }

    endpointDescList.reserve(internalEndpoints.size());
    for (const auto& internalEndpoint : internalEndpoints) {
        EndpointDesc endpointDesc{};
        if (SetEndpointDesc(internalEndpoint, endpointDesc) != HCCL_SUCCESS) {
            endpointDescList.clear();
            return RankTableQueryStatus::INVALID_DATA;
        }
        endpointDescList.push_back(std::move(endpointDesc));
    }
    return RankTableQueryStatus::SUCCESS;
}

RankTableQueryStatus RankTable::GetCommEndpointInfo(
    uint32_t rankId, const EndpointDesc& endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info)
{
    if (info == nullptr) {
        return RankTableQueryStatus::INVALID_PARAMETER;
    }

    std::string addressType;
    std::string address;
    if (GetCommAddressText(endpointDesc.commAddr, addressType, address) != HCCL_SUCCESS) {
        return RankTableQueryStatus::INVALID_PARAMETER;
    }
    std::string protocol = GetProtocolName(endpointDesc.protocol);
    if (protocol.empty()) {
        return RankTableQueryStatus::INVALID_PARAMETER;
    }

    RankEndpointInfo endpointInfo;
    RankTableQueryStatus status = FindEndpoint(rankId, addressType, address, protocol, endpointInfo);
    if (status != RankTableQueryStatus::SUCCESS) {
        return status;
    }

    switch (endpointAttr) {
        case ENDPOINT_ATTR_BW_COEFF:
            if (infoLen != sizeof(EndpointAttrBwCoeff)) {
                return RankTableQueryStatus::INVALID_PARAMETER;
            }
            *static_cast<EndpointAttrBwCoeff*>(info) = static_cast<EndpointAttrBwCoeff>(endpointInfo.ports.size());
            return RankTableQueryStatus::SUCCESS;
        case ENDPOINT_ATTR_DIE_ID:
            if (infoLen != sizeof(EndpointAttrDieId)) {
                return RankTableQueryStatus::INVALID_PARAMETER;
            }
            *static_cast<EndpointAttrDieId*>(info) = endpointInfo.localDieId;
            return RankTableQueryStatus::SUCCESS;
        case ENDPOINT_ATTR_LOCATION:
            if (infoLen != sizeof(EndpointAttrLocation)) {
                return RankTableQueryStatus::INVALID_PARAMETER;
            }
            *static_cast<EndpointAttrLocation*>(info) = endpointInfo.position == TopoAddressPosition::HOST ? 0U : 1U;
            return RankTableQueryStatus::SUCCESS;
        default:
            return RankTableQueryStatus::INVALID_PARAMETER;
    }
}

bool RankTable::Load(const char* clusterInfo)
{
    if (clusterInfo == nullptr) {
        HCCL_VM_ERROR("{}: clusterInfo is nullptr", __func__);
        return false;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_loaded) {
        return true;
    }

    std::string rawStr(clusterInfo);
    std::string jsonStr;

    bool loadFromFile = rawStr.empty() || rawStr[0] != '{';
    if (!loadFromFile) {
        jsonStr = rawStr;
    } else {
        std::ifstream ifs(rawStr);
        if (!ifs.is_open()) {
            HCCL_VM_ERROR("{}: failed to open rank table file: {}", __func__, clusterInfo);
            return false;
        }
        std::stringstream ss;
        ss << ifs.rdbuf();
        jsonStr = ss.str();
    }

    m_rankTablePath = loadFromFile ? rawStr : "";
    if (!ParseFromJsonString(jsonStr)) {
        m_rankTablePath.clear();
        return false;
    }

    // topo.json只在Load阶段解析，查询阶段不再修改单例内部数据。
    m_topologyLoaded = LoadTopologyIntoRankList();
    m_loaded = true;
    return true;
}

bool RankTable::EnsureLoaded()
{
    if (m_loaded) {
        return true;
    }

    const char* rankTablePath = std::getenv("RANK_TABLE_FILE");
    if (rankTablePath == nullptr || rankTablePath[0] == '\0') {
        HCCL_VM_ERROR("{}: RANK_TABLE_FILE is not set or is empty", __func__);
        return false;
    }
    return Load(rankTablePath);
}

bool RankTable::IsLoaded() const { return m_loaded; }

RankTableQueryStatus RankTable::GetNetLayers(uint32_t rankId, std::vector<uint32_t>& netLayers)
{
    netLayers.clear();
    if (!EnsureLoaded()) {
        return RankTableQueryStatus::LOAD_FAILED;
    }

    auto rankIt = std::find_if(m_rankList.begin(), m_rankList.end(), [rankId](const RankInfo& rankInfo) {
        return rankInfo.rankId == rankId;
    });
    if (rankIt == m_rankList.end()) {
        HCCL_VM_ERROR("{}: rank {:d} was not found in rank table", __func__, rankId);
        return RankTableQueryStatus::RANK_NOT_FOUND;
    }

    // set在插入时自动完成排序和去重，避免先填充vector再执行sort和unique。
    std::set<uint32_t> uniqueLayers;
    for (const auto& levelInfo : rankIt->levelList) {
        uniqueLayers.insert(levelInfo.netLayer);
    }
    if (uniqueLayers.empty()) {
        HCCL_VM_ERROR("{}: no network layer found for rank {:d}", __func__, rankId);
        return RankTableQueryStatus::INVALID_DATA;
    }

    // 对外仍返回连续的vector，桩函数可以安全地通过data()取得数组地址。
    netLayers.assign(uniqueLayers.begin(), uniqueLayers.end());
    return RankTableQueryStatus::SUCCESS;
}

RankTableQueryStatus RankTable::GetRanksByLayer(uint32_t rankId, uint32_t netLayer, std::vector<uint32_t>& ranks)
{
    ranks.clear();
    if (!EnsureLoaded()) {
        return RankTableQueryStatus::LOAD_FAILED;
    }

    auto currentRankIt = std::find_if(m_rankList.begin(), m_rankList.end(), [rankId](const RankInfo& rankInfo) {
        return rankInfo.rankId == rankId;
    });
    if (currentRankIt == m_rankList.end()) {
        HCCL_VM_ERROR("{}: rank {:d} was not found in rank table", __func__, rankId);
        return RankTableQueryStatus::RANK_NOT_FOUND;
    }

    auto currentLevelIt = std::find_if(
        currentRankIt->levelList.begin(), currentRankIt->levelList.end(), [netLayer](const RankLevelInfo& levelInfo) {
            return levelInfo.netLayer == netLayer;
        });
    if (currentLevelIt == currentRankIt->levelList.end()) {
        HCCL_VM_ERROR("{}: rank {:d} does not contain network layer {:d}", __func__, rankId, netLayer);
        return RankTableQueryStatus::LAYER_NOT_FOUND;
    }
    if (currentLevelIt->netInstanceId.empty()) {
        HCCL_VM_ERROR("{}: net_instance_id of rank {:d} on network layer {:d} is empty", __func__, rankId, netLayer);
        return RankTableQueryStatus::INVALID_DATA;
    }

    const std::string& netInstanceId = currentLevelIt->netInstanceId;
    for (const auto& rankInfo : m_rankList) {
        auto sameInstanceIt = std::find_if(
            rankInfo.levelList.begin(), rankInfo.levelList.end(),
            [netLayer, &netInstanceId](const RankLevelInfo& levelInfo) {
                return levelInfo.netLayer == netLayer && levelInfo.netInstanceId == netInstanceId;
            });
        if (sameInstanceIt != rankInfo.levelList.end()) {
            ranks.push_back(rankInfo.rankId);
        }
    }

    std::sort(ranks.begin(), ranks.end());
    ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
    if (ranks.empty()) {
        HCCL_VM_ERROR(
            "{}: network instance {} on layer {:d} does not contain any rank", __func__, netInstanceId, netLayer);
        return RankTableQueryStatus::INVALID_DATA;
    }
    return RankTableQueryStatus::SUCCESS;
}

RankTableQueryStatus RankTable::GetRankSizeByLayer(uint32_t rankId, uint32_t netLayer, uint32_t& rankNum)
{
    std::vector<uint32_t> ranks;
    RankTableQueryStatus status = GetRanksByLayer(rankId, netLayer, ranks);
    if (status != RankTableQueryStatus::SUCCESS) {
        return status;
    }

    rankNum = static_cast<uint32_t>(ranks.size());
    return RankTableQueryStatus::SUCCESS;
}

RankTableQueryStatus
RankTable::GetInstSizeListByLayer(uint32_t rankId, uint32_t netLayer, std::vector<uint32_t>& instSizeList)
{
    instSizeList.clear();
    if (!EnsureLoaded()) {
        return RankTableQueryStatus::LOAD_FAILED;
    }

    auto currentRankIt = std::find_if(m_rankList.begin(), m_rankList.end(), [rankId](const RankInfo& rankInfo) {
        return rankInfo.rankId == rankId;
    });
    if (currentRankIt == m_rankList.end()) {
        HCCL_VM_ERROR("{}: rank {:d} was not found in rank table", __func__, rankId);
        return RankTableQueryStatus::RANK_NOT_FOUND;
    }

    auto currentLevelIt = std::find_if(
        currentRankIt->levelList.begin(), currentRankIt->levelList.end(), [netLayer](const RankLevelInfo& levelInfo) {
            return levelInfo.netLayer == netLayer;
        });
    if (currentLevelIt == currentRankIt->levelList.end()) {
        HCCL_VM_ERROR("{}: rank {:d} does not contain network layer {:d}", __func__, rankId, netLayer);
        return RankTableQueryStatus::LAYER_NOT_FOUND;
    }

    std::map<std::string, std::set<uint32_t>> ranksByInstance;
    for (const auto& rankInfo : m_rankList) {
        for (const auto& levelInfo : rankInfo.levelList) {
            if (levelInfo.netLayer != netLayer) {
                continue;
            }
            if (levelInfo.netInstanceId.empty()) {
                HCCL_VM_ERROR(
                    "{}: net_instance_id of rank {:d} on network layer {:d} is empty", __func__, rankInfo.rankId,
                    netLayer);
                return RankTableQueryStatus::INVALID_DATA;
            }
            ranksByInstance[levelInfo.netInstanceId].insert(rankInfo.rankId);
        }
    }

    if (ranksByInstance.empty()) {
        HCCL_VM_ERROR("{}: no network instance found on network layer {:d}", __func__, netLayer);
        return RankTableQueryStatus::INVALID_DATA;
    }

    instSizeList.reserve(ranksByInstance.size());
    for (const auto& instanceRanks : ranksByInstance) {
        instSizeList.push_back(static_cast<uint32_t>(instanceRanks.second.size()));
    }
    return RankTableQueryStatus::SUCCESS;
}

RankTableQueryStatus RankTable::GetTopoTypeByLayer(uint32_t rankId, uint32_t netLayer, RankTableTopoType& topoType)
{
    if (!EnsureLoaded()) {
        return RankTableQueryStatus::LOAD_FAILED;
    }

    auto rankIt = std::find_if(m_rankList.begin(), m_rankList.end(), [rankId](const RankInfo& rankInfo) {
        return rankInfo.rankId == rankId;
    });
    if (rankIt == m_rankList.end()) {
        HCCL_VM_ERROR("{}: rank {:d} was not found in rank table", __func__, rankId);
        return RankTableQueryStatus::RANK_NOT_FOUND;
    }

    auto levelIt
        = std::find_if(rankIt->levelList.begin(), rankIt->levelList.end(), [netLayer](const RankLevelInfo& levelInfo) {
              return levelInfo.netLayer == netLayer;
          });
    if (levelIt == rankIt->levelList.end()) {
        HCCL_VM_ERROR("{}: rank {:d} does not contain network layer {:d}", __func__, rankId, netLayer);
        return RankTableQueryStatus::LAYER_NOT_FOUND;
    }

    const std::string& netType = levelIt->netType;
    if (!ParseTopoType(netType, topoType)) {
        HCCL_VM_ERROR(
            "{}: unsupported net_type '{}' for rank {:d} on network layer {:d}", __func__, netType, rankId, netLayer);
        return RankTableQueryStatus::INVALID_DATA;
    }

    return RankTableQueryStatus::SUCCESS;
}

RankTableQueryStatus
RankTable::GetTopoInstsByLayer(uint32_t rankId, uint32_t netLayer, std::vector<uint32_t>& topoInstanceIds)
{
    topoInstanceIds.clear();
    if (!EnsureLoaded()) {
        return RankTableQueryStatus::LOAD_FAILED;
    }

    auto rankIt = std::find_if(m_rankList.begin(), m_rankList.end(), [rankId](const RankInfo& rankInfo) {
        return rankInfo.rankId == rankId;
    });
    if (rankIt == m_rankList.end()) {
        HCCL_VM_ERROR("{}: rank {:d} was not found in rank table", __func__, rankId);
        return RankTableQueryStatus::RANK_NOT_FOUND;
    }

    auto layerIt
        = std::find_if(rankIt->levelList.begin(), rankIt->levelList.end(), [netLayer](const RankLevelInfo& levelInfo) {
              return levelInfo.netLayer == netLayer;
          });
    if (layerIt == rankIt->levelList.end()) {
        HCCL_VM_ERROR("{}: rank {:d} does not contain network layer {:d}", __func__, rankId, netLayer);
        return RankTableQueryStatus::LAYER_NOT_FOUND;
    }

    if (!m_topologyLoaded) {
        return RankTableQueryStatus::LOAD_FAILED;
    }

    // 实例归属在 Load 阶段按 topo.json 实例声明（local id 集合）算好并缓存在 level 上，
    // 与"本 rank 是否在该实例内有物理边"无关——单 rank 网络实例没有边，但仍是合法实例。
    topoInstanceIds = layerIt->topoInstanceIds;
    return RankTableQueryStatus::SUCCESS;
}

RankTableQueryStatus
RankTable::GetTopoType(uint32_t rankId, uint32_t netLayer, uint32_t topoInstanceId, RankTableTopoType& topoType)
{
    if (!EnsureLoaded()) {
        return RankTableQueryStatus::LOAD_FAILED;
    }

    auto rankIt = std::find_if(m_rankList.begin(), m_rankList.end(), [rankId](const RankInfo& rankInfo) {
        return rankInfo.rankId == rankId;
    });
    if (rankIt == m_rankList.end()) {
        HCCL_VM_ERROR("{}: rank {:d} was not found in rank table", __func__, rankId);
        return RankTableQueryStatus::RANK_NOT_FOUND;
    }

    auto layerIt
        = std::find_if(rankIt->levelList.begin(), rankIt->levelList.end(), [netLayer](const RankLevelInfo& levelInfo) {
              return levelInfo.netLayer == netLayer;
          });
    if (layerIt == rankIt->levelList.end()) {
        HCCL_VM_ERROR("{}: rank {:d} does not contain network layer {:d}", __func__, rankId, netLayer);
        return RankTableQueryStatus::LAYER_NOT_FOUND;
    }

    if (!m_topologyLoaded) {
        return RankTableQueryStatus::LOAD_FAILED;
    }

    // 拓扑类型来自 topo.json 的实例声明表（(netLayer, topoInstanceId) -> 类型），
    // 与是否有物理边无关，故单 rank 实例同样能返回类型。
    const auto typeIt = m_topoInstanceTypes.find(std::make_pair(netLayer, topoInstanceId));
    if (typeIt == m_topoInstanceTypes.end()) {
        HCCL_VM_ERROR(
            "{}: topology instance {:d} was not found on network layer {:d}", __func__, topoInstanceId, netLayer);
        return RankTableQueryStatus::TOPO_INSTANCE_NOT_FOUND;
    }

    topoType = typeIt->second;
    return RankTableQueryStatus::SUCCESS;
}

RankTableQueryStatus RankTable::GetEndpointList(
    uint32_t rankId, uint32_t netLayer, uint32_t topoInstanceId, std::vector<RankEndpointInfo>& endpoints)
{
    endpoints.clear();
    if (!EnsureLoaded()) {
        return RankTableQueryStatus::LOAD_FAILED;
    }

    auto rankIt = std::find_if(m_rankList.begin(), m_rankList.end(), [rankId](const RankInfo& rankInfo) {
        return rankInfo.rankId == rankId;
    });
    if (rankIt == m_rankList.end()) {
        HCCL_VM_ERROR("{}: rank {:d} was not found in rank table", __func__, rankId);
        return RankTableQueryStatus::RANK_NOT_FOUND;
    }

    auto levelIt
        = std::find_if(rankIt->levelList.begin(), rankIt->levelList.end(), [netLayer](const RankLevelInfo& levelInfo) {
              return levelInfo.netLayer == netLayer;
          });
    if (levelIt == rankIt->levelList.end()) {
        HCCL_VM_ERROR("{}: rank {:d} does not contain network layer {:d}", __func__, rankId, netLayer);
        return RankTableQueryStatus::LAYER_NOT_FOUND;
    }

    if (!m_topologyLoaded) {
        return RankTableQueryStatus::LOAD_FAILED;
    }

    using EndpointInterfaceKey = std::tuple<
        std::string, std::string, std::vector<std::string>, TopoAddressPosition, TopoLinkType, std::vector<std::string>,
        RankTableTopoType, uint32_t>;
    std::set<EndpointInterfaceKey> endpointInterfaces;

    auto hasPeerInterface
        = [this, netLayer,
           &levelIt](uint32_t peerLocalId, const std::vector<std::string>& peerPorts, bool usePcieVirtualInterface) {
              for (const auto& peerRank : m_rankList) {
                  if (peerRank.localId != peerLocalId) {
                      continue;
                  }
                  auto peerLevelIt = std::find_if(
                      peerRank.levelList.begin(), peerRank.levelList.end(),
                      [netLayer, &levelIt](const RankLevelInfo& peerLevel) {
                          return peerLevel.netLayer == netLayer && peerLevel.netInstanceId == levelIt->netInstanceId;
                      });
                  if (peerLevelIt == peerRank.levelList.end()) {
                      continue;
                  }
                  if (usePcieVirtualInterface) {
                      return true;
                  }
                  for (const auto& peerAddress : peerLevelIt->rankAddressList) {
                      if (HasCommonPort(peerAddress.ports, peerPorts)) {
                          return true;
                      }
                  }
              }
              return false;
          };

    for (const auto& edge : levelIt->topoEdgeList) {
        if (edge.netLayer != netLayer || edge.topoInstanceId != topoInstanceId) {
            continue;
        }

        const std::vector<std::string>* localPorts = nullptr;
        const std::vector<std::string>* peerPorts = nullptr;
        uint32_t peerLocalId = 0;
        if (edge.localA == rankIt->localId) {
            localPorts = &edge.localAPorts;
            if (edge.linkType == TopoLinkType::PEER_TO_PEER) {
                peerLocalId = edge.localB;
                peerPorts = &edge.localBPorts;
            }
        } else if (edge.linkType == TopoLinkType::PEER_TO_PEER && edge.localB == rankIt->localId) {
            localPorts = &edge.localBPorts;
            peerLocalId = edge.localA;
            peerPorts = &edge.localAPorts;
        } else {
            continue;
        }

        // HComm只在第0层根据点到点物理边构造rank接口；另一端没有可用
        // rank或端口时，当前端也不会生成接口。
        bool usePcieVirtualInterface = netLayer == 0 && UsesPcieVirtualInterface(edge);
        if (edge.linkType == TopoLinkType::PEER_TO_PEER) {
            if (netLayer != 0 || peerPorts == nullptr
                || !hasPeerInterface(peerLocalId, *peerPorts, usePcieVirtualInterface)) {
                continue;
            }
        }

        if (usePcieVirtualInterface) {
            endpointInterfaces.emplace(
                "ID", std::to_string(rankIt->deviceId), std::vector<std::string>{"d2h"}, edge.position, edge.linkType,
                edge.protocols, edge.topoType, edge.topoInstanceId);
            continue;
        }

        // 第0层会按通信地址合并同一条边上的端口；其他层的点到网络连接
        // 则按rank_addr_list中的每条地址记录建立接口。
        if (netLayer == 0) {
            std::map<std::pair<std::string, std::string>, std::vector<std::string>> portsByAddress;
            for (const auto& addressInfo : levelIt->rankAddressList) {
                std::vector<std::string> commonPorts = GetCommonPorts(addressInfo.ports, *localPorts);
                if (commonPorts.empty()) {
                    continue;
                }
                std::pair<std::string, std::string> addressKey{addressInfo.addressType, addressInfo.address};
                auto& addressPorts = portsByAddress[addressKey];
                addressPorts.insert(addressPorts.end(), commonPorts.begin(), commonPorts.end());
                std::sort(addressPorts.begin(), addressPorts.end());
                addressPorts.erase(std::unique(addressPorts.begin(), addressPorts.end()), addressPorts.end());
            }
            for (const auto& addressPorts : portsByAddress) {
                endpointInterfaces.emplace(
                    addressPorts.first.first, addressPorts.first.second, addressPorts.second, edge.position,
                    edge.linkType, edge.protocols, edge.topoType, edge.topoInstanceId);
            }
        } else if (edge.linkType == TopoLinkType::PEER_TO_NETWORK) {
            for (const auto& addressInfo : levelIt->rankAddressList) {
                std::vector<std::string> commonPorts = GetCommonPorts(addressInfo.ports, *localPorts);
                if (commonPorts.empty()) {
                    continue;
                }
                endpointInterfaces.emplace(
                    addressInfo.addressType, addressInfo.address, std::move(commonPorts), edge.position, edge.linkType,
                    edge.protocols, edge.topoType, edge.topoInstanceId);
            }
        }
    }

    for (const auto& endpointInterface : endpointInterfaces) {
        uint32_t localDieId = 0;
        const auto& ports = std::get<2>(endpointInterface);
        if (!ports.empty()) {
            std::string::size_type slashPos = ports.front().find('/');
            if (slashPos != std::string::npos && slashPos > 0) {
                try {
                    localDieId = static_cast<uint32_t>(std::stoul(ports.front().substr(0, slashPos)));
                } catch (const std::exception&) {
                    localDieId = 0;
                }
            }
        }

        for (const auto& protocol : std::get<5>(endpointInterface)) {
            RankEndpointInfo endpoint{};
            endpoint.addressType = std::get<0>(endpointInterface);
            endpoint.address = std::get<1>(endpointInterface);
            endpoint.ports = ports;
            endpoint.position = std::get<3>(endpointInterface);
            endpoint.protocol = protocol;
            endpoint.deviceId = rankIt->deviceId;
            endpoint.localDieId = localDieId;
            endpoints.push_back(std::move(endpoint));
        }
    }
    return RankTableQueryStatus::SUCCESS;
}

RankTableQueryStatus
RankTable::GetEndpointNum(uint32_t rankId, uint32_t netLayer, uint32_t topoInstanceId, uint32_t& endpointNum)
{
    endpointNum = 0;
    std::vector<RankEndpointInfo> endpoints;
    RankTableQueryStatus status = GetEndpointList(rankId, netLayer, topoInstanceId, endpoints);
    if (status != RankTableQueryStatus::SUCCESS) {
        return status;
    }
    if (endpoints.size() > std::numeric_limits<uint32_t>::max()) {
        HCCL_VM_ERROR(
            "{}: endpoint count exceeds uint32 range for rank {:d}, layer {:d}, topology instance {:d}", __func__,
            rankId, netLayer, topoInstanceId);
        return RankTableQueryStatus::INVALID_DATA;
    }
    endpointNum = static_cast<uint32_t>(endpoints.size());
    return RankTableQueryStatus::SUCCESS;
}

RankTableQueryStatus RankTable::FindEndpoint(
    uint32_t rankId, const std::string& addressType, const std::string& address, const std::string& protocol,
    RankEndpointInfo& endpoint)
{
    if (!EnsureLoaded()) {
        return RankTableQueryStatus::LOAD_FAILED;
    }

    if (!m_topologyLoaded) {
        return RankTableQueryStatus::LOAD_FAILED;
    }

    auto rankIt = std::find_if(m_rankList.begin(), m_rankList.end(), [rankId](const RankInfo& rankInfo) {
        return rankInfo.rankId == rankId;
    });
    if (rankIt == m_rankList.end()) {
        HCCL_VM_ERROR("{}: rank {:d} was not found in rank table", __func__, rankId);
        return RankTableQueryStatus::RANK_NOT_FOUND;
    }

    std::vector<std::pair<uint32_t, uint32_t>> layerInstances;
    for (const auto& levelInfo : rankIt->levelList) {
        for (const auto& edge : levelInfo.topoEdgeList) {
            layerInstances.emplace_back(levelInfo.netLayer, edge.topoInstanceId);
        }
    }

    std::sort(layerInstances.begin(), layerInstances.end());
    layerInstances.erase(std::unique(layerInstances.begin(), layerInstances.end()), layerInstances.end());
    for (const auto& layerInstance : layerInstances) {
        std::vector<RankEndpointInfo> endpoints;
        RankTableQueryStatus status = GetEndpointList(rankId, layerInstance.first, layerInstance.second, endpoints);
        if (status != RankTableQueryStatus::SUCCESS) {
            return status;
        }
        auto endpointIt = std::find_if(
            endpoints.begin(), endpoints.end(), [&addressType, &address, &protocol](const RankEndpointInfo& candidate) {
                return candidate.addressType == addressType && candidate.address == address
                       && candidate.protocol == protocol;
            });
        if (endpointIt != endpoints.end()) {
            endpoint = *endpointIt;
            return RankTableQueryStatus::SUCCESS;
        }
    }

    HCCL_VM_ERROR(
        "{}: endpoint was not found for rank {:d}, address type {}, address {}, protocol {}", __func__, rankId,
        addressType, address, protocol);
    return RankTableQueryStatus::ENDPOINT_NOT_FOUND;
}

RankTableQueryStatus RankTable::GetLinks(
    uint32_t rankId, uint32_t netLayer, uint32_t sourceRank, uint32_t destinationRank, std::vector<RankLinkInfo>& links)
{
    links.clear();
    if (!EnsureLoaded()) {
        return RankTableQueryStatus::LOAD_FAILED;
    }

    auto currentRankIt = std::find_if(m_rankList.begin(), m_rankList.end(), [rankId](const RankInfo& rankInfo) {
        return rankInfo.rankId == rankId;
    });
    if (currentRankIt == m_rankList.end()) {
        return RankTableQueryStatus::RANK_NOT_FOUND;
    }
    auto currentLevelIt = std::find_if(
        currentRankIt->levelList.begin(), currentRankIt->levelList.end(), [netLayer](const RankLevelInfo& levelInfo) {
            return levelInfo.netLayer == netLayer;
        });
    if (currentLevelIt == currentRankIt->levelList.end()) {
        return RankTableQueryStatus::LAYER_NOT_FOUND;
    }

    auto sourceRankIt = std::find_if(m_rankList.begin(), m_rankList.end(), [sourceRank](const RankInfo& rankInfo) {
        return rankInfo.rankId == sourceRank;
    });
    auto destinationRankIt
        = std::find_if(m_rankList.begin(), m_rankList.end(), [destinationRank](const RankInfo& rankInfo) {
              return rankInfo.rankId == destinationRank;
          });
    if (sourceRankIt == m_rankList.end() || destinationRankIt == m_rankList.end()) {
        return RankTableQueryStatus::RANK_NOT_FOUND;
    }

    auto sourceLevelIt = std::find_if(
        sourceRankIt->levelList.begin(), sourceRankIt->levelList.end(), [netLayer](const RankLevelInfo& levelInfo) {
            return levelInfo.netLayer == netLayer;
        });
    auto destinationLevelIt = std::find_if(
        destinationRankIt->levelList.begin(), destinationRankIt->levelList.end(),
        [netLayer](const RankLevelInfo& levelInfo) {
            return levelInfo.netLayer == netLayer;
        });
    if (sourceLevelIt == sourceRankIt->levelList.end() || destinationLevelIt == destinationRankIt->levelList.end()) {
        return RankTableQueryStatus::LAYER_NOT_FOUND;
    }

    if (!m_topologyLoaded) {
        return RankTableQueryStatus::LOAD_FAILED;
    }

    struct InterfaceData {
        RankEndpointInfo endpoint;
        std::vector<std::string> protocols;
        uint32_t topoInstanceId = 0;
        std::string planeId;
    };

    auto getDieId = [](const std::vector<std::string>& ports) {
        if (ports.empty()) {
            return 0U;
        }
        std::string::size_type slashPos = ports.front().find('/');
        if (slashPos == std::string::npos || slashPos == 0) {
            return 0U;
        }
        try {
            return static_cast<uint32_t>(std::stoul(ports.front().substr(0, slashPos)));
        } catch (const std::exception&) {
            return 0U;
        }
    };

    auto makeInterfaces = [netLayer, &getDieId](
                              const RankInfo& rankInfo, const RankLevelInfo& levelInfo, const TopoEdgeInfo& edge,
                              const std::vector<std::string>& localPorts) {
        std::vector<InterfaceData> interfaces;
        if (netLayer == 0 && UsesPcieVirtualInterface(edge)) {
            InterfaceData interfaceData{};
            interfaceData.endpoint.addressType = "ID";
            interfaceData.endpoint.address = std::to_string(rankInfo.deviceId);
            interfaceData.endpoint.ports = {"d2h"};
            interfaceData.endpoint.position = edge.position;
            interfaceData.endpoint.deviceId = rankInfo.deviceId;
            interfaceData.endpoint.localDieId = 0;
            interfaceData.protocols = edge.protocols;
            interfaceData.topoInstanceId = edge.topoInstanceId;
            interfaces.push_back(std::move(interfaceData));
            return interfaces;
        }

        if (netLayer == 0) {
            using AddressKey = std::pair<std::string, std::string>;
            std::map<AddressKey, std::vector<std::string>> portsByAddress;
            for (const auto& addressInfo : levelInfo.rankAddressList) {
                std::vector<std::string> commonPorts = GetCommonPorts(addressInfo.ports, localPorts);
                if (commonPorts.empty()) {
                    continue;
                }
                AddressKey key{addressInfo.addressType, addressInfo.address};
                auto& groupedPorts = portsByAddress[key];
                groupedPorts.insert(groupedPorts.end(), commonPorts.begin(), commonPorts.end());
                std::sort(groupedPorts.begin(), groupedPorts.end());
                groupedPorts.erase(std::unique(groupedPorts.begin(), groupedPorts.end()), groupedPorts.end());
            }
            for (const auto& addressPorts : portsByAddress) {
                InterfaceData interfaceData{};
                interfaceData.endpoint.addressType = addressPorts.first.first;
                interfaceData.endpoint.address = addressPorts.first.second;
                interfaceData.endpoint.ports = addressPorts.second;
                interfaceData.endpoint.position = edge.position;
                interfaceData.endpoint.deviceId = rankInfo.deviceId;
                interfaceData.endpoint.localDieId = getDieId(addressPorts.second);
                interfaceData.protocols = edge.protocols;
                interfaceData.topoInstanceId = edge.topoInstanceId;
                interfaces.push_back(std::move(interfaceData));
            }
            return interfaces;
        }

        for (const auto& addressInfo : levelInfo.rankAddressList) {
            std::vector<std::string> commonPorts = GetCommonPorts(addressInfo.ports, localPorts);
            if (commonPorts.empty()) {
                continue;
            }
            InterfaceData interfaceData{};
            interfaceData.endpoint.addressType = addressInfo.addressType;
            interfaceData.endpoint.address = addressInfo.address;
            interfaceData.endpoint.ports = std::move(commonPorts);
            interfaceData.endpoint.position = edge.position;
            interfaceData.endpoint.deviceId = rankInfo.deviceId;
            interfaceData.endpoint.localDieId = getDieId(interfaceData.endpoint.ports);
            interfaceData.protocols = edge.protocols;
            interfaceData.topoInstanceId = edge.topoInstanceId;
            interfaceData.planeId = addressInfo.planeId;
            interfaces.push_back(std::move(interfaceData));
        }
        return interfaces;
    };

    auto appendLinks = [&links](
                           const InterfaceData& sourceInterface, const InterfaceData& destinationInterface, uint8_t hop,
                           bool requireSamePortCount) {
        if (requireSamePortCount
            && sourceInterface.endpoint.ports.size() != destinationInterface.endpoint.ports.size()) {
            return;
        }
        for (const auto& protocol : sourceInterface.protocols) {
            RankLinkInfo link{};
            link.sourceEndpoint = sourceInterface.endpoint;
            link.destinationEndpoint = destinationInterface.endpoint;
            link.sourceEndpoint.protocol = protocol;
            link.destinationEndpoint.protocol = protocol;
            link.protocol = protocol;
            link.hop = hop;
            links.push_back(std::move(link));
        }
    };

    std::map<std::string, std::vector<InterfaceData>> sourceFabricInterfaces;
    std::map<std::string, std::vector<InterfaceData>> destinationFabricInterfaces;
    std::set<std::string> allowedPlanes;
    for (const auto& addressInfo : currentLevelIt->rankAddressList) {
        allowedPlanes.insert(addressInfo.planeId);
    }

    for (const auto& edge : currentLevelIt->topoEdgeList) {
        if (edge.netLayer != netLayer) {
            continue;
        }

        if (edge.linkType == TopoLinkType::PEER_TO_PEER && netLayer == 0) {
            const std::vector<std::string>* sourcePorts = nullptr;
            const std::vector<std::string>* destinationPorts = nullptr;
            if (edge.localA == sourceRankIt->localId && edge.localB == destinationRankIt->localId) {
                sourcePorts = &edge.localAPorts;
                destinationPorts = &edge.localBPorts;
            } else if (edge.localB == sourceRankIt->localId && edge.localA == destinationRankIt->localId) {
                sourcePorts = &edge.localBPorts;
                destinationPorts = &edge.localAPorts;
            }
            if (sourcePorts != nullptr && destinationPorts != nullptr) {
                auto sourceInterfaces = makeInterfaces(*sourceRankIt, *sourceLevelIt, edge, *sourcePorts);
                auto destinationInterfaces
                    = makeInterfaces(*destinationRankIt, *destinationLevelIt, edge, *destinationPorts);
                for (const auto& sourceInterface : sourceInterfaces) {
                    for (const auto& destinationInterface : destinationInterfaces) {
                        appendLinks(sourceInterface, destinationInterface, 1, false);
                    }
                }
            }
            continue;
        }

        if (edge.linkType != TopoLinkType::PEER_TO_NETWORK) {
            continue;
        }

        auto addFabricInterfaces = [&](const RankInfo& rankInfo, const RankLevelInfo& levelInfo,
                                       std::map<std::string, std::vector<InterfaceData>>& groupedInterfaces) {
            if (edge.localA != rankInfo.localId) {
                return;
            }
            auto interfaces = makeInterfaces(rankInfo, levelInfo, edge, edge.localAPorts);
            for (auto& interfaceData : interfaces) {
                std::string groupKey;
                if (netLayer == 0) {
                    groupKey = std::to_string(interfaceData.topoInstanceId);
                } else {
                    if (allowedPlanes.find(interfaceData.planeId) == allowedPlanes.end()) {
                        continue;
                    }
                    groupKey = interfaceData.planeId;
                }
                groupedInterfaces[groupKey].push_back(std::move(interfaceData));
            }
        };
        addFabricInterfaces(*sourceRankIt, *sourceLevelIt, sourceFabricInterfaces);
        addFabricInterfaces(*destinationRankIt, *destinationLevelIt, destinationFabricInterfaces);
    }

    bool isInnerNetwork = currentLevelIt->netType == "TOPO_FILE_DESC";
    for (const auto& sourceGroup : sourceFabricInterfaces) {
        auto destinationGroupIt = destinationFabricInterfaces.find(sourceGroup.first);
        if (destinationGroupIt == destinationFabricInterfaces.end() || sourceGroup.second.empty()
            || destinationGroupIt->second.empty()) {
            continue;
        }
        if (isInnerNetwork) {
            for (const auto& sourceInterface : sourceGroup.second) {
                for (const auto& destinationInterface : destinationGroupIt->second) {
                    appendLinks(sourceInterface, destinationInterface, 2, true);
                }
            }
        } else {
            appendLinks(sourceGroup.second.back(), destinationGroupIt->second.back(), 2, true);
        }
    }

    return RankTableQueryStatus::SUCCESS;
}

RankTableQueryStatus
RankTable::GetRanksByTopoInst(uint32_t rankId, uint32_t netLayer, uint32_t topoInstanceId, std::vector<uint32_t>& ranks)
{
    ranks.clear();
    if (!EnsureLoaded()) {
        return RankTableQueryStatus::LOAD_FAILED;
    }

    auto rankIt = std::find_if(m_rankList.begin(), m_rankList.end(), [rankId](const RankInfo& rankInfo) {
        return rankInfo.rankId == rankId;
    });
    if (rankIt == m_rankList.end()) {
        HCCL_VM_ERROR("{}: rank {:d} was not found in rank table", __func__, rankId);
        return RankTableQueryStatus::RANK_NOT_FOUND;
    }

    auto levelIt
        = std::find_if(rankIt->levelList.begin(), rankIt->levelList.end(), [netLayer](const RankLevelInfo& levelInfo) {
              return levelInfo.netLayer == netLayer;
          });
    if (levelIt == rankIt->levelList.end()) {
        HCCL_VM_ERROR("{}: rank {:d} does not contain network layer {:d}", __func__, rankId, netLayer);
        return RankTableQueryStatus::LAYER_NOT_FOUND;
    }

    std::vector<uint32_t> candidateRanks;
    const std::string& netInstanceId = levelIt->netInstanceId;
    for (const auto& rankInfo : m_rankList) {
        auto candidateLevelIt = std::find_if(
            rankInfo.levelList.begin(), rankInfo.levelList.end(),
            [netLayer, &netInstanceId](const RankLevelInfo& candidateLevel) {
                return candidateLevel.netLayer == netLayer && candidateLevel.netInstanceId == netInstanceId;
            });
        if (candidateLevelIt != rankInfo.levelList.end()) {
            candidateRanks.push_back(rankInfo.rankId);
        }
    }

    // HComm在单rank网络实例中会建立默认的0号拓扑实例，即使没有物理边。
    if (candidateRanks.size() == 1 && topoInstanceId == 0) {
        ranks = std::move(candidateRanks);
        return RankTableQueryStatus::SUCCESS;
    }

    for (uint32_t candidateRank : candidateRanks) {
        uint32_t endpointNum = 0;
        RankTableQueryStatus status = GetEndpointNum(candidateRank, netLayer, topoInstanceId, endpointNum);
        if (status != RankTableQueryStatus::SUCCESS) {
            return status;
        }
        if (endpointNum > 0) {
            ranks.push_back(candidateRank);
        }
    }

    std::sort(ranks.begin(), ranks.end());
    ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
    if (ranks.empty()) {
        HCCL_VM_ERROR(
            "{}: topology instance {:d} has no rank in network layer {:d} for rank {:d}", __func__, topoInstanceId,
            netLayer, rankId);
        return RankTableQueryStatus::TOPO_INSTANCE_NOT_FOUND;
    }
    return RankTableQueryStatus::SUCCESS;
}

bool RankTable::LoadTopologyIntoRankList()
{
    std::vector<std::string> candidatePaths;
    if (!m_rankTablePath.empty()) {
        candidatePaths.push_back(GetSiblingTopoPath(m_rankTablePath));
    }

    const char* rankTablePath = std::getenv("RANK_TABLE_FILE");
    if (rankTablePath != nullptr && rankTablePath[0] != '\0') {
        std::string topoPath = GetSiblingTopoPath(rankTablePath);
        if (std::find(candidatePaths.begin(), candidatePaths.end(), topoPath) == candidatePaths.end()) {
            candidatePaths.push_back(std::move(topoPath));
        }
    }

    std::ifstream rootInfoFile("/etc/hccl_rootinfo.json");
    if (rootInfoFile.is_open()) {
        try {
            json rootInfo = json::parse(rootInfoFile);
            std::string topoPath = rootInfo.value("topo_file_path", "");
            if (!topoPath.empty()
                && std::find(candidatePaths.begin(), candidatePaths.end(), topoPath) == candidatePaths.end()) {
                candidatePaths.push_back(std::move(topoPath));
            }
        } catch (const json::exception& e) {
            HCCL_VM_WARN("{}: failed to parse /etc/hccl_rootinfo.json: {}", __func__, e.what());
        }
    }

    for (const auto& topoPath : candidatePaths) {
        std::ifstream topoFile(topoPath);
        if (!topoFile.is_open()) {
            continue;
        }

        try {
            json topoJson = json::parse(topoFile);
            if (!topoJson.contains("edge_list") || !topoJson["edge_list"].is_array()) {
                HCCL_VM_ERROR("{}: topology file {} does not contain an edge_list array", __func__, topoPath);
                continue;
            }

            std::map<std::pair<uint32_t, uint32_t>, RankTableTopoType> topoInstanceTypes;
            // (netLayer, topoInstanceId) -> 该实例声明的 local id 集合（用于判定 rank 的实例归属）。
            std::map<std::pair<uint32_t, uint32_t>, std::set<uint32_t>> instanceLocalIds;
            std::vector<TopoEdgeInfo> topoEdgeList;
            bool valid = true;
            for (const auto& edge : topoJson["edge_list"]) {
                if (!edge.is_object() || !edge.contains("net_layer")) {
                    HCCL_VM_ERROR("{}: invalid edge found in topology file {}", __func__, topoPath);
                    valid = false;
                    break;
                }

                TopoEdgeInfo topoEdge{};
                topoEdge.netLayer = edge["net_layer"].get<uint32_t>();
                topoEdge.topoInstanceId = edge.value("topo_instance_id", 0U);
                std::string topoTypeName = edge.value("topo_type", "CLOS");
                if (!ParseTopoType(topoTypeName, topoEdge.topoType) || topoEdge.topoType == RankTableTopoType::CUSTOM) {
                    HCCL_VM_ERROR(
                        "{}: unsupported topo_type '{}' in topology file {}", __func__, topoTypeName, topoPath);
                    valid = false;
                    break;
                }

                std::string linkTypeName = edge.value("link_type", "");
                if (linkTypeName == "PEER2PEER") {
                    topoEdge.linkType = TopoLinkType::PEER_TO_PEER;
                } else if (linkTypeName == "PEER2NET") {
                    topoEdge.linkType = TopoLinkType::PEER_TO_NETWORK;
                } else {
                    HCCL_VM_ERROR(
                        "{}: unsupported link_type '{}' in topology file {}", __func__, linkTypeName, topoPath);
                    valid = false;
                    break;
                }

                if (!edge.contains("local_a") || !edge["local_a"].is_number_unsigned()
                    || !ParseUniqueStringList(edge, "local_a_ports", topoEdge.localAPorts)
                    || !ParseUniqueStringList(edge, "protocols", topoEdge.protocols)) {
                    HCCL_VM_ERROR(
                        "{}: edge has invalid local_a, local_a_ports or protocols in topology file {}", __func__,
                        topoPath);
                    valid = false;
                    break;
                }
                topoEdge.localA = edge["local_a"].get<uint32_t>();
                if (std::any_of(topoEdge.protocols.begin(), topoEdge.protocols.end(), [](const std::string& protocol) {
                        return !IsSupportedProtocol(protocol);
                    })) {
                    HCCL_VM_ERROR("{}: edge has unsupported protocol in topology file {}", __func__, topoPath);
                    valid = false;
                    break;
                }

                if (topoEdge.linkType == TopoLinkType::PEER_TO_PEER) {
                    if (!edge.contains("local_b") || !edge["local_b"].is_number_unsigned()
                        || !ParseUniqueStringList(edge, "local_b_ports", topoEdge.localBPorts)) {
                        HCCL_VM_ERROR(
                            "{}: peer-to-peer edge has invalid local_b or local_b_ports in topology file {}", __func__,
                            topoPath);
                        valid = false;
                        break;
                    }
                    topoEdge.localB = edge["local_b"].get<uint32_t>();
                    if (topoEdge.localA == topoEdge.localB) {
                        HCCL_VM_ERROR(
                            "{}: peer-to-peer edge uses the same local ID at both ends in topology file {}", __func__,
                            topoPath);
                        valid = false;
                        break;
                    }
                }

                std::string positionName = edge.value("position", "DEVICE");
                if (positionName == "DEVICE") {
                    topoEdge.position = TopoAddressPosition::DEVICE;
                } else if (positionName == "HOST") {
                    topoEdge.position = TopoAddressPosition::HOST;
                } else {
                    HCCL_VM_ERROR(
                        "{}: unsupported position '{}' in topology file {}", __func__, positionName, topoPath);
                    valid = false;
                    break;
                }

                const auto instanceKey = std::make_pair(topoEdge.netLayer, topoEdge.topoInstanceId);
                auto insertResult = topoInstanceTypes.emplace(instanceKey, topoEdge.topoType);
                if (!insertResult.second && insertResult.first->second != topoEdge.topoType) {
                    HCCL_VM_ERROR(
                        "{}: topology instance {:d} on layer {:d} has conflicting types in {}", __func__,
                        topoEdge.topoInstanceId, topoEdge.netLayer, topoPath);
                    valid = false;
                    break;
                }
                // 记录实例声明的 local id 集合：P2P 边的两端都属于该实例；P2NET 边只有本端。
                instanceLocalIds[instanceKey].insert(topoEdge.localA);
                if (topoEdge.linkType == TopoLinkType::PEER_TO_PEER) {
                    instanceLocalIds[instanceKey].insert(topoEdge.localB);
                }
                topoEdgeList.push_back(std::move(topoEdge));
            }

            if (!valid) {
                continue;
            }

            size_t edgeAssignmentCount = 0;
            for (auto& rankInfo : m_rankList) {
                for (auto& levelInfo : rankInfo.levelList) {
                    levelInfo.topoEdgeList.clear();

                    std::set<uint32_t> localIds;
                    for (const auto& candidateRank : m_rankList) {
                        auto candidateLevelIt = std::find_if(
                            candidateRank.levelList.begin(), candidateRank.levelList.end(),
                            [&levelInfo](const RankLevelInfo& candidateLevel) {
                                return candidateLevel.netLayer == levelInfo.netLayer
                                       && candidateLevel.netInstanceId == levelInfo.netInstanceId;
                            });
                        if (candidateLevelIt != candidateRank.levelList.end()) {
                            localIds.insert(candidateRank.localId);
                        }
                    }

                    for (const auto& topoEdge : topoEdgeList) {
                        if (topoEdge.netLayer != levelInfo.netLayer
                            || localIds.find(topoEdge.localA) == localIds.end()) {
                            continue;
                        }
                        if (topoEdge.linkType == TopoLinkType::PEER_TO_PEER
                            && localIds.find(topoEdge.localB) == localIds.end()) {
                            continue;
                        }
                        levelInfo.topoEdgeList.push_back(topoEdge);
                        ++edgeAssignmentCount;
                    }

                    // 拓扑实例归属：依据 topo.json 实例声明的 local id 集合是否包含本 rank 的 localId，
                    // 而不是"本 rank 是否在该实例内有边"——单 rank 网络实例没有物理边，但仍是合法实例
                    // （HComm 记为 0 号实例）。按 netLayer 限定，避免跨层串扰。
                    levelInfo.topoInstanceIds.clear();
                    for (const auto& instance : instanceLocalIds) {
                        if (instance.first.first != levelInfo.netLayer) {
                            continue;
                        }
                        if (instance.second.count(rankInfo.localId) != 0U) {
                            levelInfo.topoInstanceIds.push_back(instance.first.second);
                        }
                    }
                    std::sort(levelInfo.topoInstanceIds.begin(), levelInfo.topoInstanceIds.end());
                    levelInfo.topoInstanceIds.erase(
                        std::unique(levelInfo.topoInstanceIds.begin(), levelInfo.topoInstanceIds.end()),
                        levelInfo.topoInstanceIds.end());
                }
            }

            HCCL_VM_INFO(
                "{}: topology file loaded into rank list, path={}, "
                "topology instances={:d}, edges={:d}, assignments={:d}",
                __func__, topoPath, topoInstanceTypes.size(), topoEdgeList.size(), edgeAssignmentCount);
            m_topoInstanceTypes = std::move(topoInstanceTypes);
            return true;
        } catch (const json::exception& e) {
            HCCL_VM_ERROR("{}: failed to parse topology file {}: {}", __func__, topoPath, e.what());
        }
    }

    HCCL_VM_ERROR("{}: failed to load topology file for topology instance query", __func__);
    return false;
}

uint32_t RankTable::GetRankSize() const { return m_rankSize; }

int RankTable::GetDeviceId(uint32_t rank) const
{
    if (rank >= m_rankList.size()) {
        return -1;
    }
    return static_cast<int>(m_rankList[rank].deviceId);
}

int RankTable::GetRankIdByDeviceId(uint32_t deviceId) const
{
    for (const auto& r : m_rankList) {
        if (r.deviceId == deviceId) {
            return static_cast<int>(r.rankId);
        }
    }
    return -1;
}

RankTableQueryStatus RankTable::GetRankInfo(uint32_t rank, RankInfo& info) const
{
    if (rank >= m_rankList.size()) {
        return RankTableQueryStatus::RANK_NOT_FOUND;
    }
    info = m_rankList[rank];
    return RankTableQueryStatus::SUCCESS;
}

void RankTable::Reset()
{
    m_loaded = false;
    m_topologyLoaded = false;
    m_rankSize = 0;
    m_rankList.clear();
    m_rankTablePath.clear();
    m_topoInstanceTypes.clear();
}

bool RankTable::ParseFromJsonString(const std::string& jsonStr)
{
    json rankTable;
    try {
        rankTable = json::parse(jsonStr);
        if (!rankTable.is_object()) {
            HCCL_VM_ERROR("{}: rank table root is not a JSON object", __func__);
            return false;
        }

        uint32_t rankSize = 0;
        if (rankTable.contains("rank_count")) {
            rankSize = rankTable["rank_count"].get<uint32_t>();
        } else if (rankTable.contains("rank_list")) {
            rankSize = static_cast<uint32_t>(rankTable["rank_list"].size());
        } else {
            HCCL_VM_ERROR("{}: rank table missing both 'rank_count' and 'rank_list'", __func__);
            return false;
        }

        std::vector<RankInfo> rankList;
        if (rankTable.contains("rank_list") && rankTable["rank_list"].is_array()) {
            rankList.reserve(rankSize);
            for (uint32_t i = 0; i < rankTable["rank_list"].size(); i++) {
                const auto& entry = rankTable["rank_list"][i];
                RankInfo info{};
                info.rankId = entry.contains("rank_id") ? entry["rank_id"].get<uint32_t>() : i;
                info.deviceId = entry.contains("device_id") ? entry["device_id"].get<uint32_t>() : 0;
                info.localId = entry.contains("local_id") ? entry["local_id"].get<uint32_t>() : 0;

                if (entry.contains("level_list")) {
                    if (!entry["level_list"].is_array()) {
                        HCCL_VM_ERROR("{}: level_list of rank {:d} is not an array", __func__, info.rankId);
                        return false;
                    }

                    for (const auto& levelEntry : entry["level_list"]) {
                        if (!levelEntry.is_object() || !levelEntry.contains("net_layer")) {
                            HCCL_VM_ERROR("{}: invalid level_list entry found for rank {:d}", __func__, info.rankId);
                            return false;
                        }

                        RankLevelInfo levelInfo{};
                        levelInfo.netLayer = levelEntry["net_layer"].get<uint32_t>();
                        levelInfo.netInstanceId = levelEntry.value("net_instance_id", "");
                        levelInfo.netType = levelEntry.value("net_type", "");
                        levelInfo.netAttr = levelEntry.value("net_attr", "");

                        if (levelEntry.contains("rank_addr_list")) {
                            if (!levelEntry["rank_addr_list"].is_array()) {
                                HCCL_VM_ERROR(
                                    "{}: rank_addr_list of rank {:d} on layer {:d} is not an array", __func__,
                                    info.rankId, levelInfo.netLayer);
                                return false;
                            }
                            for (const auto& addressEntry : levelEntry["rank_addr_list"]) {
                                if (!addressEntry.is_object() || !addressEntry.contains("addr")
                                    || !addressEntry["addr"].is_string() || !addressEntry.contains("addr_type")
                                    || !addressEntry["addr_type"].is_string()) {
                                    HCCL_VM_ERROR(
                                        "{}: invalid rank address found for rank {:d} on layer {:d}", __func__,
                                        info.rankId, levelInfo.netLayer);
                                    return false;
                                }

                                RankAddressInfo addressInfo{};
                                addressInfo.address = addressEntry["addr"].get<std::string>();
                                addressInfo.addressType = addressEntry["addr_type"].get<std::string>();
                                addressInfo.planeId = addressEntry.value("plane_id", "0");
                                if (addressInfo.address.empty()
                                    || !ParseUniqueStringList(addressEntry, "ports", addressInfo.ports)) {
                                    HCCL_VM_ERROR(
                                        "{}: rank address has empty addr or invalid ports for rank {:d} on layer {:d}",
                                        __func__, info.rankId, levelInfo.netLayer);
                                    return false;
                                }
                                levelInfo.rankAddressList.push_back(std::move(addressInfo));
                            }
                        }
                        info.levelList.push_back(std::move(levelInfo));
                    }
                }

                rankList.push_back(std::move(info));
            }
        }

        m_rankSize = rankSize;
        m_rankList = std::move(rankList);
    } catch (const json::exception& e) {
        HCCL_VM_ERROR("{}: failed to parse rank table json: {}", __func__, e.what());
        return false;
    }

    HCCL_VM_INFO(
        "{}: rank table loaded, rankSize={:d}, rankList entries={:d}", __func__, m_rankSize, m_rankList.size());
    return true;
}

} // namespace sim

EngineCtxRegistry& EngineCtxRegistry::Instance()
{
    static EngineCtxRegistry inst;
    return inst;
}

HcclResult EngineCtxRegistry::Create(uint64_t commId, const char* ctxTag, CommEngine engine, uint64_t size, void** ctx)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    void* devAddr = nullptr;
    if (engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CPU_TS || engine == COMM_ENGINE_CCU) {
        devAddr = calloc(1, static_cast<size_t>(size));
        if (devAddr == nullptr) {
            HCCL_VM_ERROR("{}: malloc failed, size={:d}", __func__, size);
            return HCCL_E_MEMORY;
        }
    } else if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_AIV) {
        aclError aclRet = aclrtMalloc(&devAddr, static_cast<size_t>(size), ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS || devAddr == nullptr) {
            HCCL_VM_ERROR("{}: aclrtMalloc failed, ret={:d}, size={:d}", __func__, static_cast<int>(aclRet), size);
            return HCCL_E_MEMORY;
        }
    } else {
        HCCL_VM_ERROR("{}: unsupported engine={:d}", __func__, static_cast<int>(engine));
        return HCCL_E_PARA;
    }

    uint64_t addr = reinterpret_cast<uint64_t>(devAddr);
    sim::runtime::HcclEngineCtx entry{};
    entry.id = 0;
    entry.commId = commId;
    entry.ctxTag = HashCtxTag(ctxTag);
    entry.engine = static_cast<uint64_t>(engine);
    entry.size = size;
    entry.addr = addr;

    m_map[addr] = entry;
    *ctx = devAddr;
    return HCCL_SUCCESS;
}

bool EngineCtxRegistry::Get(uint64_t commId, uint64_t ctxTag, uint64_t engineVal, void** ctx, uint64_t* size)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& kv : m_map) {
        auto& entry = kv.second;
        if (entry.commId == commId && entry.ctxTag == ctxTag && entry.engine == engineVal) {
            *ctx = reinterpret_cast<void*>(entry.addr);
            *size = entry.size;
            return true;
        }
    }
    return false;
}

HcclResult EngineCtxRegistry::Copy(
    uint64_t commId, uint64_t ctxTag, uint64_t engineVal, const void* srcCtx, uint64_t size, uint64_t dstCtxOffset)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& kv : m_map) {
        auto& entry = kv.second;
        if (entry.commId == commId && entry.ctxTag == ctxTag && entry.engine == engineVal) {
            if (dstCtxOffset + size > entry.size) {
                return HCCL_E_PARA;
            }
            void* dstPtr = reinterpret_cast<void*>(entry.addr + dstCtxOffset);
            CommEngine engine = static_cast<CommEngine>(engineVal);
            if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_AIV) {
                aclError aclRet = aclrtMemcpy(
                    dstPtr, static_cast<size_t>(size), srcCtx, static_cast<size_t>(size), ACL_MEMCPY_HOST_TO_DEVICE);
                if (aclRet != ACL_SUCCESS) {
                    HCCL_VM_ERROR("{}: aclrtMemcpy H2D failed, ret={:d}", __func__, static_cast<int>(aclRet));
                    return HCCL_E_INTERNAL;
                }
            } else {
                std::memcpy(dstPtr, srcCtx, static_cast<size_t>(size));
            }
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_NOT_FOUND;
}

uint32_t EngineCtxRegistry::Destroy(uint64_t commId, uint64_t ctxTag, uint64_t engineVal)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    HCCL_VM_INFO("{}: ctxTag={:d}", __func__, ctxTag);
    uint32_t removed = 0;
    CommEngine engine = static_cast<CommEngine>(engineVal);
    bool isDeviceMem = (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_AIV);
    for (auto it = m_map.begin(); it != m_map.end();) {
        auto& entry = it->second;
        if (entry.commId == commId && entry.ctxTag == ctxTag && entry.engine == engineVal) {
            void* devAddr = reinterpret_cast<void*>(entry.addr);
            if (devAddr != nullptr) {
                if (isDeviceMem) {
                    aclrtFree(devAddr);
                } else {
                    free(devAddr);
                }
            }
            it = m_map.erase(it);
            ++removed;
        } else {
            ++it;
        }
    }
    return removed;
}

void EngineCtxRegistry::DestroyByCommId(uint64_t commId)
{
    HCCL_VM_INFO("{}: commId={:d}", __func__, commId);
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto it = m_map.begin(); it != m_map.end();) {
        if (it->second.commId == commId) {
            void* devAddr = reinterpret_cast<void*>(it->second.addr);
            if (devAddr != nullptr) {
                CommEngine engine = static_cast<CommEngine>(it->second.engine);
                if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_AIV) {
                    aclrtFree(devAddr);
                } else {
                    free(devAddr);
                }
            }
            it = m_map.erase(it);
        } else {
            ++it;
        }
    }
}

void EngineCtxRegistry::ResetAll()
{
    HCCL_VM_INFO("{}", __func__);
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& kv : m_map) {
        void* devAddr = reinterpret_cast<void*>(kv.second.addr);
        if (devAddr != nullptr) {
            CommEngine engine = static_cast<CommEngine>(kv.second.engine);
            if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_AIV) {
                aclrtFree(devAddr);
            } else {
                free(devAddr);
            }
        }
    }
    m_map.clear();
}

extern "C" void HcclEngineCtxResetAll() { EngineCtxRegistry::Instance().ResetAll(); }

void EngineCtxDestroyByCommId(uint64_t commId) { EngineCtxRegistry::Instance().DestroyByCommId(commId); }
