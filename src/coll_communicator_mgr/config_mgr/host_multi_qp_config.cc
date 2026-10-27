/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_multi_qp_config.h"

#include <cstddef>
#include <fstream>
#include <istream>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>

#include "config/base_config_pub.h"
#include "log.h"

namespace hccl {

constexpr char HOST_MULTI_QP_CONFIG_PATH[] = "/etc/hcomm.cfg";
constexpr char HOST_RDMA_UDP_PORTS_LIST_ENV[] = "HCCL_HOST_RDMA_UDP_PORTS_LIST";
constexpr uint32_t MULTI_QP_COUNT_MIN = 1U;
constexpr uint32_t MULTI_QP_COUNT_MAX = 32U;
constexpr uint32_t UDP_PORT_MIN = 1U;
constexpr uint32_t UDP_PORT_MAX = 65535U;
constexpr uint32_t UDP_PORT_COUNT_MAX = 32U;
constexpr std::size_t HOST_RDMA_UDP_PORTS_LIST_LEN_MAX = 32U * 1024U;

struct HostMultiQpRawConfig {
    std::unordered_map<uint32_t, std::string> modesByPhyId;
    std::unordered_map<uint32_t, std::string> qpCountsByPhyId;
    std::unordered_map<uint32_t, std::string> udpPortsByPhyId;
    std::unordered_set<uint32_t> configuredPhyIds;
};

static bool ParseStrictDecimal(const std::string& value, uint32_t minValue, uint32_t maxValue, uint32_t& parsed)
{
    bool parseOk = false;
    const uint32_t number = hcomm::StrToNum<uint32_t>(value, parseOk);
    if (!parseOk || number < minValue || number > maxValue) {
        return false;
    }
    parsed = number;
    return true;
}

static bool ParsePortsList(const std::string& value, std::vector<uint16_t>& ports)
{
    if (value.empty() || value.back() == ',') {
        HCCL_ERROR("[%s] UDP source port list is empty or ends with a comma.", __func__);
        return false;
    }

    std::vector<uint16_t> parsedPorts;
    size_t start = 0;
    while (true) {
        const size_t comma = value.find(',', start);
        const std::string token = value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        uint32_t port = 0;
        if (!ParseStrictDecimal(token, UDP_PORT_MIN, UDP_PORT_MAX, port)) {
            HCCL_ERROR(
                "[%s] UDP source port is invalid, expected a decimal integer in range [%u, %u].", __func__,
                UDP_PORT_MIN, UDP_PORT_MAX);
            return false;
        }
        parsedPorts.emplace_back(static_cast<uint16_t>(port));
        if (parsedPorts.size() > UDP_PORT_COUNT_MAX) {
            HCCL_ERROR(
                "[%s] UDP source port count[%zu] exceeds the maximum[%u].", __func__, parsedPorts.size(),
                UDP_PORT_COUNT_MAX);
            return false;
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1U;
    }
    ports = std::move(parsedPorts);
    return true;
}

static bool ParseConfigFileKey(const std::string& key, const std::string& prefix, uint32_t& devicePhyId)
{
    if (key.size() <= prefix.size() || key.compare(0, prefix.size(), prefix) != 0) {
        return false;
    }
    const std::string deviceSuffix = key.substr(prefix.size());
    if (deviceSuffix.size() > 1U && deviceSuffix.front() == '0') {
        return false;
    }
    return ParseStrictDecimal(deviceSuffix, 0, std::numeric_limits<uint32_t>::max(), devicePhyId);
}

static std::string TrimConfigFileField(const std::string& field)
{
    const size_t first = field.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const size_t last = field.find_last_not_of(" \t\r\n");
    return field.substr(first, last - first + 1U);
}

static HcclResult CollectConfigFileFields(std::istream& inFile, HostMultiQpRawConfig& rawConfig)
{
    std::string line;
    while (std::getline(inFile, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        const size_t equalPos = line.find('=');
        if (equalPos == std::string::npos) {
            continue;
        }

        const std::string key = TrimConfigFileField(line.substr(0, equalPos));
        const std::string value = TrimConfigFileField(line.substr(equalPos + 1U));
        std::string prefix;
        if (key.find("udp_port_mode_") == 0U) {
            prefix = "udp_port_mode_";
        } else if (key.find("multi_qp_count_") == 0U) {
            prefix = "multi_qp_count_";
        } else if (key.find("multi_qp_udp_ports_") == 0U) {
            prefix = "multi_qp_udp_ports_";
        } else {
            continue;
        }

        uint32_t devicePhyId = 0;
        if (!ParseConfigFileKey(key, prefix, devicePhyId)) {
            HCCL_ERROR(
                "[%s] invalid physical device ID suffix in host multi qp config key[%s].", __func__, key.c_str());
            return HCCL_E_PARA;
        }
        auto& fields = prefix == "udp_port_mode_"  ? rawConfig.modesByPhyId :
                       prefix == "multi_qp_count_" ? rawConfig.qpCountsByPhyId :
                                                     rawConfig.udpPortsByPhyId;
        const bool isDuplicateConfigKey = !fields.emplace(devicePhyId, value).second;
        rawConfig.configuredPhyIds.emplace(devicePhyId);
        if (isDuplicateConfigKey) {
            HCCL_WARNING(
                "[%s] duplicated host multi qp config key[%s], phyId[%u], keep the first value.", __func__, key.c_str(),
                devicePhyId);
        }
    }
    if (inFile.bad()) {
        HCCL_ERROR("[%s] read host multi qp config failed.", __func__);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

static HcclResult ParseDeviceConfigFromFile(
    uint32_t devicePhyId, const HostMultiQpRawConfig& rawConfig,
    std::unordered_map<uint32_t, HostMultiQpDeviceConfig>& configs)
{
    const auto modeIter = rawConfig.modesByPhyId.find(devicePhyId);
    const auto qpCountIter = rawConfig.qpCountsByPhyId.find(devicePhyId);
    const auto udpPortsIter = rawConfig.udpPortsByPhyId.find(devicePhyId);
    if (modeIter == rawConfig.modesByPhyId.end() || qpCountIter == rawConfig.qpCountsByPhyId.end()
        || udpPortsIter == rawConfig.udpPortsByPhyId.end()) {
        HCCL_ERROR("[%s] incomplete host multi qp config, phyId[%u].", __func__, devicePhyId);
        return HCCL_E_PARA;
    }
    if (modeIter->second != "multi_qp") {
        HCCL_ERROR("[%s] invalid udp_port_mode[%s], phyId[%u].", __func__, modeIter->second.c_str(), devicePhyId);
        return HCCL_E_PARA;
    }

    HostMultiQpDeviceConfig deviceConfig;
    if (!ParseStrictDecimal(qpCountIter->second, MULTI_QP_COUNT_MIN, MULTI_QP_COUNT_MAX, deviceConfig.qpCount)
        || !ParsePortsList(udpPortsIter->second, deviceConfig.udpPorts)) {
        HCCL_ERROR("[%s] invalid host multi qp count or ports, phyId[%u].", __func__, devicePhyId);
        return HCCL_E_PARA;
    }

    HCCL_RUN_INFO(
        "[%s] host multi qp config applied, phyId[%u] qpCount[%u] portCount[%zu].", __func__, devicePhyId,
        deviceConfig.qpCount, deviceConfig.udpPorts.size());
    configs[devicePhyId] = std::move(deviceConfig);
    return HCCL_SUCCESS;
}

static HcclResult
ParseConfigFileContent(std::istream& inFile, std::unordered_map<uint32_t, HostMultiQpDeviceConfig>& configs)
{
    HostMultiQpRawConfig rawConfig;
    CHK_RET(CollectConfigFileFields(inFile, rawConfig));
    for (const uint32_t devicePhyId : rawConfig.configuredPhyIds) {
        CHK_RET(ParseDeviceConfigFromFile(devicePhyId, rawConfig, configs));
    }
    return HCCL_SUCCESS;
}

static bool
ParseEnvUdpPortsEntry(const std::string& entry, std::unordered_map<uint32_t, HostMultiQpDeviceConfig>& configs)
{
    const size_t colon = entry.find(':');
    if (entry.empty() || colon == std::string::npos || colon == 0U || colon + 1U >= entry.size()
        || entry.find(':', colon + 1U) != std::string::npos) {
        HCCL_ERROR("[%s] device config has invalid separators or an empty field.", __func__);
        return false;
    }

    uint32_t devicePhyId = 0;
    if (!ParseStrictDecimal(entry.substr(0, colon), 0U, std::numeric_limits<uint32_t>::max(), devicePhyId)) {
        HCCL_ERROR("[%s] phy_dev_id is not a valid decimal integer.", __func__);
        return false;
    }
    std::vector<uint16_t> ports;
    if (!ParsePortsList(entry.substr(colon + 1U), ports)) {
        HCCL_ERROR("[%s] UDP source port list is invalid.", __func__);
        return false;
    }
    HostMultiQpDeviceConfig config;
    config.udpPorts = std::move(ports);
    if (!configs.emplace(devicePhyId, std::move(config)).second) {
        HCCL_ERROR("[%s] phy_dev_id[%u] is duplicated.", __func__, devicePhyId);
        return false;
    }
    return true;
}

static bool
ParseEnvUdpPortsList(const std::string& value, std::unordered_map<uint32_t, HostMultiQpDeviceConfig>& configs)
{
    configs.clear();
    if (value.empty()) {
        return true;
    }
    if (value.size() > HOST_RDMA_UDP_PORTS_LIST_LEN_MAX) {
        HCCL_ERROR(
            "[%s] HCCL_HOST_RDMA_UDP_PORTS_LIST length[%zu] exceeds the maximum[%zu].", __func__, value.size(),
            HOST_RDMA_UDP_PORTS_LIST_LEN_MAX);
        return false;
    }

    std::unordered_map<uint32_t, HostMultiQpDeviceConfig> parsedConfigs;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t semicolon = value.find(';', start);
        const std::string entry
            = value.substr(start, semicolon == std::string::npos ? std::string::npos : semicolon - start);
        if (!ParseEnvUdpPortsEntry(entry, parsedConfigs)) {
            return false;
        }
        if (semicolon == std::string::npos) {
            break;
        }
        start = semicolon + 1U;
    }
    configs = std::move(parsedConfigs);
    return true;
}

HcclResult HostMultiQpConfig::Parse()
{
    std::lock_guard<std::mutex> lock(parseMutex_);
    if (parsed_) {
        return HCCL_SUCCESS;
    }

    std::unordered_map<uint32_t, HostMultiQpDeviceConfig> configs;
    const std::string envValue = hcomm::GetEnv(HOST_RDMA_UDP_PORTS_LIST_ENV);
    if (!ParseEnvUdpPortsList(envValue, configs)) {
        HCCL_ERROR("[%s] parse HCCL_HOST_RDMA_UDP_PORTS_LIST failed.", __func__);
        return HCCL_E_PARA;
    }
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_HOST_RDMA_UDP_PORTS_LIST set by %s, device config count[%zu]",
        envValue.empty() ? "default" : "environment", configs.size());

    std::ifstream inFile;
    inFile.open(HOST_MULTI_QP_CONFIG_PATH, std::ifstream::in);
    if (!inFile) {
        HCCL_INFO("[%s] host multi qp config file is unavailable.", __func__);
    } else {
        HCCL_INFO("[%s] open host multi qp config file success.", __func__);
        CHK_RET(ParseConfigFileContent(inFile, configs));
    }
    configs_ = std::move(configs);
    parsed_ = true;
    return HCCL_SUCCESS;
}

uint32_t HostMultiQpConfig::GetQpCount(uint32_t devicePhyId) const
{
    const auto configIter = configs_.find(devicePhyId);
    return configIter == configs_.end() ? 0U : configIter->second.qpCount;
}

const std::vector<uint16_t>* HostMultiQpConfig::GetUdpPorts(uint32_t devicePhyId) const
{
    const auto configIter = configs_.find(devicePhyId);
    return configIter == configs_.end() ? nullptr : &configIter->second.udpPorts;
}

} // namespace hccl
