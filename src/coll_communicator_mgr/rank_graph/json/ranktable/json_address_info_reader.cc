/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_table_reader.h"

#include <string>
#include <vector>
#include "json/common/json_parse_util.h"
#include "json/ranktable/rank_table_keys.h"
#include "adapter_error_manager_pub.h"
#include "common/hccl_log_keywords_pub.h"
#include "rank_table_report_macro.h"

namespace Hccl {

namespace {
    // 日志打印截断长度，与 JSON schema/解析参数无关，保留在 .cc。
    constexpr int MAX_DISPLAY_LEN = 128;

    HcclResult IPV4ToAddr(const std::string& address, IpAddress& out)
    {
        if (!IpAddress::IsIPv4(address)) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({address, kRankTableKeys[RankTableKey::kAddr].name, "A valid IPv4 address"}));
            return HCCL_E_PARA;
        }
        out = IpAddress(address, AF_INET);
        return HCCL_SUCCESS;
    }

    HcclResult IPV6ToAddr(const std::string& address, IpAddress& out)
    {
        if (!IpAddress::IsIPv6(address)) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({address, kRankTableKeys[RankTableKey::kAddr].name, "A valid IPv6 address"}));
            return HCCL_E_PARA;
        }
        out = IpAddress(address, AF_INET6);
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult RankTableReader::ReadAddress(const nlohmann::json& addressInfoJson, AddressInfo& out) const
{
    out = AddressInfo{};
    CHK_RET(ReadAddrTypeAndAddr(addressInfoJson, out));
    out.planeId = addressInfoJson.value<std::string>(kRankTableKeys[RankTableKey::kPlaneId].name, "0");
    if (out.planeId.length() > MAX_VALUE_PLANEID) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {out.planeId, kRankTableKeys[RankTableKey::kPlaneId].name,
                 "length [0, " + std::to_string(MAX_VALUE_PLANEID) + "]"}));
        return HCCL_E_PARA;
    }
    return ReadPorts(addressInfoJson, out);
}

HcclResult RankTableReader::ReadAddrTypeAndAddr(const nlohmann::json& addressInfoJson, AddressInfo& out) const
{
    std::string addrTypeStr;
    HcclResult ret = GetJsonProperty(addressInfoJson, kRankTableKeys[RankTableKey::kAddrType].name, true, addrTypeStr);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(
            addressInfoJson, kRankTableKeys[RankTableKey::kAddrType].name, "non-empty string", source_);
        return ret;
    }
    if (kStrToAddrType.count(addrTypeStr) == 0) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({addrTypeStr, kRankTableKeys[RankTableKey::kAddrType].name, "EID/IPV4/IPV6"}));
        return HCCL_E_PARA;
    }
    out.addrType = kStrToAddrType.at(addrTypeStr);

    std::string address;
    ret = GetJsonProperty(addressInfoJson, kRankTableKeys[RankTableKey::kAddr].name, true, address);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(addressInfoJson, kRankTableKeys[RankTableKey::kAddr].name, "non-empty string", source_);
        return ret;
    }
    if (address.length() < MIN_VALUE_ADDR_LENGRH || address.length() > MAX_VALUE_ADDR_LENGRH) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({address, kRankTableKeys[RankTableKey::kAddr].name, "A ip address"}));
        HCCL_ERROR(
            "[%s] addr [%.*s] length is out of range [%u] to [%u]", __func__, MAX_DISPLAY_LEN, address.c_str(),
            MIN_VALUE_ADDR_LENGRH, MAX_VALUE_ADDR_LENGRH);
        return HCCL_E_PARA;
    }
    if (addrTypeStr == "IPV4") {
        CHK_RET(IPV4ToAddr(address, out.addr));
    } else if (addrTypeStr == "IPV6") {
        CHK_RET(IPV6ToAddr(address, out.addr));
    } else {
        CHK_RET(EidToAddr(address, out.addr));
    }
    return ReadBackupAddrs(addressInfoJson, out, addrTypeStr);
}

HcclResult RankTableReader::ReadPorts(const nlohmann::json& addressInfoJson, AddressInfo& out) const
{
    const nlohmann::json* portsJsons = nullptr;
    HcclResult ret = GetJsonPropertyListRef(addressInfoJson, kRankTableKeys[RankTableKey::kPorts].name, portsJsons);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(addressInfoJson, kRankTableKeys[RankTableKey::kPorts].name, "array", source_);
        return ret;
    }
    for (const auto& portsJson : *portsJsons) {
        const std::string portStr = portsJson.get<std::string>();
        if (portStr.size() < MIN_VALUE_PORT_LENGTH || portStr.size() > MAX_VALUE_PORT_LENGTH) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({portStr, "port", "length"}));
            return HCCL_E_PARA;
        }
        out.ports.emplace(portStr);
    }
    if (out.ports.size() < MIN_VALUE_PORT || out.ports.size() > MAX_VALUE_PORT) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({std::to_string(out.ports.size()), "ports count", "[1, 20]"}));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult RankTableReader::ReadBackupAddrs(
    const nlohmann::json& addressInfoJson, AddressInfo& out, const std::string& addrTypeStr) const
{
    out.backupAddrs.clear();
    const auto backupIt = addressInfoJson.find(kRankTableKeys[RankTableKey::kBackupAddr].name);
    if (backupIt == addressInfoJson.end()) {
        HCCL_WARNING("[%s] backup_addr is not configured.", __func__);
        return HCCL_SUCCESS;
    }
    return ParseBackupAddrs(*backupIt, addrTypeStr, out.backupAddrs);
}

HcclResult
RankTableReader::ParseAddrByType(const std::string& addrType, const std::string& address, IpAddress& ipAddress)
{
    return ParseAddrByType(addrType, address, ipAddress, kRankTableKeys[RankTableKey::kBackupAddr].name);
}

HcclResult RankTableReader::ParseAddrByType(
    const std::string& addrType, const std::string& address, IpAddress& ipAddress, const std::string& addrField)
{
    if (addrType == "IPV4") {
        if (!IpAddress::IsIPv4(address)) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({address, addrField, "A valid IPv4 address"}));
            return HCCL_E_PARA;
        }
        ipAddress = IpAddress(address, AF_INET);
    } else if (addrType == "IPV6") {
        if (!IpAddress::IsIPv6(address)) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({address, addrField, "A valid IPv6 address"}));
            return HCCL_E_PARA;
        }
        ipAddress = IpAddress(address, AF_INET6);
    } else {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {addrType, kRankTableKeys[RankTableKey::kAddrType].name, "IPV4 or IPV6 for " + addrField}));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult RankTableReader::ParseBackupAddrs(
    const nlohmann::json& backupAddrJson, const std::string& addrType, std::vector<IpAddress>& backupAddrs)
{
    backupAddrs.clear();
    if (!backupAddrJson.is_array()) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({"non-array", kRankTableKeys[RankTableKey::kBackupAddr].name, "array"}));
        return HCCL_E_PARA;
    }
    if (backupAddrJson.size() > MAX_VALUE_BACKUP_ADDR_SIZE) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(backupAddrJson.size()), "backup_addr size",
                 "[0, " + std::to_string(MAX_VALUE_BACKUP_ADDR_SIZE) + "]"}));
        return HCCL_E_PARA;
    }
    for (const auto& backupAddr : backupAddrJson) {
        if (!backupAddr.is_string()) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({"non-string", "backup_addr element", "string"}));
            return HCCL_E_PARA;
        }
        const std::string backupAddrStr = backupAddr.get<std::string>();
        if (backupAddrStr.length() < MIN_VALUE_ADDR_LENGRH || backupAddrStr.length() > MAX_VALUE_ADDR_LENGRH) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>(
                    {backupAddrStr, kRankTableKeys[RankTableKey::kBackupAddr].name, "invalid length"}));
            return HCCL_E_PARA;
        }
        IpAddress backupIpAddress;
        CHK_RET(ParseAddrByType(addrType, backupAddrStr, backupIpAddress));
        backupAddrs.emplace_back(std::move(backupIpAddress));
    }
    return HCCL_SUCCESS;
}

} // namespace Hccl
