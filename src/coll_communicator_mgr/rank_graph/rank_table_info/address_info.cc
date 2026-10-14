/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "address_info.h"

#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include "json_parser.h"
#include "invalid_params_exception.h"
#include "exception_util.h"
#include "adapter_error_manager_pub.h"
#include "rank_table_report_macro.h"
#include "hccl_log_keywords.h"

namespace Hccl {
using namespace std;

const unordered_map<string, AddrType> AddressInfo::strToAddrType
    = (unordered_map<string, AddrType>{{"EID", AddrType::EID}, {"IPV4", AddrType::IPV4}, {"IPV6", AddrType::IPV6}});

HcclResult AddressInfo::DeserializeAddrTypeAndAddr(const nlohmann::json& addressInfoJson, RankTableSource source)
{
    std::string addrTypeStr;
    std::string msgAddrtype = "error occurs when parser object of propName \"addr_type\"";
    TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgAddrtype, (addrTypeStr = GetJsonProperty(addressInfoJson, "addr_type")),
        addressInfoJson, "addr_type", "non-empty string", source));

    if (!IsStringInAddrType(addrTypeStr)) {
        HCCL_ERROR("[%s] failed with Invalid addrType.", __func__);
        return HCCL_E_PARA;
    }
    addrType = strToAddrType.at(addrTypeStr);

    std::string address;
    std::string msgAddr = "error occurs when parser object of propName \"addr\"";
    TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgAddr, (address = GetJsonProperty(addressInfoJson, "addr")), addressInfoJson, "addr",
        "non-empty string", source));

    if (address.length() < MIN_VALUE_ADDR_LENGRH || address.length() > MAX_VALUE_ADDR_LENGRH) {
        if (address.empty()) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({"N/A", "addr", "addr is required and should not be empty"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] addr is required and should not be empty", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA));
        } else {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({address, "addr", "A ip address"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] addr[%s] length is out of range [%u] to [%u]",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), address.c_str(), MIN_VALUE_ADDR_LENGRH,
                MAX_VALUE_ADDR_LENGRH);
        }
        HCCL_ERROR(
            "[%s] addr [%.*s] length is out of range [%u] to [%u]", __func__, MAX_DISPLAY_LEN, address.c_str(),
            MIN_VALUE_ADDR_LENGRH, MAX_VALUE_ADDR_LENGRH);
        return HCCL_E_PARA;
    }

    HCCL_INFO("[AddressInfo::%s] addrTypeStr is[%s]", __func__, addrTypeStr.c_str());
    if (addrTypeStr == "IPV4") {
        CHK_RET(IPV4ToAddr(address));
    } else if (addrTypeStr == "IPV6") {
        CHK_RET(IPV6ToAddr(address));
    } else if (addrTypeStr == "EID") {
        CHK_RET(EidToAddr(address));
    }
    // 先解析主地址以确定 addrType，再使用同一类型校验可选的 backup_addr。
    CHK_RET(DeserializeBackupAddrs(addressInfoJson, addrTypeStr));
    return HCCL_SUCCESS;
}

HcclResult AddressInfo::DeserializePorts(const nlohmann::json& addressInfoJson, RankTableSource source)
{
    nlohmann::json portsJsons;
    std::string msgPortlist = "error occurs when parser object of propName \"ports\"";
    TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgPortlist, (GetJsonPropertyList(addressInfoJson, "ports", portsJsons)),
        addressInfoJson, "ports", "array", source));

    for (auto& portsJson : portsJsons) {
        if (portsJson.get<std::string>().size() < MIN_VALUE_PORT_LENGTH
            || portsJson.get<std::string>().size() > MAX_VALUE_PORT_LENGTH) {
            HCCL_ERROR(
                "[%s] portsString [%u] length is out of range [%u] to [%u]", __func__,
                portsJson.get<std::string>().size(), MIN_VALUE_PORT_LENGTH, MAX_VALUE_PORT_LENGTH);
            return HCCL_E_PARA;
        }
        ports.emplace(portsJson);
    }
    if (ports.size() < MIN_VALUE_PORT || ports.size() > MAX_VALUE_PORT) {
        HCCL_ERROR(
            "[%s] ports [%u] length is out of range [%u] to [%u]", __func__, ports.size(), MIN_VALUE_PORT,
            MAX_VALUE_PORT);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult AddressInfo::Deserialize(const nlohmann::json& addressInfoJson, RankTableSource source)
{
    CHK_RET(DeserializeAddrTypeAndAddr(addressInfoJson, source));

    planeId = addressInfoJson.value<std::string>("plane_id", "0");
    if (planeId.length() > MAX_VALUE_PLANEID) {
        HCCL_ERROR(
            "[%s] plane_id [%s] length is out of range [%u] to [%u]", __func__, planeId.c_str(), MIN_VALUE_PLANEID,
            MAX_VALUE_PLANEID);
        return HCCL_E_PARA;
    }

    CHK_RET(DeserializePorts(addressInfoJson, source));
    return HCCL_SUCCESS;
}

HcclResult AddressInfo::ParseAddrByType(const std::string& addrType, const std::string& address, IpAddress& ipAddress)
{
    if (addrType == "IPV4") {
        CHK_PRT_RET(
            !IpAddress::IsIPv4(address),
            HCCL_ERROR("[%s] invalid IPV4 backup_addr[%.*s].", __func__, MAX_DISPLAY_LEN, address.c_str()),
            HCCL_E_PARA);
        ipAddress = IpAddress(address, AF_INET);
    } else if (addrType == "IPV6") {
        CHK_PRT_RET(
            !IpAddress::IsIPv6(address),
            HCCL_ERROR("[%s] invalid IPV6 backup_addr[%.*s].", __func__, MAX_DISPLAY_LEN, address.c_str()),
            HCCL_E_PARA);
        ipAddress = IpAddress(address, AF_INET6);
    } else {
        HCCL_ERROR("[%s] backup_addr does not support addrType[%s].", __func__, addrType.c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult AddressInfo::ParseBackupAddrs(
    const nlohmann::json& backupAddrJson, const std::string& addrType, std::vector<IpAddress>& backupAddrs)
{
    backupAddrs.clear();
    CHK_PRT_RET(!backupAddrJson.is_array(), HCCL_ERROR("[%s] backup_addr should be an array.", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        backupAddrJson.size() > MAX_VALUE_BACKUP_ADDR_SIZE,
        HCCL_ERROR(
            "[%s] backup_addr size[%zu] exceeds max[%u].", __func__, backupAddrJson.size(), MAX_VALUE_BACKUP_ADDR_SIZE),
        HCCL_E_PARA);

    for (const auto& backupAddr : backupAddrJson) {
        CHK_PRT_RET(
            !backupAddr.is_string(), HCCL_ERROR("[%s] backup_addr element should be string.", __func__), HCCL_E_PARA);
        const std::string backupAddrStr = backupAddr.get<std::string>();
        CHK_PRT_RET(
            backupAddrStr.length() < MIN_VALUE_ADDR_LENGRH || backupAddrStr.length() > MAX_VALUE_ADDR_LENGRH,
            HCCL_ERROR(
                "[%s] backup_addr[%.*s] length is out of range [%u] to [%u].", __func__, MAX_DISPLAY_LEN,
                backupAddrStr.c_str(), MIN_VALUE_ADDR_LENGRH, MAX_VALUE_ADDR_LENGRH),
            HCCL_E_PARA);
        IpAddress backupIpAddress;
        CHK_RET(ParseAddrByType(addrType, backupAddrStr, backupIpAddress));
        backupAddrs.emplace_back(backupIpAddress);
    }
    return HCCL_SUCCESS;
}

HcclResult AddressInfo::DeserializeBackupAddrs(const nlohmann::json& addressInfoJson, const std::string& addrTypeStr)
{
    backupAddrs.clear();
    // backup_addr 为可选字段，缺失时兼容未配置主备地址的 RankTable。
    if (!addressInfoJson.contains("backup_addr")) {
        HCCL_WARNING("[%s] backup_addr is not configured.", __func__);
        return HCCL_SUCCESS;
    }
    CHK_RET(ParseBackupAddrs(addressInfoJson.at("backup_addr"), addrTypeStr, backupAddrs));
    return HCCL_SUCCESS;
}

HcclResult AddressInfo::EidToAddr(std::string address)
{
    if (address.length() != URMA_EID_LEN * URMA_EID_NUM_TWO) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({address, "addr", "A valid EID"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] addr[%s] is not a valid EID: length[%zu], expected[%u]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), address.c_str(), address.length(),
            URMA_EID_LEN * URMA_EID_NUM_TWO);
        return HCCL_E_PARA;
    } else if (!IpAddress::IsEID(address)) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({address, "addr", "A valid EID"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] addr[%s] is not a valid EID format", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), address.c_str());
        return HCCL_E_PARA;
    }
    Eid eid = IpAddress::StrToEID(address);
    IpAddress ipAddress0(eid);
    addr = ipAddress0;
    return HCCL_SUCCESS;
}

HcclResult AddressInfo::IPV4ToAddr(std::string address)
{
    s32 ipFamily = AF_INET;

    if (IpAddress::IsIPv4(address)) {
        ipFamily = AF_INET;
    } else {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({address, "addr", "A valid IPv4 address"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] addr[%s] is not a valid IPv4 address", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), address.c_str());
        return HCCL_E_PARA;
    }
    IpAddress ipAddress0(address, ipFamily);
    addr = ipAddress0;
    HCCL_INFO("[AddressInfo::%s] IpAddress is[%s]", __func__, ipAddress0.Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult AddressInfo::IPV6ToAddr(std::string address)
{
    s32 ipFamily = AF_INET6;

    if (IpAddress::IsIPv6(address)) {
        ipFamily = AF_INET6;
    } else {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({address, "addr", "A valid IPv6 address"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] addr[%s] is not a valid IPv6 address", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), address.c_str());
        return HCCL_E_PARA;
    }
    IpAddress ipAddress0(address, ipFamily);
    addr = ipAddress0;
    return HCCL_SUCCESS;
}

std::string AddressInfo::Describe() const
{
    return StringFormat(
        "AddressInfo[addrType=%s, addr=%s, backupAddrSize=%u, planeId=%s, portsize=%u socketPort_=%u]",
        addrType.Describe().c_str(), addr.Describe().c_str(), static_cast<u32>(backupAddrs.size()), planeId.c_str(),
        static_cast<u32>(ports.size()), socketPort_);
}

AddressInfo::AddressInfo(BinaryStream& binStream)
{
    IpAddress address(binStream);
    addr = address;
    u32 addrTypeInt{0};
    binStream >> addrTypeInt;
    addrType = static_cast<AddrType::Value>(addrTypeInt);
    size_t portsSize{0};
    binStream >> portsSize;
    for (u32 i = 0; i < portsSize; i++) {
        string port;
        binStream >> port;
        ports.emplace(port);
    }
    binStream >> planeId;
    binStream >> socketPort_;
    size_t backupAddrSize{0};
    binStream >> backupAddrSize;
    CHK_PRT_THROW(
        backupAddrSize > MAX_VALUE_BACKUP_ADDR_SIZE,
        HCCL_ERROR("[%s] backup_addr size[%zu] exceeds max[%u].", __func__, backupAddrSize, MAX_VALUE_BACKUP_ADDR_SIZE),
        InvalidParamsException, "backup_addr size exceeds limit");
    for (size_t i = 0; i < backupAddrSize; i++) {
        IpAddress backupAddr(binStream);
        backupAddrs.emplace_back(backupAddr);
    }
}

HcclResult AddressInfo::GetBinStream(BinaryStream& binStream) const
{
    if (ports.size() == 0) {
        HCCL_ERROR("[%s] ports size is zero.", __func__);
        return HCCL_E_PARA;
    }
    addr.GetBinStream(binStream);
    binStream << static_cast<u32>(addrType);
    binStream << ports.size();
    for (auto& it : ports) {
        binStream << it;
    }
    binStream << planeId;
    binStream << socketPort_;
    binStream << backupAddrs.size();
    for (const auto& backupAddr : backupAddrs) {
        backupAddr.GetBinStream(binStream);
    }
    return HCCL_SUCCESS;
}

} // namespace Hccl
