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
#include "json/common/json_parse_util.h"
#include "json/ranktable/rank_table_keys.h"
#include "adapter_error_manager_pub.h"
#include "rank_table_report_macro.h"
#include "topo_common_types.h"

namespace Hccl {

namespace {
    // 日志打印截断长度，与 JSON schema/解析参数无关，保留在 .cc。
    constexpr unsigned int MAX_DISPLAY_LEN = 128;

    HcclResult IPV4ToAddr(const std::string& address, IpAddress& out)
    {
        if (!IpAddress::IsIPv4(address)) {
            HCCL_ERROR("[RankTableReader::%s] failed with addrs is error.", __func__);
            return HCCL_E_PARA;
        }
        out = IpAddress(address, AF_INET);
        return HCCL_SUCCESS;
    }

    HcclResult IPV6ToAddr(const std::string& address, IpAddress& out)
    {
        if (!IpAddress::IsIPv6(address)) {
            HCCL_ERROR("[RankTableReader::%s] failed with addr is error.", __func__);
            return HCCL_E_PARA;
        }
        out = IpAddress(address, AF_INET6);
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult RankTableReader::ReadControlPlane(const nlohmann::json& controlPlaneJson, ControlPlane& out) const
{
    out = ControlPlane{};
    std::string addrTypeStr;
    HcclResult ret = GetJsonProperty(controlPlaneJson, kRankTableKeys[RankTableKey::kAddrType].name, true, addrTypeStr);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(
            controlPlaneJson, kRankTableKeys[RankTableKey::kAddrType].name, "non-empty string", source_);
        return ret;
    }
    if (kStrToAddrType.count(addrTypeStr) == 0) {
        HCCL_ERROR("[RankTableReader::%s] failed with Invalid addrType.", __func__);
        return HCCL_E_PARA;
    }
    out.addrType = kStrToAddrType.at(addrTypeStr);

    std::string address;
    ret = GetJsonProperty(controlPlaneJson, kRankTableKeys[RankTableKey::kAddr].name, true, address);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(controlPlaneJson, kRankTableKeys[RankTableKey::kAddr].name, "non-empty string", source_);
        return ret;
    }
    if (address.length() < MIN_VALUE_ADDR || address.length() > MAX_VALUE_ADDR) {
        HCCL_ERROR(
            "[%s] addr [%.*s] length is out of range [%u] to [%u]", __func__, MAX_DISPLAY_LEN, address.c_str(),
            MIN_VALUE_ADDR, MAX_VALUE_ADDR);
        return HCCL_E_PARA;
    }
    if (addrTypeStr == "IPV4") {
        CHK_RET(IPV4ToAddr(address, out.addr));
    } else if (addrTypeStr == "IPV6") {
        CHK_RET(IPV6ToAddr(address, out.addr));
    } else {
        CHK_RET(EidToAddr(address, out.addr));
    }

    ret = GetJsonPropertyUInt(
        controlPlaneJson, kRankTableKeys[RankTableKey::kListenPort].name, true, 0, out.listenPort);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(
            controlPlaneJson, kRankTableKeys[RankTableKey::kListenPort].name, "0 ~ UINT32_MAX", source_);
        return ret;
    }
    if (out.listenPort < MIN_VALUE_LISTENPORT || out.listenPort > MAX_VALUE_LISTENPORT) {
        HCCL_ERROR(
            "[%s] listen_port [%u] length is out of range [%u] to [%u]", __func__, out.listenPort, MIN_VALUE_LISTENPORT,
            MAX_VALUE_LISTENPORT);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult RankTableReader::EidToAddr(const std::string& address, IpAddress& out)
{
    if (address.length() != URMA_EID_LEN * URMA_EID_NUM_TWO) {
        HCCL_ERROR("[RankTableReader::%s] failed with rankAddrs : error in length.", __func__);
        return HCCL_E_PARA;
    }
    if (!IpAddress::IsEID(address)) {
        HCCL_ERROR("[RankTableReader::%s] failed with rankAddrs : error in format.", __func__);
        return HCCL_E_PARA;
    }
    out = IpAddress(IpAddress::StrToEID(address));
    return HCCL_SUCCESS;
}

} // namespace Hccl
