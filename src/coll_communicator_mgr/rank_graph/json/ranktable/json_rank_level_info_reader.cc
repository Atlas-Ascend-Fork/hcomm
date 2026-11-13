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

namespace Hccl {

HcclResult RankTableReader::ReadLevel(const nlohmann::json& rankLevelInfoJson, RankLevelInfo& out) const
{
    out = RankLevelInfo{};
    CHK_RET(ReadNetLayerInfo(rankLevelInfoJson, out));
    CHK_RET(ReadRankAddrs(rankLevelInfoJson, out));
    BuildPortAddrMap(out);
    return HCCL_SUCCESS;
}

HcclResult RankTableReader::ReadNetLayerInfo(const nlohmann::json& rankLevelInfoJson, RankLevelInfo& out) const
{
    HcclResult ret
        = GetJsonPropertyUInt(rankLevelInfoJson, kRankTableKeys[RankTableKey::kNetLayer].name, true, 0, out.netLayer);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(
            rankLevelInfoJson, kRankTableKeys[RankTableKey::kNetLayer].name, "0 ~ UINT32_MAX", source_);
        return ret;
    }
    ret = GetJsonProperty(rankLevelInfoJson, kRankTableKeys[RankTableKey::kNetInstanceId].name, true, out.netInstId);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(
            rankLevelInfoJson, kRankTableKeys[RankTableKey::kNetInstanceId].name, "non-empty string", source_);
        return ret;
    }
    if (out.netLayer > MAX_VALUE_NETLAYER) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({std::to_string(out.netLayer), "net_layer", "[0, 7]"}));
        return HCCL_E_PARA;
    }
    if (out.netInstId.length() < MIN_VALUE_NETID || out.netInstId.length() > MAX_VALUE_NETID) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({out.netInstId, "net_instance_id", "length [1, 1024]"}));
        return HCCL_E_PARA;
    }
    out.netAttr = rankLevelInfoJson.value<std::string>(kRankTableKeys[RankTableKey::kNetAttr].name, "");
    out.pcieFallback = rankLevelInfoJson.value<bool>(kRankTableKeys[RankTableKey::kPcieFallback].name, false);
    if (rankLevelInfoJson.contains(kRankTableKeys[RankTableKey::kNetType].name)) {
        std::string netTypeStr;
        ret = GetJsonProperty(rankLevelInfoJson, kRankTableKeys[RankTableKey::kNetType].name, true, netTypeStr);
        if (ret != HCCL_SUCCESS) {
            ReportJsonConfigError(
                rankLevelInfoJson, kRankTableKeys[RankTableKey::kNetType].name, "non-empty string", source_);
            return ret;
        }
        if (kStrToNetType.count(netTypeStr) == 0) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({netTypeStr, "net_type", "valid NetType"}));
            return HCCL_E_PARA;
        }
        out.netType = kStrToNetType.at(netTypeStr);
    }
    return HCCL_SUCCESS;
}

HcclResult RankTableReader::ReadRankAddrs(const nlohmann::json& rankLevelInfoJson, RankLevelInfo& out) const
{
    const nlohmann::json* rankAddrs = nullptr;
    HcclResult ret
        = GetJsonPropertyListRef(rankLevelInfoJson, kRankTableKeys[RankTableKey::kRankAddrList].name, rankAddrs);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(rankLevelInfoJson, kRankTableKeys[RankTableKey::kRankAddrList].name, "array", source_);
        return ret;
    }
    if (rankAddrs->size() > MAX_VALUE_RANKADDR_SIZE) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({std::to_string(rankAddrs->size()), "rank_addr_list size", "[0, 24]"}));
        return HCCL_E_PARA;
    }
    out.rankAddrs.reserve(rankAddrs->size());
    for (const auto& addr : *rankAddrs) {
        AddressInfo addressInfo;
        CHK_RET(ReadAddress(addr, addressInfo));
        out.rankAddrs.emplace_back(std::move(addressInfo));
    }
    return HCCL_SUCCESS;
}

void RankTableReader::BuildPortAddrMap(RankLevelInfo& out) const
{
    for (auto& rankAddr : out.rankAddrs) {
        const IpAddress ipAddress = rankAddr.addr;
        for (auto& port : rankAddr.ports) {
            out.portAddrMap[port].push_back(ipAddress);
        }
    }
}

} // namespace Hccl
