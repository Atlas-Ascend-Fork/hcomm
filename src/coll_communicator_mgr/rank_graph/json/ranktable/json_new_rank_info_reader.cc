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
    HcclResult CheckDeviceIdAndPort(u32 deviceId, u32 devicePort)
    {
        if (deviceId > MAX_VALUE_DEVICEID) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({std::to_string(deviceId), "device_id", "[0, 512]"}));
            return HCCL_E_PARA;
        }
        if (devicePort > MAX_VALUE_TCPPORT || devicePort < MIN_VALUE_TCPPORT) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({std::to_string(devicePort), "device_port", "[1, 65535]"}));
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    HcclResult CheckLevelJsonsSize(u64 levelJsonsSize)
    {
        if (levelJsonsSize > MAX_LEVEL_lIST) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({std::to_string(levelJsonsSize), "level_list size", "[0, 8]"}));
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult RankTableReader::ReadRank(const nlohmann::json& newRankInfoJson, NewRankInfo& out) const
{
    out = NewRankInfo{};
    CHK_RET(ReadRankIdAndLocalId(newRankInfoJson, out));
    CHK_RET(ReadDeviceIdAndPort(newRankInfoJson, out));
    return ReadLevelList(newRankInfoJson, out);
}

HcclResult RankTableReader::ReadRankIdAndLocalId(const nlohmann::json& json, NewRankInfo& out) const
{
    HcclResult ret = GetJsonPropertyUInt(json, kRankTableKeys[RankTableKey::kRankId].name, true, 0, out.rankId);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(json, kRankTableKeys[RankTableKey::kRankId].name, "0 ~ UINT32_MAX", source_);
        return ret;
    }
    ret = GetJsonPropertyUInt(json, kRankTableKeys[RankTableKey::kLocalId].name, true, 0, out.localId);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(json, kRankTableKeys[RankTableKey::kLocalId].name, "0 ~ UINT32_MAX", source_);
        return ret;
    }
    if (out.localId > BACKUP_LOCAL_ID) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({std::to_string(out.localId), "local_id", "[0, 512]"}));
        return HCCL_E_PARA;
    }
    if (out.localId == BACKUP_LOCAL_ID) {
        ret = GetJsonPropertyUInt(
            json, kRankTableKeys[RankTableKey::kReplacedLocalId].name, true, 0, out.replacedLocalId);
        if (ret != HCCL_SUCCESS) {
            ReportJsonConfigError(json, kRankTableKeys[RankTableKey::kReplacedLocalId].name, "0 ~ UINT32_MAX", source_);
            return ret;
        }
        if (out.replacedLocalId > BACKUP_LOCAL_ID - 1) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({std::to_string(out.replacedLocalId), "replaced_local_id", "[0, 511]"}));
            return HCCL_E_PARA;
        }
    } else {
        out.replacedLocalId = out.localId;
    }
    return HCCL_SUCCESS;
}

HcclResult RankTableReader::ReadDeviceIdAndPort(const nlohmann::json& json, NewRankInfo& out) const
{
    HcclResult ret = GetJsonPropertyUInt(json, kRankTableKeys[RankTableKey::kDeviceId].name, true, 0, out.deviceId);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(json, kRankTableKeys[RankTableKey::kDeviceId].name, "0 ~ UINT32_MAX", source_);
        return ret;
    }
    ret = GetJsonPropertyUInt(
        json, kRankTableKeys[RankTableKey::kDevicePort].name, false, DEFAULT_VALUE_TCPPORT, out.devicePort);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(json, kRankTableKeys[RankTableKey::kDevicePort].name, "0 ~ UINT32_MAX", source_);
        return ret;
    }
    ret = GetJsonPropertyUInt(
        json, kRankTableKeys[RankTableKey::kHostPort].name, false, DEFAULT_VALUE_TCPPORT, out.hostPort);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(json, kRankTableKeys[RankTableKey::kHostPort].name, "0 ~ UINT32_MAX", source_);
        return ret;
    }
    CHK_RET(CheckDeviceIdAndPort(out.deviceId, out.devicePort));
    if (out.hostPort > MAX_VALUE_TCPPORT || out.hostPort < MIN_VALUE_TCPPORT) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({std::to_string(out.hostPort), "host_port", "[1, 65535]"}));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult RankTableReader::ReadLevelList(const nlohmann::json& json, NewRankInfo& out) const
{
    const nlohmann::json* levelJsons = nullptr;
    HcclResult ret = GetJsonPropertyListRef(json, kRankTableKeys[RankTableKey::kLevelList].name, levelJsons);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(json, kRankTableKeys[RankTableKey::kLevelList].name, "array", source_);
        return ret;
    }
    CHK_RET(CheckLevelJsonsSize(levelJsons->size()));
    out.rankLevelInfos.reserve(levelJsons->size());
    for (const auto& levelJson : *levelJsons) {
        RankLevelInfo levelInfo;
        CHK_RET(ReadLevel(levelJson, levelInfo));
        for (auto& addrsInfo : levelInfo.rankAddrs) {
            addrsInfo.socketPort_ = out.devicePort;
        }
        out.rankLevelInfos.emplace_back(std::move(levelInfo));
    }
    std::vector<u32> levelSequence;
    for (auto& levelInfo : out.rankLevelInfos) {
        levelSequence.emplace_back(levelInfo.netLayer);
    }
    for (u32 i = 1; i < levelSequence.size(); i++) {
        if (levelSequence[i] <= levelSequence[i - 1]) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>(
                    {std::to_string(levelSequence[i]), "net_layer", "strictly increasing sequence"}));
            return HCCL_E_PARA;
        }
    }
    const auto controlIt = json.find(kRankTableKeys[RankTableKey::kControlPlane].name);
    if (controlIt != json.end()) {
        CHK_RET(ReadControlPlane(*controlIt, out.controlPlane));
    }
    return HCCL_SUCCESS;
}

} // namespace Hccl
