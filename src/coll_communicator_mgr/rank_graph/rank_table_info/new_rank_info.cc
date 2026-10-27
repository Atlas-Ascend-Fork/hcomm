/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "new_rank_info.h"

#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include "json_parser.h"
#include "const_val.h"
#include "exception_util.h"
#include "adapter_error_manager_pub.h"
#include "hccl_log_keywords.h"
#include "rank_table_report_macro.h"

namespace Hccl {
HcclResult CheakDeviceIdAndDevicePort(u32 deviceId, u32& devicePort)
{
    if (deviceId > MAX_VALUE_DEVICEID) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(deviceId), "device_id", "[0, " + std::to_string(MAX_VALUE_DEVICEID) + "]"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] device_id[%u] is out of range [0, %u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), deviceId,
            MAX_VALUE_DEVICEID);
        return HCCL_E_PARA;
    }
    if (devicePort > MAX_VALUE_TCPPORT || devicePort < MIN_VALUE_TCPPORT) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(devicePort), "device_port",
                 "[" + std::to_string(MIN_VALUE_TCPPORT) + ", " + std::to_string(MAX_VALUE_TCPPORT) + "]"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] device_port[%u] is out of range [%u, %u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), devicePort,
            MIN_VALUE_TCPPORT, MAX_VALUE_TCPPORT);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CheakLevelJsonsSize(u64 levelJsonsSize)
{
    if (levelJsonsSize > MAX_LEVEL_lIST) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(levelJsonsSize), "level_list size", "[0, " + std::to_string(MAX_LEVEL_lIST) + "]"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] level_list size[%llu] exceeds maximum[%u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), levelJsonsSize,
            MAX_LEVEL_lIST);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult NewRankInfo::DeserializeRankIdAndLocalId(const nlohmann::json& newRankInfoJson, RankTableSource source)
{
    std::string msgRankid = "error occurs when parser object of propName \"rank_id\"";
    std::string msgLocalid = "error occurs when parser object of propName \"local_id\"";
    TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgRankid, (rankId = GetJsonPropertyUInt(newRankInfoJson, "rank_id")), newRankInfoJson,
        "rank_id", "0 ~ UINT32_MAX", source));
    TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgLocalid, (localId = GetJsonPropertyUInt(newRankInfoJson, "local_id")),
        newRankInfoJson, "local_id", "0 ~ UINT32_MAX", source));
    if (localId > BACKUP_LOCAL_ID) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(localId), "local_id", "[0, " + std::to_string(BACKUP_LOCAL_ID) + "]"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] local_id[%u] is out of range [0, %u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), localId, BACKUP_LOCAL_ID);
        return HCCL_E_PARA;
    }
    if (localId == BACKUP_LOCAL_ID) {
        std::string msgReplacedId = "error occurs when parser object of propName \"replaced_local_id\"";
        TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT(
            InvalidParamsException, msgReplacedId,
            (replacedLocalId = GetJsonPropertyUInt(newRankInfoJson, "replaced_local_id")), newRankInfoJson,
            "replaced_local_id", "0 ~ UINT32_MAX", source));
        if (replacedLocalId > BACKUP_LOCAL_ID - 1) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>(
                    {std::to_string(replacedLocalId), "replaced_local_id",
                     "[0, " + std::to_string(BACKUP_LOCAL_ID - 1) + "]"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] replaced_local_id[%u] is out of range [0, %u]",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), replacedLocalId, BACKUP_LOCAL_ID - 1);
            return HCCL_E_PARA;
        }
    } else {
        replacedLocalId = localId;
    }
    return HCCL_SUCCESS;
}

HcclResult NewRankInfo::DeserializeDeviceIdAndPort(const nlohmann::json& newRankInfoJson, RankTableSource source)
{
    std::string msgDeviceid = "error occurs when parser object of propName \"device_id\"";
    std::string msgdeviceport = "error occurs when parser object of propName \"device_port\"";
    std::string msghostport = "error occurs when parser object of propName \"host_port\"";
    TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgDeviceid, (deviceId = GetJsonPropertyUInt(newRankInfoJson, "device_id")),
        newRankInfoJson, "device_id", "0 ~ UINT32_MAX", source));
    TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgdeviceport,
        (devicePort = GetJsonPropertyUInt(newRankInfoJson, "device_port", false, DEFAULT_VALUE_TCPPORT)),
        newRankInfoJson, "device_port", "0 ~ UINT32_MAX", source));
    TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msghostport,
        (hostPort = GetJsonPropertyUInt(newRankInfoJson, "host_port", false, DEFAULT_VALUE_TCPPORT)), newRankInfoJson,
        "host_port", "0 ~ UINT32_MAX", source));
    CHK_RET(CheakDeviceIdAndDevicePort(deviceId, devicePort));
    if (hostPort > MAX_VALUE_TCPPORT || hostPort < MIN_VALUE_TCPPORT) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(hostPort), "host_port",
                 "[" + std::to_string(MIN_VALUE_TCPPORT) + ", " + std::to_string(MAX_VALUE_TCPPORT) + "]"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] host_port[%u] is out of range [%u, %u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), hostPort, MIN_VALUE_TCPPORT,
            MAX_VALUE_TCPPORT);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult NewRankInfo::DeserializeLevelList(const nlohmann::json& newRankInfoJson, RankTableSource source)
{
    nlohmann::json levelJsons;
    std::string msgLevellist = "error occurs when parser object of propName \"level_list\"";
    TRY_CATCH_RETURN(TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgLevellist, (GetJsonPropertyList(newRankInfoJson, "level_list", levelJsons)),
        newRankInfoJson, "level_list", "array", source));
    CHK_RET(CheakLevelJsonsSize(levelJsons.size()));
    for (auto& levelJson : levelJsons) {
        RankLevelInfo levelInfo;
        CHK_RET(levelInfo.Deserialize(levelJson, source));
        for (auto& addrsInfo : levelInfo.rankAddrs) {
            addrsInfo.socketPort_ = devicePort;
        }
        rankLevelInfos.emplace_back(levelInfo);
    }

    std::vector<u32> levelSequence;
    for (auto& levelInfos : rankLevelInfos) {
        levelSequence.emplace_back(levelInfos.netLayer);
    }

    for (u32 i = 1; i < levelSequence.size(); i++) {
        if (levelSequence[i] <= levelSequence[i - 1]) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>(
                    {std::to_string(levelSequence[i]), "net_layer", "strictly increasing sequence"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] level is not increased in sequence, rankId[%d], index[%u], "
                "levelSequence[%u]=%u, levelSequence[%u]=%u",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rankId, i, i - 1, levelSequence[i - 1], i, levelSequence[i]);
            return HCCL_E_PARA;
        }
    }

    if (newRankInfoJson.contains("control_plane")) {
        nlohmann::json controlJsons;
        std::string msgControlPlane = "error occurs when parser object of propName \"control_plane\"";
        controlJsons = newRankInfoJson.at("control_plane");
        CHK_RET(controlPlane.Deserialize(controlJsons, source));
    }
    return HCCL_SUCCESS;
}

HcclResult NewRankInfo::Deserialize(const nlohmann::json& newRankInfoJson, RankTableSource source)
{
    CHK_RET(DeserializeRankIdAndLocalId(newRankInfoJson, source));
    CHK_RET(DeserializeDeviceIdAndPort(newRankInfoJson, source));
    CHK_RET(DeserializeLevelList(newRankInfoJson, source));
    return HCCL_SUCCESS;
}

std::string NewRankInfo::Describe() const
{
    return StringFormat(
        "NewRankInfo[rankId=%d, localId=%d, replacedLocalId=%d, ranklevelInfos size=%d, device_port=%d, "
        "host_port=%d, tlsStatus=%d, hostDpuTlsStatus=%d]",
        rankId, localId, replacedLocalId, rankLevelInfos.size(), devicePort, hostPort, static_cast<int>(tlsStatus),
        static_cast<int>(hostDpuTlsStatus));
}

NewRankInfo::NewRankInfo(BinaryStream& binStream)
{
    binStream >> rankId >> localId >> replacedLocalId >> deviceId >> devicePort >> hostPort;
    HCCL_DEBUG("[NewRankInfo] localId[%d]", localId);
    size_t rankLevelNum;
    binStream >> rankLevelNum;
    for (u32 i = 0; i < rankLevelNum; i++) {
        RankLevelInfo levelInfo(binStream);
        rankLevelInfos.emplace_back(levelInfo);
    }
    ControlPlane controlPlanes(binStream);
    controlPlane = controlPlanes;
    binStream >> tlsStatus;
    binStream >> hostDpuTlsStatus;
}

HcclResult NewRankInfo::GetBinStream(bool isContainLoaId, BinaryStream& binStream) const
{
    if (rankLevelInfos.size() == 0) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({"0", "rankLevelInfos", "non-empty"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] rankLevelInfos size is zero", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    if (isContainLoaId) {
        binStream << rankId << localId << replacedLocalId << deviceId << devicePort << hostPort;
    } else {
        binStream << rankId << INVALID_RANKID << INVALID_RANKID << deviceId << devicePort << hostPort;
    }
    binStream << rankLevelInfos.size();
    HCCL_INFO("[NewRankInfo] rankLevelInfos size[%u], rankId[%d]", rankLevelInfos.size(), rankId);
    for (auto& it : rankLevelInfos) {
        TRY_CATCH_RETURN(it.GetBinStream(binStream););
    }
    TRY_CATCH_RETURN(controlPlane.GetBinStream(binStream););
    binStream << tlsStatus;
    binStream << hostDpuTlsStatus;
    return HCCL_SUCCESS;
}

} // namespace Hccl
