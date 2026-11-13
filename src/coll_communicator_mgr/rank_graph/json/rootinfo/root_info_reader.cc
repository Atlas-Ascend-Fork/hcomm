/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "root_info_reader.h"
#include "json/json_api.h"

#include <cstdlib>
#include <limits.h>
#include <string>
#include <vector>
#include "json/common/json_parse_util.h"
#include "json/ranktable/rank_table_reader.h"
#include "json/ranktable/rank_table_keys.h"
#include "address_info.h"
#include "hccp_opcode.h"
#include "log.h"
#include "adapter_error_manager_pub.h"
#include "common/hccl_log_keywords_pub.h"
#include "rank_table_report_macro.h"

namespace Hccl {

HcclResult RootInfoReader::Read(const std::string& jsonString, nlohmann::json& out) const
{
    if (jsonString.empty()) {
        HCCL_ERROR("[%s] json string length is zero", __func__);
        return HCCL_E_PARA;
    }
    try {
        out = nlohmann::json::parse(jsonString);
    } catch (nlohmann::json::exception& e) {
        HCCL_ERROR("[%s] parse json failed: %s", __func__, e.what());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult RootInfoReader::ReadFile(const std::string& filePath, nlohmann::json& out) const
{
    return LoadJsonFile(filePath, out);
}

HcclResult RootInfoReader::Check(const nlohmann::json& config) const
{
    std::string version;
    HcclResult ret = GetJsonProperty(config, kRankTableKeys[RankTableKey::kVersion].name, true, version);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(
            config, kRankTableKeys[RankTableKey::kVersion].name, "non-empty string", RankTableSource::ROOTINFO);
        return ret;
    }
    if (version != "2.0") {
        RPT_INPUT_ERR(
            true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({version, kRankTableKeys[RankTableKey::kVersion].name, "2.0"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] version[%s] is not \"2.0\"", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), version.c_str());
        return HCCL_E_PARA;
    }

    std::string topoFilePath;
    ret = GetJsonProperty(config, kTopoFilePathKey, true, topoFilePath);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(config, kTopoFilePathKey, "non-empty string", RankTableSource::ROOTINFO);
        return ret;
    }

    char resolvedPath[PATH_MAX] = {0};
    if (realpath(topoFilePath.c_str(), resolvedPath) == nullptr) {
        RPT_INPUT_ERR(
            true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({topoFilePath, kTopoFilePathKey, "valid path"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] topo_file_path[%s] is not a valid real path", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), topoFilePath.c_str());
        return HCCL_E_PARA;
    }

    u32 rankCount = 0;
    ret = GetJsonPropertyUInt(config, kRankTableKeys[RankTableKey::kRankCount].name, true, 0, rankCount);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(
            config, kRankTableKeys[RankTableKey::kRankCount].name, "0 ~ UINT32_MAX", RankTableSource::ROOTINFO);
        return ret;
    }

    nlohmann::json rankJsons;
    ret = GetJsonPropertyList(config, kRankTableKeys[RankTableKey::kRankList].name, rankJsons);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(config, kRankTableKeys[RankTableKey::kRankList].name, "array", RankTableSource::ROOTINFO);
        return ret;
    }
    if (rankCount != rankJsons.size()) {
        RPT_INPUT_ERR(
            true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({std::to_string(rankCount), "rankCount", std::to_string(rankJsons.size())}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] rankCount[%u] != rank_list size[%zu]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rankCount,
            rankJsons.size());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult RootInfoReader::GetTopoFilePath(const nlohmann::json& config, std::string& out) const
{
    HcclResult ret = GetJsonProperty(config, kTopoFilePathKey, true, out);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(config, kTopoFilePathKey, "non-empty string", RankTableSource::ROOTINFO);
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult
RootInfoReader::SelectLocalDevInfo(const nlohmann::json& config, u32 devPhyId, u32 rankId, nlohmann::json& out) const
{
    nlohmann::json rankJsons;
    HcclResult ret = GetJsonPropertyList(config, kRankTableKeys[RankTableKey::kRankList].name, rankJsons);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(config, kRankTableKeys[RankTableKey::kRankList].name, "array", RankTableSource::ROOTINFO);
        HCCL_ERROR("[%s] error occurs when parser object of propName \"rank_list\"", __func__);
        return ret;
    }

    out = nlohmann::json{};
    for (auto& rankJson : rankJsons) {
        u32 devId = 0;
        ret = GetJsonPropertyUInt(rankJson, kRankTableKeys[RankTableKey::kDeviceId].name, true, 0, devId);
        if (ret != HCCL_SUCCESS) {
            ReportJsonConfigError(
                rankJson, kRankTableKeys[RankTableKey::kDeviceId].name, "0 ~ UINT32_MAX", RankTableSource::ROOTINFO);
            return ret;
        }
        if (devId == devPhyId) {
            HCCL_INFO("[%s] find localDevInfoJson.", __func__);
            out = rankJson;
            break;
        }
    }

    if (out.empty()) {
        HCCL_ERROR("[%s] failed, no device_id matches devPhyId[%u] in rank_list.", __func__, devPhyId);
    }

    out[kRankTableKeys[RankTableKey::kRankId].name] = rankId;
    HCCL_INFO("[%s] end.", __func__);
    return HCCL_SUCCESS;
}

HcclResult
RootInfoReader::BuildLocalRankTableJson(const nlohmann::json& config, u32 rankSize, nlohmann::json& out) const
{
    std::string version;
    HcclResult ret = GetJsonProperty(config, kRankTableKeys[RankTableKey::kVersion].name, true, version);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(
            config, kRankTableKeys[RankTableKey::kVersion].name, "non-empty string", RankTableSource::ROOTINFO);
        return ret;
    }
    out[kRankTableKeys[RankTableKey::kVersion].name] = version;

    std::string detour;
    ret = GetJsonProperty(config, kRankTableKeys[RankTableKey::kDetour].name, false, detour);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(
            config, kRankTableKeys[RankTableKey::kDetour].name, "non-empty string", RankTableSource::ROOTINFO);
        return ret;
    }
    if (detour == "true") {
        out[kRankTableKeys[RankTableKey::kDetour].name] = detour;
    }

    out[kRankTableKeys[RankTableKey::kRankCount].name] = rankSize;
    return HCCL_SUCCESS;
}

HcclResult RootInfoReader::SelectHostAddr(
    nlohmann::json& localDevInfoJson, const std::function<s32(const IpAddress&)>& probe, u32 devPhyId, u32 rankId) const
{
    const bool isLevelListInvalid = localDevInfoJson.empty()
                                    || !localDevInfoJson.contains(kRankTableKeys[RankTableKey::kLevelList].name)
                                    || !localDevInfoJson[kRankTableKeys[RankTableKey::kLevelList].name].is_array();
    CHK_PRT_RET(
        isLevelListInvalid,
        HCCL_ERROR(
            "[%s] level_list is missing or is not an array, devPhyId[%u], rankId[%u].", __func__, devPhyId, rankId),
        HCCL_E_PARA);
    std::vector<nlohmann::json*> addrJsons;
    CHK_RET(CollectLayer3AddrJsons(localDevInfoJson, addrJsons));
    if (addrJsons.empty()) {
        HCCL_DEBUG("[%s] no netLayer3+ addr with backup_addr needs probing.", __func__);
        return HCCL_SUCCESS;
    }
    for (auto* addrJson : addrJsons) {
        CHK_RET(SelectAvailableHostAddr(*addrJson, probe, devPhyId, rankId));
    }
    HCCL_INFO("[%s] end, devPhyId[%u], rankId[%u], addrConfigNum[%zu].", __func__, devPhyId, rankId, addrJsons.size());
    return HCCL_SUCCESS;
}

HcclResult
RootInfoReader::BuildHostAddrCandidates(const nlohmann::json& addrJson, std::vector<IpAddress>& candidates) const
{
    std::string addrType;
    std::string primaryAddr;
    HcclResult ret = GetJsonProperty(addrJson, kRankTableKeys[RankTableKey::kAddrType].name, true, addrType);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(
            addrJson, kRankTableKeys[RankTableKey::kAddrType].name, "non-empty string", RankTableSource::ROOTINFO);
        return ret;
    }
    ret = GetJsonProperty(addrJson, kRankTableKeys[RankTableKey::kAddr].name, true, primaryAddr);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(
            addrJson, kRankTableKeys[RankTableKey::kAddr].name, "non-empty string", RankTableSource::ROOTINFO);
        return ret;
    }
    IpAddress primaryIpAddress;
    CHK_RET(RankTableReader::ParseAddrByType(
        addrType, primaryAddr, primaryIpAddress, kRankTableKeys[RankTableKey::kAddr].name));
    candidates.clear();
    candidates.emplace_back(primaryIpAddress);

    std::vector<IpAddress> backupAddrs;
    if (!addrJson.contains(kRankTableKeys[RankTableKey::kBackupAddr].name)) {
        HCCL_ERROR("[BuildHostAddrCandidates] backup_addr field missing.");
        return HCCL_E_PARA;
    }
    CHK_RET(RankTableReader::ParseBackupAddrs(
        addrJson.at(kRankTableKeys[RankTableKey::kBackupAddr].name), addrType, backupAddrs));
    candidates.insert(candidates.end(), backupAddrs.begin(), backupAddrs.end());
    return HCCL_SUCCESS;
}

HcclResult
RootInfoReader::CollectLayer3AddrJsons(nlohmann::json& localDevInfoJson, std::vector<nlohmann::json*>& addrJsons) const
{
    addrJsons.clear();
    CHK_PRT_RET(
        !localDevInfoJson.contains(kRankTableKeys[RankTableKey::kLevelList].name)
            || !localDevInfoJson.at(kRankTableKeys[RankTableKey::kLevelList].name).is_array(),
        HCCL_ERROR("[%s] level_list is missing or is not an array.", __func__), HCCL_E_PARA);
    for (auto& levelJson : localDevInfoJson.at(kRankTableKeys[RankTableKey::kLevelList].name)) {
        if (!levelJson.contains(kRankTableKeys[RankTableKey::kRankAddrList].name)
            || !levelJson[kRankTableKeys[RankTableKey::kRankAddrList].name].is_array()) {
            continue;
        }
        u32 netLayer = 0;
        HcclResult ret
            = GetJsonPropertyUInt(levelJson, kRankTableKeys[RankTableKey::kNetLayer].name, true, 0, netLayer);
        if (ret != HCCL_SUCCESS) {
            ReportJsonConfigError(
                levelJson, kRankTableKeys[RankTableKey::kNetLayer].name, "0 ~ UINT32_MAX", RankTableSource::ROOTINFO);
            return ret;
        }
        if (netLayer < kHostBackupAddrNetLayer) {
            continue;
        }
        for (auto& addrJson : levelJson[kRankTableKeys[RankTableKey::kRankAddrList].name]) {
            if (!addrJson.contains(kRankTableKeys[RankTableKey::kBackupAddr].name)) {
                HCCL_WARNING("[%s] backup_addr is not configured, use primary addr without probing.", __func__);
                continue;
            }
            addrJsons.push_back(&addrJson);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RootInfoReader::SelectAvailableHostAddr(
    nlohmann::json& addrJson, const std::function<s32(const IpAddress&)>& probe, u32 devPhyId, u32 rankId) const
{
    std::vector<IpAddress> candidates;
    CHK_RET(BuildHostAddrCandidates(addrJson, candidates));
    HCCL_INFO(
        "[%s] devPhyId[%u], rankId[%u], primaryAddr[%s], backupAddrSize[%zu], candidateSize[%zu].", __func__, devPhyId,
        rankId, candidates.front().Describe().c_str(), candidates.size() - 1, candidates.size());

    for (std::size_t idx = 0; idx < candidates.size(); ++idx) {
        const s32 ret = probe(candidates[idx]);
        if (ret == 0) {
            UpdateSelectedHostAddr(addrJson, candidates, idx, devPhyId, rankId);
            return HCCL_SUCCESS;
        }
        if (ret == HCCP_ELINKDOWN) {
            HCCL_RUN_WARNING(
                "[%s] host addr[%s] is linkdown, try next candidate, devPhyId[%u], rankId[%u], candidateIndex[%zu].",
                __func__, candidates[idx].Describe().c_str(), devPhyId, rankId, idx);
            continue;
        }
        HCCL_ERROR(
            "[%s] host addr[%s] probe failed with non-linkdown error, ret[%d], devPhyId[%u], rankId[%u], "
            "candidateIndex[%zu].",
            __func__, candidates[idx].Describe().c_str(), ret, devPhyId, rankId, idx);
        return HCCL_E_NETWORK;
    }
    HCCL_RUN_WARNING(
        "[%s] all host addr candidates are linkdown, keep primary address and continue, "
        "devPhyId[%u], rankId[%u], primaryAddr[%s], candidateSize[%zu].",
        __func__, devPhyId, rankId, candidates.front().Describe().c_str(), candidates.size());
    return HCCL_SUCCESS;
}

void RootInfoReader::UpdateSelectedHostAddr(
    nlohmann::json& addrJson, const std::vector<IpAddress>& candidates, std::size_t selectedIndex, u32 devPhyId,
    u32 rankId) const
{
    const std::string oldAddr = addrJson[kRankTableKeys[RankTableKey::kAddr].name].get<std::string>();
    addrJson[kRankTableKeys[RankTableKey::kAddr].name] = candidates[selectedIndex].GetIpStr();
    HCCL_RUN_INFO(
        "[%s] select host addr success, devPhyId[%u], rankId[%u], "
        "selectedNicRole[%s], oldHostAddr[%s], selectedHostAddr[%s], candidateIndex[%zu], tryCount[%zu].",
        __func__, devPhyId, rankId, selectedIndex == 0 ? "primary" : "backup", oldAddr.c_str(),
        candidates[selectedIndex].Describe().c_str(), selectedIndex, selectedIndex + 1);
}

bool RootInfoReader::IsLevel0Exist(const nlohmann::json& localDevInfoJson) const
{
    if (!localDevInfoJson.contains(kRankTableKeys[RankTableKey::kLevelList].name)
        || !localDevInfoJson.at(kRankTableKeys[RankTableKey::kLevelList].name).is_array()) {
        return false;
    }
    for (const auto& levelJson : localDevInfoJson.at(kRankTableKeys[RankTableKey::kLevelList].name)) {
        if (levelJson.value<u32>(kRankTableKeys[RankTableKey::kNetLayer].name, kHostBackupAddrNetLayer) == 0) {
            return true;
        }
    }
    return false;
}

HcclResult RootInfoReader::AddPcieFallbackLevel0(
    nlohmann::json& localDevInfoJson, const std::string& netInstId, bool hasVnicIp, const IpAddress& vnicIp) const
{
    if (IsLevel0Exist(localDevInfoJson)) {
        return HCCL_SUCCESS;
    }
    HCCL_WARNING(
        "[%s] no level0 in rootinfo (UB unavailable), insert pcie fallback level0, netInstId[%s].", __func__,
        netInstId.c_str());
    nlohmann::json fallbackLevel;
    fallbackLevel[kRankTableKeys[RankTableKey::kNetLayer].name] = 0;
    fallbackLevel[kRankTableKeys[RankTableKey::kNetInstanceId].name] = netInstId;
    fallbackLevel[kRankTableKeys[RankTableKey::kNetType].name] = "TOPO_FILE_DESC";
    fallbackLevel[kRankTableKeys[RankTableKey::kNetAttr].name] = "";
    fallbackLevel[kRankTableKeys[RankTableKey::kPcieFallback].name] = true;
    if (hasVnicIp) {
        nlohmann::json addrEntry;
        addrEntry[kRankTableKeys[RankTableKey::kAddrType].name] = "IPV4";
        addrEntry[kRankTableKeys[RankTableKey::kAddr].name] = vnicIp.GetIpStr();
        addrEntry[kRankTableKeys[RankTableKey::kPlaneId].name] = "0";
        addrEntry[kRankTableKeys[RankTableKey::kPorts].name] = nlohmann::json::array({std::string("d2h")});
        fallbackLevel[kRankTableKeys[RankTableKey::kRankAddrList].name] = nlohmann::json::array({addrEntry});
    } else {
        fallbackLevel[kRankTableKeys[RankTableKey::kRankAddrList].name] = nlohmann::json::array();
    }
    if (!localDevInfoJson.contains(kRankTableKeys[RankTableKey::kLevelList].name)) {
        localDevInfoJson[kRankTableKeys[RankTableKey::kLevelList].name] = nlohmann::json::array();
    }
    localDevInfoJson[kRankTableKeys[RankTableKey::kLevelList].name].push_back(fallbackLevel);
    return HCCL_SUCCESS;
}

HcclResult RootInfoReader::BuildLocalRankTable(
    const nlohmann::json& config, const nlohmann::json& localDevInfo, u32 rankSize, RankTableInfo& out) const
{
    nlohmann::json localRankTableJson;
    CHK_RET(BuildLocalRankTableJson(config, rankSize, localRankTableJson));
    localRankTableJson[kRankTableKeys[RankTableKey::kRankList].name].push_back(localDevInfo);
    RankTableReader reader(RankTableSource::ROOTINFO);
    return reader.ReadJson(localRankTableJson, out);
}

HcclResult CheckRootInfoConfig(const std::string& configJson)
{
    RootInfoReader reader;
    nlohmann::json config;
    CHK_RET(reader.Read(configJson, config));
    return reader.Check(config);
}

HcclResult GetRootInfoTopoFilePathFromConfig(const std::string& configJson, std::string& out)
{
    RootInfoReader reader;
    nlohmann::json config;
    CHK_RET(reader.Read(configJson, config));
    return reader.GetTopoFilePath(config, out);
}

HcclResult
SelectRootInfoLocalDevInfo(const std::string& configJson, u32 devPhyId, u32 rankId, std::string& localDevInfoJson)
{
    RootInfoReader reader;
    nlohmann::json config;
    CHK_RET(reader.Read(configJson, config));
    nlohmann::json localDevInfo;
    CHK_RET(reader.SelectLocalDevInfo(config, devPhyId, rankId, localDevInfo));
    localDevInfoJson = localDevInfo.dump();
    return HCCL_SUCCESS;
}

HcclResult SelectRootInfoHostAddr(
    std::string& localDevInfoJson, const std::function<s32(const IpAddress&)>& probe, u32 devPhyId, u32 rankId)
{
    RootInfoReader reader;
    nlohmann::json localDevInfo;
    CHK_RET(reader.Read(localDevInfoJson, localDevInfo));
    CHK_RET(reader.SelectHostAddr(localDevInfo, probe, devPhyId, rankId));
    localDevInfoJson = localDevInfo.dump();
    return HCCL_SUCCESS;
}

HcclResult AddRootInfoPcieFallbackLevel0(
    std::string& localDevInfoJson, const std::string& netInstId, bool hasVnicIp, const IpAddress& vnicIp)
{
    RootInfoReader reader;
    nlohmann::json localDevInfo;
    CHK_RET(reader.Read(localDevInfoJson, localDevInfo));
    CHK_RET(reader.AddPcieFallbackLevel0(localDevInfo, netInstId, hasVnicIp, vnicIp));
    localDevInfoJson = localDevInfo.dump();
    return HCCL_SUCCESS;
}

HcclResult BuildRootInfoLocalRankTable(
    const std::string& configJson, const std::string& localDevInfoJson, u32 rankSize, RankTableInfo& out)
{
    RootInfoReader reader;
    nlohmann::json config;
    nlohmann::json localDevInfo;
    CHK_RET(reader.Read(configJson, config));
    CHK_RET(reader.Read(localDevInfoJson, localDevInfo));
    return reader.BuildLocalRankTable(config, localDevInfo, rankSize, out);
}

} // namespace Hccl
