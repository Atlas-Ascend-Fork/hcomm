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

#include <algorithm>
#include <string>
#include <vector>
#include "json/common/json_parse_util.h"
#include "rank_table_fast_parser.h"
#include "rank_table_keys.h"
#include "adapter_error_manager_pub.h"
#include "rank_table_report_macro.h"
#include "common/hccl_log_keywords_pub.h"

namespace Hccl {

RankTableReader::RankTableReader(RankTableSource source) : source_(source) {}

HcclResult RankTableReader::ReadStringFast(const std::string& json, RankTableInfo& out)
{
    return ParseRankTableStringFastNoCheck(json, out);
}

HcclResult RankTableReader::Read(const std::string& jsonString, RankTableInfo& out) const
{
    if (jsonString.empty()) {
        HCCL_ERROR("[%s] json string length is zero", __func__);
        return HCCL_E_PARA;
    }
    try {
        const nlohmann::json json = nlohmann::json::parse(jsonString);
        return ReadJson(json, out);
    } catch (nlohmann::json::exception& e) {
        HCCL_ERROR("[%s] parse json failed: %s", __func__, e.what());
        return HCCL_E_PARA;
    }
}

HcclResult RankTableReader::ReadJson(const nlohmann::json& json, RankTableInfo& out) const
{
    out.version.clear();
    out.rankCount = 0;
    out.ranks.clear();
    out.detour = false;

    HcclResult ret = GetJsonProperty(json, kRankTableKeys[RankTableKey::kVersion].name, true, out.version);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(json, kRankTableKeys[RankTableKey::kVersion].name, "non-empty string", source_);
        return ret;
    }

    std::string detourStr;
    ret = GetJsonProperty(json, kRankTableKeys[RankTableKey::kDetour].name, false, detourStr);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(json, kRankTableKeys[RankTableKey::kDetour].name, "non-empty string", source_);
        return ret;
    }
    if (detourStr == "true") {
        out.detour = true;
    } else if (detourStr == "false" || detourStr == "") {
        out.detour = false;
    } else {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({detourStr, kRankTableKeys[RankTableKey::kDetour].name, "\"true\" or \"false\""}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] invalid detour value[%s]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), detourStr.c_str());
        return HCCL_E_PARA;
    }

    ret = GetJsonPropertyUInt(json, kRankTableKeys[RankTableKey::kRankCount].name, true, 0, out.rankCount);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(json, kRankTableKeys[RankTableKey::kRankCount].name, "0 ~ UINT32_MAX", source_);
        return ret;
    }

    // 与快速解析器一致: 超过 MAX_RANKCOUNT 直接拒绝, 避免伪造 rank_count 触发下游巨量分配
    if (out.rankCount > MAX_RANKCOUNT) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(out.rankCount), kRankTableKeys[RankTableKey::kRankCount].name,
                 "0 ~ " + std::to_string(MAX_RANKCOUNT)}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] rank_count[%u] exceeds maximum[%u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), out.rankCount,
            MAX_RANKCOUNT);
        return HCCL_E_PARA;
    }

    const nlohmann::json* rankJsons = nullptr;
    ret = GetJsonPropertyListRef(json, kRankTableKeys[RankTableKey::kRankList].name, rankJsons);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(json, kRankTableKeys[RankTableKey::kRankList].name, "array", source_);
        return ret;
    }
    // 声明值与实际条数取小做预留, 避免伪造字段先触发无谓的巨量分配
    out.ranks.reserve(std::min<size_t>(out.rankCount, rankJsons->size()));
    for (const auto& rankJson : *rankJsons) {
        NewRankInfo rankInfo;
        CHK_RET(ReadRank(rankJson, rankInfo));
        out.ranks.emplace_back(std::move(rankInfo));
    }

    // 与旧 Deserialize 保持一致：无论是否调用 Check，都按 rankId 排序。
    std::sort(out.ranks.begin(), out.ranks.end(), [](const NewRankInfo& a, const NewRankInfo& b) {
        return a.rankId < b.rankId;
    });
    return HCCL_SUCCESS;
}

} // namespace Hccl
