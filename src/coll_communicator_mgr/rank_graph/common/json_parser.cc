/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include "json_parser.h"
#include "adapter_error_manager_pub.h"
#include "exception_util.h"
#include "common/hccl_log_keywords_pub.h"

namespace Hccl {

std::string GetJsonProperty(const nlohmann::json& obj, const char* propName, bool required)
{
    CHK_PRT_THROW(
        propName == nullptr, HCCL_ERROR("[%s] propName is nullptr", __func__), NullPtrException, "propName is nullptr");
    if (!required) {
        return obj.value<std::string>(propName, "");
    }

    CHK_PRT_THROW(
        !obj.contains(propName), HCCL_ERROR("[Get][JsonProperty] json object does not contain property[%s].", propName),
        InvalidParamsException, "json object does not contain required property");
    std::string value = obj.at(propName).get<std::string>();
    return value;
}

u32 GetJsonPropertyUInt(const nlohmann::json& obj, const char* propName, bool required, u32 defaultValue)
{
    if (!required) {
        return obj.value<u32>(propName, defaultValue);
    }

    CHK_PRT_THROW(
        !obj.contains(propName),
        HCCL_ERROR("[Get][JsonPropertyUInt] json object does not contain property[%s].", propName),
        InvalidParamsException, "json object does not contain required property");

    s64 value = obj.at(propName).get<s64>();
    CHK_PRT_THROW(
        value < 0 || value > UINT32_MAX,
        HCCL_ERROR(
            "[Get][JsonPropertyUInt]errNo[0x%016llx]:json object "
            "property value of Name[%s] should be an unsigned 32-bit integer but actually not!",
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), propName),
        InvalidParamsException, "json property value is not an unsigned 32-bit integer");
    return static_cast<u32>(value);
}

s32 GetJsonPropertySInt(const nlohmann::json& obj, const char* propName, bool required, s32 defaultValue)
{
    if (!required) {
        return obj.value<s32>(propName, defaultValue);
    }

    CHK_PRT_THROW(
        !obj.contains(propName),
        HCCL_ERROR("[Get][JsonPropertySInt] json object does not contain property[%s].", propName),
        InvalidParamsException, "json object does not contain required property");

    s64 value = obj.at(propName).get<s64>();
    CHK_PRT_THROW(
        value < INT_MIN || value > INT_MAX,
        HCCL_ERROR(
            "[Get][JsonPropertySInt]errNo[0x%016llx]:json object "
            "property value of Name[%s] is not signed number!",
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), propName),
        InvalidParamsException, "json property value is not a signed integer");
    HCCL_INFO(
        "[Get][JsonPropertySInt]json object property value of Name [%s] is [%d]", propName, static_cast<s32>(value));
    return static_cast<s32>(value);
}

HcclResult GetJsonPropertyList(const nlohmann::json& obj, const char* propName, nlohmann::json& listObj)
{
    if (!obj.contains(propName)) {
        HCCL_ERROR("[GetJsonPropertyList] json object does not contain property[%s].", propName);
        return HCCL_E_PARA;
    }

    listObj = obj.at(propName);
    if (!listObj.is_array()) {
        HCCL_ERROR("[GetJsonPropertyList] json object property value of Name \"%s\" is not list!", propName);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

void JsonParser::ReportRanktableFileError(
    const std::string& filePath, const std::string& errorReason, HcclResult errorCode,
    const std::string& plogReason) const
{
    const std::string reportedPath = filePath.empty() ? "<empty>" : filePath;
    const std::string& reportedReason = plogReason.empty() ? errorReason : plogReason;
    RPT_INPUT_ERR(
        true, "EI0004", std::vector<std::string>({"ranktable_path", "error_reason"}),
        std::vector<std::string>({reportedPath, errorReason}));
    HCCL_ERROR(
        "[%s][%s] errNo[0x%016llx] rankTable file[%s] parse failed, reason: %s", LOG_KEYWORDS_INIT_GROUP.c_str(),
        LOG_KEYWORDS_RANKTABLE_CONFIG.c_str(), HCOM_ERROR_CODE(errorCode), reportedPath.c_str(),
        reportedReason.c_str());
}

HcclResult JsonParser::ParseFileToJson(const std::string& filePath, nlohmann::json& parseInformation) const
{
    // 校验文件是否存在
    char resolvedPath[PATH_MAX] = {0};
    if (realpath(filePath.c_str(), resolvedPath) == nullptr) {
        ReportRanktableFileError(
            filePath,
            "The rankTable file path does not exist, the permission is insufficient, or the JSON format is incorrect.",
            HcclResult::HCCL_E_PARA, StringFormat("path %s is not a valid real path", filePath.c_str()));
        return HCCL_E_PARA;
    }

    HCCL_INFO("waiting for json file load complete");
    std::ifstream infoFile(resolvedPath, std::ifstream::in);
    if (!infoFile) {
        ReportRanktableFileError(
            filePath, "Failed to open the rankTable file.", HcclResult::HCCL_E_OPEN_FILE_FAILURE,
            StringFormat("open file %s failed", resolvedPath));
        return HCCL_E_OPEN_FILE_FAILURE;
    }

    HcclResult ret = ParseInformation(parseInformation, infoFile);
    if (ret != HCCL_SUCCESS) {
        ReportRanktableFileError(filePath, "The JSON format is incorrect.", ret);
        return ret;
    }
    infoFile.close();
    return HCCL_SUCCESS;
}

} // namespace Hccl
