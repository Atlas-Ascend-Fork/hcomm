/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_JSON_PARSER_H
#define HCCLV2_JSON_PARSER_H

#include <string>
#include <linux/limits.h>
#include <fstream>
#include "nlohmann/json.hpp"
#include "log.h"
#include "string_util.h"
#include "internal_exception.h"
#include "invalid_params_exception.h"

namespace Hccl {

std::string GetJsonProperty(const nlohmann::json& obj, const char* propName, bool required = true);
u32 GetJsonPropertyUInt(const nlohmann::json& obj, const char* propName, bool required = true, u32 defaultValue = 0);
s32 GetJsonPropertySInt(const nlohmann::json& obj, const char* propName, bool required = true, s32 defaultValue = 0);
void GetJsonPropertyList(const nlohmann::json& obj, const char* propName, nlohmann::json& listObj);
class JsonParser {
public:
    template <typename T>
    HcclResult ParseString(const std::string& jsonString, T& info) const;
    template <typename T>
    HcclResult ParseFile(const std::string& filePath, T& info) const;
    template <typename T>
    void GetString(const T& info, std::string& infoStr) const;
    HcclResult ParseFileToJson(const std::string& filePath, nlohmann::json& parseInformation) const;

private:
    void ReportRanktableFileError(
        const std::string& filePath, const std::string& errorReason, HcclResult errorCode,
        const std::string& plogReason = "") const;
    template <typename T>
    HcclResult ParseInformation(nlohmann::json& parseInformation, T& information) const;
};

template <typename T>
HcclResult JsonParser::ParseString(const std::string& jsonString, T& info) const
{
    if (jsonString.empty()) {
        HCCL_ERROR("[%s] json string length is zero", __func__);
        return HCCL_E_PARA;
    }
    HCCL_INFO("waiting for json string load complete");

    nlohmann::json json;
    CHK_RET(ParseInformation(json, jsonString));

    try {
        HcclResult ret = info.Deserialize(json);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
    } catch (nlohmann::json::exception& e) {
        HCCL_ERROR("[%s] Deserialize failed: %s", __func__, e.what());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

template <typename T>
HcclResult JsonParser::ParseFile(const std::string& filePath, T& info) const
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

    nlohmann::json json;
    HcclResult parseRet = ParseInformation(json, infoFile);
    if (parseRet != HCCL_SUCCESS) {
        ReportRanktableFileError(filePath, "The JSON format is incorrect.", parseRet);
        return parseRet;
    }
    infoFile.close();

    try {
        HcclResult ret = info.Deserialize(json);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
    } catch (nlohmann::json::exception& e) {
        ReportRanktableFileError(
            filePath, "The JSON content cannot be deserialized.", HcclResult::HCCL_E_PARA, e.what());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

template <typename T>
void JsonParser::GetString(const T& info, std::string& infoStr) const
{
    nlohmann::json json;
    info.Serialize(json);
    infoStr = json.dump();
}

template <typename T>
HcclResult JsonParser::ParseInformation(nlohmann::json& parseInformation, T& information) const
{
    try {
        parseInformation = nlohmann::json::parse(information);
    } catch (const nlohmann::json::parse_error& e) {
        HCCL_ERROR("JSON parse error: %s at byte %d", e.what(), e.byte);
        return HCCL_E_PARA;
    } catch (const nlohmann::json::exception& e) {
        HCCL_ERROR("JSON parse error: %s", e.what());
        return HCCL_E_PARA;
    } catch (...) {
        HCCL_ERROR("[%s] Unknown JSON parse error", __func__);
        return HCCL_E_PARA;
    };
    return HCCL_SUCCESS;
}

} // namespace Hccl
#endif
