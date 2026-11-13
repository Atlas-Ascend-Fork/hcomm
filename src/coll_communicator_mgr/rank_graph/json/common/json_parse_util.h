/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_JSON_PARSE_UTIL_H
#define HCCLV2_JSON_PARSE_UTIL_H

#include <istream>
#include <string>
#include "nlohmann/json.hpp"
#include "log.h"
#include "string_util.h"

namespace Hccl {

HcclResult GetJsonProperty(const nlohmann::json& obj, const char* propName, bool required, std::string& value);
HcclResult
GetJsonPropertyUInt(const nlohmann::json& obj, const char* propName, bool required, u32 defaultValue, u32& value);
HcclResult
GetJsonPropertySInt(const nlohmann::json& obj, const char* propName, bool required, s32 defaultValue, s32& value);
HcclResult GetJsonPropertyList(const nlohmann::json& obj, const char* propName, nlohmann::json& listObj);
// 引用版: 不拷贝数组内容, 供大数组(如 rank_list)使用
HcclResult GetJsonPropertyListRef(const nlohmann::json& obj, const char* propName, const nlohmann::json*& listObj);

// 解析入口的定制点：默认走模型的 Deserialize；具体类型可在对应 Reader 头里特化。
template <typename T>
HcclResult ReadJson(const nlohmann::json& json, T& info)
{
    return info.Deserialize(json);
}

namespace detail {
    HcclResult ParseJsonInformation(nlohmann::json& parseInformation, const std::string& jsonString);
    HcclResult ParseJsonInformation(nlohmann::json& parseInformation, std::istream& jsonStream);
} // namespace detail

// JSON 文本 -> 模型
template <typename T>
HcclResult ParseJsonString(const std::string& jsonString, T& info)
{
    if (jsonString.empty()) {
        HCCL_ERROR("[%s] json string length is zero", __func__);
        return HCCL_E_PARA;
    }
    HCCL_INFO("waiting for json string load complete");

    nlohmann::json json;
    CHK_RET(detail::ParseJsonInformation(json, jsonString));
    try {
        return ReadJson<T>(json, info);
    } catch (nlohmann::json::exception& e) {
        HCCL_ERROR("[%s] read json failed: %s", __func__, e.what());
        return HCCL_E_PARA;
    }
}

// 文件 -> DOM（不映射模型）
HcclResult LoadJsonFile(const std::string& filePath, nlohmann::json& parseInformation);

} // namespace Hccl

#endif // HCCLV2_JSON_PARSE_UTIL_H
