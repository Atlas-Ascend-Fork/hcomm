/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "json_api.h"

#include "json/common/json_parse_util.h"
#include "json/rootinfo/root_info_reader.h"
#include "json/ranktable/changed_rank_table_reader.h"
#include "json/ranktable/rank_table_reader.h"
#include "json/topo/topo_info_reader.h"
#include "changed_rank_info.h"
#include "rank_table_info.h"
#include "topo_info.h"

namespace Hccl {

HcclResult ParseRankTableInfoJson(const std::string& jsonString, RankTableInfo& out)
{
    HcclResult ret = RankTableReader::ReadStringFast(jsonString, out);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    return out.Check();
}

HcclResult ParseChangedRankInfoJson(const std::string& jsonString, ChangedRankInfo& out)
{
    ChangedRankTableReader reader;
    return reader.ReadString(jsonString, out);
}

HcclResult ParseTopoInfoJson(const std::string& jsonString, TopoInfo& out)
{
    TopoInfoReader reader;
    HcclResult ret = reader.Read(jsonString, out);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    return out.Check();
}

HcclResult ParseTopoInfoFile(const std::string& filePath, TopoInfo& out)
{
    nlohmann::json json;
    CHK_RET(LoadJsonFile(filePath, json));
    return ReadJson<TopoInfo>(json, out);
}

HcclResult ParseRootInfoConfigFile(const std::string& filePath, std::string& configJson)
{
    RootInfoReader reader;
    nlohmann::json config;
    CHK_RET(reader.ReadFile(filePath, config));
    configJson = config.dump();
    return HCCL_SUCCESS;
}

HcclResult ParseRootInfoConfigJson(const std::string& jsonString, std::string& configJson)
{
    RootInfoReader reader;
    nlohmann::json config;
    CHK_RET(reader.Read(jsonString, config));
    configJson = config.dump();
    return HCCL_SUCCESS;
}

HcclResult ParseRootInfoTopoFilePath(const std::string& filePath, std::string& topoFilePath)
{
    RootInfoReader reader;
    nlohmann::json config;
    CHK_RET(reader.ReadFile(filePath, config));
    return reader.GetTopoFilePath(config, topoFilePath);
}

} // namespace Hccl
