/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_JSON_API_H
#define HCCLV2_JSON_API_H

#include <functional>
#include <string>
#include "types.h"

namespace Hccl {

class RankTableInfo;
class ChangedRankInfo;
class TopoInfo;
class IpAddress;

// 对外只保留三个域的解析入口，输入输出都不暴露具体 JSON 解析器。
// ranktable
HcclResult ParseRankTableInfoJson(const std::string& jsonString, RankTableInfo& out);
HcclResult ParseChangedRankInfoJson(const std::string& jsonString, ChangedRankInfo& out);
// topo
HcclResult ParseTopoInfoJson(const std::string& jsonString, TopoInfo& out);
HcclResult ParseTopoInfoFile(const std::string& filePath, TopoInfo& out);
// rootinfo：配置解析 + 本地 rank 编排
HcclResult ParseRootInfoConfigFile(const std::string& filePath, std::string& configJson);
HcclResult ParseRootInfoConfigJson(const std::string& jsonString, std::string& configJson);
HcclResult ParseRootInfoTopoFilePath(const std::string& filePath, std::string& topoFilePath);

// rootinfo 本地 rank 编排：配置文本在调用方与这些函数之间作为不透明句柄传递。
HcclResult CheckRootInfoConfig(const std::string& configJson);
HcclResult GetRootInfoTopoFilePathFromConfig(const std::string& configJson, std::string& out);
HcclResult
SelectRootInfoLocalDevInfo(const std::string& configJson, u32 devPhyId, u32 rankId, std::string& localDevInfoJson);
HcclResult SelectRootInfoHostAddr(
    std::string& localDevInfoJson, const std::function<s32(const IpAddress&)>& probe, u32 devPhyId, u32 rankId);
HcclResult AddRootInfoPcieFallbackLevel0(
    std::string& localDevInfoJson, const std::string& netInstId, bool hasVnicIp, const IpAddress& vnicIp);
HcclResult BuildRootInfoLocalRankTable(
    const std::string& configJson, const std::string& localDevInfoJson, u32 rankSize, RankTableInfo& out);

} // namespace Hccl

#endif // HCCLV2_JSON_API_H
