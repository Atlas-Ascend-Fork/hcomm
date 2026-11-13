/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ROOT_INFO_READER_H
#define HCCLV2_ROOT_INFO_READER_H

#include <cstddef>
#include <functional>
#include <string>
#include "ip_address.h"
#include "rank_table_info.h"
#include "types.h"
#include "nlohmann/json.hpp"

namespace Hccl {

// rootinfo 配置文件（/etc/hccl_rootinfo.json）的读取与字段解析。
// 这里只处理 JSON 结构和配置字段；本地地址选择、端口探测等业务逻辑仍留在 RankInfoDetectClient。
class RootInfoReader {
public:
    HcclResult Read(const std::string& json, nlohmann::json& out) const;
    HcclResult ReadFile(const std::string& filePath, nlohmann::json& out) const;
    HcclResult Check(const nlohmann::json& config) const;
    HcclResult GetTopoFilePath(const nlohmann::json& config, std::string& out) const;
    HcclResult SelectLocalDevInfo(const nlohmann::json& config, u32 devPhyId, u32 rankId, nlohmann::json& out) const;
    HcclResult BuildLocalRankTableJson(const nlohmann::json& config, u32 rankSize, nlohmann::json& out) const;
    HcclResult SelectHostAddr(
        nlohmann::json& localDevInfoJson, const std::function<s32(const IpAddress&)>& probe, u32 devPhyId,
        u32 rankId) const;
    HcclResult AddPcieFallbackLevel0(
        nlohmann::json& localDevInfoJson, const std::string& netInstId, bool hasVnicIp, const IpAddress& vnicIp) const;
    HcclResult BuildLocalRankTable(
        const nlohmann::json& config, const nlohmann::json& localDevInfo, u32 rankSize, RankTableInfo& out) const;

private:
    // rootinfo 配置中 host 备用地址所在的最小 net_layer。
    static constexpr u32 kHostBackupAddrNetLayer = 3;
    // rootinfo 顶层配置字段，不在 ranktable 通用键表内。
    static constexpr const char* kTopoFilePathKey = "topo_file_path";

    bool IsLevel0Exist(const nlohmann::json& localDevInfoJson) const;
    HcclResult BuildHostAddrCandidates(const nlohmann::json& addrJson, std::vector<IpAddress>& candidates) const;
    HcclResult CollectLayer3AddrJsons(nlohmann::json& localDevInfoJson, std::vector<nlohmann::json*>& addrJsons) const;
    HcclResult SelectAvailableHostAddr(
        nlohmann::json& addrJson, const std::function<s32(const IpAddress&)>& probe, u32 devPhyId, u32 rankId) const;
    void UpdateSelectedHostAddr(
        nlohmann::json& addrJson, const std::vector<IpAddress>& candidates, std::size_t selectedIndex, u32 devPhyId,
        u32 rankId) const;
};

} // namespace Hccl

#endif // HCCLV2_ROOT_INFO_READER_H
