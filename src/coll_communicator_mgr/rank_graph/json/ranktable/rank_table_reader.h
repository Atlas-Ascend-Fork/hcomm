/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RANK_TABLE_READER_H
#define HCCLV2_RANK_TABLE_READER_H

#include <string>
#include <vector>
#include "json/common/json_parse_util.h"
#include "rank_table_source.h"
#include "rank_table_info.h"
#include "new_rank_info.h"
#include "rank_level_info.h"
#include "address_info.h"
#include "control_plane.h"

namespace Hccl {

// ranktable 域的 DOM Reader。rank/level/address/control_plane 的解析作为本类成员，
// 不再单独暴露 Reader 类；快速路径由 ReadStringFast 提供。
class RankTableReader {
public:
    explicit RankTableReader(RankTableSource source = RankTableSource::RANKTABLE);

    // DOM 文本入口：解析后不执行 RankTableInfo::Check()。
    HcclResult Read(const std::string& json, RankTableInfo& out) const;
    // 快速路径：单遍扫描，不构建 DOM，不执行 RankTableInfo::Check()。
    static HcclResult ReadStringFast(const std::string& json, RankTableInfo& out);
    HcclResult ReadJson(const nlohmann::json& json, RankTableInfo& out) const;

    // 子模型入口：供本类内部复用，也供单测直接验证。
    HcclResult ReadRank(const nlohmann::json& json, NewRankInfo& out) const;
    HcclResult ReadLevel(const nlohmann::json& json, RankLevelInfo& out) const;
    HcclResult ReadAddress(const nlohmann::json& json, AddressInfo& out) const;
    HcclResult ReadControlPlane(const nlohmann::json& json, ControlPlane& out) const;

    static HcclResult ParseAddrByType(const std::string& addrType, const std::string& address, IpAddress& ipAddress);
    static HcclResult ParseAddrByType(
        const std::string& addrType, const std::string& address, IpAddress& ipAddress, const std::string& addrField);
    static HcclResult ParseBackupAddrs(
        const nlohmann::json& backupAddrJson, const std::string& addrType, std::vector<IpAddress>& backupAddrs);
    static HcclResult EidToAddr(const std::string& address, IpAddress& out);

private:
    HcclResult ReadRankIdAndLocalId(const nlohmann::json& json, NewRankInfo& out) const;
    HcclResult ReadDeviceIdAndPort(const nlohmann::json& json, NewRankInfo& out) const;
    HcclResult ReadLevelList(const nlohmann::json& json, NewRankInfo& out) const;
    HcclResult ReadNetLayerInfo(const nlohmann::json& json, RankLevelInfo& out) const;
    HcclResult ReadRankAddrs(const nlohmann::json& json, RankLevelInfo& out) const;
    void BuildPortAddrMap(RankLevelInfo& out) const;
    HcclResult ReadAddrTypeAndAddr(const nlohmann::json& json, AddressInfo& out) const;
    HcclResult ReadPorts(const nlohmann::json& json, AddressInfo& out) const;
    HcclResult ReadBackupAddrs(const nlohmann::json& json, AddressInfo& out, const std::string& addrTypeStr) const;

    RankTableSource source_;
};

// ParseJsonString<T> 的定制入口：把子模型解析统一转发到 RankTableReader。
template <>
inline HcclResult ReadJson<RankTableInfo>(const nlohmann::json& json, RankTableInfo& info)
{
    RankTableReader reader;
    return reader.ReadJson(json, info);
}

template <>
inline HcclResult ReadJson<NewRankInfo>(const nlohmann::json& json, NewRankInfo& info)
{
    RankTableReader reader;
    return reader.ReadRank(json, info);
}

template <>
inline HcclResult ReadJson<RankLevelInfo>(const nlohmann::json& json, RankLevelInfo& info)
{
    RankTableReader reader;
    return reader.ReadLevel(json, info);
}

template <>
inline HcclResult ReadJson<AddressInfo>(const nlohmann::json& json, AddressInfo& info)
{
    RankTableReader reader;
    return reader.ReadAddress(json, info);
}

template <>
inline HcclResult ReadJson<ControlPlane>(const nlohmann::json& json, ControlPlane& info)
{
    RankTableReader reader;
    return reader.ReadControlPlane(json, info);
}

} // namespace Hccl

#endif // HCCLV2_RANK_TABLE_READER_H
