/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RANK_TABLE_KEYS_H
#define HCCLV2_RANK_TABLE_KEYS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include "topo_common_types.h"

namespace Hccl {

// RankTable JSON 字段名集中表: 解析与查询一律查表取键, 不要在代码里散落字符串字面量。
// 枚举顺序必须与 kRankTableKeyEntries 一一对应(数量与逐项两条 static_assert 兜底), 新增字段时两处同步添加。
enum class RankTableKey : std::uint32_t {
    kVersion = 0,
    kDetour = 1,
    kRankCount = 2,
    kRankList = 3,
    kRankId = 4,
    kLocalId = 5,
    kReplacedLocalId = 6,
    kDeviceId = 7,
    kDevicePort = 8,
    kHostPort = 9,
    kLevelList = 10,
    kNetLayer = 11,
    kNetInstanceId = 12,
    kNetType = 13,
    kNetAttr = 14,
    kPcieFallback = 15,
    kRankAddrList = 16,
    kAddr = 17,
    kAddrType = 18,
    kPlaneId = 19,
    kPorts = 20,
    kListenPort = 21,
    kBackupAddrList = 22,
    kBackupAddr = 23,
    kControlPlane = 24,
    kCount
};

struct RankTableKeyEntry {
    RankTableKey key;
    const char* name;
    std::size_t len;
};

inline constexpr RankTableKeyEntry kRankTableKeyEntries[] = {
    {RankTableKey::kVersion, "version", sizeof("version") - 1},
    {RankTableKey::kDetour, "detour", sizeof("detour") - 1},
    {RankTableKey::kRankCount, "rank_count", sizeof("rank_count") - 1},
    {RankTableKey::kRankList, "rank_list", sizeof("rank_list") - 1},
    {RankTableKey::kRankId, "rank_id", sizeof("rank_id") - 1},
    {RankTableKey::kLocalId, "local_id", sizeof("local_id") - 1},
    {RankTableKey::kReplacedLocalId, "replaced_local_id", sizeof("replaced_local_id") - 1},
    {RankTableKey::kDeviceId, "device_id", sizeof("device_id") - 1},
    {RankTableKey::kDevicePort, "device_port", sizeof("device_port") - 1},
    {RankTableKey::kHostPort, "host_port", sizeof("host_port") - 1},
    {RankTableKey::kLevelList, "level_list", sizeof("level_list") - 1},
    {RankTableKey::kNetLayer, "net_layer", sizeof("net_layer") - 1},
    {RankTableKey::kNetInstanceId, "net_instance_id", sizeof("net_instance_id") - 1},
    {RankTableKey::kNetType, "net_type", sizeof("net_type") - 1},
    {RankTableKey::kNetAttr, "net_attr", sizeof("net_attr") - 1},
    {RankTableKey::kPcieFallback, "pcie_fallback", sizeof("pcie_fallback") - 1},
    {RankTableKey::kRankAddrList, "rank_addr_list", sizeof("rank_addr_list") - 1},
    {RankTableKey::kAddr, "addr", sizeof("addr") - 1},
    {RankTableKey::kAddrType, "addr_type", sizeof("addr_type") - 1},
    {RankTableKey::kPlaneId, "plane_id", sizeof("plane_id") - 1},
    {RankTableKey::kPorts, "ports", sizeof("ports") - 1},
    {RankTableKey::kListenPort, "listen_port", sizeof("listen_port") - 1},
    {RankTableKey::kBackupAddrList, "backup_addr_list", sizeof("backup_addr_list") - 1},
    {RankTableKey::kBackupAddr, "backup_addr", sizeof("backup_addr") - 1},
    {RankTableKey::kControlPlane, "control_plane", sizeof("control_plane") - 1},
};

// 逐项校验表项 key 与其下标对应的枚举值一致, 防止枚举中途插入/调整顺序时表项错位。
constexpr bool RankTableKeyEntriesBound()
{
    for (std::size_t i = 0; i < sizeof(kRankTableKeyEntries) / sizeof(kRankTableKeyEntries[0]); ++i) {
        if (kRankTableKeyEntries[i].key != static_cast<RankTableKey>(i)) {
            return false;
        }
    }
    return true;
}

static_assert(
    sizeof(kRankTableKeyEntries) / sizeof(kRankTableKeyEntries[0]) == static_cast<std::size_t>(RankTableKey::kCount),
    "RankTableKey 枚举与 kRankTableKeyEntries 表必须一一对应");
static_assert(RankTableKeyEntriesBound(), "kRankTableKeyEntries 每项 key 必须与其下标对应的 RankTableKey 枚举值一致");

// 枚举值连续，直接用下标查表；等价于以 RankTableKey 为键的哈希表，但没有哈希和动态分配开销。
class RankTableKeyTable {
public:
    constexpr const RankTableKeyEntry& operator[](RankTableKey key) const
    {
        return kRankTableKeyEntries[static_cast<std::size_t>(key)];
    }
};

inline constexpr RankTableKeyTable kRankTableKeys{};

// RankTable JSON 字符串枚举值到内部枚举的映射。
inline const std::unordered_map<std::string, AddrType> kStrToAddrType
    = {{"EID", AddrType::EID}, {"IPV4", AddrType::IPV4}, {"IPV6", AddrType::IPV6}};

inline const std::unordered_map<std::string, NetType> kStrToNetType
    = {{"1DMESH", NetType::MESH_1D},
       {"2DMESH", NetType::MESH_2D},
       {"A3_SERVER", NetType::A3_SERVER},
       {"A2_AX_SERVER", NetType::A2_AX_SERVER},
       {"TOPO_FILE_DESC", NetType::TOPO_FILE_DESC},
       {"CLOS", NetType::CLOS}};

} // namespace Hccl

#endif // HCCLV2_RANK_TABLE_KEYS_H
