/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NEW_RANK_INFO_H
#define NEW_RANK_INFO_H

// 两份.h 必须一起改：改本文件时同步改 legacy950 侧
// src/legacy/ascend950/framework/topo/new_topo_builder/rank_table_info/new_rank_info_legacy950.h。

#include <vector>
#include <string>

#include "types.h"
#include "rank_level_info.h"
#include "control_plane.h"
#include "orion_adapter_hccp.h"
namespace hccl {
constexpr unsigned int MAX_VALUE_DEVICEID = 512;
constexpr unsigned int DEFAULT_VALUE_TCPPORT = 16666;
constexpr unsigned int MAX_VALUE_TCPPORT = 65535;
constexpr unsigned int MIN_VALUE_TCPPORT = 1;
constexpr unsigned int MAX_LEVEL_lIST = 8;
const Hccl::IpAddress DEVICE_PORT_KEY_IPADDRESS = Hccl::IpAddress(); // 用于socketmanager的端口占用，vnic情况下使用
class NewRankInfo {
public:
    NewRankInfo() {};
    ~NewRankInfo() {};
    // 自定义析构会抑制隐式移动构造, 显式默认化拷贝/移动(移动避免深拷贝)
    NewRankInfo(const NewRankInfo&) = default;
    NewRankInfo& operator=(const NewRankInfo&) = default;
    NewRankInfo(NewRankInfo&&) = default;
    NewRankInfo& operator=(NewRankInfo&&) = default;

    u32 rankId{0};
    u32 deviceId{0};
    u32 localId{0};
    u32 replacedLocalId{0};
    u32 devicePort{DEFAULT_VALUE_TCPPORT};
    u32 hostPort{DEFAULT_VALUE_TCPPORT};
    std::vector<Hccl::RankLevelInfo> rankLevelInfos{};
    ControlPlane controlPlane{};
    Hccl::TlsStatus tlsStatus{Hccl::TlsStatus::UNKNOWN};
    Hccl::TlsStatus hostDpuTlsStatus{Hccl::TlsStatus::UNKNOWN};
    std::string logicSuperPodId{};
    std::string Describe() const;
    explicit NewRankInfo(Hccl::BinaryStream& binStream);
    HcclResult GetBinStream(bool isContainLoaId, Hccl::BinaryStream& binStream) const;
};

} // namespace hccl

#endif // NEW_RANK_INFO_H
