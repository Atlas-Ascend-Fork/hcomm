/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONTROL_PLANE_H
#define CONTROL_PLANE_H

#include <string>
#include "ip_address.h"
#include "topo_common_types.h"

// 两份.h 必须一起改：改本文件时同步改 legacy950 侧
// src/legacy/ascend950/framework/topo/new_topo_builder/rank_table_info/control_plane_legacy950.h。
namespace hccl {
constexpr unsigned int MIN_VALUE_LISTENPORT = 1;
constexpr unsigned int MAX_VALUE_LISTENPORT = 65536;
constexpr unsigned int MIN_VALUE_ADDR = 1;
constexpr unsigned int MAX_VALUE_ADDR = 256;
class ControlPlane {
public:
    ControlPlane() {};
    Hccl::AddrType addrType;
    Hccl::IpAddress addr;
    u32 listenPort{0};
    void GetBinStream(Hccl::BinaryStream& binaryStream) const;
    explicit ControlPlane(Hccl::BinaryStream& binStream);
    std::string Describe() const;
};

} // namespace hccl

#endif // CONTROL_PLANE_H
