/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONTROL_PLANE_LEGACY950_H
#define CONTROL_PLANE_LEGACY950_H

// 两份.h 必须一起改：改本文件时同步改 coll_comm 侧
// src/coll_communicator_mgr/rank_graph/rank_table_info/control_plane.h。

#include <string>
#include "ip_address.h"
#include "topo_common_types.h"

namespace Hccl {
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

} // namespace Hccl

#endif // CONTROL_PLANE_LEGACY950_H
