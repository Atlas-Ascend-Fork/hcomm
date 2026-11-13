/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include "control_plane.h"
#include "topo_common_types.h"
namespace Hccl {
using namespace std;

std::string ControlPlane::Describe() const
{
    return StringFormat(
        "ControlPlane[addrType=%s, addr=%s, listenPort=%u]", addrType.Describe().c_str(), addr.Describe().c_str(),
        listenPort);
}

void ControlPlane::GetBinStream(BinaryStream& binStream) const
{
    binStream << static_cast<u32>(addrType) << listenPort;
    addr.GetBinStream(binStream);
}

ControlPlane::ControlPlane(BinaryStream& binStream)
{
    u32 addrTypeInt{0};
    binStream >> addrTypeInt;
    addrType = static_cast<AddrType::Value>(addrTypeInt);
    binStream >> listenPort;
    IpAddress address(binStream);
    addr = address;
}

} // namespace Hccl
