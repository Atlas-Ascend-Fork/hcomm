/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_level_info.h"
#include <sstream>
#include <string>
#include <unordered_map>

namespace Hccl {
using namespace std;

string RankLevelInfo::Describe() const
{
    return StringFormat(
        "RankLevelInfo[net_layer=%u, net_instance_id=%s, netType=%s, rankAddrs size=%d]", netLayer, netInstId.c_str(),
        netType.Describe().c_str(), rankAddrs.size());
}

RankLevelInfo::RankLevelInfo(BinaryStream& binStream)
{
    binStream >> netLayer >> netInstId >> netAttr;
    u32 netTypeInt{0};
    binStream >> netTypeInt;
    netType = static_cast<NetType::Value>(netTypeInt);
    u32 pcieFallbackInt{0};
    binStream >> pcieFallbackInt;
    pcieFallback = (pcieFallbackInt != 0);
    size_t addrSize{0};
    binStream >> addrSize;
    HCCL_INFO(
        "[%s] net_layer[%u] net_instance_id[%s] netType[%s] addrs size[%u]", __func__, netLayer, netInstId.c_str(),
        netType.Describe().c_str(), rankAddrs.size());
    for (u32 i = 0; i < addrSize; i++) {
        AddressInfo addressInfo(binStream);
        rankAddrs.emplace_back(std::move(addressInfo));
    }

    for (auto& rankAddr : rankAddrs) {
        IpAddress ipAddress = rankAddr.addr;
        for (auto& port : rankAddr.ports) {
            if (portAddrMap.find(port) != portAddrMap.end()) {
                portAddrMap[port].push_back(ipAddress);
            } else {
                std::vector<IpAddress> ipAddrList;
                ipAddrList.push_back(ipAddress);
                portAddrMap[port] = ipAddrList;
            }
        }
    }
}

HcclResult RankLevelInfo::GetBinStream(BinaryStream& binStream) const
{
    binStream << netLayer << netInstId << netAttr << static_cast<u32>(netType);
    binStream << static_cast<u32>(pcieFallback);
    binStream << rankAddrs.size();
    HCCL_INFO(
        "[%s] net_layer[%u] net_instance_id[%s] netType[%s] addrs size[%u]", __func__, netLayer, netInstId.c_str(),
        netType.Describe().c_str(), rankAddrs.size());
    if (rankAddrs.size() == 0) {
        return HCCL_SUCCESS;
    }
    for (auto& rankAddr : rankAddrs) {
        CHK_RET(rankAddr.GetBinStream(binStream));
    }
    return HCCL_SUCCESS;
}
} // namespace Hccl
