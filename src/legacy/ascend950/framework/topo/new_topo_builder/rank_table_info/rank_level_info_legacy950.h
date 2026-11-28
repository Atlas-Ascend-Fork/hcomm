/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANK_LEVEL_INFO_LEGACY950_H
#define RANK_LEVEL_INFO_LEGACY950_H

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <regex>
#include "topo_common_types.h"
#include "address_info_legacy950.h"
// 两份.h 必须一起改：改本文件时同步改 coll_comm 侧
// src/coll_communicator_mgr/rank_graph/rank_table_info/rank_level_info.h。
// 两份归一化 namespace 后成员声明逐字一致（仅 namespace 名 Hccl/hccl + guard 名 + include 行
// （本侧 address_info_legacy950.h / coll_comm 侧 address_info.h，因 AddressInfo 同源整改）+ 同步注释不同），
// legacy950 侧无 .cc，仅作布局锁定声明（除内联空构造/析构外成员函数无 .cc 定义、调用将链接失败，生产实现只在
// coll_comm 侧 rank_level_info.cc，namespace hccl）；两份均经 Hccl:: 限定同源类型（NetType/IpAddress/
// BinaryStream 均定义在 namespace Hccl 的 topo_common_types.h，不随本类迁 ns），避免同 SO 跨 TU 声明分叉导致
// 序列化错位。
namespace Hccl {
constexpr unsigned int MIN_VALUE_NETLAYER = 0;
constexpr unsigned int MAX_VALUE_NETLAYER = 7;
constexpr unsigned int MIN_VALUE_NETID = 1;
constexpr unsigned int MAX_VALUE_NETID = 1024;
constexpr unsigned int MIN_VALUE_RANKADDR_SIZE = 0;
constexpr unsigned int MAX_VALUE_RANKADDR_SIZE = 24;
constexpr unsigned int MIN_VALUE_U32 = 0;

class RankLevelInfo {
public:
    RankLevelInfo() {};
    u32 netLayer{0};
    std::string netInstId;
    Hccl::NetType netType{Hccl::NetType::CLOS};
    std::string netAttr;
    // 无UB场景兜底标记：由rank_info_detect_client插入兜底level0时置true（JSON字段pcie_fallback），
    // 表示该层为合成的PCIe兜底层而非真实UB mesh层；正常层缺省false
    bool pcieFallback{false};
    std::vector<AddressInfo> rankAddrs;
    std::string Describe() const;
    std::map<std::string, std::vector<Hccl::IpAddress>> portAddrMap;
    explicit RankLevelInfo(Hccl::BinaryStream& binStream);
    HcclResult GetBinStream(Hccl::BinaryStream& binStream) const;
};

} // namespace Hccl

#endif // RANK_LEVEL_INFO_LEGACY950_H
