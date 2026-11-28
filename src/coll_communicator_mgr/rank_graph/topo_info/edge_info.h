/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EDGE_INFO_H
#define EDGE_INFO_H

#include <string>
#include <unordered_map>
#include <set>
#include "types.h"
#include "topo_common_types.h"
#include "ip_address.h"
// 两份.h 必须一起改：改本文件时同步改 legacy950 侧
// src/legacy/ascend950/framework/topo/new_topo_builder/topo_info/edge_info_legacy950.h。
// 两份归一化 namespace 后声明逐字一致（仅 namespace 名 Hccl/hccl + guard 名 + 同步注释不同），
// legacy950 侧无 .cc，仅作布局锁定声明（除内联空构造/析构外成员函数无 .cc 定义、调用将链接失败，生产实现只在
// coll_comm 侧 edge_info.cc，namespace hccl）；两份均经 Hccl:: 限定同源类型（LinkType/TopoType/LinkProtocol/
// AddrPosition/BinaryStream 均定义在 namespace Hccl 的 topo_common_types.h 等，不随本类迁 ns），避免同 SO
// 跨 TU 声明分叉导致序列化错位。
namespace hccl {

constexpr unsigned int MAX_VALUE_LEVEL = 7;
constexpr u32 PORT_MAX_LENGTH = 32;
constexpr u32 MAX_PORTS_SIZE = 64;

class EdgeInfo {
public:
    EdgeInfo() {};
    EdgeInfo(Hccl::BinaryStream& binaryStream);
    ~EdgeInfo() {};

    u32 netLayer{0};
    Hccl::LinkType linkType;
    Hccl::TopoType topoType{Hccl::TopoType::CLOS};
    u32 topoInstId{0};
    std::set<Hccl::LinkProtocol> protocols;

    u32 localA{0};
    std::set<std::string> localAPorts;
    u32 localB{0};
    std::set<std::string> localBPorts;
    Hccl::AddrPosition position;

    bool operator==(const EdgeInfo& other) const;
    Hccl::LinkProtocol GetLinkProtocol(std::string str) const;
    Hccl::TopoType GetTopoType(std::string topoTypeStr) const;
    Hccl::LinkType GetLinkType(std::string linkTypeStr) const;
    Hccl::AddrPosition GetAddrPosition(std::string str) const;
    std::string Describe() const;
    void GetBinStream(Hccl::BinaryStream& binaryStream) const;

private:
    const static std::unordered_map<std::string, Hccl::LinkProtocol> strToLinkProtocol;
    const static std::unordered_map<std::string, Hccl::TopoType> strToTopoType;
    const static std::unordered_map<std::string, Hccl::LinkType> strToLinkType;
    const static std::unordered_map<std::string, Hccl::AddrPosition> strToAddrPosition;
    bool CompareEndpoints(const EdgeInfo& other) const;
    std::string DescribePorts(std::set<std::string> ports) const;
};
} // namespace hccl

#endif // EDGE_INFO_H
