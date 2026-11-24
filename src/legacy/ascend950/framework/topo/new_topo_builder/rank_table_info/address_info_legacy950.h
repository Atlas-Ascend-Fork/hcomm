/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADDRESS_INFO_LEGACY950_H
#define ADDRESS_INFO_LEGACY950_H

// 两份.h 必须一起改：改本文件时同步改 coll_comm 侧
// src/coll_communicator_mgr/rank_graph/rank_table_info/address_info.h。

#include <set>
#include <string>
#include <vector>
#include "ip_address.h"
#include "topo_common_types.h"

namespace Hccl {

constexpr unsigned int MIN_VALUE_PLANEID = 0;
constexpr unsigned int MAX_VALUE_PLANEID = 1024;
constexpr unsigned int MIN_VALUE_PORT = 1;
constexpr unsigned int MAX_VALUE_PORT
    = 80; // A7场景单device最大端口数；A2/A3/A5实际配置不超16、A6不超20，此处统一放宽不做devType分档
constexpr unsigned int MIN_VALUE_PORT_LENGTH = 1;
constexpr unsigned int MAX_VALUE_PORT_LENGTH = 32;
constexpr unsigned int MIN_VALUE_ADDR_LENGRH = 1;
constexpr unsigned int MAX_VALUE_ADDR_LENGRH = 256;
constexpr unsigned int MAX_VALUE_BACKUP_ADDR_SIZE = 16;

class AddressInfo {
public:
    AddressInfo() {};
    ~AddressInfo() {};
    // 自定义析构会抑制隐式移动构造, 显式默认化拷贝/移动(移动避免深拷贝)
    AddressInfo(const AddressInfo&) = default;
    AddressInfo& operator=(const AddressInfo&) = default;
    AddressInfo(AddressInfo&&) = default;
    AddressInfo& operator=(AddressInfo&&) = default;

    Hccl::IpAddress addr;
    std::vector<Hccl::IpAddress> backupAddrs;
    u32 socketPort_{0}; // socket监听端口
    Hccl::AddrType addrType{Hccl::AddrType::EID};
    std::set<std::string> ports; // 网口标记
    std::string planeId{"0"};
    explicit AddressInfo(Hccl::BinaryStream& binStream);
    HcclResult GetBinStream(Hccl::BinaryStream& binStream) const;
    std::string Describe() const;
};

} // namespace Hccl

#endif // ADDRESS_INFO_LEGACY950_H
