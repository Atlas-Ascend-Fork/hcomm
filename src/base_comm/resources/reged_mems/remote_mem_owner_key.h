/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REMOTE_MEM_OWNER_KEY_H
#define REMOTE_MEM_OWNER_KEY_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include "endpoint_pair/endpoint_pair_pub.h"

namespace hcomm {

// 远端内存归属键：{endpointDesc, pid}，区分同一 endpointDesc 下不同远端进程导入的内存。
// desc 来自对端 memDesc 字节且 import/unimport 传同一份 memDesc，直接 memcmp 稳定。
struct RemoteMemOwnerKey {
    EndpointDesc desc{};
    uint64_t pid{0};

    bool operator==(const RemoteMemOwnerKey& other) const
    {
        return std::memcmp(&desc, &other.desc, sizeof(EndpointDesc)) == 0 && pid == other.pid;
    }
};

struct RemoteMemOwnerKeyHash {
    size_t operator()(const RemoteMemOwnerKey& key) const
    {
        size_t h1 = std::hash<EndpointDesc>{}(key.desc);
        size_t h2 = std::hash<uint64_t>{}(key.pid);
        // boost::hash_combine 组合，避免简单异或导致低位冲突
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

} // namespace hcomm
#endif // REMOTE_MEM_OWNER_KEY_H
