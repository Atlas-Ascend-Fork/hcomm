/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEV_EP_MEM_TABLE_H
#define DEV_EP_MEM_TABLE_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include "endpoint_pair/channel/channel_param_pub.h"

namespace Hccl {
class DevEpMemTable {
public:
    // ABI 头（首成员）：kernel 下发侧与查找侧校验，防悬垂/错指针写花 AICPU 堆
    struct AbiHeader {
        uint32_t magic{};
        uint32_t version{};
        uint32_t entryNum{};
        uint32_t rsv{};
    };
    struct EpMemKey {
        uint64_t addr{};
        uint64_t size{};
        bool operator<(const EpMemKey& other) const
        {
            return addr < other.addr || (addr == other.addr && size < other.size);
        }
    };
    struct EpMemEntry {
        LocUbBufLite info{};
        uint32_t ref{1};
    };

    DevEpMemTable();
    ~DevEpMemTable();

    HcclResult Update(const LocUbBufLite& entry);
    HcclResult Remove(uint64_t addr, uint64_t size);
    bool Find(uintptr_t addr, size_t size, LocUbBufLite& out) const;
    bool FindFirst(LocUbBufLite& out) const;
    bool CheckAbi() const;

    AbiHeader abi{};

private:
    using Table = std::map<EpMemKey, EpMemEntry>;
    using Snapshot = const Table;
    static std::shared_ptr<Snapshot> EmptySnapshot();

    std::mutex wMtx_;
    std::shared_ptr<Snapshot> published_{};
};
} // namespace Hccl
#endif // DEV_EP_MEM_TABLE_H
