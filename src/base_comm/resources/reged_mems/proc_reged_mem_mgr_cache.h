/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PROC_REGED_MEM_MGR_CACHE_H
#define PROC_REGED_MEM_MGR_CACHE_H

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "reged_mem_mgr.h"
#include "ip_address.h"
#include "port.h"
#include "hcomm_res_defs.h"

namespace hcomm {

struct MemMgrCacheKey {
    u32 devPhyId{0};
    CommProtocol protocol{COMM_PROTOCOL_ROCE};
    Hccl::IpAddress ip{};
    Hccl::PortDeploymentType portType{Hccl::PortDeploymentType::HOST_NET};

    bool operator==(const MemMgrCacheKey& other) const
    {
        return devPhyId == other.devPhyId && protocol == other.protocol && ip == other.ip && portType == other.portType;
    }
};

struct MemMgrCacheKeyHash {
    size_t operator()(const MemMgrCacheKey& k) const
    {
        return Hccl::HashCombine({
            std::hash<u32>{}(k.devPhyId),
            std::hash<int>{}(static_cast<int>(k.protocol)),
            std::hash<Hccl::IpAddress>{}(k.ip),
            std::hash<int>{}(static_cast<int>(k.portType)),
        });
    }
};

// 由 endpointDesc.loc.locType 推导 PortDeploymentType
inline Hccl::PortDeploymentType LocTypeToPortType(EndpointLocType locType)
{
    return (locType == ENDPOINT_LOC_TYPE_DEVICE) ? Hccl::PortDeploymentType::DEV_NET :
                                                   Hccl::PortDeploymentType::HOST_NET;
}

struct MemMgrEntry {
    // 缓存的进程级 mgr 只管本端内存（注册/注销/导出），故按 LocalRegedMemMgr 持有
    std::shared_ptr<LocalRegedMemMgr> mgrPtr{nullptr};
    u64 refCount{0};
};

/**
 * @note 进程级本端注册内存复用缓存：同网卡跨 EndpointHandle 复用（远端内存不进缓存）。
 *       EndpointMgr 值成员、进程唯一，声明序保证晚于 endpoint 死（机制见 README R7）。
 *       用法：Init 存住别名再使用；析构经持有的别名 Release，勿再取别名。
 */
class ProcRegedMemMgrCache {
public:
    ProcRegedMemMgrCache() = default;
    ~ProcRegedMemMgrCache() = default;
    // hit: refCount++ 返已有 shared_ptr; miss: 调 creator() 建实例 insert refCount=1
    std::shared_ptr<LocalRegedMemMgr>
    GetOrCreate(const MemMgrCacheKey& key, std::function<std::shared_ptr<LocalRegedMemMgr>()> creator);

    // refCount--, 归 0 则 erase cacheMap_ 条目
    void Release(const MemMgrCacheKey& key);

    ProcRegedMemMgrCache(const ProcRegedMemMgrCache&) = delete;
    ProcRegedMemMgrCache& operator=(const ProcRegedMemMgrCache&) = delete;

private:
    std::mutex mtx_;
    std::unordered_map<MemMgrCacheKey, MemMgrEntry, MemMgrCacheKeyHash> cacheMap_;
};

} // namespace hcomm

#endif // PROC_REGED_MEM_MGR_CACHE_H
