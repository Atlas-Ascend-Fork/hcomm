/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_mem_endpoint.h"
#include "log.h"
#include "hccl/hccl_res.h"
#include "adapter_rts_common.h"
#include "common/dev_type_utils_pub.h"
#include "ub_mem_reged_mem_mgr.h"
#include "proc_reged_mem_mgr_cache.h"
#include "base_comm_node_accessors.h"
#include "hccl_mem_defs.h"
#include "common/orion_adpt_utils_pub.h"

namespace hcomm {
UbMemEndpoint::UbMemEndpoint(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

UbMemEndpoint::~UbMemEndpoint() noexcept { (void)ReleaseCache(); }

HcclResult
UbMemEndpoint::AttachCache(const MemMgrCacheKey& key, const std::function<std::shared_ptr<LocalRegedMemMgr>()>& creator)
{
    cacheKey_ = key;
    cacheAlias_ = GetProcRegedMemMgrCacheHolder();
    // cache key 含 protocol 唯一决定具体 mgr 类型，static_pointer_cast 转换安全
    localMemMgr_ = std::static_pointer_cast<UbMemRegedMemMgr>(cacheAlias_->GetOrCreate(cacheKey_, creator));
    if (localMemMgr_ == nullptr) {
        HCCL_ERROR("[UbMemEndpoint][%s] localMemMgr_ is null", __func__);
        CHK_RET(ReleaseCache());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult UbMemEndpoint::ReleaseCache()
{
    if (cacheAlias_ == nullptr) {
        HCCL_WARNING("[UbMemEndpoint][%s] cacheAlias_ is null, nothing to release", __func__);
        return HCCL_E_PTR;
    }
    cacheAlias_->Release(cacheKey_);
    cacheAlias_.reset();
    return HCCL_SUCCESS;
}

HcclResult UbMemEndpoint::Init()
{
    CHK_RET(CheckDevTypeSupport(
        {DevType::DEV_TYPE_950, DevType::DEV_TYPE_960}, endpointDesc_.protocol, endpointDesc_.loc.locType));

    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr));

    s32 devId = 0;
    CHK_RET(hrtGetDevice(&devId));
    Hccl::HccpHdcManager::GetInstance().Init(devId);
    u32 devPhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(devId, devPhyId));

    if (endpointDesc_.loc.device.devPhyId != devPhyId) {
        HCCL_WARNING(
            "[UbMemEndpoint][%s] endpointDesc.loc.device.devPhyId[%u] incorrect", __func__,
            endpointDesc_.loc.device.devPhyId);
        endpointDesc_.loc.device.devPhyId = devPhyId;
    }

    MemMgrCacheKey key{devPhyId, COMM_PROTOCOL_UB_MEM, ipAddr, LocTypeToPortType(endpointDesc_.loc.locType)};
    auto createMgr = []() {
        return std::make_shared<UbMemRegedMemMgr>();
    };
    CHK_RET(AttachCache(key, createMgr));

    // ServerSocketContext：参照 UrmaEndpoint，以 devPhyId/locType/commAddr 构造，供 HcommEndpointStartListen 真实监听
    serverSocketContext_.emplace(
        Hccl::ConnectProtoType::UB, devPhyId, endpointDesc_.loc.locType, endpointDesc_.commAddr);

    return HcclResult::HCCL_SUCCESS;
}
} // namespace hcomm
