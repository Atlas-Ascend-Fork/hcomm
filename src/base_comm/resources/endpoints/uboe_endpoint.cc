/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "uboe_endpoint.h"
#include "log.h"
#include "hccl/hccl_res.h"
#include "ub_reged_mem_mgr.h"
#include "endpoint_remote_reged_mem_mgr.h"
#include "proc_reged_mem_mgr_cache.h"
#include "adapter_rts_common.h"
#include "rdma_handle_manager.h"
#include "mgr/endpoint_ctx_mgr.h"
#include "hcomm_res_mgr.h"
#include "../../common/orion_adpt_utils.h"
#include "dev_type_utils.h"

namespace hcomm {

UboeEndpoint::UboeEndpoint(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

UboeEndpoint::~UboeEndpoint() noexcept
{
    (void)ReleaseCache();
    (void)ReleaseEndpointCtx();
}

bool UboeEndpoint::IsCtxHandleValid() const
{
    if (ctxHandle_ == nullptr) {
        return false;
    }
    return Hccl::RdmaHandleManager::GetInstance().IsHandleValid(static_cast<Hccl::RdmaHandle>(ctxHandle_));
}

HcclResult UboeEndpoint::ReleaseEndpointCtx()
{
    if (endpointCtx_ == nullptr) {
        HCCL_WARNING("[UboeEndpoint][%s] endpointCtx_ is null, nothing to release", __func__);
        return HCCL_E_PTR;
    }
    // 先放掉自身引用，再触发 EndpointCtxMgr 移除
    EndpointCtxKey key = endpointCtx_->key;
    endpointCtx_.reset();
    HcommResMgr::GetInstance().GetDeviceResMgr(key.devPhyId).GetEndpointCtxMgr().Release(key);
    return HCCL_SUCCESS;
}

HcclResult
UboeEndpoint::AttachCache(const MemMgrCacheKey& key, const std::function<std::shared_ptr<LocalRegedMemMgr>()>& creator)
{
    cacheKey_ = key;
    cacheKeepAlive_ = ProcRegedMemMgrCache::GetHolder();
    // 进程级 mgr 只承担硬件注册与 alias 去重（跨 endpoint 复用）；本实例归属记录见 RecordLocalMemHandle
    localMemMgr_ = cacheKeepAlive_->GetOrCreate(cacheKey_, creator);
    if (localMemMgr_ == nullptr) {
        HCCL_ERROR("[UboeEndpoint][%s] localMemMgr_ is null", __func__);
        CHK_RET(ReleaseCache());
        return HCCL_E_INTERNAL;
    }
    remoteMemMgr_ = std::make_shared<EndpointRemoteRegedMemMgr>(ctxHandle_, ParseUbMemDesc, CreateUbRemoteBuffer);
    return HCCL_SUCCESS;
}

HcclResult UboeEndpoint::RecordLocalMemHandle(void* memHandle)
{
    return RecordLocalMemHandleImpl<UbRegedMemMgr, Hccl::LocalUbRmaBuffer>(
        memHandle, localMemMgr_, localMemMtx_, localMemRecords_);
}

HcclResult UboeEndpoint::RemoveLocalMemHandle(void* memHandle)
{
    return RemoveLocalMemHandleImpl<Hccl::LocalUbRmaBuffer>(memHandle, localMemMtx_, localMemRecords_);
}

HcclResult UboeEndpoint::GetAllLocalMemHandles(void** memHandles, uint32_t* memHandleNum)
{
    return GetAllLocalMemHandlesImpl<Hccl::LocalUbRmaBuffer>(localMemMtx_, localMemRecords_, memHandles, memHandleNum);
}

HcclResult UboeEndpoint::ReleaseCache()
{
    if (cacheKeepAlive_ == nullptr) {
        HCCL_WARNING("[UboeEndpoint][%s] cacheKeepAlive_ is null, nothing to release", __func__);
        return HCCL_E_PTR;
    }
    cacheKeepAlive_->Release(cacheKey_);
    cacheKeepAlive_.reset();
    return HCCL_SUCCESS;
}

HcclResult UboeEndpoint::Init()
{
    HCCL_INFO("[%s] localEndpoint protocol[%d]", __func__, endpointDesc_.protocol);

    CHK_RET(CheckDevTypeSupport(
        {DevType::DEV_TYPE_950, DevType::DEV_TYPE_960}, endpointDesc_.protocol, endpointDesc_.loc.locType));

    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr));

    s32 deviceLogicId;
    u32 devPhyId;
    CHK_RET(hrtGetDevice(&deviceLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(deviceLogicId, devPhyId));
    endpointDesc_.loc.device.devPhyId = devPhyId;

    Hccl::HccpHdcManager::GetInstance().Init(deviceLogicId);
    EndpointCtxKey ctxKey{devPhyId, COMM_PROTOCOL_UB_CTP, endpointDesc_.loc.locType, ipAddr};
    CHK_RET(HcommResMgr::GetInstance()
                .GetDeviceResMgr(ctxKey.devPhyId)
                .GetEndpointCtxMgr()
                .Acquire(ctxKey, true, endpointCtx_));
    CHK_PTR_NULL(endpointCtx_);
    ctxHandle_ = endpointCtx_->ctxHandle;
    HCCL_INFO(
        "%s success, devPhyId[%u], ipAddr[%s], ctxHandle[%p]", __func__, devPhyId, ipAddr.Describe().c_str(),
        ctxHandle_);

    MemMgrCacheKey key{devPhyId, COMM_PROTOCOL_UB_CTP, ipAddr, LocTypeToPortType(endpointDesc_.loc.locType)};
    auto createMgr = [this]() {
        return std::make_shared<UbRegedMemMgr>(ctxHandle_);
    };
    auto ret = AttachCache(key, createMgr);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[UboeEndpoint][%s] AttachCache failed, ret[%d]", __func__, ret);
        CHK_RET(ReleaseEndpointCtx());
        return ret;
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace hcomm
