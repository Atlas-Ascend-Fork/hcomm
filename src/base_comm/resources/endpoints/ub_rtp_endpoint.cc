/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_rtp_endpoint.h"
#include "log.h"
#include "hccl/hccl_res.h"
#include "ub_reged_mem_mgr.h"
#include "endpoint_remote_reged_mem_mgr.h"
#include "base_comm_node_accessors.h"
#include "adapter_rts_common.h"
#include "rdma_handle_manager.h"
#include "mgr/endpoint_ctx_mgr.h"
#include "common/orion_adpt_utils_pub.h"
#include "common/dev_type_utils_pub.h"

namespace hcomm {

UbRtpEndpoint::UbRtpEndpoint(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

UbRtpEndpoint::~UbRtpEndpoint() noexcept { (void)ReleaseEndpointCtx(); }

bool UbRtpEndpoint::IsCtxHandleValid() const
{
    if (ctxHandle_ == nullptr) {
        return false;
    }
    return Hccl::RdmaHandleManager::GetInstance().IsHandleValid(static_cast<Hccl::RdmaHandle>(ctxHandle_));
}

CommQueueContext* UbRtpEndpoint::GetCommQueueContext()
{
    std::call_once(jettyContextOnce_, [this] {
        jettyContext_ = std::make_unique<JettyContext>();
    });
    return jettyContext_.get();
}

HcclResult UbRtpEndpoint::ReleaseEndpointCtx()
{
    if (endpointCtx_ == nullptr) {
        HCCL_WARNING("[UbRtpEndpoint][%s] endpointCtx_ is null, nothing to release", __func__);
        return HCCL_E_PTR;
    }
    // 先放掉自身引用，再触发 EndpointCtxMgr 移除
    EndpointCtxKey key = endpointCtx_->key;
    endpointCtx_.reset();
    GetEndpointCtxMgr(key.devPhyId).Release(key);
    return HCCL_SUCCESS;
}

HcclResult UbRtpEndpoint::Init()
{
    HCCL_INFO("[%s] localEndpoint protocol[%d]", __func__, endpointDesc_.protocol);

    CHK_RET(CheckDevTypeSupport(
        {DevType::DEV_TYPE_950, DevType::DEV_TYPE_960}, endpointDesc_.protocol, endpointDesc_.loc.locType));

    // UB_RTP 直接从 EID type CommAddr 获取地址，不做 IP→EID 转换
    Hccl::IpAddress eidAddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, eidAddr));

    s32 deviceLogicId;
    u32 devPhyId;
    CHK_RET(hrtGetDevice(&deviceLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId), devPhyId));
    endpointDesc_.loc.device.devPhyId = devPhyId;

    Hccl::HccpHdcManager::GetInstance().Init(deviceLogicId);
    EndpointCtxKey ctxKey{devPhyId, COMM_PROTOCOL_UB_CTP, endpointDesc_.loc.locType, eidAddr};
    CHK_RET(GetEndpointCtxMgr(ctxKey.devPhyId).Acquire(ctxKey, false, endpointCtx_));
    CHK_PTR_NULL(endpointCtx_);
    ctxHandle_ = endpointCtx_->ctxHandle;
    HCCL_INFO(
        "%s success, devPhyId[%u], eidAddr[%s], ctxHandle[%p]", __func__, devPhyId, eidAddr.Describe().c_str(),
        ctxHandle_);

    // ub_rtp 的本端/远端内存 mgr 均为实例独有，不走进程级缓存
    EXCEPTION_CATCH(localMemMgr_ = std::make_shared<UbRegedMemMgr>(ctxHandle_), {
        CHK_RET(ReleaseEndpointCtx());
        return HCCL_E_INTERNAL;
    });
    EXCEPTION_CATCH(
        remoteMemMgr_ = std::make_shared<EndpointRemoteRegedMemMgr>(ctxHandle_, ParseUbMemDesc, CreateUbRemoteBuffer), {
            CHK_RET(ReleaseEndpointCtx());
            return HCCL_E_INTERNAL;
        });

    // 接入真实监听上下文（替代历史 no-op 语义）：承载 HcommEndpointGetListenPort（惰性创建监听并
    // 分配动态端口）与 StartListen/StopListen。devPhyId 已在上方解析写入，构造需晚于该步骤。
    // 析构时仅释放 GetListenPort 惰性创建的监听引用；StartListen 创建的监听需配对 StopListen 释放。
    EXCEPTION_CATCH(
        serverSocketContext_.emplace(
            Hccl::ConnectProtoType::UB, endpointDesc_.loc.device.devPhyId, endpointDesc_.loc.locType,
            endpointDesc_.commAddr),
        {
            CHK_RET(ReleaseEndpointCtx());
            return HCCL_E_INTERNAL;
        });

    return HcclResult::HCCL_SUCCESS;
}

} // namespace hcomm
