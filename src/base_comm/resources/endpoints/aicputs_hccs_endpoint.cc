/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_mem_defs.h"
#include "aicputs_hccs_endpoint.h"
#include "log.h"
#include "common/dev_type_utils_pub.h"
#include "net_dev/global_net_dev_manager.h"
#include "base_comm_node_accessors.h"
#include "hccs_reged_mem_mgr.h"

using namespace hccl;

namespace hcomm {
AicpuTsHccsEndpoint::AicpuTsHccsEndpoint(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

AicpuTsHccsEndpoint::~AicpuTsHccsEndpoint()
{
    try {
        // 析构顺序保持：先停止 GlobalNetDevMgr Server（子类 context 承载），再 UnRefNetDevCtx
        if (serverSocketContext_.has_value()) {
            (void)serverSocketContext_->ServerSocketStopListen(serverPort_);
        }
    } catch (...) {
    }

    regedMemMgr_ = nullptr;

    try {
        if (netDevCtx_ != nullptr) {
            (void)GetGlobalNetDevMgr(endpointDesc_.loc.device.devPhyId)
                .UnRefNetDevCtx(NicType::VNIC_TYPE, devIpAddr_, serverPort_);
            netDevCtx_ = nullptr;
        }
    } catch (...) {
    }
}

HcclResult AicpuTsHccsEndpoint::Init()
{
    HCCL_INFO(
        "[%s]localEndpoint protocol[%d], type[%d], id[%u] locType[%d], devPhyId[%u], serverIdx[%u], "
        "superDevId[%u], superPodIdx[%u]",
        __func__, endpointDesc_.protocol, endpointDesc_.commAddr.type, endpointDesc_.commAddr.id,
        endpointDesc_.loc.locType, endpointDesc_.loc.device.devPhyId, endpointDesc_.loc.device.serverIdx,
        endpointDesc_.loc.device.superDevId, endpointDesc_.loc.device.superPodIdx);

    if (endpointDesc_.loc.locType != ENDPOINT_LOC_TYPE_DEVICE) {
        HCCL_RUN_WARNING(
            "[AicpuTsHccsEndpoint][%s] locType[%s] only support[%s]", __func__,
            GetEndpointLocTypeStr(endpointDesc_.loc.locType), GetEndpointLocTypeStr(ENDPOINT_LOC_TYPE_DEVICE));
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_RET(CheckDevTypeSupport(
        {DevType::DEV_TYPE_910B, DevType::DEV_TYPE_910_93}, endpointDesc_.protocol, endpointDesc_.loc.locType));

    serverSocketContext_.emplace(endpointDesc_.loc.device.devPhyId, serverPort_);

    u32 devPhyId = endpointDesc_.loc.device.devPhyId;
    uint32_t superDevId = endpointDesc_.loc.device.superDevId;
    CHK_RET(GlobalNetDevMgr::GetDeviceVnicIP(devPhyId, superDevId, devIpAddr_));
    HCCL_INFO(
        "[AicpuTsHccsEndpoint]devPhyId[%u] superDevId[%u] devIpAddr_[%s] ", devPhyId, superDevId,
        devIpAddr_.GetReadableAddress());

    CHK_RET(GetGlobalNetDevMgr(endpointDesc_.loc.device.devPhyId)
                .RefNetDevCtx(NicType::VNIC_TYPE, devIpAddr_, serverPort_, netDevCtx_));
    EXCEPTION_CATCH(regedMemMgr_ = std::make_shared<HccsRegedMemMgr>(netDevCtx_), return HCCL_E_PARA);
    // 远端接口经 Forwarder 转发到同一组合 mgr（mgr 本身只继承 LocalRegedMemMgr）
    EXCEPTION_CATCH(
        remoteForwarder_ = std::make_shared<RemoteRegedMemMgrForwarder>(
            [this](const void* memDesc, uint32_t descLen, HcommMem* outMem) {
                return regedMemMgr_->MemoryImport(memDesc, descLen, outMem);
            },
            [this](const void* memDesc, uint32_t descLen) {
                return regedMemMgr_->MemoryUnimport(memDesc, descLen);
            }),
        return HCCL_E_PTR);
    return HCCL_SUCCESS;
}
} // namespace hcomm
