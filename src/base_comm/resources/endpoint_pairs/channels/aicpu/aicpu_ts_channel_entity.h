/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_CHANNEL_ENTITY_H
#define AICPU_TS_CHANNEL_ENTITY_H

#include "common/cast_utils_pub.h"
#include <memory>
#include <mutex>
#include <vector>
#include "hccl/hccl_types.h"
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"
#include "hcomm_res_entity_defs.h"
#include "../i_channel_entity.h"
#include "mem_device_pub.h"
#include "resources/comm_engine_res/aicpu_comm_engine_bin_proc.h"

static const uint32_t HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD = 0x0fcf0f4fU;
static const uint32_t HCOMM_AICPU_CHANNEL_CTX_VERSION = 0U;

using HcommAicpuChannelCtx = struct {
    CommAbiHeader abiHeader{HCOMM_AICPU_CHANNEL_CTX_VERSION, HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD, 0, 0};
    CommEngine engine{COMM_ENGINE_RESERVED};
    CommProtocol protocol{COMM_PROTOCOL_RESERVED};
    void* deviceChannel{nullptr};
};

/**
 * @brief AICPU 专用辅助类，持有 ctx 内存等 AICPU 专属资源，提供 AICPU 批量建链状态机接口。
 */
class AicpuTsChannelEntity : public hcomm::IChannelEntity {
public:
    // === per-channel 实例方法 ===
    HcclResult PreAllocCtx(ChannelHandle& outHandle) override
    {
        ctxMem_ = std::make_shared<hccl::DeviceMem>(hccl::DeviceMem::alloc(sizeof(HcommAicpuChannelCtx)));
        CHK_PTR_NULL(ctxMem_->ptr());
        outHandle = ReinterpretAs<ChannelHandle>(ctxMem_->ptr());
        HCCL_INFO("[AicpuTsChannelEntity::%s] pre-alloc ctx success, ctxPtr[%p]", __func__, ctxMem_->ptr());
        return HCCL_SUCCESS;
    }
    void* GetCtxPtr() const override { return ctxMem_ ? ctxMem_->ptr() : nullptr; }
    void SetCtxMem(std::shared_ptr<hccl::DeviceMem> mem) { ctxMem_ = std::move(mem); }

    // === 批量静态方法 ===
    static HcclResult HandleStatus(
        const ChannelHandle* channelList, uint32_t listNum, CommEngine engine, const HcommChannelDesc* channelDescs,
        const std::vector<int32_t>& linkStatusList, int32_t* statusList);
    // 兜底初始化：不查链路状态，对未就绪 device 实体补发 init kernel（全复用批次跳过轮询时使用）
    static HcclResult EnsureDeviceReady(const ChannelHandle* channelList, uint32_t listNum);
    static HcclResult PreAllocChannels(
        ChannelHandle* targetChannels, ChannelHandle* userChannels, HcommChannelDesc* channelDescs,
        uint32_t channelNum);
    static HcclResult TryFillCtxList(
        ChannelHandle* hostChannelHandles, uint32_t listNum, const hccl::DeviceMem& deviceChannelList,
        void*& outCtxList, bool& isCtxMode);

private:
    static HcclResult LaunchKernel(
        const ChannelHandle* channelList, uint32_t listNum, CommEngine engine, const HcommChannelDesc* channelDescs,
        aclrtBinHandle binHandle);

    std::shared_ptr<hccl::DeviceMem> ctxMem_;
};

namespace hcomm {
class AicpuTsRoceHostLocChannel;

/**
 * @brief RoCE v2 协议的 ChannelEntity 派生类，承接 RoCE v2 通道设备侧实体的构建/预分配/填充流程。
 */
class AicpuTsRoceV2ChannelEntity : public AicpuTsChannelEntity {
public:
    explicit AicpuTsRoceV2ChannelEntity(AicpuTsRoceHostLocChannel& owner);

    HcclResult BuildChannelEntityToDevice(void** devChannelPtr) override;
    HcclResult PreAllocChannelEntityToDevice(void** devChannelPtr) override;
    HcclResult FillChannelEntityToDevice() override;

private:
    AicpuTsRoceHostLocChannel& owner_;
};
} // namespace hcomm

inline HcclResult UnwrapChannelHandle(ChannelHandle& handle)
{
    if (handle == 0) {
        HCCL_ERROR("[AicpuTsChannelEntity::%s] handle is 0.", __func__);
        return HCCL_E_PTR;
    }

    const auto* entity = ReinterpretAs<const ChannelEntity*>(static_cast<uintptr_t>(handle));
    if (entity->abiHeader.magicWord != HCOMM_CHANNEL_ENTITY_MAGIC_WORD) {
        const auto* ctx = ReinterpretAs<const HcommAicpuChannelCtx*>(static_cast<uintptr_t>(handle));
        if (ctx->abiHeader.version != HCOMM_AICPU_CHANNEL_CTX_VERSION
            || ctx->abiHeader.magicWord != HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD) {
            return HCCL_SUCCESS;
        }
        if (ctx->deviceChannel == nullptr) {
            HCCL_ERROR("[AicpuTsChannelEntity::%s] ctx deviceChannel is null.", __func__);
            return HCCL_E_INTERNAL;
        }
        handle = ReinterpretAs<ChannelHandle>(ctx->deviceChannel);
        return HCCL_SUCCESS;
    }

    if (entity->abiHeader.version != HCOMM_CHANNEL_ENTITY_VERSION || entity->abiHeader.size < sizeof(ChannelEntity)) {
        HCCL_ERROR(
            "[AicpuTsChannelEntity::%s] invalid channel entity ABI, version[%u], size[%u].", __func__,
            entity->abiHeader.version, entity->abiHeader.size);
        return HCCL_E_INTERNAL;
    }
    if (entity->compatChannelHandle == 0) {
        HCCL_ERROR("[AicpuTsChannelEntity::%s] channel entity compat handle is null.", __func__);
        return HCCL_E_INTERNAL;
    }
    handle = entity->compatChannelHandle;
    return HCCL_SUCCESS;
}

#endif // AICPU_TS_CHANNEL_ENTITY_H
