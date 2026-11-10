/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_CHANNEL_PUB_H
#define AICPU_CHANNEL_PUB_H

#include <cstdint>
#include "common/cast_utils_pub.h"
#include "hccl/hccl_types.h"
#include "hcomm_res_defs.h"
#include "hcomm_res_entity_defs.h"

namespace hcomm {

inline constexpr uint32_t HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD = 0x0fcf0f4fU;
inline constexpr uint32_t HCOMM_AICPU_CHANNEL_CTX_VERSION = 0U;

struct HcommAicpuChannelCtx {
    // size 由生产方补充完整；size 小于 sizeof(HcommAicpuChannelCtx) 时解包返回 HCCL_E_INTERNAL。
    // 注意：全零（尚未初始化）的 ctx 因 magicWord 不匹配走原样透传分支，调用方须自行确认句柄有效
    // （coll_comm 适配层经 IsValidLiteTransportHandle 在册校验，原语层须保证传入的 ctx 已初始化）。
    CommAbiHeader abiHeader{HCOMM_AICPU_CHANNEL_CTX_VERSION, HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD, 0, 0};
    // 预留 8 字节：维持 ctx ABI 布局与 deviceChannel 偏移不变，禁止随意删改。
    uint8_t reserved[8]{};
    void* deviceChannel{nullptr};
};

/**
 * @brief 把 AICPU ctx / ChannelEntity 句柄解包为下层 device 句柄；原始句柄原样返回。
 * @param[in,out] handle 待解包的通道句柄
 * @return HCCL_SUCCESS 表示解包完成或无需解包；HCCL_E_PTR/HCCL_E_INTERNAL 表示入参或 ABI 非法
 * @note 边界头不打日志，错误由调用方打印，避免边界头反向依赖 legacy 日志头。
 */
inline HcclResult UnwrapChannelHandle(ChannelHandle& handle)
{
    if (handle == 0) {
        return HCCL_E_PTR;
    }

    const auto* entity = ReinterpretAs<const ChannelEntity*>(static_cast<uintptr_t>(handle));
    if (entity->abiHeader.magicWord != HCOMM_CHANNEL_ENTITY_MAGIC_WORD) {
        const auto* ctx = ReinterpretAs<const HcommAicpuChannelCtx*>(static_cast<uintptr_t>(handle));
        if (ctx->abiHeader.version != HCOMM_AICPU_CHANNEL_CTX_VERSION
            || ctx->abiHeader.magicWord != HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD) {
            return HCCL_SUCCESS;
        }
        if (ctx->abiHeader.size < sizeof(HcommAicpuChannelCtx)) {
            return HCCL_E_INTERNAL;
        }
        if (ctx->deviceChannel == nullptr) {
            return HCCL_E_INTERNAL;
        }
        handle = ReinterpretAs<ChannelHandle>(ctx->deviceChannel);
        return HCCL_SUCCESS;
    }

    if (entity->abiHeader.version != HCOMM_CHANNEL_ENTITY_VERSION || entity->abiHeader.size < sizeof(ChannelEntity)) {
        return HCCL_E_INTERNAL;
    }
    if (entity->compatChannelHandle == 0) {
        return HCCL_E_INTERNAL;
    }
    handle = entity->compatChannelHandle;
    return HCCL_SUCCESS;
}

/**
 * @brief 在册查询：返回 handle 当前是否登记在 lite transport 表中。
 * @param[in] handle 待校验的通道句柄
 * @return true 表示查询时刻在册；false 表示未注册
 * @note 仅在查询期间持有注册表锁，不保证返回后句柄仍有效；查询与使用之间不得有并发销毁，
 *       该生命周期由调用方所在通信域的通道管理流程保证。
 */
bool IsValidLiteTransportHandle(ChannelHandle handle);

} // namespace hcomm

#endif // AICPU_CHANNEL_PUB_H
