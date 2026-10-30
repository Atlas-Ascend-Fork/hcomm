/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef THREAD_ENTITY_UTILS_H
#define THREAD_ENTITY_UTILS_H

#include "hcomm_res_entity_defs.h"
#include "hccl_common.h"
#include "common/cast_utils_pub.h"

namespace hccl {
// 仅在 device 上解包；host 必须通过句柄映射查询，不能解引用 device 地址。
inline HcclResult UnwrapThreadHandle(ThreadHandle& handle)
{
    CHK_PRT_RET(handle == 0, HCCL_ERROR("[%s] thread handle is null", __func__), HCCL_E_PTR);
    const auto* entity = ReinterpretAs<const HcommThreadEntity*>(handle);
    if (entity->abiHeader.magicWord != HCOMM_THREAD_ENTITY_MAGIC_WORD) {
        return HCCL_SUCCESS; // 兼容内部传递的原始 Thread 对象地址
    }
    CHK_PRT_RET(
        entity->abiHeader.version != HCOMM_THREAD_ENTITY_VERSION || entity->abiHeader.size < sizeof(HcommThreadEntity),
        HCCL_ERROR("[%s] invalid thread entity ABI", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        entity->compatThreadHandle == 0, HCCL_ERROR("[%s] thread entity is not initialized", __func__),
        HCCL_E_INTERNAL);
    handle = entity->compatThreadHandle;
    return HCCL_SUCCESS;
}
} // namespace hccl
#endif // THREAD_ENTITY_UTILS_H
