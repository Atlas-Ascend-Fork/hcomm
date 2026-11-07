/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_QUEUE_COPY_H
#define DPU_QUEUE_COPY_H

#include <cstdint>
#include <type_traits>
#include "securec.h"

/**
 * @note SQ/CQ 元素落格与出格的拷贝：SQE/CQE 每格 64B，走 securec 的 memcpy_s 要担一次参数校验，
 *       比几条定长块搬移还贵，故小块直接整体赋值；上限只对元素变宽的情形生效，超过才回到带边界校验的路径。
 */
namespace hccl {

constexpr uint32_t DPU_ELEM_COPY_LIMIT = 1024U;

// 返回 false 只可能出在超限的慢路径上；两参同型定长，快路径不会失败
template <typename T>
inline bool CopyQueueElem(T& dst, const T& src)
{
    static_assert(std::is_trivially_copyable<T>::value, "queue element must be trivially copyable");
    if constexpr (sizeof(T) <= DPU_ELEM_COPY_LIMIT) {
        dst = src;
        return true;
    } else {
        return memcpy_s(&dst, sizeof(dst), &src, sizeof(src)) == EOK;
    }
}

} // namespace hccl

#endif // DPU_QUEUE_COPY_H
