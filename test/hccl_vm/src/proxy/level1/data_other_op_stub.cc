/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Description: 数据面其他操作打桩函数（北向劫持）
 *              BatchMode / Comm 资源管理接口，作为 Checker 的输入。
 * Create: 2026-07-20
 */

#define HCCL_VM_MODULE "OTHER_OP"

#include <cstdint>

#include "hccl_proxy_common.h"
#include "level1_proxy_common.h"
#include "sim_log.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 启动批处理模式（数据面接口）。
 *
 * 将多个通信操作打包成一个批次执行，以提高执行效率。
 * 当前 Checker 桩实现不支持 batch 缓存，直接返回成功。
 *
 * @param batchTag 输入：批次标签字符串，用于标识本次批处理。
 * @return int32_t 始终返回 0（成功）。
 */
int32_t HcommBatchModeStart(const char* batchTag)
{
    HCCL_VM_INFO(
        "{}: batchTag={}, batch cache not supported, return success", __func__, batchTag ? batchTag : "(null)");
    return 0;
}

/**
 * @brief 结束批处理模式（数据面接口）。
 *
 * 结束由 HcommBatchModeStart 启动的批处理，提交并执行批次中的所有操作。
 * 当前 Checker 桩实现不支持 batch 缓存，直接返回成功。
 *
 * @param batchTag 输入：批次标签字符串，需与 HcommBatchModeStart 传入的一致。
 * @return int32_t 始终返回 0（成功）。
 */
int32_t HcommBatchModeEnd(const char* batchTag)
{
    HCCL_VM_INFO(
        "{}: batchTag={}, batch cache not supported, return success", __func__, batchTag ? batchTag : "(null)");
    return 0;
}

/**
 * @brief 释放通信域资源（数据面接口）。
 *
 * 释放指定通信域占用的底层资源，使该通信域句柄失效。
 *
 * @note A5 平台不支持此接口，始终返回 HCCL_E_NOT_SUPPORT。
 *
 * @param commId 输入：通信域标识字符串。
 * @return int32_t 始终返回非 0（HCCL_E_NOT_SUPPORT）。
 */
int32_t HcommReleaseComm(const char* commId)
{
    (void)commId;
    HCCL_VM_ERROR("{}: not supported on A5 platform", __func__);
    return static_cast<int32_t>(HCCL_E_NOT_SUPPORT);
}

/**
 * @brief 获取通信域资源（数据面接口）。
 *
 * 根据通信域标识获取对应的通信域上下文，用于在数据面操作前绑定通信域。
 *
 * @note A5 平台不支持此接口，始终返回 HCCL_E_NOT_SUPPORT。
 *
 * @param commId 输入：通信域标识字符串。
 * @return int32_t 始终返回非 0（HCCL_E_NOT_SUPPORT）。
 */
int32_t HcommAcquireComm(const char* commId)
{
    (void)commId;
    HCCL_VM_ERROR("{}: not supported on A5 platform", __func__);
    return static_cast<int32_t>(HCCL_E_NOT_SUPPORT);
}

#ifdef __cplusplus
}
#endif
