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
 * Description: 设备侧通信域管理打桩函数（level1 劫持）
 *              HcclCommGetStatus 接口桩实现。
 * Create: 2026-07-27
 */

#define HCCL_VM_MODULE "DEV_L1_COMM"

#include <cstdint>

#include "hccl/hccl_types.h"
#include "sim_log.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 获取通信域状态（设备侧接口）。
 *
 * 查询指定通信域的当前状态。
 * 对应 hccl 仓同名接口。
 *
 * 桩实现始终返回 HCCL_COMM_STATUS_READY。
 *
 * @param commId 输入：通信域标识字符串。
 * @param status 输出：通信域状态。
 * @return HcclResult 成功返回 HCCL_SUCCESS，参数为空返回 HCCL_E_PTR。
 */
HcclResult HcclCommGetStatus(const char* commId, HcclCommStatus* status)
{
    if (commId == nullptr || status == nullptr) {
        HCCL_VM_ERROR("{}: commId or status is nullptr", __func__);
        return HCCL_E_PTR;
    }
    *status = HCCL_COMM_STATUS_READY;
    HCCL_VM_INFO("{}: commId={}, returns READY", __func__, commId);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif
