/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_ENDPOINT_C_ADPT_PUB_H
#define HCOMM_ENDPOINT_C_ADPT_PUB_H

#include <stdint.h>
#include "hcomm_res_defs.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief 通信设备Endpoint监听配置结构体
 */
#ifndef HCOMM_ENDPOINT_LISTEN_CONFIG_DEFINED
#define HCOMM_ENDPOINT_LISTEN_CONFIG_DEFINED
typedef struct {
    union {
        uint8_t raws[24]; ///< 通用数据区，用于未来扩展，如backlog, timeout等
        struct {
            uint8_t reserved[24];
        } reserved; ///< 具名占位：标准 C/C++ 均可编译
    };
} HcommEndpointListenConfig;
#endif

/**
 * @brief 启动通信设备Endpoint监听
 * @param[in] endpointHandle Endpoint句柄
 * @param[in] port 监听端口号
 * @param[in] config 监听配置参数（可为NULL，使用默认配置）
 * @return HcommResult 执行结果状态码
 * @note 启动指定Endpoint在指定端口上的监听服务
 */
extern HcommResult
HcommEndpointStartListen(EndpointHandle endpointHandle, uint32_t port, const HcommEndpointListenConfig* config);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCOMM_ENDPOINT_C_ADPT_PUB_H
