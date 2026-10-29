/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// base_comm 侧 AICPU 通信域公共常量。
#ifndef HCOMM_AICPU_COMM_DEF_PUB_H
#define HCOMM_AICPU_COMM_DEF_PUB_H

#include <cstdint>

constexpr uint32_t HCOMID_MAX_SIZE = 128;

// notify 参数缓冲与预留容量基准（非对外上限；实际可申请数量由底层资源决定）
constexpr uint32_t HCOMM_THREAD_NOTIFY_MAX_NUM = 256U;

#endif // HCOMM_AICPU_COMM_DEF_PUB_H
