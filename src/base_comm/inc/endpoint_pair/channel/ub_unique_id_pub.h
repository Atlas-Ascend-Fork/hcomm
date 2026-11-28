/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_UB_UNIQUE_ID_PUB_H
#define HCOMM_UB_UNIQUE_ID_PUB_H

#include <cstdint>

// 约束：本头文件零外部依赖（仅 <cstdint>），禁止引入 legacy 头文件
constexpr uint32_t UB_UNIQUE_ID_EXT_MAGIC_WORD = 0x55424558;
constexpr uint32_t UB_UNIQUE_ID_EXT_VERSION = 1;

#endif // HCOMM_UB_UNIQUE_ID_PUB_H
