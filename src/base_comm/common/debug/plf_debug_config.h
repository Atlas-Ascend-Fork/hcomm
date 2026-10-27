/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_PLF_DEBUG_CONFIG_H
#define HCCLV2_PLF_DEBUG_CONFIG_H

#include "common/debug/plf_debug_config_pub.h"
#include "hccl/base.h"
#include <atomic>
#include <cstdint>
#include <mutex>

namespace Hccl {

constexpr u64 PLF_ADDR_ALIGN_512B = 512ULL;

inline bool IsAddrAligned512B(const void* addr)
{
    return (reinterpret_cast<uintptr_t>(addr) & (PLF_ADDR_ALIGN_512B - 1)) == 0;
}

constexpr u64 PLF_ALG = 0x1ULL << 0;
constexpr u64 PLF_TASK = 0x1ULL << 1;
constexpr u64 PLF_RES = 0x1ULL << 2;
constexpr u64 PLF_DATA_OP = 0x1ULL << 3;
constexpr u64 PLF_CHANNEL = 0x1ULL << 4;

u64 GetPlfDebugConfigValue();
void SetPlfDebugConfigValue(u64 value);

} // namespace Hccl

#endif // HCCLV2_PLF_DEBUG_CONFIG_H
