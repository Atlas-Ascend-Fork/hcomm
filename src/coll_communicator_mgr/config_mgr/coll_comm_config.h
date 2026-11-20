/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_COMM_CONFIG_H
#define COLL_COMM_CONFIG_H

#include <cstddef>
#include <cstdint>
#include <string>

#include "hccl/hccl_types.h"
#include "net_layer_config.h"

namespace hccl {
// 这里只使用引用，避免将 legacy 配置头传播到拓扑探测等调用方。
class CommConfig;

constexpr uint32_t HCCL_COMM_CONFIG_SQ_DEPTH_VERSION = 11U;
constexpr uint32_t HCCL_COMM_CONFIG_SCQ_DEPTH_VERSION = 12U;

void InitDefaultHcclCommConfig(HcclCommConfig& config);

// 按size承接完整本地副本，保留原版本供旧业务字段消费；仅对新增选层字段检查版本和长度。
// 失败不修改输出。
// 非空input至少须可读size_t字节，并保证实际缓冲区不小于头部声明的size。
HcclResult ReadHcclCommConfig(const HcclCommConfig* input, HcclCommConfig& output);

HcclResult ApplyHcclCommConfig(
    const HcclCommConfig* hcclCommConfig, CommConfig& commConfig, uint32_t& opExpansionMode,
    int32_t processDeterministic);

} // namespace hccl

#endif /* COLL_COMM_CONFIG_H */
