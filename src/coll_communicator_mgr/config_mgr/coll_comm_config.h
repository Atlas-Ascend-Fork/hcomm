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

#include "comm_config_pub.h"
#include "hccl/hccl_types.h"

namespace hccl {
constexpr uint32_t HCCL_COMM_CONFIG_SQ_DEPTH_VERSION = 11U;
constexpr uint32_t HCCL_COMM_CONFIG_SCQ_DEPTH_VERSION = 12U;

/// 内部默认配置初始化：填充 magicWord/version 等头部字段及各哨兵值，与对外 HcclCommConfigInit 语义等价。
/// 内部代码不直接调用对外接口 HcclCommConfigInit，统一走本内部封装。
void InitDefaultHcclCommConfig(HcclCommConfig& config);

HcclResult ApplyHcclCommConfig(
    const HcclCommConfig* hcclCommConfig, CommConfig& commConfig, uint32_t& opExpansionMode,
    int32_t processDeterministic);
} // namespace hccl

#endif /* COLL_COMM_CONFIG_H */
