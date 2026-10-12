/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_COMPOSITE_OP_GROUPING_H
#define CHECKER_COMPOSITE_OP_GROUPING_H

#include <map>
#include <vector>

#include "hccl_types.h"
#include "operation_data/operation_data_types.h"

namespace HcclSim {

// 同一次集合通信中，每个物理 deviceId 最多对应一条算子记录。
using CompositeOpGroup = std::map<uint32_t, sim::operation::CompositeOpDetail>;

// 按通信域名称及 RunnerDB 提供的 opIter 归并算子。
// 输入仍以通信域内 rankId 索引，仅为兼容已有加载接口。
HcclResult GroupCompositeOpDetails(
    const std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>>& compositeData,
    std::vector<CompositeOpGroup>& opGroups);

} // namespace HcclSim

#endif // CHECKER_COMPOSITE_OP_GROUPING_H
