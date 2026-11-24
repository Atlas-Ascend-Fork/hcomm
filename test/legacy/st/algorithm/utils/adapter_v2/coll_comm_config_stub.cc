/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_comm.h"
#include "types.h"

namespace hccl {
// ST桩：真实实现(coll_comm_config.cc)的legacy头依赖链过深，ST侧仅补链接符号；
// ST用例netLayerExpression为空串时不经过此函数，透传语义保底。
HcclResult ReadHcclCommConfig(const HcclCommConfig* input, HcclCommConfig& output)
{
    if (input == nullptr) {
        return HCCL_E_PTR;
    }
    output = *input;
    return HCCL_SUCCESS;
}
} // namespace hccl
