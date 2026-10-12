/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Description: 用于1.0和2.0不同版本的兼容性处理
 * Author: huangweihao
 * Create: 2025-02-10
 */

#ifndef HCCLV2_TRANSFORM_FUNS_TYPE_H
#define HCCLV2_TRANSFORM_FUNS_TYPE_H

#include <cstdint>
#include <map>

#include "checker_def.h"
#include "data_type.h"
#include "hccl_types.h"

namespace HcclSim {
extern std::map<DataType, HcclDataType> g_DataType2CheckerDataType_aicpu;

extern std::map<uint16_t, HcclReduceOp> g_ReduceOp2CheckerReduceOp_ccu;
} // namespace HcclSim

#endif // HCCLV2_TRANSFORM_FUNS_TYPE_H
