/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_MEM_TYPE_CONVERT_PUB_H
#define HCOMM_MEM_TYPE_CONVERT_PUB_H

#include "hccl_mem_defs.h"
#include "hcomm_res_defs.h"

namespace hccl {
inline CommMemType ConvertHcclToCommMemType(HcclMemType hcclType)
{
    switch (hcclType) {
        case HCCL_MEM_TYPE_DEVICE:
            return COMM_MEM_TYPE_DEVICE;
        case HCCL_MEM_TYPE_HOST:
            return COMM_MEM_TYPE_HOST;
        default:
            return COMM_MEM_TYPE_INVALID;
    }
}

inline HcclMemType ConvertCommToHcclMemType(CommMemType commType)
{
    switch (commType) {
        case COMM_MEM_TYPE_DEVICE:
            return HCCL_MEM_TYPE_DEVICE;
        case COMM_MEM_TYPE_HOST:
            return HCCL_MEM_TYPE_HOST;
        default:
            return HCCL_MEM_TYPE_NUM;
    }
}
} // namespace hccl
#endif // MEM_TYPE_CONVERT_H
