/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 边界投影头：`HcclDpuTaskexpShmemRestore` kernel 入参块 ABI，供 coll_communicator_mgr（下发侧）与
// legacy950 kernel 侧共享。
#ifndef HCOMM_DFX_DEVICE_DPU_TASKEXP_KERNEL_PARAM_PUB_H
#define HCOMM_DFX_DEVICE_DPU_TASKEXP_KERNEL_PARAM_PUB_H

#include <cstddef>
#include <cstdint>
#include "hccl/hccl_types.h"

namespace Hccl {

/**
 * @brief `HcclDpuTaskexpShmemRestore` kernel 的入参块：下发侧（coll_communicator_mgr）与 device
 *        kernel 侧（legacy950 kernel_entrance）共享的 ABI。kernel 按裸地址解读 args，
 *        故字段顺序与尺寸不可改，两侧必须用同一份定义（本头同时编入 hcomm/hccl_v2/ccl_kernel）。
 */
struct DpuTaskexpKernelParam {
    char commId[COMM_NAME_MAX_LENGTH];
    void* taskexceptionVa;
    uint64_t memorySize;
    uint32_t deviceId;
    uint32_t reserved; // 显式补齐尾部填充：避免两侧编译器对隐式填充的取舍不一致
};
static_assert(sizeof(DpuTaskexpKernelParam) == COMM_NAME_MAX_LENGTH + 24, "DpuTaskexpKernelParam ABI");
static_assert(offsetof(DpuTaskexpKernelParam, taskexceptionVa) == COMM_NAME_MAX_LENGTH, "DpuTaskexpKernelParam ABI");
static_assert(offsetof(DpuTaskexpKernelParam, memorySize) == COMM_NAME_MAX_LENGTH + 8, "DpuTaskexpKernelParam ABI");
static_assert(offsetof(DpuTaskexpKernelParam, deviceId) == COMM_NAME_MAX_LENGTH + 16, "DpuTaskexpKernelParam ABI");

} // namespace Hccl

#endif // HCOMM_DFX_DEVICE_DPU_TASKEXP_KERNEL_PARAM_PUB_H
