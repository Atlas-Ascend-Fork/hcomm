/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_LOAD_H
#define HCOMM_CCU_LOAD_H

#include <cstdint>

#include "ccu_kernel.h"

namespace hcomm {

// 参数加载与存储类 rep 构造接口：LoadArg、Load / Store 变量块。
namespace ccu_load {

    // 参数加载类 相关接口
    CcuResult LoadArg(CcuKernel& kernel, CcuVariableHandle varHandle, uint32_t argId);
    CcuResult LoadVar(CcuKernel& kernel, uint64_t addr, CcuVariableHandle varHandle, uint32_t num);
    CcuResult
    LoadVarFromVarAddr(CcuKernel& kernel, CcuVariableHandle addrHandle, CcuVariableHandle varHandle, uint32_t num);
    CcuResult LoadAddImm(
        CcuKernel& kernel, CcuVariableHandle varHandle, uint16_t srcNum, CcuVariableHandle offsetHandle,
        uint16_t immAddValue, CcuVariableHandle dstHandle);
    CcuResult AddImmStore(
        CcuKernel& kernel, CcuVariableHandle varHandle, uint16_t dstNum, CcuVariableHandle offsetHandle,
        uint16_t immAddValue, CcuVariableHandle srcHandle);
    CcuResult StoreVar(CcuKernel& kernel, uint64_t addr, CcuVariableHandle varHandle, uint32_t num);
    CcuResult
    StoreVarToVarAddr(CcuKernel& kernel, CcuVariableHandle addrHandle, CcuVariableHandle varHandle, uint32_t num);
} // namespace ccu_load

} // namespace hcomm

#endif // HCOMM_CCU_LOAD_H
