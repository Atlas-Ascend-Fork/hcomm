/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_ARITHMETIC_H
#define HCOMM_CCU_ARITHMETIC_H

#include <cstdint>

#include "ccu_kernel.h"

namespace hcomm {

// 运算类 rep 构造接口：Variable / Address 的赋值与算术运算。
// rep 经 Variable / Address 的运算符重载追加到传入的 kernel。
namespace ccu_arithmetic {

    CcuResult VariableAssignImm(CcuKernel& kernel, CcuVariableHandle varHandle, uint64_t immediate);
    CcuResult VariableAssignVar(CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varA);
    CcuResult VariableAddVarToVar(
        CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle);
    CcuResult VariableSubVarToVar(
        CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle);
    CcuResult VariableMulVarToVar(
        CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle);
    CcuResult VariableAddImmToVar(
        CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, uint16_t immediate);
    CcuResult VariableSubImmToVar(
        CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, uint16_t immediate);
    CcuResult VariableMulImmToVar(
        CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, uint16_t immediate);
    CcuResult VariableAndVarToVar(
        CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle);
    CcuResult VariableOrVarToVar(
        CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle);
    CcuResult VariableXorVarToVar(
        CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle);
    CcuResult VariableNotVar(CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle);
    CcuResult VariableShlVarToVar(
        CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle);
    CcuResult VariableShrVarToVar(
        CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle);
    CcuResult AddressAssignImm(CcuKernel& kernel, CcuAddressHandle addrHandle, uint64_t immediate);
    CcuResult AddressAssignVar(CcuKernel& kernel, CcuAddressHandle addrHandle, CcuVariableHandle varHandle);
    CcuResult AddressAssignAddr(CcuKernel& kernel, CcuAddressHandle dstAddrHandle, CcuAddressHandle srcAddrHandle);
    CcuResult AddressAddVarToAddr(
        CcuKernel& kernel, CcuAddressHandle resAddrHandle, CcuAddressHandle lhsAddrHandle,
        CcuVariableHandle rhsVarHandle);
    CcuResult AddressAddAddrToAddr(
        CcuKernel& kernel, CcuAddressHandle resAddrHandle, CcuAddressHandle addrAHandle, CcuAddressHandle addrBHandle);
    CcuResult AddressAddAssignVar(CcuKernel& kernel, CcuAddressHandle addrHandle, CcuVariableHandle varHandle);
    CcuResult AddressAddAssignAddr(CcuKernel& kernel, CcuAddressHandle addrHandle, CcuAddressHandle otherHandle);
    CcuResult
    AddressAddImmToAddr(CcuKernel& kernel, CcuAddressHandle resAddrHandle, CcuAddressHandle addrAHandle, uint16_t imm);
} // namespace ccu_arithmetic

} // namespace hcomm

#endif // HCOMM_CCU_ARITHMETIC_H
