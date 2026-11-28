/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_arithmetic.h"

#include "ccu_log.h"
#include "config_plf_log_v2.h"

namespace hcomm {

using Hccl::PLF_DATA_OP;

CcuResult ccu_arithmetic::VariableAssignImm(CcuKernel& kernel, CcuVariableHandle varHandle, uint64_t immediate)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[VariableAssignImm] varHandle=%llu, immediate=%llu", varHandle, immediate);
    CcuRep::Variable* variable{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &variable));
    // 通过符号重载实现，内部记录rep；异常由入口 HcommCcuKernelRegister 的
    // CCU_EXCEPTION_HANDLE_BEGIN/END 统一接住，无需在此局部 try/catch。
    (*variable) = immediate;
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_arithmetic::VariableAssignVar(CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varA)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[VariableAssignVar] varHandle=%llu, varA=%llu", varHandle, varA);
    CcuRep::Variable* variable{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &variable));
    CcuRep::Variable* variableA{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varA, &variableA));
    // 通过符号重载实现，内部记录rep；异常由入口统一 catch。
    (*variable) = (*variableA);
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_arithmetic::VariableAddVarToVar(
    CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[VariableAddVarToVar] varHandle=%llu, varAHandle=%llu, varBHandle=%llu", varHandle, varAHandle,
        varBHandle);
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &resVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varAHandle, &leftVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varBHandle, &rightVar));

    // 通过符号重载实现，内部记录rep；异常由入口统一 catch。
    *resVar = *leftVar + *rightVar;
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_arithmetic::VariableSubVarToVar(
    CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &resVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varAHandle, &leftVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varBHandle, &rightVar));

    *resVar = *leftVar - *rightVar;
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_arithmetic::VariableMulVarToVar(
    CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &resVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varAHandle, &leftVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varBHandle, &rightVar));

    *resVar = *leftVar * *rightVar;
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_arithmetic::VariableAddImmToVar(
    CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, uint16_t immediate)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &resVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varAHandle, &leftVar));

    *resVar = *leftVar + immediate;
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_arithmetic::VariableSubImmToVar(
    CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, uint16_t immediate)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &resVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varAHandle, &leftVar));

    *resVar = *leftVar - immediate;
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_arithmetic::VariableMulImmToVar(
    CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, uint16_t immediate)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &resVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varAHandle, &leftVar));

    *resVar = *leftVar * immediate;
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_arithmetic::VariableAndVarToVar(
    CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &resVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varAHandle, &leftVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varBHandle, &rightVar));

    *resVar = *leftVar & *rightVar;
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_arithmetic::VariableOrVarToVar(
    CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &resVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varAHandle, &leftVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varBHandle, &rightVar));

    *resVar = *leftVar | *rightVar;
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_arithmetic::VariableXorVarToVar(
    CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &resVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varAHandle, &leftVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varBHandle, &rightVar));

    *resVar = *leftVar ^ *rightVar;
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_arithmetic::VariableNotVar(CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &resVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varAHandle, &leftVar));

    *resVar = ~(*leftVar);
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_arithmetic::VariableShlVarToVar(
    CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &resVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varAHandle, &leftVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varBHandle, &rightVar));

    *resVar = *leftVar << *rightVar;
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_arithmetic::VariableShrVarToVar(
    CcuKernel& kernel, CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &resVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varAHandle, &leftVar));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varBHandle, &rightVar));

    *resVar = *leftVar >> *rightVar;
    return CcuResult::CCU_SUCCESS;
}

// addr = 立即数 → CcuRepAssign(Address, uint64_t)
CcuResult ccu_arithmetic::AddressAssignImm(CcuKernel& kernel, CcuAddressHandle addrHandle, uint64_t immediate)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[AddressAssignImm] addrHandle=%llu, immediate=%llu", addrHandle, immediate);
    CcuRep::Address* address{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveAddress(addrHandle, &address));
    (*address) = immediate;
    return CcuResult::CCU_SUCCESS;
}

// addr = variable → CcuRepAssign(Address, Variable)
CcuResult ccu_arithmetic::AddressAssignVar(CcuKernel& kernel, CcuAddressHandle addrHandle, CcuVariableHandle varHandle)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[AddressAssignVar] addrHandle=%llu, varHandle=%llu", addrHandle, varHandle);
    CcuRep::Address* address{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveAddress(addrHandle, &address));

    CcuRep::Variable* variable{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &variable));

    (*address) = (*variable);
    return CcuResult::CCU_SUCCESS;
}
// addr = addr → CcuRepAssign(Address, Address)
CcuResult
ccu_arithmetic::AddressAssignAddr(CcuKernel& kernel, CcuAddressHandle dstAddrHandle, CcuAddressHandle srcAddrHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[AddressAssignAddr] dstAddrHandle=%llu, srcAddrHandle=%llu", dstAddrHandle, srcAddrHandle);
    CcuRep::Address* dstAddress{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveAddress(dstAddrHandle, &dstAddress));

    CcuRep::Address* srcAddress{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveAddress(srcAddrHandle, &srcAddress));

    (*dstAddress) = (*srcAddress);
    return CcuResult::CCU_SUCCESS;
}

// resAddr = lhsAddr + rhsVar → CcuRepAdd(Address, Address, Variable)
CcuResult ccu_arithmetic::AddressAddVarToAddr(
    CcuKernel& kernel, CcuAddressHandle resAddrHandle, CcuAddressHandle lhsAddrHandle, CcuVariableHandle rhsVarHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[AddressAddVarToAddr] resAddrHandle=%llu, lhsAddrHandle=%llu, rhsVarHandle=%llu", resAddrHandle,
        lhsAddrHandle, rhsVarHandle);
    CcuRep::Address *resAddr{nullptr}, *lhsAddr{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveAddress(resAddrHandle, &resAddr));
    CCU_CHK_RET(kernel.Repo().ResolveAddress(lhsAddrHandle, &lhsAddr));

    CcuRep::Variable* rhsVar{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(rhsVarHandle, &rhsVar));

    *resAddr = *lhsAddr + *rhsVar;
    return CcuResult::CCU_SUCCESS;
}

// resAddr = addrA + addrB → CcuRepAdd(Address, Address, Address)
CcuResult ccu_arithmetic::AddressAddAddrToAddr(
    CcuKernel& kernel, CcuAddressHandle resAddrHandle, CcuAddressHandle addrAHandle, CcuAddressHandle addrBHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[AddressAddAddrToAddr] resAddrHandle=%llu, addrAHandle=%llu, addrBHandle=%llu", resAddrHandle,
        addrAHandle, addrBHandle);
    CcuRep::Address *resAddr{nullptr}, *addrA{nullptr}, *addrB{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveAddress(resAddrHandle, &resAddr));
    CCU_CHK_RET(kernel.Repo().ResolveAddress(addrAHandle, &addrA));
    CCU_CHK_RET(kernel.Repo().ResolveAddress(addrBHandle, &addrB));

    *resAddr = *addrA + *addrB;
    return CcuResult::CCU_SUCCESS;
}

// addr += variable → CcuRepAdd(Address, Variable) 就地加
CcuResult
ccu_arithmetic::AddressAddAssignVar(CcuKernel& kernel, CcuAddressHandle addrHandle, CcuVariableHandle varHandle)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[AddressAddAssignVar] addrHandle=%llu, varHandle=%llu", addrHandle, varHandle);
    CcuRep::Address* address{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveAddress(addrHandle, &address));

    CcuRep::Variable* variable{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &variable));

    (*address) += (*variable);
    return CcuResult::CCU_SUCCESS;
}

// addr += addr → 等价于 addr = addr + otherAddr
CcuResult
ccu_arithmetic::AddressAddAssignAddr(CcuKernel& kernel, CcuAddressHandle addrHandle, CcuAddressHandle otherHandle)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[AddressAddAssignAddr] addrHandle=%llu, otherHandle=%llu", addrHandle, otherHandle);
    CcuRep::Address* address{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveAddress(addrHandle, &address));

    CcuRep::Address* other{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveAddress(otherHandle, &other));

    (*address) = (*address) + (*other);
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_arithmetic::AddressAddImmToAddr(
    CcuKernel& kernel, CcuAddressHandle resAddrHandle, CcuAddressHandle addrAHandle, uint16_t imm)
{
    CcuRep::Address *resAddr{nullptr}, *addrA{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveAddress(resAddrHandle, &resAddr));
    CCU_CHK_RET(kernel.Repo().ResolveAddress(addrAHandle, &addrA));

    *resAddr = *addrA + imm;
    return CcuResult::CCU_SUCCESS;
}

} // namespace hcomm
