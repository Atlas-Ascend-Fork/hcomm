/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 运算类 C 接口(CcuVariable* / CcuAddress* 赋值与算术)的端到端用例:
// 在注册流程中逐个调用 C 接口, 校验返回值及每次调用新追加的 rep 类型。
// CCU_V1 仅能翻译 ASSIGN 及不带立即数的 ADD, 其余运算只在 CCU_V2 下覆盖。
#include "ut_HcommCcuControlApi_common.h"

#include "ccu_log.h"
#include "ccu_primitives_impl.h"
#include "ccu_rep_add_v1.h"
#include "ccu_rep_and.h"
#include "ccu_rep_assign_v1.h"
#include "ccu_rep_mul_v1.h"
#include "ccu_rep_not.h"
#include "ccu_rep_or.h"
#include "ccu_rep_shl.h"
#include "ccu_rep_shr.h"
#include "ccu_rep_sub_v1.h"
#include "ccu_rep_xor.h"

namespace {
namespace CcuRep = hcomm::CcuRep;
using hcomm::CcuRep::CcuRepType;

constexpr uint32_t OPERAND_NUM = 3;

struct Operands {
    CcuVariableHandle v[OPERAND_NUM]{};
    CcuAddressHandle a[OPERAND_NUM]{};
};

void AllocOperands(Operands& ops)
{
    for (auto& var : ops.v) {
        EXPECT_EQ(CcuVariableAlloc(&var), OK);
    }
    for (auto& addr : ops.a) {
        EXPECT_EQ(CcuAddressAlloc(&addr), OK);
    }
}

uint16_t AddressIdOf(hcomm::CcuKernel* kernel, CcuAddressHandle handle)
{
    hcomm::CcuRep::Address* addr{nullptr};
    EXPECT_EQ(kernel->Repo().ResolveAddress(handle, &addr), OK);
    return addr == nullptr ? 0 : addr->Id();
}

// dst = lhs op rhs, 三个操作数分别落在 rep 的 VarC、VarA、VarB
template <typename RepT>
void ExpectVarVarRep(CcuRepCursor& reps, CcuRepType type, uint16_t dst, uint16_t lhs, uint16_t rhs)
{
    if (auto rep = TakeRep<RepT>(reps, type)) {
        EXPECT_EQ(rep->GetVarC().Id(), dst);
        EXPECT_EQ(rep->GetVarA().Id(), lhs);
        EXPECT_EQ(rep->GetVarB().Id(), rhs);
    }
}

// dst = lhs op 立即数
template <typename RepT>
void ExpectVarImmRep(CcuRepCursor& reps, CcuRepType type, uint16_t dst, uint16_t lhs, uint16_t imm)
{
    if (auto rep = TakeRep<RepT>(reps, type)) {
        EXPECT_EQ(rep->GetVarC().Id(), dst);
        EXPECT_EQ(rep->GetVarA().Id(), lhs);
        EXPECT_EQ(rep->GetImmedB(), imm);
    }
}

// dst = lhs 移位 rhs, 三个操作数分别落在 rep 的 VarD、VarN、VarM
template <typename RepT>
void ExpectShiftRep(CcuRepCursor& reps, CcuRepType type, uint16_t dst, uint16_t lhs, uint16_t rhs)
{
    if (auto rep = TakeRep<RepT>(reps, type)) {
        EXPECT_EQ(rep->GetVarD().Id(), dst);
        EXPECT_EQ(rep->GetVarN().Id(), lhs);
        EXPECT_EQ(rep->GetVarM().Id(), rhs);
    }
}

CcuResult ArithmeticV1OpsKernel(CcuKernelArg arg)
{
    (void)arg;
    Operands o{};
    AllocOperands(o);
    CcuRepCursor reps;

    EXPECT_EQ(CcuVariableAssignImm(o.v[0], 1), OK);
    reps.ExpectOne(CcuRepType::ASSIGN);
    EXPECT_EQ(CcuVariableAssignVar(o.v[1], o.v[0]), OK);
    reps.ExpectOne(CcuRepType::ASSIGN);
    EXPECT_EQ(CcuVariableAddVarToVar(o.v[2], o.v[0], o.v[1]), OK);
    reps.ExpectOne(CcuRepType::ADD);
    EXPECT_EQ(CcuAddressAssignImm(o.a[0], 0x1000), OK);
    reps.ExpectOne(CcuRepType::ASSIGN);
    EXPECT_EQ(CcuAddressAssignVar(o.a[1], o.v[0]), OK);
    reps.ExpectOne(CcuRepType::ASSIGN);
    EXPECT_EQ(CcuAddressAssignAddr(o.a[2], o.a[0]), OK);
    reps.ExpectOne(CcuRepType::ASSIGN);
    EXPECT_EQ(CcuAddressAddVarToAddr(o.a[2], o.a[0], o.v[0]), OK);
    reps.ExpectOne(CcuRepType::ADD);
    EXPECT_EQ(CcuAddressAddAddrToAddr(o.a[2], o.a[0], o.a[1]), OK);
    reps.ExpectOne(CcuRepType::ADD);
    EXPECT_EQ(CcuAddressAddAssignVar(o.a[2], o.v[0]), OK);
    reps.ExpectOne(CcuRepType::ADD);
    return OK;
}

// 句柄与立即数都是整型, 实参写串编译期发现不了, 故逐个核对 rep 上的操作数 Id 与入参一致。
CcuResult ArithmeticV2OpsKernel(CcuKernelArg arg)
{
    (void)arg;
    Operands o{};
    AllocOperands(o);
    CcuRepCursor reps;
    auto* kernel = reps.Kernel();
    const uint16_t v0 = VariableIdOf(kernel, o.v[0]);
    const uint16_t v1 = VariableIdOf(kernel, o.v[1]);
    const uint16_t v2 = VariableIdOf(kernel, o.v[2]);
    const uint16_t a0 = AddressIdOf(kernel, o.a[0]);
    const uint16_t a1 = AddressIdOf(kernel, o.a[1]);
    const uint16_t a2 = AddressIdOf(kernel, o.a[2]);
    constexpr uint64_t varImm = 1;
    constexpr uint64_t addrImm = 0x1000;
    constexpr uint16_t imm = 3;

    EXPECT_EQ(CcuVariableAssignImm(o.v[0], varImm), OK);
    if (auto rep = TakeRep<CcuRep::CcuRepAssign>(reps, CcuRepType::ASSIGN)) {
        EXPECT_EQ(rep->GetVarA().Id(), v0);
        EXPECT_EQ(rep->GetImmed(), varImm);
    }
    EXPECT_EQ(CcuVariableAssignVar(o.v[1], o.v[0]), OK);
    if (auto rep = TakeRep<CcuRep::CcuRepAssign>(reps, CcuRepType::ASSIGN)) {
        EXPECT_EQ(rep->GetVarB().Id(), v1);
        EXPECT_EQ(rep->GetVarA().Id(), v0);
    }

    EXPECT_EQ(CcuVariableAddVarToVar(o.v[2], o.v[0], o.v[1]), OK);
    ExpectVarVarRep<CcuRep::CcuRepAdd>(reps, CcuRepType::ADD, v2, v0, v1);
    EXPECT_EQ(CcuVariableSubVarToVar(o.v[2], o.v[0], o.v[1]), OK);
    ExpectVarVarRep<CcuRep::CcuRepSub>(reps, CcuRepType::SUB, v2, v0, v1);
    EXPECT_EQ(CcuVariableMulVarToVar(o.v[2], o.v[0], o.v[1]), OK);
    ExpectVarVarRep<CcuRep::CcuRepMul>(reps, CcuRepType::MUL, v2, v0, v1);
    EXPECT_EQ(CcuVariableAddImmToVar(o.v[2], o.v[0], imm), OK);
    ExpectVarImmRep<CcuRep::CcuRepAdd>(reps, CcuRepType::ADD, v2, v0, imm);
    EXPECT_EQ(CcuVariableSubImmToVar(o.v[2], o.v[0], imm), OK);
    ExpectVarImmRep<CcuRep::CcuRepSub>(reps, CcuRepType::SUB, v2, v0, imm);
    EXPECT_EQ(CcuVariableMulImmToVar(o.v[2], o.v[0], imm), OK);
    ExpectVarImmRep<CcuRep::CcuRepMul>(reps, CcuRepType::MUL, v2, v0, imm);
    EXPECT_EQ(CcuVariableAndVarToVar(o.v[2], o.v[0], o.v[1]), OK);
    ExpectVarVarRep<CcuRep::CcuRepAnd>(reps, CcuRepType::AND, v2, v0, v1);
    EXPECT_EQ(CcuVariableOrVarToVar(o.v[2], o.v[0], o.v[1]), OK);
    ExpectVarVarRep<CcuRep::CcuRepOr>(reps, CcuRepType::OR, v2, v0, v1);
    EXPECT_EQ(CcuVariableXorVarToVar(o.v[2], o.v[0], o.v[1]), OK);
    ExpectVarVarRep<CcuRep::CcuRepXor>(reps, CcuRepType::XOR, v2, v0, v1);
    EXPECT_EQ(CcuVariableNotVar(o.v[2], o.v[0]), OK);
    if (auto rep = TakeRep<CcuRep::CcuRepNot>(reps, CcuRepType::NOT)) {
        EXPECT_EQ(rep->GetVarC().Id(), v2);
        EXPECT_EQ(rep->GetVarB().Id(), v0);
    }
    EXPECT_EQ(CcuVariableShlVarToVar(o.v[2], o.v[0], o.v[1]), OK);
    ExpectShiftRep<CcuRep::CcuRepShL>(reps, CcuRepType::SHL, v2, v0, v1);
    EXPECT_EQ(CcuVariableShrVarToVar(o.v[2], o.v[0], o.v[1]), OK);
    ExpectShiftRep<CcuRep::CcuRepShR>(reps, CcuRepType::SHR, v2, v0, v1);

    EXPECT_EQ(CcuAddressAssignImm(o.a[0], addrImm), OK);
    if (auto rep = TakeRep<CcuRep::CcuRepAssign>(reps, CcuRepType::ASSIGN)) {
        EXPECT_EQ(rep->GetAddrA().Id(), a0);
        EXPECT_EQ(rep->GetImmed(), addrImm);
    }
    EXPECT_EQ(CcuAddressAssignVar(o.a[1], o.v[0]), OK);
    if (auto rep = TakeRep<CcuRep::CcuRepAssign>(reps, CcuRepType::ASSIGN)) {
        EXPECT_EQ(rep->GetAddrA().Id(), a1);
        EXPECT_EQ(rep->GetVarA().Id(), v0);
    }
    EXPECT_EQ(CcuAddressAssignAddr(o.a[2], o.a[0]), OK);
    if (auto rep = TakeRep<CcuRep::CcuRepAssign>(reps, CcuRepType::ASSIGN)) {
        EXPECT_EQ(rep->GetAddrB().Id(), a2);
        EXPECT_EQ(rep->GetAddrA().Id(), a0);
    }
    EXPECT_EQ(CcuAddressAddVarToAddr(o.a[2], o.a[0], o.v[0]), OK);
    if (auto rep = TakeRep<CcuRep::CcuRepAdd>(reps, CcuRepType::ADD)) {
        EXPECT_EQ(rep->GetAddrC().Id(), a2);
        EXPECT_EQ(rep->GetAddrA().Id(), a0);
        EXPECT_EQ(rep->GetVarB().Id(), v0);
    }
    EXPECT_EQ(CcuAddressAddAddrToAddr(o.a[2], o.a[0], o.a[1]), OK);
    if (auto rep = TakeRep<CcuRep::CcuRepAdd>(reps, CcuRepType::ADD)) {
        EXPECT_EQ(rep->GetAddrC().Id(), a2);
        EXPECT_EQ(rep->GetAddrA().Id(), a0);
        EXPECT_EQ(rep->GetAddrB().Id(), a1);
    }
    EXPECT_EQ(CcuAddressAddAssignVar(o.a[2], o.v[0]), OK);
    if (auto rep = TakeRep<CcuRep::CcuRepAdd>(reps, CcuRepType::ADD)) {
        EXPECT_EQ(rep->GetAddrA().Id(), a2);
        EXPECT_EQ(rep->GetVarB().Id(), v0);
    }
    EXPECT_EQ(CcuAddressAddImmToAddr(o.a[2], o.a[0], imm), OK);
    if (auto rep = TakeRep<CcuRep::CcuRepAdd>(reps, CcuRepType::ADD)) {
        EXPECT_EQ(rep->GetAddrC().Id(), a2);
        EXPECT_EQ(rep->GetAddrA().Id(), a0);
        EXPECT_EQ(rep->GetImmedB(), imm);
    }
    return OK;
}

CcuResult ArithmeticInvalidHandleKernel(CcuKernelArg arg)
{
    (void)arg;
    Operands o{};
    AllocOperands(o);
    CcuRepCursor reps;

    EXPECT_EQ(CcuVariableAssignImm(INVALID_HANDLE, 1), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuVariableAddVarToVar(o.v[2], o.v[0], INVALID_HANDLE), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuVariableNotVar(INVALID_HANDLE, o.v[0]), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuAddressAssignVar(o.a[0], INVALID_HANDLE), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuAddressAddAddrToAddr(o.a[2], INVALID_HANDLE, o.a[1]), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuAddressAddImmToAddr(INVALID_HANDLE, o.a[0], 3), NOT_FOUND);
    reps.ExpectNone();

    // 保证 kernel 非空, 注册流程可正常收尾
    EXPECT_EQ(CcuVariableAssignImm(o.v[0], 1), OK);
    reps.ExpectOne(CcuRepType::ASSIGN);
    return OK;
}
} // namespace

CCU_FUNC_KERNEL_TEST(Ut_CcuArithmeticApi_When_V1SupportedOps_Expect_AssignOrAddRep, ArithmeticV1OpsKernel, true)
CCU_FUNC_KERNEL_V2_TEST(Ut_CcuArithmeticApi_When_V2AllOps_Expect_MatchingRep, ArithmeticV2OpsKernel, true)
CCU_FUNC_KERNEL_TEST(
    Ut_CcuArithmeticApi_When_InvalidHandle_Expect_NotFoundAndNoRep, ArithmeticInvalidHandleKernel, true)
