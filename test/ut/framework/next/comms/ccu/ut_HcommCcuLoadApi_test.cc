/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 加载类 C 接口(任务参数装载 / 变量读写内存)的端到端用例:
// 在注册流程中逐个调用 C 接口, 校验返回值及每次调用新追加的 rep 类型。
#include "ut_HcommCcuControlApi_common.h"

#include "ccu_log.h"
#include "ccu_primitives_impl.h"
#include "ccu_rep_load_add_imm.h"
#include "ccu_rep_load_v1.h"
#include "ccu_rep_load_var_v1.h"
#include "ccu_rep_loadarg_v1.h"
#include "ccu_rep_store_add_imm.h"
#include "ccu_rep_store_v1.h"
#include "ccu_rep_store_var_v1.h"

namespace {
namespace CcuRep = hcomm::CcuRep;
using hcomm::CcuRep::CcuRepType;

constexpr uint64_t FAKE_MEM_ADDR = 0x10000;
constexpr uint32_t LOAD_VAR_NUM = 4;

struct LoadOperands {
    CcuVariableHandle v[LOAD_VAR_NUM]{};
};

void AllocLoadOperands(LoadOperands& ops)
{
    for (auto& var : ops.v) {
        EXPECT_EQ(CcuVariableAlloc(&var), OK);
    }
}

// 逐个核对 rep 上的操作数与入参一致。
CcuResult LoadAllOpsKernel(CcuKernelArg arg)
{
    (void)arg;
    LoadOperands o{};
    AllocLoadOperands(o);
    CcuRepCursor reps;
    auto* kernel = reps.Kernel();
    const uint16_t baseId = VariableIdOf(kernel, o.v[0]);
    const uint16_t offsetId = VariableIdOf(kernel, o.v[2]);
    const uint16_t addrVarId = VariableIdOf(kernel, o.v[3]);
    // 取 CCU_SQE_ARGS_LEN 使 rep 上的槽位下标与完整 argId 取值不同
    constexpr uint32_t argId = hcomm::CCU_SQE_ARGS_LEN;
    constexpr uint32_t varNum = 2; // 取 o.v[0] 起的连续两个变量
    constexpr uint16_t immAddValue = 5;
    constexpr uint16_t srcNum = 2;
    constexpr uint16_t dstNum = 2;

    EXPECT_EQ(CcuLoadArg(o.v[0], argId), OK);
    if (auto loadArg = TakeRep<CcuRep::CcuRepLoadArg>(reps, CcuRepType::LOAD_ARG)) {
        EXPECT_EQ(loadArg->GetVarId(), baseId);
        EXPECT_EQ(loadArg->GetArgId(), argId % hcomm::CCU_SQE_ARGS_LEN);
        EXPECT_EQ(loadArg->GetFullArgId(), argId);
    }
    hcomm::CcuKernelInfo info{};
    EXPECT_EQ(kernel->GetCcuKernelInfo(info), OK);
    EXPECT_EQ(info.maxTaskArgsNum, argId + 1);

    EXPECT_EQ(CcuLoadVar(FAKE_MEM_ADDR, o.v[0], varNum), OK);
    if (auto load = TakeRep<CcuRep::CcuRepLoad>(reps, CcuRepType::LOAD)) {
        EXPECT_EQ(load->GetAddr(), FAKE_MEM_ADDR);
        EXPECT_EQ(load->GetVar().Id(), baseId);
        EXPECT_EQ(load->GetNum(), varNum);
    }

    EXPECT_EQ(CcuStoreVar(FAKE_MEM_ADDR, o.v[0], varNum), OK);
    if (auto store = TakeRep<CcuRep::CcuRepStore>(reps, CcuRepType::STORE)) {
        EXPECT_EQ(store->GetAddr(), FAKE_MEM_ADDR);
        EXPECT_EQ(store->GetVar().Id(), baseId);
        EXPECT_EQ(store->GetNum(), varNum);
    }

    EXPECT_EQ(CcuLoadVarFromVarAddr(o.v[3], o.v[0], varNum), OK);
    if (auto loadVar = TakeRep<CcuRep::CcuRepLoadVar>(reps, CcuRepType::LOAD_VAR)) {
        EXPECT_EQ(loadVar->GetSrc().Id(), addrVarId);
        EXPECT_EQ(loadVar->GetVar().Id(), baseId);
        EXPECT_EQ(loadVar->GetNum(), varNum);
    }

    EXPECT_EQ(CcuStoreVarToVarAddr(o.v[3], o.v[0], varNum), OK);
    if (auto storeVar = TakeRep<CcuRep::CcuRepStoreVar>(reps, CcuRepType::STORE_VAR)) {
        EXPECT_EQ(storeVar->GetDst().Id(), addrVarId);
        EXPECT_EQ(storeVar->GetVar().Id(), baseId);
        EXPECT_EQ(storeVar->GetNum(), varNum);
    }

    EXPECT_EQ(CcuLoadAddImm(o.v[0], srcNum, o.v[2], immAddValue, o.v[3]), OK);
    if (auto loadAddImm = TakeRep<CcuRep::CcuRepLoadAddImm>(reps, CcuRepType::LOAD_ADD_IMM)) {
        EXPECT_EQ(loadAddImm->GetSrcId(), baseId);
        EXPECT_EQ(loadAddImm->GetSrcOffsetId(), offsetId);
        EXPECT_EQ(loadAddImm->GetDstId(), addrVarId);
        EXPECT_EQ(loadAddImm->GetImmAddValue(), immAddValue);
        EXPECT_EQ(loadAddImm->GetSrcNum(), srcNum);
    }

    EXPECT_EQ(CcuAddImmStore(o.v[0], dstNum, o.v[2], immAddValue, o.v[3]), OK);
    if (auto storeAddImm = TakeRep<CcuRep::CcuRepStoreAddImm>(reps, CcuRepType::STORE_ADD_IMM)) {
        EXPECT_EQ(storeAddImm->GetDstId(), baseId);
        EXPECT_EQ(storeAddImm->GetDstOffsetId(), offsetId);
        EXPECT_EQ(storeAddImm->GetSrcId(), addrVarId);
        EXPECT_EQ(storeAddImm->GetImmAddValue(), immAddValue);
        EXPECT_EQ(storeAddImm->GetDstNum(), dstNum);
    }
    return OK;
}

CcuResult LoadInvalidHandleKernel(CcuKernelArg arg)
{
    (void)arg;
    LoadOperands o{};
    AllocLoadOperands(o);
    CcuRepCursor reps;

    EXPECT_EQ(CcuLoadArg(INVALID_HANDLE, 0), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuLoadVar(FAKE_MEM_ADDR, INVALID_HANDLE, 1), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuStoreVar(FAKE_MEM_ADDR, INVALID_HANDLE, 1), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuLoadVarFromVarAddr(INVALID_HANDLE, o.v[0], 1), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuStoreVarToVarAddr(INVALID_HANDLE, o.v[0], 1), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuLoadAddImm(INVALID_HANDLE, 1, o.v[2], 0, o.v[0]), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuAddImmStore(INVALID_HANDLE, 1, o.v[2], 0, o.v[0]), NOT_FOUND);
    reps.ExpectNone();
    // rep 上的 fullArgId 为 uint16, 超出该位宽的 argId 在注册期拦下
    constexpr uint32_t overflowArgId = static_cast<uint32_t>(UINT16_MAX) + 1;
    EXPECT_EQ(CcuLoadArg(o.v[0], overflowArgId), CcuResult::CCU_E_PARA);
    reps.ExpectNone();
    // LoadArg 解析失败后 argId 不得留在任务参数账本里, 否则 maxTaskArgsNum 虚增。
    hcomm::CcuKernelInfo info{};
    EXPECT_EQ(reps.Kernel()->GetCcuKernelInfo(info), OK);
    EXPECT_EQ(info.maxTaskArgsNum, 0U);
    return OK;
}
} // namespace

CCU_FUNC_KERNEL_V2_TEST(Ut_CcuLoadApi_When_AllOps_Expect_MatchingRep, LoadAllOpsKernel, true)
CCU_FUNC_KERNEL_V2_TEST(Ut_CcuLoadApi_When_InvalidHandle_Expect_NotFoundAndNoRep, LoadInvalidHandleKernel, true)
