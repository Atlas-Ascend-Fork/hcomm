/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_load.h"

#include "ccu_rep_v1.h"
#include "ccu_log.h"
#include "config_plf_log_v2.h"

namespace hcomm {

using Hccl::PLF_DATA_OP;

constexpr uint32_t MIN_PINNED_REG_GROUP_NUM = 2; // 连续变量数 >= 2 时才需添加 pinned 寄存器组

// 加载类 相关接口
CcuResult ccu_load::LoadArg(CcuKernel& kernel, CcuVariableHandle varHandle, uint32_t argId)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[LoadArg] varHandle=%llu, argId=%u", varHandle, argId);
    if (argId > UINT16_MAX) {
        HCCL_ERROR("[%s] argId[%u] must not be greater than %u", __func__, argId, UINT16_MAX);
        return CcuResult::CCU_E_PARA;
    }
    CcuRep::Variable* var{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &var));
    kernel.AddLoadArgUsedForRepBuild(argId);
    auto loadArgRep = std::make_shared<CcuRep::CcuRepLoadArg>(
        kernel.GetInsGenerator(), *var, argId % CCU_SQE_ARGS_LEN, static_cast<uint16_t>(argId));
    kernel.Append(loadArgRep);
    return CcuResult::CCU_SUCCESS;
}

static CcuResult CheckContinuousVariables(
    CcuKernel& kernel, CcuVariableHandle varHandle, uint32_t num, const CcuRep::Variable& baseVar, const char* tag)
{
    if (num <= 1) {
        return CcuResult::CCU_SUCCESS;
    }
    for (uint32_t i = 1; i < num; i++) {
        CcuRep::Variable* nextVar{nullptr};
        CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle + i, &nextVar));
        if (nextVar->Id() != baseVar.Id() + i) {
            HCCL_ERROR(
                "[CcuKernel][%s] variables not continuous at index %u, "
                "expected Id %u but got %u",
                tag, i, baseVar.Id() + i, nextVar->Id());
            return HCCL_TO_CCU_RET(HCCL_E_PARA);
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_load::LoadVar(CcuKernel& kernel, uint64_t addr, CcuVariableHandle varHandle, uint32_t num)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[LoadVar] addr=0x%llx, varHandle=%llu, num=%u", addr, varHandle, num);
    CcuRep::Variable* var{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &var));
    CCU_CHK_RET(CheckContinuousVariables(kernel, varHandle, num, *var, "LoadVariable"));
    kernel.Append(std::make_shared<CcuRep::CcuRepLoad>(kernel.GetInsGenerator(), addr, *var, num));
    // 数据面 num>=2 走 Array<Variable>: microcode 会用到 [baseVReg, baseVReg+num)
    // 这段连续 Xn 且组内相对偏移固定, 交给 microcode 后端优化 Pass 2 作为 pinned 组.
    if (num >= MIN_PINNED_REG_GROUP_NUM) {
        kernel.AddPinnedRegGroup(*var, static_cast<uint16_t>(num));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult
ccu_load::LoadVarFromVarAddr(CcuKernel& kernel, CcuVariableHandle addrHandle, CcuVariableHandle varHandle, uint32_t num)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[LoadVarFromVarAddr] addrHandle=%llu, varHandle=%llu,"
        " num=%u",
        addrHandle, varHandle, num);
    CcuRep::Variable* addrVar{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(addrHandle, &addrVar));
    CcuRep::Variable* var{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &var));
    CCU_CHK_RET(CheckContinuousVariables(kernel, varHandle, num, *var, "LoadVar dst"));
    kernel.Append(std::make_shared<CcuRep::CcuRepLoadVar>(kernel.GetInsGenerator(), *addrVar, *var, num));
    if (num >= MIN_PINNED_REG_GROUP_NUM) {
        kernel.AddPinnedRegGroup(*var, static_cast<uint16_t>(num));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_load::LoadAddImm(
    CcuKernel& kernel, CcuVariableHandle varHandle, uint16_t srcNum, CcuVariableHandle offsetHandle,
    uint16_t immAddValue, CcuVariableHandle dstHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[LoadAddImm] varHandle=%llu, offsetHandle=%llu, immAddValue=%u, dstHandle=%llu", varHandle,
        offsetHandle, immAddValue, dstHandle);
    if (srcNum == 0) {
        HCCL_ERROR("[%s] srcNum must be greater than 0", __func__);
        return CcuResult::CCU_E_PARA;
    }
    CcuRep::Variable* baseVar(nullptr);
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &baseVar));
    CcuRep::Variable* offsetVar(nullptr);
    CCU_CHK_RET(kernel.Repo().ResolveVariable(offsetHandle, &offsetVar));
    CcuRep::Variable* dstVar(nullptr);
    CCU_CHK_RET(kernel.Repo().ResolveVariable(dstHandle, &dstVar));
    auto rep = std::make_shared<CcuRep::CcuRepLoadAddImm>(
        kernel.GetInsGenerator(), *baseVar, srcNum, *offsetVar, immAddValue, *dstVar);
    kernel.Append(rep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_load::AddImmStore(
    CcuKernel& kernel, CcuVariableHandle varHandle, uint16_t dstNum, CcuVariableHandle offsetHandle,
    uint16_t immAddValue, CcuVariableHandle srcHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[AddImmStore] varHandle=%llu, offsetHandle=%llu, immAddValue=%u, srcHandle=%llu", varHandle,
        offsetHandle, immAddValue, srcHandle);
    if (dstNum == 0) {
        HCCL_ERROR("[%s] dstNum must be greater than 0", __func__);
        return CcuResult::CCU_E_PARA;
    }
    CcuRep::Variable* baseVar(nullptr);
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &baseVar));
    CcuRep::Variable* offsetVar(nullptr);
    CCU_CHK_RET(kernel.Repo().ResolveVariable(offsetHandle, &offsetVar));
    CcuRep::Variable* srcVar(nullptr);
    CCU_CHK_RET(kernel.Repo().ResolveVariable(srcHandle, &srcVar));
    auto rep = std::make_shared<CcuRep::CcuRepStoreAddImm>(
        kernel.GetInsGenerator(), *baseVar, dstNum, *offsetVar, immAddValue, *srcVar);
    kernel.Append(rep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_load::StoreVar(CcuKernel& kernel, uint64_t addr, CcuVariableHandle varHandle, uint32_t num)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[StoreVar] addr=0x%llx, varHandle=%llu, num=%u", addr, varHandle, num);
    CcuRep::Variable* var{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &var));
    CCU_CHK_RET(CheckContinuousVariables(kernel, varHandle, num, *var, "StoreVariable"));
    kernel.Append(std::make_shared<CcuRep::CcuRepStore>(kernel.GetInsGenerator(), *var, addr, num));
    if (num >= MIN_PINNED_REG_GROUP_NUM) {
        kernel.AddPinnedRegGroup(*var, static_cast<uint16_t>(num));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult
ccu_load::StoreVarToVarAddr(CcuKernel& kernel, CcuVariableHandle addrHandle, CcuVariableHandle varHandle, uint32_t num)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[StoreVarToVarAddr] addrHandle=%llu, varHandle=%llu,"
        " num=%u",
        addrHandle, varHandle, num);
    CcuRep::Variable* addrVar{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(addrHandle, &addrVar));
    CcuRep::Variable* var{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(varHandle, &var));
    CCU_CHK_RET(CheckContinuousVariables(kernel, varHandle, num, *var, "StoreVar src"));
    kernel.Append(std::make_shared<CcuRep::CcuRepStoreVar>(kernel.GetInsGenerator(), *var, *addrVar, num));
    if (num >= MIN_PINNED_REG_GROUP_NUM) {
        kernel.AddPinnedRegGroup(*var, static_cast<uint16_t>(num));
    }
    return CcuResult::CCU_SUCCESS;
}

} // namespace hcomm
