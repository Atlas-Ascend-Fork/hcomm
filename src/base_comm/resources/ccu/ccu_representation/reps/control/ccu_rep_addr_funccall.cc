/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_v1.h"

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

#include "ccu_ins_generator_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

    using namespace Hccl;

    CcuRepAddrFuncCall::CcuRepAddrFuncCall(CcuInsGeneratorBase* insGenPtr, const Variable& funcAddrVar)
        : insGeneratorPtr_(insGenPtr),
          funcAddrVar(funcAddrVar)
    {
        type = CcuRepType::ADDR_FUNC_CALL;
        instrCount = 0;
    }

    void CcuRepAddrFuncCall::SetInArg(const Variable& var)
    {
        inArgCount++;
        if (inArgCount > CCU_FUNC_XKERNEL_ARG_MAX) {
            Hccl::THROW<Hccl::CcuApiException>("CcuAddrFuncCall Max ArgCount = %u", CCU_FUNC_XKERNEL_ARG_MAX);
        }
        inArgs.push_back(CcuRepArg(var));
    }

    void CcuRepAddrFuncCall::SetInArg(const std::vector<Variable>& varList)
    {
        inArgCount += varList.size();
        if (inArgCount > CCU_FUNC_XKERNEL_ARG_MAX) {
            Hccl::THROW<Hccl::CcuApiException>("CcuAddrFuncCall Max ArgCount = %u", CCU_FUNC_XKERNEL_ARG_MAX);
        }
        inArgs.push_back(CcuRepArg(varList));
    }

    void CcuRepAddrFuncCall::SetTransferInVars(const std::vector<Variable>& vars) { transferInVars = vars; }

    const std::vector<Variable>& CcuRepAddrFuncCall::GetTransferInVars() const { return transferInVars; }

    void CcuRepAddrFuncCall::SetSharedRetVar(const Variable& var) { sharedRetVar = std::make_shared<Variable>(var); }

    const Variable* CcuRepAddrFuncCall::GetSharedRetVar() const { return sharedRetVar.get(); }

    void CcuRepAddrFuncCall::SetJumpScratchVar(const Variable& var)
    {
        jumpScratchVar = std::make_shared<Variable>(var);
    }

    const Variable* CcuRepAddrFuncCall::GetJumpScratchVar() const { return jumpScratchVar.get(); }

    uint16_t CcuRepAddrFuncCall::InstrCount()
    {
        instrCount = inArgCount + insGeneratorPtr_->GetInstrCount(type); // 除入参处理外需要的额外指令
        return instrCount;
    }

    bool CcuRepAddrFuncCall::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        if (this->instr == nullptr) {
            this->instrId = instrId;
            this->instr = instr;
            instr += InstrCount();
            instrId += InstrCount();
        }

        translated = true;

        CHK_PRT_THROW(
            insGeneratorPtr_->CcuRepAddrFuncCallTranslate(ccuKernel, instr, instrId, this, dep)
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepAddrFuncCall][Translate] failed to translate for instrId[%u]", instrId),
            Hccl::CcuApiException, "CcuRepAddrFuncCall translate failed");

        return translated;
    }

    std::string CcuRepAddrFuncCall::Describe() { return Hccl::StringFormat("AddrFuncCall[pc]"); }

}; // namespace CcuRep
}; // namespace hcomm
