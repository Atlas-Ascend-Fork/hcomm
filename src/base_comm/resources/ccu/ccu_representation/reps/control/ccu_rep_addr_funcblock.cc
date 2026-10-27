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
#include "ccu_rep_reference_manager_v1.h"
#include "ccu_rep_translator_v1.h"

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

#include "ccu_ins_generator_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

    using namespace Hccl;

    CcuRepAddressableFuncBlock::CcuRepAddressableFuncBlock(CcuInsGeneratorBase* insGenPtr, const std::string& label)
        : CcuRepBlock(insGenPtr, label)
    {
        type = CcuRepType::ADDRESSABLE_FUNC_BLOCK;
        instrCount = 0;
    }

    std::string CcuRepAddressableFuncBlock::Describe()
    {
        return Hccl::StringFormat("AddressableFuncBlock[%s]", GetLabel().c_str());
    }

    void CcuRepAddressableFuncBlock::SetFuncManager(CcuRepReferenceManager* funcManager)
    {
        this->funcManager = funcManager;
    }

    uint16_t CcuRepAddressableFuncBlock::GetCallLayer() const { return callLayer; }

    void CcuRepAddressableFuncBlock::DefineInArg(const Variable& var)
    {
        inArgCount++;
        inArgs.push_back(CcuRepArg(var));
        HCCL_INFO("Define Input Arg: Index[%u], Type[Variable] Id[%u]", inArgs.size(), var.Id());
    }

    void CcuRepAddressableFuncBlock::DefineInArg(const std::vector<Variable>& varList)
    {
        inArgCount += varList.size();
        inArgs.push_back(CcuRepArg(varList));
        HCCL_INFO("Define Input Arg: Index[%u], Type[Variable List]: ", inArgs.size());
        for (uint32_t index = 0; index < varList.size(); index++) {
            HCCL_INFO("    Index[%u].Id[%u]", index, varList[index].Id());
        }
    }

    void CcuRepAddressableFuncBlock::SetTransferInVars(const std::vector<Variable>& vars) { transferInVars = vars; }

    const std::vector<Variable>& CcuRepAddressableFuncBlock::GetTransferInVars() const { return transferInVars; }

    void CcuRepAddressableFuncBlock::SetSharedRetVar(const Variable& var)
    {
        sharedRetVar = std::make_shared<Variable>(var);
    }

    const Variable* CcuRepAddressableFuncBlock::GetSharedRetVar() const { return sharedRetVar.get(); }

    void CcuRepAddressableFuncBlock::SetPreludeInstrCount(uint16_t count) { this->preludeInstrCount = count; }

    // 换算与 InstructionScheduler::FixReferences 的 remapGlobal 一致
    void CcuRepAddressableFuncBlock::RemapStartInstrId(const std::vector<int32_t>& origToOut, uint16_t startId)
    {
        if (instrId < startId) {
            return;
        }
        const uint32_t localId = static_cast<uint32_t>(instrId) - static_cast<uint32_t>(startId);
        if (localId >= origToOut.size() || origToOut[localId] < 0) {
            remapFailed = true;
            HCCL_ERROR(
                "[CcuRepAddressableFuncBlock::RemapStartInstrId] func %s has no mapping for local instr %u, "
                "stale pc %hu will be rejected by GetFuncPc",
                GetLabel().c_str(), localId, instrId);
            return;
        }
        remapFailed = false;
        instrId = static_cast<uint16_t>(static_cast<uint32_t>(startId) + static_cast<uint32_t>(origToOut[localId]));
    }

    std::vector<Variable> CcuRepAddressableFuncBlock::GetInArgVars() const
    {
        std::vector<Variable> vars;
        for (const auto& arg : inArgs) {
            if (arg.type == CcuArgType::VARIABLE) {
                vars.push_back(arg.var);
            } else if (arg.type == CcuArgType::VARIABLE_LIST) {
                vars.insert(vars.end(), arg.varList.begin(), arg.varList.end());
            }
        }
        return vars;
    }

    uint16_t CcuRepAddressableFuncBlock::InstrCount()
    {
        instrCount = CcuRepBlock::InstrCount() + inArgCount
                     + insGeneratorPtr_->GetInstrCount(type) // FuncBlock需要额外指令
                     + preludeInstrCount;                    // 可寻址func入口的自包含序言
        return instrCount;
    }

    bool CcuRepAddressableFuncBlock::Translate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        if (funcManager == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("funcManager is nullptr");
        }

        this->instrId = instrId;
        translated = true;

        CHK_PRT_THROW(
            insGeneratorPtr_->CcuRepAddressableFuncBlockTranslate(ccuKernel, instr, instrId, this, dep, 0)
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CcuRepAddressableFuncBlock][Translate] failed to translate inArgs processing for instrId[%u]",
                instrId),
            Hccl::CcuApiException, "CcuRepAddressableFuncBlock translate failed");
        // 使用空实现的自定义删除器，避免智能指针析构时释放对象
        auto translator = CcuRepTranslator(
            std::shared_ptr<CcuRepReferenceManager>(funcManager, []([[maybe_unused]] CcuRepReferenceManager* ptr) {}),
            dep);
        translator.Translate(
            ccuKernel, GetReps(), instr, instrId, []([[maybe_unused]] std::shared_ptr<CcuRepBase> rep) -> bool {
                return true;
            });

        CHK_PRT_THROW(
            insGeneratorPtr_->CcuRepAddressableFuncBlockTranslate(ccuKernel, instr, instrId, this, dep, 1)
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CcuRepAddressableFuncBlock][Translate] failed to translate outArgs processing for instrId[%u]",
                instrId),
            Hccl::CcuApiException, "CcuRepAddressableFuncBlock translate failed");

        return translated;
    }

}; // namespace CcuRep
}; // namespace hcomm
