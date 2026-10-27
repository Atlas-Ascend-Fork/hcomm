/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_REPRESENTATION_ADDR_FUNC_CALL_H
#define CCU_REPRESENTATION_ADDR_FUNC_CALL_H

#include "ccu_rep_base_v1.h"

namespace hcomm {
namespace CcuRep {

    // 新 callfunc 链路专用：命名 func 的按 pc 间接调用。入参统一写入跨 kernel 共用的中转寄存器，返回
    // 地址写进共享返回寄存器，目标 pc 装进共享跳转暂存寄存器后跳过去；被调 func 的入口序言再把中转
    // 寄存器拷进自己的形参。全部寄存器都来自按 die 分配的 Addr 共享资源，不依赖 CcuRepReferenceManager，
    // 与原 callfunc<> 的 CcuRepFuncCall 完全独立。
    class CcuRepAddrFuncCall : public CcuRepBase {
    public:
        explicit CcuRepAddrFuncCall(CcuInsGeneratorBase* insGenPtr, const Variable& funcAddrVar);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        uint16_t InstrCount() override;

        void SetInArg(const Variable& var);
        void SetInArg(const std::vector<Variable>& varList);

        // 组装期先登记，具体中转/返回地址/跳转暂存寄存器等 die 落定后由 BindFuncAddr 回填。
        void SetTransferInVars(const std::vector<Variable>& vars);
        const std::vector<Variable>& GetTransferInVars() const;
        void SetSharedRetVar(const Variable& var);
        const Variable* GetSharedRetVar() const;
        void SetJumpScratchVar(const Variable& var);
        const Variable* GetJumpScratchVar() const;

        Variable GetFuncAddrVar() { return funcAddrVar; }
        std::vector<CcuRepArg>& GetInArgs() { return inArgs; }
        uint32_t GetInArgCount() const { return inArgCount; }
        CcuInstr* GetInstr() { return instr; }

    private:
        CcuInsGeneratorBase* insGeneratorPtr_;

        Variable funcAddrVar;

        std::vector<CcuRepArg> inArgs;
        uint32_t inArgCount{0};

        std::vector<Variable> transferInVars;
        // Variable 的 operator= 是会 emit 指令的 DSL 赋值，这里要的是共用同一个物理资源，只能走拷贝构造
        std::shared_ptr<Variable> sharedRetVar;
        std::shared_ptr<Variable> jumpScratchVar;

        CcuInstr* instr{nullptr};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // CCU_REPRESENTATION_ADDR_FUNC_CALL_H
