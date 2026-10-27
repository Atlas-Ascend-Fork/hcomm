/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_REPRESENTATION_ADDR_FUNC_BLOCK_H
#define CCU_REPRESENTATION_ADDR_FUNC_BLOCK_H

#include "ccu_rep_block_v1.h"
#include "ccu_rep_arg_v1.h"
#include "ccu_rep_reference_manager_v1.h"

namespace hcomm {
namespace CcuRep {

    // 可寻址 func 自包含序言里保留资源的初始化指令数：reserveXn、reserveGsa(V2 下为 commXn)、reserveCke
    constexpr uint16_t FUNC_PRELUDE_FIXED_INSTR_NUM = 3;

    // 新 callfunc 链路专用：命名 / 可被 pc 寻址的 func 定义块。与原 callfunc<> 的 CcuRepFuncBlock
    // 完全独立，不共享基类以外的任何实现，入口处额外带一段自包含序言把中转寄存器拷进形参。
    class CcuRepAddressableFuncBlock : public CcuRepBlock {
    public:
        explicit CcuRepAddressableFuncBlock(CcuInsGeneratorBase* insGenPtr, const std::string& label);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        uint16_t InstrCount() override;

        void SetFuncManager(CcuRepReferenceManager* funcManager);

        // 命名 func 不支持出参，只有入参
        void DefineInArg(const Variable& var);
        void DefineInArg(const std::vector<Variable>& varList);

        uint16_t GetCallLayer() const;
        std::vector<Variable> GetInArgVars() const;

        // 入口序言把这批跨 kernel 共用的中转寄存器拷到本 func 的形参
        void SetTransferInVars(const std::vector<Variable>& vars);
        const std::vector<Variable>& GetTransferInVars() const;
        // 跨 kernel 共用的返回地址寄存器，尾声取它跳回调用点
        void SetSharedRetVar(const Variable& var);
        const Variable* GetSharedRetVar() const;
        // 自包含序言的指令数，随常量个数变化，须在常量资源确定后、统计指令数之前设置
        void SetPreludeInstrCount(uint16_t count);
        // 后端优化插 NOP 后回填入口下标；origToOut[优化前本地下标] = 优化后本地下标，负值表示无映射。
        // 无映射时不再静默保留旧 instrId，而是置失败态，由 GetFuncPc 拒绝发布过期 pc。
        void RemapStartInstrId(const std::vector<int32_t>& origToOut, uint16_t startId);
        bool RemapFailed() const { return remapFailed; }

        CcuRepReferenceManager* GetFuncManager() { return funcManager; }

        std::vector<CcuRepArg>& GetInArgs() { return inArgs; }

    private:
        CcuRepReferenceManager* funcManager{nullptr};

        std::vector<CcuRepArg> inArgs;
        uint64_t inArgCount{0};

        bool remapFailed{false};
        std::vector<Variable> transferInVars;
        // Variable 的 operator= 是会 emit 指令的 DSL 赋值，这里要的是共用同一个物理资源，只能走拷贝构造
        std::shared_ptr<Variable> sharedRetVar;
        uint16_t preludeInstrCount{0};

        uint16_t callLayer{0};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // CCU_REPRESENTATION_ADDR_FUNC_BLOCK_H
