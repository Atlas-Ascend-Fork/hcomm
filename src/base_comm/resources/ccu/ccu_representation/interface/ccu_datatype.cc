/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_datatype_v1.h"
#include "ccu_rep_v1.h"
#include "ccu_kernel_resource.h"
#include "ccu_interface_assist_v1.h"

#include "exception_util.h"
#include "ccu_api_exception.h"
#include "log.h"
#include "ccu_log.h"

namespace hcomm {
namespace CcuRep {

    uint16_t CcuPhyRes::Id() const { return id; }
    uint16_t CcuPhyRes::DieId() const { return dieId; }
    void CcuPhyRes::Reset(uint16_t id) { this->id = id; }
    void CcuPhyRes::SetDieId(uint16_t dieId) { this->dieId = dieId; }

    CcuVirRes::CcuVirRes(CcuRepContext* context) : context(context) { phyRes = std::make_shared<CcuPhyRes>(); }

    void CcuVirRes::Reset(uint16_t id) { phyRes->Reset(id); }

    void CcuVirRes::Reset(uint16_t id, uint16_t dieId)
    {
        phyRes->Reset(id);
        phyRes->SetDieId(dieId);
    }

    void CcuVirRes::SetDieId(uint16_t dieId) { phyRes->SetDieId(dieId); }

    uint16_t CcuVirRes::Id() const { return phyRes->Id(); }

    uint16_t CcuVirRes::DieId() const { return phyRes->DieId(); }

    Variable::Variable(CcuRepContext* context) : CcuVirRes(context) {}

    Variable::Variable(const Variable& other) : CcuVirRes(other.context) { phyRes = other.phyRes; }

    void Variable::operator=(Variable&& other)
    {
        phyRes = other.phyRes;
        context = other.context;
    }

    void Variable::operator=(const Variable& other)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepAssign>(context->GetInsGenerator(), *this, other)));
    }

    void Variable::operator=(uint64_t immediate)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepAssign>(context->GetInsGenerator(), *this, immediate)));
    }

    template <typename Dst, typename Lhs, typename Rhs>
    CcuResult ArithmeticAppendToContext(CcuRepContext* context, Dst& dst, CcuArithmeticOperator<Lhs, Rhs> op)
    {
        switch (op.type) {
            case CcuArithmeticOperatorType::ADDITION:
                CCU_CHK_RET(AppendToContext(
                    context, std::make_shared<CcuRepAdd>(context->GetInsGenerator(), dst, op.lhs, op.rhs)));
                break;
            case CcuArithmeticOperatorType::MULTIPLICATION:
                CCU_CHK_RET(AppendToContext(
                    context, std::make_shared<CcuRepMul>(context->GetInsGenerator(), dst, op.lhs, op.rhs)));
                break;
            case CcuArithmeticOperatorType::SUBTRACTION:
                CCU_CHK_RET(AppendToContext(
                    context, std::make_shared<CcuRepSub>(context->GetInsGenerator(), dst, op.lhs, op.rhs)));
                break;
            default:
                HCCL_ERROR("[%s] Not supported Arithmetic operator[%d].", __func__, op.type);
                return CCU_E_NOT_SUPPORT;
        }
        return CCU_SUCCESS;
    }

    CcuResult Variable::VarVarAppendToContext(CcuArithmeticOperator<Variable, Variable> op)
    {
        CCU_CHK_RET(ArithmeticAppendToContext(context, *this, op));
        return CCU_SUCCESS;
    }

    void Variable::operator=(CcuArithmeticOperator<Variable, Variable> op)
    {
        CCU_CHK_RET_THROW(Hccl::CcuApiException, "VarVarAppendToContext failed", VarVarAppendToContext(op));
    }

    CcuResult Variable::VarImmedAppendToContext(CcuArithmeticOperator<Variable, uint16_t> op)
    {
        CCU_CHK_RET(ArithmeticAppendToContext(context, *this, op));
        return CCU_SUCCESS;
    }

    void Variable::operator=(CcuArithmeticOperator<Variable, uint16_t> op)
    {
        CCU_CHK_RET_THROW(Hccl::CcuApiException, "VarImmedAppendToContext failed", VarImmedAppendToContext(op));
    }

    CcuResult Variable::AddrImmedAppendToContext(CcuArithmeticOperator<Address, uint16_t> op)
    {
        CCU_CHK_RET(ArithmeticAppendToContext(context, *this, op));
        return CCU_SUCCESS;
    }

    void Variable::operator=(CcuArithmeticOperator<Address, uint16_t> op)
    {
        CCU_CHK_RET_THROW(Hccl::CcuApiException, "AddrImmedAppendToContext failed", AddrImmedAppendToContext(op));
    }

    CcuResult Variable::AddrAddrAppendToContext(CcuArithmeticOperator<Address, Address> op)
    {
        switch (op.type) {
            case CcuArithmeticOperatorType::ADDITION: {
                CCU_CHK_RET(AppendToContext(
                    context, std::make_shared<CcuRepAdd>(context->GetInsGenerator(), *this, op.lhs, op.rhs)));
                break;
            }
            default: {
                HCCL_ERROR("[%s] Not supported Arithmetic operate in variable = addr (op) addr.", __func__);
                return CCU_E_NOT_SUPPORT;
            }
        }
        return CCU_SUCCESS;
    }

    void Variable::operator=(CcuArithmeticOperator<Address, Address> op)
    {
        CCU_CHK_RET_THROW(Hccl::CcuApiException, "AddrAddrAppendToContext failed", AddrAddrAppendToContext(op));
    }

    void Variable::operator+=(const Variable& other)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepAdd>(context->GetInsGenerator(), *this, other)));
    }

    void Variable::operator+=(const uint16_t immediate)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepAdd>(context->GetInsGenerator(), *this, immediate)));
    }

    void Variable::operator*=(const Variable& other)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepMul>(context->GetInsGenerator(), *this, other)));
    }

    void Variable::operator*=(const uint16_t immediate)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepMul>(context->GetInsGenerator(), *this, immediate)));
    }

    void Variable::operator-=(const Variable& other)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepSub>(context->GetInsGenerator(), *this, other)));
    }

    void Variable::operator-=(const uint16_t immediate)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepSub>(context->GetInsGenerator(), *this, immediate)));
    }

    void Variable::operator&=(const Variable& other)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepAnd>(context->GetInsGenerator(), *this, other)));
    }

    void Variable::operator|=(const Variable& other)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepOr>(context->GetInsGenerator(), *this, other)));
    }

    void Variable::operator^=(const Variable& other)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepXor>(context->GetInsGenerator(), *this, other)));
    }

    CcuResult Variable::VarVarLogicAppendToContext(CcuLogicOperator<Variable, Variable> op)
    {
        switch (op.type) {
            case CcuLogicOperatorType::AND: {
                CCU_CHK_RET(AppendToContext(
                    context, std::make_shared<CcuRepAnd>(context->GetInsGenerator(), *this, op.lhs, op.rhs)));
                break;
            }
            case CcuLogicOperatorType::OR: {
                CCU_CHK_RET(AppendToContext(
                    context, std::make_shared<CcuRepOr>(context->GetInsGenerator(), *this, op.lhs, op.rhs)));
                break;
            }
            case CcuLogicOperatorType::XOR: {
                CCU_CHK_RET(AppendToContext(
                    context, std::make_shared<CcuRepXor>(context->GetInsGenerator(), *this, op.lhs, op.rhs)));
                break;
            }
            default: {
                HCCL_ERROR("[%s] Not supported Logic operate between var and var", __func__);
                return CCU_E_NOT_SUPPORT;
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult Variable::AddrLogicAppendToContext(CcuLogicOperator<Variable> op)
    {
        switch (op.type) {
            case CcuLogicOperatorType::NOT: {
                CCU_CHK_RET(
                    AppendToContext(context, std::make_shared<CcuRepNot>(context->GetInsGenerator(), *this, op.lhs)));
                break;
            }
            default: {
                HCCL_ERROR("[%s] Not supported Logic operate between addr and immedB.", __func__);
                return CCU_E_NOT_SUPPORT;
            }
        }
        return CCU_SUCCESS;
    }

    void Variable::operator=(CcuLogicOperator<Variable, Variable> op)
    {
        CCU_CHK_RET_THROW(Hccl::CcuApiException, "VarVarLogicAppendToContext failed", VarVarLogicAppendToContext(op));
    }

    void Variable::operator=(CcuLogicOperator<Variable> op)
    {
        CCU_CHK_RET_THROW(Hccl::CcuApiException, "AddrLogicAppendToContext failed", AddrLogicAppendToContext(op));
    }

    void Variable::operator=(CcuShiftOperator<Variable, Variable> op)
    {
        switch (op.type) {
            case CcuShiftType::RIGHT: {
                CCU_CHK_RET_THROW(
                    Hccl::CcuApiException, "AppendToContext failed",
                    AppendToContext(
                        context, std::make_shared<CcuRepShR>(context->GetInsGenerator(), *this, op.lhs, op.rhs)));
                break;
            }
            case CcuShiftType::LEFT: {
                CCU_CHK_RET_THROW(
                    Hccl::CcuApiException, "AppendToContext failed",
                    AppendToContext(
                        context, std::make_shared<CcuRepShL>(context->GetInsGenerator(), *this, op.lhs, op.rhs)));
                break;
            }
            default: {
                Hccl::THROW<Hccl::NotSupportException>("Not supported shift bit operate between var and immed.");
            }
        }
    }

    void Variable::operator<<=(const Variable& other) const
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepShL>(context->GetInsGenerator(), *this, other)));
    }

    void Variable::operator>>=(const Variable& other) const
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepShR>(context->GetInsGenerator(), *this, other)));
    }

    Address::Address(CcuRepContext* context) : CcuVirRes(context) {}

    Address::Address(const Address& other) : CcuVirRes(other.context) { phyRes = other.phyRes; }

    Address::Address(Variable&& other) : CcuVirRes(other.GetCurContext()) { phyRes = other.GetCurPhyRes(); }

    void Address::operator=(Address&& other)
    {
        phyRes = other.phyRes;
        context = other.context;
    }

    void Address::operator=(const Address& other)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepAssign>(context->GetInsGenerator(), *this, other)));
    }

    void Address::operator=(const Variable& other)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepAssign>(context->GetInsGenerator(), *this, other)));
    }

    void Address::operator=(uint64_t immediate)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepAssign>(context->GetInsGenerator(), *this, immediate)));
    }

    CcuResult Address::VarAddrAppendToContext(CcuArithmeticOperator<Variable, Address> op)
    {
        switch (op.type) {
            case CcuArithmeticOperatorType::ADDITION: {
                CCU_CHK_RET(AppendToContext(
                    context, std::make_shared<CcuRepAdd>(context->GetInsGenerator(), *this, op.rhs, op.lhs)));
                break;
            }
            case CcuArithmeticOperatorType::MULTIPLICATION: {
                CCU_CHK_RET(AppendToContext(
                    context, std::make_shared<CcuRepMul>(context->GetInsGenerator(), *this, op.lhs, op.rhs)));
                break;
            }
            case CcuArithmeticOperatorType::SUBTRACTION: {
                CCU_CHK_RET(AppendToContext(
                    context, std::make_shared<CcuRepSub>(context->GetInsGenerator(), *this, op.rhs, op.lhs)));
                break;
            }
            default: {
                HCCL_ERROR("[%s] Not supported Arithmetic operate between var and addr.", __func__);
                return CCU_E_NOT_SUPPORT;
            }
        }
        return CCU_SUCCESS;
    }

    void Address::operator=(CcuArithmeticOperator<Variable, Address> op)
    {
        CCU_CHK_RET_THROW(Hccl::CcuApiException, "VarAddrAppendToContext failed", VarAddrAppendToContext(op));
    }

    CcuResult Address::AddrAddrAppendToContext(CcuArithmeticOperator<Address, Address> op)
    {
        switch (op.type) {
            case CcuArithmeticOperatorType::ADDITION: {
                CCU_CHK_RET(AppendToContext(
                    context, std::make_shared<CcuRepAdd>(context->GetInsGenerator(), *this, op.lhs, op.rhs)));
                break;
            }
            default: {
                HCCL_ERROR("[%s] Not supported Arithmetic operate in addr = addr (op) addr .", __func__);
                return CCU_E_NOT_SUPPORT;
            }
        }
        return CCU_SUCCESS;
    }

    void Address::operator=(CcuArithmeticOperator<Address, Address> op)
    {
        CCU_CHK_RET_THROW(Hccl::CcuApiException, "AddrAddrAppendToContext failed", AddrAddrAppendToContext(op));
    }

    CcuResult Address::VarImmedAppendToContext(CcuArithmeticOperator<Variable, uint16_t> op)
    {
        CCU_CHK_RET(ArithmeticAppendToContext(context, *this, op));
        return CCU_SUCCESS;
    }

    void Address::operator=(CcuArithmeticOperator<Variable, uint16_t> op)
    {
        CCU_CHK_RET_THROW(Hccl::CcuApiException, "VarImmedAppendToContext failed", VarImmedAppendToContext(op));
    }

    CcuResult Address::AddrImmedAppendToContext(CcuArithmeticOperator<Address, uint16_t> op)
    {
        CCU_CHK_RET(ArithmeticAppendToContext(context, *this, op));
        return CCU_SUCCESS;
    }

    void Address::operator=(CcuArithmeticOperator<Address, uint16_t> op)
    {
        CCU_CHK_RET_THROW(Hccl::CcuApiException, "AddrImmedAppendToContext failed", AddrImmedAppendToContext(op));
    }

    CcuResult Address::VarVarAppendToContext(CcuArithmeticOperator<Variable, Variable> op)
    {
        switch (op.type) {
            case CcuArithmeticOperatorType::ADDITION: {
                CCU_CHK_RET(AppendToContext(
                    context, std::make_shared<CcuRepAdd>(context->GetInsGenerator(), *this, op.lhs, op.rhs)));
                break;
            }
            case CcuArithmeticOperatorType::MULTIPLICATION: {
                CCU_CHK_RET(AppendToContext(
                    context, std::make_shared<CcuRepMul>(context->GetInsGenerator(), *this, op.lhs, op.rhs)));
                break;
            }
            default: {
                HCCL_ERROR("[%s] Not supported Arithmetic operate between var and var.", __func__);
                return CCU_E_NOT_SUPPORT;
            }
        }
        return CCU_SUCCESS;
    }

    void Address::operator=(CcuArithmeticOperator<Variable, Variable> op)
    {
        CCU_CHK_RET_THROW(Hccl::CcuApiException, "VarVarAppendToContext failed", VarVarAppendToContext(op));
    }

    void Address::operator+=(const Variable& other)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepAdd>(context->GetInsGenerator(), *this, other)));
    }

    void Address::operator+=(const uint16_t immediate)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepAdd>(context->GetInsGenerator(), *this, immediate)));
    }

    void Address::operator*=(const Variable& other)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepMul>(context->GetInsGenerator(), *this, other)));
    }

    void Address::operator*=(const uint16_t immediate)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepMul>(context->GetInsGenerator(), *this, immediate)));
    }

    void Address::operator-=(const Variable& other)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepSub>(context->GetInsGenerator(), *this, other)));
    }

    void Address::operator-=(const uint16_t immediate)
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepSub>(context->GetInsGenerator(), *this, immediate)));
    }

    void Address::operator=(CcuShiftOperator<Variable, Variable> op)
    {
        switch (op.type) {
            case CcuShiftType::LEFT: {
                CCU_CHK_RET_THROW(
                    Hccl::CcuApiException, "AppendToContext failed",
                    AppendToContext(
                        context, std::make_shared<CcuRepShL>(context->GetInsGenerator(), *this, op.lhs, op.rhs)));
                break;
            }
            case CcuShiftType::RIGHT: {
                CCU_CHK_RET_THROW(
                    Hccl::CcuApiException, "AppendToContext failed",
                    AppendToContext(
                        context, std::make_shared<CcuRepShR>(context->GetInsGenerator(), *this, op.lhs, op.rhs)));
                break;
            }
            default: {
                Hccl::THROW<Hccl::NotSupportException>("Not supported shift bit operate between var and var.");
            }
        }
    }

    void Address::operator<<=(const Variable& other) const
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepShL>(context->GetInsGenerator(), *this, other)));
    }

    void Address::operator>>=(const Variable& other) const
    {
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "AppendToContext failed",
            AppendToContext(context, std::make_shared<CcuRepShR>(context->GetInsGenerator(), *this, other)));
    }

    LocalNotify::LocalNotify(CcuRepContext* context) : CcuVirRes(context) {}

    CcuBuffer::CcuBuffer(CcuRepContext* context) : CcuVirRes(context) {}

    CcuBuf::CcuBuf(CcuRepContext* context) : CcuVirRes(context) {}
    uint16_t CcuBuffer::Id() const { return phyRes->Id() + CCUBUFFER_DIE_ID_BIT * phyRes->DieId(); }

    uint16_t CcuBuf::Id() const { return phyRes->Id() + CCUBUFFER_DIE_ID_BIT * phyRes->DieId(); }

    Executor::Executor(CcuRepContext* context) : CcuVirRes(context) {}

    CompletedEvent::CompletedEvent(CcuRepContext* context) : CcuVirRes(context) {}

}; // namespace CcuRep
}; // namespace hcomm
