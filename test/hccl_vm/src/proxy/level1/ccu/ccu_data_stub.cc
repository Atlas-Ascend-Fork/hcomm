/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define HCCL_VM_MODULE "CCU_DATA_STUB"

#include <cstdint>

#include "ccu_primitives_impl.h"
#include "sim_log.h"
#include "ccu_kernel_sim.h"

namespace {
using HcclSim::CcuSim::CheckAddrHandle;
using HcclSim::CcuSim::CheckBufHandle;
using HcclSim::CcuSim::CheckEventHandle;
using HcclSim::CcuSim::CheckLocalAddrHandle;
using HcclSim::CcuSim::CheckRemoteAddrHandle;
using HcclSim::CcuSim::CheckVarHandle;
using HcclSim::CcuSim::SimCcuHandleTag;
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

/*========== Alloc 相关接口 ==========*/

/** @brief 分配变量（XN 寄存器）对象：签发 VAR tag 句柄。
 *  exit 打印带调用者地址（libhccl 内模板代码），离线 addr2line 可反查该变量在模板源码的分配行。
 *  @return 成功返回 CCU_SUCCESS。 */
CcuResult CcuVariableAlloc(CcuVariableHandle* varHandle)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (varHandle == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    *varHandle = HcclSim::CcuSim::AllocHandle(SimCcuHandleTag::VAR);
    HCCL_VM_INFO(
        "{}: exit, varHandle=0x{:x}, caller=0x{:x}", __func__, *varHandle,
        reinterpret_cast<uint64_t>(__builtin_return_address(0)));
    return CcuResult::CCU_SUCCESS;
}

/** @brief 分配地址对象：签发 ADDR tag 句柄。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuAddressAlloc(CcuAddressHandle* addrHandle)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (addrHandle == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    *addrHandle = HcclSim::CcuSim::AllocHandle(SimCcuHandleTag::ADDR);
    HCCL_VM_INFO(
        "{}: exit, addrHandle=0x{:x}, caller=0x{:x}", __func__, *addrHandle,
        reinterpret_cast<uint64_t>(__builtin_return_address(0)));
    return CcuResult::CCU_SUCCESS;
}

/** @brief 分配事件（CKE）对象：签发全局唯一 EVENT tag 句柄（兼作 notifyId 编码原料）。
 *  @return 成功返回 CCU_SUCCESS。 */
CcuResult CcuEventAlloc(CcuEventHandle* eventHandle)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (eventHandle == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    *eventHandle = HcclSim::CcuSim::AllocEventHandle();
    HCCL_VM_INFO("{}: exit, eventHandle=0x{:x}", __func__, *eventHandle);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 分配 MS 缓冲对象：签发 BUF tag 句柄（伪地址惰性分配）。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuBufferAlloc(CcuBufferHandle* bufHandle)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (bufHandle == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    *bufHandle = HcclSim::CcuSim::AllocHandle(SimCcuHandleTag::BUF);
    HCCL_VM_INFO("{}: exit, bufHandle=0x{:x}", __func__, *bufHandle);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 组合本地内存访问三元组（地址 + token）。
 *  addrHandle/tokenHandle 为出参：为该 LocalAddr 签发专属 ADDR/VAR 句柄并写回——
 *  对齐真机固件语义（V2 下内嵌 Address 以 NoAllocTag 构造、句柄由本接口分配写回，
 *  后续 CcuAddressAssignVar 等按专属句柄寻址，每对象独立"地址寄存器"），
 *  三元组以专属句柄记录组合关系。
 *  @return 成功返回 CCU_SUCCESS。 */
CcuResult
CcuLocalAddrAlloc(CcuLocalAddrHandle* localAddrHandle, CcuAddressHandle* addrHandle, CcuVariableHandle* tokenHandle)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (localAddrHandle == nullptr || addrHandle == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    *localAddrHandle = HcclSim::CcuSim::AllocHandle(SimCcuHandleTag::LADDR);
    *addrHandle = HcclSim::CcuSim::AllocHandle(SimCcuHandleTag::ADDR);
    if (tokenHandle != nullptr) {
        *tokenHandle = HcclSim::CcuSim::AllocHandle(SimCcuHandleTag::VAR);
    }
    HcclSim::CcuSim::PrimLocalAddrAlloc(*localAddrHandle, *addrHandle, (tokenHandle != nullptr) ? *tokenHandle : 0U);
    HCCL_VM_INFO(
        "{}: exit, localAddrHandle=0x{:x}, addrHandle=0x{:x}, tokenHandle=0x{:x}", __func__, *localAddrHandle,
        *addrHandle, (tokenHandle != nullptr) ? *tokenHandle : 0U);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 组合远端内存访问三元组（地址 + token）。
 *  addrHandle/tokenHandle 为出参：为该 RemoteAddr 签发专属 ADDR/VAR 句柄并写回
 *  （出参语义与 CcuLocalAddrAlloc 一致），三元组以专属句柄记录组合关系。
 *  @return 成功返回 CCU_SUCCESS。 */
CcuResult
CcuRemoteAddrAlloc(CcuRemoteAddrHandle* remoteAddrHandle, CcuAddressHandle* addrHandle, CcuVariableHandle* tokenHandle)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (remoteAddrHandle == nullptr || addrHandle == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    *remoteAddrHandle = HcclSim::CcuSim::AllocHandle(SimCcuHandleTag::RADDR);
    *addrHandle = HcclSim::CcuSim::AllocHandle(SimCcuHandleTag::ADDR);
    if (tokenHandle != nullptr) {
        *tokenHandle = HcclSim::CcuSim::AllocHandle(SimCcuHandleTag::VAR);
    }
    HcclSim::CcuSim::PrimRemoteAddrAlloc(*remoteAddrHandle, *addrHandle, (tokenHandle != nullptr) ? *tokenHandle : 0U);
    HCCL_VM_INFO(
        "{}: exit, remoteAddrHandle=0x{:x}, addrHandle=0x{:x}, tokenHandle=0x{:x}", __func__, *remoteAddrHandle,
        *addrHandle, (tokenHandle != nullptr) ? *tokenHandle : 0U);
    return CcuResult::CCU_SUCCESS;
}

/*========== BlockAlloc 相关接口 ==========*/

/** @brief 连续块分配变量（句柄低 32 位连续，XN 连续块语义）。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuBlockVariableAlloc(CcuVariableHandle* varHandles, uint32_t count)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (varHandles == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    for (uint32_t i = 0; i < count; i++) {
        varHandles[i] = HcclSim::CcuSim::AllocHandle(SimCcuHandleTag::VAR);
    }
    HCCL_VM_INFO(
        "{}: exit, count={}, first=0x{:x}, last=0x{:x}", __func__, count, (count > 0U) ? varHandles[0] : 0U,
        (count > 0U) ? varHandles[count - 1U] : 0U);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 连续块分配事件（CKE）。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuBlockEventAlloc(CcuEventHandle* eventHandles, uint32_t count)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (eventHandles == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    for (uint32_t i = 0; i < count; i++) {
        eventHandles[i] = HcclSim::CcuSim::AllocEventHandle();
    }
    HCCL_VM_INFO(
        "{}: exit, count={}, first=0x{:x}, last=0x{:x}", __func__, count, (count > 0U) ? eventHandles[0] : 0U,
        (count > 0U) ? eventHandles[count - 1U] : 0U);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 连续块分配缓冲。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuBlockBufferAlloc(CcuBufferHandle* bufHandles, uint32_t count)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (bufHandles == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    for (uint32_t i = 0; i < count; i++) {
        bufHandles[i] = HcclSim::CcuSim::AllocHandle(SimCcuHandleTag::BUF);
    }
    HCCL_VM_INFO(
        "{}: exit, count={}, first=0x{:x}, last=0x{:x}", __func__, count, (count > 0U) ? bufHandles[0] : 0U,
        (count > 0U) ? bufHandles[count - 1U] : 0U);
    return CcuResult::CCU_SUCCESS;
}

/*========== 通道/实例级句柄派生接口 ==========*/

/** @brief 以通道对端 XN 派生变量：值依赖对端写入，回放按默认 0 解释。
 *  @return 成功返回 CCU_SUCCESS。 */
CcuResult CcuVariableCreateByChannel(ChannelHandle channel, uint32_t varIndex, CcuVariableHandle* varHandle)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (varHandle == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    *varHandle = HcclSim::CcuSim::AllocHandle(SimCcuHandleTag::VAR);
    // 登记通道派生变量来源：回放期读方据此识别交换变量并懒解析（部分交换）。
    HcclSim::CcuSim::NoteChannelVarSource(*varHandle, channel, varIndex);
    HCCL_VM_INFO(
        "{}: exit, varHandle=0x{:x}, channel={}, varIndex={} (channel-derived xn, exchange-resolved)", __func__,
        *varHandle, channel, varIndex);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 取实例级预约组内第 index 个变量（acq 句柄来自资源桩，0 视为空放行）。
 *  @return 成功返回 CCU_SUCCESS。 */
CcuResult CcuVariableGetByIndex(CcuVariableHandle acqHandle, uint32_t index, CcuVariableHandle* varHandle)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (varHandle == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    *varHandle = HcclSim::CcuSim::AllocHandle(SimCcuHandleTag::VAR);
    HCCL_VM_INFO("{}: exit, varHandle=0x{:x}, acqHandle=0x{:x}, index={}", __func__, *varHandle, acqHandle, index);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 取实例级预约组内第 index 个事件。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuEventGetByIndex(CcuEventHandle acqHandle, uint32_t index, CcuEventHandle* eventHandle)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (eventHandle == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    *eventHandle = HcclSim::CcuSim::AllocEventHandle();
    HCCL_VM_INFO("{}: exit, eventHandle=0x{:x}, acqHandle=0x{:x}, index={}", __func__, *eventHandle, acqHandle, index);
    return CcuResult::CCU_SUCCESS;
}

/*========== Variable 操作类相关接口 ==========*/

/** @brief 变量赋立即数（uint64 全宽）。@return 成功返回 CCU_SUCCESS；句柄类型不符返回 CCU_E_PARA。 */
CcuResult CcuVariableAssignImm(CcuVariableHandle resVar, uint64_t immediate)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(resVar)) {
        HCCL_VM_ERROR("{}: resVar=0x{:x} is not a variable handle", __func__, resVar);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimVarAssignImm(resVar, immediate);
    HCCL_VM_INFO("{}: exit, resVar=0x{:x}, immediate={} (0x{:x})", __func__, resVar, immediate, immediate);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 变量赋值。@return 成功返回 CCU_SUCCESS；句柄类型不符返回 CCU_E_PARA。 */
CcuResult CcuVariableAssignVar(CcuVariableHandle dstVarHandle, CcuVariableHandle srcVarHandle)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(dstVarHandle) || !CheckVarHandle(srcVarHandle)) {
        HCCL_VM_ERROR("{}: bad variable handle, dst=0x{:x}, src=0x{:x}", __func__, dstVarHandle, srcVarHandle);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimVarAssignVar(dstVarHandle, srcVarHandle);
    HCCL_VM_INFO("{}: exit, dst=0x{:x}, src=0x{:x}", __func__, dstVarHandle, srcVarHandle);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 变量相加。@return 成功返回 CCU_SUCCESS；句柄类型不符返回 CCU_E_PARA。 */
CcuResult CcuVariableAddVarToVar(CcuVariableHandle resVar, CcuVariableHandle varA, CcuVariableHandle varB)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(resVar) || !CheckVarHandle(varA) || !CheckVarHandle(varB)) {
        HCCL_VM_ERROR("{}: bad variable handle, res=0x{:x}, a=0x{:x}, b=0x{:x}", __func__, resVar, varA, varB);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimVarOpVar(resVar, varA, varB, HcclSim::CcuSim::SimCcuVarOp::ADD);
    HCCL_VM_INFO("{}: exit, res=0x{:x}, a=0x{:x}, b=0x{:x}, op=ADD", __func__, resVar, varA, varB);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 变量相减。@return 成功返回 CCU_SUCCESS；句柄类型不符返回 CCU_E_PARA。 */
CcuResult CcuVariableSubVarToVar(CcuVariableHandle resVar, CcuVariableHandle varA, CcuVariableHandle varB)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(resVar) || !CheckVarHandle(varA) || !CheckVarHandle(varB)) {
        HCCL_VM_ERROR("{}: bad variable handle, res=0x{:x}, a=0x{:x}, b=0x{:x}", __func__, resVar, varA, varB);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimVarOpVar(resVar, varA, varB, HcclSim::CcuSim::SimCcuVarOp::SUB);
    HCCL_VM_INFO("{}: exit, res=0x{:x}, a=0x{:x}, b=0x{:x}, op=SUB", __func__, resVar, varA, varB);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 变量相乘。@return 成功返回 CCU_SUCCESS；句柄类型不符返回 CCU_E_PARA。 */
CcuResult CcuVariableMulVarToVar(CcuVariableHandle resVar, CcuVariableHandle varA, CcuVariableHandle varB)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(resVar) || !CheckVarHandle(varA) || !CheckVarHandle(varB)) {
        HCCL_VM_ERROR("{}: bad variable handle, res=0x{:x}, a=0x{:x}, b=0x{:x}", __func__, resVar, varA, varB);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimVarOpVar(resVar, varA, varB, HcclSim::CcuSim::SimCcuVarOp::MUL);
    HCCL_VM_INFO("{}: exit, res=0x{:x}, a=0x{:x}, b=0x{:x}, op=MUL", __func__, resVar, varA, varB);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 变量加立即数（uint16 窄立即数，pkg_inc 契约）。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuVariableAddImmToVar(CcuVariableHandle resVar, CcuVariableHandle varA, uint16_t immediate)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(resVar) || !CheckVarHandle(varA)) {
        HCCL_VM_ERROR("{}: bad variable handle, res=0x{:x}, a=0x{:x}", __func__, resVar, varA);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimVarOpImm(resVar, varA, immediate, HcclSim::CcuSim::SimCcuVarOp::ADD);
    HCCL_VM_INFO("{}: exit, res=0x{:x}, a=0x{:x}, imm={}, op=ADD", __func__, resVar, varA, immediate);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 变量减立即数（uint16 窄立即数）。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuVariableSubImmToVar(CcuVariableHandle resVar, CcuVariableHandle varA, uint16_t immediate)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(resVar) || !CheckVarHandle(varA)) {
        HCCL_VM_ERROR("{}: bad variable handle, res=0x{:x}, a=0x{:x}", __func__, resVar, varA);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimVarOpImm(resVar, varA, immediate, HcclSim::CcuSim::SimCcuVarOp::SUB);
    HCCL_VM_INFO("{}: exit, res=0x{:x}, a=0x{:x}, imm={}, op=SUB", __func__, resVar, varA, immediate);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 变量乘立即数（uint16 窄立即数）。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuVariableMulImmToVar(CcuVariableHandle resVar, CcuVariableHandle varA, uint16_t immediate)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(resVar) || !CheckVarHandle(varA)) {
        HCCL_VM_ERROR("{}: bad variable handle, res=0x{:x}, a=0x{:x}", __func__, resVar, varA);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimVarOpImm(resVar, varA, immediate, HcclSim::CcuSim::SimCcuVarOp::MUL);
    HCCL_VM_INFO("{}: exit, res=0x{:x}, a=0x{:x}, imm={}, op=MUL", __func__, resVar, varA, immediate);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 变量按位与。@return 成功返回 CCU_SUCCESS；句柄类型不符返回 CCU_E_PARA。 */
CcuResult CcuVariableAndVarToVar(CcuVariableHandle resVar, CcuVariableHandle varA, CcuVariableHandle varB)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(resVar) || !CheckVarHandle(varA) || !CheckVarHandle(varB)) {
        HCCL_VM_ERROR("{}: bad variable handle, res=0x{:x}, a=0x{:x}, b=0x{:x}", __func__, resVar, varA, varB);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimVarOpVar(resVar, varA, varB, HcclSim::CcuSim::SimCcuVarOp::AND);
    HCCL_VM_INFO("{}: exit, res=0x{:x}, a=0x{:x}, b=0x{:x}, op=AND", __func__, resVar, varA, varB);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 变量按位或。@return 成功返回 CCU_SUCCESS；句柄类型不符返回 CCU_E_PARA。 */
CcuResult CcuVariableOrVarToVar(CcuVariableHandle resVar, CcuVariableHandle varA, CcuVariableHandle varB)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(resVar) || !CheckVarHandle(varA) || !CheckVarHandle(varB)) {
        HCCL_VM_ERROR("{}: bad variable handle, res=0x{:x}, a=0x{:x}, b=0x{:x}", __func__, resVar, varA, varB);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimVarOpVar(resVar, varA, varB, HcclSim::CcuSim::SimCcuVarOp::OR);
    HCCL_VM_INFO("{}: exit, res=0x{:x}, a=0x{:x}, b=0x{:x}, op=OR", __func__, resVar, varA, varB);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 变量按位异或。@return 成功返回 CCU_SUCCESS；句柄类型不符返回 CCU_E_PARA。 */
CcuResult CcuVariableXorVarToVar(CcuVariableHandle resVar, CcuVariableHandle varA, CcuVariableHandle varB)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(resVar) || !CheckVarHandle(varA) || !CheckVarHandle(varB)) {
        HCCL_VM_ERROR("{}: bad variable handle, res=0x{:x}, a=0x{:x}, b=0x{:x}", __func__, resVar, varA, varB);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimVarOpVar(resVar, varA, varB, HcclSim::CcuSim::SimCcuVarOp::XOR);
    HCCL_VM_INFO("{}: exit, res=0x{:x}, a=0x{:x}, b=0x{:x}, op=XOR", __func__, resVar, varA, varB);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 变量按位取反。@return 成功返回 CCU_SUCCESS；句柄类型不符返回 CCU_E_PARA。 */
CcuResult CcuVariableNotVar(CcuVariableHandle resVar, CcuVariableHandle varA)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(resVar) || !CheckVarHandle(varA)) {
        HCCL_VM_ERROR("{}: bad variable handle, res=0x{:x}, a=0x{:x}", __func__, resVar, varA);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimVarNot(resVar, varA);
    HCCL_VM_INFO("{}: exit, resVar=0x{:x}, varA=0x{:x}", __func__, resVar, varA);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 变量左移（移位量在变量中，≥64 回绕为 0）。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuVariableShlVarToVar(CcuVariableHandle resVar, CcuVariableHandle varA, CcuVariableHandle varB)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(resVar) || !CheckVarHandle(varA) || !CheckVarHandle(varB)) {
        HCCL_VM_ERROR("{}: bad variable handle, res=0x{:x}, a=0x{:x}, b=0x{:x}", __func__, resVar, varA, varB);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimVarOpVar(resVar, varA, varB, HcclSim::CcuSim::SimCcuVarOp::SHL);
    HCCL_VM_INFO("{}: exit, res=0x{:x}, a=0x{:x}, b=0x{:x}, op=SHL", __func__, resVar, varA, varB);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 变量右移（移位量在变量中，≥64 回绕为 0）。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuVariableShrVarToVar(CcuVariableHandle resVar, CcuVariableHandle varA, CcuVariableHandle varB)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(resVar) || !CheckVarHandle(varA) || !CheckVarHandle(varB)) {
        HCCL_VM_ERROR("{}: bad variable handle, res=0x{:x}, a=0x{:x}, b=0x{:x}", __func__, resVar, varA, varB);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimVarOpVar(resVar, varA, varB, HcclSim::CcuSim::SimCcuVarOp::SHR);
    HCCL_VM_INFO("{}: exit, res=0x{:x}, a=0x{:x}, b=0x{:x}, op=SHR", __func__, resVar, varA, varB);
    return CcuResult::CCU_SUCCESS;
}

/*========== Address 操作类相关接口 ==========*/

/** @brief 地址赋立即数。@return 成功返回 CCU_SUCCESS；句柄类型不符返回 CCU_E_PARA。 */
CcuResult CcuAddressAssignImm(CcuAddressHandle addr, uint64_t immediate)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckAddrHandle(addr)) {
        HCCL_VM_ERROR("{}: addr=0x{:x} is not an address handle", __func__, addr);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimAddrAssignImm(addr, immediate);
    HCCL_VM_INFO("{}: exit, addr=0x{:x}, immediate={} (0x{:x})", __func__, addr, immediate, immediate);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 地址赋值。@return 成功返回 CCU_SUCCESS；句柄类型不符返回 CCU_E_PARA。 */
CcuResult CcuAddressAssignAddr(CcuAddressHandle dstAddrHandle, CcuAddressHandle srcAddrHandle)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckAddrHandle(dstAddrHandle) || !CheckAddrHandle(srcAddrHandle)) {
        HCCL_VM_ERROR("{}: bad address handle, dst=0x{:x}, src=0x{:x}", __func__, dstAddrHandle, srcAddrHandle);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimAddrAssignAddr(dstAddrHandle, srcAddrHandle);
    HCCL_VM_INFO("{}: exit, dst=0x{:x}, src=0x{:x}", __func__, dstAddrHandle, srcAddrHandle);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 地址基值取自变量。@return 成功返回 CCU_SUCCESS；句柄类型不符返回 CCU_E_PARA。 */
CcuResult CcuAddressAssignVar(CcuAddressHandle addr, CcuVariableHandle var)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckAddrHandle(addr) || !CheckVarHandle(var)) {
        HCCL_VM_ERROR("{}: bad handle tag, addr=0x{:x}, var=0x{:x}", __func__, addr, var);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimAddrAssignVar(addr, var);
    HCCL_VM_INFO("{}: exit, addr=0x{:x}, var=0x{:x}", __func__, addr, var);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 地址加变量。@return 成功返回 CCU_SUCCESS；句柄类型不符返回 CCU_E_PARA。 */
CcuResult CcuAddressAddVarToAddr(CcuAddressHandle resAddr, CcuAddressHandle lhsAddr, CcuVariableHandle rhsVar)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckAddrHandle(resAddr) || !CheckAddrHandle(lhsAddr) || !CheckVarHandle(rhsVar)) {
        HCCL_VM_ERROR("{}: bad handle tag, res=0x{:x}, lhs=0x{:x}, var=0x{:x}", __func__, resAddr, lhsAddr, rhsVar);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimAddrOpVar(resAddr, lhsAddr, rhsVar, HcclSim::CcuSim::SimCcuVarOp::ADD);
    HCCL_VM_INFO("{}: exit, res=0x{:x}, lhs=0x{:x}, rhsVar=0x{:x}, op=ADD", __func__, resAddr, lhsAddr, rhsVar);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 地址加地址。@return 成功返回 CCU_SUCCESS；句柄类型不符返回 CCU_E_PARA。 */
CcuResult CcuAddressAddAddrToAddr(CcuAddressHandle resAddr, CcuAddressHandle addrA, CcuAddressHandle addrB)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckAddrHandle(resAddr) || !CheckAddrHandle(addrA) || !CheckAddrHandle(addrB)) {
        HCCL_VM_ERROR("{}: bad address handle, res=0x{:x}, a=0x{:x}, b=0x{:x}", __func__, resAddr, addrA, addrB);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimAddrOpAddr(resAddr, addrA, addrB, HcclSim::CcuSim::SimCcuVarOp::ADD);
    HCCL_VM_INFO("{}: exit, res=0x{:x}, a=0x{:x}, b=0x{:x}, op=ADD", __func__, resAddr, addrA, addrB);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 地址自增变量（循环体步进标准写法）。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuAddressAddAssignVar(CcuAddressHandle addr, CcuVariableHandle var)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckAddrHandle(addr) || !CheckVarHandle(var)) {
        HCCL_VM_ERROR("{}: bad handle tag, addr=0x{:x}, var=0x{:x}", __func__, addr, var);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimAddrAddAssignVar(addr, var);
    HCCL_VM_INFO("{}: exit, addr=0x{:x}, var=0x{:x} (addr += var)", __func__, addr, var);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 地址加立即数。@return 成功返回 CCU_SUCCESS；句柄类型不符返回 CCU_E_PARA。 */
CcuResult CcuAddressAddImmToAddr(CcuAddressHandle resAddr, CcuAddressHandle addrA, uint16_t imm)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckAddrHandle(resAddr) || !CheckAddrHandle(addrA)) {
        HCCL_VM_ERROR("{}: bad address handle, res=0x{:x}, a=0x{:x}", __func__, resAddr, addrA);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimAddrOpImm(resAddr, addrA, imm, HcclSim::CcuSim::SimCcuVarOp::ADD);
    HCCL_VM_INFO("{}: exit, res=0x{:x}, a=0x{:x}, imm={}, op=ADD", __func__, resAddr, addrA, imm);
    return CcuResult::CCU_SUCCESS;
}

/*========== 参数加载类相关接口 ==========*/

/** @brief 从 SQE 参数装载变量：回放期由 taskArgs[argId] 绑定。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLoadArg(CcuVariableHandle varHandle, uint32_t argId)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(varHandle)) {
        HCCL_VM_ERROR("{}: varHandle=0x{:x} is not a variable handle", __func__, varHandle);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimLoadArg(varHandle, argId);
    HCCL_VM_INFO("{}: exit, varHandle=0x{:x}, argId={}", __func__, varHandle, argId);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 从设备内存装载变量：best-effort，回放期变量保持当前值。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLoadVar(uint64_t addr, CcuVariableHandle varHandle, uint32_t num)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(varHandle)) {
        HCCL_VM_ERROR("{}: varHandle=0x{:x} is not a variable handle", __func__, varHandle);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimLoadMem(varHandle, addr, num, false);
    HCCL_VM_INFO("{}: exit, varHandle=0x{:x}, addr=0x{:x}, num={}", __func__, varHandle, addr, num);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 从变量地址装载变量：同 CcuLoadVar，best-effort。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLoadVarFromVarAddr(CcuVariableHandle addrHandle, CcuVariableHandle varHandle, uint32_t num)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(addrHandle) || !CheckVarHandle(varHandle)) {
        HCCL_VM_ERROR("{}: bad variable handle, addrVar=0x{:x}, var=0x{:x}", __func__, addrHandle, varHandle);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimLoadMem(varHandle, addrHandle, num, true);
    HCCL_VM_INFO("{}: exit, varHandle=0x{:x}, addrVar=0x{:x}, num={}", __func__, varHandle, addrHandle, num);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 变量存储到设备内存：CPU 模拟无副作用。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuStoreVar(uint64_t addr, CcuVariableHandle varHandle, uint32_t num)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(varHandle)) {
        HCCL_VM_ERROR("{}: varHandle=0x{:x} is not a variable handle", __func__, varHandle);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimStoreMem(addr, varHandle, num, false);
    HCCL_VM_INFO("{}: exit, addr=0x{:x}, varHandle=0x{:x}, num={}", __func__, addr, varHandle, num);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 变量存储到变量地址：CPU 模拟无副作用。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuStoreVarToVarAddr(CcuVariableHandle addrHandle, CcuVariableHandle varHandle, uint32_t num)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(addrHandle) || !CheckVarHandle(varHandle)) {
        HCCL_VM_ERROR("{}: bad variable handle, addrVar=0x{:x}, var=0x{:x}", __func__, addrHandle, varHandle);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimStoreMem(addrHandle, varHandle, num, true);
    HCCL_VM_INFO("{}: exit, addrVar=0x{:x}, varHandle=0x{:x}, num={}", __func__, addrHandle, varHandle, num);
    return CcuResult::CCU_SUCCESS;
}

/*========== Event 信号同步类相关接口 ==========*/

/** @brief 本地事件记录（CKE post）：回放期产 NOTIFY_RECORD。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuEventRecord(CcuEventHandle eventHandle, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckEventHandle(eventHandle)) {
        HCCL_VM_ERROR("{}: eventHandle=0x{:x} is not an event handle", __func__, eventHandle);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimEventRecord(eventHandle, mask);
    HCCL_VM_INFO("{}: exit, eventHandle=0x{:x}, mask=0x{:x}", __func__, eventHandle, mask);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 本地事件等待（CKE wait）：回放期产 NOTIFY_WAIT。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuEventWait(CcuEventHandle eventHandle, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckEventHandle(eventHandle)) {
        HCCL_VM_ERROR("{}: eventHandle=0x{:x} is not an event handle", __func__, eventHandle);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimEventWait(eventHandle, mask);
    HCCL_VM_INFO("{}: exit, eventHandle=0x{:x}, mask=0x{:x}", __func__, eventHandle, mask);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 远端通知记录：回放期经通道解析对端 rank/notifyId（channel 为 DB 主键不校验）。
 *  @return 成功返回 CCU_SUCCESS。 */
CcuResult CcuNotifyRecord(ChannelHandle channel, uint32_t remoteNotifyIdx, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    HcclSim::CcuSim::PrimNotifyRecord(channel, remoteNotifyIdx, mask);
    HCCL_VM_INFO("{}: exit, channel={}, remoteNotifyIdx={}, mask=0x{:x}", __func__, channel, remoteNotifyIdx, mask);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 远端通知等待。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuNotifyWait(ChannelHandle channel, uint32_t localNotifyIdx, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    HcclSim::CcuSim::PrimNotifyWait(channel, localNotifyIdx, mask);
    HCCL_VM_INFO("{}: exit, channel={}, localNotifyIdx={}, mask=0x{:x}", __func__, channel, localNotifyIdx, mask);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 写对端变量并通知（复合原语）：回放期产 NOTIFY_RECORD。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuWriteVariableWithNotify(
    ChannelHandle channel, CcuVariableHandle varHandle, uint32_t remoteVarIdx, uint32_t remoteNotifyIdx, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckVarHandle(varHandle)) {
        HCCL_VM_ERROR("{}: varHandle=0x{:x} is not a variable handle", __func__, varHandle);
        return CcuResult::CCU_E_PARA;
    }
    // remoteVarIdx = 对端 XN 槽号（交换区 0..3）：录制保留，回放期写方据此发布交换值。
    HcclSim::CcuSim::PrimWriteVarNotify(channel, varHandle, remoteVarIdx, remoteNotifyIdx, mask);
    HCCL_VM_INFO(
        "{}: exit, channel={}, varHandle=0x{:x}, remoteVarIdx={}, remoteNotifyIdx={}, mask=0x{:x}", __func__, channel,
        varHandle, remoteVarIdx, remoteNotifyIdx, mask);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 本地（同 device 跨 core）通知记录：notifyTag 入池。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLocalNotifyRecord(const char* notifyTag, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    HcclSim::CcuSim::PrimLocalNotifyRecord(notifyTag, mask);
    HCCL_VM_INFO("{}: exit, notifyTag={}, mask=0x{:x}", __func__, (notifyTag != nullptr) ? notifyTag : "", mask);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 本地（同 device 跨 core）通知等待。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLocalNotifyWait(const char* notifyTag, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    HcclSim::CcuSim::PrimLocalNotifyWait(notifyTag, mask);
    HCCL_VM_INFO("{}: exit, notifyTag={}, mask=0x{:x}", __func__, (notifyTag != nullptr) ? notifyTag : "", mask);
    return CcuResult::CCU_SUCCESS;
}

/*========== 本地数据拷贝相关接口 ==========*/

/** @brief 本地内存到内存拷贝：回放期产 MEM_CPY。@return 成功返回 CCU_SUCCESS；句柄类型不符返回 CCU_E_PARA。 */
CcuResult CcuLocalCopyMemToMem(
    CcuLocalAddrHandle dst, CcuLocalAddrHandle src, CcuVariableHandle len, CcuEventHandle event, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckLocalAddrHandle(dst) || !CheckLocalAddrHandle(src) || !CheckVarHandle(len) || !CheckEventHandle(event)) {
        HCCL_VM_ERROR(
            "{}: bad handle tag, dst=0x{:x}, src=0x{:x}, len=0x{:x}, event=0x{:x}", __func__, dst, src, len, event);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimLocalCopyMm(dst, src, len, event, mask);
    HCCL_VM_INFO(
        "{}: exit, dst=0x{:x}, src=0x{:x}, lenVar=0x{:x}, event=0x{:x}, mask=0x{:x}", __func__, dst, src, len, event,
        mask);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 本地内存到缓冲拷贝：缓冲侧用伪地址建模。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLocalCopyMemToBuffer(
    CcuBufferHandle dst, CcuLocalAddrHandle src, CcuVariableHandle len, CcuEventHandle event, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckBufHandle(dst) || !CheckLocalAddrHandle(src) || !CheckVarHandle(len) || !CheckEventHandle(event)) {
        HCCL_VM_ERROR(
            "{}: bad handle tag, dst=0x{:x}, src=0x{:x}, len=0x{:x}, event=0x{:x}", __func__, dst, src, len, event);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimLocalCopyMb(dst, src, len, event, mask);
    HCCL_VM_INFO(
        "{}: exit, dstBuf=0x{:x}, src=0x{:x}, lenVar=0x{:x}, event=0x{:x}, mask=0x{:x}", __func__, dst, src, len, event,
        mask);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 本地缓冲到内存拷贝。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLocalCopyBufferToMem(
    CcuLocalAddrHandle dst, CcuBufferHandle src, CcuVariableHandle len, CcuEventHandle event, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckLocalAddrHandle(dst) || !CheckBufHandle(src) || !CheckVarHandle(len) || !CheckEventHandle(event)) {
        HCCL_VM_ERROR(
            "{}: bad handle tag, dst=0x{:x}, src=0x{:x}, len=0x{:x}, event=0x{:x}", __func__, dst, src, len, event);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimLocalCopyBm(dst, src, len, event, mask);
    HCCL_VM_INFO(
        "{}: exit, dst=0x{:x}, srcBuf=0x{:x}, lenVar=0x{:x}, event=0x{:x}, mask=0x{:x}", __func__, dst, src, len, event,
        mask);
    return CcuResult::CCU_SUCCESS;
}

/*========== 本地 reduce 相关接口 ==========*/

/** @brief 本地内存归约：回放期产 REDUCE。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLocalMemReduce(
    CcuLocalAddrHandle dst, CcuLocalAddrHandle src, CcuVariableHandle len, HcclDataType dataType, HcclReduceOp opType,
    CcuEventHandle event, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckLocalAddrHandle(dst) || !CheckLocalAddrHandle(src) || !CheckVarHandle(len) || !CheckEventHandle(event)) {
        HCCL_VM_ERROR(
            "{}: bad handle tag, dst=0x{:x}, src=0x{:x}, len=0x{:x}, event=0x{:x}", __func__, dst, src, len, event);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimLocalReduceMm(
        dst, src, len, event, mask, static_cast<uint8_t>(dataType), static_cast<uint8_t>(opType));
    HCCL_VM_INFO(
        "{}: exit, dst=0x{:x}, src=0x{:x}, lenVar=0x{:x}, dataType={}, op={}", __func__, dst, src, len,
        static_cast<uint32_t>(dataType), static_cast<uint32_t>(opType));
    return CcuResult::CCU_SUCCESS;
}

/** @brief 本地多缓冲归约：句柄数组入轨迹附加存储。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLocalBufferReduce(
    CcuBufferHandle* buffers, uint32_t count, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType,
    CcuVariableHandle len, CcuEventHandle event, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    (void)outputDataType; // 与既有口径一致：任务模型仅记录入参 dataType
    if (!CheckVarHandle(len) || !CheckEventHandle(event)) {
        HCCL_VM_ERROR("{}: bad handle tag, len=0x{:x}, event=0x{:x}", __func__, len, event);
        return CcuResult::CCU_E_PARA;
    }
    if (buffers != nullptr) {
        for (uint32_t i = 0; i < count; i++) {
            if (!CheckBufHandle(buffers[i])) {
                HCCL_VM_ERROR("{}: buffers[{}]=0x{:x} is not a buffer handle", __func__, i, buffers[i]);
                return CcuResult::CCU_E_PARA;
            }
        }
    }
    HcclSim::CcuSim::PrimLocalReduceBufs(
        buffers, count, len, event, mask, static_cast<uint8_t>(dataType), static_cast<uint8_t>(opType));
    HCCL_VM_INFO(
        "{}: exit, bufCount={}, lenVar=0x{:x}, dataType={}, op={}", __func__, count, len,
        static_cast<uint32_t>(dataType), static_cast<uint32_t>(opType));
    return CcuResult::CCU_SUCCESS;
}

/*========== 远端数据传输操作 ==========*/

/** @brief 远端内存读到本地内存：回放期产跨设备 MEM_CPY。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuReadMemToMem(
    ChannelHandle channel, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle, CcuVariableHandle len,
    CcuEventHandle event, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckLocalAddrHandle(localHandle) || !CheckRemoteAddrHandle(remoteHandle) || !CheckVarHandle(len)
        || !CheckEventHandle(event)) {
        HCCL_VM_ERROR(
            "{}: bad handle tag, local=0x{:x}, remote=0x{:x}, len=0x{:x}, event=0x{:x}", __func__, localHandle,
            remoteHandle, len, event);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimRemoteReadMm(channel, localHandle, remoteHandle, len, event, mask);
    HCCL_VM_INFO(
        "{}: exit, channel={}, local=0x{:x}, remote=0x{:x}, lenVar=0x{:x}, event=0x{:x}, mask=0x{:x}", __func__,
        channel, localHandle, remoteHandle, len, event, mask);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 远端内存读到本地缓冲：缓冲侧用伪地址建模。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuReadMemToBuffer(
    ChannelHandle channel, CcuBufferHandle localHandle, CcuRemoteAddrHandle remoteHandle, CcuVariableHandle len,
    CcuEventHandle event, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckBufHandle(localHandle) || !CheckRemoteAddrHandle(remoteHandle) || !CheckVarHandle(len)
        || !CheckEventHandle(event)) {
        HCCL_VM_ERROR(
            "{}: bad handle tag, buf=0x{:x}, remote=0x{:x}, len=0x{:x}, event=0x{:x}", __func__, localHandle,
            remoteHandle, len, event);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimRemoteReadMb(channel, localHandle, remoteHandle, len, event, mask);
    HCCL_VM_INFO(
        "{}: exit, channel={}, buf=0x{:x}, remote=0x{:x}, lenVar=0x{:x}, event=0x{:x}, mask=0x{:x}", __func__, channel,
        localHandle, remoteHandle, len, event, mask);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 远端内存读到本地内存并归约：回放期产跨设备 REDUCE。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuReadMemToMemReduce(
    ChannelHandle channel, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle, CcuVariableHandle len,
    HcclDataType dataType, HcclReduceOp opType, CcuEventHandle event, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckLocalAddrHandle(localHandle) || !CheckRemoteAddrHandle(remoteHandle) || !CheckVarHandle(len)
        || !CheckEventHandle(event)) {
        HCCL_VM_ERROR(
            "{}: bad handle tag, local=0x{:x}, remote=0x{:x}, len=0x{:x}, event=0x{:x}", __func__, localHandle,
            remoteHandle, len, event);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimRemoteReadMr(
        channel, localHandle, remoteHandle, len, event, mask, static_cast<uint8_t>(dataType),
        static_cast<uint8_t>(opType));
    HCCL_VM_INFO(
        "{}: exit, channel={}, local=0x{:x}, remote=0x{:x}, lenVar=0x{:x}, dataType={}, op={}", __func__, channel,
        localHandle, remoteHandle, len, static_cast<uint32_t>(dataType), static_cast<uint32_t>(opType));
    return CcuResult::CCU_SUCCESS;
}

/** @brief 本地内存写到远端内存。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuWriteMemToMem(
    ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle, CcuVariableHandle len,
    CcuEventHandle event, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckRemoteAddrHandle(remoteHandle) || !CheckLocalAddrHandle(localHandle) || !CheckVarHandle(len)
        || !CheckEventHandle(event)) {
        HCCL_VM_ERROR(
            "{}: bad handle tag, remote=0x{:x}, local=0x{:x}, len=0x{:x}, event=0x{:x}", __func__, remoteHandle,
            localHandle, len, event);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimRemoteWriteMm(channel, remoteHandle, localHandle, len, event, mask);
    HCCL_VM_INFO(
        "{}: exit, channel={}, remote=0x{:x}, local=0x{:x}, lenVar=0x{:x}, event=0x{:x}, mask=0x{:x}", __func__,
        channel, remoteHandle, localHandle, len, event, mask);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 本地缓冲写到远端内存。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuWriteBufferToMem(
    ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuBufferHandle localHandle, CcuVariableHandle len,
    CcuEventHandle event, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckRemoteAddrHandle(remoteHandle) || !CheckBufHandle(localHandle) || !CheckVarHandle(len)
        || !CheckEventHandle(event)) {
        HCCL_VM_ERROR(
            "{}: bad handle tag, remote=0x{:x}, buf=0x{:x}, len=0x{:x}, event=0x{:x}", __func__, remoteHandle,
            localHandle, len, event);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimRemoteWriteBm(channel, remoteHandle, localHandle, len, event, mask);
    HCCL_VM_INFO(
        "{}: exit, channel={}, remote=0x{:x}, buf=0x{:x}, lenVar=0x{:x}, event=0x{:x}, mask=0x{:x}", __func__, channel,
        remoteHandle, localHandle, len, event, mask);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 本地内存写到远端内存并归约。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuWriteMemToMemReduce(
    ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle, CcuVariableHandle len,
    HcclDataType dataType, HcclReduceOp opType, CcuEventHandle event, uint16_t mask)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (!CheckRemoteAddrHandle(remoteHandle) || !CheckLocalAddrHandle(localHandle) || !CheckVarHandle(len)
        || !CheckEventHandle(event)) {
        HCCL_VM_ERROR(
            "{}: bad handle tag, remote=0x{:x}, local=0x{:x}, len=0x{:x}, event=0x{:x}", __func__, remoteHandle,
            localHandle, len, event);
        return CcuResult::CCU_E_PARA;
    }
    HcclSim::CcuSim::PrimRemoteWriteMr(
        channel, remoteHandle, localHandle, len, event, mask, static_cast<uint8_t>(dataType),
        static_cast<uint8_t>(opType));
    HCCL_VM_INFO(
        "{}: exit, channel={}, remote=0x{:x}, local=0x{:x}, lenVar=0x{:x}, dataType={}, op={}", __func__, channel,
        remoteHandle, localHandle, len, static_cast<uint32_t>(dataType), static_cast<uint32_t>(opType));
    return CcuResult::CCU_SUCCESS;
}

/*========== 控制流操作 ==========*/

/*
 * 录制期语义：控制流桩恒返回成功（不做条件判定），保证 then/else 两份分支
 * 与循环体都被 C++ 执行并录进轨迹；真实条件判定、分支裁剪与循环展开由
 * Launch 期回放器完成（ccu_kernel_sim.cc ExecuteKernelOnLaunch）。
 */

/** @brief if 块起始（变量 vs 立即数）：录制条件结构。@return 恒返回 CCU_SUCCESS。 */
CcuResult CcuIfBegin(CcuVariableHandle var, uint64_t immediate, CcuConditionType condType, const char* label)
{
    HCCL_VM_INFO("{}: enter", __func__);
    HcclSim::CcuSim::PrimIfBegin(var, immediate, false, condType, label);
    HCCL_VM_INFO(
        "{}: exit, var=0x{:x}, imm={} (0x{:x}), cond={}, label={}", __func__, var, immediate, immediate,
        static_cast<int32_t>(condType), (label != nullptr) ? label : "");
    return CcuResult::CCU_SUCCESS;
}

/** @brief if 块 else 分支：录制结构翻转点。@return 恒返回 CCU_SUCCESS。 */
CcuResult CcuIfElse(const char* label)
{
    HCCL_VM_INFO("{}: enter", __func__);
    HcclSim::CcuSim::PrimIfElse(label);
    HCCL_VM_INFO("{}: exit, label={}", __func__, (label != nullptr) ? label : "");
    return CcuResult::CCU_SUCCESS;
}

/** @brief if 块结束：录制结构收尾。@return 恒返回 CCU_SUCCESS。 */
CcuResult CcuIfEnd(const char* label)
{
    HCCL_VM_INFO("{}: enter", __func__);
    HcclSim::CcuSim::PrimIfEnd(label);
    HCCL_VM_INFO("{}: exit, label={}", __func__, (label != nullptr) ? label : "");
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 收敛挂起的 pending if：闭合栈顶连续 bodyDone 的无-else if（补录 IF_END 轨迹）。
 *
 * 宏时序（CCU_IF 退出增量）保证本调用发生在"if 体刚结束、下一个原语之前"，
 * 补录的闭合位置即回放器中条件假时的跳转落点，与真身 FlushClosablePendingIfs
 * 在此处放置 elseLabel 的语义对齐。
 * @return 恒返回 CCU_SUCCESS。
 */
CcuResult CcuFlushPendingIfs()
{
    HCCL_VM_INFO("{}: enter", __func__);
    HcclSim::CcuSim::FlushPendingIfs();
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief while 块起始（变量 vs 立即数）：事务校验 + 录制循环条件结构。
 *
 * 对齐真身 WhileBegin（ccu_kernel.cc:1599）：同 label 未闭合重开返回 CCU_E_PARA
 * （宏 ⑤⑥ 增量不执行，循环体跳过）；成功后录 WHILE_BEGIN，回放期按真实条件
 * 循环（假则 0 轮、真则多轮直到条件翻转）。
 * @return 成功返回 CCU_SUCCESS；label 重复返回 CCU_E_PARA；非录制期返回 CCU_E_PTR。
 */
CcuResult CcuWhileBegin(CcuVariableHandle var, uint64_t immediate, CcuConditionType condType, const char* label)
{
    HCCL_VM_INFO("{}: enter", __func__);
    const CcuResult ret = HcclSim::CcuSim::WhileBeginCheck(label);
    if (ret != CcuResult::CCU_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: WhileBeginCheck failed, ret={}, label={}", __func__, static_cast<int32_t>(ret),
            (label != nullptr) ? label : "");
        return ret;
    }
    HcclSim::CcuSim::PrimWhileBegin(var, immediate, false, condType, label);
    HCCL_VM_INFO(
        "{}: exit, var=0x{:x}, imm={} (0x{:x}), cond={}, label={}", __func__, var, immediate, immediate,
        static_cast<int32_t>(condType), (label != nullptr) ? label : "");
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief while 块结束：事务校验 + 录制收尾（配对 WHILE_END）。
 * @return 成功返回 CCU_SUCCESS；无匹配 Begin 返回 CCU_E_NOT_FOUND；非录制期返回 CCU_E_PTR。
 */
CcuResult CcuWhileEnd(const char* label)
{
    HCCL_VM_INFO("{}: enter", __func__);
    const CcuResult ret = HcclSim::CcuSim::WhileEndCheck(label);
    if (ret != CcuResult::CCU_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: WhileEndCheck failed, ret={}, label={}", __func__, static_cast<int32_t>(ret),
            (label != nullptr) ? label : "");
        return ret;
    }
    HcclSim::CcuSim::PrimWhileEnd(label);
    HCCL_VM_INFO("{}: exit, label={}", __func__, (label != nullptr) ? label : "");
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief do-while 块起始：事务校验 + 录制标签（体至少执行一次）。
 * @return 成功返回 CCU_SUCCESS；label 重复返回 CCU_E_PARA；非录制期返回 CCU_E_PTR。
 */
CcuResult CcuDoWhileBegin(const char* label)
{
    HCCL_VM_INFO("{}: enter", __func__);
    const CcuResult ret = HcclSim::CcuSim::DoWhileBeginCheck(label);
    if (ret != CcuResult::CCU_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: DoWhileBeginCheck failed, ret={}, label={}", __func__, static_cast<int32_t>(ret),
            (label != nullptr) ? label : "");
        return ret;
    }
    HcclSim::CcuSim::PrimDoBegin(label);
    HCCL_VM_INFO("{}: exit, label={}", __func__, (label != nullptr) ? label : "");
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief do-while 块收尾（变量 vs 立即数）：事务校验 + 录制回跳条件。
 *
 * 对齐真身 DoWhileEnd（ccu_kernel.cc:1746）：条件为真回跳 begin（正向条件，
 * 与 while 的反转相反）；六条件全支持，不支持返回 CCU_E_PARA。
 * @return 成功返回 CCU_SUCCESS；无匹配 Begin 返回 CCU_E_NOT_FOUND；非录制期返回 CCU_E_PTR。
 */
CcuResult CcuDoWhileEnd(CcuVariableHandle var, uint64_t immediate, CcuConditionType condType, const char* label)
{
    HCCL_VM_INFO("{}: enter", __func__);
    const CcuResult ret = HcclSim::CcuSim::DoWhileEndCheck(label);
    if (ret != CcuResult::CCU_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: DoWhileEndCheck failed, ret={}, label={}", __func__, static_cast<int32_t>(ret),
            (label != nullptr) ? label : "");
        return ret;
    }
    HcclSim::CcuSim::PrimDoEnd(var, immediate, false, condType, label);
    HCCL_VM_INFO(
        "{}: exit, var=0x{:x}, imm={} (0x{:x}), cond={}, label={}", __func__, var, immediate, immediate,
        static_cast<int32_t>(condType), (label != nullptr) ? label : "");
    return CcuResult::CCU_SUCCESS;
}

/** @brief if 块起始（变量 vs 变量）。@return 恒返回 CCU_SUCCESS。 */
CcuResult CcuIfBeginVar(CcuVariableHandle lhs, CcuVariableHandle rhs, CcuConditionType condType, const char* label)
{
    HCCL_VM_INFO("{}: enter", __func__);
    HcclSim::CcuSim::PrimIfBegin(lhs, rhs, true, condType, label);
    HCCL_VM_INFO(
        "{}: exit, lhs=0x{:x}, rhsVar=0x{:x}, cond={}, label={}", __func__, lhs, rhs, static_cast<int32_t>(condType),
        (label != nullptr) ? label : "");
    return CcuResult::CCU_SUCCESS;
}

/** @brief while 块起始（变量 vs 变量）。@return 恒返回 CCU_SUCCESS。 */
CcuResult CcuWhileBeginVar(CcuVariableHandle lhs, CcuVariableHandle rhs, CcuConditionType condType, const char* label)
{
    HCCL_VM_INFO("{}: enter", __func__);
    const CcuResult ret = HcclSim::CcuSim::WhileBeginCheck(label);
    if (ret != CcuResult::CCU_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: WhileBeginCheck failed, ret={}, label={}", __func__, static_cast<int32_t>(ret),
            (label != nullptr) ? label : "");
        return ret;
    }
    HcclSim::CcuSim::PrimWhileBegin(lhs, rhs, true, condType, label);
    HCCL_VM_INFO(
        "{}: exit, lhs=0x{:x}, rhsVar=0x{:x}, cond={}, label={}", __func__, lhs, rhs, static_cast<int32_t>(condType),
        (label != nullptr) ? label : "");
    return CcuResult::CCU_SUCCESS;
}

/** @brief do-while 块收尾（变量 vs 变量）：事务校验同 DoWhileEnd。@return 语义同 DoWhileEnd。 */
CcuResult CcuDoWhileEndVar(CcuVariableHandle lhs, CcuVariableHandle rhs, CcuConditionType condType, const char* label)
{
    HCCL_VM_INFO("{}: enter", __func__);
    const CcuResult ret = HcclSim::CcuSim::DoWhileEndCheck(label);
    if (ret != CcuResult::CCU_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: DoWhileEndCheck failed, ret={}, label={}", __func__, static_cast<int32_t>(ret),
            (label != nullptr) ? label : "");
        return ret;
    }
    HcclSim::CcuSim::PrimDoEnd(lhs, rhs, true, condType, label);
    HCCL_VM_INFO(
        "{}: exit, lhs=0x{:x}, rhsVar=0x{:x}, cond={}, label={}", __func__, lhs, rhs, static_cast<int32_t>(condType),
        (label != nullptr) ? label : "");
    return CcuResult::CCU_SUCCESS;
}

/*========== 函数调用操作 ==========*/

/** @brief 查找函数块：CPU 模拟按线性轨迹处理（无独立函数块结构）。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuFuncBlockLookup(const void* funcPtr, uint64_t* outHandle)
{
    HCCL_VM_INFO("{}: enter", __func__);
    (void)funcPtr;
    if (outHandle != nullptr) {
        *outHandle = 0;
    }
    return CcuResult::CCU_SUCCESS;
}

/** @brief 函数块起始：体内原语线性进入轨迹。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuFuncBlockBegin(const void* funcPtr, uint64_t* outHandle)
{
    HCCL_VM_INFO("{}: enter", __func__);
    (void)funcPtr;
    if (outHandle != nullptr) {
        *outHandle = 0;
    }
    return CcuResult::CCU_SUCCESS;
}

/** @brief 函数块结束。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuFuncBlockEnd(uint64_t handle)
{
    HCCL_VM_INFO("{}: enter", __func__);
    (void)handle;
    return CcuResult::CCU_SUCCESS;
}

/** @brief 函数块形参绑定。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuFuncDefineInArg(uint64_t handle, CcuVariableHandle formal)
{
    HCCL_VM_INFO("{}: enter", __func__);
    (void)handle;
    (void)formal;
    return CcuResult::CCU_SUCCESS;
}

/** @brief 函数调用。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuFuncCall(uint64_t handle, const CcuVariableHandle* inArgs, uint32_t numIn)
{
    HCCL_VM_INFO("{}: enter", __func__);
    (void)handle;
    (void)inArgs;
    (void)numIn;
    return CcuResult::CCU_SUCCESS;
}

/*
 * 控制流宏内部使用的标签栈接口（void 返回）：录制期与 CcuSim 标签栈联动，
 * 使 CCU_ELSE 能配对到最近的已完成 if、CCU_WHILE 能判别 do-while 尾部。
 */
void _CcuIfStackPush(const char* label)
{
    HCCL_VM_INFO("{}: enter", __func__);
    HcclSim::CcuSim::IfStackPush(label);
}

/** @brief C-ABI 包装：IfStackMarkBodyDone（供 CCU_IF 宏增量调用）。 */
void _CcuIfStackMarkBodyDone()
{
    HCCL_VM_INFO("{}: enter", __func__);
    HcclSim::CcuSim::IfStackMarkBodyDone();
}

/** @brief C-ABI 包装：IfStackPopForElse（供 CCU_ELSE 宏调用）。 */
const char* _CcuIfStackPopForElse()
{
    HCCL_VM_INFO("{}: enter", __func__);
    // 返回最近未闭合 if 的标签，使 CCU_ELSE 宏进入 else 分支录制 else 体。
    return HcclSim::CcuSim::IfStackPopForElse();
}

/** @brief C-ABI 包装：DoWhileStackPush（供 CCU_DO 宏增量调用）。 */
void _CcuDoWhileStackPush(const char* label)
{
    HCCL_VM_INFO("{}: enter", __func__);
    HcclSim::CcuSim::DoWhileStackPush(label);
}

/** @brief C-ABI 包装：DoWhileStackPopForWhile（供 CCU_WHILE 宏增量调用）。 */
const char* _CcuDoWhileStackPopForWhile()
{
    HCCL_VM_INFO("{}: enter", __func__);
    // 返回 do-while 标签使 CCU_WHILE 宏走 DoWhileEnd 分支；普通 while 返回 nullptr。
    return HcclSim::CcuSim::DoWhileStackPopForWhile();
}

/*========== 循环操作 ==========*/

/*
 * Loop/LoopGroup（大块抽象，参考 ccu-sim-stub-design.md）：
 * - 录制期：体段边界（LOOP_BODY_ENTER/EXIT）标记、组占位符（LOOP_GROUP_BEGIN/END）、
 *   成员参数归一化（CONFIG 立即数 / VAR_V1 位域 / VAR_V2 独立变量）；
 * - 回放期：大块抽象展开器消费组描述符——每模板 1 条大块 MEM_CPY task
 *   （total = (cloneNum-1)*cloneStride + (iterNum-1)*iterStride + perExecLen），
 *   模板间不合并（保真），体内不逐迭代展开（模拟无 ms 4KB 硬件限制）。
 */

/** @brief 创建硬件 Loop（分配 loopId 供体边界标记）。@return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLoopCreate(CcuLoop* loop)
{
    HCCL_VM_INFO("{}: enter", __func__);
    if (loop != nullptr) {
        *loop = HcclSim::CcuSim::AllocHandle(SimCcuHandleTag::LOOP);
        HCCL_VM_INFO("{}: exit, loop=0x{:x}", __func__, *loop);
    }
    return CcuResult::CCU_SUCCESS;
}

/** @brief 进入 Loop 体（控制流宏内部使用）：录 LOOP_BODY_ENTER 边界。@return 成功返回 CCU_SUCCESS。 */
CcuResult _CcuLoopBodyEnter(CcuLoop loop)
{
    HCCL_VM_INFO("{}: enter, loop=0x{:x}", __func__, loop);
    HcclSim::CcuSim::PrimLoopBodyEnter(static_cast<uint32_t>(loop & 0xFFFFFFFFULL));
    return CcuResult::CCU_SUCCESS;
}

/** @brief 退出 Loop 体（控制流宏内部使用）：录 LOOP_BODY_EXIT 边界。@return 成功返回 CCU_SUCCESS。 */
CcuResult _CcuLoopBodyExit(CcuLoop loop)
{
    HCCL_VM_INFO("{}: enter, loop=0x{:x}", __func__, loop);
    HcclSim::CcuSim::PrimLoopBodyExit(static_cast<uint32_t>(loop & 0xFFFFFFFFULL));
    return CcuResult::CCU_SUCCESS;
}

/** @brief 创建 LoopGroup（旧 config 入口）：录 LOOP_GROUP_BEGIN + CONFIG 参数。
 *  @return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLoopGroupCreate(CcuLoopGroup* group, uint32_t maxLoopNum, const CcuLoopGroupConfig* config)
{
    HCCL_VM_INFO("{}: enter", __func__);
    uint32_t groupIdx = UINT32_MAX;
    if (config != nullptr) {
        HcclSim::CcuSim::PrimLoopGroupBegin(
            HcclSim::CcuSim::SimCcuLoopParamSrc::CONFIG, config->cloneNum, config->cloneLoopOffset, config->addrOffset,
            config->ccuBufferOffset, config->eventOffset, 0, 0, 0, 0, groupIdx);
    } else {
        HcclSim::CcuSim::PrimLoopGroupBegin(
            HcclSim::CcuSim::SimCcuLoopParamSrc::CONFIG, 1, 0, 0, 0, 0, 0, 0, 0, 0, groupIdx);
    }
    if (group != nullptr) {
        *group = (groupIdx != UINT32_MAX) ? groupIdx + 1U : 0; // CcuLoopGroup 句柄 = groupIdx+1（非零）
        HCCL_VM_INFO("{}: exit, group=0x{:x} (groupIdx={})", __func__, *group, groupIdx);
    }
    return CcuResult::CCU_SUCCESS;
}

/** @brief 由变量创建 LoopGroup（VAR_V1：parallel/offset 位域打包变量）。
 *  @return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLoopGroupCreateFromVar(
    CcuLoopGroup* group, uint32_t maxLoopNum, CcuVariableHandle parallelVar, CcuVariableHandle offsetVar)
{
    HCCL_VM_INFO("{}: enter, parallelVar=0x{:x}, offsetVar=0x{:x}", __func__, parallelVar, offsetVar);
    uint32_t groupIdx = UINT32_MAX;
    HcclSim::CcuSim::PrimLoopGroupBegin(
        HcclSim::CcuSim::SimCcuLoopParamSrc::VAR_V1, 0, 0, 0, 0, 0, 0, parallelVar, offsetVar, 0, groupIdx);
    if (group != nullptr) {
        *group = (groupIdx != UINT32_MAX) ? groupIdx + 1U : 0;
        HCCL_VM_INFO("{}: exit, group=0x{:x} (groupIdx={})", __func__, *group, groupIdx);
    }
    return CcuResult::CCU_SUCCESS;
}

/** @brief 由变量创建 LoopGroup(V2)（iterNum/addrOffset/varOffset 独立变量）。
 *  @return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLoopGroupCreateFromVarV2(
    CcuLoopGroup* group, uint32_t maxLoopNum, CcuVariableHandle parallelVarV2, CcuVariableHandle offsetVarV2,
    CcuVariableHandle varOffsetVar)
{
    HCCL_VM_INFO(
        "{}: enter, parallelVarV2=0x{:x}, offsetVarV2=0x{:x}, varOffsetVar=0x{:x}", __func__, parallelVarV2,
        offsetVarV2, varOffsetVar);
    uint32_t groupIdx = UINT32_MAX;
    HcclSim::CcuSim::PrimLoopGroupBegin(
        HcclSim::CcuSim::SimCcuLoopParamSrc::VAR_V2, 0, 0, 0, 0, 0, 0, parallelVarV2, offsetVarV2, varOffsetVar,
        groupIdx);
    if (group != nullptr) {
        *group = (groupIdx != UINT32_MAX) ? groupIdx + 1U : 0;
        HCCL_VM_INFO("{}: exit, group=0x{:x} (groupIdx={})", __func__, *group, groupIdx);
    }
    return CcuResult::CCU_SUCCESS;
}

/** @brief 向 LoopGroup 追加 Loop（旧 config 入口：iterNum+addrOffset 立即数）。
 *  @return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLoopGroupAddLoop(CcuLoopGroup group, CcuLoop loop, const CcuLoopConfig* config)
{
    HCCL_VM_INFO("{}: enter, group=0x{:x}, loop=0x{:x}", __func__, group, loop);
    const uint32_t groupIdx = static_cast<uint32_t>((group > 0U) ? group - 1U : 0);
    HcclSim::CcuSim::PrimLoopGroupAddLoop(
        groupIdx, static_cast<uint32_t>(loop & 0xFFFFFFFFULL), HcclSim::CcuSim::SimCcuLoopParamSrc::CONFIG,
        (config != nullptr) ? config->iterNum : 1, (config != nullptr) ? config->addrOffset : 0, 0, 0, 0, 0);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 以版本化 cfg 创建 LoopGroup（等价 CcuLoopGroupCreate）。
 *  @return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLoopGroupCreateCfg(CcuLoopGroup* group, uint32_t maxLoopNum, const CcuLoopGroupCfg* cfg)
{
    HCCL_VM_INFO("{}: enter", __func__);
    uint32_t groupIdx = UINT32_MAX;
    if (cfg != nullptr) {
        HcclSim::CcuSim::PrimLoopGroupBegin(
            HcclSim::CcuSim::SimCcuLoopParamSrc::CONFIG, cfg->cloneNum, cfg->cloneLoopOffset, cfg->addrOffset,
            cfg->ccuBufferOffset, cfg->eventOffset, cfg->varOffset, 0, 0, 0, groupIdx);
    } else {
        HcclSim::CcuSim::PrimLoopGroupBegin(
            HcclSim::CcuSim::SimCcuLoopParamSrc::CONFIG, 1, 0, 0, 0, 0, 0, 0, 0, 0, groupIdx);
    }
    if (group != nullptr) {
        *group = (groupIdx != UINT32_MAX) ? groupIdx + 1U : 0;
        HCCL_VM_INFO("{}: exit, group=0x{:x} (groupIdx={})", __func__, *group, groupIdx);
    }
    return CcuResult::CCU_SUCCESS;
}

/** @brief 以版本化 cfg 向 LoopGroup 追加 Loop（等价 CcuLoopGroupAddLoop）。
 *  @return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLoopGroupAddLoopCfg(CcuLoopGroup group, CcuLoop loop, const CcuLoopCfg* cfg)
{
    HCCL_VM_INFO("{}: enter, group=0x{:x}, loop=0x{:x}", __func__, group, loop);
    const uint32_t groupIdx = static_cast<uint32_t>((group > 0U) ? group - 1U : 0);
    HcclSim::CcuSim::PrimLoopGroupAddLoop(
        groupIdx, static_cast<uint32_t>(loop & 0xFFFFFFFFULL), HcclSim::CcuSim::SimCcuLoopParamSrc::CONFIG,
        (cfg != nullptr) ? cfg->iterNum : 1, (cfg != nullptr) ? cfg->addrOffset : 0, 0, 0, 0, 0);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 由变量向 LoopGroup 追加 Loop（VAR_V1：loopParam 位域打包变量）。
 *  @return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLoopGroupAddLoopFromVar(CcuLoopGroup group, CcuLoop loop, CcuVariableHandle loopParamVar)
{
    HCCL_VM_INFO("{}: enter, group=0x{:x}, loop=0x{:x}, loopParamVar=0x{:x}", __func__, group, loop, loopParamVar);
    const uint32_t groupIdx = static_cast<uint32_t>((group > 0U) ? group - 1U : 0);
    HcclSim::CcuSim::PrimLoopGroupAddLoop(
        groupIdx, static_cast<uint32_t>(loop & 0xFFFFFFFFULL), HcclSim::CcuSim::SimCcuLoopParamSrc::VAR_V1, 0, 0,
        loopParamVar, 0, 0, 0);
    return CcuResult::CCU_SUCCESS;
}

/** @brief 由变量向 LoopGroup 追加 Loop(V2)（iterNum/addrOffset/ctxId 独立变量）。
 *  @return 成功返回 CCU_SUCCESS。 */
CcuResult CcuLoopGroupAddLoopFromVarV2(
    CcuLoopGroup group, CcuLoop loop, CcuVariableHandle iterNumVar, CcuVariableHandle addrOffsetVar,
    CcuVariableHandle ctxIdVar)
{
    HCCL_VM_INFO(
        "{}: enter, group=0x{:x}, loop=0x{:x}, iterNumVar=0x{:x}, addrOffsetVar=0x{:x}", __func__, group, loop,
        iterNumVar, addrOffsetVar);
    const uint32_t groupIdx = static_cast<uint32_t>((group > 0U) ? group - 1U : 0);
    HcclSim::CcuSim::PrimLoopGroupAddLoop(
        groupIdx, static_cast<uint32_t>(loop & 0xFFFFFFFFULL), HcclSim::CcuSim::SimCcuLoopParamSrc::VAR_V2, 0, 0, 0,
        iterNumVar, addrOffsetVar, ctxIdVar);
    return CcuResult::CCU_SUCCESS;
}

#ifdef __cplusplus
}
#endif
