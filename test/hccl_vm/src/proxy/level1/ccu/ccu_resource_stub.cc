/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define HCCL_VM_MODULE "CCU_RES_STUB"

#include <cstdint>

#include "ccu_res.h"
#include "sim_log.h"
#include "ccu_level1_res.h"

using HcclSim::CcuSim::CcuResMgr;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 在指定 CCU 实例上预约一段连续的 Variable(XN) 资源。
 *
 * 对应真身 HcommCcuVariableAlloc（ccu_launch.cc:367-380）：
 * 定位实例 → CcuVarEventResMgr::Acquire（从 XN 池切连续块 + VA 映射）。
 *
 * 桩实现：CcuResMgr::VariableAlloc 签发 acqHandle，
 * VA = dieBase(dieId) + xnAllocated[dieId] * 8（水位递增）。
 * 容量不足时返回 CCU_E_UNAVAIL（hccl 回退 AICPU 的触发点）。
 *
 * @param insHandle 输入：CCU 实例句柄（HcommCcuInsCreate 签发）。
 * @param dieId     输入：资源所在 die（[0, CCU_MAX_IODIE_NUM)）。
 * @param num       输入：预约的连续资源个数（须 > 0）。
 * @param varHandle 输出：预约句柄（非零，跨 kernel 有效）。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 预约成功
 * @retval CCU_E_PTR   varHandle 为空
 * @retval CCU_E_UNAVAIL 实例无效或 XN 容量不足（回退触发点）
 */
CcuResult HcommCcuVariableAlloc(CcuInsHandle insHandle, uint8_t dieId, uint32_t num, CcuVariableHandle* varHandle)
{
    HCCL_VM_INFO("{}: enter, insHandle=0x{:x}, dieId={}, num={}", __func__, insHandle, dieId, num);
    if (varHandle == nullptr) {
        HCCL_VM_ERROR("{}: varHandle is nullptr", __func__);
        return CcuResult::CCU_E_PTR;
    }
    *varHandle = 0;
    if (num == 0) {
        HCCL_VM_ERROR("{}: num is 0", __func__);
        return CcuResult::CCU_E_PARA;
    }
    const uint64_t acqHandle = CcuResMgr::Instance().VariableAlloc(insHandle, dieId, num);
    if (acqHandle == 0) {
        HCCL_VM_ERROR("{}: variable alloc failed, insHandle=0x{:x}, dieId={}, num={}", __func__, insHandle, dieId, num);
        return CcuResult::CCU_E_UNAVAIL;
    }
    *varHandle = acqHandle;
    HCCL_VM_INFO(
        "{}: exit, varHandle=0x{:x}, insHandle=0x{:x}, dieId={}, num={}", __func__, *varHandle, insHandle, dieId, num);
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 预约一段连续的 Event(CKE) 资源。
 *
 * 桩实现：CcuResMgr::EventAlloc 签发 acqHandle，
 * VA = dieBase + xnCapacity*8 + ckeAllocated[dieId] * 2（CKE 排在 XN 之后）。
 * 容量不足时返回 CCU_E_UNAVAIL。
 *
 * @param insHandle  输入：CCU 实例句柄。
 * @param dieId      输入：资源所在 die。
 * @param num        输入：预约个数。
 * @param eventHandle 输出：预约句柄。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 预约成功
 * @retval CCU_E_PTR   eventHandle 为空
 * @retval CCU_E_UNAVAIL 实例无效或 CKE 容量不足
 */
CcuResult HcommCcuEventAlloc(CcuInsHandle insHandle, uint8_t dieId, uint32_t num, CcuEventHandle* eventHandle)
{
    HCCL_VM_INFO("{}: enter, insHandle=0x{:x}, dieId={}, num={}", __func__, insHandle, dieId, num);
    if (eventHandle == nullptr) {
        HCCL_VM_ERROR("{}: eventHandle is nullptr", __func__);
        return CcuResult::CCU_E_PTR;
    }
    *eventHandle = 0;
    if (num == 0) {
        HCCL_VM_ERROR("{}: num is 0", __func__);
        return CcuResult::CCU_E_PARA;
    }
    const uint64_t acqHandle = CcuResMgr::Instance().EventAlloc(insHandle, dieId, num);
    if (acqHandle == 0) {
        HCCL_VM_ERROR("{}: event alloc failed, insHandle=0x{:x}, dieId={}, num={}", __func__, insHandle, dieId, num);
        return CcuResult::CCU_E_UNAVAIL;
    }
    *eventHandle = acqHandle;
    HCCL_VM_INFO(
        "{}: exit, eventHandle=0x{:x}, insHandle=0x{:x}, dieId={}, num={}", __func__, *eventHandle, insHandle, dieId,
        num);
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 查询预约句柄名下第 index 个 Variable(XN) 的虚拟地址。
 *
 * 纯查表：va = acqRecord.baseVa + index * 8（XN 步长 8 字节）。
 * Host 侧可直接读写该地址（PreSync 跨 rank 交换的数据源）。
 *
 * @param varHandle 输入：HcommCcuVariableAlloc 返回的预约句柄。
 * @param index     输入：预约段内序号（[0, num)）。
 * @param va        输出：虚拟地址。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 查询成功
 * @retval CCU_E_PTR   va 为空
 * @retval CCU_E_NOT_FOUND 句柄不存在或类型不符
 */
CcuResult HcommCcuVariableGetAddr(CcuVariableHandle varHandle, uint32_t index, uint64_t* va)
{
    HCCL_VM_INFO("{}: enter, varHandle=0x{:x}, index={}", __func__, varHandle, index);
    if (va == nullptr) {
        HCCL_VM_ERROR("{}: va is nullptr, varHandle=0x{:x}", __func__, varHandle);
        return CcuResult::CCU_E_PTR;
    }
    *va = 0;
    if (!CcuResMgr::Instance().VariableGetAddr(varHandle, index, *va)) {
        HCCL_VM_ERROR("{}: variable get addr failed, varHandle=0x{:x}, index={}", __func__, varHandle, index);
        return CcuResult::CCU_E_NOT_FOUND;
    }
    HCCL_VM_INFO("{}: exit, varHandle=0x{:x}, index={}, va=0x{:x}", __func__, varHandle, index, *va);
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 查询预约句柄名下第 index 个 Event(CKE) 的虚拟地址。
 *
 * 纯查表：va = acqRecord.baseVa + index * 2（CKE 步长 2 字节）。
 *
 * @param eventHandle 输入：HcommCcuEventAlloc 返回的预约句柄。
 * @param index       输入：预约段内序号。
 * @param va          输出：虚拟地址。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 查询成功
 * @retval CCU_E_PTR   va 为空
 * @retval CCU_E_NOT_FOUND 句柄不存在或类型不符
 */
CcuResult HcommCcuEventGetAddr(CcuEventHandle eventHandle, uint32_t index, uint64_t* va)
{
    HCCL_VM_INFO("{}: enter, eventHandle=0x{:x}, index={}", __func__, eventHandle, index);
    if (va == nullptr) {
        HCCL_VM_ERROR("{}: va is nullptr, eventHandle=0x{:x}", __func__, eventHandle);
        return CcuResult::CCU_E_PTR;
    }
    *va = 0;
    if (!CcuResMgr::Instance().EventGetAddr(eventHandle, index, *va)) {
        HCCL_VM_ERROR("{}: event get addr failed, eventHandle=0x{:x}, index={}", __func__, eventHandle, index);
        return CcuResult::CCU_E_NOT_FOUND;
    }
    HCCL_VM_INFO("{}: exit, eventHandle=0x{:x}, index={}, va=0x{:x}", __func__, eventHandle, index, *va);
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 创建 CCU 实例资源描述符。
 *
 * 对应真身 HcommCcuInsResDescCreate：按 dieId 创建一个资源描述符，
 * 初始请求量为 0，等待 SetNum 逐项设置。
 *
 * @param dieId   输入：Die ID（[0, CCU_MAX_IODIE_NUM)）。
 * @param resDesc 输出：描述符句柄。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 创建成功
 * @retval CCU_E_PTR   resDesc 为空
 * @retval CCU_E_PARA  dieId 超出范围
 */
CcuResult HcommCcuInsResDescCreate(uint32_t dieId, HcommCcuResDescHandle* resDesc)
{
    HCCL_VM_INFO("{}: enter, dieId={}", __func__, dieId);
    if (resDesc == nullptr) {
        HCCL_VM_ERROR("{}: resDesc is nullptr", __func__);
        return CcuResult::CCU_E_PTR;
    }
    *resDesc = 0;
    if (dieId >= 2) { /* CCU_MAX_IODIE_NUM */
        HCCL_VM_ERROR("{}: dieId={} out of range", __func__, dieId);
        return CcuResult::CCU_E_PARA;
    }
    const uint64_t handle = CcuResMgr::Instance().CreateResDesc(dieId);
    if (handle == 0) {
        HCCL_VM_ERROR("{}: create res desc failed, dieId={}", __func__, dieId);
        return CcuResult::CCU_E_INTERNAL;
    }
    *resDesc = handle;
    HCCL_VM_INFO("{}: exit, resDesc=0x{:x}, dieId={}", __func__, *resDesc, dieId);
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 销毁 CCU 实例资源描述符。
 *
 * @param resDesc 输入：描述符句柄。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 销毁成功
 * @retval CCU_E_NOT_FOUND 描述符未注册
 */
CcuResult HcommCcuInsResDescDestroy(HcommCcuResDescHandle resDesc)
{
    HCCL_VM_INFO("{}: enter, resDesc=0x{:x}", __func__, resDesc);
    if (!CcuResMgr::Instance().DestroyResDesc(resDesc)) {
        HCCL_VM_ERROR("{}: destroy failed, resDesc=0x{:x} not found", __func__, resDesc);
        return CcuResult::CCU_E_NOT_FOUND;
    }
    HCCL_VM_INFO("{}: exit, resDesc=0x{:x}", __func__, resDesc);
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 查询 CCU 资源描述符中已设置的 Die ID。
 *
 * @param resDesc 输入：描述符句柄。
 * @param dieId   输出：Die ID。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 查询成功
 * @retval CCU_E_PTR   dieId 为空
 * @retval CCU_E_NOT_FOUND 描述符未注册
 */
CcuResult HcommCcuInsResDescQueryDieId(HcommCcuResDescHandle resDesc, uint32_t* dieId)
{
    HCCL_VM_INFO("{}: enter, resDesc=0x{:x}", __func__, resDesc);
    if (dieId == nullptr) {
        HCCL_VM_ERROR("{}: dieId is nullptr, resDesc=0x{:x}", __func__, resDesc);
        return CcuResult::CCU_E_PTR;
    }
    *dieId = 0;
    if (!CcuResMgr::Instance().QueryResDescDieId(resDesc, *dieId)) {
        HCCL_VM_ERROR("{}: query die id failed, resDesc=0x{:x} not found", __func__, resDesc);
        return CcuResult::CCU_E_NOT_FOUND;
    }
    HCCL_VM_INFO("{}: exit, resDesc=0x{:x}, dieId={}", __func__, resDesc, *dieId);
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 按资源类型设置 CCU 资源描述符中的资源数量。
 *
 * hccl 侧 BuildAggregatedResReq 按 dieId 逐项设置（LOOP/CCU_BUF/VARIABLE/...）。
 *
 * @param resDesc 输入：描述符句柄。
 * @param resType 输入：资源类型（HcommCcuResType 枚举值）。
 * @param resNum  输入：请求量（0 表示不申请该类型）。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 设置成功
 * @retval CCU_E_PARA  resType 非法或描述符未注册
 */
CcuResult HcommCcuInsResDescSetNum(HcommCcuResDescHandle resDesc, HcommCcuResType resType, uint32_t resNum)
{
    HCCL_VM_INFO(
        "{}: enter, resDesc=0x{:x}, resType={}, resNum={}", __func__, resDesc, static_cast<uint32_t>(resType), resNum);
    if (!CcuResMgr::Instance().SetResDescNum(resDesc, static_cast<uint32_t>(resType), resNum)) {
        HCCL_VM_ERROR("{}: set num failed, resDesc=0x{:x} not found or destroyed", __func__, resDesc);
        return CcuResult::CCU_E_PARA;
    }
    HCCL_VM_INFO(
        "{}: exit, resDesc=0x{:x}, resType={}, resNum={}", __func__, resDesc, static_cast<uint32_t>(resType), resNum);
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 按资源类型查询 CCU 资源描述符中已设置的资源数量。
 *
 * @param resDesc 输入：描述符句柄。
 * @param resType 输入：资源类型。
 * @param resNum  输出：已设置的请求量。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 查询成功
 * @retval CCU_E_PTR   resNum 为空
 * @retval CCU_E_NOT_FOUND 描述符未注册或类型不匹配
 */
CcuResult HcommCcuInsResDescQueryNum(HcommCcuResDescHandle resDesc, HcommCcuResType resType, uint32_t* resNum)
{
    HCCL_VM_INFO("{}: enter, resDesc=0x{:x}, resType={}", __func__, resDesc, static_cast<uint32_t>(resType));
    if (resNum == nullptr) {
        HCCL_VM_ERROR("{}: resNum is nullptr, resDesc=0x{:x}", __func__, resDesc);
        return CcuResult::CCU_E_PTR;
    }
    *resNum = 0;
    if (!CcuResMgr::Instance().QueryResDescNum(resDesc, static_cast<uint32_t>(resType), *resNum)) {
        HCCL_VM_ERROR("{}: query num failed, resDesc=0x{:x} not found or type mismatch", __func__, resDesc);
        return CcuResult::CCU_E_NOT_FOUND;
    }
    HCCL_VM_INFO(
        "{}: exit, resDesc=0x{:x}, resType={}, resNum={}", __func__, resDesc, static_cast<uint32_t>(resType), *resNum);
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 基于资源描述符创建 CCU 实例。
 *
 * 对应真身 HcommCcuInsCreate：hccl 传入一组 desc（每 die 一个），
 * 创建 CCU 实例并返回句柄。真身在创建过程中做资源池切分。
 *
 * 桩实现：从 descs[] 取 dieMask，调 CcuResMgr::CreateIns 签发实例句柄。
 *
 * @param resDescs     输入：描述符句柄数组。
 * @param resDescNum   输入：描述符个数。
 * @param ccuInsHandle 输出：创建成功后返回的实例句柄。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 创建成功
 * @retval CCU_E_PTR   resDescs 或 ccuInsHandle 为空
 * @retval CCU_E_PARA  resDescNum 为 0 或 dieId 重复
 */
CcuResult HcommCcuInsCreate(const HcommCcuResDescHandle* resDescs, uint32_t resDescNum, CcuInsHandle* ccuInsHandle)
{
    HCCL_VM_INFO("{}: enter, resDescNum={}", __func__, resDescNum);
    if (resDescs == nullptr || ccuInsHandle == nullptr) {
        HCCL_VM_ERROR("{}: resDescs or ccuInsHandle is nullptr", __func__);
        return CcuResult::CCU_E_PTR;
    }
    *ccuInsHandle = 0;
    if (resDescNum == 0 || resDescNum > 2) { /* CCU_MAX_IODIE_NUM */
        HCCL_VM_ERROR("{}: resDescNum={} out of range", __func__, resDescNum);
        return CcuResult::CCU_E_PARA;
    }
    /* 从 descs[] 取 dieMask */
    uint32_t dieMask = 0;
    for (uint32_t i = 0; i < resDescNum; i++) {
        uint32_t dieId = 0;
        if (!CcuResMgr::Instance().QueryResDescDieId(resDescs[i], dieId)) {
            HCCL_VM_ERROR("{}: resDesc[{}]=0x{:x} not found", __func__, i, resDescs[i]);
            return CcuResult::CCU_E_PARA;
        }
        if (dieMask & (1U << dieId)) {
            HCCL_VM_ERROR("{}: dieId={} duplicated", __func__, dieId);
            return CcuResult::CCU_E_PARA;
        }
        dieMask |= (1U << dieId);
    }
    const uint64_t insHandle = CcuResMgr::Instance().CreateIns(dieMask);
    if (insHandle == 0) {
        HCCL_VM_ERROR("{}: create ins failed, dieMask=0x{:x}", __func__, dieMask);
        return CcuResult::CCU_E_INTERNAL;
    }
    /* 回填 desc 的实例归属与实批量 */
    for (uint32_t i = 0; i < resDescNum; i++) {
        (void)CcuResMgr::Instance().QueryResDescIns(insHandle, resDescs[i]);
    }
    *ccuInsHandle = insHandle;
    HCCL_VM_INFO(
        "{}: exit, insHandle=0x{:x}, dieMask=0x{:x}, resDescNum={}", __func__, *ccuInsHandle, dieMask, resDescNum);
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 使用当前 Device 上所有已使能 ioDie 的全部资源创建 CCU 实例。
 *
 * 对应真身 HcommCcuInsCreateDefault：dieIds/dieNum 为保留参数（当前版本不读取）。
 * 桩实现：默认 dieMask=0b11（两个 die 全开），调 CreateIns。
 *
 * @param dieIds      输入：保留参数（不读取）。
 * @param dieNum      输入：保留参数（不读取）。
 * @param ccuInsHandle 输出：实例句柄。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 创建成功
 * @retval CCU_E_PTR   ccuInsHandle 为空
 */
CcuResult HcommCcuInsCreateDefault(const uint32_t* dieIds, uint32_t dieNum, CcuInsHandle* ccuInsHandle)
{
    HCCL_VM_INFO("{}: enter, dieNum={}", __func__, dieNum);
    if (ccuInsHandle == nullptr) {
        HCCL_VM_ERROR("{}: ccuInsHandle is nullptr", __func__);
        return CcuResult::CCU_E_PTR;
    }
    *ccuInsHandle = 0;
    /* 默认双 die 全开 */
    constexpr uint32_t DEFAULT_DIE_MASK = 0x3;
    const uint64_t insHandle = CcuResMgr::Instance().CreateIns(DEFAULT_DIE_MASK);
    if (insHandle == 0) {
        HCCL_VM_ERROR("{}: create ins failed, dieMask=0x{:x}", __func__, DEFAULT_DIE_MASK);
        return CcuResult::CCU_E_INTERNAL;
    }
    *ccuInsHandle = insHandle;
    HCCL_VM_INFO("{}: exit, insHandle=0x{:x}, dieMask=0x{:x}", __func__, *ccuInsHandle, DEFAULT_DIE_MASK);
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 销毁 CCU 实例。
 *
 * 对应真身 HcommCcuInsDestroy。级联清理该实例下的所有描述符和预约。
 *
 * @param ccuInsHandle 输入：实例句柄。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 销毁成功
 * @retval CCU_E_NOT_FOUND 实例未注册
 */
CcuResult HcommCcuInsDestroy(CcuInsHandle ccuInsHandle)
{
    HCCL_VM_INFO("{}: enter, insHandle=0x{:x}", __func__, ccuInsHandle);
    if (!CcuResMgr::Instance().DestroyIns(ccuInsHandle)) {
        HCCL_VM_ERROR("{}: destroy failed, insHandle=0x{:x} not found", __func__, ccuInsHandle);
        return CcuResult::CCU_E_NOT_FOUND;
    }
    HCCL_VM_INFO("{}: exit, insHandle=0x{:x}", __func__, ccuInsHandle);
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 查询 CCU 实例在资源描述符所属 ioDie 上占用的资源。
 *
 * 对应真身 HcommCcuInsQueryResDesc：读取 desc.dieId 作为待查询 ioDie，
 * 查询成功后写入该 ioDie 上的占用资源数量到 desc.allocNum。
 *
 * @param ccuInsHandle 输入：实例句柄。
 * @param resDesc      输入/输出：描述符（读 dieId，写 allocNum）。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 查询成功
 * @retval CCU_E_NOT_FOUND 描述符未注册或实例无效
 */
CcuResult HcommCcuInsQueryResDesc(CcuInsHandle ccuInsHandle, HcommCcuResDescHandle resDesc)
{
    HCCL_VM_INFO("{}: enter, insHandle=0x{:x}, resDesc=0x{:x}", __func__, ccuInsHandle, resDesc);
    if (!CcuResMgr::Instance().IsValidIns(ccuInsHandle)) {
        HCCL_VM_ERROR("{}: insHandle=0x{:x} invalid", __func__, ccuInsHandle);
        return CcuResult::CCU_E_NOT_FOUND;
    }
    if (!CcuResMgr::Instance().QueryResDescIns(ccuInsHandle, resDesc)) {
        HCCL_VM_ERROR("{}: resDesc=0x{:x} not found", __func__, resDesc);
        return CcuResult::CCU_E_NOT_FOUND;
    }
    HCCL_VM_INFO("{}: exit, insHandle=0x{:x}, resDesc=0x{:x}", __func__, ccuInsHandle, resDesc);
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 查询指定 Die 上的剩余 CCU 资源（查询最大连续资源）。
 *
 * 对应真身 HcommCcuQueryRemainResDesc（ccu_res.h:158-168）：
 * 读取 desc 的 dieId 作为待查询 ioDie，查询成功后写入各资源类型的最大连续剩余数量。
 *
 * 回退触发点的审计数据：remainNum = capacity - allocated（per-die 水位差）。
 * hccl 检查后决定是否回退 AICPU（remainNum 不足 → HCCL_E_UNAVAIL → fallback）。
 *
 * @param resDesc 输入/输出：描述符（读 dieId，写 remainNum）。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 查询成功
 * @retval CCU_E_NOT_FOUND 描述符未注册
 * @retval CCU_E_UNAVAIL   指定的 Die 未使能
 */
CcuResult HcommCcuQueryRemainResDesc(HcommCcuResDescHandle resDesc)
{
    HCCL_VM_INFO("{}: enter, resDesc=0x{:x}", __func__, resDesc);
    if (!CcuResMgr::Instance().QueryRemainResDesc(resDesc)) {
        HCCL_VM_ERROR("{}: query remain failed, resDesc=0x{:x} not found", __func__, resDesc);
        return CcuResult::CCU_E_NOT_FOUND;
    }
    HCCL_VM_INFO("{}: exit, resDesc=0x{:x}", __func__, resDesc);
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 查询 CCU Kernel 的资源诉求。
 *
 * 对应真身 HcommCcuKernelQueryResReq（ccu_res_c_adpt.cc:148-198）：
 * 以查询模式执行 kernelFunc，统计资源需求写入 desc。
 *
 * 桩实现（分两档）：
 * - 当前阶段：hccl 侧 SetNum 已预估请求量（BuildAggregatedResReq），
 *   覆盖回退判定需求，此处透传 desc；
 * - 后续增强：接 CcuResMgr::QueryResourceReq 以查询模式执行 kernelFunc
 *   统计真实需求。
 *
 * @param kernelFunc 输入：CCU Kernel 函数指针。
 * @param kernelArgs 输入：CCU Kernel 参数数组。
 * @param argNum     输入：参数个数（0 或 1）。
 * @param resDesc    输入/输出：资源描述符。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 查询成功
 * @retval CCU_E_PTR   kernelFunc 为空
 * @retval CCU_E_PARA  resDesc 为 0 或 argNum > 1 或 kernelArgs 无效
 */
CcuResult HcommCcuKernelQueryResReq(
    const void* kernelFunc, const void** kernelArgs, uint32_t argNum, HcommCcuResDescHandle resDesc)
{
    HCCL_VM_INFO("{}: enter, kernelFunc={}, argNum={}, resDesc=0x{:x}", __func__, kernelFunc, argNum, resDesc);
    if (kernelFunc == nullptr) {
        HCCL_VM_ERROR("{}: kernelFunc is nullptr", __func__);
        return CcuResult::CCU_E_PTR;
    }
    if (resDesc == 0 || argNum > 1) {
        HCCL_VM_ERROR("{}: resDesc=0x{:x} or argNum={} invalid", __func__, resDesc, argNum);
        return CcuResult::CCU_E_PARA;
    }
    if (argNum == 1 && (kernelArgs == nullptr || kernelArgs[0] == nullptr)) {
        HCCL_VM_ERROR("{}: kernelArgs invalid while argNum={}", __func__, argNum);
        return CcuResult::CCU_E_PTR;
    }
    uint32_t dieId = 0;
    if (!CcuResMgr::Instance().QueryResDescDieId(resDesc, dieId)) {
        HCCL_VM_ERROR("{}: resDesc=0x{:x} not found", __func__, resDesc);
        return CcuResult::CCU_E_PARA;
    }
    if (!CcuResMgr::Instance().QueryResourceReq(resDesc, kernelFunc, kernelArgs, argNum, dieId)) {
        HCCL_VM_ERROR("{}: query resource req failed, resDesc=0x{:x}", __func__, resDesc);
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_VM_INFO("{}: exit, resDesc=0x{:x}, dieId={}", __func__, resDesc, dieId);
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 查询/计算一段本端内存区域的 CCU 访问 token。
 *
 * 对应真身 HcommCcuGetMemToken（ccu_launch.cc:405-437）：
 * 1. CcuQueryTokenInfo（内部 token 查询）
 * 2. 失败回退 RtsUbDevQueryInfo（→ level2 桩，确定性伪 token）
 * 3. CcuRep::CcuCombineTokenInfo(tokenId, tokenValue, 1) 打包为单个 uint64
 *
 * 桩实现：确定性可逆编码 token = srcVa。
 * - 纯函数、无状态、无 DB 依赖
 * - 可逆性：从 token 可还原 srcVa（checker 地址反查需要）
 * - 确定性：同一 srcVa 永远得到同一 token（多次调用幂等）
 *
 * @param srcVa     输入：内存区域起始虚地址（须非零）。
 * @param size      输入：内存区域长度（须非零）。
 * @param tokenInfo 输出：token 值。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 查询成功
 * @retval CCU_E_PTR   tokenInfo 为空
 * @retval CCU_E_PARA  srcVa 或 size 为零
 */
CcuResult HcommCcuGetMemToken(uint64_t srcVa, uint64_t size, uint64_t* tokenInfo)
{
    HCCL_VM_INFO("{}: enter, srcVa=0x{:x}, size={}", __func__, srcVa, size);
    if (tokenInfo == nullptr) {
        HCCL_VM_ERROR("{}: tokenInfo is nullptr", __func__);
        return CcuResult::CCU_E_PTR;
    }
    *tokenInfo = 0;
    if (srcVa == 0 || size == 0) {
        HCCL_VM_ERROR("{}: srcVa or size is 0, srcVa=0x{:x}, size={}", __func__, srcVa, size);
        return CcuResult::CCU_E_PARA;
    }
    /* 确定性可逆编码：token = srcVa
     * 与 level2 rtUbDevQueryInfo no-op 桩的确定性语义一致；
     * token 是不透明元数据，checker 侧按 VA 编址反查，只需可逆。 */
    *tokenInfo = srcVa;
    HCCL_VM_INFO("{}: exit, token=0x{:x}", __func__, *tokenInfo);
    return CcuResult::CCU_SUCCESS;
}

#ifdef __cplusplus
}
#endif
