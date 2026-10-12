/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define HCCL_VM_MODULE "CCU_CTRL_STUB"

#include <cstdint>

#include "ccu_launch.h"
#include "hccl/hccl_ccu_res.h"
#include "sim_log.h"
#include "sim_common_defs.h"
#include "ccu_kernel_sim.h" // 传递包含 ccu_level1_common.h（类型 + inline ToCcuResult）
#include "ccu_level1_res.h" // CcuResMgr（绑定/实例管理）

// 符号绑定说明：ccu_launch.h 将 Register 三步与 Launch 声明为 weak（HCOMM_WEAK_SYMBOL，
// hcomm 官方的上层替换机制），本文件的定义继承该属性（nm 显示 W）。真实 libhcomm 中
// 这 4 个符号同为弱定义，且 LD_PRELOAD 全局作用域中 level1 先于 libhcomm 加载，
// 因此弱定义仍然优先命中（libhccl 的弱 UND 引用与 HcclDlsym 的 RTLD_DEFAULT 探测
// 均解析到本桩）。其余 CCU 符号（资源族/绑定/88 原语）均为强定义。

// ToCcuResult 为 inline 纯函数，定义在 ccu_level1_common.h 中（§9 返回码转换）。
using HcclSim::CcuSim::CcuResMgr;
using HcclSim::CcuSim::ToCcuResult;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 将 CCU 实例绑定到指定 HCCL 通信域。
 *
 * 对应真身 HcclCommAssignCcuIns（hccl_ccu_res.h:32）：
 * 将 HcommCcuInsCreate 签发的实例绑定到通信域。
 * 绑定规则（与真身一致）：一个通信域最多绑定一个实例；
 * 重复绑定同一实例幂等；绑定不同实例返回 HCCL_E_PARA。
 * 绑定成功后，实例的所有权转移给通信域，HcclCommDestroy 时统一解绑。
 *
 * @param comm      输入：通信域句柄（由 HcclCommInitXxx 系列接口返回）。
 * @param insHandle 输入：CCU 实例句柄（由 HcommCcuInsCreate 返回）。
 *
 * @return 成功返回 HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 绑定成功
 * @retval HCCL_E_PTR comm 为空
 * @retval HCCL_E_PARA 实例无效或通信域已绑定其他实例
 */
HcclResult HcclCommAssignCcuIns(HcclComm comm, CcuInsHandle insHandle)
{
    HCCL_VM_INFO("{}: enter, comm={}, insHandle=0x{:x}", __func__, reinterpret_cast<uint64_t>(comm), insHandle);
    if (comm == nullptr) {
        HCCL_VM_ERROR("{}: comm is nullptr", __func__);
        return HCCL_E_PTR;
    }
    const uint64_t commId = reinterpret_cast<uint64_t>(comm);
    if (!CcuResMgr::Instance().BindInsToComm(commId, insHandle)) {
        HCCL_VM_ERROR(
            "{}: bind failed, commId={}, insHandle=0x{:x} invalid or already bound", __func__, commId, insHandle);
        return HCCL_E_PARA;
    }
    HCCL_VM_INFO("{}: exit, commId={}, insHandle=0x{:x}", __func__, commId, insHandle);
    return HCCL_SUCCESS;
}

/**
 * @brief 查询通信域绑定的 CCU 实例句柄（legacy 兼容路径）。
 *
 * 对应真身 HcclCommQueryCcuIns（hccl_ccu_res.h:45）：
 * 查询通过 AssignCcuIns 绑定给通信域的 CCU 实例句柄。
 * 未绑定时返回 HCCL_E_UNAVAIL，不创建句柄。
 *
 * @param comm      输入：通信域句柄。
 * @param insHandles 输出：返回的实例句柄数组（单个）。
 * @param insNum    输出：实例个数（0 或 1）。
 *
 * @return 成功返回 HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 查询成功
 * @retval HCCL_E_PTR comm、insHandles 或 insNum 为空
 * @retval HCCL_E_UNAVAIL 未绑定实例
 */
HcclResult HcclCommQueryCcuIns(HcclComm comm, CcuInsHandle* insHandles, uint32_t* insNum)
{
    HCCL_VM_INFO("{}: enter, comm={}", __func__, reinterpret_cast<uint64_t>(comm));
    if (comm == nullptr || insHandles == nullptr || insNum == nullptr) {
        HCCL_VM_ERROR("{}: comm, insHandles or insNum is nullptr", __func__);
        return HCCL_E_PTR;
    }
    *insNum = 0;
    insHandles[0] = 0;
    const uint64_t commId = reinterpret_cast<uint64_t>(comm);
    uint64_t insHandle = 0;
    if (!CcuResMgr::Instance().QueryBoundIns(commId, insHandle)) {
        HCCL_VM_INFO("{}: no ins bound, commId={}", __func__, commId);
        return HCCL_E_UNAVAIL;
    }
    insHandles[0] = insHandle;
    *insNum = 1;
    HCCL_VM_INFO("{}: exit, commId={}, insHandle=0x{:x}, insNum={}", __func__, commId, insHandle, *insNum);
    return HCCL_SUCCESS;
}

/**
 * @brief 查询通信域通过 HcclCommAssignCcuIns 绑定的 CCU 实例句柄。
 *
 * 对应真身 HcclCommQueryAssignedCcuIns（hccl_ccu_res.h:58）：
 * 与 HcclCommQueryCcuIns 语义相同（本方案中两条路径共用同一个绑定表）。
 * 未绑定时返回 HCCL_E_UNAVAIL（hccl 契约：触发新建实例路径），
 * 已绑定返回 SUCCESS + 句柄 + insNum=1（hccl 走复用路径）。
 *
 * @param comm      输入：通信域句柄。
 * @param insHandles 输出：返回的实例句柄。
 * @param insNum    输出：实例个数。
 *
 * @return 成功返回 HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 查询成功（已绑定）
 * @retval HCCL_E_PTR comm、insHandles 或 insNum 为空
 * @retval HCCL_E_UNAVAIL 未绑定实例（触发 hccl 新建路径）
 */
HcclResult HcclCommQueryAssignedCcuIns(HcclComm comm, CcuInsHandle* insHandles, uint32_t* insNum)
{
    HCCL_VM_INFO("{}: enter, comm={}", __func__, reinterpret_cast<uint64_t>(comm));
    if (comm == nullptr || insHandles == nullptr || insNum == nullptr) {
        HCCL_VM_ERROR("{}: comm, insHandles or insNum is nullptr", __func__);
        return HCCL_E_PTR;
    }
    *insNum = 0;
    insHandles[0] = 0;
    const uint64_t commId = reinterpret_cast<uint64_t>(comm);
    uint64_t insHandle = 0;
    if (!CcuResMgr::Instance().QueryBoundIns(commId, insHandle)) {
        HCCL_VM_INFO("{}: no ins bound, commId={}", __func__, commId);
        return HCCL_E_UNAVAIL;
    }
    insHandles[0] = insHandle;
    *insNum = 1;
    HCCL_VM_INFO("{}: exit, commId={}, insHandle=0x{:x}, insNum={}", __func__, commId, insHandle, *insNum);
    return HCCL_SUCCESS;
}

/**
 * @brief 开启 CCU kernel 注册事务：打开注册表事务窗口，供后续 Register 登记函数。
 *
 * 与真实语义的差异：不创建 CcuKernel、不初始化翻译器，注册期零执行。
 * @return 成功返回 CCU_SUCCESS；事务已开启返回 CCU_E_PARA。
 */
CcuResult HcommCcuKernelRegisterStart(CcuInsHandle insHandle)
{
    HCCL_VM_INFO("{}: insHandle={}", __func__, insHandle);
    return ToCcuResult(HcclSim::CcuSim::BeginRegister(insHandle));
}

/**
 * @brief 注册单个 CCU kernel：就地执行 kernelFunc 录制原语轨迹（录制-回放架构）。
 *
 * 与真实语义的差异：真实实现在此就地执行 kernelFunc 录制 IR 并计算资源占用；
 * CPU 模拟推迟到 HcommCcuKernelLaunch 时携带 taskArgs 解释执行。
 * 约束与真实实现一致：argNum 仅允许 0 或 1，argNum == 1 时 kernelArgs[0] 非空。
 * @return 成功返回 CCU_SUCCESS 并签发 kernelHandle；参数错误返回 CCU_E_PTR/CCU_E_PARA；
 *         无打开的注册事务返回 CCU_E_UNAVAIL。
 */
CcuResult HcommCcuKernelRegister(
    CcuInsHandle insHandle, uint32_t dieId, const char* kernelFuncName, const void* kernelFunc, const void** kernelArgs,
    uint32_t argNum, CcuKernelHandle* kernelHandle)
{
    HCCL_VM_INFO(
        "{}: insHandle={}, dieId={}, name={}, argNum={}", __func__, insHandle, dieId,
        (kernelFuncName != nullptr) ? kernelFuncName : "", argNum);

    if (kernelFunc == nullptr) {
        HCCL_VM_ERROR("{}: kernelFunc is null", __func__);
        return CcuResult::CCU_E_PTR;
    }
    if (kernelHandle == nullptr) {
        HCCL_VM_ERROR("{}: kernelHandle out param is null", __func__);
        return CcuResult::CCU_E_PTR;
    }
    if (argNum > 1U) {
        // 与真实实现一致：注册参数当前仅支持 0 或 1 个。
        HCCL_VM_ERROR("{}: argNum={} now only support 0 or 1", __func__, argNum);
        return CcuResult::CCU_E_PARA;
    }
    const void* kernelArg = nullptr;
    if (argNum == 1U) {
        if (kernelArgs == nullptr || kernelArgs[0] == nullptr) {
            HCCL_VM_ERROR("{}: kernelArgs is null while argNum=1", __func__);
            return CcuResult::CCU_E_PTR;
        }
        kernelArg = kernelArgs[0];
    }

    uint64_t handle = 0;
    const CcuResult ret = ToCcuResult(
        HcclSim::CcuSim::RegisterKernel(insHandle, dieId, kernelFuncName, kernelFunc, kernelArg, argNum, handle));
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }
    *kernelHandle = handle;
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 结束 CCU kernel 注册事务：关闭事务窗口，无微码翻译与指令下发。
 *
 * 与真实语义的差异：真实实现在此触发 Translate（微码翻译）与 LoadInstruction
 * （TLV 下发指令空间）；CPU 模拟不需要微码，直接关闭事务。
 * @return 成功返回 CCU_SUCCESS；事务未开启返回 CCU_E_UNAVAIL。
 */
CcuResult HcommCcuKernelRegisterEnd(CcuInsHandle insHandle)
{
    HCCL_VM_INFO("{}: insHandle={}", __func__, insHandle);
    return ToCcuResult(HcclSim::CcuSim::EndRegister(insHandle));
}

/**
 * @brief 查询 CCU kernel 的 taskArgs 数量。
 *
 * 从注册表中查找 kernel 条目，返回其 SQE 参数槽位数（RT_CCU_SQE_ARGS_LEN）。
 * kernel 未注册时返回 CCU_E_NOT_FOUND。
 *
 * @param kernelHandle 输入：kernel 句柄（Register 签发）。
 * @param taskArgsNum  输出：SQE 槽位数（13）。
 *
 * @return 成功返回 CCU_SUCCESS，其他失败。
 * @retval CCU_SUCCESS 查询成功
 * @retval CCU_E_PTR taskArgsNum 为空
 * @retval CCU_E_NOT_FOUND kernel 未注册
 */
CcuResult HcommCcuGetTaskArgsNum(CcuKernelHandle kernelHandle, uint32_t* taskArgsNum)
{
    HCCL_VM_INFO("{}: enter, kernelHandle={}", __func__, kernelHandle);
    if (taskArgsNum == nullptr) {
        HCCL_VM_ERROR("{}: taskArgsNum is nullptr, kernelHandle={}", __func__, kernelHandle);
        return CcuResult::CCU_E_PTR;
    }
    *taskArgsNum = 0;
    HcclSim::CcuSim::SimCcuKernelEntry entry;
    if (!HcclSim::CcuSim::FindKernel(kernelHandle, entry)) {
        HCCL_VM_ERROR("{}: kernel handle {} not registered", __func__, kernelHandle);
        return CcuResult::CCU_E_NOT_FOUND;
    }
    *taskArgsNum = RT_CCU_SQE_ARGS_LEN;
    HCCL_VM_INFO("{}: exit, kernelHandle={}, taskArgsNum={}", __func__, kernelHandle, *taskArgsNum);
    return CcuResult::CCU_SUCCESS;
}

/**
 * @brief 发射 CCU kernel：取回注册轨迹，携带 taskArgs 回放，生成 CPU 任务序列。
 *
 * 与真实语义的差异：真实实现通过 GeneTaskParams 组 SQE 参数后 rtCCULaunch 下发
 * 微码任务；CPU 模拟在进程内回放录制轨迹（LoadArg 绑定 taskArgs、算逻求值、
 * if/while 条件判定、LoopGroup 展开、数据原语产任务），执行结束后统一归一化
 * 并插入 TaskCollection 供 Checker 校验。
 *
 * @param threadHandle 输入：线程句柄（sim::HcclThread 主键）。
 * @param kernelHandle 输入：kernel 句柄。
 * @param taskArgs     输入：Launch 参数数组（uint64 数组）。
 * @param argNum       输入：参数个数。
 *
 * @return 成功返回 CCU_SUCCESS；参数/句柄错误返回 CCU_E_PARA/CCU_E_PTR/CCU_E_NOT_FOUND；
 *         回放或任务插入失败返回 CCU_E_INTERNAL。
 */
CcuResult
HcommCcuKernelLaunch(ThreadHandle threadHandle, CcuKernelHandle kernelHandle, const void* taskArgs, uint32_t argNum)
{
    HCCL_VM_INFO(
        "{}: enter, threadHandle={}, kernelHandle={}, argNum={}", __func__, threadHandle, kernelHandle, argNum);

    if (threadHandle == 0U) {
        HCCL_VM_ERROR("{}: thread handle is empty", __func__);
        return CcuResult::CCU_E_PARA;
    }
    if (kernelHandle == 0U) {
        HCCL_VM_ERROR("{}: kernel handle is empty", __func__);
        return CcuResult::CCU_E_PARA;
    }
    if (argNum > 0U && taskArgs == nullptr) {
        HCCL_VM_ERROR("{}: taskArgs is null while argNum={}", __func__, argNum);
        return CcuResult::CCU_E_PTR;
    }

    HcclSim::CcuSim::SimCcuKernelEntry entry;
    if (!HcclSim::CcuSim::FindKernel(kernelHandle, entry)) {
        HCCL_VM_ERROR("{}: kernel handle {} not registered", __func__, kernelHandle);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    // taskArgs 为 uint64 数组（与真实 GeneTaskParams 的口径一致，个数不受 SQE 13 槽限制）。
    const auto* args = reinterpret_cast<const uint64_t*>(taskArgs);
    return ToCcuResult(HcclSim::CcuSim::ExecuteKernelOnLaunch(threadHandle, kernelHandle, entry, args, argNum));
}

#ifdef __cplusplus
}
#endif
