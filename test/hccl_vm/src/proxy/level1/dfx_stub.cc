/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Description: DFX 诊断与 Profiling 打桩函数（北向劫持）
 *              对应 hcomm 仓 HcclDfxRegOpInfoByCommId / HcclProfilingReportOp /
 *              HcclReportAicpuKernel / HcclReportAivKernel / HcommGetProfilingSysCycleTime 接口。
 *              桩实现仅打印入参并返回成功，不实际落盘 DFX/Profiling 数据。
 * Create: 2026-07-27
 */

#define HCCL_VM_MODULE "DFX_STUB"

#include <cstdint>

#include "level1_proxy_common.h"
#include "sim_log.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 按通信域 ID 注册算子级 DFX 信息。
 *
 * 将算子执行过程中的诊断（DFX）信息注册到指定通信域上下文，
 * 供后续性能分析或故障定位使用。对应 hcomm 仓同名接口。
 *
 * 桩实现仅记录调用日志并返回成功，不实际持久化 DFX 数据。
 *
 * @param commId 输入：通信域标识字符串（C 字符串）。
 * @param hcclDfxOpInfo 输入：指向 DFX 算子信息结构体的指针，具体布局由 hcomm 定义。
 *
 * @return HcclResult 接口成功返回 HCCL_SUCCESS。
 * @retval HCCL_SUCCESS 注册成功（桩实现始终返回）
 * @retval HCCL_E_PTR commId 为空
 */
HcclResult HcclDfxRegOpInfoByCommId(char* commId, void* hcclDfxOpInfo)
{
    if (commId == nullptr) {
        HCCL_VM_ERROR("{}: commId is nullptr", __func__);
        return HCCL_E_PTR;
    }
    HCCL_VM_INFO("{}: commId={}, hcclDfxOpInfo={:p}", __func__, commId, static_cast<void*>(hcclDfxOpInfo));
    return HCCL_SUCCESS;
}

/**
 * @brief 上报算子级 Profiling 信息。
 *
 * 将通信算子的执行时间戳信息上报到 Profiling 系统，供性能分析使用。
 * 对应 hcomm 仓同名接口。
 *
 * 桩实现仅记录调用日志并返回成功，不实际上报 Profiling 数据。
 *
 * @param comm 输入：通信域句柄。
 * @param beginTime 输入：算子开始执行的时间戳（系统周期数）。
 *
 * @return HcclResult 接口成功返回 HCCL_SUCCESS。
 * @retval HCCL_SUCCESS 上报成功（桩实现始终返回）
 */
HcclResult HcclProfilingReportOp(HcclComm comm, uint64_t beginTime)
{
    uint64_t commId = reinterpret_cast<uint64_t>(comm);
    HCCL_VM_INFO("{}: commId={:d}, beginTime={:d}", __func__, commId, beginTime);
    return HCCL_SUCCESS;
}

/**
 * @brief 上报 AICPU kernel Profiling 信息。
 *
 * 将 AICPU kernel 的执行时间戳与 kernel 名称上报到 Profiling 系统。
 * 对应 hcomm 仓同名接口。
 *
 * 桩实现仅记录调用日志并返回成功，不实际上报 Profiling 数据。
 *
 * @param comm 输入：通信域句柄。
 * @param beginTime 输入：kernel 开始执行的时间戳（系统周期数）。
 * @param kernelName 输入：kernel 名称字符串。
 *
 * @return HcclResult 接口成功返回 HCCL_SUCCESS。
 * @retval HCCL_SUCCESS 上报成功（桩实现始终返回）
 */
HcclResult HcclReportAicpuKernel(HcclComm comm, uint64_t beginTime, char* kernelName)
{
    uint64_t commId = reinterpret_cast<uint64_t>(comm);
    HCCL_VM_INFO(
        "{}: commId={:d}, beginTime={:d}, kernelName={}", __func__, commId, beginTime,
        kernelName ? kernelName : "(null)");
    return HCCL_SUCCESS;
}

/**
 * @brief 上报 AIV kernel Profiling 信息。
 *
 * 将 AIV kernel 的执行时间戳上报到 Profiling 系统。
 * 对应 hcomm 仓同名接口。
 *
 * 桩实现仅记录调用日志并返回成功，不实际上报 Profiling 数据。
 *
 * @param comm 输入：通信域句柄。
 * @param beginTime 输入：kernel 开始执行的时间戳（系统周期数）。
 *
 * @return HcclResult 接口成功返回 HCCL_SUCCESS。
 * @retval HCCL_SUCCESS 上报成功（桩实现始终返回）
 */
HcclResult HcclReportAivKernel(HcclComm comm, uint64_t beginTime)
{
    uint64_t commId = reinterpret_cast<uint64_t>(comm);
    HCCL_VM_INFO("{}: commId={:d}, beginTime={:d}", __func__, commId, beginTime);
    return HCCL_SUCCESS;
}

/**
 * @brief 获取 Profiling 系统周期时间。
 *
 * 返回当前系统周期计数器值，用于 Profiling 时间戳基准。
 * 对应 hcomm 仓同名接口。
 *
 * 桩实现仅记录调用日志并返回 0，不访问真实硬件周期计数器。
 *
 * @return uint64_t 系统周期时间（桩实现始终返回 0）。
 */
uint64_t HcommGetProfilingSysCycleTime()
{
    HCCL_VM_INFO("{}: stub returns 0", __func__);
    return 0;
}

#ifdef __cplusplus
}
#endif
