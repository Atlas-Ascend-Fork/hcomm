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
 * Description: 设备侧 DFX 诊断与 Profiling 打桩函数（level1 劫持）
 *              - HcommIsSupportHcommRegOpInfo / HcommRegOpInfo /
 *                HcommIsSupportHcommRegOpTaskException / HcommRegOpTaskException
 *                算子诊断信息注册接口。
 *              - HcclDfxRegOpInfoByCommId
 *                通信域级 DFX 算子信息注册接口。
 *              - HcommProfilingInit / HcommProfilingEnd /
 *                HcommProfilingReportDeviceOp /
 *                HcommProfilingReportKernelStartTask / HcommProfilingReportKernelEndTask /
 *                HcommProfilingReportMainStreamAndFirstTask / HcommProfilingReportMainStreamAndLastTask /
 *                HcommProfilingReportDeviceHcclOpInfo
 *                设备侧 Profiling 上报接口。
 *              桩实现仅打印入参并返回成功。
 * Create: 2026-07-27
 */

#define HCCL_VM_MODULE "DEV_L1_DFX"

#include <cstddef>
#include <cstdint>

#include "hcomm_diag.h"
#include "sim_log.h"

/* ThreadHandle 在 CANN hcomm_primitives.h 中定义为 uint64_t，
 * 此处本地定义以避免引入 securec.h / acl_rt.h 等重依赖。 */
typedef uint64_t ThreadHandle;

/* HcomProInfoTmp 定义来自 hccl 仓 hcomm_device_profiling_dl.h，
 * 不在 CANN 公共头中，此处本地拷贝以匹配 ABI。 */
struct HcomProInfoTmp {
    static constexpr size_t MAX_LENGTH = 128;
    uint8_t dataType;
    uint8_t cmdType;
    uint64_t dataCount;
    uint32_t rankSize;
    uint32_t userRank;
    uint32_t blockDim = 0;
    uint64_t beginTime;
    uint32_t root;
    uint32_t slaveThreadNum;
    uint64_t commNameLen;
    uint64_t algTypeLen;
    char tag[MAX_LENGTH];
    char commName[MAX_LENGTH];
    char algType[MAX_LENGTH];
    bool isCapture = false;
    bool isAiv = false;
    uint8_t reserved[MAX_LENGTH];
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 查询是否支持注册算子信息。
 *
 * 对应 hcomm 仓 HcommIsSupport 系列接口，由 dlsym 框架在运行期动态解析。
 *
 * 桩实现始终返回 true，表示支持 HcommRegOpInfo。
 *
 * @return bool 始终返回 true。
 */
bool HcommIsSupportHcommRegOpInfo()
{
    HCCL_VM_INFO("{}: stub returns true", __func__);
    return true;
}

/**
 * @brief 注册算子信息到通信域（设备侧接口）。
 *
 * 将算子执行过程中的诊断信息注册到指定通信域上下文。
 * 对应 hcomm 仓同名接口。
 *
 * 桩实现仅记录调用日志并返回成功，不实际持久化算子信息。
 *
 * @param commId 输入：通信域标识字符串。
 * @param opInfo 输入：指向算子信息结构体的指针，具体布局由 hcomm 定义。
 * @param size 输入：算子信息数据长度。
 * @return HcclResult 成功返回 HCCL_SUCCESS，commId 为空返回 HCCL_E_PTR。
 */
HcclResult HcommRegOpInfo(const char* commId, void* opInfo, size_t size)
{
    if (commId == nullptr) {
        HCCL_VM_ERROR("{}: commId is nullptr", __func__);
        return HCCL_E_PTR;
    }
    HCCL_VM_INFO("{}: commId={}, opInfo={:p}, size={}", __func__, commId, opInfo, size);
    return HCCL_SUCCESS;
}

/**
 * @brief 查询是否支持注册 task 异常算子信息解析函数。
 *
 * 对应 hcomm 仓 HcommIsSupport 系列接口，由 dlsym 框架在运行期动态解析。
 *
 * 桩实现始终返回 true，表示支持 HcommRegOpTaskException。
 *
 * @return bool 始终返回 true。
 */
bool HcommIsSupportHcommRegOpTaskException()
{
    HCCL_VM_INFO("{}: stub returns true", __func__);
    return true;
}

/**
 * @brief 注册 task 异常算子信息解析回调（设备侧接口）。
 *
 * 注册一个回调函数，在 task 异常时解析算子信息并输出字符数组。
 * 对应 hcomm 仓同名接口。
 *
 * 桩实现仅记录调用日志并返回成功，不实际注册回调。
 *
 * @param commId 输入：通信域标识字符串。
 * @param callback 输入：解析算子信息并输出字符数组的回调函数指针。
 * @return HcclResult 成功返回 HCCL_SUCCESS，commId 为空返回 HCCL_E_PTR。
 */
HcclResult HcommRegOpTaskException(const char* commId, HcommGetOpInfoCallback callback)
{
    if (commId == nullptr) {
        HCCL_VM_ERROR("{}: commId is nullptr", __func__);
        return HCCL_E_PTR;
    }
    HCCL_VM_INFO("{}: commId={}, callback={:p}", __func__, commId, reinterpret_cast<void*>(callback));
    return HCCL_SUCCESS;
}

/**
 * @brief 按通信域 ID 注册 DFX 算子信息（设备侧接口）。
 *
 * 将算子执行过程中的诊断（DFX）信息注册到指定通信域上下文。
 * 对应 hccl 仓 hcomm_device_profiling_dl.h 同名接口。
 *
 * 桩实现仅记录调用日志并返回成功。
 *
 * @param commId 输入：通信域标识字符串。
 * @param hcclDfxOpInfo 输入：指向 DFX 算子信息结构体的指针。
 * @return HcclResult 成功返回 HCCL_SUCCESS，commId 为空返回 HCCL_E_PTR。
 */
HcclResult HcclDfxRegOpInfoByCommId(char* commId, void* hcclDfxOpInfo)
{
    if (commId == nullptr) {
        HCCL_VM_ERROR("{}: commId is nullptr", __func__);
        return HCCL_E_PTR;
    }
    HCCL_VM_INFO("{}: commId={}, hcclDfxOpInfo={:p}", __func__, commId, hcclDfxOpInfo);
    return HCCL_SUCCESS;
}

/**
 * @brief 上报 device 算子执行事件（设备侧 Profiling 接口）。
 *
 * 对应 hcomm 仓 hcomm_diag.h 同名接口。
 *
 * 桩实现仅记录调用日志并返回成功。
 *
 * @param groupname 输入：设备算子所属组名。
 * @return HcclResult 始终返回 HCCL_SUCCESS。
 */
HcclResult HcommProfilingReportDeviceOp(const char* groupname)
{
    HCCL_VM_INFO("{}: groupname={}", __func__, groupname ? groupname : "(null)");
    return HCCL_SUCCESS;
}

/**
 * @brief 上报内核启动任务事件（设备侧 Profiling 接口）。
 *
 * 对应 hcomm 仓 hcomm_diag.h 同名接口。
 *
 * 桩实现仅记录调用日志并返回成功。
 *
 * @param thread 输入：线程上下文。
 * @param groupname 输入：算子所属组名。
 * @return HcclResult 始终返回 HCCL_SUCCESS。
 */
HcclResult HcommProfilingReportKernelStartTask(uint64_t thread, const char* groupname)
{
    HCCL_VM_INFO("{}: thread={:d}, groupname={}", __func__, thread, groupname ? groupname : "(null)");
    return HCCL_SUCCESS;
}

/**
 * @brief 上报内核结束任务事件（设备侧 Profiling 接口）。
 *
 * 对应 hcomm 仓 hcomm_diag.h 同名接口。
 *
 * 桩实现仅记录调用日志并返回成功。
 *
 * @param thread 输入：线程上下文。
 * @param groupname 输入：算子所属组名。
 * @return HcclResult 始终返回 HCCL_SUCCESS。
 */
HcclResult HcommProfilingReportKernelEndTask(uint64_t thread, const char* groupname)
{
    HCCL_VM_INFO("{}: thread={:d}, groupname={}", __func__, thread, groupname ? groupname : "(null)");
    return HCCL_SUCCESS;
}

/**
 * @brief 初始化设备侧 Profiling（设备侧 Profiling 接口）。
 *
 * 注册参与 Profiling 的线程集合，为后续上报建立上下文。
 * 对应 hccl 仓 hcomm_device_profiling_dl.h 同名接口。
 *
 * 桩实现仅记录调用日志并返回成功。
 *
 * @param threads 输入：线程句柄数组。
 * @param threadNum 输入：线程数量。
 * @return HcclResult 始终返回 HCCL_SUCCESS。
 */
HcclResult HcommProfilingInit(ThreadHandle* threads, uint32_t threadNum)
{
    HCCL_VM_INFO(
        "{}: threads={:p}, threadNum={:d}", __func__, threads ? static_cast<void*>(threads) : nullptr, threadNum);
    return HCCL_SUCCESS;
}

/**
 * @brief 结束设备侧 Profiling（设备侧 Profiling 接口）。
 *
 * 注销 Profiling 线程集合，释放上下文资源。
 * 对应 hccl 仓 hcomm_device_profiling_dl.h 同名接口。
 *
 * 桩实现仅记录调用日志并返回成功。
 *
 * @param threads 输入：线程句柄数组。
 * @param threadNum 输入：线程数量。
 * @return HcclResult 始终返回 HCCL_SUCCESS。
 */
HcclResult HcommProfilingEnd(ThreadHandle* threads, uint32_t threadNum)
{
    HCCL_VM_INFO(
        "{}: threads={:p}, threadNum={:d}", __func__, threads ? static_cast<void*>(threads) : nullptr, threadNum);
    return HCCL_SUCCESS;
}

/**
 * @brief 上报主流和首个任务事件（设备侧 Profiling 接口）。
 *
 * 对应 hccl 仓 hcomm_device_profiling_dl.h 同名接口。
 *
 * 桩实现仅记录调用日志并返回成功。
 *
 * @param thread 输入：线程句柄。
 * @return HcclResult 始终返回 HCCL_SUCCESS。
 */
HcclResult HcommProfilingReportMainStreamAndFirstTask(ThreadHandle thread)
{
    HCCL_VM_INFO("{}: thread={:d}", __func__, thread);
    return HCCL_SUCCESS;
}

/**
 * @brief 上报主流和末个任务事件（设备侧 Profiling 接口）。
 *
 * 对应 hccl 仓 hcomm_device_profiling_dl.h 同名接口。
 *
 * 桩实现仅记录调用日志并返回成功。
 *
 * @param thread 输入：线程句柄。
 * @return HcclResult 始终返回 HCCL_SUCCESS。
 */
HcclResult HcommProfilingReportMainStreamAndLastTask(ThreadHandle thread)
{
    HCCL_VM_INFO("{}: thread={:d}", __func__, thread);
    return HCCL_SUCCESS;
}

/**
 * @brief 上报设备侧 HCCL 算子 Profiling 信息（设备侧 Profiling 接口）。
 *
 * 对应 hccl 仓 hcomm_device_profiling_dl.h 同名接口。
 *
 * 桩实现仅记录调用日志并返回成功，不访问 profInfo 字段。
 *
 * @param profInfo 输入：算子 Profiling 信息结构体（按值传递）。
 * @return HcclResult 始终返回 HCCL_SUCCESS。
 */
HcclResult HcommProfilingReportDeviceHcclOpInfo(HcomProInfoTmp profInfo)
{
    (void)profInfo;
    HCCL_VM_INFO("{}: stub returns SUCCESS", __func__);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif
