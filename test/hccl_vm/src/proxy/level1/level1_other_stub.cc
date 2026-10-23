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
 * Description: 其他桩函数（北向劫持）——不属于通信域初始化、控制面、数据面、公共结构的接口
 * Create: 2026-07-25
 */

#define HCCL_VM_MODULE "OTHER_STUB"

#include <cstdint>
#include <cstring>

#include "level1_proxy_common.h"
#include "sim_log.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 查询 HCOMM 是否支持 HcclChannelDestroy。
 *
 * HCCL 用该能力探针门控 CCU 跨 rank 协商（CheckCcuResNegotiation /
 * CheckCcuParamAndFallback）：探针为 false 时各 rank 会独立回退，可能出现
 * 一侧回退到 AICPU、另一侧仍在 CCU 的 engine 不一致。HCCL-VM 的
 * HcclChannelDestroy 已由北向桩实现（dpu_control_stub.cc），这里固定返回 true。
 *
 * @return bool 恒返回 true。
 */
bool HcommIsSupportHcclChannelDestroy()
{
    HCCL_VM_INFO("{}: stub returns true", __func__);
    return true;
}

/**
 * @brief 将Thread资源导出到指定通信引擎上。
 *
 * 将传入的Thread句柄数组拷贝一份到exportedThreads中，供目标通信引擎直接引用。
 * 导出后在目标通信引擎直接使用，不需要额外的导入操作。
 *
 * @param comm 输入：通信域句柄，由HcclCommInitXxx系列接口返回。
 * @param threadNum 输入：待导出的Thread数量，不能为0。
 * @param threads 输入：源Thread句柄数组，长度为threadNum。
 * @param dstCommEngine 输入：目标通信引擎类型。
 * @param exportedThreads 输出：导出的Thread句柄数组，长度为threadNum，由调用者分配。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 导出成功
 * @retval HCCL_E_PTR comm、threads或exportedThreads为空
 * @retval HCCL_E_PARA threadNum为0
 */
HcclResult HcclThreadExportToCommEngine(
    HcclComm comm, uint32_t threadNum, const ThreadHandle* threads, CommEngine dstCommEngine,
    ThreadHandle* exportedThreads)
{
    if (comm == nullptr || threads == nullptr || exportedThreads == nullptr) {
        HCCL_VM_ERROR("{}: comm, threads or exportedThreads is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (threadNum == 0) {
        HCCL_VM_ERROR("{}: threadNum is 0", __func__);
        return HCCL_E_PARA;
    }
    for (uint32_t i = 0; i < threadNum; ++i) {
        exportedThreads[i] = threads[i];
    }

    uint64_t commId = reinterpret_cast<uint64_t>(comm);
    HCCL_VM_INFO(
        "{} success, commId={:d}, threadNum={:d}, dstCommEngine={:d}", __func__, commId, threadNum,
        static_cast<int>(dstCommEngine));
    return HCCL_SUCCESS;
}

/**
 * @brief 查找aicpu task cache, 判断tag是否已经缓存
 *
 * HCCL-VM 模拟环境不维护 aicpu task cache，桩函数始终返回 cache miss
 * （*isHit=false），强制调用方走完整算子展开路径。
 *
 * @param tag 输入：缓存标识符。
 * @param isHit 输出：是否cache hit，桩函数固定输出 false。
 *
 * @return HcommResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 查询完成（始终为 miss）
 * @retval HCCL_E_PTR tag或isHit指针为空
 */
int32_t HcommAicpuTsTaskCacheLookup(const char* tag, bool* isHit)
{
    if (tag == nullptr || isHit == nullptr) {
        HCCL_VM_ERROR("{}: tag or isHit is nullptr", __func__);
        return HCCL_E_PTR;
    }

    *isHit = false;
    HCCL_VM_INFO("{} success, tag='{}', isHit=false", __func__, tag);
    return HCCL_SUCCESS;
}

/**
 * @brief cache miss下, 算子展开前, 通知aicpu task cache开始缓存task
 *
 * HCCL-VM 模拟环境不维护 aicpu task cache，桩函数为空操作，直接返回成功。
 *
 * @param tag 输入：缓存标识符。
 * @param addrs 输入：内存基址信息数组。
 * @param sizes 输入：内存大小信息数组。
 * @param count 输入：内存信息数组长度。
 *
 * @return HcommResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 空操作成功
 * @retval HCCL_E_PTR tag、addrs或sizes指针为空
 */
HcommResult HcommAicpuTsTaskCacheStart(const char* tag, void** addrs, uint64_t* sizes, uint64_t count)
{
    if (tag == nullptr || addrs == nullptr || sizes == nullptr) {
        HCCL_VM_ERROR("{}: tag, addrs or sizes is nullptr", __func__);
        return HCCL_E_PTR;
    }

    HCCL_VM_INFO("{} success, tag='{}', count={:d}", __func__, tag, count);
    return HCCL_SUCCESS;
}

/**
 * @brief cache miss下, 算子展开后, 通知aicpu task cache停止缓存task
 *
 * HCCL-VM 模拟环境不维护 aicpu task cache，桩函数为空操作，直接返回成功。
 *
 * @param tag 输入：缓存标识符。
 *
 * @return HcommResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 空操作成功
 * @retval HCCL_E_PTR tag指针为空
 */
HcommResult HcommAicpuTsTaskCacheEnd(const char* tag)
{
    if (tag == nullptr) {
        HCCL_VM_ERROR("{}: tag is nullptr", __func__);
        return HCCL_E_PTR;
    }

    HCCL_VM_INFO("{} success, tag='{}'", __func__, tag);
    return HCCL_SUCCESS;
}

/**
 * @brief cache hit下, 刷新task并下发
 *
 * HCCL-VM 模拟环境不维护 aicpu task cache（Lookup 恒返回 miss），本桩函数
 * 理论上不会被调用，保留接口仅做参数校验并返回成功。
 *
 * @param tag 输入：缓存标识符。
 * @param addrs 输入：内存基址信息数组。
 * @param sizes 输入：内存大小信息数组。
 * @param count 输入：内存信息数组长度。
 *
 * @return HcommResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 空操作成功
 * @retval HCCL_E_PTR tag、addrs或sizes指针为空
 */
HcommResult HcommAicpuTsTaskCacheExecute(const char* tag, void** addrs, uint64_t* sizes, uint64_t count)
{
    if (tag == nullptr || addrs == nullptr || sizes == nullptr) {
        HCCL_VM_ERROR("{}: tag, addrs or sizes is nullptr", __func__);
        return HCCL_E_PTR;
    }

    HCCL_VM_INFO("{} success, tag='{}', count={:d}", __func__, tag, count);
    return HCCL_SUCCESS;
}

/**
 * @brief 清理aicpu task cache中指定tag对应的cache entry
 *
 * HCCL-VM 模拟环境不维护 aicpu task cache，桩函数为空操作，直接返回成功。
 *
 * @param tag 输入：缓存标识符。
 *
 * @return HcommResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 空操作成功
 * @retval HCCL_E_PTR tag指针为空
 */
HcommResult HcommAicpuTsTaskCacheClear(const char* tag)
{
    if (tag == nullptr) {
        HCCL_VM_ERROR("{}: tag is nullptr", __func__);
        return HCCL_E_PTR;
    }

    HCCL_VM_INFO("{} success, tag='{}'", __func__, tag);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif
