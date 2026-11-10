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
 * Description: 设备侧其他操作打桩函数（level1 劫持）
 *              - HcommIsSupport* 系列能力探针（11 个）
 *              - HcclSetNotifyWaitTimeOut / HcclThreadResAcquireTimeOut /
 *                HcclThreadNotifyWaitOnThreadDefault
 *                Hccl 包装函数（委托已有 Hcomm 前缀桩）
 *              - HcommAicpuTsTaskCacheLookup / HcommAicpuTsTaskCacheStart /
 *                HcommAicpuTsTaskCacheEnd   / HcommAicpuTsTaskCacheExecute /
 *                HcommAicpuTsTaskCacheClear
 *                AICPU 任务缓存接口。HCCL-VM 模拟设备侧不维护 aicpu task cache，
 *                Lookup 始终返回 cache miss（*isHit=false），其余接口为空操作。
 * Create: 2026-08-13
 */

#define HCCL_VM_MODULE "DEV_L1_OTHER"

#include <cstdint>
#include <cstddef>

#include "hccl/hccl_types.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/table_access.h"
#include "sim_log.h"

/* HcommResult 在 CANN 公共头中通常为 int32_t 同型的枚举/别名，
 * 此处本地 typedef 以避免引入 hcomm_primitives.h 对设备侧 ABI 的额外依赖。 */
typedef int32_t HcommResult;

/* ThreadHandle 在 CANN hcomm_primitives.h 中定义为 uint64_t */
typedef uint64_t ThreadHandle;

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// HcommIsSupport* 能力探针系列
//
// 在真实 HCOMM 库中，这些函数由 DEFINE_WEAK_FUNC 宏在 *_dl.cc 中自动生成，
// 通过 dlsym 判断底层 Hcomm* 函数是否存在后设置 g_*Supported 标志。
// HCCL-VM 设备侧没有真实 HCOMM 库，weak 默认值始终返回 false，
// 导致 kernel_launch.cc 中所有依赖 IsSupport 门控的功能分支被跳过。
// 桩实现统一返回 true，使上层逻辑正常进入对应功能路径，
// 由底层 Hcomm* 设备侧桩接管实际行为。
// ============================================================================

bool HcommIsSupportHcclCommGetStatus()
{
    HCCL_VM_INFO("{}: stub returns true", __func__);
    return true;
}

/** @brief 能力探针：HcommThreadResAcquireTimeOut 存在，返回 true。 */
bool HcommIsSupportHcommThreadResAcquireTimeOut()
{
    HCCL_VM_INFO("{}: stub returns true", __func__);
    return true;
}

/** @brief 能力探针：HcommSetNotifyWaitTimeOut 存在，返回 true。 */
bool HcommIsSupportHcommSetNotifyWaitTimeOut()
{
    HCCL_VM_INFO("{}: stub returns true", __func__);
    return true;
}

/** @brief 能力探针：HcommThreadNotifyWaitOnThreadWithDefaultTimeout 存在，返回 true。 */
bool HcommIsSupportHcommThreadNotifyWaitOnThreadWithDefaultTimeout()
{
    HCCL_VM_INFO("{}: stub returns true", __func__);
    return true;
}

/** @brief 能力探针：HcommAicpuTsTaskCacheLookup 存在，返回 true。 */
bool HcommIsSupportHcommAicpuTsTaskCacheLookup()
{
    HCCL_VM_INFO("{}: stub returns true", __func__);
    return true;
}

/** @brief 能力探针：HcommAicpuTsTaskCacheStart 存在，返回 true。 */
bool HcommIsSupportHcommAicpuTsTaskCacheStart()
{
    HCCL_VM_INFO("{}: stub returns true", __func__);
    return true;
}

/** @brief 能力探针：HcommAicpuTsTaskCacheEnd 存在，返回 true。 */
bool HcommIsSupportHcommAicpuTsTaskCacheEnd()
{
    HCCL_VM_INFO("{}: stub returns true", __func__);
    return true;
}

/** @brief 能力探针：HcommAicpuTsTaskCacheExecute 存在，返回 true。 */
bool HcommIsSupportHcommAicpuTsTaskCacheExecute()
{
    HCCL_VM_INFO("{}: stub returns true", __func__);
    return true;
}

/** @brief 能力探针：HcommAicpuTsTaskCacheClear 存在，返回 true。 */
bool HcommIsSupportHcommAicpuTsTaskCacheClear()
{
    HCCL_VM_INFO("{}: stub returns true", __func__);
    return true;
}

// ============================================================================
// Hccl* 包装函数系列
//
// 这些函数在 hcomm_primitives_dl.cc 中有原始定义，将 uint32_t timeout 转换
// 为对应 Hcomm* 函数的参数类型（float/uint32_t），或直接分派到底层 Hcomm* 函数。
// HCCL-VM 设备侧提供桩以覆盖 weak 默认实现（默认实现返回 HCCL_E_NOT_SUPPORT），
// 确保 kernel_launch.cc → OpOrchestrate 的执行流程不被阻断。
// ============================================================================

/**
 * @brief 设置算子展开过程中 RTSQ 等待超时时间（Hccl 包装）。
 *
 * 委托 data_op_device_stub.cc 中的 HcommThreadResAcquireTimeOut(float)。
 *
 * @param timeout 输入：超时时间（秒）。
 * @return HcclResult 接口成功返回 HCCL_SUCCESS，不支持返回 HCCL_E_NOT_SUPPORT。
 */
HcclResult HcclThreadResAcquireTimeOut(uint32_t timeout)
{
    int32_t HcommThreadResAcquireTimeOut(float);
    return static_cast<HcclResult>(HcommThreadResAcquireTimeOut(static_cast<float>(timeout)));
}

/**
 * @brief 设置 NotifyWait 超时时间（Hccl 包装）。
 *
 * 委托 data_op_device_stub.cc 中的 HcommSetNotifyWaitTimeOut(float)。
 *
 * @param timeout 输入：超时时间（秒）。
 * @return HcclResult 接口成功返回 HCCL_SUCCESS，不支持返回 HCCL_E_NOT_SUPPORT。
 */
HcclResult HcclSetNotifyWaitTimeOut(uint32_t timeout)
{
    int32_t HcommSetNotifyWaitTimeOut(float);
    return static_cast<HcclResult>(HcommSetNotifyWaitTimeOut(static_cast<float>(timeout)));
}

/**
 * @brief 主线程等待 host stream 通知（Hccl 包装/分发）。
 *
 * 当底层 HCOMM 支持 HcommSetNotifyWaitTimeOut 和
 * HcommThreadNotifyWaitOnThreadWithDefaultTimeout 时，
 * 使用默认超时版本；否则回退到带 fallbackTimeout 的版本。
 * HCCL-VM 桩始终走默认超时路径。
 *
 * @param thread 输入：线程句柄。
 * @param notifyIdx 输入：通知索引。
 * @param fallbackTimeout 输入：回退超时时间（秒），桩中不使用。
 * @return HcclResult 接口成功返回 HCCL_SUCCESS。
 */
HcclResult HcclThreadNotifyWaitOnThreadDefault(ThreadHandle thread, uint32_t notifyIdx, uint32_t fallbackTimeout)
{
    (void)fallbackTimeout;
    int32_t HcommThreadNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle, uint32_t);
    HCCL_VM_INFO("{}: thread={:d}, notifyIdx={:d}", __func__, thread, notifyIdx);
    return static_cast<HcclResult>(HcommThreadNotifyWaitOnThreadWithDefaultTimeout(thread, notifyIdx));
}

// ============================================================================
// HcommAicpuTsTaskCache* 系列
//
// HCCL-VM 模拟设备侧不维护 aicpu task cache。
// Lookup 始终返回 cache miss（*isHit=false），强制走完整算子展开路径。
// Start/End/Execute/Clear 为空操作，仅做参数校验并返回成功。
// ============================================================================

/**
 * @brief 查找 aicpu task cache，判断 tag 是否已经缓存（设备侧接口）。
 *
 * @param tag 输入：缓存标识符。
 * @param isHit 输出：是否 cache hit，桩函数固定输出 false。
 *
 * @return int32_t 接口成功返回 HCCL_SUCCESS，参数为空返回 HCCL_E_PTR。
 */
int32_t HcommAicpuTsTaskCacheLookup(const char* tag, bool* isHit)
{
    if (tag == nullptr || isHit == nullptr) {
        HCCL_VM_ERROR("{}: tag or isHit is nullptr", __func__);
        return static_cast<int32_t>(HCCL_E_PTR);
    }

    *isHit = false;
    HCCL_VM_INFO("{} success, tag='{}', isHit=false", __func__, tag);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

/**
 * @brief cache miss 下，算子展开前，通知 aicpu task cache 开始缓存 task（设备侧接口）。
 *
 * @param tag 输入：缓存标识符。
 * @param addrs 输入：内存基址信息数组。
 * @param sizes 输入：内存大小信息数组。
 * @param count 输入：内存信息数组长度。
 *
 * @return HcommResult 接口成功返回 HCCL_SUCCESS，参数为空返回 HCCL_E_PTR。
 */
HcommResult HcommAicpuTsTaskCacheStart(const char* tag, void** addrs, uint64_t* sizes, uint64_t count)
{
    if (tag == nullptr || addrs == nullptr || sizes == nullptr) {
        HCCL_VM_ERROR("{}: tag, addrs or sizes is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }

    HCCL_VM_INFO("{} success, tag='{}', count={:d}", __func__, tag, count);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief cache miss 下，算子展开后，通知 aicpu task cache 停止缓存 task（设备侧接口）。
 *
 * @param tag 输入：缓存标识符。
 *
 * @return HcommResult 接口成功返回 HCCL_SUCCESS，参数为空返回 HCCL_E_PTR。
 */
HcommResult HcommAicpuTsTaskCacheEnd(const char* tag)
{
    if (tag == nullptr) {
        HCCL_VM_ERROR("{}: tag is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }

    HCCL_VM_INFO("{} success, tag='{}'", __func__, tag);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief cache hit 下，刷新 task 并下发（设备侧接口）。
 *
 * @param tag 输入：缓存标识符。
 * @param addrs 输入：内存基址信息数组。
 * @param sizes 输入：内存大小信息数组。
 * @param count 输入：内存信息数组长度。
 *
 * @return HcommResult 接口成功返回 HCCL_SUCCESS，参数为空返回 HCCL_E_PTR。
 */
HcommResult HcommAicpuTsTaskCacheExecute(const char* tag, void** addrs, uint64_t* sizes, uint64_t count)
{
    if (tag == nullptr || addrs == nullptr || sizes == nullptr) {
        HCCL_VM_ERROR("{}: tag, addrs or sizes is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }

    HCCL_VM_INFO("{} success, tag='{}', count={:d}", __func__, tag, count);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief 清理 aicpu task cache 中指定 tag 对应的 cache entry（设备侧接口）。
 *
 * @param tag 输入：缓存标识符。
 *
 * @return HcommResult 接口成功返回 HCCL_SUCCESS，参数为空返回 HCCL_E_PTR。
 */
HcommResult HcommAicpuTsTaskCacheClear(const char* tag)
{
    if (tag == nullptr) {
        HCCL_VM_ERROR("{}: tag is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }

    HCCL_VM_INFO("{} success, tag='{}'", __func__, tag);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

// ============================================================================
// 对称内存窗口系列桩
//
// winHandle 的编码约定：host 侧 level1 桩（comm_domain_manage_stub.cc）将
// HcclCommSymWinRegister / HcclCommSymWinGet 返回的 winHandle 编码为通信域
// DB 主键 commId（小整数，非任何进程地址空间中的指针）。真实库中 kernel 会
// 将 winHandle 当 SymmetricWindow*/HcommWindow* 解引用，VM 中若放任符号
// 穿透到 libccl_kernel.so 真身，解引用 commId 即 SIGSEGV（AlltoAllV AICPU
// V2 执行器经 hccl_sym_win_dl 直接调 HcclSymWinGetRemoteAddr，绕过
// HcclCommSymWinGet 探针，无法靠"Get 失败回退非对称"兜底）。
//
// 因此设备侧 HcclSymWinGetRemoteAddr / HcclSymWinGetPeerPointer 按 commId
// 查 Communicator 行，返回 sym_win_addr + offset（跨 rank 同址，忽略
// peerRank，与 host 桩 HcclSymWinGetPeerPointer 语义一致）。
//
// HcclCommSymWinGet 维持返回 HCCL_E_INTERNAL：AllGather 等设备侧探针路径
// 检测失败后回退到非对称路径，行为不变。
//
// HcclCommSymWinRegister / HcclCommSymWinDeregister 是 host 侧操作，
// 设备侧不支持，返回 HCCL_E_NOT_SUPPORT。
// ============================================================================

/* HcclCommWindow 在 hccl_types.h 中定义为 void*，此处直接使用。 */

/**
 * @brief 按 commId 编码解析对称窗口地址（GetRemoteAddr / GetPeerPointer 共用）。
 *
 * host 侧 level1 桩将 winHandle 编码为 Communicator 表主键 commId。
 * 对称窗口跨 rank 虚拟地址同址，对端地址 = 本端窗口基址 + offset，
 * 与 host 桩 HcclSymWinGetPeerPointer 的返回语义一致。
 *
 * @param winHandle 输入：窗口句柄（VM 编码为 commId）。
 * @param offset    输入：窗口内字节偏移。
 * @param ptr       输出：sym_win_addr + offset。
 * @return HcclResult 成功返回 HCCL_SUCCESS；空参返回 HCCL_E_PTR，
 *         commId 无对应行或未注册窗口返回 HCCL_E_INTERNAL，
 *         偏移越界返回 HCCL_E_PARA。
 */
static HcclResult ResolveSymWinAddrByCommId(HcclCommSymWindow winHandle, size_t offset, void** ptr)
{
    if (winHandle == nullptr || ptr == nullptr) {
        HCCL_VM_ERROR("{}: winHandle or ptr is nullptr", __func__);
        return HCCL_E_PTR;
    }

    const uint64_t commId = reinterpret_cast<uint64_t>(winHandle);
    auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }
    if (optComm->sym_win_registered == 0 || optComm->sym_win_addr == 0 || optComm->sym_win_size == 0) {
        HCCL_VM_ERROR("{}: symmetric window not registered for comm {:d}", __func__, commId);
        return HCCL_E_INTERNAL;
    }
    if (offset >= optComm->sym_win_size) {
        HCCL_VM_ERROR(
            "{}: offset {:d} out of window size {:d}, comm {:d}", __func__, offset, optComm->sym_win_size, commId);
        return HCCL_E_PARA;
    }

    *ptr = reinterpret_cast<void*>(optComm->sym_win_addr + offset);
    HCCL_VM_INFO("{}: commId={:d}, offset={:d}, addr={:p}", __func__, commId, offset, *ptr);
    return HCCL_SUCCESS;
}

/**
 * @brief 查询已注册对称内存窗口的句柄和偏移量（设备侧接口）。
 *
 * 设备侧不提供窗口反查：调用方（AllGatherSupportSymmetricMemory）检测失败后
 * 回退到非对称路径。窗口地址解析走 HcclSymWinGetRemoteAddr /
 * HcclSymWinGetPeerPointer（host 已先完成窗口查询，句柄以 commId 编码传入）。
 *
 * @param comm      输入：通信域句柄（opaque void*）。
 * @param ptr       输入：候选指针。
 * @param size      输入：预留大小（设备侧忽略）。
 * @param winHandle 输出：对称内存窗口资源句柄。
 * @param offset    输出：ptr 在窗口内的偏移量。
 * @return HcclResult 设备侧始终返回 HCCL_E_INTERNAL。
 */
HcclResult HcclCommSymWinGet(HcclComm comm, void* ptr, size_t size, HcclCommSymWindow* winHandle, size_t* offset)
{
    if (comm == nullptr || ptr == nullptr || winHandle == nullptr || offset == nullptr) {
        HCCL_VM_ERROR("{}: comm, ptr, winHandle or offset is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (size == 0) {
        HCCL_VM_ERROR("{}: size is 0", __func__);
        return HCCL_E_PARA;
    }

    HCCL_VM_ERROR(
        "{}: symmetric memory window lookup not supported on device side, comm={:p}, ptr={:p}, size={:d}", __func__,
        comm, ptr, size);
    return HCCL_E_INTERNAL;
}

/**
 * @brief 获取对端 rank 在对称内存窗口中的指针（设备侧接口）。
 *
 * winHandle 为 host 桩返回的 commId 编码，按 DB 解析出
 * 窗口基址 + offset 返回（跨 rank 同址，忽略 peerRank）。
 *
 * @param winHandle 输入：对称内存窗口资源句柄（VM 编码为 commId）。
 * @param offset    输入：窗口内偏移量。
 * @param peerRank  输入：对端 rank 编号（对称同址，未使用）。
 * @param ptr       输出：对端 rank 在该偏移对应的地址指针。
 * @return HcclResult 成功返回 HCCL_SUCCESS，失败见 ResolveSymWinAddrByCommId。
 */
HcclResult HcclSymWinGetPeerPointer(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    (void)peerRank;
    return ResolveSymWinAddrByCommId(winHandle, offset, ptr);
}

/**
 * @brief 获取对端 rank 的 URMA 远端地址（设备侧接口）。
 *
 * AlltoAllV 等 AICPU V2 执行器在 device 进程内直接调用本接口
 * （经 hccl_sym_win_dl，真实实现会解引用 winHandle 指向的窗口结构体）。
 * VM 中 winHandle 是 host 桩的 commId 编码而非可解引用指针，
 * 本桩按 DB 解析，语义与 HcclSymWinGetPeerPointer 一致。
 *
 * @param winHandle 输入：对称内存窗口资源句柄（VM 编码为 commId）。
 * @param offset    输入：窗口内偏移量。
 * @param peerRank  输入：对端 rank 编号（对称同址，未使用）。
 * @param ptr       输出：对端 rank 在该偏移对应的远端地址。
 * @return HcclResult 成功返回 HCCL_SUCCESS，失败见 ResolveSymWinAddrByCommId。
 */
HcclResult HcclSymWinGetRemoteAddr(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    (void)peerRank;
    return ResolveSymWinAddrByCommId(winHandle, offset, ptr);
}

/**
 * @brief 注册对称内存窗口（设备侧接口，不支持）。
 *
 * 对称窗口的注册是 host 侧操作；设备侧不支持，直接返回 HCCL_E_NOT_SUPPORT。
 *
 * @param comm      输入：通信域句柄。
 * @param addr      输入：内存基址。
 * @param size      输入：窗口大小。
 * @param winHandle 输出：注册后的窗口句柄。
 * @param flag      输入：注册标志。
 * @return HcclResult 设备侧始终返回 HCCL_E_NOT_SUPPORT。
 */
HcclResult HcclCommSymWinRegister(HcclComm comm, void* addr, uint64_t size, HcclCommSymWindow* winHandle, uint32_t flag)
{
    HCCL_VM_WARN("{}: not supported on device side", __func__);
    return HCCL_E_NOT_SUPPORT;
}

/**
 * @brief 反注册对称内存窗口（设备侧接口，不支持）。
 *
 * @param winHandle 输入：窗口句柄。
 * @return HcclResult 设备侧始终返回 HCCL_E_NOT_SUPPORT。
 */
HcclResult HcclCommSymWinDeregister(HcclCommSymWindow winHandle)
{
    HCCL_VM_WARN("{}: not supported on device side", __func__);
    return HCCL_E_NOT_SUPPORT;
}

#ifdef __cplusplus
}
#endif
