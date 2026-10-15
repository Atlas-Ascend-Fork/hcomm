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
 * Description: HostDPU 北向劫持（level1）数据面打桩函数。
 *              DPU 模板（DPUKernelRun）经 channel 原语完成 Host 网卡 RoCE 通信，
 *              本文件补齐既有 level1 未覆盖的调用面：
 *              - Nbi 非阻塞写读族：转调既有阻塞版桩（任务元数据语义一致，
 *                非阻塞语义在"记录不执行"模型下无差异）；
 *              - Drain / Fence 族：排空与保序（no-op——记录模型下无在途任务）；
 *              - DefaultTimeout / 无线程变体：以 thread=0 转调对应 OnThread 桩。
 *              DPU 调用 thread 恒为 0，任务归属由执行器回调期 TLS 上下文
 *              （dpu_level1_common.h DpuExecContext）与 DpuDeviceInfo 回退解析。
 * Create: 2026-10-09
 */

#define HCCL_VM_MODULE "DPU_DATA_STUB"

#include <cstdint>

#include "hccl/hccl_types.h"
#include "level1_proxy_common.h"
#include "sim_log.h"

#include "dpu_level1_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/*==================== 转调目标（既有阻塞版桩，data_comm/data_local_op_stub.cc） ====================*/

extern int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len);
extern int32_t HcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len);
extern int32_t HcommWriteWithNotifyOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx);
extern int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx);
extern int32_t
HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut);
extern int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeout);

/*==================== Nbi 非阻塞写读族（转调既有阻塞版） ====================*/

/**
 * @brief 通过 channel 向远端写数据并发送 notify（非阻塞变体，数据面接口）。
 *
 * HostDPU 模板数据面主写路径。任务元数据与阻塞版一致（MEM_CPY + NOTIFY_RECORD），
 * 非阻塞语义在"记录不执行"模型下无差异，直接转调 HcommWriteWithNotifyOnThread。
 *
 * @param thread          输入：线程句柄（DPU 场景恒为 0）。
 * @param channel         输入：通道句柄。
 * @param dst             输入：远端 device 内存地址。
 * @param src             输入：本地 device 内存地址。
 * @param len             输入：待拷贝字节数。
 * @param remoteNotifyIdx 输入：远端节点上的 notify 索引。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommWriteWithNotifyNbiOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    return HcommWriteWithNotifyOnThread(thread, channel, dst, src, len, remoteNotifyIdx);
}

/**
 * @brief 通过 channel 向远端写数据（非阻塞变体，数据面接口）。
 * @param thread  输入：线程句柄（DPU 场景恒为 0）。
 * @param channel 输入：通道句柄。
 * @param dst     输入：远端 device 内存地址。
 * @param src     输入：本地 device 内存地址。
 * @param len     输入：待拷贝字节数。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommWriteNbiOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    return HcommWriteOnThread(thread, channel, dst, src, len);
}

/**
 * @brief 通过 channel 向远端写数据（非阻塞、无线程变体）。
 * @param channel 输入：通道句柄。
 * @param dst     输入：远端 device 内存地址。
 * @param src     输入：本地 device 内存地址。
 * @param len     输入：待拷贝字节数。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommWriteNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    return HcommWriteOnThread(0, channel, dst, src, len);
}

/**
 * @brief 通过 channel 向远端写数据并发送 notify（非阻塞、无线程变体）。
 * @param channel         输入：通道句柄。
 * @param dst             输入：远端 device 内存地址。
 * @param src             输入：本地 device 内存地址。
 * @param len             输入：待拷贝字节数。
 * @param remoteNotifyIdx 输入：远端节点上的 notify 索引。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t
HcommWriteWithNotifyNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    return HcommWriteWithNotifyOnThread(0, channel, dst, src, len, remoteNotifyIdx);
}

/**
 * @brief 通过 channel 从远端读数据（非阻塞变体，数据面接口）。
 * @param thread  输入：线程句柄（DPU 场景恒为 0）。
 * @param channel 输入：通道句柄。
 * @param dst     输入：本地 device 内存地址（本端）。
 * @param src     输入：远端 device 内存地址（对端）。
 * @param len     输入：待读取字节数。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommReadNbiOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    return HcommReadOnThread(thread, channel, dst, src, len);
}

/**
 * @brief 通过 channel 从远端读数据（非阻塞、无线程变体）。
 * @param channel 输入：通道句柄。
 * @param dst     输入：本地 device 内存地址（本端）。
 * @param src     输入：远端 device 内存地址（对端）。
 * @param len     输入：待读取字节数。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommReadNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    return HcommReadOnThread(0, channel, dst, src, len);
}

/*==================== Drain / Fence 族（no-op） ====================*/

/**
 * @brief 等待 channel 上所有在途任务完成（排空）。
 *
 * 真实实现轮询 QP 完成计数；VM "记录不执行"模型下没有在途 WQE，直接成功。
 *
 * @param thread  输入：线程句柄（DPU 场景恒为 0）。
 * @param channel 输入：通道句柄。
 * @return int32_t 成功返回 0。
 */
int32_t HcommChannelDrainOnThread(ThreadHandle thread, ChannelHandle channel)
{
    HCCL_VM_INFO("{}: thread={:d}, channel={:d}, no-op", __func__, thread, channel);
    return 0;
}

/**
 * @brief 通道级 fence（保序，no-op）。
 * @param thread  输入：线程句柄（DPU 场景恒为 0）。
 * @param channel 输入：通道句柄。
 * @return int32_t 成功返回 0。
 */
int32_t HcommChannelFenceOnThread(ThreadHandle thread, ChannelHandle channel)
{
    HCCL_VM_INFO("{}: thread={:d}, channel={:d}, no-op", __func__, thread, channel);
    return 0;
}

/**
 * @brief 通道级 fence（无线程变体，no-op）。
 * @param channel 输入：通道句柄。
 * @return int32_t 成功返回 0。
 */
int32_t HcommChannelFence(ChannelHandle channel)
{
    HCCL_VM_INFO("{}: channel={:d}, no-op", __func__, channel);
    return 0;
}

/**
 * @brief 线程级 fence（全局 flush，no-op）。
 * @param thread 输入：线程句柄（DPU 场景恒为 0）。
 * @return int32_t 成功返回 0。
 */
int32_t HcommFenceOnThread(ThreadHandle thread)
{
    HCCL_VM_INFO("{}: thread={:d}, no-op", __func__, thread);
    return 0;
}

/**
 * @brief 全局 flush（等价 HcommFenceOnThread(0)，no-op）。
 * @return int32_t 成功返回 0。
 */
int32_t HcommFlush()
{
    HCCL_VM_INFO("{}: no-op", __func__);
    return 0;
}

/*==================== DefaultTimeout / 无线程变体（转调 OnThread 桩） ====================*/

/**
 * @brief 等待 channel 远端 notify（默认超时版本）。
 * @param thread         输入：线程句柄（DPU 场景恒为 0）。
 * @param channel        输入：通道句柄。
 * @param localNotifyIdx 输入：本端 notify 索引。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t
HcommChannelNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx)
{
    return HcommChannelNotifyWaitOnThread(thread, channel, localNotifyIdx, 0);
}

/**
 * @brief 等待 channel 远端 notify（默认超时、无线程变体）。
 * @param channel        输入：通道句柄。
 * @param localNotifyIdx 输入：本端 notify 索引。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommChannelNotifyWaitWithDefaultTimeout(ChannelHandle channel, uint32_t localNotifyIdx)
{
    return HcommChannelNotifyWaitOnThread(0, channel, localNotifyIdx, 0);
}

/**
 * @brief 通过 channel 向远端发送 notify（无线程变体）。
 * @param channel         输入：通道句柄。
 * @param remoteNotifyIdx 输入：远端节点上的 notify 索引。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommChannelNotifyRecord(ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    return HcommChannelNotifyRecordOnThread(0, channel, remoteNotifyIdx);
}

/**
 * @brief 等待 channel 远端 notify（无线程变体）。
 * @param channel        输入：通道句柄。
 * @param localNotifyIdx 输入：本端 notify 索引。
 * @param timeOut        输入：超时时间（秒，桩中不使用）。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommChannelNotifyWait(ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
{
    return HcommChannelNotifyWaitOnThread(0, channel, localNotifyIdx, timeOut);
}

/**
 * @brief 等待线程 notify（默认超时版本）。
 * @param thread    输入：线程句柄（DPU 场景恒为 0）。
 * @param notifyIdx 输入：notify 索引。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommThreadNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle thread, uint32_t notifyIdx)
{
    return HcommThreadNotifyWaitOnThread(thread, notifyIdx, 0);
}

#ifdef __cplusplus
}
#endif
