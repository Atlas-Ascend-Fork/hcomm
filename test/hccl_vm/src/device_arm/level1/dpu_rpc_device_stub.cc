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
 * Description: HostDPU 北向劫持（level1）设备侧 RPC 打桩函数。
 *              AICPU 展开模式下，设备子进程中的 DPU 模板 KernelRun 经
 *              HcommSendRequest / HcommWaitResponse 与 Host 侧 DPU 执行器
 *              完成 RPC 交互。本桩复刻真实 aicpu_ts_sync_data_c_adpt.cc 的
 *              共享内存 flag 协议（写序 data→msgId→msgTag→屏障→flag），
 *              并记录设备↔DPU 交互流（SetDpuStreamId → DpuDeviceInfo 表，
 *              与南向 ibv_dpu_stub 同源，供 Host 侧 DPU 任务归属）。
 *              协议常量本地镜像，与 Host 侧 dpu_level1_common.h 锚定同一布局。
 * Create: 2026-10-09
 */

#define HCCL_VM_MODULE "DPU_RPC_DEV"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>

#include "hccl/hccl_types.h"
#include "hccl_device_pub.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/sim_models.h"
#include "sim_log.h"

/* ThreadHandle 在 CANN hcomm_primitives.h 中定义为 uint64_t，本地镜像避免重依赖。 */
typedef uint64_t ThreadHandle;

namespace {

/*==================== 协议常量（本地镜像，锚定真实布局） ====================*/

/*
 * NPU -> DPU 请求半区布局（真实 aicpu_ts_sync_data_c_adpt.cc:71-121）：
 *   flag(1B) | msgTag(256B) | msgId(4B) | timeout(4B) | dataSize(8B) | data
 * DPU -> NPU 响应半区布局（level1 扩展，复用 timeout 槽位携带 retCode）：
 *   flag(1B) | ... | msgId(4B @257) | retCode(4B @261)
 */
constexpr uint64_t DPU_REQ_FLAG_OFF = 0;
constexpr uint64_t DPU_REQ_MSGTAG_OFF = 1;
constexpr uint64_t DPU_REQ_MSGID_OFF = 257;
constexpr uint64_t DPU_REQ_TIMEOUT_OFF = 261;
constexpr uint64_t DPU_REQ_DATASIZE_OFF = 265;
constexpr uint64_t DPU_REQ_DATA_OFF = 273;
constexpr uint64_t DPU_RSP_RETCODE_OFF = 261;

constexpr uint32_t DPU_MSG_TAG_MAX_LEN = 256;          /**< msgTag 定长空间（同 MSG_TAG_SIZE_BYTE） */
constexpr uint64_t DPU_TASK_PAYLOAD_MAX = 64UL * 1024; /**< payload 上限（与 host 侧一致） */
constexpr uint32_t DPU_RPC_WAIT_TIMEOUT_S = 1836;      /**< WaitResponse 超时（同 DPU 执行超时默认值） */

constexpr uint8_t DPU_FLAG_UNSET = 0;
constexpr uint8_t DPU_FLAG_SET = 1;

/* 与 Host 侧 dpu_level1_common.h 的布局一致性由两侧 static_assert 共同守护。 */
static_assert(DPU_REQ_MSGTAG_OFF + DPU_MSG_TAG_MAX_LEN == DPU_REQ_MSGID_OFF, "request layout drift");
static_assert(DPU_REQ_MSGID_OFF + sizeof(uint32_t) == DPU_REQ_TIMEOUT_OFF, "request layout drift");
static_assert(DPU_RSP_RETCODE_OFF == DPU_REQ_TIMEOUT_OFF, "response retCode must reuse timeout slot");

std::atomic<uint32_t> g_dpuMsgId{0}; /**< 请求 msgId 自增源（进程内唯一即可） */

/**
 * @brief 按真实写序构造请求消息并置位 flag（data→msgId→msgTag→屏障→flag=1）。
 * @param base    输入：请求半区基址（本进程可解引用地址）。
 * @param msgTag  输入：任务标识字符串。
 * @param src     输入：payload 源地址。
 * @param sizeByte 输入：payload 字节数。
 * @param msgId   输入：本次请求的消息 id。
 */
void WriteRequestMessage(uint8_t* base, const char* msgTag, const void* src, size_t sizeByte, uint32_t msgId)
{
    // 1. dataSize + data（尾部先写，保证 flag 置位时全部内容完整）
    const uint64_t dataSize = static_cast<uint64_t>(sizeByte);
    std::memcpy(base + DPU_REQ_DATASIZE_OFF, &dataSize, sizeof(dataSize));
    if (sizeByte > 0) {
        std::memcpy(base + DPU_REQ_DATA_OFF, src, sizeByte);
    }
    // 2. msgId
    std::memcpy(base + DPU_REQ_MSGID_OFF, &msgId, sizeof(msgId));
    // 3. msgTag（定长 256B，不足补零）
    std::memset(base + DPU_REQ_MSGTAG_OFF, 0, DPU_MSG_TAG_MAX_LEN);
    std::memcpy(base + DPU_REQ_MSGTAG_OFF, msgTag, std::strlen(msgTag) + 1);
    // 4. 屏障后置 flag（与真实 dmb sy 同序，QEMU 正确翻译）
#if defined(__aarch64__) || defined(__arm__)
    asm volatile("dmb sy" ::: "memory");
#else
    std::atomic_thread_fence(std::memory_order_release);
#endif
    *(base + DPU_REQ_FLAG_OFF) = DPU_FLAG_SET;
}

/**
 * @brief 幂等记录设备↔DPU 交互流 id 到 DpuDeviceInfo 表（仅变化时写库）。
 * @param streamId 输入：DPU 任务归属流（AICPU 线程的 streamId）。
 * @note DPU 模板在每次 RPC 前必调 HcommThreadSynchronize，由该处捕获线程流后
 *       经本函数写库；level1 下不依赖 level2 HAL 的 halSqCqQuery 路径。
 */
void RecordDpuStreamIdOnce(uint32_t streamId)
{
    static thread_local uint32_t s_lastRecorded = 0;
    if (streamId == 0 || streamId == s_lastRecorded) {
        return;
    }
    SetDpuStreamId(GetCurDeviceKey(), static_cast<uint64_t>(streamId));
    s_lastRecorded = streamId;
    HCCL_VM_INFO("{}: deviceId={:d}, streamId={:d}", __func__, GetCurDeviceKey(), streamId);
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 向 Host 侧 DPU 执行器发送 RPC 请求（NPU->DPU 共享内存协议）。
 *
 * 复刻真实 aicpu_ts_sync_data_c_adpt.cc 的写序：
 * data→msgId→msgTag→dmb sy→flag=1。执行器轮询到 flag 后按 msgTag 分发回调。
 *
 * @param handle   输入：请求半区的 device 映射地址（HcclDevMemAcquire 返回的 npu2DpuShmemPtr）。
 * @param msgTag   输入：任务标识（hccl 算法实例 algTag）。
 * @param src      输入：payload（序列化后的 DPURunInfo）。
 * @param sizeByte 输入：payload 字节数（≤64KB）。
 * @param msgId    输出：本次请求分配的消息 id。
 * @return int32_t 成功返回 0；参数非法/地址无法还原/超出 payload 上限返回对应错误码。
 */
int32_t HcommSendRequest(uint64_t handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId)
{
    if (handle == 0 || msgTag == nullptr || src == nullptr || msgId == nullptr) {
        HCCL_VM_ERROR(
            "{}: handle/msgTag/src/msgId invalid, handle={:#x}, msgTag={:p}, src={:p}, msgId={:p}", __func__, handle,
            static_cast<const void*>(msgTag), src, static_cast<void*>(msgId));
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    if (std::strlen(msgTag) >= DPU_MSG_TAG_MAX_LEN) {
        HCCL_VM_ERROR("{}: msgTag length {:d} exceeds max {:d}", __func__, std::strlen(msgTag), DPU_MSG_TAG_MAX_LEN);
        return static_cast<int32_t>(HCCL_E_PARA);
    }
    if (sizeByte == 0 || sizeByte > DPU_TASK_PAYLOAD_MAX) {
        HCCL_VM_ERROR("{}: sizeByte {:d} out of range (0, {:d}]", __func__, sizeByte, DPU_TASK_PAYLOAD_MAX);
        return static_cast<int32_t>(HCCL_E_PARA);
    }

    // 【核心步骤 1】device 映射地址还原为本进程（QEMU）可解引用地址
    const uint64_t localBase = GetDevMapperAddrByDevAddr(handle);
    if (localBase == 0) {
        HCCL_VM_ERROR("{}: resolve dev mapped addr failed, handle={:#x}", __func__, handle);
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    // 【核心步骤 2】按真实写序构造请求并置位 flag
    const uint32_t curMsgId = g_dpuMsgId.fetch_add(1, std::memory_order_relaxed);
    WriteRequestMessage(reinterpret_cast<uint8_t*>(localBase), msgTag, src, sizeByte, curMsgId);
    *msgId = curMsgId;
    HCCL_VM_INFO("{} success, msgTag={}, msgId={:d}, sizeByte={:d}", __func__, msgTag, curMsgId, sizeByte);
    return 0;
}

/**
 * @brief 等待 Host 侧 DPU 执行器的 RPC 响应（轮询 DPU->NPU 响应半区 flag）。
 *
 * @param handle   输入：响应半区的 device 映射地址（dpu2NpuShmemPtr）。
 * @param dst      输入：响应数据接收缓冲（当前协议无响应数据，sizeByte>0 不支持）。
 * @param sizeByte 输入：期望读取的响应数据字节数。
 * @param msgId    输出：响应携带的消息 id（调用方与 sendMsgId 比对）。
 * @return int32_t 成功返回 0；执行器上报错误时透传该错误码；超时返回 HCCL_E_TIMEOUT。
 */
int32_t HcommWaitResponse(uint64_t handle, void* dst, size_t sizeByte, uint32_t* msgId)
{
    if (handle == 0 || msgId == nullptr) {
        HCCL_VM_ERROR(
            "{}: handle or msgId invalid, handle={:#x}, msgId={:p}", __func__, handle, static_cast<void*>(msgId));
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    if (sizeByte > 0) {
        // 当前响应协议仅携带 msgId/retCode；hccl 调用点 sizeByte 恒为 0
        HCCL_VM_ERROR("{}: response data not supported, sizeByte={:d}", __func__, sizeByte);
        return static_cast<int32_t>(HCCL_E_NOT_SUPPORT);
    }
    (void)dst;

    const uint64_t localRsp = GetDevMapperAddrByDevAddr(handle);
    if (localRsp == 0) {
        HCCL_VM_ERROR("{}: resolve dev mapped addr failed, handle={:#x}", __func__, handle);
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }
    volatile uint8_t* rsp = reinterpret_cast<volatile uint8_t*>(localRsp);

    // 【核心步骤 1】轮询响应 flag（带超时守卫）
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(DPU_RPC_WAIT_TIMEOUT_S);
    while (*(rsp + DPU_REQ_FLAG_OFF) != DPU_FLAG_SET) {
        if (std::chrono::steady_clock::now() > deadline) {
            HCCL_VM_ERROR("{}: wait response timeout({:d}s), handle={:#x}", __func__, DPU_RPC_WAIT_TIMEOUT_S, handle);
            return static_cast<int32_t>(HCCL_E_TIMEOUT);
        }
        std::this_thread::yield();
    }
    std::atomic_thread_fence(std::memory_order_acquire);

    // 【核心步骤 2】读 msgId / retCode，随后清 flag（真实同款清零）
    uint32_t rspMsgId = 0;
    int32_t retCode = 0;
    std::memcpy(&rspMsgId, const_cast<const uint8_t*>(rsp + DPU_REQ_MSGID_OFF), sizeof(rspMsgId));
    std::memcpy(&retCode, const_cast<const uint8_t*>(rsp + DPU_RSP_RETCODE_OFF), sizeof(retCode));
#if defined(__aarch64__) || defined(__arm__)
    asm volatile("dmb sy" ::: "memory");
#else
    std::atomic_thread_fence(std::memory_order_release);
#endif
    *(rsp + DPU_REQ_FLAG_OFF) = DPU_FLAG_UNSET;

    *msgId = rspMsgId;
    if (retCode != 0) {
        HCCL_VM_ERROR("{}: dpu executor returned error, msgId={:d}, retCode={:d}", __func__, rspMsgId, retCode);
        return retCode;
    }
    HCCL_VM_INFO("{} success, msgId={:d}", __func__, rspMsgId);
    return 0;
}

/**
 * @brief 等待线程任务队列排空（设备侧 level1，no-op）。
 *
 * DPU 模板在每次 HcommSendRequest 前必调本接口（真实语义为轮询 RTSQ head==tail
 * 冲刷在途任务），是记录设备↔DPU 交互流的确定性锚点：本桩在此捕获该线程的
 * streamId 并写入 DpuDeviceInfo 表（供 Host 侧 DPU 任务归属），VM 无真实队列，
 * 直接返回成功。
 *
 * @param thread 输入：线程句柄（HcclThread 表主键）。
 * @return int32_t 成功返回 0。
 */
int32_t HcommThreadSynchronize(ThreadHandle thread)
{
    // 记录设备↔DPU 交互流（level1 下与设备侧其余任务同源的线程流）
    const auto threadRec = sim::runtime::Db::GetById<sim::runtime::HcclThread>(thread);
    if (threadRec.ok()) {
        RecordDpuStreamIdOnce(static_cast<uint32_t>(threadRec->streamId));
    } else {
        HCCL_VM_WARN("{}: thread {:d} not found, skip stream record", __func__, thread);
    }
    HCCL_VM_INFO("{}: thread={:d}, no-op", __func__, thread);
    return 0;
}

/**
 * @brief 等待线程退出（设备侧 level1，no-op）。
 * @param thread  输入：线程句柄（未使用）。
 * @param timeout 输入：超时时间（未使用）。
 * @return int32_t 成功返回 0。
 */
int32_t HcommThreadJoin(ThreadHandle thread, uint32_t timeout)
{
    (void)timeout;
    HCCL_VM_INFO("{}: thread={:d}, no-op", __func__, thread);
    return 0;
}

#ifdef __cplusplus
}
#endif
