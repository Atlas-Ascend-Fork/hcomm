/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CPU_DEVICE_SQ_SUBMITTER_H
#define CPU_DEVICE_SQ_SUBMITTER_H

#include <chrono>
#include <cstdint>
#include <thread>
#include "hccl_types.h"
#include "dpu_queue_copy.h"
#include "dpu_sync_layout.h"
#include "log.h"

namespace hccl {

// 环满退避节拍，与 CpuTaskPipe 的空队列退避同口径
constexpr uint32_t DPU_SQ_WAIT_INTERVAL_MS = 1U;
// 入队等环满的上界：超时返回而不是把编排线程无限挂住
constexpr uint32_t DPU_SQ_DEFAULT_TIMEOUT_MS = 10000U;
// 判满自旋的日志周期：SQ 满唯一退因是对端消费停滞，静默等待会丢掉现场
constexpr uint32_t DPU_SQ_FULL_LOG_INTERVAL_MS = 1000U;
// 满等待逐条日志内容相同（pi/ci/depth 不变），限前几条按 1s 留现场，之后改长周期心跳防刷屏
constexpr uint32_t DPU_SQ_FULL_LOG_MAX_NUM = 3U;
constexpr uint32_t DPU_SQ_FULL_LOG_BACKOFF_MS = 30000U;

/**
 * @note SQ 生产侧唯一入口：数据面把 DpuSqe 填好后交给本类入队，本类负责补 taskId、
 *       判环满、落格、发布 pi。使用者只有 CpuDeviceThread，编在 ccl_kernel 一个镜像里；
 *       做成纯头可回避「符号编在哪个 so」的问题 —— ccl_kernel 与 ccl_dpu/hcomm.so 一样没有
 *       -Wl,--no-undefined，镜像放错时仅在运行期暴露。本类只认 SQ 名片，不认线程与 CpuTaskPipe。
 */
class CpuDeviceSqSubmitter {
public:
    CpuDeviceSqSubmitter() = default;
    ~CpuDeviceSqSubmitter() = default;
    // 单一生产者：游标与 taskId 只有一份，禁止复制与移动
    CpuDeviceSqSubmitter(const CpuDeviceSqSubmitter&) = delete;
    CpuDeviceSqSubmitter& operator=(const CpuDeviceSqSubmitter&) = delete;
    CpuDeviceSqSubmitter(CpuDeviceSqSubmitter&&) = delete;
    CpuDeviceSqSubmitter& operator=(CpuDeviceSqSubmitter&&) = delete;

    /**
     * @brief 记下 SQ 名片，并取一次当前 pi 作为本地基准
     * @param[in] sqDesc SQ 半区描述（地址为 device 视角）
     * @return HcclResult 成功返回 HCCL_SUCCESS
     * @note 本侧是 pi 的唯一写者；ci 由对端推进，每次入队现读。pi 在每次 Submit 前重读，
     *       使消费侧的复位（ResetSq 清零共享 pi）能传导到生产侧
     */
    HcclResult Init(const QueueDescriptor& sqDesc);

    bool IsInited() const { return inited_; }
    const QueueDescriptor& GetSqDesc() const { return sqDesc_; }

    /**
     * @brief 通用入队：SQE 由数据面填好，本接口补 taskId、判环满、落格、发布 pi
     * @param[in,out] sqe 数据面组装好的 SQE；taskId 由本接口覆盖，调用方不必填
     * @param[out] taskId 本次分配的任务号，供编排侧留档（失败 CQE 也回抄同一 taskId）
     * @param[in] timeOutMs 环满等待上界；0 表示只探查一次，环满立即返回 HCCL_E_TIMEOUT
     * @return HcclResult 成功返回 HCCL_SUCCESS；环满等到超时返回 HCCL_E_TIMEOUT
     * @note taskId 分配权留在本接口：单调递增，失败 CQE 靠它归因到具体任务
     */
    HcclResult Submit(DpuSqe& sqe, uint32_t& taskId, uint32_t timeOutMs = DPU_SQ_DEFAULT_TIMEOUT_MS);

private:
    // 共享内存读写一律带内存序：缺了它，SQE 本体与 pi 的顺序只能靠运气
    static uint32_t LoadU32(uint64_t addr);
    static void StoreU32(uint64_t addr, uint32_t value);
    HcclResult WaitRoom(uint32_t timeOutMs) const;
    DpuSqe* SqeAt(uint32_t slot) const;

    QueueDescriptor sqDesc_{};
    uint32_t pi_{0};
    uint32_t nextTaskId_{0};
    bool inited_{false};
};

inline uint32_t CpuDeviceSqSubmitter::LoadU32(uint64_t addr)
{
    return __atomic_load_n(reinterpret_cast<const uint32_t*>(addr), __ATOMIC_ACQUIRE);
}

inline void CpuDeviceSqSubmitter::StoreU32(uint64_t addr, uint32_t value)
{
    __atomic_store_n(reinterpret_cast<uint32_t*>(addr), value, __ATOMIC_RELEASE);
}

inline DpuSqe* CpuDeviceSqSubmitter::SqeAt(uint32_t slot) const
{
    return reinterpret_cast<DpuSqe*>(sqDesc_.base + static_cast<uint64_t>(slot) * sqDesc_.elemSize);
}

inline HcclResult CpuDeviceSqSubmitter::Init(const QueueDescriptor& sqDesc)
{
    if (sqDesc.base == 0 || sqDesc.piAddr == 0 || sqDesc.ciAddr == 0 || sqDesc.elemSize != DPU_SQE_SIZE
        || sqDesc.depth == 0 || (sqDesc.depth & (sqDesc.depth - 1U)) != 0) {
        HCCL_ERROR(
            "[CpuDeviceSqSubmitter][%s] sq desc invalid, base[0x%llx] depth[%u] elemSize[%u]", __func__,
            static_cast<unsigned long long>(sqDesc.base), sqDesc.depth, sqDesc.elemSize);
        return HCCL_E_PARA;
    }
    sqDesc_ = sqDesc;
    pi_ = LoadU32(sqDesc_.piAddr);
    nextTaskId_ = 0U;
    inited_ = true;
    HCCL_INFO(
        "[CpuDeviceSqSubmitter][%s] sqElem[0x%llx] depth[%u] pi[%u]", __func__,
        static_cast<unsigned long long>(sqDesc_.base), sqDesc_.depth, pi_);
    return HCCL_SUCCESS;
}

inline HcclResult CpuDeviceSqSubmitter::WaitRoom(uint32_t timeOutMs) const
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeOutMs);
    auto lastLog = std::chrono::steady_clock::time_point::min();
    uint32_t fullLogCnt = 0U;
    for (;;) {
        // ci 只有对端会写：pi_ - ci 即当前在途条数，环深度牺牲一格让「满」与「空」判据同形
        const uint32_t ci = LoadU32(sqDesc_.ciAddr);
        if ((pi_ - ci) < (sqDesc_.depth - 1U)) {
            return HCCL_SUCCESS;
        }
        if (timeOutMs == 0U || std::chrono::steady_clock::now() >= deadline) {
            return HCCL_E_TIMEOUT;
        }
        const auto now = std::chrono::steady_clock::now();
        const uint32_t intervalMs
            = fullLogCnt < DPU_SQ_FULL_LOG_MAX_NUM ? DPU_SQ_FULL_LOG_INTERVAL_MS : DPU_SQ_FULL_LOG_BACKOFF_MS;
        if (lastLog == std::chrono::steady_clock::time_point::min()
            || now - lastLog >= std::chrono::milliseconds(intervalMs)) {
            HCCL_WARNING(
                "[CpuDeviceSqSubmitter][%s] SQ full waiting, pi[%u] ci[%u] pending[%u] depth[%u]", __func__, pi_, ci,
                pi_ - ci, sqDesc_.depth);
            lastLog = now;
            ++fullLogCnt;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(DPU_SQ_WAIT_INTERVAL_MS));
    }
}

inline HcclResult CpuDeviceSqSubmitter::Submit(DpuSqe& sqe, uint32_t& taskId, uint32_t timeOutMs)
{
    if (!inited_) {
        HCCL_ERROR("[CpuDeviceSqSubmitter][%s] submit before init", __func__);
        return HCCL_E_INTERNAL;
    }
    // 共享 pi 才是真实值：消费侧复位会把它清零，本地副本不能作为唯一依据
    pi_ = LoadU32(sqDesc_.piAddr);
    HcclResult ret = WaitRoom(timeOutMs);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CpuDeviceSqSubmitter][%s] SQ full, pi[%u] depth[%u]", __func__, pi_, sqDesc_.depth);
        return ret;
    }

    // 身份由本接口盖章：taskId 分配权单一，失败 CQE 靠它归因到具体任务
    taskId = nextTaskId_++;
    sqe.taskId = taskId;

    // 整格写全：落格即完整，消费侧取出元素不需要再补任何字段
    DpuSqe* slot = SqeAt(pi_ % sqDesc_.depth);
    if (!CopyQueueElem(*slot, sqe)) {
        HCCL_ERROR("[CpuDeviceSqSubmitter][%s] write SQE failed, taskId[%u]", __func__, taskId);
        return HCCL_E_INTERNAL;
    }
    // 顺序铁律：本体写全才发布 pi；少了这一步，消费侧可能读到半截元素
    pi_ += 1U;
    StoreU32(sqDesc_.piAddr, pi_);
    return HCCL_SUCCESS;
}

} // namespace hccl

#endif // CPU_DEVICE_SQ_SUBMITTER_H
