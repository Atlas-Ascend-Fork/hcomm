/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CPU_TASK_PIPE_H
#define CPU_TASK_PIPE_H

#include <atomic>
#include <cstdint>
#include "hccl_types.h"
#include "dpu_sync_layout.h"

namespace hccl {

// stream id 未设置时的取值：CQE 带上它是为了让异常面不去误命中真实的流
constexpr uint32_t DPU_DFX_STREAM_ID_INVALID = 0xFFFFFFFFU;

/**
 * @note SQ/CQ 队列的任务管道：一条通信域一个实例，同时持有 SQ（取任务）与 CQ（回执）两端。
 *       实例由host构造、kernel 侧只取裸指针，见 dpu_sync_layout.h 同址约定。
 *       SQ 生产侧是 device 侧 AICPU 编排，它到不了本类（本类不在 ccl_kernel 镜像里），故入队与
 *       taskId 分配归 CpuDeviceSqSubmitter（cpu_thread/cpu_device_sq_submitter.h）。
 */
class CpuTaskPipe {
public:
    CpuTaskPipe() = default;
    ~CpuTaskPipe() = default;
    // 队列语义要求单一实例、游标只有一份，禁止复制与移动
    CpuTaskPipe(const CpuTaskPipe&) = delete;
    CpuTaskPipe& operator=(const CpuTaskPipe&) = delete;
    CpuTaskPipe(CpuTaskPipe&&) = delete;
    CpuTaskPipe& operator=(CpuTaskPipe&&) = delete;

    /**
     * @brief 显式清零两页元数据，并写入 depth/elemSize
     * @param[in] layout 队列地址名片（host按半区偏移算好后传入）
     * @return HcclResult 成功返回 HCCL_SUCCESS
     * @note DpuShmemMgr::Alloc 只清零首字节，不显式初始化会让游标是垃圾值；幂等
     */
    HcclResult Init(const DpuSyncLayout& layout);

    /**
     * @brief kernel 侧主循环：阻塞轮询 SQ，直到 host 置 destroyFlag_ 或 SQE 执行失败
     * @return HcclResult 成功返回 HCCL_SUCCESS，执行失败返回该错误码
     */
    HcclResult Run();

    /**
     * @brief 写一条 CQ 回执并推进 CQ pi（kernel 侧）
     * @note CQ 满时丢弃本事件
     */
    HcclResult PushCqe(const DpuCqe& cqe);

    DpuSqMeta* SqMeta() const { return reinterpret_cast<DpuSqMeta*>(layout_.sqMeta); }
    DpuCqMeta* CqMeta() const { return reinterpret_cast<DpuCqMeta*>(layout_.cqMeta); }
    const DpuSyncLayout& GetLayout() const { return layout_; }
    bool IsInited() const { return inited_; }
    void SetNotifyInfo(uint32_t notifyNum, uint8_t* notifyAddr)
    {
        notifyNum_ = notifyNum;
        notifyAddr_ = notifyAddr;
    }
    void SetDevId(uint32_t devId) { devId_ = devId; }
    // 回执里带出的 stream id，供异常面按流定位；取自所属 CpuThread 的 rtStream
    void SetDfxStreamId(uint32_t streamId) { dfxStreamId_ = streamId; }

    // 终止与复位都由 host 控制面触发（实例 host 构造、kernel 只取裸指针，两侧同进程同地址空间），
    // 无需落共享内存；destroy 由 kernel 写回 false 作为「已退出」回执
    void RequestDestroy() { destroyFlag_.store(true, std::memory_order_release); }
    void RequestReset() { resetFlag_.store(true, std::memory_order_release); }
    // kernel 先退出后就不会再回写 destroyFlag_，此时按已退出回执，否则 host 的终止等待永远等不到
    bool IsDestroyAcked() const
    {
        return runExited_.load(std::memory_order_acquire) || !destroyFlag_.load(std::memory_order_acquire);
    }

private:
    HcclResult RunLoop();
    HcclResult ExecuteByOpKind(const DpuSqe& sqe);
    HcclResult ExecuteThreadNotifyWait(const DpuThreadNotifyWaitInfo& info);
    HcclResult ExecuteThreadNotifyRecordToDpu(const DpuThreadNotifyRecordToDpuInfo& info);
    HcclResult ExecuteThreadNotifyRecordToNpu(const DpuThreadNotifyRecordToNpuInfo& info);
    HcclResult ExecuteChannelNotifyWait(const DpuChannelNotifyWaitInfo& info);
    HcclResult ExecuteChannelNotifyRecord(const DpuChannelNotifyRecordInfo& info);
    HcclResult ExecuteChannelWriteWithNotifyNbi(const DpuChannelWriteWithNotifyInfo& info);
    HcclResult ExecuteChannelDrain(const DpuChannelDrainOrFenceInfo& info);
    HcclResult ExecuteChannelFence(const DpuChannelDrainOrFenceInfo& info);
    HcclResult ExecuteFence();
    HcclResult ExecuteLocalCopy(const DpuLocalCopyInfo& info);
    // 清空 SQ 元素区并重置 pi/ci（复位请求触发，不退出轮询）
    HcclResult ResetSq();
    DpuSqe* SqeAt(uint32_t slot) const { return reinterpret_cast<DpuSqe*>(layout_.sqElem + slot * layout_.elemSize); }
    DpuCqe* CqeAt(uint32_t slot) const { return reinterpret_cast<DpuCqe*>(layout_.cqElem + slot * layout_.elemSize); }
    // 游标与标志位由另一侧写：与 CpuDeviceSqSubmitter 的 LoadU32/StoreU32 同口径用 acquire/release 定序，
    // 值不会被缓存进寄存器，也免去 memcpy_s 的逐次参数校验
    static uint8_t ReadU8(const uint8_t* src);
    static void WriteU8(uint8_t* dst, uint8_t value);
    static uint32_t ReadU32(const uint32_t* src);
    static void WriteU32(uint32_t* dst, uint32_t value);

    DpuSyncLayout layout_{};
    uint32_t notifyNum_{0};
    uint8_t* notifyAddr_{nullptr};
    uint32_t devId_{0};
    uint32_t dfxStreamId_{DPU_DFX_STREAM_ID_INVALID};
    bool inited_{false};
    std::atomic<bool> destroyFlag_{false}; // host 写 true 请退出，kernel 写回 false 表示已退出
    std::atomic<bool> resetFlag_{false};   // host 写 true 请复位，kernel 复位后写回 false
    std::atomic<bool> runExited_{false};   // 主循环已退出：此后不会再有人回写 destroyFlag_
};

} // namespace hccl

#endif // CPU_TASK_PIPE_H
