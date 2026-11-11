/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CPU_THREAD_H
#define CPU_THREAD_H

#include <memory>
#include <mutex>
#include <string>
#include "thread.h"
#include "cpu_thread_base.h"
#include "dpu_sync_layout.h"
#include "cpu_task_pipe.h"
#include "legacy_dpu_thread/legacy_dpu_thread.h"

namespace hccl {

/**
 * @note 职责：HOST CPU 通信引擎（HostDpu kernel）的 `Thread` 抽象实现。本类即 DPU 线程：
 *       继承 `CpuThreadBase` 直接承载 SQ/CQ 队列的 DPU 域机制、dpu 流与 kernel 下发；
 *       TaskService + taskexception 通道仍收在 `legacy_` 实体里。
 *       两条通道互斥（1 thread 1 kernel），重名实现（DeInit/GetStream）由本类收编，
 *       按 legacy_ 是否已拉起区分走哪条通道。
 *       kernel 在本线程的 dpu 流上下发并阻塞不返回，直到宿主侧拉起终止握手。
 */
class CpuThread : public Thread, public CpuThreadBase {
public:
    explicit CpuThread(uint32_t notifyNum = 0);
    ~CpuThread() override;

    // Init 为 Thread 协议入口：(AICPU, CPU) 这一档在此完成队列侧 DPU 资源与 kernel 拉起，
    // 申请线程即拿到可用的队列；TaskService 通道仍由触发点显式经 InitLegacy 拉起
    HcclResult Init() override;
    // 覆盖 Thread::DeInit 与 CpuThreadBase::DeInit：按 legacy_ 是否已拉起区分终止哪一侧 kernel
    HcclResult DeInit() override;
    std::string& GetUniqueId() override;
    uint32_t GetNotifyNum() const override;
    size_t GetNotifySize() const override;
    LocalNotify* GetNotify(uint32_t index) const override;
    HcclResult SupplementNotify(uint32_t notifyNum) override;
    HcclResult PreAllocThreadEntity(ThreadHandle& handle, CommEngine targetEngine);

    // A3 Stream & A5 Stream
    bool IsDeviceA5() const override;
    Stream* GetStream() const override;
    void* GetStreamLitePtr() const override;
    s32 GetStreamId() const override;
    void LaunchTask() const override;
    void TryLaunchTask() const override;

    // Local Data Plane Functions
    HcclResult LocalNotifyRecord(uint32_t notifyId) const override;
    HcclResult LocalNotifyWait(uint32_t notifyId) const override;
    HcclResult LocalNotifyRecord(ThreadHandle dstThread, uint32_t dstNotifyIdx) const override;
    HcclResult LocalNotifyWait(uint32_t notifyIdx, uint32_t timeOut) const override;
    HcclResult LocalCopy(void* dst, const void* src, uint64_t sizeByte) const override;
    HcclResult LocalReduce(
        void* dst, const void* src, uint64_t sizeByte, HcommDataType dataType, HcommReduceOp reduceOp) const override;
    bool GetMaster() const override;
    void SetIsMaster(bool isMaster) override;

    /**
     * @brief 取 legacy 专属实体（dpu 状态与 TaskService 动词全在其上）
     */
    LegacyDpuThread* GetLegacy() const { return legacy_.get(); }

    /**
     * @brief 拉起 TaskService 通道 kernel（共享内存 → TaskService → kernel 下发）
     * @return HcclResult 被 SQ/CQ 队列占用时返回 HCCL_E_NOT_SUPPORT（1 thread 1 kernel，先到先得）
     */
    HcclResult InitLegacy(const Hccl::TaskServiceArgs& args);

    /**
     * @brief 拉起 SQ/CQ 队列 kernel（队列建立 → 轮询 kernel 下发）
     * @return HcclResult 被 TaskService 通道占用时返回 HCCL_E_NOT_SUPPORT（判据与 InitLegacy 对称）
     */
    HcclResult InitQueueMode(const Hccl::TaskServiceArgs& args);

    /**
     * @brief 取任务管道（SQ/CQ 队列拉起一次的幂等判据）
     */
    CpuTaskPipe* GetTaskPipe() const { return dpuTaskPipe_.get(); }

    /**
     * @brief 取队列名片（host 视角地址）
     */
    const DpuSyncLayout& GetQueueLayout() const { return queueLayout_; }

    HcclResult GetNotifyShmemAddrs(uint64_t& notifyHostAddr, uint64_t& notifyDevAddr) const;

    /**
     * @brief 取整张队列名片，供地址投递 kernel 按句柄回填 device 侧实体；
     *        队列描述与 notifyDevAddr 为 device 视角，notifyHostAddr 为 host 视角（SQE 执行段在 host 侧直写）
     */
    HcclResult GetDevSyncLayout(
        DpuSyncLayout& layout, QueueDescriptor& sqDesc, QueueDescriptor& cqDesc, uint64_t& notifyHostAddr,
        uint64_t& notifyDevAddr) const;

private:
    HcclResult InitQueueShmem();
    HcclResult InitNotifyShmem();
    HcclResult CreateCpuTaskPipe();
    // InitQueueMode 的下半程（XPU 切换 → 取入口 → 建流 → 下发），失败由调用方决定回滚
    HcclResult PrepareAndLaunchQueueKernel();
    HcclResult LaunchDpuQueueKernel();
    void ResetQueuePipe();
    void ResetQueueModeResources();
    // 队列通道的终止不落共享内存，改置 destroyFlag_ 等 kernel 回执（覆盖基类握手）
    HcclResult WaitKernelTerminate() override;

    bool isMaster_{false};
    // XPU 设备是进程级单例，两条通道的拉起序列必须互斥
    static std::mutex xpuLaunchMutex_;
    // dpu 的流是进程内资源，不参与跨进程序列化，故 uniqueId 保持为空
    std::string uniqueIdStr_;
    // legacy 实体：未拉起时是几乎无开销的空对象
    std::unique_ptr<LegacyDpuThread> legacy_;
    // 本类的队列任务管道与队列名片
    std::unique_ptr<CpuTaskPipe> dpuTaskPipe_;
    DpuSyncLayout queueLayout_{};
    uint32_t notifyNum_{0};
    uint8_t* notifyAddr_{nullptr};
};

} // namespace hccl
#endif // CPU_THREAD_H
