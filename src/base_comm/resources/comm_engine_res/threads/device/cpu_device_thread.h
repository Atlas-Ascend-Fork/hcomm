/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CPU_DEVICE_THREAD_H
#define CPU_DEVICE_THREAD_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include "thread.h"
#include "cpu_device_sq_submitter.h"
#include "dpu_sync_layout.h"

namespace hccl {

/**
 * @note 职责：DPU 线程在 device 侧的实体，持有 SQ 提交器、CQ 描述与 notify 寻址，供 AICPU 编排侧按
 *       handle → 实体提交 SQE，也让 CQ 守护按 handle 直接取到本线程的 CQ。
 *       本线程不持有设备流资源（无
 *       StreamLite/pImpl_/Rtsq），构造即置 fakeDeviceRes_，资源注册与任务下发按该标志统一拦截跳过，
 *       与 GE 保序桩线程共用同一套口径（见 thread.h 的 SetFakeDeviceRes 注释）。
 *       host侧对应物是薄壳 CpuThread（内部持 LegacyDpuThread），两侧由下发参数块的 threadType 配对。
 */
class CpuDeviceThread : public Thread {
public:
    CpuDeviceThread(const std::string& commId, int32_t deviceLogicId, uint32_t deviceType);
    ~CpuDeviceThread() override = default;

    // Thread 协议面：本线程无设备流/notify，取值口径与host侧 CpuThread 保持一致
    HcclResult Init() override { return HCCL_SUCCESS; }
    HcclResult DeInit() override { return HCCL_SUCCESS; }
    std::string& GetUniqueId() override { return uniqueIdStr_; }
    uint32_t GetNotifyNum() const override { return notifyNum_.load(std::memory_order_acquire); }
    size_t GetNotifySize() const override { return 0; }
    LocalNotify* GetNotify(uint32_t index) const override;
    HcclResult SupplementNotify(uint32_t notifyNum) override;
    bool IsDeviceA5() const override;
    Stream* GetStream() const override { return nullptr; }
    void* GetStreamLitePtr() const override { return nullptr; }
    s32 GetStreamId() const override { return -1; }
    void LaunchTask() const override { return; }
    void TryLaunchTask() const override { return; }
    HcclResult LocalNotifyRecord(uint32_t notifyId) const override;
    HcclResult LocalNotifyWait(uint32_t notifyId) const override;
    HcclResult LocalNotifyRecord(ThreadHandle dstThread, uint32_t dstNotifyIdx) const override;
    HcclResult LocalNotifyWait(uint32_t notifyIdx, uint32_t timeOut) const override;
    HcclResult LocalChannelNotifyWait(ChannelHandle channel, uint32_t notifyIdx, uint32_t timeout) const override;
    HcclResult LocalChannelNotifyRecord(ChannelHandle channel, uint32_t notifyIdx) const override;
    HcclResult LocalChannelWriteWithNotifyNbi(
        ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t notifyIdx) const override;
    HcclResult LocalChannelDrain(ChannelHandle channel) const override;
    HcclResult LocalChannelFence(ChannelHandle channel) const override;
    HcclResult LocalCommFence() const override;
    HcclResult LocalCopy(void* dst, const void* src, uint64_t sizeByte) const override;
    HcclResult
    LocalReduce(void* dst, const void* src, uint64_t sizeByte, HcommDataType dataType, HcommReduceOp op) const override;
    bool GetMaster() const override { return isMaster_; }
    void SetIsMaster(bool isMaster) override { isMaster_ = isMaster; }

    void SetQueueNamecard(
        const QueueDescriptor& sqDesc, const QueueDescriptor& cqDesc, uint32_t notifyNum, uint64_t notifyHostAddr,
        uint64_t notifyDevAddr);

    void UpdateNotifyInfo(uint32_t notifyNum, uint64_t notifyHostAddr, uint64_t notifyDevAddr);

    const std::string& GetCommId() const { return commId_; }
    int32_t GetDeviceLogicId() const { return deviceLogicId_; }
    uint32_t GetDeviceType() const { return deviceType_; }
    // 队列名片是否已投递（未投递意味着队列 kernel 尚未拉起，消费侧须空转）
    bool IsNamecardReady() const { return namecardReady_.load(std::memory_order_acquire); }
    // 只在本类 IsNamecardReady() 为真后调用；那时 cqDesc_ 已写全
    const QueueDescriptor& GetCqDesc() const { return cqDesc_; }
    CpuDeviceSqSubmitter& GetSqSubmitter() const { return sqSubmitter_; }

    HcclResult GetNotifyDevShmAddr(uint32_t notifyIdx, uint64_t& shmAddr) const override;
    HcclResult GetNotifyHostShmAddr(uint32_t notifyIdx, uint64_t& shmAddr) const override;

private:
    std::string commId_;
    int32_t deviceLogicId_{-1};
    uint32_t deviceType_{0};
    // 本线程身份只走下发参数块的 out-of-band 字段（CpuThread::GetUniqueId 有意恒空，不进序列化 blob）
    std::string uniqueIdStr_;
    // 名片与 notify 由 device 侧 kernel 线程写、守护与编排线程读：写侧先写全字段，最后写 notifyNum_ /
    // namecardReady_（release）；读侧须先 acquire 取到这两个值，再读其余字段
    std::atomic<uint32_t> notifyNum_{0};
    std::atomic<uint8_t*> notifyHostAddr_{nullptr};
    std::atomic<uint8_t*> notifyDevAddr_{nullptr};
    mutable CpuDeviceSqSubmitter sqSubmitter_{};
    QueueDescriptor cqDesc_{}; // 在 namecardReady_ 置真之前写全，之后不再改动
    std::atomic<bool> namecardReady_{false};
    bool isMaster_{false};
};

/**
 * @brief 按下发参数块的 threadType 造 device 侧线程实体
 * @param[in] threadType 线程类型（host 侧 CreateThread 设定并随参数块下发）
 * @param[in] commId 通信域标识（定长字符数组，须以 '\0' 结尾）
 * @param[in] deviceLogicId/deviceType 基础通信设备信息，随实体留存备查
 * @param[out] isCpuDeviceThread 是否确为本函数构建的 DPU 线程
 * @param[out] outThread 造出的实体；isCpuDeviceThread == true 时必定非空
 * @return HCCL_SUCCESS 且 isCpuDeviceThread == true：已造出 CpuDeviceThread 并登记进 cpu_device_thread_registry
 *         HCCL_SUCCESS 且 isCpuDeviceThread == false：非 DPU 线程，调用方走 AicpuTsThread 原路
 *         其他：DPU 线程但构建失败，调用方须直接上抛——退回原路会造出错类型的实体
 */
HcclResult MakeCpuDeviceThread(
    ThreadType threadType, const char* commId, int32_t deviceLogicId, uint32_t deviceType, bool& isCpuDeviceThread,
    std::shared_ptr<hccl::Thread>& outThread);

} // namespace hccl

#endif // CPU_DEVICE_THREAD_H
