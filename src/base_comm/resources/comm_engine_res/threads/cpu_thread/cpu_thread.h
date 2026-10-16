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
#include <string>
#include "thread.h"
#include "legacy_dpu_thread/legacy_dpu_thread.h"

namespace hccl {

/**
 * @note 职责：HOST CPU 通信引擎（HostDpu kernel）的 `Thread` 抽象实现。
 *       本类刻意做「薄壳」——CPU 线程要保证通用性，dpu 相关的状态与生命周期
 *       全部收在 `LegacyDpuThread` 里，本类只实现 `Thread` 的接口面并转发。
 *       1 thread 1 kernel：kernel 在本线程的 dpu 流上下发并阻塞不返回，
 *       直到宿主侧调用 `DeInit()` 完成终止握手。
 */
class CpuThread : public Thread {
public:
    CpuThread();
    ~CpuThread() override;

    // Init 为 Thread 协议入口：不转发 legacy（dpu 三步由触发点显式经 LegacyDpuThread::Init 拉起），
    // 留作后续兼容性扩展，当前无通用资源需初始化
    HcclResult Init() override;
    // DeInit 同 Init：dpu 专属 teardown 已收敛进 LegacyDpuThread 析构（ThreadMgr 析构触发），
    // 本协议入口留作后续兼容性扩展，当前无通用资源需释放
    HcclResult DeInit() override;
    std::string& GetUniqueId() override;
    uint32_t GetNotifyNum() const override;
    LocalNotify* GetNotify(uint32_t index) const override;
    HcclResult SupplementNotify(uint32_t notifyNum) override;

    // A3 Stream & A5 Stream
    bool IsDeviceA5() const override;
    Stream* GetStream() const override;
    void* GetStreamLitePtr() const override;
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
     * @brief 取 dpu 专属实体（dpu 状态与 TaskService 动词全在其上），宿主侧经此下发 kernel/注入 DPUTAG/注册回调
     * @return LegacyDpuThread* 实体指针，随本对象生命周期
     */
    LegacyDpuThread* GetLegacy() const { return legacy_.get(); }

private:
    bool isMaster_{false};
    // dpu 的流是进程内资源，不参与跨进程序列化，故 uniqueId 保持为空
    std::string uniqueIdStr_;
    std::unique_ptr<LegacyDpuThread> legacy_;
};

} // namespace hccl
#endif // CPU_THREAD_H
