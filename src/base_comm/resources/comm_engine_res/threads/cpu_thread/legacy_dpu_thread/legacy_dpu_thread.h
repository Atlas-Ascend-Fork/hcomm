/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LEGACY_DPU_THREAD_H
#define LEGACY_DPU_THREAD_H

#include <memory>
#include "cpu_thread_base.h"
#include "task_service.h"

namespace hccl {

/**
 * @note 职责：TaskService + AICPU taskexception 通道的 DPU 线程实体。
 *       dpu 域机制、dpu 流、kernel 下发、终止握手、DPUTAG 由基类 `CpuThreadBase` 承担，
 *       本类只装自己的东西：TaskService 实例、DPUTASKEXCEPTION 异常区、回调注册。
 *       kernel 在本线程的 dpu 流上阻塞不返回，直到 DeInit 完成终止握手。
 */
class LegacyDpuThread : public CpuThreadBase {
public:
    LegacyDpuThread() = default;
    // teardown 全收敛进析构：由 ThreadMgr 析构 → HcommThreadFree → 本析构触发，幂等；
    // 内部按「停 AICPU 监控 → 终止 kernel → 析构 TaskService → 释放共享内存」顺序收口
    ~LegacyDpuThread() override;
    // 基类已禁止拷贝与移动，此处仅作意图声明
    LegacyDpuThread(const LegacyDpuThread&) = delete;
    LegacyDpuThread& operator=(const LegacyDpuThread&) = delete;
    LegacyDpuThread(LegacyDpuThread&&) = delete;
    LegacyDpuThread& operator=(LegacyDpuThread&&) = delete;

    /**
     * @brief 完成整段 XPU 切换序列 + 三步收编（共享内存 → taskservice → kernel 下发）
     * @return HcclResult 成功返回 HCCL_SUCCESS
     * @note 调用前须先经 SetTaskServiceArgs 注入入参；本函数内部完成
     *       「InitShmem → CreateTaskService → PrepareDpuKernelResource → LaunchKernel」，
     *       调用方不得在 Init 前后插入任何 device 调用
     * @note 本函数是异常边界：实际流程在 InitImpl 内，异常在此统一转为错误码（调用者
     *       HcclTaskRegister 是 extern "C" 边界，异常穿出会终止进程）
     */
    HcclResult Init();

    /**
     * @brief 在 dpu 流上下发 kernel，幂等
     * @return HcclResult 成功返回 HCCL_SUCCESS
     * @note 参数块（DpuKernelLaunchParam{taskService_, deviceId}）由本类内部组装；
     *       前置条件：CreateTaskService 已完成（taskService_ 非空）
     */
    HcclResult LaunchKernel();

    /**
     * @brief 注册 DPU 任务回调，转发到宿主侧持有的 TaskService 实例
     * @param[in] taskType 任务类型
     * @param[in] callback 回调体
     * @return HcclResult 成功返回 HCCL_SUCCESS
     */
    HcclResult TaskRegister(const std::string& taskType, Hccl::CallbackTemplate callback);

    /**
     * @brief 注销 DPU 任务回调，转发到宿主侧持有的 TaskService 实例
     */
    HcclResult TaskUnRegister(const std::string& taskType);

    /**
     * @brief 取宿主侧持有的 TaskService 实例（首注册拉起一次的幂等判据）
     */
    Hccl::TaskService* GetTaskService() const { return taskService_.get(); }

    /**
     * @brief 通知 AICPU 侧 taskexception 监控线程停止并等待回执（taskExpMem 首字节 1→0 握手）
     * @return HcclResult 成功返回 HCCL_SUCCESS
     * @note 须在释放共享内存之前调用（taskExpMem_ 仍有效）
     */
    HcclResult StopTaskexpService();

    /**
     * @brief 取 DPUTASKEXCEPTION 共享内存 host 侧地址，供 dfx 异常链路读回 DPU 任务错误码
     */
    void* GetTaskExpMem() const { return taskExpMem_; }

    /**
     * @brief 取 DPUTASKEXCEPTION 共享内存 device 侧地址，供 AICPU taskexception kernel 登记与设备侧轮询
     */
    void* GetTaskExpDevMem() const { return shmemMgr_.DevPtr(Hccl::DPU_TASK_EXCEPTION_TAG); }

private:
    // Init 的实际流程（异常由 Init 统一兜底，本函数自身不捕获）
    HcclResult InitImpl();
    HcclResult InitShmem();
    HcclResult CreateTaskService();

    // ~CollComm 已在 DestroyAicpuComm 前置停过 taskexception 握手，析构二次进入 StopTaskexpService 幂等短路
    bool taskExpStopped_{false};

    // 宿主侧持有 TaskService 实例（收编后实例归属从 hccl_v2.so 迁 hcomm.so，随本对象析构）
    std::unique_ptr<Hccl::TaskService> taskService_;
    void* taskExpMem_{nullptr};   // DPUTASKEXCEPTION host 侧地址，TaskService 构造入参之一
    void* hostShareBuf_{nullptr}; // host 侧数据缓冲（100MB），TaskService 构造入参之一，非 owning 引用
};

} // namespace hccl
#endif // LEGACY_DPU_THREAD_H
