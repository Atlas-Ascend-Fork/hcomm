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

#include <atomic>
#include <memory>
#include "thread.h"
#include "dpu_types.h"
#include "dpu_shmem_mgr.h"
#include "task_service.h"

namespace hccl {

/**
 * @note 职责：HOST CPU 通信引擎（HostDpu kernel）的专属实体。
 *       CPU 线程要保证通用性，故 dpu 相关的状态与生命周期动词都收在本类，
 *       由 `CpuThread` 薄壳持有并转发，壳本身不含 dpu 细节。
 *       1 thread 1 kernel，kernel 在本线程的 dpu 流上阻塞不返回，直到 DeInit 完成终止握手。
 */
class LegacyDpuThread {
public:
    LegacyDpuThread() = default;
    // teardown 全收敛进析构：由 ThreadMgr 析构 → HcommThreadFree → 本析构触发，幂等；
    // 内部按「停 AICPU 监控 → 终止 kernel → 析构 TaskService → 释放共享内存」顺序收口
    ~LegacyDpuThread();
    // 持有裸流/上下文句柄，复制会导致重复销毁，禁止拷贝与移动
    LegacyDpuThread(const LegacyDpuThread&) = delete;
    LegacyDpuThread& operator=(const LegacyDpuThread&) = delete;
    LegacyDpuThread(LegacyDpuThread&&) = delete;
    LegacyDpuThread& operator=(LegacyDpuThread&&) = delete;

    /**
     * @brief 建 dpu 流并完成整段 XPU 切换序列 + 三步收编（共享内存 → taskservice → kernel 下发）
     * @return HcclResult 成功返回 HCCL_SUCCESS
     * @note 调用前须先经 SetTaskServiceArgs 注入入参；本函数内部完成
     *       「InitShmem → CreateTaskService → PrepareDpuKernelResource → LaunchKernel」，
     *       调用方不得在 Init 前后插入任何 device 调用
     * @note 本函数是异常边界：实际流程在 InitImpl 内，异常在此统一转为错误码（调用者
     *       HcclTaskRegister 是 extern "C" 边界，异常穿出会终止进程）
     */
    HcclResult Init();

    /**
     * @brief 终止握手 + 销毁 dpu 流 + XPU 记账，幂等
     * @return HcclResult 成功返回 HCCL_SUCCESS
     */
    HcclResult DeInit();

    /**
     * @brief 在 dpu 流上下发 kernel，幂等
     * @return HcclResult 成功返回 HCCL_SUCCESS
     * @note 参数块（DpuKernelLaunchParam{taskService_, deviceId}）由本类内部组装；
     *       前置条件：CreateTaskService 已完成（taskService_ 非空）
     */
    HcclResult LaunchKernel();

    /**
     * @brief 注入构造 TaskService 所需的入参（触发点在 HcclTaskRegister 帧内整理传入；仅身份，不含内存地址）
     */
    void SetTaskServiceArgs(const Hccl::TaskServiceArgs& args) { args_ = args; }

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

    Stream* GetStream() const { return stream_.get(); }
    DevType GetDevType() const { return devType_; }
    bool IsKernelLaunched() const { return isKernelLaunched_; }
    uint32_t GetStreamId() const { return streamId_; }
    uint32_t GetDevId() const { return args_.devId; }

    /**
     * @brief 取 DPUTASKEXCEPTION 共享内存 host 侧地址，供 dfx 异常链路读回 DPU 任务错误码
     */
    void* GetTaskExpMem() const { return taskExpMem_; }

    /**
     * @brief 取 DPUTASKEXCEPTION 共享内存 device 侧地址，供 AICPU taskexception kernel 登记与设备侧轮询
     */
    void* GetTaskExpDevMem() const { return shmemMgr_.DevPtr(Hccl::DPU_TASK_EXCEPTION_TAG); }

    /**
     * @brief 取 DPUTAG 共享内存 host 侧地址（host 侧写 dpu2npu flag 与终止握手用）
     */
    void* GetDputagAddr() const { return dputagAddr_; }

    /**
     * @brief 取 DPUTAG 共享内存 device 侧地址（仅 device 侧用）
     */
    void* GetDputagDevAddr() const { return dputagDevAddr_; }

private:
    // Init 的实际流程（异常由 Init 统一兜底，本函数自身不捕获）
    HcclResult InitImpl();
    HcclResult InitShmem();
    HcclResult CreateTaskService();
    HcclResult PrepareDpuKernelResource();
    HcclResult WaitKernelTerminate();
    HcclResult RefreshStreamId();

    rtStream_t rtStream_{nullptr};
    aclrtContext npuContext_{nullptr}; // 进入 dpu 域前保存的当前上下文
    aclrtContext dpuContext_{nullptr}; // dpu 域上下文
    aclrtFuncHandle funcHandle_{nullptr};
    void* dputagHostPtr_{nullptr}; // DPUTAG host 侧地址（终止握手信号区），InitShmem 内部置为 dputagAddr_
    uint32_t streamId_{0};
    DevType devType_{DevType::DEV_TYPE_COUNT};
    bool isKernelLaunched_{false};
    // ~CollComm 已在 DestroyAicpuComm 前置停过 taskexception 握手，析构二次进入 StopTaskexpService 幂等短路
    bool taskExpStopped_{false};
    std::unique_ptr<Stream> stream_;

    // 宿主侧持有 TaskService 实例（收编后实例归属从 hccl_v2.so 迁 hcomm.so，随本对象析构）
    std::unique_ptr<Hccl::TaskService> taskService_;
    void* dputagAddr_{nullptr}; // DPUTAG host 侧地址（TaskService 构造入参之一，兼终止握手/Suspend 写 flag）
    void* dputagDevAddr_{nullptr}; // DPUTAG device 侧地址（仅供 HcclDevMemAcquire 取用，device scatter kernel 解引用）
    void* taskExpMem_{nullptr};   // DPUTASKEXCEPTION host 侧地址，TaskService 构造入参之一
    void* hostShareBuf_{nullptr}; // host 侧数据缓冲（100MB），TaskService 构造入参之一，非 owning 引用
    DpuShmemMgr shmemMgr_;        // DPU 共享内存管理器（DPUTAG + DPUTASKEXCEPTION，独立可复用类）
    Hccl::TaskServiceArgs args_;  // 触发点传入的构造入参（仅身份：commId/devId）

    /**
     * @brief 进程级通信域计数，决定 XPU 设备何时 reset
     * @note 必须是静态量：HrtSetXpuDevice 是进程级设备切换，退化成实例成员会让
     *       reset 时机变成「本通信域计数归零」，与真实语义不符
     */
    static std::atomic<uint32_t> commNum_;
};

} // namespace hccl
#endif // LEGACY_DPU_THREAD_H
