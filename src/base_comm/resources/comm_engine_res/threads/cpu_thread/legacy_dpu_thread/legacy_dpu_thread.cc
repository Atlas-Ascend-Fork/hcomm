/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "legacy_dpu_thread.h"
#include <chrono>
#include <cstdlib>
#include <exception>
#include "securec.h"
#include "hccl_common.h"
#include "adapter_rts.h"
#include "driver/ascend_hal.h"
#include "log.h"

namespace hccl {

namespace {
    constexpr u32 WAIT_CMD_TIMEOUT = 10 * 1000; // taskexception 停止握手等待 10 秒
} // namespace

LegacyDpuThread::~LegacyDpuThread()
{
    // 收编后 teardown 全部收敛进析构：由 ThreadMgr 析构 → HcommThreadFree → 本析构触发，
    // 每一步均幂等（isKernelLaunched_/taskExpMem_/hostShareBuf_ 判空），重复调用安全。
    // 顺序约束：先停 AICPU taskexception 监控（taskExpMem_ 仍有效），再终止 kernel，再析构实例并释放内存。
    // 共享内存（DPUTAG/DPUTASKEXCEPTION）的释放由基类析构统一收口，本类只保证释放前 kernel 已退出
    (void)StopTaskexpService();
    // 派生析构体先于基类析构体执行：必须在自己这里显式终止 kernel，否则下面的 taskService_ 会先于
    // 终止握手析构，而 kernel 在该窗口内仍持其裸指针（UAF）
    (void)DeInit();

    // TaskService 实例析构须先于 hostShareBuf_ 释放（hostMem_ 非 owning 引用，否则 UAF）
    taskService_.reset();
    if (hostShareBuf_ != nullptr) {
        free(hostShareBuf_);
        hostShareBuf_ = nullptr;
    }
    taskExpMem_ = nullptr;
}

HcclResult LegacyDpuThread::Init()
{
    // 本函数是异常边界：调用者 HcclTaskRegister 是 extern "C" 边界，异常穿出会终止进程。
    // 链路上多处可抛（TaskService 构造、jsonPath 等 STL 分配、EnvConfig 解析、Stream 构造），
    // 统一在此兜底；转成错误码后由调用方走既有回滚（RollbackAcquiredThread 释放线程与内存）
    try {
        return InitImpl();
    } catch (const std::exception& e) {
        HCCL_ERROR("[LegacyDpuThread][%s] exception caught: %s", __func__, e.what());
        return HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("[LegacyDpuThread][%s] unknown exception caught", __func__);
        return HCCL_E_INTERNAL;
    }
}

HcclResult LegacyDpuThread::InitImpl()
{
    CHK_RET(hrtGetDeviceType(devType_));

    // 收编步 1：初始化共享内存（DPUTAG + DPUTASKEXCEPTION，需在 npu ctx 下、切 dpu 域之前完成）
    CHK_RET(InitShmem());

    // 收编步 2a：宿主侧同步构造 TaskService（纯 host 侧构造，无 device 调用，先于 kernel 就绪）
    CHK_RET(CreateTaskService());

    // 切 dpu 域（失败时 EnterDpuContext 自行还原 ctx）
    CHK_RET(EnterDpuContext());
    // 从切入 dpu 域起，后续任一步失败都由守卫在返回前切回 npu ctx，避免调用线程滞留 dpu 域
    CpuCtxRollback rollback(npuContext_);

    // 从此处起 rtStream_ 有可能被建出，失败出口须回收（见 CpuStreamRollback）
    CpuStreamRollback streamRollback(rtStream_, isKernelLaunched_, dpuContext_, npuContext_, stream_);

    // 准备资源：加载 libccl_dpu 并建 dpu 流
    CHK_RET(PrepareDpuKernelResource());

    // Stream 构造经 InitStream 间接调 hrt 系列可能抛异常：catch 拦截避免异常穿透 extern C 边界
    // （HcclTaskRegister），切回 npu ctx 由上方 CpuCtxRollback 守卫兜底
    try {
        stream_.reset(new (std::nothrow) Stream(rtStream_));
    } catch (...) {
        HCCL_ERROR("[LegacyDpuThread][%s] create Stream failed, exception caught", __func__);
        return HCCL_E_PTR;
    }
    CHK_SMART_PTR_NULL(stream_);

    // 收编步 2b：下发 kernel
    CHK_RET(LaunchKernel());

    HCCL_INFO("[LegacyDpuThread][%s] end, rtStream[%p]", __func__, rtStream_);
    return HCCL_SUCCESS;
}

HcclResult LegacyDpuThread::InitShmem()
{
    // 收编步 1：DPUTAG（100MB，基类公共机制）+ DPUTASKEXCEPTION（20B，本通道专属）两块共享内存的
    // 申请与注册，全部收编到 base_comm 由独立管理器 DpuShmemMgr 托管：
    // 950 不再依赖 legacy 侧单独的全局表；DPUTAG 的申请在基类，本类只补 DPUTASKEXCEPTION 这半段
    int32_t deviceLogicId = 0;
    CHK_RET(ResolveDeviceLogicId(deviceLogicId));
    CHK_RET(AllocDputagShmem(deviceLogicId));

    // DPUTASKEXCEPTION：异常区 host 侧地址
    CHK_RET(shmemMgr_.Alloc(Hccl::DPU_TASK_EXCEPTION_TAG, Hccl::DPU_TASKEXCEPTION_MEMORY_SIZE, deviceLogicId));
    taskExpMem_ = shmemMgr_.HostPtr(Hccl::DPU_TASK_EXCEPTION_TAG);

    const Hccl::DpuShmem* taskExp = shmemMgr_.Get(Hccl::DPU_TASK_EXCEPTION_TAG);
    HCCL_INFO(
        "[LegacyDpuThread][%s] taskExp va_[%p] accessVA_[%p]", __func__, taskExp != nullptr ? taskExp->va_ : nullptr,
        taskExp != nullptr ? taskExp->accessVA_ : nullptr);
    return HCCL_SUCCESS;
}

HcclResult LegacyDpuThread::CreateTaskService()
{
    // host 侧数据缓冲（100MB）：TaskService::hostMem_ 是非 owning 引用，析构须先析构实例再 free（UAF 放大器）
    hostShareBuf_ = malloc(Hccl::SHARE_HBM_MEMORY_SIZE);
    CHK_PTR_NULL(hostShareBuf_);

    // 收编后实例归属宿主侧：hcomm.so 构造 TaskService，kernel 侧 RunDpuRpcSrvLaunch 只取裸指针下发 TaskRun
    // PCIE 链路下宿主侧读 device 侧共享内存走 aclrtMemcpy, UB 链路下可直接 memcpy_s
    const Hccl::DpuShmem* dputag = shmemMgr_.Get(Hccl::DPU_TAG);
    const bool isPcieLink = dputag != nullptr && dputag->connectType_ == HOST_DEVICE_CONNECT_TYPE_PCIE;
    HCCL_INFO(
        "[LegacyDpuThread][%s] TaskServiceArgs{deviceMem:%p; deviceMemSize:%lu; hostMem:%p; hostMemSize:%lu; "
        "taskExpMem:%p; commId:%s; devId:%u; isPcieLink:%u}",
        __func__, dputagAddr_, Hccl::SHARE_HBM_MEMORY_SIZE, hostShareBuf_, Hccl::SHARE_HBM_MEMORY_SIZE, taskExpMem_,
        args_.commId.c_str(), args_.devId, static_cast<uint32_t>(isPcieLink));
    taskService_ = std::make_unique<Hccl::TaskService>(
        dputagAddr_, static_cast<int32_t>(Hccl::SHARE_HBM_MEMORY_SIZE), hostShareBuf_,
        static_cast<int32_t>(Hccl::SHARE_HBM_MEMORY_SIZE), taskExpMem_, args_.commId, args_.devId, isPcieLink,
        dputagDevAddr_);
    CHK_SMART_PTR_NULL(taskService_);
    return HCCL_SUCCESS;
}

HcclResult LegacyDpuThread::LaunchKernel()
{
    if (isKernelLaunched_) {
        HCCL_INFO("[LegacyDpuThread][%s] kernel already launched, skip", __func__);
        return HCCL_SUCCESS;
    }
    if (taskService_ == nullptr) {
        HCCL_ERROR("[LegacyDpuThread][%s] taskService is nullptr, call CreateTaskService first", __func__);
        return HCCL_E_INTERNAL;
    }

    // 参数块由本类内部组装：收编后只带宿主侧已构造的 TaskService 裸指针 + deviceId
    Hccl::DpuKernelLaunchParam hostArgs{taskService_.get(), args_.devId};
    return LaunchDpuKernelWithArgs(&hostArgs, sizeof(hostArgs));
}

HcclResult LegacyDpuThread::TaskRegister(const std::string& taskType, Hccl::CallbackTemplate callback)
{
    if (taskService_ == nullptr) {
        HCCL_ERROR("[LegacyDpuThread][%s] taskService is nullptr, register failed", __func__);
        return HCCL_E_INTERNAL;
    }
    return taskService_->TaskRegister(taskType, callback);
}

HcclResult LegacyDpuThread::TaskUnRegister(const std::string& taskType)
{
    if (taskService_ == nullptr) {
        HCCL_ERROR("[LegacyDpuThread][%s] taskService is nullptr, unregister failed", __func__);
        return HCCL_E_INTERNAL;
    }
    return taskService_->TaskUnRegister(taskType);
}

HcclResult LegacyDpuThread::StopTaskexpService()
{
    // 幂等：~CollComm 已在 DestroyAicpuComm 前置停过（taskExpStopped_ 置位），析构二次进入直接返回，
    // 避免 device 侧 IsTaskExpStopped 早退不回写 flag 导致的二次握手超时
    if (taskExpStopped_) {
        return HCCL_SUCCESS;
    }
    // 非 dpu 场景（kernel 未拉起）或异常区未申请，无共享内存可停止
    if (!isKernelLaunched_ || taskExpMem_ == nullptr) {
        return HCCL_SUCCESS;
    }
    uint8_t flag = 1;
    errno_t ret = memcpy_s(taskExpMem_, sizeof(flag), &flag, sizeof(flag)); // taskexception 首字节作为停止标志位
    if (ret != EOK) {
        HCCL_ERROR("[LegacyDpuThread][%s] set stop flag failed: %d", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    auto timeout = std::chrono::milliseconds(WAIT_CMD_TIMEOUT);
    auto startTime = std::chrono::steady_clock::now();
    do { // 等待 aicpu 背景线程 response
        ret = memcpy_s(&flag, sizeof(flag), taskExpMem_, sizeof(flag));
        if (ret != EOK) {
            HCCL_ERROR("[LegacyDpuThread][%s] read device response failed: %d", __func__, ret);
            return HCCL_E_INTERNAL;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_WARNING(
                "[LegacyDpuThread][%s] wait device destroy response timeout[%u ms].", __func__, WAIT_CMD_TIMEOUT);
            return HCCL_E_TIMEOUT;
        }
    } while (flag != 0);
    taskExpStopped_ = true;
    return HCCL_SUCCESS;
}

} // namespace hccl
