/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cpu_thread_service.h"
#include "thread_manager.h" // ThreadMgr 完整类型（GetCpuThread / HcclThreadAcquireV2 / GetThread / RollbackThread）
#include "thread.h"         // hccl::Thread 完整类型（dynamic_pointer_cast 需要）
#include "cpu_thread/cpu_thread.h"                   // CpuThread：两条通道的宿主实体与两个拉起入口
#include "kernel_launch/dpu_taskexp_kernel_launch.h" // taskexp kernel 的 L2 下发器（原弱符号转发的 legacy 实现）
#include <vector>

namespace hccl {

std::mutex CpuThreadService::cpuLaunchMutex_;

CpuThreadService::CpuThreadService(ThreadMgr* threadMgr, aclrtBinHandle binHandle)
    : threadMgr_(threadMgr),
      binHandle_(binHandle)
{}

HcclResult CpuThreadService::StopTaskexpServiceByCpuThread(const std::shared_ptr<CpuThread>& cpuThread)
{
    if (cpuThread == nullptr) {
        return HCCL_SUCCESS;
    }
    auto* legacy = cpuThread->GetLegacy();
    if (legacy == nullptr) {
        return HCCL_SUCCESS;
    }
    return legacy->StopTaskexpService();
}

HcclResult
CpuThreadService::RollbackAcquiredThread(ThreadHandle handle, std::shared_ptr<CpuThread>& cpuThread, HcclResult cause)
{
    cpuThread = nullptr; // 先断开本地持有，句柄才能真正在 HcommThreadFree 内析构
    HcclResult freeRet = threadMgr_->RollbackThread(handle);
    if (freeRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] rollback thread[0x%llx] failed, comm[%s], ret[%d]", __func__, handle, threadMgr_->GetCommId().c_str(),
            freeRet);
    }
    return cause;
}

HcclResult CpuThreadService::EnsureCpuKernelLaunched(
    HcclComm commV2, uint32_t devId, std::shared_ptr<CpuThread>& cpuThread, bool* newCreated)
{
    cpuThread = nullptr;
    // 仅创建成功路径会置 true，中途失败保持 false
    if (newCreated != nullptr) {
        *newCreated = false;
    }

    const std::string& commId = threadMgr_->GetCommId();
    std::lock_guard<std::mutex> lock(cpuLaunchMutex_);

    // 幂等主判据：查链命中且 TaskService 已就绪才算已拉起（池条目由 acquire 登记，未过 Init 的条目不可用）
    std::shared_ptr<Thread> existed;
    HcclResult ret = threadMgr_->GetCpuThread(existed);
    if (ret == HCCL_SUCCESS) {
        cpuThread = std::dynamic_pointer_cast<CpuThread>(existed);
        if (cpuThread == nullptr) {
            HCCL_ERROR("[%s] existed thread is not a CpuThread, comm[%s]", __func__, commId.c_str());
            return HCCL_E_INTERNAL;
        }
        LegacyDpuThread* existedLegacy = cpuThread->GetLegacy();
        if (existedLegacy == nullptr || existedLegacy->GetTaskService() == nullptr
            || existedLegacy->GetDputagAddr() == nullptr) {
            // 半拉起状态：不伪装成成功（下游会把未拉起的 kernel 当已拉起用），显式报错
            HCCL_ERROR("[%s] host cpu thread exists but not fully launched, comm[%s]", __func__, commId.c_str());
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }
    if (ret != HCCL_E_NOT_FOUND) {
        return ret;
    }

    // devId 只在本分支被消费；UINT32_MAX 是未设置哨兵，提前判参避免下游以 0xFFFFFFFF 失败
    CHK_PRT_RET(
        devId == UINT32_MAX, HCCL_ERROR("[%s] devId is not set, comm[%s]", __func__, commId.c_str()), HCCL_E_PARA);

    // DPUTAG/DPUTASKEXCEPTION 共享内存已收编进 LegacyDpuThread::InitShmem 由 base_comm 托管，
    // 触发点不再经 HcclDevMemAcquireV2 预申请。
    Hccl::TaskServiceArgs args{commId, devId};

    // 首次创建走标准 acquire：池登记/幂等复用/随域销毁均免费，不再手动 CreateThread + SaveThreads + RegisterCpuThread。
    // notifyNumPerThread = 0：HOST CPU 线程不申请 local notify，同步走共享内存任务队列。
    ThreadConfig config;
    HcommResult initRet = ThreadConfigInit(&config, 1);
    CHK_PRT_RET(initRet != 0, HCCL_ERROR("[%s] ThreadConfigInit failed, ret[%d]", __func__, initRet), HCCL_E_INTERNAL);

    ThreadHandle handle = 0;
    std::vector<uint32_t> threadId;
    CHK_RET(threadMgr_->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_CPU, 1, ThreadType::THREAD_TYPE_CPU, &config, &handle, threadId));

    // acquire 已把线程登记进池，此后任一失败都必须回滚（见 RollbackAcquiredThread），
    // 否则池中出现「有条目但未拉起」的条目，本函数会一直假成功、Init 永不重试
    std::shared_ptr<Thread> newThread;
    HcclResult getRet = threadMgr_->GetThread(handle, newThread);
    if (getRet != HCCL_SUCCESS) {
        return RollbackAcquiredThread(handle, cpuThread, getRet);
    }
    cpuThread = std::dynamic_pointer_cast<CpuThread>(newThread);
    if (cpuThread == nullptr) {
        HCCL_ERROR("[%s] acquire did not produce a CpuThread, comm[%s]", __func__, commId.c_str());
        return RollbackAcquiredThread(handle, cpuThread, HCCL_E_INTERNAL);
    }

    // 只拉起 TaskService 通道实体（不经 CpuThread::Init 这个通用 Thread 协议入口），dpu 专属三步收编
    // （共享内存→TaskService→kernel 下发）完全留在实体自己类里；1 thread 1 kernel 的交叉判据在 CpuThread 上。
    LegacyDpuThread* legacy = cpuThread->GetLegacy();
    if (legacy == nullptr) {
        HCCL_ERROR("[%s] legacy dpu thread is nullptr, comm[%s]", __func__, commId.c_str());
        return RollbackAcquiredThread(handle, cpuThread, HCCL_E_INTERNAL);
    }
    HcclResult legacyInitRet = cpuThread->InitLegacy(args);
    if (legacyInitRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] legacy dpu thread init failed, comm[%s], ret[%d]", __func__, commId.c_str(), legacyInitRet);
        return RollbackAcquiredThread(handle, cpuThread, legacyInitRet);
    }

    // 通信域初始化时不再拉起 AICPU taskexception kernel；改为随首次拉起（本分支）一并下发，
    // 把本 comm 的 DPUTASKEXCEPTION 共享内存地址登记到 AICPU 侧 commId→VA 登记表。
    // 仅 950 的 V2 通信域有该登记表；910 无此机制，commV2 为 nullptr，跳过。
    if (commV2 != nullptr) {
        HcclResult launchRet = LaunchDpuTaskexpShmemRestoreKernel(
            binHandle_, commId, devId, legacy->GetTaskExpDevMem(), Hccl::DPU_TASKEXCEPTION_MEMORY_SIZE);
        if (launchRet != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] launch dpu taskexp kernel failed, comm[%s], ret[%d]", __func__, commId.c_str(), launchRet);
            return RollbackAcquiredThread(handle, cpuThread, launchRet);
        }
    }

    // 与上面的查链命中判定同处临界区，故仅真正的创建者得 true
    if (newCreated != nullptr) {
        *newCreated = true;
    }
    HCCL_INFO("[%s] host cpu thread launched, comm[%s], handle[0x%llx]", __func__, commId.c_str(), handle);
    return HCCL_SUCCESS;
}

HcclResult CpuThreadService::StopTaskexpService()
{
    const std::string& commId = threadMgr_->GetCommId();
    // 三条收编期观测点：请求进入 / 查链落空 / 握手返回，用于在日志里分辨究竟哪个时机真的发出了握手。
    HCCL_INFO("[%s] stop handshake requested by ~CollComm, comm[%s]", __func__, commId.c_str());

    std::shared_ptr<Thread> thread;
    HcclResult ret = threadMgr_->GetCpuThread(thread);
    if (ret == HCCL_E_NOT_FOUND) {
        // 非 dpu 场景（无 CPU thread），无 taskexception 监控可停。
        HCCL_INFO("[%s] no host cpu thread, skip handshake, comm[%s]", __func__, commId.c_str());
        return HCCL_SUCCESS;
    }
    CHK_RET(ret);
    HcclResult stopRet = StopTaskexpServiceByCpuThread(std::dynamic_pointer_cast<CpuThread>(thread));
    HCCL_INFO("[%s] handshake returned, comm[%s], ret[%d]", __func__, commId.c_str(), stopRet);
    return stopRet;
}

HcclResult CpuThreadService::StopCpuKernel()
{
    const std::string& commId = threadMgr_->GetCommId();
    std::shared_ptr<CpuThread> cpuThread;
    HcclResult ret = threadMgr_->GetCpuThread(cpuThread);
    if (ret == HCCL_E_NOT_FOUND) {
        return HCCL_SUCCESS; // 未拉起 DPU：无 kernel 可停
    }
    CHK_RET(ret);
    LegacyDpuThread* legacy = cpuThread->GetLegacy();
    if (legacy == nullptr) {
        return HCCL_SUCCESS;
    }
    // DeInit 幂等（isKernelLaunched_ 首行清标志），~LegacyDpuThread 二次进入空转
    HcclResult deinitRet = legacy->DeInit();
    HCCL_INFO("[%s] dpu kernel stopped, comm[%s], ret[%d]", __func__, commId.c_str(), deinitRet);
    return deinitRet;
}

HcclResult
CpuThreadService::AcquireCpuTagShmem(HcclComm commV2, uint32_t devId, uint64_t* size, void** addr, bool* newCreated)
{
    if (size == nullptr || addr == nullptr) {
        HCCL_ERROR("[%s] size/addr is nullptr", __func__);
        return HCCL_E_PTR;
    }
    // DPUTAG 尺寸为 hccl 侧冻结契约（100MB），与传入尺寸不一致即判参错，避免返回口径不一致的地址
    if (*size != Hccl::SHARE_HBM_MEMORY_SIZE) {
        HCCL_ERROR("[%s] DPUTAG size mismatch, expect[%lu], got[%lu]", __func__, Hccl::SHARE_HBM_MEMORY_SIZE, *size);
        return HCCL_E_PARA;
    }

    // 预查询只用于选路；并发首次进入时后到者也会走该分支，故 created 只能由锁内判定回传
    std::shared_ptr<CpuThread> cpuThread;
    HcclResult ret = threadMgr_->GetCpuThread(cpuThread);
    bool created = false;
    if (ret == HCCL_E_NOT_FOUND) {
        CHK_RET(EnsureCpuKernelLaunched(commV2, devId, cpuThread, &created));
    } else {
        CHK_RET(ret);
    }

    auto* legacy = cpuThread->GetLegacy();
    CHK_PTR_NULL(legacy);
    // 取址方是 device scatter kernel，必须给 device 侧地址；host 侧地址只给 TaskService/握手用
    void* cpuTagAddr = legacy->GetDputagDevAddr();
    if (cpuTagAddr == nullptr) {
        HCCL_ERROR("[%s] DPUTAG device addr is nullptr, comm[%s]", __func__, threadMgr_->GetCommId().c_str());
        return HCCL_E_INTERNAL;
    }
    *addr = cpuTagAddr;
    if (newCreated != nullptr) {
        *newCreated = created;
    }
    return HCCL_SUCCESS;
}

HcclResult CpuThreadService::GetCpuTagAddrIfLaunched(void*& addr)
{
    addr = nullptr;
    std::shared_ptr<CpuThread> cpuThread;
    HcclResult ret = threadMgr_->GetCpuThread(cpuThread);
    if (ret == HCCL_E_NOT_FOUND) {
        return HCCL_SUCCESS; // 未拉起 DPU：本通信域没有 DPUTAG，无标志可写
    }
    CHK_RET(ret);
    LegacyDpuThread* legacy = cpuThread->GetLegacy();
    if (legacy == nullptr) {
        return HCCL_SUCCESS;
    }
    addr = legacy->GetDputagAddr();
    return HCCL_SUCCESS;
}

HcclResult CpuThreadService::GetTaskExpMem(void*& addr, std::shared_ptr<void>& keepAlive) const
{
    addr = nullptr;
    keepAlive = nullptr;
    std::shared_ptr<CpuThread> cpuThread;
    HcclResult ret = threadMgr_->GetCpuThread(cpuThread);
    if (ret != HCCL_SUCCESS) {
        return ret; // 该通信域未拉起 DPU 线程，无异常区
    }
    auto* legacy = cpuThread->GetLegacy();
    if (legacy == nullptr) {
        return HCCL_E_INTERNAL;
    }
    addr = legacy->GetTaskExpMem();
    // 异常区在实体析构内 FreeAll，把实体引用带出去，调用方用完 addr 之前它不会被释放
    keepAlive = cpuThread;
    return HCCL_SUCCESS;
}

HcclResult CpuThreadService::GetDpuDevId(uint32_t& devId) const
{
    devId = UINT32_MAX;
    std::shared_ptr<CpuThread> cpuThread;
    HcclResult ret = threadMgr_->GetCpuThread(cpuThread);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    auto* legacy = cpuThread->GetLegacy();
    if (legacy == nullptr) {
        return HCCL_E_INTERNAL;
    }
    devId = legacy->GetDevId();
    return HCCL_SUCCESS;
}

} // namespace hccl
