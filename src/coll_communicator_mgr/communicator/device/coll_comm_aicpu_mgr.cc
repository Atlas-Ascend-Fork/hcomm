/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_aicpu_mgr.h"
#include "ns_recovery/aicpu/ns_recovery_func_lite.h"
#include "rtsq_poll/aicpu/coll_rtsq_poll_completion_daemon.h"
#include "aicpu_daemon_service.h"
#include "hcclCommTaskExceptionLite.h"
#include "stream_task_monitor.h"
#include "coll_comm_aicpu_destroy_func.h"
#include "aicpu_indop_env.h"
#include "config_plf_log_v2.h"
#include "dlhal_function_v2.h"
#include "profiling_command_handle_lite.h"
#include "adapter_hal_pub.h"
#include "log.h"
#include <chrono>
#include <unistd.h>

thread_local CollCommAicpu* CollCommAicpuMgr::currentComm_ = nullptr;

CollCommAicpuMgr& CollCommAicpuMgr::GetInstance()
{
    static CollCommAicpuMgr instance;
    return instance;
}

// ==================== 通信域初始化 ====================

HcclResult CollCommAicpuMgr::InitComm(CommAicpuParam* commAicpuParam)
{
    CHK_PTR_NULL(commAicpuParam);

    std::string group = commAicpuParam->hcomId;
    CollCommAicpu* aicpuComm = nullptr;
    CHK_RET(AcquireAndCreateComm(group, &aicpuComm));
    if (aicpuComm == nullptr) {
        HCCL_ERROR("[CollCommAicpuMgr][InitComm] aicpuComm is null group[%s]", group.c_str());
        return HCCL_E_PTR;
    }

    HcclResult ret = aicpuComm->InitAicpuIndOp(commAicpuParam);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollCommAicpuMgr][%s]errNo[0x%016llx] Failed to init independent op comm group[%s]", __func__,
            HCCL_ERROR_CODE(ret), group.c_str()),
        ret);

    // 全局环境初始化 (call_once 保证只执行一次)
    // 注意：必须先执行 InitIndopEnv（设置 taskMonitorInterval 等环境变量），
    // 再执行 InitBackGroundThread（拉起背景线程并注册 DaemonFunc）。
    // 否则背景线程启动时 taskMonitorInterval 尚未赋值，首轮 Call() 会因
    // IsNoNeedMonitor() 返回 true 而跳过监控（后续不会自愈，因为 call_once 不再执行）。
    static std::once_flag initEnv;
    std::call_once(initEnv, [commAicpuParam, this]() {
        this->InitIndopEnv(commAicpuParam);
    });

    static std::once_flag initBackGround;
    std::call_once(initBackGround, [aicpuComm, this]() {
        this->InitBackGroundThread(aicpuComm->GetDevId());
    });

    return HCCL_SUCCESS;
}

HcclResult CollCommAicpuMgr::AcquireAndCreateComm(const std::string& group, CollCommAicpu** outComm)
{
    std::unique_lock<std::shared_mutex> rwlock(commMapMutex_);
    auto iter = commMap_.find(group);
    if (iter != commMap_.end()) {
        // 已存在 — 确保 CollCommAicpu 已创建
        if (iter->second.comm == nullptr) {
            EXCEPTION_CATCH(iter->second.comm = std::make_unique<CollCommAicpu>(), return HCCL_E_PTR);
        }
        *outComm = iter->second.comm.get();
        HCCL_INFO("[%s]Reuse existing comm group [%s]", __func__, group.c_str());
        return HCCL_SUCCESS;
    }

    // 未找到则创建新实例
    CommEntry entry;
    EXCEPTION_CATCH(entry.comm = std::make_unique<CollCommAicpu>(), return HCCL_E_PTR);

    *outComm = entry.comm.get();
    commMap_.insert({group, std::move(entry)});
    HCCL_RUN_INFO("[%s]Created new comm group [%s]", __func__, group.c_str());
    return HCCL_SUCCESS;
}

// ==================== 通信域注册表操作 ====================

CollCommAicpu* CollCommAicpuMgr::AcquireCommForUse(const std::string& group)
{
    HCCL_INFO("[CollCommAicpuMgr][%s]start, group[%s]", __func__, group.c_str());
    auto startTime = std::chrono::steady_clock::now();
    constexpr u32 pollIntervalUs = 10;
    constexpr u32 pollTimeoutMs = 10000;
    auto waitPollTimeOutMs = std::chrono::milliseconds(pollTimeoutMs);

    while (true) {
        std::unique_lock<std::shared_mutex> rwlock(commMapMutex_);
        auto iter = commMap_.find(group);
        if (iter == commMap_.end()) {
            HCCL_ERROR(
                "[CollCommAicpuMgr][%s] group[%s] not found, exist size[%zu]", __func__, group.c_str(),
                commMap_.size());
            auto curIter = commMap_.begin();
            while (curIter != commMap_.end()) {
                HCCL_ERROR("[CollCommAicpuMgr][%s] exist group [%s]", __func__, curIter->first.c_str());
                curIter++;
            }
            return nullptr;
        }

        if (!iter->second.useMtx->try_lock()) {
            auto curTime = std::chrono::steady_clock::now();
            if ((curTime - startTime) >= waitPollTimeOutMs) {
                startTime = curTime;
                HCCL_RUN_INFO("[CollCommAicpuMgr][%s]wait, comm group [%s] has been used", __func__, group.c_str());
            }
            rwlock.unlock();
            usleep(pollIntervalUs);
            continue;
        }
        iter->second.isLocked = true;
        currentComm_ = iter->second.comm.get();
        HCCL_INFO("[CollCommAicpuMgr][%s]success, group[%s]", __func__, group.c_str());
        return iter->second.comm.get();
    }
}

void CollCommAicpuMgr::ReleaseComm(const std::string& group)
{
    std::unique_lock<std::shared_mutex> rwlock(commMapMutex_);
    auto iter = commMap_.find(group);
    if (iter == commMap_.end()) {
        return;
    }
    currentComm_ = nullptr;
    if (iter->second.isLocked) {
        iter->second.isLocked = false;
        iter->second.useMtx->unlock();
    }
}

CollCommAicpu* CollCommAicpuMgr::FindCommByGroup(const std::string& group)
{
    std::shared_lock<std::shared_mutex> lock(commMapMutex_);
    auto iter = commMap_.find(group);
    if (iter == commMap_.end()) {
        return nullptr;
    }
    return iter->second.comm.get();
}

CollCommAicpu* CollCommAicpuMgr::GetCurrentComm(const std::string& group)
{
    if (group.empty()) {
        HCCL_ERROR("[CollCommAicpuMgr][%s] comm group is empty", __func__);
        return nullptr;
    }
    if (currentComm_ == nullptr) {
        HCCL_ERROR("[CollCommAicpuMgr][%s] currentComm_ is nullptr", __func__);
        return nullptr;
    }
    if (currentComm_->GetIdentifier() != group) {
        HCCL_ERROR("[CollCommAicpuMgr][%s] comm group[%s] is not current comm group", __func__, group.c_str());
        return nullptr;
    }
    return currentComm_;
}

HcclResult CollCommAicpuMgr::DestroyComm(const std::string& group)
{
    std::unique_lock<std::shared_mutex> rwlock(commMapMutex_);
    auto iter = commMap_.find(group);
    if (iter == commMap_.end()) {
        HCCL_ERROR("[CollCommAicpuMgr][%s]group[%s] is not exist", __func__, group.c_str());
        return HCCL_E_PARA;
    }

    CollCommAicpu* aicpuComm = iter->second.comm.get();
    CHK_PTR_NULL(aicpuComm);
    aicpuComm->SetCommmStatus(HcclCommStatus::HCCL_COMM_STATUS_INVALID);

    // 正在使用中，不销毁，返回重试状态让调用方稍后再试
    if (iter->second.isLocked) {
        HCCL_RUN_WARNING("[CollCommAicpuMgr][%s]comm group [%s] has been used, skip erase", __func__, group.c_str());
        return HCCL_E_AGAIN;
    }

    // 防御性检查 legacy 通信域 busy 标记，避免 isLocked 与 legacy busy 不同步时误销毁
    if (aicpuComm->GetLegacy910CollComm() != nullptr && aicpuComm->IsLegacy910CollCommBusy()) {
        HCCL_RUN_WARNING("[CollCommAicpuMgr][%s]legacy comm group [%s] is busy, skip erase", __func__, group.c_str());
        return HCCL_E_AGAIN;
    }

    // 登记表条目存的是本通信域 taskexception 共享内存的 device VA，而那块内存由宿主侧 LegacyDpuThread 持有、
    // 随宿主通信域销毁释放。条目须与通信域对象同生共死，否则同名通信域重建时 Init 的 Find 会取到已释放地址；
    // 放在这里而不是销毁命令处理里，是为了让 busy 跳过销毁时条目也一并保留（通信域还在，地址就还得有效）
    taskExpDevMemRegistry_.Erase(group, aicpuComm->GetHcclCommDfxLite()->GetTaskExpDevMem());

    commMap_.erase(group);
    HCCL_RUN_INFO("[CollCommAicpuMgr][%s]Destroy comm group [%s] success.", __func__, group.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpuMgr::GetAllComms(std::vector<std::pair<std::string, CollCommAicpu*>>& aicpuCommInfo)
{
    // 调用方必须在外部持有 commMapMutex_ 共享锁（保护遍历+访问 comm 成员的完整临界区）
    for (auto& kv : commMap_) {
        aicpuCommInfo.push_back({kv.first, kv.second.comm.get()});
    }
    return HCCL_SUCCESS;
}

std::shared_mutex& CollCommAicpuMgr::GetMutex() { return commMapMutex_; }

// ==================== 全局环境初始化 ====================

void CollCommAicpuMgr::InitIndopEnv(CommAicpuParam* commAicpuParam)
{
    hcomm::SetTaskExceptionEnable(commAicpuParam->commConfig.taskExceptionEnable);
    Hccl::SetPlfDebugConfigValue(commAicpuParam->commConfig.plfDebugConfig);
    hcomm::StreamTaskMonitor::GetInstance().SetInterval(commAicpuParam->commConfig.taskMonitorInterval);
    hcomm::StreamTaskMonitor::GetInstance().SetTaskExceptionEnable(commAicpuParam->commConfig.taskExceptionEnable);
    HCCL_RUN_INFO(
        "[%s]Env: taskExceptionEnable[%d], notifyWaitTimeout[%u s], plfDebugConfig[0x%llx], "
        "taskMonitorInterval[%u ms]",
        __func__, commAicpuParam->commConfig.taskExceptionEnable, commAicpuParam->commConfig.notifyWaitTimeout,
        commAicpuParam->commConfig.plfDebugConfig, commAicpuParam->commConfig.taskMonitorInterval);
}

void CollCommAicpuMgr::InitBackGroundThread(u32 devId)
{
    static auto commandToBackGroud = Hccl::CommandToBackGroud::Default;
    static auto daemonServiceRun = [](void* info) {
        Hccl::AicpuDaemonService::GetInstance().ServiceRun(info);
    };
    static auto daemonServiceStop = [](void* info) {
        Hccl::AicpuDaemonService::GetInstance().ServiceStop(info);
    };

    hcomm::HcclCommTaskExceptionLite::GetInstance().Init(devId);
    Hccl::AicpuDaemonService::GetInstance().Register(&hcomm::HcclCommTaskExceptionLite::GetInstance());
    Hccl::AicpuDaemonService::GetInstance().Register(&hccl::CollCommAicpuDestroyFunc::GetInstance());
    Hccl::AicpuDaemonService::GetInstance().Register(&hccl::NsRecoveryFuncLite::GetInstance());
    Hccl::AicpuDaemonService::GetInstance().Register(&hcomm::CollRtsqPollCompletionDaemon::GetInstance());
    hcomm::StreamTaskMonitor::GetInstance().Init(devId);
    Hccl::AicpuDaemonService::GetInstance().Register(&hcomm::StreamTaskMonitor::GetInstance());

    if (Hccl::StartMC2MaintenanceThread != nullptr) {
        Hccl::StartMC2MaintenanceThread(daemonServiceRun, &commandToBackGroud, daemonServiceStop, &commandToBackGroud);
        HCCL_RUN_INFO("[%s]start BackGround thread success.", __func__);
    } else {
        HCCL_WARNING("[%s]StartMC2MaintenanceThread func is nullptr", __func__);
    }
}
