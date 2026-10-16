/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cpu_thread.h"
#include <new>
#include "log.h"

namespace hccl {

CpuThread::CpuThread() : legacy_(new(std::nothrow) LegacyDpuThread()) {}

CpuThread::~CpuThread() = default;

// Init 不转发 legacy：dpu 专属三步（共享内存→TaskService→kernel 下发）是带 commId/devId 入参的一次性动作，
// 由触发点显式经 LegacyDpuThread::Init() 拉起；本类的 Thread 协议 Init 留作后续兼容性扩展，当前无通用资源需初始化。
HcclResult CpuThread::Init() { return HCCL_SUCCESS; }

// dpu 专属 teardown 已收敛进 LegacyDpuThread 析构（ThreadMgr 析构 → HcommThreadFree → ~LegacyDpuThread 触发），
// 本 Thread 协议入口留作后续兼容性扩展，当前无通用资源需释放
HcclResult CpuThread::DeInit() { return HCCL_SUCCESS; }

std::string& CpuThread::GetUniqueId() { return uniqueIdStr_; }

/**
 * @note 恒为 0：HOST CPU 线程不申请任何 local notify 资源，其同步走共享内存任务队列。
 *       返回真实值而非「请求值」是有意的——报一个拿不到的目标值会让调用方
 *       按该数量去取 notify 并拿到空指针，0 与 GetNotify 恒返回 nullptr 自洽
 */
uint32_t CpuThread::GetNotifyNum() const { return 0; }

LocalNotify* CpuThread::GetNotify([[maybe_unused]] uint32_t index) const
{
    // 本类无 notify 资源，任何 index 都取不到；不用 ERROR 级别，避免污染公共入口的正常日志
    HCCL_WARNING("[CpuThread][%s] host cpu thread has no local notify, index[%u]", __func__, index);
    return nullptr;
}

HcclResult CpuThread::SupplementNotify([[maybe_unused]] uint32_t notifyNum)
{
    // 无 notify 资源可补，按幂等空操作处理；返回 NOT_SUPPORT 会被 SupplementThreadNotify 当错误上抛
    HCCL_INFO("[CpuThread][%s] no local notify to supplement, notifyNum[%u], treat as no-op", __func__, notifyNum);
    return HCCL_SUCCESS;
}

bool CpuThread::IsDeviceA5() const
{
    if (legacy_ == nullptr) {
        return false;
    }
    const DevType devType = legacy_->GetDevType();
    return devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960;
}

Stream* CpuThread::GetStream() const { return legacy_ == nullptr ? nullptr : legacy_->GetStream(); }

void* CpuThread::GetStreamLitePtr() const { return nullptr; }

void CpuThread::LaunchTask() const { return; }

void CpuThread::TryLaunchTask() const { return; }

// 数据面走共享内存任务队列，不提供 local notify/copy/reduce 语义
HcclResult CpuThread::LocalNotifyRecord([[maybe_unused]] uint32_t notifyId) const
{
    HCCL_ERROR("[CpuThread][%s] not supported on host cpu thread", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CpuThread::LocalNotifyWait([[maybe_unused]] uint32_t notifyId) const
{
    HCCL_ERROR("[CpuThread][%s] not supported on host cpu thread", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
CpuThread::LocalNotifyRecord([[maybe_unused]] ThreadHandle dstThread, [[maybe_unused]] uint32_t dstNotifyIdx) const
{
    HCCL_ERROR("[CpuThread][%s] not supported on host cpu thread", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CpuThread::LocalNotifyWait([[maybe_unused]] uint32_t notifyIdx, [[maybe_unused]] uint32_t timeOut) const
{
    HCCL_ERROR("[CpuThread][%s] not supported on host cpu thread", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CpuThread::LocalCopy(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t sizeByte) const
{
    HCCL_ERROR("[CpuThread][%s] not supported on host cpu thread", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CpuThread::LocalReduce(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t sizeByte,
    [[maybe_unused]] HcommDataType dataType, [[maybe_unused]] HcommReduceOp reduceOp) const
{
    HCCL_ERROR("[CpuThread][%s] not supported on host cpu thread", __func__);
    return HCCL_E_NOT_SUPPORT;
}

bool CpuThread::GetMaster() const { return isMaster_; }

void CpuThread::SetIsMaster(bool isMaster) { isMaster_ = isMaster; }

} // namespace hccl
