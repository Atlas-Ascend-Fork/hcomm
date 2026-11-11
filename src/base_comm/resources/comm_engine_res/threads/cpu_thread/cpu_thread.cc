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
#include <chrono>
#include <exception>
#include <memory>
#include <new>
#include <thread>
#include "cpu_queue_kernel_entrance.h"
#include "hccl_common.h"
#include "securec.h"
#include "adapter_rts.h"
#include "orion_adapter_rts.h"
#include "log.h"

// 命名空间提醒：带 `Hccl::` 前缀的是 dpu_types.h 系列（大写命名空间），
// 不加前缀的属本命名空间 hccl（队列契约头 dpu_sync_layout.h/cpu_task_pipe.h）
namespace hccl {

constexpr char DPU_NOTIFY_TAG[] = "DPUNOTIFY";

std::mutex CpuThread::xpuLaunchMutex_;

CpuThread::CpuThread(uint32_t notifyNum) : legacy_(new(std::nothrow) LegacyDpuThread()), notifyNum_(notifyNum) {}

CpuThread::~CpuThread()
{
    // 派生析构先于基类析构执行：本类必须先终止 kernel 再收队列管道，基类析构随后才释放 DPUTAG
    (void)DeInit();
    ResetQueuePipe();
}

HcclResult CpuThread::Init()
{
    // 只有 (AICPU, CPU) 这一档走 SQ/CQ 队列通道；TaskService 通道由触发点显式经 InitLegacy 拉起
    if (GetCommEngine() != COMM_ENGINE_AICPU || GetThreadType() != THREAD_TYPE_CPU) {
        return HCCL_SUCCESS;
    }
    const s32 devId = Hccl::HrtGetDevice();
    if (devId < 0) {
        HCCL_ERROR("[CpuThread][%s] get current device failed, ret[%d]", __func__, devId);
        return HCCL_E_RUNTIME;
    }
    Hccl::TaskServiceArgs args{};
    args.devId = static_cast<uint32_t>(devId);
    return InitQueueMode(args);
}

// 1 thread 1 kernel：只能有一条通道的 kernel 跑在同一条 dpu 流上，判据落在同时看得见两个实体之处
HcclResult CpuThread::InitLegacy(const Hccl::TaskServiceArgs& args)
{
    if (legacy_ == nullptr) {
        HCCL_ERROR("[CpuThread][%s] legacy dpu thread is nullptr", __func__);
        return HCCL_E_INTERNAL;
    }
    if (dpuTaskPipe_ != nullptr) {
        HCCL_ERROR("[CpuThread][%s] thread is occupied by queue mode kernel, not supported", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    std::lock_guard<std::mutex> lock(xpuLaunchMutex_);
    legacy_->SetTaskServiceArgs(args);
    return legacy_->Init();
}

HcclResult CpuThread::InitQueueMode(const Hccl::TaskServiceArgs& args)
{
    if (legacy_ != nullptr && (legacy_->IsKernelLaunched() || legacy_->GetTaskService() != nullptr)) {
        HCCL_ERROR("[CpuThread][%s] thread is occupied by legacy kernel, not supported", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    // 幂等判据是本通道自己的产物：dpuTaskPipe_ 已在即本入口已拉起
    if (dpuTaskPipe_ != nullptr) {
        HCCL_INFO("[CpuThread][%s] queue mode already inited, skip", __func__);
        return HCCL_SUCCESS;
    }

    std::lock_guard<std::mutex> lock(xpuLaunchMutex_);
    try {
        SetTaskServiceArgs(args);

        const HcclResult initRet = [&]() -> HcclResult {
            CHK_RET(hrtGetDeviceType(devType_));
            CHK_RET(InitQueueShmem());
            CHK_RET(InitNotifyShmem());
            CHK_RET(CreateCpuTaskPipe());
            return PrepareAndLaunchQueueKernel();
        }();
        if (initRet != HCCL_SUCCESS) {
            // kernel 已下发（如切回 npu ctx 失败）时管道与共享内存仍在被轮询，须留给析构链
            // DeInit → WaitKernelTerminate 先等退出再收口，否则释放正被使用的内存造成 UAF
            if (!isKernelLaunched_) {
                // 管道与已申请的共享内存须一并收回：DpuShmemMgr::Alloc 对已存在的 tag 返回 HCCL_E_PARA，
                // 只释放pipe会让同一对象重试在 Alloc 处失败
                ResetQueueModeResources();
            }
            return initRet;
        }
        return HCCL_SUCCESS;
    } catch (const std::exception& e) {
        if (!isKernelLaunched_) {
            ResetQueueModeResources();
        }
        HCCL_ERROR("[CpuThread][%s] exception caught: %s", __func__, e.what());
        return HCCL_E_INTERNAL;
    } catch (...) {
        if (!isKernelLaunched_) {
            ResetQueueModeResources();
        }
        HCCL_ERROR("[CpuThread][%s] unknown exception caught", __func__);
        return HCCL_E_INTERNAL;
    }
}

HcclResult CpuThread::PrepareAndLaunchQueueKernel()
{
    // 切 dpu 域（整段 XPU 切换序列在基类，失败时基类自行还原 ctx）
    CHK_RET(EnterDpuContext());
    CpuCtxRollback rollback(npuContext_);
    CpuStreamRollback streamRollback(rtStream_, isKernelLaunched_, dpuContext_, npuContext_, stream_);

    // 入口名必须按队列轮询取：句柄是按名解析的，取错名等于把 TaskService 的 kernel 当队列轮询下发
    CHK_RET(PrepareDpuKernelResource(DPU_QUEUE_KERNEL_ENTRY_NAME));

    stream_.reset(new (std::nothrow) Stream(rtStream_));
    CHK_SMART_PTR_NULL(stream_);
    // 回执的 stream id 用本线程真实流：执行段写 CQE 时取这里
    dpuTaskPipe_->SetDfxStreamId(static_cast<uint32_t>(stream_->id()));

    CHK_RET(LaunchDpuQueueKernel());

    HCCL_INFO("[CpuThread][%s] end, rtStream[%p]", __func__, rtStream_);
    return HCCL_SUCCESS;
}

HcclResult CpuThread::InitQueueShmem()
{
    // 只申请 DPUTAG（基类公共机制）：不申请 DPUTASKEXCEPTION ——
    // device 侧没有 AICPU taskexception 监控线程，申请了也没有人回写停止标志，停止握手会白等满 10s
    int32_t deviceLogicId = 0;
    CHK_RET(ResolveDeviceLogicId(deviceLogicId));
    // 只申请队列所需的 256KB，尺寸由队列契约头自带常量给出
    CHK_RET(AllocDputagShmem(deviceLogicId, DPU_TAG_QUEUE_BYTES));

    // 地址一律显式给全、消费侧不做偏移算术：半区布局约定只在这里出现一次
    const uint64_t sqHalf = reinterpret_cast<uint64_t>(dputagAddr_);
    const uint64_t cqHalf = sqHalf + DPU_TAG_HALF_BYTES;
    queueLayout_.sqMeta = sqHalf + DPU_SQ_META_OFF;
    queueLayout_.sqElem = sqHalf + DPU_SQ_ELEM_OFF;
    queueLayout_.cqMeta = cqHalf + DPU_CQ_META_OFF;
    queueLayout_.cqElem = cqHalf + DPU_CQ_ELEM_OFF;
    queueLayout_.sqDepth = DPU_SQ_DEPTH;
    queueLayout_.cqDepth = DPU_CQ_DEPTH;
    queueLayout_.elemSize = DPU_SQE_SIZE;

    HCCL_INFO(
        "[CpuThread][%s] dputag[%p] sqMeta[0x%llx] sqElem[0x%llx] cqMeta[0x%llx] cqElem[0x%llx] sqDepth[%u] "
        "cqDepth[%u]",
        __func__, dputagAddr_, static_cast<unsigned long long>(queueLayout_.sqMeta),
        static_cast<unsigned long long>(queueLayout_.sqElem), static_cast<unsigned long long>(queueLayout_.cqMeta),
        static_cast<unsigned long long>(queueLayout_.cqElem), queueLayout_.sqDepth, queueLayout_.cqDepth);
    return HCCL_SUCCESS;
}

HcclResult CpuThread::InitNotifyShmem()
{
    if (notifyNum_ == 0) {
        return HCCL_SUCCESS;
    }
    int32_t deviceLogicId = 0;
    CHK_RET(ResolveDeviceLogicId(deviceLogicId));
    CHK_RET(shmemMgr_.Alloc(DPU_NOTIFY_TAG, notifyNum_, deviceLogicId));
    notifyAddr_ = static_cast<uint8_t*>(shmemMgr_.HostPtr(DPU_NOTIFY_TAG));
    CHK_PTR_NULL(notifyAddr_);
    errno_t memRet = memset_s(notifyAddr_, notifyNum_, 0, notifyNum_);
    if (memRet != EOK) {
        HCCL_ERROR("[CpuThread][%s] memset_s failed, ret[%d]", __func__, memRet);
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

HcclResult CpuThread::CreateCpuTaskPipe()
{
    // 实例归 host 侧持有（与 TaskService 同口径）：kernel 只取裸指针，故 Init 必须在 launch 之前完成
    dpuTaskPipe_ = std::unique_ptr<CpuTaskPipe>(new (std::nothrow) CpuTaskPipe());
    CHK_SMART_PTR_NULL(dpuTaskPipe_);
    CHK_RET(dpuTaskPipe_->Init(queueLayout_));
    dpuTaskPipe_->SetNotifyInfo(notifyNum_, notifyAddr_);
    dpuTaskPipe_->SetDevId(args_.devId);
    return HCCL_SUCCESS;
}

HcclResult CpuThread::LaunchDpuQueueKernel()
{
    if (isKernelLaunched_) {
        HCCL_INFO("[CpuThread][%s] kernel already launched, skip", __func__);
        return HCCL_SUCCESS;
    }
    if (dpuTaskPipe_ == nullptr) {
        HCCL_ERROR("[CpuThread][%s] dpuTaskPipe is nullptr, call CreateCpuTaskPipe first", __func__);
        return HCCL_E_INTERNAL;
    }

    DpuQueueLaunchParam hostArgs{dpuTaskPipe_.get(), args_.devId, 0};
    return LaunchDpuKernelWithArgs(&hostArgs, sizeof(hostArgs));
}

HcclResult CpuThread::GetDevSyncLayout(
    DpuSyncLayout& layout, QueueDescriptor& sqDesc, QueueDescriptor& cqDesc, uint64_t& notifyHostAddr,
    uint64_t& notifyDevAddr) const
{
    if (dpuTaskPipe_ == nullptr) {
        HCCL_ERROR("[CpuThread][%s] queue mode is not inited", __func__);
        return HCCL_E_INTERNAL;
    }
    const void* devBase = shmemMgr_.DevPtr(Hccl::DPU_TAG);
    if (devBase == nullptr || dputagAddr_ == nullptr) {
        HCCL_ERROR("[CpuThread][%s] dputag addr is null", __func__);
        return HCCL_E_PTR;
    }
    if (notifyNum_ > 0) {
        CHK_RET(GetNotifyShmemAddrs(notifyHostAddr, notifyDevAddr));
    } else {
        notifyHostAddr = 0;
        notifyDevAddr = 0;
    }

    // 同一块内存在两个地址空间里的相对偏移一致，故 device 地址 = device 基址 + (host 地址 - host 基址)
    const uint64_t hostBase = reinterpret_cast<uint64_t>(dputagAddr_);
    const uint64_t devBaseVal = reinterpret_cast<uint64_t>(devBase);
    const auto toDev = [devBaseVal, hostBase](uint64_t hostAddr) {
        return devBaseVal + (hostAddr - hostBase);
    };

    layout.sqMeta = toDev(queueLayout_.sqMeta);
    layout.sqElem = toDev(queueLayout_.sqElem);
    layout.cqMeta = toDev(queueLayout_.cqMeta);
    layout.cqElem = toDev(queueLayout_.cqElem);
    layout.sqDepth = queueLayout_.sqDepth;
    layout.cqDepth = queueLayout_.cqDepth;
    layout.elemSize = queueLayout_.elemSize;

    // SQ 只承载任务（无回执）：终止走 destroyFlag_、出错走 CQE；格宽取 SQE 语义
    sqDesc.base = layout.sqElem;
    sqDesc.depth = queueLayout_.sqDepth;
    sqDesc.elemSize = DPU_SQE_SIZE;
    sqDesc.piAddr = layout.sqMeta + offsetof(DpuSqMeta, pi);
    sqDesc.ciAddr = layout.sqMeta + offsetof(DpuSqMeta, ci);

    cqDesc.base = layout.cqElem;
    cqDesc.depth = queueLayout_.cqDepth;
    cqDesc.elemSize = DPU_CQE_SIZE; // CQ 侧的格宽与 SQE 同值，但语义上取 CQE
    cqDesc.piAddr = layout.cqMeta + offsetof(DpuCqMeta, pi);
    cqDesc.ciAddr = layout.cqMeta + offsetof(DpuCqMeta, ci);

    HCCL_INFO(
        "[CpuThread][%s] devBase[0x%llx] sqElem[0x%llx] cqElem[0x%llx]", __func__,
        static_cast<unsigned long long>(devBaseVal), static_cast<unsigned long long>(sqDesc.base),
        static_cast<unsigned long long>(cqDesc.base));
    return HCCL_SUCCESS;
}

void CpuThread::ResetQueuePipe() { dpuTaskPipe_.reset(); }

void CpuThread::ResetQueueModeResources()
{
    ResetQueuePipe();
    notifyAddr_ = nullptr;
    queueLayout_ = DpuSyncLayout{};
    int32_t deviceLogicId = 0;
    if (ResolveDeviceLogicId(deviceLogicId) != HCCL_SUCCESS) {
        return; // 设备逻辑 id 不可解析时交由析构统一释放
    }
    shmemMgr_.Free(DPU_NOTIFY_TAG, deviceLogicId);
    shmemMgr_.Free(Hccl::DPU_TAG, deviceLogicId);
    dputagAddr_ = nullptr;
    dputagDevAddr_ = nullptr;
    dputagHostPtr_ = nullptr;
}

HcclResult CpuThread::GetNotifyShmemAddrs(uint64_t& notifyHostAddr, uint64_t& notifyDevAddr) const
{
    const void* notifyDevBase = shmemMgr_.DevPtr(DPU_NOTIFY_TAG);
    if (notifyDevBase == nullptr || notifyAddr_ == nullptr) {
        HCCL_ERROR("[CpuThread][%s] notify shmem addr is null", __func__);
        return HCCL_E_PTR;
    }
    notifyHostAddr = reinterpret_cast<uint64_t>(notifyAddr_);
    notifyDevAddr = reinterpret_cast<uint64_t>(notifyDevBase);
    return HCCL_SUCCESS;
}

HcclResult CpuThread::WaitKernelTerminate()
{
    if (dpuTaskPipe_ == nullptr) {
        // 队列通道的终止只靠 destroyFlag_，无共享内存握手可等
        HCCL_ERROR("[CpuThread][%s] dpuTaskPipe is nullptr, nothing to wait", __func__);
        return HCCL_E_INTERNAL;
    }
    dpuTaskPipe_->RequestDestroy();

    // 等 kernel 写回 destroyFlag_（已退出）：与 TaskService 通道同口径，不设上限
    while (!dpuTaskPipe_->IsDestroyAcked()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return HCCL_SUCCESS;
}

// 覆盖两处同名 DeInit：legacy 实体已拉起则终止 TaskService 通道的 kernel，否则终止本类队列的 kernel，两侧各自幂等
HcclResult CpuThread::DeInit()
{
    if (legacy_ != nullptr && legacy_->IsKernelLaunched()) {
        return legacy_->DeInit();
    }
    return CpuThreadBase::DeInit();
}

std::string& CpuThread::GetUniqueId() { return uniqueIdStr_; }

uint32_t CpuThread::GetNotifyNum() const { return notifyNum_; }

HcclResult CpuThread::PreAllocThreadEntity(ThreadHandle& handle, CommEngine targetEngine)
{
    return PreAllocThreadEntityCommon(*this, threadHandleMap_, handle, targetEngine);
}

size_t CpuThread::GetNotifySize() const { return 0; }

LocalNotify* CpuThread::GetNotify([[maybe_unused]] uint32_t index) const { return nullptr; }

HcclResult CpuThread::SupplementNotify(uint32_t notifyNum)
{
    if (notifyNum == 0) {
        return HCCL_SUCCESS;
    }
    const uint32_t newNum = notifyNum_ + notifyNum;
    if (dpuTaskPipe_ == nullptr) {
        notifyNum_ = newNum;
        return HCCL_SUCCESS;
    }

    int32_t deviceLogicId = 0;
    CHK_RET(ResolveDeviceLogicId(deviceLogicId));
    dpuTaskPipe_->SetNotifyInfo(0U, nullptr);
    notifyAddr_ = nullptr;
    shmemMgr_.Free(DPU_NOTIFY_TAG, deviceLogicId);
    const HcclResult reallocRet = [&]() -> HcclResult {
        try {
            CHK_RET(shmemMgr_.Alloc(DPU_NOTIFY_TAG, newNum, deviceLogicId));
            notifyAddr_ = static_cast<uint8_t*>(shmemMgr_.HostPtr(DPU_NOTIFY_TAG));
            CHK_PTR_NULL(notifyAddr_);
            errno_t memRet = memset_s(notifyAddr_, newNum, 0, newNum);
            if (memRet != EOK) {
                HCCL_ERROR("[CpuThread][%s] memset_s failed, ret[%d]", __func__, memRet);
                return HCCL_E_MEMORY;
            }
            return HCCL_SUCCESS;
        } catch (const std::exception& e) {
            HCCL_ERROR("[CpuThread][%s] exception caught: %s", __func__, e.what());
            return HCCL_E_INTERNAL;
        } catch (...) {
            HCCL_ERROR("[CpuThread][%s] unknown exception caught", __func__);
            return HCCL_E_INTERNAL;
        }
    }();
    if (reallocRet != HCCL_SUCCESS) {
        shmemMgr_.Free(DPU_NOTIFY_TAG, deviceLogicId);
        notifyAddr_ = nullptr;
        return reallocRet;
    }
    notifyNum_ = newNum;
    dpuTaskPipe_->SetNotifyInfo(notifyNum_, notifyAddr_);
    HCCL_INFO("[CpuThread][%s] notify shmem reallocated, notifyNum[%u]", __func__, notifyNum_);
    return HCCL_SUCCESS;
}

bool CpuThread::IsDeviceA5() const
{
    // 两条通道的 devType_ 各由自己的拉起入口填入，未拉起的一侧保持 DEV_TYPE_COUNT（初值）
    DevType devType = legacy_ != nullptr ? legacy_->GetDevType() : DevType::DEV_TYPE_COUNT;
    if (devType == DevType::DEV_TYPE_COUNT) {
        devType = CpuThreadBase::GetDevType();
    }
    return devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960;
}

Stream* CpuThread::GetStream() const
{
    // TaskService 通道拉起时取 legacy 的流，否则取本类经基类持有的 dpu 流
    if (legacy_ != nullptr && legacy_->GetStream() != nullptr) {
        return legacy_->GetStream();
    }
    return CpuThreadBase::GetStream();
}

s32 CpuThread::GetStreamId() const
{
    // 流与 id 必须同源，复用 GetStream 的回退链避免两处口径漂移
    Stream* stream = GetStream();
    return stream != nullptr ? stream->id() : -1;
}

void* CpuThread::GetStreamLitePtr() const { return nullptr; }

void CpuThread::LaunchTask() const { return; }

void CpuThread::TryLaunchTask() const { return; }

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
