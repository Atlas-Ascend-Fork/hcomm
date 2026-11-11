/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/cast_utils_pub.h"
#include "common/device/plf_resource_runtime_access.h"
#include "dfx_compact_report_pub.h"
#include "dfx/device/dfx_res_pub.h"
#include <chrono>
#include <unordered_map>
#include "rtsq_a5.h"
#include "log.h"
#include "exception_util.h"
#include "internal_exception.h"
#include "sqe_build_a5.h"
#include "sqe.h"
#include "ub_conn_lite_mgr.h"
#ifdef CCL_KERNEL_AICPU
#include "aicpu_ts_primitives_c_adpt.h"
#endif
#include "aicpu_task_utils.h"
#include "aicpu_ts_thread.h"
#include "stream_lite.h"

namespace Hccl {
using namespace std;
constexpr u32 RTSQ_A5_PART_ID = 0;
constexpr u32 PRINT_INTERVAL = 30;
constexpr u32 POLL_SNAPSHOT_RESERVE = 256;
constexpr u32 DB_SEND_DEPTH_MULTIPLE = 8;
// head追踪空间上限: u32内sqDepth的最大整数倍再减一个sqDepth余量(约42.9亿):
constexpr u32 BaseSqHeadMaxSize(u32 sqDepth) { return (UINT32_MAX / sqDepth) * sqDepth - sqDepth; }

RtsqA5::RtsqA5(u32 devPhyId, u32 streamId, u32 sqId) : RtsqBase(devPhyId, streamId, sqId)
{
    InitSendTrackingState();
    SetTaskIdBySqeId();
}

RtsqA5::RtsqA5(u32 devPhyId, u32 streamId, u32 sqId, bool launchFlag) : RtsqBase(devPhyId, streamId, sqId)
{
    InitSendTrackingState();
    SetTaskIdBySqeId();
    launchFlag_ = launchFlag;
}

void RtsqA5::Reset(bool reset)
{
    RtsqBase::Reset(reset);
    InitSendTrackingState();
    dbSendHead_ = 0;
    dbSendTail_ = 0;
    pendingSqeCnt = 0;
    s32 sRet = memset_s(locBuf, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT, 0, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT);
    if (UNLIKELY(sRet != EOK)) {
        auto msg = StringFormat("[RtsqA5][Reset] locBuf memset fail. errorno[%d]", sRet);
        THROW<InternalException>(msg);
    }
    currentTransport_ = DFX_INVALID_U64;
    currentLinkType_ = static_cast<u8>(DfxLinkTypeVal::LINK_ONCHIP);
    currentTransportType_ = static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL);
    HCCL_INFO("[NsRecovery]RtsqA5::%s success", __func__);
}

void RtsqA5::InitSendTrackingState()
{
    dbSendDepth_ = sqDepth_ * DB_SEND_DEPTH_MULTIPLE;
    dbSendSlots_.assign(dbSendDepth_, DbSendSlotMeta{});
    pollSnapshot_.reserve(POLL_SNAPSHOT_RESERVE);
    lastHead_ = sqHead_;
    baseSqHead_ = sqHead_;
    baseSqHeadMaxSize_ = BaseSqHeadMaxSize(sqDepth_);
    sqTailTotal_ = baseSqHead_ + (sqTail_ + sqDepth_ - sqHead_) % sqDepth_;
}

// 计算head和tail之间的距离
u32 RtsqA5::GetTailToHeadDist() const
{
    if (UNLIKELY(sqHead_ == sqTail_)) { // 头尾相同，则距离大小为sq深度
        return sqDepth_;
    }
    return (sqTail_ < sqHead_) ? (sqHead_ - sqTail_) : (sqDepth_ - (sqTail_ - sqHead_));
}

void RtsqA5::RefreshSqHead()
{
    sqHead_ = QuerySqHead();
    u32 baseSqHead = baseSqHead_;
    u32 sqDepth = sqDepth_;
    u32 baseSqHeadMaxSize = baseSqHeadMaxSize_;

    u32 basePos = baseSqHead % sqDepth;
    u32 step = sqHead_ + sqDepth - basePos;
    if (step >= sqDepth) {
        step -= sqDepth;
    }

    u32 newBase = baseSqHead + step;
    if (newBase >= baseSqHeadMaxSize) {
        newBase -= baseSqHeadMaxSize;
    }
    baseSqHead_ = newBase;
}

// 后台线程只读: 查询硬件head并解回绕, 不写sqHead_避免与发送线程并发写
u32 RtsqA5::ReadUnwrappedSqHead() const
{
    u32 baseSqHead = baseSqHead_; // 这里必须先读baseSqHead_再QuerySqHead
    u32 hwHead = QuerySqHead();
    u32 sqDepth = sqDepth_;
    u32 baseSqHeadMaxSize = baseSqHeadMaxSize_;

    u32 basePos = baseSqHead % sqDepth;
    u32 step = hwHead + sqDepth - basePos;
    if (step >= sqDepth) {
        step -= sqDepth;
    }

    u32 newBase = baseSqHead + step;
    if (newBase >= baseSqHeadMaxSize) {
        newBase -= baseSqHeadMaxSize;
    }
    return newBase;
}

u32 RtsqA5::WrapAdd(u32 value, u32 step) const
{
    u64 sum = static_cast<u64>(value) + step;
    return (sum >= baseSqHeadMaxSize_) ? static_cast<u32>(sum - baseSqHeadMaxSize_) : static_cast<u32>(sum);
}

HcclResult RtsqA5::MakeSureAvailableSpace()
{
    u32 availableSpace = GetTailToHeadDist();
    auto startTime = std::chrono::steady_clock::now();
#ifdef CCL_KERNEL_AICPU
    sqFullTimeout_ = PlfGetSqFullTimeOut();
#endif
    const std::chrono::seconds printInterval(PRINT_INTERVAL); // 打印间隔30s
    auto lastPrintTime = std::chrono::steady_clock::now() - printInterval;
    HCCL_INFO(
        "[%s]sqId:%u, sqFullTimeout_: %u s, sqHead:%u, sqTail:%u, pendingSqeCnt:%u, dbSendHead:%u, dbSendTail:%u",
        __func__, sqId_, sqFullTimeout_, sqHead_, sqTail_, pendingSqeCnt, dbSendHead_, dbSendTail_);

    while (availableSpace <= pendingSqeCnt) {
        RefreshSqHead();
        availableSpace = GetTailToHeadDist();
        if (availableSpace > pendingSqeCnt) {
            break; // 避免head没更新导致假反压
        }

        auto curTime = std::chrono::steady_clock::now();
        if (UNLIKELY(curTime - lastPrintTime >= printInterval)) {
            HCCL_RUN_INFO(
                "[%s]while loop, sqId:%u, sqHead:%u, sqTail:%u, availableSpace:%u, pendingSqeCnt:%u, dbSendHead:%u, "
                "dbSendTail:%u, sqFullTimeout_:%u s",
                __func__, sqId_, sqHead_, sqTail_, availableSpace, pendingSqeCnt, dbSendHead_, dbSendTail_,
                sqFullTimeout_);
            lastPrintTime = curTime;
        }

        TRY_CATCH_RETURN(CheckLaunchTaskStatus(startTime, curTime, "Rtsq full"););
#ifdef CCL_KERNEL_AICPU
        HcclResult ret = PlfHandleDispatchAllStreams();
        if (UNLIKELY(ret != HCCL_SUCCESS)) {
            HCCL_ERROR("[%s] HandleDispatchAllStreams failed, ret = %d, sqId:%u", __func__, ret, sqId_);
            return ret;
        }
#endif
        if (checkOpExecStatusCallback_ != nullptr) {
            checkOpExecStatusCallback_();
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RtsqA5::CheckLaunchTaskStatus(
    const std::chrono::steady_clock::time_point& startTime, const std::chrono::steady_clock::time_point& curTime,
    const char* waitReason)
{
    bool isTimeout = (sqFullTimeout_ == 0) ? false : ((curTime - startTime) >= std::chrono::seconds(sqFullTimeout_));
    // step1 检测是否等待超时，如果超时打印等待原因的ERROR日志(如Rtsq full/dbSendSlots full)
    if (UNLIKELY(isTimeout)) {
        HCCL_ERROR(
            "%s, sqFullTimeout_:[%u s]. sqId:%u, sqHead:%u, sqTail:%u, pendingSqeCnt:%u, dbSendHead:%u, "
            "dbSendTail:%u",
            waitReason, sqFullTimeout_, sqId_, sqHead_, sqTail_, pendingSqeCnt, dbSendHead_, dbSendTail_);
    }

    HcclResult checkRet = (checkExecStatusCallback_ != nullptr) ? checkExecStatusCallback_(isTimeout) : HCCL_SUCCESS;
    // step2 通信域状态为HCCL_COMM_STATUS_SUSPENDING状态，则终止launch不抛异常
    if (UNLIKELY(checkRet == HCCL_E_SUSPENDING)) {
        pendingSqeCnt = 0;
        return HCCL_E_SUSPENDING;
    }
    // step3 调用回调检查执行状态：1、如果超时，打印taskException；2、如果通信域不可用，终止launch
    if (UNLIKELY(isTimeout || checkRet != HCCL_SUCCESS)) {
        THROW<InternalException>(StringFormat(
            "[%s]%s, stop launch Task, isTimeout[%d], checkRet[%d]", __func__, waitReason, isTimeout, checkRet));
    }
    return HCCL_SUCCESS;
}

HcclResult RtsqA5::CopySqeBufToSq(u8* sqeBuf) const
{
    u8* sqCurrAddr = ReinterpretAs<u8*>(sqBaseAddr_) + sqTail_ * RTSQ_SQE_SIZE;
    if (sqTail_ >= sqHead_) {
        u32 depthLeft = sqDepth_ - sqTail_;
        if (pendingSqeCnt <= depthLeft) { // 没有回绕
            HCCL_INFO(
                "RtsqA5::%s copy sqe from sqe buffer, sqId_: %u, streamId_: %u, cur head: %u, cur tail: %u, size: %u, "
                "depth remain: %u",
                __func__, sqId_, streamId_, sqHead_, sqTail_, pendingSqeCnt, depthLeft);
            int ret = memcpy_sp(sqCurrAddr, pendingSqeCnt * AC_SQE_SIZE, sqeBuf, pendingSqeCnt * RTSQ_SQE_SIZE);
            if (UNLIKELY(ret != 0)) {
                HCCL_ERROR("RtsqA5::%s sqe memcpy_sp failed, ret = %d", __func__, ret);
                return HCCL_E_INTERNAL;
            }
        } else {
            HCCL_INFO(
                "RtsqA5::%s copy sqe twice, sqId_: %u, streamId_: %u, cur head: %u, cur tail: %u, cnt: %u, depth "
                "remain: %u",
                __func__, sqId_, streamId_, sqHead_, sqTail_, pendingSqeCnt, depthLeft);
            // 先拷贝rtsq里剩余空间大小
            int ret = memcpy_sp(sqCurrAddr, depthLeft * AC_SQE_SIZE, sqeBuf, depthLeft * RTSQ_SQE_SIZE);
            if (ret != 0) {
                HCCL_ERROR("RtsqA5::%s rtsq remaining space memcpy_sp failed, ret = %d", __func__, ret);
                return HCCL_E_INTERNAL;
            }
            // 拷贝剩余sqe
            ret = memcpy_sp(
                ReinterpretAs<u8*>(sqBaseAddr_), sqHead_ * RTSQ_SQE_SIZE, sqeBuf + depthLeft * RTSQ_SQE_SIZE,
                (pendingSqeCnt - depthLeft) * AC_SQE_SIZE);
            if (UNLIKELY(ret != 0)) {
                HCCL_ERROR("RtsqA5::%s remaining sqe memcpy_sp failed, ret = %d", __func__, ret);
                return HCCL_E_INTERNAL;
            }
        }
    } else {
        HCCL_INFO(
            "RtsqA5::%s copy sqe from sqe buffer, tail < head, sqId_: %u, streamId_: %u, cur head: %u, cur tail: %u, "
            "size: %u",
            __func__, sqId_, streamId_, sqHead_, sqTail_, pendingSqeCnt);
        int ret = memcpy_sp(sqCurrAddr, pendingSqeCnt * AC_SQE_SIZE, sqeBuf, pendingSqeCnt * RTSQ_SQE_SIZE);
        if (UNLIKELY(ret != 0)) {
            HCCL_ERROR("RtsqA5::%s sqe memcpy_sp failed, ret = %d", __func__, ret);
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

void RtsqA5::PreLaunchSqeForCache(bool& needCacheTask)
{
    // 校验needCacheTaskCallback_
    // 注意: A5新流程下needCacheTaskCallback_一定非空; 但A5老流程下不支持aicpu task cache, needCacheTaskCallback_为空;
    //     为避免A5老流程报错, 这里为空时跳过执行而非报错
    needCacheTask = false;
    if (UNLIKELY(needCacheTaskCallback_ == nullptr)) {
        HCCL_WARNING("[RtsqA5][PreLaunchSqeForCache] needCacheTaskCallback_ is null, keep needCacheTask as false");
    } else {
        needCacheTask = needCacheTaskCallback_();
    }
}

void RtsqA5::PostLaunchSqeForCache()
{
    // 注意: 只有needCacheTask为true时才调用PostLaunchSqeForCache, 此时一定是A5新流程, 因此addSqeArrayCallback_一定非空
    if (UNLIKELY(aicpuTsThreadPtr_ == nullptr)) {
        THROW<InternalException>("[RtsqA5][PostLaunchSqeForCache] aicpuTsThreadPtr_ is null");
    }
    if (UNLIKELY(addSqeArrayCallback_ == nullptr)) {
        THROW<InternalException>("[RtsqA5][PostLaunchSqeForCache] addSqeArrayCallback_ is null");
    }
    HcclResult ret = addSqeArrayCallback_(
        this, aicpuTsThreadPtr_, pendingSqeCnt, locBuf, streamId_, pendingSqeTransports_, pendingSqeLinkTypes_,
        pendingSqeTransportTypes_);
    if (UNLIKELY(ret != HCCL_SUCCESS)) {
        THROW<InternalException>("[RtsqA5][PostLaunchSqeForCache] addSqeArrayCallback_ failed, ret %d", ret);
    }
}

// 向芯片RTSQ VA中写入 SQE，并触发芯片执行
void RtsqA5::LaunchTask()
{
    HCCL_INFO("RtsqA5::%s: START, pendingSqeCnt[%u]", __func__, pendingSqeCnt);
    if (pendingSqeCnt == 0) { // 没有SQE ，直接返回
        HCCL_INFO("RtsqA5::%s: pendingSqeCnt is %u, return", __func__, pendingSqeCnt);
        return;
    }
    // 确保 rtsq 有足够空间放pending SQE
    CHK_RET_THROW(InternalException, "[RtsqA5::LaunchTask] MakeSureAvailableSpace failed", MakeSureAvailableSpace());

    if (pendingSqeCnt == 0) {
        return;
    }

    bool needCacheTask = false;
    PreLaunchSqeForCache(needCacheTask);
    // localBuffer拷贝到 RTSQ
    CHK_RET_THROW(InternalException, "[RtsqA5::LaunchTask] CopySqeBufToSq failed", CopySqeBufToSq(locBuf));

    // 正常展开按需打印SQE
    if ((UNLIKELY((PlfGetDebugConfig() & PLF_TASK) != 0)) || UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
        PLF_CONFIG_INFO(
            PLF_TASK, "[RtsqA5][LaunchTask] dump %llu generated SQEs in stream[%u]", pendingSqeCnt, streamId_);

        int ret = HCCL_SUCCESS;
        uint8_t* sqePtr = locBuf;
        for (size_t sqeIdx = 0; sqeIdx < pendingSqeCnt; sqeIdx++) {
            PLF_CONFIG_INFO(PLF_TASK, "[RtsqA5][LaunchTask] %uth generated SQE in stream[%u]", sqeIdx, streamId_);
            ret = PlfDumpSqe(sqePtr);
            if (UNLIKELY(ret != HCCL_SUCCESS)) {
                THROW<InternalException>(StringFormat("RtsqA5::%s DumpSqeContent failed, ret = %d", __func__, ret));
            }

            sqePtr += RTSQ_SQE_SIZE;
        }
    }

    // 更新tail，触发芯片执行
    u32 newTail = (sqTail_ + pendingSqeCnt) % sqDepth_;
    ConfigSqTail(newTail);
    sqTail_ = newTail;
    sqTailTotal_ = WrapAdd(sqTailTotal_, pendingSqeCnt);

    // 缓存sqe
    if (needCacheTask) {
        PostLaunchSqeForCache();
    }
    // 清空本地的locBuffer和sqeCnt数目
    HCCL_INFO(
        "RtsqA5::%s: END, pendingSqeCnt[%u], streamId_[%u] sqHead_[%u] sqTail_[%u]", __func__, pendingSqeCnt, streamId_,
        sqHead_, sqTail_);
    pendingSqeCnt = 0;
    (void)memset_s(locBuf, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT, 0, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT); // locBuffer清零
    currentTransport_ = DFX_INVALID_U64;
    currentLinkType_ = static_cast<u8>(DfxLinkTypeVal::LINK_ONCHIP);
    currentTransportType_ = static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL);
}

void RtsqA5::RefreshSqeHeaderTaskField(Rt91095StarsSqeHeader* sqeHeaderPtr)
{
    SetSqeHeaderTaskFields(sqeHeaderPtr, taskId_);
    SetTaskIdBySqeId();
}

// 向芯片RTSQ VA中写入aicpu task cache SQE，并触发芯片执行
HcclResult RtsqA5::LaunchNewTask(uint8_t* sqeArray, uint32_t sqeCount)
{
    // 注意: cache命中时才会调用LaunchNewTask, 此时一定不存在pending SQE
    if (UNLIKELY(pendingSqeCnt > 0)) {
        HCCL_ERROR("RtsqA5::%s: pendingSqeCnt[%u] should be 0 when aicpu task cache hits!", __func__, pendingSqeCnt);
        return HCCL_E_INTERNAL;
    }

    // 临时设置pendingSqeCnt, 用于MakeSureAvailableSpace
    pendingSqeCnt = sqeCount;

    // 确保 rtsq 有足够空间放pending SQE
    CHK_RET(MakeSureAvailableSpace());

    if (pendingSqeCnt == 0) {
        return HCCL_SUCCESS;
    }

    // sqeArray拷贝到 RTSQ
    CHK_RET(CopySqeBufToSq(sqeArray));

    // 更新tail，触发芯片执行
    u32 newTail = (sqTail_ + pendingSqeCnt) % sqDepth_;
    ConfigSqTail(newTail);
    sqTail_ = newTail;
    sqTailTotal_ = WrapAdd(sqTailTotal_, pendingSqeCnt);

    HCCL_INFO(
        "RtsqA5::%s: END, pendingSqeCnt[%u], streamId_[%u] sqHead_[%u] sqTail_[%u]", __func__, pendingSqeCnt, streamId_,
        sqHead_, sqTail_);

    // 重置pendingSqeCnt
    pendingSqeCnt = 0;
    return HCCL_SUCCESS;
}

void RtsqA5::TryLaunchTask()
{
    if (pendingSqeCnt == 0) {
        return;
    }

    RefreshSqHead();
    u32 availableSpace = GetTailToHeadDist();
    if (availableSpace <= pendingSqeCnt) {
        return;
    }

    bool needCacheTask = false;
    PreLaunchSqeForCache(needCacheTask);

    CHK_RET_THROW(InternalException, "[RtsqA5::TryLaunchTask] CopySqeBufToSq failed", CopySqeBufToSq(locBuf));

    u32 newTail = (sqTail_ + pendingSqeCnt) % sqDepth_;
    ConfigSqTail(newTail);
    sqTail_ = newTail;
    sqTailTotal_ = WrapAdd(sqTailTotal_, pendingSqeCnt);

    // 缓存sqe
    if (needCacheTask) {
        PostLaunchSqeForCache();
    }
    pendingSqeCnt = 0;
    (void)memset_s(locBuf, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT, 0, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT);
    currentTransport_ = DFX_INVALID_U64;
    currentLinkType_ = static_cast<u8>(DfxLinkTypeVal::LINK_ONCHIP);
    currentTransportType_ = static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL);
    HCCL_INFO(
        "RtsqA5::%s: END, pendingSqeCnt[%u], streamId_[%u] sqHead_[%u] sqTail_[%u]", __func__, pendingSqeCnt, streamId_,
        sqHead_, sqTail_);
}

u8* RtsqA5::GetCurrSqeBuffer()
{
    lastSqeAddr_ = sqBaseAddr_ + static_cast<u64>((sqTail_ + pendingSqeCnt) % sqDepth_) * RTSQ_SQE_SIZE;
    return locBuf + pendingSqeCnt * RTSQ_SQE_SIZE;
}

u64 RtsqA5::GetSqeAddr() const { return lastSqeAddr_; }

void RtsqA5::RefreshInfo()
{
    SetTaskIdBySqeId();
    pendingSqeTransports_[pendingSqeCnt] = currentTransport_;
    pendingSqeLinkTypes_[pendingSqeCnt] = currentLinkType_;
    pendingSqeTransportTypes_[pendingSqeCnt] = currentTransportType_;
    pendingSqeCnt++;
    currentTransport_ = DFX_INVALID_U64;
    currentLinkType_ = static_cast<u8>(DfxLinkTypeVal::LINK_ONCHIP);
    currentTransportType_ = static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL);

#ifdef CCL_KERNEL_AICPU
    if (launchFlag_ && !PlfIsBatchLaunchMode()) {
        LaunchTask();
        return;
    }
#endif

    if (pendingSqeCnt != PER_LAUNCH_SQE_CNT) {
        return;
    }
    // 挂起的sqe数量为128个，则需要向芯片RTSQ中写入task
    LaunchTask();
}

void RtsqA5::NotifyWait(u32 notifyId) { NotifyWait(notifyId, GetKernelExecTimeoutFromEnvConfig()); }

void RtsqA5::NotifyWait(u32 notifyId, u32 timeout)
{
    BuildA5SqeNotifyWait(streamId_, taskId_, notifyId, timeout, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::NotifyWait: streamId %u, taskId %u, notifyId %u, timeout[%u s]", streamId_, taskId_, notifyId,
        timeout);
    RefreshInfo();
}

void RtsqA5::NotifyRecordLoc(u32 notifyId)
{
    BuildA5SqeNotifyRecord(streamId_, taskId_, notifyId, GetCurrSqeBuffer());
    HCCL_INFO("RtsqA5::NotifyRecordLoc: streamId %u, taskId %u, notifyId %u", streamId_, taskId_, notifyId);
    RefreshInfo();
}

void RtsqA5::Cnt1toNNotifyWait(u32 notifyId, u32 value)
{
    BuildA5SqeCnt1toNNotifyWait(streamId_, taskId_, notifyId, value, GetCurrSqeBuffer());
    HCCL_INFO("RtsqA5::Cnt1toNNotifyWait: streamId %u, taskId %u, notifyId %u", streamId_, taskId_, notifyId);
    RefreshInfo();
}

void RtsqA5::Cnt1toNNotifyRecord(u32 notifyId, u32 value)
{
    BuildA5SqeCnt1toNNotifyRecord(streamId_, taskId_, notifyId, value, GetCurrSqeBuffer());
    HCCL_INFO("RtsqA5::Cnt1toNNotifyRecord: streamId %u, taskId %u, notifyId %u", streamId_, taskId_, notifyId);
    RefreshInfo();
}

void RtsqA5::CntNto1NotifyWait(u32 notifyId, u32 value)
{
    BuildA5SqeCntNto1NotifyWait(streamId_, taskId_, notifyId, value, GetCurrSqeBuffer());
    HCCL_INFO("RtsqA5::CntNto1NotifyWait: streamId %u, taskId %u, notifyId %u", streamId_, taskId_, notifyId);
    RefreshInfo();
}

void RtsqA5::CntNto1NotifyRecord(u32 notifyId, u32 value)
{
    BuildA5SqeCntNto1NotifyRecord(streamId_, taskId_, notifyId, value, GetCurrSqeBuffer());
    HCCL_INFO("RtsqA5::CntNto1NotifyRecord: streamId %u, taskId %u, notifyId %u", streamId_, taskId_, notifyId);
    RefreshInfo();
}

void RtsqA5::SdmaCopy(u64 srcAddr, u64 dstAddr, u32 size, u32 partId)
{
    // 不带reduce的拷贝，opcode填0
    (void)partId;
    BuildA5SqeSdmaCopy(streamId_, taskId_, dstAddr, srcAddr, size, RTSQ_A5_PART_ID, 0, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::SdmaCopy: streamId %u, taskId %u, srcAddr 0x%llx, dstAddr 0x%llx, size %u", streamId_, taskId_,
        srcAddr, dstAddr, size);
    RefreshInfo();
}

const std::unordered_map<ReduceOp, RtStarsMemcpyAsyncOperationKind, EnumClassHash> ReduceOpToStarsOpKindMap
    = {{ReduceOp::SUM, RtStarsMemcpyAsyncOperationKind::RT_STARS_MEMCPY_ASYNC_OP_KIND_ADD},
       {ReduceOp::MAX, RtStarsMemcpyAsyncOperationKind::RT_STARS_MEMCPY_ASYNC_OP_KIND_MAX},
       {ReduceOp::MIN, RtStarsMemcpyAsyncOperationKind::RT_STARS_MEMCPY_ASYNC_OP_KIND_MIN},
       {ReduceOp::EQUAL, RtStarsMemcpyAsyncOperationKind::RT_STARS_MEMCPY_ASYNC_OP_KIND_EQUAL}};

const std::unordered_map<DataType, RtStarsMemcpyAsyncDataType, EnumClassHash> DataTypeToStarsDataTypeMap
    = {{DataType::INT8, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT8},
       {DataType::INT16, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT16},
       {DataType::INT32, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT32},
       {DataType::FP16, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_FP16},
       {DataType::FP32, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_FP32},
       {DataType::BFP16, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_BFP16}};

void RtsqA5::SdmaReduce(u64 srcAddr, u64 dstAddr, u32 size, u32 partId, const ReduceIn& reduceIn)
{
    (void)partId;
    if (UNLIKELY(
            ReduceOpToStarsOpKindMap.find(reduceIn.reduceOp) == ReduceOpToStarsOpKindMap.end()
            || DataTypeToStarsDataTypeMap.find(reduceIn.dataType) == DataTypeToStarsDataTypeMap.end())) {
        THROW<InternalException>(StringFormat(
            "Sdma does not support reduceOp %s dataType %s", reduceIn.reduceOp.Describe().c_str(),
            reduceIn.dataType.Describe().c_str()));
    }

    u8 op = static_cast<u8>(ReduceOpToStarsOpKindMap.at(reduceIn.reduceOp));
    u8 type = static_cast<u8>(DataTypeToStarsDataTypeMap.at(reduceIn.dataType));

    BuildA5SqeSdmaCopy(streamId_, taskId_, dstAddr, srcAddr, size, RTSQ_A5_PART_ID, (op | type), GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::SdmaReduce: streamId %u, taskId %u, srcAddr 0x%llx, dstAddr 0x%llx, size %u", streamId_, taskId_,
        srcAddr, dstAddr, size);
    RefreshInfo();
}

bool RtsqA5::IsRtsqQueueSpaceSufficient()
{
    // 判断逻辑与rtsq内部保持一致，rtsq剩余空间需要大于（rtsq挂起的任务数量+本次任务）
    u32 availableSpace = GetTailToHeadDist();
    if (availableSpace > pendingSqeCnt + 1) {
        return true;
    }

    // 否则的话，需要再次查询一次head，确认是否是因为head没有更新导致空间不足，如果查询后空间仍然不足，则返回false
    RefreshSqHead();
    availableSpace = GetTailToHeadDist();

    return (availableSpace > pendingSqeCnt + 1);
}

HcclResult RtsqA5::SetPreStreamSyncReady()
{
    isPreStreamSync = true;
    return HCCL_SUCCESS;
}

HcclResult RtsqA5::SetPreStreamSyncFin()
{
    isPreStreamSync = false;
    return HCCL_SUCCESS;
}

bool RtsqA5::GetPreStreamSyncStatus() { return isPreStreamSync; }

// UbDbSend的slot记录也统一走此接口, dbSqeIdx等价于pendingSqeCnt;
HcclResult RtsqA5::RecordDbSendSlot(UbTransportLiteImpl* transport, u32 dbSqeIdx, u16 seqIdx, u16 piValue)
{
    u32 dbSendTail = dbSendTail_;
    u32 dbSendDepth = dbSendDepth_;
    // 环满时等待后台线程出队腾出空位，避免覆盖未出队记录(参考MakeSureAvailableSpace的反压等待)。
    u32 nextTail = (dbSendTail + 1 == dbSendDepth) ? 0 : (dbSendTail + 1);
    // 快照须在环满等待前计算: 等待中HandleDispatchAllStreams可能下发本流pending批次推进sqTailTotal_,
    // 但本SQE落位恒为进入时的sqTailTotal_+dbSqeIdx(若已下发则成为新批第0条, 落位不变)
    u32 wrapSlotIdx = WrapAdd(sqTailTotal_, dbSqeIdx);
    if (UNLIKELY(nextTail == dbSendHead_)) {
#ifdef CCL_KERNEL_AICPU
        sqFullTimeout_ = PlfGetSqFullTimeOut();
#endif
        auto startTime = std::chrono::steady_clock::now();
        const std::chrono::seconds printInterval(PRINT_INTERVAL); // 打印间隔30s
        auto lastPrintTime = startTime - printInterval;
        do {
            auto curTime = std::chrono::steady_clock::now();
            if (UNLIKELY(curTime - lastPrintTime >= printInterval)) {
                HCCL_RUN_WARNING(
                    "[%s] jetty SQ overflow check, dbSendSlots full, while loop. sqId:%u, dbSendTail:%u, "
                    "dbSendHead:%u, dbSendDepth:%u, baseSqHead:%u",
                    __func__, sqId_, dbSendTail, dbSendHead_, dbSendDepth, baseSqHead_);
                lastPrintTime = curTime;
            }

            HcclResult ret = HCCL_SUCCESS;
            TRY_CATCH_RETURN(
                ret = CheckLaunchTaskStatus(startTime, curTime, "jetty SQ overflow check, dbSendSlots full"));
            if (UNLIKELY(ret == HCCL_E_SUSPENDING)) {
                return HCCL_SUCCESS; // 挂起: 跳过记录视同成功, 与"挂起不抛异常"语义一致
            }

#ifdef CCL_KERNEL_AICPU
            // 等待期间尝试下发所有流攒着的SQE(含本流pending批次): SQE完成推进head, 后台出队才有空间释放
            ret = PlfHandleDispatchAllStreams();
            if (UNLIKELY(ret != HCCL_SUCCESS)) {
                HCCL_ERROR("[%s] HandleDispatchAllStreams failed, ret = %d, sqId:%u", __func__, ret, sqId_);
                return ret;
            }
#endif
            if (checkOpExecStatusCallback_ != nullptr) {
                checkOpExecStatusCallback_();
            }
        } while (nextTail == dbSendHead_);
    }

    DbSendSlotMeta& slot = dbSendSlots_[dbSendTail];
    slot.transport = transport;
    slot.wrapSlotIdx = wrapSlotIdx;
    slot.seqIdx = seqIdx;
    slot.piValue = piValue;
    dbSendTail_ = nextTail;
    return HCCL_SUCCESS;
}

void RtsqA5::UbDbSend(const UbJettyLiteId& jettyLiteId, u16 piValue, u16 seqIdx, UbTransportLiteImpl* transport)
{
    // 消耗pi调用UbDbDSend，不消耗pi的也会占用rtsq的槽位但是不会调用UbDbDSend
    CHK_RET_THROW(
        InternalException, "[RtsqA5::UbDbSend] RecordDbSendSlot failed",
        RecordDbSendSlot(transport, pendingSqeCnt, seqIdx, piValue));
    // piValue需要使用u16数据类型，保证自然增长，用于判断是否翻转
    BuildA5SqeUbDbSend(streamId_, taskId_, jettyLiteId, piValue, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::UbDbSend: seqId %u, streamId %u, taskId %u, piValue(UbPi) %u, SqTail(Rtsq Pi) %u", seqIdx, streamId_,
        taskId_, piValue, sqTail_);
    RefreshInfo();
}

void RtsqA5::RdmaDbSend(const uint64_t& dbAddr, const uint64_t& dbValue)
{
    BuildA5SqeRdmaDbSend(streamId_, taskId_, dbAddr, dbValue, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::RdmaDbSend: RdmaDbSend streamId %u, taskId %u, Sqe: %s, dbAddr:0x%llx, dbValue:0x%llx, SqTail(Rtsq "
        "Pi):%u",
        streamId_, taskId_, Bytes2hex(GetCurrSqeBuffer(), RTSQ_SQE_SIZE).c_str(), dbAddr, dbValue, sqTail_);
    RefreshInfo();
}

void RtsqA5::RecordCCoreDfxTask(bool isWait) const
{
    if (aicpuTsThreadPtr_ == nullptr) {
        return;
    }
    auto* stream = static_cast<StreamLite*>(aicpuTsThreadPtr_->GetStreamLitePtr());
    if (stream == nullptr || !stream->HasReportStreamTaskCallback()) {
        return;
    }
    const void* opInfo = stream->GetLatestDfxOpInfo();
    if (opInfo == nullptr) {
        return;
    }

#ifdef CCL_KERNEL_AICPU
    // This source is also linked into libhcomm, which has no device DFX implementation.
    const auto* dfxOpInfo = static_cast<const DfxDfxOpInfo*>(opInfo);
    const auto reportOp = Hccl::GetMarkCompactReportOp();
    if (dfxOpInfo->hcclCommDfxLite != nullptr && reportOp != nullptr) {
        // NextTaskSlot may report a full queue before adding this CONDITION.
        reportOp(dfxOpInfo->hcclCommDfxLite, dfxOpInfo);
    }
#endif

    // Capture the full task ID before RefreshInfo advances it or launches the SQEs.
    auto* slot = stream->NextTaskSlot();
    *slot = DfxTaskInfo{};
    slot->taskType = isWait ? TASK_CCORE_NOTIFY_WAIT : TASK_CCORE_NOTIFY_RECORD;
    slot->sqId = stream->GetSqId();
    slot->taskId = taskId_;
    slot->dfxOpInfo = ReinterpretAs<u64>(opInfo);
    slot->linkType = LINK_ONCHIP;
    slot->transportType = DFX_TRANSPORT_TYPE_LOCAL;
}

HcclResult RtsqA5::CCoreNotifyWait(u64 waitAddr, u64 curTurnCntAddr, bool last)
{
    BuildA5SqeCCoreNotifyWait(streamId_, taskId_, waitAddr, curTurnCntAddr, last, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::CCoreNotifyWait: streamId %u, taskId %u, waitAddr %llu, curTurnCntAddr %llu, last %d", streamId_,
        taskId_, waitAddr, curTurnCntAddr, last);
    RecordCCoreDfxTask(true);
    RefreshInfo();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult RtsqA5::CCoreNotifyRecord(u64 recordAddr, u64 curTurnCntAddr)
{
    BuildA5SqeCCoreNotifyRecord(streamId_, taskId_, recordAddr, curTurnCntAddr, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::CCoreNotifyRecord: streamId %u, taskId %u, recordAddr %llu, curTurnCntAddr %llu", streamId_, taskId_,
        recordAddr, curTurnCntAddr);
    RecordCCoreDfxTask(false);
    RefreshInfo();
    return HcclResult::HCCL_SUCCESS;
}

void RtsqA5::P2PWriteValue(u64 remoteAddr, u32 writeValue)
{
    BuildA5SqeP2pWriteValue(streamId_, taskId_, remoteAddr, writeValue, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::P2PWriteValue: streamId %u, taskId %u, remoteAddr %llu, writeValue %u", streamId_, taskId_, remoteAddr,
        writeValue);
    RefreshInfo();
}

HcclResult RtsqA5::GetLastStreamIdAndTaskId(uint16_t& streamId, uint16_t& taskId) const
{
    if (pendingSqeCnt > 0) {
        const u8* lastSqe = locBuf + (pendingSqeCnt - 1U) * RTSQ_SQE_SIZE;
        auto* sqe = ReinterpretAs<const Rt91095StarsNotifySqe*>(lastSqe);
        streamId = sqe->header.rtStreamId;
        taskId = sqe->header.taskId;
        HCCL_INFO(
            "[%s] from pending, pendingSqeCnt[%u], sqId[%u], streamId[%u], taskId[%u].", __func__, pendingSqeCnt, sqId_,
            streamId, taskId);
        return HCCL_SUCCESS;
    }
    const u32 lastIdx = (sqTail_ + sqDepth_ - 1U) % sqDepth_;
    HCCL_INFO(
        "[%s] from rtsq, sqId[%u], sqTail[%u], sqDepth[%u], lastIdx[%u].", __func__, sqId_, sqTail_, sqDepth_, lastIdx);
    u8 sqeType = 0;
    u32 notifyId = 0;
    return GetSqeHeaderFieldsBySqIdx(lastIdx, streamId, taskId, sqeType, notifyId);
}

// 背景线程轮询SQ完成情况，将已完成的DbSend slot取出并上报给CiTracker
// 由RtsqPollCompletionDaemon周期性调用（约10ms）
void RtsqA5::PollCompletion()
{
    u32 newHead = ReadUnwrappedSqHead();
    u32 dbSendDepth = dbSendDepth_;
    UbTransportLiteImpl* lastTransport{nullptr};
    pollSnapshot_.clear();

    /* 回退守卫: baseSqHead_更新滞后整圈, QuerySqHead()读值偏小,
     * newHead伪回退到lastHead_之后(回退量<sqDepth); 不守卫会误出队在途DbSend,
     * 故跳过本轮保留lastHead_, 待恢复流程重锚
     *
     * baseSqHead_更新滞后小概率会发生ReadUnwrappedSqHead里,由下面两个条件触发
     * u32 baseSqHead = baseSqHead_;这里因为同步的关系baseSqHead可能读取还是旧值
     * u32 hwHead = QuerySqHead();这里可能硬件执行太快,sqHead回绕一圈后,hwHead的值比上次读到小
     * 所以newHead伪回退到lastHead_之后会发生。
     */
    u32 lastHead = lastHead_;
    u64 backward = static_cast<u64>(lastHead) + baseSqHeadMaxSize_ - newHead;
    if (backward >= baseSqHeadMaxSize_) {
        backward -= baseSqHeadMaxSize_;
    }
    if (backward != 0 && backward < sqDepth_) {
        HCCL_RUN_INFO(
            "[%s] jetty SQ overflow check, anchor lag detected, skip poll. sqId:%u, lastHead:%u, newHead:%u, "
            "baseSqHead:%u",
            __func__, sqId_, lastHead, newHead, baseSqHead_);
        return;
    }

    u32 dbSendHead = dbSendHead_;
    /* lastHead_==newHead时while不执行的两种情况：
     * 1. head无变化：SQ无新完成，跳过这是正确的
     * 2. PollCompletion10ms进一次，数据面下发太快，两轮poll间sqHead会多次回绕
     * 为了防止情况2，每次回绕累计到baseSqHead_在[0, baseSqHeadMaxSize_)的u32近满空间(约42.9亿)内追踪，
     * dbSendSlots_环满则由RecordDbSendSlot发送侧等待后台出队保护，记录不会被覆盖
     */
    while (lastHead != newHead) {
        if (dbSendHead == dbSendTail_) // dbSendSlots_队列空的情况
        {
            lastHead = newHead;
            break;
        }

        DbSendSlotMeta& slot = dbSendSlots_[dbSendHead];
        if (slot.wrapSlotIdx != lastHead) {
            lastHead = (lastHead + 1 == baseSqHeadMaxSize_) ? 0 : (lastHead + 1);
            continue;
        }

        UbTransportLiteImpl* tempTransport = slot.transport;
        if (tempTransport != lastTransport) {
            if (!pollSnapshot_.empty()) {
                UbConnLiteMgr::GetInstance().AppendCompletedCis(
                    lastTransport, pollSnapshot_.data(), pollSnapshot_.size());
                pollSnapshot_.clear();
            }
            lastTransport = tempTransport;
        }
        pollSnapshot_.emplace_back(slot.seqIdx, slot.piValue);
        dbSendHead = (dbSendHead + 1 == dbSendDepth) ? 0 : (dbSendHead + 1);
        lastHead = (lastHead + 1 == baseSqHeadMaxSize_) ? 0 : (lastHead + 1);
    }
    dbSendHead_ = dbSendHead;
    lastHead_ = lastHead;

    if (!pollSnapshot_.empty()) {
        UbConnLiteMgr::GetInstance().AppendCompletedCis(lastTransport, pollSnapshot_.data(), pollSnapshot_.size());
        pollSnapshot_.clear();
    }
}

void RtsqA5::SetCurrentTransportContext(u64 channelHandle, u8 linkType, u8 transportType)
{
    currentTransport_ = channelHandle;
    currentLinkType_ = linkType;
    currentTransportType_ = transportType;
}
} // namespace Hccl
