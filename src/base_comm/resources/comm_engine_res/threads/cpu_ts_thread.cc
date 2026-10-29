/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cpu_ts_thread.h"
#include "common/dev_type_utils_pub.h"
#include "hccl_common.h"
#include "adapter_rts.h"
#include "dfx/prof_utils_pub.h"

namespace hccl {
const std::unordered_map<HcclDataType, aclDataType> hccl2rtDataTypeMap = {
    {HCCL_DATA_TYPE_INT8, ACL_INT8},    {HCCL_DATA_TYPE_INT16, ACL_INT16}, {HCCL_DATA_TYPE_INT32, ACL_INT32},
    {HCCL_DATA_TYPE_FP16, ACL_FLOAT16}, {HCCL_DATA_TYPE_FP32, ACL_FLOAT},  {HCCL_DATA_TYPE_BFP16, ACL_BF16},
};

const std::unordered_map<HcclReduceOp, aclrtReduceKind> hccl2rtReduceOpMap = {
    {HCCL_REDUCE_SUM, ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM},
    {HCCL_REDUCE_MAX, ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX},
    {HCCL_REDUCE_MIN, ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN},
};

CpuTsThread::CpuTsThread(rtStream_t rtStream, uint32_t notifyNum, const NotifyLoadType notifyLoadType)
    : rtStream_(rtStream),
      notifyNum_(notifyNum),
      notifyLoadType_(notifyLoadType)
{}

CpuTsThread::CpuTsThread(StreamType streamType, uint32_t notifyNum, const NotifyLoadType notifyLoadType)
    : streamType_(streamType),
      notifyNum_(notifyNum),
      notifyLoadType_(notifyLoadType)
{}

CpuTsThread::~CpuTsThread() { DeInitImpl(); }

CpuTsThread::CpuTsThread(rtStream_t rtStream, LocalNotify** notifys, uint32_t notifyNum)
    : rtStream_(rtStream),
      notifyNum_(notifyNum),
      notifyLoadType_(NotifyLoadType::HOST_NOTIFY),
      injectedNotifys_(notifys, notifys + notifyNum) // 拷贝引用（借用：对象归管理器，线程不拥有）
{}

HcclResult CpuTsThread::Init()
{
    // Host 侧初始化
    CHK_RET(GetRunSideIsDevice(isDeviceSide_));
    CHK_RET(hrtGetDeviceType(devType_));
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (!isDeviceSide_) {
        s32 deviceLogicId;
        CHK_RET(hrtGetDevice(&deviceLogicId));
        CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(deviceLogicId), devId_));
        if (streamType_ == StreamType::STREAM_TYPE_DEVICE || notifyLoadType_ == NotifyLoadType::DEVICE_NOTIFY) {
            HCCL_ERROR(
                "[CpuTsThread::%s] not supported, streamType_[%d], notifyLoadType_[%d].", __func__, streamType_,
                notifyLoadType_);
            return HCCL_E_NOT_SUPPORT;
        }
        if (rtStream_ == nullptr) {
            stream_.reset(new (std::nothrow) Stream(streamType_));
            CHK_SMART_PTR_NULL(stream_);
            // 流创建/初始化失败时 Stream 构造函数仅打日志不置失败态（内部 rtStream 为空），须显式拦截
            CHK_PRT_RET(
                stream_->ptr() == nullptr,
                HCCL_ERROR(
                    "[%s] create stream failed, ptr is null, streamType[%d]", __func__, static_cast<s32>(streamType_)),
                HCCL_E_RUNTIME);
            rtStream_ = stream_->ptr();
        } else {
            stream_.reset(new (std::nothrow) Stream(rtStream_));
            CHK_SMART_PTR_NULL(stream_);
        }
        if (!injectedNotifys_.empty()) {
            // GE保序注入：借用引用已在构造时拷贝至injectedNotifys_（对象所有权始终归管理器，
            // 本线程不销毁不delete）；GetNotify/序列化经借用数组读取。管理器注销/失效销毁
            // 对象后不得再经本线程使用
            HCCL_INFO("[CpuTsThread::%s] %s, isDeviceSide[%d].", __func__, FormatThreadInfo().c_str(), isDeviceSide_);
            return HCCL_SUCCESS;
        }
        notifys_.reserve(notifyNum_);
        for (uint32_t idx = 0; idx < notifyNum_; idx++) {
            notifys_.emplace_back(nullptr);
            notifys_[idx].reset(new (std::nothrow) LocalNotify());
            CHK_SMART_PTR_NULL(notifys_[idx]);
            CHK_RET(notifys_[idx]->Init(notifyLoadType_));
            if (!(*isGe950)) {
                CHK_RET(notifys_[idx]->SetIpc());
            }
        }
        HCCL_INFO("[CpuTsThread::%s] %s, isDeviceSide[%d].", __func__, FormatThreadInfo().c_str(), isDeviceSide_);
        return HCCL_SUCCESS;
    } else {
        HCCL_ERROR("[CpuTsThread::%s] device side not supported.", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
}

HcclResult CpuTsThread::DeInit() { return DeInitImpl(); }

HcclResult CpuTsThread::DeInitImpl()
{
    HCCL_INFO("[CpuTsThread::%s] %s.", __func__, FormatThreadInfo().c_str());
    streamType_ = StreamType::STREAM_TYPE_RESERVED;
    notifyNum_ = 0;
    stream_ = nullptr;
    notifys_.clear();         // 自建notify（线程拥有）析构销毁
    injectedNotifys_.clear(); // 借用引用（对象归管理器）仅清引用，不触碰对象
    return HCCL_SUCCESS;
}

std::string& CpuTsThread::GetUniqueId()
{
    if (!uniqueIdStr_.empty()) {
        return uniqueIdStr_;
    }
    return UpdateUniqueId();
}

std::string& CpuTsThread::UpdateUniqueId()
{
    // 序列化信息。GE保序线程规避设备流构造（fakeDeviceRes_ 标记，经 HcommThreadAcquireByNotify
    // 的GE路径设置）：原实现为满足设备侧按 aicpu_ts_thread 格式解析，额外构造
    // STREAM_TYPE_DEVICE 流与 SQ/CQ 上下文设备内存——仅为填序列化字段，无功能消费者。
    // GE 路径类型标记 STREAM_TYPE_ONLINE、streamParam 零值填充，只把 notify 真实数据序列化传下去
    // （设备侧 DeviceInit 按"ONLINE + 零值"假流签名跳过流初始化，该假流可被正确解析）；
    // 非 GE 路径（OPBASE/ACLGRAPH/mainThread）维持真实构造不变
    uniqueIdStr_ = std::string();
    std::ostringstream oss;
    StreamType streamType = fakeDeviceRes_ ? StreamType::STREAM_TYPE_ONLINE : StreamType::STREAM_TYPE_DEVICE;
    oss.write(reinterpret_cast<const char_t*>(&streamType), sizeof(streamType));
    oss.write(reinterpret_cast<const char_t*>(&notifyLoadType_), sizeof(notifyLoadType_));
    oss.write(reinterpret_cast<const char_t*>(&devId_), sizeof(devId_));
    oss.write(reinterpret_cast<const char_t*>(&notifyNum_), sizeof(notifyNum_));

    if (fakeDeviceRes_) {
        HcclStreamParam streamParam{}; // 零值假流：不构造真实设备流与 SQ/CQ 上下文
        oss.write(reinterpret_cast<const char_t*>(&streamParam), sizeof(streamParam));
    } else {
        // 临时申请一条流，用于在device侧资源展开时initStream
        if (streamDevice_ == nullptr) {
            streamDevice_.reset(new (std::nothrow) Stream(streamType));
        }
        if (streamDevice_ == nullptr) {
            HCCL_ERROR("[CpuTsThread][%s]reset stream failed, stream type[%d]", __func__, streamType);
            return uniqueIdStr_;
        }
        // 流创建失败时 Stream 构造函数仅打日志不置失败态，须显式拦截，防止序列化出空流 uid
        if (streamDevice_->ptr() == nullptr) {
            HCCL_ERROR(
                "[CpuTsThread][%s]create device stream failed, ptr is null, stream type[%d]", __func__, streamType);
            return uniqueIdStr_;
        }

        uint64_t size = sizeof(SqCqeContext);
        if (sqCqeContext_.ptr() == nullptr) {
            sqCqeContext_ = DeviceMem::alloc(size);
        }
        if (sqCqeContext_.ptr() == nullptr) {
            HCCL_ERROR("[CpuTsThread][%s]alloc mem failed, mem size[%llu]", __func__, size);
            return uniqueIdStr_;
        }
        HcclResult memSetRet = hrtMemSet(sqCqeContext_.ptr(), size, size);
        if (memSetRet != HCCL_SUCCESS) {
            HCCL_ERROR("[CpuTsThread][%s]mem set failed, mem size[%llu], ptr[%p]", __func__, size, sqCqeContext_.ptr());
            return uniqueIdStr_;
        }

        HcclStreamParam streamParam;
        streamParam.streamInfo.streamIds = streamDevice_->id();
        streamParam.streamInfo.sqIds = streamDevice_->sqId();
        streamParam.streamInfo.cqIds = streamDevice_->cqId();
        streamParam.streamInfo.logicCqids = streamDevice_->logicCqId();
        streamParam.sqCqContextAddr = reinterpret_cast<uint64_t>(sqCqeContext_.ptr());
        streamParam.sqCqContextSize = sqCqeContext_.size();
        oss.write(reinterpret_cast<const char_t*>(&streamParam), sizeof(streamParam));
    }

    HcclResult ret = HCCL_SUCCESS;
    for (uint32_t idx = 0; idx < notifyNum_; idx++) {
        // GetNotify已按注入/自建模式分流（GE保序借用数组 or 自建notifys_）
        LocalNotify* notify = GetNotify(idx);
        if (notify == nullptr) {
            HCCL_ERROR("[AicpuTsThread][UpdateUniqueId]GetNotify[%u] is nullptr", idx);
            uniqueIdStr_ = std::string();
            return uniqueIdStr_;
        }
        HcclSignalInfo notifyInfo;
        ret = notify->GetNotifyData(notifyInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[AicpuTsThread][UpdateUniqueId]GetNotifyData failed, ret[%d]", ret);
            uniqueIdStr_ = std::string();
            return uniqueIdStr_;
        }
        HCCL_INFO(
            "[AicpuTsThread][UpdateUniqueId]get local notify data success, resId[%u], tsId:%d, devId[%u]",
            notifyInfo.resId, notifyInfo.tsId, notifyInfo.devId);
        oss.write(reinterpret_cast<const char_t*>(&notifyInfo), sizeof(notifyInfo));
    }
    HCCL_DEBUG("[AicpuTsThread][UpdateUniqueId] stream[%p], notifyNum[%u]", stream_->ptr(), notifyNum_);

    uniqueIdStr_ = oss.str();
    return uniqueIdStr_;
}

uint32_t CpuTsThread::GetNotifyNum() const { return notifyNum_; }

size_t CpuTsThread::GetNotifySize() const
{
    return !injectedNotifys_.empty() ? injectedNotifys_.size() : notifys_.size();
}

LocalNotify* CpuTsThread::GetNotify(uint32_t index) const
{
    bool useInjected = !injectedNotifys_.empty();
    size_t notifySize = useInjected ? injectedNotifys_.size() : notifys_.size();
    if (index >= notifySize) {
        HCCL_ERROR(
            "[CpuTsThread][GetNotify] notifySize[%zu], index[%u] out of range, useInjected[%d]", notifySize, index,
            useInjected);
        return nullptr;
    }
    return useInjected ? injectedNotifys_[index] : notifys_[index].get();
}

// A3 Stream
Stream* CpuTsThread::GetStream() const { return stream_.get(); }

// A5 Stream
void* CpuTsThread::GetStreamLitePtr() const
{
    return nullptr; // Not implemented
}

s32 CpuTsThread::GetStreamId() const { return stream_ ? stream_->id() : -1; }

void CpuTsThread::LaunchTask() const { return; }

void CpuTsThread::TryLaunchTask() const
{
    HCCL_DEBUG("[%s] CpuTsThread does not support TryLaunchTask, skip", __func__);
    return;
}

// Local Data Plane Functions
HcclResult CpuTsThread::LocalNotifyRecord([[maybe_unused]] uint32_t notifyId) const
{
    HCCL_ERROR("[CpuTsThread][%s]not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CpuTsThread::LocalNotifyWait([[maybe_unused]] uint32_t notifyId) const
{
    HCCL_ERROR("[CpuTsThread][%s]not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
CpuTsThread::LocalNotifyRecord([[maybe_unused]] ThreadHandle dstThread, [[maybe_unused]] uint32_t dstNotifyIdx) const
{
#ifndef CCL_KERNEL_AICPU
    u64 beginTime = hcomm::GetProfCycleTime();
    HCCL_INFO("[%s]dstThread[0x%llu], dstNotifyIdx[%u].", __func__, dstThread, dstNotifyIdx);
    CHK_PRT_RET(
        !hcomm::EqualToOrGreaterThan950().value_or(false),
        HCCL_ERROR("[CpuTsThread][%s]only supports DEV_TYPE_950 and later", __func__),
        HCCL_E_NOT_SUPPORT); // 只支持950及后续代际, 其他场景调用HcclLocalNotifyRecord

    Stream* stream = GetStream();
    CHK_PTR_NULL(stream);
    Thread* const dstThreadPtr = reinterpret_cast<Thread*>(dstThread);
    CHK_PTR_NULL(dstThreadPtr);
    LocalNotify* dstNotify = dstThreadPtr->GetNotify(dstNotifyIdx);
    CHK_PTR_NULL(dstNotify);

    HcclResult ret = dstNotify->Post(*stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]fail, dstThread[0x%llx], dstNotifyIdx[%u].", __func__, dstThread, dstNotifyIdx), ret);

    HcclSignalInfo signalInfo;
    CHK_RET(dstNotify->GetNotifyData(signalInfo));
    CHK_RET(ReportHostNotifyRecordTask(signalInfo.resId, beginTime, isMaster_));
#endif
    return HCCL_SUCCESS;
}

HcclResult CpuTsThread::LocalNotifyWait([[maybe_unused]] uint32_t notifyIdx, [[maybe_unused]] uint32_t timeOut) const
{
#ifndef CCL_KERNEL_AICPU
    u64 beginTime = hcomm::GetProfCycleTime();
    HCCL_INFO("[%s]notifyIdx[%u], timeOut[%u]s.", __func__, notifyIdx, timeOut);
    CHK_PRT_RET(
        !hcomm::EqualToOrGreaterThan950().value_or(false),
        HCCL_ERROR("[CpuTsThread][%s]only supports DEV_TYPE_950 and later", __func__),
        HCCL_E_NOT_SUPPORT); // 只支持950及后续代际, 其他场景调用HcclLocalNotifyWait

    Stream* stream = GetStream();
    CHK_PTR_NULL(stream);
    LocalNotify* notify = GetNotify(notifyIdx);
    CHK_PTR_NULL(notify);

    HcclResult ret = notify->Wait(*stream, timeOut);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s]fail, notifyIdx[%u], timeOut[%u]s.", __func__, notifyIdx, timeOut), ret);

    HcclSignalInfo signalInfo;
    CHK_RET(notify->GetNotifyData(signalInfo));
    CHK_RET(ReportHostNotifyWaitTask(signalInfo.resId, beginTime, isMaster_));
#endif
    return HCCL_SUCCESS;
}

HcclResult CpuTsThread::LocalCopy(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t sizeByte) const
{
#ifndef CCL_KERNEL_AICPU
    u64 beginTime = hcomm::GetProfCycleTime();
    HCCL_INFO("[%s]dst[%p], src[%p], sizeByte[%llu].", __func__, dst, src, sizeByte);
    CHK_PRT_RET(
        !hcomm::EqualToOrGreaterThan950().value_or(false),
        HCCL_ERROR("[CpuTsThread][%s]only supports DEV_TYPE_950 and later", __func__),
        HCCL_E_NOT_SUPPORT); // 只支持950及后续代际, 其他场景调用HcclLocalCopy

    if (sizeByte == 0 || src == dst) {
        HCCL_INFO("[CpuTsThread][%s]skip, dst[%p] equals src[%p] or len[%llu] equals 0", __func__, dst, src, sizeByte);
        return HCCL_SUCCESS;
    }

    Stream* stream = GetStream();
    CHK_PTR_NULL(stream);
    CHK_RET(hrtMemAsyncCopy(
        dst, sizeByte, src, sizeByte, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream->ptr()));

    CHK_RET(ReportHostLocalCopyTask(dst, src, sizeByte, beginTime, isMaster_));
#endif
    return HCCL_SUCCESS;
}

HcclResult CpuTsThread::LocalReduce(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t sizeByte,
    [[maybe_unused]] HcommDataType dataType, [[maybe_unused]] HcommReduceOp reduceOp) const
{
#ifndef CCL_KERNEL_AICPU
    u64 beginTime = hcomm::GetProfCycleTime();
    HCCL_INFO(
        "[%s]dst[%p], src[%p], sizeByte[%llu], dataType[%d], reduceOp[%d].", __func__, dst, src, sizeByte, dataType,
        reduceOp);
    CHK_PRT_RET(
        !hcomm::EqualToOrGreaterThan950().value_or(false),
        HCCL_ERROR("[CpuTsThread][%s]only supports DEV_TYPE_950 and later", __func__),
        HCCL_E_NOT_SUPPORT); // 只支持950及后续代际, 其他场景调用HcclLocalCopyReduce

    auto dataTypeIt = hccl2rtDataTypeMap.find(static_cast<HcclDataType>(dataType));
    if (dataTypeIt == hccl2rtDataTypeMap.end()) {
        HCCL_ERROR(
            "[%s]data type[%s] is not supported", __func__,
            GetDataTypeEnumStr(static_cast<HcclDataType>(dataType)).c_str());
        return HCCL_E_PARA;
    }

    auto reduceOpIt = hccl2rtReduceOpMap.find(static_cast<HcclReduceOp>(reduceOp));
    if (reduceOpIt == hccl2rtReduceOpMap.end()) {
        HCCL_ERROR(
            "[%s]reduceOp[%s] is not supported", __func__,
            GetReduceOpEnumStr(static_cast<HcclReduceOp>(reduceOp)).c_str());
        return HCCL_E_PARA;
    }

    Stream* stream = GetStream();
    CHK_PTR_NULL(stream);
    CHK_RET(hrtReduceAsync(dst, sizeByte, src, sizeByte, reduceOpIt->second, dataTypeIt->second, stream->ptr()));
    CHK_RET(ReportHostLocalReduceTask(dst, src, sizeByte, dataType, reduceOp, beginTime, isMaster_));
#endif
    return HCCL_SUCCESS;
}
bool CpuTsThread::GetMaster() const { return isMaster_; }

void CpuTsThread::SetIsMaster(bool isMaster) { isMaster_ = isMaster; }

HcclResult CpuTsThread::SupplementNotify(uint32_t notifyNum)
{
    if (streamType_ == StreamType::STREAM_TYPE_DEVICE || notifyLoadType_ == NotifyLoadType::DEVICE_NOTIFY) {
        HCCL_ERROR("[%s]Does not support this interface.", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    if (!injectedNotifys_.empty()) {
        // GE保序注入线程：notify由管理器统一创建管理（数量固定），不支持动态补充
        HCCL_ERROR("[%s]Does not support supplement for injected notify thread.", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("[%s]supplement notifyNum[%u], notifyNum_[%u]", __func__, notifyNum, notifyNum_);

    u32 currentNotifyNum = notifyNum_;
    notifyNum_ += notifyNum;
    notifys_.reserve(notifyNum_);
    for (uint32_t idx = currentNotifyNum; idx < notifyNum_; idx++) {
        notifys_.emplace_back(nullptr);
        notifys_[idx].reset(new (std::nothrow) LocalNotify());
        CHK_SMART_PTR_NULL(notifys_[idx]);
        CHK_RET(notifys_[idx]->Init(notifyLoadType_));
        if (!hcomm::EqualToOrGreaterThan950().value_or(false)) {
            CHK_RET(notifys_[idx]->SetIpc());
        }
    }

    uniqueIdStr_.clear();
    UpdateUniqueId();
    return HCCL_SUCCESS;
}
} // namespace hccl
