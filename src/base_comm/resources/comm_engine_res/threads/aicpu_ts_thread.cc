/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ts_thread.h"
#include "hccl_common.h"
#include "aicpu/aicpu_hccl_sqcq.h"
#include "types/dev_type.h"
#include "device_capacity.h"
#include <atomic>
#include <string>

namespace hccl {
AicpuTsThread::AicpuTsThread(StreamType streamType, uint32_t notifyNum, const NotifyLoadType notifyLoadType)
    : streamType_(streamType),
      notifyNum_(notifyNum),
      notifyLoadType_(notifyLoadType)
{
}

AicpuTsThread::AicpuTsThread(const std::string &uniqueIdStr) : uniqueIdStr_(uniqueIdStr)
{
}

AicpuTsThread::~AicpuTsThread()
{
    DeInit();
}

HcclResult AicpuTsThread::Init()
{
    CHK_RET(GetRunSideIsDevice(isDeviceSide_));
    if (!isDeviceSide_) {
        // host侧申请资源
        HCCL_INFO("HcclThread::%s, is hostside", __func__);
        return HostInit();
    } else {
        // device侧反序列化，恢复资源
        HCCL_INFO("HcclThread::%s, is DeviceSide", __func__);
        return DeviceInit();
    }
}

HcclResult AicpuTsThread::DeInit()
{
    streamType_ = StreamType::STREAM_TYPE_RESERVED;
    notifyNum_ = 0;
    if (stream_) {
        stream_->SetInvalidFlag();
    }
    stream_ = nullptr;
    notifys_.clear();
    if (threadDispatcherCtx_ != nullptr) {
        (void)DestroyDispatcherCtx(threadDispatcherCtx_, threadDispatcherCommId_.c_str());
        threadDispatcherCtx_ = nullptr;
        threadDispatcherCommId_ = std::string();
    }
    uniqueIdStr_ = std::string();
    devType_ = DevType::DEV_TYPE_COUNT;
    return HCCL_SUCCESS;
}

std::string &AicpuTsThread::GetUniqueId()
{
    if (!uniqueIdStr_.empty()) {
        return uniqueIdStr_;
    }

    return UpdateUniqueId();
}

std::string &AicpuTsThread::UpdateUniqueId()
{
    // 序列化信息
    std::ostringstream oss;
    oss.write(reinterpret_cast<const char_t *>(&streamType_), sizeof(streamType_));
    oss.write(reinterpret_cast<const char_t *>(&notifyLoadType_), sizeof(notifyLoadType_));
    oss.write(reinterpret_cast<const char_t *>(&devId_), sizeof(devId_));
    oss.write(reinterpret_cast<const char_t *>(&notifyNum_), sizeof(notifyNum_));

    HcclStreamParam streamParam;
    streamParam.streamInfo.streamIds = stream_->id();
    streamParam.streamInfo.sqIds = stream_->sqId();
    streamParam.streamInfo.cqIds = stream_->cqId();
    streamParam.streamInfo.logicCqids = stream_->logicCqId();
    streamParam.sqCqContextAddr = reinterpret_cast<uint64_t>(sqCqeContext_.ptr());
    streamParam.sqCqContextSize = sqCqeContext_.size();
    oss.write(reinterpret_cast<const char_t *>(&streamParam), sizeof(streamParam));

    HcclResult ret = HCCL_SUCCESS;
    for (uint32_t idx = 0; idx < notifyNum_; idx++) {
        HcclSignalInfo notifyInfo;
        ret = notifys_[idx]->GetNotifyData(notifyInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[AicpuTsThread][UpdateUniqueId]GetNotifyData failed, ret[%d]", ret);
            uniqueIdStr_ = std::string();
            return uniqueIdStr_;
        }
        HCCL_INFO("[AicpuTsThread][UpdateUniqueId]get local notify data success, resId[%u], tsId[%d], devId[%u]",
            notifyInfo.resId, notifyInfo.tsId, notifyInfo.devId);
        oss.write(reinterpret_cast<const char_t *>(&notifyInfo), sizeof(notifyInfo));
    }
    HCCL_DEBUG("[AicpuTsThread][UpdateUniqueId] stream[%p], notifyNum[%u]", stream_->ptr(), notifyNum_);

    uniqueIdStr_ = oss.str();
    return uniqueIdStr_;
}

#ifdef CCL_KERNEL_AICPU
HcclResult AicpuTsThread::BuildComStreamInfo(const HcclStreamInfo &streamInfo, HcclComStreamInfo &comStreamInfo) const
{
    comStreamInfo.sqId = streamInfo.sqIds;
    comStreamInfo.actualStreamId = streamInfo.streamIds;
    comStreamInfo.logicCqId = streamInfo.logicCqids;
    u64 sqAddr = 0;
    CHK_RET(QuerySqBaseAddr(devId_, streamInfo.sqIds, sqAddr));
    comStreamInfo.sqBaseAddr = reinterpret_cast<void *>(sqAddr);
    if (comStreamInfo.sqBaseAddr == nullptr) {
        HCCL_ERROR("[AicpuTsThread::InitStream] sqe base addr ptr is null.");
        return HCCL_E_PARA;
    }
    CHK_RET(QuerySqStatusByType(devId_, streamInfo.sqIds, DRV_SQCQ_PROP_SQ_DEPTH, comStreamInfo.sqDepth));
    HCCL_DEBUG("[AicpuTsThread::InitStream] get stream data success, "
               "streamId[%d], sqId[%d], logicCqId[%u], sqDepth[%u]",
        comStreamInfo.actualStreamId, comStreamInfo.sqId, comStreamInfo.logicCqId, comStreamInfo.sqDepth);
    return HCCL_SUCCESS;
}
#endif

HcclResult AicpuTsThread::InitStream(HcclStreamParam &streamParam)
{
#ifdef CCL_KERNEL_AICPU
    HcclStreamInfo &streamInfo = streamParam.streamInfo;

    static bool isCustom = false;
    static bool init = false;

    if (UNLIKELY(!init)) {
        uint32_t cpType = DEVDRV_PROCESS_CPTYPE_MAX;
        unsigned int hostpid = 0;
        CHK_RET(HrtHalDrvQueryProcessHostPid(getpid(), nullptr, nullptr, &hostpid, &cpType));
        isCustom = cpType == static_cast<uint32_t>(DEVDRV_PROCESS_CP2) ? true : false;
        init = true;
    }
    HcclResult ret = hrtHalResourceIdRestore(devId_, 0, DRV_STREAM_ID, streamInfo.streamIds, 0);
    // custom进程需要恢复stream资源, custom进程调用失败直接报错，aicpu进程调用失败做兼容性处理
    if (ret == HCCL_E_NOT_SUPPORT) {
        CHK_PRT_RET(isCustom,
            HCCL_ERROR(
                "%s hrtHalResourceIdRestore fail, drv not support, custom[%d], ret[%d]", __func__, isCustom, ret),
            HCCL_E_DRV);
    } else if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("%s hrtHalResourceIdRestore fail, ret[%d]", __func__, ret);
        return HCCL_E_DRV;
    }

    HcclComStreamInfo comStreamInfo{0};
    CHK_RET(BuildComStreamInfo(streamInfo, comStreamInfo));

    stream_.reset(new (std::nothrow) Stream(comStreamInfo));
    CHK_SMART_PTR_NULL(stream_);

    // 初始化stream的sqeContext
    SqCqeContext *sqCqeContext = reinterpret_cast<SqCqeContext *>(streamParam.sqCqContextAddr);
    uint64_t sqCqContextSize = streamParam.sqCqContextSize;
    if (sqCqeContext == nullptr || sqCqContextSize != sizeof(SqCqeContext)) {
        HCCL_ERROR("%s fail, sqCqeContext[%p] is null or size[%llu] is not equal to SqCqeContext size[%llu]", __func__,
            sqCqeContext, sqCqContextSize, sizeof(SqCqeContext));
        return HCCL_E_PARA;
    }
    sqCqeContext_ = DeviceMem::create(reinterpret_cast<void *>(sqCqeContext), sqCqContextSize);

    uint32_t sqTail = 0;
    uint32_t sqHead = 0;
    CHK_RET(QuerySqStatusByType(devId_, streamInfo.sqIds, DRV_SQCQ_PROP_SQ_TAIL, sqTail));
    CHK_RET(QuerySqStatusByType(devId_, streamInfo.sqIds, DRV_SQCQ_PROP_SQ_HEAD, sqHead));
    HCCL_DEBUG("[AicpuTsThread::InitStream] sqHead[%u], sqTail[%u]", sqHead, sqTail);

    ret = stream_->InitSqAndCqeContext(sqHead, sqTail, sqCqeContext);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("%s InitSqAndCqeContext failed", __func__), ret);
    HCCL_INFO("%s success, streamId[%d]", __func__, stream_->id());
#endif
    return HCCL_SUCCESS;
}

HcclResult AicpuTsThread::InitStreamLite(HcclStreamInfo &streamParam, uint32_t hostPhyId)
{
    // 在aicpu侧查询cqe时，需要使用logicCqids，而不是cqIds
    EXCEPTION_CATCH(pImpl_ = std::make_unique<Hccl::IAicpuTsThread>(
                        streamParam.streamIds, streamParam.sqIds, hostPhyId, streamParam.logicCqids),
        return HCCL_E_PTR);
    return HCCL_SUCCESS;
}

uint32_t AicpuTsThread::GetNotifyNum() const
{
    return notifyNum_;
}

LocalNotify *AicpuTsThread::GetNotify(uint32_t index) const
{
    if (UNLIKELY(index >= notifyNum_)) {
        HCCL_ERROR("[AicpuTsThread][GetNotify] notifyNum[%u], index[%u] out of range[0, %u)", notifyNum_, index,
            notifyNum_ - 1);
        return nullptr;
    }
    return notifys_[index].get();
}

// A3 Stream
Stream *AicpuTsThread::GetStream() const
{
    // 幂等绑定Thread私有dispatcher：本thread上的所有dispatcher操作统一使用私有dispatcher，
    // 保证(dispatcher, stream)配对唯一；绑定变化时才调用Set，避免热路径日志刷屏
    if (threadDispatcherCtx_ != nullptr && GetDispatcherCtx() != threadDispatcherCtx_) {
        (void)SetDispatcherCtx(threadDispatcherCtx_);
    }
    return stream_.get();
}

void AicpuTsThread::LaunchTask() const
{
    pImpl_->LaunchTask();
    return;
}

void AicpuTsThread::TryLaunchTask() const
{
    pImpl_->TryLaunchTask();
    return;
}

// Local Data Plane Functions
HcclResult AicpuTsThread::LocalNotifyWait(uint32_t notifyId) const
{
    HCCL_ERROR("[AicpuTsThread][%s] without timeout not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsThread::LocalNotifyRecord(uint32_t notifyId) const
{
    u64 beginTime = ProfGetCurCpuTimestamp();
    void *streamLitePtr = GetStreamLitePtr();
    Hccl::StreamLite *streamLite = static_cast<Hccl::StreamLite *>(streamLitePtr);
    Hccl::RtsqBase *rtsq = streamLite->GetRtsq();
    u32 taskId = rtsq->GetTaskId();

    CHK_RET(pImpl_->NotifyRecordLoc(notifyId));

    CHK_RET(ReportAicpuNotifyRecordTask(notifyId, beginTime, taskId, streamLite->GetSqId()));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsThread::LocalNotifyRecord(ThreadHandle dstThread, uint32_t dstNotifyIdx) const
{
    HCCL_ERROR("[AicpuTsThread][%s]not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsThread::LocalNotifyWait(uint32_t notifyId, uint32_t timeout) const
{
    u64 beginTime = ProfGetCurCpuTimestamp();
    void *streamLitePtr = GetStreamLitePtr();
    Hccl::StreamLite *streamLite = static_cast<Hccl::StreamLite *>(streamLitePtr);
    Hccl::RtsqBase *rtsq = streamLite->GetRtsq();
    u32 taskId = rtsq->GetTaskId();

    CHK_RET(pImpl_->NotifyWait(notifyId, timeout));

    CHK_RET(ReportAicpuNotifyWaitTask(notifyId, beginTime, taskId, streamLite->GetSqId()));

    return HCCL_SUCCESS;
}

template <typename Operation, typename ReportOp>
HcclResult AicpuTsThread::LocalProcess(
    void *dst, const void *src, uint64_t sizeByte, Operation &&op, ReportOp &&reportOp) const
{
    void *streamLitePtr = GetStreamLitePtr();
    Hccl::StreamLite *streamLite = static_cast<Hccl::StreamLite *>(streamLitePtr);
    Hccl::RtsqBase *rtsq = streamLite->GetRtsq();

    uint64_t dstAddr = reinterpret_cast<uint64_t>(dst);
    uint64_t srcAddr = reinterpret_cast<uint64_t>(src);
    uint8_t *dstByte = static_cast<uint8_t *>(dst);
    const uint8_t *srcByte = static_cast<const uint8_t *>(src);

    uint64_t remainSize = sizeByte;
    uint64_t doneSize = 0;

    while (remainSize > 0) {
        uint64_t realSize = remainSize > SDMA_SEND_MAX_SIZE ? SDMA_SEND_MAX_SIZE : remainSize;
        u64 beginTime = ProfGetCurCpuTimestamp();
        u32 taskId = rtsq->GetTaskId();

        CHK_RET(op(dstAddr + doneSize, srcAddr + doneSize, realSize));
        CHK_RET(reportOp(dstByte + doneSize, srcByte + doneSize, realSize, beginTime, taskId, streamLite->GetSqId()));

        doneSize += realSize;
        remainSize -= realSize;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsThread::LocalCopy(void *dst, const void *src, uint64_t size) const
{
    return LocalProcess(
        dst, src, size,
        [this](uint64_t dst, uint64_t src, uint64_t size) {
            return pImpl_->SdmaCopy(dst, src, size);
        },
        [this](void *dst, const void *src, uint64_t size, uint64_t beginTime, uint32_t taskId, uint32_t sqId) {
            return ReportAicpuLocalCopyTask(dst, src, size, beginTime, taskId, sqId);
        });
}

HcclResult AicpuTsThread::LocalReduce(
    void *dst, const void *src, uint64_t size, HcommDataType dataType, HcommReduceOp reduceOp) const
{
    uint32_t dataTypeRaw = static_cast<uint32_t>(dataType);
    uint32_t reduceOpRaw = static_cast<uint32_t>(reduceOp);
    return LocalProcess(
        dst, src, size,
        [this, &dataTypeRaw, &reduceOpRaw](uint64_t d, uint64_t s, uint64_t size) {
            return pImpl_->SdmaReduce(d, s, size, dataTypeRaw, reduceOpRaw);
        },
        [this, &dataType, &reduceOp](
            void *dst, const void *src, uint64_t size, uint64_t beginTime, uint32_t taskId, uint32_t sqId) {
            return ReportAicpuLocalReduceTask(dst, src, size, dataType, reduceOp, beginTime, taskId, sqId);
        });
}

// Private functions
HcclResult AicpuTsThread::HostInit()
{
    CHK_PRT_RET(!uniqueIdStr_.empty(), HCCL_ERROR("[AicpuTsThread][Init]not support init with uniqueId on host"),
        HCCL_E_NOT_SUPPORT);
    s32 deviceLogicId;
    CHK_RET(hrtGetDevice(&deviceLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(deviceLogicId), devId_));
    CHK_RET(hrtGetDeviceType(devType_));
    if (rtStream_ == nullptr) {
        stream_.reset(new (std::nothrow) Stream(streamType_));
        CHK_SMART_PTR_NULL(stream_);
        rtStream_ = stream_->ptr();
    }

    for (uint32_t idx = 0; idx < notifyNum_; idx++) {
        notifys_.emplace_back(nullptr);
        notifys_[idx].reset(new (std::nothrow) LocalNotify());
        CHK_SMART_PTR_NULL(notifys_[idx]);
        CHK_RET(notifys_[idx]->Init(notifyLoadType_));
        if (devType_ != DevType::DEV_TYPE_950) {
            CHK_RET(notifys_[idx]->SetIpc());
        }
    }

    // A5 aicpu场景thread多申请一个host类型notify，用于host&device同步
    if (devType_ == DevType::DEV_TYPE_950) {
        notifys_.emplace_back(nullptr);
        notifys_[notifyNum_].reset(new (std::nothrow) LocalNotify());
        CHK_SMART_PTR_NULL(notifys_[notifyNum_]);
        CHK_RET(notifys_[notifyNum_]->Init(NotifyLoadType::HOST_NOTIFY));
        notifyNum_ += 1;
    }

    if (streamType_ == StreamType::STREAM_TYPE_DEVICE && devType_ != DevType::DEV_TYPE_950) {
        uint64_t size = sizeof(SqCqeContext);
        sqCqeContext_ = DeviceMem::alloc(size);
        CHK_PTR_NULL(sqCqeContext_.ptr());
        CHK_RET(hrtMemSet(sqCqeContext_.ptr(), size, size));
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsThread::DeviceInit()
{
    CHK_PRT_RET(uniqueIdStr_.empty(), HCCL_ERROR("[AicpuTsThread][Init]uniqueIdStr is empty"), HCCL_E_INTERNAL);
    std::istringstream iss(uniqueIdStr_);
    CHK_RET(hrtGetDeviceType(devType_));
    uint32_t hostPhyId = 0;
    iss.read(reinterpret_cast<char_t *>(&streamType_), sizeof(streamType_));
    iss.read(reinterpret_cast<char_t *>(&notifyLoadType_), sizeof(notifyLoadType_));
    HCCL_INFO("[AicpuTsThread][Init]streamType[%d], notifyLoadType[%d].", streamType_, notifyLoadType_);
    iss.read(reinterpret_cast<char_t *>(&hostPhyId), sizeof(hostPhyId));
    CHK_RET(hrtDrvGetLocalDevIDByHostDevID(hostPhyId, &devId_));
    iss.read(reinterpret_cast<char_t *>(&notifyNum_), sizeof(notifyNum_));

    HcclStreamParam streamParam;
    iss.read(reinterpret_cast<char_t *>(&streamParam), sizeof(streamParam));
    // 91095初始化streamlite，初始化rtsq接口
    if (devType_ == DevType::DEV_TYPE_950) {
        CHK_RET(InitStreamLite(streamParam.streamInfo, hostPhyId));
    } else {
        CHK_RET(InitStream(streamParam));
    }

    // dispatcher归属Thread：随thread创建/销毁，确保(dispatcher, stream)配对稳定，
    // channel复用thread时不会出现同一stream被多个dispatcher交错写入。
    // commId用devId+进程内自增序列保证唯一（uniqueIdStr_为二进制序列，c_str会在首个\0截断，不可直接拼接）
    static std::atomic<uint64_t> threadDispatcherSeq{0};
    threadDispatcherCommId_
        = "hcomm_ts_thread_" + std::to_string(devId_) + "_" + std::to_string(threadDispatcherSeq.fetch_add(1));
    HcclResult dispatcherRet = CreateDispatcherCtx(&threadDispatcherCtx_, devId_, threadDispatcherCommId_.c_str());
    if (dispatcherRet != HCCL_SUCCESS || threadDispatcherCtx_ == nullptr) {
        // 创建失败不阻断thread初始化：任务下发时GetOpDispatcher()会显式报错，保证可诊断
        HCCL_WARNING("[AicpuTsThread][Init] create thread dispatcher failed, ret[%d], commId[%s]",
            static_cast<int>(dispatcherRet), threadDispatcherCommId_.c_str());
        threadDispatcherCtx_ = nullptr;
    } else {
        HCCL_INFO(
            "[AicpuTsThread][Init] thread dispatcher create success, commId[%s]", threadDispatcherCommId_.c_str());
    }

    notifys_.reserve(notifyNum_);

    for (uint32_t idx = 0; idx < notifyNum_; idx++) {
        notifys_.emplace_back(nullptr);
        HcclSignalInfo notifyInfo;
        iss.read(reinterpret_cast<char_t *>(&notifyInfo), sizeof(notifyInfo));
        notifys_[idx].reset(new (std::nothrow) LocalNotify());
        CHK_SMART_PTR_NULL(notifys_[idx]);
        if (devType_ == DevType::DEV_TYPE_950) {
            CHK_RET(notifys_[idx]->InitNotifyLite(notifyInfo));
            HCCL_INFO("[AicpuTsThread][Init]local notifyLite init success, resId[%u], devId[%u]", notifyInfo.resId,
                notifyInfo.devId);
        } else {
            CHK_RET(notifys_[idx]->Init(notifyInfo, notifyLoadType_));
            HCCL_INFO("[AicpuTsThread][Init]local notifyLite init success, resId[%u], tsId:%d, devId[%u]",
                notifyInfo.resId, notifyInfo.tsId, notifyInfo.devId);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTsThread::GetSqHeadAndTail(uint32_t &sqHead, uint32_t &sqTail)
{
#ifdef CCL_KERNEL_AICPU

    uint32_t sqIds = pImpl_->GetSqId();

    CHK_RET(QuerySqStatusByType(devId_, sqIds, DRV_SQCQ_PROP_SQ_TAIL, sqTail));
    CHK_RET(QuerySqStatusByType(devId_, sqIds, DRV_SQCQ_PROP_SQ_HEAD, sqHead));
#endif
    return HCCL_SUCCESS;
}

bool AicpuTsThread::GetMaster() const
{
    return isMaster_;
}

void AicpuTsThread::SetIsMaster(bool isMaster)
{
    isMaster_ = isMaster;
}

HcclResult AicpuTsThread::SupplementNotify(uint32_t notifyNum)
{
    HCCL_INFO("[%s]supplement notifyNum[%u], notifyNum_[%u]", __func__, notifyNum, notifyNum_);
    // A5 aicpu场景thread多申请一个host类型notify，用于host&device同步
    u32 beginIdx = notifyNum_;
    u32 allNotifyNum = notifyNum_ + notifyNum;
    u32 endIdx = allNotifyNum - 1;
    notifys_.resize(allNotifyNum);
    if (devType_ == DevType::DEV_TYPE_950) {
        beginIdx--;
        CHK_SMART_PTR_NULL(notifys_[beginIdx]);
        notifys_[endIdx] = std::move(notifys_[beginIdx]);
    }

    for (uint32_t idx = beginIdx; idx < endIdx; idx++) {
        notifys_[idx].reset(new (std::nothrow) LocalNotify());
        CHK_SMART_PTR_NULL(notifys_[idx]);
        CHK_RET(notifys_[idx]->Init(notifyLoadType_));
        if (devType_ != DevType::DEV_TYPE_950) {
            CHK_RET(notifys_[idx]->SetIpc());
        }
        notifyNum_++;
    }

    uniqueIdStr_.clear();
    UpdateUniqueId();
    return HCCL_SUCCESS;
}

HcclResult AicpuTsThread::GetNotifyByUniqueId(u32 &notifyNum, std::string &notifyDesc)
{
    CHK_PRT_RET(
        uniqueIdStr_.empty(), HCCL_ERROR("[AicpuTsThread][GetNotifyByUniqueId]uniqueIdStr is empty"), HCCL_E_INTERNAL);
    std::istringstream iss(uniqueIdStr_);
    StreamType streamType = StreamType::STREAM_TYPE_RESERVED;
    NotifyLoadType notifyLoadType = NotifyLoadType::HOST_NOTIFY;
    uint32_t hostPhyId = 0;
    HcclStreamParam streamParam;
    iss.read(reinterpret_cast<char_t *>(&streamType), sizeof(streamType));
    iss.read(reinterpret_cast<char_t *>(&notifyLoadType), sizeof(notifyLoadType));
    iss.read(reinterpret_cast<char_t *>(&hostPhyId), sizeof(hostPhyId));
    iss.read(reinterpret_cast<char_t *>(&notifyNum), sizeof(notifyNum));
    iss.read(reinterpret_cast<char_t *>(&streamParam), sizeof(streamParam));

    // 序列化信息
    std::ostringstream oss;
    for (uint32_t idx = 0; idx < notifyNum; idx++) {
        HcclSignalInfo notifyInfo;
        iss.read(reinterpret_cast<char_t *>(&notifyInfo), sizeof(notifyInfo));
        HCCL_INFO("[AicpuTsThread][%s]get local notify data success, resId[%u], tsId:%d, devId[%u]", __func__,
            notifyInfo.resId, notifyInfo.tsId, notifyInfo.devId);
        oss.write(reinterpret_cast<const char_t *>(&notifyInfo), sizeof(notifyInfo));
    }

    notifyDesc = oss.str();
    return HCCL_SUCCESS;
}

HcclResult AicpuTsThread::SupplementNotify(u32 notifyNum, const std::string &notifyDesc)
{
    if (notifyNum <= notifyNum_) {
        HCCL_WARNING("[%s]supplement notifyNum[%u], notifyNum_[%u]", __func__, notifyNum, notifyNum_);
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[%s]supplement notifyNum[%u], notifyNum_[%u]", __func__, notifyNum, notifyNum_);

    std::istringstream iss(notifyDesc);
    // A5 aicpu场景thread多申请一个host类型notify，用于host&device同步
    u32 beginIdx = notifyNum_;
    u32 endIdx = notifyNum - 1;
    notifys_.resize(notifyNum);
    if (devType_ == DevType::DEV_TYPE_950 && notifyNum_ > 0) {
        beginIdx--;
        CHK_SMART_PTR_NULL(notifys_[beginIdx]);
        notifys_[endIdx] = std::move(notifys_[beginIdx]);
        HCCL_INFO("[AicpuTsThread][SupplementNotify]notifyId[%u] beginIdx[%u], endIdx[%u]", notifys_[endIdx]->notifyId_,
            beginIdx, endIdx);
    }
    for (uint32_t idx = 0; idx < beginIdx; idx++) {
        HcclSignalInfo notifyInfo;
        iss.read(reinterpret_cast<char_t *>(&notifyInfo), sizeof(notifyInfo));
        HCCL_INFO("[AicpuTsThread][SupplementNotify]skip init, resId[%u], tsId:%d, devId[%u]", notifyInfo.resId,
            notifyInfo.tsId, notifyInfo.devId);
    }

    for (uint32_t idx = beginIdx; idx < endIdx; idx++) {
        HcclSignalInfo notifyInfo;
        iss.read(reinterpret_cast<char_t *>(&notifyInfo), sizeof(notifyInfo));
        notifys_[idx].reset(new (std::nothrow) LocalNotify());
        CHK_SMART_PTR_NULL(notifys_[idx]);
        if (devType_ == DevType::DEV_TYPE_950) {
            CHK_RET(notifys_[idx]->InitNotifyLite(notifyInfo));
            HCCL_INFO("[AicpuTsThread][SupplementNotify]local notifyLite init success, resId[%u], devId[%u]",
                notifyInfo.resId, notifyInfo.devId);
        } else {
            CHK_RET(notifys_[idx]->Init(notifyInfo, notifyLoadType_));
            HCCL_INFO("[AicpuTsThread][SupplementNotify]local notifyLite init success, resId[%u], tsId:%d, devId[%u]",
                notifyInfo.resId, notifyInfo.tsId, notifyInfo.devId);
        }
        notifyNum_++;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsThread::SetCheckExecStatusCallback(std::function<HcclResult(bool)> callback)
{
    Hccl::StreamLite *streamLite = static_cast<Hccl::StreamLite *>(GetStreamLitePtr());
    CHK_PTR_NULL(streamLite);
    Hccl::RtsqBase *rtsq = streamLite->GetRtsq();
    CHK_PTR_NULL(rtsq);
    rtsq->SetCheckExecStatusCallback(callback);
    return HCCL_SUCCESS;
}
} // namespace hccl
