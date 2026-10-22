/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RTSQ_A5_H
#define HCCLV2_RTSQ_A5_H
#include "rtsq_base.h"
#include "sqe_v82.h"
#include "log.h"
#include "res_pub.h"
#include <chrono>
#include <vector>

namespace hccl {
class AicpuTsThread;
}

namespace Hccl {

// RTSQ SQE 槽位元数据:记录每条DbSend SQE在SQ中的absSlotIdx、所属transport、携带的piValue。
struct DbSendSlotMeta {
    UbTransportLiteImpl* transport{nullptr};
    u32 absSlotIdx{0};
    u16 seqIdx{0};
    u16 piValue{0};
};

class RtsqA5 : public RtsqBase {
public:
    RtsqA5(u32 devPhyId, u32 streamId, u32 sqId);

    RtsqA5(u32 devPhyId, u32 streamId, u32 sqId, bool launchFlag);

    void Reset(bool reset) override;

    void LaunchTask() override;

    void TryLaunchTask() override;

    void NotifyWait(u32 notifyId) override;

    void NotifyWait(u32 notifyId, u32 timeout);

    void NotifyRecordLoc(u32 notifyId) override;

    void Cnt1toNNotifyWait(u32 notifyId, u32 value) override;

    void Cnt1toNNotifyRecord(u32 notifyId, u32 value) override;

    void CntNto1NotifyWait(u32 notifyId, u32 value) override;

    void CntNto1NotifyRecord(u32 notifyId, u32 value) override;

    void SdmaCopy(u64 srcAddr, u64 dstAddr, u32 size, u32 partId) override;

    void SdmaReduce(u64 srcAddr, u64 dstAddr, u32 size, u32 partId, const ReduceIn& reduceIn) override;

    void P2PWriteValue(u64 remoteAddr, u32 writeValue) override;

    void UbDbSend(const UbJettyLiteId& jettyLiteId, u16 piValue, u16 seqIdx, UbTransportLiteImpl* transport) override;

    void RdmaDbSend(const uint64_t& dbAddr, const uint64_t& dbValue) override;

    void UbDirectSend(const UbJettyLiteId& jettyLiteId, u32 dwqeSize, const u8* wqe) override
    {
        // 构造UBDMA的command，这个里面，SQE可能占用 128Byte 或者 192Byte
        (void)jettyLiteId;
        (void)dwqeSize;
        (void)wqe;
    }

    void UbWriteValue(u64 dbAddr, u32 piValue) override
    {
        (void)dbAddr;
        (void)piValue;
    }

    bool IsRtsqQueueSpaceSufficient() override;

    HcclResult CCoreNotifyWait(u64 waitAddr, u64 curTurnCntAddr, bool last) override;

    HcclResult CCoreNotifyRecord(u64 recordAddr, u64 curTurnCntAddr) override;

    HcclResult SetPreStreamSyncReady() override;

    HcclResult SetPreStreamSyncFin() override;

    bool GetPreStreamSyncStatus() override;

    HcclResult GetLastStreamIdAndTaskId(uint16_t& streamId, uint16_t& taskId) const override;

    // 用于aicpu task cache
    uint32_t GetPendingSqeCnt() const override { return pendingSqeCnt; }

    inline HcclResult SetAicpuTsThreadPtr(hccl::AicpuTsThread* threadPtr)
    {
        CHK_PTR_NULL(threadPtr);
        aicpuTsThreadPtr_ = threadPtr;
        return HCCL_SUCCESS;
    }

    inline HcclResult SetNeedCacheTaskCallback(std::function<bool()> callback)
    {
        CHK_PTR_NULL(callback);
        needCacheTaskCallback_ = callback;
        return HCCL_SUCCESS;
    }

    inline HcclResult SetAddSqeArrayCallback(std::function<HcclResult(
                                                 Hccl::RtsqA5*, hccl::AicpuTsThread*, uint64_t, const uint8_t*,
                                                 const uint32_t, const u64*, const u8*, const u8*)>
                                                 callback)
    {
        CHK_PTR_NULL(callback);
        addSqeArrayCallback_ = callback;
        return HCCL_SUCCESS;
    }

    void RefreshSqeHeaderTaskField(Rt91095StarsSqeHeader* sqeHeaderPtr);

    HcclResult LaunchNewTask(uint8_t* sqeArray, uint32_t sqeCount);

    // aicpu task cache命中时, 替代UbDbSend中的dbSendSlots_记录逻辑, 用于ciTracker的CI同步
    // 注意: cache命中时pendingSqeCnt为0, DbSqe在SQE数组中的偏移即为dbSqeIdx
    // 参考UbDbSend: absSlotIdx = (sqTail_ + pendingSqeCnt) % sqDepth_, 此处pendingSqeCnt等价为dbSqeIdx
    void RecordDbSendSlot(UbTransportLiteImpl* transport, u32 dbSqeIdx, u16 seqIdx, u16 piValue)
    {
        u32 absSlotIdx = (sqTail_ + dbSqeIdx) % sqDepth_;
        DbSendSlotMeta& slot = dbSendSlots_[dbSendTail_];
        slot.transport = transport;
        slot.absSlotIdx = absSlotIdx;
        slot.seqIdx = seqIdx;
        slot.piValue = piValue;
        dbSendTail_ = (dbSendTail_ + 1) % dbSendDepth_;
    }
    u64 GetSqeAddr() const override;

    void SetCurrentTransportContext(u64 channelHandle, u8 linkType, u8 transportType) override;

    void PollCompletion() override;

private:
    u32 pendingSqeCnt{0};

    u32 sqFullTimeout_ = RTSQ_FULL_TIMEOUT_DEFAULT;

    bool isPreStreamSync = false;

    bool launchFlag_ = false;

    u64 lastSqeAddr_{0};

    u8 locBuf[RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT]{0};

    u64 pendingSqeTransports_[PER_LAUNCH_SQE_CNT]{DFX_INVALID_U64};
    u8 pendingSqeLinkTypes_[PER_LAUNCH_SQE_CNT]{0};
    u8 pendingSqeTransportTypes_[PER_LAUNCH_SQE_CNT]{0};
    u64 currentTransport_{DFX_INVALID_U64};
    u8 currentLinkType_{0};
    u8 currentTransportType_{static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL)};

    std::function<bool()> needCacheTaskCallback_{nullptr};
    std::function<HcclResult(
        Hccl::RtsqA5*, hccl::AicpuTsThread*, uint64_t, const uint8_t*, const uint32_t, const u64*, const u8*,
        const u8*)>
        addSqeArrayCallback_{nullptr};
    hccl::AicpuTsThread* aicpuTsThreadPtr_{nullptr};

    u8* GetCurrSqeBuffer();

    void RefreshInfo();

    void RecordCCoreDfxTask(bool isWait);

    HcclResult CopySqeBufToSq(u8* sqeBuf) const;

    HcclResult MakeSureAvailableSpace();

    u32 GetTailToHeadDist() const;

    void CheckLaunchTaskStatus(
        const std::chrono::steady_clock::time_point& startTime, const std::chrono::steady_clock::time_point& curTime);

    void PreLaunchSqeForCache(bool& needCacheTask);

    void PostLaunchSqeForCache();

    u32 dbSendDepth_{0};
    u32 dbSendHead_{0};
    u32 dbSendTail_{0};
    u32 lastHead_{0};
    std::vector<DbSendSlotMeta> dbSendSlots_;
    std::vector<std::pair<u16, u16>> pollSnapshot_; // (seqIdx, piValue)，复用避免每次PollCompletion栈上分配
};

} // namespace Hccl

#endif
