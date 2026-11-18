/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <chrono>

#include "common/cast_utils_pub.h"
#include "roce_host_loc_transport_lite_impl.h"
#include "dfx/device/dfx_res_pub.h"
#include "binary_stream.h"
#include "data_type.h"
#include "log.h"
#include "exception_util.h"
#include "internal_exception.h"
#include "dfx/device/dfx_profiling_handler_lite_pub.h"
#include "unified_platform/pub_inc/config_plf_log_v2.h"
#include "roce_host_loc_proto_defs.h"

namespace Hccl {

RoceHostLocTransportLiteImpl::RoceHostLocTransportLiteImpl(std::vector<char>& uniqueId) { Init(uniqueId); }

RoceHostLocTransportLiteImpl::~RoceHostLocTransportLiteImpl() {}

void RoceHostLocTransportLiteImpl::Init(std::vector<char>& uniqueId)
{
    BinaryStream binaryStream(uniqueId);
    u32 type;
    binaryStream >> type;
    binaryStream >> notifyNum_;
    binaryStream >> bufferNum_;
    binaryStream >> connNum_;
    binaryStream >> qpThreshold_;

    if (connNum_ == 0 || static_cast<u64>(notifyNum_) != static_cast<u64>(hcomm::RDMA_NOTIFY_NUM) * connNum_) {
        MACRO_THROW(
            InternalException,
            StringFormat(
                "Init: invalid notifyNum_[%u] / connNum_[%u]. notifyNum must be RDMA_NOTIFY_NUM[%u] * connNum_ ",
                notifyNum_, connNum_, hcomm::RDMA_NOTIFY_NUM));
    }

    std::vector<char> locNotifyUniqueIds;
    binaryStream >> locNotifyUniqueIds;
    ParseLocNotifyVec(locNotifyUniqueIds);

    std::vector<char> rmtNotifyUniqueIds;
    binaryStream >> rmtNotifyUniqueIds;
    ParseRmtNotifyVec(rmtNotifyUniqueIds);

    std::vector<char> notifyValueBufferUniqueIds;
    binaryStream >> notifyValueBufferUniqueIds;
    ParseNotifyValueBuffer(notifyValueBufferUniqueIds);

    std::vector<char> locBufferUniqueIds;
    binaryStream >> locBufferUniqueIds;
    ParseLocBufferVec(locBufferUniqueIds);

    std::vector<char> rmtBufferUniqueIds;
    binaryStream >> rmtBufferUniqueIds;
    ParseRmtBufferVec(rmtBufferUniqueIds);

    std::vector<char> connUniqueIds;
    binaryStream >> connUniqueIds;
    ParseConnVec(connUniqueIds);
}

HcclResult RoceHostLocTransportLiteImpl::SerialPollCq()
{
    constexpr int32_t SERIAL_POLL_NUM = 128;   // 一次最多回收 128 个 CQE
    constexpr int32_t SERIAL_POLL_TIMEOUT = 0; // 不等待, Poll 到多少算多少
    std::vector<int32_t> errList = {};

    HcclResult ret = PollCq(SERIAL_POLL_NUM, SERIAL_POLL_TIMEOUT, errList);
    if (ret == HCCL_E_TIMEOUT) {
        // 无 CQE 可回收, 属正常现象, 不打 ERROR 日志
        return HCCL_SUCCESS;
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[RoceHostLocTransportLiteImpl::%s] serial poll cq failed, ret[%d].", __func__, ret);
        return ret;
    }
    return HCCL_SUCCESS;
}

void RoceHostLocTransportLiteImpl::ParseLocNotifyVec(std::vector<char>& data)
{
    if (notifyNum_ == 0) {
        HCCL_WARNING("[RoceHostLocTransportLiteImpl::%s] notifyNum is 0", __func__);
        return;
    }

    u32 notifySizePerDto = data.size() / notifyNum_;

    for (u32 idx = 0; idx < notifyNum_; idx++) {
        auto start = data.begin() + idx * notifySizePerDto;
        auto end = start + notifySizePerDto;
        std::vector<char> dto(start, end);
        localNotifies_.push_back(std::make_unique<NotifyLite>(dto));
        HCCL_INFO("locNotify idx=%u, %s", idx, localNotifies_.back()->Describe().c_str());
    }
}

void RoceHostLocTransportLiteImpl::ParseRmtNotifyVec(std::vector<char>& data)
{
    if (notifyNum_ == 0) {
        HCCL_WARNING("[RoceHostLocTransportLiteImpl::%s] notifyNum is 0", __func__);
        return;
    }

    u32 rmtBufferSizePerDto = data.size() / notifyNum_;
    HCCL_INFO(
        "[RoceHostLocTransportLiteImpl::%s] Parse remote notify num=%u, sizePerDto=%u", __func__, notifyNum_,
        rmtBufferSizePerDto);

    BinaryStream binaryStream(data);
    remoteNotifies_.clear();
    u64 addr;
    u64 size;
    u32 rkey;
    for (u32 idx = 0; idx < notifyNum_; idx++) {
        binaryStream >> addr;
        binaryStream >> size;
        binaryStream >> rkey;
        RmtRmaBufferLite rdmaBufLite(addr, size, rkey);
        HCCL_INFO("idx=%u, %s", idx, rdmaBufLite.Describe().c_str());
        remoteNotifies_.emplace_back(rdmaBufLite);
    }
}

void RoceHostLocTransportLiteImpl::ParseNotifyValueBuffer(std::vector<char>& data)
{
    HCCL_INFO("[RoceHostLocTransportLiteImpl::%s] Parse notify value buffer", __func__);

    BinaryStream binaryStream(data);
    u64 addr;
    u64 size;
    u32 lkey;
    binaryStream >> addr;
    binaryStream >> size;
    binaryStream >> lkey;
    notifyValueBuffer_ = std::make_unique<RmaBufferLite>(addr, size, lkey);
}

void RoceHostLocTransportLiteImpl::ParseLocBufferVec(std::vector<char>& data)
{
    if (bufferNum_ == 0) {
        HCCL_WARNING("[RoceHostLocTransportLiteImpl::%s] bufferNum is 0", __func__);
        return;
    }

    u32 locBufferSizePerDto = data.size() / bufferNum_;
    HCCL_INFO(
        "[RoceHostLocTransportLiteImpl::%s] Parse local buffer num=%u, sizePerDto=%u", __func__, bufferNum_,
        locBufferSizePerDto);

    BinaryStream binaryStream(data);
    locBufferVec_.clear();
    u64 addr;
    u64 size;
    u32 lkey;
    for (u32 idx = 0; idx < bufferNum_; idx++) {
        binaryStream >> addr;
        binaryStream >> size;
        binaryStream >> lkey;
        RmaBufferLite rdmaBufLite(addr, size, lkey);
        HCCL_INFO("idx=%u, %s", idx, rdmaBufLite.Describe().c_str());
        locBufferVec_.emplace_back(rdmaBufLite);
    }
}

void RoceHostLocTransportLiteImpl::ParseRmtBufferVec(std::vector<char>& data)
{
    if (bufferNum_ == 0) {
        HCCL_WARNING("[RoceHostLocTransportLiteImpl::%s] bufferNum is 0", __func__);
        return;
    }

    u32 rmtBufferSizePerDto = data.size() / bufferNum_;
    HCCL_INFO(
        "[RoceHostLocTransportLiteImpl::%s] Parse remote buffer num=%u, sizePerDto=%u", __func__, bufferNum_,
        rmtBufferSizePerDto);

    BinaryStream binaryStream(data);
    rmtBufferVec_.clear();
    u64 addr;
    u64 size;
    u32 rkey;
    for (u32 idx = 0; idx < bufferNum_; idx++) {
        binaryStream >> addr;
        binaryStream >> size;
        binaryStream >> rkey;
        RmtRmaBufferLite rdmaBufLite(addr, size, rkey);
        HCCL_INFO("idx=%u, %s", idx, rdmaBufLite.Describe().c_str());
        rmtBufferVec_.emplace_back(rdmaBufLite);
    }
}

void RoceHostLocTransportLiteImpl::ParseConnVec(std::vector<char>& data)
{
    if (connNum_ == 0) {
        HCCL_WARNING("[RoceHostLocTransportLiteImpl::%s] connNum is 0", __func__);
        return;
    }

    u32 connSizePerDto = data.size() / connNum_;
    HCCL_INFO(
        "[RoceHostLocTransportLiteImpl::%s] Parse conn num=%u, sizePerDto=%u", __func__, connNum_, connSizePerDto);
    for (u32 idx = 0; idx < connNum_; idx++) {
        auto start = data.begin() + idx * connSizePerDto;
        auto end = start + connSizePerDto;
        std::vector<char> connUniqueId(start, end);
        connUniqueIdVec_.emplace_back(connUniqueId);
        std::unique_ptr<RoceHostLocConnLite> connLite;
        connLite = std::make_unique<RoceHostLocConnLite>(connUniqueId);
        HCCL_INFO("[RoceHostLocTransportLiteImpl::%s] idx=%u, %s", __func__, idx, connLite->Describe().c_str());
        connVec_.emplace_back(std::move(connLite));
    }
}

RmaBufSliceLite RoceHostLocTransportLiteImpl::GetRmaBufSlicelite(const RmaBufferLite& lite) const
{
    return RmaBufSliceLite(lite.GetAddr(), lite.GetSize(), lite.GetLkey(), 0);
}

u32 RoceHostLocTransportLiteImpl::CalcUseQpNum(u64 totalLen, u64& tileLen, u64& tileLenTail, u64 alignment) const
{
    if (connNum_ <= 1 || totalLen == 0) {
        tileLen = totalLen;
        tileLenTail = totalLen;
        return 1;
    }

    // alignment为分片对齐粒度，普通传输为1，Reduce为元素宽度，调用方保证非0
    if (alignment == 0) {
        HCCL_ERROR("[RoceHostLocTransportLiteImpl::%s] alignment must be non-zero.", __func__);
        alignment = 1;
    }

    const u64 avgTileLen = totalLen / connNum_ + (totalLen % connNum_ != 0 ? 1 : 0);
    tileLen = std::max(static_cast<u64>(qpThreshold_), avgTileLen);
    tileLen = (tileLen / alignment + (tileLen % alignment != 0 ? 1 : 0)) * alignment;

    u64 useQpNum = totalLen / tileLen + (totalLen % tileLen != 0 ? 1 : 0);
    useQpNum = std::min(useQpNum, static_cast<u64>(connNum_));
    tileLenTail = totalLen - (useQpNum - 1) * tileLen;
    return static_cast<u32>(useQpNum);
}

RmaBufSliceLite RoceHostLocTransportLiteImpl::GetNotifySlicelite(u32 index) const
{
    (void)index;
    return RmaBufSliceLite(
        notifyValueBuffer_->GetAddr(), notifyValueBuffer_->GetSize(), notifyValueBuffer_->GetLkey(), 0);
}

HcclResult RoceHostLocTransportLiteImpl::GetRmtRmaBufSliceLite(const Buffer& rmtBuf, RmtRmaBufSliceLite& rmtSlice) const
{
    for (auto& it : rmtBufferVec_) {
        Buffer buf(it.GetAddr(), it.GetSize());
        if (buf.Contains(rmtBuf.GetAddr(), rmtBuf.GetSize())) {
            rmtSlice = RmtRmaBufSliceLite(rmtBuf.GetAddr(), rmtBuf.GetSize(), it.GetRkey(), 0, 0, UINT32_MAX);
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("[%s] %s is not in current transport", __func__, rmtBuf.Describe().c_str());
    return HCCL_E_INTERNAL;
}

HcclResult RoceHostLocTransportLiteImpl::GetRmtNotifySliceLite(u32 index, RmtRmaBufSliceLite& slice) const
{
    CHK_PRT_RET(
        index >= remoteNotifies_.size(),
        HCCL_ERROR(
            "[RoceHostLocTransportLiteImpl::%s] index[%u] out of range[%zu]", __func__, index, remoteNotifies_.size()),
        HCCL_E_PARA);

    auto& lite = remoteNotifies_[index];
    slice = RmtRmaBufSliceLite(lite.GetAddr(), lite.GetSize(), lite.GetRkey(), 0, 0, UINT32_MAX);
    return HCCL_SUCCESS;
}

std::string RoceHostLocTransportLiteImpl::Describe() const
{
    std::string desc = "RoceHostLocTransportLiteImpl[";

    u32 idx = 0;
    desc += "localNotifies=[";
    for (auto& it : localNotifies_) {
        desc += StringFormat("idx=%u, %s;", idx, it->Describe().c_str());
        idx++;
    }

    idx = 0;
    desc += "], remoteNotifies=[";
    for (auto& it : remoteNotifies_) {
        desc += StringFormat("idx=%u, %s;", idx, it.Describe().c_str());
        idx++;
    }

    idx = 0;
    desc += "], locBufferVec=[";
    for (auto& it : locBufferVec_) {
        desc += StringFormat("idx=%u, %s;", idx, it.Describe().c_str());
        idx++;
    }

    idx = 0;
    desc += "], rmtBufferVec=[";
    for (auto& it : rmtBufferVec_) {
        desc += StringFormat("idx=%u, %s;", idx, it.Describe().c_str());
        idx++;
    }

    idx = 0;
    desc += "], connVec=[";
    for (auto& it : connVec_) {
        desc += StringFormat("idx=%u, %s;", idx, it->Describe().c_str());
        idx++;
    }

    desc += "]]";
    return desc;
}

HcclResult RoceHostLocTransportLiteImpl::BuildLocRmaBufferLite(
    const uintptr_t addr, const size_t size, RmaBufferLite& rmaBufferLite)
{
    HCCL_INFO(
        "[RoceHostLocTransportLiteImpl::%s] start to find addr[0x%llx], size[0x%llx] in locBufferVec, whose size is "
        "%zu. ",
        __func__, addr, size, locBufferVec_.size());

    if (locBufferVec_.empty()) {
        HCCL_ERROR("[RoceHostLocTransportLiteImpl::%s] locBufferVec is empty.", __func__);
        return HCCL_E_INTERNAL;
    }

    bool isAddrInRange = false;
    for (auto& it : locBufferVec_) {
        Buffer iterBuf(it.GetAddr(), it.GetSize());
        if (iterBuf.Contains(addr, size)) {
            rmaBufferLite = RmaBufferLite(addr, size, it.GetLkey());
            isAddrInRange = true;
            break;
        }
    }

    if (!isAddrInRange) {
        HCCL_WARNING(
            "[RoceHostLocTransportLiteImpl::%s] addr[0x%llx], size[0x%llx] not in any range of locBufferVec. The token "
            "of the "
            "first locBuffer is used.",
            __func__, addr, size);
        rmaBufferLite = RmaBufferLite(addr, size, locBufferVec_[0].GetLkey());
        return HCCL_SUCCESS;
    }

    return HCCL_SUCCESS;
}

HcclResult RoceHostLocTransportLiteImpl::Read(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream)
{
    // 获取Profiling任务ID
    auto taskId = stream.GetRtsq()->GetTaskId();

    // 获取本端和远端Buffer切片
    SqeConfigLite cfg;
    SetFenceConfig(cfg);
    auto locRmaBufSliceLite = GetRmaBufSlicelite(loc);
    RmtRmaBufSliceLite rmtRmaBufSliceLite;
    CHK_RET(GetRmtRmaBufSliceLite(rmt, rmtRmaBufSliceLite));

    u64 tileLen = 0;
    u64 tileLenTail = 0;
    u32 useQpNum = CalcUseQpNum(locRmaBufSliceLite.GetSize(), tileLen, tileLenTail);

    // Post Wqe && return dbValue
    for (u32 i = 0; i < useQpNum; i++) {
        u64 dbAddr = 0;
        u64 dbValue = 0;
        u64 wrLen = (i == useQpNum - 1) ? tileLenTail : tileLen;
        u64 offset = tileLen * i;
        if (wrLen == 0) {
            continue;
        }
        RmaBufSliceLite locSlice(locRmaBufSliceLite.GetAddr() + offset, wrLen, locRmaBufSliceLite.GetLkey(), 0);
        RmtRmaBufSliceLite rmtSlice(
            rmtRmaBufSliceLite.GetAddr() + offset, wrLen, rmtRmaBufSliceLite.GetRkey(), 0, 0, UINT32_MAX);
        CHK_RET(connVec_[i]->Read(locSlice, rmtSlice, cfg, dbAddr, dbValue));
        CHK_RET(BuildRdmaDbSendTask(stream, dbAddr, dbValue));
    }

    // 上报Profiling任务
    ReportDmaTask(
        ReinterpretAs<const void*>(locRmaBufSliceLite.GetAddr()),
        ReinterpretAs<const void*>(rmtRmaBufSliceLite.GetAddr()), locRmaBufSliceLite.GetSize(), stream, taskId,
        TaskParamTypeVal::TASK_RDMA, DmaOp::HCCL_DMA_READ, INVALID_VALUE_NOTIFYID, UINT32_MAX,
        DfxRdmaTypeVal::DFX_RDMA_SEND_PAYLOAD, __func__);

    // 串行 PollCq: 同步回收已完成的 CQE, 不等待
    CHK_RET(SerialPollCq());
    return HCCL_SUCCESS;
}

HcclResult RoceHostLocTransportLiteImpl::Write(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream)
{
    // 获取Profiling任务ID
    auto taskId = stream.GetRtsq()->GetTaskId();

    // 获取本端和远端Buffer切片
    auto locRmaBufSliceLite = GetRmaBufSlicelite(loc);
    RmtRmaBufSliceLite rmtRmaBufSliceLite;
    CHK_RET(GetRmtRmaBufSliceLite(rmt, rmtRmaBufSliceLite));
    SqeConfigLite cfg;
    SetFenceConfig(cfg);

    u64 tileLen = 0;
    u64 tileLenTail = 0;
    u32 useQpNum = CalcUseQpNum(locRmaBufSliceLite.GetSize(), tileLen, tileLenTail);

    // Post Wqe && return dbValue
    for (u32 i = 0; i < useQpNum; i++) {
        u64 dbAddr = 0;
        u64 dbValue = 0;
        u64 wrLen = (i == useQpNum - 1) ? tileLenTail : tileLen;
        u64 offset = tileLen * i;
        if (wrLen == 0) {
            continue;
        }
        RmaBufSliceLite locSlice(locRmaBufSliceLite.GetAddr() + offset, wrLen, locRmaBufSliceLite.GetLkey(), 0);
        RmtRmaBufSliceLite rmtSlice(
            rmtRmaBufSliceLite.GetAddr() + offset, wrLen, rmtRmaBufSliceLite.GetRkey(), 0, 0, UINT32_MAX);
        CHK_RET(connVec_[i]->Write(locSlice, rmtSlice, cfg, dbAddr, dbValue));
        CHK_RET(BuildRdmaDbSendTask(stream, dbAddr, dbValue));
    }

    // 上报Profiling任务
    ReportDmaTask(
        ReinterpretAs<const void*>(locRmaBufSliceLite.GetAddr()),
        ReinterpretAs<const void*>(rmtRmaBufSliceLite.GetAddr()), locRmaBufSliceLite.GetSize(), stream, taskId,
        TaskParamTypeVal::TASK_RDMA, DmaOp::HCCL_DMA_WRITE, INVALID_VALUE_NOTIFYID, UINT32_MAX,
        DfxRdmaTypeVal::DFX_RDMA_SEND_PAYLOAD, __func__);

    // 串行 PollCq: 同步回收已完成的 CQE, 不等待
    CHK_RET(SerialPollCq());
    return HCCL_SUCCESS;
}

HcclResult RoceHostLocTransportLiteImpl::WriteReduce(
    const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const StreamLite& stream)
{
    auto taskId = stream.GetRtsq()->GetTaskId();

    SqeConfigLite cfg;
    SetFenceConfig(cfg);
    auto locRmaBufSliceLite = GetRmaBufSlicelite(loc);
    RmtRmaBufSliceLite rmtRmaBufSliceLite;
    CHK_RET(GetRmtRmaBufSliceLite(rmt, rmtRmaBufSliceLite));

    const u32 elementSize = DataTypeSizeGet(reduceIn.dataType);
    CHK_PRT_RET(
        elementSize == 0 || locRmaBufSliceLite.GetSize() % elementSize != 0,
        HCCL_ERROR(
            "[RoceHostLocTransportLiteImpl::%s] data size[%llu] is not aligned to elementSize[%u].", __func__,
            locRmaBufSliceLite.GetSize(), elementSize),
        HCCL_E_PARA);

    u64 tileLen = 0;
    u64 tileLenTail = 0;
    u32 useQpNum = CalcUseQpNum(locRmaBufSliceLite.GetSize(), tileLen, tileLenTail, elementSize);

    // Post Wqe && return dbValue
    for (u32 i = 0; i < useQpNum; i++) {
        u64 dbAddr = 0;
        u64 dbValue = 0;
        u64 wrLen = (i == useQpNum - 1) ? tileLenTail : tileLen;
        u64 offset = tileLen * i;
        if (wrLen == 0) {
            continue;
        }
        RmaBufSliceLite locSlice(locRmaBufSliceLite.GetAddr() + offset, wrLen, locRmaBufSliceLite.GetLkey(), 0);
        RmtRmaBufSliceLite rmtSlice(
            rmtRmaBufSliceLite.GetAddr() + offset, wrLen, rmtRmaBufSliceLite.GetRkey(), 0, 0, UINT32_MAX);
        CHK_RET(
            connVec_[i]->WriteReduce(locSlice, rmtSlice, cfg, reduceIn.dataType, reduceIn.reduceOp, dbAddr, dbValue));
        CHK_RET(BuildRdmaDbSendTask(stream, dbAddr, dbValue));
    }

    // 上报Profiling任务
    ReportReduceTask(
        ReinterpretAs<const void*>(locRmaBufSliceLite.GetAddr()),
        ReinterpretAs<const void*>(rmtRmaBufSliceLite.GetAddr()), locRmaBufSliceLite.GetSize(), reduceIn, stream,
        taskId, TaskParamTypeVal::TASK_REDUCE_INLINE, INVALID_VALUE_NOTIFYID, UINT32_MAX,
        DfxRdmaTypeVal::DFX_RDMA_SEND_PAYLOAD, __func__);

    // 串行 PollCq: 同步回收已完成的 CQE, 不等待
    CHK_RET(SerialPollCq());
    return HCCL_SUCCESS;
}

HcclResult RoceHostLocTransportLiteImpl::WriteWithNotify(
    const RmaBufferLite& loc, const Buffer& rmt, const WithNotifyIn& withNotify, const StreamLite& stream)
{
    auto taskId = stream.GetRtsq()->GetTaskId();

    SqeConfigLite cfg;
    SetFenceConfig(cfg);
    auto locRmaBufSliceLite = GetRmaBufSlicelite(loc);
    RmtRmaBufSliceLite rmtRmaBufSliceLite;
    CHK_RET(GetRmtRmaBufSliceLite(rmt, rmtRmaBufSliceLite));

    // Post Wqe && return dbValue
    u64 tileLen = 0;
    u64 tileLenTail = 0;
    u32 useQpNum = CalcUseQpNum(locRmaBufSliceLite.GetSize(), tileLen, tileLenTail);
    for (u32 i = 0; i < connNum_; i++) {
        u64 dbAddr = 0;
        u64 dbValue = 0;
        u32 notifyIdx = MapNotifyIdx(withNotify.index_, i);
        auto locNotifySliceLite = GetNotifySlicelite(withNotify.index_);
        RmtRmaBufSliceLite rmtNotifySliceLite;
        CHK_RET(GetRmtNotifySliceLite(notifyIdx, rmtNotifySliceLite));

        if (i < useQpNum) {
            u64 wrLen = (i == useQpNum - 1) ? tileLenTail : tileLen;
            u64 offset = tileLen * i;
            if (wrLen == 0) {
                CHK_RET(connVec_[i]->Write(locNotifySliceLite, rmtNotifySliceLite, cfg, dbAddr, dbValue));
            } else {
                RmaBufSliceLite locSlice(locRmaBufSliceLite.GetAddr() + offset, wrLen, locRmaBufSliceLite.GetLkey(), 0);
                RmtRmaBufSliceLite rmtSlice(
                    rmtRmaBufSliceLite.GetAddr() + offset, wrLen, rmtRmaBufSliceLite.GetRkey(), 0, 0, UINT32_MAX);
                CHK_RET(connVec_[i]->WriteWithNotify(
                    locSlice, rmtSlice, locNotifySliceLite, rmtNotifySliceLite, cfg, dbAddr, dbValue));
            }
        } else {
            CHK_RET(connVec_[i]->Write(locNotifySliceLite, rmtNotifySliceLite, cfg, dbAddr, dbValue));
        }
        CHK_RET(BuildRdmaDbSendTask(stream, dbAddr, dbValue));
    }

    // 上报Profiling任务
    u32 reportNotifyIdx = MapNotifyIdx(withNotify.index_, 0);
    RmtRmaBufSliceLite rmtNotifySliceLiteForReport;
    CHK_RET(GetRmtNotifySliceLite(reportNotifyIdx, rmtNotifySliceLiteForReport));
    ReportDmaTask(
        ReinterpretAs<const void*>(locRmaBufSliceLite.GetAddr()),
        ReinterpretAs<const void*>(rmtRmaBufSliceLite.GetAddr()), locRmaBufSliceLite.GetSize(), stream, taskId,
        TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY, DmaOp::HCCL_DMA_WRITE, rmtNotifySliceLiteForReport.GetNotifyId(), 1,
        DfxRdmaTypeVal::DFX_RDMA_SEND_PAYLOAD, __func__);

    // 串行 PollCq: 同步回收已完成的 CQE, 不等待
    CHK_RET(SerialPollCq());
    return HCCL_SUCCESS;
}

HcclResult RoceHostLocTransportLiteImpl::WriteReduceWithNotify(
    const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const WithNotifyIn& withNotify,
    const StreamLite& stream)
{
    auto taskId = stream.GetRtsq()->GetTaskId();

    auto locRmaBufSliceLite = GetRmaBufSlicelite(loc);
    RmtRmaBufSliceLite rmtRmaBufSliceLite;
    CHK_RET(GetRmtRmaBufSliceLite(rmt, rmtRmaBufSliceLite));
    SqeConfigLite cfg;
    SetFenceConfig(cfg);

    const u32 elementSize = DataTypeSizeGet(reduceIn.dataType);
    CHK_PRT_RET(
        elementSize == 0 || locRmaBufSliceLite.GetSize() % elementSize != 0,
        HCCL_ERROR(
            "[RoceHostLocTransportLiteImpl::%s] data size[%llu] is not aligned to elementSize[%u].", __func__,
            locRmaBufSliceLite.GetSize(), elementSize),
        HCCL_E_PARA);

    // Post Wqe && return dbValue
    u64 tileLen = 0;
    u64 tileLenTail = 0;
    u32 useQpNum = CalcUseQpNum(locRmaBufSliceLite.GetSize(), tileLen, tileLenTail, elementSize);

    for (u32 i = 0; i < connNum_; i++) {
        u64 dbAddr = 0;
        u64 dbValue = 0;
        u32 notifyIdx = MapNotifyIdx(withNotify.index_, i);
        auto locNotifySliceLite = GetNotifySlicelite(withNotify.index_);
        RmtRmaBufSliceLite rmtNotifySliceLite;
        CHK_RET(GetRmtNotifySliceLite(notifyIdx, rmtNotifySliceLite));

        if (i < useQpNum) {
            u64 wrLen = (i == useQpNum - 1) ? tileLenTail : tileLen;
            u64 offset = tileLen * i;
            if (wrLen == 0) {
                CHK_RET(connVec_[i]->Write(locNotifySliceLite, rmtNotifySliceLite, cfg, dbAddr, dbValue));
            } else {
                RmaBufSliceLite locSlice(locRmaBufSliceLite.GetAddr() + offset, wrLen, locRmaBufSliceLite.GetLkey(), 0);
                RmtRmaBufSliceLite rmtSlice(
                    rmtRmaBufSliceLite.GetAddr() + offset, wrLen, rmtRmaBufSliceLite.GetRkey(), 0, 0, UINT32_MAX);
                CHK_RET(connVec_[i]->WriteReduceWithNotify(
                    locSlice, rmtSlice, locNotifySliceLite, rmtNotifySliceLite, cfg, reduceIn.dataType,
                    reduceIn.reduceOp, dbAddr, dbValue));
            }
        } else {
            CHK_RET(connVec_[i]->Write(locNotifySliceLite, rmtNotifySliceLite, cfg, dbAddr, dbValue));
        }
        CHK_RET(BuildRdmaDbSendTask(stream, dbAddr, dbValue));
    }

    // 上报Profiling任务
    u32 reportNotifyIdx = MapNotifyIdx(withNotify.index_, 0);
    RmtRmaBufSliceLite rmtNotifySliceLiteForReport;
    CHK_RET(GetRmtNotifySliceLite(reportNotifyIdx, rmtNotifySliceLiteForReport));
    ReportReduceTask(
        ReinterpretAs<const void*>(locRmaBufSliceLite.GetAddr()),
        ReinterpretAs<const void*>(rmtRmaBufSliceLite.GetAddr()), locRmaBufSliceLite.GetSize(), reduceIn, stream,
        taskId, TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY, rmtNotifySliceLiteForReport.GetNotifyId(), 1,
        DfxRdmaTypeVal::DFX_RDMA_SEND_PAYLOAD, __func__);

    // 串行 PollCq: 同步回收已完成的 CQE, 不等待
    CHK_RET(SerialPollCq());
    return HCCL_SUCCESS;
}

HcclResult RoceHostLocTransportLiteImpl::Fence()
{
    fence_ = true;
    HCCL_INFO("[%s] SUCCESS. fence[%d]", __func__, fence_);
    return HCCL_SUCCESS;
}

HcclResult RoceHostLocTransportLiteImpl::Post(u32 index, const StreamLite& stream)
{
    auto taskId = stream.GetRtsq()->GetTaskId();

    SqeConfigLite cfg;
    SetFenceConfig(cfg);

    // Post Wqe && return dbValue
    for (u32 i = 0; i < connNum_; i++) {
        u64 dbAddr = 0;
        u64 dbValue = 0;
        u32 notifyIdx = MapNotifyIdx(index, i);
        auto locNotifySliceLite = GetNotifySlicelite(index);
        RmtRmaBufSliceLite rmtNotifySliceLite;
        CHK_RET(GetRmtNotifySliceLite(notifyIdx, rmtNotifySliceLite));

        CHK_RET(connVec_[i]->Write(locNotifySliceLite, rmtNotifySliceLite, cfg, dbAddr, dbValue));

        CHK_RET(BuildRdmaDbSendTask(stream, dbAddr, dbValue));

        if (i == 0) {
            // 上报Profiling任务
            ReportDmaTask(
                ReinterpretAs<const void*>(locNotifySliceLite.GetAddr()),
                ReinterpretAs<const void*>(rmtNotifySliceLite.GetAddr()), locNotifySliceLite.GetSize(), stream, taskId,
                TaskParamTypeVal::TASK_RDMA, DmaOp::HCCL_DMA_WRITE, rmtNotifySliceLite.GetNotifyId(), 1,
                DfxRdmaTypeVal::DFX_RDMA_SEND_NOTIFY, __func__);
        }
    }

    // 串行 PollCq: 同步回收已完成的 CQE, 不等待
    CHK_RET(SerialPollCq());
    return HCCL_SUCCESS;
}

HcclResult RoceHostLocTransportLiteImpl::PollCq(int32_t numEntries, int32_t timeOut, std::vector<int32_t>& errList)
{
    HcclResult ret = HCCL_SUCCESS;
    bool allTimeout = true;
    const auto startTime = std::chrono::steady_clock::now();

    // Poll numEntries个Cqe, 返回异常的status, 同时返回cq的db
    for (u32 i = 0; i < connVec_.size(); i++) {
        u64 dbAddr = 0;
        u64 dbValue = 0;
        int32_t remainingTimeout = timeOut;
        if (timeOut > 0) {
            const auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            remainingTimeout = (elapsed >= timeOut) ? 0 : (timeOut - static_cast<int32_t>(elapsed));
        }
        HcclResult connRet = connVec_[i]->PollCq(numEntries, remainingTimeout, errList, dbAddr, dbValue);
        if (connRet == HCCL_E_TIMEOUT) {
            // 超时为正常现象, 继续轮询其余连接
            continue;
        }
        allTimeout = false;
        if (connRet != HCCL_SUCCESS && ret == HCCL_SUCCESS) {
            // 保留首个真实错误
            ret = connRet;
        }
    }

    // 所有连接均超时, 保持单连接实现透传 HCCL_E_TIMEOUT 的语义
    if (ret == HCCL_SUCCESS && allTimeout) {
        return HCCL_E_TIMEOUT;
    }
    return ret;
}

void RoceHostLocTransportLiteImpl::WaitWithTimeout(u32 index, const StreamLite& stream, u32 timeout)
{
    auto taskId = stream.GetRtsq()->GetTaskId();

    u32 perQpTimeout = timeout;
    if (timeout > 0) {
        perQpTimeout = static_cast<u32>((static_cast<u64>(timeout) + connNum_ - 1) / connNum_);
    }
    for (u32 i = 0; i < connNum_; i++) {
        u32 notifyIdx = MapNotifyIdx(index, i);
        if (notifyIdx >= localNotifies_.size()) {
            MACRO_THROW(
                InternalException,
                StringFormat(
                    "WaitWithTimeout: notifyIdx[%u] out of range[%zu], index[%u], qpIdx[%u], connNum[%u].", notifyIdx,
                    localNotifies_.size(), index, i, connNum_));
        }
        auto notifyId = localNotifies_[notifyIdx]->GetId();
        BuildNotifyWaitTask(notifyId, stream, perQpTimeout);

        // 上报Profiling任务
        ReportNotifyWaitTask(notifyId, stream, taskId);
    }
}

// 下发Rtsq sqe, 敲DB
HcclResult
RoceHostLocTransportLiteImpl::BuildRdmaDbSendTask(const StreamLite& stream, u64 remoteAddr, u64 dbValue) const
{
    stream.GetRtsq()->SetCurrentTransportContext(
        ReinterpretAs<u64>(this), DfxLinkTypeVal::LINK_ROCE,
        static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE));
    TRY_CATCH_RETURN(stream.GetRtsq()->RdmaDbSend(remoteAddr, dbValue););
    return HCCL_SUCCESS;
}

// 下发Rtsq sqe, NotifyWait
void RoceHostLocTransportLiteImpl::BuildNotifyWaitTask(u32 notifyId, const StreamLite& stream, u32 timeout) const
{
    stream.GetRtsq()->SetCurrentTransportContext(
        ReinterpretAs<u64>(this), DfxLinkTypeVal::LINK_ROCE,
        static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE));
    stream.GetRtsq()->NotifyWait(notifyId, timeout);
}

void RoceHostLocTransportLiteImpl::SetFenceConfig(SqeConfigLite& cfg)
{
    cfg.cqeEn = true;
    cfg.fence = fence_ ? 1 : 0;
    fence_ = false;
}

void RoceHostLocTransportLiteImpl::ReportDmaTask(
    const void* src, const void* dst, u64 size, const StreamLite& stream, u32 taskId, TaskParamTypeVal taskType,
    [[maybe_unused]] DmaOp dmaOp, u64 notifyId, u32 notifyValue, u8 rdmaType, const char* funcName) const
{
    // 未开启任务上报时直接返回
    if (!IsReportTask()) {
        return;
    }

    HCCL_INFO(
        "[RoceHostLocTransportLiteImpl::%s][ProfilingTaskParam] sqId[%u], taskId[%u], taskType[%s], "
        "src[%p], dst[%p], size[%zu], notifyId[%llu], notifyValue[%u], rdmaType[%u]",
        funcName, stream.GetSqId(), taskId, TaskParamTypeValToStr(static_cast<u8>(taskType)), src, dst, size, notifyId,
        notifyValue, rdmaType);

    DfxTaskInfo* slot = stream.NextTaskSlot();
    slot->taskType = static_cast<u8>(taskType);
    slot->sqId = stream.GetSqId();
    slot->taskId = taskId;
    const void* opInfo = stream.GetLatestDfxOpInfo();
    slot->dfxOpInfo = (opInfo != nullptr) ? ReinterpretAs<u64>(opInfo) : DFX_INVALID_U64;
    slot->linkType = DfxLinkTypeVal::LINK_ROCE;
    slot->transportType = static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE);
    slot->rdmaType = rdmaType;
    slot->channelHandle = ReinterpretAs<u64>(this);
    slot->taskPara.Dma.srcAddr = ReinterpretAs<u64>(src);
    slot->taskPara.Dma.dstAddr = ReinterpretAs<u64>(dst);
    slot->taskPara.Dma.size = size;
    slot->taskPara.Dma.notifyId = static_cast<u32>(notifyId);
    PLF_CONFIG_INFO(Hccl::PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
}

void RoceHostLocTransportLiteImpl::ReportReduceTask(
    const void* src, const void* dst, u64 size, const ReduceIn& reduceIn, const StreamLite& stream, u32 taskId,
    TaskParamTypeVal taskType, u64 notifyId, u32 notifyValue, u8 rdmaType, const char* funcName) const
{
    // 未开启任务上报时直接返回
    if (!IsReportTask()) {
        return;
    }

    HCCL_INFO(
        "[RoceHostLocTransportLiteImpl::%s][ProfilingTaskParam] sqId[%u], taskId[%u], taskType[%s], "
        "src[%p], dst[%p], size[%zu], notifyId[%llu], notifyValue[%u], dataType[%d], reduceOp[%d], rdmaType[%u]",
        funcName, stream.GetSqId(), taskId, TaskParamTypeValToStr(static_cast<u8>(taskType)), src, dst, size, notifyId,
        notifyValue, static_cast<int>(reduceIn.dataType), static_cast<int>(reduceIn.reduceOp), rdmaType);

    DfxTaskInfo* slot = stream.NextTaskSlot();
    slot->taskType = static_cast<u8>(taskType);
    slot->sqId = stream.GetSqId();
    slot->taskId = taskId;
    const void* opInfo = stream.GetLatestDfxOpInfo();
    slot->dfxOpInfo = (opInfo != nullptr) ? ReinterpretAs<u64>(opInfo) : DFX_INVALID_U64;
    slot->linkType = DfxLinkTypeVal::LINK_ROCE;
    slot->transportType = static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE);
    slot->rdmaType = rdmaType;
    slot->channelHandle = ReinterpretAs<u64>(this);
    slot->taskPara.Reduce.srcAddr = ReinterpretAs<u64>(src);
    slot->taskPara.Reduce.dstAddr = ReinterpretAs<u64>(dst);
    slot->taskPara.Reduce.size = size;
    slot->taskPara.Reduce.notifyId = static_cast<u32>(notifyId);
    slot->taskPara.Reduce.reduceOp = static_cast<u8>(ConvertReduceOpToHcclReduceOp(reduceIn.reduceOp));
    PLF_CONFIG_INFO(Hccl::PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
}

void RoceHostLocTransportLiteImpl::ReportNotifyWaitTask(u64 notifyId, const StreamLite& stream, u32 taskId) const
{
    // 未开启任务上报时直接返回
    if (!IsReportTask()) {
        return;
    }

    HCCL_INFO(
        "[RoceHostLocTransportLiteImpl::%s][ProfilingTaskParam] sqId[%u], taskId[%u], notifyId[%llu]", __func__,
        stream.GetSqId(), taskId, notifyId);

    DfxTaskInfo* slot = stream.NextTaskSlot();
    slot->taskType = static_cast<u8>(TaskParamTypeVal::TASK_NOTIFY_WAIT);
    slot->sqId = stream.GetSqId();
    slot->taskId = taskId;
    const void* opInfo = stream.GetLatestDfxOpInfo();
    slot->dfxOpInfo = (opInfo != nullptr) ? ReinterpretAs<u64>(opInfo) : DFX_INVALID_U64;
    slot->linkType = DfxLinkTypeVal::LINK_ROCE;
    slot->transportType = static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE);
    slot->channelHandle = ReinterpretAs<u64>(this);
    slot->taskPara.Notify.notifyId = static_cast<u32>(notifyId);
    PLF_CONFIG_INFO(Hccl::PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
}

bool RoceHostLocTransportLiteImpl::IsReportTask() const
{
    // TaskException或Profiling开启时，允许上报
    return taskExceptionEnable_ || DfxProfilingHandlerLite::GetInstance().GetProfL1State();
}

} // namespace Hccl
