/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_transport_lite_impl.h"
#include "binary_stream.h"
#include "ub_conn_lite.h"
#include "ub_conn_lite_mgr.h"
#include "exception_util.h"
#include "internal_exception.h"
#include "sal.h"

namespace Hccl {
UbTransportLiteImpl::UbTransportLiteImpl(
    std::vector<char>& uniqueId, std::function<void(u32 streamId, u32 taskId, const TaskParam& taskParam)> callback)
{
    callback_ = nullptr;
}
UbTransportLiteImpl::UbTransportLiteImpl(std::vector<char>& uniqueId)
{
    static RmaConnLite dummyConn;
    cachedConn_ = &dummyConn;
}

UbTransportLiteImpl::~UbTransportLiteImpl() {}

std::string UbTransportLiteImpl::Describe() const { return ""; }

void UbTransportLiteImpl::ParseLocNotifyVec(std::vector<char>& data) {}

void UbTransportLiteImpl::ParseRmtBufferVec(std::vector<char>& data, RmaUbBufType rmtType) {}

void UbTransportLiteImpl::ParseLocBufferMap(std::vector<char>& data) {}

void UbTransportLiteImpl::ParseConnVec(std::vector<char>& data) {}

HcclResult UbTransportLiteImpl::BuildUbDbSendTask(const StreamLite& stream, RmaConnLite* conn, u16 pi)
{
    return HCCL_SUCCESS;
}

void UbTransportLiteImpl::BuildNotifyWaitTask(const StreamLite& stream, u32 notifyId) {}

Buffer UbTransportLiteImpl::GetRmtBuffer(u32 index)
{
    return Buffer(rmtBufferVec[index].addr, rmtBufferVec[index].size);
}

HcclResult UbTransportLiteImpl::GetRmtNotifySliceLite(u32 index, RmtRmaBufSliceLite& rmtSlice)
{
    RmtUbBufLite& lite = rmtNotifyVec[index];
    // ub conn lite 不关心rkey , rkey 设定为0
    rmtSlice = RmtRmaBufSliceLite(lite.addr, lite.size, 0, lite.tokenId, lite.tokenValue, UINT32_MAX);
    return HCCL_SUCCESS;
}

HcclResult UbTransportLiteImpl::GetRmtRmaBufSliceLite(const Buffer& rmtBuf, RmtRmaBufSliceLite& rmtSlice) const
{
    rmtSlice = RmtRmaBufSliceLite(rmtBuf.GetAddr(), rmtBuf.GetSize(), 0, 0, 0, UINT32_MAX);
    return HCCL_SUCCESS;
}

RmtRmaBufSliceLite UbTransportLiteImpl::GetRmtRmaBufSliceLite(const RmaBufferLite& lite) const
{
    return RmtRmaBufSliceLite(lite.GetAddr(), lite.GetSize(), 0, lite.GetTokenId(), lite.GetTokenValue(), UINT32_MAX);
}

HcclResult
UbTransportLiteImpl::BuildLocRmaBufferLite(const uintptr_t addr, const size_t size, RmaBufferLite& rmaBufferLite)
{
    return HCCL_SUCCESS;
}

void UbTransportLiteImpl::ClearConnOut() {}

// 检查connection不能为空
void UbTransportLiteImpl::CheckConnVec(const std::string& desc) {}

RmaBufSliceLite UbTransportLiteImpl::GetRmaBufSlicelite(const RmaBufferLite& lite) const
{
    // ub conn lite 不关心rkey , rkey 设定为0
    return RmaBufSliceLite(lite.GetAddr(), lite.GetSize(), 0, lite.GetTokenId());
}

HcclResult UbTransportLiteImpl::Post(u32 index, const StreamLite& stream) { return HCCL_SUCCESS; }

void UbTransportLiteImpl::Wait(u32 index, const StreamLite& stream) {}

void UbTransportLiteImpl::WaitWithTimeout(u32 index, const StreamLite& stream, u32 timeout) {}

void UbTransportLiteImpl::ProfilingProcess(
    void* src, void* dst, u64 size, const StreamLite& stream, DmaOp dmaOp, u32 taskId)
{}

HcclResult UbTransportLiteImpl::Read(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream)
{
    return HCCL_SUCCESS;
}

HcclResult UbTransportLiteImpl::Write(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream)
{
    return HCCL_SUCCESS;
}

void UbTransportLiteImpl::ReadReduce(
    const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const StreamLite& stream)
{}

void UbTransportLiteImpl::ReduceProfilingProcess(
    void* src, void* dst, u64 size, const ReduceIn& reduceIn, const StreamLite& stream, u32 taskId)
{}

HcclResult UbTransportLiteImpl::WriteReduce(
    const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const StreamLite& stream)
{
    return HCCL_SUCCESS;
}

void UbTransportLiteImpl::BatchTransfer(
    const std::vector<RmaBufferLite>& loc, const std::vector<Buffer>& rmt,
    const std::vector<BaseTransportLiteImpl::TransferOp>& transferOp, const StreamLite& stream)
{}

void UbTransportLiteImpl::BatchTransferAll(
    const std::vector<RmaBufferLite>& loc, const std::vector<Buffer>& rmt, const std::vector<TransferOp>& transferOp,
    const std::vector<uint32_t>& notifyIdxs, const StreamLite& stream)
{}

HcclResult UbTransportLiteImpl::WriteWithNotify(
    const RmaBufferLite& loc, const Buffer& rmt, const WithNotifyIn& withNotify, const StreamLite& stream)
{
    return HCCL_SUCCESS;
}

HcclResult UbTransportLiteImpl::WriteReduceWithNotify(
    const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const WithNotifyIn& withNotify,
    const StreamLite& stream)
{
    return HCCL_SUCCESS;
}

void UbTransportLiteImpl::BatchOneSidedRead(
    const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const StreamLite& stream)
{}

void UbTransportLiteImpl::BatchOneSidedWrite(
    const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const StreamLite& stream)
{}

Eid UbTransportLiteImpl::GetLocEid() const
{
    Eid eid{};
    return eid;
}

Eid UbTransportLiteImpl::GetRmtEid() const
{
    Eid eid{};
    return eid;
}
} // namespace Hccl
