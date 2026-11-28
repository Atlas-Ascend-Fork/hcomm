/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_trans.h"

#include <limits>

#include "ccu_rep_v1.h"
#include "ccu_api_exception.h"
#include "ccu_assist_v1.h"
#include "ccu_log.h"
#include "config_plf_log_v2.h"

namespace hcomm {

using Hccl::PLF_DATA_OP;

// 本地数据拷贝 相关实现
CcuResult ccu_trans::LocalCopyMemToBuffer(
    CcuKernel& kernel, const LocalTransOperands& op, CcuBufferHandle dstHandle, CcuLocalAddrHandle srcHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[LocalCopyMemToBuffer] dstHandle=%llu, srcHandle=%llu, lenHandle=%llu,"
        " eventHandle=%llu, mask=%u",
        dstHandle, srcHandle, op.lenHandle, op.eventHandle, op.mask);
    CcuRep::CcuBuf* dst{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveBuffer(dstHandle, &dst));
    CcuRep::LocalAddr* src{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveLocalAddr(srcHandle, &src));
    CcuRep::Variable* len{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(op.lenHandle, &len));
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveEvent(op.eventHandle, &event));
    auto rep = std::make_shared<CcuRep::CcuRepBufLocRead>(kernel.GetInsGenerator(), *src, *dst, *len, *event, op.mask);
    kernel.Append(rep);
    kernel.SetDependencyInfo(event->Id(), op.mask, rep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_trans::LocalCopyBufferToMem(
    CcuKernel& kernel, const LocalTransOperands& op, CcuLocalAddrHandle dstHandle, CcuBufferHandle srcHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[LocalCopyBufferToMem] dstHandle=%llu, srcHandle=%llu, lenHandle=%llu,"
        " eventHandle=%llu, mask=%u",
        dstHandle, srcHandle, op.lenHandle, op.eventHandle, op.mask);
    CcuRep::LocalAddr* dst{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveLocalAddr(dstHandle, &dst));
    CcuRep::CcuBuf* src{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveBuffer(srcHandle, &src));
    CcuRep::Variable* len{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(op.lenHandle, &len));
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveEvent(op.eventHandle, &event));
    auto rep = std::make_shared<CcuRep::CcuRepBufLocWrite>(kernel.GetInsGenerator(), *src, *dst, *len, *event, op.mask);
    kernel.Append(rep);
    kernel.SetDependencyInfo(event->Id(), op.mask, rep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_trans::LocalCopyMemToMem(
    CcuKernel& kernel, const LocalTransOperands& op, CcuLocalAddrHandle dstHandle, CcuLocalAddrHandle srcHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[LocalCopyMemToMem] dstHandle=%llu, srcHandle=%llu, lenHandle=%llu,"
        " eventHandle=%llu, mask=%u",
        dstHandle, srcHandle, op.lenHandle, op.eventHandle, op.mask);
    CcuRep::LocalAddr* dst{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveLocalAddr(dstHandle, &dst));
    CcuRep::LocalAddr* src{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveLocalAddr(srcHandle, &src));
    CcuRep::Variable* len{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(op.lenHandle, &len));
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveEvent(op.eventHandle, &event));
    auto rep = std::make_shared<CcuRep::CcuRepLocCpy>(kernel.GetInsGenerator(), *dst, *src, *len, *event, op.mask);
    kernel.Append(rep);
    kernel.SetDependencyInfo(event->Id(), op.mask, rep);
    return CcuResult::CCU_SUCCESS;
}

static bool isLowPrecisionIn(Hccl::DataType dataType)
{
    return dataType == Hccl::DataType::INT8 || dataType == Hccl::DataType::HIF8 || dataType == Hccl::DataType::FP8E4M3
           || dataType == Hccl::DataType::FP8E5M2;
}

static bool isLowPrecisionOut(Hccl::DataType dataType)
{
    return dataType == Hccl::DataType::FP16 || dataType == Hccl::DataType::BFP16 || dataType == Hccl::DataType::FP32;
}

static Hccl::DataType HcommDataTypeToHcclDataType(const HcclDataType dataType)
{
    static const std::map<HcclDataType, Hccl::DataType> hcclDataTypeMap = {
        {HCCL_DATA_TYPE_INT8, Hccl::DataType::INT8},       {HCCL_DATA_TYPE_INT16, Hccl::DataType::INT16},
        {HCCL_DATA_TYPE_INT32, Hccl::DataType::INT32},     {HCCL_DATA_TYPE_FP16, Hccl::DataType::FP16},
        {HCCL_DATA_TYPE_FP32, Hccl::DataType::FP32},       {HCCL_DATA_TYPE_INT64, Hccl::DataType::INT64},
        {HCCL_DATA_TYPE_UINT64, Hccl::DataType::UINT64},   {HCCL_DATA_TYPE_UINT8, Hccl::DataType::UINT8},
        {HCCL_DATA_TYPE_UINT16, Hccl::DataType::UINT16},   {HCCL_DATA_TYPE_UINT32, Hccl::DataType::UINT32},
        {HCCL_DATA_TYPE_FP64, Hccl::DataType::FP64},       {HCCL_DATA_TYPE_BFP16, Hccl::DataType::BFP16},
        {HCCL_DATA_TYPE_INT128, Hccl::DataType::INT128},
#if !defined(OPEN_BUILD_PROJECT) || defined(ORION_MODE)
        {HCCL_DATA_TYPE_HIF8, Hccl::DataType::HIF8},       {HCCL_DATA_TYPE_FP8E4M3, Hccl::DataType::FP8E4M3},
        {HCCL_DATA_TYPE_FP8E5M2, Hccl::DataType::FP8E5M2}, {HCCL_DATA_TYPE_FP8E8M0, Hccl::DataType::FP8E8M0},
#endif
    };

    auto it = hcclDataTypeMap.find(dataType);
    if (it == hcclDataTypeMap.end()) {
        return Hccl::DataType::INVALID;
    }

    return it->second;
}

static Hccl::ReduceOp HcommReduceOpToHcclReduceOp(const HcclReduceOp reduceOp)
{
    static const std::map<HcclReduceOp, Hccl::ReduceOp> hcclReduceOpMap = {
        {HCCL_REDUCE_SUM, Hccl::ReduceOp::SUM},
        {HCCL_REDUCE_PROD, Hccl::ReduceOp::PROD},
        {HCCL_REDUCE_MAX, Hccl::ReduceOp::MAX},
        {HCCL_REDUCE_MIN, Hccl::ReduceOp::MIN},
    };

    auto it = hcclReduceOpMap.find(reduceOp);
    if (it == hcclReduceOpMap.end()) {
        return Hccl::ReduceOp::INVALID;
    }

    return it->second;
}

// 本地reduce 相关实现
CcuResult ccu_trans::LocalMemReduce(
    CcuKernel& kernel, const LocalTransOperands& op, CcuLocalAddrHandle dstHandle, CcuLocalAddrHandle srcHandle,
    HcclDataType dataType, HcclReduceOp opType)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[LocalMemReduce] dstHandle=%llu, srcHandle=%llu, lenHandle=%llu, dataType=%d,"
        " op=%d, eventHandle=%llu, mask=%u",
        dstHandle, srcHandle, op.lenHandle, dataType, opType, op.eventHandle, op.mask);
    CcuRep::LocalAddr* dst{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveLocalAddr(dstHandle, &dst));
    CcuRep::LocalAddr* src{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveLocalAddr(srcHandle, &src));
    CcuRep::Variable* len{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(op.lenHandle, &len));
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveEvent(op.eventHandle, &event));
    auto opType_ = HcommReduceOpToHcclReduceOp(opType);
    auto dataType_ = HcommDataTypeToHcclDataType(dataType);

    uint16_t ubDataType;
    CCU_CHK_RET(CcuRep::GetUBDataType(dataType_, ubDataType));
    uint16_t ubReduceType;
    CCU_CHK_RET(CcuRep::GetUBReduceType(opType_, ubReduceType));
    auto rep = std::make_shared<CcuRep::CcuRepLocCpy>(
        kernel.GetInsGenerator(), *dst, *src, *len, ubDataType, ubReduceType, *event, op.mask);
    kernel.Append(rep);
    kernel.SetDependencyInfo(event->Id(), op.mask, rep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_trans::LocalBufferReduce(CcuKernel& kernel, const LocalReduceOperands& op)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[LocalBufferReduce] count=%u, dataType=%d, outDataType=%d,"
        " op=%d, lenHandle=%llu, eventHandle=%llu, mask=%u",
        op.count, op.dataType, op.outputDataType, op.opType, op.lenHandle, op.eventHandle, op.mask);
    if (op.count > std::numeric_limits<uint16_t>::max()) {
        return HCCL_TO_CCU_RET(HcclResult::HCCL_E_PARA);
    }
    CcuRep::Variable* len{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(op.lenHandle, &len));
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveEvent(op.eventHandle, &event));
    std::vector<CcuRep::CcuBuf> bufs(op.count);
    for (uint32_t i = 0; i < op.count; i++) {
        CcuRep::CcuBuf* buf{nullptr};
        CCU_CHK_RET(kernel.Repo().ResolveBuffer(op.bufHandles[i], &buf));
        bufs[i] = *buf;
    }
    auto opType_ = HcommReduceOpToHcclReduceOp(op.opType);
    auto dataType_ = HcommDataTypeToHcclDataType(op.dataType);
    auto outputDataType_ = HcommDataTypeToHcclDataType(op.outputDataType);
    if (dataType_ == Hccl::DataType::INVALID || opType_ == Hccl::ReduceOp::INVALID
        || outputDataType_ == Hccl::DataType::INVALID) {
        return HCCL_TO_CCU_RET(HcclResult::HCCL_E_PARA);
    }

    if ((opType_ == Hccl::ReduceOp::SUM && isLowPrecisionIn(dataType_) && !isLowPrecisionOut(outputDataType_))
        || (opType_ == Hccl::ReduceOp::SUM && !isLowPrecisionIn(dataType_) && dataType_ != outputDataType_)
        || (opType_ != Hccl::ReduceOp::SUM && dataType_ != outputDataType_)) {
        return HCCL_TO_CCU_RET(HcclResult::HCCL_E_NOT_SUPPORT);
    }

    // 微码支持表未命中时转成错误码返回。
    uint16_t ccuDataType = 0;
    uint16_t ccuOutputDataType = 0;
    uint16_t ccuReduceType = 0;
    try {
        ccuDataType = CcuRep::GetCcuDataType(dataType_, opType_);
        ccuOutputDataType = CcuRep::GetCcuDataType(outputDataType_, opType_);
        ccuReduceType = CcuRep::GetCcuReduceType(opType_);
    } catch (const Hccl::CcuApiException&) {
        return HCCL_TO_CCU_RET(HcclResult::HCCL_E_NOT_SUPPORT);
    }

    auto rep = std::make_shared<CcuRep::CcuRepBufReduce>(
        kernel.GetInsGenerator(), bufs, op.count, ccuDataType, ccuOutputDataType, ccuReduceType, *event, *len, op.mask);
    kernel.Append(rep);
    kernel.SetDependencyInfo(event->Id(), op.mask, rep);
    return CcuResult::CCU_SUCCESS;
}

static CcuResult ResolveBufRemoteLenEvent(
    CcuKernel& kernel, CcuBufferHandle bufHandle, CcuRemoteAddrHandle remoteHandle, CcuVariableHandle lenHandle,
    CcuEventHandle eventHandle, CcuRep::CcuBuf** buf, CcuRep::RemoteAddr** remote, CcuRep::Variable** len,
    CcuRep::CompletedEvent** event)
{
    CCU_CHK_RET(kernel.Repo().ResolveBuffer(bufHandle, buf));
    CCU_CHK_RET(kernel.Repo().ResolveRemoteAddr(remoteHandle, remote));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(lenHandle, len));
    CCU_CHK_RET(kernel.Repo().ResolveEvent(eventHandle, event));
    return CcuResult::CCU_SUCCESS;
}

static CcuResult ResolveLocalRemoteLenEvent(
    CcuKernel& kernel, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle, CcuVariableHandle lenHandle,
    CcuEventHandle eventHandle, CcuRep::LocalAddr** local, CcuRep::RemoteAddr** remote, CcuRep::Variable** len,
    CcuRep::CompletedEvent** event)
{
    CCU_CHK_RET(kernel.Repo().ResolveLocalAddr(localHandle, local));
    CCU_CHK_RET(kernel.Repo().ResolveRemoteAddr(remoteHandle, remote));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(lenHandle, len));
    CCU_CHK_RET(kernel.Repo().ResolveEvent(eventHandle, event));
    return CcuResult::CCU_SUCCESS;
}

static CcuResult ResolveRemoteLocalLenEvent(
    CcuKernel& kernel, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle, CcuVariableHandle lenHandle,
    CcuEventHandle eventHandle, CcuRep::RemoteAddr** remote, CcuRep::LocalAddr** local, CcuRep::Variable** len,
    CcuRep::CompletedEvent** event)
{
    CCU_CHK_RET(kernel.Repo().ResolveRemoteAddr(remoteHandle, remote));
    CCU_CHK_RET(kernel.Repo().ResolveLocalAddr(localHandle, local));
    CCU_CHK_RET(kernel.Repo().ResolveVariable(lenHandle, len));
    CCU_CHK_RET(kernel.Repo().ResolveEvent(eventHandle, event));
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_trans::ReadMemToMem(
    CcuKernel& kernel, const RemoteOperands& op, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[ReadMemToMem] channel=%llu, localHandle=%llu, remoteHandle=%llu,"
        " lenHandle=%llu, eventHandle=%llu, mask=%u",
        op.channel, localHandle, remoteHandle, op.lenHandle, op.eventHandle, op.mask);
    CcuRep::LocalAddr* local{nullptr};
    CcuRep::RemoteAddr* remote{nullptr};
    CcuRep::Variable* len{nullptr};
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(ResolveLocalRemoteLenEvent(
        kernel, localHandle, remoteHandle, op.lenHandle, op.eventHandle, &local, &remote, &len, &event));
    kernel.AddChannelForRepBuild(op.channel);
    auto rep = std::make_shared<CcuRep::CcuRepRead>(
        kernel.GetInsGenerator(), op.channel, *local, *remote, *len, *event, op.mask);
    kernel.Append(rep);
    kernel.SetDependencyInfo(event->Id(), op.mask, rep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_trans::ReadMemToBuffer(
    CcuKernel& kernel, const RemoteOperands& op, CcuBufferHandle bufHandle, CcuRemoteAddrHandle remoteHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[ReadMemToBuffer] channel=%llu, bufHandle=%llu, remoteHandle=%llu,"
        " lenHandle=%llu, eventHandle=%llu, mask=%u",
        op.channel, bufHandle, remoteHandle, op.lenHandle, op.eventHandle, op.mask);
    CcuRep::CcuBuf* local{nullptr};
    CcuRep::RemoteAddr* remote{nullptr};
    CcuRep::Variable* len{nullptr};
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(ResolveBufRemoteLenEvent(
        kernel, bufHandle, remoteHandle, op.lenHandle, op.eventHandle, &local, &remote, &len, &event));
    kernel.AddChannelForRepBuild(op.channel);
    auto rep = std::make_shared<CcuRep::CcuRepBufRead>(
        kernel.GetInsGenerator(), op.channel, *remote, *local, *len, *event, op.mask);
    kernel.Append(rep);
    kernel.SetDependencyInfo(event->Id(), op.mask, rep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_trans::ReadMemToMemReduce(
    CcuKernel& kernel, const RemoteOperands& op, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle,
    HcclDataType dataType, HcclReduceOp opType)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[ReadMemToMemReduce] channel=%llu, lenHandle=%llu, dataType=%d, op=%d,"
        " eventHandle=%llu, mask=%u",
        op.channel, op.lenHandle, dataType, opType, op.eventHandle, op.mask);
    CcuRep::LocalAddr* local{nullptr};
    CcuRep::RemoteAddr* remote{nullptr};
    CcuRep::Variable* len{nullptr};
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(ResolveLocalRemoteLenEvent(
        kernel, localHandle, remoteHandle, op.lenHandle, op.eventHandle, &local, &remote, &len, &event));
    auto opType_ = HcommReduceOpToHcclReduceOp(opType);
    auto dataType_ = HcommDataTypeToHcclDataType(dataType);

    uint16_t ubDataType;
    CCU_CHK_RET(CcuRep::GetUBDataType(dataType_, ubDataType));
    uint16_t ubReduceType;
    CCU_CHK_RET(CcuRep::GetUBReduceType(opType_, ubReduceType));
    kernel.AddChannelForRepBuild(op.channel);
    auto rep = std::make_shared<CcuRep::CcuRepRead>(
        kernel.GetInsGenerator(), op.channel, *local, *remote, *len, ubDataType, ubReduceType, *event, op.mask);
    kernel.Append(rep);
    kernel.SetDependencyInfo(event->Id(), op.mask, rep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_trans::WriteMemToMem(
    CcuKernel& kernel, const RemoteOperands& op, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[WriteMemToMem] channel=%llu, remoteHandle=%llu, localHandle=%llu,"
        " lenHandle=%llu, eventHandle=%llu, mask=%u",
        op.channel, remoteHandle, localHandle, op.lenHandle, op.eventHandle, op.mask);
    CcuRep::RemoteAddr* remote{nullptr};
    CcuRep::LocalAddr* local{nullptr};
    CcuRep::Variable* len{nullptr};
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(ResolveRemoteLocalLenEvent(
        kernel, remoteHandle, localHandle, op.lenHandle, op.eventHandle, &remote, &local, &len, &event));
    kernel.AddChannelForRepBuild(op.channel);
    auto rep = std::make_shared<CcuRep::CcuRepWrite>(
        kernel.GetInsGenerator(), op.channel, *remote, *local, *len, *event, op.mask);
    kernel.Append(rep);
    kernel.SetDependencyInfo(event->Id(), op.mask, rep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_trans::WriteBufferToMem(
    CcuKernel& kernel, const RemoteOperands& op, CcuRemoteAddrHandle remoteHandle, CcuBufferHandle bufHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[WriteBufferToMem] channel=%llu, remoteHandle=%llu, bufHandle=%llu,"
        " lenHandle=%llu, eventHandle=%llu, mask=%u",
        op.channel, remoteHandle, bufHandle, op.lenHandle, op.eventHandle, op.mask);
    CcuRep::CcuBuf* local{nullptr};
    CcuRep::RemoteAddr* remote{nullptr};
    CcuRep::Variable* len{nullptr};
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(ResolveBufRemoteLenEvent(
        kernel, bufHandle, remoteHandle, op.lenHandle, op.eventHandle, &local, &remote, &len, &event));
    kernel.AddChannelForRepBuild(op.channel);
    auto rep = std::make_shared<CcuRep::CcuRepBufWrite>(
        kernel.GetInsGenerator(), op.channel, *local, *remote, *len, *event, op.mask);
    kernel.Append(rep);
    kernel.SetDependencyInfo(event->Id(), op.mask, rep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_trans::WriteMemToMemReduce(
    CcuKernel& kernel, const RemoteOperands& op, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle,
    HcclDataType dataType, HcclReduceOp opType)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[WriteMemToMemReduce] channel=%llu, lenHandle=%llu, dataType=%d, op=%d,"
        " eventHandle=%llu, mask=%u",
        op.channel, op.lenHandle, dataType, opType, op.eventHandle, op.mask);
    CcuRep::RemoteAddr* remote{nullptr};
    CcuRep::LocalAddr* local{nullptr};
    CcuRep::Variable* len{nullptr};
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(ResolveRemoteLocalLenEvent(
        kernel, remoteHandle, localHandle, op.lenHandle, op.eventHandle, &remote, &local, &len, &event));
    auto opType_ = HcommReduceOpToHcclReduceOp(opType);
    auto dataType_ = HcommDataTypeToHcclDataType(dataType);

    uint16_t ubDataType;
    CCU_CHK_RET(CcuRep::GetUBDataType(dataType_, ubDataType));
    uint16_t ubReduceType;
    CCU_CHK_RET(CcuRep::GetUBReduceType(opType_, ubReduceType));
    kernel.AddChannelForRepBuild(op.channel);
    auto rep = std::make_shared<CcuRep::CcuRepWrite>(
        kernel.GetInsGenerator(), op.channel, *remote, *local, *len, ubDataType, ubReduceType, *event, op.mask);
    kernel.Append(rep);
    kernel.SetDependencyInfo(event->Id(), op.mask, rep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_trans::WriteVarAtomicAdd(
    CcuKernel& kernel, CcuVariableHandle channelIdHandle, CcuRemoteAddrHandle varAddrHandle,
    CcuVariableHandle addValueHandle, CcuEventHandle eventHandle, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[WriteVarAtomicAdd] channelIdHandle=%llu, varAddrHandle=%llu, addValueHandle=%llu, eventHandle=%llu,"
        " mask=%u",
        channelIdHandle, varAddrHandle, addValueHandle, eventHandle, mask);
    CcuRep::Variable* channelId{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(channelIdHandle, &channelId));
    CcuRep::RemoteAddr* varAddr{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveRemoteAddr(varAddrHandle, &varAddr));
    CcuRep::Variable* addValue{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(addValueHandle, &addValue));
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveEvent(eventHandle, &event));
    auto rep = std::make_shared<CcuRep::CcuRepWriteVarAtomic>(
        kernel.GetInsGenerator(), *channelId, *varAddr, *addValue, *event, mask);
    kernel.Append(rep);
    kernel.SetDependencyInfo(event->Id(), mask, rep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult ccu_trans::WriteWithCascCntInc(
    CcuKernel& kernel, CcuVariableHandle channelIdHandle, CcuRemoteAddrHandle remoteHandle,
    CcuLocalAddrHandle localHandle, CcuVariableHandle lenHandle, CcuRemoteAddrHandle inCntAddrHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[WriteWithCascCntInc] channelIdHandle=%llu, remoteHandle=%llu, localHandle=%llu, lenHandle=%llu,"
        " inCntAddrHandle=%llu",
        channelIdHandle, remoteHandle, localHandle, lenHandle, inCntAddrHandle);
    CcuRep::Variable* channelId{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(channelIdHandle, &channelId));
    CcuRep::RemoteAddr* remoteAddr{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveRemoteAddr(remoteHandle, &remoteAddr));
    CcuRep::LocalAddr* local{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveLocalAddr(localHandle, &local));
    CcuRep::Variable* len{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveVariable(lenHandle, &len));
    CcuRep::RemoteAddr* inCntAddr{nullptr};
    CCU_CHK_RET(kernel.Repo().ResolveRemoteAddr(inCntAddrHandle, &inCntAddr));
    CcuRep::Variable origLen = kernel.CreateVariableForRepBuild();
    auto rep = std::make_shared<CcuRep::CcuRepWriteWithCntInc>(
        kernel.GetInsGenerator(), *channelId, *remoteAddr, *local, *len, *inCntAddr, origLen);
    kernel.Append(rep);
    return CcuResult::CCU_SUCCESS;
}

} // namespace hcomm
