/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "step_transfer.h"
#include "channel.h"

using Hccl::BinaryStream;

namespace hcomm {

HcclResult StepTransfer::StepSend(Hccl::Socket* socket, const void* data, u64 size)
{
    if (size == 0) {
        return HCCL_SUCCESS;
    }
    if (sendOff_ > size) { // 调用方中途改小 size 的误用报错
        return HCCL_E_PARA;
    }
    if (sendOff_ == size) { // 上一步已完成本块（防御）
        sendOff_ = 0;
        return HCCL_SUCCESS;
    }
    u64 sent = 0;
    HcclResult ret = socket->ISend(static_cast<const u8*>(data) + sendOff_, size - sendOff_, sent);
    if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN) {
        return ret;
    }
    sendOff_ += sent;
    if (sendOff_ >= size) {
        sendOff_ = 0;
        return HCCL_SUCCESS;
    }
    return HCCL_E_AGAIN;
}

HcclResult StepTransfer::StepRecv(Hccl::Socket* socket, void* data, u64 size)
{
    if (size == 0) {
        return HCCL_SUCCESS;
    }
    if (recvOff_ > size) { // 调用方中途改小 size 的误用报错
        return HCCL_E_PARA;
    }
    if (recvOff_ == size) { // 上一步已完成本块（防御）
        recvOff_ = 0;
        return HCCL_SUCCESS;
    }
    u64 recvd = 0;
    HcclResult ret = socket->IRecv(static_cast<u8*>(data) + recvOff_, size - recvOff_, recvd);
    if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN) {
        return ret;
    }
    recvOff_ += recvd;
    if (recvOff_ >= size) {
        recvOff_ = 0;
        return HCCL_SUCCESS;
    }
    return HCCL_E_AGAIN;
}

HcclResult StepTransfer::StepExchange(
    Hccl::Socket* socket, const std::function<HcclResult(Hccl::BinaryStream&)>& packFn,
    const std::function<HcclResult(Hccl::BinaryStream&)>& unpackFn)
{
    if (phase_ == ExchangePhase::DONE) { // 成功终态重入返回，不重复解包
        return HCCL_SUCCESS;
    }
    if (phase_ == ExchangePhase::FAILED) { // 失败终态重入返回原错误码，不吞错
        return failureCode_;
    }
    if (phase_ == ExchangePhase::SEND_SIZE) {
        if (sendBuf_.empty()) { // 首步：打包并记录本端尺寸（跨步保留，不重复打包）
            Hccl::BinaryStream binaryStream;
            CHK_RET(packFn(binaryStream));
            binaryStream.Dump(sendBuf_);
        }
        u64 localSize = sendBuf_.size();
        HcclResult ret = StepSend(socket, &localSize, sizeof(localSize));
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        phase_ = ExchangePhase::RECV_SIZE;
    }
    if (phase_ == ExchangePhase::RECV_SIZE) {
        HcclResult ret = StepRecv(socket, &remoteSize_, sizeof(remoteSize_));
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        // 对端声明的数据长度不可信，为 0 或超过上限时拒绝分配，防止超大内存申请导致 OOM
        CHK_PRT_RET(
            remoteSize_ == 0 || remoteSize_ > hcomm::Channel::MAX_EXCHANGE_DATA_SIZE,
            HCCL_ERROR(
                "[StepTransfer] remoteSize_[%llu] is invalid, limit[%llu]", remoteSize_,
                hcomm::Channel::MAX_EXCHANGE_DATA_SIZE),
            HCCL_E_PARA);
        recvBuf_.resize(remoteSize_);
        phase_ = ExchangePhase::SEND_DATA;
    }
    if (phase_ == ExchangePhase::SEND_DATA) {
        HcclResult ret = StepSend(socket, sendBuf_.data(), sendBuf_.size());
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        phase_ = ExchangePhase::RECV_DATA;
    }
    if (phase_ == ExchangePhase::RECV_DATA) {
        HcclResult ret = StepRecv(socket, recvBuf_.data(), recvBuf_.size());
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
    }

    // 四阶段全完成，解包
    Hccl::BinaryStream recvBinStream(recvBuf_);
    HcclResult unpackRet = unpackFn(recvBinStream);
    if (unpackRet != HCCL_SUCCESS) {
        phase_ = ExchangePhase::FAILED;
        failureCode_ = unpackRet;
        return unpackRet;
    }
    phase_ = ExchangePhase::DONE;
    return HCCL_SUCCESS;
}

void StepTransfer::Reset()
{
    sendOff_ = 0;
    recvOff_ = 0;
    phase_ = ExchangePhase::SEND_SIZE;
    failureCode_ = HCCL_E_INTERNAL;
    remoteSize_ = 0;
    sendBuf_.clear();
    recvBuf_.clear();
}
} // namespace hcomm
