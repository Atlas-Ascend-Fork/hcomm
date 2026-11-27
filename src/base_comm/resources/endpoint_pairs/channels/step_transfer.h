/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STEP_TRANSFER_H
#define STEP_TRANSFER_H

#include <functional>
#include <vector>
#include "socket.h"
#include "binary_stream.h"

namespace hcomm {

/**
 * @note 职责：Peer 模式建链的数据交换器（拆步推进）。
 *       StepSend/StepRecv：单次 Socket::ISend/IRecv 有限尝试，整块完成返回 HCCL_SUCCESS 并归零
 *       已完成字节数（实例可复用于下一块）；未完成返回 HCCL_E_AGAIN（进度保留，调用方下步续传，
 *       本端 SaluSleep 1ms 防外层轮询空转）。
 *       StepExchange：发尺寸→收尺寸→发数据→收数据四阶段，packFn/unpackFn 注入通道差异；
 *       对端尺寸非法返回 HCCL_E_PARA；DONE/FAILED 终态重入幂等（不重复解包、不吞错）。
 *       同一实例仅承载一种功能的传输，实例复用须显式 Reset()。
 */
class StepTransfer {
public:
    HcclResult StepSend(const Hccl::Socket* socket, const void* data, u64 size);
    HcclResult StepRecv(const Hccl::Socket* socket, void* data, u64 size);
    HcclResult StepExchange(
        const Hccl::Socket* socket, const std::function<HcclResult(Hccl::BinaryStream&)>& packFn,
        const std::function<HcclResult(Hccl::BinaryStream&)>& unpackFn);
    void Reset();
    u64 GetSendBufSize() const { return sendBuf_.size(); }
    u64 GetRecvBufSize() const { return recvBuf_.size(); }

    enum class ExchangePhase : u32 {
        SEND_SIZE = 0, // 发送本端数据包尺寸
        RECV_SIZE,     // 接收对端数据包尺寸（完成后校验上限，防 OOM）
        SEND_DATA,     // 发送数据包
        RECV_DATA,     // 接收数据包
        DONE,          // 成功终态：解包成功进入，重入返回 SUCCESS
        FAILED,        // 失败终态：解包失败进入，重入返回 failureCode_
    };

private:
    u64 sentBytes_{0}; // 当前已完成发送的字节数（全部完成后自动归零）
    u64 recvBytes_{0}; // 当前已完成接收的字节数（全部完成后自动归零）
    ExchangePhase phase_{ExchangePhase::SEND_SIZE};
    HcclResult failureCode_{HCCL_E_INTERNAL}; // FAILED 终态记忆的解包错误码
    u64 remoteSize_{0};
    std::vector<char> sendBuf_; // 打包产物（首步生成，跨步保留）
    std::vector<char> recvBuf_; // 接收缓冲（按对端尺寸跨步扩容）
};
} // namespace hcomm
#endif
