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

// 分步传输器（Peer模式建链拆步推进）
// 功能一 单步收发（StepSend/StepRecv）：一次 Socket::ISend/IRecv 有限尝试，块内偏移内部累计，
//   整块完成返回 HCCL_SUCCESS 并归零偏移（实例可复用于下一块）；未完成返回 HCCL_E_AGAIN
//   （偏移保留，调用方下步续传）。
// 模式二 四阶段交换（StepExchange）：发尺寸→收尺寸→发数据→收数据，packFn/unpackFn 注入通道
//   打包/解包差异；已完成阶段同步直落推进，部分进度即返回 HCCL_E_AGAIN
//   对端尺寸非法返回 HCCL_E_PARA；解包成功进入 DONE 终态（重入返回 HCCL_SUCCESS，不重复解包），
//   解包失败进入 FAILED 终态（重入返回原错误码，不吞错）。
// 同一实例仅承载一种功能的传输；当前channel生命周期内无重交换场景，实例复用须显式 Reset()。
struct StepTransfer {
    u64 sendOff_{0}; // 当前发送块内偏移（块完成自动归零）
    u64 recvOff_{0}; // 当前接收块内偏移（块完成自动归零）

    enum class ExchangePhase : u32 {
        SEND_SIZE = 0, // 发送本端数据包尺寸
        RECV_SIZE,     // 接收对端数据包尺寸（完成后校验上限，防 OOM）
        SEND_DATA,     // 发送数据包
        RECV_DATA,     // 接收数据包
        DONE,          // 成功终态：解包成功进入，重入返回 SUCCESS
        FAILED,        // 失败终态：解包失败进入，重入返回 failureCode_
    };
    ExchangePhase phase_{ExchangePhase::SEND_SIZE};
    HcclResult failureCode_{HCCL_E_INTERNAL}; // FAILED 终态记忆的解包错误码
    u64 remoteSize_{0};
    std::vector<char> sendBuf_; // 打包产物（首步生成，跨步保留）
    std::vector<char> recvBuf_; // 接收缓冲（按对端尺寸跨步扩容）

    HcclResult StepSend(Hccl::Socket* socket, const void* data, u64 size);
    HcclResult StepRecv(Hccl::Socket* socket, void* data, u64 size);
    HcclResult StepExchange(
        Hccl::Socket* socket, const std::function<HcclResult(Hccl::BinaryStream&)>& packFn,
        const std::function<HcclResult(Hccl::BinaryStream&)>& unpackFn);
    void Reset();
    u64 GetSendBufSize() const { return sendBuf_.size(); }
    u64 GetRecvBufSize() const { return recvBuf_.size(); }
};
} // namespace hcomm
#endif
