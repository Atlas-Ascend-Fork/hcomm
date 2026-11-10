/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_CHANNEL_PROCESS_H__
#define __AICPU_CHANNEL_PROCESS_H__

#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "common.h"
#include "channel_param.h"
#include "base_transport_lite_impl.h"
#include "endpoint_pair/channel/device/aicpu_channel_pub.h"

class AicpuChannelProcess {
public:
    ~AicpuChannelProcess() = default;
    static HcclResult ParsePackData(std::vector<char>& data, ChannelHandle& handle);
    static HcclResult InitUrmaChannel(HcclChannelUrmaRes* commParam);
    static HcclResult AicpuChannelInit(HcclChannelUrmaRes* commParam);
    static HcclResult AicpuChannelDestroy(HcclChannelUrmaRes* commParam);
    static HcclResult InitHcommChannelRes(HcommChannelRes* commParam);

private:
    friend bool hcomm::IsValidLiteTransportHandle(ChannelHandle handle);

    static std::mutex mutex_;
    static std::unordered_map<ChannelHandle, std::unique_ptr<Hccl::BaseTransportLiteImpl>> transportMap_;
    // transpUniqueId（序列化的QP/CQ上下文，内容稳定）到已注册transport句柄的反查表，
    // 用于通道重复注册时复用原对象，避免重建transport导致软件影子状态与硬件环残留状态失配;
    // 与handleUniqueIdMap_成对维护, 且在AicpuChannelDestroy中与transportMap_同生命周期清除,
    // 防止句柄地址被后续新transport复用后, 旧表项把注册导向他人对象(跨通道串用)
    static std::map<std::vector<char>, ChannelHandle> uniqueIdHandleMap_;
    static std::unordered_map<ChannelHandle, std::vector<char>> handleUniqueIdMap_;
    // ROCE去重复用使同一transport被多个通道句柄共享, 引用计数保证所有共享通道销毁后才真正析构,
    // 避免任一通道先销毁令其余通道句柄悬垂; UB/P2P不走去重, 无计数条目, 销毁时直接析构
    static std::unordered_map<ChannelHandle, uint32_t> handleRefCnt_;
};
#endif // __AICPU_CHANNEL_PROCESS_H__
