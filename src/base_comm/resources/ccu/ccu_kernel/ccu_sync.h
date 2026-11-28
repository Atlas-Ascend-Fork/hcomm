/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_SYNC_H
#define HCOMM_CCU_SYNC_H

#include <cstdint>

#include "ccu_kernel.h"

namespace hcomm {

// 同步类 rep 构造接口：Event、Notify、CascCnt 的记录与等待。
namespace ccu_sync {

    // Event信号同步类 相关接口
    //  mask 由调用方独立传入（与 Event 句柄解耦），不再设独立的 SetEventMask 接口。
    CcuResult EventRecord(CcuKernel& kernel, CcuEventHandle eventHandle, uint32_t mask);
    CcuResult EventWait(CcuKernel& kernel, CcuEventHandle eventHandle, uint32_t mask);
    CcuResult NotifyRecord(CcuKernel& kernel, const ChannelHandle channel, uint32_t remoteNotifyIdx, uint32_t mask);
    CcuResult NotifyWait(CcuKernel& kernel, const ChannelHandle channel, uint32_t localNotifyIdx, uint32_t mask);
    CcuResult WriteVariableWithNotify(
        CcuKernel& kernel, const ChannelHandle channel, CcuVariableHandle varHandle, uint32_t remoteVarIdx,
        uint32_t remoteNotifyIdx, uint32_t mask);
    CcuResult CascCntWait(CcuKernel& kernel, HcommCcuCascCntHandle cntHandle, uint64_t tgtValue);
    CcuResult
    CascCntClear(CcuKernel& kernel, HcommCcuCascCntHandle cntHandle, CcuEventHandle eventHandle, uint32_t mask);
    // 本地（同 device 内跨 core）通知同步：用 notifyTag 字符串作为对端标识，
    // 由调用方约定生产者/消费者使用相同的 tag 字符串完成配对。
    // 与 NotifyRecord/Wait（用 ChannelHandle 标识跨 rank 通道）的对偶。
    CcuResult LocalNotifyRecord(CcuKernel& kernel, const char* notifyTag, const uint32_t mask);
    CcuResult LocalNotifyWait(CcuKernel& kernel, const char* notifyTag, const uint32_t mask);
} // namespace ccu_sync

} // namespace hcomm

#endif // HCOMM_CCU_SYNC_H
