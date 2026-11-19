/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EP_MEM_TABLE_LAUNCH_H
#define EP_MEM_TABLE_LAUNCH_H

#include <cstdint>

#include "endpoint_pair/channel/channel_param_pub.h"

namespace hcomm {
// UB 三类 device endpoint 的 DevEpMemTable 生命周期下发（AICPU kernel + stream sync）。
// bin：进程级 AicpuCommEngineBinProc（与建链/线程 kernel 同一 ccl_kernel.json）。
HcclResult LaunchEpMemTableInit(uint64_t& outTablePtr);
HcclResult LaunchEpMemTableUpdate(
    uint64_t tablePtr, EpMemTableOp op, uint64_t addr, uint64_t size, uint32_t tokenId, uint32_t tokenValue);
HcclResult LaunchEpMemTableDestroy(uint64_t tablePtr);

// 表更新纯值下发（Reg/Unreg 钩子共用）：addr/size/tokenId/tokenValue 由调用方在对象存活期提取；
// REMOVE 仅消费 addr/size，tokenId/tokenValue 传 0（kernel 侧忽略）
HcclResult SyncUbLocMemToEpTable(
    uint64_t tablePtr, EpMemTableOp op, uint64_t addr, uint64_t size, uint32_t tokenId, uint32_t tokenValue);
} // namespace hcomm

#endif // EP_MEM_TABLE_LAUNCH_H
