/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV1_BATCH_SEND_RECV_SEMANTICS_CHECKER_H
#define HCCLV1_BATCH_SEND_RECV_SEMANTICS_CHECKER_H

#include "check_utils.h"
#include "hccl_types.h"

namespace HcclSim {
HcclResult TaskCheckBatchSendRecvSemantics(
    const PhysicalMemorySemantics& memorySemantics, const DeviceBufferAddressLayouts& bufferLayouts,
    uint32_t expectedRankSize, uint64_t dataSize, const std::vector<DeviceId>& rankToDevice);
HcclResult TaskCheckBatchSendRecvRingSemantics(
    const PhysicalMemorySemantics& memorySemantics, const DeviceBufferAddressLayouts& bufferLayouts,
    uint32_t expectedRankSize, uint64_t dataSize, const std::vector<DeviceId>& rankToDevice);
} // namespace HcclSim

#endif
