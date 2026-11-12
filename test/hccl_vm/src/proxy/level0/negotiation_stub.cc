/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define HCCL_VM_MODULE "NEGOTIATION_STUB"

#include <cstdint>

#include "hccl/hccl_types.h"
#include "sim_log.h"

extern "C" {

HcclResult CheckCcuResNegotiationC(HcclComm comm, const void* param, bool localResAvailable)
{
    HCCL_VM_INFO("this func is empty");
    if (!localResAvailable) {
        return HcclResult::HCCL_E_UNAVAIL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CheckCcuParamAndFallbackC(HcclComm comm, void* param, void** topoInfo, char* algNameBuf, uint32_t algNameBufLen)
{
    HCCL_VM_INFO("this func is empty");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult NegotiateCommC(
    HcclComm comm, const char* typeTag, int op, const void* sendBuf, uint64_t sendSize, void* recvBuf,
    uint64_t recvSize)
{
    HCCL_VM_INFO("this func is empty");
    return HcclResult::HCCL_SUCCESS;
}

} // extern "C"
