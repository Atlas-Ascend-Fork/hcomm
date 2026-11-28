/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_TRANS_H
#define HCOMM_CCU_TRANS_H

#include <cstdint>

#include "ccu_kernel.h"

namespace hcomm {

// 数据搬运类 rep 构造接口：本地拷贝、本地 reduce、远端读写。
namespace ccu_trans {

    // 本地搬运公共操作数
    struct LocalTransOperands {
        CcuVariableHandle lenHandle;
        CcuEventHandle eventHandle;
        uint32_t mask;
    };

    // 本地数据拷贝 相关接口
    CcuResult LocalCopyMemToBuffer(
        CcuKernel& kernel, const LocalTransOperands& op, CcuBufferHandle dstHandle, CcuLocalAddrHandle srcHandle);
    CcuResult LocalCopyBufferToMem(
        CcuKernel& kernel, const LocalTransOperands& op, CcuLocalAddrHandle dstHandle, CcuBufferHandle srcHandle);
    CcuResult LocalCopyMemToMem(
        CcuKernel& kernel, const LocalTransOperands& op, CcuLocalAddrHandle dstHandle, CcuLocalAddrHandle srcHandle);

    // 本地reduce 相关接口
    CcuResult LocalMemReduce(
        CcuKernel& kernel, const LocalTransOperands& op, CcuLocalAddrHandle dstHandle, CcuLocalAddrHandle srcHandle,
        HcclDataType dataType, HcclReduceOp opType);
    struct LocalReduceOperands {
        const CcuBufferHandle* bufHandles;
        uint32_t count;
        HcclDataType dataType;
        HcclDataType outputDataType;
        HcclReduceOp opType;
        CcuVariableHandle lenHandle;
        CcuEventHandle eventHandle;
        uint32_t mask;
    };
    CcuResult LocalBufferReduce(CcuKernel& kernel, const LocalReduceOperands& op);

    // 远端读写公共操作数
    struct RemoteOperands {
        ChannelHandle channel;
        CcuVariableHandle lenHandle;
        CcuEventHandle eventHandle;
        uint32_t mask;
    };

    // 远端数据传输操作
    CcuResult ReadMemToMem(
        CcuKernel& kernel, const RemoteOperands& op, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle);
    CcuResult ReadMemToBuffer(
        CcuKernel& kernel, const RemoteOperands& op, CcuBufferHandle bufHandle, CcuRemoteAddrHandle remoteHandle);
    CcuResult ReadMemToMemReduce(
        CcuKernel& kernel, const RemoteOperands& op, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle,
        HcclDataType dataType, HcclReduceOp opType);
    CcuResult WriteMemToMem(
        CcuKernel& kernel, const RemoteOperands& op, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle);
    CcuResult WriteBufferToMem(
        CcuKernel& kernel, const RemoteOperands& op, CcuRemoteAddrHandle remoteHandle, CcuBufferHandle bufHandle);
    CcuResult WriteMemToMemReduce(
        CcuKernel& kernel, const RemoteOperands& op, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle,
        HcclDataType dataType, HcclReduceOp opType);
    CcuResult WriteVarAtomicAdd(
        CcuKernel& kernel, CcuVariableHandle channelIdHandle, CcuRemoteAddrHandle varAddrHandle,
        CcuVariableHandle addValueHandle, CcuEventHandle eventHandle, uint32_t mask);
    CcuResult WriteWithCascCntInc(
        CcuKernel& kernel, CcuVariableHandle channelIdHandle, CcuRemoteAddrHandle remoteHandle,
        CcuLocalAddrHandle localHandle, CcuVariableHandle lenHandle, CcuRemoteAddrHandle inCntAddrHandle);
} // namespace ccu_trans

} // namespace hcomm

#endif // HCOMM_CCU_TRANS_H
