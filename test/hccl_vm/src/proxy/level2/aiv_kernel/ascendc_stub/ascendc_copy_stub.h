/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_C_COPY_STUB_H
#define ASCEND_C_COPY_STUB_H

#include "ai_core_stub.h"
#include "aiv_task.h"
#include "ascendc_base_stub.h"
#include "ascendc_memory_stub.h"
#include "sim_log.h"

namespace AscendC {
const uint8_t DEFAULT_DATA_COPY_NBURST = 1;
const uint8_t DEFAULT_DATA_COPY_STRIDE = 0;

struct DataCopyParams {
    __aicore__ DataCopyParams() {}

    __aicore__
    DataCopyParams(const uint16_t count, const uint16_t len, const uint16_t srcStrideIn, const uint16_t dstStrideIn)
        : blockCount(count),
          blockLen(len),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn)
    {}

    uint16_t blockCount = DEFAULT_DATA_COPY_NBURST;
    uint16_t blockLen = 0;
    uint16_t srcStride = DEFAULT_DATA_COPY_STRIDE;
    uint16_t dstStride = DEFAULT_DATA_COPY_STRIDE;
};

struct DataCopyExtParams {
    __aicore__ DataCopyExtParams() {}

    __aicore__ DataCopyExtParams(
        const uint16_t count, const uint32_t len, const uint32_t srcStrideIn, const uint32_t dstStrideIn,
        const uint32_t rsvIn)
        : blockCount(count),
          blockLen(len),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn),
          rsv(rsvIn)
    {}

    uint16_t blockCount = DEFAULT_DATA_COPY_NBURST;
    uint32_t blockLen = 0;
    uint32_t srcStride = DEFAULT_DATA_COPY_STRIDE;
    uint32_t dstStride = DEFAULT_DATA_COPY_STRIDE;
    uint32_t rsv = 0; // reserved information
};

template <typename T>
struct DataCopyPadExtParams {
    __aicore__ DataCopyPadExtParams() {}

    __aicore__
    DataCopyPadExtParams(const bool isPadValue, const uint8_t leftPadValue, const uint8_t rightPadValue, T padValue)
        : isPad(isPadValue),
          leftPadding(leftPadValue),
          rightPadding(rightPadValue),
          paddingValue(padValue)
    {}

    bool isPad = false;
    uint8_t leftPadding = 0;
    uint8_t rightPadding = 0;
    T paddingValue = 0;
};

template <typename T>
__aicore__ void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count)
{
    const auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }

    const uint64_t len = static_cast<uint64_t>(count) * sizeof(T);
    // src DataSlice
    const AivSim::AivDataSlice srcSlice = AivSim::AivKernelExecutor::GetInstance().ResolveGlobalDataSlice(
        src.GetPhyAddr(), len, nullptr, static_cast<AivSim::BlockId>(curBlockIdx));
    if (srcSlice.GetDeviceId() == UINT32_MAX) {
        HCCL_VM_ERROR(
            "DataCopy resolved UB source failed, blockIdx={}, addr=0x{:x}, len={}", curBlockIdx, src.GetPhyAddr(), len);
        return;
    }
    // dst DataSlice
    const auto dstAddr = reinterpret_cast<uint64_t>(dst.GetPhyAddr());
    const AivSim::AivDataSlice dstSlice = AivSim::AivKernelExecutor::GetInstance().ResolveGlobalDataSlice(dstAddr, len);
    if (dstSlice.GetDeviceId() == UINT32_MAX) {
        HCCL_VM_ERROR("DataCopy resolved data slice failed, addr=0x{:x} len={:d}", dstAddr, len);
        return;
    }

    std::shared_ptr<AivSim::AivTask> task;
    if (aiv->GetAtomicOp() == AivSim::ReduceOp::REDUCE_RESERVED) {
        task = std::make_shared<AivSim::AivTaskMemCopy>(srcSlice, dstSlice);
    } else {
        const uint32_t dataType = AivSim::AivKernelExecutor::GetInstance().GetCurOp().dataType;
        task = std::make_shared<AivSim::AivTaskReduce>(
            srcSlice, dstSlice, dataType, static_cast<uint32_t>(aiv->GetAtomicOp()));
    }
    aiv->AppendMTE3(task);
}

template <typename T>
__aicore__ void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const uint32_t count)
{
    const auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }

    const uint64_t len = static_cast<uint64_t>(count) * sizeof(T);
    // dst DataSlice
    const AivSim::AivDataSlice dstSlice = AivSim::AivKernelExecutor::GetInstance().ResolveGlobalDataSlice(
        dst.GetPhyAddr(), len, nullptr, static_cast<AivSim::BlockId>(curBlockIdx));
    if (dstSlice.GetDeviceId() == UINT32_MAX) {
        HCCL_VM_ERROR(
            "DataCopy resolved UB destination failed, blockIdx={}, addr=0x{:x}, len={}", curBlockIdx, dst.GetPhyAddr(),
            len);
        return;
    }
    // src DataSlice
    const auto srcAddr = reinterpret_cast<uint64_t>(src.GetPhyAddr());
    const AivSim::AivDataSlice srcSlice = AivSim::AivKernelExecutor::GetInstance().ResolveGlobalDataSlice(srcAddr, len);
    if (srcSlice.GetDeviceId() == UINT32_MAX) {
        HCCL_VM_ERROR("DataCopy resolved data slice failed, addr=0x{:x} len={:d}", srcAddr, len);
        return;
    }

    auto task = std::make_shared<AivSim::AivTaskMemCopy>(srcSlice, dstSlice);
    aiv->AppendMTE2(task);
}

// DataCopyPad
//
// 入参简化(暂不模拟非对齐细节):
//   1. 只取 blockCount * blockLen 字节, 不做 burst 分段;
//   2. DataCopyExtParams::blockLen 单位是字节, 不乘 sizeof(T)(这点与 DataCopy 的
//      count 是元素数不同);
//   3. srcStride / dstStride / leftPadding / rightPadding / paddingValue 全部忽略;
//   4. 长度原样透传, 不补齐到 UB 对齐。
// 将来要真跑非对齐搬运, 在这两个重载里补分段与填充即可。

// UB -> GM, 对应 DataCopyUB2GM 的非对齐分支
template <typename T>
__aicore__ void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams& copyParams)
{
    const uint64_t len = static_cast<uint64_t>(copyParams.blockCount) * static_cast<uint64_t>(copyParams.blockLen);
    if (len == 0) {
        return;
    }

    const auto curBlockIdx = GetBlockIdx();
    auto& executor = AivSim::AivKernelExecutor::GetInstance();
    auto aiv = executor.GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }

    // src 在 UB 侧: 带 blockIdx 解析
    const AivSim::AivDataSlice srcSlice = executor.ResolveGlobalDataSlice(
        static_cast<uint64_t>(src.GetPhyAddr()), len, nullptr, static_cast<AivSim::BlockId>(curBlockIdx));
    if (srcSlice.GetDeviceId() == UINT32_MAX) {
        HCCL_VM_ERROR(
            "DataCopyPad resolved UB source failed, blockIdx={}, addr=0x{:x}, len={}", curBlockIdx, src.GetPhyAddr(),
            len);
        return;
    }
    // dst 在 GM 侧
    const auto dstAddr = reinterpret_cast<uint64_t>(dst.GetPhyAddr());
    const AivSim::AivDataSlice dstSlice = executor.ResolveGlobalDataSlice(dstAddr, len);
    if (dstSlice.GetDeviceId() == UINT32_MAX) {
        HCCL_VM_ERROR("DataCopyPad resolved data slice failed, addr=0x{:x} len={:d}", dstAddr, len);
        return;
    }

    // 与 DataCopy 一致: 置了 atomic 就走 reduce, 否则走普通拷贝
    std::shared_ptr<AivSim::AivTask> task;
    if (aiv->GetAtomicOp() == AivSim::ReduceOp::REDUCE_RESERVED) {
        task = std::make_shared<AivSim::AivTaskMemCopy>(srcSlice, dstSlice);
    } else {
        const uint32_t dataType = executor.GetCurOp().dataType;
        task = std::make_shared<AivSim::AivTaskReduce>(
            srcSlice, dstSlice, dataType, static_cast<uint32_t>(aiv->GetAtomicOp()));
    }
    aiv->AppendMTE3(task);
}

// GM -> UB, 对应 DataCopyGM2UB 的非对齐分支
template <typename T>
__aicore__ void DataCopyPad(
    const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams& copyParams,
    const DataCopyPadExtParams<T>& padParams)
{
    (void)padParams; // 填充值与左右 padding 暂不建模

    const uint64_t len = static_cast<uint64_t>(copyParams.blockCount) * static_cast<uint64_t>(copyParams.blockLen);
    if (len == 0) {
        return;
    }

    const auto curBlockIdx = GetBlockIdx();
    auto& executor = AivSim::AivKernelExecutor::GetInstance();
    auto aiv = executor.GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }

    // dst 在 UB 侧: 带 blockIdx 解析
    const AivSim::AivDataSlice dstSlice = executor.ResolveGlobalDataSlice(
        static_cast<uint64_t>(dst.GetPhyAddr()), len, nullptr, static_cast<AivSim::BlockId>(curBlockIdx));
    if (dstSlice.GetDeviceId() == UINT32_MAX) {
        HCCL_VM_ERROR(
            "DataCopyPad resolved UB destination failed, blockIdx={}, addr=0x{:x}, len={}", curBlockIdx,
            dst.GetPhyAddr(), len);
        return;
    }
    // src 在 GM 侧
    const auto srcAddr = reinterpret_cast<uint64_t>(src.GetPhyAddr());
    const AivSim::AivDataSlice srcSlice = executor.ResolveGlobalDataSlice(srcAddr, len);
    if (srcSlice.GetDeviceId() == UINT32_MAX) {
        HCCL_VM_ERROR("DataCopyPad resolved data slice failed, addr=0x{:x} len={:d}", srcAddr, len);
        return;
    }

    aiv->AppendMTE2(std::make_shared<AivSim::AivTaskMemCopy>(srcSlice, dstSlice));
}

__aicore__ inline void SetAtomicNone()
{
    auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }
    aiv->SetAtomicOp();
}

template <typename T>
__aicore__ void SetAtomicAdd()
{
    auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }
    aiv->SetAtomicOp(AivSim::ReduceOp::REDUCE_SUM);
}

template <typename T>
__aicore__ void SetAtomicMax()
{
    auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }
    aiv->SetAtomicOp(AivSim::ReduceOp::REDUCE_MAX);
}

template <typename T>
__aicore__ void SetAtomicMin()
{
    auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }
    aiv->SetAtomicOp(AivSim::ReduceOp::REDUCE_MIN);
}

template <typename T>
__aicore__ void Duplicate(const LocalTensor<T>& dstLocal, const T& scalar, const int32_t& count)
{}
} // namespace AscendC

#endif // ASCEND_C_COPY_STUB_H
