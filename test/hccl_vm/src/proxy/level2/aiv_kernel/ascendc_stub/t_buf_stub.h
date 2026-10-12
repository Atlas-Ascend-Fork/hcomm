/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_T_BUF_STUB_H
#define AIV_T_BUF_STUB_H

#include <cstdint>
#include "ascendc_base_stub.h"
#include "sim_log.h"
#include "local_tensor_stub.h"
#include "t_que_bind_stub.h"

namespace AscendC {
template <TPosition pos = TPosition::LCM>
class TBuf {
public:
    __aicore__ TBuf() = default;
    __aicore__ ~TBuf() = default;

    template <typename T>
    __aicore__ LocalTensor<T> Get()
    {
        return LocalTensor<T>{TPosition::LCM, block_.ptr, block_.len / sizeof(T)};
    }

    template <typename T>
    __aicore__ LocalTensor<T> GetWithOffset(uint32_t size, uint32_t bufOffset)
    {
        if ((size * sizeof(T) + bufOffset) >= block_.len) {
            HCCL_VM_ERROR("TBuf GetWithOffset out-of-bounds, bufOffset={} size={}", bufOffset, size);
        }
        uint64_t newAddr = reinterpret_cast<uint64_t>(block_.ptr) + bufOffset;
        T* newPtr = reinterpret_cast<T*>(newAddr);
        return LocalTensor<T>{TPosition::LCM, newPtr, size};
    }

private:
    TensorMem block_{};

    friend class TPipe;
    template <TPosition p>
    friend class TBufPool;
};

// 仅形式占位，不实现真实 UB 池语义；供 hif8 reduce_scatter 等内核编译通过。
template <TPosition pos = TPosition::LCM>
class TBufPool {
public:
    __aicore__ TBufPool() = default;
    __aicore__ ~TBufPool() = default;

    template <TPosition p>
    __aicore__ void InitBuffer(TBuf<p>& buf, uint32_t len)
    {
        buf.block_.ptr = nullptr;
        buf.block_.len = len;
    }

    template <TPosition src, TPosition dst, int32_t depth, auto mask = 0>
    __aicore__ void InitBuffer(TQueBind<src, dst, depth, mask>& que, uint8_t num, uint32_t len)
    {
        que.blocks_.clear();
        for (uint32_t i = 0; i < num; i++) {
            TensorMem tensorMem;
            tensorMem.ptr = nullptr;
            tensorMem.len = len;
            que.blocks_.push_back(tensorMem);
        }
        que.tensorLen_ = len;
    }

    __aicore__ void Reset() {}
};
} // namespace AscendC

#endif // AIV_T_BUF_STUB_H
