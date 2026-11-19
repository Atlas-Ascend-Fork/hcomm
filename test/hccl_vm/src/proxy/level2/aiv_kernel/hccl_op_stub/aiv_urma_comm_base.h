/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_URMA_COMM_BASE_H
#define AIV_URMA_COMM_BASE_H

// 仿真桩：对应 hccl/src/ops/op_common/algorithm/template/aiv/aiv_urma_comm_base.h。
// 真身依赖 devkit 的 adv_api/hcomm.h（仿真环境没有该目录），这里只保留 AIV URMA kernel
// 编译期需要的类型、常量与成员函数签名，函数体一律为空，不产生任何数据面行为。

#include "version/hcomm_version.h"
#include "version/asc_devkit_version.h"
#include "aiv_defines.h"

struct AivUrmaChannelInfo {
    uint32_t rankSize;
    uint32_t channelsPerRank[MAX_RANK_SIZE];
    uint64_t channelHandle[MAX_RANK_SIZE * MAX_CHANNELS_PER_RANK];
    uint32_t portNum[MAX_RANK_SIZE * MAX_CHANNELS_PER_RANK];
};

using ChannelHandle = uint64_t;

// ---- devkit 类型的仿真副本（adv_api/hcomm 在仿真环境不存在）----
#pragma pack(push, 1)
struct UrmaWqeEntry {
    uint32_t odr : 3;
    uint32_t fence : 1;
    uint32_t se : 1;
    uint32_t cqe : 1;
    uint32_t inlineEn : 1;
};
#pragma pack(pop)

static constexpr UrmaWqeEntry URMA_DEFAULT_CFG = {.odr = 5, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};
static constexpr UrmaWqeEntry URMA_INLINE_CFG = {.odr = 5, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 1};

struct BufDesc {
    GM_ADDR addr;
    uint32_t len;
};

enum class HcommUrmaReduceOp : uint32_t {
    MAX = 0x8U,
    MIN = 0x9U,
    SUM = 0xAU,
};

struct UbcBatchHandle {
    ChannelHandle channelHandle = 0;
    uint8_t* buffer = nullptr;
    uint32_t capacity = 0;
    uint32_t pending = 0;
    uint64_t batchId = 0;
};

template <typename T>
struct HandleTraits;

template <>
struct HandleTraits<ChannelHandle> {
    using ChannelType = ChannelHandle;
    using BatchHandleType = UbcBatchHandle;
};

template <>
struct HandleTraits<UbcBatchHandle> {
    using ChannelType = ChannelHandle;
    using BatchHandleType = UbcBatchHandle;
};

template <typename T>
using BatchHandle = typename HandleTraits<T>::BatchHandleType;

// Hcomm 全接口空实现：只保证 kernel 编译通过，不产生任何行为。
class AivUrmaHcommStub {
public:
    __aicore__ inline int32_t Init(uint8_t*, uint32_t) { return 0; }

    template <typename T>
    __aicore__ inline int32_t Init(const LocalTensor<T>&, uint32_t)
    {
        return 0;
    }

    template <typename T, typename U>
    __aicore__ inline BatchHandle<T>
    MakeBatchHandle(T, const LocalTensor<U>&, uint32_t, GM_ADDR = nullptr, GM_ADDR = nullptr)
    {
        return BatchHandle<T>{};
    }

    template <typename T, typename HandleTraits<T>::ChannelType* = nullptr>
    __aicore__ inline BatchHandle<T>& GetHandleRef(T&, uint32_t, GM_ADDR = nullptr)
    {
        static BatchHandle<T> dummy{};
        return dummy;
    }

    template <
        bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
        auto const& config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteNbi(ChannelHandle, GM_ADDR, GM_ADDR, uint64_t)
    {
        return 0;
    }

    template <auto const& config = URMA_DEFAULT_CFG, typename T, typename HandleTraits<T>::ChannelType* = nullptr>
    __aicore__ inline int32_t WriteNbi(T&, GM_ADDR, GM_ADDR, uint32_t)
    {
        return 0;
    }

    template <auto const& config = URMA_DEFAULT_CFG, typename T, typename HandleTraits<T>::ChannelType* = nullptr>
    __aicore__ inline int32_t WriteNbi(T&, GM_ADDR, const BufDesc*, uint32_t)
    {
        return 0;
    }

    template <
        typename T, bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
        auto const& config = URMA_INLINE_CFG>
    __aicore__ inline int32_t WriteValueNbi(ChannelHandle, GM_ADDR, T)
    {
        return 0;
    }

    template <
        typename T, HcommUrmaReduceOp reduceOp, bool commit = true, pipe_t commitPipe = PIPE_S,
        pipe_t reqPipe = PIPE_MTE3, auto const& config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteReduceNbi(ChannelHandle, GM_ADDR, GM_ADDR, uint64_t)
    {
        return 0;
    }

    template <
        bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
        auto const& config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteWithNotifyNbi(ChannelHandle, GM_ADDR, GM_ADDR, uint64_t, GM_ADDR, uint64_t)
    {
        return 0;
    }

    template <auto const& config = URMA_DEFAULT_CFG, typename T, typename HandleTraits<T>::ChannelType* = nullptr>
    __aicore__ inline int32_t WriteWithNotifyNbi(T&, GM_ADDR, GM_ADDR, uint32_t, GM_ADDR, uint64_t)
    {
        return 0;
    }

    template <auto const& config = URMA_DEFAULT_CFG, typename T, typename HandleTraits<T>::ChannelType* = nullptr>
    __aicore__ inline int32_t WriteWithNotifyNbi(T&, GM_ADDR, const BufDesc*, uint32_t, GM_ADDR, uint64_t)
    {
        return 0;
    }

    template <
        typename T, bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
        auto const& config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t AtomicFAA(ChannelHandle, GM_ADDR, GM_ADDR, T)
    {
        return 0;
    }

    template <
        typename T, bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
        auto const& config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t AtomicCAS(ChannelHandle, GM_ADDR, GM_ADDR, T, T)
    {
        return 0;
    }

    template <
        bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
        auto const& config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t ReadNbi(ChannelHandle, GM_ADDR, GM_ADDR, uint64_t)
    {
        return 0;
    }

    template <auto const& config = URMA_DEFAULT_CFG, typename T, typename HandleTraits<T>::ChannelType* = nullptr>
    __aicore__ inline int32_t ReadNbi(T&, GM_ADDR, GM_ADDR, uint32_t)
    {
        return 0;
    }

    template <auto const& config = URMA_DEFAULT_CFG, typename T, typename HandleTraits<T>::ChannelType* = nullptr>
    __aicore__ inline int32_t ReadNbi(T&, const BufDesc*, uint32_t, GM_ADDR)
    {
        return 0;
    }

    template <pipe_t pipe = PIPE_S>
    __aicore__ inline int32_t Commit(ChannelHandle)
    {
        return 0;
    }

    template <typename T, typename HandleTraits<T>::ChannelType* = nullptr>
    __aicore__ inline int32_t BatchCommit(T&)
    {
        return 0;
    }

    template <pipe_t pipe = PIPE_MTE3>
    __aicore__ inline int32_t Drain(ChannelHandle)
    {
        return 0;
    }

    template <pipe_t pipe = PIPE_S, typename T, typename HandleTraits<T>::ChannelType* = nullptr>
    __aicore__ inline int32_t Drain(T&)
    {
        return 0;
    }

    __aicore__ inline int32_t Lock(ChannelHandle) { return 0; }

    __aicore__ inline int32_t Unlock(ChannelHandle) { return 0; }
};

class AivUrmaCommBase {
public:
    __aicore__ inline AivUrmaCommBase() = default;

    __aicore__ inline void Init(
        GM_ADDR, uint64_t input, uint64_t output, uint32_t rank, uint32_t, uint32_t subCommRankSize, uint64_t len,
        uint32_t, uint32_t, uint32_t, uint64_t inputSliceStride, uint64_t outputSliceStride, uint64_t repeatNum,
        uint64_t inputRepeatStride, uint64_t outputRepeatStride, GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, bool,
        uint32_t numBlocks, uint32_t subCommRankLevel, uint32_t level0ChannelNumPerRank)
    {
        input_ = input;
        output_ = output;
        rank_ = rank;
        subCommRankSize_ = subCommRankSize;
        len_ = len;
        inputSliceStride_ = inputSliceStride;
        outputSliceStride_ = outputSliceStride;
        repeatNum_ = repeatNum;
        inputRepeatStride_ = inputRepeatStride;
        outputRepeatStride_ = outputRepeatStride;
        numBlocks_ = numBlocks;
        subCommRankLevel_ = subCommRankLevel;
        level0ChannelNumPerRank_ = level0ChannelNumPerRank;
        blockIdx_ = GetBlockIdx();
        notifyBase_ = FLAG1_OFFSET;
    }

    __aicore__ inline uint64_t GetPeerCommInfoBase(uint32_t) { return 0; }

    __aicore__ inline uint64_t ReadPeerOutputAddr(uint32_t) { return 0; }

    __aicore__ inline void SubmitDataOnBatch(uint64_t, uint64_t, uint64_t, uint64_t, bool, uint64_t, uint64_t) {}

    __aicore__ inline void FlushBatch() {}

    __aicore__ inline void DrainAllChannels() {}

    __aicore__ inline void WaitNotify(uint64_t, uint32_t, bool) {}

    __aicore__ inline void SendRecvOutputAddr() {}

    __aicore__ inline void LocalCopy(uint64_t, uint64_t, uint64_t) {}

    __aicore__ inline void ClearGM(uint64_t, uint64_t) {}

    __aicore__ inline uint32_t GetSubCommRank(uint32_t) { return 0; }

    __aicore__ inline uint32_t GetChannelsPerRank(uint32_t) { return 0; }

    __aicore__ inline uint64_t GetChannelHandle(uint32_t, uint32_t) { return 0; }

    __aicore__ inline uint32_t GetPortNum(uint32_t, uint32_t) { return 0; }

protected:
    AivUrmaHcommStub hcomm_;

    uint64_t input_{0};
    uint64_t output_{0};
    uint32_t rank_{0};
    uint32_t subCommRankSize_{0};
    uint64_t len_{0};
    uint64_t inputSliceStride_{0};
    uint64_t outputSliceStride_{0};
    uint64_t repeatNum_{0};
    uint64_t inputRepeatStride_{0};
    uint64_t outputRepeatStride_{0};
    uint32_t numBlocks_{0};
    uint32_t blockIdx_{0};
    uint32_t subCommRankLevel_{0};
    uint32_t level0ChannelNumPerRank_{0};
    uint64_t notifyBase_{0};
    uint32_t mySubCommRankIdx_{0};
    uint32_t assignedRankStart_{0};
    uint32_t assignedRankEnd_{0};
};

#endif // AIV_URMA_COMM_BASE_H
