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

// kernel 里直接调用了 hcomm_.WriteValueNbi，这里留同名空入口
class AivUrmaHcommStub {
public:
    __aicore__ inline void Init(uint64_t, uint32_t) {}

    template <typename T>
    __aicore__ inline void WriteValueNbi(uint64_t, GM_ADDR, uint64_t)
    {}
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
