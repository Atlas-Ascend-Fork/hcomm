/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 跨 kernel 调用命名 func 的端到端示例：把一个简化 mesh-1D AllGather 整体包进 kernel1 的命名
// ccu::Func，再由 kernel2 按 pc 调用。与单 kernel 写法的两点差别：
//   1) 运行期参数不再由被调 kernel 自己 LoadArg，而是调用方按下标填进 ccu::Array 传入；
//   2) 通道资源、rank 拓扑等编译期信息仍由 kernel1 的 lambda 捕获，参与 microcode 生成。
// kernel1 本身不会被 launch，它只提供 func 的 microcode 与入口 pc。

#ifndef CCU_XKERNEL_ALLGATHER_DEMO_H
#define CCU_XKERNEL_ALLGATHER_DEMO_H

#include "ccu_primitives.hpp"
#include "ccu_types.h"

#include <cstdint>

namespace ccu = ::AscendC::ccu;

constexpr uint32_t XK_AG_MAX_RANK_SIZE = 4;
constexpr const char* XK_AG_FUNC_NAME = "XkAllGatherFunc";

// 命名 func 的形参下标，调用方 kernel 必须按同一顺序填 Array
constexpr uint32_t XK_AG_ARG_INPUT = 0;
constexpr uint32_t XK_AG_ARG_OUTPUT = 1;
constexpr uint32_t XK_AG_ARG_TOKEN = 2;
constexpr uint32_t XK_AG_ARG_SLICE_SIZE = 3;
constexpr uint32_t XK_AG_ARG_NUM = 4;

// kernel2 的 taskArgs 布局：0 号放 func 入口 pc，其后依次是转发给 func 的实参
constexpr uint32_t XK_AG_TASK_ARG_PC = 0;
constexpr uint32_t XK_AG_TASK_ARG_NUM = XK_AG_ARG_NUM + 1;

constexpr uint32_t XK_AG_OUTPUT_XN_ID = 0;
constexpr uint32_t XK_AG_TOKEN_XN_ID = 1;
// 本 rank 的本地搬运借用 peer mask 之外的一位，避免与远端写的完成位冲突
constexpr uint16_t XK_AG_SELF_MASK = 1 << XK_AG_MAX_RANK_SIZE;
constexpr uint64_t XK_AG_LOOP_ITER_NUM = 2;

struct XkAllGatherKernelArg {
    uint32_t rankId;
    uint32_t rankSize;
    ChannelHandle channels[XK_AG_MAX_RANK_SIZE];
    uint32_t channelCount;
};

// kernel1 作用域内的资源。func body 只捕获引用，真正的分配发生在 kernel1 注册期，
// 因此这些 Xn / Event 在 func 的 microcode 里是固定编号，与调用方是谁无关。
struct XkAllGatherRes {
    const XkAllGatherKernelArg* arg{nullptr};

    // 远端 rank 的 output 基址与 mem token，由通道资源在编译期绑定
    ccu::Variable peerOutput[XK_AG_MAX_RANK_SIZE];
    ccu::Variable peerToken[XK_AG_MAX_RANK_SIZE];

    ccu::Event event;
    ccu::LocalAddr localSrc;
    ccu::LocalAddr localDst;
    ccu::RemoteAddr remoteDst[XK_AG_MAX_RANK_SIZE];
    ccu::Variable loopLen;
};

// 简化 mesh-1D AllGather：本 rank 把自己的输入片推给所有 peer 的 output 对应偏移，
// 同时把自己那一片搬到本地 output，最后等所有远端写完成。
inline void XkAllGatherBody(XkAllGatherRes& res, ccu::Array<ccu::Variable>& args)
{
    const XkAllGatherKernelArg* arg = res.arg;

    // 原算子在 kernel 顶层用 LoadArg 取这些运行期参数，改成 func 后按下标从调用方 Array 取
    ccu::Variable sliceSize;
    sliceSize = args[XK_AG_ARG_SLICE_SIZE];

    // 本 rank 在 output 里的片偏移 = sliceSize * rankId。rankId 是编译期常量，且 V1 微码
    // 没有变量乘法，这里按 rankId 累加展开。
    ccu::Variable selfOffset;
    selfOffset = 0;
    for (uint32_t i = 0; i < arg->rankId; i++) {
        selfOffset += sliceSize;
    }

    res.localSrc.addr = args[XK_AG_ARG_INPUT];
    res.localSrc.token = args[XK_AG_ARG_TOKEN];

    res.localDst.addr = args[XK_AG_ARG_OUTPUT];
    res.localDst.addr += selfOffset;
    res.localDst.token = args[XK_AG_ARG_TOKEN];

    // 1) 推给每个 peer：目标地址 = peer output 基址 + 本 rank 的片偏移
    uint32_t channelIdx = 0;
    for (uint32_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            continue;
        }
        res.remoteDst[peerId].addr = res.peerOutput[peerId];
        res.remoteDst[peerId].addr += selfOffset;
        res.remoteDst[peerId].token = res.peerToken[peerId];
        ccu::Write(
            arg->channels[channelIdx], res.remoteDst[peerId], res.localSrc, sliceSize, res.event,
            static_cast<uint16_t>(1U << peerId));
        channelIdx++;
    }

    // 2) 本地那一片用 Loop 分片搬运，验证 func body 内可以带 Loop / LoopGroup
    res.loopLen = sliceSize;
    ccu::Func loopBody([&res]() {
        ccu::LocalCopy(res.localDst, res.localSrc, res.loopLen, res.event, XK_AG_SELF_MASK);
        ccu::EventWait(res.event, XK_AG_SELF_MASK);
    });
    ccu::LoopConfig loopCfg = {.addrOffset = 0, .iterNum = XK_AG_LOOP_ITER_NUM};
    ccu::Loop loop(loopCfg, loopBody);
    ccu::LoopGroupConfig grpCfg
        = {.cloneNum = 0, .cloneLoopOffset = 0, .addrOffset = 0, .ccuBufferOffset = 0, .eventOffset = 0};
    ccu::LoopGroup group(grpCfg, 1, {loop});

    // 3) 等所有 peer 方向的写完成
    const uint16_t peerMask = static_cast<uint16_t>(((1U << arg->rankSize) - 1U) & ~(1U << arg->rankId));
    ccu::EventWait(res.event, peerMask);
}

// kernel1：只声明命名 func，不写任何 LoadArg。它不会被 launch，注册的唯一目的是
// 让这段 AllGather microcode 落到 die 上并拿到入口 pc。
inline CcuResult CcuXkAllGatherDefineKernel(CcuKernelArg arg)
{
    auto* kernelArg = static_cast<XkAllGatherKernelArg*>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize == 0 || kernelArg->rankSize > XK_AG_MAX_RANK_SIZE) {
        return CcuResult::CCU_E_PARA;
    }

    XkAllGatherRes res;
    res.arg = kernelArg;

    // 远端 output / token 走通道资源绑定，属编译期解析，不依赖本 kernel 被 launch
    uint32_t channelIdx = 0;
    for (uint32_t peerId = 0; peerId < kernelArg->rankSize; peerId++) {
        if (peerId == kernelArg->rankId) {
            continue;
        }
        res.peerOutput[peerId]
            = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[channelIdx], XK_AG_OUTPUT_XN_ID);
        res.peerToken[peerId] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[channelIdx], XK_AG_TOKEN_XN_ID);
        channelIdx++;
    }

    // 命名 func 定义在 kernel 内部，声明即注册到本 kernel，lambda 直接捕获 kernelArgs 与本 kernel 的资源。
    // 因为函数体在构造时就执行，上面那些通道资源必须先绑定好。
    ccu::Func allGather(
        [&res](ccu::Array<ccu::Variable>& args) {
            XkAllGatherBody(res, args);
        },
        XK_AG_ARG_NUM, XK_AG_FUNC_NAME);
    return CcuResult::CCU_SUCCESS;
}

// kernel2：拿到 kernel1 的 func 入口 pc 后跨 kernel 调用，实参按下标填进 Array。
inline CcuResult CcuXkAllGatherCallKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Variable pc{};
    CCU_CHK_RET(ccu::LoadArg(pc, XK_AG_TASK_ARG_PC));

    ccu::Array<ccu::Variable> args(XK_AG_ARG_NUM);
    for (uint32_t i = 0; i < XK_AG_ARG_NUM; i++) {
        CCU_CHK_RET(ccu::LoadArg(args[i], i + 1));
    }

    return ccu::CallFunc(pc, args);
}

// kernel2 变体：实参填满 32 个上限。被调 func 只声明了 XK_AG_ARG_NUM 个形参，
// 多出来的实参写进共享传参寄存器后被忽略。
inline CcuResult CcuXkAllGatherCallMaxArgKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Variable pc{};
    CCU_CHK_RET(ccu::LoadArg(pc, XK_AG_TASK_ARG_PC));

    ccu::Array<ccu::Variable> args(CCU_FUNC_XKERNEL_ARG_MAX);
    for (uint32_t i = 0; i < XK_AG_ARG_NUM; i++) {
        CCU_CHK_RET(ccu::LoadArg(args[i], i + 1));
    }
    for (uint32_t i = XK_AG_ARG_NUM; i < CCU_FUNC_XKERNEL_ARG_MAX; i++) {
        args[i] = 0;
    }

    return ccu::CallFunc(pc, args);
}

// kernel2 变体：实参超过 32 个上限，应被前端直接拒绝。
inline CcuResult CcuXkAllGatherCallOverMaxArgKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Variable pc{};
    CCU_CHK_RET(ccu::LoadArg(pc, XK_AG_TASK_ARG_PC));

    ccu::Array<ccu::Variable> args(CCU_FUNC_XKERNEL_ARG_MAX + 1);
    return ccu::CallFunc(pc, args);
}

#endif // CCU_XKERNEL_ALLGATHER_DEMO_H
