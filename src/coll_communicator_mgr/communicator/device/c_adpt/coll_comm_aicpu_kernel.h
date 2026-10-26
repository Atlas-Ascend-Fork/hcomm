/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDEPENDENT_OP_AICPU_INTERFACE_H
#define INDEPENDENT_OP_AICPU_INTERFACE_H

#include <cstdint>

// 记录 channel 列表到通信域 kernel payload：deviceHandleList 数组 + num + commTag
// 建链后 host 下发，device 侧按 commTag 定位 comm，记录 handle 列表供 Clean 使用
struct HcclAicpuCommRecordChannelsRes {
    const uint64_t* deviceHandleList{nullptr}; // device侧 channel handle 数组（ctx 模式为 ctxPtr，由 kernel 解包）
    uint32_t num{0};
    char commTag[128]{};                     // comm 标识（group）
    const uint32_t* remoteRankList{nullptr}; // device侧 remoteRank 数组，与 handleList 按下标对齐，nullptr 时跳过注册
};

// device 侧地址条目下发 payload：整批一次，每张列表按通道下标对齐
struct HcclAicpuChannelMemUpdateRes {
    const uint64_t* deviceHandleList{nullptr}; // device侧 channel handle 数组（ctx 模式为 ctxPtr，由 kernel 解包）
    const uint64_t* packAddrList{nullptr}; // 每通道条目 blob 在 device 上的起始地址
    const uint32_t* packSizeList{nullptr}; // 每通道条目 blob 字节数，0 = 跳过该通道
    uint32_t num{0};
};

extern "C" {
__attribute__((visibility("default"))) uint32_t RunAicpuIndOpChannelUpdateV2(void* args);
__attribute__((visibility("default"))) uint32_t RunAicpuIndOpThreadInit(void* args);
__attribute__((visibility("default"))) uint32_t RunAicpuIndOpNotify(void* args);
__attribute__((visibility("default"))) uint32_t RunAicpuCommInit(void* args);
__attribute__((visibility("default"))) uint32_t RunAicpuDfxInitV2(void* args);
__attribute__((visibility("default"))) uint32_t RunAicpuCommRecordChannels(void* args);
__attribute__((visibility("default"))) uint32_t RunAicpuChannelMemUpdate(void* args);
}

#endif // INDEPENDENT_OP_AICPU_INTERFACE_H
