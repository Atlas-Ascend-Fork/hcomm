/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CHANNEL_PARAM_PUB_H
#define HCOMM_CHANNEL_PARAM_PUB_H

#include <cstddef>
#include <cstdint>

#include "hccl_mem_defs.h"
#include "hccl_common.h"
#include "transport_pub.h"
#include "hccl/hccl_res.h"
#include "aicpu_operator_pub.h"
#include "endpoint_pair/channel/ub_unique_id_pub.h" // 本头文件仅做转发供下游消费点，勿删

struct DevAicpuChannelConfig {
    // 如要新增配置类字段，在此处添加
};

struct HcclChannelUrmaRes {
    char hcomId[HCOMID_MAX_LENGTH]{};    // 通信域ID 最大长度
    void* channelList{nullptr};          // 反序列后返回给host侧的device侧handle地址
    u32 listNum = 0;                     // 建链channel的总数量
    void* uniqueIdAddr{nullptr};         // 序列化后device侧地址
    u32 uniqueIdSize{0};                 // 序列化后总地址长度
    void* channelSizeAddr{nullptr};      // 存放序列化后device channel size的指针
    u32* remoteRankId{nullptr};          // 记录每个channel的对端rank
    s32 deviceLogicId{0};                // 基础通信使用
    u32 deviceType{0};                   // 基础通信使用
    DevAicpuChannelConfig channelConfig; // 收编channel配置类变量
    void* ctxList{nullptr};              // device侧ctx指针数组，非null时kernel写ctx->deviceChannel
    // device侧listNum个void*：每项为对应通道实体SqContext地址；nullptr表示该通道不绑定pi/ci
    void* entityPiCiAddrList{nullptr};
};

// 用于校验指针的合法性，防止操作到已释放的内存或非法内存
constexpr uint32_t DEV_EP_MEM_TABLE_MAGIC_WORD = 0x4D455054;
constexpr uint32_t DEV_EP_MEM_TABLE_VERSION = 1;
enum class EpMemTableOp : uint32_t {
    UPSERT = 0,
    REMOVE = 1,
};
struct LocUbBufLite {
    uint64_t addr{};
    uint64_t size{};
    uint32_t tokenId{};
    uint32_t tokenValue{};
};
static_assert(sizeof(LocUbBufLite) == 24, "LocUbBufLite ABI");
struct EpMemTableInitOutput {
    uint64_t tablePtr{0};
    uint32_t magic{0};
    uint32_t version{0};
};
static_assert(sizeof(EpMemTableInitOutput) == 16, "EpMemTableInitOutput ABI");
struct HcclAicpuEpMemTableInitRes {
    uint64_t outAddr{0};
};
static_assert(sizeof(HcclAicpuEpMemTableInitRes) == 8, "HcclAicpuEpMemTableInitRes ABI");
struct HcclAicpuEpMemTableUpdateRes {
    uint64_t tablePtr{0};
    uint32_t op{0};
    uint32_t rsv{0};
    uint64_t addr{0};
    uint64_t size{0};
    uint32_t tokenId{0};
    uint32_t tokenValue{0};
};
static_assert(sizeof(HcclAicpuEpMemTableUpdateRes) == 40, "HcclAicpuEpMemTableUpdateRes ABI");
static_assert(offsetof(HcclAicpuEpMemTableUpdateRes, op) == 8, "HcclAicpuEpMemTableUpdateRes ABI");
static_assert(offsetof(HcclAicpuEpMemTableUpdateRes, addr) == 16, "HcclAicpuEpMemTableUpdateRes ABI");
struct HcclAicpuEpMemTableDestroyRes {
    uint64_t tablePtr{0};
};
static_assert(sizeof(HcclAicpuEpMemTableDestroyRes) == 8, "HcclAicpuEpMemTableDestroyRes ABI");

#endif
