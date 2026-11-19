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
#include "endpoint_pair/channel/ub_unique_id_pub.h" // UB_UNIQUE_ID_EXT_*：零依赖薄头，本头转发

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
};

// ===== UB 建链序列化（uniqueIdV2）扩展区契约（UB_UNIQUE_ID_EXT_*）已下沉 ub_unique_id_pub.h =====
// 零依赖薄头（仅 <cstdint>），供 legacy UT/host target 直接包含，避免本头的 hccl_common/
// transport_pub（910 pub_inc）依赖链与 950 头在部分 UT target 二义；本头经 include 转发保持
// 既有消费点符号可见，勿在本头重新定义。

// ===== DevEpMemTable（device 侧 endpoint 本地内存活账本表）kernel 参数与 ABI =====
// host 与 device 共用；条目类型 LocUbBufLite 已下沉本头（原落 legacy，检视整改消除双向耦合），
// Update 入参平铺其字段仅为 kernel ABI 稳定
constexpr uint32_t DEV_EP_MEM_TABLE_MAGIC_WORD = 0x4D455054; // 'MEPT'：防悬垂/错指针哨兵
constexpr uint32_t DEV_EP_MEM_TABLE_VERSION = 1; // 语义随同包升级变更：条目键改 (addr,size)+refcount，Update/Remove
                                                 // 语义调整（表与 endpoint 同进程同包，无跨版本混布）

enum class EpMemTableOp : uint32_t {
    UPSERT = 0, // HcommMemReg 成功后：同键 ref++，新键插入（ref=1）
    REMOVE = 1, // HcommMemUnreg 成功后：ref--，归零删条目（与 host 账本严格同步）
};

// endpoint 本地内存表条目：DevEpMemTable（device 侧活账本）/ 建链序列化解析 / update kernel 共用；
// 4 个标量字段零依赖；DFX 文案由使用点内联 StringFormat（pub 头不引 legacy string_util）。
// 归属选择：与同段 ABI 类型一致取全局作用域（检视整改统一命名空间归属，勿引入 namespace 混用）
struct LocUbBufLite {
    uint64_t addr{};
    uint64_t size{};
    uint32_t tokenId{};
    uint32_t tokenValue{};
};
static_assert(sizeof(LocUbBufLite) == 24, "LocUbBufLite ABI");

// RunAicpuEpMemTableInit 输出槽（host D2H 读回并校验，仿 ctx 读回校验模式）
struct EpMemTableInitOutput {
    uint64_t tablePtr{0};
    uint32_t magic{0};
    uint32_t version{0};
};
static_assert(sizeof(EpMemTableInitOutput) == 16, "EpMemTableInitOutput ABI");

// RunAicpuEpMemTableInit 入参
struct HcclAicpuEpMemTableInitRes {
    uint64_t outAddr{0}; // 输出槽 device 地址
};
static_assert(sizeof(HcclAicpuEpMemTableInitRes) == 8, "HcclAicpuEpMemTableInitRes ABI");

// RunAicpuEpMemTableUpdate 入参：单条
struct HcclAicpuEpMemTableUpdateRes {
    uint64_t tablePtr{0};
    uint32_t op{0}; // EpMemTableOp
    uint32_t rsv{0};
    uint64_t addr{0};
    uint64_t size{0};
    uint32_t tokenId{0};
    uint32_t tokenValue{0};
};
static_assert(sizeof(HcclAicpuEpMemTableUpdateRes) == 40, "HcclAicpuEpMemTableUpdateRes ABI");
static_assert(offsetof(HcclAicpuEpMemTableUpdateRes, op) == 8, "HcclAicpuEpMemTableUpdateRes ABI");
static_assert(offsetof(HcclAicpuEpMemTableUpdateRes, addr) == 16, "HcclAicpuEpMemTableUpdateRes ABI");

// RunAicpuEpMemTableDestroy 入参
struct HcclAicpuEpMemTableDestroyRes {
    uint64_t tablePtr{0};
};
static_assert(sizeof(HcclAicpuEpMemTableDestroyRes) == 8, "HcclAicpuEpMemTableDestroyRes ABI");

#endif
