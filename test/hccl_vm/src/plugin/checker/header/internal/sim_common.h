/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_COMMON_H
#define SIM_COMMON_H
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;

using s8 = signed char;
using s16 = signed short;
using s32 = signed int;
using s64 = signed long long;

using RankId = uint32_t;
using PodId = uint32_t;
using SerId = uint32_t;
using PhyId = uint32_t;
// 物理设备 ID：全局唯一，内部逻辑统一使用 deviceId 标识设备（为多通信域执行做准备）
using DeviceId = uint32_t;
using CommId = uint64_t;
constexpr DeviceId INVALID_DEVICE_ID = std::numeric_limits<DeviceId>::max();
constexpr CommId INVALID_COMM_ID = std::numeric_limits<CommId>::max();
constexpr uint64_t SIM_MEM_BLOCK_SIZE = 0x10000000000; // 内存分配起始地址
constexpr uint64_t SIM_MEM_MASKER = 0xFFFFFF0000000000;
constexpr uint64_t SIZE_200MB = 200ULL * 1024 * 1024; // CCL 200MB
constexpr uint32_t MAX_STREAM_COUNT = 40;
constexpr uint32_t MAX_NOTIFY_COUNT = 64;

#define MAX_DEV_NUM 1024

using ChannelHandle = uint64_t;
using ThreadHandle = uint64_t;
// 通信域拓扑元数据(与include/sim_common_defs.h保持同一定义):
// 4层结构 superPodId → serverId(POD) → boardId → 叶子节点内设备物理id列表。
// 叶子节点(host节点) = server形态的server / POD形态的board(server形态只有boardId==0一个条目)
using PhyDeviceId = uint32_t;
using NodeMeta = std::vector<PhyDeviceId>;
using BoardMeta = std::map<uint32_t, NodeMeta>;
using ServerMeta = BoardMeta;
using SuperPodMeta = std::map<uint32_t, ServerMeta>;
using TopoMeta = std::map<uint32_t, SuperPodMeta>;

struct NpuPos {
    PodId superpodId; // 超节点Id
    SerId serverId;   // ServerId(POD形态下为POD id)
    PodId boardId;    // BoardId(server形态恒为0)
    PhyId phyId;      // 叶子节点(server/board)内物理Id
};

enum BufferType { INPUT = 0, OUTPUT, CCL, SCRATCH, INPUT_AIV, OUTPUT_AIV, AIV_COMMINFO, USERBUF_AIV, MS, RESERVED };

struct MemBlock {
    BufferType bufferType;
    uint64_t startAddr;
    uint64_t size;
};

#endif
