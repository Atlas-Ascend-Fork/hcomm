/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHANNEL_PARAM_H
#define CHANNEL_PARAM_H

#include "endpoint_pair/channel/channel_param_pub.h"

// base_comm 内部 P2p 通道资源：仅保留 host 填充、device 消费所需的字段（qos/transportAttr/远端内存）。
// L3/L2 用途不同，L2 侧的 notify 信号数组等字段见
// coll_communicator_mgr/resource_mgr/remote/rank_pairs/indop_channel_param.h 的 HcclChannelP2p。
struct HcommChannelP2p {
    HcclMem* remoteUserMem = nullptr; // 远端用户内存
    HcclMem remoteHcclbuffer{};       // 远端用户cclbuffer
    u32 remoteUserMemCount = 0;       // 远端用户内存数量
    hccl::TransportAttr transportAttr;
    u32 qos = 0;
};

struct HcommRoceChannelRes {
    void* localMem = nullptr;  // device 上 RoceMemDetails[localMemCount]
    void* remoteMem = nullptr; // device 上 RoceMemDetails[remoteMemCount]
    u32 localMemCount = 0;
    u32 remoteMemCount = 0;
    s64 chipId{LLONG_MAX};
    HcclQpInfoV2 QpInfo[RDMA_QP_MAX_NUM];
    u32 qpsPerConnection{1};
    void* remoteNotifyAddr = nullptr;    // 对端 NOTIFY_SRC_MEM 地址 (Fence Read 读取源)
    u32 remoteNotifyKey = 0;             // 对端 notify rkey
    void* localDataNotifyAddr = nullptr; // 本端datanotify地址
    u32 localDataNotifyKey = 0;          // 本端dataNotify key值
    u32 notifySize = sizeof(u32);        // notify 大小
    HcclSignalInfo localDataSignal{};    // 本端dataNotify signal
};

// for A2/A3 endpoint with transport
struct HcclChannelHccsRes {
    char channelTag[TAG_MAX_LENGTH]{}; // channelTag 最大长度
    HcommChannelP2p channelP2p;        // P2p资源
    u64 p2pNotifyNum{0};               // 用于linkp2p添加notify信息
    u32 deviceType{0};                 // 基础通信使用
    u32 remoteDevicePhyId{0};          // 远端 PhyId
    u32 localDevicePhyId{0};
    s32 localDeviceLogicId{0};
    hccl::MachineType machineType{hccl::MachineType::MACHINE_RESERVED_TYPE}; // client或者server
    u32 localBufSize{0};
    u32 remoteBufSize{0};
    HcclMemEx* localBufMem{nullptr};
    HcclMemEx* remoteBufMem{nullptr};
};

struct HcommDeviceInfo {
    s32 deviceLogicId{0};
    u32 devicePhyId{0};
    u32 deviceType{0};
};

struct HcommChannelRes {
    void* channelList{nullptr};         // 反序列后返回给host侧的device侧handle地址
    u32 listNum = 0;                    // 建链channel的总数量
    void* channelDataListAddr{nullptr}; // device 上 listNum 个指针，每项指向该 channel 的序列化 device 内存
    void* channelDataSizeListAddr{nullptr}; // device 上 listNum 个 u32，每项为对应 channel 序列化字节数
    void* channelTypeListAddr{nullptr}; // device 上 listNum 个 u32，每项为 hcomm::HcommChannelKind 数值（见 channel.h）
    HcommDeviceInfo deviceInfo;
    void* ctxList{nullptr}; // device侧ctx指针数组，非null时kernel写ctx->deviceChannel
};

#endif
