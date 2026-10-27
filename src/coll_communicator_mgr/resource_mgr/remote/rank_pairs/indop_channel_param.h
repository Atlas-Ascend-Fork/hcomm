/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDOP_CHANNEL_PARAM_H
#define INDOP_CHANNEL_PARAM_H

#include <climits>
#include "endpoint_pair/channel/channel_param_pub.h"
#include "aicpu_operator_pub.h"

// base_comm 内部头 resources/endpoint_pairs/channels/channel_param.h 按其自身用途另定义了字段不同的
// HcommChannelP2p（仅 qos/transportAttr/远端内存）；两者用途不同、各自定义，互不锁定。
constexpr u64 P2P_NOTIFY_MAX_NUM = 66;
struct HcclChannelP2p {
    HcclMem* remoteUserMem = nullptr;                    // 远端用户内存
    HcclMem remoteHcclbuffer{};                          // 远端用户cclbuffer
    u32 remoteUserMemCount = 0;                          // 远端用户内存数量
    HcclSignalInfo localIpcSignal[P2P_NOTIFY_MAX_NUM]{}; // localnotify
    HcclSignalInfo remoteIpcSignal[P2P_NOTIFY_MAX_NUM]{};
    hccl::TransportAttr transportAttr;
    u32 qos = 0;
};

// Roce 建链资源（仅通信域管理层 coll_communicator_mgr 使用，base_comm 不依赖）
struct HcclChannelRoce {
    // 将固定数组改为指针
    MemDetails* remoteUserHostMem = nullptr;   // 远端用户主机内存
    MemDetails* remoteUserDeviceMem = nullptr; // 远端用户设备内存
    MemDetails localHcclbuffer;                // 本端用户cclbuffer
    MemDetails remoteHcclbuffer;               // 远端用户cclbuffer
    u32 remoteUserHostMemCount = 0;            // 远端用户主机内存数量
    u32 remoteUserDeviceMemCount = 0;          // 远端用户设备内存数量
    u64 notifyValue{0};
    u32 notifyValueKey{0};
    u32 singleQPNotifyNum{0};
    u64 localNotifyList{0};
    u64 remoteNotifyList{0};
    u32 remoteNotifyKey{0};
    s64 chipId{LLONG_MAX};
    HcclQpInfoV2 QpInfo[RDMA_QP_MAX_NUM];
    u32 qpsPerConnection{1};
};

struct HcclIndOpChannelRemoteResV2 {
    u64 p2pNotifyNum = 0;        // 用于linkp2p添加notify信息
    u64 roceNotifyNum = 0;       // 主链路：用于linkroce添加notify信息
    u64 qpNum = 0;               // 主链路：QP计数，支持多个，用于linkroce添加QP信息
    u32 remoteRank = 0;          // 远端rankId
    u32 remoteWorldRank = 0;     // 远端rankWorldId
    bool isUsedRdma = false;     // 是否使用RDMA，对应Roce和P2p
    HcclChannelP2p channelP2p;   // P2p资源
    HcclChannelRoce channelRoce; // Roce资源
};

struct HcclIndOpChannelRemoteResV3 {
    char hcomId[HCOMID_MAX_LENGTH]{};                   // 通信域ID 最大长度
    char channelTag[TAG_MAX_LENGTH]{};                  // channelTag 最大长度
    CommEngine engine{};                                // 通信引擎类型
    u32 localUserRank{0};                               // 本地rankId
    u32 multiQpThreshold{0};                            // 多QP每个QP分担数据量最小阈值
    void* channelList{nullptr};                         // device侧channelList地址
    u32 listNum = 0;                                    // 建链channel的总数量
    HcclIndOpChannelRemoteResV2* remoteResV2 = nullptr; // 不同remoteRank建链的资源
};

#endif
