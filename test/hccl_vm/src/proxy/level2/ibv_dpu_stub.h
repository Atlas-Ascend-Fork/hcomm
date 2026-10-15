/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IBV_DPU_STUB_H
#define IBV_DPU_STUB_H

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <infiniband/verbs.h>

inline constexpr uint32_t INVALID_DEVICE_ID = 0xFFFFFFFF;
inline constexpr uint32_t DPU_NOTIFY_NUM_DEFAULT = 3;
inline constexpr uint32_t DPU_NOTIFY_NUM_MAX = 64;

struct DpuQpInfo {
    uint64_t dbQpId = 0;
    uint32_t qpNum = 0;
    uint32_t localDeviceId = INVALID_DEVICE_ID;
    uint32_t remoteDeviceId = INVALID_DEVICE_ID;
    uint8_t rmEid[16] = {};
    ibv_cq* sendCq = nullptr;
    ibv_cq* recvCq = nullptr;
    bool isDpu = false;
    // 评估模式新增
    uint32_t peerRankId = UINT32_MAX;
    uint8_t lcEid[16] = {};
    bool bindAttempted = false; // 防止重复绑定
};

struct IbvDpuContext {
    std::unordered_map<ibv_qp*, DpuQpInfo> qpInfo;
    std::unordered_map<ibv_cq*, ibv_qp*> sendCqToQp;
    std::unordered_map<ibv_cq*, ibv_qp*> recvCqToQp;
    std::unordered_map<ibv_cq*, int> pendingWqeCount;
    std::unordered_set<uint64_t> hybridNotifyAddrSet;
    // 评估模式新增: channel句柄(ChannelHandle,uint64_t)->(槽位 idx -> (imm, 最近post的notifyId))
    std::unordered_map<uint64_t, std::unordered_map<uint32_t, std::pair<uint32_t, uint64_t>>> channelSlots;
    // 评估模式miss去重: 已报过的 (qpn<<32|idx), 防未拦截wait入口busy-loop刷爆日志
    std::unordered_set<uint64_t> reportedMiss;
    // 评估模式通道创建计数: DpuNotifyManager位图按创建顺序连续分配, 推断imm用
    uint32_t channelCreateSeq = 0;
    std::mutex mtx;
};

extern IbvDpuContext g_dpuCtx;

// 向 g_dpuCtx 注册一个 fake QP 的映射关系（qpInfo / sendCqToQp / recvCqToQp / pendingWqeCount）。
void RegisterDpuQp(
    ibv_qp* fakeQp, uint64_t dbQpId, uint32_t qpNum, uint32_t localDeviceId, ibv_cq* sendCq, ibv_cq* recvCq);

// 评估模式时序绑定: RaGetQpAttr记录QP, 同线程下一次socket send时绑定真实对端
void DpuRecordPendingBindQp(struct ibv_qp* fakeQp);
void BindDpuPendingQps(uint32_t peerDeviceId, const uint8_t* peerEid, const uint8_t* localEid);

// 评估模式劫持Notify上下文: 记录Notify原语的(channel句柄, idx), 供ibv层建立槽位映射
void DpuSetNotifyRecordContext(uint64_t channel, uint32_t remoteNotifyIdx);
void DpuSetNotifyWaitContext(uint64_t channel, uint32_t localNotifyIdx);
void DpuClearNotifyContext(void);

// 评估模式channel创建预填: 按创建序号×notifyNum推断该通道全部idx的期望imm
// (DpuNotifyManager位图连续分配 + RoCE通道独占, post时以实际imm校准覆盖)
void DpuOnChannelCreatedEstimate(uint64_t channel, uint32_t notifyNum);

#endif
