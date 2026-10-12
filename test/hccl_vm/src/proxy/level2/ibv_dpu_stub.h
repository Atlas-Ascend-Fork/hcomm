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

struct DpuQpInfo {
    uint64_t dbQpId = 0;
    uint32_t qpNum = 0;
    uint32_t localDeviceId = INVALID_DEVICE_ID;
    uint32_t remoteDeviceId = INVALID_DEVICE_ID;
    uint8_t rmEid[16] = {};
    ibv_cq* sendCq = nullptr;
    ibv_cq* recvCq = nullptr;
    bool isDpu = false;
};

struct IbvDpuContext {
    std::unordered_map<ibv_qp*, DpuQpInfo> qpInfo;
    std::unordered_map<ibv_cq*, ibv_qp*> sendCqToQp;
    std::unordered_map<ibv_cq*, ibv_qp*> recvCqToQp;
    std::unordered_map<ibv_cq*, int> pendingWqeCount;
    std::unordered_set<uint64_t> hybridNotifyAddrSet;
    std::mutex mtx;
};

extern IbvDpuContext g_dpuCtx;

// 向 g_dpuCtx 注册一个 fake QP 的映射关系（qpInfo / sendCqToQp / recvCqToQp / pendingWqeCount）。
void RegisterDpuQp(
    ibv_qp* fakeQp, uint64_t dbQpId, uint32_t qpNum, uint32_t localDeviceId, ibv_cq* sendCq, ibv_cq* recvCq);

#endif
