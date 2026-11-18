/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file stuck_detect_counter_c_adpt.cc
 * @brief 卡住检测计数地址 C 接口适配。
 */

#include "hccl/hccl_res.h"

#include "cluster_monitor.h"
#include "coll_comm.h"
#include "coll_comm_mgr.h"
#include "hccl_communicator.h"
#include "log.h"
#include "stuck_detect_counter_types.h"

namespace {

HcclResult GetStuckDetectCounterAddrInternal(
    int32_t deviceLogicId, const std::string& commId, uint64_t* headOpCounterAddr, uint64_t* tailOpCounterAddr,
    uint64_t* addOneMemAddr)
{
    auto& monitor = hccl::CollCommMgr::GetInstance().GetClusterMonitor(deviceLogicId);
    hcomm::StuckDetectCounterAddress counterAddress;
    HcclResult ret = monitor.GetStuckDetectCounter(commId, counterAddress);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[%s] Get stuck detect counter failed, commId[%s], ret[%d].", __func__, commId.c_str(), ret);
        return ret;
    }

    *headOpCounterAddr = counterAddress.headAddr;
    *tailOpCounterAddr = counterAddress.tailAddr;
    *addOneMemAddr = counterAddress.addOneAddr;
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclStuckDetectGetCounterAddr(
    HcclComm comm, uint64_t* headOpCounterAddr, uint64_t* tailOpCounterAddr, uint64_t* addOneMemAddr)
{
    if (headOpCounterAddr == nullptr || tailOpCounterAddr == nullptr || addOneMemAddr == nullptr) {
        HCCL_WARNING("[%s] output pointer is null.", __func__);
        return HCCL_E_PTR;
    }

    *headOpCounterAddr = 0;
    *tailOpCounterAddr = 0;
    *addOneMemAddr = 0;

    if (comm == nullptr) {
        HCCL_WARNING("[%s] comm is null.", __func__);
        return HCCL_E_PTR;
    }
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_WARNING("[%s] only supports communicator V2.", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    if (collComm == nullptr) {
        HCCL_WARNING("[%s] collComm is null.", __func__);
        return HCCL_E_PTR;
    }

    const s32 deviceLogicId = collComm->GetDeviceLogicId();
    const std::string& commId = collComm->GetCommId();
    HCCL_INFO("Entry-[%s]:comm[%s], deviceLogicId[%d]", __func__, commId.c_str(), deviceLogicId);
    if (deviceLogicId < 0 || commId.empty()) {
        HCCL_WARNING("[%s] invalid comm, deviceLogicId[%d], commId[%s].", __func__, deviceLogicId, commId.c_str());
        return HCCL_E_INTERNAL;
    }

    HcclResult ret = HCCL_SUCCESS;
    EXCEPTION_CATCH(
        ret
        = GetStuckDetectCounterAddrInternal(deviceLogicId, commId, headOpCounterAddr, tailOpCounterAddr, addOneMemAddr),
        ret = HCCL_E_INTERNAL);

    if (ret != HCCL_SUCCESS) {
        *headOpCounterAddr = 0;
        *tailOpCounterAddr = 0;
        *addOneMemAddr = 0;
        HCCL_WARNING("[%s] get stuck detect counter addr failed, commId[%s], ret[%d].", __func__, commId.c_str(), ret);
        return ret;
    }

    HCCL_INFO(
        "[%s]:comm[%s] get stuck detect counter addr success, head[0x%llx], tail[0x%llx].", __func__, commId.c_str(),
        static_cast<unsigned long long>(*headOpCounterAddr), static_cast<unsigned long long>(*tailOpCounterAddr));
    return HCCL_SUCCESS;
}
