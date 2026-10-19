/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "dev_aicpu_ts_roce_dev_loc_channel.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "channel_param.h"
#include "adapter_hal_pub.h"
#include "adapter_rts_common.h"
#include "log.h"
#include "transport_pub.h"

using namespace hccl;

DevAicpuTsRoceDevLocChannel::~DevAicpuTsRoceDevLocChannel()
{
    for (auto& pair : slots_) {
        if (pair.second.link != nullptr) {
            (void)pair.second.link->DeInit();
            pair.second.link.reset();
        }
    }
    slots_.clear();
}

namespace {

constexpr u32 RDMA_QP_MAX_NUM = 32U;
constexpr u32 HCCL_MULTI_QP_THRESHOLD_DEFAULT = 0U;

constexpr u32 DEFAULT_TIMEOUT_MS = 10000;

std::atomic<u64> g_roceTagSeq{0};

HcclResult FillIbverbsDataFromRes(const HcommRoceChannelRes* res, TransportDeviceIbverbsData& ibd)
{
    std::vector<RoceMemDetails> localMd;
    auto localBase = static_cast<const RoceMemDetails*>(res->localMem);
    if (localBase != nullptr) {
        for (u32 i = 0; i < res->localMemCount; ++i) {
            localMd.push_back(localBase[i]);
        }
    }
    std::vector<RoceMemDetails> remoteMd;
    auto remoteBase = static_cast<const RoceMemDetails*>(res->remoteMem);
    if (remoteBase != nullptr) {
        for (u32 i = 0; i < res->remoteMemCount; ++i) {
            remoteMd.push_back(remoteBase[i]);
        }
    }
    HCCL_INFO(
        "[DevAicpuTsRoceDevLocChannel][Create] Roce mem from channel res: localMemCount[%u] remoteMemCount[%u] "
        "parsed local[%zu] remote[%zu]",
        res->localMemCount, res->remoteMemCount, localMd.size(), remoteMd.size());

    const u32 qpInfoSize = res->qpsPerConnection + static_cast<u32>(res->qpsPerConnection != 1U);
    if (qpInfoSize < 1U || qpInfoSize > RDMA_QP_MAX_NUM) {
        HCCL_ERROR("[DevAicpuTsRoceDevLocChannel][Create] bad qp layout qpsPerConn[%u]", res->qpsPerConnection);
        return HCCL_E_PARA;
    }
    std::vector<HcclQpInfoV2> qpVec(qpInfoSize);
    for (u32 i = 0; i < qpInfoSize; ++i) {
        qpVec[i] = res->QpInfo[i];
    }

    ibd.qpInfo = std::move(qpVec);
    ibd.qpsPerConnection = res->qpsPerConnection;
    ibd.multiQpThreshold = HCCL_MULTI_QP_THRESHOLD_DEFAULT;
    ibd.localRoceMemDetailsList = std::move(localMd);
    ibd.remoteRoceMemDetailsList = std::move(remoteMd);
    ibd.useMemDetailsMgr = true;
    ibd.remoteNotifyValueAddr = ReinterpretAs<uint64_t>(res->remoteNotifyAddr);
    ibd.remoteNotifyValueKey = res->remoteNotifyKey;
    ibd.localDataNotifyAddr = ReinterpretAs<uint64_t>(res->localDataNotifyAddr);
    ibd.localDataNotifyKey = res->localDataNotifyKey;
    ibd.notifySize = res->notifySize;
    HCCL_DEBUG(
        "[%s]remoteNotifyAddr[%llu], remoteNotifyKey[%u], localDataNotifyAddr[%llu], localDataNotifyKey[%u],"
        "notifySize[%u]",
        __func__, ibd.remoteNotifyValueAddr, ibd.remoteNotifyValueKey, ibd.localDataNotifyAddr, ibd.localDataNotifyKey,
        ibd.notifySize);

    EXCEPTION_CATCH((ibd.dataNotify = std::make_shared<LocalNotify>()), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(ibd.dataNotify);
    CHK_RET(ibd.dataNotify->Init(res->localDataSignal, NotifyLoadType::DEVICE_NOTIFY));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult DevAicpuTsRoceDevLocChannel::Create(
    const void* blob, u64 blobBytes, const HcommDeviceInfo& deviceInfo, ChannelHandle& outHandle)
{
    CHK_PTR_NULL(blob);
    if (blobBytes < sizeof(HcommRoceChannelRes)) {
        HCCL_ERROR("[DevAicpuTsRoceDevLocChannel][Create] blob too small[%llu]", static_cast<unsigned long long>(blobBytes));
        return HCCL_E_PARA;
    }
    const auto* res = static_cast<const HcommRoceChannelRes*>(blob);

    TransportDeviceIbverbsData ibd{};
    CHK_RET(FillIbverbsDataFromRes(res, ibd));
    const u32 qpInfoSize = res->qpsPerConnection + static_cast<u32>(res->qpsPerConnection != 1U);

    u32 devId = INVALID_UINT;
    CHK_RET(hrtDrvGetLocalDevIDByHostDevID(deviceInfo.devicePhyId, &devId));

    MachinePara machinePara{};
    machinePara.deviceLogicId = deviceInfo.deviceLogicId;
    machinePara.localDeviceId = deviceInfo.devicePhyId;
    DevType devType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(devType));
    machinePara.deviceType = devType;
    machinePara.isAicpuModeEn = true;
    machinePara.isIndOp = true;
    machinePara.notifyNum = 0;
    machinePara.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    // tag仅用于日志诊断的唯一标识
    machinePara.tag = "hcomm_aicpu_ts_roce_" + std::to_string(deviceInfo.deviceLogicId) + "_"
                      + std::to_string(g_roceTagSeq.fetch_add(1));
    machinePara.userMemEnable = false;
    machinePara.drainEnable = true;
    // dispatcher归属Thread（由AicpuTsThread创建/销毁），channel不持有dispatcher（dctxPtr保持默认nullptr）；
    // RoCE blob无qos字段，hcclQos保持默认INVALID_UINT，GetOpDispatcher下发时经区间映射回落SDMA_QOS_DEFAULT

    TransportPara transportPara{};
    transportPara.timeout = std::chrono::milliseconds(DEFAULT_TIMEOUT_MS);
    transportPara.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    static const std::unique_ptr<NotifyPool> kEmptyNotifyPool;
    std::shared_ptr<Transport> link;
    // Transport构造不传dispatcher，任务下发时由执行线程通过GetOpDispatcher解析Thread私有dispatcher
    link.reset(new (std::nothrow) Transport(
        TransportType::TRANS_TYPE_DEVICE_IBVERBS, transportPara, nullptr, kEmptyNotifyPool, machinePara,
        TransportDeviceP2pData(), ibd));
    if (link == nullptr) {
        HCCL_ERROR("[DevAicpuTsRoceDevLocChannel][Create] Transport alloc failed");
        return HCCL_E_PTR;
    }
    HcclResult tr = link->Init();
    if (tr != HCCL_SUCCESS) {
        link.reset();
        return tr;
    }
    tr = link->InitDrainNotifyInfo();
    if (tr != HCCL_SUCCESS) {
        link.reset();
        return tr;
    }

    outHandle = ReinterpretAs<ChannelHandle>(link.get());
    RoceSlot slot;
    slot.link = std::move(link);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        slots_.emplace(outHandle, std::move(slot));
    }

    HCCL_INFO(
        "[DevAicpuTsRoceDevLocChannel][Create] success logicId[%d] phyId[%u] devId[%u] blobBytes[%llu] "
        "localMem[%u] remoteMem[%u] qpsPerConn[%u] qpNum[%u] chipId[%lld] handle[0x%llx]",
        deviceInfo.deviceLogicId, deviceInfo.devicePhyId, devId, static_cast<unsigned long long>(blobBytes),
        res->localMemCount, res->remoteMemCount, res->qpsPerConnection, qpInfoSize, static_cast<long long>(res->chipId),
        static_cast<unsigned long long>(ReinterpretAs<uintptr_t>(outHandle)));
    return HCCL_SUCCESS;
}

bool DevAicpuTsRoceDevLocChannel::Destroy(ChannelHandle handle)
{
    RoceSlot slot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = slots_.find(handle);
        if (it == slots_.end()) {
            return false;
        }
        slot = std::move(it->second);
        slots_.erase(it);
    }
    if (slot.link != nullptr) {
        (void)slot.link->DeInit();
        slot.link.reset();
    }
    HCCL_DEBUG("[DevAicpuTsRoceDevLocChannel][Destroy] destroyed handle[0x%llx]", handle);
    return true;
}
