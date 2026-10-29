/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/cast_utils_pub.h"
#include "aiv_channel_entity.h"
#include "endpoint_pair/channel/channel_process_pub.h"
#include "channel.h"
#include "aiv_urma_channel.h"
#include "common/comm_engine_utils_pub.h"
#include "acl_device_slab_guard.h"
#include "adapter_rts.h"
#include "orion_adapter_rts.h"
#include <securec.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace hcomm;

namespace {
HcclResult SecureMemset(void* dest, size_t destMax, int value, size_t count, const char* fieldName)
{
    if (dest == nullptr) {
        HCCL_ERROR("[AivChannelEntity::%s] dest is nullptr, field[%s]", __func__, fieldName);
        return HCCL_E_PTR;
    }
    if (count > destMax) {
        HCCL_ERROR(
            "[AivChannelEntity::%s] invalid size, field[%s], count[%zu], destMax[%zu]", __func__, fieldName, count,
            destMax);
        return HCCL_E_PARA;
    }

    errno_t ret = memset_s(dest, destMax, value, count);
    if (ret != EOK) {
        HCCL_ERROR(
            "[AivChannelEntity::%s] memset_s failed, field[%s], ret[%d], count[%zu], destMax[%zu]", __func__, fieldName,
            ret, count, destMax);
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}
} // namespace

AivChannelEntity::AivChannelEntity(hcomm::AivUrmaChannel& owner) : owner_(owner) {}

HcclResult AivChannelEntity::BuildChannelEntityToDevice(void** devChannelPtr)
{
    if (devChannelPtr == nullptr) {
        HCCL_ERROR("[AivChannelEntity::%s] devChannelPtr is nullptr", __func__);
        return HCCL_E_PTR;
    }

    CHK_PTR_NULL(owner_.transport_.get());

    ChannelEntity hostChannel;
    CHK_RET(SecureMemset(&hostChannel, sizeof(ChannelEntity), 0, sizeof(ChannelEntity), "hostChannel"));

    CHK_RET(owner_.transport_->PrepareHostChannelEntity(&hostChannel));

    DeviceChannelEntityLayout layout;
    CHK_RET(BuildDeviceChannelEntityLayout(hostChannel, layout, true));
    void* slabPtr = nullptr;
    CHK_RET(AllocDeviceEntitySlab(layout.slabSize, slabPtr));
    AclDeviceSlabGuard slabGuard;
    slabGuard.Reset(slabPtr, layout.slabSize);
    uint32_t queueNum = std::max(hostChannel.sqNum, hostChannel.cqNum);
    if (owner_.IsSharedQueue() && owner_.sharedSqPiPtr_ != nullptr) {
        owner_.transport_->SetQueueIndexDeviceMem(
            owner_.sharedSqPiPtr_, owner_.sharedSqCiPtr_, owner_.sharedCqPiPtr_, owner_.sharedCqCiPtr_,
            queueNum * QUEUE_INDEX_MEM_UNIT_SIZE);
    } else {
        CHK_RET(InitQueueIndexSections(slabPtr, layout, queueNum));
        SetQueueIndexDeviceMem(slabPtr, layout, queueNum);
    }

    CHK_RET(SecureMemset(&hostChannel, sizeof(ChannelEntity), 0, sizeof(ChannelEntity), "hostChannel"));
    CHK_RET(owner_.transport_->GetHostChannelEntity(&hostChannel));
    hostChannel.abiHeader = owner_.channelDesc_.header;
    hostChannel.engine = COMM_ENGINE_AIV;
    hostChannel.protocol = owner_.channelDesc_.remoteEndpoint.protocol;

    ChannelEntity devChannel;
    CHK_RET(CopyChannelEntityArraysToSlab(slabPtr, hostChannel, layout, devChannel));
    void* entityDevPtr = nullptr;
    CHK_RET(CopyChannelEntityHeaderToSlab(slabPtr, layout, devChannel, entityDevPtr));
    ReleaseDeviceEntitySlab();
    devChannelEntitySlab_ = slabGuard.Release();
    devChannelEntitySlabSize_ = layout.slabSize;
    devChannelEntity_ = entityDevPtr;
    *devChannelPtr = devChannelEntity_;
    HCCL_INFO(
        "[AivChannelEntity::%s] Build channel entity to device success, devPtr[%p], slabPtr[%p], slabSize[%zu]",
        __func__, devChannelEntity_, devChannelEntitySlab_, devChannelEntitySlabSize_);
    return HCCL_SUCCESS;
}

HcclResult AivChannelEntity::PreAllocChannelEntityToDevice(void** devChannelPtr)
{
    if (devChannelPtr == nullptr) {
        HCCL_ERROR("[AivChannelEntity::%s] devChannelPtr is nullptr", __func__);
        return HCCL_E_PTR;
    }
    CHK_PTR_NULL(owner_.transport_.get());

    if (devChannelEntitySlab_ != nullptr) {
        *devChannelPtr = devChannelEntity_;
        HCCL_INFO("[AivChannelEntity::%s] already built, return cached devPtr[%p]", __func__, devChannelEntity_);
        return HCCL_SUCCESS;
    }

    uint32_t bufNum = 0;
    uint32_t connNum = 0;
    owner_.transport_->GetEntityCountsForLayout(bufNum, connNum);

    ChannelEntity tmp{};
    tmp.localBufferNum = bufNum;
    tmp.remoteBufferNum = bufNum;
    tmp.sqNum = connNum;
    tmp.cqNum = connNum;

    DeviceChannelEntityLayout layout;
    CHK_RET(BuildDeviceChannelEntityLayout(tmp, layout, true));

    void* slabPtr = nullptr;
    CHK_RET(AllocDeviceEntitySlab(layout.slabSize, slabPtr));
    AclDeviceSlabGuard slabGuard;
    slabGuard.Reset(slabPtr, layout.slabSize);

    uint32_t queueNum = std::max(tmp.sqNum, tmp.cqNum);
    if (owner_.IsSharedQueue() && owner_.sharedSqPiPtr_ != nullptr) {
        owner_.transport_->SetQueueIndexDeviceMem(
            owner_.sharedSqPiPtr_, owner_.sharedSqCiPtr_, owner_.sharedCqPiPtr_, owner_.sharedCqCiPtr_,
            queueNum * QUEUE_INDEX_MEM_UNIT_SIZE);
    } else {
        CHK_RET(InitQueueIndexSections(slabPtr, layout, queueNum));
    }

    devChannelEntitySlab_ = slabGuard.Release();
    devChannelEntitySlabSize_ = layout.slabSize;
    devChannelEntity_ = GetSlabPtr(devChannelEntitySlab_, layout.entitySection);
    if (!owner_.IsSharedQueue() || owner_.sharedSqPiPtr_ == nullptr) {
        SetQueueIndexDeviceMem(devChannelEntitySlab_, layout, queueNum);
    }
    *devChannelPtr = devChannelEntity_;

    HCCL_INFO(
        "[AivChannelEntity::%s] pre-alloc success, devPtr[%p], slabPtr[%p], slabSize[%zu]", __func__, devChannelEntity_,
        devChannelEntitySlab_, devChannelEntitySlabSize_);
    return HCCL_SUCCESS;
}

HcclResult AivChannelEntity::FillChannelEntityToDevice()
{
    if (devChannelEntitySlab_ == nullptr) {
        HCCL_ERROR("[AivChannelEntity::%s] devChannelEntitySlab_ is nullptr, not pre-allocated.", __func__);
        return HCCL_E_INTERNAL;
    }
    CHK_PTR_NULL(owner_.transport_.get());

    ChannelEntity hostChannel;
    CHK_RET(SecureMemset(&hostChannel, sizeof(ChannelEntity), 0, sizeof(ChannelEntity), "hostChannel"));
    CHK_RET(owner_.transport_->GetHostChannelEntity(&hostChannel));
    hostChannel.abiHeader = owner_.channelDesc_.header;
    hostChannel.engine = COMM_ENGINE_AIV;
    hostChannel.protocol = owner_.channelDesc_.remoteEndpoint.protocol;

    DeviceChannelEntityLayout layout;
    CHK_RET(BuildDeviceChannelEntityLayout(hostChannel, layout, true));
    CHK_RET(CheckSlabCapacity(layout));

    ChannelEntity devChannel;
    CHK_RET(CopyChannelEntityArraysToSlab(devChannelEntitySlab_, hostChannel, layout, devChannel));
    void* entityDevPtr = nullptr;
    CHK_RET(CopyChannelEntityHeaderToSlab(devChannelEntitySlab_, layout, devChannel, entityDevPtr));

    devChannelEntity_ = entityDevPtr;
    HCCL_INFO("[AivChannelEntity::%s] fill success, devPtr[%p]", __func__, devChannelEntity_);
    return HCCL_SUCCESS;
}

HcclResult AivChannelEntity::HandleStatus(
    const ChannelHandle* channelList, uint32_t listNum, const HcommChannelDesc* channelDescs,
    const std::vector<int32_t>& linkStatusList, int32_t* statusList)
{
    HcclResult fillRet = FillDevEntities(channelList, listNum, channelDescs, linkStatusList.data());
    if (fillRet != HCCL_SUCCESS) {
        HCCL_ERROR("[AivChannelEntity::%s] FillDevEntities failed, ret[%d]", __func__, fillRet);
        return HCCL_E_INTERNAL;
    }
    for (uint32_t i = 0; i < listNum; i++) {
        statusList[i] = linkStatusList[i];
    }
    return HCCL_SUCCESS;
}

HcclResult AivChannelEntity::PreAllocChannels(
    ChannelHandle* targetChannels, ChannelHandle* userChannels, const HcommChannelDesc* channelDescs,
    uint32_t channelNum)
{
    CHK_PTR_NULL(targetChannels);
    CHK_PTR_NULL(userChannels);
    CHK_PRT_RET(
        (channelNum == 0),
        HCCL_ERROR("[AivChannelEntity::%s] Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);

    bool needD2HMap = false;
    for (uint32_t i = 0; i < channelNum; i++) {
        CommProtocol protocol = channelDescs[i].remoteEndpoint.protocol;

        if (protocol == COMM_PROTOCOL_ROCE || protocol == COMM_PROTOCOL_UB_CTP || protocol == COMM_PROTOCOL_UBC_TP
            || protocol == COMM_PROTOCOL_UB_RTP) {
            needD2HMap = true;
            auto* channel = ReinterpretAs<Channel*>(targetChannels[i]);
            CHK_PTR_NULL(channel);
            auto* entity = channel->GetChannelEntity();
            CHK_PTR_NULL(entity);

            void* devChannelEntity = nullptr;
            HcclResult ret = entity->PreAllocChannelEntityToDevice(&devChannelEntity);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[AivChannelEntity::%s] channel[%u] PreAllocChannelEntityToDevice failed, ret[%d]", __func__, i,
                    ret),
                ret);
            CHK_PTR_NULL(devChannelEntity);
            userChannels[i] = static_cast<ChannelHandle>(ReinterpretAs<uintptr_t>(devChannelEntity));
            HCCL_INFO(
                "[AivChannelEntity::%s] channel[%u] pre-alloc dev entity success, devEntityPtr[%p]", __func__, i,
                ReinterpretAs<void*>(static_cast<uintptr_t>(userChannels[i])));
        } else {
            userChannels[i] = targetChannels[i];
            HCCL_INFO(
                "[AivChannelEntity::%s] AIV engine channel protocol not supported pre-alloc, idx[%u], protocol[%d]. "
                "Return host channel handle.",
                __func__, i, static_cast<int>(protocol));
        }
    }

    if (needD2HMap) {
        CHK_RET(ChannelProcess::FillChannelD2HMap(userChannels, targetChannels, channelNum));
    }
    return HCCL_SUCCESS;
}

HcclResult
AivChannelEntity::InitQueueIndexSections(void* slabPtr, const DeviceChannelEntityLayout& layout, uint32_t queueNum)
{
    std::vector<uint8_t> zeroQueueIndexMem(QUEUE_INDEX_MEM_UNIT_SIZE * queueNum, 0);
    CHK_RET(ZeroQueueIndexSection(slabPtr, zeroQueueIndexMem, layout.sqPiSection));
    CHK_RET(ZeroQueueIndexSection(slabPtr, zeroQueueIndexMem, layout.sqCiSection));
    CHK_RET(ZeroQueueIndexSection(slabPtr, zeroQueueIndexMem, layout.cqPiSection));
    CHK_RET(ZeroQueueIndexSection(slabPtr, zeroQueueIndexMem, layout.cqCiSection));
    return HCCL_SUCCESS;
}

void AivChannelEntity::SetQueueIndexDeviceMem(void* slabPtr, const DeviceChannelEntityLayout& layout, uint32_t queueNum)
{
    owner_.transport_->SetQueueIndexDeviceMem(
        GetSlabPtr(slabPtr, layout.sqPiSection), GetSlabPtr(slabPtr, layout.sqCiSection),
        GetSlabPtr(slabPtr, layout.cqPiSection), GetSlabPtr(slabPtr, layout.cqCiSection),
        queueNum * QUEUE_INDEX_MEM_UNIT_SIZE);
}

HcclResult AivChannelEntity::ZeroQueueIndexSection(
    void* slabPtr, const std::vector<uint8_t>& zeroQueueIndexMem, const DeviceEntitySection& section)
{
    if (section.size == 0) {
        return HCCL_SUCCESS;
    }
    void* sectionPtr = GetSlabPtr(slabPtr, section);
    CHK_PTR_NULL(sectionPtr);
    Hccl::HrtMemcpy(
        sectionPtr, section.size, zeroQueueIndexMem.data(), section.size,
        Hccl::tagRtMemcpyKind::RT_MEMCPY_HOST_TO_DEVICE);
    return HCCL_SUCCESS;
}

HcclResult AivChannelEntity::FillDevEntities(
    const ChannelHandle* channelList, uint32_t listNum, const HcommChannelDesc* channelDescs,
    const int32_t* linkStatusList)
{
    CHK_PTR_NULL(channelList);
    CHK_PTR_NULL(linkStatusList);
    CHK_PRT_RET(
        (listNum == 0), HCCL_ERROR("[AivChannelEntity::%s] Invalid listNum, listNum[%u]", __func__, listNum),
        HCCL_E_PARA);

    for (uint32_t i = 0; i < listNum; i++) {
        if (linkStatusList[i] != hcomm::HCOMM_CHANNEL_STATUS_READY) {
            continue;
        }
        CommProtocol protocol = channelDescs[i].remoteEndpoint.protocol;
        void* channelPtr = nullptr;
        CHK_RET(ChannelProcess::ChannelGet(channelList[i], &channelPtr));
        CHK_PTR_NULL(channelPtr);
        auto* channel = static_cast<Channel*>(channelPtr);
        auto* entity = channel->GetChannelEntity();
        if (entity != nullptr && entity->IsDeviceEntityReady()) {
            continue;
        }

        if (protocol == COMM_PROTOCOL_ROCE || protocol == COMM_PROTOCOL_UB_CTP || protocol == COMM_PROTOCOL_UBC_TP
            || protocol == COMM_PROTOCOL_UB_RTP) {
            CHK_PRT_RET(
                entity == nullptr,
                HCCL_ERROR(
                    "[AivChannelEntity::%s] channel[%u] has no channel entity, protocol[%d]", __func__, i,
                    static_cast<int>(protocol)),
                HCCL_E_INTERNAL);
            CHK_RET(entity->FillChannelEntityToDevice());
        }

        if (entity != nullptr) {
            entity->SetDeviceEntityReady();
            HCCL_INFO("[AivChannelEntity::%s] channel[%u] fill dev entity success.", __func__, i);
        }
    }
    return HCCL_SUCCESS;
}
