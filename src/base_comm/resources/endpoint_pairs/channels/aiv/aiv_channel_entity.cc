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
#include "aiv_roce_host_loc_channel.h"
#include "common/comm_engine_utils_pub.h"
#include "acl_device_slab_guard.h"
#include "adapter_rts.h"
#include "orion_adapter_rts.h"
#include <securec.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace hcomm {

namespace {

    constexpr size_t DEVICE_ENTITY_ALIGN_SIZE = 64;

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
                "[AivChannelEntity::%s] memset_s failed, field[%s], ret[%d], count[%zu], destMax[%zu]", __func__,
                fieldName, ret, count, destMax);
            return HCCL_E_MEMORY;
        }
        return HCCL_SUCCESS;
    }

    size_t AlignUp(size_t value, size_t alignment) { return (value + alignment - 1) / alignment * alignment; }

    template <typename T>
    HcclResult CopyArrayToSlab(
        void* slabBase, const T* hostArray, uint32_t arrayNum, const DeviceEntitySection& section, T** deviceArrayPtr,
        const char* arrayName)
    {
        CHK_PTR_NULL(deviceArrayPtr);
        if (arrayNum == 0 || hostArray == nullptr) {
            CHK_PRT_RET(
                arrayNum != 0, HCCL_ERROR("[CopyArrayToSlab] %s hostArray is nullptr, num[%u]", arrayName, arrayNum),
                HCCL_E_PTR);
            *deviceArrayPtr = nullptr;
            return HCCL_SUCCESS;
        }
        CHK_PRT_RET(
            section.size != static_cast<size_t>(arrayNum) * sizeof(T),
            HCCL_ERROR(
                "[CopyArrayToSlab] %s size mismatch, sectionSize[%zu], expect[%zu]", arrayName, section.size,
                static_cast<size_t>(arrayNum) * sizeof(T)),
            HCCL_E_PARA);
        void* sectionPtr = (section.size == 0) ?
                               nullptr :
                               reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(slabBase) + section.offset);
        CHK_PTR_NULL(sectionPtr);
        Hccl::HrtMemcpy(
            sectionPtr, section.size, hostArray, section.size, Hccl::tagRtMemcpyKind::RT_MEMCPY_HOST_TO_DEVICE);
        *deviceArrayPtr = reinterpret_cast<T*>(sectionPtr);
        HCCL_INFO(
            "[CopyArrayToSlab] %s: host[%p] -> dev[%p], num[%u], size[%zu]", arrayName, hostArray, sectionPtr, arrayNum,
            section.size);
        return HCCL_SUCCESS;
    }

    HcclResult AddDeviceEntitySection(
        size_t elemSize, uint32_t elemNum, size_t& offset, DeviceEntitySection& section, const char* sectionName)
    {
        section.offset = AlignUp(offset, DEVICE_ENTITY_ALIGN_SIZE);
        if (elemNum == 0) {
            section.size = 0;
            offset = section.offset;
            return HCCL_SUCCESS;
        }
        CHK_PRT_RET(
            elemSize != 0 && elemNum > (SIZE_MAX / elemSize),
            HCCL_ERROR(
                "[AddDeviceEntitySection] %s size overflow, elemSize[%zu], elemNum[%u]", sectionName, elemSize,
                elemNum),
            HCCL_E_PARA);
        section.size = elemSize * static_cast<size_t>(elemNum);
        CHK_PRT_RET(
            section.offset > (SIZE_MAX - section.size),
            HCCL_ERROR(
                "[AddDeviceEntitySection] %s offset overflow, offset[%zu], size[%zu]", sectionName, section.offset,
                section.size),
            HCCL_E_PARA);
        offset = section.offset + section.size;
        return HCCL_SUCCESS;
    }

} // namespace

HcclResult AivEntitySlabBase::BuildDeviceChannelEntityLayout(
    const ChannelEntity& hostChannel, DeviceChannelEntityLayout& layout, bool withQueueIdx)
{
    layout.slabSize = AlignUp(sizeof(ChannelEntity), DEVICE_ENTITY_ALIGN_SIZE);
    CHK_RET(AddDeviceEntitySection(
        sizeof(RegedNotifyEntity), hostChannel.localNotifyNum, layout.slabSize, layout.localNotifySection,
        "localNotifyAddr"));
    CHK_RET(AddDeviceEntitySection(
        sizeof(RegedNotifyEntity), hostChannel.remoteNotifyNum, layout.slabSize, layout.remoteNotifySection,
        "remoteNotifyAddr"));
    CHK_RET(AddDeviceEntitySection(
        sizeof(RegedBufferEntity), hostChannel.localBufferNum, layout.slabSize, layout.localBufferSection,
        "localBufferAddr"));
    CHK_RET(AddDeviceEntitySection(
        sizeof(RegedBufferEntity), hostChannel.remoteBufferNum, layout.slabSize, layout.remoteBufferSection,
        "remoteBufferAddr"));
    CHK_RET(AddDeviceEntitySection(
        sizeof(SqContext), hostChannel.sqNum, layout.slabSize, layout.sqContextSection, "sqContextAddr"));
    CHK_RET(AddDeviceEntitySection(
        sizeof(CqContext), hostChannel.cqNum, layout.slabSize, layout.cqContextSection, "cqContextAddr"));
    if (withQueueIdx) {
        CHK_RET(AddDeviceEntitySection(
            QUEUE_INDEX_MEM_UNIT_SIZE, hostChannel.sqNum, layout.slabSize, layout.sqPiSection, "sqPiAddr"));
        CHK_RET(AddDeviceEntitySection(
            QUEUE_INDEX_MEM_UNIT_SIZE, hostChannel.sqNum, layout.slabSize, layout.sqCiSection, "sqCiAddr"));
        CHK_RET(AddDeviceEntitySection(
            QUEUE_INDEX_MEM_UNIT_SIZE, hostChannel.cqNum, layout.slabSize, layout.cqPiSection, "cqPiAddr"));
        CHK_RET(AddDeviceEntitySection(
            QUEUE_INDEX_MEM_UNIT_SIZE, hostChannel.cqNum, layout.slabSize, layout.cqCiSection, "cqCiAddr"));
    }
    layout.slabSize = AlignUp(layout.slabSize, DEVICE_ENTITY_ALIGN_SIZE);
    return HCCL_SUCCESS;
}

HcclResult AivEntitySlabBase::AllocDeviceEntitySlab(size_t slabSize, void*& slabPtr)
{
    HcclResult ret = hrtMalloc(&slabPtr, slabSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS || slabPtr == nullptr,
        HCCL_ERROR("[AivEntitySlabBase::%s] hrtMalloc slab failed, ret[%d], size[%zu]", __func__, ret, slabSize),
        HCCL_E_MEMORY);
    return HCCL_SUCCESS;
}

void* AivEntitySlabBase::GetSlabPtr(void* base, const DeviceEntitySection& section)
{
    if (section.size == 0) {
        return nullptr;
    }
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + section.offset);
}

HcclResult AivEntitySlabBase::CheckSlabCapacity(const DeviceChannelEntityLayout& layout) const
{
    if (layout.slabSize > devChannelEntitySlabSize_) {
        HCCL_ERROR(
            "[AivEntitySlabBase::%s] slabSize[%zu] > preAllocSize[%zu]", __func__, layout.slabSize,
            devChannelEntitySlabSize_);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult AivEntitySlabBase::CopyChannelEntityArraysToSlab(
    void* slabPtr, const ChannelEntity& hostChannel, const DeviceChannelEntityLayout& layout, ChannelEntity& devChannel)
{
    devChannel = hostChannel;
    CHK_RET(CopyArrayToSlab(
        slabPtr, hostChannel.localNotifyAddr, hostChannel.localNotifyNum, layout.localNotifySection,
        &devChannel.localNotifyAddr, "localNotifyAddr"));
    CHK_RET(CopyArrayToSlab(
        slabPtr, hostChannel.remoteNotifyAddr, hostChannel.remoteNotifyNum, layout.remoteNotifySection,
        &devChannel.remoteNotifyAddr, "remoteNotifyAddr"));
    CHK_RET(CopyArrayToSlab(
        slabPtr, hostChannel.localBufferAddr, hostChannel.localBufferNum, layout.localBufferSection,
        &devChannel.localBufferAddr, "localBufferAddr"));
    CHK_RET(CopyArrayToSlab(
        slabPtr, hostChannel.remoteBufferAddr, hostChannel.remoteBufferNum, layout.remoteBufferSection,
        &devChannel.remoteBufferAddr, "remoteBufferAddr"));
    CHK_RET(CopyArrayToSlab(
        slabPtr, hostChannel.sqContextAddr, hostChannel.sqNum, layout.sqContextSection, &devChannel.sqContextAddr,
        "sqContextAddr"));
    CHK_RET(CopyArrayToSlab(
        slabPtr, hostChannel.cqContextAddr, hostChannel.cqNum, layout.cqContextSection, &devChannel.cqContextAddr,
        "cqContextAddr"));
    return HCCL_SUCCESS;
}

HcclResult AivEntitySlabBase::CopyChannelEntityHeaderToSlab(
    void* slabPtr, const DeviceChannelEntityLayout& layout, const ChannelEntity& devChannel, void*& entityDevPtr)
{
    entityDevPtr = GetSlabPtr(slabPtr, layout.entitySection);
    CHK_PTR_NULL(entityDevPtr);
    Hccl::HrtMemcpy(
        entityDevPtr, sizeof(ChannelEntity), &devChannel, sizeof(ChannelEntity),
        Hccl::tagRtMemcpyKind::RT_MEMCPY_HOST_TO_DEVICE);
    return HCCL_SUCCESS;
}

void AivEntitySlabBase::ReleaseDeviceEntitySlab()
{
    if (devChannelEntitySlab_ != nullptr) {
        HcclResult ret = hrtFree(devChannelEntitySlab_);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[AivEntitySlabBase::%s] hrtFree devChannelEntitySlab is unsuccessful, ptr[%p], size[%zu], ret[%d]",
                __func__, devChannelEntitySlab_, devChannelEntitySlabSize_, ret);
        }
        devChannelEntitySlab_ = nullptr;
        devChannelEntitySlabSize_ = 0;
    }
    devChannelEntity_ = nullptr;
}

AivChannelEntity::AivChannelEntity(AivUrmaChannel& owner) : owner_(owner) {}

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
        if (linkStatusList[i] != HCOMM_CHANNEL_STATUS_READY) {
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

AivRoceHostLocChannelEntity::AivRoceHostLocChannelEntity(AivRoceHostLocChannel& owner) : owner_(owner) {}

HcclResult AivRoceHostLocChannelEntity::BuildChannelEntityToDevice(void** devChannelPtr)
{
    CHK_PTR_NULL(devChannelPtr);

    if (devChannelEntitySlab_ != nullptr) {
        *devChannelPtr = devChannelEntitySlab_;
        HCCL_INFO("[AivRoceHostLocChannelEntity::%s] already built, return cached devPtr=%p", __func__, *devChannelPtr);
        return HCCL_SUCCESS;
    }

    ChannelEntity hostEntity{};
    std::vector<RegedBufferEntity> locBufList;
    std::vector<RegedBufferEntity> rmtBufList;
    std::vector<SqContext> sqList;
    std::vector<CqContext> cqList;
    CHK_RET(owner_.BuildHostEntity(hostEntity, locBufList, rmtBufList, sqList, cqList));

    DeviceChannelEntityLayout layout;
    CHK_RET(BuildDeviceChannelEntityLayout(hostEntity, layout, false));
    void* slabPtr = nullptr;
    CHK_RET(AllocDeviceEntitySlab(layout.slabSize, slabPtr));
    AclDeviceSlabGuard slabGuard;
    slabGuard.Reset(slabPtr, layout.slabSize);

    ChannelEntity devEntity;
    CHK_RET(CopyChannelEntityArraysToSlab(slabPtr, hostEntity, layout, devEntity));
    void* entityDevPtr = nullptr;
    CHK_RET(CopyChannelEntityHeaderToSlab(slabPtr, layout, devEntity, entityDevPtr));

    ReleaseDeviceEntitySlab();
    devChannelEntitySlab_ = slabGuard.Release();
    devChannelEntitySlabSize_ = layout.slabSize;
    devChannelEntity_ = entityDevPtr;

    *devChannelPtr = entityDevPtr;
    HCCL_INFO(
        "[AivRoceHostLocChannelEntity::%s] Success, devPtr=%p, slabPtr=%p, slabSize=%zu", __func__, *devChannelPtr,
        devChannelEntitySlab_, devChannelEntitySlabSize_);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannelEntity::PreAllocChannelEntityToDevice(void** devChannelPtr)
{
    CHK_PTR_NULL(devChannelPtr);

    if (devChannelEntitySlab_ != nullptr) {
        *devChannelPtr = devChannelEntitySlab_;
        HCCL_INFO("[AivRoceHostLocChannelEntity::%s] already built, return cached devPtr=%p", __func__, *devChannelPtr);
        return HCCL_SUCCESS;
    }

    ChannelEntity tmp{};
    tmp.localNotifyNum = 0;
    tmp.remoteNotifyNum = 0;
    tmp.localBufferNum = owner_.bufferNum_;
    tmp.remoteBufferNum = owner_.bufferNum_;
    tmp.sqNum = owner_.connNum_;
    tmp.cqNum = owner_.connNum_;

    DeviceChannelEntityLayout layout;
    CHK_RET(BuildDeviceChannelEntityLayout(tmp, layout, false));

    void* slabPtr = nullptr;
    CHK_RET(AllocDeviceEntitySlab(layout.slabSize, slabPtr));
    AclDeviceSlabGuard slabGuard;
    slabGuard.Reset(slabPtr, layout.slabSize);

    devChannelEntitySlab_ = slabGuard.Release();
    devChannelEntitySlabSize_ = layout.slabSize;
    devChannelEntity_ = GetSlabPtr(devChannelEntitySlab_, layout.entitySection);
    *devChannelPtr = devChannelEntitySlab_;

    HCCL_INFO(
        "[AivRoceHostLocChannelEntity::%s] pre-alloc success, slabPtr=%p, slabSize=%zu", __func__,
        devChannelEntitySlab_, devChannelEntitySlabSize_);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannelEntity::FillChannelEntityToDevice()
{
    if (devChannelEntitySlab_ == nullptr) {
        HCCL_ERROR("[AivRoceHostLocChannelEntity::%s] devChannelEntitySlab_ is nullptr, not pre-allocated.", __func__);
        return HCCL_E_INTERNAL;
    }
    if (owner_.channelStatus_ != ChannelStatus::READY) {
        HCCL_ERROR(
            "[AivRoceHostLocChannelEntity::%s] channel status[%d] is not ready[%d], please check.", __func__,
            owner_.channelStatus_, ChannelStatus::READY);
        return HCCL_E_INTERNAL;
    }

    ChannelEntity hostEntity{};
    std::vector<RegedBufferEntity> locBufList;
    std::vector<RegedBufferEntity> rmtBufList;
    std::vector<SqContext> sqList;
    std::vector<CqContext> cqList;
    CHK_RET(owner_.BuildHostEntity(hostEntity, locBufList, rmtBufList, sqList, cqList));

    DeviceChannelEntityLayout layout;
    CHK_RET(BuildDeviceChannelEntityLayout(hostEntity, layout, false));
    CHK_RET(CheckSlabCapacity(layout));

    ChannelEntity devEntity;
    CHK_RET(CopyChannelEntityArraysToSlab(devChannelEntitySlab_, hostEntity, layout, devEntity));
    void* entityDevPtr = nullptr;
    CHK_RET(CopyChannelEntityHeaderToSlab(devChannelEntitySlab_, layout, devEntity, entityDevPtr));

    devChannelEntity_ = entityDevPtr;
    HCCL_INFO("[AivRoceHostLocChannelEntity::%s] fill success, devPtr=%p", __func__, entityDevPtr);
    return HCCL_SUCCESS;
}

} // namespace hcomm
