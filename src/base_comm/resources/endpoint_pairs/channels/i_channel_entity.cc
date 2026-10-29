/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "i_channel_entity.h"

#include <cstdint>
#include <cstddef>

#include "log.h"
#include "adapter_rts.h"
#include "orion_adapter_rts.h"

namespace hcomm {
namespace {
    constexpr size_t DEVICE_ENTITY_ALIGN_SIZE = 64;

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
                "[IChannelEntity::%s] %s size overflow, elemSize[%zu], elemNum[%u]", __func__, sectionName, elemSize,
                elemNum),
            HCCL_E_PARA);
        section.size = elemSize * static_cast<size_t>(elemNum);
        CHK_PRT_RET(
            section.offset > (SIZE_MAX - section.size),
            HCCL_ERROR(
                "[IChannelEntity::%s] %s offset overflow, offset[%zu], size[%zu]", __func__, sectionName,
                section.offset, section.size),
            HCCL_E_PARA);
        offset = section.offset + section.size;
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult IChannelEntity::BuildDeviceChannelEntityLayout(
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

HcclResult IChannelEntity::AllocDeviceEntitySlab(size_t slabSize, void*& slabPtr)
{
    HcclResult ret = hrtMalloc(&slabPtr, slabSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS || slabPtr == nullptr,
        HCCL_ERROR("[IChannelEntity::%s] hrtMalloc slab failed, ret[%d], size[%zu]", __func__, ret, slabSize),
        HCCL_E_MEMORY);
    return HCCL_SUCCESS;
}

void* IChannelEntity::GetSlabPtr(void* base, const DeviceEntitySection& section)
{
    if (section.size == 0) {
        return nullptr;
    }
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + section.offset);
}

HcclResult IChannelEntity::CheckSlabCapacity(const DeviceChannelEntityLayout& layout) const
{
    if (layout.slabSize > devChannelEntitySlabSize_) {
        HCCL_ERROR(
            "[IChannelEntity::%s] slabSize[%zu] > preAllocSize[%zu]", __func__, layout.slabSize,
            devChannelEntitySlabSize_);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult IChannelEntity::CopyChannelEntityArraysToSlab(
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

HcclResult IChannelEntity::CopyChannelEntityHeaderToSlab(
    void* slabPtr, const DeviceChannelEntityLayout& layout, const ChannelEntity& devChannel, void*& entityDevPtr)
{
    entityDevPtr = GetSlabPtr(slabPtr, layout.entitySection);
    CHK_PTR_NULL(entityDevPtr);
    Hccl::HrtMemcpy(
        entityDevPtr, sizeof(ChannelEntity), &devChannel, sizeof(ChannelEntity),
        Hccl::tagRtMemcpyKind::RT_MEMCPY_HOST_TO_DEVICE);
    return HCCL_SUCCESS;
}

void IChannelEntity::ReleaseDeviceEntitySlab()
{
    if (devChannelEntitySlab_ != nullptr) {
        HcclResult ret = hrtFree(devChannelEntitySlab_);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[IChannelEntity::%s] hrtFree devChannelEntitySlab is unsuccessful, ptr[%p], size[%zu], ret[%d]",
                __func__, devChannelEntitySlab_, devChannelEntitySlabSize_, ret);
        }
        devChannelEntitySlab_ = nullptr;
        devChannelEntitySlabSize_ = 0;
    }
    devChannelEntity_ = nullptr;
}

} // namespace hcomm
