/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef I_CHANNEL_ENTITY_H
#define I_CHANNEL_ENTITY_H

#include "hccl/hccl_types.h"
#include "hcomm_res_defs.h"
#include "hcomm/hcomm_res_entity_defs.h"

namespace hcomm {

struct DeviceEntitySection {
    size_t offset{0};
    size_t size{0};
};

struct DeviceChannelEntityLayout {
    DeviceEntitySection entitySection{0, sizeof(ChannelEntity)};
    DeviceEntitySection localNotifySection;
    DeviceEntitySection remoteNotifySection;
    DeviceEntitySection localBufferSection;
    DeviceEntitySection remoteBufferSection;
    DeviceEntitySection sqContextSection;
    DeviceEntitySection cqContextSection;
    DeviceEntitySection sqPiSection;
    DeviceEntitySection sqCiSection;
    DeviceEntitySection cqPiSection;
    DeviceEntitySection cqCiSection;
    size_t slabSize{0};
};

class IChannelEntity {
public:
    virtual ~IChannelEntity() = default;

    virtual HcclResult PreAllocCtx(ChannelHandle& outHandle) = 0;
    virtual void* GetCtxPtr() const = 0;
    virtual bool IsDeviceEntityReady() const { return deviceEntityReady_; }
    virtual HcclResult BuildChannelEntityToDevice(void** devChannelPtr)
    {
        (void)devChannelPtr;
        return HCCL_E_NOT_SUPPORT;
    }
    virtual HcclResult PreAllocChannelEntityToDevice(void** devChannelPtr)
    {
        (void)devChannelPtr;
        return HCCL_E_NOT_SUPPORT;
    }
    virtual HcclResult FillChannelEntityToDevice() { return HCCL_E_NOT_SUPPORT; }

    virtual void SetDeviceEntityReady() { deviceEntityReady_ = true; }

    void ReleaseDeviceEntitySlab();

protected:
    bool deviceEntityReady_{false}; // 设备实体就绪状态

    HcclResult BuildDeviceChannelEntityLayout(
        const ChannelEntity& hostChannel, DeviceChannelEntityLayout& layout, bool withQueueIdx);
    HcclResult AllocDeviceEntitySlab(size_t slabSize, void*& slabPtr);
    static void* GetSlabPtr(void* base, const DeviceEntitySection& section);
    HcclResult CheckSlabCapacity(const DeviceChannelEntityLayout& layout) const;
    HcclResult CopyChannelEntityArraysToSlab(
        void* slabPtr, const ChannelEntity& hostChannel, const DeviceChannelEntityLayout& layout,
        ChannelEntity& devChannel);
    HcclResult CopyChannelEntityHeaderToSlab(
        void* slabPtr, const DeviceChannelEntityLayout& layout, const ChannelEntity& devChannel, void*& entityDevPtr);

    static constexpr size_t QUEUE_INDEX_MEM_UNIT_SIZE = sizeof(void*);

    void* devChannelEntity_{nullptr};     // 设备侧实体头地址（slab 首段）
    void* devChannelEntitySlab_{nullptr}; // 设备实体 slab 基址
    size_t devChannelEntitySlabSize_{0};  // slab 总大小
};

} // namespace hcomm

#endif // I_CHANNEL_ENTITY_H
