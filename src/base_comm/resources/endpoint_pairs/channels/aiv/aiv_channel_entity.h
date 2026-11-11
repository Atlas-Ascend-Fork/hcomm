/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_CHANNEL_ENTITY_H
#define AIV_CHANNEL_ENTITY_H

#include <vector>
#include "hccl/hccl_types.h"
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"
#include "../i_channel_entity.h"

namespace hcomm {

class AivUrmaChannel;
class AivRoceHostLocChannel;

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

/**
 * @note 职责：AIV 实体 slab 能力基类：设备实体 slab 布局计算与分配/拷贝/校验辅助方法、slab 成员与释放入口，
 *       仅供 AIV 族通道实体复用；跨引擎的 ctx/就绪标志/三段式虚接口见 IChannelEntity。
 */
class AivEntitySlabBase : public IChannelEntity {
public:
    void ReleaseDeviceEntitySlab();

protected:
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

class AivChannelEntity : public AivEntitySlabBase {
public:
    explicit AivChannelEntity(AivUrmaChannel& owner);

    HcclResult BuildChannelEntityToDevice(void** devChannelPtr) override;
    HcclResult PreAllocChannelEntityToDevice(void** devChannelPtr) override;
    HcclResult FillChannelEntityToDevice() override;

    static HcclResult HandleStatus(
        const ChannelHandle* channelList, uint32_t listNum, const HcommChannelDesc* channelDescs,
        const std::vector<int32_t>& linkStatusList, int32_t* statusList);
    static HcclResult PreAllocChannels(
        ChannelHandle* targetChannels, ChannelHandle* userChannels, const HcommChannelDesc* channelDescs,
        uint32_t channelNum);

private:
    HcclResult InitQueueIndexSections(void* slabPtr, const DeviceChannelEntityLayout& layout, uint32_t queueNum);
    void SetQueueIndexDeviceMem(void* slabPtr, const DeviceChannelEntityLayout& layout, uint32_t queueNum);
    HcclResult ZeroQueueIndexSection(
        void* slabPtr, const std::vector<uint8_t>& zeroQueueIndexMem, const DeviceEntitySection& section);

    static HcclResult FillDevEntities(
        const ChannelHandle* channelList, uint32_t listNum, const HcommChannelDesc* channelDescs,
        const int32_t* linkStatusList);

    AivUrmaChannel& owner_;
};

/**
 * @note 职责：AIV RoCE 通道实体：继承 AivEntitySlabBase（设备实体 slab 能力；
 *       ctx/就绪标志等跨引擎能力来自 IChannelEntity），设备实体 slab 三段式（Build/PreAlloc/Fill）在本类覆写，
 *       经 owner_ 驱动宿主通道；AIV 无 ctx 路径，PreAllocCtx/GetCtxPtr 直接走基类默认（NOT_SUPPORT/nullptr）。
 *       待 RoCE 数据获取收归 Transport（实体仅依赖统一数据源接口）后，本类可与 AivChannelEntity 归一。
 */
class AivRoceHostLocChannelEntity : public AivEntitySlabBase {
public:
    explicit AivRoceHostLocChannelEntity(AivRoceHostLocChannel& owner);

    HcclResult BuildChannelEntityToDevice(void** devChannelPtr) override;
    HcclResult PreAllocChannelEntityToDevice(void** devChannelPtr) override;
    HcclResult FillChannelEntityToDevice() override;

private:
    AivRoceHostLocChannel& owner_;
};

} // namespace hcomm

#endif // AIV_CHANNEL_ENTITY_H
