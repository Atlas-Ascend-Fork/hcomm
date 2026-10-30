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
#include "aicpu_ts_channel_entity.h"
#include "acl_device_slab_guard.h"
#include "roce_host_loc/aicpu_ts_roce_host_loc_channel.h"
#include "endpoint_pair/channel/channel_process_pub.h"
#include "channel.h"
#include "launch_aicpu.h"
#include "launch_device.h"
#include "common/comm_engine_utils_pub.h"
#include "adapter_rts_common.h"
#include "base_comm_node_accessors.h"

using namespace hcomm;

HcclResult AicpuTsChannelEntity::HandleStatus(
    const ChannelHandle* channelList, uint32_t listNum, CommEngine engine, const HcommChannelDesc* channelDescs,
    const std::vector<int32_t>& linkStatusList, int32_t* statusList)
{
    bool allReady = true;
    for (uint32_t i = 0; i < listNum; i++) {
        if (linkStatusList[i] != hcomm::HCOMM_CHANNEL_STATUS_READY) {
            allReady = false;
            break;
        }
    }
    if (!allReady) {
        for (uint32_t i = 0; i < listNum; i++) {
            if (linkStatusList[i] == hcomm::HCOMM_CHANNEL_STATUS_FAILED) {
                statusList[i] = hcomm::HCOMM_CHANNEL_STATUS_FAILED;
            } else if (linkStatusList[i] == hcomm::HCOMM_CHANNEL_STATUS_TIMEOUT) {
                statusList[i] = hcomm::HCOMM_CHANNEL_STATUS_TIMEOUT;
            } else {
                statusList[i] = hcomm::HCOMM_CHANNEL_STATUS_CONNECTING;
            }
        }
        return HCCL_SUCCESS;
    }
    CHK_RET(GetAicpuCommEngineBinProc().EnsureKernelBinLoaded(engine));
    HcclResult kernelRet
        = LaunchKernel(channelList, listNum, engine, channelDescs, GetAicpuCommEngineBinProc().GetBinHandle());
    if (kernelRet != HCCL_SUCCESS) {
        HCCL_ERROR("[AicpuTsChannelEntity::%s] LaunchKernel failed, ret[%d]", __func__, kernelRet);
        return HCCL_E_INTERNAL;
    }
    for (uint32_t i = 0; i < listNum; i++) {
        statusList[i] = linkStatusList[i];
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsChannelEntity::EnsureDeviceReady(const ChannelHandle* channelList, uint32_t listNum)
{
    CHK_PTR_NULL(channelList);
    CHK_PRT_RET((listNum == 0), HCCL_ERROR("[%s] Invalid listNum, listNum[%u]", __func__, listNum), HCCL_E_PARA);

    // 校验引擎并收集 desc（LaunchKernel 入参）
    std::vector<HcommChannelDesc> channelDescs(listNum);
    CommEngine engine = COMM_ENGINE_RESERVED;
    for (uint32_t i = 0; i < listNum; i++) {
        void* ch = nullptr;
        CHK_RET(ChannelProcess::ChannelGet(channelList[i], &ch));
        CHK_PTR_NULL(ch);
        auto* channel = static_cast<Channel*>(ch);
        CommEngine curEngine = channel->GetEngine();
        if (curEngine != COMM_ENGINE_AICPU && curEngine != COMM_ENGINE_AICPU_TS) {
            HCCL_ERROR(
                "[%s] unsupported engine[%s], only AICPU/AICPU_TS channels have device entity.", __func__,
                GetEnumToString(GetCommEngineStatusStrMap(), curEngine).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
        if (i == 0) {
            engine = curEngine;
        }
        channelDescs[i] = channel->GetChannelDesc();
    }

    CHK_RET(GetAicpuCommEngineBinProc().EnsureKernelBinLoaded(engine));
    return LaunchKernel(channelList, listNum, engine, channelDescs.data(), GetAicpuCommEngineBinProc().GetBinHandle());
}

HcclResult AicpuTsChannelEntity::PreAllocChannels(
    ChannelHandle* targetChannels, ChannelHandle* userChannels, [[maybe_unused]] HcommChannelDesc* channelDescs,
    uint32_t channelNum)
{
    CHK_PTR_NULL(targetChannels);
    CHK_PTR_NULL(userChannels);
    CHK_PRT_RET(
        (channelNum == 0),
        HCCL_ERROR("[AicpuTsChannelEntity::%s] Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);

    for (uint32_t i = 0; i < channelNum; i++) {
        auto* channel = ReinterpretAs<Channel*>(targetChannels[i]);
        CHK_PTR_NULL(channel);
        auto* helper = channel->GetChannelEntity();
        CHK_PTR_NULL(helper);
        CHK_RET(helper->PreAllocCtx(userChannels[i]));
        // 清零 ctx device 内存，防止脏数据 magic 误匹配
        CHK_RET(hrtMemSet(helper->GetCtxPtr(), sizeof(HcommAicpuChannelCtx), sizeof(HcommAicpuChannelCtx)));
    }

    CHK_RET(ChannelProcess::FillChannelD2HMap(userChannels, targetChannels, channelNum));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsChannelEntity::TryFillCtxList(
    ChannelHandle* hostChannelHandles, uint32_t listNum, const hccl::DeviceMem& deviceChannelList, void*& outCtxList,
    bool& isCtxMode)
{
    auto* firstCh = ReinterpretAs<Channel*>(hostChannelHandles[0]);
    CHK_PTR_NULL(firstCh);
    auto* firstHelper = firstCh->GetChannelEntity();
    CHK_PTR_NULL(firstHelper);
    isCtxMode = (firstHelper->GetCtxPtr() != nullptr);
    HCCL_INFO("[AicpuTsChannelEntity::%s] isCtxMode[%d].", __func__, isCtxMode);
    if (!isCtxMode) {
        return HCCL_SUCCESS;
    }
    std::vector<void*> ctxVec(listNum);
    for (uint32_t i = 0; i < listNum; i++) {
        auto* ch = ReinterpretAs<Channel*>(hostChannelHandles[i]);
        CHK_PTR_NULL(ch);
        auto* helper = ch->GetChannelEntity();
        ctxVec[i] = helper ? helper->GetCtxPtr() : nullptr;
    }
    HcclResult ret = hrtMemSyncCopy(
        deviceChannelList.ptr(), listNum * sizeof(void*), ctxVec.data(), listNum * sizeof(void*),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[AicpuTsChannelEntity::%s] hrtMemSyncCopy failed, ret[%d]", __func__, ret);
        return ret;
    }
    outCtxList = deviceChannelList.ptr();
    return HCCL_SUCCESS;
}

HcclResult AicpuTsChannelEntity::LaunchKernel(
    const ChannelHandle* channelList, uint32_t listNum, [[maybe_unused]] CommEngine engine,
    const HcommChannelDesc* channelDescs, aclrtBinHandle binHandle)
{
    CHK_PTR_NULL(channelList);
    CHK_PRT_RET(
        (listNum == 0), HCCL_ERROR("[AicpuTsChannelEntity::%s] Invalid listNum, listNum[%u]", __func__, listNum),
        HCCL_E_PARA);

    // 过滤出未就绪的子集，避免重复下 kernel 导致 device 侧 channel 对象泄漏
    std::vector<ChannelHandle> subHostHandles;
    std::vector<HcommChannelDesc> subDescs;
    std::vector<Channel*> subChannels;
    subHostHandles.reserve(listNum);
    subDescs.reserve(listNum);
    subChannels.reserve(listNum);
    for (uint32_t i = 0; i < listNum; i++) {
        void* ch = nullptr;
        CHK_RET(ChannelProcess::ChannelGet(channelList[i], &ch));
        CHK_PTR_NULL(ch);
        auto* channel = static_cast<Channel*>(ch);
        auto* entity = channel->GetChannelEntity();
        CHK_PTR_NULL(entity);
        if (entity->IsDeviceEntityReady()) {
            continue;
        }
        subHostHandles.push_back(ReinterpretAs<ChannelHandle>(ch));
        subDescs.push_back(channelDescs[i]);
        subChannels.push_back(channel);
    }
    if (subHostHandles.empty()) {
        HCCL_INFO("[AicpuTsChannelEntity::%s] all channels already ready, skip kernel launch.", __func__);
        return HCCL_SUCCESS;
    }
    // 序列化 + kernel launch
    std::vector<ChannelHandle> devHandles(subHostHandles.size());
    CHK_RET(ChannelProcess::LaunchChannelKernel(
        devHandles.data(), subHostHandles.data(), subDescs.data(), subHostHandles.size(), binHandle));

    // In ctx mode, read deviceChannel back from each ctx and refresh the host-to-device mapping.
    if (!subChannels.empty()) {
        auto* firstEntity = subChannels[0]->GetChannelEntity();
        if (firstEntity != nullptr && firstEntity->GetCtxPtr() != nullptr) {
            for (size_t i = 0; i < subHostHandles.size(); ++i) {
                auto* entity = subChannels[i]->GetChannelEntity();
                CHK_PTR_NULL(entity);
                void* ctxPtr = entity->GetCtxPtr();
                CHK_PTR_NULL(ctxPtr);

                HcommAicpuChannelCtx ctx{};
                CHK_RET(hrtMemSyncCopy(
                    &ctx, sizeof(ctx), ctxPtr, sizeof(ctx), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));
                CHK_PRT_RET(
                    ctx.abiHeader.version != HCOMM_AICPU_CHANNEL_CTX_VERSION
                        || ctx.abiHeader.magicWord != HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD,
                    HCCL_ERROR(
                        "[AicpuTsChannelEntity::%s] invalid AICPU channel ctx ABI, version[%u], magicWord[0x%x].",
                        __func__, ctx.abiHeader.version, ctx.abiHeader.magicWord),
                    HCCL_E_INTERNAL);
                CHK_PRT_RET(
                    ctx.deviceChannel == nullptr,
                    HCCL_ERROR("[AicpuTsChannelEntity::%s] AICPU channel ctx deviceChannel is nullptr.", __func__),
                    HCCL_E_INTERNAL);
                devHandles[i] = ReinterpretAs<ChannelHandle>(ctx.deviceChannel);
            }
            CHK_RET(ChannelProcess::FillChannelD2HMap(
                devHandles.data(), subHostHandles.data(), static_cast<uint32_t>(subHostHandles.size())));
        }
    }

    for (auto* channel : subChannels) {
        auto* entity = channel->GetChannelEntity();
        CHK_PTR_NULL(entity);
        entity->SetDeviceEntityReady();
    }
    HCCL_INFO(
        "[AicpuTsChannelEntity::%s] aicpu kernel launch success, launched[%zu]/total[%u].", __func__,
        subHostHandles.size(), listNum);
    return HCCL_SUCCESS;
}

AicpuTsRoceV2ChannelEntity::AicpuTsRoceV2ChannelEntity(AicpuTsRoceHostLocChannel& owner) : owner_(owner) {}

HcclResult AicpuTsRoceV2ChannelEntity::BuildChannelEntityToDevice(void** devChannelPtr)
{
    CHK_PTR_NULL(devChannelPtr);

    if (devChannelEntitySlab_ != nullptr) {
        *devChannelPtr = devChannelEntitySlab_;
        HCCL_INFO("[AicpuTsRoceV2ChannelEntity::%s] already built, return cached devPtr=%p", __func__, *devChannelPtr);
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
        "[AicpuTsRoceV2ChannelEntity::%s] Success, devPtr=%p, slabPtr=%p, slabSize=%zu", __func__, *devChannelPtr,
        devChannelEntitySlab_, devChannelEntitySlabSize_);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceV2ChannelEntity::PreAllocChannelEntityToDevice(void** devChannelPtr)
{
    CHK_PTR_NULL(devChannelPtr);

    if (devChannelEntitySlab_ != nullptr) {
        *devChannelPtr = devChannelEntitySlab_;
        HCCL_INFO("[AicpuTsRoceV2ChannelEntity::%s] already built, return cached devPtr=%p", __func__, *devChannelPtr);
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
        "[AicpuTsRoceV2ChannelEntity::%s] pre-alloc success, slabPtr=%p, slabSize=%zu", __func__, devChannelEntitySlab_,
        devChannelEntitySlabSize_);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceV2ChannelEntity::FillChannelEntityToDevice()
{
    if (devChannelEntitySlab_ == nullptr) {
        HCCL_ERROR("[AicpuTsRoceV2ChannelEntity::%s] devChannelEntitySlab_ is nullptr, not pre-allocated.", __func__);
        return HCCL_E_INTERNAL;
    }
    if (owner_.channelStatus_ != ChannelStatus::READY) {
        HCCL_ERROR(
            "[AicpuTsRoceV2ChannelEntity::%s] channel status[%d] is not ready[%d], please check.", __func__,
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
    HCCL_INFO("[AicpuTsRoceV2ChannelEntity::%s] fill success, devPtr=%p", __func__, entityDevPtr);
    return HCCL_SUCCESS;
}
