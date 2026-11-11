/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "thread_entity_resources.h"

#ifndef CCL_KERNEL_AICPU
#include <algorithm>
#include <cstddef>
#include <memory>
#include <sstream>
#include <utility>
#include "adapter_rts_common.h"
#include "aicpu_ts_thread.h"
#include "cpu_thread/cpu_thread.h"
#include "cpu_ts_thread.h"
#include "common/dev_type_utils_pub.h"
#include "thread.h"

namespace hccl {
namespace {
    constexpr size_t RTSQ_OFFSET = sizeof(HcommThreadEntity);
    constexpr size_t RTSQ_SQE_SIZE = 64;
    constexpr size_t RTSQ_SQE_BATCH_SIZE = 128;
    // SQE 缓存起点按单个 SQE 大小对齐。
    constexpr size_t SQE_OFFSET
        = (RTSQ_OFFSET + sizeof(HcommThreadSqContext) + RTSQ_SQE_SIZE - 1) / RTSQ_SQE_SIZE * RTSQ_SQE_SIZE;
    constexpr size_t FIXED_POD_SIZE = SQE_OFFSET + RTSQ_SQE_SIZE * RTSQ_SQE_BATCH_SIZE;
    // Notify 分段链：每段前 NOTIFY_SEGMENT_DATA_NUM 个槽为数据，末槽为指向下一段的链接；
    // 段 0 地址在 Init 后不变，旧段保留至析构，Device 侧缓存的旧指针始终有效。
    constexpr uint32_t NOTIFY_SEGMENT_DATA_NUM = 127U;

    struct HcommNotifySegmentLink {
        uint64_t nextSegmentAddr; // 下一个 Notify 分段地址；0 表示末段。
        uint8_t reserve[56];
    }; // 64B，与 RegedNotifyEntity 槽位大小一致
    static_assert(sizeof(HcommNotifySegmentLink) == sizeof(RegedNotifyEntity), "notify segment slot size mismatch");

    // nextSegmentAddr 为空表示末段。
    HcclResult WriteNotifyLink(DeviceMem& mem, const void* nextSegmentAddr)
    {
        HcommNotifySegmentLink link{};
        link.nextSegmentAddr = reinterpret_cast<uint64_t>(nextSegmentAddr);
        auto* dst = reinterpret_cast<HcommNotifySegmentLink*>(
            static_cast<RegedNotifyEntity*>(mem.ptr()) + NOTIFY_SEGMENT_DATA_NUM);
        return hrtMemSyncCopy(
            dst, sizeof(link), &link, sizeof(link), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    }

    HcclResult GetThreadExportRtsqInfo(Thread& thread, int32_t& streamId, uint32_t& sqId)
    {
        if (auto* cpuThread = dynamic_cast<CpuTsThread*>(&thread)) {
            return cpuThread->GetExportRtsqInfo(streamId, sqId);
        }
        if (auto* aicpuThread = dynamic_cast<AicpuTsThread*>(&thread)) {
            return aicpuThread->GetExportRtsqInfo(streamId, sqId);
        }
        return HCCL_E_NOT_SUPPORT;
    }
} // namespace

HcclResult BuildThreadEntityNotifies(const Thread& thread, std::vector<RegedNotifyEntity>& notifies)
{
    notifies.resize(thread.GetNotifyNum());
    for (size_t i = 0; i < notifies.size(); ++i) {
        LocalNotify* notify = thread.GetNotify(static_cast<uint32_t>(i));
        CHK_PTR_NULL(notify);
        CHK_PRT_RET(
            notify->notifyId_ > INT32_MAX, HCCL_ERROR("[%s] invalid notify ID[%u]", __func__, notify->notifyId_),
            HCCL_E_PARA);
        notifies[i] = RegedNotifyEntity{};
        notifies[i].type = REGED_NOTIFY_IPC_RT;
        notifies[i].notifyInfo.ipcRt.notifyId = static_cast<int32_t>(notify->notifyId_);
    }
    return HCCL_SUCCESS;
}

HcclResult PreAllocThreadEntity(Thread& thread, ThreadHandle& handle, CommEngine targetEngine)
{
    CHK_PRT_RET(
        targetEngine != COMM_ENGINE_AICPU && targetEngine != COMM_ENGINE_AICPU_TS,
        HCCL_ERROR("[%s] unsupported target engine[%d]", __func__, targetEngine), HCCL_E_PARA);
    handle = 0;
    if (auto* cpuThread = dynamic_cast<CpuTsThread*>(&thread)) {
        return cpuThread->PreAllocThreadEntity(handle, targetEngine);
    }
    if (auto* hostDpuThread = dynamic_cast<CpuThread*>(&thread)) {
        return hostDpuThread->PreAllocThreadEntity(handle, targetEngine);
    }
    if (auto* aicpuThread = dynamic_cast<AicpuTsThread*>(&thread)) {
        return aicpuThread->PreAllocThreadEntity(handle, targetEngine);
    }
    HCCL_ERROR("[%s] unsupported thread type for entity export, thread[%p]", __func__, &thread);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult PreAllocThreadEntityCommon(
    Thread& thread, std::unordered_map<CommEngine, ThreadHandleEntry>& entries, ThreadHandle& handle,
    CommEngine targetEngine)
{
    CHK_PRT_RET(
        targetEngine != COMM_ENGINE_AICPU && targetEngine != COMM_ENGINE_AICPU_TS,
        HCCL_ERROR("[%s] unsupported target engine[%d]", __func__, targetEngine), HCCL_E_PARA);
    handle = 0;
    auto iter = entries.find(targetEngine);
    if (iter == entries.end()) {
        EXCEPTION_CATCH(iter = entries.emplace(targetEngine, ThreadHandleEntry{}).first, return HCCL_E_MEMORY);
    }
    const bool isHostDpuThread = dynamic_cast<CpuThread*>(&thread) != nullptr;
    if (!iter->second.entityResources) {
        std::unique_ptr<ThreadEntityResources> resources;
        EXCEPTION_CATCH(resources = std::make_unique<ThreadEntityResources>(), return HCCL_E_MEMORY);
        HcommDevInfo devInfo{};
        auto isGe950 = hcomm::EqualToOrGreaterThan950();
        CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
        const bool initPodFields = *isGe950;
        if (initPodFields) {
            int32_t logicId = 0;
            CHK_RET(hrtGetDevice(&logicId));
            CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(logicId), devInfo.devPhyId));
        }
        ThreadEntityResources::InitParam param;
        param.devInfo = devInfo;
        param.threadType = thread.GetThreadType();
        param.engine = targetEngine;
        param.initPodFields = initPodFields;
        // HostDpu 队列线程没有 RTSQ/LocalNotify POD；SQ/CQ 与 notify 寻址由队列名片
        // 单独投递给 device 侧 CpuDeviceThread，Entity 只承载 ABI 与兼容句柄。
        param.hasRtsq = initPodFields && !isHostDpuThread && !thread.IsFakeDeviceRes();
        if (param.hasRtsq) {
            int32_t streamId = 0;
            uint32_t sqId = 0;
            CHK_RET(GetThreadExportRtsqInfo(thread, streamId, sqId));
            param.streamId = static_cast<uint32_t>(streamId);
            param.sqId = sqId;
        }
        if (param.initPodFields && !isHostDpuThread) {
            EXCEPTION_CATCH(CHK_RET(BuildThreadEntityNotifies(thread, param.notifies)), return HCCL_E_MEMORY);
        }
        CHK_RET(resources->Init(param));
        iter->second.entityResources = std::move(resources);
    } else if (iter->second.handle == 0) {
        // 预分配后下发失败的 entity 尚未发布，重试前按当前 Host Notify 数刷新容量。
        if (!isHostDpuThread) {
            EXCEPTION_CATCH(CHK_RET(iter->second.entityResources->RefreshNotifyBuffer(thread)), return HCCL_E_MEMORY);
        }
    }
    handle = iter->second.entityResources->GetEntityHandle();
    return HCCL_SUCCESS;
}

ThreadEntityResources* GetThreadEntityResourcesCommon(
    const std::unordered_map<CommEngine, ThreadHandleEntry>& entries, CommEngine targetEngine)
{
    auto iter = entries.find(targetEngine);
    return iter == entries.end() ? nullptr : iter->second.entityResources.get();
}

// 分配一个 Notify 段（数据槽 + 末位链接槽）并置零，失败返回 nullptr。
RegedNotifyEntity* ThreadEntityResources::NewNotifySegment()
{
    DeviceMem mem = DeviceMem::alloc((NOTIFY_SEGMENT_DATA_NUM + 1U) * sizeof(RegedNotifyEntity));
    if (mem.ptr() == nullptr || hrtMemSet(mem.ptr(), mem.size(), mem.size()) != HCCL_SUCCESS) {
        HCCL_ERROR("[NewNotifySegment] alloc or memset notify segment failed.");
        return nullptr;
    }
    RegedNotifyEntity* ptr = static_cast<RegedNotifyEntity*>(mem.ptr());
    EXCEPTION_CATCH(notifySegments_.push_back(std::move(mem)), return nullptr);
    return ptr;
}

HcclResult ThreadEntityResources::Init(const InitParam& param)
{
    CHK_PRT_RET(fixedMem_.ptr() != nullptr, HCCL_ERROR("[%s] already initialized", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        param.threadType == THREAD_TYPE_INVALID, HCCL_ERROR("[%s] thread type is invalid", __func__), HCCL_E_PARA);
    DeviceMem fixedMem = DeviceMem::alloc(param.hasRtsq ? FIXED_POD_SIZE : sizeof(HcommThreadEntity));
    CHK_PTR_NULL(fixedMem.ptr());
    CHK_RET(hrtMemSet(fixedMem.ptr(), fixedMem.size(), fixedMem.size()));
    HcommThreadEntity entity{};
    entity.abiHeader = {HCOMM_THREAD_ENTITY_VERSION, HCOMM_THREAD_ENTITY_MAGIC_WORD, sizeof(HcommThreadEntity), 0};
    entity.threadType = param.threadType;
    entity.engine = param.engine;
    if (param.initPodFields) {
        auto* base = static_cast<uint8_t*>(fixedMem.ptr());
        entity.devInfo = param.devInfo;
        if (param.hasRtsq) {
            entity.sqContextAddr = reinterpret_cast<HcommThreadSqContext*>(base + RTSQ_OFFSET);
            HcommThreadSqContext sq{};
            sq.type = HCOMM_THREAD_SQ_CONTEXT_TYPE_RTSQ;
            sq.contextInfo.rtsqContext.streamId = param.streamId;
            sq.contextInfo.rtsqContext.sqId = param.sqId;
            sq.contextInfo.rtsqContext.localSqAddr = reinterpret_cast<uintptr_t>(base + SQE_OFFSET);
            sq.contextInfo.rtsqContext.localSqDepth = RTSQ_SQE_BATCH_SIZE;
            CHK_RET(hrtMemSyncCopy(
                entity.sqContextAddr, sizeof(sq), &sq, sizeof(sq),
                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        }
    }
    CHK_RET(hrtMemSyncCopy(
        fixedMem.ptr(), sizeof(entity), &entity, sizeof(entity), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    fixedMem_ = std::move(fixedMem);
    podFieldsInited_ = param.initPodFields;
    if (param.initPodFields) {
        CHK_RET(WriteNotifyBuffer(param.notifies));
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadEntityResources::WriteNotifyBuffer(const std::vector<RegedNotifyEntity>& notifies)
{
    CHK_PTR_NULL(fixedMem_.ptr());
    // A3 不使用 Notify POD。
    if (!podFieldsInited_) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        notifies.size() > UINT32_MAX,
        HCCL_ERROR("[%s] notify count[%zu] exceeds the uint32_t ABI field", __func__, notifies.size()), HCCL_E_PARA);
    const uint32_t notifyNum = static_cast<uint32_t>(notifies.size());
    while (notifySegments_.size() * NOTIFY_SEGMENT_DATA_NUM < notifyNum) {
        RegedNotifyEntity* segment = NewNotifySegment();
        CHK_PTR_NULL(segment);
    }
    // 先连接相邻段；每次更新都补齐链接，保证上次写链接失败后可以重试。
    for (size_t i = 1; i < notifySegments_.size(); ++i) {
        CHK_RET(WriteNotifyLink(notifySegments_[i - 1], notifySegments_[i].ptr()));
    }

    // 段地址追加式增长保证旧指针稳定；段内数据全量重拷，失败重试可覆盖旧状态。
    for (uint32_t offset = 0; offset < notifyNum; offset += NOTIFY_SEGMENT_DATA_NUM) {
        const uint32_t count = std::min(notifyNum - offset, NOTIFY_SEGMENT_DATA_NUM);
        DeviceMem& segment = notifySegments_[offset / NOTIFY_SEGMENT_DATA_NUM];
        if (count > 0) {
            CHK_RET(hrtMemSyncCopy(
                segment.ptr(), count * sizeof(RegedNotifyEntity), notifies.data() + offset,
                count * sizeof(RegedNotifyEntity), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        }
    }

    // 段 0 地址不变；无 Notify 时发布空地址，消费方必须以 localNotifyNum 为准。
    auto* entity = static_cast<uint8_t*>(fixedMem_.ptr());
    const RegedNotifyEntity* localNotifyAddr
        = notifySegments_.empty() ? nullptr : static_cast<RegedNotifyEntity*>(notifySegments_[0].ptr());
    CHK_RET(hrtMemSyncCopy(
        entity + offsetof(HcommThreadEntity, localNotifyAddr), sizeof(localNotifyAddr), &localNotifyAddr,
        sizeof(localNotifyAddr), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    // 数量最后发布，避免消费者读到新数量后访问尚未写完的段数据。
    CHK_RET(hrtMemSyncCopy(
        entity + offsetof(HcommThreadEntity, localNotifyNum), sizeof(notifyNum), &notifyNum, sizeof(notifyNum),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult ThreadEntityResources::RefreshNotifyBuffer(const Thread& thread)
{
    CHK_PTR_NULL(fixedMem_.ptr());
    // A3 不使用 Notify POD，不能为会被丢弃的描述数组引入 Thread Notify 校验失败面。
    if (!podFieldsInited_) {
        return HCCL_SUCCESS;
    }
    std::vector<RegedNotifyEntity> notifies;
    CHK_RET(BuildThreadEntityNotifies(thread, notifies));
    return WriteNotifyBuffer(notifies);
}
} // namespace hccl
#endif // CCL_KERNEL_AICPU
