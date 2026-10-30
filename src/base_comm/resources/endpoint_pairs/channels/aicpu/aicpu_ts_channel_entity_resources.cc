/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ts_channel_entity_resources.h"

#include <cstring>
#include <new>
#include <type_traits>

#include "adapter_rts_common.h"
#include "aicpu_res_package_helper.h"
#include "binary_stream.h"
#include "log.h"

namespace {

using namespace hcomm;
using HcommJfsContext = decltype(((SqContext*)nullptr)->contextInfo.ubJfs);

constexpr uint32_t UB_WQE_NUM_PER_SQE = 4U; // URMA约束每个SQE包含4个WQEBB
constexpr size_t URMA_EID_SIZE = 16U;
constexpr u32 STREAM_RES_TYPE = static_cast<u32>(Hccl::AicpuResMgrType::STREAM);

class ChannelEntityReader {
public:
    ChannelEntityReader(const void* data, size_t size) : data_(static_cast<const uint8_t*>(data)), size_(size) {}

    template <typename T>
    bool Read(T& value)
    {
        static_assert(std::is_trivially_copyable<T>::value, "channel entity field must be trivially copyable");
        return ReadBytes(&value, sizeof(value));
    }

    bool ReadBytes(void* dst, size_t size)
    {
        if (offset_ > size_) {
            return false;
        }
        if (size > size_ - offset_) {
            return false;
        }
        if (size != 0) {
            (void)std::memcpy(dst, data_ + offset_, size);
        }
        offset_ += size;
        return true;
    }

    bool ReadVector(std::vector<char>& value)
    {
        size_t size = 0;
        if (!Read(size) || size > size_ - offset_) {
            return false;
        }
        value.assign(data_ + offset_, data_ + offset_ + size);
        offset_ += size;
        return true;
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t offset_{0};
};

struct ChannelRemoteResource {
    uint64_t addr;
    uint64_t size;
    uint32_t tokenId;
    uint32_t tokenValue;
    uint32_t notifyId;
};

struct ChannelLocalResource {
    uint64_t addr;
    uint64_t size;
    uint32_t tokenId;
    uint32_t tokenValue;
};

void FillUrmaProtection(ProtectionInfo& protection, uint32_t tokenId, uint32_t tokenValue)
{
    protection = {};
    protection.type = PROTECTION_TYPE_UB;
    protection.memInfo.ub.tokenId = tokenId;
    protection.memInfo.ub.tokenValue = tokenValue;
}

void FillRegedNotify(RegedNotifyEntity& notify, const ChannelRemoteResource& resource)
{
    notify = {};
    notify.type = REGED_NOTIFY_RMA_RT;
    notify.notifyInfo.rmaRt.addr = resource.addr;
    notify.notifyInfo.rmaRt.notifyId = static_cast<int32_t>(resource.notifyId);
    notify.notifyInfo.rmaRt.size = static_cast<uint32_t>(resource.size);
    FillUrmaProtection(notify.notifyInfo.rmaRt.protectionInfo, resource.tokenId, resource.tokenValue);
}

template <typename Resource>
void FillRegedBuffer(RegedBufferEntity& buffer, const Resource& resource)
{
    buffer = {};
    buffer.type = REGED_BUFFER_RMA;
    buffer.bufferInfo.rma.addr = resource.addr;
    buffer.bufferInfo.rma.size = resource.size;
    FillUrmaProtection(buffer.bufferInfo.rma.protectionInfo, resource.tokenId, resource.tokenValue);
}

bool ReadRemoteResources(const std::vector<char>& data, uint32_t count, std::vector<ChannelRemoteResource>& resources)
{
    constexpr size_t RESOURCE_SIZE = sizeof(uint64_t) * 2U + sizeof(uint32_t) * 3U;
    if (count == 0) {
        resources.clear();
        return true;
    }
    const size_t recordSize = data.size() / count;
    if (recordSize < RESOURCE_SIZE) {
        return false;
    }
    resources.resize(count);
    for (size_t index = 0; index < resources.size(); ++index) {
        ChannelEntityReader reader(data.data() + index * recordSize, recordSize);
        auto& resource = resources[index];
        if (!reader.Read(resource.addr) || !reader.Read(resource.size) || !reader.Read(resource.tokenId)
            || !reader.Read(resource.tokenValue) || !reader.Read(resource.notifyId)) {
            return false;
        }
    }
    return true;
}

bool ReadLocalResources(const std::vector<char>& data, uint32_t count, std::vector<ChannelLocalResource>& resources)
{
    constexpr size_t RESOURCE_SIZE = sizeof(uint64_t) * 2U + sizeof(uint32_t) * 2U;
    if (count == 0) {
        resources.clear();
        return true;
    }
    const size_t recordSize = data.size() / count;
    if (recordSize < RESOURCE_SIZE) {
        return false;
    }
    resources.resize(count);
    for (size_t index = 0; index < resources.size(); ++index) {
        ChannelEntityReader reader(data.data() + index * recordSize, recordSize);
        auto& resource = resources[index];
        if (!reader.Read(resource.addr) || !reader.Read(resource.size) || !reader.Read(resource.tokenId)
            || !reader.Read(resource.tokenValue)) {
            return false;
        }
    }
    return true;
}

// 平铺UB连接上下文。契约：
//   sqDepth   —— 控制面乘4填最终WQEBB深度，数据面直用
//   remoteEID —— 控制面逆序为硬件形态(WQE.rmtEid字节序)，数据面零转换直取
//   dieId/funcId —— 存于channel.devInfo(非jfs)，doorbell的唯一来源
bool FillJfsContext(const std::vector<char>& data, HcommJfsContext& jfs, HcommDevInfo& devInfo)
{
    ChannelEntityReader reader(data.data(), data.size());
    uint32_t dieId = 0;
    uint32_t funcId = 0;
    uint32_t serializedSqDepth = 0;
    uint32_t jfcPollMode = 0;
    bool dwqeCacheLocked = false;
    uint64_t dbVa = 0;
    uint64_t sqCiAddr = 0;
    uint8_t localEid[URMA_EID_SIZE]{};
    uint64_t jettyHandle = 0;
    jfs = {};
    if (!reader.Read(dieId) || !reader.Read(funcId) || !reader.Read(jfs.jfsID) || !reader.Read(jfcPollMode)
        || !reader.Read(dwqeCacheLocked) || !reader.Read(dbVa) || !reader.Read(sqCiAddr) || !reader.Read(jfs.sqVa)
        || !reader.Read(serializedSqDepth) || !reader.Read(jfs.tpID) || !reader.ReadBytes(jfs.remoteEID, URMA_EID_SIZE)
        || !reader.ReadBytes(localEid, URMA_EID_SIZE) || !reader.Read(jfs.maxReadSize) || !reader.Read(jfs.maxWriteSize)
        || !reader.Read(jettyHandle)) {
        return false;
    }
    (void)jfcPollMode;
    (void)jettyHandle;
    (void)dwqeCacheLocked; // 契约定案：恒按直写WQE处理，不再传递该标志
    jfs.wqeSize = 64U;
    jfs.dbVa = dbVa;
    jfs.tailAddr = sqCiAddr;
    jfs.sqDepth = serializedSqDepth * UB_WQE_NUM_PER_SQE;
    // 契约(CHANNEL ABI=2起)：remoteEID填最终消费值(硬件逆序形态)，数据面零转换直取。
    // 序列化流中为正序业务形态，与WQE.rmtEid硬件格式(逐字节逆序)对齐在此完成。
    for (uint32_t i = 0; i < URMA_EID_SIZE / 2U; i++) {
        uint8_t eidByte = jfs.remoteEID[i];
        jfs.remoteEID[i] = jfs.remoteEID[URMA_EID_SIZE - 1U - i];
        jfs.remoteEID[URMA_EID_SIZE - 1U - i] = eidByte;
    }
    devInfo.dieId = static_cast<uint8_t>(dieId);
    devInfo.funcId = static_cast<uint8_t>(funcId);
    return true;
}

template <typename T>
HcclResult StageToDevice(const std::vector<T>& source, hcomm::DeviceMem& mem, T*& outPtr)
{
    if (source.empty()) {
        outPtr = nullptr;
        return HCCL_SUCCESS;
    }
    mem = hcomm::DeviceMem::alloc(sizeof(T) * source.size());
    CHK_PRT_RET(
        !mem, HCCL_ERROR("[%s] alloc device memory failed, count[%zu].", __func__, source.size()), HCCL_E_MEMORY);
    CHK_RET(hrtMemSyncCopy(
        mem.ptr(), mem.size(), source.data(), sizeof(T) * source.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    outPtr = static_cast<T*>(mem.ptr());
    return HCCL_SUCCESS;
}

// 把host源数组staging到设备内存并回填设备指针

} // namespace

namespace hcomm {

HcclResult AicpuChannelEntityResources::Build(
    const std::vector<char>& packedChannelData, CommEngine engine, CommProtocol protocol, ChannelHandle originHandle,
    ChannelHandle& entityHandle)
{
    Hccl::AicpuResPackageHelper packageHelper;
    std::vector<char>& packedData = const_cast<std::vector<char>&>(packedChannelData);
    std::vector<Hccl::ModuleData> dataVec = packageHelper.ParsePackedData(packedData);
    CHK_PRT_RET(
        STREAM_RES_TYPE >= dataVec.size(),
        HCCL_ERROR("[%s] invalid packed channel data, size[%zu].", __func__, dataVec.size()), HCCL_E_PARA);

    Hccl::BinaryStream binaryStream(dataVec[STREAM_RES_TYPE].data);
    std::vector<char> transportUniqueId;
    binaryStream >> transportUniqueId;
    CHK_PRT_RET(transportUniqueId.empty(), HCCL_ERROR("[%s] transport unique id is empty.", __func__), HCCL_E_PARA);

    ChannelEntityReader reader(transportUniqueId.data(), transportUniqueId.size());
    uint32_t transportType = 0;
    uint32_t notifyNum = 0;
    uint32_t localBufferNum = 0;
    uint32_t remoteBufferNum = 0;
    uint32_t connectionNum = 0;
    std::vector<char> localNotifyData;
    std::vector<char> remoteNotifyData;
    std::vector<char> localBufferData;
    std::vector<char> remoteBufferData;
    std::vector<char> drainData;
    std::vector<char> connectionData;
    CHK_PRT_RET(
        !reader.Read(transportType) || !reader.Read(notifyNum) || !reader.Read(localBufferNum)
            || !reader.Read(remoteBufferNum) || !reader.Read(connectionNum) || !reader.ReadVector(localNotifyData)
            || !reader.ReadVector(remoteNotifyData) || !reader.ReadVector(localBufferData)
            || !reader.ReadVector(remoteBufferData) || !reader.ReadVector(drainData)
            || !reader.ReadVector(connectionData),
        HCCL_ERROR("[%s] invalid UB transport unique id, size[%zu].", __func__, transportUniqueId.size()), HCCL_E_PARA);
    CHK_PRT_RET(
        connectionNum == 0 || connectionData.size() % connectionNum != 0,
        HCCL_ERROR(
            "[%s] invalid UB connection resource, num[%u], size[%zu].", __func__, connectionNum, connectionData.size()),
        HCCL_E_PARA);
    (void)transportType;

    auto& entity = entity_;
    entity = {};
    entity.compatChannelHandle = originHandle;

    // host侧组装实体源数据；0号槽位固定预留给drain，普通资源从1号槽位开始
    CHK_PRT_RET(
        notifyNum != 0 && localNotifyData.size() / notifyNum < sizeof(uint64_t),
        HCCL_ERROR(
            "[%s] invalid local notify resource size[%zu], num[%u].", __func__, localNotifyData.size(), notifyNum),
        HCCL_E_PARA);
    std::vector<RegedNotifyEntity> localNotifies(1);
    localNotifies[0].type = REGED_NOTIFY_INVALID;
    localNotifies.reserve(static_cast<size_t>(notifyNum) + 1U);
    const size_t localNotifySize = notifyNum == 0 ? 0 : localNotifyData.size() / notifyNum;
    for (uint32_t index = 0; index < notifyNum; ++index) {
        ChannelEntityReader localNotifyReader(
            localNotifyData.data() + static_cast<size_t>(index) * localNotifySize, localNotifySize);
        uint32_t notifyId = 0;
        uint32_t devPhyId = 0;
        CHK_PRT_RET(
            !localNotifyReader.Read(notifyId) || !localNotifyReader.Read(devPhyId),
            HCCL_ERROR("[%s] invalid local notify[%u].", __func__, index), HCCL_E_PARA);
        (void)devPhyId;
        const ChannelRemoteResource res{0, sizeof(uint32_t), 0, 0, notifyId};
        RegedNotifyEntity notify{};
        FillRegedNotify(notify, res);
        localNotifies.push_back(notify);
    }

    std::vector<ChannelRemoteResource> remoteNotifyResources;
    std::vector<ChannelRemoteResource> remoteBufferResources;
    std::vector<ChannelLocalResource> localBufferResources;
    CHK_PRT_RET(
        !ReadRemoteResources(remoteNotifyData, notifyNum, remoteNotifyResources)
            || !ReadLocalResources(localBufferData, localBufferNum, localBufferResources)
            || !ReadRemoteResources(remoteBufferData, remoteBufferNum, remoteBufferResources),
        HCCL_ERROR("[%s] invalid UB buffer or notify resource.", __func__), HCCL_E_PARA);

    std::vector<RegedNotifyEntity> remoteNotifies;
    remoteNotifies.reserve(remoteNotifyResources.size());
    for (const auto& res : remoteNotifyResources) {
        RegedNotifyEntity notify{};
        FillRegedNotify(notify, res);
        remoteNotifies.push_back(notify);
    }

    std::vector<RegedBufferEntity> localBuffers;
    localBuffers.reserve(localBufferResources.size());
    for (const auto& res : localBufferResources) {
        RegedBufferEntity buffer{};
        FillRegedBuffer(buffer, res);
        localBuffers.push_back(buffer);
    }

    // 0号槽位固定预留给drain，普通远端Buffer从1号槽位开始；无drain时保持空槽。
    std::vector<RegedBufferEntity> remoteBuffers(1);
    remoteBuffers[0].type = REGED_BUFFER_INVALID;
    remoteBuffers.reserve(remoteBufferResources.size() + 1U);
    for (const auto& res : remoteBufferResources) {
        RegedBufferEntity buffer{};
        FillRegedBuffer(buffer, res);
        remoteBuffers.push_back(buffer);
    }

    std::vector<SqContext> sqContexts(1);
    sqContexts[0].type = SQ_CONTEXT_TYPE_UB_JFS;
    const size_t connectionSize = connectionData.size() / connectionNum;
    std::vector<char> firstConnection(connectionData.begin(), connectionData.begin() + connectionSize);
    CHK_PRT_RET(
        !FillJfsContext(firstConnection, sqContexts[0].contextInfo.ubJfs, entity.devInfo),
        HCCL_ERROR("[%s] invalid first UB connection resource.", __func__), HCCL_E_PARA);

    // 契约定案：drain资源固定放在localNotifies/remoteBuffers的0号槽位。
    if (!drainData.empty()) {
        std::vector<ChannelRemoteResource> drainResources;
        CHK_PRT_RET(
            !ReadRemoteResources(drainData, 2, drainResources), HCCL_ERROR("[%s] invalid UB drain resource.", __func__),
            HCCL_E_PARA);
        RegedNotifyEntity drainNotify{};
        FillRegedNotify(drainNotify, drainResources[0]);
        localNotifies[0] = drainNotify;
        RegedBufferEntity drainBuffer{};
        FillRegedBuffer(drainBuffer, drainResources[1]);
        remoteBuffers[0] = drainBuffer;
    }

    entity.abiHeader = {HCOMM_CHANNEL_ENTITY_VERSION, HCOMM_CHANNEL_ENTITY_MAGIC_WORD, sizeof(ChannelEntity), 0};
    entity.engine = engine;
    entity.protocol = protocol;
    entity.localNotifyNum = static_cast<uint32_t>(localNotifies.size());
    entity.remoteNotifyNum = static_cast<uint32_t>(remoteNotifies.size());
    entity.localBufferNum = static_cast<uint32_t>(localBuffers.size());
    entity.remoteBufferNum = static_cast<uint32_t>(remoteBuffers.size());
    entity.sqNum = static_cast<uint32_t>(sqContexts.size());
    entity.cqNum = 0;

    CHK_RET(StageToDevice(localNotifies, localNotifies_, entity.localNotifyAddr));
    CHK_RET(StageToDevice(remoteNotifies, remoteNotifies_, entity.remoteNotifyAddr));
    CHK_RET(StageToDevice(localBuffers, localBuffers_, entity.localBufferAddr));
    CHK_RET(StageToDevice(remoteBuffers, remoteBuffers_, entity.remoteBufferAddr));
    CHK_RET(StageToDevice(sqContexts, sqContexts_, entity.sqContextAddr));
    entity.cqContextAddr = nullptr;

    entityMem_ = DeviceMem::alloc(sizeof(ChannelEntity));
    CHK_PRT_RET(!entityMem_, HCCL_ERROR("[%s] alloc device entity failed.", __func__), HCCL_E_MEMORY);
    CHK_RET(hrtMemSyncCopy(
        entityMem_.ptr(), sizeof(ChannelEntity), &entity, sizeof(ChannelEntity),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    entityHandle = reinterpret_cast<ChannelHandle>(entityMem_.ptr());
    return HCCL_SUCCESS;
}

} // namespace hcomm
