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
#include "aicpu_channel_process.h"
#include "dev_aicpu_ts_channel_mgr.h"
#include "aicpu_res_package_helper.h"
#include "binary_stream.h"
#include "exception_util.h"
#include "../channel.h"
#include "aicpu_ts_channel_entity.h"
#include "ub_transport_lite_impl.h"
#include "roce_host_loc_transport_lite_impl.h"
#include "p2p_transport_lite_impl.h"
#include "aicpu_task_cache_manager.h"

#include "adapter_rts_common.h"
#include "log.h"

#include <securec.h>

#include <cstdint>
#include <vector>

std::mutex AicpuChannelProcess::mutex_;
std::unordered_map<ChannelHandle, std::unique_ptr<Hccl::BaseTransportLiteImpl>> AicpuChannelProcess::transportMap_;
std::map<std::vector<char>, ChannelHandle> AicpuChannelProcess::uniqueIdHandleMap_;
std::unordered_map<ChannelHandle, std::vector<char>> AicpuChannelProcess::handleUniqueIdMap_;
std::unordered_map<ChannelHandle, uint32_t> AicpuChannelProcess::handleRefCnt_;

namespace hcomm {
bool IsValidLiteTransportHandle(ChannelHandle handle)
{
    // 与 AicpuChannelDestroy 的 erase 互斥：调用方可能并发处于销毁/恢复路径
    std::lock_guard<std::mutex> lock(AicpuChannelProcess::mutex_);
    return AicpuChannelProcess::transportMap_.find(handle) != AicpuChannelProcess::transportMap_.end();
}
} // namespace hcomm

namespace {
// 创建 lite transport 并登记进 transportMap_（handle 即 transport 指针，primitives 层按指针解引用）。
// ROCE 无额外配置走这里；UB 需在入表前设 cache callback，故单独内联。
// 注意：本函数不加锁，调用方必须已持有 AicpuChannelProcess::mutex_。
template <typename T>
HcclResult CreateAndInsertTransport(
    std::vector<char>& uniqueId, ChannelHandle& handle,
    std::unordered_map<ChannelHandle, std::unique_ptr<Hccl::BaseTransportLiteImpl>>& transportMap)
{
    std::unique_ptr<T> impl;
    EXCEPTION_CATCH(impl = std::make_unique<T>(uniqueId), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(impl);
    handle = ReinterpretAs<uint64_t>(impl.get());
    transportMap.insert({handle, std::move(impl)});
    return HCCL_SUCCESS;
}
} // namespace

HcclResult AicpuChannelProcess::ParsePackData(std::vector<char>& data, ChannelHandle& handle)
{
    HCCL_DEBUG("[HcclCommAicpu][AicpuChannelProcess::%s] data: ptr[%p], size[%u]", __func__, data.data(), data.size());
    Hccl::BinaryStream binaryStream(data);

    std::vector<char> transpUniqueId;
    binaryStream >> transpUniqueId;

    Hccl::BinaryStream binaryStreamForType(transpUniqueId);
    u32 transType;
    binaryStreamForType >> transType;
    HCCL_INFO("[CollCommAicpu][AicpuChannelProcess::%s] transType[%u]", __func__, transType);
    if (transType == Hccl::TransportType::UB || transType == Hccl::TransportType::UBoE) {
        std::unique_ptr<Hccl::UbTransportLiteImpl> ubTransportLiteImpl;
        EXCEPTION_CATCH(
            (ubTransportLiteImpl = std::make_unique<Hccl::UbTransportLiteImpl>(transpUniqueId)), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(ubTransportLiteImpl);
        CHK_RET(ubTransportLiteImpl->SetNeedCacheTaskCallback(hcomm::AicpuTaskCacheManager::NeedCacheTask));
        CHK_RET(ubTransportLiteImpl->SetAddWqeArrayCallback(hcomm::AicpuTaskCacheManager::AddWqeArray));
        handle = ReinterpretAs<uint64_t>(ubTransportLiteImpl.get());
        transportMap_.insert({handle, std::move(ubTransportLiteImpl)});
    } else if (transType == Hccl::TransportType::ROCE) {
        // 通道重复注册（如算法切换后HcclChannelAcquire重走）时，底层QP/SQ/CQ环仍保留上一次的
        // 硬件状态（CQ环残留CQE、NIC内PI指针未回绕）。若重建transport，软件影子状态
        // (sqHead_/sqTail_/cqHead_)归零后与硬件状态失配。
        // 因此ROCE按transpUniqueId去重，已注册的通道直接复用原transport对象，保持影子状态连续。
        // 复用判定需同时满足: ①该uniqueId有历史表项; ②表项句柄仍在transportMap_中(未销毁);
        // ③反查该句柄注册时的uniqueId与本次一致(防止句柄地址被新transport复用后串到他通道)
        auto existIt = uniqueIdHandleMap_.find(transpUniqueId);
        if (existIt != uniqueIdHandleMap_.end() && transportMap_.count(existIt->second) != 0) {
            auto revIt = handleUniqueIdMap_.find(existIt->second);
            if (revIt != handleUniqueIdMap_.end() && revIt->second == transpUniqueId) {
                handle = existIt->second;
                // 复用即多一个通道句柄共享该transport, 引用计数+1; 全部共享通道销毁后才真正析构
                handleRefCnt_[handle]++;
                HCCL_INFO(
                    "[CollCommAicpu][ParsePackData] reuse registered roce transport, uniqueId size[%zu], "
                    "handle[0x%llx], refCnt[%u]",
                    transpUniqueId.size(), static_cast<unsigned long long>(handle), handleRefCnt_[handle]);
                return HCCL_SUCCESS;
            }
        }
        CHK_RET(CreateAndInsertTransport<Hccl::RoceHostLocTransportLiteImpl>(transpUniqueId, handle, transportMap_));
        uniqueIdHandleMap_[transpUniqueId] = handle;
        handleUniqueIdMap_[handle] = transpUniqueId;
        handleRefCnt_[handle] = 1;
    } else if (transType == Hccl::TransportType::P2P) {
        CHK_RET(CreateAndInsertTransport<Hccl::P2PTransportLiteImpl>(transpUniqueId, handle, transportMap_));
    } else {
        HCCL_ERROR("[AicpuChannelProcess::%s] transType[%u] is invalid", __func__, transType);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuChannelProcess::InitUrmaChannel(HcclChannelUrmaRes* commParam)
{
    HCCL_INFO(
        "[HcclCommAicpu][AicpuChannelProcess::%s] commParam->uniqueIdAddr[%p], commParam->uniqueIdSize[%u]", __func__,
        commParam->uniqueIdAddr, commParam->uniqueIdSize);

    u8* currentSrcAddr = ReinterpretAs<u8*>(commParam->uniqueIdAddr);
    u32* addSize = ReinterpretAs<u32*>(commParam->channelSizeAddr);
    for (u32 index = 0; index < commParam->listNum; index++) {
        std::vector<char> data(*addSize);

        CHK_SAFETY_FUNC_RET(memcpy_s(data.data(), data.size(), currentSrcAddr, *addSize));
        currentSrcAddr += *addSize;
        addSize++;
        Hccl::AicpuResPackageHelper helper;
        auto dataVec = helper.ParsePackedData(data);

        Hccl::AicpuResMgrType resType = Hccl::AicpuResMgrType::STREAM;
        if (static_cast<u32>(resType) >= dataVec.size()) {
            HCCL_ERROR(
                "[HcclCommAicpu][AicpuChannelProcess::%s] fail, resType[%d], dataVec size[%u]", __func__, resType,
                dataVec.size());
            return HCCL_E_PARA;
        }
        ChannelHandle channelHandle;
        CHK_RET(ParsePackData(dataVec[resType].data, channelHandle));

        if (commParam->ctxList != nullptr) {
            // ctx模式：device侧填充abiHeader + deviceChannel
            auto** ctxList = ReinterpretAs<hcomm::HcommAicpuChannelCtx**>(commParam->ctxList);
            ctxList[index]->abiHeader.version = hcomm::HCOMM_AICPU_CHANNEL_CTX_VERSION;
            ctxList[index]->abiHeader.magicWord = hcomm::HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD;
            ctxList[index]->abiHeader.size = sizeof(hcomm::HcommAicpuChannelCtx);
            ctxList[index]->deviceChannel = ReinterpretAs<void*>(channelHandle);
        } else {
            ChannelHandle* channelList = ReinterpretAs<ChannelHandle*>(commParam->channelList);
            channelList[index] = channelHandle;
        }
        HCCL_INFO(
            "[HcclCommAicpu][AicpuChannelProcess::%s] index[%u], currentSrcAddr[%p], channelSizeAddr[%p], "
            "channelHandle[0x%llx]",
            __func__, index, currentSrcAddr, commParam->channelSizeAddr, channelHandle);
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuChannelProcess::AicpuChannelInit(HcclChannelUrmaRes* commParam)
{
    HCCL_INFO(
        "[AicpuChannelProcess::%s] commParam->channelList[%p], commParam->listNum[%u], commParam->uniqueIdAddr[%p], "
        "commParam->uniqueIdSize[%u]",
        __func__, commParam->channelList, commParam->listNum, commParam->uniqueIdAddr, commParam->uniqueIdSize);

    CHK_RET(hrtSetWorkModeAicpu(true));
    CHK_RET(hrtSetlocalDevice(commParam->deviceLogicId));
    CHK_RET(hrtSetlocalDeviceType(static_cast<DevType>(commParam->deviceType)));

    std::lock_guard<std::mutex> addLock(mutex_);

    HcclResult ret = InitUrmaChannel(commParam);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AicpuChannelProcess::%s] errNo[0x%016llx] Failed to init channels", __func__, HCCL_ERROR_CODE(ret)),
        ret);

    HCCL_INFO("[AicpuChannelProcess::%s] aicpuTask End.", __func__);
    return HCCL_SUCCESS;
}

namespace {

void RollbackDestroy(DevAicpuTsChannelMgr& mgr, const std::vector<ChannelHandle>& rollback)
{
    for (const auto& h : rollback) {
        if (mgr.DestroyChannel(h)) {
            HCCL_DEBUG("[AicpuChannelProcess::%s] rollback destroyed handle[0x%llx]", __func__, h);
        } else {
            HCCL_WARNING("[AicpuChannelProcess::%s] rollback unsuccessful in destroying handle[0x%llx]", __func__, h);
        }
    }
}

HcclResult CreateSingleHcommChannel(
    DevAicpuTsChannelMgr& mgr, void* dp, u64 sz, const HcommDeviceInfo& deviceInfo, hcomm::HcommChannelKind kind,
    HcommChannelRes* commParam, u32 index, ChannelHandle* channelList, std::vector<ChannelHandle>& rollback)
{
    DevAicpuTsChannel* channel = mgr.GetOrCreateAicpuTsChannel(kind);
    if (channel == nullptr) {
        HCCL_ERROR(
            "[AicpuChannelProcess::%s] index[%u] unsupported kind[%u]", __func__, index, static_cast<uint32_t>(kind));
        RollbackDestroy(mgr, rollback);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PTR_NULL(dp);
    ChannelHandle h{};
    HcclResult pret = channel->Create(dp, sz, deviceInfo, h);
    if (pret != HCCL_SUCCESS) {
        HCCL_ERROR("[AicpuChannelProcess::%s] parse fail at index[%u]", __func__, index);
        RollbackDestroy(mgr, rollback);
        return pret;
    }
    if (commParam->ctxList != nullptr) {
        auto** ctxList = ReinterpretAs<hcomm::HcommAicpuChannelCtx**>(commParam->ctxList);
        ctxList[index]->abiHeader.version = hcomm::HCOMM_AICPU_CHANNEL_CTX_VERSION;
        ctxList[index]->abiHeader.magicWord = hcomm::HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD;
        ctxList[index]->abiHeader.size = sizeof(hcomm::HcommAicpuChannelCtx);
        ctxList[index]->deviceChannel = ReinterpretAs<void*>(h);
    } else {
        channelList[index] = h;
    }
    rollback.push_back(h);
    return HCCL_SUCCESS;
}

} // namespace

HcclResult AicpuChannelProcess::InitHcommChannelRes(HcommChannelRes* commParam)
{
    CHK_PTR_NULL(commParam);
    HCCL_INFO(
        "[AicpuChannelProcess::%s] channelList[%p], listNum[%u]", __func__, commParam->channelList, commParam->listNum);

    CHK_PTR_NULL(commParam->channelList);
    CHK_PTR_NULL(commParam->channelDataListAddr);
    CHK_PTR_NULL(commParam->channelDataSizeListAddr);
    CHK_PTR_NULL(commParam->channelTypeListAddr);

    CHK_RET(hrtSetWorkModeAicpu(true));
    CHK_RET(hrtSetlocalDevice(commParam->deviceInfo.deviceLogicId));
    CHK_RET(hrtSetlocalDeviceType(static_cast<DevType>(commParam->deviceInfo.deviceType)));

    void** dataList = ReinterpretAs<void**>(commParam->channelDataListAddr);
    auto* sizeList = ReinterpretAs<u64*>(commParam->channelDataSizeListAddr);
    auto* typeList = ReinterpretAs<u32*>(commParam->channelTypeListAddr);
    auto* channelList = ReinterpretAs<ChannelHandle*>(commParam->channelList);

    auto& mgr = DevAicpuTsChannelMgr::Instance();
    std::vector<ChannelHandle> rollback;
    rollback.reserve(commParam->listNum);

    std::lock_guard<std::mutex> addLock(mutex_);
    for (u32 index = 0; index < commParam->listNum; ++index) {
        hcomm::HcommChannelKind kind = static_cast<hcomm::HcommChannelKind>(typeList[index]);
        CHK_RET(CreateSingleHcommChannel(
            mgr, dataList[index], sizeList[index], commParam->deviceInfo, kind, commParam, index, channelList,
            rollback));
        HCCL_INFO(
            "[AicpuChannelProcess::%s] index[%u] channelHandle[0x%llx]", __func__, index,
            commParam->ctxList != nullptr ? 0 : channelList[index]);
    }

    HCCL_INFO("[AicpuChannelProcess::%s] aicpu_task End.", __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuChannelProcess::AicpuChannelDestroy(HcclChannelUrmaRes* commParam)
{
    HCCL_INFO(
        "[AicpuChannelProcess::%s] commParam->channelList[%p], commParam->listNum[%u]", __func__,
        commParam->channelList, commParam->listNum);

    auto& mgr = DevAicpuTsChannelMgr::Instance();
    std::lock_guard<std::mutex> addLock(mutex_);

    ChannelHandle* channelList = ReinterpretAs<ChannelHandle*>(commParam->channelList);
    HcclResult firstErr = HCCL_SUCCESS;
    uint32_t errCnt = 0;
    for (u32 index = 0; index < commParam->listNum; ++index) {
        ChannelHandle handle = channelList[index];
        // ctx / ChannelEntity 句柄解包为 device 句柄，与注册表/transportMap_ key 对齐；原始句柄原样返回。
        // 解包失败不中断整批销毁：记录首个错误码并跳过本句柄，其余通道继续清理，避免静默泄漏。
        HcclResult unwrapRet = hcomm::UnwrapChannelHandle(handle);
        if (unwrapRet != HCCL_SUCCESS) {
            ++errCnt;
            if (firstErr == HCCL_SUCCESS) {
                firstErr = unwrapRet;
            }
            HCCL_WARNING(
                "[%s] unwrap channel handle[0x%llx] failed, ret[0x%016llx], skip destroy", __func__, channelList[index],
                HCCL_ERROR_CODE(unwrapRet));
            continue;
        }

        auto it = transportMap_.find(handle);
        if (it != transportMap_.end()) {
            // ROCE共享transport: 引用计数递减, 归零才真正析构, 保证其余共享该句柄的通道不悬垂
            auto cntIt = handleRefCnt_.find(handle);
            if (cntIt != handleRefCnt_.end() && --cntIt->second > 0) {
                HCCL_DEBUG(
                    "[AicpuChannelProcess::%s] shared lite transport handle[0x%llx] still in use, refCnt[%u]", __func__,
                    handle, cntIt->second);
                continue;
            }
            handleRefCnt_.erase(handle);
            transportMap_.erase(it);
            // 同步清除uniqueId反查表项, 保证反查表与transportMap_同生命周期:
            // 否则残留表项持有已释放地址, 该地址被后续新transport复用后, 旧uniqueId再注册会串到他人对象
            auto revIt = handleUniqueIdMap_.find(handle);
            if (revIt != handleUniqueIdMap_.end()) {
                uniqueIdHandleMap_.erase(revIt->second);
                handleUniqueIdMap_.erase(revIt);
            }
            HCCL_DEBUG("[AicpuChannelProcess::%s] destroyed lite transport handle[0x%llx]", __func__, handle);
            continue;
        }

        if (mgr.DestroyChannel(handle)) {
            HCCL_DEBUG("[AicpuChannelProcess::%s] destroyed hcomm res handle[0x%llx]", __func__, handle);
            continue;
        }

        HCCL_WARNING(
            "[AicpuChannelProcess::%s] handle[0x%llx] not found in ub/hcomm maps, maybe already destroyed?", __func__,
            handle);
    }

    if (errCnt > 0) {
        HCCL_ERROR(
            "[%s] destroy channel failed, errCnt[%u], firstErr[0x%016llx].", __func__, errCnt,
            HCCL_ERROR_CODE(firstErr));
        return firstErr;
    }

    HCCL_INFO("[AicpuChannelProcess::%s] aicpu_task End.", __func__);
    return HCCL_SUCCESS;
}
