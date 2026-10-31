/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "local_rdma_rma_buffer_impl.h"
#include "adapter_hccp.h"
#include "hccl_network.h"
#include "network_manager_pub.h"
#include "mem_mapping_manager.h"

namespace hccl {
namespace {
    void* GetAliasDevAddr(void* addr, const LocalRdmaRmaBufferImpl& parent)
    {
        const uintptr_t childAddr = reinterpret_cast<uintptr_t>(addr);
        const uintptr_t parentAddr = reinterpret_cast<uintptr_t>(parent.GetAddr());
        const uintptr_t parentDevAddr = reinterpret_cast<uintptr_t>(parent.GetDevAddr());
        return reinterpret_cast<void*>(parentDevAddr + (childAddr - parentAddr));
    }
} // namespace

LocalRdmaRmaBufferImpl::LocalRdmaRmaBufferImpl(
    const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType)
    : RmaBuffer(netDevCtx, addr, size, memType, RmaType::RDMA_RMA)
{}

LocalRdmaRmaBufferImpl::LocalRdmaRmaBufferImpl(
    const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType, const LocalRdmaRmaBufferImpl& parent)
    : RmaBuffer(netDevCtx, addr, size, memType, RmaType::RDMA_RMA, true),
      deviceLogicId(parent.deviceLogicId),
      rdmaHandle(parent.rdmaHandle),
      mrHandle(parent.mrHandle),
      lkey(parent.lkey),
      devAddrID(parent.devAddrID),
      initialized_(true)
{
    devAddr = (memType == RmaMemType::DEVICE) ? addr : GetAliasDevAddr(addr, parent);
    HCCL_INFO("[LocalRdmaRmaBufferImpl] alias constructor, lkey[%u] mrHandle[%p]", lkey, mrHandle);
}

LocalRdmaRmaBufferImpl::~LocalRdmaRmaBufferImpl()
{
    if (!isAlias_) {
        HcclResult res = Destroy();
        if (res != HCCL_SUCCESS) {
            HCCL_ERROR("[LocalRdmaRmaBufferImpl][~LocalRdmaRmaBufferImpl]failed, ret[%d]", res);
        }
    }
}
namespace {
    // 同一 (deviceLogicId, devAddrID) 的 MR 注册信息，由引用计数管理生命周期
    struct DevAddrMrInfo {
        u32 refCount{0};
        MrHandle mrHandle{nullptr};
        u32 lkey{0};
    };

    // MR 注册表：对同一 devAddrID 只注册一次 MR，引用计数的增减与「是否最后一个持有者」
    // 的判定在同一临界区内完成，保证并发 Init/Destroy 场景下不重复注册/解注册
    class DevAddrMrRegistry {
    public:
        static DevAddrMrRegistry& GetInstance()
        {
            static DevAddrMrRegistry instance;
            return instance;
        }

        // 查找或注册：命中则复用 MR 并递增引用计数；未命中则注册新 MR 并记录
        HcclResult Acquire(
            s32 deviceLogicId, const std::string& devAddrID, const RdmaHandle rdmaHandle, MrInfoT& mrInfo,
            MrHandle& mrHandle, u32& lkey)
        {
            CHK_PRT_RET(
                deviceLogicId == INVALID_INT,
                HCCL_ERROR("[DevAddrMrRegistry][Acquire]deviceLogicId[%d] is error.", deviceLogicId), HCCL_E_PARA);
            CHK_PRT_RET(devAddrID.empty(), HCCL_ERROR("[DevAddrMrRegistry][Acquire]devAddrID is empty."), HCCL_E_PARA);
            std::lock_guard<std::mutex> lock(mutex_);
            auto& devMap = table_[deviceLogicId];
            auto it = devMap.find(devAddrID);
            if (it != devMap.end()) {
                // 复用已注册 MR：引用计数加一
                ++it->second.refCount;
                mrHandle = it->second.mrHandle;
                lkey = it->second.lkey;
                return HCCL_SUCCESS;
            }
            // 首次注册：注册过程持锁完成，避免并发同 key 重复注册
            CHK_RET(hrtRaRegGlobalMr(rdmaHandle, mrInfo, mrHandle));
            devMap[devAddrID] = DevAddrMrInfo{1, mrHandle, mrInfo.lkey};
            lkey = mrInfo.lkey;
            return HCCL_SUCCESS;
        }

        // 释放引用：递减引用计数并返回是否为最后一个持有者，由调用方决定是否解注册
        HcclResult Release(s32 deviceLogicId, const std::string& devAddrID, bool& isLastOwner)
        {
            isLastOwner = false;
            CHK_PRT_RET(
                deviceLogicId == INVALID_INT,
                HCCL_ERROR("[DevAddrMrRegistry][Release]deviceLogicId[%d] is error.", deviceLogicId), HCCL_E_PARA);
            CHK_PRT_RET(devAddrID.empty(), HCCL_ERROR("[DevAddrMrRegistry][Release]devAddrID is empty."), HCCL_E_PARA);
            std::lock_guard<std::mutex> lock(mutex_);
            auto devIt = table_.find(deviceLogicId);
            CHK_PRT_RET(
                devIt == table_.end(),
                HCCL_ERROR("[DevAddrMrRegistry][Release]deviceLogicId[%d] is not existed.", deviceLogicId),
                HCCL_E_PARA);
            auto it = devIt->second.find(devAddrID);
            CHK_PRT_RET(
                it == devIt->second.end() || it->second.refCount == 0,
                HCCL_ERROR("[DevAddrMrRegistry][Release]devAddrID is not existed."), HCCL_E_PARA);
            if (--it->second.refCount == 0) {
                devIt->second.erase(it);
                isLastOwner = true;
                HCCL_RUN_INFO("Entry-%s: deviceLogicId[%d] erased.", __func__, deviceLogicId);
            }
            return HCCL_SUCCESS;
        }

    private:
        DevAddrMrRegistry() = default;
        std::mutex mutex_;
        std::unordered_map<s32, std::unordered_map<std::string, DevAddrMrInfo>> table_;
    };
} // namespace

HcclResult LocalRdmaRmaBufferImpl::Init()
{
    if (initialized_) {
        HCCL_DEBUG("[Init] LocalRdmaRmaBuffer is initialized.");
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(netDevCtx);
    deviceLogicId = (static_cast<NetDevContext*>(netDevCtx))->GetLogicId();
    HcclIpAddress localIp = (static_cast<NetDevContext*>(netDevCtx))->GetLocalIp();
    bool isBackupIpValid = !(static_cast<NetDevContext*>(netDevCtx))->GetBackupIp().IsInvalid();
    RaResourceInfo raResourceInfo;
    CHK_RET(NetworkManager::GetInstance(deviceLogicId).GetRaResourceInfo(raResourceInfo));
    rdmaHandle = raResourceInfo.nicSocketMap[localIp].nicRdmaHandle;
    CHK_PTR_NULL(rdmaHandle);
    if (isBackupIpValid) {
        HCCL_INFO(
            "[%s] before hrtGetDevice deviceLogicId[%d], isBackupIpValid[%d]", __func__, deviceLogicId,
            isBackupIpValid);
        CHK_RET(hrtGetDevice(&deviceLogicId));
        HCCL_INFO("[%s] after hrtGetDevice deviceLogicId[%d]", __func__, deviceLogicId);
    }
    // host内存地址映射
    devAddr = addr;
    if (memType == RmaMemType::HOST) {
        CHK_RET(MemMappingManager::GetInstance(deviceLogicId).GetDevVA(deviceLogicId, addr, size, devAddr));
    }
    HCCL_DEBUG("[Init]addr[%p], size[%llu], devAddr[%p], memType[%d]", addr, size, devAddr, memType);

    // 内存注册
    MrInfoT info = {};
    info.size = size;
    info.access = RA_ACCESS_REMOTE_WRITE | RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_READ;
    info.addr = devAddr;

    std::ostringstream oss;
    oss.write(reinterpret_cast<const char_t*>(&rdmaHandle), sizeof(rdmaHandle));
    oss.write(reinterpret_cast<const char_t*>(&addr), sizeof(addr));
    oss.write(reinterpret_cast<const char_t*>(&size), sizeof(size));
    devAddrID = oss.str();

    // 内存注册：同一 devAddrID 只注册一次，重复 Init 复用同一 MR 并递增引用计数
    CHK_RET(DevAddrMrRegistry::GetInstance().Acquire(deviceLogicId, devAddrID, rdmaHandle, info, mrHandle, lkey));
    HCCL_DEBUG("[Init][RegMr] LocalRdmaRmaBuffer rdmaHandle[%p], mrHandle[%p].", rdmaHandle, mrHandle);
    // 信息保存
    initialized_ = true;
    return HCCL_SUCCESS;
}

std::string& LocalRdmaRmaBufferImpl::Serialize()
{
    if (!serializeStr_.empty()) {
        return serializeStr_;
    }
    // 序列化信息
    std::ostringstream oss;
    u8 type{static_cast<u8>(rmaType)};
    oss.write(reinterpret_cast<const char_t*>(&type), sizeof(type));
    oss.write(reinterpret_cast<const char_t*>(&addr), sizeof(addr));
    oss.write(reinterpret_cast<const char_t*>(&size), sizeof(size));
    oss.write(reinterpret_cast<const char_t*>(&devAddr), sizeof(devAddr));
    oss.write(reinterpret_cast<const char_t*>(&memType), sizeof(memType));
    oss.write(reinterpret_cast<const char_t*>(&lkey), sizeof(lkey));

    serializeStr_ = oss.str();
    return serializeStr_;
}

HcclResult LocalRdmaRmaBufferImpl::Destroy()
{
    if (addr != nullptr && initialized_) {
        // 内存解注册
        HcclResult ret = HCCL_SUCCESS;
        if (mrHandle != nullptr) {
            HCCL_DEBUG("[Destroy][DeRegMr] LocalRdmaRmaBuffer rdmaHandle[%p], mrHandle[%p].", rdmaHandle, mrHandle);

            // 防止重复释放内存，引用计数递减与「是否最后一个持有者」判定原子完成，仅最后持有者解注册
            bool isLastOwner = false;
            HcclResult retDe = DevAddrMrRegistry::GetInstance().Release(deviceLogicId, devAddrID, isLastOwner);
            if (retDe != HCCL_SUCCESS) {
                HCCL_WARNING("[Destroy][DeRegMr][Release]err[%d] release devAddrMR failed.", retDe);
            }
            if (isLastOwner) {
                ret = hrtRaDeRegGlobalMr(rdmaHandle, mrHandle);
            }

            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[LocalRdmaRmaBufferImpl][Destroy]deReg Global Mr failed, "
                    "ret[%d], dev[%d], ptr[%p], size[%llu]",
                    ret, deviceLogicId, addr, size);
            }
        }

        // host内存解映射
        if (memType == RmaMemType::HOST) {
            ret = MemMappingManager::GetInstance(deviceLogicId).ReleaseDevVA(deviceLogicId, addr, size);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[LocalRdmaRmaBufferImpl][Destroy]release dev va failed, "
                    "ret[%d], dev[%d], ptr[%p], size[%llu]",
                    ret, deviceLogicId, addr, size);
            }
        }

        addr = nullptr;
        size = 0;
        mrHandle = nullptr;
        devAddrID = std::string();
        initialized_ = false;
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult LocalRdmaRmaBufferImpl::Remap(void* addr, u64 length)
{
    CHK_PTR_NULL(addr);
    CHK_PRT_RET(length == 0, HCCL_ERROR("[Remap]memorySize[%llu] must be greater than 0.", length), HCCL_E_PARA);

    struct MemRemapInfo info = {};
    info.addr = addr;
    info.size = length;
    unsigned int num = 1;
    return HrtRaRemapMr(rdmaHandle, &info, num);
}

} // namespace hccl
