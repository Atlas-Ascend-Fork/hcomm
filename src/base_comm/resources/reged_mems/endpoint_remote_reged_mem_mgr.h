/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENDPOINT_REMOTE_REGED_MEM_MGR_H
#define ENDPOINT_REMOTE_REGED_MEM_MGR_H

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include "hccl_common.h"
#include "log.h"
#include "reged_mem_mgr.h"
#include "remote_mem_owner_key.h"
#include "common/rma_buffer_mgr_pub.h"
#include "buffer_key.h"
#include "remote_rma_buffer.h"
#include "exchange_rdma_buffer_dto.h"
#include "exchange_ub_buffer_dto.h"

namespace hcomm {
/**
 * @note 职责：Endpoint 实例级远端内存管理（仅远端接口 RemoteRegedMemMgr 的唯一实现）。
 *       每个 Endpoint 实例自持一份远端导入记录，引用计数收缩到实例内，解决多个
 *       endpoint 实例共享进程级 mgr 时同 desc 同内存重复导入报 E_AGAIN 及引用泄漏问题。
 *       本端内存（注册/注销/导出）不在这里——Endpoint 的 localMemMgr_ 直达进程级 mgr。
 *       RoCE/Ub 两族的差异（DTO 类型、远端缓冲类型）通过 MemDescParser/RemoteBufferCreator
 *       注入，见文件尾部的两族适配函数。
 */
class EndpointRemoteRegedMemMgr : public RemoteRegedMemMgr {
public:
    // 解析 memDesc：填归属键、缓冲键和可获取的内存类型，返回反序列化后的 DTO
    using MemDescParser = std::function<HcclResult(
        const void* memDesc, uint32_t descLen, RemoteMemOwnerKey& ownerKey, hccl::BufferKey<uintptr_t, u64>& bufferKey,
        CommMemType& memType, std::shared_ptr<Hccl::Serializable>& dto)>;
    // 从 DTO 构造族对应的远端缓冲
    using RemoteBufferCreator
        = std::function<std::shared_ptr<Hccl::RemoteRmaBuffer>(RdmaHandle rdmaHandle, const Hccl::Serializable& dto)>;

    using RemoteRmaBufferMgr
        = hcomm::RmaBufferMgr<hccl::BufferKey<uintptr_t, u64>, std::shared_ptr<Hccl::RemoteRmaBuffer>>;

    EndpointRemoteRegedMemMgr(RdmaHandle rdmaHandle, MemDescParser parser, RemoteBufferCreator creator)
        : rdmaHandle_(rdmaHandle),
          parser_(std::move(parser)),
          creator_(std::move(creator))
    {}

    ~EndpointRemoteRegedMemMgr() override = default;

    HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) override
    {
        HCCL_INFO("[%s] Begin", __func__);
        CHK_PTR_NULL(outMem);
        outMem->addr = nullptr;
        outMem->size = 0;
        if (memDesc == nullptr) {
            outMem->type = COMM_MEM_TYPE_INVALID;
        }
        CHK_PTR_NULL(memDesc);
        std::lock_guard<std::mutex> lock(memMtx_);

        RemoteMemOwnerKey ownerKey{};
        hccl::BufferKey<uintptr_t, u64> key(0, 0);
        CommMemType memType = COMM_MEM_TYPE_INVALID;
        std::shared_ptr<Hccl::Serializable> dto;
        const HcclResult ret = parser_(memDesc, descLen, ownerKey, key, memType, dto);
        if (ret != HCCL_SUCCESS) {
            outMem->type = COMM_MEM_TYPE_INVALID;
            HCCL_ERROR("[EndpointRemoteRegedMemMgr][MemoryImport] parse memory descriptor failed, ret[%d].", ret);
            return ret;
        }
        auto it = remoteRmaBufferMgrs_.find(ownerKey);
        if (it == remoteRmaBufferMgrs_.end()) {
            it = remoteRmaBufferMgrs_.emplace(ownerKey, std::make_unique<RemoteRmaBufferMgr>()).first;
        }
        CHK_SMART_PTR_NULL(it->second);

        // 先精确查已导入条目：命中则直接 ref+1 复用，跳过 creator_（Ub 族 creator 含硬件 import，
        // 重复导入时不做无意义的硬件 import+unimport 空转）；未命中才构造新缓冲
        auto findRef = it->second->FindAndRef(key);
        if (findRef.first) {
            if (memType != COMM_MEM_TYPE_INVALID) {
                const CommMemType cachedMemType = ToCommMemType(findRef.second->GetMemType());
                if (cachedMemType != memType) {
                    // FindAndRef 已增加引用计数，类型冲突返回前需对称撤销。
                    EXCEPTION_CATCH(static_cast<void>(it->second->Del(key)), return HCCL_E_INTERNAL);
                    HCCL_ERROR(
                        "[EndpointRemoteRegedMemMgr][MemoryImport] reused buffer memory type mismatch, "
                        "pid[%llu], key[%s], descType[%d], cachedType[%d].",
                        ownerKey.pid, key.ToString().c_str(), static_cast<int32_t>(memType),
                        static_cast<int32_t>(cachedMemType));
                    outMem->type = COMM_MEM_TYPE_INVALID;
                    return HCCL_E_PARA;
                }
            }
            outMem->addr = reinterpret_cast<void*>(findRef.second->GetAddr());
            outMem->size = findRef.second->GetSize();
            if (memType != COMM_MEM_TYPE_INVALID) {
                outMem->type = memType;
            }
            HCCL_INFO(
                "[EndpointRemoteRegedMemMgr][MemoryImport] reuse imported buffer, pid[%llu], key[%s].", ownerKey.pid,
                key.ToString().c_str());
            return HCCL_SUCCESS;
        }

        std::shared_ptr<Hccl::RemoteRmaBuffer> remoteBuffer;
        EXCEPTION_CATCH(remoteBuffer = creator_(rdmaHandle_, *dto), return HCCL_E_PTR);

        auto resultPair = it->second->Add(key, remoteBuffer);
        if (resultPair.first == it->second->End()) {
            HCCL_ERROR(
                "[EndpointRemoteRegedMemMgr][MemoryImport] key[%s] overlaps with existing imported buffer, pid[%llu].",
                key.ToString().c_str(), ownerKey.pid);
            return HCCL_E_INTERNAL;
        }
        // 新导入路径（重复导入已在 FindAndRef 命中时提前返回）
        outMem->addr = reinterpret_cast<void*>(remoteBuffer->GetAddr());
        outMem->size = remoteBuffer->GetSize();
        if (memType != COMM_MEM_TYPE_INVALID) {
            outMem->type = memType;
        }
        HCCL_INFO(
            "[EndpointRemoteRegedMemMgr][MemoryImport] success, pid[%llu], key[%s], newlyAdded.", ownerKey.pid,
            key.ToString().c_str());
        return HCCL_SUCCESS;
    }

    HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) override
    {
        HCCL_INFO("[%s] Begin", __func__);
        CHK_PTR_NULL(memDesc);
        std::lock_guard<std::mutex> lock(memMtx_);

        RemoteMemOwnerKey ownerKey{};
        hccl::BufferKey<uintptr_t, u64> key(0, 0);
        CommMemType memType = COMM_MEM_TYPE_INVALID;
        std::shared_ptr<Hccl::Serializable> dto;
        CHK_RET(parser_(memDesc, descLen, ownerKey, key, memType, dto));

        auto it = remoteRmaBufferMgrs_.find(ownerKey);
        if (it == remoteRmaBufferMgrs_.end()) {
            HCCL_ERROR(
                "[EndpointRemoteRegedMemMgr][MemoryUnimport] remote buffer manager not found, pid[%llu].",
                ownerKey.pid);
            return HCCL_E_NOT_FOUND;
        }

        bool deleted = false;
        EXCEPTION_CATCH(deleted = it->second->Del(key), return HCCL_E_NOT_FOUND);
        if (!deleted) {
            // 实例内仍有其他使用者引用，仅递减引用计数
            HCCL_INFO(
                "[EndpointRemoteRegedMemMgr][MemoryUnimport] memory reference count is larger than 0, key[%s].",
                key.ToString().c_str());
        }
        if (it->second->size() == 0) {
            remoteRmaBufferMgrs_.erase(it);
        }
        return HCCL_SUCCESS;
    }

private:
    static CommMemType ToCommMemType(HcclMemType memType)
    {
        if (memType == HCCL_MEM_TYPE_DEVICE) {
            return COMM_MEM_TYPE_DEVICE;
        }
        if (memType == HCCL_MEM_TYPE_HOST) {
            return COMM_MEM_TYPE_HOST;
        }
        return COMM_MEM_TYPE_INVALID;
    }

    // 按 {endpointDesc, pid} 分组保存本实例已导入的远端缓冲
    std::unordered_map<RemoteMemOwnerKey, std::unique_ptr<RemoteRmaBufferMgr>, RemoteMemOwnerKeyHash>
        remoteRmaBufferMgrs_{};
    RdmaHandle rdmaHandle_{nullptr}; // 本 Endpoint 实例的 rdma 句柄，导入远端内存时用来建连接
    MemDescParser parser_{};
    RemoteBufferCreator creator_{};
    mutable std::mutex memMtx_{};
};

// ---- RoCE 族适配：memDesc 解析 + 远端缓冲构造 ----
inline HcclResult ParseRoceMemDesc(
    const void* memDesc, uint32_t descLen, RemoteMemOwnerKey& ownerKey, hccl::BufferKey<uintptr_t, u64>& bufferKey,
    CommMemType&, std::shared_ptr<Hccl::Serializable>& dto)
{
    // RoCE 历史 memDesc 不携带内存类型，不返回 memType，保持修改前的行为。
    auto roceDto = std::make_shared<Hccl::ExchangeRdmaBufferDto>();
    CHK_RET(ParseMemDesc(memDesc, descLen, ownerKey.desc, ownerKey.pid, *roceDto));
    bufferKey = hccl::BufferKey<uintptr_t, u64>(static_cast<uintptr_t>(roceDto->addr), roceDto->size);
    dto = std::move(roceDto);
    return HCCL_SUCCESS;
}

inline std::shared_ptr<Hccl::RemoteRmaBuffer>
CreateRoceRemoteBuffer(RdmaHandle rdmaHandle, const Hccl::Serializable& dto)
{
    return std::make_shared<Hccl::RemoteRdmaRmaBuffer>(rdmaHandle, dto);
}

// ---- Ub 族适配：memDesc 解析 + 远端缓冲构造 ----
inline HcclResult ParseUbMemDesc(
    const void* memDesc, uint32_t descLen, RemoteMemOwnerKey& ownerKey, hccl::BufferKey<uintptr_t, u64>& bufferKey,
    CommMemType& memType, std::shared_ptr<Hccl::Serializable>& dto)
{
    auto ubDto = std::make_shared<Hccl::ExchangeUbBufferDto>();
    CHK_RET(ParseMemDesc(memDesc, descLen, ownerKey.desc, ownerKey.pid, *ubDto));
    if (ubDto->memType == HCCL_MEM_TYPE_DEVICE) {
        memType = COMM_MEM_TYPE_DEVICE;
    } else if (ubDto->memType == HCCL_MEM_TYPE_HOST) {
        memType = COMM_MEM_TYPE_HOST;
    } else {
        HCCL_ERROR("[ParseUbMemDesc] invalid memory type[%d].", static_cast<int32_t>(ubDto->memType));
        return HCCL_E_PARA;
    }
    bufferKey = hccl::BufferKey<uintptr_t, u64>(static_cast<uintptr_t>(ubDto->addr), ubDto->size);
    dto = std::move(ubDto);
    return HCCL_SUCCESS;
}

inline std::shared_ptr<Hccl::RemoteRmaBuffer> CreateUbRemoteBuffer(RdmaHandle rdmaHandle, const Hccl::Serializable& dto)
{
    return std::make_shared<Hccl::RemoteUbRmaBuffer>(rdmaHandle, dto);
}
} // namespace hcomm

#endif // ENDPOINT_REMOTE_REGED_MEM_MGR_H
