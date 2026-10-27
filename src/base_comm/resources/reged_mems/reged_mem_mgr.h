/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REGED_MEM_MGR_H
#define REGED_MEM_MGR_H

#include "common/cast_utils_pub.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include "hcomm_c_adpt.h"
#include "log.h"
#include "buffer_key.h"
#include "buffer.h"
#include "serializable.h"
#include <unistd.h>

using RdmaHandle = void*;

namespace hcomm {

/**
 * @note 职责：本端内存接口（注册/注销/导出/枚举）。实现方为进程级 mgr
 *       （RoceRegedMemMgr/UbRegedMemMgr/UbMemRegedMemMgr）或同时管两端的组合 mgr
 *       （HccsRegedMemMgr/AicpuTsRoceRegedMemMgr/PluginRegedMemMgr）。
 */
class LocalRegedMemMgr {
public:
    LocalRegedMemMgr() = default;
    virtual ~LocalRegedMemMgr() = default;

    // 注册内存
    virtual HcclResult RegisterMemory(const HcommMem* mem, const char* memTag, void** memHandle) = 0;

    // 注销内存
    virtual HcclResult UnregisterMemory(void* memHandle) = 0;

    // 导出指定内存描述，用于交换
    virtual HcclResult
    MemoryExport(const EndpointDesc& endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen)
        = 0;

    virtual HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) = 0;

protected:
    template <typename RmaBuffer>
    using RegedBufferEntry = std::pair<std::shared_ptr<RmaBuffer>, bool>;

    static HcclResult ValidateMemParams(HcommMem mem, void* const* memHandle)
    {
        CHK_PTR_NULL(memHandle);
        CHK_PTR_NULL(mem.addr);
        CHK_PRT_RET(mem.size == 0, HCCL_ERROR("[%s] mem size is zero", __func__), HCCL_E_PARA);
        CHK_PRT_RET(
            mem.type == COMM_MEM_TYPE_INVALID, HCCL_ERROR("[%s] invalid mem type [%d]", __func__, mem.type),
            HCCL_E_PARA);
        return HCCL_SUCCESS;
    }

    // MemoryExport: 从allBuffers中校验memHandle并获取buffer指针
    template <typename RmaBuffer>
    static HcclResult ValidateMemExportHandle(
        void* memHandle, const std::vector<RegedBufferEntry<RmaBuffer>>& allBuffers, RmaBuffer*& outBuffer)
    {
        auto it = std::find_if(allBuffers.begin(), allBuffers.end(), [memHandle](const auto& entry) {
            return entry.first != nullptr && entry.first.get() == memHandle;
        });
        if (it == allBuffers.end()) {
            HCCL_ERROR("[RegedMemMgr][MemoryExport] memHandle[%p] is not registered.", memHandle);
            return HCCL_E_NOT_FOUND;
        }
        outBuffer = it->first.get();
        return HCCL_SUCCESS;
    }

    // ---- 底层：tree 操作 ----

    // 用实际注册后的addr/size做key，对buffer增加引用计数
    template <typename Mgr, typename BufferPtr>
    static HcclResult AddBuffer(Mgr& mgr, const BufferPtr& buffer)
    {
        hccl::BufferKey<uintptr_t, u64> actualRegKey(
            ReinterpretAs<uintptr_t>(buffer->GetAddr()), static_cast<uint64_t>(buffer->GetSize()));
        EXCEPTION_CATCH((void)mgr->AddWithoutCheck(actualRegKey, buffer), return HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    // UnregisterMemory: IsAlias为true时，通过硬件句柄定位父buffer
    template <typename Mgr, typename BufferPtr, typename BufferVec, typename HwHandleFn, typename EqualFn>
    static BufferPtr ResolveAliasParent(
        Mgr& mgr, const hccl::BufferKey<uintptr_t, u64>& ownKey, BufferPtr buffer, BufferVec& allBuffers,
        HwHandleFn&& hwHandleGetter, EqualFn&& tokenEqual)
    {
        auto token = hwHandleGetter(buffer);
        auto findResult = mgr->Find(ownKey);
        if (findResult.first && tokenEqual(hwHandleGetter(findResult.second.get()), token)) {
            return findResult.second.get();
        }
        for (auto& entry : allBuffers) {
            const auto& ptr = entry.first;
            if (ptr.get() == buffer) {
                continue;
            }
            if (tokenEqual(hwHandleGetter(ptr.get()), token) && !ptr->IsAlias()) {
                return ptr.get();
            }
        }
        return nullptr;
    }

    // ---- 中层：注册/注销 核心逻辑 ----

    // Find命中则基于父buffer构造别名，未命中则构造新buffer并注册
    template <typename Mgr, typename FindResult, typename BufferPtr, typename MakeAlias, typename MakeNew>
    static HcclResult
    RegisterOrAlias(Mgr& mgr, const FindResult& findPair, BufferPtr& buffer, MakeAlias&& makeAlias, MakeNew&& makeNew)
    {
        if (findPair.first) {
            auto parentBuffer = findPair.second;
            EXCEPTION_CATCH((buffer = makeAlias(parentBuffer)), return HCCL_E_PTR);
            CHK_RET(AddBuffer(mgr, parentBuffer));
        } else {
            EXCEPTION_CATCH((buffer = makeNew()), return HCCL_E_PTR);
            CHK_RET(AddBuffer(mgr, buffer));
        }
        return HCCL_SUCCESS;
    }

    // ---- 上层：RegisterMemory / UnregisterMemory 通用实现 ----

    // 构造基类Buffer → RegisterOrAlias → 写回memHandle
    // makeAlias(bufPtr, parent) — 基于父buffer构造别名
    // makeNew(bufPtr)          — 构造新buffer
    // outRecords               — 可选的记录向量，注册成功时追加
    template <typename Mgr, typename RmaBuffer, typename MakeAlias, typename MakeNew>
    static HcclResult RegisterMemoryImpl(
        HcommMem mem, const char* memTag, void** memHandle, Mgr& mgr,
        std::vector<RegedBufferEntry<RmaBuffer>>& allBuffers, std::vector<std::shared_ptr<RmaBuffer>>* outRecords,
        const char* logTag, MakeAlias&& makeAlias, MakeNew&& makeNew)
    {
        CHK_RET(ValidateMemParams(mem, memHandle));

        hccl::BufferKey<uintptr_t, u64> tempKey(ReinterpretAs<uintptr_t>(mem.addr), mem.size);
        auto findPair = mgr->Find(tempKey);

        std::shared_ptr<Hccl::Buffer> localBufferPtr = nullptr;
        EXCEPTION_CATCH(
            (localBufferPtr = std::make_shared<Hccl::Buffer>(
                 ReinterpretAs<uintptr_t>(mem.addr), mem.size,
                 (mem.type == COMM_MEM_TYPE_HOST) ? HcclMemType::HCCL_MEM_TYPE_HOST : HcclMemType::HCCL_MEM_TYPE_DEVICE,
                 memTag)),
            return HCCL_E_PTR);

        std::shared_ptr<RmaBuffer> rmaBuffer;
        CHK_RET(RegisterOrAlias(
            mgr, findPair, rmaBuffer,
            [&localBufferPtr, &makeAlias](auto& parent) {
                return makeAlias(localBufferPtr, parent);
            },
            [&localBufferPtr, &makeNew]() {
                return makeNew(localBufferPtr);
            }));

        HCCL_INFO("[%s][RegisterMemory] success, key {%p, %llu}", logTag, mem.addr, mem.size);
        *memHandle = static_cast<void*>(rmaBuffer.get());
        allBuffers.emplace_back(rmaBuffer, false);
        if (outRecords != nullptr) {
            outRecords->push_back(rmaBuffer);
        }
        return HCCL_SUCCESS;
    }

    // IsAlias → ResolveAliasParent → Del → erase
    // hwHandleGetter(buffer) — 获取硬件句柄用于ResolveAliasParent
    // tokenEqual(lhs, rhs)   — 比较两个句柄是否相等
    // outRecords             — 可选的记录向量，注销成功时移除
    template <typename Mgr, typename RmaBuffer, typename HwHandleFn, typename EqualFn>
    static HcclResult UnregisterMemoryImpl(
        void* memHandle, Mgr& mgr, std::vector<RegedBufferEntry<RmaBuffer>>& allBuffers,
        std::vector<std::shared_ptr<RmaBuffer>>* outRecords, HwHandleFn&& hwHandleGetter, EqualFn&& tokenEqual)
    {
        CHK_PTR_NULL(memHandle);
        RmaBuffer* buffer = static_cast<RmaBuffer*>(memHandle);
        CHK_PTR_NULL(buffer);
        auto bufferInfo = buffer->GetBufferInfo();

        hccl::BufferKey<uintptr_t, u64> ownKey(bufferInfo.first, bufferInfo.second);
        RmaBuffer* refBuffer = buffer;
        if (buffer->IsAlias()) {
            refBuffer = ResolveAliasParent(
                mgr, ownKey, buffer, allBuffers, std::forward<HwHandleFn>(hwHandleGetter),
                std::forward<EqualFn>(tokenEqual));
            if (refBuffer == nullptr) {
                HCCL_ERROR("[UnregisterMemory] alias parent not found");
                return HCCL_E_NOT_FOUND;
            }
        }

        auto refBufferInfo = refBuffer->GetBufferInfo();
        hccl::BufferKey<uintptr_t, u64> tempKey(refBufferInfo.first, refBufferInfo.second);
        // Del returns false when ref remains nonzero; local unregister still succeeds after erasing this handle.
        EXCEPTION_CATCH((void)mgr->Del(tempKey), return HCCL_E_NOT_FOUND);
        // IsInTree判断tree中是否还有该key的引用
        auto it
            = std::find_if(allBuffers.begin(), allBuffers.end(), [buffer](const RegedBufferEntry<RmaBuffer>& entry) {
                  return entry.first.get() == buffer;
              });
        if (it != allBuffers.end()) {
            auto bufferPtr = it->first;
            if (!mgr->IsInTree(ownKey)) {
                allBuffers.erase(it);
            } else {
                it->second = true;
            }
            if (outRecords != nullptr) {
                outRecords->erase(std::remove(outRecords->begin(), outRecords->end(), bufferPtr), outRecords->end());
            }
        }
        return HCCL_SUCCESS;
    }
};

/**
 * @note 职责：远端内存接口（导入/关闭）。实现方为 Endpoint 实例级远端管理
 *       （EndpointRemoteRegedMemMgr）或同时管两端的组合 mgr（HccsRegedMemMgr 等）。
 */
class RemoteRegedMemMgr {
public:
    RemoteRegedMemMgr() = default;
    virtual ~RemoteRegedMemMgr() = default;

    // 基于内存描述导入远端内存
    virtual HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) = 0;

    // 关闭导入的远端内存
    virtual HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) = 0;
};

// 远端接口转发器：本端+远端一体的组合型 mgr（HccsRegedMemMgr 等）只继承 LocalRegedMemMgr
// （避免多继承的菱形风险），其 MemoryImport/Unimport 经此包装成 RemoteRegedMemMgr 视图供
// Endpoint 的 GetRemoteRegMemMgr() 返回。回调捕获宿主 endpoint/mgr 指针，随宿主销毁失效。
class RemoteRegedMemMgrForwarder : public RemoteRegedMemMgr {
public:
    RemoteRegedMemMgrForwarder(
        std::function<HcclResult(const void*, uint32_t, HcommMem*)> importFn,
        std::function<HcclResult(const void*, uint32_t)> unimportFn)
        : importFn_(std::move(importFn)),
          unimportFn_(std::move(unimportFn))
    {}

    HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) override
    {
        return importFn_(memDesc, descLen, outMem);
    }

    HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) override { return unimportFn_(memDesc, descLen); }

private:
    std::function<HcclResult(const void*, uint32_t, HcommMem*)> importFn_;
    std::function<HcclResult(const void*, uint32_t)> unimportFn_;
};

// memDesc 来自对端进程、长度不可信：解析前以此上限拦截异常大的 descLen，防恶意输入触发大额内存分配。
constexpr uint32_t MAX_MEM_DESC_LEN = 1024;

// memDesc 布局：[DTO 序列化数据] + [EndpointDesc] + [uint64 pid]；导出侧 BuildMemDesc、解析侧
// ParseMemDesc 在此同文件成对维护，改布局必须两处同步。pid 用于区分同一 endpointDesc 下不同进程导出的内存。
inline HcclResult BuildMemDesc(Hccl::Serializable& dto, const EndpointDesc& endpointDesc, std::vector<char>& memDesc)
{
    Hccl::BinaryStream dtoStream;
    dto.Serialize(dtoStream);
    dtoStream.Dump(memDesc);
    if (memDesc.empty()) {
        HCCL_ERROR("[RegedMemMgr][BuildMemDesc] dto serialize failed.");
        return HCCL_E_INTERNAL;
    }

    // 一次扩出 desc + pid 两段，分别从段首拷贝，避免两次 resize 各算偏移
    memDesc.resize(memDesc.size() + sizeof(EndpointDesc) + sizeof(uint64_t));
    auto* descSeg = memDesc.data() + memDesc.size() - sizeof(EndpointDesc) - sizeof(uint64_t);
    if (memcpy_s(descSeg, sizeof(EndpointDesc), &endpointDesc, sizeof(EndpointDesc)) != EOK) {
        HCCL_ERROR("[RegedMemMgr][BuildMemDesc] endpointDesc memcpy_s failed.");
        return HCCL_E_INTERNAL;
    }
    const uint64_t pid = static_cast<uint64_t>(getpid());
    if (memcpy_s(descSeg + sizeof(EndpointDesc), sizeof(uint64_t), &pid, sizeof(uint64_t)) != EOK) {
        HCCL_ERROR("[RegedMemMgr][BuildMemDesc] pid memcpy_s failed.");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

inline HcclResult
ParseMemDesc(const void* memDesc, uint32_t descLen, EndpointDesc& endpointDesc, uint64_t& pid, Hccl::Serializable& dto)
{
    const char* description = static_cast<const char*>(memDesc);
    const uint32_t minLen = static_cast<uint32_t>(sizeof(EndpointDesc) + sizeof(uint64_t));
    CHK_PRT_RET(
        descLen < minLen,
        HCCL_ERROR(
            "[RegedMemMgr][ParseMemDesc] descLen[%u] too small, expected at least[%u]; "
            "old-format desc without pid is not supported.",
            descLen, minLen),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        descLen > MAX_MEM_DESC_LEN,
        HCCL_ERROR("[RegedMemMgr][ParseMemDesc] descLen[%u] exceeds limit[%u].", descLen, MAX_MEM_DESC_LEN),
        HCCL_E_INTERNAL);

    const size_t pidOffset = descLen - sizeof(uint64_t);
    if (memcpy_s(&pid, sizeof(pid), description + pidOffset, sizeof(uint64_t)) != EOK) {
        HCCL_ERROR("[RegedMemMgr][ParseMemDesc] pid copy error.");
        return HCCL_E_INTERNAL;
    }
    if (memcpy_s(
            &endpointDesc, sizeof(EndpointDesc), description + pidOffset - sizeof(EndpointDesc), sizeof(EndpointDesc))
        != EOK) {
        HCCL_ERROR("[RegedMemMgr][ParseMemDesc] endpointDesc copy error.");
        return HCCL_E_INTERNAL;
    }

    std::vector<char> dtoBytes(description, description + pidOffset - sizeof(EndpointDesc));
    Hccl::BinaryStream dtoStream(dtoBytes);
    // memDesc 来自对端进程，字节不可信：DTO 中的长度前缀字段（如 string）被污染时可能抛异常，兜底转错误码
    EXCEPTION_CATCH(dto.Deserialize(dtoStream), return HCCL_E_INTERNAL);

    // 回读校验：解析出的 DTO 重新序列化后必须与原始 DTO 段字节一致。旧格式 desc（无 pid）按新布局
    // 解析会整体错位，回读必然不一致，要拦截，避免内存挂错归属键且无法 Unimport。
    Hccl::BinaryStream verifyStream;
    dto.Serialize(verifyStream);
    std::vector<char> verifyBytes;
    verifyStream.Dump(verifyBytes);
    CHK_PRT_RET(
        verifyBytes.size() != dtoBytes.size() || memcmp(verifyBytes.data(), dtoBytes.data(), dtoBytes.size()) != 0,
        HCCL_ERROR(
            "[RegedMemMgr][ParseMemDesc] memDesc format mismatch: expect[Dto+EndpointDesc+pid], "
            "dto re-serialize check failed, descLen[%u].",
            descLen),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}
} // namespace hcomm
#endif // REGED_MEM_MGR_H
