/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "hccl_mem_defs.h"
#include "cpu_roce_endpoint.h"
#include "hccl/hccl_res.h"
#include "log.h"
#include "roce_reged_mem_mgr.h"
#include "common/orion_adpt_utils_pub.h"
#include "endpoint_remote_reged_mem_mgr.h"
#include "proc_reged_mem_mgr_cache.h"
#include "base_comm_node_accessors.h"
#include "host_socket_handle_manager.h"
#include "adapter_rts_common.h"
#include "common/dev_type_utils_pub.h"
#include "hccp_peer_manager.h"
#include "server_socket_mgr.h"
#include "rdma_handle_manager.h"
#include "mgr/endpoint_ctx_mgr.h"
#include "hccp.h"

using Hccl::HcclException;
using std::exception;
using std::string;

namespace hcomm {
CpuRoceEndpoint::CpuRoceEndpoint(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

CpuRoceEndpoint::~CpuRoceEndpoint() noexcept
{
    (void)ReleaseCache();
    (void)ReleaseEndpointCtx();
}

bool CpuRoceEndpoint::IsCtxHandleValid() const
{
    if (ctxHandle_ == nullptr) {
        return false;
    }
    return Hccl::RdmaHandleManager::GetInstance().IsHandleValid(static_cast<Hccl::RdmaHandle>(ctxHandle_));
}

HcclResult CpuRoceEndpoint::ReleaseEndpointCtx()
{
    if (endpointCtx_ == nullptr) {
        HCCL_WARNING("[CpuRoceEndpoint][%s] endpointCtx_ is null, nothing to release", __func__);
        return HCCL_E_PTR;
    }
    // 先放掉自身引用，当use_count 只剩 map 持有的 1 份时再触发 EndpointCtxMgr 移除
    EndpointCtxKey key = endpointCtx_->key;
    endpointCtx_.reset();
    GetEndpointCtxMgr(key.devPhyId).Release(key);
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::AttachCache(
    const MemMgrCacheKey& key, const std::function<std::shared_ptr<LocalRegedMemMgr>()>& creator)
{
    cacheKey_ = key;
    cacheAlias_ = GetProcRegedMemMgrCacheHolder();
    // 进程级 mgr 只承担硬件注册与 alias 去重（跨 endpoint 复用）；本实例归属记录见 RecordLocalMemHandle
    localMemMgr_ = cacheAlias_->GetOrCreate(cacheKey_, creator);
    if (localMemMgr_ == nullptr) {
        HCCL_ERROR("[CpuRoceEndpoint][%s] localMemMgr_ is null", __func__);
        CHK_RET(ReleaseCache());
        return HCCL_E_INTERNAL;
    }
    remoteMemMgr_ = std::make_shared<EndpointRemoteRegedMemMgr>(ctxHandle_, ParseRoceMemDesc, CreateRoceRemoteBuffer);
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::RecordLocalMemHandle(void* memHandle)
{
    // 入参校验对齐共享模板 RecordLocalMemHandleImpl 首行（public virtual，防未来调用方传空）
    CHK_PTR_NULL(memHandle);
    // 与共享模板同源的反查：注册在进程级 mgr 完成后反查共享指针，记入本实例归属
    std::shared_ptr<Hccl::LocalRdmaRmaBuffer> record{};
    auto typedMgr = std::dynamic_pointer_cast<RoceRegedMemMgr>(localMemMgr_);
    CHK_PTR_NULL(typedMgr);
    CHK_RET(typedMgr->FindHandleRecord(memHandle, &record));
    // 快照语义：持锁拷贝当前代 → push_back → atomic_store 发布（旧代由在途读者自动保活）。
    // 拷贝可抛 bad_alloc（C API 边界无捕获，逃逸即 terminate）——就地转 HCCL_E_MEMORY；
    // 失败时 memRecordsPublished_ 未动（旧代仍当前代），上层 HcommMemReg 走既有回滚
    try {
        std::lock_guard<std::mutex> lock(localMemMtx_);
        auto next = std::make_shared<MemRecords>(*memRecordsPublished_);
        next->push_back(std::move(record));
        std::atomic_store(&memRecordsPublished_, std::shared_ptr<MemRecordsSnapshot>(std::move(next)));
    } catch (const std::exception& e) {
        HCCL_ERROR(
            "[CpuRoceEndpoint][%s] snapshot copy exception caught:%s, memHandle[%p]", __func__, e.what(), memHandle);
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::RemoveLocalMemHandle(void* memHandle)
{
    // 入参校验对齐共享模板 RemoveLocalMemHandleImpl 首行
    CHK_PTR_NULL(memHandle);
    std::lock_guard<std::mutex> lock(localMemMtx_);
    // 幂等/乱序注销目标态已达成：免 O(n) 拷贝直接返回（指针身份比对不解引用，悬垂句柄安全）
    const auto& cur = *memRecordsPublished_;
    if (std::none_of(cur.begin(), cur.end(), [memHandle](const auto& r) {
            return r.get() == memHandle;
        })) {
        HCCL_ERROR("[CpuRoceEndpoint][%s] memHandle[%p] not registered by this endpoint", __func__, memHandle);
        return HCCL_E_NOT_FOUND;
    }
    // 快照语义：拷贝当前代 → erase 全部指针身份匹配项（与共享模板 remove_if 全匹配语义一致）→ 发布
    try {
        auto next = std::make_shared<MemRecords>(*memRecordsPublished_);
        next->erase(
            std::remove_if(
                next->begin(), next->end(),
                [memHandle](const auto& r) {
                    return r.get() == memHandle;
                }),
            next->end());
        std::atomic_store(&memRecordsPublished_, std::shared_ptr<MemRecordsSnapshot>(std::move(next)));
    } catch (const std::exception& e) {
        HCCL_ERROR(
            "[CpuRoceEndpoint][%s] snapshot copy exception caught:%s, memHandle[%p]", __func__, e.what(), memHandle);
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

HcclResult
CpuRoceEndpoint::FindLocalBuffer(uint64_t addr, uint64_t len, std::shared_ptr<Hccl::LocalRmaBuffer>& buffer) const
{
    buffer.reset();
    // 取不可变快照后无锁遍历（插入序 = 首条命中优先级保持；本次查询内旧代经本副本保活）
    std::shared_ptr<MemRecordsSnapshot> snap = std::atomic_load(&memRecordsPublished_);
    // 保留每个注册句柄的归属，不能因范围相同/包含而丢弃仍需独立注销的记录。
    for (const auto& record : *snap) {
        CHK_SMART_PTR_NULL(record);
        const uint64_t base = record->GetAddr();
        const uint64_t size = record->GetSize();
        if (addr < base) {
            continue;
        }
        const uint64_t offset = addr - base;
        // 使用减法避免 addr + len 或 base + size 溢出造成错误命中。
        if (offset <= size && len <= size - offset) {
            buffer = record;
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_NOT_FOUND;
}

HcclResult CpuRoceEndpoint::GetAllLocalMemHandles(void** memHandles, uint32_t* memHandleNum)
{
    // 入参校验对齐共享模板 GetAllLocalMemHandlesImpl 首行（防传空出参在下方解引用段错误）
    CHK_PTR_NULL(memHandles);
    CHK_PTR_NULL(memHandleNum);
    // keep-alive 钉住当前代：data() 裸指针存活至下次 GetAll——宽于原"锁返回即失效"，
    // "同步立即消费"消费方（MyRank/UT）零改动兼容；并发 GetAll 经写锁串行（keep-alive 为共享出参缓冲）
    std::lock_guard<std::mutex> lock(localMemMtx_);
    getAllKeepAlive_ = std::atomic_load(&memRecordsPublished_);
    *memHandleNum = static_cast<uint32_t>(getAllKeepAlive_->size());
    *memHandles
        = getAllKeepAlive_->empty() ? nullptr : const_cast<void*>(static_cast<const void*>(getAllKeepAlive_->data()));
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::ReleaseCache()
{
    if (cacheAlias_ == nullptr) {
        HCCL_WARNING("[CpuRoceEndpoint][%s] cacheAlias_ is null, nothing to release", __func__);
        return HCCL_E_PTR;
    }
    cacheAlias_->Release(cacheKey_);
    cacheAlias_.reset();
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::Init()
{
    HCCL_INFO("[%s] localEndpoint protocol[%d]", __func__, endpointDesc_.protocol);

    if (endpointDesc_.loc.locType != ENDPOINT_LOC_TYPE_HOST) {
        HCCL_RUN_WARNING(
            "[CpuRoceEndpoint][%s] locType[%s] only support[%s]", __func__,
            GetEndpointLocTypeStr(endpointDesc_.loc.locType), GetEndpointLocTypeStr(ENDPOINT_LOC_TYPE_HOST));
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_RET(CheckDevTypeSupport(
        {DevType::DEV_TYPE_950, DevType::DEV_TYPE_960}, endpointDesc_.protocol, endpointDesc_.loc.locType));

    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr));
    s32 devId = 0;
    CHK_RET(hrtGetDevice(&devId));
    EXCEPTION_CATCH(Hccl::HccpPeerManager::GetInstance().Init(devId), return HCCL_E_INTERNAL);
    u32 devPhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(devId, devPhyId));
    // RdmaHandleManager 调用收口到 EndpointCtxMgr（per-device，经 hcomm::GetEndpointCtxMgr 访问器）
    EndpointCtxKey ctxKey{devPhyId, COMM_PROTOCOL_ROCE, endpointDesc_.loc.locType, ipAddr};
    CHK_RET(GetEndpointCtxMgr(ctxKey.devPhyId).Acquire(ctxKey, false, endpointCtx_));
    CHK_PTR_NULL(endpointCtx_);
    ctxHandle_ = endpointCtx_->ctxHandle;
    HCCL_INFO(
        "CpuRoceEndpoint::%s success, devPhyId[%u], ipAddr[%s], ctxHandle[%p]", __func__, devPhyId,
        ipAddr.Describe().c_str(), ctxHandle_);

    // useAllocMemBase：见 RoceRegedMemMgr 类注（本质 D2N=UB；当前以 lbMax>0 代理，待 HCCP 接口）
    Capabilities caps{};
    HcclResult capRet = GetCapabilities(caps);
    if (capRet != HCCL_SUCCESS) {
        HCCL_ERROR("[CpuRoceEndpoint][%s] GetCapabilities failed, ret[%d]", __func__, capRet);
        CHK_RET(ReleaseEndpointCtx());
        return capRet;
    }
    const bool useAllocMemBase = (caps.lbMax > 0);
    HCCL_INFO("[CpuRoceEndpoint::Init] lbMax[%d] useAllocMemBase[%d]", caps.lbMax, static_cast<int>(useAllocMemBase));

    MemMgrCacheKey key{devPhyId, COMM_PROTOCOL_ROCE, ipAddr, LocTypeToPortType(endpointDesc_.loc.locType)};
    auto createMgr = [this, useAllocMemBase]() {
        return std::make_shared<RoceRegedMemMgr>(ctxHandle_, useAllocMemBase);
    };
    auto ret = AttachCache(key, createMgr);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CpuRoceEndpoint][%s] AttachCache failed, ret[%d]", __func__, ret);
        CHK_RET(ReleaseEndpointCtx());
        return ret;
    }

    // ServerSocketContext：Init 成功路径构造，commAddr 供方法内转换 IpAddress 使用
    serverSocketContext_.emplace(Hccl::ConnectProtoType::RDMA, endpointDesc_.commAddr);
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::GetCapabilities(Capabilities& caps)
{
    HCCL_INFO("[CpuRoceEndpoint::%s] START.", __func__);
    if (!isCapabilitiesAvailable_) {
        // Init 在创建 mem mgr 前就会调用；用 ctxHandle_，不依赖 regedMemMgr_。
        CHK_PTR_NULL(ctxHandle_);
        // RoCE 下 verbs 只上报 max_msg_sz；RdevAttr 中 maxReadSize/maxWriteSize/maxMsgSize
        // 为 union，共用同一个硬件单次传输上限。
        struct RdevAttr rdevAttr {};
        int ret = RaRdevGetAttr(ctxHandle_, &rdevAttr);
        CHK_PRT_RET(
            ret != 0,
            HCCL_ERROR(
                "[CpuRoceEndpoint::GetCapabilities][RaRdevGetAttr]errNo[0x%016llx] RaRdevGetAttr fail. "
                "return[%d], params: rdmaHandle[%p]",
                HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, ctxHandle_),
            HCCL_E_NETWORK);
        // 查询结果为 0 时按约定直接报错，不做 1GiB fallback
        CHK_PRT_RET(
            rdevAttr.maxMsgSize == 0,
            HCCL_ERROR("[CpuRoceEndpoint::GetCapabilities] maxMsgSize = 0, unexpected device capability."),
            HCCL_E_PARA);
        capabilities_.maxMsgSize = rdevAttr.maxMsgSize;
        HCCL_INFO("[CpuRoceEndpoint::GetCapabilities] maxMsgSize[0x%x].", rdevAttr.maxMsgSize);

        capabilities_.lbMax = 0;
        ret = RaGetLbMax(ctxHandle_, &(capabilities_.lbMax));
        HCCL_DEBUG("[CpuRoceEndpoint::GetCapabilities] lbMax = %d.", capabilities_.lbMax);
        CHK_PRT_RET(
            ret != 0,
            HCCL_ERROR(
                "[CpuRoceEndpoint::GetCapabilities][GetLbMax]errNo[0x%016llx] RaGetLbMax fail. "
                "return[%d], params: rdmaHandle[%p], lbMax[%d]",
                HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, ctxHandle_, capabilities_.lbMax),
            HCCL_E_NETWORK);
        isCapabilitiesAvailable_ = true;
    }
    caps = capabilities_;
    HCCL_INFO("[CpuRoceEndpoint::%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}
} // namespace hcomm
