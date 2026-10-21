/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROCE_REGED_MEM_MGR_H
#define ROCE_REGED_MEM_MGR_H

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "endpoint_pair.h"
#include "reged_mem_mgr.h"
#include "rma_buffer_mgr.h"
#include "buffer_key.h"
#include "../../../../../legacy/ascend950/unified_platform/resource/buffer/local_rdma_rma_buffer_v2.h"
#include "remote_rma_buffer.h"
#include "exchange_rdma_buffer_dto.h"
#include "local_rdma_rma_buffer.h"

namespace hcomm {
/**
 * @note Host RoCE 本端 MR：
 *  - 非 MemAlloc 路径（HOST / useAllocMemBase_=false / 用户窗超出 AddressRange 上界）：
 *    走原 localRdmaRmaBufferMgr_
 *  - MemAlloc 路径（useAllocMemBase_ && DEVICE && 用户窗落在 AddressRange 内）：
 *    按 allocKey 在 allocToMrMap_ 去重；Unregister 以 handleToAllocKey_ 判别
 *  - VMM 分块映射：单次 aclrtMemGetAddressRange 往往只覆盖一块物理映射，用户窗仅
 *    超出上界时 isInAllocRange=false，回退 legacy。VMM 切窗去重不在本路径处理。
 *
 *  useAllocMemBase_ 含义：
 *  - 本质：Host 网卡与 NPU 之间的 D2N 通路为 UB（非 PCIe）时，Device MR 需按
 *    aclrtMalloc 整段 alloc 去重，否则 ubdevshm 对重叠 VA 二次登记会失败（如 -17）。
 *  - 当前：HCCP 尚无直接查询 D2N 协议类型的接口，暂以 RaGetLbMax()>0 作为 UB 侧
 *    能力代理（现网对应 1825）；后续 HCCP 提供 D2N/UB 判定后应改用该接口注入。
 */
class RoceRegedMemMgr : public LocalRegedMemMgr {
public:
    using MemKey = hccl::BufferKey<uintptr_t, u64>;
    using LocalRdmaRmaBufferMgr
        = hcomm::RmaBufferMgr<hccl::BufferKey<uintptr_t, u64>, std::shared_ptr<Hccl::LocalRdmaRmaBuffer>>;

    // allocKey → 硬件 MR；ref = 挂到该 alloc 的注册次数
    struct AllocMrEntry {
        std::shared_ptr<Hccl::LocalRdmaRmaBuffer> mr; // 非 alias，拥有硬件 MR
        uint64_t ref{0};
    };

    // useAllocMemBase：见类注（D2N=UB 时 DEVICE 按 alloc 去重；当前由 lbMax>0 代理注入）
    RoceRegedMemMgr(RdmaHandle rdmaHandle, bool useAllocMemBase = false);
    ~RoceRegedMemMgr() override;

    HcclResult RegisterMemory(const HcommMem* mem, const char* memTag, void** memHandle) override;
    HcclResult UnregisterMemory(void* memHandle) override;
    HcclResult
    MemoryExport(const EndpointDesc& endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen) override;
    HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) override;
    // 按 handle 反查注册记录的共享指针，供实例级归属记账
    HcclResult FindHandleRecord(void* memHandle, std::shared_ptr<Hccl::LocalRdmaRmaBuffer>* out) const;
    HcclResult GetMemDesc(const EndpointDesc endpointDesc, Hccl::LocalRdmaRmaBuffer* localRdmaRmaBuffer) const;

    RdmaHandle GetRdmaHandle() const { return rdmaHandle_; }

private:
    HcclResult GetMemAllocAddrRange(const HcommMem& mem, MemKey& allocKey, bool& isInAllocRange) const;

    HcclResult
    RegisterByMemAllocAddrRange(const HcommMem& mem, const char* memTag, void** memHandle, const MemKey& allocKey);
    HcclResult UnregisterByMemAllocAddrRange(Hccl::LocalRdmaRmaBuffer* buffer, const MemKey& allocKey);

    HcclResult AcquireAllocMr(const MemKey& allocKey, HcclMemType memType, const char* memTag, bool& createdMr);
    HcclResult ReleaseAllocMrRef(const MemKey& allocKey, uint64_t& allocRefAfter);
    HcclResult PublishMemHandle(
        const HcommMem& mem, const char* memTag, const MemKey& userKey, const MemKey& allocKey, bool createdMr,
        void** memHandle);
    HcclResult UnpublishMemHandle(Hccl::LocalRdmaRmaBuffer* buffer);

    RdmaHandle rdmaHandle_{nullptr};
    mutable std::mutex memMtx_;
    // 见类注：D2N=UB 时启用；当前注入源为 lbMax>0，待 HCCP D2N 查询接口替代
    bool useAllocMemBase_{false};

    // HOST / useAllocMemBase_=false
    std::unique_ptr<LocalRdmaRmaBufferMgr> localRdmaRmaBufferMgr_{};

    // useAllocMemBase_ && DEVICE：allocKey→MR / handle→allocKey
    std::map<MemKey, AllocMrEntry> allocToMrMap_{};
    std::unordered_map<Hccl::LocalRdmaRmaBuffer*, MemKey> handleToAllocKey_{};

    std::vector<RegedBufferEntry<Hccl::LocalRdmaRmaBuffer>> allRegisteredBuffers_;
    std::vector<std::shared_ptr<Hccl::LocalRdmaRmaBuffer>> handlesRecords_;
};
} // namespace hcomm

#endif // ROCE_REGED_MEM_MGR_H
