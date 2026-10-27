/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UB_REGED_MEM_MGR_H
#define UB_REGED_MEM_MGR_H

#include <memory>
#include <vector>
#include <string>
#include "endpoint_pair/endpoint_pair_pub.h"
#include "reged_mem_mgr.h"
#include "common/rma_buffer_mgr_pub.h"
#include "buffer_key.h"
#include "local_ub_rma_buffer.h"

namespace hcomm {
/**
 * @note 职责：进程级注册内存管理（仅本端）；远端内存导入/注销由 Endpoint 实例级
 *       wrapper（见 endpoint_remote_reged_mem_mgr.h）承载。
 */
class UbRegedMemMgr : public LocalRegedMemMgr {
public:
    using LocalUbRmaBufferMgr
        = hcomm::RmaBufferMgr<hccl::BufferKey<uintptr_t, u64>, std::shared_ptr<Hccl::LocalUbRmaBuffer>>;

    UbRegedMemMgr(RdmaHandle rdmaHandle);
    ~UbRegedMemMgr() override = default;

    HcclResult RegisterMemory(const HcommMem* mem, const char* memTag, void** memHandle) override;
    HcclResult UnregisterMemory(void* memHandle) override;
    HcclResult
    MemoryExport(const EndpointDesc& endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen) override;
    HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) override;
    // 按 handle 反查注册记录的共享指针，供 endpoint 实例级归属记账
    HcclResult FindHandleRecord(void* memHandle, std::shared_ptr<Hccl::LocalUbRmaBuffer>* out) const;
    HcclResult GetMemDesc(const EndpointDesc endpointDesc, Hccl::LocalUbRmaBuffer* localUbRmaBuffer) const;

    RdmaHandle GetRdmaHandle() const { return rdmaHandle_; }

private:
    RdmaHandle rdmaHandle_{nullptr};
    mutable std::mutex memMtx_;
    std::unique_ptr<LocalUbRmaBufferMgr> localUbRmaBufferMgr_{};
    std::vector<RegedBufferEntry<Hccl::LocalUbRmaBuffer>> allRegisteredBuffers_;
    std::vector<std::shared_ptr<Hccl::LocalUbRmaBuffer>> handlesRecords_;
};
} // namespace hcomm

#endif // UB_REGED_MEM_MGR_H
