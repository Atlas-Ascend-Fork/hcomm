/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_DPU_MANAGER_H
#define HCCL_DPU_MANAGER_H

#include "acl/acl_rt.h"
#include "hccl/hccl_types.h"
#include <memory>
#include <string>
#include <unordered_map>
#include "config_log.h"
#include "../../../unified_platform/resource/buffer/dev_buffer.h"

namespace hccl {

// P1：HostDpu kernel 的拉起与共享内存已收编 base_comm（LegacyDpuThread::InitShmem → DpuShmemMgr，
// HcclDevMemAcquire 在 op_base_mc2.cc 拦截 "DPUTAG" 后转发到 AcquireDpuTagShmem，910/950 共用），
// 本类不再承担 kernel 生命周期，也不再受理 DPUTAG。仅保留「非 DPUTAG」的通用 workspace 管理：
// GetDevMemWorkSpace 走 tagWorkspaceMap_ 这张 DevBuffer 表，供 MC2 侧 HcclDevMemAcquire 使用。
class DpuManager {
public:
    DpuManager();
    ~DpuManager();

    HcclResult CreateWorkspaceBuf(const char* memTag, uint64_t* size, bool* newCreated);
    std::shared_ptr<Hccl::DevBuffer> GetKFCWorkSpace(const char* memTag);
    HcclResult GetDevMemWorkSpace(const std::string& memTag, uint64_t* size, void** addr, bool* newCreated);

private:
    std::unordered_map<std::string, std::shared_ptr<Hccl::DevBuffer>> tagWorkspaceMap_;
};
} // namespace hccl

#endif // HCCL_DPU_MANAGER_H
