/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include "config_log.h"
#include "hccl_dpu_manager.h"

namespace hccl {
// 本类仅保留「非 DPUTAG」的通用 workspace 管理：GetDevMemWorkSpace 走 tagWorkspaceMap_ 这张 DevBuffer 表，
// 供 MC2 侧 HcclDevMemAcquire 使用。

DpuManager::DpuManager() {}

DpuManager::~DpuManager() {}

HcclResult DpuManager::CreateWorkspaceBuf(const char* memTag, uint64_t* size, bool* newCreated)
{
    std::string tag = memTag != nullptr ? std::string(memTag) : "";
    if (tagWorkspaceMap_.find(tag) == tagWorkspaceMap_.end()) {
        std::shared_ptr<Hccl::DevBuffer> workspace = std::make_shared<Hccl::DevBuffer>(*size);
        tagWorkspaceMap_.insert(make_pair(tag, workspace));
        HCCL_INFO(
            "[DpuManager::%s] Create tagMem[%s] WorkspaceBuf success, WorkspaceBuf = %p", __func__, tag.c_str(),
            workspace.get());
        if (newCreated != nullptr) {
            *newCreated = true;
        }
    }
    return HCCL_SUCCESS;
}

std::shared_ptr<Hccl::DevBuffer> DpuManager::GetKFCWorkSpace(const char* memTag)
{
    std::string tag = memTag != nullptr ? std::string(memTag) : "";
    auto it = tagWorkspaceMap_.find(tag);
    return it != tagWorkspaceMap_.end() ? it->second : nullptr;
}

HcclResult DpuManager::GetDevMemWorkSpace(const std::string& memTag, uint64_t* size, void** addr, bool* newCreated)
{
    auto iter = tagWorkspaceMap_.find(memTag);
    if (iter != tagWorkspaceMap_.end()) {
        std::shared_ptr<Hccl::DevBuffer> oldWorkspace = iter->second;
        if (*size != static_cast<uint64_t>(oldWorkspace.get()->GetSize())) {
            HCCL_ERROR(
                "DpuManager::GetDevMemWorkSpace, The size of oldWorkspace %p is non-consistent, "
                "target size compare now size: %llu->%llu",
                *addr, *size, oldWorkspace.get()->GetSize());
            return HCCL_E_PARA;
        }
        *addr = reinterpret_cast<void*>(oldWorkspace.get()->GetAddr());
        if (newCreated != nullptr) {
            *newCreated = false;
        }
        return HcclResult::HCCL_SUCCESS;
    }

    std::shared_ptr<Hccl::DevBuffer> newWorkspace = std::make_shared<Hccl::DevBuffer>(*size);
    tagWorkspaceMap_.insert(make_pair(memTag, newWorkspace));
    HCCL_INFO(
        "Create tagMem[%s] WorkspaceBuf success, WorkspaceBuf: %p -> %p, size[%llu]", memTag.c_str(),
        newWorkspace.get(), newWorkspace.get()->GetAddr(), *size);
    if (newCreated != nullptr) {
        *newCreated = true;
    }
    *addr = reinterpret_cast<void*>(newWorkspace.get()->GetAddr());
    return HcclResult::HCCL_SUCCESS;
}

} // namespace hccl
