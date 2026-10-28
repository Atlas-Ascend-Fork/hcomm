/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UB_MEM_ENDPOINT_H
#define UB_MEM_ENDPOINT_H

#include <memory>
#include <optional>
#include <vector>
#include "endpoint.h"
#include "server_socket_context/device_server_socket_context.h"
#include "ub_mem_reged_mem_mgr.h"

namespace hcomm {
class UbMemEndpoint : public Endpoint {
public:
    explicit UbMemEndpoint(const EndpointDesc& endpointDesc);
    ~UbMemEndpoint() noexcept override;
    HcclResult Init() override;
    // 只管本端内存（ub_mem 无远端导入，GetRemoteRegMemMgr 用基类默认返回 nullptr）
    LocalRegedMemMgr* GetLocalRegMemMgr() override { return localMemMgr_.get(); }
    void* GetRdmaHandle() override { return nullptr; }
    bool IsCtxHandleValid() const override { return false; }
    // Socket 监听上下文：Init 后非 nullptr，供 HcommEndpointStartListen 真实监听
    ServerSocketContext* GetServerSocketContext() override
    {
        return serverSocketContext_.has_value() ? &serverSocketContext_.value() : nullptr;
    }

private:
    HcclResult
    AttachCache(const MemMgrCacheKey& key, const std::function<std::shared_ptr<LocalRegedMemMgr>()>& creator);
    HcclResult ReleaseCache();

    std::shared_ptr<UbMemRegedMemMgr> localMemMgr_{};
    std::optional<DeviceServerSocketContext> serverSocketContext_{};
    MemMgrCacheKey cacheKey_{};
    // 非 owning 别名（空 deleter）：实例寿命由 EndpointMgr 声明序保证（R3），reset 不触发释放
    std::shared_ptr<ProcRegedMemMgrCache> cacheAlias_{};
};

} // namespace hcomm

#endif // UB_MEM_ENDPOINT_H
