/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_ENDPOINT_HANDLE_MGR_H
#define HCOMM_ENDPOINT_HANDLE_MGR_H

#include <memory>
#include <mutex>
#include <unordered_map>
#include "endpoint.h"
#include "hcomm_res_defs.h"

namespace hcomm {

/**
 * @note 职责：HcommEndpointMap（handle→Endpoint）全局句柄表，EndpointMgr 值成员（进程内唯一）。
 *       EndpointCtx 去重缓存（key→ctxHandle_）由同目录 EndpointCtxMgr 管理（per-device）。
 *       进程销毁兜底：BaseCommMgr::DeInit() 链调 DeInit()，遍历 endpointMap_ 强制释放残留。
 */
class EndpointHandleMgr {
public:
    EndpointHandleMgr() = default;
    ~EndpointHandleMgr() = default; // 兜底由资源树 DeInit 链承担

    EndpointHandleMgr(const EndpointHandleMgr&) = delete;
    EndpointHandleMgr& operator=(const EndpointHandleMgr&) = delete;

    // --- HcommEndpointMap 句柄表（不做去重）---
    void Add(EndpointHandle handle, std::unique_ptr<Endpoint> endpoint);
    bool Remove(EndpointHandle handle);
    bool Update(EndpointHandle handle, std::unique_ptr<Endpoint> newEndpoint);
    Endpoint* Get(EndpointHandle handle);

    // 进程销毁兜底：遍历 endpointMap_ 强制释放残留（析构链中子类会回调各设备 EndpointCtxMgr 释放 ctx）
    void DeInit();

    // RESET 兜底告警用：句柄表是否有残留
    bool HasLeftover()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return !endpointMap_.empty();
    }

private:
    std::mutex mtx_; // 仅护 endpointMap_；Endpoint 析构须在锁外（析构链回调 EndpointCtxMgr 重新拿 ctx 锁）
    std::unordered_map<EndpointHandle, std::unique_ptr<Endpoint>> endpointMap_;
};

} // namespace hcomm

#endif // HCOMM_ENDPOINT_HANDLE_MGR_H
