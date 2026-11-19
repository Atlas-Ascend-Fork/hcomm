/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROCE_ENDPOINT_H
#define ROCE_ENDPOINT_H

#include <optional>
#include <memory>
#include <mutex>
#include <vector>
#include <string>
#include "endpoint.h"
#include "server_socket_context/host_server_socket_context.h"
#include "roce_reged_mem_mgr.h"
#include "buffer/local_rdma_rma_buffer_v2.h"
#include "externalinput_pub.h"

namespace hcomm {
/**
 * @note 职责：Host CPU通信引擎+RoCE协议的通信设备Endpoint，管理通信设备上下文，以及设备上的注册内存。
 */
class CpuRoceEndpoint : public Endpoint {
public:
    explicit CpuRoceEndpoint(const EndpointDesc& endpointDesc);

    ~CpuRoceEndpoint() noexcept override;

    HcclResult Init() override;

    // 本端走进程级 mgr（硬件注册），实例级归属记录经 Record/Remove/GetAllLocalMemHandles 承载；远端走实例级 mgr
    LocalRegedMemMgr* GetLocalRegMemMgr() override { return localMemMgr_.get(); }
    RemoteRegedMemMgr* GetRemoteRegMemMgr() override { return remoteMemMgr_.get(); }
    HcclResult RecordLocalMemHandle(void* memHandle) override;
    HcclResult RemoveLocalMemHandle(void* memHandle) override;
    HcclResult
    FindLocalBuffer(uint64_t addr, uint64_t len, std::shared_ptr<Hccl::LocalRmaBuffer>& buffer) const override;
    HcclResult GetAllLocalMemHandles(void** memHandles, uint32_t* memHandleNum) override;
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override;
    ServerSocketContext* GetServerSocketContext() override
    {
        return serverSocketContext_.has_value() ? &serverSocketContext_.value() : nullptr;
    }

    struct Capabilities {
        uint64_t maxMsgSize{0}; // 硬件单次 RDMA 传输上限，RoCE 下读写共用
        int lbMax{0};
        // 按需扩展
    };
    HcclResult GetCapabilities(Capabilities& caps);

private:
    HcclResult ReleaseEndpointCtx();
    HcclResult
    AttachCache(const MemMgrCacheKey& key, const std::function<std::shared_ptr<LocalRegedMemMgr>()>& creator);
    HcclResult ReleaseCache();

    void* ctxHandle_{nullptr};
    std::shared_ptr<EndpointCtx> endpointCtx_{};
    std::shared_ptr<LocalRegedMemMgr> localMemMgr_{}; // 进程级 mgr（跨 endpoint 复用，承担硬件注册）
    // 本实例注册的 memHandle 记录（不可变快照发布）：Record/Remove 持锁拷贝当前代、私有副本改后
    // std::atomic_store 发布（构造先于发布，无并发窗口）；FindLocalBuffer 经 std::atomic_load 取快照后
    // 遍历（插入序 = 首条命中优先级保持）；旧代由在途读者的 shared_ptr 副本自动保活回收
    using MemRecords = std::vector<std::shared_ptr<Hccl::LocalRdmaRmaBuffer>>;
    using MemRecordsSnapshot = const MemRecords;
    std::mutex localMemMtx_; // 写-写串行 + 并发 GetAll 串行
    std::shared_ptr<MemRecordsSnapshot> memRecordsPublished_{std::make_shared<MemRecords>()};
    // GetAll 出参裸指针的保活：钉住最近一次 GetAll 引用的代（data() 存活至下次 GetAll）
    std::shared_ptr<MemRecordsSnapshot> getAllKeepAlive_{};
    std::shared_ptr<RemoteRegedMemMgr> remoteMemMgr_{}; // 实例级远端 mgr（EndpointRemoteRegedMemMgr，RoCE 适配）
    std::optional<HostServerSocketContext> serverSocketContext_{};
    MemMgrCacheKey cacheKey_{};
    // 非 owning 别名（空 deleter）：实例寿命由 EndpointMgr 声明序保证（R3），reset 不触发释放
    std::shared_ptr<ProcRegedMemMgrCache> cacheAlias_{};
    Capabilities capabilities_{};
    bool isCapabilitiesAvailable_{false};
};
} // namespace hcomm
#endif // ROCE_ENDPOINT_H
