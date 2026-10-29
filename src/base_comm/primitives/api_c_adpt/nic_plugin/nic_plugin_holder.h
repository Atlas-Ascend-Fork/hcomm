/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_NIC_PLUGIN_HOLDER_H
#define HCOMM_NIC_PLUGIN_HOLDER_H

#include <optional>
#include "endpoint.h"
#include "server_socket_context/server_socket_context.h"
#include "channel.h"
#include "nic_plugin_manager.h"

namespace hcomm {

/**
 * @note 职责：NIC 插件 endpoint 的 RegedMemMgr，内存方法转发到 nicOps_。
 *       GetAllMemHandles 插件 ops 表无此字段，返回 NOT_SUPPORT。
 *       本端+远端一体：只继承 LocalRegedMemMgr（避免多继承），远端接口经
 *       RemoteRegedMemMgrForwarder 包装后对外暴露（见 PluginEndpointHolder）。
 *       nicOps_/nicCtx_ 构造期注入，运行期只读。
 */
class PluginRegedMemMgr : public LocalRegedMemMgr {
public:
    PluginRegedMemMgr(HcommNicEndpointOps* nicOps, void* nicCtx) : nicOps_(nicOps), nicCtx_(nicCtx) {}
    ~PluginRegedMemMgr() override = default;

    HcclResult RegisterMemory(const HcommMem* mem, const char* memTag, void** memHandle) override
    {
        return static_cast<HcclResult>(nicOps_->registerMemory(nicCtx_, mem, memTag, memHandle));
    }
    HcclResult UnregisterMemory(void* memHandle) override
    {
        return static_cast<HcclResult>(nicOps_->unregisterMemory(nicCtx_, memHandle));
    }
    HcclResult
    MemoryExport(const EndpointDesc& endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen) override
    {
        (void)endpointDesc;
        return static_cast<HcclResult>(nicOps_->memoryExport(nicCtx_, memHandle, memDesc, memDescLen));
    }
    // 远端接口：非 RemoteRegedMemMgr 派生（组合 mgr 单继承 Local），保持 virtual 以便 Forwarder 经本类指针转发时虚派发
    virtual HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem)
    {
        return static_cast<HcclResult>(nicOps_->memoryImport(nicCtx_, memDesc, descLen, outMem));
    }
    virtual HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen)
    {
        return static_cast<HcclResult>(nicOps_->memoryUnimport(nicCtx_, memDesc, descLen));
    }
    HcclResult GetAllMemHandles([[maybe_unused]] void** memHandles, [[maybe_unused]] uint32_t* memHandleNum) override
    {
        return HCCL_E_NOT_SUPPORT;
    }

private:
    HcommNicEndpointOps* nicOps_{nullptr};
    void* nicCtx_{nullptr};
};

template <typename Ops>
void DestroyNicPluginOpsAndCtx(Ops*& nicOps, void* nicCtx)
{
    if (nicOps != nullptr) {
        if (nicOps->destroy != nullptr) {
            int32_t ret = nicOps->destroy(nicCtx);
            if (ret != HCCL_SUCCESS) {
                HCCL_WARNING("[%s] plugin destroy unsuccessfully, ret[%d].", __func__, ret);
            }
        }
        delete nicOps;
        nicOps = nullptr;
    }
}

/**
 * @note 职责：NIC 插件 endpoint 的 ServerSocketContext。
 *       GetListenPort 转发 nicOps_->getListenPort
 */
class PluginServerSocketContext : public ServerSocketContext {
public:
    PluginServerSocketContext(HcommNicEndpointOps* nicOps, void* nicCtx) : nicOps_(nicOps), nicCtx_(nicCtx) {}
    ~PluginServerSocketContext() override = default;

    HcclResult ServerSocketGetListenPort(uint32_t* port) override
    {
        return static_cast<HcclResult>(nicOps_->getListenPort(nicCtx_, port));
    }
    HcclResult ServerSocketListen([[maybe_unused]] uint32_t port) override
    {
        return HCCL_E_NOT_SUPPORT; // ops 表无 listen 字段
    }

private:
    HcommNicEndpointOps* nicOps_{nullptr};
    void* nicCtx_{nullptr};
};

/**
 * @note 职责：NIC插件Endpoint占位子类，nicOps_/nicCtx_ 从 Endpoint 基类移入此 private 成员。
 *         内存操作经 GetLocalRegMemMgr()/GetRemoteRegMemMgr() 返回组合持有的 PluginRegedMemMgr（转发到 nicOps_）。
 */
class PluginEndpointHolder : public Endpoint {
public:
    explicit PluginEndpointHolder(const EndpointDesc& endpointDesc, const NicPluginEntry* pluginEntry)
        : Endpoint(endpointDesc),
          pluginEntry_(pluginEntry)
    {}
    ~PluginEndpointHolder() override { DestroyNicPluginOpsAndCtx(nicOps_, nicCtx_); }

    const NicPluginEntry* GetPluginEntry() const { return pluginEntry_; }

    void SetNicEndpointCtx(HcommNicEndpointOps* nicOps, void* nicCtx)
    {
        nicOps_ = nicOps;
        nicCtx_ = nicCtx;
        regedMemMgr_ = std::make_shared<PluginRegedMemMgr>(nicOps, nicCtx);
        // 远端接口经 Forwarder 转发到同一组合 mgr（mgr 本身只继承 LocalRegedMemMgr）
        remoteForwarder_ = std::make_shared<RemoteRegedMemMgrForwarder>(
            [this](const void* memDesc, uint32_t descLen, HcommMem* outMem) {
                return regedMemMgr_->MemoryImport(memDesc, descLen, outMem);
            },
            [this](const void* memDesc, uint32_t descLen) {
                return regedMemMgr_->MemoryUnimport(memDesc, descLen);
            });
        serverSocketContext_.emplace(nicOps, nicCtx);
    }
    HcommNicEndpointOps* GetNicOps() const { return nicOps_; }
    void* GetNicCtx() const { return nicCtx_; }

    // 组合 mgr：本端直达对象；远端经 Forwarder 包装同一对象（避免 mgr 多继承）
    LocalRegedMemMgr* GetLocalRegMemMgr() override { return regedMemMgr_.get(); }
    RemoteRegedMemMgr* GetRemoteRegMemMgr() override { return remoteForwarder_.get(); }

    // 返回组合持有的 PluginServerSocketContext（SetNicEndpointCtx 前返回 nullptr）
    ServerSocketContext* GetServerSocketContext() override
    {
        return serverSocketContext_.has_value() ? &serverSocketContext_.value() : nullptr;
    }

    HcclResult Init() override { return HCCL_E_NOT_SUPPORT; }
    void* GetRdmaHandle() override { return nullptr; }
    bool IsCtxHandleValid() const override { return false; }

private:
    const NicPluginEntry* pluginEntry_;
    HcommNicEndpointOps* nicOps_{nullptr};
    void* nicCtx_{nullptr};
    std::shared_ptr<PluginRegedMemMgr> regedMemMgr_{};               // 组合持有 PluginRegedMemMgr
    std::shared_ptr<RemoteRegedMemMgrForwarder> remoteForwarder_{};  // 远端接口视图，SetNicEndpointCtx 构造
    std::optional<PluginServerSocketContext> serverSocketContext_{}; // 组合持有 PluginServerSocketContext
};

class PluginChannelHolder : public Channel {
public:
    explicit PluginChannelHolder(const NicPluginEntry* pluginEntry) : pluginEntry_(pluginEntry)
    {
        channelClassName_ = "PluginChannelHolder";
    }
    ~PluginChannelHolder() override { DestroyNicPluginOpsAndCtx(nicOps_, nicCtx_); }

    const NicPluginEntry* GetPluginEntry() const { return pluginEntry_; }

    void SetNicChannelCtx(HcommNicChannelOps* nicOps, void* nicCtx)
    {
        nicOps_ = nicOps;
        nicCtx_ = nicCtx;
    }
    HcommNicChannelOps* GetNicOps() const { return nicOps_; }
    void* GetNicCtx() const { return nicCtx_; }

    HcclResult GetNotifyNum(uint32_t* notifyNum) const override
    {
        (void)notifyNum;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override
    {
        (void)memNum;
        (void)remoteMem;
        (void)memInfos;
        return HCCL_E_NOT_SUPPORT;
    }
    ChannelStatus GetStatus() override { return ChannelStatus::FAILED; }
    HcclResult Clean() override { return HCCL_E_NOT_SUPPORT; }
    HcclResult Resume() override { return HCCL_E_NOT_SUPPORT; }
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override
    {
        (void)remoteNotifyIdx;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override
    {
        (void)localNotifyIdx;
        (void)timeout;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx) override
    {
        (void)dst;
        (void)src;
        (void)len;
        (void)remoteNotifyIdx;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult Write(void* dst, const void* src, uint64_t len) override
    {
        (void)dst;
        (void)src;
        (void)len;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult Read(void* dst, const void* src, uint64_t len) override
    {
        (void)dst;
        (void)src;
        (void)len;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult ChannelFence() override { return HCCL_E_NOT_SUPPORT; }
    const HcommChannelDesc& GetChannelDesc() const override
    {
        static const HcommChannelDesc kEmptyDesc{};
        return kEmptyDesc;
    }

private:
    const NicPluginEntry* pluginEntry_;
    HcommNicChannelOps* nicOps_{nullptr};
    void* nicCtx_{nullptr};
};

} // namespace hcomm

#endif // HCOMM_NIC_PLUGIN_HOLDER_H
