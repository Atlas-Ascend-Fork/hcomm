/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GLOBAL_NET_DEV_MANAGER_H
#define GLOBAL_NET_DEV_MANAGER_H

#include <atomic>
#include <set>
#include <map>
#include <mutex>
#include <memory>
#include <unordered_set>

#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>
#include "hccl_network_pub.h"
#include "hccl_socket_manager.h"
#include "hccl_net_dev.h"
#include "hccl_socket.h"

namespace hccl {
class GlobalNetDevMgr;
} // namespace hccl

// 窄访问器（base_comm_node_accessors.h）经 EnsureInited 钩子完成首用懒初始化
namespace hcomm {
hccl::GlobalNetDevMgr& GetGlobalNetDevMgr(uint32_t devPhyId);
} // namespace hcomm

namespace hccl {

// 进程粒度的endpoint的netdev管理。
// 进程级表/锁由 hcomm::EndpointProcMgr 持有（经其访问器读写），设备级实例由 hcomm::NetDevProc 持有。
class GlobalNetDevMgr {
public:
    static void MakeSocketTag(
        hccl::HcclIpAddress tagServerIp, uint32_t tagServerPort, hccl::HcclIpAddress tagClientIp,
        std::string& socketTag);
    static HcclResult GetDeviceVnicIP(u32 devicePhyId, u32 superDeviceId, hccl::HcclIpAddress& vnicIP);
    ~GlobalNetDevMgr();

    // 首用懒初始化钩子：isInited_ 检查 + Init，检查/调用顺序与原子内存序不可变；仅窄访问器调用
    HcclResult EnsureInited(u32 devicePhyId, u32 deviceLogicId);

    HcclResult RefNetDevCtx(NicType nicType, const HcclIpAddress& ipAddr, u32 port, HcclNetDevCtx& netDevCtx);
    HcclResult UnRefNetDevCtx(NicType nicType, const HcclIpAddress& ipAddr, u32 port);

    HcclResult ServerInit(u32 port);
    HcclResult ServerDeInit(u32 port);

    HcclResult ConnectToServer(
        uint32_t localPort, hccl::HcclIpAddress remoteIp, uint32_t remotePort, std::string& socketTag,
        std::shared_ptr<hccl::HcclSocket>& socket);
    HcclResult AcceptClient(
        uint32_t localPort, hccl::HcclIpAddress remoteIp, std::string& socketTag,
        std::shared_ptr<hccl::HcclSocket>& socket);
    void CloseSocket(std::shared_ptr<hccl::HcclSocket>& socket) const;

private:
    HcclResult Init(u32 devicePhyId, u32 deviceLogicId); // 仅由 EnsureInited 钩子在目标实例上调用
    void UnInit();
    HcclResult ServerDeInit(const HcclIpAddress& localIp, u32 port) const;
    HcclResult
    GetListenSocket(const HcclIpAddress& localIp, uint32_t port, std::shared_ptr<hccl::HcclSocket>& listenSocket) const;
    HcclResult AddListenSocketWhiteList(
        const HcclIpAddress& localIp, uint32_t port, const std::vector<SocketWlistInfo>& wlistInfos) const;
    HcclResult AcceptDataSocket(
        const HcclIpAddress& localIp, uint32_t port, const std::string& tag,
        std::shared_ptr<hccl::HcclSocket>& outConnected, uint32_t acceptTimeoutMs) const;
    HcclResult WaitClientSocketLinkEstablished(const std::shared_ptr<hccl::HcclSocket>& socket, s32 timeoutSec) const;

private:
    u32 devicePhyId_{INVALID_UINT};
    s32 deviceLogicId_{INVALID_INT};

    std::atomic<bool> isInited_{false};
    HcclNetDevCtx netDevCtx_{nullptr};
};

} // namespace hccl
#endif // GLOBAL_NET_DEV_MANAGER_H
