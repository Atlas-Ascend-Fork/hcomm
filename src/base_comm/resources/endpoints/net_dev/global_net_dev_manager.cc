/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "global_net_dev_manager.h"
#include <string>
#include "hccl_mem.h"
// for hccl_network.h
#include "hccl_network.h"
#include "network_manager_pub.h"
#include "dlhal_function.h"
#include "dlra_function.h"
#include "base_comm_res_per_dev.h"
#include "resources/endpoints/net_dev/endpoint_proc_mgr.h"
#include "base_comm_node_accessors.h"

using namespace hccl;

namespace hccl {
// 进程级表/锁的落点：hcomm::EndpointProcMgr 值成员
static hcomm::EndpointProcMgr& NetInfra() { return hcomm::GetEndpointProcMgr(); }

GlobalNetDevMgr::~GlobalNetDevMgr()
{
    if (isInited_.load(std::memory_order_acquire)) {
        HCCL_INFO("[GlobalNetDevMgr][%s] start.", __func__);
        UnInit();
        HCCL_INFO("[GlobalNetDevMgr][%s] end.", __func__);
    }
}

HcclResult GlobalNetDevMgr::EnsureInited(u32 devicePhyId, u32 deviceLogicId)
{
    if (isInited_.load(std::memory_order_acquire)) {
        return HCCL_SUCCESS;
    }
    return Init(devicePhyId, deviceLogicId);
}

HcclResult GlobalNetDevMgr::Init(u32 devicePhyId, u32 deviceLogicId)
{
    // init after get the lock
    std::unique_lock<std::mutex> lock(NetInfra().GetNetDevCtxMtx());
    if (isInited_.load(std::memory_order_relaxed)) {
        return HCCL_SUCCESS;
    }

    if (!NetInfra().IsDlRaInited()) {
        CHK_RET(hccl::DlRaFunction::GetInstance().DlRaFunctionInit());
        CHK_RET(hccl::DlHalFunction::GetInstance().DlHalFunctionInit());
        NetInfra().SetDlRaInited(true);
    }

    // need to check again
    if (isInited_.load(std::memory_order_relaxed)) {
        HCCL_INFO(
            "[GlobalNetDevMgr][Init]Has been inited. devicePhyId[%u], deviceLogicId[%u]", devicePhyId, deviceLogicId);
        return HCCL_SUCCESS;
    }

    devicePhyId_ = devicePhyId;
    deviceLogicId_ = deviceLogicId;
    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId, static_cast<u32>(deviceLogicId), false));
    isInited_.store(true, std::memory_order_release);
    HCCL_INFO("[GlobalNetDevMgr][Init]Init success, devicePhyId[%u], deviceLogicId[%u]", devicePhyId, deviceLogicId);
    return HCCL_SUCCESS;
}

void GlobalNetDevMgr::UnInit()
{
    if (!isInited_.load(std::memory_order_acquire)) {
        HCCL_INFO(
            "[GlobalNetDevMgr][UnInit]has been deinited. devicePhyId[%u], deviceLogicId[%d]", devicePhyId_,
            deviceLogicId_);
        return;
    }

    (void)HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, static_cast<u32>(deviceLogicId_));
    netDevCtx_ = nullptr;
    isInited_.store(false, std::memory_order_release);
    HCCL_INFO(
        "[GlobalNetDevMgr][UnInit]UnInit success. devicePhyId[%u], deviceLogicId[%d]", devicePhyId_, deviceLogicId_);
}

HcclResult GlobalNetDevMgr::GetDeviceVnicIP(u32 devicePhyId, u32 superDeviceId, hccl::HcclIpAddress& vnicIP)
{
    s32 localDeviceLogicId;
    u32 localDeviceId;
    CHK_RET(hrtGetDevice(&localDeviceLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(localDeviceLogicId), localDeviceId));

    // 先创建进程
    bool isHostUseDevNic;
    CHK_RET(IsHostUseDevNic(isHostUseDevNic));
    u32 tempDevicePhyId = hccl::DEFAULT_PHY_ID;
    HCCL_DEBUG(
        "[GlobalNetDevMgr][%s]GetDeviceVnicIP, deviceLogicId[%d], devicePhyId[%u], "
        "nicDeploy[%d], hasBackup[%d], tempDevicePhyId[%u]",
        __func__, localDeviceLogicId, devicePhyId, static_cast<int>(NICDeployment::NIC_DEPLOYMENT_DEVICE), false,
        tempDevicePhyId);
    CHK_RET(hccl::NetworkManager::GetInstance(localDeviceLogicId)
                .InitV2(NICDeployment::NIC_DEPLOYMENT_DEVICE, false, tempDevicePhyId, isHostUseDevNic));

    // 参考 Heartbeat::GetConnInfo
    // hccl::HcclIpAddress vnicIP(localDeviceId);
    if (superDeviceId != SUPER_DEVICE_ID_INVALID) {
        CHK_RET(
            hrtRaGetSingleSocketVnicIpInfo(localDeviceId, DeviceIdType::DEVICE_ID_TYPE_SDID, superDeviceId, vnicIP));
    } else {
        CHK_RET(
            hrtRaGetSingleSocketVnicIpInfo(localDeviceId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, devicePhyId, vnicIP));
    }

    HCCL_INFO(
        "[GlobalNetDevMgr][GetDeviceVnicIP] vnicIP [%s] for devicePhyId[%u], superDeviceId[%u]",
        vnicIP.GetReadableAddress(), devicePhyId, superDeviceId);

    // 销毁进程
    CHK_RET(hccl::NetworkManager::GetInstance(localDeviceLogicId)
                .DeInitV2(NICDeployment::NIC_DEPLOYMENT_DEVICE, false, false));
    return HCCL_SUCCESS;
}

HcclResult
GlobalNetDevMgr::RefNetDevCtx(NicType nicType, const HcclIpAddress& ipAddr, u32 port, HcclNetDevCtx& netDevCtx)
{
    HCCL_INFO(
        "[GlobalNetDevMgr][RefNetDevCtx] nicType[%d], ip[%s]", static_cast<int>(nicType), ipAddr.GetReadableAddress());
    std::lock_guard<std::mutex> lock(NetInfra().GetNetDevCtxMtx());

    // 进程粒度open dev，如果已open，直接复用
    PortInfo portInfo(ipAddr, port);
    if (NetInfra().GetNetDevCtxMap().find(portInfo) != NetInfra().GetNetDevCtxMap().end()) {
        netDevCtx = NetInfra().GetNetDevCtxMap()[portInfo].second;
        CHK_PTR_NULL(netDevCtx);

        auto& netDevCtxRef = NetInfra().GetNetDevCtxRefMap()[portInfo];
        netDevCtxRef.Ref();
        netDevCtx_ = netDevCtx;

        HCCL_INFO(
            "[GlobalNetDevMgr][RefNetDevCtx] nicType[%d] ip[%s] has been Ref.", static_cast<int>(nicType),
            ipAddr.GetReadableAddress());
        return HCCL_SUCCESS;
    }

    HcclNetDevCtx tempNetDevCtx;
    CHK_RET(HcclNetOpenDev(&tempNetDevCtx, nicType, devicePhyId_, deviceLogicId_, ipAddr));
    CHK_PTR_NULL(tempNetDevCtx);

    try {
        NetInfra().GetNetDevCtxMap().insert(std::make_pair(portInfo, std::make_pair(nicType, tempNetDevCtx)));
    } catch (...) {
        (void)HcclNetCloseDev(tempNetDevCtx);
        return HCCL_E_MEMORY;
    }

    Referenced ref;
    ref.Ref();
    try {
        NetInfra().GetNetDevCtxRefMap().insert(std::make_pair(portInfo, ref));
    } catch (...) {
        NetInfra().GetNetDevCtxMap().erase(portInfo);
        (void)HcclNetCloseDev(tempNetDevCtx);
        return HCCL_E_MEMORY;
    }

    netDevCtx = tempNetDevCtx;
    netDevCtx_ = netDevCtx;
    HCCL_INFO(
        "[GlobalNetDevMgr][RefNetDevCtx] nicType[%d] ip[%s] has been Init.", static_cast<int>(nicType),
        ipAddr.GetReadableAddress());
    return HCCL_SUCCESS;
}

HcclResult GlobalNetDevMgr::UnRefNetDevCtx(NicType nicType, const HcclIpAddress& ipAddr, u32 port)
{
    HCCL_INFO(
        "[GlobalNetDevMgr][UnRefNetDevCtx] nicType[%d], ip[%s]", static_cast<int>(nicType),
        ipAddr.GetReadableAddress());

    std::lock_guard<std::mutex> lock(NetInfra().GetNetDevCtxMtx());

    HcclNetDevCtx netDevCtx;
    PortInfo portInfo(ipAddr, port);
    if (NetInfra().GetNetDevCtxMap().find(portInfo) != NetInfra().GetNetDevCtxMap().end()) {
        netDevCtx = NetInfra().GetNetDevCtxMap()[portInfo].second;
        CHK_PTR_NULL(netDevCtx);

        auto& netDevCtxRef = NetInfra().GetNetDevCtxRefMap()[portInfo];
        netDevCtxRef.Unref();
        HCCL_INFO(
            "[GlobalNetDevMgr][UnRefNetDevCtx] nicType[%d] ip[%s] has been UnRef.", static_cast<int>(nicType),
            ipAddr.GetReadableAddress());

        if (netDevCtxRef.Count() == 0) {
            NetInfra().GetNetDevCtxMap().erase(portInfo);
            NetInfra().GetNetDevCtxRefMap().erase(portInfo);
            HcclNetCloseDev(netDevCtx);
            HCCL_INFO(
                "[GlobalNetDevMgr][UnRefNetDevCtx] nicType[%d] ip[%s] has been Deinit.", static_cast<int>(nicType),
                ipAddr.GetReadableAddress());
        }
    }

    if (NetInfra().GetNetDevCtxMap().empty()) {
        UnInit();
    }
    return HCCL_SUCCESS;
}

HcclResult GlobalNetDevMgr::ServerInit(u32 port)
{
    HcclIpAddress localIp{0};
    std::shared_ptr<HcclSocket> tempSocket;
    {
        std::lock_guard<std::mutex> lock(NetInfra().GetNetDevCtxMtx());
        CHK_RET(HcclNetDevGetLocalIp(netDevCtx_, localIp));
        HCCL_INFO("[GlobalNetDevMgr][ServerInit]ip[%s] port[%u]", localIp.GetReadableAddress(), port);
        EXCEPTION_CATCH((tempSocket = std::make_shared<HcclSocket>(netDevCtx_, port)), return HCCL_E_PTR);
    }

    PortInfo portInfo(localIp, port);
    std::unique_lock<std::mutex> lock(NetInfra().GetServerMapMutex());
    auto serverSocketInMap = NetInfra().GetServerSocketMap().find(portInfo);
    if (serverSocketInMap != NetInfra().GetServerSocketMap().end()) {
        auto& serverSocketRef = NetInfra().GetServerSocketRefMap()[portInfo];
        serverSocketRef.Ref();
        HCCL_INFO("[GlobalNetDevMgr][ServerInit]ip[%s] port[%u] inited", localIp.GetReadableAddress(), port);
        return HCCL_SUCCESS;
    }

    CHK_RET(tempSocket->Init());
    CHK_RET(tempSocket->Listen());
    NetInfra().GetServerSocketMap().insert(std::make_pair(portInfo, tempSocket));

    Referenced ref;
    ref.Ref();
    NetInfra().GetServerSocketRefMap().insert(std::make_pair(portInfo, ref));
    HCCL_INFO("[GlobalNetDevMgr][ServerInit]ip[%s] port[%u] init done", localIp.GetReadableAddress(), port);
    return HCCL_SUCCESS;
}

HcclResult GlobalNetDevMgr::ServerDeInit(u32 port)
{
    HcclIpAddress localIp{0};
    {
        std::lock_guard<std::mutex> lock(NetInfra().GetNetDevCtxMtx());
        CHK_RET(HcclNetDevGetLocalIp(netDevCtx_, localIp));
    }
    CHK_RET(ServerDeInit(localIp, port));

    return HCCL_SUCCESS;
}

HcclResult GlobalNetDevMgr::ServerDeInit(const HcclIpAddress& localIp, u32 port) const
{
    PortInfo portInfo(localIp, port);

    std::unique_lock<std::mutex> lock(NetInfra().GetServerMapMutex());
    auto res = NetInfra().GetServerSocketMap().find(portInfo);
    if (res == NetInfra().GetServerSocketMap().end()) {
        HCCL_INFO("[DeInit][Server]ip[%s] port[%u] not found", localIp.GetReadableAddress(), port);
        return HCCL_SUCCESS;
    }

    auto& serverSocketRef = NetInfra().GetServerSocketRefMap()[portInfo];
    serverSocketRef.Unref();

    HCCL_INFO(
        "[DeInit][Server]ip[%s] port[%u] serverSocketRef.Count() = %d", localIp.GetReadableAddress(), port,
        serverSocketRef.Count());
    if (serverSocketRef.Count() == 0) {
        HCCL_INFO("[DeInit][Server]ip[%s] port[%u]", localIp.GetReadableAddress(), port);
        NetInfra().GetServerSocketMap()[portInfo]->DeInit();
        NetInfra().GetServerSocketMap().erase(portInfo);
        NetInfra().GetServerSocketRefMap().erase(portInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult GlobalNetDevMgr::GetListenSocket(
    const HcclIpAddress& localIp, uint32_t port, std::shared_ptr<hccl::HcclSocket>& listenSocket) const
{
    PortInfo portInfo(localIp, port);
    std::lock_guard<std::mutex> lock(NetInfra().GetServerMapMutex());
    auto it = NetInfra().GetServerSocketMap().find(portInfo);
    if (it == NetInfra().GetServerSocketMap().end() || it->second == nullptr) {
        HCCL_ERROR("[GlobalNetDevMgr][%s] no listen socket for port[%u]", __func__, port);
        return HCCL_E_NOT_FOUND;
    }
    listenSocket = it->second;
    return HCCL_SUCCESS;
}

HcclResult GlobalNetDevMgr::AddListenSocketWhiteList(
    const HcclIpAddress& localIp, uint32_t port, const std::vector<SocketWlistInfo>& wlistInfos) const
{
    if (wlistInfos.empty()) {
        HCCL_ERROR("[GlobalNetDevMgr][%s] empty whitelist", __func__);
        return HCCL_E_PARA;
    }

    std::shared_ptr<hccl::HcclSocket> listenSocket;
    CHK_RET(GetListenSocket(localIp, port, listenSocket));
    std::vector<SocketWlistInfo> mutableCopy = wlistInfos;
    return listenSocket->AddWhiteList(mutableCopy);
}

HcclResult GlobalNetDevMgr::AcceptDataSocket(
    const HcclIpAddress& localIp, uint32_t port, const std::string& tag,
    std::shared_ptr<hccl::HcclSocket>& outConnected, uint32_t acceptTimeoutMs) const
{
    std::shared_ptr<hccl::HcclSocket> listenSocket;
    CHK_RET(GetListenSocket(localIp, port, listenSocket));
    return listenSocket->Accept(tag, outConnected, acceptTimeoutMs);
}

HcclResult
GlobalNetDevMgr::WaitClientSocketLinkEstablished(const std::shared_ptr<hccl::HcclSocket>& socket, s32 timeoutSec) const
{
    CHK_SMART_PTR_NULL(socket);
    u32 pollCount = 0;
    const auto startTime = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(timeoutSec > 0 ? timeoutSec : GetExternalInputHcclLinkTimeOut());
    HCCL_DEBUG("[GlobalNetDevMgr][client][WaitLink] waiting for socket link up...");
    while (true) {
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_ERROR(
                "[GlobalNetDevMgr][client][WaitLink] wait socket establish timeout, timeout[%lld s]",
                static_cast<long long>(timeout.count()));
            socket->SetStatus(hccl::HcclSocketStatus::SOCKET_TIMEOUT);
            return HCCL_E_TIMEOUT;
        }
        const hccl::HcclSocketStatus status = socket->GetStatus();
        if (status == hccl::HcclSocketStatus::SOCKET_OK) {
            HCCL_DEBUG(
                "[GlobalNetDevMgr][client][WaitLink] socket established. localIp[%s], remoteIp[%s]",
                socket->GetLocalIp().GetReadableIP(), socket->GetRemoteIp().GetReadableIP());
            return HCCL_SUCCESS;
        }
        if (status == hccl::HcclSocketStatus::SOCKET_CONNECTING) {
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
            if (pollCount % 50U == 0U) {
                HCCL_DEBUG("[GlobalNetDevMgr][client][WaitLink] socket is connecting");
            }
            ++pollCount;
            continue;
        }
        if (status == hccl::HcclSocketStatus::SOCKET_TIMEOUT) {
            return HCCL_E_TIMEOUT;
        }
        socket->SetStatus(hccl::HcclSocketStatus::SOCKET_ERROR);
        return HCCL_E_TCP_CONNECT;
    }
}

void GlobalNetDevMgr::MakeSocketTag(
    hccl::HcclIpAddress tagServerIp, uint32_t tagServerPort, hccl::HcclIpAddress tagClientIp, std::string& socketTag)
{
    socketTag = tagServerIp.GetReadableIP() + std::string(":") + std::to_string(tagServerPort) + std::string(":")
                + tagClientIp.GetReadableIP();
}

HcclResult GlobalNetDevMgr::ConnectToServer(
    uint32_t localPort, hccl::HcclIpAddress remoteIp, uint32_t remotePort, std::string& socketTag,
    std::shared_ptr<hccl::HcclSocket>& socket)
{
    HCCL_INFO("[GlobalNetDevMgr]ConnectToServer start");

    hccl::HcclIpAddress localIpAddr;
    std::shared_ptr<hccl::HcclSocket> socketTemp = nullptr;
    {
        std::lock_guard<std::mutex> lock(NetInfra().GetNetDevCtxMtx());
        CHK_PTR_NULL(netDevCtx_);
        auto* netDevCtxPtr = static_cast<hccl::NetDevContext*>(netDevCtx_);
        localIpAddr = netDevCtxPtr->GetLocalIp();

        HCCL_INFO(
            "[GlobalNetDevMgr]ConnectToServer localIp[%s] localPort[%u] remoteIp[%s] remotePort[%u] socketTag[%s]",
            localIpAddr.GetReadableIP(), localPort, remoteIp.GetReadableIP(), remotePort, socketTag.c_str());
        HCCL_INFO("[GlobalNetDevMgr][client] ConnectToServer connect to server");

        EXCEPTION_CATCH(
            socketTemp = std::make_shared<hccl::HcclSocket>(
                socketTag, netDevCtx_, remoteIp, remotePort, hccl::HcclSocketRole::SOCKET_ROLE_CLIENT),
            return HCCL_E_PTR);
    }

    CHK_SMART_PTR_NULL(socketTemp);
    CHK_RET(socketTemp->Init());
    CHK_RET(socketTemp->Connect());
    HcclResult waitRet = WaitClientSocketLinkEstablished(socketTemp, 0);
    if (waitRet != HCCL_SUCCESS) {
        socketTemp->Close();
        return waitRet;
    }

    socket = socketTemp;
    HCCL_INFO(
        "[GlobalNetDevMgr]ConnectToServer done localPort[%u] remotePort[%u]", socket->GetLocalPort(),
        socket->GetRemotePort());

    return HCCL_SUCCESS;
}

HcclResult GlobalNetDevMgr::AcceptClient(
    uint32_t localPort, hccl::HcclIpAddress remoteIp, std::string& socketTag, std::shared_ptr<hccl::HcclSocket>& socket)
{
    HCCL_INFO("[GlobalNetDevMgr]AcceptClient start");

    hccl::HcclIpAddress localIpAddr;
    {
        std::lock_guard<std::mutex> lock(NetInfra().GetNetDevCtxMtx());
        CHK_PTR_NULL(netDevCtx_);
        auto* netDevCtxPtr = static_cast<hccl::NetDevContext*>(netDevCtx_);
        localIpAddr = netDevCtxPtr->GetLocalIp();
    }

    HCCL_INFO(
        "[GlobalNetDevMgr]AcceptClient localIp[%s] localPort[%u] remoteIp[%s] socketTag[%s]",
        localIpAddr.GetReadableIP(), localPort, remoteIp.GetReadableIP(), socketTag.c_str());

    HCCL_INFO("[GlobalNetDevMgr][server] AcceptClient listen and accept");
    SocketWlistInfo wlistEntry{};
    wlistEntry.connLimit = 1U;
    const auto bin = remoteIp.GetBinaryAddress();
    wlistEntry.remoteIp.addr = bin.addr;
    wlistEntry.remoteIp.addr6 = bin.addr6;
    s32 mw = memcpy_s(wlistEntry.tag, sizeof(wlistEntry.tag), socketTag.c_str(), socketTag.size() + 1U);
    CHK_PRT_RET(mw != EOK, HCCL_ERROR("[GlobalNetDevMgr]memcpy_s whitelist tag failed"), HCCL_E_MEMORY);
    const std::vector<SocketWlistInfo> wlistVec = {wlistEntry};
    CHK_RET(AddListenSocketWhiteList(localIpAddr, localPort, wlistVec));

    std::shared_ptr<hccl::HcclSocket> socketTemp = nullptr;
    CHK_RET(AcceptDataSocket(localIpAddr, localPort, socketTag, socketTemp, 0));
    CHK_SMART_PTR_NULL(socketTemp);

    socket = socketTemp;
    HCCL_INFO(
        "[GlobalNetDevMgr]AcceptClient done localPort[%u] remotePort[%u]", socket->GetLocalPort(),
        socket->GetRemotePort());

    return HCCL_SUCCESS;
}

void GlobalNetDevMgr::CloseSocket(std::shared_ptr<hccl::HcclSocket>& socket) const
{
    if (socket != nullptr) {
        socket->Close();
        socket = nullptr;
    }
}
} // namespace hccl
