/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "server_socket_mgr.h"

namespace hcomm {

HcclResult ServerSocketMgr::ServerSocketStartListen(
    const Hccl::PortData& localPort, const Hccl::NicType nicType, const uint32_t devPhyId, uint32_t* port)
{
    if (nicType == Hccl::NicType::HOST_NIC_TYPE) {
        CHK_RET(HostSocketListen(localPort, devPhyId, port));
    } else if (nicType == Hccl::NicType::DEVICE_NIC_TYPE) {
        CHK_RET(DeviceSocketListen(localPort, devPhyId, port));
    } else {
        HCCL_ERROR("[ServerSocketMgr][%s] illegal NicType[%s]", __func__, nicType.Describe().c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult ServerSocketMgr::HostSocketListen(const Hccl::PortData& localPort, const uint32_t devPhyId, uint32_t* port)
{
    std::lock_guard<std::mutex> lock(hostMutex_);
    uint32_t requestedPort = *port;
    if (hostServerSocketMap_.find(localPort) != hostServerSocketMap_.end()
        && hostServerSocketMap_[localPort].find(requestedPort) != hostServerSocketMap_[localPort].end()) {
        if (hostServerSocketMap_[localPort][requestedPort].second == UINT32_MAX) {
            HCCL_ERROR("[ServerSocketMgr][%s]port listening count overflow UINT32_MAX", __func__);
            return HCCL_E_INTERNAL;
        }
        hostServerSocketMap_[localPort][requestedPort].second
            = hostServerSocketMap_[localPort][requestedPort].second + 1; // 计数+1
        HCCL_INFO("[ServerSocketMgr][%s] reuse serverSocket", __func__);
        return HCCL_SUCCESS;
    }

    if (socketMgrCompat_ == nullptr) {
        EXCEPTION_CATCH(socketMgrCompat_ = std::make_unique<Hccl::SocketManager>(), return HCCL_E_INTERNAL);
    }
    bool isListen = socketMgrCompat_->CheckServerPortListening(localPort, requestedPort);

    Hccl::SocketHandle socketHandle{};
    EXCEPTION_CATCH(
        socketHandle = Hccl::HostSocketHandleManager::GetInstance().Create(devPhyId, localPort.GetAddr()),
        return HCCL_E_PARA);

    std::shared_ptr<Hccl::Socket> serverSocket{};
    EXCEPTION_CATCH(
        serverSocket = std::make_shared<Hccl::Socket>(
            socketHandle, localPort.GetAddr(), requestedPort, localPort.GetAddr(), "server", Hccl::SocketRole::SERVER,
            Hccl::NicType::HOST_NIC_TYPE),
        return HCCL_E_PARA); // 端口号可能冲突，需要SE做决定
    HCCL_INFO("[ServerSocketMgr][%s] listen_socket_info[%s]", __func__, serverSocket->Describe().c_str());
    uint32_t actualPort = requestedPort;
    CHK_RET(ListenSocketIfNeeded(serverSocket.get(), isListen, actualPort, port));

    hostServerSocketMap_[localPort][actualPort] = std::make_pair(std::move(serverSocket), 1);

    return HCCL_SUCCESS;
}

HcclResult ServerSocketMgr::DeviceSocketListen(const Hccl::PortData& localPort, const uint32_t devPhyId, uint32_t* port)
{
    std::lock_guard<std::mutex> lock(deviceMutex_);
    uint32_t requestedPort = *port;
    if (deviceServerSocketMap_.find(localPort) != deviceServerSocketMap_.end()
        && deviceServerSocketMap_[localPort].find(requestedPort) != deviceServerSocketMap_[localPort].end()) {
        if (deviceServerSocketMap_[localPort][requestedPort].second == UINT32_MAX) {
            HCCL_ERROR("[ServerSocketMgr][%s]port listening count overflow UINT32_MAX", __func__);
            return HCCL_E_INTERNAL;
        }
        deviceServerSocketMap_[localPort][requestedPort].second
            = deviceServerSocketMap_[localPort][requestedPort].second + 1; // 计数+1
        HCCL_INFO("[ServerSocketMgr][%s] reuse serverSocket", __func__);
        return HCCL_SUCCESS;
    }

    if (socketMgrCompat_ == nullptr) {
        EXCEPTION_CATCH(socketMgrCompat_ = std::make_unique<Hccl::SocketManager>(), return HCCL_E_INTERNAL);
    }
    // 查询socketMgrCompat_，如果查询到已有serversocket，？new一个空壳，放进map里面，计数+1
    bool isListen = socketMgrCompat_->CheckServerPortListening(localPort, requestedPort);

    Hccl::SocketHandle socketHandle{};
    EXCEPTION_CATCH(
        socketHandle = Hccl::SocketHandleManager::GetInstance().Create(devPhyId, localPort), return HCCL_E_PARA);

    std::shared_ptr<Hccl::Socket> serverSocket;
    EXCEPTION_CATCH(
        serverSocket = std::make_shared<Hccl::Socket>(
            socketHandle, localPort.GetAddr(), requestedPort, localPort.GetAddr(), "server", Hccl::SocketRole::SERVER,
            Hccl::NicType::DEVICE_NIC_TYPE),
        return HCCL_E_PARA); // 端口号可能冲突，需要SE做决定
    HCCL_INFO("[ServerSocketMgr][%s] listen_socket_info[%s]", __func__, serverSocket->Describe().c_str());
    uint32_t actualPort = requestedPort;
    CHK_RET(ListenSocketIfNeeded(serverSocket.get(), isListen, actualPort, port));
    deviceServerSocketMap_[localPort][actualPort] = std::make_pair(std::move(serverSocket), 1);

    return HCCL_SUCCESS;
}

HcclResult ServerSocketMgr::ServerSocketStopListen(
    const Hccl::PortData& localPort, const Hccl::NicType nicType, const uint32_t port)
{
    if (nicType == Hccl::NicType::DEVICE_NIC_TYPE) {
        CHK_RET(DeviceSocketStopListen(localPort, port));
    } else if (nicType == Hccl::NicType::HOST_NIC_TYPE) {
        CHK_RET(HostSocketStopListen(localPort, port));
    } else {
        HCCL_ERROR("[ServerSocketMgr][%s] illegal NicType[%s]", __func__, nicType.Describe().c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult ServerSocketMgr::DeviceSocketStopListen(const Hccl::PortData& localPort, const uint32_t port)
{
    std::lock_guard<std::mutex> lock(deviceMutex_);
    if (deviceServerSocketMap_.find(localPort) != deviceServerSocketMap_.end()
        && deviceServerSocketMap_[localPort].find(port) != deviceServerSocketMap_[localPort].end()) {
        if (deviceServerSocketMap_[localPort][port].second == 0) {
            HCCL_ERROR("[ServerSocketMgr][%s]port[%u] listening count already zero", __func__, port);
            return HCCL_E_INTERNAL;
        }
        deviceServerSocketMap_[localPort][port].second = deviceServerSocketMap_[localPort][port].second - 1; // 计数-1
        if (deviceServerSocketMap_[localPort][port].second == 0) {
            deviceServerSocketMap_[localPort].erase(port);
            CHK_RET(DeInitCompatServerIfListening(localPort, port));
        }
        if (deviceServerSocketMap_[localPort].empty()) {
            deviceServerSocketMap_.erase(localPort);
        }
        return HCCL_SUCCESS;
    }
    HCCL_ERROR(
        "[ServerSocketMgr][%s] Can not stop listen cause {PortData[%s], port[%u]} is Not Listening", __func__,
        localPort.Describe().c_str(), port);
    return HCCL_E_NOT_FOUND;
}

HcclResult ServerSocketMgr::HostSocketStopListen(const Hccl::PortData& localPort, const uint32_t port)
{
    std::lock_guard<std::mutex> lock(hostMutex_);
    if (hostServerSocketMap_.find(localPort) != hostServerSocketMap_.end()
        && hostServerSocketMap_[localPort].find(port) != hostServerSocketMap_[localPort].end()) {
        if (hostServerSocketMap_[localPort][port].second == 0) {
            HCCL_ERROR("[ServerSocketMgr][%s]port[%u] listening count already zero", __func__, port);
            return HCCL_E_INTERNAL;
        }
        hostServerSocketMap_[localPort][port].second = hostServerSocketMap_[localPort][port].second - 1; // 计数-1
        if (hostServerSocketMap_[localPort][port].second == 0) {
            hostServerSocketMap_[localPort].erase(port);
            CHK_RET(DeInitCompatServerIfListening(localPort, port));
        }
        if (hostServerSocketMap_[localPort].empty()) {
            hostServerSocketMap_.erase(localPort);
        }
        return HCCL_SUCCESS;
    }
    HCCL_ERROR(
        "[ServerSocketMgr][%s] Can not stop listen cause {PortData[%s], port[%u]} is Not Listening", __func__,
        localPort.Describe().c_str(), port);
    return HCCL_E_NOT_FOUND;
}

HcclResult ServerSocketMgr::ListenSocketIfNeeded(
    Hccl::Socket* serverSocket, bool isListen, uint32_t& actualPort, uint32_t* outPort) const
{
    // port=0 表示自动分配端口，必须调 Listen(actualPort) 获取实际端口，不受 isListen 影响
    if (actualPort == 0) {
        EXCEPTION_CATCH(serverSocket->Listen(actualPort), return HCCL_E_INTERNAL);
        HCCL_INFO("[ServerSocketMgr][%s] allocated port[%u]", __func__, actualPort);
        *outPort = actualPort;
        return HCCL_SUCCESS;
    }
    // 指定端口且已有监听则跳过，避免重复 Listen
    if (isListen) {
        HCCL_INFO("[ServerSocketMgr][%s] port[%u] has been listen", __func__, actualPort);
        return HCCL_SUCCESS;
    }
    EXCEPTION_CATCH(serverSocket->Listen(), return HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ServerSocketMgr::DeInitCompatServerIfListening(const Hccl::PortData& localPort, const uint32_t port)
{
    if (socketMgrCompat_ == nullptr) {
        EXCEPTION_CATCH(socketMgrCompat_ = std::make_unique<Hccl::SocketManager>(), return HCCL_E_INTERNAL);
    }
    bool isListen = socketMgrCompat_->CheckServerPortListening(localPort, port);
    if (isListen) {
        Hccl::PortData portDataCopy(
            localPort.GetRankId(), localPort.GetType(), localPort.GetProto(), localPort.GetId(), localPort.GetAddr());
        socketMgrCompat_->ServerDeInit(portDataCopy);
    }
    return HCCL_SUCCESS;
}

void ServerSocketMgr::DeInitDeviceSockets(u32 devPhyId)
{
    std::lock_guard<std::mutex> lock(deviceMutex_);
    for (auto it = deviceServerSocketMap_.begin(); it != deviceServerSocketMap_.end();) {
        if (static_cast<uint32_t>(it->first.GetRankId()) == devPhyId) {
            for (auto& portEntry : it->second) {
                if (portEntry.second.first != nullptr) {
                    portEntry.second.first->Destroy();
                    portEntry.second.first.reset();
                }
            }
            it = deviceServerSocketMap_.erase(it);
        } else {
            ++it;
        }
    }
}

void ServerSocketMgr::DeInitHostSockets(u32 devPhyId)
{
    std::lock_guard<std::mutex> lock(hostMutex_);
    for (auto it = hostServerSocketMap_.begin(); it != hostServerSocketMap_.end();) {
        if (static_cast<uint32_t>(it->first.GetRankId()) == devPhyId) {
            for (auto& portEntry : it->second) {
                if (portEntry.second.first != nullptr) {
                    portEntry.second.first->Destroy();
                    portEntry.second.first.reset();
                }
            }
            it = hostServerSocketMap_.erase(it);
        } else {
            ++it;
        }
    }
}

void ServerSocketMgr::DeInit(u32 devPhyId)
{
    HCCL_INFO("[ServerSocketMgr][%s] DeInit[%u]", __func__, devPhyId);
    DeInitDeviceSockets(devPhyId);
    DeInitHostSockets(devPhyId);
}

HcclResult ServerSocketMgr::GetServerSocket(
    const Hccl::PortData& localPort, const Hccl::NicType nicType, const uint32_t port,
    std::shared_ptr<Hccl::Socket>& serverSocket)
{
    serverSocket = nullptr;
    if (nicType == Hccl::NicType::HOST_NIC_TYPE) {
        std::lock_guard<std::mutex> lock(hostMutex_);
        auto portIt = hostServerSocketMap_.find(localPort);
        if (portIt == hostServerSocketMap_.end() || portIt->second.find(port) == portIt->second.end()) {
            HCCL_ERROR("[ServerSocketMgr][%s] host serverSocket not found, port[%u]", __func__, port);
            return HCCL_E_NOT_FOUND;
        }
        serverSocket = portIt->second[port].first;
        return HCCL_SUCCESS;
    }
    if (nicType == Hccl::NicType::DEVICE_NIC_TYPE) {
        std::lock_guard<std::mutex> lock(deviceMutex_);
        auto portIt = deviceServerSocketMap_.find(localPort);
        if (portIt == deviceServerSocketMap_.end() || portIt->second.find(port) == portIt->second.end()) {
            HCCL_ERROR("[ServerSocketMgr][%s] device serverSocket not found, port[%u]", __func__, port);
            return HCCL_E_NOT_FOUND;
        }
        serverSocket = portIt->second[port].first;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[ServerSocketMgr][%s] illegal NicType[%s]", __func__, nicType.Describe().c_str());
    return HCCL_E_PARA;
}

HcclResult ServerSocketMgr::GetSocketHandle(
    const Hccl::PortData& localPort, const Hccl::NicType nicType, const uint32_t devPhyId,
    Hccl::SocketHandle& socketHandle) const
{
    if (nicType == Hccl::NicType::HOST_NIC_TYPE) {
        socketHandle = Hccl::HostSocketHandleManager::GetInstance().Get(devPhyId, localPort.GetAddr());
    } else if (nicType == Hccl::NicType::DEVICE_NIC_TYPE) {
        socketHandle = Hccl::SocketHandleManager::GetInstance().Get(devPhyId, localPort);
    } else {
        HCCL_ERROR("[ServerSocketMgr][%s] illegal NicType[%s]", __func__, nicType.Describe().c_str());
        return HCCL_E_PARA;
    }
    if (socketHandle == nullptr) {
        HCCL_ERROR(
            "[ServerSocketMgr][%s] socketHandle not found, devPhyId[%u], localPort[%s]", __func__, devPhyId,
            localPort.Describe().c_str());
        return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

std::vector<Hccl::RaSocketWhitelist> ServerSocketMgr::BuildWhiteList(const std::string& tag, uint32_t connLimit)
{
    Hccl::RaSocketWhitelist wlistEntry{};
    wlistEntry.remoteIp = Hccl::IpAddress{};
    wlistEntry.connLimit = connLimit;
    wlistEntry.tag = tag;
    return {wlistEntry};
}

HcclResult ServerSocketMgr::AddWhiteList(
    const Hccl::PortData& localPort, const Hccl::NicType nicType, const uint32_t devPhyId, const std::string& tag,
    uint32_t connLimit)
{
    Hccl::SocketHandle socketHandle{};
    CHK_RET(GetSocketHandle(localPort, nicType, devPhyId, socketHandle));

    std::vector<Hccl::RaSocketWhitelist> wlists = BuildWhiteList(tag, connLimit);
    EXCEPTION_CATCH(Hccl::HrtRaSocketWhiteListAdd(socketHandle, wlists), return HCCL_E_INTERNAL);

    HCCL_INFO(
        "[ServerSocketMgr][%s] add whitelist success, devPhyId[%u], localPort[%s], tag[%s], connLimit[%u]", __func__,
        devPhyId, localPort.Describe().c_str(), tag.c_str(), connLimit);

    return HCCL_SUCCESS;
}

HcclResult ServerSocketMgr::DeleteWhiteList(
    const Hccl::PortData& localPort, const Hccl::NicType nicType, const uint32_t devPhyId, const std::string& tag,
    uint32_t connLimit)
{
    Hccl::SocketHandle socketHandle{};
    CHK_RET(GetSocketHandle(localPort, nicType, devPhyId, socketHandle));

    std::vector<Hccl::RaSocketWhitelist> wlists = BuildWhiteList(tag, connLimit);
    EXCEPTION_CATCH(Hccl::HrtRaSocketWhiteListDel(socketHandle, wlists), return HCCL_E_INTERNAL);

    HCCL_INFO(
        "[ServerSocketMgr][%s] del whitelist success, devPhyId[%u], localPort[%s], tag[%s], connLimit[%u]", __func__,
        devPhyId, localPort.Describe().c_str(), tag.c_str(), connLimit);

    return HCCL_SUCCESS;
}

} // namespace hcomm
