/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "preempt_port_manager_v2.h"
#include <unordered_set>
#include <sstream>
#include "orion_adapter_hccp.h"
#include "hccl_common_v2.h"
#include "adapter_error_manager_pub.h"

namespace Hccl {

constexpr s32 HOST_DEVICE_ID = -1; // device id 无效值

bool PreemptPortManager::initialized = false;

PreemptPortManager::PreemptPortManager()
{
    // 根据host or device nic区分
    IpPortRef hostPortRef;
    preemptSockets_.emplace(HrtNetworkMode::PEER, hostPortRef);
    IpPortRef devPortRef;
    preemptSockets_.emplace(HrtNetworkMode::HDC, devPortRef);
    initialized = true;
}

PreemptPortManager::~PreemptPortManager()
{
    preemptSockets_.clear();
    initialized = false;
}

PreemptPortManager& PreemptPortManager::GetInstance(s32 deviceLogicId)
{
    static PreemptPortManager instance[MAX_MODULE_DEVICE_NUM];
    if (deviceLogicId == HOST_DEVICE_ID) {
        HCCL_INFO("[GetInstance] deviceLogicId[-1] is HOST_DEVICE_ID");
        return instance[0];
    }
    CHK_PRT_RET(
        (static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM || deviceLogicId < 0),
        HCCL_WARNING("[PreemptPortManager::%s] deviceLogicId[%d] is invalid", __func__, deviceLogicId), instance[0]);

    return instance[deviceLogicId];
}

void PreemptPortManager::ListenPreempt(
    const std::shared_ptr<Socket>& listenSocket, const std::vector<SocketPortRange>& portRange, u32& usePort)
{
    CHK_PRT_RET(
        !initialized,
        HCCL_ERROR("[PreemptPortManager::%s] preempt port manager has already been released.", __func__), );

    CHK_SMART_PTR_RET_NULL(listenSocket);
    NicType nicType = listenSocket->GetNicType();
    HrtNetworkMode netMode = nicType == NicType::HOST_NIC_TYPE ? HrtNetworkMode::PEER : HrtNetworkMode::HDC;
    std::lock_guard<std::mutex> lock(preemptMutex_);
    PreemptPortInRange(listenSocket, netMode, portRange, usePort);
    HCCL_INFO("[PreemptPortManager::%s] listening on port[%u] for nicType[%u] success.", __func__, usePort, nicType);
}

void PreemptPortManager::Release(const std::shared_ptr<Socket>& listenSocket)
{
    CHK_PRT_RET(
        !initialized,
        HCCL_WARNING("[PreemptPortManager::%s] preempt port manager has already been released.", __func__), );

    CHK_SMART_PTR_RET_NULL(listenSocket);
    NicType nicType = listenSocket->GetNicType();
    HrtNetworkMode netMode = nicType == NicType::HOST_NIC_TYPE ? HrtNetworkMode::PEER : HrtNetworkMode::HDC;

    std::lock_guard<std::mutex> lock(preemptMutex_);
    ReleasePreempt(preemptSockets_[netMode], listenSocket, netMode);
    HCCL_INFO("[PreemptPortManager::%s] release socket of type[%u] success.", __func__, nicType);
}

void PreemptPortManager::PreemptPortInRange(
    const std::shared_ptr<Socket>& listenSocket, HrtNetworkMode netMode, const std::vector<SocketPortRange>& portRange,
    u32& usePort)
{
    IpPortRef& portRef = preemptSockets_[netMode];
    std::string ipAddr(listenSocket->GetLocalIp().GetIpStr());
    auto portRangeIter = portRef.equal_range(ipAddr);

    // 尝试复用同IP已抢占的端口
    for (auto it = portRangeIter.first; it != portRangeIter.second; ++it) {
        usePort = it->second.first;
        // 校验候选端口是否落在本次请求的portRange范围内，不在范围内则跳过
        bool inRange = false;
        for (const auto& rangeItem : portRange) {
            if (usePort >= rangeItem.min && usePort <= rangeItem.max) {
                inRange = true;
                break;
            }
        }
        if (!inRange) {
            continue;
        }
        bool ret = listenSocket->Listen(usePort);
        if (!ret) {
            std::string errorMessage
                = "The IP address " + ipAddr + " and port " + std::to_string(usePort) + " have already been bound.";
            if (listenSocket->GetNicType() == NicType::HOST_NIC_TYPE) {
                RPT_INPUT_ERR(
                    true, "EI0019", std::vector<std::string>({"reason"}), std::vector<std::string>({errorMessage}));
            } else {
                RPT_INPUT_ERR(
                    true, "EI0020", std::vector<std::string>({"reason"}), std::vector<std::string>({errorMessage}));
            }
        }
        CHK_PRT_THROW(
            !ret, HCCL_ERROR("[PreemptPortManager::%s] usePort[%u] listen failed.", __func__, usePort),
            InvalidParamsException, "socket listen failed");
        it->second.second.Ref();
        HCCL_INFO(
            "[PreemptPortManager::%s] socket usePort[%u] has already been listened, ref count[%d].", __func__, usePort,
            it->second.second.Count());
        return;
    }

    // 收集已记录的端口，避免扫描时重复尝试已占用的端口
    std::unordered_set<u32> usedPorts;
    for (auto it = portRangeIter.first; it != portRangeIter.second; ++it) {
        usedPorts.insert(it->second.first);
    }

    // 轮询输入的端口范围，跳过已记录的端口，找到一个可用的端口
    for (auto& rangeItem : portRange) {
        for (u32 port = rangeItem.min; port <= rangeItem.max; ++port) {
            if (usedPorts.count(port) > 0) {
                continue;
            }

            if (listenSocket->Listen(port)) {
                // 抢占端口成功，将端口记录到计数器中，并作为出参返回
                usePort = port;
                auto preemptPortIter = portRef.insert({ipAddr, {usePort, Referenced()}});
                preemptPortIter->second.second.Ref();
                HCCL_INFO(
                    "[PreemptPortManager::%s] listen on ip[%s] and port[%u] success.", __func__, ipAddr.c_str(),
                    usePort);
                return;
            }

            // 当前端口已被占用，尝试抢占下一个端口
            HCCL_INFO("[PreemptPortManager::%s] could not listen on ip[%s], port[%u].", __func__, ipAddr.c_str(), port);
        }
    }
    ReportPortAllocFail(ipAddr, usePort, listenSocket->GetNicType(), portRange);
    THROW<InvalidParamsException>("No available port to listen");
}

void PreemptPortManager::ReportPortAllocFail(
    const std::string& ipAddr, u32 usePort, NicType nicType, const std::vector<SocketPortRange>& portRange)
{
    std::string portRangeStr = GetRangeStr(portRange);
    std::string errormessage;
    if (usePort == HCCL_INVALID_PORT) {
        errormessage = "The IP address " + ipAddr + " has no available port in range" + portRangeStr
                       + " (all ports in the range have already been bound).";
    } else {
        errormessage
            = "The IP address " + ipAddr + " and port " + std::to_string(usePort) + " have already been bound.";
    }
    if (nicType == NicType::HOST_NIC_TYPE) {
        RPT_INPUT_ERR(true, "EI0019", std::vector<std::string>({"reason"}), std::vector<std::string>({errormessage}));
        HCCL_ERROR(
            "[%s][%s] socket type[2], %s Please check the port status and whether the port is being used by other "
            "process.",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), errormessage.c_str());
    } else {
        RPT_INPUT_ERR(true, "EI0020", std::vector<std::string>({"reason"}), std::vector<std::string>({errormessage}));
        HCCL_ERROR(
            "[%s][%s] socket type[2], %s Please check the port status and whether the port is being used by other "
            "process.",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), errormessage.c_str());
    }
    HCCL_ERROR("[PreemptPortManager::%s] Complete polling of socket port range:%s", __func__, portRangeStr.c_str());
    HCCL_ERROR(
        "[PreemptPortManager::%s] All ports in socket port range are bound already. "
        "no available port to listen. Please check the ports status, or change the port range to listen on.",
        __func__);
    std::string envName
        = nicType == NicType::HOST_NIC_TYPE ? "HCCL_HOST_SOCKET_PORT_RANGE" : "HCCL_NPU_SOCKET_PORT_RANGE";
    HCCL_ERROR(
        "NOTICE: Users need to make sure ports in %s are available for HCCL."
        "Please double check whether the ports are used by other unexpected processes. "
        "The port ranges size should also be enough when running multi-process HCCL.",
        envName.c_str());
    HCCL_ERROR("NOTICE: The host port range size is not suggested to be smaller than the process number"
               " on current rank.");
}

void PreemptPortManager::ReleasePreempt(
    IpPortRef& portRef, const std::shared_ptr<Socket>& listenSocket, [[maybe_unused]] HrtNetworkMode netMode)
{
    std::string ipAddr(listenSocket->GetLocalIp().GetIpStr());
    u32 port = listenSocket->GetListenPort();
    HCCL_INFO("[PreemptPortManager::%s] releasing socket, ip[%s], port[%u].", __func__, ipAddr.c_str(), port);

    bool isListening = IsAlreadyListening(portRef, ipAddr, port);
    // 释放的端口并非正在抢占的端口
    CHK_PRT_RET(
        !isListening,
        HCCL_WARNING(
            "[PreemptPortManager::%s] socket ip[%s], port[%u] is not preempted or has already been released.", __func__,
            ipAddr.c_str(), port), );

    auto portRangeIter = portRef.equal_range(ipAddr);
    auto releaseIpPortIter = portRef.end();
    for (auto it = portRangeIter.first; it != portRangeIter.second; ++it) {
        if (it->second.first == port) {
            releaseIpPortIter = it;
            break;
        }
    }
    CHK_PRT_RET(
        releaseIpPortIter == portRef.end(),
        HCCL_WARNING(
            "[PreemptPortManager::%s] socket ip[%s], port[%u] is not preempted or has already been released.", __func__,
            ipAddr.c_str(), port), );
    // 释放的端口计数异常
    Referenced& ref = releaseIpPortIter->second.second;
    CHK_PRT_THROW(
        ref.Count() <= 0,
        HCCL_ERROR(
            "[PreemptPortManager::%s] ref[%d], ip[%s] port[%u] has already been released.", __func__, ref.Count(),
            ipAddr.c_str(), port),
        InvalidParamsException, "socket port duplicate release");

    // 释放绑定端口的Socket
    listenSocket->StopListen();
    int count = ref.Unref();
    CHK_PRT_RET(
        count > 0,
        HCCL_INFO(
            "[PreemptPortManager::%s] release a socket on ip[%s], port[%u], ref[%d].", __func__, ipAddr.c_str(), port,
            count), );

    // 如果端口的计数归零，则不再抢占该端口
    portRef.erase(releaseIpPortIter);
    HCCL_INFO(
        "[PreemptPortManager::%s] release preemption of socket on ip[%s], port[%u].", __func__, ipAddr.c_str(), port);
}

bool PreemptPortManager::IsAlreadyListening(const IpPortRef& ipPortRef, const std::string& ipAddr, const u32 port)
{
    auto portRangeIter = ipPortRef.equal_range(ipAddr);
    for (auto it = portRangeIter.first; it != portRangeIter.second; ++it) {
        if (it->second.first == port && it->second.second.Count() > 0) {
            return true;
        }
    }
    return false;
}

std::string PreemptPortManager::GetRangeStr(const std::vector<SocketPortRange>& portRangeVec)
{
    std::ostringstream portRangeOss;
    for (auto range : portRangeVec) {
        portRangeOss << " [" << std::to_string(range.min) << ", " << std::to_string(range.max) << "]";
    }
    return portRangeOss.str();
}

void SplitPortRange(
    const std::vector<SocketPortRange>& fullRange, std::vector<SocketPortRange>& frontHalf,
    std::vector<SocketPortRange>& backHalf)
{
    // calc the total number of ports across all ranges
    u32 totalCount = 0;
    for (const auto& range : fullRange) {
        totalCount += range.max - range.min + 1;
    }

    if (totalCount == 0) {
        HCCL_WARNING("No available port to split");
        return;
    }

    if (totalCount == 1) {
        HCCL_WARNING("Only one available port to split, return the same range for both PortRanges");
        frontHalf = fullRange;
        backHalf = fullRange;
        return;
    }

    u32 frontCount = totalCount / 2;
    u32 allocated = 0;
    for (const auto& range : fullRange) {
        u32 count = range.max - range.min + 1;
        if (allocated + count <= frontCount) {
            frontHalf.emplace_back(range);
            allocated += count;
        } else if (allocated >= frontCount) {
            backHalf.emplace_back(range);
        } else {
            // frontCount boundary falls in the middle of this range
            u32 splitCount = frontCount - allocated;
            frontHalf.emplace_back(SocketPortRange{range.min, range.min + splitCount - 1});
            backHalf.emplace_back(SocketPortRange{range.min + splitCount, range.max});
            allocated = frontCount;
        }
    }
}
} // namespace Hccl
