/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_DETECT_CONNECTION_ANOMALIES_H
#define HCOMM_DETECT_CONNECTION_ANOMALIES_H

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hccl/base.h"
#include "hcomm_res_defs.h"
#include "ip_address.h"
#include "referenced.h"
#include "port.h"
#include "socket.h"
#include "socket_config.h"
#include "socket_mgr.h"
#include "server_socket_manager.h"

namespace hcomm {

constexpr size_t DEST_MAX_LEN = 128;
constexpr uint32_t POLL_INTERVAL_100MS = 100; // 轮询间隔
constexpr uint32_t BROADCAST_WINDOW_SEC = 10; // 探测结果广播窗口（秒）
constexpr uint32_t SEND_BUDGET_MS = 200;      // 单个对端广播发送时间窗（毫秒）
constexpr uint32_t CLIENT_GETSTATUS_TIMEOUT_SEC = 1; // CreateClientSocketAndConnected 每次 GetStatus 阻塞等待时长（秒）
constexpr uint32_t ACCEPT_TIMEOUT_SEC = 1;      // AcceptSocket 单次 accept 阻塞等待时长（秒）
constexpr uint32_t RECV_POLL_INTERVAL_MS = 500; // client 收广播轮询间隔（毫秒）
constexpr uint32_t MAX_WHITE_LIST_ENTRY = 64;   // DCA 探测白名单连接上限

struct DcaPeerInfo {
    uint32_t remoteRankId{0};
    char remoteNetInstId[DEST_MAX_LEN]{}; // 对端机器标识（netLayer0 netInstId）
    uint32_t remoteDevPhyId{0};
    Hccl::IpAddress remoteIp;
    uint32_t remotePort{0};  // 对端端口（复用业务端口）
    Hccl::IpAddress localIp; // 每个rank有多个ip，需要和remoteIp配对
    EndpointDesc localEndpoint{};
    EndpointDesc remoteEndpoint{};
};

struct LocalIpInfo {
    Hccl::IpAddress ip;
    uint32_t port{0};
    EndpointDesc endpoint{};
};

inline Hccl::ConnectProtoType CommProtocolToConnectProtoType(CommProtocol commProtocol)
{
    switch (commProtocol) {
        case COMM_PROTOCOL_HCCS:
        case COMM_PROTOCOL_HCCS_ONLY:
            return Hccl::ConnectProtoType::HCCS;
        case COMM_PROTOCOL_ROCE:
            return Hccl::ConnectProtoType::RDMA;
        case COMM_PROTOCOL_PCIE:
            return Hccl::ConnectProtoType::PCIE;
        case COMM_PROTOCOL_UBOE:
        case COMM_PROTOCOL_UB_RTP:
        case COMM_PROTOCOL_UB_CTP:
        case COMM_PROTOCOL_UBC_TP:
        case COMM_PROTOCOL_UB_MEM:
        default:
            return Hccl::ConnectProtoType::UB;
    }
}

struct DetectInfo {
    int32_t localDeviceId = 0;
    int32_t remoteDeviceId = 0;
    char localIp[DEST_MAX_LEN]{};
    char remoteIp[DEST_MAX_LEN]{};
    char localNetInstId[DEST_MAX_LEN]{};  // 本端机器标识（netLayer0 netInstId）
    char remoteNetInstId[DEST_MAX_LEN]{}; // 对端机器标识（netLayer0 netInstId）
};

class DetectConnectionAnomalies {
public:
    void Init();
    void Deinit();
    void AddIpQueue(
        s32 localDevLogicId, uint32_t localDevPhyId, const std::string& localNetInstId,
        const std::vector<DcaPeerInfo>& failedPeers, const std::vector<LocalIpInfo>& localIps);

private:
    // ---------- 公共：调度与结果上报 ----------
    HcclResult Detect();
    void DetectMonitor();
    HcclResult WaitForDetect();
    HcclResult ProcessDetectionResults();
    std::vector<std::string>
    BuildGroupedDetectMessage(const std::unordered_map<std::string, DetectInfo>& snapshot) const;
    void ThreadDestroy();
    static std::string GetDetectKey(const DetectInfo& info);
    // 生成固定探测 tag：{Ip}_detect_0（IP 后缀固定 0；server 侧用本机监听 IP，client 侧用对端 IP）
    static std::string GetTag(const Hccl::IpAddress& ip);
    static uint32_t GetDfsConnFaultTime();
    // 锁内读取探测起始时间
    std::chrono::steady_clock::time_point GetStartTime() const;

    // ---------- server 侧：本端 IP → 监听 / accept / 广播 ----------
    HcclResult EnqueueLocalIpList(const std::vector<LocalIpInfo>& localIps);
    HcclResult GetLocalIpQueue();
    HcclResult ServerDetectThread(const LocalIpInfo& localInfo);
    std::pair<Hccl::PortData, Hccl::NicType> BuildLocalPortInfo(const LocalIpInfo& localInfo) const;
    HcclResult RunServerAcceptLoop(
        const LocalIpInfo& localInfo, uint32_t listenPort, Hccl::Socket* listener,
        std::vector<std::shared_ptr<Hccl::Socket>>& acceptSuccessSocketList);
    HcclResult RunBroadcastLoop(const std::vector<std::shared_ptr<Hccl::Socket>>& acceptSuccessSocketList);
    HcclResult BroadcastToSockets(const DetectInfo& info, const std::vector<std::shared_ptr<Hccl::Socket>>& sockets);

    // ---------- client 侧：失败对端 → 连接 / 收广播 ----------
    HcclResult EnqueuePeerIpList(const std::vector<DcaPeerInfo>& failedPeers);
    HcclResult GetPeerIpQueue();
    HcclResult ClientDetectThread(const DcaPeerInfo& peer);
    HcclResult CreateClientSocket(const DcaPeerInfo& peer, Hccl::Socket*& clientSocket);
    HcclResult CreateClientSocketAndConnected(const DcaPeerInfo& peer, Hccl::Socket*& clientSocket, bool& connected);
    HcclResult ReceiveDetectBroadcast(Hccl::Socket* clientSocket);
    HcclResult ConstructErrorInfo(const DcaPeerInfo& peer, DetectInfo& detectInfo);

    // ---------- 数据成员 ----------
    // 公共：入参缓存、生命周期与轮次
    s32 deviceLogicId_{-1};
    uint32_t localDevPhyId_{0};
    std::string localNetInstId_; // 本机机器标识（netLayer0 netInstId）
    std::once_flag localParamsOnce_;
    std::atomic<bool> threadRunning_{true};
    std::atomic<bool> isPrint_{false};
    Hccl::Referenced initRef_;
    std::chrono::steady_clock::time_point startTime_;
    mutable std::mutex timeMutex_;

    // 探测结果（client 写入、server 广播）
    std::unordered_map<std::string, DetectInfo> recvErrorInfoMap_; // key=检测结果（localIp-remoteIp），value=结果内容
    mutable std::mutex recvSendErrInfoMutex_;

    // monitor 调度
    std::once_flag monitorOnce_; // monitor 线程仅拉起一次（call_once）
    std::unique_ptr<std::thread> monitorThread_;

    // server 侧
    std::queue<LocalIpInfo> localIpQueue_;
    std::set<Hccl::IpAddress> knownLocalIps_;
    std::vector<std::unique_ptr<std::thread>> serverThreads_;
    mutable std::mutex serverMutex_;

    // client 侧
    std::queue<DcaPeerInfo> peerIpQueue_;
    std::set<Hccl::IpAddress> knownPeerIps_;
    std::vector<std::unique_ptr<std::thread>> clientThreads_;
    mutable std::mutex clientMutex_;
};

} // namespace hcomm

#endif // HCOMM_DETECT_CONNECTION_ANOMALIES_H
