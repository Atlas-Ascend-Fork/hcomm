/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "detect_connection_anomalies.h"

#include "socket_mgr.h"
#include "server_socket_mgr.h"

#include "orion_adpt_utils.h"
#include "base_comm_node_accessors.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "sal.h"
#include "env_config_v2.h"
#include "dev_type.h"
#include "orion_adapter_rts.h"
#include "adapter_error_manager_pub.h"
#include "hccl_common.h"
#include "log.h"

namespace hcomm {

void DetectConnectionAnomalies::Init()
{
    int ref = initRef_.Ref();
    HCCL_INFO("DetectConnectionAnomalies[Init] initRef[%d]", ref);
}

void DetectConnectionAnomalies::Deinit()
{
    int count = initRef_.Unref();
    if (count > 0) {
        HCCL_INFO("[DetectConnectionAnomalies]Deinit initRef_[%d]", count);
        return;
    } else if (count < 0) {
        HCCL_WARNING("[DetectConnectionAnomalies]Deinit failed");
        return;
    }
    ThreadDestroy();
    HCCL_INFO("DetectConnectionAnomalies[Deinit] count[%d]", count);
}

std::chrono::steady_clock::time_point DetectConnectionAnomalies::GetStartTime() const
{
    std::lock_guard<std::mutex> lk(timeMutex_);
    return startTime_;
}

HcclResult DetectConnectionAnomalies::WaitForDetect()
{
    const auto waitTime = std::chrono::seconds(GetDfsConnFaultTime() + BROADCAST_WINDOW_SEC);
    std::chrono::steady_clock::time_point localStartTime = GetStartTime();
    while (threadRunning_.load() && std::chrono::steady_clock::now() - localStartTime < waitTime) {
        std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_100MS)); // 每次休眠100毫秒
        localStartTime = GetStartTime();
    }
    if (!isPrint_.exchange(true)) {
        (void)ProcessDetectionResults();
    }
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::ProcessDetectionResults()
{
    std::string errMsg;
    HCCL_ERROR("-------------------CONNECT TIMEOUT DETECT RESULT-----------------------");
    std::unordered_map<std::string, DetectInfo> recvErrorInfo;
    {
        std::lock_guard<std::mutex> lk(recvSendErrInfoMutex_);
        recvErrorInfo = recvErrorInfoMap_;
    }
    if (!recvErrorInfo.empty()) {
        // 逐行打印聚合根因：每行一条 ERROR，避免单条日志超过日志打印长度上限被截断
        std::vector<std::string> msgLines = BuildGroupedDetectMessage(recvErrorInfo);
        for (const auto& line : msgLines) {
            HCCL_ERROR("%s", line.c_str());
            errMsg += line + "\n";
        }
        HCCL_ERROR("%s", GET_SOCKET_TIMEOUT_REASON_WITH_EVENT.c_str());
        errMsg += GET_SOCKET_TIMEOUT_REASON_WITH_EVENT;
    } else {
        errMsg = "This node detects no exception event. The possible cause is that the behaviors of different ranks "
                 "are inconsistent. The possible causes are as follows:";
        HCCL_ERROR("%s", errMsg.c_str());
        HCCL_ERROR("%s", GET_SOCKET_TIMEOUT_REASON_WITHOUT_EVENT.c_str());
        errMsg += "\n" + GET_SOCKET_TIMEOUT_REASON_WITHOUT_EVENT;
    }
    HCCL_ERROR("----------------------------------------------------------------------");
    RPT_INPUT_ERR(true, "EI0006", std::vector<std::string>({"reason"}), std::vector<std::string>({errMsg}));
    return HCCL_SUCCESS;
}

std::vector<std::string> DetectConnectionAnomalies::BuildGroupedDetectMessage(
    const std::unordered_map<std::string, DetectInfo>& recvErrorInfo) const
{
    // 按 (srcNetInstId, srcDeviceId) 分组 → (dstNetInstId → dstDeviceId 列表)
    std::map<std::tuple<std::string, int32_t>, std::map<std::string, std::vector<int32_t>>> summary;
    std::vector<std::string> result;
    for (const auto& item : recvErrorInfo) {
        const DetectInfo& info = item.second;
        summary[{info.localNetInstId, info.localDeviceId}][info.remoteNetInstId].push_back(info.remoteDeviceId);
    }
    // 确认按机器（netInstId）分组，跨机根节点聚合
    HCCL_INFO(
        "[%s] grouped by netInstId, groupNum[%zu], recvErrorInfoSize[%zu]", __func__, summary.size(),
        recvErrorInfo.size());
    std::string thisNetInstId = localNetInstId_;
    uint32_t thisDeviceId = localDevPhyId_;
    for (auto& srcEntry : summary) {
        const std::string& srcNetInstId = std::get<0>(srcEntry.first);
        const int32_t srcDeviceId = std::get<1>(srcEntry.first);
        for (auto& dstEntry : srcEntry.second) {
            const std::string& dstNetInstId = dstEntry.first;
            auto& dstDeviceIdList = dstEntry.second;
            std::sort(dstDeviceIdList.begin(), dstDeviceIdList.end());
            dstDeviceIdList.erase(std::unique(dstDeviceIdList.begin(), dstDeviceIdList.end()), dstDeviceIdList.end());
            std::string dstDeviceIdString = "[";
            for (size_t i = 0; i < dstDeviceIdList.size(); ++i) {
                if (i != 0) {
                    dstDeviceIdString += ",";
                }
                dstDeviceIdString += std::to_string(dstDeviceIdList[i]);
            }
            dstDeviceIdString += "]";
            std::ostringstream line;
            line << "This node (server " << thisNetInstId << ", device ID " << thisDeviceId
                 << ") detects that srcRank (server " << srcNetInstId << ", device ID " << srcDeviceId
                 << ") fails to connect to "
                 << "dstRank (server " << dstNetInstId << ", device ID " << dstDeviceIdString
                 << "). Continue to analyze the fault based on the logs of srcRank and dstRank.";
            result.push_back(line.str());
        }
    }
    return result;
}

void DetectConnectionAnomalies::AddIpQueue(
    s32 localDevLogicId, uint32_t localDevPhyId, const std::string& localNetInstId,
    const std::vector<DcaPeerInfo>& failedPeers, const std::vector<LocalIpInfo>& localIps)
{
    // 检查是否需要进行连接异常检测
    if (GetDfsConnFaultTime() == 0) {
        HCCL_RUN_INFO("[AddIpQueue] connection_fault_detection_time is 0, no need to detect");
        RPT_INPUT_ERR(
            true, "EI0006", std::vector<std::string>({"reason"}),
            std::vector<std::string>({GET_SOCKET_TIMEOUT_REASON_CLOSE_DETECT}));
        return;
    }
    if (!threadRunning_.load()) {
        HCCL_RUN_INFO("[AddIpQueue] detect thread is not running, skip detect");
        return;
    }
    // 检查设备类型是否支持
    if (Hccl::HrtGetDeviceType() != Hccl::DevType::DEV_TYPE_950) {
        HCCL_WARNING("[AddIpQueue] not support deviceType[%d]", Hccl::HrtGetDeviceType());
        RPT_INPUT_ERR(
            true, "EI0006", std::vector<std::string>({"reason"}),
            std::vector<std::string>({GET_SOCKET_TIMEOUT_REASON_CLOSE_DETECT}));
        return;
    }

    if (failedPeers.empty() || localIps.empty()) {
        HCCL_ERROR(
            "[%s] Empty input, skip: failedPeers size[%zu], localIps size[%zu]", __func__, failedPeers.size(),
            localIps.size());
        return;
    }

    std::call_once(localParamsOnce_, [&]() {
        localDevPhyId_ = localDevPhyId;
        localNetInstId_ = localNetInstId;
        deviceLogicId_ = localDevLogicId;
    });
    HCCL_INFO(
        "[%s] localDevPhyId[%u], localNetInstId[%s], failedPeers size[%zu], localIps size[%zu]", __func__,
        localDevPhyId, localNetInstId.c_str(), failedPeers.size(), localIps.size());

    {
        std::lock_guard<std::mutex> lk(timeMutex_);
        startTime_ = std::chrono::steady_clock::now(); // 刷新探测开始时间
    }
    Detect();
    EnqueueLocalIpList(localIps);   // server 侧入队
    EnqueuePeerIpList(failedPeers); // client 侧入队
    WaitForDetect();
}

HcclResult DetectConnectionAnomalies::EnqueueLocalIpList(const std::vector<LocalIpInfo>& localIps)
{
    // server 侧入队：新增本端 IP 入队（knownLocalIps_ 按 IP 去重）
    std::lock_guard<std::mutex> lk(serverMutex_);
    for (const auto& info : localIps) {
        if (knownLocalIps_.count(info.ip) == 0) {
            knownLocalIps_.insert(info.ip);
            localIpQueue_.push(info);
            HCCL_INFO(
                "[Add][LocalIpList]localIp[%s], locType[%d], port[%u]", info.ip.GetIpStr().c_str(),
                static_cast<int32_t>(info.endpoint.loc.locType), info.port);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::EnqueuePeerIpList(const std::vector<DcaPeerInfo>& failedPeers)
{
    // client 侧入队：失败对端入队（knownPeerIps_ 按 remoteIp 去重）
    std::lock_guard<std::mutex> lk(clientMutex_);
    for (const auto& peer : failedPeers) {
        if (knownPeerIps_.count(peer.remoteIp) == 0) {
            knownPeerIps_.insert(peer.remoteIp);
            peerIpQueue_.push(peer);
            HCCL_INFO(
                "[Add][PeerIpList]localIp[%s], remoteIp[%s], remotePort[%u], locType[%d], "
                "remoteRankId[%u], localNetInstId[%s], remoteNetInstId[%s]",
                peer.localIp.GetIpStr().c_str(), peer.remoteIp.GetIpStr().c_str(), peer.remotePort,
                static_cast<int32_t>(peer.remoteEndpoint.loc.locType), peer.remoteRankId, localNetInstId_.c_str(),
                peer.remoteNetInstId);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::Detect()
{
    // monitor 线程仅拉起一次；线程创建异常时 call_once 不置位，下次触发重试
    EXCEPTION_CATCH(
        std::call_once(
            monitorOnce_,
            [this]() {
                monitorThread_ = std::make_unique<std::thread>(&DetectConnectionAnomalies::DetectMonitor, this);
            }),
        return HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

void DetectConnectionAnomalies::DetectMonitor()
{
    SetThreadName("DcaMonitor");
    while (threadRunning_.load()) {
        GetLocalIpQueue(); // server 侧：本端 IP 队列 → 每 IP 一条 server 线程
        GetPeerIpQueue();  // client 侧：失败对端队列 → 每 peer 一条 client 线程
        std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_100MS));
    }
}

HcclResult DetectConnectionAnomalies::GetLocalIpQueue()
{
    // server 侧消费：逐条出队，每 IP 起一条 server 线程（入队已按 IP 去重，无重复）
    while (threadRunning_.load()) {
        LocalIpInfo localInfo;
        {
            std::lock_guard<std::mutex> lk(serverMutex_);
            if (localIpQueue_.empty()) {
                break;
            }
            localInfo = localIpQueue_.front();
            localIpQueue_.pop();
            EXCEPTION_CATCH(
                serverThreads_.push_back(
                    std::make_unique<std::thread>(&DetectConnectionAnomalies::ServerDetectThread, this, localInfo)),
                return HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::GetPeerIpQueue()
{
    while (threadRunning_.load()) {
        DcaPeerInfo queueEntry;
        {
            std::lock_guard<std::mutex> lk(clientMutex_);
            if (peerIpQueue_.empty()) {
                break;
            }
            queueEntry = peerIpQueue_.front();
            peerIpQueue_.pop();
            EXCEPTION_CATCH(
                clientThreads_.push_back(
                    std::make_unique<std::thread>(&DetectConnectionAnomalies::ClientDetectThread, this, queueEntry)),
                return HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}
std::pair<Hccl::PortData, Hccl::NicType>
DetectConnectionAnomalies::BuildLocalPortInfo(const LocalIpInfo& localInfo) const
{
    const bool isHost = (localInfo.endpoint.loc.locType == ENDPOINT_LOC_TYPE_HOST);
    const Hccl::ConnectProtoType protoType = CommProtocolToConnectProtoType(localInfo.endpoint.protocol);
    const Hccl::BasePortType portType = [&]() -> Hccl::BasePortType {
        if (protoType == Hccl::ConnectProtoType::HCCS || protoType == Hccl::ConnectProtoType::PCIE) {
            return Hccl::P2PPortType(protoType);
        } else if (isHost) {
            return Hccl::HostNetPortType(protoType);
        }
        return Hccl::DevNetPortType(protoType);
    }();
    Hccl::NicType nicType = isHost ? Hccl::NicType::HOST_NIC_TYPE : Hccl::NicType::DEVICE_NIC_TYPE;
    return {Hccl::PortData(static_cast<Hccl::RankId>(localDevPhyId_), portType, 0, localInfo.ip), nicType};
}

HcclResult DetectConnectionAnomalies::ServerDetectThread(const LocalIpInfo& localInfo)
{
    // 每个 server 线程只操作自己 localIp 的 socket
    SetThreadName("DcaDetectServer");
    if (deviceLogicId_ >= 0) {
        Hccl::HrtSetDevice(deviceLogicId_);
    }
    const std::string tag = GetTag(localInfo.ip);
    const auto [localPort, nicType] = BuildLocalPortInfo(localInfo);
    uint32_t listenPort = localInfo.port;

    bool listenStarted = false;
    bool whiteListAdded = false;
    std::shared_ptr<Hccl::Socket> listener = nullptr;
    std::vector<std::shared_ptr<Hccl::Socket>> acceptSuccessSocketList;
    HcclResult ret = HCCL_SUCCESS;
    do {
        ret = GetServerSocketMgr().ServerSocketStartListen(localPort, nicType, localDevPhyId_, &listenPort);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] start listen failed, ret[%d]", __func__, ret), );
        listenStarted = true;
        HCCL_INFO(
            "[CreateDetectNicLinks]Listen success, localIp[%s] tag[%s] port[%u]", localInfo.ip.GetIpStr().c_str(),
            tag.c_str(), listenPort);
        ret = GetServerSocketMgr().AddWhiteList(localPort, nicType, localDevPhyId_, tag, MAX_WHITE_LIST_ENTRY);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] add whitelist failed, ret[%d]", __func__, ret), );
        whiteListAdded = true;
        ret = GetServerSocketMgr().GetServerSocket(localPort, nicType, listenPort, listener);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] get server socket failed, ret[%d]", __func__, ret), );
        CHK_PRT_BREAK(listener == nullptr, HCCL_ERROR("[%s] listener is nullptr", __func__), ret = HCCL_E_PTR);
        ret = RunServerAcceptLoop(localInfo, listenPort, listener.get(), acceptSuccessSocketList);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] run accept loop failed, ret[%d]", __func__, ret), );
        ret = RunBroadcastLoop(acceptSuccessSocketList);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] run broadcast loop failed, ret[%d]", __func__, ret), );
    } while (false);

    // 收尾：accepted socket → 白名单 → 停止监听
    for (const auto& acceptedSocket : acceptSuccessSocketList) {
        GetSocketMgr(static_cast<s32>(localDevPhyId_)).DestroySocket(acceptedSocket.get());
    }
    if (whiteListAdded) {
        GetServerSocketMgr().DeleteWhiteList(localPort, nicType, localDevPhyId_, tag, MAX_WHITE_LIST_ENTRY);
    }
    if (listenStarted) {
        GetServerSocketMgr().ServerSocketStopListen(localPort, nicType, listenPort);
    }
    if (deviceLogicId_ >= 0) {
        Hccl::HrtResetDevice(deviceLogicId_);
    }
    return ret;
}

HcclResult DetectConnectionAnomalies::RunServerAcceptLoop(
    const LocalIpInfo& localInfo, uint32_t listenPort, Hccl::Socket* listener,
    std::vector<std::shared_ptr<Hccl::Socket>>& acceptSuccessSocketList)
{
    const std::string tag = GetTag(localInfo.ip);
    Hccl::LinkData listenLinkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localInfo.endpoint, localInfo.endpoint, listenLinkData));
    Hccl::SocketConfig acceptConfig
        = Hccl::SocketConfig(listenLinkData, tag, Hccl::SocketRole::SERVER, listenPort, true);

    const auto dfsTimeout = std::chrono::seconds(GetDfsConnFaultTime());
    auto acceptStartTime = GetStartTime();

    HCCL_INFO("[%s] accept window start, tag[%s]", __func__, tag.c_str());
    while (threadRunning_.load() && std::chrono::steady_clock::now() - acceptStartTime < dfsTimeout) {
        std::shared_ptr<Hccl::Socket> oneSocket = nullptr;
        HcclResult ret = HCCL_SUCCESS;
        EXCEPTION_CATCH(
            ret = GetSocketMgr(static_cast<s32>(localDevPhyId_))
                      .AcceptSocket(listener, acceptConfig, oneSocket, ACCEPT_TIMEOUT_SEC),
            { ret = HCCL_E_AGAIN; });
        if (ret == HCCL_SUCCESS && oneSocket != nullptr) {
            acceptSuccessSocketList.push_back(oneSocket);
            HCCL_INFO(
                "[%s] server accept ok, tag[%s], total socket count[%zu]", __func__, acceptConfig.tag.c_str(),
                acceptSuccessSocketList.size());
        }
        acceptStartTime = GetStartTime();
        std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_100MS));
    }
    // accept 窗口结束，累计 accept 到的 socket 数（对端探测连接数）
    HCCL_INFO("[%s] accept window end, total accept socket count[%zu]", __func__, acceptSuccessSocketList.size());
    return HCCL_SUCCESS;
}

HcclResult
DetectConnectionAnomalies::RunBroadcastLoop(const std::vector<std::shared_ptr<Hccl::Socket>>& acceptSuccessSocketList)
{
    const auto broadcastLoopStartTime = std::chrono::steady_clock::now();
    const auto broadcastLoopTimeOut = std::chrono::seconds(BROADCAST_WINDOW_SEC);
    // 本线程已广播结果（key=localIp-remoteIp），每条检测结果对本线程只发一次
    std::set<std::string> isSent;
    while (threadRunning_.load() && std::chrono::steady_clock::now() - broadcastLoopStartTime < broadcastLoopTimeOut) {
        // 收集未发送的故障根节点信息到toSend
        std::vector<std::pair<std::string, DetectInfo>> toSend;
        {
            std::lock_guard<std::mutex> lk(recvSendErrInfoMutex_);
            for (const auto& item : recvErrorInfoMap_) {
                if (isSent.count(item.first) == 0) { // 未发送过的信息入队
                    toSend.push_back(item);
                }
            }
        }
        // 广播故障根节点信息给自己 accept 到的 socket
        for (const auto& item : toSend) {
            BroadcastToSockets(item.second, acceptSuccessSocketList);
            isSent.insert(item.first); // 标记该信息已发送
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_100MS));
    }
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::BroadcastToSockets(
    const DetectInfo& info, const std::vector<std::shared_ptr<Hccl::Socket>>& sockets)
{
    constexpr u64 totalSize = sizeof(DetectInfo);
    constexpr uint32_t TEN_MILLISECOND_OF_USLEEP = 10000;
    std::vector<u8> sendBuf(totalSize);
    CHK_SAFETY_FUNC_RET(memcpy_s(sendBuf.data(), totalSize, &info, totalSize));
    uint32_t sentCount = 0;
    HCCL_INFO("[%s] broadcast to socket count[%zu]", __func__, sockets.size());
    for (const auto& sock : sockets) {
        u64 sendSize = 0; // 已发送长度
        // 单对端发送时间窗口200ms：EAGAIN/无进展时等待10ms重试
        const auto sendDeadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(SEND_BUDGET_MS);
        while (std::chrono::steady_clock::now() < sendDeadline) {
            u64 compSize = 0;                        // 本次发送长度
            u8* sendPos = sendBuf.data() + sendSize; // 偏移
            HcclResult ret = sock->ISendWithHeart(sendPos, totalSize - sendSize, compSize);
            if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN) {
                HCCL_WARNING(
                    "[%s] broadcast to socket[%p] failed, ret[%d], sent[%llu]/[%llu]", __func__, sock.get(), ret,
                    sendSize, totalSize);
                break;
            }
            if (ret == HCCL_E_AGAIN || compSize == 0) {
                SaluSleep(TEN_MILLISECOND_OF_USLEEP); // 10ms
                continue;
            }
            sendSize += std::min(compSize, totalSize - sendSize);
            if (sendSize == totalSize) {
                sentCount++;
                break;
            }
        }
    }
    HCCL_INFO(
        "[%s] broadcast localNetInstId[%s] remoteNetInstId[%s] to [%u] accepted sockets", __func__, info.localNetInstId,
        info.remoteNetInstId, sentCount);
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::ClientDetectThread(const DcaPeerInfo& peer)
{
    SetThreadName("DcaDetectClient");
    if (deviceLogicId_ >= 0) {
        Hccl::HrtSetDevice(deviceLogicId_);
    }
    // 探测线程启动日志：标识探测目标（对端机器 netInstId + IP + rank）
    HCCL_INFO(
        "[ClientDetect] start, remoteRankId[%u], remoteNetInstId[%s], remoteIp[%s], remotePort[%u], locType[%d]",
        peer.remoteRankId, peer.remoteNetInstId, peer.remoteIp.GetIpStr().c_str(), peer.remotePort,
        static_cast<int32_t>(peer.remoteEndpoint.loc.locType));

    Hccl::Socket* clientSocket = nullptr;
    bool connected = false;
    HcclResult ret = HCCL_SUCCESS;
    do {
        ret = CreateClientSocketAndConnected(peer, clientSocket, connected);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] create client socket failed, ret[%d]", __func__, ret), );
        CHK_PRT_BREAK(clientSocket == nullptr, HCCL_ERROR("[%s] clientSocket is nullptr", __func__), ret = HCCL_E_PTR);
        if (!connected) {
            DetectInfo detectInfo{};
            ret = ConstructErrorInfo(peer, detectInfo);
            CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] ConstructErrorInfo failed", __func__), );
            // 探测结果日志：探测对端失败，记录根因（本端机器→对端机器）
            HCCL_INFO(
                "[ClientDetect] DETECT FAIL, localNetInstId[%s] -> remoteNetInstId[%s], remoteRankId[%u], "
                "localDevId[%d] remoteDevId[%d]",
                detectInfo.localNetInstId, detectInfo.remoteNetInstId, peer.remoteRankId, detectInfo.localDeviceId,
                detectInfo.remoteDeviceId);
            {
                std::lock_guard<std::mutex> lk(recvSendErrInfoMutex_);
                recvErrorInfoMap_[GetDetectKey(detectInfo)] = detectInfo;
            }
        } else {
            // 探测结果日志：探测对端可达（级联中间节点，接收其广播来的故障根节点信息）
            HCCL_INFO(
                "[ClientDetect] DETECT OK, remoteNetInstId[%s], remoteRankId[%u] (cascade intermediate node)",
                peer.remoteNetInstId, peer.remoteRankId);
            ReceiveDetectBroadcast(clientSocket);
        }
    } while (false);

    HCCL_INFO(
        "[%s] completed, remoteRankId[%u], remoteNetInstId[%s], remoteIp[%s]", __func__, peer.remoteRankId,
        peer.remoteNetInstId, peer.remoteIp.GetIpStr().c_str());

    GetSocketMgr(static_cast<s32>(localDevPhyId_)).DestroySocket(clientSocket);
    if (deviceLogicId_ >= 0) {
        Hccl::HrtResetDevice(deviceLogicId_);
    }
    return ret;
}

HcclResult DetectConnectionAnomalies::CreateClientSocket(const DcaPeerInfo& peer, Hccl::Socket*& clientSocket)
{
    if (peer.remoteIp.IsInvalid() || peer.localIp.IsInvalid()) {
        HCCL_ERROR(
            "[%s] clientIp is invalid, localIp[%s], remoteIp[%s]", __func__, peer.localIp.GetIpStr().c_str(),
            peer.remoteIp.GetIpStr().c_str());
        return HCCL_E_NOT_FOUND;
    }

    const std::string clientTag = GetTag(peer.remoteIp);
    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(peer.localEndpoint, peer.remoteEndpoint, linkData));
    Hccl::SocketConfig socketConfig
        = Hccl::SocketConfig(linkData, clientTag, Hccl::SocketRole::CLIENT, peer.remotePort, true);

    CHK_RET(GetSocketMgr(static_cast<s32>(localDevPhyId_)).GetSocket(socketConfig, clientSocket));
    CHK_PTR_NULL(clientSocket);
    // 打印连接目标信息
    HCCL_INFO(
        "[Connect]tag[%s], localIp[%s], remoteIp[%s], port[%u], role[CLIENT], locType[%d](%s)", clientTag.c_str(),
        peer.localIp.GetIpStr().c_str(), peer.remoteIp.GetIpStr().c_str(), peer.remotePort,
        static_cast<int32_t>(peer.remoteEndpoint.loc.locType),
        (peer.remoteEndpoint.loc.locType == ENDPOINT_LOC_TYPE_HOST) ? "HOST" : "DEVICE");

    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::CreateClientSocketAndConnected(
    const DcaPeerInfo& peer, Hccl::Socket*& clientSocket, bool& connected)
{
    const uint32_t dfsTimeSec = GetDfsConnFaultTime();
    const auto connectTimeOut = std::chrono::seconds(dfsTimeSec);
    auto connectStartTime = GetStartTime();
    auto firstStartTime = connectStartTime;

    CHK_RET(CreateClientSocket(peer, clientSocket));
    CHK_PTR_NULL(clientSocket);
    connected = false;

    const std::string remoteIpStr = clientSocket->GetRemoteIp().GetIpStr();
    HCCL_INFO(
        "[%s] start connect and poll, dfsTime[%u]s, remoteIp[%s], pollInterval[%u]s", __func__, dfsTimeSec,
        remoteIpStr.c_str(), CLIENT_GETSTATUS_TIMEOUT_SEC);

    while (threadRunning_.load() && std::chrono::steady_clock::now() - connectStartTime < connectTimeOut) {
        Hccl::SocketStatus status = Hccl::SocketStatus::INIT;
        EXCEPTION_CATCH(status = clientSocket->GetStatus(CLIENT_GETSTATUS_TIMEOUT_SEC), {});
        // 打印每次轮询的状态
        HCCL_INFO(
            "[%s] poll status[%s], remoteIp[%s], elapsed[%lld]s", __func__, status.Describe().c_str(),
            remoteIpStr.c_str(),
            static_cast<long long>(
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - firstStartTime)
                    .count()));
        if (status == Hccl::SocketStatus::OK) {
            HCCL_INFO("[Detect][ConnectionAnomalies]GetStatus success, remoteIp[%s]", remoteIpStr.c_str());
            connected = true;
            return HCCL_SUCCESS;
        }
        connectStartTime = GetStartTime();
        std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_100MS));
    }
    HCCL_INFO("[Detect][ConnectionAnomalies]GetStatus timeout, remoteIp[%s]", remoteIpStr.c_str());
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::ConstructErrorInfo(const DcaPeerInfo& peer, DetectInfo& detectInfo)
{
    detectInfo.localDeviceId = static_cast<int32_t>(localDevPhyId_);
    detectInfo.remoteDeviceId = static_cast<int32_t>(peer.remoteDevPhyId);
    CHK_SAFETY_FUNC_RET(strcpy_s(detectInfo.localIp, DEST_MAX_LEN, peer.localIp.GetIpStr().c_str()));
    CHK_SAFETY_FUNC_RET(strcpy_s(detectInfo.remoteIp, DEST_MAX_LEN, peer.remoteIp.GetIpStr().c_str()));
    CHK_SAFETY_FUNC_RET(strcpy_s(detectInfo.localNetInstId, DEST_MAX_LEN, localNetInstId_.c_str()));
    CHK_SAFETY_FUNC_RET(strcpy_s(detectInfo.remoteNetInstId, DEST_MAX_LEN, peer.remoteNetInstId));
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::ReceiveDetectBroadcast(Hccl::Socket* clientSocket)
{
    CHK_PTR_NULL(clientSocket);
    HCCL_INFO("[%s] start recv broadcast window", __func__);
    constexpr u64 totalSize = sizeof(DetectInfo);
    std::vector<u8> recvBuf(totalSize);
    u64 recvSize = 0; // 已接收长度
    const auto recvEndTime
        = GetStartTime() + std::chrono::seconds(GetDfsConnFaultTime()) + std::chrono::seconds(BROADCAST_WINDOW_SEC);
    while (threadRunning_.load() && std::chrono::steady_clock::now() < recvEndTime) {
        u64 compSize = 0;                        // 本次接收长度
        u8* recvPos = recvBuf.data() + recvSize; // 偏移
        HcclResult ret = clientSocket->IRecvWithHeart(recvPos, totalSize - recvSize, compSize);
        if (ret == HCCL_SUCCESS && compSize > 0) {
            recvSize += std::min(compSize, totalSize - recvSize);
        }
        if (recvSize == totalSize) {
            recvSize = 0;
            DetectInfo detectInfo{};
            CHK_SAFETY_FUNC_RET(memcpy_s(&detectInfo, sizeof(detectInfo), recvBuf.data(), sizeof(detectInfo)));
            HCCL_INFO(
                "[%s] RECV broadcast, localNetInstId[%s] remoteNetInstId[%s], localDevId[%d] remoteDevId[%d]", __func__,
                detectInfo.localNetInstId, detectInfo.remoteNetInstId, detectInfo.localDeviceId,
                detectInfo.remoteDeviceId);
            std::lock_guard<std::mutex> lk(recvSendErrInfoMutex_);
            // 故障根节点信息入队，根据key(localIp-remoteIp)去重
            recvErrorInfoMap_[GetDetectKey(detectInfo)] = detectInfo;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(RECV_POLL_INTERVAL_MS)); // 每500ms接收一次
    }
    return HCCL_SUCCESS;
}

void DetectConnectionAnomalies::ThreadDestroy()
{
    HCCL_DEBUG("[%s] enter, destroy threads and resources", __func__);
    threadRunning_.store(false);
    // 1. 清理monitor 线程
    if (monitorThread_ && monitorThread_->joinable()) {
        monitorThread_->join();
    }
    monitorThread_.reset();
    // join 线程并清列表（线程退出即自行销毁其 socket/fd）
    auto joinAndClear = [](std::vector<std::unique_ptr<std::thread>>& threads) {
        for (auto& thread : threads) {
            if (thread && thread->joinable()) {
                thread->join();
            }
        }
        threads.clear();
    };
    // 2. server 侧：join 每条 server 线程，清空本端 IP 队列/去重集合
    {
        std::lock_guard<std::mutex> lk(serverMutex_);
        joinAndClear(serverThreads_);
        while (!localIpQueue_.empty()) {
            localIpQueue_.pop();
        }
        knownLocalIps_.clear();
    }
    // 3. client 侧：join 每条 client 线程（线程退出即自行销毁其 socket），清空对端队列/去重集合
    {
        std::lock_guard<std::mutex> lk(clientMutex_);
        joinAndClear(clientThreads_);
        while (!peerIpQueue_.empty()) {
            peerIpQueue_.pop();
        }
        knownPeerIps_.clear();
    }
}

uint32_t DetectConnectionAnomalies::GetDfsConnFaultTime()
{
    return static_cast<uint32_t>(
        Hccl::EnvConfig::GetInstance().GetLogConfig().GetDfsConfig().connectionFaultDetectionTime);
}

std::string DetectConnectionAnomalies::GetTag(const Hccl::IpAddress& ip) { return ip.GetIpStr() + "_detect_0"; }

// 检测结果 key = localIp-remoteIp
std::string DetectConnectionAnomalies::GetDetectKey(const DetectInfo& info)
{
    return std::string(info.localIp) + "-" + std::string(info.remoteIp);
}

} // namespace hcomm
