/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_info_detect_service.h"

#include <algorithm>
#include <stdio.h>
#include "rank_info_dispatcher.h"
#include "env_config/env_config_v2.h"
#include "host_buffer.h"
#include "root_handle_v2.h"
#include "hccp_peer_manager.h"
#include "orion_adapter_rts.h"
#include "preempt_port_manager_v2.h"
#include "host_socket_handle_manager.h"
#include "adapter_error_manager_pub.h"
#include "common/hccl_log_keywords_pub.h"
#include "sal.h"
#include "rank_info_detect.h"

#include <iostream>

namespace Hccl {

const u32 DISPLAY_RANKNUM_PERLINE = 8;
const u32 SOCKET_ACCEPT_TIMEOUT = 60; // Server调用Accept等待的最大超时时间 60s
const u32 SOCKET_PRINT_COUNT = 3;     // 未建链打印的数量
const u32 MAX_AGENT_BUF_SIZE = 256;

// public static 成员定义：root 进程内主线程与 accept 线程之间的同步状态
UniversalConcurrentMap<u32, u32> RankInfoDetectService::g_detectServerStatus_;

HcclResult RankInfoDetectService::MarkGrowLoopReady()
{
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (shutdown_ || threadExited_ || growState_ != GrowState::STARTING) {
            HCCL_ERROR(
                "[RankInfoDetectService::%s] grow loop not ready, state[%d].", __func__, static_cast<int>(growState_));
            return HCCL_E_INTERNAL;
        }
        growState_ = GrowState::IDLE;
    }
    cv_.notify_all();
    return HCCL_SUCCESS;
}

void RankInfoDetectService::MarkGrowThreadExited(HcclResult result)
{
    {
        std::lock_guard<std::mutex> lock(mtx_);
        threadExited_ = true;
        if (shutdown_ || growState_ == GrowState::SHUTDOWN) {
            growState_ = GrowState::SHUTDOWN;
        } else {
            growResult_ = result;
            if (growState_ != GrowState::ERROR) {
                growState_ = result == HCCL_SUCCESS ? GrowState::IDLE : GrowState::ERROR;
            }
        }
    }
    cv_.notify_all();
}

HcclResult RankInfoDetectService::Setup()
{
    // 连接所有的 rank
    HCCL_INFO("[RankInfoDetectService::%s] initial detect auto-start.", __func__);

    HcclResult ret = GetConnections(true /*isInitCollect*/);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[RankInfoDetectService::%s] initial detect GetConnections failed, ret[%d].", __func__, ret);
        CHK_RET(Disconnect());
        CHK_RET(FinishCollect(ret));
        return ret;
    }

    // 2. 接收所有rank发来的localRankTable并整合为全局RankTable
    ret = GetRankTable(true /*isInitCollect*/);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[RankInfoDetectService::%s] initial detect GetRankTable failed, ret[%d].", __func__, ret);
        CHK_RET(Disconnect());
        rankTable_ = RankTableInfo{};
        failedAgentIdList_.clear();
        CHK_RET(FinishCollect(HCCL_E_INTERNAL));
        return HCCL_E_INTERNAL;
    }
    ret = BroadcastRankTable();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[RankInfoDetectService::%s] initial detect BroadcastRankTable failed, ret[%d].", __func__, ret);
        CHK_RET(Disconnect());
        rankTable_ = RankTableInfo{};
        failedAgentIdList_.clear();
        CHK_RET(FinishCollect(HCCL_E_INTERNAL));
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO(
        "[RankInfoDetectService::%s] initial detect complete, connSockets size[%zu].", __func__, GetConnSocketCount());

    // 保留 connSockets_ 供 Grow 复用，老 rank 的 WaitForNextBroadcast 依赖持久连接。
    if (serverSocket_ != nullptr) {
        EXCEPTION_CATCH(
            serverSocket_->StopListen(),
            HCCL_ERROR("[RankInfoDetectService::%s] StopListen exception in lazy pause", __func__));
    }
    HCCL_INFO(
        "[RankInfoDetectService::%s] initial detect done, socket kept, StopListen only, port[%u].", __func__,
        listenPort_);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectService::SetupGrowLoop()
{
    threadExited_ = false;
    HCCL_INFO("[RankInfoDetectService::%s] grow loop thread start, listenPort[%u].", __func__, listenPort_);
    HcclResult loopResult = HCCL_SUCCESS;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this]() {
                return growState_ == GrowState::RUNNING || growState_ == GrowState::SHUTDOWN || shutdown_;
            });
        }
        if (shutdown_) {
            break;
        }

        HCCL_INFO(
            "[RankInfoDetectService::%s] grow round started, expectedNewRankCount[%u].", __func__,
            expectedNewRankCount_);

        HcclResult ret = GetConnections(false /*isInitCollect*/);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[RankInfoDetectService::%s] GetConnections failed, ret[%d].", __func__, ret);
            loopResult = ret;
            CHK_RET(FinishGrowCollect(loopResult));
            break;
        }

        RankTableInfo savedRankTable = rankTable_;
        ret = GetRankTableForGrow();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[RankInfoDetectService::%s] grow round GetRankTable failed, ret[%d].", __func__, ret);
            loopResult = RollbackNewConnections();
            rankTable_ = savedRankTable;
            failedAgentIdList_.clear();
            if (loopResult == HCCL_SUCCESS) {
                loopResult = HCCL_E_INTERNAL;
            }
            CHK_RET(FinishGrowCollect(loopResult));
            break;
        }
        ret = BroadcastRankTableForGrow();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[RankInfoDetectService::%s] grow round BroadcastRankTable failed, ret[%d].", __func__, ret);
            loopResult = RollbackNewConnections();
            rankTable_ = savedRankTable;
            failedAgentIdList_.clear();
            if (loopResult == HCCL_SUCCESS) {
                loopResult = HCCL_E_INTERNAL;
            }
            CHK_RET(FinishGrowCollect(loopResult));
            break;
        }
        CHK_RET(FinishGrowCollect(HCCL_SUCCESS));
    }
    threadExited_ = true;
    HCCL_INFO("[RankInfoDetectService::%s] grow loop thread exited.", __func__);
    return loopResult;
}

HcclResult RankInfoDetectService::TriggerGrowCollect(u32 newRankCount)
{
    // 事件驱动：仅 Grow IDLE 可进入下一轮；ERROR/SHUTDOWN 均视为当前 service 不可继续工作。
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]() {
            return growState_ == GrowState::IDLE || growState_ == GrowState::ERROR || growState_ == GrowState::SHUTDOWN
                   || shutdown_;
        });
        if (shutdown_ || growState_ == GrowState::SHUTDOWN) {
            HCCL_ERROR("[RankInfoDetectService::%s] service shutdown during TriggerGrowCollect.", __func__);
            return HCCL_E_INTERNAL;
        }
        if (growState_ == GrowState::ERROR) {
            HCCL_ERROR(
                "[RankInfoDetectService::%s] service unavailable, state[%d].", __func__, static_cast<int>(growState_));
            return HCCL_E_INTERNAL;
        }
        expectedNewRankCount_ = newRankCount;
        {
            std::lock_guard<std::mutex> connLock(connMtx_);
            newAgentIds_.clear();
        }
        growState_ = GrowState::RUNNING;
    }
    cv_.notify_one(); // 唤醒 Setup 线程开始本次 Grow 收集
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectService::WaitForCollectComplete(RankTableInfo& rankTable)
{
    // 事件驱动：FinishCollect 设 IDLE/ERROR 后 notify 唤醒，零 CPU 占用
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this]() {
        return growState_ == GrowState::IDLE || growState_ == GrowState::ERROR || growState_ == GrowState::SHUTDOWN
               || shutdown_;
    });
    if (shutdown_ || growState_ == GrowState::SHUTDOWN) {
        HCCL_ERROR("[RankInfoDetectService::%s] service shutdown during round.", __func__);
        return HCCL_E_INTERNAL;
    }
    if (growState_ == GrowState::ERROR) {
        HCCL_ERROR("[RankInfoDetectService::%s] collect failed, ret[%d].", __func__, growResult_);
        return growResult_;
    }

    rankTable = rankTable_;
    return growResult_;
}

HcclResult RankInfoDetectService::FinishCollect(HcclResult result)
{
    {
        std::lock_guard<std::mutex> lock(mtx_);
        collectResult_ = result;
        SetStatus(result == HCCL_SUCCESS ? RANKINFO_DETECT_SERVER_STATUS_IDLE : RANKINFO_DETECT_SERVER_STATUS_ERROR);
    }
    cv_.notify_all();
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectService::FinishGrowCollect(HcclResult result)
{
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (shutdown_ || growState_ == GrowState::ERROR || growState_ == GrowState::SHUTDOWN) {
            HCCL_WARNING(
                "[RankInfoDetectService::%s] keep terminal state[%d], ignore result[%d].", __func__,
                static_cast<int>(growState_), result);
            return HCCL_SUCCESS;
        }
        growResult_ = result;
        growState_ = result == HCCL_SUCCESS ? GrowState::IDLE : GrowState::ERROR;
    }
    cv_.notify_all();
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectService::Shutdown()
{
    {
        std::lock_guard<std::mutex> lock(mtx_);
        shutdown_ = true;
        growState_ = GrowState::SHUTDOWN;
    }
    cv_.notify_all(); // 唤醒所有阻塞在 cv_.wait() 的线程，使其检查 shutdown_ 并退出
    // 停止监听以打断 accept 线程中可能阻塞的 Accept() 调用，确保线程能及时退出
    if (serverSocket_ != nullptr) {
        EXCEPTION_CATCH(
            serverSocket_->StopListen(), HCCL_ERROR("[RankInfoDetectService::%s] StopListen exception", __func__));
    }
    // 关闭已建立的连接，打断 Grow中可能阻塞在已建连接上的 recv/send，
    // 确保 serviceThread_ 不会因阻塞 recv/send 而无法退出，避免 join 永久阻塞。
    std::vector<std::shared_ptr<Socket>> socketsToClose;
    {
        std::lock_guard<std::mutex> lock(connMtx_);
        socketsToClose.reserve(connSockets_.size());
        for (auto& iter : connSockets_) {
            socketsToClose.push_back(iter.second);
        }
        newAgentIds_.clear();
    }
    for (auto& socket : socketsToClose) {
        if (socket != nullptr) {
            EXCEPTION_CATCH(
                socket->Close(), HCCL_ERROR("[RankInfoDetectService::%s] connSocket Close exception", __func__));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectService::GetConnections(bool isInitCollect)
{
    HCCL_INFO("[RankInfoDetectService::%s] start, isInitCollect[%d].", __func__, isInitCollect);

    // 超时参数
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    bool isFirstAcceptTimeOut = false;

    // 期望等待连接的rank数量。Init 起始为1；Grow root 已知 expectedNewRankCount_，不依赖client反向告知。
    u32 expectedSocketNum = 1;
    if (!isInitCollect) {
        expectedSocketNum = expectedNewRankCount_;
    }

    // 首个connect获取到的rankSize
    u32 previousRankNum = 0;
    HcclResult collectResult = HCCL_E_TIMEOUT;

    // 获取server端socket信息
    u32 hostPort = serverSocket_->GetListenPort();
    auto hccpHostSocketHandle = HostSocketHandleManager::GetInstance().Get(devPhyId_, hostIp_);
    CHK_PRT_RET(
        hccpHostSocketHandle == nullptr,
        HCCL_ERROR("[RankInfoDetectService::%s] Get hccpHostSocketHandle fail.", __func__), HCCL_E_INTERNAL);
    // identifier_ 由主线程在 UpdateIdentifier 中写入，此处为 accept 线程读取，加锁取本地副本
    std::string localIdentifier;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        localIdentifier = identifier_;
    }
    std::string connSocketTag = RANK_INFO_DETECT_TAG + "_" + localIdentifier + "_" + std::to_string(hostPort);
    SocketStatus status = SocketStatus::INVALID;

    // 连接rankSize个client
    while (expectedSocketNum > 0) {
        auto topoExUsedTime = std::chrono::steady_clock::now() - startTime;
        if (topoExUsedTime >= timeout) {
            RPT_INPUT_ERR(
                true, "EI0015", std::vector<std::string>({"error_reason"}),
                std::vector<std::string>({StringFormat(
                    "Waiting for worker node connections timed out. "
                    "Timeout was set to %lld seconds. Expected %u nodes, received %u nodes. "
                    "Check whether worker nodes are reachable and report errors",
                    static_cast<long long>(timeout.count()), expectedSocketNum + previousRankNum, previousRankNum)}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] topo exchange server get socket timeout[%lld s], "
                "expected[%u] nodes, received[%u] nodes",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_TIMEOUT), static_cast<long long>(timeout.count()),
                expectedSocketNum + previousRankNum, previousRankNum);
            break;
        }

        // duration_cast<seconds> 会进行向下取整，不足 1s
        // 时提前跳出，确保建链超时场景，server端在client端退出前发送临终遗言
        auto topoExResTime = timeout - topoExUsedTime;
        u32 topoExRes_i = std::chrono::duration_cast<std::chrono::seconds>(topoExResTime).count();
        if (topoExRes_i == 0) {
            HCCL_ERROR(
                "[RankInfoDetectService::%s] timeout[%lld s] is exhausted", __func__,
                static_cast<long long>(timeout.count()));
            break;
        }
        bool whiteListDisable = EnvConfig::GetInstance().GetHostNicConfig().GetWhitelistDisable();
        std::shared_ptr<Socket> connSocket = std::make_shared<Socket>(
            hccpHostSocketHandle, hostIp_, hostPort, hostIp_, connSocketTag, SocketRole::SERVER, NicType::HOST_NIC_TYPE,
            whiteListDisable);
        // GetStatus 是阻塞接口，传入剩余时间作为超时上限，避免其内部超时导致外层循环超时处理失效
        EXCEPTION_CATCH(status = connSocket->GetStatus(topoExRes_i), {
            // 非本端client首次连接异常，直接重试
            if (status == SocketStatus::OK) {
                status = SocketStatus::CONNECTING;
            }
            HCCL_ERROR("[RankInfoDetectService::%s] server get socket fail", __func__);
        });
        if (status == SocketStatus::OK) {
            // 白名单校验：非白名单连接直接丢弃并继续等待其它连接
            if (!IsRemoteIpInWhiteList(connSocket->GetRemoteIp())) {
                HCCL_WARNING(
                    "[RankInfoDetectService::%s] remoteIp[%s] is not in white list, discard the connection.", __func__,
                    connSocket->GetRemoteIp().GetIpStr().c_str());
                connSocket->Close();
                continue;
            }
            if (!RecvAndVerifyRemoteAgentIdAndRankSize(connSocket, expectedSocketNum, previousRankNum, isInitCollect)) {
                collectResult = HCCL_E_INTERNAL;
                break;
            }
            expectedSocketNum--;
            isFirstAcceptTimeOut = false;
            HCCL_INFO("[RankInfoDetectService::%s] socket[%s] connect ok.", __func__, connSocket->Describe().c_str());
        } else if (status == SocketStatus::CONNECTING) {
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else if (status == SocketStatus::TIMEOUT) {
            // 避免重复打印
            if (isFirstAcceptTimeOut) {
                continue;
            }
            HCCL_ERROR(
                "[%s][%s] topo exchange server get socket timeout[%lld s]", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), static_cast<long long>(timeout.count()));
            DisplayConnectingStatus(previousRankNum, expectedSocketNum);
            isFirstAcceptTimeOut = true;
        } else {
            HCCL_ERROR("[RankInfoDetectService::%s] SocketStatus[%s] error", __func__, status.Describe().c_str());
            collectResult = HCCL_E_INTERNAL;
            break;
        }
    }

    // 如果没有连接成功的rank则退出
    CHK_PRT_RET(
        GetConnSocketCount() == 0, HCCL_ERROR("[RankInfoDetectService::%s] no rank connection success.", __func__),
        HCCL_E_INTERNAL);

    // 处理异常流程
    if (expectedSocketNum > 0) {
        // 将建立连接超时的client信息添加到failedAgentIdList_
        // Grow 路径 previousRankNum 始终为 0（不从 client 反推 rankSize），改用 expectedNewRankCount_
        CHK_RET(FailedConnectionAgentIdString(isInitCollect ? previousRankNum : expectedNewRankCount_));
        DisplayConnectedRanks();
        if (!isInitCollect) {
            HCCL_ERROR(
                "[RankInfoDetectService::%s] grow collection is incomplete, ret[%d], expectedNewRankCount[%u], "
                "remaining[%u].",
                __func__, collectResult, expectedNewRankCount_, expectedSocketNum);
            CHK_RET(RollbackNewConnections());
            failedAgentIdList_.clear();
            return collectResult;
        }
        HCCL_INFO("[RankInfoDetectService::%s] end, there exist non-connected ranks.", __func__);
    } else {
        HCCL_INFO("[RankInfoDetectService::%s] end, all agentId get connection socket success.", __func__);
    }
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectService::GetRankTable(bool isInitCollect)
{
    HCCL_INFO("[RankInfoDetectService::%s] start, isInitCollect[%d].", __func__, isInitCollect);

    if (isInitCollect) {
        // Init：从空表开始，全量 recv 所有连接（全是新连接）构建 rankTable_
        rankTable_ = RankTableInfo{};
        for (auto& iter : connSockets_) {
            vector<char> rankInfoMsg{};
            SocketAgent socketAgent(iter.second.get());
            HcclResult ret = RecvRankInfoMsg(socketAgent, rankInfoMsg);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[RankInfoDetectService::%s] RecvRankInfoMsg failed, ret[%d].", __func__, ret), ret);
            ret = ParseRankTable(rankInfoMsg, false);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[RankInfoDetectService::%s] ParseRankTable failed, ret[%d].", __func__, ret), ret);
        }
    } else {
        // Grow：不清空 rankTable_（保留老 rank 拓扑），仅从本次 Grow新增连接 recv 并 merge
        // 老 rank 在 WaitForNextBroadcast 阻塞 recv，不会 send localRankInfo，遍历 connSockets_ 会永久阻塞
        for (const auto& agentId : newAgentIds_) {
            auto iter = connSockets_.find(agentId);
            CHK_PRT_RET(
                iter == connSockets_.end(),
                HCCL_ERROR(
                    "[RankInfoDetectService::%s] newAgentId[%s] not in connSockets_.", __func__, agentId.c_str()),
                HCCL_E_INTERNAL);
            vector<char> rankInfoMsg{};
            SocketAgent socketAgent(iter->second.get());
            HcclResult ret = RecvRankInfoMsg(socketAgent, rankInfoMsg);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[RankInfoDetectService::%s] RecvRankInfoMsg failed, ret[%d].", __func__, ret), ret);
            ret = ParseRankTable(rankInfoMsg, true); // 内部 UpdateRankTable 追加 merge，不清空 rankTable_
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[RankInfoDetectService::%s] ParseRankTable failed, ret[%d].", __func__, ret), ret);
        }
    }

    SortRankTable();

    // 更新当前阶段
    currentStep_++;

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectService::GetRankTableForGrow()
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    // Grow：不清空 rankTable_（保留老 rank 拓扑），仅从本次 Grow 新增连接 recv 并 merge。
    // 先持锁拷贝连接快照，锁外阻塞接收，避免 Shutdown/回滚与容器遍历并发。
    std::unordered_map<std::string, std::shared_ptr<Socket>> sockets;
    std::vector<std::string> agentIds;
    {
        std::lock_guard<std::mutex> lock(connMtx_);
        sockets = connSockets_;
        agentIds = newAgentIds_;
    }
    for (const auto& agentId : agentIds) {
        auto iter = sockets.find(agentId);
        CHK_PRT_RET(
            iter == sockets.end(),
            HCCL_ERROR("[RankInfoDetectService::%s] newAgentId[%s] not in connSockets_.", __func__, agentId.c_str()),
            HCCL_E_INTERNAL);
        vector<char> rankInfoMsg{};
        SocketAgent socketAgent(iter->second.get());
        HcclResult ret = RecvRankInfoMsg(socketAgent, rankInfoMsg);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[RankInfoDetectService::%s] RecvRankInfoMsg failed, ret[%d].", __func__, ret), ret);
        ret = ParseRankTable(rankInfoMsg, true);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[RankInfoDetectService::%s] ParseRankTable failed, ret[%d].", __func__, ret), ret);
    }

    SortRankTable();
    currentStep_++;
    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectService::BroadcastRankTable()
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    // 广播全局ranktable
    std::shared_ptr<RankInfoDispatcher> dispatcher = std::make_shared<RankInfoDispatcher>(this);
    CHK_RET(dispatcher->BroadcastRankTable(connSockets_, rankTable_, failedAgentIdList_, currentStep_));

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectService::BroadcastRankTableForGrow()
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    std::shared_ptr<RankInfoDispatcher> dispatcher = std::make_shared<RankInfoDispatcher>(this);
    auto sockets = SnapshotConnSockets();
    CHK_RET(dispatcher->BroadcastRankTable(sockets, rankTable_, failedAgentIdList_, currentStep_));

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectService::Disconnect()
{
    std::unordered_map<std::string, std::shared_ptr<Socket>> sockets;
    {
        std::lock_guard<std::mutex> lock(connMtx_);
        sockets.swap(connSockets_);
        newAgentIds_.clear();
    }
    for (auto& iter : sockets) {
        if (iter.second != nullptr) {
            EXCEPTION_CATCH(
                iter.second->Close(), HCCL_ERROR("[RankInfoDetectService::%s] socket Close exception", __func__));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectService::RollbackNewConnections()
{
    std::vector<std::shared_ptr<Socket>> socketsToClose;
    size_t newAgentIdCount = 0;
    {
        std::lock_guard<std::mutex> lock(connMtx_);
        newAgentIdCount = newAgentIds_.size();
        for (const auto& agentId : newAgentIds_) {
            auto iter = connSockets_.find(agentId);
            if (iter == connSockets_.end()) {
                HCCL_WARNING(
                    "[RankInfoDetectService::%s] newAgentId[%s] not in connSockets_.", __func__, agentId.c_str());
                continue;
            }
            socketsToClose.push_back(iter->second);
            connSockets_.erase(iter);
        }
        newAgentIds_.clear();
    }
    HCCL_INFO("[RankInfoDetectService::%s] start, newAgentIdCount[%zu].", __func__, newAgentIdCount);
    for (auto& socket : socketsToClose) {
        EXCEPTION_CATCH(socket->Close(), HCCL_ERROR("[RankInfoDetectService::%s] socket Close exception", __func__));
    }
    return HCCL_SUCCESS;
}

bool RankInfoDetectService::RecvRemoteAgentId(SocketAgent& connSocketAgent, std::string& agentId)
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    // 接收消息
    u64 revMsgLen = 0;
    char msg[MAX_AGENT_BUF_SIZE] = {0};
    bool ret = connSocketAgent.RecvMsg(msg, revMsgLen);
    CHK_PRT_RET(
        !ret || revMsgLen >= MAX_AGENT_BUF_SIZE,
        HCCL_ERROR("[RankInfoDetectService::%s] recv error, revMsgLen[%llu].", __func__, revMsgLen), false);

    // 解析agentId
    msg[revMsgLen] = '\0';
    agentId = msg;

    HCCL_INFO("[RankInfoDetectService::%s] agentId[%s]", __func__, agentId.c_str());
    return true;
}

bool RankInfoDetectService::RecvRemoteRankSize(SocketAgent& connSocketAgent, u32& rankSize)
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    // 接收rankSize
    u64 revMsgLen = 0;
    bool ret = connSocketAgent.RecvMsg(&rankSize, revMsgLen, sizeof(rankSize));
    CHK_PRT_RET(
        !ret, HCCL_ERROR("[RankInfoDetectService::%s] RecvMsg fail, revMsgLen[%llu].", __func__, revMsgLen), false);

    HCCL_INFO("[RankInfoDetectService::%s] rankSize[%u]", __func__, rankSize);
    return true;
}

// 接收客户端发送的字节流形式的rankinfo消息
HcclResult RankInfoDetectService::RecvRankInfoMsg(SocketAgent& connSocketAgent, vector<char>& rankInfoMsg) const
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    u64 revMsgLen = 0;
    std::unique_ptr<HostBuffer> msg = std::make_unique<HostBuffer>(MAX_BUFFER_LEN);
    char* msgAddr = reinterpret_cast<char*>(msg->GetAddr());
    bool recvOk = false;
    TRY_CATCH_RETURN(recvOk = connSocketAgent.RecvMsg(msgAddr, revMsgLen););
    CHK_PRT_RET(
        !recvOk, HCCL_ERROR("[RankInfoDetectService::%s] RecvMsg fail, revMsgLen[%llu]", __func__, revMsgLen),
        HCCL_E_NETWORK);

    // 以vector<char>格式保存
    rankInfoMsg.resize(revMsgLen);
    rankInfoMsg.assign(msgAddr, msgAddr + revMsgLen);

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
    return HCCL_SUCCESS;
}

// 解析接收到的rank table信息
HcclResult RankInfoDetectService::ParseRankTable(vector<char>& rankInfoMsg, bool isGrowCollect)
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    // 消息格式: [ranktable数据(n字节)][step(4字节)]
    BinaryStream binStream(rankInfoMsg);

    // 解析localRankInfo
    RankTableInfo localRankInfo;
    TRY_CATCH_RETURN(localRankInfo = RankTableInfo(binStream););
    localRankInfo.Dump();

    // 解析step
    u32 receivedStep;
    TRY_CATCH_RETURN(binStream >> receivedStep;);

    // 校验step是否匹配
    u32 expectedStep = isGrowCollect ? 0 : currentStep_;
    RPT_INPUT_ERR(
        receivedStep != expectedStep, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
        std::vector<std::string>({std::to_string(receivedStep), "step", std::to_string(expectedStep)}));
    // Init 要求 step 与服务端一致；Grow的新 rank 首次接入，固定从 step0 开始
    CHK_PRT_RET(
        receivedStep != expectedStep,
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] Step mismatch: received %u, expected %u", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), receivedStep, expectedStep),
        HCCL_E_PARA);

    // 添加到rankTable_
    CHK_RET(rankTable_.UpdateRankTable(localRankInfo));

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
    return HCCL_SUCCESS;
}

bool RankIdCompare(const NewRankInfo& i, const NewRankInfo& j) { return (i.rankId < j.rankId); }

void RankInfoDetectService::SortRankTable()
{
    std::sort(rankTable_.ranks.begin(), rankTable_.ranks.end(), RankIdCompare);
}

HcclResult RankInfoDetectService::FailedConnectionAgentIdString(u32 rankSize)
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    std::vector<bool> connectedRank(rankSize, false);
    auto sockets = SnapshotConnSockets();
    for (auto it : sockets) {
        u32 rankid = 0;
        HcclResult ret = SalStrToULong(it.first, HCCL_BASE_DECIMAL, rankid);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[RankInfoDetectService::%s] agentId[%s] strToULong fail.", __func__, it.first.c_str()),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(
            rankid >= rankSize,
            HCCL_ERROR("[RankInfoDetectService::%s] invalid rank id[%u], rankSize[%u].", __func__, rankid, rankSize),
            HCCL_E_INTERNAL);
        connectedRank[rankid] = true;
    }

    for (u32 i = 0; i < rankSize; i++) {
        if (!connectedRank[i]) {
            if (!failedAgentIdList_.empty()) {
                failedAgentIdList_ += ',';
            }
            failedAgentIdList_ += std::to_string(i);
        }
    }

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
    return HCCL_SUCCESS;
}

// 白名单开启时校验远端IP是否在拓扑探测白名单内：仅校验remoteIp(tag 由驱动层白名单校验)
bool RankInfoDetectService::IsRemoteIpInWhiteList(const IpAddress& remoteIp) const
{
    bool whiteListDisable = EnvConfig::GetInstance().GetHostNicConfig().GetWhitelistDisable();
    if (whiteListDisable) {
        HCCL_DEBUG("[RankInfoDetectService::%s] whitelist is disabled, skip remote ip check.", __func__);
        return true;
    }

    for (const auto& wlistEntry : wlistInfo_) {
        if (remoteIp == wlistEntry.remoteIp) {
            HCCL_DEBUG(
                "[RankInfoDetectService::%s] remoteIp[%s] is in white list.", __func__, remoteIp.GetIpStr().c_str());
            return true;
        }
    }

    return false;
}

// 校验相关方法
bool RankInfoDetectService::RecvAndVerifyRemoteAgentIdAndRankSize(
    std::shared_ptr<Socket> connSocket, u32& expectedSocketNum, u32& previousRankSize, bool isInitCollect)
{
    HCCL_INFO("[RankInfoDetectService::%s] start, isInitCollect[%d].", __func__, isInitCollect);
    SocketAgent socketAgent(connSocket.get());

    // 接收AgentId
    std::string agentId = "";
    bool ret = RecvRemoteAgentId(socketAgent, agentId);
    CHK_PRT_RET(!ret, HCCL_ERROR("[RankInfoDetectService::%s] RecvRemoteAgentId fail.", __func__), false);

    // 保存connSocket
    bool duplicateAgentId = false;
    {
        std::lock_guard<std::mutex> lock(connMtx_);
        if (shutdown_) {
            return false;
        }
        auto iter = connSockets_.find(agentId);
        duplicateAgentId = iter != connSockets_.end();
        if (!duplicateAgentId) {
            connSockets_.insert({agentId, connSocket});
            // Grow：记录本次 Grow新增连接的 agentId，后续 GetRankTable 只从新连接接收信息
            if (!isInitCollect) {
                newAgentIds_.push_back(agentId);
            }
        }
    }
    CHK_PRT_RET(
        duplicateAgentId,
        HCCL_ERROR("[RankInfoDetectService::%s] agentId[%s] has been connected.", __func__, agentId.c_str()), false);

    // 接收RankSize
    u32 rankSize = 0;
    ret = RecvRemoteRankSize(socketAgent, rankSize);
    CHK_PRT_RET(!ret, HCCL_ERROR("[RankInfoDetectService::%s] RecvRemoteAgentId fail.", __func__), false);

    // Init：靠首个 client 反推 rankSize 并校验所有 client rankSize 一致；
    // Grow：root 已知 expectedNewRankCount_，不依赖 client 反向告知、也不校验 rankSize
    //          （Grow新 rank 发的 rankSize 语义不同，强校验会误判 fail）
    if (isInitCollect) {
        expectedSocketNum = (previousRankSize == 0) ? rankSize : expectedSocketNum;
        CHK_PRT_RET(
            !VerifyRemoteRankSize(previousRankSize, rankSize),
            HCCL_ERROR("[RankInfoDetectService::%s] VerifyRemoteRankSize fail, rankSize[%u]", __func__, rankSize),
            false);
    }

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
    return true;
}

bool RankInfoDetectService::VerifyRemoteRankSize(u32& previousRankSize, u32 remoteRankSize) const
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    if (previousRankSize == 0) {
        previousRankSize = remoteRankSize;
    } else {
        if (previousRankSize != remoteRankSize) {
            HCCL_ERROR(
                "[RankInfoDetectService::%s] VerifyRemoteRankSize failed. remoteRankSize[%u] is different "
                "from previousRankSize[%u].",
                __func__, remoteRankSize, previousRankSize);
            return false;
        }
    }

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
    return true;
}

// DFX相关方法
void RankInfoDetectService::DisplayConnectedRanks()
{
    vector<std::string> ranksInfo;
    auto sockets = SnapshotConnSockets();
    for (const auto& it : sockets) {
        ranksInfo.push_back(it.first);
    }
    u64 ranksLen = ranksInfo.size();
    u64 lineNum = (ranksInfo.size() % DISPLAY_RANKNUM_PERLINE == 0) ? (ranksInfo.size() / DISPLAY_RANKNUM_PERLINE) :
                                                                      (ranksInfo.size() / DISPLAY_RANKNUM_PERLINE + 1);
    HCCL_ERROR(
        "[RankInfoDetectService::%s] total connected num is [%llu],line num is [%llu]", __func__, ranksLen, lineNum);
    for (u64 i = 0; i < lineNum; i++) {
        std::string tmpRankList;
        for (u32 j = 0; j < DISPLAY_RANKNUM_PERLINE; j++) {
            u32 ranksInfoIndex = i * DISPLAY_RANKNUM_PERLINE + j;
            if (ranksInfoIndex < ranksInfo.size()) {
                tmpRankList += "[" + ranksInfo[ranksInfoIndex] + "]";
            } else {
                break;
            }
            tmpRankList += ((j == DISPLAY_RANKNUM_PERLINE - 1 || ranksInfoIndex == ranksInfo.size() - 1) ? ";" : ",");
        }
        HCCL_ERROR("[RankInfoDetectService::%s] connected rankinfo[LINE %llu]: %s", __func__, i, tmpRankList.c_str());
    }
}

void RankInfoDetectService::DisplayConnectingStatus(u32 totalSockets, u32 waitSockets)
{
    if (totalSockets == 0 && waitSockets == 1) {
        HCCL_INFO("[RankInfoDetectService::%s] wait for first connection.", __func__);
        return;
    }

    std::vector<bool> rankinfos(totalSockets, false);
    auto sockets = SnapshotConnSockets();
    for (auto it : sockets) { // 建立映射
        u32 rankid = 0;
        HcclResult ret = SalStrToULong(it.first, HCCL_BASE_DECIMAL, rankid);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[RankInfoDetectService::%s] agentId[%s] strToULong fail.", __func__, it.first.c_str()), );
        CHK_PRT_RET(
            rankid >= totalSockets,
            HCCL_ERROR(
                "[RankInfoDetectService::%s] invalid rankid[%u], rankSize[%u].", __func__, rankid, totalSockets), );
        rankinfos[rankid] = true;
    }

    u32 unRankCount = 0; // 只打印前三条未建链的rank
    std::vector<std::string> unsocketinfos;
    for (u32 rankid = 0; rankid < totalSockets; rankid++) {
        if (unRankCount >= SOCKET_PRINT_COUNT) {
            break;
        }
        if (!rankinfos[rankid]) {
            unRankCount++;
            std::string rankID = std::to_string(rankid);
            std::string agentID = std::string(16 - rankID.length(), '0') + rankID;
            unsocketinfos.push_back(agentID);
        }
    }

    std::string infoStr = "succ sockets are [" + std::to_string((totalSockets - waitSockets))
                          + "], waiting sockets is [" + std::to_string(waitSockets) + "], wait sockets rankid: ";
    for (u32 index = 0; index < unsocketinfos.size(); index++) {
        if (index == (unsocketinfos.size() - 1)) {
            infoStr += "[" + unsocketinfos[index] + "]";
        } else {
            infoStr += "[" + unsocketinfos[index] + "],";
        }
    }

    HCCL_INFO("[RankInfoDetectService::%s] %s", __func__, infoStr.c_str());
}

HcclResult RankInfoDetectService::TearDown()
{
    bool expected = false;
    if (!tornDown_.compare_exchange_strong(expected, true)) {
        HCCL_INFO("[RankInfoDetectService::%s] already torn down, skip.", __func__);
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    if (serverSocket_ == nullptr) {
        HCCL_INFO("[RankInfoDetectService::%s] serverSocket is null", __func__);
        return HCCL_SUCCESS;
    }
    CHK_RET(Disconnect());

    // 如果白名单使能则删除白名单
    if (!EnvConfig::GetInstance().GetHostNicConfig().GetWhitelistDisable()) {
        if (wlistInfo_.size() == 0) {
            HCCL_ERROR("whitelist is empty");
        } else {
            SocketHandle hostSocketHandle = HostSocketHandleManager::GetInstance().Get(devPhyId_, hostIp_);
            if (hostSocketHandle == nullptr) {
                HCCL_ERROR("[RankInfoDetectService::%s] host socket handle is null, skip whitelist delete.", __func__);
            } else {
                EXCEPTION_CATCH(
                    HrtRaSocketWhiteListDel(hostSocketHandle, wlistInfo_),
                    HCCL_ERROR("[RankInfoDetectService::%s] WhiteListDel exception", __func__));
            }
        }
    }

    s32 logicDevId = HrtGetDevice();
    if (EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange().size() > 0
        || EnvConfig::GetInstance().GetHostNicConfig().GetIfBasePort() == HCCL_INVALID_PORT) {
        // 若开启抢占监听端口
        EXCEPTION_CATCH(
            PreemptPortManager::GetInstance(logicDevId).Release(serverSocket_),
            HCCL_ERROR("[RankInfoDetectService::%s] PreemptPort Release exception", __func__));
    } else {
        // 停止监听
        EXCEPTION_CATCH(
            serverSocket_->StopListen(), HCCL_ERROR("[RankInfoDetectService::%s] StopListen exception", __func__));
    }

    // 先释放监听 socket 对象，再释放其底层 handle 引用，避免重复 StopListen。
    serverSocket_ = nullptr;
    if (HostSocketHandleManager::GetInstance().Get(devPhyId_, hostIp_) != nullptr) {
        EXCEPTION_CATCH(
            HostSocketHandleManager::GetInstance().Destroy(devPhyId_, hostIp_),
            HCCL_ERROR("[RankInfoDetectService::%s] HostSocketHandle Destroy exception", __func__));
    }

    // deinit ra
    EXCEPTION_CATCH(
        HccpPeerManager::GetInstance().DeInit(logicDevId),
        HCCL_ERROR("[RankInfoDetectService::%s] HccpPeerManager DeInit exception", __func__));

    // 清理 g_detectServerStatus_ 中本端口的状态条目，避免端口复用时新域读到残留的 RUNNING/ERROR/IDLE
    g_detectServerStatus_.Erase(listenPort_);
    HCCL_INFO("[RankInfoDetectService::%s] erased server status for port[%u].", __func__, listenPort_);

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
    return HCCL_SUCCESS;
}

RankInfoDetectService::~RankInfoDetectService() { DECTOR_TRY_CATCH("RankInfoDetectService", TearDown()); }

} // namespace Hccl
