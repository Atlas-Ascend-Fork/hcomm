/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl_rt.h"
#include "rank_info_detect.h"
#include <thread>
#include "sal.h"
#include "hccp_peer_manager.h"
#include "hccp_hdc_manager.h"
#include "internal_exception.h"
#include "orion_adapter_rts.h"
#include "whitelist.h"
#include "host_socket_handle_manager.h"
#include "env_config/env_config_v2.h"
#include "bootstrap_ip.h"
#include "preempt_port_manager_v2.h"
#include "common/hccl_log_keywords_pub.h"

namespace Hccl {

constexpr u32 HOST_CONTROL_BASE_PORT = 60016; // 控制面起始port，当前专用于拓扑探测使用，范围在[60016,60031]
constexpr u32 HOST_CONTROL_PORT_COUNT = 15;
constexpr u32 HCCL_WHITELIST_ON = 1;
constexpr u32 HOST_SOCKET_CONN_LIMIT = 8; // HCCL_AISERVER_DEVICE_NUM (8)

// 触发 rank info 的Grow收集，返回合并后的 rankTable
HcclResult RankInfoDetect::TriggerGrowCollect(u32 newRankCount, RankTableInfo& mergedRankTable)
{
    if (rankInfoDetectService_ == nullptr) {
        HCCL_ERROR("[RankInfoDetect::%s] rankInfoDetectService_ is null, not a root process.", __func__);
        return HCCL_E_INTERNAL;
    }
    CHK_PRT_RET(
        !IsReadyForGrow(),
        HCCL_ERROR(
            "[RankInfoDetect::%s] detect service is not ready for grow, state[%d], threadExited[%d].", __func__,
            static_cast<int>(rankInfoDetectService_->GetGrowState()),
            static_cast<int>(rankInfoDetectService_->IsThreadExited())),
        HCCL_E_UNAVAIL);
    // lazy 模式：标准 init Setup() 后线程已退出，但 serverSocket 未销毁（只 StopListen + Disconnect）
    if (rankInfoDetectService_->IsThreadExited()) {
        if (!rankInfoDetectService_->IsSupportGrowLoop()) {
            HCCL_ERROR("[RankInfoDetect::%s] grow service is in terminal state.", __func__);
            return HCCL_E_INTERNAL;
        }
        if (serviceThread_.joinable()) {
            serviceThread_.join();
        }
        // 复用 Init 阶段保留的 serverSocket：Steal 后直接 Listen，无需 ServerInit / 白名单 / RA 重建
        shared_ptr<Socket> serverSocket = rankInfoDetectService_->StealServerSocket();
        CHK_PTR_NULL(serverSocket);
        // socket 保留，尝试重新 Listen，若端口被占用则返回错误。
        TRY_CATCH_RETURN(serverSocket->Listen());

        // 从旧 service move 走 rankTable_ 和 connSockets_，避免新 service 为空导致不完整 / 老连接丢失。
        auto prevRankTable = rankInfoDetectService_->StealRankTable();
        auto prevConnSockets = rankInfoDetectService_->StealConnSockets();
        // 先构造新 service，替换后再启动线程，保证线程创建失败时仍由 Shutdown 统一回收。
        auto newService = std::make_shared<RankInfoDetectService>(devPhyId_, serverSocket, identifier_, wlistInfo_);
        newService->SetRankTable(std::move(prevRankTable));
        newService->SetConnSockets(std::move(prevConnSockets));

        auto service = newService;
        auto errorContext = hrtErrMGetErrorContextPub();
        s32 devLogicIdCopy = devLogicId_;
        u32 listenPort = serverSocket->GetListenPort();
        // 先替换 service，确保线程创建失败时 Shutdown 能回收本次重建的资源。
        rankInfoDetectService_ = newService;
        HcclResult threadRet = HCCL_SUCCESS;
        EXCEPTION_CATCH(
            (serviceThread_ = std::thread([service, listenPort, errorContext, devLogicIdCopy]() {
                 hrtErrMSetErrorContextPub(errorContext);
                 HCCL_INFO("[RankInfoDetect] grow loop thread start, listenPort[%u].", listenPort);

                 HcclResult ret = HCCL_SUCCESS;
                 EXCEPTION_CATCH(HrtSetDevice(devLogicIdCopy), ret = HCCL_E_INTERNAL);
                 if (ret != HCCL_SUCCESS) {
                     service->MarkGrowThreadExited(ret);
                     HCCL_ERROR("[RankInfoDetect] grow loop thread set device failed, ret[%d].", ret);
                     return;
                 }

                 EXCEPTION_CATCH(ret = service->SetupGrowLoop(), ret = HCCL_E_INTERNAL);
                 HcclResult resetRet = HCCL_SUCCESS;
                 EXCEPTION_CATCH(HrtResetDevice(devLogicIdCopy), resetRet = HCCL_E_INTERNAL);
                 if (ret == HCCL_SUCCESS) {
                     ret = resetRet;
                 }
                 service->MarkGrowThreadExited(ret);
                 HCCL_INFO("[RankInfoDetect] grow loop thread end, ret[%d].", ret);
             })),
            threadRet = HCCL_E_INTERNAL);
        if (threadRet != HCCL_SUCCESS) {
            service->MarkGrowThreadExited(threadRet);
            HCCL_ERROR("[RankInfoDetect::%s] create grow loop thread failed.", __func__);
            return threadRet;
        }
        persistentStarted_ = true;
        CHK_RET(service->MarkGrowLoopReady());
        HCCL_INFO(
            "[RankInfoDetect::%s] grow loop thread restarted, reusing serverSocket, port[%u].", __func__, listenPort);
    }
    CHK_RET(rankInfoDetectService_->TriggerGrowCollect(newRankCount));
    CHK_RET(rankInfoDetectService_->WaitForCollectComplete(mergedRankTable));
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetect::WaitForNextBroadcast(RankTableInfo& rankTable, uint32_t expectedRankCount)
{
    if (rankInfoDetectClient == nullptr) {
        HCCL_ERROR("[RankInfoDetect::%s] rankInfoDetectClient is null.", __func__);
        return HCCL_E_INTERNAL;
    }
    CHK_RET(rankInfoDetectClient->WaitForNextBroadcast(rankTable, expectedRankCount));
    rankTable_ = rankTable;
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetect::StopPersistentServer()
{
    if (rankInfoDetectService_ == nullptr && !serviceThread_.joinable()) {
        if (persistentStarted_) {
            HCCL_WARNING(
                "[RankInfoDetect::%s] persistentStarted_ is true but service/thread is absent, reset state.", __func__);
        } else {
            HCCL_INFO("[RankInfoDetect::%s] no persistent server resource, skip.", __func__);
        }
        persistentStarted_ = false;
        return HCCL_SUCCESS;
    }
    if (rankInfoDetectService_) {
        EXCEPTION_CATCH(
            rankInfoDetectService_->Shutdown(),
            HCCL_ERROR("[RankInfoDetect::%s] rankInfoDetectService Shutdown exception", __func__));
    }
    if (serviceThread_.joinable()) {
        EXCEPTION_CATCH(
            serviceThread_.join(), HCCL_ERROR("[RankInfoDetect::%s] serviceThread join exception", __func__));
    }
    rankInfoDetectService_.reset();
    persistentStarted_ = false;
    HCCL_INFO("[RankInfoDetect::%s] persistent server stopped.", __func__);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetect::Shutdown()
{
    if (shutdownDone_.load()) {
        HCCL_WARNING(
            "[RankInfoDetect::%s] already shutdown, skip, shutdownDone[%d].", __func__,
            static_cast<int>(shutdownDone_.load()));
        return HCCL_SUCCESS;
    }

    CHK_RET(StopPersistentServer());
    if (rankInfoDetectClient != nullptr) {
        CHK_RET(rankInfoDetectClient->TearDown());
        rankInfoDetectClient.reset();
    } else {
        // SetupAgent 失败路径兜底：SetupAgent 已调 HccpPeerManager::Init，但 client 未创建成功，此处显式 DeInit 释放。
        if (!persistentStarted_) {
            EXCEPTION_CATCH(
                HccpPeerManager::GetInstance().DeInit(HrtGetDevice()),
                HCCL_ERROR("[RankInfoDetect::%s] HccpPeerManager DeInit exception (agent fallback)", __func__));
        }
    }
    shutdownDone_ = true;
    HCCL_INFO(
        "[RankInfoDetect::%s] rank info detect shutdown complete, shutdownDone_[%d], persistentStarted_[%d].", __func__,
        shutdownDone_.load(), static_cast<int>(persistentStarted_));
    return HCCL_SUCCESS;
}

RankInfoDetect::RankInfoDetect()
{
    userDevId_ = HrtGetDevice();
    aclError aclRet = aclrtGetLogicDevIdByUserDevId(userDevId_, &devLogicId_); // userDevId 转 logicDevId
    CHK_PRT_THROW(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR(
            "[RankInfoDetect::%s] aclrtGetLogicDevIdByUserDevId failed, userDevId[%d], ret[%d]", __func__, userDevId_,
            aclRet),
        InternalException, "get logicDevId fail");
    s32 deviceNum = HrtGetDeviceCount();
    CHK_PRT_THROW(
        userDevId_ >= deviceNum,
        HCCL_ERROR("[RankInfoDetect::%s] userDevId[%d] is invalid, deviceNum[%d].", __func__, userDevId_, deviceNum),
        InternalException, "get hostIp fail");
    // 临时处理，当前拓扑探测阶段本质上都是使用的userDevId_，当前获取到devLogicId_是为了后续整改
    // 如果使用devLogicId_会在多进程各自设置了RT_VISIBLE_DEVICES环境变量时，可能会因为当前进行无法访问某些devLogicId_代表的rank而报错
    devLogicId_ = userDevId_;

    s32 phyDevId = 0;
    aclRet = aclrtGetPhyDevIdByUserDevId(userDevId_, &phyDevId); // userDevId 转 phyDevId
    CHK_PRT_THROW(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR(
            "[RankInfoDetect::%s] aclrtGetPhyDevIdByUserDevId failed, userDevId[%d], ret[%d]", __func__, userDevId_,
            aclRet),
        InternalException, "get phyDevId fail");
    devPhyId_ = static_cast<u32>(phyDevId);

    HCCL_INFO(
        "[RankInfoDetect::%s] end, deviceNum[%d], userDevId_[%d], devLogicId_[%d], devPhyId_[%u].", __func__, deviceNum,
        userDevId_, devLogicId_, devPhyId_);
}

RankInfoDetect::~RankInfoDetect()
{
    // 析构兜底：运行期（HcclCommDestroy 后 ref=0）：Shutdown 正常执行，停线程 + 关连接 + 释放资源。
    if (!shutdownDone_) {
        DECTOR_TRY_CATCH("RankInfoDetect", Shutdown());
    }
    if (serviceThread_.joinable()) {
        HCCL_RUN_INFO(
            "[RankInfoDetect][~RankInfoDetect] detach joinable serviceThread, identifier[%s].", identifier_.c_str());
        serviceThread_.detach();
    }
}

HcclResult RankInfoDetect::SetupServer(HcclRootHandleV2& rootHandle)
{
    HCCL_DEBUG("[RankInfoDetect::%s] setup server start.", __func__);

    // host网卡使能
    TRY_CATCH_RETURN(HccpPeerManager::GetInstance().Init(devLogicId_););

    // 获取LocalHostIP
    TRY_CATCH_RETURN(hostIp_ = GetBootstrapIp(devPhyId_););
    CHK_PRT_RET(hostIp_.IsInvalid(), HCCL_ERROR("[RankInfoDetect::%s] get hostIp fail.", __func__), HCCL_E_INTERNAL);

    // 获取端口号port
    TRY_CATCH_RETURN(hostPort_ = GetHostListenPort());

    // 1. 创建serverSocket，启动监听并获取实际端口
    shared_ptr<Socket> serverSocket = nullptr;
    TRY_CATCH_RETURN(serverSocket = ServerInit(););

    // 2. 构建rootHandle
    HcclResult rootRet = GetRootHandle(rootHandle);
    if (rootRet != HCCL_SUCCESS) {
        HCCL_ERROR("[RankInfoDetect::%s] GetRootHandle failed, ret[%d].", __func__, rootRet);
        CHK_RET(ReleaseServerResource(serverSocket));
        return rootRet;
    }
    rootHandle_ = rootHandle;

    // 3. 实际端口和identifier就绪后，使用最终tag下发白名单
    HcclResult ret = HCCL_SUCCESS;
    TRY_CATCH_RETURN(ret = GetHandleAndAddHostSocketWhitelist(););
    if (ret == HCCL_E_PTR) {
        HCCL_ERROR("[RankInfoDetect::%s] get host socket handle failed, ret[%d].", __func__, ret);
        CHK_RET(ReleaseServerResource(serverSocket));
        return HcclResult::HCCL_E_PTR;
    }
    if (ret == HCCL_E_NETWORK) {
        HCCL_ERROR("[RankInfoDetect::%s] add host socket whitelist failed, ret[%d].", __func__, ret);
        CHK_RET(ReleaseServerResource(serverSocket));
        return HcclResult::HCCL_E_NETWORK;
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[RankInfoDetect::%s] configure host socket whitelist failed, ret[%d].", __func__, ret);
        CHK_RET(ReleaseServerResource(serverSocket));
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 创建 rankInfoDetectService_ 并拉起持久线程
    if (persistentStarted_) {
        HCCL_RUN_WARNING("[RankInfoDetect::%s] serviceThread_ still running, stopping before reassign.", __func__);
        CHK_RET(StopPersistentServer());
    }
    rankInfoDetectService_ = std::make_shared<RankInfoDetectService>(devPhyId_, serverSocket, identifier_, wlistInfo_);
    auto service = rankInfoDetectService_; // shared_ptr 副本，供线程持有
    u32 listenPort = serverSocket->GetListenPort();
    auto errorContext = hrtErrMGetErrorContextPub();
    // 在创建线程前同步设置 RUNNING 状态，确保 SetupServer 返回时 g_detectServerStatus_ 中已有 key
    RankInfoDetectService::SetServerStatus(listenPort, RANKINFO_DETECT_SERVER_STATUS_RUNNING);

    HcclResult threadRet = HCCL_SUCCESS;
    EXCEPTION_CATCH(
        (serviceThread_ = std::thread(
             &RankInfoDetect::SetupRankInfoDetectService, this, errorContext, serverSocket, devLogicId_, devPhyId_,
             identifier_, wlistInfo_)),
        threadRet = HCCL_E_INTERNAL);
    if (threadRet != HCCL_SUCCESS) {
        service->MarkThreadExited();
        RankInfoDetectService::SetServerStatus(listenPort, RANKINFO_DETECT_SERVER_STATUS_ERROR);
        HCCL_ERROR("[RankInfoDetect::%s] create persistent service thread failed.", __func__);
        return threadRet;
    }
    // serviceThread_ 线程已成功创建，需要上述步骤完成再设置 persistentStarted_，避免 StopPersistentServer 误判。
    persistentStarted_ = true;

    HCCL_DEBUG("[RankInfoDetect::%s] setup server end.", __func__);
    return HcclResult::HCCL_SUCCESS;
}

SocketHandle RankInfoDetect::GetHostSocketHandle()
{
    HCCL_DEBUG("[RankInfoDetect::%s] server get host socket handle start.", __func__);

    // 获取socket句柄
    SocketHandle hostSocketHandle = HostSocketHandleManager::GetInstance().Create(devPhyId_, hostIp_);

    // 白名单tag依赖实际监听端口和identifier，在GetRootHandle后再构造并下发
    hostSocketWlist_.clear();
    if (!EnvConfig::GetInstance().GetHostNicConfig().GetWhitelistDisable()) {
        Whitelist::GetInstance().GetHostWhiteList(hostSocketWlist_);
        CHK_PRT_THROW(
            hostSocketWlist_.empty(), HCCL_ERROR("[%s] whitelist file have no valid host ip.", __func__),
            InternalException, "get host ip error");
    }

    HCCL_DEBUG("[RankInfoDetect::%s] get host socket handle success, socketHandle[%p].", __func__, hostSocketHandle);
    return hostSocketHandle;
}

shared_ptr<Socket> RankInfoDetect::ServerInit()
{
    HCCL_DEBUG("[RankInfoDetect::%s] server init start.", __func__);

    bool whiteListDisable = EnvConfig::GetInstance().GetHostNicConfig().GetWhitelistDisable();
    SocketHandle hccpHostSocketHandle = GetHostSocketHandle();
    std::shared_ptr<Socket> serverSocket = std::make_shared<Socket>(
        hccpHostSocketHandle, hostIp_, hostPort_, hostIp_, "server", SocketRole::SERVER, NicType::HOST_NIC_TYPE,
        whiteListDisable);
    if (hostPort_ == HCCL_INVALID_PORT) {
        auto portRange = EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange();
        if (portRange.empty()) {
            SocketPortRange defaultRange = {HOST_CONTROL_BASE_PORT, HOST_CONTROL_BASE_PORT + HOST_CONTROL_PORT_COUNT};
            portRange.push_back(defaultRange);
        } else {
            // 控制面使用后半端口范围
            std::vector<SocketPortRange> frontHalf;
            std::vector<SocketPortRange> backHalf;
            SplitPortRange(portRange, frontHalf, backHalf);
            portRange = backHalf;
        }
        PreemptPortManager::GetInstance(devLogicId_).ListenPreempt(serverSocket, portRange, hostPort_);
    } else {
        serverSocket->Listen();
    }

    HCCL_INFO("[RankInfoDetect::%s] serverSocket[%s] listen success.", __func__, serverSocket->Describe().c_str());
    return serverSocket;
}

HcclResult RankInfoDetect::GetHandleAndAddHostSocketWhitelist()
{
    // 同一对象重试SetupServer时，先清理上一次生成的条目，避免旧tag被重复下发
    wlistInfo_.clear();
    if (hostSocketWlist_.empty()) {
        HCCL_INFO("[RankInfoDetect::%s] hostSocketWlist_ is empty, skip whitelist configuration.", __func__);
        return HCCL_SUCCESS;
    }

    SocketHandle hostSocketHandle = HostSocketHandleManager::GetInstance().Get(devPhyId_, hostIp_);
    CHK_PRT_RET(
        hostSocketHandle == nullptr,
        HCCL_ERROR(
            "[RankInfoDetect::%s] get host socket handle failed before adding whitelist, "
            "devPhyId[%u], hostIp[%s], identifier[%s], listenPort[%u].",
            __func__, devPhyId_, hostIp_.GetIpStr().c_str(), identifier_.c_str(), hostPort_),
        HCCL_E_PTR);
    CHK_RET(AddHostSocketWhitelist(hostSocketHandle, hostSocketWlist_));
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetect::AddHostSocketWhitelist(SocketHandle& socketHandle, const vector<IpAddress>& hostSocketWlist)
{
    HCCL_DEBUG("[RankInfoDetect::%s] start, hostSocketWlist size[%zu].", __func__, hostSocketWlist.size());

    for (auto& ipAddress : hostSocketWlist) {
        RaSocketWhitelist info{};
        info.remoteIp = ipAddress;
        info.connLimit = HOST_SOCKET_CONN_LIMIT;
        info.tag = RANK_INFO_DETECT_TAG + "_" + identifier_ + "_" + std::to_string(hostPort_);
        wlistInfo_.push_back(info);
    }

    TRY_CATCH_RETURN(HrtRaSocketWhiteListAdd(socketHandle, wlistInfo_));

    HCCL_DEBUG("[RankInfoDetect::%s] end, add wlistInfo size[%zu] success.", __func__, wlistInfo_.size());
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetect::ReleaseServerResource(shared_ptr<Socket>& serverSocket)
{
    if (serverSocket == nullptr) {
        HCCL_INFO("[RankInfoDetect::%s] serverSocket is null, skip release.", __func__);
        return HCCL_SUCCESS;
    }
    // 删除已下发的白名单条目（GetHandleAndAddHostSocketWhitelist 可能已执行）
    if (!wlistInfo_.empty()) {
        SocketHandle handle = HostSocketHandleManager::GetInstance().Get(devPhyId_, hostIp_);
        if (handle != nullptr) {
            HrtRaSocketWhiteListDel(handle, wlistInfo_);
        }
        wlistInfo_.clear();
    }
    // 释放已抢占的端口
    s32 devLogicId = HrtGetDevice();
    if (EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange().size() > 0
        || EnvConfig::GetInstance().GetHostNicConfig().GetIfBasePort() == HCCL_INVALID_PORT) {
        PreemptPortManager::GetInstance(devLogicId).Release(serverSocket);
    } else {
        EXCEPTION_CATCH(serverSocket->StopListen(),
                        HCCL_ERROR("[RankInfoDetect::%s] exception when releasing server socket", __func__);
                        return HCCL_E_INTERNAL);
    }
    serverSocket = nullptr;
    HCCL_INFO("[RankInfoDetect::%s] released server socket and port.", __func__);
    return HCCL_SUCCESS;
}

std::shared_ptr<Socket> RankInfoDetect::ClientInit(const HcclRootHandleV2& rootHandle)
{
    HCCL_DEBUG("[RankInfoDetect::%s] client init start devPhyId_[%u].", __func__, devPhyId_);

    // 获取socket句柄
    SocketHandle hostSocketHandle = HostSocketHandleManager::GetInstance().Create(devPhyId_, hostIp_);

    // 获取server端ip和port
    IpAddress serverIp = IpAddress(std::string(rootHandle.ip));
    u32 serverPort = rootHandle.listenPort;

    // 创建clientSocket
    bool whiteListDisable = EnvConfig::GetInstance().GetHostNicConfig().GetWhitelistDisable();
    std::string tag = RANK_INFO_DETECT_TAG + "_" + rootHandle.identifier + "_" + std::to_string(serverPort);
    std::shared_ptr<Socket> clientSocket = std::make_shared<Socket>(
        hostSocketHandle, hostIp_, serverPort, serverIp, tag, SocketRole::CLIENT, NicType::HOST_NIC_TYPE,
        whiteListDisable);

    HCCL_INFO("[RankInfoDetect::%s] clientSocket[%s] init end.", __func__, clientSocket->Describe().c_str());
    return clientSocket;
}

HcclResult RankInfoDetect::SetupAgent(u32 rankSize, u32 rankId, const HcclRootHandleV2& rootHandle)
{
    HCCL_DEBUG("[RankInfoDetect::%s] setup agent start.", __func__);
    rankId_ = rankId;
    rootHandle_ = rootHandle;

    // 网卡使能
    TRY_CATCH_RETURN(HccpPeerManager::GetInstance().Init(devLogicId_););
    TRY_CATCH_RETURN(HccpHdcManager::GetInstance().Init(devLogicId_););

    // 获取LocalHostIP
    TRY_CATCH_RETURN(hostIp_ = GetBootstrapIp(devPhyId_););
    CHK_PRT_RET(hostIp_.IsInvalid(), HCCL_ERROR("[RankInfoDetect::%s] get hostIp fail.", __func__), HCCL_E_INTERNAL);

    // 创建clientSocket
    std::shared_ptr<Socket> clientSocket;
    TRY_CATCH_RETURN(clientSocket = ClientInit(rootHandle););

    // 1. 创建RankInfoDetectClient对象
    TRY_CATCH_RETURN(rankInfoDetectClient
                     = std::make_shared<RankInfoDetectClient>(devPhyId_, rankSize, rankId, clientSocket););

    // 2. 调用RankInfoDetectClient.Setup, 获取rankTable
    CHK_RET(rankInfoDetectClient->Setup(rankTable_));

    HCCL_DEBUG("[RankInfoDetect::%s] setup agent end.", __func__);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetect::SetupRankInfoDetectService(
    ErrContextPub errorContext, shared_ptr<Socket> serverSocket, s32 devLogicId, u32 devPhyId, std::string identifier,
    [[maybe_unused]] vector<RaSocketWhitelist> wlistInfo)
{
    hrtErrMSetErrorContextPub(errorContext);
    HCCL_INFO(
        "[RankInfoDetect::%s] start, devLogicId[%d], devPhyId[%u], identifier[%s].", __func__, devLogicId, devPhyId,
        identifier.c_str());

    u32 hostPort = serverSocket->GetListenPort();
    HCCL_INFO("[RankInfoDetect::%s] listen port[%u].", __func__, hostPort);

    if (rankInfoDetectService_ == nullptr) {
        HCCL_ERROR("[RankInfoDetect::%s] rankInfoDetectService_ is null, SetupServer not called.", __func__);
        // 失败路径同样需要把状态从 RUNNING 改为 ERROR，避免残留 RUNNING 导致 WaitForCollectComplete 永久阻塞
        RankInfoDetectService::SetServerStatus(hostPort, RANKINFO_DETECT_SERVER_STATUS_ERROR);
        return HCCL_E_INTERNAL;
    }

    bool hasException = false;
    HcclResult setupRet = HCCL_SUCCESS;
    EXCEPTION_CATCH(HrtSetDevice(devLogicId), hasException = true);
    if (!hasException) {
        // Setup 只负责收集；完成状态由线程包装层统一发布。
        EXCEPTION_CATCH(setupRet = rankInfoDetectService_->Setup(), setupRet = HCCL_E_INTERNAL);
        if (setupRet != HCCL_SUCCESS) {
            HCCL_ERROR("[RankInfoDetect] persistent service setup failed, ret[%d].", setupRet);
            hasException = true;
        }

        HcclResult resetRet = HCCL_SUCCESS;
        EXCEPTION_CATCH(HrtResetDevice(devLogicId), resetRet = HCCL_E_INTERNAL);
        if (setupRet == HCCL_SUCCESS) {
            setupRet = resetRet;
        }
    }

    if (hasException && setupRet == HCCL_SUCCESS) {
        setupRet = HCCL_E_INTERNAL;
    }
    rankInfoDetectService_->MarkThreadExited();
    if (setupRet == HCCL_SUCCESS) {
        RankInfoDetectService::SetServerStatus(hostPort, RANKINFO_DETECT_SERVER_STATUS_IDLE);
    } else {
        RankInfoDetectService::SetServerStatus(hostPort, RANKINFO_DETECT_SERVER_STATUS_ERROR);
        HCCL_ERROR("[RankInfoDetect::%s] end, status error.", __func__);
    }

    HCCL_DEBUG("[RankInfoDetect::%s] end, ret[%d].", __func__, setupRet);
    return setupRet;
}

u32 RankInfoDetect::GetHostListenPort()
{
    // 端口监听范围配置
    u32 listenPort = HCCL_INVALID_PORT;
    auto portRange = EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange();
    if (portRange.size() > 0) {
        HCCL_INFO("[RankInfoDetect::%s] SocketPortRange is configured.", __func__);
        return listenPort;
    }

    // Host网卡起始端口号
    u32 basePort = EnvConfig::GetInstance().GetHostNicConfig().GetIfBasePort();
    if (basePort != HCCL_INVALID_PORT) {
        listenPort = basePort + devPhyId_;
        HCCL_INFO("[RankInfoDetect::%s] BasePort is configured, listenPort[%u].", __func__, listenPort);
        return listenPort;
    }

    // 无环境变量设置，返回HCCL_INVALID_PORT触发PreemptPortManager轮询查找端口[60016, 60031]
    listenPort = HCCL_INVALID_PORT;
    HCCL_INFO(
        "[RankInfoDetect::%s] No port configuration, using default port range[%u, %u]", __func__,
        HOST_CONTROL_BASE_PORT, HOST_CONTROL_BASE_PORT + HOST_CONTROL_PORT_COUNT);
    return listenPort;
}

HcclResult RankInfoDetect::SetDerivationIdentifier(const std::string& topParentId, uint32_t generationIndex)
{
    derivationIdentifierOverride_ = topParentId + "_" + std::to_string(generationIndex + 1);
    HCCL_INFO("[RankInfoDetect::%s] set derivation identifier[%s].", __func__, derivationIdentifierOverride_.c_str());
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetect::GetRootHandle(HcclRootHandleV2& rootHandle)
{
    u64 timestamp = 0;
    HcclResult ret = SalGetCurrentTimestamp(timestamp);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[RankInfoDetect::%s] get timestamp failed, ret[%d].", __func__, static_cast<int>(ret)),
        HCCL_E_INTERNAL);
    if (!derivationIdentifierOverride_.empty()) {
        identifier_ = derivationIdentifierOverride_;
    } else {
        identifier_ = hostIp_.GetIpStr();
        identifier_.append("_");
        identifier_.append(to_string(hostPort_));
        identifier_.append("_");
        identifier_.append(to_string(devPhyId_));
        identifier_.append("_");
        identifier_.append(to_string(timestamp));
    }
    CHK_PRT_RET(
        (identifier_.length() >= ROOTINFO_IDENTIFIER_MAX_LENGTH),
        HCCL_ERROR("[RankInfoDetect::%s] rootInfo identifier len[%u] is invalid.", __func__, identifier_.length()),
        HCCL_E_INTERNAL);

    s32 sRet = memcpy_s(
        &rootHandle.identifier[0], sizeof(rootHandle.identifier), identifier_.c_str(), (identifier_.length() + 1));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[RankInfoDetect::%s] memcpy failed. ret[%d], params: destMaxSize[%zu], count[%zu]", __func__, sRet,
            sizeof(rootHandle.identifier), (identifier_.length() + 1)),
        HCCL_E_INTERNAL);

    sRet = strncpy_s(
        rootHandle.ip, sizeof(rootHandle.ip), hostIp_.GetIpStr().c_str(), strlen(hostIp_.GetIpStr().c_str()));
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[RankInfoDetect::%s] strncpy failed [%d]", __func__, sRet), HCCL_E_INTERNAL);

    rootHandle.listenPort = hostPort_;
    rootHandle.netMode = HrtNetworkMode::HDC;

    HCCL_INFO(
        "[RankInfoDetect::%s] rootInfo: ip[%s] port[%u] identifier[%s]", __func__, rootHandle.ip, rootHandle.listenPort,
        identifier_.c_str());
    return HCCL_SUCCESS;
}

void RankInfoDetect::GetRankTable(RankTableInfo& ranktable) const { ranktable = rankTable_; }

HcclResult RankInfoDetect::WaitComplete(u32 listenPort, u32 listenStatus) const
{
    const auto start = chrono::steady_clock::now();
    std::chrono::seconds timeout(0);
    TRY_CATCH_RETURN(timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut()));

    u32 status = RANKINFO_DETECT_SERVER_STATUS_RUNNING;
    while (true) {
        if (!RankInfoDetectService::TryGetServerStatus(listenPort, status)) {
            // 非 root 直接返回 SUCCESS。
            HCCL_INFO(
                "[RankInfoDetect::%s] detect server listenPort[%u] not found (non-root no-op).", __func__, listenPort);
            return HcclResult::HCCL_SUCCESS;
        }
        if (status == RANKINFO_DETECT_SERVER_STATUS_ERROR) {
            HCCL_ERROR("[RankInfoDetect::%s] topo detect failed, port[%u].", __func__, listenPort);
            return HcclResult::HCCL_E_INTERNAL;
        } else if (status == listenStatus) {
            HCCL_INFO("[RankInfoDetect::%s] topoExchangeServer port[%u] completed.", __func__, listenPort);
            return HcclResult::HCCL_SUCCESS;
        } else {
            const auto elapsed = chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - start);
            if (elapsed > timeout) {
                RPT_INPUT_ERR(
                    true, "EI0015", std::vector<std::string>({"error_reason"}),
                    std::vector<std::string>({StringFormat(
                        "Receiving message from the root node timed out "
                        "after %lld seconds. Timeout was set to %lld seconds. Check whether node %s reports an error",
                        static_cast<long long>(elapsed.count()), static_cast<long long>(timeout.count()),
                        identifier_.c_str())}));
                HCCL_ERROR(
                    "[%s][%s] errNo[0x%016llx] topo exchange server get socket timeout, listenPort[%u], "
                    "elapsed[%lld s] > timeout[%lld s], identifier[%s]",
                    LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(),
                    HCOM_ERROR_CODE(HcclResult::HCCL_E_TIMEOUT), listenPort, static_cast<long long>(elapsed.count()),
                    static_cast<long long>(timeout.count()), identifier_.c_str());
                return HcclResult::HCCL_E_TIMEOUT;
            }
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
            continue;
        }
    };

    return HCCL_SUCCESS;
}

} // namespace Hccl
