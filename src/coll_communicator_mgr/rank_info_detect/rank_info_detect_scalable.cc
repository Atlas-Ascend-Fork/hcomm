/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_info_detect_scalable.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#include "binary_stream.h"
#include "bootstrap_ip.h"
#include "common/hccl_log_keywords_pub.h"
#include "env_config/env_config_v2.h"
#include "hccp_peer_manager.h"
#include "host_socket_handle_manager.h"
#include "internal_exception.h"
#include "log.h"
#include "orion_adapter_rts.h"
#include "preempt_port_manager_v2.h"
#include "sal.h"
#include "securec.h"
#include "socket_agent.h"
#include "whitelist.h"

namespace Hccl {

namespace {
    const std::string MESH_TAG_PREFIX = "rank_info_detect_scalable_mesh_tag"; // mesh 白名单 tag 前缀
    constexpr u32 MESH_SOCKET_CONN_LIMIT = 1;                                 // 每对 root 仅一条 mesh 连接

    // mesh tag 构造（connect/accept、白名单下发共用）：编号归一化后拼接，双侧独立计算一致且每对唯一。
    std::string BuildMeshTag(u32 smallerRootIndex, u32 largerRootIndex, u32 serverMeshPort)
    {
        return MESH_TAG_PREFIX + "_" + std::to_string(std::min(smallerRootIndex, largerRootIndex)) + "_"
               + std::to_string(std::max(smallerRootIndex, largerRootIndex)) + "_" + std::to_string(serverMeshPort);
    }

    // 小编号 root 主动连接大编号 root，accept 侧只接收较小编号
    std::vector<u32> MeshAcceptPeers(const ScalableServerInfo& info)
    {
        std::vector<u32> acceptPeers;
        for (u32 peer = 0; peer < info.rootIndex; ++peer) {
            acceptPeers.push_back(peer);
        }
        return acceptPeers;
    }

    // mesh 白名单条目组装（peer 列表 × host 白名单 ip，tag 按对端有序 index 构造）
    std::vector<RaSocketWhitelist>
    BuildMeshWlistEntries(const ScalableServerInfo& info, u32 listenPort, const std::vector<IpAddress>& hostSocketWlist)
    {
        std::vector<RaSocketWhitelist> wlistInfo;
        for (u32 peer : MeshAcceptPeers(info)) {
            const std::string tag
                = BuildMeshTag(std::min(peer, info.rootIndex), std::max(peer, info.rootIndex), listenPort);
            for (auto& ipAddress : hostSocketWlist) {
                RaSocketWhitelist wl{};
                wl.remoteIp = ipAddress;
                wl.connLimit = MESH_SOCKET_CONN_LIMIT;
                wl.tag = tag;
                wlistInfo.push_back(wl);
            }
        }
        return wlistInfo;
    }
} // namespace

namespace {
    // 注册表外层串行锁：内层 UniversalConcurrentMap 锁只保护单次操作。
    std::mutex g_scalableDetectRegistryMutex;
} // namespace

std::mutex& GetScalableDetectRegistryMutex() { return g_scalableDetectRegistryMutex; }

void RankInfoDetectScalable::EraseFromRegistry()
{
    std::lock_guard<std::mutex> lock(GetScalableDetectRegistryMutex());
    GetScalableDetectRegistry().Erase(rootHandle_.identifier);
}

UniversalConcurrentMap<std::string, std::shared_ptr<RankInfoDetectScalable>>& GetScalableDetectRegistry()
{
    static UniversalConcurrentMap<std::string, std::shared_ptr<RankInfoDetectScalable>> registry;
    return registry;
}

RankInfoDetectScalable::RankInfoDetectScalable(s32 devLogicId, s32 userDevId, u32 devPhyId)
    : devLogicId_(devLogicId),
      userDevId_(userDevId),
      devPhyId_(devPhyId)
{
    s32 deviceNum = HrtGetDeviceCount();
    CHK_PRT_THROW(
        userDevId_ >= deviceNum,
        HCCL_ERROR(
            "[RankInfoDetectScalable::%s] userDevId[%d] is invalid, deviceNum[%d].", __func__, userDevId_, deviceNum),
        InternalException, "userDevId is invalid");
    // 临时处理，当前拓扑探测阶段本质上都是使用的userDevId_，当前获取到devLogicId_是为了后续整改
    // 如果使用devLogicId_会在多进程各自设置了RT_VISIBLE_DEVICES环境变量时，可能会因为当前进行无法访问某些devLogicId_代表的rank而报错
    devLogicId_ = userDevId_;
}

RankInfoDetectScalable::~RankInfoDetectScalable() = default;

namespace {
    // 控制面端口范围整备：无配置时给默认 [60016, 60031]，有配置取后半段
    void PrepareControlPortRange(std::vector<SocketPortRange>& portRange)
    {
        if (portRange.empty()) {
            SocketPortRange defaultRange = {HOST_CONTROL_BASE_PORT, HOST_CONTROL_BASE_PORT + HOST_CONTROL_PORT_COUNT};
            portRange.push_back(defaultRange);
            return;
        }
        std::vector<SocketPortRange> frontHalf;
        std::vector<SocketPortRange> backHalf;
        SplitPortRange(portRange, frontHalf, backHalf);
        portRange = backHalf;
    }

    // HCCL_IF_BASE_PORT 固定端口模式下，mesh 监听端口相对 server 监听端口的偏移，
    // 对齐 A3 TOPO_MESH_PORT_OFFSET 取值 32：offset 须大于单机最大 devPhyId，
    // 使 mesh 端口区间整体错开 server 区间，避免同机多卡跨进程撞端口。
    constexpr u32 MESH_PORT_OFFSET = 32;

    u32 DecideListenPort(u32 devPhyId, u32 portOffset)
    {
        auto portRange = EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange();
        if (portRange.size() > 0) {
            HCCL_INFO("[%s] SocketPortRange is configured.", __func__);
            return HCCL_INVALID_PORT;
        }
        u32 basePort = EnvConfig::GetInstance().GetHostNicConfig().GetIfBasePort();
        if (basePort != HCCL_INVALID_PORT) {
            u32 listenPort = basePort + devPhyId + portOffset;
            HCCL_INFO("[%s] BasePort is configured, listenPort[%u].", __func__, listenPort);
            return listenPort;
        }
        HCCL_INFO(
            "[%s] No port configuration, using default port range[%u, %u]", __func__, HOST_CONTROL_BASE_PORT,
            HOST_CONTROL_BASE_PORT + HOST_CONTROL_PORT_COUNT);
        return HCCL_INVALID_PORT;
    }

    // 从区间中剔除 excludePort
    void ExcludePortFromRange(std::vector<SocketPortRange>& portRange, u32 excludePort)
    {
        if (excludePort == HCCL_INVALID_PORT) {
            return;
        }
        std::vector<SocketPortRange> adjusted;
        for (const auto& range : portRange) {
            if (excludePort < range.min || excludePort > range.max) {
                adjusted.emplace_back(range);
                continue;
            }
            if (excludePort > range.min) {
                adjusted.emplace_back(SocketPortRange{range.min, excludePort - 1});
            }
            if (excludePort < range.max) {
                adjusted.emplace_back(SocketPortRange{excludePort + 1, range.max});
            }
        }
        portRange = std::move(adjusted);
    }

    // 固定端口直接 Listen；否则按控制面 range 抢占。
    HcclResult ListenOrPreempt(
        s32 devLogicId, const std::shared_ptr<Socket>& listenSocket, u32& port, u32 excludePort = HCCL_INVALID_PORT)
    {
        if (port != HCCL_INVALID_PORT) {
            HCCL_INFO("[%s] socket listen on port[%u]", __func__, port);
            TRY_CATCH_RETURN(listenSocket->Listen(););
            return HCCL_SUCCESS;
        }
        auto portRange = EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange();
        PrepareControlPortRange(portRange);
        ExcludePortFromRange(portRange, excludePort);
        PreemptPortManager::GetInstance(devLogicId).ListenPreempt(listenSocket, portRange, port);
        return HCCL_SUCCESS;
    }
} // namespace

// 监听 socket 失败自清
void RankInfoDetectScalable::ReleaseListenSocketResources(
    const std::shared_ptr<Socket>& listenSocket, const IpAddress& hostIp, u32 portOffset) const
{
    if (DecideListenPort(devPhyId_, portOffset) == HCCL_INVALID_PORT) {
        TRY_CATCH_PRINT_ERROR(PreemptPortManager::GetInstance(devLogicId_).Release(listenSocket););
    } else {
        TRY_CATCH_PRINT_ERROR(listenSocket->StopListen(););
    }
    TRY_CATCH_PRINT_ERROR(HostSocketHandleManager::GetInstance().Destroy(devPhyId_, hostIp););
}

HcclResult RankInfoDetectScalable::SetupMeshListen(u32& meshPort, u32 serverPort)
{
    IpAddress hostIp = IpAddress(std::string(rootHandle_.ip));

    // 获取端口号
    // mesh 端口与 server 同款决策：固定端口模式下 mesh = server + MESH_PORT_OFFSET；
    // 抢占模式下从区间剔除 serverPort 后轮询选新端口（防 PreemptPortManager 复用段撞端口）。
    meshPort = DecideListenPort(devPhyId_, MESH_PORT_OFFSET);

    // 创建socket，启动监听
    bool whiteListDisable = EnvConfig::GetInstance().GetHostNicConfig().GetWhitelistDisable();
    SocketHandle hostSocketHandle = HostSocketHandleManager::GetInstance().Create(devPhyId_, hostIp);
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        hostSocketHandle, hostIp, meshPort, hostIp, MESH_TAG_PREFIX + "_" + std::to_string(devPhyId_),
        SocketRole::SERVER, NicType::HOST_NIC_TYPE, whiteListDisable);

    HcclResult listenRet = ListenOrPreempt(devLogicId_, listenSocket, meshPort, serverPort);
    if (listenRet != HCCL_SUCCESS) {
        ReleaseListenSocketResources(listenSocket, hostIp, MESH_PORT_OFFSET);
        return listenRet;
    }

    if (listenSocket->GetListenPort() == 0 || meshPort == HCCL_INVALID_PORT) {
        HCCL_ERROR("[RankInfoDetectScalable::%s] mesh listen port setup failed, port[%u].", __func__, meshPort);
        ReleaseListenSocketResources(listenSocket, hostIp, MESH_PORT_OFFSET);
        return HCCL_E_INTERNAL;
    }
    meshListenSocket_ = listenSocket;
    HCCL_INFO(
        "[RankInfoDetectScalable::%s] mesh listen ready, port[%u], identifier[%s].", __func__, meshPort,
        rootHandle_.identifier);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectScalable::SetupServerListen(IpAddress& hostIp, u32& hostPort)
{
    // 获取hostIP
    TRY_CATCH_RETURN(hostIp = GetBootstrapIp(devPhyId_););
    CHK_PRT_RET(
        hostIp.IsInvalid(), HCCL_ERROR("[RankInfoDetectScalable::%s] get hostIp fail.", __func__), HCCL_E_INTERNAL);

    // 获取端口号
    hostPort = DecideListenPort(devPhyId_, 0);

    // 创建socket，启动监听
    bool whiteListDisable = EnvConfig::GetInstance().GetHostNicConfig().GetWhitelistDisable();
    SocketHandle hostSocketHandle = HostSocketHandleManager::GetInstance().Create(devPhyId_, hostIp);
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        hostSocketHandle, hostIp, hostPort, hostIp, "server", SocketRole::SERVER, NicType::HOST_NIC_TYPE,
        whiteListDisable);

    HcclResult listenRet = ListenOrPreempt(devLogicId_, listenSocket, hostPort);
    if (listenRet != HCCL_SUCCESS) {
        ReleaseListenSocketResources(listenSocket, hostIp, 0);
        return listenRet;
    }

    if (listenSocket->GetListenPort() == 0 || hostPort == HCCL_INVALID_PORT) {
        HCCL_ERROR("[RankInfoDetectScalable::%s] server listen port setup failed, port[%u].", __func__, hostPort);
        ReleaseListenSocketResources(listenSocket, hostIp, 0);
        return HCCL_E_INTERNAL;
    }
    serverListenSocket_ = listenSocket;
    serverHostSocketHandle_ = hostSocketHandle;
    HCCL_INFO("[RankInfoDetectScalable::%s] server socket listen ready, port[%u].", __func__, hostPort);
    return HCCL_SUCCESS;
}

// rootHandle_ 的 identifier 生成与填充
HcclResult RankInfoDetectScalable::BuildRootHandle(const IpAddress& hostIp, u32 hostPort)
{
    u64 timestamp = 0;
    CHK_PRT_RET(
        SalGetCurrentTimestamp(timestamp) != HCCL_SUCCESS,
        HCCL_ERROR("[RankInfoDetectScalable::%s] get timestamp failed.", __func__), HCCL_E_INTERNAL);
    std::string identifier = "scalable_" + hostIp.GetIpStr() + "_" + std::to_string(hostPort) + "_"
                             + std::to_string(devPhyId_) + "_" + std::to_string(timestamp);
    CHK_PRT_RET(
        identifier.length() >= ROOTINFO_IDENTIFIER_MAX_LENGTH,
        HCCL_ERROR("[RankInfoDetectScalable::%s] identifier len[%zu] is invalid.", __func__, identifier.length()),
        HCCL_E_INTERNAL);
    s32 sRet = strncpy_s(
        rootHandle_.ip, sizeof(rootHandle_.ip), hostIp.GetIpStr().c_str(), strlen(hostIp.GetIpStr().c_str()));
    CHK_PRT_RET(
        sRet != EOK, HCCL_ERROR("[RankInfoDetectScalable::%s] strncpy ip failed [%d].", __func__, sRet),
        HCCL_E_INTERNAL);
    sRet = memcpy_s(
        &rootHandle_.identifier[0], sizeof(rootHandle_.identifier), identifier.c_str(), identifier.length() + 1);
    CHK_PRT_RET(
        sRet != EOK, HCCL_ERROR("[RankInfoDetectScalable::%s] memcpy identifier failed [%d].", __func__, sRet),
        HCCL_E_INTERNAL);
    rootHandle_.listenPort = hostPort;
    rootHandle_.netMode = HrtNetworkMode::HDC;
    HCCL_INFO(
        "[RankInfoDetectScalable::%s] rootInfo: ip[%s] port[%u] identifier[%s]", __func__, rootHandle_.ip,
        rootHandle_.listenPort, identifier.c_str());
    return HCCL_SUCCESS;
}

// server 服务白名单下发（tag = RANK_INFO_DETECT_TAG_{identifier}_{port}）
HcclResult RankInfoDetectScalable::AddServerSocketWhiteList()
{
    serverWlistInfo_.clear();
    if (EnvConfig::GetInstance().GetHostNicConfig().GetWhitelistDisable()) {
        return HCCL_SUCCESS;
    }
    if (serverHostSocketHandle_ == nullptr) {
        HCCL_ERROR(
            "[RankInfoDetectScalable::%s] host socket handle is null before adding whitelist, "
            "devPhyId[%u], identifier[%s], listenPort[%u].",
            __func__, devPhyId_, rootHandle_.identifier, rootHandle_.listenPort);
        return HCCL_E_PTR;
    }
    std::vector<IpAddress> hostSocketWlist;
    Whitelist::GetInstance().GetHostWhiteList(hostSocketWlist);
    if (hostSocketWlist.empty()) {
        HCCL_ERROR("[%s] whitelist file have no valid host ip.", __func__);
        return HCCL_E_INTERNAL;
    }

    const std::string tag = RANK_INFO_DETECT_TAG + "_" + std::string(rootHandle_.identifier) + "_"
                            + std::to_string(rootHandle_.listenPort);
    for (auto& ipAddress : hostSocketWlist) {
        RaSocketWhitelist info{};
        info.remoteIp = ipAddress;
        info.connLimit = HOST_SOCKET_CONN_LIMIT;
        info.tag = tag;
        serverWlistInfo_.push_back(info);
    }

    TRY_CATCH_RETURN(HrtRaSocketWhiteListAdd(serverHostSocketHandle_, serverWlistInfo_));
    HCCL_DEBUG("[RankInfoDetectScalable::%s] end, add wlistInfo size[%zu] success.", __func__, serverWlistInfo_.size());
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectScalable::SetupScalableRoot(HcclScalableRootHandleV2& outHandle)
{
    HCCL_DEBUG("[RankInfoDetectScalable::%s] start.", __func__);
    // host网卡使能
    TRY_CATCH_RETURN(HccpPeerManager::GetInstance().Init(devLogicId_););
    IpAddress hostIp{};
    u32 hostPort = HCCL_INVALID_PORT;
    // 1. server 监听
    HcclResult ret = SetupServerListen(hostIp, hostPort);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[RankInfoDetectScalable::%s] SetupServerListen failed, ret[%d].", __func__, static_cast<int>(ret));
        CleanupOnSetupFailure();
        return ret;
    }
    // 2. 组rootHandle
    ret = BuildRootHandle(hostIp, hostPort);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[RankInfoDetectScalable::%s] BuildRootHandle failed, ret[%d].", __func__, static_cast<int>(ret));
        CleanupOnSetupFailure();
        return ret;
    }
    // 3. 白名单下发
    ret = AddServerSocketWhiteList();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[RankInfoDetectScalable::%s] add whitelist failed, ret[%d].", __func__, static_cast<int>(ret));
        CleanupOnSetupFailure();
        return ret;
    }

    // 4. mesh 监听
    u32 meshPort = HCCL_INVALID_PORT;
    ret = SetupMeshListen(meshPort, hostPort);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[RankInfoDetectScalable::%s] SetupMeshListen failed, ret[%d].", __func__, static_cast<int>(ret));
        CleanupOnSetupFailure();
        return ret;
    }

    outHandle.rootHandle = rootHandle_;
    outHandle.meshPort = meshPort;

    // 5. 注册表保活：server 线程等待注入期间依赖对象成员，对象须跨越 Get→Init 空窗存活。注册表访问统一持外层锁
    std::shared_ptr<RankInfoDetectScalable> self = shared_from_this();
    {
        std::lock_guard<std::mutex> lock(GetScalableDetectRegistryMutex());
        GetScalableDetectRegistry().EmplaceAndUpdate(
            rootHandle_.identifier, [self](std::shared_ptr<RankInfoDetectScalable>& slot) {
                slot = self;
            });
    }

    // 6. 拉起 scalable server 线程
    bool isThreadsReady = true;
    EXCEPTION_CATCH(
        {
            std::thread threadHandle([self, errorContext = hrtErrMGetErrorContextPub()] {
                const u32 serverPort = self->rootHandle_.listenPort;
                bool workerFailed = false;
                EXCEPTION_CATCH(self->ScalableServerWorker(errorContext), workerFailed = true);
                if (workerFailed) {
                    self->HandleWorkerException(serverPort);
                }
                self->EraseFromRegistry();
            });
            threadHandle.detach();
        },
        isThreadsReady = false);

    if (!isThreadsReady) {
        HCCL_ERROR("[RankInfoDetectScalable::%s] create server thread failed.", __func__);
        EraseFromRegistry();
        CleanupOnSetupFailure();
        return HCCL_E_MEMORY;
    }
    HCCL_INFO(
        "[RankInfoDetectScalable::%s] end, serverPort[%u], meshPort[%u], identifier[%s].", __func__, hostPort, meshPort,
        rootHandle_.identifier);
    return HCCL_SUCCESS;
}

void RankInfoDetectScalable::ScalableServerWorker(ErrContextPub errorContext)
{
    hrtErrMSetErrorContextPub(errorContext);
    HCCL_INFO(
        "[RankInfoDetectScalable::%s] start, identifier[%s], deviceLogicId[%d].", __func__, rootHandle_.identifier,
        devLogicId_);
    const u32 serverPort = rootHandle_.listenPort;
    RankInfoDetectService::SetServerStatus(serverPort, RANKINFO_DETECT_SERVER_STATUS_RUNNING);

    bool deviceSetFailed = false;
    EXCEPTION_CATCH(HrtSetDevice(devLogicId_), deviceSetFailed = true);
    if (deviceSetFailed) {
        HCCL_ERROR("[RankInfoDetectScalable::%s] set device failed, deviceLogicId[%d].", __func__, devLogicId_);
        RankInfoDetectService::SetServerStatus(serverPort, RANKINFO_DETECT_SERVER_STATUS_ERROR);
        // 未成功 SetDevice，收尾不做 HrtResetDevice，避免 device 引用配对错误。
        EraseFromRegistry();
        ReleaseAgentResources();
        ReleaseListenResources();
        return;
    }
    deviceSet_ = true;

    // 1. 阻塞等待
    HcclResult ret = WaitScalableInfoReady();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[RankInfoDetectScalable::%s] wait scalable info failed, ret[%d].", __func__, static_cast<int>(ret));
        RankInfoDetectService::SetServerStatus(serverPort, RANKINFO_DETECT_SERVER_STATUS_ERROR);
        // 收尾第一步先摘除注册表条目，关闭注入窗口，再释放资源。
        EraseFromRegistry();
        // 超时路径无 service：先补齐 TearDown 对 server 监听承担的清理，再做统一收尾（内含 device reset）。
        ReleaseAgentResources();
        ReleaseListenResources();
        return;
    }
    const u32 groupSize = scalableInfo_.groupSize;

    // 2. server 监听 socket
    HcclResult runRet = HCCL_E_INTERNAL;
    auto service = std::make_shared<RankInfoDetectService>(
        devPhyId_, serverListenSocket_, scalableInfo_.identifier, serverWlistInfo_, scalableInfo_.groupFirstRank);
    EXCEPTION_CATCH(
        (runRet = service->SetupScalable(
             groupSize,
             [this](RankTableInfo& rankTable) {
                 HcclResult gatherRet = RootMeshAllGather(rankTable);
                 CHK_PRT_RET(
                     gatherRet != HCCL_SUCCESS,
                     HCCL_ERROR("[RankInfoDetectScalable] mesh allgather failed, ret[%d]", static_cast<int>(gatherRet)),
                     gatherRet);
                 if (scalableInfo_.nRoot <= 1) {
                     return HCCL_SUCCESS; // 单 root 本地表即全局表
                 }
                 return MergeMeshPartials(scalableInfo_.nRoot, rankTable);
             })),
        runRet = HCCL_E_INTERNAL);

    (void)SetDetectServerFinalStatus(serverPort, runRet != HCCL_SUCCESS);
    EraseFromRegistry();
    // 提前析构 service：TearDown 的 HostSocketHandleManager::Destroy 在此归还 server 监听的 handle 引用
    service.reset();
    // server 监听的清理已由 TearDown 承担，只做统一收尾
    ReleaseListenResources();

    HCCL_INFO("[RankInfoDetectScalable::ScalableServerWorker] end.");
}

void RankInfoDetectScalable::ReleaseAgentResources()
{
    if (!EnvConfig::GetInstance().GetHostNicConfig().GetWhitelistDisable() && !serverWlistInfo_.empty()) {
        TRY_CATCH_PRINT_ERROR(HrtRaSocketWhiteListDel(serverHostSocketHandle_, serverWlistInfo_););
    }
    if (serverListenSocket_ != nullptr) {
        if (DecideListenPort(devPhyId_, 0) == HCCL_INVALID_PORT) {
            // 若抢占监听端口
            TRY_CATCH_PRINT_ERROR(PreemptPortManager::GetInstance(devLogicId_).Release(serverListenSocket_););
        } else {
            // 停止监听
            TRY_CATCH_PRINT_ERROR(serverListenSocket_->StopListen(););
        }
        // server 监听的 Create 引用归还
        TRY_CATCH_PRINT_ERROR(
            HostSocketHandleManager::GetInstance().Destroy(devPhyId_, serverListenSocket_->GetLocalIp()););
    }
}

void RankInfoDetectScalable::ReleaseMeshResources()
{
    TRY_CATCH_PRINT_ERROR((void)DeleteMeshSocketWhiteList(scalableInfo_););
    if (meshListenSocket_ != nullptr) {
        IpAddress meshHostIp = meshListenSocket_->GetLocalIp();
        if (DecideListenPort(devPhyId_, MESH_PORT_OFFSET) == HCCL_INVALID_PORT) {
            // 若抢占监听端口
            TRY_CATCH_PRINT_ERROR(PreemptPortManager::GetInstance(devLogicId_).Release(meshListenSocket_););
        }
        TRY_CATCH_PRINT_ERROR(meshListenSocket_->Destroy(););
        meshListenSocket_.reset();
        // mesh 的 Create 引用归还（server 监听的已由 TearDown 或 ReleaseAgentResources 归还）。
        TRY_CATCH_PRINT_ERROR(HostSocketHandleManager::GetInstance().Destroy(devPhyId_, meshHostIp););
    }
    partials_.clear();
}

// SetupScalableRoot 失败路径回滚：调用线程为 Get 阶段主线程，不做 HrtResetDevice。
void RankInfoDetectScalable::CleanupOnSetupFailure()
{
    ReleaseAgentResources();
    ReleaseMeshResources();
    TRY_CATCH_PRINT_ERROR(HccpPeerManager::GetInstance().DeInit(devLogicId_););
}

// server 线程捕获异常收尾
void RankInfoDetectScalable::HandleWorkerException(u32 serverPort)
{
    TRY_CATCH_PRINT_ERROR((void)SetDetectServerFinalStatus(serverPort, true););
    TRY_CATCH_PRINT_ERROR(EraseFromRegistry(););
    TRY_CATCH_PRINT_ERROR(ReleaseAgentResources(););
    TRY_CATCH_PRINT_ERROR(ReleaseListenResources(););
}

// 线程出口统一收尾。调用顺序约束：本函数须在 server 监听的 handle 引用已归还之后调用。
// 仅 HrtSetDevice 成功时执行 HrtResetDevice，保证 device 引用计数配对。
void RankInfoDetectScalable::ReleaseListenResources()
{
    ReleaseMeshResources();
    if (serverListenSocket_ != nullptr) {
        TRY_CATCH_PRINT_ERROR(serverListenSocket_->Destroy(););
    }
    TRY_CATCH_PRINT_ERROR(HccpPeerManager::GetInstance().DeInit(devLogicId_););
    if (deviceSet_) {
        TRY_CATCH_PRINT_ERROR(HrtResetDevice(devLogicId_););
    }
}

HcclResult RankInfoDetectScalable::SetDetectServerFinalStatus(u32 serverPort, bool failed) const
{
    const u32 finalStatus = failed ? RANKINFO_DETECT_SERVER_STATUS_ERROR : RANKINFO_DETECT_SERVER_STATUS_IDLE;
    RankInfoDetectService::SetServerStatus(serverPort, finalStatus);
    if (failed) {
        HCCL_ERROR("[RankInfoDetectScalable::%s] end, status error.", __func__);
    } else {
        HCCL_INFO("[RankInfoDetectScalable::%s] end, status idle.", __func__);
    }
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectScalable::AddMeshSocketWhiteList(const ScalableServerInfo& info)
{
    if (meshListenSocket_ == nullptr || info.nRoot <= 1) {
        return HCCL_SUCCESS; // nRoot<=1 无 mesh 链路
    }
    std::vector<IpAddress> hostSocketWlist;
    if (!EnvConfig::GetInstance().GetHostNicConfig().GetWhitelistDisable()) {
        Whitelist::GetInstance().GetHostWhiteList(hostSocketWlist);
    }
    if (hostSocketWlist.empty()) {
        return HCCL_SUCCESS;
    }
    std::vector<RaSocketWhitelist> wlistInfo
        = BuildMeshWlistEntries(info, meshListenSocket_->GetListenPort(), hostSocketWlist);
    if (wlistInfo.empty()) {
        return HCCL_SUCCESS;
    }
    SocketHandle socketHandle = serverHostSocketHandle_;
    CHK_PRT_RET(
        socketHandle == nullptr,
        HCCL_ERROR("[RankInfoDetectScalable::%s] get host socket handle failed, devPhyId[%u].", __func__, devPhyId_),
        HCCL_E_PTR);
    TRY_CATCH_RETURN(HrtRaSocketWhiteListAdd(socketHandle, wlistInfo));
    HCCL_INFO(
        "[RankInfoDetectScalable::%s] mesh whitelist added, entries[%zu], rootIndex[%u], nRoot[%u].", __func__,
        wlistInfo.size(), info.rootIndex, info.nRoot);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectScalable::DeleteMeshSocketWhiteList(const ScalableServerInfo& info) const
{
    if (meshListenSocket_ == nullptr || info.nRoot <= 1) {
        return HCCL_SUCCESS;
    }
    std::vector<IpAddress> hostSocketWlist;
    if (!EnvConfig::GetInstance().GetHostNicConfig().GetWhitelistDisable()) {
        Whitelist::GetInstance().GetHostWhiteList(hostSocketWlist);
    }
    if (hostSocketWlist.empty()) {
        return HCCL_SUCCESS;
    }
    std::vector<RaSocketWhitelist> wlistInfo
        = BuildMeshWlistEntries(info, meshListenSocket_->GetListenPort(), hostSocketWlist);
    if (wlistInfo.empty()) {
        return HCCL_SUCCESS;
    }
    SocketHandle socketHandle = HostSocketHandleManager::GetInstance().Get(devPhyId_, IpAddress(rootHandle_.ip));
    if (socketHandle == nullptr) {
        return HCCL_SUCCESS; // 句柄已释放，无需清理
    }
    TRY_CATCH_RETURN(HrtRaSocketWhiteListDel(socketHandle, wlistInfo));
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectScalable::SetScalableServerInfo(const ScalableServerInfo& info)
{
    {
        std::unique_lock<std::mutex> lock(scalableInfoMutex_);
        // 重复注入拦截：同一对象只允许注入一次
        if (scalableInfoReady_.load()) {
            HCCL_ERROR(
                "[RankInfoDetectScalable::%s] scalable info already injected, reject duplicate, nRoot[%u], "
                "rootIndex[%u].",
                __func__, info.nRoot, info.rootIndex);
            return HCCL_E_INTERNAL;
        }
        scalableInfo_ = info;
        scalableInfoReady_.store(true);
    }
    scalableInfoReadyCv_.notify_all();
    HCCL_INFO(
        "[RankInfoDetectScalable::%s] scalable info injected, nRoot[%u], rootIndex[%u], groupSize[%u].", __func__,
        info.nRoot, info.rootIndex, info.groupSize);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectScalable::WaitScalableInfoReady()
{
    std::unique_lock<std::mutex> lock(scalableInfoMutex_);
    if (scalableInfoReadyCv_.wait_for(
            lock, std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut()), [this] {
                return scalableInfoReady_.load();
            })) {
        return HCCL_SUCCESS;
    }
    HCCL_ERROR(
        "[%s][%s] errNo[0x%016llx] wait scalable info timeout, identifier[%s]", LOG_KEYWORDS_INIT_GROUP.c_str(),
        LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_TIMEOUT), rootHandle_.identifier);
    return HCCL_E_TIMEOUT;
}

namespace {
    // mesh 交换总预算内的剩余等待秒数：不足 1s 按 1s 计
    u32 RemainingMeshWaitSeconds(const std::chrono::steady_clock::time_point& startTime, u32 timeoutSec)
    {
        const auto elapsedSec = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - startTime).count());
        if (elapsedSec >= timeoutSec) {
            return 0;
        }
        return std::max<u32>(static_cast<u32>(timeoutSec - elapsedSec), 1);
    }

    // connect socket 作用域守卫：正常/异常路径统一 Close 并归还句柄引用（Create/Destroy 按次配对）。
    class MeshConnectSocketGuard {
    public:
        MeshConnectSocketGuard(std::shared_ptr<Socket> socket, u32 devPhyId)
            : socket_(std::move(socket)),
              devPhyId_(devPhyId)
        {}
        ~MeshConnectSocketGuard()
        {
            if (socket_ == nullptr) {
                return;
            }
            const IpAddress localIp = socket_->GetLocalIp();
            TRY_CATCH_PRINT_ERROR(socket_->Close(););
            TRY_CATCH_PRINT_ERROR(HostSocketHandleManager::GetInstance().Destroy(devPhyId_, localIp););
        }
        MeshConnectSocketGuard(const MeshConnectSocketGuard&) = delete;
        MeshConnectSocketGuard& operator=(const MeshConnectSocketGuard&) = delete;
        MeshConnectSocketGuard(MeshConnectSocketGuard&&) = delete;
        MeshConnectSocketGuard& operator=(MeshConnectSocketGuard&&) = delete;

    private:
        std::shared_ptr<Socket> socket_{nullptr};
        u32 devPhyId_{0};
    };
} // namespace

// root 间 mesh 全交换：每对 root 只交换各自组内子表，收齐 nRoot-1 份对端子表后由调用方合并
HcclResult RankInfoDetectScalable::RootMeshAllGather(const RankTableInfo& localRankTable)
{
    HCCL_INFO(
        "[RankInfoDetectScalable::%s] start, nRoot[%u], rootIndex[%u].", __func__, scalableInfo_.nRoot,
        scalableInfo_.rootIndex);
    const u32 nRoot = scalableInfo_.nRoot;
    CHK_PRT_RET(
        nRoot == 0 || scalableInfo_.rootIndex >= nRoot, HCCL_ERROR("[%s] invalid root count/index", __func__),
        HCCL_E_PARA);
    if (nRoot == 1) {
        HCCL_INFO("[RankInfoDetectScalable::%s] nRoot is 1, skip mesh exchange, local table is global.", __func__);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        scalableInfo_.meshInfos.size() != nRoot, HCCL_ERROR("[%s] incomplete mesh endpoint list", __func__),
        HCCL_E_PARA);
    CHK_SMART_PTR_NULL(meshListenSocket_);
    for (const auto& info : scalableInfo_.meshInfos) {
        CHK_PRT_RET(
            info.meshPort == 0 || info.meshPort == HCCL_INVALID_PORT || IpAddress(info.ip).IsInvalid(),
            HCCL_ERROR("[%s] invalid mesh info", __func__), HCCL_E_PARA);
    }
    BinaryStream stream;
    CHK_RET(localRankTable.GetBinStream(true, stream));
    partials_.assign(nRoot, std::vector<char>{});
    stream.Dump(partials_[scalableInfo_.rootIndex]);
    meshRecvCount_.store(0);
    meshFailed_.store(false);

    // mesh 交换总预算 = link timeout；accept/connect 两线程并发等待，各自独立累计耗时。
    const u32 timeoutSec = static_cast<u32>(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());

    HcclResult acceptRet = HCCL_SUCCESS;
    std::thread acceptThread;
    if (scalableInfo_.rootIndex > 0) {
        ErrContextPub errorContext = hrtErrMGetErrorContextPub();
        EXCEPTION_CATCH(
            acceptThread = std::thread([this, &acceptRet, errorContext, timeoutSec] {
                hrtErrMSetErrorContextPub(errorContext);
                acceptRet = MeshAcceptSmallerRoots(timeoutSec);
                if (acceptRet != HCCL_SUCCESS) {
                    meshFailed_.store(true);
                }
            }),
            return HCCL_E_MEMORY);
    }
    // 线程启动后不提前返回：即使主动连接失败/抛异常，也先 join 接受线程。
    HcclResult connectRet = HCCL_E_INTERNAL;
    TRY_CATCH_PRINT_ERROR(connectRet = MeshConnectToLargerRoots(timeoutSec););
    if (connectRet != HCCL_SUCCESS) {
        meshFailed_.store(true);
    }
    if (acceptThread.joinable()) {
        acceptThread.join();
    }
    CHK_RET(acceptRet);
    CHK_RET(connectRet);
    CHK_PRT_RET(
        meshRecvCount_.load() != nRoot - 1,
        HCCL_ERROR(
            "[%s] missing mesh partials, received[%u], expected[%u]", __func__, meshRecvCount_.load(), nRoot - 1),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

// 创建 socket 连接更大编号的 root
std::shared_ptr<Socket> RankInfoDetectScalable::MakeMeshSocket(u32 peer)
{
    const u32 self = scalableInfo_.rootIndex;
    const u32 serverPort = scalableInfo_.meshInfos[peer].meshPort;
    const IpAddress localIp(rootHandle_.ip);
    const IpAddress peerIp(scalableInfo_.meshInfos[peer].ip);
    auto handle = HostSocketHandleManager::GetInstance().Create(devPhyId_, localIp);
    const bool disable = EnvConfig::GetInstance().GetHostNicConfig().GetWhitelistDisable();
    return std::make_shared<Socket>(
        handle, localIp, serverPort, peerIp, BuildMeshTag(self, peer, serverPort), SocketRole::CLIENT,
        NicType::HOST_NIC_TYPE, disable);
}

HcclResult RankInfoDetectScalable::MeshConnectToLargerRoots(u32 timeoutSec)
{
    const auto startTime = std::chrono::steady_clock::now();
    TRY_CATCH_RETURN(for (u32 peer = scalableInfo_.rootIndex + 1; peer < scalableInfo_.nRoot; ++peer) {
        if (meshFailed_.load()) {
            return HCCL_E_INTERNAL;
        }
        const u32 remainTimeSec = RemainingMeshWaitSeconds(startTime, timeoutSec);
        if (remainTimeSec == 0) {
            HCCL_ERROR(
                "[%s] mesh connect budget exhausted, peer[%u], my rootIndex[%u], nRoot[%u], timeout[%u s]", __func__,
                peer, scalableInfo_.rootIndex, scalableInfo_.nRoot, timeoutSec);
            return HCCL_E_TIMEOUT;
        }
        auto socket = MakeMeshSocket(peer);
        CHK_SMART_PTR_NULL(socket);
        // 守卫统一承担 Close 与句柄引用归还（含异常路径）
        MeshConnectSocketGuard socketGuard(socket, devPhyId_);
        socket->Connect();
        const auto status = socket->GetStatus(remainTimeSec);
        HcclResult ret = (status == SocketStatus::OK) ?
                             HCCL_SUCCESS :
                             (status == SocketStatus::TIMEOUT ? HCCL_E_TIMEOUT : HCCL_E_TCP_CONNECT);
        if (ret == HCCL_SUCCESS) {
            ret = ExchangeMeshPartial(*socket, peer, true);
        }
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] connect/exchange root[%u] failed, ret[%d]", __func__, peer, ret),
            ret);
    });
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectScalable::MeshAcceptSmallerRoots(u32 timeoutSec)
{
    TRY_CATCH_RETURN(HrtSetDevice(devLogicId_););
    HcclResult ret = HCCL_E_INTERNAL;
    TRY_CATCH_PRINT_ERROR(ret = MeshAcceptSmallerRootsCore(timeoutSec););
    TRY_CATCH_PRINT_ERROR(HrtResetDevice(devLogicId_););
    return ret;
}

HcclResult RankInfoDetectScalable::MeshAcceptSmallerRootsCore(u32 timeoutSec)
{
    const auto startTime = std::chrono::steady_clock::now();
    TRY_CATCH_RETURN(for (u32 peer = 0; peer < scalableInfo_.rootIndex; ++peer) {
        if (meshFailed_.load()) {
            return HCCL_E_INTERNAL;
        }
        const u32 remainTimeSec = RemainingMeshWaitSeconds(startTime, timeoutSec);
        if (remainTimeSec == 0) {
            HCCL_ERROR(
                "[%s] mesh accept budget exhausted, peer[%u], my rootIndex[%u], timeout[%u s]", __func__, peer,
                scalableInfo_.rootIndex, timeoutSec);
            return HCCL_E_TIMEOUT;
        }
        const auto tag = BuildMeshTag(peer, scalableInfo_.rootIndex, meshListenSocket_->GetListenPort());
        std::shared_ptr<Socket> socket;
        CHK_RET(meshListenSocket_->Accept(tag, socket, remainTimeSec));
        CHK_SMART_PTR_NULL(socket);
        CHK_RET(ExchangeMeshPartial(*socket, peer, false));
        socket->Close();
    });
    return HCCL_SUCCESS;
}

// 连接方先发后收，接受方先收后发；只交换各自子表，不转发其他 root 的子表。
HcclResult RankInfoDetectScalable::ExchangeMeshPartial(Socket& socket, u32 peer, bool sendFirst)
{
    if (sendFirst) {
        CHK_RET(SendMeshPartial(socket));
        CHK_RET(RecvMeshPartial(socket, peer));
    } else {
        CHK_RET(RecvMeshPartial(socket, peer));
        CHK_RET(SendMeshPartial(socket));
    }
    meshRecvCount_.fetch_add(1);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectScalable::SendMeshPartial(Socket& socket) const
{
    const u32 self = scalableInfo_.rootIndex;
    const auto& data = partials_[self];
    CHK_PRT_RET(
        data.empty() || data.size() > MAX_BUFFER_LEN,
        HCCL_ERROR("[%s] invalid local partial size[%zu]", __func__, data.size()), HCCL_E_PARA);
    CHK_PRT_RET(!socket.Send(&self, sizeof(self)), HCCL_ERROR("[%s] send root index failed", __func__), HCCL_E_NETWORK);
    SocketAgent agent(&socket);
    return agent.SendMsg(data.data(), data.size());
}

HcclResult RankInfoDetectScalable::RecvMeshPartial(Socket& socket, u32 peer)
{
    u32 source = SCALABLE_INVALID_INDEX;
    u64 size = 0;
    CHK_PRT_RET(
        !socket.Recv(&source, sizeof(source)), HCCL_ERROR("[%s] recv root index failed", __func__), HCCL_E_NETWORK);
    CHK_PRT_RET(
        peer >= partials_.size() || source != peer || peer == scalableInfo_.rootIndex || !partials_[peer].empty(),
        HCCL_ERROR("[%s] unexpected/duplicate root index[%u], expected[%u]", __func__, source, peer), HCCL_E_PARA);
    CHK_PRT_RET(!socket.Recv(&size, sizeof(size)), HCCL_ERROR("[%s] recv length failed", __func__), HCCL_E_NETWORK);
    CHK_PRT_RET(
        size == 0 || size > MAX_BUFFER_LEN,
        HCCL_ERROR("[%s] invalid partial length[%llu]", __func__, static_cast<unsigned long long>(size)), HCCL_E_PARA);
    std::vector<char> data(static_cast<size_t>(size));
    CHK_PRT_RET(
        !socket.Recv(data.data(), static_cast<u32>(size)), HCCL_ERROR("[%s] recv partial failed", __func__),
        HCCL_E_NETWORK);
    partials_[peer] = std::move(data);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectScalable::MergeMeshPartials(u32 nRoot, RankTableInfo& rankTable)
{
    RankTableInfo merged{};
    for (u32 i = 0; i < nRoot; ++i) {
        CHK_PRT_RET(
            partials_[i].empty(), HCCL_ERROR("[%s] missing partial for root[%u]", __func__, i), HCCL_E_INTERNAL);
        BinaryStream stream(partials_[i]);
        RankTableInfo peerTable(stream);
        CHK_PRT_RET(
            peerTable.rankCount != peerTable.ranks.size(),
            HCCL_ERROR("[%s] rank count mismatch from root[%u]", __func__, i), HCCL_E_INTERNAL);
        if (i == 0) {
            merged = peerTable;
        } else {
            merged.ranks.insert(merged.ranks.end(), peerTable.ranks.begin(), peerTable.ranks.end());
            merged.rankCount += peerTable.rankCount;
        }
    }
    // 各组子表按 rootIndex 顺序拼接
    std::sort(merged.ranks.begin(), merged.ranks.end(), [](const hccl::NewRankInfo& lhs, const hccl::NewRankInfo& rhs) {
        return lhs.rankId < rhs.rankId;
    });
    rankTable = std::move(merged);
    return HCCL_SUCCESS;
}

} // namespace Hccl
