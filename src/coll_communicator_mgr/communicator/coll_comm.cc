/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm.h"
#include "rank_info_detect.h"
#include "root_info_detect_bridge.h"
#include "exception_handler.h"
#include "rank_graph_v2.h"
#include "kfc.h"
#include "dlhal_function.h"
#include "hcclCommTaskException.h"
#include "ubmem_symmetric_memory.h"
#include "hccl_team_mgr.h"
#include "hccl_team_c_adpt.h"
#include "hcomm_team.h"
#include "hcomm_team_c_adpt.h"
#include "hccl/hccl_channel.h"
#include "hccl/hccl_rank_graph.h"
#include "common/debug/plf_debug_config_pub.h"
#include "coll_comm_mgr.h"
#include "adapter_rts_common.h"
#include "common/hccl_log_keywords_pub.h"
#include "orion_adapter_rts.h"
#include "coll_comm_config.h"
#include "hccl/hccl_comm.h"
#include "securec.h"
#include <cstring>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <set>
#include <unordered_set>
#include <vector>
#include "launch_aicpu.h"
#include "launch_device.h"
#include "flush_manager.h"

namespace hccl {
namespace {
    std::unordered_map<HcclCommSymWindow, HcclComm> g_hcommWindowCommMap{};
    std::mutex g_hcommWindowCommMapMutex{};
    // DPUTAG共享内存协议约定，与DPU侧实现（legacy/ascend950/framework/communicator）保持一致：
    // 工作区大小100MB，前半区为NPU->DPU，后半区为DPU->NPU，flag位于各半区首字节
    constexpr u64 DPU_SHARE_HBM_MEMORY_SIZE = 100 * 1024 * 1024;
    constexpr u64 DPU_SHARE_HBM_MEMORY_HALF = DPU_SHARE_HBM_MEMORY_SIZE / 2;
    // dpu2npu半区flag语义：0=空闲，1=响应就绪，2=对端异常退出（读侧见base_comm/primitives/api_c_adpt下
    // aicpu_ts_sync_data_c_adpt.cc中的WaitFlagReady
    constexpr uint8_t DPU2NPU_FLAG_ABORT = 2;

    // rootHandle.identifier 格式为 hostIp_hostPort_devPhyId_timestamp，取倒数第二段。
    bool ParseRootDevPhyId(const Hccl::HcclRootHandleV2& rootHandle, u32& rootDevPhyId)
    {
        const std::string identifier(rootHandle.identifier);
        const size_t timestampSep = identifier.rfind('_');
        if (timestampSep == std::string::npos || timestampSep == 0) {
            return false;
        }
        const size_t devPhyIdSep = identifier.rfind('_', timestampSep - 1);
        if (devPhyIdSep == std::string::npos || devPhyIdSep + 1 >= timestampSep) {
            return false;
        }
        u64 parsed = 0;
        for (size_t i = devPhyIdSep + 1; i < timestampSep; ++i) {
            const char ch = identifier[i];
            if (ch < '0' || ch > '9') {
                return false;
            }
            parsed = parsed * 10 + static_cast<u64>(ch - '0');
            if (parsed > UINT32_MAX) {
                return false;
            }
        }
        rootDevPhyId = static_cast<u32>(parsed);
        return true;
    }

    // 注册本 rank 的 host RDMA flush handle：flush 只在 host 网卡参与通信时才需排空，故只挑 rank graph
    // 中 HOST 位置且支持 ROCE 的接口。FlushManager 与 CollComm 同处 hcomm.so，直接调用即可。
    void InitFlushHandle(void* rankGraphV2, s32 deviceLogicId, const std::string& commId)
    {
        auto* graph = static_cast<Hccl::RankGraph*>(rankGraphV2);
        if (graph == nullptr) {
            return;
        }
        auto peer = graph->GetPeer(graph->GetMyRank());
        if (peer == nullptr) {
            HCCL_WARNING("[CollComm][InitFlushHandle] peer of my rank is null, skip. commId[%s]", commId.c_str());
            return;
        }
        // 先筛出本 rank 的 host ROCE 接口：devPhyId 取不到时要能报出「0/N 一个都没注册」，
        // 否则与「本 rank 无该类接口」在日志上无从区分，flush 整体失效会被当成正常
        std::vector<Hccl::IpAddress> hostRoceIps;
        for (const auto& iface : peer->GetIfaces()) {
            if (iface == nullptr || iface->GetPos() != Hccl::AddrPosition::HOST
                || iface->GetLinkProtocols().count(Hccl::LinkProtocol::ROCE) == 0) {
                continue;
            }
            hostRoceIps.push_back(iface->GetAddr());
        }
        // 无 host ROCE 接口的通信域本就无需排空：提前返回，避免下面的取 id 失败被误报成 flush 失效
        if (hostRoceIps.empty()) {
            HCCL_INFO(
                "[CollComm][InitFlushHandle] no host roce iface, nothing to register. commId[%s]", commId.c_str());
            return;
        }
        u32 devPhyId = 0;
        if (hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId), devPhyId) != HCCL_SUCCESS) {
            // Flush() 对 map 内缺失的 ip 无任何动作，此处失败即本通信域 flush 整体失效且不可回补
            HCCL_ERROR(
                "[CollComm][InitFlushHandle] get devicePhyId failed, deviceLogicId[%d], 0/%zu host roce iface "
                "registered, Flush is disabled for this comm, commId[%s]",
                deviceLogicId, hostRoceIps.size(), commId.c_str());
            return;
        }
        u32 failNum = 0;
        for (const auto& ip : hostRoceIps) {
            if (Hccl::FlushManager::GetInstance().initFlushHandle(ip, devPhyId) != HCCL_SUCCESS) {
                ++failNum;
            }
        }
        // 注册失败不阻断通信域建立（与收编前一致），但须留痕：Flush() 对缺失的 ip 会静默跳过，排空随之失效
        if (failNum != 0) {
            HCCL_WARNING(
                "[CollComm][InitFlushHandle] %u/%zu host roce iface flush register failed, "
                "Flush() will skip them. commId[%s]",
                failNum, hostRoceIps.size(), commId.c_str());
        }
        HCCL_INFO(
            "[CollComm][InitFlushHandle] finish. host roce iface[%zu]. commId[%s]", hostRoceIps.size(), commId.c_str());
    }
} // namespace

HcclResult RecordHcommWindowOwner(HcclCommSymWindow winHandle, HcclComm comm)
{
    CHK_PTR_NULL(winHandle);
    CHK_PTR_NULL(comm);
    std::lock_guard<std::mutex> lock(g_hcommWindowCommMapMutex);
    bool inserted = false;
    EXCEPTION_CATCH(inserted = g_hcommWindowCommMap.emplace(winHandle, comm).second, return HCCL_E_MEMORY);
    CHK_PRT_RET(!inserted, HCCL_ERROR("[%s] window[%p] owner already exists", __func__, winHandle), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult GetHcommWindowComm(HcclCommSymWindow winHandle, HcclComm& comm)
{
    CHK_PTR_NULL(winHandle);
    comm = nullptr;
    std::lock_guard<std::mutex> lock(g_hcommWindowCommMapMutex);
    auto iter = g_hcommWindowCommMap.find(winHandle);
    if (iter == g_hcommWindowCommMap.end()) {
        return HCCL_E_NOT_FOUND;
    }
    comm = iter->second;
    return HCCL_SUCCESS;
}

void EraseHcommWindowOwner(HcclCommSymWindow winHandle)
{
    std::lock_guard<std::mutex> lock(g_hcommWindowCommMapMutex);
    g_hcommWindowCommMap.erase(winHandle);
}

void SymmetricMemoryDeleter::operator()(SymmetricMemory* ptr) const { delete ptr; }

CollComm::CollComm(
    void* comm, uint32_t rankId, const std::string& commName, const ManagerCallbacks& callbacks,
    CollCommInitMode initMode)
    : comm_(comm),
      rankId_(rankId),
      commId_(commName),
      config_(commName),
      callbacks_(callbacks),
      initMode_(initMode)
{
    groupScheduleMgr = std::make_shared<GroupScheduleMgr>();
}

CollComm::~CollComm()
{
    if (!IsFullMode()) { // SimpleMode是简化版collComm，只初始化了myrank、rankgraph未初始化以下资源，不需要析构处理
        return;
    }

    if (dcaInited_) {
        hccl::CollCommMgr::GetInstance().GetDetectConnectionAnomalies(deviceLogicId_).Deinit();
        dcaInited_ = false;
    }

    // 先注销TaskException，再销毁通信域资源，防止通信域资源销毁后rts回调TaskException
    hcomm::TaskExceptionHost* handler = hcomm::TaskExceptionHost::GetInstance(deviceLogicId_);
    if (handler != nullptr) {
        (void)handler->UnRegister(reinterpret_cast<u64>(this));
    }

    CHK_PRT(HcclBinaryUnLoad());
    if (derivedBinHandle_ != nullptr) {
        (void)aclrtBinaryUnLoad(derivedBinHandle_);
        derivedBinHandle_ = nullptr;
    }

    if (ubMemSymmetricMemory_ != nullptr) {
        ubMemSymmetricMemory_.reset();
    }
    // 兜底释放所有team的syncMem本地内存
    HcclTeamMgr::GetInstance().ClearByCollComm(this);
    // 兜底释放所有未注销的 HcommWindow device 副本（legacySymWin 部分由 symmetricMemory_ 析构清理）
    std::vector<HcclCommSymWindow> unregisteredWins;
    {
        std::unique_lock<std::shared_mutex> lock(hcommWindowMutex_);
        unregisteredWins.reserve(hcommToSymMap_.size());
        for (auto& pair : hcommToSymMap_) {
            unregisteredWins.emplace_back(pair.first);
        }
        hcommToSymMap_.clear();
        symToHcommMap_.clear();
    }
    // 全局window索引与L3回填资源的清理在成员锁外执行，避免持锁访问全局单例造成数据竞争
    for (HcclCommSymWindow win : unregisteredWins) {
        EraseHcommWindowOwner(win);
        (void)HcommTeamWindowDeregister(win); // 释放 L3 回填资源（remoteMems/sizes）
    }
    HCCL_INFO("[CollComm][~CollComm] collComm deinit");
    // 先停宿主侧 taskexception 握手再发 DESTROY_AICPU_COMM，否则 device 侧被置 INVALID 后不再回写停止标志，
    // ~LegacyDpuThread 里的轮询会 10s 超时报 WARNING（非 dpu 场景内部幂等短路）。
    // 本帧是兜底触发点：权威触发点是 CommunicatorImpl::DestroyImpl（设备侧仍有效时握手）；
    // 两点共用 LegacyDpuThread::StopTaskexpService 的 taskExpStopped_ 幂等位，谁先到谁做，后者短路。
    // 此处失败即意味兜底也没握上（多为超时），升级到 WARNING 让 10s 阻塞在外部可观测。
    ThreadMgr* threadMgr = GetThreadMgr();
    CpuThreadService* cpuService = (threadMgr == nullptr) ? nullptr : threadMgr->GetCpuThreadService();
    if (cpuService == nullptr) {
        // 无 commEngineResMgr_（simpleMode 或 Init 中途失败）即无 dpu 线程，握手从未发起，不算失败
        HCCL_DEBUG("[CollComm][~CollComm] no dpu service, skip taskexp handshake, commId[%s]", commId_.c_str());
    } else {
        HcclResult stopRet = cpuService->StopTaskexpService();
        if (stopRet != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[CollComm][~CollComm] stop taskexp handshake failed, commId[%s], ret[%d]", commId_.c_str(),
                static_cast<int>(stopRet));
        }
        // dfx 成员 hcclCommDfx_ 在成员析构阶段先于 commEngineResMgr_ 释放，而 kernel 错误路径经
        // getDpuTaskInfoCallback 读 dfx；须先停 kernel 关掉该 UAF 窗口（DeInit 幂等，析构二次进入空转）
        HcclResult deinitRet = cpuService->StopCpuKernel();
        if (deinitRet != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[CollComm][~CollComm] stop dpu kernel failed, commId[%s], ret[%d]", commId_.c_str(),
                static_cast<int>(deinitRet));
        }
    }
    (void)DestroyAicpuComm();
    // rankInfoDetect_ 的 Shutdown 由显式 destroy 路径（HcclCommDestroy）调用，
    // 不在析构中调用：exit() 静态析构阶段 RA（HccpPeerManager）可能已 finalize，
    // 此时 Shutdown → TearDown → Socket::Destroy → RaSocketListenStop 会因 RA 函数指针为 NULL 而 core。
    // 进程退出时未显式 destroy 的 detect 资源由 OS 自动回收（socket fd、线程）。
    HCCL_RUN_INFO("[CollComm][~CollComm] cclBuffer free, commId[%s].", commId_.c_str());
}

HcclResult CollComm::Init(void* rankGraph, aclrtBinHandle binHandle, HcclMem cclBuffer, uint32_t opExpansionMode)
{
    CHK_RET(CollCommMgr::GetInstance().InitConfigMgr());
    if (IsFullMode()) { // A5和下一代
        return InitFullMode(rankGraph, binHandle, cclBuffer, opExpansionMode);
    } else { // A2/A3使用简化版CollComm
        return InitSimpleMode(rankGraph, binHandle, cclBuffer, opExpansionMode);
    }
}

HcclResult
CollComm::InitSimpleMode(void* rankGraph, aclrtBinHandle binHandle, HcclMem cclBuffer, uint32_t opExpansionMode)
{
    CHK_PTR_NULL(rankGraph);

    EXCEPTION_HANDLE_BEGIN

    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());

    // SimpleMode: A2/A3的RankGraph是保存在hccl::Communicator中的静态对象裸指针，CollComm不负责释放
    rankgraph_ = static_cast<RankGraph*>(rankGraph);

    uint32_t rankNum = 0;
    CHK_PTR_NULL(rankgraph_);
    CHK_RET(rankgraph_->GetRankSize(&rankNum));

    EXCEPTION_CATCH(
        myRank_ = std::make_shared<MyRank>(binHandle, rankId_, config_, callbacks_, rankgraph_, rankIpPortMap_),
        return HCCL_E_PTR);

    CHK_RET(myRank_->Init(cclBuffer, opExpansionMode, rankNum));

    commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;

    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult
CollComm::InitFullMode(void* rankGraph, aclrtBinHandle binHandle, HcclMem cclBuffer, uint32_t opExpansionMode)
{
    CHK_PTR_NULL(rankGraph);
    CHK_RET(Hccl::GetEnvPlfDebugConfig().Parse());

    EXCEPTION_HANDLE_BEGIN

    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    rankGraphOwner_ = std::make_unique<RankGraphV2>(rankGraph);
    rankgraph_ = rankGraphOwner_.get();
    uint32_t rankNum = 0;
    CHK_PTR_NULL(rankgraph_);
    CHK_RET(rankgraph_->GetRankSize(&rankNum));
    CHK_RET(GetRankIpPortMap());

    u32 threadNum = 0xffffffff;
    u32 notifyNumPerThread = 0xffffffff;
    if (!commEngineResMgr_) {
        EXCEPTION_CATCH(commEngineResMgr_ = std::make_unique<CommEngineResMgr>(), return HCCL_E_PTR);
        CHK_PRT(commEngineResMgr_->Init(threadNum, notifyNumPerThread, commId_, binHandle, callbacks_));
    }

    if (!contextMgr_) {
        EXCEPTION_CATCH(contextMgr_ = std::make_unique<ContextManager>(), return HCCL_E_PTR);
    }

    EXCEPTION_CATCH(
        myRank_ = std::make_shared<MyRank>(binHandle, rankId_, config_, callbacks_, rankgraph_, rankIpPortMap_),
        return HCCL_E_PTR);
    CHK_RET(myRank_->Init(cclBuffer, opExpansionMode, rankNum));
    CHK_RET(hrtGetDevice(&deviceLogicId_));
    PrintCommKeyInfo(rankNum);
    CHK_RET(InitWorldTeams());
    CHK_RET(InitSymmetricMemory());

    CHK_RET(InitHDCommunicate());

    if (!hcclCommDfx_) {
        EXCEPTION_CATCH(hcclCommDfx_ = std::make_unique<HcclCommDfx>(), return HCCL_E_PTR);
    }
    CHK_RET(hcclCommDfx_->Init(deviceLogicId_, commId_, rankId_));
    CHK_RET(InitTaskExceptionHandler());

    CHK_RET(InitKfcAndRegisterCollComm());

    Hccl::HcclCommunicator* comV2 = static_cast<Hccl::HcclCommunicator*>(comm_);
    CHK_PTR_NULL(comV2);
    CHK_RET(comV2->GetCclBufferSharedPtr(cclBuffer_));
    hccl::CollCommMgr::GetInstance().GetDetectConnectionAnomalies(deviceLogicId_).Init();
    dcaInited_ = true;

    // flush 收编后落在这里：原 CommunicatorImpl::InitFlushHandle 的时机与本函数同属通信域建立期
    InitFlushHandle(rankGraph, deviceLogicId_, commId_);

    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

void CollComm::PrintCommKeyInfo(uint32_t rankNum) const
{
    HCCL_RUN_INFO(
        "[%s]identifier[%s] rankSize[%u] comm[%p]", LOG_KEYWORDS_COMMUNICATOR.c_str(), commId_.c_str(), rankNum, comm_);
    HCCL_RUN_INFO("[%s]userRank[%u] deviceLogicId[%d]", LOG_KEYWORDS_LOCALRANK.c_str(), rankId_, deviceLogicId_);
}

HcclResult CollComm::InitSymmetricMemory()
{
    uint32_t rankSize = GetRankSize();
    HCCL_RUN_INFO(
        "[CollComm][InitSymmetricMemory] commId[%s], rank[%u], rankSize[%u].", commId_.c_str(), rankId_, rankSize);

    // URMA与UB Memory可以同时存在；两种资源最终发布到同一个HcommWindow的不同子结构。
    EXCEPTION_CATCH(
        symmetricMemory_.reset(new SymmetricMemory(rankId_, rankSize, 0, SymmetricMemoryMode::URMA)),
        return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(symmetricMemory_);

    HcommTeamHandle lsaTeam = nullptr;
    HcclResult getLsaTeamRet = HcclTeamMgr::GetInstance().GetLsaTeam(this, lsaTeam);
    if (getLsaTeamRet == HCCL_SUCCESS) {
        uint32_t lsaNetLayer = 0;
        HcommResult getNetLayerRet = HcommTeamGetNetLayer(lsaTeam, &lsaNetLayer);
        CHK_PRT_RET(
            getNetLayerRet != HCOMM_SUCCESS,
            HCCL_ERROR("[%s] get LSA team net layer failed, team[%p], ret[%d]", __func__, lsaTeam, getNetLayerRet),
            static_cast<HcclResult>(getNetLayerRet));
        std::vector<uint32_t> worldRankIds
            = HcclTeamMgr::GetInstance().GetPrebuiltWorldTeamRanks(this, COMM_PROTOCOL_UB_MEM, lsaNetLayer);
        CHK_PRT_RET(worldRankIds.empty(), HCCL_ERROR("[%s] UB worldTeam has no members", __func__), HCCL_E_INTERNAL);
        EXCEPTION_CATCH(
            ubMemSymmetricMemory_ = std::make_unique<UbMemSymmetricMemory>(this, lsaTeam, lsaNetLayer, worldRankIds),
            return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(ubMemSymmetricMemory_);
        return ubMemSymmetricMemory_->Init();
    }
    // 未预制LSA Team表示当前通信域仅使用URMA，属于正常场景；其他返回码表示查询异常。
    CHK_PRT_RET(
        getLsaTeamRet != HCCL_E_NOT_FOUND, HCCL_ERROR("[%s] get LSA team failed, ret[%d]", __func__, getLsaTeamRet),
        getLsaTeamRet);
    return HCCL_SUCCESS;
}

HcclResult CollComm::CreatePrebuiltWorldTeam(
    CommProtocol protocol, uint32_t netLayer, const uint32_t* ranks, uint32_t rankNum, uint32_t selfMemberId)
{
    HcommTeamCreateDesc hcommDesc{};
    (void)HcommTeamCreateDescInit(&hcommDesc);
    hcommDesc.memberNum = rankNum;
    hcommDesc.selfMemberId = selfMemberId;
    hcommDesc.netLayer = netLayer;
    hcommDesc.protocol = protocol;
    hcommDesc.requirement.barrierCount = protocol == COMM_PROTOCOL_UB_MEM ? 0 : 1;

    HcommTeamHandle newTeam = nullptr;
    uint64_t syncMemSize = 0;
    HcommResult createRet = HcommTeamCreate(nullptr, &hcommDesc, &newTeam, &syncMemSize);
    CHK_PRT_RET(
        createRet != 0,
        HCCL_ERROR(
            "[CollComm][%s] HcommTeamCreate failed, protocol[%d] netLayer[%u] ret[%d]", __func__, protocol, netLayer,
            createRet),
        HCCL_E_INTERNAL);

    HcclResult regRet
        = HcclTeamMgr::GetInstance().RegisterPrebuiltWorldTeam(newTeam, this, protocol, netLayer, ranks, rankNum);
    if (regRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CollComm][%s] RegisterPrebuiltWorldTeam failed, protocol[%d] netLayer[%u] ret[%d]", __func__, protocol,
            netLayer, regRet);
        (void)HcommTeamDestroy(newTeam);
        return regRet;
    }
    HCCL_INFO(
        "[CollComm][%s] prebuilt worldTeam for protocol[%d] netLayer[%u], handle[%p]", __func__, protocol, netLayer,
        newTeam);
    return HCCL_SUCCESS;
}

void CollComm::CollectLayerReachableRanks(
    uint32_t netLayer, const uint32_t* ranks, uint32_t rankNum, ProtocolRankMap& reachableRanksByProtocol) const
{
    for (uint32_t peerIndex = 0; peerIndex < rankNum; ++peerIndex) {
        if (ranks[peerIndex] == rankId_) {
            continue;
        }
        CommLink* links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = rankgraph_->GetLinks(netLayer, rankId_, ranks[peerIndex], &links, &linkNum);
        if (ret != HCCL_SUCCESS || links == nullptr || linkNum == 0) {
            HCCL_INFO(
                "[CollComm][%s] netLayer[%u] selfRank[%u] remoteRank[%u] has no valid links, skip", __func__, netLayer,
                rankId_, ranks[peerIndex]);
            continue;
        }
        for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
            const CommLink& link = links[linkIndex];
            auto protocolIt = reachableRanksByProtocol.find(link.linkAttr.linkProtocol);
            if (protocolIt == reachableRanksByProtocol.end()) {
                continue;
            }
            if (std::find(protocolIt->second.begin(), protocolIt->second.end(), ranks[peerIndex])
                == protocolIt->second.end()) {
                protocolIt->second.emplace_back(ranks[peerIndex]);
            }
        }
    }
}

HcclResult
CollComm::CreateUrmaWorldTeams(uint32_t netLayer, uint32_t selfRankId, const ProtocolRankMap& reachableRanksByProtocol)
{
    static const std::set<CommProtocol> urmaProtocols
        = {COMM_PROTOCOL_UB_CTP, COMM_PROTOCOL_UBC_TP, COMM_PROTOCOL_UBOE, COMM_PROTOCOL_UB_RTP};
    for (CommProtocol protocol : urmaProtocols) {
        auto protocolIt = reachableRanksByProtocol.find(protocol);
        if (protocolIt == reachableRanksByProtocol.end() || protocolIt->second.empty()) {
            continue;
        }
        HcommTeamHandle worldTeam = HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(this, protocol, netLayer);
        if (worldTeam != nullptr) {
            continue;
        }
        std::vector<uint32_t> teamRanks = protocolIt->second;
        teamRanks.emplace_back(selfRankId);
        std::sort(teamRanks.begin(), teamRanks.end());
        auto selfIt = std::find(teamRanks.begin(), teamRanks.end(), selfRankId);
        uint32_t selfMemberId = static_cast<uint32_t>(selfIt - teamRanks.begin());
        CHK_RET(CreatePrebuiltWorldTeam(
            protocol, netLayer, teamRanks.data(), static_cast<uint32_t>(teamRanks.size()), selfMemberId));
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::InitWorldTeamLayer(uint32_t netLayer, uint32_t selfRankId, UbWorldTeamCandidate& ubCandidate)
{
    uint32_t* ranks = nullptr;
    uint32_t rankNum = 0;
    CHK_RET(rankgraph_->GetInstRanksByNetLayer(netLayer, &ranks, &rankNum));
    if (ranks == nullptr || rankNum <= 1) {
        HCCL_INFO("[CollComm][%s] netLayer[%u] has no valid ranks, rankNum[%u], skip", __func__, netLayer, rankNum);
        return HCCL_SUCCESS;
    }
    const uint32_t* selfIt = std::find(ranks, ranks + rankNum, selfRankId);
    if (selfIt == ranks + rankNum) {
        HCCL_INFO("[CollComm][%s] self rank[%u] is not in netLayer[%u], skip", __func__, selfRankId, netLayer);
        return HCCL_SUCCESS;
    }
    uint32_t selfMemberId = static_cast<uint32_t>(selfIt - ranks);
    ProtocolRankMap reachableRanksByProtocol
        = {{COMM_PROTOCOL_UB_CTP, {}},
           {COMM_PROTOCOL_UBC_TP, {}},
           {COMM_PROTOCOL_UBOE, {}},
           {COMM_PROTOCOL_UB_RTP, {}},
           {COMM_PROTOCOL_UB_MEM, {}}};
    CollectLayerReachableRanks(netLayer, ranks, rankNum, reachableRanksByProtocol);
    CHK_RET(CreateUrmaWorldTeams(netLayer, selfRankId, reachableRanksByProtocol));

    auto ubMemIt = reachableRanksByProtocol.find(COMM_PROTOCOL_UB_MEM);
    CHK_PRT_RET(
        ubMemIt == reachableRanksByProtocol.end(),
        HCCL_ERROR("[CollComm][%s] UB Memory protocol entry is missing", __func__), HCCL_E_INTERNAL);
    const auto& ubMemReachableRanks = ubMemIt->second;
    uint32_t leftIndex = (selfMemberId + rankNum - 1) % rankNum;
    uint32_t rightIndex = (selfMemberId + 1) % rankNum;
    bool hasLeftUbMemLink = std::find(ubMemReachableRanks.begin(), ubMemReachableRanks.end(), ranks[leftIndex])
                            != ubMemReachableRanks.end();
    bool hasRightUbMemLink = std::find(ubMemReachableRanks.begin(), ubMemReachableRanks.end(), ranks[rightIndex])
                             != ubMemReachableRanks.end();
    if (hasLeftUbMemLink && hasRightUbMemLink && (ubCandidate.ranks.empty() || netLayer > ubCandidate.netLayer)) {
        ubCandidate.netLayer = netLayer;
        ubCandidate.selfMemberId = selfMemberId;
        ubCandidate.ranks.assign(ranks, ranks + rankNum);
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::InitWorldTeams()
{
    CHK_PTR_NULL(rankgraph_);
    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(rankgraph_->GetNetLayers(&netLayers, &netLayerNum));
    CHK_PRT_RET(
        netLayers == nullptr || netLayerNum == 0,
        HCCL_ERROR("[CollComm][%s] GetNetLayers returned empty, commId[%s]", __func__, commId_.c_str()),
        HCCL_E_INTERNAL);

    UbWorldTeamCandidate ubCandidate;
    for (uint32_t layerIndex = 0; layerIndex < netLayerNum; ++layerIndex) {
        CHK_RET(InitWorldTeamLayer(netLayers[layerIndex], GetMyRankId(), ubCandidate));
    }
    if (ubCandidate.ranks.empty()) {
        return HCCL_SUCCESS;
    }
    HcommTeamHandle worldTeam
        = HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(this, COMM_PROTOCOL_UB_MEM, ubCandidate.netLayer);
    if (worldTeam == nullptr) {
        CHK_RET(CreatePrebuiltWorldTeam(
            COMM_PROTOCOL_UB_MEM, ubCandidate.netLayer, ubCandidate.ranks.data(),
            static_cast<uint32_t>(ubCandidate.ranks.size()), ubCandidate.selfMemberId));
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::RegisterSymmetricMemoryResource(void* ptr, size_t size, SymmetricMemoryResource& resource)
{
    CHK_PTR_NULL(ptr);
    CHK_PRT_RET(
        size == 0, HCCL_ERROR("[CollComm][RegisterSymmetricMemoryResource] invalid symmetric memory size 0."),
        HCCL_E_PARA);
    CHK_SMART_PTR_NULL(myRank_);

    CommMems* commMems = myRank_->GetCommMems();
    CHK_PTR_NULL(commMems);

    CommMem commMem{};
    commMem.type = COMM_MEM_TYPE_DEVICE;
    commMem.addr = ptr;
    commMem.size = static_cast<uint64_t>(size);
    resource.memTag = std::string(HCCL_SYMMETRIC_MEMORY_TAG_PREFIX) + commId_ + "_addr_"
                      + std::to_string(reinterpret_cast<uintptr_t>(ptr)) + "_size_" + std::to_string(size);
    HcclResult ret = commMems->CommRegMem(resource.memTag, commMem, &resource.memHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollComm][RegisterSymmetricMemoryResource] CommRegMem failed, tag[%s], ptr[%p], "
            "size[%zu], ret[%d].",
            resource.memTag.c_str(), ptr, size, ret),
        ret);

    HCCL_RUN_INFO(
        "[CollComm][RegisterSymmetricMemoryResource] register symmetric memory success, group[%s], "
        "tag[%s], ptr[%p], size[%zu], memHandle[%p].",
        commId_.c_str(), resource.memTag.c_str(), ptr, size, resource.memHandle);
    return HCCL_SUCCESS;
}

void CollComm::UnregisterSymmetricMemoryResource(const SymmetricMemoryResource& resource)
{
    if (resource.memHandle == nullptr || resource.memTag.empty()) {
        HCCL_WARNING(
            "[CollComm][UnregisterSymmetricMemoryResource] resource is empty, tag[%s], memHandle[%p].",
            resource.memTag.c_str(), resource.memHandle);
        return;
    }
    if (myRank_ == nullptr) {
        HCCL_WARNING(
            "[CollComm][UnregisterSymmetricMemoryResource] myRank is null, skip CommUnregMem, "
            "tag[%s], memHandle[%p].",
            resource.memTag.c_str(), resource.memHandle);
        return;
    }
    CommMems* commMems = myRank_->GetCommMems();
    if (commMems == nullptr) {
        HCCL_WARNING(
            "[CollComm][UnregisterSymmetricMemoryResource] commMems is null, skip CommUnregMem, "
            "tag[%s], memHandle[%p].",
            resource.memTag.c_str(), resource.memHandle);
        return;
    }
    HcclResult ret = commMems->CommUnregMem(resource.memTag, resource.memHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING(
            "[CollComm][UnregisterSymmetricMemoryResource] CommUnregMem is unsuccessful, tag[%s], "
            "memHandle[%p], ret[%d].",
            resource.memTag.c_str(), resource.memHandle, ret);
    }
    ret = myRank_->UnregMemByTag(resource.memTag);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CollComm][UnregisterSymmetricMemoryResource] UnregMemByTag failed, tag[%s], ret[%d].",
            resource.memTag.c_str(), ret);
        return;
    }
    HCCL_INFO(
        "[CollComm][UnregisterSymmetricMemoryResource] unregister symmetric memory success, "
        "tag[%s], memHandle[%p].",
        resource.memTag.c_str(), resource.memHandle);
}

/* 单 team 补交换：对每个 peer 建 1 条 channel（CreateChannels 内部有复用逻辑），symm memHandle 挂到
 * channelDesc 参与交换，建链后 UpdateSymmetricRemoteMem 双回填 SymmetricWindow/HcommWindow。 */
HcclResult CollComm::ReExchangeChannelsForTeam(
    HcommTeamHandle team, CommEngine engine, uint32_t netLayer, std::vector<HcclMemHandle>& symMemHandles)
{
    ChannelsCreateCtx ctx{};
    CHK_RET(GetTeamMemberInfo(team, rankId_, ctx));
    ctx.channelsByMember.assign(ctx.memberNum, {});
    for (uint32_t m = 0; m < ctx.memberNum; m++) {
        if (m == ctx.selfMemberId) {
            continue; // 不建到自己的 channel
        }
        uint32_t peerRank = ctx.rankIds[m];
        HcclChannelDesc channelDesc = {};
        HcclResult ret = HcclChannelDescInit(&channelDesc, 1);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[CollComm][%s] HcclChannelDescInit failed, ret[%d]", __func__, ret), ret);
        CommLink* links = nullptr;
        uint32_t linkNum = 0;
        ret = rankgraph_->GetLinks(netLayer, rankId_, peerRank, &links, &linkNum);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS || links == nullptr || linkNum == 0,
            HCCL_ERROR("[CollComm][%s] no link from rank[%u] to rank[%u]", __func__, rankId_, peerRank),
            HCCL_E_NOT_FOUND);
        channelDesc.remoteRank = peerRank;
        channelDesc.channelProtocol = links[0].linkAttr.linkProtocol;
        channelDesc.localEndpoint = links[0].srcEndpointDesc;
        channelDesc.remoteEndpoint = links[0].dstEndpointDesc;
        channelDesc.memHandles = symMemHandles.data(); // 外部已判空，此处确保有内容
        channelDesc.memHandleNum = static_cast<uint32_t>(symMemHandles.size());
        ctx.channelsByMember[m].assign(1, 0);
        ret = myRank_->CreateChannels(engine, commId_, &channelDesc, 1, ctx.channelsByMember[m].data());
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollComm][%s] CreateChannels failed, peerRank[%u] ret[%d]", __func__, peerRank, ret), ret);

        // 从新 channel 取回交换到的 remoteMem 并回填（SymmetricWindow + HcommWindow）
        CommMem* remoteMems = nullptr;
        uint32_t memNum = 0;
        std::vector<std::string> memTags;
        CHK_RET(myRank_->ChannelGetRemoteMems(ctx.channelsByMember[m][0], &memNum, &remoteMems, memTags));
        if (memNum > 0) {
            CHK_RET(UpdateSymmetricRemoteMem(peerRank, remoteMems, memTags));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::ReExchangeWindowsForBoundTeams()
{
    // 已建链 subTeam（无则常规时序 window 先于 Team，无需补交换）
    std::vector<HcommTeamHandle> subTeams = HcclTeamMgr::GetInstance().GetLinkedSubTeams(this);
    if (subTeams.empty()) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(rankgraph_);

    for (HcommTeamHandle team : subTeams) {
        // 运行时查 team 的 netLayer/engine（建链成功时由 HcclTeamCreate 写入 L3）
        uint32_t netLayer = 0;
        HcommResult nlRet = HcommTeamGetNetLayer(team, &netLayer);
        CHK_PRT_RET(
            nlRet != 0,
            HCCL_ERROR("[CollComm][%s] HcommTeamGetNetLayer failed, team[%p] ret[%d]", __func__, team, nlRet),
            static_cast<HcclResult>(nlRet));
        CommEngine engine = COMM_ENGINE_RESERVED;
        HcommResult engineRet = HcommTeamGetEngine(team, &engine);
        CHK_PRT_RET(
            engineRet != 0,
            HCCL_ERROR("[CollComm][%s] HcommTeamGetEngine failed, team[%p] ret[%d]", __func__, team, engineRet),
            HCCL_E_INTERNAL);

        // 注册 pending window 的 memHandle（新注册 window 的 CommRegMem 延迟到此刻执行）
        CHK_RET(RegisterPendingSymmetricMemHandles());
        std::vector<HcclMemHandle> symMemHandles;
        CHK_RET(GetAllRegisteredSymMemHandles(symMemHandles));
        if (symMemHandles.empty()) {
            continue;
        }
        CHK_RET(ReExchangeChannelsForTeam(team, engine, netLayer, symMemHandles));
        HCCL_INFO("[CollComm][%s] reexchange done, team[%p]", __func__, team);
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::RegisterHcommWindowMapping(HcclCommSymWindow devWin, void* devLegacySymWin)
{
    std::unique_lock<std::shared_mutex> lock(hcommWindowMutex_);
    bool inserted = false;
    HcclResult insertRet = HCCL_SUCCESS;
    EXCEPTION_CATCH(inserted = hcommToSymMap_.emplace(devWin, devLegacySymWin).second, insertRet = HCCL_E_MEMORY);
    CHK_PRT_RET(
        insertRet != HCCL_SUCCESS, HCCL_ERROR("[%s] add HcommWindow[%p] mapping failed", __func__, devWin), insertRet);
    CHK_PRT_RET(!inserted, HCCL_ERROR("[%s] HcommWindow[%p] already exists", __func__, devWin), HCCL_E_INTERNAL);
    auto reverseIt = symToHcommMap_.find(devLegacySymWin);
    if (reverseIt != symToHcommMap_.end()) {
        reverseIt->second = devWin;
        return HCCL_SUCCESS;
    }
    insertRet = HCCL_SUCCESS;
    EXCEPTION_CATCH(symToHcommMap_.emplace(devLegacySymWin, devWin), insertRet = HCCL_E_MEMORY);
    if (insertRet != HCCL_SUCCESS) {
        hcommToSymMap_.erase(devWin);
        return insertRet;
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::FindLegacySymmetricWindow(HcclCommSymWindow devWin, void*& devLegacySymWin)
{
    std::shared_lock<std::shared_mutex> lock(hcommWindowMutex_);
    auto hcommIt = hcommToSymMap_.find(devWin);
    CHK_PRT_RET(
        hcommIt == hcommToSymMap_.end(), HCCL_ERROR("[%s] HcommWindow[%p] is not registered", __func__, devWin),
        HCCL_E_NOT_FOUND);
    devLegacySymWin = hcommIt->second;
    return HCCL_SUCCESS;
}

HcclResult CollComm::UnregisterHcommWindowMapping(HcclCommSymWindow devWin, void*& devLegacySymWin)
{
    std::unique_lock<std::shared_mutex> lock(hcommWindowMutex_);
    auto hcommIt = hcommToSymMap_.find(devWin);
    CHK_PRT_RET(
        hcommIt == hcommToSymMap_.end(), HCCL_ERROR("[%s] HcommWindow[%p] is not registered", __func__, devWin),
        HCCL_E_NOT_FOUND);
    devLegacySymWin = hcommIt->second;
    hcommToSymMap_.erase(hcommIt);
    symToHcommMap_.erase(devLegacySymWin);
    return HCCL_SUCCESS;
}

void CollComm::RemoveHcommWindow(HcclCommSymWindow devWin)
{
    void* devLegacySymWin = nullptr;
    HcclResult ret = UnregisterHcommWindowMapping(devWin, devLegacySymWin);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[%s] unregister HcommWindow[%p] mapping unsuccessfully, ret[%d]", __func__, devWin, ret);
        return;
    }
    (void)HcommTeamWindowDeregister(devWin);
    CHK_PRT(symmetricMemory_->DeregisterUrmaSymmetricMem(devLegacySymWin));
}

HcclResult
CollComm::PrepareSharedSymmetricWindow(void* ptr, size_t size, void*& devLegacySymWin, HcclCommSymWindow& devWin)
{
    CHK_RET(symmetricMemory_->RegisterUrmaSymmetricMem(ptr, size, &devLegacySymWin));
    HcommResult ret = HcommTeamWindowRegister(devLegacySymWin, &devWin);
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[%s] register HcommWindow failed, ret[%d]", __func__, ret);
        CHK_PRT(symmetricMemory_->DeregisterUrmaSymmetricMem(devLegacySymWin));
        return HCCL_E_INTERNAL;
    }
    HcclResult hcclRet = RegisterHcommWindowMapping(devWin, devLegacySymWin);
    if (hcclRet != HCCL_SUCCESS) {
        (void)HcommTeamWindowDeregister(devWin);
        CHK_PRT(symmetricMemory_->DeregisterUrmaSymmetricMem(devLegacySymWin));
        return hcclRet;
    }

    ret = HcommTeamWindowSetSelfInfo(devWin, ptr, size, nullptr, 0);
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[%s] set HcommWindow self information failed, ret[%d]", __func__, ret);
        RemoveHcommWindow(devWin);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::RegisterWindow(HcclComm comm, void* ptr, size_t size, HcclCommSymWindow* winHandle)
{
    CHK_SMART_PTR_NULL(symmetricMemory_);
    CHK_PTR_NULL(winHandle);

    void* devLegacySymWin = nullptr;
    HcclCommSymWindow devWin = nullptr;
    CHK_RET(PrepareSharedSymmetricWindow(ptr, size, devLegacySymWin, devWin));

    // URMA专用的Window后注册补交换：若Team已完成Channel建链，则携带新Window的memHandle
    //    重新执行通道交换并回填远端内存信息；常规的Window先注册、Team后建链时序下为空操作。
    HcclResult ret = ReExchangeWindowsForBoundTeams();
    if (ret != HCCL_SUCCESS) {
        RemoveHcommWindow(devWin);
        return ret;
    }

    // UB Memory与URMA共用同一个HcommWindow；URMA维护netWin/legacySymWindow，UB Memory补充lsaWin。
    if (ubMemSymmetricMemory_ != nullptr) {
        ret = ubMemSymmetricMemory_->RegisterWindow(ptr, size, devWin, comm);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CollComm][%s] UB Memory window registration failed, commId[%s], ptr[%p], size[%zu], ret[%d]",
                __func__, commId_.c_str(), ptr, size, ret);
            RemoveHcommWindow(devWin);
            return ret;
        }
    } else if (IsFullMode()) {
        ret = RecordHcommWindowOwner(devWin, comm);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[CollComm][%s] record HcommWindow[%p] owner failed, ret[%d]", __func__, devWin, ret);
            RemoveHcommWindow(devWin);
            return ret;
        }
    }

    *winHandle = devWin;

    HCCL_RUN_INFO(
        "[CollComm][RegisterWindow] success, HcommWindow[%p], devLegacySymWin[%p], ptr[%p], size[%zu]", devWin,
        devLegacySymWin, ptr, size);
    return HCCL_SUCCESS;
}

HcclResult CollComm::DeregisterWindow(HcclCommSymWindow winHandle)
{
    CHK_SMART_PTR_NULL(symmetricMemory_);

    // 先通过Host索引取得底层URMA Window，完成窗口内容清理后再销毁外层HcommWindow。
    void* devLegacySymWin = nullptr;
    CHK_RET(FindLegacySymmetricWindow(winHandle, devLegacySymWin));
    HcclResult firstError = HCCL_SUCCESS;

    // UB Memory侧先注销：失败时不推进URMA侧清理；
    // 避免legacy SymmetricWindow先被销毁而UB窗口记录仍ACTIVE导致的悬空引用。
    if (ubMemSymmetricMemory_ != nullptr) {
        HcclResult ubDeregRet = ubMemSymmetricMemory_->DeregisterWindow(winHandle);
        if (ubDeregRet != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CollComm][%s] deregister UB Memory window[%p] failed, ret[%d]", __func__, winHandle, ubDeregRet);
            return ubDeregRet;
        }
    }

    // 以下各步任一失败都记录首个错误码并继续清理，保证Host索引与HcommWindow始终被移除，
    // 避免中途失败后索引残留悬空。
    SymmetricMemoryResource resource;
    HcclResult getResourceRet = symmetricMemory_->GetRegisteredMemoryResource(devLegacySymWin, resource);
    // 清理 tagToHcommMap_ 中对应的 tag 条目
    if (getResourceRet == HCCL_SUCCESS && !resource.memTag.empty()) {
        std::unique_lock<std::shared_mutex> lock(hcommWindowMutex_);
        tagToHcommMap_.erase(resource.memTag);
    } else if (getResourceRet != HCCL_SUCCESS && getResourceRet != HCCL_E_NOT_FOUND) {
        HCCL_WARNING(
            "[CollComm][DeregisterWindow] get registered symmetric memory resource unsuccessfully, "
            "devLegacySymWin[%p], ret[%d].",
            devLegacySymWin, getResourceRet);
    }

    HcclResult urmaDeregRet = symmetricMemory_->DeregisterUrmaSymmetricMem(devLegacySymWin);
    if (urmaDeregRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CollComm][%s] deregister URMA symmetric memory[%p] failed, ret[%d]", __func__, devLegacySymWin,
            urmaDeregRet);
        firstError = urmaDeregRet;
    }
    if (getResourceRet == HCCL_SUCCESS && urmaDeregRet == HCCL_SUCCESS) {
        // 对称内存窗口注销后，同步删除本地memTag到memHandle索引。
        {
            std::unique_lock<std::shared_mutex> lock(registeredSymMemHandleMapMtx_);
            registeredSymMemHandleMap_.erase(resource.memTag);
        }
        UnregisterSymmetricMemoryResource(resource);
    }

    void* detachedLegacySymWin = nullptr;
    HcclResult unregisterMappingRet = UnregisterHcommWindowMapping(winHandle, detachedLegacySymWin);
    if (unregisterMappingRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CollComm][%s] unregister HcommWindow[%p] mapping failed, ret[%d]", __func__, winHandle,
            unregisterMappingRet);
        if (firstError == HCCL_SUCCESS) {
            firstError = unregisterMappingRet;
        }
    } else if (detachedLegacySymWin != devLegacySymWin) {
        HCCL_ERROR(
            "[CollComm][%s] HcommWindow[%p] legacy window changed from[%p] to[%p]", __func__, winHandle,
            devLegacySymWin, detachedLegacySymWin);
        if (firstError == HCCL_SUCCESS) {
            firstError = HCCL_E_INTERNAL;
        }
    }

    // 先移除Host索引，再销毁Device HcommWindow，避免使用已注销的winHandle执行索引操作。
    HcommResult unregRet = HcommTeamWindowDeregister(winHandle);
    if (unregRet != HCOMM_SUCCESS) {
        HCCL_ERROR("[CollComm][%s] deregister HcommWindow[%p] failed, ret[%d]", __func__, winHandle, unregRet);
        if (firstError == HCCL_SUCCESS) {
            firstError = HCCL_E_INTERNAL;
        }
    }
    return firstError;
}

HcclResult CollComm::GetCommSymWin(void* ptr, size_t size, HcclCommSymWindow* winHandle, size_t* offset)
{
    CHK_SMART_PTR_NULL(symmetricMemory_);
    // FindUrmaSymmetricWindow 返回 devLegacySymWin，经 symToHcommMap_ O(1) 反查 HcommWindow handle
    void* devLegacySymWin = nullptr;
    HcclResult ret = symmetricMemory_->FindUrmaSymmetricWindow(ptr, size, &devLegacySymWin, offset);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[CollComm][GetCommSymWin] FindUrmaSymmetricWindow failed, ret[%d]", ret), ret);
    if (devLegacySymWin == nullptr) {
        // 保持原有查询语义：未命中不是错误，由调用方根据空句柄回退到普通内存路径。
        *winHandle = nullptr;
        *offset = 0;
        return HCCL_SUCCESS;
    }
    {
        std::shared_lock<std::shared_mutex> lock(hcommWindowMutex_);
        auto symIt = symToHcommMap_.find(devLegacySymWin);
        CHK_PRT_RET(
            symIt == symToHcommMap_.end(),
            HCCL_ERROR("[CollComm][GetCommSymWin] no HcommWindow for devLegacySymWin[%p]", devLegacySymWin),
            HCCL_E_NOT_FOUND);
        *winHandle = symIt->second;
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::RegisterPendingSymmetricMemHandles()
{
    if (symmetricMemory_ == nullptr) {
        return HCCL_SUCCESS;
    }

    std::vector<SymmetricMemoryRegisterInfo> registerInfos;
    // HcclCommSymWinRegister只记录窗口，真正CommRegMem延迟到ChannelAcquire阶段执行。
    CHK_RET(symmetricMemory_->GetPendingRegisterInfos(registerInfos));
    if (registerInfos.empty()) {
        return HCCL_SUCCESS;
    }

    for (const SymmetricMemoryRegisterInfo& registerInfo : registerInfos) {
        SymmetricMemoryResource resource;
        HcclResult ret = RegisterSymmetricMemoryResource(registerInfo.userVa, registerInfo.userSize, resource);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CollComm][RegisterPendingSymmetricMemHandles] register symmetric memory failed, "
                "win[%p], userVa[%p], size[%zu], ret[%d].",
                registerInfo.devWin, registerInfo.userVa, registerInfo.userSize, ret);
            return ret;
        }

        ret = symmetricMemory_->SetRegisteredMemoryResource(registerInfo.devWin, resource);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CollComm][RegisterPendingSymmetricMemHandles] save symmetric memory resource failed, "
                "win[%p], userVa[%p], size[%zu], ret[%d].",
                registerInfo.devWin, registerInfo.userVa, registerInfo.userSize, ret);
            UnregisterSymmetricMemoryResource(resource);
            return ret;
        }
        // 注册成功后加入本地索引，供新建通道和后续复用通道使用。
        {
            std::unique_lock<std::shared_mutex> lock(registeredSymMemHandleMapMtx_);
            registeredSymMemHandleMap_[resource.memTag] = static_cast<HcclMemHandle>(resource.memHandle);
        }
        // 登记 memTag→HcommWindow 映射，供 UpdateHcommWindowRemoteMem 直接查
        {
            std::unique_lock<std::shared_mutex> lock(hcommWindowMutex_);
            // symToHcommMap_ 反查 devLegacySymWin 对应的 HcommWindow handle；查不到则不登记该 memTag
            auto symIt = symToHcommMap_.find(registerInfo.devWin);
            if (symIt != symToHcommMap_.end()) {
                tagToHcommMap_[resource.memTag] = symIt->second;
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult CollComm::GetAllRegisteredSymMemHandles(std::vector<HcclMemHandle>& memHandles) const
{
    memHandles.clear();
    std::shared_lock<std::shared_mutex> lock(registeredSymMemHandleMapMtx_);
    for (const auto& registeredMem : registeredSymMemHandleMap_) {
        memHandles.emplace_back(registeredMem.second);
    }
    HCCL_INFO("[CollComm][GetAllRegisteredSymMemHandles] registeredMemHandleNum[%zu].", memHandles.size());
    return HCCL_SUCCESS;
}

HcclResult CollComm::GetRemoteMissingSymMemHandles(
    const std::vector<std::string>& remoteMemTags, std::vector<HcclMemHandle>& memHandles) const
{
    memHandles.clear();
    const std::unordered_set<std::string> remoteMemTagSet(remoteMemTags.begin(), remoteMemTags.end());
    std::shared_lock<std::shared_mutex> lock(registeredSymMemHandleMapMtx_);
    // 返回目标通道远端尚未拥有的本地句柄。
    for (const auto& registeredMem : registeredSymMemHandleMap_) {
        if (remoteMemTagSet.find(registeredMem.first) == remoteMemTagSet.end()) {
            memHandles.emplace_back(registeredMem.second);
        }
    }
    HCCL_INFO(
        "[CollComm][GetRemoteMissingSymMemHandles] remoteMemTagNum[%zu], missingMemHandleNum[%zu].",
        remoteMemTags.size(), memHandles.size());
    return HCCL_SUCCESS;
}

HcclResult CollComm::UpdateSymmetricRemoteMem(
    uint32_t remoteRank, const CommMem* remoteMems, const std::vector<std::string>& memTags)
{
    if (symmetricMemory_ == nullptr) {
        return HCCL_SUCCESS;
    }
    CHK_RET(symmetricMemory_->UpdateRemoteMem(remoteRank, remoteMems, memTags));
    // 同时回填 HcommWindow.netWin.baseRemoteMemAddr 偏移表（window 属于通信域，由 CollComm 统一回填）
    return UpdateHcommWindowRemoteMem(remoteRank, remoteMems, memTags);
}

HcclResult CollComm::UpdateHcommWindowRemoteMem(
    uint32_t remoteRank, const CommMem* remoteMems, const std::vector<std::string>& memTags)
{
    // L2 计算层分段信息：sizes 为各层 worldTeam 大小（变长，下标=netLayer），slots 为 remoteRank 在各层的最终槽位
    std::vector<uint32_t> sizes;
    HcclTeamMgr::GetInstance().GetWorldTeamSizesPerNetLayer(this, sizes);
    std::vector<std::pair<uint32_t, uint32_t>> layerSlots;
    HcclTeamMgr::GetInstance().GetRankLayerSlots(this, remoteRank, layerSlots);
    // 本端（self rank）层槽位：与 remoteRank 同一算法（sum(sizes[0..L-1]) + 该层 worldTeamId）。
    // 每个对端 rank 到达都会调一次本函数，这里登记是幂等的（重复写相同值无害）
    std::vector<std::pair<uint32_t, uint32_t>> selfLayerSlots;
    HcclTeamMgr::GetInstance().GetRankLayerSlots(this, rankId_, selfLayerSlots);
    std::vector<uint32_t> selfSlots;
    for (const auto& layerSlot : selfLayerSlots) {
        uint32_t offset = 0;
        for (uint32_t l = 0; l < layerSlot.first && l < sizes.size(); ++l) {
            offset += sizes[l];
        }
        selfSlots.emplace_back(offset + layerSlot.second);
    }
    std::vector<uint32_t> slots;
    for (const auto& layerSlot : layerSlots) {
        uint32_t offset = 0;
        for (uint32_t l = 0; l < layerSlot.first && l < sizes.size(); ++l) {
            offset += sizes[l];
        }
        slots.emplace_back(offset + layerSlot.second);
    }

    CHK_PTR_NULL(remoteMems);
    std::shared_lock<std::shared_mutex> lock(hcommWindowMutex_);
    for (size_t i = 0; i < memTags.size(); ++i) {
        if (memTags[i].empty()) {
            continue;
        }
        // 直接查 tagToHcommMap_ 得 HcommWindow handle（void*，与 HcclCommSymWindow 一致）
        auto it = tagToHcommMap_.find(memTags[i]);
        if (it == tagToHcommMap_.end()) {
            continue;
        }
        // 先补齐本端槽位（VA/size 已在 RegisterWindow 登记，这里只登记槽位；幂等）
        if (!selfSlots.empty()) {
            HcommResult selfRet = HcommTeamWindowSetSelfInfo(
                it->second, nullptr, 0, selfSlots.data(), static_cast<uint32_t>(selfSlots.size()));
            CHK_PRT_RET(
                selfRet != 0,
                HCCL_ERROR("[CollComm][UpdateHcommWindowRemoteMem] set self slots failed, ret[%d]", selfRet),
                static_cast<HcclResult>(selfRet));
        }
        HcommResult hRet = HcommTeamUpdateWindowRemoteMemByRank(
            it->second, sizes.data(), static_cast<uint32_t>(sizes.size()), slots.data(),
            static_cast<uint32_t>(slots.size()), &remoteMems[i]);
        CHK_PRT_RET(
            hRet != 0,
            HCCL_ERROR(
                "[CollComm][UpdateHcommWindowRemoteMem] failed, handle[%p] remoteRank[%u] ret[%d]", it->second,
                remoteRank, hRet),
            static_cast<HcclResult>(hRet));
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::InitKfcAndRegisterCollComm()
{
    myRank_->SetKfcControlTransfer(kfcControlTransferH2D_, kfcStatusTransferD2H_);
    commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;
    return HCCL_SUCCESS;
}

HcclResult CollComm::ShutdownRankInfoDetect()
{
    if (rankInfoDetect_ != nullptr) {
        CHK_RET(rankInfoDetect_->Shutdown());
    }
    // 防御性兜底：清除 CommunicatorImpl 侧的 shared_ptr 副本，避免 detect 空壳对象残留到进程退出。
    // Shutdown 幂等，即使后续 ~RankInfoDetect 再调也不会重复清理
    Hccl::HcclCommunicator* comV2 = static_cast<Hccl::HcclCommunicator*>(comm_);
    if (comV2 != nullptr && comV2->GetRankInfoDetect() != nullptr) {
        comV2->SetRankInfoDetect(nullptr);
    }
    rankInfoDetect_.reset();
    return HCCL_SUCCESS;
}

HcclResult CollComm::DestroyAicpuComm()
{
    CHK_PTR_NULL(callbacks_.getAicpuCommState);
    if (callbacks_.getAicpuCommState()) {
        CHK_SMART_PTR_NULL(kfcControlTransferH2D_);
        CHK_SMART_PTR_NULL(kfcStatusTransferD2H_);

        Hccl::KfcCommand opCmd = Hccl::KfcCommand::DESTROY_AICPU_COMM;
        CHK_RET(kfcControlTransferH2D_->Put(0, sizeof(Hccl::KfcCommand), reinterpret_cast<uint8_t*>(&opCmd)));
        HCCL_RUN_INFO(
            "[%s]group[%s] send Hccl::KfcCommand[%d] success", __func__, commId_.c_str(), static_cast<int>(opCmd));

        Hccl::KfcExecStatus opInfo;
        constexpr u32 WAIT_CMD_TIMEOUT = 10 * 1000; // 最大等待10秒
        auto timeout = std::chrono::milliseconds(WAIT_CMD_TIMEOUT);
        auto startTime = std::chrono::steady_clock::now();

        while (true) {
            CHK_RET(kfcStatusTransferD2H_->Get(0, sizeof(Hccl::KfcExecStatus), reinterpret_cast<uint8_t*>(&opInfo)));
            if (opInfo.kfcStatus == Hccl::KfcStatus::DESTROY_AICPU_COMM_DONE) {
                HCCL_RUN_INFO("[%s]get Hccl::KfcStatus[%d] success", __func__, static_cast<int>(opInfo.kfcStatus));
                return HCCL_SUCCESS;
            } else if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
                HCCL_ERROR(
                    "[%s]timeout, maxTime[%u ms] and get the opExecStatus is [%s].", __func__, WAIT_CMD_TIMEOUT,
                    opInfo.kfcStatus.Describe().c_str());
                return HCCL_E_TIMEOUT;
            }
            usleep(TEN_MILLISECOND_OF_USLEEP);
        }
    }
    return HCCL_SUCCESS;
}

uint32_t CollComm::GetMyRankId() const { return rankId_; }

void CollComm::SetRootHandle(const Hccl::HcclRootHandleV2& handle)
{
    rootHandle_ = handle;
    // rootHandle 是通信域级 root 身份的唯一来源，detect 是后续 Grow/Shrink 的实际执行者。
    if (rankInfoDetect_ != nullptr) {
        rankInfoDetect_->SetRootHandle(handle);
    }
}

void CollComm::SetRankInfoDetect(std::shared_ptr<Hccl::RankInfoDetect> detect)
{
    rankInfoDetect_ = std::move(detect);
    // detect 接管后立即同步当前域的完整 rootHandle（含无效/无 root 状态）。
    if (rankInfoDetect_ != nullptr) {
        rankInfoDetect_->SetRootHandle(rootHandle_);
    }
}

void CollComm::TransferRankInfoDetect(CollComm& target)
{
    auto detect = TakeRankInfoDetect();
    target.SetRankInfoDetect(std::move(detect));
    if (target.GetRankInfoDetect() != nullptr) {
        HCCL_INFO("[CollComm][%s] RankInfoDetect ownership transferred to derived comm.", __func__);
    }
}

HcclResult CollComm::TakeRankInfoDetectFromCommunicator()
{
    Hccl::HcclCommunicator* comV2 = static_cast<Hccl::HcclCommunicator*>(comm_);
    if (comV2 == nullptr) {
        HCCL_INFO("[CollComm][%s] communicator is null, skip.", __func__);
        return HCCL_SUCCESS;
    }
    // move 语义取走所有权：CommunicatorImpl 不再持有，避免两份 shared_ptr 导致 Shutdown 时机不确定。
    std::shared_ptr<void> ctx = comV2->GetRankInfoDetect();
    comV2->SetRankInfoDetect(nullptr);
    auto detect = std::static_pointer_cast<Hccl::RankInfoDetect>(ctx);
    if (detect != nullptr) {
        const Hccl::HcclRootHandleV2& rootHandle = detect->GetRootHandleInfo();
        CHK_PRT_RET(
            rootHandle.listenPort == HCCL_INVALID_PORT,
            HCCL_ERROR("[CollComm][%s] rootHandle is not initialized by rank info detect.", __func__), HCCL_E_INTERNAL);
        // rootRankId 允许在首次 Grow/Shrink 前补取，init 阶段只要求 rootHandle 端点有效。
        CHK_PRT_RET(
            rootHandle.rootRankId != UINT32_MAX && rootHandle.rootRankId >= GetRankSize(),
            HCCL_ERROR(
                "[CollComm][%s] invalid root rankId[%u], rankSize[%u].", __func__, rootHandle.rootRankId,
                GetRankSize()),
            HCCL_E_INTERNAL);
        SetRootHandle(rootHandle);
        SetRankInfoDetect(detect);
        CHK_RET(InitDerivationFields());
        HCCL_INFO("[CollComm][%s] RankInfoDetect taken from communicator.", __func__);
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::GetCollCommRootInfo(Hccl::HcclRootHandleV2& rootInfo)
{
    if (IsRootEndpointValid()) {
        if (rootHandle_.rootRankId == UINT32_MAX) {
            // rootRankId 在 init 阶段可能未知：root 直接使用自身 rank，非 root 从本地拓扑解析。
            CHK_RET(ResolveRootRankIdForDerivation());
            SetRootHandle(rootHandle_);
        }
        CHK_PRT_RET(
            generationIndex_ == UINT32_MAX,
            HCCL_ERROR("[CollComm][%s] generationIndex overflow, current[%u].", __func__, generationIndex_),
            HCCL_E_PARA);
        rootInfo = rootHandle_;
        CHK_SAFETY_FUNC_RET(memcpy_s(
            rootInfo.topParentIdentifier, sizeof(rootInfo.topParentIdentifier), topParentIdentifier_.c_str(),
            topParentIdentifier_.size() + 1));
        rootInfo.generationIndex = generationIndex_ + 1;
        rootInfo.valid = 1;
        CHK_PRT_RET(
            rootInfo.rootRankId == UINT32_MAX || rootInfo.rootRankId >= GetRankSize(),
            HCCL_ERROR(
                "[CollComm][%s] invalid rootHandle, rootRankId[%u], rankSize[%u].", __func__, rootInfo.rootRankId,
                GetRankSize()),
            HCCL_E_INTERNAL);
        HCCL_INFO(
            "[CollComm][%s] rootHandle already valid. rootRankId[%u] listenPort[%u] topParentId[%s] genIndex[%u].",
            __func__, rootInfo.rootRankId, rootInfo.listenPort, topParentIdentifier_.c_str(), generationIndex_);
        return HCCL_SUCCESS;
    }

    // root 重选结果已暂存时直接复用，避免重复调用创建多个 server。
    if (pendingDetect_ != nullptr) {
        auto detect = std::static_pointer_cast<Hccl::RankInfoDetect>(pendingDetect_);
        CHK_PTR_NULL(detect);
        rootInfo = detect->GetRootHandleInfo();
        HCCL_INFO(
            "[CollComm][%s] new-root re-selection reused. rootRankId[%u] listenPort[%u] topParentId[%s] genIndex[%u].",
            __func__, rootInfo.rootRankId, rootInfo.listenPort, topParentIdentifier_.c_str(), generationIndex_);
        return HCCL_SUCCESS;
    }

    HCCL_INFO(
        "[CollComm][%s] new-root re-selection. topParentId[%s] genIndex[%u].", __func__, topParentIdentifier_.c_str(),
        generationIndex_);
    const Hccl::RootInfoDetectBridge* bridge = Hccl::GetRootInfoDetectBridge();
    CHK_PRT_RET(
        bridge == nullptr || bridge->getRootInfoWithDerivation == nullptr,
        HCCL_ERROR("[CollComm][%s] RootInfoDetectBridge is not registered.", __func__), HCCL_E_INTERNAL);

    HcclRootInfo rootInfoData{};
    Hccl::RootInfoDetectBridge::DetectContext detectContext;
    CHK_RET(bridge->getRootInfoWithDerivation(
        &rootInfoData, topParentIdentifier_.c_str(), generationIndex_, detectContext));
    auto detect = std::static_pointer_cast<Hccl::RankInfoDetect>(detectContext);
    CHK_PTR_NULL(detect);

    rootInfo = detect->GetRootHandleInfo();
    rootInfo.rootRankId = GetMyRankId();
    detect->SetRootHandle(rootInfo);
    SetPendingDetect(std::move(detectContext));

    HCCL_INFO(
        "[CollComm][%s] new-root re-selection success. rootRankId[%u] listenPort[%u] topParentId[%s] genIndex[%u].",
        __func__, rootInfo.rootRankId, rootInfo.listenPort, topParentIdentifier_.c_str(), generationIndex_);
    return HCCL_SUCCESS;
}

HcclResult CollComm::LoadCclKernelBinHandle(aclrtBinHandle& outBinHandle)
{
    std::string hcclJsonPath;
    CHK_RET(GetKernelFilePath(hcclJsonPath));
    hcclJsonPath += "ccl_kernel.json";
    CHK_RET(LoadBinaryFromFile(hcclJsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0, outBinHandle));
    HCCL_INFO("[CollComm::LoadCclKernelBinHandle] loaded ccl_kernel binary, path[%s].", hcclJsonPath.c_str());
    return HCCL_SUCCESS;
}

bool CollComm::IsRoot() const
{
    auto rankInfoDetect = GetRankInfoDetect();
    if (rankInfoDetect == nullptr) {
        return false;
    }
    return rankInfoDetect->IsRoot();
}

HcclResult CollComm::GetParentRankId(u32& parentRankId) const
{
    Hccl::HcclCommunicator* comV2 = static_cast<Hccl::HcclCommunicator*>(comm_);
    CHK_PTR_NULL(comV2);
    parentRankId = comV2->GetRankInParentComm();
    return HCCL_SUCCESS;
}

HcclResult CollComm::InitHDCommunicate()
{
    // 初始化aicpu进程 host-device 共享内存
    EXCEPTION_CATCH(
        (kfcControlTransferH2D_
         = std::make_shared<hccl::HDCommunicate>(deviceLogicId_, HCCL_HDC_TYPE_H2D, sizeof(Hccl::KfcCommand))),
        return HCCL_E_PTR);
    CHK_RET(kfcControlTransferH2D_->InitHost());

    EXCEPTION_CATCH(
        (kfcStatusTransferD2H_
         = std::make_shared<hccl::HDCommunicate>(deviceLogicId_, HCCL_HDC_TYPE_D2H, sizeof(Hccl::KfcExecStatus))),
        return HCCL_E_PTR);
    CHK_RET(kfcStatusTransferD2H_->InitHost());

    return HCCL_SUCCESS;
}

HcclResult CollComm::GetHDCommunicate(
    HDCommunicateParams& kfcControlTransferH2DParams, HDCommunicateParams& kfcStatusTransferD2HParams)
{
    CHK_SMART_PTR_NULL(kfcControlTransferH2D_);
    CHK_SMART_PTR_NULL(kfcStatusTransferD2H_);
    kfcControlTransferH2DParams = kfcControlTransferH2D_->GetCommunicateParams();
    kfcStatusTransferD2HParams = kfcStatusTransferD2H_->GetCommunicateParams();
    HCCL_INFO("%s success, group[%s]", __func__, commId_.c_str());
    return HCCL_SUCCESS;
}

HcclCommStatus CollComm::GetCommStatus() const
{
    std::lock_guard<std::mutex> lock(commMutex_);
    return commStatus_;
}

HcclResult CollComm::Suspend()
{
    HCCL_RUN_INFO("[CollComm][NsRecovery][%s] commId[%s] start to suspend.", __func__, commId_.c_str());
    {
        std::lock_guard<std::mutex> lock(commMutex_);
        if (commStatus_ == HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING) {
            HCCL_WARNING(
                "[CollComm][NsRecovery][%s] commId[%s] The current communication has been suspended, no need to "
                "suspend again.",
                __func__, commId_.c_str());
            return HcclResult::HCCL_SUCCESS;
        }

        CHK_SMART_PTR_NULL(myRank_);

        commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING;
    }

    // 通信域异常退出时，AICPU侧可能正阻塞在HcommWaitResponse中轮询DPU响应（见WaitFlagReady），
    // 此处向DPU->NPU半区首字节写入异常退出标志，令其立即结束等待，不再等到自身超时。
    // 经 CpuThreadService 取地址（只查不拉起，挂起不新建 DPU 线程）。
    HCCL_INFO("[CollComm][NsRecovery][%s] commId[%s] start to terminate WaitResponse", __func__, commId_.c_str());
    void* addr{nullptr};
    ThreadMgr* threadMgr = GetThreadMgr();
    CpuThreadService* cpuService = (threadMgr == nullptr) ? nullptr : threadMgr->GetCpuThreadService();
    HcclResult addrRet = (cpuService == nullptr) ? HCCL_E_PTR : cpuService->GetCpuTagAddrIfLaunched(addr);
    // addr为nullptr说明本通信域未拉起DPU（没有DPUTAG），此时无事可写
    if (addrRet != HCCL_SUCCESS || addr == nullptr) {
        // 通知失败不阻断挂起流程，AICPU侧仍会按自身超时退出等待
        HCCL_RUN_INFO(
            "[CollComm][NsRecovery][%s] get dpu share mem failed, skip write terminate flag, commId[%s], "
            "addr[%p], ret[%d]",
            __func__, commId_.c_str(), addr, addrRet);
        return myRank_->StopLaunch();
    }
    void* dpu2npuFlag = static_cast<uint8_t*>(addr) + DPU_SHARE_HBM_MEMORY_HALF;
    // 与DPU线程写同一flag的方式保持一致（legacy_dpu_thread/task_service.cc SynchronizeControlInfo为release语义），
    // 避免对同一字节的原子/非原子混合访问
    static_cast<std::atomic<uint8_t>*>(dpu2npuFlag)->store(DPU2NPU_FLAG_ABORT, std::memory_order_release);
    HCCL_RUN_INFO(
        "[CollComm][NsRecovery][%s] write terminate flag success, commId[%s], flagAddr[%p]", __func__, commId_.c_str(),
        dpu2npuFlag);

    return myRank_->StopLaunch();
}

HcclResult CollComm::Clean()
{
    HCCL_RUN_INFO("[CollComm][NsRecovery][%s] commId[%s] start to clean.", __func__, commId_.c_str());
    {
        std::lock_guard<std::mutex> lock(commMutex_);
        if (commStatus_ != HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING) {
            HCCL_ERROR(
                "[CollComm][NsRecovery][%s] commId[%s] The current communication is not suspended, cannot clean, "
                "status is [%u]",
                __func__, commId_.c_str(), static_cast<uint32_t>(commStatus_));
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        if (isCleaned_) {
            HCCL_WARNING(
                "[CollComm][NsRecovery][%s] commId[%s] The current communication has been cleaned, no need to clean "
                "again.",
                __func__, commId_.c_str());
            return HcclResult::HCCL_SUCCESS;
        }

        CHK_SMART_PTR_NULL(myRank_);

        isCleaned_ = true;
    }

    // 先清理Host
    return myRank_->Clean();
}

HcclResult CollComm::Resume()
{
    {
        std::lock_guard<std::mutex> lock(commMutex_);
        if (commStatus_ == HcclCommStatus::HCCL_COMM_STATUS_INVALID) {
            HCCL_ERROR(
                "[CollComm][NsRecovery][%s] commId[%s] Comm has been error, can not resume now!", __func__,
                commId_.c_str());
            return HcclResult::HCCL_E_INTERNAL;
        }
        if (commStatus_ != HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING) {
            HCCL_WARNING(
                "[CollComm][NsRecovery][%s] commId[%s] The current communication is normal, no need to resume, status "
                "is [%u]",
                __func__, commId_.c_str(), static_cast<uint32_t>(commStatus_));
            return HcclResult::HCCL_SUCCESS;
        }

        HCCL_INFO("[CollComm][NsRecovery][%s] commId[%s] start to Resume.", __func__, commId_.c_str());
        CHK_SMART_PTR_NULL(myRank_);
        auto ret = myRank_->Resume();
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CollComm][NsRecovery][%s] commId[%s] failed, ret = 0x%016llx", __func__, commId_.c_str(),
                HCCL_ERROR_CODE(ret));
            return ret;
        }

        if (commEngineResMgr_ != nullptr) {
            auto notifyRet = commEngineResMgr_->ResetCommLocalNotifies();
            if (notifyRet != HcclResult::HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[CollComm][NsRecovery][%s] commId[%s] ResetCommLocalNotifies failed, ret = 0x%016llx", __func__,
                    commId_.c_str(), HCCL_ERROR_CODE(notifyRet));
                return notifyRet;
            }
        }

        commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;
        isCleaned_ = false;
    }
    HCCL_INFO("[CollComm][NsRecovery][%s] commId[%s] resume success.", __func__, commId_.c_str());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CollComm::InitTaskExceptionHandler()
{
    hcomm::TaskExceptionHost* handler = hcomm::TaskExceptionHost::GetInstance(deviceLogicId_);
    CHK_PTR_NULL(handler);
    CHK_RET(handler->Register(reinterpret_cast<u64>(this)));
    return HCCL_SUCCESS;
}

Hccl::ErrorMessageReport CollComm::GetAicpuTaskException()
{
    Hccl::ErrorMessageReport errorMessage;
    CHK_PRT_RET(kfcStatusTransferD2H_ == nullptr, HCCL_ERROR("[%s]fail, d2h is nullptr", __func__), errorMessage);

    HcclResult ret = kfcStatusTransferD2H_->Get(
        sizeof(Hccl::KfcStatus) + sizeof(Hccl::KfcErrType), sizeof(errorMessage),
        reinterpret_cast<uint8_t*>(&errorMessage));

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]fail, group [%s], ret[%d]", __func__, commId_.c_str(), static_cast<int>(ret)), errorMessage);
    HCCL_INFO("[%s]group[%s] success", __func__, commId_.c_str());
    return errorMessage;
}

uint32_t CollComm::UpdateIndex() { return index_ += 1; }

HcclResult CollComm::GetRankIpPortMap()
{
    Hccl::HcclCommunicator* commV2 = static_cast<Hccl::HcclCommunicator*>(comm_);
    CHK_PTR_NULL(commV2);
    CHK_RET(commV2->GetRankIpPortMap(rankIpPortMap_));
    CHK_PTR_NULL(rankIpPortMap_);
    // rankIpPortMap_ 在单卡多进程场景下，用于保证端口不冲突
    // 该映射表记录了：Rank ID -> (IP地址 -> 已占用的端口号)
    return HCCL_SUCCESS;
}

HcclResult CollComm::GetHcclBinHandle(aclrtBinHandle& binHcclHandle, const std::string& soName)
{
    CHK_PRT_RET(soName.empty(), HCCL_ERROR("[%s]soName is empty.", __func__), HCCL_E_PARA);
    std::lock_guard<std::mutex> lock(binHcclmutex_);
    HCCL_DEBUG("[%s] GetHcclBinHandle", __func__);
    if (binHcclHandle_ == nullptr) {
        std::string hcclJsonPath;
        CHK_RET(GetKernelFilePath(hcclJsonPath));
        std::string jsonName = soName;
        const std::string soSuffix = ".so";
        if (jsonName.size() > soSuffix.size() && jsonName.rfind(soSuffix) == jsonName.size() - soSuffix.size()) {
            jsonName.resize(jsonName.size() - soSuffix.size());
        }
        jsonName += ".json";
        hcclJsonPath += jsonName;
        HcclResult ret
            = LoadBinaryFromFile(hcclJsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0, binHcclHandle_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s]errNo[0x%016llx]load aicpu file fail, path[%s] optionType[%u] cpuKernelMode[%u].", __func__,
                HCCL_ERROR_CODE(ret), hcclJsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0),
            ret);

        HCCL_INFO(
            "[%s]load aicpu file success, soName[%s] path[%s] optionType[%u] cpuKernelMode[%u].", __func__,
            soName.c_str(), hcclJsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0);
    }
    binHcclHandle = binHcclHandle_;
    return HCCL_SUCCESS;
}

HcclResult CollComm::HcclBinaryUnLoad()
{
    std::lock_guard<std::mutex> lock(binHcclmutex_);
    if (binHcclHandle_ == nullptr) {
        HCCL_RUN_WARNING("[%s] binHcclHandle is nullptr", __func__);
        return HCCL_SUCCESS;
    }

    HCCL_DEBUG("[%s]aclrtBinaryUnLoad binHcclHandle", __func__);
    aclError ret = aclrtBinaryUnLoad(binHcclHandle_);
    binHcclHandle_ = nullptr;
    if (ret != 0) {
        HCCL_RUN_WARNING("[%s]aclrtBinaryUnLoad is unsuccessful, aclRet[%d]", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::InitDerivationFields()
{
    // 新 rank 派生身份初始化：优先从 rootHandle 的谱系字段回填 topParentIdentifier_/generationIndex_，
    if (rootHandle_.valid == 1 && rootHandle_.topParentIdentifier[0] != '\0') {
        topParentIdentifier_ = std::string(rootHandle_.topParentIdentifier);
        generationIndex_ = rootHandle_.generationIndex;
        HCCL_INFO(
            "[CollComm][InitDerivationFields] lineage from rootHandle. commId[%s], topParentId[%s], genIndex[%u].",
            commId_.c_str(), topParentIdentifier_.c_str(), generationIndex_);
    } else {
        // 首父建域：优先使用 rootHandle.identifier，避免用户 config.hcclCommName 进入 identifier 链
        topParentIdentifier_ = rootHandle_.identifier[0] != '\0' ? std::string(rootHandle_.identifier) : commId_;
        generationIndex_ = 0;
        HCCL_INFO(
            "[CollComm][InitDerivationFields] first-parent build. commId[%s], topParentId[%s], genIndex[0].",
            commId_.c_str(), topParentIdentifier_.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::BuildDerivedCommParams(
    const std::vector<uint32_t>& survivorRanks, uint32_t myNewRankId, const std::string& commName,
    const DerivationInfo& derivationInfo, const Hccl::RankTableInfo& mergedRankTable, DerivedCommContext& ctx)
{
    Hccl::HcclCommunicator* parentCommunicator = static_cast<Hccl::HcclCommunicator*>(comm_);
    CHK_PTR_NULL(parentCommunicator);
    CHK_RET(parentCommunicator->GetDevType(ctx.devType));

    // commName 在此处即确定，CreateDerivedComm 构造/Init HcclCommunicator 时即固化
    ctx.subParams.commId = commName;
    ctx.subParams.myRank = myNewRankId;
    if (derivationInfo.isGrow) {
        ctx.subParams.rankSize = mergedRankTable.rankCount;
        ctx.subParams.rankInParentComm = myNewRankId;
    } else {
        ctx.subParams.rankSize = static_cast<u32>(survivorRanks.size());
        ctx.subParams.rankInParentComm = survivorRanks[myNewRankId];
    }
    ctx.subParams.devType = ctx.devType;
    ctx.subParams.devUsed = true;
    ctx.subParams.isWorldGroup = true;
    return HCCL_SUCCESS;
}

HcclResult CollComm::CreateDerivedComm(
    const std::vector<uint32_t>& survivorRanks, const HcclCommConfig& subConfig, const DerivationInfo& derivationInfo,
    const Hccl::RankTableInfo& mergedRankTable, DerivedCommContext& ctx)
{
    Hccl::HcclCommunicator* parentCommunicator = static_cast<Hccl::HcclCommunicator*>(comm_);
    CHK_PTR_NULL(parentCommunicator);

    if (derivationInfo.isGrow) {
        HCCL_INFO("[CollComm][CreateDerivedComm] Grow path, rankCount[%u].", mergedRankTable.rankCount);
        const HcclCommConfig* configPtr = &subConfig;
        ctx.drivedHcclComm = std::make_shared<Hccl::HcclCommunicator>(ctx.subParams, configPtr);
        CHK_RET(ctx.drivedHcclComm->Init(mergedRankTable));
    } else {
        HcclCommConfig configCopy = subConfig;
        CHK_RET(parentCommunicator->CreateSubComm(ctx.subParams, survivorRanks, ctx.drivedHcclComm, configCopy));
    }

    CHK_RET(ctx.drivedHcclComm->GetRankGraphV2(ctx.rankGraph));
    uintptr_t cclBufferAddr = 0;
    size_t cclBufferSize = 0;
    HcclMemType cclBufferMemType = HcclMemType::HCCL_MEM_TYPE_DEVICE;
    CHK_RET(ctx.drivedHcclComm->HcclGetCclBuffer(cclBufferAddr, cclBufferSize, cclBufferMemType));
    ctx.cclBuffer = HcclMem{cclBufferMemType, reinterpret_cast<void*>(cclBufferAddr), cclBufferSize};
    ctx.opExpansionMode = subConfig.hcclOpExpansionMode;
    HCCL_INFO(
        "[CollComm][%s][Grow] newComm cclBuffer addr[%p] cclBufferSize[%llu] opExpansionMode[%u]", __func__,
        reinterpret_cast<void*>(cclBufferAddr), cclBufferSize, ctx.opExpansionMode);
    return HCCL_SUCCESS;
}

HcclResult CollComm::SetupDerivedCollComm(
    std::unique_ptr<CollComm>& collComm, const DerivedCommParams& derivedParams, DerivedCommContext& ctx)
{
    // 从 derivedParams 中解引用，避免函数体内频繁 params.xxx
    const std::string& commName = derivedParams.commName;
    const uint32_t myNewRankId = derivedParams.rankId;
    const HcclCommConfig& subConfig = derivedParams.subConfig;
    const bool inheritParentConfig = derivedParams.inheritParentConfig;
    const DerivationInfo& derivationInfo = derivedParams.derivationInfo;
    const CollCommInitMode initMode = derivedParams.initMode;

    const uint32_t childGenIndex = generationIndex_ + 1;
    CHK_PRT_RET(
        generationIndex_ == UINT32_MAX,
        HCCL_ERROR("[%s] generationIndex overflow, current[%u].", __func__, generationIndex_), HCCL_E_PARA);

    // commId 已在 BuildDerivedCommParams 中设置（CreateDerivedComm 构造 HcclCommunicator 前固化）
    EXCEPTION_CATCH(
        collComm = std::make_unique<CollComm>(ctx.drivedHcclComm.get(), myNewRankId, commName, callbacks_, initMode),
        return HCCL_E_PTR);

    // 持有 HcclCommunicator 的 shared_ptr，防止 CreateDerivedComm 返回后 ctx.drivedHcclComm 释放导致 comm_ 悬垂
    collComm->hcclCommunicator_ = ctx.drivedHcclComm;

    // 需注意：哨兵值 HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET 表示用户未设置 deterministic
    int32_t processDeterministic = (subConfig.hcclDeterministic == HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET) ?
                                       static_cast<int32_t>(config_.GetConfigDeterministic()) :
                                       static_cast<int32_t>(subConfig.hcclDeterministic);
    if (inheritParentConfig) {
        collComm->GetCommConfig() = config_;
    } else {
        CHK_RET(ApplyHcclCommConfig(&subConfig, collComm->GetCommConfig(), ctx.opExpansionMode, processDeterministic));
    }
    collComm->hcclCommConfig_ = subConfig;

    // Init 需要 binHandle 但不回写 binHcclHandle_。
    // 原始建域路径由 hcclComm::InitBinHandle 加载并独占 binary 生命周期，CollComm::binHcclHandle_ 始终为 nullptr。
    aclrtBinHandle binHandle = nullptr;
    CHK_RET(LoadCclKernelBinHandle(binHandle));
    HCCL_INFO(
        "[CollComm][SetupDerivedCollComm] Init derived comm, binHandle[%p], opExpansionMode[%u].", binHandle,
        ctx.opExpansionMode);
    HcclResult initRet = collComm->Init(ctx.rankGraph, binHandle, ctx.cclBuffer, ctx.opExpansionMode);
    if (initRet != HCCL_SUCCESS) {
        aclError ret = aclrtBinaryUnLoad(binHandle);
        if (ret != ACL_SUCCESS) {
            HCCL_ERROR(
                "[CollComm][SetupDerivedCollComm] Init failed, initRet[%d], and aclrtBinaryUnLoad also failed, "
                "ret[%d].",
                initRet, ret);
        }
        return initRet;
    }
    collComm->derivedBinHandle_ = binHandle;

    // 派生身份字段（从 this 继承）
    collComm->topParentIdentifier_ = topParentIdentifier_;
    collComm->generationIndex_ = childGenIndex;
    collComm->rootHandle_ = derivationInfo.rootHandle;

    collComm->SetInitSource(GetInitSource());
    collComm->ownedCommunicator_ = ctx.drivedHcclComm;
    CollCommMgr::GetInstance().RegisterCollComm(collComm.get());
    return HCCL_SUCCESS;
}

HcclResult CollComm::CreateDerivedComm(std::unique_ptr<CollComm>& collComm, const DerivedCommParams& params)
{
    const auto& survivorRanks = params.survivorRanks;
    const uint32_t myNewRankId = params.rankId;
    const std::string& commName = params.commName;
    const HcclCommConfig& subConfig = params.subConfig;
    const DerivationInfo& derivationInfo = params.derivationInfo;
    const bool inheritParentConfig = params.inheritParentConfig;
    const CollCommInitMode initMode = params.initMode;
    const Hccl::RankTableInfo& mergedRankTable = params.mergedRankTable;

    uint32_t childGenIndex = generationIndex_ + 1;
    CHK_PRT_RET(
        generationIndex_ == UINT32_MAX,
        HCCL_ERROR("[CollComm][%s] generationIndex overflow, current[%u].", __func__, generationIndex_), HCCL_E_PARA);

    HCCL_INFO(
        "[CollComm][%s] commName[%s], myNewRankId[%u], genIndex[%u], topParentId[%s], "
        "survivorCount[%zu], inheritParent[%d].",
        __func__, commName.c_str(), myNewRankId, childGenIndex, GetTopParentIdentifier().c_str(), survivorRanks.size(),
        static_cast<int>(inheritParentConfig));

    CHK_PRT_RET(
        !derivationInfo.isGrow && survivorRanks.empty(), HCCL_ERROR("[CollComm][%s] survivorRanks is empty.", __func__),
        HCCL_E_PARA);

    DerivedCommContext ctx;
    CHK_RET(BuildDerivedCommParams(survivorRanks, myNewRankId, commName, derivationInfo, mergedRankTable, ctx));
    CHK_RET(CreateDerivedComm(survivorRanks, subConfig, derivationInfo, mergedRankTable, ctx));
    CHK_RET(SetupDerivedCollComm(collComm, params, ctx));

    HCCL_INFO(
        "[CollComm][%s] SUCCESS. commName[%s], myNewRankId[%u], rankSize[%u].", __func__, commName.c_str(), myNewRankId,
        ctx.subParams.rankSize);
    return HCCL_SUCCESS;
}

// 在业务逻辑层校验 Shrink 流程中 rankId 和 ranksCount 等参数的合法性
HcclResult CollComm::ValidateShrinkParams(const uint32_t* excludeRanksList, uint32_t excludeRanksCount)
{
    CHK_PRT_RET(
        excludeRanksCount == 0, HCCL_ERROR("[ValidateShrinkParams] excludeRanksCount must be greater than 0."),
        HCCL_E_PARA);
    CHK_PTR_NULL(excludeRanksList);

    uint32_t rankSize = GetRankSize();
    uint32_t myRank = GetMyRankId();

    // excludeRanksCount 不得大于等于 rankSize，否则全部 rank 被剔除，无法构成新通信域
    CHK_PRT_RET(
        excludeRanksCount >= rankSize,
        HCCL_ERROR(
            "[CommShrink][%s] excludeRanksCount[%u] >= rankSize[%u], cannot exclude all ranks.", __func__,
            excludeRanksCount, rankSize),
        HCCL_E_PARA);

    // 逐个校验 excludeRanksList：rankId 不得越界，且不得包含自身 myRank
    for (uint32_t i = 0; i < excludeRanksCount; ++i) {
        uint32_t r = excludeRanksList[i];
        CHK_PRT_RET(
            r >= rankSize,
            HCCL_ERROR("[CommShrink][%s] excludeRank[%u] out of range, rankSize[%u].", __func__, r, rankSize),
            HCCL_E_PARA);
        // 不允许剔除自身 rank，避免新通信域中没有自己，导致后续通信无法进行
        CHK_PRT_RET(
            r == myRank,
            HCCL_ERROR(
                "[CommShrink][%s] myRank[%u] is in excludeRanksList, self-exclusion forbidden.", __func__, myRank),
            HCCL_E_PARA);
    }

    return HCCL_SUCCESS;
}

HcclResult CollComm::ValidateGrowParams(uint32_t nRanks, uint32_t rank, const HcclRootInfo* rootInfo)
{
    CHK_PRT_RET(nRanks == 0, HCCL_ERROR("[ValidateGrowParams] nRanks must be greater than 0."), HCCL_E_PARA);
    CHK_PRT_RET(
        rank >= nRanks, HCCL_ERROR("[ValidateGrowParams][%s] rank[%u] >= nRanks[%u].", __func__, rank, nRanks),
        HCCL_E_PARA);
    // nRanks 必须大于当前 rankSize（扩容后总数 > 当前数），防止 newRankCount = nRanks - rankSize 无符号下溢
    CHK_PRT_RET(
        nRanks <= GetRankSize(),
        HCCL_ERROR(
            "[ValidateGrowParams][%s] nRanks[%u] must be greater than current rankSize[%u].", __func__, nRanks,
            GetRankSize()),
        HCCL_E_PARA);
    CHK_PRT_RET(
        GetInitSource() == CommInitSource::clusterInfo,
        HCCL_ERROR("[ValidateGrowParams][%s] clusterInfo-initialized communicator does not support Grow.", __func__),
        HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

// 从父域 rank 列表中过滤掉被剔除的 rank，构建存活 rank 列表，并计算自身在新域中的 rank 编号
HcclResult CollComm::BuildShrinkSurvivorRanks(
    const uint32_t* excludeRanksList, uint32_t excludeRanksCount, ShrinkSurvivorResult& result)
{
    uint32_t rankSize = GetRankSize();
    uint32_t myRank = GetMyRankId();
    uint32_t rootRankId = GetRootRankId();
    // 使用 endpoint 有效性判断“当前域是否有 root”，与 rootRankId 是否已解析解耦。
    const bool hasRoot = IsRootEndpointValid();
    HCCL_INFO("[%s] excludeRanksCount[%u], rankSize[%u], myRank[%u].", __func__, excludeRanksCount, rankSize, myRank);
    // root 存在时 rootRankId 必须已初始化且落在当前域内；root 被剔除后的派生域允许 root 不存在
    if (hasRoot) {
        CHK_PRT_RET(
            rootRankId == UINT32_MAX,
            HCCL_ERROR("[%s] rootRankId is not initialized while rootHandle is valid.", __func__), HCCL_E_INTERNAL);
        CHK_PRT_RET(
            rootRankId >= rankSize,
            HCCL_ERROR("[%s] rootRankId[%u] out of range, rankSize[%u].", __func__, rootRankId, rankSize),
            HCCL_E_INTERNAL);
    }
    // 转为 unordered_set，循环查找 O(1)
    std::unordered_set<uint32_t> excludeSet(excludeRanksList, excludeRanksList + excludeRanksCount);

    result.survivorRanks.clear();
    result.survivorRanks.reserve(rankSize - excludeRanksCount); // 预分配，避免 push_back 时多次扩容
    result.myNewRank = 0;
    result.rootExcluded = false;
    result.childRootRank = hasRoot ? rootRankId : UINT32_MAX;
    bool foundRootRank = false;

    // 遍历父域所有 rank，跳过被剔除的
    uint32_t newRankId = 0;
    for (uint32_t r = 0; r < rankSize; ++r) {
        if (excludeSet.count(r) > 0) {
            continue;
        }
        result.survivorRanks.push_back(r);
        // 记录自身在新域中的 rank 编号
        if (r == myRank) {
            result.myNewRank = newRankId;
        }
        // 记录 root 在新域中的 rank 编号
        if (hasRoot && r == rootRankId) {
            result.childRootRank = newRankId;
            foundRootRank = true;
        }
        ++newRankId;
    }

    // root 是否被剔除（在 excludeSet 中），存活则 childRootRank 已在循环中赋值
    result.rootExcluded = hasRoot && (excludeSet.count(rootRankId) > 0);

    HCCL_INFO(
        "[%s] survivorCount[%zu], myRank[%u] remapped to myNewRank[%u], hasRoot[%d], rootExcluded[%d], "
        "childRootRank[%u].",
        __func__, result.survivorRanks.size(), myRank, result.myNewRank, static_cast<int>(hasRoot), result.rootExcluded,
        result.childRootRank);
    return HCCL_SUCCESS;
}

HcclResult CollComm::ResolveRootRankIdForDerivation()
{
    // rootHandle 已清空表示当前派生域本来就没有 root，后续 Shrink/Grow 无需再解析。
    if (!IsRootEndpointValid()) {
        HCCL_INFO("[%s] current comm has no root, skip root rank resolution.", __func__);
        return HCCL_SUCCESS;
    }
    if (GetRootRankId() != UINT32_MAX) {
        CHK_PRT_RET(
            GetRootRankId() >= GetRankSize(),
            HCCL_ERROR("[%s] invalid root rankId[%u], rankSize[%u].", __func__, GetRootRankId(), GetRankSize()),
            HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }
    if (IsRoot()) {
        rootHandle_.rootRankId = GetMyRankId();
        HCCL_INFO("[%s] local rank is root, rootRankId[%u].", __func__, rootHandle_.rootRankId);
        return HCCL_SUCCESS;
    }

    u32 rootDevPhyId = UINT32_MAX;
    CHK_PRT_RET(
        !ParseRootDevPhyId(rootHandle_, rootDevPhyId),
        HCCL_ERROR("[%s] cannot parse devPhyId from rootHandle identifier[%s].", __func__, rootHandle_.identifier),
        HCCL_E_INTERNAL);

    // rankGraph 是通信域内的权威拓扑：V1 的 GraphRankInfo 与 V2 的 Peer 都携带 devicePhyId。
    std::vector<u32> graphCandidates;
    if (auto* rankGraphV1 = dynamic_cast<RankGraphV1*>(rankgraph_); rankGraphV1 != nullptr) {
        void* graph = nullptr;
        u32 graphLen = 0;
        if (rankGraphV1->GetRankGraphInfo(RANK_GRAPH_910_93, &graph, &graphLen) == HCCL_SUCCESS && graph != nullptr) {
            const auto* rankInfos = static_cast<const GraphRankInfo*>(graph);
            const u32 rankCount = graphLen / sizeof(GraphRankInfo);
            for (u32 i = 0; i < rankCount; ++i) {
                if (rankInfos[i].deviceInfo.devicePhyId == static_cast<s32>(rootDevPhyId)) {
                    graphCandidates.push_back(rankInfos[i].rankId);
                }
            }
        }
    } else if (auto* rankGraphV2 = dynamic_cast<RankGraphV2*>(rankgraph_); rankGraphV2 != nullptr) {
        u32 rankSize = 0;
        if (rankGraphV2->GetRankSize(&rankSize) == HCCL_SUCCESS) {
            for (u32 rankId = 0; rankId < rankSize; ++rankId) {
                u32 deviceId = UINT32_MAX;
                if (rankGraphV2->GetDeviceId(rankId, &deviceId) == HCCL_SUCCESS && deviceId == rootDevPhyId) {
                    graphCandidates.push_back(rankId);
                }
            }
        }
    }

    if (graphCandidates.size() == 1) {
        rootHandle_.rootRankId = graphCandidates[0];
        HCCL_INFO(
            "[%s] resolved root rankId[%u] from rankGraph, devPhyId[%u].", __func__, rootHandle_.rootRankId,
            rootDevPhyId);
        return HCCL_SUCCESS;
    }

    // 多节点 deviceId 会重复，rankGraph 无法唯一确定时用 rankTable 的 control plane 地址消歧。
    // RankTableInfo::NewRankInfo::deviceId 来自 ranktable 的 device_id 字段，本仓写入的是 devPhyId。
    std::vector<u32> tableCandidates;
    std::vector<u32> addrCandidates;
    auto detect = GetRankInfoDetect();
    if (detect != nullptr) {
        Hccl::RankTableInfo rankTable{};
        detect->GetRankTable(rankTable);
        const Hccl::IpAddress rootIp(std::string(rootHandle_.ip));
        for (const auto& rankInfo : rankTable.ranks) {
            if (rankInfo.deviceId != rootDevPhyId) {
                continue;
            }
            tableCandidates.push_back(rankInfo.rankId);
            if (!rootIp.IsInvalid() && !rankInfo.controlPlane.addr.IsInvalid()
                && rankInfo.controlPlane.addr == rootIp) {
                addrCandidates.push_back(rankInfo.rankId);
            }
        }
    }
    if (addrCandidates.size() == 1) {
        rootHandle_.rootRankId = addrCandidates[0];
    } else if (tableCandidates.size() == 1) {
        rootHandle_.rootRankId = tableCandidates[0];
    }

    CHK_PRT_RET(
        rootHandle_.rootRankId == UINT32_MAX || rootHandle_.rootRankId >= GetRankSize(),
        HCCL_ERROR(
            "[%s] cannot resolve root rankId, rootHandle[ip=%s, port=%u, identifier=%s], devPhyId[%u], "
            "rankGraphMatches[%zu], rankTableMatches[%zu], rankTableAddrMatches[%zu], rankSize[%u].",
            __func__, rootHandle_.ip, rootHandle_.listenPort, rootHandle_.identifier, rootDevPhyId,
            graphCandidates.size(), tableCandidates.size(), addrCandidates.size(), GetRankSize()),
        HCCL_E_INTERNAL);
    HCCL_INFO(
        "[%s] resolved root rankId[%u] from rankTable, devPhyId[%u].", __func__, rootHandle_.rootRankId, rootDevPhyId);
    return HCCL_SUCCESS;
}

HcclResult CollComm::CommShrink(
    const uint32_t* excludeRanksList, uint32_t excludeRanksCount, std::unique_ptr<CollComm>& newCollComm,
    const HcclCommConfig* config)
{
    HCCL_INFO("[%s] START. excludeRanksCount[%u].", __func__, excludeRanksCount);

    // detect 所有权已移交给链尾派生域，旧通信域只能由用户消费 ERROR 后重新触发新链路
    CHK_PRT_RET(
        !HasDetectOwnership(),
        HCCL_ERROR("[%s] detect ownership has been transferred to a derived communicator.", __func__),
        HCCL_E_NOT_SUPPORT);

    // clusterInfo 建域方式的通信域不支持 Shrink
    CHK_PRT_RET(
        GetInitSource() == CommInitSource::clusterInfo,
        HCCL_ERROR("[%s] clusterInfo-initialized communicator does not support Shrink.", __func__), HCCL_E_NOT_SUPPORT);

    // 防止 excludeRanksCount >= rankSize 导致 reserve 下溢、自身被剔除等越界风险。
    CHK_RET(ValidateShrinkParams(excludeRanksList, excludeRanksCount));
    // rootRankId 在 init 阶段可能不可知；Shrink 前从本端 rankGraph/rankTable 解析，不做 socket 交互。
    CHK_RET(ResolveRootRankIdForDerivation());

    // 两路径 config：config != nullptr 用用户 config（未设置字段走默认）；config == nullptr 继承父域全量 config
    HcclCommConfig subConfig;
    bool inheritParentConfig = false;
    if (config != nullptr) {
        subConfig = *config;
    } else {
        subConfig = this->GetHcclCommConfig();
        inheritParentConfig = true;
    }

    // 构建存活 rank 列表，计算自身在新域中的 rank 编号
    ShrinkSurvivorResult result;
    CHK_RET(BuildShrinkSurvivorRanks(excludeRanksList, excludeRanksCount, result));
    uint32_t newRankSize = static_cast<uint32_t>(result.survivorRanks.size());

    HCCL_INFO("[%s] rootExcluded[%d], childRootRank[%u].", __func__, result.rootExcluded, result.childRootRank);

    // 构造派生 identifier，校验长度不超限
    CHK_PRT_RET(
        GetGenerationIndex() == UINT32_MAX,
        HCCL_ERROR("[%s] generationIndex overflow, current[%u].", __func__, GetGenerationIndex()), HCCL_E_PARA);
    uint32_t childGenIndex = GetGenerationIndex() + 1;
    std::string identifierNew = GetTopParentIdentifier() + "_" + std::to_string(childGenIndex);
    CHK_PRT_RET(
        identifierNew.size() >= Hccl::ROOTINFO_IDENTIFIER_MAX_LENGTH,
        HCCL_ERROR(
            "[%s] identifier length[%zu] exceeds max[%u].", __func__, identifierNew.size(),
            Hccl::ROOTINFO_IDENTIFIER_MAX_LENGTH),
        HCCL_E_PARA);

    // 所有路径统一覆盖 commName 为派生 identifier
    errno_t sRet
        = memcpy_s(subConfig.hcclCommName, COMM_NAME_MAX_LENGTH, identifierNew.c_str(), identifierNew.size() + 1);
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[%s] memcpy_s commName failed.", __func__), HCCL_E_MEMORY);

    // 构建 DerivationInfo：root 被剔除则清空 rootHandle，否则继承
    DerivationInfo derivationInfo;
    derivationInfo.parentRankIds = result.survivorRanks;
    if (result.rootExcluded) {
        derivationInfo.rootHandle = {};
        HCCL_INFO("[%s] root excluded, rootHandle cleared.", __func__);
    } else {
        derivationInfo.rootHandle = GetCollCommRootHandle();
        derivationInfo.rootHandle.rootRankId = result.childRootRank;
        HCCL_INFO("[%s] root alive, rootHandle inherited, rootRankId[%u].", __func__, result.childRootRank);
    }

    // 创建派生通信域（CreateDerivedComm 内部已注册到 CollCommMgr）
    std::unique_ptr<CollComm> derivedComm;
    Hccl::RankTableInfo emptyRankTable{};
    DerivedCommParams derivedParams{
        result.survivorRanks, result.myNewRank,           identifierNew, subConfig, derivationInfo,
        inheritParentConfig,  CollCommInitMode::fullMode, emptyRankTable};
    CHK_RET(CreateDerivedComm(derivedComm, derivedParams));

    newCollComm = std::move(derivedComm);

    HCCL_INFO("[%s] SUCCESS. newRankSize[%u], myNewRank[%u].", __func__, newRankSize, result.myNewRank);
    return HCCL_SUCCESS;
}

HcclResult CollComm::HandleGrowRootReselect(
    uint32_t nRanks, uint32_t rank, const HcclRootInfo* rootInfo, const HcclCommConfig* config,
    Hccl::RankTableInfo& mergedRankTable)
{
    Hccl::HcclRootHandleV2 newRootHandle{};
    CHK_SAFETY_FUNC_RET(memcpy_s(&newRootHandle, sizeof(newRootHandle), rootInfo->internal, sizeof(newRootHandle)));

    // root 重选：协调者进程已通过 HcclCommGetRootInfo → SetPendingDetect 存入 CollComm
    auto existingDetect = std::static_pointer_cast<Hccl::RankInfoDetect>(TakePendingDetect());
    std::shared_ptr<Hccl::RankInfoDetect> reselectDetect;
    Hccl::HcclRootHandleV2 detectRootHandle = newRootHandle;
    // 重选需要 initial detect：协调者复用已启动 initial detect 的 server detect，并补自身 client 连接。
    if (existingDetect != nullptr) {
        // 协调者节点（root 节点）复用 server detect，更新 rootRankId 并补自身 client 连接
        reselectDetect = existingDetect;
        HCCL_INFO("[%s][RootReselect] coordinator, reuse server detect from bridge.", __func__);
        CHK_PRT_RET(
            !reselectDetect->IsReadyForGrow(),
            HCCL_ERROR(
                "[%s][RootReselect] detect service is not ready for grow, previous round has not completed.", __func__),
            HCCL_E_UNAVAIL);
        detectRootHandle = reselectDetect->GetRootHandleInfo();
        detectRootHandle.rootRankId = rank;
        reselectDetect->SetRootHandle(detectRootHandle);
        HCCL_INFO("[%s][RootReselect] coordinator root rankId[%u].", __func__, rank);
    } else {
        // 非 root 节点需要新建本地 detect，连接协调者的 server detect。
        EXCEPTION_CATCH(reselectDetect = std::make_shared<Hccl::RankInfoDetect>(), return HCCL_E_MEMORY);
        HCCL_INFO("[%s][RootReselect] non-coordinator, new local detect.", __func__);
    }

    // 所有 rank 都走同一次 initial detect：SetupAgent 建链并收广播，WaitComplete 等 initial detect 完成。
    HcclResult ret = reselectDetect->SetupAgent(nRanks, rank, detectRootHandle);
    if (ret != HCCL_SUCCESS) {
        EXCEPTION_CATCH(
            reselectDetect->Shutdown(), HCCL_ERROR("[%s] detect Shutdown after SetupAgent fail.", __func__));
        return ret;
    }
    ret = reselectDetect->WaitComplete(detectRootHandle.listenPort, Hccl::RANKINFO_DETECT_SERVER_STATUS_IDLE);
    if (ret != HCCL_SUCCESS) {
        EXCEPTION_CATCH(
            reselectDetect->Shutdown(), HCCL_ERROR("[%s] detect Shutdown after WaitComplete fail.", __func__));
        return ret;
    }
    reselectDetect->GetRankTable(mergedRankTable);
    const Hccl::HcclRootHandleV2& detectedRootHandle = reselectDetect->GetRootHandleInfo();
    CHK_PRT_RET(
        detectedRootHandle.listenPort == HCCL_INVALID_PORT || detectedRootHandle.rootRankId == UINT32_MAX
            || detectedRootHandle.rootRankId >= nRanks,
        HCCL_ERROR(
            "[%s][RootReselect] invalid rootHandle, rankId[%u], nRanks[%u].", __func__, detectedRootHandle.rootRankId,
            nRanks),
        HCCL_E_INTERNAL);
    newRootHandle = detectedRootHandle;
    SetRootHandle(newRootHandle);
    SetRankInfoDetect(reselectDetect);
    return HCCL_SUCCESS;
}

HcclResult CollComm::HandleGrowNormal(uint32_t nRanks, Hccl::RankTableInfo& mergedRankTable)
{
    auto detect = GetRankInfoDetect();
    CHK_PRT_RET(detect == nullptr, HCCL_ERROR("[%s] rankInfoDetect_ is null.", __func__), HCCL_E_INTERNAL);
    if (detect->IsRoot()) {
        uint32_t newRankCount = nRanks - GetRankSize();
        HCCL_INFO("[%s] root rank, trigger grow collect. newRankCount[%u].", __func__, newRankCount);
        CHK_RET(detect->TriggerGrowCollect(newRankCount, mergedRankTable));
    } else {
        HCCL_INFO("[%s] non-root rank, wait for next broadcast.", __func__);
        CHK_RET(detect->WaitForNextBroadcast(mergedRankTable, nRanks));
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::CommGrow(
    uint32_t nRanks, uint32_t rank, std::unique_ptr<CollComm>& newCollComm, const HcclRootInfo* rootInfo,
    const HcclCommConfig* config)
{
    const char* rootInfoDesc = rootInfo == nullptr ? "null(root-survives)" : "non-null(root-reselect)";
    HCCL_INFO(
        "[%s] START. nRanks[%u], rank[%u], myRank[%u], rootRankId[%u], rootInfo[%s].", __func__, nRanks, rank,
        GetMyRankId(), GetRootRankId(), rootInfoDesc);

    // detect 所有权已移交给链尾派生域，旧通信域只能由用户消费 ERROR 后重新触发新链路
    CHK_PRT_RET(
        !HasDetectOwnership(),
        HCCL_ERROR("[%s] detect ownership has been transferred to a derived communicator.", __func__),
        HCCL_E_NOT_SUPPORT);

    // 入参校验
    CHK_RET(ValidateGrowParams(nRanks, rank, rootInfo));

    Hccl::RankTableInfo mergedRankTable{};
    // HcclCommGetRootInfo 创建的新 root server 已暂存在 pendingDetect_ 中，优先按 root 重选处理
    bool rootReselect = HasPendingDetect();
    if (!rootReselect && IsRootEndpointValid() && rootInfo != nullptr) {
        // root 存活：比对 rootHandle_ 与传入 rootInfo 的 listenPort / identifier 是否一致
        Hccl::HcclRootHandleV2 inputRootHandle{};
        CHK_SAFETY_FUNC_RET(
            memcpy_s(&inputRootHandle, sizeof(inputRootHandle), rootInfo->internal, sizeof(inputRootHandle)));
        const Hccl::HcclRootHandleV2& existingHandle = GetCollCommRootHandle();
        if (inputRootHandle.listenPort != existingHandle.listenPort
            || strncmp(inputRootHandle.identifier, existingHandle.identifier, Hccl::ROOTINFO_IDENTIFIER_MAX_LENGTH)
                   != 0) {
            HCCL_WARNING(
                "[%s] rootInfo mismatch with existing rootHandle. "
                "existing port[%u] identifier[%s], input port[%u] identifier[%s]. "
                "Switching to root-reselect path.",
                __func__, existingHandle.listenPort, existingHandle.identifier, inputRootHandle.listenPort,
                inputRootHandle.identifier);
            rootReselect = true;
        }
    } else if (!rootReselect && !IsRootEndpointValid()) {
        // rootHandle 无效（Shrink 剔除或首父未初始化），必须走 root 重选
        rootReselect = true;
    }
    if (rootReselect) {
        CHK_PRT_RET(
            rootInfo == nullptr, HCCL_ERROR("[%s] root reselect requires non-null rootInfo.", __func__), HCCL_E_PARA);
        CHK_PRT_RET(
            rank != GetMyRankId(),
            HCCL_ERROR("[%s] root reselect rank[%u] must equal current rank[%u].", __func__, rank, GetMyRankId()),
            HCCL_E_PARA);
        CHK_RET(HandleGrowRootReselect(nRanks, rank, rootInfo, config, mergedRankTable));
    } else {
        CHK_RET(ResolveRootRankIdForDerivation());
        CHK_RET(HandleGrowNormal(nRanks, mergedRankTable));
    }

    uint32_t newRankSize = mergedRankTable.rankCount;
    HCCL_INFO("[%s] merged rankCount[%u], building full rank graph.", __func__, newRankSize);

    // 校验收集到的 rankCount 与调用方 nRanks 一致，防止拓扑收集不完整
    CHK_PRT_RET(
        newRankSize != nRanks,
        HCCL_ERROR(
            "[%s] merged rankCount[%u] != nRanks[%u], topology collection incomplete.", __func__, newRankSize, nRanks),
        HCCL_E_INTERNAL);

    // 构造派生 identifier
    CHK_PRT_RET(
        GetGenerationIndex() == UINT32_MAX,
        HCCL_ERROR("[%s] generationIndex overflow, current[%u].", __func__, GetGenerationIndex()), HCCL_E_PARA);
    uint32_t childGenIndex = GetGenerationIndex() + 1;
    std::string identifierNew = GetTopParentIdentifier() + "_" + std::to_string(childGenIndex);

    CHK_PRT_RET(
        identifierNew.size() >= Hccl::ROOTINFO_IDENTIFIER_MAX_LENGTH,
        HCCL_ERROR(
            "[%s] identifier length[%zu] exceeds max[%u].", __func__, identifierNew.size(),
            Hccl::ROOTINFO_IDENTIFIER_MAX_LENGTH),
        HCCL_E_PARA);

    // 两路径 config（与 Shrink 一致）
    HcclCommConfig subConfig;
    bool inheritParentConfig = false;
    if (config != nullptr) {
        subConfig = *config;
    } else {
        subConfig = this->GetHcclCommConfig();
        inheritParentConfig = true;
    }
    CHK_SAFETY_FUNC_RET(
        memcpy_s(subConfig.hcclCommName, COMM_NAME_MAX_LENGTH, identifierNew.c_str(), identifierNew.size() + 1));

    DerivationInfo derivationInfo;
    derivationInfo.isGrow = true;
    derivationInfo.rootHandle = GetCollCommRootHandle();
    CHK_PRT_RET(
        derivationInfo.rootHandle.listenPort == HCCL_INVALID_PORT || derivationInfo.rootHandle.rootRankId == UINT32_MAX
            || derivationInfo.rootHandle.rootRankId >= nRanks,
        HCCL_ERROR(
            "[%s] invalid rootHandle, rankId[%u], nRanks[%u].", __func__, derivationInfo.rootHandle.rootRankId, nRanks),
        HCCL_E_INTERNAL);

    // 存活 rank 在新域中保持原编号（0 ~ parentRankSize-1），新 rank 追加编号
    uint32_t myNewRank = GetMyRankId();

    // Grow 不是父域子集，survivorRanks 传空；mergedRankTable 传拓扑收集结果
    static const std::vector<uint32_t> emptySurvivorRanks;
    std::unique_ptr<CollComm> derivedComm;
    DerivedCommParams params{
        emptySurvivorRanks,         myNewRank,      identifierNew, subConfig, derivationInfo, inheritParentConfig,
        CollCommInitMode::fullMode, mergedRankTable};
    CHK_RET(CreateDerivedComm(derivedComm, params));

    newCollComm = std::move(derivedComm);

    HCCL_INFO("[%s] SUCCESS. newRankSize[%u], myNewRank[%u].", __func__, newRankSize, myNewRank);
    return HCCL_SUCCESS;
}

HcclResult CollComm::DetectRankTableForNewRank(
    uint32_t nRanks, uint32_t rank, const HcclRootInfo* rootInfo, GrowNewRankDetectResult& result)
{
    CHK_PTR_NULL(rootInfo);
    CHK_PRT_RET(rank >= nRanks, HCCL_ERROR("[%s] rank[%u] >= nRanks[%u].", __func__, rank, nRanks), HCCL_E_PARA);

    CHK_SAFETY_FUNC_RET(memcpy_s(
        &result.newRootHandle, sizeof(result.newRootHandle), rootInfo->internal, sizeof(result.newRootHandle)));

    // 新 rank 进程本地创建 detect client，后续由 newCollComm::rankInfoDetect_ 持有；
    // root 端 server 位于协调者进程，shared_ptr 不能也不应跨进程保活。
    EXCEPTION_CATCH(result.reselectDetect = std::make_shared<Hccl::RankInfoDetect>(), return HCCL_E_MEMORY);
    HcclResult agentRet = result.reselectDetect->SetupAgent(nRanks, rank, result.newRootHandle);
    if (agentRet != HCCL_SUCCESS) {
        // SetupAgent 失败：detect 可能持有 client socket/host handle/RA 引用（R-07 泄漏场景 5）
        EXCEPTION_CATCH(
            result.reselectDetect->Shutdown(), HCCL_ERROR("[%s] detect Shutdown after SetupAgent fail.", __func__));
        return agentRet;
    }
    HcclResult ret = result.reselectDetect->WaitComplete(
        result.newRootHandle.listenPort, Hccl::RANKINFO_DETECT_SERVER_STATUS_IDLE);
    if (ret != HCCL_SUCCESS) {
        EXCEPTION_CATCH(
            result.reselectDetect->Shutdown(), HCCL_ERROR("[%s] detect Shutdown after WaitComplete fail.", __func__));
        return ret;
    }
    result.newRootHandle = result.reselectDetect->GetRootHandleInfo();
    result.reselectDetect->GetRankTable(result.mergedRankTable);

    HCCL_INFO("[%s] initial full-collection complete, rankCount[%u].", __func__, result.mergedRankTable.rankCount);
    return HCCL_SUCCESS;
}

std::string CollComm::BuildIdentifierFromRootHandle(const Hccl::HcclRootHandleV2& rootHandle)
{
    if (rootHandle.valid == 1 && rootHandle.topParentIdentifier[0] != '\0') {
        return std::string(rootHandle.topParentIdentifier) + "_" + std::to_string(rootHandle.generationIndex);
    }
    return std::string(rootHandle.identifier);
}

// ── CommGrowNewRank 内部 helper：首父建域，创建 HcclCommunicator 并提取 rankGraph / cclBuffer ──
HcclResult CollComm::CreateCommunicatorForNewRank(
    uint32_t nRanks, uint32_t rank, const GrowNewRankDetectResult& detectResult, const HcclCommConfig* config,
    GrowNewRankCommResult& result)
{
    Hccl::DevType devType = Hccl::HrtGetDeviceType();
    std::string identifierNew = BuildIdentifierFromRootHandle(detectResult.newRootHandle);
    CHK_PRT_RET(
        identifierNew.size() >= Hccl::ROOTINFO_IDENTIFIER_MAX_LENGTH,
        HCCL_ERROR(
            "[%s] identifier length[%zu] exceeds max[%u].", __func__, identifierNew.size(),
            Hccl::ROOTINFO_IDENTIFIER_MAX_LENGTH),
        HCCL_E_PARA);
    Hccl::CommParams params;
    params.commId = identifierNew;
    params.myRank = rank;
    params.rankSize = nRanks;
    params.rankInParentComm = rank;
    params.devType = devType;
    params.devUsed = true;
    params.isWorldGroup = true;
    params.commDepth = 0;

    if (config != nullptr) {
        result.subConfig = *config;
    } else {
        InitDefaultHcclCommConfig(result.subConfig);
    }
    CHK_SAFETY_FUNC_RET(
        memcpy_s(result.subConfig.hcclCommName, COMM_NAME_MAX_LENGTH, identifierNew.c_str(), identifierNew.size() + 1));

    const HcclCommConfig* configPtr = &result.subConfig;
    result.drivedHcclComm = std::make_shared<Hccl::HcclCommunicator>(params, configPtr);
    CHK_RET(result.drivedHcclComm->Init(detectResult.mergedRankTable));

    CHK_RET(result.drivedHcclComm->GetRankGraphV2(result.rankGraph));
    uintptr_t cclBufferAddr = 0;
    size_t cclBufferSize = 0;
    HcclMemType cclBufferMemType = HcclMemType::HCCL_MEM_TYPE_DEVICE;
    CHK_RET(result.drivedHcclComm->HcclGetCclBuffer(cclBufferAddr, cclBufferSize, cclBufferMemType));
    result.cclBuffer = HcclMem{cclBufferMemType, reinterpret_cast<void*>(cclBufferAddr), cclBufferSize};

    result.opExpansionMode = result.subConfig.hcclOpExpansionMode;
    return HCCL_SUCCESS;
}

HcclResult CollComm::CommGrowNewRank(
    uint32_t nRanks, uint32_t rank, const HcclRootInfo* rootInfo, const HcclCommConfig* config,
    std::unique_ptr<CollComm>& newCollComm)
{
    HCCL_INFO("[%s] START. nRanks[%u], rank[%u], rootInfo[non-null(new-rank-join)].", __func__, nRanks, rank);
    // 1. 拓扑探测 + 全量收集
    GrowNewRankDetectResult detectResult;
    CHK_RET(DetectRankTableForNewRank(nRanks, rank, rootInfo, detectResult));

    // 2. 创建 HcclCommunicator，提取 rankGraph / cclBuffer
    GrowNewRankCommResult commResult;
    CHK_RET(CreateCommunicatorForNewRank(nRanks, rank, detectResult, config, commResult));

    // 3. 组装 CollComm
    std::string identifierNew = BuildIdentifierFromRootHandle(detectResult.newRootHandle);
    ManagerCallbacks callbacks;
    callbacks.getAicpuCommState = []() {
        return false;
    };
    callbacks.setAicpuCommState = [](bool) {};
    callbacks.kernelLaunchAicpuCommInit = []() {
        return HCCL_SUCCESS;
    };
    callbacks.reportProfilingKernel = [](uint64_t, std::string) {
        return HCCL_SUCCESS;
    };
    EXCEPTION_CATCH(
        newCollComm = std::make_unique<CollComm>(
            commResult.drivedHcclComm.get(), rank, identifierNew, callbacks, CollCommInitMode::fullMode),
        return HCCL_E_PTR);

    // 未设置 deterministic。新 rank 无父域可继承，与首父建域一致：哨兵值时从环境变量获取 fallback
    HcclConfigValue processDeterministicConfig{};
    if (commResult.subConfig.hcclDeterministic == HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET) {
        CHK_RET(HcclGetConfig(HCCL_DETERMINISTIC, &processDeterministicConfig));
    }
    int32_t processDeterministic = processDeterministicConfig.value;
    CHK_RET(ApplyHcclCommConfig(
        &commResult.subConfig, newCollComm->GetCommConfig(), commResult.opExpansionMode, processDeterministic));
    newCollComm->hcclCommConfig_ = commResult.subConfig;

    // 新 rank 无父域，加载 ccl_kernel 二进制供 Init 使用。
    // 不缓存到 binHcclHandle_，binary 生命周期由 hcclComm::InitBinHandle/BinaryUnLoad 管理
    aclrtBinHandle binHandle = nullptr;
    CHK_RET(LoadCclKernelBinHandle(binHandle));
    HcclResult initRet
        = newCollComm->Init(commResult.rankGraph, binHandle, commResult.cclBuffer, commResult.opExpansionMode);
    if (initRet != HCCL_SUCCESS) {
        aclError ret = aclrtBinaryUnLoad(binHandle);
        if (ret != ACL_SUCCESS) {
            HCCL_ERROR("[__func__] aclrtBinaryUnLoad is unsuccessful, ret[%d]", ret);
        }
        return initRet;
    }
    newCollComm->derivedBinHandle_ = binHandle;

    const Hccl::HcclRootHandleV2& detectedRootHandle = detectResult.reselectDetect->GetRootHandleInfo();
    CHK_PRT_RET(
        detectedRootHandle.listenPort == HCCL_INVALID_PORT || detectedRootHandle.rootRankId == UINT32_MAX
            || detectedRootHandle.rootRankId >= nRanks,
        HCCL_ERROR("[%s] invalid rootHandle, rankId[%u], nRanks[%u].", __func__, detectedRootHandle.rootRankId, nRanks),
        HCCL_E_INTERNAL);
    newCollComm->SetRootHandle(detectedRootHandle);
    // 派生身份初始化：从 rootHandle 的谱系字段回填 topParentIdentifier_/generationIndex_，
    CHK_RET(newCollComm->InitDerivationFields());
    newCollComm->SetRankInfoDetect(detectResult.reselectDetect);
    HCCL_INFO("[%s] root rankId resolved[%u] from rootHandle broadcast.", __func__, detectedRootHandle.rootRankId);
    newCollComm->SetInitSource(CommInitSource::rootInfo);

    // ownedCommunicator_ 持有 drivedHcclComm
    newCollComm->ownedCommunicator_ = commResult.drivedHcclComm;
    newCollComm->hcclCommunicator_ = commResult.drivedHcclComm;

    CollCommMgr::GetInstance().RegisterCollComm(newCollComm.get());

    HCCL_INFO("[%s] SUCCESS. nRanks[%u], rank[%u].", __func__, nRanks, rank);
    return HCCL_SUCCESS;
}
} // namespace hccl
