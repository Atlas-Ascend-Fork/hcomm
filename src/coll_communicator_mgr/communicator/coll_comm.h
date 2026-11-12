/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_COMM_H
#define COLL_COMM_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "my_rank.h"
#include "rank_graph.h"
#include "comm_config_pub.h"
#include "comm_engine_res_manager.h"
#include "independent_op_context_manager.h"
#include "comm_mem_manager.h"
#include "channel_manager.h"
#include "hcclCommDfx.h"
#include "rank_graph_v2.h"
#include "error_message_v2.h"
#include "manager_common.h"
#include "group_schedule_mgr.h"
#include "op_group_queue_mgr.h"
#include "include/hccl_communicator.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_types.h"
#include "symmetric_memory/symmetric_memory.h"
#include "hcomm_team_entity_defs.h"
#include "../rank_info_detect/root_handle_v2.h"
#include "rank_table_info.h"
namespace Hccl {
class DevBuffer;
class RankInfoDetect;
} // namespace Hccl
namespace hccl {
class SymmetricMemory;
class UbMemSymmetricMemory;

// 记录A5 FullMode统一HcommWindow所属通信域，用于解注册时定位CollComm。
HcclResult RecordHcommWindowOwner(HcclCommSymWindow winHandle, HcclComm comm);
HcclResult GetHcommWindowComm(HcclCommSymWindow winHandle, HcclComm& comm);
void EraseHcommWindowOwner(HcclCommSymWindow winHandle);

struct SymmetricMemoryResource;
struct SymmetricMemoryDeleter {
    void operator()(SymmetricMemory* ptr) const;
};
/**
 * @note 职责：集合通信通信域上下文管理，包括RankGraph和本rank信息资源等内容。
 * 当前需包含原有的91092/91093的通信域、原有的91095的通信域void
 * *指针、及新独立算子架构的通信域（支持91092/91093/91095...）。
 */
enum class CollCommInitMode {
    fullMode, // 全功能模式：给A5及后续新架构使用，完整的CollComm初始化和资源管理
    simpleMode // 简化模式：给A2/A3老芯片使用，由于架构限制，仅将RankGraph、MyRank等放入CollComm管理
};

/**
 * @brief 通信域初始化来源。clusterInfo 建域的通信域不支持 Shrink。
 */
enum class CommInitSource {
    rootInfo,   // 通过 HcclCommInitRootInfo 建域
    clusterInfo // 通过 HcclCommInitClusterInfo 建域
};

class CollComm {
public:
    CollComm(
        void* comm, uint32_t rankId, const std::string& commName, const ManagerCallbacks& callbacks,
        CollCommInitMode initMode = CollCommInitMode::fullMode);
    ~CollComm();

    struct DerivationInfo {
        Hccl::HcclRootHandleV2 rootHandle{}; // root 身份，Grow 继承；Shrink root 被剔除时清空
        std::vector<uint32_t> parentRankIds; // 存活父域 rankId，用于日志/DFX
        bool isGrow{false};                  // true=Grow 派生；false=Shrink 派生
    };

    /// 派生通信域创建参数，聚合 survivorRanks/rankId/commName 等入参，避免函数签名过长
    struct DerivedCommParams {
        std::vector<uint32_t> survivorRanks;                   // 存活 rank 列表（升序）
        uint32_t rankId{};                                     // 本 rank 在新域中的编号
        std::string commName;                                  // 派生通信域名称
        HcclCommConfig subConfig{};                            // 派生域配置
        DerivationInfo derivationInfo;                         // 派生身份信息
        bool inheritParentConfig{false};                       // true 时直接继承父域全量 config
        CollCommInitMode initMode{CollCommInitMode::fullMode}; // 初始化模式
        Hccl::RankTableInfo mergedRankTable{};
    };

    struct ShrinkSurvivorResult {
        std::vector<uint32_t> survivorRanks; // 存活 rank 列表（升序）
        uint32_t myNewRank{UINT32_MAX};      // 本 rank 在新域中的编号
        bool rootExcluded{false};            // root rank 是否被剔除
        uint32_t childRootRank{UINT32_MAX};  // root 在新域中的编号（root 存活时有效）
    };

    // 初始化通信域
    HcclResult Init(void* rankGraph, aclrtBinHandle binHandle, HcclMem cclBuffer, uint32_t opExpansionMode = 0);

    inline CommConfig& GetCommConfig() { return config_; }
    inline const HcclCommConfig& GetHcclCommConfig() const { return hcclCommConfig_; }
    inline void SetHcclCommConfig(const HcclCommConfig& config) { hcclCommConfig_ = config; }
    inline RankGraph* GetRankGraph() { return rankgraph_; }
    inline CommEngineResMgr* GetCommEngineResMgr() { return commEngineResMgr_.get(); }
    // 经 CommEngineResMgr 取线程管理器；未初始化时返回 nullptr（HostDpu 收编链的宿主侧入口）
    inline ThreadMgr* GetThreadMgr()
    {
        return commEngineResMgr_ == nullptr ? nullptr : commEngineResMgr_->GetThreadMgr();
    }
    inline ContextManager* GetContextManager() { return contextMgr_.get(); }
    inline CommMemMgr* GetCommMemMgr() { return commMemMgr_.get(); }
    inline ChannelManager* GetChannelManager() { return channelMgr_.get(); }
    void* GetCommunicatorV2() { return comm_; }

    // 派生通信域（Shrink/Grow）持有 HcclCommunicator 的 shared_ptr，用于注册到 hcclGroupMap
    // 父域返回 nullptr
    std::shared_ptr<Hccl::HcclCommunicator> GetHcclCommunicatorSharedPtr() const { return hcclCommunicator_; }

    // 显式释放 rankInfoDetect 资源（Shutdown persistent server + client socket）
    HcclResult ShutdownRankInfoDetect();

    // 获取MyRank
    MyRank* GetMyRank() const { return myRank_.get(); }

    // 获取Rank ID
    uint32_t GetMyRankId() const;

    // -- 派生身份与 root 身份管理 --
    const std::string& GetTopParentIdentifier() const { return topParentIdentifier_; }
    void SetTopParentIdentifier(const std::string& id) { topParentIdentifier_ = id; }
    HcclResult InitDerivationFields();
    uint32_t GetGenerationIndex() const { return generationIndex_; }
    void SetGenerationIndex(uint32_t idx) { generationIndex_ = idx; }
    uint32_t GetRootRankId() const { return rootHandle_.rootRankId; }
    const Hccl::HcclRootHandleV2& GetCollCommRootHandle() const { return rootHandle_; }
    void SetRootHandle(const Hccl::HcclRootHandleV2& handle);
    // 获取 Grow 使用的 rootInfo；root endpoint 无效时可能触发 root 重选并创建新 server。
    HcclResult GetCollCommRootInfo(Hccl::HcclRootHandleV2& rootInfo);
    bool IsRootHandleValid() const
    {
        return rootHandle_.listenPort != HCCL_INVALID_PORT && rootHandle_.rootRankId != UINT32_MAX;
    }
    bool IsRootEndpointValid() const
    {
        return rootHandle_.listenPort != HCCL_INVALID_PORT && rootHandle_.identifier[0] != '\0';
    }

    // 获取 ManagerCallbacks
    const ManagerCallbacks& GetCallbacks() const { return callbacks_; }
    void SetCallbacks(const ManagerCallbacks& callbacks) { callbacks_ = callbacks; }

    // 检查是否为 rootInfo 初始化的通信域
    CommInitSource GetInitSource() const { return initSource_; }
    void SetInitSource(CommInitSource src) { initSource_ = src; }

    // 常驻 RankInfoDetect 所有权链：只有链尾派生通信域持有
    std::shared_ptr<Hccl::RankInfoDetect> GetRankInfoDetect() const { return rankInfoDetect_; }
    void SetRankInfoDetect(std::shared_ptr<Hccl::RankInfoDetect> detect);
    // 派生成功后把 detect 移交给新通信域；旧通信域永久失去 Shrink/Grow 能力
    std::shared_ptr<Hccl::RankInfoDetect> TakeRankInfoDetect()
    {
        hasDetectOwnership_ = false;
        return std::move(rankInfoDetect_);
    }
    // 派生域所有后置注册成功后，再提交 detect 所有权转移
    void TransferRankInfoDetect(CollComm& target);
    bool HasDetectOwnership() const { return hasDetectOwnership_; }

    // Grow root 重选时，HcclCommGetRootInfo 创建的 server detect 暂存于此，
    void SetPendingDetect(std::shared_ptr<void> detect) { pendingDetect_ = std::move(detect); }
    std::shared_ptr<void> TakePendingDetect() { return std::move(pendingDetect_); }
    bool HasPendingDetect() const { return pendingDetect_ != nullptr; }
    HcclResult TakeRankInfoDetectFromCommunicator();
    bool IsRoot() const;

    // 加载 ccl_kernel 二进制到 outBinHandle，不缓存到 binHcclHandle_。
    static HcclResult LoadCclKernelBinHandle(aclrtBinHandle& outBinHandle);

    // 派生通信域统一工厂入口，shrink 和 grow 都会调用
    HcclResult CreateDerivedComm(std::unique_ptr<CollComm>& collComm, const DerivedCommParams& params);

    // -- 通信域缩容/扩容 --
    HcclResult CommShrink(
        const uint32_t* excludeRanksList, uint32_t excludeRanksCount, std::unique_ptr<CollComm>& newCollComm,
        const HcclCommConfig* config = nullptr);
    HcclResult CommGrow(
        uint32_t nRanks, uint32_t rank, std::unique_ptr<CollComm>& newCollComm, const HcclRootInfo* rootInfo,
        const HcclCommConfig* config = nullptr);

    // CommGrowNewRank 内部 helper：新 rank 进程本地创建 detect client，收集拓扑表并返回
    static HcclResult CommGrowNewRank(
        uint32_t nRanks, uint32_t rank, const HcclRootInfo* rootInfo, const HcclCommConfig* config,
        std::unique_ptr<CollComm>& newCollComm);

    HcclResult ValidateShrinkParams(const uint32_t* excludeRanksList, uint32_t excludeRanksCount);
    HcclResult ValidateGrowParams(uint32_t nRanks, uint32_t rank, const HcclRootInfo* rootInfo);

    // CommGrowNewRank 内部 helper：新 rank 拓扑探测 + 全量收集
    struct GrowNewRankDetectResult {
        Hccl::HcclRootHandleV2 newRootHandle{};
        std::shared_ptr<Hccl::RankInfoDetect> reselectDetect;
        Hccl::RankTableInfo mergedRankTable{};
    };
    static HcclResult DetectRankTableForNewRank(
        uint32_t nRanks, uint32_t rank, const HcclRootInfo* rootInfo, GrowNewRankDetectResult& result);

    // CommGrowNewRank 内部 helper：首父建域，创建 HcclCommunicator 并提取 rankGraph / cclBuffer
    struct GrowNewRankCommResult {
        std::shared_ptr<Hccl::HcclCommunicator> drivedHcclComm;
        void* rankGraph{nullptr};
        HcclMem cclBuffer{};
        HcclCommConfig subConfig{};
        uint32_t opExpansionMode{0};
    };
    static HcclResult CreateCommunicatorForNewRank(
        uint32_t nRanks, uint32_t rank, const GrowNewRankDetectResult& detectResult, const HcclCommConfig* config,
        GrowNewRankCommResult& result);

    // 获取devicelogicId
    s32 GetDeviceLogicId() const { return deviceLogicId_; }

    // 是否为全功能模式（fullMode），供外部 owner 在注册/注销前判断是否需要管理
    bool IsFullMode() const { return initMode_ == CollCommInitMode::fullMode; }

    // 获取Rank数量
    uint32_t GetRankSize() const
    {
        if (rankgraph_ == nullptr) {
            HCCL_ERROR("[CollComm]get ranksize failed");
            return 0;
        }
        uint32_t rankSize{0};
        HcclResult ret = rankgraph_->GetRankSize(&rankSize);
        if (ret != 0) {
            HCCL_ERROR("[CollComm]get ranksize failed");
            return 0;
        }
        return rankSize;
    }

    // 获取HcclCommDfx
    HcclCommDfx* GetHcclCommDfx() { return hcclCommDfx_.get(); }
    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> GetDfxCallback()
    {
        if (hcclCommDfx_ == nullptr) {
            HCCL_ERROR("[CollComm]CollComm DfxCallBack failed. hcclCommDfx is nullptr");
            return nullptr;
        }
        return hcclCommDfx_->GetCallback();
    }
    const std::string& GetCommId() const { return commId_; }
    HcclResult
    GetHDCommunicate(HDCommunicateParams& kfcControlTransferH2DParams, HDCommunicateParams& kfcStatusTransferD2HParams);
    Hccl::ErrorMessageReport GetAicpuTaskException();
    HcclResult GetParentRankId(u32& parentRankId) const;
    uint32_t UpdateIndex();

    HcclCommStatus GetCommStatus() const;
    HcclResult Suspend();
    HcclResult Clean();
    HcclResult Resume();
    HcclResult RegisterWindow(HcclComm comm, void* ptr, size_t size, HcclCommSymWindow* winHandle);
    HcclResult DeregisterWindow(HcclCommSymWindow winHandle);
    HcclResult GetCommSymWin(void* ptr, size_t size, HcclCommSymWindow* winHandle, size_t* offset);
    HcclResult RegisterPendingSymmetricMemHandles();
    // 获取本地全部已完成注册的对称内存句柄。
    HcclResult GetAllRegisteredSymMemHandles(std::vector<HcclMemHandle>& memHandles) const;
    HcclResult GetRemoteMissingSymMemHandles(
        const std::vector<std::string>& remoteMemTags, std::vector<HcclMemHandle>& memHandles) const;
    HcclResult
    UpdateSymmetricRemoteMem(uint32_t remoteRank, const CommMem* remoteMems, const std::vector<std::string>& memTags);
    HcclResult
    UpdateHcommWindowRemoteMem(uint32_t remoteRank, const CommMem* remoteMems, const std::vector<std::string>& memTags);
    HcclResult GetHcclBinHandle(aclrtBinHandle& binHcclHandle, const std::string& soName);
    std::shared_ptr<class GroupScheduleMgr> groupScheduleMgr{nullptr}; // for group
    OpGroupQueueMgr groupQueueMgr{};                                   // for group enqueue

private:
    using ProtocolRankMap = std::unordered_map<CommProtocol, std::vector<uint32_t>>;

    static std::string BuildIdentifierFromRootHandle(const Hccl::HcclRootHandleV2& rootHandle);

    struct UbWorldTeamCandidate {
        uint32_t netLayer{0};
        uint32_t selfMemberId{0};
        std::vector<uint32_t> ranks;
    };

    HcclResult DestroyAicpuComm();
    HcclResult InitHDCommunicate();
    HcclResult InitTaskExceptionHandler();
    HcclResult InitKfcAndRegisterCollComm();
    void PrintCommKeyInfo(uint32_t rankNum) const;
    HcclResult GetRankIpPortMap();
    HcclResult InitSymmetricMemory();
    HcclResult PrepareSharedSymmetricWindow(void* ptr, size_t size, void*& devLegacySymWin, HcclCommSymWindow& devWin);
    // 维护HcommWindow与URMA底层Window的双向索引。
    HcclResult RegisterHcommWindowMapping(HcclCommSymWindow devWin, void* devLegacySymWin);
    HcclResult FindLegacySymmetricWindow(HcclCommSymWindow devWin, void*& devLegacySymWin);
    HcclResult UnregisterHcommWindowMapping(HcclCommSymWindow devWin, void*& devLegacySymWin);
    void RemoveHcommWindow(HcclCommSymWindow devWin);
    HcclResult RegisterSymmetricMemoryResource(void* ptr, size_t size, SymmetricMemoryResource& resource);
    void UnregisterSymmetricMemoryResource(const SymmetricMemoryResource& resource);

    // 通信域初始化时按 protocol + netLayer 创建 A5 URMA/UB Memory 预制 worldTeam。
    HcclResult InitWorldTeams();
    HcclResult InitWorldTeamLayer(uint32_t netLayer, uint32_t selfRankId, UbWorldTeamCandidate& ubCandidate);
    // 按协议收集本 Rank 在指定 NetLayer 内的可达 Rank，不在收集阶段创建 WorldTeam。
    void CollectLayerReachableRanks(
        uint32_t netLayer, const uint32_t* ranks, uint32_t rankNum, ProtocolRankMap& reachableRanksByProtocol) const;
    // 使用各 URMA 协议自己的可达成员集合预制 WorldTeam。
    HcclResult
    CreateUrmaWorldTeams(uint32_t netLayer, uint32_t selfRankId, const ProtocolRankMap& reachableRanksByProtocol);
    // 创建并注册预制worldTeam（worldTeam不通信，不创建syncMem）
    HcclResult CreatePrebuiltWorldTeam(
        CommProtocol protocol, uint32_t netLayer, const uint32_t* ranks, uint32_t rankNum, uint32_t selfMemberId);
    // window后注册补交换：对该通信域下所有已建链Team重新调HcclChannelAcquire，把新window的memHandle
    // 带入交换并回填（syncMem已交换不重复）。无已建链Team（常规时序window先于Team）为空操作。
    HcclResult ReExchangeWindowsForBoundTeams();
    // 单team补交换：组装channelDesc建链并回填（symm memHandle挂desc参与交换）
    HcclResult ReExchangeChannelsForTeam(
        HcommTeamHandle team, CommEngine engine, uint32_t netLayer, std::vector<HcclMemHandle>& symMemHandles);

    /*
     * CollComm初始化方式：
     *      fullMode：给A5及后续新架构使用，完整的CollComm初始化和资源管理
     *      SimpleMode：给A2/A3老芯片使用，由于架构限制，仅将RankGraph、MyRank等放入CollComm管理，简化CollComm实现
     */
    HcclResult InitFullMode(void* rankGraph, aclrtBinHandle binHandle, HcclMem cclBuffer, uint32_t opExpansionMode);
    HcclResult InitSimpleMode(void* rankGraph, aclrtBinHandle binHandle, HcclMem cclBuffer, uint32_t opExpansionMode);

    HcclResult HcclBinaryUnLoad();

    // -- Shrink/Grow private helpers --
    struct DerivedCommContext {
        Hccl::DevType devType{};
        Hccl::CommParams subParams{};
        std::shared_ptr<Hccl::HcclCommunicator> drivedHcclComm;
        void* rankGraph{nullptr};
        HcclMem cclBuffer{};
        uint32_t opExpansionMode{0};
    };
    HcclResult BuildDerivedCommParams(
        const std::vector<uint32_t>& survivorRanks, uint32_t myNewRankId, const std::string& commName,
        const DerivationInfo& derivationInfo, const Hccl::RankTableInfo& mergedRankTable, DerivedCommContext& ctx);
    HcclResult CreateDerivedComm(
        const std::vector<uint32_t>& survivorRanks, const HcclCommConfig& subConfig,
        const DerivationInfo& derivationInfo, const Hccl::RankTableInfo& mergedRankTable, DerivedCommContext& ctx);
    HcclResult SetupDerivedCollComm(
        std::unique_ptr<CollComm>& collComm, const DerivedCommParams& derivedParams, DerivedCommContext& ctx);
    // 静态扩容 root 重选路径，抽取为独立方法
    HcclResult HandleGrowRootReselect(
        uint32_t nRanks, uint32_t rank, const HcclRootInfo* rootInfo, const HcclCommConfig* config,
        Hccl::RankTableInfo& mergedRankTable);
    // 静态扩容正常 Grow 路径
    HcclResult HandleGrowNormal(uint32_t nRanks, Hccl::RankTableInfo& mergedRankTable);
    HcclResult BuildShrinkSurvivorRanks(
        const uint32_t* excludeRanksList, uint32_t excludeRanksCount, ShrinkSurvivorResult& result);
    // Grow/Shrink 前在本地解析当前域 rootRankId，不依赖 root 侧 socket 交互。
    HcclResult ResolveRootRankIdForDerivation();

    void* comm_{nullptr};
    // 持有 HcclCommunicator 的 shared_ptr，延长其生命周期至 CollComm 析构。
    std::shared_ptr<Hccl::HcclCommunicator> hcclCommunicator_{nullptr};
    uint32_t rankId_{};
    std::string commId_;
    CommConfig config_{};
    HcclCommConfig hcclCommConfig_{}; ///< 存储建域时使用的 HcclCommConfig，Shrink INHERIT_PARENT 路径直接取用
    HcclCommStatus commStatus_{HcclCommStatus::HCCL_COMM_STATUS_INVALID};

    ManagerCallbacks callbacks_;
    s32 deviceLogicId_{0};
    bool dcaInited_{false}; // 保证 DetectConnectionAnomalies 的 Deinit 与 Init 严格配对
    uint32_t index_{0};

    // 持有CommunicatorImpl中cclBuffer的shared_ptr拷贝，延长其生命周期
    std::shared_ptr<Hccl::DevBuffer> cclBuffer_{nullptr};

    RankGraph* rankgraph_{nullptr};
    std::unique_ptr<RankGraph> rankGraphOwner_{nullptr};
    std::unique_ptr<CommEngineResMgr> commEngineResMgr_{nullptr};
    std::unique_ptr<ContextManager> contextMgr_{nullptr};
    std::unique_ptr<CommMemMgr> commMemMgr_{nullptr};
    std::unique_ptr<ChannelManager> channelMgr_{nullptr};
    std::shared_ptr<MyRank> myRank_{};
    std::unique_ptr<HcclCommDfx> hcclCommDfx_{nullptr};
    uintptr_t addr_{0};
    std::size_t size_{0};
    HcclMemType memType_{HcclMemType::HCCL_MEM_TYPE_DEVICE};

    // NS recover
    bool isCleaned_{false};

    std::shared_ptr<HDCommunicate> kfcControlTransferH2D_{nullptr};
    std::shared_ptr<HDCommunicate> kfcStatusTransferD2H_{nullptr};
    Hccl::RankIpPortMapPtr rankIpPortMap_;

    CollCommInitMode initMode_{CollCommInitMode::fullMode}; // 初始化模式

    // -- 派生身份字段（Shrink/Grow 共用）--
    std::string topParentIdentifier_;     // 链首父标识，派生 identifier 的基底
    uint32_t generationIndex_{0};         // 本代序号，shrink/grow 派生时父+1
    Hccl::HcclRootHandleV2 rootHandle_{}; // root 身份，Grow 用；Shrink root 被剔除时清空
    // 本进程拓扑探测 session；由链尾派生通信域独占，父域派生成功后不再持有
    std::shared_ptr<Hccl::RankInfoDetect> rankInfoDetect_{nullptr};
    bool hasDetectOwnership_{true}; // detect 所有权仍在当前通信域，派生后置 false
    // Grow root 重选时暂存的 server detect（HcclCommGetRootInfo 存入，HandleGrowRootReselect 取走）。
    std::shared_ptr<void> pendingDetect_{nullptr};
    std::shared_ptr<Hccl::HcclCommunicator> ownedCommunicator_{nullptr};
    // 通信域初始化来源，ranktable或者rootinfo
    CommInitSource initSource_{CommInitSource::rootInfo};

    std::unique_ptr<SymmetricMemory, SymmetricMemoryDeleter> symmetricMemory_{nullptr};
    std::unique_ptr<UbMemSymmetricMemory> ubMemSymmetricMemory_;
    // 保护registeredSymMemHandleMap_：查询时加共享锁，注册和注销时加独占锁。
    mutable std::shared_mutex registeredSymMemHandleMapMtx_;
    // 本地已注册对称内存索引：memTag -> memHandle，供建链时查询全部或远端缺失的句柄。
    std::unordered_map<std::string, HcclMemHandle> registeredSymMemHandleMap_{};
    aclrtBinHandle binHcclHandle_{nullptr};
    std::mutex binHcclmutex_;
    aclrtBinHandle derivedBinHandle_{nullptr}; // 派生域 LoadCclKernelBinHandle 加载的句柄，~CollComm 时卸载
    mutable std::mutex commMutex_;

    // HcommWindow 管理：winHandle→SymmetricWindow device 指针（HcommWindow.symWindow 字段值）
    std::unordered_map<void*, void*> hcommToSymMap_; // key=HcommWindow device, value=SymmetricWindow device
    // SymmetricWindow device→HcommWindow device 反向映射（RegisterWindow 双向登记，GetCommSymWin 反查）
    std::unordered_map<void*, void*> symToHcommMap_;
    // memTag→HcommWindow device（RegisterPendingSymmetricMemHandles 时登记，UpdateHcommWindowRemoteMem 直接查）
    std::unordered_map<std::string, void*> tagToHcommMap_;
    std::shared_mutex hcommWindowMutex_; // 读写锁：查询类（GetCommSymWin/UpdateHcommWindowRemoteMem）多读
};
} // namespace hccl

#endif // COLL_COMM_H
