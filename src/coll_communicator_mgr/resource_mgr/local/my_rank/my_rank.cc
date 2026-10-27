/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "my_rank.h"
#include "endpoint_pair/channel/channel_c_adpt_pub.h"
#include "endpoint_pair/endpoint_c_adpt_pub.h"
#include "comm_mem_c_adpt.h"
#include "hcomm_res_defs.h"
#include "hccl_res.h"
#include "../common/loggers/channel_logger.h" // 日志记录器
#include "hcclCommDfx.h"
#include "config/env_config.h"
#include "env_config/env_config_v2.h"
#include "endpoint_pair/channel/channel_process_pub.h"
#include "coll_comm_aicpu_kernel.h"
#include "launch_aicpu.h"
#include "comm_engine/ccu/ccu_device_res_pub.h"
#include "comm_engine/ccu/ccu_device_pub.h"
#include "config_log.h"
#include "common/comm_engine_utils_pub.h"
#include "hcom_common.h"
#include "op_base.h"
#include "ccu_res.h"
#include "coll_comm_mgr.h"
#include "new_rank_info.h"
#include "roce_channel_desc_configurator.h"

#include <acl/acl.h>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include "shared_jetty_channel_pool.h"
#include "common/hccl_log_keywords_pub.h"
#include "common/orion_adpt_utils_pub.h"
#include "common/dev_type_utils_pub.h"

using namespace hcomm;

namespace MyRankUtils {

uint32_t ResolveUbCommDomainQos(const hccl::CommConfig& commConfig)
{
    if (commConfig.GetConfigHcclQos() == HCCL_COMM_QOS_CONFIG_NOT_SET) {
        return EnvConfig::UB_QOS_DEFAULT;
    }
    return commConfig.GetConfigHcclQos();
}

HcommChannelDesc ChannelDescHccl2Hcomm(const HcclChannelDesc& hcclDesc, const hccl::CommConfig& commConfig)
{
    HcommChannelDesc hcommDesc{};
    (void)HcommChannelDescInit(&hcommDesc, 1);
    hcommDesc.remoteEndpoint = hcclDesc.remoteEndpoint;
    hcommDesc.notifyNum = hcclDesc.notifyNum;
    hcommDesc.memHandles = reinterpret_cast<HcommMemHandle*>(hcclDesc.memHandles);
    hcommDesc.memHandleNum = hcclDesc.memHandleNum;
    (void)memcpy_s(hcommDesc.raws, sizeof(hcommDesc.raws), hcclDesc.raws, sizeof(hcommDesc.raws));
    // RoCE：透传原始 hcclQos（可为 NOT_SET），由 CheckRoceAttr/ApplyRoceQosCompatToSlTc 决定是否映射 SL/TC
    if (hcclDesc.channelProtocol == COMM_PROTOCOL_ROCE) {
        hcommDesc.qos = commConfig.GetConfigHcclQos();
        hcommDesc.roceAttr.retryCnt = hcclDesc.roceAttr.retryCnt;
        hcommDesc.roceAttr.retryInterval = hcclDesc.roceAttr.retryInterval;
        hcommDesc.roceAttr.sl = hcclDesc.roceAttr.sl;
        hcommDesc.roceAttr.tc = hcclDesc.roceAttr.tc;
        return hcommDesc;
    }
    // UB 等：未配置时落默认 4，供下游 Jetty/TP 使用
    hcommDesc.qos = ResolveUbCommDomainQos(commConfig);
    if (hcclDesc.channelProtocol == COMM_PROTOCOL_UB_MEM) {
        hcommDesc.ubMemAttr.pathMode = hcclDesc.ubMemAttr.pathMode;
    }
    return hcommDesc;
}

/* 公共模块函数返回值定义，跟业务层同步 */
const std::unordered_map<CommProtocol, std::string> HCOM_COMM_PROTOCOL_STR_MAP
    = {{COMM_PROTOCOL_RESERVED, "RESERVED"}, {COMM_PROTOCOL_HCCS, "HCCS"},     {COMM_PROTOCOL_ROCE, "ROCE"},
       {COMM_PROTOCOL_PCIE, "PCIE"},         {COMM_PROTOCOL_SIO, "SIO"},       {COMM_PROTOCOL_UB_CTP, "UB_CTP"},
       {COMM_PROTOCOL_UBC_TP, "UBC_TP"},     {COMM_PROTOCOL_UB_MEM, "UB_MEM"}, {COMM_PROTOCOL_UBOE, "UBOE"},
       {COMM_PROTOCOL_UB_RTP, "UB_RTP"}};

inline std::string GetCommProtocolEnumStr(CommProtocol protocol)
{
    auto iter = HCOM_COMM_PROTOCOL_STR_MAP.find(protocol);
    if (iter == HCOM_COMM_PROTOCOL_STR_MAP.end()) {
        return "CommProtocol(" + std::to_string(protocol) + ")";
    } else {
        return iter->second;
    }
}

inline const char* GetTlsTypeStr(EndpointLocType localType)
{
    return localType == ENDPOINT_LOC_TYPE_HOST ? "HostDpu" : "Device";
}

inline const char* GetTlsStatusStr(Hccl::TlsStatus tlsStatus)
{
    if (tlsStatus == Hccl::TlsStatus::ENABLE) {
        return "ENABLE";
    }
    if (tlsStatus == Hccl::TlsStatus::DISABLE) {
        return "DISABLE";
    }
    return "UNKNOWN";
}

// 批量建链状态归类：只要还有通道处于 CONNECTING（非 READY 非终态）就返回 HCCL_E_AGAIN 继续轮询；
// 全批 READY 返回 HCCL_SUCCESS；有 FAILED / TIMEOUT 返回 HCCL_E_NETWORK；其余（RES_LOC_UNAVAIL /
// RES_RMT_UNAVAIL）返回 HCCL_E_UNAVAIL。
// 注：枚举值经 hcomm:: 限定——全局枚举 HcommChannelStatus 与本枚举的 READY / CONNECTING 同名
static HcclResult ClassifyChannelStatus(const int32_t* statusList, uint32_t channelNum)
{
    uint32_t connectingCount = 0;
    uint32_t failCount = 0;
    bool hasResUnavailable = false;
    for (uint32_t i = 0; i < channelNum; ++i) {
        if (statusList[i] == hcomm::HCOMM_CHANNEL_STATUS_READY) {
            continue;
        }
        if (statusList[i] == hcomm::HCOMM_CHANNEL_STATUS_CONNECTING) {
            connectingCount++;
        } else if (
            statusList[i] == hcomm::HCOMM_CHANNEL_STATUS_FAILED
            || statusList[i] == hcomm::HCOMM_CHANNEL_STATUS_TIMEOUT) {
            // 失败与超时归一类，对外都报 HCCL_E_NETWORK
            failCount++;
        } else {
            hasResUnavailable = true;
        }
    }
    if (connectingCount > 0) {
        return HCCL_E_AGAIN;
    }
    if (failCount > 0) {
        return HCCL_E_NETWORK;
    }
    if (hasResUnavailable) {
        return HCCL_E_UNAVAIL;
    }
    return HCCL_SUCCESS;
}

} // namespace MyRankUtils

namespace hccl {

constexpr uint32_t UNREUSE_CHANNEL_IDX = 0xFFFFFFFF;

MyRank::MyRank(
    aclrtBinHandle binHandle, uint32_t rankId, const CommConfig& config, const ManagerCallbacks& callbacks,
    RankGraph* rankGraph, const Hccl::RankIpPortMapPtr& rankIpPortMap)
    : binHandle_(binHandle),
      rankId_(rankId),
      config_(config),
      callbacks_(callbacks),
      rankGraph_(rankGraph),
      rankIpPortMap_(rankIpPortMap)
{}

void MyRank::SetCcuInstance(CcuInsHandle ccuInsHandle)
{
    ccuInsHandle_ = ccuInsHandle;
    ccuInstanceEntity_ = HcommCcuInstanceEntity{};
}

void MyRank::SetAssignedCcuInstance(CcuInsHandle ccuInsHandle)
{
    assignedCcuInsHandle_ = ccuInsHandle;
    assignedCcuInstanceEntity_ = HcommCcuInstanceEntity{};
}

MyRank::~MyRank()
{
    HCCL_INFO("[MyRank][~MyRank] MyRank deinit, rankId_[%u], devLogicId_[%d]", rankId_, devLogicId_);
    bool isDiffDevId = false;
    int32_t threadDevId = INVALID_INT;
    if (HcclDeviceRefresh(threadDevId) != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] HcclDeviceRefresh failed, cleanup in current device context", __func__);
    } else {
        HCCL_INFO("[%s] curDeviceLogicId[%d], threadDevId[%d]", __func__, devLogicId_, threadDevId);
        if (devLogicId_ != threadDevId) {
            if (hrtSetDevice(devLogicId_) == HCCL_SUCCESS) {
                isDiffDevId = true;
            } else {
                HCCL_ERROR("[%s] hrtSetDevice(%d) failed", __func__, devLogicId_);
            }
        }
    }
    // 共享 Jetty Channel 不归 rankPairMgr_ 管理，需在 rankPairMgr_ 析构前独立清理
    (void)SharedJettyChannelPool::GetInstance().DestroyAllByMyRank(this);
    // 先清空反查索引，避免 rankPairMgr_ 析构 EndpointPair 时仍持有指向其的裸指针；
    // 持锁保证与并发 DestroyChannels 的索引读写一致
    {
        std::lock_guard<std::mutex> lock(channelIndexMtx_);
        handleToEpPair_.clear();
    }
    // 析构有时序要求
    rankPairMgr_ = nullptr; // 内部会销毁channel，可能需要返还endpoint与ccu资源
    endpointMgr_ = nullptr; // 内部会销毁endpoint，可能需要返回ccu资源

    struct ResourceCleanupGuard {
        explicit ResourceCleanupGuard(MyRank& myRank) : myRank_(myRank) {}
        ~ResourceCleanupGuard() noexcept
        {
            myRank_.ccuInsHandle_ = 0;
            myRank_.assignedCcuInsHandle_ = 0;
            myRank_.ccuInstanceEntity_ = HcommCcuInstanceEntity{};
            myRank_.assignedCcuInstanceEntity_ = HcommCcuInstanceEntity{};

            if (myRank_.ccuDrvHandle_) {
                myRank_.ccuDrvHandle_ = nullptr; // 先减少引用计数，再尝试关闭
                (void)CcuDeinitFeature(myRank_.devLogicId_);
                // 尝试关闭CCU功能，最后一个调用时会关闭CCU驱动
            }

            myRank_.ReleaseCcuMsCommReservation();

            myRank_.commMems_ = nullptr;
            myRank_.nsRecoveryProcessor_ = nullptr;
        }

        MyRank& myRank_;
    } cleanupGuard(*this);

    if (ccuInsHandle_ != 0 || assignedCcuInsHandle_ != 0) { // 内部清理CCU资源，关闭CCU通道
        // 销毁通信域自有的 ccuInstance（QueryCcuIns 创建）
        if (ccuInsHandle_ != 0) {
            CHK_PRT(static_cast<HcclResult>(HcommCcuInsDestroy(ccuInsHandle_)));
        }
        // 销毁通过 Assign 绑定的 ccuInstance（所有权已转移给通信域）
        if (assignedCcuInsHandle_ != 0 && assignedCcuInsHandle_ != ccuInsHandle_) {
            CHK_PRT(static_cast<HcclResult>(HcommCcuInsDestroy(assignedCcuInsHandle_)));
        }
    }
    // 切换回原来的 DeviceId
    if (isDiffDevId) {
        CHK_PRT(hrtSetDevice(threadDevId));
        CHK_PRT(HcclDeviceRefresh(threadDevId));
    }
}

HcclResult MyRank::GetLocalTlsStatus(EndpointLocType localType, Hccl::TlsStatus& tlsStatus) const
{
    tlsStatus = Hccl::TlsStatus::UNKNOWN;
    s32 deviceLogicId = -1;
    u32 devicePhyId = INVALID_UINT;
    CHK_RET(hrtGetDevice(&deviceLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId), devicePhyId));

    RaInfo info{};
    info.mode = localType == ENDPOINT_LOC_TYPE_HOST ? NetworkMode::NETWORK_PEER_ONLINE : NetworkMode::NETWORK_OFFLINE;
    info.phyId = devicePhyId;
    return Hccl::HrtRaGetTlsStatus(&info, tlsStatus);
}

void MyRank::GetAbnormalChannelTlsStatus(
    const HcclChannelDesc* channelDescs, const int32_t* statusList, uint32_t channelNum,
    std::vector<Hccl::TlsStatus>& tlsStatusList) const
{
    tlsStatusList.assign(channelNum, Hccl::TlsStatus::UNKNOWN);
    for (uint32_t i = 0; i < channelNum; ++i) {
        if (statusList[i] == hcomm::HCOMM_CHANNEL_STATUS_READY) {
            continue;
        }
        Hccl::TlsStatus tlsStatus = Hccl::TlsStatus::UNKNOWN;
        HcclResult ret = GetLocalTlsStatus(channelDescs[i].localEndpoint.loc.locType, tlsStatus);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[GetLocalTlsStatus] Can not get TlsStatus, channelIndex[%u], ret[%d]", i, ret);
            continue;
        }
        tlsStatusList[i] = tlsStatus;
    }
}

HcclResult MyRank::RegisterCommMemsToEndpoint(EndpointHandle epHandle)
{
    std::vector<HcclMem> memVec;
    std::vector<std::string> memTag;
    uint64_t version = 0;
    CHK_RET(commMems_->GetAllMemory(memVec, memTag, version));
    HCCL_INFO("[%s] got %zu memory regions to register, version[%llu]", __func__, memVec.size(), version);
    CHK_RET(endpointMgr_->RegisterMemory(epHandle, memTag, memVec, version));
    return HCCL_SUCCESS;
}

HcclResult MyRank::PrepareMemHandles(
    EndpointHandle epHandle, void** memHandles, uint32_t memHandleNum, std::vector<MemHandle>& memHandleVec)
{
    // 从 CommMems 提取该 channel 需要的 tag 列表
    // GetTagsFromHandles 始终 push cclBuffer；用户 handles 异常时内部跳过，不阻断注册
    std::vector<std::string> memTags;
    CHK_RET(commMems_->GetTagsFromHandles(memHandles, memHandleNum, memTags));

    // 确保 CommMems 全量内存已注册到该 endpoint（版本一致则跳过）
    CHK_RET(RegisterCommMemsToEndpoint(epHandle));

    // 从 endpoint 查询指定 tag 的 MemHandle
    CHK_RET(endpointMgr_->GetMemHandlesByTags(epHandle, memTags, memHandleVec));
    return HCCL_SUCCESS;
}

HcclResult MyRank::UnregMemByTag(const std::string& tag)
{
    CHK_PTR_NULL(endpointMgr_);
    return endpointMgr_->UnregMemByTag(tag);
}

HcclResult MyRank::ReserveCcuMsCommOrFallback()
{
    if (opExpansionMode_ != CCU_MS_MODE) {
        return HCCL_SUCCESS;
    }

    bool reserved = false;
    CHK_RET(CollCommMgr::GetInstance().TryReserveCcuMsComm(devLogicId_, config_.GetConfigCommName(), reserved));
    if (reserved) {
        ccuMsCommReserved_ = true;
        return HCCL_SUCCESS;
    }

    opExpansionMode_ = CCU_SCHED_MODE;
    HCCL_RUN_WARNING(
        "[MyRank][%s] CCU_MS comm already exists on device[%d], fallback to CCU_SCHED, rankId[%u].", __func__,
        devLogicId_, rankId_);
    return HCCL_SUCCESS;
}

void MyRank::ReleaseCcuMsCommReservation()
{
    if (!ccuMsCommReserved_) {
        return;
    }
    CollCommMgr::GetInstance().ReleaseCcuMsComm(devLogicId_, config_.GetConfigCommName());
    ccuMsCommReserved_ = false;
}

void MyRank::ReconcileCcuMsCommReservation(HcclResult initRet)
{
    if (initRet != HCCL_SUCCESS || opExpansionMode_ != CCU_MS_MODE) {
        ReleaseCcuMsCommReservation();
    }
}

HcclResult MyRank::TryInitCcuInstanceOnDemand()
{
    auto ccuInsType = OpExpansionModeToCcuInstanceType(opExpansionMode_);
    if (ccuInsType == CcuInstanceType::CCU_UNUSED) {
        ccuInsHandle_ = 0;
        return HcclResult::HCCL_SUCCESS;
    }

    if (mainBoardType_ == Hccl::HcclMainboardId::MAINBOARD_OTHERS) {
        CHK_RET(CcuGetMainboardType(devLogicId_, mainBoardType_));
    }

    if (mainBoardType_ == Hccl::HcclMainboardId::MAINBOARD_PCIE_STD
        && ccuInsType == CcuInstanceType::CCU_MS) { // 标卡环境下配置CCU_MS拦截报错
        HCCL_ERROR(
            "[%s] ccuInstanceType[%d] not support in %s", __func__, ccuInsType, mainBoardType_.Describe().c_str());
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    // 无UB盖板兜底场景：CCU依赖UB硬件，直接回退AICPU_TS，
    // 避免向设备侧发起必然失败的CCU驱动拉起（会产生ERROR日志且初始化失败）；
    // 依据RankGraph持久化的pcie_fallback显式标记判断，不影响有UB场景下CCU驱动真实故障的显式报错
    if (rankGraph_ != nullptr && rankGraph_->IsLevel0PcieFallback()) {
        opExpansionMode_ = AICPU_TS_MODE;
        ccuInsHandle_ = 0;
        HCCL_RUN_WARNING(
            "[MyRank][%s] level0 is pcie fallback (UB unavailable), fallback to aicpu, rankId[%u].", __func__, rankId_);
        return HcclResult::HCCL_SUCCESS;
    }
    // 拉起ccu驱动
    if (!ccuDrvHandle_) {
        auto ccuInitRet = CcuInitFeature(devLogicId_, ccuDrvHandle_);
        // ccu驱动拉起失败，直接回退至aicpu ts
        if (ccuInitRet == CcuResult::CCU_E_DRV_BUSY) {
            opExpansionMode_ = AICPU_TS_MODE;
            ccuInsHandle_ = 0;
            HCCL_RUN_WARNING(
                "[MyRank][%s] is unsuccessful in initializing ccu driver, "
                "fallback to aicpu, rankId[%u].",
                __func__, rankId_);
            return HcclResult::HCCL_SUCCESS;
        }

        // 预期外返回值属于错误
        if (ccuInitRet != CcuResult::CCU_SUCCESS) {
            HCCL_ERROR("[%s] failed, ret[%d] is not expected.", __func__, ccuInitRet);
            ccuInsHandle_ = 0;
            return static_cast<HcclResult>(ccuInitRet);
        }
    }

    // ccu驱动拉起成功
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MyRank::TryInitCcuInstance()
{
    CHK_RET(ReserveCcuMsCommOrFallback());

    // ccu instance 不在 init 时创建，由 QueryCcuIns 创建或由 Assign 显式绑定；
    // 此处仅拉起 ccu 驱动。
    HcclResult ret = TryInitCcuInstanceOnDemand();
    ReconcileCcuMsCommReservation(ret);
    return ret;
}

HcclResult MyRank::GetDevicePortInternal(uint32_t rank, uint32_t* devPort, EndpointLocType locType)
{
    CHK_PTR_NULL(devPort);
    CHK_PTR_NULL(rankGraph_);

    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    // v1 模式 (mode_ == 0): 强制转换为 RankGraphV1 调用 GetDevicePort
    // v2 模式 (mode_ != 0): 使用 rankGraph_->GetDevicePort()
    if (devType == DevType::DEV_TYPE_910B) {
        RankGraphV1* rankGraphV1 = static_cast<RankGraphV1*>(rankGraph_);
        CHK_RET(rankGraphV1->GetDevicePort(rank, devPort));
    } else {
        CHK_RET(rankGraph_->GetListenPort(rank, devPort, locType));
    }
    return HCCL_SUCCESS;
}

HcclResult MyRank::GetDevicePortByAddr(uint32_t rank, const Hccl::IpAddress& addr, uint32_t* port) const
{
    CHK_PTR_NULL(port);
    CHK_PTR_NULL(rankIpPortMap_);
    auto iterRank = rankIpPortMap_->find(rank);
    if (iterRank == rankIpPortMap_->end()) {
        HCCL_INFO("[MyRank][%s] no port entry for rank[%u] addr[%s].", __func__, rank, addr.GetIpStr().c_str());
        return HCCL_E_NOT_FOUND;
    }
    auto iterAddr = iterRank->second.find(addr);
    if (iterAddr == iterRank->second.end()) {
        HCCL_INFO("[MyRank][%s] no port entry for rank[%u] addr[%s].", __func__, rank, addr.GetIpStr().c_str());
        return HCCL_E_NOT_FOUND;
    }
    *port = iterAddr->second;
    return HCCL_SUCCESS;
}

HcclResult MyRank::GetListenPortByAddr(
    uint32_t rank, const Hccl::IpAddress& addr, EndpointLocType locType, CommEngine engine, uint32_t* port)
{
    CHK_PTR_NULL(port);
    // 两个查表方法二选一：host 侧或 CPU 引擎走原有 rank 级查询，device 侧查两级 IP 端口表
    if (locType != EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE || engine == COMM_ENGINE_CPU) {
        return GetDevicePortInternal(rank, port, locType);
    } else {
        // 查表成功直接返回；未查到（含异常返回值）统一走 fallback 端口，保证建链不因端口表缺失中断
        HcclResult ret = GetDevicePortByAddr(rank, addr, port);
        if (ret == HCCL_SUCCESS) {
            return ret;
        }

        u32 fallbackPort = Hccl::DEFAULT_VALUE_TCPPORT;
        auto portRanges = Hccl::EnvConfig::GetInstance().GetHostNicConfig().GetDeviceSocketPortRange();
        if (!portRanges.empty()) {
            fallbackPort = portRanges[0].min;
        }
        HCCL_INFO(
            "[MyRank][%s] no port entry for rank[%u] addr[%s], use fallback port[%u].", __func__, rank,
            addr.GetIpStr().c_str(), fallbackPort);
        *port = fallbackPort;
        return HCCL_SUCCESS;
    }
}

HcclResult MyRank::Init(HcclMem cclBuffer, const uint32_t opExpansionMode, uint32_t rankNum)
{
    // EXCEPTION_HANDLE_BEGIN
    CHK_RET(hrtGetDevice(&devLogicId_));

    // ns recovery processor初始化
    EXCEPTION_CATCH(nsRecoveryProcessor_ = std::make_unique<NsRecoveryProcessor>(), return HCCL_E_PTR);

    // 创建通信内存管理器
    EXCEPTION_CATCH(commMems_ = std::make_unique<CommMems>(config_.GetConfigBufferSize()), return HCCL_E_PTR);

    // 初始化通信内存
    CHK_RET(commMems_->Init(cclBuffer));

    EXCEPTION_CATCH(engineCtxs_ = std::make_unique<EngineCtxs>(), return HCCL_E_PTR);

    // 通信域配置config优先级更高，当配置默认展开模式时，读取环境变量配置
    opExpansionMode_ = opExpansionMode;
    if (opExpansionMode_ == DEFAULT_MODE) {
        // 环境变量模块已处理，当用户未配置时，输出ccu sched模式
        auto accelerator = Hccl::EnvConfig::GetInstance().GetAlgoConfig().GetHcclAccelerator();
        HCCL_RUN_INFO("[MyRank][%s] set op expansion mode by env[%s].", __func__, accelerator.Describe().c_str());
        opExpansionMode_ = static_cast<uint32_t>(accelerator);
    }

    // 仅自定义算子ccu流程初始化资源
    if (ccuInsHandle_ == 0 && rankNum != 1 && (opExpansionMode_ == CCU_MS_MODE || opExpansionMode_ == CCU_SCHED_MODE)) {
        const uint32_t originOpExpansionMode = opExpansionMode_; // 记录原始加速模式，避免中间执行修改后丢失
        auto ret = TryInitCcuInstance();
        if (ret != HcclResult::HCCL_SUCCESS) { // 申请成功与回退成功都属于成功，其他均非预期
            HCCL_ERROR(
                "[MyRank][%s] failed to init ccu instance, op expansion mode[%u].", __func__, originOpExpansionMode);
            return ret;
        }
    }

    // 创建端点管理器
    EXCEPTION_CATCH(endpointMgr_ = std::make_unique<hccl::EndpointMgr>(), return HCCL_E_PTR);

    // rankPairMgr_初始化
    EXCEPTION_CATCH(rankPairMgr_ = std::make_unique<RankPairMgr>(rankIpPortMap_), return HCCL_E_PTR);

    // EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult MyRank::QueryListenPort(
    uint32_t localRank, uint32_t remoteRank, const EndpointDesc& localEndpointDesc,
    const EndpointDesc& remoteEndpointDesc, uint32_t& listenPort, HcommChannelDesc& hcommDesc, CommEngine engine)
{
    // 查询该socket链接的server端监听的端口（监听方的选择策略需要跟SocketConfig中保持一致）
    Hccl::IpAddress localIpAddr{};
    Hccl::IpAddress remoteIpAddr{};
    CHK_RET(CommAddrToIpAddress(localEndpointDesc.commAddr, localIpAddr));
    CHK_RET(CommAddrToIpAddress(remoteEndpointDesc.commAddr, remoteIpAddr));
    // 角色判定策略：优先按IP比较，IP较小的一方为server；
    // 非对称DPU场景下双端可能复用同一IP，此时按rankId比较，rankId较小的一方为server
    // 与SocketConfig构造函数中角色判定逻辑保持一致，避免双端角色不一致导致建链死锁
    bool isServer = (localIpAddr < remoteIpAddr) || (localIpAddr == remoteIpAddr && localRank < remoteRank);
    // 查询双端devPort（按 rank + IP 查两级端口表，与监听方使用同一数据源）
    uint32_t rmtPort = 0;
    CHK_RET(GetListenPortByAddr(remoteRank, remoteIpAddr, remoteEndpointDesc.loc.locType, engine, &rmtPort));
    if (rmtPort > Hccl::MAX_VALUE_TCPPORT) {
        HCCL_ERROR(
            "[%s] Invalid port[%u] of Rank[%u], max valid port is %u", __func__, rmtPort, remoteRank,
            Hccl::MAX_VALUE_TCPPORT);
        return HCCL_E_PARA;
    }
    if (isServer) {
        uint32_t locPort = 0;
        CHK_RET(GetListenPortByAddr(localRank, localIpAddr, localEndpointDesc.loc.locType, engine, &locPort));
        if (locPort > Hccl::MAX_VALUE_TCPPORT) {
            HCCL_ERROR(
                "[%s] Invalid port[%u] of Rank[%u], max valid port is %u", __func__, locPort, localRank,
                Hccl::MAX_VALUE_TCPPORT);
            return HCCL_E_PARA;
        }
        listenPort = locPort;
        hcommDesc.role = HcommSocketRole::HCOMM_SOCKET_ROLE_SERVER;
        hcommDesc.port = static_cast<uint16_t>(locPort); // HcommChannelDesc.port中填监听端口号
    } else {
        listenPort = rmtPort;
        hcommDesc.role = HcommSocketRole::HCOMM_SOCKET_ROLE_CLIENT;
        // HcommChannelDesc.port中填对端端口号(此场景下对端端口号也就是监听端口号)
        hcommDesc.port = static_cast<uint16_t>(rmtPort);
    }

    HCCL_INFO(
        "[%s] localRank[%u] remoteRank[%u] localIp[%s] remoteIp[%s] role[%s] listenPort[%u]", __func__, localRank,
        remoteRank, localIpAddr.GetIpStr().c_str(), remoteIpAddr.GetIpStr().c_str(), isServer ? "SERVER" : "CLIENT",
        listenPort);

    return HCCL_SUCCESS;
}

HcclResult MyRank::GetEndpointPairFromChannel(
    const HcclChannelDesc& channelDesc, uint32_t channelIndex, uint32_t channelNum, uint32_t& remoteRank,
    hcomm::EndpointPair*& endpointPair, RankPair*& rankPair)
{
    remoteRank = channelDesc.remoteRank;
    HCCL_INFO(
        "[%s][%u/%u] remoteRank[%u] localProtocol[%d] remoteProtocol[%d]", __func__, channelIndex + 1, channelNum,
        remoteRank, channelDesc.localEndpoint.protocol, channelDesc.remoteEndpoint.protocol);

    const RankIdPair rankIdPair = std::make_pair(rankId_, remoteRank);
    const EndpointDescPair endpointDescPair = std::make_pair(channelDesc.localEndpoint, channelDesc.remoteEndpoint);
    CHK_RET(rankPairMgr_->Get(rankIdPair, rankPair));
    CHK_PTR_NULL(rankPair);
    CHK_RET(rankPair->GetEndpointPair(endpointDescPair, endpointPair));
    CHK_PTR_NULL(endpointPair);
    return HCCL_SUCCESS;
}

void MyRank::FillEndpointPairIpIndex(
    hcomm::EndpointPair* endpointPair, uint32_t remoteRank, const EndpointDesc& localEp,
    const EndpointDesc& remoteEp) const
{
    if (rankGraph_ == nullptr || endpointPair == nullptr) {
        return;
    }
    Hccl::IpAddress localIpAddr{};
    Hccl::IpAddress remoteIpAddr{};
    if (CommAddrToIpAddress(localEp.commAddr, localIpAddr) != HCCL_SUCCESS
        || CommAddrToIpAddress(remoteEp.commAddr, remoteIpAddr) != HCCL_SUCCESS) {
        HCCL_WARNING("[MyRank::FillEndpointPairIpIndex] CommAddrToIpAddress failed, keep ipIndex default 0.");
        return;
    }
    uint32_t localIpIndex = 0;
    uint32_t remoteIpIndex = 0;
    HcclResult res = rankGraph_->GetIpIndex(rankId_, localIpAddr, localIpIndex);
    if (res != HCCL_SUCCESS && res != HCCL_E_NOT_SUPPORT) {
        HCCL_WARNING("[MyRank::FillEndpointPairIpIndex] GetIpIndex failed for localRank, keep ipIndex default 0.");
    }
    res = rankGraph_->GetIpIndex(remoteRank, remoteIpAddr, remoteIpIndex);
    if (res != HCCL_SUCCESS && res != HCCL_E_NOT_SUPPORT) {
        HCCL_WARNING("[MyRank::FillEndpointPairIpIndex] GetIpIndex failed for remoteRank, keep ipIndex default 0.");
    }
    endpointPair->SetIpIndexPair(localIpIndex, remoteIpIndex);
}

inline std::string AddProtocolToSocketTag(const std::string& socketTag, const HcclChannelDesc* channelDescs)
{
    std::string newSocketTag = socketTag + "_protocol_" + std::to_string(channelDescs->channelProtocol);
    return newSocketTag;
}

HcclResult MyRank::BatchServerInitForChannels(
    const HcclChannelDesc* channelDescs, uint32_t channelNum, const std::string& socketTag,
    ReuseSocketIdxMap& reuseSocketIdxMap)
{
    // 批量获取socket，与server监听隔离开
    for (uint32_t i = 0; i < channelNum; ++i) {
        hcomm::EndpointPair* endpointPair = nullptr;
        RankPair* rankPair = nullptr;
        uint32_t remoteRank = 0;

        CHK_RET(GetEndpointPairFromChannel(channelDescs[i], i, channelNum, remoteRank, endpointPair, rankPair));

        // 查 local/remote addr 的 ipIndex 注入 EndpointPair，供 SocketConfig 拼 hccpTag 使用
        FillEndpointPairIpIndex(
            endpointPair, remoteRank, channelDescs[i].localEndpoint, channelDescs[i].remoteEndpoint);

        if (reuseSocketIdxMap.find(rankPair) == reuseSocketIdxMap.end()) {
            std::unordered_map<hcomm::EndpointPair*, u32> endpointPair2Idx{};
            endpointPair2Idx.emplace(endpointPair, 0);
            reuseSocketIdxMap.emplace(rankPair, endpointPair2Idx);
        } else if (reuseSocketIdxMap[rankPair].find(endpointPair) == reuseSocketIdxMap[rankPair].end()) {
            reuseSocketIdxMap[rankPair].emplace(endpointPair, 0);
        }
        u32& reuseIdx = reuseSocketIdxMap[rankPair][endpointPair];

        uint32_t devicePhyId;
        uint32_t remoteDevicePhyId;
        rankGraph_->GetDeviceId(rankId_, &devicePhyId);
        rankGraph_->GetDeviceId(remoteRank, &remoteDevicePhyId);

        const std::string socketTagAddProto = AddProtocolToSocketTag(socketTag, &channelDescs[i]);
        auto ret = endpointPair->ServerInit(
            rankId_, remoteRank, socketTagAddProto, reuseIdx, devicePhyId, remoteDevicePhyId);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] ServerInitFailed, channelIndex[%u], remoteRank[%u], protocol[%d] reuseIdx[%u]", __func__, i,
                remoteRank, channelDescs[i].localEndpoint.protocol, reuseIdx),
            ret);

        HCCL_INFO(
            "[%s][%u/%u] server listen successfully, remoteRank[%u], reuseIdx[%u]", __func__, i + 1, channelNum,
            remoteRank, reuseIdx);
    }
    return HCCL_SUCCESS;
}

HcclResult MyRank::BatchGetSocketsForChannels(
    const HcclChannelDesc* channelDescs, uint32_t channelNum, const std::string& socketTag,
    std::vector<HcommChannelDesc>& hcommDescs, ReuseSocketIdxMap& reuseSocketIdxMap, CommEngine engine)
{
    for (uint32_t i = 0; i < channelNum; ++i) {
        hcomm::EndpointPair* endpointPair = nullptr;
        RankPair* rankPair = nullptr;
        uint32_t remoteRank = 0;

        CHK_RET(GetEndpointPairFromChannel(channelDescs[i], i, channelNum, remoteRank, endpointPair, rankPair));

        // 查 local/remote addr 的 ipIndex 注入 EndpointPair，供 SocketConfig 拼 hccpTag 使用
        FillEndpointPairIpIndex(
            endpointPair, remoteRank, channelDescs[i].localEndpoint, channelDescs[i].remoteEndpoint);

        uint32_t listenPort = 0;
        CHK_RET(QueryListenPort(
            rankId_, remoteRank, channelDescs[i].localEndpoint, channelDescs[i].remoteEndpoint, listenPort,
            hcommDescs[i], engine));

        u32& reuseIdx = reuseSocketIdxMap[rankPair][endpointPair];
        uint32_t devicePhyId;
        uint32_t remoteDevicePhyId;
        rankGraph_->GetDeviceId(rankId_, &devicePhyId);
        rankGraph_->GetDeviceId(remoteRank, &remoteDevicePhyId);
        HCCL_INFO(
            "[MyRank][BatchCreateSockets] rankId_[%u] devicePhyId[%u] remoteRank[%u] remoteDevicePhyId[%u]", rankId_,
            devicePhyId, remoteRank, remoteDevicePhyId);
        Hccl::Socket* socket = nullptr;
        const std::string socketTagAddProto = AddProtocolToSocketTag(socketTag, &channelDescs[i]);
        auto ret = endpointPair->GetConnectedSocket(
            rankId_, remoteRank, socketTagAddProto, reuseIdx, listenPort, socket, devicePhyId, remoteDevicePhyId);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] failed to get socket, channelIndex[%u], remoteRank[%u], protocol[%d], reuseIdx[%u], tag[%s]",
                __func__, i, remoteRank, channelDescs[i].localEndpoint.protocol, reuseIdx, socketTagAddProto.c_str()),
            ret);
        CHK_PTR_NULL(socket);

        hcommDescs[i].socket = reinterpret_cast<HcommSocket>(socket);

        HCCL_INFO(
            "[%s][%u/%u] socket created successfully, remoteRank[%u], socket[%p] reuseIdx[%u]", __func__, i + 1,
            channelNum, remoteRank, socket, reuseIdx);
        reuseIdx++;
    }
    return HCCL_SUCCESS;
}

HcclResult MyRank::BatchCreateSockets(
    const HcclChannelDesc* channelDescs, uint32_t channelNum, const std::string& socketTag,
    std::vector<HcommChannelDesc>& hcommDescs, CommEngine engine)
{
    CHK_PTR_NULL(channelDescs);
    CHK_PRT_RET(channelNum == 0, HCCL_ERROR("[%s] invalid param: channelNum is zero", __func__), HCCL_E_PARA);

    ReuseSocketIdxMap reuseSocketIdxMap{};
    // socket服务器首先监听
    CHK_RET(BatchServerInitForChannels(channelDescs, channelNum, socketTag, reuseSocketIdxMap));
    // socket添加白名单以及进行连接，获取最后的socket
    CHK_RET(BatchGetSocketsForChannels(channelDescs, channelNum, socketTag, hcommDescs, reuseSocketIdxMap, engine));
    return HCCL_SUCCESS;
}

HcclResult MyRank::NeedCheckConsistency(bool& needCheck) const
{
    needCheck = false;
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(
        !isGe950.has_value(), HCCL_ERROR("[MyRank][%s] get device type failed, cannot check consistency.", __func__),
        HCCL_E_RUNTIME);
    if (!(*isGe950)) {
        HCCL_INFO("[MyRank][%s] devType is not 950 or later, no need to check consistency.", __func__);
        return HCCL_SUCCESS;
    }

    const int32_t mode = Hccl::EnvConfig::GetInstance().GetLogConfig().GetDfsConfig().rankConsistentState;
    needCheck = (mode == Hccl::RANK_CONSISTENT_STATE_FIRST || mode == Hccl::RANK_CONSISTENT_STATE_ON);
    HCCL_INFO("[MyRank][%s] rankConsistentState[%d], needCheck[%d].", __func__, mode, static_cast<int32_t>(needCheck));
    return HCCL_SUCCESS;
}

HcclResult MyRank::BatchExchangeAndCheckConsistency(
    const HcclChannelDesc* channelDescs, const ChannelHandle* channelHandles, uint32_t channelNum)
{
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channelHandles);
    CHK_PRT_RET(channelNum == 0, HCCL_ERROR("[%s] invalid param: channelNum is zero", __func__), HCCL_E_PARA);

    bool needCheck = false;
    CHK_RET(NeedCheckConsistency(needCheck));
    if (!needCheck) {
        HCCL_INFO("[MyRank][%s] no need to check consistency, skip.", __func__);
        return HCCL_SUCCESS;
    }

    // 仅需远端rank列表，无需向ExchangeInfoMgr透传完整channelDesc
    std::vector<uint32_t> remoteRanks(channelNum);
    for (uint32_t i = 0; i < channelNum; ++i) {
        remoteRanks[i] = channelDescs[i].remoteRank;
    }

    auto startConsistency = std::chrono::steady_clock::now();
    CHK_RET(exchangeInfoMgr_.BatchExchangeAndCheckConsistency(
        remoteRanks.data(), channelHandles, channelNum, collCommConfigConsistency_));
    auto endConsistency = std::chrono::steady_clock::now();
    auto durationConsistency
        = std::chrono::duration_cast<std::chrono::microseconds>(endConsistency - startConsistency).count();
    HCCL_INFO(
        "[MyRank][%s] BatchExchangeAndCheckConsistency Time Elapsed [%lld]us, channelNum [%u]", __func__,
        durationConsistency, channelNum);
    return HCCL_SUCCESS;
}

constexpr uint32_t MEM_HANDLE_NUM_MAX = 256; // memHandleNum的默认限制最大为256
constexpr uint32_t NOTIFY_NUM_MAX = 64;      // notifynum 的默认限制最大为64

static bool HasDuplicateChannelDesc(const HcclChannelDesc* channelDesc, uint32_t channelNum)
{
    // 通过localEndpoint, remoteEndpoint 和 channelProtocol 判断是否有重复的 channelDesc
    std::unordered_map<EndpointDescPair, std::unordered_set<int32_t>, EndpointDescPairHash, EndpointDescPairEqual>
        descSet;
    descSet.reserve(channelNum);
    for (u32 index = 0; index < channelNum; ++index) {
        EndpointDescPair endpointPair{channelDesc[index].localEndpoint, channelDesc[index].remoteEndpoint};
        auto& protocolSet = descSet[std::move(endpointPair)];
        const int32_t protocol = static_cast<int32_t>(channelDesc[index].channelProtocol);
        if (!protocolSet.insert(protocol).second) {
            return true;
        }
    }
    return false;
}

HcclResult MyRank::CheckChannelParam(CommEngine engine, const HcclChannelDesc* channelDesc, uint32_t channelNum) const
{
    for (u32 index = 0; index < channelNum; ++index) {
        if (engine == COMM_ENGINE_AIV) {
            CHK_PRT_RET(
                (channelDesc[index].memHandleNum > MEM_HANDLE_NUM_MAX),
                HCCL_ERROR(
                    "[%s]Channeldesc[%u] invalid memHandleNum, memHandleNum[%u], max channel num[%u]", __func__, index,
                    channelDesc[index].memHandleNum, MEM_HANDLE_NUM_MAX),
                HCCL_E_PARA);
            CHK_PRT_RET(
                (channelDesc[index].memHandleNum != 0 && channelDesc[index].memHandles == nullptr),
                HCCL_ERROR("[%s]Channeldesc[%u] invalid memHandles, memHandles is null", __func__, index), HCCL_E_PARA);
        } else {
            if (channelDesc[index].memHandleNum != 0) {
                HCCL_WARNING(
                    "[%s]Channeldesc[%u] memHandleNum[%u] is non-zero, memHandle exchange is not supported.", __func__,
                    index, channelDesc[index].memHandleNum);
            }
        }

        // HcclChannelAcquire 不支持 CPU+UB
        if (engine == COMM_ENGINE_CPU && channelDesc[index].channelProtocol != COMM_PROTOCOL_ROCE) {
            HCCL_ERROR(
                "[%s] Channeldesc[%u/%u] COMM_ENGINE_CPU only support COMM_PROTOCOL_ROCE, but got protocol[%d]",
                __func__, index, channelNum, channelDesc[index].channelProtocol);
            return HCCL_E_PARA;
        }
        CHK_PRT_RET(
            channelDesc[index].notifyNum > NOTIFY_NUM_MAX,
            HCCL_ERROR(
                "[%s]Channeldesc[%u] invalid notifyNum [%u], max notify num[%u]", __func__, index,
                channelDesc[index].notifyNum, NOTIFY_NUM_MAX),
            HCCL_E_PARA);
    }

    if (engine == COMM_ENGINE_CCU) {
        if (HasDuplicateChannelDesc(channelDesc, channelNum)) {
            HCCL_ERROR("[%s]Duplicate channelDesc found in CCU engine.", __func__);
            return HCCL_E_NOT_SUPPORT;
        }
    }

    return HCCL_SUCCESS;
}

// 批量创建channels，如果CCU资源不足（如Xn, Cke, channel ctx, jetty ctx, wqebb）会失败，返回HCCL_E_UNAVAIL
HcclResult MyRank::BatchCreateChannels(
    CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum,
    std::vector<HcommChannelDesc>& hcommDescs, ChannelHandle* channelHandles, ChannelHandle* deviceChannelHandles,
    std::vector<std::vector<MemHandle>>& allHandles)
{
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channelHandles);
    CHK_PTR_NULL(deviceChannelHandles);
    CHK_PRT_RET(channelNum == 0, HCCL_ERROR("[%s] invalid param: channelNum is zero", __func__), HCCL_E_PARA);

    // 持锁保护 newChannels_/handleToEpPair_；失败路径调用的 DestroyNewChannels 由本函数持锁，内部不再加锁
    std::lock_guard<std::mutex> lock(channelIndexMtx_);

    uint32_t localRank = rankId_;
    CHK_SMART_PTR_NULL(commMems_);
    CHK_PTR_NULL(endpointMgr_);
    std::unordered_map<RankPair*, std::unordered_map<CommEngine, std::unordered_map<hcomm::EndpointPair*, u32>>>
        reuseChannelIdxMap{};

    // 记录本轮新申请的channel
    newChannels_.clear();
    bool isAllSuccess = true;

    for (uint32_t i = 0; i < channelNum; ++i) {
        const EndpointDesc& localEndpointDesc = channelDescs[i].localEndpoint;
        const EndpointDesc& remoteEndpointDesc = channelDescs[i].remoteEndpoint;
        uint32_t remoteRank = channelDescs[i].remoteRank;

        HCCL_RUN_INFO(
            "[%s]userRank[%u] remoteRank[%u] localProtocol[%d] remoteProtocol[%d] engine[%s] channel[%u/%u]",
            LOG_KEYWORDS_LINK_INFO.c_str(), localRank, remoteRank, localEndpointDesc.protocol,
            remoteEndpointDesc.protocol, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), i + 1,
            channelNum);

        EndpointHandle epHandle = nullptr;
        auto ret = endpointMgr_->Get(localEndpointDesc, epHandle);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] failed to get endpoint, channelIndex[%u], remoteRank[%u], protocol[%d]", __func__, i, remoteRank,
                localEndpointDesc.protocol),
            ret);
        CHK_PTR_NULL(epHandle);

        // 启动监听（按 rank + IP 查两级端口表，与连接方 QueryListenPort 使用同一数据源）
        uint32_t listenPort = 0;
        Hccl::IpAddress localIpAddr{};
        CHK_RET(CommAddrToIpAddress(localEndpointDesc.commAddr, localIpAddr));
        CHK_RET(GetListenPortByAddr(localRank, localIpAddr, localEndpointDesc.loc.locType, engine, &listenPort));
        CHK_RET(static_cast<HcclResult>(HcommEndpointStartListen(epHandle, listenPort, nullptr)));

        HCCL_INFO(
            "[%s][%u/%u] remoteRank[%u] epHandle[%p] protocol[%d]", __func__, i + 1, channelNum, remoteRank, epHandle,
            localEndpointDesc.protocol);

        // 注册内存
        CHK_RET(PrepareMemHandles(epHandle, channelDescs[i].memHandles, channelDescs[i].memHandleNum, allHandles[i]));
        HCCL_INFO(
            "[%s][%u/%u] remoteRank[%u] got %zu user memory handles", __func__, i + 1, channelNum, remoteRank,
            allHandles[i].size());

        hcommDescs[i].exchangeAllMems = false;
        hcommDescs[i].memHandles = allHandles[i].data();
        hcommDescs[i].memHandleNum = allHandles[i].size();

        hcomm::EndpointPair* endpointPair = nullptr;
        RankIdPair rankIdPair = std::make_pair(localRank, remoteRank);
        EndpointDescPair endpointDescPair = std::make_pair(localEndpointDesc, remoteEndpointDesc);
        RankPair* rankPair = nullptr;
        CHK_RET(rankPairMgr_->Get(rankIdPair, rankPair));
        CHK_PTR_NULL(rankPair);
        CHK_RET(rankPair->GetEndpointPair(endpointDescPair, endpointPair));
        CHK_PTR_NULL(endpointPair);

        if (reuseChannelIdxMap.find(rankPair) == reuseChannelIdxMap.end()) {
            std::unordered_map<CommEngine, std::unordered_map<hcomm::EndpointPair*, u32>> engine2EndpointPairMap{};
            std::unordered_map<hcomm::EndpointPair*, u32> endpointPair2Idx{};
            endpointPair2Idx.emplace(endpointPair, 0);
            engine2EndpointPairMap.emplace(engine, endpointPair2Idx);
            reuseChannelIdxMap.emplace(rankPair, engine2EndpointPairMap);
        } else if (reuseChannelIdxMap[rankPair].find(engine) == reuseChannelIdxMap[rankPair].end()) {
            std::unordered_map<hcomm::EndpointPair*, u32> endpointPair2Idx{};
            endpointPair2Idx.emplace(endpointPair, 0);
            reuseChannelIdxMap[rankPair].emplace(engine, endpointPair2Idx);
        } else if (
            reuseChannelIdxMap[rankPair][engine].find(endpointPair) == reuseChannelIdxMap[rankPair][engine].end()) {
            reuseChannelIdxMap[rankPair][engine].emplace(endpointPair, 0);
        }

        u32& reuseIdx = reuseChannelIdxMap[rankPair][engine][endpointPair];
        u32 idx = reuseIdx;
        /* hostNIC -- DeviceNic（transport不复用link/Channel），此流程也是新创建channel，需要计入isNewChannel */
        if (localEndpointDesc.loc.locType != remoteEndpointDesc.loc.locType) {
            idx = UNREUSE_CHANNEL_IDX;
        }
        bool isNewChannel = (endpointPair->IsChannelNotExist(engine, reuseIdx) || (idx == UNREUSE_CHANNEL_IDX));

        // CreateChannel 返回 HCCL_E_UNAVAIL 表示资源不足创建失败
        ret = endpointPair->CreateChannel(
            epHandle, engine, idx, &hcommDescs[i], channelHandles + i, deviceChannelHandles + i);
        if (ret == HCCL_E_TIMEOUT || ret == HCCL_E_INTERNAL) {
            Hccl::TlsStatus tlsStatus = Hccl::TlsStatus::UNKNOWN;
            CHK_PRT_CONT(
                GetLocalTlsStatus(localEndpointDesc.loc.locType, tlsStatus) != HCCL_SUCCESS,
                HCCL_WARNING("[GetLocalTlsStatus] Can not get TlsStatus"));
            HCCL_ERROR(
                "[%s] failed to create channel, channelIndex[%u], localRank[%u], remoteRank[%u], protocol[%s], "
                "tlsType[%s], tlsStatus[%s], ret[%d]",
                __func__, i, localRank, remoteRank,
                MyRankUtils::GetCommProtocolEnumStr(localEndpointDesc.protocol).c_str(),
                MyRankUtils::GetTlsTypeStr(localEndpointDesc.loc.locType), MyRankUtils::GetTlsStatusStr(tlsStatus),
                ret);
        }
        if (ret == HCCL_E_UNAVAIL) {
            // 申请channel因资源不足失败，清理已申请的channel
            HCCL_RUN_WARNING(
                "[%s] create channel unsuccessfully, channelIndex[%u], remoteRank[%u], engine[%s], reuseIdx[%u], need "
                "clean "
                "new channels",
                __func__, i + 1, remoteRank, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), reuseIdx);
            isAllSuccess = false;
            break;
        }
        // 记录新申请的channel信息，用于清理临时资源
        if (isNewChannel) {
            newChannels_.emplace_back(std::make_pair(i, reuseIdx));
        }

        if (ret != HCCL_SUCCESS) {
            if (ret != HCCL_E_TIMEOUT && ret != HCCL_E_INTERNAL) {
                HCCL_ERROR(
                    "[%s] failed to create channel, channelIndex[%u], remoteRank[%u], engine[%s], reuseIndex[%u]",
                    __func__, i + 1, remoteRank, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),
                    reuseIdx);
            }
            return ret;
        }
        if (idx != UNREUSE_CHANNEL_IDX) {
            reuseIdx++;
        }

        // 登记 handle -> EndpointPair 反查索引；真实槽位由 EndpointPair::handleToLoc_ 维护
        handleToEpPair_[channelHandles[i]] = endpointPair;

        HCCL_INFO(
            "[%s][%u/%u] channel created successfully, remoteRank[%u], channelHandle[%p]", __func__, i + 1, channelNum,
            remoteRank, channelHandles[i]);
    }

    // 如果申请失败，清理endpoint pair中记录的channel handle
    if (!isAllSuccess) {
        HCCL_RUN_WARNING(
            "[%s] create channel unsuccessfully, destroy new channels num[%zu], engine[%s]", __func__,
            newChannels_.size(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        CHK_RET(DestroyNewChannels(engine, channelDescs, newChannels_));
        return HCCL_E_UNAVAIL;
    }

    return HCCL_SUCCESS;
}

HcclResult MyRank::DestroyNewChannels(
    CommEngine engine, const HcclChannelDesc* channelDescs, const std::vector<std::pair<u32, u32>>& newChannels)
{
    HcclResult firstErr = HCCL_SUCCESS;
    uint32_t localRank = rankId_;
    for (auto idxPairIter = std::rbegin(newChannels); idxPairIter != std::rend(newChannels);
         ++idxPairIter) { // 由于新申请的在申请过的后面，所以要从后往前找reuseIdx销毁
        auto idxPair = *idxPairIter;
        const EndpointDesc& localEndpointDesc = channelDescs[idxPair.first].localEndpoint;
        const EndpointDesc& remoteEndpointDesc = channelDescs[idxPair.first].remoteEndpoint;
        uint32_t remoteRank = channelDescs[idxPair.first].remoteRank;
        hcomm::EndpointPair* endpointPair = nullptr;
        RankIdPair rankIdPair = std::make_pair(localRank, remoteRank);
        EndpointDescPair endpointDescPair = std::make_pair(localEndpointDesc, remoteEndpointDesc);
        RankPair* rankPair = nullptr;
        CHK_RET(rankPairMgr_->Get(rankIdPair, rankPair));
        CHK_PTR_NULL(rankPair);
        CHK_RET(rankPair->GetEndpointPair(endpointDescPair, endpointPair));
        CHK_PTR_NULL(endpointPair);
        // DestroyChannel 会 erase 向量导致下标变化, 需先取出 handle
        ChannelHandle handleToErase = 0;
        endpointPair->GetChannelHandle(engine, idxPair.second, handleToErase);
        // 单个 channel 销毁失败不中断其余清理；记录首个错误，最终统一清空本次新建列表
        HcclResult destroyRet = endpointPair->DestroyChannel(engine, idxPair.second);
        if (destroyRet != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[%s] DestroyChannel failed, engine[%s] reuseIdx[%u] ret[%d], continue.", __func__,
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), idxPair.second, destroyRet);
            if (firstErr == HCCL_SUCCESS) {
                firstErr = destroyRet;
            }
        }
        if (handleToErase != 0) {
            handleToEpPair_.erase(handleToErase);
        }
    }
    newChannels_.clear();
    return firstErr;
}

HcclResult
MyRank::QueryOneChannel(CommEngine engine, const HcclChannelDesc& channelDesc, u32 reuseIdx, ChannelHandle& handle)
{
    handle = 0;
    const RankIdPair rankIdPair = std::make_pair(rankId_, channelDesc.remoteRank);
    const EndpointDescPair endpointDescPair = std::make_pair(channelDesc.localEndpoint, channelDesc.remoteEndpoint);

    RankPair* rankPair = nullptr;
    if (rankPairMgr_->Find(rankIdPair, rankPair) != HCCL_SUCCESS || rankPair == nullptr) {
        return HCCL_SUCCESS;
    }
    hcomm::EndpointPair* epPair = nullptr;
    if (rankPair->GetEndpointPair(endpointDescPair, epPair) != HCCL_SUCCESS || epPair == nullptr) {
        return HCCL_SUCCESS;
    }
    ChannelHandle slotHandle = 0;
    if (epPair->GetChannelHandle(engine, reuseIdx, slotHandle)) {
        handle = slotHandle;
    }
    return HCCL_SUCCESS;
}

HcclResult MyRank::QueryChannels(
    CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle* channels)
{
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(channelNum == 0, HCCL_ERROR("[%s] invalid param: channelNum is zero", __func__), HCCL_E_PARA);

    HCCL_INFO(
        "[MyRank][%s] Enter engine[%s] channelNum[%u] rankId[%u]", __func__,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum, rankId_);

    // 与 BatchCreateChannels 保持一致的 reuseIdx 累计逻辑
    std::unordered_map<RankIdPair, std::unordered_map<EndpointDescPair, std::unordered_map<CommEngine, u32>>>
        reuseIdxMap{};

    for (uint32_t i = 0; i < channelNum; ++i) {
        channels[i] = 0;
        const auto& channelDesc = channelDescs[i];
        uint32_t remoteRank = channelDesc.remoteRank;
        const RankIdPair rankIdPair = std::make_pair(rankId_, remoteRank);
        const EndpointDescPair endpointDescPair = std::make_pair(channelDesc.localEndpoint, channelDesc.remoteEndpoint);

        u32& reuseIdx = reuseIdxMap[rankIdPair][endpointDescPair][engine];
        u32 idx = reuseIdx;
        if (channelDesc.localEndpoint.loc.locType != channelDesc.remoteEndpoint.loc.locType) {
            idx = UNREUSE_CHANNEL_IDX;
        }

        // 仅当非 UNREUSE 且槽位存在时返回 handle
        if (idx != UNREUSE_CHANNEL_IDX) {
            (void)QueryOneChannel(engine, channelDesc, reuseIdx, channels[i]);
        }

        HCCL_INFO(
            "[MyRank][%s] [%u/%u] remoteRank[%u] exist[%s] handle[0x%llx] reuseIdx[%u] unreuse[%d]", __func__, i + 1,
            channelNum, remoteRank, channels[i] != 0 ? "yes" : "no", channels[i], reuseIdx, idx == UNREUSE_CHANNEL_IDX);

        // 与 BatchCreateChannels 一致: 非 UNREUSE 才递增 reuseIdx(引用, 直接改 map 内值)
        if (idx != UNREUSE_CHANNEL_IDX) {
            reuseIdx++;
        }
    }

    // 对发生句柄转换的引擎，经平台 H2D 反向映射把 host 句柄转换为用户实际使用的句柄
    // （device 句柄），保证 Query 返回值与 HcclChannelAcquire 出参一致
    if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_AIV) {
        for (uint32_t i = 0; i < channelNum; ++i) {
            if (channels[i] != 0) {
                ChannelHandle deviceHandle = 0;
                if (hcomm::ChannelProcess::ResolveHostHandleToDevice(channels[i], deviceHandle) == HCCL_SUCCESS
                    && deviceHandle != 0) {
                    channels[i] = deviceHandle;
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

// 记录批量销毁过程中的首个错误与对应计数，供 DestroyOneChannel 复用
static void RecordDestroyError(HcclResult& firstErr, u32& errCnt, HcclResult err)
{
    errCnt++;
    if (firstErr == HCCL_SUCCESS) {
        firstErr = err;
    }
}

HcclResult MyRank::DestroyOneChannel(
    ChannelHandle userHandle, u32 index, HcclResult& firstErr, u32& invalidHandleCnt, u32& failedCnt)
{
    // 反查索引以 host 句柄为键：AIV/AICPU_TS 入参为 device 句柄，先经 D2H 映射解析为 host 句柄
    ChannelHandle hostHandle = userHandle;
    ChannelHandle resolved = 0;
    if (hcomm::ChannelProcess::ResolveUserHandleToHost(userHandle, resolved) == HCCL_SUCCESS && resolved != 0) {
        hostHandle = resolved;
    }
    auto it = handleToEpPair_.find(hostHandle);
    if (it == handleToEpPair_.end()) {
        HCCL_ERROR("[%s] channel handle[0x%llx] not found, channelIndex[%u].", __func__, userHandle, index);
        RecordDestroyError(firstErr, invalidHandleCnt, HCCL_E_NOT_FOUND);
        return HCCL_SUCCESS;
    }
    hcomm::EndpointPair* epPair = it->second;
    if (epPair == nullptr) {
        // 反查索引条目为空指针（异常数据），清理并按无效句柄容错
        HCCL_ERROR("[%s] channel handle[0x%llx] endpoint pair is null, channelIndex[%u].", __func__, userHandle, index);
        handleToEpPair_.erase(it);
        RecordDestroyError(firstErr, invalidHandleCnt, HCCL_E_NOT_FOUND);
        return HCCL_SUCCESS;
    }
    CommEngine engine = COMM_ENGINE_RESERVED;
    u32 reuseIdx = 0;
    if (!epPair->FindChannelLoc(hostHandle, engine, reuseIdx)) {
        HCCL_ERROR("[%s] channel handle[0x%llx] FindChannelLoc failed, channelIndex[%u].", __func__, userHandle, index);
        RecordDestroyError(firstErr, invalidHandleCnt, HCCL_E_NOT_FOUND);
        return HCCL_SUCCESS;
    }
    // 暂只支持 CCU 引擎： 其他场景的 channel 销毁无法保证资源完整释放
    if (engine != COMM_ENGINE_CCU) {
        HCCL_WARNING(
            "[%s] channel handle[0x%llx] engine[%s] not supported, only CCU engine supported, channelIndex[%u].",
            __func__, userHandle, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), index);
        RecordDestroyError(firstErr, failedCnt, HCCL_E_NOT_SUPPORT);
        return HCCL_SUCCESS;
    }
    HcclResult destroyRet = epPair->DestroyChannel(engine, reuseIdx);
    if (destroyRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] DestroyChannel failed, handle[0x%llx] ret[%d], continue.", __func__, hostHandle, destroyRet);
        RecordDestroyError(firstErr, failedCnt, destroyRet);
        return HCCL_SUCCESS;
    }
    handleToEpPair_.erase(it);
    return HCCL_SUCCESS;
}

HcclResult MyRank::DestroyChannels(const ChannelHandle* channels, uint32_t channelNum)
{
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(channelNum == 0, HCCL_ERROR("[%s] invalid param: channelNum is zero", __func__), HCCL_E_PARA);

    std::lock_guard<std::mutex> lock(channelIndexMtx_);

    HCCL_INFO("[MyRank][%s] Enter channelNum[%u] rankId[%u]", __func__, channelNum, rankId_);

    HcclResult firstErr = HCCL_SUCCESS;
    u32 invalidHandleCnt = 0;
    u32 failedCnt = 0;

    for (uint32_t i = 0; i < channelNum; ++i) {
        (void)DestroyOneChannel(channels[i], i, firstErr, invalidHandleCnt, failedCnt);
    }

    if (firstErr != HCCL_SUCCESS) {
        u32 destroyedCnt = channelNum - invalidHandleCnt - failedCnt;
        HCCL_ERROR(
            "[%s] finished with errors, total[%u] destroyed[%u] failed[%u] invalidHandle[%u] firstErr[%d].", __func__,
            channelNum, destroyedCnt, failedCnt, invalidHandleCnt, static_cast<s32>(firstErr));
        return firstErr;
    }
    return HCCL_SUCCESS;
}

HcclResult MyRank::WaitForChannelsReady(
    const HcclChannelDesc* channelDescs, ChannelHandle* channelHandles, uint32_t channelNum, CommEngine engine)
{
    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();

    HCCL_INFO(
        "[%s] start waiting channels ready, channelNum[%u], timeout[%lld]sec", __func__, channelNum, timeout.count());

    std::vector<int32_t> statusVec(channelNum, hcomm::HCOMM_CHANNEL_STATUS_CONNECTING);
    int32_t* statusList = statusVec.data();
    uint32_t retryCount = 0; // 轮询次数

    auto printChannelErrors = [&](int64_t elapsed) {
        // TLS 探测与错误表打印均按 HcommChannelLinkStatus 语义解读 statusList
        std::vector<Hccl::TlsStatus> tlsStatusList(channelNum, Hccl::TlsStatus::UNKNOWN);
        GetAbnormalChannelTlsStatus(channelDescs, statusList, channelNum, tlsStatusList);
        logger::ChannelLogger::PrintChannelErrorDetails(
            rankId_, channelNum, channelDescs, channelHandles, statusList, static_cast<uint64_t>(elapsed),
            tlsStatusList.data());
    };

    while (true) {
        HcclResult ret = static_cast<HcclResult>(HcommChannelGetStatus(channelHandles, channelNum, statusList));
        // HcommChannelGetStatus 的返回码只表示接口自身调用结果，批次结论由 statusList 归类得出
        if (ret == HCCL_SUCCESS) {
            ret = MyRankUtils::ClassifyChannelStatus(statusList, channelNum);
        }

        // 1. 检查超时
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_ERROR(
                "[%s][%s] wait channel ready timeout after %lld sec, channelNum[%u], elapsed[%lld]ms, retryCount[%u]",
                LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_TIMEOUT.c_str(), timeout.count(), channelNum, elapsed,
                retryCount);
            RPT_INPUT_ERR(
                true, "EI0006", std::vector<std::string>({"reason"}),
                std::vector<std::string>({GET_SOCKET_TIMEOUT_REASON_CLOSE_DETECT}));
            printChannelErrors(elapsed);
            StartDcaDetect(channelDescs, statusList, channelNum, engine);
            return HCCL_E_TIMEOUT;
        }

        // 2. 处理重试（去除频繁的重试日志，一秒可能重试上千次）
        if (ret == HCCL_E_AGAIN) {
            retryCount++;
            continue;
        }

        // 3. 处理资源不足（属于可预期回退场景）
        if (ret == HCCL_E_UNAVAIL) {
            HCCL_WARNING(
                "[%s] channel connect resource unavailable, channelNum[%u], retryCount[%u]", __func__, channelNum,
                retryCount);
            return ret;
        }

        // 4. 处理失败
        if (ret != HCCL_SUCCESS) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_ERROR(
                "[%s] channel connect failed, channelNum[%u], ret[%d], elapsed[%lld]ms, retryCount[%u]", __func__,
                channelNum, ret, elapsed, retryCount);
            printChannelErrors(elapsed);
            StartDcaDetect(channelDescs, statusList, channelNum, engine);
            return ret;
        }

        // 5. 正常情况：所有通道连接成功
        auto elapsed
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                  .count();
        HCCL_INFO(
            "[%s] all channels connected successfully, channelNum[%u], elapsed[%lld]ms, retryCount[%u]", __func__,
            channelNum, elapsed, retryCount);
        break;
    }
    return HCCL_SUCCESS;
}

void MyRank::StartDcaDetect(
    const HcclChannelDesc* channelDescs, const int32_t* statusList, uint32_t channelNum, CommEngine engine)
{
    // 收集仍在建链、建链失败或建链超时的对端
    std::unordered_map<uint32_t, uint32_t> failedRanks;
    CollectFailedRanks(channelDescs, statusList, channelNum, failedRanks);
    StartDcaDetectByFailedRanks(channelDescs, channelNum, failedRanks, engine);
}

void MyRank::StartDcaDetectByFailedRanks(
    const HcclChannelDesc* channelDescs, uint32_t channelNum, const std::unordered_map<uint32_t, uint32_t>& failedRanks,
    CommEngine engine)
{
    if (channelDescs == nullptr || channelNum == 0) {
        HCCL_ERROR(
            "[%s] invalid param, channelDescs[%p], channelNum[%u], skip DCA detect", __func__, channelDescs,
            channelNum);
        return;
    }
    if (failedRanks.empty()) {
        HCCL_WARNING("[%s] failedRanks is empty, skip DCA detect", __func__);
        return;
    }
    DevType devType = DevType::DEV_TYPE_COUNT;
    HcclResult devRet = hrtGetDeviceType(devType);
    if (devRet != HCCL_SUCCESS || devType != DevType::DEV_TYPE_950) {
        HCCL_WARNING(
            "[%s] device gate not pass, ret[%d], deviceType[%d], skip DCA detect", __func__, devRet,
            static_cast<int32_t>(devType));
        return;
    }
    std::vector<hcomm::DcaPeerInfo> peers;
    if (CollectDcaPeerInfo(channelDescs, failedRanks, peers, engine) != HCCL_SUCCESS || peers.empty()) {
        HCCL_WARNING("[%s] collect dca peer info failed, skip DCA detect", __func__);
        return;
    }

    std::vector<hcomm::LocalIpInfo> localIps;
    if (CollectDcaLocalInfo(localIps, engine) != HCCL_SUCCESS || localIps.empty()) {
        HCCL_WARNING("[%s] collect dca local info failed, skip DCA detect", __func__);
        return;
    }

    uint32_t localDevPhyId = UINT32_MAX;
    if (hrtGetDevicePhyIdByIndex(static_cast<u32>(devLogicId_), localDevPhyId) != HCCL_SUCCESS) {
        HCCL_WARNING("[%s] get device phyId failed, devLogicId[%d], skip DCA detect", __func__, devLogicId_);
        return;
    }

    std::string localNetInstId;
    GetNetInstId(rankId_, localNetInstId);

    HCCL_INFO(
        "[%s] start detect, localDevPhyId[%u], localNetInstId[%s], localIps[%zu], failed peers[%zu]", __func__,
        localDevPhyId, localNetInstId.c_str(), localIps.size(), peers.size());

    hccl::CollCommMgr::GetInstance()
        .GetDetectConnectionAnomalies(devLogicId_)
        .AddIpQueue(devLogicId_, localDevPhyId, localNetInstId, peers, localIps);
}

HcclResult MyRank::GetNetInstId(uint32_t remoteRank, std::string& netInstId)
{
    // 参考集群心跳：用 netLayer=0 的 netInstance ID 标记机器，相当于server ID
    CHK_PTR_NULL(rankGraph_);
    RankGraphV2* rankGraphV2 = dynamic_cast<RankGraphV2*>(rankGraph_);
    if (rankGraphV2 == nullptr) {
        HCCL_WARNING("[%s] rankGraph is not RankGraphV2, GetNetInstId not supported", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    return rankGraphV2->GetNetInstId(remoteRank, netInstId);
}

void MyRank::CollectFailedRanks(
    const HcclChannelDesc* channelDescs, const int32_t* statusList, uint32_t channelNum,
    std::unordered_map<uint32_t, uint32_t>& failedRanks)
{
    if (channelDescs == nullptr || statusList == nullptr || channelNum == 0) {
        HCCL_ERROR(
            "[%s] invalid param, channelDescs[%p], statusList[%p], channelNum[%u]", __func__, channelDescs, statusList,
            channelNum);
        return;
    }
    for (uint32_t i = 0; i < channelNum; ++i) {
        if (statusList[i] == hcomm::HCOMM_CHANNEL_STATUS_CONNECTING
            || statusList[i] == hcomm::HCOMM_CHANNEL_STATUS_FAILED
            || statusList[i] == hcomm::HCOMM_CHANNEL_STATUS_TIMEOUT) {
            failedRanks.emplace(channelDescs[i].remoteRank, i);
        }
    }
}

HcclResult MyRank::CollectDcaPeerInfo(
    const HcclChannelDesc* channelDescs, const std::unordered_map<uint32_t, uint32_t>& failedRanks,
    std::vector<hcomm::DcaPeerInfo>& peers, CommEngine engine)
{
    for (const auto& failedRank : failedRanks) {
        uint32_t remoteRank = failedRank.first;
        uint32_t descIdx = failedRank.second;
        const auto& desc = channelDescs[descIdx];
        EndpointLocType locType = desc.remoteEndpoint.loc.locType;
        Hccl::IpAddress remoteIp;
        if (CommAddrToIpAddress(desc.remoteEndpoint.commAddr, remoteIp) != HCCL_SUCCESS) {
            continue; // 取不到 IP 跳过（异常防御）
        }
        Hccl::IpAddress localIp;
        (void)CommAddrToIpAddress(desc.localEndpoint.commAddr, localIp);

        uint32_t remotePort = GetDcaPort(remoteRank, remoteIp, locType, engine);

        uint32_t remoteDevPhyId = 0;
        if (locType == ENDPOINT_LOC_TYPE_HOST) {
            (void)rankGraph_->GetDeviceId(remoteRank, &remoteDevPhyId);
        } else {
            remoteDevPhyId = desc.remoteEndpoint.loc.device.devPhyId;
        }
        // 机器标识用 netLayer0 NetInstId（机器唯一）：本端传本机 rank，对端传 remoteRank
        hcomm::DcaPeerInfo peer{};
        peer.remoteRankId = remoteRank;
        peer.remoteDevPhyId = remoteDevPhyId;
        peer.remoteIp = remoteIp;
        peer.remotePort = remotePort;
        peer.localIp = localIp;
        peer.localEndpoint = desc.localEndpoint;
        peer.remoteEndpoint = desc.remoteEndpoint;
        // 对端机器标识（netLayer0 NetInstId，机器唯一）；本端标识由 InjectDetectBatch 一次获取传 AddIpQueue
        std::string remoteNetInstId;
        (void)GetNetInstId(remoteRank, remoteNetInstId);
        (void)snprintf_s(peer.remoteNetInstId, DEST_MAX_LEN, DEST_MAX_LEN - 1, "%.127s", remoteNetInstId.c_str());
        peers.push_back(peer);
    }
    return HCCL_SUCCESS;
}

uint32_t MyRank::GetDcaPort(uint32_t rank, const Hccl::IpAddress& ip, EndpointLocType locType, CommEngine engine)
{
    uint32_t port = 0;
    // device 侧按 (rank, IP) 两级端口表查询，host/CPU 引擎走 rank 级查询
    HcclResult ret = GetListenPortByAddr(rank, ip, locType, engine, &port);
    if (ret != HCCL_SUCCESS || port > Hccl::MAX_VALUE_TCPPORT) {
        HCCL_WARNING(
            "[%s] Get listen port of Rank[%u] ip[%s] locType[%d] failed, ret[%d], use default port", __func__, rank,
            ip.GetIpStr().c_str(), static_cast<int>(locType), ret);
        port = (locType == ENDPOINT_LOC_TYPE_HOST) ? Hccl::DEFAULT_LISTENING_PORT : Hccl::DEFAULT_VALUE_TCPPORT;
    }
    return port;
}

HcclResult MyRank::CollectEndpointDescs(std::vector<EndpointDesc>& descs) const
{
    descs.clear();
    CHK_PTR_NULL(rankGraph_);
    RankGraphV2* rankGraphV2 = dynamic_cast<RankGraphV2*>(rankGraph_);
    if (rankGraphV2 == nullptr) {
        HCCL_WARNING("[%s] rankGraph is not RankGraphV2, collect endpoint descs not supported", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum = 0;
    HcclResult ret = rankGraphV2->GetNetLayers(&netLayers, &netLayerNum);
    if (ret != HCCL_SUCCESS || netLayers == nullptr || netLayerNum == 0) {
        HCCL_ERROR("[%s] GetNetLayers failed, ret[%d]", __func__, ret);
        return HCCL_E_NOT_FOUND;
    }
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        uint32_t netLayer = netLayers[layerIdx];
        uint32_t* topoInsts = nullptr;
        uint32_t topoInstNum = 0;
        ret = rankGraphV2->GetTopoInstsByLayer(netLayer, &topoInsts, &topoInstNum);
        if (ret != HCCL_SUCCESS || topoInsts == nullptr) {
            HCCL_ERROR("[%s] GetTopoInstsByLayer failed, netLayer[%u], ret[%d]", __func__, netLayer, ret);
            continue;
        }
        for (uint32_t topoInstIdx = 0; topoInstIdx < topoInstNum; ++topoInstIdx) {
            uint32_t descNum = 0;
            ret = rankGraphV2->GetEndpointNum(netLayer, topoInsts[topoInstIdx], &descNum);
            if (ret != HCCL_SUCCESS || descNum == 0) {
                HCCL_ERROR(
                    "[%s] GetEndpointNum failed, netLayer[%u] topoInstId[%u], ret[%d], descNum[%u]", __func__, netLayer,
                    topoInsts[topoInstIdx], ret, descNum);
                continue;
            }
            std::vector<EndpointDesc> layerDescs(descNum);
            ret = rankGraphV2->GetEndpointDesc(netLayer, topoInsts[topoInstIdx], &descNum, layerDescs.data());
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[%s] GetEndpointDesc failed, netLayer[%u] topoInstId[%u], ret[%d]", __func__, netLayer,
                    topoInsts[topoInstIdx], ret);
                continue;
            }
            // GetEndpointDesc 写回的 descNum 为实际填充条数，按实际条数截断
            if (descNum < layerDescs.size()) {
                layerDescs.resize(descNum);
            }
            descs.insert(descs.end(), layerDescs.begin(), layerDescs.end());
        }
    }
    return HCCL_SUCCESS;
}

HcclResult MyRank::CollectDcaLocalInfo(std::vector<hcomm::LocalIpInfo>& localIps, CommEngine engine)
{
    // 收集本 rank 全量建链地址和端口（按 IP 去重）
    localIps.clear();
    std::vector<EndpointDesc> descs;
    CHK_RET(CollectEndpointDescs(descs));

    std::set<Hccl::IpAddress> knownIps;
    for (const auto& desc : descs) {
        EndpointLocType locType = desc.loc.locType;
        if (locType != ENDPOINT_LOC_TYPE_DEVICE && locType != ENDPOINT_LOC_TYPE_HOST) {
            continue;
        }
        Hccl::IpAddress ip;
        if (CommAddrToIpAddress(desc.commAddr, ip) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[%s] CommAddrToIpAddress failed, locType[%d], protocol[%d]", __func__, static_cast<int>(locType),
                static_cast<int32_t>(desc.protocol));
            continue;
        }
        if (knownIps.count(ip) == 0) {
            knownIps.insert(ip);
            hcomm::LocalIpInfo info;
            info.ip = ip;
            info.port = GetDcaPort(rankId_, info.ip, locType, engine);
            info.endpoint = desc;
            localIps.push_back(info);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult MyRank::ConfigSqDepthByExpansionMode(CommEngine engine, HcommChannelDesc& hcommDesc) const
{
    const u32 configuredSqDepth = config_.GetConfigSqDepth();
    if (configuredSqDepth != HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET) {
        const CommProtocol remoteProtocol = hcommDesc.remoteEndpoint.protocol;
        if (engine == COMM_ENGINE_AIV
            && (remoteProtocol == COMM_PROTOCOL_UBC_TP || remoteProtocol == COMM_PROTOCOL_UBC_CTP
                || remoteProtocol == COMM_PROTOCOL_UBG)) {
            hcommDesc.ubAttr.sqDepth = configuredSqDepth;
            return HCCL_SUCCESS;
        } else {
            HCCL_WARNING(
                "[%s] configured sqDepth[%u] is not supported when engine[%s] protocol[%s].", __func__,
                configuredSqDepth, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),
                MyRankUtils::GetCommProtocolEnumStr(remoteProtocol).c_str());
        }
    }

    constexpr u32 CCU_MS_MODE_DEPTH = 128;
    constexpr u32 CCU_SCHED_MODE_DEPTH = 16;
    if (engine == COMM_ENGINE_CCU) {
        if (opExpansionMode_ == CCU_MS_MODE) {
            hcommDesc.ubAttr.sqDepth = CCU_MS_MODE_DEPTH;
        } else if (opExpansionMode_ == CCU_SCHED_MODE) {
            hcommDesc.ubAttr.sqDepth = CCU_SCHED_MODE_DEPTH;
        } else {
            HCCL_ERROR("[%s] unexpected op expansion mode[%u] for ccu,", __func__, opExpansionMode_);
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

void MyRank::LogChannelCreationInfo(
    CommEngine engine, const std::string& commTag, const HcclChannelDesc* channelDescs, uint32_t channelNum,
    const ChannelHandle* hostChannelHandleList) const
{
    for (u32 i = 0; i < channelNum; ++i) {
        u32 remoteRank = channelDescs[i].remoteRank;
        HcclCommDfx::AddChannelRemoteRankId(commTag, hostChannelHandleList[i], remoteRank);
        // 打印UB通道建链信息
        if (channelDescs[i].localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_DEVICE
            && channelDescs[i].remoteEndpoint.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
            HCCL_CONFIG_DEBUG(
                HCCL_RES,
                "create channel info:channel handle[%s] comm tag[%s] protocol[%s]"
                " local rank[%u] local dev phyid[%u] remote rank[%u] remote dev phyid[%u] engine[%s]",
                std::to_string(hostChannelHandleList[i]).c_str(), commTag.c_str(),
                MyRankUtils::GetCommProtocolEnumStr(channelDescs[i].localEndpoint.protocol).c_str(), rankId_,
                channelDescs[i].localEndpoint.loc.device.devPhyId, remoteRank,
                channelDescs[i].remoteEndpoint.loc.device.devPhyId,
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        } else {
            HCCL_CONFIG_DEBUG(
                HCCL_RES,
                "create channel info:channel handle[%s] comm tag[%s] protocol[%s]"
                " local rank[%u] remote rank[%u] engine[%s]",
                std::to_string(hostChannelHandleList[i]).c_str(), commTag.c_str(),
                MyRankUtils::GetCommProtocolEnumStr(channelDescs[i].localEndpoint.protocol).c_str(), rankId_,
                remoteRank, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        }
    }
}

HcclResult MyRank::FinalizeChannelsByEngine(
    CommEngine engine, const std::string& commTag, uint32_t channelNum, ChannelHandle* hostChannelHandleList,
    const ChannelHandle* deviceChannelHandleList, ChannelHandle* channelHandles)
{
    if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
        // 输出 handle = device 句柄（ctx 模式为 ctxPtr）
        CHK_SAFETY_FUNC_RET(memcpy_s(
            channelHandles, channelNum * sizeof(ChannelHandle), deviceChannelHandleList,
            channelNum * sizeof(ChannelHandle)));

        // ns recovery：channelHandles = device 句柄，hostChannelHandleList_ = host Channel*
        nsRecoveryProcessor_->AddNsRecoveryData(engine, channelHandles, hostChannelHandleList, channelNum, commTag);

        return HCCL_SUCCESS;
    }

    if (engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CCU || engine == COMM_ENGINE_AIV) {
        CHK_SAFETY_FUNC_RET(memcpy_s(
            channelHandles, channelNum * sizeof(ChannelHandle), hostChannelHandleList,
            channelNum * sizeof(ChannelHandle)));
        return HCCL_SUCCESS;
    }

    HCCL_ERROR(
        "[MyRank][%s] unsupported comm engine[%s].", __func__,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
    return HCCL_E_NOT_SUPPORT;
}

// 记录 channel kernel 下发：建链后把本 comm 的 handle 列表记录到 CollCommAicpu，
// 并按 remoteRank 注册 per-comm DFX 表
static HcclResult LaunchRunAicpuCommRecordChannels(
    const ChannelHandle* deviceChannelHandleList, const u32* remoteRankList, u32 channelNum, const std::string& commTag,
    aclrtBinHandle binHandle)
{
    CHK_PTR_NULL(deviceChannelHandleList);
    CHK_PTR_NULL(remoteRankList);

    hccl::DeviceMem deviceChannelList = hccl::DeviceMem::alloc(channelNum * sizeof(ChannelHandle));
    CHK_PTR_NULL(deviceChannelList.ptr());
    CHK_RET(hrtMemSyncCopy(
        deviceChannelList.ptr(), channelNum * sizeof(ChannelHandle), deviceChannelHandleList,
        channelNum * sizeof(ChannelHandle), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    hccl::DeviceMem deviceRemoteRankList = hccl::DeviceMem::alloc(channelNum * sizeof(u32));
    CHK_PTR_NULL(deviceRemoteRankList.ptr());
    CHK_RET(hrtMemSyncCopy(
        deviceRemoteRankList.ptr(), channelNum * sizeof(u32), remoteRankList, channelNum * sizeof(u32),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    HcclAicpuCommRecordChannelsRes channelParam{};
    channelParam.deviceHandleList = reinterpret_cast<const uint64_t*>(deviceChannelList.ptr());
    channelParam.remoteRankList = reinterpret_cast<const uint32_t*>(deviceRemoteRankList.ptr());
    channelParam.num = channelNum;
    size_t copyLen = sizeof(channelParam.commTag) - 1;
    if (commTag.size() < copyLen) {
        copyLen = commTag.size();
    }
    CHK_SAFETY_FUNC_RET(memcpy_s(channelParam.commTag, sizeof(channelParam.commTag), commTag.c_str(), copyLen));
    channelParam.commTag[copyLen] = '\0';

    hccl::Stream localStream = hccl::Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    constexpr u32 aicpuStreamMode = 1;
    CHK_RET(hrtStreamSetMode(localStream.ptr(), aicpuStreamMode));

    hccl::DeviceMem paramAddr = hccl::DeviceMem::alloc(sizeof(HcclAicpuCommRecordChannelsRes));
    CHK_PTR_NULL(paramAddr.ptr());
    CHK_RET(hrtMemSyncCopy(
        paramAddr.ptr(), sizeof(HcclAicpuCommRecordChannelsRes), &channelParam, sizeof(HcclAicpuCommRecordChannelsRes),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    uint64_t context = reinterpret_cast<uint64_t>(paramAddr.ptr());

    uint32_t envTimeout = Hccl::EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut();
    s32 timeOut = envTimeout + 25 > std::numeric_limits<u16>::max() ? std::numeric_limits<u16>::max() :
                                                                      envTimeout + 25; // 多25s，避免超时

    CHK_RET(hccl::AicpuAclKernelLaunch(
        localStream.ptr(), static_cast<void*>(&context), sizeof(context), binHandle, "RunAicpuCommRecordChannels", true,
        static_cast<u16>(timeOut)));
    CHK_RET(hcclStreamSynchronize(localStream.ptr(), timeOut));

    HCCL_INFO("[%s] RunAicpuCommRecordChannels launch success, channelNum[%u].", __func__, channelNum);
    return HCCL_SUCCESS;
}

// 复用通道追加内存后的 device 侧补齐：只推「非本次新建 且 本次追加过内存」的通道——
// 新建通道由建链 kernel 连同地址表一起下发，未追加内存的通道 device 侧无需变更。
// newChannels 的 first 是本批通道下标；allHandles[i] 即写入 hcommDescs[i].memHandleNum 的同一个值
static HcclResult LaunchRunAicpuChannelMemUpdate(
    const ChannelHandle* hostChannelHandleList, const ChannelHandle* deviceChannelHandleList, u32 channelNum,
    const std::vector<std::pair<u32, u32>>& newChannels, const std::vector<std::vector<MemHandle>>& allHandles,
    aclrtBinHandle binHandle)
{
    CHK_PTR_NULL(hostChannelHandleList);
    CHK_PTR_NULL(deviceChannelHandleList);

    std::vector<bool> isNewChannel(channelNum, false);
    for (const auto& it : newChannels) {
        if (it.first < channelNum) {
            isNewChannel[it.first] = true;
        }
    }

    // 第一轮：逐通道问所需字节数并累计总长，顺带记下各自在段内的偏移
    std::vector<char> packHost;
    std::vector<uint32_t> packSizeList(channelNum, 0);
    std::vector<uint32_t> packOffsetList(channelNum, 0);
    u32 totalSize = 0;
    for (u32 i = 0; i < channelNum; ++i) {
        if (isNewChannel[i] || allHandles[i].size() <= 1) {
            continue;
        }
        uint32_t packSize = 0;
        CHK_RET(static_cast<HcclResult>(HcommChannelGetMemUpdatePack(hostChannelHandleList[i], nullptr, &packSize)));
        if (packSize == 0) {
            continue;
        }
        packSizeList[i] = packSize;
        packOffsetList[i] = totalSize;
        totalSize += packSize;
    }
    if (totalSize == 0) {
        return HCCL_SUCCESS;
    }

    // 第二轮：把各通道条目填进同一段 host 缓冲
    packHost.resize(totalSize);
    for (u32 i = 0; i < channelNum; ++i) {
        if (packSizeList[i] == 0) {
            continue;
        }
        uint32_t packSize = packSizeList[i];
        CHK_RET(static_cast<HcclResult>(
            HcommChannelGetMemUpdatePack(hostChannelHandleList[i], packHost.data() + packOffsetList[i], &packSize)));
    }

    hccl::Stream localStream = hccl::Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    constexpr u32 aicpuStreamMode = 1;
    CHK_RET(hrtStreamSetMode(localStream.ptr(), aicpuStreamMode));

    hccl::DeviceMem packDev = hccl::DeviceMem::alloc(totalSize);
    CHK_PTR_NULL(packDev.ptr());
    CHK_RET(hrtMemSyncCopy(
        packDev.ptr(), totalSize, packHost.data(), totalSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    const uint64_t packBase = reinterpret_cast<uint64_t>(packDev.ptr());

    std::vector<uint64_t> deviceHandleList(channelNum, 0);
    std::vector<uint64_t> packAddrList(channelNum, 0);
    for (u32 i = 0; i < channelNum; ++i) {
        deviceHandleList[i] = deviceChannelHandleList[i];
        packAddrList[i] = packSizeList[i] == 0 ? 0 : packBase + packOffsetList[i];
    }

    hccl::DeviceMem deviceHandleListDev = hccl::DeviceMem::alloc(channelNum * sizeof(uint64_t));
    CHK_PTR_NULL(deviceHandleListDev.ptr());
    CHK_RET(hrtMemSyncCopy(
        deviceHandleListDev.ptr(), channelNum * sizeof(uint64_t), deviceHandleList.data(),
        channelNum * sizeof(uint64_t), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    hccl::DeviceMem packAddrListDev = hccl::DeviceMem::alloc(channelNum * sizeof(uint64_t));
    CHK_PTR_NULL(packAddrListDev.ptr());
    CHK_RET(hrtMemSyncCopy(
        packAddrListDev.ptr(), channelNum * sizeof(uint64_t), packAddrList.data(), channelNum * sizeof(uint64_t),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    hccl::DeviceMem packSizeListDev = hccl::DeviceMem::alloc(channelNum * sizeof(uint32_t));
    CHK_PTR_NULL(packSizeListDev.ptr());
    CHK_RET(hrtMemSyncCopy(
        packSizeListDev.ptr(), channelNum * sizeof(uint32_t), packSizeList.data(), channelNum * sizeof(uint32_t),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    HcclAicpuChannelMemUpdateRes channelParam{};
    channelParam.deviceHandleList = reinterpret_cast<const uint64_t*>(deviceHandleListDev.ptr());
    channelParam.packAddrList = reinterpret_cast<const uint64_t*>(packAddrListDev.ptr());
    channelParam.packSizeList = reinterpret_cast<const uint32_t*>(packSizeListDev.ptr());
    channelParam.num = channelNum;

    hccl::DeviceMem paramAddr = hccl::DeviceMem::alloc(sizeof(HcclAicpuChannelMemUpdateRes));
    CHK_PTR_NULL(paramAddr.ptr());
    CHK_RET(hrtMemSyncCopy(
        paramAddr.ptr(), sizeof(HcclAicpuChannelMemUpdateRes), &channelParam, sizeof(HcclAicpuChannelMemUpdateRes),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    uint64_t context = reinterpret_cast<uint64_t>(paramAddr.ptr());

    uint32_t envTimeout = Hccl::EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut();
    s32 timeOut = envTimeout + 25 > std::numeric_limits<u16>::max() ? std::numeric_limits<u16>::max() :
                                                                      envTimeout + 25; // 多25s，避免超时

    CHK_RET(hccl::AicpuAclKernelLaunch(
        localStream.ptr(), static_cast<void*>(&context), sizeof(context), binHandle, "RunAicpuChannelMemUpdate", true,
        static_cast<u16>(timeOut)));
    CHK_RET(hcclStreamSynchronize(localStream.ptr(), timeOut));

    HCCL_INFO(
        "[%s] RunAicpuChannelMemUpdate launch success, channelNum[%u], totalSize[%u].", __func__, channelNum,
        totalSize);
    return HCCL_SUCCESS;
}

HcclResult MyRank::CreateChannels(
    CommEngine engine, const std::string& commTag, const HcclChannelDesc* channelDescs, uint32_t channelNum,
    ChannelHandle* channelHandles)
{
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channelHandles);
    CHK_PRT_RET(channelNum == 0, HCCL_ERROR("[%s] invalid param: channelNum is zero", __func__), HCCL_E_PARA);

    HCCL_INFO(
        "[CreateChannels][Enter] engine[%s] commTag[%s] channelNum[%u] rankId[%u]",
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), commTag.c_str(), channelNum, rankId_);

    // 参数检查
    CHK_RET(CheckChannelParam(engine, channelDescs, channelNum));

    std::vector<ChannelHandle> hostChannelHandles(channelNum);
    ChannelHandle* hostChannelHandleList = hostChannelHandles.data();
    // 与 hostChannelHandles 一一对应：device 可见句柄由 CreateChannel 出参回填，AICPU 引擎的输出/DFX 下发都用它
    std::vector<ChannelHandle> deviceChannelHandles(channelNum);
    ChannelHandle* deviceChannelHandleList = deviceChannelHandles.data();

    auto& rdmaConfig = Hccl::EnvConfig::GetInstance().GetRdmaConfig();
    std::vector<HcommChannelDesc> hcommDescs(channelNum);
    std::vector<std::vector<MemHandle>> allHandles(channelNum);
    RoceChannelDescConfigurator roceDescConfigurator(channelNum);
    for (u32 i = 0; i < channelNum; ++i) {
        hcommDescs[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDescs[i], config_);
        hcommDescs[i].roceAttr.qpThreshold = rdmaConfig.GetRdmaMultiQpThreshold();
        CHK_RET(ConfigSqDepthByExpansionMode(engine, hcommDescs[i]));
        CHK_RET(roceDescConfigurator.FillRoceSrcPortList(channelDescs[i], i, hcommDescs[i]));
    }

    auto start = std::chrono::steady_clock::now();
    std::string socketTag = commTag + "_engine_" + std::to_string(engine);
    CHK_RET(BatchCreateSockets(channelDescs, channelNum, socketTag, hcommDescs, engine));
    CHK_RET_UNAVAIL(BatchCreateChannels(
        engine, channelDescs, channelNum, hcommDescs, hostChannelHandleList, deviceChannelHandleList, allHandles));

    // 锁内快照本次新建列表：connect 阶段不再持 channelIndexMtx_，避免长耗时 IO 阻塞
    // Query/Destroy；回滚时基于快照重新取锁清理，保证 newChannels_ 读写均在锁内
    std::vector<std::pair<u32, u32>> newChannelsSnapshot;
    {
        std::lock_guard<std::mutex> lock(channelIndexMtx_);
        newChannelsSnapshot = newChannels_;
    }

    if (!newChannelsSnapshot.empty()) {
        HcclResult connRet = WaitForChannelsReady(channelDescs, hostChannelHandleList, channelNum, engine);
        if (connRet == HCCL_E_UNAVAIL && engine == COMM_ENGINE_CCU) {
            // CCU 场景额外回滚本次新建的 channel，避免资源残留
            HCCL_RUN_WARNING(
                "[%s] WaitForChannelsReady is unsuccessful[%d], engine[%s], new channels num[%u]", __func__, connRet,
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), newChannelsSnapshot.size());
            std::lock_guard<std::mutex> lock(channelIndexMtx_);
            HcclResult destroyRet = DestroyNewChannels(engine, channelDescs, newChannelsSnapshot);
            if (destroyRet != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[%s] DestroyNewChannels failed[%d] during rollback, connRet[%d], "
                    "residual newChannels[%zu] may leak.",
                    __func__, destroyRet, connRet, newChannels_.size());
            }
        }
        CHK_RET_UNAVAIL(connRet);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        HCCL_RUN_INFO(
            "[MyRank][CreateChannels] CreateChannels Time Elapsed [%lld]us, channelNum [%u]", duration, channelNum);
    }

    // 借用channel内保存的socket，完成一致性校验必要的数据交换
    HcclResult ret = BatchExchangeAndCheckConsistency(channelDescs, hostChannelHandleList, channelNum);
    // 一致性校验超时/网络异常（host 同步收发失败映射为 E_TCP_CONNECT）：触发故障根节点探测
    if (ret == HCCL_E_TIMEOUT || ret == HCCL_E_TCP_CONNECT) {
        std::unordered_map<uint32_t, uint32_t> failedRanks; // remoteRank → 首条未完成 channel 下标
        for (uint32_t i = 0; i < channelNum; ++i) {
            HcommSocket rawSocket = hcommDescs[i].socket;
            Hccl::Socket* socket = static_cast<Hccl::Socket*>(rawSocket);
            if (socket == nullptr || socket->GetSocketStatus() != Hccl::SocketStatus::OK) {
                failedRanks.emplace(channelDescs[i].remoteRank, i);
            }
        }
        StartDcaDetectByFailedRanks(channelDescs, channelNum, failedRanks, engine);
    }
    CHK_RET(ret);

    // 添加初始化时进行填表
    LogChannelCreationInfo(engine, commTag, channelDescs, channelNum, hostChannelHandleList);

    // device DFX 填表：数据源为描述符的 remoteRank
    if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
        // 全复用批次不轮询，init kernel 无触发机会（ctx 全零，kernel 解引用会崩溃）；
        // 兜底不查链路状态、仅补发未就绪实体
        CHK_RET(static_cast<HcclResult>(HcommChannelEnsureDeviceReady(hostChannelHandleList, channelNum)));

        // 记录 kernel 依赖 g_configEnv（由 RunAicpuCommInit 写入），先确保 commInit 已执行
        if (!callbacks_.getAicpuCommState()) {
            HcclResult initRet = callbacks_.kernelLaunchAicpuCommInit();
            CHK_PRT_RET(
                initRet != HCCL_SUCCESS,
                HCCL_ERROR("[%s] kernelLaunchAicpuCommInit failed, return [%d].", __func__, initRet), initRet);
            callbacks_.setAicpuCommState(true);
        }
        // 建链后记录 channel：存档 handle 列表供后台线程 Clean 使用，并按 remoteRank 注册 per-comm DFX 表
        std::vector<u32> remoteRankList(channelNum);
        for (u32 i = 0; i < channelNum; ++i) {
            remoteRankList[i] = channelDescs[i].remoteRank;
        }
        CHK_RET(LaunchRunAicpuCommRecordChannels(
            deviceChannelHandleList, remoteRankList.data(), channelNum, commTag, binHandle_));

        // device 侧地址条目下发：复用槽位建链时该通道不下发建链包，本次追加的对端内存由本流程补齐
        CHK_RET(LaunchRunAicpuChannelMemUpdate(
            hostChannelHandleList, deviceChannelHandleList, channelNum, newChannelsSnapshot, allHandles, binHandle_));
    }

    return FinalizeChannelsByEngine(
        engine, commTag, channelNum, hostChannelHandleList, deviceChannelHandleList, channelHandles);
}

HcclResult MyRank::ChannelGetHcclBuffer(ChannelHandle channel, void** buffer, uint64_t* size)
{
    CHK_PTR_NULL(buffer);
    CHK_PTR_NULL(size);

    u32 memNum = 0;
    CommMem* remoteMem = nullptr;
    char** memTags = nullptr;
    CHK_RET(static_cast<HcclResult>(HcommChannelGetRemoteMems(channel, &memNum, &remoteMem, &memTags)));
    if (memNum > 0) {
        CHK_PTR_NULL(remoteMem);
        // AicpuTsHccsChannel不使用memTag，返回为空，默认索引0为cclBuffer
        if (memTags == nullptr) {
            *buffer = remoteMem[0].addr;
            *size = remoteMem[0].size;
            HCCL_INFO("[%s] Found HcclBuffer : addr=%p, size=%llu", __func__, *buffer, *size);
            return HCCL_SUCCESS;
        }
        for (u32 i = 0; i < memNum; ++i) {
            std::string tag = memTags[i];
            if (tag == "HcclBuffer") {
                *buffer = remoteMem[i].addr;
                *size = remoteMem[i].size;
                HCCL_INFO("[%s] Found HcclBuffer : addr=%p, size=%llu", __func__, *buffer, *size);
                return HCCL_SUCCESS;
            }
            HCCL_INFO("[%s] Found %s : addr=%p, size=%llu", __func__, memTags[i], remoteMem[i].addr, remoteMem[i].size);
        }
    }
    HCCL_ERROR("[%s] HcclBuffer not found.", __func__);
    return HCCL_E_INTERNAL;
}

HcclResult
MyRank::ChannelGetRemoteMems(ChannelHandle channel, uint32_t* memNum, CommMem** remoteMem, char*** memTags) const
{
    CHK_PTR_NULL(remoteMem);
    CHK_PTR_NULL(memTags);
    CHK_PTR_NULL(memNum);
    CHK_RET(static_cast<HcclResult>(HcommChannelGetRemoteMems(channel, memNum, remoteMem, memTags)));
    // 添加空指针检查，防止返回的指针为空
    if (*memNum > 0) {
        CHK_PTR_NULL(*remoteMem);
        CHK_PTR_NULL(*memTags);
    }
    HCCL_INFO("[%s] success. memNum[%u]", __func__, *memNum);
    return HCCL_SUCCESS;
}

HcclResult MyRank::ChannelGetRemoteMems(
    ChannelHandle channel, uint32_t* memNum, CommMem** remoteMem, std::vector<std::string>& memTags) const
{
    CHK_PTR_NULL(remoteMem);
    CHK_PTR_NULL(memNum);
    char** rawTags = nullptr;
    CHK_RET(static_cast<HcclResult>(HcommChannelGetRemoteMems(channel, memNum, remoteMem, &rawTags)));
    // 添加空指针检查，防止返回的指针为空
    if (*memNum > 0) {
        CHK_PTR_NULL(*remoteMem);
        CHK_PTR_NULL(rawTags);
        memTags.reserve(*memNum);
        for (uint32_t i = 0; i < *memNum; ++i) {
            memTags.emplace_back(rawTags[i] == nullptr ? "" : rawTags[i]);
        }
    }
    HCCL_INFO("[%s] success. memNum[%u]", __func__, *memNum);
    return HCCL_SUCCESS;
}

std::vector<ChannelHandle> MyRank::GetAllChannelList()
{
    ChannelTable channelTable = rankPairMgr_->GetChannelTable();
    std::vector<ChannelHandle> channelList;
    for (const auto& rankPair : channelTable) {
        for (const auto& endPointPair : rankPair.second) {
            for (const auto& comEngines : endPointPair.second) {
                channelList.insert(channelList.end(), comEngines.second.begin(), comEngines.second.end());
            }
        }
    }

    return channelList;
}

void MyRank::SetKfcControlTransfer(
    std::shared_ptr<HDCommunicate> kfcControlTransferH2D, std::shared_ptr<HDCommunicate> kfcStatusTransferD2H)
{
    if (nsRecoveryProcessor_ == nullptr) {
        HCCL_ERROR("[MyRank][SetKfcControlTransfer] nsRecoveryProcessor_ is null, cannot set KFC control transfer.");
        return;
    }
    nsRecoveryProcessor_->SetKfcControlTransfer(kfcControlTransferH2D, kfcStatusTransferD2H);
}

HcclResult MyRank::StopLaunch()
{
    HCCL_INFO("[NsRecovery][StopLaunch] MyRank::StopLaunch start!");
    auto ret = nsRecoveryProcessor_->StopLaunch();
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[NsRecovery][StopLaunch] MyRank::StopLaunch failed, ret = 0x%016llx", HCCL_ERROR_CODE(ret));
    }
    HCCL_INFO("[NsRecovery][StopLaunch] MyRank::StopLaunch success!");
    return ret;
}

HcclResult MyRank::Clean()
{
    HCCL_INFO("[NsRecovery][Clean] MyRank::Clean start!");
    auto channelList = GetAllChannelList();
    if (channelList.empty()) {
        HCCL_INFO("[NsRecovery][Clean] Channel list empty, No need to clean!");
        return HcclResult::HCCL_SUCCESS;
    }
    auto ret = ChannelProcess::ChannelClean(channelList.data(), channelList.size());
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[NsRecovery][Clean] MyRank::Clean failed, ret = 0x%016llx", HCCL_ERROR_CODE(ret));
        return ret;
    }

    ret = nsRecoveryProcessor_->Clean();
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[NsRecovery][Clean] MyRank::Clean failed, ret = 0x%016llx", HCCL_ERROR_CODE(ret));
        return ret;
    }

    HCCL_INFO("[NsRecovery][Clean] MyRank::Clean success!");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MyRank::Resume()
{
    HCCL_INFO("[NsRecovery][Resume] MyRank::Resume start!");
    auto channelList = GetAllChannelList();
    if (channelList.empty()) {
        HCCL_INFO("[NsRecovery][Resume] Resume list empty, No need to resume!");
        return HcclResult::HCCL_SUCCESS;
    }

    auto ret = ChannelProcess::ChannelResume(channelList.data(), channelList.size());
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[NsRecovery][Resume] MyRank::Resume failed, ret = 0x%016llx", HCCL_ERROR_CODE(ret));
        return ret;
    }

    ret = nsRecoveryProcessor_->Resume(binHandle_);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[NsRecovery][Resume] MyRank::Resume failed, ret = 0x%016llx", HCCL_ERROR_CODE(ret));
        return ret;
    }

    HCCL_INFO("[NsRecovery][Resume] MyRank::Resume success!");
    return HCCL_SUCCESS;
}

CollCommConfigConsistency& MyRank::GetCollCommConfigConsistency() { return collCommConfigConsistency_; }

} // namespace hccl
