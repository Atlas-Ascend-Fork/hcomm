/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include <cstdlib>
#include <fstream>
#include <limits.h>
#include <cstring>
#include "rank_info_detect_client.h"
#include "ip_address.h"
#include "bootstrap_ip.h"
#include "root_handle_v2.h"
#include "env_config/env_config_v2.h"
#include "host_buffer.h"
#include "binary_stream.h"
#include "hccp_peer_manager.h"
#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"
#include "host_socket_handle_manager.h"
#include "socket_manager.h"
#include "topo_addr_info.h"
#include "adapter_error_manager_pub.h"
#include "common/hccl_log_keywords_pub.h"
#include "network_api_exception.h"
#include "phy_topo_builder.h"
#include "preempt_port_manager_v2.h"
#include "json/json_api.h"

namespace Hccl {
namespace {
    constexpr u32 HOST_DATA_BASE_PORT = 60000;
    constexpr u32 HOST_DATA_PORT_COUNT = 15;
    // 无UB场景下level0的兜底netInstId前缀，仅用于日志辨识实例用途；
    // 下游识别兜底层一律依据pcie_fallback标记字段，不依赖此名称。
    // 后缀拼接本机host ip：同一服务器内rank共享同一level0实例（PCIe fullmesh），跨服务器实例隔离，
    // 与原始level0按server划分（server mac）的语义一致，跨机流量走上层网络（RoCE等）
    constexpr const char* PCIE_FALLBACK_NET_INST_ID = "l0_pcie_fallback";

    // 无UB场景兜底：探测不到UB时rootinfo中rank无level0，插入一个带本卡vnic地址的兜底level0，
    // 保证rankGraph按level0建图不抛异常；链路由RankGraphBuilder按PCIE兜底合成。
    // rank_addr_list需填入本卡vnic IP：SocketManager的端口map按各层rank_addr_list的addr索引，
    // 缺失会导致设备端口查询返回0，server随机监听、client连0端口，建链失败。
    HcclResult InsertPcieFallbackLevel0(std::string& localDevInfoJson, u32 devPhyId)
    {
        const std::string fallbackNetInstId
            = StringFormat("%s_%s", PCIE_FALLBACK_NET_INST_ID, GetBootstrapIp(devPhyId).GetIpStr().c_str());
        IpAddress vnicIp;
        bool vnicQueryOk = false;
        try {
            HrtRaSocketGetVnicIpInfos(devPhyId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, devPhyId, vnicIp);
            vnicQueryOk = true;
        } catch (const HcclException& e) {
            HCCL_ERROR(
                "[%s] get local vnic ip failed, pcie fallback level0 has no rank_addr entry, devPhyId[%u], reason[%s].",
                __func__, devPhyId, e.what());
        }
        return AddRootInfoPcieFallbackLevel0(localDevInfoJson, fallbackNetInstId, vnicQueryOk, vnicIp);
    }

    std::string QueryTopoFilePathByDevice()
    {
        const size_t bufSize = 1024;
        auto userDevId = HrtGetDevice();
        auto devPhyId = HrtGetDevicePhyIdByUserDevId(userDevId);
        std::vector<char> buffer(bufSize, '\0');
        int result = TopoAddrInfoGetTopoFilePath(devPhyId, buffer.data(), buffer.size());
        CHK_PRT_THROW(
            result != 0, HCCL_ERROR("[%s] Get topo file path failed.", __func__), InvalidParamsException,
            "Get topo file path failed.");
        return std::string(buffer.data());
    }

    HcclResult CheckTopoFilePath(const std::string& topoFilePath)
    {
        char resolvedPath[PATH_MAX] = {0};
        CHK_PRT_RET(
            realpath(topoFilePath.c_str(), resolvedPath) == nullptr,
            HCCL_ERROR("[%s] topo_file_path[%s] is not a valid real path", __func__, topoFilePath.c_str()),
            HCCL_E_PARA);
        return HCCL_SUCCESS;
    }

    std::string QueryRootInfoTopoFilePath()
    {
        std::string filePath = "/etc/hccl_rootinfo.json";
        std::string configJson{};
        std::string topoFilePath{};
        std::ifstream file(filePath);
        if (file.good()) {
            HcclResult parseRet = ParseRootInfoConfigFile(filePath, configJson);
            if (parseRet == HcclResult::HCCL_E_PARA) {
                THROW<InvalidParamsException>(StringFormat("[%s] ParseFileToJson failed with param error", __func__));
            } else if (parseRet != HcclResult::HCCL_SUCCESS) {
                THROW<InternalException>(StringFormat("[%s] ParseFileToJson failed, ret[%d]", __func__, parseRet));
            }
            HcclResult jsonRet = GetRootInfoTopoFilePathFromConfig(configJson, topoFilePath);
            if (jsonRet != HCCL_SUCCESS) {
                THROW<InvalidParamsException>(
                    StringFormat("error occurs when parser object of propName \"topo_file_path\""));
            }
        } else {
            topoFilePath = QueryTopoFilePathByDevice();
        }

        HcclResult topoRet = CheckTopoFilePath(topoFilePath);
        CHK_PRT_THROW(
            topoRet != HCCL_SUCCESS, HCCL_ERROR("[%s] CheckTopoFilePath failed, ret[%d]", __func__, topoRet),
            InvalidParamsException, "CheckTopoFilePath failed");
        return topoFilePath;
    }
} // namespace

HcclResult RankInfoDetectClient::Setup(RankTableInfo& rankTable)
{
    // 1. 构造localRankTable
    RankTableInfo localRankTable{};
    HcclResult ret = ConstructRankTable(localRankTable);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[RankInfoDetectClient::%s] ConstructRankTable failed, ret[%d].", __func__, ret), ret);

    // 若启用单卡多进程抢占端口则执行
    TRY_CATCH_RETURN(SocketManager::ServerInitAll(localRankTable.ranks[0]););
    ret = HostListenPortDetect(localRankTable.ranks[0]);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[RankInfoDetectClient::%s] HostListenPortDetect failed, ret[%d].", __func__, ret), ret);

    // 2. 连接root节点
    CHK_RET(Connect());

    // 3. 发送本端agentId和rankSize
    CHK_RET(SendAgentIdAndRankSize());

    // 4. 发送给root节点
    CHK_RET(SendLocalRankTable(localRankTable));

    // 5. 接收完整 rankTable
    CHK_RET(RecvRankTable());
    rankTable = rankTable_;
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectClient::WaitForNextBroadcast(RankTableInfo& rankTable, uint32_t expectedRankCount)
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);
    vector<char> rankInfoMsg{};
    HcclResult ret = RecvRankTableMsg(rankInfoMsg);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[RankInfoDetectClient::%s] RecvRankTableMsg failed, ret[%d].", __func__, ret),
        ret);
    ParseRankTable(rankInfoMsg);
    // 校验 rankTable 有效性，防止空表/0 rankCount 被当合法结果使用
    CHK_PRT_RET(
        rankTable_.rankCount == 0,
        HCCL_ERROR("[RankInfoDetectClient::%s] parsed rankTable rankCount is 0, invalid.", __func__), HCCL_E_INTERNAL);
    // Grow rankCount 变化：先用预期值校验，通过后再更新 rankSize_
    rankSize_ = expectedRankCount;
    ret = VerifyRankTable();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[RankInfoDetectClient::%s] VerifyRankTable failed, ret[%d].", __func__, ret),
        ret);
    rankTable = rankTable_;
    HCCL_INFO("[RankInfoDetectClient::%s] end, rankCount[%u].", __func__, rankTable_.rankCount);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectClient::Connect()
{
    TRY_CATCH_RETURN(clientSocket_->Connect(););
    HcclResult ret = CheckStatus();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[RankInfoDetectClient::%s] CheckStatus failed, ret[%d].", __func__, ret), ret);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectClient::CheckStatus()
{
    HCCL_DEBUG("[RankInfoDetectClient::%s] start.", __func__);

    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());

    while (true) {
        bool isTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
        if (isTimeout) {
            const auto elapsed
                = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - startTime);
            RPT_INPUT_ERR(
                isTimeout, "EI0015", std::vector<std::string>({"error_reason"}),
                std::vector<std::string>({StringFormat(
                    "Receiving message from the root node timed out "
                    "after %lld seconds. Timeout was set to %lld seconds. "
                    "Check whether node rankId[%u] reports an error",
                    static_cast<long long>(elapsed.count()), static_cast<long long>(timeout.count()), rankId_)}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] topo exchange agent get socket timeout, timeout[%lld s], rankId[%u]",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_TIMEOUT), static_cast<long long>(timeout.count()), rankId_);
            // 建链超时后，sleep 20s，避免上层应用提前退出，确保其他正常 client 能够收到 server 发出的临终遗言
            sleep(WAIT_ERROR_BROADCAST_TIME);
            return HCCL_E_TIMEOUT;
        }

        if (clientSocket_->GetStatus() == SocketStatus::OK) {
            HCCL_DEBUG("[RankInfoDetectClient::%s] client get socket connection success.", __func__);
            break;
        }
    }

    HCCL_INFO("[RankInfoDetectClient::%s] end, connect ok.", __func__);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectClient::SendAgentIdAndRankSize()
{
    HCCL_DEBUG("[RankInfoDetectClient::%s] start.", __func__);

    // 发送agentId
    std::string rankID = std::to_string(rankId_);
    std::string agentID = std::string(16 - rankID.length(), '0') + rankID;
    CHK_RET(socketAgent_.SendMsg(agentID.c_str(), agentID.size()));

    // 发送rankSize
    CHK_RET(socketAgent_.SendMsg(&rankSize_, sizeof(rankSize_)));

    HCCL_INFO(
        "[RankInfoDetectClient::%s] send agentID[%s] and rankSize_[%u] end.", __func__, agentID.c_str(), rankSize_);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectClient::SendLocalRankTable(const RankTableInfo& localRankTable)
{
    HCCL_DEBUG("[RankInfoDetectClient::%s] start.", __func__);

    // 消息格式: [ranktable数据(n字节)][step(4字节)]
    BinaryStream binaryStream;
    CHK_RET(localRankTable.GetBinStream(true, binaryStream));
    binaryStream << currentStep_;

    // 字节流转换为vector<char>格式
    vector<char> sendMsg;
    TRY_CATCH_RETURN(binaryStream.Dump(sendMsg););

    // 发送
    CHK_RET(socketAgent_.SendMsg(sendMsg.data(), sendMsg.size()));

    HCCL_INFO("[RankInfoDetectClient::%s] end, currentStep_[%u].", __func__, currentStep_);
    currentStep_++;
    return HCCL_SUCCESS;
}

void RankInfoDetectClient::ConstructSingleRank(RankTableInfo& localRankTable)
{
    localRankTable.version = "2.0";
    localRankTable.rankCount = 1;
    NewRankInfo rankInfo{};
    rankInfo.rankId = rankId_;
    rankInfo.deviceId = devPhyId_;
    rankInfo.rankLevelInfos.emplace_back(RankLevelInfo{});
    CHK_PRT_CONT(GetLocalTlsStatus(rankInfo.tlsStatus), HCCL_WARNING("[GetLocalTlsStatus] Can not get TlsStatus"));
    CHK_PRT_CONT(
        GetLocalHostDpuTlsStatus(rankInfo.hostDpuTlsStatus),
        HCCL_WARNING("[GetLocalHostDpuTlsStatus] Can not get Host DPU TlsStatus"));
    localRankTable.ranks.emplace_back(rankInfo);

    // 打印
    localRankTable.Dump();
    HCCL_INFO(
        "[RankInfoDetectClient::%s] end, single rank, localRankTable[%s].", __func__,
        localRankTable.Describe().c_str());
}

HcclResult RankInfoDetectClient::ConstructRankTable(RankTableInfo& localRankTable)
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);

    // 单P场景处理
    CHK_PRT_RET((rankSize_ == 1), ConstructSingleRank(localRankTable), HCCL_SUCCESS);

    // 1. 解析文件topoInfo.json
    std::string filePath = "/etc/hccl_rootinfo.json";
    std::string configJson{};
    std::ifstream file(filePath);
    if (file.good()) {
        HcclResult parseRet = ParseRootInfoConfigFile(filePath, configJson);
        if (parseRet != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] ParseFileToJson failed, filePath[%s], ret[%d]",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(parseRet),
                filePath.c_str(), parseRet);
            return parseRet;
        }
    } else {
        size_t bufSize = 0;
        s32 result = TopoAddrInfoGetSize(devPhyId_, &bufSize); // 获取rankInfo大小，用于提前分配内存
        if (result != 0) {
            RPT_INPUT_ERR(
                true, "EI0015", std::vector<std::string>({"error_reason"}),
                std::vector<std::string>(
                    {StringFormat("TopoAddrInfoGetSize failed, devPhyId[%u], result[%d]", devPhyId_, result)}));
        } else if (bufSize > MAX_BUFFER_LEN) {
            RPT_INPUT_ERR(
                true, "EI0015", std::vector<std::string>({"error_reason"}),
                std::vector<std::string>(
                    {StringFormat("bufSize[%zu] exceeds max[%u], devPhyId[%u]", bufSize, MAX_BUFFER_LEN, devPhyId_)}));
        }
        CHK_PRT_RET(
            result != 0 || bufSize > MAX_BUFFER_LEN,
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] Get rankinfo size failed, devPhyId[%u], result[%d], bufSize[%zu]",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), devPhyId_, result, bufSize),
            HCCL_E_PARA);
        std::vector<char> buffer(bufSize, '\0');
        result = TopoAddrInfoGet(devPhyId_, buffer.data(), &bufSize); // 获取rankInfo 并更新大小
        RPT_INPUT_ERR(
            result != 0, "EI0015", std::vector<std::string>({"error_reason"}),
            std::vector<std::string>({StringFormat(
                "TopoAddrInfoGet failed, devPhyId[%u], bufSize[%zu], result[%d]", devPhyId_, bufSize, result)}));
        CHK_PRT_RET(
            result != 0,
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] Get rankinfo failed, devPhyId[%u], bufSize[%zu], result[%d]",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), devPhyId_, bufSize, result),
            HCCL_E_PARA);
        std::string jsonString(buffer.data(), bufSize);
        CHK_RET(ParseRootInfoConfigJson(jsonString, configJson));
    }
    CHK_RET(CheckRootInfoConfig(configJson));

    // 2. 获取当前devPhyId_对应的devInfo
    std::string localDevInfoJson{};
    CHK_RET(SelectRootInfoLocalDevInfo(configJson, devPhyId_, rankId_, localDevInfoJson));
    // 3. 在反序列化和上报本地 RankTable 前改写 addr，确保后续全局 RankTable 和 RankGraph 使用选中地址
    CHK_RET(SelectRootInfoHostAddr(
        localDevInfoJson,
        [this](const IpAddress& addr) {
            return ProbeHostRoceAddr(addr);
        },
        devPhyId_, rankId_));

    // 无UB兜底：探测不到level0时插入PCIe兜底层，仅在level_list存在（RoCE主备探测已通过）后执行
    CHK_RET(InsertPcieFallbackLevel0(localDevInfoJson, devPhyId_));

    // 4. 组本地 RankTableInfo。rootinfo 不走 Check()，与旧 isCheck=false 一致。
    CHK_RET(BuildRootInfoLocalRankTable(configJson, localDevInfoJson, rankSize_, localRankTable));
    RPT_INPUT_ERR(
        localRankTable.ranks.empty(), "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
        std::vector<std::string>({"0", "rank_list", "non-empty"}));
    CHK_PRT_RET(
        localRankTable.ranks.empty(),
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] local rank table has no rank", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA)),
        HCCL_E_PARA);
    CHK_PRT_CONT(
        GetLocalTlsStatus(localRankTable.ranks[0].tlsStatus),
        HCCL_WARNING("[GetLocalTlsStatus] Can not get TlsStatus"));
    CHK_PRT_CONT(
        GetLocalHostDpuTlsStatus(localRankTable.ranks[0].hostDpuTlsStatus),
        HCCL_WARNING("[GetLocalHostDpuTlsStatus] Can not get Host DPU TlsStatus"));
    HCCL_INFO("[RankInfoDetectClient::%s] end.", __func__);
    return HCCL_SUCCESS;
}

s32 RankInfoDetectClient::ProbeHostRoceAddr(const IpAddress& hostAddr) const
{
    // hostAddr 来自 netLayer3 的 rank_addr_list，是 RoCE 数据通信地址；它不同于 RootInfoDetect
    // 使用的 rootHandle.ip，后者只负责控制面 socket 建链。
    RaInterface raInterface{devPhyId_, hostAddr};
    RdmaHandle rdmaHandle = nullptr;
    const s32 ret = HrtRaRdmaInit(HrtNetworkMode::PEER, raInterface, rdmaHandle);
    if (ret != 0) {
        return ret;
    }
    if (rdmaHandle != nullptr) {
        // RDMA handle 仅用于可用性探测，不参与后续通信，探测成功后立即释放。
        HrtRaRdmaDeInit(rdmaHandle, HrtNetworkMode::PEER);
    }
    HCCL_INFO(
        "[%s] host addr probe success, devPhyId[%u], rankId[%u], hostAddr[%s].", __func__, devPhyId_, rankId_,
        hostAddr.Describe().c_str());
    return 0;
}

HcclResult RankInfoDetectClient::RecvRankTableMsg(vector<char>& rankInfoMsg)
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);

    // 接收数据
    u64 revMsgLen = 0;
    std::unique_ptr<HostBuffer> msg = std::make_unique<HostBuffer>(MAX_BUFFER_LEN);
    char* msgAddr = reinterpret_cast<char*>(msg->GetAddr());
    bool recvOk = false;
    TRY_CATCH_RETURN(recvOk = socketAgent_.RecvMsg(msgAddr, revMsgLen););
    CHK_PRT_RET(!recvOk, HCCL_ERROR("RankInfoDetectClient::%s, recv rankTable error.", __func__), HCCL_E_NETWORK);

    // 以vector<char>格式保存
    rankInfoMsg.resize(revMsgLen);
    rankInfoMsg.assign(msgAddr, msgAddr + revMsgLen);

    HCCL_INFO("[RankInfoDetectClient::%s] end, revMsgLen[%llu].", __func__, revMsgLen);
    return HCCL_SUCCESS;
}

// 解析接收到的rank table信息
void RankInfoDetectClient::ParseRankTable(vector<char>& rankInfoMsg)
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);

    // 消息格式: [ranktable大小(u32, 4字节)][ranktable数据(n字节)][step(4字节)][failedAgentIdList]
    BinaryStream binStream(rankInfoMsg);

    // 解析localRankInfo
    rankTable_ = RankTableInfo(binStream);
    rankTable_.Dump();

    // 解析step
    u32 receivedStep;
    binStream >> receivedStep;

    // 解析failedAgentIdList
    std::string failedAgentIdList;
    binStream >> failedAgentIdList;
    if (failedAgentIdList.size() > 0) {
        // 建链失败时，打印 root 节点发来的临终遗言
        HCCL_ERROR(
            "[%s][%s] Failed to connect agent, failedRankIdList[%s]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), failedAgentIdList.c_str());
    }

    HCCL_INFO("[RankInfoDetectClient::%s] end.", __func__);
}

HcclResult RankInfoDetectClient::RecvRankTable()
{
    // 获取rankTable
    vector<char> rankInfoMsg{};
    CHK_RET(RecvRankTableMsg(rankInfoMsg));

    // 解析rankTable
    TRY_CATCH_RETURN(ParseRankTable(rankInfoMsg););

    // 校验
    CHK_RET(VerifyRankTable());
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectClient::VerifyRankTable()
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);

    // 校验rankCount符合预期
    if (rankTable_.rankCount != rankSize_) {
        RPT_INPUT_ERR(
            true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({std::to_string(rankTable_.rankCount), "rank_count", std::to_string(rankSize_)}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] rank_count[%u] does not match rankSize_[%u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rankTable_.rankCount,
            rankSize_);
        return HCCL_E_PARA;
    }

    // 校验rankTable内容
    CHK_RET(rankTable_.Check());
    // TLS开关一致性校验
    HcclResult ret = VerifyTlsConsistency();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[RankInfoDetectClient::%s] tls consistency verify failed, ret[%d]", __func__, ret), HCCL_E_PARA);

    HcclResult hostDpuRet = VerifyHostDpuTlsConsistency();
    CHK_PRT_RET(
        hostDpuRet != HCCL_SUCCESS,
        HCCL_ERROR("[RankInfoDetectClient::%s] hostDpuTls consistency verify failed, ret[%d]", __func__, hostDpuRet),
        HCCL_E_PARA);

    HCCL_INFO("[RankInfoDetectClient::%s] end.", __func__);
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectClient::GetLocalTlsStatus(TlsStatus& tlsStatus) const
{
    struct RaInfo raInfo {};
    raInfo.mode = NetworkMode::NETWORK_OFFLINE;
    raInfo.phyId = devPhyId_;
    return HrtRaGetTlsStatus(&raInfo, tlsStatus);
}

HcclResult RankInfoDetectClient::GetLocalHostDpuTlsStatus(TlsStatus& tlsStatus) const
{
    struct RaInfo raInfo {};
    raInfo.mode = NetworkMode::NETWORK_PEER_ONLINE;
    raInfo.phyId = devPhyId_;
    return HrtRaGetTlsStatus(&raInfo, tlsStatus);
}

void RankInfoDetectClient::GenerateTlsStatusStr(std::string& tlsStatusStr, const std::vector<u32>& tlsStatusRanks) const
{
    tlsStatusStr.clear();
    for (const auto& rank : tlsStatusRanks) {
        tlsStatusStr += std::to_string(rank) + ",";
    }
    if (!tlsStatusStr.empty() && tlsStatusStr.back() == ',') {
        tlsStatusStr.pop_back();
    }
}

void RankInfoDetectClient::ReportTlsConfigurationError(
    const std::string& tlsType, const std::string& tlsInconsistentTlsType, const std::string& tlsEnableRankStr,
    const std::string& tlsDisableRankStr, const std::string& tlsUnknownRankStr) const
{
    std::string expectMessage = "\"All ranks are consistent. Current status: rankList for enabled tls: "
                                + tlsEnableRankStr + "; rankList for disabled tls: " + tlsDisableRankStr;
    if (!tlsUnknownRankStr.empty()) {
        expectMessage += "; rankList for query failure tls: " + tlsUnknownRankStr;
    }
    expectMessage += "\"";
    std::string errormessage = "Value \"" + tlsInconsistentTlsType + "\" for config \"" + tlsType
                               + "\" is invalid. Expected: " + expectMessage;

    RPT_INPUT_ERR(
        true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
        std::vector<std::string>({tlsInconsistentTlsType, "\"" + tlsType + "\"", expectMessage}));

    HCCL_ERROR(
        "[%s][%s] errNo[0x%016llx] %s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
        HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), errormessage.c_str());
}

HcclResult RankInfoDetectClient::VerifyTlsConsistency() const
{
    auto getTlsStatus = [](const NewRankInfo& rankInfo) {
        return rankInfo.tlsStatus;
    };
    return VerifyTlsConsistencyByTlsType("tls", getTlsStatus);
}

HcclResult RankInfoDetectClient::VerifyHostDpuTlsConsistency() const
{
    auto getTlsStatus = [](const NewRankInfo& rankInfo) {
        return rankInfo.hostDpuTlsStatus;
    };
    return VerifyTlsConsistencyByTlsType("hostDpuTls", getTlsStatus);
}

HcclResult RankInfoDetectClient::VerifyTlsConsistencyByTlsType(
    const std::string& tlsType, const std::function<TlsStatus(const NewRankInfo&)>& getTlsStatus) const
{
    bool isSupportCheckTlsStatus = true; // 用于标识是否存在不支持查询Tls开关状态的情况
    bool isTlsConsistent = true;         // 用于标识TLS开关状态是否一致
    std::vector<u32> tlsEnableRank;
    std::vector<u32> tlsDisableRank;
    std::vector<u32> tlsUnknownRank;

    for (const auto& rankInfo : rankTable_.ranks) {
        const TlsStatus tlsStatus = getTlsStatus(rankInfo);
        if (tlsStatus == TlsStatus::ENABLE) {
            tlsEnableRank.push_back(rankInfo.rankId);
        } else if (tlsStatus == TlsStatus::DISABLE) {
            tlsDisableRank.push_back(rankInfo.rankId);
        } else {
            isSupportCheckTlsStatus = false;
            tlsUnknownRank.push_back(rankInfo.rankId);
        }
    }

    // 将卡的信息汇总成string
    std::string tlsEnableRankStr;
    std::string tlsDisableRankStr;
    std::string tlsUnknownRankStr;
    GenerateTlsStatusStr(tlsEnableRankStr, tlsEnableRank);
    GenerateTlsStatusStr(tlsDisableRankStr, tlsDisableRank);
    if (!isSupportCheckTlsStatus) {
        GenerateTlsStatusStr(tlsUnknownRankStr, tlsUnknownRank);
    }

    std::string tlsInconsistentTlsType;
    if (!tlsEnableRank.empty() && !tlsDisableRank.empty()) {
        isTlsConsistent = false;
        tlsInconsistentTlsType = (tlsDisableRank.size() <= tlsEnableRank.size()) ? "Disable" : "Enable";
    }

    // 四种不同情况
    if (isTlsConsistent && isSupportCheckTlsStatus) {
        // 1.通信域所有卡都支持查询TLS开关状态，并且TLS开关状态都是一致的。
        HCCL_INFO("[Verify][%sConsistency] All ranks %sStatus are consistent", tlsType.c_str(), tlsType.c_str());
    } else if (!isTlsConsistent && isSupportCheckTlsStatus) {
        // 2.通信域所有卡都支持查询TLS开关状态，但是TLS开关状态存在不一致，报错。
        ReportTlsConfigurationError(
            tlsType, tlsInconsistentTlsType, tlsEnableRankStr, tlsDisableRankStr, tlsUnknownRankStr);
        return HCCL_E_PARA;
    } else if (isTlsConsistent && !isSupportCheckTlsStatus) {
        // 3.通信域内的部分卡不支持查询TLS开关状态，目前能查询到的卡的TLS开关状态是一致的，打印warning提醒
        HCCL_WARNING(
            "[Verify][%sConsistency] Some ranks do not support to check %sStatus, "
            "not support rankId: [%s]",
            tlsType.c_str(), tlsType.c_str(), tlsUnknownRankStr.c_str());
    } else {
        // 4.通信域内的部分卡不支持查询TLS开关状态，但是目前能查询到的卡的TLS开关状态已经不一致，报错
        ReportTlsConfigurationError(
            tlsType, tlsInconsistentTlsType, tlsEnableRankStr, tlsDisableRankStr, tlsUnknownRankStr);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult RankInfoDetectClient::HostListenPortDetect(NewRankInfo& rankInfo)
{
    std::string topoPath;
    TRY_CATCH_RETURN(topoPath = QueryRootInfoTopoFilePath(););
    if (PhyTopoBuilder::GetInstance().Build(topoPath) != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] PhyTopoBuilder Build failed", __func__);
        return HCCL_E_INTERNAL;
    }
    auto devLogicId = HrtGetDevice();
    u32 devPhyId = rankInfo.deviceId;
    for (auto& rankLevelInfo : rankInfo.rankLevelInfos) {
        shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> graph
            = PhyTopo::GetInstance()->GetTopoGraph(rankLevelInfo.netLayer);
        if (graph == nullptr) {
            HCCL_DEBUG("[RankInfoDetectClient::%s]Can't find the layout %u Graph!", __func__, rankLevelInfo.netLayer);
            continue;
        }
        std::vector<std::shared_ptr<PhyTopo::Link>> links = graph->GetEdges(rankInfo.localId);
        for (auto& link : links) {
            if (link->GetSourceIFace()->GetPos() != AddrPosition::HOST) {
                continue;
            }
            const std::set<LinkProtocol>& protocols = link->GetLinkProtocols();
            for (auto& protocol : protocols) {
                LinkProtoType protoType = LinkProtocol2LinkProtoType(protocol);
                if (protoType != LinkProtoType::RDMA || rankLevelInfo.rankAddrs.empty()) {
                    continue;
                }
                HCCL_DEBUG("[SocketManager::%s] find the host rdma link %s", __func__, link->Describe().c_str());
                const IpAddress& hostIp = rankLevelInfo.rankAddrs[0].addr;
                const bool isHostSocketPortRangeConfigured
                    = !EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange().empty();
                if (isHostSocketPortRangeConfigured) {
                    // Primary/backup candidate selection is completed by SelectRootInfoHostAddr.
                    // This check only validates the selected address for host socket listening.
                    // It does not switch candidates: linkdown skips listening; other errors terminate initialization.
                    const s32 probeRet = ProbeHostRoceAddr(hostIp);
                    if (probeRet == HCCP_ELINKDOWN) {
                        HCCL_RUN_WARNING(
                            "[RankInfoDetectClient::%s] host dpu address[%s] is linkdown, skip host socket listen and "
                            "continue communication domain initialization.",
                            __func__, hostIp.Describe().c_str());
                        return HCCL_SUCCESS;
                    }
                    CHK_PRT_RET(
                        probeRet != 0,
                        HCCL_ERROR(
                            "[RankInfoDetectClient::%s] probe host dpu address[%s] failed, ret[%d].", __func__,
                            hostIp.Describe().c_str(), probeRet),
                        HCCL_E_NETWORK);
                }
                uint32_t hostPort = 0;
                TRY_CATCH_RETURN(SetupHostListenPort(devLogicId, devPhyId, hostIp, hostPort););
                rankInfo.hostPort = hostPort;
                return HCCL_SUCCESS;
            }
        }
    }
    return HCCL_SUCCESS;
}

void RankInfoDetectClient::SetupHostListenPort(
    u32 devLogicId, u32 devPhyId, const IpAddress& hostIp, uint32_t& hostPort)
{
    std::lock_guard<std::mutex> lock(hostSocketLock_);
    u32 listenPort = HCCL_INVALID_PORT;
    auto portRange = EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange();
    u32 basePort = EnvConfig::GetInstance().GetHostNicConfig().GetIfBasePort();
    if (portRange.empty() && basePort != HCCL_INVALID_PORT) {
        listenPort = basePort + devPhyId;
        HCCL_INFO("[RankInfoDetectClient::%s] BasePort is configured, listenPort[%u].", __func__, listenPort);
        hostPort = listenPort;
        return;
    }

    if (portRange.empty()) {
        HCCL_INFO(
            "[RankInfoDetectClient::%s] No port configuration, using default port range[%u, %u]", __func__,
            HOST_DATA_BASE_PORT, HOST_DATA_BASE_PORT + HOST_DATA_PORT_COUNT);
        SocketPortRange defaultRange = {HOST_DATA_BASE_PORT, HOST_DATA_BASE_PORT + HOST_DATA_PORT_COUNT};
        portRange.push_back(defaultRange);
    } else {
        // 数据面使用前半端口范围
        std::vector<SocketPortRange> frontHalf;
        std::vector<SocketPortRange> backHalf;
        SplitPortRange(portRange, frontHalf, backHalf);
        portRange = frontHalf;
    }

    SocketHandle hostSocketHandle = HostSocketHandleManager::GetInstance().Create(devPhyId, hostIp);
    hostSocket_ = std::make_shared<Socket>(
        hostSocketHandle, hostIp, HCCL_INVALID_PORT, hostIp, "hostport_preempt", SocketRole::SERVER,
        NicType::HOST_NIC_TYPE);
    PreemptPortManager::GetInstance(devLogicId).ListenPreempt(hostSocket_, portRange, listenPort);
    HCCL_INFO("[RankInfoDetectClient::%s] preempt hostPort[%u] success.", __func__, listenPort);

    // 登记到进程级 map，供算子下发阶段复用，避免跨阶段端口竞争
    DevNetPortType portType(ConnectProtoType::RDMA);
    PortData portData(static_cast<RankId>(devPhyId), portType, 0, hostIp);
    SocketManager socketMgr(0, devPhyId, static_cast<u32>(devLogicId), "hostport_preempt");
    hostSocketRegistered_ = socketMgr.RegisterHostListenSocket(portData, hostSocket_);

    hostPort = listenPort;
}

void RankInfoDetectClient::SocketTearDown(u32 devPhyId)
{
    std::lock_guard<std::mutex> lock(hostSocketLock_);
    if (hostSocket_ == nullptr) {
        HCCL_INFO("[RankInfoDetectClient::%s] hostSocket is null, skip.", __func__);
        return;
    }
    const IpAddress& hostIp = hostSocket_->GetLocalIp();
    auto devLogicId = HrtGetDevice();
    if (hostSocketRegistered_) {
        // 已登记到 SocketManager::GetServerSocketMap()，所有权已转移给 map，
        // 由算子下发阶段 HostSocketStopListen refcount 归 0 时清理，此处跳过 Release/Destroy
        HCCL_INFO(
            "[RankInfoDetectClient::%s] hostSocket already registered to ServerSocketMap, "
            "skip Release/Destroy, only release local ref.",
            __func__);
    } else {
        // 未转移所有权的 hostSocket_ 均持有 HostSocketHandleManager 引用；抢占端口需先 Release
        if (EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange().size() > 0
            || EnvConfig::GetInstance().GetHostNicConfig().GetIfBasePort() == HCCL_INVALID_PORT) {
            EXCEPTION_CATCH(
                PreemptPortManager::GetInstance(devLogicId).Release(hostSocket_),
                HCCL_ERROR("[RankInfoDetectClient::%s] PreemptPort Release exception", __func__));
        }
        EXCEPTION_CATCH(
            HostSocketHandleManager::GetInstance().Destroy(devPhyId, hostIp),
            HCCL_ERROR("[RankInfoDetectClient::%s] HostSocketHandle Destroy exception", __func__));
    }
    hostSocket_ = nullptr;
}

HcclResult RankInfoDetectClient::TearDown()
{
    bool expected = false;
    if (!shutdownDone_.compare_exchange_strong(expected, true)) {
        HCCL_INFO("[RankInfoDetectClient::%s] already torn down, skip.", __func__);
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);
    SocketTearDown(devPhyId_);

    if (clientSocket_ != nullptr) {
        IpAddress localIp = clientSocket_->GetLocalIp();
        EXCEPTION_CATCH(
            clientSocket_->Close(), HCCL_ERROR("[RankInfoDetectClient::%s] clientSocket Close exception", __func__));
        if (HostSocketHandleManager::GetInstance().Get(devPhyId_, localIp) != nullptr) {
            EXCEPTION_CATCH(
                HostSocketHandleManager::GetInstance().Destroy(devPhyId_, localIp),
                HCCL_ERROR("[RankInfoDetectClient::%s] HostSocketHandle Destroy exception", __func__));
        }
        clientSocket_.reset();
    }

    // 同步释放 RA 引用：detach 线程无法保证在进程退出前完成，也会与 manager 静态析构产生竞态。
    EXCEPTION_CATCH(
        HccpPeerManager::GetInstance().DeInit(HrtGetDevice()),
        HCCL_ERROR("[RankInfoDetectClient::TearDown] DeInit exception"));

    HCCL_INFO("[RankInfoDetectClient::%s] end.", __func__);
    return HCCL_SUCCESS;
}

// 析构零副作用：TearDown() 由 RankInfoDetect::Shutdown() 在运行期显式调用（RA、runtime 一定存活）。
// 不在析构链中调 TearDown，避免 exit() 静态析构阶段 RA/runtime 已 finalize 导致 core。
RankInfoDetectClient::~RankInfoDetectClient() {}

} // namespace Hccl
