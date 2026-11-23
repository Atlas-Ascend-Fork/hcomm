/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RANK_INFO_DETECT_CLIENT_H
#define HCCLV2_RANK_INFO_DETECT_CLIENT_H

#include <cstdint>
#include <functional>
#include <atomic>
#include <thread>
#include <vector>
#include <string>
#include "socket.h"
#include "new_rank_info.h"
#include "rank_table_info.h"
#include "internal_exception.h"
#include "timeout_exception.h"
#include "socket_exception.h"
#include "socket_agent.h"
#include "ip_address.h"
#include "coll_comm_config.h"

namespace Hccl {

#ifdef OPEN_HCCL_TEST
const u32 WAIT_ERROR_BROADCAST_TIME = 1;
#endif

constexpr u32 MAX_ERROR_BROADCAST_WAIT_SEC = 300;

// 失败 agent 上报等待窗口 = min(max(linkTimeOutSec, 0) + SOCKET_LINK_EXTRA_TIMEOUT, MAX_ERROR_BROADCAST_WAIT_SEC)。
u32 GetErrorBroadcastWaitSeconds(s32 linkTimeOutSec);

// 遗言日志的失败列表打印前缀长度上限：超长截断为前缀 + "..."，失败总数恒可见
constexpr u32 MAX_FAILED_LIST_PRINT_LEN = 256;

// 遗言日志格式化：返回截断后的打印串
std::string FormatFailedAgentIdList(const std::string& failedAgentIdList, u64& failedCount, bool& parseOk);

class RankInfoDetectClient {
public:
    RankInfoDetectClient(u32 devPhyId, u32 rankSize, u32 rankId, const std::shared_ptr<Socket>& clientSocket)
        : devPhyId_(devPhyId),
          rankSize_(rankSize),
          rankId_(rankId),
          clientSocket_(clientSocket),
          socketAgent_(clientSocket.get())
    {}
    ~RankInfoDetectClient();

    HcclResult TearDown();

    HcclResult Setup(RankTableInfo& rankTable);
    // Setup 前按值保存入口已解析的策略；探测阶段不重新读取环境变量。
    void SetNetLayerSelection(const hccl::NetLayerSelection& selection) { netLayerSelection_ = selection; }

    // expectedRankCount 扩容后预期总 rank 数，用于校验接收到的 rankCount。
    HcclResult WaitForNextBroadcast(RankTableInfo& rankTable, uint32_t expectedRankCount);

private:
    hccl::NetLayerSelection netLayerSelection_{};
    NetLayerCheckInfo localNetLayerCheck_{};
    NetLayerCheckInfo remoteNetLayerCheck_{};
    u32 devPhyId_{0};
    u32 rankSize_{0};
    u32 rankId_{0};
    std::shared_ptr<Socket> clientSocket_{nullptr};
    u32 currentStep_{0};
    RankTableInfo rankTable_{};
    SocketAgent socketAgent_;
    std::mutex hostSocketLock_;
    std::shared_ptr<Socket> hostSocket_ = nullptr;
    bool hostSocketRegistered_ = false; // 标记 hostSocket_ 是否已登记到 SocketManager::GetServerSocketMap()
    std::atomic<bool> shutdownDone_{false};

    void SetupHostListenPort(u32 devLogicId, u32 devPhyId, const IpAddress& hostIp, uint32_t& hostPort);
    void SocketTearDown(u32 devPhyId);
    HcclResult Connect();
    HcclResult CheckStatus();
    HcclResult SendAgentIdAndRankSize();
    HcclResult SendLocalRankTable(const RankTableInfo& localRankTable);
    HcclResult ConstructRankTable(RankTableInfo& localRankTable);
    HcclResult VerifyRankTable();
    // 按本 rank 候选层生成上报结果；解析失败也写入报文，交由 root 汇总。
    void PrepareNetLayerCheck(const hccl::NewRankInfo& rankInfo);
    // 校验 root 汇总结果及本地有效层；显式选层遇旧 root 时拒绝继续。
    HcclResult VerifyNetLayerConsistency() const;
    HcclResult RecvRankTable();
    HcclResult RecvRankTableMsg(vector<char>& rankInfoMsg);
    // 解析广播内容，选层报文错误直接返回；原有反序列化异常由调用方处理。
    HcclResult ParseRankTable(vector<char>& rankInfoMsg);
    s32 ProbeHostRoceAddr(const IpAddress& hostAddr) const;
    void ConstructSingleRank(RankTableInfo& localRankTable);
    HcclResult GetLocalTlsStatus(TlsStatus& tlsStatus) const;
    HcclResult GetLocalHostDpuTlsStatus(TlsStatus& tlsStatus) const;
    HcclResult VerifyTlsConsistency() const;
    HcclResult VerifyHostDpuTlsConsistency() const;
    HcclResult VerifyTlsConsistencyByTlsType(
        const std::string& tlsType, const std::function<TlsStatus(const hccl::NewRankInfo&)>& getTlsStatus) const;
    void GenerateTlsStatusStr(std::string& tlsStatusStr, const std::vector<u32>& tlsStatusRanks) const;
    void ReportTlsConfigurationError(
        const std::string& tlsType, const std::string& tlsInconsistentTlsType, const std::string& tlsEnableRankStr,
        const std::string& tlsDisableRankStr, const std::string& tlsUnknownRankStr) const;
    HcclResult HostListenPortDetect(hccl::NewRankInfo& rankInfo);
};

} // namespace Hccl
#endif // HCCLV2_RANK_INFO_DETECT_CLIENT_H
