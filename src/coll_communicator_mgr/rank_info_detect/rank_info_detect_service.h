/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RANK_INFO_DETECT_SERVICE_H
#define HCCLV2_RANK_INFO_DETECT_SERVICE_H

#include <vector>
#include <climits>
#include "socket.h"
#include <unordered_map>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include "universal_concurrent_map.h"
#include "new_rank_info.h"
#include "rank_table_info.h"
#include "internal_exception.h"
#include "timeout_exception.h"
#include "socket_exception.h"
#include "socket_agent.h"

namespace Hccl {

// 拓扑探测服务状态常量（g_detectServerStatus_ 的 value）
constexpr u32 RANKINFO_DETECT_SERVER_STATUS_IDLE = 0;
constexpr u32 RANKINFO_DETECT_SERVER_STATUS_RUNNING = 1;
constexpr u32 RANKINFO_DETECT_SERVER_STATUS_ERROR = 2;

class RankInfoDetectService {
public:
    // Grow 独立状态机：Init 继续使用 g_detectServerStatus_ 的 IDLE/RUNNING/ERROR。
    enum class GrowState : u8 {
        STARTING = 0, // service 已构造，Grow 线程尚未就绪
        IDLE,         // Grow 线程已就绪，可触发下一轮
        RUNNING,      // 本轮 Grow 已触发并执行中
        ERROR,        // 本轮失败，当前 service 不再复用
        SHUTDOWN      // service 已关闭
    };

    // 设置和获取 root 进程内主线程与 accept 线程之间的同步状态。
    static void SetServerStatus(u32 port, u32 status)
    {
        g_detectServerStatus_.EmplaceAndUpdate(port, [status](u32& v) {
            v = status;
        });
    }
    static bool TryGetServerStatus(u32 port, u32& status)
    {
        auto it = g_detectServerStatus_.Find(port);
        if (it.second) {
            status = it.first->second;
            return true;
        }
        return false;
    }

    RankInfoDetectService(
        u32 devPhyId, std::shared_ptr<Socket> serverSocket, std::string identifier, vector<RaSocketWhitelist> wlistInfo)
        : devPhyId_(devPhyId),
          serverSocket_(serverSocket),
          listenPort_(serverSocket_->GetListenPort()),
          hostIp_(serverSocket_->GetLocalIp()),
          identifier_(identifier),
          wlistInfo_(wlistInfo)
    {}
    ~RankInfoDetectService();

    HcclResult Setup();
    // Grow 收集循环：等待 TriggerGrowCollect 唤醒并执行增量收集，由 RankInfoDetect::TriggerGrowCollect 懒启动。
    HcclResult SetupGrowLoop();
    // root 侧：通知 accept 线程开始Grow Grow 收集（设置 expectedNewRankCount 并唤醒）
    HcclResult TriggerGrowCollect(u32 newRankCount);
    // 阻塞等待当前收集收集+广播完成，输出 mergedRankTable
    HcclResult WaitForCollectComplete(RankTableInfo& rankTable);
    // 通知 accept 线程退出（通信域销毁时调用）
    HcclResult Shutdown();
    // 完成后线程退出时置 true，TriggerGrowCollect 据此判断是否需重启 SetupGrowLoop 线程
    bool IsThreadExited() const { return threadExited_.load(); }
    GrowState GetGrowState() const
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return growState_;
    }
    // 上一轮 Init/Grow 已完成，当前 service 可进入下一轮
    bool IsReadyForGrow() const
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (shutdown_ || growState_ == GrowState::ERROR || growState_ == GrowState::SHUTDOWN) {
            return false;
        }
        if (threadExited_) {
            return growState_ == GrowState::STARTING || growState_ == GrowState::IDLE;
        }
        return growState_ == GrowState::IDLE;
    }
    // 旧 service 的 Grow 线程退出后，仅非终态允许重建 Grow 线程
    bool IsSupportGrowLoop() const
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return !shutdown_ && growState_ != GrowState::ERROR && growState_ != GrowState::SHUTDOWN;
    }
    // Grow 服务线程已启动，发布 IDLE，等待 TriggerGrowCollect
    HcclResult MarkGrowLoopReady();
    // Grow 线程退出：记录本轮结果，失败进入 ERROR 终态
    void MarkGrowThreadExited(HcclResult result);
    // 标记服务线程已退出
    void MarkThreadExited() { threadExited_.store(true); }

    std::shared_ptr<Socket> StealServerSocket() { return std::move(serverSocket_); }
    // lazy 重启时旧 service move 给新 service，避免父域拓扑丢失
    RankTableInfo StealRankTable() { return std::move(rankTable_); }
    // 新 service 从旧 service 继承父域拓扑
    void SetRankTable(RankTableInfo&& rankTable) { rankTable_ = std::move(rankTable); }
    // 转移 connSockets_ 所有权
    std::unordered_map<std::string, std::shared_ptr<Socket>> StealConnSockets()
    {
        std::lock_guard<std::mutex> lock(connMtx_);
        return std::move(connSockets_);
    }
    void SetConnSockets(std::unordered_map<std::string, std::shared_ptr<Socket>>&& sockets)
    {
        std::lock_guard<std::mutex> lock(connMtx_);
        connSockets_ = std::move(sockets);
    }

private:
    u32 devPhyId_{0};
    shared_ptr<Socket> serverSocket_{nullptr};
    u32 listenPort_{0}; ///< 缓存 listen port，TearDown 置空 serverSocket_ 后 GetStatus/SetStatus 仍可用
    std::unordered_map<std::string, std::shared_ptr<Socket>> connSockets_{};
    IpAddress hostIp_{};
    RankTableInfo rankTable_{};
    std::string failedAgentIdList_{};
    u32 currentStep_{0};
    std::string identifier_{};
    vector<RaSocketWhitelist> wlistInfo_{};

    HcclResult GetConnections(bool isInitCollect);
    HcclResult GetRankTable(bool isInitCollect = true);
    HcclResult GetRankTableForGrow();
    HcclResult BroadcastRankTable();
    HcclResult BroadcastRankTableForGrow();
    // === Init/Grow 共用数据与同步 ===
    std::atomic<bool> shutdown_{false};
    u32 expectedNewRankCount_{0};
    HcclResult collectResult_{HCCL_SUCCESS};

    // 事件驱动唤醒：替代 SaluSleep 轮询，idle 时零 CPU，Trigger/Finish 唤醒纳秒级响应
    mutable std::mutex mtx_{};
    mutable std::condition_variable cv_{};
    // 保护 connSockets_/newAgentIds_；锁内只做容器操作，阻塞 socket 操作在锁外执行
    mutable std::mutex connMtx_{};

    std::atomic<bool> tornDown_{false}; // TearDown 防重入 flag
    std::atomic<bool> threadExited_{false}; // Setup/Grow 线程退出后置 true，TriggerGrowCollect 据此判断是否需重启线程
    GrowState growState_{GrowState::STARTING};
    HcclResult growResult_{HCCL_SUCCESS};

    // 本次 Grow新增连接的 agentId 列表（后续收集只从新连接 recv info）
    std::vector<std::string> newAgentIds_{};

    std::unordered_map<std::string, std::shared_ptr<Socket>> SnapshotConnSockets() const
    {
        std::lock_guard<std::mutex> lock(connMtx_);
        return connSockets_;
    }
    size_t GetConnSocketCount() const
    {
        std::lock_guard<std::mutex> lock(connMtx_);
        return connSockets_.size();
    }

    // Init 状态查询/设置：通过 g_detectServerStatus_ map（key=listenPort）
    u32 GetStatus() const
    {
        u32 status = RANKINFO_DETECT_SERVER_STATUS_IDLE;
        TryGetServerStatus(listenPort_, status);
        return status;
    }
    void SetStatus(u32 status) const { SetServerStatus(listenPort_, status); }

    static UniversalConcurrentMap<u32, u32> g_detectServerStatus_;

    /** @brief 当前收集结束：设置 collectResult_，成功设 IDLE，失败设 ERROR。 */
    HcclResult FinishCollect(HcclResult result);
    /** @brief Grow 当前轮结束：只更新 Grow 状态，不影响 Init 状态。 */
    HcclResult FinishGrowCollect(HcclResult result);
    HcclResult Disconnect();

    /** @brief Grow收集失败时回滚本次 Grow新增连接，保留老 rank 连接。 */
    HcclResult RollbackNewConnections();
    HcclResult TearDown();
    bool RecvRemoteAgentId(SocketAgent& connSocketAgent, std::string& agentId);
    bool RecvRemoteRankSize(SocketAgent& connSocketAgent, u32& rankSize);
    HcclResult RecvRankInfoMsg(SocketAgent& connSocketAgent, vector<char>& rankInfoMsg) const;
    void SortRankTable();
    HcclResult ParseRankTable(vector<char>& rankInfoMsg, bool isGrowCollect);

    // 异常流程处理方法
    HcclResult FailedConnectionAgentIdString(u32 rankSize);

    // 校验相关方法
    bool RecvAndVerifyRemoteAgentIdAndRankSize(
        std::shared_ptr<Socket> connSocket, u32& expectedSocketNum, u32& previousRankSize, bool isInitCollect);
    bool VerifyRemoteRankSize(u32& previousRankSize, u32 remoteRankSize) const;
    // 白名单开启时校验远端IP是否在拓扑探测白名单内：仅校验remoteIp(tag 由驱动层白名单校验)
    bool IsRemoteIpInWhiteList(const IpAddress& remoteIp) const;

    // DFX相关方法
    void DisplayConnectingStatus(u32 totalSockets, u32 waitSockets);
    void DisplayConnectedRanks();
};

} // namespace Hccl
#endif // HCCLV2_RANK_INFO_DETECT_SERVICE_H
