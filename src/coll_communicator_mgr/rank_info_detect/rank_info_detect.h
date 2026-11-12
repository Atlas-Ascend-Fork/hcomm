/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RANK_INFO_DETECT_H
#define HCCL_RANK_INFO_DETECT_H

#include "types.h"
#include "socket.h"
#include "ip_address.h"
#include <vector>
#include "rank_table_info.h"
#include "root_handle_v2.h"
#include "orion_adapter_hccp.h"
#include "rank_info_detect_service.h"
#include "rank_info_detect_client.h"
#include "socket_handle_manager.h"
#include "universal_concurrent_map.h"
#include "adapter_error_manager_pub.h"
#include <atomic>
#include <climits>
#include <mutex>

namespace Hccl {

class RankInfoDetect {
public:
    RankInfoDetect();
    ~RankInfoDetect();

    HcclResult SetupServer(HcclRootHandleV2& rootHandle);
    HcclResult SetupAgent(u32 rankSize, u32 rankId, const HcclRootHandleV2& rootHandle);
    const HcclRootHandleV2& GetRootHandleInfo() const { return rootHandle_; }
    void SetRootHandle(const HcclRootHandleV2& rootHandle) { rootHandle_ = rootHandle; }
    void GetRankTable(RankTableInfo& ranktable) const;
    HcclResult WaitComplete(u32 listenPort, u32 listenStatus) const;

    // === Grow 持久化接口 ===
    /** @brief root 侧：触发Grow Grow 拓扑收集并等待完成。 */
    HcclResult TriggerGrowCollect(u32 newRankCount, RankTableInfo& mergedRankTable);
    /** @brief non-root 侧：阻塞等待 root 的Grow广播。 */
    HcclResult WaitForNextBroadcast(RankTableInfo& rankTable, uint32_t expectedRankCount);
    /** @brief 停止持久服务（通信域销毁时调用）。 */
    HcclResult StopPersistentServer();
    /** @brief 显式释放本进程持有的探测资源；最后一个 CollComm 引用释放时由析构兜底调用。 */
    HcclResult Shutdown();

    /** @brief root 重选时设置派生 identifier（topParentId + "_" + (genIndex+1)），在 SetupServer 前调用。 */
    HcclResult SetDerivationIdentifier(const std::string& topParentId, uint32_t generationIndex);

    // root 持有 rankInfoDetectService_
    bool IsRoot() const { return rankInfoDetectService_ != nullptr; }
    // root 的上一轮探测已完成，当前 service 可触发后续 Grow
    bool IsReadyForGrow() const
    {
        return rankInfoDetectService_ != nullptr && rankInfoDetectService_->IsReadyForGrow();
    }

    /// 返回对象构造时绑定的物理设备 ID，供调用方判断是否属于当前线程设备
    u32 GetDevPhyId() const { return devPhyId_; }

private:
    s32 devLogicId_{0};
    s32 userDevId_{0};
    u32 devPhyId_{0};
    RankTableInfo rankTable_{};
    IpAddress hostIp_{};
    u32 hostPort_{HCCL_INVALID_PORT};
    u32 rankId_{UINT_MAX};
    HcclRootHandleV2 rootHandle_{}; // root 身份唯一来源；rootRankId 为当前域内 root rank
    vector<IpAddress> hostSocketWlist_{};
    vector<RaSocketWhitelist> wlistInfo_{};
    std::string identifier_{};
    std::string derivationIdentifierOverride_{}; // root 重选时覆盖 identifier_ 的派生标识
    std::shared_ptr<RankInfoDetectClient> rankInfoDetectClient;
    std::shared_ptr<RankInfoDetectService> rankInfoDetectService_{
        nullptr}; ///< Root 侧拓扑探测服务，Setup 后保留以复用 socket
    std::thread serviceThread_;
    bool persistentStarted_{false};
    HcclResult SetupRankInfoDetectService(
        ErrContextPub errorContext, shared_ptr<Socket> serverSocket, s32 devLogicId, u32 devPhyId,
        std::string identifier, vector<RaSocketWhitelist> wlistInfo);
    std::shared_ptr<Socket> ServerInit();
    std::shared_ptr<Socket> ClientInit(const HcclRootHandleV2& rootHandle);
    HcclResult GetHandleAndAddHostSocketWhitelist();
    HcclResult AddHostSocketWhitelist(SocketHandle& socketHandle, const vector<IpAddress>& hostSocketWlist);
    u32 GetHostListenPort();
    HcclResult GetRootHandle(HcclRootHandleV2& rootHandle);
    SocketHandle GetHostSocketHandle();

    /// SetupServer 失败时释放已抢占的端口和白名单条目，避免资源泄漏
    HcclResult ReleaseServerResource(shared_ptr<Socket>& serverSocket);

    std::atomic<bool> shutdownDone_{false};
};

} // namespace Hccl

#endif // HCCL_VIRT_TOPO_DETECT_H
