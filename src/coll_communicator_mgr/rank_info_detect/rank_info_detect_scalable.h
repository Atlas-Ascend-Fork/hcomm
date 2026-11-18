/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RANK_INFO_DETECT_SCALABLE_H
#define HCCL_RANK_INFO_DETECT_SCALABLE_H

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "ip_address.h"
#include "rank_info_detect.h"
#include "rank_table_info.h"
#include "root_handle_v2.h"
#include "root_info_detect_bridge.h"
#include "socket.h"
#include "types.h"
#include "universal_concurrent_map.h"

namespace Hccl {

constexpr u32 SCALABLE_INVALID_INDEX = 0xFFFFFFFF;

// 每个 root 的 full mesh 连接信息
struct RootMeshInfo {
    char ip[IP_ADDRESS_BUFFER_LEN]{0};
    u32 meshPort{HCCL_INVALID_PORT};
};

// 所有 root 的信息，用于root间mesh建链
struct ScalableServerInfo {
    u32 nRoot{0};                          // root 总数
    u32 rootIndex{SCALABLE_INVALID_INDEX}; // 本 root 序号
    u32 groupSize{0};                      // 本组 rank 个数
    u32 groupFirstRank{0};                 // 本组首 rank（组内最小 rankId，即 root 的 rankId）
    std::string identifier{};              // rootHandle 标识符
    std::vector<RootMeshInfo> meshInfos;   // 全量 root ip+meshPort，下标 = rootIndex
};

// scalable 多 root 探测本体。仅 root rank 在 HcclGetRootInfoScalableV2 阶段创建
class RankInfoDetectScalable : public std::enable_shared_from_this<RankInfoDetectScalable> {
public:
    explicit RankInfoDetectScalable(s32 devLogicId, s32 userDevId, u32 devPhyId);
    ~RankInfoDetectScalable();

    // Get 阶段：组内 server 监听 + mesh 监听，拉起 server 线程
    HcclResult SetupScalableRoot(HcclScalableRootHandleV2& outHandle);

    // Init 阶段（root rank）：按 accept 侧链路 tag 下发/清理 mesh 白名单
    HcclResult AddMeshSocketWhiteList(const ScalableServerInfo& info);
    HcclResult DeleteMeshSocketWhiteList(const ScalableServerInfo& info) const;

    // Init 阶段（root rank）：注入 mesh 全量信息并唤醒 server 线程
    HcclResult SetScalableServerInfo(const ScalableServerInfo& info);

    // server 线程内阻塞等待注入 ScalableServerInfo
    HcclResult WaitScalableInfoReady();

private:
    void ScalableServerWorker(ErrContextPub errorContext);
    HcclResult SetupServerListen(IpAddress& hostIp, u32& hostPort);
    HcclResult BuildRootHandle(const IpAddress& hostIp, u32 hostPort);
    HcclResult AddServerSocketWhiteList();

    // 线程出口清理资源
    void ReleaseAgentResources();
    void ReleaseMeshResources();
    void EraseFromRegistry();
    // SetupScalableRoot 失败路径回滚
    void CleanupOnSetupFailure();
    // server 线程未捕获异常收尾
    void HandleWorkerException(u32 serverPort);
    void ReleaseListenResources();

    HcclResult SetupMeshListen(u32& meshPort, u32 serverPort);
    void ReleaseListenSocketResources(
        const std::shared_ptr<Socket>& listenSocket, const IpAddress& hostIp, u32 portOffset) const;
    HcclResult RootMeshAllGather(const RankTableInfo& localRankTable);
    HcclResult MeshConnectToLargerRoots(u32 timeoutSec);
    HcclResult MeshAcceptSmallerRoots(u32 timeoutSec);
    HcclResult MeshAcceptSmallerRootsCore(u32 timeoutSec);
    HcclResult ExchangeMeshPartial(Socket& socket, u32 peer, bool sendFirst);
    HcclResult SendMeshPartial(Socket& socket) const;
    HcclResult RecvMeshPartial(Socket& socket, u32 peer);
    HcclResult MergeMeshPartials(u32 nRoot, RankTableInfo& rankTable);
    std::shared_ptr<Socket> MakeMeshSocket(u32 peer);
    HcclResult SetDetectServerFinalStatus(u32 serverPort, bool failed) const;

    s32 devLogicId_{0};
    s32 userDevId_{0};
    u32 devPhyId_{0};
    bool deviceSet_{false}; // HrtSetDevice 是否成功，决定线程收尾是否执行 HrtResetDevice
    SocketHandle serverHostSocketHandle_{nullptr};        // server 服务 host socket 句柄（白名单下发用）
    std::shared_ptr<Socket> serverListenSocket_{nullptr}; // 组内 server 服务监听（SetupServerListen 产出）
    vector<RaSocketWhitelist> serverWlistInfo_{};         // server 服务白名单（AddServerSocketWhiteList 产出）
    HcclRootHandleV2 rootHandle_{};                       // 组内 server 服务 handle（BuildRootHandle 产出）
    std::shared_ptr<Socket> meshListenSocket_{nullptr};
    ScalableServerInfo scalableInfo_{};
    std::mutex scalableInfoMutex_{}; // 保护 scalableInfo_ 注入
    std::condition_variable scalableInfoReadyCv_{};
    std::atomic<bool> scalableInfoReady_{false};

    std::vector<std::vector<char>> partials_{}; // 存放所有root的ranktable
    std::atomic<u32> meshRecvCount_{0};
    std::atomic<bool> meshFailed_{false};
};

// 进程级注册表：Get 阶段按 identifier 存 root 探测对象，Init 阶段按 identifier 查找对象后注入
UniversalConcurrentMap<std::string, std::shared_ptr<RankInfoDetectScalable>>& GetScalableDetectRegistry();

// 注册表外层串行锁：注册表 Find/插入/删除及 Find 返回值（shared_ptr）的取出须全程持同一把锁。
std::mutex& GetScalableDetectRegistryMutex();

} // namespace Hccl

#endif // HCCL_RANK_INFO_DETECT_SCALABLE_H
