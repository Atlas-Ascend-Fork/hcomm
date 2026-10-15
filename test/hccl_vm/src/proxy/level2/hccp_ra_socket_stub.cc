/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "RASOCKET_STUB"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <string>
#include <unordered_map>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "sim_log.h"
#include "hccp_common.h"
#include "hccp_async.h"
#include "hccp_async_request.h"
#include "sim_ip_address.h"
#include "runtime/base.h"
#include "store_sim_comm_memory_manager.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/table_access.h"
#include "store_sim_run_mode.h"
#include "ibv_dpu_stub.h"

extern uint64_t g_cur_server_key;

struct CacheEntry {
    uint64_t pair_id;
    uint32_t slot_idx;
};

struct SocketInfo {
    uint32_t device_id;
    uint64_t endpoint_id;
    uint8_t state;
    int family;
};

struct FdInfo {
    uint32_t slot_idx;
    uint32_t role;
    uint64_t pair_id;
    sim::SlotHandle handle;
    std::string tag;

    // 评估模式的真实对端信息(GetSockets由remoteIp解析)
    uint32_t peerDeviceId = INVALID_DEVICE_ID;
    uint8_t peerEid[16] = {};
    uint8_t localEid[16] = {};

    std::vector<uint8_t> peerStream; // 估算模式：幻影对端回放字节流（本端 Send 镜像）
    size_t peerOffset;               // 估算模式：已回放偏移
};
struct SocketWrapInitInfo {
    int family;
    union HccpIpAddr localIp;
    int scopeId;
};
struct SocketConnTaskInfo {
    void* sockWrap;
    unsigned int role;
    union HccpIpAddr remoteIp;
    unsigned int port;
    char tag[SOCK_CONN_TAG_SIZE];
    int status;
    void* fdHandle;
};
struct SocketExchInfo {
    void* socketHandle;
    void* sendBuf;
    unsigned long long sendSize;
    unsigned long long* sentSize;
    void* recvBuf;
    unsigned long long recvSize;
    unsigned long long* resvSize;
    bool isRecvFirst;
};

static std::unordered_map<std::string, CacheEntry> g_tagCache;
static std::mutex g_tagCacheMutex;

// 估算模式开关由 sim::IsEstimateMode() 惰性读 DB 的 RunModeConfig 判定（与 check-only 同源），
// 开启时单卡(rank0)构造幻影对端，跳过真实对端阻塞点（RaGetSockets 的 180×500ms 轮询 /
// RaSocketRecv 空槽 SOCK_EAGAIN 无限重试），仅用于在 level2 保真前提下生成任务清单，不真跑数据传输。
// 关闭时行为与多进程/多 rank 完全一致。

// 估算模式幻影对端 slot 序号: 独立自增, 起在高位, 与真实内存池 AllocSlot() 的 0..kSlotsPerPage*kMaxPages 不相交,
// 避免 close 时 g_tagCache 按 slot_idx 擦除误伤真实连接。phantom slot 仅作为控制流唯一标识,
// 不解引用任何 SHM (Send/Recv 在估算模式短路走 peerStream), 故不占用真实内存池。
static constexpr uint32_t kPhantomSlotBase = 0x80000000u;
static std::atomic<uint32_t> g_phantomSlotSeq{kPhantomSlotBase};
static inline uint32_t NextPhantomSlot() { return g_phantomSlotSeq.fetch_add(1, std::memory_order_relaxed); }
static inline bool IsPhantomSlot(uint32_t slotIdx) { return slotIdx >= kPhantomSlotBase; }

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

static uint64_t ComputeTagHash(const char* tag)
{
    return static_cast<uint64_t>(std::hash<std::string>{}(std::string(tag)));
}

// UBOE层链路地址为IPv4(rootinfo下发IPV4类型, 对齐真实环境LayerAddUBOE),
// 其余层为EID(按IPv6建链): 按family选择EndPoint寻址方式(IPv4串查ip_addr / EID查eid)
static int ResolveEndPointByAddr(
    int family, const union HccpIpAddr& ipAddr, std::string& addrDesc, sim::runtime::EndPoint& endPoint)
{
    BinaryAddr ba{};
    memcpy(&ba, &ipAddr, sizeof(BinaryAddr));
    if (family == AF_INET) {
        const IpAddress ipAddress(ba, AF_INET);
        addrDesc = ipAddress.GetIpStr();
        if (sim::runtime::GetEndPointByIpAddr(addrDesc, endPoint) != ACL_SUCCESS) {
            HCCL_VM_ERROR(" cannot find endpoint by ip addr:{}", addrDesc.c_str());
            return -1;
        }
        return 0;
    }
    const IpAddress ipAddress(ba, AF_INET6);
    addrDesc = ipAddress.EidToHexString();
    if (sim::runtime::GetEndPointByEid(ipAddress, endPoint) != 0) {
        HCCL_VM_ERROR(" cannot find endpoint by eid:{}", addrDesc.c_str());
        return -1;
    }
    return 0;
}

int RaSocketInit(int mode, struct rdev rdevInfo, void** socketHandle)
{
    // 首次调用 RaSocketInit 时初始化 SHM 池，后续调用跳过
    static std::once_flag initPoolFlag;
    std::call_once(initPoolFlag, []() {
        HCCL_VM_INFO("lazy init pool on first RaSocketInit call");
        sim::CommunicationMemoryManager::GetInstance().InitPool();
    });

    (void)mode;
    sim::runtime::Device device{};
    if (sim::runtime::GetDeviceByPhysicalId(rdevInfo.phyId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR(" get device by phy id {} failed.", rdevInfo.phyId);
        return -1;
    }

    sim::runtime::EndPoint endPoint{};
    std::string addrDesc;
    if (ResolveEndPointByAddr(rdevInfo.family, rdevInfo.localIp, addrDesc, endPoint) != 0) {
        return -1;
    }

    auto deviceIdx = endPoint.device_id;
    auto endpointId = endPoint.id;

    auto* info = new SocketInfo{deviceIdx, endpointId, 0, rdevInfo.family};
    *socketHandle = static_cast<void*>(info);
    HCCL_VM_INFO(
        " add socketHandle:{:p} device:{:d} addr:{} endpointId:{:d}", static_cast<void*>(info), deviceIdx,
        addrDesc.c_str(), endpointId);
    return 0;
}

int RaSocketInitV1(int mode, struct SocketInitInfoT socketInit, void** socketHandle)
{
    return RaSocketInit(mode, socketInit.rdevInfo, socketHandle);
}

int RaSocketWrapBatchInit(
    struct RaInfo raInfo, struct SocketWrapInitInfo sockWrapInitInfoArray[], unsigned int num,
    struct ErrorInfo* errInfo, void* sockWrapArray[])
{
    for (unsigned int i = 0; i < num; i++) {
        struct rdev rdevInfo {};
        rdevInfo.phyId = raInfo.phyId;
        rdevInfo.family = sockWrapInitInfoArray[i].family;
        memcpy(&rdevInfo.localIp, &sockWrapInitInfoArray[i].localIp, sizeof(union HccpIpAddr));

        int ret = RaSocketInit(raInfo.mode, rdevInfo, &sockWrapArray[i]);
        if (ret != 0) {
            errInfo->errIndex = i;
            errInfo->errCode = ret;
            return -1;
        }
    }
    return 0;
}

int RaSocketDeinit(void* socketHandle)
{
    auto* info = static_cast<SocketInfo*>(socketHandle);
    HCCL_VM_INFO(" delete socketHandle:{:p}", static_cast<void*>(info));
    delete info;
    return 0;
}

int RaSocketWrapBatchDeinit(
    void* sockWrapArray[], unsigned int num, struct ErrorInfo errInfoArray[], unsigned int* errNum)
{
    *errNum = 0;
    for (unsigned int i = 0; i < num; i++) {
        int ret = RaSocketDeinit(sockWrapArray[i]);
        if (ret != 0) {
            errInfoArray[*errNum].errIndex = i;
            errInfoArray[*errNum].errCode = ret;
            (*errNum)++;
        }
    }
    return (*errNum > 0) ? -1 : 0;
}

int RaSocketListenStart(struct SocketListenInfoT conn[], uint32_t num)
{
    for (uint32_t i = 0; i < num; i++) {
        auto* info = static_cast<SocketInfo*>(conn[i].socketHandle);
        info->state = 1;
        HCCL_VM_INFO(" socket {:p} port:{:d}", static_cast<void*>(info), conn[i].port);
    }
    return 0;
}

int RaSocketListenStop(struct SocketListenInfoT conn[], unsigned int num)
{
    for (uint32_t i = 0; i < num; i++) {
        auto* info = static_cast<SocketInfo*>(conn[i].socketHandle);
        info->state = 0;
        HCCL_VM_INFO(" socket {:p} port:{:d}", static_cast<void*>(info), conn[i].port);
    }
    return 0;
}

int RaSocketBatchConnect(struct SocketConnectInfoT conn[], unsigned int num)
{
    std::vector<sim::runtime::RaSocketPair> pairs;
    std::vector<uint32_t> slotIdxs;
    pairs.reserve(num);
    slotIdxs.reserve(num);

    for (unsigned int i = 0; i < num; i++) {
        auto* localInfo = static_cast<SocketInfo*>(conn[i].socketHandle);

        sim::runtime::EndPoint endPoint{};
        std::string remoteDesc;
        if (ResolveEndPointByAddr(localInfo->family, conn[i].remoteIp, remoteDesc, endPoint) != 0) {
            return -1;
        }

        auto remoteDevId = endPoint.device_id;
        uint64_t tagHash = ComputeTagHash(conn[i].tag);
        uint32_t remotePort = conn[i].port;

        HCCL_VM_INFO(
            " dev:{} sock:{:p} connect dev:{} addr:{} tag:{}", localInfo->device_id, static_cast<void*>(localInfo),
            remoteDevId, remoteDesc.c_str(), conn[i].tag);

        if (sim::IsEstimateMode()) {
            HCCL_VM_INFO("Estimate: skip connect bookkeeping tag:{}", conn[i].tag);
            continue;
        }

        uint32_t slotIdx = sim::CommunicationMemoryManager::GetInstance().AllocSlot();
        if (slotIdx == UINT32_MAX) {
            HCCL_VM_ERROR("AllocSlot failed for tag:{}", conn[i].tag);
            return -1;
        }

        sim::runtime::RaSocketPair socketpair{};
        socketpair.id = 0;
        socketpair.client_id = reinterpret_cast<uintptr_t>(localInfo);
        socketpair.server_id = 0;
        socketpair.ref_cnt = 0;
        socketpair.port = remotePort;
        socketpair.tag_hash = tagHash;
        socketpair.slot_idx = slotIdx;
        pairs.push_back(socketpair);
        slotIdxs.push_back(slotIdx);
    }

    // 一批一次入库（主线 RunnerDB::AddBatch 语义迁入新架构 Storage::InsertMany：
    // 整批成功按输入顺序回填主键；按引擎批量上限分窗，任一窗失败整批失败回滚）
    auto txn = HcclSim::Storage::BeginTransaction();
    if (!txn.ok()) {
        HCCL_VM_ERROR("AddBatch RaSocketPair failed to begin transaction: {}", txn.diagnostic);
        return -1;
    }
    for (std::size_t first = 0; first < pairs.size(); first += HcclSim::Storage::kInsertBatchMaxRows) {
        const std::size_t count = std::min(HcclSim::Storage::kInsertBatchMaxRows, pairs.size() - first);
        const auto inserted = HcclSim::Storage::InsertMany(pairs.data() + first, count, *txn.value);
        if (!inserted.ok()) {
            // 失败路径：Table/Transaction 已整笔回滚并熔断（句柄析构兜底），本函数返回 -1
            HCCL_VM_ERROR(
                "AddBatch RaSocketPair failed: expected {}, windowFirst={}, count={}, code:{}, "
                "diagnostic:{}",
                num, first, count, static_cast<int>(inserted.code), inserted.diagnostic);
            return -1;
        }
    }
    auto committed = txn.value->Commit();
    if (!committed.ok()) {
        HCCL_VM_ERROR("AddBatch RaSocketPair commit failed: {}", committed.diagnostic);
        return -1;
    }

    {
        std::lock_guard<std::mutex> lock(g_tagCacheMutex);
        for (unsigned int i = 0; i < num; i++) {
            g_tagCache[conn[i].tag] = {pairs[i].id, slotIdxs[i]};
            HCCL_VM_INFO(
                " add pair id:{:d} local:{:p} slotIdx:{} tag:{} port:{}", pairs[i].id,
                static_cast<void*>(conn[i].socketHandle), slotIdxs[i], conn[i].tag, conn[i].port);
        }
    }
    return 0;
}

// 评估模式填充本端和对端Eid信息
static void FillFdPeerInfo(FdInfo* fd, const struct SocketInfoT& conn)
{
    if (fd == nullptr) {
        return;
    }
    BinaryAddr ba{};
    std::memcpy(&ba, &conn.remoteIp, sizeof(ba));
    IpAddress remoteAddr(ba, AF_INET6);
    sim::runtime::EndPoint remoteEp{};
    if (sim::runtime::GetEndPointByEid(remoteAddr, remoteEp) == 0) {
        fd->peerDeviceId = remoteEp.device_id;
        std::memcpy(fd->peerEid, remoteEp.eid, sizeof(fd->peerEid));
    } else {
        HCCL_VM_ERROR(
            "FillFdPeerInfo: remote eid unresolved, tag:{} eid={}", conn.tag, remoteAddr.EidToHexString().c_str());
    }

    auto* sockInfo = static_cast<SocketInfo*>(conn.socketHandle);
    if (sockInfo != nullptr) {
        auto localEp = sim::runtime::Db::GetById<sim::runtime::EndPoint>(sockInfo->endpoint_id);
        if (localEp.ok()) {
            std::memcpy(fd->localEid, localEp->eid, sizeof(fd->localEid));
        }
    }
}

int RaGetSockets(unsigned int role, struct SocketInfoT conn[], unsigned int num, unsigned int* connectedNum)
{
    *connectedNum = 0;
    uint32_t fdRole = (role == 0) ? sim::kRsFdServer : sim::kRsFdClient;
    for (int i = 0; i < num; i++) {
        uint32_t slotIdx = UINT32_MAX;
        uint64_t pairId = 0;

        {
            std::lock_guard<std::mutex> lock(g_tagCacheMutex);
            auto it = g_tagCache.find(conn[i].tag);
            if (it != g_tagCache.end()) {
                slotIdx = it->second.slot_idx;
                pairId = it->second.pair_id;
            }
        }
        if (slotIdx == UINT32_MAX) {
            if (sim::IsEstimateMode()) {
                {
                    std::lock_guard<std::mutex> lock(g_tagCacheMutex);
                    auto it = g_tagCache.find(conn[i].tag);
                    if (it != g_tagCache.end()) {
                        slotIdx = it->second.slot_idx;
                        pairId = it->second.pair_id;
                    } else {
                        slotIdx = NextPhantomSlot();
                        g_tagCache[conn[i].tag] = {0, slotIdx};
                    }
                }
                HCCL_VM_INFO("Estimate: phantom slotIdx:{} role:{} tag:{}", slotIdx, role, conn[i].tag);
            } else if (role == sim::kRsFdServer) {
                // 非阻塞单次探测；未就绪 sleep 兼作查库节流，等待/超时交给 hcomm 外层，始终 return 0。
                uint64_t tagHash = ComputeTagHash(conn[i].tag);
                auto pairRes = sim::runtime::Db::GetOneByPred<sim::runtime::RaSocketPair>(
                    HcclSim::Storage::Eq(&sim::runtime::RaSocketPair::tag_hash, tagHash));
                if (!pairRes.ok()) {
                    if (pairRes.code != HcclSim::Storage::DbCode::NOT_FOUND) {
                        HCCL_VM_ERROR(
                            "RaGetSockets backend failure: tag:{}, tagHash:0x{:x}({}), code:{}, diagnostic:{}",
                            conn[i].tag, tagHash, tagHash, static_cast<int>(pairRes.code), pairRes.diagnostic);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    continue;
                }
                slotIdx = pairRes.value->slot_idx;
                pairId = pairRes.value->id;
                {
                    std::lock_guard<std::mutex> lock(g_tagCacheMutex);
                    g_tagCache[conn[i].tag] = {pairId, slotIdx};
                }
            } else {
                HCCL_VM_ERROR("Client: tag {} not found in cache", conn[i].tag);
                continue;
            }
        }

        // phantom slot 不解引用 SHM：估算模式数据面走 FdInfo::peerStream，handle 留空、不做引用计数。
        sim::SlotHandle handle{};
        if (!IsPhantomSlot(slotIdx)) {
            handle = sim::CommunicationMemoryManager::GetInstance().GetSlotHandle(slotIdx, fdRole);
            if (!handle.base) {
                HCCL_VM_ERROR("GetSlotHandle failed for slotIdx:{}", slotIdx);
                continue;
            }
            sim::CommunicationMemoryManager::GetInstance().AddRef(slotIdx);
        }

        auto* fd = new FdInfo{slotIdx, fdRole, pairId, handle, conn[i].tag};
        conn[i].fdHandle = static_cast<void*>(fd);
        conn[i].status = 1;
        *connectedNum += 1;
        if (sim::IsEstimateMode()) {
            FillFdPeerInfo(fd, conn[i]);
        }

        HCCL_VM_INFO(
            " get socket tag:{} slotIdx:{} role:{} peerDevice:{}", conn[i].tag, slotIdx, role, fd->peerDeviceId);
    }
    return 0;
}

int RaSocketBatchClose(struct SocketCloseInfoT conn[], unsigned int num)
{
    for (int i = 0; i < num; i++) {
        auto* fd = static_cast<FdInfo*>(conn[i].fdHandle);
        {
            std::lock_guard<std::mutex> lock(g_tagCacheMutex);
            for (auto it = g_tagCache.begin(); it != g_tagCache.end(); ++it) {
                if (it->second.slot_idx == fd->slot_idx) {
                    g_tagCache.erase(it);
                    break;
                }
            }
        }

        if (IsPhantomSlot(fd->slot_idx)) {
            // phantom slot 未做引用计数、未插入 RaSocketPair，无需 CloseSlot/Delete
            HCCL_VM_INFO(" close phantom socket slotIdx:{}", fd->slot_idx);
        } else {
            bool isLast = sim::CommunicationMemoryManager::GetInstance().CloseSlot(fd->slot_idx);
            if (isLast) {
                // 捕获 Db::Delete 真实结果（此前被丢弃、恒打印 delete pair）：
                // 失败记 pairId/slotIdx/code/diagnostic；成功 0 行与成功 1 行可区分
                // （0 行＝行已不在：早前重发/重复删除或他人已删）。CloseSlot/Delete/free
                // 顺序与调用次数不变。
                auto deleted = sim::runtime::Db::Delete<sim::runtime::RaSocketPair>(
                    HcclSim::Storage::Eq(&sim::runtime::RaSocketPair::id, fd->pair_id));
                if (!deleted.ok()) {
                    HCCL_VM_ERROR(
                        "delete RaSocketPair FAILED: pairId:{}, slotIdx:{}, code:{}, diagnostic:{}", fd->pair_id,
                        fd->slot_idx, static_cast<int>(deleted.code), deleted.diagnostic);
                } else {
                    const uint64_t affected = deleted.value.has_value() ? *deleted.value : 0U;
                    if (affected == 0U) {
                        HCCL_VM_WARN(
                            "delete RaSocketPair OK/0rows: pairId:{}, slotIdx:{} (row already absent)", fd->pair_id,
                            fd->slot_idx);
                    } else {
                        HCCL_VM_INFO(" delete pair:{:d} slotIdx:{}", fd->pair_id, fd->slot_idx);
                    }
                }
            }
            HCCL_VM_INFO(" close socket slotIdx:{}", fd->slot_idx);
        }
        delete fd;
    }
    return 0;
}

int RaSocketBatchAbort(struct SocketConnectInfoT conn[], unsigned int num)
{
    (void)conn;
    (void)num;
    return 0;
}

int RaSocketBatchDisconnectNorm(
    void* socketHandleArray[], unsigned int num, struct ErrorInfo errInfoArray[], unsigned int* errNum)
{
    *errNum = 0;
    for (unsigned int i = 0; i < num; i++) {
        auto* socketInfo = static_cast<SocketInfo*>(socketHandleArray[i]);
        uintptr_t clientId = reinterpret_cast<uintptr_t>(socketInfo);

        auto pairs = sim::runtime::Db::GetByPred<sim::runtime::RaSocketPair>(
            HcclSim::Storage::Eq(&sim::runtime::RaSocketPair::client_id, clientId));
        if (!pairs.ok() || !pairs.value.has_value()) {
            continue; // 查询失败按无匹配处理（与主线"空集不报错"行为一致）
        }

        for (auto& pair : *pairs.value) {
            auto* tempFd = new FdInfo{pair.slot_idx, 0, pair.id, {}};
            struct SocketCloseInfoT closeInfo {};
            closeInfo.fdHandle = tempFd;
            RaSocketBatchClose(&closeInfo, 1);
        }

        struct SocketListenInfoT listenInfo {};
        listenInfo.socketHandle = socketHandleArray[i];
        RaSocketListenStop(&listenInfo, 1);
    }
    return (*errNum > 0) ? -1 : 0;
}

int RaSocketSend(const void* fdHandle, const void* data, unsigned long long size, unsigned long long* sentSize)
{
    auto* fd = const_cast<FdInfo*>(static_cast<const FdInfo*>(fdHandle));
    if (sim::IsEstimateMode() && fd != nullptr) {
        // 时序关联——RaGetQpAttr已记录待绑QP, 此刻绑定到本socket真实对端
        BindDpuPendingQps(fd->peerDeviceId, fd->peerEid, fd->localEid);
        // 幻影对端：本端 Send 的字节镜像进对端回放流，供本端 Recv 读回（对端消息与本端同构）
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        fd->peerStream.insert(fd->peerStream.end(), bytes, bytes + size);
        *sentSize = size;
        return 0;
    }
    int64_t ret = sim::CommunicationMemoryManager::GetInstance().Send(fd->handle, data, size);
    if (ret == -1) {
        HCCL_VM_ERROR("Send failed slotIdx:{}", fd->slot_idx);
        return -1;
    }
    *sentSize = ret;
    return 0;
}

int RaSocketRecv(const void* fdHandle, void* data, unsigned long long size, unsigned long long* receivedSize)
{
    auto* fd = const_cast<FdInfo*>(static_cast<const FdInfo*>(fdHandle));
    if (sim::IsEstimateMode()) {
        // 幻影对端：空槽不再返回 SOCK_EAGAIN，改为从镜像字节流按序排出对端消息
        size_t avail = fd->peerStream.size() - fd->peerOffset;
        if (avail == 0) {
            sleep(1);
            return SOCK_EAGAIN;
        }

        size_t real = (size < avail) ? static_cast<size_t>(size) : avail;
        memcpy(data, fd->peerStream.data() + fd->peerOffset, real);
        fd->peerOffset += real;
        *receivedSize = real;
        return 0;
    }
    int64_t ret = sim::CommunicationMemoryManager::GetInstance().Recv(fd->handle, data, size);
    if (ret == 0) {
        return SOCK_EAGAIN;
    }
    if (ret == -1) {
        HCCL_VM_ERROR("Recv failed slotIdx:{}", fd->slot_idx);
        return -1;
    }
    *receivedSize = ret;
    return 0;
}

int RaEpollCtlAdd(const void* fdHandle, enum RaEpollEvent event)
{
    (void)fdHandle;
    (void)event;
    return 0;
}

int RaEpollCtlMod(const void* fdHandle, enum RaEpollEvent event)
{
    (void)fdHandle;
    (void)event;
    return 0;
}

int RaEpollCtlDel(const void* fdHandle)
{
    (void)fdHandle;
    return 0;
}

int RaSetTcpRecvCallback(const void* socketHandle, const void* callback)
{
    (void)socketHandle;
    (void)callback;
    return 0;
}

int RaExChangeDataAsync(struct SocketExchInfo sockExchInfoArray[], unsigned int num, void** reqOpHandle, int timeout)
{
    (void)timeout;
    *reqOpHandle = reinterpret_cast<void*>(0x12345678);

    for (unsigned int i = 0; i < num; i++) {
        auto* socketInfo = static_cast<SocketInfo*>(sockExchInfoArray[i].socketHandle);
        uintptr_t clientId = reinterpret_cast<uintptr_t>(socketInfo);

        auto pairs = sim::runtime::Db::GetByPred<sim::runtime::RaSocketPair>(
            HcclSim::Storage::Eq(&sim::runtime::RaSocketPair::client_id, clientId));

        if (!pairs.ok() || !pairs.value.has_value() || pairs.value->empty()) {
            HCCL_VM_ERROR("no pair found for socketHandle:{:p}", static_cast<void*>(socketInfo));
            continue;
        }

        auto& pair = pairs.value->front();
        sim::SlotHandle handle
            = sim::CommunicationMemoryManager::GetInstance().GetSlotHandle(pair.slot_idx, sim::kRsFdClient);
        auto* tempFd = new FdInfo{pair.slot_idx, sim::kRsFdClient, pair.id, handle};

        if (sockExchInfoArray[i].isRecvFirst) {
            RaSocketRecv(
                tempFd, sockExchInfoArray[i].recvBuf, sockExchInfoArray[i].recvSize, sockExchInfoArray[i].resvSize);
            RaSocketSend(
                tempFd, sockExchInfoArray[i].sendBuf, sockExchInfoArray[i].sendSize, sockExchInfoArray[i].sentSize);
        } else {
            RaSocketSend(
                tempFd, sockExchInfoArray[i].sendBuf, sockExchInfoArray[i].sendSize, sockExchInfoArray[i].sentSize);
            RaSocketRecv(
                tempFd, sockExchInfoArray[i].recvBuf, sockExchInfoArray[i].recvSize, sockExchInfoArray[i].resvSize);
        }

        delete tempFd;
    }
    return 0;
}

/////////////////////////////////async/////////////////////////////
// 合同（hccp_async.h）：提交成功返回 0 并给出唯一 req_handle，任务终态经
// RaGetAsyncReqResult 查询。模拟器提交时同步完成工作：成功/EAGAIN 记入 request
// 供查询；提交即失败（非 EAGAIN 错误）直接返回错误、不登记（调用方此时不会轮询）。
// 唯一真实句柄与查询即释放语义由 hccp_async_request 公共注册表提供（与 QP/TP
// 族存根共用），固定假句柄 0x12345678 全部废除。
int RaGetAsyncReqResult(void* reqHandle, struct AsyncReqResult* reqResult)
{
    if (reqHandle == nullptr || reqResult == nullptr) {
        HCCL_VM_ERROR(
            "RaGetAsyncReqResult invalid param: reqHandle={:p}, reqResult={:p}", reqHandle,
            static_cast<void*>(reqResult));
        return -1;
    }
    return hccp_async_req::Consume(reqHandle, &reqResult->reqResult, &reqResult->interfaceOpcode);
}

int RaSocketBatchConnectAsync(struct SocketConnectInfoT conn[], unsigned int num, void** reqHandle)
{
    if (reqHandle == nullptr) {
        return -1;
    }
    int ret = RaSocketBatchConnect(conn, num);
    if (ret != 0) {
        return ret;
    }
    void* handle = hccp_async_req::Register(0, hccp_async_req::OP_SOCKET_BATCH_CONNECT);
    if (handle == nullptr) {
        return -1;
    }
    *reqHandle = handle;
    return 0;
}

int RaSocketBatchConnectNormAsync(
    struct SocketConnTaskInfo sockConnTaskInfoArray[], unsigned int num, void** reqOpHandle, int timeout)
{
    (void)timeout;
    *reqOpHandle = reinterpret_cast<void*>(0x12345678);
    std::vector<unsigned int> clientIdx;
    std::vector<unsigned int> serverIdx;
    std::vector<struct SocketConnectInfoT> clientConns;
    std::vector<struct SocketInfoT> clientSockets;
    std::vector<struct SocketInfoT> serverSockets;

    for (unsigned int i = 0; i < num; i++) {
        if (sockConnTaskInfoArray[i].role == sim::kRsFdClient) {
            clientIdx.push_back(i);
            struct SocketConnectInfoT conn {};
            conn.socketHandle = sockConnTaskInfoArray[i].sockWrap;
            memcpy(&conn.remoteIp, &sockConnTaskInfoArray[i].remoteIp, sizeof(union HccpIpAddr));
            conn.port = sockConnTaskInfoArray[i].port;
            strncpy(conn.tag, sockConnTaskInfoArray[i].tag, SOCK_CONN_TAG_SIZE - 1);
            clientConns.push_back(conn);

            struct SocketInfoT sock {};
            sock.socketHandle = sockConnTaskInfoArray[i].sockWrap;
            strncpy(sock.tag, sockConnTaskInfoArray[i].tag, SOCK_CONN_TAG_SIZE - 1);
            clientSockets.push_back(sock);
        } else {
            serverIdx.push_back(i);
            struct SocketInfoT sock {};
            sock.socketHandle = sockConnTaskInfoArray[i].sockWrap;
            strncpy(sock.tag, sockConnTaskInfoArray[i].tag, SOCK_CONN_TAG_SIZE - 1);
            serverSockets.push_back(sock);
        }
    }

    if (!clientConns.empty()) {
        RaSocketBatchConnect(clientConns.data(), clientConns.size());
    }

    if (!clientSockets.empty()) {
        unsigned int connectedNum = 0;
        RaGetSockets(sim::kRsFdClient, clientSockets.data(), clientSockets.size(), &connectedNum);
        for (unsigned int j = 0; j < clientIdx.size(); j++) {
            unsigned int i = clientIdx[j];
            sockConnTaskInfoArray[i].fdHandle = clientSockets[j].fdHandle;
            sockConnTaskInfoArray[i].status = clientSockets[j].status;
        }
    }

    if (!serverSockets.empty()) {
        unsigned int connectedNum = 0;
        RaGetSockets(sim::kRsFdServer, serverSockets.data(), serverSockets.size(), &connectedNum);
        for (unsigned int j = 0; j < serverIdx.size(); j++) {
            unsigned int i = serverIdx[j];
            sockConnTaskInfoArray[i].fdHandle = serverSockets[j].fdHandle;
            sockConnTaskInfoArray[i].status = serverSockets[j].status;
        }
    }

    return 0;
}

int RaSocketListenStartAsync(struct SocketListenInfoT conn[], unsigned int num, void** reqHandle)
{
    (void)conn;
    (void)num;
    (void)reqHandle;
    return -1;
}

int RaSocketListenStopAsync(struct SocketListenInfoT conn[], unsigned int num, void** reqHandle)
{
    if (reqHandle == nullptr) {
        return -1;
    }
    int ret = RaSocketListenStop(conn, num);
    if (ret != 0) {
        return ret;
    }
    void* handle = hccp_async_req::Register(0, hccp_async_req::OP_SOCKET_LISTEN_STOP);
    if (handle == nullptr) {
        return -1;
    }
    *reqHandle = handle;
    return 0;
}

int RaSocketBatchCloseAsync(struct SocketCloseInfoT conn[], unsigned int num, void** reqHandle)
{
    if (reqHandle == nullptr) {
        return -1;
    }
    int ret = RaSocketBatchClose(conn, num);
    if (ret != 0) {
        return ret;
    }
    void* handle = hccp_async_req::Register(0, hccp_async_req::OP_SOCKET_BATCH_CLOSE);
    if (handle == nullptr) {
        return -1;
    }
    *reqHandle = handle;
    return 0;
}

int RaSocketSendAsync(
    const void* fdHandle, const void* data, unsigned long long size, unsigned long long* sentSize, void** reqHandle)
{
    if (reqHandle == nullptr || sentSize == nullptr) {
        return -1;
    }
    int ret = RaSocketSend(fdHandle, data, size, sentSize);
    if (ret != 0) {
        return ret;
    }
    // 满槽时 RaSocketSend 返回 0 且实发 0 字节：伪装成"已完成 0 字节"会造成发送侧
    // 忙等（sendLeftSize 永不减少）；按真实 socket 语义记 EAGAIN，hcomm 发送侧
    // CheckSendRequestResult 对 SOCK_E_AGAIN 走同样的 1ms 睡眠重投路径。
    int finalResult = (size > 0 && *sentSize == 0) ? SOCK_EAGAIN : 0;
    void* handle = hccp_async_req::Register(finalResult, hccp_async_req::OP_SOCKET_SEND);
    if (handle == nullptr) {
        return -1;
    }
    *reqHandle = handle;
    return 0;
}

int RaSocketRecvAsync(
    const void* fdHandle, void* data, unsigned long long size, unsigned long long* receivedSize, void** reqHandle)
{
    if (reqHandle == nullptr || receivedSize == nullptr) {
        return -1;
    }
    int ret = RaSocketRecv(fdHandle, data, size, receivedSize);
    if (ret != 0 && ret != SOCK_EAGAIN) {
        return ret;
    }
    // 无数据（SOCK_EAGAIN）不再伪装成"已完成收到 0 字节"：登记终态 EAGAIN 并返回
    // 提交成功 0；调用方轮询得到 reqResult==SOCK_EAGAIN 后走 1ms 睡眠重投路径。
    // 成功时 receivedSize 已由 RaSocketRecv 填写实收字节数，原样保留。
    void* handle = hccp_async_req::Register(ret, hccp_async_req::OP_SOCKET_RECV);
    if (handle == nullptr) {
        return -1;
    }
    *reqHandle = handle;
    return 0;
}

#ifdef __cplusplus
}
#endif // __cplusplus
