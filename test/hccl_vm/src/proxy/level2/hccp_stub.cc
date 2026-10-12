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
#define HCCL_VM_MODULE "HCCP_STUB"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <unordered_map>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "atrace_types.h"
#include "dtype_common.h"
#include "sim_log.h"
#include "sim_capacity_limits.h"
#include "hccp_common.h"
#include "hccp_ctx.h"
#include "hccp_async_request.h"
#include "hccp_tlv.h"
#include "sim_ip_address.h"
#include "sim_common_defs.h"
#include "mem.h"
#include "rts_device.h"
#include "runtime/base.h"
#include "store_sim_device_memory_manager.h"
#include "store_sim_memory_manager.h"
#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/table_access.h"
#include "ibv_dpu_stub.h"
#include "sim_aicpu_pipe_msg.h"
#include "sim_sub_process_manager.h"

extern uint64_t g_cur_server_key;

struct FakeQpCache {
    struct ibv_qp* fakeQp;
    struct ibv_cq* sendCq;
    struct ibv_cq* recvCq;
    bool ownsCq;         // CQ 是否由本处内部创建（如 loopback），需在销毁时 delete
    uint64_t pairedQpId; // loopback 配对的另一端 QP DB id，0 表示无配对
    struct ibv_qp* pairedFakeQp;
    struct ibv_cq* pairedSendCq;
    struct ibv_cq* pairedRecvCq;
};
struct WhiteListInfo {
    void* sockWrap;
    struct SocketWlistInfoT* wlistDataArray;
    unsigned int num;
};
struct RmaImportInfo {
    void* rmaWrap;
    struct MrImportInfoT* memImpInfoArray;
    unsigned int memImpNum;
    struct QpImportInfoT* jettyImportInfoArray;
    unsigned int jettyImpNum;
};

// Jetty相关结构体（复用SDK的QpCreateAttr和QpCreateInfo）
struct JettyResInfo {
    struct QpCreateAttr in;
    struct QpCreateInfo out;
};

struct JettyRelatedResInfo {
    bool jfcShare;
    struct CqInfoT* jfcResInfoArray;
    unsigned int jfcNum;
    struct JettyResInfo* jettyResInfoArray;
    unsigned int jettyNum;
};

// TP相关结构体
struct TpResInfo {
    struct GetTpCfg cfg;
    unsigned int getAttrBitmap;
    unsigned int setAttrBitmap;
    struct TpAttr attr;
    struct HccpTpInfo tpInfo;
};

// Segment相关结构体
struct SegmentResInfo {
    struct MrRegInfoT* regInfoArray;
    unsigned int num;
};

// 批量RMA资源准备的主结构
struct RmaResInfo {
    void* rmaWrap;
    struct JettyRelatedResInfo jettyRelatedResInfo;
    struct TpResInfo tpResInfo;
    struct SegmentResInfo segResInfo;
};

struct RmaResErrorInfo {
    int resType; // 1.jetty相关 2.Tp相关 3.Segment相关 4.其他
    struct ErrorInfo errInfo;
};

static std::mutex g_fakeQpCacheMutex;
static std::unordered_map<uint64_t, FakeQpCache> g_fakeQpCache;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

static void* AllocateWqeBuff()
{
    uint8_t rspCmd;
    uint32_t rspLen = 0;
    RspGetDevPtrPayload rspPayload{};
    if (sim::GetAicpuProcMgr().Request(
            PIPE_CMD_GET_WQE_PTR, nullptr, 0, rspCmd, &rspPayload, sizeof(rspPayload), rspLen)
        != 0) {
        HCCL_VM_ERROR("Request PIPE_CMD_GET_WQE_PTR failed.");
        return nullptr;
    }
    HCCL_VM_INFO("alloc wqe buff ptr:{:p}", (void*)rspPayload.ptr);
    return (void*)rspPayload.ptr;
}

static void FreeWqeBuff(void* ptr)
{
    uint8_t rspCmd;
    uint32_t rspLen = 0;
    ReqDevPtrPayload freePayload{};
    RspFreeDevPtrPayload rspPayload{};
    freePayload.ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
    if (sim::GetAicpuProcMgr().Request(
            PIPE_CMD_FREE_WQE_PTR, &freePayload, sizeof(freePayload), rspCmd, &rspPayload, sizeof(rspPayload), rspLen)
        != 0) {
        HCCL_VM_ERROR("Request PIPE_CMD_FREE_WQE_PTR failed ");
    }
    HCCL_VM_INFO("free wqe buff ptr:{:p}", ptr);
    return;
}

int SimRaCustomChannel(void* tlvHandle, struct TlvMsg* sendMsg, struct TlvMsg* recvMsg);

//////////////////////RDMA/////////////////////////////

int RaSocketGetVnicIpInfos(
    unsigned int phyId, enum IdType type, unsigned int ids[], unsigned int num, struct IpInfo infos[])
{
    for (unsigned int i = 0; i < num; i++) {
        sim::runtime::Device device{};
        if (sim::runtime::GetDeviceByLogicId(ids[i], device) != ACL_SUCCESS) {
            HCCL_VM_ERROR("get device by phy id {} deviceId:{:d} failed.", phyId, ids[i]);
            continue;
        }
        uint64_t deviceIdx = device.id;
        auto endPointsResult = sim::runtime::Db::GetByPred<sim::runtime::EndPoint>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::EndPoint::device_id, deviceIdx),
            HcclSim::Storage::Eq(&sim::runtime::EndPoint::status, 1)));
        if (!endPointsResult.ok() || !endPointsResult.value.has_value() || endPointsResult.value->empty()) {
            HCCL_VM_ERROR("no endpoints found for device {:d}", deviceIdx);
            continue;
        }
        const auto& endPoints = *endPointsResult.value;

        infos[i].family = AF_INET6;
        std::string ipaddr = std::string(endPoints[0].ip_addr);
        if (ipaddr.find(':') == std::string::npos) {
            ipaddr = "::" + ipaddr;
        }
        IpAddress addr(ipaddr, AF_INET6);
        infos[i].ip.addr6 = addr.GetBinaryAddress().addr6;
        HCCL_VM_INFO("phyId:{:d} ids:{}, type:{:d} infos[{:d}]={}", phyId, ids[i], (int)type, i, endPoints[0].ip_addr);
    }

    return 0;
}

int RaGetInterfaceVersion(unsigned int phyId, unsigned int interfaceOpcode, unsigned int* interfaceVersion)
{
    *interfaceVersion = 2; // GET_UBOE_FLAG_ENABLE_VERSION
    return 0;
}

int RaGetTsqpDepth(void* rdevHandle, unsigned int* tempDepth, unsigned int* qpNum)
{
    HCCL_VM_WARN("is empty");
    return 0;
}
int RaSetTsqpDepth(void* rdevHandle, unsigned int tempDepth, unsigned int* qpNum)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaRdevGetSupportLite(void* rdmaHandle, int* supportLite)
{
    if (rdmaHandle == nullptr || supportLite == nullptr) {
        HCCL_VM_ERROR("invalid params, rdmaHandle: {}, supportLite: {}", rdmaHandle, (void*)supportLite);
        return -1;
    }
    *supportLite = 1;
    uint64_t handleVal = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdmaHandle));
    HCCL_VM_INFO("rdmaHandle:{:d} supportLite:1", handleVal);
    return 0;
}

int RaSocketSetWhiteListStatus(unsigned int enable)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaSocketGetWhiteListStatus(unsigned int* enable)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

extern struct ibv_context* CreateFakeIbvContext();
extern struct ibv_cq* CreateFakeCq(uint32_t cqe);
extern struct ibv_qp* CreateFakeQp(struct ibv_cq* sendCq, struct ibv_cq* recvCq, uint32_t qpNum);

static std::mutex g_fakeCqMapMutex;
static std::unordered_map<uint64_t, std::pair<struct ibv_cq*, struct ibv_cq*>> g_fakeCqByRaCqId;

int RaNormalQpCreate(void* rdevHandle, struct ibv_qp_init_attr* qpInitAttr, void** qpHandle, void** qp)
{
    if (rdevHandle == nullptr || qpInitAttr == nullptr || qpHandle == nullptr || qp == nullptr) {
        HCCL_VM_ERROR(
            "RaNormalQpCreate invalid params: rdevHandle={}, qpInitAttr={}, qpHandle={}, qp={}", (void*)rdevHandle,
            (void*)qpInitAttr, (void*)qpHandle, (void*)qp);
        return -1;
    }

    uint64_t raDevId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdevHandle));

    auto raDevOpt = sim::runtime::Db::GetById<sim::runtime::RaDevice>(raDevId);
    if (!raDevOpt.ok()) {
        HCCL_VM_ERROR("RaDevice {:d} not found", raDevId);
        return -1;
    }

    static std::atomic<uint32_t> s_normalQpNumCounter{1};
    uint32_t qpNum = (static_cast<uint32_t>(getpid() & 0xFFFF) << 16) | s_normalQpNumCounter.fetch_add(1);

    struct ibv_cq* sendCq = qpInitAttr->send_cq;
    struct ibv_cq* recvCq = qpInitAttr->recv_cq;
    if (sendCq == nullptr || recvCq == nullptr) {
        HCCL_VM_ERROR(
            "RaNormalQpCreate: qpInitAttr CQs are null sendCq={:p} recvCq={:p}", (void*)sendCq, (void*)recvCq);
        return -1;
    }

    auto* fakeQp = CreateFakeQp(sendCq, recvCq, qpNum);
    if (fakeQp == nullptr) {
        HCCL_VM_ERROR("RaNormalQpCreate failed to create fake QP");
        return -1;
    }

    sim::runtime::RaQP qpInfo{};
    qpInfo.state = 0;
    qpInfo.ra_dev_id = raDevId;
    qpInfo.qp_num = qpNum;
    qpInfo.type = qpInitAttr->qp_type;
    qpInfo.send_cq_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sendCq));
    qpInfo.recv_cq_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(recvCq));
    qpInfo.mode = 3;
    qpInfo.pid = static_cast<uint64_t>(getpid());

    qpInfo.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::RaQP>(qpInfo);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return -1;
    }
    auto id = inserted.value->value;
    void* qpId = reinterpret_cast<void*>(static_cast<uintptr_t>(id));
    *qpHandle = qpId;
    *qp = fakeQp;

    {
        std::lock_guard<std::mutex> lock(g_fakeQpCacheMutex);
        g_fakeQpCache[id] = {fakeQp, sendCq, recvCq, false, 0, nullptr, nullptr, nullptr};
    }

    RegisterDpuQp(fakeQp, id, qpNum, static_cast<uint32_t>(raDevOpt.value->device_id), sendCq, recvCq);

    HCCL_VM_INFO(
        "RaNormalQpCreate raDev={:d}, id={:d}, qp_num={:d}, type={:d}, "
        "qpHandle={:p}, qp={:p}, sendCq={:p}, recvCq={:p}",
        raDevId, id, qpInfo.qp_num, qpInfo.type, qpId, *qp, (void*)sendCq, (void*)recvCq);
    return 0;
}

// 清理单个 fake QP 的 g_dpuCtx 映射并释放 fake 资源（fakeQp、ownsCq 时的 CQ）。
// 本端与 loopback 配对端共用同一套逻辑，避免两份实现漂移。
static void CleanupOneFakeQp(const FakeQpCache& ctx)
{
    if (ctx.fakeQp != nullptr) {
        std::lock_guard<std::mutex> lock(g_dpuCtx.mtx);
        g_dpuCtx.qpInfo.erase(ctx.fakeQp);
        if (ctx.sendCq != nullptr) {
            g_dpuCtx.sendCqToQp.erase(ctx.sendCq);
            g_dpuCtx.pendingWqeCount.erase(ctx.sendCq);
        }
        if (ctx.recvCq != nullptr) {
            g_dpuCtx.recvCqToQp.erase(ctx.recvCq);
        }
    }
    delete ctx.fakeQp;
    if (ctx.ownsCq) {
        delete ctx.sendCq;
        delete ctx.recvCq;
    }
}

// 线程安全地从 g_fakeQpCache 查询 fakeQp 指针，未命中返回 nullptr。
static struct ibv_qp* LookupFakeQp(uint64_t qpId)
{
    std::lock_guard<std::mutex> lock(g_fakeQpCacheMutex);
    auto it = g_fakeQpCache.find(qpId);
    return (it != g_fakeQpCache.end()) ? it->second.fakeQp : nullptr;
}

// 统一清理 fake QP/CQ 资源：从 g_fakeQpCache 移除、清 g_dpuCtx 映射、delete 指针。
// ownsCq=true 时 CQ 由本处内部创建（如 loopback），需 delete；否则 CQ 归 RaCqCreate 所有，不在此 delete。
// 若存在 loopback 配对端，一并清理其 fake 资源、g_dpuCtx 映射与 DB 记录，避免泄漏。
// 注意：本端 QP 的 DB 记录(RaQP)由调用方删除，配对端 DB 记录在此删除。
static void CleanupFakeQpResources(uint64_t qpId)
{
    FakeQpCache entry{};
    FakeQpCache paired{};
    bool hasPaired = false;

    {
        std::lock_guard<std::mutex> lock(g_fakeQpCacheMutex);
        auto it = g_fakeQpCache.find(qpId);
        if (it == g_fakeQpCache.end()) {
            return;
        }
        entry = it->second;
        g_fakeQpCache.erase(it);

        if (entry.pairedQpId != 0 && entry.pairedFakeQp != nullptr) {
            auto pit = g_fakeQpCache.find(entry.pairedQpId);
            if (pit != g_fakeQpCache.end()) {
                paired = pit->second;
                g_fakeQpCache.erase(pit);
                hasPaired = true;
            }
        }
    }

    CleanupOneFakeQp(entry);

    if (hasPaired) {
        CleanupOneFakeQp(paired);
        (void)sim::runtime::Db::Delete<sim::runtime::RaQP>(
            HcclSim::Storage::Eq(&sim::runtime::RaQP::id, entry.pairedQpId));
        HCCL_VM_INFO("CleanupFakeQpResources: cleaned paired QP id={:d} for QP id={:d}", entry.pairedQpId, qpId);
    }
}

int RaNormalQpDestroy(void* qpHandle)
{
    if (qpHandle == nullptr) {
        HCCL_VM_ERROR("qpHandle is null");
        return HCCL_E_PTR;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));

    CleanupFakeQpResources(qpId);

    (void)sim::runtime::Db::Delete<sim::runtime::RaQP>(HcclSim::Storage::Eq(&sim::runtime::RaQP::id, qpId));

    HCCL_VM_INFO("RaNormalQpDestroy deleted QP id={:d}", qpId);
    return 0;
}

int RaMrReg(void* qpHandle, struct MrInfoT* info)
{
    if (qpHandle == nullptr || info == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, info: {}", qpHandle, (void*)info);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(qpId);
    if (!qpOpt.ok()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    sim::runtime::RaMR mr{};
    mr.addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(info->addr));
    mr.length = info->size;
    mr.local_key = (qpId << 32) | (rand() & 0xFFFFFFFF);
    mr.remote_key = mr.local_key | 0x1;

    mr.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::RaMR>(mr);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return -1;
    }
    auto mrId = inserted.value->value;
    info->lkey = mr.local_key;
    info->rkey = mr.remote_key;

    HCCL_VM_INFO(
        "QP {:d} reg MR id:{:d}, addr:{:x}, len:{:d}, lkey:{:x}, rkey:{:x}", qpId, mrId, mr.addr, mr.length,
        mr.local_key, mr.remote_key);
    return 0;
}

int RaMrDereg(void* qpHandle, struct MrInfoT* info)
{
    if (qpHandle == nullptr || info == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, info: {}", qpHandle, (void*)info);
        return -1;
    }

    auto mrResult = sim::runtime::Db::GetByPred<sim::runtime::RaMR>(
        HcclSim::Storage::Eq(&sim::runtime::RaMR::local_key, info->lkey));
    if (!mrResult.ok() || !mrResult.value.has_value()) {
        return -1;
    }
    const auto& mrList = *mrResult.value;

    if (mrList.empty()) {
        HCCL_VM_WARN("MR with lkey {:x} not found", info->lkey);
        return 0;
    }

    for (const auto& mr : mrList) {
        (void)sim::runtime::Db::Delete<sim::runtime::RaMR>(HcclSim::Storage::Eq(&sim::runtime::RaMR::id, mr.id));
        HCCL_VM_INFO("dereg MR id:{:d}, lkey:{:x}", mr.id, mr.local_key);
    }

    return 0;
}

int RaRegisterMr(const void* rdmaHandle, struct MrInfoT* info, void** mrHandle)
{
    if (rdmaHandle == nullptr || info == nullptr || mrHandle == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, rdmaHandle: {}, info: {}, mrHandle: {}", rdmaHandle, (void*)info, (void*)mrHandle);
        return -1;
    }

    uint64_t raDevId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdmaHandle));

    sim::runtime::RaMR mr{};
    mr.addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(info->addr));
    mr.length = info->size;
    mr.local_key = (raDevId << 32) | (rand() & 0xFFFFFFFF);
    mr.remote_key = mr.local_key | 0x1;

    mr.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::RaMR>(mr);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return -1;
    }
    auto mrId = inserted.value->value;
    *mrHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(mrId));
    info->lkey = mr.local_key;
    info->rkey = mr.remote_key;

    HCCL_VM_INFO("RaDev {:d} reg MR id:{:d}, addr:{:x}, len:{:d}", raDevId, mrId, mr.addr, mr.length);
    return 0;
}

int RaRemapMr(const void* rdmaHandle, struct MemRemapInfo info[], unsigned int num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaDeregisterMr(const void* rdmaHandle, void* mrHandle)
{
    if (rdmaHandle == nullptr || mrHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, rdmaHandle: {}, mrHandle: {}", rdmaHandle, mrHandle);
        return -1;
    }

    uint64_t mrId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mrHandle));
    auto mrOpt = sim::runtime::Db::GetById<sim::runtime::RaMR>(mrId);
    if (!mrOpt.ok()) {
        HCCL_VM_WARN("MR {:d} not found", mrId);
        return 0;
    }

    (void)sim::runtime::Db::Delete<sim::runtime::RaMR>(HcclSim::Storage::Eq(&sim::runtime::RaMR::id, mrId));
    HCCL_VM_INFO("dereg MR id:{:d}", mrId);
    return 0;
}

int RaSendWr(void* qpHandle, struct SendWr* wr, struct SendWrRsp* opRsp)
{
    if (qpHandle == nullptr || wr == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, wr: {}", qpHandle, (void*)wr);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(qpId);
    if (!qpOpt.ok()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (qpOpt.value->state != 3) {
        HCCL_VM_ERROR("QP {:d} not in RTS state, current state:{:d}", qpId, qpOpt.value->state);
        return -1;
    }

    HCCL_VM_INFO("QP {:d} send op:{:d}, bufNum:{:d}, dstAddr:{:x}", qpId, wr->op, wr->bufNum, wr->dstAddr);
    return 0;
}

int RaSetQpAttrQos(void* qpHandle, struct QosAttr* attr)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaSetQpAttrTimeout(void* qpHandle, unsigned int* timeout)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaSetQpAttrRetryCnt(void* qpHandle, unsigned int* retryCnt)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaGetCqeErrInfo(unsigned int phyId, struct CqeErrInfo* info)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaCreateSrq(const void* rdmaHandle, struct SrqAttr* attr)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaDestroySrq(const void* rdmaHandle, struct SrqAttr* attr)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaCreateEventHandle(int* eventHandle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaCtlEventHandle(int eventHandle, const void* fdHandle, int opcode, enum RaEpollEvent event)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaWaitEventHandle(
    int eventHandle, struct SocketEventInfoT* eventInfos, int timeout, unsigned int maxevents, unsigned int* eventsNum)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaDestroyEventHandle(int* eventHandle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaCreateCompChannel(const void* rdmaHandle, void** compChannel)
{
    *compChannel = reinterpret_cast<void*>(static_cast<uintptr_t>(0xabcdU));
    return ((rdmaHandle == nullptr) || (compChannel == nullptr)) ? -1 : 0;
}

int RaDestroyCompChannel(const void* rdmaHandle, void* compChannel)
{
    return ((rdmaHandle == nullptr) || (compChannel == nullptr)) ? -1 : 0;
}

int RaLoopbackQpCreate(void* rdevHandle, struct LoopbackQpPair* qpPair, void** qpHandle)
{
    if (rdevHandle == nullptr || qpPair == nullptr || qpHandle == nullptr) {
        HCCL_VM_ERROR("RaLoopbackQpCreate invalid params");
        return -1;
    }

    uint64_t raDevId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdevHandle));
    auto raDevOpt = sim::runtime::Db::GetById<sim::runtime::RaDevice>(raDevId);
    if (!raDevOpt.ok()) {
        HCCL_VM_ERROR("RaLoopbackQpCreate: RaDevice {:d} not found", raDevId);
        return -1;
    }

    static std::atomic<uint32_t> s_lbQpNumCounter{0x80000000};
    uint32_t qpNum0 = s_lbQpNumCounter.fetch_add(1);
    uint32_t qpNum1 = s_lbQpNumCounter.fetch_add(1);

    auto* sendCq0 = CreateFakeCq(16);
    auto* recvCq0 = CreateFakeCq(16);
    auto* qp0 = CreateFakeQp(sendCq0, recvCq0, qpNum0);

    auto* sendCq1 = CreateFakeCq(16);
    auto* recvCq1 = CreateFakeCq(16);
    auto* qp1 = CreateFakeQp(sendCq1, recvCq1, qpNum1);

    sim::runtime::RaQP qpInfo0{};
    qpInfo0.state = 0;
    qpInfo0.ra_dev_id = raDevId;
    qpInfo0.qp_num = qpNum0;
    qpInfo0.type = 2;
    qpInfo0.mode = 3;
    qpInfo0.pid = static_cast<uint64_t>(getpid());
    qpInfo0.id = 0;
    auto inserted0 = sim::runtime::Db::Add<sim::runtime::RaQP>(qpInfo0);
    if (!inserted0.ok() || !inserted0.value.has_value()) {
        return -1;
    }
    auto id0 = inserted0.value->value;

    sim::runtime::RaQP qpInfo1{};
    qpInfo1.state = 0;
    qpInfo1.ra_dev_id = raDevId;
    qpInfo1.qp_num = qpNum1;
    qpInfo1.type = 2;
    qpInfo1.mode = 3;
    qpInfo1.pid = static_cast<uint64_t>(getpid());
    qpInfo1.id = 0;
    auto inserted1 = sim::runtime::Db::Add<sim::runtime::RaQP>(qpInfo1);
    if (!inserted1.ok() || !inserted1.value.has_value()) {
        return -1;
    }
    auto id1 = inserted1.value->value;

    qpPair->ibvQp0 = qp0;
    qpPair->ibvQp1 = qp1;
    *qpHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(id0));

    {
        std::lock_guard<std::mutex> lock(g_fakeQpCacheMutex);
        // loopback 的 CQ 由本处内部创建（ownsCq=true），销毁时需 delete；
        // 同时记录配对端，以便任一端销毁时清理另一端的 fake 资源与 g_dpuCtx 映射。
        g_fakeQpCache[id0] = {qp0, sendCq0, recvCq0, true, id1, qp1, sendCq1, recvCq1};
        g_fakeQpCache[id1] = {qp1, sendCq1, recvCq1, true, id0, qp0, sendCq0, recvCq0};
    }

    const uint32_t localDeviceId = static_cast<uint32_t>(raDevOpt.value->device_id);
    RegisterDpuQp(qp0, id0, qpNum0, localDeviceId, sendCq0, recvCq0);
    RegisterDpuQp(qp1, id0, qpNum1, localDeviceId, sendCq1, recvCq1);

    HCCL_VM_INFO(
        "RaLoopbackQpCreate raDev={:d}, id0={:d}, id1={:d}, qpNum0={:d}, qpNum1={:d}, "
        "qp0={:p}, qp1={:p}",
        raDevId, id0, id1, qpNum0, qpNum1, (void*)qp0, (void*)qp1);
    return 0;
}

int RaQpConnectAsync(void* qpHandle, const void* fdHandle)
{
    if (qpHandle == nullptr) {
        HCCL_VM_ERROR("qpHandle is null");
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(qpId);
    if (!qpOpt.ok()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    auto update = sim::runtime::Db::Update<sim::runtime::RaQP>(
        HcclSim::Storage::Eq(&sim::runtime::RaQP::id, qpId), HcclSim::Storage::Set(&sim::runtime::RaQP::state, 3));
    if (!update.ok()) {
        return -1;
    }

    HCCL_VM_INFO("QP {:d} connected, state -> RTS", qpId);
    return 0;
}

int RaSendWrV2(void* qpHandle, struct SendWrV2* wr, struct SendWrRsp* opRsp)
{
    if (qpHandle == nullptr || wr == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, wr: {}", qpHandle, (void*)wr);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(qpId);
    if (!qpOpt.ok()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (qpOpt.value->state != 3) {
        HCCL_VM_ERROR("QP {:d} not in RTS state", qpId);
        return -1;
    }

    HCCL_VM_INFO("QP {:d} send V2", qpId);
    return 0;
}

int RaPollCq(void* qpHandle, bool isSendCq, unsigned int numEntries, void* wc)
{
    if (qpHandle == nullptr) {
        HCCL_VM_ERROR("qpHandle is null");
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(qpId);
    if (!qpOpt.ok()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    uint64_t cqId = isSendCq ? qpOpt.value->send_cq_handle : qpOpt.value->recv_cq_handle;

    auto cqOpt = sim::runtime::Db::GetById<sim::runtime::RaCQ>(cqId);
    if (!cqOpt.ok()) {
        HCCL_VM_WARN("CQ {:d} not found for QP {:d}", cqId, qpId);
        return 0;
    }

    auto cqeResult = sim::runtime::Db::GetByPred<sim::runtime::RaCQE>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::RaCQE::cq_handle, cqId),
        HcclSim::Storage::Eq(&sim::runtime::RaCQE::status, 0)));
    if (!cqeResult.ok() || !cqeResult.value.has_value()) {
        return -1;
    }
    const auto& cqeList = *cqeResult.value;

    unsigned int polled = 0;
    for (const auto& cqe : cqeList) {
        if (polled >= numEntries) {
            break;
        }
        (void)sim::runtime::Db::Delete<sim::runtime::RaCQE>(HcclSim::Storage::Eq(&sim::runtime::RaCQE::id, cqe.id));
        polled++;
    }

    HCCL_VM_INFO("QP {:d} CQ {:d} polled {:d} entries (isSendCq:{:d})", qpId, cqId, polled, isSendCq);
    return polled;
}

int RaRecvWrlist(void* qpHandle, struct RecvWrlistData* wr, unsigned int recvNum, unsigned int* completeNum)
{
    if (qpHandle == nullptr || wr == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, wr: {}", qpHandle, (void*)wr);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(qpId);
    if (!qpOpt.ok()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (qpOpt.value->state != 3) {
        HCCL_VM_ERROR("QP {:d} not in RTS state", qpId);
        return -1;
    }

    if (completeNum != nullptr) {
        *completeNum = recvNum;
    }

    HCCL_VM_INFO("QP {:d} post {:d} recv WRs", qpId, recvNum);
    return 0;
}

int RaGetQpContext(void* qpHandle, void** qp, void** sendCq, void** recvCq)
{
    if (qpHandle == nullptr) {
        HCCL_VM_ERROR("qpHandle is null");
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(qpId);
    if (!qpOpt.ok()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (qp != nullptr) {
        struct ibv_qp* fakeQp = LookupFakeQp(qpId);
        *qp = (fakeQp != nullptr) ? fakeQp : qpHandle;
    }
    if (sendCq != nullptr) {
        *sendCq = reinterpret_cast<void*>(static_cast<uintptr_t>(qpOpt.value->send_cq_handle));
    }
    if (recvCq != nullptr) {
        *recvCq = reinterpret_cast<void*>(static_cast<uintptr_t>(qpOpt.value->recv_cq_handle));
    }

    HCCL_VM_INFO("QP {:d} sendCq:{:d} recvCq:{:d}", qpId, qpOpt.value->send_cq_handle, qpOpt.value->recv_cq_handle);
    return 0;
}

int RaQpBatchModify(void* rdmaHandle, void* qpHandle[], unsigned int num, int expectStatus)
{
    if (qpHandle == nullptr || num == 0) {
        HCCL_VM_ERROR("invalid params, rdmaHandle: {}", rdmaHandle);
        return -1;
    }

    unsigned int modified = 0;
    for (unsigned int i = 0; i < num; i++) {
        if (qpHandle[i] == nullptr) {
            continue;
        }
        uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle[i]));
        auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(qpId);
        if (!qpOpt.ok()) {
            HCCL_VM_WARN("QP {:d} not found", qpId);
            continue;
        }

        auto update = sim::runtime::Db::Update<sim::runtime::RaQP>(
            HcclSim::Storage::Eq(&sim::runtime::RaQP::id, qpId),
            HcclSim::Storage::Set(&sim::runtime::RaQP::state, expectStatus));
        if (!update.ok()) {
            return -1;
        }
        modified++;
    }

    HCCL_VM_INFO("batch modify {:d} QPs to state {:d}", modified, expectStatus);
    return 0;
}

int RaRdevGetCqeErrInfoList(void* rdmaHandle, struct CqeErrInfo* infoList, unsigned int* num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaRdevGetHandle(unsigned int phyId, void** rdmaHandle)
{
    if (rdmaHandle == nullptr) {
        HCCL_VM_ERROR("rdmaHandle is null");
        return -1;
    }

    sim::runtime::Device device{};
    if (sim::runtime::GetDeviceByPhysicalId(phyId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by phy id {} failed.", phyId);
        return -1;
    }

    auto raDevRes = sim::runtime::Db::GetOneByPred<sim::runtime::RaDevice>(
        HcclSim::Storage::Eq(&sim::runtime::RaDevice::device_id, device.id));

    if (raDevRes.ok()) {
        *rdmaHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(raDevRes.value->id));
        HCCL_VM_INFO("found RaDevice id:{:d} for phyId:{:d}", raDevRes.value->id, phyId);
        return 0;
    }

    HCCL_VM_ERROR("no RaDevice found for phyId:{:d}", phyId);
    return -1;
}

int RaSaveSnapshot(struct RaInfo* info, enum SaveSnapshotAction action)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaRestoreSnapshot(struct RaInfo* info)
{
    HCCL_VM_WARN("is empty");
    return 0;
}
int RaRdevInitWithBackup(
    struct RdevInitInfo* initInfo, struct rdev* rdevInfo, struct rdev* backupRdevInfo, void** rdmaHandle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaGetQpStatus(void* qpHandle, int* status)
{
    if (qpHandle == nullptr || status == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, status: {}", qpHandle, (void*)status);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(qpId);
    if (!qpOpt.ok()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    *status = qpOpt.value->state;
    HCCL_VM_INFO("QP {:d} status:{:d}", qpId, *status);
    return 0;
}

int RaSendWrlist(
    void* qpHandle, struct SendWrlistData wr[], struct SendWrRsp opRsp[], unsigned int sendNum,
    unsigned int* completeNum)
{
    if (qpHandle == nullptr || wr == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, wr: {}", qpHandle, (void*)wr);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(qpId);
    if (!qpOpt.ok()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (qpOpt.value->state != 3) {
        HCCL_VM_ERROR("QP {:d} not in RTS state", qpId);
        return -1;
    }

    unsigned int completed = sendNum;

    if (completeNum != nullptr) {
        *completeNum = completed;
    }

    HCCL_VM_INFO("QP {:d} send {:d} WRs, completed:{:d}", qpId, sendNum, completed);
    return 0;
}

int RaSendWrlistExt(
    void* qpHandle, struct SendWrlistDataExt wr[], struct SendWrRsp opRsp[], unsigned int sendNum,
    unsigned int* completeNum)
{
    if (qpHandle == nullptr || wr == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, wr: {}", qpHandle, (void*)wr);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(qpId);
    if (!qpOpt.ok()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (qpOpt.value->state != 3) {
        HCCL_VM_ERROR("QP {:d} not in RTS state", qpId);
        return -1;
    }

    unsigned int completed = sendNum;

    if (completeNum != nullptr) {
        *completeNum = completed;
    }

    HCCL_VM_INFO("QP {:d} send {:d} WRs (Ext), completed:{:d}", qpId, sendNum, completed);
    return 0;
}

int RaSendNormalWrlist(
    void* qpHandle, struct WrInfo wr[], struct SendWrRsp opRsp[], unsigned int sendNum, unsigned int* completeNum)
{
    if (qpHandle == nullptr || wr == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, wr: {}", qpHandle, (void*)wr);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(qpId);
    if (!qpOpt.ok()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (qpOpt.value->state != 3) {
        HCCL_VM_ERROR("QP {:d} not in RTS state", qpId);
        return -1;
    }

    unsigned int completed = sendNum;

    if (completeNum != nullptr) {
        *completeNum = completed;
    }

    HCCL_VM_INFO("QP {:d} send {:d} normal WRs, completed:{:d}", qpId, sendNum, completed);
    return 0;
}

int RaGetNotifyBaseAddr(void* rdevHandle, unsigned long long* va, unsigned long long* size)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaGetNotifyMrInfo(void* rdevHandle, struct MrInfoT* info)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaInit(struct RaInitConfig* config)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaDeinit(struct RaInitConfig* config)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaGetTlsEnable(struct RaInfo* info, bool* tlsEnable)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaGetHccnCfg(struct RaInfo* info, enum HccnCfgKey key, char* value, unsigned int* valueLen)
{
    // 对齐真实实现: hccp ra_common.c RaGetHccnCfg -> rs.c RsGetHccnCfg -> file_opt.c FileReadCfg
    static constexpr unsigned int HCCN_CFG_MSG_DATA_LEN = 2048U;
    static constexpr unsigned int RA_MAX_PHY_ID_NUM = 64U;
    static const char* HCCN_OFFLINE_CFGFILE_PATH = "/etc/hccl.cfg";
    static const char* HCCN_PEER_ONLINE_CFGFILE_PATH = "/etc/hcomm.cfg";
    static const char* HCCN_CFG_KEY_NAMES[]
        = {"udp_port_mode", "multi_qp_count", "multi_qp_udp_ports", "resv_mem", "qos_dscp"};

    if (info == nullptr || value == nullptr || valueLen == nullptr) {
        HCCL_VM_ERROR("invalid params, info: {}, value: {}, valueLen: {}", (void*)info, (void*)value, (void*)valueLen);
        return -EINVAL;
    }
    if (*valueLen < HCCN_CFG_MSG_DATA_LEN) {
        HCCL_VM_ERROR("valueLen[{}] < min_len[{}]", *valueLen, HCCN_CFG_MSG_DATA_LEN);
        return -EINVAL;
    }
    if (info->phyId >= RA_MAX_PHY_ID_NUM) {
        HCCL_VM_ERROR("phyId[{}] must be smaller than {}", info->phyId, RA_MAX_PHY_ID_NUM);
        return -EINVAL;
    }
    auto cfgKey = static_cast<unsigned int>(key);
    if (cfgKey >= static_cast<unsigned int>(HCCN_CFG_KEY_INVALID)) {
        HCCL_VM_ERROR("key[{}] should < {}", cfgKey, static_cast<unsigned int>(HCCN_CFG_KEY_INVALID));
        return -EINVAL;
    }

    const char* cfgFilePath = nullptr;
    if (info->mode == NETWORK_PEER_ONLINE) {
        cfgFilePath = HCCN_PEER_ONLINE_CFGFILE_PATH;
    } else if (info->mode == NETWORK_OFFLINE) {
        cfgFilePath = HCCN_OFFLINE_CFGFILE_PATH;
    } else {
        HCCL_VM_ERROR("do not support mode[{}] phyId[{}]", info->mode, info->phyId);
        return -ENOTSUP;
    }

    const unsigned int bufLen = *valueLen;
    *valueLen = 0;
    std::ifstream cfgFile(cfgFilePath);
    if (!cfgFile.is_open()) {
        HCCL_VM_WARN("open hccn cfg file[{}] failed, treat as empty cfg", cfgFilePath);
        return 0;
    }
    const std::string confName = std::string(HCCN_CFG_KEY_NAMES[cfgKey]) + "_" + std::to_string(info->phyId);
    std::string line;
    while (std::getline(cfgFile, line)) {
        if (line.size() < 3 || line[0] == '#') {
            continue;
        }
        const auto eqPos = line.find('=');
        if (eqPos == std::string::npos || eqPos != confName.size() || line.compare(0, eqPos, confName) != 0) {
            continue;
        }
        const std::string confValue = line.substr(eqPos + 1);
        if (confValue.size() + 1 > bufLen) {
            HCCL_VM_ERROR("value_len[{}] > buf_len[{}]", confValue.size() + 1, bufLen);
            return -ENOMEM;
        }
        std::copy(confValue.begin(), confValue.end(), value);
        value[confValue.size()] = '\0';
        if (!confValue.empty()) {
            *valueLen = static_cast<unsigned int>(confValue.size()) + 1;
        }
        HCCL_VM_INFO("get hccn cfg[{}] value[{}] len[{}] from file[{}]", confName, value, *valueLen, cfgFilePath);
        return 0;
    }
    HCCL_VM_INFO("hccn cfg[{}] not found in file[{}], treat as empty cfg", confName, cfgFilePath);
    return 0;
}

int RaRdevInitV2(struct RdevInitInfo initInfo, struct rdev rdevInfo, void** rdmaHandle)
{
    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        HCCL_VM_ERROR("GetCurServerId failed");
        return -1;
    }

    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        HCCL_VM_ERROR("GetCurrRunnerTls failed");
        return -1;
    }
    auto currCtx = sim::runtime::Db::GetById<sim::runtime::Context>(runner.current_ctx_id);
    if (!currCtx.ok()) {
        HCCL_VM_ERROR("can not get CurrContext: {:d}", runner.current_ctx_id);
        return -1;
    }

    sim::runtime::RaDevice dev{};
    dev.device_id = currCtx.value->device_id;
    dev.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::RaDevice>(dev);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return -1;
    }
    auto id = inserted.value->value;

    *rdmaHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(id));
    HCCL_VM_INFO("add radev id {:d}", id);
    return 0;
}

int RaRdevInit(int mode, unsigned int notifyType, struct rdev rdevInfo, void** rdmaHandle)
{
    struct RdevInitInfo initInfo;
    RaRdevInitV2(initInfo, rdevInfo, rdmaHandle);
    return 0;
}

int RaRdevDeinit(void* rdmaHandle, unsigned int notifyType)
{
    uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdmaHandle));
    (void)sim::runtime::Db::Delete<sim::runtime::RaDevice>(HcclSim::Storage::Eq(&sim::runtime::RaDevice::id, id));
    HCCL_VM_INFO("delete id {:d}", id);
    return 0;
}

int RaCqCreate(void* rdevHandle, struct CqAttr* attr)
{
    uint64_t raDevId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdevHandle));
    sim::runtime::RaCQ cq{};
    cq.ra_dev_id = raDevId;
    cq.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::RaCQ>(cq);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return -1;
    }
    auto id = inserted.value->value;

    *(attr->qpContext) = CreateFakeIbvContext();

    uint32_t sendDepth = attr->sendCqDepth > 0 ? static_cast<uint32_t>(attr->sendCqDepth) : 128;
    uint32_t recvDepth = attr->recvCqDepth > 0 ? static_cast<uint32_t>(attr->recvCqDepth) : 128;
    auto* sendCq = CreateFakeCq(sendDepth);
    auto* recvCq = CreateFakeCq(recvDepth);
    if (sendCq == nullptr || recvCq == nullptr) {
        HCCL_VM_ERROR("RaCqCreate failed to create fake CQ, sendCq={}, recvCq={}", (void*)sendCq, (void*)recvCq);
        delete sendCq;
        delete recvCq;
        return -1;
    }

    *(attr->ibSendCq) = sendCq;
    *(attr->ibRecvCq) = recvCq;

    {
        std::lock_guard<std::mutex> lock(g_fakeCqMapMutex);
        g_fakeCqByRaCqId[id] = {sendCq, recvCq};
    }

    HCCL_VM_INFO("RaDev {:d} add RaCQ id:{:d}, sendCq:{:p}, recvCq:{:p}", raDevId, id, (void*)sendCq, (void*)recvCq);
    return 0;
}

int RaCqDestroy(void* rdevHandle, struct CqAttr* attr)
{
    uint64_t raDevId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdevHandle));
    auto* sendCq = (attr && attr->ibSendCq) ? *(attr->ibSendCq) : nullptr;
    auto* recvCq = (attr && attr->ibRecvCq) ? *(attr->ibRecvCq) : nullptr;

    {
        std::lock_guard<std::mutex> lock(g_fakeCqMapMutex);
        for (auto it = g_fakeCqByRaCqId.begin(); it != g_fakeCqByRaCqId.end(); ++it) {
            if (it->second.first == sendCq && it->second.second == recvCq) {
                (void)sim::runtime::Db::Delete<sim::runtime::RaCQ>(
                    HcclSim::Storage::Eq(&sim::runtime::RaCQ::id, it->first));
                g_fakeCqByRaCqId.erase(it);
                break;
            }
        }
    }

    delete sendCq;
    delete recvCq;

    HCCL_VM_INFO("RaDev {:d} delete RaCQ, sendCq:{:p}, recvCq:{:p}", raDevId, (void*)sendCq, (void*)recvCq);
    return 0;
}

int RaQpCreate(void* rdevHandle, int flag, int qpMode, void** qpHandle)
{
    if (rdevHandle == nullptr || qpHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, rdevHandle: {}, qpHandle: {}", rdevHandle, (void*)qpHandle);
        return -1;
    }

    uint64_t raDevId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdevHandle));

    auto raDevOpt = sim::runtime::Db::GetById<sim::runtime::RaDevice>(raDevId);
    if (!raDevOpt.ok()) {
        HCCL_VM_ERROR("RaDevice {:d} not found", raDevId);
        return -1;
    }

    static std::atomic<uint32_t> s_raQpNumCounter{1};

    sim::runtime::RaQP qp{};
    qp.state = 0;
    qp.ra_dev_id = raDevId;
    qp.qp_num = s_raQpNumCounter.fetch_add(1);
    qp.type = qpMode;
    qp.send_cq_handle = 0;
    qp.recv_cq_handle = 0;
    qp.peer_qpn = 0;
    qp.perr_lid = 0;
    qp.taJettyId = 0;

    qp.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::RaQP>(qp);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return -1;
    }
    auto id = inserted.value->value;
    *qpHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(id));

    HCCL_VM_INFO("RaDev {:d} create QP id:{:d}, qp_num:{:d}, type:{:d}", raDevId, id, qp.qp_num, qp.type);
    return 0;
}

int RaQpCreateWithAttrs(void* rdevHandle, struct QpExtAttrs* extAttrs, void** qpHandle)
{
    if (rdevHandle == nullptr || qpHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, rdevHandle: {}, qpHandle: {}", rdevHandle, (void*)qpHandle);
        return -1;
    }

    uint64_t raDevId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdevHandle));

    auto raDevOpt = sim::runtime::Db::GetById<sim::runtime::RaDevice>(raDevId);
    if (!raDevOpt.ok()) {
        HCCL_VM_ERROR("RaDevice {:d} not found", raDevId);
        return -1;
    }

    static std::atomic<uint32_t> s_qpWithAttrsCounter{1000};

    sim::runtime::RaQP qp{};
    qp.state = 0;
    qp.ra_dev_id = raDevId;
    qp.qp_num = s_qpWithAttrsCounter.fetch_add(1);
    qp.send_cq_handle = 0;
    qp.recv_cq_handle = 0;
    qp.peer_qpn = 0;
    qp.perr_lid = 0;
    qp.taJettyId = 0;

    if (extAttrs != nullptr) {
        qp.type = extAttrs->qpMode;
    } else {
        qp.type = 0;
    }

    qp.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::RaQP>(qp);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return -1;
    }
    auto id = inserted.value->value;
    *qpHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(id));

    HCCL_VM_INFO("RaDev {:d} create QP id:{:d}, qp_num:{:d}, qpMode:{:d}", raDevId, id, qp.qp_num, qp.type);
    return 0;
}

int RaAiQpCreate(void* rdevHandle, struct QpExtAttrs* attrs, struct AiQpInfo* info, void** qpHandle)
{
    int ret = RaQpCreate(rdevHandle, 0, 0, qpHandle);
    return ret;
}

int RaTypicalQpCreate(void* rdevHandle, int flag, int qpMode, struct TypicalQp* qpInfo, void** qpHandle)
{
    int ret = RaQpCreate(rdevHandle, 0, 0, qpHandle);
    return ret;
}

int RaCtxInit(struct CtxInitCfg* cfg, struct CtxInitAttr* attr, void** ctxHandle)
{
    // 判断eid是否为全0
    static constexpr uint8_t zeroEid[URMA_EID_LEN] = {0};
    if (memcmp(attr->ub.eid.raw, zeroEid, URMA_EID_LEN) == 0) {
        HCCL_VM_WARN("skip ctx info. eid is zero");
        return 0;
    }

    Eid simEid{};
    memcpy(simEid.raw, attr->ub.eid.raw, sizeof(simEid));
    auto addr = IpAddress(simEid);
    auto eidStr = addr.EidToHexString();
    HCCL_VM_INFO("eid:{}", eidStr.c_str());

    sim::runtime::EndPoint endPoint{};
    if (sim::runtime::GetEndPointByEidOrIpV4(addr, endPoint) != 0) {
        HCCL_VM_ERROR("cannot find endpoint eid:{}", eidStr.c_str());
        return -1;
    }

    sim::runtime::RaContext devRaCtx{};
    devRaCtx.device_id = endPoint.device_id;
    devRaCtx.endpoint_id = endPoint.id;
    devRaCtx.eidIndex = attr->ub.eidIndex;
    devRaCtx.mode = cfg->mode;
    devRaCtx.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::RaContext>(devRaCtx);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return -1;
    }
    auto ctxId = inserted.value->value;

    sim::runtime::RaTp tp{};
    tp.ctx_handle = ctxId;
    tp.id = 0;
    auto tpInserted = sim::runtime::Db::Add<sim::runtime::RaTp>(tp);
    if (!tpInserted.ok()) {
        // 失败路径回滚已插入的 RaContext，避免遗留孤儿记录。
        (void)sim::runtime::Db::Delete<sim::runtime::RaContext>(
            HcclSim::Storage::Eq(&sim::runtime::RaContext::id, ctxId));
        return -1;
    }

    HCCL_VM_INFO("add ctx id {:d}, eid {}", ctxId, eidStr.c_str());
    *ctxHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(ctxId));
    return 0;
}

int RaCtxDeinit(void* ctxHandle)
{
    uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    HCCL_VM_INFO("soft delete ctx id {:d}, for checker use after", id);
    return 0;
}

int RaGetDevBaseAttr(void* ctxHandle, struct DevBaseAttr* attr)
{
    uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    auto devCtx = sim::runtime::Db::GetById<sim::runtime::RaContext>(id);
    if (!devCtx.ok()) {
        HCCL_VM_ERROR("can not find dev by id:{:d}", id);
        return -1;
    }

    auto endPoint = sim::runtime::Db::GetById<sim::runtime::EndPoint>(devCtx.value->endpoint_id);
    if (!endPoint.ok()) {
        HCCL_VM_ERROR("can not find endpoint:{:d}", devCtx.value->endpoint_id);
        return -1;
    }

    attr->ub.dieId = static_cast<uint32_t>(endPoint.value->die_id);
    attr->ub.funcId = endPoint.value->func_id;

    for (int i = 0; i < MAX_PRIORITY_CNT; i++) {
        CtxSlInfo& priorityInfo = attr->ub.priorityInfo[i];
        priorityInfo.tpType.bs.rtp = 1;
        priorityInfo.tpType.bs.ctp = 1;
    }

    attr->sqMaxDepth = 8192;                // max_jfs_depth
    attr->rqMaxDepth = 32768;               // max_jfr_depth
    attr->sqMaxSge = 13;                    // max_jfs_sge
    attr->rqMaxSge = 4;                     // max_jfr_sge
    attr->maxReadSize = 256 * 1024 * 1024;  // max_read_size
    attr->maxWriteSize = 256 * 1024 * 1024; // max_write_size
    attr->maxMsgSize = 65536;               // max_msg_size

    return 0;
}

int RaQpDestroy(void* qpHandle)
{
    if (qpHandle == nullptr) {
        HCCL_VM_ERROR("qpHandle is null");
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(qpId);
    if (!qpOpt.ok()) {
        HCCL_VM_WARN("QP {:d} not found", qpId);
        return 0;
    }

    // 统一清理 fake QP/CQ 资源与 g_dpuCtx 映射（含 loopback 配对端），避免悬空指针与 double-free
    CleanupFakeQpResources(qpId);

    auto mrResult
        = sim::runtime::Db::GetByPred<sim::runtime::RaMR>(HcclSim::Storage::Eq(&sim::runtime::RaMR::vptr_id, qpId));
    if (!mrResult.ok() || !mrResult.value.has_value()) {
        return -1;
    }
    const auto& mrList = *mrResult.value;
    for (const auto& mr : mrList) {
        (void)sim::runtime::Db::Delete<sim::runtime::RaMR>(HcclSim::Storage::Eq(&sim::runtime::RaMR::id, mr.id));
        HCCL_VM_INFO("cleanup MR {:d} for QP {:d}", mr.id, qpId);
    }

    auto cqeResult = sim::runtime::Db::GetByPred<sim::runtime::RaCQE>(HcclSim::Storage::Or(
        HcclSim::Storage::Eq(&sim::runtime::RaCQE::cq_handle, qpOpt.value->send_cq_handle),
        HcclSim::Storage::Eq(&sim::runtime::RaCQE::cq_handle, qpOpt.value->recv_cq_handle)));
    if (!cqeResult.ok() || !cqeResult.value.has_value()) {
        return -1;
    }
    const auto& cqeList = *cqeResult.value;
    for (const auto& cqe : cqeList) {
        (void)sim::runtime::Db::Delete<sim::runtime::RaCQE>(HcclSim::Storage::Eq(&sim::runtime::RaCQE::id, cqe.id));
    }

    HCCL_VM_INFO("destroy QP {:d}, qp_num:{:d}", qpId, qpOpt.value->qp_num);
    (void)sim::runtime::Db::Delete<sim::runtime::RaQP>(HcclSim::Storage::Eq(&sim::runtime::RaQP::id, qpId));
    return 0;
}

int RaCtxQpCreate(void* ctxHandle, struct QpCreateAttr* attr, struct QpCreateInfo* info, void** qpHandle)
{
    if (ctxHandle == nullptr || attr == nullptr || info == nullptr || qpHandle == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, ctxHandle: {}, attr: {}, info: {}, qpHandle: {}", ctxHandle, (void*)attr, (void*)info,
            (void*)qpHandle);
        return -1;
    }

    uint64_t ctxId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));

    // 1. 先做不持有资源即可失败的校验（context/endpoint 查库），再申请编号与 WQE。
    auto devCtx = sim::runtime::Db::GetById<sim::runtime::RaContext>(ctxId);
    if (!devCtx.ok()) {
        HCCL_VM_ERROR("can not find Context:{:d}", ctxId);
        return -1;
    }

    auto endPoint = sim::runtime::Db::GetById<sim::runtime::EndPoint>(devCtx.value->endpoint_id);
    if (!endPoint.ok()) {
        HCCL_VM_ERROR("can not find endpoint:{:d}", devCtx.value->endpoint_id);
        return -1;
    }

    uint32_t rankId = 0;
    auto commRes = sim::runtime::Db::GetOneByPred<sim::runtime::Communicator>(
        HcclSim::Storage::Eq(&sim::runtime::Communicator::device_id, endPoint.value->device_id));
    if (commRes.ok()) {
        rankId = commRes.value->rank_id;
    }

    // 2. 资源申请。编号合同见 sim_capacity_limits.h：USER_CTL_NORMAL(AICPU) 按 rank 进程内
    // 分配真实硬件编号 [5312,9407]（CAS 有界、单调不复用、耗尽稳定失败）；其余模式沿用
    // caller 下发的编号（各模式编号空间按硬件合同不相交）。数据库行主键仅作句柄使用，
    // 不写入 SQE 16 位字段——行主键随表规模无界增长，256p 起经 16 位字段回绕成错误编号
    // （b4 实测行主键 65536~65792 回绕为 0~256，命中错误行导致数据面 SQE 丢弃）。
    // 编号消耗策略：WQE 分配或行插入失败时编号视为已消耗（不回收，防旧引用复活）。
    static std::atomic<uint32_t> s_aicpuJettyHwIdSeq{0};
    uint64_t sqBuffer = 0;
    uint8_t sqBufferMallocFlag = 0;
    uint32_t aicpuJettyHwId = 0;
    void* ownedWqeBuf = nullptr; // 仅跟踪本函数分配的 WQE；caller 传入的 buffVa 不归本函数管理
    bool aicpuMode = (attr->ub.mode == JettyMode::JETTY_MODE_USER_CTL_NORMAL) && sim::GetAicpuProcMgr().IsAlive();
    if (aicpuMode) {
        if (!HcclSim::TryReserveAicpuJettyHwId(s_aicpuJettyHwIdSeq, aicpuJettyHwId)) {
            HCCL_VM_ERROR(
                "aicpu jetty hw id exhausted, capacity[{}], owner pid[{}]",
                HcclSim::HCCL_VM_AICPU_JETTY_CAPACITY_PER_RANK, getpid());
            return -1;
        }
        sqBuffer = attr->ub.extMode.sq.buffVa;
        if (sqBuffer == 0) { // 判断是否需要由桩函数分配sqBuffer
            ownedWqeBuf = AllocateWqeBuff();
            if (ownedWqeBuf == nullptr) {
                HCCL_VM_ERROR("get wqe buff failed, jetty hw id[{}] consumed", aicpuJettyHwId);
                return -1;
            }
            sqBufferMallocFlag = 1;
            sqBuffer = reinterpret_cast<uint64_t>(ownedWqeBuf);
        }
    }

    // 3. 行组装与插入。插入失败时释放本函数分配的 WQE（caller 传入的绝不能在此释放）。
    sim::runtime::RaJetty jty{};
    jty.ctx_handle = ctxId;
    // aicpu 模式行记录进程内硬件编号；其余模式保留 caller 下发的原始编号。
    jty.jetty_id = aicpuMode ? aicpuJettyHwId : attr->ub.jettyId;
    jty.rank_id = rankId;
    jty.dieId = endPoint.value->die_id;
    jty.type = (uint8_t)attr->transportMode;
    jty.send_cq_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(attr->scqHandle));
    jty.recv_cq_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(attr->rcqHandle));
    jty.mode = attr->ub.mode;
    jty.sqBuffer = sqBuffer;
    jty.sqBufType = sqBufferMallocFlag;
    jty.pid = getpid();

    jty.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::RaJetty>(jty);
    if (!inserted.ok() || !inserted.value.has_value()) {
        if (ownedWqeBuf != nullptr) {
            FreeWqeBuff(ownedWqeBuf); // 桩分配的 WQE 随失败释放；编号已消耗（不复用）
        }
        return -1;
    }

    // 4. 插入成功后才向调用方发布句柄与编号；失败路径不留下可被当成成功的输出。
    auto id = inserted.value->value;
    *qpHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(id));

    if (info != nullptr) {
        // info->key.value / *qpHandle 为行主键（跨进程唯一的数据库句柄，供 QpImport/QpModify/
        // QpDestroy 使用）；info->ub.id 是 guest 写入 SQE 16 位字段的 jetty 硬件编号。
        *(uint64_t*)(info->key.value) = id;
        info->key.size = sizeof(uint64_t);
        info->ub.sqBuffVa = sqBuffer;
        info->ub.id = aicpuMode ? aicpuJettyHwId : attr->ub.jettyId;
    }

    HCCL_VM_INFO(
        "Ctx:{:d} create QP id:{:d}, scqHandle:{:d}, rcqHandle:{:d}, JettyId:{:d}, mode:{:d}, mem:{:d}", ctxId, id,
        jty.send_cq_handle, jty.recv_cq_handle, jty.jetty_id, jty.mode, sqBufferMallocFlag);
    return 0;
}

int RaCtxQpImport(void* ctxHandle, struct QpImportInfoT* qpInfo, void** remQpHandle)
{
    uint64_t remoteQpId = *(uint64_t*)(qpInfo->in.key.value);
    auto remoteQpOpt = sim::runtime::Db::GetById<sim::runtime::RaJetty>(remoteQpId);
    if (!remoteQpOpt.ok()) {
        HCCL_VM_ERROR("remote QP {:d} not found", remoteQpId);
        return -1;
    }

    auto rmtRaCtxOpt = sim::runtime::Db::GetById<sim::runtime::RaContext>(remoteQpOpt.value->ctx_handle);
    if (!rmtRaCtxOpt.ok()) {
        HCCL_VM_ERROR("remote RaContext {:d} not found", remoteQpOpt.value->ctx_handle);
        return -1;
    }

    uint64_t localRaCtxId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    auto localRaCtxOpt = sim::runtime::Db::GetById<sim::runtime::RaContext>(localRaCtxId);
    if (!localRaCtxOpt.ok()) {
        HCCL_VM_ERROR("local RaContext {:d} not found", localRaCtxId);
        return -1;
    }
    uint64_t localEndpointId = localRaCtxOpt.value->endpoint_id;
    uint64_t remoteEndpointId = rmtRaCtxOpt.value->endpoint_id;
    auto pairOpt = sim::runtime::Db::GetOneByPred<sim::runtime::EndPointPair>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::EndPointPair::local_enpoint_id, localEndpointId),
        HcclSim::Storage::Eq(&sim::runtime::EndPointPair::remote_enpoint_id, remoteEndpointId)));

    if (pairOpt.code == HcclSim::Storage::DbCode::NOT_FOUND) {
        sim::runtime::EndPointPair endpointPair{};
        endpointPair.local_enpoint_id = localEndpointId;
        endpointPair.remote_enpoint_id = remoteEndpointId;
        endpointPair.tp_type = qpInfo->in.ub.tpType;
        endpointPair.id = 0;
        auto inserted = sim::runtime::Db::Add<sim::runtime::EndPointPair>(endpointPair);
        if (!inserted.ok() || !inserted.value.has_value()) {
            return -1;
        }
    } else if (!pairOpt.ok()) {
        return -1;
    }

    // 回填归属: 被import的远端jetty属于本端endpoint到远端endpoint的链路；本端jetty的归属由对端import时回填
    (void)sim::runtime::Db::Update<sim::runtime::RaJetty>(
        HcclSim::Storage::Eq(&sim::runtime::RaJetty::id, remoteQpId),
        HcclSim::Storage::Set(&sim::runtime::RaJetty::peer_endpoint_id, localEndpointId));

#if 0
    sim::operation::JettyMapTab jettyMapInfo {};
    jettyMapInfo.id = 0;
    jettyMapInfo.opDetailId = 0;
    jettyMapInfo.srcDieId = localRaCtxOpt.value->eidIndex;
    jettyMapInfo.dstDieId = rmtRaCtxOpt.value->eidIndex;
    jettyMapInfo.srcRankId = localRaCtxOpt.value->device_id;
    jettyMapInfo.dstRankId = rmtRaCtxOpt.value->device_id;
    memcpy(jettyMapInfo.leid, &localEndpointId, sizeof(localEndpointId));
    memcpy(jettyMapInfo.reid, &remoteEndpointId, sizeof(remoteEndpointId));
    jettyMapInfo.protocol = 0;
    auto ret = sim::operation::InsertJettyMap(jettyMapInfo);
    if (ret != 0) {
        HCCL_VM_ERROR("insert jetty map failed");
        return -1;
    }
#endif

    *remQpHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(remoteQpId));
    HCCL_VM_INFO(
        "import remote QpId:{:d}, pair l:{:d}, r:{:d}, tp_type:{:d}", remoteQpId, localEndpointId, remoteEndpointId,
        qpInfo->in.ub.tpType);
    return 0;
}

int RaCtxQpImportAsync(void* ctxHandle, struct QpImportInfoT* info, void** remQpHandle, void** reqHandle)
{
    if (ctxHandle == nullptr || info == nullptr || remQpHandle == nullptr || reqHandle == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, ctxHandle: {}, info: {}, remQpHandle: {}, reqHandle: {}", ctxHandle, (void*)info,
            (void*)remQpHandle, (void*)reqHandle);
        return -1;
    }

    auto ret = RaCtxQpImport(ctxHandle, info, remQpHandle);
    *reqHandle = hccp_async_req::Register((ret == 0) ? 0 : static_cast<int>(ret), hccp_async_req::OP_QP_IMPORT);
    if (*reqHandle == nullptr) {
        return -1;
    }

    HCCL_VM_INFO(
        "ctx {:d} import QP {:d}", static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle)),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(*remQpHandle)));
    return ret;
}

int RaCtxQpBind(void* qpHandle, void* remQpHandle)
{
    if (qpHandle == nullptr || remQpHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, remQpHandle: {}", qpHandle, remQpHandle);
        return -1;
    }

    uint64_t localQpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    uint64_t remoteQpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(remQpHandle));

    auto localQpOpt = sim::runtime::Db::GetById<sim::runtime::RaJetty>(localQpId);
    if (!localQpOpt.ok()) {
        HCCL_VM_ERROR("local QP {:d} not found", localQpId);
        return -1;
    }

    auto remoteQpOpt = sim::runtime::Db::GetById<sim::runtime::RaJetty>(remoteQpId);
    if (!remoteQpOpt.ok()) {
        HCCL_VM_ERROR("remote QP {:d} not found", remoteQpId);
        return -1;
    }

    auto update = sim::runtime::Db::Update<sim::runtime::RaJetty>(
        HcclSim::Storage::Eq(&sim::runtime::RaJetty::id, localQpId),
        HcclSim::Storage::Set(&sim::runtime::RaJetty::peer_jetty_handle, remoteQpId),
        HcclSim::Storage::Set(&sim::runtime::RaJetty::state, 3));
    if (!update.ok()) {
        return -1;
    }

    HCCL_VM_INFO("bind local QP {:d} remote QP {:d}", localQpId, remoteQpId);
    return 0;
}

int RaCtxQpUnimport(void* ctxHandle, void* remQpHandle)
{
    uint64_t localRaDevId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    uint64_t remoteQpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(remQpHandle));
    auto remoteQpOpt = sim::runtime::Db::GetById<sim::runtime::RaJetty>(remoteQpId);
    if (!remoteQpOpt.ok()) {
        HCCL_VM_ERROR("remote QP {:d} not found", remoteQpId);
        return -1;
    }

    auto rmtRaCtxOpt = sim::runtime::Db::GetById<sim::runtime::RaContext>(remoteQpOpt.value->ctx_handle);
    if (!rmtRaCtxOpt.ok()) {
        HCCL_VM_ERROR("remote RaContext {:d} not found", remoteQpOpt.value->ctx_handle);
        return -1;
    }

    HCCL_VM_INFO("unimport rmtQp {:d}  pair l:{:d}, r:{:d}", remoteQpId, localRaDevId, rmtRaCtxOpt.value->id);
    return 0;
}

static int ResolveRemoteDeviceFromQp(uint64_t localQpId, uint32_t remoteQpn, uint8_t gid[16])
{
    auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(localQpId);
    if (!qpOpt.ok())
        return static_cast<int>(INVALID_DEVICE_ID);

    // 1. GID-based: 用远端IP直接匹配远端EndPoint → 远端Device (最可靠)
    //    gid[12..15] 携带远端 IPv4 地址，能唯一定位远端设备。
    //    注意: 不能依赖 RaDevice.endpoint_id (该字段从未被赋值),
    //    也不能用 device_id 查 EndPoint (一个设备有多个EndPoint, 会命中错误的一个)。
    if (gid != nullptr) {
        char ipStr[64]{};
        std::snprintf(ipStr, sizeof(ipStr), "%u.%u.%u.%u", gid[12], gid[13], gid[14], gid[15]);

        auto matchedEp = sim::runtime::Db::GetOneByPred<sim::runtime::EndPoint>(
            HcclSim::Storage::Eq(&sim::runtime::EndPoint::ip_addr, std::string(ipStr)));
        if (matchedEp.ok()) {
            return static_cast<int>(matchedEp.value->device_id);
        }
    }

    // 2. Fallback: 通过全局唯一 qp_num 跨进程查找远端 QP → RaDevice → Device
    auto remoteQpsResult = sim::runtime::Db::GetByPred<sim::runtime::RaQP>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::RaQP::qp_num, remoteQpn),
        HcclSim::Storage::Ne(&sim::runtime::RaQP::pid, static_cast<uint64_t>(getpid()))));
    if (remoteQpsResult.ok() && remoteQpsResult.value.has_value() && remoteQpsResult.value->size() == 1) {
        auto remoteDevOpt = sim::runtime::Db::GetById<sim::runtime::RaDevice>((*remoteQpsResult.value)[0].ra_dev_id);
        if (remoteDevOpt.ok()) {
            return static_cast<int>(remoteDevOpt.value->device_id);
        }
    }

    return static_cast<int>(INVALID_DEVICE_ID);
}

int RaTypicalQpModify(void* qpHandle, struct TypicalQp* localQpInfo, struct TypicalQp* remoteQpInfo)
{
    if (qpHandle == nullptr) {
        HCCL_VM_ERROR("qpHandle is null");
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(qpId);
    if (!qpOpt.ok()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (remoteQpInfo != nullptr) {
        auto update = sim::runtime::Db::Update<sim::runtime::RaQP>(
            HcclSim::Storage::Eq(&sim::runtime::RaQP::id, qpId), HcclSim::Storage::Set(&sim::runtime::RaQP::state, 3),
            HcclSim::Storage::Set(&sim::runtime::RaQP::peer_qpn, remoteQpInfo->qpn),
            HcclSim::Storage::Set(&sim::runtime::RaQP::perr_lid, remoteQpInfo->psn));
        if (!update.ok()) {
            return -1;
        }

        int remoteDeviceId = ResolveRemoteDeviceFromQp(qpId, remoteQpInfo->qpn, remoteQpInfo->gid);
        struct ibv_qp* fakeQp = LookupFakeQp(qpId);
        if (fakeQp != nullptr) {
            std::lock_guard<std::mutex> lock(g_dpuCtx.mtx);
            auto qpIt = g_dpuCtx.qpInfo.find(fakeQp);
            if (qpIt != g_dpuCtx.qpInfo.end()) {
                qpIt->second.remoteDeviceId = static_cast<uint32_t>(remoteDeviceId);
                qpIt->second.isDpu = true;
                std::memcpy(qpIt->second.rmEid, remoteQpInfo->gid, 16);
            }
        }

        HCCL_VM_INFO(
            "QP {:d} modify to RTS, peer_qpn:{:d}, psn:{:d}, remoteDevice:{:d}", qpId, remoteQpInfo->qpn,
            remoteQpInfo->psn, remoteDeviceId);
    } else {
        auto update = sim::runtime::Db::Update<sim::runtime::RaQP>(
            HcclSim::Storage::Eq(&sim::runtime::RaQP::id, qpId), HcclSim::Storage::Set(&sim::runtime::RaQP::state, 3));
        if (!update.ok()) {
            return -1;
        }
        HCCL_VM_INFO("QP {:d} modify to RTS", qpId);
    }

    return 0;
}

int RaTypicalSendWr(void* qpHandle, struct SendWr* wr, struct SendWrRsp* opRsp)
{
    if (qpHandle == nullptr || wr == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, wr: {}", qpHandle, (void*)wr);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(qpId);
    if (!qpOpt.ok()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (qpOpt.value->state != 3) {
        HCCL_VM_ERROR("QP {:d} not in RTS state", qpId);
        return -1;
    }

    HCCL_VM_INFO("QP {:d} typical send op:{:d}, bufNum:{:d}", qpId, wr->op, wr->bufNum);
    return 0;
}

int RaRdevGetPortStatus(void* rdmaHandle, enum PortStatus* status)
{
    if (rdmaHandle == nullptr || status == nullptr) {
        HCCL_VM_ERROR("invalid params, rdmaHandle: {}, status: {}", rdmaHandle, (void*)status);
        return -1;
    }

    *status = PORT_STATUS_ACTIVE;
    HCCL_VM_INFO("rdmaHandle:{:d} port status:UP", static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdmaHandle)));
    return 0;
}

int RaGetQpAttr(void* qpHandle, struct QpAttr* attr)
{
    if (qpHandle == nullptr || attr == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, attr: {}", qpHandle, (void*)attr);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = sim::runtime::Db::GetById<sim::runtime::RaQP>(qpId);
    if (!qpOpt.ok()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    attr->qpn = qpOpt.value->qp_num;
    attr->udpSport = 0;
    attr->psn = 0;
    attr->gidIdx = 0;
    std::memset(attr->gid, 0, sizeof(attr->gid));

    auto devOpt = sim::runtime::Db::GetById<sim::runtime::RaDevice>(qpOpt.value->ra_dev_id);
    if (devOpt.ok()) {
        auto epOpt = sim::runtime::Db::GetOneByPred<sim::runtime::EndPoint>(
            HcclSim::Storage::Eq(&sim::runtime::EndPoint::device_id, devOpt.value->device_id));
        if (epOpt.ok() && epOpt.value->ip_addr[0] != '\0') {
            unsigned int a = 0, b = 0, c = 0, d = 0;
            if (std::sscanf(epOpt.value->ip_addr, "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
                uint32_t ipBytes = (a << 24) | (b << 16) | (c << 8) | d;
                uint32_t netIp = htonl(ipBytes);
                std::memcpy(&attr->gid[12], &netIp, 4);
                attr->gid[10] = 0xFF;
                attr->gid[11] = 0xFF;
            }
        }
    }

    HCCL_VM_INFO(
        "QP {:d} qpn:{:d} gid[12..15]={:d}.{:d}.{:d}.{:d}", qpId, attr->qpn, attr->gid[12], attr->gid[13],
        attr->gid[14], attr->gid[15]);
    return 0;
}

int RaSocketWhiteListAdd(void* socketHandle, struct SocketWlistInfoT whiteList[], unsigned int num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaSocketWhiteListDel(void* socketHandle, struct SocketWlistInfoT whiteList[], unsigned int num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaSocketWhiteListBatchAdd(struct WhiteListInfo wlistInfoArray[], unsigned int num, struct ErrorInfo* errInfo)
{
    for (unsigned int i = 0; i < num; i++) {
        int ret
            = RaSocketWhiteListAdd(wlistInfoArray[i].sockWrap, wlistInfoArray[i].wlistDataArray, wlistInfoArray[i].num);
        if (ret != 0) {
            errInfo->errIndex = i;
            errInfo->errCode = ret;
            return -1;
        }
    }
    return 0;
}

int RaSocketWhiteListBatchDel(
    struct WhiteListInfo wlistInfoArray[], unsigned int num, struct ErrorInfo errInfoArray[], unsigned int* errNum)
{
    *errNum = 0;
    for (unsigned int i = 0; i < num; i++) {
        int ret
            = RaSocketWhiteListDel(wlistInfoArray[i].sockWrap, wlistInfoArray[i].wlistDataArray, wlistInfoArray[i].num);
        if (ret != 0) {
            errInfoArray[*errNum].errIndex = i;
            errInfoArray[*errNum].errCode = ret;
            (*errNum)++;
        }
    }
    return (*errNum > 0) ? -1 : 0;
}

int RaSocketAcceptCreditAdd(struct SocketListenInfoT conn[], unsigned int num, unsigned int creditLimit)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaGetIfnum(struct RaGetIfattr* config, unsigned int* num)
{
    sim::runtime::Device device{};
    if (sim::runtime::GetDeviceByPhysicalId(config->phyId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by logic id {} failed.", config->phyId);
        return -1;
    }

    uint64_t deviceIdx = device.id;
    auto endPointsResult = sim::runtime::Db::GetByPred<sim::runtime::EndPoint>(
        HcclSim::Storage::Eq(&sim::runtime::EndPoint::device_id, deviceIdx));
    if (!endPointsResult.ok() || !endPointsResult.value.has_value()) {
        return -1;
    }
    const auto& endPoints = *endPointsResult.value;

    *num = static_cast<unsigned int>(endPoints.size());
    HCCL_VM_WARN("Get num:{:d}", *num);
    return 0;
}

int RaGetIfaddrs(struct RaGetIfattr* config, struct InterfaceInfo interfaceInfos[], unsigned int* num)
{
    sim::runtime::Device device{};
    if (sim::runtime::GetDeviceByPhysicalId(config->phyId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by logic id {} failed.", config->phyId);
        return -1;
    }

    uint64_t deviceIdx = device.id;
    auto endPointsResult = sim::runtime::Db::GetByPred<sim::runtime::EndPoint>(
        HcclSim::Storage::Eq(&sim::runtime::EndPoint::device_id, deviceIdx));
    if (!endPointsResult.ok() || !endPointsResult.value.has_value()) {
        return -1;
    }
    const auto& endPoints = *endPointsResult.value;

    *num = static_cast<unsigned int>(endPoints.size());
    for (unsigned int i = 0; i < *num; i++) {
        interfaceInfos[i].family = AF_INET6;
        sprintf(interfaceInfos[i].ifname, "%s", endPoints.at(i).ip_addr);
        interfaceInfos[i].scopeId = 0;
        std::string ipaddr = std::string(endPoints.at(i).ip_addr);
        if (ipaddr.find(':') == std::string::npos) {
            ipaddr = "::" + ipaddr;
        }
        IpAddress addr(ipaddr, AF_INET6);
        interfaceInfos[i].ifaddr.ip.addr6 = addr.GetBinaryAddress().addr6;

        HCCL_VM_INFO("get Ip addr is ipaddr[{:d}]={}", i, endPoints.at(i).ip_addr);
    }
    return 0;
}

int RaTlvInit(struct TlvInitInfo* initInfo, unsigned int* bufferSize, void** tlvHandle)
{
    auto phyId = initInfo->phyId;
    sim::runtime::RaTlv raTlv{};
    raTlv.physical_id = phyId;
    raTlv.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::RaTlv>(raTlv);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return -1;
    }
    auto tlvId = inserted.value->value;
    *tlvHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(tlvId));
    return 0;
}

int RaTlvDeinit(void* tlvHandle)
{
    auto tlvId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tlvHandle));
    (void)sim::runtime::Db::Delete<sim::runtime::RaTlv>(HcclSim::Storage::Eq(&sim::runtime::RaTlv::id, tlvId));
    return 0;
}

struct ccu_mem_info {
    unsigned int long long mem_va;
    unsigned int mem_size;
    unsigned int resv[1];
};

struct ccu_mem_rsp {
    unsigned int die_id;
    unsigned int num;
    struct ccu_mem_info list[64U];
};

int GetCcuMemInfo(struct TlvMsg* sendMsg, struct TlvMsg* recvMsg)
{
    auto sendData = (CcuMemReq*)sendMsg->data;
    auto dieId = sendData->udieIdx;

    struct ccu_mem_rsp rsp {};
    rsp.die_id = dieId;
    rsp.num = 0;

    uint64_t ccu_die0_base_addr = 0x123123123;
    uint64_t ccu_die1_base_addr = 0x456456456;

    // ccu指令空间： 1MByte
    ccu_mem_info ccu_inst_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_inst_mem.mem_va = ccu_die0_base_addr;
    } else if (dieId == 1) {
        ccu_inst_mem.mem_va = ccu_die1_base_addr;
    }
    ccu_inst_mem.mem_size = 0x100000;
    rsp.list[rsp.num++] = ccu_inst_mem;

    uint64_t ccu_die0_gsa_offset = ccu_die0_base_addr + 0x100000;
    uint64_t ccu_die1_gsa_offset = ccu_die1_base_addr + 0x100000;
    // ccu gsa寄存器： size = 24KByte, 实际预留32KByte
    ccu_mem_info ccu_gsa_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_gsa_mem.mem_va = ccu_die0_gsa_offset;
    } else if (dieId == 1) {
        ccu_gsa_mem.mem_va = ccu_die1_gsa_offset;
    }
    ccu_gsa_mem.mem_size = 0x6000;
    rsp.list[rsp.num++] = ccu_gsa_mem;

    // ccu Xn寄存器： size = 24KByte, 实际预留32KByte
    uint64_t ccu_die0_xn_offset = ccu_die0_gsa_offset + 0x8000;
    uint64_t ccu_die1_xn_offset = ccu_die1_gsa_offset + 0x8000;
    ccu_mem_info ccu_xn_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_xn_mem.mem_va = ccu_die0_xn_offset;
    } else if (dieId == 1) {
        ccu_xn_mem.mem_va = ccu_die1_xn_offset;
    }
    ccu_xn_mem.mem_size = 0x6000;
    rsp.list[rsp.num++] = ccu_xn_mem;

    // ccu CKB寄存器： size = 8KByte, 实际预留32KByte
    uint64_t ccu_die0_ckb_offset = ccu_die0_xn_offset + 0x8000;
    uint64_t ccu_die1_ckb_offset = ccu_die1_xn_offset + 0x8000;
    ccu_mem_info ccu_ckb_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_ckb_mem.mem_va = ccu_die0_ckb_offset;
    } else if (dieId == 1) {
        ccu_ckb_mem.mem_va = ccu_die1_ckb_offset;
    }
    ccu_ckb_mem.mem_size = 0x2000;
    rsp.list[rsp.num++] = ccu_ckb_mem;

    // ccu PFE表： size = 256Byte, 实际预留32KByte
    uint64_t ccu_die0_pfe_offset = ccu_die0_ckb_offset + 0x8000;
    uint64_t ccu_die1_pfe_offset = ccu_die1_ckb_offset + 0x8000;
    ccu_mem_info ccu_pfe_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_pfe_mem.mem_va = ccu_die0_pfe_offset;
    } else if (dieId == 1) {
        ccu_pfe_mem.mem_va = ccu_die1_pfe_offset;
    }
    ccu_pfe_mem.mem_size = 0x100;
    rsp.list[rsp.num++] = ccu_pfe_mem;

    // ccu channel映射表： size = 8KByte, 实际预留32KByte
    uint64_t ccu_die0_channel_offset = ccu_die0_pfe_offset + 0x8000;
    uint64_t ccu_die1_channel_offset = ccu_die1_pfe_offset + 0x8000;
    ccu_mem_info ccu_channel_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_channel_mem.mem_va = ccu_die0_channel_offset;
    } else if (dieId == 1) {
        ccu_channel_mem.mem_va = ccu_die1_channel_offset;
    }
    ccu_channel_mem.mem_size = 0x2000;
    rsp.list[rsp.num++] = ccu_channel_mem;

    // ccu jetty context表： size = 4KByte, 实际预留32KByte
    uint64_t ccu_die0_ctx_offset = ccu_die0_channel_offset + 0x8000;
    uint64_t ccu_die1_ctx_offset = ccu_die1_channel_offset + 0x8000;
    ccu_mem_info ccu_ctx_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_ctx_mem.mem_va = ccu_die0_ctx_offset;
    } else if (dieId == 1) {
        ccu_ctx_mem.mem_va = ccu_die1_ctx_offset;
    }
    ccu_ctx_mem.mem_size = 0x1000;
    rsp.list[rsp.num++] = ccu_ctx_mem;

    // ccu mission context表： size = 1KByte, 实际预留32KByte
    uint64_t ccu_die0_mission_offset = ccu_die0_ctx_offset + 0x8000;
    uint64_t ccu_die1_mission_offset = ccu_die1_ctx_offset + 0x8000;
    ccu_mem_info ccu_mission_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_mission_mem.mem_va = ccu_die0_mission_offset;
    } else if (dieId == 1) {
        ccu_mission_mem.mem_va = ccu_die1_mission_offset;
    }
    ccu_mission_mem.mem_size = 0x1000;
    rsp.list[rsp.num++] = ccu_mission_mem;

    // ccu loop context表： size = 12.5KB, 实际预留32KByte
    uint64_t ccu_die0_loop_offset = ccu_die0_mission_offset + 0x8000;
    uint64_t ccu_die1_loop_offset = ccu_die1_mission_offset + 0x8000;
    ccu_mem_info ccu_loop_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_loop_mem.mem_va = ccu_die0_loop_offset;
    } else if (dieId == 1) {
        ccu_loop_mem.mem_va = ccu_die1_loop_offset;
    }
    ccu_loop_mem.mem_size = 0x3000;
    rsp.list[rsp.num++] = ccu_loop_mem;

    memcpy(recvMsg->data, &rsp, sizeof(rsp));
    return 0;
}

int RaTlvRequest(void* tlvHandle, unsigned int moduleType, struct TlvMsg* sendMsg, struct TlvMsg* recvMsg)
{
    HCCL_VM_INFO("Enter into tlv request...module: {}, msg:{}", moduleType, static_cast<int>(sendMsg->type));
    switch (sendMsg->type) {
        case TlvCcuMsgType::MSG_TYPE_CCU_INIT:
            HCCL_VM_WARN("not support CCU_INIT");
            return 0;
        case TlvCcuMsgType::MSG_TYPE_CCU_UNINIT:
            HCCL_VM_WARN("not support CCU_UNINIT");
            return 0;
        case TlvCcuMsgType::MSG_TYPE_CCU_GET_MEM_INFO:
            GetCcuMemInfo(sendMsg, recvMsg);
            break;
        case TlvCcuMsgType::MSG_TYPE_CCU_DISPATCH_CMD:
            SimRaCustomChannel(tlvHandle, sendMsg, recvMsg);
            break;
        default:
            break;
    }
    return 0;
}

int RaPingInit(struct PingInitAttr* initAttr, struct PingInitInfo* initInfo, void** pingHandle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaPingDeinit(void* pingHandle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaPingTargetAdd(void* pingHandle, struct PingTargetInfo target[], uint32_t num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaPingTaskStart(void* pingHandle, struct PingTaskAttr* attr)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaPingGetResults(void* pingHandle, struct PingTargetResult target[], uint32_t* num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaPingTargetDel(void* pingHandle, struct PingTargetCommInfo target[], uint32_t num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaPingTaskStop(void* pingHandle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaCtxTokenIdAlloc(void* ctxHandle, struct HccpTokenId* info, void** tokenIdHandle)
{
    if (ctxHandle == nullptr || info == nullptr || tokenIdHandle == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, ctxHandle: {}, info: {}, tokenIdHandle: {}", ctxHandle, (void*)info, (void*)tokenIdHandle);
        return -1;
    }
    sim::runtime::RaTokenId tokenId{};
    tokenId.ctx_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    tokenId.token_id = ((uint32_t)(tokenId.ctx_handle >> 32)) ^ ((uint32_t)rand());
    info->tokenId = tokenId.token_id;
    tokenId.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::RaTokenId>(tokenId);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return -1;
    }
    auto id = inserted.value->value;
    *tokenIdHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(id));
    return 0;
}

int RaGetSecRandom(struct RaInfo* info, uint32_t* value)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaCtxLmemUnregister(void* ctxHandle, void* lmemHandle)
{
    if (ctxHandle == nullptr || lmemHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, ctxHandle: {}, lmemHandle: {}", ctxHandle, lmemHandle);
        return -1;
    }
    uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(lmemHandle));
    auto removed = sim::runtime::Db::Delete<sim::runtime::RaLmem>(HcclSim::Storage::Eq(&sim::runtime::RaLmem::id, id));
    if (!removed.ok()) {
        // 删除失败/未命中都会泄漏 checker 依赖的注册行：显式暴露，不再静默吞掉
        HCCL_VM_ERROR("delete lmem id {:d} failed, ret: {}", id, static_cast<uint32_t>(removed.code));
    } else if (removed.affectedRows == 0U) {
        HCCL_VM_ERROR("delete lmem id {:d} matched no row (leak suspect)", id);
    } else {
        HCCL_VM_INFO("delete lmem id {:d}, for checker use after", id);
    }

    return 0;
}

int RaCtxRmemUnimport(void* ctxHandle, void* rmemHandle)
{
    if (ctxHandle == nullptr || rmemHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, ctxHandle: {}, rmemHandle: {}", ctxHandle, rmemHandle);
        return -1;
    }
    uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rmemHandle));
    auto removed = sim::runtime::Db::Delete<sim::runtime::RaRmem>(HcclSim::Storage::Eq(&sim::runtime::RaRmem::id, id));
    if (!removed.ok()) {
        HCCL_VM_ERROR("delete rmem id {:d} failed, ret: {}", id, static_cast<uint32_t>(removed.code));
    } else if (removed.affectedRows == 0U) {
        HCCL_VM_ERROR("delete rmem id {:d} matched no row (leak suspect)", id);
    } else {
        HCCL_VM_INFO("delete rmem id {:d}, for checker use after", id);
    }
    return 0;
}

int RaBatchRmaUnImport(void* rmaWrapArray[], unsigned int num, struct RmaConnErrorInfo errInfo[], unsigned int* errNum)
{
    *errNum = 0;
    for (unsigned int i = 0; i < num; i++) {
        uint64_t ctxId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rmaWrapArray[i]));

        auto jettys = sim::runtime::Db::GetByPred<sim::runtime::RaJetty>(
            HcclSim::Storage::Eq(&sim::runtime::RaJetty::ctx_handle, ctxId));
        if (jettys.ok() && jettys.value.has_value()) {
            for (auto& j : *jettys.value) {
                void* remQpHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(j.id));
                int ret = RaCtxQpUnimport(rmaWrapArray[i], remQpHandle);
                if (ret != 0) {
                    errInfo[*errNum].errInfo.errIndex = i;
                    errInfo[*errNum].errInfo.errCode = ret;
                    errInfo[*errNum].resType = 0;
                    (*errNum)++;
                }
            }
        }

        auto rmems = sim::runtime::Db::GetByPred<sim::runtime::RaRmem>(
            HcclSim::Storage::Eq(&sim::runtime::RaRmem::ctx_handle, ctxId));
        if (rmems.ok() && rmems.value.has_value()) {
            for (auto& m : *rmems.value) {
                void* rmemHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(m.id));
                int ret = RaCtxRmemUnimport(rmaWrapArray[i], rmemHandle);
                if (ret != 0) {
                    errInfo[*errNum].errInfo.errIndex = i;
                    errInfo[*errNum].errInfo.errCode = ret;
                    errInfo[*errNum].resType = 1;
                    (*errNum)++;
                }
            }
        }
    }
    return (*errNum > 0) ? -1 : 0;
}

int RaCtxQpUnimportAsync(void* remQpHandle, void** reqHandle)
{
    (void)remQpHandle;
    *reqHandle = hccp_async_req::Register(0, hccp_async_req::OP_QP_UNIMPORT);
    if (*reqHandle == nullptr) {
        return -1;
    }
    return 0;
}

int RaCtxLmemUnregisterAsync(void* ctxHandle, void* lmemHandle, void** reqHandle)
{
    // 修复：旧实现只回请求句柄、不删除 RaLmem 行——异步注销路径的注册行永久泄漏
    // （checker 依赖的注册记录被无限累积）。与 RaCtxQpDestroyAsync 同一模式：
    // 模拟器内同步完成删除，再返回请求句柄。
    const int ret = RaCtxLmemUnregister(ctxHandle, lmemHandle);
    *reqHandle = hccp_async_req::Register((ret == 0) ? 0 : ret, hccp_async_req::OP_LMEM_UNREGISTER);
    if (*reqHandle == nullptr) {
        return -1;
    }
    return ret;
}

int RaCtxQpDestroyAsync(void* qpHandle, void** reqHandle)
{
    if (qpHandle == nullptr || reqHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, reqHandle: {}", qpHandle, (void*)reqHandle);
        return -1;
    }

    int ret = RaCtxQpDestroy(qpHandle);
    *reqHandle = hccp_async_req::Register((ret == 0) ? 0 : ret, hccp_async_req::OP_QP_DESTROY);
    if (*reqHandle == nullptr) {
        return -1;
    }

    HCCL_VM_INFO(
        "destroy QP {:d}, reqHandle:{:x}", static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle)),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(*reqHandle)));
    return ret;
}

int RaCtxQpDestroyBatchAsync(void* ctxHandle, void* qpHandle[], unsigned int* num, void** reqHandle)
{
    HCCL_VM_INFO("enter num {:d}", *num);
    int ret = 0;
    for (uint32_t i = 0; i < *num; i++) {
        ret |= RaCtxQpDestroyAsync(qpHandle[i], reqHandle);
    }
    return ret;
}

int RaCtxRmemImport(void* ctxHandle, struct MrImportInfoT* rmemInfo, void** rmemHandle)
{
    uint64_t remoteMemId = *(uint64_t*)(rmemInfo->in.key.value);
    auto remoteMemOpt = sim::runtime::Db::GetById<sim::runtime::RaLmem>(remoteMemId);
    if (!remoteMemOpt.ok()) {
        HCCL_VM_ERROR("remote Mem {:d} not found", remoteMemId);
        return -1;
    }

    // remote_eid 直接取注册时保存在 RaLmem 的 endpoint_id 快照（RaCtxLmemRegister
    // 已校验远端 RaContext 存在并保存），不再按远端 ctx_handle 二次查询。
    sim::runtime::RaRmem mem{};
    mem.ctx_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    mem.remote_key = remoteMemOpt.value->mem_key;
    mem.target_seg_handle = remoteMemId;
    mem.remote_eid = remoteMemOpt.value->endpoint_id;
    mem.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::RaRmem>(mem);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return -1;
    }
    // 修复句柄错位：旧实现把远端 RaLmem 的 id 当作句柄返回，Unimport 拿它按
    // RaRmem 主键删除——永远删不中本行（注册行泄漏），且 RaLmem/RaRmem 两表
    // 自增序列独立、id 撞号时会误删无关 RaRmem 行（checker 判据数据被破坏）。
    // 句柄改为本 RaRmem 行自己的 id（与 RaCtxLmemRegister 的既有合同一致）；
    // 远端 id 已在 target_seg_handle 字段保留。
    const uint64_t rmemId = inserted.value->value;
    rmemInfo->out.ub.targetSegHandle = rmemId;
    *rmemHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(rmemId));
    return 0;
}

int RaBatchRmaImport(struct RmaImportInfo rmaConnInfoArray[], unsigned int num, struct RmaConnErrorInfo* errInfo)
{
    for (unsigned int i = 0; i < num; i++) {
        for (unsigned int j = 0; j < rmaConnInfoArray[i].memImpNum; j++) {
            void* rmemHandle = nullptr;
            int ret
                = RaCtxRmemImport(rmaConnInfoArray[i].rmaWrap, &rmaConnInfoArray[i].memImpInfoArray[j], &rmemHandle);
            if (ret != 0) {
                errInfo->errInfo.errIndex = i;
                errInfo->errInfo.errCode = ret;
                errInfo->resType = 1;
                return -1;
            }
        }

        for (unsigned int k = 0; k < rmaConnInfoArray[i].jettyImpNum; k++) {
            void* remQpHandle = nullptr;
            int ret = RaCtxQpImport(
                rmaConnInfoArray[i].rmaWrap, &rmaConnInfoArray[i].jettyImportInfoArray[k], &remQpHandle);
            if (ret != 0) {
                errInfo->errInfo.errIndex = i;
                errInfo->errInfo.errCode = ret;
                errInfo->resType = 0;
                return -1;
            }
        }
    }
    return 0;
}

int RaCtxChanCreate(void* ctxHandle, struct ChanInfoT* chanInfo, void** chanHandle)
{
    uint64_t raCtxId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    sim::runtime::RaChan ch{};
    ch.ctx_handle = raCtxId;
    ch.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::RaChan>(ch);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return -1;
    }
    auto id = inserted.value->value;
    *(chanHandle) = reinterpret_cast<void*>(static_cast<uintptr_t>(id));

    HCCL_VM_WARN("RaChan {:d} add RaChan id:{:d}", raCtxId, id);
    return 0;
}

int RaCtxChanDestroy(void* ctxHandle, void* chanHandle)
{
    uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    HCCL_VM_INFO("soft delete chan id {:d}, for checker use after", id);
    return 0;
}

int RaCtxQpDestroy(void* qpHandle)
{
    if (qpHandle == nullptr) {
        HCCL_VM_ERROR("qpHandle is null");
        return -1;
    }

    uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto jty = sim::runtime::Db::GetById<sim::runtime::RaJetty>(id);
    if (!jty.ok()) {
        HCCL_VM_ERROR("Ra Jetty {:d} not found", id);
        return -1;
    }

    if (jty.value->sqBufType == 1) {
        void* sqBuf = reinterpret_cast<void*>(static_cast<uintptr_t>(jty.value->sqBuffer));
        // WQE 缓冲区由 HCCP 模拟层分配, 使用对应释放接口回收, 避免走 ACL 内存释放路径.
        FreeWqeBuff(sqBuf);
    }

    HCCL_VM_INFO("soft delete jetty id {:d}, for checker use after", id);
    return 0;
}

int RaCtxTokenIdFree(void* ctxHandle, void* tokenIdHandle)
{
    if (ctxHandle == nullptr || tokenIdHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, ctxHandle: {}, tokenIdHandle: {}", ctxHandle, tokenIdHandle);
        return -1;
    }
    uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tokenIdHandle));
    HCCL_VM_INFO("soft delete token id {:d}, for checker use after", id);
    return 0;
}

int RaCtxLmemRegister(void* ctxHandle, struct MrRegInfoT* lmemInfo, void** lmemHandle)
{
    if (ctxHandle == nullptr || lmemInfo == nullptr || lmemHandle == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, ctxHandle: {}, lmemInfo: {}, lmemHandle: {}", ctxHandle, (void*)lmemInfo,
            (void*)lmemHandle);
        return -1;
    }
    const uint64_t ctxId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    sim::runtime::RaLmem memInfo{};
    memInfo.ctx_handle = ctxId;
    memInfo.addr = lmemInfo->in.mem.addr;
    memInfo.size = lmemInfo->in.mem.size;
    uint64_t token = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(lmemInfo->in.ub.tokenIdHandle));
    auto tokenInfo = sim::runtime::Db::GetById<sim::runtime::RaTokenId>(token);
    if (!tokenInfo.ok()) {
        HCCL_VM_ERROR("tokenHandle {:d} not found", token);
        return -1;
    }
    // 所属 RaContext 必须存在（缺失即明确失败，不写不完整行）；endpoint_id
    // 快照保存在本行，Import 免按远端 ctx_handle 二次查询 RaContext。
    auto raCtxInfo = sim::runtime::Db::GetById<sim::runtime::RaContext>(ctxId);
    if (!raCtxInfo.ok()) {
        HCCL_VM_ERROR("RaContext {:d} not found", ctxId);
        return -1;
    }
    memInfo.endpoint_id = raCtxInfo.value->endpoint_id;
    memInfo.token_id = tokenInfo.value->token_id;
    memInfo.mem_key = (token << 32) | (rand() & 0xFFFFFFFF);
    memInfo.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::RaLmem>(memInfo);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return -1;
    }
    auto id = inserted.value->value;
    *reinterpret_cast<uint64_t*>(lmemInfo->out.key.value) = id;
    lmemInfo->out.key.size = sizeof(uint64_t);
    lmemInfo->out.ub.tokenId = memInfo.token_id;
    lmemInfo->out.ub.targetSegHandle = id;

    *lmemHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(id));
    return 0;
}

int RaCtxLmemRegisterAsync(void* ctxHandle, struct MrRegInfoT* lmemInfo, void** lmemHandle, void** reqHandle)
{
    HCCL_VM_ERROR("not support");
    return -1;
}

int RaGetTpInfoListAsync(
    void* ctxHandle, struct GetTpCfg* cfg, struct HccpTpInfo infoList[], unsigned int* num, void** reqHandle)
{
    (void)ctxHandle;
    (void)cfg;
    *reqHandle = hccp_async_req::Register(0, hccp_async_req::OP_TP_INFO_LIST);
    if (*reqHandle == nullptr) {
        return -1;
    }
    *num = 1;
    infoList[0].tpHandle = 0x12345678;
    return 0;
}

int RaGetEidByIpAsync(void* ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int* num, void** reqHandle)
{
    HCCL_VM_ERROR("not support");
    return -1;
}

int RaGetTpAttrAsync(void* ctxHandle, uint64_t tpHandle, uint32_t* attrBitmap, struct TpAttr* attr, void** reqHandle)
{
    (void)ctxHandle;
    (void)tpHandle;
    (void)attrBitmap;
    *reqHandle = hccp_async_req::Register(0, hccp_async_req::OP_TP_ATTR_GET);
    if (*reqHandle == nullptr) {
        return -1;
    }
    attr->slBitmap = 0xe;
    return 0;
}

int RaCtxQpQueryBatch(void* qpHandle[], struct JettyAttr attr[], unsigned int* num)
{
    HCCL_VM_ERROR("not support");
    return -1;
}

int RaCtxQpUnbind(void* qpHandle)
{
    if (qpHandle == nullptr) {
        HCCL_VM_ERROR("qpHandle is null");
        return -1;
    }

    uint64_t jtyId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto jtyOpt = sim::runtime::Db::GetById<sim::runtime::RaJetty>(jtyId);
    if (!jtyOpt.ok()) {
        HCCL_VM_WARN("QP {:d} not found", jtyId);
        return 0;
    }

    auto update = sim::runtime::Db::Update<sim::runtime::RaJetty>(
        HcclSim::Storage::Eq(&sim::runtime::RaJetty::id, jtyId),
        HcclSim::Storage::Set(&sim::runtime::RaJetty::peer_jetty_handle, 0),
        HcclSim::Storage::Set(&sim::runtime::RaJetty::state, 0));
    if (!update.ok()) {
        return -1;
    }

    HCCL_VM_INFO("unbind QP {:d}, reset to RESET state", jtyId);
    return 0;
}

int RaBatchSendWr(
    void* qpHandle, struct SendWrData wrList[], struct SendWrResp opResp[], unsigned int num, unsigned int* completeNum)
{
    HCCL_VM_ERROR("not support");
    return -1;
}

int RaCtxCqCreate(void* ctxHandle, struct CqInfoT* info, void** cqHandle)
{
    *cqHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(0xabcdU));
    return 0;
}

int RaCtxCqDestroy(void* ctxHandle, void* cqHandle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaCtxUpdateCi(void* qpHandle, uint16_t ci)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int ra_get_async_req_result(void* req_handle, int* req_result)
{
    if (req_handle == nullptr || req_result == nullptr) {
        HCCL_VM_ERROR("invalid params, req_handle: {}, req_result: {}", req_handle, (void*)req_result);
        return -1;
    }

    *req_result = 1;
    HCCL_VM_INFO("req_handle:{:x} result:completed", static_cast<uint64_t>(reinterpret_cast<uintptr_t>(req_handle)));
    return 0;
}

int ra_get_qp_context(void* qpHandle, void** qp, void** sendCq, void** recvCq)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int ra_get_tsqp_depth(void* rdev_handle, unsigned int* temp_depth, unsigned int* qp_num)
{
    *temp_depth = 1;
    *qp_num = 1;
    return 0;
}

int ra_set_tsqp_depth(void* rdev_handle, unsigned int temp_depth, unsigned int* qp_num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int ra_get_notify_mr_info(void* handle, struct mr_info* mrInfo)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int ra_send_wrlist_ext(
    void* qp_handle, struct send_wrlist_data_ext wr[], struct send_wr_rsp op_rsp[], unsigned int send_num,
    unsigned int* complete_num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int ra_register_mr(const void* handle, struct mr_info* mrInfo, void** mrHandle)
{
    *mrHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(0xabcdU));
    return ((handle == nullptr) || (mrInfo == nullptr)) ? -1 : 0;
}

int ra_deregister_mr(const void* handle, void* mrHandle)
{
    return ((handle == nullptr) || (mrHandle == nullptr)) ? -1 : 0;
}

int ra_is_first_used(int ins_id)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int ra_epoll_ctl_add(const void* fd_handle, RaEpollEvent event)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int ra_epoll_ctl_mod(const void* fd_handle, RaEpollEvent event)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int ra_epoll_ctl_del(const void* fd_handle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaCtxQpCreateAsync(
    void* ctxHandle, struct QpCreateAttr* attr, struct QpCreateInfo* info, void** qpHandle, void** reqHandle)
{
    if (ctxHandle == nullptr || attr == nullptr || qpHandle == nullptr || reqHandle == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, ctxHandle: {}, attr: {}, qpHandle: {}, reqHandle: {}", ctxHandle, (void*)attr,
            (void*)qpHandle, (void*)reqHandle);
        return -1;
    }

    int ret = RaCtxQpCreate(ctxHandle, attr, info, qpHandle);
    if (ret != 0) {
        HCCL_VM_ERROR("RaCtxQpCreate failed");
        return ret;
    }

    *reqHandle = hccp_async_req::Register(0, hccp_async_req::OP_QP_CREATE);
    if (*reqHandle == nullptr) {
        return -1;
    }
    HCCL_VM_INFO(
        "ctx {:d} async create QP {:d}, reqHandle:{:x}, jettyId:{:d}, jettyyMode:{:d}",
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle)),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(*qpHandle)),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(*reqHandle)), attr->ub.jettyId,
        static_cast<uint32_t>(attr->ub.mode));

    return 0;
}

int RaSetTpAttrAsync(void* ctxHandle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr* attr, void** reqHandle)
{
    (void)ctxHandle;
    (void)tpHandle;
    (void)attrBitmap;
    (void)attr;
    *reqHandle = hccp_async_req::Register(0, hccp_async_req::OP_TP_ATTR_SET);
    if (*reqHandle == nullptr) {
        return -1;
    }
    return 0;
}

int RaCtxGetAuxInfo(void* ctxHandle, struct HccpAuxInfoIn* in, struct HccpAuxInfoOut* out)
{
    HCCL_VM_ERROR("Not support yet");
    return -1;
}

int RaCtxGetCrErrInfoList(void* ctxHandle, struct CrErrInfo* infoList, unsigned int* num)
{
    HCCL_VM_ERROR("Not support yet");
    return -1;
}

TraStatus AtraceSubmit(TraHandle handle, const void* buffer, uint32_t bufSize)
{
    if (handle == 0) {
        return 0;
    } else if (handle == 1) {
        HCCL_VM_ERROR("invalid params, handle: {}", handle);
        return -1;
    }
    return 0;
}

rtError_t rtPointerGetAttributes(rtPointerAttributes_t* attributes, const void* ptr)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

rtError_t rtStreamGetCqid(const rtStream_t stm, uint32_t* cqId, uint32_t* logicCqId)
{
    static uint32_t i = 0U;
    *logicCqId = i++;
    return 0;
}

int RaGetEidByIp(void* ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int* num)
{
    if (ctxHandle == nullptr || ip == nullptr || eid == nullptr || num == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, ctxHandle: {:p}, ip: {:p}, eid: {:p}, num: {}", ctxHandle, (void*)ip, (void*)eid,
            (void*)num);
        return -1;
    }

    if (*num == 0 || *num > HCCP_EID_IP_QUERY_MAX_NUM) {
        HCCL_VM_ERROR("num({}) must be in range [1, {}]", *num, HCCP_EID_IP_QUERY_MAX_NUM);
        return -1;
    }

    for (uint32_t i = 0; i < *num; i++) {
        BinaryAddr ipAddr;
        ipAddr.addr = ip[i].ip.addr;
        ipAddr.addr6 = ip[i].ip.addr6;

        IpAddress addr(ipAddr, ip[i].family);
        auto simEid = addr.GetEid();
        for (uint32_t j = 0; j < URMA_EID_LEN; j++) {
            eid[i].raw[j] = simEid.raw[j];
        }
    }

    return 0;
}

int RaGetIpByEid(void* ctxHandle, union HccpEid eid[], struct IpInfo ip[], unsigned int* num)
{
    if (ctxHandle == nullptr || ip == nullptr || eid == nullptr || num == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, ctxHandle: {:p}, ip: {:p}, eid: {:p}, num: {}", ctxHandle, (void*)ip, (void*)eid,
            (void*)num);
        return -1;
    }

    if (*num == 0 || *num > HCCP_EID_IP_QUERY_MAX_NUM) {
        HCCL_VM_ERROR("num({}) must be in range [1, {}]", *num, HCCP_EID_IP_QUERY_MAX_NUM);
        return -1;
    }

    for (uint32_t i = 0; i < *num; i++) {
        Eid simEid;
        for (uint32_t j = 0; j < URMA_EID_LEN; j++) {
            simEid.raw[j] = eid[i].raw[j];
        }
        auto ipAddr = IpAddress(simEid);
        auto ipInfo = ipAddr.GetBinaryAddress();
        ip[i].ip.addr = ipInfo.addr;
        ip[i].ip.addr6 = ipInfo.addr6;
        ip[i].family = ipAddr.GetFamily();
    }

    return 0;
}

int RaGetIpByEidAsync(void* ctxHandle, union HccpEid eid[], struct IpInfo ip[], unsigned int* num, void** reqHandle)
{
    HCCL_VM_WARN("Enter RaGetIpByEidAsync");
    if (reqHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, reqHandle: {:p}", (void*)reqHandle);
        return -1;
    }
    return RaGetIpByEid(ctxHandle, eid, ip, num);
}

int RaGetLbMax(void* rdevHandle, int* lbMax)
{
    (void)rdevHandle;
    if (lbMax != nullptr) {
        *lbMax = 0;
    }
    HCCL_VM_INFO("RaGetLbMax stub: return lbMax=0");
    return 0;
}

int RaBatchRmaResPrepare(struct RmaResInfo rmaResInfoArray[], unsigned int num, struct RmaResErrorInfo* errInfo)
{
    if (rmaResInfoArray == nullptr || errInfo == nullptr) {
        HCCL_VM_ERROR("invalid params, rmaResInfoArray: {:p}, errInfo: {:p}", (void*)rmaResInfoArray, (void*)errInfo);
        return -1;
    }

    for (unsigned int i = 0; i < num; i++) {
        void* rmaWrap = rmaResInfoArray[i].rmaWrap;
        // 1. JFC资源创建
        struct JettyRelatedResInfo& jettyRelated = rmaResInfoArray[i].jettyRelatedResInfo;
        for (unsigned int j = 0; j < jettyRelated.jfcNum; j++) {
            void* cqHandle = nullptr;
            int ret = RaCtxCqCreate(rmaWrap, &jettyRelated.jfcResInfoArray[j], &cqHandle);
            if (ret != 0) {
                errInfo->resType = 1;
                errInfo->errInfo.errIndex = i;
                errInfo->errInfo.errCode = ret;
                return -1;
            }
        }
        // 2. Jetty资源创建
        for (unsigned int k = 0; k < jettyRelated.jettyNum; k++) {
            void* qpHandle = nullptr;
            int ret = RaCtxQpCreate(
                rmaWrap, &jettyRelated.jettyResInfoArray[k].in, &jettyRelated.jettyResInfoArray[k].out, &qpHandle);
            if (ret != 0) {
                errInfo->resType = 1;
                errInfo->errInfo.errIndex = i;
                errInfo->errInfo.errCode = ret;
                return -1;
            }
        }

        // 3. TP资源处理：GetTpInfoList → GetTpAttr → SetTpAttr
        struct TpResInfo& tpRes = rmaResInfoArray[i].tpResInfo;
        unsigned int tpNum = 1;
        struct HccpTpInfo tpInfoList[1];
        void* reqHandle = nullptr;
        int ret = RaGetTpInfoListAsync(rmaWrap, &tpRes.cfg, tpInfoList, &tpNum, &reqHandle);
        if (ret != 0) {
            errInfo->resType = 2;
            errInfo->errInfo.errIndex = i;
            errInfo->errInfo.errCode = ret;
            return -1;
        }

        if (tpNum > 0) {
            tpRes.tpInfo = tpInfoList[0];
            // GetTpAttr
            if (tpRes.getAttrBitmap != 0) {
                uint32_t bitmap = tpRes.getAttrBitmap;
                ret = RaGetTpAttrAsync(rmaWrap, tpRes.tpInfo.tpHandle, &bitmap, &tpRes.attr, &reqHandle);
                if (ret != 0) {
                    errInfo->resType = 2;
                    errInfo->errInfo.errIndex = i;
                    errInfo->errInfo.errCode = ret;
                    return -1;
                }
            }

            // SetTpAttr
            if (tpRes.setAttrBitmap != 0) {
                ret = RaSetTpAttrAsync(rmaWrap, tpRes.tpInfo.tpHandle, tpRes.setAttrBitmap, &tpRes.attr, &reqHandle);
                if (ret != 0) {
                    errInfo->resType = 2;
                    errInfo->errInfo.errIndex = i;
                    errInfo->errInfo.errCode = ret;
                    return -1;
                }
            }
        }

        // 4. Segment/MR资源注册
        struct SegmentResInfo& segRes = rmaResInfoArray[i].segResInfo;
        for (unsigned int m = 0; m < segRes.num; m++) {
            void* lmemHandle = nullptr;
            ret = RaCtxLmemRegister(rmaWrap, &segRes.regInfoArray[m], &lmemHandle);
            if (ret != 0) {
                errInfo->resType = 3;
                errInfo->errInfo.errIndex = i;
                errInfo->errInfo.errCode = ret;
                return -1;
            }
        }
    }
    return 0;
}

int RaBatchRmaResRelease(
    void* rmaWrapArray[], unsigned int num, struct RmaResErrorInfo errInfoArray[], unsigned int* errNum)
{
    if (rmaWrapArray == nullptr || errInfoArray == nullptr || errNum == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, rmaWrapArray: {:p}, errInfoArray: {:p}, errNum: {:p}", (void*)rmaWrapArray,
            (void*)errInfoArray, (void*)errNum);
        return -1;
    }
    *errNum = 0;
    for (unsigned int i = 0; i < num; i++) {
        uint64_t ctxId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rmaWrapArray[i]));

        // 1. Release JFC/CQ resources
        auto jfcs = sim::runtime::Db::GetByPred<sim::runtime::RaJfc>(
            HcclSim::Storage::Eq(&sim::runtime::RaJfc::ctx_handle, ctxId));
        if (jfcs.ok() && jfcs.value.has_value()) {
            for (auto& j : *jfcs.value) {
                void* cqHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(j.id));
                int ret = RaCtxCqDestroy(rmaWrapArray[i], cqHandle);
                if (ret != 0) {
                    errInfoArray[*errNum].errInfo.errIndex = i;
                    errInfoArray[*errNum].errInfo.errCode = ret;
                    errInfoArray[*errNum].resType = 1; // JFC/CQ related
                    (*errNum)++;
                }
            }
        }

        // 2. Release Jetty resources
        auto jettys = sim::runtime::Db::GetByPred<sim::runtime::RaJetty>(
            HcclSim::Storage::Eq(&sim::runtime::RaJetty::ctx_handle, ctxId));
        if (jettys.ok() && jettys.value.has_value()) {
            for (auto& j : *jettys.value) {
                void* qpHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(j.id));
                int ret = RaCtxQpDestroy(qpHandle);
                if (ret != 0) {
                    errInfoArray[*errNum].errInfo.errIndex = i;
                    errInfoArray[*errNum].errInfo.errCode = ret;
                    errInfoArray[*errNum].resType = 1; // Jetty related
                    (*errNum)++;
                }
            }
        }

        auto lmems = sim::runtime::Db::GetByPred<sim::runtime::RaLmem>(
            HcclSim::Storage::Eq(&sim::runtime::RaLmem::ctx_handle, ctxId));
        if (lmems.ok() && lmems.value.has_value()) {
            for (auto& m : *lmems.value) {
                void* lmemHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(m.id));
                int ret = RaCtxLmemUnregister(rmaWrapArray[i], lmemHandle);
                if (ret != 0) {
                    errInfoArray[*errNum].errInfo.errIndex = i;
                    errInfoArray[*errNum].errInfo.errCode = ret;
                    errInfoArray[*errNum].resType = 3; // Segment related
                    (*errNum)++;
                }
            }
        }
    }
    return (*errNum > 0) ? -1 : 0;
}

#ifdef __cplusplus
}
#endif // __cplusplus
