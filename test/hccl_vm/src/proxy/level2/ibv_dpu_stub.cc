/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define HCCL_VM_MODULE "IBV_DPU_STUB"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include "sim_log.h"
#include "store_sim_store_pub.h"
#include "sim_common_defs.h"
#include "storage/table_access.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "ibv_dpu_stub.h"
#include "hccl_proxy_common.h"

static uint64_t GetElapsedUs()
{
    static std::atomic<uint64_t> s_firstUs{0};
    auto nowUs
        = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
              .count();
    uint64_t firstUs = s_firstUs.load(std::memory_order_relaxed);
    if (firstUs == 0) {
        firstUs = nowUs;
        s_firstUs.store(firstUs, std::memory_order_relaxed);
    }
    return nowUs - firstUs;
}

static uint16_t GetTidShort()
{
    return static_cast<uint16_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFF);
}

IbvDpuContext g_dpuCtx;

static constexpr uint8_t DPU_PROTOCOL_ROCE = 2;

static uint64_t GetDpuStreamIdByDeviceId(uint32_t deviceId)
{
    auto dpuRet = sim::runtime::Db::GetOneByPred<sim::runtime::DpuDeviceInfo>(
        HcclSim::Storage::Eq(&sim::runtime::DpuDeviceInfo::device_id, deviceId));
    if (dpuRet.ok()) {
        if (dpuRet->stream_id != 0) {
            return dpuRet->stream_id;
        }
    }

    HCCL_VM_ERROR("cannot find streamId on dpu, deviceId={}", deviceId);
    return 0;
}

// HCCL算子入口通过 g_cur_comm_key 设置当前通信域; DPU stub 生成的任务在此上下文内
// 同步执行, 故可复用该 thread_local 值填充 commId, 并据此解析 rankId, 使 checker
// 能识别任务的通信域与身份 (见 task_meta_translator_v3.cc MakeTaskPosition).
// 但 ibv_post_send/ibv_poll_cq 可能在 HCCL 的 HostCpuRoceChannel worker 线程上被
// 调用, 该线程未继承算子入口设置的 g_cur_comm_key (仍为 0). 此时回退到按 deviceId
// 从 Communicator 表查找该设备所属的通信域成员行, 获取其 id (即 commId) 和 rankId.
static void FillTaskCommInfo(HcclTaskMetaData& task, uint32_t deviceId)
{
    task.commId = sim::runtime::g_cur_comm_key;
    if (task.commId != 0) {
        if (!sim::runtime::GetCommRankByDeviceId(task.commId, deviceId, task.rankId)) {
            HCCL_VM_ERROR("dpu task cannot resolve rankId, commId={}, deviceId={}", task.commId, deviceId);
        }
        return;
    }

    auto members = sim::runtime::Db::GetByPred<sim::runtime::Communicator>(
        HcclSim::Storage::Eq(&sim::runtime::Communicator::device_id, deviceId));
    if (!members.ok() || !members.value.has_value() || members->empty()) {
        HCCL_VM_ERROR("dpu task has no communicator, deviceId={}", deviceId);
        return;
    }
    task.commId = members->front().id;
    task.rankId = members->front().rank_id;
}

static std::atomic<uint32_t> s_notifySeq{1};

static inline uint32_t MakeUniqueNotifyId(uint32_t deviceId)
{
    return (1u << 31) | (((deviceId + 1) & 0xFFFFu) << 16)
           | (s_notifySeq.fetch_add(1, std::memory_order_relaxed) & 0xFFFFu);
}

void RegisterDpuQp(
    ibv_qp* fakeQp, uint64_t dbQpId, uint32_t qpNum, uint32_t localDeviceId, ibv_cq* sendCq, ibv_cq* recvCq)
{
    std::lock_guard<std::mutex> lock(g_dpuCtx.mtx);
    g_dpuCtx.qpInfo[fakeQp] = {dbQpId, qpNum, localDeviceId, INVALID_DEVICE_ID, {}, sendCq, recvCq, false};
    g_dpuCtx.sendCqToQp[sendCq] = fakeQp;
    g_dpuCtx.recvCqToQp[recvCq] = fakeQp;
    g_dpuCtx.pendingWqeCount[sendCq] = 0;
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int IbvPostSendStub(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr);
int IbvPostRecvStub(struct ibv_qp* qp, struct ibv_recv_wr* wr, struct ibv_recv_wr** bad_wr);
int IbvPollCqStub(struct ibv_cq* cq, int num_entries, struct ibv_wc* wc);

struct ibv_context* CreateFakeIbvContext()
{
    static struct ibv_context_ops fakeOps = {
        .poll_cq = IbvPollCqStub,
        .post_send = IbvPostSendStub,
        .post_recv = IbvPostRecvStub,
    };
    static struct ibv_context fakeCtx {};
    fakeCtx.ops = fakeOps;
    return &fakeCtx;
}

struct ibv_cq* CreateFakeCq(uint32_t cqe)
{
    auto* cq = new ibv_cq{};
    cq->context = CreateFakeIbvContext();
    cq->cqe = cqe;
    return cq;
}

struct ibv_qp* CreateFakeQp(struct ibv_cq* sendCq, struct ibv_cq* recvCq, uint32_t qpNum)
{
    auto* qp = new ibv_qp{};
    qp->context = CreateFakeIbvContext();
    qp->send_cq = sendCq;
    qp->recv_cq = recvCq;
    qp->qp_num = qpNum;
    qp->state = IBV_QPS_RESET;
    return qp;
}

static void EmitNotifyRecordTask(struct ibv_send_wr* wr, const DpuQpInfo& qpInfo)
{
    uint32_t checkerNotifyId = MakeUniqueNotifyId(qpInfo.localDeviceId);

    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    task.deviceId = qpInfo.localDeviceId;
    task.streamId = GetDpuStreamIdByDeviceId(qpInfo.localDeviceId);
    task.taskData.notify.srcDeviceId = qpInfo.localDeviceId;
    task.taskData.notify.notifyId = checkerNotifyId;
    task.taskData.notify.dstDeviceId = qpInfo.remoteDeviceId;
    task.taskData.notify.notifyCount = 1;
    task.taskData.notify.protocol = DPU_PROTOCOL_ROCE;

    FillTaskCommInfo(task, qpInfo.localDeviceId);

    uint32_t idx = 0;
    InsertTaskToCollection(&task, &idx);

    if (qpInfo.remoteDeviceId != INVALID_DEVICE_ID) {
        sim::runtime::DpuPendingNotify pending{};
        pending.dstDeviceId = qpInfo.remoteDeviceId;
        pending.notifyId = checkerNotifyId;
        pending.srcDeviceId = qpInfo.localDeviceId;
        pending.immData = wr->imm_data;
        pending.consumed = 0;
        pending.id = 0;
        auto inserted = sim::runtime::Db::Add<sim::runtime::DpuPendingNotify>(pending);
        auto pid = inserted.ok() && inserted.value.has_value() ? inserted.value->value : 0;
        HCCL_VM_INFO(
            "postSend::NotifyRecord: DpuPendingNotify INSERT id={} dstDevice={} srcDevice={} notifyId={} immData={}",
            pid, pending.dstDeviceId, pending.srcDeviceId, pending.notifyId, pending.immData);
    } else {
        HCCL_VM_WARN(
            "postSend::NotifyRecord: DpuPendingNotify SKIPPED (remoteDeviceId=INVALID) localDevice={}",
            qpInfo.localDeviceId);
    }
}

static void EmitWriteWithNotifyTask(struct ibv_send_wr* wr, const DpuQpInfo& qpInfo)
{
    HcclTaskMetaData memTask{};
    memTask.taskType = HccLTaskMetaType::MEM_CPY;
    memTask.deviceId = qpInfo.localDeviceId;
    memTask.streamId = GetDpuStreamIdByDeviceId(qpInfo.localDeviceId);
    memTask.taskData.transMem.srcDeviceId = qpInfo.localDeviceId;
    memTask.taskData.transMem.srcOffset
        = sim::GetVirPtrByDevPtr(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wr->sg_list[0].addr)));
    memTask.taskData.transMem.dstDeviceId = qpInfo.remoteDeviceId;
    memTask.taskData.transMem.dstOffset
        = sim::TransRemoteAddrToVirtualByDeviceId(wr->wr.rdma.remote_addr, qpInfo.remoteDeviceId);
    memTask.taskData.transMem.len = wr->sg_list[0].length;
    memTask.taskData.transMem.protocol = DPU_PROTOCOL_ROCE;

    FillTaskCommInfo(memTask, qpInfo.localDeviceId);

    uint32_t idx = 0;
    InsertTaskToCollection(&memTask, &idx);

    uint32_t checkerNotifyId = MakeUniqueNotifyId(qpInfo.localDeviceId);

    HcclTaskMetaData nrTask{};
    nrTask.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    nrTask.deviceId = qpInfo.localDeviceId;
    nrTask.streamId = GetDpuStreamIdByDeviceId(qpInfo.localDeviceId);
    nrTask.taskData.notify.srcDeviceId = qpInfo.localDeviceId;
    nrTask.taskData.notify.notifyId = checkerNotifyId;
    nrTask.taskData.notify.dstDeviceId = qpInfo.remoteDeviceId;
    nrTask.taskData.notify.notifyCount = 1;
    nrTask.taskData.notify.protocol = DPU_PROTOCOL_ROCE;

    FillTaskCommInfo(nrTask, qpInfo.localDeviceId);

    InsertTaskToCollection(&nrTask, &idx);

    if (qpInfo.remoteDeviceId != INVALID_DEVICE_ID) {
        sim::runtime::DpuPendingNotify pending{};
        pending.dstDeviceId = qpInfo.remoteDeviceId;
        pending.notifyId = checkerNotifyId;
        pending.srcDeviceId = qpInfo.localDeviceId;
        pending.immData = wr->imm_data;
        pending.consumed = 0;
        pending.id = 0;
        auto inserted = sim::runtime::Db::Add<sim::runtime::DpuPendingNotify>(pending);
        auto pid = inserted.ok() && inserted.value.has_value() ? inserted.value->value : 0;
        HCCL_VM_INFO(
            "postSend::WriteWithNotify: DpuPendingNotify INSERT id={} dstDevice={} srcDevice={} notifyId={} immData={}",
            pid, pending.dstDeviceId, pending.srcDeviceId, pending.notifyId, pending.immData);
    } else {
        HCCL_VM_WARN(
            "postSend::WriteWithNotify: DpuPendingNotify SKIPPED (remoteDeviceId=INVALID) localDevice={}",
            qpInfo.localDeviceId);
    }
}

static void EmitMemCpyTask(struct ibv_send_wr* wr, const DpuQpInfo& qpInfo, bool isRead)
{
    if (wr->num_sge == 0 || wr->sg_list == nullptr)
        return;

    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::MEM_CPY;
    task.deviceId = qpInfo.localDeviceId;
    task.streamId = GetDpuStreamIdByDeviceId(qpInfo.localDeviceId);

    if (isRead) {
        task.taskData.transMem.srcDeviceId = qpInfo.remoteDeviceId;
        task.taskData.transMem.srcOffset
            = sim::TransRemoteAddrToVirtualByDeviceId(wr->wr.rdma.remote_addr, qpInfo.remoteDeviceId);
        task.taskData.transMem.dstDeviceId = qpInfo.localDeviceId;
        task.taskData.transMem.dstOffset
            = sim::GetVirPtrByDevPtr(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wr->sg_list[0].addr)));
    } else {
        task.taskData.transMem.srcDeviceId = qpInfo.localDeviceId;
        task.taskData.transMem.srcOffset
            = sim::GetVirPtrByDevPtr(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wr->sg_list[0].addr)));
        task.taskData.transMem.dstDeviceId = qpInfo.remoteDeviceId;
        task.taskData.transMem.dstOffset
            = sim::TransRemoteAddrToVirtualByDeviceId(wr->wr.rdma.remote_addr, qpInfo.remoteDeviceId);
    }
    task.taskData.transMem.len = wr->sg_list[0].length;
    task.taskData.transMem.protocol = DPU_PROTOCOL_ROCE;

    FillTaskCommInfo(task, qpInfo.localDeviceId);

    uint32_t idx = 0;
    InsertTaskToCollection(&task, &idx);
}

static void EmitHybridNotifyRecordTask(struct ibv_send_wr* wr, const DpuQpInfo& qpInfo)
{
    HCCL_VM_INFO("EmitHybridNotifyRecordTask STUB");
    uint32_t checkerNotifyId = MakeUniqueNotifyId(qpInfo.localDeviceId);

    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    task.deviceId = qpInfo.localDeviceId;
    task.streamId = GetDpuStreamIdByDeviceId(qpInfo.localDeviceId);
    task.taskData.notify.srcDeviceId = qpInfo.localDeviceId;
    task.taskData.notify.notifyId = checkerNotifyId;
    task.taskData.notify.dstDeviceId = qpInfo.remoteDeviceId;
    task.taskData.notify.notifyCount = 1;
    task.taskData.notify.protocol = DPU_PROTOCOL_ROCE;

    FillTaskCommInfo(task, qpInfo.localDeviceId);

    uint32_t idx = 0;
    InsertTaskToCollection(&task, &idx);

    if (qpInfo.remoteDeviceId != INVALID_DEVICE_ID) {
        sim::runtime::DpuPendingNotify pending{};
        pending.dstDeviceId = qpInfo.remoteDeviceId;
        pending.notifyId = checkerNotifyId;
        pending.srcDeviceId = qpInfo.localDeviceId;
        pending.immData = 0;
        pending.consumed = 0;
        pending.id = 0;
        auto inserted = sim::runtime::Db::Add<sim::runtime::DpuPendingNotify>(pending);
        auto pid = inserted.ok() && inserted.value.has_value() ? inserted.value->value : 0;
        HCCL_VM_INFO(
            "postSend::HybridNotify: DpuPendingNotify INSERT id={} dstDevice={} srcDevice={} notifyId={}", pid,
            pending.dstDeviceId, pending.srcDeviceId, pending.notifyId);
    } else {
        HCCL_VM_WARN(
            "postSend::HybridNotify: DpuPendingNotify SKIPPED (remoteDeviceId=INVALID) localDevice={}",
            qpInfo.localDeviceId);
    }
}

static bool IsHybridNotifyAddr(uint64_t remoteAddr) { return g_dpuCtx.hybridNotifyAddrSet.count(remoteAddr) > 0; }

static void ProcessSendWr(struct ibv_send_wr* wr, const DpuQpInfo& qpInfo)
{
    switch (wr->opcode) {
        case IBV_WR_RDMA_WRITE_WITH_IMM:
            if (wr->num_sge > 0 && wr->sg_list && wr->sg_list[0].length > 0) {
                EmitWriteWithNotifyTask(wr, qpInfo);
            } else {
                EmitNotifyRecordTask(wr, qpInfo);
            }
            break;

        case IBV_WR_RDMA_WRITE:
            if (wr->num_sge > 0 && wr->sg_list && IsHybridNotifyAddr(wr->wr.rdma.remote_addr)) {
                EmitHybridNotifyRecordTask(wr, qpInfo);
            } else {
                EmitMemCpyTask(wr, qpInfo, false);
            }
            break;

        case IBV_WR_RDMA_READ:
            EmitMemCpyTask(wr, qpInfo, true);
            break;

        default:
            HCCL_VM_WARN("IbvPostSendStub: unhandled opcode {}", static_cast<int>(wr->opcode));
            break;
    }
}

static uint32_t ConsumeDpuPendingNotifyAndEmit(const DpuQpInfo& qpInfo)
{
    uint32_t myDeviceId = qpInfo.localDeviceId;
    uint32_t expectedSrcDeviceId = qpInfo.remoteDeviceId;

    auto pending = sim::runtime::Db::GetOneByPred<sim::runtime::DpuPendingNotify>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::DpuPendingNotify::dstDeviceId, myDeviceId),
        HcclSim::Storage::Eq(&sim::runtime::DpuPendingNotify::srcDeviceId, expectedSrcDeviceId),
        HcclSim::Storage::Eq(&sim::runtime::DpuPendingNotify::consumed, 0)));

    if (!pending.ok()) {
        return UINT32_MAX;
    }

    sim::runtime::Db::Update<sim::runtime::DpuPendingNotify>(
        HcclSim::Storage::Eq(&sim::runtime::DpuPendingNotify::id, pending->id),
        HcclSim::Storage::Set(&sim::runtime::DpuPendingNotify::consumed, 1));

    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    task.deviceId = myDeviceId;
    task.streamId = GetDpuStreamIdByDeviceId(myDeviceId);
    task.taskData.notify.srcDeviceId = pending->srcDeviceId;
    task.taskData.notify.notifyId = pending->notifyId;
    task.taskData.notify.dstDeviceId = myDeviceId;
    task.taskData.notify.notifyCount = 1;
    task.taskData.notify.protocol = DPU_PROTOCOL_ROCE;

    FillTaskCommInfo(task, myDeviceId);

    uint32_t idx = 0;
    InsertTaskToCollection(&task, &idx);

    HCCL_VM_INFO(
        "pollCq::consumePending: FOUND id={} dstDevice={} srcDevice={} notifyId={} immData={} -> NOTIFY_WAIT emitted",
        pending->id, pending->dstDeviceId, pending->srcDeviceId, pending->notifyId, pending->immData);

    return pending->immData;
}

int IbvPostSendStub(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr)
{
    // 锁顺序说明：全程持有 g_dpuCtx.mtx，以保证 qpInfo map 查询、pendingWqeCount 计数与
    // ProcessSendWr 产生的 notify 配对/任务插入的一致性。RunnerDB/InsertTaskToCollection
    // 内部锁不会反向获取 g_dpuCtx.mtx，无锁循环。
    std::lock_guard<std::mutex> lock(g_dpuCtx.mtx);
    auto it = g_dpuCtx.qpInfo.find(qp);
    if (it == g_dpuCtx.qpInfo.end()) {
        return 0;
    }

    int wrCount = 0;
    for (struct ibv_send_wr* cur = wr; cur != nullptr; cur = cur->next) {
        wrCount++;
    }
    if (qp->send_cq != nullptr) {
        g_dpuCtx.pendingWqeCount[qp->send_cq] += wrCount;
    }

    if (!it->second.isDpu) {
        return 0;
    }

    const auto& qpInfo = it->second;
    for (struct ibv_send_wr* cur = wr; cur != nullptr; cur = cur->next) {
        ProcessSendWr(cur, qpInfo);
    }

    HCCL_VM_INFO(
        "[IBV-POST] tid={:#x} elapsed={}us qp={} local={} remote={} firstOp={} firstImm={} WRcount={}", GetTidShort(),
        GetElapsedUs(), (void*)qp, qpInfo.localDeviceId, qpInfo.remoteDeviceId, static_cast<int>(wr->opcode),
        wr->imm_data, wrCount);

    return 0;
}

// recv 为 no-op：hostdpu 模拟层中接收完成依赖 send 侧生成的 notify 在 poll_cq 中配对，
// 此处不维护 recv 队列，bad_wr 出参不设置。
int IbvPostRecvStub(struct ibv_qp* qp, struct ibv_recv_wr* wr, struct ibv_recv_wr** bad_wr) { return 0; }

int IbvPollCqStub(struct ibv_cq* cq, int num_entries, struct ibv_wc* wc)
{
    if (num_entries < 1 || wc == nullptr)
        return 0;

    // 锁顺序说明：同 IbvPostSendStub，poll_cq 需与 post_send 对 g_dpuCtx 的 pendingWqeCount/
    // notify 配对状态做原子化读写，故全程持 g_dpuCtx.mtx。
    std::lock_guard<std::mutex> lock(g_dpuCtx.mtx);

    auto recvIt = g_dpuCtx.recvCqToQp.find(cq);
    if (recvIt == g_dpuCtx.recvCqToQp.end()) {
        auto& wqeCnt = g_dpuCtx.pendingWqeCount[cq];
        int toComplete = std::min(static_cast<int>(num_entries), wqeCnt);
        for (int i = 0; i < toComplete; i++) {
            std::memset(&wc[i], 0, sizeof(struct ibv_wc));
            wc[i].status = IBV_WC_SUCCESS;
            wc[i].opcode = IBV_WC_RDMA_WRITE;
        }
        wqeCnt -= toComplete;
        return toComplete;
    }

    ibv_qp* qp = recvIt->second;
    DpuQpInfo qpInfo{};
    auto qIt = g_dpuCtx.qpInfo.find(qp);
    if (qIt != g_dpuCtx.qpInfo.end()) {
        qpInfo = qIt->second;
    }

    uint32_t matched = ConsumeDpuPendingNotifyAndEmit(qpInfo);
    if (matched == UINT32_MAX) {
        return 0;
    }

    std::memset(&wc[0], 0, sizeof(struct ibv_wc));
    wc[0].status = IBV_WC_SUCCESS;
    wc[0].opcode = IBV_WC_RECV_RDMA_WITH_IMM;
    wc[0].wc_flags = IBV_WC_WITH_IMM;
    wc[0].imm_data = matched;
    wc[0].qp_num = qp->qp_num;
    return 1;
}

#ifdef __cplusplus
}
#endif // __cplusplus
