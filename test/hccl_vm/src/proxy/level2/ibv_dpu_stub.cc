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
#include <dlfcn.h>
#include <mutex>
#include <thread>
#include "sim_log.h"
#include "store_sim_store_pub.h"
#include "store_sim_run_mode.h"
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

// 评估模式原语拦截上下文: post/wait 类原语记录(channel句柄, idx)
enum class DpuPrimMode { NONE, RECORD, WAIT };
static thread_local struct {
    DpuPrimMode mode = DpuPrimMode::NONE;
    uint64_t channel = 0;
    uint32_t notifyIdx = 0;
} t_dpuNotifyCtx;

void DpuSetNotifyRecordContext(uint64_t channel, uint32_t remoteNotifyIdx)
{
    t_dpuNotifyCtx.mode = DpuPrimMode::RECORD;
    t_dpuNotifyCtx.channel = channel;
    t_dpuNotifyCtx.notifyIdx = remoteNotifyIdx;
}

void DpuSetNotifyWaitContext(uint64_t channel, uint32_t localNotifyIdx)
{
    t_dpuNotifyCtx.mode = DpuPrimMode::WAIT;
    t_dpuNotifyCtx.channel = channel;
    t_dpuNotifyCtx.notifyIdx = localNotifyIdx;
}

void DpuClearNotifyContext(void)
{
    t_dpuNotifyCtx.mode = DpuPrimMode::NONE;
    t_dpuNotifyCtx.channel = 0;
    t_dpuNotifyCtx.notifyIdx = 0;
}

// DPU 任务公共元数据: taskSrc + profInfo + rank + eids
static void FillDpuTaskMeta(HcclTaskMetaData& task, const DpuQpInfo& qpInfo, ProfTaskParamType profTaskType)
{
    task.taskSrc = TASK_SRC_DPU;
    task.profInfo = HcclTaskProfInfo(profTaskType, ProfLinkType::LINK_ROCE, ProfTransportType::TRANSPORT_ROCE);
    FillTaskCommInfo(task, qpInfo.localDeviceId);
    std::memcpy(task.lcEid, qpInfo.lcEid, sizeof(task.lcEid));
    std::memcpy(task.rmEid, qpInfo.rmEid, sizeof(task.rmEid));
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

// 时序绑定: RaGetQpAttr记录QP, 同线程下一次socket send绑定真实对端
static thread_local std::vector<struct ibv_qp*> t_pendingBindQps;

void DpuRecordPendingBindQp(struct ibv_qp* fakeQp)
{
    if (fakeQp == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_dpuCtx.mtx);
    auto it = g_dpuCtx.qpInfo.find(fakeQp);
    if (it != g_dpuCtx.qpInfo.end() && !it->second.bindAttempted) {
        t_pendingBindQps.push_back(fakeQp);
    }
}

// 同线程socket send时绑定待关联QP到该socket的真实对端
void BindDpuPendingQps(uint32_t peerDeviceId, const uint8_t* peerEid, const uint8_t* localEid)
{
    if (!sim::IsEstimateMode() || t_pendingBindQps.empty()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(g_dpuCtx.mtx);
        for (auto* qp : t_pendingBindQps) {
            auto it = g_dpuCtx.qpInfo.find(qp);
            if (it == g_dpuCtx.qpInfo.end() || it->second.bindAttempted) {
                continue;
            }

            // 更新DpuQpInfo信息
            it->second.bindAttempted = true;
            if (peerDeviceId == INVALID_DEVICE_ID) {
                continue; // 本socket对端未解析, 放弃该QP(保持未绑定, 不误绑到其他socket)
            }

            std::memcpy(it->second.lcEid, localEid, sizeof(it->second.lcEid));
            BinaryAddr ba{};
            std::memcpy(&ba, peerEid, sizeof(ba));
            IpAddress peerAddr(ba, AF_INET6);
            uint32_t peerRank = UINT32_MAX;
            if (!sim::runtime::GetRankIdByEid(peerAddr, peerRank)) {
                HCCL_VM_ERROR(
                    "bind: peerRank unresolved dev={} eid={}", peerDeviceId, peerAddr.EidToHexString().c_str());
                peerRank = UINT32_MAX;
            }

            it->second.peerRankId = peerRank;
            HCCL_VM_INFO(
                "bind: qp={:p} peerDev={} peerRank={}", static_cast<void*>(qp), peerDeviceId, peerRank,
                peerAddr.EidToHexString().c_str());
        }
    }
    t_pendingBindQps.clear();
}

#ifdef __cplusplus
// 评估模式通道创建预填: 按创建序号推断槽位imm
void DpuOnChannelCreatedEstimate(uint64_t channel, uint32_t notifyNum)
{
    if (!sim::IsEstimateMode() || channel == 0) {
        return;
    }
    if (notifyNum == 0 || notifyNum > DPU_NOTIFY_NUM_MAX) {
        HCCL_VM_WARN(
            "prefill: ch={:x} invalid notifyNum({}), fallback to {}", channel, notifyNum, DPU_NOTIFY_NUM_DEFAULT);
        notifyNum = DPU_NOTIFY_NUM_DEFAULT;
    }

    std::lock_guard<std::mutex> lock(g_dpuCtx.mtx);
    uint32_t seq = g_dpuCtx.channelCreateSeq;
    g_dpuCtx.channelCreateSeq += 1;
    uint32_t base = seq * notifyNum;
    auto& slots = g_dpuCtx.channelSlots[channel];
    for (uint32_t idx = 0; idx < notifyNum; idx++) {
        slots[idx] = {base + idx, 0}; // notifyId占位0, post校准时覆盖
    }
    HCCL_VM_INFO("prefill: ch={:x} seq={} num={} imm[{}..{}]", channel, seq, notifyNum, base, base + notifyNum - 1);
}

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

// 评估模式: 记录槽位上下文(channel idx -> imm/notifyId)
static void RecordDpuSlotContextEstimate(const DpuQpInfo& qpInfo, const char* tag, uint32_t notifyId, uint32_t immData)
{
    if (t_dpuNotifyCtx.mode == DpuPrimMode::RECORD && t_dpuNotifyCtx.channel != 0) {
        // 推断imm与实际不符时告警(推断假设失效), 以实际为准
        auto& slots = g_dpuCtx.channelSlots[t_dpuNotifyCtx.channel];
        auto it = slots.find(t_dpuNotifyCtx.notifyIdx);
        if (it != slots.end() && it->second.first != immData) {
            HCCL_VM_ERROR(
                "slotCtx calib: ch={:x} idx={} inferred imm={} actual imm={} — assumption broken",
                t_dpuNotifyCtx.channel, t_dpuNotifyCtx.notifyIdx, it->second.first, immData);
        }
        slots[t_dpuNotifyCtx.notifyIdx] = {immData, notifyId};
        HCCL_VM_INFO(
            "slotCtx: {} ch={:x} idx={} imm={} nid={}", tag, t_dpuNotifyCtx.channel, t_dpuNotifyCtx.notifyIdx, immData,
            notifyId);
    } else {
        HCCL_VM_ERROR(
            "post not intercepted: {} primMode={} ch={:x} — add interception in hcomm_api_stub.cc", tag,
            static_cast<int>(t_dpuNotifyCtx.mode), t_dpuNotifyCtx.channel);
    }
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

// 评估模式：deviceId字段承载通信域rankId
static void EmitNotifyRecordTaskEstimate(struct ibv_send_wr* wr, const DpuQpInfo& qpInfo)
{
    uint32_t checkerNotifyId = MakeUniqueNotifyId(qpInfo.localDeviceId);

    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    task.deviceId = qpInfo.localDeviceId;
    task.streamId = GetDpuStreamIdByDeviceId(qpInfo.localDeviceId);
    FillDpuTaskMeta(task, qpInfo, ProfTaskParamType::TASK_DPU_INLINE_WRITE);
    task.taskData.notify.srcDeviceId = task.rankId;
    task.taskData.notify.notifyId = checkerNotifyId;
    task.taskData.notify.dstDeviceId = qpInfo.peerRankId;
    task.taskData.notify.notifyCount = 1;
    task.taskData.notify.protocol = DPU_PROTOCOL_ROCE;

    uint32_t idx = 0;
    InsertTaskToCollection(&task, &idx);

    RecordDpuSlotContextEstimate(qpInfo, "NotifyRecord", checkerNotifyId, wr->imm_data);
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

// 评估模式：Dpu_Write_With_Notify复合任务不拆分，deviceId字段承载通信域rankId
static void EmitWriteWithNotifyTaskEstimate(struct ibv_send_wr* wr, const DpuQpInfo& qpInfo)
{
    uint32_t checkerNotifyId = MakeUniqueNotifyId(qpInfo.localDeviceId);

    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::WRITE_WITH_NOTIFY;
    task.deviceId = qpInfo.localDeviceId;
    task.streamId = GetDpuStreamIdByDeviceId(qpInfo.localDeviceId);
    FillDpuTaskMeta(task, qpInfo, ProfTaskParamType::TASK_DPU_WRITE_WITH_NOTIFY);
    WriteWithNotifyTask& wn = task.taskData.writeWithNotify;
    wn.srcDeviceId = task.rankId;
    wn.srcOffset = sim::GetVirPtrByDevPtr(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wr->sg_list[0].addr)));
    wn.dstDeviceId = qpInfo.peerRankId;
    wn.dstOffset = wr->wr.rdma.remote_addr;
    wn.len = wr->sg_list[0].length;
    wn.notifyId = checkerNotifyId;
    wn.isReduce = 0;
    wn.dataType = 0;
    wn.reduceOp = 0;
    wn.protocol = DPU_PROTOCOL_ROCE;
    wn.notifyCount = 1;

    uint32_t idx = 0;
    InsertTaskToCollection(&task, &idx);

    RecordDpuSlotContextEstimate(qpInfo, "WriteWithNotify", checkerNotifyId, wr->imm_data);
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

// 评估模式：deviceId字段承载通信域rankId
static void EmitMemCpyTaskEstimate(struct ibv_send_wr* wr, const DpuQpInfo& qpInfo, bool isRead)
{
    if (wr->num_sge == 0 || wr->sg_list == nullptr)
        return;

    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::MEM_CPY;
    task.deviceId = qpInfo.localDeviceId;
    task.streamId = GetDpuStreamIdByDeviceId(qpInfo.localDeviceId);
    FillDpuTaskMeta(task, qpInfo, ProfTaskParamType::TASK_RDMA);
    if (isRead) {
        task.taskData.transMem.srcDeviceId = qpInfo.peerRankId;
        task.taskData.transMem.srcOffset = wr->wr.rdma.remote_addr;
        task.taskData.transMem.dstDeviceId = task.rankId;
        task.taskData.transMem.dstOffset
            = sim::GetVirPtrByDevPtr(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wr->sg_list[0].addr)));
    } else {
        task.taskData.transMem.srcDeviceId = task.rankId;
        task.taskData.transMem.srcOffset
            = sim::GetVirPtrByDevPtr(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wr->sg_list[0].addr)));
        task.taskData.transMem.dstDeviceId = qpInfo.peerRankId;
        task.taskData.transMem.dstOffset = wr->wr.rdma.remote_addr;
    }
    task.taskData.transMem.len = wr->sg_list[0].length;
    task.taskData.transMem.protocol = DPU_PROTOCOL_ROCE;

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

// 评估模式：deviceId字段承载通信域rankId
static void EmitHybridNotifyRecordTaskEstimate(struct ibv_send_wr* wr, const DpuQpInfo& qpInfo)
{
    uint32_t checkerNotifyId = MakeUniqueNotifyId(qpInfo.localDeviceId);

    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    task.deviceId = qpInfo.localDeviceId;
    task.streamId = GetDpuStreamIdByDeviceId(qpInfo.localDeviceId);
    FillDpuTaskMeta(task, qpInfo, ProfTaskParamType::TASK_DPU_INLINE_WRITE);
    task.taskData.notify.srcDeviceId = task.rankId;
    task.taskData.notify.notifyId = checkerNotifyId;
    task.taskData.notify.dstDeviceId = qpInfo.peerRankId;
    task.taskData.notify.notifyCount = 1;
    task.taskData.notify.protocol = DPU_PROTOCOL_ROCE;

    uint32_t idx = 0;
    InsertTaskToCollection(&task, &idx);

    RecordDpuSlotContextEstimate(qpInfo, "HybridNotify", checkerNotifyId, 0);
}

static bool IsHybridNotifyAddr(uint64_t remoteAddr) { return g_dpuCtx.hybridNotifyAddrSet.count(remoteAddr) > 0; }

static void ProcessSendWr(struct ibv_send_wr* wr, const DpuQpInfo& qpInfo)
{
    switch (wr->opcode) {
        case IBV_WR_RDMA_WRITE_WITH_IMM:
            if (wr->num_sge > 0 && wr->sg_list && wr->sg_list[0].length > 0) {
                sim::IsEstimateMode() ? EmitWriteWithNotifyTaskEstimate(wr, qpInfo) :
                                        EmitWriteWithNotifyTask(wr, qpInfo);
            } else {
                sim::IsEstimateMode() ? EmitNotifyRecordTaskEstimate(wr, qpInfo) : EmitNotifyRecordTask(wr, qpInfo);
            }
            break;

        case IBV_WR_RDMA_WRITE:
            if (wr->num_sge > 0 && wr->sg_list && IsHybridNotifyAddr(wr->wr.rdma.remote_addr)) {
                sim::IsEstimateMode() ? EmitHybridNotifyRecordTaskEstimate(wr, qpInfo) :
                                        EmitHybridNotifyRecordTask(wr, qpInfo);
            } else {
                sim::IsEstimateMode() ? EmitMemCpyTaskEstimate(wr, qpInfo, false) : EmitMemCpyTask(wr, qpInfo, false);
            }
            break;

        case IBV_WR_RDMA_READ:
            sim::IsEstimateMode() ? EmitMemCpyTaskEstimate(wr, qpInfo, true) : EmitMemCpyTask(wr, qpInfo, true);
            break;

        default:
            HCCL_VM_WARN("IbvPostSendStub: unhandled opcode {}", static_cast<int>(wr->opcode));
            break;
    }
}

// 评估模式：deviceId字段承载通信域rankId
static void EmitNotifyWaitTaskEstimate(uint64_t notifyId, const DpuQpInfo& qpInfo)
{
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    task.deviceId = qpInfo.localDeviceId;
    task.streamId = GetDpuStreamIdByDeviceId(qpInfo.localDeviceId);
    FillDpuTaskMeta(task, qpInfo, ProfTaskParamType::TASK_DPU_NOTIFY_WAIT);
    task.taskData.notify.srcDeviceId = qpInfo.peerRankId;
    task.taskData.notify.notifyId = notifyId;
    task.taskData.notify.dstDeviceId = task.rankId;
    task.taskData.notify.notifyCount = 1;
    task.taskData.notify.protocol = DPU_PROTOCOL_ROCE;

    uint32_t idx = 0;
    InsertTaskToCollection(&task, &idx);
}

static uint32_t ConsumeDpuPendingNotifyAndEmit(const DpuQpInfo& qpInfo)
{
    uint32_t myDeviceId = qpInfo.localDeviceId;
    uint32_t expectedSrcDeviceId = qpInfo.remoteDeviceId;

    // 评估模式精确路径: wait原语上下文 + 槽位映射 -> 期望imm精确已知,
    // 直接合成"虚拟对端post", imm必过hcomm硬校验
    if (sim::IsEstimateMode() && t_dpuNotifyCtx.mode == DpuPrimMode::WAIT && t_dpuNotifyCtx.channel != 0) {
        auto chIt = g_dpuCtx.channelSlots.find(t_dpuNotifyCtx.channel);
        if (chIt != g_dpuCtx.channelSlots.end()) {
            auto slotIt = chIt->second.find(t_dpuNotifyCtx.notifyIdx);
            if (slotIt != chIt->second.end()) {
                EmitNotifyWaitTaskEstimate(slotIt->second.second, qpInfo);
                HCCL_VM_INFO(
                    "exact: qpn={} ch={:x} idx={} imm={} nid={}", qpInfo.qpNum, t_dpuNotifyCtx.channel,
                    t_dpuNotifyCtx.notifyIdx, slotIt->second.first, slotIt->second.second);
                return slotIt->second.first;
            }
        }
    }

    // 评估模式不消费pending表(无写入者), exact未命中即未拦截入口
    // 每(qpn, idx)组合只报一次, 避免busy-loop刷爆日志
    if (sim::IsEstimateMode()) {
        uint64_t missKey = (static_cast<uint64_t>(qpInfo.qpNum) << 32) | t_dpuNotifyCtx.notifyIdx;
        if (g_dpuCtx.reportedMiss.insert(missKey).second) {
            HCCL_VM_ERROR(
                "wait not intercepted: qpn={} primMode={} ch={:x} idx={} — add interception in hcomm_api_stub.cc",
                qpInfo.qpNum, static_cast<int>(t_dpuNotifyCtx.mode), t_dpuNotifyCtx.channel, t_dpuNotifyCtx.notifyIdx);
        }
        return UINT32_MAX;
    }

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
