/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Description: 数据面通信操作打桩函数（北向劫持）
 *              通过 Channel 进行跨节点通信操作，作为 Checker 的输入。
 * Create: 2026-07-15
 */

#define HCCL_VM_MODULE "COMM_OP"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>

#include "hccl_proxy_common.h"
#include "level1_proxy_common.h"
#include "sim_log.h"
#include "sim_common_defs.h"
#include "sim_capacity_limits.h"
#include "runtime_state/sim_models.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "store_sim_store_pub.h"

#include "dpu/dpu_level1_common.h"

namespace {

bool GetTaskDeviceId(uint64_t commId, uint32_t rankId, uint32_t& deviceId)
{
    sim::runtime::Device device{};
    if (sim::runtime::GetDeviceByCommRank(commId, rankId, device) == ACL_SUCCESS) {
        deviceId = static_cast<uint32_t>(device.id);
        return true;
    }
    const int rankTableDeviceId = sim::RankTable::Instance().GetDeviceId(rankId);
    if (rankTableDeviceId < 0) {
        HCCL_VM_ERROR("cannot map commId={}, rankId={} to deviceId", commId, rankId);
        return false;
    }
    const auto dbDevice = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(HcclSim::Storage::Or(
        HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, static_cast<uint32_t>(rankTableDeviceId)),
        HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, static_cast<uint32_t>(rankTableDeviceId))));
    if (!dbDevice.ok()) {
        HCCL_VM_ERROR(
            "cannot resolve database deviceId for commId={}, rankId={}, physicalDeviceId={}", commId, rankId,
            rankTableDeviceId);
        return false;
    }
    deviceId = static_cast<uint32_t>(dbDevice->id);
    return true;
}

bool NormalizeLevel1Task(HcclTaskMetaData* task)
{
    if (task == nullptr) {
        return false;
    }
    const auto communicator = sim::runtime::Db::GetById<sim::runtime::Communicator>(task->commId);
    if (!communicator.ok()) {
        HCCL_VM_ERROR("cannot normalize task: commId={} not found", task->commId);
        return false;
    }
    const uint32_t generatedRankId = task->rankId;
    const uint32_t localRankId = communicator->rank_id;
    task->rankId = localRankId;
    const auto normalizeEndpoint = [generatedRankId, localRankId](uint32_t& rankId) {
        if (rankId == generatedRankId) {
            rankId = localRankId;
        }
    };
    uint32_t deviceId = 0;
    if (!GetTaskDeviceId(task->commId, task->rankId, deviceId)) {
        return false;
    }
    task->deviceId = deviceId;

    switch (task->taskType) {
        case HccLTaskMetaType::MEM_CPY:
            normalizeEndpoint(task->taskData.transMem.srcDeviceId);
            normalizeEndpoint(task->taskData.transMem.dstDeviceId);
            if (!GetTaskDeviceId(task->commId, task->taskData.transMem.srcDeviceId, task->taskData.transMem.srcDeviceId)
                || !GetTaskDeviceId(
                    task->commId, task->taskData.transMem.dstDeviceId, task->taskData.transMem.dstDeviceId)) {
                return false;
            }
            break;
        case HccLTaskMetaType::REDUCE:
            normalizeEndpoint(task->taskData.reduce.srcDeviceId);
            normalizeEndpoint(task->taskData.reduce.dstDeviceId);
            if (!GetTaskDeviceId(task->commId, task->taskData.reduce.srcDeviceId, task->taskData.reduce.srcDeviceId)
                || !GetTaskDeviceId(
                    task->commId, task->taskData.reduce.dstDeviceId, task->taskData.reduce.dstDeviceId)) {
                return false;
            }
            break;
        case HccLTaskMetaType::NOTIFY_RECORD:
        case HccLTaskMetaType::NOTIFY_WAIT:
            normalizeEndpoint(task->taskData.notify.srcDeviceId);
            normalizeEndpoint(task->taskData.notify.dstDeviceId);
            if (!GetTaskDeviceId(task->commId, task->taskData.notify.srcDeviceId, task->taskData.notify.srcDeviceId)
                || !GetTaskDeviceId(
                    task->commId, task->taskData.notify.dstDeviceId, task->taskData.notify.dstDeviceId)) {
                return false;
            }
            break;
        default:
            break;
    }
    task->rankId = UINT32_MAX;
    return true;
}

/**
 * @brief 归一化并插入任务到任务集合（本文件数据面桩的统一出口）。
 * @param task  输入输出：任务元数据（先 Normalize，再按 DPU 上下文打 protocol 标注）。
 * @param index 输出：插入后的任务索引。
 * @return HcclVmResult 成功返回 HCCL_SIM_SUCCESS，归一化失败返回 HCCL_SIM_E_INTERNAL。
 */
HcclSim::HcclVmResult InsertLevel1Task(HcclTaskMetaData* task, uint32_t* index)
{
    if (!NormalizeLevel1Task(task)) {
        return HcclSim::HCCL_SIM_E_INTERNAL;
    }
    // DPU 执行器回调期产生的任务：地址从设备映射空间翻译为 VM 虚拟空间
    // （checker 布局空间），并统一标注 protocol=ROCE；非 DPU 上下文均为 no-op
    if (task != nullptr && HcclSim::DpuSim::IsDpuContext()) {
        HcclSim::DpuSim::TranslateDpuTaskAddress(*task);
        HcclSim::DpuSim::ApplyDpuTaskProtocol(*task);
    }
    return InsertTaskToCollection(task, index);
}

/*==================== HostDPU（level1）DPU 上下文回退与反向通道解析 ====================*/

// 反向通道轮询等待参数（与设备侧 data_op_device_stub.cc 同口径）
constexpr uint32_t REVERSE_CHANNEL_WAIT_MAX_MS = 10000;
constexpr uint32_t REVERSE_CHANNEL_WAIT_POLL_INTERVAL_MS = 10;

/**
 * @brief DPU 上下文（执行器回调期，thread=0）的任务归属回退。
 *
 * DPU 数据面调用不携带有效线程句柄，无法经 HcclThread 解析 commId/streamId；
 * 回退到执行器回调前置位的 TLS 上下文（dpu_level1_common.h DpuExecContext）。
 *
 * @param commId   输出：通信域 id。
 * @param streamId 输出：DPU 任务归属流（DpuDeviceInfo.stream_id）。
 * @return 处于 DPU 上下文返回 true，否则 false。
 */
bool ResolveDpuFallbackCtx(uint64_t& commId, uint64_t& streamId)
{
    const auto& tls = HcclSim::DpuSim::GetDpuExecTls();
    if (!tls.active) {
        return false;
    }
    commId = tls.commId;
    streamId = tls.dpuStreamId;
    return true;
}

/**
 * @brief 查找指向本端的反向 channel 并读取其 notifyId[notifyIdx]（DPU 上下文专用）。
 *
 * NOTIFY_RECORD 向对端发信号，须引用**接收方**通道上的 notify 槽位 id，
 * 使远端 rank 的 NOTIFY_WAIT 任务按同一资源配对（Checker SyncResourceKey）。
 * 对端通道可能尚未入库（跨 rank 建链时序），按既有口径轮询等待至 10s。
 * 移植自设备侧 data_op_device_stub.cc GetReverseChannelNotifyId。
 *
 * @param localCommId 输入：本端通信域 id（Communicator 表主键）。
 * @param localRank   输入：本端 rank id。
 * @param remoteRank  输入：对端 rank id。
 * @param engine      输入：本端 channel 的引擎类型（反向 channel 必须同 engine）。
 * @param notifyIdx   输入：对端 notify 槽位索引。
 * @param notifyId    输出：对端通道 notifyId[notifyIdx] 的 Notify 记录 id。
 * @return int32_t 成功返回 0；对端通信域/通道未就绪或索引越界返回 HCCL_E_NOT_FOUND。
 */
int32_t GetReverseChannelNotifyIdForDpu(
    uint64_t localCommId, uint32_t localRank, uint32_t remoteRank, uint8_t engine, uint32_t notifyIdx,
    uint64_t& notifyId)
{
    const auto localComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(localCommId);
    if (!localComm.ok()) {
        HCCL_VM_ERROR("{}: local communicator={} not found", __func__, localCommId);
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(REVERSE_CHANNEL_WAIT_MAX_MS);
    while (true) {
        // 按 rank_id + rank_size + comm_hash + comm_id 严格匹配对端通信域（避免同名多通信域误匹配）
        auto remoteComms = sim::runtime::Db::GetByPred<sim::runtime::Communicator>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Communicator::rank_id, static_cast<uint64_t>(remoteRank)),
            HcclSim::Storage::Eq(&sim::runtime::Communicator::rank_size, localComm->rank_size),
            HcclSim::Storage::Eq(&sim::runtime::Communicator::comm_hash, localComm->comm_hash),
            HcclSim::Storage::Eq(&sim::runtime::Communicator::comm_id, std::string(localComm->comm_id))));
        if (remoteComms.ok() && !remoteComms->empty()) {
            const uint64_t remoteCommId = remoteComms->front().id;
            auto reverseChannels = sim::runtime::Db::GetByPred<sim::runtime::HcclChannel>(HcclSim::Storage::And(
                HcclSim::Storage::Eq(&sim::runtime::HcclChannel::commId, remoteCommId),
                HcclSim::Storage::Eq(&sim::runtime::HcclChannel::remoteRankId, static_cast<uint64_t>(localRank)),
                HcclSim::Storage::Eq(&sim::runtime::HcclChannel::engine, static_cast<uint64_t>(engine)),
                HcclSim::Storage::Eq(&sim::runtime::HcclChannel::status, true)));
            if (reverseChannels.ok() && !reverseChannels->empty()) {
                const auto& reverseCh = reverseChannels->front();
                if (notifyIdx < reverseCh.notifyNum) {
                    notifyId = reverseCh.notifyId[notifyIdx];
                    return 0;
                }
                HCCL_VM_ERROR(
                    "{}: notifyIdx {:d} out of range (notifyNum={:d})", __func__, notifyIdx, reverseCh.notifyNum);
                return static_cast<int32_t>(HCCL_E_NOT_FOUND);
            }
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            HCCL_VM_ERROR(
                "{}: reverse channel not ready after {:d}ms (localRank={:d}, remoteRank={:d})", __func__,
                REVERSE_CHANNEL_WAIT_MAX_MS, localRank, remoteRank);
            return static_cast<int32_t>(HCCL_E_NOT_FOUND);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(REVERSE_CHANNEL_WAIT_POLL_INTERVAL_MS));
    }
}

/**
 * @brief 选择 NOTIFY_RECORD 任务的 notifyId（DPU 上下文取反向通道槽位）。
 *
 * 常规路径（AIV/AICPU 等）：取本端通道 notifyId[remoteNotifyIdx] 的 Notify 记录 id。
 * DPU 上下文（HostDPU，执行器回调期）：取**接收方**反向通道的同索引槽位，
 * 使远端 rank 的 NOTIFY_WAIT（取本端槽位）按同一逻辑资源配对；
 * 反向解析失败时告警并退化回本端通道槽位，保证任务仍可记录。
 *
 * @param channel        输入：本端通道记录指针（可为空，空表示通道未入库）。
 * @param channelHandle  输入：通道句柄（退化场景的兜底 notifyId）。
 * @param remoteNotifyIdx 输入：远端 notify 槽位索引。
 * @param localRank      输入：本端 rank id。
 * @param remoteRank     输入：对端 rank id。
 * @param caller         输入：调用方函数名（日志用）。
 * @return 用于任务元数据的 notifyId。
 */
uint64_t ResolveRecordNotifyId(
    const sim::runtime::HcclChannel* channel, uint64_t channelHandle, uint32_t remoteNotifyIdx, uint32_t localRank,
    uint32_t remoteRank, const char* caller)
{
    // DPU 上下文：优先解析反向通道（接收方）槽位
    if (HcclSim::DpuSim::IsDpuContext() && channel != nullptr) {
        const auto& tls = HcclSim::DpuSim::GetDpuExecTls();
        uint64_t reverseNotifyId = 0;
        const int32_t ret = GetReverseChannelNotifyIdForDpu(
            tls.commId, localRank, remoteRank, channel->engine, remoteNotifyIdx, reverseNotifyId);
        if (ret == 0) {
            return reverseNotifyId;
        }
        HCCL_VM_WARN("{}: {} resolve reverse channel notifyId failed, ret={:d}, fallback local", caller, __func__, ret);
    }
    // 常规路径：本端通道槽位
    if (channel != nullptr) {
        if (remoteNotifyIdx < channel->notifyNum) {
            return channel->notifyId[remoteNotifyIdx];
        }
        HCCL_VM_WARN(
            "{}: remoteNotifyIdx {:d} out of range (notifyNum={:d})", caller, remoteNotifyIdx, channel->notifyNum);
    }
    return channelHandle;
}

/**
 * @brief 生成 notify 任务 id（DPU 上下文按事件唯一编码，其余场景维持原编码）。
 *
 * DPU（RoCE）语义下同一槽位可被连续多次 record（如 NHR 多 slice，FIFO 逐次消费），
 * 用事件级唯一 id 还原 record/wait 的 1:1 配对，避免槽位复用被 Checker 误判为
 * 多打一冲突；RECORD 侧与 WAIT 侧各自进程对同一槽位（同一 Notify 行 id）独立计数，
 * 模板中同槽位事件的顺序与数量一一对应，序号天然对齐，无需跨进程协调。
 *
 * @param rankIdForEncode 输入：非 DPU 场景下 MakeNotifyTaskId 的 rank 分量。
 * @param notifyIdVal     输入：槽位对应的 Notify 表行 id。
 * @return 用于任务元数据的 notify id。
 */
uint64_t MakeDpuAwareNotifyTaskId(uint32_t rankIdForEncode, uint64_t notifyIdVal)
{
    if (HcclSim::DpuSim::IsDpuContext()) {
        return HcclSim::DpuSim::EncodeDpuNotifyTaskId(notifyIdVal, HcclSim::DpuSim::NextDpuNotifySeq(notifyIdVal));
    }
    return HcclSim::MakeNotifyTaskId(rankIdForEncode, notifyIdVal);
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 通过 channel 向远端内存写入数据（数据面接口）。
 *
 * 将 src 指向的 len 字节的数据，通过 channel 写入到远端 dst 所指向的内存中。
 * 接口调用方为 src 所在节点（本端为 local/src，远端为 remote/dst）。
 *
 * 本桩以 MEM_CPY 任务记录到 TaskCollection 与 DB，供 Checker 做冲突分析。
 * 不做实际的跨节点数据传输。
 *
 * @param thread  输入：线程句柄。
 * @param channel 输入：通道句柄，由 HcommChannelCreate / HcclChannelAcquire 创建。
 * @param dst     输入：远端 device 内存地址（对端）。
 * @param src     输入：本地 device 内存地址（本端）。
 * @param len     输入：待拷贝字节数。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    if (dst == nullptr || src == nullptr) {
        HCCL_VM_ERROR("{}: dst or src is nullptr", __func__);
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    if (len == 0u) {
        HCCL_VM_ERROR("{}: len is 0", __func__);
        return static_cast<int32_t>(HCCL_E_PARA);
    }

    HCCL_VM_INFO(
        "{}: thread={:d}, channel={:d}, dst={:p}, src={:p}, len={:d}", __func__, thread, channel, dst, src, len);

    // 【核心步骤 1】获取当前 rank ID，用于标记操作发起方
    uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());

    // 【核心步骤 2】查询线程信息，获取所属通信域和流 ID
    // - commId: 关联操作到通信域
    // - streamId: 确定操作执行的流
    uint64_t commId = 0u;
    uint64_t streamId = 0u;
    auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(thread);
    if (optThread.ok()) {
        commId = optThread->commId;
        streamId = optThread->streamId;
    } else if (!ResolveDpuFallbackCtx(commId, streamId)) { // DPU 回调期（thread=0）经 TLS 回退归属
        HCCL_VM_WARN("{}: thread {:d} not found", __func__, thread);
    }

    // 【核心步骤 3】查询通道信息，获取远端 rank ID
    // - channel 记录了 local ↔ remote 的连接关系
    // - remoteRankId 标识数据的目标节点
    uint32_t remoteRankId = 0u;
    auto optChannel = sim::runtime::Db::GetById<sim::runtime::HcclChannel>(channel);
    if (optChannel.ok()) {
        remoteRankId = static_cast<uint32_t>(optChannel->remoteRankId);
    } else {
        HCCL_VM_WARN("{}: channel {:d} not found", __func__, channel);
    }

    // 【核心步骤 4】构建 MEM_CPY 任务元数据
    // Write 操作语义（从调用方视角）：
    //   - 调用方是 src 所在节点（local = src）
    //   - dst 是远端节点（remote = dst）
    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::MEM_CPY;
    taskMetaData.commId = commId;
    taskMetaData.rankId = curRank; // 操作发起方
    taskMetaData.streamId = streamId;

    // 写入远端：srcRankId = 本地, dstRankId = 远端
    taskMetaData.taskData.transMem.srcDeviceId = curRank;
    taskMetaData.taskData.transMem.srcOffset = reinterpret_cast<uint64_t>(src);
    taskMetaData.taskData.transMem.dstDeviceId = remoteRankId;
    taskMetaData.taskData.transMem.dstOffset = reinterpret_cast<uint64_t>(dst);
    taskMetaData.taskData.transMem.len = len;

    // 【核心步骤 5】插入任务到集合，返回任务索引
    // - Checker 通过此索引追踪操作执行顺序
    // - 失败则回滚所有操作
    uint32_t taskIndex = 0u;
    auto insertRet = InsertLevel1Task(&taskMetaData, &taskIndex);
    if (insertRet != HcclSim::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("{}: InsertTaskToCollection failed, ret={:d}", __func__, static_cast<int>(insertRet));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, channel={:d}, "
        "remoteRank={:d}, taskIndex={:d}",
        __func__, thread, channel, remoteRankId, taskIndex);
    return 0;
}

/**
 * @brief 通过 channel 向远端内存写数据并做 reduce（数据面接口）。
 *
 * 将 src 中 count * sizeof(dataType) 字节的数据，通过 channel 与远端 dst
 * 所指向的相同长度的内存数据进行 reduceOp 操作，结果输出到远端 dst 中。
 * 接口调用方为 src 所在节点（本端为 local/src，远端为 remote/dst）。
 *
 * @param thread   输入：线程句柄。
 * @param channel  输入：通道句柄。
 * @param dst      输入：远端 device 内存地址（对端）。
 * @param src      输入：本地 device 内存地址（本端）。
 * @param count    输入：元素数量（非字节数）。
 * @param dataType 输入：元素数据类型。
 * @param reduceOp 输入：reduce 操作类型。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommWriteReduceOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    if (dst == nullptr || src == nullptr) {
        HCCL_VM_ERROR("{}: dst or src is nullptr", __func__);
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    if (count == 0u) {
        HCCL_VM_ERROR("{}: count is 0", __func__);
        return static_cast<int32_t>(HCCL_E_PARA);
    }

    HCCL_VM_INFO(
        "{}: thread={:d}, channel={:d}, dst={:p}, src={:p}, count={:d}, dataType={:d}, reduceOp={:d}", __func__, thread,
        channel, dst, src, count, static_cast<int>(dataType), static_cast<int>(reduceOp));

    // 【核心步骤 1】获取当前 rank ID
    uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());

    // 【核心步骤 2】查询线程的 commId 和 streamId
    uint64_t commId = 0u;
    uint64_t streamId = 0u;
    auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(thread);
    if (optThread.ok()) {
        commId = optThread->commId;
        streamId = optThread->streamId;
    } else if (!ResolveDpuFallbackCtx(commId, streamId)) { // DPU 回调期（thread=0）经 TLS 回退归属
        HCCL_VM_WARN("{}: thread {:d} not found", __func__, thread);
    }

    // 【核心步骤 3】查询通道的远端 rank ID
    uint32_t remoteRankId = 0u;
    auto optChannel = sim::runtime::Db::GetById<sim::runtime::HcclChannel>(channel);
    if (optChannel.ok()) {
        remoteRankId = static_cast<uint32_t>(optChannel->remoteRankId);
    } else {
        HCCL_VM_WARN("{}: channel {:d} not found", __func__, channel);
    }

    // 【核心步骤 4】构建 REDUCE 任务元数据
    // WriteReduce 操作语义（从调用方视角）：
    //   - 调用方是 src 所在节点（local = src）
    //   - dst 是远端节点（remote = dst）
    //   - 在远端执行 reduce：local_data ⊕ remote_data → remote_data
    uint32_t typeSize = 0;
    sim::GetDataTypeSize(static_cast<HcclDataType>(dataType), typeSize);
    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::REDUCE;
    taskMetaData.commId = commId;
    taskMetaData.rankId = curRank;
    taskMetaData.streamId = streamId;

    // Write: 本地是 src，远端是 dst
    taskMetaData.taskData.reduce.srcDeviceId = curRank;
    taskMetaData.taskData.reduce.srcOffset = reinterpret_cast<uint64_t>(src);
    taskMetaData.taskData.reduce.dstDeviceId = remoteRankId;
    taskMetaData.taskData.reduce.dstOffset = reinterpret_cast<uint64_t>(dst);
    taskMetaData.taskData.reduce.dataCount = count * typeSize;
    taskMetaData.taskData.reduce.dataType = static_cast<uint8_t>(dataType);
    taskMetaData.taskData.reduce.reduceOp = static_cast<uint8_t>(reduceOp);

    // 【核心步骤 5】插入 REDUCE 任务
    uint32_t taskIndex = 0u;
    auto insertRet = InsertLevel1Task(&taskMetaData, &taskIndex);
    if (insertRet != HcclSim::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("{}: InsertTaskToCollection failed, ret={:d}", __func__, static_cast<int>(insertRet));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, channel={:d}, "
        "remoteRank={:d}, taskIndex={:d}",
        __func__, thread, channel, remoteRankId, taskIndex);
    return 0;
}

/**
 * @brief 通过 channel 从远端内存读数据并做 reduce（数据面接口）。
 *
 * 从远端 src 读取 count * sizeof(dataType) 字节的数据，与本端 dst 所指向
 * 的相同长度的内存数据做 reduceOp 操作，结果输出到本端 dst 中。
 * 接口调用方为 dst 所在节点（本端是 local/dst，远端是 remote/src）。
 *
 * @param thread   输入：线程句柄。
 * @param channel  输入：通道句柄。
 * @param dst      输入/输出：本地 device 内存地址。
 * @param src      输入：远端 device 内存地址。
 * @param count    输入：元素数量（非字节数）。
 * @param dataType 输入：元素数据类型。
 * @param reduceOp 输入：reduce 操作类型。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommReadReduceOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    if (dst == nullptr || src == nullptr) {
        HCCL_VM_ERROR("{}: dst or src is nullptr", __func__);
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    if (count == 0u) {
        HCCL_VM_ERROR("{}: count is 0", __func__);
        return static_cast<int32_t>(HCCL_E_PARA);
    }

    HCCL_VM_INFO(
        "{}: thread={:d}, channel={:d}, dst={:p}, src={:p}, "
        "count={:d}, dataType={:d}, reduceOp={:d}",
        __func__, thread, channel, dst, src, count, static_cast<int>(dataType), static_cast<int>(reduceOp));

    // 【核心步骤 1】获取当前 rank ID
    uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());

    // 【核心步骤 2】查询线程的 commId 和 streamId
    uint64_t commId = 0u;
    uint64_t streamId = 0u;
    auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(thread);
    if (optThread.ok()) {
        commId = optThread->commId;
        streamId = optThread->streamId;
    } else if (!ResolveDpuFallbackCtx(commId, streamId)) { // DPU 回调期（thread=0）经 TLS 回退归属
        HCCL_VM_WARN("{}: thread {:d} not found", __func__, thread);
    }

    // 【核心步骤 3】查询通道的远端 rank ID
    uint32_t remoteRankId = 0u;
    auto optChannel = sim::runtime::Db::GetById<sim::runtime::HcclChannel>(channel);
    if (optChannel.ok()) {
        remoteRankId = static_cast<uint32_t>(optChannel->remoteRankId);
    } else {
        HCCL_VM_WARN("{}: channel {:d} not found", __func__, channel);
    }

    // 【核心步骤 4】构建 REDUCE 任务元数据
    // ReadReduce 操作语义（从调用方视角，与 WriteReduce 相反）：
    //   - 调用方是 dst 所在节点（local = dst）
    //   - src 是远端节点（remote = src）
    //   - 从远端读取数据并在本地执行 reduce：local_data ⊕ remote_data → local_data
    uint32_t typeSize = 0;
    sim::GetDataTypeSize(static_cast<HcclDataType>(dataType), typeSize);
    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::REDUCE;
    taskMetaData.commId = commId;
    taskMetaData.rankId = curRank;
    taskMetaData.streamId = streamId;

    // Read: 远端是 src，本地是 dst
    taskMetaData.taskData.reduce.srcDeviceId = remoteRankId;
    taskMetaData.taskData.reduce.srcOffset = reinterpret_cast<uint64_t>(src);
    taskMetaData.taskData.reduce.dstDeviceId = curRank;
    taskMetaData.taskData.reduce.dstOffset = reinterpret_cast<uint64_t>(dst);
    taskMetaData.taskData.reduce.dataCount = count * typeSize;
    taskMetaData.taskData.reduce.dataType = static_cast<uint8_t>(dataType);
    taskMetaData.taskData.reduce.reduceOp = static_cast<uint8_t>(reduceOp);

    // 【核心步骤 5】插入 REDUCE 任务
    uint32_t taskIndex = 0u;
    auto insertRet = InsertLevel1Task(&taskMetaData, &taskIndex);
    if (insertRet != HcclSim::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("{}: InsertTaskToCollection failed, ret={:d}", __func__, static_cast<int>(insertRet));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, channel={:d}, "
        "remoteRank={:d}, taskIndex={:d}",
        __func__, thread, channel, remoteRankId, taskIndex);
    return 0;
}

/**
 * @brief 通过 channel 向远端发送 notify 信号（数据面接口）。
 *
 * 在指定 channel 上向远端的 remoteNotifyIdx 索引位置发送通知信号。
 * 与 HcommThreadNotifyRecordOnThread 的区别：本接口通过 channel 发送到远端节点，
 * 而不是在本地的线程间发送。
 *
 * @param thread         输入：发起 notify 的线程句柄。
 * @param channel        输入：通道句柄。
 * @param remoteNotifyIdx 输入：远端节点上的 notify 索引。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    HCCL_VM_INFO("{}: thread={:d}, channel={:d}, remoteNotifyIdx={:d}", __func__, thread, channel, remoteNotifyIdx);

    // 【核心步骤 1】获取当前 rank ID（通知发送方）
    uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());

    // 【核心步骤 2】查询线程的 commId 和 streamId
    uint64_t commId = 0u;
    uint64_t streamId = 0u;
    auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(thread);
    if (optThread.ok()) {
        commId = optThread->commId;
        streamId = optThread->streamId;
    } else if (!ResolveDpuFallbackCtx(commId, streamId)) { // DPU 回调期（thread=0）经 TLS 回退归属
        HCCL_VM_WARN("{}: thread {:d} not found", __func__, thread);
    }

    // 【核心步骤 3】查询通道的远端 rank ID（通知接收方）
    uint32_t remoteRankId = 0u;
    auto optChannel = sim::runtime::Db::GetById<sim::runtime::HcclChannel>(channel);
    if (optChannel.ok()) {
        remoteRankId = static_cast<uint32_t>(optChannel->remoteRankId);
    } else {
        HCCL_VM_WARN("{}: channel {:d} not found", __func__, channel);
    }

    // 【核心步骤 4】构建 NOTIFY_RECORD 任务元数据
    // Channel notify 语义（与 Thread Notify 的关键差异）：
    //   - ThreadNotifyRecord: dstRankId=本端（同一节点内，目标线程）
    //   - ChannelNotifyRecord: dstRankId=远端（跨节点通信，目标节点）
    // notifyId 取本通道 notifyId 数组中 remoteNotifyIdx 对应的真实 Notify 记录 id；
    // DPU 上下文（HostDPU）取接收方反向通道槽位（ResolveRecordNotifyId 内置退化策略）；
    // 若通道不存在或索引越界，退化为通道句柄，仍可记录任务（仅告警）
    const uint64_t notifyIdVal = ResolveRecordNotifyId(
        optChannel.ok() ? &(*optChannel) : nullptr, channel, remoteNotifyIdx, curRank, remoteRankId, __func__);

    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    taskMetaData.commId = commId;
    taskMetaData.rankId = curRank; // 操作发起方（本端）
    taskMetaData.streamId = streamId;

    // notifyId = 通道真实 Notify id；notifyCount = 远端的 notify 槽位索引（目标节点的接收位）
    // srcRankId = 本端（发送方），dstRankId = 远端（接收方）
    const uint64_t notifyTaskId = MakeDpuAwareNotifyTaskId(remoteRankId, notifyIdVal);
    taskMetaData.taskData.notify.notifyId = notifyTaskId;
    taskMetaData.taskData.notify.notifyCount = static_cast<uint8_t>(remoteNotifyIdx);
    taskMetaData.taskData.notify.srcDeviceId = curRank;
    taskMetaData.taskData.notify.dstDeviceId = remoteRankId;

    // 【核心步骤 5】插入任务到集合
    uint32_t taskIndex = 0u;
    auto insertRet = InsertLevel1Task(&taskMetaData, &taskIndex);
    if (insertRet != HcclSim::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("{}: InsertTaskToCollection failed, ret={:d}", __func__, static_cast<int>(insertRet));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, channel={:d}, "
        "remoteRank={:d}, remoteNotifyIdx={:d}, taskIndex={:d}",
        __func__, thread, channel, remoteRankId, remoteNotifyIdx, taskIndex);
    return 0;
}

/**
 * @brief 通过 channel 向远端写数据并发送 notify（数据面接口）。
 *
 * 将 src 中 len 字节的数据通过 channel 写入远端 dst，写入完成后向远端的
 * remoteNotifyIdx 索引位置发送通知信号。这是一个原子操作，保证数据写入
 * 和 notify 发送的顺序性。
 *
 * @param thread          输入：线程句柄。
 * @param channel         输入：通道句柄。
 * @param dst             输入：远端 device 内存地址。
 * @param src             输入：本地 device 内存地址。
 * @param len             输入：待拷贝字节数。
 * @param remoteNotifyIdx 输入：远端节点上的 notify 索引。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommWriteWithNotifyOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    if (dst == nullptr || src == nullptr) {
        HCCL_VM_ERROR("{}: dst or src is nullptr", __func__);
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    if (len == 0u) {
        HCCL_VM_ERROR("{}: len is 0", __func__);
        return static_cast<int32_t>(HCCL_E_PARA);
    }

    HCCL_VM_INFO(
        "{}: thread={:d}, channel={:d}, dst={:p}, src={:p}, "
        "len={:d}, remoteNotifyIdx={:d}",
        __func__, thread, channel, dst, src, len, remoteNotifyIdx);

    // 【核心步骤 1】获取当前 rank ID 和线程上下文
    uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());

    uint64_t commId = 0u;
    uint64_t streamId = 0u;
    auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(thread);
    if (optThread.ok()) {
        commId = optThread->commId;
        streamId = optThread->streamId;
    } else if (!ResolveDpuFallbackCtx(commId, streamId)) { // DPU 回调期（thread=0）经 TLS 回退归属
        HCCL_VM_WARN("{}: thread {:d} not found", __func__, thread);
    }

    // 【核心步骤 2】查询通道的远端 rank ID
    uint32_t remoteRankId = 0u;
    auto optChannel = sim::runtime::Db::GetById<sim::runtime::HcclChannel>(channel);
    if (optChannel.ok()) {
        remoteRankId = static_cast<uint32_t>(optChannel->remoteRankId);
    } else {
        HCCL_VM_WARN("{}: channel {:d} not found", __func__, channel);
    }

    // 【核心步骤 3】构建并记录 MEM_CPY 任务（第一阶段）
    // 语义：将本地 src 数据写入远端 dst
    HcclTaskMetaData memTask{};
    memTask.taskType = HccLTaskMetaType::MEM_CPY;
    memTask.commId = commId;
    memTask.rankId = curRank;
    memTask.streamId = streamId;

    // Write 方向：本地 → 远端
    memTask.taskData.transMem.srcDeviceId = curRank;
    memTask.taskData.transMem.srcOffset = reinterpret_cast<uint64_t>(src);
    memTask.taskData.transMem.dstDeviceId = remoteRankId;
    memTask.taskData.transMem.dstOffset = reinterpret_cast<uint64_t>(dst);
    memTask.taskData.transMem.len = len;

    uint32_t memTaskIndex = 0u;
    auto insertRet = InsertLevel1Task(&memTask, &memTaskIndex);
    if (insertRet != HcclSim::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("{}: InsertTaskToCollection (MEM_CPY) failed, ret={:d}", __func__, static_cast<int>(insertRet));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    // 【核心步骤 4】构建并记录 NOTIFY_RECORD 任务（第二阶段）
    // 语义：数据写入完成后，向远端发送通知信号
    // notifyId 取本通道 notifyId 数组中 remoteNotifyIdx 对应的真实 Notify 记录 id；
    // DPU 上下文（HostDPU）取接收方反向通道槽位（ResolveRecordNotifyId 内置退化策略）
    // srcRankId/dstRankId 标识通知的发送方和接收方
    const uint64_t notifyIdVal = ResolveRecordNotifyId(
        optChannel.ok() ? &(*optChannel) : nullptr, channel, remoteNotifyIdx, curRank, remoteRankId, __func__);

    HcclTaskMetaData notifyTask{};
    notifyTask.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    notifyTask.commId = commId;
    notifyTask.rankId = curRank;
    notifyTask.streamId = streamId;

    notifyTask.taskData.notify.notifyId = MakeDpuAwareNotifyTaskId(remoteRankId, notifyIdVal); // 通道真实 Notify id
    notifyTask.taskData.notify.notifyCount = static_cast<uint8_t>(remoteNotifyIdx); // 通知计数值（索引）
    notifyTask.taskData.notify.srcDeviceId = curRank;                               // 发送方：本地
    notifyTask.taskData.notify.dstDeviceId = remoteRankId;                          // 接收方：远端

    uint32_t notifyTaskIndex = 0u;
    insertRet = InsertLevel1Task(&notifyTask, &notifyTaskIndex);
    if (insertRet != HcclSim::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: InsertTaskToCollection (NOTIFY_RECORD) failed, ret={:d}", __func__, static_cast<int>(insertRet));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, channel={:d}, remoteRank={:d}, memTaskIndex={:d}, notifyTaskIndex={:d}", __func__,
        thread, channel, remoteRankId, memTaskIndex, notifyTaskIndex);
    return 0;
}

/**
 * @brief 通过 channel 向远端写数据并做 reduce，然后发送 notify（数据面接口）。
 *
 * 将 src 中 count * sizeof(dataType) 字节的数据通过 channel 与远端 dst
 * 所指向的相同长度的内存数据进行 reduceOp 操作，结果输出到远端 dst 中，
 * 操作完成后向远端的 remoteNotifyIdx 索引位置发送通知信号。
 * 这是一个原子操作，保证 reduce 和 notify 发送的顺序性。
 *
 * @param thread          输入：线程句柄。
 * @param channel         输入：通道句柄。
 * @param dst             输入：远端 device 内存地址。
 * @param src             输入：本地 device 内存地址。
 * @param count           输入：元素数量（非字节数）。
 * @param dataType        输入：元素数据类型。
 * @param reduceOp        输入：reduce 操作类型。
 * @param remoteNotifyIdx 输入：远端节点上的 notify 索引。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommWriteReduceWithNotifyOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp, uint32_t remoteNotifyIdx)
{
    if (dst == nullptr || src == nullptr) {
        HCCL_VM_ERROR("{}: dst or src is nullptr", __func__);
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    if (count == 0u) {
        HCCL_VM_ERROR("{}: count is 0", __func__);
        return static_cast<int32_t>(HCCL_E_PARA);
    }

    HCCL_VM_INFO(
        "{}: thread={:d}, channel={:d}, dst={:p}, src={:p}, "
        "count={:d}, dataType={:d}, reduceOp={:d}, remoteNotifyIdx={:d}",
        __func__, thread, channel, dst, src, count, static_cast<int>(dataType), static_cast<int>(reduceOp),
        remoteNotifyIdx);

    // 【核心步骤 1】获取当前 rank ID 和线程上下文
    uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());

    uint64_t commId = 0u;
    uint64_t streamId = 0u;
    auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(thread);
    if (optThread.ok()) {
        commId = optThread->commId;
        streamId = optThread->streamId;
    } else if (!ResolveDpuFallbackCtx(commId, streamId)) { // DPU 回调期（thread=0）经 TLS 回退归属
        HCCL_VM_WARN("{}: thread {:d} not found", __func__, thread);
    }

    // 【核心步骤 2】查询通道的远端 rank ID
    uint32_t remoteRankId = 0u;
    auto optChannel = sim::runtime::Db::GetById<sim::runtime::HcclChannel>(channel);
    if (optChannel.ok()) {
        remoteRankId = static_cast<uint32_t>(optChannel->remoteRankId);
    } else {
        HCCL_VM_WARN("{}: channel {:d} not found", __func__, channel);
    }

    // 第一步：记录 REDUCE 任务
    uint32_t typeSize = 0;
    sim::GetDataTypeSize(static_cast<HcclDataType>(dataType), typeSize);
    HcclTaskMetaData reduceTask{};
    reduceTask.taskType = HccLTaskMetaType::REDUCE;
    reduceTask.commId = commId;
    reduceTask.rankId = curRank;
    reduceTask.streamId = streamId;

    // Write: 本端是 src，远端是 dst
    reduceTask.taskData.reduce.srcDeviceId = curRank;
    reduceTask.taskData.reduce.srcOffset = reinterpret_cast<uint64_t>(src);
    reduceTask.taskData.reduce.dstDeviceId = remoteRankId;
    reduceTask.taskData.reduce.dstOffset = reinterpret_cast<uint64_t>(dst);
    reduceTask.taskData.reduce.dataCount = count * typeSize;
    reduceTask.taskData.reduce.dataType = static_cast<uint8_t>(dataType);
    reduceTask.taskData.reduce.reduceOp = static_cast<uint8_t>(reduceOp);

    uint32_t reduceTaskIndex = 0u;
    auto insertRet = InsertLevel1Task(&reduceTask, &reduceTaskIndex);
    if (insertRet != HcclSim::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("{}: InsertTaskToCollection (REDUCE) failed, ret={:d}", __func__, static_cast<int>(insertRet));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    // 【核心步骤 4】构建并记录 NOTIFY_RECORD 任务（第二阶段）
    // 语义：Reduce 操作完成后，向远端发送通知信号
    // notifyId 取本通道 notifyId 数组中 remoteNotifyIdx 对应的真实 Notify 记录 id
    // srcRankId/dstRankId 标识通知的发送方和接收方
    uint64_t notifyIdVal = static_cast<uint64_t>(channel);
    if (optChannel.ok()) {
        if (remoteNotifyIdx < optChannel->notifyNum) {
            notifyIdVal = optChannel->notifyId[remoteNotifyIdx];
        } else {
            HCCL_VM_WARN(
                "{}: remoteNotifyIdx {:d} out of range (notifyNum={:d})", __func__, remoteNotifyIdx,
                optChannel->notifyNum);
        }
    }

    HcclTaskMetaData notifyTask{};
    notifyTask.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    notifyTask.commId = commId;
    notifyTask.rankId = curRank;
    notifyTask.streamId = streamId;

    notifyTask.taskData.notify.notifyId = MakeDpuAwareNotifyTaskId(remoteRankId, notifyIdVal); // 通道真实 Notify id
    notifyTask.taskData.notify.notifyCount = static_cast<uint8_t>(remoteNotifyIdx); // 通知计数值（索引）
    notifyTask.taskData.notify.srcDeviceId = curRank;                               // 发送方：本地
    notifyTask.taskData.notify.dstDeviceId = remoteRankId;                          // 接收方：远端

    uint32_t notifyTaskIndex = 0u;
    insertRet = InsertLevel1Task(&notifyTask, &notifyTaskIndex);
    if (insertRet != HcclSim::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: InsertTaskToCollection (NOTIFY_RECORD) failed, ret={:d}", __func__, static_cast<int>(insertRet));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO("{} success, reduceTaskId={:d}, notifyTaskId={:d}", __func__, reduceTaskIndex, notifyTaskIndex);
    return 0;
}

/**
 * @brief 通过 channel 从远端内存读取数据（数据面接口）。
 *
 * 从远端 src 中读取 len 字节的数据，通过 channel 读取到本端 dst 所指向的内存中。
 * 接口调用方为 dst 所在节点（本端是 local/dst，远端是 remote/src）。
 *
 * 本桩以 MEM_CPY 任务记录到 TaskCollection 与 DB，供 Checker 做冲突分析。
 * 不做实际的跨节点数据传输。
 *
 * @param thread  输入：线程句柄。
 * @param channel 输入：通道句柄，由 HcommChannelCreate / HcclChannelAcquire 创建。
 * @param dst     输入：本地 device 内存地址（本端）。
 * @param src     输入：远端 device 内存地址（对端）。
 * @param len     输入：待读取字节数。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    if (dst == nullptr || src == nullptr) {
        HCCL_VM_ERROR("{}: dst or src is nullptr", __func__);
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    if (len == 0u) {
        HCCL_VM_ERROR("{}: len is 0", __func__);
        return static_cast<int32_t>(HCCL_E_PARA);
    }

    HCCL_VM_INFO(
        "{}: thread={:d}, channel={:d}, dst={:p}, src={:p}, len={:d}", __func__, thread, channel, dst, src, len);

    // 【核心步骤 1】获取当前 rank ID
    uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());

    // 【核心步骤 2】查询线程的 commId 和 streamId
    uint64_t commId = 0u;
    uint64_t streamId = 0u;
    auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(thread);
    if (optThread.ok()) {
        commId = optThread->commId;
        streamId = optThread->streamId;
    } else if (!ResolveDpuFallbackCtx(commId, streamId)) { // DPU 回调期（thread=0）经 TLS 回退归属
        HCCL_VM_WARN("{}: thread {:d} not found", __func__, thread);
    }

    // 【核心步骤 3】查询通道的远端 rank ID
    uint32_t remoteRankId = 0u;
    auto optChannel = sim::runtime::Db::GetById<sim::runtime::HcclChannel>(channel);
    if (optChannel.ok()) {
        remoteRankId = static_cast<uint32_t>(optChannel->remoteRankId);
    } else {
        HCCL_VM_WARN("{}: channel {:d} not found", __func__, channel);
    }

    // 【核心步骤 4】构建 MEM_CPY 任务元数据
    // Read 操作语义（从调用方视角，与 Write 相反）：
    //   - 调用方是 dst 所在节点（local = dst）
    //   - src 是远端节点（remote = src）
    //   - 数据流方向：远端 src → 本端 dst
    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::MEM_CPY;
    taskMetaData.commId = commId;
    taskMetaData.rankId = curRank; // 操作发起方（本端 = dst 侧）
    taskMetaData.streamId = streamId;

    // Read 语义：srcRankId=远端，dstRankId=本端
    taskMetaData.taskData.transMem.srcDeviceId = remoteRankId;
    taskMetaData.taskData.transMem.srcOffset = reinterpret_cast<uint64_t>(src);
    taskMetaData.taskData.transMem.dstDeviceId = curRank;
    taskMetaData.taskData.transMem.dstOffset = reinterpret_cast<uint64_t>(dst);
    taskMetaData.taskData.transMem.len = len;

    // 【核心步骤 5】插入任务到集合
    uint32_t taskIndex = 0u;
    auto insertRet = InsertLevel1Task(&taskMetaData, &taskIndex);
    if (insertRet != HcclSim::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("{}: InsertTaskToCollection failed, ret={:d}", __func__, static_cast<int>(insertRet));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, channel={:d}, "
        "remoteRank={:d}, taskIndex={:d}",
        __func__, thread, channel, remoteRankId, taskIndex);
    return 0;
}

/**
 * @brief 通过 channel 等待远端发送的 notify 信号（数据面接口）。
 *
 * 在指定 channel 上等待来自远端的 localNotifyIdx 索引位置的通知信号。
 * 与 HcommThreadNotifyWaitOnThread 的区别：本接口等待的是远端通过 channel 发送
 * 的 notify 信号，而不是本地线程间的 notify 信号。
 *
 * @param thread        输入：等待 notify 的线程句柄。
 * @param channel       输入：通道句柄。
 * @param localNotifyIdx 输入：本端节点上的 notify 索引（被远端 record 的目标索引）。
 * @param timeout       输入：超时时间（秒），0 表示永久等待。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t
HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout)
{
    HCCL_VM_INFO(
        "{}: thread={:d}, channel={:d}, "
        "localNotifyIdx={:d}, timeout={:d}",
        __func__, thread, channel, localNotifyIdx, timeout);

    // 【核心步骤 1】获取当前 rank ID（等待方/接收方）
    uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());

    // 【核心步骤 2】查询线程的 commId 和 streamId
    uint64_t commId = 0u;
    uint64_t streamId = 0u;
    auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(thread);
    if (optThread.ok()) {
        commId = optThread->commId;
        streamId = optThread->streamId;
    } else if (!ResolveDpuFallbackCtx(commId, streamId)) { // DPU 回调期（thread=0）经 TLS 回退归属
        HCCL_VM_WARN("{}: thread {:d} not found", __func__, thread);
    }

    // 【核心步骤 3】查询通道的远端 rank ID（信号发送方）
    // ChannelNotifyWait 等待的是远端通过该 channel 发来的 notify 信号，
    // 因此 remoteRankId 对应远端的 srcRankId（信号发送方）。
    uint32_t remoteRankId = 0u;
    auto optChannel = sim::runtime::Db::GetById<sim::runtime::HcclChannel>(channel);
    if (optChannel.ok()) {
        remoteRankId = static_cast<uint32_t>(optChannel->remoteRankId);
    } else {
        HCCL_VM_WARN("{}: channel {:d} not found", __func__, channel);
    }

    // 【核心步骤 4】构建 NOTIFY_WAIT 任务元数据
    // Channel notify wait 语义（与 Thread NotifyWait 的关键差异）：
    //   - ThreadNotifyWait: srcRankId=本端（同一节点内，信号发送者也在本节点）
    //   - ChannelNotifyWait: srcRankId=远端（跨节点通信，信号来自远端节点）
    // notifyId 取本通道 notifyId 数组中 localNotifyIdx 对应的真实 Notify 记录 id；
    // 若通道不存在或索引越界，退化为通道句柄，仍可记录任务（仅告警）
    uint64_t notifyIdVal = static_cast<uint64_t>(channel);
    if (optChannel.ok()) {
        if (localNotifyIdx < optChannel->notifyNum) {
            notifyIdVal = optChannel->notifyId[localNotifyIdx];
        } else {
            HCCL_VM_WARN(
                "{}: localNotifyIdx {:d} out of range (notifyNum={:d})", __func__, localNotifyIdx,
                optChannel->notifyNum);
        }
    }

    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    taskMetaData.commId = commId;
    taskMetaData.rankId = curRank; // 操作发起方（本端 = 等待方）
    taskMetaData.streamId = streamId;

    // notifyId = 通道真实 Notify id；notifyCount = 本端 notify 槽位索引（被 record 的目标位）
    // srcRankId = 远端（信号发送方），dstRankId = 本端（等待方）
    const uint64_t notifyTaskId = MakeDpuAwareNotifyTaskId(curRank, notifyIdVal);
    taskMetaData.taskData.notify.notifyId = notifyTaskId;
    taskMetaData.taskData.notify.notifyCount = static_cast<uint8_t>(localNotifyIdx);
    taskMetaData.taskData.notify.srcDeviceId = remoteRankId;
    taskMetaData.taskData.notify.dstDeviceId = curRank;

    // 【核心步骤 5】插入任务到集合
    uint32_t taskIndex = 0u;
    auto insertRet = InsertLevel1Task(&taskMetaData, &taskIndex);
    if (insertRet != HcclSim::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("{}: InsertTaskToCollection failed, ret={:d}", __func__, static_cast<int>(insertRet));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, channel={:d}, "
        "remoteRank={:d}, localNotifyIdx={:d}, taskIndex={:d}",
        __func__, thread, channel, remoteRankId, localNotifyIdx, taskIndex);
    return 0;
}

#ifdef __cplusplus
}
#endif
