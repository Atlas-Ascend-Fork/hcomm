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

#include <cstdint>
#include <cstring>

#include "hccl_proxy_common.h"
#include "level1_proxy_common.h"
#include "sim_log.h"
#include "sim_common_defs.h"
#include "sim_capacity_limits.h"
#include "runtime_state/sim_models.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "store_sim_store_pub.h"

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

HcclSim::HcclVmResult InsertLevel1Task(HcclTaskMetaData* task, uint32_t* index)
{
    if (!NormalizeLevel1Task(task)) {
        return HcclSim::HCCL_SIM_E_INTERNAL;
    }
    return InsertTaskToCollection(task, index);
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
    } else {
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
    } else {
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
    } else {
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
    } else {
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
    // 若通道不存在或索引越界，退化为通道句柄，仍可记录任务（仅告警）
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

    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    taskMetaData.commId = commId;
    taskMetaData.rankId = curRank; // 操作发起方（本端）
    taskMetaData.streamId = streamId;

    // notifyId = 通道真实 Notify id；notifyCount = 远端的 notify 槽位索引（目标节点的接收位）
    // srcRankId = 本端（发送方），dstRankId = 远端（接收方）
    const uint64_t notifyTaskId = HcclSim::MakeNotifyTaskId(remoteRankId, notifyIdVal);
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
    } else {
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

    notifyTask.taskData.notify.notifyId = HcclSim::MakeNotifyTaskId(remoteRankId, notifyIdVal); // 通道真实 Notify id
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
    } else {
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

    notifyTask.taskData.notify.notifyId = HcclSim::MakeNotifyTaskId(remoteRankId, notifyIdVal); // 通道真实 Notify id
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
    } else {
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
    } else {
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
    const uint64_t notifyTaskId = HcclSim::MakeNotifyTaskId(curRank, notifyIdVal);
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
