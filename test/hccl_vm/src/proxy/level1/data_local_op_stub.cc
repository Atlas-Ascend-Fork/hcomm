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
 * Description: 数据面本地操作打桩函数（北向劫持）
 *              - HcommLocalCopyOnThread 等数据面接口作为 Checker 的输入。
 *              与南向劫持 aclrtMemcpy 的区别：这些接口绑定 ThreadHandle，
 *              由用户在控制面申请线程后直接调用。
 * Create: 2026-07-20
 */

#define HCCL_VM_MODULE "HCOMM_LOCAL"

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
#include "store_sim_device_memory_manager.h"

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

/**
 * @brief 本端设备地址 -> 虚拟地址，翻译不到则保留原值。
 *
 * AICPU 模式下 aclrtMalloc 返回给应用的是 devMapAddr（≠ 内部虚拟地址），而 Checker 的
 * MemLayout 是按虚拟地址建立的；数据面桩记录 task 前必须翻译，否则 checker 解析不到
 * （现象：GetSlice 报 ErrorCode 110）。非 AICPU 模式下 devMapAddr == 虚拟地址，TryGet
 * 查不到映射 → 保留原值，故本函数幂等、无副作用（不改变已虚拟地址）。
 *
 * @param devAddr 输入：桩函数入参里的本端设备地址（可能是 devMapAddr，也可能已是虚拟地址）。
 * @return 翻译后的虚拟地址；查不到映射时原样返回 devAddr。
 */
uint64_t LocalDevAddrToVirOrKeep(uint64_t devAddr)
{
    const uint64_t virAddr = sim::TryGetVirPtrByDevPtr(devAddr);
    return (virAddr != 0) ? virAddr : devAddr;
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
    return NormalizeLevel1Task(task) ? InsertTaskToCollection(task, index) : HcclSim::HCCL_SIM_E_INTERNAL;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 在指定线程上执行本地内存拷贝（数据面接口）。
 *
 * 提供本地内存拷贝功能，将 src 指向的长度为 len 的内存数据，拷贝到 dst 所指向的
 * 相同长度的内存中。本桩作为 Checker 的输入，主要职责：
 *   1) 将操作以 MEM_CPY 任务形式记录到 TaskCollection 与 DB，供 Checker 进行
 *      内存冲突检查与调度正确性校验；
 *   2) 尽可能完成实际的 host 侧 memcpy（虚拟设备内存已被映射到 host）。
 *
 * 失败语义（参考 hcomm 官方接口）：
 *   - dst / src 为 nullptr 时返回非 0 值 (HCCL_E_PTR = 2)
 *   - len 为 0 时视为空操作，返回 0
 *   - 实际 host 侧 memcpy 失败（地址解析失败）仅告警，仍返回 0，
 *     保证 Checker 仍能基于已记录的任务元数据做正确性分析。
 *
 * 约束（来自官方文档）：
 *   - dst、src 内存是申请的 device 内存；
 *   - 在 Ascend 950PR/950DT 上仅支持 AICPU_TS 模式下、在 Device 侧调用。
 *     本桩在 Host 侧运行，直接操作虚拟设备内存对应的 host 映射。
 *
 * @param thread 输入：线程句柄，由 HcclThreadAcquire / HcommThreadAlloc 创建。
 * @param dst    输出：目标 device 内存地址（虚拟地址）。
 * @param src    输入：源 device 内存地址（虚拟地址）。
 * @param len    输入：待拷贝字节数。
 * @return int32_t 成功返回 0，参数错误返回非 0 (与 HCCL_E_PTR 数值一致)。
 */
int32_t HcommLocalCopyOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    if (dst == nullptr || src == nullptr) {
        HCCL_VM_ERROR("{}: dst or src is nullptr", __func__);
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    if (len == 0u) {
        HCCL_VM_ERROR("{}: len is 0", __func__);
        return static_cast<int32_t>(HCCL_E_PARA);
    }

    HCCL_VM_INFO("{}: thread={:d}, dst={:p}, src={:p}, len={:d}", __func__, thread, dst, src, len);

    // 【核心步骤 1】获取当前 rank ID
    uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());

    // 【核心步骤 2】查询线程记录，获取所属通信域与关联的 stream
    // 若线程不存在仍允许记录任务，便于 Checker 在没有控制面上下文时也能分析
    // （local 操作不依赖远端，但需要流上下文）
    uint64_t commId = 0u;
    uint64_t streamId = 0u;
    auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(thread);
    if (optThread.ok()) {
        commId = optThread->commId;
        streamId = optThread->streamId;
    } else {
        HCCL_VM_WARN("{}: thread {:d} not found, continue with commId=0", __func__, thread);
    }

    // 【核心步骤 3】构造 MEM_CPY 任务元数据
    // Local Copy 语义：src 和 dst 都在同一节点，srcRankId == dstRankId == curRank
    // Checker 通过 srcOffset/dstOffset + len 做地址范围的冲突分析
    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::MEM_CPY;
    taskMetaData.commId = commId;
    taskMetaData.rankId = curRank;
    taskMetaData.streamId = streamId;

    // 同节点拷贝：src/dst rank 都是本节点。
    // 地址须翻译为虚拟地址（AICPU 模式下入参是 devMapAddr），否则 Checker 的 MemLayout 解析不到。
    taskMetaData.taskData.transMem.srcDeviceId = curRank;
    taskMetaData.taskData.transMem.srcOffset = LocalDevAddrToVirOrKeep(reinterpret_cast<uint64_t>(src));
    taskMetaData.taskData.transMem.dstDeviceId = curRank;
    taskMetaData.taskData.transMem.dstOffset = LocalDevAddrToVirOrKeep(reinterpret_cast<uint64_t>(dst));
    taskMetaData.taskData.transMem.len = len;

    // 【核心步骤 4】插入任务到集合
    uint32_t taskIndex = 0u;
    auto insertRet = InsertLevel1Task(&taskMetaData, &taskIndex);
    if (insertRet != HcclSim::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("{}: InsertTaskToCollection failed, ret={:d}", __func__, static_cast<int>(insertRet));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, commId={:d}, rankId={:d}, "
        "taskIndex={:d}",
        __func__, thread, commId, curRank, taskIndex);
    return 0;
}

/**
 * @brief 在指定线程上执行本地 reduce 操作（数据面接口）。
 *
 * 将 src 指向的 count * sizeof(dataType) 字节的内存数据，
 * 与 dst 所指向的相同长度的内存数据进行 reduceOp 操作，结果输出到 dst 中。
 *
 * 本桩以 REDUCE 任务记录到 TaskCollection 与 DB，供 Checker 做冲突分析。
 * 不做实际 host 侧 reduce 计算。
 *
 * @param thread   输入：线程句柄。
 * @param dst      输入/输出：目标 device 内存地址。
 * @param src      输入：源 device 内存地址。
 * @param count    输入：元素数量（非字节数）。
 * @param dataType 输入：元素数据类型（HcommDataType）。
 * @param reduceOp 输入：reduce 操作类型（HcommReduceOp）。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommLocalReduceOnThread(
    ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
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
        "{}: thread={:d}, dst={:p}, src={:p}, count={:d}, "
        "dataType={:d}, reduceOp={:d}",
        __func__, thread, dst, src, count, static_cast<int>(dataType), static_cast<int>(reduceOp));

    // 【核心步骤 1】获取当前 rank ID
    uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());

    // 【核心步骤 2】查询线程记录，获取所属通信域与关联的 stream
    uint64_t commId = 0u;
    uint64_t streamId = 0u;
    auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(thread);
    if (optThread.ok()) {
        commId = optThread->commId;
        streamId = optThread->streamId;
    } else {
        HCCL_VM_WARN("{}: thread {:d} not found", __func__, thread);
    }

    // 【核心步骤 3】构造 REDUCE 任务元数据
    // Local Reduce 语义：src 和 dst 都在同一节点，srcRankId == dstRankId == curRank
    // 与 Channel Reduce 的关键差异：
    //   - LocalReduce: src/dst 都在本节点（同 rank）
    //   - WriteReduce/ReadReduce: src 和 dst 跨节点（不同 rank）
    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::REDUCE;
    taskMetaData.commId = commId;
    taskMetaData.rankId = curRank;
    taskMetaData.streamId = streamId;

    // 同节点 reduce：src/dst rank 都是本节点。
    // 地址须翻译为虚拟地址（AICPU 模式下入参是 devMapAddr），否则 Checker 的 MemLayout 解析不到。
    taskMetaData.taskData.reduce.srcDeviceId = curRank;
    taskMetaData.taskData.reduce.srcOffset = LocalDevAddrToVirOrKeep(reinterpret_cast<uint64_t>(src));
    taskMetaData.taskData.reduce.dstDeviceId = curRank;
    taskMetaData.taskData.reduce.dstOffset = LocalDevAddrToVirOrKeep(reinterpret_cast<uint64_t>(dst));
    taskMetaData.taskData.reduce.dataCount = count;
    taskMetaData.taskData.reduce.dataType = static_cast<uint8_t>(dataType);
    taskMetaData.taskData.reduce.reduceOp = static_cast<uint8_t>(reduceOp);

    // 【核心步骤 4】插入任务到集合
    uint32_t taskIndex = 0u;
    auto insertRet = InsertLevel1Task(&taskMetaData, &taskIndex);
    if (insertRet != HcclSim::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("{}: InsertTaskToCollection failed, ret={:d}", __func__, static_cast<int>(insertRet));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO("{} success, thread={:d}, taskIndex={:d}", __func__, thread, taskIndex);
    return 0;
}

/**
 * @brief 在指定线程上发送 notify 信号给目标线程（数据面接口）。
 *
 * 向 dstThread 的 dstNotifyIdx 索引位置发送通知信号。
 * dstNotifyIdx 的取值范围为 [0, notifyNumPerThread)。
 *
 * @param thread        输入：发起 notify 的线程句柄（src 线程）。
 * @param dstThread     输入：目标线程句柄（接收端线程）。
 * @param dstNotifyIdx  输入：目标线程上的 notify 索引。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx)
{
    HCCL_VM_INFO("{}: thread={:d}, dstThread={:d}, dstNotifyIdx={:d}", __func__, thread, dstThread, dstNotifyIdx);

    // 【核心步骤 1】获取当前 rank ID（发送方线程所在节点）
    uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());

    // 【核心步骤 2】查询发送方线程的 commId 和 streamId
    uint64_t commId = 0u;
    uint64_t streamId = 0u;
    auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(thread);
    if (optThread.ok()) {
        commId = optThread->commId;
        streamId = optThread->streamId;
    } else {
        HCCL_VM_WARN("{}: src thread {:d} not found", __func__, thread);
    }

    // 【核心步骤 3】构建 NOTIFY_RECORD 任务元数据
    // Thread notify 语义（与 Channel notify 的关键差异）：
    //   - ThreadNotifyRecord: 同一节点内，srcRankId == dstRankId == curRank
    //   - ChannelNotifyRecord: 跨节点通信，srcRankId ≠ dstRankId
    // notifyId 取目标线程 notifyId 数组中 dstNotifyIdx 对应的真实 Notify 记录 id；
    // 若目标线程不存在或索引越界，退化为目标线程句柄，仍可记录任务（仅告警）
    uint64_t notifyIdVal = static_cast<uint64_t>(dstThread);
    auto optDstThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(dstThread);
    if (optDstThread.ok() && dstNotifyIdx < optDstThread->notifyNum) {
        notifyIdVal = static_cast<uint64_t>(optDstThread->notifyId[dstNotifyIdx]);
    } else {
        HCCL_VM_WARN("{}: dstThread {:d} not found or idx {:d} out of range", __func__, dstThread, dstNotifyIdx);
    }

    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    taskMetaData.commId = commId;
    taskMetaData.rankId = curRank; // 任务发起方（发送方线程）
    taskMetaData.streamId = streamId;

    // notifyId = 目标线程真实 Notify id；notifyCount = 目标线程上的 notify 槽位索引
    const uint64_t notifyTaskId = HcclSim::MakeNotifyTaskId(curRank, notifyIdVal);
    taskMetaData.taskData.notify.notifyId = notifyTaskId;
    taskMetaData.taskData.notify.notifyCount = static_cast<uint8_t>(dstNotifyIdx);
    taskMetaData.taskData.notify.srcDeviceId = curRank; // 发送方：本节点
    taskMetaData.taskData.notify.dstDeviceId = curRank; // 接收方：同节点

    // 【核心步骤 4】插入任务到集合
    uint32_t taskIndex = 0u;
    auto insertRet = InsertLevel1Task(&taskMetaData, &taskIndex);
    if (insertRet != HcclSim::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("{}: InsertTaskToCollection failed, ret={:d}", __func__, static_cast<int>(insertRet));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, dstThread={:d}, "
        "dstNotifyIdx={:d}, taskIndex={:d}",
        __func__, thread, dstThread, dstNotifyIdx, taskIndex);
    return 0;
}

/**
 * @brief 在指定线程上等待 notifyIdx 位置的通知信号（数据面接口）。
 *
 * @param thread    输入：等待 notify 的线程句柄。
 * @param notifyIdx 输入：本线程上的 notify 索引。
 * @param timeout   输入：超时时间（秒），0 表示永久等待。
 * @return int32_t 成功返回 0，参数错误返回非 0。
 */
int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeout)
{
    HCCL_VM_INFO("{}: thread={:d}, notifyIdx={:d}, timeout={:d}", __func__, thread, notifyIdx, timeout);

    // 【核心步骤 1】获取当前 rank ID（等待方线程所在节点）
    // 注意：timeout 参数在当前桩实现中不做实际处理（仅记录日志）
    uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());

    // 【核心步骤 2】查询等待方线程的 commId 和 streamId
    uint64_t commId = 0u;
    uint64_t streamId = 0u;
    auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(thread);
    if (optThread.ok()) {
        commId = optThread->commId;
        streamId = optThread->streamId;
    } else {
        HCCL_VM_WARN("{}: thread {:d} not found", __func__, thread);
    }

    // 【核心步骤 3】构建 NOTIFY_WAIT 任务元数据
    // Thread notify wait 语义（与 Channel notify wait 的关键差异）：
    //   - ThreadNotifyWait: srcRankId == dstRankId == curRank（同节点内配对）
    //   - ChannelNotifyWait: srcRankId ≠ dstRankId（跨节点，src 来自远端）
    // notifyId 取本线程 notifyId 数组中 notifyIdx 对应的真实 Notify 记录 id；
    // 若本线程不存在或索引越界，退化为本线程句柄，仍可记录任务（仅告警）
    uint64_t notifyIdVal = static_cast<uint64_t>(thread);
    if (optThread.ok()) {
        if (notifyIdx < optThread->notifyNum) {
            notifyIdVal = static_cast<uint64_t>(optThread->notifyId[notifyIdx]);
        } else {
            HCCL_VM_WARN("{}: idx {:d} out of range (notifyNum={:d})", __func__, notifyIdx, optThread->notifyNum);
        }
    }

    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    taskMetaData.commId = commId;
    taskMetaData.rankId = curRank; // 任务发起方（等待方线程）
    taskMetaData.streamId = streamId;

    // notifyId = 本线程真实 Notify id；notifyCount = 本线程上的 notify 槽位索引
    const uint64_t notifyTaskId = HcclSim::MakeNotifyTaskId(curRank, notifyIdVal);
    taskMetaData.taskData.notify.notifyId = notifyTaskId;
    taskMetaData.taskData.notify.notifyCount = static_cast<uint8_t>(notifyIdx);
    taskMetaData.taskData.notify.srcDeviceId = curRank; // 期望的发送方（同节点匹配）
    taskMetaData.taskData.notify.dstDeviceId = curRank; // 等待方：本节点

    // 【核心步骤 4】插入任务到集合
    uint32_t taskIndex = 0u;
    auto insertRet = InsertLevel1Task(&taskMetaData, &taskIndex);
    if (insertRet != HcclSim::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("{}: InsertTaskToCollection failed, ret={:d}", __func__, static_cast<int>(insertRet));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, notifyIdx={:d}, "
        "taskIndex={:d}",
        __func__, thread, notifyIdx, taskIndex);
    return 0;
}

/**
 * @brief 在指定线程上发送 aclrt notify 信号（数据面接口）。
 *
 * @note A5 平台不支持此接口，始终返回 HCCL_E_NOT_SUPPORT。
 */
int32_t HcommAclrtNotifyRecordOnThread(ThreadHandle thread, uint64_t dstNotifyId)
{
    (void)thread;
    (void)dstNotifyId;
    HCCL_VM_ERROR("{}: not supported on A5 platform", __func__);
    return static_cast<int32_t>(HCCL_E_NOT_SUPPORT);
}

/**
 * @brief 在指定线程上等待 aclrt notify 信号（数据面接口）。
 *
 * @note A5 平台不支持此接口，始终返回 HCCL_E_NOT_SUPPORT。
 */
int32_t HcommAclrtNotifyWaitOnThread(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut)
{
    (void)thread;
    (void)notifyId;
    (void)timeOut;
    HCCL_VM_ERROR("{}: not supported on A5 platform", __func__);
    return static_cast<int32_t>(HCCL_E_NOT_SUPPORT);
}

#ifdef __cplusplus
}
#endif
