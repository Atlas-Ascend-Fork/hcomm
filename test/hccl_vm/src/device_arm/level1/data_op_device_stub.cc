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
 * Description: 设备侧数据面操作打桩函数（level1 劫持）
 *              设备侧数据面接口在 Device 侧被真实 AICPU 实现调用，本桩劫持这些调用，
 *              将其转换为 HcclTaskMetaData 任务元数据并写入设备任务表，供 Checker 做冲突分析。
 *              本桩不做真实的数据搬运 / 归约计算 / notify 阻塞。
 *              接口分组：
 *              - HcommAcquireComm / HcommReleaseComm
 *                通信域资源获取与释放（仅校验通信域存在，不维护全局当前通信域）。
 *              - HcommBatchModeStart / HcommBatchModeEnd
 *                批处理模式边界（桩不缓存任务，直接返回成功）。
 *              - HcommThreadNotifyRecordOnThread / HcommThreadNotifyWaitOnThread /
 *                HcommThreadNotifyWaitOnThreadWithDefaultTimeout
 *                本地线程间 notify 记录与等待（src/dst 同 rank）。
 *              - HcommChannelNotifyRecordOnThread / HcommChannelNotifyWaitOnThreadWithDefaultTimeout
 *                跨节点 Channel notify 记录与等待（src/dst 跨 rank）。
 *              - HcommBatchTransferOnThread / HcclHcommBatchTransferOnThread
 *                批量传输（按描述符展开为 MEM_CPY / REDUCE / NOTIFY_RECORD 任务序列）。
 *              - HcommWriteOnThread / HcommReadOnThread
 *                单条跨节点内存写入/读取（Write: src=本端→远端；Read: 远端→本端）。
 *              - HcommWriteReduceOnThread / HcommReadReduceOnThread
 *                单条跨节点归约写入/读取（WriteReduce: src=本端→远端；ReadReduce: 远端→本端）。
 *              - HcommLocalCopyOnThread / HcommLocalReduceOnThread
 *                本地内存拷贝与归约（src/dst 同 rank，不跨节点）。
 *              - HcommAclrtNotifyRecordOnThread / HcommAclrtNotifyWaitOnThread
 *                ACL notify 记录与等待（notifyId 直接由调用方给出，反查远端 rank）。
 *              - HcommThreadResAcquireTimeOut / HcommSetNotifyWaitTimeOut
 *                超时配置（桩仅校验参数，不保存状态）。
 * Create: 2026-07-27
 */

#define HCCL_VM_MODULE "DEV_L1_DATA_OP"

#include <cstdint>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <vector>

#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "storage/table_access.h"
#include "hccl_device_pub.h"
#include "hccl_task_collection.h"
#include "hccl/hccl_types.h"
#include "sim_capacity_limits.h"
#include "sim_log.h"
#include "runtime_state/sim_models.h"

/* ThreadHandle / ChannelHandle 在 CANN hcomm_primitives.h 中定义为 uint64_t，
 * 此处本地定义以避免引入 securec.h / acl_rt.h 等重依赖。 */
typedef uint64_t ThreadHandle;
typedef uint64_t ChannelHandle;

/* HcommDataType / HcommReduceOp 在正式头文件中均为4字节枚举。 */
typedef int HcommDataType;
typedef int HcommReduceOp;

/* HComm/HCCL正式批量描述符的最小ABI镜像。 */
typedef union HcommBatchTransferInfoAbi {
    uint8_t raws[56];
    struct {
        uint64_t len;
        void* dst;
        void* src;
    } write;
    struct {
        uint64_t len;
        void* dst;
        void* src;
    } read;
    struct {
        uint64_t count;
        void* dst;
        void* src;
        HcommReduceOp reduceOp;
        HcommDataType dataType;
    } reduce;
    struct {
        uint32_t notifyIdx;
    } notifyRecord;
    struct {
        uint64_t len;
        void* dst;
        void* src;
        uint32_t notifyIdx;
    } writeWithNotify;
    struct {
        uint64_t count;
        void* dst;
        void* src;
        HcommReduceOp reduceOp;
        HcommDataType dataType;
        uint32_t notifyIdx;
    } writeReduceWithNotify;
} HcommBatchTransferInfoAbi;

typedef struct HcommBatchTransferDesc {
    int32_t transType;
    uint8_t reserved[4];
    HcommBatchTransferInfoAbi transferInfo;
} HcommBatchTransferDesc;

typedef struct HcclHcommBatchTransferDesc {
    int32_t transType;
    uint8_t reserved[4];
    HcommBatchTransferInfoAbi transferInfo;
} HcclHcommBatchTransferDesc;

static_assert(sizeof(ThreadHandle) == sizeof(uint64_t), "ThreadHandle ABI mismatch");
static_assert(sizeof(ChannelHandle) == sizeof(uint64_t), "ChannelHandle ABI mismatch");
static_assert(sizeof(HcommDataType) == sizeof(int), "HcommDataType ABI mismatch");
static_assert(sizeof(HcommReduceOp) == sizeof(int), "HcommReduceOp ABI mismatch");
static_assert(sizeof(HcommBatchTransferDesc) == 64, "batch descriptor size mismatch");
static_assert(alignof(HcommBatchTransferDesc) == 8, "batch descriptor alignment mismatch");
static_assert(offsetof(HcommBatchTransferDesc, transType) == 0, "transType offset mismatch");
static_assert(offsetof(HcommBatchTransferDesc, reserved) == 4, "reserved offset mismatch");
static_assert(offsetof(HcommBatchTransferDesc, transferInfo) == 8, "transferInfo offset mismatch");
static_assert(sizeof(HcclHcommBatchTransferDesc) == 64, "HCCL batch descriptor size mismatch");
static_assert(alignof(HcclHcommBatchTransferDesc) == 8, "HCCL batch descriptor alignment mismatch");
static_assert(offsetof(HcclHcommBatchTransferDesc, transType) == 0, "HCCL transType offset mismatch");
static_assert(offsetof(HcclHcommBatchTransferDesc, reserved) == 4, "HCCL reserved offset mismatch");
static_assert(offsetof(HcclHcommBatchTransferDesc, transferInfo) == 8, "HCCL transferInfo offset mismatch");

namespace {

// 线程/Channel 上 notify 槽位上限（与 HcclThread.notifyId[40]/HcclChannel.notifyId[64] 对齐）。
constexpr uint32_t MAX_NOTIFY_PER_THREAD = 40;
constexpr uint32_t MAX_NOTIFY_PER_CHANNEL = 64;
// 桩使用的 notify 等待固定默认超时（秒）；仅用于日志，不写入任务元数据。
constexpr uint32_t DEFAULT_NOTIFY_WAIT_TIMEOUT = 1836;
// 跨 rank 反向 Channel 就绪等待参数，与 ctrl_comm_resource_stub 的
// REMOTE_MEM_WAIT_MAX_MS / REMOTE_MEM_WAIT_POLL_INTERVAL_MS 语义一致：
// 总等待 10s，每 10ms 轮询一次 RunnerDB。
constexpr uint32_t REVERSE_CHANNEL_WAIT_MAX_MS = 10000;
constexpr uint32_t REVERSE_CHANNEL_WAIT_POLL_INTERVAL_MS = 10;
// 任务元数据中 jettyId 的占位值：未关联具体 jetty 时填 uint32_t 最大值。
constexpr uint32_t UNKNOWN_JETTY_ID = std::numeric_limits<uint32_t>::max();

// 批量传输描述符的 transType 枚举，与真实实现的 HComm 批量传输类型一一对应。
enum TransferType : int32_t {
    TRANSFER_TYPE_WRITE = 0,                    // 本端 src → 远端 dst 的内存写入
    TRANSFER_TYPE_WRITE_REDUCE = 1,             // 本端 src → 远端 dst 的归约写入
    TRANSFER_TYPE_WRITE_WITH_NOTIFY = 2,        // 本端 src → 远端 dst 写入后发 notify
    TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY = 3, // 本端 src → 远端 dst 归约写入后发 notify
    TRANSFER_TYPE_READ = 4,                     // 远端 src → 本端 dst 的内存读取
    TRANSFER_TYPE_READ_REDUCE = 5,              // 远端 src → 本端 dst 的归约读取
    TRANSFER_TYPE_NOTIFY_RECORD = 6,            // 仅记录一条 notify record
    TRANSFER_TYPE_NOTIFY_WAIT = 7,              // notify wait（显式超时）
    TRANSFER_TYPE_NOTIFY_WAIT_DEFAULT = 8,      // notify wait（默认超时）
};

// ============================================================================
// 资源校验与解析辅助函数
//
// 这些函数把"句柄 → 数据库记录 → 稳定ID / rank"的解析逻辑集中在一起，
// 使每个公开数据面接口只需关注自身语义（写/读/notify 方向），不必重复 DB 查询与校验。
// 失败时统一返回 HcclResult 错误码并打印日志，调用方按非 SUCCESS 直接返回。
// ============================================================================

// 校验线程句柄，并从数据库读取对应的线程资源。
// thread==0 视为空指针；查不到记录视为资源已释放。
int32_t GetThreadResource(ThreadHandle handle, sim::runtime::HcclThread& thread)
{
    if (handle == 0) {
        HCCL_VM_ERROR("{} failed: handle=0, ret={:d}", __func__, static_cast<int32_t>(HCCL_E_PTR));
        return static_cast<int32_t>(HCCL_E_PTR);
    }

    auto threadRecord = sim::runtime::Db::GetById<sim::runtime::HcclThread>(handle);
    if (!threadRecord.ok()) {
        HCCL_VM_ERROR(
            "{} failed: handle={:d} not found, ret={:d}", __func__, handle, static_cast<int32_t>(HCCL_E_NOT_FOUND));
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }
    thread = *threadRecord;
    return static_cast<int32_t>(HCCL_SUCCESS);
}

// 校验通道句柄，并读取已连接的Channel资源。
// 与线程校验的区别：额外校验 channel.status，未连接的通道返回 HCCL_E_UNAVAIL。
int32_t GetChannelResource(ChannelHandle handle, sim::runtime::HcclChannel& channel)
{
    if (handle == 0) {
        HCCL_VM_ERROR("{} failed: channel=0, ret={:d}", __func__, static_cast<int32_t>(HCCL_E_PTR));
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    auto record = sim::runtime::Db::GetById<sim::runtime::HcclChannel>(handle);
    if (!record.ok()) {
        HCCL_VM_ERROR(
            "{} failed: channel={:d} not found, ret={:d}", __func__, handle, static_cast<int32_t>(HCCL_E_NOT_FOUND));
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }
    if (!record->status) {
        HCCL_VM_ERROR(
            "{} failed: channel={:d} unavailable, ret={:d}", __func__, handle, static_cast<int32_t>(HCCL_E_UNAVAIL));
        return static_cast<int32_t>(HCCL_E_UNAVAIL);
    }
    channel = *record;
    return static_cast<int32_t>(HCCL_SUCCESS);
}

// 校验Channel Notify索引，并解析稳定的Notify数据库ID。
// Channel 的 notifyId 数组元素已是 uint64_t（无需类型提升），
// 越界返回 HCCL_E_PARA（参数错误而非指针错误）。
int32_t GetChannelNotifyId(
    const sim::runtime::HcclChannel& channel, ChannelHandle handle, uint32_t notifyIdx, uint64_t& notifyId)
{
    if (notifyIdx >= channel.notifyNum || notifyIdx >= MAX_NOTIFY_PER_CHANNEL) {
        HCCL_VM_ERROR(
            "{} failed: channel={:d}, notifyIdx={:d}, notifyNum={:d}, ret={:d}", __func__, handle, notifyIdx,
            channel.notifyNum, static_cast<int32_t>(HCCL_E_PARA));
        return static_cast<int32_t>(HCCL_E_PARA);
    }
    notifyId = channel.notifyId[notifyIdx];
    if (notifyId == 0 || !sim::runtime::Db::GetById<sim::runtime::Notify>(notifyId).ok()) {
        HCCL_VM_ERROR(
            "{} failed: channel={:d}, notifyIdx={:d}, notifyId={:d} not found, ret={:d}", __func__, handle, notifyIdx,
            notifyId, static_cast<int32_t>(HCCL_E_NOT_FOUND));
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }
    return static_cast<int32_t>(HCCL_SUCCESS);
}

// 校验任务所属通信域，并取得设备进程的当前rank。
// 必须满足：thread.commId 对应的通信域存在，且通信域 myRank == 当前设备 rank，
// 否则说明任务不属于当前设备进程，返回 HCCL_E_INTERNAL。
// communicatorOut 非空时回写已取得的通信域行（成功路径），调用方需要
// rank_size 等字段时直接复用，不再为同一 commId 二次查询。
int32_t GetThreadTaskPosition(
    const sim::runtime::HcclThread& thread, ThreadHandle handle, uint32_t& rankId,
    sim::runtime::Communicator* communicatorOut = nullptr)
{
    if (thread.commId > std::numeric_limits<uint16_t>::max()) {
        HCCL_VM_ERROR(
            "{} failed: thread={:d}, commId={:d} exceeds uint16_t, ret={:d}", __func__, handle, thread.commId,
            static_cast<int32_t>(HCCL_E_PARA));
        return static_cast<int32_t>(HCCL_E_PARA);
    }

    auto communicator = sim::runtime::Db::GetById<sim::runtime::Communicator>(thread.commId);
    if (!communicator.ok()) {
        HCCL_VM_ERROR(
            "{} failed: thread={:d}, commId={:d} not found, ret={:d}", __func__, handle, thread.commId,
            static_cast<int32_t>(HCCL_E_NOT_FOUND));
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }

    auto rankRet = GetCurRankId(&rankId);
    if (rankRet != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS || communicator->rank_id != static_cast<uint64_t>(rankId)) {
        HCCL_VM_ERROR(
            "{} failed: thread={:d}, commId={:d}, deviceRank={:d}, commRank={:d}, ret={:d}", __func__, handle,
            thread.commId, rankId, communicator->rank_id, static_cast<int32_t>(HCCL_E_INTERNAL));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }
    if (communicatorOut != nullptr) {
        *communicatorOut = *communicator.value;
    }
    return static_cast<int32_t>(HCCL_SUCCESS);
}

// 同时解析线程和Channel，并校验远端rank属于线程通信域。
// 这是所有 Channel 数据面接口（write/read/notify/batch）的公共前置步骤：
//   1) 取线程资源（含 commId / streamId）
//   2) 取已连接通道资源（含 remoteRankId）
//   3) 校验线程通信域与当前设备 rank 一致（顺带取得通信域行）
//   4) 校验远端 rank 不越界，并记录到 remoteRankId
// 线程与通道通信域不一致时仅告警（真实实现未显式禁止），仍以源线程通信域记录任务。
// rank_size 边界校验复用步骤 3 取得的通信域行，同一 commId 只查询一次。
int32_t GetChannelTaskContext(
    ThreadHandle threadHandle, ChannelHandle channelHandle, sim::runtime::HcclThread& thread,
    sim::runtime::HcclChannel& channel, uint32_t& rankId, uint32_t& remoteRankId)
{
    int32_t ret = GetThreadResource(threadHandle, thread);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: GetThreadResource returned {:d}, thread={:d}, channel={:d}", __func__, ret, threadHandle,
            channelHandle);
        return ret;
    }
    ret = GetChannelResource(channelHandle, channel);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: GetChannelResource returned {:d}, thread={:d}, channel={:d}", __func__, ret, threadHandle,
            channelHandle);
        return ret;
    }
    sim::runtime::Communicator communicator{};
    ret = GetThreadTaskPosition(thread, threadHandle, rankId, &communicator);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: GetThreadTaskPosition returned {:d}, thread={:d}, channel={:d}", __func__, ret, threadHandle,
            channelHandle);
        return ret;
    }
    if (channel.remoteRankId > std::numeric_limits<uint32_t>::max() || channel.remoteRankId >= communicator.rank_size) {
        HCCL_VM_ERROR(
            "{} failed: thread={:d}, channel={:d}, remoteRank={:d}, rankSize={:d}, ret={:d}", __func__, threadHandle,
            channelHandle, channel.remoteRankId, communicator.rank_size, static_cast<int32_t>(HCCL_E_PARA));
        return static_cast<int32_t>(HCCL_E_PARA);
    }
    remoteRankId = static_cast<uint32_t>(channel.remoteRankId);
    if (thread.commId != channel.commId) {
        HCCL_VM_WARN(
            "{}: thread commId={:d} differs from channel commId={:d}, thread={:d}, channel={:d}", __func__,
            thread.commId, channel.commId, threadHandle, channelHandle);
    }
    return static_cast<int32_t>(HCCL_SUCCESS);
}

// ============================================================================
// 任务元数据构造辅助函数
//
// 所有构造函数都先 memset 整结构零初始化，再逐字段赋值。
// 这样保证 taskData 联合体中未被显式赋值的字段（如 protocol / notifyCount / resv）
// 确定为 0，避免 Checker 基于脏数据做误判。
// jettyId 统一填 UNKNOWN_JETTY_ID（占位），表示数据面任务不绑定具体 jetty。
// ============================================================================

// 构造通用 Notify 任务（src/dst 可跨 rank）。
// 通过 srcRankId/dstRankId 的方向区分 record（本端→远端）与 wait（远端→本端）。
HcclTaskMetaData MakeNotifyTask(
    HccLTaskMetaType taskType, const sim::runtime::HcclThread& thread, uint32_t rankId, uint32_t srcRankId,
    uint32_t dstRankId, uint64_t notifyId)
{
    HcclTaskMetaData taskMeta;
    std::memset(static_cast<void*>(&taskMeta), 0, sizeof(taskMeta));
    taskMeta.taskType = taskType;
    taskMeta.commId = thread.commId;
    taskMeta.rankId = rankId;
    taskMeta.streamId = thread.streamId;
    taskMeta.jettyId = UNKNOWN_JETTY_ID;
    taskMeta.taskData.notify.srcDeviceId = srcRankId;
    taskMeta.taskData.notify.dstDeviceId = dstRankId;
    if (taskType == HccLTaskMetaType::NOTIFY_RECORD) {
        taskMeta.taskData.notify.notifyId = HcclSim::MakeNotifyTaskId(dstRankId, notifyId);
    } else if (taskType == HccLTaskMetaType::NOTIFY_WAIT) {
        taskMeta.taskData.notify.notifyId = HcclSim::MakeNotifyTaskId(srcRankId, notifyId);
    }
    return taskMeta;
}

// 构造内存传输任务（MEM_CPY）。
// srcRankId/dstRankId 由调用方按 Write/Read/Local 方向传入：
//   - Write：src=本端, dst=远端
//   - Read ：src=远端, dst=本端
//   - Local：src=dst=本端
uint64_t TransRemoteAddrToVirtualByCommRank(uint64_t commId, uint64_t devAddr, uint32_t rankId)
{
    sim::runtime::Device device{};
    if (sim::runtime::GetDeviceByCommRank(commId, rankId, device) != ACL_SUCCESS || device.id > UINT32_MAX) {
        HCCL_VM_ERROR("{} failed: commId={:d}, rankId={:d}", __func__, commId, rankId);
        return 0;
    }
    return TransRemoteAddrToVirtualByDeviceId(devAddr, static_cast<uint32_t>(device.id));
}

HcclTaskMetaData MakeCopyTask(
    const sim::runtime::HcclThread& thread, uint32_t rankId, uint32_t srcRankId, const void* src, uint32_t dstRankId,
    const void* dst, uint64_t len)
{
    HcclTaskMetaData taskMeta;
    std::memset(static_cast<void*>(&taskMeta), 0, sizeof(taskMeta));
    taskMeta.taskType = HccLTaskMetaType::MEM_CPY;
    taskMeta.commId = thread.commId;
    taskMeta.rankId = rankId;
    taskMeta.streamId = thread.streamId;
    taskMeta.jettyId = UNKNOWN_JETTY_ID;
    taskMeta.taskData.transMem.srcDeviceId = srcRankId;
    const uint64_t srcDevAddr = reinterpret_cast<uint64_t>(src);
    taskMeta.taskData.transMem.srcOffset = (srcRankId == rankId) ?
                                               TransLocalAddrToVirtual(srcDevAddr) :
                                               TransRemoteAddrToVirtualByCommRank(thread.commId, srcDevAddr, srcRankId);
    taskMeta.taskData.transMem.dstDeviceId = dstRankId;
    const uint64_t dstDevAddr = reinterpret_cast<uint64_t>(dst);
    taskMeta.taskData.transMem.dstOffset = (dstRankId == rankId) ?
                                               TransLocalAddrToVirtual(dstDevAddr) :
                                               TransRemoteAddrToVirtualByCommRank(thread.commId, dstDevAddr, dstRankId);
    taskMeta.taskData.transMem.len = len;
    return taskMeta;
}

// 构造Reduce任务。
// dataCount 字段存储字节长度（由调用方完成 count * typeSize 换算），便于 Checker 直接做地址范围分析。
HcclTaskMetaData MakeReduceTask(
    const sim::runtime::HcclThread& thread, uint32_t rankId, uint32_t srcRankId, const void* src, uint32_t dstRankId,
    const void* dst, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    HcclTaskMetaData taskMeta;
    std::memset(static_cast<void*>(&taskMeta), 0, sizeof(taskMeta));
    taskMeta.taskType = HccLTaskMetaType::REDUCE;
    taskMeta.commId = thread.commId;
    taskMeta.rankId = rankId;
    taskMeta.streamId = thread.streamId;
    taskMeta.jettyId = UNKNOWN_JETTY_ID;
    taskMeta.taskData.reduce.srcDeviceId = srcRankId;
    const uint64_t srcDevAddr = reinterpret_cast<uint64_t>(src);
    taskMeta.taskData.reduce.srcOffset = (srcRankId == rankId) ?
                                             TransLocalAddrToVirtual(srcDevAddr) :
                                             TransRemoteAddrToVirtualByCommRank(thread.commId, srcDevAddr, srcRankId);
    taskMeta.taskData.reduce.dstDeviceId = dstRankId;
    const uint64_t dstDevAddr = reinterpret_cast<uint64_t>(dst);
    taskMeta.taskData.reduce.dstOffset = (dstRankId == rankId) ?
                                             TransLocalAddrToVirtual(dstDevAddr) :
                                             TransRemoteAddrToVirtualByCommRank(thread.commId, dstDevAddr, dstRankId);
    taskMeta.taskData.reduce.dataCount = count;
    taskMeta.taskData.reduce.dataType = static_cast<uint8_t>(dataType);
    taskMeta.taskData.reduce.reduceOp = static_cast<uint8_t>(reduceOp);
    return taskMeta;
}

// 将单条任务写入设备任务表，并统一转换数据库错误码。
// InsertTaskToCollectionDev 返回非 0 表示入库失败，统一映射为 HCCL_E_INTERNAL。
int32_t InsertCollectedTask(const HcclTaskMetaData& task)
{
    HcclTaskMetaData normalizedTask = task;
    const auto communicator = sim::runtime::Db::GetById<sim::runtime::Communicator>(normalizedTask.commId);
    if (!communicator.ok()) {
        HCCL_VM_ERROR("{} failed: commId={:d} not found", __func__, normalizedTask.commId);
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }

    auto getDeviceId = [&normalizedTask](uint32_t rankId, uint32_t& deviceId) {
        sim::runtime::Device device{};
        if (sim::runtime::GetDeviceByCommRank(normalizedTask.commId, rankId, device) != ACL_SUCCESS) {
            HCCL_VM_ERROR(
                "{} failed: cannot map commId={:d}, rankId={:d} to deviceId", __func__, normalizedTask.commId, rankId);
            return false;
        }
        deviceId = static_cast<uint32_t>(device.id);
        return true;
    };

    uint32_t taskDeviceId = 0;
    if (!getDeviceId(normalizedTask.rankId, taskDeviceId)) {
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }
    normalizedTask.deviceId = taskDeviceId;
    switch (normalizedTask.taskType) {
        case HccLTaskMetaType::MEM_CPY:
            if (!getDeviceId(normalizedTask.taskData.transMem.srcDeviceId, normalizedTask.taskData.transMem.srcDeviceId)
                || !getDeviceId(
                    normalizedTask.taskData.transMem.dstDeviceId, normalizedTask.taskData.transMem.dstDeviceId)) {
                return static_cast<int32_t>(HCCL_E_INTERNAL);
            }
            break;
        case HccLTaskMetaType::REDUCE:
            if (!getDeviceId(normalizedTask.taskData.reduce.srcDeviceId, normalizedTask.taskData.reduce.srcDeviceId)
                || !getDeviceId(
                    normalizedTask.taskData.reduce.dstDeviceId, normalizedTask.taskData.reduce.dstDeviceId)) {
                return static_cast<int32_t>(HCCL_E_INTERNAL);
            }
            break;
        case HccLTaskMetaType::NOTIFY_RECORD:
        case HccLTaskMetaType::NOTIFY_WAIT:
            if (!getDeviceId(normalizedTask.taskData.notify.srcDeviceId, normalizedTask.taskData.notify.srcDeviceId)
                || !getDeviceId(
                    normalizedTask.taskData.notify.dstDeviceId, normalizedTask.taskData.notify.dstDeviceId)) {
                return static_cast<int32_t>(HCCL_E_INTERNAL);
            }
            break;
        default:
            break;
    }
    normalizedTask.rankId = UINT32_MAX;
    const int32_t insertRet = InsertTaskToCollectionDev(&normalizedTask);
    if (insertRet != 0) {
        HCCL_VM_ERROR(
            "{} failed: taskType={:d}, insertRet={:d}, ret={:d}", __func__, static_cast<int32_t>(task.taskType),
            insertRet, static_cast<int32_t>(HCCL_E_INTERNAL));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }
    return static_cast<int32_t>(HCCL_SUCCESS);
}

// ============================================================================
// 参数校验辅助函数
// ============================================================================

// 校验A5 Reduce枚举并取得元素字节数。
// 真实 A5 AICPU 仅支持固定的 dataType 子集与 reduceOp 0~3（SUM/PROD/MAX/MIN），
// 本函数按 A5 支持矩阵做白名单校验，并计算 count * typeSize 的溢出。
int32_t ValidateA5Reduce(HcommDataType dataType, HcommReduceOp reduceOp, uint64_t count, uint32_t& typeSize)
{
    switch (dataType) {
        case 0:
        case 7:
        case 14:
        case 15:
        case 16:
        case 17: {
            typeSize = 1;
            break;
        }
        case 1:
        case 3:
        case 8:
        case 11: {
            typeSize = 2;
            break;
        }
        case 2:
        case 4:
        case 9: {
            typeSize = 4;
            break;
        }
        case 5:
        case 6:
        case 10: {
            typeSize = 8;
            break;
        }
        case 12: {
            typeSize = 16;
            break;
        }
        default:
            HCCL_VM_ERROR(
                "{} failed: dataType={:d} unsupported on A5, ret={:d}", __func__, dataType,
                static_cast<int32_t>(HCCL_E_PARA));
            return static_cast<int32_t>(HCCL_E_PARA);
    }
    if (reduceOp < 0 || reduceOp > 3) {
        HCCL_VM_ERROR(
            "{} failed: reduceOp={:d} unsupported on A5, ret={:d}", __func__, reduceOp,
            static_cast<int32_t>(HCCL_E_PARA));
        return static_cast<int32_t>(HCCL_E_PARA);
    }
    if (count != 0 && count > std::numeric_limits<uint64_t>::max() / typeSize) {
        HCCL_VM_ERROR(
            "{} failed: count={:d}, dataType={:d} byte length overflow, ret={:d}", __func__, count, dataType,
            static_cast<int32_t>(HCCL_E_PARA));
        return static_cast<int32_t>(HCCL_E_PARA);
    }
    return static_cast<int32_t>(HCCL_SUCCESS);
}

// 校验float超时并按真实实现截断为uint32_t。
// 真实实现将 float→uint32_t 做向零截断；NaN / 负数 / 超过 uint32_t 上限均视为非法。
int32_t NormalizeTimeout(float timeOut, uint32_t& normalized)
{
    if (std::isnan(timeOut) || timeOut < 0.0F
        || static_cast<double>(timeOut) > static_cast<double>(std::numeric_limits<uint32_t>::max())) {
        HCCL_VM_ERROR("{} failed: timeOut={} invalid, ret={:d}", __func__, timeOut, static_cast<int32_t>(HCCL_E_PARA));
        return static_cast<int32_t>(HCCL_E_PARA);
    }
    normalized = static_cast<uint32_t>(timeOut);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

// 通过notifyId反查同通信域的Channel，返回远端rankId。
// ACL notify 接口没有显式的远端 rank 参数，但 notifyId 可能登记在某条 Channel 的
// notifyId 数组中；若命中则说明该 notify 关联跨节点通信，据此还原远端 rank 方向。
bool FindRemoteRankByNotifyId(uint64_t commId, uint64_t notifyId, uint32_t& remoteRankId)
{
    auto channels = sim::runtime::Db::GetByPred<sim::runtime::HcclChannel>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::HcclChannel::commId, commId),
        HcclSim::Storage::Eq(&sim::runtime::HcclChannel::status, true)));
    if (channels.ok() && channels.value.has_value()) {
        for (const auto& ch : *channels.value) {
            for (uint16_t i = 0; i < ch.notifyNum; ++i) {
                if (ch.notifyId[i] == notifyId) {
                    remoteRankId = static_cast<uint32_t>(ch.remoteRankId);
                    return true;
                }
            }
        }
    }
    return false;
}

// 轮询等待对端通信域与指向本端的反向 Channel 就绪。
// 跨 rank 场景下两侧 Channel 独立创建：本端 channel 连接成功后立即发 NOTIFY_RECORD 时，
// 对端反向 channel 可能尚未入库（高并发下窗口可达数百毫秒），单次查询会误报
// reverse channel not found。Channel 记录入库时 notifyId 数组已同步填写完成，
// 因此只需等待记录可见即可保证后续 notifyId 解析可用。
// 对端通信域按 localComm 的 rank_size / comm_hash / comm_id 严格匹配，避免同名多通信域
// 场景下按 rank 误匹配。以 REVERSE_CHANNEL_WAIT_POLL_INTERVAL_MS 为间隔轮询，总时长不超过
// REVERSE_CHANNEL_WAIT_MAX_MS（10s）；remoteCommId 在查到对端通信域后即回填，
// 便于超时日志定位是对端通信域缺失还是反向 channel 缺失。
int32_t WaitForReverseChannelResource(
    const sim::runtime::Communicator& localComm, uint32_t localRank, uint32_t remoteRank, uint8_t engine,
    uint64_t& remoteCommId, sim::runtime::HcclChannel& reverseChannel)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(REVERSE_CHANNEL_WAIT_MAX_MS);
    while (true) {
        auto remoteComms = sim::runtime::Db::GetByPred<sim::runtime::Communicator>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Communicator::rank_id, static_cast<uint64_t>(remoteRank)),
            HcclSim::Storage::Eq(&sim::runtime::Communicator::rank_size, localComm.rank_size),
            HcclSim::Storage::Eq(&sim::runtime::Communicator::comm_hash, localComm.comm_hash),
            HcclSim::Storage::Eq(&sim::runtime::Communicator::comm_id, std::string(localComm.comm_id))));
        if (remoteComms.ok() && remoteComms.value.has_value() && !remoteComms.value->empty()) {
            remoteCommId = remoteComms.value->front().id;
            auto reverseChannels = sim::runtime::Db::GetByPred<sim::runtime::HcclChannel>(HcclSim::Storage::And(
                HcclSim::Storage::Eq(&sim::runtime::HcclChannel::commId, remoteCommId),
                HcclSim::Storage::Eq(&sim::runtime::HcclChannel::remoteRankId, static_cast<uint64_t>(localRank)),
                HcclSim::Storage::Eq(&sim::runtime::HcclChannel::engine, static_cast<uint64_t>(engine)),
                HcclSim::Storage::Eq(&sim::runtime::HcclChannel::status, true)));
            if (reverseChannels.ok() && reverseChannels.value.has_value() && !reverseChannels.value->empty()) {
                reverseChannel = reverseChannels.value->front();
                return static_cast<int32_t>(HCCL_SUCCESS);
            }
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            HCCL_VM_ERROR(
                "{} failed: reverse channel not ready after {:d}ms "
                "(remoteCommId={:d}, localRank={:d}, remoteRank={:d}), ret={:d}",
                __func__, REVERSE_CHANNEL_WAIT_MAX_MS, remoteCommId, localRank, remoteRank,
                static_cast<int32_t>(HCCL_E_NOT_FOUND));
            return static_cast<int32_t>(HCCL_E_NOT_FOUND);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(REVERSE_CHANNEL_WAIT_POLL_INTERVAL_MS));
    }
}

// 查找指向本端的反向 channel，并读取其 notifyId[notifyIdx]。
// NOTIFY_RECORD 向对端发信号，需读取对端 channel（反向 channel）上的 notifyId，
// 而非本端 channel 的 notifyId。
// engine 为本端 channel 的引擎类型：反向 channel 必须与本端 channel 同 engine，
// 否则会读到对端其它 engine channel 上的 notifyId，造成两侧编号错位。
int32_t GetReverseChannelNotifyId(
    uint64_t localCommId, uint32_t localRank, uint32_t remoteRank, uint8_t engine, uint32_t notifyIdx,
    uint64_t& notifyId)
{
    const auto localComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(localCommId);
    if (!localComm.ok()) {
        HCCL_VM_ERROR("{} failed: local communicator={} not found", __func__, localCommId);
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }
    uint64_t remoteCommId = 0;
    sim::runtime::HcclChannel reverseCh{};
    int32_t ret = WaitForReverseChannelResource(*localComm, localRank, remoteRank, engine, remoteCommId, reverseCh);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: reverse channel not found (remoteCommId={:d}, localRank={:d}, remoteRank={:d}), ret={:d}",
            __func__, remoteCommId, localRank, remoteRank, ret);
        return ret;
    }
    return GetChannelNotifyId(reverseCh, reverseCh.id, notifyIdx, notifyId);
}

// ============================================================================
// 公共采集路径
//
// 以下函数被公开桩接口复用，避免在每个公开接口里重复"校验+构造+入库"的样板。
// 本身不对外导出，仅供 HcommBatchTransfer* 系列调用。
// ============================================================================

// 预校验并展开一个批量描述符。
// 按描述符的 transType 选择对应的任务构造方式，方向语义如下：
//   - WRITE / WRITE_REDUCE / WRITE_*_WITH_NOTIFY：本端(src) → 远端(dst)，方向 = local→remote
//   - READ / READ_REDUCE：远端(src) → 本端(dst)，方向 = remote→local
//   - NOTIFY_RECORD：本端 → 远端的 notify 记录
//   - NOTIFY_WAIT*：设备侧批量传输不支持纯 wait，返回 HCCL_E_NOT_SUPPORT
// 任意校验失败立即返回，不向 tasks 追加部分结果。
int32_t ExpandBatchDescriptor(
    const HcommBatchTransferDesc& desc, uint32_t index, const sim::runtime::HcclThread& thread,
    const sim::runtime::HcclChannel& channel, uint32_t localRank, uint32_t remoteRank,
    std::vector<HcclTaskMetaData>& tasks)
{
    uint32_t typeSize = 0;
    uint64_t notifyId = 0;
    int32_t ret = static_cast<int32_t>(HCCL_SUCCESS);

    switch (desc.transType) {
        case TRANSFER_TYPE_WRITE:
            // 本端 src → 远端 dst 的内存写入
            if (desc.transferInfo.write.dst == nullptr || desc.transferInfo.write.src == nullptr) {
                ret = static_cast<int32_t>(HCCL_E_PTR);
                break;
            }
            tasks.push_back(MakeCopyTask(
                thread, localRank, localRank, desc.transferInfo.write.src, remoteRank, desc.transferInfo.write.dst,
                desc.transferInfo.write.len));
            return static_cast<int32_t>(HCCL_SUCCESS);
        case TRANSFER_TYPE_READ:
            // 远端 src → 本端 dst 的内存读取（方向与 WRITE 相反）
            if (desc.transferInfo.read.dst == nullptr || desc.transferInfo.read.src == nullptr) {
                ret = static_cast<int32_t>(HCCL_E_PTR);
                break;
            }
            tasks.push_back(MakeCopyTask(
                thread, localRank, remoteRank, desc.transferInfo.read.src, localRank, desc.transferInfo.read.dst,
                desc.transferInfo.read.len));
            return static_cast<int32_t>(HCCL_SUCCESS);
        case TRANSFER_TYPE_WRITE_REDUCE:
        case TRANSFER_TYPE_READ_REDUCE: {
            // 归约传输：先校验 A5 支持矩阵，再按方向构造 REDUCE 任务
            if (desc.transferInfo.reduce.dst == nullptr || desc.transferInfo.reduce.src == nullptr) {
                ret = static_cast<int32_t>(HCCL_E_PTR);
                break;
            }
            ret = ValidateA5Reduce(
                desc.transferInfo.reduce.dataType, desc.transferInfo.reduce.reduceOp, desc.transferInfo.reduce.count,
                typeSize);
            if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
                break;
            }
            // WRITE_REDUCE: src=本端, dst=远端；READ_REDUCE: src=远端, dst=本端
            const bool isWrite = (desc.transType == TRANSFER_TYPE_WRITE_REDUCE);
            tasks.push_back(MakeReduceTask(
                thread, localRank, isWrite ? localRank : remoteRank, desc.transferInfo.reduce.src,
                isWrite ? remoteRank : localRank, desc.transferInfo.reduce.dst,
                desc.transferInfo.reduce.count * typeSize, desc.transferInfo.reduce.dataType,
                desc.transferInfo.reduce.reduceOp));
            return static_cast<int32_t>(HCCL_SUCCESS);
        }
        case TRANSFER_TYPE_WRITE_WITH_NOTIFY:
            // 本端 src → 远端 dst 写入后，向远端发 notify（两段任务：MEM_CPY + NOTIFY_RECORD）
            if (desc.transferInfo.writeWithNotify.dst == nullptr || desc.transferInfo.writeWithNotify.src == nullptr) {
                ret = static_cast<int32_t>(HCCL_E_PTR);
                break;
            }
            ret = GetReverseChannelNotifyId(
                thread.commId, localRank, remoteRank, channel.engine, desc.transferInfo.writeWithNotify.notifyIdx,
                notifyId);
            if (ret != static_cast<int32_t>(HCCL_SUCCESS))
                break;
            tasks.push_back(MakeCopyTask(
                thread, localRank, localRank, desc.transferInfo.writeWithNotify.src, remoteRank,
                desc.transferInfo.writeWithNotify.dst, desc.transferInfo.writeWithNotify.len));
            tasks.push_back(
                MakeNotifyTask(HccLTaskMetaType::NOTIFY_RECORD, thread, localRank, localRank, remoteRank, notifyId));
            return static_cast<int32_t>(HCCL_SUCCESS);
        case TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY:
            // 本端 src → 远端 dst 归约写入后，向远端发 notify（两段任务：REDUCE + NOTIFY_RECORD）
            if (desc.transferInfo.writeReduceWithNotify.dst == nullptr
                || desc.transferInfo.writeReduceWithNotify.src == nullptr) {
                ret = static_cast<int32_t>(HCCL_E_PTR);
                break;
            }
            ret = ValidateA5Reduce(
                desc.transferInfo.writeReduceWithNotify.dataType, desc.transferInfo.writeReduceWithNotify.reduceOp,
                desc.transferInfo.writeReduceWithNotify.count, typeSize);
            if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
                break;
            }
            ret = GetReverseChannelNotifyId(
                thread.commId, localRank, remoteRank, channel.engine, desc.transferInfo.writeReduceWithNotify.notifyIdx,
                notifyId);
            if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
                break;
            }
            tasks.push_back(MakeReduceTask(
                thread, localRank, localRank, desc.transferInfo.writeReduceWithNotify.src, remoteRank,
                desc.transferInfo.writeReduceWithNotify.dst, desc.transferInfo.writeReduceWithNotify.count * typeSize,
                desc.transferInfo.writeReduceWithNotify.dataType, desc.transferInfo.writeReduceWithNotify.reduceOp));
            tasks.push_back(
                MakeNotifyTask(HccLTaskMetaType::NOTIFY_RECORD, thread, localRank, localRank, remoteRank, notifyId));
            return static_cast<int32_t>(HCCL_SUCCESS);
        case TRANSFER_TYPE_NOTIFY_RECORD:
            // 仅记录一条本端→远端的 notify record
            ret = GetReverseChannelNotifyId(
                thread.commId, localRank, remoteRank, channel.engine, desc.transferInfo.notifyRecord.notifyIdx,
                notifyId);
            if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
                break;
            }
            tasks.push_back(
                MakeNotifyTask(HccLTaskMetaType::NOTIFY_RECORD, thread, localRank, localRank, remoteRank, notifyId));
            return static_cast<int32_t>(HCCL_SUCCESS);
        case TRANSFER_TYPE_NOTIFY_WAIT:
        case TRANSFER_TYPE_NOTIFY_WAIT_DEFAULT:
            // 设备侧批量传输不支持纯 wait 类型
            ret = static_cast<int32_t>(HCCL_E_NOT_SUPPORT);
            break;
        default:
            ret = static_cast<int32_t>(HCCL_E_PARA);
            break;
    }
    HCCL_VM_ERROR("{} failed: descIndex={:d}, transferType={:d}, ret={:d}", __func__, index, desc.transType, ret);
    return ret;
}

// 全量预校验批量描述符，再按描述符顺序写入任务数据库。
// 采用"先全部展开再统一入库"的策略：任一描述符校验失败则整批失败，避免部分任务入库
// 导致 Checker 看到不完整的任务序列。
int32_t
CollectBatchTransfers(ThreadHandle thread, ChannelHandle channel, const HcommBatchTransferDesc* descs, uint32_t descNum)
{
    if (descs == nullptr) {
        HCCL_VM_ERROR("{} failed: transferDescs is nullptr, ret={:d}", __func__, static_cast<int32_t>(HCCL_E_PTR));
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    if (descNum == 0) {
        HCCL_VM_ERROR("{} failed: transferDescNum=0, ret={:d}", __func__, static_cast<int32_t>(HCCL_E_PARA));
        return static_cast<int32_t>(HCCL_E_PARA);
    }
    sim::runtime::HcclThread threadRes;
    sim::runtime::HcclChannel channelRes;
    uint32_t rankId = 0;
    uint32_t remoteRankId = 0;
    int32_t ret = GetChannelTaskContext(thread, channel, threadRes, channelRes, rankId, remoteRankId);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: GetChannelTaskContext returned {:d}, thread={:d}, channel={:d}", __func__, ret, thread,
            channel);
        return ret;
    }

    std::vector<HcclTaskMetaData> tasks;
    try {
        tasks.reserve(static_cast<size_t>(descNum) * 2U);
        for (uint32_t i = 0; i < descNum; ++i) {
            ret = ExpandBatchDescriptor(descs[i], i, threadRes, channelRes, rankId, remoteRankId, tasks);
            if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
                HCCL_VM_ERROR(
                    "{} failed: ExpandBatchDescriptor returned {:d}, descIndex={:d}, thread={:d}, channel={:d}",
                    __func__, ret, i, thread, channel);
                return ret;
            }
        }
    } catch (const std::bad_alloc&) {
        HCCL_VM_ERROR(
            "{} failed: transferDescNum={:d}, allocation failed, ret={:d}", __func__, descNum,
            static_cast<int32_t>(HCCL_E_MEMORY));
        return static_cast<int32_t>(HCCL_E_MEMORY);
    }
    for (const auto& task : tasks) {
        ret = InsertCollectedTask(task);
        if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
            HCCL_VM_ERROR(
                "{} failed: InsertCollectedTask returned {:d}, taskType={:d}, thread={:d}, channel={:d}", __func__, ret,
                static_cast<int32_t>(task.taskType), thread, channel);
            return ret;
        }
    }
    HCCL_VM_INFO(
        "{} success, thread={:d}, channel={:d}, transferDescNum={:d}, taskNum={:d}", __func__, thread, channel, descNum,
        tasks.size());
    return static_cast<int32_t>(HCCL_SUCCESS);
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 获取通信域资源（设备侧数据面接口）。
 *
 * 根据通信域标识确认对应通信域仍然存在，不维护全局“当前通信域”状态。
 * 对应 hcomm 仓同名接口。
 *
 * @param commId 输入：通信域标识字符串。
 * @return int32_t 成功返回 HCCL_SUCCESS；commId 为空返回 HCCL_E_PTR；
 *                 通信域不存在返回 HCCL_E_NOT_FOUND。
 */
int32_t HcommAcquireComm(const char* commId)
{
    HCCL_VM_INFO("{}: commId={}", __func__, commId == nullptr ? "(null)" : commId);

    if (commId == nullptr) {
        HCCL_VM_ERROR("{} failed: commId is nullptr, ret={:d}", __func__, static_cast<int32_t>(HCCL_E_PTR));
        return static_cast<int32_t>(HCCL_E_PTR);
    }

    // 按名称在通信域表中查找；commName 为定长数组，需用 strncmp 比较完整字符串
    auto communicators = sim::runtime::Db::GetByPred<sim::runtime::Communicator>(
        HcclSim::Storage::Eq(&sim::runtime::Communicator::comm_id, std::string(commId)));
    if (!communicators.ok() || !communicators.value.has_value() || communicators.value->empty()) {
        HCCL_VM_ERROR(
            "{} failed: commId={} not found, ret={:d}", __func__, commId, static_cast<int32_t>(HCCL_E_NOT_FOUND));
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }

    HCCL_VM_INFO("{} success, commId={}, communicatorId={:d}", __func__, commId, communicators.value->front().id);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

/**
 * @brief 释放通信域资源（设备侧数据面接口）。
 *
 * 根据通信域标识确认对应通信域仍然存在，结束本次使用但不销毁通信域记录。
 * 对应 hcomm 仓同名接口。
 *
 * @param commId 输入：通信域标识字符串。
 * @return int32_t 成功返回 HCCL_SUCCESS；commId 为空返回 HCCL_E_PTR；
 *                 通信域不存在返回 HCCL_E_NOT_FOUND。
 */
int32_t HcommReleaseComm(const char* commId)
{
    HCCL_VM_INFO("{}: commId={}", __func__, commId == nullptr ? "(null)" : commId);

    if (commId == nullptr) {
        HCCL_VM_ERROR("{} failed: commId is nullptr, ret={:d}", __func__, static_cast<int32_t>(HCCL_E_PTR));
        return static_cast<int32_t>(HCCL_E_PTR);
    }

    auto communicators = sim::runtime::Db::GetByPred<sim::runtime::Communicator>(
        HcclSim::Storage::Eq(&sim::runtime::Communicator::comm_id, std::string(commId)));
    if (!communicators.ok() || !communicators.value.has_value() || communicators.value->empty()) {
        HCCL_VM_ERROR(
            "{} failed: commId={} not found, ret={:d}", __func__, commId, static_cast<int32_t>(HCCL_E_NOT_FOUND));
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }

    HCCL_VM_INFO("{} success, commId={}, communicatorId={:d}", __func__, commId, communicators.value->front().id);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

/**
 * @brief 启动批处理模式（设备侧数据面接口）。
 *
 * 将多个通信操作打包成一个批次执行，以提高执行效率。
 * 对应 hcomm 仓同名接口。
 *
 * 桩不缓存任务；数据面任务仍由各操作接口按调用顺序直接采集。
 * batchTag 为空时使用真实实现定义的默认标签。
 *
 * @param batchTag 输入：批次标签字符串，用于标识本次批处理。
 * @return int32_t 始终返回 HCCL_SUCCESS。
 */
int32_t HcommBatchModeStart(const char* batchTag)
{
    const char* effectiveTag = (batchTag == nullptr) ? "" : batchTag;
    HCCL_VM_INFO("{} success, batchTag={}", __func__, effectiveTag);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

/**
 * @brief 结束批处理模式（设备侧数据面接口）。
 *
 * 结束由 HcommBatchModeStart 启动的批处理，提交并执行批次中的所有操作。
 * 对应 hcomm 仓同名接口。
 *
 * 桩不提交或清理任务缓存；此前的数据面任务已经由对应操作接口完成采集。
 * batchTag 为空时使用真实实现定义的默认标签。
 *
 * @param batchTag 输入：批次标签字符串，需与 HcommBatchModeStart 传入的一致。
 * @return int32_t 始终返回 HCCL_SUCCESS。
 */
int32_t HcommBatchModeEnd(const char* batchTag)
{
    const char* effectiveTag = (batchTag == nullptr) ? "" : batchTag;
    HCCL_VM_INFO("{} success, batchTag={}", __func__, effectiveTag);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

/**
 * @brief 在指定线程上发送 notify 信号给目标线程（设备侧数据面接口）。
 *
 * 向 dstThread 的 dstNotifyIdx 索引位置发送通知信号。
 * 对应 hcomm 仓同名接口。
 *
 * 在源线程的stream上生成NOTIFY_RECORD任务，Notify资源取自目标线程的指定索引，
 * 并将任务元数据写入设备侧任务表。
 *
 * @param thread 输入：发起 notify 的线程句柄（src 线程）。
 * @param dstThread 输入：目标线程句柄（接收端线程）。
 * @param dstNotifyIdx 输入：目标线程上的 notify 索引。
 * @return int32_t 成功返回HCCL_SUCCESS；空句柄返回HCCL_E_PTR；资源不存在返回
 *                 HCCL_E_NOT_FOUND；通信域字段越界返回HCCL_E_PARA；入库失败返回
 *                 HCCL_E_INTERNAL。
 */
int32_t HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx)
{
    HCCL_VM_INFO("{}: thread={:d}, dstThread={:d}, dstNotifyIdx={:d}", __func__, thread, dstThread, dstNotifyIdx);

    // 【核心步骤 1】校验源线程资源（任务记录在源线程 stream 上）
    sim::runtime::HcclThread srcThread{};
    int32_t ret = GetThreadResource(thread, srcThread);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: GetThreadResource(src) returned {:d}, thread={:d}, dstThread={:d}", __func__, ret, thread,
            dstThread);
        return ret;
    }

    // 【核心步骤 2】校验目标线程资源并解析 notifyId
    sim::runtime::HcclThread dstThreadRes{};
    ret = GetThreadResource(dstThread, dstThreadRes);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: GetThreadResource(dst) returned {:d}, thread={:d}, dstThread={:d}", __func__, ret, thread,
            dstThread);
        return ret;
    }

    uint64_t notifyIdVal = 0;
    if (dstNotifyIdx >= dstThreadRes.notifyNum || dstNotifyIdx >= MAX_NOTIFY_PER_THREAD) {
        HCCL_VM_ERROR(
            "{} failed: dstNotifyIdx={:d} out of range (notifyNum={:d}), thread={:d}, dstThread={:d}, ret={:d}",
            __func__, dstNotifyIdx, dstThreadRes.notifyNum, thread, dstThread, static_cast<int32_t>(HCCL_E_PTR));
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    notifyIdVal = static_cast<uint64_t>(dstThreadRes.notifyId[dstNotifyIdx]);
    if (notifyIdVal == 0 || !sim::runtime::Db::GetById<sim::runtime::Notify>(notifyIdVal).ok()) {
        HCCL_VM_ERROR(
            "{} failed: notifyId={:d} not found in DB, thread={:d}, dstThread={:d}, dstNotifyIdx={:d}, ret={:d}",
            __func__, notifyIdVal, thread, dstThread, dstNotifyIdx, static_cast<int32_t>(HCCL_E_NOT_FOUND));
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }

    // 【核心步骤 3】校验源线程所属通信域与当前 rank 一致
    uint32_t curRank = 0;
    ret = GetThreadTaskPosition(srcThread, thread, curRank);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR("{} failed: GetThreadTaskPosition returned {:d}, thread={:d}", __func__, ret, thread);
        return ret;
    }

    // 跨通信域用法未被真实实现显式禁止，仅告警
    if (srcThread.commId != dstThreadRes.commId) {
        HCCL_VM_WARN(
            "{}: src commId={:d} differs from dst commId={:d}, "
            "thread={:d}, dstThread={:d}",
            __func__, srcThread.commId, dstThreadRes.commId, thread, dstThread);
    }

    // 【核心步骤 4】构建 NOTIFY_RECORD 任务元数据
    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    taskMetaData.commId = srcThread.commId;
    taskMetaData.rankId = curRank;
    taskMetaData.streamId = srcThread.streamId;

    taskMetaData.taskData.notify.notifyId = HcclSim::MakeNotifyTaskId(curRank, notifyIdVal);
    taskMetaData.taskData.notify.notifyCount = static_cast<uint8_t>(dstNotifyIdx);
    taskMetaData.taskData.notify.srcDeviceId = curRank;
    taskMetaData.taskData.notify.dstDeviceId = curRank;

    // 【核心步骤 5】插入任务到集合
    const int32_t insertRet = InsertCollectedTask(taskMetaData);
    if (insertRet != 0) {
        HCCL_VM_ERROR(
            "{} failed: InsertTaskToCollectionDev failed, "
            "insertRet={:d}, ret={:d}",
            __func__, insertRet, static_cast<int32_t>(HCCL_E_INTERNAL));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, dstThread={:d}, "
        "dstNotifyIdx={:d}, commId={:d}, rankId={:d}, streamId={:d}, notifyId={:d}",
        __func__, thread, dstThread, dstNotifyIdx, srcThread.commId, curRank, srcThread.streamId, notifyIdVal);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

/**
 * @brief 在指定线程上等待 notifyIdx 位置的通知信号（设备侧数据面接口）。
 *
 * 阻塞等待本线程上 notifyIdx 索引位置的通知信号到达。
 * 对应 hcomm 仓同名接口。
 *
 * 在指定线程的stream上生成NOTIFY_WAIT任务，Notify资源取自该线程的指定索引，
 * 并将任务元数据写入设备侧任务表；桩不执行真实阻塞。
 *
 * @param thread 输入：等待 notify 的线程句柄。
 * @param notifyIdx 输入：本线程上的 notify 索引。
 * @param timeOut 输入：超时时间（秒），0 表示永久等待；当前任务元数据不保存该字段。
 * @return int32_t 成功返回HCCL_SUCCESS；空句柄返回HCCL_E_PTR；资源不存在返回
 *                 HCCL_E_NOT_FOUND；通信域字段越界返回HCCL_E_PARA；入库失败返回
 *                 HCCL_E_INTERNAL。
 */
int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut)
{
    HCCL_VM_INFO("{}: thread={:d}, notifyIdx={:d}, timeout={:d}", __func__, thread, notifyIdx, timeOut);

    // 【核心步骤 1】校验等待方线程资源
    sim::runtime::HcclThread threadRes{};
    int32_t ret = GetThreadResource(thread, threadRes);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: GetThreadResource returned {:d}, thread={:d}, notifyIdx={:d}", __func__, ret, thread,
            notifyIdx);
        return ret;
    }

    // 【核心步骤 2】解析通知 ID 并校验在 DB 中存在
    uint64_t notifyIdVal = 0;
    if (notifyIdx >= threadRes.notifyNum || notifyIdx >= MAX_NOTIFY_PER_THREAD) {
        HCCL_VM_ERROR(
            "{} failed: notifyIdx={:d} out of range (notifyNum={:d}), thread={:d}, ret={:d}", __func__, notifyIdx,
            threadRes.notifyNum, thread, static_cast<int32_t>(HCCL_E_PTR));
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    notifyIdVal = static_cast<uint64_t>(threadRes.notifyId[notifyIdx]);
    if (notifyIdVal == 0 || !sim::runtime::Db::GetById<sim::runtime::Notify>(notifyIdVal).ok()) {
        HCCL_VM_ERROR(
            "{} failed: notifyId={:d} not found in DB, thread={:d}, notifyIdx={:d}, ret={:d}", __func__, notifyIdVal,
            thread, notifyIdx, static_cast<int32_t>(HCCL_E_NOT_FOUND));
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }

    // 【核心步骤 3】校验线程所属通信域与当前 rank 一致
    uint32_t curRank = 0;
    ret = GetThreadTaskPosition(threadRes, thread, curRank);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR("{} failed: GetThreadTaskPosition returned {:d}, thread={:d}", __func__, ret, thread);
        return ret;
    }

    // 【核心步骤 4】构建 NOTIFY_WAIT 任务元数据
    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    taskMetaData.commId = threadRes.commId;
    taskMetaData.rankId = curRank;
    taskMetaData.streamId = threadRes.streamId;

    taskMetaData.taskData.notify.notifyId = HcclSim::MakeNotifyTaskId(curRank, notifyIdVal);
    taskMetaData.taskData.notify.notifyCount = static_cast<uint8_t>(notifyIdx);
    taskMetaData.taskData.notify.srcDeviceId = curRank;
    taskMetaData.taskData.notify.dstDeviceId = curRank;

    // 【核心步骤 5】插入任务到集合
    const int32_t insertRet = InsertCollectedTask(taskMetaData);
    if (insertRet != 0) {
        HCCL_VM_ERROR(
            "{} failed: InsertTaskToCollectionDev failed, "
            "insertRet={:d}, ret={:d}",
            __func__, insertRet, static_cast<int32_t>(HCCL_E_INTERNAL));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, notifyIdx={:d}, "
        "commId={:d}, rankId={:d}, streamId={:d}, notifyId={:d}",
        __func__, thread, notifyIdx, threadRes.commId, curRank, threadRes.streamId, notifyIdVal);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

/**
 * @brief 在指定线程上使用默认超时等待通知信号（设备侧数据面接口）。
 *
 * 使用固定默认超时时间，采集本线程notifyIdx对应的等待任务。
 * 真实实现从 g_threadLaunchCtx 获取默认超时后委托给 HcommThreadNotifyWaitOnThread。
 * 对应 hcomm 仓同名接口。
 *
 * 桩生成NOTIFY_WAIT任务但不执行真实阻塞，超时值不写入现有任务元数据。
 *
 * @param thread 输入：等待 notify 的线程句柄。
 * @param notifyIdx 输入：本线程上的 notify 索引。
 * @return int32_t 成功返回HCCL_SUCCESS，参数或资源错误返回对应HcclResult。
 */
int32_t HcommThreadNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle thread, uint32_t notifyIdx)
{
    const uint32_t timeOut = DEFAULT_NOTIFY_WAIT_TIMEOUT;
    return HcommThreadNotifyWaitOnThread(thread, notifyIdx, timeOut);
}

/**
 * @brief 在指定线程上通过 channel 发送远端 notify 信号（设备侧数据面接口）。
 *
 * 向 dstThread 关联的 channel 对应的远端 remoteNotifyIdx 位置发送通知信号。
 * 对应 hcomm 仓同名接口。
 *
 * 桩生成本地rank到Channel远端rank的NOTIFY_RECORD任务。
 *
 * @param thread 输入：发起 notify 的线程句柄。
 * @param channel 输入：通道句柄。
 * @param remoteNotifyIdx 输入：远端 notify 索引。
 * @return int32_t 成功返回HCCL_SUCCESS，参数或资源错误返回对应HcclResult。
 */
int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    HCCL_VM_INFO("{}: thread={:d}, channel={:d}, remoteNotifyIdx={:d}", __func__, thread, channel, remoteNotifyIdx);

    // 【核心步骤 1】校验线程资源并获取所属通信域与 stream
    sim::runtime::HcclThread threadRes{};
    int32_t ret = GetThreadResource(thread, threadRes);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: GetThreadResource returned {:d}, thread={:d}, channel={:d}", __func__, ret, thread, channel);
        return ret;
    }

    // 【核心步骤 2】校验线程所属通信域与当前 rank 一致
    uint32_t curRank = 0;
    ret = GetThreadTaskPosition(threadRes, thread, curRank);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR("{} failed: GetThreadTaskPosition returned {:d}, thread={:d}", __func__, ret, thread);
        return ret;
    }

    // 【核心步骤 3】查询通道资源，校验已连接，获取远端 rank ID（通知接收方）
    sim::runtime::HcclChannel channelResource{};
    ret = GetChannelResource(channel, channelResource);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: GetChannelResource returned {:d}, thread={:d}, channel={:d}", __func__, ret, thread, channel);
        return ret;
    }
    // 校验 remoteRankId 在通信域 rankSize 范围内
    auto communicator = sim::runtime::Db::GetById<sim::runtime::Communicator>(threadRes.commId);
    if (!communicator.ok()) {
        HCCL_VM_ERROR(
            "{} failed: communicator={:d} not found, thread={:d}, ret={:d}", __func__, threadRes.commId, thread,
            static_cast<int32_t>(HCCL_E_NOT_FOUND));
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }

    if (channelResource.remoteRankId > std::numeric_limits<uint32_t>::max()
        || channelResource.remoteRankId >= communicator->rank_size) {
        HCCL_VM_ERROR(
            "{} failed: remoteRankId={:d} >= rankSize={:d}, thread={:d}, channel={:d}, ret={:d}", __func__,
            channelResource.remoteRankId, communicator->rank_size, thread, channel, static_cast<int32_t>(HCCL_E_PARA));
        return static_cast<int32_t>(HCCL_E_PARA);
    }
    uint32_t remoteRankId = static_cast<uint32_t>(channelResource.remoteRankId);

    // 【核心步骤 4】查找指向本端的反向 channel 上的 notifyId
    // remoteNotifyIdx 指向的是**对端** channel 上的索引，而非本端 channel。
    // Channel 创建流程：本端用本端 commId 创建指向对端 remoteRankId 的 channel，
    // 对端则用对端自己的 commId（与本端 commId 不同）创建一条 remoteRankId=curRank 的
    // 反向 channel。因此需要先通过 remoteRankId 在通信域表中找到对端自己的 commId，
    // 再用该 commId + 本端 channel 的 engine 精确匹配反向 channel（同 rank 对可能
    // 存在多个 engine 的 channel，不按 engine 过滤会命中旧 channel 的 notifyId）。
    // 跨 rank 两侧独立建链，本端就绪时对端反向 channel 可能尚未入库，此处按
    // 10ms 间隔轮询等待，总超时 10s（与 ctrl_comm_resource_stub 远端内存等待一致），
    // 超时后按原语义返回 HCCL_E_NOT_FOUND。
    uint64_t remoteCommId = 0;
    sim::runtime::HcclChannel reverseCh{};
    ret = WaitForReverseChannelResource(
        *communicator, curRank, remoteRankId, channelResource.engine, remoteCommId, reverseCh);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: reverse channel not found "
            "(remoteCommId={:d}, curRank={:d}, remoteRankId={:d}), thread={:d}, channel={:d}, ret={:d}",
            __func__, remoteCommId, curRank, remoteRankId, thread, channel, ret);
        return ret;
    }
    if (remoteNotifyIdx >= reverseCh.notifyNum || remoteNotifyIdx >= MAX_NOTIFY_PER_CHANNEL) {
        HCCL_VM_ERROR(
            "{} failed: remoteNotifyIdx={:d} out of range "
            "(reverseChannel.notifyNum={:d}, reverseChannel={:d}), thread={:d}, channel={:d}, ret={:d}",
            __func__, remoteNotifyIdx, reverseCh.notifyNum, reverseCh.id, thread, channel,
            static_cast<int32_t>(HCCL_E_PARA));
        return static_cast<int32_t>(HCCL_E_PARA);
    }
    uint64_t notifyIdVal = reverseCh.notifyId[remoteNotifyIdx];
    if (notifyIdVal == 0 || !sim::runtime::Db::GetById<sim::runtime::Notify>(notifyIdVal).ok()) {
        HCCL_VM_ERROR(
            "{} failed: notifyId={:d} not found in DB "
            "(reverseChannel={:d}, remoteNotifyIdx={:d}), thread={:d}, channel={:d}, ret={:d}",
            __func__, notifyIdVal, reverseCh.id, remoteNotifyIdx, thread, channel,
            static_cast<int32_t>(HCCL_E_NOT_FOUND));
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }

    // 【核心步骤 5】构建 NOTIFY_RECORD 任务元数据
    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    taskMetaData.commId = threadRes.commId;
    taskMetaData.rankId = curRank;
    taskMetaData.streamId = threadRes.streamId;

    taskMetaData.taskData.notify.notifyId = HcclSim::MakeNotifyTaskId(remoteRankId, notifyIdVal);
    taskMetaData.taskData.notify.notifyCount = static_cast<uint8_t>(remoteNotifyIdx);
    taskMetaData.taskData.notify.srcDeviceId = curRank;
    taskMetaData.taskData.notify.dstDeviceId = remoteRankId;

    // 【核心步骤 6】插入任务到集合
    const int32_t insertRet = InsertCollectedTask(taskMetaData);
    if (insertRet != 0) {
        HCCL_VM_ERROR(
            "{} failed: InsertTaskToCollectionDev failed, "
            "insertRet={:d}, ret={:d}",
            __func__, insertRet, static_cast<int32_t>(HCCL_E_INTERNAL));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, channel={:d}, "
        "remoteRank={:d}, remoteNotifyIdx={:d}, commId={:d}, rankId={:d}, streamId={:d}, notifyId={:d}",
        __func__, thread, channel, remoteRankId, remoteNotifyIdx, threadRes.commId, curRank, threadRes.streamId,
        notifyIdVal);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

/**
 * @brief 在指定线程上使用默认超时等待 channel 通知信号（设备侧数据面接口）。
 *
 * 使用固定默认超时时间，采集Channel上localNotifyIdx对应的等待任务。
 * 真实实现从 g_threadLaunchCtx 获取默认超时后委托给 HcommChannelNotifyWaitOnThread。
 * 对应 hcomm 仓同名接口。
 *
 * 桩生成Channel远端rank到本地rank的NOTIFY_WAIT任务，但不执行真实阻塞。
 *
 * @param thread 输入：等待 notify 的线程句柄。
 * @param channel 输入：通道句柄。
 * @param localNotifyIdx 输入：本地 notify 索引。
 * @return int32_t 成功返回HCCL_SUCCESS，参数或资源错误返回对应HcclResult。
 */
int32_t
HcommChannelNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx)
{
    const uint32_t timeOut = DEFAULT_NOTIFY_WAIT_TIMEOUT;
    HCCL_VM_INFO(
        "{}: thread={:d}, channel={:d}, "
        "localNotifyIdx={:d}, timeOut={:d}",
        __func__, thread, channel, localNotifyIdx, timeOut);

    // 【核心步骤 1】校验线程资源并获取所属通信域与 stream
    sim::runtime::HcclThread threadRes{};
    int32_t ret = GetThreadResource(thread, threadRes);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: GetThreadResource returned {:d}, thread={:d}, channel={:d}", __func__, ret, thread, channel);
        return ret;
    }

    // 【核心步骤 2】校验线程所属通信域与当前 rank 一致
    uint32_t curRank = 0;
    ret = GetThreadTaskPosition(threadRes, thread, curRank);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR("{} failed: GetThreadTaskPosition returned {:d}, thread={:d}", __func__, ret, thread);
        return ret;
    }

    // 【核心步骤 3】查询通道资源，校验已连接，获取远端 rank ID（信号发送方）
    sim::runtime::HcclChannel channelResource{};
    ret = GetChannelResource(channel, channelResource);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: GetChannelResource returned {:d}, thread={:d}, channel={:d}", __func__, ret, thread, channel);
        return ret;
    }
    // 校验 remoteRankId 在通信域 rankSize 范围内
    auto communicator = sim::runtime::Db::GetById<sim::runtime::Communicator>(threadRes.commId);
    if (!communicator.ok()) {
        HCCL_VM_ERROR(
            "{} failed: communicator={:d} not found, thread={:d}, ret={:d}", __func__, threadRes.commId, thread,
            static_cast<int32_t>(HCCL_E_NOT_FOUND));
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }
    if (channelResource.remoteRankId > std::numeric_limits<uint32_t>::max()
        || channelResource.remoteRankId >= communicator->rank_size) {
        HCCL_VM_ERROR(
            "{} failed: remoteRankId={:d} >= rankSize={:d}, thread={:d}, channel={:d}, ret={:d}", __func__,
            channelResource.remoteRankId, communicator->rank_size, thread, channel, static_cast<int32_t>(HCCL_E_PARA));
        return static_cast<int32_t>(HCCL_E_PARA);
    }
    uint32_t remoteRankId = static_cast<uint32_t>(channelResource.remoteRankId);

    // 【核心步骤 4】解析本地通知 ID 并校验在 DB 中存在
    if (localNotifyIdx >= channelResource.notifyNum || localNotifyIdx >= MAX_NOTIFY_PER_CHANNEL) {
        HCCL_VM_ERROR(
            "{} failed: localNotifyIdx={:d} out of range (notifyNum={:d}), thread={:d}, channel={:d}, ret={:d}",
            __func__, localNotifyIdx, channelResource.notifyNum, thread, channel, static_cast<int32_t>(HCCL_E_PARA));
        return static_cast<int32_t>(HCCL_E_PARA);
    }
    uint64_t notifyIdVal = channelResource.notifyId[localNotifyIdx];
    if (notifyIdVal == 0 || !sim::runtime::Db::GetById<sim::runtime::Notify>(notifyIdVal).ok()) {
        HCCL_VM_ERROR(
            "{} failed: notifyId={:d} not found in DB, thread={:d}, channel={:d}, localNotifyIdx={:d}, ret={:d}",
            __func__, notifyIdVal, thread, channel, localNotifyIdx, static_cast<int32_t>(HCCL_E_NOT_FOUND));
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }

    // 【核心步骤 5】构建 NOTIFY_WAIT 任务元数据
    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    taskMetaData.commId = threadRes.commId;
    taskMetaData.rankId = curRank;
    taskMetaData.streamId = threadRes.streamId;

    taskMetaData.taskData.notify.notifyId = HcclSim::MakeNotifyTaskId(curRank, notifyIdVal);
    taskMetaData.taskData.notify.notifyCount = static_cast<uint8_t>(localNotifyIdx);
    taskMetaData.taskData.notify.srcDeviceId = remoteRankId;
    taskMetaData.taskData.notify.dstDeviceId = curRank;

    // 【核心步骤 6】插入任务到集合
    const int32_t insertRet = InsertCollectedTask(taskMetaData);
    if (insertRet != 0) {
        HCCL_VM_ERROR(
            "{} failed: InsertTaskToCollectionDev failed, "
            "insertRet={:d}, ret={:d}",
            __func__, insertRet, static_cast<int32_t>(HCCL_E_INTERNAL));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, channel={:d}, "
        "remoteRank={:d}, localNotifyIdx={:d}, commId={:d}, rankId={:d}, streamId={:d}, notifyId={:d}",
        __func__, thread, channel, remoteRankId, localNotifyIdx, threadRes.commId, curRank, threadRes.streamId,
        notifyIdVal);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

/**
 * @brief 在指定线程上执行批量传输（设备侧数据面接口）。
 *
 * 通过指定 channel 在线程上批量执行 read/write/reduce/notify 等传输操作。
 * transferDescs 数组中每个元素描述一个传输项（类型、源地址、目的地址、长度等）。
 * 对应 hcomm 仓同名接口。
 *
 * 桩先校验全部描述符，再按顺序生成MEM_CPY、REDUCE和NOTIFY_RECORD任务。
 *
 * @param thread 输入：执行传输的线程句柄。
 * @param channel 输入：通道句柄。
 * @param transferDescs 输入：批量传输描述符数组。
 * @param transferDescNum 输入：描述符数组元素个数。
 * @return int32_t 成功返回HCCL_SUCCESS，非法描述符返回对应HcclResult。
 */
int32_t HcommBatchTransferOnThread(
    ThreadHandle thread, ChannelHandle channel, const HcommBatchTransferDesc* transferDescs, uint32_t transferDescNum)
{
    HCCL_VM_INFO("{}: thread={:d}, channel={:d}, transferDescNum={:d}", __func__, thread, channel, transferDescNum);
    // 内部路径先全量校验并展开描述符，再按顺序写入任务表。
    return CollectBatchTransfers(thread, channel, transferDescs, transferDescNum);
}

/**
 * @brief 在指定线程上执行批量传输（设备侧 hccl wrapper 接口）。
 *
 * hccl 仓 hcomm_primitives_dl 中的 wrapper 接口，签名与 HcommBatchTransferOnThread 一致，
 * 但使用 HcclHcommBatchTransferDesc 描述符类型。libhccl.so 通过此 wrapper 委托到 hcomm 实现。
 * 对应 hccl 仓同名接口。
 *
 * 桩校验HCCL私有ABI后复用内部批量转换逻辑，不调用另一个公开桩函数。
 *
 * @param thread 输入：执行传输的线程句柄。
 * @param channel 输入：通道句柄。
 * @param transferDescs 输入：批量传输描述符数组。
 * @param transferDescNum 输入：描述符数组元素个数。
 * @return int32_t 成功返回HCCL_SUCCESS，非法描述符返回对应HcclResult。
 */
int32_t HcclHcommBatchTransferOnThread(
    ThreadHandle thread, ChannelHandle channel, const HcclHcommBatchTransferDesc* transferDescs,
    uint32_t transferDescNum)
{
    HCCL_VM_INFO("{}: thread={:d}, channel={:d}, transferDescNum={:d}", __func__, thread, channel, transferDescNum);
    // HCCL私有描述符已由静态断言证明与HComm ABI一致，调用内部采集路径。
    return CollectBatchTransfers(
        thread, channel, reinterpret_cast<const HcommBatchTransferDesc*>(transferDescs), transferDescNum);
}

/**
 * @brief 在指定线程上通过指定通道执行跨节点内存写入（设备侧数据面接口）。
 *
 * 将本端 src 指向的 len 字节数据通过 channel 写入远端 rank 的 dst。
 * 对应 hcomm 仓同名接口。
 *
 * 桩按 Write 语义（src=本端, dst=远端）生成 MEM_CPY 任务，不执行真实内存写入，
 * 数据正确性由 Runner 依据任务元数据模拟执行。
 *
 * @param thread  输入：执行写入的线程句柄。
 * @param channel 输入：通道句柄（含远端 rank 信息）。
 * @param dst     输入：远端 device 内存地址（对端）。
 * @param src     输入：本地 device 内存地址（本端）。
 * @param len     输入：待写入字节数。
 * @return int32_t 成功返回HCCL_SUCCESS，参数或资源错误返回对应HcclResult。
 */
int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    if (dst == nullptr || src == nullptr || len == 0u) {
        HCCL_VM_ERROR("{}: invalid parameter, dst={:p}, src={:p}, len={:d}", __func__, dst, src, len);
        return static_cast<int32_t>(HCCL_E_PARA);
    }

    HCCL_VM_INFO(
        "{}: thread={:d}, channel={:d}, dst={:p}, src={:p}, len={:d}", __func__, thread, channel, dst, src, len);

    // 【核心步骤 1】解析线程/通道资源，获取当前 rank 与远端 rank
    sim::runtime::HcclThread threadRes{};
    sim::runtime::HcclChannel channelRes{};
    uint32_t curRank = 0;
    uint32_t remoteRankId = 0;
    int32_t ret = GetChannelTaskContext(thread, channel, threadRes, channelRes, curRank, remoteRankId);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: GetChannelTaskContext returned {:d}, thread={:d}, channel={:d}", __func__, ret, thread,
            channel);
        return ret;
    }

    // 【核心步骤 2】构造 MEM_CPY 任务（Write 语义：src=本端, dst=远端）
    const HcclTaskMetaData task = MakeCopyTask(threadRes, curRank, curRank, src, remoteRankId, dst, len);

    // 【核心步骤 3】插入任务到集合
    ret = InsertCollectedTask(task);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: InsertCollectedTask returned {:d}, thread={:d}, channel={:d}", __func__, ret, thread, channel);
        return ret;
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, channel={:d}, commId={:d}, rankId={:d}, "
        "streamId={:d}, remoteRank={:d}, src={:p}, dst={:p}, len={:d}",
        __func__, thread, channel, threadRes.commId, curRank, threadRes.streamId, remoteRankId, src, dst, len);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

/**
 * @brief 在指定线程上通过指定通道执行跨节点内存读取（设备侧数据面接口）。
 *
 * 将远端 rank 的 src 指向的 len 字节数据通过 channel 读取到本端 dst。
 * 对应 hcomm 仓同名接口。
 *
 * 桩按 Read 语义（src=远端, dst=本端）生成 MEM_CPY 任务，不执行真实内存读取，
 * 数据正确性由 Runner 依据任务元数据模拟执行。
 *
 * @param thread  输入：执行读取的线程句柄。
 * @param channel 输入：通道句柄（含远端 rank 信息）。
 * @param dst     输入：本地 device 内存地址（本端）。
 * @param src     输入：远端 device 内存地址（对端）。
 * @param len     输入：待读取字节数。
 * @return int32_t 成功返回HCCL_SUCCESS，参数或资源错误返回对应HcclResult。
 */
int32_t HcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    if (dst == nullptr || src == nullptr || len == 0u) {
        HCCL_VM_ERROR("{}: invalid parameter, dst={:p}, src={:p}, len={:d}", __func__, dst, src, len);
        return static_cast<int32_t>(HCCL_E_PARA);
    }

    HCCL_VM_INFO(
        "{}: thread={:d}, channel={:d}, dst={:p}, src={:p}, len={:d}", __func__, thread, channel, dst, src, len);

    // 【核心步骤 1】解析线程/通道资源，获取当前 rank 与远端 rank
    sim::runtime::HcclThread threadRes{};
    sim::runtime::HcclChannel channelRes{};
    uint32_t curRank = 0;
    uint32_t remoteRankId = 0;
    int32_t ret = GetChannelTaskContext(thread, channel, threadRes, channelRes, curRank, remoteRankId);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: GetChannelTaskContext returned {:d}, thread={:d}, channel={:d}", __func__, ret, thread,
            channel);
        return ret;
    }

    // 【核心步骤 2】构造 MEM_CPY 任务（Read 语义：src=远端, dst=本端）
    const HcclTaskMetaData task = MakeCopyTask(threadRes, curRank, remoteRankId, src, curRank, dst, len);

    // 【核心步骤 3】插入任务到集合
    ret = InsertCollectedTask(task);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: InsertCollectedTask returned {:d}, thread={:d}, channel={:d}", __func__, ret, thread, channel);
        return ret;
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, channel={:d}, commId={:d}, rankId={:d}, "
        "streamId={:d}, remoteRank={:d}, src={:p}, dst={:p}, len={:d}",
        __func__, thread, channel, threadRes.commId, curRank, threadRes.streamId, remoteRankId, src, dst, len);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

/**
 * @brief 在指定线程上通过指定通道执行跨节点归约写入（设备侧数据面接口）。
 *
 * 将本端 src 指向的 count 个 dataType 元素通过 channel 与远端 rank 的 dst
 * 做归约操作，结果输出到远端 dst。对应 hcomm 仓同名接口。
 *
 * 桩按 WriteReduce 语义（src=本端, dst=远端）生成 REDUCE 任务，不执行真实归约，
 * 数据正确性由 Runner 依据任务元数据模拟执行。
 *
 * @param thread   输入：执行归约写入的线程句柄。
 * @param channel  输入：通道句柄（含远端 rank 信息）。
 * @param dst      输入：远端 device 内存地址（对端）。
 * @param src      输入：本地 device 内存地址（本端）。
 * @param count    输入：元素个数。
 * @param dataType 输入：元素数据类型。
 * @param reduceOp 输入：归约操作类型。
 * @return int32_t 成功返回HCCL_SUCCESS，参数或资源错误返回对应HcclResult。
 */
int32_t HcommWriteReduceOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    if (dst == nullptr || src == nullptr || count == 0u) {
        HCCL_VM_ERROR("{}: invalid parameter, dst={:p}, src={:p}, count={:d}", __func__, dst, src, count);
        return static_cast<int32_t>(HCCL_E_PARA);
    }

    HCCL_VM_INFO(
        "{}: thread={:d}, channel={:d}, dst={:p}, src={:p}, "
        "count={:d}, dataType={:d}, reduceOp={:d}",
        __func__, thread, channel, dst, src, count, dataType, reduceOp);

    // 【核心步骤 1】按A5支持矩阵校验数据类型、归约操作，并取得元素字节数
    uint32_t typeSize = 0;
    int32_t ret = ValidateA5Reduce(dataType, reduceOp, count, typeSize);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: ValidateA5Reduce returned {:d}, thread={:d}, dataType={:d}, reduceOp={:d}", __func__, ret,
            thread, dataType, reduceOp);
        return ret;
    }

    // 【核心步骤 2】解析线程/通道资源，获取当前 rank 与远端 rank
    sim::runtime::HcclThread threadRes{};
    sim::runtime::HcclChannel channelRes{};
    uint32_t curRank = 0;
    uint32_t remoteRankId = 0;
    ret = GetChannelTaskContext(thread, channel, threadRes, channelRes, curRank, remoteRankId);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: GetChannelTaskContext returned {:d}, thread={:d}, channel={:d}", __func__, ret, thread,
            channel);
        return ret;
    }

    // 【核心步骤 3】构造 REDUCE 任务（WriteReduce 语义：src=本端, dst=远端）
    const HcclTaskMetaData task
        = MakeReduceTask(threadRes, curRank, curRank, src, remoteRankId, dst, count * typeSize, dataType, reduceOp);

    // 【核心步骤 4】插入任务到集合
    ret = InsertCollectedTask(task);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: InsertCollectedTask returned {:d}, thread={:d}, channel={:d}", __func__, ret, thread, channel);
        return ret;
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, channel={:d}, commId={:d}, rankId={:d}, "
        "streamId={:d}, remoteRank={:d}, src={:p}, dst={:p}, count={:d}",
        __func__, thread, channel, threadRes.commId, curRank, threadRes.streamId, remoteRankId, src, dst, count);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

/**
 * @brief 在指定线程上通过指定通道执行跨节点归约读取（设备侧数据面接口）。
 *
 * 从远端 rank 的 src 读取 count 个 dataType 元素，与本端 dst 的数据做归约操作，
 * 结果输出到本端 dst。对应 hcomm 仓同名接口。
 *
 * 桩按 ReadReduce 语义（src=远端, dst=本端）生成 REDUCE 任务，不执行真实归约，
 * 数据正确性由 Runner 依据任务元数据模拟执行。
 *
 * @param thread   输入：执行归约读取的线程句柄。
 * @param channel  输入：通道句柄（含远端 rank 信息）。
 * @param dst      输入：本地 device 内存地址（本端）。
 * @param src      输入：远端 device 内存地址（对端）。
 * @param count    输入：元素个数。
 * @param dataType 输入：元素数据类型。
 * @param reduceOp 输入：归约操作类型。
 * @return int32_t 成功返回HCCL_SUCCESS，参数或资源错误返回对应HcclResult。
 */
int32_t HcommReadReduceOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    if (dst == nullptr || src == nullptr || count == 0u) {
        HCCL_VM_ERROR("{}: invalid parameter, dst={:p}, src={:p}, count={:d}", __func__, dst, src, count);
        return static_cast<int32_t>(HCCL_E_PARA);
    }

    HCCL_VM_INFO(
        "{}: thread={:d}, channel={:d}, dst={:p}, src={:p}, "
        "count={:d}, dataType={:d}, reduceOp={:d}",
        __func__, thread, channel, dst, src, count, dataType, reduceOp);

    // 【核心步骤 1】按A5支持矩阵校验数据类型、归约操作，并取得元素字节数
    uint32_t typeSize = 0;
    int32_t ret = ValidateA5Reduce(dataType, reduceOp, count, typeSize);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: ValidateA5Reduce returned {:d}, thread={:d}, dataType={:d}, reduceOp={:d}", __func__, ret,
            thread, dataType, reduceOp);
        return ret;
    }

    // 【核心步骤 2】解析线程/通道资源，获取当前 rank 与远端 rank
    sim::runtime::HcclThread threadRes{};
    sim::runtime::HcclChannel channelRes{};
    uint32_t curRank = 0;
    uint32_t remoteRankId = 0;
    ret = GetChannelTaskContext(thread, channel, threadRes, channelRes, curRank, remoteRankId);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: GetChannelTaskContext returned {:d}, thread={:d}, channel={:d}", __func__, ret, thread,
            channel);
        return ret;
    }

    // 【核心步骤 3】构造 REDUCE 任务（ReadReduce 语义：src=远端, dst=本端）
    const HcclTaskMetaData task
        = MakeReduceTask(threadRes, curRank, remoteRankId, src, curRank, dst, count * typeSize, dataType, reduceOp);

    // 【核心步骤 4】插入任务到集合
    ret = InsertCollectedTask(task);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: InsertCollectedTask returned {:d}, thread={:d}, channel={:d}", __func__, ret, thread, channel);
        return ret;
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, channel={:d}, commId={:d}, rankId={:d}, "
        "streamId={:d}, remoteRank={:d}, src={:p}, dst={:p}, count={:d}",
        __func__, thread, channel, threadRes.commId, curRank, threadRes.streamId, remoteRankId, src, dst, count);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

/**
 * @brief 在指定线程上执行本地拷贝（设备侧数据面接口）。
 *
 * 将 src 指向的 len 字节数据拷贝到 dst，在本地引擎内执行（不跨 rank）。
 * 对应 hcomm 仓同名接口。
 *
 * 桩生成同rank的MEM_CPY任务，不执行真实内存拷贝。
 *
 * @param thread 输入：执行拷贝的线程句柄。
 * @param dst 输入：目的地址。
 * @param src 输入：源地址。
 * @param len 输入：拷贝的字节数。
 * @return int32_t 成功返回HCCL_SUCCESS，参数或资源错误返回对应HcclResult。
 */
int32_t HcommLocalCopyOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    if (dst == nullptr || src == nullptr || len == 0u) {
        HCCL_VM_ERROR("{}: invalid parameter, dst={:p}, src={:p}, len={:d}", __func__, dst, src, len);
        return static_cast<int32_t>(HCCL_E_PARA);
    }

    HCCL_VM_INFO("{}: thread={:d}, dst={:p}, src={:p}, len={:d}", __func__, thread, dst, src, len);

    // 【核心步骤 1】获取当前 rank ID
    uint32_t curRank = 0;
    auto rankRet = GetCurRankId(&curRank);
    if (rankRet != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("{} failed: GetCurRankId failed, ret={:d}", __func__, static_cast<int32_t>(HCCL_E_INTERNAL));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    // 【核心步骤 2】查询线程记录，获取所属通信域与关联的 stream
    // 与 notify 类接口（GetThreadResource 强校验）不同：本地操作不跨 rank，允许线程记录
    // 缺失并降级为 commId=0/streamId=0 继续采集，弱校验即可满足 Checker 的地址范围分析。
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
    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::MEM_CPY;
    taskMetaData.commId = commId;
    taskMetaData.rankId = curRank;
    taskMetaData.streamId = streamId;

    taskMetaData.taskData.transMem.srcDeviceId = curRank;
    taskMetaData.taskData.transMem.srcOffset = TransLocalAddrToVirtual(reinterpret_cast<uint64_t>(src));
    taskMetaData.taskData.transMem.dstDeviceId = curRank;
    taskMetaData.taskData.transMem.dstOffset = TransLocalAddrToVirtual(reinterpret_cast<uint64_t>(dst));
    taskMetaData.taskData.transMem.len = len;

    // 【核心步骤 4】插入任务到集合
    const int32_t insertRet = InsertCollectedTask(taskMetaData);
    if (insertRet != 0) {
        HCCL_VM_ERROR(
            "{} failed: InsertTaskToCollectionDev failed, "
            "insertRet={:d}, ret={:d}",
            __func__, insertRet, static_cast<int32_t>(HCCL_E_INTERNAL));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, commId={:d}, rankId={:d}, "
        "streamId={:d}, src={:p}, dst={:p}, len={:d}",
        __func__, thread, commId, curRank, streamId, src, dst, len);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

/**
 * @brief 在指定线程上执行本地归约（设备侧数据面接口）。
 *
 * 将 src 指向的 count 个 dataType 元素归约到 dst，在本地引擎内执行（不跨 rank）。
 * 对应 hcomm 仓同名接口。
 *
 * 桩按A5支持矩阵校验后生成同rank的REDUCE任务，不执行真实归约。
 *
 * @param thread 输入：执行归约的线程句柄。
 * @param dst 输入：目的地址。
 * @param src 输入：源地址。
 * @param count 输入：元素个数。
 * @param dataType 输入：数据类型。
 * @param reduceOp 输入：归约操作类型。
 * @return int32_t 成功返回HCCL_SUCCESS，参数或资源错误返回对应HcclResult。
 */
int32_t HcommLocalReduceOnThread(
    ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    if (dst == nullptr || src == nullptr || count == 0u) {
        HCCL_VM_ERROR("{}: invalid parameter, dst={:p}, src={:p}, count={:d}", __func__, dst, src, count);
        return static_cast<int32_t>(HCCL_E_PARA);
    }

    HCCL_VM_INFO(
        "{}: thread={:d}, dst={:p}, src={:p}, count={:d}, dataType={:d}, reduceOp={:d}", __func__, thread, dst, src,
        count, dataType, reduceOp);

    // 【核心步骤 1】按A5支持矩阵校验数据类型、归约操作，并取得元素字节数
    uint32_t typeSize = 0;
    int32_t ret = ValidateA5Reduce(dataType, reduceOp, count, typeSize);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR(
            "{} failed: ValidateA5Reduce returned {:d}, thread={:d}, dataType={:d}, reduceOp={:d}", __func__, ret,
            thread, dataType, reduceOp);
        return ret;
    }

    // 【核心步骤 2】获取当前 rank ID
    uint32_t curRank = 0;
    auto rankRet = GetCurRankId(&curRank);
    if (rankRet != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("{} failed: GetCurRankId failed, ret={:d}", __func__, static_cast<int32_t>(HCCL_E_INTERNAL));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    // 【核心步骤 3】查询线程记录，获取所属通信域与关联的 stream
    // 同 LocalCopy：本地归约不跨 rank，允许线程记录缺失并降级为 commId=0/streamId=0 继续采集。
    uint64_t commId = 0u;
    uint64_t streamId = 0u;
    auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(thread);
    if (optThread.ok()) {
        commId = optThread->commId;
        streamId = optThread->streamId;
    } else {
        HCCL_VM_WARN("{}: thread {:d} not found", __func__, thread);
    }

    // 【核心步骤 4】构造 REDUCE 任务元数据
    // Local Reduce 语义：src 和 dst 都在同一节点，srcRankId == dstRankId == curRank
    // dataCount 存储字节长度（count * typeSize），便于 Checker 直接做地址范围分析
    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::REDUCE;
    taskMetaData.commId = commId;
    taskMetaData.rankId = curRank;
    taskMetaData.streamId = streamId;

    taskMetaData.taskData.reduce.srcDeviceId = curRank;
    taskMetaData.taskData.reduce.srcOffset = TransLocalAddrToVirtual(reinterpret_cast<uint64_t>(src));
    taskMetaData.taskData.reduce.dstDeviceId = curRank;
    taskMetaData.taskData.reduce.dstOffset = TransLocalAddrToVirtual(reinterpret_cast<uint64_t>(dst));
    taskMetaData.taskData.reduce.dataCount = count * typeSize;
    taskMetaData.taskData.reduce.dataType = static_cast<uint8_t>(dataType);
    taskMetaData.taskData.reduce.reduceOp = static_cast<uint8_t>(reduceOp);

    // 【核心步骤 5】插入任务到集合
    const int32_t insertRet = InsertCollectedTask(taskMetaData);
    if (insertRet != 0) {
        HCCL_VM_ERROR(
            "{} failed: InsertTaskToCollectionDev failed, "
            "insertRet={:d}, ret={:d}",
            __func__, insertRet, static_cast<int32_t>(HCCL_E_INTERNAL));
        return static_cast<int32_t>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, thread={:d}, commId={:d}, rankId={:d}, "
        "streamId={:d}, count={:d}, dataType={:d}, reduceOp={:d}",
        __func__, thread, commId, curRank, streamId, count, dataType, reduceOp);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

/**
 * @brief 在指定线程上发送 aclrt notify 信号（设备侧数据面接口）。
 *
 * 通过 aclrt 机制向目标 notifyId 发送通知信号。
 * 对应 hcomm 仓同名接口。
 *
 * 桩校验ACL Notify资源并生成同rank的NOTIFY_RECORD任务。
 *
 * @param thread 输入：发起 notify 的线程句柄。
 * @param dstNotifyId 输入：目标 notify 标识。
 * @return int32_t 成功返回HCCL_SUCCESS，参数或资源错误返回对应HcclResult。
 */
int32_t HcommAclrtNotifyRecordOnThread(ThreadHandle thread, uint64_t dstNotifyId)
{
    HCCL_VM_INFO("{}: thread={:d}, dstNotifyId={:d}", __func__, thread, dstNotifyId);

    // 【核心步骤 1】解析执行record任务的线程和任务位置。
    sim::runtime::HcclThread threadResource{};
    int32_t ret = GetThreadResource(thread, threadResource);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR("{} failed: GetThreadResource returned {:d}, thread={:d}", __func__, ret, thread);
        return ret;
    }
    uint32_t rankId = 0;
    ret = GetThreadTaskPosition(threadResource, thread, rankId);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR("{} failed: GetThreadTaskPosition returned {:d}, thread={:d}", __func__, ret, thread);
        return ret;
    }

    // 【核心步骤 2】校验ACL Notify ID对应数据库中的稳定Notify资源。
    if (dstNotifyId == 0) {
        HCCL_VM_ERROR(
            "{} failed: thread={:d}, notifyId=0, ret={:d}", __func__, thread, static_cast<int32_t>(HCCL_E_PTR));
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    if (dstNotifyId > std::numeric_limits<uint32_t>::max()) {
        HCCL_VM_ERROR(
            "{} failed: thread={:d}, notifyId={:d} exceeds uint32_t, ret={:d}", __func__, thread, dstNotifyId,
            static_cast<int32_t>(HCCL_E_PARA));
        return static_cast<int32_t>(HCCL_E_PARA);
    }
    if (!sim::runtime::Db::GetById<sim::runtime::Notify>(dstNotifyId).ok()) {
        HCCL_VM_ERROR(
            "{} failed: thread={:d}, notifyId={:d} not found, ret={:d}", __func__, thread, dstNotifyId,
            static_cast<int32_t>(HCCL_E_NOT_FOUND));
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }

    // 【核心步骤 3】构造NOTIFY_RECORD任务并入库。
    // ACL notify没有显式远端rank，若notifyId命中某Channel的notifyId数组，
    // 则说明该notify关联跨节点通信，dstRankId记为该Channel的远端rank。
    uint32_t dstRankId = rankId;
    uint32_t remoteRankId = 0;
    if (FindRemoteRankByNotifyId(threadResource.commId, dstNotifyId, remoteRankId)) {
        dstRankId = remoteRankId;
    } else {
        HCCL_VM_ERROR(
            "{} failed: notifyId={:d} not bound to any cross-node channel, "
            "ret={:d}",
            __func__, dstNotifyId, static_cast<int32_t>(HCCL_E_NOT_FOUND));
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }
    HcclTaskMetaData task
        = MakeNotifyTask(HccLTaskMetaType::NOTIFY_RECORD, threadResource, rankId, rankId, dstRankId, dstNotifyId);
    ret = InsertCollectedTask(task);
    if (ret == static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_INFO(
            "{} success, taskType=NOTIFY_RECORD, thread={:d}, notifyId={:d}, rankId={:d}", __func__, thread,
            dstNotifyId, rankId);
    }
    return ret;
}

/**
 * @brief 在指定线程上等待 aclrt notify 信号（设备侧数据面接口）。
 *
 * 通过 aclrt 机制阻塞等待指定 notifyId 的通知信号到达。
 * 对应 hcomm 仓同名接口。
 *
 * 桩校验ACL Notify资源并生成同rank的NOTIFY_WAIT任务，不执行真实阻塞。
 *
 * @param thread 输入：等待 notify 的线程句柄。
 * @param notifyId 输入：等待的 notify 标识。
 * @param timeOut 输入：超时时间（秒），0 表示永久等待。
 * @return int32_t 成功返回HCCL_SUCCESS，参数或资源错误返回对应HcclResult。
 */
int32_t HcommAclrtNotifyWaitOnThread(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut)
{
    HCCL_VM_INFO("{}: thread={:d}, notifyId={:d}, timeOut={:d}", __func__, thread, notifyId, timeOut);

    // 【核心步骤 1】解析执行wait任务的线程和任务位置。
    sim::runtime::HcclThread threadResource{};
    int32_t ret = GetThreadResource(thread, threadResource);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR("{} failed: GetThreadResource returned {:d}, thread={:d}", __func__, ret, thread);
        return ret;
    }
    uint32_t rankId = 0;
    ret = GetThreadTaskPosition(threadResource, thread, rankId);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR("{} failed: GetThreadTaskPosition returned {:d}, thread={:d}", __func__, ret, thread);
        return ret;
    }

    // 【核心步骤 2】校验ACL Notify ID，不修改Notify执行状态。
    if (notifyId == 0) {
        HCCL_VM_ERROR("{}: thread={:d}, notifyId=0, ret={:d}", __func__, thread, static_cast<int32_t>(HCCL_E_PTR));
        return static_cast<int32_t>(HCCL_E_PTR);
    }
    if (notifyId > std::numeric_limits<uint32_t>::max()) {
        HCCL_VM_ERROR(
            "{}: thread={:d}, notifyId={:d} exceeds uint32_t, ret={:d}", __func__, thread, notifyId,
            static_cast<int32_t>(HCCL_E_PARA));
        return static_cast<int32_t>(HCCL_E_PARA);
    }
    if (!sim::runtime::Db::GetById<sim::runtime::Notify>(notifyId).ok()) {
        HCCL_VM_ERROR(
            "{}: thread={:d}, notifyId={:d} not found, ret={:d}", __func__, thread, notifyId,
            static_cast<int32_t>(HCCL_E_NOT_FOUND));
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }

    // 【核心步骤 3】构造NOTIFY_WAIT任务并入库，timeout单位为秒。
    // 与 record 对称：若notifyId命中某Channel，则srcRankId记为该Channel的远端rank
    // （表示等待远端通过该channel发来的notify）。
    uint32_t srcRankId = rankId;
    uint32_t remoteRankId = 0;
    if (FindRemoteRankByNotifyId(threadResource.commId, notifyId, remoteRankId)) {
        srcRankId = remoteRankId;
    } else {
        HCCL_VM_ERROR(
            "{} failed: notifyId={:d} not bound to any cross-node channel, ret={:d}", __func__, notifyId,
            static_cast<int32_t>(HCCL_E_NOT_FOUND));
        return static_cast<int32_t>(HCCL_E_NOT_FOUND);
    }
    HcclTaskMetaData task
        = MakeNotifyTask(HccLTaskMetaType::NOTIFY_WAIT, threadResource, rankId, srcRankId, rankId, notifyId);
    ret = InsertCollectedTask(task);
    if (ret == static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_INFO(
            "{} success, taskType=NOTIFY_WAIT, thread={:d}, notifyId={:d}, timeOutSeconds={:d}, rankId={:d}", __func__,
            thread, notifyId, timeOut, rankId);
    }
    return ret;
}

/**
 * @brief 设置线程资源获取超时时间（设备侧数据面接口）。
 *
 * 配置后续 HcommThreadResAcquire 系列接口的超时阈值。
 * 对应 hcomm 仓同名接口。
 *
 * 桩按真实实现校验并截断float，不保存状态（模拟器不感知超时）。
 *
 * @param timeOut 输入：超时时间（秒）。
 * @return int32_t 成功返回HCCL_SUCCESS，非法超时返回HCCL_E_PARA。
 */
int32_t HcommThreadResAcquireTimeOut(float timeOut)
{
    HCCL_VM_INFO("{}: timeOut={}", __func__, timeOut);

    // 【核心步骤 1】校验float范围并按真实实现向零截断。
    uint32_t normalized = 0;
    int32_t ret = NormalizeTimeout(timeOut, normalized);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR("{} failed: NormalizeTimeout returned {:d}, timeOut={}", __func__, ret, timeOut);
        return ret;
    }

    HCCL_VM_INFO("{} success, timeOutSeconds={:d}", __func__, normalized);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

/**
 * @brief 设置 notify 等待超时时间（设备侧数据面接口）。
 *
 * 配置后续 HcommThreadNotifyWaitOnThread 系列接口的默认超时阈值。
 * 对应 hcomm 仓同名接口。
 *
 * 桩按真实实现校验并截断float，不保存状态（模拟器不感知超时）。
 *
 * @param timeOut 输入：超时时间（秒）。
 * @return int32_t 成功返回HCCL_SUCCESS，非法超时返回HCCL_E_PARA。
 */
int32_t HcommSetNotifyWaitTimeOut(float timeOut)
{
    HCCL_VM_INFO("{}: timeOut={}", __func__, timeOut);

    // 【核心步骤 1】校验float范围并按真实实现向零截断。
    uint32_t normalized = 0;
    int32_t ret = NormalizeTimeout(timeOut, normalized);
    if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
        HCCL_VM_ERROR("{} failed: NormalizeTimeout returned {:d}, timeOut={}", __func__, ret, timeOut);
        return ret;
    }

    HCCL_VM_INFO("{} success, timeOutSeconds={:d}", __func__, normalized);
    return static_cast<int32_t>(HCCL_SUCCESS);
}

#ifdef __cplusplus
}
#endif
