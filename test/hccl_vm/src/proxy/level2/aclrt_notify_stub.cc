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
#define HCCL_VM_MODULE "NOTIFY_STUB"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <utility>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "store_sim_store_pub.h"
#include "sim_log.h"
#include "runtime/base.h"
#include "runtime/event.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "storage/table_access.h"
#include "sim_capacity_limits.h"
#include "store_sim_run_mode.h"
#include "graph_capture.h"

namespace {

class DeviceNotifyIdAllocator {
public:
    bool Allocate(uint32_t& deviceNotifyId)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (nextId_ >= HCCL_VM_DEVICE_NOTIFY_ID_CAPACITY) {
            if (!freeList_.empty()) {
                deviceNotifyId = freeList_.back();
                freeList_.pop_back();
                return true;
            }
            return false;
        }
        deviceNotifyId = nextId_++;
        return true;
    }

    void Release(uint32_t deviceNotifyId)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        freeList_.push_back(deviceNotifyId);
    }

private:
    std::mutex mutex_;
    uint32_t nextId_ = 0;
    std::vector<uint32_t> freeList_;
};

DeviceNotifyIdAllocator g_deviceNotifyIds;

bool GetNotifyTaskId(aclrtNotify notify, uint64_t& notifyTaskId)
{
    uint64_t storageId = (uint32_t)(uintptr_t)notify;
    auto record = sim::runtime::Db::GetById<sim::runtime::Notify>(storageId);
    if (!record.ok()) {
        HCCL_VM_ERROR("notify not found:{:d}", storageId);
        return false;
    }
    notifyTaskId = record->device_notify_seq;
    return true;
}

// 解析 notify 任务的通信域身份（commId 兼入参/出参，为 0 时回退默认通信域）
static bool ResolveNotifyCommIdentity(uint64_t& commId, uint32_t deviceId, uint32_t& rankId)
{
    if (commId != 0) {
        return sim::runtime::GetCommRankByDeviceId(commId, deviceId, rankId);
    }

    // 未经过算子入口劫持（如 HcclBarrier）时 commId 为 0：
    // 按 deviceId 查 Communicator 表取首行作默认身份，commId 回写供任务标注
    auto members = sim::runtime::Db::GetByPred<sim::runtime::Communicator>(
        HcclSim::Storage::Eq(&sim::runtime::Communicator::device_id, deviceId));
    if (!members.ok() || !members.value.has_value() || members->empty()) {
        HCCL_VM_ERROR("notify fallback found no communicator, deviceId={}", deviceId);
        return false;
    }
    commId = members->front().id;
    rankId = members->front().rank_id;
    HCCL_VM_WARN(
        "notify task falls back to default communicator, commId={}, rankId={}, deviceId={}", commId, rankId, deviceId);
    return true;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void PrintTaskMetaData(const HcclTaskMetaData& taskMeta)
{
    pid_t pid = getpid();
    HcclSim::Eid rmEidTmp{};
    HcclSim::Eid lcEidTmp{};
    (void)memcpy(rmEidTmp.raw, taskMeta.rmEid, HcclSim::URMA_EID_LEN);
    (void)memcpy(lcEidTmp.raw, taskMeta.lcEid, HcclSim::URMA_EID_LEN);
    const std::string eidInfo = ", rmEid[" + HcclSim::IpAddress(rmEidTmp).EidToHexString() + "], lcEid["
                                + HcclSim::IpAddress(lcEidTmp).EidToHexString() + "]";
    switch (taskMeta.taskType) {
        case HccLTaskMetaType::MEM_CPY:
            HCCL_VM_INFO(
                "pid[{}]: deviceId[{}], commId[{}], rankId[{}], streamId[{}], taskType[MEM_CPY], jettyId[{}], "
                "srcOffset[{}], dstOffset[{}], len[{}], srcDeviceId[{}], dstDeviceId[{}]{}",
                pid, taskMeta.deviceId, taskMeta.commId, taskMeta.rankId, taskMeta.streamId, taskMeta.jettyId,
                taskMeta.taskData.transMem.srcOffset, taskMeta.taskData.transMem.dstOffset,
                taskMeta.taskData.transMem.len, taskMeta.taskData.transMem.srcDeviceId,
                taskMeta.taskData.transMem.dstDeviceId, eidInfo);
            break;
        case HccLTaskMetaType::REDUCE:
            HCCL_VM_INFO(
                "pid[{}]: deviceId[{}], commId[{}], rankId[{}], streamId[{}], taskType[REDUCE], jettyId[{}], "
                "srcOffset[{}], dstOffset[{}], len[{}], srcDeviceId[{}], dstDeviceId[{}], reduceOp[{}], dataType[{}]{}",
                pid, taskMeta.deviceId, taskMeta.commId, taskMeta.rankId, taskMeta.streamId, taskMeta.jettyId,
                taskMeta.taskData.reduce.srcOffset, taskMeta.taskData.reduce.dstOffset,
                taskMeta.taskData.reduce.dataCount, taskMeta.taskData.reduce.srcDeviceId,
                taskMeta.taskData.reduce.dstDeviceId, taskMeta.taskData.reduce.reduceOp,
                taskMeta.taskData.reduce.dataType, eidInfo);
            break;
        case HccLTaskMetaType::NOTIFY_WAIT:
            HCCL_VM_INFO(
                "pid[{}]: deviceId[{}], commId[{}], rankId[{}], streamId[{}], taskType[NOTIFY_WAIT], jettyId[{}], "
                "notifyId[{}], recordDeviceId[{}], waitDeviceId[{}]{}",
                pid, taskMeta.deviceId, taskMeta.commId, taskMeta.rankId, taskMeta.streamId, taskMeta.jettyId,
                taskMeta.taskData.notify.notifyId, taskMeta.taskData.notify.srcDeviceId,
                taskMeta.taskData.notify.dstDeviceId, eidInfo);
            break;
        case HccLTaskMetaType::NOTIFY_RECORD:
            HCCL_VM_INFO(
                "pid[{}]: deviceId[{}], commId[{}], rankId[{}], streamId[{}], taskType[NOTIFY_RECORD], jettyId[{}], "
                "notifyId[{}], recordDeviceId[{}], waitDeviceId[{}]{}",
                pid, taskMeta.deviceId, taskMeta.commId, taskMeta.rankId, taskMeta.streamId, taskMeta.jettyId,
                taskMeta.taskData.notify.notifyId, taskMeta.taskData.notify.srcDeviceId,
                taskMeta.taskData.notify.dstDeviceId, eidInfo);
            break;
        default:
            break;
    }
}

aclError aclrtCreateNotify(aclrtNotify* notify, uint64_t flag)
{
    (void)flag;
    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (runner.current_ctx_id == 0) {
        HCCL_VM_ERROR("invalid param");
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = sim::runtime::Db::GetById<sim::runtime::Context>(runner.current_ctx_id);
    if (!currCtx.ok()) {
        HCCL_VM_ERROR("CurrContext not found:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto& currCtxId = currCtx.value->id;

    // GetCurrDeviceId 返回 Device 表主键；先取逻辑设备号，再由映射函数确定 Notify 所属 rank。
    const auto currentDevice = sim::runtime::Db::GetById<sim::runtime::Device>(sim::runtime::GetCurrDeviceId());
    if (!currentDevice.ok()) {
        HCCL_VM_ERROR("current device not found for notify");
        return ACL_ERROR_INVALID_PARAM;
    }
    uint32_t rankId = currentDevice.value->logic_id;
    uint64_t serverTmp = 0;
    sim::runtime::GetRankIdByMPI(rankId, serverTmp);
    if (rankId >= HcclSim::GetMaxRanks()) {
        HCCL_VM_ERROR("invalid rank for notify allocation: {:d}", rankId);
        return ACL_ERROR_INVALID_PARAM;
    }

    uint32_t deviceNotifyId = 0;
    if (!g_deviceNotifyIds.Allocate(deviceNotifyId)) {
        HCCL_VM_ERROR("device notify id capacity exhausted");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    sim::runtime::Notify tmp{};
    tmp.create_ctx_id = currCtxId;
    tmp.device_notify_seq = HcclSim::MakeNotifyTaskId(rankId, deviceNotifyId);
    tmp.value = 0;
    tmp.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::Notify>(tmp);
    if (!inserted.ok() || !inserted.value.has_value()) {
        g_deviceNotifyIds.Release(deviceNotifyId);
        HCCL_VM_ERROR("failed to record notify for context:{:d}", currCtxId);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    auto res = inserted.value->value;

    *notify = (aclrtNotify)res;
    HCCL_VM_DEBUG("id:{:d}", res);
    return ACL_SUCCESS;
}

aclError aclrtDestroyNotify(aclrtNotify notify)
{
    const uint64_t storageId = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(notify));
    HCCL_VM_DEBUG("notify: {:d}", storageId);
    const auto record = sim::runtime::Db::GetById<sim::runtime::Notify>(storageId);
    auto deleted
        = sim::runtime::Db::Delete<sim::runtime::Notify>(HcclSim::Storage::Eq(&sim::runtime::Notify::id, storageId));
    if (!deleted.ok() || !deleted.value.has_value() || *deleted.value == 0) {
        HCCL_VM_ERROR("failed to destroy notify:{:d}", storageId);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    if (record.ok()) {
        g_deviceNotifyIds.Release(HcclSim::GetDeviceNotifyId(record->device_notify_seq));
    }
    return ACL_SUCCESS;
}

aclError aclrtGetNotifyId(aclrtNotify notify, uint32_t* notifyId)
{
    if (notifyId == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    uint64_t notifyTaskId = 0;
    if (!GetNotifyTaskId(notify, notifyTaskId)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *notifyId = HcclSim::GetDeviceNotifyId(notifyTaskId);
    HCCL_VM_DEBUG("notifyId: {:d}", *notifyId);
    return ACL_SUCCESS;
}

aclError aclrtRecordNotify(aclrtNotify notify, aclrtStream stream)
{
    uint64_t streamId = reinterpret_cast<uint64_t>(stream);
    uint64_t notifyId = 0;
    if (!GetNotifyTaskId(notify, notifyId)) {
        return ACL_ERROR_INVALID_PARAM;
    }

    uint32_t deviceId = (uint32_t)sim::runtime::GetCurrDeviceId();

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    taskMetaData.profInfo = HcclTaskProfInfo(
        ProfTaskParamType::TASK_NOTIFY_RECORD, ProfLinkType::LINK_ONCHIP,
        ProfTransportType::TRANSPORT_LOCAL); // hcomm 本地 NotifyRecord
    taskMetaData.deviceId = deviceId;
    taskMetaData.commId = sim::runtime::g_cur_comm_key;
    if (!ResolveNotifyCommIdentity(taskMetaData.commId, deviceId, taskMetaData.rankId)) {
        HCCL_VM_ERROR("notify record task has no valid communicator, deviceId={}", deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.notify.notifyId = notifyId;
    taskMetaData.taskData.notify.notifyCount = 0; // notify value
    // 评估模式: deviceId字段承载通信域rankId; 主线保持真实deviceId供checker/runner消费
    if (sim::IsEstimateMode()) {
        taskMetaData.taskData.notify.srcDeviceId = taskMetaData.rankId;
        taskMetaData.taskData.notify.dstDeviceId = taskMetaData.rankId;
    } else {
        taskMetaData.taskData.notify.srcDeviceId = deviceId;
        taskMetaData.taskData.notify.dstDeviceId = deviceId;
    }
    PrintTaskMetaData(taskMetaData);

    // 图模式采集态：只录制 NOTIFY_RECORD，不进 DB（重放期 ReplayModelActions 在 START_SUB/END_SUB 之间重插）
    if (IsCapturingStream(streamId)) {
        return RecordThreadTaskAction(streamId, taskMetaData) ? ACL_SUCCESS : ACL_ERROR_INTERNAL_ERROR;
    }

    uint32_t index{0};
    HCCL_VM_DEBUG("Get notify task, id={:d}, streamId={:d}", notifyId, streamId);
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("InsertTaskToCollection fail");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    return ACL_SUCCESS;
}

aclError aclrtWaitAndResetNotify(aclrtNotify notify, aclrtStream stream, uint32_t timeout)
{
    (void)timeout;
    uint64_t streamId = reinterpret_cast<uint64_t>(stream);
    const uint64_t storageId = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(notify));
    uint64_t notifyId = 0;
    if (!GetNotifyTaskId(notify, notifyId)) {
        return ACL_ERROR_INVALID_PARAM;
    }

    uint32_t deviceId = (uint32_t)sim::runtime::GetCurrDeviceId();

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    taskMetaData.profInfo = HcclTaskProfInfo(
        ProfTaskParamType::TASK_NOTIFY_WAIT, ProfLinkType::LINK_ONCHIP,
        ProfTransportType::TRANSPORT_LOCAL); // hcomm 本地 NotifyWait
    taskMetaData.deviceId = deviceId;
    taskMetaData.commId = sim::runtime::g_cur_comm_key;
    if (!ResolveNotifyCommIdentity(taskMetaData.commId, deviceId, taskMetaData.rankId)) {
        HCCL_VM_ERROR("notify wait task has no valid communicator, deviceId={}", deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.notify.notifyId = notifyId;
    taskMetaData.taskData.notify.notifyCount = 0; // notify value
    // 评估模式: deviceId字段承载通信域rankId; 主线保持真实deviceId供checker/runner消费
    if (sim::IsEstimateMode()) {
        taskMetaData.taskData.notify.srcDeviceId = taskMetaData.rankId;
        taskMetaData.taskData.notify.dstDeviceId = taskMetaData.rankId;
    } else {
        taskMetaData.taskData.notify.srcDeviceId = deviceId;
        taskMetaData.taskData.notify.dstDeviceId = deviceId;
    }
    PrintTaskMetaData(taskMetaData);

    // 图模式采集态：只录制 NOTIFY_WAIT，不 reset notify、不进 DB（重放期 ReplayModelActions 再 reset + 重插）
    if (IsCapturingStream(streamId)) {
        return RecordThreadTaskAction(streamId, taskMetaData) ? ACL_SUCCESS : ACL_ERROR_INTERNAL_ERROR;
    }

    // reset notify（主线语义：等待后复位 value；复位按存储主键，任务 id 与存储主键分离）。
    // 失败判定只看操作本身（对齐主线 RunnerDB::Update 返回语义）：MariaDB 的 UPDATE
    // affected 计数只含值真正变化的行——Notify.value 本就为 0 时 SET value=0 命中行
    // 但 0 变化，不能据此判"notify not found"（SQLite 的计数按匹配行，两者天然不同）。
    auto res = sim::runtime::Db::Update<sim::runtime::Notify>(
        HcclSim::Storage::Eq(&sim::runtime::Notify::id, storageId),
        HcclSim::Storage::Set(&sim::runtime::Notify::value, 0));
    if (!res.ok()) {
        HCCL_VM_ERROR("notify not found:{:d}", storageId);
        return ACL_ERROR_INVALID_PARAM;
    }

    uint32_t index{0};
    HCCL_VM_DEBUG("Get notify task, id={:d}, streamId={:d}", notifyId, streamId);
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("InsertTaskToCollection fail");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    return ACL_SUCCESS;
}

aclError aclrtNotifyBatchReset(aclrtNotify* notifies, size_t num)
{
    for (int i = 0; i < num; i++) {
        uint64_t notifyId = (uint32_t)(uintptr_t)notifies[i];

        // reset notify
        auto res = sim::runtime::Db::Update<sim::runtime::Notify>(
            HcclSim::Storage::Eq(&sim::runtime::Notify::id, notifyId),
            HcclSim::Storage::Set(&sim::runtime::Notify::value, 0));
        if (!res.ok() || !res.value.has_value() || *res.value == 0) {
            HCCL_VM_ERROR("notify not found:{:d}", notifyId);
            return ACL_ERROR_INVALID_PARAM;
        }
    }
    return ACL_SUCCESS;
}

aclError aclrtNotifyGetExportKey(aclrtNotify notify, char* key, size_t len, uint64_t flags)
{
    (void)flags;
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;

    std::string notifyKeyStr = std::to_string(notifyId);

    sim::runtime::IpcNotify tmp{};
    tmp.notify_id = notifyId;
    // tmp.flag = (uint8_t)flags;
    memcpy(tmp.name_or_key, notifyKeyStr.data(), notifyKeyStr.length());
    if (len > 0) {
        size_t copyLen = std::min(len - 1, notifyKeyStr.length());
        memcpy(key, notifyKeyStr.data(), copyLen);
        key[copyLen] = '\0';
    }
    tmp.create_pid = getpid();
    tmp.id = 0;
    auto res = sim::runtime::Db::Add<sim::runtime::IpcNotify>(tmp);
    if (!res.ok()) {
        return ACL_ERROR_INVALID_PARAM;
    }
    HCCL_VM_DEBUG("id:{:d}", notifyId);
    return ACL_SUCCESS;
}

aclError aclrtNotifySetImportPid(aclrtNotify notify, int32_t* pid, size_t num)
{
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;
    HCCL_VM_DEBUG("id:{:d}", notifyId);

    auto ipcNotify = sim::runtime::Db::GetOneByPred<sim::runtime::IpcNotify>(
        HcclSim::Storage::Eq(&sim::runtime::IpcNotify::notify_id, notifyId));
    if (!ipcNotify.ok()) {
        HCCL_VM_ERROR("notify in ipc notify not found:{:d}", notifyId);
        return ACL_ERROR_INVALID_PARAM;
    }

    for (int i = 0; i < num; i++) {
        sim::runtime::IpcNotifyVistorList vistor;
        vistor.ipc_id = ipcNotify->id;
        vistor.visitor_pid = pid[i];
        vistor.id = 0;
        auto inserted = sim::runtime::Db::Add<sim::runtime::IpcNotifyVistorList>(vistor);
        if (!inserted.ok()) {
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    return ACL_SUCCESS;
}

aclError aclrtNotifyImportByKey(aclrtNotify* notify, const char* key, uint64_t flags)
{
    (void)flags;
    // 名字前缀查找:memcmp(名字, strlen(key)) 的迁移前语义等价于 TEXT 列 PREFIX 比较。
    auto ipcNotify = sim::runtime::Db::GetOneByPred<sim::runtime::IpcNotify>(
        HcclSim::Storage::StartsWith(&sim::runtime::IpcNotify::name_or_key, std::string(key)));
    if (!ipcNotify.ok()) {
        HCCL_VM_ERROR("notify in ipc notify key not found:{}", key);
        return ACL_ERROR_INVALID_PARAM;
    }
    *notify = (aclrtNotify)ipcNotify->notify_id;
    HCCL_VM_DEBUG("id:{:d}", ipcNotify->notify_id);
    return ACL_SUCCESS;
}

////////////////////////////rt 接口/////////////////////////////////
rtError_t rtsNotifyCreate(rtNotify_t* notify, uint64_t flag) { return aclrtCreateNotify(notify, flag); }

rtError_t rtNotifyWait(rtNotify_t notify, rtStream_t stm) { return aclrtWaitAndResetNotify(notify, stm, 0); }

rtError_t rtGetNotifyAddress(rtNotify_t notify, uint64_t* const notifyAddres)
{
    uint64_t notifyTaskId = 0;
    if (!GetNotifyTaskId(notify, notifyTaskId)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *notifyAddres = HcclSim::GetDeviceNotifyId(notifyTaskId);
    return RT_ERROR_NONE;
}

rtError_t rtNotifyGetPhyInfo(rtNotify_t notify, uint32_t* phyDevId, uint32_t* tsId)
{
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;
    // reset notify
    auto res = sim::runtime::Db::GetById<sim::runtime::Notify>(notifyId);
    if (!res.ok()) {
        HCCL_VM_ERROR("notify not found:{:d}", notifyId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto createCtx = sim::runtime::Db::GetById<sim::runtime::Context>(res.value->create_ctx_id);
    if (!createCtx.ok()) {
        HCCL_VM_ERROR("create ctx not found:{:d}", res.value->create_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto devRes = sim::runtime::Db::GetById<sim::runtime::Device>(createCtx.value->device_id);
    if (!devRes.ok()) {
        HCCL_VM_ERROR("device id not found:{:d}", createCtx.value->device_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    *phyDevId = devRes.value->physical_id;
    *tsId = 0;
    return RT_ERROR_NONE;
}

rtError_t rtNotifyGetPhyInfoExt(rtNotify_t notify, rtNotifyPhyInfo* notifyInfo)
{
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;
    // reset notify
    auto res = sim::runtime::Db::GetById<sim::runtime::Notify>(notifyId);
    if (!res.ok()) {
        HCCL_VM_ERROR("notify not found:{:d}", notifyId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto createCtx = sim::runtime::Db::GetById<sim::runtime::Context>(res.value->create_ctx_id);
    if (!createCtx.ok()) {
        HCCL_VM_ERROR("create ctx not found:{:d}", res.value->create_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto devRes = sim::runtime::Db::GetById<sim::runtime::Device>(createCtx.value->device_id);
    if (!devRes.ok()) {
        HCCL_VM_ERROR("device id not found:{:d}", createCtx.value->device_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    notifyInfo->phyId = devRes.value->physical_id;
    notifyInfo->tsId = 0;
    notifyInfo->idType = 0;
    notifyInfo->shrId = (uint32_t)notifyId;
    notifyInfo->flag = 0;
    return RT_ERROR_NONE;
}

rtError_t rtNotifyGetAddrOffset(rtNotify_t notify, uint64_t* devAddrOffset)
{
    (void)notify;
    (void)devAddrOffset;
    return RT_ERROR_NONE;
}

#ifdef __cplusplus
}
#endif // __cplusplus
