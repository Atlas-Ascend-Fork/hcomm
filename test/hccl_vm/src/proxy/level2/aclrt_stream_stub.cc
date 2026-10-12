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
#include "runtime_state/sim_models.h"
#define HCCL_VM_MODULE "STREAM_STUB"

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fcntl.h>
#include <iostream>
#include <sys/file.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "store_dump_shm_data.h"
#include "sim_log.h"
#include "runtime/base.h"
#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "storage/table_access.h"
#include "store_sim_store_pub.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

aclError aclrtCreateStreamWithConfig(aclrtStream* stream, uint32_t priority, uint32_t flag)
{
    (void)flag;
    (void)priority;
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
        HCCL_VM_ERROR("ctx not found:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto& currCtxId = currCtx.value->id;

    sim::runtime::Stream streamTmp{};
    streamTmp.ctx_id = currCtxId;
    streamTmp.activated = 1;
    streamTmp.priority = priority;
    streamTmp.user_tag = flag;
    streamTmp.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::Stream>(streamTmp);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto res = inserted.value->value;

    *stream = (aclrtStream)res;
    HCCL_VM_DEBUG("id:{:d}", res);
    sim::runtime::SetLastStreamIdTls(res);
    sim::operation::AddSyncStreamIter(res);
    return ACL_SUCCESS;
}

aclError aclrtCreateStream(aclrtStream* stream) { return aclrtCreateStreamWithConfig(stream, 0, 0); }

aclError aclrtDestroyStream(aclrtStream stream)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d}", streamId);
    (void)sim::runtime::Db::Delete<sim::runtime::Stream>(HcclSim::Storage::Eq(&sim::runtime::Stream::id, streamId));
    sim::operation::RemoveSyncStreamIter(streamId);
    return ACL_SUCCESS;
}

aclError aclrtDestroyStreamForce(aclrtStream stream) { return aclrtDestroyStream(stream); }

aclError aclrtActiveStream(aclrtStream activeStream, aclrtStream stream)
{
    (void)stream;
    uint64_t activStreamId = (uint64_t)(uintptr_t)activeStream;
    HCCL_VM_DEBUG("id:{:d}", activStreamId);
    auto res = sim::runtime::Db::Update<sim::runtime::Stream>(
        HcclSim::Storage::Eq(&sim::runtime::Stream::id, activStreamId),
        HcclSim::Storage::Set(&sim::runtime::Stream::activated, 1));
    if (!res.ok() || !res.value.has_value() || *res.value == 0) {
        HCCL_VM_ERROR("stream not found:{:d}", activStreamId);
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

aclError aclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d} mode:{:d}", streamId, mode);
    (void)sim::runtime::Db::Update<sim::runtime::Stream>(
        HcclSim::Storage::Eq(&sim::runtime::Stream::id, streamId),
        HcclSim::Storage::Set(&sim::runtime::Stream::failure_mode, mode));

    return ACL_SUCCESS;
}

aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout)
{
    (void)timeout;
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    // GetMode();
    std::string mode = "checker";
    const int WAIT_COUNTDOWN = 10; // 等待20s
    HCCL_VM_DEBUG("id:{:d}", streamId);
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeStream(aclrtStream stream)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    uint64_t syncIdx{UINT64_MAX};
    if (!sim::operation::NextStreamSyncIdx(streamId, syncIdx)) {
        HCCL_VM_ERROR("Get stream syncIdx failed, streamId={:d}", streamId);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::SYNC_STREAM;
    taskMetaData.commId = 0;
    taskMetaData.deviceId = sim::runtime::GetCurrDeviceId();
    taskMetaData.rankId = UINT32_MAX;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.syncStreamTask.syncIdx = syncIdx;

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("InsertTaskToCollection fail");
        return ACL_ERROR_INTERNAL_ERROR;
    }
    HCCL_VM_DEBUG("Add SYNC_STREAM task, streamId={:d}, syncIdx={:d}", streamId, syncIdx);

    //  把 aclrtSynchronizeStream 入参流作为当前 opdetail 记录的真实主流，幂等更新
    // （图模式下 opdetail.streamId 采集期记的是 master_stream，此处修正为重放触发流 resources_str.stream）
    // 主流发现走摘要接口：只取最新 detail 的 id/streamId（不读 opDetail/opExtInfo BLOB）。
    uint32_t latestOpDetailId{0};
    uint64_t latestStreamId{0};
    if (sim::operation::QueryLatestOpDetailStreamByDeviceId(taskMetaData.deviceId, latestOpDetailId, latestStreamId)
        == 0) {
        if (latestStreamId != streamId) {
            if (sim::operation::UpdateOpDetailMainStream(latestOpDetailId, streamId) != 0) {
                HCCL_VM_ERROR(
                    "UpdateOpDetailMainStream fail, opDetailId={}, streamId={}->{}", latestOpDetailId, latestStreamId,
                    streamId);
                return ACL_ERROR_INTERNAL_ERROR;
            }
            HCCL_VM_INFO(
                "update real main stream, opDetailId={}, streamId={} -> {}", latestOpDetailId, latestStreamId,
                streamId);
        }
    }

    // 记录 synchronize_strategy：aclrtSynchronizeStream 是 DeviceStatus 的写入源，
    // 查到对应 device 的记录就 Update，查不到就 Add。
    auto streamRecord = sim::runtime::Db::GetById<sim::runtime::Stream>(streamId);
    if (!streamRecord.ok()) {
        HCCL_VM_ERROR("stream not found:{:d}", streamId);
        return ACL_ERROR_INVALID_PARAM;
    }
    // dump 未开启时不维护 DeviceStatus（其唯一消费者是 RunnerListen 的 dump 触发），
    // 同时跳过 DeviceStatus 所需的 stream/context/device 查询链路.
    const char* enableDumpData = std::getenv("HCCLVM_ENABLE_DUMP_DATA");
    const bool dumpEnabled
        = (enableDumpData != nullptr && !std::string(enableDumpData).empty() && std::string(enableDumpData) != "0");
    if (dumpEnabled) {
        // 记录 synchronize_strategy：aclrtSynchronizeStream 是 DeviceStatus 的写入源，
        // 查到对应 device 的记录就 Update，查不到就 Add。
        auto streamRecord = sim::runtime::Db::GetById<sim::runtime::Stream>(streamId);
        if (!streamRecord.ok()) {
            HCCL_VM_ERROR("stream not found:{:d}", streamId);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto context = sim::runtime::Db::GetById<sim::runtime::Context>(streamRecord->ctx_id);
        if (!context.ok()) {
            HCCL_VM_ERROR("context not found:{:d}", streamRecord->ctx_id);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto device = sim::runtime::Db::GetById<sim::runtime::Device>(context->device_id);
        if (!device.ok()) {
            HCCL_VM_ERROR("device not found:{:d}", context->device_id);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto deviceStatus = sim::runtime::Db::GetOneByPred<sim::runtime::DeviceStatus>(
            HcclSim::Storage::Eq(&sim::runtime::DeviceStatus::device_id, device->id));
        if (deviceStatus.ok()) {
            (void)sim::runtime::Db::Update<sim::runtime::DeviceStatus>(
                HcclSim::Storage::Eq(&sim::runtime::DeviceStatus::id, deviceStatus->id),
                HcclSim::Storage::Set(&sim::runtime::DeviceStatus::synchronize_strategy, (uint32_t)1));
        } else {
            sim::runtime::DeviceStatus newStatus{};
            newStatus.device_id = device->id;
            newStatus.synchronize_strategy = 1;
            (void)sim::runtime::Db::Add<sim::runtime::DeviceStatus>(newStatus);
        }
    }

    auto pluginRows
        = sim::runtime::Db::GetByPred<sim::runtime::Plugin>(HcclSim::Storage::Eq(&sim::runtime::Plugin::tag, "runner"));
    if (!pluginRows.ok() || !pluginRows.value.has_value() || pluginRows.value->empty()) { // Runner 未启动
        return ACL_SUCCESS;
    }

    // Runner 已启动，等待stream上所有task运行结束
    while (true) {
        uint64_t totalTasks = 0;
        uint64_t doneTasks = 0;
        if (sim::operation::QueryOpTaskCompletionByStreamId(streamId, totalTasks, doneTasks) != 0) {
            HCCL_VM_ERROR("QueryOpTasksByStreamId fail");
            return ACL_ERROR_INTERNAL_ERROR;
        }

        HCCL_VM_INFO("Waiting for stream[{:d}]... {:d}/{:d}", streamId, doneTasks, totalTasks);

        if (doneTasks == totalTasks) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return ACL_SUCCESS;
}

aclError aclrtStreamAbort(aclrtStream stream)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d}", streamId);
    (void)sim::runtime::Db::Update<sim::runtime::Stream>(
        HcclSim::Storage::Eq(&sim::runtime::Stream::id, streamId),
        HcclSim::Storage::Set(&sim::runtime::Stream::activated, 0));

    return ACL_SUCCESS;
}

aclError aclrtStreamQuery(aclrtStream stream, aclrtStreamStatus* status)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d}", streamId);
    auto res = sim::runtime::Db::GetById<sim::runtime::Stream>(streamId);
    if (!res.ok()) {
        HCCL_VM_ERROR("stream not found:{:d}", streamId);
        return ACL_ERROR_INVALID_PARAM;
    }
    *status = (aclrtStreamStatus)res.value->task_complete_status;
    return ACL_SUCCESS;
}

aclError aclrtGetStreamAvailableNum(uint32_t* streamCount)
{
    (void)streamCount;
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
        HCCL_VM_ERROR("ctx not found:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto& devId = currCtx.value->device_id;

    auto device = sim::runtime::Db::GetById<sim::runtime::Device>(devId);
    if (!device.ok()) {
        HCCL_VM_ERROR("device not found:{:d}", devId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto currCtxs = sim::runtime::Db::GetByPred<sim::runtime::Context>(
        HcclSim::Storage::Eq(&sim::runtime::Context::device_id, devId));
    if (!currCtxs.ok() || !currCtxs.value.has_value()) {
        return ACL_ERROR_INVALID_PARAM;
    }

    // 上下文集合内查流:ctx_id IN (当前设备全部上下文 ID),一次下推。
    std::vector<uint64_t> ctxIds;
    ctxIds.reserve(currCtxs.value->size());
    for (const auto& ctx : *currCtxs.value) {
        ctxIds.push_back(ctx.id);
    }
    // 数据库端聚合计数：只取行数，不加载行集。
    auto currStreams = sim::runtime::Db::Count<sim::runtime::Stream>(
        HcclSim::Storage::In(&sim::runtime::Stream::ctx_id, std::move(ctxIds)));
    if (!currStreams.ok()) {
        return ACL_ERROR_INVALID_PARAM;
    }

    if (memcmp(device.value->soc_version, "A3", strlen("A3") == 0)) {
        *streamCount = 1984 - static_cast<uint32_t>(*currStreams);
    }

    return ACL_SUCCESS;
}

aclError aclrtStreamGetId(aclrtStream stream, int32_t* streamId)
{
    *streamId = (uint32_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d}", *streamId);
    return ACL_SUCCESS;
}

aclError aclrtSetStreamOverflowSwitch(aclrtStream stream, uint32_t flag)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d} flag:{:d}", streamId, flag);
    (void)sim::runtime::Db::Update<sim::runtime::Stream>(
        HcclSim::Storage::Eq(&sim::runtime::Stream::id, streamId),
        HcclSim::Storage::Set(&sim::runtime::Stream::overflow_switch, flag));
    return ACL_SUCCESS;
}

aclError aclrtGetStreamOverflowSwitch(aclrtStream stream, uint32_t* flag)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d}", streamId);
    auto res = sim::runtime::Db::GetById<sim::runtime::Stream>(streamId);
    if (!res.ok()) {
        HCCL_VM_ERROR("stream not found:{:d}", streamId);
        return ACL_ERROR_INVALID_PARAM;
    }
    *flag = res.value->overflow_switch;
    return ACL_SUCCESS;
}

aclError aclrtSetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue* value)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d}", streamId);
    auto res = sim::runtime::Db::GetById<sim::runtime::Stream>(streamId);
    if (!res.ok()) {
        HCCL_VM_ERROR("stream not found:{:d}", streamId);
        return ACL_ERROR_INVALID_PARAM;
    }

    // 按属性类型显式更新对应字段;未知属性不产生变更(与迁移前无字段被改写的语义一致)。
    aclrtStreamAttrValue tmp = *value;
    if (stmAttrType == ACL_STREAM_ATTR_FAILURE_MODE) {
        (void)sim::runtime::Db::Update<sim::runtime::Stream>(
            HcclSim::Storage::Eq(&sim::runtime::Stream::id, streamId),
            HcclSim::Storage::Set(&sim::runtime::Stream::failure_mode, tmp.failureMode));
    } else if (stmAttrType == ACL_STREAM_ATTR_FLOAT_OVERFLOW_CHECK) {
        (void)sim::runtime::Db::Update<sim::runtime::Stream>(
            HcclSim::Storage::Eq(&sim::runtime::Stream::id, streamId),
            HcclSim::Storage::Set(&sim::runtime::Stream::overflow_switch, tmp.overflowSwitch));
    } else if (stmAttrType == ACL_STREAM_ATTR_USER_CUSTOM_TAG) {
        (void)sim::runtime::Db::Update<sim::runtime::Stream>(
            HcclSim::Storage::Eq(&sim::runtime::Stream::id, streamId),
            HcclSim::Storage::Set(&sim::runtime::Stream::user_tag, tmp.userCustomTag));
    } else {
        return ACL_SUCCESS;
    }

    return ACL_SUCCESS;
}

aclError aclrtGetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue* value)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d}", streamId);
    auto res = sim::runtime::Db::GetById<sim::runtime::Stream>(streamId);
    if (!res.ok()) {
        HCCL_VM_ERROR("stream not found:{:d}", streamId);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (stmAttrType == ACL_STREAM_ATTR_FAILURE_MODE) {
        value->failureMode = res.value->failure_mode;
    } else if (stmAttrType == ACL_STREAM_ATTR_FLOAT_OVERFLOW_CHECK) {
        value->overflowSwitch = res.value->overflow_switch;
    } else if (stmAttrType == ACL_STREAM_ATTR_USER_CUSTOM_TAG) {
        value->userCustomTag = res.value->user_tag;
    }
    return ACL_SUCCESS;
}

aclError aclrtStreamStop(aclrtStream stream)
{
    (void)stream;
    return ACL_SUCCESS;
}

rtError_t rtStreamSynchronize(rtStream_t stream)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d}", streamId);
    return aclrtSynchronizeStream((aclrtStream)stream);
}

rtError_t rtStreamCreateWithFlags(rtStream_t* stm, int32_t priority, uint32_t flags)
{
    return aclrtCreateStreamWithConfig((aclrtStream*)stm, priority, flags);
}

rtError_t rtStreamGetSqid(const rtStream_t stream, uint32_t* sqId)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    *sqId = streamId;
    HCCL_VM_DEBUG("id:{:d}", streamId);
    return ACL_SUCCESS;
}

rtError_t rtGetTaskIdAndStreamID(uint32_t* taskId, uint32_t* streamId)
{
    *streamId = (uint32_t)sim::runtime::GetLastStreamIdTls();
    *taskId = (uint32_t)sim::runtime::GetLastTaskIdTls();
    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
