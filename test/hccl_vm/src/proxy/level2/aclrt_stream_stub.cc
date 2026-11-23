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
#include "device_spec/device_spec_types.h"
#include "storage/table_access.h"
#include "store_sim_store_pub.h"
#include "resource_limit_manager.h"

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
    const auto deviceKey = currCtx.value->device_id;
    if (!HCCL_VM_ACQUIRE_STREAM(deviceKey)) {
        return ACL_ERROR_RT_NO_STREAM_RESOURCE;
    }

    sim::runtime::Stream streamTmp{};
    streamTmp.ctx_id = currCtxId;
    streamTmp.activated = 1;
    streamTmp.priority = priority;
    streamTmp.user_tag = flag;
    streamTmp.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::Stream>(streamTmp);
    if (!inserted.ok() || !inserted.value.has_value()) {
        HCCL_VM_RELEASE_STREAM(deviceKey);
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
    // 计数回减（内部反查所属卡并跳过默认流）须在删除行之前完成；stub 不持有 deviceKey。
    HCCL_VM_RELEASE_STREAM_BY_ID(streamId);
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

/**
 * @brief 查询 HCCLVM_ENABLE_DUMP_DATA 开关是否使能（RunnerListen/bin dump 与
 *        all_rank_input_output.txt dump 共用该开关）。
 *
 * @return bool 已设置且非 "0" 返回 true；未设置/为空/"0" 返回 false。
 */
static bool IsDumpDataEnabled()
{
    const char* enableDumpData = std::getenv("HCCLVM_ENABLE_DUMP_DATA");
    return enableDumpData != nullptr && !std::string(enableDumpData).empty() && std::string(enableDumpData) != "0";
}

/**
 * @brief 等待 stream 上的 runner 任务全部执行完成（doneTasks==totalTasks 轮询）。
 *
 * @param streamId 输入：待等待完成的 stream id。
 * @return aclError 全部完成返回 ACL_SUCCESS；完成度查询失败返回 ACL_ERROR_INTERNAL_ERROR。
 */
static aclError WaitStreamTasksDone(uint64_t streamId)
{
    while (true) {
        uint64_t totalTasks = 0;
        uint64_t doneTasks = 0;
        if (sim::operation::QueryOpTaskCompletionByStreamId(streamId, totalTasks, doneTasks) != 0) {
            HCCL_VM_ERROR("QueryOpTasksByStreamId fail");
            return ACL_ERROR_INTERNAL_ERROR;
        }

        HCCL_VM_INFO("Waiting for stream[{:d}]... {:d}/{:d}", streamId, doneTasks, totalTasks);

        if (doneTasks == totalTasks) {
            return ACL_SUCCESS;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

/**
 * @brief 置位本 stream 所属设备的 DeviceStatus.synchronize_strategy=1。
 *
 * 该信号的消费者（RunnerListen 的 bin dump、runner 侧 all_rank_input_output.txt dump）
 * 据此判定"全部设备同步就绪"再读设备 buffer：须在 stream 任务全部完成后置位，若在
 * synchronize 入口就置位，dump 会与仍在执行的任务竞争（实测读到计算中的 rank
 * output 全 0）。查到 DeviceStatus 记录就 Update，查不到就 Add。
 *
 * @param streamRecord 输入：stream 记录（提供 ctx_id 归属链）。
 * @return aclError 成功返回 ACL_SUCCESS；context/device 记录缺失返回 ACL_ERROR_INVALID_PARAM。
 */
static aclError MarkSyncStrategyReady(const sim::runtime::Stream& streamRecord)
{
    auto context = sim::runtime::Db::GetById<sim::runtime::Context>(streamRecord.ctx_id);
    if (!context.ok()) {
        HCCL_VM_ERROR("context not found:{:d}", streamRecord.ctx_id);
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
    return ACL_SUCCESS;
}

/**
 * @brief 登记 SYNC_STREAM 任务到任务集合（供 runner/checker 建图识别同步点）。
 *
 * @param streamId 输入：待同步的 stream id。
 * @return aclError 登记成功返回 ACL_SUCCESS；syncIdx 获取或任务插入失败返回
 *         ACL_ERROR_INTERNAL_ERROR。
 */
static aclError RegisterSyncStreamTask(uint64_t streamId)
{
    uint64_t syncIdx{UINT64_MAX};
    if (!sim::operation::NextStreamSyncIdx(streamId, syncIdx)) {
        HCCL_VM_ERROR("Get stream syncIdx failed, streamId={:d}", streamId);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::SYNC_STREAM;
    taskMetaData.profInfo = HcclTaskProfInfo(
        ProfTaskParamType::UNKNOWN, ProfLinkType::UNKNOWN, ProfTransportType::UNKNOWN); // hcomm profiling 无对应类型
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
    return ACL_SUCCESS;
}

/**
 * @brief 同步 stream：登记 SYNC_STREAM 任务、修正 opdetail 主流、任务完成后置位同步就绪信号。
 *
 * 行为要点：1) 插 SYNC_STREAM 任务并修正最新 opdetail 的主流记录；2) runner 插件在
 * 运行时等待 stream 任务被 runner 执行完毕（无 runner 时无任务可等，直接继续）；
 * 3) dump 开关使能时置位 DeviceStatus.synchronize_strategy=1（RunnerListen 的 bin
 * dump 与 runner 的 all_rank_input_output dump 的就绪信号，不装 runner 也须置位，
 * 否则性能仿真落盘不触发）。
 *
 * @param stream 输入：待同步的 stream 句柄。
 * @return aclError 成功返回 ACL_SUCCESS；syncIdx 获取/任务插入/主流修正/等待失败返回
 *         对应错误码；stream/context/device 记录缺失返回 ACL_ERROR_INVALID_PARAM。
 */
aclError aclrtSynchronizeStream(aclrtStream stream)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    const aclError registerRet = RegisterSyncStreamTask(streamId);
    if (registerRet != ACL_SUCCESS) {
        return registerRet;
    }

    // 主流修正不再挂在本接口上：
    // - eager 模式：OpDetail.streamId 在算子入口 RecordOpDbInfo 已记录 API 入参流（即真实主流），
    //   而算子执行期可能同步任意内部流（如嵌套 warmup 后外层同步 warmup 流、warmup 同步用户流），
    //   以“同步流”覆写“当前算子主流”会造成主流串写（嵌套场景两算子主流互换）；
    // - 图模式重放：真实主流 = rtModelExecute 入参流，已在该桩内直接回写克隆 OpDetail
    //   （见 aclrt_model_stub.cc 的 UpdateOpDetailMainStream），无需经本接口猜测。

    // 记录 synchronize_strategy：aclrtSynchronizeStream 是 DeviceStatus 的写入源，
    // 查到对应 device 的记录就 Update，查不到就 Add。
    auto streamRecord = sim::runtime::Db::GetById<sim::runtime::Stream>(streamId);
    if (!streamRecord.ok()) {
        HCCL_VM_ERROR("stream not found:{:d}", streamId);
        return ACL_ERROR_INVALID_PARAM;
    }
    // dump 未开启时不维护 DeviceStatus（其唯一消费者是 RunnerListen 的 dump 触发），
    // 同时跳过 DeviceStatus 所需的 stream/context/device 查询链路.
    const bool dumpEnabled = IsDumpDataEnabled();

    auto pluginRows
        = sim::runtime::Db::GetByPred<sim::runtime::Plugin>(HcclSim::Storage::Eq(&sim::runtime::Plugin::tag, "runner"));
    const bool runnerStarted = pluginRows.ok() && pluginRows.value.has_value() && !pluginRows.value->empty();
    if (runnerStarted) {
        // Runner 已启动，等待stream上所有task运行结束后再置位就绪信号
        const aclError waitRet = WaitStreamTasksDone(streamId);
        if (waitRet != ACL_SUCCESS) {
            return waitRet;
        }
    }

    // 就绪信号置位：须在 stream 任务全部完成后（无 runner 时无任务可等直接置位），
    // 详见 MarkSyncStrategyReady 注释。
    if (dumpEnabled) {
        return MarkSyncStrategyReady(*streamRecord);
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
