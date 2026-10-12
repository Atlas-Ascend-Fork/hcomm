/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 实现分区与 db_sim_runner_ops.h 保持相同顺序：薄兼容接口（模板在头文件）→
// Device/Server/Topology → Context/Stream/Task → Runtime State/Lifecycle。
// 业务函数（GetRankSize/GetHostSize 等）内部只经 sim::runtime::Db 兼容接口执行
// 既有原子操作；DDL 与 codec 在头文件。SetRunMode 的 Clear+Insert 复合流程
// 是实现边界内少数保留表句柄形态的流程（合同禁止 Clear 进事务）。
#include "runtime_state/db_sim_runner_ops.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <utility>

#include "sim_log.h"
#include "runtime_state/sim_models.h"
#include "runtime_state/db_sim_runner_common.h"
#include "storage/table_access.h"

extern uint64_t g_cur_server_key;
namespace sim::runtime {
thread_local Runner g_runner;

thread_local uint64_t g_last_streamId;
thread_local uint64_t g_last_taskId;
thread_local int g_tsId = 0;

// ==== 薄兼容接口分区：sim::runtime::Db 全部为头文件内联模板（一对一转发） ====

// ==== Device / Server / Topology 分区 ====

uint64_t GetServerKeyById(uint32_t superPodIdx, uint32_t serverIdx)
{
    auto serverRet = Db::GetOneByPred<Server>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&Server::pod_id, superPodIdx), HcclSim::Storage::Eq(&Server::server_id, serverIdx)));
    if (!serverRet.ok()) {
        HCCL_VM_ERROR("can not find server by id: {:d}, {:d}", superPodIdx, serverIdx);
        return 0;
    }
    return serverRet->id;
}

// ==== Context / Stream / Task 分区 ====

void SetLastStreamIdTls(uint64_t streamId) { g_last_streamId = streamId; }

void SetLastTaskIdTls(uint64_t taskId) { g_last_taskId = taskId; }

uint64_t GetLastStreamIdTls() { return g_last_streamId; }

uint64_t GetLastTaskIdTls() { return g_last_taskId; }

uint32_t GetRankSize()
{
    // 上游语义：通信域 rank 总数取自当前线程 Communicator 表行（Rank 表已随 CheckerL2 基线移除）。
    if (g_cur_comm_key == 0) {
        HCCL_VM_ERROR("current communicator is not set");
        return 0;
    }
    auto selfMember = Db::GetById<Communicator>(g_cur_comm_key);
    if (!selfMember.ok()) {
        HCCL_VM_ERROR("communicator member not found by id:{}", g_cur_comm_key);
        return 0;
    }
    return selfMember->rank_size;
}

uint32_t GetHostSize()
{
    // 数据库端 COUNT 聚合，不加载行集；失败按 0 返回（与迁移前空表语义一致）。
    return static_cast<uint32_t>(Db::Count<Host>(HcclSim::Storage::All<Host>()).value_or(0U));
}

// ==== Runtime State / Lifecycle 分区 ====
bool InsertRunner(uint64_t serverKey)
{
    try {
        auto curServer = Db::GetById<Server>(serverKey);
        if (!curServer.ok()) {
            HCCL_VM_ERROR("can not get server by key: {:d}", serverKey);
            return false;
        }

        auto ret = Db::GetOneByPred<Host>(HcclSim::Storage::Eq(&Host::server_id, serverKey));
        if (!ret.ok()) {
            HCCL_VM_ERROR("cannot find host by server key: {:d}", serverKey);
            return false;
        }
        const auto& host = *ret;
        auto hostKey = host.id;

        // 插入runner
        sim::runtime::Runner runner{};
        runner.host_id = hostKey;
        runner.pid = getpid();
        runner.thread_id = pthread_self();

        // todo: runner未实际插入数据库，这里需要重新赋值
        g_runner.host_id = hostKey;
        g_runner.pid = getpid();
        g_runner.thread_id = pthread_self();

        g_cur_server_key = serverKey;
        HCCL_VM_INFO("Init host ip= {}, server key= {}.", host.ip_addr, g_cur_server_key);

        runner.id = 0;
        auto inserted = Db::Add(runner);
        if (!inserted.ok() || !inserted.value.has_value()) {
            return false;
        }
        g_runner.id = inserted.value->value;
        return true;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("storage exception: {}", e.what());
        return false;
    }
}

bool GetCurrRunnerTls(uint64_t serverKey, Runner& runner)
{
    if (g_runner.id == 0 && serverKey == 0) {
        HCCL_VM_ERROR("can not get runner by server key: {:d}", serverKey);
        return false;
    }
    if (g_runner.id == 0) {
        InsertRunner(serverKey);
    }
    runner = g_runner;
    return true;
}

bool SetCurrCtxTls(uint64_t ctx)
{
    try {
        auto& currRunnerId = g_runner.id;
        g_runner.current_ctx_id = ctx;
        auto updated = Db::Update<Runner>(
            HcclSim::Storage::Eq(&Runner::id, currRunnerId), HcclSim::Storage::Set(&Runner::current_ctx_id, ctx));
        return updated.ok();
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("storage exception: {}", e.what());
        return false;
    }
}

uint64_t GetCurrDeviceId()
{
    auto currCtx = Db::GetById<Context>(g_runner.current_ctx_id);
    if (!currCtx.ok()) {
        // not find
        HCCL_VM_ERROR("can not get CurrContext: {:d}", g_runner.current_ctx_id);
        return 0;
    }

    auto dev = Db::GetById<Device>(currCtx.value->device_id);
    if (!dev.ok()) {
        // not find
        HCCL_VM_ERROR("can not get device: {:d}", currCtx.value->device_id);
        return 0;
    }

    return currCtx.value->device_id;
}

uint64_t GetCurrDeviceKey()
{
    auto currCtx = Db::GetById<Context>(g_runner.current_ctx_id);
    if (!currCtx.ok()) {
        // not find
        HCCL_VM_ERROR("can not get CurrContext: {:d}", g_runner.current_ctx_id);
        return 0;
    }

    return currCtx.value->device_id;
}

uint64_t GetDeviceIdByCtxId(uint64_t ctxId)
{
    auto currCtx = Db::GetById<Context>(ctxId);
    if (!currCtx.ok()) {
        // not find
        HCCL_VM_ERROR("can not get CurrContext: {:d}", ctxId);
        return 0;
    }

    auto device = Db::GetById<Device>(currCtx.value->device_id);
    if (!device.ok()) {
        // not find
        HCCL_VM_ERROR("can not get device: {:d}", currCtx.value->device_id);
        return 0;
    }

    return device.value->id;
}

void SetTsDevice(int tsId) { g_tsId = tsId; }

bool SetRunMode(bool checkOnlyMode)
{
    // 运行模式单行表语义：清空（重置自增序列）后写入新模式；主键由数据库生成回填。
    // Clear 不进入事务（合同禁止）：与迁移前 DELETE + INSERT 的非原子顺序一致。
    auto runModeTable = HcclSim::Storage::AcquireTable<RunModeConfig>();
    if (!runModeTable.ok()) {
        HCCL_VM_ERROR("Failed to acquire run mode table: {}", runModeTable.diagnostic);
        return false;
    }
    RunModeConfig runMode{};
    runMode.mode = checkOnlyMode ? 1 : 0;
    if (!runModeTable->Clear(HcclSim::Storage::ClearMode::RESET_SEQUENCE).ok()) {
        HCCL_VM_ERROR("Failed to reset run mode table.");
        return false;
    }
    runMode.id = 0;
    auto inserted = runModeTable->Insert(runMode);
    if (!inserted.ok()) {
        HCCL_VM_ERROR("Failed to insert run mode: {}", inserted.diagnostic);
        return false;
    }
    return true;
}
} // namespace sim::runtime
