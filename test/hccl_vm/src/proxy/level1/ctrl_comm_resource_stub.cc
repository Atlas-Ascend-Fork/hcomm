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
 * Description: 通信域资源管理打桩函数（北向劫持）
 * Create: 2026-07-20
 */

#define HCCL_VM_MODULE "CDR_STUB"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "acl/acl_rt.h"
#include "hccl/hccl_res.h"
#include "hccl_proxy_common.h"
#include "level1_proxy_common.h"
#include "sim_log.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "operation_data/operation_data_ops.h"
#include "sim_sub_process_manager.h"

extern uint64_t g_cur_server_key;

static void TryLaunchAicpuDevProcForRank(int32_t rankId, uint32_t deviceKey)
{
    auto& procMgr = sim::GetAicpuProcMgr();
    if (procMgr.IsAlive()) {
        return;
    }

    auto config = sim::CreateAicpuDeviceConfig(static_cast<uint32_t>(rankId), deviceKey);
    if (procMgr.CreateProcess(config) != 0) {
        HCCL_VM_ERROR("failed to create device process.");
        exit(EXIT_FAILURE);
    }

    HCCL_VM_INFO("device process for rankId[{}] launched, pid={}", rankId, procMgr.GetPid());
}

constexpr uint32_t MAX_THREAD_PER_COMM = 40;
constexpr uint64_t MAX_NOTIFY_PER_COMM = 640;
constexpr uint32_t MAX_CHANNEL_NUM = 1024 * 1024;
constexpr uint64_t CCU_DEFAULT_NOTIFY_NUM = 8;

// 与 sim::runtime::HcclThread.notifyId[40] / sim::runtime::HcclChannel.notifyId[64] 数组容量对齐
constexpr uint32_t MAX_NOTIFY_PER_THREAD = 40;
constexpr uint32_t MAX_NOTIFY_PER_CHANNEL = 64;

// 等待远端 HcclMem 注册记录就绪的轮询参数
constexpr uint32_t REMOTE_MEM_WAIT_MAX_MS = 10000;
constexpr uint32_t REMOTE_MEM_WAIT_POLL_INTERVAL_MS = 10;

// 释放线程记录上挂载的 Notify 记录（按 notifyId 数组逐条删除）
static inline void ReleaseThreadNotifies(const sim::runtime::HcclThread& rec)
{
    for (uint32_t n = 0; n < rec.notifyNum && n < MAX_NOTIFY_PER_THREAD; ++n) {
        if (rec.notifyId[n] != 0u) {
            sim::runtime::Db::Delete<sim::runtime::Notify>(
                HcclSim::Storage::Eq(&sim::runtime::Notify::id, rec.notifyId[n]));
        }
    }
}

// 释放通道记录上挂载的 Notify 记录（按 notifyId 数组逐条删除）
static inline void ReleaseChannelNotifies(const sim::runtime::HcclChannel& rec)
{
    for (uint32_t n = 0; n < rec.notifyNum && n < MAX_NOTIFY_PER_CHANNEL; ++n) {
        if (rec.notifyId[n] != 0u) {
            sim::runtime::Db::Delete<sim::runtime::Notify>(
                HcclSim::Storage::Eq(&sim::runtime::Notify::id, rec.notifyId[n]));
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 获取通信域中的HCCL通信内存。
 *
 * 返回通信域中本端rank的HCCL通信内存地址和大小。该内存由库内管理，调用者严禁释放。
 * 内存大小在通信域初始化时已分配，后续调用仅查询并返回已分配的内存信息。
 *
 * @param comm 输入：通信域句柄，由HcclCommInitXxx系列接口返回。
 * @param buffer 输出：HCCL通信内存地址。
 * @param size 输出：HCCL通信内存大小（字节）。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 获取成功
 * @retval HCCL_E_PTR comm、buffer或size指针为空
 * @retval HCCL_E_INTERNAL 内部错误（通信域不存在或未找到关联的HCCL Buffer）
 */
HcclResult HcclGetHcclBuffer(HcclComm comm, void** buffer, uint64_t* size)
{
    if (comm == nullptr) {
        HCCL_VM_ERROR("{}: comm is nullptr", __func__);
        return HCCL_E_PTR;
    }

    uint64_t commId = reinterpret_cast<uint64_t>(comm);

    auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    // 查询通信域关联的缓冲区记录
    // 每个通信域应有且仅有 1 个 HcclBuffer（在 Init 时创建）
    auto buffers = sim::runtime::Db::GetByPred<sim::runtime::HcclBuffer>(
        HcclSim::Storage::Eq(&sim::runtime::HcclBuffer::commId, commId));
    if (!buffers.ok() || buffers->size() != 1) {
        // size != 1 表示数据不一致（可能是并发问题或数据库损坏）
        HCCL_VM_ERROR(
            "{}: no HcclBuffer found for commId={:d},size={:d}", __func__, commId, buffers.ok() ? buffers->size() : 0U);
        return HCCL_E_INTERNAL;
    }

    const auto& hcclBuf = (*buffers)[0];
    *buffer = reinterpret_cast<void*>(hcclBuf.addr);
    *size = hcclBuf.size;
    sim::operation::UpdateOpMemCclBuffer(sim::GetVirPtrByDevPtr(reinterpret_cast<uint64_t>(*buffer)), *size);

    HCCL_VM_INFO("{} success, commId={}, buffer={}, size={}", __func__, commId, hcclBuf.addr, *size);
    return HCCL_SUCCESS;
}

/**
 * @brief 基于通信域获取通信线程。
 *
 * 根据传入的引擎类型、线程数量和每条线程的Notify数量，在数据库中创建对应数量的
 * HcclThread记录，并将每条线程的句柄（DB主键）填充到threads输出数组中。
 * 同一通信域内最多申请40条流、640个同步资源。
 *
 * @param comm 输入：通信域句柄。
 * @param engine 输入：通信引擎类型。
 * @param threadNum 输入：通信线程数量。
 * @param notifyNumPerThread 输入：每个通信线程中的同步资源（Notify）数量。
 * @param threads 输出：返回的通信线程句柄数组，需传入threadNum大小。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 获取成功
 * @retval HCCL_E_PTR comm或threads指针为空
 * @retval HCCL_E_PARA threadNum为0或超过单域上限
 * @retval HCCL_E_INTERNAL 通信域不存在或DB写入失败或超出资源上限
 */
HcclResult HcclThreadAcquire(
    HcclComm comm, CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread, ThreadHandle* threads)
{
    if (comm == nullptr) {
        HCCL_VM_ERROR("{}: comm is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (threadNum == 0) {
        HCCL_VM_ERROR("{}: threadNum is 0", __func__);
        return HCCL_E_PARA;
    }

    uint64_t commId = reinterpret_cast<uint64_t>(comm);

    auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    auto existingThreads = sim::runtime::Db::GetByPred<sim::runtime::HcclThread>(
        HcclSim::Storage::Eq(&sim::runtime::HcclThread::commId, commId));

    uint32_t existingThreadCount = existingThreads.ok() ? static_cast<uint32_t>(existingThreads->size()) : 0U;
    if (existingThreadCount + threadNum > MAX_THREAD_PER_COMM) {
        HCCL_VM_ERROR(
            "{}: threadNum {:d} exceeds limit, existing {:d}, max {:d}", __func__, threadNum, existingThreadCount,
            MAX_THREAD_PER_COMM);
        return HCCL_E_INTERNAL;
    }

    uint64_t existingNotifyCount = 0;
    if (existingThreads.ok()) {
        for (const auto& t : *existingThreads) {
            existingNotifyCount += t.notifyNum;
        }
    }
    // 计算新增 Notify 总数 = 线程数 × 每线程 Notify 数
    // 使用 uint64_t 防止乘法溢出（threadNum 最大 40，notifyNumPerThread 可能很大）
    uint64_t newNotifyCount = static_cast<uint64_t>(threadNum) * notifyNumPerThread;
    if (existingNotifyCount + newNotifyCount > MAX_NOTIFY_PER_COMM) {
        HCCL_VM_ERROR(
            "{}: notify count exceeds limit, existing {:d}, new {:d}, max {:d}", __func__, existingNotifyCount,
            newNotifyCount, MAX_NOTIFY_PER_COMM);
        return HCCL_E_INTERNAL;
    }

    // 单线程 Notify 数量不得超过 HcclThread.notifyId 数组容量
    if (notifyNumPerThread > MAX_NOTIFY_PER_THREAD) {
        HCCL_VM_ERROR(
            "{}: notifyNumPerThread={:d} exceeds per-thread cap {:d}", __func__, notifyNumPerThread,
            MAX_NOTIFY_PER_THREAD);
        return HCCL_E_PARA;
    }

    // 获取当前 runner，用于为 Notify 记录填写 create_ctx_id
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(g_cur_server_key, runner)) {
        HCCL_VM_ERROR("{}: GetCurrRunnerTls failed", __func__);
        return HCCL_E_INTERNAL;
    }
    uint64_t createCtxId = runner.current_ctx_id;

    auto rollbackPrevThreads = [threads](uint32_t upTo) {
        for (uint32_t j = 0; j < upTo; ++j) {
            if (threads[j] == 0u) {
                continue;
            }
            auto prev = sim::runtime::Db::GetById<sim::runtime::HcclThread>(threads[j]);
            if (prev.ok()) {
                ReleaseThreadNotifies(*prev);
            }
            sim::runtime::Db::Delete<sim::runtime::HcclThread>(
                HcclSim::Storage::Eq(&sim::runtime::HcclThread::id, threads[j]));
            threads[j] = 0u;
        }
    };

    // 批量创建线程记录，每个线程分配独立的 DB 主键作为句柄
    // 每条线程同步在 Notify 表中插入 notifyNumPerThread 条记录，并将 id 存入 notifyId 数组
    for (uint32_t i = 0; i < threadNum; i++) {
        sim::runtime::HcclThread hcclThread{};
        hcclThread.commId = commId;
        hcclThread.engine = static_cast<uint8_t>(engine);
        hcclThread.notifyNum = static_cast<uint16_t>(notifyNumPerThread);
        aclrtStream stream = nullptr;
        if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
            HCCL_VM_ERROR("{}: aclrtCreateStream failed (i={:d})", __func__, i);
            rollbackPrevThreads(i);
            return HCCL_E_INTERNAL;
        }
        hcclThread.streamId = reinterpret_cast<uint64_t>(stream);

        // 逐条创建 Notify 记录：device_notify_seq / value 不填写，create_ctx_id 取自 runner
        bool notifyOk = true;
        for (uint32_t n = 0; n < notifyNumPerThread; ++n) {
            sim::runtime::Notify notify{};
            notify.create_ctx_id = createCtxId;
            auto nidResult = sim::runtime::Db::Add<sim::runtime::Notify>(notify);
            if (!nidResult.ok()) {
                HCCL_VM_ERROR("{}: failed to add Notify record (i={:d}, n={:d})", __func__, i, n);
                for (uint32_t k = 0; k < n; ++k) {
                    sim::runtime::Db::Delete<sim::runtime::Notify>(
                        HcclSim::Storage::Eq(&sim::runtime::Notify::id, hcclThread.notifyId[k]));
                }
                notifyOk = false;
                break;
            }
            hcclThread.notifyId[n] = static_cast<uint32_t>(notify.id);
        }
        if (!notifyOk) {
            rollbackPrevThreads(i);
            return HCCL_E_INTERNAL;
        }

        auto threadIdResult = sim::runtime::Db::Add<sim::runtime::HcclThread>(hcclThread);
        if (!threadIdResult.ok()) {
            HCCL_VM_ERROR("{}: failed to add HcclThread index {:d}", __func__, i);
            ReleaseThreadNotifies(hcclThread);
            rollbackPrevThreads(i);
            return HCCL_E_INTERNAL;
        }
        threads[i] = static_cast<ThreadHandle>(hcclThread.id);
    }

    HCCL_VM_INFO(
        "{} success, commId={:d}, engine={:d}, threadNum={:d}, "
        "notifyNumPerThread={:d}",
        __func__, commId, static_cast<int>(engine), threadNum, notifyNumPerThread);
    return HCCL_SUCCESS;
}

/**
 * @brief 基于通信域获取通信线程（每线程独立配置Notify数量）。
 *
 * 与HcclThreadAcquire的区别：本接口通过ThreadConfig数组为每条线程单独指定Notify数量，
 * 而非所有线程共用同一个notifyNumPerThread。调用前需用ThreadConfigInit初始化config数组。
 * 同一通信域内最多申请40条流、640个同步资源。
 *
 * @param comm 输入：通信域句柄。
 * @param engine 输入：通信引擎类型。
 * @param threadNum 输入：通信线程数量。
 * @param type 输入：线程类型，不能为THREAD_TYPE_INVALID。
 * @param config 输入：每线程的配置数组，长度为threadNum，需先用ThreadConfigInit初始化。
 * @param threads 输出：返回的通信线程句柄数组，需传入threadNum大小。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 获取成功
 * @retval HCCL_E_PTR comm、threads或config指针为空
 * @retval HCCL_E_PARA threadNum为0、线程类型无效、config未初始化或超过单线程上限
 * @retval HCCL_E_INTERNAL 通信域不存在、DB写入失败或超出资源上限
 */
HcclResult HcclThreadAcquireWithConfig(
    HcclComm comm, CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config,
    ThreadHandle* threads)
{
    if (comm == nullptr) {
        HCCL_VM_ERROR("{}: comm is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (config == nullptr) {
        HCCL_VM_ERROR("{}: config is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (threadNum == 0) {
        HCCL_VM_ERROR("{}: threadNum is 0", __func__);
        return HCCL_E_PARA;
    }
    if (type == THREAD_TYPE_INVALID) {
        HCCL_VM_ERROR("{}: thread type {:d} is invalid", __func__, static_cast<int>(type));
        return HCCL_E_PARA;
    }
    // 校验每条线程的config已通过ThreadConfigInit初始化，并满足单线程Notify上限
    for (uint32_t i = 0; i < threadNum; ++i) {
        if (config[i].header.magicWord != HCOMM_THREAD_CONFIG_MAGIC_WORD) {
            HCCL_VM_ERROR(
                "{}: config[{:d}] magicWord=0x{:08x} mismatch, "
                "expected=0x{:08x}, call ThreadConfigInit first",
                __func__, i, config[i].header.magicWord, HCOMM_THREAD_CONFIG_MAGIC_WORD);
            return HCCL_E_PARA;
        }
        if (config[i].notifyNumPerThread > MAX_NOTIFY_PER_THREAD) {
            HCCL_VM_ERROR(
                "{}: config[{:d}] notifyNumPerThread={:d} "
                "exceeds per-thread cap {:d}",
                __func__, i, config[i].notifyNumPerThread, MAX_NOTIFY_PER_THREAD);
            return HCCL_E_PARA;
        }
    }

    uint64_t commId = reinterpret_cast<uint64_t>(comm);

    auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    auto existingThreads = sim::runtime::Db::GetByPred<sim::runtime::HcclThread>(
        HcclSim::Storage::Eq(&sim::runtime::HcclThread::commId, commId));

    uint32_t existingThreadCount = existingThreads.ok() ? static_cast<uint32_t>(existingThreads->size()) : 0U;
    if (existingThreadCount + threadNum > MAX_THREAD_PER_COMM) {
        HCCL_VM_ERROR(
            "{}: threadNum {:d} exceeds limit, existing {:d}, max {:d}", __func__, threadNum, existingThreadCount,
            MAX_THREAD_PER_COMM);
        return HCCL_E_INTERNAL;
    }

    uint64_t existingNotifyCount = 0;
    if (existingThreads.ok()) {
        for (const auto& t : *existingThreads) {
            existingNotifyCount += t.notifyNum;
        }
    }
    // 计算新增 Notify 总数 = 各线程 config[i].notifyNumPerThread 之和
    uint64_t newNotifyCount = 0;
    for (uint32_t i = 0; i < threadNum; ++i) {
        newNotifyCount += config[i].notifyNumPerThread;
    }
    if (existingNotifyCount + newNotifyCount > MAX_NOTIFY_PER_COMM) {
        HCCL_VM_ERROR(
            "{}: notify count exceeds limit, existing {:d}, new {:d}, max {:d}", __func__, existingNotifyCount,
            newNotifyCount, MAX_NOTIFY_PER_COMM);
        return HCCL_E_INTERNAL;
    }

    // 获取当前 runner，用于为 Notify 记录填写 create_ctx_id
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(g_cur_server_key, runner)) {
        HCCL_VM_ERROR("{}: GetCurrRunnerTls failed", __func__);
        return HCCL_E_INTERNAL;
    }
    uint64_t createCtxId = runner.current_ctx_id;

    auto rollbackPrevThreads = [threads](uint32_t upTo) {
        for (uint32_t j = 0; j < upTo; ++j) {
            if (threads[j] == 0u) {
                continue;
            }
            auto prev = sim::runtime::Db::GetById<sim::runtime::HcclThread>(threads[j]);
            if (prev.ok()) {
                ReleaseThreadNotifies(*prev);
            }
            sim::runtime::Db::Delete<sim::runtime::HcclThread>(
                HcclSim::Storage::Eq(&sim::runtime::HcclThread::id, threads[j]));
            threads[j] = 0u;
        }
    };

    // 批量创建线程记录，每条线程按 config[i].notifyNumPerThread 分配 Notify
    for (uint32_t i = 0; i < threadNum; i++) {
        uint16_t notifyNumPerThread = config[i].notifyNumPerThread;

        sim::runtime::HcclThread hcclThread{};
        hcclThread.commId = commId;
        hcclThread.engine = static_cast<uint8_t>(engine);
        hcclThread.notifyNum = notifyNumPerThread;
        aclrtStream stream = nullptr;
        if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
            HCCL_VM_ERROR("{}: aclrtCreateStream failed (i={:d})", __func__, i);
            rollbackPrevThreads(i);
            return HCCL_E_INTERNAL;
        }
        hcclThread.streamId = reinterpret_cast<uint64_t>(stream);

        bool notifyOk = true;
        for (uint32_t n = 0; n < notifyNumPerThread; ++n) {
            sim::runtime::Notify notify{};
            notify.create_ctx_id = createCtxId;
            auto nidResult = sim::runtime::Db::Add<sim::runtime::Notify>(notify);
            if (!nidResult.ok()) {
                HCCL_VM_ERROR("{}: failed to add Notify record (i={:d}, n={:d})", __func__, i, n);
                for (uint32_t k = 0; k < n; ++k) {
                    sim::runtime::Db::Delete<sim::runtime::Notify>(
                        HcclSim::Storage::Eq(&sim::runtime::Notify::id, hcclThread.notifyId[k]));
                }
                notifyOk = false;
                break;
            }
            hcclThread.notifyId[n] = static_cast<uint32_t>(notify.id);
        }
        if (!notifyOk) {
            rollbackPrevThreads(i);
            return HCCL_E_INTERNAL;
        }

        auto threadIdResult = sim::runtime::Db::Add<sim::runtime::HcclThread>(hcclThread);
        if (!threadIdResult.ok()) {
            HCCL_VM_ERROR("{}: failed to add HcclThread index {:d}", __func__, i);
            ReleaseThreadNotifies(hcclThread);
            rollbackPrevThreads(i);
            return HCCL_E_INTERNAL;
        }
        threads[i] = static_cast<ThreadHandle>(hcclThread.id);
    }

    HCCL_VM_INFO(
        "{} success, commId={:d}, engine={:d}, threadNum={:d}, "
        "type={:d}, totalNotify={:d}",
        __func__, commId, static_cast<int>(engine), threadNum, static_cast<int>(type), newNotifyCount);
    return HCCL_SUCCESS;
}

/**
 * @brief 基于已有runtime stream获取指定notifyNum的通信线程资源。
 *
 * 与HcclThreadAcquire的区别：本接口一次只创建1条线程，并将其与调用者传入的
 * aclrtStream（runtime流表中的流ID）绑定。适用于HOST CPU+TS、CCU引擎等需要
 * 复用已有流的场景。
 * 同一通信域内最多申请40条流、640个同步资源。
 *
 * @param comm 输入：通信域句柄。
 * @param engine 输入：通信引擎类型。
 * @param stream 输入：已有runtime流（aclrtStream为stream表的id）。
 * @param notifyNum 输入：线程中的同步资源（Notify）数量。
 * @param thread 输出：返回的通信线程句柄（单条）。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 获取成功
 * @retval HCCL_E_PTR comm或thread指针为空
 * @retval HCCL_E_INTERNAL 通信域不存在、DB写入失败或超出资源上限
 */
// 进程级主流线程缓存：键为 (commId, engine, stream)，值为已分配的 ThreadHandle。
// 与 hcomm ThreadMgr::mainThread_ 语义一致——同通信域内同 stream 复用同一条线程，
// 避免同一 stream 反复申请线程撞硬件配额（每通信域 40 线程上限）。
static std::mutex g_streamThreadMapMtx;
static std::map<std::tuple<uint64_t, int, aclrtStream>, ThreadHandle> g_streamThreadMap;

// 补充既有线程 Notify 的助手（定义于本文件 HcclDedicatedThreadAcquire 段，前置声明复用）
static HcclResult SupplementDedicatedThreadNotify(ThreadHandle handle, uint32_t target, uint64_t createCtxId);

/**
 * @brief 校验新增一条流线程的通信域级配额（线程条数与 Notify 总数上限）。
 * @param commId    输入：通信域 id。
 * @param notifyNum 输入：本次申请线程携带的 Notify 数量。
 * @return HcclResult 配额内返回 HCCL_SUCCESS，超限返回 HCCL_E_INTERNAL。
 */
static HcclResult CheckStreamThreadQuota(uint64_t commId, uint32_t notifyNum)
{
    auto existingThreads = sim::runtime::Db::GetByPred<sim::runtime::HcclThread>(
        HcclSim::Storage::Eq(&sim::runtime::HcclThread::commId, commId));
    const uint32_t existingThreadCount = existingThreads.ok() ? static_cast<uint32_t>(existingThreads->size()) : 0U;
    if (existingThreadCount + 1 > MAX_THREAD_PER_COMM) {
        HCCL_VM_ERROR(
            "{}: thread count exceeds limit, existing {:d}, max {:d}", __func__, existingThreadCount,
            MAX_THREAD_PER_COMM);
        return HCCL_E_INTERNAL;
    }
    uint64_t existingNotifyCount = 0;
    if (existingThreads.ok()) {
        for (const auto& t : *existingThreads) {
            existingNotifyCount += t.notifyNum;
        }
    }
    if (existingNotifyCount + notifyNum > MAX_NOTIFY_PER_COMM) {
        HCCL_VM_ERROR(
            "{}: notify count exceeds limit, existing {:d}, new {:d}, max {:d}", __func__, existingNotifyCount,
            notifyNum, MAX_NOTIFY_PER_COMM);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 创建流线程记录（含逐条 Notify 与失败回滚）。
 * @param commId      输入：通信域 id。
 * @param engine      输入：通信引擎类型。
 * @param stream      输入：关联流（为空时自动创建新流）。
 * @param notifyNum   输入：线程携带的 Notify 数量。
 * @param createCtxId 输入：Notify 记录的 create_ctx_id。
 * @param thread      输出：返回的线程句柄（DB 主键）。
 * @return HcclResult 成功返回 HCCL_SUCCESS，否则返回对应错误码。
 */
static HcclResult CreateStreamThreadRecord(
    uint64_t commId, CommEngine engine, aclrtStream stream, uint32_t notifyNum, uint64_t createCtxId,
    ThreadHandle* thread)
{
    sim::runtime::HcclThread hcclThread{};
    hcclThread.commId = commId;
    hcclThread.engine = static_cast<uint8_t>(engine);
    hcclThread.notifyNum = static_cast<uint16_t>(notifyNum);
    aclrtStream actualStream = stream;
    if (actualStream == nullptr && aclrtCreateStream(&actualStream) != ACL_SUCCESS) {
        HCCL_VM_ERROR("{}: aclrtCreateStream failed", __func__);
        return HCCL_E_INTERNAL;
    }
    hcclThread.streamId = reinterpret_cast<uint64_t>(actualStream);

    // 逐条创建 Notify 记录：device_notify_seq / value 不填写，create_ctx_id 取自 runner
    for (uint32_t n = 0; n < notifyNum; ++n) {
        sim::runtime::Notify notify{};
        notify.create_ctx_id = createCtxId;
        auto nidResult = sim::runtime::Db::Add<sim::runtime::Notify>(notify);
        if (!nidResult.ok()) {
            HCCL_VM_ERROR("{}: failed to add Notify record (n={:d})", __func__, n);
            ReleaseThreadNotifies(hcclThread);
            return HCCL_E_INTERNAL;
        }
        hcclThread.notifyId[n] = static_cast<uint32_t>(notify.id);
    }

    if (!sim::runtime::Db::Add<sim::runtime::HcclThread>(hcclThread).ok()) {
        HCCL_VM_ERROR("{}: failed to add HcclThread record", __func__);
        ReleaseThreadNotifies(hcclThread);
        return HCCL_E_INTERNAL;
    }
    *thread = static_cast<ThreadHandle>(hcclThread.id);
    return HCCL_SUCCESS;
}

/**
 * @brief 基于通信域和流获取通信线程（主流获取）。
 *
 * 与真实 hcomm ThreadMgr::mainThread_ 语义一致：同一通信域内同一 (engine, stream)
 * 复用同一条线程（notify 不足时增量补充），避免算子迭代反复申请耗尽
 * 每通信域 40 线程的硬件配额；未命中时创建新线程并缓存映射。
 *
 * @param comm      输入：通信域句柄。
 * @param engine    输入：通信引擎类型。
 * @param stream    输入：关联流（nullptr 时按 (commId,engine,nullptr) 键复用/创建）。
 * @param notifyNum 输入：线程携带的 Notify 数量。
 * @param thread    输出：返回的线程句柄。
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 获取成功
 * @retval HCCL_E_PTR comm指针为空
 * @retval HCCL_E_PARA notifyNum超过单线程上限
 * @retval HCCL_E_INTERNAL 通信域不存在、DB写入失败或超出资源上限
 */
HcclResult HcclThreadAcquireWithStream(
    HcclComm comm, CommEngine engine, aclrtStream stream, uint32_t notifyNum, ThreadHandle* thread)
{
    if (comm == nullptr || thread == nullptr) {
        HCCL_VM_ERROR("{}: comm or thread is nullptr", __func__);
        return HCCL_E_PTR;
    }
    const uint64_t commId = reinterpret_cast<uint64_t>(comm);
    if (!sim::runtime::Db::GetById<sim::runtime::Communicator>(commId).ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }
    if (notifyNum > MAX_NOTIFY_PER_THREAD) {
        HCCL_VM_ERROR("{}: notifyNum={:d} exceeds per-thread cap {:d}", __func__, notifyNum, MAX_NOTIFY_PER_THREAD);
        return HCCL_E_PARA;
    }

    // 【核心步骤 1】按 (commId, engine, stream) 查询缓存的主流线程，命中则复用
    // 并增量补充 Notify（与真实 ThreadMgr::mainThread_ 复用 + SupplementNotify 语义一致）
    sim::runtime::Runner runner{};
    if (!sim::runtime::GetCurrRunnerTls(g_cur_server_key, runner)) {
        HCCL_VM_ERROR("{}: GetCurrRunnerTls failed", __func__);
        return HCCL_E_INTERNAL;
    }
    const auto key = std::make_tuple(commId, static_cast<int>(engine), stream);
    {
        std::lock_guard<std::mutex> lock(g_streamThreadMapMtx);
        const auto it = g_streamThreadMap.find(key);
        if (it != g_streamThreadMap.end()) {
            if (sim::runtime::Db::GetById<sim::runtime::HcclThread>(static_cast<uint64_t>(it->second)).ok()) {
                *thread = it->second;
                return SupplementDedicatedThreadNotify(*thread, notifyNum, runner.current_ctx_id);
            }
            g_streamThreadMap.erase(it); // 缓存句柄失效（通信域已销毁），走创建路径
        }
    }

    // 【核心步骤 2】未命中：配额校验 + 创建线程 + 缓存映射
    const HcclResult quotaRet = CheckStreamThreadQuota(commId, notifyNum);
    if (quotaRet != HCCL_SUCCESS) {
        return quotaRet;
    }
    const HcclResult ret = CreateStreamThreadRecord(commId, engine, stream, notifyNum, runner.current_ctx_id, thread);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    {
        std::lock_guard<std::mutex> lock(g_streamThreadMapMtx);
        g_streamThreadMap[key] = *thread;
    }
    HCCL_VM_INFO(
        "{} success, commId={:d}, engine={:d}, stream={:p}, notifyNum={:d}, thread={:d}", __func__, commId,
        static_cast<int>(engine), stream, notifyNum, *thread);
    return HCCL_SUCCESS;
}

// 进程级专用线程缓存：键为 (commId, useType)，值为已分配的 ThreadHandle。
// 与 hcomm ThreadMgr::dedicatedThreadMap_ 语义一致——同通信域内同 useType 复用同一线程。
static std::mutex g_dedicatedThreadMapMtx;
static std::map<std::pair<uint64_t, int>, ThreadHandle> g_dedicatedThreadMap;

/**
 * @brief 为已存在的专用线程补充 Notify 至目标数量。
 *
 * 复用专用线程时，若本次请求的 notifyNumPerThread 大于线程当前已持有的 Notify 数，
 * 需新增 (target - current) 条 Notify 记录并追加到 HcclThread.notifyId 数组末尾。
 * 当前已持有数量 >= target 时为无操作。失败时已新增的 Notify 记录会被回滚删除。
 *
 * @param handle   目标线程句柄（DB 主键）。
 * @param target   目标 Notify 数量。
 * @param createCtxId 新增 Notify 记录的 create_ctx_id。
 *
 * @return HcclResult 成功返回 HCCL_SUCCESS，其他失败。
 */
static HcclResult SupplementDedicatedThreadNotify(ThreadHandle handle, uint32_t target, uint64_t createCtxId)
{
    if (target > MAX_NOTIFY_PER_THREAD) {
        HCCL_VM_ERROR("{}: target notifyNum={:d} exceeds per-thread cap {:d}", __func__, target, MAX_NOTIFY_PER_THREAD);
        return HCCL_E_PARA;
    }
    auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(static_cast<uint64_t>(handle));
    if (!optThread.ok()) {
        HCCL_VM_ERROR("{}: dedicated thread {:d} not found in DB", __func__, handle);
        return HCCL_E_INTERNAL;
    }
    uint32_t current = optThread->notifyNum;
    if (current >= target) {
        return HCCL_SUCCESS;
    }

    // 预分配新增 Notify 记录，失败时整批回滚
    std::vector<uint32_t> newNotifyIds;
    newNotifyIds.reserve(target - current);
    for (uint32_t n = current; n < target; ++n) {
        sim::runtime::Notify notify{};
        notify.create_ctx_id = createCtxId;
        auto nidResult = sim::runtime::Db::Add<sim::runtime::Notify>(notify);
        if (!nidResult.ok()) {
            HCCL_VM_ERROR("{}: failed to add Notify record (n={:d})", __func__, n);
            for (uint32_t id : newNotifyIds) {
                sim::runtime::Db::Delete<sim::runtime::Notify>(HcclSim::Storage::Eq(&sim::runtime::Notify::id, id));
            }
            return HCCL_E_INTERNAL;
        }
        newNotifyIds.push_back(static_cast<uint32_t>(notify.id));
    }

    // 追加到线程记录的 notifyId 数组并更新 notifyNum
    std::vector<uint8_t> notifyIdBlob(sizeof(optThread->notifyId));
    std::memcpy(notifyIdBlob.data(), optThread->notifyId, sizeof(optThread->notifyId));
    uint16_t newNum = optThread->notifyNum;
    for (uint32_t id : newNotifyIds) {
        if (newNum >= MAX_NOTIFY_PER_THREAD) {
            break;
        }
        std::memcpy(notifyIdBlob.data() + newNum * sizeof(uint32_t), &id, sizeof(id));
        newNum++;
    }
    auto updateResult = sim::runtime::Db::Update<sim::runtime::HcclThread>(
        HcclSim::Storage::Eq(&sim::runtime::HcclThread::id, static_cast<uint64_t>(handle)),
        HcclSim::Storage::Assignment<sim::runtime::HcclThread>{
            HcclSim::Storage::RecordSchema<sim::runtime::HcclThread>::FieldIdOf(&sim::runtime::HcclThread::notifyId),
            false, HcclSim::Storage::DbScalar{notifyIdBlob}},
        HcclSim::Storage::Set(&sim::runtime::HcclThread::notifyNum, newNum));
    if (!updateResult.ok()) {
        HCCL_VM_ERROR("{}: Update HcclThread {:d} failed", __func__, handle);
        for (uint32_t id : newNotifyIds) {
            sim::runtime::Db::Delete<sim::runtime::Notify>(HcclSim::Storage::Eq(&sim::runtime::Notify::id, id));
        }
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

bool HcommIsSupportHcclDedicatedThreadAcquire()
{
    HCCL_VM_INFO("{}: stub returns true", __func__);
    return false;
}

/**
 * @brief 基于通信域获取专用通信线程。
 *
 * 每个通信域内对同一种 useType 维护一个专用线程：若已存在则直接复用并按需补充 Notify
 * （当请求的 notifyNumPerThread 大于线程当前已持有数量时补足差额，否则不变），不存在则
 * 调用与 HcclThreadAcquire 一致的逻辑创建 1 条线程（engine=COMM_ENGINE_CPU，
 * threadNum=1）并缓存映射。对于 HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE，未命中时
 * 不创建，直接返回 thread=0（与 hcomm 图模式行为一致）。
 * 同一通信域内最多申请40条流、640个同步资源。
 *
 * @param comm 输入：通信域句柄。
 * @param useType 输入：专用线程使用类型，不能为 HCCL_DED_THREAD_TYPE_INVALID。
 * @param notifyNumPerThread 输入：该线程需持有的 Notify 数量。
 * @param thread 输出：返回的线程句柄（单条，GE 未命中时为 0）。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 获取成功
 * @retval HCCL_E_PTR comm或thread指针为空
 * @retval HCCL_E_PARA useType无效或notifyNumPerThread超过单线程上限
 * @retval HCCL_E_INTERNAL 通信域不存在、DB写入失败或超出资源上限
 */
HcclResult HcclDedicatedThreadAcquire(
    HcclComm comm, HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread)
{
    if (comm == nullptr) {
        HCCL_VM_ERROR("{}: comm is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (thread == nullptr) {
        HCCL_VM_ERROR("{}: thread is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (useType == HCCL_DED_THREAD_TYPE_INVALID) {
        HCCL_VM_ERROR("{}: useType {:d} is invalid", __func__, static_cast<int>(useType));
        return HCCL_E_PARA;
    }
    if (notifyNumPerThread > MAX_NOTIFY_PER_THREAD) {
        HCCL_VM_ERROR(
            "{}: notifyNumPerThread={:d} exceeds per-thread cap {:d}", __func__, notifyNumPerThread,
            MAX_NOTIFY_PER_THREAD);
        return HCCL_E_PARA;
    }

    *thread = 0;
    uint64_t commId = reinterpret_cast<uint64_t>(comm);

    auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    // 获取当前 runner，用于为 Notify 记录填写 create_ctx_id
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(g_cur_server_key, runner)) {
        HCCL_VM_ERROR("{}: GetCurrRunnerTls failed", __func__);
        return HCCL_E_INTERNAL;
    }
    uint64_t createCtxId = runner.current_ctx_id;

    auto key = std::make_pair(commId, static_cast<int>(useType));
    std::lock_guard<std::mutex> lock(g_dedicatedThreadMapMtx);

    auto it = g_dedicatedThreadMap.find(key);
    if (it != g_dedicatedThreadMap.end()) {
        // 校验缓存句柄在 DB 中仍然有效（通信域销毁/线程释放后会失效，此时走重建路径）
        auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(static_cast<uint64_t>(it->second));
        if (optThread.ok()) {
            *thread = it->second;
            HcclResult ret = SupplementDedicatedThreadNotify(*thread, notifyNumPerThread, createCtxId);
            if (ret != HCCL_SUCCESS) {
                HCCL_VM_ERROR(
                    "{}: supplement notify failed, thread={:d}, target={:d}", __func__, it->second, notifyNumPerThread);
                return ret;
            }
            HCCL_VM_INFO(
                "{} reuse dedicated thread, commId={:d}, useType={:d}, "
                "thread={:d}, notifyNumPerThread={:d}",
                __func__, commId, static_cast<int>(useType), *thread, notifyNumPerThread);
            return HCCL_SUCCESS;
        }
        // 缓存失效，清除并走创建路径
        g_dedicatedThreadMap.erase(it);
    }

    // AICPU_LAUNCH_GE 未命中时：图模式下不创建，直接返回 thread=0（与 hcomm 行为一致）
    if (useType == HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE) {
        HCCL_VM_INFO(
            "{} dedicated thread not found for AICPU_LAUNCH_GE, "
            "return thread=0, commId={:d}",
            __func__, commId);
        return HCCL_SUCCESS;
    }

    // 资源上限校验：新增 1 条线程 + notifyNumPerThread 个 Notify
    auto existingThreads = sim::runtime::Db::GetByPred<sim::runtime::HcclThread>(
        HcclSim::Storage::Eq(&sim::runtime::HcclThread::commId, commId));
    const uint32_t existingThreadNum = existingThreads.ok() ? static_cast<uint32_t>(existingThreads->size()) : 0U;
    if (existingThreadNum + 1 > MAX_THREAD_PER_COMM) {
        HCCL_VM_ERROR(
            "{}: thread count exceeds limit, existing {:d}, max {:d}", __func__, existingThreadNum,
            MAX_THREAD_PER_COMM);
        return HCCL_E_INTERNAL;
    }
    uint64_t existingNotifyCount = 0;
    if (existingThreads.ok()) {
        for (const auto& t : *existingThreads) {
            existingNotifyCount += t.notifyNum;
        }
    }
    if (existingNotifyCount + notifyNumPerThread > MAX_NOTIFY_PER_COMM) {
        HCCL_VM_ERROR(
            "{}: notify count exceeds limit, existing {:d}, new {:d}, max {:d}", __func__, existingNotifyCount,
            notifyNumPerThread, MAX_NOTIFY_PER_COMM);
        return HCCL_E_INTERNAL;
    }

    // 创建专用线程记录（threadNum=1, engine=COMM_ENGINE_CPU）
    sim::runtime::HcclThread hcclThread{};
    hcclThread.commId = commId;
    hcclThread.engine = static_cast<uint8_t>(COMM_ENGINE_CPU);
    hcclThread.notifyNum = static_cast<uint16_t>(notifyNumPerThread);
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        HCCL_VM_ERROR("{}: aclrtCreateStream failed", __func__);
        return HCCL_E_INTERNAL;
    }
    hcclThread.streamId = reinterpret_cast<uint64_t>(stream);

    bool notifyOk = true;
    for (uint32_t n = 0; n < notifyNumPerThread; ++n) {
        sim::runtime::Notify notify{};
        notify.create_ctx_id = createCtxId;
        auto nidResult = sim::runtime::Db::Add<sim::runtime::Notify>(notify);
        if (!nidResult.ok()) {
            HCCL_VM_ERROR("{}: failed to add Notify record (n={:d})", __func__, n);
            for (uint32_t k = 0; k < n; ++k) {
                sim::runtime::Db::Delete<sim::runtime::Notify>(
                    HcclSim::Storage::Eq(&sim::runtime::Notify::id, hcclThread.notifyId[k]));
            }
            notifyOk = false;
            break;
        }
        hcclThread.notifyId[n] = static_cast<uint32_t>(notify.id);
    }
    if (!notifyOk) {
        return HCCL_E_INTERNAL;
    }

    auto threadIdResult = sim::runtime::Db::Add<sim::runtime::HcclThread>(hcclThread);
    if (!threadIdResult.ok()) {
        HCCL_VM_ERROR("{}: failed to add HcclThread record", __func__);
        ReleaseThreadNotifies(hcclThread);
        return HCCL_E_INTERNAL;
    }

    g_dedicatedThreadMap[key] = static_cast<ThreadHandle>(hcclThread.id);
    *thread = static_cast<ThreadHandle>(hcclThread.id);

    HCCL_VM_INFO(
        "{} success, commId={:d}, useType={:d}, thread={:d}, "
        "notifyNumPerThread={:d}",
        __func__, commId, static_cast<int>(useType), *thread, notifyNumPerThread);
    return HCCL_SUCCESS;
}

/**
 * @brief 获取Thread底层资源信息。
 *
 * 根据线程句柄和资源类型，从数据库中查询HcclThread记录并返回对应的资源信息。
 * 当前仅支持THREAD_RES_TYPE_STREAM类型，返回关联的runtime流句柄。
 *
 * @param comm 输入：通信域句柄。
 * @param thread 输入：线程句柄，由HcclThreadAcquire/HcclThreadAcquireWithStream创建。
 * @param resType 输入：底层资源类型，当前仅支持THREAD_RES_TYPE_STREAM。
 * @param infoLen 输入：info缓冲区长度，必须等于目标资源类型的大小。
 * @param info 输出：指向资源信息的缓冲区指针，具体内容由resType决定。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 获取成功
 * @retval HCCL_E_PTR comm或info指针为空
 * @retval HCCL_E_PARA resType不支持或infoLen不匹配
 * @retval HCCL_E_INTERNAL 通信域不存在或thread未找到
 */
HcclResult
HcclThreadResGetInfo(HcclComm comm, ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void** info)
{
    if (comm == nullptr) {
        HCCL_VM_ERROR("{}: comm is nullptr", __func__);
        return HCCL_E_PTR;
    }

    if (resType != THREAD_RES_TYPE_STREAM) {
        HCCL_VM_ERROR("{}: unsupported resType {:d}", __func__, static_cast<int>(resType));
        return HCCL_E_PARA;
    }
    if (infoLen != sizeof(ThreadResTypeStream)) {
        HCCL_VM_ERROR(
            "{}: infoLen {:d} does not match ThreadResTypeStream size {:d}", __func__, infoLen,
            static_cast<uint32_t>(sizeof(ThreadResTypeStream)));
        return HCCL_E_PARA;
    }

    auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(thread);
    if (!optThread.ok()) {
        HCCL_VM_ERROR("{}: HcclThread {:d} not found", __func__, thread);
        return HCCL_E_INTERNAL;
    }

    // 从数据库读取存储的 stream 地址，转换回指针类型
    // 注意：此转换依赖于 stream 在创建后未被释放
    ThreadResTypeStream stream = reinterpret_cast<aclrtStream>(optThread->streamId);
    *reinterpret_cast<ThreadResTypeStream*>(info) = stream;

    HCCL_VM_INFO("{} success, thread={:d}, stream={:p}", __func__, thread, stream);
    return HCCL_SUCCESS;
}

/**
 * @brief 基于通信域获取多个通信通道。
 *
 * 根据传入的HcclChannelDesc数组，检查通信域中是否已存在匹配的通道
 * （commId + engine + remoteRankId + channelProtocol），若存在则复用，否则新建。
 * 对于CCU引擎，notifyNum固定为8，不支持外部配置。
 * channelNum的取值范围为(0, 1024 * 1024]。
 *
 * @param comm 输入：通信域句柄。
 * @param engine 输入：通信引擎类型。
 * @param channelDescs 输入：通道描述数组，由HcclChannelDescInit初始化。
 * @param channelNum 输入：通道数量，取值范围(0, 1024 * 1024]。
 * @param channels 输出：返回的通道句柄数组，需传入channelNum大小。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 成功
 * @retval HCCL_E_PTR comm、channelDescs或channels指针为空
 * @retval HCCL_E_PARA channelNum为0或超过上限
 * @retval HCCL_E_INTERNAL 通信域不存在或DB写入失败
 */
HcclResult HcclChannelAcquire(
    HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle* channels)
{
    if (comm == nullptr || channelDescs == nullptr) {
        HCCL_VM_ERROR("{}: comm or channelDescs is nullptr", __func__);
        return HCCL_E_PTR;
    }

    if (channelNum == 0 || channelNum > MAX_CHANNEL_NUM) {
        HCCL_VM_ERROR("{}: channelNum {:d} out of range (0, {:d}]", __func__, channelNum, MAX_CHANNEL_NUM);
        return HCCL_E_PARA;
    }

    uint64_t commId = reinterpret_cast<uint64_t>(comm);

    auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    // 获取当前 runner，用于为 Notify 记录填写 create_ctx_id
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(g_cur_server_key, runner)) {
        HCCL_VM_ERROR("{}: GetCurrRunnerTls failed", __func__);
        return HCCL_E_INTERNAL;
    }
    uint64_t createCtxId = runner.current_ctx_id;

    // 本次调用内每个 remoteRank 已处理的描述符数量：用于把同 remoteRank 的第 k 个描述符
    // 映射到第 k 条通道，避免并发多路（channelNum>1 指向同一 remoteRank）被合并成同一条通道。
    std::map<uint64_t, uint32_t> remoteOrdinal;

    // 遍历所有通道描述，逐个创建或复用通道
    for (uint32_t i = 0; i < channelNum; i++) {
        const HcclChannelDesc& desc = channelDescs[i];
        uint64_t remoteRankId = desc.remoteRank;

        // CCU 引擎限制：NotifyNum 固定为 8，忽略用户配置
        // 非 CCU 引擎使用用户指定的 notifyNum
        uint64_t notifyNum = (engine == COMM_ENGINE_CCU) ? CCU_DEFAULT_NOTIFY_NUM : desc.notifyNum;

        // CCU 引擎不支持外部 memHandle 交换（仅使用通信域内置缓冲区）
        bool isCcuWithExtraMem = (engine == COMM_ENGINE_CCU) && (desc.memHandleNum > 0);
        if (isCcuWithExtraMem) {
            HCCL_VM_ERROR("{}: CCU engine does not support external memHandle exchange", __func__);
            return HCCL_E_PARA;
        }

        // 单通道 Notify 数量不得超过 HcclChannel.notifyId 数组容量
        if (notifyNum > MAX_NOTIFY_PER_CHANNEL) {
            HCCL_VM_ERROR(
                "{}: notifyNum={:d} exceeds per-channel cap {:d} (i={:d})", __func__, notifyNum, MAX_NOTIFY_PER_CHANNEL,
                i);
            return HCCL_E_PARA;
        }

        // 通道复用逻辑：取同 (commId, engine, remoteRankId) 键下、按主键升序的第 ordinal 条通道复用；
        // 数量不足则新建。ordinal 取本次调用内该 remoteRank 的描述符序号，使并发多路各自映射到
        // 独立通道（各自独立 notify 数组），对齐真机/南向语义，避免 notify 撞号。
        const uint32_t ordinal = remoteOrdinal[remoteRankId]++;
        auto existingChannels = sim::runtime::Db::GetByPred<sim::runtime::HcclChannel>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::HcclChannel::commId, commId),
            HcclSim::Storage::Eq(&sim::runtime::HcclChannel::engine, static_cast<uint8_t>(engine)),
            HcclSim::Storage::Eq(&sim::runtime::HcclChannel::remoteRankId, remoteRankId)));

        if (existingChannels.ok() && existingChannels->size() > ordinal) {
            channels[i] = static_cast<ChannelHandle>((*existingChannels)[ordinal].id);
            continue;
        }

        sim::runtime::HcclChannel hcclChannel{};
        hcclChannel.commId = commId;
        hcclChannel.engine = static_cast<uint8_t>(engine);
        hcclChannel.remoteRankId = remoteRankId;
        hcclChannel.notifyNum = static_cast<uint16_t>(notifyNum);
        hcclChannel.status = true;

        // 逐条创建 Notify 记录：device_notify_seq / value 不填写，create_ctx_id 取自 runner
        bool notifyOk = true;
        for (uint32_t n = 0; n < notifyNum; ++n) {
            sim::runtime::Notify notify{};
            notify.create_ctx_id = createCtxId;
            auto nidResult = sim::runtime::Db::Add<sim::runtime::Notify>(notify);
            if (!nidResult.ok()) {
                HCCL_VM_ERROR("{}: failed to add Notify record (i={:d}, n={:d})", __func__, i, n);
                for (uint32_t k = 0; k < n; ++k) {
                    sim::runtime::Db::Delete<sim::runtime::Notify>(
                        HcclSim::Storage::Eq(&sim::runtime::Notify::id, hcclChannel.notifyId[k]));
                }
                notifyOk = false;
                break;
            }
            hcclChannel.notifyId[n] = notify.id;
        }
        if (!notifyOk) {
            return HCCL_E_INTERNAL;
        }

        auto channelIdResult = sim::runtime::Db::Add<sim::runtime::HcclChannel>(hcclChannel);
        if (!channelIdResult.ok()) {
            HCCL_VM_ERROR("{}: failed to add HcclChannel index {:d}", __func__, i);
            ReleaseChannelNotifies(hcclChannel);
            return HCCL_E_INTERNAL;
        }
        channels[i] = static_cast<ChannelHandle>(hcclChannel.id);
    }

    HCCL_VM_INFO(
        "{} success, commId={:d}, engine={:d}, channelNum={:d}", __func__, commId, static_cast<int>(engine),
        channelNum);
    return HCCL_SUCCESS;
}

/**
 * @brief 查询通信通道是否已存在。
 *
 * 根据传入的HcclChannelDesc数组，检查通信域中是否已存在匹配的通道
 * （commId + engine + remoteRankId）。已存在的通道在channels中返回其句柄，
 * 不存在的返回0（空句柄），由调用方决定是否新建（通常随后调用HcclChannelAcquire）。
 *
 * 该接口为CANN 9.3.0新激活的北向接口：真实HCCL在CcuAcquireKernelChannels中会先
 * 调用本接口查询可复用通道。若不在此处打桩，调用会落到真实libhcomm实现，而真实
 * 实现会按真实hcclComm解引用北向的通信域句柄（北向为DB行号假句柄），导致段错误。
 *
 * @param comm 输入：通信域句柄。
 * @param engine 输入：通信引擎类型。
 * @param channelDescs 输入：通道描述数组，由HcclChannelDescInit初始化。
 * @param channelNum 输入：通道数量，取值范围(0, 1024 * 1024]。
 * @param channels 输出：查询结果句柄数组，需传入channelNum大小；不存在的通道置0。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 查询成功
 * @retval HCCL_E_PTR comm、channelDescs或channels指针为空
 * @retval HCCL_E_PARA channelNum为0或超过上限
 * @retval HCCL_E_INTERNAL 通信域不存在
 */
HcclResult HcclChannelQuery(
    HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle* channels)
{
    if (comm == nullptr || channelDescs == nullptr || channels == nullptr) {
        HCCL_VM_ERROR("{}: comm, channelDescs or channels is nullptr", __func__);
        return HCCL_E_PTR;
    }

    if (channelNum == 0 || channelNum > MAX_CHANNEL_NUM) {
        HCCL_VM_ERROR("{}: channelNum {:d} out of range (0, {:d}]", __func__, channelNum, MAX_CHANNEL_NUM);
        return HCCL_E_PARA;
    }

    uint64_t commId = reinterpret_cast<uint64_t>(comm);

    auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    uint32_t existedNum = 0;
    // 与 HcclChannelAcquire 保持同一序号口径：同 remoteRank 的第 k 个描述符查询第 k 条通道。
    std::map<uint64_t, uint32_t> remoteOrdinal;
    for (uint32_t i = 0; i < channelNum; i++) {
        const HcclChannelDesc& desc = channelDescs[i];
        uint64_t remoteRankId = desc.remoteRank;
        const uint32_t ordinal = remoteOrdinal[remoteRankId]++;

        auto existingChannels = sim::runtime::Db::GetByPred<sim::runtime::HcclChannel>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::HcclChannel::commId, commId),
            HcclSim::Storage::Eq(&sim::runtime::HcclChannel::engine, static_cast<uint8_t>(engine)),
            HcclSim::Storage::Eq(&sim::runtime::HcclChannel::remoteRankId, remoteRankId)));

        if (existingChannels.ok() && existingChannels.value.has_value() && existingChannels.value->size() > ordinal) {
            channels[i] = static_cast<ChannelHandle>((*existingChannels.value)[ordinal].id);
            ++existedNum;
        } else {
            channels[i] = 0;
        }
    }

    HCCL_VM_INFO(
        "{} success, commId={:d}, engine={:d}, channelNum={:d}, existedNum={:d}", __func__, commId,
        static_cast<int>(engine), channelNum, existedNum);
    return HCCL_SUCCESS;
}

/**
 * @brief 轮询等待远端通信域的 HcclBuffer 记录就绪。
 *
 * 跨 rank 高并发执行时，本端可能先于对端执行到需要查询远端 HcclBuffer 的位置
 * （对端进程尚未完成通信域初始化与 HcclBuffer 分配）。本函数以 10ms 为间隔
 * 周期性查询存储层中属于指定 remoteCommId 的 sim::runtime::HcclBuffer 记录，
 * 直到查询结果非空或总等待时长超过 10s 为止。
 *
 * @param[in]  remoteCommId 远端通信域的 id（sim::runtime::Communicator::id）。
 * @param[out] outRecords   成功时存放首次查询到的 HcclBuffer 记录；超时返回空。
 *
 * @return true  至少查到一条记录，outRecords 非空。
 * @return false 等待超时仍未查到，已在内部输出 HCCL_VM_ERROR。
 */
static bool WaitForRemoteHcclBufferRecords(uint64_t remoteCommId, std::vector<sim::runtime::HcclBuffer>& outRecords)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(REMOTE_MEM_WAIT_MAX_MS);
    while (true) {
        outRecords.clear();
        auto records = sim::runtime::Db::GetByPred<sim::runtime::HcclBuffer>(
            HcclSim::Storage::Eq(&sim::runtime::HcclBuffer::commId, remoteCommId));
        if (records.ok()) {
            outRecords = *records;
        }
        if (!outRecords.empty()) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            HCCL_VM_ERROR(
                "{}: no HcclBuffer records for remoteCommId={:d} after {:d}ms", __func__, remoteCommId,
                REMOTE_MEM_WAIT_MAX_MS);
            outRecords.clear();
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(REMOTE_MEM_WAIT_POLL_INTERVAL_MS));
    }
}

static bool WaitForRemoteCommunicator(
    uint64_t remoteRankId, const sim::runtime::Communicator& localComm,
    std::vector<sim::runtime::Communicator>& outComms)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(REMOTE_MEM_WAIT_MAX_MS);
    while (true) {
        outComms.clear();
        auto comms = sim::runtime::Db::GetByPred<sim::runtime::Communicator>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Communicator::rank_id, remoteRankId),
            HcclSim::Storage::Eq(&sim::runtime::Communicator::rank_size, localComm.rank_size),
            HcclSim::Storage::Eq(&sim::runtime::Communicator::comm_hash, localComm.comm_hash),
            HcclSim::Storage::Eq(&sim::runtime::Communicator::comm_id, std::string(localComm.comm_id))));
        if (comms.ok()) {
            outComms = *comms;
        }
        if (!outComms.empty()) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            HCCL_VM_ERROR(
                "{}: no remote communicator found after {:d}ms, "
                "remoteRankId={:d}, rankSize={:d}",
                __func__, REMOTE_MEM_WAIT_MAX_MS, remoteRankId, localComm.rank_size);
            outComms.clear();
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(REMOTE_MEM_WAIT_POLL_INTERVAL_MS));
    }
}

/**
 * @brief 获取指定channel对端rank的HCCL通信缓存。
 *
 * 根据channel句柄查询对应的通信通道记录，取出其对端rankId，再在本端通信域所属
 * 的同组（相同rankSize）中查找对端rank的通信域，返回对端通信域已分配的HCCL Buffer
 * 地址和大小。该内存由库内管理，调用者严禁释放。
 *
 * @param comm 输入：通信域句柄（本端），需与channel所属通信域一致。
 * @param channel 输入：通信通道句柄，由HcclChannelAcquire返回。
 * @param buffer 输出：对端rank的HCCL通信内存地址。
 * @param size 输出：对端rank的HCCL通信内存大小（字节）。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 获取成功
 * @retval HCCL_E_PTR comm、buffer或size指针为空
 * @retval HCCL_E_INTERNAL channel不存在、comm与channel不归属同一通信域、
 *                         对端同组通信域不存在或等待对端HcclBuffer就绪超时(10s)
 */
HcclResult HcclChannelGetHcclBuffer(HcclComm comm, ChannelHandle channel, void** buffer, uint64_t* size)
{
    if (comm == nullptr || buffer == nullptr || size == nullptr) {
        HCCL_VM_ERROR("{}: comm, buffer or size is nullptr", __func__);
        return HCCL_E_PTR;
    }

    auto optChannel = sim::runtime::Db::GetById<sim::runtime::HcclChannel>(channel);
    if (!optChannel.ok()) {
        HCCL_VM_ERROR("{}: channel {:d} not found", __func__, channel);
        return HCCL_E_INTERNAL;
    }

    const uint64_t localCommId = reinterpret_cast<uint64_t>(comm);
    if (optChannel->commId != localCommId) {
        HCCL_VM_ERROR(
            "{}: channel {:d} does not belong to comm {:d}, "
            "channel.commId={:d}",
            __func__, channel, localCommId, optChannel->commId);
        return HCCL_E_INTERNAL;
    }

    const uint64_t remoteRankId = optChannel->remoteRankId;

    auto optLocalComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(localCommId);
    if (!optLocalComm.ok()) {
        HCCL_VM_ERROR("{}: local communicator {:d} not found", __func__, localCommId);
        return HCCL_E_INTERNAL;
    }
    const uint32_t localRankSize = optLocalComm->rank_size;

    std::vector<sim::runtime::Communicator> remoteComms;
    if (!WaitForRemoteCommunicator(remoteRankId, *optLocalComm, remoteComms)) {
        HCCL_VM_ERROR(
            "{}: timed out waiting for remote communicator, "
            "channel={:d}, localCommId={:d}, remoteRankId={:d}, rankSize={:d}",
            __func__, channel, localCommId, remoteRankId, localRankSize);
        return HCCL_E_INTERNAL;
    }
    if (remoteComms.size() != 1) {
        HCCL_VM_ERROR(
            "{}: ambiguous remote communicator, matched={:d}, "
            "remoteRankId={:d}, rankSize={:d}",
            __func__, remoteComms.size(), remoteRankId, localRankSize);
        return HCCL_E_INTERNAL;
    }

    const uint64_t remoteCommId = remoteComms[0].id;
    std::vector<sim::runtime::HcclBuffer> buffers;
    if (!WaitForRemoteHcclBufferRecords(remoteCommId, buffers)) {
        HCCL_VM_ERROR(
            "{}: timed out waiting for HcclBuffer, "
            "channel={:d}, localCommId={:d}, remoteRankId={:d}, remoteCommId={:d}",
            __func__, channel, localCommId, remoteRankId, remoteCommId);
        return HCCL_E_INTERNAL;
    }

    const auto& hcclBuf = buffers[0];
    *buffer = reinterpret_cast<void*>(hcclBuf.addr);
    *size = hcclBuf.size;

    HCCL_VM_INFO(
        "{} success, channel={:d}, localCommId={:d}, "
        "remoteRankId={:d}, remoteCommId={:d}, buffer={:p}, size={:d}",
        __func__, channel, localCommId, remoteRankId, remoteCommId, *buffer, *size);
    return HCCL_SUCCESS;
}

/**
 * @brief 轮询等待远端通信域的 HcclMem 注册记录就绪。
 *
 * 跨 rank 执行时，本端可能先于对端执行到需要查询远端 HcclMem 的位置。本函数
 * 以 10ms 为间隔周期性查询存储层中属于指定 remoteCommId 的
 * sim::runtime::HcclMem 记录，直到查询结果非空或总等待时长超过 10s 为止。
 *
 * @param[in]  remoteCommId 远端通信域的 id（sim::runtime::Communicator::id）。
 * @param[out] outRecords   成功时存放首次查询到的 HcclMem 记录；超时返回空。
 *
 * @return true  至少查到一条记录，outRecords 非空。
 * @return false 等待超时仍未查到，已在内部输出 HCCL_VM_ERROR。
 */
static bool WaitForRemoteMemRecords(uint64_t remoteCommId, std::vector<sim::runtime::HcclMem>& outRecords)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(REMOTE_MEM_WAIT_MAX_MS);
    while (true) {
        outRecords.clear();
        auto records = sim::runtime::Db::GetByPred<sim::runtime::HcclMem>(
            HcclSim::Storage::Eq(&sim::runtime::HcclMem::commId, remoteCommId));
        if (records.ok()) {
            outRecords = *records;
        }
        if (!outRecords.empty()) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            HCCL_VM_ERROR(
                "{}: no HcclMem records for remoteCommId={:d} after {:d}ms", __func__, remoteCommId,
                REMOTE_MEM_WAIT_MAX_MS);
            outRecords.clear();
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(REMOTE_MEM_WAIT_POLL_INTERVAL_MS));
    }
}

/**
 * @brief 获取通信通道中交换的远端内存信息。
 *
 * 真实实现中，通道创建时交换的远端内存信息存储在 Channel 对象内部，
 * 本桩通过 sim::runtime::HcclMem 表（按远端 commId 查询）返回远端 rank 注册的内存。
 * 返回的指针在下次调用本接口前有效（与真实实现的缓存语义一致）。
 *
 * @param comm 输入：通信域句柄。
 * @param channel 输入：通信通道句柄。
 * @param memNum 输出：远端内存数量。
 * @param remoteMems 输出：远端内存信息数组（CommMem*）。
 * @param memTags 输出：内存标签字符串数组（char**）。
 *
 * @return HcclResult 成功返回 HCCL_SUCCESS，其他失败。
 */
HcclResult
HcclChannelGetRemoteMems(HcclComm comm, ChannelHandle channel, uint32_t* memNum, CommMem** remoteMems, char*** memTags)
{
    if (comm == nullptr || memNum == nullptr || remoteMems == nullptr || memTags == nullptr) {
        HCCL_VM_ERROR("{}: comm, memNum, remoteMems or memTags is nullptr", __func__);
        return HCCL_E_PTR;
    }
    *memNum = 0;
    *remoteMems = nullptr;
    *memTags = nullptr;

    auto optChannel = sim::runtime::Db::GetById<sim::runtime::HcclChannel>(channel);
    if (!optChannel.ok()) {
        HCCL_VM_ERROR("{}: channel {:d} not found", __func__, channel);
        return HCCL_E_INTERNAL;
    }

    const uint64_t localCommId = reinterpret_cast<uint64_t>(comm);
    if (optChannel->commId != localCommId) {
        HCCL_VM_ERROR(
            "{}: channel {:d} does not belong to comm {:d}, "
            "channel.commId={:d}",
            __func__, channel, localCommId, optChannel->commId);
        return HCCL_E_INTERNAL;
    }

    // 通过 channel.remoteRankId 查找远端 rank 的通信域
    const uint64_t remoteRankId = optChannel->remoteRankId;
    auto optLocalComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(localCommId);
    if (!optLocalComm.ok()) {
        HCCL_VM_ERROR("{}: local communicator {:d} not found", __func__, localCommId);
        return HCCL_E_INTERNAL;
    }
    const uint32_t localRankSize = optLocalComm->rank_size;
    std::vector<sim::runtime::Communicator> remoteComms;
    if (!WaitForRemoteCommunicator(remoteRankId, *optLocalComm, remoteComms)) {
        HCCL_VM_ERROR(
            "{}: timed out waiting for remote communicator, "
            "channel={:d}, localCommId={:d}, remoteRankId={:d}, rankSize={:d}",
            __func__, channel, localCommId, remoteRankId, localRankSize);
        return HCCL_E_INTERNAL;
    }
    if (remoteComms.size() != 1) {
        HCCL_VM_ERROR(
            "{}: ambiguous remote communicator, matched={:d}, "
            "remoteRankId={:d}, rankSize={:d}",
            __func__, remoteComms.size(), remoteRankId, localRankSize);
        return HCCL_E_INTERNAL;
    }

    const uint64_t remoteCommId = remoteComms[0].id;
    std::vector<sim::runtime::HcclMem> memRecords;
    if (!WaitForRemoteMemRecords(remoteCommId, memRecords)) {
        HCCL_VM_ERROR(
            "{}: timed out waiting for remoteCommId={:d}, "
            "channel={:d}, localCommId={:d}, remoteRankId={:d}",
            __func__, remoteCommId, channel, localCommId, remoteRankId);
        return HCCL_E_INTERNAL;
    }

    static thread_local std::vector<CommMem> memBuf;
    static thread_local std::vector<std::string> tagBuf;
    static thread_local std::vector<char*> tagPtrBuf;

    memBuf.clear();
    tagBuf.clear();
    tagPtrBuf.clear();
    memBuf.reserve(memRecords.size());
    tagBuf.reserve(memRecords.size());
    tagPtrBuf.reserve(memRecords.size());

    for (const auto& m : memRecords) {
        CommMem mem{};
        mem.type = static_cast<CommMemType>(m.memType);
        mem.addr = reinterpret_cast<void*>(m.addr);
        mem.size = m.size;
        memBuf.push_back(mem);
        tagBuf.emplace_back(m.mem_tag_str);
        tagPtrBuf.push_back(const_cast<char*>(tagBuf.back().c_str()));
    }

    *memNum = static_cast<uint32_t>(memBuf.size());
    *remoteMems = memBuf.data();
    *memTags = tagPtrBuf.data();

    HCCL_VM_INFO(
        "{} success, channel={:d}, localCommId={:d}, "
        "remoteRankId={:d}, remoteCommId={:d}, memNum={:d}, "
        "lastMemAddr={:p}, lastMemSize={:d}, lastMemTag='{}'",
        __func__, channel, localCommId, remoteRankId, remoteCommId, *memNum,
        static_cast<const void*>((*remoteMems)[*memNum - 1].addr), (*remoteMems)[*memNum - 1].size,
        (*memTags)[*memNum - 1]);
    return HCCL_SUCCESS;
}

/**
 * @brief 创建算子通信引擎上下文。
 *
 * 根据指定的ctxTag和engine类型，为通信域分配一块指定大小的Device内存，
 * 并在全局EngineCtxRegistry中创建对应记录。ctxTag为nullptr时视为空字符串。
 * ctxTag最大长度为HCCL_RES_TAG_MAX_LEN(255)。
 *
 * @param comm 输入：通信域句柄。
 * @param ctxTag 输入：引擎标签字符串，最大长度255，为nullptr时视为空字符串。
 * @param engine 输入：通信引擎类型。
 * @param size 输入：ctx内存大小（字节），不能为0。
 * @param ctx 输出：分配的通信引擎上下文地址（Device内存）。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 创建成功
 * @retval HCCL_E_PTR comm或ctx为空
 * @retval HCCL_E_PARA ctxTag超长或size为0
 * @retval HCCL_E_INTERNAL 通信域不存在或内存分配失败
 */
HcclResult HcclEngineCtxCreate(HcclComm comm, const char* ctxTag, CommEngine engine, uint64_t size, void** ctx)
{
    if (comm == nullptr) {
        HCCL_VM_ERROR("{}: comm is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (ctxTag != nullptr && std::strlen(ctxTag) > HCCL_RES_TAG_MAX_LEN) {
        HCCL_VM_ERROR("{}: ctxTag length {:d} exceeds max {:d}", __func__, std::strlen(ctxTag), HCCL_RES_TAG_MAX_LEN);
        return HCCL_E_PARA;
    }
    if (size == 0) {
        HCCL_VM_ERROR("{}: size is 0", __func__);
        return HCCL_E_PARA;
    }

    uint64_t commId = reinterpret_cast<uint64_t>(comm);
    auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
        int32_t rankId = static_cast<int32_t>(sim::GetCurrRankId());
        uint32_t devKey = static_cast<uint32_t>(sim::runtime::GetCurrDeviceKey());
        TryLaunchAicpuDevProcForRank(rankId, devKey);
    }

    HcclResult ret = EngineCtxRegistry::Instance().Create(commId, ctxTag, engine, size, ctx);
    if (ret != HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: EngineCtxRegistry::Create failed, ret={:d}, size={:d}", __func__, static_cast<int>(ret), size);
        return ret;
    }

    HCCL_VM_INFO(
        "{} success, commId={:d}, ctxTag={}, engine={:d}, "
        "size={:d}, ctx={:p}",
        __func__, commId, ctxTag ? ctxTag : "", static_cast<int>(engine), size, *ctx);
    return HCCL_SUCCESS;
}

/**
 * @brief 获取算子通信引擎上下文。
 *
 * 根据ctxTag和engine类型查询已创建的EngineCtx记录，返回其地址和大小。
 * 若未找到则返回HCCL_E_NOT_FOUND。
 *
 * @param comm 输入：通信域句柄。
 * @param ctxTag 输入：引擎标签字符串，为nullptr时视为空字符串。
 * @param engine 输入：通信引擎类型。
 * @param ctx 输出：已存在的通信引擎上下文地址。
 * @param size 输出：已存在的ctx内存大小（字节）。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 获取成功
 * @retval HCCL_E_PTR comm、ctx或size为空
 * @retval HCCL_E_PARA ctxTag超长
 * @retval HCCL_E_NOT_FOUND 未找到对应的EngineCtx
 */
HcclResult HcclEngineCtxGet(HcclComm comm, const char* ctxTag, CommEngine engine, void** ctx, uint64_t* size)
{
    if (comm == nullptr || ctx == nullptr || size == nullptr) {
        HCCL_VM_ERROR("{}: comm, ctx or size is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (ctxTag != nullptr && std::strlen(ctxTag) > HCCL_RES_TAG_MAX_LEN) {
        HCCL_VM_ERROR("{}: ctxTag length {:d} exceeds max {:d}", __func__, std::strlen(ctxTag), HCCL_RES_TAG_MAX_LEN);
        return HCCL_E_PARA;
    }

    uint64_t commId = reinterpret_cast<uint64_t>(comm);
    uint64_t tagHash = HashCtxTag(ctxTag);
    uint64_t engineVal = static_cast<uint64_t>(engine);

    if (!EngineCtxRegistry::Instance().Get(commId, tagHash, engineVal, ctx, size)) {
        return HCCL_E_NOT_FOUND;
    }

    HCCL_VM_INFO(
        "{} success, commId={:d}, ctxTag={}, engine={:d}, "
        "ctx={:p}, size={:d}",
        __func__, commId, ctxTag ? ctxTag : "", static_cast<int>(engine), *ctx, *size);
    return HCCL_SUCCESS;
}

/**
 * @brief 拷贝算子通信引擎上下文。
 *
 * 根据ctxTag查找目标ctx记录，将srcCtx中size字节的数据拷贝到
 * 目标ctx的(addr + dstCtxOffset)位置。
 *
 * @param comm 输入：通信域句柄。
 * @param engine 输入：通信引擎类型。
 * @param ctxTag 输入：目标引擎标签字符串，为nullptr时视为空字符串。
 * @param srcCtx 输入：源数据指针（Device内存）。
 * @param size 输入：拷贝的字节数，不能为0。
 * @param dstCtxOffset 输入：目标ctx内部的字节偏移。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 拷贝成功
 * @retval HCCL_E_PTR comm或srcCtx为空
 * @retval HCCL_E_PARA ctxTag超长或size为0或越界
 * @retval HCCL_E_NOT_FOUND 目标ctx不存在
 * @retval HCCL_E_INTERNAL 拷贝失败
 */
HcclResult HcclEngineCtxCopy(
    HcclComm comm, CommEngine engine, const char* ctxTag, const void* srcCtx, uint64_t size, uint64_t dstCtxOffset)
{
    if (comm == nullptr || srcCtx == nullptr) {
        HCCL_VM_ERROR("{}: comm or srcCtx is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (ctxTag != nullptr && std::strlen(ctxTag) > HCCL_RES_TAG_MAX_LEN) {
        HCCL_VM_ERROR("{}: ctxTag length {:d} exceeds max {:d}", __func__, std::strlen(ctxTag), HCCL_RES_TAG_MAX_LEN);
        return HCCL_E_PARA;
    }
    if (size == 0) {
        HCCL_VM_ERROR("{}: size is 0", __func__);
        return HCCL_E_PARA;
    }

    uint64_t commId = reinterpret_cast<uint64_t>(comm);
    uint64_t tagHash = HashCtxTag(ctxTag);
    uint64_t engineVal = static_cast<uint64_t>(engine);

    HcclResult ret = EngineCtxRegistry::Instance().Copy(commId, tagHash, engineVal, srcCtx, size, dstCtxOffset);
    if (ret == HCCL_E_NOT_FOUND) {
        HCCL_VM_ERROR("{}: dst ctx not found, commId={:d}, ctxTag={}", __func__, commId, ctxTag ? ctxTag : "");
        return HCCL_E_NOT_FOUND;
    }
    if (ret == HCCL_E_PARA) {
        HCCL_VM_ERROR("{}: dstCtx overflow, offset={:d} + size={:d}", __func__, dstCtxOffset, size);
        return HCCL_E_PARA;
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_VM_ERROR("{}: aclrtMemcpy failed, ret={:d}", __func__, static_cast<int>(ret));
        return ret;
    }

    HCCL_VM_INFO(
        "{} success, commId={:d}, engine={:d}, ctxTag={}, "
        "srcCtx={:p}, size={:d}, dstCtxOffset={:d}",
        __func__, commId, static_cast<int>(engine), ctxTag ? ctxTag : "", srcCtx, size, dstCtxOffset);
    return HCCL_SUCCESS;
}

/**
 * @brief 销毁通信引擎资源上下文。
 *
 * 根据ctxTag和engine类型查询对应的EngineCtx记录，释放其Device内存并从全局map删除记录。
 * 若已不存在则视为已销毁，返回成功（幂等）。
 *
 * @param comm 输入：通信域句柄。
 * @param ctxTag 输入：引擎标签字符串，为nullptr时视为空字符串。
 * @param engine 输入：通信引擎类型。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 销毁成功（含已不存在的情况）
 * @retval HCCL_E_PTR comm为空
 * @retval HCCL_E_PARA ctxTag超长
 */
HcclResult HcclEngineCtxDestroy(HcclComm comm, const char* ctxTag, CommEngine engine)
{
    if (comm == nullptr) {
        HCCL_VM_ERROR("{}: comm is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (ctxTag != nullptr && std::strlen(ctxTag) > HCCL_RES_TAG_MAX_LEN) {
        HCCL_VM_ERROR("{}: ctxTag length {:d} exceeds max {:d}", __func__, std::strlen(ctxTag), HCCL_RES_TAG_MAX_LEN);
        return HCCL_E_PARA;
    }

    uint64_t commId = reinterpret_cast<uint64_t>(comm);
    uint64_t tagHash = HashCtxTag(ctxTag);
    uint64_t engineVal = static_cast<uint64_t>(engine);

    uint32_t removed = EngineCtxRegistry::Instance().Destroy(commId, tagHash, engineVal);

    HCCL_VM_INFO(
        "{} done, commId={:d}, ctxTag={}, engine={:d}, removed={:d}", __func__, commId, ctxTag ? ctxTag : "",
        static_cast<int>(engine), removed);
    return HCCL_SUCCESS;
}

/**
 * @brief 向通信域注册内存。
 *
 * 将用户指定的内存段（地址+大小+类型）与一个字符串标签 memTag 绑定，
 * 记录到通信域对应的 HcclMem 表中。通信域内以 memTag 作为 key 唯一标识该内存。
 * 注册成功后返回不透明的 HcclMemHandle（DB PK），可被后续通道创建时使用。
 * memTag 最大长度为 HCCL_RES_TAG_MAX_LEN(255)。
 *
 * @param comm 输入：通信域句柄。
 * @param memTag 输入：内存字符串标签，以"\0"结尾，最大长度255。
 * @param mem 输入：待注册的内存信息（type/addr/size）。
 * @param memHandle 输出：返回的内存句柄。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 注册成功
 * @retval HCCL_E_PTR comm或mem为空
 * @retval HCCL_E_INTERNAL 通信域不存在/标签重复/DB写入失败
 */
HcclResult HcclCommMemReg(HcclComm comm, const char* memTag, const CommMem* mem, HcclMemHandle* memHandle)
{
    if (comm == nullptr) {
        HCCL_VM_ERROR("{}: comm is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (mem == nullptr) {
        HCCL_VM_ERROR("{}: mem is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (memHandle == nullptr) {
        HCCL_VM_ERROR("{}: memHandle is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (memTag == nullptr) {
        HCCL_VM_ERROR("{}: memTag is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (std::strlen(memTag) > HCCL_RES_TAG_MAX_LEN) {
        HCCL_VM_ERROR("{}: memTag length {:d} exceeds max {:d}", __func__, std::strlen(memTag), HCCL_RES_TAG_MAX_LEN);
        return HCCL_E_PARA;
    }

    uint64_t commId = reinterpret_cast<uint64_t>(comm);
    auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    uint64_t tagHash = HashCtxTag(memTag);

    // 检查 memTag 是否已在本通信域内注册（memTag 作为唯一键）
    // 同一通信域内不允许重复注册相同的 memTag
    auto existing = sim::runtime::Db::GetByPred<sim::runtime::HcclMem>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::HcclMem::commId, commId),
        HcclSim::Storage::Eq(&sim::runtime::HcclMem::memTag, tagHash)));
    if (existing.ok() && !existing->empty()) {
        HCCL_VM_ERROR("{}: memTag '{}' already registered in comm {:d}", __func__, memTag, commId);
        return HCCL_E_INTERNAL;
    }

    // 创建新的内存注册记录，存储用户提供的内存信息
    // memHandle 将作为后续通道创建时的引用标识
    sim::runtime::HcclMem hcclMem{};
    hcclMem.commId = commId;
    hcclMem.memTag = tagHash;
    hcclMem.memType = static_cast<uint64_t>(mem->type);
    // 存储内存地址和大小，供后续通道创建和检查使用
    hcclMem.addr = reinterpret_cast<uint64_t>(mem->addr);
    hcclMem.size = mem->size;
    // 保存原始标签字符串，供 HcclChannelGetRemoteMems 返回
    std::strncpy(hcclMem.mem_tag_str, memTag, sizeof(hcclMem.mem_tag_str) - 1);
    hcclMem.mem_tag_str[sizeof(hcclMem.mem_tag_str) - 1] = '\0';
    auto memIdResult = sim::runtime::Db::Add<sim::runtime::HcclMem>(hcclMem);
    if (!memIdResult.ok()) {
        HCCL_VM_ERROR("{}: failed to add HcclMem record", __func__);
        return HCCL_E_INTERNAL;
    }

    // 返回 DB 主键作为内存句柄（HcclMemHandle = void*）
    // 后续可通过此句柄查询或注销内存注册
    *memHandle = reinterpret_cast<HcclMemHandle>(hcclMem.id);
    HCCL_VM_INFO(
        "{} success, commId={:d}, memTag={}, memType={:d}, "
        "memAddr={:p}, memSize={:d}, handleId={:d}",
        __func__, commId, memTag, static_cast<int>(mem->type), mem->addr, mem->size, hcclMem.id);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif
