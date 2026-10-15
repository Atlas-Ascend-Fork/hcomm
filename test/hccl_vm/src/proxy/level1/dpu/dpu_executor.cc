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
 * Description: HostDPU 北向劫持（level1）DPU 执行器。
 *              替代真实 libccl_dpu.so 的 RunDpuRpcSrvLaunch/TaskService::TaskRun：
 *              以独立线程轮询各通信域 DPUTAG 共享内存的请求半区 flag，
 *              解析 msgTag 后经 DpuCallbackRegistry 分发回调（HcclLaunchDPUKernel
 *              → DPUKernelRun，HCCL 真实代码），回调产生的数据面调用落入 level1
 *              数据面桩生成任务元数据；结果经响应半区回写，供设备侧
 *              HcommWaitResponse 读取。单线程串行消费，保证 DPU 任务相对顺序。
 * Create: 2026-10-09
 */

#define HCCL_VM_MODULE "DPU_EXEC"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#include "dpu_level1_res.h"

#include "acl/acl_rt.h"
#include "ascend_hal.h"
#include "hccl/hccl_types.h"
#include "hccl_proxy_common.h"
#include "level1_proxy_common.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/sim_models.h"
#include "sim_log.h"

/* g_cur_server_key 由公共运行时定义（与 ctrl_comm_resource_stub.cc 同来源）。 */
extern uint64_t g_cur_server_key;

namespace HcclSim {
namespace DpuSim {

    namespace {

        std::mutex g_execMutex;                 /**< 执行器启动互斥锁 */
        std::atomic<bool> g_execRunning{false}; /**< 执行器线程是否已拉起（进程级幂等） */

        /**
         * @brief 从请求半区读取一条完整请求（flag 已由调用方确认为 SET）。
         * @param req       输入：请求半区基址（host 可解引用）。
         * @param msgTag    输出：任务标识字符串（截断到 255 字节内保证 NUL 结尾）。
         * @param msgId     输出：请求消息 id。
         * @param dataSize  输出：payload 字节数（不含头部）。
         * @param payload   输出：payload 拷贝缓冲（调用方持有，容量至少 DPU_TASK_PAYLOAD_MAX）。
         * @return 成功返回 true；dataSize 超限返回 false。
         * @note 读前以 acquire 屏障保证设备侧写入（dmb sy）对本线程可见。
         */
        bool ReadOneRequest(
            const volatile uint8_t* req, std::string& msgTag, uint32_t& msgId, uint64_t& dataSize, uint8_t* payload)
        {
            std::atomic_thread_fence(std::memory_order_acquire);

            // 1. msgTag：256B 定长空间，要求空间内存在 NUL（与真实 ReadTaskType 一致）；
            //    volatile 内存先整体拷贝到本地再解析，避免限定符丢失与重复读不一致
            char tagBuf[DPU_MSG_TAG_MAX_LEN] = {};
            std::memcpy(tagBuf, const_cast<const uint8_t*>(req + DPU_REQ_MSGTAG_OFF), DPU_MSG_TAG_MAX_LEN);
            const void* nulPos = std::memchr(tagBuf, '\0', DPU_MSG_TAG_MAX_LEN);
            if (nulPos == nullptr) {
                HCCL_VM_ERROR("{}: msgTag has no NUL within {:d} bytes", __func__, DPU_MSG_TAG_MAX_LEN);
                return false;
            }
            msgTag.assign(tagBuf, static_cast<const char*>(nulPos) - tagBuf);

            // 2. msgId / dataSize / payload
            std::memcpy(&msgId, const_cast<const uint8_t*>(req + DPU_REQ_MSGID_OFF), sizeof(msgId));
            std::memcpy(&dataSize, const_cast<const uint8_t*>(req + DPU_REQ_DATASIZE_OFF), sizeof(dataSize));
            if (dataSize > DPU_TASK_PAYLOAD_MAX) {
                HCCL_VM_ERROR("{}: dataSize={:d} exceeds max {:d}", __func__, dataSize, DPU_TASK_PAYLOAD_MAX);
                return false;
            }
            if (dataSize > 0) {
                std::memcpy(payload, const_cast<const uint8_t*>(req + DPU_REQ_DATA_OFF), dataSize);
            }
            return true;
        }

        /**
         * @brief 回写响应半区并置位响应 flag（msgId + retCode）。
         * @param rsp     输入：响应半区基址（host 可解引用）。
         * @param msgId   输入：回传的请求消息 id。
         * @param retCode 输入：回调执行结果（0=成功，其余为错误码，设备侧快速失败）。
         * @note 先写内容再以 release 语义置 flag=1，保证设备侧读到 flag 时内容完整。
         */
        void WriteResponse(volatile uint8_t* rsp, uint32_t msgId, int32_t retCode)
        {
            std::memcpy(const_cast<uint8_t*>(rsp + DPU_RSP_RETCODE_OFF), &retCode, sizeof(retCode));
            std::memcpy(const_cast<uint8_t*>(rsp + DPU_REQ_MSGID_OFF), &msgId, sizeof(msgId));
            std::atomic_thread_fence(std::memory_order_release);
            *(rsp + DPU_REQ_FLAG_OFF) = DPU_FLAG_SET;
        }

        /**
         * @brief 构造回调期的 DpuExecContext 并置位到 TLS。
         * @param region 输入：当前请求的共享内存区域（提供 commId）。
         * @param ctx    输出：填充 rankId/deviceId/dpuStreamId 的执行上下文。
         * @return 成功返回 true；通信域查询失败返回 false。
         */
        bool BuildDpuExecContext(const DpuMemRegion& region, DpuExecContext& ctx)
        {
            ctx = DpuExecContext{};
            ctx.active = true;
            ctx.commId = region.commId;

            // rankId/deviceId：取自本进程通信域成员行（commId 即 Communicator 表主键）
            const auto comm = sim::runtime::Db::GetById<sim::runtime::Communicator>(region.commId);
            if (!comm.ok()) {
                HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, region.commId);
                return false;
            }
            ctx.rankId = comm->rank_id;
            ctx.deviceId = static_cast<uint32_t>(comm->device_id);

            // DPU 任务归属流：设备侧 HcommSendRequest 桩已写入 DpuDeviceInfo（与南向同源）
            const auto dpuInfo = sim::runtime::Db::GetOneByPred<sim::runtime::DpuDeviceInfo>(
                HcclSim::Storage::Eq(&sim::runtime::DpuDeviceInfo::device_id, ctx.deviceId));
            if (dpuInfo.ok()) {
                ctx.dpuStreamId = dpuInfo->stream_id;
            } else {
                HCCL_VM_WARN("{}: DpuDeviceInfo not found, deviceId={:d}, streamId fallback 0", __func__, ctx.deviceId);
                ctx.dpuStreamId = 0;
            }
            return true;
        }

        /**
         * @brief 处理单个通信域上的一条待处理请求（读消息→查表→执行→回执）。
         * @param region  输入：请求的共享内存区域。
         * @param payload 输入：可复用的 payload 拷贝缓冲（容量 DPU_TASK_PAYLOAD_MAX）。
         * @return 无；内部错误经响应 retCode 上报给请求方，不向外传播。
         */
        void HandleOneRequest(const DpuMemRegion& region, uint8_t* payload)
        {
            volatile uint8_t* req = region.npu2dpu;
            std::string msgTag;
            uint32_t msgId = 0;
            uint64_t dataSize = 0;
            int32_t retCode = 0;

            // 【核心步骤 1】读出完整请求并立即清请求 flag（释放写方缓冲，对齐真实 ProcessTaskOk 顺序）
            if (!ReadOneRequest(req, msgTag, msgId, dataSize, payload)) {
                HCCL_VM_ERROR("{}: bad request, commId={:d}", __func__, region.commId);
                retCode = static_cast<int32_t>(HCCL_E_INTERNAL);
                WriteResponse(region.dpu2npu, msgId, retCode);
                *(req + DPU_REQ_FLAG_OFF) = DPU_FLAG_UNSET;
                return;
            }

            // 【核心步骤 2】按 (commId, msgTag) 查回调
            DpuCallback cb = nullptr;
            if (!DpuCallbackRegistry::Instance().Lookup(region.commId, msgTag, cb)) {
                HCCL_VM_ERROR("{}: callback not found, commId={:d}, msgTag={}", __func__, region.commId, msgTag);
                WriteResponse(region.dpu2npu, msgId, static_cast<int32_t>(HCCL_E_NOT_FOUND));
                *(req + DPU_REQ_FLAG_OFF) = DPU_FLAG_UNSET;
                return;
            }

            // 【核心步骤 3】置位 DPU 执行上下文（数据面任务归属依赖），执行回调
            DpuExecContext& tls = GetDpuExecTls();
            if (BuildDpuExecContext(region, tls)) {
                try {
                    retCode = cb(reinterpret_cast<uint64_t>(payload), static_cast<int32_t>(dataSize));
                } catch (...) {
                    HCCL_VM_ERROR("{}: callback exception, commId={:d}, msgTag={}", __func__, region.commId, msgTag);
                    retCode = static_cast<int32_t>(HCCL_E_INTERNAL);
                }
            } else {
                retCode = static_cast<int32_t>(HCCL_E_INTERNAL);
            }
            tls = DpuExecContext{}; // 复位 TLS（含异常路径）

            // 【核心步骤 4】回写响应（msgId + retCode），置响应 flag，清请求 flag
            WriteResponse(region.dpu2npu, msgId, retCode);
            *(req + DPU_REQ_FLAG_OFF) = DPU_FLAG_UNSET;
            HCCL_VM_INFO(
                "{}: commId={:d}, msgTag={}, msgId={:d}, ret={:d}", __func__, region.commId, msgTag, msgId, retCode);
        }

        /**
         * @brief 为执行器线程绑定设备上下文（Runner 行 + 默认 Context + device key）。
         * @param deviceId 输入：设备 id（Device 表主键）。
         * @return 成功返回 true。
         * @note 不可省略：回调内 aclrtMemcpy 等经 GetCurrDeviceKey/当前 Context 解析
         *       虚拟内存，执行器新线程不绑 ctx 则全部失败。
         */
        bool BindExecutorCtxTls(uint32_t deviceId)
        {
            sim::runtime::Runner runner{};
            if (!sim::runtime::GetCurrRunnerTls(g_cur_server_key, runner)) {
                HCCL_VM_ERROR("{}: GetCurrRunnerTls failed", __func__);
                return false;
            }
            sim::runtime::Context ctx{};
            if (sim::runtime::GetContextByDevId(deviceId, ctx) != ACL_SUCCESS) {
                HCCL_VM_ERROR("{}: GetContextByDevId failed, deviceId={:d}", __func__, deviceId);
                return false;
            }
            if (!sim::runtime::SetCurrCtxTls(ctx.id)) {
                HCCL_VM_ERROR("{}: SetCurrCtxTls failed, ctxId={:d}", __func__, ctx.id);
                return false;
            }
            sim::runtime::g_cur_device_key = deviceId;
            return true;
        }

        /**
         * @brief DPU 执行器主循环：轮询各通信域请求半区，串行分发处理。
         * @param deviceId 输入：本进程设备 id（用于绑定设备上下文）。
         * @note 常驻 detach 线程；进程退出时随进程结束。空转时 sleep 降低 CPU 占用。
         */
        void DpuExecutorMain(uint32_t deviceId)
        {
            if (!BindExecutorCtxTls(deviceId)) {
                HCCL_VM_ERROR("{}: bind ctx failed, executor exit", __func__);
                return;
            }
            // payload 复用缓冲（串行消费，每请求覆盖写）
            uint8_t* payload = new (std::nothrow) uint8_t[DPU_TASK_PAYLOAD_MAX];
            if (payload == nullptr) {
                HCCL_VM_ERROR("{}: alloc payload buffer failed", __func__);
                return;
            }
            HCCL_VM_INFO("{}: dpu executor started, deviceId={:d}", __func__, deviceId);

            while (g_execRunning.load(std::memory_order_relaxed)) {
                const auto snapshot = DpuMemRegistry::Instance().Snapshot();
                bool handledAny = false;
                for (const auto& region : snapshot) {
                    if (*(region.npu2dpu + DPU_REQ_FLAG_OFF) != DPU_FLAG_SET) {
                        continue;
                    }
                    HandleOneRequest(region, payload);
                    handledAny = true;
                }
                if (!handledAny) {
                    std::this_thread::sleep_for(std::chrono::microseconds(DPU_EXEC_IDLE_SLEEP_US));
                }
            }

            delete[] payload;
            HCCL_VM_INFO("{}: dpu executor exit", __func__);
        }

    } // namespace

    /**
     * @brief 幂等拉起 DPU 执行器线程（进程级单例，double-checked）。
     * @param deviceId 输入：本进程设备 id（首次拉起时用于绑定设备上下文）。
     * @return 已运行或拉起成功返回 true；线程创建失败返回 false。
     * @note 由 HcclDevMemAcquire（首分配 DPUTAG）与 HcclTaskRegister（兜底）调用。
     */
    bool EnsureDpuExecutorStarted(uint32_t deviceId)
    {
        if (g_execRunning.load(std::memory_order_acquire)) {
            return true;
        }
        std::lock_guard<std::mutex> lock(g_execMutex);
        if (g_execRunning.load(std::memory_order_relaxed)) {
            return true;
        }
        g_execRunning.store(true, std::memory_order_release);
        std::thread(DpuExecutorMain, deviceId).detach();
        HCCL_VM_INFO("{}: dpu executor thread launched, deviceId={:d}", __func__, deviceId);
        return true;
    }

    /**
     * @brief 按通信域清理 DPU 资源（通信域销毁钩子）。
     * @param commId 输入：通信域 id（Communicator 表主键）。
     * @note 清理顺序：先摘除 DpuMemRegistry 条目（执行器下轮不再访问），
     *       再释放共享内存（halHostUnregister 解除 host 注册 + aclrtFree 释放设备内存，
     *       与真实 DestroyKFCWorkSpaceVA 的 HrtHalHostUnregister + HrtFree 顺序一致），
     *       最后移除该 comm 全部回调注册。
     */
    void DpuCleanupByCommId(uint64_t commId)
    {
        DpuMemRegion region{};
        if (DpuMemRegistry::Instance().Remove(commId, region)) {
            uint32_t deviceId = 0;
            const auto comm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
            if (comm.ok()) {
                deviceId = static_cast<uint32_t>(comm->device_id);
            }
            // region.devAddr 为 aclrtMalloc 返回的设备地址，按真实销毁顺序释放
            (void)halHostUnregister(region.devAddr, deviceId);
            (void)aclrtFree(region.devAddr);
            HCCL_VM_INFO("{}: freed dpu shmem, commId={:d}, devAddr={:p}", __func__, commId, region.devAddr);
        }
        const auto removed = DpuCallbackRegistry::Instance().RemoveByCommId(commId);
        HCCL_VM_INFO("{}: commId={:d}, removedCallbacks={:d}", __func__, commId, removed);
    }

} // namespace DpuSim
} // namespace HcclSim
