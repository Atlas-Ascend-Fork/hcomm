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
 * @file barrier_real_wait.cc
 * @brief 方案二"屏障真等"共享实现（level1 join 桩 / level2 halSqCqQuery SQ_HEAD 桩共用）。
 *
 * 背景与两套形态见 docs/int64_prod_reduce_aicpu_hijack_design.md §9.2：
 * - level2（2a′）：halSqCqQuery 的 SQ_HEAD 分支在查询内阻塞等待该 sqId 流任务全部回放
 *   完成，再按原契约应答 head=GetSqTail（计数契约不变，等待藏进查询）——hcomm 库内
 *   HcommThreadJoin/MakeSureAvailableSpace/PollCompletion 等全部 head 消费者统一获得反压；
 * - level1：北向劫持使 SQE/doorbell 不再生成、hal 路径整体短路，故真等做在 L1 库既有的
 *   HcommThreadJoin no-op 桩内（全表等待，与已验证初版语义一致）。
 *
 * 防挂死闸门（2026-10-21 挂起事故教训）：runner 安装门禁（Plugin 表）/ 等待上限 / 零进度
 * 停滞 fail-open；任何异常一律放行回退基线时序，绝不无限挂起。
 *
 * 门控（2026-11 起为算子属性自动判定，不再使用环境变量/开关文件）：按"本设备最新
 * 落库算子"的 opDetail（opType/dataType/reduceType）判定是否命中内联 CPU 归约路径
 * （reduce 类 + 64bit|PROD，与 hccl LocalReduce→AicpuReduce 分支及 checker 语义盲区
 * 判定一致），命中才真等；其余算子保持基线时序，无需任何手工开关。
 */

// 日志模块 tag：本 TU 仅编入 device 可执行文件（CMake 全局 -DHCCL_VM_MODULE="DEVICE"），
// 沿用 DEVICE 染色；日志行内以 [BarrierRealWait] 前缀作为检索标识。
#include "barrier_real_wait.h"

#include <unistd.h>

#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "hccl/hccl_types.h"
#include "hccl_device_pub.h"
#include "operation_data/operation_data_ops.h"
#include "operation_data/operation_data_types.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/sim_models.h"
#include "sim_common_defs.h"
#include "sim_log.h"

namespace sim {

namespace {

    /* 真等墙钟上限（秒）：即使调用方语义允许更久，也不把 host pipe RPC 挂住数分钟以上。 */
    constexpr uint32_t kMaxRealWaitSec = 300;
    /* 零进度停滞判定（秒）：pending 持续不变 ⇒ runner 未在消费本轮 DB（崩溃/残存旧 runner
     * 绑定已被 envInit 删除的 DB inode）→ fail-open。健康参照：成功轮 11 任务亚秒级清零。 */
    constexpr uint32_t kStallFailOpenSec = 60;
    /* 完成度轮询间隔：runner 主循环 1ms 轮询 DB，2ms 粒度足以贴住回放进度。 */
    constexpr uint32_t kPollIntervalMs = 2;
    /* 等待进度日志节流间隔。 */
    constexpr uint32_t kProgressLogIntervalMs = 500;

    // 等待进度日志节流表：hal SQ_HEAD 会被多线程查询（kernel 编排线程 + aicpu_daemon_service
    // 后台线程），按 (tag, streamId) 维度节流需互斥保护。
    std::mutex& ThrottleMutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    std::map<std::string, std::chrono::steady_clock::time_point>& ThrottleMap()
    {
        static std::map<std::string, std::chrono::steady_clock::time_point> lastLog;
        return lastLog;
    }

    // runner 是否已由用户安装（`hccl-vm plugin install @runner`）。判据与 host 侧
    // aclrt_stream_stub.cc 完全一致——runtime DB(hccl_sim.db) 的 Plugin 表存在 tag=="runner" 行。
    // 未安装 ⇒ 任务无人回放、host 侧也不等待，真等没有意义（2026-10-21 挂起事故根因），
    // 直接回退基线。结果静态缓存：安装状态在一次运行内不变（start 会清库重建）。
    bool RunnerInstalled()
    {
        static const bool installed = [] {
            auto pluginRows = sim::runtime::Db::GetByPred<sim::runtime::Plugin>(
                HcclSim::Storage::Eq(&sim::runtime::Plugin::tag, "runner"));
            if (!pluginRows.ok()) {
                // 查询失败保守视为已安装：让停滞/上限闸门兜底，避免因偶发查询失败静默退化基线。
                HCCL_VM_WARN("[BarrierRealWait] query Plugin(runner) failed, assume installed");
                return true;
            }
            return pluginRows.value.has_value() && !pluginRows.value->empty();
        }();
        return installed;
    }

    // 节流进度日志（500ms/键）。
    void
    ThrottledWaitingLog(const char* callerTag, uint64_t streamId, bool wholeTable, uint64_t pending, uint64_t total)
    {
        const std::string key = std::string(callerTag) + (wholeTable ? "/all" : ("/" + std::to_string(streamId)));
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(ThrottleMutex());
        auto& lastLog = ThrottleMap();
        auto it = lastLog.find(key);
        if (it != lastLog.end() && now - it->second < std::chrono::milliseconds(kProgressLogIntervalMs)) {
            return;
        }
        lastLog[key] = now;
        HCCL_VM_INFO("[BarrierRealWait] {} waiting runner replay: {}/{} task(s) pending", callerTag, pending, total);
    }

    // 自动门控判定谓词：当前算子是否会走"device 侧 AICPU 内联 CPU 归约"路径——
    // reduce 类算子且命中 64bit 数据类型（INT64/UINT64/FP64）或 PROD 归约。与 hccl
    // LocalReduce→AicpuReduce 分支命中条件（alg_data_trans_wrapper.cc:905）及 checker
    // 语义盲区判定（checker.cc IsReduceSemanticBlindSpotOp）严格一致：该类归约不产生
    // REDUCE 任务制品、数据效果不进 runner 回放序列，屏障必须真等（方案二适用域）；
    // 其余算子（SDMA inline reduce / CCU / AIV）任务制品齐全，保持基线时序。
    bool IsInlineCpuReduceOp(uint16_t opType, uint16_t dataType, uint16_t reduceType)
    {
        const bool isReduceFamily = (opType == HCCL_CMD_ALLREDUCE) || (opType == HCCL_CMD_REDUCE)
                                    || (opType == HCCL_CMD_REDUCE_SCATTER) || (opType == HCCL_CMD_REDUCE_SCATTER_V);
        if (!isReduceFamily) {
            return false;
        }
        const bool is64BitType = (dataType == HCCL_DATA_TYPE_INT64) || (dataType == HCCL_DATA_TYPE_UINT64)
                                 || (dataType == HCCL_DATA_TYPE_FP64);
        const bool isProdOp = (reduceType == HCCL_REDUCE_PROD);
        return is64BitType || isProdOp;
    }

    // 自动门控判定缓存：按"本设备最新算子行 id"缓存判定结果——同一算子执行期内
    // （device 进程经 pipe RPC 串行执行 kernel，host 在 kernel 返回前不会落库下一行）
    // id 不变，每次 SQ_HEAD/join 调用只付一次两列投影查询；算子切换（id 变化）才
    // 重新解析 BLOB 判定并打一条 decision 日志（继承旧 gate-init 自检日志的诊断职责：
    // 一眼区分"没生效"的三种情形——未调用/gate OFF/gate ON）。
    std::mutex& GateMutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    struct GateDecision {
        uint32_t opDetailId = 0; // 最近一次判定所依据的算子行 id（0 = 尚无算子行）
        bool evaluated = false;  // 已按 opDetailId 完成判定（缓存命中键）
        bool noOpLogged = false; // "尚无算子行"提示只打一次（init 阶段每次查询都会走到）
        bool gate = false;       // 判定结果：true = 真等
        uint16_t opType = 0;
        uint16_t dataType = 0;
        uint16_t reduceType = 0;
    };

    GateDecision& GateState()
    {
        static GateDecision state;
        return state;
    }

    // 判定失败一律 OFF（基线时序，fail-open）：真等只是时序修复，判定不了就不改变现状。
    bool RealWaitAutoGateHit()
    {
        std::unique_lock<std::mutex> lock(GateMutex());
        auto& state = GateState();
        const uint32_t deviceKey = GetCurDeviceKey();
        uint32_t latestId = 0;
        std::vector<uint8_t> blob;
        // deviceKey 未设置（PIPE_RSP_SET_DEV_ID 之前）不查库：无身份可判，静默 OFF。
        // rc: 1=命中最新算子行；0=尚无算子行（init kernel 阶段，高频合法态）；-1=查询失败。
        const int rc = (deviceKey == 0) ? 0 : operation::QueryLatestOpDetailBlobByDeviceId(deviceKey, latestId, blob);
        if (rc != 1) {
            // 尚无任何算子落库（RunAicpuCommInit 等 init kernel 阶段）或查询失败：
            // 内联 CPU 归约只可能出现在算子 kernel 内，判 OFF 不影响修复目标。
            // 提示日志整个"无算子"阶段只打一条（noOpLogged），避免逐次查询刷屏。
            if (!state.noOpLogged) {
                state.noOpLogged = true;
                lock.unlock();
                HCCL_VM_INFO(
                    "[BarrierRealWait] gate decision: no op recorded for deviceKey={} yet (rc={}), "
                    "gate=OFF(baseline)",
                    deviceKey, rc);
            }
            return false;
        }
        if (state.evaluated && state.opDetailId == latestId) {
            return state.gate; // 同一算子：判定缓存命中，静默返回
        }

        state = GateDecision{};
        state.opDetailId = latestId;
        state.evaluated = true;
        if (blob.size() < sizeof(OpDetails)) {
            state.gate = false;
            lock.unlock();
            HCCL_VM_WARN(
                "[BarrierRealWait] gate decision: opDetailId={} blob truncated ({} < {}), gate=OFF(baseline)", latestId,
                blob.size(), sizeof(OpDetails));
            return false;
        }
        OpDetails details{};
        std::memcpy(&details, blob.data(), sizeof(OpDetails));
        state.opType = details.opType;
        state.dataType = details.dataType;
        state.reduceType = details.reduceType;
        state.gate = IsInlineCpuReduceOp(details.opType, details.dataType, details.reduceType);
        lock.unlock();
        HCCL_VM_INFO(
            "[BarrierRealWait] gate decision (auto by op attrs): pid={}, ppid={}, deviceKey={}, opDetailId={}, "
            "opType={}, dataType={}, reduceType={}, gate={}",
            static_cast<uint64_t>(getpid()), static_cast<uint64_t>(getppid()), deviceKey, latestId, details.opType,
            details.dataType, details.reduceType, state.gate ? "ON(real-wait)" : "OFF(baseline)");
        return state.gate;
    }

} // namespace

bool BarrierRealWait(uint64_t streamId, bool wholeTable, const char* callerTag)
{
    if (!RealWaitAutoGateHit()) {
        return false; // 自动门控未命中：调用方走基线语义
    }
    if (!RunnerInstalled()) {
        static const bool logged = [] {
            HCCL_VM_INFO("[BarrierRealWait] runner plugin NOT installed (Plugin table has no tag='runner'), real-wait "
                         "skipped — baseline timing (install via `hccl-vm plugin install @runner` to enable)");
            return true;
        }();
        (void)logged;
        return true;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(kMaxRealWaitSec);
    auto lastProgress = std::chrono::steady_clock::now();
    uint64_t lastPending = UINT64_MAX;
    uint64_t pending = 0;
    uint64_t total = 0;
    bool waited = false;
    while (true) {
        const int rc = wholeTable ? operation::QueryPendingOpTaskCountDev(total, pending) :
                                    operation::QueryPendingOpTaskCountByStreamDev(streamId, total, pending);
        if (rc != 0) {
            // fail-open：完成度查询异常不挂 kernel，按基线放行，问题经 WARN 暴露。
            HCCL_VM_WARN(
                "[BarrierRealWait] {} query pending tasks failed (ppid={}), fail-open pass", callerTag,
                static_cast<uint64_t>(getppid()));
            return true;
        }
        if (pending == 0) {
            if (waited) {
                HCCL_VM_INFO("[BarrierRealWait] {} drained: all {} recorded task(s) done, pass", callerTag, total);
            }
            return true;
        }
        waited = true;
        const auto now = std::chrono::steady_clock::now();
        if (pending != lastPending) {
            lastPending = pending;
            lastProgress = now;
        }
        // 停滞闸门：runner 已装却零进度（崩溃/残存旧 runner 绑定已删 DB inode）→ fail-open。
        if (now - lastProgress >= std::chrono::seconds(kStallFailOpenSec)) {
            HCCL_VM_ERROR(
                "[BarrierRealWait] {} zero progress for {}s (pending {}/{} unchanged; runner installed but not "
                "consuming this run's DB? crashed / stale runner bound to wiped DB inode?) — fail-open",
                callerTag, kStallFailOpenSec, pending, total);
            return true;
        }
        // 上限闸门。
        if (now >= deadline) {
            HCCL_VM_ERROR(
                "[BarrierRealWait] {} wait cap {}s reached: {}/{} task(s) still pending — fail-open", callerTag,
                kMaxRealWaitSec, pending, total);
            return true;
        }
        ThrottledWaitingLog(callerTag, streamId, wholeTable, pending, total);
        std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
    }
}

} // namespace sim
