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
 * Description: CCU CPU 模拟引擎实现（录制-回放两段式）。
 *              - 录制（注册期，kernelArg 存活）：TLS 录制上下文，kernelFunc 执行期间原语桩
 *                调 Prim* 接口把语义按原始参数追加为轨迹；
 *              - 回放（Launch 期）：TLS 执行上下文（taskArgs/变量/地址/控制流帧/任务记录），
 *                逐条回放轨迹——LoadArg 绑定、算逻求值、if 帧抑制、while/do-while 游标循环、
 *                数据/同步原语录制为 CPU 任务序列；
 *              - 任务生成：回放结束后做 rank->device 归一化（与 data_comm_op_stub.cc 的
 *                level1 口径一致）并插入 TaskCollection，同时落盘 dump 文件便于排障。
 * Create: 2026-09-12
 */

#define HCCL_VM_MODULE "CCU_KERNEL_SIM"

#include "ccu_kernel_sim.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unistd.h>

#include "hccl_proxy_common.h"
#include "level1_proxy_common.h"
#include "sim_common_api.h"
#include "sim_log.h"
#include "runtime_state/sim_models.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "operation_data/operation_data_ops.h"
#include "storage/table_access.h"
#include "store_sim_store_pub.h"

namespace HcclSim {
namespace CcuSim {
    namespace {

        /** @brief 本地 CKE 事件位资源 id 编码基址：与 DB notifyId、通道位 id、线程句柄空间天然隔离。 */
        constexpr uint64_t CCU_EVENT_NOTIFY_ID_BASE = 0xE000000000000000ULL;

        /** @brief 通道位资源 id 编码基址：与事件位 id、DB notifyId 空间隔离。 */
        constexpr uint64_t CCU_CHANNEL_NOTIFY_ID_BASE = 0xD000000000000000ULL;

        /** @brief CCU 缓冲(MS)地址窗口基址：与 checker `CCU_MS_WINDOW_BASE` 一致（南向实测 MS 偏移起点）。 */
        constexpr uint64_t CCU_MS_BASE = 0x08000000ULL;

        /** @brief MS 分配粒度：4KB（与 checker `GenSliceFromMs` 的 msId*4096 口径一致）。 */
        constexpr uint64_t CCU_MS_GRANULARITY = 4096ULL;

        /** @brief MS 窗口容量：与 checker `CCU_MS_WINDOW_CAP` 一致（64MB）。 */
        constexpr uint64_t CCU_MS_WINDOW_CAP = 64ULL * 1024ULL * 1024ULL;

        /** @brief 回放步数上限（死循环/超大展开防御）。 */
        constexpr uint64_t CCU_REPLAY_MAX_STEPS = 64ULL * 1024ULL * 1024ULL;

        /** @brief 单个 while 的迭代次数上限：条件变量永不更新时的死循环快速失败（远小于全局步数上限）。 */
        constexpr uint64_t CCU_REPLAY_MAX_LOOP_ITER = 1000ULL * 1000ULL;

        /** @brief 交换资源类型：0=XN(Var)，1=CKE（CKE 当前经 NOTIFY 任务表达，不入库）。 */
        constexpr uint32_t CCU_SYNC_RES_XN = 0;

        /** @brief 交换值有界等待参数（复用 DB-as-IPC 轮询模式）。 */
        constexpr uint32_t CCU_SYNC_WAIT_TIMEOUT_MS = 10000U;
        constexpr uint32_t CCU_SYNC_WAIT_POLL_MS = 10U;

        /**
         * @brief 交换资源逻辑键（对齐 hcomm CcuTransport 每 channel 预留 XN/CKE 0..3）。
         *
         * opIter = 算子迭代（= OpDetailTab.opIter，两侧进程按同一算子序列自增，天然跨 rank 对齐）；
         * round = 轮次（写方 CCU launch 序号，**进程内全局计数、不按 op 重置**）；
         * srcRank = 写方 rank（= 读方视角的 peer）；targetRank = 被写入方 rank（= 读方）；
         * connOrdinal = 同 peer 多 channel 连接序号；
         * 不含 commId（两侧进程的 Communicator id 不同，无法对齐——见设计文档 §3.4.4）。
         *
         * **peer 隔离**（2026-09-21 修复）：必须带 srcRank。仅靠 targetRank+connOrdinal 无法区分
         * 同一读方的多个写方（多 rank 时不同 peer 会算出相同 connOrdinal），导致写方互相覆盖、
         * 读方取到别的 peer 的地址（现象：MEM_CPY src 落在非本设备地址段）。
         *
         * **op 隔离**（2026-09-2x 修复）：必须带 opIter。round 是进程内全局 launch 计数，同一
         * (round, src, dst, id) 坐标会被后续 op 复用；表又跨 op 累积，裁剪时会把前一 op 尚未被
         * 读走的行删掉，落后 rank 只能取到别的 op 的地址（现象：连跑用例时 checker 报 MEM_CPY
         * 目标地址落不到本 op 内存布局，单跑不触发）。
         *
         * **轮次隔离**（设计 §3.4.5）：同一 op 内一轮一行、不覆盖。读方按自己所在 op+轮次精确取，
         * 避免两进程异步漂移（NOTIFY_WAIT 在仿真中不真阻塞回放）导致读到"未来轮次"或"旧轮次"的值。
         */
        struct SimCcuSyncKey {
            uint32_t opIter{0};
            uint32_t round{0};
            uint32_t srcRank{0};
            uint32_t targetRank{0};
            uint32_t connOrdinal{0};
            uint32_t dieId{0};
            uint32_t resType{0};
            uint32_t resId{0};
        };

        /** @brief 交换坐标条件（不含 opIter/round）：跨 op 裁剪与同 op 查询的公共坐标部分（条件数据库端下推）。 */
        HcclSim::Storage::Condition<sim::runtime::CcuSyncResTab> SyncCoordsCond(const SimCcuSyncKey& key)
        {
            return HcclSim::Storage::And(
                HcclSim::Storage::Eq(&sim::runtime::CcuSyncResTab::srcRank, key.srcRank),
                HcclSim::Storage::Eq(&sim::runtime::CcuSyncResTab::targetRank, key.targetRank),
                HcclSim::Storage::Eq(&sim::runtime::CcuSyncResTab::connOrdinal, key.connOrdinal),
                HcclSim::Storage::Eq(&sim::runtime::CcuSyncResTab::dieId, key.dieId),
                HcclSim::Storage::Eq(&sim::runtime::CcuSyncResTab::resType, key.resType),
                HcclSim::Storage::Eq(&sim::runtime::CcuSyncResTab::resId, key.resId));
        }

        /** @brief 同 op 坐标条件（含 opIter，不含 round）：取/兜底"本 op 最近一轮"的已有值用。 */
        HcclSim::Storage::Condition<sim::runtime::CcuSyncResTab> SameOpSyncCoordsCond(const SimCcuSyncKey& key)
        {
            return HcclSim::Storage::And(
                HcclSim::Storage::Eq(&sim::runtime::CcuSyncResTab::opIter, key.opIter), SyncCoordsCond(key));
        }

        /** @brief 全键条件（opIter + 坐标 + 轮次）：一次交换的唯一标识。 */
        HcclSim::Storage::Condition<sim::runtime::CcuSyncResTab> SyncKeyCond(const SimCcuSyncKey& key)
        {
            return HcclSim::Storage::And(
                HcclSim::Storage::Eq(&sim::runtime::CcuSyncResTab::opIter, key.opIter),
                HcclSim::Storage::Eq(&sim::runtime::CcuSyncResTab::round, key.round), SyncCoordsCond(key));
        }

        /** @brief 轮次保留窗口：同一 op 内每个坐标只保留最近 N 轮的行，更早的删除（防表无限增长）。 */
        constexpr uint32_t CCU_SYNC_ROUND_KEEP = 2U;

        /** @brief op 保留窗口：同一坐标只保留最近 N 个 op 的行，更早的删除（防连跑时表无限增长）。 */
        constexpr uint32_t CCU_SYNC_OP_KEEP = 8U;

        /**
         * @brief 裁剪交换表历史：同一坐标只保留最近 CCU_SYNC_OP_KEEP 个 op，op 内只保留最近
         *        CCU_SYNC_ROUND_KEEP 轮的行，防连跑用例下表无限增长。
         * @note 裁剪按"同坐标跨 op"判定（SameSyncResCoords，不含 opIter），否则旧 op 的行永不被
         *       清理；而读取/兜底按含 opIter 的精确坐标判定（SameSyncCoords），因此不会误删本 op
         *       尚需的行——这正是 op 隔离的关键：落后 rank 需要的行不会被后续 op 裁掉。
         */
        void TrimSyncResHistory(const SimCcuSyncKey& key)
        {
            const uint32_t minKeepRound
                = (key.round + 1U > CCU_SYNC_ROUND_KEEP) ? key.round - (CCU_SYNC_ROUND_KEEP - 1U) : 0U;
            const uint32_t minKeepOp = (key.opIter >= CCU_SYNC_OP_KEEP) ? key.opIter - (CCU_SYNC_OP_KEEP - 1U) : 0U;
            if (minKeepRound == 0U && minKeepOp == 0U) {
                return;
            }
            // 裁剪谓词迁入数据库端条件：同坐标（不含 opIter/round）且
            // （opIter 过老整段清除 或 本 op 内 round 过老清除）。
            const auto stale = sim::runtime::Db::GetByPred<sim::runtime::CcuSyncResTab>(HcclSim::Storage::And(
                SyncCoordsCond(key),
                HcclSim::Storage::Or(
                    HcclSim::Storage::Lt(&sim::runtime::CcuSyncResTab::opIter, minKeepOp),
                    HcclSim::Storage::And(
                        HcclSim::Storage::Eq(&sim::runtime::CcuSyncResTab::opIter, key.opIter),
                        HcclSim::Storage::Lt(&sim::runtime::CcuSyncResTab::round, minKeepRound)))));
            if (!stale.ok() || !stale.value.has_value()) {
                return;
            }
            for (const auto& row : *stale.value) {
                (void)sim::runtime::Db::Delete<sim::runtime::CcuSyncResTab>(
                    HcclSim::Storage::Eq(&sim::runtime::CcuSyncResTab::id, row.id));
            }
        }

        /** @brief 写方发布交换值（键含轮次，一轮一行；同轮次幂等重发只更新值）。 */
        void PublishSyncRes(const SimCcuSyncKey& key, uint64_t value, uint32_t srcRank)
        {
            const auto existing = sim::runtime::Db::GetOneByPred<sim::runtime::CcuSyncResTab>(SyncKeyCond(key));
            if (existing.ok()) {
                // 命中行已含相同 opIter（键条件含 opIter），仅更新值/写方/写方进程
                (void)sim::runtime::Db::Update<sim::runtime::CcuSyncResTab>(
                    HcclSim::Storage::Eq(&sim::runtime::CcuSyncResTab::id, existing->id),
                    HcclSim::Storage::Set(&sim::runtime::CcuSyncResTab::value, value),
                    HcclSim::Storage::Set(&sim::runtime::CcuSyncResTab::srcRank, srcRank),
                    HcclSim::Storage::Set(&sim::runtime::CcuSyncResTab::ownerPid, static_cast<uint64_t>(getpid())));
            } else {
                sim::runtime::CcuSyncResTab rec{};
                rec.opIter = key.opIter;
                rec.round = key.round;
                rec.targetRank = key.targetRank;
                rec.connOrdinal = key.connOrdinal;
                rec.dieId = key.dieId;
                rec.resType = key.resType;
                rec.resId = key.resId;
                rec.value = value;
                rec.srcRank = srcRank;
                rec.ownerPid = static_cast<uint64_t>(getpid());
                (void)sim::runtime::Db::Add<sim::runtime::CcuSyncResTab>(rec);
                TrimSyncResHistory(key);
            }
            HCCL_VM_INFO(
                "ccu publish sync res: opIter={}, round={}, targetRank={}, conn={}, die={}, type={}, id={}, "
                "value=0x{:x}",
                key.opIter, key.round, key.targetRank, key.connOrdinal, key.dieId, key.resType, key.resId, value);
        }

        /**
         * @brief 读方按**本读方所在轮次**取交换值（有界轮询）。
         *
         * 轮次隔离：只在本轮次的行上等待；超时兜底取该坐标"最近一轮"的已有值并 WARN
         * （交换值基本是跨轮稳定的缓冲区地址，取最近值远优于回落 0——0 会让上层算出非法地址）。
         *
         * @return true = 取到值（正常或兜底）；false = 该坐标从未有任何轮次的行。
         */
        /**
         * @brief 交换值等待超时兜底：取**本 op 内**该坐标"最近一轮"的已有值并 WARN。
         *
         * 兜底只在同一 opIter 内取（SameSyncCoords 含 opIter）：交换值（缓冲区地址）在同一 op 内跨轮
         * 稳定，取本 op 的最近轮安全；**绝不取别的 op 的值**——不同 op 地址不同，取到会导致上层算出
         * 非法/越界地址（历史现象：连跑用例 checker 报 MEM_CPY 地址落不到本 op 内存布局）。
         *
         * @return true = 取到同 op 兜底值；false = 本 op 该坐标从未发布过任何轮次的行（调用方保留默认 0）。
         */
        bool WaitSyncResFallback(const SimCcuSyncKey& key, uint64_t& value)
        {
            const auto history = sim::runtime::Db::GetByPred<sim::runtime::CcuSyncResTab>(SameOpSyncCoordsCond(key));
            bool hasLatest = false;
            uint32_t latestRound = 0;
            uint64_t latestValue = 0;
            if (history.ok() && history.value.has_value()) {
                for (const auto& cand : *history.value) {
                    if (!hasLatest || cand.round > latestRound) {
                        hasLatest = true;
                        latestRound = cand.round;
                        latestValue = cand.value;
                    }
                }
            }
            if (hasLatest) {
                value = latestValue;
                HCCL_VM_WARN(
                    "ccu wait sync res: opIter={}, round={} not published yet, fall back to round={} value=0x{:x} "
                    "(targetRank={}, conn={}, die={}, type={}, id={})",
                    key.opIter, key.round, latestRound, value, key.targetRank, key.connOrdinal, key.dieId, key.resType,
                    key.resId);
                return true;
            }
            HCCL_VM_WARN(
                "ccu wait sync res timeout: no row at all (opIter={}, round={}, targetRank={}, conn={}, die={}, "
                "type={}, id={})",
                key.opIter, key.round, key.targetRank, key.connOrdinal, key.dieId, key.resType, key.resId);
            return false;
        }

        /**
         * @brief 交换值等待：有界轮询本轮次的行；超时回落到 WaitSyncResFallback 取"最近一轮"已有值。
         * @return true = 取到值（本轮或兜底）；false = 该坐标从未有任何轮次的行。
         */
        bool WaitSyncRes(const SimCcuSyncKey& key, uint64_t& value)
        {
            const auto deadline
                = std::chrono::steady_clock::now() + std::chrono::milliseconds(CCU_SYNC_WAIT_TIMEOUT_MS);
            while (true) {
                const auto row = sim::runtime::Db::GetOneByPred<sim::runtime::CcuSyncResTab>(SyncKeyCond(key));
                if (row.ok()) {
                    value = row->value;
                    HCCL_VM_INFO(
                        "ccu wait sync res: opIter={}, round={}, targetRank={}, conn={}, die={}, type={}, id={}, "
                        "value=0x{:x} (from rank={})",
                        key.opIter, key.round, key.targetRank, key.connOrdinal, key.dieId, key.resType, key.resId,
                        value, row->srcRank);
                    return true;
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    return WaitSyncResFallback(key, value);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(CCU_SYNC_WAIT_POLL_MS));
            }
        }

        /** @brief if/else 控制帧。 */
        struct SimCcuIfFrame {
            bool ifTaken{false}; /**< if 分支条件判定结果。 */
            bool inElse{false};  /**< 当前是否处于 else 分支。 */
        };

        /** @brief 录制的任务记录（插入 TaskCollection 前的中间表示）。 */
        struct SimCcuTaskRecord {
            HccLTaskMetaType kind{HccLTaskMetaType::INVALID};
            uint32_t srcRank{0}; /**< rank 语义，插入前归一化为 deviceId。 */
            uint64_t srcOffset{0};
            uint32_t dstRank{0};
            uint64_t dstOffset{0};
            uint64_t len{0};        /**< MEM_CPY 字节数。 */
            uint64_t dataCount{0};  /**< REDUCE 元素个数。 */
            uint8_t dataType{0};    /**< REDUCE：HcclDataType。 */
            uint8_t reduceOp{0};    /**< REDUCE：HcclReduceOp。 */
            uint64_t notifyId{0};   /**< NOTIFY：配对 id。 */
            uint8_t notifyCount{0}; /**< NOTIFY：附加信息（事件 mask 截断）。 */
            std::string opName;     /**< 产生本记录的原语名（dump 用）。 */
        };

        /** @brief 录制期 if 标签栈项（CCU_IF/CCU_ELSE 宏配合，与真身 IfLabelEntry 对齐）。 */
        struct SimCcuIfLabelEntry {
            std::string label;
            bool bodyDone{false};
        };

        /**
         * @brief 注册期录制上下文：kernelFunc 执行期间原语桩的唯一写入对象。
         *
         * 生命周期不变量（设计文档 §2.3）：
         *  - I1 仅注册线程可见（thread_local）：kernelFunc 与原语桩严格同线程；
         *  - I3 同一线程不并发注册两个 kernel（active 重入检查拒绝）；
         *  - I5 录制窗口内签发的句柄携带 kernelIdx，与注册表 kernelHandle 同源编号。
         */
        struct SimCcuRecordContext {
            bool active{false};
            bool isFlushing{false}; /**< flush 防重入（对齐真身 isFlushing_）。 */
            uint32_t kernelIdx{0};  /**< 录制 kernel 序号（注册表提前占号，与 kernelHandle 同值）。 */
            uint64_t handleSeq{0};  /**< kernel 内句柄序列（句柄低 32 位）。 */
            std::vector<SimCcuPrim> trace;           /**< 构建中的轨迹。 */
            std::vector<std::string> strings;        /**< label/notifyTag 字符串池。 */
            std::vector<uint64_t> extraBufs;         /**< LOCAL_REDUCE_BUFS 句柄数组存储。 */
            std::vector<SimCcuIfLabelEntry> ifStack; /**< if 标签栈。 */
            std::vector<std::string> doWhileStack;   /**< do-while 标签栈（CCU_DO/CCU_WHILE 宏配合）。 */
            std::set<std::string> whileOpenLabels; /**< 未闭合 while 的 label（对齐真身 pendingWhileCtx_ 查重）。 */
            std::set<std::string> doWhileOpenLabels; /**< 未闭合 do-while 的 label（对齐 pendingDoWhileCtx_）。 */
            /* Loop/LoopGroup 录制态 */
            uint32_t loopSeq{0};                    /**< loop 序号分配器。 */
            uint32_t curLoopDepth{0};               /**< 当前 loop 体嵌套深度（>0 时禁 if/while/Loop）。 */
            std::vector<uint32_t> loopBodyEnterIdx; /**< 每个 loop 的 LOOP_BODY_ENTER 轨迹索引。 */
            std::vector<uint32_t> loopBodyExitIdx;  /**< 每个 loop 的 LOOP_BODY_EXIT 轨迹索引。 */
            std::vector<SimCcuLoopGroup> groups;    /**< 按 GroupCreate 顺序。 */
            std::map<uint64_t, std::pair<uint64_t, uint32_t>> channelVarSources; /**< 通道派生变量来源（交换区）。 */
            CcuResult funcRet{CcuResult::CCU_SUCCESS}; /**< kernelFunc 返回值（异常时 CCU_E_INTERNAL）。 */
        };

        /** @brief Launch 期回放上下文：回放器的唯一读写对象。 */
        struct SimCcuExecContext {
            bool active{false};
            uint64_t kernelHandle{0};
            uint64_t threadHandle{0};
            uint64_t commId{0};
            uint64_t streamId{0};
            uint64_t deviceId{0};
            uint32_t rankId{0};
            uint32_t dieId{0};
            uint64_t launchIdx{0};
            std::vector<uint64_t> taskArgs;         /**< 全量 Launch 参数（已做地址转换）。 */
            std::map<uint64_t, uint64_t> varValues; /**< varHandle -> 当前值。 */
            std::map<uint64_t, uint64_t> addrBases; /**< addrHandle -> 基值。 */
            std::map<uint64_t, std::pair<uint64_t, uint64_t>> localAddrParts;  /**< localAddr -> (addr, token)。 */
            std::map<uint64_t, std::pair<uint64_t, uint64_t>> remoteAddrParts; /**< remoteAddr -> (addr, token)。 */
            std::map<uint64_t, uint64_t> bufferOffsets;                        /**< bufHandle -> 伪地址。 */
            uint64_t bufferCursor{0};
            std::vector<SimCcuIfFrame> ifFrames;                                 /**< 控制流帧栈。 */
            std::vector<SimCcuTaskRecord> records;                               /**< 任务序列。 */
            std::map<uint64_t, std::pair<uint64_t, uint32_t>> channelVarSources; /**< 交换变量来源（entry 拷贝）。 */
            std::set<uint64_t> resolvedChannelVars; /**< 已解析的交换变量句柄（避免重复等待）。 */
        };

        /* 前向声明（定义在 GetChannelRemoteRank 之后）：交换变量首次读取处懒解析。 */
        bool ResolveChannelVarIfNeeded(SimCcuExecContext* ctx, uint64_t varHandle);

        thread_local SimCcuRecordContext g_tlsRecordCtx;
        thread_local SimCcuExecContext g_tlsExecCtx;

        /** @brief 执行期全局唯一句柄/序号分配器（Event 句柄全局唯一兼作 notifyId 编码原料）。 */
        std::atomic<uint64_t> g_globalHandleSeq{1};
        std::atomic<uint64_t> g_globalEventSeq{1};
        std::atomic<uint64_t> g_launchIdxSeq{1};

        /** @brief 进程级 kernel 注册表。 */
        struct SimCcuRegistryState {
            std::mutex mutex;
            std::map<uint64_t, SimCcuKernelEntry> kernelByHandle; /**< kernelHandle -> 条目。 */
            std::set<uint64_t> registeringIns;                    /**< 处于注册事务中的实例。 */
            uint64_t nextKernelHandle{1};
        };

        /** @brief 进程级 kernel 注册表单例（内核句柄表 + 注册事务集合 + 句柄分配器）。 */
        SimCcuRegistryState& RegistryState()
        {
            static SimCcuRegistryState instance;
            return instance;
        }

        /** @brief 取当前线程的活跃录制上下文；未处于录制期返回 nullptr。 */
        SimCcuRecordContext* ActiveRecordCtx() { return g_tlsRecordCtx.active ? &g_tlsRecordCtx : nullptr; }

        /** @brief 取当前线程的活跃回放上下文；未处于回放期返回 nullptr。 */
        SimCcuExecContext* ActiveExecCtx() { return g_tlsExecCtx.active ? &g_tlsExecCtx : nullptr; }

        /** @brief 控制流抑制判断：任一外层帧未落在选中分支上则不求值/不录制。 */
        bool ShouldRecordCtx(const SimCcuExecContext& ctx)
        {
            for (const SimCcuIfFrame& frame : ctx.ifFrames) {
                if (!frame.inElse && !frame.ifTaken) {
                    return false;
                }
                if (frame.inElse && frame.ifTaken) {
                    return false;
                }
            }
            return true;
        }

        /** @brief 事件位资源 id：真机 (设备, CKE寄存器, bit) 三维度资源身份的 u64 压平。
         *  布局：0xE 段 | rank(16b,bit[55:40]) | eventSeq(32b,bit[39:4]) | bit(4b,bit[3:0])。
         *  rank 为设备维度（CKE 寄存器为 per-die 硬件资源，且事件句柄每进程独立从 1 起编号，
         *  缺设备维度会导致跨进程同号串扰）；eventSeq 取事件句柄低 32 位（进程内唯一）；
         *  bit 为 CKE 位序号（mask≤16bit）。真机/wait 与 record 双方在同一 rank 内配对。 */
        uint64_t EventBitNotifyId(uint32_t rankId, uint64_t eventHandle, uint32_t bit)
        {
            return CCU_EVENT_NOTIFY_ID_BASE | (static_cast<uint64_t>(rankId & 0xFFFFU) << 40U)
                   | ((eventHandle & 0xFFFFFFFFULL) << 4U) | static_cast<uint64_t>(bit & 0xFU);
        }

        /** @brief 通道位资源 id：真机 (设备对方向, CKE槽位, bit) 的 u64 压平。
         *  布局：0xD 段 | srcRank(16b,bit[59:44]) | dstRank(16b,bit[43:28]) | notifyIdx(8b,bit[15:8]) | bit(4b)。
         *  两侧无需通信即可推导同一 id：record 方由 (ctx.rankId, channel.remoteRankId)、wait 方由
         *  (channel.remoteRankId, ctx.rankId) 得到同一方向化 rank 对；不依赖任何单侧实体
         *  （channel 句柄 / DB notifyId 均为单侧值，两侧不同号会导致跨 rank 配对失败）。 */
        uint64_t ChannelBitNotifyId(uint32_t srcRank, uint32_t dstRank, uint32_t notifyIdx, uint32_t bit)
        {
            return CCU_CHANNEL_NOTIFY_ID_BASE | (static_cast<uint64_t>(srcRank & 0xFFFFU) << 44U)
                   | (static_cast<uint64_t>(dstRank & 0xFFFFU) << 28U)
                   | (static_cast<uint64_t>(notifyIdx & 0xFFU) << 8U) | static_cast<uint64_t>(bit & 0xFU);
        }

        /** @brief 算逻求值 lhs <op> rhs（ADD/SUB/MUL/AND/OR/XOR/SHL/SHR；移位 >=64 归 0）。 */
        uint64_t ApplyVarOpValue(uint64_t lhs, uint64_t rhs, SimCcuVarOp op)
        {
            switch (op) {
                case SimCcuVarOp::ADD:
                    return lhs + rhs;
                case SimCcuVarOp::SUB:
                    return lhs - rhs;
                case SimCcuVarOp::MUL:
                    return lhs * rhs;
                case SimCcuVarOp::AND:
                    return lhs & rhs;
                case SimCcuVarOp::OR:
                    return lhs | rhs;
                case SimCcuVarOp::XOR:
                    return lhs ^ rhs;
                case SimCcuVarOp::SHL:
                    return (rhs >= 64U) ? 0U : (lhs << rhs);
                case SimCcuVarOp::SHR:
                    return (rhs >= 64U) ? 0U : (lhs >> rhs);
                default:
                    return 0U;
            }
        }

        /** @brief 向回放上下文追加一条任务记录（外层 if 未命中时抑制）。 */
        void AppendRecord(SimCcuExecContext& ctx, const SimCcuTaskRecord& record)
        {
            if (!ShouldRecordCtx(ctx)) {
                return;
            }
            ctx.records.push_back(record);
        }

        /** @brief 轨迹条目类型短名表（回放期指令打印用；与 SimCcuPrimKind 语义一一对应）。 */
        const std::pair<SimCcuPrimKind, const char*> kPrimKindNames[] = {
            {SimCcuPrimKind::VAR_ASSIGN_IMM, "var<=imm"},
            {SimCcuPrimKind::VAR_ASSIGN_VAR, "var<=var"},
            {SimCcuPrimKind::VAR_OP_VAR, "var=varOpVar"},
            {SimCcuPrimKind::VAR_OP_IMM, "var=varOpImm"},
            {SimCcuPrimKind::VAR_NOT, "var=~var"},
            {SimCcuPrimKind::ADDR_ASSIGN_IMM, "addr<=imm"},
            {SimCcuPrimKind::ADDR_ASSIGN_ADDR, "addr<=addr"},
            {SimCcuPrimKind::ADDR_ASSIGN_VAR, "addr<=var"},
            {SimCcuPrimKind::ADDR_OP_VAR, "addr=addrOpVar"},
            {SimCcuPrimKind::ADDR_OP_ADDR, "addr=addrOpAddr"},
            {SimCcuPrimKind::ADDR_OP_IMM, "addr=addrOpImm"},
            {SimCcuPrimKind::ADDR_ADDASSIGN_VAR, "addr+=var"},
            {SimCcuPrimKind::LOAD_ARG, "loadArg"},
            {SimCcuPrimKind::LOAD_MEM, "loadMem"},
            {SimCcuPrimKind::LOAD_MEM_VAR, "loadMemVar"},
            {SimCcuPrimKind::STORE_MEM, "storeMem"},
            {SimCcuPrimKind::STORE_MEM_VAR, "storeMemVar"},
            {SimCcuPrimKind::LOCAL_ADDR_ALLOC, "localAddrAlloc"},
            {SimCcuPrimKind::REMOTE_ADDR_ALLOC, "remoteAddrAlloc"},
            {SimCcuPrimKind::EVENT_RECORD, "eventRecord"},
            {SimCcuPrimKind::EVENT_WAIT, "eventWait"},
            {SimCcuPrimKind::NOTIFY_RECORD, "notifyRecord"},
            {SimCcuPrimKind::NOTIFY_WAIT, "notifyWait"},
            {SimCcuPrimKind::WRITE_VAR_NOTIFY, "writeVarNotify"},
            {SimCcuPrimKind::LOCAL_NOTIFY_RECORD, "localNotifyRecord"},
            {SimCcuPrimKind::LOCAL_NOTIFY_WAIT, "localNotifyWait"},
            {SimCcuPrimKind::LOCAL_COPY_MM, "localCopyMM"},
            {SimCcuPrimKind::LOCAL_COPY_MB, "localCopyMB"},
            {SimCcuPrimKind::LOCAL_COPY_BM, "localCopyBM"},
            {SimCcuPrimKind::LOCAL_REDUCE_MM, "localReduceMM"},
            {SimCcuPrimKind::LOCAL_REDUCE_BUFS, "localReduceBufs"},
            {SimCcuPrimKind::REMOTE_READ_MM, "readMM"},
            {SimCcuPrimKind::REMOTE_READ_MB, "readMB"},
            {SimCcuPrimKind::REMOTE_READ_MR, "readMR"},
            {SimCcuPrimKind::REMOTE_WRITE_MM, "writeMM"},
            {SimCcuPrimKind::REMOTE_WRITE_BM, "writeBM"},
            {SimCcuPrimKind::REMOTE_WRITE_MR, "writeMR"},
            {SimCcuPrimKind::IF_BEGIN, "ifBegin"},
            {SimCcuPrimKind::IF_ELSE, "ifElse"},
            {SimCcuPrimKind::IF_END, "ifEnd"},
            {SimCcuPrimKind::WHILE_BEGIN, "whileBegin"},
            {SimCcuPrimKind::WHILE_END, "whileEnd"},
            {SimCcuPrimKind::DO_BEGIN, "doBegin"},
            {SimCcuPrimKind::DO_END, "doEnd"},
            {SimCcuPrimKind::LOOP_BODY_ENTER, "loopBodyEnter"},
            {SimCcuPrimKind::LOOP_BODY_EXIT, "loopBodyExit"},
            {SimCcuPrimKind::LOOP_GROUP_BEGIN, "loopGroupBegin"},
            {SimCcuPrimKind::LOOP_GROUP_END, "loopGroupEnd"},
        };

        /** @brief 轨迹条目类型短名（回放期指令打印用）。 */
        const char* PrimKindName(SimCcuPrimKind kind)
        {
            for (const auto& kv : kPrimKindNames) {
                if (kv.first == kind) {
                    return kv.second;
                }
            }
            return "unknown";
        }

        /*--------------------------------------------------------------------------
         * rank -> device 归一化（与 data_comm_op_stub.cc 的 level1 任务口径保持一致）
         *----------------------------------------------------------------------*/
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
         * @brief 归一化单个端点：录制 rank 语义改写为本地 rank，并把 rank 解析为 deviceId（原地写回）。
         * @return 解析成功返回 true。
         */
        bool NormalizeTaskEndpoint(uint64_t commId, uint32_t generatedRankId, uint32_t localRankId, uint32_t& rankId)
        {
            if (rankId == generatedRankId) {
                rankId = localRankId;
            }
            return GetTaskDeviceId(commId, rankId, rankId);
        }

        /** @brief 任务 rank→device 归一化：改写本地 rank 并解析各端点 deviceId（供插入 TaskCollection）。 */
        bool NormalizeSimTask(HcclTaskMetaData* task)
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
            uint32_t deviceId = 0;
            if (!GetTaskDeviceId(task->commId, task->rankId, deviceId)) {
                return false;
            }
            task->deviceId = deviceId;

            switch (task->taskType) {
                case HccLTaskMetaType::MEM_CPY:
                    return NormalizeTaskEndpoint(
                               task->commId, generatedRankId, localRankId, task->taskData.transMem.srcDeviceId)
                           && NormalizeTaskEndpoint(
                               task->commId, generatedRankId, localRankId, task->taskData.transMem.dstDeviceId);
                case HccLTaskMetaType::REDUCE:
                    return NormalizeTaskEndpoint(
                               task->commId, generatedRankId, localRankId, task->taskData.reduce.srcDeviceId)
                           && NormalizeTaskEndpoint(
                               task->commId, generatedRankId, localRankId, task->taskData.reduce.dstDeviceId);
                case HccLTaskMetaType::NOTIFY_RECORD:
                case HccLTaskMetaType::NOTIFY_WAIT:
                    return NormalizeTaskEndpoint(
                               task->commId, generatedRankId, localRankId, task->taskData.notify.srcDeviceId)
                           && NormalizeTaskEndpoint(
                               task->commId, generatedRankId, localRankId, task->taskData.notify.dstDeviceId);
                default:
                    return true;
            }
        }

        /** @brief 任务节点终态描述（归一化+插入后，排障打印用；偏移/notifyId 按十六进制）。 */
        std::string DescribeCcuTask(const HcclTaskMetaData& task, const std::string& opName)
        {
            std::ostringstream oss;
            oss << "op=" << opName << ", type=" << task.TaskTypeName() << ", commId=" << task.commId
                << ", rankId=" << task.rankId << ", deviceId=" << task.deviceId << ", streamId=" << task.streamId;
            switch (task.taskType) {
                case HccLTaskMetaType::MEM_CPY:
                    oss << ", src(dev=" << task.taskData.transMem.srcDeviceId << ")@0x" << std::hex
                        << task.taskData.transMem.srcOffset << std::dec
                        << ", dst(dev=" << task.taskData.transMem.dstDeviceId << ")@0x" << std::hex
                        << task.taskData.transMem.dstOffset << std::dec << ", len=" << task.taskData.transMem.len;
                    break;
                case HccLTaskMetaType::REDUCE:
                    oss << ", src(dev=" << task.taskData.reduce.srcDeviceId << ")@0x" << std::hex
                        << task.taskData.reduce.srcOffset << std::dec
                        << ", dst(dev=" << task.taskData.reduce.dstDeviceId << ")@0x" << std::hex
                        << task.taskData.reduce.dstOffset << std::dec
                        << ", dataCount=" << task.taskData.reduce.dataCount
                        << ", dataType=" << static_cast<uint32_t>(task.taskData.reduce.dataType)
                        << ", reduceOp=" << static_cast<uint32_t>(task.taskData.reduce.reduceOp);
                    break;
                case HccLTaskMetaType::NOTIFY_RECORD:
                case HccLTaskMetaType::NOTIFY_WAIT:
                    oss << ", src(dev=" << task.taskData.notify.srcDeviceId
                        << "), dst(dev=" << task.taskData.notify.dstDeviceId << "), notifyId=0x" << std::hex
                        << task.taskData.notify.notifyId << std::dec
                        << ", notifyCount=" << static_cast<uint32_t>(task.taskData.notify.notifyCount);
                    break;
                default:
                    break;
            }
            return oss.str();
        }

        /** @brief dump 单条任务记录（MEM_CPY/REDUCE/NOTIFY）到输出流。 */
        void DumpOneRecord(std::ostream& out, const SimCcuTaskRecord& rec, size_t idx)
        {
            out << "[" << idx << "] op=" << rec.opName;
            switch (rec.kind) {
                case HccLTaskMetaType::MEM_CPY:
                    out << ", kind=MEM_CPY, src(rank)=" << rec.srcRank << ", srcOffset=0x" << std::hex << rec.srcOffset
                        << std::dec << ", dst(rank)=" << rec.dstRank << ", dstOffset=0x" << std::hex << rec.dstOffset
                        << std::dec << ", len=" << rec.len;
                    break;
                case HccLTaskMetaType::REDUCE:
                    out << ", kind=REDUCE, src(rank)=" << rec.srcRank << ", srcOffset=0x" << std::hex << rec.srcOffset
                        << std::dec << ", dst(rank)=" << rec.dstRank << ", dstOffset=0x" << std::hex << rec.dstOffset
                        << std::dec << ", dataCount=" << rec.dataCount
                        << ", dataType=" << static_cast<uint32_t>(rec.dataType)
                        << ", reduceOp=" << static_cast<uint32_t>(rec.reduceOp);
                    break;
                case HccLTaskMetaType::NOTIFY_RECORD:
                case HccLTaskMetaType::NOTIFY_WAIT:
                    out << ", kind=" << (rec.kind == HccLTaskMetaType::NOTIFY_RECORD ? "NOTIFY_RECORD" : "NOTIFY_WAIT")
                        << ", src(rank)=" << rec.srcRank << ", dst(rank)=" << rec.dstRank << ", notifyId=0x" << std::hex
                        << rec.notifyId << std::dec << ", notifyCount=" << static_cast<uint32_t>(rec.notifyCount);
                    break;
                default:
                    break;
            }
            out << "\n";
        }

        /** @brief dump 变量/地址终值表（句柄 -> 回放结束时的值，与录制日志的 alloc 行对照）。 */
        void DumpVarTable(std::ostream& out, const SimCcuExecContext& ctx)
        {
            out << "\n[var table] handle -> final value:\n";
            for (const auto& kv : ctx.varValues) {
                out << "  var 0x" << std::hex << kv.first << std::dec << " = " << kv.second << " (0x" << std::hex
                    << kv.second << std::dec << ")\n";
            }
            for (const auto& kv : ctx.addrBases) {
                out << "  addr 0x" << std::hex << kv.first << std::dec << " = " << kv.second << " (0x" << std::hex
                    << kv.second << std::dec << ")\n";
            }
        }

        /** @brief 把录制的任务序列落盘（排障用，失败仅告警不影响主流程）。 */
        void DumpSequenceFile(const SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, uint32_t insertCount)
        {
            std::ostringstream name;
            name << "ccu_sim_seq_device_" << ctx.deviceId << "_launch_" << ctx.launchIdx << ".txt";
            const std::string path = InstallPath::ResolveToInstallRoot("data/" + name.str());
            std::ofstream out(path, std::ios::out | std::ios::trunc);
            if (!out.is_open()) {
                HCCL_VM_WARN("cannot open dump file {}, skip sequence dump", path);
                return;
            }
            out << "ccu cpu-sim sequence: kernel=" << entry.funcName << ", kernelHandle=" << ctx.kernelHandle
                << ", launchIdx=" << ctx.launchIdx << ", deviceId=" << ctx.deviceId << ", rankId=" << ctx.rankId
                << ", dieId=" << entry.dieId << ", streamId=" << ctx.streamId << ", commId=" << ctx.commId
                << ", argNum=" << ctx.taskArgs.size() << ", primCount=" << entry.trace.size()
                << ", recordCount=" << ctx.records.size() << ", insertedTaskCount=" << insertCount << "\n";
            for (size_t i = 0; i < ctx.taskArgs.size(); ++i) {
                out << "[SQE Arg][" << i << "]: 0x" << std::hex << ctx.taskArgs[i] << std::dec << "\n";
            }
            for (size_t i = 0; i < ctx.records.size(); ++i) {
                DumpOneRecord(out, ctx.records[i], i);
            }
            HCCL_VM_INFO("dumped ccu cpu-sim sequence to {}", path);
            DumpVarTable(out, ctx);
        }

        /**
         * @brief 由回放任务记录构造 taskmeta。
         * @return false = 记录类型不支持（应跳过）。
         */
        bool BuildTaskFromRecord(const SimCcuExecContext& ctx, const SimCcuTaskRecord& rec, HcclTaskMetaData& task)
        {
            task.taskType = rec.kind;
            task.commId = ctx.commId;
            task.rankId = ctx.rankId;
            task.streamId = ctx.streamId;
            task.deviceId = ctx.deviceId;
            switch (rec.kind) {
                case HccLTaskMetaType::MEM_CPY:
                    task.taskData.transMem.srcDeviceId = rec.srcRank;
                    task.taskData.transMem.srcOffset = rec.srcOffset;
                    task.taskData.transMem.dstDeviceId = rec.dstRank;
                    task.taskData.transMem.dstOffset = rec.dstOffset;
                    task.taskData.transMem.len = rec.len;
                    return true;
                case HccLTaskMetaType::REDUCE:
                    task.taskData.reduce.srcDeviceId = rec.srcRank;
                    task.taskData.reduce.srcOffset = rec.srcOffset;
                    task.taskData.reduce.dstDeviceId = rec.dstRank;
                    task.taskData.reduce.dstOffset = rec.dstOffset;
                    task.taskData.reduce.dataCount = rec.dataCount;
                    task.taskData.reduce.dataType = rec.dataType;
                    task.taskData.reduce.reduceOp = rec.reduceOp;
                    return true;
                case HccLTaskMetaType::NOTIFY_RECORD:
                case HccLTaskMetaType::NOTIFY_WAIT:
                    task.taskData.notify.srcDeviceId = rec.srcRank;
                    task.taskData.notify.notifyId = rec.notifyId;
                    task.taskData.notify.dstDeviceId = rec.dstRank;
                    task.taskData.notify.notifyCount = rec.notifyCount;
                    return true;
                default:
                    return false;
            }
        }

        /**
         * @brief 归一化并插入单条任务记录。
         * @param inserted 输出：是否成功插入（未支持类型为 false）。
         * @return SUCCESS 表示成功或跳过；否则为归一化/插入错误码。
         */
        HcclVmResult InsertOneRecord(const SimCcuExecContext& ctx, const SimCcuTaskRecord& rec, bool& inserted)
        {
            inserted = false;
            HcclTaskMetaData task;
            if (!BuildTaskFromRecord(ctx, rec, task)) {
                return HCCL_SIM_SUCCESS; // 未支持类型：跳过
            }
            if (!NormalizeSimTask(&task)) {
                HCCL_VM_ERROR("cannot normalize ccu cpu-sim task, task=[{}]", DescribeCcuTask(task, rec.opName));
                return HCCL_SIM_E_INTERNAL;
            }
            uint32_t index = 0;
            const HcclVmResult ret = InsertTaskToCollection(&task, &index);
            if (ret != HCCL_SIM_SUCCESS) {
                HCCL_VM_ERROR(
                    "insert ccu cpu-sim task failed, ret={}, kind={}, op={}", static_cast<int32_t>(ret),
                    static_cast<int32_t>(rec.kind), rec.opName);
                return ret;
            }
            HCCL_VM_INFO("ccu cpu-sim tasknode[{}]: {}", index, DescribeCcuTask(task, rec.opName));
            inserted = true;
            return HCCL_SIM_SUCCESS;
        }

        /** @brief 收尾：录制序列 -> 归一化 -> 插入 TaskCollection -> dump -> 清理 TLS。 */
        HcclVmResult FinishExecution(const SimCcuKernelEntry& entry)
        {
            SimCcuExecContext& ctx = g_tlsExecCtx;
            // 【收尾步骤 1】逐条回放记录 -> taskmeta -> 归一化 -> 插入 TaskCollection。
            uint32_t insertCount = 0;
            HcclVmResult firstErr = HCCL_SIM_SUCCESS;
            for (const SimCcuTaskRecord& rec : ctx.records) {
                bool inserted = false;
                const HcclVmResult ret = InsertOneRecord(ctx, rec, inserted);
                if (ret != HCCL_SIM_SUCCESS && firstErr == HCCL_SIM_SUCCESS) {
                    firstErr = ret;
                }
                if (inserted) {
                    ++insertCount;
                }
            }

            // 【收尾步骤 2】落盘 dump + 汇总日志。
            DumpSequenceFile(ctx, entry, insertCount);
            HCCL_VM_INFO(
                "CCU kernel replayed on cpu, kernel={}, kernelHandle={}, launchIdx={}, deviceId={}, "
                "primCount={}, recordCount={}, insertedTaskCount={}",
                entry.funcName, ctx.kernelHandle, ctx.launchIdx, ctx.deviceId, entry.trace.size(), ctx.records.size(),
                insertCount);
            // 【收尾步骤 3】清理 TLS 回放上下文。
            g_tlsExecCtx = SimCcuExecContext();
            return firstErr;
        }

        /*==================== 录制：Prim 追加辅助 ====================*/

        /**
         * @brief 闭合栈顶连续 bodyDone 的挂起 if（对齐真身 FlushClosablePendingIfs）。
         *
         * 每闭合一个补录一条 IF_END 轨迹（无 else 的 if，闭合位置即当前轨迹末尾——
         * 与真身在此处放置 elseLabel 作跳转目标的语义一致）。
         * isFlushing 防重入（PrimIfEnd -> AppendPrim 会再进本函数）。
         */
        SimCcuPrim& AppendPrim(SimCcuPrimKind kind); // 前置声明（CloseClosableIfs 与其相互引用）
        uint32_t PoolString(const char* text);       // 前置声明（定义在下方匿名命名空间）

        /** @brief 闭合栈顶连续 bodyDone 的挂起 if（各补录一条 IF_END），带 isFlushing 重入保护。 */
        void CloseClosableIfs(SimCcuRecordContext* ctx)
        {
            if (ctx == nullptr || ctx->isFlushing) {
                return;
            }
            ctx->isFlushing = true;
            while (!ctx->ifStack.empty() && ctx->ifStack.back().bodyDone) {
                const std::string label = ctx->ifStack.back().label;
                ctx->ifStack.pop_back();
                SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::IF_END);
                prim.strIdx = PoolString(label.c_str());
            }
            ctx->isFlushing = false;
        }

        /**
         * @brief 向录制上下文追加一条轨迹（非录制期打点丢弃）。
         *
         * 对齐真身 CcuKernel::Append（ccu_kernel.cc:1385-1389）：任何轨迹条目落位前
         * 先闭合栈顶已 bodyDone 的挂起 if——保证无 else 的 if 其 IF_END 恰好落在
         * "if 体结束、下一原语之前"（真身 elseLabel 落位位置），而非延迟到后续宏增量。
         */
        SimCcuPrim& AppendPrim(SimCcuPrimKind kind)
        {
            static SimCcuPrim dummy; // 非录制期返回的丢弃槽（线程安全性：仅本线程写，值无意义）
            SimCcuRecordContext* ctx = ActiveRecordCtx();
            if (ctx == nullptr) {
                dummy = SimCcuPrim();
                dummy.kind = kind;
                return dummy;
            }
            CloseClosableIfs(ctx);
            ctx->trace.emplace_back();
            SimCcuPrim& prim = ctx->trace.back();
            prim.kind = kind;
            return prim;
        }

        /** @brief 字符串入池（label/notifyTag），返回下标。 */
        uint32_t PoolString(const char* text)
        {
            SimCcuRecordContext* ctx = ActiveRecordCtx();
            if (ctx == nullptr) {
                return 0;
            }
            ctx->strings.emplace_back((text != nullptr) ? text : "");
            return static_cast<uint32_t>(ctx->strings.size() - 1U);
        }

        /*==================== 回放：求值/录制辅助（操作 EXEC 上下文）====================*/

        bool ExecSetVarValue(SimCcuExecContext* ctx, uint64_t varHandle, uint64_t value)
        {
            if (ctx == nullptr) {
                return false;
            }
            if (!ShouldRecordCtx(*ctx)) {
                return false;
            }
            ctx->varValues[varHandle] = value;
            return true;
        }

        /** @brief 读变量值（交换变量首次读取处懒解析跨 rank 值；缺失按 XN 上电默认 0）。 */
        bool ExecGetVarValue(SimCcuExecContext* ctx, uint64_t varHandle, uint64_t& value)
        {
            value = 0;
            if (ctx == nullptr) {
                return false;
            }
            // 交换变量：首次读取处懒解析（此时本端 PreSync 已写完、对端值已发布；见设计 §3.4.4）。
            (void)ResolveChannelVarIfNeeded(ctx, varHandle);
            const auto iter = ctx->varValues.find(varHandle);
            if (iter == ctx->varValues.end()) {
                // 与 XN 上电默认 0 的行为一致。
                return true;
            }
            value = iter->second;
            return true;
        }

        /** @brief 设置地址句柄基值（外层 if 未命中时抑制）。 */
        bool ExecSetAddrBase(SimCcuExecContext* ctx, uint64_t addrHandle, uint64_t base)
        {
            if (ctx == nullptr || !ShouldRecordCtx(*ctx)) {
                return false;
            }
            ctx->addrBases[addrHandle] = base;
            return true;
        }

        /** @brief 读地址句柄基值（缺失告警并返回 false）。 */
        bool ExecGetAddrBase(SimCcuExecContext* ctx, uint64_t addrHandle, uint64_t& base)
        {
            base = 0;
            if (ctx == nullptr) {
                return false;
            }
            const auto iter = ctx->addrBases.find(addrHandle);
            if (iter == ctx->addrBases.end()) {
                HCCL_VM_WARN("ccu address {} has no base yet", addrHandle);
                return false;
            }
            base = iter->second;
            return true;
        }

        /** @brief 本地偏移归一化：设备地址可换算虚拟地址时返回虚拟地址，否则原样返回。 */
        uint64_t NormalizeLocalOffset(uint64_t offset)
        {
            const uint64_t virAddr = sim::TryGetVirPtrByDevPtr(offset);
            return (virAddr != 0) ? virAddr : offset;
        }

        /** @brief 由本地地址句柄求偏移（取其 addr 句柄基值后做虚拟地址归一化）。 */
        bool ExecGetLocalAddrOffset(SimCcuExecContext* ctx, uint64_t localAddrHandle, uint64_t& offset)
        {
            offset = 0;
            if (ctx == nullptr) {
                return false;
            }
            const auto iter = ctx->localAddrParts.find(localAddrHandle);
            if (iter == ctx->localAddrParts.end()) {
                HCCL_VM_WARN("ccu local addr {} has no parts, cannot resolve offset", localAddrHandle);
                return false;
            }
            uint64_t base = 0;
            if (!ExecGetAddrBase(ctx, iter->second.first, base)) {
                return false;
            }
            offset = NormalizeLocalOffset(base);
            return true;
        }

        /** @brief 由远端地址句柄求偏移（level1 口径裸透传，设备归属交由 checker 校验）。 */
        bool ExecGetRemoteAddrOffset(
            SimCcuExecContext* ctx, uint64_t remoteAddrHandle, uint32_t remoteRankId, uint64_t& offset)
        {
            offset = 0;
            if (ctx == nullptr) {
                return false;
            }
            const auto iter = ctx->remoteAddrParts.find(remoteAddrHandle);
            if (iter == ctx->remoteAddrParts.end()) {
                HCCL_VM_WARN("ccu remote addr {} has no parts, cannot resolve offset", remoteAddrHandle);
                return false;
            }
            uint64_t base = 0;
            if (!ExecGetAddrBase(ctx, iter->second.first, base)) {
                return false;
            }
            // 远端地址按 level1 口径裸透传：checker 的 GetSlice 依据 level0 RecordOpDbInfo 登记的
            // per-rank 布局完成 地址->offset 匹配与设备归属校验（actualDeviceId vs 任务 src/dst），
            // proxy 侧无需自行做 dev->vir 换算（原换算的 VirtualMemBlock.rank_id 存设备号而此处
            // 传的是通信 rank，键必然失配；dev==vir 恒等时换算也只是返回原值）。
            (void)remoteRankId;
            offset = base;
            return true;
        }

        /**
         * @brief 取 CCU 缓冲(MS)端点地址：MS 窗口内**按腿长分配**（4KB 对齐、连续累进）。
         *
         * 背景（设计 §3.7 + loop 合并方案 S2'）：CCU 缓冲(MS)在真身里是 CCU 寄存器空间的一块。
         * level1 采用 **MS 地址段**编址——checker 在 `GetTaskDataSlice` 按地址段识别为
         * `MemType::MS_CCU`（MS 切片免边界校验，故合并后的大区间也不越界）。该段只由 level1 产出，
         * 南向 CCU 指令路径不触发，因此**无需注册内存布局块**。
         *
         * 编址：`addr = CCU_MS_BASE + 已分配游标`，长度按 `len` 4KB 对齐后累进；
         * 同一 `bufHandle` 的多次调用（如同一腿的 MB 写 / BM 读、或归约的多个槽）命中同一区间。
         *
         * @param len 本腿从该缓冲端点搬运的长度（决定区间大小；同句柄只需首次传入与实际相当的值）。
         * @return 缓冲端点地址；窗口耗尽时回绕到窗口起点并 WARN。
         */
        uint64_t ExecGetBufferOffset(SimCcuExecContext* ctx, uint64_t bufHandle, uint64_t len)
        {
            if (ctx == nullptr) {
                return 0;
            }
            const auto iter = ctx->bufferOffsets.find(bufHandle);
            if (iter != ctx->bufferOffsets.end()) {
                return iter->second;
            }
            const uint64_t need = (len == 0U ? 1U : len);
            const uint64_t alignedLen = (need + CCU_MS_GRANULARITY - 1U) & ~(CCU_MS_GRANULARITY - 1U);
            if (ctx->bufferCursor + alignedLen > CCU_MS_WINDOW_CAP) {
                HCCL_VM_WARN(
                    "ccu MS window exhausted (cursor={}, need={}, cap={}), wrap to window base", ctx->bufferCursor,
                    alignedLen, CCU_MS_WINDOW_CAP);
                ctx->bufferCursor = 0U;
            }
            const uint64_t offset = CCU_MS_BASE + static_cast<uint64_t>(ctx->bufferCursor);
            ctx->bufferCursor += alignedLen;
            ctx->bufferOffsets[bufHandle] = offset;
            return offset;
        }

        /** @brief 取第 argId 个 Launch 参数（越界报错返回 false）。 */
        bool ExecGetTaskArg(SimCcuExecContext* ctx, uint32_t argId, uint64_t& value)
        {
            value = 0;
            if (ctx == nullptr) {
                return false;
            }
            if (argId >= ctx->taskArgs.size()) {
                HCCL_VM_ERROR("ccu load arg out of range, argId={}, argNum={}", argId, ctx->taskArgs.size());
                return false;
            }
            value = ctx->taskArgs[argId];
            return true;
        }

        /** @brief 条件求值：lhs condType rhs（rhs 为立即数或变量句柄）。 */
        bool ExecEvalPrimCond(
            SimCcuExecContext* ctx, uint64_t lhsVar, uint64_t rhs, bool isVarCompare, CcuConditionType condType,
            bool& result)
        {
            result = false;
            uint64_t lhs = 0;
            if (!ExecGetVarValue(ctx, lhsVar, lhs)) {
                return false;
            }
            uint64_t rhsValue = rhs;
            if (isVarCompare) {
                if (!ExecGetVarValue(ctx, rhs, rhsValue)) {
                    return false;
                }
            }
            switch (condType) {
                case CCU_CONDITION_EQ:
                    result = (lhs == rhsValue);
                    return true;
                case CCU_CONDITION_NE:
                    result = (lhs != rhsValue);
                    return true;
                case CCU_CONDITION_LT:
                    result = (lhs < rhsValue);
                    return true;
                case CCU_CONDITION_LE:
                    result = (lhs <= rhsValue);
                    return true;
                case CCU_CONDITION_GT:
                    result = (lhs > rhsValue);
                    return true;
                case CCU_CONDITION_GE:
                    result = (lhs >= rhsValue);
                    return true;
                default:
                    HCCL_VM_WARN("unknown ccu condition type {}", static_cast<int32_t>(condType));
                    return false;
            }
        }

        /** @brief 压入 if 控制帧（记录条件判定结果）。 */
        void ExecIfBegin(SimCcuExecContext* ctx, bool taken)
        {
            if (ctx == nullptr) {
                return;
            }
            SimCcuIfFrame frame;
            frame.ifTaken = taken;
            ctx->ifFrames.push_back(frame);
        }

        /** @brief 将当前 if 帧切换到 else 分支。 */
        void ExecIfElse(SimCcuExecContext* ctx)
        {
            if (ctx == nullptr || ctx->ifFrames.empty()) {
                return;
            }
            ctx->ifFrames.back().inElse = true;
        }

        /** @brief 弹出 if 控制帧。 */
        void ExecIfEnd(SimCcuExecContext* ctx)
        {
            if (ctx == nullptr || ctx->ifFrames.empty()) {
                return;
            }
            ctx->ifFrames.pop_back();
        }

        /*---- 回放：任务录制（内部均做控制流抑制判断）----*/

        void ExecRecordMemCpy(
            SimCcuExecContext* ctx, uint32_t srcRank, uint64_t srcOffset, uint32_t dstRank, uint64_t dstOffset,
            uint64_t len, const char* opName)
        {
            if (ctx == nullptr) {
                return;
            }
            HCCL_VM_INFO(
                "ccu replay task: MEM_CPY {} src(rank={})@0x{:x} -> dst(rank={})@0x{:x}, len={}",
                (opName != nullptr) ? opName : "", srcRank, srcOffset, dstRank, dstOffset, len);
            SimCcuTaskRecord record;
            record.kind = HccLTaskMetaType::MEM_CPY;
            record.srcRank = srcRank;
            record.srcOffset = srcOffset;
            record.dstRank = dstRank;
            record.dstOffset = dstOffset;
            record.len = len;
            record.opName = (opName != nullptr) ? opName : "";
            AppendRecord(*ctx, record);
        }

        /** @brief 录制 REDUCE 任务（dataCount 口径为字节数）并打印。 */
        void ExecRecordReduce(
            SimCcuExecContext* ctx, uint32_t srcRank, uint64_t srcOffset, uint32_t dstRank, uint64_t dstOffset,
            uint64_t byteLen, uint8_t dataType, uint8_t reduceOp, const char* opName)
        {
            if (ctx == nullptr) {
                return;
            }
            uint32_t typeSize = 1;
            if (sim::GetDataTypeSize(static_cast<HcclDataType>(dataType), typeSize) != 0 || typeSize == 0U) {
                HCCL_VM_WARN(
                    "ccu reduce has unknown data type {}, falls back to 1 byte per element",
                    static_cast<uint32_t>(dataType));
                typeSize = 1;
            }
            HCCL_VM_INFO(
                "ccu replay task: REDUCE {} src(rank={})@0x{:x} -> dst(rank={})@0x{:x}, byteLen={}, "
                "elemCount={}, dataType={}, op={}",
                (opName != nullptr) ? opName : "", srcRank, srcOffset, dstRank, dstOffset, byteLen, byteLen / typeSize,
                static_cast<uint32_t>(dataType), static_cast<uint32_t>(reduceOp));
            SimCcuTaskRecord record;
            record.kind = HccLTaskMetaType::REDUCE;
            record.srcRank = srcRank;
            record.srcOffset = srcOffset;
            record.dstRank = dstRank;
            record.dstOffset = dstOffset;
            record.dataCount = byteLen; // dataCount 口径为字节数：对齐 level1（data_comm_op_stub.cc:297）与
                                        // checker（task_meta_translator_v3.cc:406）
            record.dataType = dataType;
            record.reduceOp = reduceOp;
            record.opName = (opName != nullptr) ? opName : "";
            AppendRecord(*ctx, record);
        }

        /** @brief 录制 NOTIFY_RECORD/NOTIFY_WAIT 任务（notifyId 配对）。 */
        void ExecRecordNotify(
            SimCcuExecContext* ctx, bool isRecord, uint32_t srcRank, uint32_t dstRank, uint64_t notifyId,
            uint8_t notifyCount, const char* opName)
        {
            if (ctx == nullptr) {
                return;
            }
            HCCL_VM_INFO(
                "ccu replay task: {} {} src(rank={}) -> dst(rank={}), notifyId=0x{:x}, count={}",
                isRecord ? "NOTIFY_RECORD" : "NOTIFY_WAIT", (opName != nullptr) ? opName : "", srcRank, dstRank,
                notifyId, static_cast<uint32_t>(notifyCount));
            SimCcuTaskRecord record;
            record.kind = isRecord ? HccLTaskMetaType::NOTIFY_RECORD : HccLTaskMetaType::NOTIFY_WAIT;
            record.srcRank = srcRank;
            record.dstRank = dstRank;
            record.notifyId = notifyId;
            record.notifyCount = notifyCount;
            record.opName = (opName != nullptr) ? opName : "";
            AppendRecord(*ctx, record);
        }

        /** @brief 事件同步按 CKE 位拆分下发：mask 逐 bit 分解、每 bit 一条任务，
         *  id 由 (设备, CKE寄存器, bit) 共同确定——对齐真机 (ckeId, mask) 资源身份与
         *  checker 的 1:1 notifyId 配对模型（多 bit wait = 多个独立位等待的 AND-join）。 */
        void ExecRecordEventBits(
            SimCcuExecContext* ctx, bool isRecord, uint64_t eventHandle, uint16_t mask, const char* opName)
        {
            if (ctx == nullptr) {
                return;
            }
            for (uint32_t bit = 0; bit < 16U; ++bit) {
                const uint16_t bitMask = static_cast<uint16_t>(1U << bit);
                if ((mask & bitMask) == 0U) {
                    continue;
                }
                ExecRecordNotify(
                    ctx, isRecord, ctx->rankId, ctx->rankId, EventBitNotifyId(ctx->rankId, eventHandle, bit),
                    static_cast<uint8_t>(bit), opName);
            }
        }

        /** @brief 通道同步按 CKE 位拆分下发（语义同 ExecRecordEventBits，id 为通道位资源）。 */
        void ExecRecordChannelBits(
            SimCcuExecContext* ctx, bool isRecord, uint32_t srcRank, uint32_t dstRank, uint32_t notifyIdx,
            uint16_t mask, const char* opName)
        {
            if (ctx == nullptr) {
                return;
            }
            for (uint32_t bit = 0; bit < 16U; ++bit) {
                const uint16_t bitMask = static_cast<uint16_t>(1U << bit);
                if ((mask & bitMask) == 0U) {
                    continue;
                }
                ExecRecordNotify(
                    ctx, isRecord, srcRank, dstRank, ChannelBitNotifyId(srcRank, dstRank, notifyIdx, bit),
                    static_cast<uint8_t>(bit), opName);
            }
        }

        /** @brief 录制 MEM_CPY，并在完成后 record 事件位（数据原语事件语义 = 完成置位）。 */
        void ExecRecordMemCpyWithEvent(
            SimCcuExecContext* ctx, uint32_t srcRank, uint64_t srcOffset, uint32_t dstRank, uint64_t dstOffset,
            uint64_t len, uint64_t eventHandle, uint16_t mask, const char* opName)
        {
            if (ctx == nullptr) {
                return;
            }
            ExecRecordMemCpy(ctx, srcRank, srcOffset, dstRank, dstOffset, len, opName);
            // 数据原语的 event 语义为完成置位（无前置等待）：搬运完成后 record mask 位——
            // 与模板用法互证（Read(e,1<<i) 产位 + EventWait(allBit) 消费位；体拷贝后显式 EventWait）。
            if (eventHandle != 0 && mask != 0) {
                ExecRecordEventBits(ctx, true, eventHandle, mask, "event-record");
            }
        }

        /** @brief 录制 REDUCE，并在完成后 record 事件位。 */
        void ExecRecordReduceWithEvent(
            SimCcuExecContext* ctx, uint32_t srcRank, uint64_t srcOffset, uint32_t dstRank, uint64_t dstOffset,
            uint64_t byteLen, uint8_t dataType, uint8_t reduceOp, uint64_t eventHandle, uint16_t mask,
            const char* opName)
        {
            if (ctx == nullptr) {
                return;
            }
            ExecRecordReduce(ctx, srcRank, srcOffset, dstRank, dstOffset, byteLen, dataType, reduceOp, opName);
            if (eventHandle != 0 && mask != 0) {
                ExecRecordEventBits(ctx, true, eventHandle, mask, "event-record");
            }
        }

        /*---- 通道信息（DB 查询封装，回放期使用）----*/

        bool GetChannelRemoteRank(uint64_t channelHandle, uint32_t& remoteRankId)
        {
            remoteRankId = 0;
            const auto optChannel = sim::runtime::Db::GetById<sim::runtime::HcclChannel>(channelHandle);
            if (!optChannel.ok()) {
                HCCL_VM_WARN("ccu channel {} not found, remote rank falls back to 0", channelHandle);
                return false;
            }
            remoteRankId = static_cast<uint32_t>(optChannel->remoteRankId);
            return true;
        }

        /**
         * @brief 连接序号：本端（本通信域）到该 peer 的 channel 中，id 更小的同 peer channel 数。
         *
         * RunnerDB 为全 rank 共享，必须按 commId 限定到本进程本通信域，否则会把其它 rank 的
         * 通道计入 ordinal：多 rank 时同一读方的不同 peer 会算出相同 ordinal，交换键互相覆盖。
         * 单连接（1 channel/peer）恒为 0；multi-jetty 时两侧按 id 升序对齐（记录为设计假设）。
         */
        uint32_t GetChannelOrdinal(uint64_t channelHandle, uint32_t remoteRankId)
        {
            const auto self = sim::runtime::Db::GetById<sim::runtime::HcclChannel>(channelHandle);
            if (!self.ok()) {
                HCCL_VM_WARN("ccu channel {} not found when computing ordinal, fallback to 0", channelHandle);
                return 0;
            }
            const uint64_t commId = self->commId;
            const auto channels = sim::runtime::Db::GetByPred<sim::runtime::HcclChannel>(HcclSim::Storage::And(
                HcclSim::Storage::Eq(&sim::runtime::HcclChannel::commId, commId),
                HcclSim::Storage::Eq(&sim::runtime::HcclChannel::remoteRankId, remoteRankId)));
            uint32_t ordinal = 0;
            if (channels.ok() && channels.value.has_value()) {
                for (const auto& ch : *channels.value) {
                    if (ch.id < channelHandle) {
                        ++ordinal;
                    }
                }
            }
            return ordinal;
        }

        /**
         * @brief 交换变量首次读取处懒解析。
         *
         * 条件：handle ∈ entry.channelVarSources 且未解析。此时本端 PreSync 已完成（模板时序
         * 先写后读），可直接有界等待对端发布值（不会双向死锁）。
         */
        bool ResolveChannelVarIfNeeded(SimCcuExecContext* ctx, uint64_t varHandle)
        {
            if (ctx == nullptr) {
                return false;
            }
            const auto srcIter = ctx->channelVarSources.find(varHandle);
            if (srcIter == ctx->channelVarSources.end() || ctx->resolvedChannelVars.count(varHandle) != 0U) {
                return false;
            }
            ctx->resolvedChannelVars.insert(varHandle); // 无论成败都只解析一次，避免重复等待
            const uint64_t channel = srcIter->second.first;
            const uint32_t varIndex = srcIter->second.second;
            uint32_t remoteRank = 0;
            if (!GetChannelRemoteRank(channel, remoteRank)) {
                return false;
            }
            SimCcuSyncKey key;
            key.opIter = sim::operation::GetCurOpIter(); // op 身份 = 当前正在处理的算子（跨 rank 一致）
            key.round = static_cast<uint32_t>(ctx->launchIdx); // 轮次 = 本读方所在 launch 序号
            key.srcRank = remoteRank;     // 写方 = 对端 peer（修复：仅靠 connOrdinal 无法区分多写方）
            key.targetRank = ctx->rankId; // 读方 = 本端
            key.connOrdinal = GetChannelOrdinal(channel, remoteRank);
            key.dieId = ctx->dieId;
            key.resType = CCU_SYNC_RES_XN;
            key.resId = varIndex;
            uint64_t resolved = 0;
            if (WaitSyncRes(key, resolved)) {
                ctx->varValues[varHandle] = resolved;
                return true;
            }
            HCCL_VM_WARN(
                "ccu channel var 0x{:x} unresolved (channel={}, varIndex={}), keep default 0", varHandle, channel,
                varIndex);
            return false;
        }

        /** @brief 轨迹后处理：填充 while/do-while 的循环配对索引。 */
        bool LinkTraceLoops(std::vector<SimCcuPrim>& trace)
        {
            std::vector<size_t> stack; // WHILE_BEGIN / DO_BEGIN 的索引栈
            for (size_t i = 0; i < trace.size(); ++i) {
                const SimCcuPrimKind kind = trace[i].kind;
                if (kind == SimCcuPrimKind::WHILE_BEGIN || kind == SimCcuPrimKind::DO_BEGIN) {
                    stack.push_back(i);
                } else if (kind == SimCcuPrimKind::WHILE_END || kind == SimCcuPrimKind::DO_END) {
                    if (stack.empty()) {
                        HCCL_VM_ERROR("ccu trace loop end without begin at index {}", i);
                        return false;
                    }
                    const size_t beginIdx = stack.back();
                    stack.pop_back();
                    trace[beginIdx].link = static_cast<uint32_t>(i); // begin -> end
                    trace[i].link = static_cast<uint32_t>(beginIdx); // end -> begin
                }
            }
            if (!stack.empty()) {
                HCCL_VM_ERROR("ccu trace has {} unclosed loop begin(s)", stack.size());
                return false;
            }
            return true;
        }

    } // namespace

    /*==================== 注册事务 ====================*/

    HcclVmResult BeginRegister(uint64_t insHandle)
    {
        SimCcuRegistryState& state = RegistryState();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.registeringIns.count(insHandle) != 0U) {
            HCCL_VM_ERROR("ccu register transaction already opened, insHandle={}", insHandle);
            return HCCL_SIM_E_PARA;
        }
        state.registeringIns.insert(insHandle);
        HCCL_VM_INFO("ccu kernel register transaction begin, insHandle={}", insHandle);
        return HCCL_SIM_SUCCESS;
    }

    /**
     * @brief 注册并就地录制 CCU kernel：注册期执行 kernelFunc 采集轨迹（此刻 kernelArg 存活），
     *        成功后登记到注册表并返回内核句柄。
     */
    /**
     * @brief 校验注册事务并提前占号（kernelIdx 与注册表 kernelHandle 同源编号，同一锁内）。
     * @return false = 无打开的注册事务。
     */
    bool ReserveKernelHandle(uint64_t insHandle, uint64_t& pendingHandle)
    {
        SimCcuRegistryState& state = RegistryState();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.registeringIns.count(insHandle) == 0U) {
            HCCL_VM_ERROR("ccu kernel register rejected: no open register transaction, insHandle={}", insHandle);
            return false;
        }
        pendingHandle = state.nextKernelHandle;
        state.nextKernelHandle += 1U;
        return true;
    }

    /** @brief 注册期就地执行 kernelFunc（异常归一为 CCU_E_INTERNAL）。 */
    CcuResult RunKernelFuncForRecord(const void* func, const char* funcName, const void* kernelArg, uint32_t argNum)
    {
        CcuResult funcRet = CcuResult::CCU_SUCCESS;
        try {
            if (argNum == 0U) {
                const auto kernelFunc = reinterpret_cast<SimCcuKernelFuncNoArg>(const_cast<void*>(func));
                funcRet = kernelFunc();
            } else {
                const auto kernelFunc = reinterpret_cast<SimCcuKernelFuncOneArg>(const_cast<void*>(func));
                funcRet = kernelFunc(const_cast<CcuKernelArg>(kernelArg));
            }
        } catch (const std::exception& e) {
            HCCL_VM_ERROR(
                "ccu kernel function threw during recording: kernel={}, what={}", (funcName != nullptr) ? funcName : "",
                e.what());
            funcRet = CcuResult::CCU_E_INTERNAL;
        } catch (...) {
            HCCL_VM_ERROR(
                "ccu kernel function threw unknown exception during recording, kernel={}",
                (funcName != nullptr) ? funcName : "");
            funcRet = CcuResult::CCU_E_INTERNAL;
        }
        return funcRet;
    }

    /**
     * @brief 闭合残留 if 并校验轨迹良构（配对/label 闭合）。
     *
     * 与真身 ccu_kernel_mgr.cc:194 一致：kernelFunc 执行完后闭合残留的未匹配 if，
     * 保证轨迹中每个 IF_BEGIN 都有 IF_END（回放器帧栈配平的前提）。
     */
    HcclVmResult ValidateRecordedTrace(SimCcuRecordContext& rctx)
    {
        rctx.active = true;
        FlushPendingIfs();
        rctx.active = false;
        HcclVmResult ret = HCCL_SIM_SUCCESS;
        if (!rctx.ifStack.empty()) {
            HCCL_VM_ERROR(
                "ccu record finished with {} unclosed if(s), top label={}", rctx.ifStack.size(),
                rctx.ifStack.back().label);
            ret = HCCL_SIM_E_INTERNAL;
        }
        if (ret == HCCL_SIM_SUCCESS && !LinkTraceLoops(rctx.trace)) {
            ret = HCCL_SIM_E_INTERNAL;
        }
        if (ret == HCCL_SIM_SUCCESS && !rctx.whileOpenLabels.empty()) {
            HCCL_VM_ERROR(
                "ccu record finished with {} unclosed while(s), top label={}", rctx.whileOpenLabels.size(),
                *rctx.whileOpenLabels.begin());
            ret = HCCL_SIM_E_INTERNAL;
        }
        if (ret == HCCL_SIM_SUCCESS && !rctx.doWhileOpenLabels.empty()) {
            HCCL_VM_ERROR(
                "ccu record finished with {} unclosed do-while(s), top label={}", rctx.doWhileOpenLabels.size(),
                *rctx.doWhileOpenLabels.begin());
            ret = HCCL_SIM_E_INTERNAL;
        }
        return ret;
    }

    /** @brief 把录制结果构建为 kernel 条目并登记到注册表（含 dladdr 排障日志）。 */
    void StoreRecordedKernel(
        uint64_t handle, uint64_t insHandle, uint32_t dieId, const char* funcName, const void* func, uint32_t argNum,
        SimCcuRecordContext& rctx)
    {
        SimCcuKernelEntry entry;
        entry.insHandle = insHandle;
        entry.dieId = dieId;
        entry.funcName = (funcName != nullptr) ? funcName : "";
        entry.func = func;
        entry.kernelArg = nullptr; // 仅供注册期使用，录制完成后立即置空防误用
        entry.argNum = argNum;
        entry.ownerPid = static_cast<uint64_t>(getpid()); // I5 进程封闭：回放期自检
        entry.trace = std::move(rctx.trace);
        entry.strings = std::move(rctx.strings);
        entry.extraBufs = std::move(rctx.extraBufs);
        entry.loopBodyEnterIdx = std::move(rctx.loopBodyEnterIdx);
        entry.loopBodyExitIdx = std::move(rctx.loopBodyExitIdx);
        entry.groups = std::move(rctx.groups);
        entry.channelVarSources = std::move(rctx.channelVarSources);

        SimCcuRegistryState& state = RegistryState();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.kernelByHandle[handle] = std::move(entry);
        // 打印 kernelFunc 与所属 SO 基址：配合分配桩 exit 日志的 caller=0xADDR，
        // 离线用 addr2line -e libhccl.so (caller - base) 反查变量分配的模板源码行。
        const void* funcAddr = state.kernelByHandle[handle].func;
        Dl_info dlInfo{};
        if (dladdr(funcAddr, &dlInfo) != 0 && dlInfo.dli_fname != nullptr) {
            HCCL_VM_INFO(
                "ccu kernel registered (recorded), kernelHandle={}, insHandle={}, dieId={}, name={}, "
                "argNum={}, primCount={}, strCount={}, funcAddr=0x{:x}, so={}@0x{:x}",
                handle, insHandle, dieId, funcName, argNum, state.kernelByHandle[handle].trace.size(),
                state.kernelByHandle[handle].strings.size(), reinterpret_cast<uint64_t>(funcAddr), dlInfo.dli_fname,
                reinterpret_cast<uint64_t>(dlInfo.dli_fbase));
        } else {
            HCCL_VM_INFO(
                "ccu kernel registered (recorded), kernelHandle={}, insHandle={}, dieId={}, name={}, "
                "argNum={}, primCount={}, strCount={}, funcAddr=0x{:x}",
                handle, insHandle, dieId, funcName, argNum, state.kernelByHandle[handle].trace.size(),
                state.kernelByHandle[handle].strings.size(), reinterpret_cast<uint64_t>(funcAddr));
        }
    }

    /**
     * @brief 注册并就地录制 CCU kernel：校验注册事务 → 占号 → 执行 kernelFunc 录制 → 校验轨迹 → 登记。
     * @return 成功返回 HCCL_SIM_SUCCESS，并输出内核句柄。
     */
    HcclVmResult RegisterKernel(
        uint64_t insHandle, uint32_t dieId, const char* funcName, const void* func, const void* kernelArg,
        uint32_t argNum, uint64_t& kernelHandle)
    {
        kernelHandle = 0;
        if (func == nullptr) {
            return HCCL_SIM_E_PTR;
        }
        // 【注册步骤 1】校验注册事务并提前占号（录制失败时该号作废/跳号，无害）。
        uint64_t pendingHandle = 0;
        if (!ReserveKernelHandle(insHandle, pendingHandle)) {
            return HCCL_SIM_E_UNAVAIL;
        }

        // 【注册步骤 2】注册期就地执行 kernelFunc 录制轨迹：此刻 kernelArg（hccl 侧 resRequest 持有的
        // shared_ptr 对象）必然存活；Launch 期该对象已析构，禁止再触碰。
        SimCcuRecordContext& rctx = g_tlsRecordCtx;
        if (rctx.active) {
            HCCL_VM_ERROR("ccu record context already active (nested register?), reject");
            return HCCL_SIM_E_INTERNAL;
        }
        rctx = SimCcuRecordContext();
        rctx.active = true;
        rctx.kernelIdx = static_cast<uint32_t>(pendingHandle & 0xFFFFU);
        const CcuResult funcRet = RunKernelFuncForRecord(func, funcName, kernelArg, argNum);
        rctx.active = false;
        if (funcRet != CcuResult::CCU_SUCCESS) {
            HCCL_VM_ERROR(
                "ccu kernel function returned error during recording, kernel={}, ret={}",
                (funcName != nullptr) ? funcName : "", static_cast<int32_t>(funcRet));
            g_tlsRecordCtx = SimCcuRecordContext();
            return HCCL_SIM_E_INTERNAL;
        }

        // 【注册步骤 3】闭合残留 if 并校验轨迹良构（while/do-while label 闭合、循环配对）。
        const HcclVmResult vret = ValidateRecordedTrace(rctx);
        if (vret != HCCL_SIM_SUCCESS) {
            g_tlsRecordCtx = SimCcuRecordContext();
            return vret;
        }

        // 【注册步骤 4】构建 kernel 条目并登记到注册表（含 dladdr 排障日志）。
        StoreRecordedKernel(pendingHandle, insHandle, dieId, funcName, func, argNum, rctx);
        g_tlsRecordCtx = SimCcuRecordContext();
        kernelHandle = pendingHandle;
        return HCCL_SIM_SUCCESS;
    }

    /** @brief 结束注册事务（从注册中集合移除 insHandle）。 */
    HcclVmResult EndRegister(uint64_t insHandle)
    {
        SimCcuRegistryState& state = RegistryState();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.registeringIns.count(insHandle) == 0U) {
            HCCL_VM_ERROR("ccu register end rejected: no open register transaction, insHandle={}", insHandle);
            return HCCL_SIM_E_UNAVAIL;
        }
        state.registeringIns.erase(insHandle);
        HCCL_VM_INFO(
            "ccu kernel register transaction end, insHandle={}, totalRegisteredKernels={}", insHandle,
            state.kernelByHandle.size());
        return HCCL_SIM_SUCCESS;
    }

    /** @brief 按内核句柄查注册表，命中则拷贝出条目。 */
    bool FindKernel(uint64_t kernelHandle, SimCcuKernelEntry& entry)
    {
        SimCcuRegistryState& state = RegistryState();
        std::lock_guard<std::mutex> lock(state.mutex);
        const auto iter = state.kernelByHandle.find(kernelHandle);
        if (iter == state.kernelByHandle.end()) {
            return false;
        }
        entry = iter->second;
        return true;
    }

    /*==================== Launch 回放 ====================*/

    /**
     * @brief 回放前置（核心步骤 0~3）：进程封闭自检 / 线程-通信域 / rank-device / 参数地址转换，
     *        并建立 TLS 回放上下文 g_tlsExecCtx。
     * @return 成功返回 HCCL_SIM_SUCCESS。
     */
    static HcclVmResult PrepareReplayContext(
        uint64_t threadHandle, uint64_t kernelHandle, const SimCcuKernelEntry& entry, const uint64_t* taskArgs,
        uint32_t argNum)
    {
        // 【核心步骤 0】进程封闭自检（I5）：轨迹句柄/录制状态只在录制进程内有意义，
        // 跨进程回放属于架构误用，fail-fast 拒绝。
        if (entry.ownerPid != static_cast<uint64_t>(getpid())) {
            HCCL_VM_ERROR(
                "ccu replay rejected: kernel recorded in pid={} but replayed in pid={}, kernel={}, "
                "kernelHandle={}",
                entry.ownerPid, static_cast<uint64_t>(getpid()), entry.funcName, kernelHandle);
            return HCCL_SIM_E_INTERNAL;
        }

        // 【核心步骤 1】线程上下文：commId/streamId（查不到仅告警，仍可回放）。
        uint64_t commId = 0;
        uint64_t streamId = 0;
        const auto optThread = sim::runtime::Db::GetById<sim::runtime::HcclThread>(threadHandle);
        if (optThread.ok()) {
            commId = optThread->commId;
            streamId = optThread->streamId;
        } else {
            HCCL_VM_WARN("ccu launch thread {} not found, continue with commId=0", threadHandle);
        }

        // 【核心步骤 2】rank/device 与 launch 序号。
        const uint32_t rankId = sim::GetCurrRankId();
        const uint64_t deviceId = sim::runtime::GetCurrDeviceId();
        const uint64_t launchIdx = g_launchIdxSeq.fetch_add(1U);

        // 【核心步骤 3】建立 TLS 回放上下文；taskArgs 逐个做设备地址->虚拟地址转换
        // （与 level2 rtCCULaunch 的 SQE 参数口径一致，标量参数不受影响）。
        // 参数个数不受 SQE 13 槽限制：hccl 模板按真实诉求传入任意个数。
        constexpr uint32_t CCU_LAUNCH_ARGS_MAX = 1024U;
        if (argNum > CCU_LAUNCH_ARGS_MAX) {
            HCCL_VM_ERROR("ccu launch argNum={} exceeds defensive limit {}", argNum, CCU_LAUNCH_ARGS_MAX);
            return HCCL_SIM_E_PARA;
        }
        SimCcuExecContext& ctx = g_tlsExecCtx;
        ctx = SimCcuExecContext();
        ctx.active = true;
        ctx.kernelHandle = kernelHandle;
        ctx.threadHandle = threadHandle;
        ctx.commId = commId;
        ctx.streamId = streamId;
        ctx.deviceId = deviceId;
        ctx.rankId = rankId;
        ctx.dieId = entry.dieId;
        ctx.launchIdx = launchIdx;
        ctx.channelVarSources = entry.channelVarSources; /**< 交换变量来源（首次读取处懒解析）。 */
        ctx.taskArgs.reserve(argNum);
        for (uint32_t i = 0; i < argNum; ++i) {
            const uint64_t rawArg = (taskArgs != nullptr) ? taskArgs[i] : 0U;
            const uint64_t virAddr = sim::TryGetVirPtrByDevPtr(rawArg);
            const uint64_t argValue = (virAddr != 0) ? virAddr : rawArg;
            ctx.taskArgs.push_back(argValue);
            if (virAddr != 0) {
                HCCL_VM_INFO(
                    "ccu launch arg converted, launchIdx={}, argId={}, devAddr=0x{:x}, virAddr=0x{:x}", launchIdx, i,
                    rawArg, virAddr);
            }
        }
        return HCCL_SIM_SUCCESS;
    }

    // LoopGroup 展开器（大块抽象，参考 ccu-sim-stub-design.md §6.4）：
    // 不复制体段逐条产 task，而是按参数解码出 clone/iter/len，每模板产 1 条大块 MEM_CPY / REDUCE task。
    // 由 ReplayTrace 的 LOOP_GROUP_BEGIN 分支调用（ctx = g_tlsExecCtx 回放上下文）。

    /** @brief 解码 LoopGroup 组级并行参数（clone 数/步进、有效成员数、克隆施加成员位）。 */
    void DecodeLoopGroupParams(
        SimCcuExecContext& ctx, const SimCcuLoopGroup& grp, uint64_t& groupCloneNum, uint64_t& cloneStride,
        uint64_t& repeatLoopIndex, uint64_t& totalLoopNum)
    {
        // 对齐 hccl CalGoSize 语义（ccu_kernel_alg_base.cc + ccu_kernel_utils.cc GetParallelParam）：
        // 尾块(loop0=residual)与整块(loop1=memSlice)共组，parallelParam 位域声明——
        //   有效成员 = 前 totalLoopNum 个（其余为占位绑定，真机不执行）；
        //   克隆（repeatNum+1 次）只施加于第 repeatLoopIndex 个成员，其余单次。
        groupCloneNum = 1;
        cloneStride = 0;
        repeatLoopIndex = 0;
        totalLoopNum = grp.members.size(); // CONFIG/未知来源：全成员有效（保持原行为）
        if (grp.src == SimCcuLoopParamSrc::CONFIG) {
            groupCloneNum = grp.cfgCloneNum;
            cloneStride = grp.cfgAddrOffset;
            return;
        }
        uint64_t parallelPacked = 0;
        uint64_t offsetPacked = 0;
        if (!ExecGetVarValue(&ctx, grp.parallelVar, parallelPacked)
            || !ExecGetVarValue(&ctx, grp.offsetVar, offsetPacked)) {
            return;
        }
        // 位域布局见 hccl ccu_kernel_utils.cc:61-76 GetParallelParam：
        //   V1: repeatNum[61:55](7b) | repeatLoopIndex[54:48](7b) | totalLoopNum[47:41](7b)
        //   V2: repeatNum[28:19](9b) | repeatLoopIndex[18:10](9b) | totalLoopNum[9:0](10b)
        // offsetParam 两版布局相同：gsaStride[52:21](32b)
        if (grp.src == SimCcuLoopParamSrc::VAR_V2) {
            const uint64_t repeatNum = (parallelPacked >> 19U) & 0x1FFULL;
            repeatLoopIndex = (parallelPacked >> 10U) & 0x1FFULL;
            totalLoopNum = parallelPacked & 0x3FFULL;
            groupCloneNum = repeatNum + 1U;
        } else { // VAR_V1
            const uint64_t repeatNum = (parallelPacked >> 55U) & 0x7FULL;
            repeatLoopIndex = (parallelPacked >> 48U) & 0x7FULL;
            totalLoopNum = (parallelPacked >> 41U) & 0x7FULL;
            groupCloneNum = repeatNum + 1U;
        }
        cloneStride = (offsetPacked >> 21U) & 0xFFFFFFFFULL;
        if (totalLoopNum == 0U || totalLoopNum > grp.members.size()) {
            HCCL_VM_WARN(
                "ccu loop-group totalLoopNum={} abnormal (members={}), fallback to all members", totalLoopNum,
                grp.members.size());
            totalLoopNum = grp.members.size();
        }
    }

    /** @brief 解码单个 LoopGroup 成员的迭代参数（iterNum / iterStride）。 */
    void DecodeLoopMemberParams(
        SimCcuExecContext& ctx, const SimCcuLoopMember& member, uint64_t& iterNum, uint64_t& iterStride)
    {
        iterNum = 1;
        iterStride = 0;
        if (member.src == SimCcuLoopParamSrc::CONFIG) {
            iterNum = member.cfgIterNum;
            iterStride = member.cfgAddrOffset;
            return;
        }
        if (member.src == SimCcuLoopParamSrc::VAR_V1) {
            uint64_t packed = 0;
            if (ExecGetVarValue(&ctx, member.paramVar, packed)) {
                iterNum = packed & 0x1FFFULL;                 // DecodeLoopParam: [12:0]=iterNum
                iterStride = (packed >> 13U) & 0xFFFFFFFFULL; // [44:13]=gsaStride
            }
            return;
        }
        uint64_t v = 0; // VAR_V2：iterNum/addrOffset 为独立变量
        if (ExecGetVarValue(&ctx, member.iterNumVar, v)) {
            iterNum = v;
        }
        if (ExecGetVarValue(&ctx, member.addrOffsetVar, v)) {
            iterStride = v;
        }
    }

    /** @brief 该原语是否为 loop 体内可出现的数据原语（远端读/写 + 本地拷贝/归约）。 */
    bool IsLoopGroupDataOp(SimCcuPrimKind kind)
    {
        return (kind >= SimCcuPrimKind::REMOTE_READ_MM && kind <= SimCcuPrimKind::REMOTE_WRITE_MR)
               || (kind >= SimCcuPrimKind::LOCAL_COPY_MM && kind <= SimCcuPrimKind::LOCAL_REDUCE_BUFS);
    }

    /** @brief 展开 loop 体内远端读原语（ReadMM/ReadMR/ReadMB）为大块任务。 */
    void EmitLoopGroupRemoteRead(
        SimCcuExecContext& ctx, const SimCcuPrim& bp, uint64_t totalLen, uint8_t dataType, uint8_t reduceOp,
        uint16_t bpMask)
    {
        uint32_t remoteRank = 0;
        uint64_t remoteOffset = 0;
        uint64_t localOffset = 0;
        switch (bp.kind) {
            case SimCcuPrimKind::REMOTE_READ_MM:
                if (GetChannelRemoteRank(bp.channel, remoteRank)
                    && ExecGetRemoteAddrOffset(&ctx, bp.remote, remoteRank, remoteOffset)
                    && ExecGetLocalAddrOffset(&ctx, bp.local, localOffset)) {
                    ExecRecordMemCpyWithEvent(
                        &ctx, remoteRank, remoteOffset, ctx.rankId, localOffset, totalLen, bp.event, bpMask,
                        "CcuReadMemToMem");
                }
                break;
            case SimCcuPrimKind::REMOTE_READ_MR:
                if (GetChannelRemoteRank(bp.channel, remoteRank)
                    && ExecGetRemoteAddrOffset(&ctx, bp.remote, remoteRank, remoteOffset)
                    && ExecGetLocalAddrOffset(&ctx, bp.local, localOffset)) {
                    ExecRecordReduceWithEvent(
                        &ctx, remoteRank, remoteOffset, ctx.rankId, localOffset, totalLen, dataType, reduceOp, bp.event,
                        bpMask, "CcuReadMemToMemReduce");
                }
                break;
            default: // REMOTE_READ_MB
                if (GetChannelRemoteRank(bp.channel, remoteRank)
                    && ExecGetRemoteAddrOffset(&ctx, bp.remote, remoteRank, remoteOffset)) {
                    const uint64_t bufOffset = ExecGetBufferOffset(&ctx, bp.local, totalLen);
                    ExecRecordMemCpyWithEvent(
                        &ctx, remoteRank, remoteOffset, ctx.rankId, bufOffset, totalLen, bp.event, bpMask,
                        "CcuReadMemToBuffer");
                }
                break;
        }
    }

    /** @brief 展开 loop 体内远端写原语（WriteMM/WriteMR/WriteMB）为大块任务。 */
    void EmitLoopGroupRemoteWrite(
        SimCcuExecContext& ctx, const SimCcuPrim& bp, uint64_t totalLen, uint8_t dataType, uint8_t reduceOp,
        uint16_t bpMask)
    {
        uint32_t remoteRank = 0;
        uint64_t remoteOffset = 0;
        uint64_t localOffset = 0;
        switch (bp.kind) {
            case SimCcuPrimKind::REMOTE_WRITE_MM:
                if (GetChannelRemoteRank(bp.channel, remoteRank)
                    && ExecGetRemoteAddrOffset(&ctx, bp.local, remoteRank, remoteOffset)
                    && ExecGetLocalAddrOffset(&ctx, bp.remote, localOffset)) {
                    ExecRecordMemCpyWithEvent(
                        &ctx, ctx.rankId, localOffset, remoteRank, remoteOffset, totalLen, bp.event, bpMask,
                        "CcuWriteMemToMem");
                }
                break;
            case SimCcuPrimKind::REMOTE_WRITE_MR:
                if (GetChannelRemoteRank(bp.channel, remoteRank)
                    && ExecGetRemoteAddrOffset(&ctx, bp.local, remoteRank, remoteOffset)
                    && ExecGetLocalAddrOffset(&ctx, bp.remote, localOffset)) {
                    ExecRecordReduceWithEvent(
                        &ctx, ctx.rankId, localOffset, remoteRank, remoteOffset, totalLen, dataType, reduceOp, bp.event,
                        bpMask, "CcuWriteMemToMemReduce");
                }
                break;
            default: // REMOTE_WRITE_BM
                if (GetChannelRemoteRank(bp.channel, remoteRank)
                    && ExecGetRemoteAddrOffset(&ctx, bp.local, remoteRank, remoteOffset)) {
                    const uint64_t bufOffset = ExecGetBufferOffset(&ctx, bp.remote, totalLen);
                    ExecRecordMemCpyWithEvent(
                        &ctx, ctx.rankId, bufOffset, remoteRank, remoteOffset, totalLen, bp.event, bpMask,
                        "CcuWriteBufferToMem");
                }
                break;
        }
    }

    /** @brief 展开 loop 体内本地拷贝原语（CopyMM/CopyMB/CopyBM）为大块任务。 */
    void EmitLoopGroupLocalCopy(SimCcuExecContext& ctx, const SimCcuPrim& bp, uint64_t totalLen, uint16_t bpMask)
    {
        switch (bp.kind) {
            case SimCcuPrimKind::LOCAL_COPY_MM: {
                uint64_t srcOffset = 0;
                uint64_t dstOffset = 0;
                if (ExecGetLocalAddrOffset(&ctx, bp.local, srcOffset)
                    && ExecGetLocalAddrOffset(&ctx, bp.dst, dstOffset)) {
                    ExecRecordMemCpyWithEvent(
                        &ctx, ctx.rankId, srcOffset, ctx.rankId, dstOffset, totalLen, bp.event, bpMask,
                        "CcuLocalCopyMemToMem");
                }
                break;
            }
            case SimCcuPrimKind::LOCAL_COPY_MB: {
                uint64_t srcOffset = 0;
                if (ExecGetLocalAddrOffset(&ctx, bp.local, srcOffset)) {
                    const uint64_t bufOffset = ExecGetBufferOffset(&ctx, bp.dst, totalLen);
                    ExecRecordMemCpyWithEvent(
                        &ctx, ctx.rankId, srcOffset, ctx.rankId, bufOffset, totalLen, bp.event, bpMask,
                        "CcuLocalCopyMemToBuffer");
                }
                break;
            }
            default: { // LOCAL_COPY_BM
                uint64_t dstOffset = 0;
                if (ExecGetLocalAddrOffset(&ctx, bp.dst, dstOffset)) {
                    const uint64_t bufOffset = ExecGetBufferOffset(&ctx, bp.local, totalLen);
                    ExecRecordMemCpyWithEvent(
                        &ctx, ctx.rankId, bufOffset, ctx.rankId, dstOffset, totalLen, bp.event, bpMask,
                        "CcuLocalCopyBufferToMem");
                }
                break;
            }
        }
    }

    /** @brief 展开 loop 体内本地归约原语（ReduceMM/ReduceBufs）为大块任务。 */
    void EmitLoopGroupLocalReduce(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& bp, uint64_t totalLen,
        uint8_t dataType, uint8_t reduceOp, uint16_t bpMask)
    {
        if (bp.kind == SimCcuPrimKind::LOCAL_REDUCE_MM) {
            uint64_t srcOffset = 0;
            uint64_t dstOffset = 0;
            if (ExecGetLocalAddrOffset(&ctx, bp.local, srcOffset) && ExecGetLocalAddrOffset(&ctx, bp.dst, dstOffset)) {
                ExecRecordReduceWithEvent(
                    &ctx, ctx.rankId, srcOffset, ctx.rankId, dstOffset, totalLen, dataType, reduceOp, bp.event, bpMask,
                    "CcuLocalMemReduce");
            }
            return;
        }
        // LOCAL_REDUCE_BUFS：count-1 条（src=buf[1..]、dst=buf[0]，首元素累积），首条带事件包裹
        const uint32_t bufCount = bp.aux;
        if (bufCount >= 2U && bp.extOff + bufCount <= entry.extraBufs.size()) {
            const uint64_t dstOffset = ExecGetBufferOffset(&ctx, entry.extraBufs[bp.extOff], totalLen);
            for (uint32_t k = 1U; k < bufCount; ++k) {
                const uint64_t srcOffset = ExecGetBufferOffset(&ctx, entry.extraBufs[bp.extOff + k], totalLen);
                ExecRecordReduceWithEvent(
                    &ctx, ctx.rankId, srcOffset, ctx.rankId, dstOffset, totalLen, dataType, reduceOp,
                    (k == 1U) ? bp.event : 0U, (k == 1U) ? bpMask : 0U, "CcuLocalBufferReduce");
            }
        }
    }

    /** @brief 按类别分派 loop 体内单条数据原语的展开。 */
    void EmitLoopGroupMemberInstr(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& bp, uint64_t totalLen,
        uint8_t dataType, uint8_t reduceOp, uint16_t bpMask)
    {
        const SimCcuPrimKind kind = bp.kind;
        if (kind >= SimCcuPrimKind::REMOTE_READ_MM && kind <= SimCcuPrimKind::REMOTE_READ_MR) {
            EmitLoopGroupRemoteRead(ctx, bp, totalLen, dataType, reduceOp, bpMask);
        } else if (kind >= SimCcuPrimKind::REMOTE_WRITE_MM && kind <= SimCcuPrimKind::REMOTE_WRITE_MR) {
            EmitLoopGroupRemoteWrite(ctx, bp, totalLen, dataType, reduceOp, bpMask);
        } else if (kind >= SimCcuPrimKind::LOCAL_COPY_MM && kind <= SimCcuPrimKind::LOCAL_COPY_BM) {
            EmitLoopGroupLocalCopy(ctx, bp, totalLen, bpMask);
        } else if (kind >= SimCcuPrimKind::LOCAL_REDUCE_MM && kind <= SimCcuPrimKind::LOCAL_REDUCE_BUFS) {
            EmitLoopGroupLocalReduce(ctx, entry, bp, totalLen, dataType, reduceOp, bpMask);
        }
    }

    /**
    /** @brief 处理 loop 体内单条指令（事件位或数据原语）；返回是否产出了内容。 */
    bool EmitLoopGroupBodyInstr(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& bp, uint32_t memberLoopIdx,
        uint64_t cloneNum, uint64_t cloneStride, uint64_t iterNum, uint64_t iterStride)
    {
        if (bp.kind == SimCcuPrimKind::EVENT_WAIT || bp.kind == SimCcuPrimKind::EVENT_RECORD) {
            const uint16_t m = static_cast<uint16_t>(bp.aux & 0xFFFFU);
            if (bp.event != 0 && m != 0) {
                ExecRecordEventBits(
                    &ctx, bp.kind == SimCcuPrimKind::EVENT_RECORD, bp.event, m,
                    (bp.kind == SimCcuPrimKind::EVENT_RECORD) ? "CcuEventRecord" : "CcuEventWait");
            }
            return true;
        }
        if (!IsLoopGroupDataOp(bp.kind)) {
            return false; // 非数据/同步原语不出现于 loop 体（LOOP_BLOCK 保证），防御跳过
        }
        uint64_t perExecLen = 0;
        if (!ExecGetVarValue(&ctx, bp.lenVar, perExecLen)) {
            HCCL_VM_WARN("ccu replay loop-group member {} instr lenVar unresolved, skip instr", memberLoopIdx);
            return false;
        }
        const uint64_t totalLen = (cloneNum - 1U) * cloneStride + (iterNum - 1U) * iterStride + perExecLen;
        const uint16_t bpMask = (bp.kind == SimCcuPrimKind::LOCAL_REDUCE_BUFS) ?
                                    static_cast<uint16_t>((bp.aux2 >> 16U) & 0xFFFFU) :
                                    static_cast<uint16_t>(bp.aux & 0xFFFFU);
        const uint8_t dataType = static_cast<uint8_t>((bp.aux2 >> 8U) & 0xFFU);
        const uint8_t reduceOp = static_cast<uint8_t>(bp.aux2 & 0xFFU);
        EmitLoopGroupMemberInstr(ctx, entry, bp, totalLen, dataType, reduceOp, bpMask);
        HCCL_VM_INFO(
            "ccu replay loop-group merged instr: member loop={}, kind={}, iterNum={}, cloneNum={}, "
            "perExecLen={}, total={}",
            memberLoopIdx, PrimKindName(bp.kind), iterNum, cloneNum, perExecLen, totalLen);
        return true;
    }

    /**
     * @brief 展开单个 LoopGroup 成员：扫描其体段，逐条数据原语产出大块任务。
     *
     * 体段扫描：模板内全部数据原语逐条大块展开（Step1 拷贝 → Step2 归约 → Step3 写出均保留，
     * reduce 语义不丢）；每条按自身 lenVar 求值 perExecLen（Step3 的 loopLenExp 与 Step1/2 的 loopLen 可能不同）：
     * total = (cloneNum-1)*cloneStride + (iterNum-1)*iterStride + perExecLen。
     */
    void EmitLoopGroupMember(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuLoopGroup& grp, const SimCcuPrim& groupPrim,
        const SimCcuLoopMember& member, size_t memberIdx, uint64_t groupCloneNum, uint64_t cloneStride,
        uint64_t repeatLoopIndex, HcclVmResult& replayErr)
    {
        const std::vector<SimCcuPrim>& trace = entry.trace;
        // 【成员步骤 1】解码迭代参数 + 计算该成员的克隆次数（VAR 组仅克隆 repeatLoopIndex 成员）。
        uint64_t iterNum = 1;
        uint64_t iterStride = 0;
        DecodeLoopMemberParams(ctx, member, iterNum, iterStride);
        // 克隆施加位：VAR 组只克隆 repeatLoopIndex 成员（尾块/整块单列）；CONFIG 组维持"克隆施加全部成员"的原行为
        const uint64_t cloneNum = (grp.src == SimCcuLoopParamSrc::CONFIG) ?
                                      groupCloneNum :
                                      ((memberIdx == repeatLoopIndex) ? groupCloneNum : 1U);
        if (iterNum == 0U || cloneNum == 0U) {
            return; // 空成员
        }
        // 【成员步骤 2】校验体段边界（loopId 越界记错并跳过）。
        if (member.loopIdx >= entry.loopBodyEnterIdx.size() || member.loopIdx >= entry.loopBodyExitIdx.size()) {
            HCCL_VM_ERROR(
                "ccu replay failed: loop member bad loopIdx, kind={}, dst=0x{:x} src=0x{:x}",
                static_cast<int32_t>(groupPrim.kind), groupPrim.dst, groupPrim.src);
            if (replayErr == HCCL_SIM_SUCCESS) {
                replayErr = HCCL_SIM_E_INTERNAL;
            }
            return;
        }
        const uint32_t bodyStart = entry.loopBodyEnterIdx[member.loopIdx] + 1U;
        const uint32_t bodyEnd = entry.loopBodyExitIdx[member.loopIdx];
        // 【成员步骤 3】扫描体段，逐条数据/同步原语展开为大块任务；无内容则告警。
        bool bodyHasInstr = false;
        for (uint32_t j = bodyStart; j < bodyEnd && j < trace.size(); ++j) {
            if (EmitLoopGroupBodyInstr(
                    ctx, entry, trace[j], member.loopIdx, cloneNum, cloneStride, iterNum, iterStride)) {
                bodyHasInstr = true;
            }
        }
        if (!bodyHasInstr) {
            HCCL_VM_WARN("ccu replay loop-group member {} has no data/sync op in body, skip", member.loopIdx);
        }
    }

    /** @brief LoopGroup 展开入口：查组、外层 if 抑制判断、遍历成员并展开为大块任务。 */
    static void EmitLoopGroupTasks(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        // 【展开步骤 1】查组 + 外层 if 抑制判断（组越界/被抑制直接返回）。
        const uint32_t groupIdx = static_cast<uint32_t>(prim.dst);
        if (groupIdx >= entry.groups.size()) {
            HCCL_VM_ERROR(
                "ccu replay failed: loop group bad index, kind={}, dst=0x{:x} src=0x{:x}",
                static_cast<int32_t>(prim.kind), prim.dst, prim.src);
            if (replayErr == HCCL_SIM_SUCCESS) {
                replayErr = HCCL_SIM_E_INTERNAL;
            }
            return;
        }
        if (!recordable) {
            return; // 外层 if 抑制时整组跳过
        }
        const SimCcuLoopGroup& grp = entry.groups[groupIdx];
        HCCL_VM_INFO(
            "ccu replay instr[{}]: loopGroupBegin groupIdx={}, members={}", instrIdx, groupIdx, grp.members.size());
        // 【展开步骤 2】解码组级并行参数（clone 数/步进、有效成员数、克隆施加成员位）。
        uint64_t groupCloneNum = 1;
        uint64_t cloneStride = 0;
        uint64_t repeatLoopIndex = 0;
        uint64_t totalLoopNum = grp.members.size();
        DecodeLoopGroupParams(ctx, grp, groupCloneNum, cloneStride, repeatLoopIndex, totalLoopNum);
        // 【展开步骤 3】遍历成员并展开为大块任务（占位成员跳过）。
        for (size_t memberIdx = 0; memberIdx < grp.members.size(); ++memberIdx) {
            if (memberIdx >= totalLoopNum) {
                // 占位成员（如纯尾块场景下 loop1 的整块绑定）：真机不执行，跳过
                HCCL_VM_INFO(
                    "ccu replay loop-group member {} is placeholder (totalLoopNum={}), skip", memberIdx, totalLoopNum);
                continue;
            }
            EmitLoopGroupMember(
                ctx, entry, grp, prim, grp.members[memberIdx], memberIdx, groupCloneNum, cloneStride, repeatLoopIndex,
                replayErr);
        }
    }

    // 【核心步骤 4/5】回放轨迹 + 收尾（读 g_tlsExecCtx，由 PrepareReplayContext 建立）。
    /** @brief 回放错误记录：打印并（首次）置 replayErr = HCCL_SIM_E_INTERNAL。 */
    void MarkReplayErr(HcclVmResult& replayErr, const char* why, const SimCcuPrim& prim)
    {
        HCCL_VM_ERROR(
            "ccu replay failed: {}, kind={}, dst=0x{:x} src=0x{:x}", why, static_cast<int32_t>(prim.kind), prim.dst,
            prim.src);
        if (replayErr == HCCL_SIM_SUCCESS) {
            replayErr = HCCL_SIM_E_INTERNAL;
        }
    }

    /** @brief 叶子原语展开：VAR_ASSIGN_IMM。 */
    static bool ExecVarAssignImmPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        (void)ExecSetVarValue(&ctx, prim.dst, prim.src);
        if (recordable) {
            HCCL_VM_INFO("ccu replay var: 0x{:x} <= imm {} (0x{:x})", prim.dst, prim.src, prim.src);
        }
        return true;
    }

    /** @brief 叶子原语展开：VAR_ASSIGN_VAR。 */
    static bool ExecVarAssignVarPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        uint64_t value = 0;
        if (recordable && ExecGetVarValue(&ctx, prim.src, value)) {
            (void)ExecSetVarValue(&ctx, prim.dst, value);
            HCCL_VM_INFO("ccu replay var: 0x{:x} <= var 0x{:x} = {} (0x{:x})", prim.dst, prim.src, value, value);
        }
        return true;
    }

    /** @brief 叶子原语展开：VAR_OP_VAR。 */
    static bool ExecVarOpVarPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        uint64_t lhs = 0;
        uint64_t rhs = 0;
        if (recordable && ExecGetVarValue(&ctx, prim.varA, lhs) && ExecGetVarValue(&ctx, prim.varB, rhs)) {
            const uint64_t result = ApplyVarOpValue(lhs, rhs, static_cast<SimCcuVarOp>(prim.op));
            (void)ExecSetVarValue(&ctx, prim.dst, result);
            HCCL_VM_INFO(
                "ccu replay var: 0x{:x} = 0x{:x} op 0x{:x} ({} op {}) = {} (0x{:x})", prim.dst, prim.varA, prim.varB,
                lhs, rhs, result, result);
        }
        return true;
    }

    /** @brief 叶子原语展开：VAR_OP_IMM。 */
    static bool ExecVarOpImmPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        uint64_t lhs = 0;
        if (recordable && ExecGetVarValue(&ctx, prim.varA, lhs)) {
            const uint64_t result = ApplyVarOpValue(lhs, prim.imm, static_cast<SimCcuVarOp>(prim.op));
            (void)ExecSetVarValue(&ctx, prim.dst, result);
            HCCL_VM_INFO(
                "ccu replay var: 0x{:x} = 0x{:x} op imm ({} op {}) = {} (0x{:x})", prim.dst, prim.varA, lhs, prim.imm,
                result, result);
        }
        return true;
    }

    /** @brief 叶子原语展开：VAR_NOT。 */
    static bool ExecVarNotPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        uint64_t lhs = 0;
        if (recordable && ExecGetVarValue(&ctx, prim.src, lhs)) {
            const uint64_t result = ~lhs;
            (void)ExecSetVarValue(&ctx, prim.dst, result);
            HCCL_VM_INFO(
                "ccu replay var: 0x{:x} = ~0x{:x} (~{}) = {} (0x{:x})", prim.dst, prim.src, lhs, result, result);
        }
        return true;
    }

    /** @brief 叶子原语展开：ADDR_ASSIGN_IMM。 */
    static bool ExecAddrAssignImmPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        (void)ExecSetAddrBase(&ctx, prim.dst, prim.src);
        if (recordable) {
            HCCL_VM_INFO("ccu replay addr: 0x{:x} <= imm {} (0x{:x})", prim.dst, prim.src, prim.src);
        }
        return true;
    }

    /** @brief 叶子原语展开：ADDR_ASSIGN_ADDR。 */
    static bool ExecAddrAssignAddrPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        uint64_t base = 0;
        if (recordable && ExecGetAddrBase(&ctx, prim.src, base)) {
            (void)ExecSetAddrBase(&ctx, prim.dst, base);
            HCCL_VM_INFO("ccu replay addr: 0x{:x} <= addr 0x{:x} = {} (0x{:x})", prim.dst, prim.src, base, base);
        }
        return true;
    }

    /** @brief 叶子原语展开：ADDR_ASSIGN_VAR。 */
    static bool ExecAddrAssignVarPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        uint64_t value = 0;
        if (recordable && ExecGetVarValue(&ctx, prim.src, value)) {
            (void)ExecSetAddrBase(&ctx, prim.dst, value);
            HCCL_VM_INFO("ccu replay addr: 0x{:x} <= var 0x{:x} = {} (0x{:x})", prim.dst, prim.src, value, value);
        }
        return true;
    }

    /** @brief 叶子原语展开：ADDR_OP_VAR。 */
    static bool ExecAddrOpVarPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        uint64_t base = 0;
        uint64_t rhs = 0;
        if (recordable && ExecGetAddrBase(&ctx, prim.src, base) && ExecGetVarValue(&ctx, prim.varB, rhs)) {
            const uint64_t result = ApplyVarOpValue(base, rhs, static_cast<SimCcuVarOp>(prim.op));
            (void)ExecSetAddrBase(&ctx, prim.dst, result);
            HCCL_VM_INFO(
                "ccu replay addr: 0x{:x} = 0x{:x} op var 0x{:x} ({} op {}) = {} (0x{:x})", prim.dst, prim.src,
                prim.varB, base, rhs, result, result);
        }
        return true;
    }

    /** @brief 叶子原语展开：ADDR_OP_ADDR。 */
    static bool ExecAddrOpAddrPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        uint64_t lhs = 0;
        uint64_t rhs = 0;
        if (recordable && ExecGetAddrBase(&ctx, prim.src, lhs) && ExecGetAddrBase(&ctx, prim.varB, rhs)) {
            const uint64_t result = ApplyVarOpValue(lhs, rhs, static_cast<SimCcuVarOp>(prim.op));
            (void)ExecSetAddrBase(&ctx, prim.dst, result);
            HCCL_VM_INFO(
                "ccu replay addr: 0x{:x} = 0x{:x} op 0x{:x} ({} op {}) = {} (0x{:x})", prim.dst, prim.src, prim.varB,
                lhs, rhs, result, result);
        }
        return true;
    }

    /** @brief 叶子原语展开：ADDR_OP_IMM。 */
    static bool ExecAddrOpImmPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        uint64_t base = 0;
        if (recordable && ExecGetAddrBase(&ctx, prim.src, base)) {
            const uint64_t result = ApplyVarOpValue(base, prim.imm, static_cast<SimCcuVarOp>(prim.op));
            (void)ExecSetAddrBase(&ctx, prim.dst, result);
            HCCL_VM_INFO(
                "ccu replay addr: 0x{:x} = 0x{:x} op imm ({} op {}) = {} (0x{:x})", prim.dst, prim.src, base, prim.imm,
                result, result);
        }
        return true;
    }

    /** @brief 叶子原语展开：ADDR_ADDASSIGN_VAR。 */
    static bool ExecAddrAddassignVarPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        uint64_t base = 0;
        uint64_t value = 0;
        if (recordable && ExecGetAddrBase(&ctx, prim.dst, base) && ExecGetVarValue(&ctx, prim.src, value)) {
            const uint64_t result = base + value;
            (void)ExecSetAddrBase(&ctx, prim.dst, result);
            HCCL_VM_INFO(
                "ccu replay addr: 0x{:x} += 0x{:x} ({} + {}) = {} (0x{:x})", prim.dst, prim.src, base, value, result,
                result);
        }
        return true;
    }

    /** @brief 叶子原语展开：LOAD_ARG。 */
    static bool ExecLoadArgPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        (void)entry;
        (void)instrIdx;
        uint64_t value = 0;
        if (recordable) {
            if (!ExecGetTaskArg(&ctx, aux, value)) {
                MarkReplayErr(replayErr, "load arg out of range", prim);
            } else {
                ctx.varValues[prim.dst] = value;
                HCCL_VM_INFO("ccu replay var: 0x{:x} <= taskArgs[{}] = {} (0x{:x})", prim.dst, aux, value, value);
            }
        }
        return true;
    }

    /** @brief 叶子原语展开：LOAD_MEM / LOAD_MEM_VAR。 */
    static bool ExecLoadMemPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        (void)entry;
        (void)recordable;
        (void)replayErr;
        // best-effort：CPU 模拟不读设备内存内容，变量保持当前值（默认 0）。
        HCCL_VM_INFO(
            "ccu replay instr[{}]: {} var=0x{:x}, src=0x{:x}, num={} (keep value)", instrIdx, PrimKindName(prim.kind),
            prim.dst, prim.src, aux);
        return true;
    }

    /** @brief 叶子原语展开：STORE_MEM / STORE_MEM_VAR。 */
    static bool ExecStoreMemPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        (void)entry;
        (void)recordable;
        (void)replayErr;
        // 存变量到内存：CPU 模拟无副作用。
        HCCL_VM_INFO(
            "ccu replay instr[{}]: {} dst=0x{:x}, var=0x{:x}, num={} (no side effect)", instrIdx,
            PrimKindName(prim.kind), prim.dst, prim.src, aux);
        return true;
    }

    /** @brief 叶子原语展开：LOCAL_ADDR_ALLOC。 */
    static bool ExecLocalAddrAllocPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        (void)entry;
        (void)replayErr;
        if (recordable) {
            ctx.localAddrParts[prim.dst] = std::make_pair(prim.addr, prim.token);
            HCCL_VM_INFO(
                "ccu replay instr[{}]: localAddrAlloc local=0x{:x} (addr=0x{:x}, token=0x{:x})", instrIdx, prim.dst,
                prim.addr, prim.token);
        }
        return true;
    }

    /** @brief 叶子原语展开：REMOTE_ADDR_ALLOC。 */
    static bool ExecRemoteAddrAllocPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        (void)entry;
        (void)replayErr;
        if (recordable) {
            ctx.remoteAddrParts[prim.dst] = std::make_pair(prim.addr, prim.token);
            HCCL_VM_INFO(
                "ccu replay instr[{}]: remoteAddrAlloc remote=0x{:x} (addr=0x{:x}, token=0x{:x})", instrIdx, prim.dst,
                prim.addr, prim.token);
        }
        return true;
    }

    /** @brief 叶子原语展开：EVENT_RECORD。 */
    static bool ExecEventRecordPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        const uint16_t mask = static_cast<uint16_t>(aux & 0xFFFFU);
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        if (recordable && prim.event != 0 && mask != 0) {
            ExecRecordEventBits(&ctx, true, prim.event, mask, "CcuEventRecord");
        }
        return true;
    }

    /** @brief 叶子原语展开：EVENT_WAIT。 */
    static bool ExecEventWaitPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        const uint16_t mask = static_cast<uint16_t>(aux & 0xFFFFU);
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        if (recordable && prim.event != 0 && mask != 0) {
            ExecRecordEventBits(&ctx, false, prim.event, mask, "CcuEventWait");
        }
        return true;
    }

    /** @brief 叶子原语展开：NOTIFY_RECORD。 */
    static bool ExecNotifyRecordPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        if (!recordable) {
            return true;
        }
        uint32_t remoteRank = 0;
        if (GetChannelRemoteRank(prim.channel, remoteRank)) {
            // 通道位资源：mask 存于 aux2（aux 为 notifyIdx/CKE 槽位），id 由 (src,dst,槽位,bit) 共同确定
            const uint16_t chanMask = static_cast<uint16_t>(prim.aux2 & 0xFFFFU);
            ExecRecordChannelBits(&ctx, true, ctx.rankId, remoteRank, aux, chanMask, "CcuNotifyRecord");
        }
        return true;
    }

    /** @brief 叶子原语展开：NOTIFY_WAIT。 */
    static bool ExecNotifyWaitPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        if (!recordable) {
            return true;
        }
        uint32_t remoteRank = 0;
        if (GetChannelRemoteRank(prim.channel, remoteRank)) {
            // src=远端（信号发送方）、dst=本端（等待方），对齐 level1 口径（data_comm_op_stub.cc:891-892）
            const uint16_t chanMask = static_cast<uint16_t>(prim.aux2 & 0xFFFFU);
            ExecRecordChannelBits(&ctx, false, remoteRank, ctx.rankId, aux, chanMask, "CcuNotifyWait");
        }
        return true;
    }

    /** @brief 叶子原语展开：WRITE_VAR_NOTIFY。 */
    static bool ExecWriteVarNotifyPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        if (!recordable) {
            return true;
        }
        uint32_t remoteRank = 0;
        if (GetChannelRemoteRank(prim.channel, remoteRank)) {
            const uint16_t chanMask = static_cast<uint16_t>(prim.aux2 & 0xFFFFU);
            ExecRecordChannelBits(&ctx, true, ctx.rankId, remoteRank, aux, chanMask, "CcuWriteVariableWithNotify");
            // 交换值发布：本端变量值 -> 对端 XN[remoteVarIdx]（部分交换，per-channel 键）。
            // 模板时序保证本端写先于对端读，故此处发布即可解除读方等待。
            uint64_t varValue = 0;
            if (ExecGetVarValue(&ctx, prim.src, varValue)) {
                SimCcuSyncKey key;
                key.opIter = sim::operation::GetCurOpIter(); // op 身份 = 当前正在处理的算子（跨 rank 一致）
                key.round = static_cast<uint32_t>(ctx.launchIdx); // 轮次 = 本 launch 序号
                key.srcRank = ctx.rankId;                         // 写方 = 本端
                key.targetRank = remoteRank;
                key.connOrdinal = GetChannelOrdinal(prim.channel, remoteRank);
                key.dieId = ctx.dieId; // 单 die 对称假设（HcclChannel 无 remoteDieId 字段）
                key.resType = CCU_SYNC_RES_XN;
                key.resId = prim.extOff; // remoteVarIdx
                PublishSyncRes(key, varValue, ctx.rankId);
            }
        }
        return true;
    }

    /** @brief 叶子原语展开：LOCAL_NOTIFY_RECORD。 */
    static bool ExecLocalNotifyRecordPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const std::vector<std::string>& strings = entry.strings;
        const char* str = (prim.strIdx < strings.size()) ? strings[prim.strIdx].c_str() : "";
        const uint32_t aux = prim.aux;
        const uint16_t mask = static_cast<uint16_t>(aux & 0xFFFFU);
        (void)entry;
        (void)replayErr;
        if (recordable) {
            HCCL_VM_INFO("ccu replay instr[{}]: localNotifyRecord tag={}", instrIdx, str);
            ExecRecordNotify(
                &ctx, true, ctx.rankId, ctx.rankId, HashCtxTag(str), static_cast<uint8_t>(mask & 0xFFU),
                "CcuLocalNotifyRecord");
        }
        return true;
    }

    /** @brief 叶子原语展开：LOCAL_NOTIFY_WAIT。 */
    static bool ExecLocalNotifyWaitPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const std::vector<std::string>& strings = entry.strings;
        const char* str = (prim.strIdx < strings.size()) ? strings[prim.strIdx].c_str() : "";
        const uint32_t aux = prim.aux;
        const uint16_t mask = static_cast<uint16_t>(aux & 0xFFFFU);
        (void)entry;
        (void)replayErr;
        if (recordable) {
            HCCL_VM_INFO("ccu replay instr[{}]: localNotifyWait tag={}", instrIdx, str);
            ExecRecordNotify(
                &ctx, false, ctx.rankId, ctx.rankId, HashCtxTag(str), static_cast<uint8_t>(mask & 0xFFU),
                "CcuLocalNotifyWait");
        }
        return true;
    }

    /** @brief 叶子原语展开：LOCAL_COPY_MM。 */
    static bool ExecLocalCopyMmPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        const uint16_t mask = static_cast<uint16_t>(aux & 0xFFFFU);
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        if (!recordable) {
            return true;
        }
        uint64_t srcOffset = 0;
        uint64_t dstOffset = 0;
        uint64_t lenValue = 0;
        if (ExecGetLocalAddrOffset(&ctx, prim.local, srcOffset) && ExecGetLocalAddrOffset(&ctx, prim.dst, dstOffset)
            && ExecGetVarValue(&ctx, prim.lenVar, lenValue)) {
            ExecRecordMemCpyWithEvent(
                &ctx, ctx.rankId, srcOffset, ctx.rankId, dstOffset, lenValue, prim.event, mask, "CcuLocalCopyMemToMem");
        }
        return true;
    }

    /** @brief 叶子原语展开：LOCAL_COPY_MB。 */
    static bool ExecLocalCopyMbPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        const uint16_t mask = static_cast<uint16_t>(aux & 0xFFFFU);
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        if (!recordable) {
            return true;
        }
        uint64_t srcOffset = 0;
        uint64_t lenValue = 0;
        if (ExecGetLocalAddrOffset(&ctx, prim.local, srcOffset) && ExecGetVarValue(&ctx, prim.lenVar, lenValue)) {
            ExecRecordMemCpyWithEvent(
                &ctx, ctx.rankId, srcOffset, ctx.rankId, ExecGetBufferOffset(&ctx, prim.dst, lenValue), lenValue,
                prim.event, mask, "CcuLocalCopyMemToBuffer");
        }
        return true;
    }

    /** @brief 叶子原语展开：LOCAL_COPY_BM。 */
    static bool ExecLocalCopyBmPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        const uint16_t mask = static_cast<uint16_t>(aux & 0xFFFFU);
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        if (!recordable) {
            return true;
        }
        uint64_t dstOffset = 0;
        uint64_t lenValue = 0;
        if (ExecGetLocalAddrOffset(&ctx, prim.dst, dstOffset) && ExecGetVarValue(&ctx, prim.lenVar, lenValue)) {
            ExecRecordMemCpyWithEvent(
                &ctx, ctx.rankId, ExecGetBufferOffset(&ctx, prim.local, lenValue), ctx.rankId, dstOffset, lenValue,
                prim.event, mask, "CcuLocalCopyBufferToMem");
        }
        return true;
    }

    /** @brief 叶子原语展开：LOCAL_REDUCE_MM。 */
    static bool ExecLocalReduceMmPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        const uint32_t aux2 = prim.aux2;
        const uint8_t dataType = static_cast<uint8_t>((aux2 >> 8U) & 0xFFU);
        const uint8_t reduceOp = static_cast<uint8_t>(aux2 & 0xFFU);
        const uint16_t mask = static_cast<uint16_t>(aux & 0xFFFFU);
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        if (!recordable) {
            return true;
        }
        uint64_t srcOffset = 0;
        uint64_t dstOffset = 0;
        uint64_t lenValue = 0;
        if (ExecGetLocalAddrOffset(&ctx, prim.local, srcOffset) && ExecGetLocalAddrOffset(&ctx, prim.dst, dstOffset)
            && ExecGetVarValue(&ctx, prim.lenVar, lenValue)) {
            ExecRecordReduceWithEvent(
                &ctx, ctx.rankId, srcOffset, ctx.rankId, dstOffset, lenValue, dataType, reduceOp, prim.event, mask,
                "CcuLocalMemReduce");
        }
        return true;
    }

    /** @brief 叶子原语展开：LOCAL_REDUCE_BUFS。 */
    static bool ExecLocalReduceBufsPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        const uint32_t aux2 = prim.aux2;
        const uint8_t dataType = static_cast<uint8_t>((aux2 >> 8U) & 0xFFU);
        const uint8_t reduceOp = static_cast<uint8_t>(aux2 & 0xFFU);
        (void)instrIdx;
        if (!recordable) {
            return true;
        }
        // 多缓冲归约：**首元素为累积目标 dst**，其余为 src。
        // 真身 CcuRepBufReduce 把 mem[0..count-1].Id() 原序写入指令 msId[]，
        // 语义为 msId[0]=dst、msId[1..]=src；模板 LocalReduce(&ccuBuf[base], size, ...)
        // 亦以首元素累积、随后从首元素写回输出。适配成单 (src,dst) 任务时须按其真实方向展开。
        // 注意：本条目 aux=count / prim.lenVar=lenVar / prim.event=event / ((aux2 >> 16U) & 0xFFFFU)=mask（aux 已被
        // count 占用）。
        const uint32_t count = aux;
        const uint16_t bufMask = static_cast<uint16_t>(((aux2 >> 16U) & 0xFFFFU));
        if (count == 0U || prim.extOff + count > entry.extraBufs.size()) {
            MarkReplayErr(replayErr, "local buffer reduce bad ext range", prim);
        }
        uint64_t lenValue = 0;
        if (!ExecGetVarValue(&ctx, prim.lenVar, lenValue)) {
        }
        const uint64_t dstOffset = ExecGetBufferOffset(&ctx, entry.extraBufs[prim.extOff], lenValue);
        for (uint32_t k = 1U; k < count; ++k) {
            const uint64_t srcOffset = ExecGetBufferOffset(&ctx, entry.extraBufs[prim.extOff + k], lenValue);
            ExecRecordReduceWithEvent(
                &ctx, ctx.rankId, srcOffset, ctx.rankId, dstOffset, lenValue, dataType, reduceOp,
                (k == 1U) ? prim.event : 0U, (k == 1U) ? bufMask : 0U, "CcuLocalBufferReduce");
        }
        return true;
    }

    /** @brief 叶子原语展开：REMOTE_READ_MM。 */
    static bool ExecRemoteReadMmPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        const uint16_t mask = static_cast<uint16_t>(aux & 0xFFFFU);
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        if (!recordable) {
            return true;
        }
        uint32_t remoteRank = 0;
        uint64_t remoteOffset = 0;
        uint64_t localOffset = 0;
        uint64_t lenValue = 0;
        if (GetChannelRemoteRank(prim.channel, remoteRank)
            && ExecGetRemoteAddrOffset(&ctx, prim.remote, remoteRank, remoteOffset)
            && ExecGetLocalAddrOffset(&ctx, prim.local, localOffset) && ExecGetVarValue(&ctx, prim.lenVar, lenValue)) {
            ExecRecordMemCpyWithEvent(
                &ctx, remoteRank, remoteOffset, ctx.rankId, localOffset, lenValue, prim.event, mask, "CcuReadMemToMem");
        }
        return true;
    }

    /** @brief 叶子原语展开：REMOTE_READ_MB。 */
    static bool ExecRemoteReadMbPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        const uint16_t mask = static_cast<uint16_t>(aux & 0xFFFFU);
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        if (!recordable) {
            return true;
        }
        uint32_t remoteRank = 0;
        uint64_t remoteOffset = 0;
        uint64_t lenValue = 0;
        if (GetChannelRemoteRank(prim.channel, remoteRank)
            && ExecGetRemoteAddrOffset(&ctx, prim.remote, remoteRank, remoteOffset)
            && ExecGetVarValue(&ctx, prim.lenVar, lenValue)) {
            ExecRecordMemCpyWithEvent(
                &ctx, remoteRank, remoteOffset, ctx.rankId, ExecGetBufferOffset(&ctx, prim.local, lenValue), lenValue,
                prim.event, mask, "CcuReadMemToBuffer");
        }
        return true;
    }

    /** @brief 叶子原语展开：REMOTE_READ_MR。 */
    static bool ExecRemoteReadMrPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        const uint32_t aux2 = prim.aux2;
        const uint8_t dataType = static_cast<uint8_t>((aux2 >> 8U) & 0xFFU);
        const uint8_t reduceOp = static_cast<uint8_t>(aux2 & 0xFFU);
        const uint16_t mask = static_cast<uint16_t>(aux & 0xFFFFU);
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        if (!recordable) {
            return true;
        }
        uint32_t remoteRank = 0;
        uint64_t remoteOffset = 0;
        uint64_t localOffset = 0;
        uint64_t lenValue = 0;
        if (GetChannelRemoteRank(prim.channel, remoteRank)
            && ExecGetRemoteAddrOffset(&ctx, prim.remote, remoteRank, remoteOffset)
            && ExecGetLocalAddrOffset(&ctx, prim.local, localOffset) && ExecGetVarValue(&ctx, prim.lenVar, lenValue)) {
            ExecRecordReduceWithEvent(
                &ctx, remoteRank, remoteOffset, ctx.rankId, localOffset, lenValue, dataType, reduceOp, prim.event, mask,
                "CcuReadMemToMemReduce");
        }
        return true;
    }

    /** @brief 叶子原语展开：REMOTE_WRITE_MM。 */
    static bool ExecRemoteWriteMmPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        const uint16_t mask = static_cast<uint16_t>(aux & 0xFFFFU);
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        if (!recordable) {
            return true;
        }
        uint32_t remoteRank = 0;
        uint64_t remoteOffset = 0;
        uint64_t localOffset = 0;
        uint64_t lenValue = 0;
        if (GetChannelRemoteRank(prim.channel, remoteRank)
            && ExecGetRemoteAddrOffset(&ctx, prim.local, remoteRank, remoteOffset)
            && ExecGetLocalAddrOffset(&ctx, prim.remote, localOffset) && ExecGetVarValue(&ctx, prim.lenVar, lenValue)) {
            ExecRecordMemCpyWithEvent(
                &ctx, ctx.rankId, localOffset, remoteRank, remoteOffset, lenValue, prim.event, mask,
                "CcuWriteMemToMem");
        }
        return true;
    }

    /** @brief 叶子原语展开：REMOTE_WRITE_BM。 */
    static bool ExecRemoteWriteBmPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        const uint16_t mask = static_cast<uint16_t>(aux & 0xFFFFU);
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        if (!recordable) {
            return true;
        }
        uint32_t remoteRank = 0;
        uint64_t remoteOffset = 0;
        uint64_t lenValue = 0;
        if (GetChannelRemoteRank(prim.channel, remoteRank)
            && ExecGetRemoteAddrOffset(&ctx, prim.local, remoteRank, remoteOffset)
            && ExecGetVarValue(&ctx, prim.lenVar, lenValue)) {
            ExecRecordMemCpyWithEvent(
                &ctx, ctx.rankId, ExecGetBufferOffset(&ctx, prim.remote, lenValue), remoteRank, remoteOffset, lenValue,
                prim.event, mask, "CcuWriteBufferToMem");
        }
        return true;
    }

    /** @brief 叶子原语展开：REMOTE_WRITE_MR。 */
    static bool ExecRemoteWriteMrPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        const uint32_t aux = prim.aux;
        const uint32_t aux2 = prim.aux2;
        const uint8_t dataType = static_cast<uint8_t>((aux2 >> 8U) & 0xFFU);
        const uint8_t reduceOp = static_cast<uint8_t>(aux2 & 0xFFU);
        const uint16_t mask = static_cast<uint16_t>(aux & 0xFFFFU);
        (void)entry;
        (void)instrIdx;
        (void)replayErr;
        if (!recordable) {
            return true;
        }
        uint32_t remoteRank = 0;
        uint64_t remoteOffset = 0;
        uint64_t localOffset = 0;
        uint64_t lenValue = 0;
        if (GetChannelRemoteRank(prim.channel, remoteRank)
            && ExecGetRemoteAddrOffset(&ctx, prim.local, remoteRank, remoteOffset)
            && ExecGetLocalAddrOffset(&ctx, prim.remote, localOffset) && ExecGetVarValue(&ctx, prim.lenVar, lenValue)) {
            ExecRecordReduceWithEvent(
                &ctx, ctx.rankId, localOffset, remoteRank, remoteOffset, lenValue, dataType, reduceOp, prim.event, mask,
                "CcuWriteMemToMemReduce");
        }
        return true;
    }

    /** @brief 叶子原语处理函数签名（与 Exec<Kind>Prim 一致）。 */
    using LeafPrimFn
        = bool (*)(SimCcuExecContext&, const SimCcuKernelEntry&, const SimCcuPrim&, size_t, bool, HcclVmResult&);

    /** @brief 叶子原语 种类 -> 处理函数 分发表。 */
    const std::pair<SimCcuPrimKind, LeafPrimFn> kLeafPrimHandlers[] = {
        {SimCcuPrimKind::VAR_ASSIGN_IMM, &ExecVarAssignImmPrim},
        {SimCcuPrimKind::VAR_ASSIGN_VAR, &ExecVarAssignVarPrim},
        {SimCcuPrimKind::VAR_OP_VAR, &ExecVarOpVarPrim},
        {SimCcuPrimKind::VAR_OP_IMM, &ExecVarOpImmPrim},
        {SimCcuPrimKind::VAR_NOT, &ExecVarNotPrim},
        {SimCcuPrimKind::ADDR_ASSIGN_IMM, &ExecAddrAssignImmPrim},
        {SimCcuPrimKind::ADDR_ASSIGN_ADDR, &ExecAddrAssignAddrPrim},
        {SimCcuPrimKind::ADDR_ASSIGN_VAR, &ExecAddrAssignVarPrim},
        {SimCcuPrimKind::ADDR_OP_VAR, &ExecAddrOpVarPrim},
        {SimCcuPrimKind::ADDR_OP_ADDR, &ExecAddrOpAddrPrim},
        {SimCcuPrimKind::ADDR_OP_IMM, &ExecAddrOpImmPrim},
        {SimCcuPrimKind::ADDR_ADDASSIGN_VAR, &ExecAddrAddassignVarPrim},
        {SimCcuPrimKind::LOAD_ARG, &ExecLoadArgPrim},
        {SimCcuPrimKind::LOAD_MEM, &ExecLoadMemPrim},
        {SimCcuPrimKind::LOAD_MEM_VAR, &ExecLoadMemPrim},
        {SimCcuPrimKind::STORE_MEM, &ExecStoreMemPrim},
        {SimCcuPrimKind::STORE_MEM_VAR, &ExecStoreMemPrim},
        {SimCcuPrimKind::LOCAL_ADDR_ALLOC, &ExecLocalAddrAllocPrim},
        {SimCcuPrimKind::REMOTE_ADDR_ALLOC, &ExecRemoteAddrAllocPrim},
        {SimCcuPrimKind::EVENT_RECORD, &ExecEventRecordPrim},
        {SimCcuPrimKind::EVENT_WAIT, &ExecEventWaitPrim},
        {SimCcuPrimKind::NOTIFY_RECORD, &ExecNotifyRecordPrim},
        {SimCcuPrimKind::NOTIFY_WAIT, &ExecNotifyWaitPrim},
        {SimCcuPrimKind::WRITE_VAR_NOTIFY, &ExecWriteVarNotifyPrim},
        {SimCcuPrimKind::LOCAL_NOTIFY_RECORD, &ExecLocalNotifyRecordPrim},
        {SimCcuPrimKind::LOCAL_NOTIFY_WAIT, &ExecLocalNotifyWaitPrim},
        {SimCcuPrimKind::LOCAL_COPY_MM, &ExecLocalCopyMmPrim},
        {SimCcuPrimKind::LOCAL_COPY_MB, &ExecLocalCopyMbPrim},
        {SimCcuPrimKind::LOCAL_COPY_BM, &ExecLocalCopyBmPrim},
        {SimCcuPrimKind::LOCAL_REDUCE_MM, &ExecLocalReduceMmPrim},
        {SimCcuPrimKind::LOCAL_REDUCE_BUFS, &ExecLocalReduceBufsPrim},
        {SimCcuPrimKind::REMOTE_READ_MM, &ExecRemoteReadMmPrim},
        {SimCcuPrimKind::REMOTE_READ_MB, &ExecRemoteReadMbPrim},
        {SimCcuPrimKind::REMOTE_READ_MR, &ExecRemoteReadMrPrim},
        {SimCcuPrimKind::REMOTE_WRITE_MM, &ExecRemoteWriteMmPrim},
        {SimCcuPrimKind::REMOTE_WRITE_BM, &ExecRemoteWriteBmPrim},
        {SimCcuPrimKind::REMOTE_WRITE_MR, &ExecRemoteWriteMrPrim},
    };

    /**
     * @brief 叶子原语执行分派（VAR/ADDR/LOAD/STORE/地址分配/事件/通知/本地-远端数据搬运）。
     *
     * 均不涉及控制流跳转（不改变指令游标），由 ReplayTrace 的 switch default 分支调用；
     * 按 kind 查 kLeafPrimHandlers 分发表。
     * @return false 表示未知原语（由调用方按错误处理）。
     */
    static bool ExecLeafPrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t instrIdx,
        bool recordable, HcclVmResult& replayErr)
    {
        for (const auto& handler : kLeafPrimHandlers) {
            if (handler.first == prim.kind) {
                return handler.second(ctx, entry, prim, instrIdx, recordable, replayErr);
            }
        }
        return false;
    }

    /** @brief 控制流指令的回放动作。 */
    enum class ReplayFlow {
        kNext,  /**< 顺序执行下一条（游标由主循环自增）。 */
        kJump,  /**< 已改写游标 i，主循环直接 continue。 */
        kAbort, /**< 终止回放。 */
    };

    /** @brief 回放 IF_BEGIN / IF_ELSE / IF_END（外层 if 未命中时不再求值条件）。 */
    ReplayFlow
    ReplayIfPrim(SimCcuExecContext& ctx, const SimCcuPrim& prim, size_t& i, const char* str, HcclVmResult& replayErr)
    {
        switch (prim.kind) {
            case SimCcuPrimKind::IF_BEGIN: {
                bool taken = false;
                // 外层 if 已抑制时不再求值（避免误判/触发交换变量解析阻塞）；压入 taken=false 帧保持配平。
                if (ShouldRecordCtx(ctx)
                    && !ExecEvalPrimCond(
                        &ctx, prim.lhsVar, prim.rhs, (prim.flags & 0x1U) != 0U, static_cast<CcuConditionType>(prim.op),
                        taken)) {
                    MarkReplayErr(replayErr, "if condition eval failed", prim);
                    taken = false;
                }
                ExecIfBegin(&ctx, taken);
                HCCL_VM_INFO(
                    "ccu replay if: label={}, taken={} (lhsVar=0x{:x}, rhs=0x{:x})", str, taken, prim.lhsVar, prim.rhs);
                break;
            }
            case SimCcuPrimKind::IF_ELSE:
                HCCL_VM_INFO("ccu replay instr[{}]: ifElse label={}", i, str);
                ExecIfElse(&ctx);
                break;
            default: // IF_END
                HCCL_VM_INFO("ccu replay instr[{}]: ifEnd label={}", i, str);
                ExecIfEnd(&ctx);
                break;
        }
        return ReplayFlow::kNext;
    }

    /** @brief 回放 WHILE_BEGIN / WHILE_END（含外层 if 抑制与单循环迭代上限兜底）。 */
    ReplayFlow ReplayWhilePrim(
        SimCcuExecContext& ctx, const SimCcuPrim& prim, size_t& i, const char* str,
        std::map<size_t, uint64_t>& whileIterCount, HcclVmResult& replayErr)
    {
        const bool recordable = ShouldRecordCtx(ctx);
        if (prim.kind == SimCcuPrimKind::WHILE_BEGIN) {
            bool loop = false;
            // 外层 if 抑制时体内计数器更新被抑制，仍求值条件必恒真 → 死循环；对外层抑制直接跳过体段。
            if (recordable
                && !ExecEvalPrimCond(
                    &ctx, prim.lhsVar, prim.rhs, (prim.flags & 0x1U) != 0U, static_cast<CcuConditionType>(prim.op),
                    loop)) {
                MarkReplayErr(replayErr, "while condition eval failed", prim);
                loop = false;
            }
            if (!recordable) {
                HCCL_VM_INFO(
                    "ccu replay while: label={} suppressed by outer if, skip body (to [{}])", str, prim.link + 1U);
            } else {
                HCCL_VM_INFO(
                    "ccu replay while: label={}, enter={} (lhsVar=0x{:x}, rhs=0x{:x})", str, loop, prim.lhsVar,
                    prim.rhs);
            }
            if (!loop) {
                i = prim.link + 1U; // 条件为假：跳过循环体，落到配对 WHILE_END 之后
                return ReplayFlow::kJump;
            }
            return ReplayFlow::kNext;
        }
        // WHILE_END
        if (!recordable) {
            HCCL_VM_INFO("ccu replay instr[{}]: whileEnd label={} suppressed by outer if, fall through", i, str);
            return ReplayFlow::kNext;
        }
        const uint64_t loopIter = ++whileIterCount[prim.link];
        if (loopIter > CCU_REPLAY_MAX_LOOP_ITER) {
            HCCL_VM_ERROR(
                "ccu replay while loop exceeded {} iterations (begin=[{}], label={}), possible infinite loop",
                CCU_REPLAY_MAX_LOOP_ITER, prim.link, str);
            replayErr = HCCL_SIM_E_INTERNAL;
            return ReplayFlow::kAbort;
        }
        HCCL_VM_INFO("ccu replay instr[{}]: whileEnd label={} (loop back to [{}])", i, str, prim.link);
        i = prim.link; // 无条件回跳配对 WHILE_BEGIN 重新求值
        return ReplayFlow::kJump;
    }

    /** @brief 回放 DO_BEGIN / DO_END（体至少执行一次；外层 if 抑制视为退出）。 */
    ReplayFlow
    ReplayDoPrim(SimCcuExecContext& ctx, const SimCcuPrim& prim, size_t& i, const char* str, HcclVmResult& replayErr)
    {
        if (prim.kind == SimCcuPrimKind::DO_BEGIN) {
            return ReplayFlow::kNext; // 体至少执行一次，直落
        }
        const bool recordable = ShouldRecordCtx(ctx);
        bool loop = false;
        // 外层 if 抑制时不求值条件：体内更新被抑制会导致条件恒真 → 幽灵迭代。
        if (recordable
            && !ExecEvalPrimCond(
                &ctx, prim.lhsVar, prim.rhs, (prim.flags & 0x1U) != 0U, static_cast<CcuConditionType>(prim.op), loop)) {
            MarkReplayErr(replayErr, "do-while condition eval failed", prim);
            loop = false;
        }
        loop = loop && recordable;
        HCCL_VM_INFO("ccu replay instr[{}]: do-while end label={}, reloop={} (back to [{}])", i, str, loop, prim.link);
        if (loop) {
            i = prim.link;
            return ReplayFlow::kJump;
        }
        return ReplayFlow::kNext;
    }

    /** @brief 回放 LOOP_BODY_ENTER / LOOP_BODY_EXIT（体段由 loop-group 展开器消费时跳到体段尾）。 */
    ReplayFlow ReplayLoopBodyPrim(const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t& i)
    {
        if (prim.kind != SimCcuPrimKind::LOOP_BODY_ENTER) {
            return ReplayFlow::kNext; // LOOP_BODY_EXIT：线性经过直接越过
        }
        const uint32_t enterLoopId = static_cast<uint32_t>(prim.dst);
        bool bodyInGroup = false;
        for (const SimCcuLoopGroup& grp : entry.groups) {
            for (const SimCcuLoopMember& grpMember : grp.members) {
                if (grpMember.loopIdx == enterLoopId) {
                    bodyInGroup = true;
                    break;
                }
            }
            if (bodyInGroup) {
                break;
            }
        }
        if (bodyInGroup && enterLoopId < entry.loopBodyExitIdx.size() && entry.loopBodyExitIdx[enterLoopId] > i) {
            HCCL_VM_INFO(
                "ccu replay instr[{}]: loopBodyEnter loopId={} (skip to body exit [{}], "
                "body consumed by loop-group expander)",
                i, enterLoopId, entry.loopBodyExitIdx[enterLoopId]);
            i = entry.loopBodyExitIdx[enterLoopId];
            return ReplayFlow::kJump;
        }
        HCCL_VM_INFO("ccu replay instr[{}]: loopBodyEnter loopId={} (bare loop, linear replay)", i, enterLoopId);
        return ReplayFlow::kNext;
    }

    /** @brief 回放单条指令：按类别派发（控制流/循环/叶子原语）。 */
    ReplayFlow ReplayOnePrim(
        SimCcuExecContext& ctx, const SimCcuKernelEntry& entry, const SimCcuPrim& prim, size_t& i, const char* str,
        bool recordable, std::map<size_t, uint64_t>& whileIterCount, HcclVmResult& replayErr)
    {
        switch (prim.kind) {
            case SimCcuPrimKind::IF_BEGIN:
            case SimCcuPrimKind::IF_ELSE:
            case SimCcuPrimKind::IF_END:
                return ReplayIfPrim(ctx, prim, i, str, replayErr);
            case SimCcuPrimKind::WHILE_BEGIN:
            case SimCcuPrimKind::WHILE_END:
                return ReplayWhilePrim(ctx, prim, i, str, whileIterCount, replayErr);
            case SimCcuPrimKind::DO_BEGIN:
            case SimCcuPrimKind::DO_END:
                return ReplayDoPrim(ctx, prim, i, str, replayErr);
            case SimCcuPrimKind::LOOP_BODY_ENTER:
            case SimCcuPrimKind::LOOP_BODY_EXIT:
                return ReplayLoopBodyPrim(entry, prim, i);
            case SimCcuPrimKind::LOOP_GROUP_BEGIN:
                // LoopGroup 展开：每条体指令产 1 条大块 MEM_CPY/REDUCE（详见 EmitLoopGroupTasks）。
                EmitLoopGroupTasks(ctx, entry, prim, i, recordable, replayErr);
                return ReplayFlow::kNext;
            case SimCcuPrimKind::LOOP_GROUP_END:
                return ReplayFlow::kNext; // 组结束占位（展开已在 BEGIN 处完成）
            default:
                if (!ExecLeafPrim(ctx, entry, prim, i, recordable, replayErr)) {
                    MarkReplayErr(replayErr, "unknown prim kind", prim);
                }
                return ReplayFlow::kNext;
        }
    }

    /** @brief 回放轨迹主循环：逐条指令派发（控制流/循环/叶子原语），结束后收尾。 */
    static HcclVmResult ReplayTrace(const SimCcuKernelEntry& entry)
    {
        SimCcuExecContext& ctx = g_tlsExecCtx;
        // 【核心步骤 4】回放轨迹：LoadArg 绑定 -> 算逻/地址求值 -> if 帧抑制 ->
        // while/do-while 游标循环 -> 数据/同步原语录制任务。
        const std::vector<SimCcuPrim>& trace = entry.trace;
        const std::vector<std::string>& strings = entry.strings;
        HcclVmResult replayErr = HCCL_SIM_SUCCESS;
        const char* emptyStr = "";

        size_t i = 0;
        uint64_t steps = 0;
        std::map<size_t, uint64_t> whileIterCount; // 每个 WHILE_BEGIN 索引的迭代计数（死循环防御）
        // 【回放步骤 1】步数上限防御；【回放步骤 2】逐条指令派发（ReplayOnePrim）；【回放步骤 3】按 flow 推进游标。
        while (i < trace.size()) {
            if (++steps > CCU_REPLAY_MAX_STEPS) {
                HCCL_VM_ERROR("ccu replay exceeded step limit {}, possible infinite loop", CCU_REPLAY_MAX_STEPS);
                replayErr = HCCL_SIM_E_INTERNAL;
                break;
            }
            const SimCcuPrim& prim = trace[i];
            const bool recordable = ShouldRecordCtx(ctx);
            const char* str = (prim.strIdx < strings.size()) ? strings[prim.strIdx].c_str() : emptyStr;
            const ReplayFlow flow = ReplayOnePrim(ctx, entry, prim, i, str, recordable, whileIterCount, replayErr);
            if (flow == ReplayFlow::kAbort) {
                break;
            }
            if (flow == ReplayFlow::kJump) {
                continue;
            }
            ++i;
        }

        // 【核心步骤 5】收尾：任务序列插入 + dump + 清理（无论回放成败都收尾）。
        const HcclVmResult finishRet = FinishExecution(entry);
        return (replayErr != HCCL_SIM_SUCCESS) ? replayErr : finishRet;
    }

    /** @brief Launch 期入口：组装回放上下文后回放轨迹并收尾。 */
    HcclVmResult ExecuteKernelOnLaunch(
        uint64_t threadHandle, uint64_t kernelHandle, const SimCcuKernelEntry& entry, const uint64_t* taskArgs,
        uint32_t argNum)
    {
        // 【步骤 0~3】组装回放上下文。
        const HcclVmResult prep = PrepareReplayContext(threadHandle, kernelHandle, entry, taskArgs, argNum);
        if (prep != HCCL_SIM_SUCCESS) {
            return prep;
        }
        // 【步骤 4~5】回放轨迹并收尾（任务插入 + dump + TLS 清理）。
        return ReplayTrace(entry);
    }

    /*==================== 录制期查询/句柄分配 ====================*/

    /** @brief 当前线程是否正处于注册期录制（kernelFunc 执行期）。 */
    bool IsRecording() { return ActiveRecordCtx() != nullptr; }

    /** @brief 当前线程是否处于 Launch 回放期。 */
    bool IsExecuting() { return ActiveExecCtx() != nullptr; }

    /** @brief 当前回放上下文的 rank（未回放时返回 UINT32_MAX）。 */
    uint32_t CurRankId()
    {
        const SimCcuExecContext* ctx = ActiveExecCtx();
        return (ctx != nullptr) ? ctx->rankId : UINT32_MAX;
    }

    /** @brief 非录制期防御签发使用的保留 kernelIdx（避免与真实 kernel 命名空间碰撞）。 */
    constexpr uint32_t CCU_HANDLE_KERNEL_ORPHAN = 0x3FFFU;

    /** @brief 桩侧句柄 tag 校验：0 视为空句柄放行。 */
    bool CheckHandleTagOf(uint64_t handle, SimCcuHandleTag expect)
    {
        return (handle == 0U) || (SimCcuHandleTagOf(handle) == static_cast<uint8_t>(expect));
    }

    /** @brief 分配原语句柄（录制期 kernel 命名空间；非录制期防御分支用全局序列 + 保留 kernelIdx）。 */
    uint64_t AllocHandle(SimCcuHandleTag tag)
    {
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx != nullptr) {
            ctx->handleSeq += 1U;
            return MakeSimCcuHandle(tag, ctx->kernelIdx, static_cast<uint32_t>(ctx->handleSeq & 0xFFFFFFFFULL));
        }
        // 防御分支（理论不可达：桩仅在录制窗口被调用）：全局序列 + 保留 kernelIdx，仅保证唯一非零。
        return MakeSimCcuHandle(
            tag, CCU_HANDLE_KERNEL_ORPHAN, static_cast<uint32_t>(g_globalHandleSeq.fetch_add(1U) & 0xFFFFFFFFULL));
    }

    /** @brief 分配全局唯一 Event(CKE) 句柄（低 32 位序列编码进 notifyId，跨 Launch 唯一）。 */
    uint64_t AllocEventHandle()
    {
        // Event 句柄全局唯一（R2 例外：notifyId 配对要求跨 Launch 唯一），
        // 不入 kernel 命名空间；句柄低 32 位（seq）编码进 EventBitNotifyId 后
        // 仍与 DB notifyId、通道位 id 值域隔离。
        const uint64_t seq = g_globalEventSeq.fetch_add(1U);
        return MakeSimCcuHandle(SimCcuHandleTag::EVENT, 0U, static_cast<uint32_t>(seq & 0xFFFFFFFFULL));
    }

    /** @brief 登记通道派生变量来源 varHandle→(channel, varIndex)，供回放期懒解析交换值。 */
    void NoteChannelVarSource(uint64_t varHandle, uint64_t channelHandle, uint32_t varIndex)
    {
        SimCcuRecordContext* rctx = ActiveRecordCtx();
        if (rctx == nullptr) {
            // 非录制期调用（理论不可达：桩仅在录制窗口被调用）；静默返回，回放侧退化为默认值 0。
            HCCL_VM_WARN("ccu channel var source noted outside record window, var=0x{:x}", varHandle);
            return;
        }
        rctx->channelVarSources[varHandle] = {channelHandle, varIndex};
        HCCL_VM_INFO("ccu channel var bound, var=0x{:x}, channel={}, varIndex={}", varHandle, channelHandle, varIndex);
    }

    /* Check*Handle 函数族已 constexpr inline 定义在 ccu_level1_common.h 中 */

    /*==================== 录制：Prim* 实现 ====================*/

    /** @brief 录制 var = 立即数（VAR_ASSIGN_IMM）。 */
    void PrimVarAssignImm(uint64_t varHandle, uint64_t immediate)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::VAR_ASSIGN_IMM);
        prim.dst = varHandle;
        prim.src = immediate;
    }

    /** @brief 录制 var = var（VAR_ASSIGN_VAR）。 */
    void PrimVarAssignVar(uint64_t dstVar, uint64_t srcVar)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::VAR_ASSIGN_VAR);
        prim.dst = dstVar;
        prim.src = srcVar;
    }

    /** @brief 录制 var = varA op varB（VAR_OP_VAR）。 */
    void PrimVarOpVar(uint64_t resVar, uint64_t varA, uint64_t varB, SimCcuVarOp op)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::VAR_OP_VAR);
        prim.dst = resVar;
        prim.varA = varA;
        prim.varB = varB;
        prim.op = static_cast<uint8_t>(op);
    }

    /** @brief 录制 var = varA op 立即数（VAR_OP_IMM）。 */
    void PrimVarOpImm(uint64_t resVar, uint64_t varA, uint64_t immediate, SimCcuVarOp op)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::VAR_OP_IMM);
        prim.dst = resVar;
        prim.varA = varA;
        prim.imm = immediate;
        prim.op = static_cast<uint8_t>(op);
    }

    /** @brief 录制 var = ~varA（VAR_NOT）。 */
    void PrimVarNot(uint64_t resVar, uint64_t varA)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::VAR_NOT);
        prim.dst = resVar;
        prim.src = varA;
    }

    /** @brief 录制 addr = 立即数（ADDR_ASSIGN_IMM）。 */
    void PrimAddrAssignImm(uint64_t addrHandle, uint64_t immediate)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::ADDR_ASSIGN_IMM);
        prim.dst = addrHandle;
        prim.src = immediate;
    }

    /** @brief 录制 addr = addr（ADDR_ASSIGN_ADDR）。 */
    void PrimAddrAssignAddr(uint64_t dstAddr, uint64_t srcAddr)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::ADDR_ASSIGN_ADDR);
        prim.dst = dstAddr;
        prim.src = srcAddr;
    }

    /** @brief 录制 addr = var（ADDR_ASSIGN_VAR）。 */
    void PrimAddrAssignVar(uint64_t addrHandle, uint64_t varHandle)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::ADDR_ASSIGN_VAR);
        prim.dst = addrHandle;
        prim.src = varHandle;
    }

    /** @brief 录制 addr = addr op var（ADDR_OP_VAR）。 */
    void PrimAddrOpVar(uint64_t resAddr, uint64_t lhsAddr, uint64_t rhsVar, SimCcuVarOp op)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::ADDR_OP_VAR);
        prim.dst = resAddr;
        prim.src = lhsAddr;
        prim.varB = rhsVar;
        prim.op = static_cast<uint8_t>(op);
    }

    /** @brief 录制 addr = addr op addr（ADDR_OP_ADDR）。 */
    void PrimAddrOpAddr(uint64_t resAddr, uint64_t addrA, uint64_t addrB, SimCcuVarOp op)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::ADDR_OP_ADDR);
        prim.dst = resAddr;
        prim.src = addrA;
        prim.varB = addrB;
        prim.op = static_cast<uint8_t>(op);
    }

    /** @brief 录制 addr = addr op 立即数（ADDR_OP_IMM）。 */
    void PrimAddrOpImm(uint64_t resAddr, uint64_t addrA, uint64_t immediate, SimCcuVarOp op)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::ADDR_OP_IMM);
        prim.dst = resAddr;
        prim.src = addrA;
        prim.imm = immediate;
        prim.op = static_cast<uint8_t>(op);
    }

    /** @brief 录制 addr += var（ADDR_ADDASSIGN_VAR）。 */
    void PrimAddrAddAssignVar(uint64_t addrHandle, uint64_t varHandle)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::ADDR_ADDASSIGN_VAR);
        prim.dst = addrHandle;
        prim.src = varHandle;
    }

    /** @brief 录制 var = LoadArg(argId)（LOAD_ARG，回放期绑定 Launch 参数）。 */
    void PrimLoadArg(uint64_t varHandle, uint32_t argId)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::LOAD_ARG);
        prim.dst = varHandle;
        prim.aux = argId;
    }

    /** @brief 录制 var = LoadMem(addr, num)（LOAD_MEM / LOAD_MEM_VAR）。 */
    void PrimLoadMem(uint64_t varHandle, uint64_t addr, uint32_t num, bool addrIsVar)
    {
        SimCcuPrim& prim = AppendPrim(addrIsVar ? SimCcuPrimKind::LOAD_MEM_VAR : SimCcuPrimKind::LOAD_MEM);
        prim.dst = varHandle;
        prim.src = addr;
        prim.aux = num;
    }

    /** @brief 录制 StoreMem(addr, var, num)（STORE_MEM / STORE_MEM_VAR）。 */
    void PrimStoreMem(uint64_t varOrAddrHandle, uint64_t varHandle, uint32_t num, bool addrIsVar)
    {
        SimCcuPrim& prim = AppendPrim(addrIsVar ? SimCcuPrimKind::STORE_MEM_VAR : SimCcuPrimKind::STORE_MEM);
        prim.dst = varOrAddrHandle;
        prim.src = varHandle;
        prim.aux = num;
    }

    /** @brief 录制本地地址分配（LOCAL_ADDR_ALLOC）。 */
    void PrimLocalAddrAlloc(uint64_t localAddrHandle, uint64_t addrHandle, uint64_t tokenHandle)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::LOCAL_ADDR_ALLOC);
        prim.dst = localAddrHandle;
        prim.addr = addrHandle;
        prim.token = tokenHandle;
    }

    /** @brief 录制远端地址分配（REMOTE_ADDR_ALLOC）。 */
    void PrimRemoteAddrAlloc(uint64_t remoteAddrHandle, uint64_t addrHandle, uint64_t tokenHandle)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::REMOTE_ADDR_ALLOC);
        prim.dst = remoteAddrHandle;
        prim.addr = addrHandle;
        prim.token = tokenHandle;
    }

    /** @brief 录制事件位置位（EVENT_RECORD，mask 逐 bit 拆分）。 */
    void PrimEventRecord(uint64_t eventHandle, uint16_t mask)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::EVENT_RECORD);
        prim.event = eventHandle;
        prim.aux = mask;
    }

    /** @brief 录制事件位等待（EVENT_WAIT，mask 逐 bit 拆分）。 */
    void PrimEventWait(uint64_t eventHandle, uint16_t mask)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::EVENT_WAIT);
        prim.event = eventHandle;
        prim.aux = mask;
    }

    /** @brief 录制通道通知置位（NOTIFY_RECORD）。 */
    void PrimNotifyRecord(uint64_t channel, uint32_t notifyIdx, uint16_t mask)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::NOTIFY_RECORD);
        prim.channel = channel;
        prim.aux = notifyIdx;
        prim.aux2 = mask;
    }

    /** @brief 录制通道通知等待（NOTIFY_WAIT）。 */
    void PrimNotifyWait(uint64_t channel, uint32_t notifyIdx, uint16_t mask)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::NOTIFY_WAIT);
        prim.channel = channel;
        prim.aux = notifyIdx;
        prim.aux2 = mask;
    }

    /** @brief 录制“写远端变量 + 通知”（WRITE_VAR_NOTIFY，PreSync 交换机制）。 */
    void PrimWriteVarNotify(
        uint64_t channel, uint64_t varHandle, uint32_t remoteVarIdx, uint32_t remoteNotifyIdx, uint16_t mask)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::WRITE_VAR_NOTIFY);
        prim.channel = channel;
        prim.src = varHandle;
        prim.extOff = remoteVarIdx; // 对端 XN 槽号（交换区 0..3）：回放期写方据此发布交换值
        prim.aux = remoteNotifyIdx;
        prim.aux2 = mask;
    }

    /** @brief 录制按 tag 的本端通知置位（LOCAL_NOTIFY_RECORD）。 */
    void PrimLocalNotifyRecord(const char* notifyTag, uint16_t mask)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::LOCAL_NOTIFY_RECORD);
        prim.strIdx = PoolString(notifyTag);
        prim.aux = mask;
    }

    /** @brief 录制按 tag 的本端通知等待（LOCAL_NOTIFY_WAIT）。 */
    void PrimLocalNotifyWait(const char* notifyTag, uint16_t mask)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::LOCAL_NOTIFY_WAIT);
        prim.strIdx = PoolString(notifyTag);
        prim.aux = mask;
    }

    /** @brief 录制本地 mem→mem 拷贝（LOCAL_COPY_MM）。 */
    void PrimLocalCopyMm(uint64_t dstLocal, uint64_t srcLocal, uint64_t lenVar, uint64_t event, uint16_t mask)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::LOCAL_COPY_MM);
        prim.dst = dstLocal;
        prim.local = srcLocal;
        prim.lenVar = lenVar;
        prim.event = event;
        prim.aux = mask;
    }

    /** @brief 录制本地 mem→缓冲(MS) 拷贝（LOCAL_COPY_MB）。 */
    void PrimLocalCopyMb(uint64_t dstBuf, uint64_t srcLocal, uint64_t lenVar, uint64_t event, uint16_t mask)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::LOCAL_COPY_MB);
        prim.dst = dstBuf;
        prim.local = srcLocal;
        prim.lenVar = lenVar;
        prim.event = event;
        prim.aux = mask;
    }

    /** @brief 录制本地缓冲(MS)→mem 拷贝（LOCAL_COPY_BM）。 */
    void PrimLocalCopyBm(uint64_t dstLocal, uint64_t srcBuf, uint64_t lenVar, uint64_t event, uint16_t mask)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::LOCAL_COPY_BM);
        prim.dst = dstLocal;
        prim.local = srcBuf;
        prim.lenVar = lenVar;
        prim.event = event;
        prim.aux = mask;
    }

    /** @brief 录制本地 mem→mem 归约（LOCAL_REDUCE_MM）。 */
    void PrimLocalReduceMm(
        uint64_t dstLocal, uint64_t srcLocal, uint64_t lenVar, uint64_t event, uint16_t mask, uint8_t dataType,
        uint8_t reduceOp)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::LOCAL_REDUCE_MM);
        prim.dst = dstLocal;
        prim.local = srcLocal;
        prim.lenVar = lenVar;
        prim.event = event;
        prim.aux = mask;
        prim.aux2 = (static_cast<uint32_t>(dataType) << 8U) | reduceOp;
    }

    /** @brief 录制多缓冲归约（LOCAL_REDUCE_BUFS，buffers[1..]→buffers[0]）。 */
    void PrimLocalReduceBufs(
        const uint64_t* buffers, uint32_t count, uint64_t lenVar, uint64_t event, uint16_t mask, uint8_t dataType,
        uint8_t reduceOp)
    {
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx == nullptr) {
            return;
        }
        const uint32_t extOff = static_cast<uint32_t>(ctx->extraBufs.size());
        for (uint32_t k = 0; k < count; ++k) {
            ctx->extraBufs.push_back((buffers != nullptr) ? buffers[k] : 0U);
        }
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::LOCAL_REDUCE_BUFS);
        prim.extOff = extOff;
        prim.aux = count; // count 占用 aux，mask 打包进 aux2 高 16 位
        prim.lenVar = lenVar;
        prim.event = event;
        prim.aux2 = (static_cast<uint32_t>(mask) << 16U) | (static_cast<uint32_t>(dataType) << 8U) | reduceOp;
    }

    /** @brief 录制远端读 mem→本地 mem（REMOTE_READ_MM）。 */
    void PrimRemoteReadMm(
        uint64_t channel, uint64_t localH, uint64_t remoteH, uint64_t lenVar, uint64_t event, uint16_t mask)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::REMOTE_READ_MM);
        prim.channel = channel;
        prim.local = localH;
        prim.remote = remoteH;
        prim.lenVar = lenVar;
        prim.event = event;
        prim.aux = mask;
    }

    /** @brief 录制远端读 mem→本地缓冲(MS)（REMOTE_READ_MB）。 */
    void
    PrimRemoteReadMb(uint64_t channel, uint64_t bufH, uint64_t remoteH, uint64_t lenVar, uint64_t event, uint16_t mask)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::REMOTE_READ_MB);
        prim.channel = channel;
        prim.local = bufH;
        prim.remote = remoteH;
        prim.lenVar = lenVar;
        prim.event = event;
        prim.aux = mask;
    }

    /** @brief 录制远端读并归约（REMOTE_READ_MR）。 */
    void PrimRemoteReadMr(
        uint64_t channel, uint64_t localH, uint64_t remoteH, uint64_t lenVar, uint64_t event, uint16_t mask,
        uint8_t dataType, uint8_t reduceOp)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::REMOTE_READ_MR);
        prim.channel = channel;
        prim.local = localH;
        prim.remote = remoteH;
        prim.lenVar = lenVar;
        prim.event = event;
        prim.aux = mask;
        prim.aux2 = (static_cast<uint32_t>(dataType) << 8U) | reduceOp;
    }

    /** @brief 录制本地 mem→远端 mem（REMOTE_WRITE_MM）。 */
    void PrimRemoteWriteMm(
        uint64_t channel, uint64_t remoteH, uint64_t localH, uint64_t lenVar, uint64_t event, uint16_t mask)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::REMOTE_WRITE_MM);
        prim.channel = channel;
        prim.local = remoteH;
        prim.remote = localH;
        prim.lenVar = lenVar;
        prim.event = event;
        prim.aux = mask;
    }

    /** @brief 录制本地缓冲(MS)→远端 mem（REMOTE_WRITE_BM）。 */
    void
    PrimRemoteWriteBm(uint64_t channel, uint64_t remoteH, uint64_t bufH, uint64_t lenVar, uint64_t event, uint16_t mask)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::REMOTE_WRITE_BM);
        prim.channel = channel;
        prim.local = remoteH;
        prim.remote = bufH;
        prim.lenVar = lenVar;
        prim.event = event;
        prim.aux = mask;
    }

    /** @brief 录制本地写远端并归约（REMOTE_WRITE_MR）。 */
    void PrimRemoteWriteMr(
        uint64_t channel, uint64_t remoteH, uint64_t localH, uint64_t lenVar, uint64_t event, uint16_t mask,
        uint8_t dataType, uint8_t reduceOp)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::REMOTE_WRITE_MR);
        prim.channel = channel;
        prim.local = remoteH;
        prim.remote = localH;
        prim.lenVar = lenVar;
        prim.event = event;
        prim.aux = mask;
        prim.aux2 = (static_cast<uint32_t>(dataType) << 8U) | reduceOp;
    }

    bool IsInsideLoopBody(const char* what); // 前置声明（定义在 while 校验区）

    /** @brief 录制 if 起始（IF_BEGIN；label 入池，loop 体内调用告警）。 */
    void PrimIfBegin(uint64_t lhsVar, uint64_t rhs, bool isVarCompare, CcuConditionType condType, const char* label)
    {
        // LOOP_BLOCK 守卫：真身拒绝 if 入 loop 体（ccu_kernel.cc:1453）；我们桩恒 SUCCESS
        // 保两分支录制，但体段含控制流会使大块抽象展开器无法安全复制——ERROR 告警。
        (void)IsInsideLoopBody("if-begin");
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::IF_BEGIN);
        prim.lhsVar = lhsVar;
        prim.rhs = rhs;
        prim.flags = isVarCompare ? 0x1U : 0U;
        prim.op = static_cast<uint8_t>(condType);
        prim.strIdx = PoolString(label);
    }

    /** @brief 录制 if/else 分界（IF_ELSE）。 */
    void PrimIfElse(const char* label)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::IF_ELSE);
        prim.strIdx = PoolString(label);
    }

    /** @brief 录制 if 结束（IF_END）。 */
    void PrimIfEnd(const char* label)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::IF_END);
        prim.strIdx = PoolString(label);
    }

    /** @brief 录制 while 起始（WHILE_BEGIN；label 入池）。 */
    void PrimWhileBegin(uint64_t lhsVar, uint64_t rhs, bool isVarCompare, CcuConditionType condType, const char* label)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::WHILE_BEGIN);
        prim.lhsVar = lhsVar;
        prim.rhs = rhs;
        prim.flags = isVarCompare ? 0x1U : 0U;
        prim.op = static_cast<uint8_t>(condType);
        prim.strIdx = PoolString(label);
    }

    /** @brief 录制 while 结束（WHILE_END）。 */
    void PrimWhileEnd(const char* label)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::WHILE_END);
        prim.strIdx = PoolString(label);
    }

    /** @brief 录制 do-while 起始（DO_BEGIN）。 */
    void PrimDoBegin(const char* label)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::DO_BEGIN);
        prim.strIdx = PoolString(label);
    }

    /*==================== 录制期 while/do-while 事务校验 ====================*/

    /**
     * @brief LOOP_BLOCK 守卫：当前处于 loop 体内时拒绝控制流/循环原语。
     *
     * 对齐真身（ccu_kernel.cc:1453/1598/2320 的 LOOP_BLOCK 检查）：loop 体必须为
     * 直线代码（无 if/while/嵌套 Loop），这是大块抽象展开器"安全复制体段"的前提。
     * 返回 true 表示违规（调用方应拒绝操作）。
     */
    bool IsInsideLoopBody(const char* what)
    {
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx == nullptr) {
            return false;
        }
        if (ctx->curLoopDepth > 0U) {
            HCCL_VM_ERROR("ccu {} rejected: not allowed inside a ccu::Loop body (depth={})", what, ctx->curLoopDepth);
            return true;
        }
        return false;
    }

    /*
     * 对齐真身 pendingWhileCtx_/pendingDoWhileCtx_ 的查重与配对语义：
     *  - Begin 查重（同 label 未闭合重开）→ CCU_E_PARA（宏将跳过循环体）；
     *  - End 查找（无匹配 Begin）→ CCU_E_NOT_FOUND；
     *  - 非录制期调用 → CCU_E_PTR（对齐真身 GetCurrentKernel 为空的错误码）。
     */

    CcuResult WhileBeginCheck(const char* label)
    {
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx == nullptr) {
            HCCL_VM_ERROR("ccu while-begin check: no kernel is recording");
            return CcuResult::CCU_E_PTR;
        }
        if (IsInsideLoopBody("while-begin")) {
            return CcuResult::CCU_E_INTERNAL; // 对齐真身 LatchBodyError(CCU_E_INTERNAL)
        }
        const std::string labelStr((label != nullptr) ? label : "");
        if (ctx->whileOpenLabels.count(labelStr) != 0U) {
            HCCL_VM_ERROR(
                "ccu while-begin check: label '{}' already has a pending WhileBegin without WhileEnd", labelStr);
            return CcuResult::CCU_E_PARA;
        }
        ctx->whileOpenLabels.insert(labelStr);
        return CcuResult::CCU_SUCCESS;
    }

    /** @brief CCU_WHILE 收尾事务校验：无匹配 WhileBegin 返回 CCU_E_NOT_FOUND。 */
    CcuResult WhileEndCheck(const char* label)
    {
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx == nullptr) {
            HCCL_VM_ERROR("ccu while-end check: no kernel is recording");
            return CcuResult::CCU_E_PTR;
        }
        const std::string labelStr((label != nullptr) ? label : "");
        if (ctx->whileOpenLabels.erase(labelStr) == 0U) {
            HCCL_VM_ERROR("ccu while-end check: no matching WhileBegin for label '{}'", labelStr);
            return CcuResult::CCU_E_NOT_FOUND;
        }
        return CcuResult::CCU_SUCCESS;
    }

    /** @brief CCU_DO 起始事务校验：同 label 未闭合重开返回 CCU_E_PARA。 */
    CcuResult DoWhileBeginCheck(const char* label)
    {
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx == nullptr) {
            HCCL_VM_ERROR("ccu do-while-begin check: no kernel is recording");
            return CcuResult::CCU_E_PTR;
        }
        const std::string labelStr((label != nullptr) ? label : "");
        if (ctx->doWhileOpenLabels.count(labelStr) != 0U) {
            HCCL_VM_ERROR(
                "ccu do-while-begin check: label '{}' already has a pending DoWhileBegin without DoWhileEnd", labelStr);
            return CcuResult::CCU_E_PARA;
        }
        ctx->doWhileOpenLabels.insert(labelStr);
        return CcuResult::CCU_SUCCESS;
    }

    /** @brief do-while 收尾事务校验：无匹配 DoWhileBegin 返回 CCU_E_NOT_FOUND。 */
    CcuResult DoWhileEndCheck(const char* label)
    {
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx == nullptr) {
            HCCL_VM_ERROR("ccu do-while-end check: no kernel is recording");
            return CcuResult::CCU_E_PTR;
        }
        const std::string labelStr((label != nullptr) ? label : "");
        if (ctx->doWhileOpenLabels.erase(labelStr) == 0U) {
            HCCL_VM_ERROR("ccu do-while-end check: no matching DoWhileBegin for label '{}'", labelStr);
            return CcuResult::CCU_E_NOT_FOUND;
        }
        return CcuResult::CCU_SUCCESS;
    }

    /** @brief 录制 do-while 结束条件（DO_END）。 */
    void PrimDoEnd(uint64_t lhsVar, uint64_t rhs, bool isVarCompare, CcuConditionType condType, const char* label)
    {
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::DO_END);
        prim.lhsVar = lhsVar;
        prim.rhs = rhs;
        prim.flags = isVarCompare ? 0x1U : 0U;
        prim.op = static_cast<uint8_t>(condType);
        prim.strIdx = PoolString(label);
    }

    /*==================== 录制期控制流宏标签栈 ====================*/

    void IfStackPush(const char* label)
    {
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx == nullptr) {
            return;
        }
        SimCcuIfLabelEntry entry;
        entry.label = (label != nullptr) ? label : "";
        ctx->ifStack.push_back(std::move(entry));
    }

    /** @brief 标记栈顶 CCU_IF 的 if 体已执行完（允许后续 CCU_ELSE 配对 / flush 闭合）。 */
    void IfStackMarkBodyDone()
    {
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx == nullptr) {
            return;
        }
        if (ctx->ifStack.empty()) {
            HCCL_VM_ERROR("ccu if-stack mark-body-done: stack is empty");
            return;
        }
        // 标记栈顶（最近的 CCU_IF）body 已执行完，允许后续 CCU_ELSE 配对 / flush 闭合。
        ctx->ifStack.back().bodyDone = true;
    }

    /** @brief 弹出栈顶 if 标签供 CCU_ELSE 使用，返回池内保活指针（无匹配返回 nullptr）。 */
    const char* IfStackPopForElse()
    {
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx == nullptr) {
            return nullptr;
        }
        // 与真身 IfLabelStackPopForElse 语义一致：仅检查栈顶，不向下搜索。
        if (ctx->ifStack.empty()) {
            HCCL_VM_ERROR("ccu if-stack pop-for-else: orphan CCU_ELSE, no matching CCU_IF on the stack");
            return nullptr;
        }
        if (!ctx->ifStack.back().bodyDone) {
            HCCL_VM_ERROR(
                "ccu if-stack pop-for-else: CCU_ELSE called while top if-body is still in body, label={}",
                ctx->ifStack.back().label);
            return nullptr;
        }
        // 栈项弹出即析构，label 必须拷贝返回；宏在后续增量中仍会使用该指针
        // （CcuIfElse/CcuIfEnd 入参），借用字符串池保活（append-only，不会失效）。
        const std::string label = ctx->ifStack.back().label;
        ctx->ifStack.pop_back();
        ctx->strings.push_back(label);
        return ctx->strings.back().c_str();
    }

    /** @brief 闭合所有可闭合的挂起 if（AppendPrim 兜底 / RegisterKernel 收尾共用）。 */
    void FlushPendingIfs()
    {
        // 与真身 FlushClosablePendingIfs 语义一致（CcuFlushPendingIfs 桩 / AppendPrim 兜底共用）。
        CloseClosableIfs(ActiveRecordCtx());
    }

    /** @brief 压入 do-while 标签（CCU_DO 收尾时调用）。 */
    void DoWhileStackPush(const char* label)
    {
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx == nullptr) {
            return;
        }
        ctx->doWhileStack.emplace_back((label != nullptr) ? label : "");
    }

    /** @brief 弹出 do-while 标签供 CCU_WHILE 作为其结束条件；返回池内保活指针。 */
    const char* DoWhileStackPopForWhile()
    {
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx == nullptr || ctx->doWhileStack.empty()) {
            return nullptr;
        }
        // 栈项弹出即析构，label 必须拷贝返回；宏在后续增量中仍会使用该指针
        // （CcuDoWhileEnd 入参），借用字符串池保活（append-only，不会失效）。
        // 与 IfStackPopForElse 同口径，避免返回悬垂指针（review #1.2）。
        const std::string label = ctx->doWhileStack.back();
        ctx->doWhileStack.pop_back();
        ctx->strings.push_back(label);
        return ctx->strings.back().c_str();
    }

    /*==================== 录制：Loop/LoopGroup ====================*/

    /** @brief 录制 loop 体段开始标记（LOOP_BODY_ENTER）并推进 loop 嵌套深度。 */
    void PrimLoopBodyEnter(uint32_t loopId)
    {
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx == nullptr) {
            return;
        }
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::LOOP_BODY_ENTER);
        prim.dst = loopId;
        if (loopId >= ctx->loopBodyEnterIdx.size()) {
            ctx->loopBodyEnterIdx.resize(loopId + 1U, 0U);
            ctx->loopBodyExitIdx.resize(loopId + 1U, 0U);
        }
        // 记录边界索引需在 AppendPrim 返回后取 trace 末尾（AppendPrim 内部可能先 flush 补录条目）。
        ctx->loopBodyEnterIdx[loopId] = static_cast<uint32_t>(ctx->trace.size() - 1U);
        ctx->curLoopDepth += 1U;
        HCCL_VM_INFO(
            "ccu loop body enter, loopId={}, traceIdx={}, depth={}", loopId, ctx->loopBodyEnterIdx[loopId],
            ctx->curLoopDepth);
    }

    /** @brief 录制 loop 体段结束标记（LOOP_BODY_EXIT）并回退 loop 嵌套深度。 */
    void PrimLoopBodyExit(uint32_t loopId)
    {
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx == nullptr) {
            return;
        }
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::LOOP_BODY_EXIT);
        prim.dst = loopId;
        if (loopId >= ctx->loopBodyExitIdx.size()) {
            ctx->loopBodyExitIdx.resize(loopId + 1U, 0U);
        }
        ctx->loopBodyExitIdx[loopId] = static_cast<uint32_t>(ctx->trace.size() - 1U);
        if (ctx->curLoopDepth > 0U) {
            ctx->curLoopDepth -= 1U;
        }
        HCCL_VM_INFO(
            "ccu loop body exit, loopId={}, traceIdx={}, bodyPrimCount={}", loopId, ctx->loopBodyExitIdx[loopId],
            ctx->loopBodyExitIdx[loopId] - ctx->loopBodyEnterIdx[loopId] - 1U);
    }

    /**
     * @brief 录制 LoopGroup 开始：登记组级并行参数来源并回填 groupIdx。
     */
    void PrimLoopGroupBegin(
        SimCcuLoopParamSrc src, uint64_t cfgCloneNum, uint64_t cfgCloneLoopOffset, uint64_t cfgAddrOffset,
        uint64_t cfgCcuBufferOffset, uint64_t cfgEventOffset, uint64_t cfgVarOffset, uint64_t parallelVar,
        uint64_t offsetVar, uint64_t varOffsetVar, uint32_t& groupIdx)
    {
        groupIdx = UINT32_MAX;
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx == nullptr) {
            return;
        }
        SimCcuLoopGroup group;
        group.src = src;
        group.cfgCloneNum = cfgCloneNum;
        group.cfgCloneLoopOffset = cfgCloneLoopOffset;
        group.cfgAddrOffset = cfgAddrOffset;
        group.cfgCcuBufferOffset = cfgCcuBufferOffset;
        group.cfgEventOffset = cfgEventOffset;
        group.cfgVarOffset = cfgVarOffset;
        group.parallelVar = parallelVar;
        group.offsetVar = offsetVar;
        group.varOffsetVar = varOffsetVar;

        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::LOOP_GROUP_BEGIN);
        const uint32_t beginIdx = static_cast<uint32_t>(ctx->trace.size() - 1U);
        group.traceBeginIdx = beginIdx;
        prim.dst = static_cast<uint64_t>(ctx->groups.size());
        ctx->groups.push_back(std::move(group));
        groupIdx = static_cast<uint32_t>(ctx->groups.size() - 1U);
        HCCL_VM_INFO(
            "ccu loop group begin, groupIdx={}, traceIdx={}, src={}", groupIdx, beginIdx, static_cast<int32_t>(src));
    }

    /** @brief 录制 LoopGroup 成员（迭代数/步进及其参数来源）。 */
    void PrimLoopGroupAddLoop(
        uint32_t groupIdx, uint32_t loopIdx, SimCcuLoopParamSrc src, uint64_t cfgIterNum, uint64_t cfgAddrOffset,
        uint64_t paramVar, uint64_t iterNumVar, uint64_t addrOffsetVar, uint64_t ctxIdVar)
    {
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx == nullptr || groupIdx >= ctx->groups.size()) {
            HCCL_VM_ERROR("ccu loop group add-loop rejected: bad groupIdx={}", groupIdx);
            return;
        }
        SimCcuLoopMember member;
        member.loopIdx = loopIdx;
        member.src = src;
        member.cfgIterNum = cfgIterNum;
        member.cfgAddrOffset = cfgAddrOffset;
        member.paramVar = paramVar;
        member.iterNumVar = iterNumVar;
        member.addrOffsetVar = addrOffsetVar;
        member.ctxIdVar = ctxIdVar;
        // AddLoop 时刻全量快照 varValues（ccu-sim-stub-design.md §4.3：主/尾 group 复用
        // 同一 ccu::Loop 时取值不同——主 group loopLen[0]=memSlice、尾 group=residual）。
        member.varSnapshot = g_tlsExecCtx.active ? g_tlsExecCtx.varValues : std::map<uint64_t, uint64_t>{};
        // 注：录制期 varValues 不存在（求值在回放期），快照退化为空表——参数本身以
        // 句柄/立即数记录，回放期由展开器从 Launch 上下文求值。快照结构保留，供未来
        // "录制期已知常量"场景使用。
        ctx->groups[groupIdx].members.push_back(std::move(member));
        HCCL_VM_INFO(
            "ccu loop group add-loop, groupIdx={}, memberIdx={}, loopIdx={}, src={}", groupIdx,
            ctx->groups[groupIdx].members.size() - 1U, loopIdx, static_cast<int32_t>(src));
    }

    /** @brief 录制 LoopGroup 结束标记。 */
    void PrimLoopGroupEnd(uint32_t groupIdx)
    {
        SimCcuRecordContext* ctx = ActiveRecordCtx();
        if (ctx == nullptr || groupIdx >= ctx->groups.size()) {
            HCCL_VM_ERROR("ccu loop group end rejected: bad groupIdx={}", groupIdx);
            return;
        }
        SimCcuPrim& prim = AppendPrim(SimCcuPrimKind::LOOP_GROUP_END);
        ctx->groups[groupIdx].traceEndIdx = static_cast<uint32_t>(ctx->trace.size() - 1U);
        prim.dst = groupIdx;
        HCCL_VM_INFO(
            "ccu loop group end, groupIdx={}, traceIdx=[{},{}], memberCount={}", groupIdx,
            ctx->groups[groupIdx].traceBeginIdx, ctx->groups[groupIdx].traceEndIdx,
            ctx->groups[groupIdx].members.size());
    }

} // namespace CcuSim
} // namespace HcclSim
