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
 * Description: HostDPU 北向劫持（level1）设备任务转发环消费者实现。
 *              host 执行器线程按环序消费 device 进程转发的任务记录并落库，
 *              重建"每流单写者、落库序 = 设备程序序"的真实 hcomm 不变量
 *              （根因与协议详见 dpu_level1_common.h 第 1b 节注释）。
 * Create: 2026-10-29
 */

#define HCCL_VM_MODULE "DPU_RING"

#include "dpu_task_ring.h"

#include <chrono>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unistd.h>

#include "operation_data/operation_data_ops.h"
#include "sim_log.h"
#include "storage/query_spec.h"
#include "storage/table_access.h"

namespace HcclSim {
namespace DpuSim {

    namespace {

        /**
         * @brief 消费位点表：commId -> 下一个待消费 ticket（进程内状态，不落库）。
         * @note 消费位点只存于 host 进程；device 侧经槽位 FREE/READY 与生产者协作。
         *       互斥锁串行化执行器线程与清理线程的并发排空。
         */
        std::mutex g_ringHeadMutex;
        std::map<uint64_t, uint64_t> g_ringHeads; /**< commId -> head（下一个待消费 ticket） */

        /** 已就"槽位未就绪"告警过的通信域（防止执行器每轮刷错误日志）。 */
        std::set<uint64_t> g_notReadyWarned;

        /* 消费者等生产者写完的有界自旋上限（正常为微秒级，仅异常路径触发）。 */
        constexpr auto RING_NOT_READY_TIMEOUT = std::chrono::seconds(2);

        /**
         * @brief 有界自旋等待槽位 READY（生产者已写完本槽）。
         * @param state 输入：槽位状态字（跨进程共享内存）。
         * @return true=就绪；false=超时（生产者异常，本轮放弃，由下轮重试）。
         */
        bool WaitEntryReady(const uint32_t* state)
        {
            const auto deadline = std::chrono::steady_clock::now() + RING_NOT_READY_TIMEOUT;
            while (__atomic_load_n(state, __ATOMIC_ACQUIRE) != DPU_DEV_TASK_ENTRY_READY) {
                if (std::chrono::steady_clock::now() > deadline) {
                    return false;
                }
                std::this_thread::yield();
            }
            return true;
        }

    } // namespace

    void DrainDevTaskRing(const DpuMemRegion& region, uint64_t markExclusive)
    {
        if (region.npu2dpu == nullptr) {
            return;
        }
        uint8_t* ring = region.npu2dpu + DPU_DEV_TASK_RING_OFF;
        uint64_t* tailPtr = reinterpret_cast<uint64_t*>(ring);
        const uint64_t published = __atomic_load_n(tailPtr, __ATOMIC_ACQUIRE);
        if (published == 0) {
            return; // 尚无记录（device 从未推送）
        }

        // 排水目标：请求回调前按水位精确排水（markExclusive），其余路径排空全部
        // 已发布记录。水位由 device 在置 flag 前写入（tail 快照，故恒 <= published）。
        const uint64_t target = (markExclusive < published) ? markExclusive : published;

        std::lock_guard<std::mutex> lock(g_ringHeadMutex);
        uint64_t& head = g_ringHeads[region.commId];
        if (head >= target) {
            return; // 水位 <= 已消费位点（陈旧/初始值），空排水
        }

        // 与 device 直写路径同一物理表：device 进程 ppid == 本进程 pid；
        // opDetailId 原样携带 device 推送时刻的值（InsertOpTask 的直写路径也会
        // 在插入时刻覆写为 g_currOpDetailId，语义逐字节一致）。
        const uint32_t pid = static_cast<uint32_t>(getpid());
        // 严格按水位排水：[head, target)。用 published 会把 mark 之后（本请求
        // 回调之后）的设备推送提前落库，与回调直写在同一条设备流上交织，
        // 破坏"落库序 = 设备程序序"（ErrorCode 102 假死锁根因）。
        for (; head < target; ++head) {
            auto* entry = reinterpret_cast<DpuDevTaskRingEntry*>(
                ring + DPU_DEV_TASK_RING_HDR + (head % DPU_DEV_TASK_RING_CAP) * DPU_DEV_TASK_RING_ENTRY_BYTES);
            if (!WaitEntryReady(&entry->state)) {
                if (g_notReadyWarned.insert(region.commId).second) {
                    HCCL_VM_ERROR(
                        "{}: ring entry not ready, commId={:d}, head={:d}, published={:d}, retry next round", __func__,
                        region.commId, head, published);
                }
                return; // 保持 head 不动，下一轮重试
            }
            g_notReadyWarned.erase(region.commId);

            sim::operation::OpTaskTab rec{};
            rec.id = 0;
            rec.pid = pid;
            rec.opDetailId = entry->opDetailId;
            rec.deviceId = entry->meta.deviceId;
            rec.streamId = entry->meta.streamId;
            rec.taskType = static_cast<uint32_t>(entry->meta.taskType);
            const uint8_t* blob = reinterpret_cast<const uint8_t*>(&entry->meta);
            rec.optaskMeta.assign(blob, blob + sizeof(HcclTaskMetaData));

            // 数据已拷出，先释放槽位让 device 复用，再落库（落库失败仅告警，
            // 记录仍按环序前进，避免毒化整环）。
            __atomic_store_n(&entry->state, DPU_DEV_TASK_ENTRY_FREE, __ATOMIC_RELEASE);
            auto inserted
                = HcclSim::Storage::Insert<sim::operation::OpTaskTab>(HcclSim::Storage::TableScope::Process(pid), rec);
            if (!inserted.ok()) {
                HCCL_VM_ERROR(
                    "{}: drain insert failed, commId={:d}, ticket={:d}, diagnostic={}", __func__, region.commId, head,
                    inserted.diagnostic);
            }
        }
    }

    void DiscardDevTaskRingState(uint64_t commId)
    {
        std::lock_guard<std::mutex> lock(g_ringHeadMutex);
        (void)g_ringHeads.erase(commId);
        (void)g_notReadyWarned.erase(commId);
    }

} // namespace DpuSim
} // namespace HcclSim
