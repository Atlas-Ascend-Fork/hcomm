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
 * Description: HostDPU 北向劫持（level1）进程内资源管理实现。
 *              实现 DpuCallbackRegistry / DpuMemRegistry 两个进程级单例，
 *              以及 dpu_level1_common.h 声明的 TLS 上下文与任务 protocol 标注。
 * Create: 2026-10-09
 */

#define HCCL_VM_MODULE "DPU_RES"

#include "dpu_level1_res.h"

#include "hccl_proxy_common.h"
#include "sim_log.h"

namespace HcclSim {
namespace DpuSim {

    namespace {
        /** DPU 执行上下文（线程局部；仅执行器线程在回调期间置位）。 */
        thread_local DpuExecContext g_dpuExecTls;
    } // namespace

    /*==================== TLS 上下文与 protocol 标注 ====================*/

    DpuExecContext& GetDpuExecTls() { return g_dpuExecTls; }

    /** 判断当前线程是否处于 DPU 执行器回调上下文（实现见头文件注释）。 */
    bool IsDpuContext() { return g_dpuExecTls.active; }

    /**
     * @brief 给 DPU 上下文产生的任务打 protocol 标注（ROCE）。
     * @param task 输入输出：任务元数据；仅 DPU 上下文（IsDpuContext）内生效，
     *             仅 MEM_CPY / REDUCE / NOTIFY_RECORD / NOTIFY_WAIT 四类带
     *             protocol 字段的任务被标注，其余类型不修改。
     */
    void ApplyDpuTaskProtocol(HcclTaskMetaData& task)
    {
        if (!IsDpuContext()) {
            return;
        }
        switch (task.taskType) {
            case HccLTaskMetaType::MEM_CPY:
                task.taskData.transMem.protocol = DPU_TASK_PROTOCOL_ROCE;
                break;
            case HccLTaskMetaType::REDUCE:
                task.taskData.reduce.protocol = DPU_TASK_PROTOCOL_ROCE;
                break;
            case HccLTaskMetaType::NOTIFY_RECORD:
            case HccLTaskMetaType::NOTIFY_WAIT:
                task.taskData.notify.protocol = DPU_TASK_PROTOCOL_ROCE;
                break;
            default:
                break;
        }
    }

    /*==================== DPU notify 事件序号分配 ====================*/

    namespace {
        std::mutex g_seqMutex;                       /**< 序号表互斥锁 */
        std::map<uint64_t, uint32_t> g_dpuNotifySeq; /**< 槽位行id -> 下一事件序号 */
    } // namespace

    /** 编码一次 DPU RECORD/WAIT 事件的唯一 notify id（布局见 dpu_level1_common.h 常量注释）。 */
    uint64_t EncodeDpuNotifyTaskId(uint64_t slotNotifyId, uint32_t seq)
    {
        return DPU_NOTIFY_ID_TAG | (slotNotifyId << 32) | seq;
    }

    /** 分配指定槽位的下一个事件序号（进程内单调递增，mutex 保护）。 */
    uint32_t NextDpuNotifySeq(uint64_t slotNotifyId)
    {
        std::lock_guard<std::mutex> lock(g_seqMutex);
        return g_dpuNotifySeq[slotNotifyId]++;
    }

    /*==================== DPU 任务地址翻译 ====================*/

    namespace {
        /**
         * @brief 将单个端点地址从设备映射空间翻译为 VM 虚拟地址（失败时保留 0 并告警）。
         * @param offset   输入：设备映射地址（dev_mapped 空间）。
         * @param deviceId 输入：该端点属主设备（Normalize 后的 DB Device 主键）。
         * @param tag      输入：日志标签（src/dst）。
         * @return VM 虚拟地址；翻译失败返回 0（与南向行为一致，任务仍可记录定位问题）。
         */
        uint64_t TranslateOneEndpoint(uint64_t offset, uint32_t deviceId, const char* tag)
        {
            const uint64_t virAddr = sim::TransRemoteAddrToVirtualByDeviceId(offset, deviceId);
            if (virAddr == 0) {
                HCCL_VM_ERROR("{}: translate failed, {} addr=0x{:x}, deviceId={:d}", __func__, tag, offset, deviceId);
            }
            return virAddr;
        }
    } // namespace

    /** 翻译 DPU 任务的端点地址（dev_mapped → VM 虚拟；语义见 dpu_level1_common.h 声明注释）。 */
    void TranslateDpuTaskAddress(HcclTaskMetaData& task)
    {
        // DPU 数据面地址来自设备侧序列化的 DPURunInfo（dev_mapped 空间），checker 布局
        // 在 VM 虚拟（start_ptr）空间；与南向 ibv_dpu_stub 的翻译路径同源。
        if (task.taskType == HccLTaskMetaType::MEM_CPY) {
            task.taskData.transMem.srcOffset
                = TranslateOneEndpoint(task.taskData.transMem.srcOffset, task.taskData.transMem.srcDeviceId, "src");
            task.taskData.transMem.dstOffset
                = TranslateOneEndpoint(task.taskData.transMem.dstOffset, task.taskData.transMem.dstDeviceId, "dst");
            return;
        }
        if (task.taskType == HccLTaskMetaType::REDUCE) {
            task.taskData.reduce.srcOffset
                = TranslateOneEndpoint(task.taskData.reduce.srcOffset, task.taskData.reduce.srcDeviceId, "src");
            task.taskData.reduce.dstOffset
                = TranslateOneEndpoint(task.taskData.reduce.dstOffset, task.taskData.reduce.dstDeviceId, "dst");
        }
    }

    /*==================== DpuCallbackRegistry ====================*/

    DpuCallbackRegistry& DpuCallbackRegistry::Instance()
    {
        static DpuCallbackRegistry instance;
        return instance;
    }

    /** 注册 DPU 回调（insert-if-absent；语义与边界见 dpu_level1_res.h 类注释）。 */
    bool DpuCallbackRegistry::Register(uint64_t commId, const std::string& msgTag, DpuCallback cb)
    {
        if (cb == nullptr || msgTag.empty()) {
            HCCL_VM_ERROR(
                "{}: invalid args, commId={:d}, msgTag empty={}, cb={:p}", __func__, commId, msgTag.empty(),
                reinterpret_cast<void*>(cb));
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto key = std::make_pair(commId, msgTag);
        if (callbacks_.find(key) != callbacks_.end()) {
            HCCL_VM_INFO("{}: callback already registered, commId={:d}, msgTag={}", __func__, commId, msgTag);
            return true;
        }
        callbacks_.emplace(key, cb);
        HCCL_VM_INFO("{} success, commId={:d}, msgTag={}, total={:d}", __func__, commId, msgTag, callbacks_.size());
        return true;
    }

    /** 注销指定 (commId, msgTag) 的回调。 */
    bool DpuCallbackRegistry::UnRegister(uint64_t commId, const std::string& msgTag)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto erased = callbacks_.erase(std::make_pair(commId, msgTag));
        HCCL_VM_INFO("{}, commId={:d}, msgTag={}, erased={:d}", __func__, commId, msgTag, erased);
        return erased > 0;
    }

    /** 按 (commId, msgTag) 查询回调，命中回填 out。 */
    bool DpuCallbackRegistry::Lookup(uint64_t commId, const std::string& msgTag, DpuCallback& out)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = callbacks_.find(std::make_pair(commId, msgTag));
        if (it == callbacks_.end()) {
            return false;
        }
        out = it->second;
        return true;
    }

    /** 移除指定通信域的全部回调（通信域销毁路径），返回移除条数。 */
    uint32_t DpuCallbackRegistry::RemoveByCommId(uint64_t commId)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        uint32_t removed = 0;
        for (auto it = callbacks_.begin(); it != callbacks_.end();) {
            if (it->first.first == commId) {
                it = callbacks_.erase(it);
                ++removed;
            } else {
                ++it;
            }
        }
        HCCL_VM_INFO("{}: commId={:d}, removed={:d}", __func__, commId, removed);
        return removed;
    }

    /*==================== DpuMemRegistry ====================*/

    DpuMemRegistry& DpuMemRegistry::Instance()
    {
        static DpuMemRegistry instance;
        return instance;
    }

    /** 登记一个通信域的 DPU 共享内存区域（同 commId 重复登记返回 false）。 */
    bool DpuMemRegistry::Add(const DpuMemRegion& region)
    {
        if (region.npu2dpu == nullptr || region.dpu2npu == nullptr || region.halfSize == 0) {
            HCCL_VM_ERROR(
                "{}: invalid region, commId={:d}, npu2dpu={:p}, dpu2npu={:p}, halfSize={:d}", __func__, region.commId,
                static_cast<void*>(region.npu2dpu), static_cast<void*>(region.dpu2npu), region.halfSize);
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (regions_.find(region.commId) != regions_.end()) {
            HCCL_VM_WARN("{}: region already exists, commId={:d}", __func__, region.commId);
            return false;
        }
        regions_.emplace(region.commId, region);
        HCCL_VM_INFO(
            "{} success, commId={:d}, npu2dpu={:p}, halfSize={:d}", __func__, region.commId,
            static_cast<void*>(region.npu2dpu), region.halfSize);
        return true;
    }

    /** 摘除指定通信域的内存区域并取出内容（供调用方释放内存）。 */
    bool DpuMemRegistry::Remove(uint64_t commId, DpuMemRegion& out)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = regions_.find(commId);
        if (it == regions_.end()) {
            return false;
        }
        out = it->second;
        regions_.erase(it);
        HCCL_VM_INFO("{}: commId={:d} removed", __func__, commId);
        return true;
    }

    /** 获取当前登记区域的快照（副本），供执行器无锁轮询。 */
    std::vector<DpuMemRegion> DpuMemRegistry::Snapshot()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<DpuMemRegion> snapshot;
        snapshot.reserve(regions_.size());
        for (const auto& pair : regions_) {
            snapshot.push_back(pair.second);
        }
        return snapshot;
    }

} // namespace DpuSim
} // namespace HcclSim
