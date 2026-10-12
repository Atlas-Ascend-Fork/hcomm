/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define HCCL_VM_MODULE "CCU_L1_RES"

#include "ccu_level1_res.h"

#include <cerrno>
#include <csignal>
#include <cstring>

#include "sim_log.h"
#include "runtime_state/sim_models.h"
#include "sim_common_api.h"
#include "level1_proxy_common.h"
#include "hccl_proxy_common.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/table_access.h"

namespace HcclSim {
namespace CcuSim {

    namespace {
        /* V1(A5/950) 容量（与 level2 SetCcuV1ResourceBasicInfo 一致） */
        constexpr CcuCapacity CCU_V1_CAPACITY = {3072, 3072, 1024, 1536, 200, 16, 32 * 1024};
        /* V2(A6/960) 容量 */
        constexpr CcuCapacity CCU_V2_CAPACITY = {16384, 0, 1024, 1536, 512, 16, 32 * 1024};
        /* die 基址（与 level2 resourceAddr 一致） */
        constexpr uint64_t DIE_BASE_ADDR[2] = {0x123123123ULL, 0x456456456ULL};
        /* 资源单元步长（B，XN/CKE/GSA 均为 8） */
        constexpr uint32_t RES_STRIDE = 8;
        /* 版本化区偏移（权威来源：hcomm ccu_res_specs.{h,cc}）
         *   V1 = CCUM(0x800000) + INS(0x100000) → GSA(0x8000) → XN(0x8000) → CKE
         *   V2 = CCUM(0x0)      + INS(0x100000) → XN(0x40000) → CKE
         * V2 偏移与 checker findTypeByAddr / runner ccu_resource_manager 完全一致。 */
        constexpr uint64_t V1_GSA_OFFSET = 0x900000ULL;
        constexpr uint64_t V1_XN_OFFSET = 0x908000ULL;
        constexpr uint64_t V1_CKE_OFFSET = 0x910000ULL;
        constexpr uint64_t V2_XN_OFFSET = 0x100000ULL;
        constexpr uint64_t V2_CKE_OFFSET = 0x140000ULL;

        /** @brief 进程是否仍存活（用于交换表 run 内清理：只删"写方进程已消失"的上一轮遗留行）。 */
        bool IsProcessAlive(uint64_t pid)
        {
            if (pid == 0) {
                return false;
            }
            if (kill(static_cast<pid_t>(pid), 0) == 0) {
                return true;
            }
            return errno == EPERM; // 进程存在但无信号权限 → 视为存活
        }
    } // namespace

    /** @brief 获取资源管理器进程级单例（首次调用时构造）。 */
    CcuResMgr& CcuResMgr::Instance()
    {
        static CcuResMgr instance;
        return instance;
    }

    /** @brief 签发下一个 64bit 资源句柄（进程内自增，恒非零）。 */
    uint64_t CcuResMgr::NextHandle() { return nextHandle_++; }

    /*==================== 实例生命周期 ====================*/

    uint64_t CcuResMgr::CreateIns(uint32_t dieMask)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // 交换表 run 内清理：DB(hccl_sim.db) 跨 run 保留，而交换键的 round 从 1 重新计数，
        // 若不清会命中上一轮遗留的同 round 行。本进程首次建实例时清理一次（早于任何 launch/publish）。
        // **只删"写方进程已不存在"的行**：两进程启动无同步，可能对端已抢跑并发布了本轮的行；
        // 无条件 DeleteAll 会误删对端刚写入的行，导致读方查不到 → 等满超时（实测 3 次×10s）。
        if (!exchangeTableCleared_) {
            const auto rows = sim::runtime::Db::GetByPred<sim::runtime::CcuSyncResTab>(
                HcclSim::Storage::All<sim::runtime::CcuSyncResTab>());
            uint32_t removed = 0;
            if (rows.ok() && rows.value.has_value()) {
                for (const auto& row : *rows.value) {
                    if (!IsProcessAlive(row.ownerPid)) {
                        (void)sim::runtime::Db::Delete<sim::runtime::CcuSyncResTab>(
                            HcclSim::Storage::Eq(&sim::runtime::CcuSyncResTab::id, row.id));
                        ++removed;
                    }
                }
            }
            exchangeTableCleared_ = true;
            if (removed != 0) {
                HCCL_VM_INFO("ccu sync table run-start cleanup, removedStaleRows={}", removed);
            }
        }
        const uint64_t handle = NextHandle();
        SimCcuInsDesc desc;
        desc.insHandle = handle;
        desc.dieMask = dieMask;
        insPool_[handle] = desc;
        HCCL_VM_INFO("ccu ins created, insHandle={}, dieMask=0x{:x}", handle, dieMask);
        return handle;
    }

    /**
     * @brief 销毁 CCU 实例：级联标记其描述符/清零预约，并擦除实例条目、清理指向它的通信域绑定。
     */
    bool CcuResMgr::DestroyIns(uint64_t insHandle)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = insPool_.find(insHandle);
        if (it == insPool_.end()) {
            HCCL_VM_WARN("ccu ins destroy: insHandle={} not found", insHandle);
            return false;
        }
        // 级联清理：标记实例/描述符、清零预约，并**擦除实例条目 + 清理指向它的通信域绑定**。
        // 否则 HcclCommQueryAssignedCcuIns 会继续返回已销毁句柄、hccl 复用死实例（review #1.3）。
        it->second.destroyed = true;
        for (auto& kv : resDescPool_) {
            if (kv.second.insHandle == insHandle) {
                kv.second.destroyed = true;
            }
        }
        for (auto& kv : acqVarPool_) {
            if (kv.second.insHandle == insHandle) {
                kv.second.num = 0;
            }
        }
        for (auto bindIt = commBinding_.begin(); bindIt != commBinding_.end();) {
            if (bindIt->second == insHandle) {
                HCCL_VM_INFO("ccu ins destroy: unbind commId={} from destroyed insHandle={}", bindIt->first, insHandle);
                bindIt = commBinding_.erase(bindIt);
            } else {
                ++bindIt;
            }
        }
        insPool_.erase(it);
        HCCL_VM_INFO("ccu ins destroyed, insHandle={}", insHandle);
        return true;
    }

    /** @brief 实例句柄是否有效（存在且未被销毁）。 */
    bool CcuResMgr::IsValidIns(uint64_t insHandle) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = insPool_.find(insHandle);
        return (it != insPool_.end()) && !it->second.destroyed;
    }

    /*==================== 通信域绑定 ====================*/

    bool CcuResMgr::BindInsToComm(uint64_t commId, uint64_t insHandle)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // 只允许绑定仍存活的实例（对齐接口契约"实例无效返回 HCCL_E_PARA"；review #1.3）。
        const auto insIt = insPool_.find(insHandle);
        if (insIt == insPool_.end() || insIt->second.destroyed) {
            HCCL_VM_ERROR("ccu ins bind rejected: insHandle=0x{:x} invalid or destroyed", insHandle);
            return false;
        }
        auto it = commBinding_.find(commId);
        if (it != commBinding_.end()) {
            if (it->second == insHandle) {
                return true; /* 幂等 */
            }
            HCCL_VM_ERROR("ccu ins bind rejected: commId={} already bound to insHandle={}", commId, it->second);
            return false;
        }
        commBinding_[commId] = insHandle;
        HCCL_VM_INFO("ccu ins bound, commId={}, insHandle={}", commId, insHandle);
        return true;
    }

    /** @brief 解绑通信域与 CCU 实例的映射（当前无调用者，见检视 1.3 遗留项）。 */
    bool CcuResMgr::UnbindInsFromComm(uint64_t commId)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = commBinding_.find(commId);
        if (it == commBinding_.end()) {
            return false;
        }
        HCCL_VM_INFO("ccu ins unbound, commId={}, insHandle={}", commId, it->second);
        commBinding_.erase(it);
        return true;
    }

    /** @brief 查询通信域绑定的实例句柄；若绑定项指向的实例已失效则视为未绑定。 */
    bool CcuResMgr::QueryBoundIns(uint64_t commId, uint64_t& insHandle) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = commBinding_.find(commId);
        if (it == commBinding_.end()) {
            return false;
        }
        // 防御：绑定项指向的实例必须仍存活；否则视为未绑定（防脏映射返回已销毁句柄，review #1.3）。
        const auto insIt = insPool_.find(it->second);
        if (insIt == insPool_.end() || insIt->second.destroyed) {
            HCCL_VM_WARN("ccu query bound ins: commId={} bound to invalid insHandle=0x{:x}", commId, it->second);
            return false;
        }
        insHandle = it->second;
        return true;
    }

    /*==================== 资源描述符（多资源类型） ====================*/

    uint64_t CcuResMgr::CreateResDesc(uint32_t dieId)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const uint64_t handle = NextHandle();
        SimCcuInsResDesc desc;
        desc.descHandle = handle;
        desc.dieId = dieId;
        resDescPool_[handle] = desc;
        HCCL_VM_INFO("ccu ins res desc created, descHandle={}, dieId={}", handle, dieId);
        return handle;
    }

    /** @brief 销毁资源描述符（仅置 destroyed 标记，不擦除条目）。 */
    bool CcuResMgr::DestroyResDesc(uint64_t descHandle)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = resDescPool_.find(descHandle);
        if (it == resDescPool_.end()) {
            HCCL_VM_WARN("ccu ins res desc destroy: descHandle={} not found", descHandle);
            return false;
        }
        it->second.destroyed = true;
        return true;
    }

    /** @brief 设置描述符某资源类型的请求量 reqNum（描述符缺失/已销毁时拒绝）。 */
    bool CcuResMgr::SetResDescNum(uint64_t descHandle, uint32_t resType, uint32_t reqNum)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = resDescPool_.find(descHandle);
        if (it == resDescPool_.end() || it->second.destroyed) {
            HCCL_VM_ERROR("ccu ins res desc set-num: descHandle={} not found or destroyed", descHandle);
            return false;
        }
        if (resType >= CCU_RES_TYPE_MAX) {
            HCCL_VM_ERROR("ccu ins res desc set-num: resType={} out of range", resType);
            return false;
        }
        it->second.reqNum[resType] = reqNum;
        return true;
    }

    /** @brief 读取描述符某资源类型的请求量 reqNum。 */
    bool CcuResMgr::QueryResDescNum(uint64_t descHandle, uint32_t resType, uint32_t& reqNum) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = resDescPool_.find(descHandle);
        if (it == resDescPool_.end() || it->second.destroyed || resType >= CCU_RES_TYPE_MAX) {
            return false;
        }
        reqNum = it->second.reqNum[resType];
        return true;
    }

    /** @brief 读取描述符所属的 dieId。 */
    bool CcuResMgr::QueryResDescDieId(uint64_t descHandle, uint32_t& dieId) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = resDescPool_.find(descHandle);
        if (it == resDescPool_.end() || it->second.destroyed) {
            return false;
        }
        dieId = it->second.dieId;
        return true;
    }

    /**
     * @brief 按请求量计算实批量 allocNum = min(reqNum, remaining)，并回填描述符所属实例句柄。
     * @return 描述符缺失/已销毁返回 false。
     */
    bool CcuResMgr::QueryResDescIns(uint64_t insHandle, uint64_t descHandle)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = resDescPool_.find(descHandle);
        if (it == resDescPool_.end() || it->second.destroyed) {
            return false;
        }
        it->second.insHandle = insHandle;
        /* 按类型计算实批 = min(请求, 剩余) */
        const uint32_t dieId = it->second.dieId;
        const CcuCapacity cap = GetCapacity();
        for (uint32_t t = 0; t < CCU_RES_TYPE_MAX; t++) {
            if (it->second.reqNum[t] == 0) {
                continue;
            }
            uint32_t remaining = 0;
            switch (t) {
                case 2:
                    remaining = RemainingXn(dieId);
                    break; /* VARIABLE */
                case 4:
                    remaining = RemainingCke(dieId);
                    break; /* EVENT */
                case 1:
                    remaining = cap.msNum;
                    break; /* CCU_BUF */
                case 0:
                    remaining = cap.loopEngineNum;
                    break; /* LOOP */
                case 6:
                    remaining = cap.instructionNum;
                    break; /* INSTRUCTION */
                default:
                    remaining = it->second.reqNum[t];
                    break;
            }
            it->second.allocNum[t] = (it->second.reqNum[t] <= remaining) ? it->second.reqNum[t] : remaining;
        }
        return true;
    }

    /** @brief 回填描述符各资源类型的剩余量 remainNum（供 hccl 做 CCU→AICPU 回退判定）。 */
    bool CcuResMgr::QueryRemainResDesc(uint64_t descHandle)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = resDescPool_.find(descHandle);
        if (it == resDescPool_.end() || it->second.destroyed) {
            return false;
        }
        const uint32_t dieId = it->second.dieId;
        const CcuCapacity cap = GetCapacity();
        for (uint32_t t = 0; t < CCU_RES_TYPE_MAX; t++) {
            uint32_t remain = 0;
            switch (t) {
                case 2:
                    remain = RemainingXn(dieId);
                    break;
                case 4:
                    remain = RemainingCke(dieId);
                    break;
                case 1:
                    remain = cap.msNum;
                    break;
                case 0:
                    remain = cap.loopEngineNum;
                    break;
                case 6:
                    remain = cap.instructionNum;
                    break;
                default:
                    remain = 0;
                    break;
            }
            it->second.remainNum[t] = remain;
        }
        HCCL_VM_INFO("ccu remain res desc, descHandle={}, dieId={}", descHandle, dieId);
        return true;
    }

    /*==================== 查询模式资源统计 ====================*/

    bool CcuResMgr::QueryResourceReq(
        uint64_t descHandle, const void* kernelFunc, const void** kernelArgs, uint32_t argNum, uint32_t dieId)
    {
        (void)kernelFunc;
        (void)kernelArgs;
        (void)argNum;
        (void)dieId;
        /* TODO: 以查询模式执行 kernelFunc 统计资源——需要与录制引擎配合：
         * 1. 开临时录制事务（queryOnly，不注册）
         * 2. 执行 kernelFunc
         * 3. 从录制上下文提取 handleSeq 按类型计数
         * 4. 写入 desc.reqNum[]
         * 当前阶段 desc.reqNum 由 hccl 侧 SetNum 预估，已覆盖回退判定需求。 */
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = resDescPool_.find(descHandle);
        if (it == resDescPool_.end() || it->second.destroyed) {
            return false;
        }
        HCCL_VM_INFO("ccu query resource req (pass-through), descHandle=0x{:x}", descHandle);
        return true;
    }

    /*==================== 实例级变量/事件 ====================*/

    uint64_t CcuResMgr::VariableAlloc(uint64_t insHandle, uint32_t dieId, uint32_t num)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // 已持 mutex_，禁止调用 IsValidIns（其内部对同一把非递归 mutex_ 二次加锁会自死锁），锁内直查。
        const auto insIter = insPool_.find(insHandle);
        if (insIter == insPool_.end() || insIter->second.destroyed) {
            HCCL_VM_ERROR("ccu variable alloc: insHandle={} invalid", insHandle);
            return 0;
        }
        if (dieId >= 2 || num == 0) {
            HCCL_VM_ERROR("ccu variable alloc: dieId={} or num={} invalid", dieId, num);
            return 0;
        }
        if (xnAllocated_[dieId] + num > GetCapacity().xnNum) {
            HCCL_VM_ERROR(
                "ccu variable alloc: xn exhausted, dieId={}, allocated={}, reqNum={}, capacity={}", dieId,
                xnAllocated_[dieId], num, GetCapacity().xnNum);
            return 0;
        }
        const uint64_t handle = NextHandle();
        SimCcuAcquiredVar acq;
        acq.acqHandle = handle;
        acq.insHandle = insHandle;
        acq.dieId = dieId;
        acq.num = num;
        acq.isEvent = false;
        acq.baseVa = GetXnBaseAddr(dieId) + static_cast<uint64_t>(xnAllocated_[dieId]) * RES_STRIDE;
        acqVarPool_[handle] = acq;
        xnAllocated_[dieId] += num;
        HCCL_VM_INFO(
            "ccu variable acquired, acqHandle={}, insHandle={}, dieId={}, num={}, baseVa=0x{:x}", handle, insHandle,
            dieId, num, acq.baseVa);
        return handle;
    }

    /**
     * @brief 预约一段连续 Event(CKE) 资源并记录 VA：baseVa = CKE 区基址 + 分配水位*步长。
     * @return 预约句柄；实例失效/参数非法/容量不足返回 0。
     */
    uint64_t CcuResMgr::EventAlloc(uint64_t insHandle, uint32_t dieId, uint32_t num)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // 同 VariableAlloc：锁内直查，避免 IsValidIns 二次加锁自死锁。
        const auto insIter = insPool_.find(insHandle);
        if (insIter == insPool_.end() || insIter->second.destroyed) {
            HCCL_VM_ERROR("ccu event alloc: insHandle={} invalid", insHandle);
            return 0;
        }
        if (dieId >= 2 || num == 0) {
            HCCL_VM_ERROR("ccu event alloc: dieId={} or num={} invalid", dieId, num);
            return 0;
        }
        if (ckeAllocated_[dieId] + num > GetCapacity().ckeNum) {
            HCCL_VM_ERROR(
                "ccu event alloc: cke exhausted, dieId={}, allocated={}, reqNum={}, capacity={}", dieId,
                ckeAllocated_[dieId], num, GetCapacity().ckeNum);
            return 0;
        }
        const uint64_t handle = NextHandle();
        SimCcuAcquiredVar acq;
        acq.acqHandle = handle;
        acq.insHandle = insHandle;
        acq.dieId = dieId;
        acq.num = num;
        acq.isEvent = true;
        acq.baseVa = GetCkeBaseAddr(dieId) + static_cast<uint64_t>(ckeAllocated_[dieId]) * RES_STRIDE;
        acqVarPool_[handle] = acq;
        ckeAllocated_[dieId] += num;
        HCCL_VM_INFO(
            "ccu event acquired, acqHandle={}, insHandle={}, dieId={}, num={}, baseVa=0x{:x}", handle, insHandle, dieId,
            num, acq.baseVa);
        return handle;
    }

    /** @brief 取 Variable 预约第 index 个元素的 VA（越界/类型不符/已失效返回 false）。 */
    bool CcuResMgr::VariableGetAddr(uint64_t acqHandle, uint32_t index, uint64_t& va) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = acqVarPool_.find(acqHandle);
        if (it == acqVarPool_.end() || it->second.num == 0 || it->second.isEvent || index >= it->second.num) {
            return false;
        }
        va = it->second.baseVa + index * sizeof(uint64_t);
        return true;
    }

    /** @brief 取 Event 预约第 index 个元素的 VA（越界/类型不符/已失效返回 false）。 */
    bool CcuResMgr::EventGetAddr(uint64_t acqHandle, uint32_t index, uint64_t& va) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = acqVarPool_.find(acqHandle);
        if (it == acqVarPool_.end() || it->second.num == 0 || !it->second.isEvent || index >= it->second.num) {
            return false;
        }
        va = it->second.baseVa + index * RES_STRIDE;
        return true;
    }

    /*==================== 容量查询 ====================*/

    SimCcuVersion CcuResMgr::GetCcuVersion() const
    {
        const uint64_t deviceId = sim::runtime::GetCurrDeviceId();
        const auto optDevice = sim::runtime::Db::GetById<sim::runtime::Device>(deviceId);
        if (!optDevice.ok()) {
            HCCL_VM_WARN("ccu version: device id={} not found, fallback to V1", deviceId);
            return SimCcuVersion::V1;
        }
        if (strcmp(optDevice->soc_version, "Ascend960") == 0) {
            return SimCcuVersion::V2;
        }
        if (strcmp(optDevice->soc_version, "Ascend950") != 0) {
            HCCL_VM_WARN("ccu version: unknown soc_version='{}', fallback to V1", optDevice->soc_version);
        }
        return SimCcuVersion::V1;
    }

    /** @brief 按 CCU 版本返回资源地址布局（V1 含 GSA 区，V2 无 GSA）。 */
    CcuResLayout CcuResMgr::GetLayout() const
    {
        CcuResLayout layout;
        layout.version = GetCcuVersion();
        layout.xnStride = RES_STRIDE;
        layout.gsaStride = RES_STRIDE;
        layout.ckeStride = RES_STRIDE;
        if (layout.version == SimCcuVersion::V2) {
            layout.xnOffset = V2_XN_OFFSET;
            layout.ckeOffset = V2_CKE_OFFSET;
            layout.gsaOffset = 0; // V2 无 GSA
        } else {
            layout.gsaOffset = V1_GSA_OFFSET;
            layout.xnOffset = V1_XN_OFFSET;
            layout.ckeOffset = V1_CKE_OFFSET;
        }
        return layout;
    }

    /** @brief XN 区基址 = dieBase + xnOffset。 */
    uint64_t CcuResMgr::GetXnBaseAddr(uint32_t dieId) const { return GetDieBaseAddr(dieId) + GetLayout().xnOffset; }

    /** @brief GSA 区基址 = dieBase + gsaOffset（V2 无 GSA 返回 0）。 */
    uint64_t CcuResMgr::GetGsaBaseAddr(uint32_t dieId) const
    {
        const CcuResLayout layout = GetLayout();
        if (layout.gsaOffset == 0) {
            return 0; // V2 无 GSA
        }
        return GetDieBaseAddr(dieId) + layout.gsaOffset;
    }

    /** @brief CKE 区基址 = dieBase + ckeOffset。 */
    uint64_t CcuResMgr::GetCkeBaseAddr(uint32_t dieId) const { return GetDieBaseAddr(dieId) + GetLayout().ckeOffset; }

    /** @brief 当前设备的 CCU 容量（按 soc_version 分 V1/V2 档）。 */
    CcuCapacity CcuResMgr::GetCapacity() const
    {
        if (GetCcuVersion() == SimCcuVersion::V2) {
            return CCU_V2_CAPACITY;
        }
        return CCU_V1_CAPACITY;
    }

    /** @brief per-die 固定基址（die0=0x123123123, die1=0x456456456）；dieId 越界回退 die0。 */
    uint64_t CcuResMgr::GetDieBaseAddr(uint32_t dieId) const
    {
        if (dieId >= 2) {
            return DIE_BASE_ADDR[0];
        }
        return DIE_BASE_ADDR[dieId];
    }

    /** @brief 指定 die 的剩余 XN 数 = capacity - 分配水位（dieId 越界返回 0）。 */
    uint32_t CcuResMgr::RemainingXn(uint32_t dieId) const
    {
        if (dieId >= 2) {
            return 0;
        }
        const uint32_t cap = GetCapacity().xnNum;
        return (xnAllocated_[dieId] < cap) ? (cap - xnAllocated_[dieId]) : 0;
    }

    /** @brief 指定 die 的剩余 GSA 数（V2 容量为 0）。 */
    uint32_t CcuResMgr::RemainingGsa(uint32_t dieId) const
    {
        if (dieId >= 2) {
            return 0;
        }
        const uint32_t cap = GetCapacity().gsaNum; // V2 = 0
        return (gsaAllocated_[dieId] < cap) ? (cap - gsaAllocated_[dieId]) : 0;
    }

    /** @brief 指定 die 的剩余 CKE 数。 */
    uint32_t CcuResMgr::RemainingCke(uint32_t dieId) const
    {
        if (dieId >= 2) {
            return 0;
        }
        const uint32_t cap = GetCapacity().ckeNum;
        return (ckeAllocated_[dieId] < cap) ? (cap - ckeAllocated_[dieId]) : 0;
    }

} // namespace CcuSim
} // namespace HcclSim
