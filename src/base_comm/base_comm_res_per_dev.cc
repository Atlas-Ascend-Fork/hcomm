/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base_comm_res_per_dev.h"

#include "hccl_common.h"
#include "adapter_rts_common.h"

// orion 通用平台层单例（legacy950，libhccl_v2.so）
#include "tp_manager.h"
#include "endpoint_monitor.h"
#include "ccu_component.h"
#include "ccu_res_batch_allocator_legacy.h"
#include "ccu_context_mgr_imp.h"
#include "ccu_res_specs_legacy.h"
// 开源开放 ccu 单例（base_comm 自有）
#include "ccu_comp.h"
#include "ccu_res_specs.h"
#include "ccu_pfe_cfg_mgr.h"
#include "resources/ccu/ccu_device/ccu_res_batch_allocator.h"

namespace hcomm {
BaseCommResPerDev::BaseCommResPerDev(uint32_t devPhyId, bool skipAnchor) : devPhyId_(devPhyId)
{
    // 三个 Proc 节点为值成员，随构造建齐（构造均平凡，不触设备/legacy 单例）
    if (!skipAnchor) {
        AnchorDeviceSingletons(devPhyId_);
    }
}

// 设备级 legacy 依赖唯一清单：树根构造期备份槽预热锁死析构序，子树构造期按本槽再调（正常路径 no-op）
// 不经窄访问器（构造期重入 FindOrCreate 内部锁死锁）；依据见 README R2/R8
void AnchorDeviceSingletons(uint32_t devPhyId)
{
    Hccl::TpManager::GetInstance(devPhyId);
    (void)EndpointMonitor::GetHolder(devPhyId);

    if (devPhyId == MAX_MODULE_DEVICE_NUM) {
        // 备份槽位无真实设备，跳过 phyId->logicId 转换（VM/在位卡不足时转换必打 ACL [error]），直取数组备份槽
        // 入参语义是 logicId，但槽位 65 即备份槽（合法 logicId 只到 64），phyId/logicId 两种解读收敛到同一槽
        (void)Hccl::CcuResSpecifications::GetInstance(static_cast<int32_t>(devPhyId));
    } else {
        // 探测性转换：设备在位则锚定规格单例；失败（设备不存在）跳过，退化为 legacy 首触时序
        uint32_t userDevId = 0;
        if (hrtTryGetDeviceIndexByPhyId(devPhyId, userDevId) == HCCL_SUCCESS) {
            (void)Hccl::CcuResSpecifications::GetInstance(static_cast<int32_t>(userDevId));
        } else {
            HCCL_WARNING(
                "[AnchorDeviceSingletons] hrtTryGetDeviceIndexByPhyId failed, devPhyId[%u], "
                "skip anchoring Hccl::CcuResSpecifications (fallback to legacy first-touch order)",
                devPhyId);
        }
    }
    Hccl::CcuComponent::GetInstance(devPhyId);
    Hccl::CcuResBatchAllocator::GetInstance(devPhyId);
    Hccl::CtxMgrImp::GetInstance(devPhyId);

    // 树值成员（HccpTlvHdcMgr/TpMgr/CCU 三件套）无需首触，只锚定仍为单例的四个：
    // 950 混跑走 legacy（Hccl::）路径用不到它们，960 开源路径/UT 会真实首触，锚定锁死其析构序
    CcuResSpecifications::GetInstance(devPhyId);
    CcuComponent::GetInstance(devPhyId);
    CcuResBatchAllocator::GetInstance(devPhyId);
    CcuPfeCfgMgr::GetInstance(devPhyId);
}

BaseCommResPerDev& BaseCommResPerDevMgr::FindOrCreate(uint32_t devPhyId)
{
    // 越界钳位 65 号备份设备（恢复 upstream InitDevice 语义；65 合法静默，>65 告警）
    if (devPhyId > MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[BaseCommResPerDevMgr][%s] devPhyId[%u] out of range, clamp to backup device %u.", __func__, devPhyId,
            MAX_MODULE_DEVICE_NUM);
        devPhyId = MAX_MODULE_DEVICE_NUM;
    }
    // 快路径：无锁读已发布槽位
    if (auto* p = published_[devPhyId].load(std::memory_order_acquire); p != nullptr) {
        return *p;
    }
    // 锁外首建：构造含锚定与设备探测、耗时不可控，不持全局锁——避免阻塞其它设备首建、杜绝构造期重入
    // 自死锁；锚定幂等（GetInstance 语义），并发重复构造的迟到副本随作用域在锁释放后析构丢弃
    auto node = std::make_unique<BaseCommResPerDev>(devPhyId, false);
    // 慢路径：锁内只做复查与发布（deinited_ 必须锁内复查，堵死"构造期间发生 DeInit"的窗口）
    std::lock_guard<std::mutex> lock(mtx_);
    auto& slot = active_[devPhyId];
    if (slot != nullptr) {
        return *slot; // 并发首建迟到：丢弃自构副本，返回既有实例
    }
    // DeInit 后拒绝再建：清理动作链已停跑，新建子树的设备资源无人归还，返回惰性兜底实例
    if (deinited_.load()) {
        HCCL_WARNING(
            "[BaseCommResPerDevMgr][%s] already deinited, reject creating device subtree, devPhyId[%u].", __func__,
            devPhyId);
        return reserve_;
    }
    // 入槽即发布（强异常保证：构造抛异常不留残留槽）
    slot = std::move(node);
    published_[devPhyId].store(slot.get(), std::memory_order_release);
    return *slot;
}

std::vector<BaseCommResPerDev*> BaseCommResPerDevMgr::SnapshotActive()
{
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<BaseCommResPerDev*> snapshot;
    for (const auto& slot : active_) {
        if (slot != nullptr) {
            snapshot.push_back(slot.get());
        }
    }
    return snapshot;
}

void BaseCommResPerDevMgr::RetireAll()
{
    std::lock_guard<std::mutex> lock(mtx_);
    for (uint32_t i = 0; i < SLOT_NUM; ++i) {
        published_[i].store(nullptr, std::memory_order_release);
        if (active_[i] != nullptr) {
            retired_[i] = std::move(active_[i]);
        }
    }
}

} // namespace hcomm
