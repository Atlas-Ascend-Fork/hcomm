/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_mgr.h"
#include "cluster_monitor.h"
#include "res_mgr_c_adpt_pub.h"
#include "hcom_common.h"
#include "shared_jetty_channel_pool.h"
#include "dpu_thread_service.h" // DpuThreadService 完整类型（GetDpuTaskExpMem/GetDpuDevId 转发）

namespace hccl {

CollCommMgr& CollCommMgr::GetInstance()
{
    // 确保 base_comm 层单例先于 CollCommMgr 构造，从而后于 CollCommMgr 析构，
    // 避免 ~CollCommMgr() 访问已销毁的 base_comm 单例。
    // magic static 保证仅执行一次，避免热路径重复调用和并发访问。
    // 使用 init-capture 捕获外层函数名：__func__在lambda体内输出在不同 gcc 版本下行为不同，捕获可消除差异。
    static const bool baseCommReady = [func = __func__]() {
        s32 devLogicId = 0;
        HcclResult ret = hrtGetDevice(&devLogicId);
        uint32_t devPhyId = 0U;
        if (ret == HCCL_SUCCESS) {
            ret = hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(devLogicId), devPhyId);
        }
        if (ret != HCCL_SUCCESS) {
            // devPhyId 保持为 0 作为回退值：此处首要目的是触发 base_comm 单例构造以保证析构顺序，
            // phyId 正确性是次要的。后续 LegacyGetOpHcomInfo 中会用正确的 devId 重新调用
            // InitBaseCommRes 覆盖初始化。
            HCCL_WARNING(
                "[CollCommMgr][%s] get deviceId unsuccessfully, ret[%d], use default devPhyId[0].", func,
                static_cast<s32>(ret));
        }
        HcommResult hRet = HcommResMgrInit(devPhyId);
        if (hRet != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[CollCommMgr][%s] HcommResMgrInit is unsuccessful, ret[%d], "
                "base_comm singleton construct-order not guaranteed.",
                func, static_cast<s32>(hRet));
        }
        return true;
    }();
    (void)baseCommReady;
    // 预热 SharedJettyChannelPool：使其先于 CollCommMgr 构造，从而后于 CollCommMgr 析构。
    // 否则 SharedJettyChannelPool 在运行期由 HcommChannelCreate 首次触发，晚于 CollCommMgr 构造，
    // 进程退出时先于 CollCommMgr 析构；
    // ~CollCommMgr → ~hcclComm → ~CollComm → ~MyRank
    // → DestroyAllByMyRank → CollectMyRankChannelsLocked 会命中已析构的
    // SharedJettyChannelPool 静态对象（mtx_/rankPools_ 已销毁）。
    static const bool sharedJettyChannelPoolReady = []() {
        (void)SharedJettyChannelPool::GetInstance();
        return true;
    }();
    (void)sharedJettyChannelPoolReady;
    static CollCommMgr instance;
    return instance;
}

HcclResult CollCommMgr::InitConfigMgr() { return configMgr_.Init(); }

CollCommMgr::~CollCommMgr()
{
    HCCL_INFO("[CollCommMgr][~CollCommMgr] destruct begin.");
    for (auto& monitor : clusterMonitor_) {
        (void)monitor.DeInit();
    }
    HCCL_INFO("[CollCommMgr][~CollCommMgr] destruct end.");
}

hcomm::ClusterMonitor& CollCommMgr::GetClusterMonitor(s32 deviceLogicId)
{
    if (static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[ClusterMonitor][%s]deviceLogicId[%d] >= %u, not valid", __func__, deviceLogicId, MAX_MODULE_DEVICE_NUM);
        return clusterMonitor_[0];
    }
    return clusterMonitor_[deviceLogicId];
}

hcomm::DetectConnectionAnomalies& CollCommMgr::GetDetectConnectionAnomalies(s32 deviceLogicId)
{
    if (static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[DetectConnectionAnomalies][%s]deviceLogicId[%d] >= %u, invalid", __func__, deviceLogicId,
            MAX_MODULE_DEVICE_NUM);
        return detectConnectionAnomalies_[0];
    }
    return detectConnectionAnomalies_[deviceLogicId];
}

HcclResult CollCommMgr::TryReserveCcuMsComm(s32 deviceLogicId, const std::string& commId, bool& reserved)
{
    reserved = false;
    if (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM || commId.empty()) {
        HCCL_ERROR(
            "[%s] invalid parameter, deviceLogicId[%d], max device num[%u], commId empty[%d].", __func__, deviceLogicId,
            MAX_MODULE_DEVICE_NUM, commId.empty());
        return HCCL_E_PARA;
    }

    std::lock_guard<std::mutex> lock(ccuMsCommMutex_);
    auto& owner = ccuMsCommIds_[deviceLogicId];
    if (owner.empty()) {
        owner = commId;
        reserved = true;
    }
    return HCCL_SUCCESS;
}

void CollCommMgr::ReleaseCcuMsComm(s32 deviceLogicId, const std::string& commId)
{
    if (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[%s] deviceLogicId[%d] is not valid, max device num[%u].", __func__, deviceLogicId, MAX_MODULE_DEVICE_NUM);
        return;
    }

    std::lock_guard<std::mutex> lock(ccuMsCommMutex_);
    auto& owner = ccuMsCommIds_[deviceLogicId];
    if (owner == commId) {
        owner.clear();
    }
}

OrderLaunchThreadMgr& CollCommMgr::GetOrderLaunchThreadMgr(s32 deviceLogicId)
{
    if (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[CollCommMgr][%s]deviceLogicId[%d] >= %u, not valid", __func__, deviceLogicId, MAX_MODULE_DEVICE_NUM);
        return orderLaunchThreadMgrs_[0];
    }
    return orderLaunchThreadMgrs_[deviceLogicId];
}

void CollCommMgr::RegisterCollComm(CollComm* collComm)
{
    std::lock_guard<std::mutex> lock(mutex_);
    allCollComms_[collComm->GetCommId()] = collComm;
    // 注册到需要的地方
    taskAbortHandler_.Register(collComm);
    (void)GetOrderLaunchThreadMgr(collComm->GetDeviceLogicId()).RegisterOrderLaunch(collComm->GetCommId());
}

void CollCommMgr::UnregisterCollComm(CollComm* collComm)
{
    std::lock_guard<std::mutex> lock(mutex_);
    allCollComms_.erase(collComm->GetCommId());
    // 从通信域里面注销
    taskAbortHandler_.UnRegister(collComm);
    (void)GetClusterMonitor(collComm->GetDeviceLogicId()).UnRegisterToClusterMonitor(collComm);
    (void)GetOrderLaunchThreadMgr(collComm->GetDeviceLogicId()).UnRegisterOrderLaunch(collComm->GetCommId());
}

const std::unordered_map<std::string, CollComm*>& CollCommMgr::GetAllCollComms() const { return allCollComms_; }

CollComm* CollCommMgr::FindCollCommLocked(const std::string& commId)
{
    auto iter = allCollComms_.find(commId);
    return iter == allCollComms_.end() ? nullptr : iter->second;
}

HcclResult CollCommMgr::GetCollComm(const std::string& commId, CollComm*& collComm)
{
    std::lock_guard<std::mutex> lock(mutex_);
    collComm = FindCollCommLocked(commId);
    if (collComm == nullptr) {
        HCCL_WARNING("[CollCommMgr][%s] commId[%s] not found", __func__, commId.c_str());
        return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommMgr::GetDpuTaskExpMem(const std::string& commId, DpuTaskExpMemRef& ref)
{
    ref.addr = nullptr;
    ref.keepAlive = nullptr;
    // 反查必须与后续取值同锁：注销路径 UnregisterCollComm 摘除条目、其后成员析构释放 CollComm，
    // 都要先持有 mutex_，故锁内不存在被并发释放的中间态 CollComm
    std::lock_guard<std::mutex> lock(mutex_);
    CollComm* collComm = FindCollCommLocked(commId);
    if (collComm == nullptr) {
        HCCL_WARNING("[CollCommMgr][%s] commId[%s] not found", __func__, commId.c_str());
        return HCCL_E_NOT_FOUND;
    }
    ThreadMgr* threadMgr = collComm->GetThreadMgr();
    CHK_PTR_NULL(threadMgr);
    DpuThreadService* dpuService = threadMgr->GetDpuThreadService();
    CHK_PTR_NULL(dpuService);
    return dpuService->GetTaskExpMem(ref.addr, ref.keepAlive);
}

HcclResult CollCommMgr::GetDpuDevId(const std::string& commId, uint32_t& devId)
{
    std::lock_guard<std::mutex> lock(mutex_);
    CollComm* collComm = FindCollCommLocked(commId);
    if (collComm == nullptr) {
        HCCL_WARNING("[CollCommMgr][%s] commId[%s] not found", __func__, commId.c_str());
        return HCCL_E_NOT_FOUND;
    }
    ThreadMgr* threadMgr = collComm->GetThreadMgr();
    CHK_PTR_NULL(threadMgr);
    DpuThreadService* dpuService = threadMgr->GetDpuThreadService();
    CHK_PTR_NULL(dpuService);
    return dpuService->GetDpuDevId(devId);
}

std::vector<std::string> CollCommMgr::SnapshotCollComms()
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> snapshot;
    snapshot.reserve(allCollComms_.size());
    for (const auto& iter : allCollComms_) {
        snapshot.push_back(iter.first);
    }
    return snapshot;
}

void CollCommMgr::InitBaseCommRes(uint32_t devId) const { (void)HcommResMgrInit(devId); }

HcclOpInfoCtx& CollCommMgr::LegacyGetOpHcomInfo(uint32_t devId)
{
    if (devId >= MAX_MODULE_DEVICE_NUM + 1) {
        devId = MAX_MODULE_DEVICE_NUM;
    }
    // baseCommInited_ 无需加锁：本函数在生产路径中始终由 LegacyGetHcclExistDeviceOpInfoCtx /
    // LegacyGetHcclOpInfoCtx 在 opHcomInfosMutex_ 锁内调用
    if (!baseCommInited_[devId]) {
        InitBaseCommRes(devId);
        baseCommInited_[devId] = true;
    }
    return opHcomInfos_[devId];
}

HcclOpInfoCtx& CollCommMgr::LegacyGetHcclExistDeviceOpInfoCtx(s32 devId)
{
    std::lock_guard<std::mutex> lock(opHcomInfosMutex_);
    auto& opHcomInfo = LegacyGetOpHcomInfo(devId);
    if (!opHcomInfo.isUsed) {
        HCCL_INFO("[LegacyGetHcclOpInfoCtx] Set device, use devId[%d] ", devId);
        auto& backUpOpHcomInfo = LegacyGetOpHcomInfo(MAX_MODULE_DEVICE_NUM);
        if (backUpOpHcomInfo.isUsed) {
            HcclSetThreadDeviceId(static_cast<s32>(MAX_MODULE_DEVICE_NUM));
            HCCL_INFO("[LegacyGetHcclOpInfoCtx] Used cover bottom devId[%u]", MAX_MODULE_DEVICE_NUM);
            return backUpOpHcomInfo;
        }
    }

    HCCL_INFO("[LegacyGetHcclExistDeviceOpInfoCtx] use devId[%d] opHcomInfos", devId);
    opHcomInfo.isUsed = true;
    return opHcomInfo;
}

HcclOpInfoCtx& CollCommMgr::LegacyGetHcclOpInfoCtx()
{
    if (HcclGetDeviceId() == HCCL_SUCCESS) {
        return LegacyGetHcclExistDeviceOpInfoCtx(HcclGetThreadDeviceId());
    }

    std::lock_guard<std::mutex> lock(opHcomInfosMutex_);
    for (u32 i = 0; i < MAX_MODULE_DEVICE_NUM; i++) {
        auto& opHcomInfo = LegacyGetOpHcomInfo(i);
        if (opHcomInfo.isUsed) {
            HcclSetThreadDeviceId(static_cast<s32>(i));
            HCCL_INFO("[LegacyGetHcclOpInfoCtx] Not set device, Used devId[%u] ", i);
            return opHcomInfo;
        }
    }

    HcclSetThreadDeviceId(static_cast<s32>(MAX_MODULE_DEVICE_NUM));
    auto& backUpOpHcomInfo = LegacyGetOpHcomInfo(MAX_MODULE_DEVICE_NUM);
    backUpOpHcomInfo.isUsed = true;
    HCCL_INFO("[LegacyGetHcclOpInfoCtx] Used cover bottom devId[%u]", MAX_MODULE_DEVICE_NUM);
    return backUpOpHcomInfo;
}

} // namespace hccl
