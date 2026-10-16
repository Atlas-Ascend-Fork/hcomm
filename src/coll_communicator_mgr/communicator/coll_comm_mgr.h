/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_COMM_MGR_H
#define COLL_COMM_MGR_H

#include <unordered_set>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <array>
#include "coll_comm.h"
#include "coll_comm_config_mgr.h"
#include "cluster_monitor.h"
#include "ns_recovery/task_abort_handler.h"
#include "legacy_op_hcom_info.h"
#include "order_launch_thread_mgr.h"

namespace hccl {
/**
 * @brief HostDpu 的 DPUTASKEXCEPTION 异常区取用句柄：地址 + 保活引用
 * @note 异常区由 DPU 线程实体持有、随通信域销毁释放，故 addr 只在 keepAlive 存续期间有效。
 *       保活引用按 void 擦除类型：本层接口只出 POD 与标准库类型，不带入 base_comm 的线程类名。
 */
struct DpuTaskExpMemRef {
    void* addr{nullptr};
    std::shared_ptr<void> keepAlive;
};
/**
 * @note 职责：实现多个集合通信通信域上下文的创建、销毁管理，及多通信域资源、信息的共享等。
 */
class CollCommMgr {
public:
    static CollCommMgr& GetInstance();
    void RegisterCollComm(CollComm* collComm);
    void UnregisterCollComm(CollComm* collComm);
    const std::unordered_map<std::string, CollComm*>& GetAllCollComms() const;

    /**
     * @brief 按 commId 查通信域，与 RegisterCollComm/UnregisterCollComm 同锁互斥
     * @param[in] commId 通信域 id
     * @param[out] collComm 查到的通信域指针
     * @return HcclResult 成功返回 HCCL_SUCCESS，未注册返回 HCCL_E_NOT_FOUND
     * @note 返回的裸指针只在锁内有效：UnregisterCollComm 摘除条目后 CollComm 即可能析构，
     *       锁外解引用是 UAF；要取 CollComm 上的成员请用 GetDpuTaskExpMem 这类锁内完成的接口。
     *       GetAllCollComms() 返回裸引用且不加锁，与持锁的 UnregisterCollComm 并发即数据竞争，不得用于新链路
     */
    HcclResult GetCollComm(const std::string& commId, CollComm*& collComm);

    /**
     * @brief 经 commId 反查 HostDpu 的 DPUTASKEXCEPTION 异常区，不创建。
     * @param[in] commId 通信域 id
     * @param[out] ref 异常区地址与保活引用；未拉起 DPU 时 addr 置 nullptr
     * @return HCCL_SUCCESS 查到；HCCL_E_NOT_FOUND 通信域未注册或未拉起 DPU 线程；其余为具体错误码
     * @note commId → CollComm → CommEngineResMgr → ThreadMgr → DpuThreadService 查找链的宿主侧入口，
     *       供 dfx 异常链路（无 CollComm*，只有 commId）反查使用。
     *       反查在同一把锁内完成，依赖「条目先摘除、对象后析构」的注销序：注销方摘除条目必须持
     *       mutex_（~hcclComm 先 UnregisterCollComm，其成员 collComm_ 才开始析构），而对象析构本身不受
     *       该锁保护，故持锁查找期间 CollComm 必然存活。新增「未注销即析构」路径会破坏这一前提。
     *       ref.keepAlive 承担锁外那段窗口：调用方须持有它直到用完 ref.addr。
     */
    HcclResult GetDpuTaskExpMem(const std::string& commId, DpuTaskExpMemRef& ref);

    /**
     * @brief 经 commId 反查 HostDpu 线程绑定的设备号，不创建。
     * @param[in] commId 通信域 id
     * @param[out] devId 设备号；未查到时不改写入参
     * @return HCCL_SUCCESS 查到；HCCL_E_NOT_FOUND 通信域未注册或未拉起 DPU 线程；其余为具体错误码
     */
    HcclResult GetDpuDevId(const std::string& commId, uint32_t& devId);

    /**
     * @brief 持锁拷贝一份通信域 id 快照，供调用方在锁外遍历后经 GetCollComm 反查
     * @return std::vector<std::string> 通信域 id 快照
     * @note 不用「锁内回调」形态：回调中若再调本类接口会自锁（mutex_ 非递归）。
     *       快照只保证取快照那一刻的一致性，遍历期间通信域仍可能被注销，调用方需经 GetCollComm
     *       反查确认仍存活，不得直接解引用快照对象
     */
    std::vector<std::string> SnapshotCollComms();
    hcomm::ClusterMonitor& GetClusterMonitor(s32 deviceLogicId);
    HcclResult TryReserveCcuMsComm(s32 deviceLogicId, const std::string& commId, bool& reserved);
    void ReleaseCcuMsComm(s32 deviceLogicId, const std::string& commId);
    OrderLaunchThreadMgr& GetOrderLaunchThreadMgr(s32 deviceLogicId);
    HcclTaskAbortHandler& GetTaskAbortHandler() { return taskAbortHandler_; }
    void InitBaseCommRes(uint32_t devId) const;
    HcclResult InitConfigMgr();
    const CollCommConfigMgr& GetConfigMgr() const { return configMgr_; }
    ~CollCommMgr();

    // 以下接口以 Legacy 前缀标记，表示用于兼容历史老接口，仅做 bug 修复与兼容维护，不再承接新特性、不再继续演进
    HcclOpInfoCtx& LegacyGetOpHcomInfo(uint32_t devId);
    HcclOpInfoCtx& LegacyGetHcclExistDeviceOpInfoCtx(s32 devId);
    HcclOpInfoCtx& LegacyGetHcclOpInfoCtx();

private:
    CollComm* FindCollCommLocked(const std::string& commId); // 调用方须持有 mutex_

    std::unordered_map<std::string, CollComm*> allCollComms_;
    std::array<hcomm::ClusterMonitor, MAX_MODULE_DEVICE_NUM> clusterMonitor_;
    std::array<std::string, MAX_MODULE_DEVICE_NUM> ccuMsCommIds_{};
    std::array<OrderLaunchThreadMgr, MAX_MODULE_DEVICE_NUM> orderLaunchThreadMgrs_;
    HcclTaskAbortHandler taskAbortHandler_;
    CollCommConfigMgr configMgr_;

    std::array<HcclOpInfoCtx, MAX_MODULE_DEVICE_NUM + 1> opHcomInfos_;
    std::array<bool, MAX_MODULE_DEVICE_NUM + 1> baseCommInited_{};
    std::mutex opHcomInfosMutex_;

    std::mutex mutex_;
    std::mutex ccuMsCommMutex_;
};
} // namespace hccl
#endif // COLL_COMM_MGR_H
