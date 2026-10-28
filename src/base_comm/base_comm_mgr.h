/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_BASE_COMM_MGR_H
#define HCOMM_BASE_COMM_MGR_H

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include "hccl/hccl_types.h"
#include "common/exit_profile.h"
#include "common/dlsym_proc.h"
#include "config_mgr/config_mgr_proc.h"
#include "dfx/dfx_procs.h"
#include "legacy950_res_mgr_proc.h"
#include "handle_registry_procs.h"
#include "base_comm_res_per_dev.h"
#include "resources/comm_engine_res/notify_mgr_proc.h"
#include "resources/endpoints/net_dev/endpoint_proc_mgr.h"
#include "resources/endpoints/mgr/endpoint_mgr.h"

namespace hcomm {

// 树亡标志查询：全树（含设备子树）析构完成后返回 true；供锚定单例（晚于树死）析构链判活（README R10）
bool IsResourceTreeTeardownFinished();

/**
 * @brief 进程级资源树根，base_comm 唯一树根（唯一新增单例；存量函数级 static 工厂见 README R1 例外清单）。
 *
 * 树形态、析构契约、生命周期双路径、锚定与访问器等演进规则见 src/base_comm/README.md。
 */
class BaseCommMgr {
public:
    static BaseCommMgr& GetInstance();

    // 设备状态回调注册（幂等；注销与标志复位在 DeInit 收尾）。regName 是 ACL 按名注销的配对键，严禁改名
    static void RegisterDeviceResetCallback();
    static void RegisterDeviceRefreshCallback();
    static void UnregisterDeviceRefreshCallback();

    BaseCommMgr(const BaseCommMgr&) = delete;
    BaseCommMgr& operator=(const BaseCommMgr&) = delete;

    // 设备层：首访懒建该设备子树（加锁），未使用的设备不建
    BaseCommResPerDev& GetBaseCommResPerDev(uint32_t devPhyId);

    // 树根直属节点（值成员，构造按声明序自动完成，永不重建）：一行内联，返回引用
    DfxProc& GetDfxProc() { return dfxProc_; }
    DlsymProc& GetDlsymProc() { return dlsymProc_; }
    ConfigMgrProc& GetConfigMgrProc() { return configMgrProc_; }
    HandleRegistryMgrProc& GetHandleRegistryMgrProc() { return handleRegistryMgrProc_; }
    CommEngineBinMgrProc& GetCommEngineBinMgrProc() { return commEngineBinMgrProc_; }
    EndpointProcMgr& GetEndpointProcMgr() { return endpointProcMgr_; }
    NotifyMgrProc& GetNotifyMgrProc() { return notifyMgrProc_; }
    EndpointMgr& GetEndpointMgr() { return endpointMgr_; }

    // 显式按契约顺序清理（动作路径），幂等；EXIT_PROFILE_RESET 跳过触碰设备的动作，仅 host 侧清理
    HcclResult DeInit(ExitProfile profile = ExitProfile::EXIT_PROFILE_GRACEFUL);

private:
    BaseCommMgr();
    ~BaseCommMgr();

    // ===== 成员声明顺序 = 析构契约反序（先声明的后死；顺序依据见 README R3）=====
    // 最先声明 = 全树最后死：析构置位树亡标志
    struct TeardownFlagGuard {
        ~TeardownFlagGuard();
    };
    TeardownFlagGuard teardownFlagGuard_;
    std::atomic<bool> isDeInited_{false};         // 析构期回调链读此标志
    std::atomic<bool> lastDeinitWasReset_{false}; // RESET 后析构穿透的告警依据

    // 节点均为值成员，构造/析构严格按声明序
    Legacy950ResMgrProc legacy950ResMgrProc_; // legacy950 锚定节点
    ConfigMgrProc configMgrProc_;             // 配置
    DlsymProc dlsymProc_;                     // 动态库符号表
    DfxProc dfxProc_;                         // 异常上报/profiling
    EndpointProcMgr endpointProcMgr_;         // 进程级网络资源（网卡上下文/listen socket 表）
    NotifyMgrProc notifyMgrProc_; // notify 管理（进程级语义；channel 析构要还 notify，晚于句柄表死）
    CommEngineBinMgrProc commEngineBinMgrProc_;   // 卸 bin
    BaseCommResPerDevMgr perDevResMgr_;           // 设备子树存取器（现役表 + 退役区 + 兜底）
    HandleRegistryMgrProc handleRegistryMgrProc_; // 句柄表（残留 channel 析构触设备子树，须先死）
    EndpointMgr endpointMgr_;                     // Endpoint 族：全树最先死
};

} // namespace hcomm

#endif // HCOMM_BASE_COMM_MGR_H
