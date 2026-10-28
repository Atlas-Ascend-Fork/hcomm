/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_BASE_COMM_NODE_ACCESSORS_H
#define HCOMM_BASE_COMM_NODE_ACCESSORS_H

#include <cstdint>
#include <memory>
#include "base_comm_mgr.h"
#include "base_comm_res_per_dev.h"
#include "common/dlsym_proc.h"
#include "config_mgr/config_mgr_proc.h"
#include "resources/comm_engine_res/comm_engine_procs.h"
#include "resources/endpoints/mgr/endpoint_mgr.h"
#include "resources/endpoints/net_dev/endpoint_proc_mgr.h"
#include "handle_registry_procs.h"

/**
 * @file base_comm_node_accessors.h
 * @brief 窄访问器：资源树实例的短自由函数入口（namespace hcomm，返回引用）。
 *        仅供 host 侧包含（AICPU 侧无资源树）；使用约束与入参语义见 README R7/R9。
 */

namespace hcomm {

// ===== 进程级 =====
inline DlUrmaFunction& GetDlUrmaFunction() { return BaseCommMgr::GetInstance().GetDlsymProc().GetDlUrmaFunction(); }
inline ConfigMgr& GetConfigMgr() { return BaseCommMgr::GetInstance().GetConfigMgrProc().GetConfigMgr(); }
// legacy910 hccl::CommConfiger 转发访问（本体在 libhccl_v2.so，白名单内）
inline hccl::CommConfiger& GetCommConfiger() { return BaseCommMgr::GetInstance().GetConfigMgrProc().GetCommConfiger(); }
inline AicpuCommEngineBinProc& GetAicpuCommEngineBinProc()
{
    return BaseCommMgr::GetInstance().GetCommEngineBinMgrProc().GetAicpuCommEngineBinProc();
}
inline EndpointHandleMgr& GetEndpointHandleMgr()
{
    return BaseCommMgr::GetInstance().GetEndpointMgr().GetEndpointHandleMgr();
}
inline SharedJettyMgr& GetSharedJettyMgr() { return BaseCommMgr::GetInstance().GetEndpointMgr().GetSharedJettyMgr(); }
inline ProcRegedMemMgrCache& GetProcRegedMemMgrCache()
{
    return BaseCommMgr::GetInstance().GetEndpointMgr().GetProcRegedMemMgrCache();
}
// 非 owning 别名（空 deleter）：实例生命周期由 EndpointMgr 成员声明序保证，无需 shared_ptr 保活
inline std::shared_ptr<ProcRegedMemMgrCache> GetProcRegedMemMgrCacheHolder()
{
    auto& cache = GetProcRegedMemMgrCache();
    return std::shared_ptr<ProcRegedMemMgrCache>(&cache, [](ProcRegedMemMgrCache* ptr) {
        (void)ptr;
    });
}

inline EndpointProcMgr& GetEndpointProcMgr() { return BaseCommMgr::GetInstance().GetEndpointProcMgr(); }

inline ServerSocketManager& GetServerSocketManager() { return GetEndpointProcMgr().GetServerSocketManager(); }

inline ChannelHandleRegistryProc& GetChannelHandleRegistryProc()
{
    return BaseCommMgr::GetInstance().GetHandleRegistryMgrProc().GetChannelHandleRegistryProc();
}
inline ThreadHandleRegistryProc& GetThreadHandleRegistryProc()
{
    return BaseCommMgr::GetInstance().GetHandleRegistryMgrProc().GetThreadHandleRegistryProc();
}
inline DpuNotifyManager& GetDpuNotifyManager()
{
    return BaseCommMgr::GetInstance().GetNotifyMgrProc().GetDpuNotifyManager();
}

// 设备级访问器：越界回落备份设备并回写 devPhyId_（定义在 base_comm_node_accessors.cc）
EidInfoMgr& GetEidInfoMgr(uint32_t devPhyId);

// ===== 设备级 =====
EndpointCtxMgr& GetEndpointCtxMgr(uint32_t devPhyId);
SocketMgr& GetSocketMgr(int32_t devPhyId);
hccl::GlobalNetDevMgr& GetGlobalNetDevMgr(uint32_t devPhyId);

CcuKernelMgr& GetCcuKernelMgr(int32_t deviceLogicId);
CcuInstanceMgr& GetCcuInstanceMgr(int32_t deviceLogicId);
CcuVarEventResMgr& GetCcuVarEventResMgr(int32_t deviceLogicId);
TpMgr& GetTpMgr(uint32_t devPhyId);
HccpTlvHdcMgr& GetHccpTlvHdcMgr(uint32_t devPhyId);
// 入参 deviceLogicId（与原 GetInstance 键型一致），内部转 phyId；越界/转换失败回落 0 号设备（同基线槽位 0）
SocketProcess& GetSocketProcess(int32_t deviceLogicId);

} // namespace hcomm

#endif // HCOMM_BASE_COMM_NODE_ACCESSORS_H
