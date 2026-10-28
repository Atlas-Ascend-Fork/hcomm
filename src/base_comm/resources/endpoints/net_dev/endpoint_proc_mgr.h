/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_RESOURCES_ENDPOINTS_NET_DEV_ENDPOINT_PROC_MGR_H
#define HCOMM_RESOURCES_ENDPOINTS_NET_DEV_ENDPOINT_PROC_MGR_H

#include <map>
#include <memory>
#include <mutex>
#include "hccl/hccl_types.h"
#include "hccl_common.h"
#include "common/exit_profile.h"
#include "resources/endpoints/net_dev/global_net_dev_manager.h"
#include "resources/endpoints/server_socket/server_socket_manager.h"

namespace hcomm {

/**
 * @brief PortInfo 键网络资源族占位节点（进程级，EndpointProcMgr 子节点），当前无实体。
 */
class PortResProc {
public:
    void DeInit() {}

public:
    PortResProc() = default;
    ~PortResProc() = default;

private:
    PortResProc(const PortResProc&) = delete;
    PortResProc& operator=(const PortResProc&) = delete;
};

/**
 * @brief Endpoint 进程级资源管理节点：netDevCtx/serverSocket 两表两锁 + isDlRaInited_、
 * ServerSocketManager 聚合壳（map 以 PortData.RankId 区分设备）。
 * 晚于所有设备子树与 EndpointMgr 死，早于 Legacy950 死（依据见 README R3）。
 */

class EndpointProcMgr {
public:
    // 当前无动作（no-op）：还网卡上下文/关 listen socket 等设备动作由析构链兜底
    void DeInit(ExitProfile profile = ExitProfile::EXIT_PROFILE_GRACEFUL)
    {
        (void)profile;
        portResProc_.DeInit();
    }

    std::map<hccl::PortInfo, std::pair<NicType, HcclNetDevCtx>>& GetNetDevCtxMap() { return netDevCtxMap_; }
    std::map<hccl::PortInfo, hccl::Referenced>& GetNetDevCtxRefMap() { return netDevCtxRefMap_; }
    std::mutex& GetNetDevCtxMtx() { return netDevCtxMtx_; }
    bool IsDlRaInited() const { return isDlRaInited_; }
    void SetDlRaInited(bool v) { isDlRaInited_ = v; }
    std::mutex& GetServerMapMutex() { return serverMapMutex_; }
    std::map<hccl::PortInfo, std::shared_ptr<hccl::HcclSocket>>& GetServerSocketMap() { return serverSocketMap_; }
    std::map<hccl::PortInfo, hccl::Referenced>& GetServerSocketRefMap() { return serverSocketRefMap_; }
    PortResProc& GetPortResProc() { return portResProc_; }
    // 构造时锚定 legacy950 的 (Host)SocketHandleManager，保证其晚于本对象析构
    ServerSocketManager& GetServerSocketManager() { return serverSocketManager_; }

public:
    EndpointProcMgr() = default;
    ~EndpointProcMgr() = default;

private:
    EndpointProcMgr(const EndpointProcMgr&) = delete;
    EndpointProcMgr& operator=(const EndpointProcMgr&) = delete;

    // 声明顺序 = 析构契约反序（依据见 README R3）：互斥锁最后死；serverSocketManager_ 与表互不引用，顺序无约束
    std::mutex netDevCtxMtx_;
    std::mutex serverMapMutex_;
    std::map<hccl::PortInfo, std::pair<NicType, HcclNetDevCtx>> netDevCtxMap_;
    std::map<hccl::PortInfo, hccl::Referenced> netDevCtxRefMap_;
    bool isDlRaInited_{false};
    std::map<hccl::PortInfo, std::shared_ptr<hccl::HcclSocket>> serverSocketMap_;
    std::map<hccl::PortInfo, hccl::Referenced> serverSocketRefMap_;
    ServerSocketManager serverSocketManager_;
    PortResProc portResProc_; // 占位节点（预留扩展位）
};

} // namespace hcomm

#endif // HCOMM_RESOURCES_ENDPOINTS_NET_DEV_ENDPOINT_PROC_MGR_H
