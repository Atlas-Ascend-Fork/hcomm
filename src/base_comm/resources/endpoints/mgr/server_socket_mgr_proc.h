/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_SERVER_SOCKET_MGR_PROC_H
#define HCOMM_SERVER_SOCKET_MGR_PROC_H

#include "hccl/hccl_types.h"
#include "resources/endpoint_pairs/sockets/socket_mgr.h"

namespace hcomm {

/**
 * @brief server socket 族节点（设备级）：SocketMgr 值成员。
 * 进程级的 ServerSocketMgr 聚合壳由 EndpointProcMgr 持有；
 * devPhyId 越界（含 65）由窄访问器回落 0 号设备子树（README R9，基线无备份槽语义）。
 * 声明顺序 = 析构契约反序（依据见 README R3）。
 */
class ServerSocketMgrProc {
public:
    void DeInit() {}

    SocketMgr& GetSocketMgr() { return socketMgr_; }

public:
    ServerSocketMgrProc() = default;
    ~ServerSocketMgrProc() = default;

private:
    ServerSocketMgrProc(const ServerSocketMgrProc&) = delete;
    ServerSocketMgrProc& operator=(const ServerSocketMgrProc&) = delete;

    SocketMgr socketMgr_;
};

} // namespace hcomm

#endif // HCOMM_SERVER_SOCKET_MGR_PROC_H
