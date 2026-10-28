/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_NET_DEV_PROC_H
#define HCOMM_NET_DEV_PROC_H

#include "hccl/hccl_types.h"
#include "resources/endpoints/net_dev/global_net_dev_manager.h"

namespace hcomm {

/**
 * @brief 网卡设备级实例节点（设备级）：GlobalNetDevMgr 值成员（phyId↔logicId 一一对应）。
 * 进程级表/锁由 EndpointProcMgr 持有；设备子树内最后死（依据见 README R3）。
 */
class NetDevProc {
public:
    void DeInit() {}

    hccl::GlobalNetDevMgr& GetNetDev() { return netDev_; }

public:
    NetDevProc() = default;
    ~NetDevProc() = default;

private:
    NetDevProc(const NetDevProc&) = delete;
    NetDevProc& operator=(const NetDevProc&) = delete;

    hccl::GlobalNetDevMgr netDev_; // 随设备子树懒建/回收
};

} // namespace hcomm

#endif // HCOMM_NET_DEV_PROC_H
