/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "legacy950_res_mgr_proc.h"

// legacy950（Hccl::，libhccl_v2.so）单例
#include "hccp_hdc_manager.h"
#include "hccp_peer_manager.h"
#include "hccp_tlv_hdc_manager.h"
#include "rdma_handle_manager.h"
#include "inner_net_dev_manager.h"
#include "socket_handle_manager.h"
#include "host_socket_handle_manager.h"
// legacy910（libhccl_v2.so）通信域配置单例
#include "comm_configer.h"
// 设备级 legacy 依赖清单（备份槽预热）
#include "base_comm_res_per_dev.h"

namespace hcomm {

// 构造即锚定：仅首触做单例生命周期控制（先构造的后析构），本节点首建 = 下列单例晚于全树死亡
Legacy950ResMgrProc::Legacy950ResMgrProc()
{
    // CommConfiger 最先锚定 = 最晚析构；ExceptionHandle/ExceptionCallbackMgr 为 AICPU 专属，host 无需锚定
    hccl::CommConfiger::GetInstance();
    Hccl::HccpHdcManager::GetInstance();
    Hccl::HccpPeerManager::GetInstance();
    Hccl::HccpTlvHdcManager::GetInstance();
    Hccl::RdmaHandleManager::GetInstance();
    Hccl::InnerNetDevManager::GetInstance();
    Hccl::SocketHandleManager::GetInstance();
    Hccl::HostSocketHandleManager::GetInstance();

    // 设备级 static 数组预热：设备子树懒建的逐槽首触晚于树根构造，按静态析构逆序锁不死
    // （反而先于树死亡，树析构期 ~CcuDrvHandle 链回调即 UB）；以备份槽调用依赖清单，
    // 数组在树根构造期内完成注册 = 锁死晚于全树死亡（构造均平凡，不触设备/ACL，README R8）
    AnchorDeviceSingletons(MAX_MODULE_DEVICE_NUM);
}

} // namespace hcomm
