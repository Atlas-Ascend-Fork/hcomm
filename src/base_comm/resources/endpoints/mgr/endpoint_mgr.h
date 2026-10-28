/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_ENDPOINT_MGR_H
#define HCOMM_ENDPOINT_MGR_H

#include <array>
#include "hccl/hccl_types.h"
#include "hccl_common.h"
#include "common/exit_profile.h"
#include "resources/endpoints/mgr/endpoint_handle_mgr.h"
#include "resources/endpoint_pairs/channels/shared_jetty_mgr.h"
#include "resources/reged_mems/proc_reged_mem_mgr_cache.h"
#include "common/eid_info_mgr.h"

namespace hcomm {

/**
 * @brief Endpoint 族管理器（进程级，树根直属）：EndpointHandleMgr 整表、SharedJettyMgr、
 * ProcRegedMemMgrCache、EidInfoMgr（按 devPhyId 数组索引；调用方持 logicId 时须先经
 * hrtGetDevicePhyIdByIndex 转 phyId 传入）。
 * 与 coll_communicator_mgr 侧 hccl::EndpointMgr 同名不同层（仓内 Channel/MyRank 同款惯例），
 * 靠命名空间区分。
 * 声明顺序 = 析构契约反序（依据见 README R3）：全树最先死。
 */
class EndpointMgr {
public:
    // 清 endpoint 句柄表残留（幂等）；辅助表无 DeInit，记录随 endpoint 析构链摘除
    // RESET 跳过：endpoint 析构链要回调设备，设备不可用时不能执行
    void DeInit(ExitProfile profile = ExitProfile::EXIT_PROFILE_GRACEFUL)
    {
        if (profile == ExitProfile::EXIT_PROFILE_RESET) {
            return;
        }
        endpointHandleMgr_.DeInit();
    }

    EndpointHandleMgr& GetEndpointHandleMgr() { return endpointHandleMgr_; }
    SharedJettyMgr& GetSharedJettyMgr() { return sharedJettyMgr_; }
    ProcRegedMemMgrCache& GetProcRegedMemMgrCache() { return procRegedMemMgrCache_; }
    std::array<EidInfoMgr, MAX_MODULE_DEVICE_NUM + 1>& GetEidInfoMgrs() { return eidInfoMgrs_; }

public:
    EndpointMgr() = default;
    ~EndpointMgr() = default;

private:
    EndpointMgr(const EndpointMgr&) = delete;
    EndpointMgr& operator=(const EndpointMgr&) = delete;

    // 声明顺序 = 析构契约反序（依据见 README R3）：endpointHandleMgr_ 必须最后声明（最先死）——
    // endpoint 析构经非 owning 别名 cacheAlias_ 解引用 procRegedMemMgrCache_（R7 别名规则）；
    // 其余辅助表间无约束
    SharedJettyMgr sharedJettyMgr_;
    ProcRegedMemMgrCache procRegedMemMgrCache_;
    std::array<EidInfoMgr, MAX_MODULE_DEVICE_NUM + 1> eidInfoMgrs_;
    EndpointHandleMgr endpointHandleMgr_;
};

} // namespace hcomm

#endif // HCOMM_ENDPOINT_MGR_H
