/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_INSTANCE_PROC_H
#define HCOMM_CCU_INSTANCE_PROC_H

#include "hccl/hccl_types.h"
#include "common/exit_profile.h"
#include "resources/ccu/ccu_kernel/ccu_kernel_mgr.h"
#include "resources/ccu/ccu_instance/ccu_instance_mgr.h"
#include "resources/ccu/ccu_instance/ccu_var_event_res_mgr.h"
#include "common/tp_mgr.h"

namespace hcomm {

/**
 * @brief CCU 实例资源节点（设备级，每 devPhyId 一份值成员）。
 *
 * 声明序 = 析构契约反序：ccuInstanceMgr_ 先死（insMap_ 析构 CcuInstance 时回调
 * CcuKernelMgr::UnRegister / CcuVarEventResMgr::ReleaseByInstance，两者必须存活）。
 * static reserve* 为越界 logicId 回落的进程级兜底（从不 Init）。
 */
class CcuInstanceProc {
public:
    CcuInstanceProc() = default;
    ~CcuInstanceProc() = default;
    CcuInstanceProc(const CcuInstanceProc&) = delete;
    CcuInstanceProc& operator=(const CcuInstanceProc&) = delete;

    void DeInit(ExitProfile profile = ExitProfile::EXIT_PROFILE_GRACEFUL);

    CcuKernelMgr& GetCcuKernelMgr() { return ccuKernelMgr_; }
    CcuInstanceMgr& GetCcuInstanceMgr() { return ccuInstanceMgr_; }
    CcuVarEventResMgr& GetCcuVarEventResMgr() { return ccuVarEventResMgr_; }
    TpMgr& GetTpMgr() { return tpMgr_; }

    static CcuKernelMgr& GetReserveCcuKernelMgr() { return reserveCcuKernelMgr_; }
    static CcuInstanceMgr& GetReserveCcuInstanceMgr() { return reserveCcuInstanceMgr_; }
    static CcuVarEventResMgr& GetReserveCcuVarEventResMgr() { return reserveCcuVarEventResMgr_; }

private:
    CcuVarEventResMgr ccuVarEventResMgr_;
    TpMgr tpMgr_;
    CcuKernelMgr ccuKernelMgr_;
    CcuInstanceMgr ccuInstanceMgr_;

    static CcuVarEventResMgr reserveCcuVarEventResMgr_;
    static CcuKernelMgr reserveCcuKernelMgr_;
    static CcuInstanceMgr reserveCcuInstanceMgr_;
};

} // namespace hcomm

#endif // HCOMM_CCU_INSTANCE_PROC_H
