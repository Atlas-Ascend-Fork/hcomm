/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_HDC_PROC_H
#define HCOMM_HDC_PROC_H

#include "hccl/hccl_types.h"
#include "common/exit_profile.h"
#include "common/hccp_tlv_hdc_mgr.h"
#include "log.h"

namespace hcomm {

/**
 * @brief HDC 节点（设备级），值成员持有 HccpTlvHdcMgr
 */
class HdcProc {
public:
    // 引擎族收尾（幂等）：graceful 执行 TLV 去初始化（触设备）；RESET 跳过并告警（README R5）
    void DeInit(ExitProfile profile = ExitProfile::EXIT_PROFILE_GRACEFUL)
    {
        if (!hccpTlvHdcMgr_.IsInited()) {
            return;
        }
        if (profile == ExitProfile::EXIT_PROFILE_RESET) {
            HCCL_WARNING(
                "[HdcProc][%s] RESET profile: skip HDC deinit (device may be unavailable), "
                "residual cleanup deferred to destructor",
                __func__);
            return;
        }
        (void)hccpTlvHdcMgr_.Deinit();
    }

    HccpTlvHdcMgr& GetHccpTlvHdcMgr() { return hccpTlvHdcMgr_; }

public:
    ~HdcProc() = default;

public:
    HdcProc() = default;

private:
    HdcProc(const HdcProc&) = delete;
    HdcProc& operator=(const HdcProc&) = delete;

    HccpTlvHdcMgr hccpTlvHdcMgr_;
};

} // namespace hcomm

#endif // HCOMM_HDC_PROC_H
