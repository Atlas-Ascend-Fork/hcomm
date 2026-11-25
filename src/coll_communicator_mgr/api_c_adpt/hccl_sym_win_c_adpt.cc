/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_sym_win.h"
#include "hcomm_team_mgr.h"
#include "log.h"

using namespace hcomm;

// 适配层将 HcommResult 直接 static_cast 为 HcclResult，依赖两套错误枚举数值一致；
// 任一侧重排/插码时在此编译期暴露，避免静默返回语义错误的错误码。
static_assert(static_cast<int32_t>(HCCL_E_PARA) == static_cast<int32_t>(HCOMM_E_PARA), "error code mismatch: PARA");
static_assert(static_cast<int32_t>(HCCL_E_PTR) == static_cast<int32_t>(HCOMM_E_PTR), "error code mismatch: PTR");
static_assert(
    static_cast<int32_t>(HCCL_E_NOT_SUPPORT) == static_cast<int32_t>(HCOMM_E_NOT_SUPPORT),
    "error code mismatch: NOT_SUPPORT");
static_assert(
    static_cast<int32_t>(HCCL_E_NOT_FOUND) == static_cast<int32_t>(HCOMM_E_NOT_FOUND),
    "error code mismatch: NOT_FOUND");

HcclResult HcclSymWinGetPeerDevicePointer(HcclCommSymWindow winHandle, size_t offset, uint32_t peerMemberId, void** ptr)
{
    CHK_PRT_RET(ptr == nullptr, HCCL_ERROR("[%s] ptr is nullptr", __func__), HCCL_E_PTR);
    *ptr = nullptr;
    CHK_PRT_RET(winHandle == nullptr, HCCL_ERROR("[%s] nullptr parameter", __func__), HCCL_E_PTR);
    const HcommResult ret = HcommTeamMgr::GetInstance().WindowGetPeerPointer(winHandle, offset, peerMemberId, ptr);
    // 非 UB 窗口返回 NOT_SUPPORT 属能力探测结果，调用方会回退 legacy 查询，不按错误上报
    if (ret == HCOMM_E_NOT_SUPPORT) {
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCOMM_SUCCESS, HCCL_ERROR("[%s] failed, handle[%p], ret[%d]", __func__, winHandle, ret),
        static_cast<HcclResult>(ret));
    return HCCL_SUCCESS;
}
