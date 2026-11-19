/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include <memory>
#include "hccl/hccl_res.h"
#include "hccl_comm_pub.h"
#include "coll_comm_mgr.h"
#include "orion_adapter_rts.h"
#include "adapter_rts_common.h" // hrtGetStreamId：注册审计日志打印流ID
#include "hcom_common.h"
#include "param_check_basic_v2.h"
#include "log.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcomSetAttachedStream(const char* group, u32 graphId, const rtStream_t* stream, s32 len)
{
    EXCEPTION_HANDLE_BEGIN
    HCCL_INFO(
        "[HcomSetAttachedStream] entry, group[%s], graphId[%u], stream[%p], len[%d]",
        group == nullptr ? "nullptr" : group, graphId, stream, len);

    CHK_PRT_RET(len < 0, HCCL_ERROR("[HcomSetAttachedStream] len is %d", len), HCCL_E_PARA);
    CHK_PTR_NULL(stream);
    if (group == nullptr) {
        group = HCCL_WORLD_GROUP;
    }

    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        if (len <= 1) {
            HCCL_WARNING("[HcomSetAttachedStream] len <= 1, no stream");
            return HCCL_SUCCESS;
        }
        void* attachedStream = stream[0];
        s32 deviceLogicId = 0;
        CHK_RET(Hccl::HrtGetDevice(deviceLogicId));
        auto& mgr = hccl::CollCommMgr::GetInstance().GetOrderLaunchThreadMgr(deviceLogicId);
        CHK_RET(mgr.SetAttachedStream(std::string(group), graphId, attachedStream));
        // len >= 2 时第二条流为通信域粒度 GE 提前展开流，注入到通信域 ThreadMgr（仅存流，首次获取时创建 thread）
        std::shared_ptr<hccl::hcclComm> hcclCommInV2 = nullptr;
        if (HcomGetCommByGroup(group, hcclCommInV2) != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[HcomSetAttachedStream] get comm by group failed, skip GE unfold stream inject, group[%s]", group);
            return HCCL_SUCCESS;
        }
        hccl::CollComm* collComm = hcclCommInV2->GetCollComm();
        if (collComm == nullptr) {
            HCCL_WARNING("[HcomSetAttachedStream] collComm is null, skip GE unfold stream inject, group[%s]", group);
            return HCCL_SUCCESS;
        }
        hccl::CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        if (engineResMgr == nullptr) {
            HCCL_WARNING(
                "[HcomSetAttachedStream] engineResMgr is null, skip GE unfold stream inject, group[%s]", group);
            return HCCL_SUCCESS;
        }
        HcclResult injectRet = engineResMgr->SetAttachedStream(stream[1]);
        // 注册审计日志：通信域/图/控制流/提前展开流一次打全（流ID查询失败保持INVALID便于识别）
        s32 ctrlStreamId = INVALID_INT;
        s32 unfoldStreamId = INVALID_INT;
        (void)hrtGetStreamId(stream[0], ctrlStreamId);
        (void)hrtGetStreamId(stream[1], unfoldStreamId);
        HCCL_RUN_INFO(
            "[HcomSetAttachedStream] register attached streams: group[%s], commId[%s], graphId[%u], "
            "ctrlStream[%p](id[%d]), unfoldStream[%p](id[%d])",
            group, hcclCommInV2->GetIdentifier().c_str(), graphId, stream[0], ctrlStreamId, stream[1], unfoldStreamId);
        return injectRet;
    }());

    std::shared_ptr<hccl::hcclComm> hcclComm = nullptr;
    std::vector<rtStream_t> rtStream(stream, stream + len);
    if (HcomGetCommByGroup(group, hcclComm) == HCCL_SUCCESS) {
        CHK_RET(hcclComm->SetAttachedStream(graphId, rtStream));
    } else {
        HCCL_WARNING("[HcomSetAttachedStream] HcclCommBase now don't support set attached stream");
    }
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif
