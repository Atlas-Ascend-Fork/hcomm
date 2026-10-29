/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_loopcall_v1.h"

#include "ccu_api_exception.h"
#include "ccu_log.h"

namespace hcomm {
namespace CcuRep {

    LoopCall::LoopCall(CcuRepContext* context, const std::string& label) : context(context), label(label)
    {
        // repLoopCall = std::make_shared<CcuRepLoopCall>(label);
    }

    CcuResult LoopCall::AppendToContext()
    {
        if (context == nullptr) {
            HCCL_ERROR("[%s] context is nullptr, loopCall", __func__);
            return CCU_E_PTR;
        }
        CCU_TRY_CATCH_RETURN(context->Append(repLoopCall););
        return CCU_SUCCESS;
    }

}; // namespace CcuRep
}; // namespace hcomm
