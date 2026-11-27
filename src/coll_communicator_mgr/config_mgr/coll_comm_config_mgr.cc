/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_config_mgr.h"

namespace hccl {

HcclResult CollCommConfigMgr::Init()
{
    std::lock_guard<std::mutex> lock(initMutex_);
    if (initialized_) {
        return HCCL_SUCCESS;
    }

    CHK_RET(envUbConfig_.Parse());
    CHK_RET(hostMultiQpConfig_.Parse());
    CHK_RET(envAicpuOrderLaunchConfig_.Parse());
    initialized_ = true;
    return HCCL_SUCCESS;
}

} // namespace hccl
