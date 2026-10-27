/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_HCCL_COMM_OP_H
#define HCOMM_HCCL_COMM_OP_H

#include <functional>
#include "hcomm_res_defs.h"
#include "hccl_common.h"
#include "task_param.h"

int32_t HcommChannelRegisterDfx(
    ChannelHandle channel, std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback);

#endif // HCOMM_HCCL_COMM_OP_H
