/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CHANNEL_DFX_PUB_H
#define HCOMM_CHANNEL_DFX_PUB_H

#include <functional>
#include "hcomm_res_defs.h"
#include "hccl_common.h"
#include "task_param.h"

HcommResult
HcommDpuChannelRegisterDfx(ChannelHandle channel, std::function<HcclResult(const Hccl::TaskParam&, u64)> callback);

#endif // HCOMM_CHANNEL_DFX_PUB_H
