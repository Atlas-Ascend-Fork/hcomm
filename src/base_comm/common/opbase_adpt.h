/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_OPBASE_ADPT_H
#define HCCL_OPBASE_ADPT_H

#include "hccl_types.h"
#include "hccl/base.h"

extern thread_local s32 g_hcclDeviceId;
s32 HcclGetThreadDeviceId();
void HcclSetThreadDeviceId(s32 deviceId);
HcclResult HcclDeviceRefresh(s32& deviceLogicId);

#endif
