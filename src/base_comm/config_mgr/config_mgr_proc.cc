/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "config_mgr_proc.h"

// legacy910（libhccl_v2.so）Meyers 单例，仅经 ConfigMgrProc 转发访问
#include "comm_configer.h"

namespace hcomm {

hccl::CommConfiger& ConfigMgrProc::GetCommConfiger() { return hccl::CommConfiger::GetInstance(); }

} // namespace hcomm
