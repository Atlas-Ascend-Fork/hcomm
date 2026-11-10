/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_TYPES_PUB_H
#define DPU_TYPES_PUB_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include "hccl_types.h"
#include "task_param.h"

namespace Hccl {

// DPU 任务回调 ABI（跨 so 共用：宿主侧 TaskService 注册、hccl_v2 兼容入口转发）。
using CallbackTemplate = std::function<int32_t(uint64_t, int32_t)>;
using ProfCallbackTemplate = std::function<HcclResult(const TaskParam&, uint64_t)>;
using ReportCallbackTemplate = std::function<HcclResult()>;
using GetDpuTaskInfoCallbackTemplate = std::function<std::pair<uint32_t, uint32_t>()>;

} // namespace Hccl

#endif // DPU_TYPES_PUB_H
