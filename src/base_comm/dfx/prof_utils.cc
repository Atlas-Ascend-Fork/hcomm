/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dfx/prof_utils_pub.h"
#include "dfx_dlprof_function.h"

namespace hcomm {
// 首次调用时完成一次 DfxDlProfFunctionInit（dlopen libprofapi.so），Init 自带双检锁，重复调用幂等
static void EnsureProfFunctionInit() { (void)Hccl::DfxDlProfFunction::GetInstance().DfxDlProfFunctionInit(); }

uint64_t GetProfCycleTime()
{
    EnsureProfFunctionInit();
    return Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
}

uint64_t ProfStr2Id(const char* hashInfo, uint32_t length)
{
    EnsureProfFunctionInit();
    return Hccl::DfxDlProfFunction::GetInstance().dlMsprofStr2Id(hashInfo, length);
}
} // namespace hcomm
