/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_thread_process.h"

extern "C" {
__attribute__((visibility("default"))) uint32_t RunAicpuThreadInit(void* args)
{
    CHK_PTR_NULL(args);
    uint64_t devAddr = *reinterpret_cast<uint64_t*>(args);
    ThreadMgrAicpuParam* param = reinterpret_cast<ThreadMgrAicpuParam*>(devAddr);
    HCCL_INFO(
        "[RunAicpuThreadInit] threadNum[%u], deviceLogicId[%d], deviceType[%u]", param->threadNum, param->deviceLogicId,
        param->deviceType);
    return AicpuThreadProcess::AicpuThreadInit(param);
}

__attribute__((visibility("default"))) uint32_t RunAicpuThreadDestroy(void* args)
{
    CHK_PTR_NULL(args);
    uint64_t devAddr = *reinterpret_cast<uint64_t*>(args);
    ThreadMgrAicpuParam* param = reinterpret_cast<ThreadMgrAicpuParam*>(devAddr);
    HCCL_INFO("[RunAicpuThreadDestroy] threadNum[%u]", param->threadNum);
    return AicpuThreadProcess::AicpuThreadDestroy(param);
}

__attribute__((visibility("default"))) uint32_t RunAicpuThreadSupplementNotify(void* args)
{
    CHK_PTR_NULL(args);
    uint64_t devAddr = *reinterpret_cast<uint64_t*>(args);
    ThreadMgrAicpuParam* param = reinterpret_cast<ThreadMgrAicpuParam*>(devAddr);
    HCCL_INFO(
        "[RunAicpuThreadSupplementNotify] threadNum[%u], deviceLogicId[%d], deviceType[%u]", param->threadNum,
        param->deviceLogicId, param->deviceType);
    return AicpuThreadProcess::AicpuThreadSupplementNotify(param);
}
}
