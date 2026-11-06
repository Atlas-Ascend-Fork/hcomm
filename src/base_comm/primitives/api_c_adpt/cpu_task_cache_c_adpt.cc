/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_primitives.h"
#include "hccl_types.h"
#include "log.h"

HcommResult HcommAicpuTsTaskCacheLookup(const char* tag, bool* isHit)
{
    (void)tag;
    (void)isHit;
    HCCL_ERROR("[%s] aicpu task cache is not supported on host.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcommResult HcommAicpuTsTaskCacheStart(const char* tag, void** addrs, uint64_t* sizes, uint64_t count)
{
    (void)tag;
    (void)addrs;
    (void)sizes;
    (void)count;
    HCCL_ERROR("[%s] aicpu task cache is not supported on host.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcommResult HcommAicpuTsTaskCacheEnd(const char* tag)
{
    (void)tag;
    HCCL_ERROR("[%s] aicpu task cache is not supported on host.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcommResult HcommAicpuTsTaskCacheExecute(const char* tag, void** addrs, uint64_t* sizes, uint64_t count)
{
    (void)tag;
    (void)addrs;
    (void)sizes;
    (void)count;
    HCCL_ERROR("[%s] aicpu task cache is not supported on host.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcommResult HcommAicpuTsTaskCacheClear(const char* tag)
{
    (void)tag;
    HCCL_ERROR("[%s] aicpu task cache is not supported on host.", __func__);
    return HCCL_E_NOT_SUPPORT;
}
