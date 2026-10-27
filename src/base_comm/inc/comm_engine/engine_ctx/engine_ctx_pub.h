/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// EngineCtx C interfaces (internal use only, not exported). Relocated from common/hcomm_c_adpt_pub.h.
#ifndef HCOMM_ENGINE_CTX_PUB_H
#define HCOMM_ENGINE_CTX_PUB_H

#include <stdint.h>
#include "hcomm_res_defs.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

HcommResult HcommEngineCtxCreate(CommEngine engine, uint64_t size, void** ctx);

HcommResult HcommEngineCtxDestroy(CommEngine engine, void* ctx);

HcommResult HcommEngineCtxCopy(CommEngine engine, void* dstCtx, const void* srcCtx, uint64_t size);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCOMM_ENGINE_CTX_PUB_H
