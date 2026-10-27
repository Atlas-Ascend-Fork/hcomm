/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 内存类型（MemHandle/CommMemInfo），C++ only，仅 coll_comm 消费。
// memTag 长度上限 HCOMM_RES_TAG_MAX_LEN 定义于 include/hcomm_res_defs.h（HcommMemReg 契约）。
#ifndef HCOMM_MEM_C_ADPT_H
#define HCOMM_MEM_C_ADPT_H

#include "hcomm_res_defs.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef HcommMemHandle MemHandle;

struct CommMemInfo {
    CommMem mem{};
    void* bufferHandle{nullptr};
    char memTag[HCOMM_RES_TAG_MAX_LEN + 1] = {0};
};

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCOMM_MEM_C_ADPT_H
