/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_RES_MGR_C_ADPT_PUB_H
#define HCOMM_RES_MGR_C_ADPT_PUB_H

#include <stdint.h>
#include "hcomm_res_defs.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// 默认实参是 C++ 语法：仅对 C++ 暴露，C 下不带默认值（须显式传参）以保证本头可被 C 独立编译。
#ifdef __cplusplus
HcommResult HcommResMgrInit(uint32_t devPhyId = UINT32_MAX);
#else
HcommResult HcommResMgrInit(uint32_t devPhyId);
#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCOMM_RES_MGR_C_ADPT_PUB_H
