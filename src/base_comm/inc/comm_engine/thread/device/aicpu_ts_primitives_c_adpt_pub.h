/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 设备侧(AICPU)线程 primitives C 接口边界头。当前仅提供「按索引查询线程内 notify 的 runtime notifyId」，
// 供 coll_comm 的 dev_coll_comm_c_adpt.cc 解析 DfxOpInfo 的 cpuWaitAicpuNotifyId 使用。
#ifndef HCOMM_AICPU_TS_PRIMITIVES_C_ADPT_PUB_H
#define HCOMM_AICPU_TS_PRIMITIVES_C_ADPT_PUB_H

#include <stdint.h>
#include "hcomm_res_defs.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief 查询线程内指定索引 notify 的 runtime notifyId。
 * @param[in] thread 线程句柄（device 侧 AicpuTsThread）
 * @param[in] notifyIdx notify 索引
 * @param[out] notifyId 输出的 runtime notifyId
 * @return HcommResult HCOMM_SUCCESS 成功，否则对应错误码
 */
HcommResult HcommThreadGetNotifyId(ThreadHandle thread, uint32_t notifyIdx, uint32_t* notifyId);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCOMM_AICPU_TS_PRIMITIVES_C_ADPT_PUB_H
