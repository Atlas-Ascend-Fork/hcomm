/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// CCU instance public interface. Keeps the L2 layer off the internal CcuInstanceMgr.
#ifndef HCOMM_CCU_INSTANCE_PUB_H
#define HCOMM_CCU_INSTANCE_PUB_H

#include <cstdint>
#include "ccu_types.h"
#include "hcomm/ccu/hcomm_res_ccu_entity_defs.h"

namespace hcomm {

// Returns true if the CCU instance identified by insHandle exists on the device.
bool CcuInstanceExists(int32_t deviceLogicId, CcuInsHandle insHandle);

// 构建 CCU 实例完整实体快照（内部填充 ABI 头与 ccuInsKey 后查询）。
CcuResult CcuQueryInstanceEntity(int32_t deviceLogicId, CcuInsHandle insHandle, HcommCcuInstanceEntity& insEntity);

} // namespace hcomm

#endif // HCOMM_CCU_INSTANCE_PUB_H
