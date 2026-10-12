/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONVERT_CONFIG_GENERATOR_V3_H
#define HCCLV2_CCU_CONVERT_CONFIG_GENERATOR_V3_H

#include "hccl_types.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {

    // 生成 convert-ccu-microcode 所需的 JSON 配置文件（rule1 + rule3），所有 device/die 统一保存。
    // rule1.max_used_xn_id   : 来自 AllRankParamRecorder::curXn 中各 (deviceId, dieId) 的最大 xnId
    // rule3.reg_value_table  : 来自 AllRankParamRecorder::loopGroupRegSnapshot 中各 LoopGroup 处的 xpId/xmId value
    // 输出路径由 DumpV3Manager 管理，文件名固定为 ccu_convert_config.json。
    HcclResult DumpConvertConfig();

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif // HCCLV2_CCU_CONVERT_CONFIG_GENERATOR_V3_H
