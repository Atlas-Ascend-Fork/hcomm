/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Description: 用于存放指令公共功能文件
 * Author: zhanhaifeng、caiyifan
 * Create: 2025-06-19
 */

#ifndef HCCLV2_CCU_TRANSFORM_TASK_COMMON_V3_H
#define HCCLV2_CCU_TRANSFORM_TASK_COMMON_V3_H

#include <cstdint>
#include <map>
#include <set>
#include <queue>
#include <vector>
#include <memory>
#include <unordered_map>
#include <hccl_types.h>

#include "base.h"
#include "log.h"
#include "data_slice.h"
#include "ccu_microcode_v1.h"
#include "ccu_instr_info.h"
#include "../task_def_v3.h"
#include "storage_manager.h"
#include "data_type.h"

using namespace hcomm;
namespace HcclSim {
namespace TaskGraphGeneratorV3 {

    using AllRankChannelInfoV3 = std::map<DeviceId, std::map<uint32_t, ChannelsPerDie>>;
    extern AllRankChannelInfoV3& g_allRankChannelInfo;
    using AllDeviceHalfRTTInfoV3 = std::map<DeviceId, std::map<uint32_t, std::vector<HcclSim::HalfRTTInfo>>>;
    extern AllDeviceHalfRTTInfoV3& g_allDeviceHalfRTTInfo;

    std::string ParseMSList(const CcuRep::CcuInstr* instr);

    // CCU 资源统计（dfx）：channel 表查询统一封装。翻译层引用 channel 必须经由这两个函数，
    // 以便将 channelId 记入当前算子桶（引用口径，见 docs/ccu_resource_statistics_design.md 3.1）；
    // 封装之外禁止直接访问 g_allRankChannelInfo。
    bool HasChannelInfo(DeviceId deviceId, uint32_t dieId, uint32_t channelId);
    const RemoteDieInfo& GetChannelInfo(DeviceId deviceId, uint32_t dieId, uint32_t channelId);
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif
