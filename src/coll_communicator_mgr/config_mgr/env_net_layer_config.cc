/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "env_net_layer_config.h"

#include "config/base_config_pub.h"
#include "log.h"

namespace hccl {
EnvNetLayerConfig& EnvNetLayerConfig::GetInstance()
{
    static EnvNetLayerConfig config;
    return config;
}

HcclResult EnvNetLayerConfig::GetSelection(NetLayerSelection& selection)
{
    std::lock_guard<std::mutex> lock(parseMutex_);
    if (!parsed_) {
        const std::string expression = hcomm::GetEnv("HCCL_USE_NET_LAYER");
        CHK_RET(NetLayerSelection::Parse(expression, selection_));
        if (expression.empty()) {
            HCCL_RUN_INFO("[HCCL_ENV] HCCL_USE_NET_LAYER is not set.");
        } else {
            HCCL_RUN_INFO(
                "[HCCL_ENV] HCCL_USE_NET_LAYER set by environment to [%s]",
                selection_.GetNormalizedExpression().c_str());
        }
        parsed_ = true;
    }
    selection = selection_;
    return HCCL_SUCCESS;
}

void EnvNetLayerConfig::ResetParsed()
{
    std::lock_guard<std::mutex> lock(parseMutex_);
    parsed_ = false;
}
} // namespace hccl
