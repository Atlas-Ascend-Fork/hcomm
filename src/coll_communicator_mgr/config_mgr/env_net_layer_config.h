/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENV_NET_LAYER_CONFIG_H
#define ENV_NET_LAYER_CONFIG_H

#include <mutex>

#include "net_layer_config.h"

namespace hccl {
// 仅管理 HCCL_USE_NET_LAYER；通信域参数优先级由 net_layer_config 处理。
class EnvNetLayerConfig {
public:
    static EnvNetLayerConfig& GetInstance();

    // 首次成功解析后进程内复用，包括未配置的默认值；失败不缓存且不修改输出。
    HcclResult GetSelection(NetLayerSelection& selection);
    // 仅供 UT 隔离不同进程配置场景，不用于运行时改配。
    void ResetParsed();

private:
    EnvNetLayerConfig() = default;
    std::mutex parseMutex_;
    bool parsed_{false};
    NetLayerSelection selection_;
};
} // namespace hccl

#endif // ENV_NET_LAYER_CONFIG_H
