/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_COMM_CONFIG_MGR_H
#define COLL_COMM_CONFIG_MGR_H

#include <mutex>

#include "env_aicpu_order_launch_config.h"
#include "env_ub_config.h"
#include "host_multi_qp_config.h"

namespace hccl {

/**
 * @brief 集合通信域进程级配置管理器。
 *
 * 统一持有并初始化集合通信域共享配置，由CollCommMgr管理其生命周期。
 */
class CollCommConfigMgr {
public:
    HcclResult Init();

    const EnvUbConfig& GetEnvUbConfig() const { return envUbConfig_; }
    const HostMultiQpConfig& GetHostMultiQpConfig() const { return hostMultiQpConfig_; }
    const EnvAicpuOrderLaunchConfig& GetEnvAicpuOrderLaunchConfig() const { return envAicpuOrderLaunchConfig_; }

private:
    EnvUbConfig envUbConfig_;
    HostMultiQpConfig hostMultiQpConfig_;
    EnvAicpuOrderLaunchConfig envAicpuOrderLaunchConfig_;
    std::mutex initMutex_;
    bool initialized_{false};
};

} // namespace hccl

#endif // COLL_COMM_CONFIG_MGR_H
