/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENV_AICPU_ORDER_LAUNCH_CONFIG_H
#define ENV_AICPU_ORDER_LAUNCH_CONFIG_H

#include <mutex>

#include "config_mgr/base_config.h"

namespace hccl {

/**
 * @brief AI CPU保序任务下发开关（HCCL_AICPU_ORDER_LAUNCH_ENABLE，HCCL 语义的环境变量落在通信域管理层）。
 *
 * CollCommConfigMgr::Init()调用Parse()完成解析：非法配置当场报错（EI0001），
 * 合法值缓存在字段内；读取期通过 GetAicpuOrderLaunchEnable() 直接取缓存值，不再解析。
 * hccl 侧经 HcclConfigGetInfo 的 HCCL_CONFIG_TYPE_AICPU_ORDER_LAUNCH_ENABLE 查询项获取该值。
 */
class EnvAicpuOrderLaunchConfig {
public:
    HcclResult Parse();
    bool GetAicpuOrderLaunchEnable() const { return aicpuOrderLaunchEnable_.Get(); }

private:
    // 默认值：开启AI CPU保序任务下发
    static constexpr bool AICPU_ORDER_LAUNCH_ENABLE_DEFAULT = true;

    // 环境变量字段（解析器复用通用模板，取值仅支持1或0，默认开启）
    hcomm::EnvField<bool> aicpuOrderLaunchEnable_{
        "HCCL_AICPU_ORDER_LAUNCH_ENABLE", AICPU_ORDER_LAUNCH_ENABLE_DEFAULT, hcomm::StrToBool};

    // 只缓存成功：首次（通信域初始化）解析成功后不再重复；失败不缓存，每次初始化尝试均报错
    std::mutex parseMutex_;
    bool parsed_{false};
    HcclResult parseRet_{HCCL_SUCCESS};
};

} // namespace hccl

#endif // ENV_AICPU_ORDER_LAUNCH_CONFIG_H
