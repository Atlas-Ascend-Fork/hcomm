/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENV_LOGIC_SUPERPOD_CONFIG_H
#define ENV_LOGIC_SUPERPOD_CONFIG_H

#include <mutex>
#include "config/base_config_pub.h"

namespace hccl {

// 进程共享配置首次Parse成功后缓存（包括空值），失败不缓存。
class EnvLogicSuperPodConfig {
public:
    EnvLogicSuperPodConfig();
    static EnvLogicSuperPodConfig& GetInstance();
    HcclResult Parse();
    const std::string& GetLogicSuperPodId() const { return logicSuperPodId_.Get(); }

private:
    hcomm::EnvField<std::string> logicSuperPodId_;
    std::mutex parseMutex_;
    bool parsed_{false};
};

} // namespace hccl

#endif // ENV_LOGIC_SUPERPOD_CONFIG_H
