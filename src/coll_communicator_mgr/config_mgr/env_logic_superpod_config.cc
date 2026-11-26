/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "env_logic_superpod_config.h"
#include "log.h"

namespace hccl {

EnvLogicSuperPodConfig::EnvLogicSuperPodConfig()
    : logicSuperPodId_(
          "HCCL_LOGIC_SUPERPOD_ID", std::string{},
          [](const std::string& value, bool& parseOk) {
              parseOk = true;
              return value == "EmptyString" ? std::string{} : value;
          },
          [](const std::string& value) {
              return value.size() <= MAX_LEN_OF_LOGIC_SUPERPOD_ID;
          })
{}

EnvLogicSuperPodConfig& EnvLogicSuperPodConfig::GetInstance()
{
    static EnvLogicSuperPodConfig config;
    return config;
}

HcclResult EnvLogicSuperPodConfig::Parse()
{
    std::lock_guard<std::mutex> lock(parseMutex_);
    if (parsed_) {
        return HCCL_SUCCESS;
    }
    HcclResult ret = logicSuperPodId_.Parse();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[Parse][LogicSuperPodId]Invalid HCCL_LOGIC_SUPERPOD_ID env len, len is bigger than [%u].",
            MAX_LEN_OF_LOGIC_SUPERPOD_ID);
        return ret;
    }
    const auto& logicId = logicSuperPodId_.Get();
    if (logicId.empty()) {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_LOGIC_SUPERPOD_ID is not set, default value[%s]", logicId.c_str());
    } else {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_LOGIC_SUPERPOD_ID set by environment to [%s]", logicId.c_str());
    }
    parsed_ = true;
    return HCCL_SUCCESS;
}

} // namespace hccl
