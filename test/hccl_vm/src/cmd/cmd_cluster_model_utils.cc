/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cmd_cluster_model_utils.h"
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "sim_common_api.h"
#include "sim_log.h"

namespace {
constexpr const char* HVM_MODEL_ENV_KEY = "HCCL_VM_MODEL";
constexpr const char* HVM_POD_NUM_ENV_KEY = "HCCL_VM_POD_NUM";
constexpr const char* HVM_SERVER_COUNT_ENV_KEY = "HCCL_VM_SERVER_COUNT";
constexpr const char* HVM_RANK_NUM_ENV_KEY = "HCCL_VM_RANK_NUM";
constexpr const char* HVM_PROCESSES_PER_SERVER_ENV_KEY = "HCCL_VM_PROCESSES_PER_SERVER";
} // namespace

bool ParseYamlTopo(const std::string& fileName, TopoMeta& topo) { return ParseTopoMetaYaml(fileName, topo); }
