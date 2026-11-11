/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_PLF_RESOURCE_RUNTIME_PUB_H
#define HCOMM_PLF_RESOURCE_RUNTIME_PUB_H

#include <cstdint>
#include "hccl/base.h"

namespace Hccl {

// 临时桥接接口：legacy 目录尚未日落期间，由 base_comm_aicpu 注册资源层所需运行时能力。
// legacy 目录移除后，本接口及对应注册实现需一并删除。
struct PlfResourceRuntimeOps {
    bool (*isBatchLaunchMode)();
    u32 (*getSqFullTimeOut)();
    HcclResult (*handleDispatchAllStreams)();
    bool (*getProfL1State)();
    HcclResult (*dumpSqe)(const uint8_t* sqePtr);
    HcclResult (*dumpWqe)(const uint8_t* wqePtr);
    u64 (*getPlfDebugConfig)();
};

using PlfMarkCompactReportOp = void (*)(void* dfxLite, const void* opInfo);
using PlfDeviceExecTimeoutGetter = u32 (*)();

// 27*68 notifywait 默认等待时长，与 EnvConfig.hcclExecTimeout 默认值保持一致。
constexpr u32 PLF_DEVICE_EXEC_TIMEOUT_DEFAULT = 1836U;

const PlfResourceRuntimeOps& GetPlfResourceRuntimeOps();

} // namespace Hccl

extern "C" void HcclRegisterPlfResourceRuntimeOps(const Hccl::PlfResourceRuntimeOps* ops);
extern "C" const Hccl::PlfResourceRuntimeOps* HcclGetPlfResourceRuntimeOps();
extern "C" void HcclSetPlfMarkCompactReportOp(Hccl::PlfMarkCompactReportOp op);
extern "C" Hccl::PlfMarkCompactReportOp HcclGetPlfMarkCompactReportOp();
extern "C" void HcclSetPlfDeviceExecTimeoutGetter(Hccl::PlfDeviceExecTimeoutGetter getter);
extern "C" u32 HcclGetPlfDeviceExecTimeout();

#endif // HCOMM_PLF_RESOURCE_RUNTIME_PUB_H
