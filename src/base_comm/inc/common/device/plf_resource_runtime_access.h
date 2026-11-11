/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_PLF_RESOURCE_RUNTIME_ACCESS_H
#define HCOMM_PLF_RESOURCE_RUNTIME_ACCESS_H

#include <cstdint>
#include "hccl/base.h"

// base_comm 内部桥接头：仅用于 legacy 资源层迁移期，不对外暴露。
#ifdef HCOMM_PLF_RESOURCE
#include <atomic>
#include "common/device/plf_resource_runtime_pub.h"
#include "log.h"

namespace Hccl {
inline bool PlfIsBatchLaunchMode()
{
    const auto& ops = GetPlfResourceRuntimeOps();
    return ops.isBatchLaunchMode == nullptr ? false : ops.isBatchLaunchMode();
}

inline u32 PlfGetSqFullTimeOut()
{
    const auto& ops = GetPlfResourceRuntimeOps();
    return ops.getSqFullTimeOut == nullptr ? PLF_DEVICE_EXEC_TIMEOUT_DEFAULT : ops.getSqFullTimeOut();
}

inline HcclResult PlfHandleDispatchAllStreams()
{
    const auto& ops = GetPlfResourceRuntimeOps();
    return ops.handleDispatchAllStreams == nullptr ? HCCL_E_INTERNAL : ops.handleDispatchAllStreams();
}

inline bool PlfGetProfL1State()
{
    const auto& ops = GetPlfResourceRuntimeOps();
    if (ops.getProfL1State == nullptr) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            HCCL_WARNING("[PlfResourceRuntime] getProfL1State callback is not registered");
        }
        return false;
    }
    return ops.getProfL1State();
}

inline HcclResult PlfDumpSqe(const uint8_t* sqePtr)
{
    const auto& ops = GetPlfResourceRuntimeOps();
    return ops.dumpSqe == nullptr ? HCCL_E_INTERNAL : ops.dumpSqe(sqePtr);
}

inline HcclResult PlfDumpWqe(const uint8_t* wqePtr)
{
    const auto& ops = GetPlfResourceRuntimeOps();
    return ops.dumpWqe == nullptr ? HCCL_E_INTERNAL : ops.dumpWqe(wqePtr);
}

inline u64 PlfGetDebugConfig()
{
    const auto& ops = GetPlfResourceRuntimeOps();
    return ops.getPlfDebugConfig == nullptr ? 0 : ops.getPlfDebugConfig();
}
} // namespace Hccl
#else
#include "aicpu_task_utils.h"
#include "dfx/device/dfx_profiling_handler_lite_pub.h"
#include "plf_debug_config.h"

bool IsBatchLaunchMode();
u32 GetSqFullTimeOut();
HcclResult HandleDispatchAllStreams();

namespace Hccl {
inline bool PlfIsBatchLaunchMode() { return IsBatchLaunchMode(); }
inline u32 PlfGetSqFullTimeOut() { return GetSqFullTimeOut(); }
inline HcclResult PlfHandleDispatchAllStreams() { return HandleDispatchAllStreams(); }
inline bool PlfGetProfL1State() { return DfxProfilingHandlerLite::GetInstance().GetProfL1State(); }
inline HcclResult PlfDumpSqe(const uint8_t* sqePtr) { return hcomm::AicpuTaskUtils::DumpSqeContent(sqePtr); }
inline HcclResult PlfDumpWqe(const uint8_t* wqePtr) { return hcomm::AicpuTaskUtils::DumpWqeContent(wqePtr); }
inline u64 PlfGetDebugConfig() { return GetPlfDebugConfigValue(); }
} // namespace Hccl
#endif

#endif // HCOMM_PLF_RESOURCE_RUNTIME_ACCESS_H
