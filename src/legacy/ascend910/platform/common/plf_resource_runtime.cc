/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/device/plf_resource_runtime_pub.h"
#include "rtsq_base.h"
#include "log.h"
#include <atomic>

namespace Hccl {
namespace {

    // legacy 目录日落前的临时回调表，资源层通过它访问 base_comm 运行时能力。
    // legacy 目录移除后，本文件及注册接口需一并删除。
    bool DefaultIsBatchLaunchMode() { return false; }

    u32 DefaultGetSqFullTimeOut() { return RTSQ_FULL_TIMEOUT_DEFAULT; }

    HcclResult DefaultHandleDispatchAllStreams() { return HCCL_E_INTERNAL; }

    HcclResult DefaultDumpSqe(const uint8_t* sqePtr)
    {
        (void)sqePtr;
        return HCCL_SUCCESS;
    }

    HcclResult DefaultDumpWqe(const uint8_t* wqePtr)
    {
        (void)wqePtr;
        return HCCL_SUCCESS;
    }

    u64 DefaultGetPlfDebugConfig() { return 0; }

    const PlfResourceRuntimeOps DEFAULT_OPS = {
        DefaultIsBatchLaunchMode, DefaultGetSqFullTimeOut, DefaultHandleDispatchAllStreams, nullptr,
        DefaultDumpSqe,           DefaultDumpWqe,          DefaultGetPlfDebugConfig,
    };

    std::atomic<const PlfResourceRuntimeOps*> g_plfResourceRuntimeOps{};
    std::atomic<PlfMarkCompactReportOp> g_markCompactReportOp{};
    std::atomic<PlfDeviceExecTimeoutGetter> g_deviceExecTimeoutGetter{};
    constexpr u32 DEFAULT_DEVICE_EXEC_TIMEOUT = PLF_DEVICE_EXEC_TIMEOUT_DEFAULT;

} // namespace

const PlfResourceRuntimeOps& GetPlfResourceRuntimeOps()
{
    const auto* ops = g_plfResourceRuntimeOps.load(std::memory_order_acquire);
    if (ops == nullptr) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            HCCL_WARNING("[PlfResourceRuntime] runtime ops not registered, fallback to DEFAULT_OPS");
        }
        return DEFAULT_OPS;
    }
    return *ops;
}

static void SetPlfResourceRuntimeOps(const PlfResourceRuntimeOps* ops)
{
    g_plfResourceRuntimeOps.store(ops == nullptr ? &DEFAULT_OPS : ops, std::memory_order_release);
}

static void SetMarkCompactReportOpState(PlfMarkCompactReportOp op)
{
    g_markCompactReportOp.store(op, std::memory_order_release);
}

static PlfMarkCompactReportOp GetMarkCompactReportOpState()
{
    return g_markCompactReportOp.load(std::memory_order_acquire);
}

static void SetDeviceExecTimeoutGetterState(PlfDeviceExecTimeoutGetter getter)
{
    g_deviceExecTimeoutGetter.store(getter, std::memory_order_release);
}

static u32 GetDeviceExecTimeoutState()
{
    const auto getter = g_deviceExecTimeoutGetter.load(std::memory_order_acquire);
    return getter == nullptr ? DEFAULT_DEVICE_EXEC_TIMEOUT : getter();
}

} // namespace Hccl

extern "C" void HcclRegisterPlfResourceRuntimeOps(const Hccl::PlfResourceRuntimeOps* ops)
{
    Hccl::SetPlfResourceRuntimeOps(ops);
}

extern "C" void HcclSetPlfMarkCompactReportOp(Hccl::PlfMarkCompactReportOp op)
{
    Hccl::SetMarkCompactReportOpState(op);
}

extern "C" Hccl::PlfMarkCompactReportOp HcclGetPlfMarkCompactReportOp() { return Hccl::GetMarkCompactReportOpState(); }

extern "C" void HcclSetPlfDeviceExecTimeoutGetter(Hccl::PlfDeviceExecTimeoutGetter getter)
{
    Hccl::SetDeviceExecTimeoutGetterState(getter);
}

extern "C" u32 HcclGetPlfDeviceExecTimeout() { return Hccl::GetDeviceExecTimeoutState(); }
