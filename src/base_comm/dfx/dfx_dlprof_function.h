/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// dlopen libprofapi.so 获取 msprof 上报/查询函数（dlsym 全部成功后覆盖 stub，失败保留 stub）。
// coll_comm 经 inc/dfx/prof_utils_pub.h 的 GetProfCycleTime/ProfStr2Id 使用，
// 两接口内部自带首次 DfxDlProfFunctionInit（dlopen libprofapi.so），无需显式初始化。
#ifndef HCOMM_DFX_DLPROF_FUNCTION_H
#define HCOMM_DFX_DLPROF_FUNCTION_H

#include <functional>
#include <mutex>
#include <atomic>
#include <dlfcn.h>
#include <hccl/hccl_types.h>
#include "orion_adapter_rts.h"

namespace Hccl {
class DfxDlProfFunction {
public:
    virtual ~DfxDlProfFunction();
    static DfxDlProfFunction& GetInstance();
    HcclResult DfxDlProfFunctionInit();
    std::function<int32_t(uint32_t, ProfCommandHandle)> dlMsprofRegisterCallback{};
    std::function<int32_t(uint16_t, uint32_t, const char*)> dlMsprofRegTypeInfo{};
    std::function<int32_t(uint32_t, const MsprofApi*)> dlMsprofReportApi{};
    std::function<int32_t(uint32_t, const VOID_PTR, uint32_t)> dlMsprofReportCompactInfo{};
    std::function<int32_t(uint32_t, const VOID_PTR, uint32_t)> dlMsprofReportAdditionalInfo{};
    std::function<int32_t(uint32_t, const VOID_PTR, uint32_t)> dlMsprofReportBatchAdditionalInfo{};
    std::function<uint64_t(const char*, uint32_t)> dlMsprofStr2Id{};
    std::function<uint64_t(void)> dlMsprofSysCycleTime{};

private:
    std::atomic<bool> initializedFlag_{false};
    void* handle_{};
    std::mutex handleMutex_;
    DfxDlProfFunction(const DfxDlProfFunction&) = delete;
    DfxDlProfFunction& operator=(const DfxDlProfFunction&) = delete;
    DfxDlProfFunction();
    HcclResult DfxDlProfFunctionInterInit();
    void DfxDlProfFunctionStubInit();
};
} // namespace Hccl
#endif
