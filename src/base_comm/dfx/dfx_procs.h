/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_DFX_DFX_PROCS_H
#define HCOMM_DFX_DFX_PROCS_H

#include "hccl/hccl_types.h"
#include "hccl_common.h"

namespace hcomm {

/**
 * @brief Profiling 占位节点（当前无实体）。
 */
class ProfilingProc {
public:
    void DeInit() {}

    ~ProfilingProc() = default;

public:
    ProfilingProc() = default;

private:
    ProfilingProc(const ProfilingProc&) = delete;
    ProfilingProc& operator=(const ProfilingProc&) = delete;
};

/**
 * @brief DFX 聚合节点。成员声明顺序 = 析构契约反序（README R3）。
 */
class DfxProc {
public:
    void DeInit() { profilingProc_.DeInit(); }

    ProfilingProc& GetProfilingProc() { return profilingProc_; }

public:
    ~DfxProc() = default;

public:
    DfxProc() = default;

private:
    DfxProc(const DfxProc&) = delete;
    DfxProc& operator=(const DfxProc&) = delete;

    // ExceptionHandle/ExceptionCallbackMgr 为 AICPU 专属，host 树不持有
    ProfilingProc profilingProc_; // [6] 基础设施
};

} // namespace hcomm

#endif // HCOMM_DFX_DFX_PROCS_H
