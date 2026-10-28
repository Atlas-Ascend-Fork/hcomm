/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_AICPU_COMM_ENGINE_BIN_PROC_H
#define HCOMM_AICPU_COMM_ENGINE_BIN_PROC_H

#include <mutex>
#include "hccl/hccl_types.h"
#include "acl/acl_rt.h"
#include "hcomm_res_defs.h"

namespace hcomm {

/**
 * @brief AICPU 通信引擎 kernel bin 句柄（进程级节点，全进程一份）
 *
 * 无卸载逻辑，bin 句柄随进程退出由 OS 回收
 */
class AicpuCommEngineBinProc {
public:
    void DeInit() {}

    HcclResult EnsureKernelBinLoaded(CommEngine engine);
    aclrtBinHandle GetBinHandle() const { return binHandle_; }

public:
    ~AicpuCommEngineBinProc() = default;

public:
    AicpuCommEngineBinProc() = default;

private:
    AicpuCommEngineBinProc(const AicpuCommEngineBinProc&) = delete;
    AicpuCommEngineBinProc& operator=(const AicpuCommEngineBinProc&) = delete;

    aclrtBinHandle binHandle_{nullptr};
    std::mutex binHandleMtx_;
};

} // namespace hcomm

#endif // HCOMM_AICPU_COMM_ENGINE_BIN_PROC_H
