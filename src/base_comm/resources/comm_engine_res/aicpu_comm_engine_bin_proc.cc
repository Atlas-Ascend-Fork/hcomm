/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_comm_engine_bin_proc.h"

#include "log.h"
#include "launch_aicpu.h"
#include "launch_device.h"
#include "common/comm_engine_utils_pub.h"

namespace hcomm {

HcclResult AicpuCommEngineBinProc::EnsureKernelBinLoaded(CommEngine engine)
{
    if (engine != COMM_ENGINE_AICPU && engine != COMM_ENGINE_AICPU_TS) {
        HCCL_INFO(
            "[%s] engine[%s] kernel loading not required", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        return HCCL_SUCCESS;
    }
    std::lock_guard<std::mutex> lock(binHandleMtx_);
    if (binHandle_ != nullptr) {
        return HCCL_SUCCESS;
    }
    std::string jsonPath;
    CHK_RET(hccl::GetKernelFilePath(jsonPath));
    jsonPath += "ccl_kernel.json";

    HcclResult ret = hccl::LoadBinaryFromFile(jsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0, binHandle_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] load aicpu file fail, path[%s]", __func__, jsonPath.c_str()), ret);
    return HCCL_SUCCESS;
}

} // namespace hcomm
