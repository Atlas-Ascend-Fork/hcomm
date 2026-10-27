/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include <string>
#include "dpu_taskexp_kernel_launch.h"
#include "dfx/device/dpu_taskexp_kernel_param_pub.h"
#include "launch_aicpu.h"
#include "manager_common.h"
#include "comm_configer.h"

namespace hccl {
namespace {
    constexpr char DPU_TASKEXP_KERNEL_NAME[] = "HcclDpuTaskexpShmemRestore";
} // namespace

HcclResult LaunchDpuTaskexpShmemRestoreKernel(
    aclrtBinHandle binHandle, const std::string& commId, uint32_t devId, void* taskexpVa, uint64_t memorySize)
{
    CHK_PTR_NULL(binHandle);
    CHK_PTR_NULL(taskexpVa);

    Hccl::DpuTaskexpKernelParam param{};
    errno_t sRet = strncpy_s(param.commId, sizeof(param.commId), commId.c_str(), sizeof(param.commId) - 1);
    CHK_PRT_RET(
        sRet != EOK, HCCL_ERROR("[%s] strncpy_s commId failed, comm[%s], ret[%d]", __func__, commId.c_str(), sRet),
        HCCL_E_INTERNAL);
    param.taskexceptionVa = taskexpVa;
    param.memorySize = memorySize;
    param.deviceId = devId;

    // 与 RunAicpuCommInit 等 L2 AICPU kernel 同款下发路径：局部流 + AICPU 模式 + 下发后阻塞同步
    Stream localStream(StreamType::STREAM_TYPE_ONLINE);
    constexpr u32 aicpuStreamMode = 1;
    CHK_RET(hrtStreamSetMode(localStream.ptr(), aicpuStreamMode));

    u16 timeOut = NOTIFY_DEFAULT_WAIT_TIME > std::numeric_limits<uint16_t>::max() ?
                      std::numeric_limits<uint16_t>::max() :
                      static_cast<u16>(NOTIFY_DEFAULT_WAIT_TIME);
    CHK_RET(AicpuAclKernelLaunch(
        localStream.ptr(), static_cast<void*>(&param), sizeof(param), binHandle, DPU_TASKEXP_KERNEL_NAME, true,
        timeOut));
    // AICPU 侧登记完成后本次下发即结束，同步等待避免后续流程读到未登记的登记表
    CHK_RET(hcclStreamSynchronize(localStream.ptr(), CommConfiger::GetInstance().GetCommConfigExecTimeOut(commId)));
    HCCL_INFO("[%s] launch dpu taskexp kernel success, comm[%s], devId[%u]", __func__, commId.c_str(), devId);
    return HCCL_SUCCESS;
}

} // namespace hccl
