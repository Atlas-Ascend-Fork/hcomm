/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dpu_task_exception_lite.h"
#include "hcclCommTaskExceptionLite.h"
#include "stream_lite.h"
#include "aicpu_indop_env.h"
#include "hcomm_task_scheduler_error.h"
#include "task_struct_v2.h"
#include "dlhal_function_v2.h"

namespace hcomm {

HcclResult DpuTaskExceptionLite::IsHandleDpuStop(uint8_t* taskexceptionVa, bool& isStop)
{
    uint8_t stopSignal = 0;
    errno_t ret = memcpy_s(
        &stopSignal, sizeof(stopSignal), taskexceptionVa,
        sizeof(stopSignal)); // 读标志位,第1字节，存放host侧发送是否停止的信号。
    if (ret != EOK) {
        HCCL_ERROR("[DpuTaskExceptionLite::%s] memcpy_s failed on flag, return[%d].", __func__, ret);
        return HCCL_E_MEMORY;
    }
    if (stopSignal == 1) {
        isStop = true;
        stopSignal = 0;
        ret = memcpy_s(
            taskexceptionVa, sizeof(stopSignal), &stopSignal,
            sizeof(stopSignal)); // 读标志位,第1字节，存放host侧发送是否停止的信号。
        if (ret != EOK) {
            HCCL_ERROR("[DpuTaskExceptionLite::%s] memcpy_s failed on flag, return[%d].", __func__, ret);
            return HCCL_E_MEMORY;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult DpuTaskExceptionLite::HandleDpuTaskexception(CollCommAicpu* aicpuComm)
{
    if (hcomm::GetTaskExceptionEnable() == false) { // taskException没开则跳过
        return HCCL_SUCCESS;
    }
    // 轮询taskexception共享内存
    auto commId = aicpuComm->GetIdentifier();
    auto* hcclCommDfxLite = aicpuComm->GetHcclCommDfxLite();
    CHK_PTR_NULL(hcclCommDfxLite);
    if (hcclCommDfxLite->IsTaskExpStopped()) {
        return HCCL_SUCCESS;
    }
    auto taskexceptionVa = reinterpret_cast<uint8_t*>(hcclCommDfxLite->GetTaskExpDevMem());
    if (taskexceptionVa == nullptr) {
        return HCCL_SUCCESS; // 非dpu场景
    }
    // 查是否要停止
    bool isStop = false;
    CHK_RET(IsHandleDpuStop(taskexceptionVa, isStop));
    if (isStop) {
        hcclCommDfxLite->MarkTaskExpStopped();
        return HCCL_SUCCESS;
    }
    // 查是否有错误
    uint16_t errorCode = 0;
    errno_t ret = memcpy_s(
        &errorCode, sizeof(errorCode), taskexceptionVa + sizeof(uint8_t),
        sizeof(errorCode)); // 读标志位,第2-3字节，存放HcclResult。
    if (ret != EOK) {
        HCCL_ERROR("[DpuTaskExceptionLite::%s] memcpy_s failed on errorCode, return[%d].", __func__, ret);
        return HCCL_E_MEMORY;
    }
    if (errorCode != 0) {
        // 触发taskexception
        HCCL_ERROR(
            "[DpuTaskExceptionLite][DPU] taskexceptionVa[%p], errorCode[0x%x], devId[%u], commId[%s]", taskexceptionVa,
            errorCode, aicpuComm->GetDevId(), commId.c_str());
        // 1、取notify，并构造rtLogicCqReport_t
        const auto curDfxOpInfo = static_cast<const Hccl::DfxDfxOpInfo*>(hcclCommDfxLite->GetLatestDfxOpInfo());
        CHK_PTR_NULL(curDfxOpInfo);
        u32 notifyId = curDfxOpInfo->cpuWaitAicpuNotifyId;
        rtLogicCqReport_t exceptionInfo{};
        // 2、调用SendTaskExceptionByMBox触发taskexception回调
        CHK_RET(HcclCommTaskExceptionLite::GetInstance().SendTaskExceptionByMBox(notifyId, 0, exceptionInfo));
        // 3、标志位置0
        ret = memset_s(taskexceptionVa + sizeof(uint8_t), sizeof(uint16_t), 0, sizeof(uint16_t)); // 标志位置0
        if (ret != EOK) {
            HCCL_ERROR("[DpuTaskExceptionLite::%s] memset_s failed on flag, return[%d].", __func__, ret);
            return HCCL_E_MEMORY;
        }
    }
    return HCCL_SUCCESS;
}

} // namespace hcomm
