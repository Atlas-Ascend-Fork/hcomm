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
#include "common/cast_utils_pub.h"
#include "dfx/device/dfx_res_pub.h"
#include "hcclCommTaskExceptionLite.h"
#include "stream_lite.h"
#include "common/device/aicpu_indop_env_pub.h"
#include "dpu_sync_layout.h"
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
    auto taskexceptionVa = ReinterpretAs<uint8_t*>(hcclCommDfxLite->GetTaskExpDevMem());
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

HcclResult DpuTaskExceptionLite::HandleDpuCq(CollCommAicpu* aicpuComm, const hccl::QueueDescriptor& cqDesc)
{
    CHK_PTR_NULL(aicpuComm);
    if (cqDesc.depth == 0) {
        return HCCL_SUCCESS; // 名片不可用，空转
    }
    if (aicpuComm->IsErrorReported()) {
        return HCCL_SUCCESS; // 防重：与 task exception 上报共用同一份「已上报」状态
    }

    // CQ 有一格回执：消费一格并上报。游标走 acquire：生产侧是「先写 CQE 再 release 发布 pi」，
    // 这里先 acquire 读 pi 才能保证随后读到的 CQE 内容是完整的
    const uint32_t pi = __atomic_load_n(reinterpret_cast<const uint32_t*>(cqDesc.piAddr), __ATOMIC_ACQUIRE);
    const uint32_t ci = __atomic_load_n(reinterpret_cast<const uint32_t*>(cqDesc.ciAddr), __ATOMIC_ACQUIRE);
    errno_t ret = EOK;
    if (pi == ci) {
        return HCCL_SUCCESS;
    }

    hccl::DpuCqe cqe{};
    const uint64_t slotAddr = cqDesc.base + static_cast<uint64_t>(ci % cqDesc.depth) * cqDesc.elemSize;
    ret = memcpy_s(&cqe, sizeof(cqe), reinterpret_cast<const void*>(slotAddr), sizeof(cqe));
    if (ret != EOK) {
        HCCL_ERROR(
            "[DpuTaskExceptionLite::%s] memcpy_s failed on cqe, comm[%s], ci[%u], return[%d].", __func__,
            aicpuComm->GetIdentifier().c_str(), ci, ret);
        return HCCL_E_MEMORY;
    }
    HCCL_ERROR(
        "[DpuTaskExceptionLite::%s] cqe error, comm[%s], hcclErrCode[%d], taskId[%u], streamId[%u]", __func__,
        aicpuComm->GetIdentifier().c_str(), cqe.hcclErrCode, cqe.taskId, cqe.streamId);
    // 先上报再推进：单槽队列推进即丢，上报失败会静默丢错
    const HcclResult reportRet = ReportDpuCqError(aicpuComm);
    if (reportRet == HCCL_E_AGAIN) {
        return HCCL_SUCCESS; // 尚无算子上下文，保留 CQE 等下次重报
    }
    if (reportRet != HCCL_SUCCESS) {
        return reportRet;
    }
    // release 推进：本侧已读完该格，生产侧见到新 ci 后才可复用
    const uint32_t nextCi = ci + 1U;
    __atomic_store_n(reinterpret_cast<uint32_t*>(cqDesc.ciAddr), nextCi, __ATOMIC_RELEASE);
    return HCCL_SUCCESS;
}

HcclResult DpuTaskExceptionLite::ReportDpuCqError(CollCommAicpu* aicpuComm)
{
    // 取 host 等 device 的 notifyId，投 MBox 触发 host 侧回调
    const auto curDfxOpInfo
        = static_cast<const Hccl::DfxDfxOpInfo*>(aicpuComm->GetHcclCommDfxLite()->GetLatestDfxOpInfo());
    if (curDfxOpInfo == nullptr) {
        HCCL_WARNING(
            "[DpuTaskExceptionLite::%s] no op info yet, comm[%s]", __func__, aicpuComm->GetIdentifier().c_str());
        return HCCL_E_AGAIN; // 还没有算子就没有上报通道，不置「已上报」，等下次下发后重报
    }
    u32 notifyId = curDfxOpInfo->cpuWaitAicpuNotifyId;
    rtLogicCqReport_t exceptionInfo{};
    CHK_RET(HcclCommTaskExceptionLite::GetInstance().SendTaskExceptionByMBox(notifyId, 0, exceptionInfo));
    aicpuComm->SetErrorReported(true);
    return HCCL_SUCCESS;
}

} // namespace hcomm
