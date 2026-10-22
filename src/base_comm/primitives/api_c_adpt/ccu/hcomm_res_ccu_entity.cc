/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_res_ccu_entity.h"

#include "acl/acl_rt.h"

#include "ccu_dev_mgr_imp.h"
#include "ccu_instance.h"
#include "channel_process.h"
#include "exception_handler.h"
#include "hcomm_adapter_hccp.h"
#include "log.h"

namespace {
constexpr uint32_t CCU_INSTR_WIDTH_BYTES = 32U; // 单条 CCU 微码指令宽度

bool CheckSubmitInstsParams(
    int32_t deviceLogicId, uint32_t dieId, const void* data, uint32_t instNum, uint64_t& byteSize)
{
    if (data == nullptr || instNum == 0) {
        HCCL_ERROR(
            "[HcommCcuSubmitInsts] invalid submit params, devLogicId[%d] dieId[%u] instNum[%u] dataNull[%d].",
            deviceLogicId, dieId, instNum, data == nullptr ? 1 : 0);
        return false;
    }
    if (dieId >= hcomm::HCOMM_CCU_MAX_DIE_NUM) {
        HCCL_ERROR(
            "[HcommCcuSubmitInsts] dieId[%u] should be less than %u.", dieId,
            static_cast<uint32_t>(hcomm::HCOMM_CCU_MAX_DIE_NUM));
        return false;
    }
    byteSize = static_cast<uint64_t>(instNum) * CCU_INSTR_WIDTH_BYTES;
    if (byteSize > static_cast<uint64_t>(0xFFFFFFFFULL)) {
        HCCL_ERROR(
            "[HcommCcuSubmitInsts] byteSize[%llu] exceeds uint32 limit.", static_cast<unsigned long long>(byteSize));
        return false;
    }
    return true;
}

HcclResult CopyInstsToDevice(uint64_t byteSize, const void* data, void*& devMem)
{
    HcclResult ret = static_cast<HcclResult>(aclrtMalloc(&devMem, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if (ret != HcclResult::HCCL_SUCCESS || devMem == nullptr) {
        HCCL_ERROR(
            "[HcommCcuSubmitInsts] alloc staging device memory failed, byteSize[%llu] ret[%d].",
            static_cast<unsigned long long>(byteSize), ret);
        return ret == HcclResult::HCCL_SUCCESS ? HcclResult::HCCL_E_INTERNAL : ret;
    }
    ret = static_cast<HcclResult>(aclrtMemcpy(devMem, byteSize, data, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    if (ret != HcclResult::HCCL_SUCCESS) {
        (void)aclrtFree(devMem);
        devMem = nullptr;
        HCCL_ERROR("[HcommCcuSubmitInsts] copy instructions to device failed, ret[%d].", ret);
    }
    return ret;
}

HcclResult SubmitInstsToDevice(
    int32_t deviceLogicId, uint32_t dieId, uint32_t startInstructionId, uint32_t instNum, uint64_t byteSize,
    void* devMem)
{
    hcomm::CustomChannelInfoIn inBuff{};
    hcomm::CustomChannelInfoOut outBuff{};
    inBuff.op = hcomm::CcuOpcodeType::CCU_U_OP_SET_INSTRUCTION;
    inBuff.offsetStartIdx = startInstructionId;
    inBuff.data.dataInfo.udieIdx = dieId;
    inBuff.data.dataInfo.dataArraySize = 1;
    inBuff.data.dataInfo.dataLen = static_cast<uint32_t>(byteSize);
    inBuff.data.dataInfo.dataArray[0].insinfo.resourceAddr = reinterpret_cast<uint64_t>(devMem);

    HcclResult ret
        = hcomm::HccpRaTlvCcuCustomChannel(deviceLogicId, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcommCcuSubmitInsts] failed to call ccu driver, devLogicId[%d] dieId[%u] "
            "startInstrId[%u] instNum[%u] ret[%d].",
            deviceLogicId, dieId, startInstructionId, instNum, ret);
    }
    return ret;
}
} // namespace

// 查询 CCU Channel 实体快照，remoteBuffers 借用 hcomm 线程本地数组。
HcommResult HcommCcuChannelGetEntity(ChannelHandle channel, HcommCcuChannelEntity* channelEntity)
{
    EXCEPTION_HANDLE_BEGIN
    return hcomm::ChannelProcess::CcuChannelGetEntity(channel, channelEntity);
    EXCEPTION_HANDLE_END
}

// 下发 CCU 微码指令到设备，data 仅本次调用内借用。
HcommResult HcommCcuSubmitInsts(
    int32_t deviceLogicId, uint32_t dieId, uint32_t startInstructionId, const void* data, uint32_t instNum)
{
    uint64_t byteSize = 0;
    if (!CheckSubmitInstsParams(deviceLogicId, dieId, data, instNum, byteSize)) {
        return static_cast<int32_t>(HcclResult::HCCL_E_PARA);
    }

    // C ABI 边界不允许 C++ 异常穿透
    void* devMem = nullptr;
    try {
        // host -> device 暂存（调用级临时缓冲；下发完成即归还）
        HcclResult ret = CopyInstsToDevice(byteSize, data, devMem);
        if (ret != HcclResult::HCCL_SUCCESS) {
            return static_cast<int32_t>(ret);
        }

        ret = SubmitInstsToDevice(deviceLogicId, dieId, startInstructionId, instNum, byteSize, devMem);
        (void)aclrtFree(devMem);
        return static_cast<int32_t>(ret);
    } catch (...) {
        (void)aclrtFree(devMem);
        HCCL_ERROR("[HcommCcuSubmitInsts] caught exception, devLogicId[%d].", deviceLogicId);
        return static_cast<int32_t>(HcclResult::HCCL_E_INTERNAL);
    }
}

// 按设备和 die 申请连续指令空间；指令槽位由 per-device CCU 分配器管理，
// 最后一个 CCU feature 引用释放触发 CcuDeinitFeature 时随组件整体回收。
HcommResult HcommCcuAllocInstSpace(int32_t deviceLogicId, uint32_t dieId, uint32_t instNum, uint32_t* startInstId)
{
    if (startInstId == nullptr) {
        HCCL_ERROR(
            "[%s] startInstId is nullptr, devLogicId[%d] dieId[%u] instNum[%u] ret[%d].", __func__, deviceLogicId,
            dieId, instNum, static_cast<int32_t>(HcclResult::HCCL_E_PTR));
        return static_cast<int32_t>(HcclResult::HCCL_E_PTR);
    }
    *startInstId = 0;
    if (instNum == 0 || dieId >= hcomm::HCOMM_CCU_MAX_DIE_NUM) {
        HCCL_ERROR(
            "[%s] invalid instruction space request, devLogicId[%d] dieId[%u] instNum[%u].", __func__, deviceLogicId,
            dieId, instNum);
        return static_cast<int32_t>(HcclResult::HCCL_E_PARA);
    }
    hcomm::ResInfo range{};
    const HcclResult ret = hcomm::CcuDevMgrImp::AllocIns(deviceLogicId, dieId, instNum, range);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_WARNING(
            "[%s] allocate instruction space failed, devLogicId[%d] dieId[%u] instNum[%u] ret[%d].", __func__,
            deviceLogicId, dieId, instNum, ret);
        return static_cast<int32_t>(ret);
    }
    *startInstId = range.startId;
    return static_cast<int32_t>(HcclResult::HCCL_SUCCESS);
}
