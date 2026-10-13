/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_res_ccu.h"

#include "acl/acl_rt.h"

#include "ccu_dev_mgr_imp.h"
#include "ccu_instance.h"
#include "channel_process.h"
#include "exception_handler.h"
#include "hcomm_adapter_hccp.h"
#include "log.h"

namespace {
constexpr uint32_t CCU_INSTR_WIDTH_BYTES = 32U; // 单条 CCU 微码指令宽度
} // namespace

// 查询 CCU Channel 实体快照，remoteBuffers 借用 hcomm 线程本地数组。
HcommResult HcommCcuGetChannelEntity(ChannelHandle channel, HcommCcuChannelEntity* channelEntity)
{
    EXCEPTION_HANDLE_BEGIN
    return hcomm::ChannelProcess::CcuChannelGetEntity(channel, channelEntity);
    EXCEPTION_HANDLE_END
}

// 下发 CCU 微码指令到设备，data 仅本次调用内借用。
HcommResult HcommCcuSubmitInsts(
    int32_t deviceLogicId, uint32_t dieId, uint32_t startInstructionId, const void* data, uint32_t instNum)
{
    if (data == nullptr || instNum == 0) {
        return static_cast<int32_t>(HcclResult::HCCL_E_PARA);
    }
    if (dieId >= hcomm::HCOMM_CCU_MAX_DIE_NUM) {
        HCCL_ERROR(
            "[HcommCcuSubmitInsts] dieId[%u] should be less than %u.", dieId,
            static_cast<uint32_t>(hcomm::HCOMM_CCU_MAX_DIE_NUM));
        return static_cast<int32_t>(HcclResult::HCCL_E_PARA);
    }
    const uint64_t byteSize = static_cast<uint64_t>(instNum) * CCU_INSTR_WIDTH_BYTES;
    if (byteSize > static_cast<uint64_t>(0xFFFFFFFFULL)) {
        HCCL_ERROR(
            "[HcommCcuSubmitInsts] byteSize[%llu] exceeds uint32 limit.", static_cast<unsigned long long>(byteSize));
        return static_cast<int32_t>(HcclResult::HCCL_E_PARA);
    }

    // C ABI 边界不允许 C++ 异常穿透
    try {
        // host -> device 暂存（按需扩容的调用级缓冲；下发完成即归还）
        void* devMem = nullptr;
        HcclResult ret = static_cast<HcclResult>(aclrtMalloc(&devMem, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (ret != HcclResult::HCCL_SUCCESS || devMem == nullptr) {
            HCCL_ERROR(
                "[HcommCcuSubmitInsts] alloc staging device memory failed, byteSize[%llu] ret[%d].",
                static_cast<unsigned long long>(byteSize), ret);
            return static_cast<int32_t>(ret);
        }
        ret = static_cast<HcclResult>(aclrtMemcpy(devMem, byteSize, data, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
        if (ret != HcclResult::HCCL_SUCCESS) {
            (void)aclrtFree(devMem);
            HCCL_ERROR("[HcommCcuSubmitInsts] copy instructions to device failed, ret[%d].", ret);
            return static_cast<int32_t>(ret);
        }

        hcomm::CustomChannelInfoIn inBuff{};
        hcomm::CustomChannelInfoOut outBuff{};
        inBuff.op = hcomm::CcuOpcodeType::CCU_U_OP_SET_INSTRUCTION;
        inBuff.offsetStartIdx = startInstructionId;
        inBuff.data.dataInfo.udieIdx = dieId;
        inBuff.data.dataInfo.dataArraySize = 1;
        inBuff.data.dataInfo.dataLen = static_cast<uint32_t>(byteSize);
        inBuff.data.dataInfo.dataArray[0].insinfo.resourceAddr = reinterpret_cast<uint64_t>(devMem);

        ret = hcomm::HccpRaTlvCcuCustomChannel(
            deviceLogicId, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));
        (void)aclrtFree(devMem);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HcommCcuSubmitInsts] failed to call ccu driver, devLogicId[%d] dieId[%u] "
                "startInstrId[%u] instNum[%u] ret[%d].",
                deviceLogicId, dieId, startInstructionId, instNum, ret);
            return static_cast<int32_t>(ret);
        }
        return static_cast<int32_t>(HcclResult::HCCL_SUCCESS);
    } catch (...) {
        HCCL_ERROR("[HcommCcuSubmitInsts] caught exception, devLogicId[%d].", deviceLogicId);
        return static_cast<int32_t>(HcclResult::HCCL_E_INTERNAL);
    }
}

// 按设备和 die 申请连续指令空间；调用方须保存申请结果，并通过 HcommCcuReleaseInstSpace 成对归还。
HcommResult HcommCcuAllocInstSpace(int32_t deviceLogicId, uint32_t dieId, uint32_t instNum, uint32_t* startInstId)
{
    if (startInstId == nullptr) {
        return static_cast<int32_t>(HcclResult::HCCL_E_PTR);
    }
    *startInstId = 0;
    if (instNum == 0 || dieId >= hcomm::HCOMM_CCU_MAX_DIE_NUM) {
        return static_cast<int32_t>(HcclResult::HCCL_E_PARA);
    }
    hcomm::ResInfo range{};
    const HcclResult ret = hcomm::CcuDevMgrImp::AllocIns(deviceLogicId, dieId, instNum, range);
    if (ret == HcclResult::HCCL_SUCCESS) {
        *startInstId = range.startId;
    }
    return static_cast<int32_t>(ret);
}

// 按设备和 die 归还连续指令空间；空间由 asc-comm 数据面按申请结果成对归还。
HcommResult HcommCcuReleaseInstSpace(int32_t deviceLogicId, uint32_t dieId, uint32_t instNum, uint32_t startInstId)
{
    if (instNum == 0 || dieId >= hcomm::HCOMM_CCU_MAX_DIE_NUM) {
        return static_cast<int32_t>(HcclResult::HCCL_E_PARA);
    }
    const HcclResult ret = hcomm::CcuDevMgrImp::ReleaseIns(deviceLogicId, dieId, {startInstId, instNum});
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcommCcuReleaseInstSpace] failed to release instruction space, devLogicId[%d] dieId[%u] "
            "startInstId[%u] instNum[%u] ret[%d].",
            deviceLogicId, dieId, startInstId, instNum, ret);
    }
    return static_cast<int32_t>(ret);
}
