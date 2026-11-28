/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ep_mem_table_launch.h"

#include <cstddef>

#include "base_comm_node_accessors.h" // GetAicpuCommEngineBinProc
#include "launch_aicpu.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "adapter_rts_common.h"
#include "common/cast_utils_pub.h"
#include "log.h"

namespace hcomm {
namespace {
    constexpr uint32_t AICPU_LAUNCH_TIMEOUT_SEC = 120;

    HcclResult EnsureAicpuBinHandle(aclrtBinHandle& binHandle)
    {
        auto& binProc = GetAicpuCommEngineBinProc();
        CHK_RET(binProc.EnsureKernelBinLoaded(COMM_ENGINE_AICPU_TS));
        binHandle = binProc.GetBinHandle();
        CHK_PTR_NULL(binHandle);
        return HCCL_SUCCESS;
    }

    HcclResult CreateLocalAicpuStream(hccl::Stream& stream)
    {
        stream = hccl::Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
        constexpr uint32_t aicpuStreamMode = 1;
        return hrtStreamSetMode(stream.ptr(), aicpuStreamMode);
    }

    HcclResult LaunchEpMemKernel(const char* kernelName, const void* param, size_t paramSize)
    {
        aclrtBinHandle binHandle{nullptr};
        CHK_RET(EnsureAicpuBinHandle(binHandle));
        hccl::Stream stream;
        CHK_RET(CreateLocalAicpuStream(stream));

        hccl::DeviceMem paramDev = hccl::DeviceMem::alloc(static_cast<u64>(paramSize));
        CHK_PTR_NULL(paramDev.ptr());
        CHK_RET(hrtMemSyncCopy(
            paramDev.ptr(), paramSize, param, paramSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

        uint64_t context = ReinterpretAs<uint64_t>(paramDev.ptr());
        CHK_RET(hccl::AicpuAclKernelLaunch(
            stream.ptr(), ReinterpretAs<void*>(&context), sizeof(context), binHandle, kernelName, true,
            static_cast<u16>(AICPU_LAUNCH_TIMEOUT_SEC)));
        // 需通过 hcclStreamSynchronize 来阻塞host线程直到下kernel更新device内存表完成，保证时序安全
        CHK_RET(hcclStreamSynchronize(stream.ptr(), static_cast<s32>(AICPU_LAUNCH_TIMEOUT_SEC)));
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult LaunchEpMemTableInit(uint64_t& outTablePtr)
{
    outTablePtr = 0;
    EpMemTableInitOutput output{};
    hccl::DeviceMem outDev = hccl::DeviceMem::alloc(sizeof(output));
    CHK_PTR_NULL(outDev.ptr());
    CHK_RET(hrtMemSyncCopy(
        outDev.ptr(), sizeof(output), &output, sizeof(output), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    HcclAicpuEpMemTableInitRes param{};
    param.outAddr = ReinterpretAs<uint64_t>(outDev.ptr());
    HcclResult launchRet = LaunchEpMemKernel("RunAicpuEpMemTableInit", &param, sizeof(param));
    HcclResult d2hRet = hrtMemSyncCopy(
        &output, sizeof(output), outDev.ptr(), sizeof(output), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST);
    if (launchRet != HCCL_SUCCESS) {
        if (d2hRet == HCCL_SUCCESS && output.tablePtr != 0 && output.magic == DEV_EP_MEM_TABLE_MAGIC_WORD
            && output.version == DEV_EP_MEM_TABLE_VERSION) {
            HcclResult destroyRet = LaunchEpMemTableDestroy(output.tablePtr);
            HCCL_WARNING(
                "[%s] launch failed ret[%d] but table[0x%llx] built, reclaim ret[%d], fail means leak until "
                "process exit",
                __func__, static_cast<int>(launchRet), output.tablePtr, static_cast<int>(destroyRet));
        } else {
            HCCL_ERROR(
                "[%s] launch failed ret[%d], d2hRet[%d]; table not built or addr unreadable, leaked until "
                "process exit",
                __func__, static_cast<int>(launchRet), static_cast<int>(d2hRet));
        }
        return launchRet;
    }
    CHK_PRT_RET(
        d2hRet != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] D2H readback failed, ret[%d]; table addr unreadable, leaked until process exit", __func__, d2hRet),
        d2hRet);
    if (output.tablePtr == 0 || output.magic != DEV_EP_MEM_TABLE_MAGIC_WORD
        || output.version != DEV_EP_MEM_TABLE_VERSION) {
        HCCL_ERROR(
            "[%s] bad output, tablePtr[0x%llx] magic[0x%x] version[%u]", __func__, output.tablePtr, output.magic,
            output.version);
        if (output.tablePtr != 0) {
            HcclResult destroyRet = LaunchEpMemTableDestroy(output.tablePtr);
            HCCL_WARNING(
                "[%s] reclaim table[0x%llx] ret[%d], fail means leak until process exit", __func__, output.tablePtr,
                destroyRet);
        }
        return HCCL_E_INTERNAL;
    }
    outTablePtr = output.tablePtr;
    HCCL_INFO("[%s] success, tablePtr[0x%llx]", __func__, outTablePtr);
    return HCCL_SUCCESS;
}

HcclResult LaunchEpMemTableUpdate(
    uint64_t tablePtr, EpMemTableOp op, uint64_t addr, uint64_t size, uint32_t tokenId, uint32_t tokenValue)
{
    HcclAicpuEpMemTableUpdateRes param{};
    param.tablePtr = tablePtr;
    param.op = static_cast<uint32_t>(op);
    param.addr = addr;
    param.size = size;
    param.tokenId = tokenId;
    param.tokenValue = tokenValue;
    return LaunchEpMemKernel("RunAicpuEpMemTableUpdate", &param, sizeof(param));
}

HcclResult LaunchEpMemTableDestroy(uint64_t tablePtr)
{
    HcclAicpuEpMemTableDestroyRes param{};
    param.tablePtr = tablePtr;
    return LaunchEpMemKernel("RunAicpuEpMemTableDestroy", &param, sizeof(param));
}

HcclResult SyncUbLocMemToEpTable(
    uint64_t tablePtr, EpMemTableOp op, uint64_t addr, uint64_t size, uint32_t tokenId, uint32_t tokenValue)
{
    CHK_PRT_RET(tablePtr == 0, HCCL_ERROR("[%s] ep mem table not inited", __func__), HCCL_E_INTERNAL);
    return LaunchEpMemTableUpdate(tablePtr, op, addr, size, tokenId, tokenValue);
}
} // namespace hcomm
