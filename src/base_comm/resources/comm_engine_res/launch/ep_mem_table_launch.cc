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
    constexpr uint32_t AICPU_LAUNCH_TIMEOUT_SEC = 120; // 对齐 ThreadKernelLaunchForBase 默认值

    HcclResult EnsureAicpuBinHandle(aclrtBinHandle& binHandle)
    {
        // 进程级 bin 单例，Ensure 幂等（先例：aicpu_ts_channel_entity.cc EnsureDeviceReady）
        auto& binProc = GetAicpuCommEngineBinProc();
        CHK_RET(binProc.EnsureKernelBinLoaded(COMM_ENGINE_AICPU_TS));
        binHandle = binProc.GetBinHandle();
        CHK_PTR_NULL(binHandle);
        return HCCL_SUCCESS;
    }

    HcclResult CreateLocalAicpuStream(hccl::Stream& stream)
    {
        // 自建 online stream + AICPU 模式（先例：aicpu_launch_manager.cc CreateLocalStream）
        stream = hccl::Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
        constexpr uint32_t aicpuStreamMode = 1;
        return hrtStreamSetMode(stream.ptr(), aicpuStreamMode);
    }

    // param → device 内存 → launch → sync（骨架同 LaunchRunAicpuChannelMemUpdate，my_rank.cc）
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
        // ★ 时序锚点：sync 完成后表更新全局可见——后续 channel 建链 kernel（不同 stream 靠此定序）
        //   与已建通道数据面原语均可立即查到
        CHK_RET(hcclStreamSynchronize(stream.ptr(), static_cast<s32>(AICPU_LAUNCH_TIMEOUT_SEC)));
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult LaunchEpMemTableInit(uint64_t& outTablePtr)
{
    outTablePtr = 0;

    // 输出槽：host 侧零值结构 H2D 下发，等效清零（防脏 magic 误匹配；先例 PreAllocChannels 清零）
    EpMemTableInitOutput output{};
    hccl::DeviceMem outDev = hccl::DeviceMem::alloc(sizeof(output));
    CHK_PTR_NULL(outDev.ptr());
    CHK_RET(hrtMemSyncCopy(
        outDev.ptr(), sizeof(output), &output, sizeof(output), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    HcclAicpuEpMemTableInitRes param{};
    param.outAddr = ReinterpretAs<uint64_t>(outDev.ptr());
    HcclResult launchRet = LaunchEpMemKernel("RunAicpuEpMemTableInit", &param, sizeof(param));

    // D2H 读回 + ABI 校验（仿 ctx 读回校验：aicpu_ts_channel_entity.cc LaunchKernel）。
    // 读回无条件执行：launch/sync 失败时 kernel 可能已执行完（如 sync 超时慢启动），表已建而
    // host 未知地址——不做读回则每次失败泄漏一份 device 表；kernel 未跑时槽保持 H2D 零值，读回无害
    HcclResult d2hRet = hrtMemSyncCopy(
        &output, sizeof(output), outDev.ptr(), sizeof(output), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST);
    // 启动/同步失败：读回成功且三元组校验合法（tablePtr!=0 且 magic/version 合法——kernel 写序
    // tablePtr→magic→version，magic/version 可见 ⇒ 构造已完整 ⇒ destroy 安全；仅凭非零 tablePtr
    // 发起 destroy 会在 CheckAbi 前解引用垃圾半截值，故门比校验失败分支更保守）则尽力回收
    // 已建表后返回原始错误码（保留编排层回滚语义）；kernel 未跑或写半截 → 泄漏不可避免，记日志
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
        // 表已建但校验不过：地址已读回则尽力回收（destroy kernel 侧 CheckAbi 哨兵防误删坏地址）；
        // 回收失败或 tablePtr==0 时只能依赖进程退出回收，显式记日志便于定位
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
    // 纯值入参：对象存活期由调用方保证（Reg 路径刚注册存活 / Unreg 路径 host 已前置提取）；
    // REMOVE 仅消费 addr/size，tokenId/tokenValue 由 Unreg 路径填 0（kernel 侧忽略）
    return LaunchEpMemTableUpdate(tablePtr, op, addr, size, tokenId, tokenValue);
}
} // namespace hcomm
