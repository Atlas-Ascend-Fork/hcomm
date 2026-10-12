/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_aicpu_pipe_handler.h"
#include "sim_aicpu_pipe_msg.h"
#include <unistd.h>
#include <string>
#include <dlfcn.h>
#include "sim_log.h"
#include "sim_common_api.h"
#include "operation_data/operation_data_ops.h"
#include "hccl_device_pub.h"
#include "store_sim_memory_manager.h"
#include "sim_kernel_lib_mgr.h"
#include "aicpu_args_stub.h"

namespace sim {

int HandlePipeCmdSetDevId(uint8_t* payload, uint32_t payloadLen)
{
    SetDevIdPayload* req = reinterpret_cast<SetDevIdPayload*>(payload);
    uint32_t curRankId = static_cast<uint32_t>(req->rankId);
    uint64_t curDeviceKey = req->deviceKey;
    SetCurRankId(curRankId);
    SetCurDeviceKey(curDeviceKey);
    // 设备侧实际收到的身份参数（host aclrtSetDevice 下发）：commRank=通信 rank、
    // deviceKey=Device 表主键（与逻辑设备号不同口径，不按名称混用）
    HCCL_VM_INFO(
        "Process[{}] PIPE_RSP_SET_DEV_ID, set commRank = [{}], deviceKey(DeviceIdPrimaryKey) = [{}]", getpid(),
        curRankId, curDeviceKey);
    uint64_t donePayload = 0;
    DeviceSendMsg(PIPE_RSP_SET_DEV_ID, &donePayload, sizeof(donePayload));
    return 0;
}

int HandlePipeCmdGetDevPtr(uint8_t* payload, uint32_t payloadLen)
{
    DevMemOpPayload* req = reinterpret_cast<DevMemOpPayload*>(payload);
    RspGetDevPtrPayload donePayload{};
    void* shmptr = sim::MemoryManager::GetInstance().AcquireMemByName(req->memName);
    if (shmptr == nullptr) {
        donePayload.ptr = 0;
        HCCL_VM_ERROR("acquire {} shm failed.", req->memName);
    } else {
        donePayload.ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(shmptr));
    }
    HCCL_VM_INFO("[device] acquire {} shm ptr:{:p}", req->memName, shmptr);
    DeviceSendMsg(PIPE_RSP_GET_DEV_PTR, &donePayload, sizeof(donePayload));
    return 0;
}

static int ExecuteAicpuKernel(uint32_t rankId, ExecKernelPayload* kernelReq, void* argsPtr)
{
    std::string kernelSo = kernelReq->soName;
    std::string kernelName = kernelReq->kernelName;

    HCCL_VM_INFO("rankId[{}] kernel[{}] start run...", rankId, kernelName);
    std::string libDir = InstallPath::ResolveToInstallRoot("lib/" + GetArchStr()) + "/" + kernelSo;
    KernelFn fn = sim::KernelLibManager::GetInstance().GetOrLoadFunc(libDir, kernelName);
    if (!fn) {
        HCCL_VM_ERROR("rankId[{}] kernel[{}] failed to resolve function in {}", rankId, kernelName, libDir);
        return -1;
    }

    if (argsPtr == nullptr) {
        HCCL_VM_ERROR("[device] rankId[{}] init func handle failed null ptr.", rankId);
        return -1;
    }

    // CCU退化AICPU场景此处使用的内存先于device进程启动前分配，使用前需转换。
    // 冻结 HCOMM 48b54e99 消费合同（已逐字段核对 aicpu_init_param.h /
    // hdc_param.h / coll_comm_aicpu.cc / hdc.cc / hdc_lite.cc，模拟器
    // aicpu_args_stub.h 镜像与真身布局一致）：
    //   - 块级可选：buffLen==0 的 HDCommunicateParams 块 InitHDCommunicate
    //     不消费（仅在 buffLen!=0 时 InitDevice）；
    //   - 块内必需：被消费块的 deviceAddr 与 readCacheAddr 均经 CHK_PTR_NULL，
    //     为 0 即 HCCL_E_PARA。
    // 因此：先整体换算暂存，必需地址（所在块 buffLen!=0）映射失败立即沿既有
    // 非零 status 链返回，不把 0 地址交给 kernel；全部通过才写回（避免半成功
    // 覆写后丢失原始地址）；可选块的映射结果照旧写回（保持原语义）。
    if (kernelName == "RunAicpuIndOpCommInit" || kernelName == "RunAicpuCommInit") {
        CommAicpuParam* param = reinterpret_cast<CommAicpuParam*>(argsPtr);
        const uint64_t ctrlDevOrig = param->kfcControlTransferH2DParams.deviceAddr;
        const uint64_t ctrlCacheOrig = param->kfcControlTransferH2DParams.readCacheAddr;
        const uint64_t statusDevOrig = param->kfcStatusTransferD2HParams.deviceAddr;
        const uint64_t ctrlDevMapped = GetDevMapperAddrByDevAddr(ctrlDevOrig);
        const uint64_t ctrlCacheMapped = GetDevMapperAddrByDevAddr(ctrlCacheOrig);
        const uint64_t statusDevMapped = GetDevMapperAddrByDevAddr(statusDevOrig);
        const bool ctrlRequired = param->kfcControlTransferH2DParams.buffLen != 0;
        const bool statusRequired = param->kfcStatusTransferD2HParams.buffLen != 0;
        if ((ctrlRequired && (ctrlDevMapped == 0 || ctrlCacheMapped == 0))
            || (statusRequired && statusDevMapped == 0)) {
            HCCL_VM_ERROR(
                "rankId[{}] kernel[{}] comm init required addr translate FAILED: "
                "ctrlRequired[{}] ctrlDev[0x{:x}->0x{:x}] ctrlCache[0x{:x}->0x{:x}], "
                "statusRequired[{}] statusDev[0x{:x}->0x{:x}]",
                rankId, kernelName, ctrlRequired, ctrlDevOrig, ctrlDevMapped, ctrlCacheOrig, ctrlCacheMapped,
                statusRequired, statusDevOrig, statusDevMapped);
            return 1;
        }
        param->kfcControlTransferH2DParams.deviceAddr = ctrlDevMapped;
        param->kfcControlTransferH2DParams.readCacheAddr = ctrlCacheMapped;
        param->kfcStatusTransferD2HParams.deviceAddr = statusDevMapped;
    }

    uint32_t kernelRet = fn(argsPtr);
    if (kernelRet != 0) {
        HCCL_VM_ERROR("rankId[{}] kernel[{}] returned error: {}", rankId, kernelName, kernelRet);
        return static_cast<int>(kernelRet);
    }
    HCCL_VM_INFO("rankId[{}] kernel[{}] finish run...", rankId, kernelName);
    return 0;
}

int HandlePipeCmdExecKernel(uint8_t* payload, uint32_t payloadLen)
{
    if (payloadLen < sizeof(ExecKernelPayload)) {
        HCCL_VM_ERROR("[device] EXEC_KERNEL payload too small: {} < {}", payloadLen, sizeof(ExecKernelPayload));
        RspExecKernelPayload donePayload{};
        donePayload.status = -1;
        DeviceSendMsg(PIPE_RSP_EXEC_KERNEL, &donePayload, sizeof(donePayload));
        return -1;
    }

    ExecKernelPayload* kernelReq = reinterpret_cast<ExecKernelPayload*>(payload);
    uint32_t rankId = 0;
    GetCurRankId(&rankId);
    std::string kernelName(kernelReq->kernelName);
    uint32_t argsLen = payloadLen - sizeof(ExecKernelPayload);
    void* argsPtr = payload + sizeof(ExecKernelPayload);
    HCCL_VM_INFO("[device] rankId[{}] executing kernel:{}, argsLen:{}", rankId, kernelName, argsLen);

    uint64_t commId = 0;
    uint32_t opRankId = 0;
    uint32_t opDeviceId = 0;
    if (kernelReq->opDetailId != 0) {
        // 只使用 host 随本次 kernel 显式传入的算子记录，不按时间或 pid 猜测。
        if (sim::operation::QueryOpDetailIdentity(kernelReq->opDetailId, commId, opRankId, opDeviceId) != 0
            || commId == 0) {
            HCCL_VM_ERROR("[device] invalid task identity for opDetailId={}, commId={}", kernelReq->opDetailId, commId);
            RspExecKernelPayload donePayload{-1};
            DeviceSendMsg(PIPE_RSP_EXEC_KERNEL, &donePayload, sizeof(donePayload));
            return -1;
        }
        // InsertOpTask 仍通过该值把 device 任务写入本次 opDetail 的任务表；
        // 它不参与任务 metadata 的身份生成。
        sim::operation::g_currOpDetailId = kernelReq->opDetailId;
    } else {
        // 普通 kernel 不产生带通信域身份的 SQE，执行前必须清掉上一次算子的上下文。
        sim::operation::g_currOpDetailId = 0;
    }

    int execRet = ExecuteAicpuKernel(rankId, kernelReq, argsPtr);
    sim::operation::g_currOpDetailId = 0;
    RspExecKernelPayload donePayload{};
    donePayload.status = static_cast<int32_t>(execRet);
    DeviceSendMsg(PIPE_RSP_EXEC_KERNEL, &donePayload, sizeof(donePayload));
    return execRet;
}

int HandlePipeCmdFreeDevPtr(uint8_t* payload, uint32_t payloadLen)
{
    DevMemOpPayload* req = reinterpret_cast<DevMemOpPayload*>(payload);
    sim::MemoryManager::GetInstance().ReleaseMemByName(req->memName);
    HCCL_VM_INFO("[device] release memName:{} shm", req->memName);
    RspFreeDevPtrPayload donePayload{};
    donePayload.status = 0;
    DeviceSendMsg(PIPE_RSP_FREE_DEV_PTR, &donePayload, sizeof(donePayload));
    return 0;
}

int HandlePipeCmdGetWqePtr(uint8_t* payload, uint32_t payloadLen)
{
    RspGetDevPtrPayload donePayload{};
    void* ptr = malloc(2 * 1024 * 1024);
    if (ptr == nullptr) {
        donePayload.ptr = 0;
    } else {
        donePayload.ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
    }
    HCCL_VM_INFO("[device] alloc wqe buffer :{:p}", ptr);

    DeviceSendMsg(PIPE_RSP_GET_WQE_PTR, &donePayload, sizeof(donePayload));
    return 0;
}

int HandlePipeCmdFreeWqePtr(uint8_t* payload, uint32_t payloadLen)
{
    ReqDevPtrPayload* req = reinterpret_cast<ReqDevPtrPayload*>(payload);
    void* ptr = (void*)(req->ptr);
    free(ptr);
    RspFreeDevPtrPayload donePayload{};
    donePayload.status = 0;
    HCCL_VM_INFO("[device] free wqe buffer :{:p}", ptr);

    DeviceSendMsg(PIPE_RSP_FREE_WQE_PTR, &donePayload, sizeof(donePayload));
    return 0;
}

} // namespace sim
