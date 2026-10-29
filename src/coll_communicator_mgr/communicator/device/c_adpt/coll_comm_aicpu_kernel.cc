/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_aicpu_kernel.h"
#include "aicpu_init_param.h"
#include "coll_comm_aicpu_mgr.h"
#include "coll_comm_aicpu_kernel_adpt.h"
#include "common/dev_type_utils_pub.h"
#include "base_transport_lite_impl.h"
#include "aicpu_indop_env.h"
#include "aicpu_ts_channel_entity.h"
#include "aicpu_channel_process.h"
#include "framework/aicpu_hccl_process.h"
#include "log.h"
#include "common/cast_utils_pub.h"

extern "C" {
__attribute__((visibility("default"))) uint32_t RunAicpuIndOpThreadInit(void* args)
{
    CHK_PTR_NULL(args);
    uint64_t devAddr = *ReinterpretAs<uint64_t*>(args);
    ThreadMgrAicpuParam* param = ReinterpretAs<ThreadMgrAicpuParam*>(devAddr);
    CHK_PTR_NULL(param);
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (hcomm::EqualToOrGreaterThan950().value_or(false)) {
        HCCL_INFO(
            "[RunAicpuIndOpThreadInit] group[%s], threadNum[%u], deviceType[%u]", param->hcomId, param->threadNum,
            devType);
        return CollCommAicpuKernelAdptInitThreads(param);
    }
    return AicpuHcclProcess::AicpuIndOpThreadInit(param);
}

__attribute__((visibility("default"))) uint32_t RunAicpuIndOpNotify(void* args)
{
    CHK_PTR_NULL(args);
    uint64_t devAddr = *ReinterpretAs<uint64_t*>(args);
    NotifyMgrAicpuParam* param = ReinterpretAs<NotifyMgrAicpuParam*>(devAddr);
    CHK_PTR_NULL(param);
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (hcomm::EqualToOrGreaterThan950().value_or(false)) {
        HCCL_INFO(
            "[RunAicpuIndOpNotify] group[%s], notifyNum[%u], deviceType[%u]", param->hcomId, param->notifyNum, devType);
        return CollCommAicpuKernelAdptInitNotify(param);
    }
    return AicpuHcclProcess::AicpuIndOpNotifyInit(param);
}

__attribute__((visibility("default"))) uint32_t RunAicpuIndOpChannelUpdateV2(void* args)
{
    HCCL_RUN_INFO("RunAicpuIndOpChannelUpdateV2 start.");
    CHK_PTR_NULL(args);
    uint64_t devAddr = *ReinterpretAs<uint64_t*>(args);
    HcclChannelUrmaRes* commParam = ReinterpretAs<HcclChannelUrmaRes*>(devAddr);
    CHK_PTR_NULL(commParam);
    return CollCommAicpuKernelAdptUpdateChannel(commParam);
}

__attribute__((visibility("default"))) uint32_t RunAicpuCommInit(void* args)
{
    CHK_PRT_RET(args == nullptr, HCCL_ERROR("[%s]args is null.", __func__), HCCL_E_PTR);

    CommAicpuParam* commAicpuParam = static_cast<CommAicpuParam*>(args);
    CHK_RET(hrtSetWorkModeAicpu(true));
    CHK_RET(hrtSetlocalDevice(commAicpuParam->deviceLogicId));
    CHK_RET(hrtSetlocalDeviceType(static_cast<DevType>(commAicpuParam->deviceType)));

    // hrtSet* 前置后，谓词经缓存的 localDeviceType 取数，与原判断等价
    if (hcomm::EqualToOrGreaterThan950().value_or(false)) {
        HCCL_INFO(
            "[RunAicpuCommInit] group[%s], deviceLogicId[%u], devicePhyId[%u], deviceType[%u]", commAicpuParam->hcomId,
            commAicpuParam->deviceLogicId, commAicpuParam->devicePhyId, commAicpuParam->deviceType);
        return CollCommAicpuMgr::GetInstance().InitComm(commAicpuParam);
    }
    return AicpuHcclProcess::AicpuIndOpCommInit(commAicpuParam);
}

__attribute__((visibility("default"))) uint32_t RunAicpuDfxInitV2(void* args)
{
    HCCL_RUN_INFO("RunAicpuDfxInitV2 start.");
    CHK_PRT_RET(args == nullptr, HCCL_ERROR("[%s]args is null.", __func__), HCCL_E_PTR);
    struct InitTask {
        u64 context;
        char commTag[256];
    };
    InitTask* ctxArgs = static_cast<InitTask*>(args);
    CHK_PRT_RET(ctxArgs == nullptr, HCCL_ERROR("[%s]ctxArgs is null.", __func__), HCCL_E_PTR);
    HcclDfxOpInfo* dfxOpInfo = reinterpret_cast<HcclDfxOpInfo*>(ctxArgs->context);
    CollCommAicpu* currentComm = CollCommAicpuMgr::GetInstance().GetCurrentComm();
    CHK_PTR_NULL(currentComm);
    return currentComm->InitDfxOpInfo(dfxOpInfo);
}

__attribute__((visibility("default"))) uint32_t RunAicpuCommRecordChannels(void* args)
{
    HCCL_RUN_INFO("RunAicpuCommRecordChannels start.");
    CHK_PTR_NULL(args);
    uint64_t devAddr = *reinterpret_cast<uint64_t*>(args);
    HcclAicpuCommRecordChannelsRes* commParam = reinterpret_cast<HcclAicpuCommRecordChannelsRes*>(devAddr);
    CHK_PTR_NULL(commParam);
    CHK_PTR_NULL(commParam->deviceHandleList);

    CollCommAicpu* collComm = CollCommAicpuMgr::GetInstance().FindCommByGroup(std::string(commParam->commTag));
    if (collComm == nullptr) {
        HCCL_ERROR("[RunAicpuCommRecordChannels] group[%s] not found", commParam->commTag);
        return HCCL_E_NOT_FOUND;
    }
    CHK_RET(
        collComm->RecordChannels(reinterpret_cast<const ChannelHandle*>(commParam->deviceHandleList), commParam->num));
    if (commParam->remoteRankList == nullptr) {
        return HCCL_SUCCESS;
    }
    hccl::HcclCommDfxLite* dfxLite = collComm->GetHcclCommDfxLite();
    CHK_PTR_NULL(dfxLite);
    const ChannelHandle* handleList = reinterpret_cast<const ChannelHandle*>(commParam->deviceHandleList);
    const bool taskExceptionEnable = hcomm::GetTaskExceptionEnable();
    for (uint32_t i = 0; i < commParam->num; ++i) {
        ChannelHandle handle = handleList[i];
        // ctx 模式下 handle 为 ctxPtr，解包为 deviceChannel
        (void)UnwrapChannelHandle(handle);
        // 未在册跳过，避免按错误 vtable 虚调用
        if (!AicpuChannelProcess::IsLiteTransportHandle(handle)) {
            continue;
        }
        auto* transport = reinterpret_cast<Hccl::BaseTransportLiteImpl*>(handle);
        transport->SetTaskExceptionEnable(taskExceptionEnable);
        dfxLite->AddChannelRemoteRankId(handle, commParam->remoteRankList[i]);
    }
    return HCCL_SUCCESS;
}

// device 侧地址条目下发：把条目录进已有 transport 的表
__attribute__((visibility("default"))) uint32_t RunAicpuChannelMemUpdate(void* args)
{
    HCCL_RUN_INFO("RunAicpuChannelMemUpdate start.");
    CHK_PTR_NULL(args);
    uint64_t devAddr = *reinterpret_cast<uint64_t*>(args);
    HcclAicpuChannelMemUpdateRes* commParam = reinterpret_cast<HcclAicpuChannelMemUpdateRes*>(devAddr);
    CHK_PTR_NULL(commParam);
    CHK_PTR_NULL(commParam->deviceHandleList);
    CHK_PTR_NULL(commParam->packAddrList);
    CHK_PTR_NULL(commParam->packSizeList);

    for (u32 index = 0; index < commParam->num; ++index) {
        if (commParam->packSizeList[index] == 0) {
            continue;
        }
        ChannelHandle handle = commParam->deviceHandleList[index];
        // ctx 模式下入参为 ctxPtr，解包为 ctx->deviceChannel；非 ctx 模式原样返回
        HcclResult unwrapRet = UnwrapChannelHandle(handle);
        if (unwrapRet != HCCL_SUCCESS) {
            return unwrapRet;
        }
        // 解包仅校验 ctx 头部，句柄可能非 lite transport 或已销毁悬垂，未在册跳过，避免按错误 vtable 虚调用
        if (!AicpuChannelProcess::IsLiteTransportHandle(handle)) {
            HCCL_WARNING(
                "[%s] index[%u] handle[0x%llx] not lite transport, skip mem update", __func__, index,
                static_cast<unsigned long long>(handle));
            continue;
        }
        auto* transport = reinterpret_cast<Hccl::BaseTransportLiteImpl*>(handle);

        const char* packSrc = reinterpret_cast<const char*>(commParam->packAddrList[index]);
        std::vector<char> packBuf(packSrc, packSrc + commParam->packSizeList[index]);
        Hccl::BinaryStream binaryStream(packBuf);
        u32 entryNum = 0;
        binaryStream >> entryNum;
        HCCL_INFO(
            "RunAicpuChannelMemUpdate index[%u], entryNum[%u], packSize[%u].", index, entryNum,
            commParam->packSizeList[index]);
        for (u32 entryIdx = 0; entryIdx < entryNum; ++entryIdx) {
            Hccl::RmaBufEntryLite entry;
            Hccl::UnpackRmaBufEntry(binaryStream, entry);
            HcclResult entryRet = transport->UpdateMemInfoEntry(entry);
            if (entryRet != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "RunAicpuChannelMemUpdate UpdateMemInfoEntry failed, index[%u], entryIdx[%u], ret[%d].", index,
                    entryIdx, entryRet);
                return entryRet;
            }
        }
    }
    return HCCL_SUCCESS;
}
}
