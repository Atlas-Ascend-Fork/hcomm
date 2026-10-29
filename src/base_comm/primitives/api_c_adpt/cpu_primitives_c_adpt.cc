/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/cast_utils_pub.h"
#include "hccl_api_data.h"
#include "new/hccl_primitive_local.h"
#include "new/hccl_primitive_remote.h"
#include "thread.h"
#include "launch_context.h"
#include "host/host_cpu_roce_channel.h"
#include "hcclCommOp.h"
#include "comm_engine/thread/dfx/thread_dfx_pub.h"
#include "endpoint_pair/channel/dfx/channel_dfx_pub.h"
#include "adapter_prof.h"
#include "hccl_diag.h"
#include "exception_handler.h"
#include "task_param.h"
#include "nic_plugin_manager.h"
#include "common/dev_type_utils_pub.h"
#include "hcomm_c_adpt_common.h"
#include "nic_plugin_holder.h"

using Hccl::LogAddrAlignCheck512B;
using Hccl::LogDescsAddrAlignCheck512B;

using namespace hccl;
thread_local LaunchContext g_threadLaunchCtx;

void AddThreadWithTag(ThreadHandle thread) { g_threadLaunchCtx.AddThreadWithTag(thread); }

bool IsSupportReduce(HcommDataType dataType, HcommReduceOp op)
{
    bool checkDataType
        = (dataType == HCOMM_DATA_TYPE_FP32 || dataType == HCOMM_DATA_TYPE_FP16 || dataType == HCOMM_DATA_TYPE_INT8
           || dataType == HCOMM_DATA_TYPE_INT16 || dataType == HCOMM_DATA_TYPE_INT32
           || dataType == HCOMM_DATA_TYPE_BFP16);
    bool checkReduceType = (op == HCOMM_REDUCE_SUM || op == HCOMM_REDUCE_MAX || op == HCOMM_REDUCE_MIN);
    return checkDataType && checkReduceType;
}

int32_t HcommSetNotifyWaitTimeOut(float timeOut)
{
    if (std::isnan(timeOut) || timeOut < 0.0f || timeOut > static_cast<float>(UINT32_MAX)) {
        HCCL_ERROR("[%s] in cpu timeOut[%f]s is invalid.", __func__, timeOut);
        return HCCL_E_PARA;
    }
    uint32_t timeOutInt = static_cast<uint32_t>(timeOut);
    HCCL_INFO("[%s] START in cpu. timeOut[%u]s.", __func__, timeOutInt);
    return g_threadLaunchCtx.SetNotifyWaitTimeOut(timeOutInt);
}

int32_t HcommThreadResAcquireTimeOut(float timeOut)
{
    HCCL_ERROR("[%s] timeOut[%f]s, not support in cpu.", __func__, timeOut);
    return HCCL_E_NOT_SUPPORT;
}

int32_t HcommLocalCopyOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    HCCL_INFO("[%s] START. thread[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, dst, src, len);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    AddThreadWithTag(thread);
    LogAddrAlignCheck512B(__func__, dst, src);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    if (hcomm::EqualToOrGreaterThan950().value_or(false)) {
        CHK_RET(threadPtr->LocalCopy(dst, src, len));
    } else {
        HcclBuf srcBuf{const_cast<void*>(src), len, nullptr};
        HcclBuf dstBuf{dst, len, nullptr};
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        HcclResult ret = HcclLocalCopy(stream, &dstBuf, &srcBuf);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL. thread[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, dst, src, len),
            ret);
    }
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommLocalReduceOnThread(
    ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    HCCL_INFO(
        "[%s] START. thread[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d].", __func__,
        thread, dst, src, count, dataType, reduceOp);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    CHK_PRT_RET(
        (IsSupportReduce(dataType, reduceOp) == false),
        HCCL_ERROR(
            "[HcommLocalReduceOnThread] Not support reduce, "
            "dst[%p], src[%p], count[%llu], dataType[%d], reduceOp[%d]",
            dst, src, count, dataType, reduceOp),
        HCCL_E_PARA);
    AddThreadWithTag(thread);
    LogAddrAlignCheck512B(__func__, dst, src);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    uint64_t len = count * SIZE_TABLE[dataType];

    if (hcomm::EqualToOrGreaterThan950().value_or(false)) {
        CHK_RET(threadPtr->LocalReduce(dst, src, len, dataType, reduceOp));
    } else {
        HcclBuf srcBuf{const_cast<void*>(src), len, nullptr};
        HcclBuf dstBuf{dst, len, nullptr};
        HcclReduceInfo reduceInfo{static_cast<HcclDataType>(dataType), static_cast<HcclReduceOp>(reduceOp)};
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        HcclResult ret = HcclLocalCopyReduce(stream, &dstBuf, &srcBuf, reduceInfo);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL. thread[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d].",
                __func__, thread, dst, src, count, dataType, reduceOp),
            ret);
    }
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx)
{
    HCCL_INFO(
        "[%s] START. thread[0x%llx], dstThread[0x%llx], dstNotifyIdx[%u].", __func__, thread, dstThread, dstNotifyIdx);

    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    if (hcomm::EqualToOrGreaterThan950().value_or(false)) {
        HcclResult ret = threadPtr->LocalNotifyRecord(dstThread, dstNotifyIdx);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL. thread[0x%llx], dstThread[0x%llx], notifyIdx[%u].", __func__, thread, dstThread,
                dstNotifyIdx),
            ret);
    } else {
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        LocalNotify* notify = GetNotify(dstThread, dstNotifyIdx);
        CHK_PTR_NULL(notify);

        HcclResult ret = HcclLocalNotifyRecord(stream, notify);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL. thread[0x%llx], dstThread[0x%llx], notifyIdx[%u].", __func__, thread, dstThread,
                dstNotifyIdx),
            ret);
    }

    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut)
{
    HCCL_INFO("[%s] START. thread[0x%llx], notifyIdx[%u], timeOut[%u]s.", __func__, thread, notifyIdx, timeOut);

    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    if (hcomm::EqualToOrGreaterThan950().value_or(false)) {
        HcclResult ret = threadPtr->LocalNotifyWait(notifyIdx, timeOut);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s] FAIL. thread[0x%llx], notifyIdx[%u], timeOut[%u]s.", __func__, thread, notifyIdx, timeOut),
            ret);
    } else {
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);
        LocalNotify* notify = GetNotify(thread, notifyIdx);
        CHK_PTR_NULL(notify);

        HcclResult ret = HcclLocalNotifyWait(stream, notify, timeOut);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s] FAIL. thread[0x%llx], notifyIdx[%u], timeOut[%u]s.", __func__, thread, notifyIdx, timeOut),
            ret);
    }
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommAclrtNotifyRecordOnThread(ThreadHandle thread, uint64_t dstNotifyId)
{
    HCCL_INFO("[%s] START. thread[0x%llx], dstNotifyId[%u].", __func__, thread, dstNotifyId);

    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    HcclResult ret = HcclLocalBareNotifyRecord(stream, dstNotifyId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] FAIL. thread[0x%llx], dstNotifyId[%u].", __func__, thread, dstNotifyId),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommAclrtNotifyWaitOnThread(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut)
{
    HCCL_INFO("[%s] START. thread[0x%llx], notifyId[%llu], timeOut[%u]s.", __func__, thread, notifyId, timeOut);

    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    HcclResult ret = HcclLocalBareNotifyWait(stream, notifyId, timeOut);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s] FAIL. thread[0x%llx], notifyId[%llu], timeOut[%u]s.", __func__, thread, notifyId, timeOut),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

HcclResult CommTaskPrepare(const char* key, uint32_t keyLen) // host ffts+使用
{
    std::string keyStr = "temp_key";
    if (key != nullptr && keyLen != 0) {
        keyStr = std::string(key, keyLen);
        HCCL_DEBUG("[CommTaskPrepare]key[%s], keyLen[%u]", key, keyLen);
    } else {
        HCCL_DEBUG("[CommTaskPrepare]disable cache, key[0x%llx], keyLen[%u]", key, keyLen);
    }

    return HcclTaskPrepare(const_cast<char_t*>(keyStr.c_str()), keyStr.length());
}

HcclResult CommTaskLaunch(ThreadHandle* threads, uint32_t threadNum) // host ffts+或aicpu stars使用"
{
    CHK_PTR_NULL(threads);
    CHK_PRT_RET(threadNum < 1, HCCL_ERROR("[CommTaskLaunch]threadNum is less than 1"), HCCL_E_PARA);

    Thread* threadPtr = ReinterpretAs<Thread*>(threads[0]);
    CHK_PTR_NULL(threadPtr);

    std::vector<hccl::Stream> streams;
    for (uint32_t i = 0; i < threadNum; i++) {
        hccl::Stream* stream = GetStream(threads[i]);
        CHK_PTR_NULL(stream);
        streams.push_back(*stream);
    }

    return HcclTaskLaunch(streams.data(), threadNum);
}

HcclResult DispatchAllStreams(const ThreadHandle* threads, uint32_t threadNum)
{
    CHK_PTR_NULL(threads);
    CHK_PRT_RET(threadNum < 1, HCCL_ERROR("[DispatchAllStreams]threadNum is less than 1"), HCCL_E_PARA);

    HCCL_WARNING("[DispatchAllStreams] DispatchAllStreams is only supported on A5 device, skip");
    return HCCL_E_NOT_SUPPORT;
}

int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    LogAddrAlignCheck512B(__func__, dst, src);
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->writeOnThread(ch->GetNicCtx(), thread, dst, src, len);
    }
    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, channel,
        dst, src, len);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclBuf locBuf{const_cast<void*>(src), len, nullptr};
    HcclBuf rmtBuf{dst, len, nullptr};

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    HcclResult ret = HcclRemoteWrite(stream, ReinterpretAs<void*>(channel), &rmtBuf, &locBuf);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread,
            channel, dst, src, len),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommBatchTransferOnThread(
    ThreadHandle thread, ChannelHandle channel, const HcommBatchTransferDesc* transferDescs, uint32_t transferDescNum)
{
    LogDescsAddrAlignCheck512B(__func__, transferDescs, transferDescNum);
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->batchTransferOnThread(ch->GetNicCtx(), thread, transferDescs, transferDescNum);
    }
    HCCL_ERROR(" [HcommBatchTransferOnThread] not support in cpu");
    return HCCL_E_NOT_SUPPORT;
}

int32_t HcommWriteReduceOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    LogAddrAlignCheck512B(__func__, dst, src);
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->writeReduceOnThread(ch->GetNicCtx(), thread, dst, src, count, dataType, reduceOp);
    }

    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], "
        "reduceOp[%d].",
        __func__, thread, channel, dst, src, count, dataType, reduceOp);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    CHK_PRT_RET(
        (IsSupportReduce(dataType, reduceOp) == false),
        HCCL_ERROR(
            "[HcommWriteReduceOnThread] Not support reduce, "
            "dst[%p], src[%p], count[%llu], dataType[%d], reduceOp[%d]",
            dst, src, count, dataType, reduceOp),
        HCCL_E_PARA);
    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    uint64_t len = count * SIZE_TABLE[dataType];

    HcclBuf locBuf{const_cast<void*>(src), len, nullptr};
    HcclBuf rmtBuf{dst, len, nullptr};
    HcclReduceInfo reduceInfo{static_cast<HcclDataType>(dataType), static_cast<HcclReduceOp>(reduceOp)};

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    HcclResult ret = HcclRemoteWriteReduce(stream, ReinterpretAs<void*>(channel), &rmtBuf, &locBuf, reduceInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], "
            "reduceOp[%d].",
            __func__, thread, channel, dst, src, count, dataType, reduceOp),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

HcclResult CommWriteReduceWithNotify(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp, uint32_t remoteNotifyIdx)
{
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);

    CHK_PRT_RET(
        (IsSupportReduce(dataType, reduceOp) == false),
        HCCL_ERROR(
            "[CommWriteReduceWithNotify] Not support reduce, "
            "dst[%p], src[%p], count[%llu], dataType[%d], reduceOp[%d]",
            dst, src, count, dataType, reduceOp),
        HCCL_E_PARA);
    AddThreadWithTag(thread);
    HcclBuf locBuf{const_cast<void*>(src), count * SIZE_TABLE[dataType], nullptr};
    HcclBuf rmtBuf{dst, count * SIZE_TABLE[dataType], nullptr};
    HcclReduceInfo reduceInfo{static_cast<HcclDataType>(dataType), static_cast<HcclReduceOp>(reduceOp)};

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    return HcclRemoteWriteReduceWithNotify(
        stream, ReinterpretAs<void*>(channel), &rmtBuf, &locBuf, reduceInfo, remoteNotifyIdx);
}

int32_t HcommWriteWithNotifyOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    LogAddrAlignCheck512B(__func__, dst, src);
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->writeWithNotifyOnThread(ch->GetNicCtx(), thread, dst, src, len, remoteNotifyIdx);
    }

    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu], remoteNotifyIdx[%u].",
        __func__, thread, channel, dst, src, len, remoteNotifyIdx);

    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);

    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclBuf locBuf{const_cast<void*>(src), len, nullptr};
    HcclBuf rmtBuf{dst, len, nullptr};

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    HcclResult ret
        = HcclRemoteWriteWithNotify(stream, ReinterpretAs<void*>(channel), &rmtBuf, &locBuf, remoteNotifyIdx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu], remoteNotifyIdx[%u].",
            __func__, thread, channel, dst, src, len, remoteNotifyIdx),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommWriteReduceWithNotifyOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp, uint32_t remoteNotifyIdx)
{
    LogAddrAlignCheck512B(__func__, dst, src);
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->writeReduceWithNotifyOnThread(
            ch->GetNicCtx(), thread, dst, src, count, dataType, reduceOp, remoteNotifyIdx);
    }

    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], "
        "reduceOp[%d], remoteNotifyIdx[%u].",
        __func__, thread, channel, dst, src, count, dataType, reduceOp, remoteNotifyIdx);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    uint64_t len = count * SIZE_TABLE[dataType];

    HcclResult ret = HCCL_SUCCESS;

    ret = HCCL_E_NOT_SUPPORT;
    (void)len;

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], "
            "reduceOp[%d], remoteNotifyIdx[%u].",
            __func__, thread, channel, dst, src, count, dataType, reduceOp, remoteNotifyIdx),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    LogAddrAlignCheck512B(__func__, dst, src);
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->readOnThread(ch->GetNicCtx(), thread, dst, src, len);
    }

    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, channel,
        dst, src, len);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclBuf locBuf{dst, len, nullptr};
    HcclBuf rmtBuf{const_cast<void*>(src), len, nullptr};

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    HcclResult ret = HcclRemoteRead(stream, ReinterpretAs<void*>(channel), &locBuf, &rmtBuf);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread,
            channel, dst, src, len),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommReadReduceOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    LogAddrAlignCheck512B(__func__, dst, src);
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->readReduceOnThread(ch->GetNicCtx(), thread, dst, src, count, dataType, reduceOp);
    }

    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], "
        "reduceOp[%d].",
        __func__, thread, channel, dst, src, count, dataType, reduceOp);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    CHK_PRT_RET(
        (IsSupportReduce(dataType, reduceOp) == false),
        HCCL_ERROR(
            "[HcommReadReduceOnThread] Not support reduce, "
            "dst[%p], src[%p], count[%llu], dataType[%d], reduceOp[%d]",
            dst, src, count, dataType, reduceOp),
        HCCL_E_PARA);
    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    uint64_t len = count * SIZE_TABLE[dataType];

    HcclBuf locBuf{dst, len, nullptr};
    HcclBuf rmtBuf{const_cast<void*>(src), len, nullptr};
    HcclReduceInfo reduceInfo{static_cast<HcclDataType>(dataType), static_cast<HcclReduceOp>(reduceOp)};

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    HcclResult ret = HcclRemoteReadReduce(stream, ReinterpretAs<void*>(channel), &locBuf, &rmtBuf, reduceInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], "
            "reduceOp[%d].",
            __func__, thread, channel, dst, src, count, dataType, reduceOp),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommWriteNbiOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->writeNbiOnThread(ch->GetNicCtx(), thread, dst, src, len);
    }
    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, channel,
        dst, src, len);

    (void)thread;
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);

    HcclResult ret = HCCL_SUCCESS;
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (*isGe950) {
        auto* const channelPtr = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(channelPtr);
        ret = channelPtr->Write(dst, src, len);
    } else {
        ret = HCCL_E_NOT_SUPPORT;
    }
    if (ret == HCCL_E_AGAIN) {
        return ret;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread,
            channel, dst, src, len),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommWriteNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->writeNbi(ch->GetNicCtx(), dst, src, len);
    }
    return HcommWriteNbiOnThread(0, channel, dst, src, len);
}

int32_t HcommWriteWithNotifyNbiOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->writeWithNotifyNbiOnThread(ch->GetNicCtx(), thread, dst, src, len, remoteNotifyIdx);
    }
    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu], remoteNotifyIdx[%u].",
        __func__, thread, channel, dst, src, len, remoteNotifyIdx);

    (void)thread;
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);
    HcclResult ret = HCCL_SUCCESS;
    /* 复合条件含 thread == 0 分支，取数失败须显式返回错误，不得整体 value_or(false) 误放行 */
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_INTERNAL);
    if (*isGe950 || thread == 0) {
        auto* const channelPtr = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(channelPtr);
        ret = channelPtr->WriteWithNotify(dst, src, len, remoteNotifyIdx);
    } else {
        ret = HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu], remoteNotifyIdx[%u].",
            __func__, thread, channel, dst, src, len, remoteNotifyIdx),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t
HcommWriteWithNotifyNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->writeWithNotifyNbi(ch->GetNicCtx(), dst, src, len, remoteNotifyIdx);
    }
    return HcommWriteWithNotifyNbiOnThread(0, channel, dst, src, len, remoteNotifyIdx);
}

int32_t HcommReadNbiOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->readNbiOnThread(ch->GetNicCtx(), thread, dst, src, len);
    }
    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, channel,
        dst, src, len);

    (void)thread;
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);

    HcclResult ret = HCCL_SUCCESS;
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (*isGe950) {
        auto* const channelPtr = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(channelPtr);
        ret = channelPtr->Read(dst, src, len);
    } else {
        ret = HCCL_E_NOT_SUPPORT;
    }
    if (ret == HCCL_E_AGAIN) {
        return ret;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread,
            channel, dst, src, len),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommReadNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->readNbi(ch->GetNicCtx(), dst, src, len);
    }
    return HcommReadNbiOnThread(0, channel, dst, src, len);
}

int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->notifyRecordOnThread(ch->GetNicCtx(), thread, remoteNotifyIdx);
    }
    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], remoteNotifyIdx[%u].", __func__, thread, channel,
        remoteNotifyIdx);

    HcclResult ret = HCCL_SUCCESS;
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    /* 910B 分支无对应谓词，保留原始 devType 判断；上方 CHK_RET 已保证取数成功，value_or(false) 与原行为等价 */
    if (hcomm::EqualToOrGreaterThan950().value_or(false) || (thread == 0 && devType == DevType::DEV_TYPE_910B)) {
        auto* const channelPtr = ReinterpretAs<hcomm::Channel*>(channel);
        CHK_PTR_NULL(channelPtr);
        ret = channelPtr->NotifyRecord(remoteNotifyIdx);
    } else { // Non-950 devices use thread-based notify.
        AddThreadWithTag(thread);

        Thread* threadPtr = ReinterpretAs<Thread*>(thread);
        CHK_PTR_NULL(threadPtr);

        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        ret = HcclRemoteNotifyRecord(stream, ReinterpretAs<void*>(channel), remoteNotifyIdx);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], remoteNotifyIdx[%u].", __func__, thread, channel,
            remoteNotifyIdx),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommChannelNotifyRecord(ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->notifyRecord(ch->GetNicCtx(), remoteNotifyIdx);
    }
    HCCL_INFO("[%s] START. channel[0x%llx], remoteNotifyIdx[%u].", __func__, channel, remoteNotifyIdx);

    HcclResult ret = HCCL_SUCCESS;
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (*isGe950) {
        auto* const channelPtr = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(channelPtr);
        ret = channelPtr->NotifyRecord(remoteNotifyIdx);
    } else {
        ret = HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s] FAIL. channel[0x%llx], remoteNotifyIdx[%u].", __func__, channel, remoteNotifyIdx), ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t
HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->notifyWaitOnThread(ch->GetNicCtx(), thread, localNotifyIdx, timeOut);
    }
    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], localNotifyIdx[%u], timeOut[%u]s.", __func__, thread, channel,
        localNotifyIdx, timeOut);

    HcclResult ret = HCCL_SUCCESS;
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    /* 910B 分支无对应谓词，保留原始 devType 判断；上方 CHK_RET 已保证取数成功，value_or(false) 与原行为等价 */
    if (hcomm::EqualToOrGreaterThan950().value_or(false) || (thread == 0 && devType == DevType::DEV_TYPE_910B)) {
        auto* const channelPtr = ReinterpretAs<hcomm::Channel*>(channel);
        CHK_PTR_NULL(channelPtr);
        ret = channelPtr->NotifyWait(localNotifyIdx, timeOut);
    } else { // Non-950 devices use thread-based notify.
        AddThreadWithTag(thread);

        Thread* threadPtr = ReinterpretAs<Thread*>(thread);
        CHK_PTR_NULL(threadPtr);

        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        ret = HcclRemoteNotifyWait(stream, ReinterpretAs<void*>(channel), localNotifyIdx, timeOut);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], localNotifyIdx[%u], timeOut[%u]s.", __func__, thread, channel,
            localNotifyIdx, timeOut),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommChannelNotifyWait(ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->notifyWait(ch->GetNicCtx(), localNotifyIdx, timeOut);
    }
    HCCL_INFO(
        "[%s] START. channel[0x%llx], localNotifyIdx[%u], timeOut[%u].", __func__, channel, localNotifyIdx, timeOut);

    HcclResult ret = HCCL_SUCCESS;
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (*isGe950) {
        auto* const channelPtr = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(channelPtr);
        ret = channelPtr->NotifyWait(localNotifyIdx, timeOut);
    } else {
        ret = HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. channel[0x%llx], localNotifyIdx[%u], timeOut[%u].", __func__, channel, localNotifyIdx, timeOut),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

HcclResult CommFence(ThreadHandle thread, ChannelHandle channel) // 控制前后的任务保序
{
    HCCL_DEBUG("[CommFence] thread[0x%llx], channel[0x%llx].", thread, channel);

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    return HcclRemoteFence(stream, ReinterpretAs<void*>(channel), false);
}

int32_t HcommSetLaunchMode(const char* launchTag, HcommLaunchMode mode)
{
    HCCL_DEBUG("HcommSetLaunchMode launchTag[%s]", launchTag);
    return g_threadLaunchCtx.SetLaunchMode(launchTag, mode);
}

int32_t HcommBatchModeStart(const char* batchTag) { return HcommSetLaunchMode(batchTag, HCOMM_LAUNCH_MODE_BATCH); }

int32_t HcommBatchModeEnd(const char* batchTag) { return HcommSetLaunchMode(batchTag, HCOMM_LAUNCH_MODE_EAGER); }

HcommResult
HcommThreadRegisterDfx(ThreadHandle thread, std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback)
{
    Thread* threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);
    CHK_RET(threadPtr->SetAddTaskInfoCallback(callback));
    return HCCL_SUCCESS;
}

HcommResult HcommThreadRegisterCheckExecStatus(ThreadHandle thread, std::function<HcclResult(bool)> callback)
{
    Thread* threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);
    CHK_RET(threadPtr->SetCheckExecStatusCallback(callback));
    return HCCL_SUCCESS;
}

HcommResult
HcommDpuChannelRegisterDfx(ChannelHandle channel, std::function<HcclResult(const Hccl::TaskParam&, u64)> callback)
{
    auto* const hostCpuRoceChannelPtr = ReinterpretAs<hcomm::HostCpuRoceChannel*>(channel);
    CHK_PTR_NULL(hostCpuRoceChannelPtr);
    CHK_RET(hostCpuRoceChannelPtr->SetDfxCallback(callback));
    return HCCL_SUCCESS;
}

int32_t HcommChannelDrainOnThread(ThreadHandle thread, ChannelHandle channel)
{
    HCCL_INFO("[%s] START. thread[0x%llx], channel[0x%llx].", __func__, thread, channel);

    int32_t ret = HCCL_SUCCESS;
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        ret = ch->GetNicOps()->drainOnThread(ch->GetNicCtx(), thread);
    } else {
        auto isGe950 = hcomm::EqualToOrGreaterThan950();
        CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
        if (*isGe950) {
            auto* ch = CHANNEL_FROM_HANDLE(channel);
            CHK_PTR_NULL(ch);
            ret = ch->ChannelDrain();
        } else {
            Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
            CHK_PTR_NULL(threadPtr);
            Stream* stream = GetStream(thread);
            CHK_PTR_NULL(stream);
            ret = HcclRemoteDrain(stream, ReinterpretAs<void*>(channel));
        }
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] FAIL. thread[0x%llx], channel[0x%llx].", __func__, thread, channel), ret);

    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommChannelFenceOnThread(ThreadHandle thread, ChannelHandle channel)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->fenceOnThread(ch->GetNicCtx(), thread);
    }
    HCCL_INFO("[%s] START. thread[0x%llx], channel[0x%llx].", __func__, thread, channel);

    (void)thread;
    HcclResult ret = HCCL_SUCCESS;
    /* 复合条件含 thread == 0 分支，取数失败须显式返回错误，不得整体 value_or(false) 误放行 */
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_INTERNAL);
    if (*isGe950 || thread == 0) {
        auto* const channelPtr = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(channelPtr);
        ret = channelPtr->ChannelFence();
    } else {
        ret = HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] FAIL. thread[0x%llx], channel[0x%llx].", __func__, thread, channel), ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommChannelFence(ChannelHandle channel)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->fence(ch->GetNicCtx());
    }
    return HcommChannelFenceOnThread(0, channel);
}

HcclResult
HcclDfxRegOpInfo([[maybe_unused]] HcclComm comm, [[maybe_unused]] void* hcclDfxOpInfo) // 兼容性接口，后续删除
{
    HCCL_WARNING("%s not support", __func__);
    return HCCL_SUCCESS;
}

int32_t
HcommChannelNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = PLUGIN_CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->notifyWaitOnThreadWithDefaultTimeout(ch->GetNicCtx(), thread, localNotifyIdx);
    }
    HCCL_ERROR(
        "[%s] thread[0x%llx], channel[0x%llx], localNotifyIdx[%u] not support in cpu.", __func__, thread, channel,
        localNotifyIdx);
    return HCCL_E_NOT_SUPPORT;
}

int32_t HcommThreadNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle thread, uint32_t notifyIdx)
{
    HCCL_INFO("[%s] data cpu START. thread[0x%llx], notifyIdx[%u].", __func__, thread, notifyIdx);

    uint32_t notifyWaitTimeout;
    g_threadLaunchCtx.GetNotifyWaitTimeOut(notifyWaitTimeout);

    HCCL_DEBUG("[%s] data cpu Using default timeout: %u s", __func__, notifyWaitTimeout);

    int32_t ret = HcommThreadNotifyWaitOnThread(thread, notifyIdx, notifyWaitTimeout);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] data cpu HcommThreadNotifyWaitOnThread FAILED. thread[0x%llx], notifyIdx[%u], ret[%d]", __func__,
            thread, notifyIdx, ret);
        return ret;
    }

    HCCL_INFO("[%s] data cpu SUCCESS.", __func__);
    return HCCL_SUCCESS;
}
