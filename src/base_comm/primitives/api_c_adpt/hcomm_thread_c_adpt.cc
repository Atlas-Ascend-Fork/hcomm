/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "hcomm_c_adpt.h"
#include "hcomm_c_adpt_common.h"
#include "hcomm_thread_c_adpt.h"
#include "hcomm_res.h"
#include "hcomm_res_defs.h"
#include "hcomm_result_defs.h"
#include "hccl/hccl_res.h"
#include "res_pub.h"
#include "log.h"
#include "thread.h"
#include "cpu_ts_thread.h"
#include "aicpu_launch_manager.h"
#include "param_check_pub.h"
#include "comm_engine_utils.h"
#include "exception_handler.h"
#include "adapter_rts_common.h"
#include "../hcomm_res_mgr.h"

using namespace hcomm;

// 创建并初始化一个线程，登记自身句柄映射（供按引擎查线程时查到自己），并加入待保存列表
static HcclResult CreateAndInitThread(
    CommEngine engine, ThreadType type, hccl::StreamType streamType, hccl::NotifyLoadType notifyLoadType,
    uint32_t notifyNum, uint32_t index, std::vector<std::shared_ptr<hccl::Thread>>& newThreads)
{
    std::shared_ptr<hccl::Thread> threadPtr;
    HcclResult ret = hccl::CreateThread(engine, type, streamType, notifyNum, notifyLoadType, threadPtr);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] Failed to create thread at index[%u], ret[%d]", __func__, index, ret),
        ret);
    ret = threadPtr->Init();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] Failed to init thread at index[%u], ret[%d]", __func__, index, ret), ret);
    // 在线程内部登记"引擎->句柄"映射，后续 FindThreadByCommEngine 按引擎查线程时能查到自己
    ThreadHandle selfHandle = reinterpret_cast<ThreadHandle>(threadPtr.get());
    CHK_RET(threadPtr->AddThreadHandleToMap(engine, selfHandle));
    newThreads.emplace_back(std::move(threadPtr));
    return HCCL_SUCCESS;
}

HcommResult
HcommThreadAlloc(CommEngine engine, uint32_t threadNum, const uint32_t* notifyNumPerThread, ThreadHandle* threads)
{
    CHK_PTR_NULL(threads);
    CHK_PTR_NULL(notifyNumPerThread);
    (void)HcommResMgrInit();
    CHK_PRT_RET(
        threadNum == 0 || threadNum > HCOMM_THREADNUM_MAX_NUM,
        HCCL_ERROR(
            "[%s] Validate thread params failed. ThreadNum %u, range (0, %u]", __func__, threadNum,
            HCOMM_THREADNUM_MAX_NUM),
        HCOMM_E_PARA);
    HCCL_INFO(
        "[%s] ThreadAcquire begin. engine[%s], threadNum[%u], threads[%p]", __func__,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, threads);
    CHK_RET(RefreshCommEngineContext(engine));

    // 1. 获取引擎对应的类型
    hccl::NotifyLoadType notifyLoadType;
    hccl::StreamType streamType;
    CHK_RET(hccl::CommEngineToNotifyLoadType(engine, notifyLoadType));
    CHK_RET(hccl::CommEngineToStreamType(engine, streamType));

    // 2. 逐线程创建（每个线程 notifyNum 独立）
    std::vector<std::shared_ptr<hccl::Thread>> newThreads;
    newThreads.reserve(threadNum);
    for (uint32_t i = 0; i < threadNum; ++i) {
        CHK_RET(hccl::ValidateThreadParams(1, notifyNumPerThread[i]));
        CHK_RET(CreateAndInitThread(
            engine, THREAD_TYPE_TS, streamType, notifyLoadType, notifyNumPerThread[i], i, newThreads));
    }

    // 3. 登记线程句柄，生成对外的句柄值（若失败，newThreads 析构时会自动释放已创建的线程）
    CHK_RET(HcommResMgr::EnsureKernelBinLoaded(engine));
    CHK_RET(hccl::StoreThreadHandles(newThreads, threads, engine, HcommResMgr::GetBinHandle()));

    // 4. 最后才放入全局线程表：若先放入而句柄登记失败，线程会留在表里却没人拿到有效句柄，造成泄漏
    CHK_RET(hccl::SaveThreads(newThreads));

    HCCL_INFO(
        "[HcommThreadAlloc] ThreadAcquire done: engine[%s] threadNum[%u]",
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum);
    return HCOMM_SUCCESS;
}

HcommResult HcommThreadAllocWithConfig(
    CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads)
{
    return HcommThreadAllocWithCommConfig(engine, "", threadNum, type, config, threads);
}

HcommResult HcommThreadAllocWithCommConfigCheck(
    CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads)
{
    CHK_PTR_NULL(threads);
    CHK_PTR_NULL(config);
    CHK_PRT_RET(
        type == THREAD_TYPE_INVALID,
        HCCL_ERROR("[%s] thread type[%d] is invalid", __func__, static_cast<int32_t>(type)), HCOMM_E_PARA);
    CHK_PRT_RET(
        engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_CPU_TS,
        HCCL_ERROR(
            "[%s] commEngine[%d] CPU_TS/AICPU_TS not supported, use engine with ThreadType instead", __func__,
            static_cast<int32_t>(engine)),
        HCOMM_E_PARA);
    CHK_PRT_RET(threadNum == 0, HCCL_ERROR("[%s] threadNum[%u] is invalid", __func__, threadNum), HCOMM_E_PARA);
    HcommResult hcommRet = HcommResMgrInit();
    CHK_PRT_RET(
        hcommRet != HCCL_SUCCESS,
        HCCL_ERROR("[%s] HcommResMgrInit failed, ret[%d]", __func__, static_cast<int32_t>(hcommRet)), hcommRet);
    CHK_RET(RefreshCommEngineContext(engine));
    return HCOMM_SUCCESS;
}

HcommResult HcommThreadAllocWithCommConfig(
    CommEngine engine, const char* commId, uint32_t threadNum, ThreadType type, const ThreadConfig* config,
    ThreadHandle* threads)
{
    CHK_RET((HcclResult)HcommThreadAllocWithCommConfigCheck(engine, threadNum, type, config, threads));
    const std::string commIdStr = (commId != nullptr) ? std::string(commId) : std::string();
    HCCL_INFO(
        "[%s] begin. engine[%d], threadType[%d], threadNum[%u], threads[%p]", __func__, engine,
        static_cast<int32_t>(type), threadNum, threads);

    hccl::NotifyLoadType notifyLoadType;
    hccl::StreamType streamType;
    CHK_RET(hccl::GetNotifyLoadType(engine, type, notifyLoadType));
    CHK_RET(hccl::GetStreamType(engine, type, streamType));

    std::vector<std::shared_ptr<hccl::Thread>> newThreads;
    newThreads.reserve(threadNum);
    for (uint32_t i = 0; i < threadNum; ++i) {
        CHK_PRT_RET(
            config[i].header.magicWord != HCOMM_THREAD_CONFIG_MAGIC_WORD,
            HCCL_ERROR(
                "[%s] config[%u] magicWord[0x%x] mismatch, expected[0x%x], call ThreadConfigInit first", __func__, i,
                config[i].header.magicWord, HCOMM_THREAD_CONFIG_MAGIC_WORD),
            HCOMM_E_PARA);
        CHK_RET(hccl::ValidateThreadParams(1, config[i].notifyNumPerThread));
        CHK_RET(
            CreateAndInitThread(engine, type, streamType, notifyLoadType, config[i].notifyNumPerThread, i, newThreads));
    }

    CHK_RET(HcommResMgr::EnsureKernelBinLoaded(engine));
    CHK_RET(hccl::StoreThreadHandles(newThreads, commIdStr, threads, engine, HcommResMgr::GetBinHandle()));
    CHK_RET(hccl::SaveThreads(newThreads));

    HCCL_INFO(
        "[%s] done: engine[%d] threadType[%d] threadNum[%u]", __func__, engine, static_cast<int32_t>(type), threadNum);
    return HCOMM_SUCCESS;
}

HcommResult HcommThreadFree(const ThreadHandle* threads, uint32_t threadNum)
{
    CHK_PTR_NULL(threads);
    HcommResult hcommRet = HcommResMgrInit();
    CHK_PRT_RET(
        hcommRet != HCCL_SUCCESS,
        HCCL_ERROR("[%s] HcommResMgrInit failed, ret[%d]", __func__, static_cast<int32_t>(hcommRet)), hcommRet);
    return hccl::FreeThreads(threads, threadNum, HcommResMgr::GetBinHandle());
}

HcommResult HcommThreadAllocWithStream(CommEngine engine, aclrtStream stream, uint32_t notifyNum, ThreadHandle* thread)
{
    CHK_PTR_NULL(thread);

    HcommResult hcommRet = HcommResMgrInit();
    CHK_PRT_RET(
        hcommRet != HCCL_SUCCESS,
        HCCL_ERROR("[%s] HcommResMgrInit failed, ret[%d]", __func__, static_cast<int32_t>(hcommRet)), hcommRet);
    hccl::NotifyLoadType notifyLoadType;
    CHK_RET(hccl::CommHostEngineToNotifyLoadType(engine, notifyLoadType));
    std::shared_ptr<hccl::Thread> handle;
    EXCEPTION_CATCH(
        handle = std::make_shared<hccl::CpuTsThread>(stream, notifyNum, notifyLoadType), return HCOMM_E_PTR);
    handle->SetCommEngine(engine);
    CHK_RET(handle->Init());
    handle->SetIsMaster(true);

    // 登记句柄映射并生成返回句柄（若中途失败，newThreads 析构时会自动释放已创建的线程）
    std::vector<std::shared_ptr<hccl::Thread>> newThreads{std::move(handle)};
    CHK_RET(newThreads[0]->AddThreadHandleToMap(engine, reinterpret_cast<ThreadHandle>(newThreads[0].get())));
    CHK_RET(HcommResMgr::EnsureKernelBinLoaded(engine));
    CHK_RET(hccl::StoreThreadHandles(newThreads, thread, engine, HcommResMgr::GetBinHandle()));

    // 最后才放入全局线程表，防止线程已入表但句柄未生成，外部拿到无效句柄
    CHK_RET(hccl::SaveThreads(newThreads));

    HCCL_INFO(
        "[ThreadMgr] ThreadAcquireWithStream done: engine[%s] stream[%p], "
        "notifyNum[%u]",
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), stream, notifyNum);
    return HCOMM_SUCCESS;
}

HcommResult HcommThreadAcquireByNotify(rtStream_t stream, void** notifys, uint32_t notifyNum, ThreadHandle* thread)
{
    CHK_PTR_NULL(thread);
    CHK_PTR_NULL(notifys);
    CHK_PRT_RET(
        notifyNum == 0U || notifyNum > hccl::HCCL_THREAD_NOTIFY_MAX_NUM,
        HCCL_ERROR(
            "[%s] notifyNum[%u] is invalid, range (0, %u]", __func__, notifyNum, hccl::HCCL_THREAD_NOTIFY_MAX_NUM),
        HCCL_E_PARA);
    for (uint32_t i = 0; i < notifyNum; i++) {
        CHK_PRT_RET(notifys[i] == nullptr, HCCL_ERROR("[%s] notifys[%u] is nullptr", __func__, i), HCCL_E_PARA);
    }
    HcommResult hcommRet = HcommResMgrInit();
    CHK_PRT_RET(
        hcommRet != HCCL_SUCCESS,
        HCCL_ERROR("[%s] HcommResMgrInit failed, ret[%d]", __func__, static_cast<int32_t>(hcommRet)), hcommRet);
    // C ABI面以void**传递（保持纯C类型），此处还原为实际类型LocalNotify**。
    // 纯借用契约：对象所有权始终归调用方，CpuTsThread构造仅拷贝引用（不接管不销毁）
    hccl::LocalNotify** notifyObjs = reinterpret_cast<hccl::LocalNotify**>(notifys);
    std::shared_ptr<hccl::Thread> handle;
    EXCEPTION_CATCH(handle = std::make_shared<hccl::CpuTsThread>(stream, notifyObjs, notifyNum), return HCCL_E_PTR);
    handle->SetCommEngine(COMM_ENGINE_CPU_TS);
    CHK_RET(handle->Init());
    handle->SetIsMaster(true);
    handle->SetFakeDeviceRes();

    std::vector<std::shared_ptr<hccl::Thread>> newThreads{std::move(handle)};
    CHK_RET(
        newThreads[0]->AddThreadHandleToMap(COMM_ENGINE_CPU_TS, reinterpret_cast<ThreadHandle>(newThreads[0].get())));
    CHK_RET(hccl::SaveThreads(newThreads));
    *thread = reinterpret_cast<ThreadHandle>(newThreads[0].get());

    HCCL_INFO("[ThreadMgr] ThreadAcquireByNotify done: stream[%p], notifyNum[%u]", stream, notifyNum);
    return HCCL_SUCCESS;
}

HcommResult
HcommThreadSupplementNotify(const ThreadHandle* handles, uint32_t threadNum, const uint32_t* supplementNotifyNums)
{
    CHK_PTR_NULL(handles);
    CHK_PTR_NULL(supplementNotifyNums);
    HcommResult hcommRet = HcommResMgrInit();
    CHK_PRT_RET(
        hcommRet != HCCL_SUCCESS,
        HCCL_ERROR("[%s] HcommResMgrInit failed, ret[%d]", __func__, static_cast<int32_t>(hcommRet)), hcommRet);

    std::vector<std::shared_ptr<hccl::Thread>> needSupplementThread;
    std::unique_ptr<ThreadHandle[]> threadHandle;
    EXCEPTION_CATCH(threadHandle = std::make_unique<ThreadHandle[]>(threadNum), return HCOMM_E_PTR);

    uint32_t idx = 0;
    for (uint32_t i = 0; i < threadNum; ++i) {
        std::shared_ptr<hccl::Thread> threadPtr;
        CHK_RET(hccl::LookupThreadByHandle(handles[i], threadPtr));
        if (supplementNotifyNums[i] == 0) {
            HCCL_INFO(
                "[%s] thread[0x%llx] supplementNotifyNums[%u] is 0, skip supplement", __func__, handles[i],
                supplementNotifyNums[i]);
            continue;
        }
        CHK_RET(threadPtr->SupplementNotify(supplementNotifyNums[i]));
        needSupplementThread.push_back(std::move(threadPtr));
        threadHandle[idx++] = handles[i];
    }

    // 设备侧 kernel launch（仅 AICPU 引擎触发）
    if (!needSupplementThread.empty() && needSupplementThread[0]->GetCommEngine() == COMM_ENGINE_AICPU) {
        CHK_RET(HcommResMgr::EnsureKernelBinLoaded(COMM_ENGINE_AICPU));
        HcclResult ret = hccl::AicpuLaunchMgr::SupplementNotifyKernelLaunch(
            needSupplementThread, std::string(""), threadHandle, HcommResMgr::GetBinHandle());
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] SupplementNotifyKernelLaunch failed, ret[%d]", __func__, ret),
            static_cast<HcommResult>(ret));
    }
    return HCOMM_SUCCESS;
}

HcommResult HcommThreadGetNotifyNum(ThreadHandle thread, uint32_t* notifyNum)
{
    CHK_PTR_NULL(notifyNum);
    (void)HcommResMgrInit();
    std::shared_ptr<hccl::Thread> threadPtr;
    // 查不到说明线程已被释放（句柄失效），直接报错返回，不能再使用
    CHK_RET(hccl::LookupThreadByHandle(thread, threadPtr));
    CHK_PTR_NULL(threadPtr.get());
    *notifyNum = threadPtr->GetNotifyNum();
    HCCL_INFO("[%s] thread[0x%llx] notifyNum[%u]", __func__, thread, *notifyNum);
    return HCOMM_SUCCESS;
}

HcommResult HcommThreadResetNotifies(ThreadHandle thread)
{
    (void)HcommResMgrInit();
    std::shared_ptr<hccl::Thread> threadPtr;
    // 查不到说明线程及其 notify 已被释放，直接报错返回，不能再使用
    HcclResult ret = hccl::LookupThreadByHandle(thread, threadPtr);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] thread not found, thread[0x%llx], ret[%d]", __func__, thread, ret),
        static_cast<HcommResult>(ret));
    CHK_PTR_NULL(threadPtr.get());
    const uint32_t notifyNum = threadPtr->GetNotifyNum();
    for (uint32_t i = 0; i < notifyNum; ++i) {
        hccl::LocalNotify* notify = threadPtr->GetNotify(i);
        if (notify == nullptr || notify->ptr() == nullptr) {
            continue;
        }
        HcclResult rst = hrtNotifyReset(notify->ptr());
        CHK_PRT_RET(
            rst != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] hrtNotifyReset failed, thread[0x%llx], notifyIdx[%u], notifyId[%u], ret[%d]", __func__, thread, i,
                notify->notifyId_, rst),
            (HcommResult)rst);
        HCCL_INFO(
            "[%s] reset notify success, thread[0x%llx], notifyIdx[%u], notifyId[%u]", __func__, thread, i,
            notify->notifyId_);
    }
    return HCOMM_SUCCESS;
}

HcommResult HcommThreadExportToCommEngineAiCpu(
    const ThreadHandle* handles, const std::string& commIdStr, uint32_t threadNum, CommEngine dstEngine,
    ThreadHandle* outHandles)
{
    // 导出到 AICPU：先在线程内部映射表查，已有 AICPU 句柄直接用；
    // 没有的收集起来，批量下发 kernel 创建后再登记映射
    std::vector<std::shared_ptr<hccl::Thread>> hostThreads;
    std::vector<uint32_t> missIdx;
    for (uint32_t i = 0; i < threadNum; ++i) {
        std::shared_ptr<hccl::Thread> threadPtr;
        CHK_RET(hccl::LookupThreadByHandle(handles[i], threadPtr));
        hccl::Thread* exported = threadPtr->FindThreadByCommEngine(dstEngine);
        if (exported != nullptr) {
            outHandles[i] = ReinterpretAs<ThreadHandle>(exported);
        } else {
            hostThreads.push_back(std::move(threadPtr));
            missIdx.push_back(i);
        }
    }
    if (!hostThreads.empty()) {
        CHK_RET(HcommResMgr::EnsureKernelBinLoaded(dstEngine));
        std::unique_ptr<ThreadHandle[]> aicpuHandle;
        EXCEPTION_CATCH(aicpuHandle = std::make_unique<ThreadHandle[]>(hostThreads.size()), return HCOMM_E_PTR);
        HcclResult ret = hccl::AicpuLaunchMgr::ThreadKernelLaunchForComm(
            hostThreads, commIdStr, aicpuHandle, HcommResMgr::GetBinHandle());
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] ThreadKernelLaunchForComm failed, ret[%d]", __func__, ret),
            static_cast<HcommResult>(ret));
        for (size_t i = 0; i < hostThreads.size(); ++i) {
            outHandles[missIdx[i]] = aicpuHandle[i];
            CHK_RET(hostThreads[i]->AddThreadHandleToMap(dstEngine, aicpuHandle[i]));
            // 登记 device→host 句柄映射（g_ThreadD2HMap），供后续导出回 CPU/CCU 时反查
            ThreadHandle hostHandle = ReinterpretAs<ThreadHandle>(hostThreads[i].get());
            CHK_RET(hccl::FillThreadD2HMap(&aicpuHandle[i], &hostHandle, 1));
        }
    }

    return HCOMM_SUCCESS;
}

HcommResult HcommThreadExportToCommEngine(
    const ThreadHandle* handles, const char* commId, uint32_t threadNum, CommEngine dstEngine, ThreadHandle* outHandles)
{
    CHK_PTR_NULL(handles);
    CHK_PTR_NULL(outHandles);
    HcommResult hcommRet = HcommResMgrInit();
    CHK_PRT_RET(
        hcommRet != HCCL_SUCCESS,
        HCCL_ERROR("[%s] HcommResMgrInit failed, ret[%d]", __func__, static_cast<int32_t>(hcommRet)), hcommRet);
    CHK_RET(RefreshCommEngineContext(dstEngine));
    const std::string commIdStr = (commId != nullptr) ? std::string(commId) : std::string();
    switch (dstEngine) {
        case COMM_ENGINE_CPU:
        case COMM_ENGINE_CPU_TS:
        case COMM_ENGINE_CCU: {
            // 导出到 CPU/CCU：入参是 device 侧句柄，按 device→host 映射表换回 host 侧句柄
            for (uint32_t i = 0; i < threadNum; ++i) {
                CHK_RET(hccl::LookupD2HHandle(handles[i], outHandles[i]));
            }
            return HCOMM_SUCCESS;
        }
        case COMM_ENGINE_AICPU:
        case COMM_ENGINE_AICPU_TS: {
            CHK_RET(static_cast<HcclResult>(
                HcommThreadExportToCommEngineAiCpu(handles, commIdStr, threadNum, dstEngine, outHandles)));
            break;
        }
        default:
            HCCL_ERROR("[%s] unsupported dstEngine[%d]", __func__, static_cast<int32_t>(dstEngine));
            return HCOMM_E_PARA;
    }
    return HCOMM_SUCCESS;
}

HcommResult HcommThreadResGetInfo(ThreadHandle handle, ThreadResType resType, uint32_t infoLen, void** info)
{
    CHK_PTR_NULL(info);
    CHK_PRT_RET(handle == 0, HCCL_ERROR("[%s] thread is 0", __func__), HCOMM_E_PTR);
    CHK_PRT_RET(
        resType == ThreadResType::THREAD_RES_TYPE_INVALID,
        HCCL_ERROR("[%s] resType[%d] is invalid", __func__, static_cast<int32_t>(resType)), HCOMM_E_PARA);
    std::shared_ptr<hccl::Thread> threadPtr;
    CHK_RET(hccl::LookupThreadByHandle(handle, threadPtr));
    CHK_PTR_NULL(threadPtr.get());
    HCCL_INFO(
        "[%s] begin, thread[0x%llx], resType[%d], infoLen[%u]", __func__, threadPtr, static_cast<int32_t>(resType),
        infoLen);

    CHK_PRT_RET(
        infoLen != sizeof(ThreadResTypeStream),
        HCCL_ERROR(
            "[%s] infoLen[%u] mismatch sizeof(ThreadResTypeStream)[%zu]", __func__, infoLen,
            sizeof(ThreadResTypeStream)),
        HCOMM_E_PARA);

    hccl::Stream* streamPtr = threadPtr->GetStream();
    CHK_PTR_NULL(streamPtr);
    ThreadResTypeStream stream = streamPtr->ptr();
    CHK_PTR_NULL(stream);

    *info = stream;

    HCCL_INFO(
        "[%s] success, thread[0x%llx] resType[%d] stream[%p]", __func__, threadPtr, static_cast<int32_t>(resType),
        *info);
    return HCOMM_SUCCESS;
}
