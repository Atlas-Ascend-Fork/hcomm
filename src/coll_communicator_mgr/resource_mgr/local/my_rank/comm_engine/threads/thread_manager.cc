/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "thread_manager.h"
#include "hcomm/hcomm_res_expt.h"
#include "dpu_thread_service.h"
#include <algorithm>
#include <cstring>
#include "comm_engine/thread/thread_pub.h"
#include "dfx/res_pub.h"
#include "comm_engine/thread/thread_c_adpt_pub.h"
#include "independent_op.h"
#include "common/comm_engine_utils_pub.h"
#include "hcomm_res.h"
#include "comm_engine/aicpu_launch_manager_pub.h" // HCOMM_THREADNUM_MAX_NUM 常量
#include "dtype_common.h"
#include "adapter_rts_common.h"
#include "comm_engine/thread/dfx/thread_dfx_pub.h"
#include "dfx/prof_utils_pub.h"
#include <mutex>
#include <vector>
namespace hccl {

ThreadMgr::ThreadMgr(
    uint32_t threadNum, uint32_t notifyNumPerThread, std::string commId, aclrtBinHandle binHandle,
    const ManagerCallbacks& callbacks)
    : threadNum_(threadNum),
      notifyNumPerThread_(notifyNumPerThread),
      commId_(commId),
      dpuThreadService_(std::make_unique<DpuThreadService>(this, binHandle)),
      callbacks_(callbacks)
{}

ThreadMgr::~ThreadMgr()
{
    // 1. 释放专用线程（dedicatedThreadMap_ 全类型）。此前仅释放 AICPU_LAUNCH 一种，
    // ORDER_LAUNCH_DEVICE（deviceOrderThread）被遗漏——其线程与自建流只能活到进程退出
    // 静态析构（此时 ctx/流已被 runtime 回收，销毁必然失败并产生退出期错误风暴）
    FreeDedicatedThreads();

    // 线程对象由底层接口 HcommThreadFree 统一释放：这里若直接删，底层的全局线程表会残留句柄，造成泄漏
    // 2. 释放复用池线程（engineToThreadsMap_）
    FreeEngineToThreads();
    // 3. 释放主流线程（mainThread_）
    FreeMainThreads();

    HCCL_INFO("[~ThreadMgr] Hcom[%s] destroy done.", commId_.c_str());
}

void ThreadMgr::FreeDedicatedThreads() noexcept
{
    // dedicatedThreadMap_ 以 HcclDedicatedThreadType 枚举为 key，容量上界即枚举值个数：
    // 栈上固定数组收集句柄（零堆分配，规避析构路径上 vector 扩容抛 bad_alloc 的问题）
    constexpr u32 DEDICATED_THREAD_MAX_NUM = HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE + 1;
    ThreadHandle handles[DEDICATED_THREAD_MAX_NUM] = {0};
    u32 num = 0;
    {
        std::lock_guard<std::mutex> lock(dedicatedThreadMutex_);
        for (auto& pair : dedicatedThreadMap_) {
            if (pair.second != 0 && num < DEDICATED_THREAD_MAX_NUM) {
                handles[num++] = pair.second;
            }
        }
        dedicatedThreadMap_.clear();
    }
    if (num == 0) {
        return;
    }

    // HcommThreadFree 内部保证销毁时序：先经 ThreadKernelLaunchDestroy 拆除 device 侧线程
    // （移出守护线程轮询列表），再释放 host 侧线程与 SQ/CQ 资源，避免设备侧 UAF
    HcommResult ret = HcommThreadFree(handles, num);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[~ThreadMgr] dedicated threads free failed, Hcom[%s], num[%u], ret[%d]", commId_.c_str(), num, ret);
    }
}

void ThreadMgr::FreeEngineToThreads() noexcept
{
    std::lock_guard<std::mutex> lock(engineToThreadMutex_);
    for (auto& kv : engineToThreadsMap_) {
        for (auto& meta : kv.second) {
            if (meta.handle == 0) {
                continue;
            }
            ThreadHandle handle = meta.handle;
            HcommResult ret = HcommThreadFree(&handle, 1);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[~ThreadMgr] engineToThreads free failed, engine[%u] type[%d] handle[0x%llx] ret[%d]",
                    static_cast<uint32_t>(kv.first.first), static_cast<int32_t>(kv.first.second),
                    static_cast<uint64_t>(handle), ret);
            }
        }
        kv.second.clear();
    }
}

void ThreadMgr::FreeMainThreads() noexcept
{
    // 释放主流线程（mainThread_：stream 到 ThreadMeta 的映射，HcclThreadAcquireWithStream 分配）
    std::lock_guard<std::mutex> lock(mainThreadMutex_);
    for (auto& kv : mainThread_) {
        if (kv.second.handle == 0) {
            continue;
        }
        ThreadHandle handle = kv.second.handle;
        HcommResult ret = HcommThreadFree(&handle, 1);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[~ThreadMgr] mainThread free failed, handle[0x%llx], ret[%d]", static_cast<uint64_t>(handle), ret);
        }
    }
    mainThread_.clear();
}

// AICPU 首次触碰 device 前预置公共域初始化（独立锁串行化 check+launch+set，comm 生命周期内仅执行一次；
// 调用点分散在 threadMutex_ 与 dedicatedThreadMutex_ 两条路径，故用独立叶子锁防重，回调不回调本类无死锁风险）
HcclResult ThreadMgr::EnsureAicpuCommInit(CommEngine engine)
{
    if (engine != COMM_ENGINE_AICPU) {
        return HCCL_SUCCESS;
    }
    std::lock_guard<std::mutex> lock(aicpuCommInitMutex_);
    if (callbacks_.getAicpuCommState()) {
        return HCCL_SUCCESS;
    }
    HcclResult ret = callbacks_.kernelLaunchAicpuCommInit();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] kernelLaunchAicpuCommInit failed, ret[%d].", __func__, ret), ret);
    callbacks_.setAicpuCommState(true);
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclThreadAcquireV2(
    CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads,
    std::vector<uint32_t>& threadId)
{
    CHK_PTR_NULL(threads);
    CHK_PTR_NULL(config);
    if (threadNum == 0) {
        HCCL_ERROR("[ThreadMgr][HcclThreadAcquire] threadNum is 0");
        return HCCL_E_PARA;
    }

    for (u32 i = 0; i < threadNum; ++i) {
        CHK_PRT_RET(
            config[i].header.magicWord != HCOMM_THREAD_CONFIG_MAGIC_WORD,
            HCCL_ERROR(
                "[ThreadMgr][%s] config[%u] magicWord[0x%x] mismatch, expected[0x%x]", __func__, i,
                config[i].header.magicWord, HCOMM_THREAD_CONFIG_MAGIC_WORD),
            HCCL_E_PARA);
    }

    // AICPU 线程创建前置 device 侧公共域初始化（覆盖首扩容场景）
    CHK_RET(EnsureAicpuCommInit(engine));

    std::lock_guard<std::mutex> lock(threadMutex_);
    std::lock_guard<std::mutex> engineToThreadMtx(engineToThreadMutex_);
    HCCL_INFO(
        "[ThreadMgr][%s] Hcom[%s] HcclThreadAcquire begin, max: engine[%s] threadNum[%u],"
        "notifyPerThread[%u], need: threadNum[%u], threadType[%d]",
        __func__, commId_.c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum_,
        notifyNumPerThread_, threadNum, static_cast<int32_t>(type));

    // 1. 复用池已有线程：notify 充足则直接返回
    auto it = engineToThreadsMap_.find(std::make_pair(engine, type));
    if (it == engineToThreadsMap_.end()) {
        it = engineToThreadsMap_.emplace(std::make_pair(engine, type), std::vector<ThreadMeta>{}).first;
    }
    auto& threadVec = it->second;
    CHK_RET(SupplementNotify(threadVec, threadNum, config));

    // 2. 扩容：threadVec.size() 扩充到 threadNum
    CHK_RET(SupplementThread(engine, threadVec, type, threadNum, config));

    // 3. 返回 threadHandle + sqId
    for (u32 idx = 0; idx < threadNum; idx++) {
        threads[idx] = threadVec[idx].handle;
        uint32_t id = threadVec[idx].sqId;
        threadId.push_back(id);
    }

    HCCL_INFO(
        "[ThreadMgr][%s] Hcom[%s] HcclThreadAcquire done: engine[%s] threadNum[%u]%s", __func__, commId_.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum,
        (engine == COMM_ENGINE_AICPU) ? " (AICPU token ready)" : "");
    return HCCL_SUCCESS;
}

HcclResult
ThreadMgr::SupplementNotify(std::vector<ThreadMeta>& threadVec, uint32_t threadNum, const ThreadConfig* config)
{
    u32 existNum = std::min(static_cast<u32>(threadVec.size()), threadNum);
    if (existNum == 0) {
        return HCCL_SUCCESS;
    }

    // 1. 检查 notify 不足的thread
    std::vector<ThreadHandle> needSupplement(existNum);
    std::vector<uint32_t> supplementNums(existNum);
    u32 totalSupplement = 0;
    for (u32 i = 0; i < existNum; ++i) {
        needSupplement[i] = threadVec[i].handle;
        u32 cur = threadVec[i].notifyNum;
        u32 target = config[i].notifyNumPerThread;
        if (target > cur) {
            supplementNums[i] = target - cur;
            totalSupplement += (target - cur);
        } else {
            supplementNums[i] = 0;
        }
    }

    // 3. 批量补充
    HcommResult ret = HcommThreadSupplementNotify(needSupplement.data(), existNum, supplementNums.data());
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcommThreadSupplementNotify failed, ret[%d]", __func__, ret),
        static_cast<HcclResult>(ret));

    usedNotifyNum_ += totalSupplement;
    for (u32 i = 0; i < existNum; ++i) {
        // 缓存记录已分配容量：需求减小时底层不释放，保留原值，避免需求恢复时重复增配
        threadVec[i].notifyNum = std::max(threadVec[i].notifyNum, static_cast<uint32_t>(config[i].notifyNumPerThread));
    }

    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::SupplementThread(
    CommEngine engine, std::vector<ThreadMeta>& threadVec, ThreadType type, uint32_t threadNum,
    const ThreadConfig* config)
{
    // 1. 检查thread数量是否满足threadNum需求
    if (threadVec.size() >= threadNum) {
        return HCCL_SUCCESS;
    }

    // 2. 扩容：threadVec.size() 扩充到 threadNum
    u32 supplementThreadNum = threadNum - static_cast<u32>(threadVec.size());
    u32 offset = static_cast<u32>(threadVec.size());
    std::vector<ThreadHandle> newHandles(supplementThreadNum);
    HcommResult ret = HcommThreadAllocWithCommConfig(
        engine, commId_.c_str(), supplementThreadNum, type, &config[offset], newHandles.data());
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcommThreadAllocWithConfig failed, ret[%d]", __func__, ret),
        static_cast<HcclResult>(ret));

    // 3. 查询 stream/sqId（HOST CPU dpu 线程的流在 LegacyDpuThread::Init 内部才建，acquire 时尚未就绪，走空流分支）
    std::vector<ThreadMeta> metas;
    metas.reserve(supplementThreadNum);
    for (u32 i = 0; i < supplementThreadNum; ++i) {
        ThreadMeta meta;
        meta.handle = newHandles[i];
        meta.engine = engine;
        meta.type = type;
        meta.notifyNum = config[offset + i].notifyNumPerThread;
        if (engine == CommEngine::COMM_ENGINE_CPU && type == ThreadType::THREAD_TYPE_CPU) {
            // 1 thread 1 kernel 的 dpu 线程不参与 TS stream/notify 机制，stream/sqId 恒为空
            meta.stream = nullptr;
            meta.sqId = 0;
        } else {
            ThreadResTypeStream stream = nullptr;
            uint32_t size = sizeof(ThreadResTypeStream);
            HcclResult infoRet
                = HcclThreadResGetInfo(newHandles[i], THREAD_RES_TYPE_STREAM, size, reinterpret_cast<void**>(&stream));
            if (infoRet != HCCL_SUCCESS || stream == nullptr) {
                HCCL_ERROR(
                    "[ThreadMgr][%s] HcclThreadResGetInfo failed, ret[%d], stream[%p]", __func__, infoRet, stream);
                // 回滚本批已分配但未入池的句柄，避免泄漏
                (void)HcommThreadFree(newHandles.data(), supplementThreadNum);
                return (infoRet != HCCL_SUCCESS) ? infoRet : HCCL_E_INTERNAL;
            }
            meta.stream = stream;
            uint32_t sqId = 0;
            if (hrtStreamGetSqid(meta.stream, &sqId) != HCCL_SUCCESS) {
                HCCL_ERROR("[ThreadMgr][%s] hrtStreamGetSqid failed stream[%p]", __func__, meta.stream);
                // 回滚本批已分配但未入池的句柄，避免泄漏
                (void)HcommThreadFree(newHandles.data(), supplementThreadNum);
                return HCCL_E_INTERNAL;
            }
            meta.sqId = sqId;
        }
        metas.emplace_back(std::move(meta));
    }

    // 全部查询成功：统一入池
    threadVec.insert(threadVec.end(), metas.begin(), metas.end());
    totalThreadCount_ += supplementThreadNum;
    for (u32 i = 0; i < supplementThreadNum; ++i) {
        usedNotifyNum_ += config[offset + i].notifyNumPerThread;
    }

    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclThreadAcquire(
    CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads,
    std::vector<uint32_t>& threadId)
{
    // 旧接口直接转调 V2：V2 用 engineToThreadsMap_ 复用池管理线程，能复用已有线程，减少重复创建
    return HcclThreadAcquireV2(engine, threadNum, type, config, threads, threadId);
}

HcclResult ThreadMgr::HcclGetNotifyNumInThread(ThreadHandle thread, uint32_t* notifyNum)
{
    CHK_PTR_NULL(notifyNum);
    HcommResult ret = HcommThreadGetNotifyNum(thread, notifyNum);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[ThreadMgr][%s] HcommThreadGetNotifyNum failed, ret[%d]", __func__, ret),
        static_cast<HcclResult>(ret));
    HCCL_INFO("[ThreadMgr] Hcom[%s] HcclGetNotifyNumInThread done: notifyPerThread[%u]", commId_.c_str(), *notifyNum);
    return HCCL_SUCCESS;
}

HcclResult
ThreadMgr::HcclThreadAcquireWithStream(CommEngine engine, rtStream_t stream, uint32_t notifyNum, ThreadHandle* thread)
{
    CHK_PTR_NULL(thread);

    std::lock_guard<std::mutex> lock(mainThreadMutex_);
    auto it = mainThread_.find(stream);
    if (it != mainThread_.end()) {
        // 1. 复用已有线程：notify 不足则补充，传入增量值
        if (it->second.notifyNum < notifyNum) {
            u32 supplementNum = notifyNum - it->second.notifyNum;
            ThreadHandle handle = it->second.handle;
            HcommResult ret = HcommThreadSupplementNotify(&handle, 1, &supplementNum);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcommThreadSupplementNotify failed, ret[%d]", __func__, ret),
                static_cast<HcclResult>(ret));
            it->second.notifyNum = notifyNum;
        }
        *thread = it->second.handle;
        return HCCL_SUCCESS;
    }

    // 2. 无复用线程：新申请
    HcommResult ret = HcommThreadAllocWithStream(engine, stream, notifyNum, thread);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcommThreadAllocWithStream failed, ret[%d]", __func__, ret),
        static_cast<HcclResult>(ret));
    ThreadMeta meta;
    meta.handle = *thread;
    meta.engine = engine;
    meta.notifyNum = notifyNum;
    meta.stream = stream;
    mainThread_.emplace(stream, meta);

    HCCL_INFO(
        "[ThreadMgr] Hcom[%s] HcclThreadAcquireWithStream done: engine[%s] stream[%p],"
        "notifyNum[%u]",
        commId_.c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), stream, notifyNum);
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclThreadExportToCommEngine(
    uint32_t threadNum, const ThreadHandle* threads, CommEngine dstCommEngine, ThreadHandle* exportedThreads)
{
    CHK_PTR_NULL(threads);
    CHK_PTR_NULL(exportedThreads);
    // 导出线程前需预调 commInit（由 aicpuCommState 标志控制）
    u64 beginTime = 0;
    if (dstCommEngine == COMM_ENGINE_AICPU || dstCommEngine == COMM_ENGINE_AICPU_TS) {
        if (!callbacks_.getAicpuCommState()) {
            HcclResult ret = callbacks_.kernelLaunchAicpuCommInit();
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[%s] kernelLaunchAicpuCommInit failed, return [%d].", __func__, ret),
                ret);
            callbacks_.setAicpuCommState(true);
        }
        beginTime = hcomm::GetProfCycleTime();
    }

    // 导出线程
    HcommResult ret
        = HcommThreadExportToCommEngine(threads, commId_.c_str(), threadNum, dstCommEngine, exportedThreads);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[ThreadMgr][%s] HcommThreadExportToCommEngine failed, ret[%d]", __func__, ret),
        static_cast<HcclResult>(ret));

    if ((dstCommEngine == COMM_ENGINE_AICPU || dstCommEngine == COMM_ENGINE_AICPU_TS)
        && callbacks_.reportProfilingKernel != nullptr) {
        HcclResult profRet = callbacks_.reportProfilingKernel(beginTime, "RunAicpuIndOpThreadInit");
        CHK_PRT_RET(
            profRet != HCCL_SUCCESS,
            HCCL_ERROR("[ThreadMgr][%s] ReportProfilingAiCpukernelLaunch failed, return [%d].", __func__, profRet),
            profRet);
    }

    return HCCL_SUCCESS;
}

HcclResult
ThreadMgr::HcclThreadResGetInfo(ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void** info) const
{
    CHK_PRT_RET(
        resType != ThreadResType::THREAD_RES_TYPE_STREAM,
        HCCL_ERROR("[%s] failed. resType[%d] is not supported.", __func__, static_cast<int32_t>(resType)),
        HCCL_E_NOT_SUPPORT);

    HcommResult ret = HcommThreadResGetInfo(thread, resType, infoLen, info);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcommThreadGetStreamInfo failed, ret[%d]", __func__, ret),
        static_cast<HcclResult>(ret));
    HCCL_INFO(
        "[%s] success. thread[0x%llx] resType[%d] info[%p]", __func__, thread, static_cast<int32_t>(resType), *info);
    return HCCL_SUCCESS;
}

HcclResult
ThreadMgr::HcclUnfoldThreadAcquire(HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread)
{
    CHK_PRT_RET(thread == nullptr, HCCL_ERROR("[%s] thread is null", __func__), HCCL_E_PTR);
    auto it = dedicatedThreadMap_.find(useType);
    if (it != dedicatedThreadMap_.end()) {
        *thread = it->second;
        HCCL_INFO(
            "[%s] reuse dedicated thread, HcclDedicatedThreadType[%u], thread[0x%llx]", __func__, useType, *thread);
        // 补充 notify（notifyNumPerThread 为目标值，转为增量传入）
        ThreadHandle handle = *thread;
        uint32_t currentNotifyNum = 0;
        HcommResult ret = HcommThreadGetNotifyNum(handle, &currentNotifyNum);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcommThreadGetNotifyNum failed, ret[%d]", __func__, ret),
            static_cast<HcclResult>(ret));
        if (notifyNumPerThread > currentNotifyNum) {
            uint32_t addNotifyNum = notifyNumPerThread - currentNotifyNum;
            ret = HcommThreadSupplementNotify(&handle, 1, &addNotifyNum);
        }
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcommThreadSupplementNotify failed, ret[%d]", __func__, ret),
            static_cast<HcclResult>(ret));
    } else {
        if (useType == HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE) {
            CHK_RET(HcclGeUnfoldThreadAcquire(useType, notifyNumPerThread, thread));
            return HCCL_SUCCESS;
        }
        // DFX回调必须先经HcclDedicatedThreadDfx记录（per-comm恒定），未记录按致命处理。
        // GE分支在HcclGeUnfoldThreadAcquire内注册no-op
        if (dfxCallback_ == nullptr) {
            HCCL_ERROR("[%s] dfxCallback not recorded, record via HcclDedicatedThreadDfx first", __func__);
            return HCCL_E_INTERNAL;
        }
        CommEngine engine = CommEngine::COMM_ENGINE_CPU;
        uint32_t notifyNumPerThreadVec[1] = {notifyNumPerThread};
        HcclResult ret = static_cast<HcclResult>(HcommThreadAlloc(engine, 1, notifyNumPerThreadVec, thread));
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[%s] Failed to cache dedicated thread, HcclDedicatedThreadType[%u], ret[%d]", __func__, useType, ret);
            return ret;
        }
        dedicatedThreadMap_[useType] = *thread;
        // 注册失败按致命处理
        int dfxRet = HcommThreadRegisterDfx(*thread, dfxCallback_);
        CHK_PRT_RET(
            dfxRet != HCCL_SUCCESS, HCCL_ERROR("[%s] HcommThreadRegisterDfx failed, ret:[%d]", __func__, dfxRet),
            HCCL_E_INTERNAL);
        HCCL_RUN_INFO(
            "[ThreadMgr][%s] created opbase unfold thread, HcclDedicatedThreadType[%u], "
            "thread[0x%llx], notifyNumPerThread[%u], comm[%s]",
            __func__, useType, *thread, notifyNumPerThread, commId_.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::SetAttachedStream(rtStream_t stream)
{
    CHK_PRT_RET(
        stream == nullptr, HCCL_ERROR("[%s] stream is nullptr, Hcom[%s]", __func__, commId_.c_str()), HCCL_E_PTR);
    std::lock_guard<std::mutex> lock(attachedStreamMutex_);
    attachedStream_ = stream;
    HCCL_INFO(
        "[ThreadMgr][%s] Hcom[%s] attached stream[%p] stored, thread will be created on first GE acquire", __func__,
        commId_.c_str(), stream);
    return HCCL_SUCCESS;
}

HcclResult
ThreadMgr::HcclGeUnfoldThreadAcquire(HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread)
{
    *thread = 0;
    rtStream_t stream = nullptr;
    {
        std::lock_guard<std::mutex> lock(attachedStreamMutex_);
        stream = attachedStream_;
    }
    if (stream == nullptr) {
        HCCL_WARNING(
            "[%s] dedicated thread not found, HcclDedicatedThreadType[%u], attached stream not set, "
            "return threadHandle[0]",
            __func__, useType);
        return HCCL_SUCCESS;
    }

    std::lock_guard<std::mutex> lock(mainThreadMutex_);
    auto it = mainThread_.find(stream);
    if (it != mainThread_.end()) {
        // 1. 复用已有线程：notify 不足则经 L0 接口补充
        if (it->second.notifyNum < notifyNumPerThread) {
            u32 supplementNum = notifyNumPerThread - it->second.notifyNum;
            ThreadHandle handle = it->second.handle;
            HcommResult ret = HcommThreadSupplementNotify(&handle, 1, &supplementNum);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcommThreadSupplementNotify failed, ret[%d]", __func__, ret),
                static_cast<HcclResult>(ret));
            it->second.notifyNum = notifyNumPerThread;
        }
        *thread = it->second.handle;
        HCCL_INFO(
            "[ThreadMgr][%s] reuse GE dedicated thread[0x%llx] by attached stream[%p], Hcom[%s]", __func__, *thread,
            stream, commId_.c_str());
        return HCCL_SUCCESS;
    }

    // 2. 无复用线程：经 L0 接口新申请并登记 ThreadMeta
    HcommResult ret = HcommThreadAllocWithStream(CommEngine::COMM_ENGINE_CPU_TS, stream, notifyNumPerThread, thread);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcommThreadAllocWithStream failed, ret[%d]", __func__, ret),
        static_cast<HcclResult>(ret));
    ThreadMeta meta;
    meta.handle = *thread;
    meta.engine = CommEngine::COMM_ENGINE_CPU_TS;
    meta.notifyNum = notifyNumPerThread;
    meta.stream = stream;
    mainThread_.emplace(stream, meta);

    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> dfxCallback
        = [](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle) {
              (void)streamId;
              (void)taskId;
              (void)taskParam;
              (void)handle;
              return HCCL_SUCCESS;
          };
    int dfxRet = HcommThreadRegisterDfx(*thread, dfxCallback);
    if (dfxRet != 0) {
        HCCL_WARNING("[%s] HcommThreadRegisterDfx failed, ret[%d], thread[0x%llx]", __func__, dfxRet, *thread);
    }

    // 展开流建链审计日志：通信域/展开流/线程/notify数（流ID查询失败保持INVALID便于识别）
    s32 unfoldStreamId = INVALID_INT;
    (void)hrtGetStreamId(stream, unfoldStreamId);
    HCCL_RUN_INFO(
        "[ThreadMgr][%s] created GE dedicated thread by unfold stream: commId[%s], unfoldStream[%p](id[%d]), "
        "thread[0x%llx], notifyNumPerThread[%u]",
        __func__, commId_.c_str(), stream, unfoldStreamId, *thread, notifyNumPerThread);
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclDeviceOrderThreadCreate(
    HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread)
{
    CHK_PRT_RET(thread == nullptr, HCCL_ERROR("[%s] thread is null", __func__), HCCL_E_PTR);
    auto it = dedicatedThreadMap_.find(useType);
    if (it != dedicatedThreadMap_.end() && it->second != 0) {
        *thread = it->second;
        HCCL_INFO("[%s] reuse device order thread[0x%llx], comm[%s]", __func__, *thread, commId_.c_str());
        return HCCL_SUCCESS;
    }

    ThreadConfig config;
    HcommResult initRet = ThreadConfigInit(&config, 1);
    CHK_PRT_RET(initRet != 0, HCCL_ERROR("[%s] ThreadConfigInit failed, ret[%d]", __func__, initRet), HCCL_E_INTERNAL);
    config.notifyNumPerThread = static_cast<uint16_t>(notifyNumPerThread);

    // AICPU 专用线程创建前置 device 侧公共域初始化
    CHK_RET(EnsureAicpuCommInit(CommEngine::COMM_ENGINE_AICPU));

    // AICPU 专用线程统一走底层 C 接口创建，且必须带 commId：带 commId 时线程会绑定通信域（ForComm 路径）。
    // 不带 commId 的 HcommThreadAlloc 走 ForBase 路径，创建的线程不绑定通信域，这里不能用
    HcclResult ret = static_cast<HcclResult>(HcommThreadAllocWithCommConfig(
        CommEngine::COMM_ENGINE_AICPU, commId_.c_str(), 1, THREAD_TYPE_TS, &config, thread));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcommThreadAllocWithCommConfig failed, ret[%d]", __func__, ret), ret);

    dedicatedThreadMap_[useType] = *thread;

    HCCL_INFO("[%s] created device order thread[0x%llx], comm[%s]", __func__, *thread, commId_.c_str());
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclDedicatedThreadAcquire(
    HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread)
{
    CHK_PRT_RET(thread == nullptr, HCCL_ERROR("[%s] thread is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(
        useType == HCCL_DED_THREAD_TYPE_INVALID, HCCL_ERROR("[%s] HcclDedicatedThreadType is invalid", __func__),
        HCCL_E_PARA);
    HCCL_INFO("Entry-%s: HcclDedicatedThreadType[%u] notifyNumPerThread[%u]", __func__, useType, notifyNumPerThread);

    std::lock_guard<std::mutex> lock(dedicatedThreadMutex_);
    if (useType == HCCL_DED_THREAD_TYPE_AICPU_LAUNCH || useType == HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE) {
        CHK_RET(HcclUnfoldThreadAcquire(useType, notifyNumPerThread, thread));
    } else if (useType == HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE) {
        CHK_RET(HcclDeviceOrderThreadCreate(useType, notifyNumPerThread, thread));
    } else {
        HCCL_ERROR("[%s] unsupported HcclDedicatedThreadType[%u]", __func__, useType);
        return HCCL_E_NOT_SUPPORT;
    }

    HCCL_INFO(
        "[%s] success, HcclDedicatedThreadType[%u], thread[0x%llx], notifyNumPerThread[%u]", __func__, useType, *thread,
        notifyNumPerThread);
    return HCCL_SUCCESS;
}

HcclResult
ThreadMgr::HcclDedicatedThreadDfx(std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> dfxCallback)
{
    std::lock_guard<std::mutex> lock(dedicatedThreadMutex_);
    if (dfxCallback_ == nullptr && dfxCallback != nullptr) {
        dfxCallback_ = dfxCallback; // 仅首次记录非空回调，per-comm恒定不覆盖
    }
    return HCCL_SUCCESS;
}

// Reset 家族：L1 只持有 ThreadHandle，notify reset 通过 L0 C 接口 HcommThreadResetNotifies 完成，
HcclResult ThreadMgr::ResetThreadPoolLocalNotifies()
{
    std::lock_guard<std::mutex> lock(engineToThreadMutex_);
    for (auto& kv : engineToThreadsMap_) {
        for (auto& meta : kv.second) {
            if (meta.handle == 0) {
                continue;
            }
            HcclResult ret = static_cast<HcclResult>(HcommThreadResetNotifies(meta.handle));
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[ThreadMgr][ResetThreadPoolLocalNotifies] reset notifies failed, Hcom[%s], thread[0x%llx], "
                    "ret[0x%016llx]",
                    commId_.c_str(), static_cast<uint64_t>(meta.handle), HCCL_ERROR_CODE(ret)),
                ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::ResetMainThreadLocalNotifies()
{
    std::lock_guard<std::mutex> lock(mainThreadMutex_);
    for (auto& pair : mainThread_) {
        if (pair.second.handle == 0) {
            continue;
        }
        HcclResult ret = static_cast<HcclResult>(HcommThreadResetNotifies(pair.second.handle));
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ThreadMgr][ResetMainThreadLocalNotifies] reset notifies failed, Hcom[%s], stream[%p], "
                "thread[0x%llx], ret[0x%016llx]",
                commId_.c_str(), pair.first, static_cast<uint64_t>(pair.second.handle), HCCL_ERROR_CODE(ret)),
            ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::ResetDedicatedThreadLocalNotifies()
{
    std::lock_guard<std::mutex> lock(dedicatedThreadMutex_);
    for (auto& pair : dedicatedThreadMap_) {
        // device侧保序流不会保存在g_ThreadMap，不在此处清理，在ResetThreadPoolLocalNotifiles清
        if (pair.first == HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE) {
            continue;
        }
        if (pair.second == 0) {
            continue;
        }
        HcclResult ret = static_cast<HcclResult>(HcommThreadResetNotifies(pair.second));
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ThreadMgr][ResetDedicatedThreadLocalNotifies] reset notifies failed, Hcom[%s], "
                "HcclDedicatedThreadType[%u], thread[0x%llx], ret[0x%016llx]",
                commId_.c_str(), static_cast<uint32_t>(pair.first), static_cast<uint64_t>(pair.second),
                HCCL_ERROR_CODE(ret)),
            ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::ResetOrderLaunchThreadLocalNotifies()
{
    for (ThreadHandle handle : orderLaunchThreads_) {
        if (handle == 0) {
            continue;
        }
        HcclResult ret = static_cast<HcclResult>(HcommThreadResetNotifies(handle));
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ThreadMgr][ResetOrderLaunchThreadLocalNotifies] reset notifies failed, Hcom[%s], thread[0x%llx], "
                "ret[0x%016llx]",
                commId_.c_str(), static_cast<uint64_t>(handle), HCCL_ERROR_CODE(ret)),
            ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::ResetThreadLocalNotifies()
{
    // pool / main / dedicated / orderLaunch 可能指向同一 Thread（orderLaunch 常是已 Acquire
    // 的 handle 再 Register）。同一 LocalNotify 可能对 RTS 多次 hrtNotifyReset；
    // hrtNotifyReset → aclrtNotifyBatchReset 清零触发态，重复调用视为幂等，可接受。
    HCCL_INFO("[ThreadMgr][ResetThreadLocalNotifies] start, Hcom[%s]", commId_.c_str());
    CHK_RET(ResetThreadPoolLocalNotifies());
    CHK_RET(ResetMainThreadLocalNotifies());
    CHK_RET(ResetDedicatedThreadLocalNotifies());
    CHK_RET(ResetOrderLaunchThreadLocalNotifies());
    HCCL_INFO("[ThreadMgr][ResetThreadLocalNotifies] finish, Hcom[%s]", commId_.c_str());
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::RegisterOrderLaunchThread(ThreadHandle thread)
{
    orderLaunchThreads_.insert(thread);
    HCCL_INFO(
        "[ThreadMgr][%s] register order launch thread[0x%llx], comm[%s], total[%zu]", __func__, thread, commId_.c_str(),
        orderLaunchThreads_.size());
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::GetCpuThreadHandle(ThreadHandle& handle) const
{
    handle = 0;
    std::lock_guard<std::mutex> lock(engineToThreadMutex_);
    auto it = engineToThreadsMap_.find(std::make_pair(CommEngine::COMM_ENGINE_CPU, ThreadType::THREAD_TYPE_CPU));
    if (it == engineToThreadsMap_.end() || it->second.empty()) {
        HCCL_WARNING("[ThreadMgr][%s] no host cpu thread registered, comm[%s]", __func__, commId_.c_str());
        return HCCL_E_NOT_FOUND;
    }
    // 该键下恒为唯一一条（1 thread 1 kernel）；多条目时 [0] 未必是 dpu 线程，
    // 静默取首条会把 DPU 动作落到别的 CpuThread 上，故留痕
    if (it->second.size() > 1) {
        HCCL_WARNING(
            "[ThreadMgr][%s] %zu threads under (CPU, CPU) key, take the first, comm[%s]", __func__, it->second.size(),
            commId_.c_str());
    }
    handle = it->second[0].handle;
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::RollbackThread(ThreadHandle handle)
{
    if (handle == 0) {
        HCCL_ERROR("[ThreadMgr][%s] handle is invalid, comm[%s]", __func__, commId_.c_str());
        return HCCL_E_PARA;
    }

    ThreadMeta meta;
    {
        // 加锁次序与 HcclThreadAcquireV2 一致（threadMutex_ → engineToThreadMutex_），避免反向嵌套
        std::lock_guard<std::mutex> lock(threadMutex_);
        std::lock_guard<std::mutex> engineToThreadMtx(engineToThreadMutex_);
        bool found = false;
        for (auto& kv : engineToThreadsMap_) {
            auto& threadVec = kv.second;
            for (auto it = threadVec.begin(); it != threadVec.end(); ++it) {
                if (it->handle != handle) {
                    continue;
                }
                meta = *it;
                threadVec.erase(it);
                found = true;
                break;
            }
            if (found) {
                break;
            }
        }
        if (!found) {
            // 句柄不在复用池：非本池所有，不代为释放，避免误释放他人线程
            HCCL_ERROR("[ThreadMgr][%s] thread[0x%llx] not in pool, comm[%s]", __func__, handle, commId_.c_str());
            return HCCL_E_NOT_FOUND;
        }

        // 与 SupplementThread 的入池记账严格对冲：否则配额只减不还，反复失败后 acquire 被判配额耗尽
        totalThreadCount_ = (totalThreadCount_ > 0) ? (totalThreadCount_ - 1) : 0;
        usedNotifyNum_ = (usedNotifyNum_ > meta.notifyNum) ? (usedNotifyNum_ - meta.notifyNum) : 0;
    }

    // 释放放在锁外：缩小锁窗口，避免运行期持锁执行 device 侧销毁等慢操作阻塞 Acquire
    HcommResult ret = HcommThreadFree(&handle, 1);
    HCCL_INFO(
        "[ThreadMgr][%s] rollback thread[0x%llx] done, comm[%s], ret[%d]", __func__, handle, commId_.c_str(),
        static_cast<int32_t>(ret));
    return (ret == HCCL_SUCCESS) ? HCCL_SUCCESS : static_cast<HcclResult>(ret);
}

} // namespace hccl
