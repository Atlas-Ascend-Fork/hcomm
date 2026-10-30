/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <unordered_set>
#include "thread.h"
#include "dfx/res_pub.h"
#include "cpu_ts_thread.h"
#include "aicpu_ts_thread.h"
#ifndef CCL_KERNEL_AICPU
#include "cpu_thread.h"
#endif
#include "comm_engine/thread/device/device_thread_pub.h"
#include "common/result_defs_pub.h"
#include "hcomm_res_defs.h"
#include "exception_handler.h"
#include <mutex>
#include <unordered_map>
#include "sal_pub.h"
#include "stream_lite.h"
#include "task_info.h"
#include "common/comm_engine_utils_pub.h"
#include "comm_engine/aicpu_launch_manager_pub.h"
#include "aicpu_thread_launch_mgr.h"
#include "dfx/device/dfx_profiling_handler_lite_pub.h"
#include "dfx/prof_utils_pub.h"
#ifdef CCL_KERNEL_AICPU
#include "common/device/aicpu_indop_env_pub.h"
#include "rtsq_a5.h"
#include "aicpu_task_cache_manager.h"
#endif
#include "adapter_rts_common.h"
#ifndef CCL_KERNEL_AICPU
#include "base_comm_node_accessors.h"
#include "handle_registry_procs.h"
#endif

using namespace std;

namespace hccl {

HcclResult CreateThread(
    CommEngine engine, [[maybe_unused]] ThreadType threadType, StreamType streamType, uint32_t notifyNum,
    NotifyLoadType loadType, shared_ptr<Thread>& out_thread)
{
    out_thread = nullptr; // 初始化出参

    // HOST CPU 线程（HostDpu kernel，1 thread 1 kernel）：不参与 TS 的 stream/notify 机制，
    // 只持有 dpu 流与整段 XPU 切换序列，故独立于 CpuTsThread 单独分派
#ifndef CCL_KERNEL_AICPU
    if (engine == COMM_ENGINE_CPU && threadType == THREAD_TYPE_CPU) {
        EXCEPTION_CATCH(out_thread = make_shared<CpuThread>(), return HCCL_E_PTR);
    } else
#endif
        if (engine == COMM_ENGINE_CPU_TS || engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CCU
            || engine == COMM_ENGINE_AIV) {
        EXCEPTION_CATCH(out_thread = make_shared<CpuTsThread>(streamType, notifyNum, loadType), return HCCL_E_PTR);
    } else if (engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_AICPU) {
        EXCEPTION_CATCH(out_thread = make_shared<AicpuTsThread>(streamType, notifyNum, loadType), return HCCL_E_PTR);
    } else {
        return HCCL_E_NOT_SUPPORT;
    }

    out_thread->SetCommEngine(engine);
    return HCCL_SUCCESS;
}

HcclResult CommHostEngineToNotifyLoadType(CommEngine engine, NotifyLoadType& type)
{
    switch (engine) {
        case COMM_ENGINE_CPU:
        case COMM_ENGINE_CPU_TS:
        case COMM_ENGINE_CCU:
            type = NotifyLoadType::HOST_NOTIFY;
            break;
        default:
            HCCL_ERROR(
                "[ThreadMgr] Unsupported comm engine type: %s",
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CommEngineToNotifyLoadType(CommEngine engine, NotifyLoadType& type)
{
    switch (engine) {
        case COMM_ENGINE_CPU:
        case COMM_ENGINE_CPU_TS:
        case COMM_ENGINE_CCU:
            type = NotifyLoadType::HOST_NOTIFY;
            break;
        case COMM_ENGINE_AICPU:
        case COMM_ENGINE_AICPU_TS:
            type = NotifyLoadType::HOST_NOTIFY;
            break;
        default:
            HCCL_ERROR(
                "[ThreadMgr] Unknown comm engine type: %s",
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CommEngineToStreamType(CommEngine engine, StreamType& type)
{
    switch (engine) {
        case COMM_ENGINE_CPU:
        case COMM_ENGINE_CPU_TS:
        case COMM_ENGINE_CCU:
            type = StreamType::STREAM_TYPE_ONLINE; // 单算子使用online，图模式使用offline
            break;
        case COMM_ENGINE_AICPU:
        case COMM_ENGINE_AICPU_TS:
            type = StreamType::STREAM_TYPE_DEVICE;
            break;
        // 暂不支持AIV
        case COMM_ENGINE_AIV:
        default:
            HCCL_ERROR(
                "[ThreadMgr] Unknown comm engine type: %s",
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

struct EnumPairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const
    {
        const std::size_t h1 = std::hash<T1>{}(p.first);
        const std::size_t h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

const std::unordered_map<std::pair<CommEngine, ThreadType>, NotifyLoadType, EnumPairHash> NOTIFY_TYPE_CONVERT = {
    {{COMM_ENGINE_CPU, THREAD_TYPE_TS}, NotifyLoadType::HOST_NOTIFY},
    {{COMM_ENGINE_CPU, THREAD_TYPE_CPU}, NotifyLoadType::HOST_NOTIFY},
    {{COMM_ENGINE_CCU, THREAD_TYPE_TS}, NotifyLoadType::HOST_NOTIFY},
    {{COMM_ENGINE_AIV, THREAD_TYPE_TS}, NotifyLoadType::HOST_NOTIFY},
    {{COMM_ENGINE_AICPU, THREAD_TYPE_TS}, NotifyLoadType::HOST_NOTIFY},
};

const std::unordered_map<std::pair<CommEngine, ThreadType>, StreamType, EnumPairHash> STREAM_TYPE_CONVERT = {
    {{COMM_ENGINE_CPU, THREAD_TYPE_TS}, StreamType::STREAM_TYPE_ONLINE},
    {{COMM_ENGINE_CPU, THREAD_TYPE_CPU}, StreamType::STREAM_TYPE_ONLINE},
    {{COMM_ENGINE_CCU, THREAD_TYPE_TS}, StreamType::STREAM_TYPE_ONLINE},
    {{COMM_ENGINE_AIV, THREAD_TYPE_TS}, StreamType::STREAM_TYPE_ONLINE},
    {{COMM_ENGINE_AICPU, THREAD_TYPE_TS}, StreamType::STREAM_TYPE_DEVICE},
};

HcclResult GetNotifyLoadType(CommEngine engine, ThreadType threadType, NotifyLoadType& type)
{
    auto iter = NOTIFY_TYPE_CONVERT.find(std::make_pair(engine, threadType));
    if (iter == NOTIFY_TYPE_CONVERT.end()) {
        HCCL_ERROR("[GetNotifyLoadType] not support comm engine type: %d, thread type: %d", engine, threadType);
        return HCCL_E_PARA;
    }
    type = iter->second;
    return HCCL_SUCCESS;
}

HcclResult GetStreamType(CommEngine engine, ThreadType threadType, StreamType& type)
{
    auto iter = STREAM_TYPE_CONVERT.find(std::make_pair(engine, threadType));
    if (iter == STREAM_TYPE_CONVERT.end()) {
        HCCL_ERROR("[GetStreamType] not support comm engine type: %d, thread type: %d", engine, threadType);
        return HCCL_E_PARA;
    }
    type = iter->second;
    return HCCL_SUCCESS;
}

#ifndef CCL_KERNEL_AICPU
using hcomm::DeviceThreadKey;

static hcomm::ThreadHandleRegistryProc& ThreadRegistry() { return hcomm::GetThreadHandleRegistryProc(); }

HcclResult Thread::PreAllocThreadEntity(ThreadHandle& handle, CommEngine targetEngine)
{
    CHK_PRT_RET(
        targetEngine != COMM_ENGINE_AICPU && targetEngine != COMM_ENGINE_AICPU_TS,
        HCCL_ERROR("[%s] unsupported target engine[%d]", __func__, targetEngine), HCCL_E_PARA);
    auto iter = threadHandleMap_.find(targetEngine);
    if (iter == threadHandleMap_.end()) {
        EXCEPTION_CATCH(iter = threadHandleMap_.emplace(targetEngine, ThreadHandleEntry{}).first, return HCCL_E_MEMORY);
    }
    if (!iter->second.entityResources) {
        std::unique_ptr<ThreadEntityResources> resources;
        EXCEPTION_CATCH(resources = std::make_unique<ThreadEntityResources>(), return HCCL_E_MEMORY);
        HcommDevInfo devInfo{};
        auto isGe950 = hcomm::EqualToOrGreaterThan950();
        CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
        const bool initPodFields = *isGe950;
        if (initPodFields) {
            int32_t logicId = 0;
            CHK_RET(hrtGetDevice(&logicId));
            CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(logicId), devInfo.devPhyId));
        }
        CHK_RET(resources->Init(devInfo, GetNotifyNum(), initPodFields));
        iter->second.entityResources = std::move(resources);
    } else if (iter->second.handle == 0) {
        // 预分配后下发失败的 entity 尚未发布，重试前按当前 Host Notify 数刷新容量。
        // 已发布 entity 的扩容由 SupplementNotify 流程处理，避免设备侧使用期间替换 Notify 数组。
        CHK_RET(iter->second.entityResources->UpdateNotifyBuffer(GetNotifyNum()));
    }
    handle = iter->second.entityResources->GetEntityHandle();
    return HCCL_SUCCESS;
}

HcclResult Thread::ConvertThreadEntityToRawHandle(ThreadHandle& handle, CommEngine targetEngine)
{
    if (!IsFakeDeviceRes()) {
        return HCCL_SUCCESS;
    }
    auto iter = threadHandleMap_.find(targetEngine);
    CHK_PRT_RET(
        iter == threadHandleMap_.end() || !iter->second.entityResources,
        HCCL_ERROR("[%s] temporary thread entity not found, engine[%d]", __func__, targetEngine), HCCL_E_INTERNAL);
    CHK_PRT_RET(
        iter->second.handle != 0 || iter->second.entityResources->GetEntityHandle() != handle,
        HCCL_ERROR("[%s] cannot convert a published or mismatched thread entity", __func__), HCCL_E_PARA);

    ThreadHandle rawHandle = 0;
    const auto* entity = reinterpret_cast<const uint8_t*>(handle);
    CHK_RET(hrtMemSyncCopy(
        &rawHandle, sizeof(rawHandle), entity + offsetof(HcommThreadEntity, compatThreadHandle), sizeof(rawHandle),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));
    CHK_PRT_RET(rawHandle == 0, HCCL_ERROR("[%s] device thread is not initialized", __func__), HCCL_E_INTERNAL);

    // GE 假流仍由设备通信域持有，旧 HCOMM 原语直接从该对象取 notify。
    // 初始化 kernel 已同步且 Entity 尚未发布，临时 Entity/SQ 缓存/notify 描述数组可以立即回收。
    iter->second.entityResources.reset();
    handle = rawHandle;
    return HCCL_SUCCESS;
}

ThreadEntityResources* Thread::GetThreadEntityResources(CommEngine targetEngine) const
{
    auto iter = threadHandleMap_.find(targetEngine);
    return iter == threadHandleMap_.end() ? nullptr : iter->second.entityResources.get();
}

HcclResult SaveThreads(const vector<shared_ptr<Thread>>& newThreads)
{
    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    lock_guard<shared_mutex> lock(ThreadRegistry().EngineThreadMapMtx());
    for (const auto& threadPtr : newThreads) {
        ThreadHandle handle = reinterpret_cast<ThreadHandle>(threadPtr.get());

        if (ThreadRegistry().EngineThreadMap().find(handle) != ThreadRegistry().EngineThreadMap().end()) {
            HCCL_ERROR("[%s] thread handle already exists [0x%llx] in ThreadMap", __func__, handle);
            return HCCL_E_INTERNAL;
        }
        DeviceThreadKey key{deviceId, handle};
        if (ThreadRegistry().EngineThreadD2HMap().find(key) != ThreadRegistry().EngineThreadD2HMap().end()) {
            HCCL_ERROR(
                "[%s] thread handle already exists [0x%llx] in EngineThreadD2HMap, deviceId[%d]", __func__, handle,
                deviceId);
            return HCCL_E_INTERNAL;
        }

        ThreadRegistry().EngineThreadMap().emplace(handle, threadPtr);
        ThreadRegistry().EngineThreadD2HMap().emplace(key, handle);
    }
    return HCCL_SUCCESS;
}

HcclResult FillThreadD2HMap(ThreadHandle* deviceThreadHandles, ThreadHandle* hostThreadHandles, uint32_t listNum)
{
    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    lock_guard<shared_mutex> lock(ThreadRegistry().EngineThreadMapMtx());
    for (uint32_t idx = 0; idx < listNum; idx++) {
        auto deviceThreadHandle = deviceThreadHandles[idx];
        auto hostThreadHandle = hostThreadHandles[idx];
        HCCL_INFO(
            "%s deviceId[%d], deviceThreadHandle[0x%llx], hostThreadHandle[0x%llx]", __func__, deviceId,
            deviceThreadHandle, hostThreadHandle);
        DeviceThreadKey key{deviceId, deviceThreadHandle};
        auto exist = ThreadRegistry().EngineThreadD2HMap().find(key);
        if (exist != ThreadRegistry().EngineThreadD2HMap().end()) {
            if (exist->second == hostThreadHandle) {
                continue; // 幂等登记：同device句柄同host映射，跳过
            }
            int32_t existEngine = -1;
            auto existThread = ThreadRegistry().EngineThreadMap().find(exist->second);
            if (existThread != ThreadRegistry().EngineThreadMap().end()) {
                existEngine = static_cast<int32_t>(existThread->second->GetCommEngine());
            }
            HCCL_RUN_WARNING(
                "[FillThreadD2HMap][COLLISION] device handle[0x%llx] on deviceId[%d] reused: "
                "host[0x%llx](engine[%d]) -> host[0x%llx]",
                deviceThreadHandle, deviceId, exist->second, existEngine, hostThreadHandle);
            exist->second = hostThreadHandle; // 覆盖残留条目：新device对象已占用该地址，新映射为权威映射
            continue;
        }
        ThreadRegistry().EngineThreadD2HMap().emplace(key, hostThreadHandle);
    }

    return HCCL_SUCCESS;
}

HcclResult StoreThreadHandles(
    vector<shared_ptr<Thread>>& newThreads, const std::string& commId, ThreadHandle* threads, CommEngine engine,
    aclrtBinHandle binHandle)
{
    CHK_PTR_NULL(threads);
    if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
        // AICPU引擎处理逻辑
        unique_ptr<ThreadHandle[]> aicpuHandle;
        EXCEPTION_CATCH(aicpuHandle = make_unique<ThreadHandle[]>(newThreads.size()), return HCCL_E_PTR);
        CHK_PTR_NULL(binHandle);
        for (size_t i = 0; i < newThreads.size(); ++i) {
            CHK_RET(newThreads[i]->PreAllocThreadEntity(aicpuHandle[i], engine));
        }
        HcclResult ret;
        if (commId == "") {
            ret = AicpuThreadLaunchMgr::ThreadKernelLaunchForBase(newThreads, aicpuHandle, binHandle);
        } else {
            ret = AicpuThreadLaunchMgr::ThreadKernelLaunchForComm(newThreads, commId, aicpuHandle, binHandle);
        }

        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[StoreThreadHandles] AiCpuKernelLaunch failed, engine[%s], return[%d].",
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), ret),
            ret);

        // 保存并映射AICPU线程句柄
        for (size_t i = 0; i < newThreads.size(); ++i) {
            threads[i] = aicpuHandle[i];
            CHK_RET(newThreads[i]->AddThreadHandleToMap(engine, aicpuHandle[i]));
            ThreadHandle hostHandle = reinterpret_cast<ThreadHandle>(newThreads[i].get());
            CHK_RET(FillThreadD2HMap(&aicpuHandle[i], &hostHandle, 1));
            HCCL_INFO(
                "[StoreThreadHandles] AICPU engine[%s] threadArray[%zu] = [%lu]",
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), i, threads[i]);
        }
    } else {
        for (size_t i = 0; i < newThreads.size(); ++i) {
            threads[i] = reinterpret_cast<ThreadHandle>(newThreads[i].get());
            HCCL_INFO(
                "[StoreThreadHandles] Host engine[%s] threadArray[%zu] = [%lu]",
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), i, threads[i]);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult StoreThreadHandles(
    vector<shared_ptr<Thread>>& newThreads, ThreadHandle* threads, CommEngine engine, aclrtBinHandle binHandle)
{
    return StoreThreadHandles(newThreads, "", threads, engine, binHandle);
}

static HcclResult FreeThreadHandlesLocked(
    const ThreadHandle* threads, uint32_t threadNum, vector<ThreadHandle>& deviceHandles,
    vector<shared_ptr<Thread>>& hostThreadsToRelease)
{
    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    lock_guard<shared_mutex> lock(ThreadRegistry().EngineThreadMapMtx());
    for (uint32_t i = 0; i < threadNum; ++i) {
        const ThreadHandle inHandle = threads[i];

        DeviceThreadKey key{deviceId, inHandle};
        auto itH = ThreadRegistry().EngineThreadD2HMap().find(key);
        if (itH == ThreadRegistry().EngineThreadD2HMap().end()) {
            HCCL_ERROR(
                "[%s] failed to find handle mapping in EngineThreadD2HMap, deviceId[%d], inHandle[0x%llx].", __func__,
                deviceId, inHandle);
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        const ThreadHandle mappedHandle = itH->second;

        auto itC = ThreadRegistry().EngineThreadMap().find(mappedHandle);
        if (itC == ThreadRegistry().EngineThreadMap().end()) {
            HCCL_ERROR(
                "[%s] failed to find thread in EngineThreadMap, deviceId[%d], inHandle[0x%llx], mappedHandle[0x%llx].",
                __func__, deviceId, inHandle, mappedHandle);
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        if (inHandle != mappedHandle) {
            deviceHandles.push_back(inHandle);
        }

        // 暂缓析构 host 侧线程，待 device 侧线程销毁（停止守护线程轮询）后再释放其 SQ/CQ 资源
        hostThreadsToRelease.emplace_back(itC->second);

        HCCL_INFO(
            "[%s] erase thread: deviceId[%d], inHandle[0x%llx], mappedHandle[0x%llx], ptr[%p]", __func__, deviceId,
            inHandle, mappedHandle, itC->second.get());
        ThreadRegistry().EngineThreadMap().erase(itC);
        // 同步摘除order控制流标记：防线程释放后句柄地址被新线程复用时误判为order
        {
            lock_guard<mutex> orderLock(ThreadRegistry().OrderCtrlMtx());
            ThreadRegistry().OrderCtrlThreads().erase(mappedHandle);
        }

        for (auto it = ThreadRegistry().EngineThreadD2HMap().begin();
             it != ThreadRegistry().EngineThreadD2HMap().end();) {
            if (it->second == mappedHandle && it->first.deviceId == deviceId) {
                it = ThreadRegistry().EngineThreadD2HMap().erase(it);
            } else {
                ++it;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult FreeThreads(const ThreadHandle* threads, uint32_t threadNum, aclrtBinHandle binHandle)
{
    CHK_PRT_RET(threads == nullptr, HCCL_ERROR("[HcommThreadfree] threads is null."), HCCL_E_PARA);
    if (threadNum == 0) {
        HCCL_ERROR("[%s] Validate thread params failed. ThreadNum %u, range (0, +inf)", __func__, threadNum);
        return HCCL_E_PARA;
    }
    HCCL_INFO("[%s] begin to free %u threads", __func__, threadNum);

    vector<ThreadHandle> deviceHandles; // 存放device侧的handle
    vector<shared_ptr<Thread>> hostThreadsToRelease; // 暂缓析构的 host 侧线程，避免 CQ 先于 device 侧线程销毁被释放

    CHK_RET(FreeThreadHandlesLocked(threads, threadNum, deviceHandles, hostThreadsToRelease));

    // 先销毁 device 侧线程（将其从守护线程轮询列表 threads_ 移除），再释放 host 侧 SQ/CQ 资源，
    // 避免守护线程在 CQ 已释放后仍对其发起查询导致「参数无效」报错
    if (!deviceHandles.empty()) {
        CHK_RET(AicpuThreadLaunchMgr::ThreadKernelLaunchDestroy(deviceHandles.data(), deviceHandles.size(), binHandle));
    }
    HCCL_INFO("[%s] %u threads freed successfully.", __func__, threadNum);
    return HCCL_SUCCESS;
}

HcclResult SupplementThreadNotify(ThreadHandle handle, uint32_t notifyNum)
{
    shared_lock<shared_mutex> lock(ThreadRegistry().EngineThreadMapMtx());
    auto it = ThreadRegistry().EngineThreadMap().find(handle);
    CHK_PRT_RET(
        it == ThreadRegistry().EngineThreadMap().end(),
        HCCL_ERROR("[%s] thread handle[0x%llx] not found in EngineThreadMap.", __func__, handle), HCCL_E_NOT_FOUND);
    if (it->second->GetNotifyNum() >= notifyNum) {
        return HCCL_SUCCESS;
    }
    u32 supplementNum = notifyNum - it->second->GetNotifyNum();
    HCCL_INFO(
        "[%s] supplement notify num:[%u], current notify num:[%u], target notify num:[%u]", __func__, supplementNum,
        it->second->GetNotifyNum(), notifyNum);
    return it->second->SupplementNotify(supplementNum);
}

HcclResult LookupThreadByHandle(ThreadHandle handle, std::shared_ptr<Thread>& outThread)
{
    shared_lock<shared_mutex> lock(ThreadRegistry().EngineThreadMapMtx());
    auto it = ThreadRegistry().EngineThreadMap().find(handle);
    if (it == ThreadRegistry().EngineThreadMap().end()) {
        // try find device handle
        int32_t deviceId = 0;
        CHK_RET(hrtGetDevice(&deviceId));
        DeviceThreadKey key{deviceId, handle};
        auto device_it = ThreadRegistry().EngineThreadD2HMap().find(key);
        CHK_PRT_RET(
            device_it == ThreadRegistry().EngineThreadD2HMap().end(),
            HCCL_ERROR(
                "[%s] device handle[0x%llx] not found in EngineThreadD2HMap, deviceId[%d].", __func__, handle,
                deviceId),
            HCCL_E_NOT_FOUND);
        it = ThreadRegistry().EngineThreadMap().find(device_it->second);
        CHK_PRT_RET(
            it == ThreadRegistry().EngineThreadMap().end(),
            HCCL_ERROR("[%s] thread handle[0x%llx] not found in EngineThreadMap.", __func__, handle), HCCL_E_NOT_FOUND);
    }

    outThread = it->second;
    return HCCL_SUCCESS;
}

HcclResult LookupD2HHandle(ThreadHandle deviceHandle, ThreadHandle& outHostHandle)
{
    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));
    shared_lock<shared_mutex> lock(ThreadRegistry().EngineThreadMapMtx());
    DeviceThreadKey key{deviceId, deviceHandle};
    auto it = ThreadRegistry().EngineThreadD2HMap().find(key);
    CHK_PRT_RET(
        it == ThreadRegistry().EngineThreadD2HMap().end(),
        HCCL_ERROR(
            "[%s] device handle[0x%llx] not found in EngineThreadD2HMap, deviceId[%d].", __func__, deviceHandle,
            deviceId),
        HCCL_E_NOT_FOUND);
    outHostHandle = it->second;
    return HCCL_SUCCESS;
}

void MarkOrderCtrlThread(ThreadHandle hostHandle)
{
    lock_guard<mutex> lock(ThreadRegistry().OrderCtrlMtx());
    ThreadRegistry().OrderCtrlThreads().insert(hostHandle);
}

bool IsOrderCtrlThread(ThreadHandle hostHandle)
{
    lock_guard<mutex> lock(ThreadRegistry().OrderCtrlMtx());
    return ThreadRegistry().OrderCtrlThreads().find(hostHandle) != ThreadRegistry().OrderCtrlThreads().end();
}
#endif

HcclResult Thread::AddThreadHandleToMap(CommEngine commEngine, ThreadHandle threadHandle)
{
    auto iter = threadHandleMap_.find(commEngine);
    if (iter != threadHandleMap_.end() && iter->second.handle != 0 && iter->second.handle != threadHandle) {
        HCCL_ERROR(
            "[Thread][%s]Mapping already exists:commEngine[%s], threadHandle[%lu], new threadHandle[%lu]", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), commEngine).c_str(), iter->second.handle, threadHandle);
        return HCCL_E_PARA;
    }

    EXCEPTION_CATCH(threadHandleMap_[commEngine].handle = threadHandle, return HCCL_E_MEMORY);
    return HCCL_SUCCESS;
}

ThreadHandle Thread::FindThreadHandleByCommEngine(CommEngine commEngine) const
{
    auto iter = threadHandleMap_.find(commEngine);
    if (iter != threadHandleMap_.end()) {
        return iter->second.handle;
    }

    return 0;
}

HcclResult Thread::ReportAicpuNotifyWaitTask(u64 notifyId, u64 beginTime, u32 taskId, u32 sqId) const
{
    if (!IsReportTask()) {
        return HCCL_SUCCESS;
    }
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    taskParam.beginTime = beginTime;
    taskParam.taskPara.Notify.notifyID = notifyId;
    taskParam.taskPara.Notify.value = 1;
    taskParam.endTime = ProfGetCurCpuTimestamp();
    CHK_PTR_NULL(callback_);
    CHK_RET(callback_(sqId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO(
        "[Thread][%s] sqId[%u], taskId[%u], notifyId[%llu], %s", __func__, sqId, taskId, notifyId,
        taskParam.Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult Thread::ReportHostNotifyWaitTask(
    [[maybe_unused]] u64 notifyId, [[maybe_unused]] u64 beginTime, [[maybe_unused]] bool isMaster) const
{
#ifndef CCL_KERNEL_AICPU
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    taskParam.beginTime = beginTime;
    taskParam.taskPara.Notify.notifyID = notifyId;
    taskParam.taskPara.Notify.value = 1;
    taskParam.isMaster = isMaster;
    u32 taskId = 0;
    u32 streamId = 0;
    hrtGetTaskIdAndStreamID(taskId, streamId);
    taskParam.endTime = hcomm::GetProfCycleTime();
    HCCL_INFO("[ReportHostNotifyWaitTask] time is %llu", taskParam.endTime);
    CHK_PTR_NULL(callback_);
    CHK_RET(callback_(streamId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO(
        "[Thread][%s] streamId[%u], taskId[%u], notifyId[%llu], %s", __func__, streamId, taskId, notifyId,
        taskParam.Describe().c_str());
#endif
    return HCCL_SUCCESS;
}

HcclResult Thread::ReportAicpuNotifyRecordTask(u64 notifyId, u64 beginTime, u32 taskId, u32 sqId) const
{
    if (!IsReportTask()) {
        return HCCL_SUCCESS;
    }
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_RECORD;
    taskParam.beginTime = beginTime;
    taskParam.taskPara.Notify.notifyID = notifyId;
    taskParam.taskPara.Notify.value = 1;
    taskParam.endTime = ProfGetCurCpuTimestamp();
    CHK_PTR_NULL(callback_);
    CHK_RET(callback_(sqId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO(
        "[Thread][%s] sqId[%u], taskId[%u], notifyId[%llu], %s", __func__, sqId, taskId, notifyId,
        taskParam.Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult Thread::ReportHostNotifyRecordTask(
    [[maybe_unused]] u64 notifyId, [[maybe_unused]] u64 beginTime, [[maybe_unused]] bool isMaster) const
{
#ifndef CCL_KERNEL_AICPU
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_RECORD;
    taskParam.beginTime = beginTime;
    taskParam.taskPara.Notify.notifyID = notifyId;
    taskParam.taskPara.Notify.value = 1;
    taskParam.isMaster = isMaster;
    u32 taskId = 0;
    u32 streamId = 0;
    hrtGetTaskIdAndStreamID(taskId, streamId);
    taskParam.endTime = hcomm::GetProfCycleTime();
    HCCL_INFO("[ReportHostNotifyRecordTask] time is %llu", taskParam.endTime);
    CHK_PTR_NULL(callback_);
    CHK_RET(callback_(streamId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO(
        "[Thread][%s] streamId[%u], taskId[%u], notifyId[%llu], %s", __func__, streamId, taskId, notifyId,
        taskParam.Describe().c_str());
#endif
    return HCCL_SUCCESS;
}

HcclResult Thread::ReportHostLocalCopyTask(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t sizeByte,
    [[maybe_unused]] u64 beginTime, [[maybe_unused]] bool isMaster) const
{
#ifndef CCL_KERNEL_AICPU
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_SDMA;
    taskParam.beginTime = beginTime;
    taskParam.taskPara.DMA.src = src;
    taskParam.taskPara.DMA.dst = dst;
    taskParam.taskPara.DMA.size = sizeByte;
    taskParam.taskPara.DMA.notifyID = DFX_INVALID_U64;
    taskParam.taskPara.DMA.linkType = Hccl::DfxLinkType::ONCHIP;
    taskParam.taskPara.DMA.dmaOp = Hccl::DmaOp::HCCL_DMA_READ;
    taskParam.isMaster = isMaster;

    u32 taskId = 0;
    u32 streamId = 0;
    hrtGetTaskIdAndStreamID(taskId, streamId);
    taskParam.endTime = hcomm::GetProfCycleTime();
    CHK_PTR_NULL(callback_);
    CHK_RET(callback_(streamId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO(
        "[Thread][%s] streamId[%u], taskId[%u], src[%p], dst[%p], len[%llu] %s", __func__, streamId, taskId, src, dst,
        sizeByte, taskParam.Describe().c_str());
#endif
    return HCCL_SUCCESS;
}

HcclResult Thread::ReportAicpuLocalCopyTask(
    void* dst, const void* src, uint64_t sizeByte, u64 beginTime, u32 taskId, u32 sqId) const
{
    if (!IsReportTask()) {
        return HCCL_SUCCESS;
    }
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_SDMA;
    taskParam.beginTime = beginTime;
    taskParam.taskPara.DMA.src = src;
    taskParam.taskPara.DMA.dst = dst;
    taskParam.taskPara.DMA.size = sizeByte;
    taskParam.taskPara.DMA.notifyID = DFX_INVALID_U64;
    taskParam.taskPara.DMA.linkType = Hccl::DfxLinkType::ONCHIP;
    taskParam.taskPara.DMA.dmaOp = Hccl::DmaOp::HCCL_DMA_READ;
    taskParam.endTime = ProfGetCurCpuTimestamp();
    CHK_PTR_NULL(callback_);
    CHK_RET(callback_(sqId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO(
        "[Thread][%s] sqId[%u], taskId[%u], src[%p], dst[%p], len[%llu] %s", __func__, sqId, taskId, src, dst, sizeByte,
        taskParam.Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult Thread::ReportAicpuLocalReduceTask(
    void* dst, const void* src, uint64_t sizeByte, HcommDataType dataType, HcommReduceOp reduceOp, u64 beginTime,
    u32 taskId, u32 sqId) const
{
    if (!IsReportTask()) {
        return HCCL_SUCCESS;
    }
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_REDUCE_INLINE;
    taskParam.beginTime = beginTime;
    taskParam.taskPara.Reduce.src = src;
    taskParam.taskPara.Reduce.dst = dst;
    taskParam.taskPara.Reduce.size = sizeByte;
    taskParam.taskPara.Reduce.notifyID = DFX_INVALID_U64;
    taskParam.taskPara.Reduce.linkType = Hccl::DfxLinkType::ONCHIP;
    taskParam.taskPara.Reduce.dataType = static_cast<HcclDataType>(dataType);
    taskParam.taskPara.Reduce.reduceOp = static_cast<HcclReduceOp>(reduceOp);
    CHK_PTR_NULL(callback_);
    CHK_RET(callback_(sqId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO(
        "[Thread][%s] sqId[%u], taskId[%u], src[%p], dst[%p], len[%llu], dataType[%d], reduceOp[%d], %s", __func__,
        sqId, taskId, src, dst, sizeByte, dataType, reduceOp, taskParam.Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult Thread::ReportHostLocalReduceTask(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t sizeByte,
    [[maybe_unused]] HcommDataType dataType, [[maybe_unused]] HcommReduceOp reduceOp, [[maybe_unused]] u64 beginTime,
    [[maybe_unused]] bool isMaster) const
{
#ifndef CCL_KERNEL_AICPU
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_REDUCE_INLINE;
    taskParam.beginTime = beginTime;
    taskParam.taskPara.Reduce.src = src;
    taskParam.taskPara.Reduce.dst = dst;
    taskParam.taskPara.Reduce.size = sizeByte;
    taskParam.taskPara.Reduce.notifyID = DFX_INVALID_U64;
    taskParam.taskPara.Reduce.linkType = Hccl::DfxLinkType::ONCHIP;
    taskParam.taskPara.Reduce.dataType = static_cast<HcclDataType>(dataType);
    taskParam.taskPara.Reduce.reduceOp = static_cast<HcclReduceOp>(reduceOp);
    taskParam.isMaster = isMaster;
    u32 taskId = 0;
    u32 streamId = 0;
    hrtGetTaskIdAndStreamID(taskId, streamId);
    taskParam.endTime = hcomm::GetProfCycleTime();

    CHK_PTR_NULL(callback_);
    CHK_RET(callback_(streamId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO(
        "[Thread][%s] streamId[%u], taskId[%u], src[%p], dst[%p], len[%llu], dataType[%d], reduceOp[%d] %s", __func__,
        streamId, taskId, src, dst, sizeByte, dataType, reduceOp, taskParam.Describe().c_str());
#endif
    return HCCL_SUCCESS;
}

bool Thread::IsReportTask() const
{
#ifdef CCL_KERNEL_AICPU
    return hcomm::GetTaskExceptionEnable() || Hccl::DfxProfilingHandlerLite::GetInstance().GetProfL1State();
#endif
    return true;
}

std::string Thread::FormatThreadInfo() const
{
    s32 streamId = GetStreamId();
    size_t notifySize = GetNotifySize();
    std::ostringstream oss;
    oss << "threadHandle[0x" << std::hex << reinterpret_cast<uintptr_t>(this) << std::dec << "], ";
    oss << "streamId[" << streamId << "], ";
    oss << "notifyIds[";
    for (size_t idx = 0; idx < notifySize; ++idx) {
        if (idx != 0) {
            oss << ",";
        }
        LocalNotify* notify = GetNotify(static_cast<uint32_t>(idx));
        if (notify != nullptr) {
            oss << notify->notifyId_;
        } else {
            oss << "null";
        }
    }
    oss << "]";
    return oss.str();
}

} // namespace hccl

#ifdef CCL_KERNEL_AICPU
namespace {
std::unordered_map<ThreadHandle, std::shared_ptr<hccl::Thread>> g_collCommThreadMap;
std::mutex g_collCommThreadMapMtx;
// 进程级销毁回调：HcommThreadDestroy 移除句柄后触发，供上层擦除缓存的裸句柄（防地址复用误判）
HcommThreadDestroyCallback g_threadDestroyCallback = nullptr;
} // namespace
#endif

HcommResult HcommThreadCreateAicpu(const char* uniqueId, size_t uniqueIdLen, ThreadHandle* handle)
{
#ifndef CCL_KERNEL_AICPU
    (void)uniqueId;
    (void)uniqueIdLen;
    (void)handle;
    return static_cast<HcommResult>(HCOMM_E_NOT_SUPPORT);
#else
    CHK_PTR_NULL(uniqueId);
    CHK_PTR_NULL(handle);
    std::shared_ptr<hccl::AicpuTsThread> thread;
    EXCEPTION_CATCH(
        thread = std::make_shared<hccl::AicpuTsThread>(std::string(uniqueId, uniqueIdLen)), return HCOMM_E_PTR);
    HcclResult ret = thread->Init();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] init aicpu thread failed, ret[%d]", __func__, ret),
        static_cast<HcommResult>(ret));
    ThreadHandle newHandle = reinterpret_cast<ThreadHandle>(thread.get());
    {
        std::lock_guard<std::mutex> lock(g_collCommThreadMapMtx);
        g_collCommThreadMap[newHandle] = thread;
    }
    *handle = newHandle;
    HCCL_INFO("[%s] created aicpu thread[0x%llx]", __func__, static_cast<unsigned long long>(newHandle));
    return HCOMM_SUCCESS;
#endif
}

HcommResult HcommThreadFillEntity(ThreadHandle handle, HcommThreadEntity* entity)
{
#ifndef CCL_KERNEL_AICPU
    (void)handle;
    (void)entity;
    return static_cast<HcommResult>(HCOMM_E_NOT_SUPPORT);
#else
    CHK_PTR_NULL(entity);
    std::shared_ptr<hccl::Thread> thread;
    {
        std::lock_guard<std::mutex> lock(g_collCommThreadMapMtx);
        auto it = g_collCommThreadMap.find(handle);
        CHK_PRT_RET(
            it == g_collCommThreadMap.end(),
            HCCL_ERROR("[%s] invalid thread handle[0x%llx]", __func__, static_cast<unsigned long long>(handle)),
            HCOMM_E_PTR);
        thread = it->second;
    }
    auto* threadPtr = dynamic_cast<hccl::AicpuTsThread*>(thread.get());
    CHK_PTR_NULL(threadPtr);
    return static_cast<HcommResult>(threadPtr->FillHcommThreadEntity(*entity));
#endif
}

HcommResult HcommThreadUnwrapHandle(ThreadHandle* handle)
{
#ifndef CCL_KERNEL_AICPU
    (void)handle;
    return static_cast<HcommResult>(HCOMM_E_NOT_SUPPORT);
#else
    CHK_PTR_NULL(handle);
    return static_cast<HcommResult>(hccl::UnwrapThreadHandle(*handle));
#endif
}

HcommResult HcommThreadGetStreamLite(ThreadHandle handle, void** streamLite)
{
#ifndef CCL_KERNEL_AICPU
    (void)handle;
    (void)streamLite;
    return static_cast<HcommResult>(HCOMM_E_NOT_SUPPORT);
#else
    CHK_PTR_NULL(streamLite);
    CHK_PRT_RET(handle == 0, HCCL_ERROR("[%s] thread is 0", __func__), HCOMM_E_PTR);
    std::shared_ptr<hccl::Thread> thread;
    {
        std::lock_guard<std::mutex> lock(g_collCommThreadMapMtx);
        auto it = g_collCommThreadMap.find(handle);
        CHK_PRT_RET(
            it == g_collCommThreadMap.end(),
            HCCL_ERROR("[%s] invalid thread handle[0x%llx]", __func__, static_cast<unsigned long long>(handle)),
            HCOMM_E_PTR);
        thread = it->second; // 持引用期间对象不会被析构
    }
    CHK_PTR_NULL(thread);
    *streamLite = thread->GetStreamLitePtr();
    CHK_PRT_RET(
        *streamLite == nullptr,
        HCCL_ERROR("[%s] streamLite is null, thread[0x%llx]", __func__, static_cast<unsigned long long>(handle)),
        HCOMM_E_PTR);
    return HCOMM_SUCCESS;
#endif
}

HcommResult HcommThreadIsFakeDeviceRes(ThreadHandle handle, bool* isFake)
{
#ifndef CCL_KERNEL_AICPU
    (void)handle;
    (void)isFake;
    return static_cast<HcommResult>(HCOMM_E_NOT_SUPPORT);
#else
    CHK_PTR_NULL(isFake);
    CHK_PRT_RET(handle == 0, HCCL_ERROR("[%s] thread is 0", __func__), HCOMM_E_PTR);
    std::shared_ptr<hccl::Thread> thread;
    {
        std::lock_guard<std::mutex> lock(g_collCommThreadMapMtx);
        auto it = g_collCommThreadMap.find(handle);
        CHK_PRT_RET(
            it == g_collCommThreadMap.end(),
            HCCL_ERROR("[%s] invalid thread handle[0x%llx]", __func__, static_cast<unsigned long long>(handle)),
            HCOMM_E_PTR);
        thread = it->second; // 持引用期间对象不会被析构
    }
    CHK_PTR_NULL(thread);
    *isFake = thread->IsFakeDeviceRes();
    return HCOMM_SUCCESS;
#endif
}

HcommResult HcommThreadRegisterCacheCallback(ThreadHandle handle)
{
#ifndef CCL_KERNEL_AICPU
    (void)handle;
    return static_cast<HcommResult>(HCOMM_E_NOT_SUPPORT);
#else
    CHK_PRT_RET(handle == 0, HCCL_ERROR("[%s] thread is 0", __func__), HCOMM_E_PTR);
    std::shared_ptr<hccl::Thread> thread;
    {
        std::lock_guard<std::mutex> lock(g_collCommThreadMapMtx);
        auto it = g_collCommThreadMap.find(handle);
        CHK_PRT_RET(
            it == g_collCommThreadMap.end(),
            HCCL_ERROR("[%s] invalid thread handle[0x%llx]", __func__, static_cast<unsigned long long>(handle)),
            HCOMM_E_PTR);
        thread = it->second; // 持引用期间对象不会被析构
    }
    auto* threadPtr = dynamic_cast<hccl::AicpuTsThread*>(thread.get());
    CHK_PRT_RET(
        threadPtr == nullptr,
        HCCL_ERROR("[%s] thread handle type mismatch[0x%llx]", __func__, static_cast<unsigned long long>(handle)),
        HCOMM_E_PTR);
    auto* streamLite = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
    CHK_PTR_NULL(streamLite);
    auto* rtsqA5 = dynamic_cast<Hccl::RtsqA5*>(streamLite->GetRtsq());
    CHK_PTR_NULL(rtsqA5);
    CHK_RET(rtsqA5->SetAicpuTsThreadPtr(threadPtr));
    CHK_RET(rtsqA5->SetNeedCacheTaskCallback(hcomm::AicpuTaskCacheManager::NeedCacheTask));
    CHK_RET(rtsqA5->SetAddSqeArrayCallback(hcomm::AicpuTaskCacheManager::AddSqeArray));
    return HCOMM_SUCCESS;
#endif
}

HcommResult HcommThreadDestroy(ThreadHandle handle)
{
#ifndef CCL_KERNEL_AICPU
    (void)handle;
    return static_cast<HcommResult>(HCOMM_E_NOT_SUPPORT);
#else
    CHK_PRT_RET(handle == 0, HCCL_ERROR("[%s] thread is 0", __func__), HCOMM_E_PTR);
    HcommThreadDestroyCallback callback = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_collCommThreadMapMtx);
        size_t erased = g_collCommThreadMap.erase(handle);
        CHK_PRT_RET(
            erased == 0,
            HCCL_WARNING(
                "[%s] thread handle[0x%llx] not found in registry", __func__, static_cast<unsigned long long>(handle)),
            HCOMM_E_PTR);
        callback = g_threadDestroyCallback;
    }
    // 在锁外回调：销毁回调（如 order 控制流 Registry 反注册）持各自互斥，避免锁序嵌套
    if (callback != nullptr) {
        callback(handle);
    }
    return HCOMM_SUCCESS;
#endif
}

HcommResult HcommThreadRegisterDestroyCallback(HcommThreadDestroyCallback callback)
{
#ifndef CCL_KERNEL_AICPU
    (void)callback;
    return static_cast<HcommResult>(HCOMM_E_NOT_SUPPORT);
#else
    std::lock_guard<std::mutex> lock(g_collCommThreadMapMtx);
    g_threadDestroyCallback = callback;
    return HCOMM_SUCCESS;
#endif
}
