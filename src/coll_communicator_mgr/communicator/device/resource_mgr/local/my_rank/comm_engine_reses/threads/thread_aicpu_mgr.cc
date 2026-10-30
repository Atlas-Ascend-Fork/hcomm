/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "thread_aicpu_mgr.h"
#include "comm_engine/thread/dfx/thread_dfx_pub.h"
#include "comm_engine/thread/device/device_thread_pub.h"
#include "log.h"
#include <sstream>
#include <iomanip>
#include <mutex>
#include <unordered_map>

std::mutex ThreadAicpuMgr::s_orderCtrlMutex;
std::unordered_map<ThreadHandle, ThreadAicpuMgr::OrderCtrlEntry> ThreadAicpuMgr::g_orderCtrlThreads;
std::once_flag ThreadAicpuMgr::s_destroyCbOnce;

ThreadAicpuMgr::ThreadAicpuMgr(hccl::HcclCommDfxLite& dfx, std::function<HcclResult(bool)> checkExecStatusCallback)
    : dfx_(dfx),
      checkExecStatusCallback_(std::move(checkExecStatusCallback))
{
    // 进程级注册一次：base_comm 销毁线程对象时回调擦除 Registry 条目，保证 Registry 与
    // HcommThreadDestroy 状态同步（防悬垂裸句柄/地址复用误判）
    std::call_once(s_destroyCbOnce, []() {
        (void)HcommThreadRegisterDestroyCallback(&ThreadAicpuMgr::UnregisterOrderCtrlThread);
    });
}

ThreadAicpuMgr::~ThreadAicpuMgr()
{
    std::unique_lock<std::shared_mutex> rwLock(threadMutex_);
    for (auto handle : threads_) {
        HcommThreadRegisterCheckExecStatus(handle, nullptr);
        (void)HcommThreadDestroy(handle);
    }
    for (auto handle : cpuExportThread_) {
        (void)HcommThreadDestroy(handle);
    }
    threads_.clear();
    cpuExportThread_.clear();
}

void ThreadAicpuMgr::RegisterOrderCtrlThread(ThreadHandle thread, ThreadAicpuMgr* refMgr, const std::string& commId)
{
    std::lock_guard<std::mutex> lock(s_orderCtrlMutex);
    auto it = g_orderCtrlThreads.find(thread);
    if (it == g_orderCtrlThreads.end()) {
        OrderCtrlEntry entry;
        entry.thread = thread;
        entry.refMgrs.insert(refMgr);
        auto ret = g_orderCtrlThreads.emplace(thread, std::move(entry));
        HCCL_INFO(
            "[ThreadAicpuMgr][%s] order ctrl thread[0x%llx] registered, comm[%s], mgr[%p], refs[%zu]", __func__,
            static_cast<unsigned long long>(thread), commId.c_str(), static_cast<const void*>(refMgr),
            ret.first->second.refMgrs.size());
    } else {
        // host侧线程缓存保证每(线程,comm)仅一次miss建链，重复登记为防御性幂等
        it->second.refMgrs.insert(refMgr);
    }
}

void ThreadAicpuMgr::UnregisterOrderCtrlThread(ThreadHandle thread)
{
    std::lock_guard<std::mutex> lock(s_orderCtrlMutex);
    size_t erased = g_orderCtrlThreads.erase(thread);
    if (erased > 0) {
        HCCL_INFO(
            "[ThreadAicpuMgr][%s] order ctrl thread[0x%llx] unregistered", __func__,
            static_cast<unsigned long long>(thread));
    }
}

HcclResult ThreadAicpuMgr::InitThreads(ThreadMgrAicpuParam* param)
{
    CHK_PTR_NULL(param);
    u32 threadNum = param->threadNum;
    auto** entityList = static_cast<HcommThreadEntity**>(param->deviceHandle);
    CHK_PTR_NULL(entityList);
    for (u32 i = 0; i < threadNum; ++i) {
        CHK_PTR_NULL(entityList[i]);
    }
    std::string hcomId(param->hcomId);
    // 创建期即分流：真线程入outHandles（comm容器），GE桩入stubHandles（cpuExportThread_，comm
    // 容器），order控制流真线程（OPBASE/ACLGRAPH）入进程级Registry
    std::vector<ThreadHandle> outHandles;
    std::vector<ThreadHandle> stubHandles;
    std::vector<ThreadHandle> orderCtrlHandles;
    outHandles.reserve(threadNum);
    stubHandles.reserve(threadNum);
    orderCtrlHandles.reserve(threadNum);
    // 失败时释放本函数已创建（已登记进 base_comm 注册表）的线程，避免泄漏；order 控制流线程同样纳入，
    // 既避免设备线程资源泄漏，也保证 Registry 与 HcommThreadDestroy 状态一致
    auto cleanupCreatedThreads = [&outHandles, &stubHandles, &orderCtrlHandles]() {
        for (auto handle : outHandles) {
            (void)HcommThreadDestroy(handle);
        }
        for (auto handle : stubHandles) {
            (void)HcommThreadDestroy(handle);
        }
        for (auto handle : orderCtrlHandles) {
            (void)HcommThreadDestroy(handle);
        }
        outHandles.clear();
        stubHandles.clear();
        orderCtrlHandles.clear();
    };
    // 校验/查询类失败统一置 loopRet 后 break，循环后统一 cleanupCreatedThreads 释放已创建线程，
    // 与 HcommThreadCreateAicpu/DFX 注册等失败路径的资源回收语义保持一致
    HcclResult loopRet = HCCL_SUCCESS;
    for (u32 i = 0; i < threadNum; ++i) {
        const ThreadParamEntry* entry = GetThreadParamEntry(param, i);
        if (entry == nullptr) {
            HCCL_ERROR("[ThreadAicpuMgr][%s] thread[%u] entry invalid", __func__, i);
            loopRet = HCCL_E_PARA;
            break;
        }
        uint32_t uidLen = 0;
        const char* uidData = GetThreadUid(param, i, uidLen);
        if (uidData == nullptr) {
            HCCL_ERROR("[ThreadAicpuMgr][%s] thread[%u] uid invalid, uidLen[%u]", __func__, i, uidLen);
            loopRet = HCCL_E_PARA;
            break;
        }
        std::string thdUniqueId(uidData, uidLen);
        if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
            std::ostringstream oss;
            oss << "thread uid[" << i << "] raw bytes: ";
            constexpr u32 HEX_WIDTH = 2;
            u32 dumpLen = uidLen > THREAD_UNIQUE_ID_MAX_SIZE ? THREAD_UNIQUE_ID_MAX_SIZE : uidLen;
            for (u32 j = 0; j < dumpLen; ++j) {
                oss << std::hex << std::setw(HEX_WIDTH) << std::setfill('0')
                    << static_cast<unsigned int>(static_cast<unsigned char>(uidData[j])) << " ";
            }
            HCCL_INFO("[ThreadAicpuMgr][%s] %s", __func__, oss.str().c_str());
        }
        ThreadHandle handle = 0;
        HcommResult ret = HcommThreadCreateAicpu(thdUniqueId.c_str(), thdUniqueId.size(), &handle);
        if (ret != HCOMM_SUCCESS) {
            HCCL_ERROR(
                "[ThreadAicpuMgr][%s] comm identifier[%s], init threads num[%u] failed at index %u", __func__,
                hcomId.c_str(), param->threadNum, i);
            cleanupCreatedThreads();
            return static_cast<HcclResult>(ret);
        }
        ret = HcommThreadFillEntity(handle, entityList[i]);
        if (ret != HCOMM_SUCCESS) {
            (void)HcommThreadDestroy(handle);
            cleanupCreatedThreads();
            return static_cast<HcclResult>(ret);
        }
        // 真线程和 Notify 桩均按输入顺序填充 Host 预分配的实体。
        entityList[i]->compatThreadHandle = handle;
        HCCL_INFO(
            "[ThreadAicpuMgr][%s] entityList[%u] compatThreadHandle = [%llu]", __func__, i,
            static_cast<unsigned long long>(handle));
        // 用显式分类标志分流真/桩（与 base 的 thread->IsFakeDeviceRes() 等价）；不以 StreamLite 能力探测，
        // 否则非 950 真线程（无 pImpl_/StreamLite）会被误判为桩，且探测路径会对桩线程刷 ERROR 日志
        bool isFake = false;
        HcommResult fakeRet = HcommThreadIsFakeDeviceRes(handle, &isFake);
        if (fakeRet != HCOMM_SUCCESS) {
            HCCL_ERROR(
                "[ThreadAicpuMgr][%s] HcommThreadIsFakeDeviceRes failed, handle[0x%llx], ret[%d]", __func__,
                static_cast<unsigned long long>(handle), fakeRet);
            // 本次已创建但尚未分类的句柄不在任何分类容器中，cleanupCreatedThreads 覆盖不到，须直接释放
            (void)HcommThreadDestroy(handle);
            loopRet = HCCL_E_PTR;
            break;
        }
        if (isFake) {
            stubHandles.push_back(handle);
            continue;
        }
        if (entry->orderCtrlFlag != 0) {
            // order控制流（OPBASE/ACLGRAPH，context级共享）：注入进程级Registry（不进comm容器，
            // ~ThreadAicpuMgr不释放）。不挂comm级DFX/Cache回调（Registry存活期长于comm，回调引用
            // dfx_会悬垂；DFX降级不影响record/wait）
            orderCtrlHandles.push_back(handle);
            RegisterOrderCtrlThread(handle, this, hcomId);
            continue;
        }
        HCCL_INFO(
            "[ThreadAicpuMgr][%s] thread[%llu] dispatch to comm container, comm[%s]", __func__,
            static_cast<unsigned long long>(handle), hcomId.c_str());
        outHandles.push_back(handle);
    }

    // 循环内校验/查询失败：释放本函数已创建并登记进 base_comm 注册表的线程，避免泄漏
    if (loopRet != HCCL_SUCCESS) {
        cleanupCreatedThreads();
        return loopRet;
    }

    // 注册仅真线程（桩线程无StreamLite/Rtsq，Registry线程不挂comm级回调）。
    // 注册中途失败须清理本函数已创建的线程（此时 threads_/cpuExportThread_ 尚未填充，
    // 否则线程滞留在 base_comm 全局注册表无人释放，且已挂 DFX 回调悬垂）。
    for (size_t i = 0; i < outHandles.size(); ++i) {
        HcclResult regRet = RegisterThreadAddDfxTaskInfo(outHandles[i]);
        if (regRet != HCCL_SUCCESS) {
            cleanupCreatedThreads();
            return regRet;
        }
        regRet = static_cast<HcclResult>(HcommThreadRegisterCacheCallback(outHandles[i]));
        if (regRet != HCCL_SUCCESS) {
            cleanupCreatedThreads();
            return regRet;
        }
    }
    std::unique_lock<std::shared_mutex> rwLock(threadMutex_);
    threads_.insert(threads_.end(), outHandles.begin(), outHandles.end());
    cpuExportThread_.insert(cpuExportThread_.end(), stubHandles.begin(), stubHandles.end());
    HCCL_INFO(
        "[ThreadAicpuMgr][%s] comm identifier[%s], init threads num[%u] success", __func__, hcomId.c_str(), threadNum);
    return HCCL_SUCCESS;
}

HcclResult ThreadAicpuMgr::RegisterThreadAddDfxTaskInfo(ThreadHandle thread)
{
    int32_t ret = HcommThreadRegisterCheckExecStatus(thread, checkExecStatusCallback_);
    if (ret != 0) {
        HCCL_ERROR(
            "[%s]HcommThreadRegisterCheckExecStatus failed, ret[%d], thread[0x%llx], checkExecStatusCallback[%p]",
            __func__, ret, static_cast<unsigned long long>(thread),
            static_cast<const void*>(&checkExecStatusCallback_));
        return HCCL_E_PTR;
    }

    std::function<void(Hccl::TaskInfoCircularQueue*)> reportCallback = [this](Hccl::TaskInfoCircularQueue* taskQueue) {
        dfx_.ReportStreamTask(taskQueue);
    };
    ret = HcommNewThreadRegisterDfx(thread, reportCallback);
    if (ret != 0) {
        HCCL_ERROR("[%s] HcommNewThreadRegisterDfx failed, ret[%d], thread[0x%llx]", __func__, ret, thread);
        return HCCL_E_PTR;
    }

    std::function<const void*()> getLatestOpInfoCallback = [this]() -> const void* {
        return dfx_.GetLatestDfxOpInfo();
    };
    ret = HcommNewThreadRegisterGetLatestDfxOpInfo(thread, getLatestOpInfoCallback);
    if (ret != 0) {
        HCCL_ERROR(
            "[%s] HcommNewThreadRegisterGetLatestDfxOpInfo failed, ret[%d], thread[0x%llx]", __func__, ret, thread);
        return HCCL_E_PTR;
    }

    return HCCL_SUCCESS;
}
