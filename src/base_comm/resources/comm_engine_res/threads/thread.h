/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef THREAD_H
#define THREAD_H

#include <string>
#include <vector>
#include <memory>
#include "hccl_types.h"
#include "hccl_common.h"
#include "res_pub.h"
#include "hcomm_primitives.h"
#include "hcomm_res_defs.h"
#include "local_notify.h"
#include "stream_pub.h"
#include "acl/acl_rt.h"
#include "adapter_hal_pub.h"
#include "device_capacity.h"
#include "task_param.h"
#include "sal_pub.h"
#include "stream_lite.h"
#include "task_info.h"
#include "adapter_prof.h"

namespace hccl {

constexpr u32 HCOMM_NOTIFY_MAX_NUM = 256;
// 常量在 res_pub.h 中定义，此处仅起别名，老代码里 HCCL_ 前缀的写法继续可用
constexpr u32 HCCL_THREAD_NOTIFY_MAX_NUM = HCOMM_THREAD_NOTIFY_MAX_NUM;
/**
 * @note 职责：通信引擎的Thread的C++抽象接口类，表达并行资源，内部包含thread间的同步Notify。
 */
class Thread {
public:
    virtual ~Thread() = default;
    virtual HcclResult Init() = 0;
    virtual HcclResult DeInit() = 0;
    virtual std::string& GetUniqueId() = 0;
    virtual uint32_t GetNotifyNum() const = 0;
    virtual LocalNotify* GetNotify(uint32_t index) const = 0;
    virtual HcclResult SupplementNotify(uint32_t notifyNum) = 0;

    // A3 Stream & A5 Stream
    virtual bool IsDeviceA5() const = 0;
    virtual Stream* GetStream() const = 0;
    virtual void* GetStreamLitePtr() const = 0;
    virtual void LaunchTask() const = 0;
    virtual void TryLaunchTask() const = 0;

    // Local Data Plane Functions
    virtual HcclResult LocalNotifyRecord(uint32_t notifyId) const = 0;
    virtual HcclResult LocalNotifyWait(uint32_t notifyId) const = 0;

    virtual HcclResult LocalNotifyRecord(ThreadHandle dstThread, uint32_t dstNotifyIdx) const = 0;
    virtual HcclResult LocalNotifyWait(uint32_t notifyIdx, uint32_t timeOut) const = 0;

    virtual HcclResult LocalCopy(void* dst, const void* src, uint64_t sizeByte) const = 0;
    virtual HcclResult
    LocalReduce(void* dst, const void* src, uint64_t sizeByte, HcommDataType dataType, HcommReduceOp reduceOp) const
        = 0;
    virtual bool GetMaster() const = 0;
    virtual void SetIsMaster(bool isMaster) = 0;

    HcclResult AddThreadHandleToMap(CommEngine commEngine, ThreadHandle threadHandle);
    Thread* FindThreadByCommEngine(CommEngine commEngine);
    HcclResult SetAddTaskInfoCallback(std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback)
    {
        CHK_PTR_NULL(callback);
        callback_ = callback;
        return HCCL_SUCCESS;
    }
    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> GetCallback() { return callback_; }
    virtual HcclResult SetCheckExecStatusCallback([[maybe_unused]] std::function<HcclResult(bool)> callback)
    {
        return HCCL_SUCCESS;
    }

    void SetCommEngine(CommEngine engine) { engine_ = engine; }
    CommEngine GetCommEngine() const { return engine_; }

    // host侧流导出线程规避设备流资源构造的标记：host侧（如GE保序线程经 HcommThreadAcquireByNotify
    // 创建）置位后，导出序列化标记 ONLINE 类型 + 零值 streamParam，不构造真实设备流/SQ/CQ；
    // 设备侧 DeviceInit 解析到"ONLINE + 零值"假流签名时置位。置位线程不持有设备流资源（无
    // StreamLite/pImpl_/Rtsq），相关的资源注册与任务下发统一按此标志拦截跳过
    void SetFakeDeviceRes() { fakeDeviceRes_ = true; }
    bool IsFakeDeviceRes() const { return fakeDeviceRes_; }

protected:
    HcclResult ReportAicpuNotifyWaitTask(u64 notifyId, u64 beginTime, u32 taskId, u32 sqId) const;
    HcclResult ReportHostNotifyWaitTask(u64 notifyId, u64 beginTime, bool isMaster) const;
    HcclResult ReportAicpuNotifyRecordTask(u64 notifyId, u64 beginTime, u32 taskId, u32 sqId) const;
    HcclResult ReportHostNotifyRecordTask(u64 notifyId, u64 beginTime, bool isMaster) const;
    HcclResult
    ReportAicpuLocalCopyTask(void* dst, const void* src, uint64_t sizeByte, u64 beginTime, u32 taskId, u32 sqId) const;
    HcclResult
    ReportHostLocalCopyTask(void* dst, const void* src, uint64_t sizeByte, u64 beginTime, bool isMaster) const;
    HcclResult ReportAicpuLocalReduceTask(
        void* dst, const void* src, uint64_t sizeByte, HcommDataType dataType, HcommReduceOp reduceOp, u64 beginTime,
        u32 taskId, u32 sqId) const;
    HcclResult ReportHostLocalReduceTask(
        void* dst, const void* src, uint64_t sizeByte, HcommDataType dataType, HcommReduceOp reduceOp, u64 beginTime,
        bool isMaster) const;
    bool IsReportTask() const;

private:
    std::unordered_map<CommEngine, ThreadHandle>
        threadHandleMap_; // CPU_TS上的ThreadHandle与其他引擎上的ThreadHandle的映射
    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback_; // 上报task信息的回调函数
    CommEngine engine_ = COMM_ENGINE_RESERVED;                                  // 创建时设置的引擎类型

protected:
    bool fakeDeviceRes_{false}; // host侧流导出线程标记，见 SetFakeDeviceRes 注释
};

inline Stream* GetStream(uint64_t thread)
{
    Thread* threadPtr = reinterpret_cast<Thread*>(thread);
    if (UNLIKELY(threadPtr == nullptr)) {
        HCCL_ERROR("[Thread][GetStream]thread is nullptr");
        return nullptr;
    }
    return threadPtr->GetStream();
}

inline LocalNotify* GetNotify(uint64_t thread, uint32_t index)
{
    Thread* threadPtr = reinterpret_cast<Thread*>(thread);
    if (UNLIKELY(threadPtr == nullptr)) {
        HCCL_ERROR("[Thread][GetNotify]thread is nullptr");
        return nullptr;
    }
    return threadPtr->GetNotify(index);
}

HcclResult CreateThread(
    CommEngine engine, StreamType streamType, uint32_t notifyNum, NotifyLoadType loadType,
    std::shared_ptr<Thread>& out_thread);
HcclResult CommEngineToNotifyLoadType(CommEngine engine, NotifyLoadType& type);
HcclResult CommHostEngineToNotifyLoadType(CommEngine engine, NotifyLoadType& type);
HcclResult CommEngineToStreamType(CommEngine engine, StreamType& type);
HcclResult GetNotifyLoadType(CommEngine engine, ThreadType threadType, NotifyLoadType& type);
HcclResult GetStreamType(CommEngine engine, ThreadType threadType, StreamType& type);
HcclResult ValidateThreadParams(uint32_t threadNum, uint32_t notifyNumPerThread);
HcclResult SaveThreads(const std::vector<std::shared_ptr<hccl::Thread>>& newThreads);
HcclResult FillThreadD2HMap(ThreadHandle* deviceThreadHandles, ThreadHandle* hostThreadHandles, uint32_t listNum);
HcclResult StoreThreadHandles(
    std::vector<std::shared_ptr<hccl::Thread>>& newThreads, ThreadHandle* threads, CommEngine engine,
    aclrtBinHandle binHandle);
HcclResult StoreThreadHandles(
    std::vector<std::shared_ptr<hccl::Thread>>& newThreads, const std::string& commId, ThreadHandle* threads,
    CommEngine engine, aclrtBinHandle binHandle);
HcclResult FreeThreads(const ThreadHandle* threads, uint32_t threadNum, aclrtBinHandle binHandle);
HcclResult SupplementThreadNotify(ThreadHandle handle, uint32_t notifyNum);
/**
 * @brief 按句柄查全局线程表，获取对应的 shared_ptr<Thread>
 * @param[in] handle 线程句柄
 * @param[out] outThread 输出的线程共享指针
 * @return HcclResult 成功返回 HCCL_SUCCESS，未找到返回 HCCL_E_NOT_FOUND
 */
HcclResult LookupThreadByHandle(ThreadHandle handle, std::shared_ptr<hccl::Thread>& outThread);

/**
 * @brief 按 device 句柄查 device→host 映射表，返回对应的 host 句柄
 * @param[in] deviceHandle device 侧线程句柄
 * @param[out] outHostHandle 输出的 host 侧线程句柄
 * @return HcclResult 成功返回 HCCL_SUCCESS，未找到返回 HCCL_E_NOT_FOUND
 */
HcclResult LookupD2HHandle(ThreadHandle deviceHandle, ThreadHandle& outHostHandle);

/**
 * @brief 登记线程为order控制流线程（进程级集合，key=host侧线程对象句柄，即g_ThreadMap的key，
 *        与export入参句柄可能为device句柄无关——登记/查询统一在host对象地址空间）。幂等；
 *        由HcclDedicatedThreadAcquire保序分支（仅OPBASE/ACLGRAPH，context级共享线程）获取即登记，
 *        线程释放时自动摘除
 * @param[in] hostHandle host侧线程对象句柄（g_ThreadMap的key）
 */
void MarkOrderCtrlThread(ThreadHandle hostHandle);

/**
 * @brief 判定线程是否order控制流线程（PrepareThreadMgrParam按下发批次逐线程查询，
 *        填充opParam.orderCtrlFlag传device侧）
 * @param[in] hostHandle host侧线程对象句柄（g_ThreadMap的key）
 * @return true=order控制流线程
 */
bool IsOrderCtrlThread(ThreadHandle hostHandle);
} // namespace hccl
#endif // THREAD_H
