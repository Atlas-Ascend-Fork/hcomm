/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef THREAD_MANAGER_H
#define THREAD_MANAGER_H
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <map>
#include <functional>
#include "hccl/hccl_res.h"
#include "hccl_independent_common.h"
#include "hcomm_res_defs.h"
#include "log.h"
#include "manager_common.h"
#include "cpu_thread_service.h"
#include "task_param.h"

namespace hccl {

// 前置声明：本头文件也会被 device 侧包含，不能在此引入 base_comm 的 thread.h
class Thread;
class CpuThread;

// L1 层缓存用的值类型，替代 shared_ptr<Thread>，不依赖 base_comm 私有 C++ 对象
struct ThreadMeta {
    ThreadHandle handle{0}; // 线程句柄（L0 opaque）
    uint32_t notifyNum{0};  // 已分配notify容量（单调不减；需求减小时底层不释放，保留原值）
    CommEngine engine{COMM_ENGINE_RESERVED};
    ThreadType type{THREAD_TYPE_INVALID};
    rtStream_t stream{nullptr};
    uint32_t sqId{0}; // 流队列 id
};

class ThreadMgr {
public:
    // binHandle 只透传给 CpuThreadService（taskexp kernel 下发按名字解析函数句柄），本类不持有
    ThreadMgr(
        uint32_t threadNum, uint32_t notifyNumPerThread, std::string commId, aclrtBinHandle binHandle,
        const ManagerCallbacks& callbacks);
    ~ThreadMgr();
    HcclResult HcclThreadAcquire(
        CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads,
        std::vector<uint32_t>& threadId);
    HcclResult HcclThreadAcquireV2(
        CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads,
        std::vector<uint32_t>& threadId);
    HcclResult
    HcclThreadAcquireWithStream(CommEngine engine, rtStream_t stream, uint32_t notifyNum, ThreadHandle* thread);
    HcclResult HcclGetNotifyNumInThread(ThreadHandle thread, uint32_t* notifyNum);
    HcclResult HcclThreadExportToCommEngine(
        uint32_t threadNum, const ThreadHandle* threads, CommEngine dstCommEngine, ThreadHandle* exportedThreads);
    HcclResult HcclThreadResGetInfo(ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void** info) const;
    HcclResult
    HcclDedicatedThreadAcquire(HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread);
    HcclResult HcclDedicatedThreadDfx(std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> dfxCallback);
    HcclResult RegisterOrderLaunchThread(ThreadHandle thread);
    HcclResult SetAttachedStream(rtStream_t stream);
    HcclResult ResetThreadLocalNotifies();
    u32 GetThreadNum() const { return threadNum_; }
    u32 GetNotifyNumPerThread() const { return notifyNumPerThread_; }
    const std::string& GetCommId() const { return commId_; }

    /**
     * @brief 按句柄反查 base_comm 全局线程表中的 Thread 对象，供收编后的 ThreadMgr → Thread 查找链使用
     * @param[in] handle 线程句柄（L0 opaque）
     * @param[out] thread 查到的线程对象
     * @return HcclResult 成功返回 HCCL_SUCCESS，未找到返回 HCCL_E_NOT_FOUND
     * @note 返回 shared_ptr 而非裸 Thread*：保证调用期间对象存活。
     *       本接口只做句柄反查，不再校验该句柄是否属于本 ThreadMgr 实例，跨通信域隔离由调用方保证
     */
    HcclResult GetThread(ThreadHandle handle, std::shared_ptr<Thread>& thread) const;

    /**
     * @brief 查 HOST CPU 线程（HostDpu kernel，1 thread 1 kernel）的 Thread 对象
     * @param[out] cpuThread 查到的线程对象；未创建时置 nullptr
     * @return HCCL_SUCCESS 查到；HCCL_E_NOT_FOUND 未创建
     * @note 返回 shared_ptr 而非裸 Thread*：保证调用期间对象存活。本接口按 (COMM_ENGINE_AICPU, THREAD_TYPE_CPU)
     *       与 (COMM_ENGINE_CPU, THREAD_TYPE_CPU) 两个池键查线程池（算子侧保证一个通信域只走一条通道，故至多命中
     *       其一），不做跨通信域校验（跨通信域隔离由调用方 CollComm 保证）
     */
    HcclResult GetCpuThread(std::shared_ptr<Thread>& cpuThread) const;
    HcclResult GetCpuThread(std::shared_ptr<CpuThread>& cpuThread) const;

    /**
     * @brief 回滚一次 HcclThreadAcquireV2 刚拿到的复用池线程：从池中摘除并释放
     * @param[in] handle 待回滚的句柄
     * @return HCCL_SUCCESS 已回滚；HCCL_E_NOT_FOUND 该句柄不在复用池（非本池所有，不做释放）
     * @note 供「acquire 成功但后续拉起失败」的补偿路径使用：池中残留未拉起的条目后，GetCpuThread 的
     *       查链命中会被当成已拉起，且同参数 acquire 会复用该条目而永不重试。仅覆盖
     *       engineToThreadsMap_，mainThread_ / dedicatedThreadMap_ 不受影响
     */
    HcclResult RollbackThread(ThreadHandle handle);

    /**
     * @brief 取 HostDpu 线程生命周期服务（EnsureCpuKernelLaunched / StopTaskexpService / AcquireCpuTagShmem），
     *        由本类每通信域持有一份。
     * @return 服务对象指针，随 ThreadMgr 生命周期持有，恒非空
     */
    CpuThreadService* GetCpuThreadService() { return cpuThreadService_.get(); }

private:
    // CPU 类线程补配 notify：队列通道还要把新的地址与数量同步给 device 侧实体
    HcclResult
    SupplementCpuThreadNotify(const ThreadHandle* handles, const uint32_t* supplementNums, uint32_t threadNum);
    HcclResult EnsureAicpuCommInit(CommEngine engine);
    HcclResult
    HcclUnfoldThreadAcquire(HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread);
    HcclResult
    HcclGeUnfoldThreadAcquire(HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread);
    HcclResult
    HcclDeviceOrderThreadCreate(HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread);
    HcclResult ResetThreadPoolLocalNotifies();        // 线程池 engineToThreadsMap_
    HcclResult ResetMainThreadLocalNotifies();        // 主线程 mainThread_
    HcclResult ResetDedicatedThreadLocalNotifies();   // 专用线程 dedicatedThreadMap_
    HcclResult ResetOrderLaunchThreadLocalNotifies(); // 保序流线程 orderLaunchThreads_
    void FreeEngineToThreads();
    void FreeMainThreads();
    void FreeDedicatedThreads(); // 专用线程 dedicatedThreadMap_（全类型）
    HcclResult SupplementNotify(std::vector<ThreadMeta>& threadVec, uint32_t threadNum, const ThreadConfig* config);
    HcclResult SupplementThread(
        CommEngine engine, std::vector<ThreadMeta>& threadVec, ThreadType type, uint32_t threadNum,
        const ThreadConfig* config);

    u32 threadNum_ = 0;
    u32 notifyNumPerThread_ = 0;
    std::string commId_;

    // HostDpu 线程生命周期服务（每通信域一份，构造时以 this 注入；仅存指针不访问，可安全提前构造）
    std::unique_ptr<CpuThreadService> cpuThreadService_;

    u64 usedNotifyNum_ = 0;
    u32 totalThreadCount_ = 0; // 仅 HcclThreadAcquire 累加
    std::mutex threadMutex_;

    std::mutex mainThreadMutex_;
    std::map<rtStream_t, ThreadMeta> mainThread_;

    mutable std::mutex engineToThreadMutex_; // GetCpuThread 为 const 只读接口，仍须持锁串行化查表
    std::map<std::pair<CommEngine, ThreadType>, std::vector<ThreadMeta>> engineToThreadsMap_;

    ManagerCallbacks callbacks_;

    std::mutex dedicatedThreadMutex_;
    std::unordered_map<HcclDedicatedThreadType, ThreadHandle> dedicatedThreadMap_;
    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> dfxCallback_;

    std::mutex aicpuCommInitMutex_; // AICPU 公共域初始化防重叶子锁，串行化 check+launch+set

    std::mutex attachedStreamMutex_;
    rtStream_t attachedStream_{nullptr}; // GE 附属流（HcomSetAttachedStream 注入，仅持有引用，不管理生命周期）

    std::unordered_set<ThreadHandle>
        orderLaunchThreads_; // 保序流（非所有权，资源由 OrderLaunchThreadMgr 管理，同 comm 操作串行无需加锁）
};
} // namespace hccl
#endif
