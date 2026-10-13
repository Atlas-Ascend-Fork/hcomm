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
#include "hccl/hccl_res.h"
#include "hccl_independent_common.h"
#include "hcomm_res_defs.h"
#include "log.h"
#include "manager_common.h"

namespace hccl {

// L1 层缓存用的值类型，替代 shared_ptr<Thread>，不依赖 base_comm 私有 C++ 对象
struct ThreadMeta {
    ThreadHandle handle{0}; // 线程句柄（L0 opaque）
    uint32_t notifyNum{0};  // 当前 notify 数量
    CommEngine engine{COMM_ENGINE_RESERVED};
    ThreadType type{THREAD_TYPE_INVALID};
    rtStream_t stream{nullptr};
    uint32_t sqId{0}; // 流队列 id
};

class ThreadMgr {
public:
    ThreadMgr(uint32_t threadNum, uint32_t notifyNumPerThread, std::string commId, const ManagerCallbacks& callbacks);
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
    HcclResult RegisterOrderLaunchThread(ThreadHandle thread);
    HcclResult SetAttachedStream(rtStream_t stream);
    HcclResult ResetThreadLocalNotifies();
    u32 GetThreadNum() const { return threadNum_; }
    u32 GetNotifyNumPerThread() const { return notifyNumPerThread_; }

private:
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
    HcommResult FreeEngineToThreads();
    HcommResult FreeMainThreads();
    void FreeDedicatedThreads(); // 专用线程 dedicatedThreadMap_（全类型）
    HcclResult SupplementNotify(std::vector<ThreadMeta>& threadVec, uint32_t threadNum, const ThreadConfig* config);
    HcclResult SupplementThread(
        CommEngine engine, std::vector<ThreadMeta>& threadVec, ThreadType type, uint32_t threadNum,
        const ThreadConfig* config);

    u32 threadNum_ = 0;
    u32 notifyNumPerThread_ = 0;
    std::string commId_;

    u64 usedNotifyNum_ = 0;
    u32 totalThreadCount_ = 0; // 仅 HcclThreadAcquire 累加
    std::mutex threadMutex_;

    std::mutex mainThreadMutex_;
    std::map<rtStream_t, ThreadMeta> mainThread_;

    std::mutex engineToThreadMutex_;
    std::map<std::pair<CommEngine, ThreadType>, std::vector<ThreadMeta>> engineToThreadsMap_;

    ManagerCallbacks callbacks_;

    std::mutex dedicatedThreadMutex_;
    std::unordered_map<HcclDedicatedThreadType, ThreadHandle> dedicatedThreadMap_;

    std::mutex aicpuCommInitMutex_; // AICPU 公共域初始化防重叶子锁，串行化 check+launch+set
    std::mutex attachedStreamMutex_;
    rtStream_t attachedStream_{nullptr}; // GE 附属流（HcomSetAttachedStream 注入，仅持有引用，不管理生命周期）

    std::unordered_set<ThreadHandle>
        orderLaunchThreads_; // 保序流（非所有权，资源由 OrderLaunchThreadMgr 管理，同 comm 操作串行无需加锁）
};
} // namespace hccl
#endif
