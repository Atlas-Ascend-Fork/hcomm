/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef THREAD_AICPU_MGR_H
#define THREAD_AICPU_MGR_H

#include "common.h"
#include "comm_engine/aicpu_launch_manager_pub.h"
#include "comm_engine/thread/device/device_thread_pub.h"
#include "hcclCommDfxLite.h"
#include <shared_mutex>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <functional>

class ThreadAicpuMgr {
public:
    ThreadAicpuMgr(hccl::HcclCommDfxLite& dfx, std::function<HcclResult(bool)> checkExecStatusCallback);
    ~ThreadAicpuMgr();
    HcclResult InitThreads(ThreadMgrAicpuParam* param);
    // L1 层只持 handle，不暴露 base_comm 私有 Thread 类
    const std::vector<ThreadHandle>& GetAllThread() { return threads_; }
    std::shared_mutex& GetThreadMutex() { return threadMutex_; }

    // order控制流device线程Registry（进程级）：承接context级共享的OPBASE/ACLGRAPH导出线程，不随comm
    // 析构（防跨comm复用句柄UAF）；本表线程常驻至进程退出、不做DeInit。refMgrs仅观测；GE桩走cpuExportThread_
    static void RegisterOrderCtrlThread(ThreadHandle thread, ThreadAicpuMgr* refMgr, const std::string& commId);
    // base_comm 设备线程销毁回调：同步擦除 Registry 中已析构线程的悬垂裸句柄，避免 AICPU 同尺寸
    // 对象地址复用后被 find 命中旧条目误判"已注册"
    static void UnregisterOrderCtrlThread(ThreadHandle thread);

private:
    HcclResult RegisterThreadAddDfxTaskInfo(ThreadHandle thread);

    std::shared_mutex threadMutex_;
    // threads_：持有设备流资源的真线程（Resume/异常CQE/dfx遍历均依赖其StreamLite/Rtsq）。
    // cpuExportThread_：host侧（CPU_TS）导出的notify桩线程（无设备流资源，GE保序线程）。
    std::vector<ThreadHandle> threads_;
    std::vector<ThreadHandle> cpuExportThread_;
    hccl::HcclCommDfxLite& dfx_;
    std::function<HcclResult(bool)> checkExecStatusCallback_;

    struct OrderCtrlEntry {
        ThreadHandle thread;                         // 仅记录 handle；真实所有权在 base_comm 注册表
        std::unordered_set<ThreadAicpuMgr*> refMgrs; // 引用方comm集合（仅观测）
    };
    static std::mutex s_orderCtrlMutex;                                         // Registry互斥
    static std::unordered_map<ThreadHandle, OrderCtrlEntry> g_orderCtrlThreads; // order控制流device线程
    static std::once_flag s_destroyCbOnce;                                      // 销毁回调注册一次性
};

#endif // THREAD_AICPU_MGR_H
