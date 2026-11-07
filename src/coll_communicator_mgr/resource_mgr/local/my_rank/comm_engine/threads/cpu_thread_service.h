/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CPU_THREAD_SERVICE_H
#define CPU_THREAD_SERVICE_H

#include <functional>
#include <memory>
#include <mutex>
#include "hccl/hccl_res.h"
#include "task_param.h"

namespace hccl {

class ThreadMgr;
class CpuThread;
class LegacyDpuThread;

// HostDpu CPU 线程（1 thread 1 kernel）的宿主侧生命周期服务，收编原 api_c_adpt/coll_comm_dpu_c_adpt
// 的三个导出函数：确保 kernel 拉起（幂等）、停止 taskexception 握手、取 DPUTAG 共享内存地址。
// 由 ThreadMgr 每通信域持有一份，方法内经 threadMgr_ 复用线程池查链/acquire/回滚能力。
class CpuThreadService {
public:
    explicit CpuThreadService(ThreadMgr* threadMgr, aclrtBinHandle binHandle);

    HcclResult
    EnsureCpuKernelLaunched(HcclComm commV2, uint32_t devId, std::shared_ptr<CpuThread>& cpuThread, bool* newCreated);
    HcclResult StopTaskexpService();
    // 终止 DPU kernel 的 WaitKernelTerminate 握手；须在 ~CollComm 释放 dfx 之前调用（kernel 错误路径经回调读 dfx）
    HcclResult StopCpuKernel();
    HcclResult AcquireCpuTagShmem(HcclComm commV2, uint32_t devId, uint64_t* size, void** addr, bool* newCreated);

    // 只查不拉起：挂起路径不应因取地址而新建 DPU 线程；本通信域未拉起 DPU 时 addr 置 nullptr
    HcclResult GetCpuTagAddrIfLaunched(void*& addr);

    // 取 DPUTASKEXCEPTION 共享内存 host 侧地址，供 dfx 异常链路读回错误码；
    // 异常区随 DPU 线程实体释放，keepAlive 是类型擦除的保活引用，持有期间 addr 才有效
    HcclResult GetTaskExpMem(void*& addr, std::shared_ptr<void>& keepAlive) const;
    // 取本通信域 DPU 线程绑定的设备号，供异常链路按 deviceid 过滤
    HcclResult GetDpuDevId(uint32_t& devId) const;

private:
    // acquire 之后任一失败都要把线程从池中摘除并释放：池中残留「有条目但未拉起」的条目后，
    // 后续调用的查链命中会被当成已拉起而假成功，且同参数 acquire 会复用该条目、Init 再无重试机会。
    HcclResult RollbackAcquiredThread(ThreadHandle handle, std::shared_ptr<CpuThread>& cpuThread, HcclResult cause);
    // 停止握手共性收口：非 dpu 场景（无 CPU thread / 无 legacy 实体）视为无监控可停，返回成功
    static HcclResult StopTaskexpServiceByCpuThread(const std::shared_ptr<CpuThread>& cpuThread);

    ThreadMgr* threadMgr_;
    // 本通信域的 kernel 二进制句柄（随通信域初始化下传）：taskexp kernel 下发要按 kernel 名解析函数句柄
    aclrtBinHandle binHandle_{nullptr};
    // 首次拉起（创建 CpuThread + Init）是进程级共享状态变更，跨通信域串行化避免并发双建竞态
    static std::mutex cpuLaunchMutex_;
};

} // namespace hccl

#endif // CPU_THREAD_SERVICE_H
