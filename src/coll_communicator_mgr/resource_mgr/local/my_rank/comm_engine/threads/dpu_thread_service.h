/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_THREAD_SERVICE_H
#define DPU_THREAD_SERVICE_H

#include <memory>
#include <mutex>
#include "hccl/hccl_res.h"
#include "dpu_types.h" // 回调 ABI（Hccl::CallbackTemplate / ProfCallbackTemplate / ReportCallbackTemplate 等）

namespace hccl {

class ThreadMgr;

// HostDpu CPU 线程（1 thread 1 kernel）的宿主侧生命周期服务，收编原 api_c_adpt/coll_comm_dpu_c_adpt
// 的三个导出函数：确保 kernel 拉起（幂等）、停止 taskexception 握手、取 DPUTAG 共享内存地址。
// 由 ThreadMgr 每通信域持有一份，方法内经 threadMgr_ 复用线程池查链/acquire/回滚能力。
//
// 本类是 coll_communicator_mgr 里唯一接触 base_comm 线程实体的地方，故本头文件也不得出现其线程类名。
class DpuThreadService {
public:
    explicit DpuThreadService(ThreadMgr* threadMgr, aclrtBinHandle binHandle);

    HcclResult EnsureDpuKernelLaunched(HcclComm commV2, uint32_t devId, bool* newCreated);
    HcclResult StopTaskexpService();
    HcclResult AcquireDpuTagShmem(HcclComm commV2, uint32_t devId, uint64_t* size, void** addr, bool* newCreated);

    // 只查不拉起：挂起路径不应因取地址而新建 DPU 线程；本通信域未拉起 DPU 时 addr 置 nullptr
    HcclResult GetDputagAddrIfLaunched(void*& addr);

    // 以下窄接口代调用方取回实体并转发动词/取值，使 L2 其余层次不必接触 base_comm 的线程对象
    HcclResult RegisterTaskCallback(const char* msgTag, Hccl::CallbackTemplate callback);
    HcclResult UnRegisterTaskCallback(const char* msgTag);
    HcclResult RegisterProfCallback(Hccl::ProfCallbackTemplate profCallback);
    HcclResult RegisterReportCallback(Hccl::ReportCallbackTemplate reportCallback);
    HcclResult RegisterDpuTaskInfoCallback(Hccl::GetDpuTaskInfoCallbackTemplate getDpuTaskInfoCallback);
    // 取 DPU 流 id，供 dfx 侧记录任务来源
    HcclResult GetDpuStreamId(uint32_t& streamId) const;
    // 取 DPUTASKEXCEPTION 共享内存 host 侧地址，供 dfx 异常链路读回错误码；
    // 异常区随 DPU 线程实体释放，keepAlive 是类型擦除的保活引用，持有期间 addr 才有效
    HcclResult GetTaskExpMem(void*& addr, std::shared_ptr<void>& keepAlive) const;
    // 取本通信域 DPU 线程绑定的设备号，供异常链路按 deviceid 过滤
    HcclResult GetDpuDevId(uint32_t& devId) const;

private:
    ThreadMgr* threadMgr_;
    // 本通信域的 kernel 二进制句柄（随通信域初始化下传）：taskexp kernel 下发要按 kernel 名解析函数句柄
    aclrtBinHandle binHandle_{nullptr};
    // 同一 comm 的并发首拉会经池键复用拿到同一个 CpuThread，重复 Init 会双份申请共享内存并下发第二个 kernel，
    // 故临界区须覆盖 Init 与下发；锁随本类每 comm 一份，只挡本通信域，不串行其它通信域的首拉
    std::mutex dpuLaunchMutex_;
};

} // namespace hccl

#endif // DPU_THREAD_SERVICE_H
