/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TASK_SERVICE_H
#define TASK_SERVICE_H

#include <functional>
#include <unordered_map>
#include <mutex>
#include <string>
#include <cstdint>
#include "hccl_types.h"
#include "task_param.h"
#include "dpu_types.h"

namespace Hccl {
/**
 * 1. 使用共享 HBM 内存传递任务信息和数据
 * 2. 内存布局(shmemPtr_)分为两块等长区域：
 *    - NPU -> DPU (npu2dpuShmem)
 * +----------------+---------------------------+------------------+--------------------+----------------+
 * | flag (uint8_t) | taskType (256字节定长空间) | msgId (uint32_t) | timeout (uint32_t) |      data      |
 * +----------------+---------------------------+------------------+--------------------+----------------+
 *    - DPU -> NPU (dpu2npuShmem)
 * +----------------+---------------------------+------------------+--------------------+
 * | flag (uint8_t) | taskType (256字节定长空间) | msgId (uint32_t) | timeout (uint32_t) |
 * +----------------+---------------------------+------------------+--------------------+
 * 3. host侧内存装载npu2dpuShmem 的 data部分
 * +----------------+
 * |      data      |
 * +----------------+
 */
class TaskService {
public:
    TaskService() = default;
    ~TaskService();
    TaskService(
        void* deviceMem, int32_t deviceMemSize, void* hostMem, int32_t hostMemSize, void* taskExpMem,
        std::string commId, uint32_t devId);
    HcclResult TaskRun();
    HcclResult TaskRegister(std::string taskType, CallbackTemplate callback);
    HcclResult TaskUnRegister(std::string taskType);
    HcclResult TaskProfRegister(ProfCallbackTemplate profCallback);
    HcclResult TaskReportRegister(ReportCallbackTemplate reportCallback);
    HcclResult TaskGetDpuTaskInfoRegister(GetDpuTaskInfoCallbackTemplate getDpuTaskInfoCallback);

    /**
     * @brief 取 DPUTASKEXCEPTION 共享内存 host 侧地址，供 dfx 异常链路读回 DPU 任务错误码
     */
    void* GetTaskExpMem() const { return taskExpMem_; }

private:
    HcclResult WriteFlag(uint8_t* flagPtr, uint8_t newFlag) const;
    // 下列方法的 ctrlHdr 缓冲尺寸由 task_service.cc 的 CTRL_HDR_READ_LEN（编译期常量）统一固定，
    // 不再以形参传入长度，避免调用方传入的上界与栈缓冲实际尺寸脱钩
    HcclResult ReadFlag(uint8_t* ctrlHdr, uint8_t& flag) const;
    HcclResult ReadTaskType(const uint8_t* ctrlHdr, const uint8_t* srcTaskTypePtr, std::string& taskTypeStr) const;
    HcclResult ExecuteTask(uint8_t* ctrlHdr, uint8_t* srcPtr, std::string taskTypeStr);
    HcclResult SynchronizeControlInfo();
    HcclResult ProcessTaskOk(uint8_t* ctrlHdr, uint8_t* srcFlagPtr, uint8_t* srcTaskTypePtr);
    HcclResult ExecuteTaskClean() const;
    HcclResult ExecuteTaskexception(int32_t ret, u32 taskId, u32 streamId);
    HcclResult ExecuteExit(uint8_t* srcFlagPtr) const;
    HcclResult TaskRunImpl();

private:
    std::mutex cbMtx_; // callbacks_ 的并发保护，替代原 g_serMapMutex 对回调表的职责
    std::unordered_map<std::string, CallbackTemplate> callbacks_;
    ProfCallbackTemplate profCallback_{nullptr};
    ReportCallbackTemplate reportCallback_{nullptr};
    GetDpuTaskInfoCallbackTemplate getDpuTaskInfoCallback_{nullptr};
    void* npu2dpuMem_{nullptr};
    void* dpu2npuMem_{nullptr};
    int32_t shmemSize_{0};
    int32_t leftSize_{0}; // npu2dpuMem_中除去控制信息后剩余的可用空间大小
    void* hostMem_{nullptr};
    int32_t hostMemSize_{0};
    void* taskExpMem_{nullptr}; // DPUTASKEXCEPTION 共享内存 host 侧地址，异常时回写错误码
    std::string commId_{};
    uint32_t devId_{UINT32_MAX};
};
} // namespace Hccl

#endif // TASK_SERVICE_H
