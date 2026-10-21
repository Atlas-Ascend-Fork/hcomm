/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_service.h"
#include <algorithm>
#include "profiling/dlprof_function_v2.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include "acl/acl_rt.h"
#include "log.h"
#include "dpu_kernel_entrance.h"
#include "dpu_types.h"

namespace Hccl {
constexpr uint32_t CTRL_HDR_FLAG_LENGTH = 1;
constexpr uint32_t TASKTYPE_ADDR_LENGTH = 256;
constexpr uint32_t CTRL_HDR_MSG_ID_LEN = 4;
constexpr uint32_t TIMEOUT_SIZE_BYTE = 4;      // timeout字段长度为4字节，表示超时时间，单位为秒
constexpr uint32_t CTRL_HDR_DATA_SIZE_LEN = 8; // size_t 在不同平台上长度不同，取最大值
constexpr uint32_t CTRL_HDR_DEFAULT_DATA_LEN = 512;

// 控制信息长度：|flag|taskType|msgId|timeout|dataSize|，其后的 data 段长度由 dataSize 字段在运行期给出
constexpr uint32_t CTRL_HDR_BASE_LEN
    = CTRL_HDR_FLAG_LENGTH + TASKTYPE_ADDR_LENGTH + CTRL_HDR_MSG_ID_LEN + TIMEOUT_SIZE_BYTE + CTRL_HDR_DATA_SIZE_LEN;
// 单次从 deviceMem 读入的长度 = 控制信息 + 随行默认数据。必须是编译期常量：它既是 ctrlHdr 栈缓冲尺寸，
// 也是 ReadFlag 的拷贝上限；写成运行期值即成 VLA（非标准 C++），且任一加数改为可配置长度会放大为栈溢出
constexpr uint32_t CTRL_HDR_READ_LEN = CTRL_HDR_BASE_LEN + CTRL_HDR_DEFAULT_DATA_LEN;

constexpr uint8_t MEMORY_DEVIDE = 2;

TaskService::TaskService(
    void* deviceMem, int32_t deviceMemSize, void* hostMem, int32_t hostMemSize, void* taskExpMem, std::string commId,
    uint32_t devId)
    : npu2dpuMem_(deviceMem),
      shmemSize_(deviceMemSize / MEMORY_DEVIDE),
      hostMem_(hostMem),
      hostMemSize_(hostMemSize),
      taskExpMem_(taskExpMem),
      commId_(commId),
      devId_(devId)
{
    int32_t controlSize = static_cast<int32_t>(CTRL_HDR_BASE_LEN); // 与 TaskRunImpl 读入口径同源，避免两处算法漂移
    if (shmemSize_ < controlSize) {
        leftSize_ = 0;
    } else {
        leftSize_ = shmemSize_ - controlSize;
    }
    dpu2npuMem_ = static_cast<uint8_t*>(npu2dpuMem_) + shmemSize_;
}

TaskService::~TaskService() = default;

HcclResult TaskService::TaskRegister(std::string taskType, CallbackTemplate callback)
{
    HCCL_INFO("[TaskService::%s] taskType[%s]", __func__, taskType.c_str());
    std::lock_guard<std::mutex> lock(cbMtx_);
    callbacks_.insert({taskType, callback});
    return HCCL_SUCCESS;
}

HcclResult TaskService::TaskUnRegister(std::string taskType)
{
    HCCL_INFO("[TaskService::%s] taskType[%s]", __func__, taskType.c_str());
    std::lock_guard<std::mutex> lock(cbMtx_);
    if (callbacks_.find(taskType) == callbacks_.end()) {
        HCCL_WARNING("[TaskService::%s] TaskType Not Found", __func__);
        return HCCL_E_NOT_FOUND;
    }
    callbacks_.erase(taskType);
    return HCCL_SUCCESS;
}

HcclResult TaskService::TaskProfRegister(ProfCallbackTemplate profCallback)
{
    std::lock_guard<std::mutex> lock(cbMtx_);
    if (profCallback_ != nullptr) {
        return HCCL_SUCCESS;
    }
    profCallback_ = profCallback;
    return HCCL_SUCCESS;
}

HcclResult TaskService::TaskReportRegister(ReportCallbackTemplate reportCallback)
{
    std::lock_guard<std::mutex> lock(cbMtx_);
    if (reportCallback_ != nullptr) {
        return HCCL_SUCCESS;
    }
    reportCallback_ = reportCallback;
    return HCCL_SUCCESS;
}

HcclResult TaskService::TaskGetDpuTaskInfoRegister(GetDpuTaskInfoCallbackTemplate getDpuTaskInfoCallback)
{
    std::lock_guard<std::mutex> lock(cbMtx_);
    getDpuTaskInfoCallback_ = getDpuTaskInfoCallback;
    return HCCL_SUCCESS;
}

HcclResult TaskService::WriteFlag(uint8_t* flagPtr, uint8_t newFlag) const
{
    errno_t ret = memcpy_s(flagPtr, sizeof(newFlag), &newFlag, sizeof(newFlag));
    if (ret != EOK) {
        HCCL_ERROR("[TaskService::TaskRun] set flag failed: %d", ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult TaskService::ReadFlag(uint8_t* ctrlHdr, uint8_t& flag) const
{
    // dest 容量与拷贝长度都用编译期常量，不接收调用方传入的长度，避免上界被形参放大
    errno_t ret = memcpy_s(ctrlHdr, CTRL_HDR_READ_LEN, npu2dpuMem_, CTRL_HDR_READ_LEN);
    if (ret != EOK) {
        HCCL_ERROR("[TaskService::%s] memcpy_s failed on flag, return[%d].", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    flag = *ctrlHdr;
    return HCCL_SUCCESS;
}

HcclResult
TaskService::ReadTaskType(const uint8_t* ctrlHdr, const uint8_t* srcTaskTypePtr, std::string& taskTypeStr) const
{
    CHK_PTR_NULL(srcTaskTypePtr);
    // 读 taskType
    char* taskType = new char[TASKTYPE_ADDR_LENGTH];
    CHK_PTR_NULL(taskType);
    int ret = memcpy_s(
        taskType, (sizeof(char) * TASKTYPE_ADDR_LENGTH), ctrlHdr + CTRL_HDR_FLAG_LENGTH,
        (sizeof(char) * TASKTYPE_ADDR_LENGTH));
    if (ret != EOK) {
        HCCL_ERROR("[%s] memcpy failed on taskType, return[%d].", __func__, ret);
        delete[] taskType;
        return HCCL_E_INTERNAL;
    }
    //  查找 \0
    auto it = std::find(taskType, taskType + TASKTYPE_ADDR_LENGTH, '\0');
    if (it == taskType + TASKTYPE_ADDR_LENGTH) {
        HCCL_ERROR("[TaskService::TaskRun] No Null Character Found Within TaskType Max Length");
        delete[] taskType;
        return HCCL_E_PARA;
    }
    taskTypeStr = std::string(taskType);
    delete[] taskType;
    HCCL_INFO("[TaskService::TaskRun] read taskType = %s", taskTypeStr.c_str());

    uint32_t msgId{0};
    ret = memcpy_s(&msgId, sizeof(msgId), ctrlHdr + CTRL_HDR_FLAG_LENGTH + TASKTYPE_ADDR_LENGTH, sizeof(msgId));
    if (ret != EOK) {
        HCCL_ERROR("[%s] memcpy failed on msgId, return[%d].", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[TaskService::TaskRun] read msgId = %u.", msgId);
    return HCCL_SUCCESS;
}

// 共享内存排布：|stop flag[1]|hcclret[2]|dstret[2]|opIndex[4]|taskId[4]|streamId[4]|reserved[3]|
// 其中，stop flag为aicpu侧读取是否停止的标志位, hcclret为aicpu背景线程读取是否有错的标志位,
// dstret为host侧taskexception回调读取是否有错的标志位,
// opIndex为算子序号，taskId/streamId为host侧taskexception回调读取最后入队DPU任务的标识
HcclResult TaskService::ExecuteTaskexception(int32_t ret, u32 taskId, u32 streamId)
{
    if (taskExpMem_ == nullptr) {
        HCCL_ERROR(
            "[TaskService::ExecuteTaskexception] taskExpMem is nullptr, commId[%s] devId[%u].", commId_.c_str(),
            devId_);
        return HCCL_E_NOT_FOUND;
    }
    HcclResult hcclRet = static_cast<HcclResult>(ret);
    uint8_t* basePtr = static_cast<uint8_t*>(taskExpMem_);
    uint8_t* hcclRetPtr = basePtr + sizeof(uint8_t);
    uint8_t* dstRetPtr = hcclRetPtr + sizeof(uint16_t);
    uint8_t* taskIdPtr = dstRetPtr + sizeof(uint16_t) + sizeof(u32); // 跳过 opIndex[4]
    uint8_t* streamIdPtr = taskIdPtr + sizeof(u32);
    errno_t memRet = memcpy_s(hcclRetPtr, sizeof(uint16_t), &hcclRet, sizeof(uint16_t)); // aicpu背景线程轮询的标志位
    if (memRet != 0) {
        HCCL_ERROR("[TaskService::ExecuteTaskexception] memcpy ret for device failed.");
        return HCCL_E_MEMORY;
    }
    memRet = memcpy_s(dstRetPtr, sizeof(uint16_t), &hcclRet, sizeof(uint16_t)); // host侧taskexception回调读出错误码
    if (memRet != 0) {
        HCCL_ERROR("[TaskService::ExecuteTaskexception] memcpy ret for host failed.");
        return HCCL_E_MEMORY;
    }
    memRet = memcpy_s(taskIdPtr, sizeof(u32), &taskId, sizeof(u32));
    if (memRet != 0) {
        HCCL_ERROR("[TaskService::ExecuteTaskexception] memcpy taskId failed.");
        return HCCL_E_MEMORY;
    }
    memRet = memcpy_s(streamIdPtr, sizeof(u32), &streamId, sizeof(u32));
    if (memRet != 0) {
        HCCL_ERROR("[TaskService::ExecuteTaskexception] memcpy streamId failed.");
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

HcclResult TaskService::ExecuteTaskClean() const
{
    errno_t ret = memset_s(hostMem_, hostMemSize_, 0, hostMemSize_);
    if (ret != EOK) {
        HCCL_ERROR("memset hostMem failed: %d", ret);
        return HCCL_E_MEMORY;
    }
    ret = memset_s(npu2dpuMem_, shmemSize_, 0, shmemSize_);
    if (ret != EOK) {
        HCCL_ERROR("memset npu2dpuMem_ failed: %d", ret);
        return HCCL_E_MEMORY;
    }
    ret = memset_s(dpu2npuMem_, shmemSize_, 0, shmemSize_);
    if (ret != EOK) {
        HCCL_ERROR("memset dpu2npuMem_ failed: %d", ret);
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

HcclResult TaskService::ExecuteTask(uint8_t* ctrlHdr, uint8_t* srcPtr, std::string taskTypeStr)
{
    uint64_t beginTime = Hccl::DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    // 回调表在 host 侧 TaskRegister/UnRegister 与 dpu 侧 ExecuteTask 之间共享，取出回调副本后在锁外执行，
    // 避免回调体（用户数据面）再触发注册/注销时自锁
    CallbackTemplate callback;
    {
        std::lock_guard<std::mutex> lock(cbMtx_);
        auto itFunc = callbacks_.find(taskTypeStr);
        if (itFunc == callbacks_.end()) {
            HCCL_ERROR("[TaskService::TaskRun] Callback of taskType[%s] Not Found", taskTypeStr.c_str());
            std::string taskTypeMsg{"map{"};
            for (const auto& pair : callbacks_) {
                taskTypeMsg += pair.first;
            }
            taskTypeMsg += "}";
            HCCL_ERROR("[TaskService::TaskRun] Callback key : %s", taskTypeMsg.c_str());
            return HCCL_E_NOT_FOUND;
        }
        callback = itFunc->second;
    }

    // copy data
    // ctrlHdr 是 1 字节对齐的栈缓冲，reinterpret_cast 直读 size_t 属未对齐 UB，改 memcpy_s 读
    uint64_t dataLen{0};
    errno_t ret = memcpy_s(
        &dataLen, sizeof(dataLen),
        ctrlHdr + CTRL_HDR_FLAG_LENGTH + TASKTYPE_ADDR_LENGTH + CTRL_HDR_MSG_ID_LEN + TIMEOUT_SIZE_BYTE,
        sizeof(dataLen));
    if (ret != EOK) {
        HCCL_ERROR("[%s] memcpy failed on dataLen, return[%d].", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    if (dataLen > static_cast<uint64_t>(leftSize_) || dataLen > static_cast<uint64_t>(hostMemSize_)) {
        HCCL_ERROR(
            "[TaskService::%s] dataLen[%llu] larger than leftSize[%d] or hostMemSize[%d]", __func__, dataLen, leftSize_,
            hostMemSize_);
        return HCCL_E_PARA;
    }
    uint32_t ctrlHdrLen = CTRL_HDR_BASE_LEN;
    /* ctrlHdr提前从deviceMem copy一定长度，如果长度够，直接从ctrlHdr copy，减少一次aclmemcpy耗时 */
    uint8_t* dataPtr = nullptr;
    if (CTRL_HDR_READ_LEN < ctrlHdrLen + dataLen) {
        dataPtr = srcPtr + ctrlHdrLen;
    } else {
        dataPtr = ctrlHdr + ctrlHdrLen;
    }
    ret = memcpy_s(hostMem_, leftSize_, dataPtr, dataLen);
    if (ret != EOK) {
        HCCL_ERROR("control data memcpy failed: %d", ret);
        return HCCL_E_MEMORY;
    }
    auto callbackRet = callback(reinterpret_cast<uint64_t>(hostMem_), dataLen);
    HcclResult retVal = HCCL_SUCCESS;
    if (callbackRet != 0) {
        // 读取最后入队DPU任务的taskId/streamId，写入共享内存供host侧taskexception回调使用
        u32 taskId = 0;
        u32 streamId = 0;
        // 单槽回调与注册同锁，取副本后锁外执行，避免回调体再触发注册时自锁（同 callbacks_ 口径）
        GetDpuTaskInfoCallbackTemplate getDpuTaskInfoCb;
        {
            std::lock_guard<std::mutex> lock(cbMtx_);
            getDpuTaskInfoCb = getDpuTaskInfoCallback_;
        }
        if (getDpuTaskInfoCb != nullptr) {
            auto [tid, sid] = getDpuTaskInfoCb();
            taskId = tid;
            streamId = sid;
            HCCL_INFO(
                "[%s] dpu task failed, callbackRet[%d], taskId[%u], streamId[%u], beginTime[%llu].", __func__,
                callbackRet, taskId, streamId, beginTime);
        }
        // 写DPUTASKEXCEPTION
        CHK_RET(ExecuteTaskexception(callbackRet, taskId, streamId));
        retVal = HCCL_E_INTERNAL;
    }
    return retVal;
}

HcclResult TaskService::SynchronizeControlInfo()
{
    CHK_PTR_NULL(npu2dpuMem_);
    CHK_PTR_NULL(dpu2npuMem_);
    // npu2dpu -> dpu2npu
    int32_t controlDataSize = sizeof(uint8_t) + sizeof(char) * TASKTYPE_ADDR_LENGTH + sizeof(uint32_t);
    HCCL_INFO("[TaskService::TaskRun] Send response: npu2dpu -> dpu2npu memcpy");
    errno_t ret = memcpy_s(dpu2npuMem_, controlDataSize, npu2dpuMem_, controlDataSize);
    if (ret != EOK) {
        HCCL_ERROR("[TaskService::TaskRun] npu2dpu -> dpu2npu memcpy failed: %d", ret);
        return HCCL_E_INTERNAL;
    }

    std::atomic_thread_fence(std::memory_order_seq_cst);

    uint8_t newFlag = 1;
    HCCL_INFO("[TaskService::TaskRun] Send response: Set dpu2npu flag -> 1");
    static_cast<std::atomic<uint8_t>*>(dpu2npuMem_)->store(newFlag, std::memory_order_release);
    return HCCL_SUCCESS;
}

HcclResult TaskService::ProcessTaskOk(uint8_t* ctrlHdr, uint8_t* srcFlagPtr, uint8_t* srcTaskTypePtr)
{
    std::string taskTypeStr;
    HCCL_INFO("[TaskService::TaskRun] flag = %u.", TASK_OK);
    HCCL_INFO("[TaskService::TaskRun] Set npu2dpu flag -> %u.", TASK_UNSET);
    CHK_RET(WriteFlag(srcFlagPtr, TASK_UNSET));
    CHK_RET(ReadTaskType(ctrlHdr, srcTaskTypePtr, taskTypeStr));
    CHK_RET(ExecuteTask(ctrlHdr, srcFlagPtr, taskTypeStr));
    // 取副本后锁外执行，避免回调体再触发注册时自锁（同 callbacks_ 口径）
    ReportCallbackTemplate reportCb;
    {
        std::lock_guard<std::mutex> lock(cbMtx_);
        reportCb = reportCallback_;
    }
    if (reportCb != nullptr) {
        CHK_RET(reportCb());
    }
    CHK_RET(SynchronizeControlInfo());
    return HCCL_SUCCESS;
}

HcclResult TaskService::ExecuteExit(uint8_t* srcFlagPtr) const
{
    CHK_RET(WriteFlag(srcFlagPtr, TASK_TERMINATE_RESPONSE));
    HCCL_INFO("[TaskService::TaskRun] Exiting.");
    return HCCL_SUCCESS;
}

HcclResult TaskService::TaskRun()
{
    // 预检：以下失败都发生在握手开始之前，device 侧并未进入等待；且 leftSize_ 与 hostMemSize_ 的
    // 关系已证明共享内存尺寸与预期布局不符，此时 npu2dpuMem_ 指向的映射不可信，故直接返回、
    // 不向其回写任何内容。回写（终止响应）只留给「握手已开始」的运行期失败，见下方 TaskRunImpl 之后。
    CHK_PTR_NULL(hostMem_);
    CHK_PTR_NULL(npu2dpuMem_);
    CHK_PTR_NULL(dpu2npuMem_);
    if (leftSize_ <= 0) {
        HCCL_ERROR("[TaskService::%s] dataSize[%d] illegal", __func__, leftSize_);
        return HCCL_E_INTERNAL;
    }
    if (leftSize_ > hostMemSize_) {
        HCCL_ERROR("[TaskService::%s] hostMemSize[%d] less than dataSize[%d]", __func__, hostMemSize_, leftSize_);
        return HCCL_E_INTERNAL;
    }

    HcclResult ret = TaskRunImpl();
    // 收编前由 kernel 入口在 TaskRun 失败时写退出信号，现下沉进本类：异常退出也要让 NPU 侧看到终止响应
    if (ret != HCCL_SUCCESS) {
        uint8_t newFlag = TASK_TERMINATE_RESPONSE;
        errno_t cpyRet = memcpy_s(static_cast<uint8_t*>(npu2dpuMem_), sizeof(newFlag), &newFlag, sizeof(newFlag));
        if (cpyRet != EOK) {
            HCCL_ERROR("[TaskService::%s] set exit flag failed: %d", __func__, cpyRet);
            return HCCL_E_INTERNAL;
        }
    }
    return ret;
}

HcclResult TaskService::TaskRunImpl()
{
    // 入参自检已由 TaskRun() 前置完成（那里不属于「握手已开始」，故不回写 device 内存）
    HCCL_INFO(
        "[TaskService::%s] TaskService{npu2dpuMem:%p; dpu2npuMem:%p; hostMem:%p}", __func__, npu2dpuMem_, dpu2npuMem_,
        hostMem_);
    uint8_t flag{0};
    uint8_t* srcFlagPtr = static_cast<uint8_t*>(npu2dpuMem_);
    uint8_t* dstTimeoutPtr
        = static_cast<uint8_t*>(dpu2npuMem_) + sizeof(flag) + sizeof(char) * TASKTYPE_ADDR_LENGTH + CTRL_HDR_MSG_ID_LEN;
    uint8_t* srcTaskTypePtr = srcFlagPtr + sizeof(flag);
    uint8_t ctrlHdr[CTRL_HDR_READ_LEN] = {0}; // 定长栈缓冲：尺寸为编译期常量，非 VLA
    u32 timeout = GetDpuExecTimeout();
    CHK_SAFETY_FUNC_RET(memcpy_s(dstTimeoutPtr, sizeof(timeout), &timeout, sizeof(timeout)));

    while (true) {
        CHK_RET(ReadFlag(ctrlHdr, flag));
        switch (flag) {
            case TASK_UNSET:
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            case TASK_OK:
                CHK_RET(ProcessTaskOk(ctrlHdr, srcFlagPtr, srcTaskTypePtr));
                continue;
            case TASK_TERMINATE:
                HCCL_INFO("[TaskService::TaskRun] flag = %u.", flag);
                HCCL_INFO("[TaskService::TaskRun] Set npu2dpu flag -> %u. Task Run END.", TASK_TERMINATE_RESPONSE);
                CHK_RET(ExecuteExit(srcFlagPtr));
                return HCCL_SUCCESS;
            case TASK_TERMINATE_RESPONSE:
                HCCL_INFO("[TaskService::TaskRun] flag = %u. Exiting.", flag);
                return HCCL_SUCCESS;
            default:
                HCCL_INFO("[TaskService::TaskRun] flag = %u. Unimplemented flag. Exiting.", flag);
                return HCCL_SUCCESS;
        }
    }
    return HCCL_SUCCESS;
}
} // namespace Hccl
