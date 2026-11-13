/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <cstring>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include "acl/acl_rt.h"
#define private public
#include "hostdpu/task_service.h"
#include "hostdpu/dpu_kernel_entrance.h"
#undef private

using namespace Hccl;

class TaskServiceTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TaskServiceTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "TaskServiceTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        g_callbackResult = 0;
        g_taskExpMemMap.clear();
    }

    virtual void TearDown() { g_taskExpMemMap.clear(); }

public:
    static int32_t g_callbackResult;
    std::string commId = "test";
    uint32_t devId = 0;
};
int32_t TaskServiceTest::g_callbackResult = 0;

// ===== Helper: set up TASK_OK payload in shared memory =====
static void
SetupOkTask(uint8_t* base, const std::string& taskType, uint32_t msgId, uint32_t dataSize, uint32_t timeout = 0)
{
    char* taskTypePtr = reinterpret_cast<char*>(base + 1);
    memset(static_cast<void*>(taskTypePtr), 0, 256);
    size_t copyLen = taskType.size();
    if (copyLen > 255) {
        copyLen = 255;
    }
    memcpy(static_cast<void*>(taskTypePtr), taskType.c_str(), copyLen);
    taskTypePtr[copyLen] = '\0';
    uint32_t* msgIdPtr = reinterpret_cast<uint32_t*>(base + 1 + 256);
    *msgIdPtr = msgId;
    uint32_t* timeoutPtr = reinterpret_cast<uint32_t*>(base + 1 + 256 + sizeof(uint32_t));
    *timeoutPtr = timeout;
    size_t* dataSizePtr = reinterpret_cast<size_t*>(base + 1 + 256 + sizeof(uint32_t) + sizeof(uint32_t));
    *dataSizePtr = static_cast<size_t>(dataSize);
    base[0] = 1; // TASK_OK
}

// ===== Helper: wait for TaskRun to enter the while-loop =====
// deviceMem 初始化为0，模拟 AllocAndRegKFCWorkSpace 在申请注册时已置零
static void WaitForTaskRunLoop(uint8_t* flagPtr)
{
    auto start = std::chrono::steady_clock::now();
    while (*flagPtr != 0) { // TASK_UNSET
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        auto elapsed
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        ASSERT_LT(elapsed, 5000) << "Timeout waiting for TaskRun to enter loop";
    }
}

// ===== Callback stubs =====
static int32_t TrackingCallback(uint64_t /*addr*/, int32_t /*size*/)
{
    TaskServiceTest::g_callbackResult = 42; // signal that callback was called
    return 0;
}

static int32_t FailingCallback(uint64_t /*addr*/, int32_t /*size*/)
{
    TaskServiceTest::g_callbackResult = -1;
    return -1;
}

// ==================== Existing tests (backward compat) ====================

TEST_F(TaskServiceTest, Ut_TaskRun_When_DataSizeIsZero_Expect_ReturnError)
{
    // shmemSize_ < controlSize => hostSize_ = 0 => TaskRun returns HCCL_E_INTERNAL
    constexpr int32_t controlSize
        = sizeof(uint8_t) + sizeof(char) * 256 + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);
    constexpr int32_t deviceMemSize = controlSize * 1; // less than controlSize * 2
    constexpr int32_t hostMemSize = 1024;

    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);

    TaskService taskService(deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId);

    HcclResult ret = taskService.TaskRun();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(TaskServiceTest, Ut_TaskRun_When_DataSizeExceedsHostMemSize_Expect_ReturnError)
{
    // 测试 dataSize_ > hostMemSize_ 的情况
    constexpr int32_t controlSize = sizeof(uint8_t) + sizeof(char) * 256 + sizeof(uint32_t) + sizeof(uint32_t);
    constexpr int32_t deviceMemSize = (controlSize + 2048) * 2; // dataSize_ = 2048
    constexpr int32_t hostMemSize = 1024;                       // hostMemSize_ < dataSize_

    // 分配内存
    void* deviceMem = reinterpret_cast<void*>(0x1234);
    void* hostMem = reinterpret_cast<void*>(0x4321);

    TaskService taskService(deviceMem, deviceMemSize, hostMem, hostMemSize, commId, devId);

    // 调用TaskRun,应该返回HCCL_E_INTERNAL
    HcclResult ret = taskService.TaskRun();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// ==================== Register / UnRegister ====================

TEST_F(TaskServiceTest, Ut_TaskRegister_When_Normal_Expect_Success)
{
    std::vector<uint8_t> deviceMem(1024);
    std::vector<uint8_t> hostMem(512);
    TaskService taskService(deviceMem.data(), 1024, hostMem.data(), 512, commId, devId);

    HcclResult ret = taskService.TaskRegister("testTask", TrackingCallback);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TaskServiceTest, Ut_TaskUnRegister_When_TypeExists_Expect_Success)
{
    std::vector<uint8_t> deviceMem(1024);
    std::vector<uint8_t> hostMem(512);
    TaskService taskService(deviceMem.data(), 1024, hostMem.data(), 512, commId, devId);

    taskService.TaskRegister("testTask", TrackingCallback);
    HcclResult ret = taskService.TaskUnRegister("testTask");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TaskServiceTest, Ut_TaskUnRegister_When_TypeNotExists_Expect_NotFound)
{
    std::vector<uint8_t> deviceMem(1024);
    std::vector<uint8_t> hostMem(512);
    TaskService taskService(deviceMem.data(), 1024, hostMem.data(), 512, commId, devId);

    HcclResult ret = taskService.TaskUnRegister("nonexistent");
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

// ==================== TaskReportRegister ====================

TEST_F(TaskServiceTest, Ut_TaskReportRegister_When_NotRegistered_Expect_SuccessAndCallbackStored)
{
    std::vector<uint8_t> deviceMem(1024);
    std::vector<uint8_t> hostMem(512);
    TaskService taskService(deviceMem.data(), 1024, hostMem.data(), 512, commId, devId);

    ReportCallbackTemplate callback = []() -> HcclResult {
        return HCCL_SUCCESS;
    };
    HcclResult ret = taskService.TaskReportRegister(callback);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(taskService.reportCallback_, nullptr);
    g_taskServiceMap.clear();
}

TEST_F(TaskServiceTest, Ut_TaskReportRegister_When_AlreadyRegistered_Expect_SuccessAndCallbackUnchanged)
{
    std::vector<uint8_t> deviceMem(1024);
    std::vector<uint8_t> hostMem(512);
    TaskService taskService(deviceMem.data(), 1024, hostMem.data(), 512, commId, devId);

    int32_t calledBy = 0;
    ReportCallbackTemplate firstCallback = [&calledBy]() -> HcclResult {
        calledBy = 1;
        return HCCL_SUCCESS;
    };
    taskService.TaskReportRegister(firstCallback);

    ReportCallbackTemplate secondCallback = [&calledBy]() -> HcclResult {
        calledBy = 2;
        return HCCL_E_INTERNAL;
    };
    HcclResult ret = taskService.TaskReportRegister(secondCallback);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    taskService.reportCallback_();
    EXPECT_EQ(calledBy, 1);
    g_taskServiceMap.clear();
}

// ==================== TaskRun exit flags (no callback needed) ====================

TEST_F(TaskServiceTest, Ut_TaskRun_When_TerminateFlag_Expect_Success)
{
    constexpr int32_t deviceMemSize = 1024;
    constexpr int32_t hostMemSize = 512;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);

    TaskService taskService(deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId);

    HcclResult result = HCCL_SUCCESS;
    std::thread taskThread([&taskService, &result]() {
        result = taskService.TaskRun();
    });

    WaitForTaskRunLoop(deviceMem.data());

    uint8_t* flagPtr = deviceMem.data();
    *flagPtr = 2; // TASK_TERMINATE

    taskThread.join();

    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(*flagPtr, 3); // TASK_TERMINATE_RESPONSE
}

TEST_F(TaskServiceTest, Ut_TaskRun_When_TerminateResponseFlag_Expect_Success)
{
    constexpr int32_t deviceMemSize = 1024;
    constexpr int32_t hostMemSize = 512;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);

    TaskService taskService(deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId);

    HcclResult result = HCCL_SUCCESS;
    std::thread taskThread([&taskService, &result]() {
        result = taskService.TaskRun();
    });

    WaitForTaskRunLoop(deviceMem.data());

    uint8_t* flagPtr = deviceMem.data();
    *flagPtr = 3; // TASK_TERMINATE_RESPONSE

    taskThread.join();

    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(TaskServiceTest, Ut_TaskRun_When_InvalidFlag_Expect_Success)
{
    constexpr int32_t deviceMemSize = 1024;
    constexpr int32_t hostMemSize = 512;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);

    TaskService taskService(deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId);

    HcclResult result = HCCL_SUCCESS;
    std::thread taskThread([&taskService, &result]() {
        result = taskService.TaskRun();
    });

    WaitForTaskRunLoop(deviceMem.data());

    uint8_t* flagPtr = deviceMem.data();
    *flagPtr = 255; // unknown / invalid flag

    taskThread.join();

    EXPECT_EQ(result, HCCL_SUCCESS);
}

// ==================== Exec timeout (SetDpuExecTimeout) ====================

TEST_F(TaskServiceTest, Ut_TaskRun_When_ExecTimeoutSet_Expect_WriteTimeout)
{
    constexpr int32_t deviceMemSize = 1024;
    constexpr int32_t hostMemSize = 512;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);

    TaskService taskService(deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId);
    SetDpuExecTimeout(100);

    HcclResult result = HCCL_SUCCESS;
    std::thread taskThread([&taskService, &result]() {
        result = taskService.TaskRun();
    });

    WaitForTaskRunLoop(deviceMem.data());

    uint8_t* flagPtr = deviceMem.data();
    *flagPtr = 2; // TASK_TERMINATE

    taskThread.join();

    EXPECT_EQ(result, HCCL_SUCCESS);
    uint32_t timeoutVal = 0;
    memcpy(&timeoutVal, deviceMem.data() + deviceMemSize / 2 + 1 + 256 + sizeof(uint32_t), sizeof(timeoutVal));
    EXPECT_EQ(timeoutVal, 100U);
}

TEST_F(TaskServiceTest, Ut_TaskRun_When_ExecTimeoutDefault_Expect_WriteDefault)
{
    constexpr int32_t deviceMemSize = 1024;
    constexpr int32_t hostMemSize = 512;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);

    TaskService taskService(deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId);
    SetDpuExecTimeout(DPU_EXEC_TIMEOUT_DEFAULT_S);

    HcclResult result = HCCL_SUCCESS;
    std::thread taskThread([&taskService, &result]() {
        result = taskService.TaskRun();
    });

    WaitForTaskRunLoop(deviceMem.data());

    uint8_t* flagPtr = deviceMem.data();
    *flagPtr = 2; // TASK_TERMINATE

    taskThread.join();

    EXPECT_EQ(result, HCCL_SUCCESS);
    uint32_t timeoutVal = 0;
    memcpy(&timeoutVal, deviceMem.data() + deviceMemSize / 2 + 1 + 256 + sizeof(uint32_t), sizeof(timeoutVal));
    EXPECT_EQ(timeoutVal, DPU_EXEC_TIMEOUT_DEFAULT_S);
}

// ==================== TASK_OK with unknown callback ====================

TEST_F(TaskServiceTest, Ut_TaskRun_When_TaskOkCallbackNotFound_Expect_NotFound)
{
    constexpr int32_t deviceMemSize = 1024;
    constexpr int32_t hostMemSize = 512;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);

    TaskService taskService(deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId);
    taskService.TaskRegister("existingTask", TrackingCallback);

    HcclResult result = HCCL_SUCCESS;
    std::thread taskThread([&taskService, &result]() {
        result = taskService.TaskRun();
    });

    WaitForTaskRunLoop(deviceMem.data());
    SetupOkTask(deviceMem.data(), "nonexistentTask", 42, 0);
    // flag is now TASK_OK (1) from SetupOkTask

    taskThread.join();

    EXPECT_EQ(result, HCCL_E_NOT_FOUND);
}

// ==================== TASK_OK with data size exceeding hostSize ====================

TEST_F(TaskServiceTest, Ut_TaskRun_When_TaskOkDataTooLarge_Expect_Error)
{
    constexpr int32_t deviceMemSize = 1024;
    constexpr int32_t hostMemSize = 512;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);
    // deviceMemSize=1024 => shmemSize_=512 => leftSize_=512-273=239

    TaskService taskService(deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId);
    taskService.TaskRegister("testTask", TrackingCallback);

    HcclResult result = HCCL_SUCCESS;
    std::thread taskThread([&taskService, &result]() {
        result = taskService.TaskRun();
    });

    WaitForTaskRunLoop(deviceMem.data());
    SetupOkTask(deviceMem.data(), "testTask", 42, 500); // dataSize=500 > leftSize_=239
    // flag is now TASK_OK (1) from SetupOkTask

    taskThread.join();

    EXPECT_EQ(result, HCCL_E_PARA);
}

// ==================== TASK_OK with callback returning error ====================

TEST_F(TaskServiceTest, Ut_TaskRun_When_TaskOkCallbackFails_Expect_Error)
{
    constexpr int32_t deviceMemSize = 1024;
    constexpr int32_t hostMemSize = 512;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);

    TaskService taskService(deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId);
    taskService.TaskRegister("testTask", FailingCallback);

    g_taskExpMemMap[commId][devId] = nullptr;

    HcclResult result = HCCL_SUCCESS;
    std::thread taskThread([&taskService, &result]() {
        result = taskService.TaskRun();
    });

    WaitForTaskRunLoop(deviceMem.data());
    SetupOkTask(deviceMem.data(), "testTask", 42, 16); // dataSize > 0 to bypass ExecuteTask's dataSize==0 early return
    // flag is now TASK_OK (1) from SetupOkTask

    taskThread.join();

    EXPECT_EQ(result, HCCL_E_INTERNAL);
}

// ==================== TASK_OK full success flow ====================

TEST_F(TaskServiceTest, Ut_TaskRun_When_TaskOkFlow_Expect_Success)
{
    constexpr int32_t deviceMemSize = 1024;
    constexpr int32_t hostMemSize = 512;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);

    uint8_t* dpu2npuMem = deviceMem.data() + deviceMemSize / 2; // second half

    TaskService taskService(deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId);
    taskService.TaskRegister("myTask", TrackingCallback);

    HcclResult result = HCCL_SUCCESS;
    std::thread taskThread([&taskService, &result]() {
        result = taskService.TaskRun();
    });

    WaitForTaskRunLoop(deviceMem.data());

    // Prepare and issue TASK_OK
    const std::string taskType = "myTask";
    const uint32_t msgId = 12345;
    const uint32_t dataSize = 16;
    for (uint32_t i = 0; i < dataSize; i++) {
        deviceMem[273 + i] = static_cast<uint8_t>(i); // fill data area
    }
    SetupOkTask(deviceMem.data(), taskType, msgId, dataSize);
    // flag is now TASK_OK (1) from SetupOkTask

    // Wait for callback to be invoked
    auto start = std::chrono::steady_clock::now();
    while (TaskServiceTest::g_callbackResult != 42) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        auto elapsed
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        ASSERT_LT(elapsed, 5000) << "Timeout waiting for callback";
    }

    // Stop the loop
    *deviceMem.data() = 2; // TASK_TERMINATE
    taskThread.join();

    EXPECT_EQ(result, HCCL_SUCCESS);

    // After TaskRun: npu2dpu flag should be TERMINATE_RESPONSE
    // (but the TASK_OK path set it to UNSET first, then TERMINATE set it to TERMINATE_RESPONSE)
    // After TASK_OK processing, flag is UNSET, then we wrote TERMINATE, TaskRun wrote TERMINATE_RESPONSE
    EXPECT_EQ(*deviceMem.data(), 3); // TASK_TERMINATE_RESPONSE

    // Verify dpu2npu response (SynchronizeControlInfo copies control info)
    EXPECT_EQ(dpu2npuMem[0], 1); // dpu2npu flag should be 1 (sent by SynchronizeControlInfo)

    // Verify dpu2npu task type matches
    char* dpuTaskType = reinterpret_cast<char*>(dpu2npuMem + 1);
    EXPECT_STREQ(dpuTaskType, taskType.c_str());

    // Verify dpu2npu msgId matches
    uint32_t* dpuMsgId = reinterpret_cast<uint32_t*>(dpu2npuMem + 1 + 256);
    EXPECT_EQ(*dpuMsgId, msgId);
}

TEST_F(TaskServiceTest, Ut_ExecuteTaskexception_When_TaskExpShmemNull_Expect_ReturnSuccess)
{
    std::vector<uint8_t> deviceMem(1024);
    std::vector<uint8_t> hostMem(512);
    TaskService taskService(deviceMem.data(), 1024, hostMem.data(), 512, commId, devId);

    g_taskExpMemMap[commId][devId] = nullptr;

    HcclResult ret = taskService.ExecuteTaskexception(1, 100, 200);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(TaskServiceTest, Ut_ExecuteTaskexception_When_TaskExpShmemValid_Expect_WriteBothPositions)
{
    std::vector<uint8_t> deviceMem(1024);
    std::vector<uint8_t> hostMem(512);
    TaskService taskService(deviceMem.data(), 1024, hostMem.data(), 512, commId, devId);

    std::vector<uint8_t> taskexpShmem(20, 0);
    g_taskExpMemMap[commId][devId] = taskexpShmem.data();

    int32_t errCode = 1;
    u32 taskId = 100;
    u32 streamId = 200;
    HcclResult ret = taskService.ExecuteTaskexception(errCode, taskId, streamId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint16_t hcclRetVal = 0;
    memcpy(&hcclRetVal, taskexpShmem.data() + sizeof(uint8_t), sizeof(uint16_t));
    EXPECT_EQ(hcclRetVal, static_cast<uint16_t>(errCode));

    uint16_t dstRetVal = 0;
    memcpy(&dstRetVal, taskexpShmem.data() + sizeof(uint8_t) + sizeof(uint16_t), sizeof(uint16_t));
    EXPECT_EQ(dstRetVal, static_cast<uint16_t>(errCode));
}

TEST_F(TaskServiceTest, Ut_ExecuteTaskexception_When_CommIdNotInMap_Expect_InsertAndReturnSuccess)
{
    std::vector<uint8_t> deviceMem(1024);
    std::vector<uint8_t> hostMem(512);
    std::string uniqueCommId = "notInMap_" + std::to_string(reinterpret_cast<uintptr_t>(this));
    TaskService taskService(deviceMem.data(), 1024, hostMem.data(), 512, uniqueCommId, devId);

    HcclResult ret = taskService.ExecuteTaskexception(1, 100, 200);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(TaskServiceTest, Ut_ExecuteTaskClean_When_Normal_Expect_ReturnSuccess)
{
    constexpr int32_t deviceMemSize = 1024;
    constexpr int32_t hostMemSize = 512;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0xAA);
    std::vector<uint8_t> hostMem(hostMemSize, 0xBB);

    TaskService taskService(deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId);

    HcclResult ret = taskService.ExecuteTaskClean();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int32_t i = 0; i < hostMemSize; i++) {
        EXPECT_EQ(hostMem[i], 0);
    }
    for (int32_t i = 0; i < deviceMemSize; i++) {
        EXPECT_EQ(deviceMem[i], 0);
    }
}

TEST_F(TaskServiceTest, Ut_ExecuteTask_When_CallbackReturnsNonZero_Expect_CleanAndTaskexception)
{
    constexpr int32_t deviceMemSize = 1024;
    constexpr int32_t hostMemSize = 512;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);

    TaskService taskService(deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId);
    taskService.TaskRegister("testTask", FailingCallback);

    std::vector<uint8_t> taskexpShmem(20, 0);
    g_taskExpMemMap[commId][devId] = taskexpShmem.data();

    HcclResult result = HCCL_SUCCESS;
    std::thread taskThread([&taskService, &result]() {
        result = taskService.TaskRun();
    });

    WaitForTaskRunLoop(deviceMem.data());
    SetupOkTask(deviceMem.data(), "testTask", 42, 16);

    taskThread.join();

    EXPECT_EQ(result, HCCL_E_INTERNAL);

    uint16_t hcclRetVal = 0;
    memcpy(&hcclRetVal, taskexpShmem.data() + sizeof(uint8_t), sizeof(uint16_t));
    EXPECT_NE(hcclRetVal, 0);
}

TEST_F(TaskServiceTest, Ut_TaskRun_When_ProcessTaskOkFails_Expect_WriteTerminateResponseAndReturnError)
{
    constexpr int32_t deviceMemSize = 1024;
    constexpr int32_t hostMemSize = 512;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);

    TaskService taskService(deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId);
    taskService.TaskRegister("testTask", FailingCallback);

    std::vector<uint8_t> taskexpShmem(20, 0);
    g_taskExpMemMap[commId][devId] = taskexpShmem.data();

    HcclResult result = HCCL_SUCCESS;
    std::thread taskThread([&taskService, &result]() {
        result = taskService.TaskRun();
    });

    WaitForTaskRunLoop(deviceMem.data());
    SetupOkTask(deviceMem.data(), "testTask", 42, 16);

    taskThread.join();

    EXPECT_EQ(result, HCCL_E_INTERNAL);
    EXPECT_EQ(deviceMem[0], 0);
}

TEST_F(TaskServiceTest, Ut_TaskRun_When_TaskOkCallbackFails_Expect_TaskexceptionShmemWritten)
{
    constexpr int32_t deviceMemSize = 1024;
    constexpr int32_t hostMemSize = 512;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);

    TaskService taskService(deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId);
    taskService.TaskRegister("testTask", FailingCallback);

    std::vector<uint8_t> taskexpShmem(20, 0);
    g_taskExpMemMap[commId][devId] = taskexpShmem.data();

    HcclResult result = HCCL_SUCCESS;
    std::thread taskThread([&taskService, &result]() {
        result = taskService.TaskRun();
    });

    WaitForTaskRunLoop(deviceMem.data());
    SetupOkTask(deviceMem.data(), "testTask", 42, 16);

    taskThread.join();

    EXPECT_EQ(result, HCCL_E_INTERNAL);

    uint16_t hcclRetVal = 0;
    memcpy(&hcclRetVal, taskexpShmem.data() + sizeof(uint8_t), sizeof(uint16_t));
    EXPECT_NE(hcclRetVal, 0);

    uint16_t dstRetVal = 0;
    memcpy(&dstRetVal, taskexpShmem.data() + sizeof(uint8_t) + sizeof(uint16_t), sizeof(uint16_t));
    EXPECT_NE(dstRetVal, 0);
}

TEST_F(TaskServiceTest, Ut_RunDpuRpcSrvLaunch_When_TaskExpMemNull_Expect_ReturnPara)
{
    struct TestDpuKernelLaunchParam {
        u64 memorySize;
        void* deviceMem;
        void* hostMem;
        uint32_t deviceId;
        std::string commId;
        void* taskExpMem;
        void* deviceMemDev;
        bool isPcieLink;
    };

    std::vector<uint8_t> deviceMemBuf(1024);
    std::vector<uint8_t> hostMemBuf(512);
    std::string testCommId = "runDpuTest_" + std::to_string(reinterpret_cast<uintptr_t>(this));

    TestDpuKernelLaunchParam params;
    params.memorySize = 1024;
    params.deviceMem = deviceMemBuf.data();
    params.hostMem = hostMemBuf.data();
    params.deviceId = 0;
    params.commId = testCommId;
    params.taskExpMem = nullptr;
    params.deviceMemDev = nullptr;
    params.isPcieLink = false;

    uint32_t ret = RunDpuRpcSrvLaunch(reinterpret_cast<uint64_t>(reinterpret_cast<void*>(&params)));
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// ==================== ExecuteTask data copy paths ====================

constexpr uint32_t CTRL_HDR_LEN_IN_UT = 273; // |flag[1]|taskType[256]|msgId[4]|timeout[4]|dataSize[8]|

static std::vector<uint8_t> g_expectedPayload;
static bool g_payloadMatched = false;
static uint32_t g_aclrtMemcpyCalls = 0;
static size_t g_capturedDestMax = 0;
static const void* g_capturedSrc = nullptr;
static aclrtMemcpyKind g_capturedMemcpyKind = ACL_MEMCPY_DEFAULT;

// deviceMem 为 host 访问地址对应的共享内存，dmaSrcMem 为 device 侧地址对应的共享内存（为空表示不构造）
static void PreparePayload(std::vector<uint8_t>& deviceMem, std::vector<uint8_t>* dmaSrcMem, uint32_t dataSize)
{
    g_expectedPayload.assign(dataSize, 0);
    for (uint32_t i = 0; i < dataSize; ++i) {
        uint8_t value = static_cast<uint8_t>((i % 251) + 1); // 控制头之后为 data
        deviceMem[CTRL_HDR_LEN_IN_UT + i] = value;
        if (dmaSrcMem != nullptr) {
            (*dmaSrcMem)[CTRL_HDR_LEN_IN_UT + i] = value;
        }
        g_expectedPayload[i] = value;
    }
}

static int32_t PayloadCheckCallback(uint64_t addr, int32_t size)
{
    TaskServiceTest::g_callbackResult = 42;
    g_payloadMatched
        = (static_cast<int32_t>(g_expectedPayload.size()) == size)
          && (memcmp(reinterpret_cast<void*>(addr), g_expectedPayload.data(), g_expectedPayload.size()) == 0);
    return 0;
}

static aclError FakeAclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind)
{
    ++g_aclrtMemcpyCalls;
    g_capturedDestMax = destMax;
    g_capturedSrc = src;
    g_capturedMemcpyKind = kind;
    if (count > destMax) {
        return ACL_ERROR_INVALID_PARAM;
    }
    memcpy(dst, src, count);
    return ACL_SUCCESS;
}

static aclError FailingAclrtMemcpy(void*, size_t, const void*, size_t, aclrtMemcpyKind)
{
    ++g_aclrtMemcpyCalls;
    return ACL_ERROR_INVALID_PARAM;
}

static uint32_t g_captureModeExchangeCalls = 0;
static bool g_allExchangeModeRelaxed = true;

static aclError FakeExchangeCaptureMode(aclmdlRICaptureMode* mode)
{
    ++g_captureModeExchangeCalls;
    if ((mode == nullptr) || (*mode != aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_RELAXED)) {
        g_allExchangeModeRelaxed = false;
    }
    return ACL_SUCCESS;
}

static HcclResult RunTaskOkFlow(TaskService& taskService, std::vector<uint8_t>& deviceMem, uint32_t dataSize)
{
    HcclResult result = HCCL_SUCCESS;
    std::thread taskThread([&taskService, &result]() {
        result = taskService.TaskRun();
    });

    WaitForTaskRunLoop(deviceMem.data());
    SetupOkTask(deviceMem.data(), "testTask", 42, dataSize);

    auto start = std::chrono::steady_clock::now();
    while (TaskServiceTest::g_callbackResult != 42) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        auto elapsed
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        if (elapsed >= 5000) {
            break;
        }
    }

    *deviceMem.data() = 2; // TASK_TERMINATE
    taskThread.join();
    return result;
}

// dataSize <= CTRL_HDR_DEFAULT_DATA_LEN(512)，payload 已在预拷贝的 ctrlHdr 中，走 memcpy_s 而非 aclrtMemcpy
TEST_F(TaskServiceTest, Ut_ExecuteTask_When_DataInCtrlHdr_Expect_NoAclrtMemcpyAndContentMatch)
{
    constexpr int32_t deviceMemSize = 1024;
    constexpr int32_t hostMemSize = 512;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);
    std::vector<uint8_t> dmaSrcMem(deviceMemSize, 0);

    TaskService taskService(
        deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId, true, dmaSrcMem.data());
    taskService.TaskRegister("testTask", PayloadCheckCallback);

    constexpr uint32_t dataSize = 16;
    PreparePayload(deviceMem, &dmaSrcMem, dataSize);
    g_payloadMatched = false;
    g_aclrtMemcpyCalls = 0;
    MOCKER(aclrtMemcpy).stubs().will(invoke(FakeAclrtMemcpy));

    HcclResult result = RunTaskOkFlow(taskService, deviceMem, dataSize);

    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_TRUE(g_payloadMatched);
    EXPECT_EQ(g_aclrtMemcpyCalls, 0U);
    GlobalMockObject::verify();
}

// dataSize > 512，payload 未被预拷贝完整，需回源 srcPtr 拷贝；非PCIE链路走 memcpy_s
TEST_F(TaskServiceTest, Ut_ExecuteTask_When_DataExceedsCtrlHdr_Expect_CopyFromSrcPtr)
{
    constexpr int32_t deviceMemSize = 4096; // shmemSize_ = 2048, leftSize_ = 1775
    constexpr int32_t hostMemSize = 2048;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);

    TaskService taskService(deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId);
    taskService.TaskRegister("testTask", PayloadCheckCallback);

    constexpr uint32_t dataSize = 1024;
    PreparePayload(deviceMem, nullptr, dataSize);
    g_payloadMatched = false;
    g_captureModeExchangeCalls = 0;
    g_allExchangeModeRelaxed = true;
    MOCKER(aclmdlRICaptureThreadExchangeMode).stubs().will(invoke(FakeExchangeCaptureMode));

    HcclResult result = RunTaskOkFlow(taskService, deviceMem, dataSize);

    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_TRUE(g_payloadMatched);
    // 非 PCIE 链路直接 memcpy_s，不应触发 capture mode 交换
    EXPECT_EQ(g_captureModeExchangeCalls, 0U);
    GlobalMockObject::verify();
}

// PCIe链路且 dataSize > 512，源在device，走 aclrtMemcpy 且 kind 为 DEVICE_TO_HOST、destMax 为目的缓冲区容量
TEST_F(TaskServiceTest, Ut_ExecuteTask_When_PcieLinkDataExceedsCtrlHdr_Expect_DeviceToHostMemcpy)
{
    constexpr int32_t deviceMemSize = 4096;
    constexpr int32_t hostMemSize = 2048;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);
    // device 侧源缓冲区与 host 访问缓冲区独立，确保断言能区分 dmaSrcMem 与 npu2dpuMem_
    std::vector<uint8_t> dmaSrcMem(deviceMemSize, 0);

    TaskService taskService(
        deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId, true, dmaSrcMem.data());
    taskService.TaskRegister("testTask", PayloadCheckCallback);

    constexpr uint32_t dataSize = 1024;
    PreparePayload(deviceMem, &dmaSrcMem, dataSize);
    g_payloadMatched = false;
    g_aclrtMemcpyCalls = 0;
    g_capturedDestMax = 0;
    g_capturedSrc = nullptr;
    g_capturedMemcpyKind = ACL_MEMCPY_DEFAULT;
    g_captureModeExchangeCalls = 0;
    g_allExchangeModeRelaxed = true;
    MOCKER(aclrtMemcpy).stubs().will(invoke(FakeAclrtMemcpy));
    MOCKER(aclmdlRICaptureThreadExchangeMode).stubs().will(invoke(FakeExchangeCaptureMode));

    HcclResult result = RunTaskOkFlow(taskService, deviceMem, dataSize);

    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_TRUE(g_payloadMatched);
    EXPECT_EQ(g_aclrtMemcpyCalls, 1U);
    EXPECT_EQ(g_capturedDestMax, static_cast<size_t>(hostMemSize));
    // DEVICE_TO_HOST 的源必须是 device 侧地址，不能复用 host 访问地址
    EXPECT_EQ(g_capturedSrc, static_cast<const void*>(dmaSrcMem.data() + CTRL_HDR_LEN_IN_UT));
    EXPECT_NE(g_capturedSrc, static_cast<const void*>(deviceMem.data() + CTRL_HDR_LEN_IN_UT));
    EXPECT_EQ(g_capturedMemcpyKind, ACL_MEMCPY_DEVICE_TO_HOST);
    // HrtMemcpy 必须先把线程 capture mode 切到 RELAXED，图捕获场景下同步拷贝才能完成
    EXPECT_GT(g_captureModeExchangeCalls, 0U);
    EXPECT_TRUE(g_allExchangeModeRelaxed);
    GlobalMockObject::verify();
}

// aclrtMemcpy 失败不应终止任务：回退 memcpy_s 完成拷贝，仍返回成功且数据一致
TEST_F(TaskServiceTest, Ut_ExecuteTask_When_AclrtMemcpyFails_Expect_FallbackMemcpySuccess)
{
    constexpr int32_t deviceMemSize = 4096; // shmemSize_ = 2048, leftSize_ = 1775
    constexpr int32_t hostMemSize = 2048;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);
    std::vector<uint8_t> dmaSrcMem(deviceMemSize, 0);

    TaskService taskService(
        deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId, true, dmaSrcMem.data());
    taskService.TaskRegister("testTask", PayloadCheckCallback);

    constexpr uint32_t dataSize = 1024;
    PreparePayload(deviceMem, &dmaSrcMem, dataSize);
    g_payloadMatched = false;
    SetupOkTask(deviceMem.data(), "testTask", 42, dataSize);
    // ctrlHdr 为 512 字节，容纳不下 1024 字节 payload，触发 aclrtMemcpy 分支
    std::vector<uint8_t> ctrlHdr(512, 0);
    memcpy(ctrlHdr.data(), deviceMem.data(), ctrlHdr.size());

    g_aclrtMemcpyCalls = 0;
    MOCKER(aclrtMemcpy).stubs().will(invoke(FailingAclrtMemcpy));

    HcclResult result = taskService.ExecuteTask(ctrlHdr.data(), ctrlHdr.size(), deviceMem.data(), "testTask");

    EXPECT_EQ(result, HCCL_SUCCESS); // 回退 memcpy_s 后任务继续
    EXPECT_EQ(g_aclrtMemcpyCalls, 1U);
    EXPECT_TRUE(g_payloadMatched); // 回退路径数据完整
    GlobalMockObject::verify();
}

// PCIE 链路但未提供 device 侧源地址时，TaskRun 预检应直接失败，避免把 host 访问地址当 device 地址使用
TEST_F(TaskServiceTest, Ut_TaskRun_When_PcieLinkWithoutDmaSrcMem_Expect_ReturnPtrError)
{
    constexpr int32_t deviceMemSize = 4096;
    constexpr int32_t hostMemSize = 2048;
    std::vector<uint8_t> deviceMem(deviceMemSize, 0);
    std::vector<uint8_t> hostMem(hostMemSize, 0);

    TaskService taskService(deviceMem.data(), deviceMemSize, hostMem.data(), hostMemSize, commId, devId, true);
    EXPECT_EQ(taskService.TaskRun(), HCCL_E_PTR);
}
